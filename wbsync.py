"""whiteboard-sync — keep Todoist in sync with the office whiteboards.

Reads the boards through kinect-knob's full-resolution camera snapshot
(``GET /api/snapshot`` — unmirrored, so the writing is readable), has a
local Ollama vision model transcribe the open items (zero marginal cost,
fully private — chosen because scans run frequently), and reconciles them
into Todoist via Home Assistant:

* LEFT board  -> work items,     tagged ["whiteboard", "work"]
* RIGHT board -> personal items, tagged ["whiteboard", "personal"]
* New item on a board            -> ``todoist.new_task`` into the Inbox
* Item erased or checkbox ticked -> after it has been missing for
  ``WB_MISSING_TO_COMPLETE`` consecutive scans, ``todo.update_item``
  marks the Todoist task completed (the guard absorbs one bad OCR read)

Scans run every ``WB_INTERVAL_S`` seconds, only while ``WB_PRESENCE_ENTITY``
is away (that's also when nobody blocks the camera's view of the boards),
and the Claude call is skipped entirely when the cropped board region hasn't
changed since the last processed scan — so vision costs accrue only when
something was actually written or erased.

OCR of handwriting is noisy run-to-run, so reconciliation is fuzzy
(difflib ratio >= 0.8) and the FIRST transcription of an item is kept as
canonical — it must stay byte-identical to the Todoist task summary or the
completion call can't find it.
"""
from __future__ import annotations

import base64
import difflib
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

log = logging.getLogger("wbsync")

FUZZY_MATCH_RATIO = 0.8
HTTP_PORT = 8430

READ_PROMPT = """\
This photo shows two whiteboards side by side.
The LEFT board holds WORK to-do items. The RIGHT board holds PERSONAL to-do items.
Transcribe every item that is still OPEN: its checkbox is empty (not ticked)
and it is not crossed out or erased. Skip completed items, headings, name tags,
stickers, photos, and anything that is not a list item.
Write each item as a short task phrase without bullet or checkbox characters,
fixing obvious handwriting artifacts. If a board is empty or unreadable,
return an empty list for it."""

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "work": {"type": "array", "items": {"type": "string"}},
        "personal": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["work", "personal"],
    "additionalProperties": False,
}


def parse_region(spec: str) -> tuple[int, int, int, int]:
    """'x1,y1,x2,y2' -> tuple; raises ValueError on nonsense."""
    parts = [int(p) for p in spec.split(",")]
    if len(parts) != 4 or parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise ValueError(f"bad region {spec!r}, want 'x1,y1,x2,y2'")
    return tuple(parts)  # type: ignore[return-value]


@dataclass
class Config:
    ha_url: str
    ha_token: str
    ollama_url: str
    kinect_url: str
    region: tuple[int, int, int, int]
    presence_entity: str
    todo_entity: str
    todoist_project: str
    interval_s: int
    change_threshold: float
    missing_to_complete: int
    scan_when_home: bool
    model: str
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ.get
        return cls(
            ha_url=env("HA_URL", "").rstrip("/"),
            ha_token=env("HA_TOKEN", ""),
            ollama_url=env("OLLAMA_URL", "http://192.168.2.229:11434").rstrip("/"),
            kinect_url=env("KINECT_URL", "").rstrip("/"),
            region=parse_region(env("WB_REGION", "600,280,1320,810")),
            presence_entity=env("WB_PRESENCE_ENTITY", ""),
            todo_entity=env("WB_TODO_ENTITY", "todo.inbox"),
            todoist_project=env("WB_TODOIST_PROJECT", "Inbox"),
            interval_s=int(env("WB_INTERVAL_S", "900")),
            change_threshold=float(env("WB_CHANGE_THRESHOLD", "4.0")),
            missing_to_complete=int(env("WB_MISSING_TO_COMPLETE", "2")),
            scan_when_home=env("WB_SCAN_WHEN_HOME", "false").lower() in ("1", "true", "yes"),
            model=env("WB_MODEL", "qwen3-vl:8b"),
            data_dir=Path(env("WB_DATA_DIR", "/data")),
        )


class HomeAssistant:
    def __init__(self, url: str, token: str):
        self._url = url
        self._s = requests.Session()
        self._s.headers["Authorization"] = f"Bearer {token}"

    def state(self, entity: str) -> Optional[str]:
        try:
            r = self._s.get(f"{self._url}/api/states/{entity}", timeout=10)
            r.raise_for_status()
            return r.json().get("state")
        except requests.RequestException as exc:
            log.warning("HA state(%s) failed: %s", entity, exc)
            return None

    def call(self, domain: str, service: str, data: dict) -> bool:
        try:
            r = self._s.post(f"{self._url}/api/services/{domain}/{service}", json=data, timeout=15)
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.warning("HA %s.%s failed: %s", domain, service, exc)
            return False


class BoardReader:
    """Local Ollama vision call: JPEG in, {'work': [...], 'personal': [...]} out.

    Uses Ollama structured outputs (the ``format`` field takes a JSON schema)
    so the reply parses without prompt gymnastics; temperature 0 keeps
    transcriptions stable between scans, which the fuzzy reconciler relies on."""

    def __init__(self, url: str, model: str):
        self._url = url.rstrip("/")
        self._model = model

    def read(self, jpeg: bytes) -> dict:
        r = requests.post(
            f"{self._url}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "format": READ_SCHEMA,
                # qwen3-vl is a thinking model; unchecked it can burn its whole
                # budget "thinking" and return empty content. Transcription
                # needs no reasoning — turn it off (also much faster).
                "think": False,
                # num_ctx: the image alone is ~4k tokens; the ollama server's
                # default 4096 context truncated the prompt at the boundary
                # (intermittent empty/garbled reads until this was raised).
                "options": {"temperature": 0, "num_ctx": 6144},
                "messages": [{
                    "role": "user",
                    "content": READ_PROMPT,
                    "images": [base64.standard_b64encode(jpeg).decode()],
                }],
            },
            timeout=600,  # first call loads 6 GB into VRAM; Pascal inference is leisurely
        )
        r.raise_for_status()
        content = r.json()["message"].get("content", "")
        if not content.strip():
            raise RuntimeError("model returned empty content")
        return json.loads(content)


def fuzzy_match(needle: str, haystack: list[str]) -> Optional[str]:
    """Best fuzzy match for OCR'd text, or None. Handwriting reads vary
    slightly between scans; exact matching would duplicate tasks."""
    best, best_ratio = None, 0.0
    for candidate in haystack:
        ratio = difflib.SequenceMatcher(None, needle.lower(), candidate.lower()).ratio()
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio
    return best if best_ratio >= FUZZY_MATCH_RATIO else None


class Syncer:
    def __init__(self, cfg: Config, ha: HomeAssistant, reader):
        self.cfg = cfg
        self.ha = ha
        self.reader = reader
        self._lock = threading.Lock()
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = cfg.data_dir / "state.json"
        self._baseline_path = cfg.data_dir / "baseline.npy"
        self.state: dict = {"items": []}  # [{text, board, missing}]
        if self._state_path.is_file():
            self.state = json.loads(self._state_path.read_text())
        self.last_result: dict = {}
        self.last_scan_t: float = 0.0

    # -- imaging --------------------------------------------------------
    def _fetch_crop(self) -> tuple[bytes, np.ndarray]:
        r = requests.get(f"{self.cfg.kinect_url}/api/snapshot", timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        crop = img.crop(self.cfg.region)
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), gray

    def _changed(self, gray: np.ndarray) -> bool:
        if not self._baseline_path.is_file():
            return True
        baseline = np.load(self._baseline_path)
        if baseline.shape != gray.shape:
            return True
        return float(np.abs(gray - baseline).mean()) >= self.cfg.change_threshold

    # -- reconciliation ---------------------------------------------------
    def reconcile(self, seen: dict, apply: bool = True) -> dict:
        """Diff OCR results against known state. With ``apply`` (the default),
        push adds/completions to HA and persist; with ``apply=False`` (dry
        run) just report what WOULD happen, touching nothing.
        Returns {'added': [...], 'completed': [...]} of item texts."""
        added, completed = [], []
        for board in ("work", "personal"):
            seen_texts = [t.strip() for t in seen.get(board, []) if t.strip()]
            known = [i for i in self.state["items"] if i["board"] == board]
            known_texts = [i["text"] for i in known]

            matched_known = set()
            for text in seen_texts:
                match = fuzzy_match(text, known_texts)
                if match is not None:
                    matched_known.add(match)  # keep canonical text (= Todoist summary)
                    continue
                if not apply:
                    added.append(text)
                    continue
                if self.ha.call("todoist", "new_task", {
                    "content": text,
                    "project": self.cfg.todoist_project,
                    "labels": ["whiteboard", board],
                }):
                    self.state["items"].append({"text": text, "board": board, "missing": 0})
                    added.append(text)

            for item in known:
                if item["text"] in matched_known:
                    if apply:
                        item["missing"] = 0
                    continue
                if not apply:
                    if item["missing"] + 1 >= self.cfg.missing_to_complete:
                        completed.append(item["text"])
                    continue
                item["missing"] += 1
                if item["missing"] >= self.cfg.missing_to_complete:
                    if self.ha.call("todo", "update_item", {
                        "entity_id": self.cfg.todo_entity,
                        "item": item["text"],
                        "status": "completed",
                    }):
                        self.state["items"].remove(item)
                        completed.append(item["text"])
        if apply:
            self._state_path.write_text(json.dumps(self.state, indent=2))
        return {"added": added, "completed": completed}

    # -- scan -------------------------------------------------------------
    def scan(self, force: bool = False, dry: bool = False) -> dict:
        with self._lock:
            self.last_scan_t = time.time()
            try:
                jpeg, gray = self._fetch_crop()
            except requests.RequestException as exc:
                return self._done({"ok": False, "error": f"snapshot failed: {exc}"})
            if not force and not self._changed(gray):
                return self._done({"ok": True, "changed": False})
            try:
                seen = self.reader.read(jpeg)
            except Exception as exc:  # noqa: BLE001 — one bad read must not kill the loop
                log.warning("board read failed: %s", exc)
                return self._done({"ok": False, "error": f"read failed: {exc}"})
            result = self.reconcile(seen, apply=not dry)
            if not dry:
                np.save(self._baseline_path, gray)  # only after a processed read
            return self._done({"ok": True, "changed": True, "dry": dry, "seen": seen, **result})

    def _done(self, result: dict) -> dict:
        result["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_result = result
        log.info("scan: %s", json.dumps(result))
        return result

    def away(self) -> bool:
        if not self.cfg.presence_entity:
            return True
        return self.ha.state(self.cfg.presence_entity) != "home"


def serve_http(syncer: Syncer) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                self._json(200, {"status": "ok"})
            elif self.path == "/state":
                self._json(200, {"items": syncer.state["items"], "last": syncer.last_result})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/scan"):
                self._json(200, syncer.scan(force="force" in self.path, dry="dry" in self.path))
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, *args):  # quiet
            pass

    ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler).serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    cfg = Config.from_env()
    ha = HomeAssistant(cfg.ha_url, cfg.ha_token)
    reader = BoardReader(cfg.ollama_url, cfg.model)
    try:
        tags = requests.get(f"{cfg.ollama_url}/api/tags", timeout=5).json()
        if not any(m["name"].startswith(cfg.model) for m in tags.get("models", [])):
            log.warning("model %r not found in ollama — pull it or set WB_MODEL", cfg.model)
    except requests.RequestException as exc:
        log.warning("ollama not reachable at %s: %s", cfg.ollama_url, exc)
    syncer = Syncer(cfg, ha, reader)
    threading.Thread(target=serve_http, args=(syncer,), daemon=True).start()
    log.info("whiteboard-sync up: model %s, every %ss while %s is away; region %s",
             cfg.model, cfg.interval_s, cfg.presence_entity or "(no presence gate)", cfg.region)

    while True:
        time.sleep(30)
        if time.time() - syncer.last_scan_t < cfg.interval_s:
            continue
        if not cfg.scan_when_home and not syncer.away():
            syncer.last_scan_t = time.time()  # re-check one interval later
            continue
        syncer.scan()


if __name__ == "__main__":
    main()
