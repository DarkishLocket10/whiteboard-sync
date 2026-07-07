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

A dashboard on port 8430 (``GET /``) shows what the service is doing:
presence gate, tracked items, scan history with change scores, and the
latest camera crop.

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
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageOps

log = logging.getLogger("wbsync")

FUZZY_MATCH_RATIO = 0.8
HTTP_PORT = 8430
HISTORY_KEEP = 400  # scan/skip records kept in memory and in data/history.jsonl

# Runtime-tunable settings (dashboard toggles, persisted in data/settings.json;
# env vars provide the defaults). key -> bool | (type, lo, hi) | enum tuple.
SETTING_BOUNDS = {
    "enabled": bool,             # master switch for scheduled scans
    "presence_gate": bool,       # only scan while the presence entity is away
    "change_detection": bool,    # skip the vision read when the crop hasn't changed
    "obstruction_guard": bool,   # skip reconcile when the model sees an obstruction
    "enhance": bool,             # autocontrast + unsharp mask on the crop
    "interval_s": (int, 60, 86400),
    "change_threshold": (float, 0.1, 50.0),
    "missing_to_complete": (int, 1, 10),
    "capture_frames": (int, 1, 32),      # frames stacked by kinect-knob per photo
    "capture_quality": (int, 50, 100),   # JPEG quality sent to the vision model
    "upscale": (int, 1, 2),              # LANCZOS upscale factor before the model
    "capture_format": ("jpeg", "png"),   # encoding sent to the vision model
}

READ_PROMPT = """\
This photo shows two whiteboards side by side.
The LEFT board holds WORK to-do items. The RIGHT board holds PERSONAL to-do items.
First, set "obstructed" to true if a person, chair, or any object blocks or
covers ANY part of either whiteboard, or if a board is not fully visible in
the photo — otherwise false.
Then transcribe every item that is still OPEN: its checkbox is empty (not
ticked) and it is not crossed out or erased. Skip completed items, headings,
name tags, stickers, photos, and anything that is not a list item.
Write each item as a short task phrase without bullet or checkbox characters,
fixing obvious handwriting artifacts. If a board is empty or unreadable,
return an empty list for it."""

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "obstructed": {"type": "boolean"},
        "work": {"type": "array", "items": {"type": "string"}},
        "personal": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["obstructed", "work", "personal"],
    "additionalProperties": False,
}


def parse_region(spec: str) -> tuple[int, int, int, int]:
    """'x1,y1,x2,y2' -> tuple; raises ValueError on nonsense."""
    parts = [int(p) for p in spec.split(",")]
    if len(parts) != 4 or parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise ValueError(f"bad region {spec!r}, want 'x1,y1,x2,y2'")
    return tuple(parts)  # type: ignore[return-value]


def write_json_atomic(path: Path, obj) -> None:
    """A crash mid-write must not corrupt state — write aside, then rename."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


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
            model=env("WB_MODEL", "qwen3-vl:8b-instruct"),
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
        self._history_path = cfg.data_dir / "history.jsonl"
        self.crop_path = cfg.data_dir / "last_crop.jpg"
        self.state: dict = {"items": []}  # [{text, board, missing}]
        if self._state_path.is_file():
            self.state = json.loads(self._state_path.read_text())
        # items_view is what HTTP threads serve: a snapshot swapped whole, so
        # a scan mutating self.state mid-serialisation can't tear a response.
        self.items_view: list[dict] = [dict(i) for i in self.state["items"]]
        self.last_result: dict = {}
        self.last_scan_t: float = 0.0
        self.scanning = False
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.loop_beat = time.time()
        self.gate: dict = {}  # last presence-gate skip, for the dashboard
        self.counters = {"scans": 0, "changed": 0, "unchanged": 0, "obstructed": 0,
                         "errors": 0, "added": 0, "completed": 0, "skips_home": 0,
                         "skips_off": 0}
        self._presence: Optional[str] = None
        self._presence_t = 0.0
        self._settings_path = cfg.data_dir / "settings.json"
        self.settings = {
            "enabled": True,
            "presence_gate": not cfg.scan_when_home,
            "change_detection": True,
            "obstruction_guard": True,
            "enhance": True,
            "interval_s": cfg.interval_s,
            "change_threshold": cfg.change_threshold,
            "missing_to_complete": cfg.missing_to_complete,
            "capture_frames": 8,
            "capture_quality": 92,
            "upscale": 1,
            "capture_format": "jpeg",
        }
        if self._settings_path.is_file():
            try:
                saved = json.loads(self._settings_path.read_text())
                self.settings.update(
                    {k: v for k, v in saved.items() if k in self.settings})
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("settings.json unreadable (%s), using defaults", exc)
        self.crop_t: Optional[float] = (
            self.crop_path.stat().st_mtime if self.crop_path.is_file() else None)
        self.history: list[dict] = []
        if self._history_path.is_file():
            for line in self._history_path.read_text().splitlines()[-HISTORY_KEEP:]:
                try:
                    self.history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            # rewrite the trimmed tail so the file can't grow without bound
            self._history_path.write_text(
                "".join(json.dumps(e) + "\n" for e in self.history))

    # -- settings ---------------------------------------------------------
    def update_settings(self, patch: dict) -> dict:
        """Validate + apply a runtime settings patch from the dashboard.
        Returns {'applied': {...}, 'rejected': {key: reason}}."""
        applied, rejected = {}, {}
        for key, val in patch.items():
            spec = SETTING_BOUNDS.get(key)
            if spec is None:
                rejected[key] = "unknown setting"
                continue
            try:
                if spec is bool:
                    if not isinstance(val, bool):
                        raise ValueError("want true/false")
                    new = val
                elif callable(spec[0]):
                    typ, lo, hi = spec
                    new = typ(val)
                    if not lo <= new <= hi:
                        raise ValueError(f"out of range {lo}..{hi}")
                else:  # enum of strings
                    new = str(val).lower()
                    if new not in spec:
                        raise ValueError(f"want one of {spec}")
            except (TypeError, ValueError) as exc:
                rejected[key] = str(exc)
                continue
            if self.settings[key] != new:
                applied[key] = new
        if applied:
            self.settings.update(applied)
            write_json_atomic(self._settings_path, self.settings)
            self._record({"type": "settings", "changed": applied})
            log.info("settings changed: %s", json.dumps(applied))
        return {"applied": applied, "rejected": rejected}

    # -- imaging --------------------------------------------------------
    def _fetch_crop(self) -> tuple[bytes, np.ndarray]:
        s = self.settings
        # Ask kinect-knob for a stacked, losslessly-encoded photo; an older
        # kinect-knob simply ignores unknown params and returns its JPEG.
        params = {"format": "png"}
        if s["capture_frames"] > 1:
            params["frames"] = s["capture_frames"]
        r = requests.get(f"{self.cfg.kinect_url}/api/snapshot", params=params,
                         timeout=15 + 2 * s["capture_frames"])
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        crop = img.crop(self.cfg.region).convert("RGB")
        # The baseline compares the RAW crop — toggling enhance/upscale must
        # not read as "the board changed".
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
        if s["enhance"]:
            crop = ImageOps.autocontrast(crop, cutoff=1)
            crop = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
        if s["upscale"] > 1:
            crop = crop.resize(
                (crop.width * s["upscale"], crop.height * s["upscale"]), Image.LANCZOS)
        buf = io.BytesIO()
        if s["capture_format"] == "png":
            crop.save(buf, format="PNG")
        else:
            crop.save(buf, format="JPEG", quality=s["capture_quality"])
        payload = buf.getvalue()
        try:
            disp = io.BytesIO()
            crop.save(disp, format="JPEG", quality=90)  # dashboard copy = model's view
            tmp = self.crop_path.with_name(self.crop_path.name + ".tmp")
            tmp.write_bytes(disp.getvalue())
            tmp.replace(self.crop_path)  # never serve a half-written image
            self.crop_t = time.time()
        except OSError as exc:
            log.warning("crop save failed: %s", exc)
        return payload, gray

    def _diff_score(self, gray: np.ndarray) -> Optional[float]:
        """Mean abs pixel difference vs the last processed read. None means
        no usable baseline (first run, corrupt file, region resize) — always
        treated as changed."""
        if not self._baseline_path.is_file():
            return None
        try:
            baseline = np.load(self._baseline_path)
        except Exception as exc:  # noqa: BLE001 — corrupt baseline must not stop scans
            log.warning("baseline unreadable (%s), re-reading board", exc)
            return None
        if baseline.shape != gray.shape:
            return None
        return float(np.abs(gray - baseline).mean())

    # -- reconciliation ---------------------------------------------------
    def reconcile(self, seen: dict, apply: bool = True) -> dict:
        """Diff OCR results against known state. With ``apply`` (the default),
        push adds/completions to HA and persist; with ``apply=False`` (dry
        run) just report what WOULD happen, touching nothing.
        Returns {'added': [...], 'completed': [...]} of item texts plus a
        'failed' count of HA calls that didn't land (they retry next scan)."""
        added, completed, failed = [], [], 0
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
                else:
                    failed += 1

            for item in known:
                if item["text"] in matched_known:
                    if apply:
                        item["missing"] = 0
                    continue
                if not apply:
                    if item["missing"] + 1 >= self.settings["missing_to_complete"]:
                        completed.append(item["text"])
                    continue
                item["missing"] += 1
                if item["missing"] >= self.settings["missing_to_complete"]:
                    if self.ha.call("todo", "update_item", {
                        "entity_id": self.cfg.todo_entity,
                        "item": item["text"],
                        "status": "completed",
                    }):
                        self.state["items"].remove(item)
                        completed.append(item["text"])
                    else:
                        failed += 1
        if apply:
            write_json_atomic(self._state_path, self.state)
            self.items_view = [dict(i) for i in self.state["items"]]
        return {"added": added, "completed": completed, "failed": failed}

    # -- scan -------------------------------------------------------------
    def scan(self, force: bool = False, dry: bool = False, trigger: str = "manual") -> dict:
        with self._lock:
            self.last_scan_t = time.time()
            self.scanning = True
            t0 = time.monotonic()
            try:
                result = self._scan_inner(force, dry)
            except Exception as exc:  # noqa: BLE001 — a scan must never kill its caller
                log.exception("scan failed unexpectedly")
                result = {"ok": False, "error": f"internal: {exc}"}
            finally:
                self.scanning = False
            result["trigger"] = trigger
            result["took_s"] = round(time.monotonic() - t0, 1)
            return self._done(result)

    def _scan_inner(self, force: bool, dry: bool) -> dict:
        try:
            jpeg, gray = self._fetch_crop()
        except requests.RequestException as exc:
            return {"ok": False, "error": f"snapshot failed: {exc}"}
        s = self.settings
        diff = self._diff_score(gray)
        base = {"diff": None if diff is None else round(diff, 2)}
        if (not force and s["change_detection"] and diff is not None
                and diff < s["change_threshold"]):
            return {"ok": True, "changed": False, **base}
        try:
            seen = self.reader.read(jpeg)
        except Exception as exc:  # noqa: BLE001 — one bad read must not kill the loop
            log.warning("board read failed: %s", exc)
            return {"ok": False, "error": f"read failed: {exc}", **base}
        if seen.get("obstructed"):
            if s["obstruction_guard"]:
                # Someone/something is blocking a board (presence lag, guest,
                # chair). Items behind the obstruction would read as "missing"
                # and could be falsely completed — touch nothing, retry later.
                return {"ok": True, "changed": True, "obstructed": True, **base}
            base["obstructed_ignored"] = True
        result = self.reconcile(seen, apply=not dry)
        if not dry and not result["failed"]:
            # A failed HA push means an item is not yet in Todoist; keeping
            # the old baseline forces a re-read (and a retry) next interval.
            np.save(self._baseline_path, gray)
        return {"ok": True, "changed": True, "dry": dry, "seen": seen, **base, **result}

    def _done(self, result: dict) -> dict:
        result["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["t"] = round(time.time(), 1)
        self.last_result = result
        c = self.counters
        c["scans"] += 1
        if not result.get("ok"):
            c["errors"] += 1
        elif result.get("obstructed"):
            c["obstructed"] += 1
        elif not result.get("changed"):
            c["unchanged"] += 1
        else:
            c["changed"] += 1
        if not result.get("dry"):
            c["added"] += len(result.get("added", []))
            c["completed"] += len(result.get("completed", []))
        self._record({"type": "scan", **{k: v for k, v in result.items() if k != "seen"}})
        log.info("scan: %s", json.dumps(result))
        return result

    def _record(self, event: dict) -> None:
        """Append to the dashboard's history (memory + jsonl under /data)."""
        event.setdefault("at", time.strftime("%Y-%m-%d %H:%M:%S"))
        event.setdefault("t", round(time.time(), 1))
        self.history.append(event)
        if len(self.history) > HISTORY_KEEP:
            del self.history[: len(self.history) - HISTORY_KEEP]
        try:
            with self._history_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as exc:
            log.warning("history append failed: %s", exc)

    def away(self) -> bool:
        if not self.cfg.presence_entity:
            return True
        return self.ha.state(self.cfg.presence_entity) != "home"

    def tick(self) -> None:
        """One 30s beat of the main loop: scan when the interval has elapsed
        and the gates allow. Split out of main() so it's testable."""
        self.loop_beat = time.time()
        s = self.settings
        if time.time() - self.last_scan_t < s["interval_s"]:
            return
        if not s["enabled"]:
            self.last_scan_t = time.time()
            self._gate_skip("disabled")
            return
        if s["presence_gate"] and not self.away():
            self.last_scan_t = time.time()  # re-check one interval later
            self._gate_skip("home")
            return
        self.scan(trigger="interval")

    def _gate_skip(self, reason: str) -> None:
        self.counters["skips_home" if reason == "home" else "skips_off"] += 1
        self.gate = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "t": round(time.time(), 1), "reason": reason}
        self._record({"type": "skip", "reason": reason})

    # -- dashboard --------------------------------------------------------
    def presence_state(self) -> Optional[str]:
        """Presence for the dashboard, cached 30s — the status endpoint is
        polled every few seconds and must not hammer Home Assistant."""
        if not self.cfg.presence_entity:
            return None
        if time.time() - self._presence_t > 30:
            self._presence = self.ha.state(self.cfg.presence_entity)
            self._presence_t = time.time()
        return self._presence

    def status(self) -> dict:
        cfg = self.cfg
        now = time.time()
        state = self.presence_state()
        return {
            "now": time.strftime("%Y-%m-%d %H:%M:%S"),
            "now_t": round(now, 1),
            "started_at": self.started_at,
            "scanning": self.scanning,
            "loop_beat_age_s": round(now - self.loop_beat),
            "next_due_s": max(0, round(self.last_scan_t + self.settings["interval_s"] - now)),
            "presence": {"entity": cfg.presence_entity, "state": state,
                         "away": not cfg.presence_entity or state != "home",
                         "gates_scans": bool(cfg.presence_entity)
                                        and self.settings["presence_gate"]},
            "gate": self.gate,
            "settings": dict(self.settings),
            "items": self.items_view,
            "last": self.last_result,
            "history": self.history,
            "counters": dict(self.counters),
            "crop_t": self.crop_t,
            "config": {"model": cfg.model, "interval_s": cfg.interval_s,
                       "region": list(cfg.region),
                       "change_threshold": cfg.change_threshold,
                       "missing_to_complete": cfg.missing_to_complete,
                       "todo_entity": cfg.todo_entity,
                       "todoist_project": cfg.todoist_project},
        }


# Monitoring dashboard, served at GET /. Self-contained (inline CSS/JS, no
# external assets); data comes from GET /api/status polled every 5s. Colors
# follow the validated reference dataviz palette, light + dark.
DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>whiteboard-sync</title>
<style>
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --series:#2a78d6; --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
}
@media (prefers-color-scheme: dark){:root{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --series:#3987e5;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1020px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:14px}
h1{font-size:17px;font-weight:600;margin:0}
h1 small{color:var(--muted);font-weight:400;font-size:12px;margin-left:8px}
.pill{display:inline-flex;align-items:center;gap:7px;padding:4px 12px;border:1px solid var(--border);
  border-radius:999px;background:var(--surface);font-size:13px;color:var(--ink2)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex:none}
.pill.good .dot{background:var(--good)} .pill.warn .dot{background:var(--warn)}
.pill.crit .dot{background:var(--crit)} .pill.busy .dot{background:var(--series);animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.3}}
.spacer{flex:1}
button{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);
  border:1px solid var(--border);border-radius:8px;padding:6px 14px;cursor:pointer}
button:hover{border-color:var(--axis)} button:disabled{opacity:.45;cursor:default}
#note{font-size:12px;color:var(--ink2);min-height:16px;margin:0 0 12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:12px}
.tile,.card{background:var(--surface);border:1px solid var(--border);border-radius:10px}
.tile{padding:12px 14px}
.tile .label{font-size:12px;color:var(--ink2)}
.tile .value{font-size:21px;font-weight:600;margin:2px 0}
.tile .sub{font-size:12px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
@media(max-width:720px){.grid2{grid-template-columns:1fr}}
.card{padding:14px 16px;margin-bottom:12px}
.card h2{font-size:13px;font-weight:600;margin:0 0 10px;color:var(--ink2)}
.card h2 small{color:var(--muted);font-weight:400;float:right}
ul.items{list-style:none;margin:0;padding:0}
ul.items li{display:flex;align-items:center;justify-content:space-between;gap:8px;
  padding:5px 0;border-top:1px solid var(--grid);font-size:13px}
ul.items li:first-child{border-top:0}
.chip{font-size:11px;color:var(--ink2);border:1px solid var(--border);border-radius:999px;
  padding:1px 8px;white-space:nowrap;display:inline-flex;align-items:center;gap:5px}
.chip .dot{width:7px;height:7px}
.empty{color:var(--muted);font-size:13px}
.setgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:0 22px}
.setrow{display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:6px 0;border-top:1px solid var(--grid);font-size:13px}
.setrow label{color:var(--ink2)}
input[type=checkbox]{width:16px;height:16px;accent-color:var(--series);margin:0}
input[type=number],select{font:inherit;font-size:13px;color:var(--ink);background:var(--page);
  border:1px solid var(--border);border-radius:6px;padding:3px 6px;width:82px}
img#crop{display:block;width:100%;border-radius:8px;border:1px solid var(--border)}
#chartwrap{position:relative}
svg{display:block;width:100%;height:auto}
#tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--axis);
  border-radius:8px;padding:6px 10px;font-size:12px;color:var(--ink);display:none;
  box-shadow:0 2px 10px rgba(0,0,0,.12);white-space:nowrap;z-index:2}
#tip .t{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-size:11px;font-weight:600;color:var(--muted);text-align:left;padding:2px 8px 6px}
td{padding:5px 8px;border-top:1px solid var(--grid);vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.skip td{color:var(--muted)}
td .dot{display:inline-block;margin-right:6px}
footer{color:var(--muted);font-size:12px;margin-top:16px;line-height:1.7}
.scroll{overflow-x:auto}
</style>
</head>
<body>
<main>
<header>
  <h1>whiteboard-sync<small>boards &rarr; Todoist</small></h1>
  <span class="pill" id="pill"><span class="dot"></span><span id="pilltext">connecting&hellip;</span></span>
  <span class="spacer"></span>
  <button id="btn-scan">Scan now</button>
  <button id="btn-dry">Dry run</button>
</header>
<div id="note"></div>

<section class="tiles">
  <div class="tile"><div class="label">Sync state</div><div class="value" id="t-state">&ndash;</div><div class="sub" id="t-state-sub"></div></div>
  <div class="tile"><div class="label" id="t-next-label">Next check</div><div class="value" id="t-next">&ndash;</div><div class="sub" id="t-next-sub"></div></div>
  <div class="tile"><div class="label">Tracked items</div><div class="value" id="t-items">&ndash;</div><div class="sub" id="t-items-sub"></div></div>
  <div class="tile"><div class="label">Last scan</div><div class="value" id="t-last">&ndash;</div><div class="sub" id="t-last-sub"></div></div>
</section>

<div class="grid2">
  <section class="card"><h2>Work board</h2><ul class="items" id="list-work"></ul></section>
  <section class="card"><h2>Personal board</h2><ul class="items" id="list-personal"></ul></section>
</div>

<section class="card"><h2>Controls <small>changes apply immediately and persist</small></h2>
  <div id="settings" class="setgrid"></div>
</section>

<section class="card"><h2>Latest board photo <small id="crop-cap"></small></h2>
  <img id="crop" alt="Cropped whiteboard snapshot" hidden>
  <div class="empty" id="crop-empty">No snapshot captured yet.</div>
</section>

<section class="card"><h2>Change score per scan <small>vs threshold &mdash; details in the table below</small></h2>
  <div id="chartwrap"><svg id="chart" viewBox="0 0 880 170" role="img" style="display:none"
    aria-label="Change score per scan"></svg><div id="tip"></div></div>
  <div class="empty" id="chart-empty">No scans recorded yet.</div>
</section>

<section class="card"><h2>History <small id="counters"></small></h2>
  <div class="scroll"><table>
    <thead><tr><th class="num">Time</th><th>Event</th><th class="num">Diff</th><th>Tasks</th><th class="num">Took</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
</section>

<footer id="meta"></footer>
</main>
<script>
'use strict';
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
let st = null, fetchedAt = 0, cropShown = null;

const ago = (t) => {
  if (!t) return 'never';
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 90) return Math.round(s) + 's ago';
  if (s < 5400) return Math.round(s / 60) + ' min ago';
  if (s < 129600) return (s / 3600).toFixed(1) + ' h ago';
  return Math.round(s / 86400) + ' d ago';
};
const mmss = (s) => { s = Math.max(0, Math.round(s));
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
const tstr = (t) => { const d = new Date(t * 1000);
  const today = new Date().toDateString() === d.toDateString();
  return (today ? '' : (d.getMonth() + 1) + '/' + d.getDate() + ' ') +
    d.toTimeString().slice(0, 8); };
const who = () => ((st.presence.entity || '').replace('person.', '').split('_')[0] || 'you');

function outcome(r) {
  if (r.type === 'settings') return 'settings: ' + Object.entries(r.changed || {})
    .map(([k, v]) => k + '=' + v).join(', ');
  if (r.type === 'skip') return r.reason === 'home'
    ? 'skipped — ' + who() + ' was home' : 'skipped — scheduled scans off';
  if (r.error) return 'error: ' + r.error;
  if (r.obstructed) return 'board obstructed — left alone';
  if (r.changed === false) return 'no change';
  const a = (r.added || []).length, c = (r.completed || []).length;
  let s = r.dry ? 'dry run: would add ' + a + ', complete ' + c
                : a + ' added, ' + c + ' completed';
  if (r.obstructed_ignored) s += ' (obstruction ignored)';
  if (r.failed) s += ' (' + r.failed + ' HA call' + (r.failed > 1 ? 's' : '') + ' failed, will retry)';
  return s;
}
function kind(r) {
  if (r.type === 'skip' || r.type === 'settings' || r.changed === false) return 'muted';
  if (r.error) return 'crit';
  if (r.failed || r.obstructed || r.obstructed_ignored) return 'warn';
  return 'good';
}

const SETTINGS_UI = [
  ['enabled', 'bool', 'Scheduled scans'],
  ['presence_gate', 'bool', 'Only scan while away'],
  ['change_detection', 'bool', 'Skip unchanged frames'],
  ['obstruction_guard', 'bool', 'Skip when obstructed'],
  ['enhance', 'bool', 'Enhance photo (contrast+sharpen)'],
  ['interval_s', 'num', 'Scan interval (s)', { min: 60, max: 86400, step: 60 }],
  ['change_threshold', 'num', 'Change threshold', { min: 0.1, max: 50, step: 0.5 }],
  ['missing_to_complete', 'num', 'Misses to complete', { min: 1, max: 10, step: 1 }],
  ['capture_frames', 'num', 'Frames stacked per photo', { min: 1, max: 32, step: 1 }],
  ['capture_quality', 'num', 'JPEG quality to model', { min: 50, max: 100, step: 1 }],
  ['upscale', 'sel', 'Upscale for model', [['1', '1x'], ['2', '2x']]],
  ['capture_format', 'sel', 'Image format to model', [['jpeg', 'JPEG'], ['png', 'PNG']]],
];
let settingsBuilt = false;
function buildSettings() {
  const host = $('settings');
  for (const [key, type, label, opt] of SETTINGS_UI) {
    const row = el('div', 'setrow');
    const lab = el('label', null, label); lab.htmlFor = 'set-' + key;
    row.appendChild(lab);
    let inp;
    if (type === 'bool') { inp = el('input'); inp.type = 'checkbox'; }
    else if (type === 'num') { inp = el('input'); inp.type = 'number';
      inp.min = opt.min; inp.max = opt.max; inp.step = opt.step; }
    else { inp = el('select');
      for (const [v, l] of opt) { const o = el('option', null, l); o.value = v; inp.appendChild(o); } }
    inp.id = 'set-' + key;
    inp.addEventListener('change', async () => {
      let val;
      if (type === 'bool') val = inp.checked;
      else if (type === 'num') val = Number(inp.value);
      else val = type === 'sel' && key === 'upscale' ? Number(inp.value) : inp.value;
      try {
        const r = await fetch('/api/settings', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ [key]: val }) });
        const j = await r.json();
        const rej = j.rejected && j.rejected[key];
        $('note').textContent = rej ? label + ': ' + rej
          : Object.keys(j.applied || {}).length ? 'Saved: ' + label : '';
      } catch (e) { $('note').textContent = 'Save failed: ' + e; }
      refresh();
    });
    row.appendChild(inp); host.appendChild(row);
  }
  settingsBuilt = true;
}
function syncSettings() {
  if (!settingsBuilt) buildSettings();
  for (const [key, type] of SETTINGS_UI) {
    const inp = $('set-' + key);
    if (document.activeElement === inp) continue;
    if (type === 'bool') inp.checked = !!st.settings[key];
    else inp.value = String(st.settings[key]);
  }
}
const KINDCOLOR = { good: 'var(--good)', warn: 'var(--warn)', crit: 'var(--crit)', muted: 'var(--muted)' };

function pill(cls, text) { const p = $('pill'); p.className = 'pill ' + cls; $('pilltext').textContent = text; }

function render() {
  const last = st.last && st.last.at ? st.last : null;
  const p = st.presence;
  const stalled = st.loop_beat_age_s > st.config.interval_s + 900;

  if (stalled) pill('crit', 'Scan loop stalled — restart the container');
  else if (st.scanning) pill('busy', 'Scanning now…');
  else if (!st.settings.enabled) pill('pause', 'Scheduled scans off');
  else if (last && last.error) pill('warn', 'Last scan failed — retrying on schedule');
  else if (last && last.failed) pill('warn', 'HA push failed — will retry');
  else if (p.gates_scans && !p.away) pill('pause', 'Paused — ' + who() + ' is home');
  else pill('good', 'Watching the boards');

  $('t-state').textContent = stalled ? 'Stalled' : st.scanning ? 'Scanning' :
    !st.settings.enabled ? 'Off' : (p.gates_scans && !p.away) ? 'Paused' : 'Active';
  $('t-state-sub').textContent = p.entity
    ? who() + ' is ' + (p.state === null ? 'unknown' : p.state === 'home' ? 'home' : 'away')
    : 'no presence gate';

  $('t-next-label').textContent = !st.settings.enabled ? 'Next enabled check'
    : (p.gates_scans && !p.away) ? 'Next presence check' : 'Next scan';
  $('t-next-sub').textContent = 'every ' + Math.round(st.settings.interval_s / 60) + ' min' +
    (st.settings.presence_gate ? ' while away' : '');

  const items = st.items || [];
  const w = items.filter((i) => i.board === 'work').length;
  $('t-items').textContent = items.length;
  $('t-items-sub').textContent = w + ' work · ' + (items.length - w) + ' personal';

  $('t-last').textContent = last ? ago(last.t) : 'never';
  $('t-last-sub').textContent = last ? outcome(last) : 'no scans since start';

  for (const board of ['work', 'personal']) {
    const ul = $('list-' + board); ul.textContent = '';
    const rows = items.filter((i) => i.board === board);
    if (!rows.length) { ul.appendChild(el('li', 'empty', 'nothing tracked')); continue; }
    for (const it of rows) {
      const li = el('li'); li.appendChild(el('span', null, it.text));
      if (it.missing > 0) {
        const chip = el('span', 'chip');
        const d = el('span', 'dot'); d.style.background = 'var(--serious)';
        chip.appendChild(d);
        chip.appendChild(document.createTextNode(
          'missing ' + it.missing + '/' + st.settings.missing_to_complete));
        li.appendChild(chip);
      }
      ul.appendChild(li);
    }
  }

  if (st.crop_t) {
    if (st.crop_t !== cropShown) { $('crop').src = '/crop.jpg?t=' + st.crop_t; cropShown = st.crop_t; }
    $('crop').hidden = false; $('crop-empty').hidden = true;
    $('crop-cap').textContent = 'captured ' + ago(st.crop_t);
  }

  syncSettings(); chart(); table();

  const c = st.counters, cfg = st.config, set = st.settings;
  $('counters').textContent = 'since start: ' + c.scans + ' scans · ' + c.changed +
    ' processed · ' + c.unchanged + ' unchanged · ' + c.obstructed + ' obstructed · ' +
    c.errors + ' errors · ' + c.skips_home + ' home-skips · ' + c.skips_off + ' off-skips';
  $('meta').textContent = 'model ' + cfg.model + ' · crop region ' + cfg.region.join(',') +
    ' · photo: ' + set.capture_frames + ' frames stacked, ' + set.capture_format +
    (set.capture_format === 'jpeg' ? ' q' + set.capture_quality : '') +
    (set.upscale > 1 ? ', ' + set.upscale + 'x upscale' : '') +
    (set.enhance ? ', enhanced' : '') +
    ' · Todoist project "' + cfg.todoist_project + '" via ' + cfg.todo_entity +
    ' · service up since ' + st.started_at +
    ' · tasks this session: ' + c.added + ' added, ' + c.completed + ' completed';
  countdown();
}

function chart() {
  const svg = $('chart');
  const scans = (st.history || []).filter((h) => h.type === 'scan' && h.diff != null).slice(-48);
  svg.textContent = '';
  const has = scans.length > 0;
  svg.style.display = has ? '' : 'none';
  $('chart-empty').style.display = has ? 'none' : '';
  if (!has) return;
  const W = 880, H = 170, L = 10, R = 84, T = 12, B = 26;
  const thr = st.config.change_threshold;
  const maxV = Math.max(thr * 1.5, ...scans.map((s) => s.diff)) * 1.05;
  const y = (v) => H - B - (v / maxV) * (H - T - B);
  const slot = (W - L - R) / scans.length;
  const bw = Math.max(3, Math.min(24, slot - 2));
  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, at) => { const n = document.createElementNS(NS, tag);
    for (const k in at) n.setAttribute(k, at[k]); return n; };

  const thrY = y(thr);
  svg.appendChild(S('line', { x1: L, x2: W - R + 6, y1: thrY, y2: thrY,
    stroke: 'var(--axis)', 'stroke-width': 1 }));
  const lbl = S('text', { x: W - R + 10, y: thrY + 4, fill: 'var(--muted)', 'font-size': 11 });
  lbl.textContent = 'threshold ' + thr;
  svg.appendChild(lbl);

  scans.forEach((s, i) => {
    const x = L + i * slot + (slot - bw) / 2;
    const top = Math.min(y(s.diff), H - B - 1), h = H - B - top;
    const r = Math.min(4, bw / 2, h);
    svg.appendChild(S('path', { fill: 'var(--series)', d:
      'M' + x + ',' + (H - B) + ' L' + x + ',' + (top + r) +
      ' Q' + x + ',' + top + ' ' + (x + r) + ',' + top +
      ' L' + (x + bw - r) + ',' + top +
      ' Q' + (x + bw) + ',' + top + ' ' + (x + bw) + ',' + (top + r) +
      ' L' + (x + bw) + ',' + (H - B) + ' Z' }));
    const hit = S('rect', { x: L + i * slot, y: T, width: slot, height: H - T - B,
      fill: 'transparent' });
    hit.addEventListener('mouseenter', () => tip(s, L + i * slot + slot / 2, top));
    hit.addEventListener('mouseleave', () => { $('tip').style.display = 'none'; });
    svg.appendChild(hit);
  });

  svg.appendChild(S('line', { x1: L, x2: W - R + 6, y1: H - B, y2: H - B,
    stroke: 'var(--axis)', 'stroke-width': 1 }));
  const t0 = S('text', { x: L, y: H - 8, fill: 'var(--muted)', 'font-size': 11 });
  t0.textContent = tstr(scans[0].t);
  svg.appendChild(t0);
  if (scans.length > 1) {
    const t1 = S('text', { x: W - R + 6, y: H - 8, fill: 'var(--muted)', 'font-size': 11,
      'text-anchor': 'end' });
    t1.textContent = tstr(scans[scans.length - 1].t);
    svg.appendChild(t1);
  }
}

function tip(s, vx, vy) {
  const box = $('tip'), wrap = $('chartwrap'), svg = $('chart');
  box.textContent = '';
  box.appendChild(el('div', 't', tstr(s.t)));
  box.appendChild(el('div', null, 'diff ' + s.diff));
  box.appendChild(el('div', null, outcome(s)));
  const k = svg.getBoundingClientRect().width / 880;
  box.style.display = 'block';
  const bx = Math.min(Math.max(vx * k - 60, 0), wrap.clientWidth - box.offsetWidth - 4);
  box.style.left = bx + 'px';
  box.style.top = Math.max(vy * k - box.offsetHeight - 10, 0) + 'px';
}

function table() {
  const tb = $('rows'); tb.textContent = '';
  const rows = (st.history || []).slice(-60).reverse();
  if (!rows.length) { const tr = el('tr'); const td = el('td', 'empty', 'nothing yet');
    td.colSpan = 5; tr.appendChild(td); tb.appendChild(tr); return; }
  for (const r of rows) {
    const tr = el('tr', kind(r) === 'muted' ? 'skip' : null);
    tr.appendChild(el('td', 'num', tstr(r.t)));
    const ev = el('td');
    const d = el('span', 'dot'); d.style.background = KINDCOLOR[kind(r)];
    ev.appendChild(d); ev.appendChild(document.createTextNode(outcome(r)));
    tr.appendChild(ev);
    tr.appendChild(el('td', 'num', r.diff != null ? r.diff.toFixed(1) : ''));
    const a = (r.added || []).length, c = (r.completed || []).length;
    tr.appendChild(el('td', null, r.type === 'scan' && (a || c)
      ? (a ? '+' + a + ' ' : '') + (c ? '−' + c : '') : ''));
    tr.appendChild(el('td', 'num', r.took_s != null ? r.took_s + 's' : ''));
    tb.appendChild(tr);
  }
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    st = await r.json(); fetchedAt = Date.now();
    render();
  } catch (e) { pill('crit', 'Cannot reach whiteboard-sync'); }
}

function countdown() {
  if (!st) return;
  if (st.scanning) { $('t-next').textContent = 'now'; return; }
  const left = st.next_due_s - (Date.now() - fetchedAt) / 1000;
  $('t-next').textContent = mmss(left);
}

async function doScan(dry) {
  $('btn-scan').disabled = $('btn-dry').disabled = true;
  $('note').textContent = (dry ? 'Dry-run scan' : 'Scan') +
    ' started — the vision model can take a few minutes on first load…';
  try {
    const r = await fetch('/scan?force=1' + (dry ? '&dry=1' : ''), { method: 'POST' });
    const j = await r.json();
    $('note').textContent = 'Result: ' + outcome(j);
  } catch (e) { $('note').textContent = 'Scan request failed: ' + e; }
  $('btn-scan').disabled = $('btn-dry').disabled = false;
  refresh();
}
$('btn-scan').addEventListener('click', () => doScan(false));
$('btn-dry').addEventListener('click', () => doScan(true));

refresh();
setInterval(refresh, 5000);
setInterval(countdown, 1000);
</script>
</body>
</html>
"""


def serve_http(syncer: Syncer) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            self._bytes(code, json.dumps(obj).encode(), "application/json")

        def _bytes(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._bytes(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/healthz":
                # A stale loop beat means the scan loop is dead or stuck; the
                # longest legit gap is one interval plus the 600s model timeout.
                age = time.time() - syncer.loop_beat
                stalled = age > syncer.settings["interval_s"] + 900
                self._json(500 if stalled else 200,
                           {"status": "loop stalled" if stalled else "ok",
                            "loop_beat_age_s": round(age)})
            elif path == "/api/status":
                self._json(200, syncer.status())
            elif path == "/state":
                self._json(200, {"items": syncer.items_view, "last": syncer.last_result})
            elif path == "/crop.jpg":
                try:
                    self._bytes(200, syncer.crop_path.read_bytes(), "image/jpeg")
                except OSError:
                    self._json(404, {"error": "no crop captured yet"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            q = parse_qs(url.query, keep_blank_values=True)

            def flag(name: str) -> bool:
                vals = q.get(name)
                return bool(vals) and vals[0].lower() not in ("0", "false")

            if url.path == "/scan":
                self._json(200, syncer.scan(force=flag("force"), dry=flag("dry")))
            elif url.path == "/api/settings":
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    patch = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(patch, dict):
                        raise ValueError("want a JSON object of settings")
                except (json.JSONDecodeError, ValueError) as exc:
                    self._json(400, {"error": f"bad body: {exc}"})
                    return
                self._json(200, syncer.update_settings(patch))
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
    log.info("whiteboard-sync up: model %s, every %ss while %s is away; region %s; "
             "dashboard on :%s", cfg.model, cfg.interval_s,
             cfg.presence_entity or "(no presence gate)", cfg.region, HTTP_PORT)

    while True:
        time.sleep(30)
        try:
            syncer.tick()
        except Exception:  # noqa: BLE001 — the scan loop must survive anything a tick throws
            log.exception("tick failed")


if __name__ == "__main__":
    main()
