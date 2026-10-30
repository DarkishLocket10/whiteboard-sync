"""whiteboard-sync — keep Todoist in sync with the office whiteboards.

Reads the boards through kinect-knob's full-resolution camera snapshot
(``GET /api/snapshot`` — unmirrored, so the writing is readable), has a
local Ollama vision model transcribe the open items (zero marginal cost,
fully private — chosen because scans run frequently), and reconciles them
into Todoist via Home Assistant:

* LEFT board  -> work items,     tagged ["work"]
* RIGHT board -> personal items, tagged ["personal"]
* New item on a board            -> ``todoist.new_task`` into the Inbox,
  after ``sightings_to_add`` consecutive reads (guards against OCR
  hallucinations and fragments of half-hidden lines becoming tasks)
* Checkbox TICKED (check/X/fill) -> ``todo.update_item`` completes the task
  after ``ticked_to_complete`` sightings (default 2) on CLEAR reads — on an
  obstructed board checkbox judgments are untrustworthy and are ignored in
  both directions; the item stays tracked as "done" while it remains on the
  board so tick misreads can't re-add it
* Item ERASED                    -> after it has been missing for
  ``WB_MISSING_TO_COMPLETE`` consecutive scans, completed the same way
  (the guard absorbs one bad OCR read)

Scans run every ``WB_INTERVAL_S`` seconds, only while ``WB_PRESENCE_ENTITY``
is away (that's also when nobody blocks the camera's view of the boards),
and the vision read is skipped entirely when ToF depth says someone is
physically at a board (their body hides parts of lines and the fragments
become tasks), when the frame is too dim to trust (dim reads add partial
garbage), or when the cropped board region hasn't changed since the last
processed scan — so vision costs accrue only when something was actually
written or erased.

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
import re
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
EDIT_LOCK_TIMEOUT_S = 5.0  # how long a dashboard edit waits out a running scan
# When ONE read would complete this many items by ABSENCE while still seeing
# other items on the board, suspect a garbage read and demand an extra miss.
MASS_ERASE_MIN = 4
# A tick STREAK may only start within this many scans of a real frame change
# (diff >= threshold). A new mark necessarily changes pixels; a tick that
# "appears" while the scene has been static for longer is a misread. The
# window > 0 covers detection lag (a tick missed on its change frame and
# first read a scan or two later).
TICK_START_CHANGE_WINDOW = 2
# Low light (mean crop luminance, 0-255; measured 2026-07-09: lit office
# ~140, office at night with residual light ~31). The Kinect exposes no
# exposure control (freenect2 binding has none — long exposures are
# impossible), so frame STACKING is the synthetic long exposure: at
# luminance ~31 a single frame read garbled text and called almost every
# checkbox ticked, while a 32-frame stack transcribed perfectly. Even
# stacked, dark-frame CHECKBOX judgments stay completion-grade unreliable,
# and a pitch-black frame reads as an EMPTY board — which absence logic
# would treat as a real wipe.
DIM_LUMA = 55        # default dim_luma setting. Below it: skip the scan when
                     # dim_skip is on (dim reads still ADD partial garbage —
                     # protection only deferred completions); with dim_skip
                     # off, stack LOWLIGHT_FRAMES + defer all completions
DARK_LUMA = 18       # below: unreadable — always skip, even when forced
LOWLIGHT_FRAMES = 32
# Depth guard: kinect-knob's /api/region_depth (self-illuminated ToF, aligned
# to the color frame) says whether something PHYSICALLY sits in front of each
# board half. p90 of a half is the board plane (holds as long as ≥10% of it
# is visible); p10 more than this much nearer means an object covers ≥10% of
# the half. Field values 2026-07-09: clear plane p10≈1915/p90≈1988; a person
# in front read p10≈988 against p90≈1997.
DEPTH_OBSTRUCTION_MM = 400


def depth_blocked(stats: Optional[dict]) -> bool:
    """Is a board half physically obstructed, per region_depth stats? Falls
    to False (no evidence) on missing/degenerate data — the model's visual
    flag still applies independently."""
    if not stats or stats.get("valid_frac", 0) < 0.5 or "p10_mm" not in stats:
        return False
    return stats["p10_mm"] < stats["p90_mm"] - DEPTH_OBSTRUCTION_MM
# Presence: kinect-knob's /api/presence reports, per REGION, the fraction of
# it sitting in front of the learned empty plane. That is a different and far
# more sensitive question than depth_blocked() above asks. depth_blocked
# compares percentiles WITHIN one frame, so it only fires once an object
# covers ~10% of a board half — a head leaning in front of a line or two
# covers a few percent and sails straight through, which is how partially
# hidden lines kept being transcribed into fragmentary Todoist tasks. Against
# a learned plane, a few percent is unmistakable.
PRESENCE_OCCUPANCY = 0.015   # default presence_occupancy setting
PRESENCE_MARGIN_PX = 120     # widen the asked-about region by this much: a
                             # head just OUTSIDE the crop is still about to be
                             # inside it, and its shadow/arm already is
PRESENCE_POLL_S = 3.0        # background poll cadence (one cheap HTTP call)
PRESENCE_STALE_S = 30.0      # readings older than this tell us nothing
# kinect-knob switches its camera off overnight (KK_OFF_WINDOW) and answers
# 503 {"camera": "off"} meanwhile: poll that slowly, and don't scan at all.
PRESENCE_OFF_POLL_S = 60.0

# Framing. The Kinect v2 colour camera has no optical zoom, pan or exposure
# control the service can use, so "zoom" and "pan" are the crop rectangle cut
# from the fixed 1920x1080 frame — a tighter crop spends the model's pixel
# budget on the boards instead of the wall above them.
FRAME_W, FRAME_H = 1920, 1080
MIN_REGION_W, MIN_REGION_H = 160, 120   # smaller is a slip of the mouse
# The largest image ever sent to the model. The default 720x530 crop at 2x
# (1440x1060, ~1.53 MP) is known to fit num_ctx 6144 together with the
# prompt; a bigger image silently truncates the prompt (see OllamaReader), so
# upscaling is clamped to this rather than trusted.
MODEL_MAX_PIXELS = 1_600_000


def board_split(region: tuple[int, int, int, int], split: int = 0) -> int:
    """The x (full-frame px) dividing the work board from the personal one.
    0 — or anything outside the region — means the region's midpoint. Only
    the presence and depth guards use it; the model itself sees the whole
    crop and tells left from right on its own."""
    x1, _, x2, _ = region
    return split if x1 < split < x2 else (x1 + x2) // 2


# The writing usually fills only the top of each board; the rest is blank
# board, and whoever sits at the desk in front of it shows up there (live
# 2026-10-30: the chair back and the user's head sit below the last line on
# both boards all day). ``items_bottom`` (full-frame y, 0 = off) marks where
# the writing ends, so the presence and depth guards watch only the written
# part — a head in front of blank board space hides nothing. Below that line
# the asked-about region only reaches this far, not the full side margin.
ITEMS_BOTTOM_MARGIN_PX = 30


def items_zone(region: tuple[int, int, int, int],
               items_bottom: int = 0) -> tuple[int, int, int, int]:
    """The part of the crop that holds writing: the region cut off at
    ``items_bottom``, or the whole region when that is 0 / outside it."""
    x1, y1, x2, y2 = region
    if y1 < items_bottom < y2:
        y2 = items_bottom
    return x1, y1, x2, y2


def model_scale(w: int, h: int, upscale: float) -> float:
    """The resize factor actually applied before the model: the requested
    upscale, clamped so the result stays within MODEL_MAX_PIXELS (which
    shrinks a crop that is already bigger than that)."""
    return min(float(upscale), (MODEL_MAX_PIXELS / max(1, w * h)) ** 0.5)


class CameraOff(requests.RequestException):
    """kinect-knob is up but its camera is deliberately off (the scheduled
    night window). A RequestException so every existing "snapshot failed"
    handler still catches it; scans and the scheduler single it out to stay
    quiet instead of logging an error every interval all night."""


def camera_off_detail(r) -> Optional[str]:
    """The 'scheduled off' detail if ``r`` is kinect-knob's camera-off 503."""
    if getattr(r, "status_code", 200) != 503:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if isinstance(data, dict) and data.get("camera") == "off":
        return str(data.get("detail") or "camera off")
    return None


class PresenceWatcher(threading.Thread):
    """Watches, continuously, whether a human is in front of either board.

    Polling in the background rather than asking once at scan time is the
    whole point. A scan takes seconds — fetch a stacked photo, run a vision
    model — and the old single pre-flight check could only ever say "clear a
    moment ago". Somebody who sat down during the capture, or who had just
    stood up as it began, produced exactly the half-occluded frame that turns
    into a fragmentary task. Here the scan can ask "has this board been clear
    for the last N seconds", and ask again afterwards about the window the
    photo actually spans.

    Degrades honestly: an unreachable or too-old endpoint reports ``unknown``,
    never ``clear``, and the caller decides what to do with no evidence.
    """

    def __init__(self, cfg, settings: dict):
        super().__init__(name="presence", daemon=True)
        self.cfg = cfg
        self._settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.started_t = time.time()
        self.supported: Optional[bool] = None   # None = not yet established
        self.error: Optional[str] = None
        # board -> {"occupancy", "blocked", "t", "last_blocked_t", "plane_mm"}
        self.boards: dict[str, dict] = {}
        self.scene: dict = {}                   # whole-room presence, if offered
        self.camera_off: Optional[str] = None   # kinect-knob's "off until" detail

    # -- geometry -------------------------------------------------------
    def regions(self) -> dict[str, tuple[int, int, int, int]]:
        """The two board halves, widened by PRESENCE_MARGIN_PX — and, with
        an items zone set, cut off just below the last written line."""
        bottom = int(self._settings.get("items_bottom", 0) or 0)
        x1, y1, x2, y2 = items_zone(self.cfg.region, bottom)
        mid = board_split(self.cfg.region, self._settings.get("split", 0))
        m = PRESENCE_MARGIN_PX
        mb = ITEMS_BOTTOM_MARGIN_PX if y2 < self.cfg.region[3] else m
        return {
            "work": (x1 - m, y1 - m, mid + m, y2 + mb),
            "personal": (mid - m, y1 - m, x2 + m, y2 + mb),
        }

    # -- polling --------------------------------------------------------
    def run(self) -> None:
        while not self._stop.wait(PRESENCE_OFF_POLL_S if self.camera_off
                                  else PRESENCE_POLL_S):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 — a watcher must never die
                log.debug("presence poll failed", exc_info=True)

    def stop(self) -> None:
        self._stop.set()

    def poll(self) -> None:
        if not self.cfg.kinect_url:
            return
        threshold = float(self._settings.get("presence_occupancy", PRESENCE_OCCUPANCY))
        now = time.time()
        fresh: dict[str, dict] = {}
        scene: dict = {}
        for board, (rx1, ry1, rx2, ry2) in self.regions().items():
            try:
                r = requests.get(f"{self.cfg.kinect_url}/api/presence",
                                 params={"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
                                 timeout=5)
                off = camera_off_detail(r)
                if off is not None:
                    with self._lock:
                        if not self.camera_off:
                            log.info("kinect-knob camera is off (%s) — "
                                     "presence polling slowed", off)
                        self.camera_off = off
                        self.error = "camera off"
                        self.boards = {}
                    return
                if r.status_code == 404:
                    # Either an older kinect-knob with no /api/presence, or a
                    # backend with no depth. Either way: fall back.
                    with self._lock:
                        self.supported = False
                        self.error = "kinect-knob has no /api/presence"
                    return
                r.raise_for_status()
                data = r.json()
            except (requests.RequestException, ValueError) as exc:
                with self._lock:
                    self.error = str(exc)
                    # Drop the readings now rather than letting them age out:
                    # a camera that is restarting has no opinion on who is
                    # at the boards, and "unknown" is the honest answer.
                    self.boards = {}
                return
            region = data.get("region") or {}
            occ = float(region.get("occupancy", 0.0))
            prev = self.boards.get(board, {})
            blocked = occ >= threshold
            fresh[board] = {
                "occupancy": round(occ, 4),
                "blocked": blocked,
                "plane_mm": region.get("plane_mm"),
                "nearest_mm": region.get("nearest_mm"),
                "t": now,
                "last_blocked_t": now if blocked else prev.get("last_blocked_t", 0.0),
            }
            scene = {"present": bool(data.get("present")),
                     "occupancy": data.get("occupancy"),
                     "since_seen_s": data.get("since_seen_s"),
                     "available": bool(data.get("available"))}
        with self._lock:
            if self.error is not None or self.camera_off or not self.boards:
                # Back from an outage (or the first answer): the settle window
                # restarts — time nobody was watching is not time "clear".
                self.started_t = now
            self.supported = True
            self.error = None
            self.camera_off = None
            self.boards = fresh
            self.scene = scene

    # -- queries --------------------------------------------------------
    def status(self, board: str) -> str:
        """'blocked' | 'clear' | 'unknown' — right now."""
        with self._lock:
            rec = self.boards.get(board)
            supported = self.supported
        if not supported or rec is None:
            return "unknown"
        if time.time() - rec["t"] > PRESENCE_STALE_S:
            return "unknown"
        return "blocked" if rec["blocked"] else "clear"

    def clear_for_s(self, board: str) -> Optional[float]:
        """Seconds this board has been continuously clear, or None if unknown."""
        with self._lock:
            rec = self.boards.get(board)
            supported = self.supported
        if not supported or rec is None or time.time() - rec["t"] > PRESENCE_STALE_S:
            return None
        if rec["blocked"]:
            return 0.0
        # Never seen blocked: the honest answer is "for as long as I have been
        # watching", not infinity — a watcher that started ten seconds ago has
        # not yet earned a thirty-second settle window.
        last = max(rec.get("last_blocked_t") or 0.0, self.started_t)
        return time.time() - last

    def blocked_boards(self, min_clear_s: float) -> list[str]:
        """Boards that are occupied, or that have not been clear long enough.

        ``min_clear_s`` is a settle window: somebody who just stood up is
        still leaning over the desk, and the frame they left behind is the one
        with half a sleeve across a line.
        """
        out = []
        for board in ("work", "personal"):
            state = self.status(board)
            if state == "blocked":
                out.append(board)
                continue
            if state == "clear":
                clear = self.clear_for_s(board)
                if clear is not None and clear < min_clear_s:
                    out.append(board)
        return out

    def blocked_since(self, since_t: float) -> list[str]:
        """Boards somebody stepped in front of at any point since ``since_t``.

        The post-capture question: the photo is already taken, and the only
        thing that matters is whether anybody was in shot while it was.
        """
        out = []
        with self._lock:
            supported = self.supported
            boards = dict(self.boards)
        if not supported:
            return []
        for board, rec in boards.items():
            if rec.get("last_blocked_t", 0.0) >= since_t:
                out.append(board)
        return sorted(out)

    def snapshot(self) -> dict:
        with self._lock:
            boards = {b: dict(r) for b, r in self.boards.items()}
            out = {"supported": self.supported, "error": self.error,
                   "camera_off": self.camera_off,
                   "scene": dict(self.scene)}
        now = time.time()
        for board, rec in boards.items():
            rec["age_s"] = round(now - rec["t"], 1)
            rec["clear_for_s"] = (None if rec["blocked"] else round(
                now - max(rec.get("last_blocked_t") or 0.0, self.started_t), 1))
        out["boards"] = boards
        out["margin_px"] = PRESENCE_MARGIN_PX
        return out


HISTORY_KEEP = 400  # scan/skip records kept in memory and in data/history.jsonl

# Runtime-tunable settings (dashboard toggles, persisted in data/settings.json;
# env vars provide the defaults). key -> bool | (type, lo, hi) | enum tuple.
SETTING_BOUNDS = {
    "enabled": bool,             # master switch for scheduled scans
    "presence_gate": bool,       # only scan while the presence entity is away
    "change_detection": bool,    # skip the vision read when the crop hasn't changed
    "obstruction_guard": bool,   # skip reconcile when the model sees an obstruction
    "enhance": bool,             # autocontrast + unsharp mask on the crop
    "due_today": bool,           # new tasks get due "today" (Todoist Today view)
    "interval_s": (int, 60, 86400),
    "change_threshold": (float, 0.1, 50.0),
    "missing_to_complete": (int, 1, 10),
    "ticked_to_complete": (int, 1, 10),  # tick sightings before completing
    "sightings_to_add": (int, 1, 10),    # reads a NEW item needs before its task is created
    "tick_confirm": bool,                # re-photograph + re-read before tick completions
    "low_light_boost": bool,             # dim frame -> re-capture with a 32-frame stack
    "depth_guard": bool,                 # ToF depth says a board is physically blocked
    "depth_skip": bool,                  # someone at a board -> skip the scan entirely
    "presence_occupancy": (float, 0.002, 0.5),   # region occupancy that counts as "a human is there"
    "presence_clear_s": (int, 0, 600),           # a board must have been clear this long
    "presence_recheck": bool,                    # re-check AFTER the photo, not just before
    "presence_required": bool,                   # no presence evidence -> skip rather than read
    "dim_skip": bool,                    # dim frame -> skip instead of reading protected
    "dim_luma": (int, 10, 150),          # mean luma below which a frame counts as dim
    "capture_frames": (int, 1, 32),      # frames stacked by kinect-knob per photo
    "capture_quality": (int, 50, 100),   # JPEG quality sent to the vision model
    "upscale": (float, 1.0, 3.0),        # LANCZOS upscale before the model (capped, see MODEL_MAX_PIXELS)
    "capture_format": ("jpeg", "png"),   # encoding sent to the vision model
    "region": "region",                  # crop x1,y1,x2,y2 in the 1920x1080 frame (the "zoom + pan")
    "split": "split",                    # x dividing work|personal for the presence guards; 0 = midpoint
    "items_bottom": "items_bottom",      # y where the writing ends (guards ignore below); 0 = whole crop
    "presence_overrides_model": bool,    # depth says the writing is clear -> ignore the model's "obstructed"
}
# What the framing editor previews as a draft and saves together.
FRAMING_KEYS = ("region", "split", "items_bottom", "enhance", "upscale", "capture_quality",
                "capture_format", "capture_frames")

# Tick detection is a PER-ITEM forced choice, not two separate lists. The
# old "open list + done list" schema let the model transcribe a whole board
# without ever judging a single checkbox, and in production it put every
# item in "open" (benchmarked 2026-07-08 on a live frame: 10/21 checkbox
# states correct, all 11 ticks missed). Requiring {"text", "checkbox"} per
# item fixed that, and naming the mark SHAPES as enum values (a thin slash
# doesn't activate a generic "marked" concept, but activates "slash")
# scored 77/84 over four clear frames with ZERO false "marked" — false
# negatives are the safe direction, because a missed tick still completes
# later through the erased-item path. Two variants that scored worse and
# flipped errors to false POSITIVES (whole board read as ticked, which
# would complete live tasks): a free-text "describe the mark first" field,
# and a six-value enum with "struck_text" as its own checkbox category.
# Don't reintroduce either; benchmark any change with tools/bench_read.py.
READ_PROMPT = """\
This photo shows two whiteboards side by side.
The LEFT board holds WORK to-do items. The RIGHT board holds PERSONAL to-do items.
Every to-do item has a small hand-drawn square checkbox at the START of its line.

Transcribe every readable to-do item into the "work" and "personal" arrays,
one object per item, top to bottom. Write "text" as a short task phrase
without bullet or checkbox characters, fixing obvious handwriting artifacts.

Then, for EACH item separately, look closely at the small square at the start
of ITS line and set "checkbox":
- "empty": the inside of the square is blank white — no stroke at all.
- "slash": one thin diagonal or straight line crosses the square (a hasty
  tick often looks like this).
- "check_x_or_scribble": a check mark, an X, or a scribble in the square;
  ALSO use this when the item's text is struck through or crossed out,
  whatever its checkbox looks like.
Decide each checkbox on its own evidence; most boards contain a mix.

Skip headings, name tags, stickers, photos, and anything that is not a
to-do list item. Erased items are not listed.
A person or object may hide part of a board. Still transcribe every item you
can actually read — NEVER return empty arrays just because a board is
partially hidden. Read around the blockage.
After transcribing, judge whether the ITEM LISTS themselves are blocked:
set "work_obstructed" to true if a person or object hides or overlaps any of
the LEFT board's written items, their checkboxes, or the space immediately
around/between them — and "personal_obstructed" likewise for the RIGHT
board. A person or object in front of BLANK board space, away from all the
written items, does NOT count as obstruction — otherwise false."""

_READ_ITEM = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "checkbox": {"type": "string",
                     "enum": ["empty", "slash", "check_x_or_scribble"]},
    },
    "required": ["text", "checkbox"],
    "additionalProperties": False,
}

READ_SCHEMA = {
    "type": "object",
    "properties": {
        # transcriptions FIRST: the obstruction judgment must not talk the
        # model out of reading the items it can see
        "work": {"type": "array", "items": _READ_ITEM},
        "personal": {"type": "array", "items": _READ_ITEM},
        "work_obstructed": {"type": "boolean"},
        "personal_obstructed": {"type": "boolean"},
    },
    "required": ["work", "personal", "work_obstructed", "personal_obstructed"],
    "additionalProperties": False,
}


MIN_ITEM_CHARS = 2   # alphanumerics a read line needs to count as an item


def normalise_read(raw: dict) -> dict:
    """Model output -> the reconciler's shape: {"work": [open texts],
    "work_done": [ticked texts], ..., "*_obstructed": bool}. Tolerates junk
    entries (non-dict items, blank text) rather than failing the scan.
    ``{board}_order`` keeps the model's top-to-bottom transcription order
    with open and done interleaved — the split lists lose it, and the
    dashboard sorts items by their position on the physical board."""
    seen = {"work_obstructed": bool(raw.get("work_obstructed")),
            "personal_obstructed": bool(raw.get("personal_obstructed"))}
    for board in ("work", "personal"):
        open_, done, order = [], [], []
        for entry in raw.get(board) or []:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text") or "").strip()
            # A lone letter is never a to-do: "U" — a stroke of a half-wiped
            # board — became a real Todoist task on 2026-10-26.
            if len(re.sub(r"[^0-9A-Za-z]", "", text)) >= MIN_ITEM_CHARS:
                # any named mark shape counts as ticked; a missing/unknown
                # checkbox value degrades to open (the safe direction)
                box = entry.get("checkbox") or "empty"
                (done if box in ("slash", "check_x_or_scribble") else open_
                 ).append(text)
                order.append(text)
        seen[board] = open_
        seen[f"{board}_done"] = done
        seen[f"{board}_order"] = order
    return seen


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
    ticked_to_complete: int = 1

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
            # 2, not 1: a single-frame tick false positive was observed live
            # (2026-07-08, "Pulse leaderboard update") — one misread frame
            # must not complete a real task.
            ticked_to_complete=int(env("WB_TICKED_TO_COMPLETE", "2")),
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

    def open_items(self, entity: str) -> Optional[list[str]]:
        """Summaries of the to-do list's OPEN items, or None if HA can't say."""
        try:
            r = self._s.post(f"{self._url}/api/services/todo/get_items?return_response",
                             json={"entity_id": entity, "status": ["needs_action"]},
                             timeout=15)
            r.raise_for_status()
            items = r.json()["service_response"][entity]["items"]
            return [i["summary"] for i in items]
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            log.warning("HA todo.get_items failed: %s", exc)
            return None


READ_TIMEOUT_S = 240
MODEL_FAILING_AFTER = 3   # consecutive failed reads before /healthz goes red


class BoardReader:
    """Local Ollama vision call: JPEG in, {'work': [...], 'personal': [...]} out.

    Uses Ollama structured outputs (the ``format`` field takes a JSON schema)
    so the reply parses without prompt gymnastics; temperature 0 keeps
    transcriptions stable between scans, which the fuzzy reconciler relies on."""

    def __init__(self, url: str, model: str):
        self._url = url.rstrip("/")
        self._model = model

    def unload(self) -> None:
        """Ask Ollama to drop the model so the next read starts a fresh
        runner. 2026-10-28 21:35 the llama-server runner wedged (another
        client had just reloaded it with a bigger context): every request
        hung until its timeout for ~44 h, text-only prompts included, and
        ``ollama stop`` was the whole fix."""
        try:
            requests.post(f"{self._url}/api/generate",
                          json={"model": self._model, "keep_alive": 0}, timeout=30)
            log.warning("vision model unloaded after a timed-out read — the "
                        "next read loads a fresh runner")
        except requests.RequestException as exc:
            log.warning("could not unload the vision model: %s", exc)

    def read(self, jpeg: bytes) -> dict:
        try:
            return self._read(jpeg)
        except requests.Timeout:
            self.unload()
            raise

    def _read(self, jpeg: bytes) -> dict:
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
            # (connect, read). A warm read takes ~10 s and a cold load ~45 s;
            # 600 s only ever measured a wedged runner, holding the scan lock
            # (and the dashboard's edits) for ten minutes per attempt.
            timeout=(5, READ_TIMEOUT_S),
        )
        r.raise_for_status()
        content = r.json()["message"].get("content", "")
        if not content.strip():
            raise RuntimeError("model returned empty content")
        return normalise_read(json.loads(content))


# --------------------------------------------------------------- matching
# This writer's lowercase "i" is a tiny detached dot with almost no stem, so
# the model regularly transcribes it as a period: "Organ.ze K.tchen Sp.ces",
# "Rev.ew", "Ed.t", "F.n.sh", "n.cer" are all real reads off these boards. A
# period BETWEEN two letters is never punctuation in a to-do phrase, so it is
# safe to read it back as an "i" — but for COMPARISON ONLY. Stored text stays
# byte-identical to the Todoist summary (see CLAUDE.md), so this never
# rewrites an item from a guess; it only stops the drift breaking a match.
_DOT_AS_I = re.compile(r"(?<=[A-Za-z])[.,](?=[A-Za-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# A truncated read is scored by CONTAINMENT, not edit distance.
MIN_CONTAINMENT_CHARS = 6
CONTAINMENT_SCORE = 0.92
# Floor for the one-to-one assignment. Lower than FUZZY_MATCH_RATIO on
# purpose: pairing the read as a SET means a weak pair is only accepted when
# nothing better competes for either side, so the hard 0.8 cliff is no longer
# what protects against mismatches.
#
# No threshold separates every case cleanly. Measured on this board's real
# history: a legitimate reword ("Add LOOPS logging to Posthog" -> "Implement
# Loops logging via Posthog") scores 0.730, while two GENUINELY different
# items that share vocabulary ("Work on role play generation" vs "Break down
# Role Play Version 3") score 0.724 — six thousandths apart. Assignment is
# what carries it rather than the number: when both of those items are on the
# board each matches ITSELF at ~1.0 and wins its pairing, so the confusable
# pair only forms when one of them is absent from the read entirely. That
# leaves the item uncreated for one scan and self-corrects on the next; the
# sightings_to_add and mass-erase guards still sit behind it.
ASSIGN_FLOOR = 0.68
# Vertical position is a tie-breaker only. The user reorders items by erasing
# and rewriting them, so distance must never PENALISE a pair.
POSITION_BONUS = 0.08


def norm_for_match(text: str) -> str:
    """Comparison-only normalisation. Never stored, never sent to Todoist."""
    return _NON_ALNUM.sub(" ", _DOT_AS_I.sub("i", text).lower()).strip()


def match_score(a: str, b: str) -> float:
    """How likely two texts are the same board item, 0..1.

    Truncation is scored by containment rather than edit ratio. A read that
    lost its start or end — "Update Flight" for "Update Flight RISK ticket",
    "Remove Ampli+" for "Remove Amplitude tracking", "pe Express" for "Europe
    Express Change Request!" — scores only 0.50-0.68 by SequenceMatcher and
    used to fall off the 0.8 cliff, which completed the real task and re-added
    the fragment as a new one. "Europe Express Change Request!" churned six
    add/complete cycles that way.
    """
    na, nb = norm_for_match(a), norm_for_match(b)
    if not na or not nb:
        return 0.0
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(short) >= MIN_CONTAINMENT_CHARS and short in long_:
        return CONTAINMENT_SCORE
    return difflib.SequenceMatcher(None, na, nb).ratio()


def assign_matches(reads: list[str], known: list[str],
                   read_pos: Optional[dict] = None,
                   known_pos: Optional[dict] = None,
                   floor: float = ASSIGN_FLOOR) -> dict[str, str]:
    """Pair this read's lines one-to-one with the tracked items.

    The old path matched each read line independently against the whole known
    list and took its own best hit above a hard 0.8. Two failure modes came
    out of that: a lightly reworded line ("Add LOOPS logging to Posthog" ->
    "Implement Loops logging via Posthog", ratio 0.73) fell off the cliff and
    became a false completion PLUS a duplicate task; and two read lines could
    both claim the same tracked item.

    Pairing the read against the list as a SET fixes both. The strongest pairs
    are taken first and the remainder match by elimination, so one badly
    drifted line still lands on its real item instead of orphaning it. The
    user both reorders items and lightly rewords them, so text carries the
    decision and position only breaks ties.
    """
    read_pos = read_pos or {}
    known_pos = known_pos or {}
    scored = []
    for r in reads:
        for k in known:
            s = match_score(r, k)
            if s < floor:
                continue
            if r in read_pos and k in known_pos:
                s += POSITION_BONUS / (1.0 + abs(read_pos[r] - known_pos[k]))
            scored.append((s, r, k))
    scored.sort(key=lambda p: -p[0])
    out: dict[str, str] = {}
    used_r: set[str] = set()
    used_k: set[str] = set()
    for _s, r, k in scored:
        if r in used_r or k in used_k:
            continue
        out[r] = k
        used_r.add(r)
        used_k.add(k)
    return out


def fuzzy_match(needle: str, haystack: list[str]) -> Optional[str]:
    """Best fuzzy match for OCR'd text, or None. Handwriting reads vary
    slightly between scans; exact matching would duplicate tasks."""
    best, best_ratio = None, 0.0
    for candidate in haystack:
        ratio = match_score(needle, candidate)
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio
    return best if best_ratio >= FUZZY_MATCH_RATIO else None


def confirm_ticks(seen: dict, seen2: Optional[dict]) -> dict:
    """Tick consensus between two independent reads of two fresh photos.
    A tick survives only if BOTH reads saw it ticked; a tick the second
    read disagrees with (or, when the second read failed entirely,
    ``seen2 is None``, every tick) is demoted to an OPEN sighting — the
    item still counts as present, it just earns no tick evidence this
    scan. Texts stay canonical from the first read; obstruction flags OR.
    Demoted texts are also listed under ``{board}_unsure``: reconcile
    freezes their tick streaks instead of resetting them — a confirm read
    that failed (model timeout, somebody walked in) or disagreed is no
    evidence that the box is empty. Until 2026-10-30 a timed-out confirm
    read wiped a real streak ("LR blindspot alignment") back to zero.

    This exists because tick false positives were observed to REPEAT
    across consecutive reads of a transiently weird scene (2026-07-08,
    ~23:20: an empty box read ticked on two captures minutes apart while
    the board read unobstructed) — a per-scan streak alone cannot stop a
    correlated repeat, but three agreeing misreads are much rarer."""
    merged = dict(seen)
    for board in ("work", "personal"):
        done1 = merged.get(f"{board}_done", [])
        if seen2 is None:
            confirmed, demoted = [], list(done1)
        else:
            done2 = [t.strip() for t in seen2.get(f"{board}_done", []) if t.strip()]
            confirmed = [t for t in done1 if fuzzy_match(t, done2)]
            demoted = [t for t in done1 if t not in confirmed]
        merged[f"{board}_done"] = confirmed
        merged[board] = list(merged.get(board, [])) + demoted
        merged[f"{board}_unsure"] = list(merged.get(f"{board}_unsure", [])) + demoted
        if seen2 is not None:
            merged[f"{board}_obstructed"] = bool(
                merged.get(f"{board}_obstructed")
                or seen2.get(f"{board}_obstructed"))
    return merged


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
        self.state: dict = {"items": []}  # [{text, board, status, missing, ticked}]
        if self._state_path.is_file():
            self.state = json.loads(self._state_path.read_text())
        for idx, item in enumerate(self.state["items"]):  # migrate old state files
            item.setdefault("status", "open")
            item.setdefault("ticked", 0)
            # board position; refreshed from the next processed read's
            # top-to-bottom order — until then, keep today's display order
            item.setdefault("pos", idx)
        # items_view is what HTTP threads serve: a snapshot swapped whole, so
        # a scan mutating self.state mid-serialisation can't tear a response.
        self.items_view: list[dict] = [dict(i) for i in self.state["items"]]
        self.last_result: dict = {}
        self.last_scan_t: float = 0.0
        self.scans_since_change = 0  # see TICK_START_CHANGE_WINDOW
        self.scanning = False
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.loop_beat = time.time()
        self.gate: dict = {}  # last presence-gate skip, for the dashboard
        self._camera_off_noted = False   # night-window skip recorded once
        # Vision-model health, for the dashboard and /healthz: a wedged model
        # once went 44 h unnoticed because the loop itself stayed healthy.
        self.model_health = {"failures": 0, "last_ok_t": None,
                             "last_error": None, "last_error_t": None}
        self.last_read: dict = {}   # the last scan that actually read the boards
        self.counters = {"scans": 0, "changed": 0, "unchanged": 0, "obstructed": 0,
                         "errors": 0, "added": 0, "completed": 0, "skips_home": 0,
                         "skips_off": 0, "skips_person": 0, "skips_dim": 0,
                         "skips_person_late": 0, "skips_camera_off": 0}
        self._presence: Optional[str] = None
        self._presence_t = 0.0
        # Did the last _depth_obstructed() call actually reach the endpoint?
        # [] means "clear" and "couldn't ask" alike, and those must not be
        # confused when deciding whether there is any evidence at all.
        self._depth_ok: Optional[bool] = None
        self._settings_path = cfg.data_dir / "settings.json"
        self.settings = {
            "enabled": True,
            "presence_gate": not cfg.scan_when_home,
            "change_detection": True,
            "obstruction_guard": True,
            "enhance": True,
            "due_today": True,
            "interval_s": cfg.interval_s,
            "change_threshold": cfg.change_threshold,
            "missing_to_complete": cfg.missing_to_complete,
            "ticked_to_complete": cfg.ticked_to_complete,
            "sightings_to_add": 2,
            "tick_confirm": True,
            "low_light_boost": True,
            "depth_guard": True,
            "depth_skip": True,
            "presence_occupancy": PRESENCE_OCCUPANCY,
            "presence_clear_s": 20,
            "presence_recheck": True,
            "presence_required": False,
            "dim_skip": True,
            "dim_luma": DIM_LUMA,
            "capture_frames": 8,
            "capture_quality": 92,
            "upscale": 1,
            "capture_format": "jpeg",
            # WB_REGION is only the default; the dashboard's framing editor
            # overrides it here, and cfg.region follows the live value.
            "region": list(cfg.region),
            "split": 0,
            "items_bottom": 0,
            "presence_overrides_model": True,
        }
        self.default_region = list(cfg.region)   # WB_REGION, for "reset"
        if self._settings_path.is_file():
            try:
                saved = json.loads(self._settings_path.read_text())
                clean, bad = self.validate_settings(
                    {k: v for k, v in saved.items() if k in self.settings})
                for key, why in bad.items():
                    log.warning("settings.json: ignoring %s (%s)", key, why)
                self.settings.update(clean)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("settings.json unreadable (%s), using defaults", exc)
        cfg.region = tuple(self.settings["region"])
        # Watches, continuously, whether a human is in front of either board.
        # Started by main(); constructed here so it shares the live settings
        # dict (threshold changes from the dashboard apply on the next poll).
        self.watcher = PresenceWatcher(cfg, self.settings)
        self.crop_t: Optional[float] = (
            self.crop_path.stat().st_mtime if self.crop_path.is_file() else None)
        self.history: list[dict] = []
        if self._history_path.is_file():
            for line in self._history_path.read_text().splitlines()[-HISTORY_KEEP:]:
                try:
                    self.history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            self._compact_history()
            for e in reversed(self.history):
                if e.get("type") == "scan" and e.get("changed"):
                    self.last_read = e
                    break
        self._appends = 0

    # -- settings ---------------------------------------------------------
    @staticmethod
    def validate_settings(patch: dict) -> tuple[dict, dict]:
        """Type-check and bound a settings patch WITHOUT applying it.
        Returns ({key: clean value}, {key: reason rejected})."""
        clean, rejected = {}, {}
        for key, val in patch.items():
            spec = SETTING_BOUNDS.get(key)
            if spec is None:
                rejected[key] = "unknown setting"
                continue
            try:
                if spec == "region":
                    if isinstance(val, str):
                        val = val.split(",")
                    if not isinstance(val, (list, tuple)) or len(val) != 4:
                        raise ValueError("want [x1, y1, x2, y2]")
                    x1, y1, x2, y2 = (int(round(float(v))) for v in val)
                    if not (0 <= x1 < x2 <= FRAME_W and 0 <= y1 < y2 <= FRAME_H):
                        raise ValueError(f"must lie inside the {FRAME_W}x{FRAME_H} frame")
                    if x2 - x1 < MIN_REGION_W or y2 - y1 < MIN_REGION_H:
                        raise ValueError(f"smaller than {MIN_REGION_W}x{MIN_REGION_H}")
                    new = [x1, y1, x2, y2]
                elif spec == "split":
                    new = int(round(float(val)))
                    if not 0 <= new <= FRAME_W:
                        raise ValueError(f"out of range 0..{FRAME_W}")
                elif spec == "items_bottom":
                    new = int(round(float(val)))
                    if not 0 <= new <= FRAME_H:
                        raise ValueError(f"out of range 0..{FRAME_H}")
                elif spec is bool:
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
            clean[key] = new
        return clean, rejected

    def update_settings(self, patch: dict) -> dict:
        """Validate + apply a runtime settings patch from the dashboard.
        Returns {'applied': {...}, 'rejected': {key: reason}}."""
        clean, rejected = self.validate_settings(patch)
        applied = {k: v for k, v in clean.items() if self.settings[k] != v}
        if applied:
            self.settings.update(applied)
            if "region" in applied:
                self.cfg.region = tuple(applied["region"])
                # The change-detection baseline is the OLD framing; diffing a
                # new one against it is meaningless (and a same-size pan
                # would not even trip the shape check). Start over: the next
                # scan reads the board fresh and becomes the new baseline.
                self._baseline_path.unlink(missing_ok=True)
            write_json_atomic(self._settings_path, self.settings)
            self._record({"type": "settings", "changed": applied})
            log.info("settings changed: %s", json.dumps(applied))
        return {"applied": applied, "rejected": rejected}

    # -- manual item edits (dashboard) --------------------------------------
    def _save_items(self) -> None:
        write_json_atomic(self._state_path, self.state)
        self.items_view = [dict(i) for i in self.state["items"]]

    def _find_item(self, board: str, text: str) -> Optional[dict]:
        for item in self.state["items"]:
            if item["board"] == board and item["text"] == text:
                return item
        return None

    def edit_item(self, board: str, text: str, new_text: str) -> tuple[int, dict]:
        """Rename a tracked item from the dashboard. For open items the
        Todoist task is renamed too (state text is canonical — it must stay
        byte-identical to the task summary or completion can't find it);
        if that HA call fails nothing changes locally. Returns (http status,
        body)."""
        new_text = (new_text or "").strip()
        if not new_text:
            return 400, {"error": "new text is empty"}
        # Scans hold the lock for their whole run (the vision call can take
        # minutes) — don't hang the dashboard waiting, tell the user instead.
        if not self._lock.acquire(timeout=EDIT_LOCK_TIMEOUT_S):
            return 409, {"error": "a scan is running — try again in a moment"}
        try:
            item = self._find_item(board, text)
            if item is None:
                return 404, {"error": "item not found — state changed, refreshing"}
            if new_text == text:
                return 200, {"ok": True, "item": dict(item)}
            if self._find_item(board, new_text) is not None:
                return 400, {"error": "another item on this board already has that text"}
            if item["status"] == "open":
                if not self.ha.call("todo", "update_item", {
                        "entity_id": self.cfg.todo_entity,
                        "item": text, "rename": new_text}):
                    return 502, {"error": "Todoist rename via HA failed — item unchanged"}
            item["text"] = new_text
            self._save_items()
            self._record({"type": "manual", "action": "edit", "board": board,
                          "from": text, "to": new_text})
            log.info("manual edit [%s] %r -> %r", board, text, new_text)
            body = {"ok": True, "item": dict(item)}
            if item["status"] == "open" and difflib.SequenceMatcher(
                    None, text.lower(), new_text.lower()).ratio() < FUZZY_MATCH_RATIO:
                body["warning"] = ("new text no longer resembles the board writing — "
                                   "the next scan may re-add what's written and count "
                                   "this item missing")
            return 200, body
        finally:
            self._lock.release()

    def remove_item(self, board: str, text: str) -> tuple[int, dict]:
        """Remove a tracked item from the dashboard: delete its Todoist task
        (open items only — done/removed ones have no live task) and stop
        treating it as a to-do. The item is NOT dropped from state while its
        writing may still be on the board — a dropped item would come back
        as a brand-new task on the next processed read. Status "removed"
        behaves like "done": ignored by reconciliation, purged after it has
        been erased for ``missing_to_complete`` scans."""
        if not self._lock.acquire(timeout=EDIT_LOCK_TIMEOUT_S):
            return 409, {"error": "a scan is running — try again in a moment"}
        try:
            item = self._find_item(board, text)
            if item is None:
                return 404, {"error": "item not found — state changed, refreshing"}
            if item["status"] == "open":
                if not self.ha.call("todo", "remove_item", {
                        "entity_id": self.cfg.todo_entity, "item": text}):
                    return 502, {"error": "Todoist delete via HA failed — item kept"}
            item["status"] = "removed"
            item["missing"] = 0
            item["ticked"] = 0
            self._save_items()
            self._record({"type": "manual", "action": "remove", "board": board,
                          "text": text})
            log.info("manual remove [%s] %r", board, text)
            return 200, {"ok": True, "item": dict(item)}
        finally:
            self._lock.release()

    def reopen_item(self, board: str, text: str) -> tuple[int, dict]:
        """Mark a done/removed item as NOT done from the dashboard, by
        creating a FRESH Todoist task with the canonical text and flipping
        the item back to "open". A completed task can't be un-completed
        through HA's todo entity (Todoist only lists active tasks there)
        and a removed item's task was deleted — a new task is the reliable
        path either way; it also does the right thing for tick-misread
        "done" ghosts that never had a task at all. NB: if the checkbox is
        still marked on the physical board, the next clear scans will
        simply complete the item again — the mark must be wiped too (the
        dashboard's confirm dialog says so)."""
        if not self._lock.acquire(timeout=EDIT_LOCK_TIMEOUT_S):
            return 409, {"error": "a scan is running — try again in a moment"}
        try:
            item = self._find_item(board, text)
            if item is None:
                return 404, {"error": "item not found — state changed, refreshing"}
            if item["status"] == "open":
                return 200, {"ok": True, "item": dict(item)}  # nothing to do
            if item["status"] == "pending":
                return 400, {"error": "item is still unconfirmed — it has no "
                                      "Todoist task to reopen"}
            if not self._push_task(text, board):
                return 502, {"error": "Todoist task creation via HA failed — "
                                      "item unchanged"}
            item["status"] = "open"
            item["missing"] = 0
            item["ticked"] = 0
            self._save_items()
            self._record({"type": "manual", "action": "reopen", "board": board,
                          "text": text})
            log.info("manual reopen [%s] %r", board, text)
            return 200, {"ok": True, "item": dict(item)}
        finally:
            self._lock.release()

    # -- imaging --------------------------------------------------------
    def _grab_frame(self, frames: int) -> Image.Image:
        """One full 1920x1080 photo from kinect-knob, ``frames`` stacked."""
        # Ask kinect-knob for a stacked, losslessly-encoded photo; an older
        # kinect-knob simply ignores unknown params and returns its JPEG.
        params = {"format": "png"}
        if frames > 1:
            params["frames"] = frames
        r = requests.get(f"{self.cfg.kinect_url}/api/snapshot", params=params,
                         timeout=15 + 2 * frames)
        off = camera_off_detail(r)
        if off is not None:
            raise CameraOff(off)
        r.raise_for_status()
        ctype = getattr(r, "headers", {}).get("Content-Type", "image/")
        if not ctype.startswith("image/"):
            raise requests.RequestException(
                f"snapshot returned {ctype}, not an image")
        try:
            return Image.open(io.BytesIO(r.content))
        except OSError as exc:  # truncated / corrupt image
            raise requests.RequestException(f"unreadable snapshot: {exc}") from exc

    @staticmethod
    def _prepare(img: Image.Image, region, s: dict) -> tuple[bytes, np.ndarray, Image.Image]:
        """Frame -> (bytes for the model, raw grey crop, the model's image).
        Shared by scans and the framing preview, so a preview is exactly
        what a scan with those settings would send."""
        crop = img.crop(tuple(region)).convert("RGB")
        # The baseline compares the RAW crop — toggling enhance/upscale must
        # not read as "the board changed".
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
        if s["enhance"]:
            crop = ImageOps.autocontrast(crop, cutoff=1)
            crop = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
        scale = model_scale(crop.width, crop.height, s["upscale"])
        if abs(scale - 1.0) > 0.01:
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)),
                               Image.LANCZOS)
        buf = io.BytesIO()
        if s["capture_format"] == "png":
            crop.save(buf, format="PNG")
        else:
            crop.save(buf, format="JPEG", quality=s["capture_quality"])
        return buf.getvalue(), gray, crop

    def _fetch_crop(self, frames: Optional[int] = None) -> tuple[bytes, np.ndarray]:
        s = self.settings
        img = self._grab_frame(frames or s["capture_frames"])
        payload, gray, crop = self._prepare(img, self.cfg.region, s)
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

    def live_frame(self, frames: int = 1) -> tuple[bytes, str]:
        """The whole uncropped camera frame, for the framing editor."""
        params = {"frames": frames} if frames > 1 else {}
        r = requests.get(f"{self.cfg.kinect_url}/api/snapshot", params=params,
                         timeout=15 + 2 * frames)
        off = camera_off_detail(r)
        if off is not None:
            raise CameraOff(off)
        r.raise_for_status()
        return r.content, r.headers.get("Content-Type", "image/jpeg")

    def framing_preview(self, body: dict) -> tuple[int, dict]:
        """What the model WOULD see under a draft framing (region, split and
        image settings layered over the live ones) — and, with read=true,
        what it reads off it. Touches nothing: no state, no baseline, no
        dashboard crop, no Todoist. This is how a framing gets judged before
        it is saved."""
        draft = {k: v for k, v in body.items() if k in FRAMING_KEYS}
        clean, rejected = self.validate_settings(draft)
        if rejected:
            return 400, {"error": "bad framing", "rejected": rejected}
        s = {**self.settings, **clean}
        t0 = time.monotonic()
        try:
            img = self._grab_frame(s["capture_frames"])
        except requests.RequestException as exc:
            return 502, {"error": f"snapshot failed: {exc}"}
        payload, gray, crop = self._prepare(img, s["region"], s)
        mime = "image/png" if s["capture_format"] == "png" else "image/jpeg"
        x1, y1, x2, y2 = s["region"]
        out = {
            "image": f"data:{mime};base64," + base64.standard_b64encode(payload).decode(),
            "region": s["region"],
            "split": board_split(tuple(s["region"]), s["split"]),
            "crop": [x2 - x1, y2 - y1],
            "model": [crop.width, crop.height],
            "bytes": len(payload),
            "luma": round(float(gray.mean())),
            # upscale asked for more than MODEL_MAX_PIXELS allows
            "capped": model_scale(x2 - x1, y2 - y1, s["upscale"]) < s["upscale"] - 0.01,
            "capture_s": round(time.monotonic() - t0, 1),
        }
        if body.get("read"):
            t1 = time.monotonic()
            try:
                out["read"] = self.reader.read(payload)
            except Exception as exc:  # noqa: BLE001 — report it, don't 500
                out["read_error"] = str(exc)
            out["read_s"] = round(time.monotonic() - t1, 1)
        return 200, out

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
    def _push_task(self, text: str, board: str) -> bool:
        task = {
            "content": text,
            "project": self.cfg.todoist_project,
            "labels": [board],  # just "work" / "personal" (user preference)
        }
        if self.settings["due_today"]:
            # Todoist parses this in the ACCOUNT's timezone — safer
            # than a date computed on this container's (UTC) clock.
            task["due_date_string"] = "today"
        return self.ha.call("todoist", "new_task", task)

    def _complete_in_todoist(self, item: dict) -> str:
        """"ok" | "orphaned" | "failed".

        "orphaned": the call failed and the task is not open in Todoist at all
        — completed, renamed or deleted there directly. Retrying can never
        succeed, so the caller settles the item locally. Before this existed
        three such items failed on EVERY scan for a week (2026-10): each
        failure also blocked the change-detection baseline (so every scan was
        "changed") and kept a tick-confirm photo + second model read firing."""
        if self.ha.call("todo", "update_item", {
                "entity_id": self.cfg.todo_entity,
                "item": item["text"],
                "status": "completed"}):
            return "ok"
        lookup = getattr(self.ha, "open_items", None)
        open_now = lookup(self.cfg.todo_entity) if lookup else None
        if open_now is not None and item["text"] not in open_now:
            log.warning("[%s] %r has no open Todoist task (completed or "
                        "deleted there) — settling it locally", item["board"],
                        item["text"])
            return "orphaned"
        return "failed"

    def reconcile(self, seen: dict, apply: bool = True,
                  tick_start_ok: bool = True,
                  force_protect: tuple = ()) -> dict:
        """Diff OCR results against known state. With ``apply`` (the default),
        push adds/completions to HA and persist; with ``apply=False`` (dry
        run) just report what WOULD happen, touching nothing.
        ``tick_start_ok=False`` (the scene has been static for more than
        ``TICK_START_CHANGE_WINDOW`` scans) blocks NEW tick streaks from
        starting: a real new tick necessarily changed pixels recently, so a
        tick first sighted on a long-static scene is a misread — observed
        live 2026-07-09 00:39, when three near-identical static frames
        agreed on the same false tick and defeated both the streak and the
        confirm photo. Streaks already under way still accrue and complete.
        Returns {'added': [...], 'completed': [...]} of item texts plus
        'pending' (new texts still waiting for confirming sightings), a
        'failed' count of HA calls that didn't land (they retry next scan),
        the 'protected' boards, and 'slowed' boards where the mass-erase
        guard deferred completions.

        Addition — a brand-new text does NOT immediately become a Todoist
        task: it is tracked as status "pending" and the task is only created
        after ``sightings_to_add`` consecutive processed reads (default 2).
        Hallucinated items and fragments of half-hidden lines almost never
        repeat identically between reads; real items always do. A single
        miss on a clear board drops a pending item, and one first seen
        ticked converts straight to "done" without ever creating a task.

        On a PROTECTED board (flagged obstructed while the guard is on) the
        checkbox judgment itself is untrusted — observed live 2026-07-08:
        with a person in front of a board, empty boxes read "marked" and
        marked boxes read "empty", varying frame to frame, while clear-board
        reads were stable. Item TEXT is still trusted there (it resets
        absence counters and accrues pending sightings), but tick streaks
        freeze in both directions and no tick may complete a task.

        Completion signals, strongest first:

        * A TICKED checkbox (the board's ``*_done`` list) on a CLEAR board:
          the task completes after ``ticked_to_complete`` consecutive
          sightings (default 2 — a single-frame misread must not complete a
          real task). The item then stays tracked with status "done" for as
          long as it remains on the board, so a later misread of its tick
          can never re-create the task.
        * ABSENCE (in neither list) means erased — but only on a clear
          board. After ``missing_to_complete`` consecutive misses the task
          completes and the item is dropped. On a PROTECTED board (flagged
          obstructed while the guard is on) absence proves nothing: no miss
          is counted, though visible new items still accrue sightings.
          Mass-erase guard: when a single read would complete
          ``MASS_ERASE_MIN`` or more items by absence while the board still
          shows OTHER items (the signature of a garbage read — a real wipe
          reads as an empty board), each completion needs one extra miss.
        * Items first seen already ticked never create Todoist tasks; done
          and manually-removed items are purged (no HA call) once they've
          been erased for ``missing_to_complete`` scans."""
        guard = self.settings["obstruction_guard"]
        # ``force_protect`` boards are protected regardless of the
        # obstruction guard toggle — used for dim frames (both boards) and
        # for depth-detected physical obstructions: text sightings still
        # keep items alive, but nothing may complete off those reads.
        protected = [b for b in ("work", "personal")
                     if b in force_protect
                     or (guard and seen.get(f"{b}_obstructed"))]
        need_seen = self.settings["sightings_to_add"]
        base_miss = self.settings["missing_to_complete"]
        added, completed, pending, slowed, failed = [], [], [], [], 0
        parked, orphaned = [], []
        for board in ("work", "personal"):
            seen_open = [t.strip() for t in seen.get(board, []) if t.strip()]
            seen_done = [t.strip() for t in seen.get(f"{board}_done", []) if t.strip()]
            known = [i for i in self.state["items"] if i["board"] == board]
            known_texts = [i["text"] for i in known]

            # top-to-bottom position of each seen text on the board this
            # read; tracked items get restamped every time they're sighted
            # so the dashboard mirrors the physical board's order
            order_idx: dict[str, int] = {}
            for i, t in enumerate(seen.get(f"{board}_order",
                                           seen_open + seen_done)):
                order_idx.setdefault(t.strip(), i)
            pos_seen: dict[str, int] = {}  # canonical text -> board position

            # Pair the WHOLE read against the WHOLE tracked list in one go,
            # before deciding anything. Matching line-by-line let a reworded
            # line orphan its item (completion + duplicate) and let two lines
            # claim the same item; assignment resolves both globally.
            known_pos = {i["text"]: i.get("pos", 0) for i in known}
            assigned = assign_matches(seen_open + seen_done, known_texts,
                                      order_idx, known_pos)
            taken = set(assigned.values())

            def _match_for(text: str) -> Optional[str]:
                """Assigned partner, else a fallback among items still free.

                The fallback keeps two existing behaviours: items appended to
                ``known_texts`` DURING this scan (new pendings) are matchable,
                and nothing already claimed can be claimed twice.
                """
                hit = assigned.get(text)
                if hit is None:
                    hit = fuzzy_match(text, [t for t in known_texts
                                             if t not in taken])
                if hit is not None:
                    taken.add(hit)
                return hit

            open_matched, done_matched, unsure_matched = set(), set(), set()
            unsure = {t.strip() for t in seen.get(f"{board}_unsure", [])}
            for text in seen_open:
                match = _match_for(text)
                if match is not None:
                    open_matched.add(match)  # keep canonical text (= Todoist summary)
                    if text in unsure:
                        unsure_matched.add(match)
                    pos_seen[match] = order_idx.get(text, len(order_idx))
                    continue
                if need_seen > 1:
                    pending.append(text)
                    if apply:
                        self.state["items"].append(
                            {"text": text, "board": board, "status": "pending",
                             "missing": 0, "ticked": 0, "seen": 1,
                             "pos": order_idx.get(text, len(order_idx))})
                        known_texts.append(text)  # no same-scan duplicate pendings
                    continue
                if not apply:
                    added.append(text)
                    continue
                if self._push_task(text, board):
                    self.state["items"].append({"text": text, "board": board,
                                                "status": "open", "missing": 0,
                                                "ticked": 0,
                                                "pos": order_idx.get(text, len(order_idx))})
                    known_texts.append(text)
                    added.append(text)
                else:
                    failed += 1
            for text in seen_done:
                match = _match_for(text)
                if match is not None:
                    done_matched.add(match)
                    pos_seen[match] = order_idx.get(text, len(order_idx))
                elif apply and board not in protected:
                    # First seen already ticked (on a TRUSTED read): it was
                    # finished before we ever tracked it — no Todoist task,
                    # but remember it so a misread of its tick can't create
                    # one later. Never from a protected board: a false
                    # "marked" there would park a real new item as done and
                    # its task would never be created. And never off ONE read
                    # when ticks need a streak: it starts as a pending item
                    # one tick in, and parks only if the tick holds (see the
                    # pending branch below); read open instead, it becomes a
                    # normal task.
                    pos = order_idx.get(text, len(order_idx))
                    if self.settings["ticked_to_complete"] <= 1 and tick_start_ok:
                        self.state["items"].append({"text": text, "board": board,
                                                    "status": "done", "missing": 0,
                                                    "ticked": 0, "pos": pos})
                        parked.append(text)
                        log.info("parked [%s] %r as done: first seen ticked",
                                 board, text)
                    else:
                        self.state["items"].append({"text": text, "board": board,
                                                    "status": "pending", "missing": 0,
                                                    "ticked": 1 if tick_start_ok else 0,
                                                    "seen": 1, "pos": pos})
                        pending.append(text)
                    known_texts.append(text)
            if apply:
                for item in known:
                    if item["text"] in pos_seen:
                        item["pos"] = pos_seen[item["text"]]

            # Mass-erase guard (see docstring): completing many items by
            # absence in one read, while the read still shows other items,
            # looks like a hallucinated/partial read — not like an erased
            # board. Slow those completions down by one extra miss.
            miss_needed = base_miss
            if board not in protected and (seen_open or seen_done):
                doomed = [i for i in known
                          if i["status"] == "open"
                          and i["text"] not in open_matched
                          and i["text"] not in done_matched
                          and i["missing"] + 1 >= base_miss]
                if len(doomed) >= MASS_ERASE_MIN:
                    miss_needed = base_miss + 1
                    slowed.append(board)

            for item in known:
                sighted = item["text"] in open_matched or item["text"] in done_matched
                # A tick only counts as a tick on a clear board; on a
                # protected board it degrades to a plain text sighting.
                ticked = item["text"] in done_matched and board not in protected

                if item["status"] == "pending":
                    # No Todoist task exists yet. Sightings must be
                    # CONSECUTIVE: one miss on a clear board kills the item
                    # — that strictness is the whole anti-hallucination
                    # point, and a real item just starts accruing again.
                    #
                    # A trusted tick here is a STREAK, exactly as on an open
                    # item: ``ticked_to_complete`` consecutive ticks (plus the
                    # tick-confirm photo) park it as done without a task —
                    # finished before we ever believed in it. Until 2026-10
                    # ONE tick read parked it on the spot, silently: a single
                    # misread box made a real new item vanish with no task and
                    # no trace ("Review UK Retreat Photos"). A static-scene
                    # tick cannot START a streak (see tick_start_ok).
                    need_tick = self.settings["ticked_to_complete"]
                    streak = item.get("ticked", 0)
                    tick_counts = ticked and (tick_start_ok or streak > 0)
                    if not apply:
                        if tick_counts and streak + 1 >= need_tick:
                            parked.append(item["text"])
                        elif sighted and not tick_counts and item["seen"] + 1 >= need_seen:
                            added.append(item["text"])
                        elif sighted:
                            pending.append(item["text"])
                        continue
                    if tick_counts:
                        item["ticked"] = streak + 1
                        item["missing"] = 0
                        if item["ticked"] >= need_tick:
                            item.pop("seen", None)
                            item["status"] = "done"
                            item["ticked"] = 0
                            parked.append(item["text"])
                            log.info("parked [%s] %r as done without a task: "
                                     "ticked on %d consecutive reads before it "
                                     "was ever confirmed", board, item["text"],
                                     need_tick)
                        else:
                            pending.append(item["text"])
                    elif sighted:
                        if (board not in protected and not ticked
                                and item["text"] not in unsure_matched):
                            item["ticked"] = 0   # read open: the streak was a misread
                        item["seen"] += 1
                        if item["seen"] >= need_seen:
                            if self._push_task(item["text"], board):
                                item.pop("seen", None)
                                item["status"] = "open"
                                added.append(item["text"])
                            else:
                                failed += 1  # stays pending; retries next scan
                        else:
                            pending.append(item["text"])
                    elif board not in protected:
                        self.state["items"].remove(item)
                    continue

                if item["status"] != "open":
                    # "done": task already completed. "removed": task deleted
                    # from the dashboard. Either way the item just hasn't
                    # been erased yet. Purge once it actually leaves the board.
                    if not apply:
                        continue
                    if sighted:
                        item["missing"] = 0
                    elif board not in protected:
                        item["missing"] += 1
                        if item["missing"] >= base_miss:
                            self.state["items"].remove(item)
                    continue

                if ticked:  # trusted tick: clear board only
                    if item["ticked"] == 0 and not tick_start_ok:
                        # long-static scene: a "new" tick is a misread (see
                        # docstring) — count the text sighting, nothing else
                        if apply:
                            item["missing"] = 0
                        continue
                    if not apply:
                        if item["ticked"] + 1 >= self.settings["ticked_to_complete"]:
                            completed.append(item["text"])
                        continue
                    item["missing"] = 0
                    item["ticked"] += 1
                    if item["ticked"] >= self.settings["ticked_to_complete"]:
                        outcome = self._complete_in_todoist(item)
                        if outcome == "failed":
                            failed += 1
                        else:
                            item["status"] = "done"
                            item["ticked"] = 0
                            (completed if outcome == "ok" else orphaned).append(item["text"])
                    continue
                if sighted:
                    if apply:
                        item["missing"] = 0
                        if board not in protected and item["text"] not in unsure_matched:
                            # Read as open on a trusted frame: any tick
                            # streak was a misread — reset it. On a protected
                            # frame, or when the confirm read failed or
                            # disagreed, the streak freezes instead (that
                            # judgment proves nothing either way).
                            item["ticked"] = 0
                    continue
                # absent from both lists: erased, or hidden behind something
                if board in protected:
                    continue  # hidden vs erased is unknowable — wait
                if not apply:
                    if item["missing"] + 1 >= miss_needed:
                        completed.append(item["text"])
                    continue
                item["missing"] += 1
                item["ticked"] = 0
                if item["missing"] >= miss_needed:
                    outcome = self._complete_in_todoist(item)
                    if outcome == "failed":
                        failed += 1
                    else:
                        self.state["items"].remove(item)
                        (completed if outcome == "ok" else orphaned).append(item["text"])
        if apply:
            self._save_items()
        out = {"added": added, "completed": completed, "pending": pending,
               "failed": failed, "protected": protected, "slowed": slowed}
        if parked:
            out["parked"] = parked
        if orphaned:
            out["orphaned"] = orphaned
        return out

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

    def _depth_obstructed(self) -> list[str]:
        """Boards with something PHYSICALLY in front of them, per
        kinect-knob's aligned-depth stats (self-illuminated ToF: works day
        and pitch-black night, and unlike the model's visual judgment it
        can't be fooled by lighting — the visual flag was observed flapping
        at night on a clear board). Best-effort: [] when the endpoint is
        unavailable (older kinect-knob) or the data is degenerate."""
        x1, y1, x2, y2 = items_zone(self.cfg.region,
                                    self.settings.get("items_bottom", 0))
        mid = board_split(self.cfg.region, self.settings.get("split", 0))
        blocked = []
        for board, (rx1, rx2) in (("work", (x1, mid)), ("personal", (mid, x2))):
            try:
                r = requests.get(f"{self.cfg.kinect_url}/api/region_depth",
                                 params={"x1": rx1, "y1": y1,
                                         "x2": rx2, "y2": y2}, timeout=5)
                r.raise_for_status()
                stats = r.json()
            except (requests.RequestException, ValueError) as exc:
                log.debug("region_depth unavailable: %s", exc)
                self._depth_ok = False
                return []
            if depth_blocked(stats):
                blocked.append(board)
        self._depth_ok = True
        return blocked

    def _physically_blocked(self, since_t: float) -> list[str]:
        """Boards something physically covered while the photo was taken —
        the reconcile-time depth guard.

        Prefers the presence watcher (occupancy against the LEARNED empty
        board, clipped to the items zone) over the per-frame percentile test.
        The percentile test cannot tell furniture from a person: from
        2026-10-26 the desk chair in front of the personal board's blank lower
        half tripped it on every one of 57 reads, so every personal tick was
        thrown away for four days while presence (which had learned the chair
        as background) read the board clear."""
        w = self.watcher
        boards = ("work", "personal")
        if w.supported and all(w.status(b) != "unknown" for b in boards):
            return sorted({b for b in boards if w.status(b) == "blocked"}
                          | set(w.blocked_since(since_t)))
        return self._depth_obstructed()

    def _depth_vouched(self, since_t: float) -> list[str]:
        """Boards whose writing depth positively saw clear for the whole
        photo — the evidence that overrules the model's "obstructed" flag.

        The model sets that flag whenever a person is anywhere in shot, even
        in front of blank board space well below the writing (live
        2026-10-30: a frame with every tick read correctly came back with
        both boards "obstructed", which threw every tick away). Depth is the
        reliable signal (CLAUDE.md); the flag only rules where depth has no
        fresh opinion."""
        w = self.watcher
        if not self.settings["presence_overrides_model"] or not w.supported:
            return []
        late = set(w.blocked_since(since_t))
        return [b for b in ("work", "personal")
                if w.status(b) == "clear" and b not in late]

    @staticmethod
    def _overrule(seen: dict, vouched: list[str]) -> list[str]:
        """Clear the model's obstruction flag on ``vouched`` boards; returns
        the boards it actually overruled."""
        out = [b for b in vouched if seen.get(f"{b}_obstructed")]
        for b in out:
            seen[f"{b}_obstructed"] = False
        return out

    def _presence_blocked(self) -> tuple[list[str], str]:
        """Boards a human is at — the question, and the evidence for it.

        Returns ``(boards, source)`` where source is "presence" (kinect-knob's
        region occupancy, the sensitive signal), "depth" (the older
        percentile guard) or "none" (no evidence at all).

        Occupancy is preferred because the percentile guard cannot see a head.
        It compares p10 against p90 within a single frame, so it needs an
        object covering roughly a tenth of a board half before it fires;
        somebody leaning in front of two or three lines covers a few percent.
        Those scans read the visible fragments of the hidden lines, the
        fragments accrued sightings, and the sightings became real Todoist
        tasks — the failure this method exists to end.
        """
        if self.watcher.supported is None:
            # First scan after boot: don't go in blind, pay one round-trip.
            try:
                self.watcher.poll()
            except Exception:  # noqa: BLE001
                log.debug("initial presence poll failed", exc_info=True)
        if self.watcher.supported:
            blocked = self.watcher.blocked_boards(self.settings["presence_clear_s"])
            if blocked:
                return blocked, "presence"
            if all(self.watcher.status(b) == "clear" for b in ("work", "personal")):
                return [], "presence"
            # A board with no fresh reading (camera restarting, endpoint
            # erroring) is NOT clear; fall through to the weaker evidence.
        # No occupancy signal (older kinect-knob, no depth backend, endpoint
        # unreachable): the percentile guard is weaker but far better than
        # nothing — it still catches a whole body at a board.
        blocked = self._depth_obstructed()
        if blocked or self._depth_ok:
            return blocked, "depth"
        return [], "none"

    def _tick_completion_imminent(self, seen: dict,
                                  force_protect: tuple = ()) -> bool:
        """Would reconciling ``seen`` complete a task via its ticks?"""
        guard = self.settings["obstruction_guard"]
        need = self.settings["ticked_to_complete"]
        for board in ("work", "personal"):
            if board in force_protect or (guard and seen.get(f"{board}_obstructed")):
                continue  # protected: ticks are ignored there anyway
            done = [t.strip() for t in seen.get(f"{board}_done", []) if t.strip()]
            if not done:
                continue
            for item in self.state["items"]:
                if (item["board"] == board and item["status"] in ("open", "pending")
                        and item.get("ticked", 0) + 1 >= need
                        and fuzzy_match(item["text"], done)):
                    return True
        return False

    def _wipe_imminent(self, seen: dict, force_protect: tuple = ()) -> list[str]:
        """Boards this read shows EMPTY while absence would complete
        ``MASS_ERASE_MIN``+ tracked tasks. A real wipe reads empty and
        completes at the normal threshold (by design); the second photo only
        rules out the read itself being the empty thing — a truncated model
        reply or a frame from a camera that was coming back up."""
        out = []
        base_miss = self.settings["missing_to_complete"]
        for board in ("work", "personal"):
            if board in force_protect or seen.get(board) or seen.get(f"{board}_done"):
                continue
            doomed = [i for i in self.state["items"]
                      if i["board"] == board and i["status"] == "open"
                      and i["missing"] + 1 >= base_miss]
            if len(doomed) >= MASS_ERASE_MIN:
                out.append(board)
        return out

    def _scan_inner(self, force: bool, dry: bool) -> dict:
        s = self.settings
        if not force and s["depth_skip"]:
            # Someone physically at a board, established BEFORE any photo is
            # taken: their body hides parts of lines and the model turns the
            # visible fragments into pending sightings that become real
            # Todoist tasks. Reading through an obstruction has no upside on
            # this cadence — skip and catch up when they step away. Force
            # (the dashboard's Scan-now button) still scans, with the blocked
            # boards protected by the reconcile-time depth guard as before.
            present, via = self._presence_blocked()
            if present:
                return {"ok": True, "changed": False, "person": present,
                        "person_via": via}
            if via == "none" and s["presence_required"]:
                # No evidence either way. Default is to read anyway (the old
                # behaviour, with every downstream guard still in force); set
                # presence_required when a silent camera should mean "wait".
                return {"ok": True, "changed": False, "person_unknown": True}
        capture_t0 = time.time()
        try:
            jpeg, gray = self._fetch_crop()
        except CameraOff as exc:
            # kinect-knob's scheduled night off: not an error, nothing seen.
            return {"ok": True, "changed": False, "camera_off": str(exc)}
        except requests.RequestException as exc:
            return {"ok": False, "error": f"snapshot failed: {exc}"}
        luma = float(gray.mean())
        base = {"luma": round(luma)}
        if luma < DARK_LUMA:
            # Unreadable. NEVER reconcile a black frame: it reads as an
            # empty board, which absence logic would complete as a wipe.
            return {"ok": True, "changed": False, "dark": True, **base}
        dim = luma < s["dim_luma"]
        if dim and not force and s["dim_skip"]:
            # Dim reads are false-positive factories: even the 32-frame
            # stack leaves garbled fragments that accrue sightings and land
            # in Todoist as partial text (protection only deferred
            # COMPLETIONS — adds still went through). Nobody writes on the
            # boards in the dark; skip and let the first lit scan catch up.
            # A forced scan still reads via the boosted, protected path.
            return {"ok": True, "changed": False, "dim_skipped": True, **base}
        if dim and s["low_light_boost"] and s["capture_frames"] < LOWLIGHT_FRAMES:
            # Synthetic long exposure: the Kinect has no exposure control,
            # but averaging 32 frames buys the same noise reduction. At
            # night this is the difference between garbled text with almost
            # every box read ticked and a near-perfect transcription.
            try:
                jpeg, gray = self._fetch_crop(frames=LOWLIGHT_FRAMES)
                luma = float(gray.mean())
                dim = luma < s["dim_luma"]
                base["luma"] = round(luma)
                if luma < DARK_LUMA:  # lights went out mid-stack
                    return {"ok": True, "changed": False, "dark": True, **base}
            except requests.RequestException as exc:
                log.warning("low-light re-capture failed, using first: %s", exc)
        diff = self._diff_score(gray)
        base["diff"] = None if diff is None else round(diff, 2)
        if diff is None or diff >= s["change_threshold"]:
            self.scans_since_change = 0
        else:
            self.scans_since_change += 1
        # Pending items confirm through consecutive PROCESSED reads, but the
        # board doesn't change while they wait — skipping unchanged frames
        # would strand them forever. Keep reading until none are pending.
        has_pending = any(i["status"] == "pending" for i in self.state["items"])
        if (not force and s["change_detection"] and not has_pending
                and diff is not None and diff < s["change_threshold"]):
            return {"ok": True, "changed": False, **base}
        if not force and s["depth_skip"] and s["presence_recheck"]:
            # The pre-flight check could only say "clear a moment ago", and a
            # scan is not instantaneous: fetching a stacked photo takes
            # seconds, and a 32-frame low-light stack takes longer. Somebody
            # who sat down WHILE the shutter was open produced exactly the
            # half-occluded frame the pre-flight check was meant to prevent.
            # Ask again about the window the photo actually spans — and do it
            # before the model read, so a ruined frame costs nothing.
            late = self.watcher.blocked_since(capture_t0)
            if late:
                return {"ok": True, "changed": False, "person": late,
                        "person_late": True, **base}
            if self.watcher.supported and any(
                    self.watcher.status(b) == "unknown" for b in ("work", "personal")):
                # The watcher lost its view while the shutter was open — most
                # likely the camera restarting under the capture. Nobody can
                # vouch for this frame; the next interval costs nothing.
                return {"ok": True, "changed": False, "person_unknown": True, **base}
        try:
            seen = self.reader.read(jpeg)
        except Exception as exc:  # noqa: BLE001 — one bad read must not kill the loop
            log.warning("board read failed: %s", exc)
            self._model_failed(exc)
            return {"ok": False, "error": f"read failed: {exc}", **base}
        self._model_ok()
        force_protect: tuple = ()
        vouched: list[str] = []
        if dim:
            force_protect = ("work", "personal")
        elif self.settings["depth_guard"]:
            force_protect = tuple(self._physically_blocked(capture_t0))
            if force_protect:
                base["depth_blocked"] = list(force_protect)
            vouched = [b for b in self._depth_vouched(capture_t0)
                       if b not in force_protect]
        seen = dict(seen)   # flags get overruled below; never touch the reader's dict
        overruled = set(self._overrule(seen, vouched))
        if (not dry and not dim and self.settings["tick_confirm"]
                and self._tick_completion_imminent(seen, force_protect)):
            # About to complete a task off this read's ticks: re-photograph
            # and re-read, keeping only ticks BOTH reads agree on (see
            # confirm_ticks). A failed second read demotes every tick to an
            # open sighting — deferring a completion is always safe.
            seen2 = None
            confirm_t0 = time.time()
            try:
                jpeg2, _ = self._fetch_crop()
                if s["presence_recheck"] and self.watcher.blocked_since(confirm_t0):
                    # Somebody walked into the confirmation photo. Leaving
                    # seen2 as None demotes every tick to a plain sighting,
                    # which is the safe direction.
                    log.info("someone stepped in during the tick-confirm "
                             "photo; ticks deferred")
                else:
                    try:
                        seen2 = self.reader.read(jpeg2)
                    except Exception as exc:  # noqa: BLE001
                        self._model_failed(exc)
                        raise
            except Exception as exc:  # noqa: BLE001
                log.warning("tick-confirm read failed (%s); ticks deferred", exc)
            seen = confirm_ticks(seen, seen2)
            overruled |= set(self._overrule(seen, vouched))
            base["tick_confirmed"] = seen2 is not None
        if not dry:
            wipe = self._wipe_imminent(seen, force_protect)
            if wipe:
                # Second photo + read before an empty read completes a stack
                # of tasks. Still empty: a real wipe, proceed. Anything there
                # (or no second read): absence proves nothing this scan.
                try:
                    jpeg3, _ = self._fetch_crop()
                    seen3 = self.reader.read(jpeg3)
                except Exception as exc:  # noqa: BLE001
                    log.warning("wipe-confirm read failed (%s); completions deferred", exc)
                    seen3 = None
                disputed = [b for b in wipe if seen3 is None
                            or seen3.get(b) or seen3.get(f"{b}_done")]
                base["wipe_confirmed"] = [b for b in wipe if b not in disputed]
                if disputed:
                    log.info("empty read of %s disputed by a second read; "
                             "absence deferred", disputed)
                    force_protect = tuple(sorted(set(force_protect) | set(disputed)))
        blocked = [b for b in ("work", "personal") if seen.get(f"{b}_obstructed")]
        if blocked:
            base["obstructed"] = blocked
        if overruled:
            base["obstruction_overruled"] = sorted(overruled)
        if dim:
            base["dim"] = True
        result = self.reconcile(
            seen, apply=not dry,
            tick_start_ok=self.scans_since_change <= TICK_START_CHANGE_WINDOW,
            force_protect=force_protect)
        if not dry and not result["failed"]:
            # A failed HA push means an item is not yet in Todoist; keeping
            # the old baseline forces a re-read (and a retry) next interval.
            np.save(self._baseline_path, gray)
        return {"ok": True, "changed": True, "dry": dry, "seen": seen, **base, **result}

    def _done(self, result: dict) -> dict:
        result["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["t"] = round(time.time(), 1)
        self.last_result = result
        if result.get("ok") and result.get("changed"):
            self.last_read = {k: v for k, v in result.items() if k != "seen"}
        c = self.counters
        c["scans"] += 1
        if not result.get("ok"):
            c["errors"] += 1
        elif result.get("camera_off"):
            c["skips_camera_off"] += 1
        elif result.get("person"):
            c["skips_person_late" if result.get("person_late") else "skips_person"] += 1
        elif result.get("person_unknown"):
            c["skips_person"] += 1
        elif result.get("dim_skipped"):
            c["skips_dim"] += 1
        elif not result.get("changed"):
            c["unchanged"] += 1
        else:
            c["changed"] += 1
        if result.get("obstructed"):
            c["obstructed"] += 1
        if not result.get("dry"):
            c["added"] += len(result.get("added", []))
            c["completed"] += len(result.get("completed", []))
        self._record({"type": "scan", **{k: v for k, v in result.items() if k != "seen"}})
        log.info("scan: %s", json.dumps(result))
        return result

    def _compact_history(self) -> None:
        """Rewrite the file as the in-memory tail, so it can't grow without
        bound (at a 6 min interval it gained ~30 KB a day between boots)."""
        try:
            write_text = "".join(json.dumps(e) + "\n" for e in self.history)
            tmp = self._history_path.with_name(self._history_path.name + ".tmp")
            tmp.write_text(write_text)
            tmp.replace(self._history_path)
        except OSError as exc:
            log.warning("history compaction failed: %s", exc)

    def _model_ok(self) -> None:
        self.model_health.update(failures=0, last_ok_t=round(time.time(), 1))

    def _model_failed(self, exc: Exception) -> None:
        h = self.model_health
        h["failures"] += 1
        h["last_error"] = str(exc)[:300]
        h["last_error_t"] = round(time.time(), 1)

    def model_failing(self) -> bool:
        return self.model_health["failures"] >= MODEL_FAILING_AFTER

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
        self._appends = getattr(self, "_appends", 0) + 1
        if self._appends >= HISTORY_KEEP:
            self._appends = 0
            self._compact_history()

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
        if self.watcher.camera_off:
            # kinect-knob's night window: nothing to photograph. Record it
            # once per window, not every interval all night.
            self.last_scan_t = time.time()
            if not self._camera_off_noted:
                self._camera_off_noted = True
                self._gate_skip("camera_off")
            return
        self._camera_off_noted = False
        self.scan(trigger="interval")

    def _gate_skip(self, reason: str) -> None:
        self.counters[{"home": "skips_home", "camera_off": "skips_camera_off"}
                      .get(reason, "skips_off")] += 1
        self.gate = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "t": round(time.time(), 1), "reason": reason}
        self._record({"type": "skip", "reason": reason})

    def relearn_presence(self) -> tuple[int, dict]:
        """Make kinect-knob re-learn the empty scene, then re-poll.

        The escape hatch for a board that reads permanently occupied: a box
        left on the ledge, a chair pushed into shot, the camera nudged. The
        detector absorbs genuinely motionless objects on its own, but only
        after a long window (it must not absorb a person working there), so
        this is the way to say "that is furniture now, get on with it"."""
        if not self.cfg.kinect_url:
            return 400, {"error": "no KINECT_URL configured"}
        try:
            r = requests.post(f"{self.cfg.kinect_url}/api/presence/relearn", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            return 502, {"error": f"kinect-knob relearn failed: {exc}"}
        try:
            self.watcher.poll()
        except Exception:  # noqa: BLE001
            log.debug("post-relearn poll failed", exc_info=True)
        self._record({"type": "manual", "action": "relearn"})
        log.info("presence background models relearned by request")
        return 200, {"ok": True, "watcher": self.watcher.snapshot()}

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
            "watcher": self.watcher.snapshot(),
            "settings": dict(self.settings),
            "items": self.items_view,
            "last": self.last_result,
            "last_read": self.last_read,
            "model": {**self.model_health, "failing": self.model_failing()},
            "history": self.history,
            "counters": dict(self.counters),
            "crop_t": self.crop_t,
            "config": {"model": cfg.model, "interval_s": cfg.interval_s,
                       "region": list(cfg.region),
                       "default_region": self.default_region,
                       "split": board_split(cfg.region, self.settings["split"]),
                       "items_bottom_margin_px": ITEMS_BOTTOM_MARGIN_PX,
                       "frame": [FRAME_W, FRAME_H],
                       "model_max_pixels": MODEL_MAX_PIXELS,
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
  padding:3px 4px;margin:0 -4px;border-top:1px solid var(--grid);font-size:13px;
  border-radius:7px}
ul.items li:first-child{border-top:0}
ul.items li:hover{background:color-mix(in srgb,var(--ink) 4%,transparent)}
ul.items li.done .txt{text-decoration:line-through;color:var(--muted)}
ul.items li.busy{opacity:.5;pointer-events:none}
/* The text IS the edit affordance: one click opens an inline field. The old
   dashboard put edits behind a blocking window.prompt(), which is slow, ugly
   and cannot be tabbed through. */
ul.items li .txt{flex:1;overflow-wrap:anywhere;cursor:text;padding:3px 4px;
  margin:-3px -4px;border-radius:5px}
ul.items li .txt:hover{box-shadow:inset 0 -1px 0 var(--axis)}
ul.items li .txt:focus-visible{outline:2px solid var(--series);outline-offset:1px}
ul.items li input.edit{flex:1;font:inherit;font-size:13px;color:var(--ink);
  background:var(--page);border:1px solid var(--series);border-radius:6px;
  padding:2px 5px;margin:-3px 0;min-width:0;width:100%}
ul.items li .end{display:flex;align-items:center;gap:4px;flex:none}
button.ib{padding:0;width:26px;height:24px;line-height:22px;font-size:13px;
  border-radius:6px;color:var(--muted);background:transparent;border-color:transparent}
button.ib:hover{color:var(--ink);background:var(--surface);border-color:var(--border)}
button.ib.danger:hover{color:var(--crit)}
#toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:9;
  display:none;align-items:center;gap:14px;background:var(--surface);color:var(--ink);
  border:1px solid var(--axis);border-radius:10px;padding:9px 14px;font-size:13px;
  box-shadow:0 4px 18px rgba(0,0,0,.18);max-width:min(560px,92vw)}
#toast .msg{overflow-wrap:anywhere}
#toast button{padding:3px 11px;font-size:12px}
.hint{font-size:11px;color:var(--muted);font-weight:400;margin-left:8px}
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
/* Framing editor. The camera can't zoom or pan, so the box IS the zoom and
   pan: everything outside it is dimmed, the dashed line is the work|personal
   divider the presence guards use. All geometry is in % of the frame, so it
   scales with the card. */
.stage{position:relative;overflow:hidden;border-radius:8px;border:1px solid var(--border);
  background:#000;aspect-ratio:16/9;touch-action:none;user-select:none;cursor:crosshair}
.stage img{display:block;width:100%;height:100%;object-fit:fill;pointer-events:none}
.stage .loading{position:absolute;inset:0;display:grid;place-items:center;color:#bbb;font-size:13px}
#fr-box{position:absolute;box-shadow:0 0 0 9999px rgba(0,0,0,.55);outline:2px solid #fab219;
  cursor:move;touch-action:none}
#fr-box:focus-visible{outline:3px solid #3987e5}
#fr-box .h{position:absolute;width:14px;height:14px;background:#fab219;border:2px solid #0b0b0b;
  border-radius:3px;margin:-7px 0 0 -7px}
#fr-box .h.nw{left:0;top:0;cursor:nwse-resize} #fr-box .h.n{left:50%;top:0;cursor:ns-resize}
#fr-box .h.ne{left:100%;top:0;cursor:nesw-resize} #fr-box .h.e{left:100%;top:50%;cursor:ew-resize}
#fr-box .h.se{left:100%;top:100%;cursor:nwse-resize} #fr-box .h.s{left:50%;top:100%;cursor:ns-resize}
#fr-box .h.sw{left:0;top:100%;cursor:nesw-resize} #fr-box .h.w{left:0;top:50%;cursor:ew-resize}
#fr-split{position:absolute;top:0;bottom:0;width:16px;margin-left:-8px;cursor:col-resize}
#fr-split::after{content:"";position:absolute;left:7px;top:0;bottom:0;border-left:2px dashed #fab219}
#fr-split.auto::after{opacity:.55}
#fr-box .tag{position:absolute;top:4px;font-size:11px;color:#0b0b0b;background:#fab219;
  padding:0 6px;border-radius:4px;pointer-events:none;white-space:nowrap}
.frbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin:10px 0 0}
.frbar .grp{display:flex;align-items:center;gap:5px;font-size:13px;color:var(--ink2)}
.frbar button{padding:4px 10px}
.frbar button.sq{width:30px;padding:4px 0}
.frbar input[type=number]{width:70px}
.frbar input[type=range]{width:140px;accent-color:var(--series)}
.frbar .primary{background:var(--series);color:#fff;border-color:transparent}
.frbar .primary:disabled{background:var(--surface);color:var(--ink)}
#fr-info{font-size:12px;color:var(--ink2);margin:8px 0 0;font-variant-numeric:tabular-nums}
#fr-info .warn{color:var(--serious)}
#fr-dirty{color:var(--serious)}
.frout{display:grid;grid-template-columns:3fr 2fr;gap:12px;margin-top:12px}
@media(max-width:720px){.frout{grid-template-columns:1fr}}
.frout img{display:block;width:100%;border-radius:8px;border:1px solid var(--border)}
.frout.stale img{opacity:.45}
.frout h3{font-size:12px;font-weight:600;color:var(--ink2);margin:0 0 4px}
.frout ul{list-style:none;margin:0 0 10px;padding:0;font-size:13px}
.frout li{padding:2px 0;border-top:1px solid var(--grid)}
.frout li.done{color:var(--muted);text-decoration:line-through}
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
  <button id="btn-relearn" title="Tell the camera the boards are empty right now — use after moving the camera, the boards, or anything parked in front of them">Relearn empty boards</button>
</header>
<div id="note"></div>

<section class="tiles">
  <div class="tile"><div class="label">Sync state</div><div class="value" id="t-state">&ndash;</div><div class="sub" id="t-state-sub"></div></div>
  <div class="tile"><div class="label" id="t-next-label">Next check</div><div class="value" id="t-next">&ndash;</div><div class="sub" id="t-next-sub"></div></div>
  <div class="tile"><div class="label">Tracked items</div><div class="value" id="t-items">&ndash;</div><div class="sub" id="t-items-sub"></div></div>
  <div class="tile"><div class="label">Last scan</div><div class="value" id="t-last">&ndash;</div><div class="sub" id="t-last-sub"></div></div>
</section>

<div class="grid2">
  <section class="card"><h2>Work board <span class="hint">click text to edit · Tab for the next · × to delete</span><small id="pres-work"></small></h2><ul class="items" id="list-work"></ul></section>
  <section class="card"><h2>Personal board<small id="pres-personal"></small></h2><ul class="items" id="list-personal"></ul></section>
</div>

<section class="card"><h2>Controls <small>changes apply immediately and persist</small></h2>
  <div id="settings" class="setgrid"></div>
</section>

<section class="card"><h2>Camera framing
    <span class="hint">drag the box to pan · drag a handle to zoom · drag outside to draw a new box · dashed line = where the work board ends</span>
    <small id="fr-dirty"></small></h2>
  <div class="stage" id="fr-stage">
    <img id="fr-img" alt="Full camera frame" hidden>
    <div class="loading" id="fr-loading">Loading camera frame&hellip;</div>
    <div id="fr-box" tabindex="0" aria-label="Crop box: arrow keys pan, + and - zoom">
      <span class="tag" id="fr-tag-l">work</span><span class="tag" id="fr-tag-r">personal</span>
      <div id="fr-split" title="Work | personal divider for the presence guard. Double-click to return to the middle."></div>
      <i class="h nw"></i><i class="h n"></i><i class="h ne"></i><i class="h e"></i>
      <i class="h se"></i><i class="h s"></i><i class="h sw"></i><i class="h w"></i>
    </div>
  </div>
  <div class="frbar">
    <span class="grp">Zoom <button class="sq" id="fr-zout" title="Zoom out">&minus;</button>
      <input type="range" id="fr-zoom" min="1" max="8" step="0.05" aria-label="Zoom">
      <button class="sq" id="fr-zin" title="Zoom in">+</button></span>
    <span class="grp">Pan <button class="sq" data-pan="-1,0" title="Left">&larr;</button><button class="sq" data-pan="0,-1" title="Up">&uarr;</button><button class="sq" data-pan="0,1" title="Down">&darr;</button><button class="sq" data-pan="1,0" title="Right">&rarr;</button></span>
    <span class="grp">x1 <input type="number" id="fr-x1" step="2"> y1 <input type="number" id="fr-y1" step="2">
      x2 <input type="number" id="fr-x2" step="2"> y2 <input type="number" id="fr-y2" step="2">
      split <input type="number" id="fr-sp" step="2" title="0 = middle of the box"></span>
  </div>
  <div class="setgrid" id="fr-settings" style="margin-top:8px"></div>
  <div id="fr-info"></div>
  <div class="frbar">
    <button id="fr-refresh" title="Take a fresh full-frame photo">Refresh frame</button>
    <button id="fr-preview" title="Show exactly what the model would be sent">Preview model view</button>
    <button id="fr-read" title="Preview, then run the model on it. Nothing is saved and Todoist is not touched">Test read</button>
    <span class="spacer"></span>
    <button id="fr-reset" title="Back to WB_REGION from .env (not saved until you press Save)">Default</button>
    <button id="fr-revert" title="Discard unsaved changes">Revert</button>
    <button id="fr-save" class="primary">Save framing</button>
  </div>
  <div class="frout" id="fr-out" hidden>
    <div><h3 id="fr-out-cap"></h3><img id="fr-out-img" alt="What the model is sent"></div>
    <div id="fr-out-read"></div>
  </div>
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
<div id="toast" role="status" aria-live="polite">
  <span class="msg"></span><button id="toast-act" hidden></button>
</div>
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
  if (r.type === 'manual') return r.action === 'edit'
    ? 'edited (' + r.board + '): “' + r.from + '” → “' + r.to + '”'
    : (r.action === 'reopen' ? 'reopened' : 'removed') +
      ' (' + r.board + '): “' + r.text + '”';
  if (r.type === 'skip') return r.reason === 'home'
    ? 'skipped — ' + who() + ' was home'
    : r.reason === 'camera_off' ? 'camera off for the night — scans paused'
    : 'skipped — scheduled scans off';
  if (r.error) return 'error: ' + r.error;
  if (r.camera_off) return 'camera off (' + r.camera_off + ') — skipped';
  if (r.dark) return 'too dark to read (luma ' + r.luma + ') — skipped';
  if (r.person) return (r.person_late
      ? 'skipped — someone stepped in front of the ' + r.person.join(' + ') + ' board'
        + (r.person.length > 1 ? 's' : '') + ' while the photo was being taken'
      : 'skipped — someone at the ' + r.person.join(' + ') + ' board'
        + (r.person.length > 1 ? 's' : ''))
    + (r.person_via === 'depth' ? ' (depth percentiles)' : '');
  if (r.person_unknown) return 'skipped — no presence evidence from the camera';
  if (r.type === 'manual' && r.action === 'relearn')
    return 'camera re-learned the empty boards';
  if (r.dim_skipped) return 'too dim to trust (luma ' + r.luma + ') — skipped';
  if (r.obstructed === true) return 'board obstructed — left alone';  // legacy records
  if (r.changed === false) return 'no change';
  const a = (r.added || []).length, c = (r.completed || []).length;
  let s = r.dry ? 'dry run: would add ' + a + ', complete ' + c
                : a + ' added, ' + c + ' completed';
  if (Array.isArray(r.pending) && r.pending.length)
    s += ', ' + r.pending.length + ' new awaiting confirmation';
  if (Array.isArray(r.obstructed) && r.obstructed.length)
    s += ' (' + r.obstructed.join(' + ') + ' board blocked — completions deferred)';
  if (Array.isArray(r.slowed) && r.slowed.length)
    s += ' (mass-erase guard: ' + r.slowed.join(' + ') + ' completions slowed)';
  if (Array.isArray(r.parked) && r.parked.length)
    s += ', ' + r.parked.length + ' ticked before confirmation (no task): ' + r.parked.join(', ');
  if (Array.isArray(r.orphaned) && r.orphaned.length)
    s += ', ' + r.orphaned.length + ' already gone from Todoist (settled): ' + r.orphaned.join(', ');
  if (r.tick_confirmed === false)
    s += ' (confirm photo failed — tick completions deferred)';
  if (r.dim) s += ' (low light: 32-frame stack, completions deferred)';
  if (Array.isArray(r.depth_blocked) && r.depth_blocked.length)
    s += ' (depth: ' + r.depth_blocked.join(' + ') + ' physically blocked)';
  if (r.obstructed_ignored) s += ' (obstruction ignored)';               // legacy
  if (r.failed) s += ' (' + r.failed + ' HA call' + (r.failed > 1 ? 's' : '') + ' failed, will retry)';
  return s;
}
function kind(r) {
  if (r.type === 'skip' || r.type === 'settings' || r.type === 'manual'
      || r.changed === false) return 'muted';
  if (r.error) return 'crit';
  if (r.failed || r.obstructed || r.obstructed_ignored) return 'warn';
  return 'good';
}

const SETTINGS_UI = [
  ['enabled', 'bool', 'Scheduled scans'],
  ['presence_gate', 'bool', 'Only scan while away'],
  ['change_detection', 'bool', 'Skip unchanged frames'],
  ['obstruction_guard', 'bool', 'Skip when obstructed'],
  ['due_today', 'bool', 'New tasks due today'],
  ['interval_s', 'num', 'Scan interval (s)', { min: 60, max: 86400, step: 60 }],
  ['change_threshold', 'num', 'Change threshold', { min: 0.1, max: 50, step: 0.5 }],
  ['missing_to_complete', 'num', 'Misses to complete', { min: 1, max: 10, step: 1 }],
  ['ticked_to_complete', 'num', 'Ticks to complete', { min: 1, max: 10, step: 1 }],
  ['sightings_to_add', 'num', 'Sightings to create task', { min: 1, max: 10, step: 1 }],
  ['tick_confirm', 'bool', 'Confirm ticks (2nd photo)'],
  ['low_light_boost', 'bool', 'Low light: stack 32 frames'],
  ['depth_guard', 'bool', 'Depth guard (ToF obstruction)'],
  ['depth_skip', 'bool', 'Skip scan when someone is at a board'],
  ['presence_occupancy', 'num', 'Person sensitivity (board coverage)', { min: 0.002, max: 0.5, step: 0.002 }],
  ['presence_clear_s', 'num', 'Board must be clear for (s)', { min: 0, max: 600, step: 5 }],
  ['presence_recheck', 'bool', 'Re-check for a person after the photo'],
  ['presence_required', 'bool', 'Wait when presence is unknown'],
  ['dim_skip', 'bool', 'Skip scan when light is dim'],
  ['dim_luma', 'num', 'Dim threshold (luma 0-255)', { min: 10, max: 150, step: 5 }],
];
// Photo + image settings live in the framing card: they are drafted, previewed
// and saved together with the crop box, not applied one control at a time.
const FRAMING_UI = [
  ['enhance', 'bool', 'Enhance photo (contrast+sharpen)'],
  ['capture_frames', 'num', 'Frames stacked per photo', { min: 1, max: 32, step: 1 }],
  ['upscale', 'sel', 'Upscale for model', [['1', '1x'], ['1.5', '1.5x'], ['2', '2x'], ['2.5', '2.5x'], ['3', '3x']]],
  ['capture_format', 'sel', 'Image format to model', [['jpeg', 'JPEG'], ['png', 'PNG (lossless)']]],
  ['capture_quality', 'num', 'JPEG quality to model', { min: 50, max: 100, step: 1 }],
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
// ---- framing editor -------------------------------------------------------
// fr is the DRAFT: the crop box + image settings being edited. It follows the
// live settings until the first edit, then holds still until Save or Revert.
// Coordinates are always full-frame camera pixels (1920x1080).
const FR_KEYS = ['region', 'split', ...FRAMING_UI.map((f) => f[0])];
let fr = null, frDirty = false, frGen = 0, frShown = -1;
const FW = () => st.config.frame[0], FH = () => st.config.frame[1];
const MIN_W = 160, MIN_H = 120;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const liveFraming = () => Object.fromEntries(FR_KEYS.map((k) =>
  [k, Array.isArray(st.settings[k]) ? st.settings[k].slice() : st.settings[k]]));
const sameFraming = (a, b) => JSON.stringify(FR_KEYS.map((k) => a[k])) ===
  JSON.stringify(FR_KEYS.map((k) => b[k]));
const splitX = () => { const [x1, , x2] = fr.region;
  return fr.split > x1 && fr.split < x2 ? fr.split : Math.round((x1 + x2) / 2); };

function frSet(region, split) {
  // Keep the box inside the frame and at least MIN_W x MIN_H, rounded to even px.
  let [x1, y1, x2, y2] = region.map((v) => Math.round(v / 2) * 2);
  x1 = clamp(x1, 0, FW() - MIN_W); y1 = clamp(y1, 0, FH() - MIN_H);
  x2 = clamp(x2, x1 + MIN_W, FW()); y2 = clamp(y2, y1 + MIN_H, FH());
  fr.region = [x1, y1, x2, y2];
  if (split !== undefined) fr.split = split;
  frChanged();
}
function frChanged() {
  frDirty = !sameFraming(fr, liveFraming());
  frGen++;
  frDraw();
}
function frScale(w, h) {
  return Math.min(Number(fr.upscale), Math.sqrt(st.config.model_max_pixels / (w * h)));
}
function frDraw() {
  const [x1, y1, x2, y2] = fr.region, w = x2 - x1, h = y2 - y1;
  const box = $('fr-box');
  box.style.left = (x1 / FW() * 100) + '%'; box.style.top = (y1 / FH() * 100) + '%';
  box.style.width = (w / FW() * 100) + '%'; box.style.height = (h / FH() * 100) + '%';
  const sx = splitX();
  $('fr-split').style.left = ((sx - x1) / w * 100) + '%';
  $('fr-split').classList.toggle('auto', !(fr.split > x1 && fr.split < x2));
  $('fr-tag-l').style.left = '4px';
  $('fr-tag-r').style.left = 'calc(' + ((sx - x1) / w * 100) + '% + 6px)';
  for (const [k, v] of [['x1', x1], ['y1', y1], ['x2', x2], ['y2', y2], ['sp', fr.split]]) {
    if (document.activeElement !== $('fr-' + k)) $('fr-' + k).value = v;
  }
  if (document.activeElement !== $('fr-zoom')) $('fr-zoom').value = (FW() / w).toFixed(2);
  for (const [key, type] of FRAMING_UI) {
    const inp = $('fr-set-' + key);
    if (document.activeElement === inp) continue;
    if (type === 'bool') inp.checked = !!fr[key]; else inp.value = String(fr[key]);
  }
  $('fr-set-capture_quality').disabled = fr.capture_format === 'png';
  const sc = frScale(w, h), mw = Math.round(w * sc), mh = Math.round(h * sc);
  const capped = sc < Number(fr.upscale) - 0.01;
  $('fr-info').innerHTML = '';
  $('fr-info').append(
    'Crop ' + w + '×' + h + ' px of ' + FW() + '×' + FH() + ' · zoom ' + (FW() / w).toFixed(2) +
    '× · model is sent ' + mw + '×' + mh + ' (' + (mw * mh / 1e6).toFixed(2) + ' MP' +
    (sc > 1.01 ? ', ' + sc.toFixed(2) + '× upscale' : sc < 0.99 ? ', shrunk ' + sc.toFixed(2) + '×' : '') + ')');
  if (capped) $('fr-info').append(el('span', 'warn', ' · upscale capped at ' +
    (st.config.model_max_pixels / 1e6).toFixed(1) + ' MP so the image fits the model context'));
  $('fr-dirty').textContent = frDirty ? 'unsaved changes' : 'saved';
  $('fr-save').disabled = !frDirty; $('fr-revert').disabled = !frDirty;
  $('fr-out').classList.toggle('stale', frShown !== frGen);
}

function frZoom(factor) {
  // Zoom about the box centre, keeping its aspect ratio.
  const [x1, y1, x2, y2] = fr.region, cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  let w = clamp((x2 - x1) / factor, MIN_W, FW()), h = (y2 - y1) * w / (x2 - x1);
  if (h > FH()) { w *= FH() / h; h = FH(); }
  if (h < MIN_H) { w *= MIN_H / h; h = MIN_H; }
  let nx = clamp(cx - w / 2, 0, FW() - w), ny = clamp(cy - h / 2, 0, FH() - h);
  frSet([nx, ny, nx + w, ny + h]);
}
function frPan(dx, dy) {
  const [x1, y1, x2, y2] = fr.region, w = x2 - x1, h = y2 - y1;
  const nx = clamp(x1 + dx, 0, FW() - w), ny = clamp(y1 + dy, 0, FH() - h);
  const sx = fr.split > x1 && fr.split < x2 ? fr.split + (nx - x1) : fr.split;
  frSet([nx, ny, nx + w, ny + h], sx);
}

// Pointer editing: drag the box = pan, a handle = resize that edge/corner,
// the dashed line = move the divider, empty frame = draw a fresh box.
let drag = null;
function frPoint(e) {
  const r = $('fr-stage').getBoundingClientRect();
  return [(e.clientX - r.left) / r.width * FW(), (e.clientY - r.top) / r.height * FH()];
}
function frDown(e) {
  if (!fr || e.button > 0) return;
  const t = e.target, p = frPoint(e);
  let mode = 'draw';
  if (t.id === 'fr-split') mode = 'split';
  else if (t.classList.contains('h')) mode = [...t.classList].find((c) => c !== 'h');
  else if (t.closest('#fr-box')) mode = 'move';
  drag = { mode, p0: p, r0: fr.region.slice(), s0: fr.split };
  $('fr-stage').setPointerCapture(e.pointerId);
  e.preventDefault();
}
function frMove(e) {
  if (!drag) return;
  const [px, py] = frPoint(e), dx = px - drag.p0[0], dy = py - drag.p0[1];
  let [x1, y1, x2, y2] = drag.r0;
  const m = drag.mode;
  if (m === 'move') {
    const w = x2 - x1, h = y2 - y1;
    const nx = clamp(x1 + dx, 0, FW() - w), ny = clamp(y1 + dy, 0, FH() - h);
    const s0 = drag.s0, keep = s0 > x1 && s0 < x2;
    frSet([nx, ny, nx + w, ny + h], keep ? s0 + (nx - x1) : s0);
    return;
  }
  if (m === 'split') { frSet(fr.region, Math.round(clamp(px, x1 + 20, x2 - 20) / 2) * 2); return; }
  if (m === 'draw') {
    const [ax, ay] = drag.p0;
    if (Math.abs(dx) < 8 && Math.abs(dy) < 8) return;   // a click, not a drag
    x1 = Math.min(ax, px); x2 = Math.max(ax, px); y1 = Math.min(ay, py); y2 = Math.max(ay, py);
    frSet([clamp(x1, 0, FW()), clamp(y1, 0, FH()), clamp(x2, 0, FW()), clamp(y2, 0, FH())], 0);
    return;
  }
  if (m.includes('w')) x1 = clamp(x1 + dx, 0, x2 - MIN_W);
  if (m.includes('e')) x2 = clamp(x2 + dx, x1 + MIN_W, FW());
  if (m.includes('n')) y1 = clamp(y1 + dy, 0, y2 - MIN_H);
  if (m.includes('s')) y2 = clamp(y2 + dy, y1 + MIN_H, FH());
  frSet([x1, y1, x2, y2]);
}
function frUp() { drag = null; }

function frLoadFrame() {
  const img = $('fr-img');
  $('fr-loading').hidden = false; $('fr-loading').textContent = 'Loading camera frame…';
  img.onload = () => { img.hidden = false; $('fr-loading').hidden = true; };
  img.onerror = () => { $('fr-loading').hidden = false;
    $('fr-loading').textContent = 'Camera frame unavailable — is kinect-knob running? Try Refresh frame.'; };
  img.src = '/api/frame?t=' + Date.now();
}

async function frPreview(read) {
  const btns = ['fr-preview', 'fr-read'].map($);
  btns.forEach((b) => { b.disabled = true; });
  const gen = frGen, label = read ? $('fr-read') : $('fr-preview'), old = label.textContent;
  label.textContent = read ? 'Reading… (can take 30 s)' : 'Capturing…';
  try {
    const r = await fetch('/api/framing/preview', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...Object.fromEntries(FR_KEYS.map((k) => [k, fr[k]])), read }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error + (j.rejected ? ': ' + JSON.stringify(j.rejected) : ''));
    $('fr-out').hidden = false;
    $('fr-out-img').src = j.image;
    $('fr-out-cap').textContent = 'Model view ' + j.model[0] + '×' + j.model[1] + ' · ' +
      Math.round(j.bytes / 1024) + ' KB · luma ' + j.luma +
      (j.luma < st.settings.dim_luma ? ' (dim — scheduled scans would skip)' : '') +
      ' · ' + j.capture_s + ' s to capture';
    const out = $('fr-out-read'); out.innerHTML = '';
    if (read) frRenderRead(out, j); else
      out.append(el('div', 'empty', 'Test read runs the vision model on this exact image — nothing is saved and Todoist is not touched.'));
    frShown = gen; frDraw();
  } catch (e) { $('note').textContent = 'Framing preview failed: ' + e.message; }
  finally { label.textContent = old; btns.forEach((b) => { b.disabled = false; }); }
}
function frRenderRead(out, j) {
  if (j.read_error) { out.append(el('div', 'empty', 'Read failed after ' + j.read_s + ' s: ' + j.read_error)); return; }
  const rd = j.read;
  out.append(el('h3', null, 'Model read in ' + j.read_s + ' s'));
  for (const [board, name] of [['work', 'Work (left)'], ['personal', 'Personal (right)']]) {
    const done = new Set(rd[board + '_done'] || []), order = rd[board + '_order'] || [];
    out.append(el('h3', null, name + ' · ' + order.length + ' items' +
      (rd[board + '_obstructed'] ? ' · model says OBSTRUCTED' : '')));
    const ul = el('ul');
    if (!order.length) ul.append(el('li', 'done', 'nothing read'));
    for (const t of order) ul.append(el('li', done.has(t) ? 'done' : null, (done.has(t) ? '☑ ' : '☐ ') + t));
    out.append(ul);
  }
}

async function frSave() {
  $('fr-save').disabled = true;
  try {
    const r = await fetch('/api/settings', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.fromEntries(FR_KEYS.map((k) => [k, fr[k]]))) });
    const j = await r.json();
    const rej = Object.entries(j.rejected || {});
    if (rej.length) { $('note').textContent = 'Not saved: ' + rej.map(([k, v]) => k + ' ' + v).join('; '); return; }
    $('note').textContent = 'Framing saved — the next scan reads the boards fresh with it' +
      (j.applied && j.applied.region ? '. If presence starts misbehaving, press Relearn empty boards.' : '.');
    await refresh();
    fr = liveFraming(); frDirty = false;   // same framing: a preview of it is still current
  } catch (e) { $('note').textContent = 'Save failed: ' + e; }
  finally { frDraw(); }
}

function frBuild() {
  const host = $('fr-settings');
  for (const [key, type, label, opt] of FRAMING_UI) {
    const row = el('div', 'setrow');
    const lab = el('label', null, label); lab.htmlFor = 'fr-set-' + key;
    row.appendChild(lab);
    let inp;
    if (type === 'bool') { inp = el('input'); inp.type = 'checkbox'; }
    else if (type === 'num') { inp = el('input'); inp.type = 'number';
      inp.min = opt.min; inp.max = opt.max; inp.step = opt.step; }
    else { inp = el('select');
      for (const [v, l] of opt) { const o = el('option', null, l); o.value = v; inp.appendChild(o); } }
    inp.id = 'fr-set-' + key;
    inp.addEventListener('change', () => {
      if (type === 'bool') fr[key] = inp.checked;
      else if (type === 'num') fr[key] = clamp(Number(inp.value) || opt.min, opt.min, opt.max);
      else fr[key] = key === 'upscale' ? Number(inp.value) : inp.value;
      frChanged();
    });
    row.appendChild(inp); host.appendChild(row);
  }
  const stage = $('fr-stage');
  stage.addEventListener('pointerdown', frDown);
  stage.addEventListener('pointermove', frMove);
  stage.addEventListener('pointerup', frUp);
  stage.addEventListener('pointercancel', frUp);
  $('fr-split').addEventListener('dblclick', () => frSet(fr.region, 0));
  $('fr-box').addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 40 : 4;
    const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (moves[e.key]) { frPan(...moves[e.key]); e.preventDefault(); }
    else if (e.key === '+' || e.key === '=') { frZoom(1.05); e.preventDefault(); }
    else if (e.key === '-') { frZoom(1 / 1.05); e.preventDefault(); }
  });
  $('fr-zin').onclick = () => frZoom(1.1);
  $('fr-zout').onclick = () => frZoom(1 / 1.1);
  $('fr-zoom').addEventListener('input', () => {
    const w = fr.region[2] - fr.region[0];
    frZoom(Number($('fr-zoom').value) / (FW() / w));
  });
  for (const b of document.querySelectorAll('[data-pan]')) {
    b.onclick = (e) => { const [dx, dy] = b.dataset.pan.split(',').map(Number);
      const step = e.shiftKey ? 80 : 16; frPan(dx * step, dy * step); };
  }
  for (const k of ['x1', 'y1', 'x2', 'y2']) {
    $('fr-' + k).addEventListener('change', () => {
      const r = ['x1', 'y1', 'x2', 'y2'].map((n) => Number($('fr-' + n).value) || 0);
      frSet(r);
    });
  }
  $('fr-sp').addEventListener('change', () => frSet(fr.region, Math.max(0, Number($('fr-sp').value) || 0)));
  $('fr-refresh').onclick = frLoadFrame;
  $('fr-preview').onclick = () => frPreview(false);
  $('fr-read').onclick = () => frPreview(true);
  $('fr-save').onclick = frSave;
  $('fr-revert').onclick = () => { fr = liveFraming(); frChanged(); };
  $('fr-reset').onclick = () => frSet(st.config.default_region.slice(), 0);
  frLoadFrame();
}
function frSync() {
  if (!fr) { fr = liveFraming(); frBuild(); }
  else if (!frDirty && !drag && !sameFraming(fr, liveFraming())) fr = liveFraming();
  frDraw();
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
  else if (last && Array.isArray(last.obstructed) && last.obstructed.length)
    pill('warn', last.obstructed.join(' + ') + ' board blocked — completions deferred');
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
  const open = items.filter((i) => i.status === 'open');
  const w = open.filter((i) => i.board === 'work').length;
  $('t-items').textContent = open.length;
  const doneN = items.filter((i) => i.status === 'done').length;
  const removedN = items.filter((i) => i.status === 'removed').length;
  const pendingN = items.filter((i) => i.status === 'pending').length;
  $('t-items-sub').textContent = w + ' work · ' + (open.length - w) + ' personal' +
    (pendingN ? ' · ' + pendingN + ' unconfirmed' : '') +
    (doneN ? ' · ' + doneN + ' ticked on board' : '') +
    (removedN ? ' · ' + removedN + ' removed' : '');

  $('t-last').textContent = last ? ago(last.t) : 'never';
  $('t-last-sub').textContent = last ? outcome(last) : 'no scans since start';

  lists();
  presenceChips();

  if (st.crop_t) {
    if (st.crop_t !== cropShown) { $('crop').src = '/crop.jpg?t=' + st.crop_t; cropShown = st.crop_t; }
    $('crop').hidden = false; $('crop-empty').hidden = true;
    $('crop-cap').textContent = 'captured ' + ago(st.crop_t);
  }

  syncSettings(); frSync(); chart(); table();

  const c = st.counters, cfg = st.config, set = st.settings;
  $('counters').textContent = 'since start: ' + c.scans + ' scans · ' + c.changed +
    ' processed · ' + c.unchanged + ' unchanged · ' + c.obstructed + ' obstructed · ' +
    c.errors + ' errors · ' + c.skips_home + ' home-skips · ' + c.skips_off + ' off-skips · ' +
    (c.skips_person || 0) + ' person-skips · ' +
    (c.skips_person_late || 0) + ' spoiled mid-photo · ' +
    (c.skips_dim || 0) + ' dim-skips · ' + (c.skips_camera_off || 0) + ' camera-off';
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

async function itemCall(path, body, okMsg, it, onOk) {
  // Mark the row busy immediately so a second click cannot race the first:
  // edits and deletes take a Home Assistant round-trip, and the scan lock can
  // hold them for seconds.
  const key = it ? keyOf(it) : null;
  if (key) { busyKeys.add(key); lists(); }
  let ok = false, warning = null;
  try {
    const r = await fetch(path, { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    ok = !j.error;
    warning = j.warning || null;
    $('note').textContent = j.error ? 'Failed: ' + j.error
      : (okMsg || '') + (warning ? ' — warning: ' + warning : '');
  } catch (e) { $('note').textContent = 'Request failed: ' + e; }
  if (key) busyKeys.delete(key);
  await refresh();
  // Outside the try on purpose: a bug in a toast callback must not be
  // reported to the user as "the request failed" when the request succeeded.
  if (ok && warning) toast(warning);
  if (ok && onOk) onOk();
  return ok;
}
function presenceChips() {
  // Say plainly, per board, whether the camera can see a person in front of
  // it — the thing that decides whether a scan happens at all.
  const w = (st.watcher) || {};
  for (const board of ['work', 'personal']) {
    const host = $('pres-' + board);
    if (!host) continue;
    if (w.supported === false) { host.textContent = 'no depth presence signal'; continue; }
    const rec = (w.boards || {})[board];
    if (!rec || rec.age_s > 30) {
      host.textContent = w.error ? 'presence unknown — camera unreachable'
                                 : 'presence unknown';
      continue;
    }
    if (rec.blocked) {
      host.textContent = 'someone is at this board (' +
        (rec.occupancy * 100).toFixed(1) + '% covered) — not scanning';
      continue;
    }
    const need = st.settings.presence_clear_s || 0;
    const clear = rec.clear_for_s;
    host.textContent = (clear !== null && clear < need)
      ? 'clear for ' + Math.round(clear) + 's — waiting for ' + need + 's'
      : 'clear';
  }
}

// -- item list + inline editing ---------------------------------------------
// Editing used to go through window.prompt() and deleting through
// window.confirm(): two blocking modal windows per change, no keyboard flow,
// and a 5s poll that would rebuild the list underneath you. Now the text IS
// the field, Enter commits, Tab walks the list, and delete is one click with
// an undo — so cleaning up a board's worth of OCR wobble takes seconds.
let editing = null;      // {board, text} while an inline editor is open
let pendingEdit = null;  // {board, index} -> open an editor after the next build
let busyKeys = new Set();

const keyOf = (it) => it.board + '\u0000' + it.text;

function itemState(it) {
  if (it.status === 'done') return ['var(--good)', 'done — erase whenever'];
  if (it.status === 'removed') return ['var(--muted)', 'removed — ignored until erased'];
  if (it.status === 'pending')
    return ['var(--warn)', 'seen ' + it.seen + '/' + st.settings.sightings_to_add + ' — no task yet'];
  if (it.ticked > 0)
    return ['var(--good)', 'ticked ' + it.ticked + '/' + st.settings.ticked_to_complete];
  if (it.missing > 0)
    return ['var(--serious)', 'missing ' + it.missing + '/' + st.settings.missing_to_complete];
  return null;
}

function lists() {
  // Never rebuild under an open editor: the 5s poll would blow away what is
  // being typed. Everything else on the page keeps refreshing.
  if (editing) return;
  const items = st.items || [];
  for (const board of ['work', 'personal']) {
    const ul = $('list-' + board); ul.textContent = '';
    const rows = items.filter((i) => i.board === board)
      .sort((a, b) => (a.pos ?? 1e9) - (b.pos ?? 1e9));  // as written on the board
    if (!rows.length) { ul.appendChild(el('li', 'empty', 'nothing tracked')); continue; }
    rows.forEach((it, index) => {
      const cls = [];
      if (it.status === 'done' || it.status === 'removed') cls.push('done');
      if (busyKeys.has(keyOf(it))) cls.push('busy');
      const li = el('li', cls.join(' ') || null);

      const txt = el('span', 'txt', it.text);
      txt.tabIndex = 0;
      txt.title = 'Click to edit · Delete to remove';
      txt.addEventListener('click', () => startEdit(li, it, board, index));
      txt.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' || ev.key === 'F2') {
          ev.preventDefault(); startEdit(li, it, board, index);
        } else if (ev.key === 'Delete') {
          ev.preventDefault(); removeItem(it);
        }
      });
      li.appendChild(txt);

      const end = el('span', 'end');
      const state = itemState(it);
      if (state) {
        const chip = el('span', 'chip');
        const d = el('span', 'dot'); d.style.background = state[0];
        chip.appendChild(d);
        chip.appendChild(document.createTextNode(state[1]));
        end.appendChild(chip);
      }
      if (it.status === 'done' || it.status === 'removed') {
        const br = el('button', 'ib', '\u21a9');
        br.title = 'Mark as not done — creates a fresh Todoist task';
        br.addEventListener('click', () => reopenItem(it));
        end.appendChild(br);
      }
      if (it.status !== 'removed') {
        const bx = el('button', 'ib danger', '\u00d7');
        bx.title = it.status === 'open'
          ? 'Delete — removes the Todoist task (undoable)' : 'Remove from tracking';
        bx.addEventListener('click', () => removeItem(it));
        end.appendChild(bx);
      }
      li.appendChild(end);
      ul.appendChild(li);

      if (pendingEdit && pendingEdit.board === board && pendingEdit.index === index) {
        pendingEdit = null;
        startEdit(li, it, board, index);
      }
    });
  }
  if (pendingEdit) pendingEdit = null;   // that row went away (renamed/removed)
}

function startEdit(li, it, board, index) {
  if (editing) return;
  const txt = li.querySelector('.txt');
  if (!txt) return;
  editing = { board: it.board, text: it.text };
  const inp = el('input', 'edit');
  inp.type = 'text';
  inp.value = it.text;
  inp.setAttribute('aria-label', 'Item text');
  li.replaceChild(inp, txt);
  inp.focus();
  inp.select();

  let settled = false;
  const finish = (save, thenEdit) => {
    if (settled) return;
    settled = true;
    const next = inp.value.trim();
    editing = null;
    if (thenEdit !== undefined) pendingEdit = { board: board, index: thenEdit };
    if (!save || !next || next === it.text) { lists(); return; }
    itemCall('/api/items/edit',
             { board: it.board, text: it.text, new_text: next },
             'Renamed to \u201c' + next + '\u201d', it);
  };
  inp.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); finish(true); }
    else if (ev.key === 'Escape') { ev.preventDefault(); finish(false); }
    else if (ev.key === 'Tab') {
      // Commit and walk: the fast path for fixing up a whole board.
      ev.preventDefault();
      finish(true, index + (ev.shiftKey ? -1 : 1));
    }
  });
  inp.addEventListener('blur', () => finish(true));
}

function removeItem(it) {
  // No confirm dialog. Deleting an OPEN item removes its Todoist task, and
  // reopen puts that task back verbatim — a real undo, so the honest UI is
  // "do it, offer the undo". The other states touch no live task, so there is
  // nothing an undo could faithfully restore and none is offered.
  const undoable = it.status === 'open';
  itemCall('/api/items/remove', { board: it.board, text: it.text }, null, it,
           () => undoable
             ? toast('Deleted \u201c' + it.text + '\u201d', 'Undo',
                     () => itemCall('/api/items/reopen',
                                    { board: it.board, text: it.text },
                                    'Restored \u201c' + it.text + '\u201d', it))
             : toast(it.status === 'pending'
                       ? 'Dropped \u201c' + it.text + '\u201d — it had no task yet'
                       : 'Removed \u201c' + it.text + '\u201d from tracking'));
}

function reopenItem(it) {
  itemCall('/api/items/reopen', { board: it.board, text: it.text }, null, it,
           () => toast('Reopened \u201c' + it.text + '\u201d as a fresh Todoist ' +
                       'task. If the checkbox is still marked on the board, wipe ' +
                       'it off — otherwise the next scans just complete it again.'));
}

let toastTimer = null;
function toast(msg, actionLabel, onAction, ms) {
  const box = $('toast'), act = $('toast-act');
  box.querySelector('.msg').textContent = msg;
  act.hidden = !actionLabel;
  act.textContent = actionLabel || '';
  act.onclick = actionLabel ? () => { hideToast(); onAction(); } : null;
  box.style.display = 'flex';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, ms || (actionLabel ? 9000 : 5000));
}
function hideToast() { clearTimeout(toastTimer); $('toast').style.display = 'none'; }

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
$('btn-relearn').addEventListener('click', async () => {
  const b = $('btn-relearn'); b.disabled = true;
  try {
    const j = await (await fetch('/api/presence/relearn', { method: 'POST' })).json();
    toast(j.error ? 'Relearn failed: ' + j.error
                  : 'The camera has re-learned the empty boards.');
  } catch (e) { toast('Relearn failed: ' + e); }
  b.disabled = false;
  refresh();
});
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
                # A loop that runs but whose every read fails is just as dead
                # to the user (2026-10-28: 44 h of model timeouts behind "ok").
                failing = syncer.model_failing()
                self._json(500 if stalled else 503 if failing else 200,
                           {"status": "loop stalled" if stalled
                                      else "vision model failing" if failing else "ok",
                            "loop_beat_age_s": round(age),
                            "model": syncer.model_health})
            elif path == "/api/status":
                self._json(200, syncer.status())
            elif path == "/state":
                self._json(200, {"items": syncer.items_view, "last": syncer.last_result})
            elif path == "/crop.jpg":
                try:
                    self._bytes(200, syncer.crop_path.read_bytes(), "image/jpeg")
                except OSError:
                    self._json(404, {"error": "no crop captured yet"})
            elif path == "/api/frame":
                # Full uncropped frame for the framing editor.
                q = parse_qs(urlparse(self.path).query)
                try:
                    frames = max(1, min(LOWLIGHT_FRAMES, int((q.get("frames") or ["1"])[0])))
                    body, ctype = syncer.live_frame(frames)
                    self._bytes(200, body, ctype)
                except ValueError:
                    self._json(400, {"error": "frames must be an integer"})
                except requests.RequestException as exc:
                    self._json(502, {"error": f"snapshot failed: {exc}"})
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
            elif url.path == "/api/framing/preview":
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("want a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    self._json(400, {"error": f"bad body: {exc}"})
                    return
                code, resp = syncer.framing_preview(body)
                self._json(code, resp)
            elif url.path == "/api/presence/relearn":
                # Ask kinect-knob to forget the learned empty-board plane.
                # Needed after the camera or the boards move — and the escape
                # hatch if something parked in front of a board has latched it
                # "occupied" and scans have stopped.
                code, resp = syncer.relearn_presence()
                self._json(code, resp)
            elif url.path in ("/api/items/edit", "/api/items/remove",
                              "/api/items/reopen"):
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("want a JSON object")
                    board, text = body.get("board"), body.get("text")
                    if board not in ("work", "personal") or not isinstance(text, str):
                        raise ValueError("want board ('work'|'personal') and text")
                    if url.path.endswith("/edit") and not isinstance(body.get("new_text"), str):
                        raise ValueError("want new_text")
                except (json.JSONDecodeError, ValueError) as exc:
                    self._json(400, {"error": f"bad body: {exc}"})
                    return
                if url.path.endswith("/edit"):
                    code, resp = syncer.edit_item(board, text, body["new_text"])
                elif url.path.endswith("/remove"):
                    code, resp = syncer.remove_item(board, text)
                else:
                    code, resp = syncer.reopen_item(board, text)
                self._json(code, resp)
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
    # Watch for a human at the boards continuously, not once per scan: the
    # question "was anybody in shot while that photo was taken" cannot be
    # answered by a check that ran before it.
    syncer.watcher.start()
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
