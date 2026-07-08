"""Benchmark tick-detection prompt/schema variants against ground truth.

Run before changing READ_PROMPT/READ_SCHEMA — never tune those blind:

    mkdir -p /tmp/bench && cp data/last_crop.jpg /tmp/bench/kinect_crop.jpg
    docker run --rm --network host -v /tmp/bench:/t -v $PWD/tools:/tools \
        whiteboard-sync:local python3 /tools/bench_read.py http://192.168.2.229:11434 1

GT below is the boards as photographed on 2026-07-08 (checkbox truth
verified by zooming the checkbox columns). If the boards have changed,
re-establish ground truth from a fresh photo FIRST, then update GT.
Score to beat: V9 (deployed since 2026-07-08 evening: per-item enum with
NAMED mark shapes empty/slash/check_x_or_scribble) = 77/84 over four clear
frames, zero false positives. V1 (generic empty/marked) = 72/84. V0 (the
original two-list schema) = 10/21 on one frame: all 11 ticks missed.
False "marked" positives are the dangerous direction (they complete live
tasks) and must stay at ZERO; false negatives only delay completion until
the item is erased. Variants that flipped errors to false positives — do
not retry: free-text "describe the mark" field; "struck_text" as its own
checkbox enum value; 3x upscale. Obstructed frames flip checkbox reads
randomly (that is why reconcile distrusts ticks on protected boards) —
bench on CLEAR frames only.
"""
import base64, difflib, io, json, sys, time
import requests
from PIL import Image

OLLAMA = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.2.229:11434"
MODEL = "qwen3-vl:8b-instruct"

GT = {
    "work": [
        ("REP dashboard changes", True),
        ("Admin area changes", True),
        ("Pulse leaderboard update", False),
        ("Leah - Feedback from Christina", False),
        ("Leah - BlindSpot Visual in dashboard", False),
        ("Ash - data dashboard for AI RP", True),
        ("Shafi - Fix Plays giving cursing", True),
        ("Shafi - Plays demo", True),
        ("Ticket updates", True),
        ("Review Plays", True),
        ("Delete buttons", True),
    ],
    "personal": [
        ("Recycle boxes", False),
        ("do laundry", False),
        ("LCBO", True),
        ("Bluetooth Adapter", True),
        ("Finish bedroom", False),
        ("Sew Pillowcase", False),
        ("Try out new Vacuum", False),
        ("Finish BMO worksheets", False),
        ("Inform dad to call Rogers", False),
        ("Send amazon Package back", True),
    ],
}

V0_PROMPT = """\
This photo shows two whiteboards side by side.
The LEFT board holds WORK to-do items. The RIGHT board holds PERSONAL to-do items.
Every to-do item has a small square checkbox drawn beside it.
Sort every readable to-do item into one of two lists per board:
- OPEN ("work" / "personal"): the checkbox is empty and the text is not
  crossed out.
- DONE ("work_done" / "personal_done"): the checkbox contains a check mark,
  an X, or a scribble filling it, OR the item text is struck through.
Skip headings, name tags, stickers, photos, and anything that is not a
to-do list item. Erased items appear in neither list.
A person or object may hide part of a board. Still transcribe every item you
can actually read — NEVER return empty lists just because a board is
partially hidden. Read around the blockage.
After transcribing, set "work_obstructed" to true if a person, chair, or any
object hides ANY part of the LEFT board, and "personal_obstructed" to true if
anything hides ANY part of the RIGHT board — otherwise false.
Write each item as a short task phrase without bullet or checkbox characters,
fixing obvious handwriting artifacts."""

V0_SCHEMA = {
    "type": "object",
    "properties": {
        "work": {"type": "array", "items": {"type": "string"}},
        "work_done": {"type": "array", "items": {"type": "string"}},
        "personal": {"type": "array", "items": {"type": "string"}},
        "personal_done": {"type": "array", "items": {"type": "string"}},
        "work_obstructed": {"type": "boolean"},
        "personal_obstructed": {"type": "boolean"},
    },
    "required": ["work", "work_done", "personal", "personal_done",
                 "work_obstructed", "personal_obstructed"],
    "additionalProperties": False,
}

ITEM = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "checkbox": {"type": "string", "enum": ["empty", "marked"]},
    },
    "required": ["text", "checkbox"],
    "additionalProperties": False,
}
V1_SCHEMA = {
    "type": "object",
    "properties": {
        "work": {"type": "array", "items": ITEM},
        "personal": {"type": "array", "items": ITEM},
        "work_obstructed": {"type": "boolean"},
        "personal_obstructed": {"type": "boolean"},
    },
    "required": ["work", "personal", "work_obstructed", "personal_obstructed"],
    "additionalProperties": False,
}

V1_PROMPT = """\
This photo shows two whiteboards side by side.
The LEFT board holds WORK to-do items. The RIGHT board holds PERSONAL to-do items.
Every to-do item has a small hand-drawn square checkbox at the START of its line.

Transcribe every readable to-do item into the "work" and "personal" arrays,
one object per item, top to bottom. Write "text" as a short task phrase
without bullet or checkbox characters, fixing obvious handwriting artifacts.

Then, for EACH item separately, look closely at the small square at the start
of ITS line and set "checkbox":
- "empty": the inside of the square is blank white.
- "marked": ANY pen stroke touches the inside of the square — a check mark,
  a tick, an X, a diagonal slash, or a scribble. A hasty tick often looks
  like one short diagonal line crossing the box. Also use "marked" if the
  item's text is struck through.
Decide each checkbox on its own evidence; most boards contain a mix of
empty and marked boxes.

Skip headings, name tags, stickers, photos, and anything that is not a
to-do list item. Erased items are not listed.
A person or object may hide part of a board. Still transcribe every item you
can actually read — NEVER return empty arrays just because a board is
partially hidden. Read around the blockage.
After transcribing, set "work_obstructed" to true if a person, chair, or any
object hides ANY part of the LEFT board, and "personal_obstructed" to true if
anything hides ANY part of the RIGHT board — otherwise false."""

BOARD_ITEM = ITEM
V2_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": BOARD_ITEM},
        "obstructed": {"type": "boolean"},
    },
    "required": ["items", "obstructed"],
    "additionalProperties": False,
}
V2_PROMPT = """\
This photo shows ONE whiteboard holding a to-do list.
Every to-do item has a small hand-drawn square checkbox at the START of its line.

Transcribe every readable to-do item into "items", one object per item, top
to bottom. Write "text" as a short task phrase without bullet or checkbox
characters, fixing obvious handwriting artifacts.

Then, for EACH item separately, look closely at the small square at the start
of ITS line and set "checkbox":
- "empty": the inside of the square is blank white.
- "marked": ANY pen stroke touches the inside of the square — a check mark,
  a tick, an X, a diagonal slash, or a scribble. A hasty tick often looks
  like one short diagonal line crossing the box. Also use "marked" if the
  item's text is struck through.
Decide each checkbox on its own evidence; most boards contain a mix of
empty and marked boxes.

Skip headings, name tags, stickers, photos, and anything that is not a
to-do list item. Erased items are not listed.
A person or object may hide part of the board. Still transcribe every item
you can actually read — NEVER return an empty array just because the board
is partially hidden. Read around the blockage.
Set "obstructed" to true if a person, chair, or any object hides ANY part of
this board — otherwise false."""


# V9 == the deployed prompt/schema (keep in sync with wbsync.READ_PROMPT).
ITEM9 = {"type": "object", "properties": {
    "text": {"type": "string"},
    "checkbox": {"type": "string",
                 "enum": ["empty", "slash", "check_x_or_scribble"]}},
    "required": ["text", "checkbox"], "additionalProperties": False}
V9_SCHEMA = {"type": "object", "properties": {
    "work": {"type": "array", "items": ITEM9},
    "personal": {"type": "array", "items": ITEM9},
    "work_obstructed": {"type": "boolean"},
    "personal_obstructed": {"type": "boolean"}},
    "required": ["work", "personal", "work_obstructed", "personal_obstructed"],
    "additionalProperties": False}
V9_PROMPT = V1_PROMPT.replace(
    """- "empty": the inside of the square is blank white.
- "marked": ANY pen stroke touches the inside of the square — a check mark,
  a tick, an X, a diagonal slash, or a scribble. A hasty tick often looks
  like one short diagonal line crossing the box. Also use "marked" if the
  item's text is struck through.
Decide each checkbox on its own evidence; most boards contain a mix of
empty and marked boxes.""",
    """- "empty": the inside of the square is blank white — no stroke at all.
- "slash": one thin diagonal or straight line crosses the square (a hasty
  tick often looks like this).
- "check_x_or_scribble": a check mark, an X, or a scribble in the square;
  ALSO use this when the item's text is struck through or crossed out,
  whatever its checkbox looks like.
Decide each checkbox on its own evidence; most boards contain a mix.""")


def b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode()


def call(prompt, schema, image_b64, num_ctx=6144):
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL, "stream": False, "format": schema, "think": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    }, timeout=600)
    r.raise_for_status()
    out = json.loads(r.json()["message"]["content"])
    return out, time.time() - t0


def score(name, got):  # got: {"work": [(text, checked)], "personal": [...]}
    total_ok = 0
    total = 0
    lines = []
    for board in ("work", "personal"):
        gt = GT[board]
        got_items = got.get(board, [])
        got_texts = [t for t, _ in got_items]
        for text, checked in gt:
            best, ratio = None, 0.0
            for i, cand in enumerate(got_texts):
                r = difflib.SequenceMatcher(None, text.lower(), cand.lower()).ratio()
                if r > ratio:
                    best, ratio = i, r
            total += 1
            if best is None or ratio < 0.6:
                lines.append(f"    MISS  [{board}] {text!r} not transcribed")
                continue
            got_checked = got_items[best][1]
            if got_checked == checked:
                total_ok += 1
            else:
                lines.append(f"    WRONG [{board}] {text!r}: gt={'done' if checked else 'open'} got={'done' if got_checked else 'open'} (as {got_texts[best]!r})")
        extra = [t for i, (t, _) in enumerate(got_items)
                 if max((difflib.SequenceMatcher(None, t.lower(), g.lower()).ratio()
                        for g, _ in gt), default=0) < 0.5]
        for t in extra:
            lines.append(f"    EXTRA [{board}] {t!r}")
    print(f"  {name}: {total_ok}/{total} checkbox states correct")
    for line in lines:
        print(line)
    return total_ok


img = Image.open("/t/kinect_crop.jpg")
W, H = img.size

runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for n in range(runs):
    print(f"=== run {n + 1} ===")

    out, dt = call(V0_PROMPT, V0_SCHEMA, b64(img))
    got = {b: [(t, False) for t in out.get(b, [])] +
              [(t, True) for t in out.get(f"{b}_done", [])]
           for b in ("work", "personal")}
    score(f"V0 current lists      ({dt:4.1f}s)", got)

    out, dt = call(V1_PROMPT, V1_SCHEMA, b64(img))
    got = {b: [(i["text"], i["checkbox"] == "marked") for i in out.get(b, [])]
           for b in ("work", "personal")}
    score(f"V1 per-item checkbox  ({dt:4.1f}s)", got)

    out, dt = call(V9_PROMPT, V9_SCHEMA, b64(img))
    got = {b: [(i["text"], i["checkbox"] != "empty") for i in out.get(b, [])]
           for b in ("work", "personal")}
    score(f"V9 DEPLOYED mark-shapes ({dt:4.1f}s)", got)

    left = img.crop((0, 0, int(W * 0.64), H))
    right = img.crop((int(W * 0.62), 0, W, H))
    got = {}
    dts = 0.0
    for board, half in (("work", left), ("personal", right)):
        out, dt = call(V2_PROMPT, V2_SCHEMA, b64(half))
        dts += dt
        got[board] = [(i["text"], i["checkbox"] == "marked")
                      for i in out.get("items", [])]
    score(f"V2 split boards       ({dts:4.1f}s)", got)
