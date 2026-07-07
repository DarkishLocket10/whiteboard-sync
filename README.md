# whiteboard-sync

**Write a task on the office whiteboard → it appears in Todoist.
Erase it (or tick its checkbox) → the Todoist task completes.**

No app, no photo ritual, no "remember to transcribe the board later." A
Kinect that's already pointed at the boards photographs them on a schedule, a
local vision model reads the handwriting, and the difference between what the
board says and what Todoist knows becomes task creations and completions.
Everything runs on your own hardware — no cloud vision API, no images leaving
the house, zero marginal cost per scan.

There are two boards side by side: the **left board is work**, the **right
board is personal**. Tasks created from them land in Todoist tagged
`whiteboard` + `work` / `personal`.

A built-in dashboard (below) shows exactly what the service is doing and lets
you flip every behaviour at runtime.

---

## How it works

```
kinect-knob (:8420)                         whiteboard-sync (:8430)
┌──────────────────────┐   GET /api/snapshot?frames=8&format=png
│ Kinect v2, 1080p     │ ─────────────────────────────►  ┌────────────────┐
│ stacks 8 consecutive │        one denoised photo       │ crop the board │
│ frames into one      │                                 │ region         │
│ "proper photo"       │                                 └───────┬────────┘
└──────────────────────┘                                         ▼
                                            changed since last processed scan?
                                                  │ no → done (cheap, no model)
                                                  ▼ yes
                                     local qwen3-vl via Ollama transcribes
                                     every OPEN item on each board
                                                  │ {"work": [...], "personal": [...]}
                                                  ▼
                                     fuzzy diff vs data/state.json
                                      │ new item        → todoist.new_task
                                      │ checkbox ticked → todo.update_item (completed)
                                      │ missing twice   → todo.update_item (completed)
                                      ▼
                                        Home Assistant ──► Todoist
```

Stage by stage:

1. **Photograph.** whiteboard-sync asks kinect-knob for a snapshot. Rather
   than grabbing one frame off the live stream, kinect-knob averages **8
   consecutive frames** into a single photo — the scene is static, so
   temporal stacking cuts sensor noise like a long exposure (the Kinect v2's
   color feed offers no exposure control, so stacking is the strongest
   quality lever available). The photo arrives as lossless PNG so the
   handwriting is never double-JPEG-compressed.

2. **Crop and compare.** The configured board region (`WB_REGION`) is cropped
   out, lightly enhanced (autocontrast + unsharp mask), and compared
   pixel-wise against the last *processed* photo. If the mean difference is
   under the threshold, nothing changed — the scan ends before the vision
   model is ever invoked.

3. **Read.** The crop goes to a local Ollama vision model
   (`qwen3-vl:8b-instruct` by default) with a JSON-schema-constrained prompt
   that sorts every readable item into **open** (checkbox empty) or **done**
   (checkbox ticked/X'd/filled, or text struck through) per board — including
   items on a partially blocked board — and flags **per board** whether
   anything obstructs it.

4. **Reconcile.** OCR of handwriting wobbles between scans, so matching is
   fuzzy (`difflib` ratio ≥ 0.8) and the *first* transcription of an item is
   kept as canonical — it must stay byte-identical to the Todoist task
   summary, or the completion call couldn't find its task later. New open
   items fire `todoist.new_task`. Completions have two signals:
   - **A ticked checkbox** is a positive, directly-readable signal: the task
     completes on the next scan (tunable). You don't have to erase anything —
     the ticked item stays tracked as "done" while it remains on the board,
     so a later misread of its tick can never re-create the task. Items that
     *first* appear already ticked never become tasks at all.
   - **Absence** (erased) completes after **two consecutive misses** (one
     bad read is forgiven), and only on an unobstructed board.

5. **Guard rails.**
   - If a person or chair blocks part of a board, that board's visible new
     items still sync, but nothing on it is counted missing — items hidden
     behind you must not be "completed." The other board reconciles
     normally, so sitting in front of one board never freezes the other.
   - If a Home Assistant call fails, the change-detection baseline is *not*
     advanced, so the next scan re-reads the board and retries the push.
     Nothing is silently lost.
   - Scans only run while you're **away** (per a Home Assistant presence
     entity) — which is also when nobody is sitting in front of the boards.

## The dashboard

Open **`http://<host>:8430/`**.

- **Status at a glance** — watching / paused (you're home) / scanning /
  stalled, with a countdown to the next scan or presence check.
- **Tracked items per board**, with miss-counters on items that look erased.
- **The latest board photo** — exactly what the vision model saw, stacking
  and enhancement included.
- **Change score per scan** charted against the threshold — this is the data
  you tune `WB_CHANGE_THRESHOLD` with.
- **Full history** of every scan, skip, and settings change (persisted, so it
  survives restarts), with per-scan diff scores and durations.
- **Controls** — every toggle below, applied live, no restart.
- **Scan now / Dry run** buttons. Dry run reports what *would* be added or
  completed without touching Todoist.

## What you need

| Piece | Why |
|---|---|
| [kinect-knob](https://github.com/DarkishLocket10/kinect-knob) on the same LAN | Owns the Kinect and serves `/api/snapshot` (libfreenect2 is single-process — this service never opens the camera itself) |
| [Ollama](https://ollama.com) with a vision model | The handwriting OCR. `ollama pull qwen3-vl:8b-instruct` (~6 GB VRAM) |
| Home Assistant with the [Todoist integration](https://www.home-assistant.io/integrations/todoist/) | Provides `todoist.new_task`, `todo.update_item`, and the presence entity |
| Docker + compose | Everything runs in one small container |

## Quick start

```bash
git clone git@github.com:DarkishLocket10/whiteboard-sync.git
cd whiteboard-sync
cp .env.example .env        # fill in HA_TOKEN; check the URLs
```

Find your crop region once — grab a snapshot and measure the box that
contains both boards (x1,y1,x2,y2 in the 1920×1080 frame):

```bash
curl "http://<kinect-host>:8420/api/snapshot" -o frame.jpg
# open frame.jpg, note the pixel box, put it in WB_REGION in .env
```

Then:

```bash
docker compose up -d --build
curl localhost:8430/healthz
curl -X POST 'localhost:8430/scan?force=1&dry=1'   # end-to-end test, touches nothing
```

Open `http://<host>:8430/`, check the photo card shows your boards cleanly,
and you're done — write something on a board and force a scan.

## Configuration

Environment variables (in `.env`) set the **defaults**:

| Variable | Default | Meaning |
|---|---|---|
| `HA_URL` / `HA_TOKEN` | — | Home Assistant base URL + long-lived token |
| `OLLAMA_URL` | `http://…:11434` | Ollama server |
| `WB_MODEL` | `qwen3-vl:8b-instruct` | Vision model for transcription |
| `KINECT_URL` | — | kinect-knob base URL |
| `WB_REGION` | `600,280,1320,810` | Crop box of the boards in the snapshot |
| `WB_PRESENCE_ENTITY` | — | HA `person.*` entity that gates scanning |
| `WB_TODO_ENTITY` | `todo.inbox` | HA todo entity used for completions |
| `WB_TODOIST_PROJECT` | `Inbox` | Project new tasks land in |
| `WB_INTERVAL_S` | `900` | Seconds between scans / presence checks |
| `WB_CHANGE_THRESHOLD` | `4.0` | Mean pixel diff that counts as "changed" |
| `WB_MISSING_TO_COMPLETE` | `2` | Consecutive misses (erased) before completing |
| `WB_TICKED_TO_COMPLETE` | `1` | Consecutive ticked sightings before completing |
| `WB_SCAN_WHEN_HOME` | `false` | `true` disables the presence gate |
| `WB_DATA_DIR` | `/data` | Persistence directory (volume-mounted) |

**Runtime settings** (the dashboard's Controls card, or
`POST /api/settings`) override the defaults live and persist in
`data/settings.json`:

`enabled` · `presence_gate` · `change_detection` · `obstruction_guard` ·
`enhance` · `due_today` (new tasks land in Todoist's Today view) ·
`interval_s` · `change_threshold` · `missing_to_complete` ·
`ticked_to_complete` · `capture_frames` (1–32 stacked per photo) ·
`capture_quality` · `capture_format` (`jpeg`/`png` to the model) ·
`upscale` (1–2×)

## HTTP API (port 8430)

| Endpoint | What it does |
|---|---|
| `GET /` | The dashboard |
| `GET /healthz` | Liveness — returns **500 if the scan loop has stalled**, so Docker healthchecks catch a wedged loop, not just a dead HTTP thread |
| `GET /api/status` | Everything the dashboard shows, as JSON |
| `GET /state` | Tracked items + last scan result (compact) |
| `GET /crop.jpg` | The last photo exactly as the model saw it |
| `POST /scan` | Scan now. `?force=1` bypasses change detection, `?dry=1` reports without touching Todoist |
| `POST /api/settings` | Apply a JSON object of runtime settings |

## Data directory (`./data`, volume-mounted)

| File | Contents | Safe to delete? |
|---|---|---|
| `state.json` | Tracked items (canonical text, board, miss count) | **No** while tasks are open — texts must match Todoist summaries |
| `baseline.npy` | Change-detection reference frame | Yes (forces one full re-read) |
| `history.jsonl` | Scan/skip/settings history for the dashboard | Yes (dashboard loses history) |
| `settings.json` | Runtime settings overrides | Yes (reverts to env defaults) |
| `last_crop.jpg` | Latest processed crop | Yes |

## Troubleshooting

- **History says "board blocked — completions deferred."** Someone (you?)
  is between the camera and that board, or a chair drifted into frame. Check
  the photo card — the model is probably right. Visible items still sync;
  only completions on the blocked board wait until the view clears. If it
  never clears, something is parked in front of the board.
- **Nothing has scanned for hours.** Look at the status pill — you're
  probably home, and scans are presence-gated. The history shows each skip.
  Toggle "Only scan while away" off if you want scans regardless.
- **Snapshot failures in history.** kinect-knob restarts itself to recover
  the Kinect from USB stalls; a scan that lands mid-restart fails and retries
  next interval. Persistent failures → check `KINECT_URL` and kinect-knob.
- **Model reads garbage / empty.** Is the model pulled
  (`docker exec ollama ollama list`)? Handwriting too small in the crop? Try
  `capture_frames: 16` and `upscale: 2x` from the Controls card and inspect
  `/crop.jpg` — if *you* can't read it, the model can't either.
- **A task was renamed in Todoist and now won't complete.** Expected: the
  canonical text must match the Todoist summary. Complete it by hand;
  everything else is unaffected.

## Development

```bash
# run the tests inside the app image (no local Python needed)
docker run --rm -v "$PWD:/w" -w /w whiteboard-sync:local \
  sh -c "pip install -q pytest && python -m pytest -q"

# deploy after a change
docker compose up -d --build     # rebuilds in seconds
```

The whole service is one file (`wbsync.py`) on purpose — stdlib HTTP server,
requests, Pillow, numpy, and an embedded dashboard. The test suite covers
reconciliation, the guard rails, the settings layer, and the scan loop's
failure modes with fakes — no camera, Ollama, or Home Assistant needed.

## License

MIT — see [LICENSE](LICENSE).
