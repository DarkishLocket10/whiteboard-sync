# whiteboard-sync

Point a Kinect at your whiteboards; whatever you write appears in Todoist,
and whatever you erase (or tick off) gets completed. Scans happen only while
you're away — which is also when nobody is sitting in front of the boards.

```
kinect-knob (/api/snapshot, 1080p) ──► crop boards ──► changed since last scan?
                                                        │ no → done (no API cost)
                                                        ▼ yes
                                            local qwen3-vl (Ollama) transcription
                                                        │ {"work": [...], "personal": [...]}
                                                        ▼
                       fuzzy diff vs data/state.json (handwriting OCR wobbles)
                        │ new item → todoist.new_task (labels: whiteboard, work|personal)
                        │ missing 2 scans → todo.update_item status=completed
                        ▼
                              Home Assistant → Todoist
```

## Setup

1. `cp .env.example .env` and fill in `HA_TOKEN`.
2. Adjust `WB_REGION` (crop of the boards in the 1920x1080 snapshot) — grab
   `curl kinect:8420/api/snapshot -o f.jpg` and measure once.
3. `docker compose up -d --build`
4. Test: `curl -X POST 'localhost:8430/scan?force=1'` and check Todoist.

## Dashboard & endpoints (port 8430)

Open `http://<host>:8430/` for the live dashboard: sync state and presence
gate, tracked items per board, the latest camera crop, change scores vs the
threshold (useful for tuning `WB_CHANGE_THRESHOLD`), and a history of every
scan and skip. Buttons trigger a forced or dry-run scan.

- `GET /` — dashboard
- `GET /healthz` — liveness (returns 500 once the scan loop has stalled)
- `GET /state` — tracked items + last scan result
- `GET /api/status` — everything the dashboard shows, as JSON
- `GET /crop.jpg` — most recent cropped board photo
- `POST /scan` (`?force=1` bypass change detection, `?dry=1` report only,
  touch nothing) — scan now
- `POST /api/settings` — runtime toggles/tuning as a JSON object; the
  dashboard's Controls card uses this. Settings: `enabled`, `presence_gate`,
  `change_detection`, `obstruction_guard`, `enhance`, `interval_s`,
  `change_threshold`, `missing_to_complete`, `capture_frames` (multi-frame
  stacking via kinect-knob), `capture_quality`, `capture_format`, `upscale`.

Scan history persists in `data/history.jsonl`, runtime settings in
`data/settings.json`, the latest crop in `data/last_crop.jpg`.
