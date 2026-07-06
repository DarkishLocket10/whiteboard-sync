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

## Endpoints (port 8430)

- `GET /healthz` — liveness
- `GET /state` — tracked items + last scan result
- `POST /scan` (`?force=1` to bypass change detection) — scan now
