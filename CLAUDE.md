# whiteboard-sync — agent notes (Unraid deployment copy)

This checkout at `/mnt/user/appdata/whiteboard-sync` IS the live deployment;
the `whiteboard-sync` container builds from it via docker compose.

Reads the two office whiteboards through **kinect-knob's** camera
(`GET :8420/api/snapshot`, full-res unmirrored 1080p) every 15 min while
the user is away, transcribes open items with a local qwen3-vl model via Ollama, and syncs
them to Todoist via Home Assistant. Left board = work, right = personal;
new items → `todoist.new_task` (labels `whiteboard` + board); erased or
checked items → `todo.update_item` completed after 2 consecutive misses.

## The change → deploy loop

1. Edit; keep tests green:
   `docker run --rm -v $PWD:/w -w /w whiteboard-sync:local sh -c "pip install -q pytest && python -m pytest -q"`
2. Commit and push.
3. `docker compose up -d --build` (builds in seconds).
4. Verify: `curl localhost:8430/healthz`, `curl localhost:8430/state`,
   force a scan with `curl -X POST 'localhost:8430/scan?force=1'`
   (`&dry=1` to preview without touching Todoist), or open the dashboard
   at `http://<host>:8430/`.

## Sharp edges

- `.env` is gitignored and holds the HA token. Scans need the Ollama model
  pulled: `docker exec ollama ollama pull qwen3-vl:8b-instruct`.
- `data/state.json` item texts are canonical — they must stay byte-identical
  to the Todoist task summaries or completions can't find their task. Don't
  "clean up" state by hand while tasks are open.
- The change-detection baseline (`data/baseline.npy`) only updates after a
  *processed* read whose HA pushes ALL succeeded, so a failed model call or
  a failed Todoist push retries on the next interval.
- Dashboard state lives in `data/history.jsonl` (scan/skip records, trimmed
  to the last 400 on boot) and `data/last_crop.jpg`. Safe to delete; the
  dashboard just loses its history.
- kinect-knob owns the camera (libfreenect2 is single-process); this service
  must never try to open the Kinect itself — always go through the snapshot
  endpoint. If snapshots 404, kinect-knob was probably restarted seconds ago
  (USB-stall recovery) — the next interval will succeed.
