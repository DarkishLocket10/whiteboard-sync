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
  "clean up" state by hand while tasks are open. Items carry a `status`:
  "open" (task live in Todoist), "done" (task completed but the ticked
  item is still physically on the board — kept so tick misreads can't
  re-create the task; purged after it's erased), "removed" (manually
  removed on the dashboard: Todoist task deleted, kept like "done" so the
  writing still on the board can't re-create it), or "pending" (new text
  awaiting `sightings_to_add` consecutive reads before its task is
  created; one miss on a clear board kills it).
- The vision schema asks for a PER-ITEM `{"text", "checkbox": empty|marked}`
  decision. Do not regress to separate open/done lists: benchmarked
  2026-07-08 on a live frame, the list schema missed ALL 11 ticks (10/21
  checkbox states right); the per-item schema scored 18/21 with only safe
  false negatives. A "describe the mark first" field flipped errors to
  false positives (whole board read as ticked) — also don't do that.
- Reconcile guards, both bench-driven by real incidents from 2026-07-08
  history: new items wait for `sightings_to_add` (default 2) consecutive
  reads (a scan hallucinated "Buy cake"/"Call mom" into real Todoist tasks;
  fragments of half-hidden lines became tasks), and a mass-erase guard
  demands one extra miss when ≥4 items would absence-complete in one scan
  while the board still shows other items (17 tasks were once falsely
  completed by one garbage read). A real full-board wipe reads as an EMPTY
  board and is not slowed. Change detection is bypassed while any item is
  pending, so sightings can accrue on an unchanged board.
- Item edits belong on the dashboard, not in the JSON: the ✎/× buttons hit
  `POST /api/items/edit` (renames the Todoist task too, keeping state
  canonical) and `POST /api/items/remove` (deletes the Todoist task via
  `todo.remove_item`). Both take `{board, text}` (+`new_text` for edit),
  refuse with 409 while a scan holds the lock, and fail without touching
  state if the HA call fails.
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
