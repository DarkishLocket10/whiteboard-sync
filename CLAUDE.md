# whiteboard-sync — agent notes (Unraid deployment copy)

This checkout at `/mnt/user/appdata/whiteboard-sync` IS the live deployment;
the `whiteboard-sync` container builds from it via docker compose.

Reads the two office whiteboards through **kinect-knob's** camera
(`GET :8420/api/snapshot`, full-res unmirrored 1080p) every 15 min while
the user is away, transcribes open items with a local qwen3-vl model via Ollama, and syncs
them to Todoist via Home Assistant. Left board = work, right = personal;
new items → `todoist.new_task` (label `work` / `personal`); erased or
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
- Reconcile guards, all driven by real incidents from 2026-07-08 history:
  new items wait for `sightings_to_add` (default 2) consecutive reads (a
  scan hallucinated "Buy cake"/"Call mom" into real Todoist tasks;
  fragments of half-hidden lines became tasks); a mass-erase guard demands
  one extra miss when ≥4 items would absence-complete in one scan while
  the board still shows other items (17 tasks were once falsely completed
  by one garbage read — a real full-board wipe reads as an EMPTY board and
  is not slowed); and on an OBSTRUCTED board checkbox judgments are
  ignored in BOTH directions (empty boxes read "marked" and vice versa,
  varying frame to frame, when someone sits in front — item text is still
  trusted there: it resets absence counters and accrues pending
  sightings). Tick completions need `ticked_to_complete` (default 2)
  consecutive CLEAR-board sightings, and then a `tick_confirm` second
  photo+read in the same scan that must agree (a transiently weird scene
  — chair moved, lighting — once produced the SAME false tick on two
  reads minutes apart while the board read unobstructed; streaks alone
  can't stop correlated repeats). Disputed or unconfirmable ticks demote
  to plain sightings — deferring a completion is always safe.
- On top of all that, a tick streak may only START within
  `TICK_START_CHANGE_WINDOW` scans of a real frame change (diff ≥
  threshold): a new mark necessarily changes pixels, so a tick first
  sighted on a long-static scene is a misread. This is what finally
  stopped the static-scene failure (2026-07-09 00:39: with
  change_detection toggled off, three near-identical static frames agreed
  on the same false tick — streak AND confirm photo were defeated by
  correlation, 3 tasks falsely completed). The gate holds regardless of
  the change_detection / obstruction_guard dashboard toggles. Streaks
  already under way still finish on static frames (the mark persisting IS
  the confirmation), and it also blocks static-tick pending→done ghosts.
- Model reads are NOT deterministic even at temperature 0 on an identical
  image (marginal checkboxes flip between runs), and false-positive bursts
  are scene-state-dependent, not image-enhancement-dependent (enhance
  on/off benchmarked identical). Don't chase single-run bench results;
  run multiple frames/runs and judge false positives strictly.
- Depth guard: kinect-knob's `/api/region_depth` (self-illuminated ToF,
  color-aligned, works in pitch black) reports per-board-half depth
  percentiles; `depth_blocked()` calls a half obstructed when p10 sits
  ≥400mm in front of p90 (the board plane, ~1950mm on this setup).
  Depth-protected boards are force-protected INDEPENDENT of the
  obstruction_guard toggle — this is the reliable obstruction signal (the
  model's visual flag flaps at night). IR itself CANNOT read the boards:
  512×424 over the full FOV leaves ~150px for both boards, and the
  coloured inks are near-invisible at 860nm (verified with
  `/api/snapshot?ir=1` 2026-07-09).
- Depth presence SKIP (2026-07-13, `depth_skip` toggle, default on): a
  scheduled scan first asks `_depth_obstructed()` BEFORE taking a photo;
  if someone is physically at either board the scan is skipped outright
  (history: "someone at the … board"). Rationale: reading through a
  person only ever contributed partial-line fragments that accrued
  sightings and became real Todoist tasks — Yash sits at the work board
  all day, so those scans were pure downside. Consequence: while he's at
  the desk NOTHING scans (personal board included) until he steps away;
  that matches the original "scan while away" design. Forced scans
  (`/scan?force=1`, the dashboard button) bypass the skip and fall back
  to the old read-through-with-depth-guard behavior, as does turning
  `depth_skip` off. Depth endpoint unreachable → no evidence → scan
  proceeds (old guards still apply).
- Low light (the color camera: kinect-knob gained a native exposure shim
  2026-07-09 — kk_exposure.cpp — but it's capped per-frame and currently
  pinned to a short shutter for gesture blur, so for whiteboard photos):
  frame stacking is the synthetic long exposure. Measured 2026-07-09 ~02:50 (luma ~31 vs ~140
  lit): a single frame garbled text and read almost every box ticked; a
  32-frame stack transcribed near-perfectly but checkbox judgments stayed
  completion-grade unreliable. Even so, "near-perfect" still leaked:
  protection only deferred COMPLETIONS, so garbled fragments from dim
  stacks accrued sightings and landed in Todoist as partial text. Hence
  (2026-07-13): scheduled scans SKIP dim frames outright (`dim_skip`
  toggle, default on; threshold is the `dim_luma` setting, default 55,
  tunable on the dashboard — history: "too dim to trust"). Forced scans,
  or `dim_skip` off, use the old path: re-capture with 32 frames and
  reconcile with BOTH boards protected (regardless of the
  obstruction_guard toggle). Pitch-dark frames (< DARK_LUMA) always skip,
  forced or not — a black frame reads as an EMPTY board, which absence
  logic would complete as a full wipe.
- Marker-COLOUR detection was benchmarked 2026-07-09 and NOT shipped: a
  per-item colour enum after the checkbox field is checkbox-safe (77/84,
  0 FP unchanged), but the Kinect renders ink nearly desaturated — black
  reads 100%, red ~50%, blue ~25%, green/purple ~0% (they read
  blue/black). Boosting saturation before the model lifts colour to
  ~68/84 but WRECKS tick detection (8–11 false positives) — never apply
  saturation/colour enhancement to the image the checkbox read uses. Change detection is bypassed while any item is
  pending, so sightings can accrue on an unchanged board.
- Item edits belong on the dashboard, not in the JSON: the ✎/↩/× buttons
  hit `POST /api/items/edit` (renames the Todoist task too, keeping state
  canonical), `POST /api/items/reopen` (mark as NOT done: creates a fresh
  Todoist task — a completed task can't be un-completed through HA's todo
  entity and a removed one's task was deleted — and flips the item back
  to open; if the board's checkbox is still marked the scans will just
  complete it again, the dialog warns), and `POST /api/items/remove`
  (deletes the Todoist task via `todo.remove_item`). All take
  `{board, text}` (+`new_text` for edit), refuse with 409 while a scan
  holds the lock, and fail without touching state if the HA call fails.
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
