"""Reconciliation and helpers — the logic that decides what hits Todoist."""
import json

import pytest

from wbsync import (Config, Syncer, confirm_ticks, fuzzy_match,
                    normalise_read, parse_region)


class FakeHA:
    def __init__(self):
        self.calls = []
        self.ok = True            # flip to False to simulate HA being down
        self.presence = "not_home"

    def call(self, domain, service, data):
        self.calls.append((domain, service, data))
        return self.ok

    def state(self, entity):
        return self.presence


class FixedReader:
    def __init__(self, seen):
        self.seen = seen

    def read(self, jpeg):
        return self.seen


def fake_fetch(syncer):
    import numpy as np
    syncer._fetch_crop = lambda: (b"jpeg", np.zeros((5, 5), dtype=np.float32))


def make_syncer(tmp_path, missing_to_complete=2, sightings_to_add=1):
    cfg = Config(
        ha_url="http://ha", ha_token="t", ollama_url="http://ollama",
        kinect_url="http://kk", region=(0, 0, 10, 10),
        presence_entity="person.x", todo_entity="todo.inbox",
        todoist_project="Inbox", interval_s=900, change_threshold=4.0,
        missing_to_complete=missing_to_complete, scan_when_home=False,
        model="qwen3-vl:8b-instruct", data_dir=tmp_path,
    )
    s = Syncer(cfg, FakeHA(), reader=None)
    # Most tests predate the sighting streak and exercise immediate adds;
    # in-memory only, so reloads still see the persisted/default value.
    s.settings["sightings_to_add"] = sightings_to_add
    return s


def test_parse_region():
    assert parse_region("600,280,1320,810") == (600, 280, 1320, 810)
    with pytest.raises(ValueError):
        parse_region("10,10,5,5")


def test_fuzzy_match_tolerates_ocr_wobble():
    known = ["Bluetooth Adapter", "do laundry"]
    assert fuzzy_match("bluetooth adaptor", known) == "Bluetooth Adapter"
    assert fuzzy_match("completely different", known) is None


def test_new_items_create_tagged_todoist_tasks(tmp_path):
    s = make_syncer(tmp_path)
    result = s.reconcile({"work": ["Fix dashboard"], "personal": ["do laundry"]})
    assert sorted(result["added"]) == ["Fix dashboard", "do laundry"]
    tasks = [c for c in s.ha.calls if c[:2] == ("todoist", "new_task")]
    assert {t[2]["content"]: t[2]["labels"] for t in tasks} == {
        "Fix dashboard": ["whiteboard", "work"],
        "do laundry": ["whiteboard", "personal"],
    }


def test_rescan_with_wobbly_ocr_adds_nothing(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Bluetooth Adapter"], "personal": []})
    s.ha.calls.clear()
    result = s.reconcile({"work": ["bluetooth adaptor"], "personal": []})  # OCR wobble
    assert result["added"] == [] and result["completed"] == []
    assert s.ha.calls == []
    # canonical text preserved — it must keep matching the Todoist summary
    assert s.state["items"][0]["text"] == "Bluetooth Adapter"


def test_erased_item_completes_after_two_misses(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Fix dashboard"], "personal": []})
    s.ha.calls.clear()

    first = s.reconcile({"work": [], "personal": []})   # miss 1: guarded
    assert first["completed"] == [] and s.ha.calls == []

    second = s.reconcile({"work": [], "personal": []})  # miss 2: complete
    assert second["completed"] == ["Fix dashboard"]
    assert s.ha.calls == [("todo", "update_item", {
        "entity_id": "todo.inbox", "item": "Fix dashboard", "status": "completed",
    })]
    assert s.state["items"] == []


def test_reappearing_item_resets_the_miss_counter(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Fix dashboard"], "personal": []})
    s.reconcile({"work": [], "personal": []})               # miss 1
    s.reconcile({"work": ["Fix dashboard"], "personal": []})  # seen again
    assert s.state["items"][0]["missing"] == 0
    s.ha.calls.clear()
    s.reconcile({"work": [], "personal": []})               # miss 1 again — no completion
    assert s.ha.calls == []


def test_state_persists_across_restarts(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Fix dashboard"], "personal": []})
    reloaded = make_syncer(tmp_path)
    assert reloaded.state["items"] == [{"text": "Fix dashboard", "board": "work",
                                        "status": "open", "missing": 0, "ticked": 0}]


def test_legacy_state_migrates_to_open_status(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps(
        {"items": [{"text": "old", "board": "work", "missing": 1}]}))
    s = make_syncer(tmp_path)
    assert s.state["items"][0]["status"] == "open"
    assert s.state["items"][0]["ticked"] == 0
    assert s.state["items"][0]["missing"] == 1  # counter survives the migration


def test_obstructed_board_defers_completions_but_still_adds(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Fix dashboard"], "personal": ["do laundry"]})
    s.ha.calls.clear()

    # A person blocks the WORK board; one new item is still readable there,
    # and the PERSONAL board is clear with its item erased.
    s.reader = FixedReader({"work": ["New visible item"], "personal": [],
                            "work_obstructed": True, "personal_obstructed": False})
    fake_fetch(s)
    result = s.scan(force=True)

    assert result["obstructed"] == ["work"]
    assert result["protected"] == ["work"]
    assert result["added"] == ["New visible item"]   # visible adds still land
    by_text = {i["text"]: i for i in s.state["items"]}
    assert by_text["Fix dashboard"]["missing"] == 0  # hidden: NOT counted missing
    assert by_text["do laundry"]["missing"] == 1     # clear board reconciles

    result = s.scan(force=True)                      # second miss on clear board
    assert result["completed"] == ["do laundry"]
    assert by_text["Fix dashboard"]["missing"] == 0  # still protected


def test_dry_run_touches_nothing(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["existing"], "personal": []})
    s.reconcile({"work": [], "personal": []})  # miss 1 for "existing"
    s.ha.calls.clear()
    preview = s.reconcile({"work": ["brand new"], "personal": []}, apply=False)
    assert preview["added"] == ["brand new"]
    assert preview["completed"] == ["existing"]     # would complete on next real scan
    assert s.ha.calls == []                          # nothing pushed
    assert len(s.state["items"]) == 1                # state unchanged
    assert s.state["items"][0]["missing"] == 1       # counter unchanged


def test_failed_ha_push_keeps_baseline_so_next_scan_retries(tmp_path):
    s = make_syncer(tmp_path)
    s.reader = FixedReader({"obstructed": False, "work": ["new item"], "personal": []})
    fake_fetch(s)
    s.ha.ok = False  # HA down: the push fails
    result = s.scan(force=True)
    assert result["failed"] == 1 and result["added"] == []
    # baseline untouched -> the change re-detects and the push retries
    assert not (tmp_path / "baseline.npy").exists()
    s.ha.ok = True
    result = s.scan(force=True)
    assert result["added"] == ["new item"]
    assert (tmp_path / "baseline.npy").exists()


def test_scan_history_is_recorded_and_survives_restart(tmp_path):
    s = make_syncer(tmp_path)
    s.reader = FixedReader({"obstructed": False, "work": ["task"], "personal": []})
    fake_fetch(s)
    s.scan(force=True)
    assert s.history[-1]["type"] == "scan"
    assert s.history[-1]["added"] == ["task"]
    assert "seen" not in s.history[-1]  # raw transcriptions stay out of history
    reloaded = make_syncer(tmp_path)
    assert reloaded.history[-1]["added"] == ["task"]


def test_tick_skips_and_records_when_home(tmp_path):
    s = make_syncer(tmp_path)  # reader=None: an actual scan attempt would error
    s.ha.presence = "home"
    s.tick()
    assert s.counters["skips_home"] == 1
    assert s.history[-1]["type"] == "skip" and s.history[-1]["reason"] == "home"
    assert s.last_scan_t > 0  # re-check deferred one interval


def test_scan_survives_unexpected_errors(tmp_path):
    s = make_syncer(tmp_path)

    def boom():
        raise ValueError("boom")

    s._fetch_crop = boom
    result = s.scan(force=True)
    assert result["ok"] is False and "boom" in result["error"]
    assert s.counters["errors"] == 1


def test_settings_validate_persist_and_reload(tmp_path):
    s = make_syncer(tmp_path)
    out = s.update_settings({"interval_s": 300, "capture_frames": 999,
                             "capture_format": "png", "bogus": 1})
    assert out["applied"] == {"interval_s": 300, "capture_format": "png"}
    assert "out of range" in out["rejected"]["capture_frames"]
    assert out["rejected"]["bogus"] == "unknown setting"
    reloaded = make_syncer(tmp_path)
    assert reloaded.settings["interval_s"] == 300
    assert reloaded.settings["capture_format"] == "png"
    assert s.history[-1]["type"] == "settings"


def test_presence_gate_toggle_scans_while_home(tmp_path):
    s = make_syncer(tmp_path)
    s.reader = FixedReader({"obstructed": False, "work": [], "personal": []})
    fake_fetch(s)
    s.ha.presence = "home"
    s.update_settings({"presence_gate": False})
    s.tick()
    assert s.counters["scans"] == 1 and s.counters["skips_home"] == 0


def test_disabled_toggle_skips_all_scans(tmp_path):
    s = make_syncer(tmp_path)  # away per default FakeHA, but disabled wins
    s.update_settings({"enabled": False})
    s.tick()
    assert s.counters["scans"] == 0 and s.counters["skips_off"] == 1
    assert s.history[-1] == {"type": "skip", "reason": "disabled",
                             "at": s.history[-1]["at"], "t": s.history[-1]["t"]}


def test_obstruction_guard_toggle(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["existing"], "personal": []})
    s.reader = FixedReader({"work": [], "personal": [],
                            "work_obstructed": True, "personal_obstructed": False})
    fake_fetch(s)
    s.ha.calls.clear()
    s.update_settings({"missing_to_complete": 1})

    result = s.scan(force=True)             # guard on: work board protected
    assert result["obstructed"] == ["work"] and result["completed"] == []
    assert s.ha.calls == []

    s.update_settings({"obstruction_guard": False})
    result = s.scan(force=True)             # guard off: flags ignored entirely
    assert result["protected"] == []
    assert result["completed"] == ["existing"]


def test_change_detection_toggle_forces_reads(tmp_path):
    import numpy as np

    s = make_syncer(tmp_path)
    s.reader = FixedReader({"obstructed": False, "work": [], "personal": []})
    fake_fetch(s)
    s.scan(force=True)                       # establishes the baseline
    result = s.scan()                        # identical frame: skipped
    assert result["changed"] is False
    s.update_settings({"change_detection": False})
    result = s.scan()                        # toggle off: always read
    assert result["changed"] is True


def test_boards_are_reconciled_independently(tmp_path):
    s = make_syncer(tmp_path)
    # Same text on both boards must be two distinct items
    s.reconcile({"work": ["call mom"], "personal": ["call mom"]})
    assert len(s.state["items"]) == 2
    s.ha.calls.clear()
    # Erasing it from ONE board must not touch the other's counter
    s.reconcile({"work": [], "personal": ["call mom"]})
    s.reconcile({"work": [], "personal": ["call mom"]})
    completed = [c for c in s.ha.calls if c[1] == "update_item"]
    assert len(completed) == 1
    assert s.state["items"] == [{"text": "call mom", "board": "personal",
                                 "status": "open", "missing": 0, "ticked": 0}]


def test_new_tasks_get_due_today_unless_disabled(tmp_path):
    s = make_syncer(tmp_path)
    # NB: the two task names must be fuzzy-DISSIMILAR or the second read
    # matches the first item and never creates a task
    s.reconcile({"work": ["buy milk"], "personal": []})
    assert s.ha.calls[-1][2]["due_date_string"] == "today"
    s.update_settings({"due_today": False})
    s.reconcile({"work": ["call the plumber"], "personal": []})
    assert s.ha.calls[-1][2]["content"] == "call the plumber"
    assert "due_date_string" not in s.ha.calls[-1][2]


def test_ticked_item_completes_and_stays_tracked_as_done(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Fix dashboard"], "personal": []})
    s.ha.calls.clear()

    result = s.reconcile({"work": [], "work_done": ["Fix dashboard"], "personal": []})
    assert result["completed"] == ["Fix dashboard"]  # one tick sighting is enough
    assert s.ha.calls == [("todo", "update_item", {
        "entity_id": "todo.inbox", "item": "Fix dashboard", "status": "completed",
    })]
    assert s.state["items"][0]["status"] == "done"   # kept while on the board

    # Still ticked next scan: nothing fires again.
    s.ha.calls.clear()
    result = s.reconcile({"work": [], "work_done": ["Fix dashboard"], "personal": []})
    assert result["completed"] == [] and s.ha.calls == []

    # Tick MISREAD as open later: must not re-create the task.
    result = s.reconcile({"work": ["Fix dashboard"], "personal": []})
    assert result["added"] == [] and s.ha.calls == []
    assert s.state["items"][0]["status"] == "done"


def test_done_item_purges_after_erase_without_ha_call(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Fix dashboard"], "personal": []})
    s.reconcile({"work": [], "work_done": ["Fix dashboard"], "personal": []})
    s.ha.calls.clear()
    s.reconcile({"work": [], "personal": []})        # erased: miss 1, still tracked
    assert len(s.state["items"]) == 1
    s.reconcile({"work": [], "personal": []})        # miss 2: purged silently
    assert s.state["items"] == [] and s.ha.calls == []


def test_item_first_seen_ticked_never_creates_a_task(tmp_path):
    s = make_syncer(tmp_path)
    result = s.reconcile({"work": [], "work_done": ["old finished thing"], "personal": []})
    assert result["added"] == [] and s.ha.calls == []
    assert s.state["items"][0]["status"] == "done"
    # ...even when its tick is later misread as an open item
    result = s.reconcile({"work": ["old finished thing"], "personal": []})
    assert result["added"] == [] and s.ha.calls == []


def test_ticked_to_complete_guard(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"ticked_to_complete": 2})
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    r1 = s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    assert r1["completed"] == [] and s.ha.calls == []
    assert s.state["items"][0]["ticked"] == 1
    r2 = s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    assert r2["completed"] == ["task"]
    # a wobbly tick that reads open in between resets the streak
    s.reconcile({"work": ["another"], "personal": []})
    s.ha.calls.clear()
    s.reconcile({"work": [], "work_done": ["another"], "personal": []})
    s.reconcile({"work": ["another"], "personal": []})
    assert s.state["items"][-1]["ticked"] == 0


def test_tick_on_obstructed_board_is_only_a_sighting(tmp_path):
    """Checkbox reads flip randomly on obstructed frames (observed live
    2026-07-08): a tick there must not complete, and must not create ghosts."""
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    result = s.reconcile({"work": [], "work_done": ["task"], "personal": [],
                          "work_obstructed": True, "personal_obstructed": False})
    assert result["protected"] == ["work"] and result["completed"] == []
    assert s.ha.calls == []
    item = s.state["items"][0]
    assert item["status"] == "open"
    assert item["missing"] == 0        # the TEXT sighting still counts
    assert item["ticked"] == 0         # ...but no tick streak accrues
    # the same tick on a CLEAR frame completes normally
    result = s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    assert result["completed"] == ["task"]


def test_obstructed_frames_freeze_tick_streak_both_ways(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"ticked_to_complete": 2})
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    s.reconcile({"work": [], "work_done": ["task"], "personal": []})  # clear tick 1
    assert s.state["items"][0]["ticked"] == 1
    # obstructed frame misreads it as open: streak must NOT reset
    s.reconcile({"work": ["task"], "personal": [],
                 "work_obstructed": True, "personal_obstructed": False})
    assert s.state["items"][0]["ticked"] == 1
    # next clear tick completes (2 trusted sightings total)
    result = s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    assert result["completed"] == ["task"]


def test_first_seen_ticked_on_obstructed_board_creates_no_ghost(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": [], "work_done": ["brand new"], "personal": [],
                 "work_obstructed": True, "personal_obstructed": False})
    assert s.state["items"] == []      # a false "marked" here would park a
    # real item as done and its task would never be created
    s.reconcile({"work": ["brand new"], "personal": []})
    assert s.state["items"][0]["status"] == "open"   # clear read: tracked


def test_pending_accrues_from_text_sighting_on_obstructed_board(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.reconcile({"work": ["new item"], "personal": []})
    # obstructed frame misreads its checkbox as marked: the TEXT repeating is
    # still confirmation — accrue; do NOT convert to done off an untrusted tick
    s.reconcile({"work": [], "work_done": ["new item"], "personal": [],
                 "work_obstructed": True, "personal_obstructed": False})
    assert s.state["items"][0]["status"] == "open"
    assert s.ha.calls[-1][:2] == ("todoist", "new_task")


def test_dry_run_reports_tick_completions_without_touching(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    preview = s.reconcile({"work": [], "work_done": ["task"], "personal": []},
                          apply=False)
    assert preview["completed"] == ["task"]
    assert s.ha.calls == []
    assert s.state["items"][0]["status"] == "open"   # untouched


def test_edit_renames_todoist_task_and_state(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Bluetooth Adapter"], "personal": []})
    s.ha.calls.clear()
    code, body = s.edit_item("work", "Bluetooth Adapter", "Bluetooth Adaptor ")
    assert code == 200 and body["ok"]
    assert "warning" not in body                     # similar text: no warning
    assert s.ha.calls == [("todo", "update_item", {
        "entity_id": "todo.inbox",
        "item": "Bluetooth Adapter", "rename": "Bluetooth Adaptor",
    })]
    assert s.state["items"][0]["text"] == "Bluetooth Adaptor"
    # the rename persisted — a restart must still see the canonical text
    assert make_syncer(tmp_path).state["items"][0]["text"] == "Bluetooth Adaptor"
    # ...and future completions use the NEW canonical text
    s.ha.calls.clear()
    s.reconcile({"work": [], "personal": []})
    s.reconcile({"work": [], "personal": []})
    assert s.ha.calls[-1][2]["item"] == "Bluetooth Adaptor"


def test_edit_warns_when_new_text_wont_fuzzy_match(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["buy milk"], "personal": []})
    code, body = s.edit_item("work", "buy milk", "call the plumber")
    assert code == 200 and "warning" in body


def test_edit_failed_ha_rename_changes_nothing(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.ok = False
    code, body = s.edit_item("work", "task", "renamed")
    assert code == 502 and "error" in body
    assert s.state["items"][0]["text"] == "task"


def test_edit_done_item_skips_ha_call(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    s.ha.calls.clear()
    code, _ = s.edit_item("work", "task", "task (was misread)")
    assert code == 200 and s.ha.calls == []
    assert s.state["items"][0]["text"] == "task (was misread)"


def test_edit_rejects_empty_duplicate_and_unknown(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["one", "completely different"], "personal": []})
    assert s.edit_item("work", "one", "   ")[0] == 400
    assert s.edit_item("work", "one", "completely different")[0] == 400
    assert s.edit_item("work", "no such item", "x")[0] == 404
    assert s.edit_item("personal", "one", "x")[0] == 404  # wrong board


def test_remove_deletes_todoist_task_and_ignores_until_erased(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["misread item"], "personal": []})
    s.ha.calls.clear()
    code, body = s.remove_item("work", "misread item")
    assert code == 200 and body["ok"]
    assert s.ha.calls == [("todo", "remove_item", {
        "entity_id": "todo.inbox", "item": "misread item"})]
    assert s.state["items"][0]["status"] == "removed"
    # still written on the board: NOT re-added as a new task
    s.ha.calls.clear()
    result = s.reconcile({"work": ["misread item"], "personal": []})
    assert result["added"] == [] and s.ha.calls == []
    # erased: purged after missing_to_complete scans, without any HA call
    s.reconcile({"work": [], "personal": []})
    s.reconcile({"work": [], "personal": []})
    assert s.state["items"] == [] and s.ha.calls == []


def test_remove_done_item_skips_ha_call(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    s.ha.calls.clear()
    code, _ = s.remove_item("work", "task")
    assert code == 200 and s.ha.calls == []
    assert s.state["items"][0]["status"] == "removed"


def test_remove_failed_ha_delete_keeps_item_open(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.ok = False
    code, _ = s.remove_item("work", "task")
    assert code == 502
    assert s.state["items"][0]["status"] == "open"


def test_manual_edits_refuse_rather_than_block_during_a_scan(tmp_path, monkeypatch):
    import wbsync
    monkeypatch.setattr(wbsync, "EDIT_LOCK_TIMEOUT_S", 0.05)
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s._lock.acquire()  # a scan is holding the lock
    try:
        assert s.edit_item("work", "task", "renamed")[0] == 409
        assert s.remove_item("work", "task")[0] == 409
        assert s.reopen_item("work", "task")[0] == 409
    finally:
        s._lock.release()


def test_manual_edits_land_in_history(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.edit_item("work", "task", "task v2")
    assert s.history[-1]["type"] == "manual" and s.history[-1]["action"] == "edit"
    s.remove_item("work", "task v2")
    assert s.history[-1]["action"] == "remove" and s.history[-1]["text"] == "task v2"


def test_reopen_done_item_creates_fresh_task_and_reopens(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.reconcile({"work": [], "work_done": ["task"], "personal": []})  # completed
    assert s.state["items"][0]["status"] == "done"
    s.ha.calls.clear()
    code, body = s.reopen_item("work", "task")
    assert code == 200 and body["ok"]
    assert s.ha.calls == [("todoist", "new_task", {
        "content": "task", "project": "Inbox",
        "labels": ["whiteboard", "work"], "due_date_string": "today",
    })]
    item = s.state["items"][0]
    assert item["status"] == "open" and item["ticked"] == 0 and item["missing"] == 0
    assert make_syncer(tmp_path).state["items"][0]["status"] == "open"  # persisted
    assert s.history[-1] == {"type": "manual", "action": "reopen",
                             "board": "work", "text": "task",
                             "at": s.history[-1]["at"], "t": s.history[-1]["t"]}
    # still ticked on the physical board: the scans will complete it again
    result = s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    assert result["completed"] == ["task"]


def test_reopen_removed_item_restores_it(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.remove_item("work", "task")
    s.ha.calls.clear()
    code, _ = s.reopen_item("work", "task")
    assert code == 200
    assert s.ha.calls[-1][:2] == ("todoist", "new_task")
    assert s.state["items"][0]["status"] == "open"


def test_reopen_ghost_done_item_creates_its_first_task(tmp_path):
    s = make_syncer(tmp_path)
    # first seen already ticked: tracked as done, never had a Todoist task
    s.reconcile({"work": [], "work_done": ["misread tick"], "personal": []})
    s.ha.calls.clear()
    code, _ = s.reopen_item("work", "misread tick")
    assert code == 200
    assert s.ha.calls[-1][:2] == ("todoist", "new_task")


def test_reopen_edge_cases(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["open item"], "personal": []})   # open, task exists
    s.settings["sightings_to_add"] = 2
    s.reconcile({"work": ["open item", "confirmed"], "personal": []})  # pending
    assert s.state["items"][-1]["status"] == "pending"
    s.ha.calls.clear()
    assert s.reopen_item("work", "no such")[0] == 404
    assert s.reopen_item("work", "confirmed")[0] == 400    # pending: no task yet
    code, _ = s.reopen_item("work", "open item")           # already open: no-op
    assert code == 200 and s.ha.calls == []


def test_reopen_failed_ha_push_changes_nothing(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    s.ha.ok = False
    code, _ = s.reopen_item("work", "task")
    assert code == 502
    assert s.state["items"][0]["status"] == "done"


# -- model output normalisation ---------------------------------------------

def test_normalise_read_maps_checkbox_states_to_lists():
    raw = {
        "work": [{"text": "open item", "checkbox": "empty"},
                 {"text": "slashed item", "checkbox": "slash"},
                 {"text": "no box read"},                     # unknown -> open
                 {"text": "  ", "checkbox": "empty"},         # blank: dropped
                 "garbage string",                            # junk: dropped
                 {"checkbox": "slash"}],                      # no text: dropped
        "personal": [{"text": "chore", "checkbox": "check_x_or_scribble"}],
        "work_obstructed": True,
    }
    assert normalise_read(raw) == {
        "work": ["open item", "no box read"], "work_done": ["slashed item"],
        "personal": [], "personal_done": ["chore"],
        "work_obstructed": True, "personal_obstructed": False,
    }


# -- sighting streak for new items ------------------------------------------

def test_new_item_needs_two_sightings_before_task_created(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    r1 = s.reconcile({"work": ["real new item"], "personal": []})
    assert r1["added"] == [] and r1["pending"] == ["real new item"]
    assert s.ha.calls == []                          # nothing hit Todoist yet
    assert s.state["items"][0] == {"text": "real new item", "board": "work",
                                   "status": "pending", "missing": 0,
                                   "ticked": 0, "seen": 1}
    r2 = s.reconcile({"work": ["real new item"], "personal": []})
    assert r2["added"] == ["real new item"] and r2["pending"] == []
    assert s.ha.calls[-1][:2] == ("todoist", "new_task")
    assert s.state["items"][0]["status"] == "open"
    assert "seen" not in s.state["items"][0]


def test_hallucinated_read_never_reaches_todoist(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.reconcile({"work": ["Buy cake", "Call mom"], "personal": ["Do homework"]})
    # next read is sane again: the fabrications miss once and die
    s.reconcile({"work": [], "personal": []})
    assert s.ha.calls == []
    assert s.state["items"] == []


def test_pending_survives_obstructed_board_but_not_a_clear_miss(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=3)
    s.reconcile({"work": ["fragment"], "personal": []})
    # hidden board: pending neither accrues nor dies
    s.reconcile({"work": [], "personal": [], "work_obstructed": True})
    assert s.state["items"][0]["seen"] == 1
    s.reconcile({"work": ["fragment"], "personal": []})
    assert s.state["items"][0]["seen"] == 2
    s.reconcile({"work": [], "personal": []})        # clear board, one miss
    assert s.state["items"] == [] and s.ha.calls == []


def test_pending_item_first_ticked_becomes_done_without_task(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.reconcile({"work": ["quick job"], "personal": []})
    s.reconcile({"work": [], "work_done": ["quick job"], "personal": []})
    assert s.ha.calls == []                          # no add, no completion
    assert s.state["items"][0]["status"] == "done"


def test_pending_push_failure_retries_next_scan(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.reconcile({"work": ["new item"], "personal": []})
    s.ha.ok = False
    r = s.reconcile({"work": ["new item"], "personal": []})
    assert r["failed"] == 1 and r["added"] == []
    assert s.state["items"][0]["status"] == "pending"
    s.ha.ok = True
    r = s.reconcile({"work": ["new item"], "personal": []})
    assert r["added"] == ["new item"]
    assert s.state["items"][0]["status"] == "open"


def test_pending_dry_run_previews_without_touching(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.reconcile({"work": ["new item"], "personal": []})
    s.ha.calls.clear()
    preview = s.reconcile({"work": ["new item"], "personal": []}, apply=False)
    assert preview["added"] == ["new item"]          # would confirm next scan
    assert s.ha.calls == []
    assert s.state["items"][0]["seen"] == 1          # untouched


def test_scan_keeps_reading_unchanged_frames_while_pending_exists(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.reader = FixedReader({"work": ["new item"], "personal": []})
    fake_fetch(s)
    s.scan(force=True)                               # pending, baseline saved
    result = s.scan()                                # identical frame
    assert result["changed"] is True                 # NOT skipped: pending exists
    assert result["added"] == ["new item"]
    result = s.scan()                                # no pending left now
    assert result["changed"] is False                # skip resumes


# -- mass-erase guard --------------------------------------------------------

def test_garbage_read_defers_mass_completions_one_extra_scan(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    texts = ["alpha task", "beta errand", "gamma chore", "delta fix", "epsilon job"]
    s.settings["sightings_to_add"] = 1
    s.reconcile({"work": texts, "personal": []})
    s.settings["sightings_to_add"] = 2
    s.ha.calls.clear()

    s.reconcile({"work": ["totally fabricated"], "personal": []})      # miss 1
    r = s.reconcile({"work": ["different nonsense"], "personal": []})  # miss 2
    assert r["completed"] == [] and r["slowed"] == ["work"]            # deferred
    assert [c for c in s.ha.calls if c[1] == "update_item"] == []
    r = s.reconcile({"work": ["yet more junk"], "personal": []})       # miss 3
    assert sorted(r["completed"]) == sorted(texts)   # persists -> believed


def test_resighting_after_garbage_read_resets_cleanly(tmp_path):
    s = make_syncer(tmp_path)
    texts = ["alpha task", "beta errand", "gamma chore", "delta fix", "epsilon job"]
    s.reconcile({"work": texts, "personal": []})
    s.settings["sightings_to_add"] = 2
    s.ha.calls.clear()
    s.reconcile({"work": ["totally fabricated"], "personal": []})      # miss 1
    s.reconcile({"work": ["different nonsense"], "personal": []})      # miss 2 deferred
    s.reconcile({"work": texts, "personal": []})                       # good read again
    assert all(i["missing"] == 0 for i in s.state["items"])
    assert [c for c in s.ha.calls if c[1] == "update_item"] == []
    assert len(s.state["items"]) == 5                # fabrications died as pending


# -- tick confirmation (second photo before completing) ---------------------

class SeqReader:
    """Returns queued reads in order; raises when the queue is empty."""
    def __init__(self, seens):
        self.seens = list(seens)
        self.calls = 0

    def read(self, jpeg):
        self.calls += 1
        nxt = self.seens.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def test_confirm_ticks_keeps_agreed_demotes_disputed():
    seen = {"work": ["open item"], "work_done": ["agreed", "disputed"],
            "personal": [], "personal_done": [],
            "work_obstructed": False, "personal_obstructed": False}
    merged = confirm_ticks(seen, {"work": [], "work_done": ["agreed!"],  # OCR wobble
                                  "personal": [], "personal_done": [],
                                  "work_obstructed": False,
                                  "personal_obstructed": True})
    assert merged["work_done"] == ["agreed"]        # canonical text from read 1
    assert merged["work"] == ["open item", "disputed"]  # demoted to a sighting
    assert merged["personal_obstructed"] is True    # obstruction flags OR


def test_confirm_ticks_failed_second_read_demotes_everything():
    seen = {"work": [], "work_done": ["task"], "personal": [],
            "personal_done": [], "work_obstructed": False,
            "personal_obstructed": False}
    merged = confirm_ticks(seen, None)
    assert merged["work_done"] == [] and merged["work"] == ["task"]


def confirm_syncer(tmp_path, reads):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    s.reader = SeqReader(reads)
    fake_fetch(s)
    return s


TICKED = {"work": [], "work_done": ["task"], "personal": [],
          "work_obstructed": False, "personal_obstructed": False}
OPEN = {"work": ["task"], "personal": [],
        "work_obstructed": False, "personal_obstructed": False}


def test_imminent_tick_completion_triggers_confirm_read(tmp_path):
    s = confirm_syncer(tmp_path, [TICKED, TICKED])
    result = s.scan(force=True)
    assert s.reader.calls == 2                       # confirm read happened
    assert result["completed"] == ["task"]           # both reads agree


def test_disputed_tick_defers_completion(tmp_path):
    s = confirm_syncer(tmp_path, [TICKED, OPEN])
    result = s.scan(force=True)
    assert s.reader.calls == 2
    assert result["completed"] == [] and s.ha.calls == []
    assert s.state["items"][0]["status"] == "open"
    assert s.state["items"][0]["missing"] == 0       # still counts as sighted


def test_failed_confirm_read_defers_completion(tmp_path):
    s = confirm_syncer(tmp_path, [TICKED, RuntimeError("model hiccup")])
    result = s.scan(force=True)
    assert result["ok"] is True                      # scan itself survives
    assert result["tick_confirmed"] is False
    assert result["completed"] == [] and s.ha.calls == []


def test_no_confirm_read_when_nothing_would_complete(tmp_path):
    s = confirm_syncer(tmp_path, [OPEN])
    s.update_settings({"ticked_to_complete": 2})
    s.reader.seens = [TICKED]                        # tick 1 of 2: not imminent
    result = s.scan(force=True)
    assert s.reader.calls == 1                       # single read only
    assert result["completed"] == []
    assert s.state["items"][0]["ticked"] == 1


def test_confirm_read_can_be_disabled(tmp_path):
    s = confirm_syncer(tmp_path, [TICKED])
    s.update_settings({"tick_confirm": False})
    result = s.scan(force=True)
    assert s.reader.calls == 1
    assert result["completed"] == ["task"]


def test_dry_run_never_takes_a_confirm_read(tmp_path):
    s = confirm_syncer(tmp_path, [TICKED])
    result = s.scan(force=True, dry=True)
    assert s.reader.calls == 1
    assert result["completed"] == ["task"] and s.ha.calls == []


def test_true_board_wipe_still_completes_at_normal_threshold(tmp_path):
    s = make_syncer(tmp_path)
    texts = ["alpha task", "beta errand", "gamma chore", "delta fix", "epsilon job"]
    s.reconcile({"work": texts, "personal": []})
    s.ha.calls.clear()
    s.reconcile({"work": [], "personal": []})        # empty board: miss 1
    r = s.reconcile({"work": [], "personal": []})    # miss 2: real wipe, no slowdown
    assert sorted(r["completed"]) == sorted(texts) and r["slowed"] == []
    assert s.state["items"] == []
