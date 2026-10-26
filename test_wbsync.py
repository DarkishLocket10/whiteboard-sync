"""Reconciliation and helpers — the logic that decides what hits Todoist."""
import json

import pytest

from wbsync import (ASSIGN_FLOOR, Config, Syncer, assign_matches,
                    confirm_ticks, depth_blocked, fuzzy_match, match_score,
                    norm_for_match, normalise_read, parse_region)


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


def fake_fetch(syncer, luma=150.0):
    """Stub the camera with a well-lit constant frame (dark frames are
    skipped and dim ones distrusted — see DARK_LUMA / DIM_LUMA)."""
    import numpy as np
    syncer._fetch_crop = lambda frames=None: (
        b"jpeg", np.full((5, 5), luma, dtype=np.float32))


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


def test_word_internal_dot_reads_back_as_an_i():
    # This writer's lowercase "i" is a detached dot with no stem, so the model
    # transcribes it as a period. All of these are real reads off the boards.
    assert norm_for_match("Organ.ze K.tchen Sp.ces") == norm_for_match(
        "Organize Kitchen Spices")
    assert norm_for_match("Rev.ew") == norm_for_match("Review")
    assert norm_for_match('Make AI roleplays "n.cer"') == norm_for_match(
        'Make AI roleplays "nicer"')
    # A trailing period is punctuation, not a letter, and must not become "i".
    assert norm_for_match("Send invoice.") == "send invoice"


def test_truncated_reads_still_match_their_item():
    # Fragments that used to fall off the 0.8 cliff, complete the real task
    # and re-add themselves as a duplicate. "Europe Express Change Request!"
    # churned six add/complete cycles this way.
    for whole, fragment in [
            ("Update Flight RISK ticket", "Update Flight"),
            ("Remove Amplitude tracking", "Remove Ampli+"),
            ("Europe Express Change Request!", "pe Express"),
            ("Europe Express Change Request!", "nge Request!")]:
        assert match_score(whole, fragment) >= ASSIGN_FLOOR, (whole, fragment)


def test_lightly_reworded_item_still_matches():
    # The user rewords for clarity when rewriting; this pair caused a real
    # false completion plus a duplicate task on 2026-09-21.
    assert match_score("Add LOOPS logging to Posthog",
                       "Implement Loops logging via Posthog") >= ASSIGN_FLOOR


def test_unrelated_items_do_not_match():
    for a, b in [("Send Invoice", "Clean fridge"),
                 ("Buy milk", "Buy cake"),
                 ("Organize Kitchen Spices", "Clean fridge")]:
        assert match_score(a, b) < ASSIGN_FLOOR, (a, b)


def test_assignment_is_one_to_one_and_prefers_the_better_pair():
    known = ["Update Flight RISK ticket", "Update Laura's Small ticket"]
    # Both reads look like both items; assignment must not give either item
    # to two lines, and each line must land on its own best partner.
    got = assign_matches(["Update Flight", "Update Laura's"], known)
    assert got["Update Flight"] == "Update Flight RISK ticket"
    assert got["Update Laura's"] == "Update Laura's Small ticket"
    assert len(set(got.values())) == 2


def test_assignment_leaves_genuinely_new_text_unmatched():
    known = ["Clean fridge", "Send Invoice"]
    got = assign_matches(["Clean fridge", "Book dentist appointment"], known)
    assert got["Clean fridge"] == "Clean fridge"
    assert "Book dentist appointment" not in got


def test_reworded_item_neither_completes_nor_duplicates(tmp_path):
    # End to end: an item rewritten with different wording must stay the SAME
    # tracked item - no completion, no second Todoist task.
    s = make_syncer(tmp_path)
    fake_fetch(s)
    s.reconcile({"work": ["Add LOOPS logging to Posthog"], "personal": []})
    assert [i["text"] for i in s.state["items"]] == ["Add LOOPS logging to Posthog"]
    s.ha.calls.clear()
    res = s.reconcile({"work": ["Implement Loops logging via Posthog"],
                       "personal": []})
    assert res["added"] == [] and res["completed"] == []
    assert len(s.state["items"]) == 1
    assert s.state["items"][0]["missing"] == 0
    assert s.ha.calls == []


def test_new_items_create_tagged_todoist_tasks(tmp_path):
    s = make_syncer(tmp_path)
    result = s.reconcile({"work": ["Fix dashboard"], "personal": ["do laundry"]})
    assert sorted(result["added"]) == ["Fix dashboard", "do laundry"]
    tasks = [c for c in s.ha.calls if c[:2] == ("todoist", "new_task")]
    assert {t[2]["content"]: t[2]["labels"] for t in tasks} == {
        "Fix dashboard": ["work"],
        "do laundry": ["personal"],
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
                                        "status": "open", "missing": 0,
                                        "ticked": 0, "pos": 0}]


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
                                 "status": "open", "missing": 0, "ticked": 0,
                                 "pos": 0}]


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
        "labels": ["work"], "due_date_string": "today",
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
        "work_order": ["open item", "slashed item", "no box read"],
        "personal": [], "personal_done": ["chore"],
        "personal_order": ["chore"],
        "work_obstructed": True, "personal_obstructed": False,
    }


# -- board order --------------------------------------------------------------

def test_items_carry_their_position_on_the_board(tmp_path):
    s = make_syncer(tmp_path)
    # board top-to-bottom: alpha (open), beta (ticked), gamma (open) —
    # the split open/done lists alone would lose beta's middle position
    s.reconcile({"work": ["alpha task", "gamma chore"],
                 "work_done": ["beta errand"],
                 "work_order": ["alpha task", "beta errand", "gamma chore"],
                 "personal": []})
    by_text = {i["text"]: i["pos"] for i in s.state["items"]}
    assert by_text == {"alpha task": 0, "beta errand": 1, "gamma chore": 2}


def test_positions_restamp_when_the_board_is_rewritten(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["alpha task", "gamma chore"], "personal": []})
    # rewritten in the opposite order (fuzzy text wobble included)
    s.reconcile({"work": ["gamma chore!", "alpha task"],
                 "work_order": ["gamma chore!", "alpha task"],
                 "personal": []})
    by_text = {i["text"]: i["pos"] for i in s.state["items"]}
    assert by_text == {"gamma chore": 0, "alpha task": 1}


def test_positions_are_per_board_and_survive_reload(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["work thing"], "personal": ["personal thing"]})
    assert all(i["pos"] == 0 for i in s.state["items"])  # independent boards
    reloaded = make_syncer(tmp_path)
    assert all(i["pos"] == 0 for i in reloaded.state["items"])


def test_legacy_items_without_pos_keep_file_order(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"items": [
        {"text": "first", "board": "work", "missing": 0},
        {"text": "second", "board": "work", "missing": 0}]}))
    s = make_syncer(tmp_path)
    assert [i["pos"] for i in s.state["items"]] == [0, 1]


def test_dry_run_does_not_touch_positions(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["alpha task", "gamma chore"], "personal": []})
    s.reconcile({"work": ["gamma chore", "alpha task"], "personal": []},
                apply=False)
    by_text = {i["text"]: i["pos"] for i in s.state["items"]}
    assert by_text == {"alpha task": 0, "gamma chore": 1}  # unchanged


# -- sighting streak for new items ------------------------------------------

def test_new_item_needs_two_sightings_before_task_created(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    r1 = s.reconcile({"work": ["real new item"], "personal": []})
    assert r1["added"] == [] and r1["pending"] == ["real new item"]
    assert s.ha.calls == []                          # nothing hit Todoist yet
    assert s.state["items"][0] == {"text": "real new item", "board": "work",
                                   "status": "pending", "missing": 0,
                                   "ticked": 0, "seen": 1, "pos": 0}
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


# -- static-scene tick gate ---------------------------------------------------

def test_tick_streak_cannot_start_on_a_long_static_scene(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    r = s.reconcile({"work": [], "work_done": ["task"], "personal": []},
                    tick_start_ok=False)
    assert r["completed"] == [] and s.ha.calls == []
    item = s.state["items"][0]
    assert item["status"] == "open" and item["ticked"] == 0
    assert item["missing"] == 0                      # text sighting still counts
    # same tick near a real frame change: completes normally
    r = s.reconcile({"work": [], "work_done": ["task"], "personal": []})
    assert r["completed"] == ["task"]


def test_started_streak_may_finish_on_static_frames(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"ticked_to_complete": 2})
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    s.reconcile({"work": [], "work_done": ["task"], "personal": []})  # change frame
    assert s.state["items"][0]["ticked"] == 1
    # the mark persisting on later static frames is confirmation, not a start
    r = s.reconcile({"work": [], "work_done": ["task"], "personal": []},
                    tick_start_ok=False)
    assert r["completed"] == ["task"]


def test_pending_static_tick_does_not_become_a_done_ghost(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.reconcile({"work": ["new item"], "personal": []})
    r = s.reconcile({"work": [], "work_done": ["new item"], "personal": []},
                    tick_start_ok=False)
    item = s.state["items"][0]
    assert item["status"] == "open"                  # counted as a sighting:
    assert s.ha.calls[-1][:2] == ("todoist", "new_task")  # task got created


def test_scan_loop_tracks_scene_staleness(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"change_detection": False, "tick_confirm": False})
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    s.reader = SeqReader([OPEN, OPEN, OPEN, OPEN, TICKED, TICKED])
    fake_fetch(s)
    for _ in range(4):                               # frames identical throughout
        s.scan(force=True)
    assert s.scans_since_change == 3                 # first scan had no baseline
    result = s.scan(force=True)                      # tick appears on static scene
    assert result["completed"] == []                 # blocked: cannot start
    assert s.state["items"][0]["ticked"] == 0
    # now the frame actually changes: the tick may start and complete
    fake_fetch(s, luma=120.0)
    result = s.scan(force=True)
    assert s.scans_since_change == 0
    assert result["completed"] == ["task"]


# -- low light -----------------------------------------------------------------

def test_pitch_dark_frames_never_reconcile(tmp_path):
    """A black frame reads as an empty board — absence logic would complete
    everything as a wipe. Dark scans must skip before reading anything."""
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.reader = SeqReader([])                  # any read attempt would blow up
    fake_fetch(s, luma=5.0)
    result = s.scan(force=True)
    assert result["ok"] is True and result["dark"] is True
    assert s.reader.calls == 0
    assert s.state["items"][0]["missing"] == 0


def test_dim_frames_boost_stack_and_defer_completions(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    fetches = []
    import numpy as np

    def fetch(frames=None):
        fetches.append(frames)
        return b"jpeg", np.full((5, 5), 30, dtype=np.float32)

    s._fetch_crop = fetch
    s.reader = SeqReader([TICKED])
    result = s.scan(force=True)
    assert fetches == [None, 32]              # low-light re-capture happened
    assert result["dim"] is True and result["luma"] == 30
    assert result["completed"] == [] and s.ha.calls == []   # distrusted
    assert s.reader.calls == 1                # ...and no confirm read either
    item = s.state["items"][0]
    assert item["ticked"] == 0 and item["missing"] == 0     # text still counts


def test_dim_absence_counts_no_misses_even_with_guard_off(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"obstruction_guard": False, "missing_to_complete": 1})
    s.reconcile({"work": ["task"], "personal": []})
    s.ha.calls.clear()
    result = s.reconcile({"work": [], "personal": []},
                         force_protect=("work", "personal"))
    assert result["completed"] == [] and s.ha.calls == []
    assert s.state["items"][0]["missing"] == 0


def test_low_light_boost_can_be_disabled(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"low_light_boost": False})
    s.reconcile({"work": ["task"], "personal": []})
    fetches = []
    import numpy as np

    def fetch(frames=None):
        fetches.append(frames)
        return b"jpeg", np.full((5, 5), 30, dtype=np.float32)

    s._fetch_crop = fetch
    s.reader = SeqReader([OPEN])
    result = s.scan(force=True)
    assert fetches == [None]                  # no re-capture
    assert result["dim"] is True              # but reads stay distrusted


def test_dim_frames_skip_entirely_on_scheduled_scans(tmp_path):
    """Dim reads are false-positive factories: protection only deferred
    COMPLETIONS, so garbled fragments still accrued sightings and landed in
    Todoist as partial text. A scheduled dim scan must not read at all."""
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["task"], "personal": []})
    s.reader = SeqReader([])                  # any read attempt would blow up
    fetches = []
    import numpy as np

    def fetch(frames=None):
        fetches.append(frames)
        return b"jpeg", np.full((5, 5), 30, dtype=np.float32)

    s._fetch_crop = fetch
    result = s.scan()
    assert result["ok"] is True and result["dim_skipped"] is True
    assert result["luma"] == 30
    assert fetches == [None]                  # no 32-frame re-capture either
    assert s.reader.calls == 0
    assert s.state["items"][0]["missing"] == 0
    assert s.counters["skips_dim"] == 1


def test_dim_skip_disabled_reads_via_boosted_protected_path(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"dim_skip": False})
    s.reconcile({"work": ["task"], "personal": []})
    fetches = []
    import numpy as np

    def fetch(frames=None):
        fetches.append(frames)
        return b"jpeg", np.full((5, 5), 30, dtype=np.float32)

    s._fetch_crop = fetch
    s.reader = SeqReader([TICKED])
    result = s.scan()
    assert fetches == [None, 32]              # old behavior: boost + protect
    assert result["dim"] is True and result["completed"] == []


def test_dim_threshold_is_tunable(tmp_path):
    """Lowering dim_luma lets marginal-light frames through as normal reads."""
    s = make_syncer(tmp_path)
    s.update_settings({"dim_luma": 25})
    s.reconcile({"work": ["task"], "personal": []})
    s.reader = SeqReader([OPEN])
    fake_fetch(s, luma=30.0)
    result = s.scan()
    assert "dim_skipped" not in result and "dim" not in result
    assert result["changed"] is True and s.reader.calls == 1


# -- depth guard ---------------------------------------------------------------

def test_depth_blocked_rule():
    # field values 2026-07-09: a person in front vs a clear board plane
    assert depth_blocked({"p10_mm": 988, "p90_mm": 1997, "valid_frac": 0.97})
    assert not depth_blocked({"p10_mm": 1915, "p90_mm": 1988, "valid_frac": 0.99})
    assert not depth_blocked({"p10_mm": 500, "p90_mm": 2000, "valid_frac": 0.3})
    assert not depth_blocked({"valid_frac": 0.9})    # degenerate: no percentiles
    assert not depth_blocked(None)


def test_depth_guard_protects_a_board_the_model_calls_clear(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"obstruction_guard": False, "tick_confirm": False})
    s.reconcile({"work": ["task"], "personal": ["chore"]})
    s.ha.calls.clear()
    s._depth_obstructed = lambda: ["work"]
    s.reader = SeqReader([{"work": [], "work_done": ["task"],
                           "personal": [], "personal_done": ["chore"],
                           "work_obstructed": False,
                           "personal_obstructed": False}])
    fake_fetch(s)
    result = s.scan(force=True)
    assert result["depth_blocked"] == ["work"]
    # work tick ignored (physically blocked); personal tick completes
    assert result["completed"] == ["chore"]
    by_text = {i["text"]: i for i in s.state["items"]}
    assert by_text["task"]["status"] == "open" and by_text["task"]["ticked"] == 0
    assert by_text["chore"]["status"] == "done"


def test_depth_guard_can_be_disabled(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"depth_guard": False, "tick_confirm": False})
    calls = []
    s._depth_obstructed = lambda: calls.append(1) or ["work"]
    s.reconcile({"work": ["task"], "personal": []})
    s.reader = SeqReader([TICKED])
    fake_fetch(s)
    result = s.scan(force=True)
    assert calls == []                       # depth endpoint never queried
    assert result["completed"] == ["task"]


def test_depth_endpoint_failure_degrades_to_no_evidence(tmp_path):
    s = make_syncer(tmp_path)                # cfg.kinect_url is unreachable
    s.update_settings({"tick_confirm": False})
    s.reconcile({"work": ["task"], "personal": []})
    s.reader = SeqReader([TICKED])
    fake_fetch(s)
    result = s.scan(force=True)              # real _depth_obstructed -> []
    assert "depth_blocked" not in result
    assert result["completed"] == ["task"]   # model-only path still works


# -- depth presence skip (someone sitting at a board) --------------------------

def test_person_at_board_skips_scan_entirely(tmp_path):
    """ToF depth says someone is at a board BEFORE any photo is taken: no
    capture, no model read. Their body hides parts of lines and the visible
    fragments would accrue sightings and become real Todoist tasks."""
    s = make_syncer(tmp_path)
    s._depth_obstructed = lambda: ["work"]
    s.reader = SeqReader([])                  # any read attempt would blow up
    fetches = []
    s._fetch_crop = lambda frames=None: fetches.append(frames)
    result = s.scan()
    assert result["ok"] is True and result["changed"] is False
    assert result["person"] == ["work"]
    assert fetches == [] and s.reader.calls == 0
    assert s.counters["skips_person"] == 1
    assert s.history[-1]["person"] == ["work"]


def test_person_skip_can_be_disabled(tmp_path):
    """With depth_skip off the old behavior holds: read through the person,
    reconcile-time depth guard protecting the blocked board."""
    s = make_syncer(tmp_path)
    s.update_settings({"depth_skip": False, "tick_confirm": False})
    s.reconcile({"work": ["task"], "personal": []})
    s._depth_obstructed = lambda: ["work"]
    s.reader = SeqReader([TICKED])
    fake_fetch(s)
    result = s.scan()
    assert "person" not in result
    assert result["depth_blocked"] == ["work"]
    assert result["completed"] == [] and s.reader.calls == 1
    assert s.state["items"][0]["status"] == "open"


def test_forced_scan_overrides_person_skip(tmp_path):
    """The dashboard's Scan-now button is pressed while sitting at the desk —
    it must still scan, with the blocked board protected as before."""
    s = make_syncer(tmp_path)
    s.update_settings({"tick_confirm": False})
    s.reconcile({"work": ["task"], "personal": []})
    s._depth_obstructed = lambda: ["work"]
    s.reader = SeqReader([TICKED])
    fake_fetch(s)
    result = s.scan(force=True)
    assert "person" not in result
    assert result["depth_blocked"] == ["work"]
    assert result["completed"] == [] and s.reader.calls == 1


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


# ---------------------------------------------------------------------------
# Presence: "don't scan while a human is in front of the board"
#
# The old guard asked kinect-knob's region_depth whether p10 sat 400mm in
# front of p90 — a test that needs an object covering ~10% of a board half.
# A head leaning in front of two lines covers a few percent, so it read
# "clear", the model transcribed the fragments it could see, and the
# fragments became Todoist tasks. These cover the replacement.
# ---------------------------------------------------------------------------
import time as _time

from wbsync import PresenceWatcher


class FakePresenceAPI:
    """Stands in for kinect-knob's /api/presence."""

    def __init__(self, occupancy=0.0, status=200, raises=False):
        self.occupancy = occupancy          # float, or {board_x1: float}
        self.status = status
        self.raises = raises
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append(params)
        if self.raises:
            raise __import__("requests").RequestException("boom")
        occ = self.occupancy
        if isinstance(occ, dict):
            occ = occ.get(params["x1"], 0.0)

        class R:
            status_code = self.status

            def raise_for_status(_self):
                if self.status >= 400:
                    raise __import__("requests").HTTPError(str(self.status))

            def json(_self):
                return {"present": occ > 0, "occupancy": occ, "available": True,
                        "region": {"occupancy": occ, "plane_mm": 1950,
                                   "nearest_mm": 1200 if occ else None,
                                   "valid_frac": 0.99, "age_s": 0.2}}
        return R()


def make_watcher(monkeypatch, api, region=(600, 280, 1320, 810), settings=None):
    cfg = type("C", (), {"kinect_url": "http://kk", "region": region})()
    w = PresenceWatcher(cfg, settings or {"presence_occupancy": 0.015})
    monkeypatch.setattr("wbsync.requests.get", api)
    return w


def test_watcher_asks_about_both_board_halves_with_a_margin(monkeypatch):
    """A head just OUTSIDE the crop is about to be inside it, and its arm
    already is — so the region asked about is wider than the crop."""
    api = FakePresenceAPI()
    w = make_watcher(monkeypatch, api)
    regions = w.regions()
    assert set(regions) == {"work", "personal"}
    (wx1, wy1, wx2, wy2) = regions["work"]
    assert wx1 == 600 - 120 and wy1 == 280 - 120
    assert wx2 == 960 + 120 and wy2 == 810 + 120
    assert regions["personal"][0] == 960 - 120
    assert regions["personal"][2] == 1320 + 120


def test_a_head_sized_occupancy_blocks_the_board(monkeypatch):
    """4% of a board half — invisible to the percentile guard, decisive here."""
    w = make_watcher(monkeypatch, FakePresenceAPI(occupancy=0.04))
    w.poll()
    assert w.supported is True
    assert w.status("work") == "blocked"
    assert w.blocked_boards(min_clear_s=0) == ["work", "personal"]


def test_a_clear_board_is_clear(monkeypatch):
    w = make_watcher(monkeypatch, FakePresenceAPI(occupancy=0.0001))
    w.poll()
    assert w.status("work") == "clear"
    assert w.blocked_boards(min_clear_s=0) == []


def test_only_the_occupied_half_is_blocked(monkeypatch):
    """Sitting at the work board must not stop the personal board being read
    — except that it does, by the pre-existing whole-scan skip. What matters
    here is that the SIGNAL is per board."""
    api = FakePresenceAPI(occupancy={600 - 120: 0.05, 960 - 120: 0.0})
    w = make_watcher(monkeypatch, api)
    w.poll()
    assert w.status("work") == "blocked"
    assert w.status("personal") == "clear"


def test_a_board_must_be_clear_for_the_settle_window(monkeypatch):
    """Somebody who just stood up is still leaning over the desk."""
    w = make_watcher(monkeypatch, FakePresenceAPI(occupancy=0.05))
    w.poll()                                   # blocked now
    w._settings["presence_occupancy"] = 0.015
    api2 = FakePresenceAPI(occupancy=0.0)
    import wbsync
    wbsync.requests.get = api2
    w.poll()                                   # clear, but only just
    assert w.status("work") == "clear"
    assert w.blocked_boards(min_clear_s=30) == ["work", "personal"]
    assert w.blocked_boards(min_clear_s=0) == []


def test_blocked_since_answers_about_the_photo_window(monkeypatch):
    """The post-capture question: was anybody in shot WHILE that photo was
    taken? A pre-flight check can never answer it."""
    w = make_watcher(monkeypatch, FakePresenceAPI(occupancy=0.0))
    w.poll()
    t_capture = _time.time()
    import wbsync
    wbsync.requests.get = FakePresenceAPI(occupancy=0.05)
    w.poll()                                   # somebody arrived mid-capture
    assert w.blocked_since(t_capture) == ["personal", "work"]
    assert w.blocked_since(_time.time() + 10) == []


def test_missing_endpoint_is_reported_not_guessed(monkeypatch):
    """An older kinect-knob has no /api/presence. That must read as 'no
    occupancy signal', never as 'nobody is there'."""
    w = make_watcher(monkeypatch, FakePresenceAPI(status=404))
    w.poll()
    assert w.supported is False
    assert w.status("work") == "unknown"
    assert w.blocked_boards(min_clear_s=0) == []


def test_unreachable_endpoint_reads_unknown(monkeypatch):
    w = make_watcher(monkeypatch, FakePresenceAPI(raises=True))
    w.poll()
    assert w.status("work") == "unknown"
    assert w.error


def test_stale_readings_read_unknown(monkeypatch):
    w = make_watcher(monkeypatch, FakePresenceAPI(occupancy=0.0))
    w.poll()
    assert w.status("work") == "clear"
    for rec in w.boards.values():
        rec["t"] -= 120                        # older than PRESENCE_STALE_S
    assert w.status("work") == "unknown"
    assert w.clear_for_s("work") is None


# -- how the syncer uses it -------------------------------------------------


def _stub_watcher(s, blocked=(), late=(), supported=True):
    s.watcher.supported = supported
    s.watcher.blocked_boards = lambda min_clear_s: list(blocked)
    s.watcher.blocked_since = lambda since_t: list(late)


def test_occupancy_is_preferred_over_the_percentile_guard(tmp_path):
    s = make_syncer(tmp_path)
    _stub_watcher(s, blocked=["work"])
    s._depth_obstructed = lambda: pytest.fail("should not fall back to depth")
    assert s._presence_blocked() == (["work"], "presence")


def test_the_percentile_guard_still_covers_an_older_kinect_knob(tmp_path):
    s = make_syncer(tmp_path)
    s.watcher.supported = False
    s._depth_obstructed = lambda: ["personal"]
    s._depth_ok = True
    assert s._presence_blocked() == (["personal"], "depth")


def test_no_evidence_is_reported_as_no_evidence(tmp_path):
    s = make_syncer(tmp_path)
    s.watcher.supported = False
    s._depth_obstructed = lambda: []
    s._depth_ok = False
    assert s._presence_blocked() == ([], "none")


def test_presence_skip_takes_no_photo(tmp_path):
    s = make_syncer(tmp_path)
    _stub_watcher(s, blocked=["work"])
    s.reader = SeqReader([])
    fetches = []
    s._fetch_crop = lambda frames=None: fetches.append(frames)
    result = s.scan()
    assert result["ok"] is True and result["changed"] is False
    assert result["person"] == ["work"] and result["person_via"] == "presence"
    assert fetches == [] and s.reader.calls == 0
    assert s.counters["skips_person"] == 1


def test_somebody_who_arrives_during_the_photo_voids_it(tmp_path):
    """THE regression. The pre-flight check said clear; he sat down while the
    stack was being captured; the old code read the photo anyway and turned
    his half-hidden lines into tasks. The re-check must throw the frame away
    — and before the model read, so a ruined frame costs nothing."""
    s = make_syncer(tmp_path)
    _stub_watcher(s, blocked=[], late=["work"])
    fake_fetch(s)
    s.reader = SeqReader([])                   # any read would blow up
    result = s.scan()
    assert result["ok"] is True and result["changed"] is False
    assert result["person"] == ["work"] and result["person_late"] is True
    assert s.reader.calls == 0
    assert s.counters["skips_person_late"] == 1


def test_the_recheck_can_be_turned_off(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"presence_recheck": False, "tick_confirm": False})
    _stub_watcher(s, blocked=[], late=["work"])
    fake_fetch(s)
    s.reader = FixedReader({"work": ["Fix dashboard"], "personal": []})
    result = s.scan()
    assert result["changed"] is True


def test_a_forced_scan_ignores_presence(tmp_path):
    """The dashboard's Scan-now button means now."""
    s = make_syncer(tmp_path)
    s.update_settings({"tick_confirm": False})
    _stub_watcher(s, blocked=["work"], late=["work"])
    fake_fetch(s)
    s.reader = FixedReader({"work": ["Fix dashboard"], "personal": []})
    result = s.scan(force=True)
    assert result["changed"] is True


def test_presence_required_waits_when_there_is_no_evidence(tmp_path):
    """Opt-in strictness: a silent camera means 'wait', not 'read anyway'."""
    s = make_syncer(tmp_path)
    s.update_settings({"presence_required": True})
    s.watcher.supported = False
    s._depth_obstructed = lambda: []
    s._depth_ok = False
    fetches = []
    s._fetch_crop = lambda frames=None: fetches.append(frames)
    result = s.scan()
    assert result["person_unknown"] is True
    assert fetches == []


def test_without_presence_required_no_evidence_still_scans(tmp_path):
    """Default stays backward compatible: every downstream guard still runs."""
    s = make_syncer(tmp_path)
    s.update_settings({"tick_confirm": False})
    s.watcher.supported = False
    s._depth_obstructed = lambda: []
    s._depth_ok = False
    fake_fetch(s)
    s.reader = FixedReader({"work": ["Fix dashboard"], "personal": []})
    assert s.scan()["changed"] is True


def test_a_freshly_started_watcher_has_not_earned_a_settle_window(monkeypatch):
    """"Never seen anybody" is not the same as "clear for hours". A watcher
    that came up ten seconds ago cannot vouch for the preceding minute."""
    w = make_watcher(monkeypatch, FakePresenceAPI(occupancy=0.0))
    w.poll()
    assert w.clear_for_s("work") < 5
    assert w.blocked_boards(min_clear_s=60) == ["work", "personal"]
    w.started_t -= 600
    assert w.clear_for_s("work") > 595
    assert w.blocked_boards(min_clear_s=60) == []


# ---------------------------------------------------------------- framing
# The camera can't zoom or pan; the crop region is the zoom and pan, and the
# dashboard's framing editor changes it at runtime.

from wbsync import MODEL_MAX_PIXELS, board_split, model_scale  # noqa: E402


def frame_png(w=1920, h=1080, shade=200):
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", (w, h), (shade, shade, shade)).save(buf, format="PNG")
    return buf.getvalue()


def test_region_is_validated_against_the_frame(tmp_path):
    s = make_syncer(tmp_path)
    ok, bad = s.validate_settings({"region": [600, 380, 1280, 930]})
    assert ok == {"region": [600, 380, 1280, 930]} and not bad
    assert s.validate_settings({"region": "600,380,1280,930"})[0]["region"] == [600, 380, 1280, 930]
    for nonsense in ([0, 0, 2000, 500], [500, 500, 400, 900], [0, 0, 50, 50],
                     [1, 2, 3], "a,b,c,d", None):
        assert "region" in s.validate_settings({"region": nonsense})[1], nonsense


def test_saving_a_region_moves_the_live_crop_and_drops_the_baseline(tmp_path):
    """Diffing a new framing against the old baseline is meaningless — a
    same-size pan would not even trip the shape check."""
    import numpy as np
    s = make_syncer(tmp_path)
    np.save(s._baseline_path, np.zeros((5, 5), dtype=np.float32))
    out = s.update_settings({"region": [600, 380, 1280, 930], "split": 940})
    assert out["applied"] == {"region": [600, 380, 1280, 930], "split": 940}
    assert s.cfg.region == (600, 380, 1280, 930)
    assert not s._baseline_path.exists()
    # unrelated settings leave the baseline alone
    np.save(s._baseline_path, np.zeros((5, 5), dtype=np.float32))
    s.update_settings({"interval_s": 600})
    assert s._baseline_path.exists()


def test_saved_region_survives_a_restart_and_env_stays_the_default(tmp_path):
    s = make_syncer(tmp_path)
    s.update_settings({"region": [600, 380, 1280, 930]})
    s2 = make_syncer(tmp_path)
    assert s2.cfg.region == (600, 380, 1280, 930)
    assert s2.default_region == [0, 0, 10, 10]   # what WB_REGION said


def test_a_corrupt_saved_region_falls_back_to_the_default(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"region": [5, 5, 1, 1],
                                                        "interval_s": 600}))
    s = make_syncer(tmp_path)
    assert s.cfg.region == (0, 0, 10, 10)
    assert s.settings["interval_s"] == 600


def test_board_split_defaults_to_the_midpoint():
    assert board_split((600, 280, 1320, 810)) == 960
    assert board_split((600, 280, 1320, 810), 940) == 940
    assert board_split((600, 280, 1320, 810), 1500) == 960   # outside the box


def test_watcher_divides_the_boards_at_the_saved_split(monkeypatch):
    w = make_watcher(monkeypatch, FakePresenceAPI(),
                     settings={"presence_occupancy": 0.015, "split": 940})
    regions = w.regions()
    assert regions["work"][2] == 940 + 120
    assert regions["personal"][0] == 940 - 120


def test_upscale_is_capped_to_what_the_model_context_holds():
    assert model_scale(720, 530, 2) == 2                       # today's default fits
    big = model_scale(680, 540, 3)
    assert big < 3 and 680 * 540 * big * big <= MODEL_MAX_PIXELS * 1.0001
    assert model_scale(1920, 1080, 1) < 1                      # whole frame shrinks


def test_framing_preview_renders_the_draft_and_touches_nothing(tmp_path, monkeypatch):
    s = make_syncer(tmp_path)
    s.reader = FixedReader({"work": ["a"], "work_done": [], "work_order": ["a"],
                            "personal": [], "personal_done": [], "personal_order": [],
                            "work_obstructed": False, "personal_obstructed": False})

    class R:
        content = frame_png()
        def raise_for_status(self):
            pass
    monkeypatch.setattr("wbsync.requests.get", lambda *a, **k: R())
    before = json.dumps(s.settings)
    code, out = s.framing_preview({"region": [600, 380, 1280, 930], "upscale": 2,
                                   "capture_format": "jpeg", "read": True})
    assert code == 200
    assert out["crop"] == [680, 550] and out["model"] == [1360, 1100]
    assert out["image"].startswith("data:image/jpeg;base64,")
    assert out["read"]["work"] == ["a"]
    assert json.dumps(s.settings) == before                    # draft, not saved
    assert s.cfg.region == (0, 0, 10, 10)
    assert not s.crop_path.exists() and not s._baseline_path.exists()


def test_framing_preview_rejects_a_bad_draft(tmp_path):
    s = make_syncer(tmp_path)
    code, out = s.framing_preview({"region": [0, 0, 5000, 10]})
    assert code == 400 and "region" in out["rejected"]


# -- 2026-10-26 review fixes ---------------------------------------------------
def test_one_tick_read_no_longer_parks_a_pending_item(tmp_path):
    """The silent pending->done path: ONE misread box used to park a real new
    item as done — no task, no log, no trace."""
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.update_settings({"ticked_to_complete": 2})
    s.reconcile({"work": ["Pay rent"], "personal": []})
    r = s.reconcile({"work": [], "work_done": ["Pay rent"], "personal": []})
    item = s.state["items"][0]
    assert item["status"] == "pending" and item["ticked"] == 1
    assert "parked" not in r and s.ha.calls == []
    # read open next: the tick was a misread — the task gets created
    r = s.reconcile({"work": ["Pay rent"], "personal": []})
    assert r["added"] == ["Pay rent"] and s.state["items"][0]["status"] == "open"


def test_pending_item_ticked_on_consecutive_reads_parks_and_says_so(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.update_settings({"ticked_to_complete": 2})
    s.reconcile({"work": ["quick job"], "personal": []})
    s.reconcile({"work": [], "work_done": ["quick job"], "personal": []})
    r = s.reconcile({"work": [], "work_done": ["quick job"], "personal": []})
    assert r["parked"] == ["quick job"] and s.ha.calls == []
    assert s.state["items"][0]["status"] == "done"


def test_first_seen_ticked_needs_the_streak_too(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.update_settings({"ticked_to_complete": 2})
    s.reconcile({"work": [], "work_done": ["new thing"], "personal": []})
    assert s.state["items"][0]["status"] == "pending"
    s.reconcile({"work": ["new thing"], "personal": []})     # tick was a misread
    assert s.state["items"][0]["status"] == "open"
    assert s.ha.calls[-1][:2] == ("todoist", "new_task")


def test_pending_tick_triggers_the_confirm_photo(tmp_path):
    s = make_syncer(tmp_path, sightings_to_add=2)
    s.update_settings({"ticked_to_complete": 2})
    s.reconcile({"work": ["quick job"], "personal": []})
    s.reconcile({"work": [], "work_done": ["quick job"], "personal": []})
    assert s._tick_completion_imminent({"work_done": ["quick job"]})


class OrphanHA(FakeHA):
    """HA whose to-do list no longer has the task (done in Todoist directly)."""
    def __init__(self, open_now):
        super().__init__()
        self.open_now = open_now

    def call(self, domain, service, data):
        self.calls.append((domain, service, data))
        return service != "update_item"     # completions 400

    def open_items(self, entity):
        return self.open_now


def test_completion_of_a_task_gone_from_todoist_settles_instead_of_retrying(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Call Bright Shores"], "personal": []})
    s.ha = OrphanHA(open_now=["something else"])
    s.reconcile({"work": [], "personal": []})
    r = s.reconcile({"work": [], "personal": []})
    assert r["failed"] == 0 and r["orphaned"] == ["Call Bright Shores"]
    assert s.state["items"] == []


def test_completion_failure_with_the_task_still_open_keeps_retrying(tmp_path):
    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Get medication"], "personal": []})
    s.ha = OrphanHA(open_now=["Get medication"])
    r = s.reconcile({"work": [], "work_done": ["Get medication"], "personal": []})
    assert r["failed"] == 1 and "orphaned" not in r
    assert s.state["items"][0]["status"] == "open"


def test_camera_off_scans_are_quiet_skips_not_errors(tmp_path):
    from wbsync import CameraOff

    s = make_syncer(tmp_path)

    def off(frames=None):
        raise CameraOff("scheduled off until 06:30")

    s._fetch_crop = off
    s.settings["depth_skip"] = False
    r = s.scan(force=True)
    assert r["ok"] and r["camera_off"] and s.counters["errors"] == 0
    assert s.counters["skips_camera_off"] == 1


def test_scheduler_records_the_night_window_once(tmp_path):
    s = make_syncer(tmp_path)      # reader=None: a real scan attempt would error
    s.watcher.camera_off = "scheduled off until 06:30"
    for _ in range(3):
        s.last_scan_t = 0
        s.tick()
    skips = [h for h in s.history if h.get("reason") == "camera_off"]
    assert len(skips) == 1 and s.counters["errors"] == 0


def test_camera_off_detail_reads_kinect_knobs_503():
    from wbsync import camera_off_detail

    class R:
        status_code = 503

        def json(self):
            return {"error": "camera off", "camera": "off",
                    "detail": "scheduled off until 06:30"}

    assert camera_off_detail(R()) == "scheduled off until 06:30"
    R.status_code = 200
    assert camera_off_detail(R()) is None


def test_presence_without_fresh_readings_is_not_clear(tmp_path):
    s = make_syncer(tmp_path)
    s.watcher.supported = True       # worked once...
    s.watcher.boards = {}            # ...but has no current readings
    s._depth_obstructed = lambda: []
    s._depth_ok = False
    blocked, via = s._presence_blocked()
    assert blocked == [] and via == "none"


def test_empty_read_disputed_by_a_second_read_completes_nothing(tmp_path):
    s = make_syncer(tmp_path)
    texts = ["a one", "b two", "c three", "d four"]
    s.reconcile({"work": texts, "personal": []})
    s.reconcile({"work": [], "personal": []})               # miss 1
    reads = iter([{"work": [], "personal": []},             # empty read...
                  {"work": texts, "personal": []}])         # ...second disagrees

    class Seq:
        def read(self, jpeg):
            return next(reads)

    s.reader = Seq()
    fake_fetch(s)
    s.settings["tick_confirm"] = False
    r = s.scan(force=True)
    assert r["completed"] == [] and r["protected"] == ["work"]
    assert len(s.state["items"]) == 4
