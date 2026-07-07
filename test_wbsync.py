"""Reconciliation and helpers — the logic that decides what hits Todoist."""
import json

import pytest

from wbsync import Config, Syncer, fuzzy_match, parse_region


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


def make_syncer(tmp_path, missing_to_complete=2):
    cfg = Config(
        ha_url="http://ha", ha_token="t", ollama_url="http://ollama",
        kinect_url="http://kk", region=(0, 0, 10, 10),
        presence_entity="person.x", todo_entity="todo.inbox",
        todoist_project="Inbox", interval_s=900, change_threshold=4.0,
        missing_to_complete=missing_to_complete, scan_when_home=False,
        model="qwen3-vl:8b-instruct", data_dir=tmp_path,
    )
    return Syncer(cfg, FakeHA(), reader=None)


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
    assert reloaded.state["items"] == [{"text": "Fix dashboard", "board": "work", "missing": 0}]


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
    assert s.state["items"] == [{"text": "call mom", "board": "personal", "missing": 0}]
