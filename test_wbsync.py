"""Reconciliation and helpers — the logic that decides what hits Todoist."""
import json

import pytest

from wbsync import Config, Syncer, fuzzy_match, parse_region


class FakeHA:
    def __init__(self):
        self.calls = []

    def call(self, domain, service, data):
        self.calls.append((domain, service, data))
        return True

    def state(self, entity):
        return "not_home"


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


def test_obstructed_scan_touches_nothing(tmp_path):
    import numpy as np

    s = make_syncer(tmp_path)
    s.reconcile({"work": ["Fix dashboard"], "personal": []})
    s.ha.calls.clear()

    class ObstructedReader:
        def read(self, jpeg):
            return {"obstructed": True, "work": [], "personal": []}

    s.reader = ObstructedReader()
    s._fetch_crop = lambda: (b"jpeg", np.zeros((5, 5), dtype=np.float32))
    result = s.scan(force=True)
    assert result["obstructed"] is True
    assert s.ha.calls == []                          # no completions fired
    assert s.state["items"][0]["missing"] == 0       # no miss counted
    assert not (tmp_path / "baseline.npy").exists()  # baseline untouched


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
