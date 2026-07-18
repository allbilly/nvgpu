"""Offline hypothesis board, lifecycle, and plan-check."""

from pathlib import Path

from falcon_oracle.hypotheses import list_hypotheses
from falcon_oracle.lifecycle import evaluate_lifecycle, validate_h76_sequence
from falcon_oracle.plan_check import check_plan_file
import json

BRINGUP = Path(__file__).resolve().parents[1] / "corpus" / "bringup"
PLANS = BRINGUP / "plans"


def test_hypothesis_board_has_h76_through_h80():
    ids = {h.id for h in list_hypotheses()}
    assert {"H52", "H70", "H75", "H76", "H77", "H78", "H79", "H80"} <= ids


def test_lifecycle_blocks_without_pramin():
    report = evaluate_lifecycle({})
    assert not report.ok
    assert any(s.stage == "bar1" and s.status == "blocked" for s in report.stages)


def test_lifecycle_opens_with_pramin_live():
    report = evaluate_lifecycle({"pramin_live": True, "devinit_post": True})
    assert report.ok
    assert any(s.stage == "devinit_post" and s.status == "observed" for s in report.stages)


def test_h76_sequence_fixture_valid():
    data = json.loads((BRINGUP / "h76_init_io_sequence.json").read_text())
    assert not validate_h76_sequence(data["steps"])


def test_plan_allows_h76_only():
    report = check_plan_file(PLANS / "night41q_h76_only.json")
    assert report.verdict == "ALLOW"
    assert report.ok


def test_plan_refuses_full_cold_on_replug():
    report = check_plan_file(PLANS / "bad_full_cold_on_replug.json")
    assert report.verdict == "REFUSE"
    assert any(f.code == "refuse_h57_replay" for f in report.findings)


def test_plan_refuses_h69():
    report = check_plan_file(PLANS / "bad_promote_619f04.json")
    assert report.verdict == "REFUSE"
    assert any(f.code.startswith("refuse_h69") for f in report.findings)
