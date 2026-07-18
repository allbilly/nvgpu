"""Offline entry-probe and MEMX diff (no GPU)."""

from pathlib import Path

from falcon_oracle.bringup import cold_gate
from falcon_oracle.entry_probe import evaluate_entry_snapshot, load_entry_snapshot
from falcon_oracle.memx_diff import diff_memx_commands, diff_memx_fixtures
from falcon_oracle.memx_lint import lint_memx_fixture
from falcon_oracle.memx_script import MemxCommand

SNAPSHOTS = Path(__file__).resolve().parents[1] / "corpus" / "snapshots"
FIXTURES = Path(__file__).resolve().parents[1] / "corpus" / "memx" / "fixtures"
CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "pmu_bar1_bootstrap"


def test_night41h_snapshot_is_cold_replug():
    report = evaluate_entry_snapshot(load_entry_snapshot(SNAPSHOTS / "night41h_replug_cold.json"))
    assert report.classification == "COLD_REPLUG"
    assert not report.posted_baseline_ok
    assert not report.allow_full_cold_replay


def test_stub_pramin_snapshot():
    report = evaluate_entry_snapshot(load_entry_snapshot(SNAPSHOTS / "stub_pramin_after_memif.json"))
    assert report.classification == "STUB_PRAMIN"


def test_posted_candidate_snapshot():
    report = evaluate_entry_snapshot(load_entry_snapshot(SNAPSHOTS / "posted_baseline_candidate.json"))
    assert report.classification == "POSTED_CANDIDATE"
    assert report.posted_baseline_ok


def test_cold_gate_no_go_on_cold_replug_snapshot():
    report = cold_gate(CORPUS, snapshot=SNAPSHOTS / "night41h_replug_cold.json")
    assert report.verdict == "NO_GO"
    assert report.entry_classification == "COLD_REPLUG"


def test_cold_gate_conditional_on_posted_candidate():
    report = cold_gate(CORPUS, snapshot=SNAPSHOTS / "posted_baseline_candidate.json")
    assert report.verdict == "CONDITIONAL"
    assert report.entry_classification == "POSTED_CANDIDATE"


def test_night41f_order_fixture_lints():
    report = lint_memx_fixture(FIXTURES / "10_night41f_order.json")
    assert report.ok
    assert any(f.code == "host_enable_after_leave" for f in report.findings)


def test_memx_diff_identical():
    assert not diff_memx_fixtures(FIXTURES / "08_enter_leave.json", FIXTURES / "08_enter_leave.json")


def test_memx_diff_finds_first_divergence():
    diffs = diff_memx_commands(
        [MemxCommand.enter(), MemxCommand.leave()],
        [MemxCommand.enter(), MemxCommand.wr32(1, 2), MemxCommand.leave()],
    )
    assert diffs
    assert diffs[0].kind in {"command", "length"}
