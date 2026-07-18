"""Bring-up helpers tied to progress.md Night41."""

from pathlib import Path

from falcon_oracle.bringup import (
    diagnose_bringup,
    diff_pad_enter_vs_memx,
    explain_bootstrap,
)
from falcon_oracle.semantic_memx import execute_semantic_memx_enter_leave

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "pmu_bar1_bootstrap"


def test_explain_bootstrap_night41():
    report = explain_bootstrap(CORPUS)
    assert report.stop_reason == "done"
    assert report.xfer_count == 3
    assert report.rising_edge_pause
    assert report.loadback.ok
    assert report.loadback.matched_bytes == 40


def test_enter_diff_pad_vs_memx():
    ok, text = diff_pad_enter_vs_memx(CORPUS)
    assert ok
    assert "0x1620" in text
    assert "pause SET" in text


def test_memx_enter_leave_completes():
    result = execute_semantic_memx_enter_leave()
    assert result.stop_reason == "done"
    # After leave, stock path restores 0x1620 with 0xaa2|1 flavor.
    assert result.host[0x1620] & 0xAAB == 0xAAB or result.host[0x1620] & 0xAA2


def test_diagnose_marks_h52_offline_and_night41s_post_state():
    report = diagnose_bringup(CORPUS)
    assert report.falcon_offline_ok
    by_id = {h.id: h for h in report.hypotheses}
    assert by_id["H52"].status == "offline_ok"
    assert by_id["H70"].status == "needs_live"
    assert by_id["H73"].status == "secondary"
    assert by_id["H77"].status == "confirmed_post_boundary"
    assert by_id["H79"].status == "open"
    assert by_id["H80"].status == "confirmed"
    assert any("H79" in action or "0xa2ba" in action or "8fe8" in action.lower()
               for action in report.next_actions)


def test_cold_gate_never_unconditional_go():
    from falcon_oracle.bringup import cold_gate

    report = cold_gate(CORPUS)
    assert report.verdict == "CONDITIONAL"
    assert report.verdict != "GO"
