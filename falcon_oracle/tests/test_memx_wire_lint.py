"""MEMX wire encode/decode and lint (progress.md H29 / ownership)."""

from pathlib import Path

import pytest

from falcon_oracle.bringup import cold_gate
from falcon_oracle.memx_lint import lint_memx_commands, lint_memx_fixture
from falcon_oracle.memx_script import MemxCommand, load_memx_fixture
from falcon_oracle.memx_wire import decode_memx_words, encode_memx_packets, encode_memx_words

FIXTURES = Path(__file__).resolve().parents[1] / "corpus" / "memx" / "fixtures"
CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "pmu_bar1_bootstrap"
MUTATIONS = Path(__file__).resolve().parent / "fixtures"


def test_encode_enter_leave_headers():
    words = encode_memx_words([MemxCommand.enter(), MemxCommand.leave()])
    assert words == [0x00000001, 0x00000002]


def test_encode_wr32_coalesce():
    packets = encode_memx_packets(
        [
            MemxCommand.enter(),
            MemxCommand.wr32(0x10F808, 0x1),
            MemxCommand.wr32(0x10F80C, 0x2),
            MemxCommand.leave(),
        ]
    )
    assert [p.mthd for p in packets] == [1, 3, 2]
    assert packets[1].data == (0x10F808, 0x1, 0x10F80C, 0x2)
    assert packets[1].header == (4 << 16) | 3


def test_wait_and_delay_flush_separately():
    packets = encode_memx_packets(
        [
            MemxCommand.enter(),
            MemxCommand.wait(0x10F900, 0xF, 0xA, 500_000),
            MemxCommand.delay(1000),
            MemxCommand.leave(),
        ]
    )
    assert [p.mthd for p in packets] == [1, 4, 5, 2]
    assert packets[1].header == (4 << 16) | 4
    assert packets[2].header == (1 << 16) | 5


def test_roundtrip_fixture_ladder():
    for name in sorted(FIXTURES.glob("*.json")):
        commands, _seed, _meta = load_memx_fixture(name)
        words = encode_memx_words(commands)
        back = decode_memx_words(words)

        def non_wr32(ops):
            return [c.op for c in ops if c.op.name != "WR32"]

        def wr32_flat(ops):
            flat = []
            for c in ops:
                if c.op.name == "WR32":
                    flat.extend(c.args)
            return flat

        # WR32 may coalesce across commands; other ops and pairs must match.
        assert non_wr32(back) == non_wr32(commands)
        assert wr32_flat(back) == wr32_flat(commands)


def test_lint_fixture_ladder_ok():
    for name in sorted(FIXTURES.glob("*.json")):
        report = lint_memx_fixture(name)
        assert report.ok, format_failures(name, report)


def format_failures(name, report):
    return f"{name}: " + "; ".join(f"{f.code}:{f.message}" for f in report.errors())


def test_lint_h29_wait_too_long():
    report = lint_memx_commands(
        [
            MemxCommand.enter(),
            MemxCommand.wait(0x10F900, 0xF, 0xA, 50_000_000),
            MemxCommand.leave(),
        ]
    )
    assert not report.ok
    assert any(f.code == "h29_wait_bound" for f in report.errors())


def test_lint_wr32_outside_enter():
    report = lint_memx_commands([MemxCommand.wr32(0x10F808, 1)])
    assert not report.ok
    assert any(f.code == "wr32_outside_enter" for f in report.errors())


def test_lint_enable_inside_enter_warning():
    report = lint_memx_commands(
        [
            MemxCommand.enter(),
            MemxCommand.wr32(0x1704, 0x80000001),
            MemxCommand.leave(),
        ]
    )
    assert report.ok  # warning only
    assert any(f.code == "enable_inside_enter" for f in report.findings)


def test_cold_gate_conditional_without_script():
    report = cold_gate(CORPUS)
    assert report.verdict == "CONDITIONAL"
    assert report.falcon_offline_ok
    assert any("H70" in r for r in report.reasons)


def test_cold_gate_with_good_fixture():
    report = cold_gate(CORPUS, fixture=FIXTURES / "08_enter_leave.json")
    assert report.verdict == "CONDITIONAL"
    assert report.script_ok is True


def test_cold_gate_rejects_bad_script():
    bad = MUTATIONS / "mutation_wr32_outside_enter.json"
    report = cold_gate(CORPUS, fixture=bad)
    assert report.verdict == "NO_GO"
    assert report.script_ok is False


def test_fixture_ladder_includes_night41f_order():
    assert (FIXTURES / "10_night41f_order.json").is_file()
