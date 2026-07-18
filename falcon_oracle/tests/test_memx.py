"""MEMX semantic script executor + fixture ladder (plan.md §18.2)."""

from pathlib import Path

import pytest

from falcon_oracle.errors import DeviceModelError
from falcon_oracle.memx_script import (
    MemxCommand,
    execute_memx_fixture,
    execute_memx_script,
    load_memx_fixture,
)
from falcon_oracle.semantic_memx import execute_semantic_memx_enter_leave

FIXTURES = Path(__file__).resolve().parents[1] / "corpus" / "memx" / "fixtures"
MUTATIONS = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize(
    "name",
    [
        "01_empty.json",
        "02_one_wr32.json",
        "03_two_wr32.json",
        "04_wr32_group.json",
        "05_wait_success.json",
        "06_wait_timeout.json",
        "07_delay.json",
        "08_enter_leave.json",
        "09_enter_wr32_leave.json",
    ],
)
def test_fixture_ladder_runs(name: str):
    result = execute_memx_fixture(FIXTURES / name)
    assert result.stop_reason == "done"
    assert not result.in_enter


def test_wait_success_satisfied():
    result = execute_memx_fixture(FIXTURES / "05_wait_success.json")
    assert result.wait_satisfied == [True]


def test_wait_timeout_unsatisfied():
    result = execute_memx_fixture(FIXTURES / "06_wait_timeout.json")
    assert result.wait_satisfied == [False]


def test_wr32_group_serial_host_state():
    result = execute_memx_fixture(FIXTURES / "04_wr32_group.json")
    assert result.host[0x10F808] == 0x72A00000
    assert result.host[0x10F80C] == 0x1
    assert result.host[0x10F810] == 0x2


def test_enter_leave_matches_semantic_host():
    script = execute_memx_fixture(FIXTURES / "08_enter_leave.json")
    semantic = execute_semantic_memx_enter_leave()
    assert script.host[0x1620] == semantic.host[0x1620]
    assert script.host[0x26F0] == semantic.host[0x26F0]


def test_mutation_wr32_outside_enter():
    commands, seed, meta = load_memx_fixture(MUTATIONS / "mutation_wr32_outside_enter.json")
    assert meta["expect_error"]
    with pytest.raises(DeviceModelError, match="outside ENTER"):
        execute_memx_script(commands, seed_host=seed or None)


def test_mutation_wrong_wait_never_matches_then_leaves():
    # Historical: wrong training-status address → WAIT times out, script continues.
    result = execute_memx_script(
        [
            MemxCommand.enter(),
            MemxCommand.wait(0x10F900, 0xF, 0xA, 2000),  # seed default 0
            MemxCommand.leave(),
        ]
    )
    assert result.wait_satisfied == [False]
    assert result.stop_reason == "done"
    assert not result.in_enter
