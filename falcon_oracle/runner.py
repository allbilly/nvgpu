"""High-level corpus runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bus.base import NullBus
from .bus.gk104_fake_adapter import GK104FakeBus
from .bus.scripted import ScriptedBus
from .bus.sparse_memory import SparseMemory
from .devices.pmu_mmio import PmuMmioWindow
from .executor import ExecutionResult, FalconExecutor
from .manifests import (
    load_image_memory,
    load_initial_dmem,
    load_symbols,
)
from .state import FalconMemory, FalconState
from .trace import TraceCollector, TraceLevel


def make_bootstrap_state(corpus_dir: Path | str) -> tuple[Any, FalconState]:
    corpus_dir = Path(corpus_dir)
    manifest, imem, _ = load_image_memory(corpus_dir)
    dmem = FalconMemory(
        load_initial_dmem(corpus_dir, manifest.dmem_size),
        name="dmem",
    )
    state = FalconState(pc=manifest.entry_address, imem=imem, dmem=dmem)
    return manifest, state


def _wrap_pmu_bus(bus: Any, *, seed_host: dict[int, int] | None = None) -> Any:
    """Attach the PMU MMIO window used by stock wr32/rd32 (call 0x34)."""
    if isinstance(bus, NullBus):
        return bus
    window = PmuMmioWindow(inner=bus, trace=None)
    seeds = {0x1620: 0x00000AAB, 0x26F0: 0x00000001}
    if seed_host:
        seeds.update(seed_host)
    for addr, val in seeds.items():
        window.seed_host(addr, val)
    return window


def execute_falcon_bootstrap(
    corpus_dir: Path | str,
    *,
    mode: str = "fake",
    scenario_yaml: str | None = None,
    trace_level: TraceLevel = TraceLevel.EXTERNAL,
    max_instructions: int = 100_000,
) -> ExecutionResult:
    corpus_dir = Path(corpus_dir)
    manifest, state = make_bootstrap_state(corpus_dir)
    symbols = load_symbols(corpus_dir)
    sym_map = {pc: name for pc, name in symbols.items()}
    trace = TraceCollector(level=trace_level)

    if mode == "cpu":
        bus: Any = NullBus()
    elif mode == "scripted":
        if not scenario_yaml:
            raise ValueError("scripted mode requires scenario_yaml")
        bus = _wrap_pmu_bus(ScriptedBus.from_yaml(scenario_yaml, trace=trace))
    else:
        bus = _wrap_pmu_bus(
            GK104FakeBus(
                vram=SparseMemory(default=0x00),
                trace=trace,
                pause_ack_delay=1,
                pause_clear_delay=1,
                xfer_complete_after_polls=1,
            )
        )

    exe = FalconExecutor(
        state=state,
        bus=bus,
        trace=trace,
        symbols=sym_map,
        done_pc=manifest.done_pc,
        done_magic_addr=manifest.done_magic_addr,
        done_magic_value=manifest.done_magic_value,
    )
    return exe.run(max_instructions=max_instructions)


def main(argv=None):
    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
