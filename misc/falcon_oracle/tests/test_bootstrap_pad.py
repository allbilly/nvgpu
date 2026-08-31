"""End-to-end PMU bootstrap pad execution."""

from pathlib import Path

from falcon_oracle.bus.gk104_fake_adapter import GK104FakeBus
from falcon_oracle.diff import compare_traces
from falcon_oracle.executor import FalconExecutor
from falcon_oracle.manifests import load_image_memory, load_initial_dmem, load_symbols
from falcon_oracle.runner import _wrap_pmu_bus, execute_falcon_bootstrap
from falcon_oracle.semantic import execute_semantic_bootstrap
from falcon_oracle.state import FalconMemory, FalconState
from falcon_oracle.trace import TraceCollector, TraceLevel, canonical_trace_hash

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "pmu_bar1_bootstrap"


def test_bootstrap_pad_e2e_fake():
    result = execute_falcon_bootstrap(CORPUS, mode="fake", trace_level=TraceLevel.EXTERNAL)
    assert result.stop_reason == "done"
    assert result.state.dmem.read_u32(0xD70) == 0x40C0B005
    # Re-run to inspect VRAM via dedicated bus
    manifest, imem, _ = load_image_memory(CORPUS)
    dmem = FalconMemory(load_initial_dmem(CORPUS, manifest.dmem_size), name="dmem")
    state = FalconState(pc=manifest.entry_address, imem=imem, dmem=dmem)
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    inner = GK104FakeBus(trace=trace)
    bus = _wrap_pmu_bus(inner)
    FalconExecutor(
        state=state,
        bus=bus,
        trace=trace,
        symbols=load_symbols(CORPUS),
        done_pc=manifest.done_pc,
        done_magic_addr=manifest.done_magic_addr,
        done_magic_value=manifest.done_magic_value,
    ).run(max_instructions=100_000)

    assert inner.vram.read(0x60200, 16) == (CORPUS / "fragment_instance.bin").read_bytes()
    assert inner.vram.read(0x40000, 8) == (CORPUS / "fragment_pde.bin").read_bytes()
    assert inner.vram.read(0x50000, 16) == (CORPUS / "fragment_pte.bin").read_bytes()

    xfers = [e for e in trace.events if e.kind.name == "XFER_START"]
    assert len(xfers) == 3
    assert [e.size for e in xfers] == [16, 8, 16]
    assert [e.address for e in xfers] == [0x60200, 0x40000, 0x50000]


def test_bootstrap_deterministic_trace_hash():
    a = execute_falcon_bootstrap(CORPUS, mode="fake")
    b = execute_falcon_bootstrap(CORPUS, mode="fake")
    assert a.trace_hash == b.trace_hash
    assert a.trace_hash == canonical_trace_hash(a.events)


def test_semantic_vs_falcon():
    sem = execute_semantic_bootstrap(CORPUS)
    fal = execute_falcon_bootstrap(CORPUS, mode="fake")
    diffs = compare_traces(sem.events, fal.events, external_only=True, stop_at_first=True)
    assert not diffs, diffs[0].detail if diffs else ""
    assert fal.stop_reason == "done"
