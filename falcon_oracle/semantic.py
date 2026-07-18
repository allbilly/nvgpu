"""Semantic bootstrap path emitting the same normalized external events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bus.gk104_fake_adapter import (
    FB_PAUSE_CLEAR,
    FB_PAUSE_SET,
    GK104FakeBus,
    PAUSE_BIT,
)
from .bus.sparse_memory import SparseMemory
from .bus.types import XferRequest
from .devices.pmu_mmio import PmuMmioWindow, wr32_via_window
from .manifests import load_initial_dmem
from .trace import EventKind, TraceCollector, TraceLevel, canonical_trace_hash
from .values import u32


@dataclass
class SemanticResult:
    events: list
    vram: SparseMemory
    dmem: bytearray
    stop_reason: str
    trace_hash: str


def execute_semantic_bootstrap(
    corpus_dir: Path | str,
    *,
    bus: GK104FakeBus | None = None,
    trace_level: TraceLevel = TraceLevel.EXTERNAL,
) -> SemanticResult:
    """Host-side semantic model of the PMU pad's external effects."""
    corpus_dir = Path(corpus_dir)
    dmem = load_initial_dmem(corpus_dir, 4096)
    trace = TraceCollector(level=trace_level)
    inner = bus or GK104FakeBus(trace=trace, pause_ack_delay=1, pause_clear_delay=1)
    window = PmuMmioWindow(inner=inner, trace=None)
    window.seed_host(0x1620, 0x00000AAB)
    window.seed_host(0x26F0, 0x00000001)
    window.attach_dmem_provider(lambda addr, size: bytes(dmem[addr : addr + size]))

    # Pad ENTER preamble (simplified vs stock MEMX; see bringup enter-diff).
    wr32_via_window(window, 0x1620, 0x8)
    wr32_via_window(window, 0x26F0, 0x0)
    window.mmio_write32(FB_PAUSE_CLEAR, PAUSE_BIT)
    window.advance(1)
    window.mmio_write32(FB_PAUSE_SET, PAUSE_BIT)
    window.advance(1)

    status = window.mmio_read32(0x07C0)
    assert status & PAUSE_BIT, f"pause not set: {status:#x}"

    transfers = [
        (0x0D80, 0x60200, 16),
        (0x0D90, 0x40000, 8),
        (0x0DA0, 0x50000, 16),
    ]
    for src, dst, size in transfers:
        token = window.xfer_start(
            XferRequest(
                source_space="dmem",
                source_address=src,
                destination_space="direct_vram",
                destination_address=dst,
                size=size,
                direction="dmem_to_direct_vram",
                port=0,
            )
        )
        while True:
            st = window.xfer_poll(token)
            if st.phase.name == "COMPLETE":
                break
            window.advance(1)

    window.mmio_write32(FB_PAUSE_CLEAR, PAUSE_BIT)
    window.advance(1)
    wr32_via_window(window, 0x26F0, 0x1)
    wr32_via_window(window, 0x1620, 0xAAB)

    dmem[0x0D70:0x0D74] = u32(0x40C0B005).to_bytes(4, "little")
    trace.emit(EventKind.HALT, pc=0xBEF, value=0x40C0B005, metadata={"reason": "done"})

    return SemanticResult(
        events=list(trace.events),
        vram=inner.vram,
        dmem=dmem,
        stop_reason="done",
        trace_hash=canonical_trace_hash(trace.events),
    )
