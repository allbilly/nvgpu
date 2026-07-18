"""Semantic stock MEMX ENTER/LEAVE (Nouveau memx.fuc, GF119+ path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bus.gk104_fake_adapter import (
    FB_PAUSE_CLEAR,
    FB_PAUSE_SET,
    FB_PAUSE_STATUS,
    GK104FakeBus,
    PAUSE_BIT,
)
from .devices.pmu_mmio import PmuMmioWindow, rd32_via_window, wr32_via_window
from .trace import EventKind, TraceCollector, TraceLevel, canonical_trace_hash
from .values import u32


@dataclass
class MemxSemanticResult:
    events: list
    stop_reason: str
    trace_hash: str
    host: dict[int, int]


def execute_semantic_memx_enter_leave(
    *,
    seed_host: dict[int, int] | None = None,
    pause_ack_delay: int = 1,
    pause_clear_delay: int = 1,
) -> MemxSemanticResult:
    """Execute memx_func_enter + memx_func_leave host effects.

    Source: ref/linux/.../pmu/fuc/memx.fuc (GF119 / GK104 chipset path).
    """
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    inner = GK104FakeBus(
        trace=trace,
        pause_ack_delay=pause_ack_delay,
        pause_clear_delay=pause_clear_delay,
    )
    bus = PmuMmioWindow(inner=inner, trace=None)
    host = {
        0x1620: 0x00000AAB,
        0x26F0: 0x00000001,
    }
    if seed_host:
        host.update({u32(k): u32(v) for k, v in seed_host.items()})
    for addr, val in host.items():
        bus.seed_host(addr, val)
        # Also mirror into inner mmio so direct reads work if needed.
        inner._inner.mmio[addr] = val

    def host_rd(addr: int) -> int:
        val = rd32_via_window(bus, addr)
        # Keep seed mirror updated from window data.
        host[addr] = val
        # Re-emit as ordinary external read for diffing (window markers dropped).
        # rd32_via_window already caused inner read attempt; emit explicit:
        return val

    def host_wr(addr: int, val: int) -> None:
        wr32_via_window(bus, addr, val)
        host[addr] = u32(val)

    # --- memx_func_enter (non-GT215) ---
    # mov $r6 0x001620; imm32(~0xaa2); rd32; and; wr32
    r8 = host_rd(0x1620)
    r8 = u32(r8 & ~0x00000AA2)
    host_wr(0x1620, r8)
    # imm32(~1); rd32; and; wr32
    r8 = host_rd(0x1620)
    r8 = u32(r8 & ~0x00000001)
    host_wr(0x1620, r8)
    # mov $r6 0x0026f0; rd32; and ~1; wr32
    r8 = host_rd(0x26F0)
    r8 = u32(r8 & ~0x00000001)
    host_wr(0x26F0, r8)

    # pause SET + wait bit2
    bus.mmio_write32(FB_PAUSE_SET, PAUSE_BIT)
    bus.advance(pause_ack_delay)
    status = bus.mmio_read32(FB_PAUSE_STATUS)
    assert status & PAUSE_BIT, f"MEMX ENTER pause not set: {status:#x}"
    trace.emit(EventKind.MARKER, metadata={"phase": "enter_done"})

    # --- memx_func_leave ---
    bus.mmio_write32(FB_PAUSE_CLEAR, PAUSE_BIT)
    bus.advance(pause_clear_delay)
    status = bus.mmio_read32(FB_PAUSE_STATUS)
    assert not (status & PAUSE_BIT), f"MEMX LEAVE pause still set: {status:#x}"

    # 0x26f0 |= 1
    r8 = host_rd(0x26F0)
    host_wr(0x26F0, u32(r8 | 1))
    # 0x1620 |= 1; then |= 0xaa2
    r8 = host_rd(0x1620)
    host_wr(0x1620, u32(r8 | 1))
    r8 = host_rd(0x1620)
    host_wr(0x1620, u32(r8 | 0x00000AA2))
    trace.emit(EventKind.MARKER, metadata={"phase": "leave"})
    trace.emit(EventKind.HALT, metadata={"reason": "memx_enter_leave_done"})

    return MemxSemanticResult(
        events=list(trace.events),
        stop_reason="done",
        trace_hash=canonical_trace_hash(trace.events),
        host=host,
    )


def execute_semantic_memx_wr32_group(
    writes: list[tuple[int, int]],
    *,
    seed_host: dict[int, int] | None = None,
) -> MemxSemanticResult:
    """Serial WR32 group under an already-entered pause (caller must ENTER)."""
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    inner = GK104FakeBus(trace=trace)
    # Assume ENTER already done.
    inner.in_enter = True
    inner._seen_rising_edge = True
    inner._inner.mmio[FB_PAUSE_STATUS] = PAUSE_BIT
    bus = PmuMmioWindow(inner=inner, trace=None)
    host = dict(seed_host or {})
    for addr, val in host.items():
        bus.seed_host(addr, val)
    for addr, val in writes:
        wr32_via_window(bus, addr, val)
        host[u32(addr)] = u32(val)
    trace.emit(EventKind.HALT, metadata={"reason": "wr32_group_done"})
    return MemxSemanticResult(
        events=list(trace.events),
        stop_reason="done",
        trace_hash=canonical_trace_hash(trace.events),
        host=host,
    )
