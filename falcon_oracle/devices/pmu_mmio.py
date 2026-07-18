"""PMU Falcon IO MMIO window (Nouveau kernel.fuc wr32/rd32).

On GK104 the PMU reaches host BAR0 registers through:

  I[0x07a0] MMIO_ADDR
  I[0x07a4] MMIO_DATA
  I[0x07ac] MMIO_CTRL  (OP + TRIGGER + STATUS)

Stock ``wr32`` / ``rd32`` and the ``nv_iowr`` macro clear ``$r0``.  The
bootstrap pad's ``call 0x34`` target is this helper, not a bare store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..trace import EventKind, TraceCollector
from ..values import u32

# Falcon IO addresses (NVKM_FALCON_UNSHIFTED_IO / GF119+).
MMIO_ADDR = 0x07A0
MMIO_DATA = 0x07A4
MMIO_CTRL = 0x07AC

CTRL_TRIGGER = 0x00010000
CTRL_STATUS = 0x00007000
CTRL_STATUS_IDLE = 0x00000000
CTRL_MASK_B32_0 = 0x000000F0
CTRL_OP_RD = 0x00000001
CTRL_OP_WR = 0x00000002

WINDOW_REGS = frozenset({MMIO_ADDR, MMIO_DATA, MMIO_CTRL})


class _InnerBus(Protocol):
    def mmio_read32(self, address: int) -> int: ...

    def mmio_write32(self, address: int, value: int) -> None: ...

    def advance(self, logical_steps: int = 1) -> None: ...


@dataclass
class PmuMmioWindow:
    """Wrap a bus so Falcon IO window accesses become host MMIO ops."""

    inner: Any
    trace: TraceCollector | None = None
    complete_immediately: bool = True
    addr: int = 0
    data: int = 0
    ctrl: int = 0
    pending_op: str | None = None
    host_reads: dict[int, int] = field(default_factory=dict)

    def seed_host(self, address: int, value: int) -> None:
        self.host_reads[u32(address)] = u32(value)

    def mmio_read32(self, address: int) -> int:
        address = u32(address)
        if address == MMIO_ADDR:
            return self.addr
        if address == MMIO_DATA:
            return self.data
        if address == MMIO_CTRL:
            # STATUS idle unless a deferred op is pending.
            if self.pending_op and not self.complete_immediately:
                return self.ctrl | CTRL_STATUS
            return (self.ctrl & ~CTRL_STATUS) | CTRL_STATUS_IDLE
        return self.inner.mmio_read32(address)

    def mmio_write32(self, address: int, value: int) -> None:
        address = u32(address)
        value = u32(value)
        if address == MMIO_ADDR:
            self.addr = value
            return
        if address == MMIO_DATA:
            self.data = value
            return
        if address == MMIO_CTRL:
            self.ctrl = value
            if value & CTRL_TRIGGER:
                self._fire(value)
            return
        self.inner.mmio_write32(address, value)

    def _fire(self, ctrl: int) -> None:
        op = ctrl & 0x3
        if op == CTRL_OP_WR:
            self.inner.mmio_write32(self.addr, self.data)
            self.host_reads[self.addr] = self.data
            if self.trace:
                self.trace.emit(
                    EventKind.MARKER,
                    address=self.addr,
                    value=self.data,
                    metadata={"pmu_mmio": "wr32", "ctrl": ctrl},
                )
        elif op == CTRL_OP_RD:
            if self.addr in self.host_reads:
                self.data = self.host_reads[self.addr]
            else:
                try:
                    self.data = u32(self.inner.mmio_read32(self.addr))
                except Exception:
                    self.data = 0
                self.host_reads[self.addr] = self.data
            if self.trace:
                self.trace.emit(
                    EventKind.MARKER,
                    address=self.addr,
                    value=self.data,
                    metadata={"pmu_mmio": "rd32", "ctrl": ctrl},
                )
        self.ctrl = (ctrl & ~CTRL_TRIGGER & ~CTRL_STATUS) | CTRL_STATUS_IDLE
        self.pending_op = None

    def xfer_start(self, request):
        return self.inner.xfer_start(request)

    def xfer_poll(self, token):
        return self.inner.xfer_poll(token)

    def advance(self, logical_steps: int = 1) -> None:
        if hasattr(self.inner, "advance"):
            self.inner.advance(logical_steps)

    def attach_dmem_provider(self, reader) -> None:
        if hasattr(self.inner, "attach_dmem_provider"):
            self.inner.attach_dmem_provider(reader)

    @property
    def vram(self):
        return getattr(self.inner, "vram", None)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def wr32_via_window(bus: Any, addr: int, data: int) -> None:
    """Perform one stock wr32 transaction through a (possibly wrapped) bus."""
    bus.mmio_write32(MMIO_ADDR, u32(addr))
    bus.mmio_write32(MMIO_DATA, u32(data))
    bus.mmio_write32(
        MMIO_CTRL,
        CTRL_OP_WR | CTRL_MASK_B32_0 | CTRL_TRIGGER,
    )


def rd32_via_window(bus: Any, addr: int) -> int:
    bus.mmio_write32(MMIO_ADDR, u32(addr))
    bus.mmio_write32(
        MMIO_CTRL,
        CTRL_OP_RD | CTRL_TRIGGER,
    )
    return u32(bus.mmio_read32(MMIO_DATA))
