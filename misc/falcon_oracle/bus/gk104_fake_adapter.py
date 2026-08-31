"""Minimal GK104 fake adapter used by the oracle MVP.

The plan wants the project's existing GK104 fake extracted behind this
adapter.  Edits outside ``falcon_oracle/`` are forbidden for this
implementation pass, so the adapter embeds the pause/XFER rules the pad
needs and optionally wraps an external object if the caller supplies one.
"""

from __future__ import annotations

from typing import Any

from ..errors import DeviceModelError, UnsupportedMMIO, UnsupportedXferMode
from ..trace import EventKind, TraceCollector
from ..values import u32
from .scripted import ScriptedBus, TriggerRule
from .sparse_memory import SparseMemory
from .types import XferPhase, XferRequest, XferStatus


# Falcon IO addresses used by the PMU pad for FB pause ENTER/LEAVE.
FB_PAUSE_STATUS = 0x07C0
FB_PAUSE_SET = 0x07E0
FB_PAUSE_CLEAR = 0x07E4
PAUSE_BIT = 0x4


class GK104FakeBus:
    """Pause-edge + direct-VRAM XFER model matching pad expectations."""

    def __init__(
        self,
        *,
        vram: SparseMemory | None = None,
        trace: TraceCollector | None = None,
        pause_ack_delay: int = 1,
        pause_clear_delay: int = 1,
        xfer_complete_after_polls: int = 1,
        external: Any | None = None,
        require_enter_before_xfer: bool = True,
    ) -> None:
        self.external = external
        self.trace = trace
        self.require_enter_before_xfer = require_enter_before_xfer
        self.in_enter = False
        self._seen_rising_edge = False
        rules = [
            TriggerRule(
                kind="mmio_write",
                address=FB_PAUSE_SET,
                value=PAUSE_BIT,
                after_steps=pause_ack_delay,
                set_mmio={FB_PAUSE_STATUS: PAUSE_BIT},
            ),
            TriggerRule(
                kind="mmio_write",
                address=FB_PAUSE_CLEAR,
                value=PAUSE_BIT,
                after_steps=pause_clear_delay,
                set_mmio={FB_PAUSE_STATUS: 0},
            ),
        ]
        self._inner = ScriptedBus(
            initial_mmio={FB_PAUSE_STATUS: 0, FB_PAUSE_SET: 0, FB_PAUSE_CLEAR: 0},
            rules=rules,
            vram=vram or SparseMemory(default=0x00),
            xfer_complete_after_polls=xfer_complete_after_polls,
            trace=trace,
            strict_mmio=False,
        )

    @property
    def vram(self) -> SparseMemory:
        return self._inner.vram

    def attach_dmem_provider(self, reader) -> None:
        self._inner.attach_dmem_provider(reader)

    def mmio_read32(self, address: int) -> int:
        if self.external is not None and hasattr(self.external, "mmio_read32"):
            return int(self.external.mmio_read32(address))
        return self._inner.mmio_read32(address)

    def mmio_write32(self, address: int, value: int) -> None:
        address = u32(address)
        value = u32(value)
        if address == FB_PAUSE_SET and value == PAUSE_BIT:
            prev = self._inner.mmio.get(FB_PAUSE_STATUS, 0)
            if prev & PAUSE_BIT:
                # Sticky-high: not a rising edge.
                self._seen_rising_edge = False
            else:
                self._seen_rising_edge = True
                self.in_enter = True
        if address == FB_PAUSE_CLEAR and value == PAUSE_BIT:
            self.in_enter = False
        if self.external is not None and hasattr(self.external, "mmio_write32"):
            self.external.mmio_write32(address, value)
            return
        self._inner.mmio_write32(address, value)

    def xfer_start(self, request: XferRequest) -> int:
        if self.require_enter_before_xfer and not self.in_enter:
            raise DeviceModelError(
                "XFER outside ENTER/LEAVE ownership region",
                details={"request": request},
            )
        if self.require_enter_before_xfer and not self._seen_rising_edge:
            raise DeviceModelError(
                "XFER without rising-edge pause acknowledgement",
                details={"request": request},
            )
        if request.target_mode != "direct_vram":
            raise UnsupportedXferMode("VM-mode XFER not supported in MVP")
        return self._inner.xfer_start(request)

    def xfer_poll(self, token: int) -> XferStatus:
        return self._inner.xfer_poll(token)

    def advance(self, logical_steps: int = 1) -> None:
        self._inner.advance(logical_steps)

    def loadback(self, address: int, size: int) -> bytes:
        return self.vram.read(address, size)
