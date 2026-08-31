"""Minimal and null bus implementations."""

from __future__ import annotations

from ..errors import UnsupportedMMIO, UnsupportedXferMode, DeviceModelError
from .types import XferRequest, XferStatus


class NullBus:
    """CPU-only mode: any external access stops with UnsupportedMMIO/Xfer."""

    def mmio_read32(self, address: int) -> int:
        raise UnsupportedMMIO("CPU-only bus rejects MMIO read", details={"address": address})

    def mmio_write32(self, address: int, value: int) -> None:
        raise UnsupportedMMIO(
            "CPU-only bus rejects MMIO write",
            details={"address": address, "value": value},
        )

    def xfer_start(self, request: XferRequest) -> int:
        raise UnsupportedXferMode("CPU-only bus rejects XFER", details={"request": request})

    def xfer_poll(self, token: int) -> XferStatus:
        raise DeviceModelError("CPU-only bus has no XFER tokens", details={"token": token})

    def advance(self, logical_steps: int = 1) -> None:
        return
