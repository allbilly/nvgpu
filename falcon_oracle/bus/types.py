"""Shared XFER / bus value types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol


class XferDirection(str, Enum):
    DMEM_TO_DIRECT_VRAM = "dmem_to_direct_vram"
    DIRECT_VRAM_TO_DMEM = "direct_vram_to_dmem"


class XferPhase(Enum):
    CREATED = auto()
    PENDING = auto()
    COMPLETE = auto()
    FAULT = auto()


@dataclass(frozen=True)
class XferRequest:
    source_space: str
    source_address: int
    destination_space: str
    destination_address: int
    size: int
    direction: str
    port: int
    target_mode: str = "direct_vram"


@dataclass
class XferStatus:
    token: int
    phase: XferPhase
    fault_reason: str | None = None


class FalconBus(Protocol):
    def mmio_read32(self, address: int) -> int: ...

    def mmio_write32(self, address: int, value: int) -> None: ...

    def xfer_start(self, request: XferRequest) -> int: ...

    def xfer_poll(self, token: int) -> XferStatus: ...

    def advance(self, logical_steps: int = 1) -> None: ...
