"""Structured Falcon oracle failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FalconOracleError(Exception):
    message: str
    pc: int | None = None
    raw: bytes | None = None
    mnemonic: str | None = None
    registers: dict[str, int] | None = None
    last_external_event: Any | None = None
    source_symbol: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.message]
        if self.pc is not None:
            parts.append(f"pc=0x{self.pc:x}")
        if self.mnemonic:
            parts.append(f"insn={self.mnemonic}")
        if self.source_symbol:
            parts.append(f"symbol={self.source_symbol}")
        if self.raw is not None:
            parts.append(f"raw={self.raw.hex()}")
        return ": ".join(parts) if len(parts) > 1 else self.message


class DecodeError(FalconOracleError):
    pass


class UnsupportedInstruction(FalconOracleError):
    pass


class InvalidInstructionLength(FalconOracleError):
    pass


class InvalidIMEMAccess(FalconOracleError):
    pass


class InvalidDMEMAccess(FalconOracleError):
    pass


class InvalidAlignment(FalconOracleError):
    pass


class InvalidRegister(FalconOracleError):
    pass


class InvalidBranchTarget(FalconOracleError):
    pass


class DeviceModelError(FalconOracleError):
    pass


class UnsupportedMMIO(FalconOracleError):
    pass


class UnsupportedXferMode(FalconOracleError):
    pass


class XferTimeout(FalconOracleError):
    pass


class WaitTimeout(FalconOracleError):
    pass


class DivergenceError(FalconOracleError):
    pass
