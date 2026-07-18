"""Decoded Falcon instruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecodedInstruction:
    pc: int
    size: int
    raw: bytes
    mnemonic: str
    operands: tuple[Any, ...]
    branch_target: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def form(self) -> str:
        return str(self.metadata.get("form", self.mnemonic))
