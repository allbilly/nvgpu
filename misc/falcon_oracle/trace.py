"""Normalized oracle event trace."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Iterable


class EventKind(Enum):
    INSTRUCTION = auto()
    REGISTER_WRITE = auto()
    FLAG_WRITE = auto()
    DMEM_READ = auto()
    DMEM_WRITE = auto()
    MMIO_READ = auto()
    MMIO_WRITE = auto()
    XFER_START = auto()
    XFER_COMPLETE = auto()
    WAIT_BEGIN = auto()
    WAIT_END = auto()
    CALL = auto()
    RETURN = auto()
    SLEEP = auto()
    WAKE = auto()
    HALT = auto()
    FAULT = auto()
    MARKER = auto()


EXTERNAL_KINDS = frozenset(
    {
        EventKind.MMIO_READ,
        EventKind.MMIO_WRITE,
        EventKind.XFER_START,
        EventKind.XFER_COMPLETE,
        EventKind.SLEEP,
        EventKind.WAKE,
        EventKind.HALT,
        EventKind.FAULT,
        EventKind.MARKER,
    }
)

EFFECT_KINDS = EXTERNAL_KINDS | frozenset(
    {
        EventKind.REGISTER_WRITE,
        EventKind.FLAG_WRITE,
        EventKind.DMEM_WRITE,
        EventKind.CALL,
        EventKind.RETURN,
        EventKind.WAIT_BEGIN,
        EventKind.WAIT_END,
    }
)


@dataclass(frozen=True)
class OracleEvent:
    sequence: int
    kind: EventKind
    pc: int | None = None
    raw_instruction: bytes | None = None
    mnemonic: str | None = None
    address: int | None = None
    value: int | None = None
    size: int | None = None
    source_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceLevel(str, Enum):
    EXTERNAL = "external"
    EFFECTS = "effects"
    INSTRUCTIONS = "instructions"


class TraceCollector:
    def __init__(self, level: TraceLevel = TraceLevel.EXTERNAL) -> None:
        self.level = TraceLevel(level)
        self.events: list[OracleEvent] = []
        self._seq = 0

    def emit(self, kind: EventKind, **kwargs: Any) -> OracleEvent:
        if not self._accepted(kind):
            # Still allocate sequence for determinism only when accepted.
            # Rejected events are dropped entirely (not sequenced).
            return OracleEvent(sequence=-1, kind=kind, **kwargs)
        event = OracleEvent(sequence=self._seq, kind=kind, **kwargs)
        self._seq += 1
        self.events.append(event)
        return event

    def _accepted(self, kind: EventKind) -> bool:
        if self.level is TraceLevel.INSTRUCTIONS:
            return True
        if self.level is TraceLevel.EFFECTS:
            return kind in EFFECT_KINDS or kind is EventKind.INSTRUCTION
        return kind in EXTERNAL_KINDS

    def clear(self) -> None:
        self.events.clear()
        self._seq = 0


def event_to_dict(event: OracleEvent) -> dict[str, Any]:
    data = asdict(event)
    data["kind"] = event.kind.name
    if event.raw_instruction is not None:
        data["raw_instruction"] = event.raw_instruction.hex()
    # Drop empty optional fields for stable JSON.
    return {k: v for k, v in data.items() if v is not None and v != {} and v != []}


def events_to_jsonl(events: Iterable[OracleEvent]) -> str:
    lines = [
        json.dumps(event_to_dict(e), sort_keys=True, separators=(",", ":"))
        for e in events
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def events_from_jsonl(text: str) -> list[OracleEvent]:
    out: list[OracleEvent] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        raw = data.get("raw_instruction")
        out.append(
            OracleEvent(
                sequence=data["sequence"],
                kind=EventKind[data["kind"]],
                pc=data.get("pc"),
                raw_instruction=bytes.fromhex(raw) if raw else None,
                mnemonic=data.get("mnemonic"),
                address=data.get("address"),
                value=data.get("value"),
                size=data.get("size"),
                source_symbol=data.get("source_symbol"),
                metadata=data.get("metadata") or {},
            )
        )
    return out


def externally_visible(events: Iterable[OracleEvent]) -> list[OracleEvent]:
    return [e for e in events if e.kind in EXTERNAL_KINDS]


def trace_hash(events: Iterable[OracleEvent]) -> str:
    payload = events_to_jsonl(events).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_external_events(events: Iterable[OracleEvent]) -> list[OracleEvent]:
    """External events with stable renumbered sequences and no markers."""
    out: list[OracleEvent] = []
    for event in externally_visible(events):
        if event.kind is EventKind.MARKER:
            continue
        out.append(
            OracleEvent(
                sequence=len(out),
                kind=event.kind,
                pc=event.pc,
                raw_instruction=event.raw_instruction,
                mnemonic=event.mnemonic,
                address=event.address,
                value=event.value,
                size=event.size,
                source_symbol=event.source_symbol,
                metadata=dict(event.metadata),
            )
        )
    return out


def canonical_trace_hash(events: Iterable[OracleEvent]) -> str:
    return trace_hash(canonical_external_events(events))
