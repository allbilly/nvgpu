"""Trace differential comparison."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .errors import DivergenceError
from .trace import EventKind, OracleEvent, externally_visible


class DivergenceClass(str, Enum):
    MISSING_EVENT = "MISSING_EVENT"
    EXTRA_EVENT = "EXTRA_EVENT"
    EVENT_KIND_MISMATCH = "EVENT_KIND_MISMATCH"
    ADDRESS_MISMATCH = "ADDRESS_MISMATCH"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    ORDER_MISMATCH = "ORDER_MISMATCH"
    WAIT_CONDITION_MISMATCH = "WAIT_CONDITION_MISMATCH"
    XFER_MODE_MISMATCH = "XFER_MODE_MISMATCH"
    BRANCH_MISMATCH = "BRANCH_MISMATCH"
    FINAL_STATE_MISMATCH = "FINAL_STATE_MISMATCH"
    STOP_REASON_MISMATCH = "STOP_REASON_MISMATCH"


@dataclass
class Divergence:
    klass: DivergenceClass
    index: int
    reference: OracleEvent | None
    candidate: OracleEvent | None
    detail: str


def _norm_meta(event: OracleEvent) -> OracleEvent:
    # Drop volatile helper metadata for comparison of external effects.
    meta = {
        k: v
        for k, v in event.metadata.items()
        if k in {"direction", "port", "source_space", "destination_space", "token", "name", "reason"}
    }
    return OracleEvent(
        sequence=event.sequence,
        kind=event.kind,
        pc=None if event.kind != EventKind.FAULT else event.pc,
        address=event.address,
        value=event.value,
        size=event.size,
        metadata=meta,
    )


def compare_traces(
    reference: Iterable[OracleEvent],
    candidate: Iterable[OracleEvent],
    *,
    external_only: bool = True,
    stop_at_first: bool = True,
) -> list[Divergence]:
    ref = list(reference)
    cand = list(candidate)
    if external_only:
        ref = externally_visible(ref)
        cand = externally_visible(cand)
    # Ignore pure MARKER noise unless both sides use the same markers.
    ref = [_norm_meta(e) for e in ref if e.kind != EventKind.MARKER]
    cand = [_norm_meta(e) for e in cand if e.kind != EventKind.MARKER]

    diffs: list[Divergence] = []
    n = max(len(ref), len(cand))
    for i in range(n):
        r = ref[i] if i < len(ref) else None
        c = cand[i] if i < len(cand) else None
        if r is None:
            diffs.append(Divergence(DivergenceClass.EXTRA_EVENT, i, r, c, "extra candidate event"))
        elif c is None:
            diffs.append(Divergence(DivergenceClass.MISSING_EVENT, i, r, c, "missing candidate event"))
        elif r.kind != c.kind:
            diffs.append(Divergence(DivergenceClass.EVENT_KIND_MISMATCH, i, r, c, f"{r.kind} != {c.kind}"))
        elif r.address != c.address:
            diffs.append(Divergence(DivergenceClass.ADDRESS_MISMATCH, i, r, c, f"{r.address} != {c.address}"))
        elif r.size != c.size and r.size is not None and c.size is not None:
            diffs.append(Divergence(DivergenceClass.SIZE_MISMATCH, i, r, c, f"{r.size} != {c.size}"))
        elif r.value != c.value and r.kind in {
            EventKind.MMIO_WRITE,
            EventKind.MMIO_READ,
            EventKind.XFER_START,
            EventKind.XFER_COMPLETE,
        }:
            # For XFER_START, value carries source address in our encoding.
            diffs.append(Divergence(DivergenceClass.VALUE_MISMATCH, i, r, c, f"{r.value} != {c.value}"))
        if diffs and stop_at_first:
            break
    return diffs


def format_divergence(
    div: Divergence,
    *,
    corpus: str | None = None,
    falcon_state: dict | None = None,
) -> str:
    lines = ["FIRST DIVERGENCE", ""]
    if corpus:
        lines += [f"Corpus:", f"  {corpus}", ""]
    if div.reference:
        r = div.reference
        lines += [
            "Semantic event:",
            f"  {r.kind.name} addr={r.address} value={r.value} size={r.size}",
            "",
        ]
    if div.candidate:
        c = div.candidate
        lines += [
            "Falcon event:",
            f"  {c.kind.name} addr={c.address} value={c.value} size={c.size}",
            "",
        ]
    lines.append(f"Class: {div.klass.value}")
    lines.append(f"Detail: {div.detail}")
    if falcon_state:
        lines.append("")
        lines.append("Falcon state:")
        for k, v in falcon_state.items():
            lines.append(f"  {k}={v}")
    return "\n".join(lines)


def assert_traces_equal(reference, candidate, **kwargs) -> None:
    diffs = compare_traces(reference, candidate, **kwargs)
    if diffs:
        raise DivergenceError(format_divergence(diffs[0]), details={"divergences": diffs})
