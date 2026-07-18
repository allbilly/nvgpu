"""Trace serialization and diff."""

from falcon_oracle.diff import compare_traces
from falcon_oracle.trace import (
    EventKind,
    OracleEvent,
    events_from_jsonl,
    events_to_jsonl,
    trace_hash,
)


def test_jsonl_roundtrip():
    events = [
        OracleEvent(sequence=0, kind=EventKind.MMIO_WRITE, address=0x7E0, value=4, size=4),
        OracleEvent(sequence=1, kind=EventKind.XFER_START, address=0x60200, value=0xD80, size=16),
    ]
    text = events_to_jsonl(events)
    back = events_from_jsonl(text)
    assert len(back) == 2
    assert back[0].kind is EventKind.MMIO_WRITE
    assert back[1].size == 16
    assert trace_hash(events) == trace_hash(back)


def test_diff_detects_value_mismatch():
    a = [OracleEvent(sequence=0, kind=EventKind.MMIO_WRITE, address=1, value=1, size=4)]
    b = [OracleEvent(sequence=0, kind=EventKind.MMIO_WRITE, address=1, value=2, size=4)]
    diffs = compare_traces(a, b, external_only=True)
    assert diffs and diffs[0].klass.value == "VALUE_MISMATCH"
