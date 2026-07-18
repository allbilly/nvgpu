"""Append-only JSONL three-space traces and comparison/slicing helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


TRACE_FIELD_NAMES = (
  "sequence",
  "timestamp_ns",
  "phase",
  "cs",
  "ip",
  "instruction_bytes",
  "operation",
  "address_space",
  "direction",
  "width",
  "bdf",
  "bar",
  "raw_address",
  "canonical_offset",
  "value",
  "read_result",
  "firmware_service",
  "posted_flush",
  "dependency_tag",
)


@dataclass
class TraceEvent:
  sequence: int = 0
  timestamp_ns: int = 0
  phase: str = ""
  cs: int = 0
  ip: int = 0
  instruction_bytes: str = ""
  operation: str = ""
  address_space: str = ""  # pci | io | mmio | firmware | cpu | checkpoint
  direction: str = ""  # read | write | none
  width: int = 0
  bdf: str = ""
  bar: Optional[int] = None
  raw_address: int = 0
  canonical_offset: int = 0
  value: Optional[int] = None
  read_result: Optional[int] = None
  firmware_service: str = ""
  posted_flush: bool = False
  dependency_tag: str = ""

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "TraceEvent":
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Divergence:
  index: int
  reason: str
  left: Optional[TraceEvent]
  right: Optional[TraceEvent]

  def __str__(self) -> str:
    return f"divergence@{self.index}: {self.reason}"


@dataclass
class CompareResult:
  match: bool
  divergence: Optional[Divergence] = None
  left_count: int = 0
  right_count: int = 0
  normalized: bool = False
  categories: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class SliceReport:
  sink_index: int
  retained: List[TraceEvent]
  stable_producers: List[int]
  polling: List[int]
  volatile_reads: List[int]
  address_allocation: List[int]
  control_flow: List[int]


class TraceLog:
  """Append-only in-memory + optional JSONL sink."""

  def __init__(self, path: Optional[Path | str] = None) -> None:
    self.events: List[TraceEvent] = []
    self._path = Path(path) if path else None
    self._fp = None
    if self._path is not None:
      self._path.parent.mkdir(parents=True, exist_ok=True)
      self._fp = open(self._path, "a", encoding="utf-8")

  def close(self) -> None:
    if self._fp is not None:
      self._fp.close()
      self._fp = None

  def __enter__(self) -> "TraceLog":
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()

  def append(self, event: TraceEvent) -> TraceEvent:
    if event.timestamp_ns == 0:
      event.timestamp_ns = time.time_ns()
    event.sequence = len(self.events)
    self.events.append(event)
    if self._fp is not None:
      self._fp.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")
      self._fp.flush()
    return event

  def record(self, **kwargs: Any) -> TraceEvent:
    return self.append(TraceEvent(**kwargs))


def load_jsonl(path: Path | str) -> List[TraceEvent]:
  events: List[TraceEvent] = []
  with open(path, "r", encoding="utf-8") as fp:
    for line in fp:
      line = line.strip()
      if not line:
        continue
      events.append(TraceEvent.from_dict(json.loads(line)))
  return events


def dump_jsonl(events: Sequence[TraceEvent], path: Path | str) -> None:
  with open(path, "w", encoding="utf-8") as fp:
    for ev in events:
      fp.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")


def normalize_bar_addresses(
  events: Sequence[TraceEvent],
  *,
  bar_bases: Optional[Dict[int, int]] = None,
) -> List[TraceEvent]:
  """Cross-run normalization of assigned BAR addresses; retain raw_address."""
  bases = dict(bar_bases or {})
  # Infer BAR bases from first mmio events if not supplied.
  if not bases:
    for ev in events:
      if ev.address_space == "mmio" and ev.bar is not None and ev.raw_address:
        # raw = base + offset → base = raw - canonical_offset when offset known
        if ev.canonical_offset is not None:
          bases.setdefault(ev.bar, (ev.raw_address - ev.canonical_offset) & 0xFFFFFFFF)
  out: List[TraceEvent] = []
  for ev in events:
    d = ev.to_dict()
    if ev.address_space == "mmio" and ev.bar is not None and ev.bar in bases:
      base = bases[ev.bar]
      if ev.raw_address:
        d["canonical_offset"] = (ev.raw_address - base) & 0xFFFFFFFF
      # keep raw_address unchanged
    out.append(TraceEvent.from_dict(d))
  return out


def compare_traces(
  left: Sequence[TraceEvent],
  right: Sequence[TraceEvent],
  *,
  volatile_fields: Optional[Set[str]] = None,
  ignore_poll_repeats: bool = False,
  normalize_bars: bool = False,
) -> CompareResult:
  """Exact comparison with first-divergence reporting.

  No implicit width/address/ordering normalization.  Volatile read fields and
  poll-loop repetition must be declared explicitly.
  """
  volatile_fields = set(volatile_fields or ())
  a = list(normalize_bar_addresses(left) if normalize_bars else left)
  b = list(normalize_bar_addresses(right) if normalize_bars else right)

  if ignore_poll_repeats:
    a = _collapse_polls(a)
    b = _collapse_polls(b)

  n = min(len(a), len(b))
  for i in range(n):
    diff = _event_diff(a[i], b[i], volatile_fields)
    if diff:
      return CompareResult(
        match=False,
        divergence=Divergence(i, diff, a[i], b[i]),
        left_count=len(a),
        right_count=len(b),
        normalized=normalize_bars,
      )
  if len(a) != len(b):
    longer = "left" if len(a) > len(b) else "right"
    missing = a[n] if len(a) > len(b) else b[n]
    return CompareResult(
      match=False,
      divergence=Divergence(
        n,
        f"extra event on {longer}: {missing.operation} {missing.address_space}",
        a[n] if len(a) > n else None,
        b[n] if len(b) > n else None,
      ),
      left_count=len(a),
      right_count=len(b),
      normalized=normalize_bars,
    )
  return CompareResult(match=True, left_count=len(a), right_count=len(b), normalized=normalize_bars)


def _event_diff(a: TraceEvent, b: TraceEvent, volatile: Set[str]) -> Optional[str]:
  keys = (
    "operation", "address_space", "direction", "width", "bar",
    "canonical_offset", "bdf", "posted_flush", "firmware_service",
  )
  for k in keys:
    if getattr(a, k) != getattr(b, k):
      return f"field {k}: {getattr(a, k)!r} != {getattr(b, k)!r}"
  # Values / read results — respect volatile declarations.
  if "value" not in volatile and a.direction == "write" and a.value != b.value:
    return f"write value {a.value!r} != {b.value!r}"
  if "read_result" not in volatile and a.direction == "read" and a.read_result != b.read_result:
    return f"read_result {a.read_result!r} != {b.read_result!r}"
  if "raw_address" not in volatile and a.raw_address != b.raw_address:
    # Allow raw BAR address drift when canonical matches and raw was set.
    if a.address_space == "mmio" and a.canonical_offset == b.canonical_offset:
      pass
    elif a.raw_address and b.raw_address:
      return f"raw_address {a.raw_address:#x} != {b.raw_address:#x}"
  return None


def _collapse_polls(events: Sequence[TraceEvent]) -> List[TraceEvent]:
  """Collapse consecutive identical reads (declared poll-loop repetition)."""
  out: List[TraceEvent] = []
  for ev in events:
    if (
      out
      and ev.direction == "read"
      and out[-1].direction == "read"
      and out[-1].address_space == ev.address_space
      and out[-1].canonical_offset == ev.canonical_offset
      and out[-1].width == ev.width
      and out[-1].bar == ev.bar
      and out[-1].operation == ev.operation
    ):
      continue
    out.append(ev)
  return out


def backward_slice(
  events: Sequence[TraceEvent],
  sink_index: int,
  *,
  volatile_offsets: Optional[Set[Tuple[str, int]]] = None,
) -> SliceReport:
  """Retain causal predecessors of the first successful PRAMIN checkpoint.

  Keeps writes, branch-controlling reads, PCI/BAR setup, posting reads,
  and delays.  Does not minimize before the producer works.
  """
  volatile_offsets = set(volatile_offsets or ())
  if sink_index < 0 or sink_index >= len(events):
    raise IndexError(f"sink_index {sink_index} out of range")

  retained_idx: List[int] = []
  stable: List[int] = []
  polling: List[int] = []
  volatile_reads: List[int] = []
  addr_alloc: List[int] = []
  control_flow: List[int] = []

  # Inclusive prefix through sink — complete first producer prefix.
  for i in range(sink_index + 1):
    ev = events[i]
    retained_idx.append(i)
    key = (ev.address_space, ev.canonical_offset)
    if ev.direction == "write":
      stable.append(i)
      if ev.address_space == "pci" and 0x10 <= (ev.canonical_offset & 0xFF) <= 0x24:
        addr_alloc.append(i)
    elif ev.direction == "read":
      if key in volatile_offsets:
        volatile_reads.append(i)
      elif ev.dependency_tag == "poll" or (
        i > 0
        and events[i - 1].direction == "read"
        and events[i - 1].canonical_offset == ev.canonical_offset
        and events[i - 1].address_space == ev.address_space
      ):
        polling.append(i)
      else:
        stable.append(i)
    elif ev.address_space in ("firmware", "cpu") or ev.operation in ("delay", "checkpoint"):
      if ev.operation == "checkpoint":
        control_flow.append(i)
      else:
        stable.append(i)
    if ev.posted_flush:
      stable.append(i)

  return SliceReport(
    sink_index=sink_index,
    retained=[events[i] for i in retained_idx],
    stable_producers=sorted(set(stable)),
    polling=polling,
    volatile_reads=volatile_reads,
    address_allocation=addr_alloc,
    control_flow=control_flow,
  )


def find_pramin_live_index(
  events: Sequence[TraceEvent],
  *,
  pramin_offset: int = 0x700000,
) -> Optional[int]:
  for i, ev in enumerate(events):
    if (
      ev.operation == "checkpoint"
      and ev.dependency_tag in ("pramin-live", "pramin")
    ):
      return i
    if (
      ev.address_space == "mmio"
      and ev.direction == "read"
      and ev.canonical_offset == pramin_offset
      and ev.read_result is not None
      and ev.read_result != 0xFFFFFFFF
      and (ev.read_result & 0xFFFFFF00) != 0xBAD0FB00
    ):
      return i
  return None
