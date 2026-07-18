"""Diff two MEMX command streams (source-literal oracle guardrail).

Stops at the first divergence in opcode / args / wire words — the H41-style
check progress.md asked for without requiring a cold run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memx_script import MemxCommand, load_memx_fixture, parse_memx_commands
from .memx_wire import encode_memx_words


@dataclass
class MemxDiff:
    index: int
    kind: str  # command | wire | length
    left: str
    right: str


def _cmd_key(cmd: MemxCommand) -> str:
    args = ",".join(f"{a:#x}" for a in cmd.args)
    return f"{cmd.op.name}({args})" if args else cmd.op.name


def diff_memx_commands(
    left: Any,
    right: Any,
    *,
    stop_at_first: bool = True,
) -> list[MemxDiff]:
    a = parse_memx_commands(left)
    b = parse_memx_commands(right)
    diffs: list[MemxDiff] = []
    n = max(len(a), len(b))
    for i in range(n):
        if i >= len(a):
            diffs.append(MemxDiff(i, "length", "<missing>", _cmd_key(b[i])))
        elif i >= len(b):
            diffs.append(MemxDiff(i, "length", _cmd_key(a[i]), "<missing>"))
        elif a[i] != b[i]:
            diffs.append(MemxDiff(i, "command", _cmd_key(a[i]), _cmd_key(b[i])))
        if diffs and stop_at_first:
            return diffs

    wa = encode_memx_words(a)
    wb = encode_memx_words(b)
    if wa != wb:
        for i, (x, y) in enumerate(zip(wa, wb)):
            if x != y:
                diffs.append(MemxDiff(i, "wire", f"{x:#010x}", f"{y:#010x}"))
                if stop_at_first:
                    return diffs
        if len(wa) != len(wb):
            diffs.append(
                MemxDiff(
                    min(len(wa), len(wb)),
                    "wire",
                    f"len={len(wa)}",
                    f"len={len(wb)}",
                )
            )
    return diffs


def diff_memx_fixtures(
    left_path: Path | str,
    right_path: Path | str,
    *,
    stop_at_first: bool = True,
) -> list[MemxDiff]:
    left, _, _ = load_memx_fixture(left_path)
    right, _, _ = load_memx_fixture(right_path)
    return diff_memx_commands(left, right, stop_at_first=stop_at_first)


def format_memx_diff(diffs: list[MemxDiff]) -> str:
    if not diffs:
        return "OK: MEMX streams match"
    d = diffs[0]
    lines = [
        "MEMX stream divergence",
        f"  kind:  {d.kind}",
        f"  index: {d.index}",
        f"  left:  {d.left}",
        f"  right: {d.right}",
    ]
    if len(diffs) > 1:
        lines.append(f"  (+{len(diffs) - 1} more)")
    return "\n".join(lines)
