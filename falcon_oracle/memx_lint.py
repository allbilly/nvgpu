"""Offline MEMX script lint for Kepler bring-up (progress.md Night40/41).

Catches cold-run burners before EXEC:

- H29: WAIT nsec must be in 1..5_000_000 (50 ms WAIT wedged PMU)
- ENTER/LEAVE balance and WR32 ownership region
- Nouveau DMEM capacity (0x800-byte memx data segment)
- Optional Night41f root-order hints (ENTER before root stores, LEAVE before 0x1704)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .memx_script import MemxCommand, MemxOp, load_memx_fixture, parse_memx_commands
from .memx_wire import MEMX_DATA_WORDS, wire_footprint_words
from .values import u32

# progress.md H29 / nvbios_init.py KEPLER_TRAIN_WAIT_NS bound.
H29_WAIT_MAX_NS = 5_000_000
H29_WAIT_MIN_NS = 1

# Night41f VRAM root targets (progress.md).
NIGHT41F_ROOT_ADDRS = frozenset({0x60200, 0x40000, 0x50000, 0x60208})
BAR1_ENABLE_ADDR = 0x1704


@dataclass
class LintFinding:
    severity: str  # error | warning | note
    code: str
    message: str
    index: int | None = None


@dataclass
class LintReport:
    ok: bool
    findings: list[LintFinding] = field(default_factory=list)
    wire_words: int = 0
    command_count: int = 0
    wr32_words: int = 0

    def errors(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity == "error"]


def lint_memx_commands(commands: Any) -> LintReport:
    cmds = parse_memx_commands(commands)
    findings: list[LintFinding] = []
    in_enter = False
    enter_count = 0
    leave_count = 0
    wr32_words = 0
    saw_root_wr = False
    saw_enable = False
    root_before_enter = False
    enable_inside_enter = False

    for i, cmd in enumerate(cmds):
        if cmd.op is MemxOp.ENTER:
            if in_enter:
                findings.append(
                    LintFinding("error", "nested_enter", "ENTER while already inside ENTER", i)
                )
            in_enter = True
            enter_count += 1
        elif cmd.op is MemxOp.LEAVE:
            if not in_enter:
                findings.append(
                    LintFinding("error", "leave_without_enter", "LEAVE without matching ENTER", i)
                )
            in_enter = False
            leave_count += 1
        elif cmd.op is MemxOp.WR32:
            if not in_enter:
                findings.append(
                    LintFinding(
                        "error",
                        "wr32_outside_enter",
                        "WR32 outside ENTER/LEAVE ownership region",
                        i,
                    )
                )
            args = cmd.args
            wr32_words += len(args)
            for a, d in zip(args[0::2], args[1::2]):
                addr = u32(a)
                if addr in NIGHT41F_ROOT_ADDRS:
                    saw_root_wr = True
                    if enter_count == 0:
                        root_before_enter = True
                if addr == BAR1_ENABLE_ADDR:
                    saw_enable = True
                    if in_enter:
                        enable_inside_enter = True
                # Hostile absolute clears that Night40 blamed for BAR0 death when
                # issued incorrectly — note only; ENTER itself RMWs these.
                if addr in (0x1620, 0x26F0) and in_enter:
                    findings.append(
                        LintFinding(
                            "note",
                            "pause_reg_wr32",
                            f"WR32 {addr:#x}={u32(d):#x} inside ENTER "
                            "(stock MEMX ENTER/LEAVE also touch these)",
                            i,
                        )
                    )
        elif cmd.op is MemxOp.WAIT:
            nsec = u32(cmd.args[3]) if len(cmd.args) >= 4 else 0
            if not H29_WAIT_MIN_NS <= nsec <= H29_WAIT_MAX_NS:
                findings.append(
                    LintFinding(
                        "error",
                        "h29_wait_bound",
                        f"WAIT nsec={nsec} outside H29 bound "
                        f"{H29_WAIT_MIN_NS}..{H29_WAIT_MAX_NS} "
                        "(Night40aq: 50ms WAIT wedged PMU)",
                        i,
                    )
                )
            if not in_enter:
                findings.append(
                    LintFinding(
                        "warning",
                        "wait_outside_enter",
                        "WAIT outside ENTER (stock ram_wait runs inside atomic ENTER)",
                        i,
                    )
                )
        elif cmd.op is MemxOp.DELAY:
            nsec = u32(cmd.args[0]) if cmd.args else 0
            if nsec > H29_WAIT_MAX_NS:
                findings.append(
                    LintFinding(
                        "warning",
                        "long_delay",
                        f"DELAY nsec={nsec} > 5ms — legal but slow; H29 bound is for WAIT",
                        i,
                    )
                )

    if in_enter:
        findings.append(
            LintFinding("error", "unclosed_enter", "script ends inside ENTER (missing LEAVE)")
        )
    if enter_count != leave_count:
        findings.append(
            LintFinding(
                "error",
                "enter_leave_mismatch",
                f"ENTER count {enter_count} != LEAVE count {leave_count}",
            )
        )

    wire_words = wire_footprint_words(cmds)
    if wire_words > MEMX_DATA_WORDS:
        findings.append(
            LintFinding(
                "error",
                "dmem_capacity",
                f"wire footprint {wire_words} words exceeds memx data segment "
                f"{MEMX_DATA_WORDS} words (0x{MEMX_DATA_BYTES:x} bytes)",
            )
        )

    if root_before_enter:
        findings.append(
            LintFinding(
                "error",
                "roots_before_enter",
                "Night41f root VRAM WR32 before any ENTER",
            )
        )
    if enable_inside_enter:
        findings.append(
            LintFinding(
                "warning",
                "enable_inside_enter",
                "0x1704 BAR1 enable inside ENTER — Night41 order is LEAVE then enable",
            )
        )
    if saw_root_wr and not saw_enable:
        findings.append(
            LintFinding(
                "note",
                "roots_without_enable",
                "script writes Night41f roots but never 0x1704 (ok for MEMIF-only probe)",
            )
        )

    ok = not any(f.severity == "error" for f in findings)
    return LintReport(
        ok=ok,
        findings=findings,
        wire_words=wire_words,
        command_count=len(cmds),
        wr32_words=wr32_words,
    )


def lint_memx_fixture(path: Path | str) -> LintReport:
    commands, _seed, meta = load_memx_fixture(path)
    report = lint_memx_commands(commands)
    host_after = meta.get("host_after") or []
    if host_after:
        after = parse_memx_commands(host_after)
        for cmd in after:
            if cmd.op is MemxOp.WR32:
                for addr in cmd.args[0::2]:
                    if u32(addr) == BAR1_ENABLE_ADDR:
                        report.findings.append(
                            LintFinding(
                                "note",
                                "host_enable_after_leave",
                                "host_after enables 0x1704 after MEMX LEAVE (Night41 order)",
                            )
                        )
        # Reject enable buried inside MEMX when host_after already has it — already warned.
    report.ok = not any(f.severity == "error" for f in report.findings)
    return report


def format_lint(report: LintReport, *, path: str | None = None) -> str:
    title = f"MEMX lint ({path})" if path else "MEMX lint"
    lines = [
        title,
        f"  result:     {'OK' if report.ok else 'FAIL'}",
        f"  commands:   {report.command_count}",
        f"  wr32_words: {report.wr32_words}",
        f"  wire_words: {report.wire_words}/{MEMX_DATA_WORDS}",
    ]
    if report.findings:
        lines.append("  findings:")
        for f in report.findings:
            loc = f" @{f.index}" if f.index is not None else ""
            lines.append(f"    [{f.severity}] {f.code}{loc}: {f.message}")
    else:
        lines.append("  findings: (none)")
    return "\n".join(lines)
