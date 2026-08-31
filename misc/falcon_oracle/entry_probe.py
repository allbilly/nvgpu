"""Offline entry-probe evaluator (progress.md Night41h / H70 posted baseline).

Classifies a *recorded* BAR0 MMIO snapshot — never touches hardware.
Live bring-up should dump these fields once, then score them here before
spending another full cold path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .values import parse_int, u32

# Night41h five-field probe + H70 posted-baseline fields (progress.md).
REG_PMC_BOOT_0 = 0x000000
REG_DEVINIT_MARKER = 0x02240C
REG_GPC_STATUS = 0x409604
REG_BAR1_CFG = 0x001700
REG_PRAMIN_WORD = 0x700000  # via PRAMIN window; snapshot stores the visible word
REG_ROM_WINDOW = 0x619F04
REG_ROM_MIRROR = 0x088050

GK104_PMC_BOOT_0 = 0x0E4040A2
NIGHT41H_GPC_GATED = 0xBADF1200
PRAMIN_VIRGIN = 0xFFFFFFFF
PRAMIN_GOLDEN_HINT = 0x0000BEEF  # golden Nouveau pre-RAMMAP read class

# Sequential PBUS walk failure class (Night41g+).
BAD0FB_HI = 0xBAD0FB00


@dataclass
class FieldCheck:
    name: str
    address: int
    actual: int | None
    ok: bool
    detail: str


@dataclass
class EntryProbeReport:
    classification: str  # COLD_REPLUG | STUB_PRAMIN | POSTED_CANDIDATE | BAR0_DEAD | UNKNOWN
    allow_full_cold_replay: bool
    posted_baseline_ok: bool
    fields: list[FieldCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)


def _get(regs: dict[int, int], addr: int) -> int | None:
    if addr in regs:
        return u32(regs[addr])
    # allow sparse string-keyed dicts already normalized
    return None


def _is_bad0fb(val: int) -> bool:
    return (u32(val) & 0xFFFFFF00) == BAD0FB_HI


def _is_stub_pramin(val: int | None) -> bool:
    if val is None:
        return True
    v = u32(val)
    return v == PRAMIN_VIRGIN or _is_bad0fb(v)


def load_entry_snapshot(path: Path | str) -> dict[int, int]:
    data = json.loads(Path(path).read_text())
    raw = data.get("registers") or data.get("mmio") or data
    if not isinstance(raw, dict):
        raise ValueError("snapshot needs registers/mmio object")
    out: dict[int, int] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        out[u32(parse_int(k))] = u32(parse_int(v))
    return out


def evaluate_entry_snapshot(registers: dict[int, int] | dict[str, Any]) -> EntryProbeReport:
    """Score a snapshot against Night41h / H70 criteria (offline)."""
    regs: dict[int, int] = {}
    for k, v in registers.items():
        regs[u32(parse_int(k))] = u32(parse_int(v))

    fields: list[FieldCheck] = []
    notes: list[str] = []
    next_actions: list[str] = []

    pmc = _get(regs, REG_PMC_BOOT_0)
    marker = _get(regs, REG_DEVINIT_MARKER)
    gpc = _get(regs, REG_GPC_STATUS)
    bar1 = _get(regs, REG_BAR1_CFG)
    pramin = _get(regs, REG_PRAMIN_WORD)
    rom = _get(regs, REG_ROM_WINDOW)
    mirror = _get(regs, REG_ROM_MIRROR)

    bar0_ok = pmc == GK104_PMC_BOOT_0
    fields.append(
        FieldCheck(
            "PMC_BOOT_0",
            REG_PMC_BOOT_0,
            pmc,
            bar0_ok,
            f"want GK104 {GK104_PMC_BOOT_0:#x}" if pmc is not None else "missing",
        )
    )
    if pmc == 0xFFFFFFFF:
        return EntryProbeReport(
            classification="BAR0_DEAD",
            allow_full_cold_replay=False,
            posted_baseline_ok=False,
            fields=fields,
            notes=["PMC_BOOT_0=0xffffffff — BAR0/link dead; replug/power before anything"],
            next_actions=["Restore BAR0 (enclosure power), then re-sample entry probe only"],
        )

    fields.append(
        FieldCheck(
            "devinit_marker",
            REG_DEVINIT_MARKER,
            marker,
            marker is not None,
            "Night41h cold had 0; completed-devinit marker is not PRAMIN activation",
        )
    )
    fields.append(
        FieldCheck(
            "gpc_status",
            REG_GPC_STATUS,
            gpc,
            gpc is not None,
            f"Night41h gated sentinel {NIGHT41H_GPC_GATED:#x}",
        )
    )
    fields.append(
        FieldCheck(
            "bar1_cfg",
            REG_BAR1_CFG,
            bar1,
            bar1 is not None,
            "Night41h cold had 0",
        )
    )
    stub = _is_stub_pramin(pramin)
    fields.append(
        FieldCheck(
            "pramin_word",
            REG_PRAMIN_WORD,
            pramin,
            not stub,
            "non-stub required for H70 posted baseline "
            f"(reject {PRAMIN_VIRGIN:#x} and bad0fb*)",
        )
    )

    rom_bit3 = ((rom or 0) >> 3) & 1 if rom is not None else 0
    # H68 golden correlation: 0x619f04 ≈ 0xfffe09; H69: writing it alone is insufficient.
    shadow_base = (rom or 0) & 0xFFFFF000 if rom is not None else 0
    fields.append(
        FieldCheck(
            "rom_window",
            REG_ROM_WINDOW,
            rom,
            rom is not None and rom_bit3 == 1 and shadow_base != 0,
            "H70 wants bit3=1 and nonzero shadow base; H69: do not promote MMIO-only writes",
        )
    )
    if mirror is not None:
        fields.append(
            FieldCheck(
                "rom_mirror",
                REG_ROM_MIRROR,
                mirror,
                True,
                "H69 closed for tested BAR0 mirror writes — observe only",
            )
        )

    # Classification
    night41h = (
        bar0_ok
        and marker == 0
        and gpc == NIGHT41H_GPC_GATED
        and (bar1 or 0) == 0
        and pramin == PRAMIN_VIRGIN
    )
    posted = (
        bar0_ok
        and not stub
        and rom is not None
        and rom_bit3 == 1
        and shadow_base != 0
    )

    if night41h:
        classification = "COLD_REPLUG"
        notes.append(
            "Matches Night41h entry-only replug: BAR0 healthy, no POST, PRAMIN virgin"
        )
        notes.append("Electrical replug ≠ firmware POST (progress.md Night41h)")
        next_actions.extend(
            [
                "Do NOT replay full H57 cold path for PRAMIN activation",
                "Obtain posted baseline (boot/primary on x86, unbind without power loss) "
                "or capture option-ROM PCI/VGA/MMIO producer (H73/H75)",
            ]
        )
        allow = False
    elif bar0_ok and stub:
        classification = "STUB_PRAMIN"
        notes.append("BAR0 live but PRAMIN stub/virgin — same H57/H70 boundary")
        next_actions.append("Stop before another MEMIF-roots→physical-BAR1 replay")
        allow = False
    elif posted:
        classification = "POSTED_CANDIDATE"
        notes.append(
            "Snapshot meets H70 field predicates (bit3, shadow base, non-stub PRAMIN)"
        )
        notes.append(
            "Still verify VM-disabled physical BAR1 visibility before full bring-up"
        )
        next_actions.append(
            "If physical BAR1 is live, proceed to reuse exact MEMIF roots / channel bring-up"
        )
        allow = True  # allow continuing *runtime* bring-up, not "invent POST"
    else:
        classification = "UNKNOWN"
        notes.append("Incomplete snapshot or mixed state — fill Night41h five fields + 0x619f04")
        next_actions.append("Re-sample entry probe only (no NVINIT/RAM/PMU) and re-score")
        allow = False

    if rom == 0xFFFE09:
        notes.append("0x619f04=0xfffe09 matches golden correlation (H68); H69 says insufficient alone")

    return EntryProbeReport(
        classification=classification,
        allow_full_cold_replay=False,  # never greenlight full PRAMIN-activation replay
        posted_baseline_ok=posted,
        fields=fields,
        notes=notes,
        next_actions=next_actions,
    )


def format_entry_probe(report: EntryProbeReport) -> str:
    lines = [
        "Kepler entry probe (offline snapshot)",
        f"  classification:     {report.classification}",
        f"  posted_baseline:    {'OK' if report.posted_baseline_ok else 'NO'}",
        f"  full_cold_replay:   NEVER" if not report.allow_full_cold_replay else "  full_cold_replay:   ?",
        "  fields:",
    ]
    for f in report.fields:
        actual = f"{f.actual:#x}" if f.actual is not None else "<missing>"
        mark = "ok" if f.ok else "no"
        lines.append(f"    [{mark}] {f.name} @ {f.address:#x} = {actual} — {f.detail}")
    if report.notes:
        lines.append("  notes:")
        for n in report.notes:
            lines.append(f"    - {n}")
    if report.next_actions:
        lines.append("  next:")
        for a in report.next_actions:
            lines.append(f"    - {a}")
    return "\n".join(lines)
