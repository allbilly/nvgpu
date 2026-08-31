"""Nouveau vs Python cold lifecycle matrix (progress.md H77 / roadmap §8).

Offline checklist only: requires an observed register/write consequence before
the next stage is allowed in a *plan*. Never programs the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Nouveau-ish cold ownership order (progress.md roadmap item 8 + H77).
NOUVEAU_STAGES = (
    "devinit_preinit",  # e.g. VGA unlock CR3f=0x57
    "devinit_post",  # BIT-I / nvbios_post
    "devinit_init",
    "fb_oneinit",
    "fb_init",
    "ram_init",
    "instmem",
    "bar2_bootstrap",  # consumes existing PRAMIN
    "bar1",
    "pmu",  # Nouveau: after FB/RAM relative to Python's early PMU
    "gr_oneinit_pgob",
)

# Documented Python cold order skew (H77).
PYTHON_SKEW = (
    "pmc_gr_reset_enable",
    "devinit_post",
    "pmu_pgob",
    "ram_init",
)


@dataclass
class StageGate:
    stage: str
    required_observation: str
    status: str  # pending | observed | blocked


@dataclass
class LifecycleReport:
    ok: bool
    stages: list[StageGate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def default_stage_requirements() -> list[tuple[str, str]]:
    return [
        ("devinit_preinit", "CR3f unlock or equivalent preinit write observed"),
        ("devinit_post", "BIT-I scripts completed; 0x2240c marker optional (not PRAMIN)"),
        ("fb_oneinit", "FB oneinit consequence observed"),
        ("ram_init", "train nibbles / timing programmed (not a PRAMIN activator)"),
        ("instmem", "instmem acquisition observed"),
        ("bar2_bootstrap", "PRAMIN already live (H70) — bootstrap consumes, does not create"),
        ("bar1", "VM-disabled physical BAR1 non-stub before enable experiments"),
        ("pmu", "PMU path only after POST ownership understood"),
    ]


def evaluate_lifecycle(
    observed: dict[str, bool] | None = None,
    *,
    prefer_nouveau_order: bool = True,
) -> LifecycleReport:
    """Score which lifecycle stages have offline-recorded observations."""
    observed = observed or {}
    stages: list[StageGate] = []
    blocked = False
    for stage, req in default_stage_requirements():
        if blocked:
            status = "blocked"
        elif observed.get(stage):
            status = "observed"
        else:
            status = "pending"
            # bar2/bar1 block if PRAMIN not posted
            if stage in {"bar2_bootstrap", "bar1"} and not observed.get("pramin_live"):
                status = "blocked"
                blocked = True
        stages.append(StageGate(stage=stage, required_observation=req, status=status))

    notes = [
        "H77: Python enables GR/PMC then POST/PMU then RAM; Nouveau POSTs earlier and FB/RAM before PMU",
        "H70: bar2_bootstrap requires pre-existing PRAMIN — do not invent it with runtime MMIO",
        "Roadmap: require an observed consequence before spending another cold run on the next stage",
    ]
    if prefer_nouveau_order:
        notes.append(f"Prefer Nouveau stage order: {' → '.join(NOUVEAU_STAGES[:8])} …")
        notes.append(f"Avoid unexamined Python skew: {' → '.join(PYTHON_SKEW)}")

    ok = not any(s.status == "blocked" for s in stages if s.stage in {"bar2_bootstrap", "bar1"})
    # Matrix itself is informational; ok means no hard block without pramin_live claim.
    if not observed.get("pramin_live"):
        ok = False
    return LifecycleReport(ok=ok, stages=stages, notes=notes)


def format_lifecycle(report: LifecycleReport) -> str:
    lines = [
        "Kepler lifecycle matrix (offline)",
        f"  pramin_gate: {'OPEN' if report.ok else 'CLOSED (need H70 posted PRAMIN)'}",
        "  stages:",
    ]
    for s in report.stages:
        lines.append(f"    [{s.status:8}] {s.stage}: {s.required_observation}")
    lines.append("  notes:")
    for n in report.notes:
        lines.append(f"    - {n}")
    return "\n".join(lines)


def validate_h76_sequence(steps: list[dict[str, Any]]) -> list[str]:
    """Structural checks for the H76 INIT_IO sequence fixture."""
    errors: list[str] = []
    if not steps:
        return ["empty H76 sequence"]
    addrs = [str(s.get("addr", "")).lower() for s in steps if s.get("op") in {"mask", "wr32"}]
    if "0x614100" not in addrs and "0x00614100" not in addrs:
        # accept 0x614100 forms
        if not any(a.endswith("614100") for a in addrs):
            errors.append("missing 0x614100 ops")
    if not any(s.get("op") == "port_rmw" and int(str(s.get("port", "0")), 0) == 0x3C3 for s in steps):
        errors.append("missing final port_rmw 0x3c3")
    pmc = [s for s in steps if str(s.get("addr", "")).lower() in {"0x200", "0x000200", "0x00000200"}]
    if len(pmc) < 2:
        errors.append("expected PMC_ENABLE (0x200) bit30 clear then set")
    sleeps = [s for s in steps if s.get("op") == "sleep_ms"]
    if len(sleeps) < 2:
        errors.append("expected two 10ms sleeps")
    return errors
