"""Offline live-plan checker — refuse progress.md known-burners without touching GPU."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lifecycle import validate_h76_sequence


@dataclass
class PlanFinding:
    severity: str  # error | warning | note
    code: str
    message: str


@dataclass
class PlanReport:
    ok: bool
    verdict: str  # ALLOW | REFUSE | WARN
    findings: list[PlanFinding] = field(default_factory=list)
    plan_name: str | None = None


def load_plan(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def check_plan(plan: dict[str, Any], *, sequence: dict[str, Any] | None = None) -> PlanReport:
    """Score an intended cold/live plan against progress.md refuse rules."""
    findings: list[PlanFinding] = []
    name = plan.get("name")
    intent = str(plan.get("intent") or "")
    entry = str(plan.get("entry_assumption") or "")
    actions = {str(a) for a in (plan.get("actions") or [])}
    forbid = {str(a) for a in (plan.get("forbid") or [])}

    # --- hard refuses ---
    if intent == "full_cold_pramin_replay" or "full_cold_pramin_replay" in actions:
        if entry in {"COLD_REPLUG", "STUB_PRAMIN", ""}:
            findings.append(
                PlanFinding(
                    "error",
                    "refuse_h57_replay",
                    "Full cold PRAMIN/BAR1 replay on unposted entry burns a run "
                    "(progress.md: stop replaying full cold path for PRAMIN activation)",
                )
            )

    if intent == "write_619f04_activator" or "write_619f04" in actions or "write_619f04_activator" in actions:
        findings.append(
            PlanFinding(
                "error",
                "refuse_h69",
                "H69 closed: do not promote 0x619f04 / 0x088050 MMIO as PRAMIN activator",
            )
        )

    if "write_088050" in actions and "write_619f04" in actions:
        findings.append(
            PlanFinding("error", "refuse_h69_pair", "H69 pair write is disproven for PRAMIN"),
        )

    if intent == "reclock_cold_fb" or "reclock_before_compute" in actions:
        findings.append(
            PlanFinding(
                "error",
                "refuse_h44_h30",
                "Defer reclocking-only H44/H30 until compute works at 324 MHz",
            )
        )

    if "vga_alias_preamble_only" in actions or intent == "h74_vga_alias":
        findings.append(
            PlanFinding(
                "error",
                "refuse_h74",
                "H74 disproven: BAR0 VGA-alias preamble alone did not instantiate PRAMIN",
            )
        )

    # --- H76 historical bounded discriminator / regression shape ---
    if intent == "h76_init_io_only" or "probe_nouveau_init_io" in actions:
        bad = actions & {"ram_init", "pmu_memx", "gr_enable", "bar1_enable", "nvinit"}
        if bad:
            findings.append(
                PlanFinding(
                    "error",
                    "h76_not_alone",
                    f"H76 probe must run alone before other init; remove {sorted(bad)}",
                )
            )
        else:
            findings.append(
                PlanFinding(
                    "note",
                    "h76_ok",
                    "H76 INIT_IO-only shape is valid, but Night41q already closed it as a PBUS activator",
                )
            )
        if sequence is not None:
            errs = validate_h76_sequence(sequence.get("steps") or [])
            for e in errs:
                findings.append(PlanFinding("error", "h76_sequence", e))
        required_forbid = {"ram_init", "pmu_memx", "gr_enable"}
        missing = required_forbid - forbid
        if missing:
            findings.append(
                PlanFinding(
                    "warning",
                    "h76_forbid_incomplete",
                    f"plan should forbid {sorted(missing)} alongside H76-only probe",
                )
            )

    # --- H75 capture ---
    if intent == "h75_capture" or "capture_io_bar_prefix" in actions:
        spaces = set(plan.get("spaces") or [])
        need = {"pci_config", "x86_io", "bar0_mmio"}
        if spaces and not need.issubset(spaces):
            findings.append(
                PlanFinding(
                    "error",
                    "h75_spaces",
                    f"H75 needs {sorted(need)}; mmiotrace alone is insufficient (H72/H73)",
                )
            )
        elif not spaces:
            findings.append(
                PlanFinding(
                    "warning",
                    "h75_declare_spaces",
                    "Declare spaces: pci_config, x86_io, bar0_mmio for H75 capture",
                )
            )
        else:
            findings.append(
                PlanFinding("note", "h75_capture_ok", "H75 multi-space capture plan shape looks ok")
            )

    # --- H77 warning ---
    if plan.get("lifecycle_order") == "python_legacy" and not plan.get("accept_h77_risk"):
        findings.append(
            PlanFinding(
                "warning",
                "h77_order",
                "Python legacy order (GR/PMC→POST→PMU→RAM) differs from Nouveau; "
                "set accept_h77_risk or prefer the Nouveau-literal H77 lifecycle",
            )
        )

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    if errors:
        verdict = "REFUSE"
        ok = False
    elif warnings:
        verdict = "WARN"
        ok = True
    else:
        verdict = "ALLOW"
        ok = True

    return PlanReport(ok=ok, verdict=verdict, findings=findings, plan_name=name)


def check_plan_file(
    path: Path | str,
    *,
    sequence_path: Path | str | None = None,
) -> PlanReport:
    plan = load_plan(path)
    sequence = None
    if sequence_path is not None:
        sequence = json.loads(Path(sequence_path).read_text())
    elif plan.get("hypothesis") == "H76" or plan.get("intent") == "h76_init_io_only":
        default_seq = (
            Path(__file__).resolve().parent / "corpus" / "bringup" / "h76_init_io_sequence.json"
        )
        if default_seq.is_file():
            sequence = json.loads(default_seq.read_text())
    return check_plan(plan, sequence=sequence)


def format_plan_report(report: PlanReport) -> str:
    lines = [
        "Kepler live-plan check (offline — no GPU)",
        f"  plan:    {report.plan_name or '<unnamed>'}",
        f"  verdict: {report.verdict}",
    ]
    if report.findings:
        lines.append("  findings:")
        for f in report.findings:
            lines.append(f"    [{f.severity}] {f.code}: {f.message}")
    return "\n".join(lines)
