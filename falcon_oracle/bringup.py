"""Bring-up oriented helpers tied to examples_kepler/progress.md Night41+."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus.gk104_fake_adapter import FB_PAUSE_CLEAR, FB_PAUSE_SET, FB_PAUSE_STATUS, PAUSE_BIT
from .diff import compare_traces
from .runner import execute_falcon_bootstrap
from .semantic_memx import execute_semantic_memx_enter_leave
from .trace import EventKind, TraceLevel, canonical_external_events, externally_visible

# Night41f VRAM layout (progress.md): INST=0x60000, PGD=0x40000, SPT=0x50000.
# Pad stores instance root at INST+0x200.
NIGHT41F_TRANSFERS = (
    {"name": "instance", "dmem": 0x0D80, "vram": 0x60200, "size": 16},
    {"name": "pde", "dmem": 0x0D90, "vram": 0x40000, "size": 8},
    {"name": "pte", "dmem": 0x0DA0, "vram": 0x50000, "size": 16},
)


@dataclass
class LoadbackReport:
    ok: bool
    fragments: list[dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0
    matched_bytes: int = 0


@dataclass
class BootstrapExplain:
    stop_reason: str
    done_magic: int | None
    rising_edge_pause: bool
    xfer_count: int
    transfers: list[dict[str, Any]]
    loadback: LoadbackReport
    enter_mmio: list[dict[str, Any]]
    leave_mmio: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


def verify_loadback(vram, corpus_dir: Path | str) -> LoadbackReport:
    corpus_dir = Path(corpus_dir)
    fragments = []
    matched = 0
    total = 0
    ok = True
    for spec in NIGHT41F_TRANSFERS:
        expected = (corpus_dir / f"fragment_{spec['name']}.bin").read_bytes()
        assert len(expected) == spec["size"]
        actual = vram.read(spec["vram"], spec["size"])
        mism = [(i, expected[i], actual[i]) for i in range(len(expected)) if expected[i] != actual[i]]
        total += len(expected)
        matched += len(expected) - len(mism)
        if mism:
            ok = False
        fragments.append(
            {
                "name": spec["name"],
                "vram": spec["vram"],
                "size": spec["size"],
                "ok": not mism,
                "first_mismatch": (
                    {
                        "offset": mism[0][0],
                        "expected": mism[0][1],
                        "actual": mism[0][2],
                    }
                    if mism
                    else None
                ),
            }
        )
    return LoadbackReport(ok=ok, fragments=fragments, total_bytes=total, matched_bytes=matched)


def explain_bootstrap(corpus_dir: Path | str) -> BootstrapExplain:
    """Run the pad and summarize Night41-relevant outcomes."""
    from .bus.gk104_fake_adapter import GK104FakeBus
    from .executor import FalconExecutor
    from .manifests import load_image_memory, load_initial_dmem, load_symbols
    from .state import FalconMemory, FalconState
    from .trace import TraceCollector
    from .devices.pmu_mmio import PmuMmioWindow

    corpus_dir = Path(corpus_dir)
    manifest, imem, _ = load_image_memory(corpus_dir)
    dmem = FalconMemory(load_initial_dmem(corpus_dir, manifest.dmem_size), name="dmem")
    state = FalconState(pc=manifest.entry_address, imem=imem, dmem=dmem)
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    inner = GK104FakeBus(trace=trace)
    bus = PmuMmioWindow(inner=inner, trace=None)
    # Seed host regs that memx/pad RMW via wr32.
    bus.seed_host(0x1620, 0x00000AAB)
    bus.seed_host(0x26F0, 0x00000001)

    result = FalconExecutor(
        state=state,
        bus=bus,
        trace=trace,
        symbols=load_symbols(corpus_dir),
        done_pc=manifest.done_pc,
        done_magic_addr=manifest.done_magic_addr,
        done_magic_value=manifest.done_magic_value,
    ).run(max_instructions=100_000)

    events = externally_visible(result.events)
    xfers = [e for e in events if e.kind is EventKind.XFER_START]
    transfers = [
        {
            "vram": e.address,
            "dmem": e.value,
            "size": e.size,
        }
        for e in xfers
    ]

    # Rising edge: SET write while status was clear beforehand is modeled by fake.
    rising = bool(getattr(inner, "_seen_rising_edge", False))

    writes = [e for e in events if e.kind is EventKind.MMIO_WRITE]
    # Split roughly at first XFER.
    first_xfer_seq = xfers[0].sequence if xfers else 10**9
    enter_mmio = [
        {"address": e.address, "value": e.value}
        for e in writes
        if e.sequence < first_xfer_seq
    ]
    leave_mmio = [
        {"address": e.address, "value": e.value}
        for e in writes
        if e.sequence > (xfers[-1].sequence if xfers else -1)
    ]

    try:
        done_magic = state.dmem.read_u32(0x0D70)
    except Exception:
        done_magic = None

    loadback = verify_loadback(inner.vram, corpus_dir)
    notes = []
    if result.stop_reason != "done":
        notes.append(f"stop_reason={result.stop_reason} (want done)")
    if not rising:
        notes.append("no rising-edge pause observed (sticky-pause risk; see progress Night40)")
    if len(xfers) != 3:
        notes.append(f"expected 3 XFERs, got {len(xfers)}")
    if not loadback.ok:
        notes.append(
            f"loadback {loadback.matched_bytes}/{loadback.total_bytes} "
            "(Night41d required 40/40 before 0x1704 enable)"
        )
    else:
        notes.append("Night41f loadback 40/40 exact (H52-class MEMIF view)")

    # Address sanity vs Night41f
    expected_vram = [t["vram"] for t in NIGHT41F_TRANSFERS]
    got_vram = [t["vram"] for t in transfers]
    if got_vram != expected_vram:
        notes.append(f"VRAM targets {got_vram} != Night41f {expected_vram}")

    return BootstrapExplain(
        stop_reason=result.stop_reason,
        done_magic=done_magic,
        rising_edge_pause=rising,
        xfer_count=len(xfers),
        transfers=transfers,
        loadback=loadback,
        enter_mmio=enter_mmio,
        leave_mmio=leave_mmio,
        notes=notes,
    )


def format_explain(report: BootstrapExplain) -> str:
    lines = [
        "PMU BAR1 bootstrap explain (Night41 bring-up)",
        f"  stop:            {report.stop_reason}",
        f"  done magic:      {report.done_magic:#x}" if report.done_magic is not None else "  done magic:      <unset>",
        f"  rising-edge:     {report.rising_edge_pause}",
        f"  xfers:           {report.xfer_count}",
        f"  loadback:        {report.loadback.matched_bytes}/{report.loadback.total_bytes}",
        "  transfers:",
    ]
    for t in report.transfers:
        lines.append(f"    DMEM {t['dmem']:#x} -> VRAM {t['vram']:#x} size={t['size']}")
    lines.append("  enter MMIO:")
    for w in report.enter_mmio:
        lines.append(f"    W {w['address']:#x} = {w['value']:#x}")
    lines.append("  leave MMIO:")
    for w in report.leave_mmio:
        lines.append(f"    W {w['address']:#x} = {w['value']:#x}")
    if report.notes:
        lines.append("  notes:")
        for n in report.notes:
            lines.append(f"    - {n}")
    return "\n".join(lines)


@dataclass
class HypothesisNote:
    id: str
    status: str  # offline_ok | out_of_scope | needs_live | open
    summary: str


@dataclass
class DiagnoseReport:
    falcon_offline_ok: bool
    stop_reason: str
    loadback_ok: bool
    rising_edge: bool
    xfer_count: int
    enter_diff_ok: bool
    hypotheses: list[HypothesisNote] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def diagnose_bringup(corpus_dir: Path | str) -> DiagnoseReport:
    """Triage Kepler bring-up: what the oracle can prove offline vs H70/H73."""
    corpus_dir = Path(corpus_dir)
    pad = explain_bootstrap(corpus_dir)
    enter_ok, _ = diff_pad_enter_vs_memx(corpus_dir)

    falcon_ok = (
        pad.stop_reason == "done"
        and pad.loadback.ok
        and pad.rising_edge_pause
        and pad.xfer_count == 3
        and enter_ok
    )

    hypotheses = [
        HypothesisNote(
            id="H52",
            status="offline_ok" if pad.loadback.ok else "open",
            summary=(
                "PMU MEMIF loadback 40/40 after ENTER/xdst/LEAVE"
                if pad.loadback.ok
                else f"loadback {pad.loadback.matched_bytes}/{pad.loadback.total_bytes}"
            ),
        ),
        HypothesisNote(
            id="H41-B",
            status="offline_ok" if enter_ok else "open",
            summary="Pad and stock MEMX both issue rising-edge FB_PAUSE SET",
        ),
        HypothesisNote(
            id="H57",
            status="out_of_scope",
            summary="MEMIF-visible roots ≠ PBUS physical BAR1 — oracle cannot invent PBUS",
        ),
        HypothesisNote(
            id="H70",
            status="needs_live",
            summary=(
                "Golden pre-runtime producer remains a platform explanation, but "
                "Night41s proved corrected NVINIT POST can activate PRAMIN from cold."
            ),
        ),
        HypothesisNote(
            id="H73",
            status="secondary",
            summary=(
                "H70 refinement: PCI config / legacy VGA I/O / reset sequencing "
                "absent from mmiotrace — live capture required"
            ),
        ),
        HypothesisNote(
            id="H75",
            status="secondary",
            summary="Option-ROM I/O-BAR prefix contingency — live A/B, not Falcon simulation",
        ),
        HypothesisNote(
            id="H76",
            status="closed",
            summary=(
                "INIT_IO 0x3c3 sequence ported and cold-tested; "
                "did not instantiate PBUS/PRAMIN"
            ),
        ),
        HypothesisNote(
            id="H77",
            status="confirmed_post_boundary",
            summary="Night41s corrected NVINIT POST activated fixed-PA PRAMIN before RAM",
        ),
        HypothesisNote(
            id="H78",
            status="cold_supported",
            summary="NVINIT parity cluster enabled the successful POST; exact member remains open",
        ),
        HypothesisNote(
            id="H79",
            status="open",
            summary=(
                "Leading: activator in (0xa138,0xa43c] of 0x8fe8; next stop 0xa2ba"
            ),
        ),
        HypothesisNote(
            id="H80",
            status="confirmed",
            summary=(
                "Night41u reproduced end-only fixed-PA activation; "
                "Night41t mid-POST 0x1700 sampling was the virgin artifact"
            ),
        ),
    ]

    next_actions: list[str] = []
    notes = list(pad.notes)
    if falcon_ok:
        notes.append(
            "Falcon-offline path is green: do not burn cold runs re-proving H52 MEMIF"
        )
        next_actions.extend(
            [
                "Night41ab: PREFIX=2 STOP_OFFSET=0xa2ba --probe-nouveau-post-script-bisect (H79)",
                "Never mid-POST 0x1700 sample (H80 confirmed)",
                "Treat H75/H70/H73 as a secondary native-I/O/posted-baseline branch",
                "Use `memx-run` fixtures to validate MEMX script semantics before cold RAM EXEC",
                "Do not promote 0x619f04 / 0x088050 alone (H69 closed)",
            ]
        )
    else:
        next_actions.append("Fix Falcon-offline failures first (explain / loadback / enter-diff)")
        if not pad.loadback.ok:
            next_actions.append("Investigate MEMIF xdst / rising-edge pause before live BAR1")
        if not enter_ok:
            next_actions.append("Compare pad ENTER vs stock memx_func_enter (enter-diff)")

    return DiagnoseReport(
        falcon_offline_ok=falcon_ok,
        stop_reason=pad.stop_reason,
        loadback_ok=pad.loadback.ok,
        rising_edge=pad.rising_edge_pause,
        xfer_count=pad.xfer_count,
        enter_diff_ok=enter_ok,
        hypotheses=hypotheses,
        next_actions=next_actions,
        notes=notes,
    )


def format_diagnose(report: DiagnoseReport) -> str:
    lines = [
        "Kepler bring-up diagnose (progress.md Night41+)",
        f"  falcon_offline:  {'OK' if report.falcon_offline_ok else 'FAIL'}",
        f"  stop:            {report.stop_reason}",
        f"  rising-edge:     {report.rising_edge}",
        f"  xfers:           {report.xfer_count}",
        f"  loadback:        {'40/40' if report.loadback_ok else 'FAIL'}",
        f"  enter-diff:      {'OK' if report.enter_diff_ok else 'FAIL'}",
        "  hypotheses:",
    ]
    for h in report.hypotheses:
        lines.append(f"    [{h.status:12}] {h.id}: {h.summary}")
    if report.next_actions:
        lines.append("  next:")
        for a in report.next_actions:
            lines.append(f"    - {a}")
    if report.notes:
        lines.append("  notes:")
        for n in report.notes:
            lines.append(f"    - {n}")
    return "\n".join(lines)


@dataclass
class ColdGateReport:
    verdict: str  # NO_GO | CONDITIONAL
    falcon_offline_ok: bool
    script_ok: bool | None
    entry_classification: str | None = None
    posted_baseline_ok: bool | None = None
    reasons: list[str] = field(default_factory=list)
    diagnose: DiagnoseReport | None = None
    lint: Any = None
    entry: Any = None


def cold_gate(
    corpus_dir: Path | str,
    *,
    fixture: Path | str | None = None,
    snapshot: Path | str | None = None,
) -> ColdGateReport:
    """Gate a cold run: Falcon-offline must be green; PRAMIN still needs H70 POST.

    Never returns an unconditional GO for physical BAR1/PRAMIN — progress.md
    H70/H73 remain live prerequisites. Optional ``snapshot`` is a recorded
    MMIO dump (offline); this never touches the GPU.
    """
    from .entry_probe import evaluate_entry_snapshot, load_entry_snapshot
    from .memx_lint import lint_memx_fixture

    diag = diagnose_bringup(corpus_dir)
    reasons: list[str] = []
    hard_fail = False
    script_ok: bool | None = None
    lint_report = None
    entry_report = None
    entry_classification = None
    posted_ok = None

    if not diag.falcon_offline_ok:
        hard_fail = True
        reasons.append("Falcon-offline path failed (explain/loadback/enter-diff)")

    if fixture is not None:
        lint_report = lint_memx_fixture(fixture)
        script_ok = lint_report.ok
        if not lint_report.ok:
            hard_fail = True
            reasons.append(
                "MEMX script lint failed: "
                + ", ".join(f.code for f in lint_report.errors())
            )

    if snapshot is not None:
        entry_report = evaluate_entry_snapshot(load_entry_snapshot(snapshot))
        entry_classification = entry_report.classification
        posted_ok = entry_report.posted_baseline_ok
        if entry_report.classification in {
            "COLD_REPLUG",
            "STUB_PRAMIN",
            "BAR0_DEAD",
            "UNKNOWN",
        }:
            hard_fail = True
            reasons.append(
                f"entry probe {entry_report.classification}: "
                "do not spend a full cold path on PRAMIN activation"
            )
        elif entry_report.classification == "POSTED_CANDIDATE":
            reasons.append(
                "entry probe POSTED_CANDIDATE — verify VM-disabled physical BAR1 "
                "before channel bring-up (H70 field predicates only)"
            )

    if hard_fail:
        verdict = "NO_GO"
    else:
        verdict = "CONDITIONAL"
        if not any("H70" in r or "POSTED_CANDIDATE" in r for r in reasons):
            reasons.append(
                "Falcon-offline OK (H52 MEMIF class) — do not re-prove pad xdst on cold silicon"
            )
            reasons.append(
                "Physical PRAMIN/BAR1 still requires H70/H73 posted baseline "
                "(or a newly sourced pre-runtime producer); oracle cannot invent this"
            )
            reasons.append(
                "progress.md: stop replaying full cold path for PRAMIN activation"
            )

    return ColdGateReport(
        verdict=verdict,
        falcon_offline_ok=diag.falcon_offline_ok,
        script_ok=script_ok,
        entry_classification=entry_classification,
        posted_baseline_ok=posted_ok,
        reasons=reasons,
        diagnose=diag,
        lint=lint_report,
        entry=entry_report,
    )


def format_cold_gate(report: ColdGateReport) -> str:
    lines = [
        "Kepler cold-run gate (offline — no GPU access)",
        f"  verdict:         {report.verdict}",
        f"  falcon_offline:  {'OK' if report.falcon_offline_ok else 'FAIL'}",
    ]
    if report.script_ok is not None:
        lines.append(f"  memx_script:     {'OK' if report.script_ok else 'FAIL'}")
    if report.entry_classification is not None:
        lines.append(f"  entry_probe:     {report.entry_classification}")
        lines.append(
            f"  posted_baseline: {'OK' if report.posted_baseline_ok else 'NO'}"
        )
    lines.append("  reasons:")
    for r in report.reasons:
        lines.append(f"    - {r}")
    if report.verdict == "CONDITIONAL":
        lines.append("  live priority: H79 prefix bisect k=4; H80 confirmed (no mid-POST 0x1700)")
    return "\n".join(lines)


def diff_pad_enter_vs_memx(corpus_dir: Path | str) -> tuple[bool, str]:
    """Compare pad ENTER host effects vs stock memx_func_enter (GF119 path)."""
    from .bus.gk104_fake_adapter import GK104FakeBus
    from .devices.pmu_mmio import PmuMmioWindow
    from .executor import FalconExecutor
    from .manifests import load_image_memory, load_initial_dmem, load_symbols
    from .state import FalconMemory, FalconState
    from .trace import TraceCollector

    corpus_dir = Path(corpus_dir)
    # Pad until first XFER only
    manifest, imem, _ = load_image_memory(corpus_dir)
    dmem = FalconMemory(load_initial_dmem(corpus_dir, manifest.dmem_size), name="dmem")
    state = FalconState(pc=manifest.entry_address, imem=imem, dmem=dmem)
    pad_trace = TraceCollector(level=TraceLevel.EXTERNAL)
    pad_bus = PmuMmioWindow(inner=GK104FakeBus(trace=pad_trace), trace=None)
    pad_bus.seed_host(0x1620, 0x00000AAB)
    pad_bus.seed_host(0x26F0, 0x00000001)
    exe = FalconExecutor(
        state=state,
        bus=pad_bus,
        trace=pad_trace,
        symbols=load_symbols(corpus_dir),
        done_pc=manifest.done_pc,
        done_magic_addr=manifest.done_magic_addr,
        done_magic_value=manifest.done_magic_value,
    )
    while state.status == "running" and state.instruction_count < 100_000:
        exe.step()
        # Stop once first xdst has been issued (after ENTER ack).
        if any(e.kind is EventKind.XFER_START for e in pad_trace.events):
            break

    pad_enter = [
        e
        for e in canonical_external_events(pad_trace.events)
        if e.kind in {EventKind.MMIO_WRITE, EventKind.MMIO_READ}
        and e.kind is not EventKind.XFER_START
    ]
    # Drop events at/after first xfer by rebuilding from pre-xfer only
    pad_enter = []
    for e in canonical_external_events(pad_trace.events):
        if e.kind is EventKind.XFER_START:
            break
        if e.kind in {EventKind.MMIO_WRITE, EventKind.MMIO_READ}:
            pad_enter.append(e)

    memx = execute_semantic_memx_enter_leave(seed_host={0x1620: 0xAAB, 0x26F0: 0x1})
    memx_enter = []
    saw_pause_set = False
    for e in canonical_external_events(memx.events):
        if e.kind in {EventKind.MMIO_WRITE, EventKind.MMIO_READ}:
            memx_enter.append(e)
        if e.kind is EventKind.MMIO_WRITE and e.address == FB_PAUSE_SET:
            saw_pause_set = True
            # Include the following pause status read(s), then stop before LEAVE clear.
            continue
        if saw_pause_set and e.kind is EventKind.MMIO_WRITE and e.address == FB_PAUSE_CLEAR:
            memx_enter.pop()  # drop the leave clear we just appended
            break

    lines = [
        "Pad ENTER vs stock memx_func_enter (GF119)",
        "",
        "Pad ENTER host effects:",
    ]
    for e in pad_enter:
        lines.append(f"  {e.kind.name:10} {e.address:#x} = {e.value:#x}")
    lines.append("")
    lines.append("MEMX ENTER host effects:")
    for e in memx_enter:
        lines.append(f"  {e.kind.name:10} {e.address:#x} = {e.value:#x}")
    lines.append("")
    # Highlight known intentional pad simplification from progress Night40/41.
    lines.append("Bring-up notes (progress.md):")
    lines.append("  - Stock MEMX RMW-clears 0x1620 (~0xaa2, then ~bit0) and 0x26f0 (~bit0).")
    lines.append("  - Pad uses absolute wr32(0x1620,8) then wr32(0x26f0,0) before pause SET.")
    lines.append("  - Both must observe rising-edge FB_PAUSE before first xdst (H52 path).")
    same_pause = any(e.address == FB_PAUSE_SET and e.value == PAUSE_BIT for e in pad_enter) and any(
        e.address == FB_PAUSE_SET and e.value == PAUSE_BIT for e in memx_enter
    )
    ok = same_pause and len(pad_enter) > 0 and len(memx_enter) > 0
    if ok:
        lines.append("  - RESULT: both paths issue pause SET; sequences intentionally differ.")
    else:
        lines.append("  - RESULT: pause SET missing on one path — pad/MEMX ENTER broken.")
    return ok, "\n".join(lines)
