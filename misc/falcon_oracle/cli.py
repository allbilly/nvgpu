"""CLI entry: python -m falcon_oracle <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .bus.gk104_fake_adapter import GK104FakeBus
from .coverage import format_coverage, report_corpus_coverage
from .diff import compare_traces, format_divergence
from .executor import FalconExecutor
from .manifests import load_image_memory, load_initial_dmem, load_symbols, verify_corpus
from .runner import execute_falcon_bootstrap
from .semantic import execute_semantic_bootstrap
from .state import FalconMemory, FalconState
from .trace import TraceCollector, TraceLevel, events_from_jsonl, events_to_jsonl, externally_visible


def cmd_coverage(args: argparse.Namespace) -> int:
    verify_corpus(args.corpus)
    report = report_corpus_coverage(args.corpus)
    print(format_coverage(report))
    return 0 if report.get("manifest_ok") else 1


def cmd_disasm(args: argparse.Namespace) -> int:
    from .decoder import FalconDecoder
    from .values import parse_int

    image = Path(args.image).read_bytes()
    base = parse_int(args.base)
    imem = FalconMemory(bytearray(image), name="imem", base=base)
    decoder = FalconDecoder()
    for insn in decoder.decode_all(imem):
        ops = " ".join(str(o) for o in insn.operands)
        print(f"{insn.pc:08x}: {insn.raw.hex(' '):<20} {insn.mnemonic} {ops}".rstrip())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scenario = Path(args.scenario).read_text() if args.scenario else None
    mode = "scripted" if scenario else args.mode
    result = execute_falcon_bootstrap(
        args.corpus,
        mode=mode,
        scenario_yaml=scenario,
        trace_level=TraceLevel(args.trace_level),
        max_instructions=args.max_instructions,
    )
    text = events_to_jsonl(result.events)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    meta = {
        "stop_reason": result.stop_reason,
        "instruction_count": result.instruction_count,
        "trace_hash": result.trace_hash,
        "pc": result.state.pc,
        "status": result.state.status,
    }
    print(json.dumps(meta, sort_keys=True), file=sys.stderr)
    return 0 if result.stop_reason == "done" else 1


def cmd_diff(args: argparse.Namespace) -> int:
    ref = events_from_jsonl(Path(args.reference).read_text())
    cand = events_from_jsonl(Path(args.candidate).read_text())
    diffs = compare_traces(
        ref,
        cand,
        external_only=args.external_only,
        stop_at_first=args.stop_at_first,
    )
    if not diffs:
        print("OK: traces match")
        return 0
    print(format_divergence(diffs[0]))
    return 1


def cmd_diff_semantic(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    sem = execute_semantic_bootstrap(corpus)
    fal = execute_falcon_bootstrap(corpus, mode="fake")

    from .bringup import verify_loadback
    from .runner import _wrap_pmu_bus

    # Recover VRAM via a dedicated explain-style run for byte check.
    from .bringup import NIGHT41F_TRANSFERS
    from .devices.pmu_mmio import PmuMmioWindow

    manifest, imem, _ = load_image_memory(corpus)
    dmem = FalconMemory(load_initial_dmem(corpus, manifest.dmem_size), name="dmem")
    state = FalconState(pc=manifest.entry_address, imem=imem, dmem=dmem)
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    inner = GK104FakeBus(trace=trace)
    bus = _wrap_pmu_bus(inner)
    FalconExecutor(
        state=state,
        bus=bus,
        trace=trace,
        symbols=load_symbols(corpus),
        done_pc=manifest.done_pc,
        done_magic_addr=manifest.done_magic_addr,
        done_magic_value=manifest.done_magic_value,
    ).run(max_instructions=100_000)

    report = verify_loadback(inner.vram, corpus)
    if not report.ok:
        print(f"VRAM loadback failed: {report.matched_bytes}/{report.total_bytes}")
        for frag in report.fragments:
            if not frag["ok"]:
                print(f"  {frag['name']} @ {frag['vram']:#x}: {frag['first_mismatch']}")
        return 1

    diffs = compare_traces(sem.events, fal.events, external_only=True, stop_at_first=True)
    if diffs:
        print(format_divergence(diffs[0], corpus=str(corpus)))
        return 1
    print("OK: semantic and falcon external traces match; Night41f loadback 40/40")
    print(
        json.dumps(
            {"semantic_hash": sem.trace_hash, "falcon_hash": fal.trace_hash},
            sort_keys=True,
        )
    )
    return 0 if fal.stop_reason == "done" else 1


def cmd_explain(args: argparse.Namespace) -> int:
    from .bringup import explain_bootstrap, format_explain

    report = explain_bootstrap(args.corpus)
    print(format_explain(report))
    return 0 if report.stop_reason == "done" and report.loadback.ok else 1


def cmd_enter_diff(args: argparse.Namespace) -> int:
    from .bringup import diff_pad_enter_vs_memx

    ok, text = diff_pad_enter_vs_memx(args.corpus)
    print(text)
    return 0 if ok else 1


def cmd_loadback(args: argparse.Namespace) -> int:
    from .bringup import explain_bootstrap

    report = explain_bootstrap(args.corpus)
    print(
        json.dumps(
            {
                "ok": report.loadback.ok,
                "matched_bytes": report.loadback.matched_bytes,
                "total_bytes": report.loadback.total_bytes,
                "fragments": report.loadback.fragments,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if report.loadback.ok else 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .bringup import diagnose_bringup, format_diagnose

    report = diagnose_bringup(args.corpus)
    print(format_diagnose(report))
    return 0 if report.falcon_offline_ok else 1


def cmd_memx_run(args: argparse.Namespace) -> int:
    from .errors import DeviceModelError
    from .memx_script import execute_memx_fixture, load_memx_fixture

    fixture = Path(args.fixture)
    _commands, _seed, meta = load_memx_fixture(fixture)
    try:
        result = execute_memx_fixture(fixture)
    except DeviceModelError as exc:
        if meta.get("expect_error") and meta["expect_error"].lower() in str(exc).lower():
            print(json.dumps({"ok": True, "expected_error": str(exc)}, sort_keys=True))
            return 0
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "name": meta.get("name"),
                "stop_reason": result.stop_reason,
                "commands_executed": result.commands_executed,
                "wait_satisfied": result.wait_satisfied,
                "trace_hash": result.trace_hash,
                "host": {f"0x{k:x}": v for k, v in sorted(result.host.items())},
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def cmd_memx_lint(args: argparse.Namespace) -> int:
    from .memx_lint import format_lint, lint_memx_fixture

    report = lint_memx_fixture(args.fixture)
    print(format_lint(report, path=str(args.fixture)))
    return 0 if report.ok else 1


def cmd_memx_encode(args: argparse.Namespace) -> int:
    from .memx_script import load_memx_fixture
    from .memx_wire import decode_memx_words, encode_memx_packets, encode_memx_words

    commands, _seed, meta = load_memx_fixture(args.fixture)
    words = encode_memx_words(commands)
    packets = encode_memx_packets(commands)
    roundtrip = decode_memx_words(words)
    print(
        json.dumps(
            {
                "name": meta.get("name"),
                "packets": [
                    {"mthd": p.mthd, "size": len(p.data), "header": f"0x{p.header:08x}"}
                    for p in packets
                ],
                "words": [f"0x{w:08x}" for w in words],
                "word_count": len(words),
                "roundtrip_ops": [c.op.name for c in roundtrip],
            },
            indent=2,
        )
    )
    return 0


def cmd_cold_gate(args: argparse.Namespace) -> int:
    from .bringup import cold_gate, format_cold_gate

    report = cold_gate(args.corpus, fixture=args.fixture, snapshot=args.snapshot)
    print(format_cold_gate(report))
    # CONDITIONAL is exit 0 (offline ready); NO_GO is 1.
    return 0 if report.verdict == "CONDITIONAL" else 1


def cmd_entry_probe(args: argparse.Namespace) -> int:
    from .entry_probe import (
        evaluate_entry_snapshot,
        format_entry_probe,
        load_entry_snapshot,
    )

    report = evaluate_entry_snapshot(load_entry_snapshot(args.snapshot))
    print(format_entry_probe(report))
    # POSTED_CANDIDATE → 0; everything else → 1 (including COLD_REPLUG).
    return 0 if report.classification == "POSTED_CANDIDATE" else 1


def cmd_memx_diff(args: argparse.Namespace) -> int:
    from .memx_diff import diff_memx_fixtures, format_memx_diff

    diffs = diff_memx_fixtures(args.left, args.right, stop_at_first=args.stop_at_first)
    print(format_memx_diff(diffs))
    return 0 if not diffs else 1


def cmd_hypotheses(args: argparse.Namespace) -> int:
    from .hypotheses import format_hypotheses

    print(format_hypotheses(args.board))
    return 0


def cmd_lifecycle(args: argparse.Namespace) -> int:
    from .lifecycle import evaluate_lifecycle, format_lifecycle

    observed = {}
    if args.pramin_live:
        observed["pramin_live"] = True
    if args.observed:
        for item in args.observed:
            observed[item] = True
    report = evaluate_lifecycle(observed)
    print(format_lifecycle(report))
    return 0 if report.ok else 1


def cmd_plan_check(args: argparse.Namespace) -> int:
    from .plan_check import check_plan_file, format_plan_report

    report = check_plan_file(args.plan, sequence_path=args.sequence)
    print(format_plan_report(report))
    return 0 if report.verdict == "ALLOW" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="falcon_oracle")
    p.add_argument("--version", action="version", version=f"falcon_oracle {__version__}")
    sp = p.add_subparsers(dest="cmd", required=True)

    c = sp.add_parser("coverage", help="Verify corpus decode coverage")
    c.add_argument("--corpus", type=Path, required=True)
    c.set_defaults(func=cmd_coverage)

    d = sp.add_parser("disasm", help="Disassemble with local decoder")
    d.add_argument("--image", type=Path, required=True)
    d.add_argument("--base", default="0xb14")
    d.set_defaults(func=cmd_disasm)

    r = sp.add_parser("run", help="Execute corpus")
    r.add_argument("--corpus", type=Path, required=True)
    r.add_argument("--scenario", type=Path, default=None)
    r.add_argument("--mode", choices=["fake", "scripted", "cpu"], default="fake")
    r.add_argument("--trace-level", default="external")
    r.add_argument("--output", type=Path, default=None)
    r.add_argument("--max-instructions", type=int, default=100_000)
    r.set_defaults(func=cmd_run)

    df = sp.add_parser("diff", help="Compare two JSONL traces")
    df.add_argument("--reference", type=Path, required=True)
    df.add_argument("--candidate", type=Path, required=True)
    df.add_argument("--external-only", action="store_true")
    df.add_argument("--stop-at-first", action="store_true", default=True)
    df.set_defaults(func=cmd_diff)

    ds = sp.add_parser("diff-semantic", help="Compare semantic vs falcon bootstrap")
    ds.add_argument("--corpus", type=Path, required=True)
    ds.set_defaults(func=cmd_diff_semantic)

    ex = sp.add_parser("explain", help="Night41 bring-up summary for the PMU pad")
    ex.add_argument("--corpus", type=Path, required=True)
    ex.set_defaults(func=cmd_explain)

    ed = sp.add_parser("enter-diff", help="Diff pad ENTER vs stock memx_func_enter")
    ed.add_argument("--corpus", type=Path, required=True)
    ed.set_defaults(func=cmd_enter_diff)

    lb = sp.add_parser("loadback", help="Verify Night41f 40-byte MEMIF loadback")
    lb.add_argument("--corpus", type=Path, required=True)
    lb.set_defaults(func=cmd_loadback)

    dg = sp.add_parser(
        "diagnose",
        help="Triage Falcon-offline (H52) vs live POST (H70/H73) for Kepler bring-up",
    )
    dg.add_argument("--corpus", type=Path, required=True)
    dg.set_defaults(func=cmd_diagnose)

    mr = sp.add_parser("memx-run", help="Run a MEMX semantic fixture (plan.md ladder)")
    mr.add_argument("--fixture", type=Path, required=True)
    mr.set_defaults(func=cmd_memx_run)

    ml = sp.add_parser("memx-lint", help="Lint MEMX script (H29 wait, ENTER ownership, capacity)")
    ml.add_argument("--fixture", type=Path, required=True)
    ml.set_defaults(func=cmd_memx_lint)

    me = sp.add_parser("memx-encode", help="Encode fixture to Nouveau 0x10a1c4 wire words")
    me.add_argument("--fixture", type=Path, required=True)
    me.set_defaults(func=cmd_memx_encode)

    cg = sp.add_parser(
        "cold-gate",
        help="Cold-run gate (offline): Falcon OK → CONDITIONAL; never touches GPU",
    )
    cg.add_argument("--corpus", type=Path, required=True)
    cg.add_argument("--fixture", type=Path, default=None, help="Optional MEMX script to lint")
    cg.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Optional recorded MMIO snapshot JSON (no live read)",
    )
    cg.set_defaults(func=cmd_cold_gate)

    ep = sp.add_parser(
        "entry-probe",
        help="Classify a recorded entry MMIO snapshot (Night41h / H70) — offline only",
    )
    ep.add_argument("--snapshot", type=Path, required=True)
    ep.set_defaults(func=cmd_entry_probe)

    md = sp.add_parser("memx-diff", help="Diff two MEMX fixtures (first divergence)")
    md.add_argument("--left", type=Path, required=True)
    md.add_argument("--right", type=Path, required=True)
    md.add_argument("--stop-at-first", action="store_true", default=True)
    md.set_defaults(func=cmd_memx_diff)

    hy = sp.add_parser("hypotheses", help="Show progress.md hypothesis board (offline)")
    hy.add_argument("--board", type=Path, default=None)
    hy.set_defaults(func=cmd_hypotheses)

    lc = sp.add_parser("lifecycle", help="Nouveau/Python lifecycle matrix gate (offline)")
    lc.add_argument(
        "--pramin-live",
        action="store_true",
        help="Claim posted PRAMIN already observed (H70)",
    )
    lc.add_argument(
        "--observed",
        action="append",
        default=[],
        help="Mark a stage as observed (repeatable)",
    )
    lc.set_defaults(func=cmd_lifecycle)

    pc = sp.add_parser(
        "plan-check",
        help="Refuse known-bad plans; validate the historical H76-only sequence",
    )
    pc.add_argument("--plan", type=Path, required=True)
    pc.add_argument("--sequence", type=Path, default=None, help="Optional H76 sequence JSON")
    pc.set_defaults(func=cmd_plan_check)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
