"""CLI: python3 -m x86rom_py <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_rom, write_census_json
from .constants import LIVE_ACK_TOKEN
from .executor import live_ack_ok, run_rom, run_until_entry_target
from .rom import DEFAULT_ROM_PATH, load_rom
from .trace import (
  backward_slice,
  compare_traces,
  dump_jsonl,
  find_pramin_live_index,
  load_jsonl,
)


def cmd_inspect(args: argparse.Namespace) -> int:
  rom = load_rom(args.rom)
  pcir = rom.pcir
  print(f"container: {rom.container_path}")
  print(f"container_sha256: {rom.container_sha256}")
  print(f"legacy_offset: {rom.image_offset:#x}")
  print(f"legacy_size: {rom.size:#x}")
  print(f"legacy_sha256: {rom.sha256}")
  print(f"checksum: {rom.checksum:#x}")
  print(
    f"PCIR@{pcir.offset:#x}: {pcir.vendor_id:04x}:{pcir.device_id:04x} "
    f"type={pcir.code_type} indicator={pcir.indicator:#x} "
    f"final={pcir.is_final_image}"
  )
  print(f"entry: {rom.entry_offset:#x} (C000:{rom.entry_offset:04x})")
  return 0


def cmd_analyze(args: argparse.Namespace) -> int:
  rom = load_rom(args.rom)
  report = analyze_rom(rom)
  if args.json:
    write_census_json(report, args.json)
    print(f"wrote census to {args.json}")
  print(f"entry_path: {[hex(x) for x in report.entry_path]}")
  print(f"INT sites: {len(report.interrupts)}")
  print(f"IN sites: {len(report.io_ins)}  OUT sites: {len(report.io_outs)}")
  print(f"far call/retf: {len(report.far_calls)}  far jmp: {len(report.far_jmps)}")
  print(f"pci mech immediates: {len(report.pci_mech)}")
  print(f"llvm_available: {report.llvm_available} lines={report.llvm_disasm_lines}")
  print(f"unknown_opcode_ledger: {len(report.unknown_ledger)} entries")
  if args.verbose:
    for hit in report.io_outs[:20]:
      print(f"  OUT @{hit.offset:#06x} {hit.mnemonic} {hit.detail}")
  return 0


def cmd_compare(args: argparse.Namespace) -> int:
  a = load_jsonl(args.trace_a)
  b = load_jsonl(args.trace_b)
  volatile = set(args.volatile.split(",")) if args.volatile else set()
  result = compare_traces(
    a, b,
    volatile_fields=volatile,
    ignore_poll_repeats=args.ignore_polls,
    normalize_bars=args.normalize_bars,
  )
  if result.match:
    print(f"match: {result.left_count} events")
    return 0
  d = result.divergence
  assert d is not None
  print(f"DIVERGE @ {d.index}: {d.reason}")
  if d.left:
    print(f"  left:  {d.left.operation} {d.left.address_space} {d.left.direction} "
          f"off={d.left.canonical_offset:#x} w={d.left.width}")
  if d.right:
    print(f"  right: {d.right.operation} {d.right.address_space} {d.right.direction} "
          f"off={d.right.canonical_offset:#x} w={d.right.width}")
  return 1


def cmd_replay(args: argparse.Namespace) -> int:
  events = load_jsonl(args.golden)
  sink = find_pramin_live_index(events)
  if sink is None:
    sink = len(events) - 1 if events else 0
  report = backward_slice(events, sink)
  print(f"golden events: {len(events)}")
  print(f"sink_index: {sink}")
  print(f"retained: {len(report.retained)}")
  print(f"stable_producers: {len(report.stable_producers)}")
  print(f"polling: {len(report.polling)}")
  print(f"volatile_reads: {len(report.volatile_reads)}")
  print(f"address_allocation: {len(report.address_allocation)}")
  if args.slice_out:
    dump_jsonl(report.retained, args.slice_out)
    print(f"wrote slice to {args.slice_out}")
  print("replay preview only (ModelBus/ReplayBus execution: use tests / live)")
  return 0


def cmd_run(args: argparse.Namespace) -> int:
  """Offline executor until stop (default: entry target 0x2caa)."""
  rom = load_rom(args.rom)
  if args.until == "entry":
    result = run_until_entry_target(rom)
  else:
    stop_ips = set()
    if args.stop_at_ip is not None:
      stop_ips.add(int(args.stop_at_ip, 0))
    result = run_rom(
      rom,
      max_insns=args.max_insns,
      stop_at_ips=stop_ips or None,
      stop_at_pramin_live=args.stop_at == "pramin-live",
    )
  stop = result.stop
  print(f"stop: {stop.reason} at {stop.cs:04x}:{stop.ip:04x}")
  print(f"bytes: {stop.bytes_at_ip.hex()}")
  if stop.message:
    print(f"message: {stop.message}")
  print(f"insns: {result.insn_count}  ops: {result.op_count}")
  print(f"reached entry path ips: "
        f"{sorted(hex(x) for x in result.reached_ips if x in (0x3, 0x50, 0x2caa))}")
  if args.trace_out and result.trace is not None:
    dump_jsonl(result.trace.events, args.trace_out)
    print(f"wrote trace {args.trace_out} ({len(result.trace.events)} events)")
  return 0 if stop.reason in ("breakpoint", "rom-return", "pramin-live") else 1


def cmd_live(args: argparse.Namespace) -> int:
  if not live_ack_ok(args.live_ack):
    print(
      f"refusing live: need --live-ack {LIVE_ACK_TOKEN}",
      file=sys.stderr,
    )
    return 2
  if not args.trace_out:
    print("refusing live: --trace-out is mandatory before first write", file=sys.stderr)
    return 2
  print(
    "live path scaffolds LiveBus over TinyGPU/LinuxPCIDevice; "
    "requires fresh replug + unbound GK104. Not executing writes in this build "
    "without an attached transport module.",
    file=sys.stderr,
  )
  # Identity gate only unless a device factory is injected later.
  print(json.dumps({
    "backend": args.backend,
    "live_ack": args.live_ack,
    "trace_out": args.trace_out,
    "stop_at": args.stop_at,
    "status": "armed-but-no-transport",
  }))
  return 3


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="x86rom_py", description="GK104 legacy option-ROM producer toolkit")
  p.add_argument("--rom", type=Path, default=DEFAULT_ROM_PATH)
  sub = p.add_subparsers(dest="cmd", required=True)

  s = sub.add_parser("inspect", help="Validate and print pinned ROM identity")
  s.set_defaults(func=cmd_inspect)

  s = sub.add_parser("analyze", help="Static census / entry-path verification")
  s.add_argument("--json", type=Path, help="Write machine-readable census JSON")
  s.add_argument("-v", "--verbose", action="store_true")
  s.set_defaults(func=cmd_analyze)

  s = sub.add_parser("compare", help="Compare two JSONL traces")
  s.add_argument("trace_a", type=Path)
  s.add_argument("trace_b", type=Path)
  s.add_argument("--volatile", default="", help="Comma-separated volatile fields")
  s.add_argument("--ignore-polls", action="store_true")
  s.add_argument("--normalize-bars", action="store_true")
  s.set_defaults(func=cmd_compare)

  s = sub.add_parser("replay", help="Backward-slice / preview a golden trace")
  s.add_argument("golden", type=Path)
  s.add_argument("--slice-out", type=Path)
  s.set_defaults(func=cmd_replay)

  s = sub.add_parser("run", help="Offline real-mode execution")
  s.add_argument("--until", choices=("entry", "budget"), default="entry")
  s.add_argument("--stop-at-ip", default=None)
  s.add_argument("--stop-at", choices=("pramin-live",), default=None)
  s.add_argument("--max-insns", type=int, default=50_000)
  s.add_argument("--trace-out", type=Path)
  s.set_defaults(func=cmd_run)

  s = sub.add_parser("live", help="Guarded live producer experiment")
  s.add_argument("--backend", choices=("tinygpu", "linux"), default="tinygpu")
  s.add_argument("--live-ack", required=True)
  s.add_argument("--trace-out", type=Path, required=True)
  s.add_argument("--stop-at", default="pramin-live")
  s.set_defaults(func=cmd_live)

  return p


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":
  sys.exit(main())
