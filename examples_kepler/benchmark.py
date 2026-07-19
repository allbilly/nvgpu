#!/usr/bin/env python3
"""Live GK104 add/mul launch matrix (TinyGPU / shared PCIe path).

Runs the same case table used to validate multi-CTA and the VA 0x67000 PTE
hole fix.  Default: both operations.  Examples::

  python3 examples_kepler/benchmark.py
  python3 examples_kepler/benchmark.py --op mul
  python3 examples_kepler/benchmark.py --op add,mul --quick

Requires TinyGPU at /tmp/tinygpu.sock (or KEPLER_TINYGPU_SOCK).  Wall times
are dominated by cold bring-up (~10s); compare paths, not absolute GPU FLOPs.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ADD_PY = SCRIPT_DIR / "add.py"
LOG_DIR = REPO_ROOT / "logs"

# (label, env overrides).  Empty MULTI_CTA → auto.
CASES = [
  ("N=8 auto", {"KEPLER_N": "8"}),
  ("N=512 auto", {"KEPLER_N": "512"}),
  ("N=512 BLOCK=256 auto", {"KEPLER_N": "512", "KEPLER_BLOCK": "256"}),
  ("N=512 MULTI_CTA=0", {"KEPLER_N": "512", "KEPLER_MULTI_CTA": "0"}),
  ("N=20480 auto", {"KEPLER_N": "20480", "KEPLER_BLOCK": "1024"}),
  ("N=24576 MULTI_CTA=1", {
      "KEPLER_N": "24576", "KEPLER_MULTI_CTA": "1", "KEPLER_BLOCK": "1024",
      "KEPLER_DUMP_MISMATCH": "1"}),
  ("N=32768 auto", {
      "KEPLER_N": "32768", "KEPLER_BLOCK": "1024", "KEPLER_DUMP_MISMATCH": "1"}),
  ("N=32768 MULTI_CTA=1", {
      "KEPLER_N": "32768", "KEPLER_MULTI_CTA": "1", "KEPLER_BLOCK": "1024",
      "KEPLER_DUMP_MISMATCH": "1"}),
  ("N=65536 auto", {
      "KEPLER_N": "65536", "KEPLER_BLOCK": "1024", "KEPLER_DUMP_MISMATCH": "1"}),
  ("N=65536 MULTI_CTA=0", {
      "KEPLER_N": "65536", "KEPLER_MULTI_CTA": "0", "KEPLER_BLOCK": "1024"}),
]

QUICK_CASES = [
  "N=512 BLOCK=256 auto",
  "N=20480 auto",
  "N=32768 MULTI_CTA=1",
  "N=65536 auto",
]

BASE_ENV = {
  "KEPLER_ALLOW_DIRTY": "1",
  "KEPLER_SEED": "42",
  "KEPLER_AUTO_WARM_CONTINUE": "1",
  "KEPLER_RPC_LIGHT": "1",
}


def _ensure_sock() -> None:
  sock = os.environ.get("KEPLER_TINYGPU_SOCK", "/tmp/tinygpu.sock")
  try:
    import socket
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(0.5)
    s.connect(sock)
    s.close()
    return
  except OSError:
    pass
  app = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"
  if not os.path.isfile(app):
    raise SystemExit(f"TinyGPU sock missing ({sock}) and {app} not found")
  subprocess.run(["pkill", "-f", "TinyGPU server"], check=False)
  time.sleep(0.3)
  try:
    os.unlink(sock)
  except FileNotFoundError:
    pass
  log = open("/tmp/tinygpu-server.log", "ab")
  subprocess.Popen([app, "server", sock], stdout=log, stderr=log)
  for _ in range(20):
    time.sleep(0.25)
    try:
      import socket
      s = socket.socket(socket.AF_UNIX)
      s.settimeout(0.5)
      s.connect(sock)
      s.close()
      return
    except OSError:
      continue
  raise SystemExit(f"TinyGPU server did not create {sock}")


def _slug(op: str, label: str) -> str:
  return re.sub(r"[^A-Za-z0-9]+", "_", f"{op}_{label}").strip("_")


def _parse_launch(text: str) -> str:
  m = re.search(r"\[kepler\] launch N=\d+ ([^\n]+)", text)
  if not m:
    return "?"
  launch = m.group(1).strip()
  if "multi-CTA" in launch:
    g = re.search(r"grid=\([^)]+\)", launch)
    return f"multi-CTA {g.group(0)}" if g else "multi-CTA"
  if "channel_windows" in launch:
    return launch
  return launch


def _parse_status(text: str, rc: int) -> str:
  if re.search(r"hardware_demo=ok", text):
    return "ok"
  mism = re.search(r"mismatch ranges[^\n]*", text)
  if mism:
    return f"FAIL {mism.group(0)}"
  m = re.search(r"mismatches=(\d+)/\d+", text)
  if m and int(m.group(1)) > 0:
    return f"FAIL mismatches={m.group(1)}"
  if "AssertionError" in text or rc != 0:
    err = re.search(r"AssertionError:.*", text)
    return f"FAIL {err.group(0)}" if err else f"FAIL rc={rc}"
  return f"rc={rc}"


def run_case(op: str, label: str, extra: dict[str, str]) -> tuple[str, str, str, float]:
  LOG_DIR.mkdir(exist_ok=True)
  slug = _slug(op, label)
  log_path = LOG_DIR / f"bench-{slug}.log"
  err_path = LOG_DIR / f"bench-{slug}.err"
  env = {**os.environ, **BASE_ENV, **extra, "KEPLER_OPERATION": op}
  if op == "mul":
    env["KEPLER_CUBIN"] = str(SCRIPT_DIR / "mul_kepler.cubin")
  else:
    env["KEPLER_CUBIN"] = str(SCRIPT_DIR / "add_kepler.cubin")
  t0 = time.time()
  with open(log_path, "w") as logf, open(err_path, "w") as errf:
    proc = subprocess.run(
        [sys.executable, "-u", str(ADD_PY)],
        env=env, stdout=logf, stderr=errf, cwd=str(REPO_ROOT))
  dt = time.time() - t0
  text = log_path.read_text(errors="replace") + err_path.read_text(errors="replace")
  return label, _parse_launch(text), _parse_status(text, proc.returncode), dt


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--op", default="add,mul",
                  help="Comma-separated operations: add, mul (default: both)")
  ap.add_argument("--quick", action="store_true",
                  help="Only the four load-bearing multi-CTA / large-N cases")
  ap.add_argument("--no-sock-start", action="store_true",
                  help="Do not attempt to start TinyGPU if sock is missing")
  args = ap.parse_args()
  ops = [o.strip() for o in args.op.split(",") if o.strip()]
  for o in ops:
    if o not in ("add", "mul"):
      raise SystemExit(f"unsupported op {o!r}; expected add or mul")
  cases = CASES
  if args.quick:
    want = set(QUICK_CASES)
    cases = [(lab, env) for lab, env in CASES if lab in want]

  if not args.no_sock_start:
    _ensure_sock()

  rows: list[tuple[str, str, str, str, str]] = []
  failed = 0
  print(f"{'op':3} | {'case':22} | {'path':42} | {'result':6} | time", flush=True)
  print("-" * 100, flush=True)
  for op in ops:
    for label, extra in cases:
      lab, path, status, dt = run_case(op, label, extra)
      rows.append((op, lab, path, status, f"{dt:.1f}s"))
      if status != "ok":
        failed += 1
      print(f"{op:3} | {lab:22} | {path:42} | {status:6} | {dt:.1f}s", flush=True)

  print("\n| Op | Case | Path | Result | Runtime |")
  print("|---|---|---|---|---|")
  for op, lab, path, status, dt in rows:
    print(f"| {op} | {lab} | {path} | {status} | {dt} |")
  print(f"\nbenchmark={'ok' if failed == 0 else 'FAIL'} "
        f"cases={len(rows)} failed={failed}", flush=True)
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
