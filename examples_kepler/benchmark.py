#!/usr/bin/env python3
"""Live GK104 add/mul launch matrix (TinyGPU / shared PCIe path).

Reports both process wall time and host-measured kernel time (submit
semaphore poll until done).  Wall time is dominated by cold bring-up (~10s);
kernel_time_ms is what to compare for GPC boost experiments.

Examples::

  python3 examples_kepler/benchmark.py --op add --quick
  python3 examples_kepler/benchmark.py --op add --quick --boost
  python3 examples_kepler/benchmark.py --op add --compare-boost --quick
  python3 examples_kepler/benchmark.py --op add --compare-boost --quick --repeat 10

Requires TinyGPU at /tmp/tinygpu.sock (or KEPLER_TINYGPU_SOCK).
"""
from __future__ import annotations

import argparse
import math
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

BOOST_ENV = {
  "KEPLER_EXPERIMENTAL_PSTATE": "1",
  "KEPLER_EXPERIMENTAL_PSTATE_MEM": "0",
  "KEPLER_PSTATE_IDX": "1",
}


def _ensure_sock() -> None:
  sock = os.environ.get("KEPLER_TINYGPU_SOCK", "/tmp/tinygpu.sock")
  import socket
  try:
    s = socket.socket(socket.AF_UNIX)
    try:
      s.settimeout(0.5)
      s.connect(sock)
      return
    finally:
      s.close()
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
  with open("/tmp/tinygpu-server.log", "ab") as log:
    subprocess.Popen([app, "server", sock], stdout=log, stderr=log)
    for _ in range(20):
      time.sleep(0.25)
      try:
        s = socket.socket(socket.AF_UNIX)
        try:
          s.settimeout(0.5)
          s.connect(sock)
          return
        finally:
          s.close()
      except OSError:
        continue
  raise SystemExit(f"TinyGPU server did not create {sock}")


def _slug(op: str, label: str, tag: str = "") -> str:
  return re.sub(r"[^A-Za-z0-9]+", "_", f"{op}_{tag}_{label}" if tag else f"{op}_{label}").strip("_")


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
  if "AssertionError" in text or "TimeoutError" in text or rc != 0:
    err = re.search(r"(AssertionError|TimeoutError|RuntimeError):.*", text)
    return f"FAIL {err.group(0)}" if err else f"FAIL rc={rc}"
  return f"rc={rc}"


def _parse_kernel_ms(text: str) -> float | None:
  totals = [float(x) for x in re.findall(
      r"\[kepler\] kernel_time_total_ms=([0-9.]+)", text)]
  if totals:
    return totals[-1]
  parts = [float(x) for x in re.findall(
      r"\[kepler\] kernel_time_ms=([0-9.]+)", text)]
  if not parts:
    return None
  return sum(parts)


def _parse_gpc_coef(text: str) -> str:
  afters = re.findall(
      r"clk after experimental-pstate:.*?0x137004=(0x[0-9a-f]+)", text)
  if afters:
    return afters[-1]
  return "-"


def run_case(op: str, label: str, extra: dict[str, str], *,
             tag: str = "", trial: int | None = None
             ) -> tuple[str, str, str, float, float | None, str]:
  LOG_DIR.mkdir(exist_ok=True)
  slug = _slug(op, label, tag)
  if trial is not None:
    slug = f"{slug}_t{trial}"
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
  wall = time.time() - t0
  text = log_path.read_text(errors="replace") + err_path.read_text(errors="replace")
  return (label, _parse_launch(text), _parse_status(text, proc.returncode),
          wall, _parse_kernel_ms(text), _parse_gpc_coef(text))


def _fmt_kernel(ms: float | None) -> str:
  if ms is None:
    return "-"
  if ms < 1.0:
    return f"{ms * 1000:.0f}µs"
  if ms < 100.0:
    return f"{ms:.2f}ms"
  return f"{ms:.1f}ms"


def _mean_std(xs: list[float]) -> tuple[float, float]:
  if not xs:
    return float("nan"), float("nan")
  m = sum(xs) / len(xs)
  if len(xs) == 1:
    return m, 0.0
  var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
  return m, math.sqrt(var)


def _fmt_wall_ms(mean: float, std: float) -> str:
  return f"{mean:.1f}±{std:.1f}s"


def _fmt_kern_ms(mean: float, std: float) -> str:
  if mean < 1.0 and std < 1.0:
    return f"{mean * 1000:.0f}±{std * 1000:.0f}µs"
  return f"{mean:.2f}±{std:.2f}ms"


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--op", default="add,mul",
                  help="Comma-separated operations: add, mul (default: both)")
  ap.add_argument("--quick", action="store_true",
                  help="Only the four load-bearing multi-CTA / large-N cases")
  ap.add_argument("--boost", action="store_true",
                  help="Enable experimental GPC boost (PSTATE_IDX=1, MEM=0)")
  ap.add_argument("--compare-boost", action="store_true",
                  help="Run each case twice: baseline then boost; print delta")
  ap.add_argument("--repeat", type=int, default=1,
                  help="Repeat each (op,case,mode) N times; report mean±std")
  ap.add_argument("--no-sock-start", action="store_true",
                  help="Do not attempt to start TinyGPU if sock is missing")
  args = ap.parse_args()
  if args.repeat < 1:
    raise SystemExit("--repeat must be >= 1")
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

  modes: list[tuple[str, dict[str, str]]] = [("", {})]
  if args.compare_boost:
    modes = [("base", {}), ("boost", dict(BOOST_ENV))]
  elif args.boost:
    modes = [("boost", dict(BOOST_ENV))]

  # key -> aggregates
  agg: dict[tuple[str, str, str], dict] = {}
  failed = 0
  n_runs = 0
  hdr = (f"{'op':3} | {'mode':5} | {'case':22} | {'t':>2} | "
         f"{'result':6} | {'wall':7} | {'kernel':9} | coef")
  print(hdr, flush=True)
  print("-" * len(hdr), flush=True)
  for trial in range(1, args.repeat + 1):
    for op in ops:
      for label, extra in cases:
        for mode, mextra in modes:
          mode_name = mode or ("boost" if args.boost else "base")
          tag = mode or "run"
          lab, path, status, wall, kern, coef = run_case(
              op, label, {**extra, **mextra}, tag=tag, trial=trial)
          n_runs += 1
          if status != "ok":
            failed += 1
          key = (op, mode_name, lab)
          slot = agg.setdefault(key, {
              "path": path, "walls": [], "kerns": [], "coefs": [],
              "oks": 0, "fails": 0})
          slot["path"] = path
          slot["walls"].append(wall)
          if kern is not None:
            slot["kerns"].append(kern)
          if coef != "-":
            slot["coefs"].append(coef)
          if status == "ok":
            slot["oks"] += 1
          else:
            slot["fails"] += 1
          print(f"{op:3} | {mode_name:5} | {lab:22} | {trial:2d} | "
                f"{status:6} | {wall:6.1f}s | {_fmt_kernel(kern):9} | "
                f"{coef}", flush=True)

  print("\n=== mean ± sample std (n={}) ===".format(args.repeat), flush=True)
  print(f"{'op':3} | {'mode':5} | {'case':22} | {'ok':4} | "
        f"{'wall':14} | {'kernel':16} | coef | Δk(mean)", flush=True)
  print("-" * 100, flush=True)

  summary_rows = []
  base_kern_mean: dict[tuple[str, str], float] = {}
  for (op, mode_name, lab), slot in agg.items():
    wm, ws = _mean_std(slot["walls"])
    km, ks = _mean_std(slot["kerns"]) if slot["kerns"] else (float("nan"), float("nan"))
    coef = slot["coefs"][-1] if slot["coefs"] else "-"
    ok_s = f"{slot['oks']}/{slot['oks'] + slot['fails']}"
    dk = ""
    if mode_name == "base" and slot["kerns"]:
      base_kern_mean[(op, lab)] = km
    elif mode_name == "boost" and (op, lab) in base_kern_mean and slot["kerns"]:
      d = km - base_kern_mean[(op, lab)]
      dk = f"{d * 1000:+.0f}µs" if abs(d) < 1 else f"{d:+.2f}ms"
    print(f"{op:3} | {mode_name:5} | {lab:22} | {ok_s:4} | "
          f"{_fmt_wall_ms(wm, ws):14} | {_fmt_kern_ms(km, ks):16} | "
          f"{coef} | {dk}", flush=True)
    summary_rows.append((op, mode_name, lab, ok_s, wm, ws, km, ks, coef, dk))

  print("\n| Op | Mode | Case | OK | Wall | Kernel | Coef | Δk |")
  print("|---|---|---|---|---|---|---|---|")
  for op, mode, lab, ok_s, wm, ws, km, ks, coef, dk in summary_rows:
    print(f"| {op} | {mode} | {lab} | {ok_s} | {_fmt_wall_ms(wm, ws)} | "
          f"{_fmt_kern_ms(km, ks)} | {coef} | {dk or '-'} |")
  print(f"\nbenchmark={'ok' if failed == 0 else 'FAIL'} "
        f"runs={n_runs} failed={failed} repeat={args.repeat}", flush=True)
  print("Note: kernel_time is host GP_PUT→semaphore (TinyGPU RPC noise); "
        "mean±std over --repeat.", flush=True)
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
