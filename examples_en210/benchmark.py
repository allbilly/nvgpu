#!/usr/bin/env python3
"""Live GT218 add launch matrix (TinyGPU socket path).

Reports process wall time and host-measured kernel time (submit to
GPPut poll).  Wall time is dominated by cold bring-up (~10s);
kernel_time_ms is what to compare across N/block configurations.

Examples::

  python3 benchmark.py --quick
  python3 benchmark.py --quick --repeat 5
  python3 benchmark.py

Requires TinyGPU at /tmp/tinygpu.sock.
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
ADD_PY = SCRIPT_DIR / "add.py"
LOG_DIR = SCRIPT_DIR / "logs"

# (label, env overrides).  EN210_BLOCK defaults to N (single-CTA) unless set.
CASES = [
  ("N=4 block=4",        {"EN210_N": "4",      "EN210_BLOCK": "4"}),
  ("N=16 block=4",       {"EN210_N": "16",     "EN210_BLOCK": "4"}),
  ("N=64 block=32",      {"EN210_N": "64",     "EN210_BLOCK": "32"}),
  ("N=256 block=256",    {"EN210_N": "256",    "EN210_BLOCK": "256"}),
  ("N=256 block=128",    {"EN210_N": "256",    "EN210_BLOCK": "128"}),
  ("N=1024 block=512",   {"EN210_N": "1024",   "EN210_BLOCK": "512"}),
  ("N=4096 block=512",   {"EN210_N": "4096",   "EN210_BLOCK": "512"}),
  ("N=4096 block=256",   {"EN210_N": "4096",   "EN210_BLOCK": "256"}),
  ("N=8192 block=512",   {"EN210_N": "8192",   "EN210_BLOCK": "512"}),
  ("N=16384 block=512",  {"EN210_N": "16384",  "EN210_BLOCK": "512"}),
]

QUICK_CASES = [
  "N=4 block=4",
  "N=256 block=256",
  "N=1024 block=512",
  "N=16384 block=512",
]

BASE_ENV = {
  "EN210_ALLOW_DIRTY": "1",
}


def _ensure_sock() -> None:
  sock = os.environ.get("EN210_TINYGPU_SOCK", "/tmp/tinygpu.sock")
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


def _slug(label: str, tag: str = "") -> str:
  return re.sub(r"[^A-Za-z0-9]+", "_", f"{tag}_{label}" if tag else label).strip("_")


def _parse_status(text: str, rc: int) -> str:
  if re.search(r"hardware_demo=ok", text):
    return "ok"
  m = re.search(r"mismatches=(\d+)/\d+", text)
  if m and int(m.group(1)) > 0:
    return f"FAIL mism={m.group(1)}"
  if "hardware_demo=FAIL" in text:
    return "FAIL"
  if "AssertionError" in text or "TimeoutError" in text or "RuntimeError" in text or rc != 0:
    err = re.search(r"((?:Assertion|Timeout|Runtime)Error):.*", text)
    return f"FAIL {err.group(0)}" if err else f"FAIL rc={rc}"
  return f"rc={rc}"


def _parse_kernel_ms(text: str) -> float | None:
  m = re.search(r"\[en210\] kernel_time_ms=([0-9.]+)", text)
  return float(m.group(1)) if m else None


def _parse_grid(text: str) -> str:
  m = re.search(r"\[en210\] kernel_time_ms=[^\n]*grid=(\d+)", text)
  return f"grid={m.group(1)}" if m else "?"


def run_case(label: str, extra: dict[str, str], *,
             tag: str = "", trial: int | None = None
             ) -> tuple[str, str, str, float, float | None]:
  LOG_DIR.mkdir(exist_ok=True)
  slug = _slug(label, tag)
  if trial is not None:
    slug = f"{slug}_t{trial}"
  log_path = LOG_DIR / f"bench-{slug}.log"
  err_path = LOG_DIR / f"bench-{slug}.err"
  env = {**os.environ, **BASE_ENV, **extra}
  t0 = time.time()
  with open(log_path, "w") as logf, open(err_path, "w") as errf:
    proc = subprocess.run(
        [sys.executable, "-u", str(ADD_PY)],
        env=env, stdout=logf, stderr=errf, cwd=str(SCRIPT_DIR))
  wall = time.time() - t0
  text = log_path.read_text(errors="replace") + err_path.read_text(errors="replace")
  return (label, _parse_grid(text), _parse_status(text, proc.returncode),
          wall, _parse_kernel_ms(text))


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


def _fmt_wall(mean: float, std: float) -> str:
  return f"{mean:.1f}±{std:.1f}s"


def _fmt_kern(mean: float, std: float) -> str:
  if mean < 1.0 and std < 1.0:
    return f"{mean * 1000:.0f}±{std * 1000:.0f}µs"
  return f"{mean:.2f}±{std:.2f}ms"


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--quick", action="store_true",
                  help="Only the four load-bearing cases")
  ap.add_argument("--repeat", type=int, default=1,
                  help="Repeat each case N times; report mean±std")
  ap.add_argument("--no-sock-start", action="store_true",
                  help="Do not attempt to start TinyGPU if sock is missing")
  args = ap.parse_args()
  if args.repeat < 1:
    raise SystemExit("--repeat must be >= 1")

  cases = CASES
  if args.quick:
    want = set(QUICK_CASES)
    cases = [(lab, env) for lab, env in CASES if lab in want]

  if not args.no_sock_start:
    _ensure_sock()

  agg: dict[str, dict] = {}
  failed = 0
  n_runs = 0
  hdr = (f"{'case':22} | {'t':>2} | {'result':10} | {'wall':7} | "
         f"{'kernel':9} | {'launch':10}")
  print(hdr, flush=True)
  print("-" * len(hdr), flush=True)
  for trial in range(1, args.repeat + 1):
    for label, extra in cases:
      lab, grid, status, wall, kern = run_case(label, extra, tag="run", trial=trial)
      n_runs += 1
      if status != "ok":
        failed += 1
      slot = agg.setdefault(label, {
          "grid": grid, "walls": [], "kerns": [], "oks": 0, "fails": 0})
      slot["grid"] = grid
      slot["walls"].append(wall)
      if kern is not None:
        slot["kerns"].append(kern)
      if status == "ok":
        slot["oks"] += 1
      else:
        slot["fails"] += 1
      print(f"{lab:22} | {trial:2d} | {status:10} | {wall:6.1f}s | "
            f"{_fmt_kernel(kern):9} | {grid:10}", flush=True)

  print(f"\n=== mean ± sample std (n={args.repeat}) ===", flush=True)
  print(f"{'case':22} | {'ok':4} | {'wall':14} | {'kernel':16} | {'launch':10}",
        flush=True)
  print("-" * 80, flush=True)

  summary_rows = []
  for label, slot in agg.items():
    wm, ws = _mean_std(slot["walls"])
    km, ks = _mean_std(slot["kerns"]) if slot["kerns"] else (float("nan"), float("nan"))
    ok_s = f"{slot['oks']}/{slot['oks'] + slot['fails']}"
    grid = slot["grid"]
    print(f"{label:22} | {ok_s:4} | {_fmt_wall(wm, ws):14} | "
          f"{_fmt_kern(km, ks):16} | {grid:10}", flush=True)
    summary_rows.append((label, ok_s, wm, ws, km, ks, grid))

  print("\n| Case | OK | Wall | Kernel | Launch |")
  print("|---|---|---|---|---|")
  for lab, ok_s, wm, ws, km, ks, grid in summary_rows:
    print(f"| {lab} | {ok_s} | {_fmt_wall(wm, ws)} | "
          f"{_fmt_kern(km, ks)} | {grid} |")
  print(f"\nbenchmark={'ok' if failed == 0 else 'FAIL'} "
        f"runs={n_runs} failed={failed} repeat={args.repeat}", flush=True)
  print("Note: kernel_time is host submit→poll (includes RPC overhead); "
        "mean±std over --repeat.", flush=True)
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
