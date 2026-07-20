#!/usr/bin/env python3
"""Live GK104 add/mul benchmark for add_660ti.py (TinyGPU USB4 path).

Sweeps N from 1 to 524288, measuring wall time and host-measured kernel time
(GP_PUT → semaphore done).  Reports per-window breakdown for windowed runs.

The benchmark respects KEPLER_MAX_WINDOWS=64 (the H25 sysmem leak fix allows
64 windows; previously 64+ crashed macOS via mmap exhaustion).

Examples::

  python3 examples_kepler/benchmark_660ti.py
  python3 examples_kepler/benchmark_660ti.py --op add --quick
  python3 examples_kepler/benchmark_660ti.py --op add --repeat 3
  python3 examples_kepler/benchmark_660ti.py --op mul --n 256,4096,16384
  python3 examples_kepler/benchmark_660ti.py --boost --compare-boost --quick

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
ADD_PY = SCRIPT_DIR / "add_660ti.py"
LOG_DIR = REPO_ROOT / "logs"

# N values for the full sweep.  Covers single-CTA, multi-CTA, and windowed.
FULL_N_VALUES = [1, 8, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072,
                 262144, 524288, 1048576]

# Quick subset: one from each path tier.
QUICK_N_VALUES = [256, 8192, 16384, 65536, 262144]

BASE_ENV = {
    "KEPLER_ALLOW_DIRTY": "1",
    "KEPLER_SEED": "42",
    "KEPLER_AUTO_WARM_CONTINUE": "1",
    "KEPLER_LIVE_ACK": "completion-abort-risk",
    "KEPLER_RAMCFG_STRAP": "5",
    "KEPLER_MAX_WINDOWS": "64",
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


def _parse_launch_path(text: str) -> str:
    """Extract the launch mode: single-CTA, multi-CTA, or windowed."""
    m = re.search(r"\[kepler\] launch N=\d+ ([^\n]+)", text)
    if not m:
        return "?"
    launch = m.group(1).strip()
    if "multi-CTA" in launch:
        g = re.search(r"grid=\((\d+),", launch)
        return f"multi-CTA({g.group(1)})" if g else "multi-CTA"
    if "channel_windows" in launch:
        w = re.search(r"channel_windows=(\d+)", launch)
        return f"windowed({w.group(1)})" if w else "windowed"
    return "single-CTA"


def _parse_status(text: str, rc: int) -> str:
    if re.search(r"hardware_demo=ok", text):
        return "ok"
    m = re.search(r"mismatches=(\d+)/\d+", text)
    if m and int(m.group(1)) > 0:
        return f"FAIL mism={m.group(1)}"
    if "AssertionError" in text or "TimeoutError" in text or "RuntimeError" in text:
        err = re.search(r"((?:Assertion|Timeout|Runtime|Value)Error):[^\n]*", text)
        return f"FAIL {err.group(0)[:60]}" if err else f"FAIL rc={rc}"
    if rc != 0:
        return f"FAIL rc={rc}"
    return f"rc={rc}"


def _parse_kernel_ms(text: str) -> float | None:
    """Sum kernel_time_ms across all windows (or single for non-windowed)."""
    totals = re.findall(
        r"\[kepler\] kernel_time_total_ms=([0-9.]+)", text)
    if totals:
        return float(totals[-1])
    parts = [float(x) for x in re.findall(
        r"\[kepler\] kernel_time_ms=([0-9.]+)", text)]
    return sum(parts) if parts else None


def _parse_window_count(text: str) -> int:
    """Count channel windows actually executed."""
    return len(re.findall(r"\[kepler\] channel window offset=", text))


def _parse_gpc_coef(text: str) -> str:
    afters = re.findall(
        r"clk after experimental-pstate:.*?0x137004=(0x[0-9a-f]+)", text)
    return afters[-1] if afters else "-"


def run_case(op: str, n: int, extra: dict[str, str], *,
             tag: str = "", trial: int | None = None
             ) -> dict:
    """Run one benchmark case and return parsed results."""
    LOG_DIR.mkdir(exist_ok=True)
    slug = f"{op}_n{n}"
    if tag:
        slug += f"_{tag}"
    if trial is not None:
        slug += f"_t{trial}"
    log_path = LOG_DIR / f"bench660-{slug}.log"
    err_path = LOG_DIR / f"bench660-{slug}.err"
    env = {**os.environ, **BASE_ENV, **extra,
           "KEPLER_OPERATION": op, "KEPLER_N": str(n)}
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
    text = (log_path.read_text(errors="replace")
            + err_path.read_text(errors="replace"))
    return {
        "op": op,
        "n": n,
        "tag": tag,
        "trial": trial,
        "path": _parse_launch_path(text),
        "status": _parse_status(text, proc.returncode),
        "wall_s": wall,
        "kernel_ms": _parse_kernel_ms(text),
        "windows": _parse_window_count(text),
        "coef": _parse_gpc_coef(text),
        "log": str(log_path),
    }


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
    if math.isnan(mean):
        return "-"
    if mean < 1.0 and std < 1.0:
        return f"{mean * 1000:.0f}±{std * 1000:.0f}µs"
    return f"{mean:.2f}±{std:.2f}ms"


def _fmt_throughput(n: int, ms: float) -> str:
    """Elements per second from N and kernel time."""
    if ms <= 0 or math.isnan(ms):
        return "-"
    eps = n / (ms / 1000.0)
    if eps >= 1e9:
        return f"{eps / 1e9:.2f}G/s"
    if eps >= 1e6:
        return f"{eps / 1e6:.1f}M/s"
    return f"{eps / 1e3:.0f}K/s"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="add,mul",
                    help="Comma-separated: add, mul (default: both)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick subset: 5 representative N values")
    ap.add_argument("--n", type=str, default=None,
                    help="Comma-separated N values (overrides default sweep)")
    ap.add_argument("--boost", action="store_true",
                    help="Enable experimental GPC boost")
    ap.add_argument("--compare-boost", action="store_true",
                    help="Run each case baseline then boost; print delta")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat each case N times; report mean±std")
    ap.add_argument("--no-sock-start", action="store_true",
                    help="Do not auto-start TinyGPU if sock is missing")
    args = ap.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    ops = [o.strip() for o in args.op.split(",") if o.strip()]
    for o in ops:
        if o not in ("add", "mul"):
            raise SystemExit(f"unsupported op {o!r}; expected add or mul")

    if args.n:
        n_values = [int(x.strip()) for x in args.n.split(",")]
    elif args.quick:
        n_values = QUICK_N_VALUES
    else:
        n_values = FULL_N_VALUES

    if not args.no_sock_start:
        _ensure_sock()

    modes: list[tuple[str, dict[str, str]]] = [("", {})]
    if args.compare_boost:
        modes = [("base", {}), ("boost", dict(BOOST_ENV))]
    elif args.boost:
        modes = [("boost", dict(BOOST_ENV))]

    # key = (op, mode_name, n) -> aggregated results
    agg: dict[tuple[str, str, int], dict] = {}
    failed = 0
    n_runs = 0

    hdr = (f"{'op':3} | {'mode':5} | {'N':>7} | {'t':>2} | "
           f"{'result':12} | {'path':14} | {'win':>3} | "
           f"{'wall':7} | {'kernel':9} | {'thruput':8} | coef")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for trial in range(1, args.repeat + 1):
        for op in ops:
            for n in n_values:
                for mode, mextra in modes:
                    mode_name = mode or ("boost" if args.boost else "base")
                    r = run_case(op, n, mextra, tag=mode or "run",
                                 trial=trial)
                    n_runs += 1
                    if r["status"] != "ok":
                        failed += 1
                    key = (op, mode_name, n)
                    slot = agg.setdefault(key, {
                        "path": r["path"], "windows": r["windows"],
                        "walls": [], "kerns": [], "coefs": [],
                        "oks": 0, "fails": 0})
                    slot["path"] = r["path"]
                    slot["windows"] = r["windows"]
                    slot["walls"].append(r["wall_s"])
                    if r["kernel_ms"] is not None:
                        slot["kerns"].append(r["kernel_ms"])
                    if r["coef"] != "-":
                        slot["coefs"].append(r["coef"])
                    if r["status"] == "ok":
                        slot["oks"] += 1
                    else:
                        slot["fails"] += 1
                    print(f"{op:3} | {mode_name:5} | {n:7d} | {trial:2d} | "
                          f"{r['status']:12} | {r['path']:14} | "
                          f"{r['windows']:3d} | {r['wall_s']:6.1f}s | "
                          f"{_fmt_kernel(r['kernel_ms']):9} | "
                          f"{_fmt_throughput(n, r['kernel_ms'] or 0):8} | "
                          f"{r['coef']}", flush=True)

    # Summary table
    print(f"\n=== mean ± sample std (repeat={args.repeat}) ===", flush=True)
    sum_hdr = (f"{'op':3} | {'mode':5} | {'N':>7} | {'ok':4} | "
               f"{'path':14} | {'win':>3} | {'wall':14} | {'kernel':16} | "
               f"{'thruput':8} | coef | Δk(mean)")
    print(sum_hdr, flush=True)
    print("-" * len(sum_hdr), flush=True)

    summary_rows = []
    base_kern: dict[tuple[str, int], float] = {}
    for (op, mode_name, n), slot in sorted(agg.items()):
        wm, ws = _mean_std(slot["walls"])
        km, ks = (_mean_std(slot["kerns"]) if slot["kerns"]
                  else (float("nan"), float("nan")))
        coef = slot["coefs"][-1] if slot["coefs"] else "-"
        ok_s = f"{slot['oks']}/{slot['oks'] + slot['fails']}"
        dk = ""
        if mode_name == "base" and not math.isnan(km):
            base_kern[(op, n)] = km
        elif mode_name == "boost" and (op, n) in base_kern and not math.isnan(km):
            d = km - base_kern[(op, n)]
            dk = f"{d * 1000:+.0f}µs" if abs(d) < 1 else f"{d:+.2f}ms"
        thr = _fmt_throughput(n, km)
        print(f"{op:3} | {mode_name:5} | {n:7d} | {ok_s:4} | "
              f"{slot['path']:14} | {slot['windows']:3d} | "
              f"{_fmt_wall(wm, ws):14} | {_fmt_kern(km, ks):16} | "
              f"{thr:8} | {coef} | {dk}", flush=True)
        summary_rows.append((op, mode_name, n, ok_s, slot["path"],
                             slot["windows"], wm, ws, km, ks, thr, coef, dk))

    # Markdown table
    print(f"\n| Op | Mode | N | OK | Path | Win | Wall | Kernel | Thruput | Coef | Δk |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in summary_rows:
        op, mode, n, ok_s, path, win, wm, ws, km, ks, thr, coef, dk = row
        print(f"| {op} | {mode} | {n} | {ok_s} | {path} | {win} | "
              f"{_fmt_wall(wm, ws)} | {_fmt_kern(km, ks)} | {thr} | "
              f"{coef} | {dk or '-'} |")

    print(f"\nbenchmark={'ok' if failed == 0 else 'FAIL'} "
          f"runs={n_runs} failed={failed} repeat={args.repeat}", flush=True)
    print("Note: kernel_time = host GP_PUT→semaphore poll (includes TinyGPU "
          "RPC overhead).  Wall = full process time (cold bring-up + "
          "context copy + L2 warming + kernel).", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
