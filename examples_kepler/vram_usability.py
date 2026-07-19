#!/usr/bin/env python3
"""Probe whether the full 4 GiB GK104 framebuffer is uniquely R/W-able.

Boots the shared TinyGPU path, reads per-FBP size straps, then walks VRAM
through the BAR0 PRAMIN window (1 MiB, retargeted via 0x1700).  Sparse unique
markers detect dead ranges and bit19 / wrap aliasing without writing all 4 GiB.

  python3 examples_kepler/vram_usability.py
  KEPLER_VRAM_STRIDE=0x400000 python3 examples_kepler/vram_usability.py  # denser
"""
from __future__ import annotations

import os
import struct
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "ref")):
  if _p not in sys.path:
    sys.path.insert(0, _p)

os.environ.setdefault("KEPLER_LIVE_ACK", "completion-abort-risk")
os.environ.setdefault("KEPLER_RPC_LIGHT", "1")
os.environ.setdefault("KEPLER_BAR1_AFTER_POST", "1")
# Force classic PRAMIN (not BAR1 substitute) so addresses > mapped BAR1 work.
os.environ["KEPLER_FORCE_PRAMIN"] = "1"

import examples_kepler.add as pcie  # noqa: E402  (standalone stack)
import examples_kepler.add as mac  # noqa: E402

VRAM_SIZE = 4 << 30  # GTX 770 4GB


def _marker(pa: int) -> int:
  return (0xC0DE0000 ^ ((pa >> 12) & 0xffff) ^ ((pa >> 20) & 0xfff) << 8) & 0xffffffff


def _ensure_pramin(dev) -> None:
  """Prefer PRAMIN window over BAR1 identity diversion for >128 MiB PAs."""
  if getattr(dev, "_bar1_identity_ready", False):
    print("[vram] clearing _bar1_identity_ready so PRAMIN covers full FB",
          flush=True)
    setattr(dev, "_bar1_identity_ready", False)


def _store(dev, pa: int, val: int) -> None:
  pcie._gk104_pramin_write(dev, pa, struct.pack("<I", val))


def _load(dev, pa: int) -> int:
  return pcie._gk104_pramin_read32(dev, pa) & 0xffffffff


def _fbp_report(dev) -> int:
  """Sum per-FBP amounts (Nouveau gf100_ram_probe_fbpa_amount style)."""
  fbp_nr = (dev.read32(0x022438) & 0xffffffff) or 4
  # 0x022438 encoding varies; also try topology bits.
  parts = []
  total = 0
  for i, reg in enumerate((0x11020c, 0x11120c, 0x11220c, 0x11320c)):
    raw = dev.read32(reg) & 0xffffffff
    # Amount is often in MiB in low bits; accept several encodings.
    mib = raw & 0xffff
    if mib == 0 or mib > 8192:
      mib = (raw >> 20) & 0xfff
    parts.append((reg, raw, mib))
    if 0 < mib <= 4096:
      total += mib
  print(f"[vram] FBP straps 0x22438={dev.read32(0x022438):#x} "
        f"0x22554={dev.read32(0x022554):#x}", flush=True)
  for reg, raw, mib in parts:
    print(f"[vram]   {reg:#x}={raw:#010x} (~{mib} MiB decode)", flush=True)
  print(f"[vram]   summed FBP decode ≈ {total} MiB "
        f"({total / 1024:.2f} GiB)", flush=True)
  try:
    bar_addr, bar_sz = dev.dev_impl.hw.bar_info(1)
    print(f"[vram] PCI BAR1={bar_addr:#x} size={bar_sz:#x} "
          f"({bar_sz / (1 << 20):.0f} MiB aperture)", flush=True)
  except Exception as e:
    print(f"[vram] PCI BAR1 info unavailable: {e}", flush=True)
  return total << 20 if total else VRAM_SIZE


def sweep(dev, size: int, stride: int) -> tuple[int, int, list]:
  """Write unique markers, then verify. Returns (ok, fail, failures)."""
  pas = list(range(0, size, stride))
  if (size - 4) not in pas:
    pas.append(size - 4)
  # Always include classic bit19 pair bases near 0 / mid / high.
  for base in (0, 0x400000, 0x8000000, 0x10000000, 0x80000000, 0xFF000000):
    if base < size:
      pas.append(base)
      if base + 0x80000 < size:
        pas.append(base + 0x80000)
  pas = sorted(set(p & ~3 for p in pas if 0 <= p < size))
  print(f"[vram] sweep points={len(pas)} stride={stride:#x} "
        f"size={size:#x}", flush=True)
  t0 = time.perf_counter()
  for i, pa in enumerate(pas):
    _store(dev, pa, _marker(pa))
    if (i + 1) % 32 == 0 or i + 1 == len(pas):
      print(f"[vram]   wrote {i + 1}/{len(pas)} "
            f"(last={pa:#x})", flush=True)
  fails = []
  for i, pa in enumerate(pas):
    got = _load(dev, pa)
    want = _marker(pa)
    if got != want:
      fails.append((pa, want, got))
    if (i + 1) % 64 == 0 or i + 1 == len(pas):
      print(f"[vram]   verified {i + 1}/{len(pas)} fails={len(fails)}",
            flush=True)
  dt = time.perf_counter() - t0
  print(f"[vram] sweep done in {dt:.1f}s ok={len(pas) - len(fails)} "
        f"fail={len(fails)}", flush=True)
  return len(pas) - len(fails), len(fails), fails


def bit19_banks(dev, size: int, n_banks: int = 32) -> tuple[int, int]:
  """Write both halves of 1 MiB banks; both must survive."""
  ok = fail = 0
  step = max(1, (size // (1 << 20)) // n_banks)
  print(f"[vram] bit19 bank check every {step} MiB ({n_banks} samples)",
        flush=True)
  for bi in range(0, size // (1 << 20), step):
    lo = bi << 20
    hi = lo + 0x80000
    if hi + 4 > size:
      break
    a = (0xB1000000 | (bi & 0xffff)) & 0xffffffff
    b = (0xB2000000 | (bi & 0xffff)) & 0xffffffff
    _store(dev, lo, a)
    _store(dev, hi, b)
    ga, gb = _load(dev, lo), _load(dev, hi)
    if ga == a and gb == b:
      ok += 1
    else:
      fail += 1
      print(f"[vram]   BIT19 FAIL bank={bi} lo={lo:#x}:{ga:#x}/{a:#x} "
            f"hi={hi:#x}:{gb:#x}/{b:#x}", flush=True)
      if fail >= 8:
        print("[vram]   (stopping bit19 after 8 fails)", flush=True)
        break
  print(f"[vram] bit19 banks ok={ok} fail={fail}", flush=True)
  return ok, fail


def main() -> int:
  stride = int(os.environ.get("KEPLER_VRAM_STRIDE", "0x1000000"), 0)  # 16 MiB
  target = int(os.environ.get("KEPLER_VRAM_SIZE", hex(VRAM_SIZE)), 0)

  pcie.set_pci_transport_factory(mac._MacPCIDeviceFactory())
  print("[vram] booting NVDevice…", flush=True)
  t0 = time.perf_counter()
  # Quiet bring-up noise unless KEPLER_VRAM_VERBOSE=1
  if os.environ.get("KEPLER_VRAM_VERBOSE", "0") == "0":
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      dev = pcie.NVDevice("NV", backend="hardware")
  else:
    dev = pcie.NVDevice("NV", backend="hardware")
  print(f"[vram] boot {time.perf_counter() - t0:.1f}s "
        f"boot0={dev.read32(0):#x}", flush=True)

  try:
    _ensure_pramin(dev)
    decoded = _fbp_report(dev)
    # Prefer strap sum when sensible; else assume 4 GiB SKU.
    size = target
    if 0x40000000 <= decoded <= 0x100000000:
      size = min(target, decoded)
      print(f"[vram] using size={size:#x} from FBP straps (capped by target)",
            flush=True)
    else:
      print(f"[vram] FBP decode unusable; probing target={size:#x}", flush=True)

    # Sanity: first dword R/W
    _store(dev, 0, 0x11112222)
    if _load(dev, 0) != 0x11112222:
      print("[vram] FAIL: PA0 not writable via PRAMIN", flush=True)
      return 1

    ok, fail, fails = sweep(dev, size, stride)
    b_ok, b_fail = bit19_banks(dev, size)

    print("\n=== VRAM usability ===", flush=True)
    print(f"target={size:#x} ({size / (1 << 30):.2f} GiB)", flush=True)
    print(f"sparse markers: ok={ok} fail={fail}", flush=True)
    print(f"bit19 banks:    ok={b_ok} fail={b_fail}", flush=True)
    for pa, want, got in fails[:16]:
      print(f"  FAIL pa={pa:#x} want={want:#x} got={got:#x}", flush=True)
    if len(fails) > 16:
      print(f"  … {len(fails) - 16} more", flush=True)

    usable = fail == 0 and b_fail == 0
    print(f"\nvram_usability={'ok' if usable else 'FAIL'} "
          f"full_{size / (1 << 30):.2f}GiB={'yes' if usable else 'no'}",
          flush=True)
    return 0 if usable else 1
  finally:
    try:
      dev.close()
    except Exception:
      pass


if __name__ == "__main__":
  raise SystemExit(main())
