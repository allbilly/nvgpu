#!/usr/bin/env python3
"""Print eGPU VRAM size (live + selftest).

Runs against the same vendored NV stack as examples/add.py (RTX 3080 eGPU over
PCIe/USB4 on macOS, routed through /Applications/TinyGPU.app). We do NOT touch
add.py — we import it as a module to reuse the boot path, autogen constants,
and the eGPU's already-proven RPC plumbing.
Two sources, both reported:

  1. PCI BAR1 aperture  — `pci_dev.bar_info(1)` returns (phys_addr, size)
     from TinyGPU.app. That's the VRAM BAR; with reBAR enabled it's the
     full VRAM, otherwise it's a 256 MB window onto the full VRAM.
  2. RM control probe   — `NV2080_CTRL_CMD_FB_GET_INFO` on the device's
     subdevice handle (would be authoritative). NOTE: the vendored autogen
     declares the params struct as 16 B opaque, but the real wire format
     is 56 B. Without a fully-described params struct the vendored
     `iface.rm_control` sends the wrong paramsSize and RM returns 0
     (no error). The probe still runs and is reported honestly; the
     authoritative number requires extending the autogen (out of scope
     for this script — `add.py` is not modified).

Usage:
  python3 examples/vram.py             # live: boots the eGPU and prints
  python3 examples/vram.py --selftest  # offline: validates pack/unpack + sizes
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
import time

# Reuse the vendored stack: autogen, NVDev, NVDevice, PCIIface, NV_GSP, etc.
# add.py is a single-file driver — importing it boots nothing on its own.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import add  # noqa: E402

# The vendored autogen (`nv_570`) declares `struct_NV2080_CTRL_FB_GET_INFO_PARAMS`
# as a 16-byte opaque blob, but the real wire format is 56 bytes:
#   u32 index, u32 reserved, u32 length, u32 reserved, u32 data[8]
# We use a flat c_ubyte array to get a 56-byte payload; the vendored
# `iface.rm_control` will serialize exactly 56 bytes for paramsSize.
# (The 16-byte autogen struct can't be used as-is — see print_vram() comment.)
_PARAMS_SIZE = 56
_PARAMS_T = ctypes.c_ubyte * _PARAMS_SIZE


def _make_params(index: int) -> "_PARAMS_T":
    """Allocate a 56-byte params buffer with index/length pre-filled."""
    buf = _PARAMS_T()  # ctypes zeros the buffer
    # +0x00: index (u32 LE), +0x08: length (u32 LE)
    struct.pack_into("<I", buf, 0x00, index)
    struct.pack_into("<I", buf, 0x08, 8)  # ask RM for 8 bytes
    return buf


def _read_u64(params_buf) -> int:
    """Read the u64 result from data[0:2] at +0x10 of the 56-byte params."""
    return struct.unpack_from("<Q", bytes(params_buf), 0x10)[0]


def _fb_get_info(dev, index: int) -> int:
    """Issue NV2080_CTRL_CMD_FB_GET_INFO on dev.subdevice; return u64 result."""
    params_in = _make_params(index)
    params_out = dev.iface.rm_control(dev.subdevice,
                                      add.nv_gpu.NV2080_CTRL_CMD_FB_GET_INFO,
                                      params_in)
    return _read_u64(params_out)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB" if n else "0"


def _mib(n: int) -> str:
    return f"{n / (1024 ** 2):.0f} MiB" if n else "0"

# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------
# tinygrad's NV stack reads the GPU's physical VRAM during boot from the
# `NV_PGC6_AON_SECURE_SCRATCH_GROUP_42` register and stashes it on
# `dev_impl.vram_size` (see add.py:667 — `self.vram_size = self.reg(...).read() << 20`).
# That is the authoritative physical framebuffer size, independent of
# reBAR/BAR1 aperture and independent of the (broken) RM info controls.
def print_vram(device: str = "NV") -> int:
    t0 = time.perf_counter()
    dev = add.NVDevice(device)
    boot_s = time.perf_counter() - t0
    iface = type(dev.iface).__name__
    is_remote = not dev.iface.is_local()
    print(f"device={dev.device}  iface={iface}  remote={is_remote}  boot={boot_s:.2f}s")

    # (1) Authoritative physical VRAM — read from the GPU's own scratch reg
    # during boot. This is the number tinygrad itself uses to size the
    # page-table allocator and WPR layout (see add.py:667, 1011).
    phys_vram = int(dev.iface.dev_impl.vram_size)
    large_bar = bool(getattr(dev.iface.dev_impl, "large_bar", False))
    print(f"Phys VRAM (reg):  {_gb(phys_vram)}  ({phys_vram} B)  large_bar={large_bar}")

    # (2) BAR1 aperture — the host-side view. With reBAR/large BAR, equals
    # (1); without, it's a 256 MiB window.
    bar_paddr, bar_size = dev.iface.pci_dev.bar_info(dev.iface.vram_bar)
    print(f"PCI BAR{dev.iface.vram_bar}:  phys=0x{bar_paddr:x}  size={_mib(bar_size)} ({bar_size} B)")

    # (3) RM control — best-effort cross-check.
    # The vendored autogen declares `struct_NV2080_CTRL_FB_GET_INFO_PARAMS` as
    # a 16 B opaque blob, but the real wire format is 56 B. We send 56 B via a
    # flat c_ubyte buffer, but RM zeros the response because the autogen-side
    # paramsSize sent to RM is the wrong shape. Reported for completeness; the
    # authoritative number is (1).
    rm_results = {}
    for label, sym in (("TOTAL_RAM_SIZE",  "NV2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE"),
                       ("USABLE_RAM_SIZE", "NV2080_CTRL_FB_INFO_INDEX_USABLE_RAM_SIZE"),
                       ("RAM_SIZE",        "NV2080_CTRL_FB_INFO_INDEX_RAM_SIZE")):
        try:
            rm_results[label] = _fb_get_info(dev, getattr(add.nv_gpu, sym))
        except Exception as e:
            rm_results[label] = f"<error: {type(e).__name__}: {e}>"
    for label, val in rm_results.items():
        if isinstance(val, int):
            print(f"RM {label:18s} {_gb(val)}  ({val} B)")
        else:
            print(f"RM {label:18s} {val}")
    rm_ok = all(isinstance(v, int) and v > 0 for v in rm_results.values())
    if not rm_ok:
        print("RM probe:        vendored autogen doesn't model "
              "FB_GET_INFO_PARAMS (16 B vs 56 B on the wire) — see Phys VRAM above")

    # (4) reBAR state — if BAR1 < phys VRAM, reBAR is off and the host
    # only sees a window onto the framebuffer.
    rebar_disabled = (bar_size < phys_vram)
    print(f"reBAR:           {'disabled (BAR1 is a window; rest paged via sysmem)' if rebar_disabled else 'enabled (BAR1 covers full VRAM)'}")
    return phys_vram

def selftest() -> None:
    # ctypes array: sizeof() = 56, bytes() = 56-byte payload, no alignment.
    assert ctypes.sizeof(_PARAMS_T) == 56
    assert len(_PARAMS_T()) == 56
    p = _make_params(0xdeadbeef)
    assert ctypes.sizeof(p) == 56
    assert bytes(p)[:0x10] == struct.pack("<II", 0xdeadbeef, 0) + struct.pack("<II", 8, 0)
    # Inject a u64 result at +0x10, simulate a response, read it back.
    raw = bytearray(bytes(p))
    struct.pack_into("<Q", raw, 0x10, 0x1122334455667788)
    p2 = _PARAMS_T.from_buffer(raw)
    assert _read_u64(p2) == 0x1122334455667788
    # Length on the input was 8; an RM response with length=0 must not crash.
    raw2 = bytearray(56)
    p3 = _PARAMS_T.from_buffer(raw2)
    assert _read_u64(p3) == 0

    # Required autogen symbols must be present
    for sym in ("NV2080_CTRL_CMD_FB_GET_INFO",
                "NV2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE",
                "NV2080_CTRL_FB_INFO_INDEX_USABLE_RAM_SIZE",
                "NV2080_CTRL_FB_INFO_INDEX_RAM_SIZE"):
        assert hasattr(add.nv_gpu, sym), f"missing autogen symbol: {sym}"

    # Formatter sanity
    assert _gb(1024 ** 3) == "1.00 GB"
    assert _gb(0) == "0"
    assert _mib(256 << 20) == "256 MiB"
    print("vram_selftest=ok struct_size=56 roundtrip=ok autogen_symbols=ok")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    total = print_vram()
    # Sanity floor: any real NV GPU has >= 1 GiB. If this fires, the boot
    # didn't actually read the scratch reg (or this is not an NV GPU).
    assert total > (1 << 30), f"phys VRAM implausibly small: {total} B"
    print(f"\nVRAM: {total / (1024 ** 3):.1f} GB")


if __name__ == "__main__":
    main()
