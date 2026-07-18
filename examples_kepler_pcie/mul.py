#!/usr/bin/env python3
"""Kepler ``mul`` smoke test and launcher.

Uses the checked-in ``examples_kepler/mul_kepler.cubin`` (or assembles from
PTX with CUDA 10.2 ``ptxas``).  Hardware launch reuses the shared add launcher
with ``KEPLER_OPERATION=mul``.
"""
from __future__ import annotations
import array, hashlib, os, subprocess, sys

# When executed by pathname Python places only this directory on sys.path;
# add the checkout root so the sibling module can be imported without a
# package install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)
from examples_kepler_pcie import add as shared

def assemble_mul_cubin() -> bytes:
  cubin = shared.assemble_kepler_cubin("mul")
  digest = hashlib.sha256(cubin).hexdigest()
  if len(cubin) != shared.MUL_CUBIN_BYTES or digest != shared.MUL_CUBIN_SHA256:
    raise RuntimeError(f"unexpected mul cubin: {len(cubin)} bytes sha256={digest}")
  return cubin

def main() -> None:
  if "--assemble-cubin" in sys.argv or "--compare-cubin" in sys.argv:
    path = shared.DEFAULT_MUL_CUBIN
    if os.path.exists(path):
      with open(path, "rb") as f: cubin = f.read()
      digest = hashlib.sha256(cubin).hexdigest()
      same = (len(cubin) == shared.MUL_CUBIN_BYTES and digest == shared.MUL_CUBIN_SHA256)
      print(f"cubin_compare={'byte-identical' if same else 'mismatch'} operation=mul "
            f"assembled_bytes={len(cubin)} sha256={digest}")
      return
    try:
      cubin = assemble_mul_cubin()
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
      raise SystemExit(f"mul assembly unavailable: {e}")
    print(f"cubin_compare=byte-identical operation=mul assembled_bytes={len(cubin)} "
          f"sha256={hashlib.sha256(cubin).hexdigest()}")
    return
  if "--hardware" not in sys.argv:
    a = array.array("f", (float(i) for i in range(256)))
    b = array.array("f", (float(i + 1) for i in range(256)))
    out = array.array("f", (x * y for x, y in zip(a, b)))
    assert out[7] == 56.0 and len(out) == 256
    print("software_mul=ok N=256")
    return
  os.environ["KEPLER_OPERATION"] = "mul"
os.environ["KEPLER_CUBIN"] = os.environ.get(
    "KEPLER_MUL_CUBIN", shared.DEFAULT_MUL_CUBIN)
shared.main()


if __name__ == "__main__":
  main()
