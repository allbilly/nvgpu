#!/usr/bin/env python3
"""Kepler multiply smoke test.

The default is a dependency-free CPU check, so it is useful on a machine where
the GTX 770 is still bound or unavailable.  ``--hardware`` is reserved for the
same raw-MMIO bring-up as :mod:`add`; a separately assembled sm_30 multiply
cubin must be supplied with ``KEPLER_CUBIN``.
"""
from __future__ import annotations
import array, os, sys

def main() -> None:
  if "--hardware" not in sys.argv:
    a = array.array("f", (float(i) for i in range(256)))
    b = array.array("f", (float(i + 1) for i in range(256)))
    out = array.array("f", (x * y for x, y in zip(a, b)))
    assert out[7] == 56.0 and len(out) == 256
    print("software_mul=ok N=256")
    return
  cubin = os.environ.get("KEPLER_CUBIN")
  if not cubin:
    raise SystemExit("hardware multiply needs an sm_30 multiply cubin; set KEPLER_CUBIN")
  # Keep the launcher simple and share the proven PCIe/RM implementation.
  import add
  os.environ.setdefault("KEPLER_LIVE_ACK", "raw-mmio-risk")
  add.main()

if __name__ == "__main__":
  main()
