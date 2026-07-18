#!/usr/bin/env python3
"""GK104 macOS TinyGPU multiply entrypoint.

Same TinyGPU transport and bring-up as ``examples_kepler/add.py``, but launches
the sm_30 ``mul`` cubin (``out[i] = a[i] * b[i]``) and checks that result.

Live path::

  python3 examples_kepler/mul.py
"""
from __future__ import annotations
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _path in (REPO_ROOT, SCRIPT_DIR, os.path.join(REPO_ROOT, "ref")):
  if _path not in sys.path:
    sys.path.insert(0, _path)

os.environ["KEPLER_OPERATION"] = "mul"
_mul_cubin = os.path.join(SCRIPT_DIR, "mul_kepler.cubin")
# Force the mul image: a lingering KEPLER_CUBIN=add_kepler.cubin from an add
# session must not win (setdefault would keep the wrong ELF).
_cur = os.environ.get("KEPLER_CUBIN", "")
if (not _cur or os.path.basename(_cur) == "add_kepler.cubin"
    or os.environ.get("KEPLER_MUL_CUBIN")):
  os.environ["KEPLER_CUBIN"] = os.environ.get("KEPLER_MUL_CUBIN", _mul_cubin)

import add as kepler_add  # noqa: E402  (same-dir TinyGPU wrapper)


def main() -> None:
  if "--software" in sys.argv:
    import array
    a = array.array("f", (float(i) for i in range(256)))
    b = array.array("f", (float(i + 1) for i in range(256)))
    out = array.array("f", (x * y for x, y in zip(a, b)))
    assert out[7] == 56.0 and len(out) == 256
    print("software_mul=ok N=256")
    return
  kepler_add.main()


if __name__ == "__main__":
  main()
