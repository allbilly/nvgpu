#!/usr/bin/env python3
"""Linux entrypoint for the GK104 stack.

The full self-contained implementation lives in ``examples_kepler/add.py``
(same shape as ``examples/add.py``).  This file re-exports that module so
existing ``from examples_kepler_pcie import add`` / ``python3
examples_kepler_pcie/add.py`` callers keep working.  On Linux the standalone
file selects ``LinuxPCIDevice`` (sysfs BAR mmap) by default.
"""
from __future__ import annotations
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEPLER_DIR = os.path.join(REPO_ROOT, "examples_kepler")
TINYGRAD_ROOT = os.path.join(REPO_ROOT, "ref")
for _path in (REPO_ROOT, TINYGRAD_ROOT, KEPLER_DIR):
  if _path not in sys.path:
    sys.path.insert(0, _path)

_IMPL_PATH = os.path.join(KEPLER_DIR, "add.py")
_spec = importlib.util.spec_from_file_location("examples_kepler_add", _IMPL_PATH)
if _spec is None or _spec.loader is None:
  raise ImportError(f"cannot load GK104 stack from {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _impl
_spec.loader.exec_module(_impl)

# Full re-export so `from examples_kepler_pcie import add as shared` stays complete.
globals().update(_impl.__dict__)
# Keep __file__ pointing at the standalone implementation so source-contract
# tests and inspect helpers see the real stack (not this thin entry).
main = _impl.main

if __name__ == "__main__":
  main()
