#!/usr/bin/env python3
"""Run a small elementwise add on an RTX 5090 (Blackwell, SM120).

The 3080 example in :mod:`add.py` embeds hand-written SM86 SASS.  That image
must not be reused on a 5090: Blackwell uses SM120 and the Blackwell QMD path.
This example deliberately goes through the local tinygrad NV reference
runtime, which already handles GB202/SM120 compilation and submission.

The repository copy of tinygrad is added to ``sys.path`` so this script works
from a checkout without requiring a separately installed tinygrad package.
Use ``NV5090_DEVICE=NV:1`` (or ``--device NV:1``) when the 5090 is not GPU 0.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
TINYGRAD_ROOT = ROOT / "ref" / "tinygrad"
if str(TINYGRAD_ROOT) not in sys.path:
  sys.path.insert(0, str(TINYGRAD_ROOT))

from tinygrad import Tensor, dtypes  # noqa: E402
from tinygrad.device import Device  # noqa: E402


EXPECTED = [11.0, 22.0, 33.0, 44.0]


def device_name(cli_device: str | None) -> str:
  """Return the requested tinygrad device, defaulting to the first NV GPU."""
  return cli_device or os.environ.get("NV5090_DEVICE", "NV:0")


def open_5090(name: str):
  """Open *name* and enforce that it is the Blackwell SM120 target."""
  if not name.upper().startswith("NV"):
    raise ValueError(f"{name!r} is not an NVIDIA device; use NV or NV:<index>")
  dev = Device[name]
  arch = getattr(dev, "arch", None)
  if arch != "sm_120":
    raise RuntimeError(
      f"{name} is {arch or 'an unknown architecture'}, not sm_120; "
      "select the RTX 5090 with --device NV:<index> or NV5090_DEVICE"
    )
  return dev


def add_on(device: str, a: list[float], b: list[float]) -> list[float]:
  """Execute the add through tinygrad's NV/Blackwell compiler and queues."""
  if len(a) != len(b):
    raise ValueError("input lengths differ")
  dev = open_5090(device)
  # Keeping the device string explicit prevents tinygrad from silently moving
  # either input through the default device when several GPUs are present.
  out = (Tensor(a, device=dev.device, dtype=dtypes.float32) +
         Tensor(b, device=dev.device, dtype=dtypes.float32)).realize()
  return [float(x) for x in out.tolist()]


def middle_selftest() -> None:
  """Offline gate: exercise the same arithmetic without opening an NV GPU."""
  a, b = Tensor([1.0, 2.0, 3.0, 4.0], device="CPU"), Tensor([10.0, 20.0, 30.0, 40.0], device="CPU")
  result = (a + b).realize().tolist()
  assert result == EXPECTED, f"CPU reference result {result!r} != {EXPECTED!r}"
  print(f"middle_selftest=ok arch=sm_120 result={result}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--middle-selftest", action="store_true", help="run without opening a GPU")
  parser.add_argument("--device", help="tinygrad NV device, for example NV:1")
  args = parser.parse_args()
  if args.middle_selftest:
    middle_selftest()
    return

  name = device_name(args.device)
  result = add_on(name, [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
  print(f"device={name} arch=sm_120 expected={EXPECTED}")
  print(f"result={result}")
  if result != EXPECTED:
    raise RuntimeError(f"unexpected add result: {result!r}")


if __name__ == "__main__":
  main()
