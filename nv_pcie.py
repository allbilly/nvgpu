#!/usr/bin/env python3
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref" / "tinygrad"))

from tinygrad.device import Device


def open_pcie_device():
  dev = Device["NV"]
  iface_name = type(dev.iface).__name__
  if iface_name != "PCIIface":
    raise RuntimeError(f"expected PCIIface for this example path, got {iface_name}")
  return dev
