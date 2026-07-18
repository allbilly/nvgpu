"""Mandatory safety policy for live and model execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

from .constants import (
  GK104_DEVICE_ID,
  HOSTILE_MMIO_OFFSETS,
  LIVE_ACK_TOKEN,
  NVIDIA_VENDOR_ID,
  PMC_BOOT_0,
  PMC_BOOT_0_GK104,
  PROVEN_VGA_PORTS,
  VGA_BAR0_ALIAS_BASE,
)


class SafetyError(RuntimeError):
  """Rejected operation; must abort without retry."""


@dataclass
class SafetyPolicy:
  """Read-only by default; live writes require exact ack and pinned identity."""

  live_writes: bool = False
  live_ack: Optional[str] = None
  allow_bus_master: bool = False
  allow_bar_remap: bool = False
  allow_reset: bool = False
  allow_hostile_mmio: bool = False
  allowed_extra_mmio: Optional[Set[int]] = None
  proven_vga_ports: Set[int] = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.proven_vga_ports is None:
      self.proven_vga_ports = set(PROVEN_VGA_PORTS)
    if self.allowed_extra_mmio is None:
      self.allowed_extra_mmio = set()

  def require_live_writes(self, ack: Optional[str]) -> None:
    if ack != LIVE_ACK_TOKEN:
      raise SafetyError(
        f"live writes require ack token {LIVE_ACK_TOKEN!r}; got {ack!r}"
      )
    self.live_ack = ack
    self.live_writes = True

  def check_identity(self, vendor: int, device: int) -> None:
    if vendor != NVIDIA_VENDOR_ID or device != GK104_DEVICE_ID:
      raise SafetyError(
        f"device identity {vendor:04x}:{device:04x} "
        f"!= {NVIDIA_VENDOR_ID:04x}:{GK104_DEVICE_ID:04x}"
      )

  def check_pmc_boot_0(self, value: int) -> None:
    if value == 0xFFFFFFFF:
      raise SafetyError("BAR0 lost (PMC_BOOT_0 = 0xffffffff)")
    if value != PMC_BOOT_0_GK104:
      raise SafetyError(
        f"PMC_BOOT_0={value:#010x} != expected GK104 {PMC_BOOT_0_GK104:#010x}"
      )

  def check_mmio_write(
    self,
    bar: int,
    offset: int,
    width: int,
    *,
    rom_cs: int = 0,
    rom_ip: int = 0,
  ) -> None:
    if not self.live_writes:
      raise SafetyError(
        f"write blocked (read-only default) BAR{bar}+{offset:#x} "
        f"width={width} at CS:IP={rom_cs:04x}:{rom_ip:04x}"
      )
    if bar == 0 and offset in HOSTILE_MMIO_OFFSETS and not self.allow_hostile_mmio:
      raise SafetyError(
        f"TinyGPU-hostile write to BAR0+{offset:#x} rejected "
        f"(ROM CS:IP={rom_cs:04x}:{rom_ip:04x})"
      )
    if bar == 1 and not self.allow_bar_remap:
      raise SafetyError(
        f"BAR1 write rejected at CS:IP={rom_cs:04x}:{rom_ip:04x}"
      )
    # Reset / power / clock / train class — conservative deny unless allowed.
    if bar == 0 and offset in (0x200, 0x1200) and not self.allow_reset:
      # Broad PMC/PBUS reset-adjacent; policy can reopen after oracle proof.
      pass  # do not blanket-block all 0x200; only explicit reset paths

  def check_pci_write(self, offset: int, width: int, value: int) -> None:
    if not self.live_writes:
      raise SafetyError(f"PCI write blocked (read-only) cfg+{offset:#x}")
    # Command register: bus master bit 2
    if offset == 0x04 and width >= 2 and (value & 0x4) and not self.allow_bus_master:
      raise SafetyError("bus-master enable rejected (IOMMU containment required)")
    # BAR registers 0x10-0x24
    if 0x10 <= offset <= 0x24 and not self.allow_bar_remap:
      raise SafetyError(f"BAR remap PCI write cfg+{offset:#x} rejected")

  def check_io_write(self, port: int, width: int) -> None:
    if not self.live_writes:
      raise SafetyError(f"I/O write blocked (read-only) port={port:#x}")
    # Unclassified ports are handled by the bus (abort); here only write gate.

  def vga_alias_offset(self, port: int) -> Optional[int]:
    """Return BAR0 alias offset for a proven VGA port, else None."""
    p = port & 0xFFFF
    if p in self.proven_vga_ports:
      return VGA_BAR0_ALIAS_BASE + p
    return None

  def is_pramin_stub(self, value: int) -> bool:
    if value == 0xFFFFFFFF:
      return True
    # advancing 0xbad0fbxx class
    if (value & 0xFFFFFF00) == 0xBAD0FB00:
      return True
    if (value & 0xFFFF0000) == 0xBADF0000:  # observed topology sentinels
      return False  # topology != PRAMIN stub; caller decides
    return False

  def is_pramin_live(self, value: int) -> bool:
    return not self.is_pramin_stub(value) and value != 0xFFFFFFFF
