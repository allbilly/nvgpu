"""Conventional memory, IVT, BDA, stack, and ROM mapping at 0xC0000."""

from __future__ import annotations

from typing import Optional, Tuple

from .bus import Bus, BusError
from .constants import ROM_SEGMENT


class MemoryError(RuntimeError):
  """Unsupported physical aperture or illegal ROM write."""


class RealModeMemory:
  """1 MiB conventional memory with optional A20 wrap and BAR translation."""

  def __init__(
    self,
    rom_bytes: bytes,
    *,
    rom_segment: int = ROM_SEGMENT,
    bus: Optional[Bus] = None,
    size: int = 0x100000,
    allow_a20_wrap: bool = True,
    rom_shadow_writable: bool = True,
  ) -> None:
    self.ram = bytearray(size)
    self.rom_segment = rom_segment & 0xFFFF
    self.rom_phys = self.rom_segment << 4
    self.rom_bytes = bytes(rom_bytes)
    self.rom_end = self.rom_phys + len(self.rom_bytes)
    self.bus = bus
    self.allow_a20_wrap = allow_a20_wrap
    # BIOS copies the option ROM into conventional memory at C000:0 so the
    # image may self-modify.  That shadow is RAM.  Writes to the PCI expansion
    # ROM BAR aperture remain forbidden (handled via bus physical decode).
    self.rom_shadow_writable = rom_shadow_writable
    self._map_rom()

  def _map_rom(self) -> None:
    end = min(len(self.ram), self.rom_end)
    n = end - self.rom_phys
    if n > 0:
      self.ram[self.rom_phys:end] = self.rom_bytes[:n]

  def phys(self, addr: int) -> int:
    if self.allow_a20_wrap:
      return addr & 0xFFFFF
    return addr & 0xFFFFFFFF

  def linear(self, seg: int, off: int) -> int:
    return self.phys(((seg & 0xFFFF) << 4) + (off & 0xFFFF))

  def in_rom(self, phys: int) -> bool:
    p = self.phys(phys)
    return self.rom_phys <= p < self.rom_end

  def read8(self, phys: int) -> int:
    p = self.phys(phys)
    if p < len(self.ram):
      return self.ram[p]
    return self._phys_bus_read(p, 1)

  def write8(self, phys: int, value: int) -> None:
    p = self.phys(phys)
    if self.in_rom(p) and not self.rom_shadow_writable:
      raise MemoryError(f"write to ROM at phys {p:#x}")
    if p < len(self.ram):
      self.ram[p] = value & 0xFF
      return
    self._phys_bus_write(p, 1, value & 0xFF)

  def read16(self, phys: int) -> int:
    return self.read8(phys) | (self.read8(phys + 1) << 8)

  def write16(self, phys: int, value: int) -> None:
    self.write8(phys, value & 0xFF)
    self.write8(phys + 1, (value >> 8) & 0xFF)

  def read32(self, phys: int) -> int:
    return (
      self.read8(phys)
      | (self.read8(phys + 1) << 8)
      | (self.read8(phys + 2) << 16)
      | (self.read8(phys + 3) << 24)
    )

  def write32(self, phys: int, value: int) -> None:
    for i in range(4):
      self.write8(phys + i, (value >> (8 * i)) & 0xFF)

  def read_seg(self, seg: int, off: int, width: int) -> int:
    p = self.linear(seg, off)
    if width == 1:
      return self.read8(p)
    if width == 2:
      return self.read16(p)
    if width == 4:
      return self.read32(p)
    raise MemoryError(f"bad width {width}")

  def write_seg(self, seg: int, off: int, width: int, value: int) -> None:
    p = self.linear(seg, off)
    if width == 1:
      self.write8(p, value)
    elif width == 2:
      self.write16(p, value)
    elif width == 4:
      self.write32(p, value)
    else:
      raise MemoryError(f"bad width {width}")

  def fetch(self, seg: int, ip: int, n: int) -> bytes:
    p = self.linear(seg, ip)
    out = bytearray()
    for i in range(n):
      out.append(self.read8(p + i))
    return bytes(out)

  def install_ivt_entry(self, vector: int, cs: int, ip: int) -> None:
    off = (vector & 0xFF) * 4
    self.write16(off, ip & 0xFFFF)
    self.write16(off + 2, cs & 0xFFFF)

  def get_ivt_entry(self, vector: int) -> Tuple[int, int]:
    off = (vector & 0xFF) * 4
    ip = self.read16(off)
    cs = self.read16(off + 2)
    return cs, ip

  def _phys_bus_read(self, phys: int, width: int) -> int:
    if self.bus is None:
      raise MemoryError(f"physical read outside conventional RAM: {phys:#x}")
    hit = self._resolve_mmio(phys)
    if hit is None:
      raise MemoryError(f"unsupported physical aperture read {phys:#x}")
    bar, offset = hit
    return self.bus.mmio_read(bar, offset, width)

  def _phys_bus_write(self, phys: int, width: int, value: int) -> None:
    if self.bus is None:
      raise MemoryError(f"physical write outside conventional RAM: {phys:#x}")
    hit = self._resolve_mmio(phys)
    if hit is None:
      raise MemoryError(f"unsupported physical aperture write {phys:#x}")
    bar, offset = hit
    self.bus.mmio_write(bar, offset, width, value)

  def _resolve_mmio(self, phys: int) -> Optional[Tuple[int, int]]:
    bars = getattr(self.bus, "bars", None)
    if not bars:
      return None
    for bar, win in bars.items():
      if getattr(win, "is_io", False):
        continue
      if win.base <= phys < win.base + win.size:
        return bar, phys - win.base
    return None
