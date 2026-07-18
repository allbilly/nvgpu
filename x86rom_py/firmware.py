"""Versioned firmware profile and reached BIOS interrupt services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from .constants import GK104_DEVICE_ID, NVIDIA_VENDOR_ID, ROM_SEGMENT
from .trace import TraceLog

if TYPE_CHECKING:
  from .bus import Bus
  from .cpu import CPU
  from .memory import RealModeMemory


class FirmwareError(RuntimeError):
  """Unknown interrupt function — abort rather than invent success."""


@dataclass
class PciFunction:
  bus: int
  device: int
  function: int
  vendor_id: int = NVIDIA_VENDOR_ID
  device_id: int = GK104_DEVICE_ID

  @property
  def bdf(self) -> str:
    return f"0000:{self.bus:02x}:{self.device:02x}.{self.function}"


@dataclass
class FirmwareProfile:
  """Deterministic BIOS environment for the pinned Palit ROM."""

  version: str = "palit-gtx770-v1"
  entry_ax: int = 0
  entry_bx: int = 0
  entry_cx: int = 0
  entry_dx: int = 0
  entry_si: int = 0
  entry_di: int = 0
  entry_ds: int = 0
  entry_es: int = 0
  entry_ss: int = 0x0000
  entry_sp: int = 0x7C00
  rom_segment: int = ROM_SEGMENT
  pci: PciFunction = field(default_factory=lambda: PciFunction(0, 9, 0))
  # Equipment word (INT 11h) and memory size KB (INT 12h).
  equipment_word: int = 0x0021  # diskette + coprocessor + 80x25
  memory_kb: int = 640
  # Video mode state for INT 10h.
  video_mode: int = 0x03
  # Timing: INT 15h AH=86h / AH=83h
  allow_timing: bool = True
  # PCI BIOS present.
  pci_bios: bool = True
  # Extra canned responses keyed by (int_no, ax).
  canned: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)

  def apply_entry_state(self, cpu: "CPU") -> None:
    cpu.eax = self.entry_ax
    cpu.ebx = self.entry_bx
    cpu.ecx = self.entry_cx
    cpu.edx = self.entry_dx
    cpu.esi = self.entry_si
    cpu.edi = self.entry_di
    cpu.ds = self.entry_ds
    cpu.es = self.entry_es
    cpu.ss = self.entry_ss
    cpu.esp = self.entry_sp
    cpu.cs = self.rom_segment
    cpu.eip = 0x0003
    cpu.eflags = 0x00000202  # IF set, reserved bit1

  def install_bda(self, mem: "RealModeMemory") -> None:
    # BDA at 0x400
    mem.write16(0x410, self.equipment_word)
    mem.write16(0x413, self.memory_kb)
    mem.write8(0x449, self.video_mode)
    mem.write16(0x44A, 80)  # columns
    # Soft reset flag / POST etc. left zero.


class FirmwareServices:
  """Dispatch BIOS interrupts reached by this ROM only."""

  def __init__(
    self,
    profile: FirmwareProfile,
    bus: "Bus",
    mem: "RealModeMemory",
    *,
    trace: Optional[TraceLog] = None,
  ) -> None:
    self.profile = profile
    self.bus = bus
    self.mem = mem
    self.trace = trace

  def handle(self, cpu: "CPU", vector: int) -> None:
    ah = (cpu.eax >> 8) & 0xFF
    al = cpu.eax & 0xFF
    ax = cpu.eax & 0xFFFF
    service = f"INT{vector:02X}_AH{ah:02X}"
    if self.trace is not None:
      self.trace.record(
        operation="firmware",
        address_space="firmware",
        direction="none",
        firmware_service=service,
        cs=cpu.cs,
        ip=cpu.eip & 0xFFFF,
        value=ax,
      )

    key = (vector, ax)
    if key in self.profile.canned:
      self._apply_regs(cpu, self.profile.canned[key])
      return

    if vector == 0x10:
      self._int10(cpu, ah, al)
    elif vector == 0x11:
      cpu.eax = (cpu.eax & 0xFFFF0000) | (self.profile.equipment_word & 0xFFFF)
    elif vector == 0x12:
      cpu.eax = (cpu.eax & 0xFFFF0000) | (self.profile.memory_kb & 0xFFFF)
    elif vector == 0x15:
      self._int15(cpu, ah, al)
    elif vector == 0x16:
      # Keyboard: return ZF=1 (no key) for non-blocking check AH=01
      if ah == 0x01:
        cpu.set_flag("ZF", True)
      elif ah == 0x00:
        # Blocking read — return space
        cpu.eax = (cpu.eax & 0xFFFF0000) | 0x3920
        cpu.set_flag("ZF", False)
      else:
        raise FirmwareError(f"unsupported INT 16h AH={ah:#x}")
    elif vector == 0x1A:
      self._int1a(cpu, ah, al)
    else:
      raise FirmwareError(f"unsupported interrupt INT {vector:#x} AH={ah:#x}")

  def _apply_regs(self, cpu: "CPU", regs: Dict[str, int]) -> None:
    for name, val in regs.items():
      if name == "CF":
        cpu.set_flag("CF", bool(val))
      elif hasattr(cpu, name.lower()):
        setattr(cpu, name.lower(), val)
      elif name.upper() in ("AX", "BX", "CX", "DX", "SI", "DI", "BP", "SP"):
        setattr(cpu, "e" + name.lower(), val & 0xFFFF)

  def _int10(self, cpu: "CPU", ah: int, al: int) -> None:
    if ah == 0x00:  # set mode
      self.profile.video_mode = al
      self.mem.write8(0x449, al)
    elif ah == 0x0F:  # get mode
      cpu.eax = (cpu.eax & 0xFFFF0000) | ((80 & 0xFF) << 8) | (self.profile.video_mode & 0xFF)
      cpu.ebx = (cpu.ebx & 0xFFFFFF00) | 0x00  # page 0
    elif ah == 0x01:  # set cursor shape — ignore
      pass
    elif ah == 0x02:  # set cursor pos — ignore
      pass
    elif ah == 0x03:  # get cursor
      cpu.ecx = 0x0607
      cpu.edx = 0
    elif ah == 0x0E:  # teletype — ignore
      pass
    elif ah == 0x11:
      # Character generator — font services (includes AL=04 path at 0x2f0a).
      # Acknowledge without side effects for offline conformance.
      pass
    elif ah == 0x12:
      # Alternate select — return safe defaults
      if al == 0x10:  # get EGA info
        cpu.ebx = (cpu.ebx & 0xFFFF0000) | 0x0003
        cpu.ecx = (cpu.ecx & 0xFFFF0000) | 0x0000
      else:
        pass
    else:
      raise FirmwareError(f"unsupported INT 10h AH={ah:#x} AL={al:#x}")

  def _int15(self, cpu: "CPU", ah: int, al: int) -> None:
    if ah == 0x86:  # wait CX:DX microseconds
      if not self.profile.allow_timing:
        raise FirmwareError("INT 15h AH=86h timing disabled")
      duration = ((cpu.ecx & 0xFFFF) << 16) | (cpu.edx & 0xFFFF)
      self.bus.delay_us(duration)
      cpu.set_flag("CF", False)
    elif ah == 0x83:  # event wait — stub success
      cpu.set_flag("CF", False)
    elif ah == 0x88:  # extended memory size
      cpu.eax = (cpu.eax & 0xFFFF0000) | 0x0000
      cpu.set_flag("CF", False)
    elif ah == 0xC0:  # return system config pointer — not present
      cpu.set_flag("CF", True)
      cpu.eax = (cpu.eax & 0xFFFF0000) | 0x8600
    elif ah == 0xE8 and al == 0x20:
      # Query system address map — return empty / CF
      cpu.set_flag("CF", True)
    else:
      raise FirmwareError(f"unsupported INT 15h AH={ah:#x} AL={al:#x}")

  def _int1a(self, cpu: "CPU", ah: int, al: int) -> None:
    if ah == 0x00:  # get system time
      cpu.ecx = 0
      cpu.edx = 0
      cpu.eax = cpu.eax & 0xFFFFFF00  # AL=0 midnight flag
    elif ah == 0x01:  # set system time
      pass
    elif ah == 0x02:  # get real-time clock
      cpu.ecx = 0x0000
      cpu.edx = 0x0000
      cpu.set_flag("CF", False)
    elif ah == 0xB1:
      self._pci_bios(cpu, al)
    else:
      raise FirmwareError(f"unsupported INT 1Ah AH={ah:#x}")

  def _pci_bios(self, cpu: "CPU", al: int) -> None:
    if not self.profile.pci_bios:
      cpu.set_flag("CF", True)
      cpu.eax = (cpu.eax & 0xFFFF0000) | 0xFF00
      return
    # AL = PCI BIOS function
    if al == 0x01:  # installation check
      cpu.edx = 0x20494350  # 'PCI '
      cpu.eax = (cpu.eax & 0xFFFF0000) | 0x0001  # AH=0 success, AL=hw mech
      cpu.ebx = (cpu.ebx & 0xFFFF0000) | 0x0210  # version 2.10
      cpu.ecx = (cpu.ecx & 0xFFFF0000) | 0x0000  # last bus
      cpu.set_flag("CF", False)
      return
    if al == 0x02:  # find PCI device
      vendor = cpu.edx & 0xFFFF
      device = (cpu.ecx & 0xFFFF)
      index = cpu.esi & 0xFFFF
      pci = self.profile.pci
      if vendor == pci.vendor_id and device == pci.device_id and index == 0:
        bx = (pci.bus << 8) | (pci.device << 3) | pci.function
        cpu.ebx = (cpu.ebx & 0xFFFF0000) | bx
        cpu.eax = cpu.eax & 0xFFFF00FF  # AH=0
        cpu.set_flag("CF", False)
      else:
        cpu.eax = (cpu.eax & 0xFFFF00FF) | 0x8600  # device not found
        cpu.set_flag("CF", True)
      return
    if al == 0x03:  # find PCI class code
      cpu.eax = (cpu.eax & 0xFFFF00FF) | 0x8600
      cpu.set_flag("CF", True)
      return
    if al in (0x08, 0x09, 0x0A):  # read config byte/word/dword
      width = {0x08: 1, 0x09: 2, 0x0A: 4}[al]
      bx = cpu.ebx & 0xFFFF
      di = cpu.edi & 0xFFFF
      bus, devfn = (bx >> 8) & 0xFF, bx & 0xFF
      pci = self.profile.pci
      if bus == pci.bus and devfn == ((pci.device << 3) | pci.function):
        val = self.bus.pci_read(di, width)
        if width == 1:
          cpu.ecx = (cpu.ecx & 0xFFFFFF00) | (val & 0xFF)
        elif width == 2:
          cpu.ecx = (cpu.ecx & 0xFFFF0000) | (val & 0xFFFF)
        else:
          cpu.ecx = val & 0xFFFFFFFF
        cpu.eax = cpu.eax & 0xFFFF00FF
        cpu.set_flag("CF", False)
      else:
        cpu.eax = (cpu.eax & 0xFFFF00FF) | 0x8700
        cpu.set_flag("CF", True)
      return
    if al in (0x0B, 0x0C, 0x0D):  # write config byte/word/dword
      width = {0x0B: 1, 0x0C: 2, 0x0D: 4}[al]
      bx = cpu.ebx & 0xFFFF
      di = cpu.edi & 0xFFFF
      bus, devfn = (bx >> 8) & 0xFF, bx & 0xFF
      pci = self.profile.pci
      if bus == pci.bus and devfn == ((pci.device << 3) | pci.function):
        if width == 1:
          val = cpu.ecx & 0xFF
        elif width == 2:
          val = cpu.ecx & 0xFFFF
        else:
          val = cpu.ecx & 0xFFFFFFFF
        self.bus.pci_write(di, width, val)
        cpu.eax = cpu.eax & 0xFFFF00FF
        cpu.set_flag("CF", False)
      else:
        cpu.eax = (cpu.eax & 0xFFFF00FF) | 0x8700
        cpu.set_flag("CF", True)
      return
    raise FirmwareError(f"unsupported PCI BIOS INT 1Ah/B1 AL={al:#x}")
