"""Three-space hardware bus: PCI config, legacy I/O, and BAR MMIO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .constants import (
  PCI_CONFIG_ADDR_PORT,
  PCI_CONFIG_DATA_PORT,
  PROVEN_VGA_PORTS,
  VGA_BAR0_ALIAS_BASE,
)
from .safety import SafetyError, SafetyPolicy
from .trace import TraceEvent, TraceLog


class BusError(RuntimeError):
  """Unsupported or divergent bus operation."""


@dataclass
class ProbeResult:
  name: str
  ok: bool
  detail: str = ""
  values: Dict[str, int] = field(default_factory=dict)


class Bus:
  """Common three-space bus protocol."""

  def pci_read(self, offset: int, width: int) -> int:
    raise NotImplementedError

  def pci_write(self, offset: int, width: int, value: int) -> None:
    raise NotImplementedError

  def io_read(self, port: int, width: int) -> int:
    raise NotImplementedError

  def io_write(self, port: int, width: int, value: int) -> None:
    raise NotImplementedError

  def mmio_read(self, bar: int, offset: int, width: int) -> int:
    raise NotImplementedError

  def mmio_write(self, bar: int, offset: int, width: int, value: int) -> None:
    raise NotImplementedError

  def delay_us(self, duration: int) -> None:
    pass

  def checkpoint(self, name: str) -> ProbeResult:
    return ProbeResult(name=name, ok=True)


def _mask(width: int) -> int:
  if width == 1:
    return 0xFF
  if width == 2:
    return 0xFFFF
  if width == 4:
    return 0xFFFFFFFF
  raise BusError(f"unsupported width {width}")


def _merge(old: int, value: int, offset: int, width: int) -> int:
  """Merge a width-sized value into a 32-bit dword at byte offset."""
  shift = (offset & 3) * 8
  m = _mask(width) << shift
  return (old & ~m) | ((value & _mask(width)) << shift)


def _extract(dword: int, offset: int, width: int) -> int:
  shift = (offset & 3) * 8
  return (dword >> shift) & _mask(width)


@dataclass
class BarWindow:
  base: int
  size: int
  is_io: bool = False


class ModelBus(Bus):
  """Deterministic register/config model for unit tests."""

  def __init__(
    self,
    *,
    vendor: int = 0x10DE,
    device: int = 0x1184,
    bars: Optional[Dict[int, BarWindow]] = None,
    mmio: Optional[Dict[Tuple[int, int], int]] = None,
    io_ports: Optional[Dict[int, int]] = None,
    safety: Optional[SafetyPolicy] = None,
    trace: Optional[TraceLog] = None,
    bdf: str = "0000:09:00.0",
  ) -> None:
    self.vendor = vendor
    self.device = device
    self.bdf = bdf
    self.safety = safety or SafetyPolicy(live_writes=True, live_ack="model")
    # Model bus allows writes for offline tests; identity still checked on live.
    self.safety.live_writes = True
    self.trace = trace
    self.config = bytearray(256)
    self.config[0:2] = vendor.to_bytes(2, "little")
    self.config[2:4] = device.to_bytes(2, "little")
    self.config[4] = 0x00  # command
    self.config[5] = 0x00
    self.config[6] = 0x10  # status
    self.config[0x0A] = 0x00
    self.config[0x0B] = 0x03  # VGA display controller
    self.bars = bars or {
      0: BarWindow(base=0xA0000000, size=0x1000000, is_io=False),
      1: BarWindow(base=0xB0000000, size=0x10000000, is_io=False),
      2: BarWindow(base=0xC0000000, size=0x2000000, is_io=False),
      # GK104's legacy native-I/O aperture is PCI BAR5 (config dword 0x24).
      # The option ROM discovers this exact dword before issuing the H75
      # +0/+4 and indexed +8/+12 accesses.
      5: BarWindow(base=0x0000E000, size=0x80, is_io=True),
    }
    self._program_bars()
    # Sparse MMIO: key (bar, byte_offset) -> byte value; also dword helpers.
    self._mmio_bytes: Dict[Tuple[int, int], int] = {}
    if mmio:
      for (bar, off), val in mmio.items():
        self._store_mmio(bar, off, 4, val)
    # PMC_BOOT_0 default
    self._store_mmio(0, 0, 4, 0x0E4040A2)
    self._io: Dict[int, int] = dict(io_ports or {})
    self._cfg_addr = 0  # 0xCF8 latch
    self.cs = 0
    self.ip = 0
    self.phase = "model"

  def _program_bars(self) -> None:
    for bar, win in self.bars.items():
      reg = 0x10 + bar * 4
      if win.is_io:
        val = (win.base & ~0x3) | 0x1
      else:
        val = win.base & ~0xF
      self.config[reg:reg + 4] = val.to_bytes(4, "little")

  def set_cs_ip(self, cs: int, ip: int) -> None:
    self.cs, self.ip = cs & 0xFFFF, ip & 0xFFFF

  def _emit(self, **kwargs: Any) -> None:
    if self.trace is None:
      return
    kwargs.setdefault("cs", self.cs)
    kwargs.setdefault("ip", self.ip)
    kwargs.setdefault("phase", self.phase)
    kwargs.setdefault("bdf", self.bdf)
    self.trace.record(**kwargs)

  def _store_mmio(self, bar: int, offset: int, width: int, value: int) -> None:
    for i in range(width):
      self._mmio_bytes[(bar, offset + i)] = (value >> (8 * i)) & 0xFF

  def _load_mmio(self, bar: int, offset: int, width: int) -> int:
    val = 0
    for i in range(width):
      val |= self._mmio_bytes.get((bar, offset + i), 0xFF) << (8 * i)
    return val

  def pci_read(self, offset: int, width: int) -> int:
    if offset < 0 or offset + width > len(self.config):
      raise BusError(f"PCI read OOB {offset:#x} width={width}")
    raw = int.from_bytes(self.config[offset:offset + width], "little")
    self._emit(
      operation="pci_read", address_space="pci", direction="read",
      width=width, canonical_offset=offset, raw_address=offset,
      read_result=raw,
    )
    return raw

  def pci_write(self, offset: int, width: int, value: int) -> None:
    self.safety.check_pci_write(offset, width, value)
    value &= _mask(width)
    # BAR size probes: write all-ones then read back sizing — model keeps programmed base
    # unless value is a size probe pattern; for simplicity accept programmed writes when allowed.
    if 0x10 <= offset <= 0x24 and not self.safety.allow_bar_remap:
      # Allow writes that restore the same base (ROM may rewrite BARs with same value).
      cur = int.from_bytes(self.config[offset:offset + width], "little")
      if (value & ~0xF) != (cur & ~0xF) and value != 0xFFFFFFFF & _mask(width):
        raise SafetyError(f"BAR remap rejected cfg+{offset:#x} val={value:#x}")
    for i in range(width):
      self.config[offset + i] = (value >> (8 * i)) & 0xFF
    self._emit(
      operation="pci_write", address_space="pci", direction="write",
      width=width, canonical_offset=offset, raw_address=offset, value=value,
    )

  def _resolve_bar(self, phys: int) -> Optional[Tuple[int, int]]:
    for bar, win in self.bars.items():
      if win.is_io:
        continue
      if win.base <= phys < win.base + win.size:
        return bar, phys - win.base
    return None

  def _resolve_io_bar(self, port: int) -> Optional[Tuple[int, int]]:
    for bar, win in self.bars.items():
      if not win.is_io:
        continue
      base = win.base & ~0x3
      if base <= port < base + win.size:
        return bar, port - base
    return None

  def io_read(self, port: int, width: int) -> int:
    port &= 0xFFFF
    # PCI mechanism #1
    if port in (PCI_CONFIG_ADDR_PORT, PCI_CONFIG_ADDR_PORT + 1,
                PCI_CONFIG_ADDR_PORT + 2, PCI_CONFIG_ADDR_PORT + 3):
      shift = (port - PCI_CONFIG_ADDR_PORT) * 8
      val = (self._cfg_addr >> shift) & _mask(width)
      self._emit(
        operation="io_read", address_space="io", direction="read",
        width=width, raw_address=port, canonical_offset=port, read_result=val,
      )
      return val
    if PCI_CONFIG_DATA_PORT <= port < PCI_CONFIG_DATA_PORT + 4:
      cfg_off = (self._cfg_addr & 0xFC) + (port - PCI_CONFIG_DATA_PORT)
      # Only respond when enable bit set and bus/dev/fn match our model endpoint.
      if self._cfg_addr & 0x80000000:
        val = self.pci_read(cfg_off, width)
      else:
        val = _mask(width)
      self._emit(
        operation="io_read", address_space="io", direction="read",
        width=width, raw_address=port, canonical_offset=port, read_result=val,
        dependency_tag="pci_mech1",
      )
      return val

    # Proven VGA → byte-width BAR0 alias (split wider accesses into bytes).
    if any((port + i) in PROVEN_VGA_PORTS for i in range(width)):
      val = 0
      for i in range(width):
        p = (port + i) & 0xFFFF
        if p not in PROVEN_VGA_PORTS:
          raise BusError(f"partial VGA port span {port:#x}+{width}")
        alias = VGA_BAR0_ALIAS_BASE + p
        b = self.mmio_read(0, alias, 1)
        val |= (b & 0xFF) << (8 * i)
        self._emit(
          operation="io_read", address_space="io", direction="read",
          width=1, raw_address=p, canonical_offset=p, read_result=b,
          dependency_tag="vga_alias",
        )
      return val

    io_hit = self._resolve_io_bar(port)
    if io_hit is not None:
      bar, off = io_hit
      val = self._load_mmio(bar, off, width)
      self._emit(
        operation="io_read", address_space="io", direction="read",
        width=width, raw_address=port, canonical_offset=off, bar=bar,
        read_result=val, dependency_tag="io_bar",
      )
      return val

    if port in self._io:
      val = self._io[port] & _mask(width)
      self._emit(
        operation="io_read", address_space="io", direction="read",
        width=width, raw_address=port, canonical_offset=port, read_result=val,
      )
      return val

    raise BusError(f"unclassified I/O read port={port:#x} width={width}")

  def io_write(self, port: int, width: int, value: int) -> None:
    port &= 0xFFFF
    value &= _mask(width)
    self.safety.check_io_write(port, width)

    if PCI_CONFIG_ADDR_PORT <= port < PCI_CONFIG_ADDR_PORT + 4:
      shift = (port - PCI_CONFIG_ADDR_PORT) * 8
      m = _mask(width) << shift
      self._cfg_addr = (self._cfg_addr & ~m) | (value << shift)
      self._emit(
        operation="io_write", address_space="io", direction="write",
        width=width, raw_address=port, canonical_offset=port, value=value,
        dependency_tag="pci_mech1",
      )
      return
    if PCI_CONFIG_DATA_PORT <= port < PCI_CONFIG_DATA_PORT + 4:
      cfg_off = (self._cfg_addr & 0xFC) + (port - PCI_CONFIG_DATA_PORT)
      if self._cfg_addr & 0x80000000:
        self.pci_write(cfg_off, width, value)
      self._emit(
        operation="io_write", address_space="io", direction="write",
        width=width, raw_address=port, canonical_offset=port, value=value,
        dependency_tag="pci_mech1",
      )
      return

    if any((port + i) in PROVEN_VGA_PORTS for i in range(width)):
      for i in range(width):
        p = (port + i) & 0xFFFF
        if p not in PROVEN_VGA_PORTS:
          raise BusError(f"partial VGA port span {port:#x}+{width}")
        alias = VGA_BAR0_ALIAS_BASE + p
        self.mmio_write(0, alias, 1, (value >> (8 * i)) & 0xFF)
        self._emit(
          operation="io_write", address_space="io", direction="write",
          width=1, raw_address=p, canonical_offset=p,
          value=(value >> (8 * i)) & 0xFF,
          dependency_tag="vga_alias",
        )
      return

    io_hit = self._resolve_io_bar(port)
    if io_hit is not None:
      bar, off = io_hit
      self._store_mmio(bar, off, width, value)
      self._emit(
        operation="io_write", address_space="io", direction="write",
        width=width, raw_address=port, canonical_offset=off, bar=bar,
        value=value, dependency_tag="io_bar",
      )
      return

    self._io[port] = value
    self._emit(
      operation="io_write", address_space="io", direction="write",
      width=width, raw_address=port, canonical_offset=port, value=value,
    )

  def mmio_read(self, bar: int, offset: int, width: int) -> int:
    val = self._load_mmio(bar, offset, width)
    win = self.bars.get(bar)
    raw = (win.base + offset) if win and not win.is_io else offset
    self._emit(
      operation="mmio_read", address_space="mmio", direction="read",
      width=width, bar=bar, canonical_offset=offset, raw_address=raw,
      read_result=val,
    )
    return val

  def mmio_write(self, bar: int, offset: int, width: int, value: int) -> None:
    self.safety.check_mmio_write(bar, offset, width, rom_cs=self.cs, rom_ip=self.ip)
    value &= _mask(width)
    # Never combine adjacent byte writes — store exactly `width` bytes.
    self._store_mmio(bar, offset, width, value)
    win = self.bars.get(bar)
    raw = (win.base + offset) if win and not win.is_io else offset
    self._emit(
      operation="mmio_write", address_space="mmio", direction="write",
      width=width, bar=bar, canonical_offset=offset, raw_address=raw,
      value=value,
    )

  def delay_us(self, duration: int) -> None:
    self._emit(
      operation="delay", address_space="cpu", direction="none",
      width=0, value=duration,
    )

  def checkpoint(self, name: str) -> ProbeResult:
    pramin = self.mmio_read(0, 0x700000, 4)
    pmc = self.mmio_read(0, 0, 4)
    live = self.safety.is_pramin_live(pramin)
    detail = f"PRAMIN={pramin:#010x} PMC_BOOT_0={pmc:#010x}"
    self._emit(
      operation="checkpoint", address_space="checkpoint", direction="none",
      dependency_tag=name, read_result=pramin, value=pmc,
    )
    return ProbeResult(
      name=name, ok=True, detail=detail,
      values={"pramin": pramin, "pmc_boot_0": pmc, "pramin_live": int(live)},
    )


class RecordingBus(Bus):
  """Records all activity from another bus."""

  def __init__(self, inner: Bus, trace: Optional[TraceLog] = None) -> None:
    self.inner = inner
    self.trace = trace or TraceLog()
    self.cs = 0
    self.ip = 0
    self.phase = "record"

  def set_cs_ip(self, cs: int, ip: int) -> None:
    self.cs, self.ip = cs & 0xFFFF, ip & 0xFFFF
    if hasattr(self.inner, "set_cs_ip"):
      self.inner.set_cs_ip(cs, ip)

  def _emit(self, **kwargs: Any) -> None:
    kwargs.setdefault("cs", self.cs)
    kwargs.setdefault("ip", self.ip)
    kwargs.setdefault("phase", self.phase)
    self.trace.record(**kwargs)

  def pci_read(self, offset: int, width: int) -> int:
    val = self.inner.pci_read(offset, width)
    self._emit(
      operation="pci_read", address_space="pci", direction="read",
      width=width, canonical_offset=offset, raw_address=offset, read_result=val,
    )
    return val

  def pci_write(self, offset: int, width: int, value: int) -> None:
    self.inner.pci_write(offset, width, value)
    self._emit(
      operation="pci_write", address_space="pci", direction="write",
      width=width, canonical_offset=offset, raw_address=offset, value=value,
    )

  def io_read(self, port: int, width: int) -> int:
    val = self.inner.io_read(port, width)
    self._emit(
      operation="io_read", address_space="io", direction="read",
      width=width, raw_address=port, canonical_offset=port, read_result=val,
    )
    return val

  def io_write(self, port: int, width: int, value: int) -> None:
    self.inner.io_write(port, width, value)
    self._emit(
      operation="io_write", address_space="io", direction="write",
      width=width, raw_address=port, canonical_offset=port, value=value,
    )

  def mmio_read(self, bar: int, offset: int, width: int) -> int:
    val = self.inner.mmio_read(bar, offset, width)
    self._emit(
      operation="mmio_read", address_space="mmio", direction="read",
      width=width, bar=bar, canonical_offset=offset, raw_address=offset,
      read_result=val,
    )
    return val

  def mmio_write(self, bar: int, offset: int, width: int, value: int) -> None:
    self.inner.mmio_write(bar, offset, width, value)
    self._emit(
      operation="mmio_write", address_space="mmio", direction="write",
      width=width, bar=bar, canonical_offset=offset, raw_address=offset,
      value=value,
    )

  def delay_us(self, duration: int) -> None:
    self.inner.delay_us(duration)
    self._emit(operation="delay", address_space="cpu", direction="none", value=duration)

  def checkpoint(self, name: str) -> ProbeResult:
    res = self.inner.checkpoint(name)
    self._emit(
      operation="checkpoint", address_space="checkpoint", direction="none",
      dependency_tag=name, read_result=res.values.get("pramin"),
    )
    return res


@dataclass
class ExpectedOp:
  operation: str
  address_space: str
  direction: str
  width: int
  canonical_offset: int = 0
  bar: Optional[int] = None
  value: Optional[int] = None
  read_result: Optional[int] = None
  raw_address: int = 0


class ReplayBus(Bus):
  """Consumes expected reads and checks emitted ops against a golden trace."""

  def __init__(
    self,
    expected: Sequence[TraceEvent | ExpectedOp],
    *,
    read_results: Optional[Dict[int, int]] = None,
  ) -> None:
    self.expected: List[Any] = list(expected)
    self.index = 0
    self.read_results = dict(read_results or {})
    self.cs = 0
    self.ip = 0

  def set_cs_ip(self, cs: int, ip: int) -> None:
    self.cs, self.ip = cs & 0xFFFF, ip & 0xFFFF

  def _next(self, operation: str, address_space: str, direction: str,
            width: int, **fields: Any) -> Any:
    if self.index >= len(self.expected):
      raise BusError(
        f"replay: extra op {operation} {address_space} {direction} "
        f"width={width} at CS:IP={self.cs:04x}:{self.ip:04x}"
      )
    exp = self.expected[self.index]
    self.index += 1
    if isinstance(exp, TraceEvent):
      e_op, e_as, e_dir, e_w = exp.operation, exp.address_space, exp.direction, exp.width
      e_off = exp.canonical_offset
      e_bar = exp.bar
      e_val = exp.value
      e_rr = exp.read_result
    else:
      e_op, e_as, e_dir, e_w = exp.operation, exp.address_space, exp.direction, exp.width
      e_off = exp.canonical_offset
      e_bar = exp.bar
      e_val = exp.value
      e_rr = exp.read_result
    if (e_op, e_as, e_dir, e_w) != (operation, address_space, direction, width):
      raise BusError(
        f"replay divergence@{self.index - 1}: expected "
        f"{e_op}/{e_as}/{e_dir}/w{e_w}, got "
        f"{operation}/{address_space}/{direction}/w{width}"
      )
    if "canonical_offset" in fields and fields["canonical_offset"] != e_off:
      raise BusError(
        f"replay offset {fields['canonical_offset']:#x} != expected {e_off:#x}"
      )
    if "bar" in fields and e_bar is not None and fields["bar"] != e_bar:
      raise BusError(f"replay bar {fields['bar']} != expected {e_bar}")
    if direction == "write" and e_val is not None and fields.get("value") != e_val:
      raise BusError(f"replay write value {fields.get('value'):#x} != {e_val:#x}")
    return e_rr

  def pci_read(self, offset: int, width: int) -> int:
    rr = self._next("pci_read", "pci", "read", width, canonical_offset=offset)
    if rr is None:
      rr = self.read_results.get(self.index - 1, 0)
    return int(rr)

  def pci_write(self, offset: int, width: int, value: int) -> None:
    self._next("pci_write", "pci", "write", width, canonical_offset=offset, value=value)

  def io_read(self, port: int, width: int) -> int:
    rr = self._next("io_read", "io", "read", width, canonical_offset=port)
    if rr is None:
      rr = self.read_results.get(self.index - 1, 0)
    return int(rr)

  def io_write(self, port: int, width: int, value: int) -> None:
    self._next("io_write", "io", "write", width, canonical_offset=port, value=value)

  def mmio_read(self, bar: int, offset: int, width: int) -> int:
    rr = self._next("mmio_read", "mmio", "read", width, canonical_offset=offset, bar=bar)
    if rr is None:
      rr = self.read_results.get(self.index - 1, 0)
    return int(rr)

  def mmio_write(self, bar: int, offset: int, width: int, value: int) -> None:
    self._next(
      "mmio_write", "mmio", "write", width,
      canonical_offset=offset, bar=bar, value=value,
    )

  def delay_us(self, duration: int) -> None:
    if self.index < len(self.expected):
      exp = self.expected[self.index]
      op = exp.operation if isinstance(exp, TraceEvent) else exp.operation
      if op == "delay":
        self.index += 1

  def checkpoint(self, name: str) -> ProbeResult:
    return ProbeResult(name=name, ok=True)

  def done(self) -> bool:
    return self.index >= len(self.expected)


class LiveBus(Bus):
  """Wraps APLRemotePCIDevice or LinuxPCIDevice without duplicating transports.

  Expects duck-typed methods: read_config/write_config, mmio_read/mmio_write
  (bytes) or mmio_read32/mmio_write32, optional read8/write8, bar_info.
  """

  def __init__(
    self,
    device: Any,
    *,
    safety: Optional[SafetyPolicy] = None,
    trace: Optional[TraceLog] = None,
    bdf: str = "",
  ) -> None:
    self.dev = device
    self.safety = safety or SafetyPolicy()
    self.trace = trace
    self.bdf = bdf
    self.cs = 0
    self.ip = 0
    self.phase = "live"
    self._cfg_addr = 0
    self._bar_bases: Dict[int, int] = {}
    if hasattr(device, "bar_info"):
      for bar in range(6):
        try:
          base, size = device.bar_info(bar)
          if size:
            self._bar_bases[bar] = int(base)
        except Exception:
          pass

  def set_cs_ip(self, cs: int, ip: int) -> None:
    self.cs, self.ip = cs & 0xFFFF, ip & 0xFFFF

  def _emit(self, **kwargs: Any) -> None:
    if self.trace is None:
      return
    kwargs.setdefault("cs", self.cs)
    kwargs.setdefault("ip", self.ip)
    kwargs.setdefault("phase", self.phase)
    kwargs.setdefault("bdf", self.bdf)
    self.trace.record(**kwargs)

  def pci_read(self, offset: int, width: int) -> int:
    if hasattr(self.dev, "read_config"):
      val = int(self.dev.read_config(offset, width))
    else:
      raise BusError("device lacks read_config")
    self._emit(
      operation="pci_read", address_space="pci", direction="read",
      width=width, canonical_offset=offset, read_result=val,
    )
    return val & _mask(width)

  def pci_write(self, offset: int, width: int, value: int) -> None:
    self.safety.check_pci_write(offset, width, value)
    if hasattr(self.dev, "write_config"):
      self.dev.write_config(offset, value & _mask(width), width)
    else:
      raise BusError("device lacks write_config")
    self._emit(
      operation="pci_write", address_space="pci", direction="write",
      width=width, canonical_offset=offset, value=value & _mask(width),
    )

  def mmio_read(self, bar: int, offset: int, width: int) -> int:
    val = self._dev_mmio_read(bar, offset, width)
    raw = self._bar_bases.get(bar, 0) + offset
    self._emit(
      operation="mmio_read", address_space="mmio", direction="read",
      width=width, bar=bar, canonical_offset=offset, raw_address=raw,
      read_result=val,
    )
    if bar == 0 and offset == 0 and val == 0xFFFFFFFF:
      raise SafetyError("BAR0 lost (all ones)")
    return val

  def mmio_write(self, bar: int, offset: int, width: int, value: int) -> None:
    self.safety.check_mmio_write(bar, offset, width, rom_cs=self.cs, rom_ip=self.ip)
    self._dev_mmio_write(bar, offset, width, value & _mask(width))
    raw = self._bar_bases.get(bar, 0) + offset
    self._emit(
      operation="mmio_write", address_space="mmio", direction="write",
      width=width, bar=bar, canonical_offset=offset, raw_address=raw,
      value=value & _mask(width),
    )

  def _dev_mmio_read(self, bar: int, offset: int, width: int) -> int:
    if width == 1 and hasattr(self.dev, "mmio_read"):
      data = self.dev.mmio_read(bar, offset, 1)
      return data[0] if isinstance(data, (bytes, bytearray)) else int(data) & 0xFF
    if width == 4 and hasattr(self.dev, "mmio_read32"):
      return int(self.dev.mmio_read32(bar, offset)) & 0xFFFFFFFF
    if hasattr(self.dev, "mmio_read"):
      data = self.dev.mmio_read(bar, offset, width)
      if isinstance(data, (bytes, bytearray)):
        return int.from_bytes(data[:width], "little")
      return int(data) & _mask(width)
    raise BusError("device lacks mmio_read")

  def _dev_mmio_write(self, bar: int, offset: int, width: int, value: int) -> None:
    if width == 1 and hasattr(self.dev, "mmio_write"):
      self.dev.mmio_write(bar, offset, bytes((value & 0xFF,)))
      return
    if width == 4 and hasattr(self.dev, "mmio_write32"):
      self.dev.mmio_write32(bar, offset, value & 0xFFFFFFFF)
      return
    if hasattr(self.dev, "mmio_write"):
      self.dev.mmio_write(bar, offset, value.to_bytes(width, "little"))
      return
    raise BusError("device lacks mmio_write")

  def io_read(self, port: int, width: int) -> int:
    port &= 0xFFFF
    if PCI_CONFIG_ADDR_PORT <= port < PCI_CONFIG_ADDR_PORT + 4:
      shift = (port - PCI_CONFIG_ADDR_PORT) * 8
      val = (self._cfg_addr >> shift) & _mask(width)
      self._emit(
        operation="io_read", address_space="io", direction="read",
        width=width, raw_address=port, canonical_offset=port, read_result=val,
      )
      return val
    if PCI_CONFIG_DATA_PORT <= port < PCI_CONFIG_DATA_PORT + 4:
      cfg_off = (self._cfg_addr & 0xFC) + (port - PCI_CONFIG_DATA_PORT)
      if self._cfg_addr & 0x80000000:
        val = self.pci_read(cfg_off, width)
      else:
        val = _mask(width)
      self._emit(
        operation="io_read", address_space="io", direction="read",
        width=width, raw_address=port, canonical_offset=port, read_result=val,
      )
      return val
    if any((port + i) in PROVEN_VGA_PORTS for i in range(width)):
      val = 0
      for i in range(width):
        p = (port + i) & 0xFFFF
        if p not in PROVEN_VGA_PORTS:
          raise BusError(f"partial VGA port span {port:#x}+{width}")
        alias = VGA_BAR0_ALIAS_BASE + p
        b = self.mmio_read(0, alias, 1)
        val |= (b & 0xFF) << (8 * i)
      return val
    raise BusError(
      f"unclassified live I/O port {port:#x} — TinyGPU has no native PCI I/O; abort"
    )

  def io_write(self, port: int, width: int, value: int) -> None:
    port &= 0xFFFF
    value &= _mask(width)
    self.safety.check_io_write(port, width)
    if PCI_CONFIG_ADDR_PORT <= port < PCI_CONFIG_ADDR_PORT + 4:
      shift = (port - PCI_CONFIG_ADDR_PORT) * 8
      m = _mask(width) << shift
      self._cfg_addr = (self._cfg_addr & ~m) | (value << shift)
      self._emit(
        operation="io_write", address_space="io", direction="write",
        width=width, raw_address=port, canonical_offset=port, value=value,
      )
      return
    if PCI_CONFIG_DATA_PORT <= port < PCI_CONFIG_DATA_PORT + 4:
      cfg_off = (self._cfg_addr & 0xFC) + (port - PCI_CONFIG_DATA_PORT)
      if self._cfg_addr & 0x80000000:
        self.pci_write(cfg_off, width, value)
      self._emit(
        operation="io_write", address_space="io", direction="write",
        width=width, raw_address=port, canonical_offset=port, value=value,
      )
      return
    if any((port + i) in PROVEN_VGA_PORTS for i in range(width)):
      for i in range(width):
        p = (port + i) & 0xFFFF
        if p not in PROVEN_VGA_PORTS:
          raise BusError(f"partial VGA port span {port:#x}+{width}")
        alias = VGA_BAR0_ALIAS_BASE + p
        self.mmio_write(0, alias, 1, (value >> (8 * i)) & 0xFF)
      return
    raise BusError(
      f"unclassified live I/O port {port:#x} — abort rather than invent a mapping"
    )

  def delay_us(self, duration: int) -> None:
    self._emit(operation="delay", address_space="cpu", direction="none", value=duration)

  def checkpoint(self, name: str) -> ProbeResult:
    pmc = self.mmio_read(0, 0, 4)
    self.safety.check_pmc_boot_0(pmc)
    pramin = self.mmio_read(0, 0x700000, 4)
    live = self.safety.is_pramin_live(pramin)
    tag = "pramin-live" if live and name == "pramin-live" else name
    self._emit(
      operation="checkpoint", address_space="checkpoint", direction="none",
      dependency_tag=tag, read_result=pramin, value=pmc,
    )
    return ProbeResult(
      name=name, ok=True,
      detail=f"PRAMIN={pramin:#010x} live={live}",
      values={"pramin": pramin, "pmc_boot_0": pmc, "pramin_live": int(live)},
    )
