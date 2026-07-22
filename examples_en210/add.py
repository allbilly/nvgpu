#!/usr/bin/env python3
"""Compute kernel launch for ASUS EN210 (GT218, GeForce 210, sm_12).

Standalone self-contained script: runs a vector-add kernel out[i] = a[i] + b[i].
Inlines VBIOS devinit, FIFO/GR init, channel creation, and compute launch.

Transport: TinyGPU.app DriverKit Unix socket (/tmp/tinygpu.sock) on macOS.

Environment variables:
  EN210_N       Number of elements (default: 4)
  EN210_BLOCK   Block dimension (default: N)

Binary data files (in same directory):
  en210.rom     GT218 VBIOS dump (64KB)
  ctxprog.py    Generated context program
  ctxvals.bin   GR context values (310KB)

Kernel SASS is embedded in this file (88 bytes, verified from CUDA 6.5 nvcc).
If ptxas 6.5 is available (EN210_PTXAS env var or cuda65-bin/ptxas), it will
compile the embedded PTX source instead.  add_sass.bin is optional (fallback).
"""
from __future__ import annotations
import os, sys, struct, time, socket, enum

# ============================================================================
# VBIOS init script interpreter (ported from Nouveau nvbios_init.c)
# ============================================================================
class NvbiosI2CError(RuntimeError):
  """Protocol/bus absence error with the errno Nouveau returns to NVINIT."""
  def __init__(self, message: str, errno: int = 5):
    super().__init__(message)
    self.errno = int(errno)

  @property
  def u8(self) -> int:
    return (-self.errno) & 0xff

class NvbiosInit:
  """State for one nvbios_exec() invocation."""

  def __init__(self, dev, image: bytes, debug: bool = False):
    self.dev = dev
    self.image = image
    self.debug = debug

    # nvbios_init state
    self.offset = 0
    self.execute = 1
    self.nested = 0
    self.repeat = 0
    self.repend = 0
    self.ramcfg = 0
    self.head = -1
    self.or_ = -1
    self.link = 0
    self.outp = None

    # parsed BIT tables
    self.tables = self._parse_bit_tables()

    # cache for ramcfg
    self._ramcfg_value = None
    self._i2c_inited: set[int] = set()

  # --------------------------------------------------------------------------
  # BIT table helpers
  # --------------------------------------------------------------------------
  def _bit_entry(self, ident: bytes):
    """Return (data_offset, version) of the BIT table entry for `ident` or (0,0)."""
    bit = self.image.find(b"BIT")
    if bit < 2 or bit + 12 > len(self.image):
      return 0, 0
    hlen, rlen, count = self.image[bit + 6], self.image[bit + 7], self.image[bit + 8]
    for i in range(count):
      p = bit - 2 + hlen + i * rlen
      if p + 6 > len(self.image):
        break
      if self.image[p] == ident[0]:
        return struct.unpack_from("<H", self.image, p + 4)[0], self.image[p + 1]
    return 0, 0

  def _parse_bit_tables(self) -> dict:
    """Read the pointer table from the BIT 'I' record."""
    off, _ = self._bit_entry(b"I")
    t = {}
    if off and off + 18 <= len(self.image):
      t["script"]       = struct.unpack_from("<H", self.image, off + 0)[0]
      t["macro_index"]  = struct.unpack_from("<H", self.image, off + 2)[0]
      t["macro"]        = struct.unpack_from("<H", self.image, off + 4)[0]
      t["condition"]    = struct.unpack_from("<H", self.image, off + 6)[0]
      t["io_condition"] = struct.unpack_from("<H", self.image, off + 8)[0]
      t["io_flag"]      = struct.unpack_from("<H", self.image, off + 10)[0]
      t["function"]     = struct.unpack_from("<H", self.image, off + 12)[0]
      t["unknown"]      = struct.unpack_from("<H", self.image, off + 14)[0]
      t["xlat"]         = struct.unpack_from("<H", self.image, off + 16)[0]
    # M table for RAMCFG
    moff, mver = self._bit_entry(b"M")
    if moff and moff + 7 <= len(self.image):
      t["m"] = moff
      t["m_ver"] = mver
      # M v2 layout: count @0, xlat @1, M0203T pointer @3
      if mver == 2 and moff + 7 <= len(self.image):
        t["m_xlat"] = struct.unpack_from("<H", self.image, moff + 1)[0]
        t["m0203t"] = struct.unpack_from("<H", self.image, moff + 3)[0]
      elif mver == 1 and moff + 5 <= len(self.image):
        t["m_xlat"] = struct.unpack_from("<H", self.image, moff + 3)[0]
        t["m0203t"] = 0
    else:
      t["m"] = 0
      t["m_ver"] = 0
    return t

  # --------------------------------------------------------------------------
  # Image accessors
  # --------------------------------------------------------------------------
  def rd08(self, addr: int) -> int:
    return self.image[addr] if addr < len(self.image) else 0
  def rd16(self, addr: int) -> int:
    return struct.unpack_from("<H", self.image, addr)[0] if addr + 2 <= len(self.image) else 0
  def rd32(self, addr: int) -> int:
    return struct.unpack_from("<I", self.image, addr)[0] if addr + 4 <= len(self.image) else 0

  # --------------------------------------------------------------------------
  # Register / I/O accessors
  # --------------------------------------------------------------------------
  def _nvreg(self, reg: int) -> int:
    """init_nvreg: strip lower bits and per-head/per-or translations."""
    reg &= ~0x00000003
    # For GK104 (NV_50+) head/OR translations.  We have only one head/or.
    if reg & 0x80000000:
      reg += (0 if self.head < 0 else self.head) * 0x800
      reg &= ~0x80000000
    if reg & 0x40000000:
      reg += (0 if self.or_ < 0 else self.or_) * 0x800
      reg &= ~0x40000000
      if reg & 0x20000000:
        reg += (0 if self.link else 0) * 0x80
        reg &= ~0x20000000
    return reg

  def _reg_rd32(self, reg: int) -> int:
    reg = self._nvreg(reg)
    if not self.init_exec():
      return 0
    try:
      return self.dev.read32(reg)
    except Exception as e:
      raise RuntimeError(f"NVINIT read32({reg:#010x}) failed: {e}") from e

  def _reg_wr32(self, reg: int, val: int) -> None:
    reg = self._nvreg(reg)
    if not self.init_exec():
      return
    try:
      self.dev.write32(reg, val)
      if self.debug:
        print(f"  [nvbios] WR {reg:#010x} = {val:#010x}")
    except Exception as e:
      raise RuntimeError(
          f"NVINIT write32({reg:#010x}, {val:#010x}) failed: {e}") from e

  def _reg_mask(self, reg: int, mask: int, val: int) -> int:
    """init_mask: (r & ~mask) | val."""
    reg = self._nvreg(reg)
    if not self.init_exec():
      return 0
    try:
      old = self.dev.read32(reg) & 0xffffffff
      mask &= 0xffffffff
      val &= 0xffffffff
      # Nouveau deliberately does not clip val to mask.  INIT_OR_REG passes
      # mask=0 and relies on this to OR its payload into the existing value.
      new = ((old & (~mask & 0xffffffff)) | val) & 0xffffffff
      self.dev.write32(reg, new)
      if self.debug:
        print(f"  [nvbios] MASK {reg:#010x} & ~{mask:#010x} | {val:#010x} = {new:#010x}")
      return old
    except Exception as e:
      raise RuntimeError(f"NVINIT mask({reg:#010x}) failed: {e}") from e

  def _vgai_rd(self, port: int, index: int) -> int:
    """Read one Nouveau VGA indexed register through the BAR0 VGA window."""
    if port not in (0x03c4, 0x03ce, 0x03d4) or not self.init_exec():
      return 0
    self._port_wr(port, index)
    value = self._port_rd(port + 1)
    if os.environ.get("KEPLER_VBIOS_IO_TRACE", "0") == "1":
      print(f"[nvbios] VGA RD port={port:#06x} index={index:#04x} "
            f"value={value:#04x}", flush=True)
    return value

  def _vgai_wr(self, port: int, index: int, val: int) -> None:
    if port not in (0x03c4, 0x03ce, 0x03d4) or not self.init_exec():
      return
    self._port_wr(port, index)
    self._port_wr(port + 1, val)
    if os.environ.get("KEPLER_VBIOS_IO_TRACE", "0") == "1":
      print(f"[nvbios] VGA WR port={port:#06x} index={index:#04x} "
            f"value={val & 0xff:#04x}", flush=True)

  def unlock_vga_crtc(self) -> None:
    """Match ``nvkm_devinit_preinit()`` for NV50-and-newer boards.

    Nouveau unlocks extended CRTC registers before it evaluates any NVINIT
    condition.  GK104 uses CR3f=0x57 (pre-NV50 chips use CR1f instead).
    """
    self._vgai_wr(0x03d4, 0x3f, 0x57)

  def option_rom_vga_enable_prefix(self) -> None:
    """Replay the reachable legacy-ROM VGA-enable prefix before NVINIT.

    The Palit GK104 x86 image enters this VGA-only sequence through
    ``0x0050 -> 0x2caa -> 0x657e -> 0x67ae``.  It enables VGA decode through
    ports 0x3c3/0x3c2, performs the two readbacks in that exact order, and
    then unlocks the extended CRTC registers.  Later reachable ROM code also
    uses the device's native PCI I/O BAR; that separate prefix is deliberately
    not represented here.  The real transport exposes these VGA ports through
    NVIDIA's BAR0 alias at 0x601000 + port.
    """
    self._port_wr(0x03c3, 0x01)
    self._port_rd(0x03c3)
    self._port_wr(0x03c2, 0x01)
    self._port_rd(0x03c3)
    self.unlock_vga_crtc()

  def _port_rd(self, port: int) -> int:
    if not self.init_exec():
      return 0
    reg = 0x601000 + (port & 0xffff)
    try:
      if hasattr(self.dev, "read8"):
        return self.dev.read8(reg) & 0xff
      # Offline fakes generally expose only dword MMIO.  Real transports use
      # read8 above because VGA index/data ports have access side effects.
      word = self.dev.read32(reg & ~3)
      return (word >> ((reg & 3) * 8)) & 0xff
    except Exception as e:
      raise RuntimeError(f"NVINIT rdport({port:#06x}) failed: {e}") from e

  def _port_wr(self, port: int, val: int) -> None:
    if not self.init_exec():
      return
    reg = 0x601000 + (port & 0xffff)
    val &= 0xff
    try:
      if hasattr(self.dev, "write8"):
        self.dev.write8(reg, val)
        return
      # Test-only compatibility for dword-backed fake MMIO.
      aligned = reg & ~3
      shift = (reg & 3) * 8
      old = self.dev.read32(aligned)
      self.dev.write32(aligned, (old & ~(0xff << shift)) | (val << shift))
    except Exception as e:
      raise RuntimeError(
          f"NVINIT wrport({port:#06x}, {val:#04x}) failed: {e}") from e

  # ------------------------------------------------------------------------
  # GK104 DCB I2C (Nouveau gf119_i2c_bus + internal bit transfer)
  # ------------------------------------------------------------------------
  def _i2c_bus_reg(self, index: int) -> int:
    """Resolve an NVINIT I2C index to the GK104 GF119-style bus register."""
    dcb = self.rd16(0x36)
    if not dcb or dcb + 6 > len(self.image):
      raise NvbiosI2CError("NVINIT I2C: DCB table missing", errno=19)
    table = self.rd16(dcb + 4)
    if not table or table + 5 > len(self.image):
      raise NvbiosI2CError("NVINIT I2C: DCB I2C table missing", errno=19)
    ver = self.rd08(table)
    hdr = self.rd08(table + 1)
    count = self.rd08(table + 2)
    length = self.rd08(table + 3)
    if ver < 0x30 or ver > 0x41 or not hdr or not length:
      raise NvbiosI2CError(
          f"NVINIT I2C: unsupported DCB table version {ver:#x}", errno=19)
    if index in (0xff, 0x80):
      index = self.rd08(table + 4) & 0x0f
    elif index == 0x81:
      index = (self.rd08(table + 4) >> 4) & 0x0f
    if index >= count:
      raise NvbiosI2CError(
          f"NVINIT I2C: bus index {index:#x} out of range", errno=19)
    entry = table + hdr + index * length
    if entry + length > len(self.image):
      raise NvbiosI2CError("NVINIT I2C: truncated DCB entry", errno=19)
    if ver >= 0x41:
      info = self.rd16(entry)
      drive = info & 0x1f
      if drive == 0x1f:
        raise NvbiosI2CError(
            f"NVINIT I2C: CCB {index:#x} has no I2C drive", errno=19)
    else:
      type_ = self.rd08(entry + 3)
      if type_ != 0x05:  # DCB_I2C_NVIO_BIT
        raise NvbiosI2CError(
            f"NVINIT I2C: CCB {index:#x} type {type_:#x} is not NVIO_BIT",
            errno=19)
      drive = self.rd08(entry) & 0x0f
    return 0x00d014 + drive * 0x20

  @staticmethod
  def _i2c_delay(usec: int) -> None:
    time.sleep(max(0, int(usec)) / 1_000_000.0)

  def _i2c_mask(self, bus: int, mask: int, data: int) -> None:
    old = self.dev.read32(bus) & 0xffffffff
    self.dev.write32(bus, (old & ~mask) | (data & mask))

  def _i2c_scl(self, bus: int, state: bool) -> None:
    self._i2c_mask(bus, 0x00000001, 0x00000001 if state else 0)

  def _i2c_sda(self, bus: int, state: bool) -> None:
    self._i2c_mask(bus, 0x00000002, 0x00000002 if state else 0)

  def _i2c_sense_scl(self, bus: int) -> bool:
    return bool(self.dev.read32(bus) & 0x00000010)

  def _i2c_sense_sda(self, bus: int) -> bool:
    return bool(self.dev.read32(bus) & 0x00000020)

  def _i2c_raise_scl(self, bus: int) -> None:
    self._i2c_scl(bus, True)
    for _ in range(2200):  # Nouveau T_TIMEOUT/T_RISEFALL = 2.2ms/1us
      self._i2c_delay(1)
      if self._i2c_sense_scl(bus):
        return
    raise NvbiosI2CError(f"NVINIT I2C: SCL stuck low at {bus:#x}")

  def _i2c_start(self, bus: int) -> None:
    if not self._i2c_sense_scl(bus) or not self._i2c_sense_sda(bus):
      self._i2c_scl(bus, False)
      self._i2c_sda(bus, True)
      self._i2c_raise_scl(bus)
    self._i2c_sda(bus, False)
    self._i2c_delay(5)
    self._i2c_scl(bus, False)
    self._i2c_delay(5)

  def _i2c_stop(self, bus: int) -> None:
    self._i2c_scl(bus, False)
    self._i2c_sda(bus, False)
    self._i2c_delay(1)
    self._i2c_scl(bus, True)
    self._i2c_delay(5)
    self._i2c_sda(bus, True)
    self._i2c_delay(5)

  def _i2c_bit_write(self, bus: int, bit: int) -> None:
    self._i2c_sda(bus, bool(bit))
    self._i2c_delay(1)
    self._i2c_raise_scl(bus)
    self._i2c_delay(5)
    self._i2c_scl(bus, False)
    self._i2c_delay(5)

  def _i2c_bit_read(self, bus: int) -> int:
    self._i2c_sda(bus, True)
    self._i2c_delay(1)
    self._i2c_raise_scl(bus)
    self._i2c_delay(5)
    bit = int(self._i2c_sense_sda(bus))
    self._i2c_scl(bus, False)
    self._i2c_delay(5)
    return bit

  def _i2c_put_byte(self, bus: int, value: int) -> None:
    for bit in range(7, -1, -1):
      self._i2c_bit_write(bus, (value >> bit) & 1)
    if self._i2c_bit_read(bus):
      raise NvbiosI2CError(
          f"NVINIT I2C: NACK writing {value & 0xff:#04x}")

  def _i2c_get_byte(self, bus: int, last: bool) -> int:
    value = 0
    for bit in range(7, -1, -1):
      value |= self._i2c_bit_read(bus) << bit
    self._i2c_bit_write(bus, 1 if last else 0)
    return value

  def _i2c_xfer(self, index: int, addr: int,
                messages: list[tuple[bool, bytes | int]]) -> list[bytes]:
    bus = self._i2c_bus_reg(index)
    if bus not in self._i2c_inited:
      # gf119_i2c_bus_init(): release both lines and enable the port.
      self.dev.write32(bus, 0x00000007)
      self._i2c_inited.add(bus)
    reads: list[bytes] = []
    try:
      for is_read, payload in messages:
        self._i2c_start(bus)
        self._i2c_put_byte(bus, ((addr & 0x7f) << 1) | int(is_read))
        if is_read:
          count = int(payload)
          reads.append(bytes(self._i2c_get_byte(bus, i + 1 == count)
                             for i in range(count)))
        else:
          for value in bytes(payload):
            self._i2c_put_byte(bus, value)
      return reads
    finally:
      self._i2c_stop(bus)

  def _i2c_read_reg(self, index: int, addr: int, reg: int) -> int:
    reads = self._i2c_xfer(index, addr,
                           [(False, bytes((reg & 0xff,))), (True, 1)])
    value = reads[0][0]
    if os.environ.get("KEPLER_VBIOS_I2C_TRACE", "0") == "1":
      print(f"[nvbios] I2C RD index={index:#x} addr={addr:#x} "
            f"reg={reg:#x} value={value:#x}", flush=True)
    return value

  def _i2c_write_reg(self, index: int, addr: int, reg: int, value: int) -> None:
    self._i2c_xfer(index, addr,
                   [(False, bytes((reg & 0xff, value & 0xff)))])
    if os.environ.get("KEPLER_VBIOS_I2C_TRACE", "0") == "1":
      print(f"[nvbios] I2C WR index={index:#x} addr={addr:#x} "
            f"reg={reg:#x} value={value & 0xff:#x}", flush=True)

  # --------------------------------------------------------------------------
  # Execute flag helpers
  # --------------------------------------------------------------------------
  def init_exec(self) -> bool:
    return self.execute == 1 or (self.execute & 5) == 5

  def init_exec_set(self, exec_: bool) -> None:
    if exec_:
      self.execute &= 0xfd
    else:
      self.execute |= 0x02

  def init_exec_inv(self) -> None:
    self.execute ^= 0x02

  def init_exec_force(self, exec_: bool) -> None:
    if exec_:
      self.execute |= 0x04
    else:
      self.execute &= 0xfb

  # --------------------------------------------------------------------------
  # Condition / strap / ramcfg helpers
  # --------------------------------------------------------------------------
  def _ramcfg_count(self) -> int:
    m = self.tables.get("m", 0)
    mver = self.tables.get("m_ver", 0)
    if m and m + 3 <= len(self.image):
      if mver == 1 and self.rd08(m + 5) >= 5:
        return self.rd08(m + 2)
      if mver == 2 and self.rd08(m + 0) >= 3:
        return self.rd08(m + 0)
    return 0

  def _ramcfg_index(self) -> int:
    if self._ramcfg_value is not None:
      return self._ramcfg_value
    # Optional override for cold eGPU bring-up when 0x101000 is unreadable
    # (returns 0) or for offline golden-trace replay.  The Palit GTX 770 ROM's
    # Nouveau baseline uses hardware strap 6 (0x101000=0x8040509a).
    override = os.environ.get("KEPLER_RAMCFG_STRAP")
    if override is not None and override != "":
      strap = int(override, 0) & 0xf
    else:
      strap_reg = self._reg_rd32(0x101000)
      strap = (strap_reg & 0x0000003c) >> 2
      self._ramcfg_strap_reg = strap_reg
    if not self.tables.get("m", 0):
      self._ramcfg_value = strap
      return strap
    m = self.tables["m"]
    mver = self.tables.get("m_ver", 0)
    # M0203E lookup if table is present
    m0203t = self.tables.get("m0203t", 0)
    if m0203t and m0203t + 6 <= len(self.image) and mver == 2 and self.rd08(m + 0) >= 7:
      tver = self.rd08(m0203t + 0)
      if tver == 0x10:
        thdr = self.rd08(m0203t + 1)
        tlen = self.rd08(m0203t + 2)
        tcnt = self.rd08(m0203t + 3)
        for i in range(tcnt):
          e = m0203t + thdr + i * tlen
          if e + 2 > len(self.image):
            break
          entry_strap = (self.rd08(e + 0) & 0xf0) >> 4
          entry_group = (self.rd08(e + 1) & 0x0f)
          if entry_strap == strap:
            self._ramcfg_value = entry_group
            return entry_group
    # fallback xlat table
    xlat = self.tables.get("m_xlat", 0)
    if xlat and xlat + strap < len(self.image):
      strap = self.rd08(xlat + strap)
    self._ramcfg_value = strap
    return strap

  def rammap_scripts(self) -> list[int]:
    """Return the GK104 BIT-P RAMMAP init-script pointers.

    Nouveau's ``gk104_ram_init()`` executes these scripts before it touches
    framebuffer allocations.  They are distinct from the BIT-I devinit
    scripts and contain the mode-specific GDDR5 controller setup.
    """
    poff, _ = self._bit_entry(b"P")
    if not poff or poff + 8 > len(self.image):
      return []
    rammap = self.rd32(poff + 4)
    if not rammap or rammap + 0x18 > len(self.image):
      return []
    count = self.rd08(rammap + 0x14)
    table = self.rd32(rammap + 0x10)
    if not table or table + count * 4 > len(self.image):
      return []
    return [self.rd32(table + i * 4) for i in range(count)
            if self.rd32(table + i * 4)]

  def rammap_config(self, freq_mhz: int = 1000) -> dict:
    """Decode the v0x11 RAMMAP/RAMCFG entry selected for ``freq_mhz``.

    ``gk104_ram_init()`` only consumes the mode scripts and training tables;
    the clock transition path consumes the RAMCFG and timing fields.  Keeping
    this decoder here makes that second phase use the same ROM/strap selection
    as the first phase instead of duplicating offsets in the launcher.
    """
    poff, _ = self._bit_entry(b"P")
    if not poff or poff + 12 > len(self.image):
      raise ValueError("GK104 BIT-P table not found")
    rammap = self.rd32(poff + 4)
    if not rammap or rammap + 6 > len(self.image):
      raise ValueError("GK104 RAMMAP table not found")
    ver = self.rd08(rammap)
    hdr = self.rd08(rammap + 1)
    length = self.rd08(rammap + 2)
    sub_size = self.rd08(rammap + 3)
    sub_count = self.rd08(rammap + 4)
    count = self.rd08(rammap + 5)
    if ver != 0x11 or not length or not sub_size:
      raise ValueError(f"unsupported GK104 RAMMAP version {ver:#x}")

    selected = None
    entries = []
    for i in range(count):
      entry = rammap + hdr + i * (length + sub_count * sub_size)
      if entry + length + sub_count * sub_size > len(self.image):
        break
      lo, hi = self.rd16(entry), self.rd16(entry + 2)
      item = (lo, hi, i, entry)
      entries.append(item)
      if lo <= int(freq_mhz) <= hi:
        selected = item
        break
    if selected is None:
      if not entries:
        raise ValueError("GK104 RAMMAP has no entries")
      selected = min(entries,
                     key=lambda item: min(abs(freq_mhz - item[0]),
                                          abs(freq_mhz - item[1])))
    lo, hi, entry_index, entry = selected
    ramcfg_index = self._ramcfg_index()
    if ramcfg_index >= sub_count:
      raise ValueError(f"RAMCFG strap index {ramcfg_index} >= {sub_count}")
    cfg = entry + length + ramcfg_index * sub_size
    if cfg + sub_size > len(self.image):
      raise ValueError("GK104 RAMCFG entry is truncated")

    # nvbios_rammapSp(), v0x11.
    b = self.rd08
    c = {
      "rammap_index": entry_index, "rammap_min": lo, "rammap_max": hi,
      "rammap_11_08_01": self.rd08(entry + 8) & 1,
      "rammap_11_08_0c": (self.rd08(entry + 8) >> 2) & 3,
      "rammap_11_08_10": (self.rd08(entry + 8) >> 4) & 1,
      "rammap_11_09_01ff": self.rd32(entry + 9) & 0x1ff,
      "rammap_11_0a_03fe": (self.rd32(entry + 9) >> 9) & 0x1ff,
      "rammap_11_0a_0400": (self.rd32(entry + 9) >> 18) & 1,
      "rammap_11_0a_0800": (self.rd32(entry + 9) >> 19) & 1,
      "rammap_11_0b_01f0": (self.rd32(entry + 9) >> 20) & 0x1f,
      "rammap_11_0b_0200": (self.rd32(entry + 9) >> 25) & 1,
      "rammap_11_0b_0400": (self.rd32(entry + 9) >> 26) & 1,
      "rammap_11_0b_0800": (self.rd32(entry + 9) >> 27) & 1,
      "rammap_11_0d": self.rd08(entry + 0x0d),
      "rammap_11_0e": self.rd08(entry + 0x0e),
      "rammap_11_0f": self.rd08(entry + 0x0f),
      "rammap_11_11_0c": (self.rd08(entry + 0x11) >> 2) & 3,
      "ramcfg_index": ramcfg_index, "ramcfg_timing": b(cfg),
      "ramcfg_11_01_01": b(cfg + 1) & 1,
      "ramcfg_11_01_02": (b(cfg + 1) >> 1) & 1,
      "ramcfg_11_01_04": (b(cfg + 1) >> 2) & 1,
      "ramcfg_11_01_08": (b(cfg + 1) >> 3) & 1,
      "ramcfg_11_01_10": (b(cfg + 1) >> 4) & 1,
      "ramcfg_DLLoff": (b(cfg + 1) >> 5) & 1,
      "ramcfg_11_01_40": (b(cfg + 1) >> 6) & 1,
      "ramcfg_11_01_80": (b(cfg + 1) >> 7) & 1,
      "ramcfg_11_02_03": b(cfg + 2) & 3,
      "ramcfg_11_02_04": (b(cfg + 2) >> 2) & 1,
      "ramcfg_11_02_08": (b(cfg + 2) >> 3) & 1,
      "ramcfg_11_02_10": (b(cfg + 2) >> 4) & 1,
      "ramcfg_11_02_40": (b(cfg + 2) >> 6) & 1,
      "ramcfg_11_02_80": (b(cfg + 2) >> 7) & 1,
      "ramcfg_11_03_0f": b(cfg + 3) & 0x0f,
      "ramcfg_11_03_30": (b(cfg + 3) >> 4) & 3,
      "ramcfg_11_03_c0": (b(cfg + 3) >> 6) & 3,
      "ramcfg_11_03_f0": (b(cfg + 3) >> 4) & 0x0f,
      "ramcfg_11_04": b(cfg + 4),
      "ramcfg_11_06": b(cfg + 6),
      "ramcfg_11_07_02": (b(cfg + 7) >> 1) & 1,
      "ramcfg_11_07_04": (b(cfg + 7) >> 2) & 1,
      "ramcfg_11_07_08": (b(cfg + 7) >> 3) & 1,
      "ramcfg_11_07_10": (b(cfg + 7) >> 4) & 1,
      "ramcfg_11_07_40": (b(cfg + 7) >> 6) & 1,
      "ramcfg_11_07_80": (b(cfg + 7) >> 7) & 1,
      "ramcfg_11_08_01": b(cfg + 8) & 1,
      "ramcfg_11_08_02": (b(cfg + 8) >> 1) & 1,
      "ramcfg_11_08_04": (b(cfg + 8) >> 2) & 1,
      "ramcfg_11_08_08": (b(cfg + 8) >> 3) & 1,
      "ramcfg_11_08_10": (b(cfg + 8) >> 4) & 1,
      "ramcfg_11_08_20": (b(cfg + 8) >> 5) & 1,
      "ramcfg_11_09": b(cfg + 9),
    }
    # nvbios_timingEp(), v0x20.  BIT-P+0x08 is the timing table pointer.
    timing_table = self.rd32(poff + 8)
    if not timing_table or self.rd08(timing_table) != 0x20:
      raise ValueError("GK104 v0x20 timing table not found")
    timing_hdr = self.rd08(timing_table + 1)
    timing_len = self.rd08(timing_table + 2)
    timing_count = self.rd08(timing_table + 5)
    timing_index = c["ramcfg_timing"]
    if timing_index >= timing_count:
      raise ValueError(f"timing index {timing_index} >= {timing_count}")
    timing = timing_table + timing_hdr + timing_index * timing_len
    if timing + timing_len > len(self.image) or timing_len < 0x33:
      raise ValueError("GK104 timing entry is truncated")
    c["timing_index"] = timing_index
    c["timing"] = [self.rd32(timing + i * 4) for i in range(11)]
    c["timing_20_2e_03"] = self.rd08(timing + 0x2e) & 3
    c["timing_20_2e_30"] = (self.rd08(timing + 0x2e) >> 4) & 3
    c["timing_20_2e_c0"] = (self.rd08(timing + 0x2e) >> 6) & 3
    c["timing_20_2f_03"] = self.rd08(timing + 0x2f) & 3
    t2c = self.rd16(timing + 0x2c)
    c["timing_20_2c_003f"] = t2c & 0x3f
    c["timing_20_2c_1fc0"] = (t2c >> 6) & 0x7f
    t30 = self.rd08(timing + 0x30)
    c["timing_20_30_07"] = t30 & 7
    c["timing_20_30_f8"] = (t30 >> 3) & 0x1f
    t31 = self.rd16(timing + 0x31)
    c["timing_20_31_0007"] = t31 & 7
    c["timing_20_31_0078"] = (t31 >> 3) & 0xf
    c["timing_20_31_0780"] = (t31 >> 7) & 0xf
    c["timing_20_31_0800"] = (t31 >> 11) & 1
    c["timing_20_31_7000"] = (t31 >> 12) & 7
    c["timing_20_31_8000"] = (t31 >> 15) & 1
    return c

  def rammap_diffs(self) -> dict:
    """Return Nouveau-style ``ram->diff`` flags for the ROM RAMMAP entries.

    The GK104 driver compares each successive RAMMAP/RAMCFG entry while
    constructing its transition program.  These flags describe ROM variation;
    they are not equivalent to testing whether the selected value is nonzero.
    Keeping that distinction matters for fields which must explicitly remain
    zero in the selected configuration.
    """
    poff, _ = self._bit_entry(b"P")
    if not poff or poff + 12 > len(self.image):
      raise ValueError("GK104 BIT-P table not found")
    rammap = self.rd32(poff + 4)
    if not rammap or rammap + 6 > len(self.image):
      raise ValueError("GK104 RAMMAP table not found")
    hdr = self.rd08(rammap + 1)
    length = self.rd08(rammap + 2)
    sub_size = self.rd08(rammap + 3)
    sub_count = self.rd08(rammap + 4)
    count = self.rd08(rammap + 5)
    stride = length + sub_count * sub_size
    if not hdr or not length or not stride or not count:
      raise ValueError("GK104 RAMMAP has no entries")
    configs = []
    for i in range(count):
      entry = rammap + hdr + i * stride
      if entry + length > len(self.image):
        break
      lo, hi = self.rd16(entry), self.rd16(entry + 2)
      # Select this exact entry through the public decoder.  Midpoints avoid
      # ambiguity at adjacent range boundaries.
      freq = lo if hi <= lo else (lo + hi) // 2
      try:
        cfg = self.rammap_config(freq)
      except ValueError:
        continue
      if cfg["rammap_index"] == i:
        configs.append(cfg)
    if len(configs) < 2:
      return {k: False for k in (
        "rammap_11_0a_03fe", "rammap_11_09_01ff",
        "rammap_11_0a_0400", "rammap_11_0a_0800", "rammap_11_0b_01f0",
        "rammap_11_0b_0200", "rammap_11_0d", "rammap_11_0f",
        "rammap_11_0e", "rammap_11_0b_0800", "rammap_11_0b_0400",
        "ramcfg_11_01_01", "ramcfg_11_01_02", "ramcfg_11_01_10",
        "ramcfg_11_02_03", "ramcfg_11_08_20", "timing_20_30_07")}
    fields = (
      "rammap_11_0a_03fe", "rammap_11_09_01ff",
      "rammap_11_0a_0400", "rammap_11_0a_0800", "rammap_11_0b_01f0",
      "rammap_11_0b_0200", "rammap_11_0d", "rammap_11_0f",
      "rammap_11_0e", "rammap_11_0b_0800", "rammap_11_0b_0400",
      "ramcfg_11_01_01", "ramcfg_11_01_02", "ramcfg_11_01_10",
      "ramcfg_11_02_03", "ramcfg_11_08_20", "timing_20_30_07")
    first = configs[0]
    return {field: any(cfg[field] != first[field] for cfg in configs[1:])
            for field in fields}

  def refpll_limits(self) -> dict | None:
    """Decode the ROM's BIT-C memory reference-PLL limits.

    GK104's ``gt215_pll_calc()`` does not use a generic VCO range.  It uses
    the per-board limits from the type-0x0c PLL entry (including the allowed
    P/M/N ranges and reference clock).  Returning those fields keeps the
    userspace RAM transition on the same coefficient path as Nouveau.
    """
    coff, cver = self._bit_entry(b"C")
    if not coff:
      return None
    # pll_limits_table(): BIT-C v1 stores the limits-table pointer at +8;
    # v2 stores a 32-bit pointer at +0.
    if cver == 1 and coff + 10 <= len(self.image):
      table = self.rd16(coff + 8)
    elif cver == 2 and coff + 4 <= len(self.image):
      table = self.rd32(coff)
    else:
      return None
    if not table or table + 4 > len(self.image):
      return None
    ver, hdr, length, count = (self.rd08(table + i) for i in range(4))
    if ver < 0x30 or not hdr or not length:
      return None
    for i in range(count):
      entry = table + hdr + i * length
      if entry + length > len(self.image) or self.rd08(entry) != 0x0c:
        continue
      if ver >= 0x50:
        # GK104 uses v0x40, but leave newer tables explicitly unsupported
        # rather than guessing the pointer layout.
        return None
      if entry + 11 > len(self.image):
        return None
      limits = self.rd16(entry + 1)
      if not limits or limits + 14 > len(self.image):
        return None
      if ver == 0x40:
        # nvbios_pll_parse(), case 0x40.
        return {
          "refclk": self.rd16(entry + 9) * 1000,
          "min_freq": self.rd16(limits + 0) * 1000,
          "max_freq": self.rd16(limits + 2) * 1000,
          "min_inputfreq": self.rd16(limits + 4) * 1000,
          "max_inputfreq": self.rd16(limits + 6) * 1000,
          "min_m": self.rd08(limits + 8),
          "max_m": self.rd08(limits + 9),
          "min_n": self.rd08(limits + 10),
          "max_n": self.rd08(limits + 11),
          "min_p": self.rd08(limits + 12),
          "max_p": self.rd08(limits + 13),
        }
      return None
    return None

  def dcb_gpios(self) -> list[dict]:
    """Return all DCB GPIO descriptors needed by GK104 ``gpio_reset``."""
    result = []
    dcb = self.rd16(0x36)
    if not dcb or dcb + 0x0c > len(self.image):
      return result
    dver, dhdr = self.rd08(dcb), self.rd08(dcb + 1)
    if dver < 0x30 or dhdr < 0x0c:
      return result
    table = self.rd16(dcb + 0x0a)
    if not table or table + 4 > len(self.image):
      return result
    ver = self.rd08(table)
    hdr = self.rd08(table + 1)
    count = self.rd08(table + 2)
    length = self.rd08(table + 3)
    if not hdr or not length or ver > 0x41:
      return result
    for i in range(count):
      entry = table + hdr + i * length
      if entry + length > len(self.image):
        break
      if ver < 0x40:
        info = self.rd16(entry)
        line = info & 0x1f
        func = (info >> 5) & 0x3f
        log0, log1 = (info >> 11) & 3, (info >> 13) & 3
        defs = int(bool(info & 0x8000))
        unk0 = unk1 = 0
      elif ver < 0x41:
        info = self.rd32(entry)
        line = info & 0x1f
        func = (info >> 8) & 0xff
        log0, log1 = (info >> 27) & 3, (info >> 29) & 3
        defs = int(bool(info & 0x80000000))
        unk0 = unk1 = 0
      else:
        info = self.rd32(entry)
        info1 = self.rd08(entry + 4)
        line = info & 0x3f
        func = (info >> 8) & 0xff
        log0, log1 = (info1 >> 4) & 3, (info1 >> 6) & 3
        defs = int(bool(info & 0x80))
        unk0 = (info >> 16) & 0xff
        unk1 = (info >> 24) & 0x1f
      result.append({
          "line": line, "func": func, "log0": log0, "log1": log1,
          "defs": defs, "unk0": unk0, "unk1": unk1,
          "reg": 0x00d610 + line * 4,
      })
    return result

  def dcb_gpio(self, function: int) -> dict | None:
    """Return one DCB GPIO function descriptor for a GK104 ROM.

    The GDDR5 transition uses DCB tags 0x18 (memory-voltage select) and 0x2e
    (voltage-control).  Nouveau maps a matching line to ``0xd610 + line*4``
    and derives the two logical levels from the entry's log bits.
    """
    for gpio in self.dcb_gpios():
      if gpio["func"] == int(function):
        return gpio
    return None

  def _gpio_reset(self, excluded: set[int] | None = None) -> tuple[int, list[int]]:
    """Port ``gf119_gpio_reset`` for GK104's DCB v0x41 table."""
    excluded = set() if excluded is None else excluded
    count = 0
    changed = []
    for gpio in self.dcb_gpios():
      if gpio["func"] == 0xff or gpio["func"] in excluded:
        continue
      level = gpio["log1"] if gpio["defs"] else gpio["log0"]
      want = (level ^ 2) << 12
      old = self._reg_mask(gpio["reg"], 0x00003000, want)
      if (old & 0x00003000) != want:
        changed.append(gpio["func"])
      self._reg_mask(0x00d604, 0x00000001, 0x00000001)
      self._reg_mask(gpio["reg"], 0x000000ff, gpio["unk0"])
      if gpio["unk1"]:
        self._reg_mask(0x00d740 + (gpio["unk1"] - 1) * 4,
                       0x000000ff, gpio["line"])
      count += 1
    return count, changed

  def _m0205_training_entry(self, index: int):
    """Decode one M0205E entry and select its current RAMCFG byte."""
    m = self.tables.get("m", 0)
    if not m or m + 9 > len(self.image):
      return None
    base = self.rd32(m + 5)
    if not base or base + 6 > len(self.image) or self.rd08(base) != 0x10:
      return None
    hdr, length = self.rd08(base + 1), self.rd08(base + 2)
    ssz, snr, count = (self.rd08(base + i) for i in (3, 4, 5))
    if index >= count:
      return None
    entry = base + hdr + index * (length + snr * ssz)
    data = entry + length + self._ramcfg_index() * ssz
    if data >= len(self.image):
      return None
    return self.rd08(entry) & 0x0f, self.rd08(data)

  def _m0209_values(self, index: int) -> list[int] | None:
    """Decode one packed M0209 training data entry (Nouveau M0209.c)."""
    m = self.tables.get("m", 0)
    if not m or m + 13 > len(self.image):
      return None
    base = self.rd32(m + 9)
    if not base or base + 5 > len(self.image) or self.rd08(base) != 0x10:
      return None
    hdr, length, sample_size, count = (self.rd08(base + i) for i in (1, 2, 3, 4))
    if index >= count:
      return None
    entry = base + hdr + index * (length + sample_size)
    if entry + length + sample_size > len(self.image):
      return None
    bits = self.rd08(entry) & 0x3f
    modulo = self.rd08(entry + 1)
    mode = self.rd08(entry + 2) & 0x07
    remap_index = self.rd08(entry + 3)
    if not bits or not modulo:
      return None
    data = entry + length
    mask = (1 << bits) - 1
    values = []
    for i in range(0x100):
      bit = (i % modulo) * bits
      raw = self.rd32(data + bit // 8) >> (bit & 7)
      values.append(raw & mask)
    if mode == 2:
      remap = self._m0209_values(remap_index)
      if remap is None:
        return None
      values = [remap[v] for v in values]
    elif mode != 1:
      return None
    return values

  def ram_training(self) -> dict[int, list[int]]:
    """Decode the GDDR5 training arrays consumed by gk104_ram_train_init()."""
    result = {}
    for i in range(0x100):
      item = self._m0205_training_entry(i)
      if item is None:
        continue
      typ, data_index = item
      values = self._m0209_values(data_index)
      if values is not None and typ in (0x00, 0x01, 0x04, 0x06, 0x07, 0x08, 0x09):
        result[typ] = values
    required = (0x00, 0x01, 0x04, 0x06, 0x07, 0x08, 0x09)
    missing = [typ for typ in required if typ not in result]
    if missing:
      raise ValueError(f"GK104 VBIOS missing GDDR5 training types: {missing}")
    return result

  def _condition_table(self) -> int:
    return self.tables.get("condition", 0)

  def _io_condition_table(self) -> int:
    return self.tables.get("io_condition", 0)

  def _io_flag_condition_table(self) -> int:
    return self.tables.get("io_flag", 0)

  def _condition_met(self, cond: int) -> bool:
    table = self._condition_table()
    if table and table + cond * 12 + 12 <= len(self.image):
      reg = self.rd32(table + cond * 12 + 0)
      mask = self.rd32(table + cond * 12 + 4)
      val = self.rd32(table + cond * 12 + 8)
      return (self._reg_rd32(reg) & mask) == val
    return False

  def _io_condition_met(self, cond: int) -> bool:
    table = self._io_condition_table()
    if table and table + cond * 5 + 5 <= len(self.image):
      port = self.rd16(table + cond * 5 + 0)
      index = self.rd08(table + cond * 5 + 2)
      mask = self.rd08(table + cond * 5 + 3)
      val = self.rd08(table + cond * 5 + 4)
      return (self._vgai_rd(port, index) & mask) == val
    return False

  def _io_flag_condition_met(self, cond: int) -> bool:
    table = self._io_flag_condition_table()
    if table and table + cond * 9 + 9 <= len(self.image):
      port = self.rd16(table + cond * 9 + 0)
      index = self.rd08(table + cond * 9 + 2)
      mask = self.rd08(table + cond * 9 + 3)
      shift = self.rd08(table + cond * 9 + 4)
      data = self.rd16(table + cond * 9 + 5)
      dmask = self.rd08(table + cond * 9 + 7)
      val = self.rd08(table + cond * 9 + 8)
      ioval = (self._vgai_rd(port, index) & mask) >> shift
      return (self.rd08(data + ioval) & dmask) == val
    return False

  @staticmethod
  def _shift(data: int, shift: int) -> int:
    if shift < 0x80:
      return data >> shift
    return data << (0x100 - shift)

  # --------------------------------------------------------------------------
  # Core execution loop
  # --------------------------------------------------------------------------
  def run_script(self, start: int, stop_before: int | None = None) -> None:
    if self.nested == 0:
      # nvbios_init() constructs a fresh execution state for every top-level
      # script.  Only nested SUB/CALL execution inherits conditions/selectors.
      self.execute = 1
      self.repeat = 0
      self.repend = 0
      self.ramcfg = 0
      self.head = -1
      self.or_ = -1
      self.link = 0
      self.outp = None
      # Nouveau's nvbios_init() stack object is fresh for each top-level
      # invocation.  The strap cache belongs to that object too; only nested
      # SUB/CALL execution may inherit it.
      self._ramcfg_value = None
    self.offset = start
    self.nested += 1
    while self.offset:
      # Night41x proved activator is inside 0x8fe8 after 0x87e5.  Nested
      # bisection stops before a top-level ROM offset without mid-POST
      # 0x1700 samples.  Nested SUB/CALL may still run past stop_before.
      if (stop_before is not None and self.nested == 1 and
          self.offset >= int(stop_before)):
        break
      op = self.rd08(self.offset)
      handler = self._handlers.get(op)
      if handler is None:
        raise RuntimeError(f"NVINIT unknown opcode 0x{op:02x} at offset 0x{self.offset:04x}")
      handler(self)
    self.nested -= 1

  # --------------------------------------------------------------------------
  # Opcode handlers
  # --------------------------------------------------------------------------
  def _op_reset_begun(self):
    self.offset += 1
  def _op_reset_end(self):
    self.offset += 1
  def _op_done(self):
    self.offset = 0
  def _op_resume(self):
    self.offset += 1
    self.init_exec_set(True)
  def _op_not(self):
    self.offset += 1
    self.init_exec_inv()

  def _op_zm_reg(self):
    addr = self.rd32(self.offset + 1)
    data = self.rd32(self.offset + 5)
    # Nouveau init_zm_reg(): PMC_ENABLE bit 0 is the master-enable invariant.
    # The Palit cold script's first write is 0x2020 at ROM 0x87e6, so omitting
    # this source-level special case transiently disables the master domain.
    if addr == 0x000200:
      data |= 0x00000001
    self._reg_wr32(addr, data)
    self.offset += 9

  def _op_nv_reg(self):
    addr = self.rd32(self.offset + 1)
    mask = self.rd32(self.offset + 5)
    data = self.rd32(self.offset + 9)
    self._reg_mask(addr, ~mask, data)
    self.offset += 13

  def _op_zm_reg16(self):
    addr = self.rd32(self.offset + 1)
    data = self.rd16(self.offset + 5)
    self._reg_wr32(addr, data)
    self.offset += 7

  def _op_zm_reg_group(self):
    addr = self.rd32(self.offset + 1)
    count = self.rd08(self.offset + 5)
    self.offset += 6
    for _ in range(count):
      self._reg_wr32(addr, self.rd32(self.offset))
      self.offset += 4

  def _op_zm_reg_sequence(self):
    addr = self.rd32(self.offset + 1)
    count = self.rd08(self.offset + 5)
    self.offset += 6
    for i in range(count):
      self._reg_wr32(addr + i * 4, self.rd32(self.offset + i * 4))
    self.offset += count * 4

  def _op_zm_mask_add(self):
    addr = self.rd32(self.offset + 1)
    mask = self.rd32(self.offset + 5)
    add = self.rd32(self.offset + 9)
    data = self._reg_rd32(addr)
    data = (data & mask) | ((data + add) & ~mask)
    self._reg_wr32(addr, data)
    self.offset += 13

  def _op_andn_reg(self):
    addr = self.rd32(self.offset + 1)
    mask = self.rd32(self.offset + 5)
    self._reg_mask(addr, mask, 0)
    self.offset += 9

  def _op_or_reg(self):
    addr = self.rd32(self.offset + 1)
    mask = self.rd32(self.offset + 5)
    self._reg_mask(addr, 0, mask)
    self.offset += 9

  def _op_copy_zm_reg(self):
    src = self.rd32(self.offset + 1)
    dst = self.rd32(self.offset + 5)
    self._reg_wr32(dst, self._reg_rd32(src))
    self.offset += 9

  def _op_copy_nv_reg(self):
    sreg = self.rd32(self.offset + 1)
    shift = self.rd08(self.offset + 5)
    smask = self.rd32(self.offset + 6)
    sxor = self.rd32(self.offset + 10)
    dreg = self.rd32(self.offset + 14)
    dmask = self.rd32(self.offset + 18)
    data = self._shift(self._reg_rd32(sreg), shift)
    self._reg_mask(dreg, ~dmask, (data & smask) ^ sxor)
    self.offset += 22

  def _op_zm_reg_indirect(self):
    reg = self.rd32(self.offset + 1)
    addr = self.rd16(self.offset + 5)
    data = self.rd32(addr)
    self._reg_wr32(reg, data)
    self.offset += 7

  def _op_ram_restrict_zm_reg_group(self):
    addr = self.rd32(self.offset + 1)
    incr = self.rd08(self.offset + 5)
    num = self.rd08(self.offset + 6)
    count = self._ramcfg_count()
    index = self._ramcfg_index()
    self.offset += 7
    for _ in range(num):
      for j in range(count):
        data = self.rd32(self.offset)
        if j == index:
          self._reg_wr32(addr, data)
        self.offset += 4
      addr += incr

  def _op_ram_restrict_pll(self):
    type_ = self.rd08(self.offset + 1)
    count = self._ramcfg_count()
    index = self._ramcfg_index()
    self.offset += 2
    for i in range(count):
      freq = self.rd32(self.offset)
      if i == index:
        self._prog_pll(type_, freq)
      self.offset += 4

  def _op_zm_i2c(self):
    index = self.rd08(self.offset + 1)
    addr = self.rd08(self.offset + 2) >> 1
    count = self.rd08(self.offset + 3)
    self.offset += 4
    data = bytes(self.rd08(self.offset + i) for i in range(count))
    self.offset += count
    if self.init_exec():
      try:
        self._i2c_xfer(index, addr, [(False, data)])
      except NvbiosI2CError as e:
        print(f"[nvbios] ZM_I2C protocol failure ignored like Nouveau: {e}",
              flush=True)

  def _op_zm_i2c_byte(self):
    index = self.rd08(self.offset + 1)
    addr = self.rd08(self.offset + 2) >> 1
    count = self.rd08(self.offset + 3)
    self.offset += 4
    for _ in range(count):
      reg = self.rd08(self.offset)
      data = self.rd08(self.offset + 1)
      self.offset += 2
      if self.init_exec():
        try:
          self._i2c_write_reg(index, addr, reg, data)
        except NvbiosI2CError as e:
          print(f"[nvbios] ZM_I2C_BYTE failure ignored like Nouveau: {e}",
                flush=True)

  def _op_i2c_byte(self):
    index = self.rd08(self.offset + 1)
    addr = self.rd08(self.offset + 2) >> 1
    count = self.rd08(self.offset + 3)
    self.offset += 4
    for _ in range(count):
      reg = self.rd08(self.offset)
      mask = self.rd08(self.offset + 1)
      data = self.rd08(self.offset + 2)
      self.offset += 3
      if self.init_exec():
        try:
          value = self._i2c_read_reg(index, addr, reg)
        except NvbiosI2CError as e:
          print(f"[nvbios] I2C_BYTE read failure skipped like Nouveau: {e}",
                flush=True)
          continue
        try:
          self._i2c_write_reg(index, addr, reg, (value & mask) | data)
        except NvbiosI2CError as e:
          print(f"[nvbios] I2C_BYTE write failure ignored like Nouveau: {e}",
                flush=True)

  def _op_i2c_if(self):
    index = self.rd08(self.offset + 1)
    addr = self.rd08(self.offset + 2)
    reg = self.rd08(self.offset + 3)
    mask = self.rd08(self.offset + 4)
    data = self.rd08(self.offset + 5)
    self.offset += 6
    self.init_exec_force(True)
    try:
      value = self._i2c_read_reg(index, addr, reg)
    except NvbiosI2CError as e:
      value = e.u8
      print(f"[nvbios] I2C_IF read failure -> {value:#x} like Nouveau: {e}",
            flush=True)
    if (value & mask) != data:
      self.init_exec_set(False)
    self.init_exec_force(False)

  def _op_i2c_long_if(self):
    index = self.rd08(self.offset + 1)
    addr = self.rd08(self.offset + 2) >> 1
    reglo = self.rd08(self.offset + 3)
    reghi = self.rd08(self.offset + 4)
    mask = self.rd08(self.offset + 5)
    data = self.rd08(self.offset + 6)
    self.offset += 7
    try:
      reads = self._i2c_xfer(
          index, addr,
          [(False, bytes((reghi, reglo))), (True, 1)])
      matched = (reads[0][0] & mask) == data
    except NvbiosI2CError as e:
      print(f"[nvbios] I2C_LONG_IF read failure -> false like Nouveau: {e}",
            flush=True)
      matched = False
    if not matched:
      self.init_exec_set(False)

  def _op_tmds(self):
    self.offset += 5

  def _op_zm_tmds_group(self):
    count = self.rd08(self.offset + 2)
    self.offset += 3 + count * 2

  def _op_cr(self):
    addr = self.rd08(self.offset + 1)
    mask = self.rd08(self.offset + 2)
    data = self.rd08(self.offset + 3)
    self.offset += 4
    val = self._vgai_rd(0x03d4, addr) & mask
    self._vgai_wr(0x03d4, addr, val | data)

  def _op_zm_cr(self):
    addr = self.rd08(self.offset + 1)
    data = self.rd08(self.offset + 2)
    self.offset += 3
    self._vgai_wr(0x03d4, addr, data)

  def _op_zm_cr_group(self):
    count = self.rd08(self.offset + 1)
    self.offset += 2
    for _ in range(count):
      addr = self.rd08(self.offset)
      data = self.rd08(self.offset + 1)
      self.offset += 2
      self._vgai_wr(0x03d4, addr, data)

  def _op_cr_idx_adr_latch(self):
    addr0 = self.rd08(self.offset + 1)
    addr1 = self.rd08(self.offset + 2)
    base = self.rd08(self.offset + 3)
    count = self.rd08(self.offset + 4)
    self.offset += 5
    save0 = self._vgai_rd(0x03d4, addr0)
    for _ in range(count):
      data = self.rd08(self.offset)
      self.offset += 1
      self._vgai_wr(0x03d4, addr0, base)
      self._vgai_wr(0x03d4, addr1, data)
      base = (base + 1) & 0xff
    self._vgai_wr(0x03d4, addr0, save0)

  def _op_io(self):
    port = self.rd16(self.offset + 1)
    mask = self.rd08(self.offset + 3)
    data = self.rd08(self.offset + 4)
    self.offset += 5

    # Literal Nouveau init_io() NV50+ special case.  The Palit script executes
    # exactly INIT_IO port=0x3c3 mask=0 data=1 before its first DONE.
    if port == 0x03c3 and data == 0x01:
      self._reg_mask(0x614100, 0xf0800000, 0x00800000)
      self._reg_mask(0x00e18c, 0x00020000, 0x00020000)
      self._reg_mask(0x614900, 0xf0800000, 0x00800000)
      self._reg_mask(0x000200, 0x40000000, 0x00000000)
      if self.init_exec():
        time.sleep(0.010)
      self._reg_mask(0x00e18c, 0x00020000, 0x00000000)
      self._reg_mask(0x000200, 0x40000000, 0x40000000)
      self._reg_wr32(0x614100, 0x00800018)
      self._reg_wr32(0x614900, 0x00800018)
      if self.init_exec():
        time.sleep(0.010)
      self._reg_wr32(0x614100, 0x10000018)
      self._reg_wr32(0x614900, 0x10000018)

    value = self._port_rd(port) & mask
    self._port_wr(port, data | value)

  def _op_zm_index_io(self):
    self.offset += 5

  def _op_index_io(self):
    self.offset += 6

  def _op_condition(self):
    cond = self.rd08(self.offset + 1)
    if self.init_exec() and not self._condition_met(cond):
      self.init_exec_set(False)
    self.offset += 2

  def _op_io_condition(self):
    cond = self.rd08(self.offset + 1)
    if self.init_exec() and not self._io_condition_met(cond):
      self.init_exec_set(False)
    self.offset += 2

  def _op_io_flag_condition(self):
    cond = self.rd08(self.offset + 1)
    if self.init_exec() and not self._io_flag_condition_met(cond):
      self.init_exec_set(False)
    self.offset += 2

  def _op_strap_condition(self):
    mask = self.rd32(self.offset + 1)
    val = self.rd32(self.offset + 5)
    if self.init_exec() and (self._reg_rd32(0x101000) & mask) != val:
      self.init_exec_set(False)
    self.offset += 9

  def _op_ram_condition(self):
    mask = self.rd08(self.offset + 1)
    val = self.rd08(self.offset + 2)
    if self.init_exec() and (self._reg_rd32(0x100000) & mask) != val:
      self.init_exec_set(False)
    self.offset += 3

  def _op_generic_condition(self):
    cond = self.rd08(self.offset + 1)
    size = self.rd08(self.offset + 2)
    self.offset += 3
    # No display output available; default false for all conditions.
    if self.init_exec():
      self.init_exec_set(False)
    self.offset += size

  def _op_io_mask_or(self):
    self.offset += 2

  def _op_io_or(self):
    self.offset += 2

  def _op_condition_time(self):
    cond = self.rd08(self.offset + 1)
    retry = self.rd08(self.offset + 2)
    self.offset += 3
    if not self.init_exec():
      return
    wait = min(retry * 50, 100)
    ok = False
    while wait:
      if self._condition_met(cond):
        ok = True
        break
      time.sleep(0.020)
      wait -= 1
    if not ok:
      self.init_exec_set(False)

  def _op_time(self):
    usec = self.rd16(self.offset + 1)
    self.offset += 3
    if self.init_exec():
      if usec < 1000:
        time.sleep(usec / 1_000_000)
      else:
        time.sleep((usec + 900) / 1_000_000)

  def _op_ltime(self):
    msec = self.rd16(self.offset + 1)
    self.offset += 3
    if self.init_exec():
      time.sleep(msec / 1000)

  def _op_repeat(self):
    count = self.rd08(self.offset + 1)
    self.offset += 2
    old_repeat = self.repeat
    old_repend = self.repend
    self.repeat = self.offset
    self.repend = self.offset
    while count:
      self.offset = self.repeat
      self.run_script(self.offset)
      count -= 1
      if count:
        if self.debug:
          print(f"  [nvbios] REPEAT remaining {count}")
    self.offset = self.repend
    self.repeat = old_repeat
    self.repend = old_repend

  def _op_end_repeat(self):
    self.offset += 1
    if self.repeat:
      self.repend = self.offset
      self.offset = 0

  def _op_sub_direct(self):
    addr = self.rd16(self.offset + 1)
    if self.init_exec():
      save = self.offset + 3
      self.offset = addr
      self.run_script(self.offset)
      self.offset = save
    else:
      self.offset += 3

  def _op_sub(self):
    index = self.rd08(self.offset + 1)
    self.offset += 2
    st = self.tables.get("script", 0)
    if st and st + index * 2 + 2 <= len(self.image):
      addr = self.rd16(st + index * 2)
      if addr and self.init_exec():
        save = self.offset
        self.offset = addr
        self.run_script(self.offset)
        self.offset = save

  def _op_jump(self):
    target = self.rd16(self.offset + 1)
    if self.init_exec():
      self.offset = target
    else:
      self.offset += 3

  def _op_reset(self):
    reg = self.rd32(self.offset + 1)
    data1 = self.rd32(self.offset + 5)
    data2 = self.rd32(self.offset + 9)
    self.offset += 13
    self.init_exec_force(True)
    save = self._reg_mask(0x00184c, 0x00000f00, 0x00000000)
    self._reg_wr32(reg, data1)
    time.sleep(0.00001)
    self._reg_wr32(reg, data2)
    self._reg_wr32(0x00184c, save)
    self._reg_mask(0x001850, 0x00000001, 0x00000000)
    self.init_exec_force(False)

  def _op_macro(self):
    macro = self.rd08(self.offset + 1)
    self.offset += 2
    table = self.tables.get("macro", 0)
    if table and table + macro * 8 + 8 <= len(self.image):
      addr = self.rd32(table + macro * 8 + 0)
      data = self.rd32(table + macro * 8 + 4)
      self._reg_wr32(addr, data)

  def _op_pll(self):
    reg = self.rd32(self.offset + 1)
    freq = self.rd16(self.offset + 5) * 10
    self.offset += 7
    self._prog_pll(reg, freq)

  def _op_pll_indirect(self):
    reg = self.rd32(self.offset + 1)
    addr = self.rd16(self.offset + 5)
    freq = self.rd16(addr) * 10
    self.offset += 7
    self._prog_pll(reg, freq)

  def _op_pll2(self):
    reg = self.rd32(self.offset + 1)
    freq = self.rd32(self.offset + 5)
    self.offset += 9
    self._prog_pll(reg, freq)

  def _op_io_restrict_pll(self):
    port = self.rd16(self.offset + 1)
    index = self.rd08(self.offset + 3)
    mask = self.rd08(self.offset + 4)
    shift = self.rd08(self.offset + 5)
    iofc = self.rd08(self.offset + 6)
    count = self.rd08(self.offset + 7)
    reg = self.rd32(self.offset + 8)
    self.offset += 12
    conf = (self._vgai_rd(port, index) & mask) >> shift
    for i in range(count):
      freq = self.rd16(self.offset) * 10
      if i == conf:
        if iofc > 0 and self._io_flag_condition_met(iofc):
          freq *= 2
        self._prog_pll(reg, freq)
      self.offset += 2

  def _op_io_restrict_pll2(self):
    port = self.rd16(self.offset + 1)
    index = self.rd08(self.offset + 3)
    mask = self.rd08(self.offset + 4)
    shift = self.rd08(self.offset + 5)
    count = self.rd08(self.offset + 6)
    reg = self.rd32(self.offset + 7)
    self.offset += 11
    conf = (self._vgai_rd(port, index) & mask) >> shift
    for i in range(count):
      freq = self.rd32(self.offset)
      if i == conf:
        self._prog_pll(reg, freq)
      self.offset += 4

  def _op_io_restrict_prog(self):
    port = self.rd16(self.offset + 1)
    index = self.rd08(self.offset + 3)
    mask = self.rd08(self.offset + 4)
    shift = self.rd08(self.offset + 5)
    count = self.rd08(self.offset + 6)
    reg = self.rd32(self.offset + 7)
    self.offset += 11
    conf = (self._vgai_rd(port, index) & mask) >> shift
    for i in range(count):
      data = self.rd32(self.offset)
      if i == conf:
        self._reg_wr32(reg, data)
      self.offset += 4

  def _op_copy(self):
    reg = self.rd32(self.offset + 1)
    shift = self.rd08(self.offset + 5)
    smask = self.rd08(self.offset + 6)
    port = self.rd16(self.offset + 7)
    index = self.rd08(self.offset + 9)
    mask = self.rd08(self.offset + 10)
    self.offset += 11
    data = self._vgai_rd(port, index) & mask
    data |= self._shift(self._reg_rd32(reg), shift) & smask
    self._vgai_wr(port, index, data)

  def _op_xlat(self):
    sreg = self.rd32(self.offset + 1)
    sshift = self.rd08(self.offset + 5)
    smask = self.rd08(self.offset + 6)
    index = self.rd08(self.offset + 7)
    dreg = self.rd32(self.offset + 8)
    dmask = self.rd32(self.offset + 12)
    shift = self.rd08(self.offset + 16)
    self.offset += 17
    data = self._shift(self._reg_rd32(sreg), sshift) & smask
    # xlat table: lookup index -> data entry -> offset
    table = self.tables.get("xlat", 0)
    if table and table + index * 2 + 2 <= len(self.image):
      data_ptr = self.rd16(table + index * 2)
      if data_ptr and data_ptr + data < len(self.image):
        data = self.rd08(data_ptr + data)
    data <<= shift
    self._reg_mask(dreg, ~dmask, data)

  def _op_auxch(self):
    addr = self.rd32(self.offset + 1)
    count = self.rd08(self.offset + 5)
    self.offset += 6
    self.offset += count * 2

  def _op_zm_auxch(self):
    addr = self.rd32(self.offset + 1)
    count = self.rd08(self.offset + 5)
    self.offset += 6
    self.offset += count

  def _op_gpio(self):
    self.offset += 1
    if self.init_exec():
      count, changed = self._gpio_reset()
      if os.environ.get("KEPLER_GPIO_TRACE", "1") != "0":
        print(f"[nvbios] INIT_GPIO reset count={count} "
              f"changed={[f'{func:#x}' for func in changed]}", flush=True)

  def _op_gpio_ne(self):
    count = self.rd08(self.offset + 1)
    excluded = {self.rd08(self.offset + 2 + i) for i in range(count)}
    self.offset += 2 + count
    if self.init_exec():
      reset, changed = self._gpio_reset(excluded)
      if os.environ.get("KEPLER_GPIO_TRACE", "1") != "0":
        print(f"[nvbios] INIT_GPIO_NE reset count={reset} "
              f"changed={[f'{func:#x}' for func in changed]}", flush=True)

  def _op_compute_mem(self):
    self.offset += 1

  def _op_configure_mem(self):
    self.offset += 1

  def _op_configure_clk(self):
    self.offset += 1

  def _op_configure_preinit(self):
    self.offset += 1

  def _op_index_address_latched(self):
    count = self.rd08(self.offset + 17)
    self.offset += 19 + count * 2

  def _op_reserved(self):
    """0x92 and 0xaa: skip 1 and 4 bytes respectively."""
    if self.rd08(self.offset) == 0xaa:
      self.offset += 4
    else:
      self.offset += 1

  def _prog_pll(self, type_: int, freq: int) -> None:
    """Stub for PLL programming.  The VBIOS uses this for VPLL/MPLL, not GPC."""
    if self.debug:
      print(f"  [nvbios] PLL {type_:#010x} {freq}kHz (stub)")

  _handlers = {
    0x32: _op_io_restrict_prog,
    0x33: _op_repeat,
    0x34: _op_io_restrict_pll,
    0x36: _op_end_repeat,
    0x37: _op_copy,
    0x38: _op_not,
    0x39: _op_io_flag_condition,
    0x3a: _op_generic_condition,
    0x3b: _op_io_mask_or,
    0x3c: _op_io_or,
    0x47: _op_andn_reg,
    0x48: _op_or_reg,
    0x49: _op_index_address_latched,
    0x4a: _op_io_restrict_pll2,
    0x4b: _op_pll2,
    0x4c: _op_i2c_byte,
    0x4d: _op_zm_i2c_byte,
    0x4e: _op_zm_i2c,
    0x4f: _op_tmds,
    0x50: _op_zm_tmds_group,
    0x51: _op_cr_idx_adr_latch,
    0x52: _op_cr,
    0x53: _op_zm_cr,
    0x54: _op_zm_cr_group,
    0x56: _op_condition_time,
    0x57: _op_ltime,
    0x58: _op_zm_reg_sequence,
    0x59: _op_pll_indirect,
    0x5a: _op_zm_reg_indirect,
    0x5b: _op_sub_direct,
    0x5c: _op_jump,
    0x5e: _op_i2c_if,
    0x5f: _op_copy_nv_reg,
    0x62: _op_zm_index_io,
    0x63: _op_compute_mem,
    0x65: _op_reset,
    0x66: _op_configure_mem,
    0x67: _op_configure_clk,
    0x68: _op_configure_preinit,
    0x69: _op_io,
    0x6b: _op_sub,
    0x6d: _op_ram_condition,
    0x6e: _op_nv_reg,
    0x6f: _op_macro,
    0x71: _op_done,
    0x72: _op_resume,
    0x73: _op_strap_condition,
    0x74: _op_time,
    0x75: _op_condition,
    0x76: _op_io_condition,
    0x77: _op_zm_reg16,
    0x78: _op_index_io,
    0x79: _op_pll,
    0x7a: _op_zm_reg,
    0x87: _op_ram_restrict_pll,
    0x8c: _op_reset_begun,
    0x8d: _op_reset_end,
    0x8e: _op_gpio,
    0x8f: _op_ram_restrict_zm_reg_group,
    0x90: _op_copy_zm_reg,
    0x91: _op_zm_reg_group,
    0x92: _op_reserved,
    0x96: _op_xlat,
    0x97: _op_zm_mask_add,
    0x98: _op_auxch,
    0x99: _op_zm_auxch,
    0x9a: _op_i2c_long_if,
    0xa9: _op_gpio_ne,
    0xaa: _op_reserved,
  }



def find_vbios_scripts(image: bytes) -> list:
  """Return the list of NVINIT script offsets from the BIT-I script table."""
  bit = image.find(b"BIT")
  if bit < 2 or bit + 12 > len(image):
    return []
  hlen, rlen, count = image[bit + 6], image[bit + 7], image[bit + 8]
  ioff = 0
  for i in range(count):
    p = bit - 2 + hlen + i * rlen
    if p + 6 > len(image): break
    if image[p] == ord("I"):
      ioff = struct.unpack_from("<H", image, p + 4)[0]
      break
  if not ioff:
    raise ValueError("BIT-I record not found")
  # script table is the 16-bit pointer array at I+0
  st = struct.unpack_from("<H", image, ioff + 0)[0]
  scripts = []
  while st + 2 <= len(image):
    ptr = struct.unpack_from("<H", image, st)[0]
    if ptr == 0:
      break
    scripts.append(ptr)
    st += 2
  return scripts


def run_vbios_init(dev, image: bytes, scripts: list | None = None, debug: bool = False) -> None:
  """Execute the VBIOS devinit scripts sequentially."""
  init = NvbiosInit(dev, image, debug=debug)
  init.unlock_vga_crtc()
  if scripts is None:
    scripts = find_vbios_scripts(image)
  for s in scripts:
    if debug:
      print(f"[nvbios] === script 0x{s:04x} ===")
    init.run_script(s)


# ============================================================================
# TinyGPU transport + VRAM access
# ============================================================================

# ============================================================================
# TinyGPU transport
# ============================================================================
class RemoteCmd(enum.IntEnum):
  MAP_BAR    = 1
  CFG_READ   = 3
  CFG_WRITE  = 4
  RESET      = 5
  MMIO_READ  = 6
  MMIO_WRITE = 7

TINYGPU_SOCK = '/tmp/tinygpu.sock'

class Dev:
  def __init__(self):
    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self._sock.settimeout(10.0)
    for _ in range(100):
      try: self._sock.connect(TINYGPU_SOCK); break
      except: time.sleep(0.05)
  def _rpc(self, cmd, bar=0, a1=0, a2=0, a3=0, payload=b'', readout=0):
    self._sock.sendall(struct.pack('<BIIQQQ', int(cmd), 0, bar, a1, a2, a3) + payload)
    if payload: return None
    msg = self._recvall(17)
    status, v1, v2 = struct.unpack('<BQQ', msg)
    if status != 0:
      err = self._recvall(v1).decode() if v1 > 0 else 'unknown'
      raise RuntimeError(f'RPC failed: {err}')
    data = self._recvall(readout) if readout else b''
    return v1, v2, data
  def _recvall(self, n):
    buf = bytearray(n); got = 0
    while got < n:
      cnt = self._sock.recv_into(memoryview(buf)[got:]); got += cnt
    return bytes(buf)
  def read32(self, reg):
    _, _, d = self._rpc(RemoteCmd.MMIO_READ, 0, reg & 0xffffffff, 4, readout=4)
    return struct.unpack_from('<I', d)[0]
  def write32(self, reg, val):
    self._rpc(RemoteCmd.MMIO_WRITE, 0, reg & 0xffffffff, 4, payload=struct.pack('<I', val & 0xffffffff))
  def cfg(self, off, sz=4):
    return self._rpc(RemoteCmd.CFG_READ, 0, off, sz)[0]
  def map_bar(self, bar):
    v1, v2, _ = self._rpc(RemoteCmd.MAP_BAR, bar)
    return v1, v2
  def close(self):
    try: self._sock.close()
    except: pass

# ============================================================================
# VRAM access via PRAMIN window (BAR0 + 0x700000, window via reg 0x001700)
# ============================================================================
PRAMIN_WINDOW_REG = 0x001700
PRAMIN_BASE = 0x700000

class VRAM:
  def __init__(self, dev):
    self.dev = dev
    self._cur_window = None
  def _set_window(self, addr_base):
    page = addr_base >> 16
    if page != self._cur_window:
      self.dev.write32(PRAMIN_WINDOW_REG, page)
      self._cur_window = page
  def write32(self, vram_addr, val):
    base = vram_addr & 0xFFFFFF00000
    offset = vram_addr & 0x000000FFFFF
    self._set_window(base)
    self.dev.write32(PRAMIN_BASE + offset, val)
  def read32(self, vram_addr):
    base = vram_addr & 0xFFFFFF00000
    offset = vram_addr & 0x000000FFFFF
    self._set_window(base)
    return self.dev.read32(PRAMIN_BASE + offset)
  def write_block(self, vram_addr, data: bytes):
    for i in range(0, len(data), 4):
      val = struct.unpack_from('<I', data, i)[0] if i + 4 <= len(data) else \
            struct.unpack('<I', data[i:].ljust(4, b'\x00'))[0]
      self.write32(vram_addr + i, val)
  def read_block(self, vram_addr, size: int) -> bytes:
    data = bytearray()
    for i in range(0, size, 4):
      data.extend(struct.pack('<I', self.read32(vram_addr + i)))
    return bytes(data[:size])

# ============================================================================
# NV50 push buffer method encoding (cl506f.h)
# ============================================================================
def nvm50(subch, method, *args, inc=True):
  """Encode NV50 push buffer method.
  inc=True: incrementing; inc=False: non-incrementing.
  bits[31:29]=SEC_OP (0=inc, 2=noninc), bits[28:18]=count, bits[17:16]=TERT_OP(0),
  bits[15:13]=subch, bits[12:2]=method_dword_addr
  The method byte address goes at bits[12:0] (4-byte aligned, bits[1:0]=0).
  Hardware extracts dword addr from bits[12:2].
  """
  sec_op = 0 if inc else 2
  tert_op = 0  # INC_METHOD (GRP0) or NON_INC_METHOD (GRP2)
  count = len(args)
  hdr = (sec_op << 29) | (count << 18) | (tert_op << 16) | (subch << 13) | method
  return [hdr] + list(args)

# ============================================================================
# NV50 GPFIFO entry (cl506f.h)
# ============================================================================
def gp_entry(push_offset, length_dwords, priv=0, level=0, no_ctx_switch=0, disable=0):
  """8-byte GPFIFO entry (cl506f.h + nvif/chan506f.c).
  entry0: bit0=disable, bit1=no_context_switch, bits[31:2]=push_addr[31:2]
  entry1: bits[7:0]=addr_hi, bit8=priv, bit9=level, bits[31:10]=length_in_dwords
  length is in DWORDS (nouveau: (size >> 2) << 10), not bytes.
  """
  entry0 = (push_offset & ~3) | (no_ctx_switch << 1) | disable
  get_hi = (push_offset >> 32) & 0xFF
  entry1 = (length_dwords << 10) | (level << 9) | (priv << 8) | get_hi
  return struct.pack('<II', entry0, entry1)

# ============================================================================
# NV50 DMA object (nv50_dmaobj_bind)
# ============================================================================
def make_dma_object(start, limit, target='vram', access='rw', cls=0x00000002):
  """24-byte NV50 DMA object (nv50_dmaobj_bind format, usernv50.c).
  For non-VM target: priv=1(US), part=1(256B), comp=0(NONE), kind=0(PITCH)
  For VM target:     priv=0(VM), part=0(VM), comp=3(VM), kind=0x7f(VM)
  Constants from cl0002.h.
  """
  tgt = {'vram': 0x00010000, 'pci': 0x00020000, 'ncoh': 0x00030000, 'vm': 0}[target]
  if target == 'vm':
    acc = 0  # NV_MEM_ACCESS_VM → no access bits
    priv = 0  # NV50_DMA_V0_PRIV_VM
    comp, kind, part = 3, 0x7f, 0  # COMP_VM, KIND_VM, PART_VM
  else:
    acc = {'ro': 0x00040000, 'rw': 0x00080000}[access]
    priv = 1  # NV50_DMA_V0_PRIV_US
    comp, kind, part = 0, 0, 1  # COMP_NONE, KIND_PITCH, PART_256
  flags0 = (comp << 29) | (kind << 22) | (priv << 20) | cls | tgt | acc
  flags5 = (part << 16)
  return struct.pack('<IIIIII',
    flags0, limit & 0xFFFFFFFF, start & 0xFFFFFFFF,
    ((limit >> 32) & 0xFF) << 24 | ((start >> 32) & 0xFF),
    0, flags5)

# ============================================================================
# Constants
# ============================================================================
CHAN_ID = 1
USERD_BASE = 0xc00000
USERD_SIZE = 0x2000
# Flow control starts at 0x7c0 dwords = 0x1f00 bytes into USERD page
# But on NV50/G84, USERD maps inst block first 0x2000 bytes, control at 0x1f00
# Actually from testing: flow control is at offset 0 within USERD page
# Put=0x40, Get=0x44, Ref=0x48, GPGet=0x88, GPPut=0x8c

VRAM_BASE = 0x100000  # 1 MB into VRAM

# g84 instance block layout (all allocated within 0x10000 inst block):
# HEAD allocations:
#   0x0000: eng    (0x200 bytes, align 0)
#   0x0200: pgd    (0x4000 bytes, align 0)
#   0x4400: cache  (0x1000 bytes, align 0x400)
#   0x5400: ramfc  (0x100 bytes, align 0x100)
#   0x5500: ramht  (0x8000 bytes, align 16)
#   0xD500: (free)
# TAIL allocation:
#   0xFFE0: push DMA object (24 bytes, align 16, from tail)
INST_ENG_OFF    = 0x0000
INST_PGD_OFF    = 0x0200
INST_CACHE_OFF  = 0x4400
INST_RAMFC_OFF  = 0x5400
INST_RAMHT_OFF  = 0x5500
INST_PUSH_OFF   = 0xFFE0

# GMMU page table (1MB PGT in VRAM, identity-mapped for push buffer region)
PGT_ADDR = VRAM_BASE + 0x30000  # 1MB PGT at VRAM 0x130000

def setup_gmmu(vram, inst_addr):
  """Set up a minimal GMMU page table for identity-mapped VRAM pages.

  NV50 GMMU: 2-level page table.
  - PGD (in inst block at offset 0x0200): 11-bit index [39:29], 8-byte entries
  - PGT (1MB, 17-bit index [28:12] for 4KB pages): 8-byte entries
  PDE format: 0x003 | pgt_addr (4KB pages, 1MB PGT, VRAM target)
  PTE format: phys_addr | 0x1 (valid, VRAM aperture)
  """
  print('[gmmu] Setting up page table...')
  # Zero the PGT (1MB = 131072 entries * 8 bytes)
  # Only zero the first 512 entries (covers VA 0x00000000-0x001FF000)
  for i in range(512):
    vram.write32(PGT_ADDR + i * 8, 0)
    vram.write32(PGT_ADDR + i * 8 + 4, 0)

  # Write PDE in instance block at offset 0x0200 (pd_offset for g84)
  # PDE = 0x003 (4KB pages, 1MB PGT) | PGT_ADDR (VRAM target = 0)
  pde = 0x00000003 | PGT_ADDR
  vram.write32(inst_addr + INST_PGD_OFF, pde & 0xFFFFFFFF)
  vram.write32(inst_addr + INST_PGD_OFF + 4, (pde >> 32) & 0xFF)

  # Identity-map VRAM pages for inst (0x100000), push buf (0x110000), gpfifo (0x120000)
  # PGT index = (vaddr >> 12) & 0x1FFFF
  # PTE = phys_addr | 0x1 (valid + VRAM)
  for vaddr in range(0x100000, 0x130000, 0x1000):
    pgt_idx = (vaddr >> 12) & 0x1FFFF
    pte = vaddr | 0x1  # identity mapping: virtual = physical
    vram.write32(PGT_ADDR + pgt_idx * 8, pte & 0xFFFFFFFF)
    vram.write32(PGT_ADDR + pgt_idx * 8 + 4, (pte >> 32) & 0xFF)

  print(f'[gmmu] PGT at 0x{PGT_ADDR:08x}, PDE=0x{pde:08x}')
  print(f'[gmmu] Mapped VA 0x100000-0x12FFFF -> phys (identity)')

  # Verify PDE
  pde_r0 = vram.read32(inst_addr + INST_PGD_OFF)
  pde_r1 = vram.read32(inst_addr + INST_PGD_OFF + 4)
  print(f'[gmmu] PDE readback: 0x{pde_r1:08x}{pde_r0:08x}')

  # Verify a few PTEs
  for vaddr in [0x100000, 0x110000, 0x120000]:
    pgt_idx = (vaddr >> 12) & 0x1FFFF
    pte_r0 = vram.read32(PGT_ADDR + pgt_idx * 8)
    pte_r1 = vram.read32(PGT_ADDR + pgt_idx * 8 + 4)
    print(f'[gmmu] PTE[{pgt_idx:#x}] (VA 0x{vaddr:x}): 0x{pte_r1:08x}{pte_r0:08x}')

  # Flush GMMU TLB for all known engine IDs
  for eng_id in [0x00, 0x01, 0x06, 0x08, 0x09, 0x0a, 0x0d]:
    vram.dev.write32(0x100c80, (eng_id << 16) | 1)
    for _ in range(2000):
      if not (vram.dev.read32(0x100c80) & 0x00000001):
        break
      time.sleep(0.001)
  print('[gmmu] TLB flushed')

def fifo_init(dev):
  """nv50_fifo_init."""
  print('[fifo] Init...')
  dev.write32(0x000200, dev.read32(0x000200) & ~0x00000100)
  dev.write32(0x000200, dev.read32(0x000200) | 0x00000100)
  dev.write32(0x00250c, 0x6f3cfc34)
  dev.write32(0x002044, 0x01003fff)
  dev.write32(0x002100, 0xffffffff)  # clear intr
  dev.write32(0x002140, 0xbfffffff)  # intr enable
  for i in range(128):
    dev.write32(0x002600 + i * 4, 0)
  # Empty runlist update
  dev.write32(0x0032f4, 0)
  dev.write32(0x0032ec, 0)
  dev.write32(0x003200, 0x00000001)
  dev.write32(0x003250, 0x00000001)
  dev.write32(0x002500, 0x00000001)
  print(f'[fifo] PFIFO_ENABLE=0x{dev.read32(0x002500):08x}')

CHIPSET = 0xa8  # GT218

def gr_init(dev, ctxprog=None, ctxvals=None, ctxvals_size=0):
  """nv50_gr_init for GT218 (chipset 0xa8).
  ctxprog: list of 32-bit instructions (from ctxnv50.py)
  ctxvals: bytes buffer of default register values
  ctxvals_size: size of ctxvals buffer
  """
  print('[gr] Init...')
  # HW context switch enable
  dev.write32(0x40008c, 0x00000004)

  # Reset/enable traps
  dev.write32(0x400804, 0xc0000000)
  dev.write32(0x406800, 0xc0000000)
  dev.write32(0x400c04, 0xc0000000)
  dev.write32(0x401800, 0xc0000000)
  dev.write32(0x405018, 0xc0000000)
  dev.write32(0x402000, 0xc0000000)

  # Per-TP trap clear (chipset >= 0xa0)
  units = dev.read32(0x001540)
  print(f'[gr] Units bitmap: 0x{units:08x}')
  for i in range(16):
    if not (units & (1 << i)):
      continue
    dev.write32(0x408600 + (i << 11), 0xc0000000)
    dev.write32(0x408708 + (i << 11), 0xc0000000)
    dev.write32(0x40831c + (i << 11), 0xc0000000)

  # Interrupt enable
  dev.write32(0x400108, 0xffffffff)
  dev.write32(0x400138, 0xffffffff)
  dev.write32(0x400100, 0xffffffff)
  dev.write32(0x40013c, 0xffffffff)
  dev.write32(0x400500, 0x00010001)

  # Upload ctxprog
  if ctxprog:
    print(f'[gr] Uploading ctxprog ({len(ctxprog)} instructions)...')
    dev.write32(0x400324, 0)  # reset index
    for instr in ctxprog:
      dev.write32(0x400328, instr)
    print(f'[gr] Ctxprog uploaded, ctxvals_size={ctxvals_size}')
  else:
    print('[gr] WARNING: No ctxprog provided — GR will not work!')

  # Clear context pointers
  dev.write32(0x400824, 0)
  dev.write32(0x400828, 0)
  dev.write32(0x40082c, 0)
  dev.write32(0x400830, 0)
  dev.write32(0x40032c, 0)
  dev.write32(0x400330, 0)

  # ZCULL config (chipset 0xa8, not a0/aa/ac)
  dev.write32(0x402cc0, 0x00000000)
  dev.write32(0x402ca8, 0x00000002)

  # Zero ZCULL regions
  for i in range(8):
    dev.write32(0x402c20 + i * 0x10, 0)
    dev.write32(0x402c24 + i * 0x10, 0)
    dev.write32(0x402c28 + i * 0x10, 0)
    dev.write32(0x402c2c + i * 0x10, 0)

  print('[gr] Init complete')
  return ctxvals_size

def gr_bind_context(dev, vram, chan_id, inst_addr, ctx_addr, ctx_size):
  """Bind a GR context to the channel (g84_ectx_bind for GR engine).
  ptr0 = 0x0020 for GR engine.
  The context buffer must be in VRAM, filled with ctxvals.
  """
  print(f'[gr] Binding GR context to channel {chan_id}...')
  # eng block is at inst+0x0000 (INST_ENG_OFF)
  eng_addr = inst_addr + INST_ENG_OFF
  ptr0 = 0x0020  # GR engine
  flags = 0x00190000
  limit = ctx_addr + ctx_size - 1

  vram.write32(eng_addr + ptr0 + 0x00, flags)
  vram.write32(eng_addr + ptr0 + 0x04, limit & 0xFFFFFFFF)
  vram.write32(eng_addr + ptr0 + 0x08, ctx_addr & 0xFFFFFFFF)
  vram.write32(eng_addr + ptr0 + 0x0c,
    ((limit >> 32) & 0xFF) << 24 | (ctx_addr >> 32) & 0xFF)
  vram.write32(eng_addr + ptr0 + 0x10, 0)
  vram.write32(eng_addr + ptr0 + 0x14, 0)
  print(f'[gr] GR context: addr=0x{ctx_addr:x} size={ctx_size} limit=0x{limit:x}')

def create_channel(dev, vram, push_words=None, setup_fn=None):
  """Create a G84_CHANNEL_GPFIFO channel and submit push words via GPFIFO.
  setup_fn(vram, inst_addr) is called after RAMFC+GMMU setup, before channel bind.
  Use it to add RAMHT entries, GR context binding, extra GMMU mappings, etc.
  """
  chan_id = CHAN_ID
  inst_addr = VRAM_BASE
  push_buf_addr = VRAM_BASE + 0x10000  # 64KB push buffer
  gp_fifo_addr = VRAM_BASE + 0x20000   # GPFIFO ring (4KB = 512 entries)
  runlist_addr = VRAM_BASE + 0x21000   # Runlist

  if push_words is None:
    push_words = [0x00000000]  # NOP

  print(f'[chan] Channel {chan_id}, inst=0x{inst_addr:08x}')
  print(f'[chan] push=0x{push_buf_addr:08x}, gpfifo=0x{gp_fifo_addr:08x}')

  # Zero instance block (0x10000 bytes)
  print('[chan] Zeroing instance block...')
  for off in range(0, 0x10000, 4):
    vram.write32(inst_addr + off, 0)

  # Set up GMMU page table (identity-mapped VRAM pages)
  setup_gmmu(vram, inst_addr)

  # Write push DMA object at tail (offset 0xFFE0 in inst)
  # Try VM target with GMMU identity mapping (Tesla uses VM target for GART pushbuf)
  push_dma = make_dma_object(0, 0xFFFFFFFFFF, target='vm', access='rw')
  vram.write_block(inst_addr + INST_PUSH_OFF, push_dma)
  print(f'[chan] Push DMA at inst+0x{INST_PUSH_OFF:x}')

  # Write RAMFC at offset 0x5400 in inst
  ramfc_addr = inst_addr + INST_RAMFC_OFF
  gp_fifo_size = 0x1000  # 4KB = 512 entries
  gp_limit2 = 9  # ilog2(0x1000 / 8) = ilog2(512) = 9
  ramht_bits = 12  # order_base_2(0x8000/8) = order_base_2(0x1000) = 12

  vram.write32(ramfc_addr + 0x3c, 0x403f6078)
  vram.write32(ramfc_addr + 0x44, 0x01003fff)
  vram.write32(ramfc_addr + 0x48, INST_PUSH_OFF >> 4)  # push DMA offset >> 4
  vram.write32(ramfc_addr + 0x50, gp_fifo_addr & 0xFFFFFFFF)  # GPFIFO addr low
  vram.write32(ramfc_addr + 0x54, ((gp_fifo_addr >> 32) & 0xFFFF) | (gp_limit2 << 16))
  vram.write32(ramfc_addr + 0x60, 0x7fffffff)
  vram.write32(ramfc_addr + 0x78, 0x00000000)
  vram.write32(ramfc_addr + 0x7c, 0x30000000 | 0xfff)  # devm=0xfff
  vram.write32(ramfc_addr + 0x80, ((ramht_bits - 9) << 27) | (4 << 24) | (INST_RAMHT_OFF >> 4))
  vram.write32(ramfc_addr + 0x88, (inst_addr + INST_CACHE_OFF) >> 10)  # cache abs addr >> 10
  vram.write32(ramfc_addr + 0x98, inst_addr >> 12)  # inst abs addr >> 12

  print('[chan] RAMFC:')
  for off in [0x3c, 0x44, 0x48, 0x50, 0x54, 0x60, 0x78, 0x7c, 0x80, 0x88, 0x98]:
    print(f'  [0x{off:02x}] = 0x{vram.read32(ramfc_addr + off):08x}')

  # Write push buffer data
  for i, w in enumerate(push_words):
    vram.write32(push_buf_addr + i * 4, w)
  push_len = len(push_words) * 4
  print(f'[chan] Push buffer: {len(push_words)} words ({push_len} bytes)')

  # Zero GPFIFO ring (prevent garbage entries causing MEM_FAULT)
  for off in range(0, 0x1000, 4):
    vram.write32(gp_fifo_addr + off, 0)
  # GPFIFO entry: GET=push_buf_addr (VM addr), length in DWORDS
  entry = gp_entry(push_buf_addr, len(push_words))
  vram.write_block(gp_fifo_addr, entry)
  print(f'[chan] GPFIFO entry: GET=0x{push_buf_addr:x} len={len(push_words)} dwords')

  # Call setup_fn before binding (for RAMHT, GR context, extra GMMU, etc.)
  if setup_fn:
    setup_fn(vram, inst_addr)

  # Bind channel: g84_chan_bind writes ramfc->addr >> 8
  chan_reg = 0x002600 + chan_id * 4
  dev.write32(chan_reg, ramfc_addr >> 8)
  print(f'[chan] Bind: chan_reg = 0x{ramfc_addr >> 8:08x}')

  # Update runlist: write channel ID, commit
  vram.write32(runlist_addr, chan_id)
  dev.write32(0x0032f4, runlist_addr >> 12)
  dev.write32(0x0032ec, 1)
  for _ in range(100):
    if not (dev.read32(0x0032ec) & 0x00000100): break
    time.sleep(0.01)
  print(f'[chan] Runlist committed, 0x0032ec=0x{dev.read32(0x0032ec):08x}')

  # Start channel: set bit 31
  dev.write32(chan_reg, (ramfc_addr >> 8) | 0x80000000)
  print(f'[chan] Started: chan_reg = 0x{(ramfc_addr >> 8) | 0x80000000:08x}')

  # Clear any pending interrupts
  dev.write32(0x002100, 0xffffffff)

  # Set GPPut to trigger GPFIFO processing
  userd = USERD_BASE + chan_id * USERD_SIZE
  print(f'[chan] USERD at BAR0+0x{userd:06x}')

  # Read initial flow control state
  print('[chan] Initial USERD:')
  for name, off in [('Put', 0x40), ('Get', 0x44), ('Ref', 0x48),
                    ('GPGet', 0x88), ('GPPut', 0x8c)]:
    print(f'  {name:6s} [0x{off:02x}] = 0x{dev.read32(userd + off):08x}')

  # Write GPPut = 1 (entry index, NOT byte offset — see nvif/chan506f.c)
  dev.write32(userd + 0x8c, 1)
  print(f'[chan] Wrote GPPut=1 (entry index)')

  # Poll for completion
  for attempt in range(200):
    time.sleep(0.02)
    gp_get = dev.read32(userd + 0x88)
    gp_put = dev.read32(userd + 0x8c)
    fifo_intr = dev.read32(0x002100)
    if gp_get == gp_put or fifo_intr != 0:
      break

  print(f'[chan] After poll ({attempt+1} iterations):')
  print(f'  GPGet=0x{gp_get:08x} GPPut=0x{gp_put:08x}')
  print(f'  PFIFO_INTR=0x{fifo_intr:08x}')

  # Read all flow control fields
  print('[chan] Final USERD:')
  for name, off in [('Put', 0x40), ('Get', 0x44), ('Ref', 0x48), ('PutHi', 0x4c),
                    ('SetRef', 0x50), ('TopGet', 0x58), ('TopGetHi', 0x5c),
                    ('GetHi', 0x60), ('GPGet', 0x88), ('GPPut', 0x8c)]:
    print(f'  {name:10s} [0x{off:02x}] = 0x{dev.read32(userd + off):08x}')

  # DMA pusher error registers
  if fifo_intr & 0x1000:
    print('[chan] DMA_PUSHER error!')
    dma_get = dev.read32(0x003244)
    dma_put = dev.read32(0x003240)
    push_state = dev.read32(0x003220)
    dma_state = dev.read32(0x003228)
    ho_get = dev.read32(0x003328)
    ho_put = dev.read32(0x003320)
    ib_get = dev.read32(0x003334)
    ib_put = dev.read32(0x003330)
    err = ['NONE', 'CALL_SUBR', 'INVALID_MTHD', 'RET_SUBR',
           'INVALID_CMD', 'IB_EMPTY', 'MEM_FAULT', 'UNK'][(dma_state >> 29) & 7]
    print(f'  DMA_GET=0x{dma_get:08x} DMA_PUT=0x{dma_put:08x}')
    print(f'  HO_GET=0x{ho_get:08x} HO_PUT=0x{ho_put:08x}')
    print(f'  IB_GET=0x{ib_get:08x} IB_PUT=0x{ib_put:08x}')
    print(f'  DMA_STATE=0x{dma_state:08x} (err: {err})')
    print(f'  PUSH=0x{push_state:08x}')

  return chan_id

# ============================================================================
# Compute kernel launch (NV50_COMPUTE / NVA3_COMPUTE)
# ============================================================================

# ============================================================================
# Kernel source + Tesla SASS assembler
# ============================================================================
# CUDA source for the vector-add kernel (sm_12 / Tesla ISA):
KERNEL_CU = """
extern "C" __global__ void add(const float *a, const float *b, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = a[i] + b[i];
}
"""

# PTX 4.1 for sm_12 (compiled from KERNEL_CU via nvcc --gpu-architecture=sm_12):
KERNEL_PTX = """\
.version 4.1
.target sm_12
.address_size 32

.visible .entry add(
    .param .u32 __cudap_a,
    .param .u32 __cudap_b,
    .param .u32 __cudap_out,
    .param .u32 __cudap_N
)
{
    .reg .u32 %r<10>;
    .reg .f32 %f<4>;
    .reg .pred %p<2>;

    ld.param.u32 %r1, [__cudap_a];
    ld.param.u32 %r2, [__cudap_b];
    ld.param.u32 %r3, [__cudap_out];
    ld.param.u32 %r4, [__cudap_N];

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %ctaid.x;
    mad.lo.u32 %r8, %r7, %r6, %r5;

    setp.ge.u32 %p1, %r8, %r4;
    @%p1 ret;

    shl.b32 %r0, %r8, 2;

    add.u32 %r5, %r1, %r0;
    ld.global.f32 %f1, [%r5];

    add.u32 %r7, %r2, %r0;
    ld.global.f32 %f2, [%r7];

    add.f32 %f3, %f1, %f2;
    add.u32 %r8, %r3, %r0;
    st.global.f32 [%r8], %f3;

    ret;
}
"""

# Kernel metadata (from cubin: bar=0, reg=4, lmem=0, smem=32)
KERNEL_REGS = 4
KERNEL_SMEM = 32

# ----------------------------------------------------------------------------
# Tesla (NV50) instruction encoders
# ----------------------------------------------------------------------------
# ISA reference: envytools/docs/hw/graph/tesla/cuda/isa.rst + envydis/g80.c
# Instruction types:
#   short normal:  word0 bits 0-1 = 0  (1 dword)
#   long normal:   word0 bits 0-1 = 1, word1 bits 0-1 = 0  (2 dwords)
#   long + exit:   word0 bits 0-1 = 1, word1 bits 0-1 = 2  (2 dwords)
#   long control:  word0 bits 0-1 = 3  (2 dwords)

# Half-register: bit 0 = high/low, bits 1+ = register index
def _half(reg, high):
  return (reg << 1) | (1 if high else 0)

# Shared memory operand for short (6-bit field):
#   16-bit: (0b01 << 4) | offset  (unsigned s[])
#   32-bit: (0b11 << 4) | offset  (s[] space)
def _ssh16(off):  return (0b01 << 4) | (off & 0xF)
def _ssh32(off):  return (0b11 << 4) | (off & 0xF)

# Shared memory operand for long (7-bit field):
#   High 3 bits = subspace, low 5 bits = offset
#   32-bit s[]: subspace = 0b011
def _lsh32(off):  return (0b011 << 5) | (off & 0x1F)

# Predicates (isa.rst §Predicates)
_PRED_ALWAYS = 0x0F
_PRED_NEU    = 0x0D  # not equal or unordered (for RET)

# Long normal word0: op[31:28] areg[27:26] s3t[24] s2t[23] src2[22:16] src1[15:9] dst[8:2] type[1:0]
#   type = 1 for long normal, 3 for long control (RET, BRA, etc.)
def _lw0(op, dst, src1, src2=0, s2t=0, s3t=0, areg=0, ctrl=False):
  return (3 if ctrl else 1) | (dst << 2) | (src1 << 9) | (src2 << 16) | (s2t << 23) | (s3t << 24) | (areg << 26) | (op << 28)

# Long normal word1: secop[31:29] m2[27] m1[26] cidx[25:22] s1t[21] src3[20:14] csrc[13:12] pred[11:7] cw[6] cdst[5:4] dst_t[3] areg_hi[2] exit[1:0]
#   exit flag: bits 0-1 = 1 (empirically verified from nvcc output + envydis g80.c)
def _lw1(secop=0, areg_hi=0, dst_t=0, cdst=0, cwrite=0, pred=_PRED_ALWAYS,
         csrc=0, src3=0, s1t=0, cidx=0, m1=0, m2=0, exit=False):
  return ((1 if exit else 0) | (areg_hi << 2) | (dst_t << 3) | (cdst << 4) |
          (cwrite << 6) | (pred << 7) | (csrc << 12) | (src3 << 14) |
          (s1t << 21) | (cidx << 22) | (m1 << 26) | (m2 << 27) | (secop << 29))

# Short normal: op[31:28] areg[27:26] s1t[24] s2t[23] m3[22] src2[21:16] m2[15] src1[14:9] m1[8] dst[7:2] 0[1:0]
def _sw(op, dst, src1, src2, m1=0, m2=0, m3=0, s2t=0, s1t=0, areg=0):
  return ((dst << 2) | (m1 << 8) | (src1 << 9) | (m2 << 15) | (src2 << 16) |
          (m3 << 22) | (s2t << 23) | (s1t << 24) | (areg << 26) | (op << 28))

def _emit_long(c0, c1):
  return struct.pack('<II', c0 & 0xFFFFFFFF, c1 & 0xFFFFFFFF)

def _emit_short(c0):
  return struct.pack('<I', c0 & 0xFFFFFFFF)

# ----------------------------------------------------------------------------
# build_kernel_sass: hand-assemble the vector-add kernel for sm_12 (Tesla)
# ----------------------------------------------------------------------------
# Disassembly (from nvdisasm 6.5):
#   G2R.U16 R0H, g[0x6].U16           ; blockIdx.x -> R0 high half
#   I2I.U32.U16 R1, R0L               ; cvt u32 u16: threadIdx.x -> R1
#   IMUL32.U16.U16 R0, g[0x1].U16, R0H; R0 = blockDim * blockIdx
#   IADD32 R0, R0, R1                 ; R0 = global thread index i
#   ISET.C0 o[0x7f], g[0x7], R0, LE   ; set C0 if N <= i
#   RET C0.NEU                        ; early exit if i >= N
#   SHL R2, R0, 0x2                   ; R2 = byte offset
#   IADD32 R0, g[0x4], R2             ; R0 = &a[i]
#   IADD32 R3, g[0x5], R2             ; R3 = &b[i]
#   GLD.U32 R1, global14[R0]          ; R1 = a[i]
#   GLD.U32 R0, global14[R3]          ; R0 = b[i]
#   FADD32 R0, R1, R0                 ; R0 = a[i] + b[i]
#   IADD32 R1, g[0x6], R2             ; R1 = &out[i]
#   GST.U32 global14[R1], R0 (exit)   ; out[i] = result, then exit
#
# Shared memory layout (USER_PARAM slots):
#   g[0x1] = blockDim.x (U16)   g[0x4] = a ptr   g[0x5] = b ptr
#   g[0x6] = out ptr            g[0x7] = N
# global14 = the global memory space used by nvcc for GLD/GST

def build_kernel_sass():
  """Hand-assemble the vector-add kernel for Tesla (sm_12) ISA.

  Returns 88 bytes of SASS (14 instructions: 8 long + 6 short).
  Verified bit-exact against CUDA 6.5 nvcc output via nvdisasm.
  """
  out = bytearray()

  # G2R.U16 R0H, g[0x6].U16 — load blockIdx.x into high half of R0
  # Long mov (op=1), secop=2 selects shared-memory load, src1=raw offset
  out += _emit_long(_lw0(1, _half(0, True), 6), _lw1(secop=2, src3=1))

  # I2I.U32.U16 R1, R0L — zero-extend threadIdx.x (low half of R0) to 32-bit
  out += _emit_long(_lw0(0xA, 1, _half(0, False)), _lw1(m1=1))

  # IMUL32.U16.U16 R0, g[0x1].U16, R0H — R0 = blockDim.x * blockIdx.x
  out += _emit_short(_sw(4, 0, _ssh16(1), _half(0, True), s1t=1))

  # IADD32 R0, R0, R1 — R0 = blockIdx*blockDim + threadIdx (global index i)
  out += _emit_short(_sw(2, 0, 0, 1, m2=1))  # m2=1 → b32

  # ISET.C0 o[0x7f], g[0x7], R0, LE — set condition C0 if N <= i
  out += _emit_long(_lw0(3, 127, _lsh32(7)),
                    _lw1(secop=3, dst_t=1, cwrite=1, src3=3, s1t=1, m1=1))

  # RET C0.NEU — return if C0 is not-equal-or-unordered (i >= N)
  # RET is a long control instruction (word0 bits 0-1 = 3)
  out += _emit_long(_lw0(3, 0, 0, ctrl=True), _lw1(pred=_PRED_NEU))

  # SHL R2, R0, 0x2 — R2 = i << 2 (byte offset)
  out += _emit_long(_lw0(3, 2, 0, src2=2), _lw1(secop=6, src3=64, m1=1))

  # IADD32 R0, g[0x4], R2 — R0 = a_ptr + byte_offset
  out += _emit_short(_sw(2, 0, _ssh32(4), 2, m2=1, s1t=1))

  # IADD32 R3, g[0x5], R2 — R3 = b_ptr + byte_offset
  out += _emit_short(_sw(2, 3, _ssh32(5), 2, m2=1, s1t=1))

  # GLD.U32 R1, global14[R0] — load a[i]
  out += _emit_long(_lw0(0xD, 1, 0, 14), _lw1(secop=4, cidx=3))

  # GLD.U32 R0, global14[R3] — load b[i]
  out += _emit_long(_lw0(0xD, 0, 3, 14), _lw1(secop=4, cidx=3))

  # FADD32 R0, R1, R0 — R0 = a[i] + b[i]
  out += _emit_short(_sw(0xB, 0, 1, 0))

  # IADD32 R1, g[0x6], R2 — R1 = out_ptr + byte_offset
  out += _emit_short(_sw(2, 1, _ssh32(6), 2, m2=1, s1t=1))

  # GST.U32 global14[R1], R0 (exit) — store result, then exit
  out += _emit_long(_lw0(0xD, 0, 1, 14), _lw1(secop=5, cidx=3, exit=True))

  return bytes(out)

def assemble_kernel_sass():
  """Try to compile KERNEL_PTX via ptxas 6.5 and extract SASS.

  ptxas 6.5 is the last version supporting sm_12.  It's x86-64 only, so on
  ARM macOS we try Docker.  Returns raw SASS bytes or None.
  """
  import subprocess, tempfile
  candidates = [
    os.environ.get("EN210_PTXAS"),
    os.path.join(os.path.dirname(__file__), "cuda65-bin", "ptxas"),
  ]
  ptxas = next((p for p in candidates if p and os.path.isfile(p) and
                os.access(p, os.X_OK)), None)
  if ptxas is None:
    return None
  try:
    with tempfile.TemporaryDirectory(prefix="en210_ptxas_") as d:
      src = os.path.join(d, "add.ptx")
      cubin = os.path.join(d, "add.cubin")
      with open(src, "w") as f:
        f.write(KERNEL_PTX)
      subprocess.run([ptxas, "-arch=sm_12", src, "-o", cubin],
                     check=True, capture_output=True, timeout=60)
      with open(cubin, "rb") as f:
        blob = f.read()
      # Extract .text section from cubin ELF (ptxas 6.5 emits ELF64)
      ei_class = blob[4]
      if ei_class == 2:
        e_shoff = struct.unpack_from("<Q", blob, 0x28)[0]
        e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3a)
        e_shstrndx = struct.unpack_from("<H", blob, 0x3e)[0]
        shstr_hdr = e_shoff + e_shstrndx * e_shentsize
        shstr_off = struct.unpack_from("<Q", blob, shstr_hdr + 0x18)[0]
        for i in range(e_shnum):
          off = e_shoff + i * e_shentsize
          name_idx = struct.unpack_from("<I", blob, off)[0]
          sh_offset = struct.unpack_from("<Q", blob, off + 0x18)[0]
          sh_size = struct.unpack_from("<Q", blob, off + 0x20)[0]
          end = blob.find(b"\x00", shstr_off + name_idx)
          name = blob[shstr_off + name_idx:end].decode()
          if name.startswith(".text"):
            sass = blob[sh_offset:sh_offset + sh_size]
            print(f"[kernel] ptxas compiled {len(sass)} bytes of SASS")
            return sass
      else:
        e_shoff = struct.unpack_from("<I", blob, 0x20)[0]
        e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x2e)
        e_shstrndx = struct.unpack_from("<H", blob, 0x32)[0]
        shstr_hdr = e_shoff + e_shstrndx * e_shentsize
        shstr_off = struct.unpack_from("<I", blob, shstr_hdr + 0x10)[0]
        for i in range(e_shnum):
          off = e_shoff + i * e_shentsize
          name_idx = struct.unpack_from("<I", blob, off)[0]
          sh_offset = struct.unpack_from("<I", blob, off + 0x10)[0]
          sh_size = struct.unpack_from("<I", blob, off + 0x14)[0]
          end = blob.find(b"\x00", shstr_off + name_idx)
          name = blob[shstr_off + name_idx:end].decode()
          if name.startswith(".text"):
            sass = blob[sh_offset:sh_offset + sh_size]
            print(f"[kernel] ptxas compiled {len(sass)} bytes of SASS")
            return sass
  except OSError as e:
    if e.errno != 8:  # ENOEXEC — expected on ARM, don't print
      print(f"[kernel] ptxas failed: {e}")
  except Exception as e:
    print(f"[kernel] ptxas compilation failed: {e}")
  return None

def get_kernel_sass():
  """Return the kernel SASS bytes.

  Tries ptxas 6.5 first (if available), then hand-assembles via
  build_kernel_sass().  The hand-assembled output is verified against
  the known-good CUDA 6.5 nvcc SASS.
  """
  sass = assemble_kernel_sass()
  if sass is not None:
    return sass
  sass = build_kernel_sass()
  print(f"[kernel] Hand-assembled {len(sass)} bytes of SASS ({len(sass)//4} words)")
  return sass

# ============================================================================
# Constants
# ============================================================================
# GT218 (0xa8) uses NVA3_COMPUTE_CLASS (0x85c0), not NV50_COMPUTE (0x50c0).
# Source: mesa/src/gallium/drivers/nouveau/nv50/nv50_compute.c:54
NVA3_COMPUTE_CLASS = 0x85c0
COMPUTE_HANDLE     = 0x85c0
VRAM_DMA_HANDLE    = 0xbeef0201

# Engine IDs from g98_fifo_runl_ctor (g98.c:37-44):
#   SW/DMAOBJ: engi=0, GR: engi=1, MSPPP: engi=2, CE: engi=3, ...
GR_ENGN_ID    = 1
DMA_ENGN_ID   = 0

# Compute subchannel (mesa nv50_winsys.h:53: SUBC_CP = 6)
SUBC_CP = 6

# NV50_COMPUTE method addresses (nv50_compute.xml.h + cl50c0.h)
CP_SET_OBJECT             = 0x0000
CP_WAIT_FOR_IDLE          = 0x0110
CP_DMA_GLOBAL             = 0x01a0
CP_DMA_LOCAL              = 0x01b8
CP_DMA_STACK              = 0x01bc
CP_DMA_CODE_CB            = 0x01c0
CP_STACK_ADDRESS_HIGH     = 0x0218
CP_STACK_SIZE_LOG         = 0x0220
CP_SHADER_SCHEDULING      = 0x0290
CP_LOCAL_ADDRESS_HIGH     = 0x0294
CP_LOCAL_SIZE_LOG         = 0x029c
CP_WORK_DISTRIBUTION      = 0x02a0
CP_BLOCK_ALLOC            = 0x02b4
CP_LANES32_ENABLE         = 0x02b8
CP_CP_REG_ALLOC_TEMP      = 0x02c0
CP_BLOCKDIM_LATCH         = 0x02f8
CP_LOCAL_WARPS_LOG_ALLOC  = 0x02fc
CP_LOCAL_WARPS_NO_CLAMP   = 0x0300
CP_STACK_WARPS_LOG_ALLOC  = 0x0304
CP_STACK_WARPS_NO_CLAMP   = 0x0308
CP_CODE_CB_FLUSH          = 0x0380
CP_UNK0384                = 0x0384
CP_GRIDID                 = 0x0388
CP_LAUNCH                 = 0x0368
CP_GRIDDIM                = 0x03a4
CP_SHARED_SIZE            = 0x03a8
CP_BLOCKDIM_XY            = 0x03ac
CP_BLOCKDIM_Z             = 0x03b0
CP_CP_START_ID            = 0x03b4
CP_REG_MODE               = 0x03b8
CP_USER_PARAM_COUNT       = 0x0374

def CP_GLOBAL_ADDR_HI(i):  return 0x0400 + 0x20 * i
def CP_GLOBAL_LIMIT(i):    return 0x040c + 0x20 * i
def CP_GLOBAL_MODE(i):     return 0x0410 + 0x20 * i
def CP_USER_PARAM(i):      return 0x0600 + 0x04 * i

GLOBAL_MODE_LINEAR = 0x00000001
REG_MODE_STRIPED   = 0x00000002

# VRAM layout: PGT is at 0x130000 (1MB), so compute buffers start at 0x240000
KERNEL_ADDR   = VRAM_BASE + 0x140000  # 0x240000: kernel SASS code
INPUT_A_ADDR  = VRAM_BASE + 0x150000  # 0x250000: input array a
INPUT_B_ADDR  = VRAM_BASE + 0x160000  # 0x260000: input array b
OUTPUT_ADDR   = VRAM_BASE + 0x170000  # 0x270000: output array
STACK_ADDR    = VRAM_BASE + 0x180000  # 0x280000: stack/TLS
CTXVALS_ADDR  = VRAM_BASE + 0x190000  # 0x290000: GR context (ctxvals)

# VRAM DMA object placed in instance block after RAMHT (0x5500 + 0x8000 = 0xD500)
INST_VRAM_DMA_OFF = 0xD500

# ============================================================================
# RAMHT hash (ramht.c:22-30, chid=0 for per-channel RAMHT)
# ============================================================================
def ramht_hash(handle, bits=12):
    h = 0
    while handle:
        h ^= handle & ((1 << bits) - 1)
        handle >>= bits
    return h

def ramht_insert(vram, inst_addr, handle, context):
    idx = ramht_hash(handle)
    off = INST_RAMHT_OFF + idx * 8
    vram.write32(inst_addr + off, handle)
    vram.write32(inst_addr + off + 4, context)
    print(f'[ramht] handle=0x{handle:08x} idx=0x{idx:03x} ctx=0x{context:08x}')

# ============================================================================
# Extend GMMU mapping for compute buffers
# ============================================================================
def extend_gmmu(vram, inst_addr, vaddr_start, vaddr_end):
    """Add identity-mapped PTEs for vaddr_start to vaddr_end (4KB pages).
    Also zeros PTEs in the range first (setup_gmmu only zeros first 512 entries).
    """
    for vaddr in range(vaddr_start, vaddr_end, 0x1000):
        pgt_idx = (vaddr >> 12) & 0x1FFFF
        pte = vaddr | 0x1  # identity mapping: virtual = physical
        vram.write32(PGT_ADDR + pgt_idx * 8, pte & 0xFFFFFFFF)
        vram.write32(PGT_ADDR + pgt_idx * 8 + 4, (pte >> 32) & 0xFF)
    # Flush GMMU TLB
    for eng_id in [0x00, 0x01, 0x06, 0x08, 0x09, 0x0a, 0x0d]:
        vram.dev.write32(0x100c80, (eng_id << 16) | 1)
        for _ in range(2000):
            if not (vram.dev.read32(0x100c80) & 0x00000001):
                break
            time.sleep(0.001)
    print(f'[gmmu] Extended mapping: 0x{vaddr_start:06x}-0x{vaddr_end:06x}')

# ============================================================================
# Compute setup + launch push buffer
# ============================================================================
def build_compute_push(vram_dma_handle, kernel_addr, stack_addr,
                       input_a, input_b, output, n_elements,
                       block_dim=1, grid_dim=1):
    """Build push buffer for NV50_COMPUTE setup + launch.

    Follows mesa/src/gallium/drivers/nouveau/nv50/nv50_compute.c:
    - nv50_screen_compute_setup (lines 66-166): one-time init
    - nv50_launch_grid_with_input (lines 562-633): per-launch
    """
    w = []
    # SET_OBJECT on subchannel 6
    w += nvm50(SUBC_CP, CP_SET_OBJECT, COMPUTE_HANDLE)

    # One-time compute setup (mesa nv50_screen_compute_setup)
    w += nvm50(SUBC_CP, CP_WORK_DISTRIBUTION, 1)
    w += nvm50(SUBC_CP, CP_DMA_STACK, vram_dma_handle)
    w += nvm50(SUBC_CP, CP_STACK_ADDRESS_HIGH,
               (stack_addr >> 32) & 0xFF, stack_addr & 0xFFFFFFFF)
    w += nvm50(SUBC_CP, CP_STACK_SIZE_LOG, 4)
    w += nvm50(SUBC_CP, CP_SHADER_SCHEDULING, 1)
    w += nvm50(SUBC_CP, CP_LANES32_ENABLE, 1)
    w += nvm50(SUBC_CP, CP_REG_MODE, REG_MODE_STRIPED)
    w += nvm50(SUBC_CP, CP_UNK0384, 0x100)
    w += nvm50(SUBC_CP, CP_DMA_GLOBAL, vram_dma_handle)

    # 16 global memory slots: all full-range (kernel uses global14 for GLD/GST)
    for i in range(16):
        w += nvm50(SUBC_CP, CP_GLOBAL_ADDR_HI(i), 0, 0)
        w += nvm50(SUBC_CP, CP_GLOBAL_LIMIT(i), 0xFFFFFFFF)
        w += nvm50(SUBC_CP, CP_GLOBAL_MODE(i), GLOBAL_MODE_LINEAR)

    w += nvm50(SUBC_CP, CP_LOCAL_WARPS_LOG_ALLOC, 7)
    w += nvm50(SUBC_CP, CP_LOCAL_WARPS_NO_CLAMP, 1)
    w += nvm50(SUBC_CP, CP_STACK_WARPS_LOG_ALLOC, 7)
    w += nvm50(SUBC_CP, CP_STACK_WARPS_NO_CLAMP, 1)
    w += nvm50(SUBC_CP, CP_USER_PARAM_COUNT, 0)
    w += nvm50(SUBC_CP, CP_DMA_CODE_CB, vram_dma_handle)
    w += nvm50(SUBC_CP, CP_DMA_LOCAL, vram_dma_handle)
    w += nvm50(SUBC_CP, CP_LOCAL_ADDRESS_HIGH,
               (stack_addr >> 32) & 0xFF, stack_addr & 0xFFFFFFFF)
    w += nvm50(SUBC_CP, CP_LOCAL_SIZE_LOG, 4)

    # Per-launch setup (mesa nv50_launch_grid_with_input)
    w += nvm50(SUBC_CP, CP_CODE_CB_FLUSH, 0)
    w += nvm50(SUBC_CP, CP_CP_START_ID, kernel_addr & 0xFFFFFFFF)

    # Shared memory: align(smem + params + 0x14, 0x40)
    shared_size = (0x14 + 0x10 + 0x3F) & ~0x3F  # base + 4 params * 4 bytes
    w += nvm50(SUBC_CP, CP_SHARED_SIZE, shared_size)
    w += nvm50(SUBC_CP, CP_CP_REG_ALLOC_TEMP, 4)  # from cubin: reg=4
    w += nvm50(SUBC_CP, CP_BLOCKDIM_XY, (1 << 16) | block_dim)
    w += nvm50(SUBC_CP, CP_BLOCKDIM_Z, 1)
    w += nvm50(SUBC_CP, CP_BLOCK_ALLOC, (1 << 16) | block_dim)
    w += nvm50(SUBC_CP, CP_BLOCKDIM_LATCH, 1)
    w += nvm50(SUBC_CP, CP_GRIDDIM, (1 << 16) | grid_dim)
    w += nvm50(SUBC_CP, CP_GRIDID, 1)

    # User parameters: kernel expects a@0x0, b@0x4, out@0x8, N@0xc in shared mem.
    # USER_PARAM(i) writes to shared offset i*4, so params go to USER_PARAM(0-3).
    # No Z-counter needed for 1D launch (grid_z=1).
    param_count = 4 << 8
    w += nvm50(SUBC_CP, CP_USER_PARAM_COUNT, param_count)
    w += nvm50(SUBC_CP, CP_USER_PARAM(0), input_a)
    w += nvm50(SUBC_CP, CP_USER_PARAM(1), input_b)
    w += nvm50(SUBC_CP, CP_USER_PARAM(2), output)
    w += nvm50(SUBC_CP, CP_USER_PARAM(3), n_elements)

    # Launch + wait for idle
    w += nvm50(SUBC_CP, CP_LAUNCH, 0)
    w += nvm50(SUBC_CP, CP_WAIT_FOR_IDLE, 0)
    return w

# ============================================================================
# Channel setup callback (called by create_channel after RAMFC, before bind)
# ============================================================================
def make_setup_fn(ctxvals, ctxvals_size):
    def setup_fn(vram, inst_addr):
        # Extend GMMU for compute buffers (0x240000 - 0x2A0000)
        extend_gmmu(vram, inst_addr, 0x240000, 0x2A0000)

        # Write VRAM DMA object at inst+0xD500
        vram_dma = make_dma_object(0, 0xFFFFFFFFFF, target='vram', access='rw')
        vram.write_block(inst_addr + INST_VRAM_DMA_OFF, vram_dma)
        print(f'[chan] VRAM DMA at inst+0x{INST_VRAM_DMA_OFF:x}')

        # RAMHT entries
        # VRAM DMA: context = (DMA_ENGN_ID << 20) | (dma_offset >> 4)
        ramht_insert(vram, inst_addr, VRAM_DMA_HANDLE,
                     (DMA_ENGN_ID << 20) | (INST_VRAM_DMA_OFF >> 4))
        # Compute object: context = (GR_ENGN_ID << 20) | (gpuobj_off >> 4)
        # nv50_gr_object_bind creates a 16-byte GPU object with class ID at +0x00.
        # We place it at inst+0x40 (within eng block, VP slot is unused).
        GPUOBJ_OFF = 0x40
        vram.write32(inst_addr + GPUOBJ_OFF, NVA3_COMPUTE_CLASS)  # class ID
        vram.write32(inst_addr + GPUOBJ_OFF + 0x04, 0)
        vram.write32(inst_addr + GPUOBJ_OFF + 0x08, 0)
        vram.write32(inst_addr + GPUOBJ_OFF + 0x0c, 0)
        ramht_insert(vram, inst_addr, COMPUTE_HANDLE,
                     (GR_ENGN_ID << 20) | (GPUOBJ_OFF >> 4))

        # Bind GR context (ctxvals buffer in VRAM)
        gr_bind_context(vram.dev, vram, CHAN_ID, inst_addr,
                        CTXVALS_ADDR, ctxvals_size)

        # Upload ctxvals to VRAM
        print(f'[gr-ctx] Uploading ctxvals ({ctxvals_size} bytes) to 0x{CTXVALS_ADDR:08x}...')
        vram.write_block(CTXVALS_ADDR, ctxvals)

    return setup_fn

# ============================================================================
# Main
# ============================================================================
def run_add():
    dev = Dev()
    try:
        # Enable MSE
        cmd_reg = dev.cfg(0x04, 2)
        if not (cmd_reg & 0x0002):
            dev._rpc(4, 0, 0x04, 2, payload=struct.pack('<H', cmd_reg | 0x0002))
        dev.map_bar(0)
        dev.write32(0x000200, 0xffffffff)
        print(f'PMC_ENABLE = 0x{dev.read32(0x000200):08x}')

        # VBIOS devinit
        rom_path = os.path.join(os.path.dirname(__file__), 'en210.rom')
        with open(rom_path, 'rb') as f:
            image = f.read()
        scripts = find_vbios_scripts(image)
        for s in scripts:
            run_vbios_init(dev, image, scripts=[s], debug=False)
        print('Devinit complete.')

        # Verify VRAM
        vram = VRAM(dev)
        vram.write32(0x100000, 0xDEADBEEF)
        v = vram.read32(0x100000)
        print(f'VRAM test: {"PASS" if v == 0xDEADBEEF else "FAIL"} (0x{v:08x})')

        # FIFO init
        fifo_init(dev)

        # Load ctxprog + ctxvals
        ctxprog_globals = {}
        with open(os.path.join(os.path.dirname(__file__), 'ctxprog.py')) as f:
            exec(f.read(), ctxprog_globals)
        ctxprog = ctxprog_globals['ctxprog']
        ctxvals_size = ctxprog_globals['ctxvals_size']
        with open(os.path.join(os.path.dirname(__file__), 'ctxvals.bin'), 'rb') as f:
            ctxvals = f.read()
        print(f'[gr] ctxprog: {len(ctxprog)} instrs, ctxvals: {len(ctxvals)} bytes')

        # GR init
        gr_init(dev, ctxprog=ctxprog, ctxvals=ctxvals, ctxvals_size=ctxvals_size)

        # Upload kernel SASS to VRAM
        sass = get_kernel_sass()
        print(f'[kernel] Uploading {len(sass)} bytes to 0x{KERNEL_ADDR:08x}...')
        vram.write_block(KERNEL_ADDR, sass)
        verify = vram.read_block(KERNEL_ADDR, len(sass))
        print(f'[kernel] Upload {"verified" if verify == sass else "FAILED"}')

        # Prepare input data: a[i]=i, b[i]=i*10, expected out[i]=i*11
        N = int(os.environ.get("EN210_N", "4"))
        block_dim = int(os.environ.get("EN210_BLOCK", str(N)))
        grid_dim = (N + block_dim - 1) // block_dim
        a_bytes = struct.pack(f'{N}f', *[float(i) for i in range(N)])
        b_bytes = struct.pack(f'{N}f', *[float(i * 10) for i in range(N)])
        expected = [float(i) + float(i * 10) for i in range(N)]
        print(f'[data] N={N}, block={block_dim}, grid={grid_dim}')
        print(f'[data] expected out = {expected[:8]}{"..." if N > 8 else ""}')

        vram.write_block(INPUT_A_ADDR, a_bytes)
        vram.write_block(INPUT_B_ADDR, b_bytes)
        for off in range(0, N * 4, 4):
            vram.write32(OUTPUT_ADDR + off, 0)
        print(f'[data] a@0x{INPUT_A_ADDR:x}, b@0x{INPUT_B_ADDR:x}, out@0x{OUTPUT_ADDR:x}')

        # Build compute push buffer
        push = build_compute_push(
            vram_dma_handle=VRAM_DMA_HANDLE,
            kernel_addr=KERNEL_ADDR,
            stack_addr=STACK_ADDR,
            input_a=INPUT_A_ADDR,
            input_b=INPUT_B_ADDR,
            output=OUTPUT_ADDR,
            n_elements=N,
            block_dim=block_dim,
            grid_dim=grid_dim,
        )
        print(f'[compute] Push buffer: {len(push)} words')

        # Create channel and launch
        print('\n=== Compute Launch ===')
        setup_fn = make_setup_fn(ctxvals, ctxvals_size)
        t_kern0 = time.time()
        create_channel(dev, vram, push_words=push, setup_fn=setup_fn)
        t_kern1 = time.time()
        kernel_ms = (t_kern1 - t_kern0) * 1000.0
        print(f'[en210] kernel_time_ms={kernel_ms:.3f} N={N} block={block_dim} grid={grid_dim}')

        # Read back results
        print('\n=== Result Readback ===')
        time.sleep(0.5)

        # Check GR/FIFO status — print trap diagnostics only on error
        gr_intr = dev.read32(0x400100)
        gr_trap = dev.read32(0x400108)
        fifo_intr = dev.read32(0x002100)
        if gr_intr or gr_trap or fifo_intr:
            print(f'[gr] INTR=0x{gr_intr:08x} TRAP=0x{gr_trap:08x}')
            print(f'[fifo] INTR=0x{fifo_intr:08x}')
            units = dev.read32(0x001540)
            if gr_trap & 0x080:  # TRAP_MP
                for tpid in range(16):
                    if not (units & (1 << tpid)):
                        continue
                    tp_ustatus = dev.read32(0x40831c + (tpid << 11)) & 0x7fffffff
                    if tp_ustatus:
                        print(f'[gr] TP{tpid} ustatus=0x{tp_ustatus:08x}')
                    for mp in range(4):
                        if not (units & (1 << (mp + 24))):
                            continue
                        addr = 0x408100 + (tpid << 11) + (mp << 7)
                        status = dev.read32(addr + 0x14)
                        if status:
                            pc = dev.read32(addr + 0x24)
                            oplow = dev.read32(addr + 0x70)
                            ophigh = dev.read32(addr + 0x74)
                            print(f'[gr] MP_TRAP TP{tpid} MP{mp}: status=0x{status:08x} '
                                  f'pc=0x{pc:06x} op=0x{ophigh:08x}{oplow:08x}')
            if gr_trap & 0x100:  # TRAP_PROP
                for tpid in range(16):
                    if not (units & (1 << tpid)):
                        continue
                    ustatus = dev.read32(0x408e08 + (tpid << 11)) & 0x7fffffff
                    if ustatus:
                        fault_hi = dev.read32(0x408e08 + (tpid << 11) + 0x08)
                        fault_lo = dev.read32(0x408e08 + (tpid << 11) + 0x0c)
                        print(f'[gr] PROP_TRAP TP{tpid}: ustatus=0x{ustatus:08x} '
                              f'fault_addr=0x{fault_hi:02x}{fault_lo:08x}')
            if gr_trap & 0x001:  # DISPATCH
                addr = dev.read32(0x400808)
                class_id = dev.read32(0x400814)
                print(f'[gr] DISPATCH_TRAP: addr=0x{addr:08x} class=0x{class_id:04x}')

        results = []
        for i in range(N):
            raw = vram.read32(OUTPUT_ADDR + i * 4)
            val = struct.unpack('<f', struct.pack('<I', raw))[0]
            results.append(val)
            ok = "PASS" if val == expected[i] else "FAIL"
            print(f'  out[{i}] = {val} (expected {expected[i]})  {ok}')

        mismatches = sum(1 for r, e in zip(results, expected) if r != e)
        print(f'\n=== Summary: N={N} mismatches={mismatches}/{N} ===')
        if mismatches == 0:
            print(f'hardware_demo=ok N={N} block={block_dim} grid={grid_dim}')
        else:
            print(f'hardware_demo=FAIL N={N} mismatches={mismatches}/{N}')

    finally:
        dev.close()

def main():
    if "--probe" in sys.argv:
        dev = Dev()
        try:
            ven_dev = dev.cfg(0x00, 4)
            print(f"PCI_ID={ven_dev:#010x}")
            dev.map_bar(0)
            boot0 = dev.read32(0x000000)
            print(f"PMC_BOOT_0={boot0:#010x} chip_id={(boot0>>20)&0xFFF:#05x}")
        finally:
            dev.close()
        return
    run_add()

if __name__ == "__main__":
    main()

