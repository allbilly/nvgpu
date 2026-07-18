#!/usr/bin/env python3
"""Minimal but complete NVINIT / devinit script executor for GK104 VBIOS.

This is a straight port of the VBIOS parser in
`nvkm/subdev/bios/init.c` from the Nouveau driver.  The original
`examples_kepler/add.py` `_vbios_target_ops` is only a partial decoder and
misaligns as soon as it hits opcodes it does not recognise (e.g.
RAM_RESTRICT_ZM_REG_GROUP 0x8f), so it never actually executes the VBIOS
devinit scripts.  This module replaces it.
"""
from __future__ import annotations
import os, struct, time

# GK104 MEMX opcodes (ref/linux/.../pmu/fuc/os.h).
_MEMX_ENTER = 1
_MEMX_LEAVE = 2
_MEMX_WR32 = 3
_MEMX_WAIT = 4
_MEMX_DELAY = 5


class NvbiosI2CError(RuntimeError):
  """Protocol/bus absence error with the errno Nouveau returns to NVINIT."""
  def __init__(self, message: str, errno: int = 5):
    super().__init__(message)
    self.errno = int(errno)

  @property
  def u8(self) -> int:
    return (-self.errno) & 0xff


class _MemxBus:
  """Optional write buffer that flushes controller stores through PMU MEMX WR32.

  Nouveau's gk104_ram_calc runs as a MEMX script on the PMU.  Host BAR0 stores
  of the same sequence leave PRAMIN in the ``0xbad0fb`` stub on this eGPU.
  MEMX WR32/DELAY/WAIT EXEC work without ENTER; buffer only between pause/unpause.
  """

  def __init__(self, real):
    self.real = real
    self.memx_on = False
    self.pending: list[tuple[int, int]] = []
    self.can_memx = (
        os.environ.get("KEPLER_RAM_MEMX_WR", "1") != "0" and
        hasattr(real, "pmu_memx_exec_commands") and
        getattr(real, "_pmu_memx_data", None) is not None)
    self._wr32_flushes = 0
    self._wr32_words = 0
    self._require_memx = False
    self.atomic = os.environ.get("KEPLER_RAM_MEMX_ATOMIC", "0") == "1"
    self.commands: list[tuple[int, tuple[int, ...]]] = []
    self.shadow: dict[int, int] = {}

  def read32(self, reg: int) -> int:
    reg &= 0xffffffff
    if self.atomic and self.memx_on:
      # Nouveau snapshots register values while constructing one MEMX script.
      # Preserve queued writes in a software shadow instead of flushing the
      # transition before its ENTER/LEAVE critical section can surround it.
      if reg in self.shadow:
        return self.shadow[reg]
      value = self.real.read32(reg) & 0xffffffff
      self.shadow[reg] = value
      return value
    self.flush()
    return self.real.read32(reg) & 0xffffffff

  def write32(self, reg: int, val: int) -> None:
    val &= 0xffffffff
    reg &= 0xffffffff
    # Hard refuse (host *or* MEMX): TinyGPU dies if 0x1620/0x26f0 are stored
    # from either path (proven: MEMX WR32 of 0x1620 times out and drops the
    # link).  Nouveau applies these only inside MEMX ENTER, which we cannot use.
    if reg in (0x001620, 0x0026f0):
      print(f"[nvbios] refusing any write to {reg:#x} (TinyGPU-unsafe)",
            flush=True)
      return
    if self.memx_on:
      self.pending.append((reg, val))
      if self.atomic:
        self.shadow[reg] = val
      if len(self.pending) >= 4:
        self.flush()
      return
    # If we started in MEMX mode and it died, do not silently host-program
    # the GDDR5 controller — that sequence collapses TinyGPU BAR0.
    if self._require_memx:
      raise RuntimeError(
          f"MEMX buffering stopped; refusing host write to {reg:#x}")
    self.real.write32(reg, val)

  def flush(self) -> None:
    if not self.pending:
      return
    pairs = self.pending
    self.pending = []
    # Smaller chunks: cold PMU MEMX EXEC sometimes times out on large batches.
    chunk_n = 4
    i = 0
    while i < len(pairs):
      chunk = pairs[i:i + chunk_n]
      payload: list[int] = []
      for addr, data in chunk:
        payload.extend([addr, data])
      if self.atomic:
        self.commands.append((_MEMX_WR32, tuple(payload)))
        self._wr32_flushes += 1
        self._wr32_words += len(payload)
        i += chunk_n
        continue
      try:
        # Clear stuck PMU data semaphore before each EXEC.
        try:
          if (self.real.read32(0x10a580) & 0xffffffff) != 0:
            self.real.write32(0x10a580, 0)
        except Exception:
          pass
        self.real.pmu_memx_exec_commands(
            [(_MEMX_WR32, tuple(payload))], timeout_s=8.0)
      except (TimeoutError, RuntimeError) as e:
        # Do **not** host-fallback the rest: on TinyGPU, host stores of the
        # GDDR5 controller sequence (and especially 0x1620/0x26f0) can collapse
        # BAR0.  Abort with context so the caller can fail closed.
        self.memx_on = False
        self.can_memx = False
        addrs = [f"{a:#x}" for a, _ in chunk]
        raise RuntimeError(
            f"MEMX WR32 flush failed ({e}); aborting without host fallback "
            f"(chunk_addrs={addrs})") from e
      self._wr32_flushes += 1
      self._wr32_words += len(payload)
      i += chunk_n

  def _enter_wait_allowed(self) -> bool:
    """True when stock MEMX ENTER/LEAVE FB_PAUSE spins are intentional.

    Default atomic path still requires ``_pmu_memx_nowait`` (TinyGPU hang).
    ``KEPLER_RAM_ENTER_WAIT=1`` opts into Nouveau's wait (night40ah).
    """
    return os.environ.get("KEPLER_RAM_ENTER_WAIT", "0") != "0"

  def start_memx_buffer(self, *, preflight: bool = True) -> bool:
    if not self.can_memx:
      return False
    nowait = getattr(self.real, "_pmu_memx_nowait", False)
    if self.atomic and not nowait and not self._enter_wait_allowed():
      raise RuntimeError(
          "atomic MEMX requires patched ENTER/LEAVE firmware "
          "(_pmu_memx_nowait is false) or KEPLER_RAM_ENTER_WAIT=1")
    # Prove EXEC still works (DELAY) before buffering hundreds of WR32s.
    try:
      self.real.pmu_memx_exec_commands([(_MEMX_DELAY, (1000,))], timeout_s=3.0)
    except (TimeoutError, RuntimeError) as e:
      # Never imply host GDDR5 fallback when REQUIRE_MEMX (TinyGPU-hostile).
      if os.environ.get("KEPLER_RAM_REQUIRE_MEMX", "1") != "0":
        print(f"[nvbios] MEMX EXEC smoke failed ({e}); refusing host fallback",
              flush=True)
      else:
        print(f"[nvbios] MEMX EXEC smoke failed ({e}); using host writes",
              flush=True)
      self.can_memx = False
      return False
    if (self.atomic and preflight and
        os.environ.get("KEPLER_RAM_ATOMIC_PREFLIGHT", "1") != "0"):
      # Determine whether framebuffer pause can round-trip at all before a
      # long RAM program obscures the boundary.  ENTER and LEAVE share one
      # EXEC, so host MMIO may disappear transiently but must return before
      # the reply.  A timeout proves the pause mechanism itself is the block.
      # Under stock wait, give the PMU longer than the nowait 5s path.
      _pf_t = 30.0 if (not nowait and self._enter_wait_allowed()) else 5.0
      self.real.pmu_memx_exec_commands(
          [(_MEMX_ENTER, ()), (_MEMX_LEAVE, ())], timeout_s=_pf_t)
      print("[nvbios] atomic MEMX ENTER+LEAVE preflight completed", flush=True)
    self.shadow.clear()
    self.memx_on = True
    self._require_memx = True
    return True

  def stop_memx_buffer(self, *, label: str = "script") -> None:
    self.flush()
    if self.atomic and self.commands:
      words = sum(1 + len(payload) for _opcode, payload in self.commands)
      memx_data = getattr(self.real, "_pmu_memx_data", None)
      capacity = (memx_data[1] if isinstance(memx_data, tuple) else 0)
      if capacity and words * 4 > capacity:
        raise RuntimeError(
            f"atomic MEMX RAM script too large: {words * 4:#x}>{capacity:#x}")
      nowait = getattr(self.real, "_pmu_memx_nowait", False)
      # Stock ENTER waits forever on FB_PAUSE; host must bound the RPC.
      _exec_t = 60.0 if (not nowait and self._enter_wait_allowed()) else 15.0
      self.real.pmu_memx_exec_commands(self.commands, timeout_s=_exec_t)
      print(f"[nvbios] atomic MEMX {label} completed: "
            f"commands={len(self.commands)} bytes={words * 4:#x}", flush=True)
      self.commands.clear()
      self.shadow.clear()
    self.memx_on = False

  def memx_delay_ns(self, nsec: int) -> bool:
    if not self.memx_on or nsec <= 0:
      return False
    self.flush()
    if self.atomic:
      self.commands.append((_MEMX_DELAY, (int(nsec) & 0xffffffff,)))
      return True
    self.real.pmu_memx_exec_commands([(_MEMX_DELAY, (int(nsec) & 0xffffffff,))])
    return True

  def memx_wait(self, addr: int, mask: int, data: int,
                nsec: int = 500000) -> bool:
    """MEMX_WAIT — Nouveau ``ram_wait`` / ``nvkm_memx_wait``."""
    if not self.memx_on:
      return False
    self.flush()
    payload = (addr & 0xffffffff, mask & 0xffffffff,
               data & 0xffffffff, int(nsec) & 0xffffffff)
    if self.atomic:
      self.commands.append((_MEMX_WAIT, payload))
      return True
    self.real.pmu_memx_exec_commands([
        (_MEMX_WAIT, payload)],
        timeout_s=max(3.0, nsec / 1e6 + 1.0))
    return True

  def memx_control(self, opcode: int) -> None:
    if not (self.atomic and self.memx_on):
      raise RuntimeError("atomic MEMX control requested outside atomic buffer")
    self.flush()
    self.commands.append((int(opcode), ()))


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
  def run_script(self, start: int) -> None:
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


def find_vbios_image(data: bytes, device: int = 0x1184) -> bytes:
  """Extract the PCI ROM image for a matching GK104 device from an NVGI dump."""
  for pcir in range(len(data)):
    if data[pcir:pcir + 4] != b"PCIR" or pcir + 0x12 > len(data): continue
    vendor, dev = struct.unpack_from("<HH", data, pcir + 4)
    if vendor != 0x10de or dev != device: continue
    size = struct.unpack_from("<H", data, pcir + 0x10)[0] * 512
    base = data.rfind(b"\x55\xaa", max(0, pcir - 0x1000), pcir + 1)
    if base >= 0 and base + size <= len(data):
      return data[base:base + size]
  raise ValueError("no complete matching PCI ROM image")


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


def run_vbios_ram_init(dev, image: bytes, debug: bool = False) -> None:
  """Run the minimal GK104 GDDR5 bring-up used by Nouveau's ramgk104.c.

  This deliberately ports only the hardware-visible portion needed before
  userspace allocates framebuffer objects: all BIT-P RAMMAP scripts, the mode
  finalisation writes, and the packed M0205/M0209 training tables.  Frequency
  transition code and voltage/GPIO handling remain outside this cold-start
  path.
  """
  init = NvbiosInit(dev, image, debug=debug)
  scripts = init.rammap_scripts()
  if not scripts:
    raise ValueError("GK104 VBIOS BIT-P RAMMAP scripts not found")
  ramcfg = init._ramcfg_index()
  strap_reg = getattr(init, "_ramcfg_strap_reg", None)
  strap_src = ("override" if os.environ.get("KEPLER_RAMCFG_STRAP") not in (None, "")
               else "0x101000")
  print(f"[nvbios] RAMCFG group={ramcfg} via {strap_src}"
        + (f" raw={strap_reg:#x}" if strap_reg is not None else ""),
        flush=True)
  if (strap_src == "0x101000" and strap_reg == 0 and ramcfg == 0 and
      os.environ.get("KEPLER_RAMCFG_ALLOW_ZERO", "0") != "1"):
    # A true zero at 0x101000 is indistinguishable from an unclocked/unread
    # PFB strap on the cold eGPU path, and selects the wrong M0205/M0209
    # training tables for the checked-in Palit ROM (golden uses strap 6).
    raise RuntimeError(
        "GK104 0x101000 strap reads as 0; refusing RAM init.  "
        "Replug for a cold POST, or set KEPLER_RAMCFG_STRAP=6 for the "
        "Palit GTX 770 golden configuration "
        "(KEPLER_RAMCFG_ALLOW_ZERO=1 to override this guard)")
  save = dev.read32(0x10f65c) & 0x000000f0
  selected = save >> 4
  if debug:
    print(f"[nvbios] RAMMAP scripts={len(scripts)} selected={selected}")
  for i, script in enumerate(scripts):
    if i == selected:
      continue
    current = dev.read32(0x10f65c)
    dev.write32(0x10f65c,
                (current & ~0x000000f0) | ((i << 4) & 0x000000f0))
    init.run_script(script)
  current = dev.read32(0x10f65c)
  dev.write32(0x10f65c, (current & ~0x000000f0) | save)
  dev.write32(0x10f584, dev.read32(0x10f584) & ~0x11000000)
  dev.write32(0x10ecc0, 0xffffffff)
  dev.write32(0x10f160, dev.read32(0x10f160) | 0x00000010)

  train = init.ram_training()
  # Nouveau gk104_ram_train_init_0 warns if (mask & 0x03d3) != 0x03d3
  # (ramgk104.c:1338).  Types 00/01/04/06/07/08/09 → bits → 0x03d3.
  train_mask = 0
  for typ in train:
    train_mask |= 1 << typ
  print(f"[nvbios] GDDR5 train types={[f'{t:#x}' for t in sorted(train)]} "
        f"mask={train_mask:#x} (Nouveau wants &0x03d3==0x03d3: "
        f"{'ok' if (train_mask & 0x03d3) == 0x03d3 else 'MISSING'})",
        flush=True)
  for i in range(0x30):
    for j in (0, 4):
      dev.write32(0x10f968 + j, i << 8)
      dev.write32(0x10f920 + j,
                   (train[0x08][i] << 4) | train[0x06][i])
      dev.write32(0x10f918 + j, train[0x00][i])
      dev.write32(0x10f920 + j,
                   0x100 | (train[0x09][i] << 4) | train[0x07][i])
      dev.write32(0x10f918 + j, train[0x01][i])
  for j in (0, 4):
    for i in range(0x100):
      dev.write32(0x10f968 + j, i)
      dev.write32(0x10f900 + j, train[0x04][i])
  if debug:
    print("[nvbios] GK104 GDDR5 RAM init/training complete")


def _gk104_read_mem_clock_khz(dev, crystal_khz: int = 27_000) -> int:
  """Port gk104_clk.read_mem/read_pll/read_div for the former RAM clock."""
  def read_pll(pll: int) -> int:
    ctrl = dev.read32(pll) & 0xffffffff
    coef = dev.read32(pll + 4) & 0xffffffff
    if not (ctrl & 1):
      return 0
    p = (coef >> 16) & 0x3f
    n = (coef >> 8) & 0xff
    m = coef & 0xff
    fn = 0xf000
    if pll in (0x00e800, 0x00e820):
      source = crystal_khz
      p = 1
    elif pll == 0x132000:
      source = read_pll(0x132020)
      p = 2 if (coef & 0x10000000) else 1
    elif pll == 0x132020:
      source = read_div(0, 0x137320, 0x137330)
      fn = (dev.read32(pll + 0x10) >> 16) & 0xffff
    else:
      return 0
    if not source or not m:
      return 0
    if not p:
      p = 1
    source = (source * n) + ((((fn + 4096) & 0xffff) * source) >> 13)
    return source // (m * p)

  def read_vco(dsrc: int) -> int:
    source = dev.read32(dsrc) & 0xffffffff
    return read_pll(0x00e820 if (source & 0x00000100) else 0x00e800)

  def read_div(doff: int, dsrc: int, dctl: int) -> int:
    source = dev.read32(dsrc + doff * 4) & 0xffffffff
    control = dev.read32(dctl + doff * 4) & 0xffffffff
    select = source & 3
    if select == 0:
      return 108_000 if (source & 0x00030000) == 0x00030000 else crystal_khz
    if select == 2:
      return 100_000
    if select != 3:
      return 0
    clock = read_vco(dsrc + doff * 4)
    if control & 0x80000000:
      return (clock * 2) // ((control & 0x3f) + 2)
    return clock

  mode = dev.read32(0x1373f4) & 0xf
  if mode == 1:
    return read_pll(0x132020)
  if mode == 2:
    return read_pll(0x132000)
  return 0


def run_vbios_ram_program(dev, image: bytes, freq_mhz: int = 1000,
                          debug: bool = False, *,
                          _cfg_override: dict | None = None,
                          _allow_xition: bool = True) -> dict:
  """Program the cold GDDR5 controller phase used by GK104 reclocking.

  A card which was never POSTed has neither the EFI memory-controller state
  nor Nouveau's first ``ram->func->calc/prog`` transition.  The RAMMAP
  scripts alone therefore leave BAR1 writes nondeterministic.  This is the
  bounded mode-1 form of ``gk104_ram_calc_gddr5()``: it uses the ROM's
  RAMCFG/timing entries, including Nouveau's active former→xition→target loop,
  initializes the controller/MR/timing ports, and leaves the separate
  RAMMAP/training pass to ``run_vbios_ram_init``.

  Board DCB GPIO selection and the ROM-bounded fractional reference PLL are
  included.  Dynamic voltage tables and the high-clock/mode-2 PLL path remain
  deliberately out of scope until the known-good 648-MHz cold path is
  validated.
  """
  # Falcon TIMER_LOW and the MEMX comments define this raw value in ns.
  # Night40aq showed that a 50 ms tight MEMX_WAIT can wedge the PMU's repeated
  # MMIO-read loop instead of returning at its deadline.  Bound diagnostics
  # before any RAM script executes; Nouveau's stock value is 500 us.
  train_wait_ns = int(os.environ.get("KEPLER_TRAIN_WAIT_NS", "500000"), 0)
  if not 1 <= train_wait_ns <= 5_000_000:
    raise ValueError(
        "KEPLER_TRAIN_WAIT_NS must be 1..5000000 nanoseconds per partition; "
        f"got {train_wait_ns}")

  init = NvbiosInit(dev, image, debug=debug)
  target_cfg = init.rammap_config(int(freq_mhz))
  cfg = target_cfg if _cfg_override is None else _cfg_override
  b = cfg

  # gk104_ram_calc() can require two calc/prog iterations.  For an upward
  # transition, Nouveau starts xition from the former RAMCFG and copies these
  # three target fields into it.  If that BIOS image differs from former, the
  # clock core executes xition first and loops once for the final target.
  if _allow_xition and _cfg_override is None:
    pll = init.refpll_limits()
    crystal_khz = int(pll["refclk"]) if pll is not None else 27_000
    former_khz = _gk104_read_mem_clock_khz(dev, crystal_khz)
    target_khz = int(freq_mhz) * 1000
    if 0 < former_khz < target_khz:
      former_mhz = former_khz // 1000
      former_cfg = init.rammap_config(former_mhz)
      xition = dict(former_cfg)
      copied = ("ramcfg_11_02_04", "ramcfg_11_02_03", "timing_20_30_07")
      for field in copied:
        xition[field] = target_cfg[field]
      changed = [field for field in copied
                 if xition[field] != former_cfg[field]]
      if changed:
        print(f"[nvbios] Nouveau xition pass former={former_khz}kHz "
              f"rammap={former_cfg['rammap_index']} target={target_khz}kHz "
              f"rammap={target_cfg['rammap_index']} copied={changed}",
              flush=True)
        run_vbios_ram_program(
            dev, image, freq_mhz=former_mhz, debug=debug,
            _cfg_override=xition, _allow_xition=False)

  # TinyGPU escape hatch: night5 cleared only 0x1620[0] (not 0x26f0, not
  # unpause) and saw PRAMIN 0xbad0fb → 0xffffffff.  Full pause+unpause of both
  # regs on true cold has killed BAR0 (2026-07-16).  Step carefully.
  if os.environ.get("KEPLER_RAM_PROGRAM", "0") == "bit0-only":
    def _boot0() -> int:
      return dev.read32(0) & 0xffffffff
    def _alive(label: str) -> None:
      b0 = _boot0()
      if b0 == 0xffffffff:
        raise RuntimeError(f"bit0-only killed BAR0 at {label}")
    print("[nvbios] KEPLER_RAM_PROGRAM=bit0-only: 0x1620[0] clear only "
          "(no 0x26f0, no unpause)", flush=True)
    _alive("entry")
    r1620 = dev.read32(0x001620) & 0xffffffff
    print(f"[nvbios] 0x1620 before={r1620:#x}", flush=True)
    # Clear bit0 only; preserve 0xaa2 and everything else.
    dev.write32(0x001620, r1620 & ~0x1)
    _alive("after 0x1620 bit0 clear")
    r1620b = dev.read32(0x001620) & 0xffffffff
    print(f"[nvbios] 0x1620 after={r1620b:#x} PMC_BOOT_0={_boot0():#x}",
          flush=True)
    # Optional: continue into MEMX train now that FB is paused at bit0.
    if os.environ.get("KEPLER_RAM_AFTER_BIT0", "0") == "memx":
      print("[nvbios] KEPLER_RAM_AFTER_BIT0=memx: continuing GDDR5 MEMX train",
            flush=True)
    else:
      return cfg

  # Shadow `dev` with an optional MEMX WR32 buffer.
  # TinyGPU: host prog0 then first MEMX WR32 has timed out (0x10f200); all-MEMX
  # historically reached ~78 execs.  bit0 *before* MEMX also kills PMU acquire
  # (0x10a580=0xffffffff) — only post-MEMX bit0 is safe.
  # Optional: KEPLER_RAM_HOST_PROG0=1 restores Nouveau host-then-MEMX order.
  _real_dev = dev
  dev = _MemxBus(_real_dev)
  _host_prog0 = os.environ.get("KEPLER_RAM_HOST_PROG0", "0") == "1"
  if not _host_prog0:
    if not dev.start_memx_buffer():
      if os.environ.get("KEPLER_RAM_REQUIRE_MEMX", "1") != "0":
        raise RuntimeError(
            "PMU MEMX unavailable; refusing host GDDR5 "
            "(power-cycle eGPU and retry)")
    else:
      print("[nvbios] GDDR5 ram_program writes buffered via MEMX WR32 "
            "(all-MEMX; no host 0x10f* pre-store)", flush=True)
  elif (not dev.can_memx and
        os.environ.get("KEPLER_RAM_REQUIRE_MEMX", "1") != "0"):
    raise RuntimeError(
        "PMU MEMX unavailable; refusing host GDDR5 and bit0 pause on a "
        "possibly-wedged GPU.  Power-cycle the eGPU enclosure, restart "
        "TinyGPU server, then cold-run add.py once (no probe).")

  def mask(reg: int, m: int, value: int, *, force: bool = False) -> int:
    """Queue one ``ramfuc_mask`` write only when Nouveau would queue it."""
    old = dev.read32(reg)
    # ramfuc_mask() does not clip data to mask, and suppresses an unchanged
    # store unless ramfuc_nuke() set force on that register.
    new = ((old & ~m) | value) & 0xffffffff
    if force or old != new:
      dev.write32(reg, new)
    return old

  def nvkm_mask(reg: int, m: int, value: int) -> int:
    """Host ``nvkm_mask`` used by prog0: always issue its MMIO write."""
    old = dev.read32(reg)
    dev.write32(reg, ((old & ~m) | value) & 0xffffffff)
    return old

  def ramfuc_mask(reg: int, m: int, value: int) -> int:
    """Apply Nouveau ``ramfuc_mask`` semantics for its special raw fields.

    Unlike ``nvkm_mask``, ramfuc deliberately ORs ``value`` without clipping
    it to ``m``.  The reference-PLL setup uses mask=0 to set control bits, so
    those calls cannot be represented by the ordinary host mask helper.
    """
    return mask(reg, m, value)

  def delay_ns(nsec: int) -> None:
    """MEMX_DELAY while buffering; otherwise host sleep."""
    if nsec <= 0:
      return
    if dev.memx_delay_ns(nsec):
      return
    time.sleep(nsec / 1_000_000_000)

  def _direct_ram_block() -> None:
    nvkm_mask(0x001620, 0x00000aa2, 0)
    nvkm_mask(0x001620, 0x00000001, 0)
    nvkm_mask(0x0026f0, 0x00000001, 0)

  def _direct_ram_unblock() -> None:
    nvkm_mask(0x0026f0, 0x00000001, 0x00000001)
    nvkm_mask(0x001620, 0x00000001, 0x00000001)
    nvkm_mask(0x001620, 0x00000aa2, 0x00000aa2)

  def ram_block() -> None:
    """Begin the GDDR5 reprogram window.

    Nouveau queues MEMX ENTER (falcon writes 0x1620/0x26f0, waits FB_PAUSE).
    On TinyGPU:
      * host / MEMX-WR32 of 0x1620/0x26f0 kill BAR0 — never do that.
      * stock ENTER hangs on FB_PAUSE — requires PMU wait patch
        (``KEPLER_PMU_ENTER_NOWAIT``).
    Modes:
      ``0`` (default): skip pause; MEMX WR32 other controller regs only.
      ``bit0``: do **not** pause mid-MEMX (that drops TinyGPU BAR0 once the
        PMU is active).  After MEMX WR32 finishes, host-clear only
        ``0x1620[0]`` / ``0x26f0[0]`` then restore — night5 unstub on clean
        cold without touching ``0xaa2``.
      ``enter``: real MEMX ENTER after the wait patch (falcon ``0x1620`` stores
        still TinyGPU-hostile).
      ``direct``: full Nouveau host ``0x1620``/``0x26f0`` masks (kills TinyGPU).
    """
    block_mode = os.environ.get("KEPLER_RAM_BLOCK", "0")
    if block_mode == "atomic":
      if not getattr(dev, "atomic", False):
        raise RuntimeError(
            "KEPLER_RAM_BLOCK=atomic requires KEPLER_RAM_MEMX_ATOMIC=1")
      dev.memx_control(_MEMX_ENTER)
      print("[nvbios] queued atomic MEMX ENTER around RAM transition",
            flush=True)
      return
    if block_mode == "bit0":
      # Mid-sequence pause while MEMX is live kills BAR0 on this eGPU.
      print("[nvbios] deferring bit0 pause until after MEMX WR32", flush=True)
      return
    if block_mode == "direct":
      print("[nvbios] WARNING: KEPLER_RAM_BLOCK=direct — host 0x1620 pause "
            "can kill TinyGPU BAR0", flush=True)
      def _host_mask(reg: int, m: int, val: int) -> None:
        old = _real_dev.read32(reg) & 0xffffffff
        _real_dev.write32(reg, ((old & ~m) | val) & 0xffffffff)
      _host_mask(0x001620, 0x00000aa2, 0)
      _host_mask(0x001620, 0x00000001, 0)
      _host_mask(0x0026f0, 0x00000001, 0)
      return
    if block_mode == "enter":
      pmu_block = getattr(_real_dev, "pmu_memx_block", None)
      if (pmu_block is None or
          getattr(_real_dev, "_pmu_memx_data", None) is None or
          not getattr(_real_dev, "_pmu_memx_nowait", False)):
        print("[nvbios] KEPLER_RAM_BLOCK=enter unavailable "
              "(need PMU MEMX + ENTER_NOWAIT); skipping pause", flush=True)
        return
      # Flush any buffered WR32s before ENTER so pause applies first.
      if getattr(dev, "memx_on", False):
        dev.flush()
      pmu_block()
      print("[nvbios] FB pause via patched MEMX ENTER", flush=True)
      return
    if block_mode != "0":
      print(f"[nvbios] ignoring KEPLER_RAM_BLOCK={block_mode!r}; use atomic, "
            "bit0, enter, 0, or direct", flush=True)
    print("[nvbios] skipping FB pause (0x1620 unsafe via host and MEMX WR32)",
          flush=True)

  def ram_unblock() -> None:
    """End pause window; keep MEMX buffering until ram_program returns."""
    block_mode = os.environ.get("KEPLER_RAM_BLOCK", "0")
    if block_mode == "atomic":
      dev.memx_control(_MEMX_LEAVE)
      print("[nvbios] queued atomic MEMX LEAVE after RAM transition",
            flush=True)
      return
    if block_mode == "bit0":
      # Paired with deferred mid-sequence pause; real bit0 runs post-MEMX.
      return
    if block_mode == "enter":
      if getattr(dev, "memx_on", False):
        dev.flush()
      pmu_unblock = getattr(_real_dev, "pmu_memx_unblock", None)
      if pmu_unblock is None:
        return
      try:
        pmu_unblock()
        print("[nvbios] FB unpause via patched MEMX LEAVE", flush=True)
      except (TimeoutError, RuntimeError) as e:
        # Programming may have stuck; leave paused rather than host-unpause.
        print(f"[nvbios] MEMX LEAVE failed ({e}); leaving FB paused",
              flush=True)
      return
    if block_mode != "direct":
      return
    dev.flush()
    def _host_mask(reg: int, m: int, val: int) -> None:
      old = _real_dev.read32(reg) & 0xffffffff
      _real_dev.write32(reg, ((old & ~m) | val) & 0xffffffff)
    _host_mask(0x0026f0, 0x00000001, 0x00000001)
    _host_mask(0x001620, 0x00000001, 0x00000001)
    _host_mask(0x001620, 0x00000aa2, 0x00000aa2)

  def set_gpio_level(function: int, logical: int, delay_nsec: int) -> None:
    """Apply one Nouveau DCB GPIO level and pulse the GPIO trigger if needed."""
    gpio = init.dcb_gpio(function)
    if gpio is None:
      print(f"[nvbios] GPIO func={function:#x}: DCB entry MISSING — skipped",
            flush=True)
      return
    # `logical` selects the DCB log[0]/log[1] entry; the entry itself carries
    # the polarity encoding that Nouveau XORs with 2 before writing GPIO.
    level = gpio["log1"] if int(logical) else gpio["log0"]
    want = (int(level) ^ 2) << 12
    old = mask(gpio["reg"], 0x00003000, want)
    # Nouveau: temp = ram_mask(...); if (temp != ram_rd32(...)) trigger.
    # In atomic MEMX, host read32 still sees pre-buffer HW, so comparing a
    # re-read to ``old`` never sees a change (night40ap).  Compare the bits
    # we just programmed instead — equivalent when the write is visible.
    changed = (old & 0x00003000) != (want & 0x00003000)
    if changed:
      dev.write32(0x00d604, 1)
      delay_ns(delay_nsec)
    if os.environ.get("KEPLER_GPIO_TRACE", "1") != "0":
      print(f"[nvbios] GPIO func={function:#x} line={gpio['line']} "
            f"reg={gpio['reg']:#x} logical={int(logical)} "
            f"log={level} want_bits={want:#x} old={old:#x} "
            f"changed={changed} trig={'yes' if changed else 'no'}",
            flush=True)

  # Nouveau's transition program only masks RAMMAP fields which differ across
  # the ROM's entries.  Compute those flags once; testing the selected value
  # itself would incorrectly skip valid zero fields (or touch invariant ones).
  _diff = init.rammap_diffs()

  # Nouveau mirrors a subset of controller registers into partitions whose
  # 0x110204 configuration differs from the first active partition.  Compute
  # that mask once, so the direct host path can reproduce ram_nuts() without
  # assuming all eight GK104 partitions are identical.
  _parts_hw = dev.read32(0x022438) & 0xff
  _parts = _parts_hw
  _pmask = dev.read32(0x022554)
  if _parts == 0:
    # Nouveau carries ram->parts from FB one-init.  A cold TinyGPU script
    # build can read zero here before that topology register is latched, so
    # retain the same bounded board fallback used by train().
    _parts = int(os.environ.get("KEPLER_RAM_PARTS", "4"), 0)
  _parts = max(0, min(_parts, 16))
  _pnuts = 0
  _first_cfg = None
  _part_cfgs = []
  for _part in range(min(_parts, 16)):
    if _pmask & (1 << _part):
      continue
    _cfg1 = dev.read32(0x110204 + _part * 0x1000)
    _part_cfgs.append((_part, _cfg1 & 0xffffffff))
    if _first_cfg is not None and _first_cfg != _cfg1:
      _pnuts |= 1 << _part
    else:
      _first_cfg = _cfg1
  if os.environ.get("KEPLER_TRAIN_TRACE", "1") != "0":
    print(f"[nvbios] partition topology hw_parts={_parts_hw} parts={_parts} "
          f"pmask={_pmask:#x} pnuts={_pnuts:#x} "
          f"cfg={[f'{part}:{value:#x}' for part, value in _part_cfgs]}",
          flush=True)

  def ram_nuts(reg: int, m: int, value: int, copy: int,
               reg_data: int | None = None) -> None:
    if not _pnuts:
      return
    if reg_data is None:
      reg_data = dev.read32(reg)
    full_mask = (m | copy) & 0xffffffff
    full_data = ((value & m) | (reg_data & copy)) & 0xffffffff
    for _part in range(min(_parts, 16)):
      if not (_pnuts & (1 << _part)):
        continue
      _addr = 0x110000 + (reg & 0xfff) + _part * 0x1000
      _prev = dev.read32(_addr)
      dev.write32(_addr, ((_prev & ~full_mask) | full_data) & 0xffffffff)

  def train(m: int, value: int) -> None:
    mask(0x10f910, m, value)
    mask(0x10f914, m, value)
    if not (value & 0x80000000):
      return
    parts = dev.read32(0x022438) & 0xff
    pmask = dev.read32(0x022554) & 0xffffffff
    # Night40ao (H25): cold script-build can see 0x022438==0 before PFB
    # topology is latched; Nouveau uses ram->parts from fb oneinit, not a
    # mid-script host peek.  Fall back so MEMX_WAIT is actually queued.
    if parts == 0:
      parts = int(os.environ.get("KEPLER_RAM_PARTS", "4"), 0)
      print(f"[nvbios] train: 0x022438 was 0; using parts={parts} "
            f"(KEPLER_RAM_PARTS)", flush=True)
    waits = 0
    for part in range(min(parts, 16)):
      if pmask & (1 << part):
        continue
      addr = 0x110974 + part * 0x1000
      waits += 1
      # Nouveau ram_wait → MEMX_WAIT while buffering; else host poll.
      # Nouveau uses 500,000 ns; a timeout falls through silently (H24).
      if getattr(dev, "memx_wait", None) and dev.memx_wait(
          addr, 0x0000000f, 0x00000000, train_wait_ns):
        continue
      deadline = time.monotonic() + max(0.5, train_wait_ns / 1e9)
      while dev.read32(addr) & 0xf:
        if time.monotonic() >= deadline:
          raise TimeoutError(f"GK104 RAM training timeout on partition {part}")
        time.sleep(0.00001)
    if os.environ.get("KEPLER_TRAIN_TRACE", "1") != "0":
      print(f"[nvbios] train trigger mask={m:#x} data={value:#x} "
            f"parts={parts} pmask={pmask:#x} waits={waits}", flush=True)

  def prog0(cfg_src: dict | None = None) -> None:
    """Port gk104_ram_prog_0() for a RAMMAP entry's frequency tables.

    Nouveau's ``gk104_ram_prog`` calls this twice: once at 1000 MHz *before*
    ``ram_exec``, then again at ``next->freq`` after EXEC returns.  Pass
    ``cfg_src`` to select which RAMMAP entry's ``rammap_11_*`` fields to apply;
    default is the target ``freq_mhz`` config already in ``b``.
    """
    c = b if cfg_src is None else cfg_src
    _m = ((0x001ff000 if _diff["rammap_11_0a_03fe"] else 0) |
          (0x000001ff if _diff["rammap_11_09_01ff"] else 0))
    _v = (((c["rammap_11_0a_03fe"] << 12)
           if _diff["rammap_11_0a_03fe"] else 0) |
          (c["rammap_11_09_01ff"]
           if _diff["rammap_11_09_01ff"] else 0))
    nvkm_mask(0x10f468, _m, _v)
    nvkm_mask(0x10f420,
              0x00000001 if _diff["rammap_11_0a_0400"] else 0,
              c["rammap_11_0a_0400"] if _diff["rammap_11_0a_0400"] else 0)
    nvkm_mask(0x10f430,
              0x00000001 if _diff["rammap_11_0a_0800"] else 0,
              c["rammap_11_0a_0800"] if _diff["rammap_11_0a_0800"] else 0)
    nvkm_mask(0x10f400,
              0x0000001f if _diff["rammap_11_0b_01f0"] else 0,
              c["rammap_11_0b_01f0"] if _diff["rammap_11_0b_01f0"] else 0)
    nvkm_mask(0x10f410,
              0x00000200 if _diff["rammap_11_0b_0200"] else 0,
              (c["rammap_11_0b_0200"] << 9)
              if _diff["rammap_11_0b_0200"] else 0)
    _m = ((0x00ff0000 if _diff["rammap_11_0d"] else 0) |
          (0x0000ff00 if _diff["rammap_11_0f"] else 0))
    _v = (((c["rammap_11_0d"] << 16) if _diff["rammap_11_0d"] else 0) |
          ((c["rammap_11_0f"] << 8) if _diff["rammap_11_0f"] else 0))
    nvkm_mask(0x10f440, _m, _v)
    _m = ((0x0000ff00 if _diff["rammap_11_0e"] else 0) |
          (0x00000080 if _diff["rammap_11_0b_0800"] else 0) |
          (0x00000020 if _diff["rammap_11_0b_0400"] else 0))
    _v = (((c["rammap_11_0e"] << 8) if _diff["rammap_11_0e"] else 0) |
          ((c["rammap_11_0b_0800"] << 7)
           if _diff["rammap_11_0b_0800"] else 0) |
          ((c["rammap_11_0b_0400"] << 5)
           if _diff["rammap_11_0b_0400"] else 0))
    nvkm_mask(0x10f444, _m, _v)

  # nvkm_gddr5_calc(): derive the mode registers from the timing entry.
  timing = b["timing"]
  wl = (timing[1] >> 7) & 0xf
  cl = timing[1] & 0x1f
  wr = (timing[2] >> 16) & 0x7f
  if not (1 <= wl <= 7 and 5 <= cl <= 36 and 4 <= wr <= 35):
    raise ValueError(f"invalid GK104 GDDR5 timing WL={wl} CL={cl} WR={wr}")
  cl -= 5
  wr -= 4
  # Keep the complete controller register image.  nvkm_gddr5_calc() changes
  # only mode-register fields, and gk104_ram_calc_gddr5() preserves all upper
  # bits when issuing MR1/MR3/MR5/MR6/MR7/MR8 writes.
  mr = {i: dev.read32(addr) for i, addr in {
    0: 0x10f300, 1: 0x10f330, 3: 0x10f338, 5: 0x10f340,
    6: 0x10f344, 7: 0x10f348, 8: 0x10f354,
  }.items()}
  mr[0] = (mr[0] & ~0xf7f) | ((wr & 0xf) << 8) | ((cl & 0xf) << 3) | wl
  xd = int(not b["ramcfg_DLLoff"])
  mr[1] = ((mr[1] & ~0x0bf) |
           ((b["timing_20_2e_c0"] & 3) << 4) |
           ((b["timing_20_2e_03"] & 3) << 2) |
           (b["timing_20_2f_03"] & 3) | (xd << 7))
  # nvkm_gddr5_calc() saves the ordinary at[0] image for partition mirrors,
  # then switches the main MR1 to at[1] only when ram->pnuts is nonzero.
  mr1_nuts = mr[1]
  if _pnuts:
    mr[1] = ((mr[1] & ~0x030) |
             ((b["timing_20_2e_30"] & 3) << 4))
  mr[3] = (mr[3] & ~0x020) | (int(freq_mhz < 1000) << 5)
  mr[5] = (mr[5] & ~0x004) | (int(not b["ramcfg_11_07_02"]) << 2)
  vo = b["ramcfg_11_06"] or ((mr[6] >> 4) & 0xff)
  pd = b["ramcfg_11_01_80"] or (mr[6] & 1)
  mr[6] = (mr[6] & ~0xff1) | ((vo & 0xff) << 4) | (pd & 1)
  mr[7] = ((mr[7] & ~0x388) | ((b["ramcfg_11_02_04"] & 3) << 8) |
           ((b["ramcfg_11_02_10"] & 1) << 7) |
           ((b["ramcfg_11_01_40"] & 1) << 3))
  mr[8] = (mr[8] & ~3) | ((wr & 0x10) >> 3) | ((cl & 0x10) >> 4)
  # prog0: default all-MEMX; optional host-then-MEMX for A/B.
  # Nouveau gk104_ram_prog(): prog0(1000) BEFORE ram_exec, prog0(target) after.
  # Night40ai: apply the 1000 MHz RAMMAP tables here (pre-ENTER), not target.
  _pre_mhz = int(os.environ.get("KEPLER_RAM_PROG0_PRE_MHZ", "1000"))
  if _pre_mhz > 0 and _pre_mhz != int(freq_mhz):
    b_pre = init.rammap_config(_pre_mhz)
    print(f"[nvbios] pre-transition prog0 at {_pre_mhz} MHz "
          f"(rammap={b_pre['rammap_index']} vs target "
          f"{b['rammap_index']}@{freq_mhz})", flush=True)
    prog0(b_pre)
  else:
    prog0()

  if not getattr(dev, "memx_on", False):
    if dev.start_memx_buffer():
      print("[nvbios] GDDR5 ram_program writes buffered via MEMX WR32 "
            "(after host prog0; no host 0x1620 pause)", flush=True)
    elif os.environ.get("KEPLER_RAM_REQUIRE_MEMX", "1") != "0":
      raise RuntimeError(
          "PMU MEMX EXEC smoke failed after host prog0; refusing host "
          "GDDR5 transition (power-cycle eGPU and retry)")
  # Controller reset/refresh/precharge and early write leveling.
  # Match Nouveau's gt215_pll_calc(): use the ROM's type-0x0c limits rather
  # than a generic VCO window.  For this ROM that changes the 648-MHz result
  # from the old (P=2,N=48,M=1) guess to Nouveau's integer/fractional
  # coefficients.  The fractional accumulator is required even when the
  # requested output clock is an exact multiple of the reference clock:
  # gt215_pll_calc() is called with pfN non-NULL by gk104_ram_calc_xits().
  pll = init.refpll_limits()
  if pll is None:
    # Keep a conservative fallback for older/nonstandard dumps, but make it
    # explicit and bounded.  The checked-in Palit ROM always takes the path
    # above.
    pll = {
      "refclk": 27000, "min_freq": 600000, "max_freq": 1200000,
      "min_inputfreq": 25000, "max_inputfreq": 75000,
      "min_m": 1, "max_m": 255, "min_n": 8, "max_n": 255,
      "min_p": 1, "max_p": 7,
    }
  target_khz = int(freq_mhz) * 1000
  ref_khz = int(pll["refclk"])
  p1 = pll["max_freq"] // target_khz
  p1 = max(int(pll["min_p"]), min(int(pll["max_p"]), p1))
  l_m = (ref_khz + int(pll["max_inputfreq"])) // int(pll["max_inputfreq"])
  l_m = max(l_m, int(pll["min_m"]))
  h_m = (ref_khz + int(pll["min_inputfreq"])) // int(pll["min_inputfreq"])
  h_m = min(h_m, int(pll["max_m"]))
  l_m = min(l_m, h_m)
  best = None
  for m in range(l_m, h_m + 1):
    tmp = target_khz * p1 * m
    n = tmp // ref_khz
    rem = tmp % ref_khz
    # Match gt215_pll_calc()'s pfN branch.  It chooses the lower integer N
    # when the remainder is below half the reference clock, then represents
    # the residual in the 13-bit fractional accumulator.  The hardware value
    # is biased by 4096 and stored in both 0x132030[31:16] and
    # 0x132034[15:0].
    if rem < ref_khz // 2:
      n -= 1
      rem = tmp - n * ref_khz
    if n < int(pll["min_n"]):
      continue
    if n > int(pll["max_n"]):
      break
    fn = (((rem << 13) + ref_khz // 2) // ref_khz - 4096) & 0xffff
    # With pfN supplied Nouveau returns at the first valid M/N pair rather
    # than comparing integer-only output errors.  Preserve that ordering;
    # the fractional accumulator makes the resulting PLL output exact.
    best = (0, target_khz, p1, n, m, fn)
    break
  if best is None:
    raise ValueError(f"cannot calculate GK104 memory PLL for {freq_mhz} MHz")
  _, actual, p1, n1, m1, fn1 = best

  _pll_from = dev.read32(0x1373f4) & 0x0000000f
  # gk104_ram_calc() first asks gk104_clk_read(nv_clk_src_mem) for the
  # *former* clock.  read_mem() returns zero unless this selector is 1 or 2;
  # in that state a literal Nouveau reclock aborts before building RAMFUC.
  print(f"[nvbios] Nouveau former-memory selector 0x1373f4[3:0]="
        f"{_pll_from:#x} "
        f"({'clock path present' if _pll_from in (1, 2) else 'read_mem=0; literal reclock prerequisite missing'})",
        flush=True)

  def finish_refpll() -> None:
    """Apply the source's r1373f4_fini() sequence."""
    dev.write32(0x1373ec,
                (dev.read32(0x1373ec) & ~0x00030000) |
                (b["ramcfg_11_03_30"] << 16))
    # r1373f4_fini(): (~mode & 3) is only bit 1 in mode 1; preserve bit 0.
    mask(0x1373f0, 0x00000002, 0)
    mask(0x1373f4, 0x00000003, 0x00000001)
    mask(0x1373f4, 0x00010000, 0)
    mask(0x10f800, 0x00000030,
         ((b["ramcfg_11_03_c0"] ^ b["ramcfg_11_03_30"]) & 3) << 4)

  def program_refpll(from_mode: int = _pll_from, finish: bool = True) -> None:
    # r1373f4_init()/r1373f4_fini(), mode 1.  This is deliberately invoked
    # after refresh/precharge setup, matching gk104_ram_calc_gddr5().
    if from_mode == 2:
      # r1373f4_init() has a distinct pre-existing-clock path.  These are
      # ramfuc_mask(mask=0,data=...) operations, not ordinary masked writes.
      ramfuc_mask(0x1373f4, 0x00000000, 0x00001100)
      ramfuc_mask(0x1373f4, 0x00000000, 0x00000010)
    else:
      ramfuc_mask(0x1373f4, 0x00000000, 0x00010010)
    ramfuc_mask(0x1373f4, 0x00000003, 0x00000000)
    ramfuc_mask(0x1373f4, 0x00000010, 0x00000000)
    rcoef = (p1 << 16) | (n1 << 8) | m1
    coef_before = dev.read32(0x132024) & 0xffffffff
    frac_before = dev.read32(0x132034) & 0x0000ffff
    reprogram = coef_before != rcoef or frac_before != fn1
    print(f"[nvbios] REFPLL coefficient current={coef_before:#x}/"
          f"{frac_before:#x} wanted={rcoef:#x}/{fn1:#x} "
          f"reprogram={'yes' if reprogram else 'no'}", flush=True)
    if reprogram:
      mask(0x132000, 0x00000001, 0)
      mask(0x132020, 0x00000001, 0)
      dev.write32(0x137320, 0)
      mask(0x132030, 0xffff0000, fn1 << 16)
      mask(0x132034, 0x0000ffff, fn1)
      dev.write32(0x132024, rcoef)
      mask(0x132028, 0x00080000, 0x00080000)
      mask(0x132020, 0x00000001, 0x00000001)
      # Nouveau's MEMX program waits for REFPLL_LOCK (0x137390[17]) for at
      # most 64 us.  Poll the same condition on the direct host path rather
      # than assuming a fixed delay was sufficient.
      refpll_wait_ns = 64_000
      deadline = time.monotonic() + refpll_wait_ns / 1_000_000_000
      if not (getattr(dev, "memx_wait", None) and dev.memx_wait(
          0x137390, 0x00020000, 0x00020000, refpll_wait_ns)):
        while not (dev.read32(0x137390) & 0x00020000):
          if time.monotonic() >= deadline:
            if os.environ.get("KEPLER_RAM_STRICT_WAIT", "1") != "0":
              raise TimeoutError(
                  "GK104 reference PLL lock timeout (0x137390[17])")
            break
          time.sleep(0.00001)
      mask(0x132028, 0x00080000, 0)
    # r1373f4_init(): mode 1 selects refpll as the memory-clock source and
    # arms the transition before r1373f4_fini commits selector 1.
    ramfuc_mask(0x1373f4, 0x00000000, 0x00010100)
    ramfuc_mask(0x1373f4, 0x00000000, 0x00000010)
    if finish:
      finish_refpll()

  mask(0x10f808, 0x40000000, 0x40000000)
  ram_block()
  # gk104_ram_calc_gddr5() quiesces display memory clients for every card
  # with a display subdevice, then restores them immediately after unblock.
  # A normal GK104 Nouveau device has that subdevice even with no monitor.
  dev.write32(0x62c000, 0x0f0f0000)
  # gk104_ram_calc_gddr5(): early MR1 termination is part of the transition
  # script *after* ENTER.  The old port queued it before prog0/ENTER (H32).
  if (mr[1] & 0x03c) != 0x030:
    mask(0x10f330, 0x0000003c, mr[1] & 0x0000003c)
    ram_nuts(0x10f330, 0x0000003c, mr1_nuts & 0x0000003c,
             0x00000000, reg_data=mr1_nuts)
  # vc == !ramcfg_11_02_08; select the DCB tag-0x2e level before refresh
  # sequencing, matching Nouveau's source ordering.
  if not b["ramcfg_11_02_08"]:
    set_gpio_level(0x2e, 1, 20_000)
  mask(0x10f200, 0x00000800, 0)
  train(0x01020000, 0x000c0000)
  dev.write32(0x10f210, 0)
  delay_ns(1_000)
  dev.write32(0x10f310, 1)
  delay_ns(1_000)
  mask(0x10f200, 0x80000000, 0x80000000)
  dev.write32(0x10f314, 1)
  mask(0x10f200, 0x80000000, 0)
  dev.write32(0x10f090, 0x61)
  dev.write32(0x10f090, 0xc000007f)
  delay_ns(1_000)

  # RAMCFG-dependent controller knobs.  These masks intentionally mirror
  # gk104_ram_calc_gddr5() instead of using a fixed aggregate mask: the
  # strap-6 Palit entry exercises several of the conditional fields.
  dev.write32(0x10f698, 0)
  dev.write32(0x10f69c, 0)
  mask_824 = 0x800F07E0
  data_824 = 0x00030000
  if dev.read32(0x10f978) & 0x00800000:
    data_824 |= 0x00040000
  data_824 |= 0x800807E0
  clear_c0 = {3: 0x00000040, 2: 0x00000100,
              1: 0x80000000, 0: 0x00000400}
  clear_30 = {3: 0x00000020, 2: 0x00000080,
              1: 0x00080000, 0: 0x00000200}
  data_824 &= ~clear_c0[b["ramcfg_11_03_c0"]]
  data_824 &= ~clear_30[b["ramcfg_11_03_30"]]
  if b["ramcfg_11_02_80"]:
    mask_824 |= 0x03000000
  if b["ramcfg_11_02_40"]:
    mask_824 |= 0x00002000
  if b["ramcfg_11_07_10"]:
    mask_824 |= 0x00004000
  if b["ramcfg_11_07_08"]:
    mask_824 |= 0x00000003
  else:
    mask_824 |= 0x34000000
    if dev.read32(0x10f978) & 0x00800000:
      mask_824 |= 0x40000000
  mask(0x10f824, mask_824, data_824)
  mask(0x132040, 0x00010000, 0)
  if _pll_from == 2:
    mask(0x10f808, 0x00080000, 0)
    mask(0x10f200, 0x18008000, 0x00008000)
    ramfuc_mask(0x10f800, 0x00000000, 0x00000004)
    ramfuc_mask(0x10f830, 0x00008000, 0x01040010)
    mask(0x10f830, 0x01000000, 0)
    program_refpll(_pll_from, finish=False)
    # The source performs r1373f4_fini() only after the temporary 1373f0
    # transition, then applies the raw-mask 0x10f830 update.
    ramfuc_mask(0x1373f0, 0x00000002, 0x00000001)
    finish_refpll()
    ramfuc_mask(0x10f830, 0x00c00000, 0x00240001)
  else:
    program_refpll(_pll_from)
  # Memory-voltage GPIO (DCB tag 0x18), matching the source's mv =
  # !ramcfg_11_02_04 selection.  The Palit ROM maps this to GPIO line 7.
  set_gpio_level(0x18, int(not b["ramcfg_11_02_04"]), 64_000)
  if b["ramcfg_11_02_40"] or b["ramcfg_11_07_10"]:
    mask(0x132040, 0x00010000, 0x00010000)
    delay_ns(20_000)
  # gk104_ram_calc_gddr5() raises the controller transition bit before the
  # RAMMAP/controller fields below, then waits for it to settle immediately
  # after the second 0x10f200 update.  Keeping this ordering is important for
  # the memory-controller state machine; do not defer it until PFB timing.
  if b["ramcfg_11_07_40"]:
    mask(0x10f670, 0x80000000, 0x80000000)
  dev.write32(0x10f65c, 0x11 * cfg["rammap_11_11_0c"])
  dev.write32(0x10f6b8, 0x01010101 * b["ramcfg_11_09"])
  dev.write32(0x10f6bc, 0x01010101 * b["ramcfg_11_09"])
  if not b["ramcfg_11_07_08"] and not b["ramcfg_11_07_04"]:
    data_698 = 0x01010101 * b["ramcfg_11_04"]
    dev.write32(0x10f698, data_698)
    dev.write32(0x10f69c, data_698)
  elif not b["ramcfg_11_07_08"]:
    dev.write32(0x10f698, 0)
    dev.write32(0x10f69c, 0)
  # ram_nuke sets force for the following ramfuc_mask without clearing data.
  mask(0x10f694, 0xff00ff00, 0x01000100 * b["ramcfg_11_04"], force=True)
  mask(0x10f60c, 0x00000080, 0)
  mask_824 = 0x00070000
  data_824 = 0
  if not b["ramcfg_11_02_80"]:
    data_824 |= 0x03000000
  if not b["ramcfg_11_02_40"]:
    data_824 |= 0x00002000
  if not b["ramcfg_11_07_10"]:
    data_824 |= 0x00004000
  if not b["ramcfg_11_07_08"]:
    data_824 |= 0x00000003
  else:
    data_824 |= 0x74000000
  mask(0x10f824, mask_824, data_824)
  mask(0x10f200, 0x00001000, 0x00000000 if b["ramcfg_11_01_08"] else 0x00001000)
  if dev.read32(0x10f670) & 0x80000000:
    delay_ns(10_000)
    mask(0x10f670, 0x80000000, 0)
  mask(0x10f82c, 0x00100000, 0x00100000 if b["ramcfg_11_08_01"] else 0)
  mask(0x10f830, 0x00007000,
       (b["ramcfg_11_08_08"] << 13) |
       (b["ramcfg_11_08_04"] << 12) |
       (b["ramcfg_11_08_02"] << 14))
  # PFB timing registers.  Nouveau rotates the VBIOS array: timing[10] is
  # 0x10f248, followed by timing[0..9] at 0x10f290..0x10f2e8.
  timing_values = (timing[10], *timing[:10])
  for reg, value in zip((0x10f248, 0x10f290, 0x10f294, 0x10f298,
                         0x10f29c, 0x10f2a0, 0x10f2a4, 0x10f2a8,
                         0x10f2ac, 0x10f2cc, 0x10f2e8), timing_values):
    dev.write32(reg, value)
  if _diff["ramcfg_11_08_20"]:
    mask(0x10f200, 0x01000000,
         0x01000000 if b["ramcfg_11_08_20"] else 0)
  mask_604 = 0
  data_604 = 0
  # ram->diff is nonzero for a cold card; retain both the low mode field and
  # the 0x70000000 transition field when the selected RAMCFG enables it.
  if _diff["ramcfg_11_02_03"]:
    mask_604 |= 0x00000300
    data_604 |= b["ramcfg_11_02_03"] << 8
  if _diff["ramcfg_11_01_10"]:
    mask_604 |= 0x70000000
    if b["ramcfg_11_01_10"]:
      data_604 |= 0x70000000
  mask(0x10f604, mask_604, data_604)
  mask_614 = 0
  data_614 = 0
  if _diff["timing_20_30_07"]:
    mask_614 |= 0x70000000
    data_614 |= b["timing_20_30_07"] << 28
  if _diff["ramcfg_11_01_01"]:
    mask_614 |= 0x00000100
    if b["ramcfg_11_01_01"]:
      data_614 |= 0x00000100
  mask(0x10f614, mask_614, data_614)
  mask_610 = 0
  data_610 = 0
  if _diff["timing_20_30_07"]:
    mask_610 |= 0x70000000
    data_610 |= b["timing_20_30_07"] << 28
  if _diff["ramcfg_11_01_02"]:
    mask_610 |= 0x00000100
    if b["ramcfg_11_01_02"]:
      data_610 |= 0x00000100
  mask(0x10f610, mask_610, data_610)

  mask_808 = 0x33F00000
  data_808 = 0
  if not b["ramcfg_11_01_04"]:
    data_808 |= 0x20200000
  if not b["ramcfg_11_07_80"]:
    data_808 |= 0x12800000
  if b["ramcfg_11_03_f0"]:
    if cfg["rammap_11_08_0c"]:
      if not b["ramcfg_11_07_80"]:
        mask_808 |= 0x00000020
      else:
        data_808 |= 0x00000020
      # gk104_ram_calc_gddr5() uses mask bit 0x4 here (not the SDR/DDR
      # sibling's 0x08000004).  Keep the GDDR5 encoding exact.
      mask_808 |= 0x00000004
  else:
    mask_808 |= 0x40000020
    data_808 |= 0x00000004
  # Final 0x10f808: Nouveau ram_mask preserves bits outside ``mask_808``.
  # Cold eGPU power-on leaves residual bits (night40ah live ended
  # ``0x7aa00050`` vs offline golden ``0x72a00000``, xor ``0x08000050``).
  # Absolute write matches the computed field image on a zero baseline.
  if os.environ.get("KEPLER_RAM_808_ABSOLUTE", "1") != "0":
    final_808 = (data_808 & mask_808) | 0x40000000
    # Keep any bits Nouveau's earlier 0x40000000 set; drop cold residuals.
    old_808 = dev.read32(0x10f808) & 0xffffffff
    if (old_808 & mask_808) != (final_808 & mask_808) or (
        old_808 & ~mask_808 & ~0x40000000):
      print(f"[nvbios] 0x10f808 absolute {old_808:#x}->{final_808:#x} "
            f"(clear cold residuals outside mask {mask_808:#x})", flush=True)
    dev.write32(0x10f808, final_808)
  else:
    mask(0x10f808, mask_808, data_808)
  dev.write32(0x10f870, 0x11111111 * b["ramcfg_11_03_0f"])
  mask_100770 = ((0x00000003 if _diff["ramcfg_11_02_03"] else 0) |
                 (0x00000004 if _diff["ramcfg_11_01_10"] else 0))
  data_100770 = ((b["ramcfg_11_02_03"]
                  if _diff["ramcfg_11_02_03"] else 0) |
                 (0x00000004
                  if (_diff["ramcfg_11_01_10"] and
                      b["ramcfg_11_01_10"]) else 0))
  old_100770 = mask(0x100770, mask_100770, data_100770)
  if ((old_100770 & mask_100770 & 4) != (data_100770 & 4)):
    mask(0x100750, 0x00000008, 0x00000008)
    dev.write32(0x100710, 0)
    transition_wait_ns = 200_000
    if not (getattr(dev, "memx_wait", None) and dev.memx_wait(
        0x100710, 0x80000000, 0x80000000, transition_wait_ns)):
      deadline = time.monotonic() + transition_wait_ns / 1_000_000_000
      while not (dev.read32(0x100710) & 0x80000000):
        if time.monotonic() >= deadline:
          if os.environ.get("KEPLER_RAM_STRICT_WAIT") == "1":
            raise TimeoutError("GK104 0x100710 transition timeout")
          break
        time.sleep(0.00001)
  mask(0x100778, 0x00000700,
       (b["timing_20_30_07"] << 8) |
       (0x80000000 if b["ramcfg_11_01_01"] else 0))
  mask(0x10f250, 0x000003f0, b["timing_20_2c_003f"] << 4)
  t10 = (timing[10] >> 24) & 0x7f
  mask(0x10f24c, 0x7f000000,
       max(t10, b["timing_20_2c_1fc0"]) << 24)
  mask(0x10f224, 0x001f0000, b["timing_20_30_f8"] << 16)
  mask(0x10fec4, 0x041e0f07,
       (b["timing_20_31_0800"] << 26) |
       (b["timing_20_31_0780"] << 17) |
       (b["timing_20_31_0078"] << 8) |
       b["timing_20_31_0007"])
  mask(0x10fec8, 0x00000027,
       (b["timing_20_31_8000"] << 5) | b["timing_20_31_7000"])
  # Restore refresh and mode registers, then issue the mode-register writes.
  dev.write32(0x10f090, 0x4000007e)
  delay_ns(2_000)
  dev.write32(0x10f314, 1)
  dev.write32(0x10f310, 1)
  dev.write32(0x10f210, 0x80000000)
  mask(0x10f338, 0x00000fff, mr[3])
  dev.write32(0x10f300, mr[0])
  mask(0x10f354, 0x00000fff, mr[8]); delay_ns(1_000)
  mask(0x10f330, 0x00000fff, mr[1])
  mask(0x10f340, 0x00000fff, mr[5] & ~0x004)
  mask(0x10f344, 0x00000fff, mr[6])
  mask(0x10f348, 0x00000fff, mr[7])
  if b["ramcfg_11_02_08"]:
    set_gpio_level(0x2e, 0, 20_000)
  mask(0x10f200, 0x80000000, 0x80000000)
  dev.write32(0x10f318, 1)
  mask(0x10f200, 0x80000000, 0)
  delay_ns(1_000)
  ram_nuts(0x10f200, 0x18808800, 0x00000000, 0x18808800)
  data = dev.read32(0x10f978) & ~0x00046144
  data |= 0x0000000b
  if not b["ramcfg_11_07_08"]:
    if not b["ramcfg_11_07_04"]: data |= 0x0000200c
  else:
    data |= 0x00040044
  dev.write32(0x10f978, data)
  # Mode-1 path enables the controller's low-clock mode before the final
  # write-leveling train.
  mask(0x10f830, 0x00000001, 0x00000001)
  train_data = 0x88020000 | (0x10000000 if b["ramcfg_11_07_04"] else 0)
  if not cfg.get("rammap_11_08_10", 0): train_data |= 0x00080000
  train(0xbc0f0000, train_data)
  delay_ns(1_000)
  # Nouveau restores LP3 immediately after the final train wait and before
  # pulsing 0x10f830[24].  The old port reversed those operations (H31).
  if mask(0x10f340, 0x00000fff, mr[5]) != mr[5]:
    delay_ns(1_000)
  mask(0x10f830, 0x01000000, 0x01000000)
  mask(0x10f830, 0x01000000, 0)
  if b["ramcfg_11_07_02"]:
    train(0x80020000, 0x01000000)
  ram_unblock()
  dev.write32(0x62c000, 0x0f0f0f00)
  mask(0x10f200, 0x00000800,
       0x00000800 if cfg.get("rammap_11_08_01", 0) else 0)
  ram_nuts(0x10f200, 0x18808800,
           0x00000800 if cfg.get("rammap_11_08_01", 0) else 0,
           0x18808800)
  # Nouveau executes the target gk104_ram_prog_0() only after ram_exec()
  # returns.  Keep TinyGPU's host-hostile 0x10f4xx stores on MEMX, but use a
  # second EXEC so a timeout cannot be mistaken for failure to reach LEAVE.
  atomic = getattr(dev, "atomic", False)
  if getattr(dev, "memx_on", False) or getattr(dev, "_wr32_flushes", 0):
    print(f"[nvbios] MEMX WR32 totals: {dev._wr32_flushes} execs, "
          f"{dev._wr32_words} words", flush=True)
  if atomic:
    dev.stop_memx_buffer(label="RAM transition")
    if not dev.start_memx_buffer(preflight=False):
      raise RuntimeError("PMU MEMX unavailable for post-transition target prog0")
    prog0()
    dev.stop_memx_buffer(label="target prog0")
  else:
    prog0()
    dev.stop_memx_buffer()
  # TinyGPU bit0 unstub *after* MEMX: mid-sequence bit0 drops BAR0 once the
  # PMU is active (proven 2026-07-16 replug).  Night5 standalone bit0 still
  # converts PRAMIN stub → virgin without clearing 0xaa2.
  # After post-MEMX bit0, the host PGRAPH pack (and BLCG/LTC) collapses BAR0
  # — defer with KEPLER_RAM_BIT0_DEFER=1 until just before first PRAMIN store.
  if os.environ.get("KEPLER_RAM_BLOCK", "0") == "bit0":
    if os.environ.get("KEPLER_RAM_BIT0_DEFER", "0") == "1":
      print("[nvbios] deferring post-MEMX bit0 until first PRAMIN store "
            "(KEPLER_RAM_BIT0_DEFER=1; PGRAPH pack is TinyGPU-hostile after bit0)",
            flush=True)
    else:
      boot0 = _real_dev.read32(0) & 0xffffffff
      if boot0 == 0xffffffff:
        raise RuntimeError(
            f"BAR0 dead before post-MEMX bit0 (PMC_BOOT_0={boot0:#x}); abort")
      # Night5: clear only 0x1620[0]; no 0x26f0, no unpause (pause+unpause of
      # both killed BAR0 on true cold).  KEPLER_RAM_BIT0_FULL=1 restores old.
      r1620 = _real_dev.read32(0x001620) & 0xffffffff
      if os.environ.get("KEPLER_RAM_BIT0_FULL", "0") == "1":
        print("[nvbios] post-MEMX bit0 FULL (0x1620[0]+0x26f0 pause/unpause)",
              flush=True)
        def _host_mask(reg: int, m: int, val: int) -> None:
          old = _real_dev.read32(reg) & 0xffffffff
          _real_dev.write32(reg, ((old & ~m) | val) & 0xffffffff)
        _host_mask(0x001620, 0x00000001, 0)
        _host_mask(0x0026f0, 0x00000001, 0)
        _host_mask(0x0026f0, 0x00000001, 0x00000001)
        _host_mask(0x001620, 0x00000001, 0x00000001)
      else:
        print(f"[nvbios] post-MEMX bit0: 0x1620 {r1620:#x}->"
              f"{r1620 & ~1:#x} (no 0x26f0, no unpause)", flush=True)
        _real_dev.write32(0x001620, r1620 & ~0x1)
      boot0 = _real_dev.read32(0) & 0xffffffff
      if boot0 == 0xffffffff:
        raise RuntimeError(
            f"post-MEMX bit0 killed BAR0 (PMC_BOOT_0={boot0:#x})")
      print(f"[nvbios] post-MEMX bit0 done; PMC_BOOT_0={boot0:#x}", flush=True)
  if debug:
    print(f"[nvbios] GK104 cold GDDR5 controller programmed: "
          f"freq={freq_mhz}MHz actual={actual // 1000}MHz "
         f"PLL(P={p1},N={n1},M={m1},fN={fn1:#x}) rammap={cfg['rammap_index']} "
          f"ramcfg={cfg['ramcfg_index']} timing={cfg['timing_index']}")
  return cfg
