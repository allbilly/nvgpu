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
import struct, time

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
      if self.debug:
        print(f"  [nvbios] read32({reg:#010x}) failed: {e}")
      return 0

  def _reg_wr32(self, reg: int, val: int) -> None:
    reg = self._nvreg(reg)
    if not self.init_exec():
      return
    try:
      self.dev.write32(reg, val)
      if self.debug:
        print(f"  [nvbios] WR {reg:#010x} = {val:#010x}")
    except Exception as e:
      if self.debug:
        print(f"  [nvbios] write32({reg:#010x}, {val:#010x}) failed: {e}")

  def _reg_mask(self, reg: int, mask: int, val: int) -> int:
    """init_mask: (r & ~mask) | val."""
    reg = self._nvreg(reg)
    if not self.init_exec():
      return 0
    try:
      old = self.dev.read32(reg)
      new = (old & ~mask) | (val & mask)
      self.dev.write32(reg, new)
      if self.debug:
        print(f"  [nvbios] MASK {reg:#010x} & ~{mask:#010x} | {val:#010x} = {new:#010x}")
      return old
    except Exception as e:
      if self.debug:
        print(f"  [nvbios] mask({reg:#010x}) failed: {e}")
      return 0

  def _vgai_rd(self, port: int, index: int) -> int:
    """VGA indexed I/O (CRTC/SEQ) - not available on this platform, return 0."""
    return 0

  def _vgai_wr(self, port: int, index: int, val: int) -> None:
    pass

  def _port_rd(self, port: int) -> int:
    return 0
  def _port_wr(self, port: int, val: int) -> None:
    pass

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
    strap = (self._reg_rd32(0x101000) & 0x0000003c) >> 2
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
    count = self.rd08(self.offset + 3)
    self.offset += 4 + count

  def _op_zm_i2c_byte(self):
    count = self.rd08(self.offset + 3)
    self.offset += 4 + count * 2

  def _op_i2c_byte(self):
    count = self.rd08(self.offset + 3)
    self.offset += 4 + count * 3

  def _op_i2c_if(self):
    self.offset += 6
    if self.init_exec():
      self.init_exec_set(False)

  def _op_i2c_long_if(self):
    self.offset += 7
    if self.init_exec():
      self.init_exec_set(False)

  def _op_tmds(self):
    self.offset += 5

  def _op_zm_tmds_group(self):
    count = self.rd08(self.offset + 2)
    self.offset += 3 + count * 2

  def _op_cr(self):
    self.offset += 4

  def _op_zm_cr(self):
    self.offset += 3

  def _op_zm_cr_group(self):
    count = self.rd08(self.offset + 1)
    self.offset += 2 + count * 2

  def _op_cr_idx_adr_latch(self):
    count = self.rd08(self.offset + 4)
    self.offset += 5 + count

  def _op_io(self):
    self.offset += 5

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

  def _op_gpio_ne(self):
    count = self.rd08(self.offset + 1)
    self.offset += 2 + count

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
  if scripts is None:
    scripts = find_vbios_scripts(image)
  for s in scripts:
    if debug:
      print(f"[nvbios] === script 0x{s:04x} ===")
    init.run_script(s)
