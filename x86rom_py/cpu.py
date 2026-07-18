"""16-bit real-mode interpreter for dynamically reached Palit ROM opcodes."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .bus import Bus
from .firmware import FirmwareError, FirmwareServices
from .memory import MemoryError, RealModeMemory
from .trace import TraceLog


class CPUError(RuntimeError):
  """Unsupported opcode or illegal CPU state."""


class StopReason:
  ROM_RETURN = "rom-return"
  UNSUPPORTED_OPCODE = "unsupported-opcode"
  UNSUPPORTED_FIRMWARE = "unsupported-firmware-service"
  UNSAFE_OPERATION = "unsafe-operation"
  TRACE_DIVERGENCE = "trace-divergence"
  INSTRUCTION_BUDGET = "instruction-budget"
  OPERATION_BUDGET = "operation-budget"
  WALL_TIME_BUDGET = "wall-time-budget"
  LOOP_DETECTED = "loop-detected"
  BAR0_LOST = "bar0-lost"
  PRAMIN_LIVE = "pramin-live"
  HALT = "halt"
  MEMORY_ERROR = "memory-error"
  BUS_ERROR = "bus-error"


FLAG_CF = 0
FLAG_PF = 2
FLAG_AF = 4
FLAG_ZF = 6
FLAG_SF = 7
FLAG_TF = 8
FLAG_IF = 9
FLAG_DF = 10
FLAG_OF = 11


@dataclass
class StopInfo:
  reason: str
  cs: int
  ip: int
  bytes_at_ip: bytes
  message: str = ""
  regs: Optional[Dict[str, int]] = None


class CPU:
  """Real-mode x86 subset; opcodes added only when reached by the pinned ROM."""

  def __init__(
    self,
    mem: RealModeMemory,
    bus: Bus,
    firmware: FirmwareServices,
    *,
    trace: Optional[TraceLog] = None,
  ) -> None:
    self.mem = mem
    self.bus = bus
    self.firmware = firmware
    self.trace = trace

    self.eax = self.ebx = self.ecx = self.edx = 0
    self.esi = self.edi = self.ebp = self.esp = 0
    self.cs = self.ds = self.es = self.ss = self.fs = self.gs = 0
    self.eip = 0
    self.eflags = 0x00000002

    self.stopped: Optional[StopInfo] = None
    self.insn_count = 0
    self.op_count = 0
    self._rep_active = False

  # --- register helpers ---
  def _gpr(self, reg: int, width: int) -> int:
    if width == 1:
      if reg < 4:
        return (getattr(self, _REG32[reg]) & 0xFF)
      return (getattr(self, _REG32[reg - 4]) >> 8) & 0xFF
    name = _REG32[reg]
    val = getattr(self, name)
    return val & (0xFFFF if width == 2 else 0xFFFFFFFF)

  def _set_gpr(self, reg: int, width: int, value: int) -> None:
    if width == 1:
      if reg < 4:
        name = _REG32[reg]
        cur = getattr(self, name)
        setattr(self, name, (cur & ~0xFF) | (value & 0xFF))
      else:
        name = _REG32[reg - 4]
        cur = getattr(self, name)
        setattr(self, name, (cur & ~0xFF00) | ((value & 0xFF) << 8))
      return
    name = _REG32[reg]
    if width == 2:
      cur = getattr(self, name)
      setattr(self, name, (cur & ~0xFFFF) | (value & 0xFFFF))
    else:
      setattr(self, name, value & 0xFFFFFFFF)

  def get_flag(self, name: str) -> bool:
    bit = {
      "CF": FLAG_CF, "PF": FLAG_PF, "AF": FLAG_AF, "ZF": FLAG_ZF,
      "SF": FLAG_SF, "TF": FLAG_TF, "IF": FLAG_IF, "DF": FLAG_DF, "OF": FLAG_OF,
    }[name]
    return bool(self.eflags & (1 << bit))

  def set_flag(self, name: str, val: bool) -> None:
    bit = {
      "CF": FLAG_CF, "PF": FLAG_PF, "AF": FLAG_AF, "ZF": FLAG_ZF,
      "SF": FLAG_SF, "TF": FLAG_TF, "IF": FLAG_IF, "DF": FLAG_DF, "OF": FLAG_OF,
    }[name]
    if val:
      self.eflags |= (1 << bit)
    else:
      self.eflags &= ~(1 << bit)

  def regs_dict(self) -> Dict[str, int]:
    return {
      "EAX": self.eax, "EBX": self.ebx, "ECX": self.ecx, "EDX": self.edx,
      "ESI": self.esi, "EDI": self.edi, "EBP": self.ebp, "ESP": self.esp,
      "CS": self.cs, "DS": self.ds, "ES": self.es, "SS": self.ss,
      "FS": self.fs, "GS": self.gs, "EIP": self.eip, "EFLAGS": self.eflags,
    }

  def abort_unsupported(self, raw: bytes, msg: str = "") -> None:
    self.stopped = StopInfo(
      reason=StopReason.UNSUPPORTED_OPCODE,
      cs=self.cs,
      ip=self.eip & 0xFFFF,
      bytes_at_ip=raw,
      message=msg or f"unsupported opcode {raw.hex()}",
      regs=self.regs_dict(),
    )

  # --- flag updates ---
  def _parity(self, val: int) -> bool:
    v = val & 0xFF
    return bin(v).count("1") % 2 == 0

  def _set_szp(self, result: int, width: int) -> None:
    mask = (1 << (width * 8)) - 1
    r = result & mask
    self.set_flag("ZF", r == 0)
    self.set_flag("SF", bool(r & (1 << (width * 8 - 1))))
    self.set_flag("PF", self._parity(r))

  def _set_logic_flags(self, result: int, width: int) -> None:
    self._set_szp(result, width)
    self.set_flag("CF", False)
    self.set_flag("OF", False)
    self.set_flag("AF", False)

  def _add_flags(self, a: int, b: int, result: int, width: int, *, carry: bool = False) -> None:
    bits = width * 8
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    r = result & mask
    self._set_szp(r, width)
    if carry:
      # for ADC already included in result; CF from unsigned overflow
      self.set_flag("CF", result > mask)
    else:
      self.set_flag("CF", result > mask)
    # OF: signed overflow
    self.set_flag("OF", bool((~(a ^ b) & (a ^ r)) & sign))
    self.set_flag("AF", bool(((a ^ b ^ r) & 0x10)))

  def _sub_flags(self, a: int, b: int, result: int, width: int) -> None:
    bits = width * 8
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    r = result & mask
    self._set_szp(r, width)
    self.set_flag("CF", b > a)
    self.set_flag("OF", bool(((a ^ b) & (a ^ r)) & sign))
    self.set_flag("AF", bool(((a ^ b ^ r) & 0x10)))

  # --- stack ---
  def push_width(self, value: int, width: int) -> None:
    self.esp = (self.esp - width) & 0xFFFF
    self.mem.write_seg(self.ss, self.esp & 0xFFFF, width, value)

  def pop_width(self, width: int) -> int:
    val = self.mem.read_seg(self.ss, self.esp & 0xFFFF, width)
    self.esp = (self.esp + width) & 0xFFFF
    return val

  # --- fetch ---
  def _peek(self, n: int = 16) -> bytes:
    return self.mem.fetch(self.cs, self.eip & 0xFFFF, n)

  def step(self) -> bool:
    """Execute one instruction. Returns False if stopped."""
    if self.stopped is not None:
      return False
    start_ip = self.eip & 0xFFFF
    raw = self._peek(16)
    if hasattr(self.bus, "set_cs_ip"):
      self.bus.set_cs_ip(self.cs, start_ip)
    try:
      consumed = self._execute(raw)
    except CPUError as e:
      self.abort_unsupported(raw[:1], str(e))
      return False
    except FirmwareError as e:
      self.stopped = StopInfo(
        reason=StopReason.UNSUPPORTED_FIRMWARE,
        cs=self.cs, ip=start_ip, bytes_at_ip=raw[:8],
        message=str(e), regs=self.regs_dict(),
      )
      return False
    except MemoryError as e:
      self.stopped = StopInfo(
        reason=StopReason.MEMORY_ERROR,
        cs=self.cs, ip=start_ip, bytes_at_ip=raw[:8],
        message=str(e), regs=self.regs_dict(),
      )
      return False
    except Exception as e:
      from .bus import BusError
      from .safety import SafetyError
      if isinstance(e, SafetyError):
        self.stopped = StopInfo(
          reason=StopReason.UNSAFE_OPERATION,
          cs=self.cs, ip=start_ip, bytes_at_ip=raw[:8],
          message=str(e), regs=self.regs_dict(),
        )
        return False
      if isinstance(e, BusError):
        self.stopped = StopInfo(
          reason=StopReason.BUS_ERROR,
          cs=self.cs, ip=start_ip, bytes_at_ip=raw[:8],
          message=str(e), regs=self.regs_dict(),
        )
        return False
      raise
    if self.stopped is not None:
      return False
    self.insn_count += 1
    if self.trace is not None and consumed:
      self.trace.record(
        operation="insn",
        address_space="cpu",
        direction="none",
        cs=self.cs,
        ip=start_ip,
        instruction_bytes=raw[:consumed].hex(),
      )
    return True

  def run(
    self,
    *,
    max_insns: int = 100_000,
    stop_at_ips: Optional[set] = None,
    stop_on_return_below: Optional[int] = None,
  ) -> StopInfo:
    stop_at_ips = stop_at_ips or set()
    seen_ip: Dict[Tuple[int, int], int] = {}
    while self.stopped is None and self.insn_count < max_insns:
      ip = self.eip & 0xFFFF
      if ip in stop_at_ips:
        self.stopped = StopInfo(
          reason="breakpoint", cs=self.cs, ip=ip,
          bytes_at_ip=self._peek(8), regs=self.regs_dict(),
        )
        break
      key = (self.cs, ip)
      seen_ip[key] = seen_ip.get(key, 0) + 1
      if seen_ip[key] > 10_000:
        self.stopped = StopInfo(
          reason=StopReason.LOOP_DETECTED, cs=self.cs, ip=ip,
          bytes_at_ip=self._peek(8), message=f"CS:IP hit {seen_ip[key]} times",
          regs=self.regs_dict(),
        )
        break
      if not self.step():
        break
    if self.stopped is None:
      self.stopped = StopInfo(
        reason=StopReason.INSTRUCTION_BUDGET,
        cs=self.cs, ip=self.eip & 0xFFFF,
        bytes_at_ip=self._peek(8),
        message=f"hit max_insns={max_insns}",
        regs=self.regs_dict(),
      )
    return self.stopped

  def _execute(self, raw: bytes) -> int:
    """Decode and execute; return bytes consumed. May modify eip itself for jumps."""
    i = 0
    seg_override: Optional[int] = None
    op_size_32 = False
    addr_size_32 = False
    rep = 0  # 0 none, 1 REPE/REPZ, 2 REPNE/REPNZ, 3 REP
    lock = False

    # prefixes
    while True:
      b = raw[i]
      if b == 0xF0:
        lock = True; i += 1
      elif b == 0xF2:
        rep = 2; i += 1
      elif b == 0xF3:
        rep = 1; i += 1
      elif b == 0x2E:
        seg_override = self.cs; i += 1
      elif b == 0x3E:
        seg_override = self.ds; i += 1
      elif b == 0x26:
        seg_override = self.es; i += 1
      elif b == 0x36:
        seg_override = self.ss; i += 1
      elif b == 0x64:
        seg_override = self.fs; i += 1
      elif b == 0x65:
        seg_override = self.gs; i += 1
      elif b == 0x66:
        op_size_32 = True; i += 1
      elif b == 0x67:
        addr_size_32 = True; i += 1
      else:
        break

    op_width = 4 if op_size_32 else 2
    addr_width = 4 if addr_size_32 else 2
    opcode = raw[i]
    i += 1
    start_eip = self.eip

    def finish(n: int) -> int:
      # n = total bytes from start including prefixes; advance IP unless jmp set it
      if self.eip == start_eip:
        self.eip = (start_eip + n) & 0xFFFF
      return n

    def seg_for_mem(default_ds: bool = True) -> int:
      if seg_override is not None:
        return seg_override
      return self.ds if default_ds else self.ss

    # --- one-byte opcodes ---
    if opcode == 0x90:  # NOP
      return finish(i)
    if opcode == 0xF4:  # HLT
      self.stopped = StopInfo(
        reason=StopReason.HALT, cs=self.cs, ip=start_eip & 0xFFFF,
        bytes_at_ip=raw[:i], regs=self.regs_dict(),
      )
      return i
    if opcode == 0xFA:  # CLI
      self.set_flag("IF", False)
      return finish(i)
    if opcode == 0xFB:  # STI
      self.set_flag("IF", True)
      return finish(i)
    if opcode == 0xFC:  # CLD
      self.set_flag("DF", False)
      return finish(i)
    if opcode == 0xFD:  # STD
      self.set_flag("DF", True)
      return finish(i)
    if opcode == 0xF8:  # CLC
      self.set_flag("CF", False)
      return finish(i)
    if opcode == 0xF9:  # STC
      self.set_flag("CF", True)
      return finish(i)
    if opcode == 0xF5:  # CMC
      self.set_flag("CF", not self.get_flag("CF"))
      return finish(i)
    if opcode == 0xCC:  # INT3
      raise CPUError("INT3")
    if opcode == 0xCD:  # INT imm8
      vec = raw[i]; i += 1
      self.eip = (start_eip + i) & 0xFFFF
      self._do_int(vec)
      return i
    if opcode == 0xCF:  # IRET
      ip = self.pop_width(2)
      cs = self.pop_width(2)
      flags = self.pop_width(2)
      self.eip = ip
      self.cs = cs
      self.eflags = (self.eflags & ~0xFFFF) | (flags & 0xFFFF)
      # Detect return from option ROM (far return to caller below ROM)
      if cs != self.mem.rom_segment and cs < 0xC000:
        self.stopped = StopInfo(
          reason=StopReason.ROM_RETURN, cs=cs, ip=ip,
          bytes_at_ip=b"", message="IRET left ROM", regs=self.regs_dict(),
        )
      return i
    if opcode == 0xC3:  # RETN
      self.eip = self.pop_width(2)
      return i
    if opcode == 0xC2:  # RETN imm16
      imm = struct.unpack_from("<H", raw, i)[0]; i += 2
      self.eip = self.pop_width(2)
      self.esp = (self.esp + imm) & 0xFFFF
      return i
    if opcode == 0xCB:  # RETF
      ip = self.pop_width(2)
      cs = self.pop_width(2)
      self.eip = ip
      self.cs = cs
      if cs != self.mem.rom_segment and (cs << 4) < self.mem.rom_phys:
        self.stopped = StopInfo(
          reason=StopReason.ROM_RETURN, cs=cs, ip=ip,
          bytes_at_ip=b"", message="RETF left ROM", regs=self.regs_dict(),
        )
      return i
    if opcode == 0xCA:  # RETF imm16
      imm = struct.unpack_from("<H", raw, i)[0]; i += 2
      ip = self.pop_width(2)
      cs = self.pop_width(2)
      self.esp = (self.esp + imm) & 0xFFFF
      self.eip = ip
      self.cs = cs
      return i
    if opcode == 0xE8:  # CALL rel16/32
      if op_width == 4:
        rel = struct.unpack_from("<i", raw, i)[0]; i += 4
        next_ip = (start_eip + i) & 0xFFFFFFFF
        self.push_width(next_ip & 0xFFFF, 2)  # real-mode still 16-bit stack for near?
        # In real mode with 66h prefix, still typically push 16 on 16-bit stack.
        # Use 16-bit IP.
        self.push_width = self.push_width  # noqa keep
        self.eip = (next_ip + rel) & 0xFFFF
      else:
        rel = struct.unpack_from("<h", raw, i)[0]; i += 2
        next_ip = (start_eip + i) & 0xFFFF
        self.push_width(next_ip, 2)
        self.eip = (next_ip + rel) & 0xFFFF
      return i
    if opcode == 0x9A:  # CALL far ptr16:16
      ip = struct.unpack_from("<H", raw, i)[0]; i += 2
      cs = struct.unpack_from("<H", raw, i)[0]; i += 2
      next_ip = (start_eip + i) & 0xFFFF
      self.push_width(self.cs, 2)
      self.push_width(next_ip, 2)
      self.cs = cs
      self.eip = ip
      return i
    if opcode == 0xE9:  # JMP rel16/32
      if op_width == 4:
        rel = struct.unpack_from("<i", raw, i)[0]; i += 4
        self.eip = (start_eip + i + rel) & 0xFFFF
      else:
        rel = struct.unpack_from("<h", raw, i)[0]; i += 2
        self.eip = (start_eip + i + rel) & 0xFFFF
      return i
    if opcode == 0xEB:  # JMP rel8
      rel = struct.unpack_from("<b", raw, i)[0]; i += 1
      self.eip = (start_eip + i + rel) & 0xFFFF
      return i
    if opcode == 0xEA:  # JMP far
      ip = struct.unpack_from("<H", raw, i)[0]; i += 2
      cs = struct.unpack_from("<H", raw, i)[0]; i += 2
      self.cs = cs
      self.eip = ip
      return i

    # conditional jumps Jcc rel8
    if 0x70 <= opcode <= 0x7F:
      rel = struct.unpack_from("<b", raw, i)[0]; i += 1
      if self._jcc(opcode & 0x0F):
        self.eip = (start_eip + i + rel) & 0xFFFF
      else:
        self.eip = (start_eip + i) & 0xFFFF
      return i

    # short jumps with 0F 8x handled below

    # PUSH/POP segment
    if opcode in (0x06, 0x0E, 0x16, 0x1E):  # PUSH ES/CS/SS/DS
      seg = {0x06: self.es, 0x0E: self.cs, 0x16: self.ss, 0x1E: self.ds}[opcode]
      self.push_width(seg, 2)
      return finish(i)
    if opcode in (0x07, 0x17, 0x1F):  # POP ES/SS/DS
      val = self.pop_width(2)
      if opcode == 0x07:
        self.es = val
      elif opcode == 0x17:
        self.ss = val
      else:
        self.ds = val
      return finish(i)

    # PUSH/POP r16/r32 (66h → 32-bit operand on 386+ real mode)
    if 0x50 <= opcode <= 0x57:
      w = 4 if op_width == 4 else 2
      self.push_width(self._gpr(opcode - 0x50, w), w)
      return finish(i)
    if 0x58 <= opcode <= 0x5F:
      w = 4 if op_width == 4 else 2
      self._set_gpr(opcode - 0x58, w, self.pop_width(w))
      return finish(i)

    if opcode == 0x60:  # PUSHA
      sp = self.esp & 0xFFFF
      for reg in (self.eax, self.ecx, self.edx, self.ebx, sp, self.ebp, self.esi, self.edi):
        self.push_width(reg & 0xFFFF, 2)
      return finish(i)
    if opcode == 0x61:  # POPA
      self.edi = (self.edi & ~0xFFFF) | self.pop_width(2)
      self.esi = (self.esi & ~0xFFFF) | self.pop_width(2)
      self.ebp = (self.ebp & ~0xFFFF) | self.pop_width(2)
      self.pop_width(2)  # skip SP
      self.ebx = (self.ebx & ~0xFFFF) | self.pop_width(2)
      self.edx = (self.edx & ~0xFFFF) | self.pop_width(2)
      self.ecx = (self.ecx & ~0xFFFF) | self.pop_width(2)
      self.eax = (self.eax & ~0xFFFF) | self.pop_width(2)
      return finish(i)
    if opcode == 0x9C:  # PUSHF
      self.push_width(self.eflags & 0xFFFF, 2)
      return finish(i)
    if opcode == 0x9D:  # POPF
      self.eflags = (self.eflags & ~0xFFFF) | (self.pop_width(2) & 0xFFFF)
      self.eflags |= 0x2
      return finish(i)
    if opcode == 0x68:  # PUSH imm16/32
      if op_width == 4:
        imm = struct.unpack_from("<I", raw, i)[0]; i += 4
        self.push_width(imm & 0xFFFF, 2)
        self.push_width((imm >> 16) & 0xFFFF, 2)
      else:
        imm = struct.unpack_from("<H", raw, i)[0]; i += 2
        self.push_width(imm, 2)
      return finish(i)
    if opcode == 0x6A:  # PUSH imm8
      imm = struct.unpack_from("<b", raw, i)[0]; i += 1
      self.push_width(imm & 0xFFFF, 2)
      return finish(i)

    # ALU Acc, imm
    if opcode in (0x04, 0x05, 0x0C, 0x0D, 0x14, 0x15, 0x1C, 0x1D,
                  0x24, 0x25, 0x2C, 0x2D, 0x34, 0x35, 0x3C, 0x3D):
      which = (opcode >> 3) & 7
      w = opcode & 1
      width = 1 if w == 0 else op_width
      if width == 1:
        imm = raw[i]; i += 1
        a = self.eax & 0xFF
      elif width == 2:
        imm = struct.unpack_from("<H", raw, i)[0]; i += 2
        a = self.eax & 0xFFFF
      else:
        imm = struct.unpack_from("<I", raw, i)[0]; i += 4
        a = self.eax
      result = self._alu_op(which, a, imm, width)
      if which != 7:  # not CMP
        if width == 1:
          self.eax = (self.eax & ~0xFF) | (result & 0xFF)
        elif width == 2:
          self.eax = (self.eax & ~0xFFFF) | (result & 0xFFFF)
        else:
          self.eax = result & 0xFFFFFFFF
      return finish(i)

    # MOV r/m, r and friends — ModRM group
    if opcode in (
      0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0A, 0x0B,
      0x10, 0x11, 0x12, 0x13, 0x18, 0x19, 0x1A, 0x1B,
      0x20, 0x21, 0x22, 0x23, 0x28, 0x29, 0x2A, 0x2B,
      0x30, 0x31, 0x32, 0x33, 0x38, 0x39, 0x3A, 0x3B,
      0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8E,
      0x84, 0x85, 0x86, 0x87,
      0xFE, 0xFF,
      0x80, 0x81, 0x82, 0x83,
      0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3,
      0xF6, 0xF7,
      0x8F, 0xC6, 0xC7,
      0x69, 0x6B,
    ):
      return self._exec_modrm(opcode, raw, i, start_eip, op_width, addr_width, seg_override, finish)

    # MOV r16/32, imm
    if 0xB0 <= opcode <= 0xB7:
      self._set_gpr(opcode - 0xB0, 1, raw[i]); i += 1
      return finish(i)
    if 0xB8 <= opcode <= 0xBF:
      if op_width == 4:
        imm = struct.unpack_from("<I", raw, i)[0]; i += 4
        self._set_gpr(opcode - 0xB8, 4, imm)
      else:
        imm = struct.unpack_from("<H", raw, i)[0]; i += 2
        self._set_gpr(opcode - 0xB8, 2, imm)
      return finish(i)

    # XCHG AX,r
    if 0x91 <= opcode <= 0x97:
      r = opcode - 0x90
      a = self._gpr(0, op_width)
      b = self._gpr(r, op_width)
      self._set_gpr(0, op_width if op_width == 4 else 2, b)
      self._set_gpr(r, op_width if op_width == 4 else 2, a)
      return finish(i)

    # INC/DEC r16
    if 0x40 <= opcode <= 0x47:
      r = opcode - 0x40
      v = self._gpr(r, 2)
      res = (v + 1) & 0xFFFF
      self._set_szp(res, 2)
      self.set_flag("AF", (v & 0xF) == 0xF)
      self.set_flag("OF", v == 0x7FFF)
      self._set_gpr(r, 2, res)
      return finish(i)
    if 0x48 <= opcode <= 0x4F:
      r = opcode - 0x48
      v = self._gpr(r, 2)
      res = (v - 1) & 0xFFFF
      self._set_szp(res, 2)
      self.set_flag("AF", (v & 0xF) == 0)
      self.set_flag("OF", v == 0x8000)
      self._set_gpr(r, 2, res)
      return finish(i)

    # IN/OUT
    if opcode == 0xE4:  # IN AL, imm8
      port = raw[i]; i += 1
      self.eax = (self.eax & ~0xFF) | (self.bus.io_read(port, 1) & 0xFF)
      self.op_count += 1
      return finish(i)
    if opcode == 0xE5:  # IN AX/EAX, imm8
      port = raw[i]; i += 1
      val = self.bus.io_read(port, op_width if op_width == 2 else 2)
      if op_width == 4:
        self.eax = self.bus.io_read(port, 4)
      else:
        self.eax = (self.eax & ~0xFFFF) | (val & 0xFFFF)
      self.op_count += 1
      return finish(i)
    if opcode == 0xEC:  # IN AL, DX
      port = self.edx & 0xFFFF
      self.eax = (self.eax & ~0xFF) | (self.bus.io_read(port, 1) & 0xFF)
      self.op_count += 1
      return finish(i)
    if opcode == 0xED:  # IN AX/EAX, DX
      port = self.edx & 0xFFFF
      if op_width == 4:
        self.eax = self.bus.io_read(port, 4) & 0xFFFFFFFF
      else:
        self.eax = (self.eax & ~0xFFFF) | (self.bus.io_read(port, 2) & 0xFFFF)
      self.op_count += 1
      return finish(i)
    if opcode == 0xE6:  # OUT imm8, AL
      port = raw[i]; i += 1
      self.bus.io_write(port, 1, self.eax & 0xFF)
      self.op_count += 1
      return finish(i)
    if opcode == 0xE7:  # OUT imm8, AX
      port = raw[i]; i += 1
      self.bus.io_write(port, 2 if op_width == 2 else 4, self.eax)
      self.op_count += 1
      return finish(i)
    if opcode == 0xEE:  # OUT DX, AL
      self.bus.io_write(self.edx & 0xFFFF, 1, self.eax & 0xFF)
      self.op_count += 1
      return finish(i)
    if opcode == 0xEF:  # OUT DX, AX/EAX
      port = self.edx & 0xFFFF
      if op_width == 4:
        self.bus.io_write(port, 4, self.eax)
      else:
        self.bus.io_write(port, 2, self.eax & 0xFFFF)
      self.op_count += 1
      return finish(i)

    # LEA
    if opcode == 0x8D:
      modrm = raw[i]; i += 1
      _reg, ea, i = self._modrm_ea(modrm, raw, i, addr_width, seg_override)
      # LEA stores offset only
      self._set_gpr(_reg, op_width if op_width == 4 else 2, ea[1] if ea else 0)
      return finish(i)

    # LES / LDS
    if opcode in (0xC4, 0xC5):
      modrm = raw[i]; i += 1
      reg, ea, i = self._modrm_ea(modrm, raw, i, addr_width, seg_override)
      if ea is None:
        raise CPUError("LES/LDS on register")
      seg, off = ea
      ip_val = self.mem.read_seg(seg, off, 2)
      cs_val = self.mem.read_seg(seg, (off + 2) & 0xFFFF, 2)
      self._set_gpr(reg, 2, ip_val)
      if opcode == 0xC4:
        self.es = cs_val
      else:
        self.ds = cs_val
      return finish(i)

    # CBW / CWD
    if opcode == 0x98:
      if op_width == 4:  # CWDE
        ax = self.eax & 0xFFFF
        self.eax = ax if ax < 0x8000 else (ax | 0xFFFF0000)
      else:
        al = self.eax & 0xFF
        self.eax = (self.eax & ~0xFFFF) | (al if al < 0x80 else (al | 0xFF00))
      return finish(i)
    if opcode == 0x99:
      if op_width == 4:
        self.edx = 0 if self.eax < 0x80000000 else 0xFFFFFFFF
      else:
        self.edx = (self.edx & ~0xFFFF) | (0 if (self.eax & 0x8000) == 0 else 0xFFFF)
      return finish(i)

    # SAHF / LAHF
    if opcode == 0x9E:
      ah = (self.eax >> 8) & 0xFF
      self.eflags = (self.eflags & ~0xD5) | (ah & 0xD5)
      self.eflags |= 0x2
      return finish(i)
    if opcode == 0x9F:
      flags = (self.eflags & 0xD5) | 0x2
      self.eax = (self.eax & ~0xFF00) | (flags << 8)
      return finish(i)

    # XLAT
    if opcode == 0xD7:
      seg = seg_override if seg_override is not None else self.ds
      off = ((self.ebx & 0xFFFF) + (self.eax & 0xFF)) & 0xFFFF
      self.eax = (self.eax & ~0xFF) | self.mem.read_seg(seg, off, 1)
      return finish(i)

    # string ops
    if opcode in (0xA4, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF):
      return self._string_op(opcode, raw, i, start_eip, op_width, rep, seg_override, finish)

    # MOV mem offsets
    if opcode in (0xA0, 0xA1, 0xA2, 0xA3):
      seg = seg_override if seg_override is not None else self.ds
      if addr_width == 4:
        off = struct.unpack_from("<I", raw, i)[0]; i += 4
      else:
        off = struct.unpack_from("<H", raw, i)[0]; i += 2
      if opcode == 0xA0:
        self.eax = (self.eax & ~0xFF) | self.mem.read_seg(seg, off & 0xFFFF, 1)
      elif opcode == 0xA1:
        w = op_width
        val = self.mem.read_seg(seg, off & 0xFFFF, 2 if w == 2 else 4)
        if w == 2:
          self.eax = (self.eax & ~0xFFFF) | (val & 0xFFFF)
        else:
          self.eax = val
      elif opcode == 0xA2:
        self.mem.write_seg(seg, off & 0xFFFF, 1, self.eax & 0xFF)
      else:
        w = op_width
        if w == 2:
          self.mem.write_seg(seg, off & 0xFFFF, 2, self.eax & 0xFFFF)
        else:
          self.mem.write_seg(seg, off & 0xFFFF, 4, self.eax)
      return finish(i)

    # TEST AL/AX, imm
    if opcode == 0xA8:
      imm = raw[i]; i += 1
      self._set_logic_flags((self.eax & 0xFF) & imm, 1)
      return finish(i)
    if opcode == 0xA9:
      if op_width == 4:
        imm = struct.unpack_from("<I", raw, i)[0]; i += 4
        self._set_logic_flags(self.eax & imm, 4)
      else:
        imm = struct.unpack_from("<H", raw, i)[0]; i += 2
        self._set_logic_flags((self.eax & 0xFFFF) & imm, 2)
      return finish(i)

    # LOOP / JCXZ
    if opcode in (0xE0, 0xE1, 0xE2):
      rel = struct.unpack_from("<b", raw, i)[0]; i += 1
      cx = (self.ecx - 1) & 0xFFFF
      self.ecx = (self.ecx & ~0xFFFF) | cx
      take = False
      if opcode == 0xE2:
        take = cx != 0
      elif opcode == 0xE1:
        take = cx != 0 and self.get_flag("ZF")
      else:
        take = cx != 0 and not self.get_flag("ZF")
      if take:
        self.eip = (start_eip + i + rel) & 0xFFFF
      else:
        self.eip = (start_eip + i) & 0xFFFF
      return i
    if opcode == 0xE3:  # JCXZ
      rel = struct.unpack_from("<b", raw, i)[0]; i += 1
      if (self.ecx & 0xFFFF) == 0:
        self.eip = (start_eip + i + rel) & 0xFFFF
      else:
        self.eip = (start_eip + i) & 0xFFFF
      return i

    # two-byte opcode 0F
    if opcode == 0x0F:
      op2 = raw[i]; i += 1
      if 0x80 <= op2 <= 0x8F:  # Jcc near
        if op_width == 4:
          rel = struct.unpack_from("<i", raw, i)[0]; i += 4
        else:
          rel = struct.unpack_from("<h", raw, i)[0]; i += 2
        if self._jcc(op2 & 0x0F):
          self.eip = (start_eip + i + rel) & 0xFFFF
        else:
          self.eip = (start_eip + i) & 0xFFFF
        return i
      if op2 in (0xB6, 0xB7, 0xBE, 0xBF):  # MOVZX/MOVSX
        modrm = raw[i]; i += 1
        reg, ea, i = self._modrm_ea(modrm, raw, i, addr_width, seg_override)
        src_w = 1 if op2 in (0xB6, 0xBE) else 2
        val = self._read_ea(ea, reg if ea is None else None, src_w, modrm)
        if op2 in (0xBE, 0xBF):  # MOVSX
          bits = 8 * src_w
          if val & (1 << (bits - 1)):
            val |= ~((1 << bits) - 1)
            val &= 0xFFFFFFFF if op_width == 4 else 0xFFFF
        dest_w = 4 if op_width == 4 else 2
        self._set_gpr(reg, dest_w, val & ((1 << (8 * dest_w)) - 1))
        return finish(i)
      if op2 in (0xBC, 0xBD):  # BSF / BSR
        modrm = raw[i]; i += 1
        reg, ea, i = self._modrm_ea(modrm, raw, i, addr_width, seg_override)
        width = 4 if op_width == 4 else 2
        src = self._read_ea(ea, None, width, modrm)
        if src == 0:
          self.set_flag("ZF", True)
        else:
          self.set_flag("ZF", False)
          if op2 == 0xBC:  # BSF
            bit = 0
            while (src & (1 << bit)) == 0:
              bit += 1
          else:  # BSR
            bit = width * 8 - 1
            while (src & (1 << bit)) == 0:
              bit -= 1
          self._set_gpr(reg, width, bit)
        return finish(i)
      if op2 == 0xA0:  # PUSH FS
        self.push_width(self.fs, 2); return finish(i)
      if op2 == 0xA1:  # POP FS
        self.fs = self.pop_width(2); return finish(i)
      if op2 == 0xA8:  # PUSH GS
        self.push_width(self.gs, 2); return finish(i)
      if op2 == 0xA9:  # POP GS
        self.gs = self.pop_width(2); return finish(i)
      if op2 == 0xA2:  # CPUID — abort, not expected in option ROM
        raise CPUError("CPUID")
      raise CPUError(f"unsupported 0F {op2:02x}")

    # group already handled; fallthrough
    raise CPUError(f"unsupported opcode {opcode:02x} @ {self.cs:04x}:{start_eip:04x}")

  def _do_int(self, vector: int) -> None:
    # Soft interrupt: push flags, CS, IP then dispatch via firmware (not IVT jump),
    # matching option-ROM BIOS service model. IVT still consulted for unhandled?
    self.push_width(self.eflags & 0xFFFF, 2)
    self.push_width(self.cs, 2)
    self.push_width(self.eip & 0xFFFF, 2)
    self.set_flag("IF", False)
    self.set_flag("TF", False)
    try:
      self.firmware.handle(self, vector)
    finally:
      # BIOS services return via IRET semantics inline
      ip = self.pop_width(2)
      cs = self.pop_width(2)
      flags = self.pop_width(2)
      self.eip = ip
      self.cs = cs
      self.eflags = (self.eflags & ~0xFFFF) | (flags & 0xFFFF)

  def _jcc(self, cc: int) -> bool:
    cf, zf, sf, of, pf = (
      self.get_flag("CF"), self.get_flag("ZF"), self.get_flag("SF"),
      self.get_flag("OF"), self.get_flag("PF"),
    )
    table = {
      0x0: of, 0x1: not of,
      0x2: cf, 0x3: not cf,
      0x4: zf, 0x5: not zf,
      0x6: cf or zf, 0x7: not (cf or zf),
      0x8: sf, 0x9: not sf,
      0xA: pf, 0xB: not pf,
      0xC: sf != of, 0xD: sf == of,
      0xE: zf or (sf != of), 0xF: not zf and (sf == of),
    }
    return table[cc]

  def _alu_op(self, which: int, a: int, b: int, width: int) -> int:
    bits = width * 8
    mask = (1 << bits) - 1
    a &= mask
    b &= mask
    if which == 0:  # ADD
      r = a + b
      self._add_flags(a, b, r, width)
      return r & mask
    if which == 1:  # OR
      r = a | b
      self._set_logic_flags(r, width)
      return r
    if which == 2:  # ADC
      r = a + b + (1 if self.get_flag("CF") else 0)
      self._add_flags(a, b, r, width, carry=True)
      return r & mask
    if which == 3:  # SBB
      r = a - b - (1 if self.get_flag("CF") else 0)
      # flags approx via sub of (b+cf)
      bb = b + (1 if self.get_flag("CF") else 0)
      self._sub_flags(a, bb, r, width)
      return r & mask
    if which == 4:  # AND
      r = a & b
      self._set_logic_flags(r, width)
      return r
    if which == 5:  # SUB
      r = a - b
      self._sub_flags(a, b, r, width)
      return r & mask
    if which == 6:  # XOR
      r = a ^ b
      self._set_logic_flags(r, width)
      return r
    if which == 7:  # CMP
      r = a - b
      self._sub_flags(a, b, r, width)
      return a
    raise CPUError(f"bad alu {which}")

  def _modrm_ea(
    self, modrm: int, raw: bytes, i: int, addr_width: int, seg_override: Optional[int]
  ) -> Tuple[int, Optional[Tuple[int, int]], int]:
    mod = (modrm >> 6) & 3
    reg = (modrm >> 3) & 7
    rm = modrm & 7
    if mod == 3:
      return reg, None, i
    # 16-bit addressing
    if addr_width == 2:
      disp = 0
      if mod == 1:
        disp = struct.unpack_from("<b", raw, i)[0]; i += 1
      elif mod == 2 or (mod == 0 and rm == 6):
        disp = struct.unpack_from("<h", raw, i)[0]; i += 2
        if mod == 0 and rm == 6:
          seg = seg_override if seg_override is not None else self.ds
          return reg, (seg, disp & 0xFFFF), i
      bases = {
        0: (self.ebx + self.esi),
        1: (self.ebx + self.edi),
        2: (self.ebp + self.esi),
        3: (self.ebp + self.edi),
        4: self.esi,
        5: self.edi,
        6: self.ebp,
        7: self.ebx,
      }
      off = (bases[rm] + disp) & 0xFFFF
      if rm in (2, 3, 6) and not (mod == 0 and rm == 6):
        default_ss = True
      else:
        default_ss = False
      if seg_override is not None:
        seg = seg_override
      else:
        seg = self.ss if default_ss else self.ds
      return reg, (seg, off), i
    # 32-bit addressing (simplified, no SIB fully… handle SIB)
    if rm == 4:  # SIB
      sib = raw[i]; i += 1
      scale = (sib >> 6) & 3
      index = (sib >> 3) & 7
      base = sib & 7
      disp = 0
      if mod == 1:
        disp = struct.unpack_from("<b", raw, i)[0]; i += 1
      elif mod == 2 or (mod == 0 and base == 5):
        disp = struct.unpack_from("<i", raw, i)[0]; i += 4
      base_val = 0 if (mod == 0 and base == 5) else self._gpr(base, 4)
      idx_val = 0 if index == 4 else self._gpr(index, 4)
      off = (base_val + (idx_val << scale) + disp) & 0xFFFFFFFF
      seg = seg_override if seg_override is not None else (
        self.ss if base in (4, 5) else self.ds
      )
      return reg, (seg, off & 0xFFFF), i
    disp = 0
    if mod == 1:
      disp = struct.unpack_from("<b", raw, i)[0]; i += 1
    elif mod == 2:
      disp = struct.unpack_from("<i", raw, i)[0]; i += 4
    elif mod == 0 and rm == 5:
      disp = struct.unpack_from("<i", raw, i)[0]; i += 4
      seg = seg_override if seg_override is not None else self.ds
      return reg, (seg, disp & 0xFFFF), i
    off = (self._gpr(rm, 4) + disp) & 0xFFFF
    seg = seg_override if seg_override is not None else (
      self.ss if rm == 5 else self.ds
    )
    return reg, (seg, off), i

  def _read_ea(self, ea, reg_if_reg, width, modrm) -> int:
    if ea is None:
      return self._gpr(modrm & 7, width)
    seg, off = ea
    return self.mem.read_seg(seg, off, width)

  def _write_ea(self, ea, modrm, width, value) -> None:
    if ea is None:
      self._set_gpr(modrm & 7, width, value)
    else:
      seg, off = ea
      self.mem.write_seg(seg, off, width, value)

  def _exec_modrm(self, opcode, raw, i, start_eip, op_width, addr_width, seg_override, finish):
    modrm = raw[i]; i += 1
    reg, ea, i = self._modrm_ea(modrm, raw, i, addr_width, seg_override)

    # MOV
    if opcode in (0x88, 0x89, 0x8A, 0x8B):
      width = 1 if opcode in (0x88, 0x8A) else (4 if op_width == 4 else 2)
      if opcode in (0x88, 0x89):  # r/m <- r
        val = self._gpr(reg, width)
        self._write_ea(ea, modrm, width, val)
      else:  # r <- r/m
        val = self._read_ea(ea, None, width, modrm)
        self._set_gpr(reg, width, val)
      return finish(i)

    if opcode == 0x8C:  # MOV r/m16, Sreg
      sregs = [self.es, self.cs, self.ss, self.ds, self.fs, self.gs]
      if reg > 5:
        raise CPUError("bad sreg")
      self._write_ea(ea, modrm, 2, sregs[reg])
      return finish(i)
    if opcode == 0x8E:  # MOV Sreg, r/m16
      val = self._read_ea(ea, None, 2, modrm)
      if reg == 0: self.es = val
      elif reg == 1: self.cs = val
      elif reg == 2: self.ss = val
      elif reg == 3: self.ds = val
      elif reg == 4: self.fs = val
      elif reg == 5: self.gs = val
      else: raise CPUError("bad sreg")
      return finish(i)

    # TEST
    if opcode in (0x84, 0x85):
      width = 1 if opcode == 0x84 else (4 if op_width == 4 else 2)
      a = self._read_ea(ea, None, width, modrm)
      b = self._gpr(reg, width)
      self._set_logic_flags(a & b, width)
      return finish(i)

    # XCHG
    if opcode in (0x86, 0x87):
      width = 1 if opcode == 0x86 else (4 if op_width == 4 else 2)
      a = self._read_ea(ea, None, width, modrm)
      b = self._gpr(reg, width)
      self._write_ea(ea, modrm, width, b)
      self._set_gpr(reg, width, a)
      return finish(i)

    # ALU r/m, r and r, r/m
    if opcode <= 0x3B:
      which = (opcode >> 3) & 7
      d = (opcode >> 1) & 1
      w = opcode & 1
      width = 1 if w == 0 else (4 if op_width == 4 else 2)
      rm_val = self._read_ea(ea, None, width, modrm)
      r_val = self._gpr(reg, width)
      if d == 0:  # r/m = r/m OP r
        result = self._alu_op(which, rm_val, r_val, width)
        if which != 7:
          self._write_ea(ea, modrm, width, result)
      else:
        result = self._alu_op(which, r_val, rm_val, width)
        if which != 7:
          self._set_gpr(reg, width, result)
      return finish(i)

    # group 1 imm
    if opcode in (0x80, 0x81, 0x82, 0x83):
      which = reg
      width = 1 if opcode in (0x80, 0x82) else (4 if op_width == 4 else 2)
      rm_val = self._read_ea(ea, None, width, modrm)
      if opcode == 0x81:
        if width == 4:
          imm = struct.unpack_from("<I", raw, i)[0]; i += 4
        else:
          imm = struct.unpack_from("<H", raw, i)[0]; i += 2
      elif opcode == 0x83:
        imm = struct.unpack_from("<b", raw, i)[0]; i += 1
        imm &= (1 << (8 * width)) - 1
      else:
        imm = raw[i]; i += 1
        if width != 1:
          imm = imm if imm < 0x80 else (imm | 0xFF00)  # for 82 rare
      result = self._alu_op(which, rm_val, imm, width)
      if which != 7:
        self._write_ea(ea, modrm, width, result)
      return finish(i)

    # MOV imm to r/m
    if opcode in (0xC6, 0xC7):
      width = 1 if opcode == 0xC6 else (4 if op_width == 4 else 2)
      if width == 1:
        imm = raw[i]; i += 1
      elif width == 2:
        imm = struct.unpack_from("<H", raw, i)[0]; i += 2
      else:
        imm = struct.unpack_from("<I", raw, i)[0]; i += 4
      self._write_ea(ea, modrm, width, imm)
      return finish(i)

    # POP r/m
    if opcode == 0x8F:
      val = self.pop_width(2)
      self._write_ea(ea, modrm, 2, val)
      return finish(i)

    # shifts
    if opcode in (0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3):
      width = 1 if opcode in (0xC0, 0xD0, 0xD2) else (4 if op_width == 4 else 2)
      rm_val = self._read_ea(ea, None, width, modrm)
      if opcode in (0xC0, 0xC1):
        count = raw[i]; i += 1
      elif opcode in (0xD0, 0xD1):
        count = 1
      else:
        count = self.ecx & 0xFF
      count &= 0x1F
      which = reg
      result = self._shift_op(which, rm_val, count, width)
      self._write_ea(ea, modrm, width, result)
      return finish(i)

    # F6/F7 group
    if opcode in (0xF6, 0xF7):
      width = 1 if opcode == 0xF6 else (4 if op_width == 4 else 2)
      which = reg
      rm_val = self._read_ea(ea, None, width, modrm)
      if which == 0:  # TEST
        if width == 1:
          imm = raw[i]; i += 1
        elif width == 2:
          imm = struct.unpack_from("<H", raw, i)[0]; i += 2
        else:
          imm = struct.unpack_from("<I", raw, i)[0]; i += 4
        self._set_logic_flags(rm_val & imm, width)
      elif which == 2:  # NOT
        self._write_ea(ea, modrm, width, (~rm_val) & ((1 << (8 * width)) - 1))
      elif which == 3:  # NEG
        result = self._alu_op(5, 0, rm_val, width)
        self._write_ea(ea, modrm, width, result)
        self.set_flag("CF", rm_val != 0)
      elif which == 4:  # MUL
        if width == 1:
          r = (self.eax & 0xFF) * rm_val
          self.eax = (self.eax & ~0xFFFF) | (r & 0xFFFF)
          self.set_flag("CF", r > 0xFF); self.set_flag("OF", r > 0xFF)
        elif width == 2:
          r = (self.eax & 0xFFFF) * rm_val
          self.eax = (self.eax & ~0xFFFF) | (r & 0xFFFF)
          self.edx = (self.edx & ~0xFFFF) | ((r >> 16) & 0xFFFF)
          self.set_flag("CF", r > 0xFFFF); self.set_flag("OF", r > 0xFFFF)
        else:
          r = self.eax * rm_val
          self.eax = r & 0xFFFFFFFF
          self.edx = (r >> 32) & 0xFFFFFFFF
          self.set_flag("CF", self.edx != 0); self.set_flag("OF", self.edx != 0)
      elif which == 5:  # IMUL
        # signed multiply approximate via Python int
        def sext(v, w):
          bits = 8 * w
          if v & (1 << (bits - 1)):
            return v - (1 << bits)
          return v
        if width == 1:
          r = sext(self.eax & 0xFF, 1) * sext(rm_val, 1)
          self.eax = (self.eax & ~0xFFFF) | (r & 0xFFFF)
          ok = -0x80 <= r <= 0x7F
          self.set_flag("CF", not ok); self.set_flag("OF", not ok)
        elif width == 2:
          r = sext(self.eax & 0xFFFF, 2) * sext(rm_val, 2)
          self.eax = (self.eax & ~0xFFFF) | (r & 0xFFFF)
          self.edx = (self.edx & ~0xFFFF) | ((r >> 16) & 0xFFFF)
          ok = -0x8000 <= r <= 0x7FFF
          self.set_flag("CF", not ok); self.set_flag("OF", not ok)
        else:
          r = sext(self.eax, 4) * sext(rm_val, 4)
          self.eax = r & 0xFFFFFFFF
          self.edx = (r >> 32) & 0xFFFFFFFF
          ok = -0x80000000 <= r <= 0x7FFFFFFF
          self.set_flag("CF", not ok); self.set_flag("OF", not ok)
      elif which in (6, 7):  # DIV / IDIV
        if width == 1:
          dividend = self.eax & 0xFFFF
          if rm_val == 0:
            raise CPUError("divide by zero")
          if which == 6:
            q, r = divmod(dividend, rm_val)
          else:
            def sext16(v):
              return v - 0x10000 if v & 0x8000 else v
            def sext8(v):
              return v - 0x100 if v & 0x80 else v
            q, r = divmod(sext16(dividend), sext8(rm_val))
            q &= 0xFF; r &= 0xFF
          if q > 0xFF:
            raise CPUError("divide overflow")
          self.eax = (self.eax & ~0xFFFF) | ((r & 0xFF) << 8) | (q & 0xFF)
        elif width == 2:
          dividend = ((self.edx & 0xFFFF) << 16) | (self.eax & 0xFFFF)
          if rm_val == 0:
            raise CPUError("divide by zero")
          q, r = divmod(dividend, rm_val)
          if q > 0xFFFF:
            raise CPUError("divide overflow")
          self.eax = (self.eax & ~0xFFFF) | (q & 0xFFFF)
          self.edx = (self.edx & ~0xFFFF) | (r & 0xFFFF)
        else:
          raise CPUError("32-bit DIV not implemented")
      else:
        raise CPUError(f"F6/F7 /{which}")
      return finish(i)

    # FE/FF
    if opcode in (0xFE, 0xFF):
      width = 1 if opcode == 0xFE else (4 if op_width == 4 else 2)
      which = reg
      if which in (0, 1):  # INC/DEC
        rm_val = self._read_ea(ea, None, width, modrm)
        if which == 0:
          res = (rm_val + 1) & ((1 << (8 * width)) - 1)
          self.set_flag("AF", (rm_val & 0xF) == 0xF)
          self.set_flag("OF", rm_val == (0x7F if width == 1 else 0x7FFF if width == 2 else 0x7FFFFFFF))
        else:
          res = (rm_val - 1) & ((1 << (8 * width)) - 1)
          self.set_flag("AF", (rm_val & 0xF) == 0)
          self.set_flag("OF", rm_val == (0x80 if width == 1 else 0x8000 if width == 2 else 0x80000000))
        self._set_szp(res, width)
        self._write_ea(ea, modrm, width, res)
        return finish(i)
      if opcode == 0xFF and which == 2:  # CALL near r/m
        target = self._read_ea(ea, None, 2, modrm)
        next_ip = (start_eip + (i - 0)) & 0xFFFF
        # i already past modrm; compute consumed
        self.eip = (start_eip + i) & 0xFFFF  # temp
        # Actually next_ip should be after this insn
        consumed_so_far = i
        next_ip = (start_eip + consumed_so_far) & 0xFFFF
        self.push_width(next_ip, 2)
        self.eip = target & 0xFFFF
        return i
      if opcode == 0xFF and which == 3:  # CALL far m16:16
        if ea is None:
          raise CPUError("CALL far reg")
        seg, off = ea
        nip = self.mem.read_seg(seg, off, 2)
        ncs = self.mem.read_seg(seg, (off + 2) & 0xFFFF, 2)
        next_ip = (start_eip + i) & 0xFFFF
        self.push_width(self.cs, 2)
        self.push_width(next_ip, 2)
        self.cs = ncs
        self.eip = nip
        return i
      if opcode == 0xFF and which == 4:  # JMP near r/m
        self.eip = self._read_ea(ea, None, 2, modrm) & 0xFFFF
        return i
      if opcode == 0xFF and which == 5:  # JMP far
        if ea is None:
          raise CPUError("JMP far reg")
        seg, off = ea
        self.eip = self.mem.read_seg(seg, off, 2)
        self.cs = self.mem.read_seg(seg, (off + 2) & 0xFFFF, 2)
        return i
      if opcode == 0xFF and which == 6:  # PUSH r/m
        val = self._read_ea(ea, None, 2, modrm)
        self.push_width(val, 2)
        return finish(i)
      raise CPUError(f"FF /{which}")

    # IMUL
    if opcode in (0x69, 0x6B):
      width = 4 if op_width == 4 else 2
      rm_val = self._read_ea(ea, None, width, modrm)
      if opcode == 0x6B:
        imm = struct.unpack_from("<b", raw, i)[0]; i += 1
      elif width == 2:
        imm = struct.unpack_from("<h", raw, i)[0]; i += 2
      else:
        imm = struct.unpack_from("<i", raw, i)[0]; i += 4
      def sext(v, w):
        bits = 8 * w
        return v - (1 << bits) if v & (1 << (bits - 1)) else v
      r = sext(rm_val, width) * imm
      self._set_gpr(reg, width, r & ((1 << (8 * width)) - 1))
      ok = (-(1 << (8 * width - 1)) <= r < (1 << (8 * width - 1)))
      self.set_flag("CF", not ok); self.set_flag("OF", not ok)
      return finish(i)

    raise CPUError(f"unhandled modrm opcode {opcode:02x}")

  def _shift_op(self, which: int, val: int, count: int, width: int) -> int:
    if count == 0:
      return val
    bits = width * 8
    mask = (1 << bits) - 1
    val &= mask
    if which == 0:  # ROL
      for _ in range(count):
        cf = (val >> (bits - 1)) & 1
        val = ((val << 1) | cf) & mask
        self.set_flag("CF", bool(cf))
      self.set_flag("OF", bool(((val >> (bits - 1)) ^ (1 if self.get_flag("CF") else 0)) & 1)) if count == 1 else None
      return val
    if which == 1:  # ROR
      for _ in range(count):
        cf = val & 1
        val = ((val >> 1) | (cf << (bits - 1))) & mask
        self.set_flag("CF", bool(cf))
      return val
    if which == 2:  # RCL
      for _ in range(count):
        cf_in = 1 if self.get_flag("CF") else 0
        cf_out = (val >> (bits - 1)) & 1
        val = ((val << 1) | cf_in) & mask
        self.set_flag("CF", bool(cf_out))
      return val
    if which == 3:  # RCR
      for _ in range(count):
        cf_in = 1 if self.get_flag("CF") else 0
        cf_out = val & 1
        val = ((val >> 1) | (cf_in << (bits - 1))) & mask
        self.set_flag("CF", bool(cf_out))
      return val
    if which == 4:  # SHL/SAL
      for _ in range(count):
        self.set_flag("CF", bool((val >> (bits - 1)) & 1))
        val = (val << 1) & mask
      self._set_szp(val, width)
      return val
    if which == 5:  # SHR
      for _ in range(count):
        self.set_flag("CF", bool(val & 1))
        val = (val >> 1) & mask
      self._set_szp(val, width)
      self.set_flag("OF", False)
      return val
    if which == 7:  # SAR
      sign = (val >> (bits - 1)) & 1
      for _ in range(count):
        self.set_flag("CF", bool(val & 1))
        val = ((val >> 1) | (sign << (bits - 1))) & mask
      self._set_szp(val, width)
      return val
    raise CPUError(f"shift /{which}")

  def _string_op(self, opcode, raw, i, start_eip, op_width, rep, seg_override, finish):
    width = 1 if opcode in (0xA4, 0xA6, 0xAA, 0xAC, 0xAE) else (4 if op_width == 4 else 2)
    df = -width if self.get_flag("DF") else width
    src_seg = seg_override if seg_override is not None else self.ds

    def once() -> Optional[bool]:
      """Return True/False for scas/cmps ZF relevance; None otherwise."""
      if opcode in (0xA4, 0xA5):  # MOVS
        val = self.mem.read_seg(src_seg, self.esi & 0xFFFF, width)
        self.mem.write_seg(self.es, self.edi & 0xFFFF, width, val)
        self.esi = (self.esi + df) & 0xFFFFFFFF
        self.edi = (self.edi + df) & 0xFFFFFFFF
      elif opcode in (0xAA, 0xAB):  # STOS
        val = self.eax if width == 4 else (self.eax & ((1 << (8 * width)) - 1))
        self.mem.write_seg(self.es, self.edi & 0xFFFF, width, val)
        self.edi = (self.edi + df) & 0xFFFFFFFF
      elif opcode in (0xAC, 0xAD):  # LODS
        val = self.mem.read_seg(src_seg, self.esi & 0xFFFF, width)
        if width == 1:
          self.eax = (self.eax & ~0xFF) | val
        elif width == 2:
          self.eax = (self.eax & ~0xFFFF) | val
        else:
          self.eax = val
        self.esi = (self.esi + df) & 0xFFFFFFFF
      elif opcode in (0xAE, 0xAF):  # SCAS
        val = self.mem.read_seg(self.es, self.edi & 0xFFFF, width)
        a = self.eax & ((1 << (8 * width)) - 1)
        self._alu_op(7, a, val, width)
        self.edi = (self.edi + df) & 0xFFFFFFFF
        return self.get_flag("ZF")
      elif opcode in (0xA6, 0xA7):  # CMPS
        a = self.mem.read_seg(src_seg, self.esi & 0xFFFF, width)
        b = self.mem.read_seg(self.es, self.edi & 0xFFFF, width)
        self._alu_op(7, a, b, width)
        self.esi = (self.esi + df) & 0xFFFFFFFF
        self.edi = (self.edi + df) & 0xFFFFFFFF
        return self.get_flag("ZF")
      return None

    if rep == 0:
      once()
      return finish(i)

    # REP prefix: do not advance IP until CX exhausted / condition
    # Instruction includes prefixes — consumed length is i (opcode only past prefixes)
    # Actually i points past opcode; total from start includes prefixes.
    # For REP we keep EIP at start until done.
    count_guard = 0
    while (self.ecx & 0xFFFF) != 0:
      z = once()
      self.ecx = (self.ecx & ~0xFFFF) | ((self.ecx - 1) & 0xFFFF)
      count_guard += 1
      if count_guard > 1_000_000:
        raise CPUError("REP runaway")
      if opcode in (0xA6, 0xA7, 0xAE, 0xAF):
        if rep == 1 and not z:  # REPE: stop if not equal
          break
        if rep == 2 and z:  # REPNE: stop if equal
          break
    return finish(i)


_REG32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
