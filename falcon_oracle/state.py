"""Falcon CPU state and bounds-checked memory."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .errors import InvalidAlignment, InvalidDMEMAccess, InvalidIMEMAccess, InvalidRegister
from .values import MASK32, u32


@dataclass
class FalconMemory:
    """Little-endian Falcon local memory with strict bounds."""

    data: bytearray
    name: str = "mem"
    base: int = 0
    allow_unaligned: bool = True

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def end(self) -> int:
        return self.base + self.size

    def _check(self, offset: int, width: int, *, write: bool) -> int:
        if offset < self.base or offset + width > self.end:
            exc = InvalidIMEMAccess if self.name == "imem" else InvalidDMEMAccess
            raise exc(
                f"{self.name} {'write' if write else 'read'} out of range",
                details={
                    "offset": offset,
                    "width": width,
                    "base": self.base,
                    "end": self.end,
                },
            )
        return offset - self.base

    def read_bytes(self, offset: int, size: int) -> bytes:
        idx = self._check(offset, size, write=False)
        return bytes(self.data[idx : idx + size])

    def write_bytes(self, offset: int, data: bytes) -> None:
        idx = self._check(offset, len(data), write=True)
        self.data[idx : idx + len(data)] = data

    def read_u8(self, offset: int) -> int:
        return self.read_bytes(offset, 1)[0]

    def read_u16(self, offset: int) -> int:
        if not self.allow_unaligned and (offset & 1):
            raise InvalidAlignment("unaligned u16 read", details={"offset": offset})
        return int.from_bytes(self.read_bytes(offset & ~1 if self.name == "dmem" else offset, 2), "little")

    def read_u32(self, offset: int) -> int:
        # Falcon DMEM LD/ST force alignment on address for sized accesses.
        addr = offset & ~3 if self.name == "dmem" else offset
        if not self.allow_unaligned and (offset & 3):
            raise InvalidAlignment("unaligned u32 read", details={"offset": offset})
        return int.from_bytes(self.read_bytes(addr, 4), "little")

    def write_u8(self, offset: int, value: int) -> None:
        self.write_bytes(offset, bytes((value & 0xFF,)))

    def write_u16(self, offset: int, value: int) -> None:
        addr = offset & ~1 if self.name == "dmem" else offset
        self.write_bytes(addr, int(value & 0xFFFF).to_bytes(2, "little"))

    def write_u32(self, offset: int, value: int) -> None:
        addr = offset & ~3 if self.name == "dmem" else offset
        self.write_bytes(addr, int(value & MASK32).to_bytes(4, "little"))

    def snapshot(self) -> "FalconMemory":
        return FalconMemory(
            data=bytearray(self.data),
            name=self.name,
            base=self.base,
            allow_unaligned=self.allow_unaligned,
        )


SR_NAMES = {
    0: "iv0",
    1: "iv1",
    3: "tv",
    4: "sp",
    5: "pc",
    6: "xcbase",
    7: "xdbase",
    8: "flags",
    11: "xtargets",
    12: "tstatus",
}


@dataclass
class FalconState:
    pc: int
    registers: list[int] = field(default_factory=lambda: [0] * 16)
    specials: dict[int, int] = field(default_factory=dict)

    carry: bool = False
    overflow: bool = False
    sign: bool = False
    zero: bool = False
    predicates: list[bool] = field(default_factory=lambda: [False] * 8)

    imem: FalconMemory = field(default_factory=lambda: FalconMemory(bytearray(), name="imem"))
    dmem: FalconMemory = field(default_factory=lambda: FalconMemory(bytearray(4096), name="dmem"))

    instruction_count: int = 0
    logical_step: int = 0
    status: str = "running"
    fault_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.registers) != 16:
            raise InvalidRegister(
                "Falcon has 16 GPRs",
                details={"count": len(self.registers)},
            )
        self.registers = [u32(r) for r in self.registers]
        if 4 not in self.specials:
            self.specials[4] = 0x0FC0  # default $sp near top of 4KiB DMEM
        if 7 not in self.specials:
            self.specials[7] = 0  # $xdbase
        if 6 not in self.specials:
            self.specials[6] = 0  # $xcbase
        if 11 not in self.specials:
            self.specials[11] = 0  # $xtargets

    def get_reg(self, idx: int) -> int:
        if not 0 <= idx < 16:
            raise InvalidRegister("bad GPR", details={"index": idx})
        return self.registers[idx]

    def set_reg(self, idx: int, value: int) -> None:
        if not 0 <= idx < 16:
            raise InvalidRegister("bad GPR", details={"index": idx})
        self.registers[idx] = u32(value)

    @property
    def sp(self) -> int:
        return u32(self.specials.get(4, 0)) & ~3

    @sp.setter
    def sp(self, value: int) -> None:
        self.specials[4] = u32(value) & ~3

    @property
    def xdbase(self) -> int:
        return u32(self.specials.get(7, 0))

    @xdbase.setter
    def xdbase(self, value: int) -> None:
        self.specials[7] = u32(value)

    @property
    def xtargets(self) -> int:
        return u32(self.specials.get(11, 0))

    def get_sreg(self, idx: int) -> int:
        if idx == 5:
            return u32(self.pc)
        if idx == 8:
            return self.flags_word()
        if idx == 4:
            return self.sp
        return u32(self.specials.get(idx, 0))

    def set_sreg(self, idx: int, value: int) -> None:
        value = u32(value)
        if idx == 5:
            raise InvalidRegister("$pc is read-only via sreg write")
        if idx == 8:
            self.set_flags_word(value)
            return
        if idx == 4:
            self.sp = value
            return
        self.specials[idx] = value

    def flags_word(self) -> int:
        word = 0
        for i, p in enumerate(self.predicates):
            if p:
                word |= 1 << i
        if self.carry:
            word |= 1 << 8
        if self.overflow:
            word |= 1 << 9
        if self.sign:
            word |= 1 << 10
        if self.zero:
            word |= 1 << 11
        return word

    def set_flags_word(self, value: int) -> None:
        value = u32(value)
        self.predicates = [bool(value & (1 << i)) for i in range(8)]
        self.carry = bool(value & (1 << 8))
        self.overflow = bool(value & (1 << 9))
        self.sign = bool(value & (1 << 10))
        self.zero = bool(value & (1 << 11))

    def set_arith_flags(self, result: int, *, carry: bool | None = None, overflow: bool | None = None) -> None:
        result = u32(result)
        self.sign = bool(result & 0x80000000)
        self.zero = result == 0
        if carry is not None:
            self.carry = carry
        if overflow is not None:
            self.overflow = overflow

    def snapshot(self) -> "FalconState":
        return FalconState(
            pc=self.pc,
            registers=list(self.registers),
            specials=dict(self.specials),
            carry=self.carry,
            overflow=self.overflow,
            sign=self.sign,
            zero=self.zero,
            predicates=list(self.predicates),
            imem=self.imem.snapshot(),
            dmem=self.dmem.snapshot(),
            instruction_count=self.instruction_count,
            logical_step=self.logical_step,
            status=self.status,
            fault_reason=self.fault_reason,
        )

    def restore(self, other: "FalconState") -> None:
        self.pc = other.pc
        self.registers = list(other.registers)
        self.specials = dict(other.specials)
        self.carry = other.carry
        self.overflow = other.overflow
        self.sign = other.sign
        self.zero = other.zero
        self.predicates = list(other.predicates)
        self.imem = other.imem.snapshot()
        self.dmem = other.dmem.snapshot()
        self.instruction_count = other.instruction_count
        self.logical_step = other.logical_step
        self.status = other.status
        self.fault_reason = other.fault_reason

    def canonical_dict(self) -> dict:
        return {
            "pc": self.pc,
            "registers": list(self.registers),
            "specials": {str(k): v for k, v in sorted(self.specials.items())},
            "carry": self.carry,
            "overflow": self.overflow,
            "sign": self.sign,
            "zero": self.zero,
            "predicates": list(self.predicates),
            "instruction_count": self.instruction_count,
            "logical_step": self.logical_step,
            "status": self.status,
            "fault_reason": self.fault_reason,
            "dmem_sha256": __import__("hashlib").sha256(bytes(self.dmem.data)).hexdigest(),
        }
