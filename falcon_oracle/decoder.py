"""Coverage-driven Falcon decoder for the PMU bootstrap pad subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import DecodeError, InvalidIMEMAccess, UnsupportedInstruction
from .instruction import DecodedInstruction
from .state import FalconMemory, SR_NAMES
from .values import sign_extend, u32


SIZE_NAMES = {0: "b8", 1: "b16", 2: "b32"}


def _bits(word: int, lo: int, width: int) -> int:
    return (word >> lo) & ((1 << width) - 1)


def _read_word(imem: FalconMemory | bytes, pc: int, size: int) -> tuple[bytes, int]:
    if isinstance(imem, FalconMemory):
        if pc < imem.base or pc + size > imem.end:
            raise InvalidIMEMAccess(
                "instruction fetch past IMEM",
                pc=pc,
                details={"need": size, "imem_end": imem.end},
            )
        raw = imem.read_bytes(pc, size)
    else:
        # bytes image addressed from 0; caller maps pc.
        raise DecodeError("raw bytes decode requires FalconMemory with base", pc=pc)
    word = int.from_bytes(raw.ljust(4, b"\x00")[:4], "little")
    return raw, word


@dataclass
class FalconDecoder:
    """Strict decoder supporting only forms present in the MVP corpus."""

    def decode_one(self, imem: FalconMemory, pc: int) -> DecodedInstruction:
        if pc < imem.base or pc >= imem.end:
            raise InvalidIMEMAccess("PC outside IMEM", pc=pc, details={"imem_end": imem.end})

        # Peek first byte to classify.
        first = imem.read_u8(pc)
        size_code = first >> 6
        low6 = first & 0x3F

        # Unsized class (size_code == 3) and selected sized forms used by pad.
        if first in (0xF0, 0xF1):
            return self._decode_f0_f1(imem, pc, first)
        if first == 0xF4:
            return self._decode_f4(imem, pc)
        if first == 0xF8:
            return self._decode_f8(imem, pc)
        if first == 0xFA:
            return self._decode_fa(imem, pc)
        if first == 0xFE:
            return self._decode_fe(imem, pc)
        if first in (0xCF,):
            return self._decode_iord(imem, pc)
        if first in (0xD0,):
            return self._decode_iowr(imem, pc)

        if size_code in (0, 1, 2):
            return self._decode_sized(imem, pc, first, size_code, low6)

        raise UnsupportedInstruction(
            "unsupported opcode",
            pc=pc,
            raw=bytes((first,)),
            details={"opcode": first},
        )

    def decode_all(self, imem: FalconMemory) -> list[DecodedInstruction]:
        out: list[DecodedInstruction] = []
        pc = imem.base
        while pc < imem.end:
            insn = self.decode_one(imem, pc)
            out.append(insn)
            pc += insn.size
        return out

    def _decode_f0_f1(self, imem: FalconMemory, pc: int, first: int) -> DecodedInstruction:
        size = 3 if first == 0xF0 else 4
        raw, word = _read_word(imem, pc, size)
        reg = _bits(word, 12, 4)  # REG1 field used as DST for OL forms
        # For f0/f1 OL forms, DST is in REG1 bits (12-15) per envydis REG1 on ol0.
        # Observed encodings put DST in high nibble of byte1 = bits 12-15.
        # Actually for mov/sethi: byte1 high = reg. Confirm with f0 43 00 → reg4.
        # bits 12-15 of 0x0043f0 = 4. Yes REG1.
        # Wait f0 43 00 LE word low24 = 0x0043f0, bits12-15=4. Good.
        # But also low nibble bits8-11 = subop for OL? bits8-11 of 0x43f0 = 3 = sethi.
        subop = _bits(word, 8, 4)
        # For I8/I16 the imm starts at bit 16.
        if first == 0xF0:
            imm = _bits(word, 16, 8)
        else:
            imm = _bits(word, 16, 16)

        if subop == 7:  # mov
            simm = sign_extend(imm, 8 if first == 0xF0 else 16)
            return DecodedInstruction(
                pc=pc,
                size=size,
                raw=raw,
                mnemonic="mov",
                operands=(reg, u32(simm)),
                metadata={"form": f"mov_imm{8 if first==0xF0 else 16}", "dst": reg, "imm": u32(simm)},
            )
        if subop == 3:  # sethi
            # sethi immediate is zero-extended, shifted << 16 at execute
            return DecodedInstruction(
                pc=pc,
                size=size,
                raw=raw,
                mnemonic="sethi",
                operands=(reg, imm),
                metadata={"form": f"sethi_imm{8 if first==0xF0 else 16}", "dst": reg, "imm": imm},
            )
        if subop == 4:  # and
            return DecodedInstruction(
                pc=pc,
                size=size,
                raw=raw,
                mnemonic="and",
                operands=(reg, imm),
                metadata={"form": f"and_imm{8 if first==0xF0 else 16}", "dst": reg, "imm": imm},
            )
        raise UnsupportedInstruction(
            "unsupported f0/f1 subop",
            pc=pc,
            raw=raw,
            details={"subop": subop},
        )

    def _decode_f4(self, imem: FalconMemory, pc: int) -> DecodedInstruction:
        raw, word = _read_word(imem, pc, 3)
        subop = _bits(word, 8, 8)
        imm8 = _bits(word, 16, 8)
        if subop == 0x0E:
            target = pc + sign_extend(imm8, 8)
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="bra",
                operands=("always", target),
                branch_target=target,
                metadata={"form": "bra_always", "cc": "always", "diff": sign_extend(imm8, 8)},
            )
        if subop in (0x1B,):  # ne/nz
            target = pc + sign_extend(imm8, 8)
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="bra",
                operands=("ne", target),
                branch_target=target,
                metadata={"form": "bra_ne", "cc": "ne", "diff": sign_extend(imm8, 8)},
            )
        if subop == 0x21:  # call abs8
            target = imm8
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="call",
                operands=(target,),
                branch_target=target,
                metadata={"form": "call_abs8"},
            )
        if subop == 0x20:  # jmp abs8
            target = imm8
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="jmp",
                operands=(target,),
                branch_target=target,
                metadata={"form": "jmp_abs8"},
            )
        raise UnsupportedInstruction("unsupported f4 subop", pc=pc, raw=raw, details={"subop": subop})

    def _decode_f8(self, imem: FalconMemory, pc: int) -> DecodedInstruction:
        raw, word = _read_word(imem, pc, 2)
        subop = _bits(word, 8, 4)
        if subop == 3:
            return DecodedInstruction(
                pc=pc,
                size=2,
                raw=raw,
                mnemonic="xdwait",
                operands=(),
                metadata={"form": "xdwait"},
            )
        if subop == 0:
            return DecodedInstruction(
                pc=pc,
                size=2,
                raw=raw,
                mnemonic="ret",
                operands=(),
                metadata={"form": "ret"},
            )
        raise UnsupportedInstruction("unsupported f8 subop", pc=pc, raw=raw, details={"subop": subop})

    def _decode_fa(self, imem: FalconMemory, pc: int) -> DecodedInstruction:
        raw, word = _read_word(imem, pc, 3)
        subop = _bits(word, 16, 4)
        reg1 = _bits(word, 12, 4)
        reg2 = _bits(word, 8, 4)
        if subop == 6:
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="xdst",
                operands=(reg1, reg2),
                metadata={"form": "xdst", "src1": reg1, "src2": reg2},
            )
        if subop == 0:
            # iowr I[R], R form without immediate index
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="iowr",
                operands=(reg1, 0, reg2),
                metadata={"form": "iowr_fa", "base": reg1, "idx": 0, "src": reg2},
            )
        raise UnsupportedInstruction("unsupported fa subop", pc=pc, raw=raw, details={"subop": subop})

    def _decode_fe(self, imem: FalconMemory, pc: int) -> DecodedInstruction:
        raw, word = _read_word(imem, pc, 3)
        subop = _bits(word, 16, 4)
        reg1 = _bits(word, 12, 4)
        sreg2 = _bits(word, 8, 4)
        if subop == 0:
            name = SR_NAMES.get(sreg2, f"sr{sreg2}")
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="mov",
                operands=(("sreg", sreg2, name), ("reg", reg1)),
                metadata={"form": "mov_to_sreg", "sreg": sreg2, "src": reg1},
            )
        if subop == 1:
            name = SR_NAMES.get(_bits(word, 12, 4), f"sr{_bits(word,12,4)}")
            # mov REG2, SREG1 — SREG1 uses reg1_bf bits12, REG2 uses bits8
            sreg1 = _bits(word, 12, 4)
            reg2 = _bits(word, 8, 4)
            name = SR_NAMES.get(sreg1, f"sr{sreg1}")
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="mov",
                operands=(("reg", reg2), ("sreg", sreg1, name)),
                metadata={"form": "mov_from_sreg", "dst": reg2, "sreg": sreg1},
            )
        raise UnsupportedInstruction("unsupported fe subop", pc=pc, raw=raw, details={"subop": subop})

    def _decode_iord(self, imem: FalconMemory, pc: int) -> DecodedInstruction:
        raw, word = _read_word(imem, pc, 3)
        # cf: iord REG2, I[REG1 + off*4]
        dst = _bits(word, 8, 4)
        base = _bits(word, 12, 4)
        idx = _bits(word, 16, 8)
        return DecodedInstruction(
            pc=pc,
            size=3,
            raw=raw,
            mnemonic="iord",
            operands=(dst, base, idx),
            metadata={"form": "iord", "dst": dst, "base": base, "idx": idx},
        )

    def _decode_iowr(self, imem: FalconMemory, pc: int) -> DecodedInstruction:
        raw, word = _read_word(imem, pc, 3)
        # d0: iowr I[REG1 + off*4], REG2
        src = _bits(word, 8, 4)
        base = _bits(word, 12, 4)
        idx = _bits(word, 16, 8)
        return DecodedInstruction(
            pc=pc,
            size=3,
            raw=raw,
            mnemonic="iowr",
            operands=(base, idx, src),
            metadata={"form": "iowr", "base": base, "idx": idx, "src": src},
        )

    def _decode_sized(
        self,
        imem: FalconMemory,
        pc: int,
        first: int,
        size_code: int,
        low6: int,
    ) -> DecodedInstruction:
        sz_name = SIZE_NAMES[size_code]
        # Form 0x: st — opcode low4 = subop, size in high2. first & 0x0f for form0?
        # byte0 = (size<<6) | form_low6
        if low6 == 0x00:
            # st: T(datari)=D[REG1+off], REG2=src
            raw, word = _read_word(imem, pc, 3)
            src = _bits(word, 8, 4)
            base = _bits(word, 12, 4)
            off = _bits(word, 16, 8)
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="st",
                operands=(sz_name, base, off, src),
                metadata={"form": "st_ri", "size": sz_name, "base": base, "off": off, "src": src},
            )
        if low6 == 0x18:
            # ld: REG2=dst, T(datari)=D[REG1+off]
            raw, word = _read_word(imem, pc, 3)
            dst = _bits(word, 8, 4)
            base = _bits(word, 12, 4)
            off = _bits(word, 16, 8)
            return DecodedInstruction(
                pc=pc,
                size=3,
                raw=raw,
                mnemonic="ld",
                operands=(sz_name, dst, base, off),
                metadata={"form": "ld_ri", "size": sz_name, "dst": dst, "base": base, "off": off},
            )
        if low6 == 0x3B:
            # sub etc: OP3B N("sub"), REG1, REG2  => dst=REG1, src=REG2
            raw, word = _read_word(imem, pc, 3)
            dst = _bits(word, 12, 4)
            src = _bits(word, 8, 4)
            subop = _bits(word, 16, 4)
            if subop == 2:
                return DecodedInstruction(
                    pc=pc,
                    size=3,
                    raw=raw,
                    mnemonic="sub",
                    operands=(sz_name, dst, src),
                    metadata={"form": "sub_rr", "size": sz_name, "dst": dst, "src": src},
                )
            raise UnsupportedInstruction("unsupported 3b subop", pc=pc, raw=raw, details={"subop": subop})
        if low6 == 0x36:
            # R2SD I8
            raw, word = _read_word(imem, pc, 3)
            dst = _bits(word, 12, 4)
            subop = _bits(word, 8, 4)
            imm = _bits(word, 16, 8)
            if subop == 2:
                return DecodedInstruction(
                    pc=pc,
                    size=3,
                    raw=raw,
                    mnemonic="sub",
                    operands=(sz_name, dst, imm),
                    metadata={"form": "sub_ri", "size": sz_name, "dst": dst, "imm": imm},
                )
            raise UnsupportedInstruction("unsupported 36 subop", pc=pc, raw=raw, details={"subop": subop})
        if low6 == 0x3D:
            raw, word = _read_word(imem, pc, 2)
            dst = _bits(word, 12, 4)
            subop = _bits(word, 8, 4)
            if subop == 4:
                return DecodedInstruction(
                    pc=pc,
                    size=2,
                    raw=raw,
                    mnemonic="clear",
                    operands=(sz_name, dst),
                    metadata={"form": "clear", "size": sz_name, "dst": dst},
                )
            raise UnsupportedInstruction("unsupported 3d subop", pc=pc, raw=raw, details={"subop": subop})

        raise UnsupportedInstruction(
            "unsupported sized form",
            pc=pc,
            raw=bytes((first,)),
            details={"low6": low6, "size_code": size_code},
        )


def coverage_report(imem: FalconMemory, instructions: Iterable[DecodedInstruction]) -> dict:
    decoded_bytes = sum(i.size for i in instructions)
    forms = sorted({i.form for i in instructions})
    return {
        "image_bytes": imem.size,
        "decoded_bytes": decoded_bytes,
        "instruction_instances": sum(1 for _ in instructions),
        "unique_forms": forms,
        "unique_form_count": len(forms),
        "unknown_opcodes": 0 if decoded_bytes == imem.size else 1,
        "trailing_undecoded_bytes": imem.size - decoded_bytes,
        "complete": decoded_bytes == imem.size,
    }
