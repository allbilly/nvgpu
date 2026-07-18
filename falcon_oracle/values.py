"""32-bit Falcon value helpers."""

from __future__ import annotations

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
MASK8 = 0xFF


def u32(value: int) -> int:
    return int(value) & MASK32


def u16(value: int) -> int:
    return int(value) & MASK16


def u8(value: int) -> int:
    return int(value) & MASK8


def sign_extend(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def as_signed32(value: int) -> int:
    value = u32(value)
    return value - 0x100000000 if value & 0x80000000 else value


def parse_int(text: str | int) -> int:
    if isinstance(text, int):
        return text
    text = text.strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    return int(text, 10)


def fmt_u32(value: int) -> str:
    return f"0x{u32(value):08x}"
