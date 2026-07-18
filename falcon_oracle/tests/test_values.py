"""Level 0: values and memory."""

from falcon_oracle.state import FalconMemory
from falcon_oracle.values import sign_extend, u32


def test_u32_mask():
    assert u32(-1) == 0xFFFFFFFF
    assert u32(0x1_0000_0001) == 1


def test_sign_extend():
    assert sign_extend(0xFA, 8) == -6
    assert sign_extend(0xB002, 16) == -0x4FFE


def test_memory_bounds():
    mem = FalconMemory(bytearray(16), name="dmem", base=0)
    mem.write_u32(0, 0xA1B2C3D4)
    assert mem.read_u32(0) == 0xA1B2C3D4
    try:
        mem.read_u32(16)
        assert False, "expected fault"
    except Exception as exc:
        assert "out of range" in str(exc)


def test_imem_base_window():
    mem = FalconMemory(bytearray(b"\x01\x02\x03\x04"), name="imem", base=0xB14)
    assert mem.read_u8(0xB14) == 0x01
    assert mem.read_bytes(0xB14, 4) == b"\x01\x02\x03\x04"
