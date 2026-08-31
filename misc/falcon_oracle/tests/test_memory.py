"""Memory bounds tests."""

from falcon_oracle.errors import InvalidDMEMAccess, InvalidIMEMAccess
from falcon_oracle.state import FalconMemory
import pytest


def test_dmem_oob_write():
    mem = FalconMemory(bytearray(8), name="dmem")
    with pytest.raises(InvalidDMEMAccess):
        mem.write_u32(8, 1)


def test_imem_oob_read():
    mem = FalconMemory(bytearray(4), name="imem", base=0x100)
    with pytest.raises(InvalidIMEMAccess):
        mem.read_u8(0x104)
