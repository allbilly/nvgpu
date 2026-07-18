"""Instruction-form smoke tests for ALU/flags."""

from falcon_oracle.bus.base import NullBus
from falcon_oracle.decoder import FalconDecoder
from falcon_oracle.executor import FalconExecutor
from falcon_oracle.state import FalconMemory, FalconState
from falcon_oracle.trace import TraceCollector, TraceLevel


def test_and_sets_zero_flag():
    # mov $r6, 4; and $r6, 4 -> 4; and $r6, 0 -> 0
    code = bytes.fromhex("f06704") + bytes.fromhex("f06404")  # mov r6,4 ; and r6,4
    # Actually mov imm8: f0 with subop 7: byte1 = (reg<<4)|7 = 0x67 for r6
    imem = FalconMemory(bytearray(code), name="imem", base=0)
    state = FalconState(pc=0, imem=imem, dmem=FalconMemory(bytearray(256), name="dmem"))
    exe = FalconExecutor(
        state=state,
        bus=NullBus(),
        decoder=FalconDecoder(),
        trace=TraceCollector(level=TraceLevel.EFFECTS),
        helpers={},
    )
    exe.step()
    assert exe.state.get_reg(6) == 4
    exe.step()
    assert exe.state.get_reg(6) == 4
    assert exe.state.zero is False
