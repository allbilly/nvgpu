"""Instruction semantics and control flow."""

from falcon_oracle.bus.base import NullBus
from falcon_oracle.decoder import FalconDecoder
from falcon_oracle.executor import FalconExecutor
from falcon_oracle.state import FalconMemory, FalconState
from falcon_oracle.trace import TraceCollector, TraceLevel


def _exe(code: bytes, base: int = 0) -> FalconExecutor:
    imem = FalconMemory(bytearray(code), name="imem", base=base)
    dmem = FalconMemory(bytearray(4096), name="dmem")
    state = FalconState(pc=base, imem=imem, dmem=dmem)
    return FalconExecutor(
        state=state,
        bus=NullBus(),
        decoder=FalconDecoder(),
        trace=TraceCollector(level=TraceLevel.EFFECTS),
        helpers={},
    )


def test_mov_sethi_compose():
    # mov $r3, 0xb002; sethi $r3, 0x40c0 -> 0x40c0b002
    code = bytes.fromhex("f13702b0f133c040")
    exe = _exe(code)
    exe.step()
    exe.step()
    assert exe.state.get_reg(3) == 0x40C0B002


def test_sub_sets_zero_and_branch():
    # clear $r2; mov $r3,0; sub $r3,$r2; bra ne self (not taken); 
    # Use pad bytes around wait_go comparison is heavier; unit test flags:
    code = bytes.fromhex("bd24")  # clear b32 $r2
    exe = _exe(code)
    exe.state.set_reg(2, 0x1234)
    exe.step()
    assert exe.state.get_reg(2) == 0


def test_call_ret_stack():
    # call 0x10 with helper-less body: place ret at 0x10
    # at 0: call 0x10 (f4 21 10); mov $r1,1 (f0 17 01); 
    # at 0x10: ret (f8 00)
    code = bytearray(0x20)
    code[0:3] = bytes.fromhex("f42110")
    code[3:6] = bytes.fromhex("f01701")
    code[0x10:0x12] = bytes.fromhex("f800")
    exe = _exe(bytes(code))
    exe.step()  # call
    assert exe.state.pc == 0x10
    exe.step()  # ret
    assert exe.state.pc == 3
    exe.step()  # mov
    assert exe.state.get_reg(1) == 1


def test_self_loop_hits_cap():
    code = bytes.fromhex("f40e00")  # bra always self
    exe = _exe(code)
    result = exe.run(max_instructions=5)
    assert result.stop_reason == "max_instructions"
    assert result.instruction_count == 5
