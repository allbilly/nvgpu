"""Historical $r0 clobber regression (progress.md / macros.fuc nv_iowr)."""

from falcon_oracle.bus.gk104_fake_adapter import GK104FakeBus
from falcon_oracle.devices.pmu_mmio import PmuMmioWindow
from falcon_oracle.executor import FalconExecutor, nv_wr32_helper
from falcon_oracle.state import FalconMemory, FalconState
from falcon_oracle.trace import TraceCollector, TraceLevel


def test_nv_wr32_clears_r0():
    imem = FalconMemory(bytearray(4), name="imem", base=0)
    dmem = FalconMemory(bytearray(4096), name="dmem")
    state = FalconState(pc=0x34, imem=imem, dmem=dmem)
    state.set_reg(0, 0xABCD)
    state.set_reg(14, 0x1000)
    state.set_reg(13, 0x55)
    state.sp = 0x0FC0
    state.dmem.write_u32(state.sp, 0x100)

    inner = GK104FakeBus(trace=TraceCollector(level=TraceLevel.EXTERNAL))
    bus = PmuMmioWindow(inner=inner)
    exe = FalconExecutor(
        state=state,
        bus=bus,
        helpers={0x34: nv_wr32_helper},
        trace=TraceCollector(level=TraceLevel.EFFECTS),
    )
    exe.step()
    assert exe.state.get_reg(0) == 0
    assert bus.host_reads.get(0x1000) == 0x55
    assert exe.state.pc == 0x100


def test_bad_loop_counter_in_r0_corrupts():
    imem = FalconMemory(bytearray(4), name="imem", base=0)
    dmem = FalconMemory(bytearray(4096), name="dmem")
    state = FalconState(pc=0x34, imem=imem, dmem=dmem)
    state.set_reg(0, 5)
    state.set_reg(14, 0x2000)
    state.set_reg(13, 1)
    state.sp = 0x0FC0
    state.dmem.write_u32(state.sp, 0x200)
    bus = PmuMmioWindow(inner=GK104FakeBus())
    exe = FalconExecutor(
        state=state,
        bus=bus,
        helpers={0x34: nv_wr32_helper},
        trace=TraceCollector(level=TraceLevel.EFFECTS),
    )
    exe.step()
    assert exe.state.get_reg(0) == 0
