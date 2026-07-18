"""PMU MMIO window (kernel.fuc wr32/rd32)."""

from falcon_oracle.bus.gk104_fake_adapter import GK104FakeBus
from falcon_oracle.devices.pmu_mmio import (
    CTRL_MASK_B32_0,
    CTRL_OP_WR,
    CTRL_TRIGGER,
    MMIO_ADDR,
    MMIO_CTRL,
    MMIO_DATA,
    PmuMmioWindow,
    wr32_via_window,
)
from falcon_oracle.trace import EventKind, TraceCollector, TraceLevel


def test_wr32_window_emits_host_write_only():
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    inner = GK104FakeBus(trace=trace)
    bus = PmuMmioWindow(inner=inner)
    wr32_via_window(bus, 0x1620, 0x8)
    writes = [e for e in trace.events if e.kind is EventKind.MMIO_WRITE]
    assert len(writes) == 1
    assert writes[0].address == 0x1620
    assert writes[0].value == 0x8
    # Window regs themselves are not external host events.
    assert all(e.address not in {MMIO_ADDR, MMIO_DATA, MMIO_CTRL} for e in writes)


def test_manual_ctrl_trigger():
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    bus = PmuMmioWindow(inner=GK104FakeBus(trace=trace))
    bus.mmio_write32(MMIO_ADDR, 0x26F0)
    bus.mmio_write32(MMIO_DATA, 0)
    bus.mmio_write32(MMIO_CTRL, CTRL_OP_WR | CTRL_MASK_B32_0 | CTRL_TRIGGER)
    assert bus.host_reads[0x26F0] == 0
