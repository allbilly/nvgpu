"""Scripted bus and XFER tests."""

from falcon_oracle.bus.scripted import ScriptedBus
from falcon_oracle.bus.sparse_memory import SparseMemory
from falcon_oracle.bus.types import XferRequest
from falcon_oracle.errors import DeviceModelError, UnsupportedMMIO


def test_scheduled_mmio():
    yaml = """
schema: 1
initial_mmio: {"0x7c0": "0"}
events:
  - trigger: {kind: mmio_write, address: "0x7e0", value: "4"}
    effect: {after_steps: 2, set_mmio: {"0x7c0": "4"}}
"""
    bus = ScriptedBus.from_yaml(yaml)
    bus.mmio_write32(0x7E0, 4)
    assert bus.mmio_read32(0x7C0) == 0
    bus.advance(1)
    assert bus.mmio_read32(0x7C0) == 0
    bus.advance(1)
    assert bus.mmio_read32(0x7C0) == 4


def test_xfer_pending_overlap_rejected():
    bus = ScriptedBus(initial_mmio={}, xfer_complete_after_polls=100)
    bus.attach_dmem_provider(lambda a, s: b"\x00" * s)
    t1 = bus.xfer_start(
        XferRequest("dmem", 0, "direct_vram", 0x100, 16, "dmem_to_direct_vram", 0)
    )
    try:
        bus.xfer_start(
            XferRequest("dmem", 0, "direct_vram", 0x200, 16, "dmem_to_direct_vram", 0)
        )
        assert False
    except DeviceModelError as exc:
        assert "pending" in str(exc)
    st = bus.xfer_poll(t1)
    assert st.phase.name == "PENDING"


def test_xfer_completes_and_writes_vram():
    vram = SparseMemory(default=0)
    bus = ScriptedBus(vram=vram, xfer_complete_after_polls=1)
    payload = bytes(range(16))
    bus.attach_dmem_provider(lambda a, s: payload)
    token = bus.xfer_start(
        XferRequest("dmem", 0xD80, "direct_vram", 0x60200, 16, "dmem_to_direct_vram", 0)
    )
    st = bus.xfer_poll(token)
    assert st.phase.name == "COMPLETE"
    assert vram.read(0x60200, 16) == payload


def test_strict_unsupported_mmio():
    bus = ScriptedBus(initial_mmio={}, strict_mmio=True)
    try:
        bus.mmio_read32(0x1234)
        assert False
    except UnsupportedMMIO:
        pass
