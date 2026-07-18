"""Invalid IMEM / missing xdwait / sticky pause regressions."""

from pathlib import Path

import pytest

from falcon_oracle.bus.gk104_fake_adapter import GK104FakeBus
from falcon_oracle.bus.types import XferRequest
from falcon_oracle.errors import DeviceModelError, InvalidIMEMAccess
from falcon_oracle.executor import FalconExecutor
from falcon_oracle.manifests import load_initial_dmem
from falcon_oracle.runner import execute_falcon_bootstrap
from falcon_oracle.state import FalconMemory, FalconState
from falcon_oracle.trace import TraceCollector, TraceLevel

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "pmu_bar1_bootstrap"


def test_entry_beyond_imem_faults():
    dmem = FalconMemory(load_initial_dmem(CORPUS, 4096), name="dmem")
    imem = FalconMemory(bytearray(16), name="imem", base=0xB14)
    state = FalconState(pc=0xC00, imem=imem, dmem=dmem)
    exe = FalconExecutor(state=state, bus=GK104FakeBus())
    with pytest.raises(InvalidIMEMAccess):
        exe.step()


def test_second_xfer_while_pending():
    bus = GK104FakeBus(xfer_complete_after_polls=100)
    bus.in_enter = True
    bus._seen_rising_edge = True
    bus.attach_dmem_provider(lambda a, s: b"\x00" * s)
    bus.xfer_start(
        XferRequest("dmem", 0, "direct_vram", 0x100, 16, "dmem_to_direct_vram", 0)
    )
    with pytest.raises(DeviceModelError):
        bus.xfer_start(
            XferRequest("dmem", 0, "direct_vram", 0x200, 16, "dmem_to_direct_vram", 0)
        )


def test_sticky_pause_blocks_rising_edge_xfer():
    bus = GK104FakeBus()
    # Sticky high before ENTER
    bus._inner.mmio[0x07C0] = 4
    bus.mmio_write32(0x07E0, 4)  # SET without clear edge
    assert bus._seen_rising_edge is False
    bus.attach_dmem_provider(lambda a, s: b"\x00" * 16)
    with pytest.raises(DeviceModelError):
        bus.in_enter = True  # force ownership but still no rising edge
        bus.xfer_start(
            XferRequest("dmem", 0, "direct_vram", 0x100, 16, "dmem_to_direct_vram", 0)
        )
