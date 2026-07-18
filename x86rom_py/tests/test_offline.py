"""Offline tests for x86rom_py (stdlib unittest)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

# Allow `python3 -m unittest` from x86rom_py/ or repo root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from x86rom_py.analyze import analyze_rom, verify_entry_path
from x86rom_py.bus import ExpectedOp, ModelBus, ReplayBus, RecordingBus
from x86rom_py.constants import (
  ENTRY_PATH_TARGET,
  ENTRY_PATH_VIA,
  HOSTILE_MMIO_OFFSETS,
  LIVE_ACK_TOKEN,
  PINNED_CONTAINER_SHA256,
  PINNED_LEGACY_SHA256,
  PMC_BOOT_0_GK104,
  VGA_BAR0_ALIAS_BASE,
)
from x86rom_py.cpu import CPU, FLAG_CF, FLAG_OF, FLAG_PF, FLAG_SF, FLAG_ZF
from x86rom_py.executor import build_machine, run_until_entry_target
from x86rom_py.firmware import FirmwareProfile, FirmwareServices
from x86rom_py.memory import MemoryError, RealModeMemory
from x86rom_py.rom import RomError, load_rom, select_legacy_image
from x86rom_py.safety import SafetyError, SafetyPolicy
from x86rom_py.trace import (
  TraceEvent,
  TraceLog,
  backward_slice,
  compare_traces,
  dump_jsonl,
  load_jsonl,
)


class TestRom(unittest.TestCase):
  def test_pinned_hashes_and_metadata(self):
    rom = load_rom()
    self.assertEqual(rom.container_sha256, PINNED_CONTAINER_SHA256)
    self.assertEqual(rom.sha256, PINNED_LEGACY_SHA256)
    self.assertEqual(rom.checksum, 0)
    self.assertEqual(rom.image_offset, 0x600)
    self.assertEqual(rom.size, 0xF600)
    self.assertEqual(rom.pcir.vendor_id, 0x10DE)
    self.assertEqual(rom.pcir.device_id, 0x1184)
    self.assertEqual(rom.pcir.code_type, 0)
    self.assertFalse(rom.pcir.is_final_image)  # EFI follows

  def test_rejects_bad_checksum(self):
    rom = load_rom()
    bad = bytearray(rom.container)
    # Flip a byte inside the legacy image without fixing checksum.
    bad[0x600 + 0x100] ^= 0xFF
    with self.assertRaises(RomError):
      load_rom(container=bytes(bad), require_pinned_hashes=False)

  def test_rejects_wrong_identity(self):
    rom = load_rom()
    bad = bytearray(rom.container)
    # Patch PCIR device id inside legacy image.
    pcir = rom.image_offset + rom.pcir.offset
    bad[pcir + 6] = 0x00
    bad[pcir + 7] = 0x00
    with self.assertRaises(RomError):
      load_rom(container=bytes(bad), require_pinned_hashes=False)

  def test_never_selects_efi(self):
    rom = load_rom()
    # EFI at 0xfc00 has code type != 0
    off, image, pcir = select_legacy_image(rom.container)
    self.assertEqual(pcir.code_type, 0)
    self.assertEqual(off, 0x600)
    self.assertNotEqual(off, 0xFC00)

  def test_entry_path_static(self):
    rom = load_rom()
    path = verify_entry_path(rom.data)
    self.assertEqual(path, [0x3, ENTRY_PATH_VIA, ENTRY_PATH_TARGET])


class TestAnalyze(unittest.TestCase):
  def test_census(self):
    report = analyze_rom()
    self.assertEqual(report.entry_path[-1], ENTRY_PATH_TARGET)
    self.assertGreater(len(report.io_outs) + len(report.io_ins), 0)
    self.assertIn(0x67AE, report.vga_preamble_offsets)


class TestMemory(unittest.TestCase):
  def test_rom_map_and_reject_write(self):
    rom = load_rom()
    mem = RealModeMemory(rom.data, rom_shadow_writable=False)
    self.assertEqual(mem.read8(0xC0000), 0x55)
    self.assertEqual(mem.read8(0xC0001), 0xAA)
    with self.assertRaises(MemoryError):
      mem.write8(0xC0000, 0x00)

  def test_rom_shadow_writable_default(self):
    rom = load_rom()
    mem = RealModeMemory(rom.data)
    mem.write8(0xC0082, 0x5A)
    self.assertEqual(mem.read8(0xC0082), 0x5A)

  def test_segment_wrap(self):
    mem = RealModeMemory(b"\x55\xaa" + b"\x00" * 510)
    mem.write8(0x10, 0x5A)
    self.assertEqual(mem.read_seg(0x0000, 0x10, 1), 0x5A)
    # wrap within 1MiB
    self.assertEqual(mem.phys(0x100000), 0)


class TestCPUFlags(unittest.TestCase):
  def _cpu(self) -> CPU:
    mem = RealModeMemory(b"\x55\xaa" + b"\x00" * 0x1000)
    bus = ModelBus()
    fw = FirmwareServices(FirmwareProfile(), bus, mem)
    return CPU(mem, bus, fw)

  def test_add_flags(self):
    cpu = self._cpu()
    cpu._alu_op(0, 0xFF, 0x01, 1)
    self.assertTrue(cpu.get_flag("CF"))
    self.assertTrue(cpu.get_flag("ZF"))

  def test_xor_clears_cf_of(self):
    cpu = self._cpu()
    cpu.set_flag("CF", True)
    cpu.set_flag("OF", True)
    cpu._alu_op(6, 0x12, 0x12, 1)
    self.assertFalse(cpu.get_flag("CF"))
    self.assertFalse(cpu.get_flag("OF"))
    self.assertTrue(cpu.get_flag("ZF"))

  def test_parity(self):
    cpu = self._cpu()
    self.assertTrue(cpu._parity(0x00))
    self.assertFalse(cpu._parity(0x01))


class TestCPUControl(unittest.TestCase):
  def test_short_and_near_jmp(self):
    # Build tiny program: EB 02 / 90 90 / E9 ... → land
    # Simpler: use real ROM entry
    result = run_until_entry_target()
    self.assertEqual(result.stop.reason, "breakpoint")
    self.assertEqual(result.stop.ip, ENTRY_PATH_TARGET)
    self.assertIn(0x3, result.reached_ips)
    self.assertIn(ENTRY_PATH_VIA, result.reached_ips)
    self.assertIn(ENTRY_PATH_TARGET, result.reached_ips)

  def test_push_pop_roundtrip(self):
    cpu, bus, mem, fw, trace = build_machine()
    cpu.eax = 0x1234
    cpu.push_width(cpu.eax & 0xFFFF, 2)
    cpu.eax = 0
    cpu.eax = (cpu.eax & ~0xFFFF) | cpu.pop_width(2)
    self.assertEqual(cpu.eax & 0xFFFF, 0x1234)


class TestBus(unittest.TestCase):
  def test_pci_mech1_cf8_cfc(self):
    bus = ModelBus()
    # Enable + bus0/dev0/fn0 / offset 0
    bus.io_write(0xCF8, 4, 0x80000000)
    val = bus.io_read(0xCFC, 4)
    self.assertEqual(val & 0xFFFF, 0x10DE)
    self.assertEqual((val >> 16) & 0xFFFF, 0x1184)

  def test_vga_byte_width_alias(self):
    bus = ModelBus()
    bus.io_write(0x3C3, 1, 0x01)
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3C3, 1), 0x01)
    # Word I/O splits into successive byte-width alias transactions.
    bus.io_write(0x3D4, 2, 0x573F)
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3D4, 1), 0x3F)
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3D5, 1), 0x57)

  def test_no_combine_byte_writes(self):
    bus = ModelBus()
    bus.mmio_write(0, VGA_BAR0_ALIAS_BASE + 0x3C2, 1, 0xAB)
    bus.mmio_write(0, VGA_BAR0_ALIAS_BASE + 0x3C3, 1, 0xCD)
    # Adjacent bytes remain independent (not a single 16-bit store).
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3C2, 1), 0xAB)
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3C3, 1), 0xCD)

  def test_replay_detects_extra(self):
    expected = [
      ExpectedOp("io_write", "io", "write", 1, canonical_offset=0x3C3, value=1),
    ]
    rb = ReplayBus(expected)
    rb.io_write(0x3C3, 1, 1)
    with self.assertRaises(Exception):
      rb.io_write(0x3C2, 1, 1)

  def test_recording_bus(self):
    inner = ModelBus()
    rec = RecordingBus(inner)
    rec.io_write(0x3C3, 1, 1)
    self.assertGreaterEqual(len(rec.trace.events), 1)


class TestTrace(unittest.TestCase):
  def test_compare_and_diverge(self):
    a = [
      TraceEvent(operation="mmio_write", address_space="mmio", direction="write",
                 width=1, bar=0, canonical_offset=0x6013C3, value=1),
      TraceEvent(operation="mmio_read", address_space="mmio", direction="read",
                 width=1, bar=0, canonical_offset=0x6013C3, read_result=1),
    ]
    b = [
      TraceEvent(operation="mmio_write", address_space="mmio", direction="write",
                 width=1, bar=0, canonical_offset=0x6013C3, value=1),
      TraceEvent(operation="mmio_read", address_space="mmio", direction="read",
                 width=1, bar=0, canonical_offset=0x6013C3, read_result=0),
    ]
    ok = compare_traces(a, a)
    self.assertTrue(ok.match)
    bad = compare_traces(a, b)
    self.assertFalse(bad.match)
    self.assertEqual(bad.divergence.index, 1)

  def test_jsonl_roundtrip(self):
    ev = TraceEvent(operation="delay", address_space="cpu", direction="none", value=100)
    with tempfile.TemporaryDirectory() as td:
      path = Path(td) / "t.jsonl"
      dump_jsonl([ev], path)
      loaded = load_jsonl(path)
      self.assertEqual(loaded[0].operation, "delay")
      self.assertEqual(loaded[0].value, 100)

  def test_backward_slice(self):
    events = [
      TraceEvent(operation="pci_write", address_space="pci", direction="write",
                 width=4, canonical_offset=0x10, value=0xA0000000),
      TraceEvent(operation="mmio_write", address_space="mmio", direction="write",
                 width=1, bar=0, canonical_offset=0x6013C3, value=1),
      TraceEvent(operation="checkpoint", address_space="checkpoint", direction="none",
                 dependency_tag="pramin-live", read_result=0x12345678),
    ]
    report = backward_slice(events, 2)
    self.assertEqual(len(report.retained), 3)
    self.assertIn(0, report.address_allocation)


class TestSafety(unittest.TestCase):
  def test_readonly_default(self):
    pol = SafetyPolicy()
    with self.assertRaises(SafetyError):
      pol.check_mmio_write(0, 0x100, 4)

  def test_live_ack_token(self):
    pol = SafetyPolicy()
    with self.assertRaises(SafetyError):
      pol.require_live_writes("nope")
    pol.require_live_writes(LIVE_ACK_TOKEN)
    self.assertTrue(pol.live_writes)

  def test_hostile_mmio(self):
    pol = SafetyPolicy()
    pol.require_live_writes(LIVE_ACK_TOKEN)
    for off in HOSTILE_MMIO_OFFSETS:
      with self.assertRaises(SafetyError):
        pol.check_mmio_write(0, off, 4, rom_cs=0xC000, rom_ip=0x1234)

  def test_pramin_stub_class(self):
    pol = SafetyPolicy()
    self.assertTrue(pol.is_pramin_stub(0xFFFFFFFF))
    self.assertTrue(pol.is_pramin_stub(0xBAD0FB12))
    self.assertTrue(pol.is_pramin_live(0x00000000))
    self.assertTrue(pol.is_pramin_live(0xDEADBEEF))

  def test_pmc_boot(self):
    pol = SafetyPolicy()
    pol.check_pmc_boot_0(PMC_BOOT_0_GK104)
    with self.assertRaises(SafetyError):
      pol.check_pmc_boot_0(0xFFFFFFFF)


class TestFirmware(unittest.TestCase):
  def test_pci_bios_install_check(self):
    cpu, bus, mem, fw, _ = build_machine()
    cpu.eax = 0xB101
    fw.handle(cpu, 0x1A)
    self.assertEqual(cpu.edx, 0x20494350)
    self.assertFalse(cpu.get_flag("CF"))

  def test_unknown_int_aborts(self):
    cpu, bus, mem, fw, _ = build_machine()
    with self.assertRaises(Exception):
      fw.handle(cpu, 0xFF)


class TestVgaPreambleFixture(unittest.TestCase):
  def test_option_rom_vga_prefix_on_model_bus(self):
    """Reproduce NvbiosInit.option_rom_vga_enable_prefix event shape."""
    bus = ModelBus()
    # OUT 3c3,1; IN 3c3; OUT 3c2,1; IN 3c3; CRTC unlock via 3d4/3d5
    bus.io_write(0x3C3, 1, 0x01)
    bus.io_read(0x3C3, 1)
    bus.io_write(0x3C2, 1, 0x01)
    bus.io_read(0x3C3, 1)
    bus.io_write(0x3D4, 1, 0x3F)
    bus.io_write(0x3D5, 1, 0x57)
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3C3, 1), 0x01)
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3D4, 1), 0x3F)
    self.assertEqual(bus.mmio_read(0, VGA_BAR0_ALIAS_BASE + 0x3D5, 1), 0x57)

  def test_cpu_reaches_vga_preamble_outs(self):
    """Executor reaches 0x67ae and emits the Night41p VGA enable prefix."""
    from x86rom_py.executor import run_rom
    r = run_rom(max_insns=200_000)
    ios = [e for e in r.trace.events if e.address_space == "io" and e.dependency_tag == "vga_alias"]
    # First four guest I/O ops of the prefix (byte OUT/IN on 3c3/3c2).
    prefix = [
      (e.direction, e.canonical_offset, e.width, e.value if e.direction == "write" else e.read_result)
      for e in ios[:4]
    ]
    self.assertEqual(prefix[0], ("write", 0x3C3, 1, 1))
    self.assertEqual(prefix[1], ("read", 0x3C3, 1, 1))
    self.assertEqual(prefix[2], ("write", 0x3C2, 1, 1))
    self.assertEqual(prefix[3], ("read", 0x3C3, 1, 1))
    self.assertIn(0x67AE, r.reached_ips)
    # Extended CRTC unlock CR3f=0x57 (byte writes; may be preceded by other CRTC ops).
    unlock = [
      e for e in ios
      if e.direction == "write" and e.canonical_offset == 0x3D5 and e.value == 0x57
    ]
    index = [
      e for e in ios
      if e.direction == "write" and e.canonical_offset == 0x3D4 and e.value == 0x3F
    ]
    self.assertTrue(index, "missing CRTC index write 0x3d4=0x3f")
    self.assertTrue(unlock, "missing CRTC data write 0x3d5=0x57")

  def test_real_rom_reaches_native_io_bar_prefix(self):
    """The pinned Palit ROM discovers BAR5 and emits the bounded H75 prefix."""
    from x86rom_py.executor import run_rom

    r = run_rom(max_insns=300)
    ios = [
      e for e in r.trace.events
      if e.address_space == "io" and e.dependency_tag == "io_bar"
    ]
    prefix = [
      (e.direction, e.bar, e.canonical_offset,
       e.value if e.direction == "write" else e.read_result)
      for e in ios[:6]
    ]
    self.assertEqual(prefix, [
      ("write", 5, 0x00, 0x2469FDB9),
      ("write", 5, 0x04, 0x00000001),
      ("write", 5, 0x08, 0x00000200),
      ("read",  5, 0x0C, 0xFFFFFFFF),
      ("write", 5, 0x08, 0x00000200),
      ("write", 5, 0x0C, 0xFFFFFFFF),
    ])


class TestStopReasons(unittest.TestCase):
  def test_unsupported_opcode_stop(self):
    cpu, bus, mem, fw, _ = build_machine()
    # Plant UD2-equivalent: 0F 0B at a scratch CS by jumping into RAM
    mem.write8(0x8000, 0x0F)
    mem.write8(0x8001, 0x0B)
    cpu.cs = 0x0800
    cpu.eip = 0x0000
    stop = cpu.run(max_insns=4)
    self.assertEqual(stop.reason, "unsupported-opcode")
    self.assertIsNotNone(stop.regs)


if __name__ == "__main__":
  unittest.main()
