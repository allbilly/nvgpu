"""Orchestrate ROM load → memory/firmware/CPU → bus against stop policies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Set

from .bus import Bus, ModelBus, RecordingBus
from .constants import (
  ENTRY_PATH_TARGET,
  ENTRY_PATH_VIA,
  LIVE_ACK_TOKEN,
  ROM_SEGMENT,
  VGA_PREAMBLE_NEAR,
)
from .cpu import CPU, StopInfo, StopReason
from .firmware import FirmwareProfile, FirmwareServices
from .memory import RealModeMemory
from .rom import RomImage, load_rom
from .safety import SafetyPolicy
from .trace import TraceLog


@dataclass
class RunResult:
  stop: StopInfo
  insn_count: int
  op_count: int
  elapsed_ns: int
  trace: Optional[TraceLog]
  reached_ips: Set[int]


def build_machine(
  rom: Optional[RomImage] = None,
  *,
  bus: Optional[Bus] = None,
  profile: Optional[FirmwareProfile] = None,
  trace: Optional[TraceLog] = None,
  safety: Optional[SafetyPolicy] = None,
) -> tuple[CPU, Bus, RealModeMemory, FirmwareServices, TraceLog]:
  if rom is None:
    rom = load_rom()
  if trace is None:
    trace = TraceLog()
  if safety is None:
    safety = SafetyPolicy(live_writes=True, live_ack="model")
    safety.live_writes = True
  if bus is None:
    bus = ModelBus(safety=safety, trace=trace)
  elif not isinstance(bus, RecordingBus) and getattr(bus, "trace", None) is None:
    if hasattr(bus, "trace"):
      bus.trace = trace  # type: ignore[attr-defined]
  mem = RealModeMemory(rom.data, rom_segment=ROM_SEGMENT, bus=bus)
  profile = profile or FirmwareProfile()
  profile.install_bda(mem)
  # Seed a return frame so RETF from the ROM can observe rom-return.
  # Option ROM is CALLed far from POST; simulate caller at 0000:7C00-ish.
  fw = FirmwareServices(profile, bus, mem, trace=trace)
  cpu = CPU(mem, bus, fw, trace=trace)
  profile.apply_entry_state(cpu)
  # Place a far-return target on the stack as if BIOS did CALL FAR C000:0003
  cpu.push_width(0x0000, 2)  # return CS
  cpu.push_width(0x7C00, 2)  # return IP
  return cpu, bus, mem, fw, trace


def run_rom(
  rom: Optional[RomImage] = None,
  *,
  bus: Optional[Bus] = None,
  profile: Optional[FirmwareProfile] = None,
  trace: Optional[TraceLog] = None,
  max_insns: int = 50_000,
  max_ops: int = 100_000,
  max_wall_s: float = 30.0,
  stop_at_ips: Optional[Set[int]] = None,
  stop_at_pramin_live: bool = False,
) -> RunResult:
  cpu, bus, mem, fw, trace = build_machine(
    rom, bus=bus, profile=profile, trace=trace,
  )
  stop_at_ips = set(stop_at_ips or ())
  reached: Set[int] = set()
  t0 = time.time_ns()
  while cpu.stopped is None:
    if cpu.insn_count >= max_insns:
      cpu.stopped = StopInfo(
        reason=StopReason.INSTRUCTION_BUDGET,
        cs=cpu.cs, ip=cpu.eip & 0xFFFF,
        bytes_at_ip=cpu._peek(8),
        message=f"max_insns={max_insns}",
        regs=cpu.regs_dict(),
      )
      break
    if cpu.op_count >= max_ops:
      cpu.stopped = StopInfo(
        reason=StopReason.OPERATION_BUDGET,
        cs=cpu.cs, ip=cpu.eip & 0xFFFF,
        bytes_at_ip=cpu._peek(8),
        regs=cpu.regs_dict(),
      )
      break
    if (time.time_ns() - t0) / 1e9 > max_wall_s:
      cpu.stopped = StopInfo(
        reason=StopReason.WALL_TIME_BUDGET,
        cs=cpu.cs, ip=cpu.eip & 0xFFFF,
        bytes_at_ip=cpu._peek(8),
        regs=cpu.regs_dict(),
      )
      break
    ip = cpu.eip & 0xFFFF
    reached.add(ip)
    if ip in stop_at_ips:
      cpu.stopped = StopInfo(
        reason="breakpoint", cs=cpu.cs, ip=ip,
        bytes_at_ip=cpu._peek(8), regs=cpu.regs_dict(),
      )
      break
    if stop_at_pramin_live:
      probe = bus.checkpoint("pramin-live")
      if probe.values.get("pramin_live"):
        cpu.stopped = StopInfo(
          reason=StopReason.PRAMIN_LIVE, cs=cpu.cs, ip=ip,
          bytes_at_ip=cpu._peek(8),
          message=probe.detail, regs=cpu.regs_dict(),
        )
        break
    if not cpu.step():
      break
  assert cpu.stopped is not None
  return RunResult(
    stop=cpu.stopped,
    insn_count=cpu.insn_count,
    op_count=cpu.op_count,
    elapsed_ns=time.time_ns() - t0,
    trace=trace,
    reached_ips=reached,
  )


def run_until_entry_target(rom: Optional[RomImage] = None) -> RunResult:
  """Offline gate: execute entry until image offset 0x2caa."""
  return run_rom(rom, stop_at_ips={ENTRY_PATH_TARGET}, max_insns=32)


def run_until_vga_preamble(rom: Optional[RomImage] = None, max_insns: int = 200_000) -> RunResult:
  """Attempt to reach the known VGA OUT site at 0x67ae (may stop earlier)."""
  return run_rom(rom, stop_at_ips={VGA_PREAMBLE_NEAR}, max_insns=max_insns)


def live_ack_ok(token: Optional[str]) -> bool:
  return token == LIVE_ACK_TOKEN
