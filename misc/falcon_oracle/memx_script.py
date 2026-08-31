"""Semantic MEMX script executor (Nouveau memx.fuc + memx.c command model).

Accepts the same command shapes used by ``examples_kepler/nvbios_init.py``:

  ENTER, LEAVE, WR32(addr, data[, ...]), WAIT(addr, mask, data, nsec), DELAY(nsec)

Timing is logical-step based (never wall-clock), matching the oracle plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

from .bus.gk104_fake_adapter import (
    FB_PAUSE_CLEAR,
    FB_PAUSE_SET,
    FB_PAUSE_STATUS,
    GK104FakeBus,
    PAUSE_BIT,
)
from .devices.pmu_mmio import PmuMmioWindow, rd32_via_window, wr32_via_window
from .errors import DeviceModelError
from .trace import EventKind, TraceCollector, TraceLevel, canonical_trace_hash
from .values import parse_int, u32

# Falcon IO: PTIMER low (macros.fuc NV_PPWR_TIMER_LOW).
TIMER_LOW = 0x002C

# Match examples_kepler/nvbios_init.py opcode numbers.
class MemxOp(IntEnum):
    ENTER = 1
    LEAVE = 2
    WR32 = 3
    WAIT = 4
    DELAY = 5


@dataclass(frozen=True)
class MemxCommand:
    op: MemxOp
    args: tuple[int, ...] = ()

    @staticmethod
    def enter() -> "MemxCommand":
        return MemxCommand(MemxOp.ENTER)

    @staticmethod
    def leave() -> "MemxCommand":
        return MemxCommand(MemxOp.LEAVE)

    @staticmethod
    def wr32(*words: int) -> "MemxCommand":
        if len(words) < 2 or len(words) % 2:
            raise ValueError("WR32 needs addr/data pairs")
        return MemxCommand(MemxOp.WR32, tuple(u32(w) for w in words))

    @staticmethod
    def wait(addr: int, mask: int, data: int, nsec: int) -> "MemxCommand":
        return MemxCommand(MemxOp.WAIT, (u32(addr), u32(mask), u32(data), u32(nsec)))

    @staticmethod
    def delay(nsec: int) -> "MemxCommand":
        return MemxCommand(MemxOp.DELAY, (u32(nsec),))


@dataclass
class MemxScriptResult:
    events: list
    stop_reason: str
    trace_hash: str
    host: dict[int, int]
    in_enter: bool
    commands_executed: int
    wait_satisfied: list[bool] = field(default_factory=list)


class LogicalTimerBus:
    """Wraps a bus and exposes TIMER_LOW that advances with ``advance()``."""

    NS_PER_STEP = 1000  # 1 µs per logical step (arbitrary but deterministic)

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.time_ns = 0

    def mmio_read32(self, address: int) -> int:
        if u32(address) == TIMER_LOW:
            return u32(self.time_ns)
        return self.inner.mmio_read32(address)

    def mmio_write32(self, address: int, value: int) -> None:
        if u32(address) == TIMER_LOW:
            return
        self.inner.mmio_write32(address, value)

    def advance(self, logical_steps: int = 1) -> None:
        self.time_ns = u32(self.time_ns + logical_steps * self.NS_PER_STEP)
        if hasattr(self.inner, "advance"):
            self.inner.advance(logical_steps)

    def xfer_start(self, request):
        return self.inner.xfer_start(request)

    def xfer_poll(self, token):
        return self.inner.xfer_poll(token)

    def attach_dmem_provider(self, reader) -> None:
        if hasattr(self.inner, "attach_dmem_provider"):
            self.inner.attach_dmem_provider(reader)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def parse_memx_commands(items: Iterable[Any]) -> list[MemxCommand]:
    out: list[MemxCommand] = []
    for item in items:
        if isinstance(item, MemxCommand):
            out.append(item)
            continue
        if isinstance(item, (list, tuple)) and item:
            op = item[0]
            args = tuple(item[1:])
            if isinstance(op, str):
                name = op.upper()
                if name == "ENTER":
                    out.append(MemxCommand.enter())
                elif name == "LEAVE":
                    out.append(MemxCommand.leave())
                elif name == "WR32":
                    out.append(MemxCommand.wr32(*[parse_int(a) for a in args]))
                elif name == "WAIT":
                    out.append(MemxCommand.wait(*[parse_int(a) for a in args]))
                elif name == "DELAY":
                    out.append(MemxCommand.delay(parse_int(args[0])))
                else:
                    raise ValueError(f"unknown MEMX op {op}")
            else:
                out.append(MemxCommand(MemxOp(int(op)), tuple(u32(parse_int(a)) for a in args)))
            continue
        if isinstance(item, dict):
            op = str(item["op"]).upper()
            args = item.get("args") or []
            out.extend(parse_memx_commands([(op, *args)]))
            continue
        raise ValueError(f"bad MEMX command: {item!r}")
    return out


def load_memx_fixture(path: Path | str) -> tuple[list[MemxCommand], dict[int, int], dict[str, Any]]:
    """Load fixture JSON → (commands, seed_host, meta)."""
    import json

    data = json.loads(Path(path).read_text())
    commands = parse_memx_commands(data.get("commands") or [])
    seed_raw = data.get("seed_host") or {}
    seed_host = {u32(parse_int(k)): u32(parse_int(v)) for k, v in seed_raw.items()}
    meta = {
        "name": data.get("name"),
        "description": data.get("description"),
        "expect_error": data.get("expect_error"),
        "host_after": data.get("host_after") or [],
    }
    return commands, seed_host, meta


def execute_memx_fixture(path: Path | str, **kwargs: Any) -> MemxScriptResult:
    commands, seed_host, _meta = load_memx_fixture(path)
    seed = kwargs.pop("seed_host", None)
    if seed:
        seed_host = {**seed_host, **{u32(k): u32(v) for k, v in seed.items()}}
    return execute_memx_script(commands, seed_host=seed_host or None, **kwargs)


def execute_memx_script(
    commands: list[MemxCommand] | Iterable[Any],
    *,
    seed_host: dict[int, int] | None = None,
    pause_ack_delay: int = 1,
    pause_clear_delay: int = 1,
    require_enter_for_wr32: bool = True,
    wait_poll_steps: int = 1,
) -> MemxScriptResult:
    cmds = parse_memx_commands(commands)
    trace = TraceCollector(level=TraceLevel.EXTERNAL)
    gk = GK104FakeBus(
        trace=trace,
        pause_ack_delay=pause_ack_delay,
        pause_clear_delay=pause_clear_delay,
        require_enter_before_xfer=False,
    )
    timed = LogicalTimerBus(gk)
    bus = PmuMmioWindow(inner=timed, trace=None)

    host = {0x1620: 0x00000AAB, 0x26F0: 0x00000001}
    if seed_host:
        host.update({u32(k): u32(v) for k, v in seed_host.items()})
    for addr, val in host.items():
        bus.seed_host(addr, val)
        gk._inner.mmio[addr] = val

    in_enter = False
    wait_satisfied: list[bool] = []
    executed = 0

    def host_rd(addr: int) -> int:
        val = rd32_via_window(bus, addr)
        host[addr] = val
        return val

    def host_wr(addr: int, val: int) -> None:
        wr32_via_window(bus, addr, val)
        host[u32(addr)] = u32(val)

    def do_enter() -> None:
        nonlocal in_enter
        r8 = host_rd(0x1620)
        host_wr(0x1620, u32(r8 & ~0x00000AA2))
        r8 = host_rd(0x1620)
        host_wr(0x1620, u32(r8 & ~0x00000001))
        r8 = host_rd(0x26F0)
        host_wr(0x26F0, u32(r8 & ~0x00000001))
        bus.mmio_write32(FB_PAUSE_SET, PAUSE_BIT)
        bus.advance(pause_ack_delay)
        status = bus.mmio_read32(FB_PAUSE_STATUS)
        if not (status & PAUSE_BIT):
            raise DeviceModelError("MEMX ENTER: FB_PAUSE never set")
        in_enter = True
        gk.in_enter = True
        gk._seen_rising_edge = True

    def do_leave() -> None:
        nonlocal in_enter
        bus.mmio_write32(FB_PAUSE_CLEAR, PAUSE_BIT)
        bus.advance(pause_clear_delay)
        status = bus.mmio_read32(FB_PAUSE_STATUS)
        if status & PAUSE_BIT:
            raise DeviceModelError("MEMX LEAVE: FB_PAUSE still set")
        r8 = host_rd(0x26F0)
        host_wr(0x26F0, u32(r8 | 1))
        r8 = host_rd(0x1620)
        host_wr(0x1620, u32(r8 | 1))
        r8 = host_rd(0x1620)
        host_wr(0x1620, u32(r8 | 0x00000AA2))
        in_enter = False
        gk.in_enter = False

    def do_wr32(args: tuple[int, ...]) -> None:
        if require_enter_for_wr32 and not in_enter:
            raise DeviceModelError(
                "MEMX WR32 outside ENTER/LEAVE ownership region",
                details={"args": args},
            )
        # Serial pairs — memx_func_wr32 loops until packet length consumed.
        it = iter(args)
        for addr, data in zip(it, it):
            host_wr(addr, data)
            bus.advance(1)

    def do_wait(addr: int, mask: int, data: int, nsec: int) -> bool:
        # kernel.fuc wait: spin until (rd32(addr)&mask)==data or timeout.
        start = timed.time_ns
        satisfied = False
        while True:
            val = host_rd(addr)
            if u32(val & mask) == u32(data):
                satisfied = True
                break
            bus.advance(wait_poll_steps)
            if u32(timed.time_ns - start) >= u32(nsec):
                break
        wait_satisfied.append(satisfied)
        trace.emit(
            EventKind.MARKER,
            address=addr,
            value=data,
            metadata={
                "memx": "wait",
                "mask": mask,
                "timeout_ns": nsec,
                "satisfied": satisfied,
            },
        )
        return satisfied

    def do_delay(nsec: int) -> None:
        steps = max(1, (nsec + LogicalTimerBus.NS_PER_STEP - 1) // LogicalTimerBus.NS_PER_STEP)
        bus.advance(steps)
        trace.emit(EventKind.SLEEP, value=nsec, metadata={"memx": "delay"})

    for cmd in cmds:
        if cmd.op is MemxOp.ENTER:
            do_enter()
        elif cmd.op is MemxOp.LEAVE:
            do_leave()
        elif cmd.op is MemxOp.WR32:
            do_wr32(cmd.args)
        elif cmd.op is MemxOp.WAIT:
            do_wait(*cmd.args)
        elif cmd.op is MemxOp.DELAY:
            do_delay(cmd.args[0])
        else:
            raise DeviceModelError(f"unsupported MEMX op {cmd.op}")
        executed += 1

    trace.emit(EventKind.HALT, metadata={"reason": "memx_script_done"})
    return MemxScriptResult(
        events=list(trace.events),
        stop_reason="done",
        trace_hash=canonical_trace_hash(trace.events),
        host=host,
        in_enter=in_enter,
        commands_executed=executed,
        wait_satisfied=wait_satisfied,
    )
