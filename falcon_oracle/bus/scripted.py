"""Logical-step scripted MMIO/XFER bus for deterministic unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from ..errors import (
    DeviceModelError,
    UnsupportedMMIO,
    UnsupportedXferMode,
    XferTimeout,
)
from ..trace import EventKind, TraceCollector
from ..values import parse_int, u32
from .sparse_memory import SparseMemory
from .types import XferPhase, XferRequest, XferStatus


@dataclass
class ScheduledEffect:
    fire_at_step: int
    set_mmio: dict[int, int] = field(default_factory=dict)
    complete_xfer_token: int | None = None
    fault_xfer_token: int | None = None
    marker: str | None = None


@dataclass
class TriggerRule:
    kind: str
    address: int | None = None
    value: int | None = None
    after_steps: int = 0
    set_mmio: dict[int, int] = field(default_factory=dict)
    complete_pending_xfer: bool = False
    marker: str | None = None


class ScriptedBus:
    def __init__(
        self,
        *,
        initial_mmio: dict[int, int] | None = None,
        rules: list[TriggerRule] | None = None,
        vram: SparseMemory | None = None,
        xfer_complete_after_polls: int = 1,
        trace: TraceCollector | None = None,
        strict_mmio: bool = True,
    ) -> None:
        self.mmio: dict[int, int] = dict(initial_mmio or {})
        self.rules = list(rules or [])
        self.vram = vram or SparseMemory(default=0x00)
        self.xfer_complete_after_polls = xfer_complete_after_polls
        self.trace = trace
        self.strict_mmio = strict_mmio
        self.logical_step = 0
        self._pending_effects: list[ScheduledEffect] = []
        self._xfers: dict[int, dict[str, Any]] = {}
        self._next_token = 1
        self._pending_token: int | None = None

    @classmethod
    def from_yaml(cls, text: str, **kwargs: Any) -> "ScriptedBus":
        data = yaml.safe_load(text) or {}
        initial = {
            parse_int(k): parse_int(v) for k, v in (data.get("initial_mmio") or {}).items()
        }
        rules: list[TriggerRule] = []
        for ev in data.get("events") or []:
            trig = ev.get("trigger") or {}
            effect = ev.get("effect") or {}
            rules.append(
                TriggerRule(
                    kind=trig.get("kind", "mmio_write"),
                    address=parse_int(trig["address"]) if "address" in trig else None,
                    value=parse_int(trig["value"]) if "value" in trig else None,
                    after_steps=int(effect.get("after_steps", 0)),
                    set_mmio={
                        parse_int(k): parse_int(v)
                        for k, v in (effect.get("set_mmio") or {}).items()
                    },
                    complete_pending_xfer=bool(effect.get("complete_pending_xfer", False)),
                    marker=effect.get("marker"),
                )
            )
        default = data.get("vram_default", 0)
        return cls(
            initial_mmio=initial,
            rules=rules,
            vram=SparseMemory(default=None if default == "fault" else parse_int(default)),
            xfer_complete_after_polls=int(data.get("xfer_complete_after_polls", 1)),
            strict_mmio=bool(data.get("strict_mmio", False)),
            **kwargs,
        )

    def mmio_read32(self, address: int) -> int:
        address = u32(address)
        if address not in self.mmio:
            if self.strict_mmio:
                raise UnsupportedMMIO("MMIO not defined by scenario", details={"address": address})
            value = 0
        else:
            value = u32(self.mmio[address])
        if self.trace:
            self.trace.emit(EventKind.MMIO_READ, address=address, value=value, size=4)
        return value

    def mmio_write32(self, address: int, value: int) -> None:
        address = u32(address)
        value = u32(value)
        self.mmio[address] = value
        if self.trace:
            self.trace.emit(EventKind.MMIO_WRITE, address=address, value=value, size=4)
        self._match_triggers("mmio_write", address, value)

    def xfer_start(self, request: XferRequest) -> int:
        if request.target_mode != "direct_vram":
            raise UnsupportedXferMode(
                "only direct_vram XFER supported in MVP",
                details={"mode": request.target_mode},
            )
        if self._pending_token is not None:
            raise DeviceModelError(
                "second XFER while first still pending",
                details={"pending": self._pending_token, "request": request},
            )
        token = self._next_token
        self._next_token += 1
        self._pending_token = token
        self._xfers[token] = {
            "request": request,
            "phase": XferPhase.PENDING,
            "polls": 0,
            "bytes": None,
        }
        if self.trace:
            self.trace.emit(
                EventKind.XFER_START,
                address=request.destination_address,
                value=request.source_address,
                size=request.size,
                metadata={
                    "token": token,
                    "direction": request.direction,
                    "port": request.port,
                    "source_space": request.source_space,
                    "destination_space": request.destination_space,
                },
            )
        self._match_triggers("xfer_start", None, None)
        # Auto-schedule completion after N polls unless a rule handles it.
        if self.xfer_complete_after_polls >= 0 and not any(
            r.complete_pending_xfer for r in self.rules
        ):
            # Completion is poll-driven in xfer_poll.
            pass
        return token

    def xfer_poll(self, token: int) -> XferStatus:
        xfer = self._xfers.get(token)
        if xfer is None:
            raise DeviceModelError("unknown XFER token", details={"token": token})
        if xfer["phase"] is XferPhase.PENDING:
            xfer["polls"] += 1
            if (
                self.xfer_complete_after_polls >= 0
                and xfer["polls"] >= self.xfer_complete_after_polls
            ):
                self._complete_xfer(token)
        return XferStatus(token=token, phase=xfer["phase"], fault_reason=xfer.get("fault_reason"))

    def advance(self, logical_steps: int = 1) -> None:
        for _ in range(max(logical_steps, 0)):
            self.logical_step += 1
            self._apply_due_effects()

    def attach_dmem_provider(self, reader) -> None:
        self._dmem_reader = reader

    def _complete_xfer(self, token: int) -> None:
        xfer = self._xfers[token]
        if xfer["phase"] is not XferPhase.PENDING:
            return
        req: XferRequest = xfer["request"]
        reader = getattr(self, "_dmem_reader", None)
        if req.direction == "dmem_to_direct_vram":
            if reader is None:
                raise DeviceModelError("no DMEM provider for XFER store")
            data = reader(req.source_address, req.size)
            self.vram.write(req.destination_address, data)
            xfer["bytes"] = data
        elif req.direction == "direct_vram_to_dmem":
            data = self.vram.read(req.source_address, req.size)
            xfer["bytes"] = data
            writer = getattr(self, "_dmem_writer", None)
            if writer is None:
                raise DeviceModelError("no DMEM writer for XFER load")
            writer(req.destination_address, data)
        else:
            raise UnsupportedXferMode("bad direction", details={"direction": req.direction})
        xfer["phase"] = XferPhase.COMPLETE
        if self._pending_token == token:
            self._pending_token = None
        if self.trace:
            self.trace.emit(
                EventKind.XFER_COMPLETE,
                address=req.destination_address,
                value=req.source_address,
                size=req.size,
                metadata={"token": token, "sha256": __import__("hashlib").sha256(xfer["bytes"]).hexdigest()},
            )

    def _fault_xfer(self, token: int, reason: str) -> None:
        xfer = self._xfers[token]
        xfer["phase"] = XferPhase.FAULT
        xfer["fault_reason"] = reason
        if self._pending_token == token:
            self._pending_token = None

    def _match_triggers(self, kind: str, address: int | None, value: int | None) -> None:
        for rule in self.rules:
            if rule.kind != kind:
                continue
            if rule.address is not None and address is not None and rule.address != address:
                continue
            if rule.value is not None and value is not None and rule.value != value:
                continue
            fire_at = self.logical_step + max(int(rule.after_steps), 0)
            complete_token = self._pending_token if rule.complete_pending_xfer else None
            effect = ScheduledEffect(
                fire_at_step=fire_at,
                set_mmio=dict(rule.set_mmio),
                complete_xfer_token=complete_token,
                marker=rule.marker,
            )
            self._pending_effects.append(effect)
            if rule.after_steps <= 0:
                self._apply_due_effects()

    def _apply_due_effects(self) -> None:
        due = [e for e in self._pending_effects if e.fire_at_step <= self.logical_step]
        self._pending_effects = [
            e for e in self._pending_effects if e.fire_at_step > self.logical_step
        ]
        for effect in due:
            for addr, val in effect.set_mmio.items():
                self.mmio[addr] = u32(val)
            if effect.complete_xfer_token is not None:
                self._complete_xfer(effect.complete_xfer_token)
            if effect.fault_xfer_token is not None:
                self._fault_xfer(effect.fault_xfer_token, "scripted fault")
            if effect.marker and self.trace:
                self.trace.emit(EventKind.MARKER, metadata={"name": effect.marker})
