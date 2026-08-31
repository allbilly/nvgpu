"""Falcon CPU executor for the bootstrap-pad instruction subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .bus.types import XferPhase, XferRequest
from .decoder import FalconDecoder
from .errors import (
    DeviceModelError,
    InvalidBranchTarget,
    InvalidIMEMAccess,
    UnsupportedInstruction,
    WaitTimeout,
    XferTimeout,
)
from .instruction import DecodedInstruction
from .state import FalconState
from .trace import EventKind, TraceCollector, TraceLevel
from .values import u32


HelperFn = Callable[["FalconExecutor"], None]


@dataclass
class ExecutionResult:
    state: FalconState
    events: list
    stop_reason: str
    instruction_count: int
    trace_hash: str | None = None


@dataclass
class FalconExecutor:
    state: FalconState
    bus: Any
    decoder: FalconDecoder = field(default_factory=FalconDecoder)
    trace: TraceCollector = field(default_factory=TraceCollector)
    helpers: dict[int, HelperFn] = field(default_factory=dict)
    symbols: dict[int, str] = field(default_factory=dict)
    done_pc: int | None = None
    done_magic_addr: int | None = None
    done_magic_value: int | None = None
    max_xdwait_polls: int = 1024

    def __post_init__(self) -> None:
        # Allow bus to pull DMEM bytes for XFER stores.
        if hasattr(self.bus, "attach_dmem_provider"):
            self.bus.attach_dmem_provider(
                lambda addr, size: self.state.dmem.read_bytes(addr, size)
            )
        if not self.helpers:
            self.helpers = {0x34: nv_wr32_helper}

    def symbol_at(self, pc: int | None) -> str | None:
        if pc is None:
            return None
        return self.symbols.get(pc)

    def step(self) -> list:
        if self.state.status != "running":
            return []
        # Helper entry: execute helper body without decoding missing IMEM.
        if self.state.pc in self.helpers:
            helper = self.helpers[self.state.pc]
            helper(self)
            self.state.instruction_count += 1
            self.state.logical_step += 1
            if hasattr(self.bus, "advance"):
                self.bus.advance(1)
            return []

        try:
            insn = self.decoder.decode_one(self.state.imem, self.state.pc)
        except InvalidIMEMAccess as exc:
            self.state.status = "fault"
            self.state.fault_reason = str(exc)
            self.trace.emit(EventKind.FAULT, pc=self.state.pc, metadata={"reason": str(exc)})
            raise

        self.trace.emit(
            EventKind.INSTRUCTION,
            pc=insn.pc,
            raw_instruction=insn.raw,
            mnemonic=insn.mnemonic,
            source_symbol=self.symbol_at(insn.pc),
            metadata={"size": insn.size, "form": insn.form},
        )
        self.execute_instruction(insn)
        self.state.instruction_count += 1
        self.state.logical_step += 1
        if hasattr(self.bus, "advance"):
            self.bus.advance(1)
        self._maybe_complete()
        return []

    def run(
        self,
        *,
        max_instructions: int,
        breakpoints: set[int] | None = None,
    ) -> ExecutionResult:
        breakpoints = breakpoints or set()
        stop_reason = "max_instructions"
        try:
            while self.state.status == "running" and self.state.instruction_count < max_instructions:
                if self.state.pc in breakpoints:
                    stop_reason = "breakpoint"
                    break
                self.step()
                if self.state.status != "running":
                    stop_reason = self.state.status
                    break
            else:
                if self.state.status == "running" and self.state.instruction_count >= max_instructions:
                    stop_reason = "max_instructions"
                    self.state.status = "timeout"
        except Exception as exc:
            self.state.status = "fault"
            self.state.fault_reason = str(exc)
            stop_reason = "fault"
            self.trace.emit(
                EventKind.FAULT,
                pc=self.state.pc,
                metadata={"reason": str(exc)},
            )
            raise

        from .trace import canonical_trace_hash

        return ExecutionResult(
            state=self.state,
            events=list(self.trace.events),
            stop_reason=stop_reason,
            instruction_count=self.state.instruction_count,
            trace_hash=canonical_trace_hash(self.trace.events),
        )

    def _maybe_complete(self) -> None:
        if self.done_magic_addr is not None and self.done_magic_value is not None:
            try:
                val = self.state.dmem.read_u32(self.done_magic_addr)
            except Exception:
                return
            if val == u32(self.done_magic_value):
                if self.done_pc is None or self.state.pc == self.done_pc:
                    self.state.status = "done"
                    self.trace.emit(
                        EventKind.HALT,
                        pc=self.state.pc,
                        value=val,
                        metadata={"reason": "done"},
                    )

    def execute_instruction(self, insn: DecodedInstruction) -> None:
        handler = getattr(self, f"_op_{insn.mnemonic}", None)
        if handler is None:
            raise UnsupportedInstruction(
                "no executor handler",
                pc=insn.pc,
                raw=insn.raw,
                mnemonic=insn.mnemonic,
            )
        handler(insn)

    # --- helpers for register/flag updates ---

    def _write_reg(self, idx: int, value: int, *, pc: int) -> None:
        self.state.set_reg(idx, value)
        self.trace.emit(
            EventKind.REGISTER_WRITE,
            pc=pc,
            address=idx,
            value=u32(value),
            source_symbol=self.symbol_at(pc),
        )

    def _set_flags(self, *, pc: int, carry=None, overflow=None, sign=None, zero=None) -> None:
        if carry is not None:
            self.state.carry = bool(carry)
        if overflow is not None:
            self.state.overflow = bool(overflow)
        if sign is not None:
            self.state.sign = bool(sign)
        if zero is not None:
            self.state.zero = bool(zero)
        self.trace.emit(
            EventKind.FLAG_WRITE,
            pc=pc,
            metadata={
                "c": self.state.carry,
                "o": self.state.overflow,
                "s": self.state.sign,
                "z": self.state.zero,
            },
        )

    def _eval_cc(self, cc: str) -> bool:
        if cc in ("always",):
            return True
        if cc in ("ne", "nz"):
            return not self.state.zero
        if cc in ("e", "z"):
            return self.state.zero
        if cc in ("c", "b"):
            return self.state.carry
        if cc in ("nc", "ae", "nb"):
            return not self.state.carry
        raise UnsupportedInstruction(f"unsupported condition {cc}")

    def _op_mov(self, insn: DecodedInstruction) -> None:
        meta = insn.metadata
        if meta.get("form") == "mov_to_sreg":
            sreg = meta["sreg"]
            src = meta["src"]
            self.state.set_sreg(sreg, self.state.get_reg(src))
            self.state.pc = insn.pc + insn.size
            return
        if meta.get("form") == "mov_from_sreg":
            self._write_reg(meta["dst"], self.state.get_sreg(meta["sreg"]), pc=insn.pc)
            self.state.pc = insn.pc + insn.size
            return
        # immediate mov
        dst = meta["dst"]
        imm = meta["imm"]
        self._write_reg(dst, imm, pc=insn.pc)
        self.state.pc = insn.pc + insn.size

    def _op_sethi(self, insn: DecodedInstruction) -> None:
        dst = insn.metadata["dst"]
        imm = insn.metadata["imm"]
        cur = self.state.get_reg(dst)
        self._write_reg(dst, (cur & 0xFFFF) | ((imm & 0xFFFF) << 16), pc=insn.pc)
        self.state.pc = insn.pc + insn.size

    def _op_and(self, insn: DecodedInstruction) -> None:
        dst = insn.metadata["dst"]
        imm = insn.metadata["imm"]
        res = u32(self.state.get_reg(dst) & imm)
        self._write_reg(dst, res, pc=insn.pc)
        self.state.set_arith_flags(res, carry=False, overflow=False)
        self._set_flags(
            pc=insn.pc,
            carry=False,
            overflow=False,
            sign=self.state.sign,
            zero=self.state.zero,
        )
        self.state.pc = insn.pc + insn.size

    def _op_clear(self, insn: DecodedInstruction) -> None:
        dst = insn.metadata["dst"]
        self._write_reg(dst, 0, pc=insn.pc)
        self.state.pc = insn.pc + insn.size

    def _op_st(self, insn: DecodedInstruction) -> None:
        base = self.state.get_reg(insn.metadata["base"])
        off = insn.metadata["off"]
        src = self.state.get_reg(insn.metadata["src"])
        addr = u32(base + off)
        self.state.dmem.write_u32(addr, src)
        self.trace.emit(
            EventKind.DMEM_WRITE,
            pc=insn.pc,
            address=addr,
            value=src,
            size=4,
            source_symbol=self.symbol_at(insn.pc),
        )
        self.state.pc = insn.pc + insn.size

    def _op_ld(self, insn: DecodedInstruction) -> None:
        base = self.state.get_reg(insn.metadata["base"])
        off = insn.metadata["off"]
        addr = u32(base + off)
        val = self.state.dmem.read_u32(addr)
        self.trace.emit(
            EventKind.DMEM_READ,
            pc=insn.pc,
            address=addr,
            value=val,
            size=4,
            source_symbol=self.symbol_at(insn.pc),
        )
        self._write_reg(insn.metadata["dst"], val, pc=insn.pc)
        self.state.pc = insn.pc + insn.size

    def _op_sub(self, insn: DecodedInstruction) -> None:
        dst = insn.metadata["dst"]
        a = self.state.get_reg(dst)
        if "imm" in insn.metadata:
            b = insn.metadata["imm"]
        else:
            b = self.state.get_reg(insn.metadata["src"])
        res = u32(a - b)
        # borrow / unsigned overflow
        carry = a < b
        # signed overflow for subtraction
        sa, sb, sr = bool(a & 0x80000000), bool(b & 0x80000000), bool(res & 0x80000000)
        overflow = sa != sb and sa != sr
        self._write_reg(dst, res, pc=insn.pc)
        self.state.set_arith_flags(res, carry=carry, overflow=overflow)
        self._set_flags(
            pc=insn.pc,
            carry=carry,
            overflow=overflow,
            sign=self.state.sign,
            zero=self.state.zero,
        )
        self.state.pc = insn.pc + insn.size

    def _op_bra(self, insn: DecodedInstruction) -> None:
        cc = insn.metadata["cc"]
        target = insn.branch_target
        assert target is not None
        if self._eval_cc(cc):
            if target != insn.pc and (
                target < self.state.imem.base or target >= self.state.imem.end
            ):
                # allow helper targets and absolute firmware calls
                if target not in self.helpers:
                    # relative branches inside pad must stay in image; always bra may self-loop
                    if cc != "always":
                        raise InvalidBranchTarget(
                            "branch target outside IMEM",
                            pc=insn.pc,
                            details={"target": target},
                        )
            self.state.pc = target
        else:
            self.state.pc = insn.pc + insn.size

    def _op_call(self, insn: DecodedInstruction) -> None:
        target = insn.branch_target
        assert target is not None
        ret = insn.pc + insn.size
        self.state.sp = self.state.sp - 4
        self.state.dmem.write_u32(self.state.sp, ret)
        self.trace.emit(
            EventKind.CALL,
            pc=insn.pc,
            address=target,
            value=ret,
            source_symbol=self.symbol_at(insn.pc),
        )
        self.state.pc = target

    def _op_ret(self, insn: DecodedInstruction) -> None:
        ret = self.state.dmem.read_u32(self.state.sp)
        self.state.sp = self.state.sp + 4
        self.trace.emit(
            EventKind.RETURN,
            pc=insn.pc,
            address=ret,
            source_symbol=self.symbol_at(insn.pc),
        )
        self.state.pc = ret

    def _op_iowr(self, insn: DecodedInstruction) -> None:
        base = self.state.get_reg(insn.metadata["base"])
        idx = insn.metadata["idx"]
        src = self.state.get_reg(insn.metadata["src"])
        addr = u32(base + idx * 4)
        self.bus.mmio_write32(addr, src)
        # Historical: nv_iowr / iowr helpers clear $r0; direct iowr insn does not
        # by itself — the stock helper at 0x34 does. Keep insn semantics clean.
        self.state.pc = insn.pc + insn.size

    def _op_iord(self, insn: DecodedInstruction) -> None:
        base = self.state.get_reg(insn.metadata["base"])
        idx = insn.metadata["idx"]
        addr = u32(base + idx * 4)
        val = self.bus.mmio_read32(addr)
        self._write_reg(insn.metadata["dst"], val, pc=insn.pc)
        self.state.pc = insn.pc + insn.size

    def _op_xdst(self, insn: DecodedInstruction) -> None:
        src1 = self.state.get_reg(insn.metadata["src1"])  # ext_offset
        src2 = self.state.get_reg(insn.metadata["src2"])  # local | size<<16
        local = src2 & 0xFFFF
        size_code = (src2 >> 16) & 7
        size = 4 << size_code
        ext_base = self.state.xdbase
        ext_addr = u32((ext_base << 8) + src1)
        port = (self.state.xtargets >> 12) & 7
        req = XferRequest(
            source_space="dmem",
            source_address=local,
            destination_space="direct_vram",
            destination_address=ext_addr,
            size=size,
            direction="dmem_to_direct_vram",
            port=port,
            target_mode="direct_vram",
        )
        token = self.bus.xfer_start(req)
        self.state.specials["_last_xfer_token"] = token
        self.state.pc = insn.pc + insn.size

    def _op_xdwait(self, insn: DecodedInstruction) -> None:
        token = self.state.specials.get("_last_xfer_token")
        if token is None:
            raise DeviceModelError("xdwait with no prior xdst", pc=insn.pc)
        self.trace.emit(
            EventKind.WAIT_BEGIN,
            pc=insn.pc,
            metadata={"token": token},
            source_symbol=self.symbol_at(insn.pc),
        )
        for _ in range(self.max_xdwait_polls):
            status = self.bus.xfer_poll(token)
            if status.phase is XferPhase.COMPLETE:
                self.trace.emit(
                    EventKind.WAIT_END,
                    pc=insn.pc,
                    metadata={"token": token, "result": "complete"},
                )
                self.state.pc = insn.pc + insn.size
                return
            if status.phase is XferPhase.FAULT:
                raise XferTimeout(
                    "XFER fault during xdwait",
                    pc=insn.pc,
                    details={"token": token, "reason": status.fault_reason},
                )
            if hasattr(self.bus, "advance"):
                self.bus.advance(1)
            self.state.logical_step += 1
        raise XferTimeout("xdwait exceeded poll limit", pc=insn.pc, details={"token": token})

    def _op_jmp(self, insn: DecodedInstruction) -> None:
        self.state.pc = insn.branch_target  # type: ignore[assignment]


def nv_wr32_helper(exe: FalconExecutor) -> None:
    """Stock PMU ``wr32`` at call target 0x34 (kernel.fuc).

    Sequence (Nouveau macros.fuc + kernel.fuc, unshifted IO)::

        iowr MMIO_ADDR, $r14
        iowr MMIO_DATA, $r13
        iowr MMIO_CTRL, OP_WR|MASK_B32|TRIGGER
        # wait STATUS idle (immediate in oracle)
        clear $r0   # nv_iowr macro side effect
        ret
    """
    from .devices.pmu_mmio import (
        CTRL_MASK_B32_0,
        CTRL_OP_WR,
        CTRL_TRIGGER,
        MMIO_ADDR,
        MMIO_CTRL,
        MMIO_DATA,
    )

    addr = exe.state.get_reg(14)
    val = exe.state.get_reg(13)
    # Each nv_iowr clears $r0 after moving the Falcon IO index into it.
    exe.bus.mmio_write32(MMIO_ADDR, addr)
    exe._write_reg(0, 0, pc=0x34)
    exe.bus.mmio_write32(MMIO_DATA, val)
    exe._write_reg(0, 0, pc=0x34)
    exe.bus.mmio_write32(MMIO_CTRL, CTRL_OP_WR | CTRL_MASK_B32_0 | CTRL_TRIGGER)
    exe._write_reg(0, 0, pc=0x34)
    ret = exe.state.dmem.read_u32(exe.state.sp)
    exe.state.sp = exe.state.sp + 4
    exe.trace.emit(EventKind.RETURN, pc=0x34, address=ret, metadata={"helper": "nv_wr32"})
    exe.state.pc = ret


def nv_rd32_helper(exe: FalconExecutor) -> None:
    """Stock PMU ``rd32``: result in $r13; clears $r0."""
    from .devices.pmu_mmio import CTRL_OP_RD, CTRL_TRIGGER, MMIO_ADDR, MMIO_CTRL, MMIO_DATA

    addr = exe.state.get_reg(14)
    exe.bus.mmio_write32(MMIO_ADDR, addr)
    exe._write_reg(0, 0, pc=0x30)
    exe.bus.mmio_write32(MMIO_CTRL, CTRL_OP_RD | CTRL_TRIGGER)
    exe._write_reg(0, 0, pc=0x30)
    data = exe.bus.mmio_read32(MMIO_DATA)
    exe._write_reg(13, data, pc=0x30)
    exe._write_reg(0, 0, pc=0x30)
    ret = exe.state.dmem.read_u32(exe.state.sp)
    exe.state.sp = exe.state.sp + 4
    exe.trace.emit(EventKind.RETURN, pc=0x30, address=ret, metadata={"helper": "nv_rd32"})
    exe.state.pc = ret
