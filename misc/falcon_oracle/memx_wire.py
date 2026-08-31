"""Nouveau ``memx.c`` wire encoding for MEMX command scripts.

Host packets written through ``0x10a1c4``:

  header = (size << 16) | mthd
  followed by ``size`` data words

Coalescing matches ``memx_cmd`` / ``memx_out``: same-mthd WR32 pairs merge
until the 64-word staging buffer would overflow; WAIT/DELAY flush immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memx_script import MemxCommand, MemxOp, parse_memx_commands
from .values import u32

# Nouveau memx.c staging array size.
MEMX_STAGING_WORDS = 64
# DMEM data segment size from memx.fuc (``memx_data_head`` skip 0x0800).
MEMX_DATA_BYTES = 0x0800
MEMX_DATA_WORDS = MEMX_DATA_BYTES // 4


@dataclass(frozen=True)
class MemxPacket:
    mthd: int
    data: tuple[int, ...]

    @property
    def header(self) -> int:
        return u32((len(self.data) << 16) | self.mthd)

    @property
    def wire_words(self) -> list[int]:
        return [self.header, *[u32(d) for d in self.data]]


def encode_memx_packets(commands) -> list[MemxPacket]:
    """Encode logical commands into Nouveau wire packets (with WR32 coalesce)."""
    cmds = parse_memx_commands(commands)
    packets: list[MemxPacket] = []
    pending_mthd: int | None = None
    pending_data: list[int] = []

    def flush() -> None:
        nonlocal pending_mthd, pending_data
        if pending_mthd is None:
            return
        packets.append(MemxPacket(pending_mthd, tuple(pending_data)))
        pending_mthd = None
        pending_data = []

    def emit(mthd: int, data: list[int], *, force_flush: bool = False) -> None:
        nonlocal pending_mthd, pending_data
        if pending_mthd is not None and (
            pending_mthd != mthd or len(pending_data) + len(data) >= MEMX_STAGING_WORDS
        ):
            flush()
        if pending_mthd is None:
            pending_mthd = mthd
            pending_data = list(data)
        else:
            pending_data.extend(data)
        if force_flush or mthd in (MemxOp.WAIT, MemxOp.DELAY, MemxOp.ENTER, MemxOp.LEAVE):
            # WAIT/DELAY always flush (memx.c); ENTER/LEAVE are size-0 singleton.
            flush()

    for cmd in cmds:
        if cmd.op is MemxOp.ENTER:
            emit(int(MemxOp.ENTER), [], force_flush=True)
        elif cmd.op is MemxOp.LEAVE:
            emit(int(MemxOp.LEAVE), [], force_flush=True)
        elif cmd.op is MemxOp.WR32:
            # Feed pairs; coalesce across adjacent WR32 commands.
            args = list(cmd.args)
            for i in range(0, len(args), 2):
                emit(int(MemxOp.WR32), [args[i], args[i + 1]])
        elif cmd.op is MemxOp.WAIT:
            emit(int(MemxOp.WAIT), list(cmd.args), force_flush=True)
        elif cmd.op is MemxOp.DELAY:
            emit(int(MemxOp.DELAY), list(cmd.args), force_flush=True)
        else:
            raise ValueError(f"unsupported MEMX op {cmd.op}")
    flush()
    return packets


def encode_memx_words(commands) -> list[int]:
    words: list[int] = []
    for pkt in encode_memx_packets(commands):
        words.extend(pkt.wire_words)
    return words


def decode_memx_words(words: list[int]) -> list[MemxCommand]:
    """Decode wire words back into logical MemxCommand list."""
    out: list[MemxCommand] = []
    i = 0
    while i < len(words):
        header = u32(words[i])
        i += 1
        mthd = header & 0xFFFF
        size = (header >> 16) & 0xFFFF
        if i + size > len(words):
            raise ValueError(f"truncated packet mthd={mthd} size={size} at word {i-1}")
        data = tuple(u32(words[j]) for j in range(i, i + size))
        i += size
        if mthd == MemxOp.ENTER:
            if size:
                raise ValueError("ENTER packet must have size 0")
            out.append(MemxCommand.enter())
        elif mthd == MemxOp.LEAVE:
            if size:
                raise ValueError("LEAVE packet must have size 0")
            out.append(MemxCommand.leave())
        elif mthd == MemxOp.WR32:
            if size < 2 or size % 2:
                raise ValueError(f"WR32 size must be even >=2, got {size}")
            out.append(MemxCommand.wr32(*data))
        elif mthd == MemxOp.WAIT:
            if size != 4:
                raise ValueError(f"WAIT size must be 4, got {size}")
            out.append(MemxCommand.wait(*data))
        elif mthd == MemxOp.DELAY:
            if size != 1:
                raise ValueError(f"DELAY size must be 1, got {size}")
            out.append(MemxCommand.delay(data[0]))
        else:
            raise ValueError(f"unknown MEMX mthd {mthd}")
    return out


def wire_footprint_words(commands) -> int:
    return len(encode_memx_words(commands))
