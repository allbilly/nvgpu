#!/usr/bin/env python3
"""
Python port of the Nouveau ctxprog generator from ctxnv50.c.

Generates the ctxprog microcode (uploaded to GPU register 0x400328)
and the ctxvals buffer (default register values) for NV50-family GPUs.

Original C code: Copyright 2009 Marcin Kościelnicki, MIT licensed.
"""

import struct

# ---------------------------------------------------------------------------
# Flag definitions (from ctxnv50.c lines 23-75)
# ---------------------------------------------------------------------------

CP_FLAG_CLEAR                 = 0
CP_FLAG_SET                   = 1

CP_FLAG_SWAP_DIRECTION        = (0 * 32) + 0
CP_FLAG_SWAP_DIRECTION_LOAD   = 0
CP_FLAG_SWAP_DIRECTION_SAVE   = 1

CP_FLAG_UNK01                 = (0 * 32) + 1
CP_FLAG_UNK01_CLEAR           = 0
CP_FLAG_UNK01_SET             = 1

CP_FLAG_UNK03                 = (0 * 32) + 3
CP_FLAG_UNK03_CLEAR           = 0
CP_FLAG_UNK03_SET             = 1

CP_FLAG_USER_SAVE             = (0 * 32) + 5
CP_FLAG_USER_SAVE_NOT_PENDING = 0
CP_FLAG_USER_SAVE_PENDING     = 1

CP_FLAG_USER_LOAD             = (0 * 32) + 6
CP_FLAG_USER_LOAD_NOT_PENDING = 0
CP_FLAG_USER_LOAD_PENDING     = 1

CP_FLAG_UNK0B                 = (0 * 32) + 0xb
CP_FLAG_UNK0B_CLEAR           = 0
CP_FLAG_UNK0B_SET             = 1

CP_FLAG_XFER_SWITCH           = (0 * 32) + 0xe
CP_FLAG_XFER_SWITCH_DISABLE   = 0
CP_FLAG_XFER_SWITCH_ENABLE    = 1

CP_FLAG_STATE                 = (0 * 32) + 0x1c
CP_FLAG_STATE_STOPPED         = 0
CP_FLAG_STATE_RUNNING         = 1

CP_FLAG_UNK1D                 = (0 * 32) + 0x1d
CP_FLAG_UNK1D_CLEAR           = 0
CP_FLAG_UNK1D_SET             = 1

CP_FLAG_UNK20                 = (1 * 32) + 0
CP_FLAG_UNK20_CLEAR           = 0
CP_FLAG_UNK20_SET             = 1

CP_FLAG_STATUS                = (2 * 32) + 0
CP_FLAG_STATUS_BUSY           = 0
CP_FLAG_STATUS_IDLE           = 1

CP_FLAG_AUTO_SAVE             = (2 * 32) + 4
CP_FLAG_AUTO_SAVE_NOT_PENDING = 0
CP_FLAG_AUTO_SAVE_PENDING     = 1

CP_FLAG_AUTO_LOAD             = (2 * 32) + 5
CP_FLAG_AUTO_LOAD_NOT_PENDING = 0
CP_FLAG_AUTO_LOAD_PENDING     = 1

CP_FLAG_NEWCTX                = (2 * 32) + 10
CP_FLAG_NEWCTX_BUSY           = 0
CP_FLAG_NEWCTX_DONE           = 1

CP_FLAG_XFER                  = (2 * 32) + 11
CP_FLAG_XFER_IDLE             = 0
CP_FLAG_XFER_BUSY             = 1

CP_FLAG_ALWAYS                = (2 * 32) + 13
CP_FLAG_ALWAYS_FALSE          = 0
CP_FLAG_ALWAYS_TRUE           = 1

CP_FLAG_INTR                  = (2 * 32) + 15
CP_FLAG_INTR_NOT_PENDING      = 0
CP_FLAG_INTR_PENDING          = 1

# ---------------------------------------------------------------------------
# Instruction opcodes (from ctxnv50.c lines 77-106)
# ---------------------------------------------------------------------------

CP_CTX                   = 0x00100000
CP_CTX_COUNT             = 0x000f0000
CP_CTX_COUNT_SHIFT       = 16
CP_CTX_REG               = 0x00003fff
CP_LOAD_SR               = 0x00200000
CP_LOAD_SR_VALUE         = 0x000fffff
CP_BRA                   = 0x00400000
CP_BRA_IP                = 0x0001ff00
CP_BRA_IP_SHIFT          = 8
CP_BRA_IF_CLEAR          = 0x00000080
CP_BRA_FLAG              = 0x0000007f
CP_WAIT                  = 0x00500000
CP_WAIT_SET              = 0x00000080
CP_WAIT_FLAG             = 0x0000007f
CP_SET                   = 0x00700000
CP_SET_1                 = 0x00000080
CP_SET_FLAG              = 0x0000007f
CP_NEWCTX                = 0x00600004
CP_NEXT_TO_SWAP          = 0x00600005
CP_SET_CONTEXT_POINTER   = 0x00600006
CP_SET_XFER_POINTER      = 0x00600007
CP_ENABLE                = 0x00600009
CP_END                   = 0x0060000c
CP_NEXT_TO_CURRENT       = 0x0060000d
CP_DISABLE1              = 0x0090ffff
CP_DISABLE2              = 0x0091ffff
CP_XFER_1                = 0x008000ff
CP_XFER_2                = 0x008800ff
CP_SEEK_1                = 0x00c000ff
CP_SEEK_2                = 0x00c800ff

# ---------------------------------------------------------------------------
# Label enum (from ctxnv50.c lines 160-168)
# ---------------------------------------------------------------------------

cp_check_load       = 1
cp_setup_auto_load  = 2
cp_setup_load       = 3
cp_setup_save       = 4
cp_swap_state       = 5
cp_prepare_exit     = 6
cp_exit             = 7

# ---------------------------------------------------------------------------
# Chipset helpers (from ctxnv50.c lines 113-114)
# ---------------------------------------------------------------------------

def IS_NVA3F(x):
    return ((x) > 0xa0 and (x) < 0xaa) or (x) == 0xaf

def IS_NVAAF(x):
    return (x) >= 0xaa and (x) <= 0xac

# RAM type enum value for GDDR5 (from nvkm/subdev/fb.h)
NVKM_RAM_TYPE_GDDR5 = 10

# ---------------------------------------------------------------------------
# Grctx class - holds generator state for both PROG and VALS modes
# ---------------------------------------------------------------------------

PROG = 0  # NVKM_GRCTX_PROG
VALS = 1  # NVKM_GRCTX_VALS

class Grctx:
    def __init__(self, chipset, mode, units=0, ram_type=0):
        self.chipset = chipset
        self.mode = mode
        self.units = units
        self.ram_type = ram_type
        # PROG mode state
        self.ucode = []        # list of 32-bit instructions
        self.ctxprog_max = 512
        self.ctxprog_len = 0
        self.ctxprog_reg = 0
        self.ctxprog_label = [0] * 32
        # VALS mode state
        self.data = bytearray()  # ctxvals buffer
        # Shared state
        self.ctxvals_pos = 0
        self.ctxvals_base = 0

    # --- Helper methods (ported from ctxnv40.h) ---

    def cp_out(self, inst):
        """Emit a raw instruction (PROG mode only)."""
        if self.mode != PROG:
            return
        assert self.ctxprog_len < self.ctxprog_max, "ctxprog overflow"
        self.ucode.append(inst & 0xffffffff)
        self.ctxprog_len += 1

    def cp_lsr(self, val):
        """Emit CP_LOAD_SR with value."""
        self.cp_out(CP_LOAD_SR | (val & CP_LOAD_SR_VALUE))

    def cp_ctx(self, reg, length):
        """Emit CP_CTX instruction and advance ctxvals position."""
        self.ctxprog_reg = (reg - 0x00400000) >> 2
        self.ctxvals_base = self.ctxvals_pos
        self.ctxvals_pos = self.ctxvals_base + length
        if length > (CP_CTX_COUNT >> CP_CTX_COUNT_SHIFT):
            self.cp_lsr(length)
            length = 0
        self.cp_out(CP_CTX | (length << CP_CTX_COUNT_SHIFT) | self.ctxprog_reg)

    def cp_name(self, name):
        """Mark a branch target label (PROG mode only)."""
        if self.mode != PROG:
            return
        self.ctxprog_label[name] = self.ctxprog_len
        # Patch all prior placeholder branches targeting this label
        for i in range(self.ctxprog_len):
            if (self.ucode[i] & 0xfff00000) != 0xff400000:
                continue
            if (self.ucode[i] & CP_BRA_IP) != (name << CP_BRA_IP_SHIFT):
                continue
            self.ucode[i] = (self.ucode[i] & 0x00ff00ff) | \
                            (self.ctxprog_len << CP_BRA_IP_SHIFT)

    def _cp_bra(self, mod, flag, state, name):
        """Internal branch helper. mod: 0=bra, 1=cal, 2=ret."""
        ip = 0
        if mod != 2:
            ip = self.ctxprog_label[name] << CP_BRA_IP_SHIFT
            if ip == 0:
                ip = 0xff000000 | (name << CP_BRA_IP_SHIFT)
        self.cp_out(CP_BRA | (mod << 18) | ip | flag |
                    (0 if state else CP_BRA_IF_CLEAR))

    def cp_bra(self, flag, state, name):
        """Emit branch instruction."""
        self._cp_bra(0, flag, state, name)

    def _cp_wait(self, flag, state):
        self.cp_out(CP_WAIT | flag | (CP_WAIT_SET if state else 0))

    def cp_wait(self, flag, state):
        self._cp_wait(flag, state)

    def _cp_set(self, flag, state):
        self.cp_out(CP_SET | flag | (CP_SET_1 if state else 0))

    def cp_set(self, flag, state):
        self._cp_set(flag, state)

    def cp_pos(self, offset):
        """Set ctxvals position and emit SET_CONTEXT_POINTER."""
        self.ctxvals_pos = offset
        self.ctxvals_base = self.ctxvals_pos
        self.cp_lsr(self.ctxvals_pos)
        self.cp_out(CP_SET_CONTEXT_POINTER)

    def gr_def(self, reg, val):
        """Store default value at current ctxvals position (VALS mode only)."""
        if self.mode != VALS:
            return
        reg = (reg - 0x00400000) // 4
        reg = (reg - self.ctxprog_reg) + self.ctxvals_base
        self._wo32(reg * 4, val)

    def _wo32(self, offset, val):
        """Write a 32-bit value at byte offset in the ctxvals buffer."""
        val = val & 0xffffffff
        needed = offset + 4
        if len(self.data) < needed:
            self.data.extend(b'\x00' * (needed - len(self.data)))
        struct.pack_into('<I', self.data, offset, val)

    def xf_emit(self, num, val):
        """Emit XFER data (advances ctxvals position by num*8)."""
        if val and self.mode == VALS:
            for i in range(num):
                self._wo32(4 * (self.ctxvals_pos + (i << 3)), val)
        self.ctxvals_pos += num << 3

    def dd_emit(self, num, val):
        """Emit ddata (advances ctxvals position by num)."""
        if val and self.mode == VALS:
            for i in range(num):
                self._wo32(4 * (self.ctxvals_pos + i), val)
        self.ctxvals_pos += num


# ---------------------------------------------------------------------------
# nv50_grctx_generate - main function (lines 177-253)
# ---------------------------------------------------------------------------

def nv50_grctx_generate(ctx):
    ctx.cp_set(CP_FLAG_STATE, CP_FLAG_STATE_RUNNING)
    ctx.cp_set(CP_FLAG_XFER_SWITCH, CP_FLAG_XFER_SWITCH_ENABLE)
    # decide whether we're loading/unloading the context
    ctx.cp_bra(CP_FLAG_AUTO_SAVE, CP_FLAG_AUTO_SAVE_PENDING, cp_setup_save)
    ctx.cp_bra(CP_FLAG_USER_SAVE, CP_FLAG_USER_SAVE_PENDING, cp_setup_save)

    ctx.cp_name(cp_check_load)
    ctx.cp_bra(CP_FLAG_AUTO_LOAD, CP_FLAG_AUTO_LOAD_PENDING, cp_setup_auto_load)
    ctx.cp_bra(CP_FLAG_USER_LOAD, CP_FLAG_USER_LOAD_PENDING, cp_setup_load)
    ctx.cp_bra(CP_FLAG_ALWAYS, CP_FLAG_ALWAYS_TRUE, cp_prepare_exit)

    # setup for context load
    ctx.cp_name(cp_setup_auto_load)
    ctx.cp_out(CP_DISABLE1)
    ctx.cp_out(CP_DISABLE2)
    ctx.cp_out(CP_ENABLE)
    ctx.cp_out(CP_NEXT_TO_SWAP)
    ctx.cp_set(CP_FLAG_UNK01, CP_FLAG_UNK01_SET)
    ctx.cp_name(cp_setup_load)
    ctx.cp_out(CP_NEWCTX)
    ctx.cp_wait(CP_FLAG_NEWCTX, CP_FLAG_NEWCTX_BUSY)
    ctx.cp_set(CP_FLAG_UNK1D, CP_FLAG_UNK1D_CLEAR)
    ctx.cp_set(CP_FLAG_SWAP_DIRECTION, CP_FLAG_SWAP_DIRECTION_LOAD)
    ctx.cp_bra(CP_FLAG_UNK0B, CP_FLAG_UNK0B_SET, cp_prepare_exit)
    ctx.cp_bra(CP_FLAG_ALWAYS, CP_FLAG_ALWAYS_TRUE, cp_swap_state)

    # setup for context save
    ctx.cp_name(cp_setup_save)
    ctx.cp_set(CP_FLAG_UNK1D, CP_FLAG_UNK1D_SET)
    ctx.cp_wait(CP_FLAG_STATUS, CP_FLAG_STATUS_BUSY)
    ctx.cp_wait(CP_FLAG_INTR, CP_FLAG_INTR_PENDING)
    ctx.cp_bra(CP_FLAG_STATUS, CP_FLAG_STATUS_BUSY, cp_setup_save)
    ctx.cp_set(CP_FLAG_UNK01, CP_FLAG_UNK01_SET)
    ctx.cp_set(CP_FLAG_SWAP_DIRECTION, CP_FLAG_SWAP_DIRECTION_SAVE)

    # general PGRAPH state
    ctx.cp_name(cp_swap_state)
    ctx.cp_set(CP_FLAG_UNK03, CP_FLAG_UNK03_SET)
    ctx.cp_pos(0x00004 // 4)
    ctx.cp_ctx(0x400828, 1)  # needed. otherwise, flickering happens.
    ctx.cp_pos(0x00100 // 4)
    nv50_gr_construct_mmio(ctx)
    nv50_gr_construct_xfer1(ctx)
    nv50_gr_construct_xfer2(ctx)

    ctx.cp_bra(CP_FLAG_SWAP_DIRECTION, CP_FLAG_SWAP_DIRECTION_SAVE, cp_check_load)

    ctx.cp_set(CP_FLAG_UNK20, CP_FLAG_UNK20_SET)
    ctx.cp_set(CP_FLAG_SWAP_DIRECTION, CP_FLAG_SWAP_DIRECTION_SAVE)
    ctx.cp_lsr(ctx.ctxvals_base)
    ctx.cp_out(CP_SET_XFER_POINTER)
    ctx.cp_lsr(4)
    ctx.cp_out(CP_SEEK_1)
    ctx.cp_out(CP_XFER_1)
    ctx.cp_wait(CP_FLAG_XFER, CP_FLAG_XFER_BUSY)

    # pre-exit state updates
    ctx.cp_name(cp_prepare_exit)
    ctx.cp_set(CP_FLAG_UNK01, CP_FLAG_UNK01_CLEAR)
    ctx.cp_set(CP_FLAG_UNK03, CP_FLAG_UNK03_CLEAR)
    ctx.cp_set(CP_FLAG_UNK1D, CP_FLAG_UNK1D_CLEAR)

    ctx.cp_bra(CP_FLAG_USER_SAVE, CP_FLAG_USER_SAVE_PENDING, cp_exit)
    ctx.cp_out(CP_NEXT_TO_CURRENT)

    ctx.cp_name(cp_exit)
    ctx.cp_set(CP_FLAG_USER_SAVE, CP_FLAG_USER_SAVE_NOT_PENDING)
    ctx.cp_set(CP_FLAG_USER_LOAD, CP_FLAG_USER_LOAD_NOT_PENDING)
    ctx.cp_set(CP_FLAG_XFER_SWITCH, CP_FLAG_XFER_SWITCH_DISABLE)
    ctx.cp_set(CP_FLAG_STATE, CP_FLAG_STATE_STOPPED)
    ctx.cp_out(CP_END)
    ctx.ctxvals_pos += 0x400  # padding... no idea why you need it


# ---------------------------------------------------------------------------
# nv50_gr_construct_mmio (lines 297-782)
# ---------------------------------------------------------------------------

def nv50_gr_construct_mmio(ctx):
    chipset = ctx.chipset
    units = ctx.units

    # 0800: DISPATCH
    ctx.cp_ctx(0x400808, 7)
    ctx.gr_def(0x400814, 0x00000030)
    ctx.cp_ctx(0x400834, 0x32)
    if chipset == 0x50:
        ctx.gr_def(0x400834, 0xff400040)
        ctx.gr_def(0x400838, 0xfff00080)
        ctx.gr_def(0x40083c, 0xfff70090)
        ctx.gr_def(0x400840, 0xffe806a8)
    ctx.gr_def(0x400844, 0x00000002)
    if IS_NVA3F(chipset):
        ctx.gr_def(0x400894, 0x00001000)
    ctx.gr_def(0x4008e8, 0x00000003)
    ctx.gr_def(0x4008ec, 0x00001000)
    if chipset == 0x50:
        ctx.cp_ctx(0x400908, 0xb)
    elif chipset < 0xa0:
        ctx.cp_ctx(0x400908, 0xc)
    else:
        ctx.cp_ctx(0x400908, 0xe)

    if chipset >= 0xa0:
        ctx.cp_ctx(0x400b00, 0x1)
    if IS_NVA3F(chipset):
        ctx.cp_ctx(0x400b10, 0x1)
        ctx.gr_def(0x400b10, 0x0001629d)
        ctx.cp_ctx(0x400b20, 0x1)
        ctx.gr_def(0x400b20, 0x0001629d)

    nv50_gr_construct_mmio_ddata(ctx)

    # 0C00: VFETCH
    ctx.cp_ctx(0x400c08, 0x2)
    ctx.gr_def(0x400c08, 0x0000fe0c)

    # 1000
    if chipset < 0xa0:
        ctx.cp_ctx(0x401008, 0x4)
        ctx.gr_def(0x401014, 0x00001000)
    elif not IS_NVA3F(chipset):
        ctx.cp_ctx(0x401008, 0x5)
        ctx.gr_def(0x401018, 0x00001000)
    else:
        ctx.cp_ctx(0x401008, 0x5)
        ctx.gr_def(0x401018, 0x00004000)

    # 1400
    ctx.cp_ctx(0x401400, 0x8)
    ctx.cp_ctx(0x401424, 0x3)
    if chipset == 0x50:
        ctx.gr_def(0x40142c, 0x0001fd87)
    else:
        ctx.gr_def(0x40142c, 0x00000187)
    ctx.cp_ctx(0x401540, 0x5)
    ctx.gr_def(0x401550, 0x00001018)

    # 1800: STREAMOUT
    ctx.cp_ctx(0x401814, 0x1)
    ctx.gr_def(0x401814, 0x000000ff)
    if chipset == 0x50:
        ctx.cp_ctx(0x40181c, 0xe)
        ctx.gr_def(0x401850, 0x00000004)
    elif chipset < 0xa0:
        ctx.cp_ctx(0x40181c, 0xf)
        ctx.gr_def(0x401854, 0x00000004)
    else:
        ctx.cp_ctx(0x40181c, 0x13)
        ctx.gr_def(0x401864, 0x00000004)

    # 1C00
    ctx.cp_ctx(0x401c00, 0x1)
    if chipset == 0x50:
        ctx.gr_def(0x401c00, 0x0001005f)
    elif chipset in (0x84, 0x86, 0x94):
        ctx.gr_def(0x401c00, 0x044d00df)
    elif chipset in (0x92, 0x96, 0x98, 0xa0, 0xaa, 0xac):
        ctx.gr_def(0x401c00, 0x042500df)
    elif chipset in (0xa3, 0xa5, 0xa8, 0xaf):
        ctx.gr_def(0x401c00, 0x142500df)

    # 2000

    # 2400
    ctx.cp_ctx(0x402400, 0x1)
    if chipset == 0x50:
        ctx.cp_ctx(0x402408, 0x1)
    else:
        ctx.cp_ctx(0x402408, 0x2)
    ctx.gr_def(0x402408, 0x00000600)

    # 2800: CSCHED
    ctx.cp_ctx(0x402800, 0x1)
    if chipset == 0x50:
        ctx.gr_def(0x402800, 0x00000006)

    # 2C00: ZCULL
    ctx.cp_ctx(0x402c08, 0x6)
    if chipset != 0x50:
        ctx.gr_def(0x402c14, 0x01000000)
    ctx.gr_def(0x402c18, 0x000000ff)
    if chipset == 0x50:
        ctx.cp_ctx(0x402ca0, 0x1)
    else:
        ctx.cp_ctx(0x402ca0, 0x2)
    if chipset < 0xa0:
        ctx.gr_def(0x402ca0, 0x00000400)
    elif not IS_NVA3F(chipset):
        ctx.gr_def(0x402ca0, 0x00000800)
    else:
        ctx.gr_def(0x402ca0, 0x00000400)
    ctx.cp_ctx(0x402cac, 0x4)

    # 3000: ENG2D
    ctx.cp_ctx(0x403004, 0x1)
    ctx.gr_def(0x403004, 0x00000001)

    # 3400
    if chipset >= 0xa0:
        ctx.cp_ctx(0x403404, 0x1)
        ctx.gr_def(0x403404, 0x00000001)

    # 5000: CCACHE
    ctx.cp_ctx(0x405000, 0x1)
    if chipset == 0x50:
        ctx.gr_def(0x405000, 0x00300080)
    elif chipset in (0x84, 0xa0, 0xa3, 0xa5, 0xa8, 0xaa, 0xac, 0xaf):
        ctx.gr_def(0x405000, 0x000e0080)
    elif chipset in (0x86, 0x92, 0x94, 0x96, 0x98):
        ctx.gr_def(0x405000, 0x00000080)
    ctx.cp_ctx(0x405014, 0x1)
    ctx.gr_def(0x405014, 0x00000004)
    ctx.cp_ctx(0x40501c, 0x1)
    ctx.cp_ctx(0x405024, 0x1)
    ctx.cp_ctx(0x40502c, 0x1)

    # 6000?
    if chipset == 0x50:
        ctx.cp_ctx(0x4063e0, 0x1)

    # 6800: M2MF
    if chipset < 0x90:
        ctx.cp_ctx(0x406814, 0x2b)
        ctx.gr_def(0x406818, 0x00000f80)
        ctx.gr_def(0x406860, 0x007f0080)
        ctx.gr_def(0x40689c, 0x007f0080)
    else:
        ctx.cp_ctx(0x406814, 0x4)
        if chipset == 0x98:
            ctx.gr_def(0x406818, 0x00000f80)
        else:
            ctx.gr_def(0x406818, 0x00001f80)
        if IS_NVA3F(chipset):
            ctx.gr_def(0x40681c, 0x00000030)
        ctx.cp_ctx(0x406830, 0x3)

    # 7000: per-ROP group state
    for i in range(8):
        if units & (1 << (i + 16)):
            ctx.cp_ctx(0x407000 + (i << 8), 3)
            if chipset == 0x50:
                ctx.gr_def(0x407000 + (i << 8), 0x1b74f820)
            elif chipset != 0xa5:
                ctx.gr_def(0x407000 + (i << 8), 0x3b74f821)
            else:
                ctx.gr_def(0x407000 + (i << 8), 0x7b74f821)
            ctx.gr_def(0x407004 + (i << 8), 0x89058001)

            if chipset == 0x50:
                ctx.cp_ctx(0x407010 + (i << 8), 1)
            elif chipset < 0xa0:
                ctx.cp_ctx(0x407010 + (i << 8), 2)
                ctx.gr_def(0x407010 + (i << 8), 0x00001000)
                ctx.gr_def(0x407014 + (i << 8), 0x0000001f)
            else:
                ctx.cp_ctx(0x407010 + (i << 8), 3)
                ctx.gr_def(0x407010 + (i << 8), 0x00001000)
                if chipset != 0xa5:
                    ctx.gr_def(0x407014 + (i << 8), 0x000000ff)
                else:
                    ctx.gr_def(0x407014 + (i << 8), 0x000001ff)

            ctx.cp_ctx(0x407080 + (i << 8), 4)
            if chipset != 0xa5:
                ctx.gr_def(0x407080 + (i << 8), 0x027c10fa)
            else:
                ctx.gr_def(0x407080 + (i << 8), 0x827c10fa)
            if chipset == 0x50:
                ctx.gr_def(0x407084 + (i << 8), 0x000000c0)
            else:
                ctx.gr_def(0x407084 + (i << 8), 0x400000c0)
            ctx.gr_def(0x407088 + (i << 8), 0xb7892080)

            if chipset < 0xa0:
                ctx.cp_ctx(0x407094 + (i << 8), 1)
            elif not IS_NVA3F(chipset):
                ctx.cp_ctx(0x407094 + (i << 8), 3)
            else:
                ctx.cp_ctx(0x407094 + (i << 8), 4)
                ctx.gr_def(0x4070a0 + (i << 8), 1)

    ctx.cp_ctx(0x407c00, 0x3)
    if chipset < 0x90:
        ctx.gr_def(0x407c00, 0x00010040)
    elif chipset < 0xa0:
        ctx.gr_def(0x407c00, 0x00390040)
    else:
        ctx.gr_def(0x407c00, 0x003d0040)
    ctx.gr_def(0x407c08, 0x00000022)
    if chipset >= 0xa0:
        ctx.cp_ctx(0x407c10, 0x3)
        ctx.cp_ctx(0x407c20, 0x1)
        ctx.cp_ctx(0x407c2c, 0x1)

    if chipset < 0xa0:
        ctx.cp_ctx(0x407d00, 0x9)
    else:
        ctx.cp_ctx(0x407d00, 0x15)
    if chipset == 0x98:
        ctx.gr_def(0x407d08, 0x00380040)
    else:
        if chipset < 0x90:
            ctx.gr_def(0x407d08, 0x00010040)
        elif chipset < 0xa0:
            ctx.gr_def(0x407d08, 0x00390040)
        else:
            if ctx.ram_type != NVKM_RAM_TYPE_GDDR5:
                ctx.gr_def(0x407d08, 0x003d0040)
            else:
                ctx.gr_def(0x407d08, 0x003c0040)
        ctx.gr_def(0x407d0c, 0x00000022)

    # 8000+: per-TP state
    for i in range(10):
        if units & (1 << i):
            if chipset < 0xa0:
                base = 0x408000 + (i << 12)
            else:
                base = 0x408000 + (i << 11)
            if chipset < 0xa0:
                offset = base + 0xc00
            else:
                offset = base + 0x80
            ctx.cp_ctx(offset + 0x00, 1)
            ctx.gr_def(offset + 0x00, 0x0000ff0a)
            ctx.cp_ctx(offset + 0x08, 1)

            # per-MP state
            for j in range(2 if chipset < 0xa0 else 4):
                if not (units & (1 << (j + 24))):
                    continue
                if chipset < 0xa0:
                    offset = base + 0x200 + (j << 7)
                else:
                    offset = base + 0x100 + (j << 7)
                ctx.cp_ctx(offset, 0x20)
                ctx.gr_def(offset + 0x00, 0x01800000)
                ctx.gr_def(offset + 0x04, 0x00160000)
                ctx.gr_def(offset + 0x08, 0x01800000)
                ctx.gr_def(offset + 0x18, 0x0003ffff)
                if chipset == 0x50:
                    ctx.gr_def(offset + 0x1c, 0x00080000)
                elif chipset == 0x84:
                    ctx.gr_def(offset + 0x1c, 0x00880000)
                elif chipset == 0x86:
                    ctx.gr_def(offset + 0x1c, 0x018c0000)
                elif chipset in (0x92, 0x96, 0x98):
                    ctx.gr_def(offset + 0x1c, 0x118c0000)
                elif chipset == 0x94:
                    ctx.gr_def(offset + 0x1c, 0x10880000)
                elif chipset in (0xa0, 0xa5):
                    ctx.gr_def(offset + 0x1c, 0x310c0000)
                elif chipset in (0xa3, 0xa8, 0xaa, 0xac, 0xaf):
                    ctx.gr_def(offset + 0x1c, 0x300c0000)
                ctx.gr_def(offset + 0x40, 0x00010401)
                if chipset == 0x50:
                    ctx.gr_def(offset + 0x48, 0x00000040)
                else:
                    ctx.gr_def(offset + 0x48, 0x00000078)
                ctx.gr_def(offset + 0x50, 0x000000bf)
                ctx.gr_def(offset + 0x58, 0x00001210)
                if chipset == 0x50:
                    ctx.gr_def(offset + 0x5c, 0x00000080)
                else:
                    ctx.gr_def(offset + 0x5c, 0x08000080)
                if chipset >= 0xa0:
                    ctx.gr_def(offset + 0x68, 0x0000003e)

            if chipset < 0xa0:
                ctx.cp_ctx(base + 0x300, 0x4)
            else:
                ctx.cp_ctx(base + 0x300, 0x5)
            if chipset == 0x50:
                ctx.gr_def(base + 0x304, 0x00007070)
            elif chipset < 0xa0:
                ctx.gr_def(base + 0x304, 0x00027070)
            elif not IS_NVA3F(chipset):
                ctx.gr_def(base + 0x304, 0x01127070)
            else:
                ctx.gr_def(base + 0x304, 0x05127070)

            if chipset < 0xa0:
                ctx.cp_ctx(base + 0x318, 1)
            else:
                ctx.cp_ctx(base + 0x320, 1)
            if chipset == 0x50:
                ctx.gr_def(base + 0x318, 0x0003ffff)
            elif chipset < 0xa0:
                ctx.gr_def(base + 0x318, 0x03ffffff)
            else:
                ctx.gr_def(base + 0x320, 0x07ffffff)

            if chipset < 0xa0:
                ctx.cp_ctx(base + 0x324, 5)
            else:
                ctx.cp_ctx(base + 0x328, 4)

            if chipset < 0xa0:
                ctx.cp_ctx(base + 0x340, 9)
                offset = base + 0x340
            elif not IS_NVA3F(chipset):
                ctx.cp_ctx(base + 0x33c, 0xb)
                offset = base + 0x344
            else:
                ctx.cp_ctx(base + 0x33c, 0xd)
                offset = base + 0x344
            ctx.gr_def(offset + 0x0, 0x00120407)
            ctx.gr_def(offset + 0x4, 0x05091507)
            if chipset == 0x84:
                ctx.gr_def(offset + 0x8, 0x05100202)
            else:
                ctx.gr_def(offset + 0x8, 0x05010202)
            ctx.gr_def(offset + 0xc, 0x00030201)
            if chipset == 0xa3:
                ctx.cp_ctx(base + 0x36c, 1)

            ctx.cp_ctx(base + 0x400, 2)
            ctx.gr_def(base + 0x404, 0x00000040)
            ctx.cp_ctx(base + 0x40c, 2)
            ctx.gr_def(base + 0x40c, 0x0d0c0b0a)
            ctx.gr_def(base + 0x410, 0x00141210)

            if chipset < 0xa0:
                offset = base + 0x800
            else:
                offset = base + 0x500
            ctx.cp_ctx(offset, 6)
            ctx.gr_def(offset + 0x0, 0x000001f0)
            ctx.gr_def(offset + 0x4, 0x00000001)
            ctx.gr_def(offset + 0x8, 0x00000003)
            if chipset == 0x50 or IS_NVAAF(chipset):
                ctx.gr_def(offset + 0xc, 0x00008000)
            ctx.gr_def(offset + 0x14, 0x00039e00)
            ctx.cp_ctx(offset + 0x1c, 2)
            if chipset == 0x50:
                ctx.gr_def(offset + 0x1c, 0x00000040)
            else:
                ctx.gr_def(offset + 0x1c, 0x00000100)
            ctx.gr_def(offset + 0x20, 0x00003800)

            if chipset >= 0xa0:
                ctx.cp_ctx(base + 0x54c, 2)
                if not IS_NVA3F(chipset):
                    ctx.gr_def(base + 0x54c, 0x003fe006)
                else:
                    ctx.gr_def(base + 0x54c, 0x003fe007)
                ctx.gr_def(base + 0x550, 0x003fe000)

            if chipset < 0xa0:
                offset = base + 0xa00
            else:
                offset = base + 0x680
            ctx.cp_ctx(offset, 1)
            ctx.gr_def(offset, 0x00404040)

            if chipset < 0xa0:
                offset = base + 0xe00
            else:
                offset = base + 0x700
            ctx.cp_ctx(offset, 2)
            if chipset < 0xa0:
                ctx.gr_def(offset, 0x0077f005)
            elif chipset == 0xa5:
                ctx.gr_def(offset, 0x6cf7f007)
            elif chipset == 0xa8:
                ctx.gr_def(offset, 0x6cfff007)
            elif chipset == 0xac:
                ctx.gr_def(offset, 0x0cfff007)
            else:
                ctx.gr_def(offset, 0x0cf7f007)
            if chipset == 0x50:
                ctx.gr_def(offset + 0x4, 0x00007fff)
            elif chipset < 0xa0:
                ctx.gr_def(offset + 0x4, 0x003f7fff)
            else:
                ctx.gr_def(offset + 0x4, 0x02bf7fff)
            ctx.cp_ctx(offset + 0x2c, 1)
            if chipset == 0x50:
                ctx.cp_ctx(offset + 0x50, 9)
                ctx.gr_def(offset + 0x54, 0x000003ff)
                ctx.gr_def(offset + 0x58, 0x00000003)
                ctx.gr_def(offset + 0x5c, 0x00000003)
                ctx.gr_def(offset + 0x60, 0x000001ff)
                ctx.gr_def(offset + 0x64, 0x0000001f)
                ctx.gr_def(offset + 0x68, 0x0000000f)
                ctx.gr_def(offset + 0x6c, 0x0000000f)
            elif chipset < 0xa0:
                ctx.cp_ctx(offset + 0x50, 1)
                ctx.cp_ctx(offset + 0x70, 1)
            else:
                ctx.cp_ctx(offset + 0x50, 1)
                ctx.cp_ctx(offset + 0x60, 5)


# ---------------------------------------------------------------------------
# nv50_gr_construct_mmio_ddata (lines 795-1113)
# ---------------------------------------------------------------------------

def nv50_gr_construct_mmio_ddata(ctx):
    chipset = ctx.chipset
    base = ctx.ctxvals_pos

    # tesla state
    ctx.dd_emit(1, 0)     # 00000001 UNK0F90
    ctx.dd_emit(1, 0)     # 00000001 UNK135C

    # SRC_TIC state
    ctx.dd_emit(1, 0)     # 00000007 SRC_TILE_MODE_Z
    ctx.dd_emit(1, 2)     # 00000007 SRC_TILE_MODE_Y
    ctx.dd_emit(1, 1)     # 00000001 SRC_LINEAR #1
    ctx.dd_emit(1, 0)     # 000000ff SRC_ADDRESS_HIGH
    ctx.dd_emit(1, 0)     # 00000001 SRC_SRGB
    if chipset >= 0x94:
        ctx.dd_emit(1, 0) # 00000003 eng2d UNK0258
    ctx.dd_emit(1, 1)     # 00000fff SRC_DEPTH
    ctx.dd_emit(1, 0x100) # 0000ffff SRC_HEIGHT

    # turing state
    ctx.dd_emit(1, 0)          # 0000000f TEXTURES_LOG2
    ctx.dd_emit(1, 0)          # 0000000f SAMPLERS_LOG2
    ctx.dd_emit(1, 0)          # 000000ff CB_DEF_ADDRESS_HIGH
    ctx.dd_emit(1, 0)          # ffffffff CB_DEF_ADDRESS_LOW
    ctx.dd_emit(1, 0)          # ffffffff SHARED_SIZE
    ctx.dd_emit(1, 2)          # ffffffff REG_MODE
    ctx.dd_emit(1, 1)          # 0000ffff BLOCK_ALLOC_THREADS
    ctx.dd_emit(1, 1)          # 00000001 LANES32
    ctx.dd_emit(1, 0)          # 000000ff UNK370
    ctx.dd_emit(1, 0)          # 000000ff USER_PARAM_UNK
    ctx.dd_emit(1, 0)          # 000000ff USER_PARAM_COUNT
    ctx.dd_emit(1, 1)          # 000000ff UNK384 bits 8-15
    ctx.dd_emit(1, 0x3fffff)   # 003fffff TIC_LIMIT
    ctx.dd_emit(1, 0x1fff)     # 000fffff TSC_LIMIT
    ctx.dd_emit(1, 0)          # 0000ffff CB_ADDR_INDEX
    ctx.dd_emit(1, 1)          # 000007ff BLOCKDIM_X
    ctx.dd_emit(1, 1)          # 000007ff BLOCKDIM_XMY
    ctx.dd_emit(1, 0)          # 00000001 BLOCKDIM_XMY_OVERFLOW
    ctx.dd_emit(1, 1)          # 0003ffff BLOCKDIM_XMYMZ
    ctx.dd_emit(1, 1)          # 000007ff BLOCKDIM_Y
    ctx.dd_emit(1, 1)          # 0000007f BLOCKDIM_Z
    ctx.dd_emit(1, 4)          # 000000ff CP_REG_ALLOC_TEMP
    ctx.dd_emit(1, 1)          # 00000001 BLOCKDIM_DIRTY
    if IS_NVA3F(chipset):
        ctx.dd_emit(1, 0)      # 00000003 UNK03E8
    ctx.dd_emit(1, 1)          # 0000007f BLOCK_ALLOC_HALFWARPS
    ctx.dd_emit(1, 1)          # 00000007 LOCAL_WARPS_NO_CLAMP
    ctx.dd_emit(1, 7)          # 00000007 LOCAL_WARPS_LOG_ALLOC
    ctx.dd_emit(1, 1)          # 00000007 STACK_WARPS_NO_CLAMP
    ctx.dd_emit(1, 7)          # 00000007 STACK_WARPS_LOG_ALLOC
    ctx.dd_emit(1, 1)          # 00001fff BLOCK_ALLOC_REGSLOTS_PACKED
    ctx.dd_emit(1, 1)          # 00001fff BLOCK_ALLOC_REGSLOTS_STRIDED
    ctx.dd_emit(1, 1)          # 000007ff BLOCK_ALLOC_THREADS

    # compat 2d state
    if chipset == 0x50:
        ctx.dd_emit(4, 0)      # 0000ffff clip X, Y, W, H
        ctx.dd_emit(1, 1)      # ffffffff chroma COLOR_FORMAT
        ctx.dd_emit(1, 1)      # ffffffff pattern COLOR_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff pattern SHAPE
        ctx.dd_emit(1, 1)      # ffffffff pattern PATTERN_SELECT
        ctx.dd_emit(1, 0xa)    # ffffffff surf2d SRC_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff surf2d DMA_SRC
        ctx.dd_emit(1, 0)      # 000000ff surf2d SRC_ADDRESS_HIGH
        ctx.dd_emit(1, 0)      # ffffffff surf2d SRC_ADDRESS_LOW
        ctx.dd_emit(1, 0x40)   # 0000ffff surf2d SRC_PITCH
        ctx.dd_emit(1, 0)      # 0000000f surf2d SRC_TILE_MODE_Z
        ctx.dd_emit(1, 2)      # 0000000f surf2d SRC_TILE_MODE_Y
        ctx.dd_emit(1, 0x100)  # ffffffff surf2d SRC_HEIGHT
        ctx.dd_emit(1, 1)      # 00000001 surf2d SRC_LINEAR
        ctx.dd_emit(1, 0x100)  # ffffffff surf2d SRC_WIDTH
        ctx.dd_emit(1, 0)      # 0000ffff gdirect CLIP_B_X
        ctx.dd_emit(1, 0)      # 0000ffff gdirect CLIP_B_Y
        ctx.dd_emit(1, 0)      # 0000ffff gdirect CLIP_C_X
        ctx.dd_emit(1, 0)      # 0000ffff gdirect CLIP_C_Y
        ctx.dd_emit(1, 0)      # 0000ffff gdirect CLIP_D_X
        ctx.dd_emit(1, 0)      # 0000ffff gdirect CLIP_D_Y
        ctx.dd_emit(1, 1)      # ffffffff gdirect COLOR_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff gdirect OPERATION
        ctx.dd_emit(1, 0)      # 0000ffff gdirect POINT_X
        ctx.dd_emit(1, 0)      # 0000ffff gdirect POINT_Y
        ctx.dd_emit(1, 0)      # 0000ffff blit SRC_Y
        ctx.dd_emit(1, 0)      # ffffffff blit OPERATION
        ctx.dd_emit(1, 0)      # ffffffff ifc OPERATION
        ctx.dd_emit(1, 0)      # ffffffff iifc INDEX_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff iifc LUT_OFFSET
        ctx.dd_emit(1, 4)      # ffffffff iifc COLOR_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff iifc OPERATION

    # m2mf state
    ctx.dd_emit(1, 0)      # ffffffff m2mf LINE_COUNT
    ctx.dd_emit(1, 0)      # ffffffff m2mf LINE_LENGTH_IN
    ctx.dd_emit(2, 0)      # ffffffff m2mf OFFSET_IN, OFFSET_OUT
    ctx.dd_emit(1, 1)      # ffffffff m2mf TILING_DEPTH_OUT
    ctx.dd_emit(1, 0x100)  # ffffffff m2mf TILING_HEIGHT_OUT
    ctx.dd_emit(1, 0)      # ffffffff m2mf TILING_POSITION_OUT_Z
    ctx.dd_emit(1, 1)      # 00000001 m2mf LINEAR_OUT
    ctx.dd_emit(2, 0)      # 0000ffff m2mf TILING_POSITION_OUT_X, Y
    ctx.dd_emit(1, 0x100)  # ffffffff m2mf TILING_PITCH_OUT
    ctx.dd_emit(1, 1)      # ffffffff m2mf TILING_DEPTH_IN
    ctx.dd_emit(1, 0x100)  # ffffffff m2mf TILING_HEIGHT_IN
    ctx.dd_emit(1, 0)      # ffffffff m2mf TILING_POSITION_IN_Z
    ctx.dd_emit(1, 1)      # 00000001 m2mf LINEAR_IN
    ctx.dd_emit(2, 0)      # 0000ffff m2mf TILING_POSITION_IN_X, Y
    ctx.dd_emit(1, 0x100)  # ffffffff m2mf TILING_PITCH_IN

    # more compat 2d state
    if chipset == 0x50:
        ctx.dd_emit(1, 1)      # ffffffff line COLOR_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff line OPERATION
        ctx.dd_emit(1, 1)      # ffffffff triangle COLOR_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff triangle OPERATION
        ctx.dd_emit(1, 0)      # 0000000f sifm TILE_MODE_Z
        ctx.dd_emit(1, 2)      # 0000000f sifm TILE_MODE_Y
        ctx.dd_emit(1, 0)      # 000000ff sifm FORMAT_FILTER
        ctx.dd_emit(1, 1)      # 000000ff sifm FORMAT_ORIGIN
        ctx.dd_emit(1, 0)      # 0000ffff sifm SRC_PITCH
        ctx.dd_emit(1, 1)      # 00000001 sifm SRC_LINEAR
        ctx.dd_emit(1, 0)      # 000000ff sifm SRC_OFFSET_HIGH
        ctx.dd_emit(1, 0)      # ffffffff sifm SRC_OFFSET
        ctx.dd_emit(1, 0)      # 0000ffff sifm SRC_HEIGHT
        ctx.dd_emit(1, 0)      # 0000ffff sifm SRC_WIDTH
        ctx.dd_emit(1, 3)      # ffffffff sifm COLOR_FORMAT
        ctx.dd_emit(1, 0)      # ffffffff sifm OPERATION
        ctx.dd_emit(1, 0)      # ffffffff sifc OPERATION

    # tesla state
    ctx.dd_emit(1, 0)      # 0000000f GP_TEXTURES_LOG2
    ctx.dd_emit(1, 0)      # 0000000f GP_SAMPLERS_LOG2
    ctx.dd_emit(1, 0)      # 000000ff
    ctx.dd_emit(1, 0)      # ffffffff
    ctx.dd_emit(1, 4)      # 000000ff UNK12B0_0
    ctx.dd_emit(1, 0x70)   # 000000ff UNK12B0_1
    ctx.dd_emit(1, 0x80)   # 000000ff UNK12B0_3
    ctx.dd_emit(1, 0)      # 000000ff UNK12B0_2
    ctx.dd_emit(1, 0)      # 0000000f FP_TEXTURES_LOG2
    ctx.dd_emit(1, 0)      # 0000000f FP_SAMPLERS_LOG2
    if IS_NVA3F(chipset):
        ctx.dd_emit(1, 0)  # ffffffff
        ctx.dd_emit(1, 0)  # 0000007f MULTISAMPLE_SAMPLES_LOG2
    else:
        ctx.dd_emit(1, 0)  # 0000000f MULTISAMPLE_SAMPLES_LOG2
    ctx.dd_emit(1, 0xc)   # 000000ff SEMANTIC_COLOR.BFC0_ID
    if chipset != 0x50:
        ctx.dd_emit(1, 0)  # 00000001 SEMANTIC_COLOR.CLMP_EN
    ctx.dd_emit(1, 8)      # 000000ff SEMANTIC_COLOR.COLR_NR
    ctx.dd_emit(1, 0x14)   # 000000ff SEMANTIC_COLOR.FFC0_ID
    if chipset == 0x50:
        ctx.dd_emit(1, 0)  # 000000ff SEMANTIC_LAYER
        ctx.dd_emit(1, 0)  # 00000001
    else:
        ctx.dd_emit(1, 0)  # 00000001 SEMANTIC_PTSZ.ENABLE
        ctx.dd_emit(1, 0x29)  # 000000ff SEMANTIC_PTSZ.PTSZ_ID
        ctx.dd_emit(1, 0x27)  # 000000ff SEMANTIC_PRIM
        ctx.dd_emit(1, 0x26)  # 000000ff SEMANTIC_LAYER
        ctx.dd_emit(1, 8)     # 0000000f SMENATIC_CLIP.CLIP_HIGH
        ctx.dd_emit(1, 4)     # 000000ff SEMANTIC_CLIP.CLIP_LO
        ctx.dd_emit(1, 0x27)  # 000000ff UNK0FD4
        ctx.dd_emit(1, 0)     # 00000001 UNK1900
    ctx.dd_emit(1, 0)      # 00000007 RT_CONTROL_MAP0
    ctx.dd_emit(1, 1)      # 00000007 RT_CONTROL_MAP1
    ctx.dd_emit(1, 2)      # 00000007 RT_CONTROL_MAP2
    ctx.dd_emit(1, 3)      # 00000007 RT_CONTROL_MAP3
    ctx.dd_emit(1, 4)      # 00000007 RT_CONTROL_MAP4
    ctx.dd_emit(1, 5)      # 00000007 RT_CONTROL_MAP5
    ctx.dd_emit(1, 6)      # 00000007 RT_CONTROL_MAP6
    ctx.dd_emit(1, 7)      # 00000007 RT_CONTROL_MAP7
    ctx.dd_emit(1, 1)      # 0000000f RT_CONTROL_COUNT
    ctx.dd_emit(8, 0)      # 00000001 RT_HORIZ_UNK
    ctx.dd_emit(8, 0)      # ffffffff RT_ADDRESS_LOW
    ctx.dd_emit(1, 0xcf)   # 000000ff RT_FORMAT
    ctx.dd_emit(7, 0)      # 000000ff RT_FORMAT
    if chipset != 0x50:
        ctx.dd_emit(3, 0)  # 1, 1, 1
    else:
        ctx.dd_emit(2, 0)  # 1, 1
    ctx.dd_emit(1, 0)      # ffffffff GP_ENABLE
    ctx.dd_emit(1, 0x80)   # 0000ffff GP_VERTEX_OUTPUT_COUNT
    ctx.dd_emit(1, 4)      # 000000ff GP_REG_ALLOC_RESULT
    ctx.dd_emit(1, 4)      # 000000ff GP_RESULT_MAP_SIZE
    if IS_NVA3F(chipset):
        ctx.dd_emit(1, 3)  # 00000003
        ctx.dd_emit(1, 0)  # 00000001 UNK1418. Alone.
    if chipset != 0x50:
        ctx.dd_emit(1, 3)  # 00000003 UNK15AC
    ctx.dd_emit(1, 1)      # ffffffff RASTERIZE_ENABLE
    ctx.dd_emit(1, 0)      # 00000001 FP_CONTROL.EXPORTS_Z
    if chipset != 0x50:
        ctx.dd_emit(1, 0)  # 00000001 FP_CONTROL.MULTIPLE_RESULTS
    ctx.dd_emit(1, 0x12)   # 000000ff FP_INTERPOLANT_CTRL.COUNT
    ctx.dd_emit(1, 0x10)   # 000000ff FP_INTERPOLANT_CTRL.COUNT_NONFLAT
    ctx.dd_emit(1, 0xc)    # 000000ff FP_INTERPOLANT_CTRL.OFFSET
    ctx.dd_emit(1, 1)      # 00000001 FP_INTERPOLANT_CTRL.UMASK.W
    ctx.dd_emit(1, 0)      # 00000001 FP_INTERPOLANT_CTRL.UMASK.X
    ctx.dd_emit(1, 0)      # 00000001 FP_INTERPOLANT_CTRL.UMASK.Y
    ctx.dd_emit(1, 0)      # 00000001 FP_INTERPOLANT_CTRL.UMASK.Z
    ctx.dd_emit(1, 4)      # 000000ff FP_RESULT_COUNT
    ctx.dd_emit(1, 2)      # ffffffff REG_MODE
    ctx.dd_emit(1, 4)      # 000000ff FP_REG_ALLOC_TEMP
    if chipset >= 0xa0:
        ctx.dd_emit(1, 0)  # ffffffff
    ctx.dd_emit(1, 0)      # 00000001 GP_BUILTIN_RESULT_EN.LAYER_IDX
    ctx.dd_emit(1, 0)      # ffffffff STRMOUT_ENABLE
    ctx.dd_emit(1, 0x3fffff)  # 003fffff TIC_LIMIT
    ctx.dd_emit(1, 0x1fff)    # 000fffff TSC_LIMIT
    ctx.dd_emit(1, 0)      # 00000001 VERTEX_TWO_SIDE_ENABLE
    if chipset != 0x50:
        ctx.dd_emit(8, 0)  # 00000001
    if chipset >= 0xa0:
        ctx.dd_emit(1, 1)  # 00000007 VTX_ATTR_DEFINE.COMP
        ctx.dd_emit(1, 1)  # 00000007 VTX_ATTR_DEFINE.SIZE
        ctx.dd_emit(1, 2)  # 00000007 VTX_ATTR_DEFINE.TYPE
        ctx.dd_emit(1, 0)  # 000000ff VTX_ATTR_DEFINE.ATTR
    ctx.dd_emit(1, 4)      # 0000007f VP_RESULT_MAP_SIZE
    ctx.dd_emit(1, 0x14)   # 0000001f ZETA_FORMAT
    ctx.dd_emit(1, 1)      # 00000001 ZETA_ENABLE
    ctx.dd_emit(1, 0)      # 0000000f VP_TEXTURES_LOG2
    ctx.dd_emit(1, 0)      # 0000000f VP_SAMPLERS_LOG2
    if IS_NVA3F(chipset):
        ctx.dd_emit(1, 0)  # 00000001
    ctx.dd_emit(1, 2)      # 00000003 POLYGON_MODE_BACK
    if chipset >= 0xa0:
        ctx.dd_emit(1, 0)  # 00000003 VTX_ATTR_DEFINE.SIZE - 1
    ctx.dd_emit(1, 0)      # 0000ffff CB_ADDR_INDEX
    if chipset >= 0xa0:
        ctx.dd_emit(1, 0)  # 00000003
    ctx.dd_emit(1, 0)      # 00000001 CULL_FACE_ENABLE
    ctx.dd_emit(1, 1)      # 00000003 CULL_FACE
    ctx.dd_emit(1, 0)      # 00000001 FRONT_FACE
    ctx.dd_emit(1, 2)      # 00000003 POLYGON_MODE_FRONT
    ctx.dd_emit(1, 0x1000) # 00007fff UNK141C
    if chipset != 0x50:
        ctx.dd_emit(1, 0xe00)   # 7fff
        ctx.dd_emit(1, 0x1000)  # 7fff
        ctx.dd_emit(1, 0x1e00)  # 7fff
    ctx.dd_emit(1, 0)      # 00000001 BEGIN_END_ACTIVE
    ctx.dd_emit(1, 1)      # 00000001 POLYGON_MODE_???
    ctx.dd_emit(1, 1)      # 000000ff GP_REG_ALLOC_TEMP / 4 rounded up
    ctx.dd_emit(1, 1)      # 000000ff FP_REG_ALLOC_TEMP... without /4?
    ctx.dd_emit(1, 1)      # 000000ff VP_REG_ALLOC_TEMP / 4 rounded up
    ctx.dd_emit(1, 1)      # 00000001
    ctx.dd_emit(1, 0)      # 00000001
    ctx.dd_emit(1, 0)      # 00000001 VTX_ATTR_MASK_UNK0 nonempty
    ctx.dd_emit(1, 0)      # 00000001 VTX_ATTR_MASK_UNK1 nonempty
    ctx.dd_emit(1, 0x200)  # 0003ffff GP_VERTEX_OUTPUT_COUNT*GP_REG_ALLOC_RESULT
    if IS_NVA3F(chipset):
        ctx.dd_emit(1, 0x200)
    ctx.dd_emit(1, 0)      # 00000001
    if chipset < 0xa0:
        ctx.dd_emit(1, 1)  # 00000001
        ctx.dd_emit(1, 0x70)  # 000000ff
        ctx.dd_emit(1, 0x80)  # 000000ff
        ctx.dd_emit(1, 0)     # 000000ff
        ctx.dd_emit(1, 0)     # 00000001
        ctx.dd_emit(1, 1)     # 00000001
        ctx.dd_emit(1, 0x70)  # 000000ff
        ctx.dd_emit(1, 0x80)  # 000000ff
        ctx.dd_emit(1, 0)     # 000000ff
    else:
        ctx.dd_emit(1, 1)     # 00000001
        ctx.dd_emit(1, 0xf0)  # 000000ff
        ctx.dd_emit(1, 0xff)  # 000000ff
        ctx.dd_emit(1, 0)     # 000000ff
        ctx.dd_emit(1, 0)     # 00000001
        ctx.dd_emit(1, 1)     # 00000001
        ctx.dd_emit(1, 0xf0)  # 000000ff
        ctx.dd_emit(1, 0xff)  # 000000ff
        ctx.dd_emit(1, 0)     # 000000ff
        ctx.dd_emit(1, 9)     # 0000003f UNK114C.COMP,SIZE

    # eng2d state
    ctx.dd_emit(1, 0)      # 00000001 eng2d COLOR_KEY_ENABLE
    ctx.dd_emit(1, 0)      # 00000007 eng2d COLOR_KEY_FORMAT
    ctx.dd_emit(1, 1)      # ffffffff eng2d DST_DEPTH
    ctx.dd_emit(1, 0xcf)   # 000000ff eng2d DST_FORMAT
    ctx.dd_emit(1, 0)      # ffffffff eng2d DST_LAYER
    ctx.dd_emit(1, 1)      # 00000001 eng2d DST_LINEAR
    ctx.dd_emit(1, 0)      # 00000007 eng2d PATTERN_COLOR_FORMAT
    ctx.dd_emit(1, 0)      # 00000007 eng2d OPERATION
    ctx.dd_emit(1, 0)      # 00000003 eng2d PATTERN_SELECT
    ctx.dd_emit(1, 0xcf)   # 000000ff eng2d SIFC_FORMAT
    ctx.dd_emit(1, 0)      # 00000001 eng2d SIFC_BITMAP_ENABLE
    ctx.dd_emit(1, 2)      # 00000003 eng2d SIFC_BITMAP_UNK808
    ctx.dd_emit(1, 0)      # ffffffff eng2d BLIT_DU_DX_FRACT
    ctx.dd_emit(1, 1)      # ffffffff eng2d BLIT_DU_DX_INT
    ctx.dd_emit(1, 0)      # ffffffff eng2d BLIT_DV_DY_FRACT
    ctx.dd_emit(1, 1)      # ffffffff eng2d BLIT_DV_DY_INT
    ctx.dd_emit(1, 0)      # 00000001 eng2d BLIT_CONTROL_FILTER
    ctx.dd_emit(1, 0xcf)   # 000000ff eng2d DRAW_COLOR_FORMAT
    ctx.dd_emit(1, 0xcf)   # 000000ff eng2d SRC_FORMAT
    ctx.dd_emit(1, 1)      # 00000001 eng2d SRC_LINEAR #2

    num = ctx.ctxvals_pos - base
    ctx.ctxvals_pos = base
    if IS_NVA3F(chipset):
        ctx.cp_ctx(0x404800, num)
    else:
        ctx.cp_ctx(0x405400, num)


# ---------------------------------------------------------------------------
# nv50_gr_construct_xfer1 (lines 1188-1345)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer1(ctx):
    chipset = ctx.chipset
    units = ctx.units
    size = 0

    offset = (ctx.ctxvals_pos + 0x3f) & ~0x3f
    ctx.ctxvals_base = offset

    if chipset < 0xa0:
        # Strand 0
        ctx.ctxvals_pos = offset
        nv50_gr_construct_gene_dispatch(ctx)
        nv50_gr_construct_gene_m2mf(ctx)
        nv50_gr_construct_gene_unk24xx(ctx)
        nv50_gr_construct_gene_clipid(ctx)
        nv50_gr_construct_gene_zcull(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 1
        ctx.ctxvals_pos = offset + 0x1
        nv50_gr_construct_gene_vfetch(ctx)
        nv50_gr_construct_gene_eng2d(ctx)
        nv50_gr_construct_gene_csched(ctx)
        nv50_gr_construct_gene_ropm1(ctx)
        nv50_gr_construct_gene_ropm2(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 2
        ctx.ctxvals_pos = offset + 0x2
        nv50_gr_construct_gene_ccache(ctx)
        nv50_gr_construct_gene_unk1cxx(ctx)
        nv50_gr_construct_gene_strmout(ctx)
        nv50_gr_construct_gene_unk14xx(ctx)
        nv50_gr_construct_gene_unk10xx(ctx)
        nv50_gr_construct_gene_unk34xx(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 3: per-ROP group state
        ctx.ctxvals_pos = offset + 3
        for i in range(6):
            if units & (1 << (i + 16)):
                nv50_gr_construct_gene_ropc(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strands 4-7: per-TP state
        for i in range(4):
            ctx.ctxvals_pos = offset + 4 + i
            if units & (1 << (2 * i)):
                nv50_gr_construct_xfer_tp(ctx)
            if units & (1 << (2 * i + 1)):
                nv50_gr_construct_xfer_tp(ctx)
            if (ctx.ctxvals_pos - offset) // 8 > size:
                size = (ctx.ctxvals_pos - offset) // 8
    else:
        # Strand 0
        ctx.ctxvals_pos = offset
        nv50_gr_construct_gene_dispatch(ctx)
        nv50_gr_construct_gene_m2mf(ctx)
        nv50_gr_construct_gene_unk34xx(ctx)
        nv50_gr_construct_gene_csched(ctx)
        nv50_gr_construct_gene_unk1cxx(ctx)
        nv50_gr_construct_gene_strmout(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 1
        ctx.ctxvals_pos = offset + 1
        nv50_gr_construct_gene_unk10xx(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 2
        ctx.ctxvals_pos = offset + 2
        if chipset == 0xa0:
            nv50_gr_construct_gene_unk14xx(ctx)
        nv50_gr_construct_gene_unk24xx(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 3
        ctx.ctxvals_pos = offset + 3
        nv50_gr_construct_gene_vfetch(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 4
        ctx.ctxvals_pos = offset + 4
        nv50_gr_construct_gene_ccache(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 5
        ctx.ctxvals_pos = offset + 5
        nv50_gr_construct_gene_ropm2(ctx)
        nv50_gr_construct_gene_ropm1(ctx)
        # per-ROP context
        for i in range(8):
            if units & (1 << (i + 16)):
                nv50_gr_construct_gene_ropc(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 6
        ctx.ctxvals_pos = offset + 6
        nv50_gr_construct_gene_zcull(ctx)
        nv50_gr_construct_gene_clipid(ctx)
        nv50_gr_construct_gene_eng2d(ctx)
        if units & (1 << 0):
            nv50_gr_construct_xfer_tp(ctx)
        if units & (1 << 1):
            nv50_gr_construct_xfer_tp(ctx)
        if units & (1 << 2):
            nv50_gr_construct_xfer_tp(ctx)
        if units & (1 << 3):
            nv50_gr_construct_xfer_tp(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 7
        ctx.ctxvals_pos = offset + 7
        if chipset == 0xa0:
            if units & (1 << 4):
                nv50_gr_construct_xfer_tp(ctx)
            if units & (1 << 5):
                nv50_gr_construct_xfer_tp(ctx)
            if units & (1 << 6):
                nv50_gr_construct_xfer_tp(ctx)
            if units & (1 << 7):
                nv50_gr_construct_xfer_tp(ctx)
            if units & (1 << 8):
                nv50_gr_construct_xfer_tp(ctx)
            if units & (1 << 9):
                nv50_gr_construct_xfer_tp(ctx)
        else:
            nv50_gr_construct_gene_unk14xx(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

    ctx.ctxvals_pos = offset + size * 8
    ctx.ctxvals_pos = (ctx.ctxvals_pos + 0x3f) & ~0x3f
    ctx.cp_lsr(offset)
    ctx.cp_out(CP_SET_XFER_POINTER)
    ctx.cp_lsr(size)
    ctx.cp_out(CP_SEEK_1)
    ctx.cp_out(CP_XFER_1)
    ctx.cp_wait(CP_FLAG_XFER, CP_FLAG_XFER_BUSY)


# ---------------------------------------------------------------------------
# Gene: dispatch (lines 1351-1405)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_dispatch(ctx):
    chipset = ctx.chipset
    if chipset == 0x50:
        ctx.xf_emit(5, 0)
    elif not IS_NVA3F(chipset):
        ctx.xf_emit(6, 0)
    else:
        ctx.xf_emit(4, 0)
    if chipset == 0x50:
        ctx.xf_emit(8 * 3, 0)
    else:
        ctx.xf_emit(0x100 * 3, 0)
    ctx.xf_emit(3, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(3, 0)
    ctx.xf_emit(9, 0)
    ctx.xf_emit(9, 0)
    ctx.xf_emit(9, 0)
    ctx.xf_emit(9, 0)
    if chipset < 0x90:
        ctx.xf_emit(4, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(6 * 2, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(6 * 2, 0)
    ctx.xf_emit(2, 0)
    if chipset == 0x50:
        ctx.xf_emit(0x1c, 0)
    elif chipset < 0xa0:
        ctx.xf_emit(0x1e, 0)
    else:
        ctx.xf_emit(0x22, 0)
    ctx.xf_emit(0x15, 0)


# ---------------------------------------------------------------------------
# Gene: m2mf (lines 1407-1457)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_m2mf(ctx):
    chipset = ctx.chipset
    smallm2mf = 0
    if chipset < 0x92 or chipset == 0x98:
        smallm2mf = 1
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x21)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x2)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x2)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if smallm2mf:
        ctx.xf_emit(0x40, 0)
    else:
        ctx.xf_emit(0x100, 0)
    ctx.xf_emit(4, 0)
    if smallm2mf:
        ctx.xf_emit(0x400, 0)
    else:
        ctx.xf_emit(0x800, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(0x40, 0)
    ctx.xf_emit(0x6, 0)


# ---------------------------------------------------------------------------
# Gene: ccache (lines 1459-1525)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_ccache(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(2, 0)
    ctx.xf_emit(0x800, 0)
    if chipset in (0x50, 0x92, 0xa0):
        ctx.xf_emit(0x2b, 0)
    elif chipset == 0x84:
        ctx.xf_emit(0x29, 0)
    elif chipset in (0x94, 0x96, 0xa3):
        ctx.xf_emit(0x27, 0)
    elif chipset in (0x86, 0x98, 0xa5, 0xa8, 0xaa, 0xac, 0xaf):
        ctx.xf_emit(0x25, 0)
    ctx.xf_emit(0x100, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x30, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(0x100, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x3fffff)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x1fff)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: unk10xx (lines 1527-1585)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_unk10xx(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x80)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0x80c14)
    ctx.xf_emit(1, 0)
    if chipset == 0x50:
        ctx.xf_emit(1, 0x3ff)
    else:
        ctx.xf_emit(1, 0x7ff)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    for _ in range(8):
        if chipset in (0x50, 0x86, 0x98, 0xaa, 0xac):
            ctx.xf_emit(0xa0, 0)
        elif chipset in (0x84, 0x92, 0x94, 0x96):
            ctx.xf_emit(0x120, 0)
        elif chipset in (0xa5, 0xa8):
            ctx.xf_emit(0x100, 0)
        elif chipset in (0xa0, 0xa3, 0xaf):
            ctx.xf_emit(0x400, 0)
        ctx.xf_emit(4, 0)
        ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x80)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x27)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x26)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: unk34xx (lines 1587-1610)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_unk34xx(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x10, 0x04000000)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x20, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x04e3bfdf)
    ctx.xf_emit(1, 0x04e3bfdf)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x1fe21)
    if chipset >= 0xa0:
        ctx.xf_emit(1, 0x0fac6881)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
        ctx.xf_emit(3, 0)


# ---------------------------------------------------------------------------
# Gene: unk14xx (lines 1612-1721)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_unk14xx(ctx):
    chipset = ctx.chipset
    if chipset != 0x50:
        ctx.xf_emit(5, 0)
        ctx.xf_emit(1, 0x80c14)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0x804)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(2, 4)
        ctx.xf_emit(1, 0x8100c12)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(1, 0)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x804)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x1a)
    if chipset != 0x50:
        ctx.xf_emit(1, 0x7f)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x80c14)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x8100c12)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x8100c12)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if chipset == 0x50:
        ctx.xf_emit(1, 0x3ff)
    else:
        ctx.xf_emit(1, 0x7ff)
    ctx.xf_emit(1, 0x80c14)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x30, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x30, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0x88)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x26)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x3f800000)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x1a)
    ctx.xf_emit(1, 0x10)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(0x20, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x52)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x26)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x1a)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x00ffff00)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: zcull (lines 1723-1782)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_zcull(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(1, 0x3f)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(2, 0x04000000)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x1001)
    ctx.xf_emit(4, 0xffff)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(0x10, 0x3f800000)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 3)
    ctx.xf_emit(1, 0)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: clipid (lines 1784-1802)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_clipid(ctx):
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(2, 0x04000000)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x80)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x80)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: unk24xx (lines 1804-1885)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_unk24xx(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(0x33, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    if IS_NVA3F(chipset):
        ctx.xf_emit(4, 0)
        ctx.xf_emit(0xe10, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 0)
        ctx.xf_emit(9, 0)
        ctx.xf_emit(4, 0)
        ctx.xf_emit(0xe10, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 0)
        ctx.xf_emit(9, 0)
    else:
        ctx.xf_emit(0xc, 0)
        ctx.xf_emit(0xe10, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 0)
        ctx.xf_emit(0xc, 0)
        ctx.xf_emit(0xe10, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0x8100c12)
    if chipset != 0x50:
        ctx.xf_emit(1, 3)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x8100c12)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x80c14)
    ctx.xf_emit(1, 1)
    if chipset >= 0xa0:
        ctx.xf_emit(2, 4)
    ctx.xf_emit(1, 0x80c14)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x8100c12)
    ctx.xf_emit(1, 0x27)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    for _ in range(10):
        ctx.xf_emit(0x40, 0)
        ctx.xf_emit(0x10, 0)
        ctx.xf_emit(0x10, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x8100c12)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: vfetch (lines 1887-2071)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_vfetch(ctx):
    chipset = ctx.chipset
    acnt = 0x10
    if IS_NVA3F(chipset):
        acnt = 0x20
    if chipset >= 0xa0:
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit((acnt // 8) - 1, 0)
    ctx.xf_emit(acnt // 8, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x20)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(0xb, 0)
    elif chipset >= 0xa0:
        ctx.xf_emit(0x9, 0)
    else:
        ctx.xf_emit(0x8, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x1a)
    ctx.xf_emit(0xc, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 8)
    ctx.xf_emit(1, 0)
    if chipset == 0x50:
        ctx.xf_emit(1, 0x3ff)
    else:
        ctx.xf_emit(1, 0x7ff)
    if chipset == 0xa8:
        ctx.xf_emit(1, 0x1e00)
    ctx.xf_emit(0xc, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit((acnt // 8) - 1, 0)
    ctx.xf_emit(1, 0)
    if chipset > 0x50 and chipset < 0xa0:
        ctx.xf_emit(2, 0)
    else:
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(0x10, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(2, 0)
    else:
        ctx.xf_emit(8, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(acnt, 0)
    if chipset >= 0xa0:
        ctx.xf_emit(1, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(acnt, 0)
    ctx.xf_emit(3, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(acnt, 0)
        ctx.xf_emit(3, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(2, 0)
    else:
        ctx.xf_emit(5, 0)
    ctx.xf_emit(1, 0)
    if chipset < 0xa0:
        ctx.xf_emit(0x41, 0)
        ctx.xf_emit(0x11, 0)
    elif not IS_NVA3F(chipset):
        ctx.xf_emit(0x50, 0)
    else:
        ctx.xf_emit(0x58, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit((acnt // 8) - 1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(acnt * 4, 0)
    ctx.xf_emit(4, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(0x1d, 0)
    else:
        ctx.xf_emit(0x16, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit((acnt // 8) - 1, 0)
    if chipset < 0xa0:
        ctx.xf_emit(8, 0)
    elif IS_NVA3F(chipset):
        ctx.xf_emit(0xc, 0)
    else:
        ctx.xf_emit(7, 0)
    ctx.xf_emit(0xa, 0)
    if chipset == 0xa0:
        rep = 0xc
    else:
        rep = 4
    for _ in range(rep):
        if IS_NVA3F(chipset):
            ctx.xf_emit(0x20, 0)
        ctx.xf_emit(0x200, 0)
        ctx.xf_emit(4, 0)
        ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit((acnt // 8) - 1, 0)
    ctx.xf_emit(acnt // 8, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(7, 0)
    else:
        ctx.xf_emit(5, 0)


# ---------------------------------------------------------------------------
# Gene: eng2d (lines 2073-2133)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_eng2d(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(2, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    if chipset < 0xa0:
        ctx.xf_emit(2, 0)
        ctx.xf_emit(2, 1)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 8)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0xcf)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x15)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x4444480)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(0x27, 0)


# ---------------------------------------------------------------------------
# Gene: csched (lines 2135-2232)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_csched(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x8100c12)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x10001)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x10001)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x10001)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(0x40, 0)
    if chipset in (0x50, 0x92):
        ctx.xf_emit(8, 0)
        ctx.xf_emit(0x80, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(0x10 * 2, 0)
    elif chipset == 0x84:
        ctx.xf_emit(8, 0)
        ctx.xf_emit(0x60, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(0xc * 2, 0)
    elif chipset in (0x94, 0x96):
        ctx.xf_emit(8, 0)
        ctx.xf_emit(0x40, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(8 * 2, 0)
    elif chipset in (0x86, 0x98):
        ctx.xf_emit(4, 0)
        ctx.xf_emit(0x10, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(2 * 2, 0)
    elif chipset == 0xa0:
        ctx.xf_emit(8, 0)
        ctx.xf_emit(0xf0, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(0x1e * 2, 0)
    elif chipset == 0xa3:
        ctx.xf_emit(8, 0)
        ctx.xf_emit(0x60, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(0xc * 2, 0)
    elif chipset in (0xa5, 0xaf):
        ctx.xf_emit(8, 0)
        ctx.xf_emit(0x30, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(6 * 2, 0)
    elif chipset == 0xaa:
        ctx.xf_emit(0x12, 0)
    elif chipset in (0xa8, 0xac):
        ctx.xf_emit(4, 0)
        ctx.xf_emit(0x10, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(2 * 2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: unk1cxx (lines 2234-2328)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_unk1cxx(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0x3f800000)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0x1a)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x00ffff00)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 3)
    elif chipset >= 0xa0:
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(2, 0x04000000)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 5)
    ctx.xf_emit(1, 0x52)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 1)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(0x10, 0x3f800000)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(0x20, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x8100c12)
    ctx.xf_emit(1, 5)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0xffff)
    if chipset != 0x50:
        ctx.xf_emit(1, 3)
    if chipset < 0xa0:
        ctx.xf_emit(0x1c, 0)
    elif IS_NVA3F(chipset):
        ctx.xf_emit(0x9, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x00ffff00)
    ctx.xf_emit(1, 0x1a)
    ctx.xf_emit(1, 0)
    if chipset != 0x50:
        ctx.xf_emit(1, 3)
        ctx.xf_emit(1, 0)
    if chipset < 0xa0:
        ctx.xf_emit(0x25, 0)
    else:
        ctx.xf_emit(0x3b, 0)


# ---------------------------------------------------------------------------
# Gene: strmout (lines 2330-2370)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_strmout(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(1, 0x102)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 4)
    if chipset >= 0xa0:
        ctx.xf_emit(4, 0)
        ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    if chipset == 0x50:
        ctx.xf_emit(1, 0x3ff)
    else:
        ctx.xf_emit(1, 0x7ff)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x102)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(4, 4)
    if chipset >= 0xa0:
        ctx.xf_emit(4, 0)
        ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x20, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)


# ---------------------------------------------------------------------------
# Gene: ropm1 (lines 2372-2383)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_ropm1(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(1, 0x4e3bfdf)
    ctx.xf_emit(1, 0x4e3bfdf)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: ropm2 (lines 2385-2409)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_ropm2(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0x4e3bfdf)
    ctx.xf_emit(1, 0x4e3bfdf)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# Gene: ropc (lines 2411-2644)
# ---------------------------------------------------------------------------

def nv50_gr_construct_gene_ropc(ctx):
    chipset = ctx.chipset
    if chipset == 0x50:
        magic2 = 0x00003e60
    elif not IS_NVA3F(chipset):
        magic2 = 0x001ffe67
    else:
        magic2 = 0x00087e67
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, magic2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 0)
    if chipset >= 0xa0 and not IS_NVAAF(chipset):
        ctx.xf_emit(1, 0x15)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    if chipset == 0x86 or chipset == 0x92 or chipset == 0x98 or chipset >= 0xa0:
        ctx.xf_emit(3, 0)
        ctx.xf_emit(1, 4)
        ctx.xf_emit(1, 0x400)
        ctx.xf_emit(1, 0x300)
        ctx.xf_emit(1, 0x1001)
        if chipset != 0xa0:
            if IS_NVA3F(chipset):
                ctx.xf_emit(1, 0)
            else:
                ctx.xf_emit(1, 0x15)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x10)
    ctx.xf_emit(0x10, 0)
    ctx.xf_emit(0x10, 0x3f800000)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x3f)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    if chipset >= 0xa0:
        ctx.xf_emit(2, 0)
        ctx.xf_emit(1, 0x1001)
        ctx.xf_emit(0xb, 0)
    else:
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 0)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 0)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, magic2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x0fac6881)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(1, 1)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(1, 1)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    elif chipset >= 0xa0:
        ctx.xf_emit(2, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(2, 0)
    else:
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    if chipset >= 0xa0:
        ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if chipset >= 0xa0:
        ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x50, 0)


# ---------------------------------------------------------------------------
# xfer_unk84xx (lines 2646-2736)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer_unk84xx(ctx):
    chipset = ctx.chipset
    if chipset == 0x50:
        magic3 = 0x1000
    elif chipset in (0x86, 0x98, 0xa8, 0xaa, 0xac, 0xaf):
        magic3 = 0x1e00
    else:
        magic3 = 0
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(0x1f, 0)
    elif chipset >= 0xa0:
        ctx.xf_emit(0x0f, 0)
    else:
        ctx.xf_emit(0x10, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    if chipset >= 0xa0:
        ctx.xf_emit(1, 0x03020100)
    else:
        ctx.xf_emit(1, 0x00608080)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0x80)
    if magic3:
        ctx.xf_emit(1, magic3)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(0x1f, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0x80)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0x03020100)
    ctx.xf_emit(1, 3)
    if magic3:
        ctx.xf_emit(1, magic3)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 3)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 3)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if chipset == 0x94 or chipset == 0x96:
        ctx.xf_emit(0x1020, 0)
    elif chipset < 0xa0:
        ctx.xf_emit(0xa20, 0)
    elif not IS_NVA3F(chipset):
        ctx.xf_emit(0x210, 0)
    else:
        ctx.xf_emit(0x410, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 3)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# xfer_tprop (lines 2738-3036)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer_tprop(ctx):
    chipset = ctx.chipset
    if chipset == 0x50:
        magic1 = 0x3ff
        magic2 = 0x00003e60
    elif not IS_NVA3F(chipset):
        magic1 = 0x7ff
        magic2 = 0x001ffe67
    else:
        magic1 = 0x7ff
        magic2 = 0x00087e67
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(4, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(4, 0xffff)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 3)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    elif chipset >= 0xa0:
        ctx.xf_emit(1, 1)
        ctx.xf_emit(1, 0)
    else:
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0xcf)
    ctx.xf_emit(1, 0xcf)
    ctx.xf_emit(1, 0xcf)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(8, 1)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, magic2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 1)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    if chipset == 0x50:
        ctx.xf_emit(1, 0)
    else:
        ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0xff)
    ctx.xf_emit(1, magic1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(8, 8)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(8, 0x400)
    ctx.xf_emit(8, 0x300)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x20)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x40)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 3)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, magic2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x400)
    ctx.xf_emit(1, 0x300)
    ctx.xf_emit(1, 0x1001)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(1, 0xf)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if chipset >= 0xa0:
        ctx.xf_emit(1, 0x0fac6881)
    ctx.xf_emit(1, magic2)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    if chipset >= 0xa0:
        ctx.xf_emit(3, 0)
        ctx.xf_emit(1, 0xfac6881)
        ctx.xf_emit(4, 0)
        ctx.xf_emit(1, 4)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(2, 1)
        ctx.xf_emit(2, 0)
        ctx.xf_emit(1, 1)
        ctx.xf_emit(1, 0)
        if IS_NVA3F(chipset):
            ctx.xf_emit(0x9, 0)
        else:
            ctx.xf_emit(0x8, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(1, 0x11)
        ctx.xf_emit(7, 0)
        ctx.xf_emit(1, 0xfac6881)
        ctx.xf_emit(1, 0xf)
        ctx.xf_emit(7, 0)
        ctx.xf_emit(1, 0x11)
        ctx.xf_emit(1, 1)
        ctx.xf_emit(5, 0)
        if IS_NVA3F(chipset):
            ctx.xf_emit(1, 0)
            ctx.xf_emit(1, 1)


# ---------------------------------------------------------------------------
# xfer_tex (lines 3038-3082)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer_tex(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(2, 0)
    if chipset != 0x50:
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    if chipset == 0x50:
        ctx.xf_emit(1, 0)
    else:
        ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0x2a712488)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x4085c000)
    ctx.xf_emit(1, 0x40)
    ctx.xf_emit(1, 0x100)
    ctx.xf_emit(1, 0x10100)
    ctx.xf_emit(1, 0x02800000)
    ctx.xf_emit(1, 0)
    if chipset == 0x50:
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    elif not IS_NVAAF(chipset):
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    else:
        ctx.xf_emit(0x6, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)


# ---------------------------------------------------------------------------
# xfer_unk8cxx (lines 3084-3121)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer_unk8cxx(ctx):
    chipset = ctx.chipset
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(2, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x04e3bfdf)
    ctx.xf_emit(1, 0x04e3bfdf)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x00ffff00)
    ctx.xf_emit(1, 1)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x00ffff00)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x30201000)
    ctx.xf_emit(1, 0x70605040)
    ctx.xf_emit(1, 0xb8a89888)
    ctx.xf_emit(1, 0xf8e8d8c8)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x1a)


# ---------------------------------------------------------------------------
# xfer_tp (lines 3123-3138)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer_tp(ctx):
    chipset = ctx.chipset
    if chipset < 0xa0:
        nv50_gr_construct_xfer_unk84xx(ctx)
        nv50_gr_construct_xfer_tprop(ctx)
        nv50_gr_construct_xfer_tex(ctx)
        nv50_gr_construct_xfer_unk8cxx(ctx)
    else:
        nv50_gr_construct_xfer_tex(ctx)
        nv50_gr_construct_xfer_tprop(ctx)
        nv50_gr_construct_xfer_unk8cxx(ctx)
        nv50_gr_construct_xfer_unk84xx(ctx)


# ---------------------------------------------------------------------------
# xfer_mpc (lines 3140-3270)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer_mpc(ctx):
    chipset = ctx.chipset
    mpcnt = 2
    if chipset in (0x98, 0xaa):
        mpcnt = 1
    elif chipset in (0x50, 0x84, 0x86, 0x92, 0x94, 0x96, 0xa8, 0xac):
        mpcnt = 2
    elif chipset in (0xa0, 0xa3, 0xa5, 0xaf):
        mpcnt = 3
    for _ in range(mpcnt):
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0x80)
        ctx.xf_emit(1, 0x80007004)
        ctx.xf_emit(1, 0x04000400)
        if chipset >= 0xa0:
            ctx.xf_emit(1, 0xc0)
        ctx.xf_emit(1, 0x1000)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        if chipset == 0x86 or chipset == 0x98 or chipset == 0xa8 or IS_NVAAF(chipset):
            ctx.xf_emit(1, 0xe00)
            ctx.xf_emit(1, 0x1e00)
        ctx.xf_emit(1, 1)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
        if chipset == 0x50:
            ctx.xf_emit(2, 0x1000)
        ctx.xf_emit(1, 1)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 4)
        ctx.xf_emit(1, 2)
        if IS_NVAAF(chipset):
            ctx.xf_emit(0xb, 0)
        elif chipset >= 0xa0:
            ctx.xf_emit(0xc, 0)
        else:
            ctx.xf_emit(0xa, 0)
    ctx.xf_emit(1, 0x08100c12)
    ctx.xf_emit(1, 0)
    if chipset >= 0xa0:
        ctx.xf_emit(1, 0x1fe21)
    ctx.xf_emit(3, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(4, 0xffff)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0x10001)
    ctx.xf_emit(1, 0x10001)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x1fe21)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0x08100c12)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(7, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0xfac6881)
    ctx.xf_emit(1, 0)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 3)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    ctx.xf_emit(8, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 2)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(1, 1)
    if IS_NVA3F(chipset):
        ctx.xf_emit(1, 0)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 2)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(8, 1)
        ctx.xf_emit(1, 0)
        ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 4)
    if chipset == 0x50:
        ctx.xf_emit(0x3a0, 0)
    elif chipset < 0x94:
        ctx.xf_emit(0x3a2, 0)
    elif chipset == 0x98 or chipset == 0xaa:
        ctx.xf_emit(0x39f, 0)
    else:
        ctx.xf_emit(0x3a3, 0)
    ctx.xf_emit(1, 0x11)
    ctx.xf_emit(1, 0)
    ctx.xf_emit(1, 1)
    ctx.xf_emit(0x2d, 0)


# ---------------------------------------------------------------------------
# nv50_gr_construct_xfer2 (lines 3272-3347)
# ---------------------------------------------------------------------------

def nv50_gr_construct_xfer2(ctx):
    chipset = ctx.chipset
    units = ctx.units
    size = 0

    offset = (ctx.ctxvals_pos + 0x3f) & ~0x3f

    if chipset < 0xa0:
        for i in range(8):
            ctx.ctxvals_pos = offset + i
            if i == 0:
                ctx.xf_emit(1, 0x08100c12)
            if units & (1 << i):
                nv50_gr_construct_xfer_mpc(ctx)
            if (ctx.ctxvals_pos - offset) // 8 > size:
                size = (ctx.ctxvals_pos - offset) // 8
    else:
        # Strand 0: TPs 0, 1
        ctx.ctxvals_pos = offset
        ctx.xf_emit(1, 0x08100c12)
        if units & (1 << 0):
            nv50_gr_construct_xfer_mpc(ctx)
        if units & (1 << 1):
            nv50_gr_construct_xfer_mpc(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 1: TPs 2, 3
        ctx.ctxvals_pos = offset + 1
        if units & (1 << 2):
            nv50_gr_construct_xfer_mpc(ctx)
        if units & (1 << 3):
            nv50_gr_construct_xfer_mpc(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 2: TPs 4, 5, 6
        ctx.ctxvals_pos = offset + 2
        if units & (1 << 4):
            nv50_gr_construct_xfer_mpc(ctx)
        if units & (1 << 5):
            nv50_gr_construct_xfer_mpc(ctx)
        if units & (1 << 6):
            nv50_gr_construct_xfer_mpc(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

        # Strand 3: TPs 7, 8, 9
        ctx.ctxvals_pos = offset + 3
        if units & (1 << 7):
            nv50_gr_construct_xfer_mpc(ctx)
        if units & (1 << 8):
            nv50_gr_construct_xfer_mpc(ctx)
        if units & (1 << 9):
            nv50_gr_construct_xfer_mpc(ctx)
        if (ctx.ctxvals_pos - offset) // 8 > size:
            size = (ctx.ctxvals_pos - offset) // 8

    ctx.ctxvals_pos = offset + size * 8
    ctx.ctxvals_pos = (ctx.ctxvals_pos + 0x3f) & ~0x3f
    ctx.cp_lsr(offset)
    ctx.cp_out(CP_SET_XFER_POINTER)
    ctx.cp_lsr(size)
    ctx.cp_out(CP_SEEK_2)
    ctx.cp_out(CP_XFER_2)
    ctx.cp_wait(CP_FLAG_XFER, CP_FLAG_XFER_BUSY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ctxprog(chipset=0xa8, units=0, ram_type=0):
    """Generate the ctxprog microcode for the given chipset.

    Args:
        chipset: GPU chipset ID (default 0xa8 for GT218)
        units: PGRAPH units bitmask from register 0x1540 (default 0)
        ram_type: RAM type (default 0 = non-GDDR5)

    Returns:
        (ctxprog: list[int], ctxvals_size: int)
        ctxprog is a list of 32-bit instruction words.
        ctxvals_size is the required ctxvals buffer size in bytes.
    """
    ctx = Grctx(chipset, PROG, units=units, ram_type=ram_type)
    nv50_grctx_generate(ctx)
    return ctx.ucode, ctx.ctxvals_pos * 4


def generate_ctxvals(chipset=0xa8, units=0, ram_type=0):
    """Generate the ctxvals buffer for the given chipset.

    Args:
        chipset: GPU chipset ID (default 0xa8 for GT218)
        units: PGRAPH units bitmask from register 0x1540 (default 0)
        ram_type: RAM type (default 0 = non-GDDR5)

    Returns:
        (ctxvals: bytes, ctxvals_size: int)
        ctxvals is the default register values buffer.
        ctxvals_size is the size in bytes.
    """
    ctx = Grctx(chipset, VALS, units=units, ram_type=ram_type)
    nv50_grctx_generate(ctx)
    return bytes(ctx.data), ctx.ctxvals_pos * 4


if __name__ == '__main__':
    # Quick test: generate ctxprog and ctxvals for GT218
    prog, vals_size = generate_ctxprog(0xa8)
    print(f"ctxprog: {len(prog)} instructions, ctxvals_size: {vals_size} bytes")
    for i, instr in enumerate(prog):
        print(f"  [{i:3d}] 0x{instr:08x}")

    vals, vals_size2 = generate_ctxvals(0xa8)
    print(f"ctxvals: {len(vals)} bytes, ctxvals_size: {vals_size2} bytes")
