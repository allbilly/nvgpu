#!/usr/bin/env python3
"""Compute kernel launch for ASUS EN210 (GT218, GeForce 210, sm_12).

Runs a vector-add kernel: out[i] = a[i] + b[i].

Reuses fifo_test.py for VBIOS devinit, FIFO/GR init, and channel creation.
Adds: GR context bind, RAMHT entries, NV50_COMPUTE method sequence, launch.

Transport: TinyGPU.app DriverKit Unix socket (/tmp/tinygpu.sock) on macOS.
"""
from __future__ import annotations
import os, sys, struct, time

sys.path.insert(0, os.path.dirname(__file__))
from fifo_test import (
    Dev, VRAM, nvm50, gp_entry, make_dma_object,
    fifo_init, gr_init, gr_bind_context, setup_gmmu,
    create_channel,
    CHAN_ID, USERD_BASE, USERD_SIZE, VRAM_BASE, PGT_ADDR,
    INST_ENG_OFF, INST_PGD_OFF, INST_CACHE_OFF, INST_RAMFC_OFF,
    INST_RAMHT_OFF, INST_PUSH_OFF,
)
from nvbios_init import find_vbios_scripts, run_vbios_init

# ============================================================================
# Constants
# ============================================================================
# GT218 (0xa8) uses NVA3_COMPUTE_CLASS (0x85c0), not NV50_COMPUTE (0x50c0).
# Source: mesa/src/gallium/drivers/nouveau/nv50/nv50_compute.c:54
NVA3_COMPUTE_CLASS = 0x85c0
COMPUTE_HANDLE     = 0x85c0
VRAM_DMA_HANDLE    = 0xbeef0201

# Engine IDs from g98_fifo_runl_ctor (g98.c:37-44):
#   SW/DMAOBJ: engi=0, GR: engi=1, MSPPP: engi=2, CE: engi=3, ...
GR_ENGN_ID    = 1
DMA_ENGN_ID   = 0

# Compute subchannel (mesa nv50_winsys.h:53: SUBC_CP = 6)
SUBC_CP = 6

# NV50_COMPUTE method addresses (nv50_compute.xml.h + cl50c0.h)
CP_SET_OBJECT             = 0x0000
CP_WAIT_FOR_IDLE          = 0x0110
CP_DMA_GLOBAL             = 0x01a0
CP_DMA_LOCAL              = 0x01b8
CP_DMA_STACK              = 0x01bc
CP_DMA_CODE_CB            = 0x01c0
CP_STACK_ADDRESS_HIGH     = 0x0218
CP_STACK_SIZE_LOG         = 0x0220
CP_SHADER_SCHEDULING      = 0x0290
CP_LOCAL_ADDRESS_HIGH     = 0x0294
CP_LOCAL_SIZE_LOG         = 0x029c
CP_WORK_DISTRIBUTION      = 0x02a0
CP_BLOCK_ALLOC            = 0x02b4
CP_LANES32_ENABLE         = 0x02b8
CP_CP_REG_ALLOC_TEMP      = 0x02c0
CP_BLOCKDIM_LATCH         = 0x02f8
CP_LOCAL_WARPS_LOG_ALLOC  = 0x02fc
CP_LOCAL_WARPS_NO_CLAMP   = 0x0300
CP_STACK_WARPS_LOG_ALLOC  = 0x0304
CP_STACK_WARPS_NO_CLAMP   = 0x0308
CP_CODE_CB_FLUSH          = 0x0380
CP_UNK0384                = 0x0384
CP_GRIDID                 = 0x0388
CP_LAUNCH                 = 0x0368
CP_GRIDDIM                = 0x03a4
CP_SHARED_SIZE            = 0x03a8
CP_BLOCKDIM_XY            = 0x03ac
CP_BLOCKDIM_Z             = 0x03b0
CP_CP_START_ID            = 0x03b4
CP_REG_MODE               = 0x03b8
CP_USER_PARAM_COUNT       = 0x0374

def CP_GLOBAL_ADDR_HI(i):  return 0x0400 + 0x20 * i
def CP_GLOBAL_LIMIT(i):    return 0x040c + 0x20 * i
def CP_GLOBAL_MODE(i):     return 0x0410 + 0x20 * i
def CP_USER_PARAM(i):      return 0x0600 + 0x04 * i

GLOBAL_MODE_LINEAR = 0x00000001
REG_MODE_STRIPED   = 0x00000002

# VRAM layout: PGT is at 0x130000 (1MB), so compute buffers start at 0x240000
KERNEL_ADDR   = VRAM_BASE + 0x140000  # 0x240000: kernel SASS code
INPUT_A_ADDR  = VRAM_BASE + 0x150000  # 0x250000: input array a
INPUT_B_ADDR  = VRAM_BASE + 0x160000  # 0x260000: input array b
OUTPUT_ADDR   = VRAM_BASE + 0x170000  # 0x270000: output array
STACK_ADDR    = VRAM_BASE + 0x180000  # 0x280000: stack/TLS
CTXVALS_ADDR  = VRAM_BASE + 0x190000  # 0x290000: GR context (ctxvals)

# VRAM DMA object placed in instance block after RAMHT (0x5500 + 0x8000 = 0xD500)
INST_VRAM_DMA_OFF = 0xD500

# ============================================================================
# RAMHT hash (ramht.c:22-30, chid=0 for per-channel RAMHT)
# ============================================================================
def ramht_hash(handle, bits=12):
    h = 0
    while handle:
        h ^= handle & ((1 << bits) - 1)
        handle >>= bits
    return h

def ramht_insert(vram, inst_addr, handle, context):
    idx = ramht_hash(handle)
    off = INST_RAMHT_OFF + idx * 8
    vram.write32(inst_addr + off, handle)
    vram.write32(inst_addr + off + 4, context)
    print(f'[ramht] handle=0x{handle:08x} idx=0x{idx:03x} ctx=0x{context:08x}')

# ============================================================================
# Extend GMMU mapping for compute buffers
# ============================================================================
def extend_gmmu(vram, inst_addr, vaddr_start, vaddr_end):
    """Add identity-mapped PTEs for vaddr_start to vaddr_end (4KB pages).
    Also zeros PTEs in the range first (setup_gmmu only zeros first 512 entries).
    """
    for vaddr in range(vaddr_start, vaddr_end, 0x1000):
        pgt_idx = (vaddr >> 12) & 0x1FFFF
        pte = vaddr | 0x1  # identity mapping: virtual = physical
        vram.write32(PGT_ADDR + pgt_idx * 8, pte & 0xFFFFFFFF)
        vram.write32(PGT_ADDR + pgt_idx * 8 + 4, (pte >> 32) & 0xFF)
    # Flush GMMU TLB
    for eng_id in [0x00, 0x01, 0x06, 0x08, 0x09, 0x0a, 0x0d]:
        vram.dev.write32(0x100c80, (eng_id << 16) | 1)
        for _ in range(2000):
            if not (vram.dev.read32(0x100c80) & 0x00000001):
                break
            time.sleep(0.001)
    print(f'[gmmu] Extended mapping: 0x{vaddr_start:06x}-0x{vaddr_end:06x}')

# ============================================================================
# Compute setup + launch push buffer
# ============================================================================
def build_compute_push(vram_dma_handle, kernel_addr, stack_addr,
                       input_a, input_b, output, n_elements,
                       block_dim=1, grid_dim=1):
    """Build push buffer for NV50_COMPUTE setup + launch.

    Follows mesa/src/gallium/drivers/nouveau/nv50/nv50_compute.c:
    - nv50_screen_compute_setup (lines 66-166): one-time init
    - nv50_launch_grid_with_input (lines 562-633): per-launch
    """
    w = []
    # SET_OBJECT on subchannel 6
    w += nvm50(SUBC_CP, CP_SET_OBJECT, COMPUTE_HANDLE)

    # One-time compute setup (mesa nv50_screen_compute_setup)
    w += nvm50(SUBC_CP, CP_WORK_DISTRIBUTION, 1)
    w += nvm50(SUBC_CP, CP_DMA_STACK, vram_dma_handle)
    w += nvm50(SUBC_CP, CP_STACK_ADDRESS_HIGH,
               (stack_addr >> 32) & 0xFF, stack_addr & 0xFFFFFFFF)
    w += nvm50(SUBC_CP, CP_STACK_SIZE_LOG, 4)
    w += nvm50(SUBC_CP, CP_SHADER_SCHEDULING, 1)
    w += nvm50(SUBC_CP, CP_LANES32_ENABLE, 1)
    w += nvm50(SUBC_CP, CP_REG_MODE, REG_MODE_STRIPED)
    w += nvm50(SUBC_CP, CP_UNK0384, 0x100)
    w += nvm50(SUBC_CP, CP_DMA_GLOBAL, vram_dma_handle)

    # 16 global memory slots: all full-range (kernel uses global14 for GLD/GST)
    for i in range(16):
        w += nvm50(SUBC_CP, CP_GLOBAL_ADDR_HI(i), 0, 0)
        w += nvm50(SUBC_CP, CP_GLOBAL_LIMIT(i), 0xFFFFFFFF)
        w += nvm50(SUBC_CP, CP_GLOBAL_MODE(i), GLOBAL_MODE_LINEAR)

    w += nvm50(SUBC_CP, CP_LOCAL_WARPS_LOG_ALLOC, 7)
    w += nvm50(SUBC_CP, CP_LOCAL_WARPS_NO_CLAMP, 1)
    w += nvm50(SUBC_CP, CP_STACK_WARPS_LOG_ALLOC, 7)
    w += nvm50(SUBC_CP, CP_STACK_WARPS_NO_CLAMP, 1)
    w += nvm50(SUBC_CP, CP_USER_PARAM_COUNT, 0)
    w += nvm50(SUBC_CP, CP_DMA_CODE_CB, vram_dma_handle)
    w += nvm50(SUBC_CP, CP_DMA_LOCAL, vram_dma_handle)
    w += nvm50(SUBC_CP, CP_LOCAL_ADDRESS_HIGH,
               (stack_addr >> 32) & 0xFF, stack_addr & 0xFFFFFFFF)
    w += nvm50(SUBC_CP, CP_LOCAL_SIZE_LOG, 4)

    # Per-launch setup (mesa nv50_launch_grid_with_input)
    w += nvm50(SUBC_CP, CP_CODE_CB_FLUSH, 0)
    w += nvm50(SUBC_CP, CP_CP_START_ID, kernel_addr & 0xFFFFFFFF)

    # Shared memory: align(smem + params + 0x14, 0x40)
    shared_size = (0x14 + 0x10 + 0x3F) & ~0x3F  # base + 4 params * 4 bytes
    w += nvm50(SUBC_CP, CP_SHARED_SIZE, shared_size)
    w += nvm50(SUBC_CP, CP_CP_REG_ALLOC_TEMP, 4)  # from cubin: reg=4
    w += nvm50(SUBC_CP, CP_BLOCKDIM_XY, (1 << 16) | block_dim)
    w += nvm50(SUBC_CP, CP_BLOCKDIM_Z, 1)
    w += nvm50(SUBC_CP, CP_BLOCK_ALLOC, (1 << 16) | block_dim)
    w += nvm50(SUBC_CP, CP_BLOCKDIM_LATCH, 1)
    w += nvm50(SUBC_CP, CP_GRIDDIM, (1 << 16) | grid_dim)
    w += nvm50(SUBC_CP, CP_GRIDID, 1)

    # User parameters: kernel expects a@0x0, b@0x4, out@0x8, N@0xc in shared mem.
    # USER_PARAM(i) writes to shared offset i*4, so params go to USER_PARAM(0-3).
    # No Z-counter needed for 1D launch (grid_z=1).
    param_count = 4 << 8
    w += nvm50(SUBC_CP, CP_USER_PARAM_COUNT, param_count)
    w += nvm50(SUBC_CP, CP_USER_PARAM(0), input_a)
    w += nvm50(SUBC_CP, CP_USER_PARAM(1), input_b)
    w += nvm50(SUBC_CP, CP_USER_PARAM(2), output)
    w += nvm50(SUBC_CP, CP_USER_PARAM(3), n_elements)

    # Launch + wait for idle
    w += nvm50(SUBC_CP, CP_LAUNCH, 0)
    w += nvm50(SUBC_CP, CP_WAIT_FOR_IDLE, 0)
    return w

# ============================================================================
# Channel setup callback (called by create_channel after RAMFC, before bind)
# ============================================================================
def make_setup_fn(ctxvals, ctxvals_size):
    def setup_fn(vram, inst_addr):
        # Extend GMMU for compute buffers (0x240000 - 0x2A0000)
        extend_gmmu(vram, inst_addr, 0x240000, 0x2A0000)

        # Write VRAM DMA object at inst+0xD500
        vram_dma = make_dma_object(0, 0xFFFFFFFFFF, target='vram', access='rw')
        vram.write_block(inst_addr + INST_VRAM_DMA_OFF, vram_dma)
        print(f'[chan] VRAM DMA at inst+0x{INST_VRAM_DMA_OFF:x}')

        # RAMHT entries
        # VRAM DMA: context = (DMA_ENGN_ID << 20) | (dma_offset >> 4)
        ramht_insert(vram, inst_addr, VRAM_DMA_HANDLE,
                     (DMA_ENGN_ID << 20) | (INST_VRAM_DMA_OFF >> 4))
        # Compute object: context = (GR_ENGN_ID << 20) | (gpuobj_off >> 4)
        # nv50_gr_object_bind creates a 16-byte GPU object with class ID at +0x00.
        # We place it at inst+0x40 (within eng block, VP slot is unused).
        GPUOBJ_OFF = 0x40
        vram.write32(inst_addr + GPUOBJ_OFF, NVA3_COMPUTE_CLASS)  # class ID
        vram.write32(inst_addr + GPUOBJ_OFF + 0x04, 0)
        vram.write32(inst_addr + GPUOBJ_OFF + 0x08, 0)
        vram.write32(inst_addr + GPUOBJ_OFF + 0x0c, 0)
        ramht_insert(vram, inst_addr, COMPUTE_HANDLE,
                     (GR_ENGN_ID << 20) | (GPUOBJ_OFF >> 4))

        # Bind GR context (ctxvals buffer in VRAM)
        gr_bind_context(vram.dev, vram, CHAN_ID, inst_addr,
                        CTXVALS_ADDR, ctxvals_size)

        # Upload ctxvals to VRAM
        print(f'[gr-ctx] Uploading ctxvals ({ctxvals_size} bytes) to 0x{CTXVALS_ADDR:08x}...')
        vram.write_block(CTXVALS_ADDR, ctxvals)

    return setup_fn

# ============================================================================
# Main
# ============================================================================
def run_add():
    dev = Dev()
    try:
        # Enable MSE
        cmd_reg = dev.cfg(0x04, 2)
        if not (cmd_reg & 0x0002):
            dev._rpc(4, 0, 0x04, 2, payload=struct.pack('<H', cmd_reg | 0x0002))
        dev.map_bar(0)
        dev.write32(0x000200, 0xffffffff)
        print(f'PMC_ENABLE = 0x{dev.read32(0x000200):08x}')

        # VBIOS devinit
        rom_path = os.path.join(os.path.dirname(__file__), 'en210.rom')
        with open(rom_path, 'rb') as f:
            image = f.read()
        scripts = find_vbios_scripts(image)
        for s in scripts:
            run_vbios_init(dev, image, scripts=[s], debug=False)
        print('Devinit complete.')

        # Verify VRAM
        vram = VRAM(dev)
        vram.write32(0x100000, 0xDEADBEEF)
        v = vram.read32(0x100000)
        print(f'VRAM test: {"PASS" if v == 0xDEADBEEF else "FAIL"} (0x{v:08x})')

        # FIFO init
        fifo_init(dev)

        # Load ctxprog + ctxvals
        ctxprog_globals = {}
        with open(os.path.join(os.path.dirname(__file__), 'ctxprog.py')) as f:
            exec(f.read(), ctxprog_globals)
        ctxprog = ctxprog_globals['ctxprog']
        ctxvals_size = ctxprog_globals['ctxvals_size']
        with open(os.path.join(os.path.dirname(__file__), 'ctxvals.bin'), 'rb') as f:
            ctxvals = f.read()
        print(f'[gr] ctxprog: {len(ctxprog)} instrs, ctxvals: {len(ctxvals)} bytes')

        # GR init
        gr_init(dev, ctxprog=ctxprog, ctxvals=ctxvals, ctxvals_size=ctxvals_size)

        # Upload kernel SASS to VRAM
        sass_path = os.path.join(os.path.dirname(__file__), 'add_sass.bin')
        with open(sass_path, 'rb') as f:
            sass = f.read()
        print(f'[kernel] Uploading {len(sass)} bytes to 0x{KERNEL_ADDR:08x}...')
        vram.write_block(KERNEL_ADDR, sass)
        verify = vram.read_block(KERNEL_ADDR, len(sass))
        print(f'[kernel] Upload {"verified" if verify == sass else "FAILED"}')

        # Prepare input data: a[i]=i, b[i]=i*10, expected out[i]=i*11
        N = int(os.environ.get("EN210_N", "4"))
        block_dim = int(os.environ.get("EN210_BLOCK", str(N)))
        grid_dim = (N + block_dim - 1) // block_dim
        a_bytes = struct.pack(f'{N}f', *[float(i) for i in range(N)])
        b_bytes = struct.pack(f'{N}f', *[float(i * 10) for i in range(N)])
        expected = [float(i) + float(i * 10) for i in range(N)]
        print(f'[data] N={N}, block={block_dim}, grid={grid_dim}')
        print(f'[data] expected out = {expected[:8]}{"..." if N > 8 else ""}')

        vram.write_block(INPUT_A_ADDR, a_bytes)
        vram.write_block(INPUT_B_ADDR, b_bytes)
        for off in range(0, N * 4, 4):
            vram.write32(OUTPUT_ADDR + off, 0)
        print(f'[data] a@0x{INPUT_A_ADDR:x}, b@0x{INPUT_B_ADDR:x}, out@0x{OUTPUT_ADDR:x}')

        # Build compute push buffer
        push = build_compute_push(
            vram_dma_handle=VRAM_DMA_HANDLE,
            kernel_addr=KERNEL_ADDR,
            stack_addr=STACK_ADDR,
            input_a=INPUT_A_ADDR,
            input_b=INPUT_B_ADDR,
            output=OUTPUT_ADDR,
            n_elements=N,
            block_dim=block_dim,
            grid_dim=grid_dim,
        )
        print(f'[compute] Push buffer: {len(push)} words')

        # Create channel and launch
        print('\n=== Compute Launch ===')
        setup_fn = make_setup_fn(ctxvals, ctxvals_size)
        t_kern0 = time.time()
        create_channel(dev, vram, push_words=push, setup_fn=setup_fn)
        t_kern1 = time.time()
        kernel_ms = (t_kern1 - t_kern0) * 1000.0
        print(f'[en210] kernel_time_ms={kernel_ms:.3f} N={N} block={block_dim} grid={grid_dim}')

        # Read back results
        print('\n=== Result Readback ===')
        time.sleep(0.5)

        # Check GR/FIFO status — print trap diagnostics only on error
        gr_intr = dev.read32(0x400100)
        gr_trap = dev.read32(0x400108)
        fifo_intr = dev.read32(0x002100)
        if gr_intr or gr_trap or fifo_intr:
            print(f'[gr] INTR=0x{gr_intr:08x} TRAP=0x{gr_trap:08x}')
            print(f'[fifo] INTR=0x{fifo_intr:08x}')
            units = dev.read32(0x001540)
            if gr_trap & 0x080:  # TRAP_MP
                for tpid in range(16):
                    if not (units & (1 << tpid)):
                        continue
                    tp_ustatus = dev.read32(0x40831c + (tpid << 11)) & 0x7fffffff
                    if tp_ustatus:
                        print(f'[gr] TP{tpid} ustatus=0x{tp_ustatus:08x}')
                    for mp in range(4):
                        if not (units & (1 << (mp + 24))):
                            continue
                        addr = 0x408100 + (tpid << 11) + (mp << 7)
                        status = dev.read32(addr + 0x14)
                        if status:
                            pc = dev.read32(addr + 0x24)
                            oplow = dev.read32(addr + 0x70)
                            ophigh = dev.read32(addr + 0x74)
                            print(f'[gr] MP_TRAP TP{tpid} MP{mp}: status=0x{status:08x} '
                                  f'pc=0x{pc:06x} op=0x{ophigh:08x}{oplow:08x}')
            if gr_trap & 0x100:  # TRAP_PROP
                for tpid in range(16):
                    if not (units & (1 << tpid)):
                        continue
                    ustatus = dev.read32(0x408e08 + (tpid << 11)) & 0x7fffffff
                    if ustatus:
                        fault_hi = dev.read32(0x408e08 + (tpid << 11) + 0x08)
                        fault_lo = dev.read32(0x408e08 + (tpid << 11) + 0x0c)
                        print(f'[gr] PROP_TRAP TP{tpid}: ustatus=0x{ustatus:08x} '
                              f'fault_addr=0x{fault_hi:02x}{fault_lo:08x}')
            if gr_trap & 0x001:  # DISPATCH
                addr = dev.read32(0x400808)
                class_id = dev.read32(0x400814)
                print(f'[gr] DISPATCH_TRAP: addr=0x{addr:08x} class=0x{class_id:04x}')

        results = []
        for i in range(N):
            raw = vram.read32(OUTPUT_ADDR + i * 4)
            val = struct.unpack('<f', struct.pack('<I', raw))[0]
            results.append(val)
            ok = "PASS" if val == expected[i] else "FAIL"
            print(f'  out[{i}] = {val} (expected {expected[i]})  {ok}')

        mismatches = sum(1 for r, e in zip(results, expected) if r != e)
        print(f'\n=== Summary: N={N} mismatches={mismatches}/{N} ===')
        if mismatches == 0:
            print(f'hardware_demo=ok N={N} block={block_dim} grid={grid_dim}')
        else:
            print(f'hardware_demo=FAIL N={N} mismatches={mismatches}/{N}')

    finally:
        dev.close()

def main():
    if "--probe" in sys.argv:
        dev = Dev()
        try:
            ven_dev = dev.cfg(0x00, 4)
            print(f"PCI_ID={ven_dev:#010x}")
            dev.map_bar(0)
            boot0 = dev.read32(0x000000)
            print(f"PMC_BOOT_0={boot0:#010x} chip_id={(boot0>>20)&0xFFF:#05x}")
        finally:
            dev.close()
        return
    run_add()

if __name__ == "__main__":
    main()
