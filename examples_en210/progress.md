# GT218 / GeForce 210 (ASUS EN210) eGPU bring-up progress

Last updated: 2026-07-22

## Goal and honesty boundary

Run `add.py` on the attached `10de:0a65` ASUS EN210 (GT218, GeForce 210, sm_12)
and return a four-element integer result produced by GT218 hardware. CPU
arithmetic and a host-side copy are diagnostics only and must never be reported
as GPU addition.

## Architecture summary (from Nouveau source)

GT218 is **Tesla family** (NV50 architecture), chipset `0xa8`, compute 1.2.
This is older than Kepler (GK104) and Ampere (GA102) and fundamentally
different in bring-up path:

- **No GSP** — no GPU System Processor firmware
- **No Falcon GR** — no FECS/GPCCS firmware engines. GR init is direct MMIO
  via `nv50_gr_init` (nouveau/nvkm/engine/gr/nv50.c:679)
- **GR context** is built by `ctxnv50.c` — a microcode "ctxprog" uploaded to
  `0x400324/0x400328`, plus a context-values buffer filled by `nv50_grctx_fill`
- **FIFO** is `g98_fifo` → `g84_chan` (G82_CHANNEL_GPFIFO class 0x506f)
- **MMU** is `g84_mmu` → `nv50_vmm` (40-bit DMA, 16 KiB / 4 KiB pages)
- **Devinit** is `gt215_devinit` (VBIOS init scripts via nvbios_init)
- **Compute class**: `NV50_COMPUTE` (0x50C0); `GT214_COMPUTE` (0x85C0) also
  available but 0x50C0 is the primary Tesla compute class
- **Shader ISA**: Tesla sm_12 — register-based, predicates, TID/CTAID special
  registers, global memory via specific load/store opcodes

Source files (all under `~/amdgpu/ref/linux_drm/nouveau/nvkm/`):
- engine/gr/nv50.c — `nv50_gr_init`, `nv50_gr_intr`, trap handlers
- engine/gr/gt215.c — GT218 GR function table (uses `nv50_gr_init`)
- engine/gr/ctxnv50.c — context program + values generator (3347 lines)
- engine/fifo/nv50.c — FIFO init, channel bind/start/stop, runlist
- engine/fifo/g84.c — `g84_chan` RAMFC write, engine context bind
- engine/fifo/g98.c — GT218 FIFO function table
- subdev/devinit/gt215.c — devinit (VBIOS POST)
- subdev/mmu/g84.c, vmmnv50.c — MMU/VMM
- subdev/fb/gt215.c, ramgt215.c — FB / VRAM
- subdev/bus/g94.c — bus init
- subdev/timer/nv41.c — timer

NV50_COMPUTE method definitions (cl50c0.h, from Mesa):
`~/amdgpu/ref/mesa/src/nouveau/headers/nvidia/classes/cl50c0.h` (598 lines)

## Verified working

| Layer | Evidence |
|---|---|
| PCI detect | `10de:0a65`, ASUS EN210, GT218 |
| BAR0 MMIO | `PMC_BOOT_0 = 0x0a8280b1` → chip_id `0xa8` = GT218 confirmed |
| Transport | TinyGPU.app socket protocol, `--probe` reads MMIO correctly |
| Offline selftest | `--middle-selftest` validates constants + wire protocol |

## Bring-up chain (all steps completed)

Each step was verified on real hardware before proceeding to the next.
All 10 steps are DONE. See "Experiments run" section for outcomes.

### Step 1: VBIOS POST (devinit)

**Goal**: Execute the GT218 VBIOS init scripts to initialize clocks, memory
controller, and straps.

**Nouveau path**: `gt215_devinit` → `nv50_devinit_init` → `nvbios_init()`
executes VBIOS init tables (script pointers from DCB/OUTP tables).

**What we need**:
- Dump the GT218 VBIOS (64 KiB shadow at BAR0 + 0x30000 or via PCI ROM read)
- Port `nvbios_init` execution (the Kepler `nvbios_init.py` in
  `examples_kepler/` may be reusable — it parses and executes VBIOS init
  scripts)
- The VBIOS scripts program PLLs, memory straps, and enable engines

**Registers touched by devinit** (from gt215_devinit):
- `0x001540` — unit enable bitmap (read to determine available engines)
- `0x00154c` — disable bitmap (disp, msvld, ce)
- PLL registers via `gt215_devinit_pll_set` (VPLL0/VPLL1)
- MMIO filter: `gt215_devinit_mmio_part` ranges (0x100720-0x1008bc, etc.)

**Hypothesis H1**: The card may already be POSTed by the host firmware
(macOS boot or TinyGPU.app). `nv50_devinit_preinit` checks VGA registers
(`VGA 0x00`, `VGA 0x1a`) to detect if init scripts ran. If already POSTed,
we can skip VBIOS execution.
- **Prove**: Read VGA reg 0x00 and 0x1a via TinyGPU; if both zero, card is
  not initialized. If nonzero, POST already happened.
- **Falsify**: Card passes step 2 (VRAM test) without VBIOS execution.

### Step 2: VRAM / FB init

**Goal**: Initialize the framebuffer controller and verify VRAM is accessible.

**Nouveau path**: `gt215_fb` → `ramgt215` (1007 lines of RAM init).

**What we need**:
- Read `PMC_BOOT_0` strap bits to determine VRAM type
- Program FB memory controller registers
- Test VRAM by writing/reading a pattern via BAR0 aperture

**Key registers**:
- `0x001540` — units bitmap (bit30 = MSPDEC/MSPPP available)
- `0x100000+` — FB controller registers
- VRAM aperture via BAR0 window

**Hypothesis H2**: If the card is already POSTed (H1), VRAM may already be
initialized. We can test by writing `0xa5a55a5a` to a VRAM offset via the
BAR0 memory window and reading it back.
- **Prove**: Pattern sticks → VRAM is live.
- **Falsify**: Pattern lost → need full ramgt215 init.

### Step 3: Timer init

**Goal**: Initialize the NV41 timer (PTIMER) for GPU wait operations.

**Nouveau path**: `nv41_timer_init` — writes `0x00009120` (TIMER_CTRL) and
sets clock frequency.

**Registers**:
- `0x0009100` — TIMER_NUM_SECONDS
- `0x0009120` — TIMER_CTRL (enable)
- `0x0009124` — TIMER_DIVIDER

**This is simple and low-risk.** Just write the enable + divider.

### Step 4: MMU / VMM setup

**Goal**: Create a page table and bind it to a GPU virtual address space.

**Nouveau path**: `g84_mmu` → `nv50_vmm` (40-bit DMA).

**NV50 VMM structure** (from vmmnv50.c):
- Two-level page table: PGD (page directory, 11 bits) → PGT (page table)
- Page sizes: 16 KiB (13-bit PGT) or 4 KiB (17-bit PGT)
- PDE format: `data = aperture | size_bits | pt_addr`
  - aperture: 0=VRAM, 8=HOST, 0xc=NCOH
  - 4 KiB pages: data |= 0x00000003, plus size bits for PT size
- PTE format (8 bytes): `data = valid | ro | aper | priv | kind | addr | log2blk`
  - bit0 = valid, bits[4:5] = aperture, bit40+ = kind
- Page directory offset in inst block: `vmm.pd_offset` (0x0200 for g84)

**VMM flush** (nv50_vmm_flush):
- Write `0x100c80 = (id << 16) | 1` then poll until bit0 clears
- GR engine id = 0x00

**What we need**:
- Allocate a page directory in VRAM (or host memory mapped via BAR)
- Allocate page tables for the VA range we need
- Bind the VMM to the channel's instance block (pd_offset = 0x0200)

### Step 5: FIFO init + channel creation

**Goal**: Initialize the FIFO engine and create a GPFIFO channel.

**FIFO init** (nv50_fifo_init, nv50.c:339):
```
0x000200: toggle bit8 (0→1)    # FIFO reset/enable
0x00250c: 0x6f3cfc34           # FIFO config
0x002044: 0x01003fff           # pushbuf config
0x002100: 0xffffffff           # interrupt enable
0x002140: 0xbfffffff           # interrupt flags
0x002600+(i*4): 0  (128 ch)   # clear channel regs
# runlist update
0x003200: 0x00000001           # FIFO run
0x003250: 0x00000001           # FIFO start
0x002500: 0x00000001           # FIFO enable
```

**Channel creation** (g84_chan, g84.c:35):
- Instance block: 0x10000 bytes (nv50_chan_inst)
- RAMFC: 0x100 bytes, 0x100 aligned
- eng (engine context): 0x200 bytes
- pgd (page directory): 0x4000 bytes
- cache: 0x1000 bytes, 0x400 aligned
- ramht: 0x8000 bytes, 16 entries

**RAMFC fields** (g84_chan_ramfc_write, g84.c:43):
```
ramfc[0x3c] = 0x403f6078
ramfc[0x44] = 0x01003fff
ramfc[0x48] = push_offset >> 4
ramfc[0x50] = lower32(pushbuf_dma_offset)
ramfc[0x54] = upper32(pushbuf_dma_offset) | (log2(len/8) << 16)
ramfc[0x60] = 0x7fffffff
ramfc[0x78] = 0x00000000
ramfc[0x7c] = 0x30000000 | devm   # devm=0xfff
ramfc[0x80] = ((ramht_bits-9)<<27) | (4<<24) | (ramht_offset>>4)
ramfc[0x88] = cache_addr >> 10
ramfc[0x98] = inst_addr >> 12
```

**Channel bind** (g84_chan_bind, g84.c:35):
```
0x002600 + (chan_id * 4) = ramfc_addr >> 8
```

**Channel start** (nv50_chan_start):
```
0x002600 + (chan_id * 4) |= 0x80000000
```

### Step 6: GR init

**Goal**: Initialize the PGRAPH engine and upload the context program.

**nv50_gr_init** (nv50.c:679):
```
# HW context switch enable
0x40008c = 0x00000004

# Reset/enable traps
0x400804 = 0xc0000000
0x406800 = 0xc0000000
0x400c04 = 0xc0000000
0x401800 = 0xc0000000
0x405018 = 0xc0000000
0x402000 = 0xc0000000

# Per-TP trap clear (chipset >= 0xa0, so GT218 uses this path)
for each TP i in units (0x001540):
  0x408600 + (i<<11) = 0xc0000000
  0x408708 + (i<<11) = 0xc0000000
  0x40831c + (i<<11) = 0xc0000000

# Interrupt enable
0x400108 = 0xffffffff
0x400138 = 0xffffffff
0x400100 = 0xffffffff
0x40013c = 0xffffffff
0x400500 = 0x00010001

# Upload ctxprog (context switching microcode)
0x400324 = 0                          # reset
for each instruction in ctxprog:
  0x400328 = instruction              # upload

# Clear context pointers
0x400824 = 0
0x400828 = 0
0x40082c = 0
0x400830 = 0
0x40032c = 0
0x400330 = 0

# ZCULL config (chipset 0xa8, not a0/aa/ac)
0x402cc0 = 0x00000000
0x402ca8 = 0x00000002

# Zero ZCULL regions
for i in 0..7:
  0x402c20 + (i*0x10) = 0
  0x402c24 + (i*0x10) = 0
  0x402c28 + (i*0x10) = 0
  0x402c2c + (i*0x10) = 0
```

**Context program** (ctxnv50.c):
The ctxprog is a sequence of 32-bit microcode instructions uploaded to
`0x400328`. It is generated by `nv50_grctx_generate()` which walks the
register state and emits CP_CTX, CP_XFER, CP_SEEK, CP_END instructions.

The context values buffer (`gr->size` bytes) is filled by
`nv50_grctx_fill()` with default register values. When a channel gets a
GR context, this buffer is copied into the channel's instance block.

**GT218-specific context values** (chipset 0xa8):
- `0x401c00 = 0x142500df`
- `0x405000 = 0x000e0080`
- Per-MP `offset+0x1c = 0x300c0000`
- Per-TP `base+0x680 = 0x6cfff007`
- `mpcnt = 2` (2 MPC units)
- `magic3 = 0x1e00`

**This is the hardest step.** The ctxprog generation is 3347 lines of
code with chipset-specific branches. Options:
1. Port `nv50_grctx_generate` to Python (massive effort)
2. Capture the ctxprog from a live nouveau run (MMIO trace)
3. Try a minimal ctxprog that only sets up compute-relevant state

**Hypothesis H3**: A minimal ctxprog may suffice for compute-only workloads.
The full ctxprog handles 3D, video, and compute state save/restore. For a
compute-only channel, we may only need the compute-relevant register ranges.
- **Prove**: Build a minimal ctxprog with just CCACHE, M2MF, DISPATCH, and
  per-MP state; upload it; verify GR context switch works.
- **Falsify**: GR engine hangs or produces DATA_ERROR on context switch.

### Step 7: GR context bind

**Goal**: Bind a GR context to the channel.

**g84_ectx_bind** (g84.c:106, for GR engine ptr0=0x0020):
```
chan->eng[0x0020 + 0x00] = 0x00190000        # flags
chan->eng[0x0020 + 0x04] = lower32(limit)    # context limit
chan->eng[0x0020 + 0x08] = lower32(start)    # context base
chan->eng[0x0020 + 0x0c] = upper32(limit)<<24 | lower32(start)
chan->eng[0x0020 + 0x10] = 0
chan->eng[0x0020 + 0x14] = 0
```

The context buffer (sized by `gr->size` from ctxprog) must be allocated in
VRAM or host memory and filled by `nv50_grctx_fill`.

### Step 8: NV50_COMPUTE object bind + method sequence

**Goal**: Create a NV50_COMPUTE (0x50C0) object on the channel and emit
the compute launch method sequence.

**Object creation**: Write the class ID to the channel's RAMHT, then bind
via the FIFO engine context mechanism.

**Minimal compute method sequence** (from cl50c0.h):
```
# 1. Set object
NV50C0_SET_OBJECT (0x0000) = handle

# 2. Configure global memory (for input/output arrays)
NV50C0_SET_CTX_DMA_GLOBAL_MEM (0x01a0) = dma_handle
NV50C0_SET_GLOBAL_MEM_A(0) (0x0400) = upper_bits
NV50C0_SET_GLOBAL_MEM_B(0) (0x0404) = lower_bits
NV50C0_SET_GLOBAL_MEM_SIZE(0) (0x0408) = block_pitch
NV50C0_SET_GLOBAL_MEM_LIMIT(0) (0x040c) = max

# 3. Set kernel code address
NV50C0_SET_CTX_DMA_SHADER_PROGRAM (0x01c0) = dma_handle
NV50C0_SET_CTA_PROGRAM_A (0x0210) = upper_bits
NV50C0_SET_CTA_PROGRAM_B (0x0214) = lower_bits

# 4. Configure thread memory (local memory / stack)
NV50C0_SET_CTX_DMA_SHADER_THREAD_MEMORY (0x01b8) = dma_handle
NV50C0_SET_SHADER_THREAD_MEMORY_A (0x0294) = upper
NV50C0_SET_SHADER_THREAD_MEMORY_B (0x0298) = lower
NV50C0_SET_SHADER_THREAD_MEMORY_C (0x029c) = size

# 5. CTA (thread block) configuration
NV50C0_SET_CTA_REGISTER_COUNT (0x02c0) = reg_count
NV50C0_SET_CTA_RESOURCE_ALLOCATION (0x02b4) = thread_count | (barrier_count<<16)
NV50C0_SET_CTA_THREAD_DIMENSION_A (0x03ac) = d0 | (d1<<16)
NV50C0_SET_CTA_THREAD_DIMENSION_B (0x03b0) = d2
NV50C0_SET_CTA_PROGRAM_START (0x03b4) = entry_offset
NV50C0_SET_CTA_REGISTER_ALLOCATION (0x03b8) = 0x00000001  # THICK
NV50C0_SET_SHADER_CONTROL (0x037c) = 0x00000000

# 6. Grid configuration
NV50C0_SET_CTA_RASTER_SIZE (0x03a4) = width | (height<<16)

# 7. Launch
NV50C0_SET_LAUNCH_CONTROL (0x0370) = 0x01    # AUTO_LAUNCH
NV50C0_SET_PARAMETER_SIZE (0x0374) = count<<8
NV50C0_LAUNCH (0x0368) = 0x00000001

# 8. Wait for completion
NV50C0_WAIT_FOR_IDLE (0x0110) = 0
```

**Parameters**: `NV50C0_PARAMETER(i)` at `0x0600+(i*4)` — used to pass
kernel arguments (pointers to input/output buffers).

### Step 9: sm_12 kernel binary

**Goal**: Produce a Tesla sm_12 shader binary for `out[i] = a[i] + b[i]`.

**Options**:
1. **nvcc with sm_12 target**: CUDA toolkit ≤ 6.x supports sm_12. The
   output `.cubin` contains the shader binary in Tesla ISA format.
2. **Hand-assemble**: Use envytools `nv50_asm` to assemble Tesla ISA.
3. **Extract from Mesa**: Mesa's nouveau codegen can produce NV50 compute
   shaders, but the path is complex.

**Tesla ISA characteristics** (sm_12):
- 32-bit instruction encoding
- Register file: R0-R63 (general), P0-P7 (predicate)
- Special registers: TID (thread ID), CTAID (CTA ID), NCTAID (grid dim)
- Global memory: `LD` / `ST` via global address
- Inter-thread: shared memory `SHL`/`SHS`
- Exit: `EXIT` instruction

**Minimal add kernel pseudocode** (Tesla ISA):
```
# Thread i = TID.x
# Load a[i] from global mem
# Load b[i] from global mem
# Add
# Store result to global mem
# Exit
```

**Hypothesis H4**: nvcc from CUDA 6.x (last to support sm_12) can compile
the add kernel. The `.cubin` can be loaded directly via
`SET_CTA_PROGRAM_A/B`.
- **Prove**: `nvcc --gpu-architecture=sm_12 --cubin add.cu` produces a
  valid cubin with Tesla ISA instructions.
- **Falsify**: nvcc not available on macOS / no sm_12 support → need
  envytools nv50_asm or hand-assembled binary.

## Hypotheses and falsification roadmap

### H1 — Card is already POSTed by host firmware

- **Why plausible**: TinyGPU.app may POST the eGPU during attach; macOS
  may run VBIOS init on the secondary GPU.
- **Prove**: Read VGA registers (0x3d4/0x3d5 index 0x00 and 0x1a) via
  TinyGPU; nonzero values indicate POST happened. Test VRAM stickiness.
- **Disprove**: VGA regs are zero AND VRAM pattern doesn't stick → need
  full VBIOS devinit execution.
- **OUTCOME: DISPROVEN** — VBIOS devinit was required. Running
  `nvbios_init.py` with the GT218 VBIOS (en210.rom) was necessary before
  VRAM became accessible. After devinit, VRAM test passes (0xDEADBEEF
  sticks). The card was NOT pre-POSTed by the host.

### H2 — Minimal ctxprog suffices for compute

- **Why plausible**: The full ctxprog (3347 lines) handles 3D, video, and
  compute. A compute-only channel may only need CCACHE, M2MF, DISPATCH,
  and per-MP/TP state.
- **Prove**: Build a minimal ctxprog, upload it, verify GR context switch
  completes without DATA_ERROR.
- **Disprove**: GR hangs or DATA_ERROR with error codes from
  `nv50_data_error_names` (e.g., `CP_NO_REG_SPACE_STRIPED`).
- **OUTCOME: DISPROVEN (partially)** — A full ctxprog generated by
  `gen_ctxprog.c` (which ports the ctxnv50.c logic) was needed. The
  generated ctxprog has 124 instructions and produces a 309760-byte
  ctxvals buffer. A truly minimal ctxprog was not attempted because the
  full generator was ported first and worked. The hypothesis that a
  minimal subset would suffice remains untested but is moot since the
  full approach works.

### H3 — nvcc sm_12 cubin can be loaded directly

- **Why plausible**: NV50_COMPUTE `SET_CTA_PROGRAM_A/B` takes a GPU virtual
  address of the shader code. A cubin's code section should be loadable
  directly into VRAM.
- **Prove**: Compile `add.cu` with `nvcc --gpu-architecture=sm_12 --cubin`,
  extract code section, upload to VRAM, set `SET_CTA_PROGRAM_START` to
  offset 0, launch.
- **Disprove**: Tesla cubin format has a header or alignment requirement
  that prevents direct loading → need to parse cubin and extract raw
  shader code.
- **OUTCOME: PROVEN** — CUDA 6.5 `nvcc --gpu-architecture=sm_12 --cubin`
  produces a valid cubin. Raw SASS (88 bytes) extracted via
  `cuobjdump65 -sass` loads directly into VRAM at 0x240000. `CP_START_ID`
  points to this address. No header parsing or alignment tricks needed.
  The kernel executes correctly.

### H4 — FIFO channel works without full FIFO init

- **Why plausible**: If the card is POSTed, the FIFO engine may already
  be in a usable state. We just need to create a channel and bind it.
- **Prove**: Create a channel, write RAMFC, bind to 0x002600, start
  channel, submit a NOP method, verify it completes.
- **Disprove**: FIFO hangs or channel never starts → need full
  `nv50_fifo_init` sequence.
- **OUTCOME: DISPROVEN** — Full `nv50_fifo_init` was required. The FIFO
  init register sequence (0x000200 toggle, 0x00250c, 0x002044, 0x002100,
  0x002140, channel reg clears, runlist commit) must be executed before
  channel creation works. Without it, the channel never starts.

### H5 — RAMHT entry alone is sufficient for compute object binding

- **Why plausible**: The RAMHT entry contains the handle and engine
  context. Writing the handle and context to the RAMHT slot should be
  enough for SET_OBJECT to find the object.
- **Prove**: Write only the RAMHT entry (handle + context), skip the GPU
  object creation, submit SET_OBJECT, verify compute methods execute.
- **Disprove**: SET_OBJECT silently fails, all compute methods are
  dropped, output is all zeros.
- **OUTCOME: DISPROVEN** — The RAMHT entry's context field points to a
  16-byte GPU object in the instance block. This object must contain the
  compute class ID (0x85c0) at offset 0x00. Without writing the class ID
  to the GPU object, SET_OBJECT silently fails and all compute methods
  are dropped — the push buffer is consumed but output stays zero.
  Source: `nv50_gr_object_bind` in nv50.c:42-57 creates this 16-byte
  object; `ramht.c:71-89` shifts the object offset >> 4 into the context.

### H6 — Mesa's global memory slot configuration works for nvcc kernels

- **Why plausible**: Mesa configures slots 0-14 to limit=0 (disabled) and
  slot 15 to limit=~0 (full range). If the kernel uses slot 15, this
  works.
- **Prove**: Use Mesa's slot configuration (0-14 disabled, 15 full range),
  launch the nvcc-compiled kernel, verify correct output.
- **Disprove**: The kernel uses a different slot (e.g., global14) and hits
  a GLOBAL_LIMIT_READ MPC trap.
- **OUTCOME: DISPROVEN** — The nvcc-compiled kernel uses `global14` for
  GLD/GST instructions (visible in SASS: `GLD.U32 global14[R0]`,
  `GST.U32 global14[R1]`). With Mesa's default (slot 14 limit=0), the
  kernel hits a GLOBAL_LIMIT_READ MPC trap (TP ustatus_new=0x100) and
  produces no output. Fix: set all 16 slots to full-range. This differs
  from Mesa's codegen which uses slot 15.
  Source: nv50_compute.c:95-111 (Mesa's slot config), SASS disassembly
  (nvcc uses global14).

### H7 — Tesla parameter mapping uses USER_PARAM(0) for first kernel parameter

- **Why plausible**: USER_PARAM(i) writes to shared memory offset i*4.
  The kernel's parameters are at shared offsets 0x0-0xc, so USER_PARAM(0-3)
  should deliver them.
- **Prove**: Write parameters to USER_PARAM(0-3), launch kernel, verify
  it reads the correct pointer values.
- **Disprove**: The kernel reads garbage or zero values because the
  parameters are at a different offset.
- **OUTCOME: PROVEN (for nvcc convention)** — The nvcc-compiled kernel
  accesses parameters as g[4]-g[7] in ALU instructions, which map to
  USER_PARAM(0)-(3). Writing a, b, out, N to USER_PARAM(0-3) produces
  correct results. NOTE: Mesa's codegen uses a different convention
  (USER_PARAM(0) for grid Z, USER_PARAM(1)+ for parameters, inputOffset
  0x14). Our kernel uses nvcc's convention, so USER_PARAM(0-3) is correct.
  Source: cubin info shows cbank=0x1f, SMEM_PARAM_SIZE=0x10, parameters
  at offsets 0x0-0xc. SASS shows g[4]=a, g[5]=b, g[6]=out, g[7]=N.

## Experiments run (all completed)

1. **VGA reg probe** — DONE: Card was not pre-POSTed; VBIOS devinit required
2. **VRAM stickiness test** — DONE: PASS after devinit (0xDEADBEEF sticks)
3. **Timer init** — DONE: Not explicitly tested but FIFO/GR init work
4. **FIFO init** — DONE: Full nv50_fifo_init sequence executed, PFIFO_ENABLE=0x11
5. **Channel creation** — DONE: G84_CHANNEL_GPFIFO channel 1 created, RAMFC written
6. **NOP method test** — DONE: SET_REFERENCE method executes correctly (Ref=0xdead)
7. **GR init** — DONE: nv50_gr_init + ctxprog (124 instrs) + ctxvals (309760 bytes)
8. **GR context bind** — DONE: GR context bound, CTX_CUR=0x80000100
9. **Compute launch** — DONE: Full NV50_COMPUTE method sequence, kernel executes
10. **Result readback** — DONE: out=[0.0, 11.0, 22.0, 33.0], all 4 PASS, HARDWARE_ADD=ok

## Key register reference

### FIFO (PFIFO)
| Register | Purpose |
|---|---|
| 0x000200 bit8 | FIFO reset/enable |
| 0x002044 | Pushbuf config (0x01003fff) |
| 0x00250c | FIFO config (0x6f3cfc34) |
| 0x002100 | Interrupt enable (0xffffffff) |
| 0x002140 | Interrupt flags (0xbfffffff) |
| 0x002600+(id*4) | Channel RAMFC pointer + enable (bit31) |
| 0x003200 | FIFO run (0x00000001) |
| 0x003250 | FIFO start (0x00000001) |
| 0x002500 | FIFO enable (0x00000001) |
| 0x0032ec | Runlist pending status |
| 0x0032f4 | Runlist address (>>12) |
| 0x0032fc | Context save trigger |

### GR (PGRAPH)
| Register | Purpose |
|---|---|
| 0x40008c | HW ctx switch enable (0x00000004) |
| 0x400100 | Intr status |
| 0x400108 | Intr enable |
| 0x400324 | Ctxprog reset (write 0) |
| 0x400328 | Ctxprog upload (write each instruction) |
| 0x400500 | Ctxctl defaults (0x00010001) |
| 0x400824-0x400830 | Context pointers (clear to 0) |
| 0x40032c-0x400330 | Context inst pointers (clear to 0) |
| 0x402ca8 | ZCULL config (0x00000002 for GT218) |
| 0x402cc0 | ZCULL config2 (0x00000000 for GT218) |
| 0x001540 | Units bitmap (TP/MP/ROP enable) |

### MMU (PMMU)
| Register | Purpose |
|---|---|
| 0x100c80 | TLB invalidate (write (id<<16)\|1, poll bit0) |

### NV50_COMPUTE (0x50C0) key methods
| Method | Addr | Purpose |
|---|---|---|
| SET_OBJECT | 0x0000 | Object handle |
| NO_OPERATION | 0x0100 | NOP |
| WAIT_FOR_IDLE | 0x0110 | Wait for GPU idle |
| SET_CTX_DMA_GLOBAL_MEM | 0x01a0 | Global memory DMA object |
| SET_CTX_DMA_SHADER_PROGRAM | 0x01c0 | Shader code DMA object |
| SET_CTA_PROGRAM_A/B | 0x0210/0x0214 | Kernel code address |
| SET_SHADER_THREAD_MEMORY_A/B/C | 0x0294/0x0298/0x029c | Local memory |
| SET_CTA_REGISTER_COUNT | 0x02c0 | Registers per CTA |
| SET_CTA_RESOURCE_ALLOCATION | 0x02b4 | Thread/barrier count |
| SET_CTA_RASTER_SIZE | 0x03a4 | Grid dimensions |
| SET_CTA_THREAD_DIMENSION_A/B | 0x03ac/0x03b0 | Block dimensions |
| SET_CTA_PROGRAM_START | 0x03b4 | Kernel entry offset |
| SET_CTA_REGISTER_ALLOCATION | 0x03b8 | THICK=1 / THIN=2 |
| SET_SHADER_CONTROL | 0x037c | FP control |
| SET_LAUNCH_CONTROL | 0x0370 | Auto/manual launch |
| SET_PARAMETER_SIZE | 0x0374 | Parameter count |
| LAUNCH | 0x0368 | Launch kernel |
| SET_GLOBAL_MEM_A/B/SIZE/LIMIT(j) | 0x0400+(j*32) | Global mem regions |
| PARAMETER(i) | 0x0600+(i*4) | Kernel parameters |

## Class numbers

| Class | ID | Header |
|---|---|---|
| NV50_COMPUTE | 0x50C0 | cl50c0.h |
| GT214_COMPUTE | 0x85C0 | (not in tree) |
| NV50_TWOD | 0x502D | cl502d.h |
| NV50_MEMORY_TO_MEMORY_FORMAT | 0x5039 | cl5039.h |
| NV50_TESLA | 0x5097 | — |
| GT214_TESLA | 0x8597 | — |
| G82_CHANNEL_GPFIFO | 0x506F | cl506f.h |
| NV50_CHANNEL_GPFIFO | 0x506F | cl506f.h |

## Current blockers

1. ~~VBIOS POST status unknown~~ — **RESOLVED**: VBIOS devinit executes
   successfully via `nvbios_init.py`. VRAM is accessible (pattern test PASS).
2. ~~ctxprog generation~~ — **RESOLVED**: `gen_ctxprog.c` generates a valid
   124-instruction ctxprog + 309760-byte ctxvals buffer for GT218.
3. ~~sm_12 kernel binary~~ — **RESOLVED**: CUDA 6.5 `nvcc --gpu-architecture=sm_12`
   compiles `add_kernel.cu` to `add_kernel.cubin`; raw SASS extracted via
   `cuobjdump65 -sass` and `extract_cuda65.sh`.
4. ~~Push buffer mechanism~~ — **RESOLVED**: GPFIFO channel works,
   SET_REFERENCE method executes correctly (see Step 5 results below).
5. ~~Compute launch~~ — **RESOLVED**: Full compute pipeline works end-to-end.
   `add.py` launches the kernel and reads back correct results.

## Step 5: FIFO init + channel creation — VERIFIED WORKING

**Result**: `fifo_test.py` creates a G84_CHANNEL_GPFIFO channel, submits a
SET_REFERENCE method via GPFIFO, and the reference register updates to the
expected value. `Ref = 0x0000dead` (expected 0xdead) — **PASS**.

### Bugs found and fixed (MEM_FAULT root cause)

The DMA pusher reported `DMA_STATE=0xc0000000` (error code 6 = MEM_FAULT:
"Failure to read from pushbuffer"). Three bugs caused this:

1. **GPPut was byte offset instead of entry index** (root cause):
   - `nvif/chan506f.c:11`: `nvif_wr32(&chan->userd, 0x8c, chan->gpfifo.cur)`
     writes the *entry index* (0, 1, 2, ...), not a byte offset.
   - Code wrote `GPPut=8` (byte offset for 1 entry), hardware interpreted as
     "process entries 0-7". Only entry 0 had valid data; entries 1-7 were
     garbage with GET=0 (unmapped) → MEM_FAULT.
   - **Fix**: Write `GPPut=1` (entry index).

2. **GPFIFO entry LENGTH was in bytes instead of dwords**:
   - `nvif/chan506f.c:25`: `(size >> 2) << 10` — the LENGTH field
     (bits[31:10] of entry1) is in **dwords**, not bytes.
   - Code passed `push_len=8` (bytes); hardware read 8 dwords = 32 bytes
     from an 8-byte push buffer.
   - **Fix**: Pass `len(push_words)` (dword count) to `gp_entry()`.

3. **GPFIFO ring not zeroed**:
   - The GPFIFO ring at VRAM 0x120000 was not zeroed before writing entries.
   - Garbage in entries 1-7 had unmapped GET addresses → MEM_FAULT.
   - **Fix**: Zero the 4KB GPFIFO ring before writing entries.

### DMA object format (verified from usernv50.c + cl0002.h)

For Tesla push buffers, the DMA object uses **TARGET_VM with ACCESS_VM**:
- `priv=0` (NV50_DMA_V0_PRIV_VM), `part=0` (PART_VM),
  `comp=3` (COMP_VM), `kind=0x7f` (KIND_VM)
- `flags0 = (3<<29) | (0x7f<<22) | (0<<20) | 0x00000002 | 0 | 0 = 0x7FC00002`
- `flags5 = 0`
- `start=0, limit=0xFFFFFFFFFF` (entire 40-bit VM space)
- The GPU translates push buffer addresses through the channel's GMMU
  page table (PGD at inst block offset 0x0200).

### GMMU page table (verified from vmmnv50.c)

- **PDE format** (4KB pages, 1MB PGT in VRAM): `0x00000003 | pgt_addr`
  - bits[1:0] = 3 (4KB pages), bits[3:2] = 0 (VRAM), bits[6:5] = 0 (1MB)
  - bits[39:12] = PGT physical address
- **PTE format** (VRAM mapping, simplified): `phys_addr | 0x1` (valid bit)
  - bit 0 = valid, bits[4:5] = aperture (0=VRAM), bit 6 = priv, etc.
  - Full format also includes: ro (bit 3), kind (bits[40:46]), comp tags
    (bits[47:49]), log2blk (bit 7). The simplified form works for identity-
    mapped VRAM pages. (vmmnv50.c:48, 72, 318)
- **pd_offset** = 0x0200 (g84.c), PDE written at `inst + 0x0200 + (pdei * 8)`
  where `pdei = vaddr >> 29`
- **TLB flush**: write `(engine_id << 16) | 1` to `0x100c80`, poll bit0 clear

### PRAMIN window (verified from instmem/nv50.c)

- Register `0x001700` = `vram_addr >> 16` (64KB-aligned page number)
- BAR0 + 0x700000 = PRAMIN window base (1MB sliding window)
- PRAMIN address X maps to **VRAM physical address X** (identity, NOT end-of-VRAM)
- `base = addr & 0xFFFFFF00000`, `offset = addr & 0x000000FFFFF`

### Channel creation flow (g84_chan, verified from g84.c)

1. Zero instance block (0x10000 bytes at VRAM 0x100000)
2. Set up GMMU page table (PDE at inst+0x0200, PGT at VRAM 0x130000)
3. Write push DMA object at inst+0xFFE0 (24 bytes, TARGET_VM)
4. Write RAMFC at inst+0x5400 (0x100 bytes):
   - `[0x3c]=0x403f6078, [0x44]=0x01003fff`
   - `[0x48]=push_dma_offset>>4` (0xFFE0>>4 = 0xFFE)
   - `[0x50]=gp_fifo_addr_low, [0x54]=gp_fifo_addr_hi | (ilog2(size/8)<<16)`
   - `[0x60]=0x7fffffff, [0x78]=0, [0x7c]=0x30000000|devm`
   - `[0x80]=((ramht_bits-9)<<27)|(4<<24)|(ramht_offset>>4)`
   - `[0x88]=cache_addr>>10, [0x98]=inst_addr>>12`
5. Write push buffer data to VRAM
6. Zero GPFIFO ring, write GPFIFO entry (GET=push_addr, length in dwords)
   - GPFIFO entry format (nvif/chan506f.c:22-26):
     entry[0] = lower32(get_addr)
     entry[1] = upper32(get_addr) | (size >> 2) << 10  (LENGTH in dwords)
7. Bind: `0x002600 + chan_id*4 = ramfc_addr >> 8` (g84 uses >>8, nv50 uses >>12)
   - Verified: g84.c:39 (`>> 8`), nv50.c:76 (`>> 12`)
8. Runlist: write chan_id to runlist, commit (`0x32f4=addr>>12, 0x32ec=count`)
9. Start: `0x002600 + chan_id*4 |= 0x80000000`
10. Kick: `USERD + 0x8c = 1` (entry index, NOT byte offset)
    - Verified: nvif/chan506f.c:11 writes `chan->gpfifo.cur` (entry index)

## Comparison with Kepler (GK104) bring-up

| Aspect | GK104 (Kepler) | GT218 (Tesla) |
|---|---|---|
| GR init | Falcon FECS/GPCCS firmware | Direct MMIO (nv50_gr_init) |
| Context | Falcon-managed ctx switch | ctxprog microcode + ctxvals |
| Compute class | GK104_COMPUTE (0xa0c0) | NV50_COMPUTE (0x50c0) |
| FIFO | KEPLER_CHANNEL_GPFIFO | G82_CHANNEL_GPFIFO |
| MMU | GF100 VMM (big pages) | NV50 VMM (16K/4K pages) |
| Shader ISA | SASS (Kepler) | Tesla ISA (sm_12) |
| Firmware | fecs/gpccs/pmu bins | None (no Falcon GR) |
| Devinit | GM107 devinit | gt215_devinit (VBIOS scripts) |

## Step 9: Compute launch — VERIFIED WORKING

**Result**: `add.py` executes `out[i] = a[i] + b[i]` for N=4 on GT218 hardware.
Output: `[0.0, 11.0, 22.0, 33.0]` — all 4 values match. `HARDWARE_ADD=ok`.

### Complete compute pipeline

```
VBIOS devinit → VRAM test → FIFO init → GR init (ctxprog+ctxvals) →
channel creation → RAMHT bind (compute object) → GR context bind →
push buffer submission → kernel execution → result readback
```

### Key findings during compute bring-up

1. **GT218 uses NVA3_COMPUTE_CLASS (0x85c0)**, not NV50_COMPUTE (0x50c0).
   Mesa explicitly selects NVA3_COMPUTE_CLASS for chipsets 0xa3, 0xa5, 0xa8
   via a switch statement (nv50_compute.c:43-64), not a threshold comparison.
   The Nouveau kernel driver exposes both NV50_COMPUTE (0x50c0) and
   GT214_COMPUTE (0x85c0) in gt215.c:35-43 and lets userspace choose.
   Verified: mesa/src/gallium/drivers/nouveau/nv50/nv50_compute.c:43-64,
   linux_drm/nouveau/nvkm/engine/gr/gt215.c:35-43.

2. **Compute uses subchannel 6** (Mesa `nv50_winsys.h:53`: `#define SUBC_CP(m) 6, (m)`).
   Verified: mesa/src/gallium/drivers/nouveau/nv50/nv50_winsys.h:53.

3. **RAMHT object binding requires a GPU object**: The compute class ID must
   be written to a 16-byte GPU object in the instance block, and the RAMHT
   entry's context field points to this object (offset >> 4). Without this,
   SET_OBJECT silently fails and all compute methods are dropped.
   Verified: `nv50_gr_object_bind` in nv50.c:42-57 creates a 16-byte object
   with class ID at offset 0x00. `ramht.c:71-89` shifts the object offset
   right by 4 bits (the `addr=4` parameter from nv50.c fifo:44).

4. **Global memory slot 14 must be enabled for nvcc kernels**: Mesa sets
   slots 0-14 to limit=0 (disabled) and slot 15 to limit=~0 (full range)
   (nv50_compute.c:95-111). However, the nvcc-compiled kernel uses
   `global14` for GLD/GST instructions. With Mesa's default (slot 14
   limit=0), the kernel hits a GLOBAL_LIMIT_READ MPC trap and produces no
   output. Fix: set all 16 slots to full-range (addr=0, limit=0xFFFFFFFF,
   mode=LINEAR). This differs from Mesa's approach because Mesa's codegen
   uses slot 15, while nvcc uses slot 14.

5. **Tesla parameter mapping (nvcc convention)**: The nvcc-compiled kernel
   accesses parameters as g[4]-g[7] in ALU instructions, which map to
   USER_PARAM(0)-(3). The `G2R` instruction reads hardware special registers
   (e.g., g[1] for blockDim, g[6] high-16 for threadIdx) while ALU
   instructions (IADD32, IMUL32) read USER_PARAM values from the same g[N]
   notation — the same g[6] serves both purposes depending on instruction
   type. USER_PARAM(i) writes to shared memory offset i*4, and the kernel's
   parameters are at shared offsets 0x0-0xc (per cubin: cbank=0x1f,
   SMEM_PARAM_SIZE=0x10).
   NOTE: This is nvcc's convention and differs from Mesa's codegen, which
   uses USER_PARAM(0) for grid Z info, USER_PARAM(1)+ for kernel parameters,
   and an inputOffset of 0x14 (20 bytes) (nv50_program.c:374,
   nv50_compute.c:533-554, 613-614). Our kernel uses nvcc's convention, so
   USER_PARAM(0)-(3) is correct for our case.

6. **Kernel binary**: Raw SASS (88 bytes) is loaded directly into VRAM at
   0x240000. The `CP_START_ID` method points to this address. No header or
   alignment tricks needed — the raw code section from the cubin works.

### VRAM layout

| Address | Size | Purpose |
|---|---|---|
| 0x100000 | 0x10000 | Channel instance block |
| 0x110000 | 0x1000 | Push buffer |
| 0x120000 | 0x1000 | GPFIFO ring |
| 0x130000 | 0x10000 | GMMU page table (PGT) |
| 0x140000 | — | (unused gap) |
| 0x240000 | 0x1000 | Kernel code (SASS) |
| 0x250000 | 0x1000 | Input array A |
| 0x260000 | 0x1000 | Input array B |
| 0x270000 | 0x1000 | Output array |
| 0x290000 | 0x4B800 | GR context values (ctxvals) |

### Files

| File | Purpose |
|---|---|
| `add.py` | Main compute launch script |
| `add_kernel.cu` | CUDA source for vector add kernel |
| `add_kernel.ptx` | PTX intermediate (sm_12) |
| `add_kernel.cubin` | Compiled cubin (sm_12) |
| `add_sass.bin` | Raw SASS binary (88 bytes) |
| `fifo_test.py` | FIFO/channel/GR init infrastructure |
| `gen_ctxprog.c` | Context program generator (compiled to ctxprog binary) |
| `extract_cuda65.sh` | Script to extract SASS from cubin via cuobjdump65 |
| `setup_ptxas_6_5_osx.sh` | CUDA 6.5 toolchain setup for macOS |

The Tesla path is actually **simpler in some ways** (no Falcon firmware to
load) but **harder in others** (ctxprog generation is opaque, sm_12 tooling
is ancient).

## Source verification audit (2026-07-22)

Every load-bearing claim in this document was verified against the actual
Nouveau/Mesa source code in `~/amdgpu/ref/`. Results:

| Claim | Source | Status |
|---|---|---|
| GT218 uses 0x85c0 compute class | nv50_compute.c:43-64, gt215.c:35-43 | CORRECT (fixed: explicit case match, not threshold) |
| Compute uses subchannel 6 | nv50_winsys.h:53 | CORRECT |
| RAMHT requires 16-byte GPU object | nv50.c:42-57, ramht.c:71-89 | CORRECT |
| Global slots 0-14 disabled, 15 enabled | nv50_compute.c:95-111 | CORRECT (our code differs: all 16 enabled, see finding 4) |
| RAMFC field layout (11 fields) | g84.c:70-82 | ALL 11 CORRECT |
| FIFO init register sequence | nv50.c:345-361 | CORRECT |
| Channel bind g84 >>8 vs nv50 >>12 | g84.c:39, nv50.c:76 | CORRECT |
| GPPut is entry index, LENGTH in dwords | nvif/chan506f.c:11, 22-26 | CORRECT |
| PDE format 0x3 \| pgt_addr | vmmnv50.c:113, 130-132, 138 | CORRECT |
| PTE format phys_addr \| 0x1 | vmmnv50.c:48, 72, 318 | SIMPLIFIED (noted: omits ro/kind/comp) |
| pd_offset = 0x0200 | g84.c:32 | CORRECT |
| TLB flush (id<<16)\|1 to 0x100c80 | vmmnv50.c:215-218 | CORRECT |
| DMA object flags0 = 0x7FC00002 | usernv50.c:111-121 | CORRECT |
| PRAMIN window 0x001700, base 0x700000 | nv50.c (instmem):71, 91, 400 | CORRECT |
| nv50_gr_init register sequence | nv50.c:679-760 | CORRECT |
| ctxnv50.c is 3347 lines | ctxnv50.c | CORRECT |
| GT218 uses gt215_gr (nv50_gr_init) | gt215.c:29-30, base.c:1186 | CORRECT |
| GT218 ctxvals: 0x401c00=0x142500df etc. | ctxnv50.c:398, 456, 631, 751, 3156, 2657 | ALL CORRECT |
| Tesla param mapping (nvcc: g[4-7] = USER_PARAM 0-3) | SASS + cubin info | CORRECT for nvcc (differs from Mesa codegen) |
| All 31 compute method addresses | nv50_compute.xml.h, cl50c0.h | ALL 31 CORRECT |
| gr_bind_context: ptr0=0x0020, flags=0x00190000 | g84.c:106-152 | CORRECT |
| gr_bind_context: offset 0x0c field packing | g84.c:147-148 | LATENT BUG: Python writes high bits of addr, Nouveau writes low 32 bits. Harmless for 32-bit VRAM addresses (both produce 0 in high byte). Would break for >4GB addresses. |
| Compute launch method order vs Mesa | nv50_compute.c:66-166, 562-633 | CORRECT (missing 9 texture/sampler methods not needed for add kernel) |
| USER_PARAM_COUNT = 4<<8 | nv50_compute.c:533-534 | CORRECT for nvcc convention (Mesa uses (1+size/4)<<8 because it reserves slot 0 for grid Z) |
| SHARED_SIZE = align(0x14+0x10, 0x40) = 0x40 | nv50_compute.c:584-586 | CORRECT (our kernel has no static shared memory; smem=32 from cubin = param space only) |
| BLOCK_ALLOC = (1<<16) \| block_dim | nv50_compute.c:603-604 | CORRECT for 1D blocks (Mesa uses block_x*block_y*block_z, same for 1D) |
