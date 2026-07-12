# Progress — GTX 770 / GK104 bring-up (`examples_kepler/add.py`)

Live status of getting a Kepler `sm_30` `out[i]=a[i]+b[i]` kernel running on the
GTX 770 (GK104) eGPU over USB4 via TinyGPU. Reference plan: `plan.md`.
Linux nouveau source is at `ref/linux/` (torvalds/linux 7.2-rc2). Kepler docs /
rnndb from `envytools` (fetched on demand). Agent goal: ask each DeepWiki ref
repo to solve the current blocker and continue debug/test/fix.

## Verified on real hardware (2026-07-11)

1. **TinyGPU transport works.** `--probe` reads `PMC_BOOT_0 = 0x0e4040a2` →
   chip_id **0x0e4 = GK104** (GTX 770). Real silicon, eGPU reachable.

2. **FALCON firmware loads and FECS RUNS but does NOT post "ready".**
   - `VER(0x40912c)=0x81103` (real), `imem[0]=0xf50e9b03` **exact match** to the
     expected first FUC opcode (the earlier byte-reversed read was a BE artifact).
   - `CPUCTL(0x409100)=0x10` ⇒ FECS is executing, but `0x409800 bit31` (ready)
     is never set, all mailboxes `0x0` ⇒ FECS reaches its init but cannot read
     GPC topology, so it never signals ready.
   - This is the **GPC clock/power domain being gated**, not a firmware/csdata bug.

3. **`PMC_ENABLE = 0xffffffff` (full enable)** required to make FECS/GR registers
   readable (without it FECS sits at the `0xbad0da1f` engine power-gated sentinel).

## Completed work

- **Transport** rewritten to match `examples/add.py` (`<BQQ`, `MAP_SYSMEM_FD` +
  `recvmsg` + `mmap`, coherent sysmem). Software backend + `--middle-selftest` green.
- **Firmware verified correct**: regenerated `gk104_{fecs,gpccs}_{code,data}.bin`
  from `fuc3.h`; sizes match existing bins.
- **`falcon_load`** matches nouveau FALCON v1 (WRITE=bit24, ITAG every 64 words).
- **`grctx_gk104.py`** parses `gf100_gr_init`/`gf100_gr_pack` and resolves the 5
  csdata packs → flat `(addr,count,pitch)` lists + `method_stream()`. Verified
  output (hub 50 / gpc_0 16 / gpc_1 11 / tpc 19 / ppc 6 words).
- **`falcon_csdata_write`** implemented and CALLED in `_init_hardware` for all 5
  packs BEFORE `falcon_start` (resolves the original "csdata not loaded" issue).
- Added standalone probe CLIs: `--probe-pgob`, `--probe-gpc-clock` (traced MMIO
  with before/written/after readback).
- Added `gk104_pmu_pgob` (fuse gate + 0x020004 un-gate + 0x10a78c handshake +
  War00C800_0 c800 workaround) and best-effort PMU load; **not** the blocker.

## Blocker investigation — what was RULED OUT (do not re-litigate)

Each tested on real HW with full MMIO traces:

- **PGOB power-gate — NOT the blocker.**
  - `nvkm_fuse_read(0x31c)` → `0x02141c = 0x00000000`, **bit0 = CLEAR** ⇒ nouveau
    *intentionally skips* `gk104_pmu_pgob` on this card.
  - `0x020004` already `0x41000000` → bits `[31:30]=01b` (bit30 = power **released**)
    **before any write**, and the requested value is retained.
  - All 4 `0xc800` workaround commands returned `ready=YES` ⇒ PMU/c800 block reachable
    (TinyGPU forwards those ranges fine).
- **GPC clock source mux — NOT the blocker.** `0x137160=0` is the **valid crystal
  reference** (not "no clock"; selector `00`=crystal, `10`=fixed 100 MHz). Forcing
  `0x137160` to fixed-100MHz(`0x2`), `RPLL_e800`(`0x3`), `RPLL_e820`(`0x103`),
  fixed-108MHz(`0x30000`) all *retained* the write but GPCs stayed `0xbadf3000`,
  `0x409604=0`.
- **RED_SWITCH (0x409614) — NOT sufficient alone.** Host write
  `0x409614=0x770` (POWER+ENABLE MAIN/GPC/ROP) retained, but GPCs still gated.
- **PTHERM GR clock-gate (0x20200) — NOT the blocker.** Forced GR `ENG_CLK=RUN`
  (low byte `0x54`, nouveau `clkgate_fini`) retained; no effect on GPC access.

## THE BLOCKER — GPC PLL block is unwritable → VBIOS devinit required

- **`0x137000` (GPC PLL control, `PCLOCK.CLK0_CTRL`) is NOT writable.** Every test
  write (`0x0`, `0x10`, `0x10011`) reads back `0xbadf3000`. The GPC PLL block is
  hard power/clock-gated at a level unreachable by RED_SWITCH, therm force, or any
  `0x137160` source selection.
- The GPC clock can ONLY come from this PLL (DIV-mode VCO sources also route through
  the gated block), so the GPC domain cannot be clocked by host register writes.
- **This is exactly what the VBIOS devinit init-scripts do at POST** (release the
  GPC/PCLOCK power-domain isolation and program the PLL). Our un-POSTed eGPU never
  ran them, so the GPC PLL block is permanently gated until we run them.

### VBIOS availability vs. sandbox limitation
- Online: TechPowerUp VGA BIOS Database has GTX 770 (GK104, `10DE:1184`) dumps —
  NVIDIA ref `219286` (GK104 P2004), MSI `160365`, Asus `140907`, etc.
- **This sandbox cannot download it**: TechPowerUp serves a bot-wall HTML page
  for every attempt (curl UA / Referer / full browser headers, CDN guesses,
  `/download` variants — all return HTML, not `.rom`). `gh` CLI absent.
- **User must supply the `.rom`**: GPU-Z dump on Windows, the card's PCI Expansion
  ROM (if TinyGPU is extended to expose the ROM BAR — currently BAR0 MMIO only), or
  a non-botwalled mirror dropped into `firmware/gk104/vbios/`.

## Next steps (in order)

1. **VBIOS obtained and validated.** `Palit.GTX770.4096.131216.rom` is 169,984
   bytes, SHA-256 `d574f587406c107e963c702a7980c4773226d4815d86afc77f9a60d96df7d02d`,
   and contains two `10DE:1184` PCIR images plus a 19-entry BIT directory.
   `python3 examples_kepler/add.py --vbios-info` performs this validation.
   The dump is an `NVGI` multi-image container: the matching images begin at
   file offsets `0x600` and `0xfc00`; the inspector now reports both PCIR and
   image-relative BIT coordinates so the executor does not accidentally use
   container offsets as MMIO script pointers.
2. **Parse it** with envytools `nvbios` (or port nouveau `nvkm/subdev/bios/`):
   - devinit init-script bytecode → GR/GPC/PCLOCK domain-release ops;
   - GPC PLL (`0x137000`) limits + `refclk` via `nvbios_pll_parse`.
3. **Port a minimal devinit executor** that runs ONLY the GR/GPC/PCLOCK release ops
   (board-independent register writes; skip GPIO/voltage/fan ops).
4. After devinit, `0x137000` should become writable → program the GPC PLL from the
   parsed limits for a safe low GPC clock → GPC domain wakes (`0x409604 != 0`, GPC
   falcons un-gated).
5. Resume FECS bring-up: FECS should now post ready (`0x409800 bit31`) + valid
   `0x409804` size.
6. GR golden-context buffer (`gk104_grctx_generate_*`); FIFO/RAMIN/USERD/GPFIFO
   channel; wire launch; run `out[i]=a[i]+b[i]` on silicon.

## Key register facts (updated)

- FECS `0x409000`, GPCCS `0x41a000`, per-GPC CTXCTL `0x502000` (`PGRAPH.TP[0]`),
  GPCCS RED_SWITCH `0x41a614`. FALCON v1 load: CODE_INDEX `0x180`/DATA_INDEX
  `0x1c0` = `start|(1<<24)` WRITE; CODE `0x184`, TAG `0x188`/64w, DATA `0x1c4`;
  bootvec `0x104`; CPUCTL `0x100` START=`0x2`. csdata iface `0x1c0`/`0x1c4`,
  method word `(xfer<<26)|addr`.
- `PMC_ENABLE` = **0xffffffff** (full) required.
- **`RED_SWITCH` `0x409614`** (gf100_gr_fecs_reset): `POWER_MAIN=0x10,
  POWER_GPC=0x20, POWER_ROP=0x40, ENABLE_MAIN=0x100, ENABLE_GPC=0x200,
  ENABLE_ROP=0x400, PAUSE_GPC=0x2, PAUSE_MAIN=0x1`. Host writes `0x70` then enables
  `0x700` → releases GR/GPC/ROP power+clock. Retained but insufficient alone.
- **`0x020004`** GPC/ROP power-gate: `[31:30]=01b` (bit30) = released. Already set
  on this card; fuse `0x02141c` bit0 must be SET for nouveau to run PGOB (it is
  CLEAR here, so nouveau skips PGOB).
- **`0x137000` `PCLOCK.CLK0_CTRL` / GPC PLL** — `gf100_pll_ctrl`: ENABLE=bit0,
  PWROFF=bit1, PLL_PWR=bit16, PLL_LOCK=bit17. **Currently unwritable (0xbadf3000)**.
- **`0x137100` `SRC_SEL`** bit0 = CLK0 source sel (0=DIV, 1=PLL→0x137000).
- **`0x137160` `CLK0_DIV_SRC`** `gf100_div_src`: SRC[3:0] (0=SRC0,2=100MHz,3=SRC3),
  VCO[8] (0=RPLL_e800,1=RPLL_e820), SRC0[17:16] (0/3=27/108MHz).
- **`0x20200`** PTHERM GR clock-gate (ENG offset `0x00`): low byte `0x45`=AUTO
  (clkgate_enable), `0x54`=RUN (clkgate_fini). Forcing RUN had no effect.
- `0xbadf3000` = power/clock-gated sentinel (returns for any gated engine/block).
- VCO PLLs 0/1 (`0x00e800`/`0x00e820`) ARE programmed and running (valid coef),
  but the GPC clock does not route from them without the GPC PLL block alive.

### Current test result (2026-07-11)

- Added and validated the Palit ROM loader, BIT-I init-table discovery, GK104 PLL
  limits parsing, direct NVINIT writes, group writes, AND/OR/RESET decoding, and
  condition-table discovery in `examples_kepler/add.py`.
- Offline self-test remains green.
- Live `--probe-vbios-devinit` executes the ROM's direct power/clock writes, but
  `0x137000` still reads `0xbadf3001`, `0x409604` remains zero, and the PLL
  cannot be programmed. The remaining blocker is a real NVINIT control-flow
  executor (conditions/subscripts/resume), not ROM availability.

### Current test result (2026-07-12)

- Implemented a complete `nvbios_init.py` NVINIT executor (SUB_DIRECT/SUB/JUMP,
  control-flow, PLL, IO, MACRO, CONDITION/TIME/REPEAT, etc.) and wired it into
  `add.py`.
- Live `--probe-vbios-devinit` now executes all VBIOS init scripts and the GPC PLL
  becomes programmable: `0x137000` locks, `gpc-pll: ... lock=YES`.
- After VBIOS devinit and PGOB, `0x409604=0x40004`, `0x41a100=0x10`, `0x502100=0x10`
  (GPC/GPCCS/per-GPC FALCONs are now out of power-gate, not `0xbadf1200`).
- **Next blocker:** FECS does not post `0x409800 bit31` ready. After `falcon_start`,
  `0x409100=0x10` (FECS STOPPED/halted), `0x409400=0x300`, `0x409724=0x0`, `0x409728=0x0`,
  `0x409730=0x0`, and all `0x40980x` mailboxes are `0x0`, `0x409c18=0x0` (no ISR). The
  FALCON CPUCTL `0x10` bit does not clear and the MMIO_CTRL never becomes pending, so
  `hub.fuc` is not actually executing. The `0x2` START_TRIGGER to `0x409100` is not
  taking effect from a `0x10` STOPPED state, even though the `falcon_load` sequence
  (0x180/0x184/0x188, 0x1c0/0x1c4) appears correct and `imem[0]` matches.
  - Tried `0x400080-0x400148` (GK104 `main_0` init list) + `0x400500` PGRAPH master enable
    before `0x260=0`/`1`; no effect. `0x409100` still `0x10` and `0x409728` still `0x0`.
  - Tried `0x409100 = 0x0` and `0x12`/`0x112` start values; no effect.
  - Tried `0x409100` reset bits `0x1`/`0x4` and `0x40907c` SUBENGINE_RESET; `0x40907c=1`
    gates `FECS` to `0xbadf5000` and is not the correct path.
  - This is the hard blocker: the FALCON CPUCTL START_TRIGGER is not accepted.

### Current test result (2026-07-12) – fixed

- Root cause: `firmware/gk104/*.bin` were big-endian word dumps of the `.fuc3.h`
  arrays, but `falcon_write_imem/dmem` loads them as little-endian `u32` words.
  This byte-swapped every instruction; `FECS` decoded garbage and branched to
  unmapped virtual page `0xd8` (`UC_PC=0xd804`, `VTLB` no-hit).
- Re-extracted the binaries with `firmware/gk104/extract_fw.py` (`<I`) so the
  first FECS word is `0x039b0ef5` (a branch to the real entry) and data words
  are `0x00000300`, etc.
- Updated `add.py` `imem[0]` expectation and added a `_dump_fecs_tlb()` helper
  in the `TimeoutError` path for future diagnostics.
- `--probe-falcon` now succeeds: `FECS+GPCCS loaded and started OK`,
  `UC_PC=0x567`, `UC_CTRL=0x20` (running), `0x409800 bit31` set.

### Current test result (2026-07-12) – FIFO descriptor correction

- Read `ref/linux/.../engine/fifo/gk104.c`: RAMFC uses a push-buffer base/limit
  at `0x48/0x4c`, while the GPFIFO is a 16-byte descriptor ring and USERD
  `GP_PUT` is an entry index. Fixed `submit_launch()` to create that layout
  instead of placing method dwords directly in the ring and writing a byte
  address to `GP_PUT`.
- `--middle-selftest`, software demo, Falcon probe, and VBIOS inspectors pass.
- Full hardware `python3 examples_kepler/add.py` now reaches channel submission
  and times out with semaphore `0` (no crash). The remaining hardware blocker is
  GR context/channel execution: the current RAMIN still supplies a zero-filled
  placeholder instead of nouveau's generated GK104 context, so the compute
  engine has not yet been proven to execute the push buffer.

### Current test result (2026-07-12) – Kepler FIFO layout and runlist

- DeepWiki search did not expose pages for the requested repositories; the
  available NVIDIA/GK20A DeepWiki-adjacent source confirms GP entries use 8
  bytes and encode push-buffer length at bit 10. Local Nouveau `gk104.c` and
  `gf100.c` were authoritative for the remaining details.
- Corrected the earlier 16-byte/newer-GPU descriptor: GK104 now uses 8-byte GP
  entries, RAMFC points to the GPFIFO ring, and a real `(chid, 0)` runlist entry
  is committed through `0x2270/0x2274`.
- Live timeout diagnostics show `GP_GET=0`, PFIFO channel remains unscheduled,
  and no semaphore update. This narrows the next issue to runlist memory target/
  address or the missing GR context bind, rather than push-buffer encoding.
- TOP-table decoding confirms the GR engine is assigned to runlist 0. Added the
  Nouveau `gk104_fifo_init()` PFIFO BAR1/interrupt initialization before the
  runlist commit; the live result is unchanged (`GP_GET=0`). The sysmem-only
  BAR1/userd assumption is now the leading scheduling blocker.
- Corrected initialization ordering so PFIFO setup happens before channel bind
  and start. Hardware still clears the start state and leaves `GP_GET=0`, while
  `PFIFO_BAR1=0x10030000` and the instance bind register retain their values.
  This confirms the remaining failure is in the BAR1 aperture / PBDMA access
  path, not a late FIFO reset.
- Replaced the fake sysmem USERD with the actual PCI BAR1 address returned by
  TinyGPU (`bar_info(1)`), programmed PFIFO BAR1 from that address, and wrote
  `GP_PUT` through BAR1. The channel still does not advance; BAR1 reads return
  nonzero GPU contents but `GP_GET` does not become 1. This is the first test
  that exercises the real Nouveau USERD aperture rather than a sysmem alias.
- Added an immediate BAR1 readback: `GP_PUT` writes `1` and reads back `1`, with
  BAR1 address `0x110000000` and size `0x8000000`. Therefore the remaining
  failure is after USERD visibility—runlist/PBDMA scheduling or channel
  instance/context validation—not a transport write failure.
- Filled all remaining `gk104_chan_ramfc_write()` fields, then corrected the
  address domains using `gk104_ectx_bind()`: GR context, GPFIFO, and push-buffer
  pointers now use mapped GPU VAs (GR context `VA|4`), while the runlist commit
  retains the physical bus address. The live result is still `GP_GET=0`; the
  channel instance is now structurally much closer to Nouveau.
- Also applied `gk104_fifo_init_pbdmas()` essentials (`0x204` PBDMA enable and
  scheduler error-disable release at `0x2a04`). No change: BAR1 `GP_PUT` remains
  writable, but PBDMA never consumes the runlist entry.
- Compared the complete GK104 FIFO init sequence; no additional `0x2200` or
  `0x2628` write belongs to `gk104_fifo_init()` (those are GF100-only). The
  current failure is therefore not explained by omitted generic FIFO enables;
  next diagnostics must focus on channel instance validation/context-switch
  fault state.
- Added the previously missing `gf100_runq_init()`/`gk104_runq_init()` sequence
  for all three PBDMAs and mapped each PBDMA to runlist 0 at `0x2390`. The live
  channel still does not advance `GP_GET`; the next evidence needed is the
  per-PBDMA HCE/interrupt status and context-valid fault state.
- Corrected a diagnostic/initialization address typo: Nouveau PBDMA registers
  are based at `0x040000`, not `0x004000`. With the real registers initialized,
  PBDMA0 reports HCE `CTXNOTVALID` (`0x040108 bit31`) and channel 0, proving the
  runlist is now reaching the PBDMA. The remaining blocker is the missing valid
  GK104 GR golden context image; the zero-filled placeholder is rejected during
  context switch.
- Matched `ctxgf100.c`'s context pointer convention (`ctx + 0x80000 | 4`) and
  initialized its header words at `CB_RESERVED+0x1c` (`1,0,0,0`). The HCE fault
  remains, so a complete generated `gf100_grctx_generate_main()` image and its
  required pagepool/bundle/attrib buffers are still needed.
- Compared allocation domains: Nouveau uses `NVKM_MEM_TARGET_INST` for the
  golden context. Tested a BAR1-backed 1 MiB context region with the same
  header; HCE `CTXNOTVALID` remains. This rules out the simple sysmem-versus-
  instance-memory placement mismatch; the complete generated image and engine
  context resources are still required.
- DeepWiki's available NVIDIA command-processing page independently confirms
  that RAMFC/PBDMA state is separate from the engine context image and that
  context switching is a distinct stage after runlist selection. The local
  Nouveau source remains the GK104-specific authority; no requested mirror
  repository had a deeper indexed GK104 context page.
- Live FECS reports the GK104 golden-context image size as `0x2d000`; Nouveau
  allocates `CB_RESERVED + gr->size` and then fills it through
  `gf100_grctx_generate_main()`. The current 1 MiB allocation is large enough,
  but contains no generated image, confirming that image generation—not buffer
  capacity—is the active missing implementation.
- After correcting the PBDMA base and re-testing with the context pointer as a
  VMM GPU VA (`gr_ctx.va_addr + 0x80000 | 4`), PBDMA0 HCE status is now clear
  (`0x040108=0`) rather than `CTXNOTVALID`. Context switching therefore passes;
  the remaining failure is post-switch GPFIFO/USERD fetch (`GP_GET` stays 0).
- Added PBDMA GP state diagnostics. The HCE fault is clear, but the live
  PBDMA GP registers do not show a consumed entry; the next fix is to reconcile
  the RAMFC GPFIFO base/limit encoding with the GK104 PBDMA `GP_BASE` register
  after context load.
- Read `nouveau_channel_new()` in the Linux source: Kepler channels use a VMM,
  `args.offset = ioffset + chan->push.addr`, and `args.length = 0x2000`; the
  RAMFC limit encoding (`ilog2(length/8)=10`) matches the current script. This
  confirms the remaining GP fetch issue is the exact push/GPFIFO allocation
  address relationship, not the limit exponent.
- Decisive fix from `ref/linux/drivers/gpu/drm/nouveau/nvif/chan506f.c`:
  Kepler USERD uses `GP_GET=0x88` and host `GP_PUT=0x8c` (the script had them
  reversed). After correcting this, live hardware reports `userd_gp_get=1`,
  proving the GPFIFO descriptor is consumed. The remaining failure is now in
  pushbuffer method execution/semaphore completion, after FIFO fetch.
- Seeded the initial semaphore to `1`; the live stream now passes the wait and
  stalls after the compute launch with `last=1`. Read `gk104_compute.xml` and
  fixed CWD `CB_CONFIG_1.SIZE` from `<<8` to `<<15`, and changed hardware code
  upload to extract `.text.E_4` from the cubin instead of uploading the ELF.
  The fallback structural cubin is still not executable SASS when Docker/CUDA
  is unavailable, so the remaining live failure may now be the placeholder
  kernel itself rather than FIFO infrastructure.
- Workspace search found no verified real `sm_30` `E_4` cubin. The local
  `nvcc` is a Docker shim and Docker is unavailable; the only other cubin is a
  32-bit Multi2Sim sample for a different kernel/format.
- Fetched the requested `xiuxiazhang/KeplerAs` reference and assembled a valid
  `sm_35` control kernel (`S2R; EXIT;`) from the existing `E_4` cubin template.
  It also stalls after launch with the done semaphore at `1`, proving the
  fallback SASS is not the sole cause; the zero/un-generated GR context or
  launch-state setup still blocks even an immediate-exit kernel.
- Tested the Nouveau `gf100_gr_fecs_start_ctxsw`/bind/golden-save mailbox
  sequence. On this direct bring-up path the resume command leaves
  `0x409804=0xffffffff` and never returns status `1`, so it was removed from
  the live path rather than leaving a new hard timeout.
- Forced the uploaded `.text` allocation to a 256-byte VA alignment and
  retested the KeplerAs immediate-exit kernel; it still stalls with semaphore
  `1`. Code alignment is therefore not the remaining launch blocker.
- Replaced the incorrect G80-style 3-level page-table builder with the
  GK104/GF100 2-level format from `ref/linux/.../vmmgf100.c` and
  `vmmgk104.c`: 13-bit PGD, 15-bit small-page index, PTE address `paddr>>8`,
  system-coherent/non-coherent aperture bits, and the VMM join record at
  RAMIN `0x200`. Offline GMMU self-tests still pass.
- Replaced the semaphore stream's invalid NVC56F `0x005c` packet with the
  NV906F incrementing methods `0x10/0x14/0x18/0x1c`, matching Nouveau's
  `nvc0_fence.c`/`nvif/chan906f.c`. The hardware stream still needs a valid
  FIFO dispatch before this can be judged by the completion value.
- Added an opt-in `KEPLER_VRAM_INST=1` probe that places the channel instance
  and GR context in BAR1-backed instance memory, matching Nouveau's
  `NVKM_MEM_TARGET_INST`. This clears the earlier PBDMA signature error and
  confirms the instance-memory target mattered; the remaining live error is
  `PBDMA0 INTR=0x00004000` (`GPPTR`) with USERD `GP_GET=0`.
- Matched the GK104 GPFIFO allocation to RAMFC `LIMIT2=10` (`0x2000` bytes),
  reserved VA zero so absolute ring alignment is honored, reset reused BAR1
  USERD state, and used masked channel start/stop writes. These fixes remove
  stale signature/HCE errors, but the current BAR1-instance probe still does
  not fetch the first GPFIFO entry. Offline gates remain green:
  `kepler_selftest=ok`, `software_demo=ok N=256 launch_words=24`.
- Tested the alternate GPFIFO external-entry `BIT(9)` encoding after the
  aligned-ring/reset fixes; it produced the same `GP_GET=0`/`GPPTR` result.
  The remaining issue is therefore not the main-vs-external flag alone.
- Mirrored Nouveau's main-push VA relationship (`push_va` and
  `gpfifo_va = push_va + 0x10000`) with a `0x10000`-aligned push allocation;
  the BAR1-instance probe still reports `GPPTR` and `GP_GET=0`. The unresolved
  issue is now in the channel's live GPFIFO pointer/context wiring rather than
  ring size, VA alignment, entry flag, USERD reset, or instance placement.
- Re-read the exact GK104 paths in `gk104.c`, `gf100.c`, `vmmgf100.c`, and
  `chan506f.c`. Added an opt-in GK104 PFIFO reset using MC FIFO bit `0x100`,
  corrected channel runlist selection bits `0xf0000`, cleared runlist fault
  `0x262c`, unblocked runlist 0 at `0x2630`, and changed `0x2a04` to the exact
  Nouveau mask/value `0xbfffffff`. Also wait for the asynchronous GK104
  runlist pending bit at `0x2284` before writing USERD `GP_PUT`.
- The FIFO reset test proved the stale-cache hypothesis: PBDMA `CTRL_ADDR_LOW`
  changed from an old BAR1 address to the current USERD address. The old
  address path raised `GPPTR`; the source-shaped physical USERD path removes
  `GPPTR` and loads `GP_PUT=1`, but `GP_GET` still remains 0.
- Corrected the USERD domain in the VRAM-instance probe: Nouveau RAMFC `0x08`
  contains the USERD memory object's VRAM address, while PFIFO `0x2254` is the
  BAR1 GPU-virtual mapping base. A diagnostic BAR1-VMM replacement was tested
  and reverted because it invalidated TinyGPU's existing BAR1 mapping
  (`0xbad0...` reads); the remaining blocker is to use that existing mapping
  rather than overwrite it.
- DeepWiki search was attempted for the requested repositories. The available
  NVIDIA command-processing page confirms the same PBDMA roles (`GP_PUT` q+0,
  `GP_GET` q+0x14, `GP_BASE` q+0x48, channel q+0x08); no indexed page exposed a
  deeper GK104-specific implementation than the local Nouveau source.
- Current verified gates remain green: `py_compile`, `--middle-selftest`, and
  `NV_BACKEND=software`. Hardware semaphore-only probes are still blocked at
  `GP_GET=0`, so `hardware_demo=ok` has not yet been claimed.
- Read-only inspection of TinyGPU `0x1704` found its live BAR1 instance at
  VRAM `0x102000`. Earlier probes had allocated channel RAMIN at that exact
  address and overwrote the BAR1 instance; the allocator now starts at
  `0x400000` to preserve this reserved object. The current attached GPU still
  reports `0xbad0...` from the already-corrupted BAR1 mapping, so another live
  result requires a real device/function reset before it can validate the new
  reservation.
