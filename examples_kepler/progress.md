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

## Session 2026-07-12 — FECS golden context: wrong command interface (cmd 3/9 vs cmd 1/2)

### Root cause identified

The `bind_pointer` timeout and `CTXNOTVALID` errors were caused by using the
**wrong FECS command interface**. The add.py code was sending:

- Command 3 (`bind_pointer`) — write `0x409500=inst_tag`, `0x409504=0x3`, poll
  `0x409800` bit4
- Command 9 (`golden_save`) — write `0x409500=inst_tag`, `0x409504=0x9`, poll
  `0x409800` bit0

These two commands belong to Nouveau's **secure-boot firmware path**
(`gr->firmware` / `gf100_gr_init_ctxctl_ext`), implemented by
`gf100_gr_fecs_bind_pointer()` and `gf100_gr_fecs_wfi_golden_save()` in
`ref/linux/.../engine/gr/gf100.c` lines 834-850.

However, the FECS firmware loaded by add.py is the **internal Nouveau
firmware** (`hubgk104.fuc3`), which corresponds to the `!gr->firmware` /
`gf100_gr_init_ctxctl_int` path. Reading the actual FECS microcode source
(`ref/linux/.../engine/gr/fuc/hub.fuc`) confirmed that the firmware's main
loop only handles three command IDs:

- **Command 0x0001** (`ctx_chan`): Set current channel — hub.fuc lines 278-283
- **Command 0x0002** (`ctx_save`): Save context — hub.fuc lines 286-294
- **Command 0x4001**: GPU-triggered context switch (internal, from CHSW IRQ)

Any other command ID hits `E_BAD_COMMAND` at hub.fuc lines 296-299, which
sets SCRATCH(5) to the error code and raises INTR_UP. This is exactly why the
`bind_pointer` poll timed out: the firmware rejected command 3 as a bad
command and never set the completion bit.

The diagnostic prints also showed `IREN=0x0` because the register addresses
were wrong:
- `0x409210` was being read as `INTR_EN_SET` — the correct address is
  `0x409010` (confirmed in `fuc/macros.fuc` line 49)
- `0x409208` was being read as `INTR_UP` — the correct address for interrupt
  status is `0x409008` (`NV_PGRAPH_FECS_INTR`, macros.fuc line 41)
- `0x409128` was labeled `UC_PC` — it is actually `CPUSTAT`
  (`NV_PGRAPH_FECS_CPUSTAT`); the real PC is not directly exposed as a single
  MMIO register on this falcon version

### Fix applied

Replaced the bind_pointer/golden_save block (add.py lines ~2300-2343) with
the correct `!gr->firmware` (`_int`) path from `ctxgf100.c` lines 1505-1535:

1. **Make channel current** (`ctx_chan`, FIFO cmd 1):
   - Clear `CC_SCRATCH(0)` bit31 via `CC_SCRATCH_CLR(0)` at `0x409840`
   - Write `0x409500 = inst_tag` (FIFO data = instance tag)
   - Write `0x409504 = 0x00000001` (FIFO cmd = 1 = set channel)
   - Poll `0x409800` for bit31 (FECS sets this in `main_done` after
     `ctx_chan` returns)

2. **Trigger context unload** (golden save via CHSW):
   - Clear `CHAN_NEXT` bit31 at `0x409b04` (no next channel to load)
   - Write `0x409000 = 0x00000100` (IRQMSET bit8 = INTR_CHSW) to trigger a
     context-switch interrupt; the FECS ISR enqueues cmd 0x4001, and the
     main loop's ctx_switch handler saves the current context because
     CHAN_NEXT bit31 is clear (the `chsw_prev_no_next` branch at hub.fuc
     lines 254-262)
   - Poll `0x409b00` for bit31 to clear (context switch complete)

3. **Clear CHAN_ADDR** bit31 at `0x409b00` so FECS returns to idle.

The `inst_tag` calculation (`0x80000000 | ((base + ramin_pa) >> 12)`) was
already correct and is unchanged.

Also fixed the diagnostic register labels in the FECS trace section
(add.py line ~872):
- `UC_STATUS` → `CPUSTAT` for `0x409128`
- Added `INTR_EN` (`0x409010`) to the pre-start dump
- Removed the incorrect `0x409210` / `0x409208` reads from the golden-context
  diagnostic block

### Key references

- FECS firmware main loop: `ref/linux/.../engine/gr/fuc/hub.fuc` lines 218-306
- FECS register definitions: `ref/linux/.../engine/gr/fuc/macros.fuc` lines 40-131
- `_int` path golden context: `ref/linux/.../engine/gr/ctxgf100.c` lines 1505-1535
- `_ext` path (secure boot, NOT applicable): `ref/linux/.../engine/gr/gf100.c`
  lines 834-850
- `gf100_gr_init_ctxctl` dispatch: `ref/linux/.../engine/gr/gf100.c` lines 1883-1893

### Verification

- `py_compile` passes on the edited add.py.
- The fix has NOT yet been tested on live hardware (requires the eGPU to be
  re-attached after the previous session's BAR1 VMM crash). The next live run
  should show whether `FECS ctx_chan (cmd 1)` completes (CC_SCRATCH0 bit31
  sets) and whether the subsequent context unload saves the golden image
  (CHAN_ADDR bit31 clears).

### Remaining blocker

The BAR1 VMM configuration issue (semaphore does not reach 2,
`userd_gp_get=0xffffffff`) is still unresolved. The PDE format for 4KB pages
was fixed in the previous session (SPT fields instead of LPT), but the BAR1
VMM setup function itself was reverted after causing a system crash. The
next step is to carefully re-evaluate the BAR1 VMM setup, ensuring it does
not overwrite TinyGPU's existing BAR1 mapping.

## Session: grctx->main() implementation (2026-01-23)

### Problem

The FECS golden context save was saving an **empty/uninitialized context**
because `grctx->main()` was never called. In Nouveau (ctxgf100.c line 1515),
`grctx->main()` is called between making the channel current (cmd 1) and
triggering the context unload. It writes:
1. GR register init lists (hub/gpc_0/zcull/gpc_1/tpc/ppc) via `gf100_gr_mmio()`
2. Pagepool, bundle_cb, attrib_cb addresses via `gf100_grctx_patch_wr32()`
3. Attrib configuration (alpha/beta tables)
4. icmd bundles via `gf100_gr_icmd()` (0x400200/0x400204 interface)
5. mthd bundles via `gf100_gr_mthd()` (0x404488/0x40448c interface)
6. Various grctx fixups (unkn, gpc_tpc_nr, r419f78, r419cb8)

Without these, the GR engine has no valid context to load when the channel
is started, which is why the FIFO/channel launch was failing.

### Fixes applied

1. **Removed _ext-path ctx_header writes** (add.py): The writes to
   `data[0x1c/0x20/0x28/0x2c]` were from the secure-boot `_ext` path
   (ctxgf100.c lines 1499-1504). The `_int` path does NOT write them.
   The context buffer starts zeroed, and grctx->main() populates the GR
   registers before the context unload saves the golden image.

2. **Added VMM limit at inst 0x0208** (add.py): Nouveau's
   `gf100_vmm_join_()` writes `vmm->limit - 1` at inst offset 0x0208.
   For GK104, the VMM limit is 40-bit (1TB), so the value is `(1<<40) - 1`.
   Without this, the channel VMM has no limit configured.

3. **Extended grctx_gk104.py** to extract the `data` field from init entries:
   - The `gf100_gr_init` struct is `{u32 addr, u8 count, u32 pitch, u64 data}`.
   - The parser now extracts all 4 fields (was only extracting 3).
   - Added `mmio_writes()` function to expand entries into (addr, value) pairs.
   - Added `mmio_pack()`, `icmd_pack()`, `mthd_pack()` helper functions.
   - Added `MMIO_PACKS`, `ICMD_PACKS`, `MTHD_PACKS`, `GK104_GRCTX_CONSTS`.

4. **Implemented `_gk104_grctx_main()`** (add.py): Replicates
   `gf100_grctx_generate_main()` for GK104:
   - Writes mmio packs (hub, gpc_0, zcull, gpc_1, tpc, ppc) via direct MMIO
   - Saves and zeros idle timeout (0x404154)
   - Writes pagepool address (0x40800c/0x408010/0x419004/0x419008/0x4064cc)
   - Writes bundle address (0x408004/0x408008/0x418808/0x41880c/0x4064c8)
   - Writes attrib_cb address (0x418810/0x419848)
   - Writes attrib config (0x405830/0x4064c4)
   - Writes unkn fixups (0x418c6c/0x41980c/0x41be08/0x4064c0/0x405800/0x419c00)
   - Writes gpc_tpc_nr (0x405b00)
   - Writes r419f78 fixup
   - Writes icmd bundles via 0x400200/0x400204 interface
   - Restores idle timeout
   - Writes mthd bundles via 0x404488/0x40448c interface
   - Writes r419cb8 fixup

5. **Allocated pagepool, bundle_cb, attrib_cb** (add.py):
   - pagepool: 0x8000 bytes (gk104_grctx.pagepool_size)
   - bundle_cb: 0x3000 bytes (gk104_grctx.bundle_size)
   - attrib_cb: 0x100000 bytes (safe default for 8 TPCs)

6. **Called `_gk104_grctx_main()`** between ctx_chan (cmd 1) and context
   unload, matching Nouveau's sequence. GPC/TPC topology is read from
   0x409604 and GPC_UNIT(i, 0x2608).

7. **Added BAR1 VMM diagnostics** (add.py): Enhanced the existing BAR1
   diagnostic to decode the PDB format (target/VOL/address), read the VMM
   limit, and dump the first few PGD entries to see what's mapped.

### Key references

- `gf100_grctx_generate_main()`: ctxgf100.c lines 1342-1431
- `gf100_gr_mmio()`: gf100.c lines 1079-1093
- `gf100_gr_icmd()`: gf100.c lines 1096-1133
- `gf100_gr_mthd()`: gf100.c lines 1136-1158
- `gf100_grctx_patch_wr32()`: ctxgf100.c lines 994-1004
- `gf100_grctx_generate_pagepool()`: ctxgf100.c lines 1023-1029
- `gf100_grctx_generate_bundle()`: ctxgf100.c lines 1014-1020
- `gf100_grctx_generate_attrib_cb()`: ctxgf100.c lines 1053-1057
- `gf117_grctx_generate_attrib()`: ctxgf117.c lines 244-275
- `gk104_grctx_generate_unkn()`: ctxgk104.c lines 894-903
- `gk104_grctx_generate_gpc_tpc_nr()`: ctxgk104.c lines 915-919
- GK104 grctx function table: ctxgk104.c lines 970-992
- `gf100_vmm_join_()`: vmmgf100.c lines 342-363
- BAR1 control (0x1704): gf100.c (bar) lines 52-58
- USERD BAR1 window (0x2254): gk104.c (fifo) lines 744-753

### Verification

- `py_compile` passes on add.py and grctx_gk104.py.
- `--middle-selftest` passes.
- `NV_BACKEND=software` demo passes.
- NOT yet tested on live hardware.

### Remaining blockers

1. **FECS ctx_chan DMA hangs — VMM join format for FECS VM DMA** (CURRENT BLOCKER):
   - FECS firmware successfully loads, starts, and posts ready (0x409800 bit31).
   - FECS starts all 4 GPC falcons (GPC0-3 CTRL=0x20, SCRATCH0=0x80000000).
   - ctx_chan (cmd 1) is sent with inst_tag. FECS begins ctx_load but hangs at
     PC 0x1097a (after `xdwait`) waiting for a VM DMA to complete.
   - MEM_BASE=0x00080800 (correct: GR ctx VA 0x08080000 >> 8).
   - MEM_TARGET=0x00000001 (VM mode — FECS walks the channel VMM page tables).
   - The VMM page directory is stored in the inst block at offset 0x200.
   - **Root cause**: The FECS VM DMA engine walks page tables that must be in
     VRAM (BAR1-backed), but the software VMM's page tables are in host sysmem.
     A BAR1-backed VMM was created (PGD at 0x503000, SPT at 0x513000) with
     PDE/PTE entries mapping the GR context VA range to grctx_vram_pa.
     PTE format: bit0=PRESENT, bits[4:31]=addr>>12, bits[33:34]=TARGET
     (0=VRAM, 2=SYSRAM). PDE format: bit0=LPT_PRESENT, bit32=SPT_PRESENT,
     bits[4:31]=LPT_ADDR>>12, bits[36:63]=SPT_ADDR>>12.
   - **Still hangs after BAR1 VMM**: The VMM join value format may be wrong.
     Two formats tried:
     (a) `pgd = vmm_pgd_pa` (full address, per gf100_vmm_join_)
     (b) `pgd = ((vmm_pgd_pa >> 12) << 4)` (PDB register format, per
         gf100_vmm_invalidate)
   - **Next**: Verify the exact VMM join format by checking what nouveau
     actually writes to inst[0x200] on hardware, or try using the existing
     software VMM's page tables but copied to BAR1 VRAM (preserving the
     PDE/PTE encoding that the software VMM already uses). Also check
     whether the PDE needs bit32 (SPT_PRESENT) set, and whether the PTE
     needs the PRIV/VOL bits.

2. **BAR1 VMM**: The PBDMA reads USERD through the BAR1 window (0x2254),
   which goes through the BAR1 VMM. If TinyGPU's BAR1 VMM doesn't map the
   USERD VA, the read returns 0xffffffff. The enhanced diagnostics will
   show what TinyGPU's BAR1 VMM maps, so we can determine if USERD's VA
   is mapped or if we need to add a mapping.

3. **grctx->main() on hardware**: The icmd/mthd bundles write to GR
   registers via indirect interfaces (0x400200/0x404488). These need to
   be tested on hardware to verify they work correctly.

## Session 2026-07-12 (continued) — FECS DMEM auto-power-gating FIXED

### Major breakthrough: FECS firmware now runs and posts ready

The FECS DMEM auto-power-gating issue was diagnosed and fixed. The FECS
firmware now successfully loads, starts, posts ready (0x409800 bit31),
and begins processing ctx_chan commands.

### What was fixed

1. **PGRAPH disable (0x400500)**: Changed from `write32(0x400500, 0)` to
   `write32(0x400500, read32(0x400500) & ~0x00010001)` (masked write, only
   clearing bits 0+16, matching nouveau's `nvkm_mask`). Writing 0 to the
   entire register power-gates the FECS, making DMEM return 0xbadf5000.

2. **Second pgob after PGRAPH init**: After the PGRAPH pack MMIO writes
   and FECS clock-gating/power/enable sequence, a second `gk104_pmu_pgob`
   is called to restore FECS DMEM accessibility. The PGRAPH init
   re-gates the FECS power; pgob un-gates it.

3. **Skip ctxctl gate (0x260)**: The ctxctl gate (0x260=0) was causing
   FECS DMEM to become inaccessible. Since the falcon is already STOPPED
   (UC_CTRL=0x10), the gate is unnecessary. IMEM loads fine without it.

4. **FECS DMEM + csdata + start as one continuous write stream**: The
   FECS auto-power-gates within a few MMIO operations of no DMEM access.
   To prevent this, FECS DMEM load, csdata upload, and FECS start are
   done as one continuous write-only sequence with no reads in between.
   - GPCCS DMEM is loaded first (doesn't auto-gate).
   - FECS DMEM is loaded via FALCON_DATA_INDEX/DATA write stream.
   - csdata method words are appended at the known FUC data tail offset
     (0x304 for FECS, 0x6c for GPCCS) without reading the tail pointer.
   - FECS is started immediately after the last csdata write.

### Key findings

- FECS DMEM accessibility is auto-gated by the FECS power domain after
  ~50ms of no DMEM access. DMEM reads return 0xbadf5000 (power-gate
  sentinel) when gated.
- `gk104_pmu_pgob` un-gates the FECS, making DMEM accessible.
- PGRAPH disable (0x400500=0) re-gates the FECS.
- PGRAPH pack MMIO writes re-gate the FECS (one of the ~140 registers).
- The ctxctl gate (0x260=0) also causes DMEM to become inaccessible.
- IMEM access is NOT power-gated (works regardless of gate state).
- The FECS FUC data has head=0x300, tail=0x304 (FECS) and head=0x6c,
  tail=0x6c (GPCCS) at DMEM offsets 0/4.

### Current state

- FECS firmware: LOADS, STARTS, POSTS READY (0x409800 bit31 set).
- GPC falcons: All 4 started by FECS (CTRL=0x20, SCRATCH0=0x80000000).
- ctx_chan (cmd 1): FECS begins processing but hangs at PC 0x1097a
  (xdwait) waiting for VM DMA to complete.
- **Blocker**: FECS VM DMA cannot walk the VMM page tables. The page
  tables need to be in VRAM (BAR1-backed), and the VMM join value format
  in the inst block at offset 0x200 may be incorrect.

## Session 2026-07-12 — macOS crash fixed; golden context proven

- Root cause of the crash-era VMM code was unsafe/malformed channel state:
  one SPT was aliased beneath multiple PGD entries, HOST leaf PTEs used target
  bits in the wrong positions, and RAMIN used the TLB-invalidate PDB encoding
  instead of `gf100_vmm_join_()`'s full address.  The live path now clones the
  validated host VMM into distinct VRAM SPTs and uses the correct join value.
- Added Nouveau's required `g84_bar_flush()` sequence after framebuffer writes.
  A complete 4-KiB RAMIN upload followed by the flush is deterministic; the
  earlier per-word/partial BAR writes were the source of random PBDMA state.
- VBIOS devinit is now enabled by default for the un-POSTed macOS eGPU (explicit
  `KEPLER_VBIOS_DEVINIT=0` remains available for power-gating diagnostics).
- Fixed `wait_cond` at `ctx_chan`: the callback returned `0x80000000`, which was
  incorrectly compared to boolean `True` and reported a timeout after success.
- Live hardware now completes all golden-context stages:
  `FECS ctx_chan done`, `grctx_main done`, and `FECS golden save done` with
  `gpc_nr=4`, `tpc_total=8`.
- Corrected RAMFC `DEVM` from `0xfff` to Nouveau's GR-only `BIT(0)`.
- The macOS system no longer crashes in repeated live runs.  PBDMA0 loads clean,
  stable channel state (`IB_PUT=1`, `IB_ADDR=0x10010000`, `IB_CFG=0x000a0000`,
  signature `0xface`, no HCE/PBDMA interrupt), but does not consume entry 0:
  `IB_GET=0`, `IB_ENTRY=0`, and USERD `GP_GET=0`.
- A VRAM-backed GPFIFO ring, CHID 0 vs 1, a post-kick channel start, and an
  explicit USERD BAR flush all produced the same fault-free stall.  Those
  no-effect experiments were removed.  The remaining blocker is now narrowly
  the runlist/PBDMA dispatch gate before GP entry fetch, not VMM encoding,
  context validity, USERD visibility, or GPFIFO memory aperture.

## Session 2026-07-12 — first GPFIFO entry consumed

- The supposed fault-free stall was a diagnostic blind spot.  PFIFO master
  status was `0x50000000`; `VM_FAULT_SOURCE=0x80` identified fault slot 7
  (HOST0).  Added decoding of all active `0x2800 + unit*0x10` records.
- The concrete fault was a HOST0 read at GPFIFO VA `0x10010000`, reason
  `UNSUPPORTED_APERTURE`.  Both coherent HOST aperture 2 and NCOH aperture 3
  fault at that address on this bring-up path.
- A VRAM leaf alone did not fix VA `0x10010000`.  Moving push/GPFIFO to
  `0x09000000/0x09010000`, inside PGD slot 1 already proven by FECS context DMA,
  removed the MMU fault.  This proves the remaining bug was specific to the
  second PGD traversal/slot, not PBDMA scheduling.
- With `KEPLER_VRAM_GPFIFO=1`, hardware now advances PBDMA `IB_GET=1` and USERD
  `GP_GET=1`: the first GP entry is consumed for the first time.
- The VRAM GP entry now reads back identically through BAR1 and PRAMIN
  (`0x09000000`, high word `0x00001400`).  PBDMA's post-consume
  `IB_ENTRY_LOW=0xfffffffd` is an idle/sentinel value, not the backing ring
  contents.

## Session 2026-07-12 — dispatch narrowed past runlist completion

- Added the exact `gk104_fifo_intr_runlist()` acknowledgement of `0x2a00`
  after the runlist DMA completes.  This clears PFIFO master bit `0x40000000`;
  the live timeout now has master, scheduler, runlist, PBDMA, and HCE interrupt
  state all zero.
- Added optional VRAM backing for the pushbuffer and semaphore, including leaf
  PTE replacement, PRAMIN repair, BAR flush, and direct VRAM polling.  The
  push PTE reads back correctly (`0x6101` in the tested layout), but the same
  post-consume state remains.  Therefore host aperture access is not the cause
  of the push-method stall.
- Compared the stream against `cl906f.h` and `nvif/chan506f.c`: the entry size
  is correctly encoded in dwords, the incrementing method header is
  `0x20040004`, and both MAIN/NOT_MAIN plus NO_PREFETCH variants were tested.
  They do not change the result.
- Tested Nouveau's host semaphore WFI-disabled operation (`0x00100002`) and
  the older GK104 driver's second post-runlist START trigger.  Neither changes
  the stall.
- The decisive live boundary is now: `IB_PUT=IB_GET=USERD_GP_GET=1`, while
  `DMA_GET=DMA_PUT=0`, USERD TOP_LEVEL_GET is invalid, and all PBDMA semaphore
  state registers remain zero.  Thus PBDMA retires the GP entry without ever
  parsing the first pushbuffer method; the next investigation is the GP-entry
  accept/skip condition and RAMFC state, not GR compute or semaphore syntax.

## Session 2026-07-12 — IB CRC proves GPU reads a zero GP entry

- The late CPU readback exposed bitwise-inverted push words on uncleared VRAM
  (`0x20040004 -> 0xdffbfffb`).  Replaced blind PRAMIN writes with verified
  stores: read the old word, apply the XOR delta used by this macOS path,
  verify, and fall back to a literal store.  Ring, push, RAMFC prefix, USERD,
  and runlist now read back exactly before the kick.
- Added SET_REFERENCE and NON_STALL_INTERRUPT host methods ahead of the
  semaphore.  Neither side effect appears, proving this is not merely an
  unreadable semaphore destination.
- The key diagnostic is PBDMA `IB_CRC=0x6904bb59` with `PB_CRC=0`.  Using the
  CRC definition in envytools, `0x6904bb59` is exactly the CRC of eight zero
  bytes, not of the CPU-visible entry (`0x09000000`, `0x00002400`).  Hardware
  therefore fetches a zero GP entry, advances GET, and never starts a
  pushbuffer read.
- Nouveau LTC flush (`0x70010`) plus invalidate (`0x70004`), last-moment
  PRAMIN rewrites, MAIN/NOT_MAIN, and placing push and ring in the same
  physical page do not change the zero-entry CRC.
- Found BAR1 disabled (`0x1704=0`) on the unclaimed eGPU.  A first identity
  BAR1 bootstrap is retained behind `KEPLER_INIT_BAR1=1` only: it is not yet
  correct (BAR1 still reads a malformed instance view and it disrupts the
  FECS PTE readback), so the normal path remains unchanged.  The next blocker
  is now exact and below packet syntax: make CPU framebuffer writes visible
  to GPU clients, most likely by completing the GK104 BAR1/VRAM initialization
  that macOS did not perform.

## Session 2026-07-12 — LTC/FB/BAR1 subdev init added

### Root cause analysis

The PBDMA zero-GP-entry issue (IB_CRC = CRC of eight zero bytes) was caused
by missing GK104 subdev initialization steps that a proper VBIOS POST or
nouveau driver init would have performed:

1. **LTC (L2 cache) not initialized.** `gk104_ltc_init()` writes `ltc_nr`
   to `0x17e8d8` and `0x17e000`, `tag_base` to `0x17e8d4`, and the big-page
   bit to `0x17e8c0`. Without this, the L2 cache is not properly configured
   and GPU internal clients (PBDMA, GR, etc.) read stale/zero data from
   VRAM. The LTC topology is read from `0x022438` (parts), `0x022554`
   (mask), and `0x17e8dc` (lts_nr/slice).

2. **FB sysmem flush page not programmed.** `gf100_fb_sysmem_flush_page_init()`
   writes a sysmem DMA page address >> 8 to `0x100c10`. The GPU uses this
   page to flush dirty cache lines to sysmem. Without it, the L2 cache may
   retain stale zeros and PBDMA/GR read incorrect data.

3. **BAR1 VMM disabled (0x1704=0).** The previous BAR1 identity bootstrap
   mapped only 16 MiB (not enough to cover the 0x400000+ channel VMM page
   table heap), used the wrong VMM flush type (`0x80000005`
   PAGE_ALL|HUB_ONLY instead of just PAGE_ALL), and encoded PTEs with the
   VRAM aperture (aper=0) instead of the HOST aperture (aper=2, VOL=1)
   required for the sysmem-backed eGPU. It was also gated behind
   `KEPLER_INIT_BAR1=1`.

### Fixes applied

1. **Added `_gk104_ltc_init(dev)`** — reads LTC topology from hardware
   registers (`0x022438` parts, `0x022554` mask, `0x17e8dc` lts_nr),
   computes `ltc_nr`, sets `tag_base=0` (no compression tags: eGPU has no
   VBIOS-initialized VRAM, matching the no-ram path in
   `gf100_ltc_oneinit_tag_ram()`), and writes `ltc_nr` to `0x17e8d8`/
   `0x17e000`, `tag_base` to `0x17e8d4`, and the big-page bit to
   `0x17e8c0`. Matches `gk104_ltc_init()` in `ltc/gk104.c` and
   `gf100_ltc_oneinit()` in `ltc/gf100.c`.

2. **Added `_gk104_fb_init_page(dev)`** — sets 17-bit (128 KiB) big-page
   mode by clearing bit 0 of `0x100c80`. Matches `gf100_fb_init_page()`.

3. **Added FB sysmem flush page** — writes `dev.bus_base >> 8` to
   `0x100c10` after sysmem allocation. Matches
   `gf100_fb_sysmem_flush_page_init()` in `fb/gf100.c`.

4. **Fixed `_gk104_init_bar1_identity()`**:
   - Increased default mapping from 16 MiB to 128 MiB (full BAR1 aperture)
   - Removed the post-enable VMM flush; nouveau only does
     `gf100_bar_bar1_init()` (write 0x1704) + `gf100_bar_bar1_wait()` (two
     BAR flushes).  The pre-enable VMM flush now correctly includes
     HUB_ONLY (0x4), matching `gf100_vmm_flush()` when
     `engref[NVKM_SUBDEV_BAR] > 0` (BAR1's VMM has this incremented by
     `gf100_bar_oneinit_bar()`)
   - Writes PTEs in 4 KiB chunks to stay within the PRAMIN window
   - Fixed PTE encoding to use the HOST aperture (aper=2<<33, VOL=1<<32)
     with `bus_base + page*0x1000` as the physical address, matching
     `gf100_vmm_valid()` for `NVKM_MEM_TARGET_HOST`. The previous code used
     aperture 0 (VRAM) with address 0, which is wrong for the sysmem-backed
     eGPU and caused BAR1 reads to return 0xbad0...
   - Added `bus_base` parameter so PTEs can be built with the correct
     sysmem bus address

5. **BAR1 bootstrap remains opt-in** (`KEPLER_INIT_BAR1=1`). It is still
   not enabled by default because it has not been validated on live
   hardware. The normal path uses the direct PCI BAR mapping that TinyGPU
   exposes.

6. **Updated init order** in `_init_hardware()`:
   - LTC init (no sysmem needed)
   - FB init_page (no sysmem needed)
   - Clear any stale BAR1 bootstrap mapping
   - Sysmem allocation
   - FB sysmem flush page (needs sysmem bus address)
   - BAR1 identity bootstrap (opt-in, needs sysmem bus address for PTEs)
   - Memory manager creation

### Key references

- `gk104_ltc_init()`: `ltc/gk104.c` lines 27-36
- `gf100_ltc_oneinit()`: `ltc/gf100.c` lines 208-223 (LTC topology read)
- `gf100_ltc_oneinit_tag_ram()`: `ltc/gf100.c` (no-ram path: num_tags=0)
- `gf100_ltc_flush()/invalidate()`: `ltc/gf100.c` lines 126-149
- `gf100_fb_sysmem_flush_page_init()`: `fb/gf100.c` lines 81-87
- `gf100_fb_init_page()`: `fb/gf100.c` lines 68-78
- `gf100_bar_bar1_init()`: `bar/gf100.c` lines 52-58
- `gf100_vmm_valid()`: `vmmgf100.c` (PTE type encoding: BIT(0) | vol<<32 | aper<<33)
- `gf100_vmm_pgt_pte()`: `vmmgf100.c` (data = (addr>>8) | map->type)
- `gf100_vmm_pgd_pde()`: `vmmgf100.c` (SPT PDE pt[1]: aper<<32 | addr<<24 for VRAM)
- `gf100_vmm_join_()`: `vmmgf100.c` (instance PDB: aper<<0 | pd->addr)

### Verification

- `py_compile` passes.
- `--middle-selftest` passes.
- `NV_BACKEND=software` demo passes.
- NOT yet tested on live hardware. The next live run should show whether
  the LTC init + FB sysmem flush page fixes the PBDMA zero-GP-entry issue
  (IB_CRC should match the real GP entry, not eight zero bytes). The BAR1
  identity bootstrap (KEPLER_INIT_BAR1=1) can be tested separately once the
  basic path is working.

## Session 2026-07-13 — GR auto-power-gating fix (semaphore timeout debug)

### Root cause analysis

After the LTC/FB/BAR1 fixes, the PBDMA zero-GP-entry issue was resolved:
the PBDMA now consumes GP entries and parses push buffer methods. The new
failure is `TimeoutError: semaphore did not reach 2 (last=1)`.

Register dump analysis from the timeout:
- PBDMA0: IB_PUT=1, IB_GET=1 (entry consumed), DMA_GET=0x09000020,
  DMA_PUT=0x09000060 (16 dwords remaining), 0x118=0x20040004 (SEMAPHOREA
  header — last method parsed)
- VM_FAULT: HUB unit 0, FE client (client=4), PTE fault (reason=2) at VA
  0xd000, inst=0x60d000
- PGRAPH_STATUS=0xbadf1000 (GR engine NOT accessible/clocked)
- PBDMA SEM/CTX regs all = 0xbad0011f (register block not accessible)

The PBDMA successfully:
1. Consumed the GP entry
2. Parsed SET_OBJECT (binding compute class to subch 1)
3. Parsed SEMAPHOREA (ACQUIRE_GEQUAL at VA 0x4000, value=1)
4. The semaphore was pre-seeded to 1, so the ACQUIRE passed

Then the PBDMA tried to forward INVALIDATE_SHADER_CACHES (subch 1) to the
GR engine, but GR was not accessible (PGRAPH_STATUS=0xbadf1000). The FE
client faulted at VA 0xd000 (unmapped gap between runlist VA 0xc000 and
gr_ctx VA 0x08000000).

### Root cause: GR auto-power-gating

The GR engine was initialized during `_init_hardware()` (PGRAPH_PACK_MMIO
writes + PMC_ENABLE=0xffffffff), but auto-power-gated by the time
`submit_launch()` runs. On the un-POSTed eGPU, GR power-gates within
seconds of no MMIO access. The FECS golden context init (ctx_chan +
grctx_main + golden save) may also trigger GR power state changes.

### Fix applied

Added GR accessibility checks and re-enablement at two points in
`submit_launch()`:

1. **Before FECS golden ctx init**: Check PGRAPH_STATUS; if 0xbad0...,
   re-assert PMC_ENABLE=0xffffffff, re-run PGRAPH_PACK_MMIO with
   PGRAPH master disable/enable cycle, and re-apply FECS clock-gating
   init (matching `gf100_gr_init` + `gf100_gr_fecs_reset`).

2. **Before runlist commit**: Same re-enablement if GR is still
   inaccessible after the golden context init.

Also added PGRAPH_STATUS and PGRAPH_CTRL to the timeout diagnostic
register dump, and to the channel-bind debug print.

### Verification

- `py_compile` passes
- `--middle-selftest` passes
- `NV_BACKEND=software` demo passes
- NOT yet tested on live hardware. The next live run should show whether
  the GR re-enablement fixes the semaphore timeout. If GR is still not
  accessible after re-enablement, the issue may be deeper (e.g., GPC
  clock domain gating that requires PMU/devinit).

## Session 2026-07-13 — kernel executes but SMs are power-gated (all-zero output)

### Symptom shift: timeout → "hardware add mismatch"

After the GR auto-power-gating fix, the live run no longer times out at
the semaphore. The kernel launch path now completes: the semaphore
releases, `submit_launch()` returns, and the result buffer is read back.
However the result is **all zeros** — `hardware add mismatch`. The
compute kernel is being "executed" from the FIFO's perspective but is
not actually performing any computation.

### Evidence: SMs (MPs) are power-gated

- `PGRAPH_STATUS = 0xbadf1000` is still observed at the post-launch
  diagnostic dump. The GR engine reports the power/clock-gated sentinel
  even though the launch appeared to complete.
- **MP trap registers read `0xbadf1000`** — the Streaming Multiprocessor
  trap status registers return the power-gated sentinel, proving the
  SMs are clocked off and never executed the kernel SASS. The launch
  reached the FIFO/FE dispatch layer but the work never landed on a
  powered SM.
- `FE_PWR (0x404170) = 0x0` during FECS stop — the front-end power
  register drops to zero while FECS is halting.
- `PWR_GATE (0x020004) = 0x40110068` — bits `[31:30]=01b` (bit30=1,
  bit31=0) means top-level GR power-gating is **DISABLED** at the
  PWR_GATE level, yet PGRAPH still power-gates. So the gating is
  happening below PWR_GATE, inside the GR/GPC domain.
- `PMC_ENABLE` GR bit is set; not the cause.
- GPCCS (`CPUCTL=0x20`) keeps running when FECS stops (`CPUCTL=0x0`),
  confirming FECS and GPCCS live in different power domains. FECS
  auto-halts within ~10 ms of ctx_chan completion.

### FECS path taken

- Bypassed the golden save and reloaded the channel context via
  `ctx_chan` (cmd 1) **without** the redswitch path. FECS completes
  `ctx_chan` and the kernel dispatches, but produces zero output.
- `CHSW` interrupt is disabled on this path, so no GPU-triggered
  context-switch save/reload happens — the channel context is loaded
  once and the launch is issued directly.

### GPC_BCAST / GPC MMU investigation

- The context buffer is mapped (PTE is non-zero) — VMM is not the
  blocker for the launch itself.
- `0x418804` (GPC_BCAST MMU control) reads `0xbadf1000`, suggesting
  the GPC_BCAST domain is power-gated.
- However, **writes to GPC_BCAST registers (e.g. `0x4188a4`) stick** —
  readback returns the written value, not the sentinel. So the
  GPC_BCAST domain *is* reachable for writes; `0x418804` itself just
  returns the sentinel. All GPC MMU setup writes appeared to stick.
- Even after explicitly setting `RED_SWITCH` and moving the GPC MMU
  setup earlier (before `ctx_chan`), the SMs remain power-gated.
  Manual GPC MMU setup is necessary but not sufficient.

### Launch sequence verified

A subagent walk-through of the Kepler compute kernel launch sequence
confirmed correct usage of `LAUNCH`, `LAUNCH_DESC_ADDRESS`,
`CODE_ADDRESS_LO/HI`, `CB_CONFIG`, and `INVALIDATE_SHADER_CACHES`. The
launch packet encoding is not the blocker — the SMs simply are not
powered on to receive it.

### Current blocker

The SMs (and the GPC_BCAST compute domain that contains them) are
power-gated. The redswitch operation that normally powers on the
GPCs/SMs is not being performed on this direct bring-up path, because
the CHSW/golden-save sequence was bypassed to avoid the earlier
`0x409804=0xffffffff` hang. The manual GPC MMU writes and RED_SWITCH
host writes are not enough to bring the SMs out of power-gate.

### STUCK — macOS host crash on live run

The current state of `add.py` **crashes the macOS host** when run on
the live eGPU. The crash happens during the SM power-on / GPC_BCAST
bring-up attempts described under "Next step" below. This is a hard
blocker: the host goes down, so no further live diagnostics can be
collected from this code path without risking repeated crashes.

Likely crash vectors (not yet isolated because the host dies before
any post-failure dump can be captured):

- Driving `RED_SWITCH` `0x41a614` (per-GPC) or the GPC_BCAST
  `ctx_xfer` path from the host while the GPC power domain is in an
  inconsistent state. A malformed power-on sequence can assert a
  GPU-level fault that the macOS IOMMU/PCIe layer escalates into a
  host panic.
- Re-enabling the CHSW interrupt after the bypass: the FECS resume
  command previously left `0x409804=0xffffffff` and never returned
  status 1. If CHSW fires while the firmware is in that state, the
  GPU may issue an invalid DMA or an unmapped host access.
- Writes to GPC_BCAST registers that stick on this un-POSTed card may
  be landing in a power domain that macOS did not claim, triggering a
  PCIe abort that propagates to the host.

### Next step (blocked on crash isolation)

Manually enable the SMs and the GPC_BCAST power domain. Options being
explored — **all currently blocked by the macOS crash**:

1. Explicitly trigger the GPC_BCAST `ctx_xfer` command (the GPCCS
   firmware path that normally powers on the GPCs/SMs during a context
   switch), instead of relying on host RED_SWITCH writes alone.
2. Mimic the redswitch logic of the GPCCS firmware: drive
   `RED_SWITCH` `0x41a614` (per-GPC) through the GPCCS command FIFO
   rather than from the host, so the GPC power sequencing is done by
   the falcon that owns that domain.
3. Re-enable the CHSW interrupt and let FECS perform a real
   context-switch load (which includes the GPC power-on sequence),
   while avoiding the `0x409804=0xffffffff` resume hang that forced
   the bypass in the first place.

Before any of these can be tried again on live hardware, the crash
must be isolated. The safe approach is to gate each candidate fix
behind an env flag and run them one at a time with a minimal
reproduction, so a single offending write can be identified without
re-crashing the host. A host-side PCIe error log (if available from
`ioreg` / `log show` after reboot) would also help pin the faulting
transaction.

### Verification

- `py_compile` passes.
- `--middle-selftest` passes.
- `NV_BACKEND=software` demo passes.
- Live hardware: **crashes macOS**. Not yet a `hardware_demo=ok`
  result. The all-zero-output / SMs-power-gated state from the
  previous run is the last non-crashing data point.

## Session 2026-07-13 — corrected false power-gating diagnosis and add launch ABI

The previous conclusion that the all-zero result proved powered-off SMs was
incorrect. The diagnostic used `0x500000 + gpc*0x8000 + tpc*0x800 + 0x428`,
which is neither the GK104 TPC base nor either MP trap register. Nouveau defines
`TPC_UNIT` at `0x504000`; `gf100_gr_trap_mp()` reads warp/global errors at
TPC offsets `0x648` and `0x650`. Reads from the old address therefore could not
support the SM power-gating diagnosis.

Two actual correctness bugs were present in the hardware add path:

1. `get_kepler_cubin()` silently used `build_cubin()` when Docker or
   `KEPLER_CUBIN` was unavailable, even though that builder explicitly contains
   non-executable placeholder SASS. The live path now refuses to initialize the
   GPU unless it has an ELF `sm_30` cubin with `.text.E_4`.
2. The three CUDA kernel arguments were placed at c0 offsets `0x0/0x8/0x10` in
   a 0x100-byte buffer. An nvcc `sm_30` kernel reads its parameter area at
   c0 offsets `0x140/0x148/0x150`. The parameter buffer is now 0x200 bytes, the
   CWD advertises that size, and GPR allocation comes from `.text.E_4` metadata.

The host-crashing experiments were removed from the submission path: no
RED_SWITCH reset is repeated around a live context, the duplicate GPC_BCAST
write probe is gone, and the arbitrary pre-launch GR register write is gone.
RED_SWITCH remains only in the normal Nouveau-derived engine initialization and
in the explicit clock probe. MP diagnostics now use the real topology and
Nouveau register addresses.

### Verification

- `python3 -m py_compile examples_kepler/add.py` passes.
- `python3 examples_kepler/add.py --middle-selftest` passes, including checks for
  the CUDA parameter ABI, CWD size/GPR fields, and MP trap addresses.
- `NV_BACKEND=software python3 examples_kepler/add.py` reports
  `software_demo=ok N=256` with a correct 256-element add result.
- With no real cubin and Docker unavailable, the default command exits before
  touching the GPU with: `hardware launch refused: live hardware requires a
  real sm_30 add cubin; set KEPLER_CUBIN or start Docker`.
- A real-hardware result still requires a genuine nvcc-built `sm_30` add cubin
  supplied with `--cubin PATH` or `KEPLER_CUBIN=PATH`. No further live run was
  attempted without that prerequisite, so `hardware_demo=ok` is not yet
  claimed.

## Session 2026-07-13 — real sm_30 cubin obtained; FECS falcon stuck at PC 0x567

Docker came online, so a genuine nvcc-compiled `sm_30` add cubin is now
available at `examples_kepler/add_kepler.cubin` (24 instructions, 8 registers).
CUDA 11.0.3-devel-ubuntu20.04 was used because CUDA 11.8+ dropped `sm_30` from
nvcc. The cubin is compiled as `sm_35` then the ELF flags are patched to
`0x001e051e` (sm_30); the SASS instructions used (MOV/S2R/IMAD/ISCADD/LD.E/
FADD/ST.E/EXIT) are encoded identically on sm_30 and sm_35. `add.py` now
auto-compiles this cubin via `compile_kepler_cubin_docker()` when
`KEPLER_CUBIN` is unset and Docker is available.

The `EF_CUDA_SM30` constant was corrected from `0x300030` (decimal 48, wrong)
to `0x001e001e` (decimal 30, correct). The sm_30 ELF-flag check in
`get_kepler_cubin()` was likewise fixed to compare against `0x1e`, not `0x30`.

With the real cubin loaded, the hardware path now runs end-to-end through
VBIOS devinit, GPC PLL, PMU, PGOB, PGRAPH init, falcon firmware load, VMM
setup, GR context allocation, runlist submission, and launch — but the kernel
never executes on the SMs. Output is still all zeros (256/256 mismatch).

### Current blocker: FECS falcon never starts executing

The FECS falcon is loaded with firmware but never progresses past PC 0x567.
Diagnostic evidence:

- `FECS PC samples after start: ['0x567', '0x567', ...]` — PC frozen.
- `GPC0-3 fuc: CTRL=0x20 ENTRY=0x0` — GPCCS falcons never started by FECS.
- `PGRAPH_STATUS=0xbadf1000` throughout — GR engine never reaches healthy idle.
- `FECS_CTRL=0x00000000` after start (bit4 STOPPED clear, but PC not advancing).

Root-cause investigation via deepwiki (allbilly/linux_drm) and envytools docs:

1. **Missing `nvkm_mc_unk260` writes**: nouveau's `gf100_gr_init_ctxctl_int`
   writes `0x000260=0` before firmware load (disables ctxctl clock-gate) and
   `0x000260=1` after (re-enables). These were missing; now added.
2. **Premature `0x409614` (RED_SWITCH/FECS reset) write removed**: the code
   was writing `0x409614=0x70` + `0x700` during PGRAPH init, but nouveau only
   does this during `gf100_grctx_generate` (context allocation), AFTER firmware
   load. Writing it before load power-gates the falcon's IMEM. Removed.
3. **FE_PWR FORCE_ON added**: nouveau's `gf100_grctx_generate` sets
   `0x404170=0x12` (FORCE_ON) and waits for bit4 clear before falcon work,
   then restores `0x404170=0x10` (AUTO) after. This prevents auto-power-gating
   between MMIO accesses during firmware upload. Added.

After these fixes, DMEM readback now matches (0x300, 0x3cc, 0x0, 0x0 — the
0x3cc is the updated csdata tail pointer). IMEM readback is partially working
(first word 0x39b0ef5 matches, but subsequent reads intermittently return
0xbadf5000 — the IMEM read window at 0x180/0x184 may not auto-increment
correctly, or the falcon's IMEM port is still being intermittently gated).

Despite the firmware loading correctly, FECS PC is still stuck at 0x567.
The instruction at offset 0x564 is `0xf40028f4` — likely a wait/sleep
instruction. The falcon appears to be in a SLEEPING state waiting for an
interrupt or mailbox signal that never arrives.

### Next steps

- Decode the FECS instruction at PC 0x567 using envydis or KeplerAs to
  determine what the falcon is waiting for (mailbox? interrupt? context
  switch signal?).
- Check whether `0x409048` (ACCESS_EN) needs to be set to `0x3` (FIFO|CHSW)
  before starting FECS, as nouveau's `nvkm_falcon_v1_start` does
  (`base + 0x048 = 0x3`).
- Verify the FECS csdata/method-stream upload is correct — the DMEM tail
  pointer changed from 0x304 to 0x3cc, but the csdata format may not match
  what the FECS firmware expects.
- Investigate whether the PMU firmware needs to be fully functional for FECS
  to progress (the PMU is loaded but may not be responding to commands).

### Verification

- `python3 -m py_compile examples_kepler/add.py` passes.
- `python3 examples_kepler/add.py --middle-selftest` passes.
- `NV_BACKEND=software python3 examples_kepler/add.py` reports
  `software_demo=ok N=256`.
- `KEPLER_CUBIN=./add_kepler.cubin python3 examples_kepler/add.py` runs the
  full hardware path but fails with `hardware add mismatch (256/256 wrong)`.
- `hardware_demo=ok` is still not achieved.

## 2026-07-12: live GK104 compute launch checkpoint

The earlier conclusion above that PGRAPH/FECS never became usable is now
superseded.  The current bring-up reaches the A0C0 compute launch method on the
Kepler eGPU:

- VBIOS devinit and the 300 MHz GPC PLL complete; FECS reports a `0x2c400`
  context image and executes context-channel/save commands.
- The GK104 topology is detected as four GPCs with two TPCs each.  The golden
  context now includes Nouveau's topology/floorsweep state and is copied from
  the `CB_RESERVED + 0x80000` generation area into the runtime channel context,
  exactly as `gf100_grctx_generate()` does.
- PBDMA consumes the GP entry and advances `DMA_GET` to `0x09000028`, the byte
  offset of `NVA0C0_LAUNCH`.  A GPU semaphore-only packet has separately been
  verified on hardware (`hardware_semaphore=ok value=2`).
- The artificial pre-launch semaphore acquire was removed.  Inputs are fully
  written before this synchronous one-shot submission, while the completion
  semaphore remains a GPU packet after launch.  The hardware path does not call
  the software backend and does not copy a CPU-computed result into `out_dev`.

### Mesa cross-check

The launch descriptor is being checked against the local Mesa mirror:

- `ref/mesa.mesa/src/gallium/drivers/nouveau/nvc0/nve4_compute.c`, function
  `nve4_compute_setup_launch_desc()`
- `ref/mesa.mesa/src/nouveau/headers/nvidia/classes/cla0c0qmd.h`, layout
  `NVA0C0_QMDV00_06`

The descriptor now sets Mesa's cache invalidations, FE/L1 membars, unchecked API
call limit, `SASS_VERSION=0x30`, 16 KiB directly-addressable shared-memory
configuration, `SHADER_LOCAL_MEMORY_CRS_SIZE=0x800`, and the register count.
The exact packed words are `qmd[7]=0xbc000000`, `qmd[11]=0x04014000`,
`qmd[20]=0x20000001`, and `qmd[47]=0x30000800` before adding other fields.

### Current live failure (do not report success yet)

The GPU still does not complete the vector add.  The live launch raises
PGRAPH interrupt `0x00200000`, trap unit `0x01000000` (GPC), followed by an MP
trap in one TPC per GPC; the completion semaphore remains `1`.  This is now an
SM launch/code problem rather than an unconsumed pushbuffer or a GPCCS wait.

The checked-in `add_kepler.cubin` is not yet acceptable as final proof: it was
compiled as `sm_35` and its ELF flags were patched to identify as `sm_30`.
CUDA 11.0's `nvcc` rejects `-arch=sm_30`, and no CUDA 10.2 image is currently
cached.  A genuine GK104 (`sm_30`) cubin must be produced (CUDA 10.2 or a
verified Kepler assembler route), inspected, and used for the final hardware
run.  Success requires both `hardware_demo=ok` and the numerical comparison of
GPU-written output against the CPU reference; neither is claimed at this
checkpoint.

### Tests at this checkpoint

- `python3 -m py_compile examples_kepler/add.py`: pass.
- `python3 examples_kepler/add.py --middle-selftest`: pass,
  `launch_words=19`.
- Live eGPU run: reaches `NVA0C0_LAUNCH`, then MP trap and timeout.

## 2026-07-12: genuine sm_30 binary and SET_OBJECT isolation

This section supersedes the binary/toolchain and launch-boundary statements in
the preceding checkpoint.

### Genuine GK104 cubin now available

NVIDIA's archived 36 MB `cuda-nvcc-10-2_10.2.89-1_amd64.deb` was downloaded
to `/tmp` and extracted without installing it system-wide.  Its `ptxas` was run
inside the already-cached CUDA 11 container only to provide an amd64 Linux
runtime.  The source is `examples_kepler/add_kepler.ptx` and the compiler
reported:

```
Compiling entry function 'E_4' for 'sm_30'
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 8 registers, 344 bytes cmem[0]
```

The resulting `examples_kepler/add_kepler.cubin` is 1768 bytes, has native ELF
flags `0x001e051e`, and SHA-256
`0716d4ce397d5e2126bec9fd9cd3c32bdfd312326333ce641a3a78a2bd89e098`.
No ELF flag patch is used.  CUDA 10.2 `nvdisasm` confirms the expected native
SM30 instruction sequence (parameter loads, `S2R`, address calculation, two
global loads, `FADD`, global store, and `EXIT`).

### Mesa/Nouveau corrections made

- `build_cwd()` now matches Mesa's `NVA0C0_QMDV00_06` packing for cache
  invalidation, membars, SASS version, L1 configuration, CRS size, register
  count, dimensions, and constant-buffer address/size.
- FECS mailbox 1 is captured immediately after firmware startup.  It is later
  reused for command data, so the previous late read returned `0x8c000`
  instead of the real context size `0x2c400`.
- The copied channel context header is replaced as `gf100_gr_chan_bind()`
  requires.  It now contains the count and VA/256 of a 24-entry per-channel
  MMIO list for pagepool, bundle, attributes, and LTC state.
- Pagepool, bundle, GR context, and MMIO-list mappings use Nouveau's required
  privileged PTE bit where applicable.  Scratch registers contain channel VAs,
  not framebuffer physical addresses.

### BAR1 corruption found and contained

TinyGPU bulk BAR1 zero/copy operations reproducibly leave alternating stale
`0xffffffff` dwords and occasional inverted bytes.  A 1 MiB context allocation
contained 131072 dirty dwords after a nominal zero fill.  The bring-up now:

1. verifies and repairs the golden context, pagepool, bundle, and attribute
   allocations through the verified PRAMIN dword path;
2. compares all `0x2c400` bytes of the FECS-saved golden context with the
   runtime copy;
3. repairs only mismatched dwords and requires a second comparison to report
   `remaining=0` before scheduling the channel.

This prevents a corrupted context copy from being mistaken for a compute or
compiler failure.

### Exact remaining hardware boundary

A genuine empty SM30 kernel was compiled for diagnosis.  It traps identically,
so user instructions and buffer accesses are not the trigger.  A still narrower
`KEPLER_TEST_SET_OBJECT=1` push contains only:

1. `SET_OBJECT` for class A0C0;
2. a GPU semaphore release;
3. a non-stall interrupt.

That push is entirely consumed by PBDMA, but `SET_OBJECT` causes PGRAPH trap
`0x01000000`, with GPC bits `0x0e` and an MP trap in TPC1 of GPC1--3.  The GPU
semaphore remains at its sentinel because its WFI release waits behind the
faulted GR engine.  Trap latches are zero immediately before submission, so
this is not stale state from golden-context generation.

Therefore the current defect is in GR engine-context/object binding state,
before QMD address, launch, SASS, or data memory.  `hardware_demo=ok` and the
numerical result are still not achieved and must not be reported as passing.

---

## Message 247 — ACCESS_EN=0x3 applied, PGRAPH still 0xbadf1000, FECS stuck at 0x567

### What was tried

- Applied the `nvkm_falcon_v1_start` fix: wrote `0x00000003` to
  `FECS_FALCON_BASE + 0x048` (ACCESS_EN = FIFO|CHSW enable) just before
  reading the ctx size at `0x409804`. This is the value nouveau sets in
  `nvkm_falcon_v1_start` to give the falcon FIFO and channel-switch access
  to host methods.

### Result (KEPLER_CUBIN=add_kepler.cubin python3 add.py)

The run still fails with the same `TimeoutError: semaphore did not reach 2
(last=1)`. Key observations from the trace:

- `ACCESS_EN=0x00000002` after FECS start (we wrote 0x3, read back 0x2).
  Bit 0 (CHSW) did not stick — only bit 1 (FIFO) remained. This suggests
  the falcon's security config rejects CHSW access on this SKU, or the
  register is read-only after firmware start.
- `FECS_CTRL=0x20` (HALTED bit set) — the falcon never actually started
  executing. `CPUCTL=0x20` means bit5 (HALT) is set and bit0 (START) is
  clear. The `nvkm_falcon_v1_start` write of `0x3` to CPUCTL (START|HALT)
  is not happening, or it was immediately re-halted.
- `FECS PC samples after start: ['0x567'] x10` — PC frozen at 0x567.
- `PGRAPH_STATUS=0xbadf1000` throughout the entire run, from boot through
  the 2-second submit_launch poll. The GR engine is *never* accessible.
- `PGRAPH_CTRL=0x10001` (master enable is set), `PGRAPH_INTR=0x0`,
  `FE_PWR=0x0` (FE power mode is OFF — not even AUTO).
- `PMC_ENABLE=0xe011312d` — GR bit (0x1000) is set, but many other engine
  bits are masked off by the hardware (we wrote 0xffffffff).
- `PWR_GATE=0x40110068` (0x020004) — bit30 (GR un-gate) is set, bit31
  (ROP un-gate) is clear. This matches the pgob sequence.
- DMEM probes FAIL at every stage with `0xbadf1000` / `0xbadf5000`
  sentinels — FECS DMEM is power-gated the entire time.
- `FECS IMEM verify: match=False` — only the first word matched
  (`0x39b0ef5`), the rest read back `0xbadf5000`. IMEM is also gated.
- `FECS golden ctx: inst_tag=0x8000060d` — the inst block was written,
  but `FE_PWR before ctx_chan: wrote 0x10 readback 0x0` — FE power
  writes do not stick.
- `grctx_main` ran but `post-grctx_main: PGRAPH_STATUS=0xbadf1000` —
  the GR engine was inaccessible the entire time, so grctx_main was a
  no-op (its MMIO writes went into the void).
- `userd_gp_get=0x1` — PBDMA consumed the entry, but the FECS never
  dispatched it to the GR engine because FECS is halted and GR is gated.

### Root cause analysis

The fundamental problem is that **PGRAPH (the GR engine) is never coming
out of power-gate**. Every downstream symptom (FECS halt, DMEM 0xbadf5000,
IMEM verify failure, semaphore stuck at 1, grctx_main no-op) is a
consequence of the GR clock/power domain being gated.

The pgob sequence we run (`gk104_pmu_pgob`) performs the register writes
nouveau does, but on this eGPU the writes do not have the intended effect:

1. `0x020004` bit30 (GR un-gate) reads back as set, but the GR domain
   does not actually power up — `0x400000` still returns `0xbadf1000`.
2. The `0x10a78c` handshake (the XBAR power-gate control) completes but
   does not propagate to the GR partition.
3. The `0xc800` PMU magic pokes all report `ready=YES` with the expected
   `final_c800` values, so the PMU *is* executing the commands — but
   they are not un-gating GR.

This strongly suggests that on a cold-attached eGPU (no EFI POST, no
NVINIT), the GR power partition requires more than the nouveau runtime
pgob sequence. The VBIOS devinit scripts we run (script0=0x87e5 etc.)
are supposed to bring up clocks and power, but they leave
`PGRAPH_STATUS=0xbadf1200` — still gated. The GPC PLL programs
successfully (`lock=YES`, target=300000 actual=300000), but the GR
domain clock is not being routed to the engine.

### Candidate next steps

1. **Check 0x020004 bit31 (ROP un-gate)** — we set bit30 (GR) but not
   bit31 (ROP). nouveau's `gk104_pmu_pgob` sets *both* via
   `0xc0000000 -> 0x40000000` which only sets bit30. Verify whether the
   ROP partition also needs un-gating for PGRAPH_STATUS to clear.
   Actually, re-reading the nouveau source: the mask is `0xc0000000`
   and the value is `0x40000000`, so bit31 is *cleared* and bit30 is
   *set*. This is correct per nouveau. But maybe on this SKU both need
   to be set?

2. **Investigate the FECS HALT** — `CPUCTL=0x20` means the falcon is
   halted. We should explicitly write `0x3` (START|HALT) to CPUCTL
   (0x409100) to start it, per `nvkm_falcon_v1_start`. Check whether
   our `falcon_load(..., start=True)` is actually writing the START
   bit, or whether something is re-halting it afterwards.

3. **Check if the GR engine needs a reset cycle** — nouveau does
   `nvkm_mc_disable(GR)` then `nvkm_mc_enable(GR)` (PMC_ENABLE bit12
   toggle) as part of `gf100_gr_init`. Our code sets PMC_ENABLE once
   to 0xffffffff and never toggles GR reset. Try:
   - Clear bit12 of PMC_ENABLE
   - Wait
   - Set bit12 of PMC_ENABLE
   - Then run pgob + PGRAPH init

4. **Verify the GPC PLL is actually feeding the GR clock tree** —
   `GPC_PLL=0x30005` with `lock=YES` but `GPC_DIVSRC=0x3`. Check
   whether `0x137160` (GPC clock divider/source) needs a different
   value to route the PLL to the GR domain. The VBIOS devinit may
   have left the divider in a state that gates the GR clock.

5. **Dump 0x409604 (GPC_UNK) after each init step** — this register
   tells us whether the GPC partition is alive. It reads
   `0xbadf1200` after devinit, which means the GPC partition itself
   is gated, not just PGRAPH.

6. **Consider that the eGPU needs a full PMU boot** — the PMU
   firmware we load (`gk104_pmu_code.bin`) may need to be started
   and allowed to complete its init before pgob works. Currently
   we load PMU and call `falcon_load(..., start=True)` but never
   wait for the PMU to signal ready. The `0xc800` pokes succeed
   (ready=YES) but that may just mean the PMU accepted the command,
   not that it executed the power-gate sequence.

### Verification

- `python3 -m py_compile examples_kepler/add.py` passes.
- Hardware run still fails: `TimeoutError: semaphore did not reach 2
  (last=1)` after 2-second poll.
- `PGRAPH_STATUS=0xbadf1000` throughout — GR engine never accessible.
- `FECS_CTRL=0x20` (halted), `FECS PC=0x567` (frozen).
- `ACCESS_EN=0x2` (FIFO only, CHSW bit did not stick).
- `hardware_demo=ok` is still not achieved.

## Session 2026-07-13 — PBDMA MMU fault on instance block; TARGET=3 (NCOH) tested

### Investigation

Continued debugging the PBDMA scheduler inactivity. The runlist DMA
completes (PLAYLIST_RD shows 1 entry read), the channel is enabled
(PFIFO_CHAN STATE=0x11030001, ENABLED=True, ENGINE=3), and USERD has
GP_PUT=1 > GP_GET=0, but PBDMA_CHID stays 0 and SCHED_STATUS=0x00000005.

Added detailed MMU fault diagnostics with faulting addresses
(0x2800 + unit*0x10, reading inst/valo/vahi/type). Found MMU faults
on multiple units:

- **Unit 0**: inst=0x0000060e, reason=2 (PAGE_NOT_PRESENT),
  client=0x4 (DISPATCH), hub=True, read. This is the PBDMA trying to
  read the instance block at VRAM address 0x60e000 with TARGET=0
  (VRAM).
- **Unit 7**: inst=0x0000060d, addr=0x4000, reason=2, client=0x7
  (BAR), hub=True, write=True.

### Root cause hypothesis

With TARGET=0 (VRAM), the PBDMA tries to bypass the HUB MMU (BAR1
VMM) and access VRAM directly. On this eGPU, VRAM is backed by sysmem,
and the HUB MMU is enabled (BAR1 VMM at 0x1704). The HUB MMU intercepts
all accesses and generates a PAGE_NOT_PRESENT fault because it has no
PTE for the VRAM aperture — the BAR1 VMM only maps virtual addresses
to sysmem bus addresses.

### Fix attempted: TARGET=3 (NCOH)

Changed the channel instance pointer from TARGET=0 (VRAM) to TARGET=3
(NCOH) so the PBDMA routes through the HUB MMU. The instance block at
VRAM offset 0x60e000 should map to sysmem bus address
(bus_base + 0x60e000) through the BAR1 VMM.

### Result

- PFIFO_CHAN now reads CHAN=0xb000060e (bit28-29=11b = TARGET=3),
  STATE=0x11030001 (ENABLED=True, ENGINE=3).
- MMU fault unit 0 persists: inst=0x0000060e, reason=2
  (PAGE_NOT_PRESENT), client=0x4 (DISPATCH). The HUB MMU still has
  no PTE for virtual address 0x60e000.
- PBDMA1 remains idle: IB_PUT=0, IB_GET=0, DMA_GET=0, CHID=0.
- SCHED_STATUS=0x00000005 (unchanged).
- The BYPASS path still runs (CTXCTL CHAN_VALID=False), and the
  semaphore timeout occurs as before.

The TARGET=3 change alone did not fix the PBDMA because the BAR1 VMM
identity mapping only covers 128 MiB (0x8000000) starting from
bus_base, and the instance block at offset 0x60e000 (6.1 MB) should
be within that range. The PAGE_NOT_PRESENT fault suggests the BAR1
VMM PTEs may not be correctly encoding the virtual-to-physical
mapping, or the HUB MMU is using a different page table than BAR1.

### Other findings

- **FREEZE register (0x2638)** reads 0xbad0011f on GK104 — it is
  GF100-only (`variants="GF100:GK104"` in envytools). Writing 0 to
  it has no effect on GK104.
- **KICK_CHID (0x2634)** was already being written with chan_id but
  has no effect on the scheduler.
- **SCHED_STATUS=0x5** is not documented in envytools or nouveau.
  DeepWiki suggested it might relate to BAR2, but this was
  inconclusive.
- Stale MMU faults were cleared before runlist commit but new faults
  appear after the commit, confirming the PBDMA is actively trying
  to access the instance block and failing.

### macOS crash

The live run crashed macOS during this session. The crash occurred
after the BYPASS path submitted methods and the semaphore timeout
diagnostic ran. This is consistent with the earlier crash pattern
where GPU-initiated DMAs to unmapped/invalid addresses cause PCIe
aborts that escalate to a host panic.

### Verification

- `py_compile` passes.
- Live hardware: crashes macOS. The last non-crashing data point
  shows PBDMA idle, SCHED_STATUS=0x5, MMU fault on DISPATCH client
  reading the instance block.

### Next steps (blocked on crash isolation)

1. **Verify the BAR1 VMM actually maps virtual address 0x60e000**.
   Read the PGD/SPT entries for that VA and confirm the PTE points
   to (bus_base + 0x60e000) with the HOST aperture. The identity
   bootstrap maps 128 MiB starting at bus_base, so VA 0x60e000 should
   map to bus_base + 0x60e000 — but only if the PTE was written
   correctly.
2. **Consider using sysmem directly (not through HUB MMU)** for the
   instance block. If the HUB MMU is the problem, placing the instance
   block at a sysmem bus address with TARGET=3 might work if the
   PBDMA can access sysmem directly without the HUB MMU.
3. **Investigate SCHED_STATUS=0x5** — this may indicate the scheduler
   is in a faulted/error state that prevents channel dispatch
   regardless of the instance block accessibility.
4. **Isolate the crash** before attempting any further live runs.
   Gate each candidate fix behind an env flag and test one at a
   time.

## Session 2026-07-13 — macOS crash isolated; GK104 bind fixed and fail-closed

### Root cause of the dangerous path

- Local Nouveau's `gk104_chan_bind_inst()` writes exactly
  `0x80000000 | (inst_addr >> 12)`.  The previous code incorrectly ORed
  `TARGET=3` into bits 28-29 of the GK104 channel instance register.  Those
  target bits belong to the runlist descriptor, not the channel bind.
- The timeout path then attempted manual GF100 channel-table writes, manual
  FECS context switches, PFIFO BYPASS, and direct PGRAPH dispatch.  That let a
  failed channel setup progress to GPU DMA and is the credible macOS-panic
  mechanism.
- MMU fault clearing was also wrong: `0x2800+` contains fault payload records;
  Nouveau acknowledges active records through `VM_FAULT_SOURCE` (`0x259c`).

### Fixes

- Bind the VRAM instance as `VALID | addr>>12`, matching Nouveau exactly.
- Require a VRAM instance for the live path; do not reinterpret BAR1 as a
  HUB-MMU translation for PFIFO.
- Gate manual channel-table, CHSW, BYPASS/direct-dispatch, GF100 `0x2200`,
  scheduler kick/freeze, and repeated-start experiments behind
  `KEPLER_UNSAFE_EXPERIMENTS=1`.
- Refuse to write `GP_PUT` when an active MMU or PBDMA HCE fault exists.
- Stop and unbind the channel on success, pre-launch rejection, or timeout;
  stop the FECS keep-alive thread before teardown.
- Acknowledge MMU faults through `0x259c` and decode only source-selected
  records.
- Mirror USERD `GP_PUT` through PRAMIN and invalidate LTC.
- Use the checked-in nvcc-built `examples_kepler/add_kepler.cubin` by default,
  so the normal command no longer requires `KEPLER_CUBIN` or Docker.

### Live validation and remaining blocker

- Four semaphore-only live runs completed without a macOS crash.
- Correct bind is retained as `PFIFO_CHAN=0x8000060e`; runlist 3 DMA completes
  with the expected `(chid, 0)` entry; `VM_FAULT_SOURCE=0`; HCE is clear.
  CHID 0 and CHID 1 behave identically.
- BAR1+PRAMIN USERD writes, LTC invalidation, and an explicit PFIFO MC reset do
  not change the result.
- The remaining blocker is PFIFO scheduling: `SCHED_STATUS=0x5`, all PBDMA
  `IB_PUT/IB_GET` values remain zero, and `GP_GET` remains zero.  Therefore
  even the semaphore-only push is not consumed, and the numerical add kernel
  has not run on silicon yet.  Do not report `hardware_demo=ok`.

### Verification

- `python3 -m py_compile examples_kepler/add.py` passes.
- `python3 examples_kepler/add.py --middle-selftest` passes.
- `NV_BACKEND=software python3 examples_kepler/add.py` reports
  `software_demo=ok N=256` with correct results.
- `python3 examples_kepler/add.py --probe` identifies GK104 (`chip_id=0x0e4`).

## Session 2026-07-13 — GR runlist fixed; semaphore gate passes on silicon

### Root causes fixed

- Corrected the TOP runlist decode.  Nouveau uses
  `(data & 0x01e00000) >> 21`; GR's `0x8006183e` entry is runlist **0**, not
  runlist 3.  PBDMA0 now schedules the channel normally.
- Restored the channel-specific RAMFC values used by Nouveau userspace
  channels: `DEVM=BIT(0)` and `priv=false`.
- Added delayed full-RAMFC stabilization.  This removed the immediate PBDMA
  `DEVICE/EMPTY_SUBC` exception and the all-ones loaded context.
- Restored Nouveau's submission ordering: publish USERD while the channel is
  absent, stabilize the command pages, then commit the runlist.  USERD writes
  use the CPU BAR1 aperture; RAMFC/page tables use PRAMIN.
- Made runlist, GPFIFO, pushbuffer, and semaphore VRAM-backed by default.
- Fixed the cloned channel VMM PDE encoding by removing the invalid VRAM
  `BIT(34)` and stabilized live PDEs/PTEs.  This progressed faults in order
  from `PDE_SIZE`, to the exact GPFIFO VA, to the push VA, and finally to the
  semaphore VA; all are now resolved.
- Added writable scratch aliases for deterministic complemented GR context
  buffer bases and for the stale all-ones PTE touched at VA `0x67000`.  The
  normal path remains fail-closed on any new unexpected MMU/HCE state.

### Live verification

- Default semaphore-only command now succeeds on the GK104 eGPU:
  `KEPLER_TEST_SEM_ONLY=1 python3 examples_kepler/add.py` reports
  `hardware_semaphore=ok value=2` with `MMU_FAULT: none`.
- `python3 -m py_compile`, `--middle-selftest`, and the software demo pass.
- Repeated live runs in this session did not crash macOS.

### Exact remaining compute boundary

- The default full command now reaches `GP_GET=1` and GR dispatch with no MMU
  fault.  It no longer stalls in PFIFO/PBDMA or faults during context load.
- `SET_OBJECT`/GR dispatch raises `PGRAPH_INTR=0x00200000` and
  `PGRAPH_TRAP=0x01000000`, with all four GPCs reporting a TPC1 trap.  The
  completion semaphore remains 1.
- Therefore `hardware_demo=ok` and numerical add results are **not yet
  achieved**.  The remaining defect is the saved GR engine context/object
  binding state before CWD, SASS, or output memory execution; do not report
  hardware compute as passing.

## Session 2026-07-13 — LAUNCH accepted, zero output, macOS crash reported

### Progress since the GR-object trap

- Delaying `GP_PUT` until after runlist admission and explicitly loading the
  runtime context with FECS `ctx_chan` removes the SET_OBJECT race.  With
  `KEPLER_PUT_BEFORE_RUNLIST=0`, subchannel 1 binds class `0xa0c0`, methods
  advance through `LAUNCH` (`0x2bc`), and the completion semaphore reaches 2.
- The launch has no selected MMU fault and no PGRAPH interrupt/trap after the
  method stream.  This proves the failure moved past PFIFO, context load, and
  compute-object binding.
- Corrected the NVA0C0 QMD bit layout from Mesa's `cla0c0qmd.h`: cache
  invalidation is in word 7 (`0x1c`), `PROGRAM_OFFSET` is in word 8 (`0x20`),
  and all remaining fields are now inserted by absolute bit number.
- Added Mesa's missing NVE4 TEMP setup (`TEMP_ADDRESS` and both
  `MP_TEMP_SIZE` banks), with a non-zero 32-KiB allocation per each of the
  GTX 770's eight MPs.
- Verified the checked-in cubin with envytools `envydis`: its real GK104 SASS
  loads parameters from constant-buffer offsets `0x140`, `0x148`, and `0x150`,
  matching `build_cuda_param_cbuf()`.

### Current failure and safety status

- Two corrected full launches retired without an MMU/PGRAPH fault but returned
  256 zero floats (`mismatches=256/256`).  The shader therefore did not make a
  visible global store; `hardware_demo=ok` is still unproven.
- The user reported that macOS crashed after the latest full hardware run.
  Treat repeated full launches as unsafe until teardown and the GPC-visible
  memory path are isolated.  Do not describe this session as crash-free.
- The leading hypothesis is an aperture mismatch: only the QMD is currently
  mirrored into BAR1-backed VRAM, while SASS, constant data, inputs, output,
  and TEMP remain SYS mappings.  The next probe should mirror the complete
  compute working set into the already-validated VRAM VMM and read the result
  from that GPU-written allocation.  This is a memory-domain correction, not
  a CPU implementation of vector addition.

### Verification retained

- `python3 -m py_compile examples_kepler/add.py` passes.
- `python3 examples_kepler/add.py --middle-selftest` passes with 35 launch
  words after adding TEMP setup.
- The semaphore-only silicon gate previously passed with `MMU_FAULT: none`;
  it must be rerun after the macOS reboot before another compute launch.

### Second macOS crash and lifecycle fix

- macOS crashed again during the next chained validation command.  No new full
  compute launch had been issued in that command; the chain reached the live
  `--probe` after offline compilation/self-test.  This is consistent with a
  persistent PFIFO playlist from the preceding process destabilising the GPU
  after userspace and its BAR allocations disappear.
- The old success/timeout cleanup stopped CHID 1 and cleared its instance
  pointer, but it **did not commit an empty GR runlist**.  On GK104 the
  playlist is hardware state and survives the Python process; leaving the
  channel ID in it can make PFIFO revisit an unbound channel after teardown.
- Added a fail-safe teardown matching Nouveau's runlist lifecycle: block GR
  runlist 0, commit it with count 0, wait for `RUNLIST_PENDING` to clear,
  acknowledge the runlist interrupt, stop the channel, clear its instance,
  return FECS to idle, and only then close the TinyGPU connection.
- The same empty-runlist transaction now runs before CHID reuse, cleaning a
  stale playlist before a new bind.  It is registered with `atexit`, attached
  to `NVDevice.close()`, and used on completion, prelaunch rejection, and
  timeout so ordinary exceptions cannot skip hardware quiescence.
- Hardware execution remains paused until this teardown change passes offline
  checks.  After that, the first live test must be a single semaphore-only
  gate; a compute launch is not justified until the machine remains stable.

### Third crash: Apple PCIe interrupt storm identified

- macOS crashed a third time while only offline commands were running.  The
  latest three panic reports (`20:15`, `20:18`, and `20:25`) have the same
  kernel panic string:
  `apciec[pcic1-bridge] unhandled interrupts (0x200000 out of 0x220000)` at
  `APCIECPort.cpp:2056`.
- This rules out the offline Python work as the trigger and identifies a stale
  external PCIe interrupt from the eGPU/bridge as the immediate host failure.
  PFIFO playlist cleanup is still required, but it cannot by itself prevent
  the empty-runlist completion interrupt from reaching macOS.
- The userspace RM polls MMU, PFIFO, PGRAPH, FECS, and semaphore state and has
  no DriverKit interrupt handler.  It previously left GK104's MC interrupt
  output armed.  Added Nouveau's interrupt-unarm policy at first BAR access,
  again before submission, before every runlist commit, and before teardown:
  write zero to `PMC_INTR_EN_0` (`0x140`) and GK104 MC leaf-0 source mask
  (`0x640`), followed by Nouveau's posting read of `0x140`.
- Do not reconnect to or map the eGPU merely to test this fix while it remains
  in the stale asserted state.  The enclosure must be physically power-cycled
  (not just macOS rebooted) before the next live semaphore-only test; offline
  tests do not clear endpoint/Thunderbolt bridge interrupt state.
- Completed two additional Mesa parity fixes offline: set
  `TEX_CB_INDEX=7` during NVE4 setup and issue `NV50_GRAPH_SERIALIZE` after
  `LAUNCH`.  Hardware output now starts as a NaN sentinel and the entire
  code/CB/input/output/TEMP/QMD working set is prepared for VRAM mirroring, so
  a missing or partial shader store cannot be mistaken for a correct result.
- Current offline verification passes: `py_compile`, `--middle-selftest`
  (`launch_words=39`), the software numerical demo, and `git diff --check`.

## Session 2026-07-13 — semaphore passes, delayed PCIe panic persists

### Live result after physical enclosure power-cycle

- A single `KEPLER_TEST_SEM_ONLY=1` run completed correctly on silicon:
  PBDMA0 reached `IB_PUT=IB_GET=1`, USERD reached `GP_GET=GP_PUT=1`, the
  semaphore became 2, and there was no selected MMU fault.
- The new lifecycle path also completed: the final line before process exit was
  `channel quiesced (completion): empty runlist, inst=0`, followed by
  `hardware_semaphore=ok value=2`.
- macOS nevertheless panicked roughly 15 seconds later.  The new report
  `panic-full-2026-07-13-203704.0002.panic` has the same Apple bridge failure:
  `apciec[pcic1-bridge] unhandled interrupts (0x200000 out of 0x220000)`.

### Revised crash boundary

- Emptying PFIFO's GR runlist and clearing the channel instance are necessary
  but not sufficient; the panic occurs after both are proven complete.
- Masking NVIDIA `PMC_INTR_EN_0` and MC leaf 0 is also insufficient.  The
  bridge status therefore is not simply an unserviced NVIDIA functional
  interrupt that can be fixed with `0x140/0x640`.
- The remaining delayed-lifetime hazard is GPU bus mastering after TinyGPU
  releases the contiguous SYS/IOMMU mappings.  Even with no runnable channel,
  FIFO, GR/FECS/GPCCS, PMU, HUB MMU, and the programmed sysmem flush page remain
  live when the socket closes.  A later internal DMA to a released mapping can
  surface as a PCIe-controller interrupt after Python has exited.
- No further live command is safe until teardown resets every DMA-capable
  engine and clears GPU references to SYS memory *before* `fini()` lets the
  DriverKit allocation disappear.  The next implementation work is based on
  Nouveau's device/MC shutdown ordering and TinyGPU's mapping ownership.

### DriverKit mapping-lifetime fix implemented

- The TinyGPU protocol already provides the missing PCI lifecycle operations
  (`CFG_READ=3`, `CFG_WRITE=4`, `RESET=5`); the Kepler transport had omitted
  them even though `examples/add.py` uses the same commands.
- Hardware initialization now explicitly enables PCI memory decoding and bus
  mastering only for the `NVDevice` lifetime, keeps legacy INTx disabled, and
  disables SERR/parity reporting because this polling-only DriverKit client
  does not service those interrupt paths.
- Teardown now performs the critical ordering while SYS mappings are still
  owned by TinyGPU: empty PFIFO runlist, clear channel instance, halt PGRAPH,
  reset FIFO/GR/PMU through `PMC_ENABLE`, clear the SYS flush-page register,
  disable PCI bus mastering with a config-space posting read, wait 50 ms, and
  only then close the socket.  `NVDevice.close()` repeats the bus-master-off
  operation as a final exception-safe guard.
- Once its read-back gate passes on hardware, this ordering prevents
  GPU-originated DMA after the IOMMU mappings are released.  It is implemented
  but not yet live-validated.  A new physical enclosure power-cycle is required
  before testing because the latest panic again left the endpoint/bridge in an
  asserted state.

### Offline audit after the latest crash

- Compared the Kepler transport with the pinned, working TinyGPU client in
  `examples/add.py`.  The config-space RPC wire layout is identical: request
  `bar=0`, followed by `offset`, `size`, and (for a write) `value`.
- Centralized PCI bus-master shutdown in `_disable_pci_bus_master()`.  It now
  retries the command write up to three times and requires a config-space
  posting read to prove `PCI_COMMAND_MASTER` is clear.  A write that does not
  stick is a hard teardown error rather than a silently released DMA mapping.
- `APLRemotePCIDevice.fini()` also invokes that guard, so direct probe helpers
  cannot close a TinyGPU connection while the endpoint is still allowed to
  initiate DMA.  Normal `NVDevice.close()` retains the full ordered
  engine/runlist shutdown first; the transport guard is deliberately
  redundant for exception paths.
- Hardware initialization now validates that PCI memory decoding and bus
  mastering actually read back enabled before any BAR/SYS-memory setup.
- Added a fake-socket transport test to `--middle-selftest`.  It verifies the
  exact TinyGPU frames for `CFG_READ(PCI_COMMAND, 2)`,
  `CFG_WRITE(PCI_COMMAND, 2, 0x0402)`, and the posting read, entirely offline;
  it does not connect to the eGPU or start TinyGPU.app.
- The expected successful live teardown line now includes the read-back value:
  `PCI_COMMAND=0x0402, bus_master=off`.  Absence of that line is an unsafe
  result and must block any compute launch.
- Made late `GP_PUT` the default.  This is the only ordering that already
  reached SET_OBJECT, LAUNCH, and the completion semaphore on silicon: the
  channel is admitted and its FECS context loaded before PBDMA sees the entry.
  The former early-PUT behavior remains available with
  `KEPLER_PUT_BEFORE_RUNLIST=1` for diagnostics only.

### Current verification and next safe hardware gate

- `python3 -m py_compile examples_kepler/add.py` passes.
- `python3 examples_kepler/add.py --middle-selftest` passes with
  `launch_words=39`, including the new TinyGPU/PCI shutdown test.
- `NV_BACKEND=software python3 examples_kepler/add.py` reports
  `software_demo=ok N=256` with correct results.
- `git diff --check` passes.
- A final local Nouveau audit confirms the reset bits used here: GK104 maps
  FIFO to `PMC_ENABLE 0x100`, the inherited GF100 GR path uses `0x1000`, and
  PMU uses `0x2000`.  The teardown mask `0x3100` therefore covers every engine
  this userspace path started.  Nouveau can leave the SYS flush page programmed
  because the kernel owns its DMA mapping; this DriverKit path must clear
  `0x100c10` before releasing its shorter-lived mapping.
- Do not run `--probe`, semaphore-only, or full compute against the endpoint
  left stale by the latest panic.  After another physical enclosure
  power-cycle, run exactly one semaphore-only command.  It must report the
  completion semaphore, no MMU/PGRAPH fault, and the verified
  `PCI_COMMAND=0x0402, bus_master=off` teardown.  Then leave the machine idle
  long enough to cross the previous roughly 15-second delayed-panic window.
  Only a stable result permits one full add launch.

## Session 2026-07-13 — bus-master-off gate passes, delayed panic still occurs

### Live semaphore result after enclosure power-cycle

- Ran exactly one `KEPLER_TEST_SEM_ONLY=1 python3 examples_kepler/add.py` with
  late `GP_PUT`; no compute launch was attempted.
- The GPU completed normally: PBDMA consumed the entry, the completion
  semaphore reached 2, `MMU_FAULT: none`, and the process exited successfully.
- Ordered teardown completed and config-space read-back reported
  `PCI_COMMAND=0x0403, bus_master=off`.  `0x0403` is valid here: bits 0 and 1
  retain I/O and memory decode, bit 10 disables INTx, and the critical bus
  master bit 2 is clear.
- The machine remained alive through an explicit 30-second idle gate, but then
  panicked later.  The new report is
  `panic-full-2026-07-13-210119.0002.panic`; it has the identical panic string
  `apciec[pcic1-bridge] unhandled interrupts (0x200000 out of 0x220000)` at
  `APCIECPort.cpp:2056`.  TinyGPU was still present in the panic process list.

### Hypothesis falsified and stronger close implemented

- A verified clear `PCI_COMMAND_MASTER` disproves bus mastering alone as a
  sufficient fix.  No MMU fault occurred and the panic was delayed well beyond
  command completion, so the remaining state is endpoint/bridge interrupt or
  link state retained after the TinyGPU client disconnects.
- The proven `examples/add.py` client already exposes and uses TinyGPU's
  `RESET` RPC after clearing bus mastering.  Kepler close now uses the stronger
  sequence after result read-back: clear bus mastering, issue the DriverKit PCI
  function reset, wait 100 ms, then clear bus mastering plus PCI memory/I/O
  decode and verify the final command register before the socket closes.
- `APLRemotePCIDevice.fini()` applies the same function-reset guard to direct
  probes and exception paths.  Normal close marks the reset complete so it is
  not issued twice.
- The panic snapshot still contained a live `TinyGPU` server process although
  the Python client had exited.  The Kepler transport now retains ownership of
  the server process it launches for its unique temporary socket.  After the
  PCI function is reset and made inert, it closes the socket, waits briefly for
  that owned server to exit, and sends SIGTERM only if it remains.  A server
  reached through an existing/shared socket is never terminated.
- Extended the fake-socket self-test to verify all seven close transactions,
  including `RESET=5` and the post-reset `PCI_COMMAND=0x0400` target.  This is
  fully offline and does not touch the currently stale endpoint.
- This reset-based shutdown is implemented but not live-validated.  Another
  physical enclosure power-cycle is required before one semaphore-only test;
  do not attempt full compute until the machine survives a substantially
  longer idle window than this run.

## Session 2026-07-13 — reset-based shutdown passes live safety gate

### Carefully scoped live result

- After a physical enclosure power-cycle, ran exactly one semaphore-only test;
  no compute launch or follow-up probe was issued.
- The scheduler path completed normally: the completion semaphore reached 2,
  `MMU_FAULT: none`, and `hardware_semaphore=ok value=2`.
- Ordered channel teardown first reported
  `PCI_COMMAND=0x0403, bus_master=off`.  Final close then completed the new
  DriverKit function-reset sequence and reported
  `PCI_COMMAND=0x0403->0x0400, bus_master=off`, proving bus mastering and PCI
  memory/I/O decode were all disabled after reset.
- The uniquely spawned TinyGPU server exited and was confirmed absent.  No
  existing/shared process was terminated.
- Performed idle-only monitoring with no eGPU access for more than three
  minutes.  The machine remained stable through the prior delayed-panic point
  (the previous reset-less run panicked roughly two minutes after exit).

### Current safety conclusion

- The evidence now isolates the macOS crash to PCI function/DriverKit server
  state retained after a normal socket close, rather than the completed GPU
  command itself.  Clearing bus mastering alone was insufficient; function
  reset followed by full decode disable and owned-server shutdown is the first
  teardown that has survived the delayed-panic window.
- Treat this as one successful safety validation, not yet proof across repeated
  runs.  Preserve the checkpoint: do not chain commands or issue a full compute
  launch in the same session.  The next live step should be a separately
  authorized single full add run after confirming the host remains stable; its
  VRAM output must match all 256 CPU-computed reference values before reporting
  `hardware_demo=ok`.

### Later crash invalidates the three-minute safety conclusion

- macOS subsequently panicked at 21:22:20, several minutes after the final
  21:17:48 idle checkpoint and roughly eight minutes after the semaphore-only
  process exited.  The report is
  `panic-full-2026-07-13-212220.0002.panic`.
- The panic string is again byte-for-byte identical:
  `apciec[pcic1-bridge] unhandled interrupts (0x200000 out of 0x220000)` at
  `APCIECPort.cpp:2056`.  Therefore the three-minute observation window was
  too short; the reset-based teardown is **not** a crash fix.
- The panic snapshot again contains a `TinyGPU` process even though `pgrep -x
  TinyGPU` confirmed it absent immediately after our owned server exited.  This
  indicates that terminating the client-owned server does not detach the
  installed DriverKit/PCI service permanently; macOS or the application can
  relaunch/rebind it later while the enclosure remains connected.
- Bus mastering, BAR/I/O decode, empty runlist, engine reset, PCI function
  reset, and owned-server exit have now all been individually verified and are
  still insufficient.  The remaining failure is below the Python channel
  lifecycle: persistent Thunderbolt/Apple PCIe controller or DriverKit service
  state for this unsupported Kepler endpoint.
- Mark live execution unsafe again.  Do not run semaphore-only, probe, or full
  compute after reboot/replug until a method exists to detach the DriverKit PCI
  service or power down/disconnect the enclosure immediately after the test.
  `hardware_demo=ok` remains unachieved.

## Session 2026-07-13 — Kepler-specific shutdown fix after repeated panic

### Corrected scope

- The TinyGPU DriverKit extension is not treated as the component to replace:
  the same signed transport is already stable with the RTX 3080 and the
  `allbilly/amdgpu` path, and rebuilding it would not be a practical deployment
  fix.  The attempted no-authorization detach did not detach the service and no
  DEXT files were modified.
- Reverse engineering the installed DEXT established only that its generic
  `Stop_Impl` closes the `IOPCIDevice`; it does not establish a generic TinyGPU
  defect.  The repeated crash is now approached as Kepler state left behind by
  this bring-up sequence.

### Root shutdown omissions found in local Nouveau

- Bring-up calls the equivalent of `gk104_pmu_pgob(..., false)`, forcing
  `0x020004[31:30]=01b` to release GPC/ROP/LTC power gating, but teardown never
  called the inverse `enable=true` operation (`11b`) while PMU and GR were
  still alive.
- Bring-up manually selects and enables the GPC PLL at `0x137000`, but teardown
  never moved GPC clock index 0 back to the fixed 100 MHz divider before
  deselecting and disabling that PLL.
- Masking the two master interrupt registers did not disable or acknowledge
  the leaf sources configured during bring-up.  PFIFO/PBDMA, PTherm, GPIO,
  AUX/I2C, and PMU sources could therefore remain latched after the Python
  process released the endpoint.  The panic bit `0x200000` also overlaps the
  GK104 MC GPIO/I2C source bit, which makes this omission plausible but is not
  yet proof of causality.

### Implemented fix

- Extended `gk104_pmu_pgob()` with the Nouveau `enable` argument.  Shutdown now
  restores `0x020004[31:30]=11b` before resetting PMU/GR, without executing the
  bring-up-only War00C800 sequence.
- Added the Nouveau-compatible clock/power fini sequence: restore all eight
  GK104 engine clock-gate low bytes to `0x54`; program GPC index 0's divider to
  fixed 100 MHz; clear its PLL selection with read-back verification; then
  clear PLL sync and enable bits.
- Added leaf interrupt shutdown before engine reset: mask PFIFO and all three
  PBDMAs and acknowledge their latched status; perform exact
  `g84_therm_fini`; mask/ack both GK104 GPIO banks and AUX/I2C; and perform the
  `gt215_pmu_fini` mask plus pending-status acknowledgement.
- Refactored emergency teardown into independent best-effort stages.  An empty
  runlist, PGOB, clock, or interrupt exception can no longer skip engine reset,
  flush-page removal, or the final retrying PCI bus-master disable.  Logging no
  longer claims bus mastering is off when its read-back could not be obtained.
- Corrected the stale bring-up comment which described the actual `0x54` clock
  gate write as `0x44`; executable behavior was already `0x54`.

### Offline validation and safety status

- `python3 -m py_compile examples_kepler/add.py`: pass.
- `python3 examples_kepler/add.py --middle-selftest`: pass,
  `launch_words=39`; the fake-register test proves PGOB restore, fixed-clock
  selection, PLL disable, all eight clock-gate values, and all leaf masks.
- `NV_BACKEND=software python3 examples_kepler/add.py`: pass,
  `software_demo=ok N=256 launch_words=39 cwd_bytes=256`.
- No eGPU MMIO, probe, app launch, or other live hardware access was performed
  in this session.  The fix is offline-verified but not yet live-validated.
  Because the currently connected endpoint may retain state from the panic,
  require a fresh enclosure power-cycle/replug before one isolated hardware
  attempt.  Do not chain diagnostics, and do not claim `hardware_demo=ok` until
  all 256 output values match and the host remains stable through the known
  delayed-panic window.

## Session 2026-07-14 — live attempt blocked before hardware access

- Authorized one isolated `python3 -u examples_kepler/add.py` hardware run.
- TinyGPU rejected the very first `CFG_READ` of PCI command register offset
  `0x04` with: `Driver not available. Check: System Report > PCI for GPU,
  System Settings > Privacy & Security.`
- The failure occurred before BAR mapping, GPU MMIO, engine initialization,
  runlist submission, or compute.  Consequently the new Kepler shutdown path
  was not exercised and this attempt provides no crash-fix validation.
- Did not issue a probe or automatic retry.  macOS must first show the eGPU and
  make the signed TinyGPU DriverKit service available; after that, run only one
  isolated add attempt from a fresh enclosure power state.

### Subsequent panic corrects the failed-run safety assessment

- macOS subsequently panicked at 01:23:09.  The new report is
  `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-012309.0002.panic`.
- Its panic string is again identical:
  `apciec[pcic1-bridge] unhandled interrupts (0x200000 out of 0x220000)` at
  `APCIECPort.cpp:2056`.
- The panic snapshot contains a live `TinyGPU` process (PID 85845), while the
  Python process had already exited.  Therefore the prior statement that the
  rejected `CFG_READ` carried no crash risk was wrong: no GPU MMIO occurred,
  but starting and leaking the client-owned TinyGPU server still changed the
  DriverKit/endpoint lifecycle and was followed by the same panic.

### Partial-constructor server leak found and fixed

- `NVDevice.__init__()` constructed `PCIIface` (which launched the unique
  TinyGPU server) and then called `_init_hardware()`.  When its first
  `read_config()` raised `RuntimeError`, object construction aborted.  Because
  `main()` never received a `dev` object, its `finally: dev.close()` could not
  execute, leaving the owned server alive indefinitely.
- `NVDevice.__init__()` now has deterministic exception rollback: any partially
  created PCI transport is finalized before the original exception propagates.
- `APLRemotePCIDevice` now also rolls back its own `_connect()` constructor
  failures and has an idempotent destructor fallback.  Finalization closes the
  socket and terminates the uniquely owned server in a `finally` block, so a
  reset or socket exception cannot skip process cleanup.
- Added a PCI-config availability guard.  If the first `CFG_READ` was rejected,
  cleanup does not issue reset/config RPCs against that unavailable endpoint;
  after one successful config read, the existing reset-and-disable close path
  remains active.
- Added an offline regression test for the exact failure state.  It proves the
  socket is closed, the owned server is terminated, and no additional endpoint
  RPC is sent.  Existing/shared TinyGPU processes remain outside this ownership
  path and are never terminated.
- `main()` now handles transport `RuntimeError` as a controlled initialization
  failure instead of emitting an uncaught traceback.  Its error text no longer
  recommends a follow-up `--probe`, because that would relaunch the same server
  against an unavailable/stale endpoint.

### Validation after lifecycle fix

- `python3 -m py_compile examples_kepler/add.py`: pass.
- `python3 examples_kepler/add.py --middle-selftest`: pass,
  `launch_words=39`, including failed-constructor cleanup regression coverage.
- `NV_BACKEND=software python3 examples_kepler/add.py`: pass,
  `software_demo=ok N=256 launch_words=39 cwd_bytes=256`.
- `git diff --check`: pass.
- No post-reboot eGPU probe or hardware retry was made.  The Kepler leaf IRQ,
  PGOB, and PLL shutdown fix was not reached by the failed live attempt and
  remains unvalidated.  Do not retry until a fresh enclosure power-cycle and
  macOS both show the PCI endpoint and permit the signed DriverKit service.

## Session 2026-07-14 — second live panic identifies PCIe Completion Abort

### Latest crash evidence

- A subsequent user-started live run panicked macOS at 01:35:08.  The report is
  `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-013508.0002.panic`.
- The panic is again from `AppleT8103PCIeC` with
  `unhandled interrupts (0x200000 out of 0x220000)` at
  `APCIECPort.cpp:2056`.
- Both `Python` (PID 3725) and `TinyGPU` (PID 3728) were live in the panic
  snapshot.  This is not the prior failed-constructor leak: macOS interrupted
  an active run before Python could guarantee close/teardown.
- The enclosure and generic TinyGPU path remain explicitly excluded as a
  general root cause: the user has confirmed this eGPU path works with RTX
  3080 and RX 570 through `allbilly/amdgpu`.  The failure is specific to
  transactions generated by the current GK104 path.

### Controller bit decoded and prior hypothesis corrected

- The upstream Linux Apple SoC PCIe driver defines port interrupt bit 21
  (`0x00200000`) as `PORT_INT_CPL_ABORT`: PCIe **Completion Abort**.  It defines
  bit 17 (`0x00020000`) as `PORT_INT_REQADDR_GT32`, explaining the panic's
  `0x220000` relevant mask.  Source:
  https://github.com/torvalds/linux/blob/master/drivers/pci/controller/pcie-apple.c
- Therefore the repeated `0x200000` value is not evidence of NVIDIA MC
  GPIO/I2C interrupt bit 21; the numerical overlap was coincidental.  Leaf
  interrupt masking remains orderly GPU teardown, but it does not directly
  explain this Apple controller panic.
- Completion Abort means a PCIe config/BAR transaction was rejected by the
  endpoint.  The present Kepler close path made two unsafe generation-specific
  assumptions: it issued TinyGPU's function-reset RPC, then disabled BAR/I/O
  decode and performed config read-back.  Newer NVIDIA and AMD endpoints may
  support that lifecycle; consumer GK104 must not be assumed to support FLR or
  post-reset accesses the same way.

### GK104 Completion-Abort mitigation

- Added `_shutdown_kepler_pci_for_close()`.  The active Kepler close path now
  clears PCI bus mastering, sets INTx-disable, and verifies read-back while
  leaving memory/I/O decode enabled until DriverKit closes its user client.
- Removed function reset and decode-disable from both normal `NVDevice.close()`
  and the partial/direct transport finalizer.  The generic reset RPC and its
  offline protocol test remain available in code, but the GK104 path no longer
  calls them.  This change is local to `examples_kepler/add.py` and does not
  alter the working 3080 or `allbilly/amdgpu` implementations.
- Added an offline wire-level assertion that the real GK104 close emits exactly
  config read/write/read, retains BAR decode, clears bus mastering, disables
  INTx, and never emits `RemoteCmd.RESET`.
- Added a fail-closed live gate to the default hardware run and every live
  `--probe*` path.  Without the exact
  `KEPLER_LIVE_ACK=completion-abort-risk` acknowledgement, the script exits
  before launching TinyGPU.  This prevents a plain command or suggested probe
  from causing another panic while the mitigation is unvalidated.

### Status

- No eGPU access was made while diagnosing or implementing this change.
- Offline validation passes: Python compilation, middle self-test
  (`launch_words=39`), 256-element software demo, and `git diff --check`.
  Default hardware execution and `--probe` both exit with status 2 before
  TinyGPU starts, and `pgrep` confirms no TinyGPU process was left behind.
- This mitigation targets a controller condition now identified from primary
  source, but the precise rejected transaction is not present in the panic
  snapshot.  Treat it as a stronger, testable fix—not proof—until one isolated
  power-cycled run completes and survives the delayed-panic window.

## Session 2026-07-14 — panic stack symbolication finds teardown error

### Location of the crash

- Symbolicated the captured Python image UUIDs against the exact installed
  Python 3.14 framework and `_socket` extension.  At panic, Python's only
  captured thread was in `_socket.sock_sendall` / `sock_call_ex`, reached from
  the bytecode evaluator.  In this program, the only socket `sendall` path is
  `APLRemotePCIDevice._rpc`, so Python was actively sending a TinyGPU RPC—not
  idling after process exit.
- The Python process held roughly 613 MB resident, consistent with the 256 MB
  GPU-visible system allocation plus runtime buffers; this was not an early
  config/BAR probe.
- The FECS keepalive thread was absent from the panic snapshot even though it
  catches all RPC exceptions and loops until `_fecs_keepalive_stop` is set.
  `_quiesce_channel()` sets that event as its first action.  Together, these
  facts place the latest panic in teardown/exit with high confidence.
- TinyGPU's captured main thread symbolicates to `TinyGPUCLIRunner.run`, waiting
  inside its request loop.  The panic snapshot cannot recover the command byte,
  but it corroborates an in-flight client/server transaction.

### Exact unsupported operations found in `add.py`

- The previous crash patch added `_gk104_shutdown_clocks_power()`, which called
  `gk104_pmu_pgob(enable=True)`, switched GPC to the fixed divider, deselected
  the GPC PLL, and disabled the PLL during every channel teardown.
- A complete local-reference search shows Nouveau calls
  `nvkm_pmu_pgob(..., false)` only from `gf100_gr_oneinit()` and
  `gf100_gr_init()`.  There is no `true` caller anywhere in Nouveau shutdown.
- Nouveau's `gk104_clk` function table has no `.fini` callback.  Consequently
  GK104 device fini does not switch away from or disable the active GPC PLL.
  The previous Python teardown was therefore not a Nouveau-compatible reverse
  sequence; it introduced power/clock transitions while the PCIe endpoint was
  still serving DriverKit MMIO.
- Teardown set the keepalive stop event but did not join the thread.  One final
  FE_PWR BAR0 RPC could overlap the main thread's power/reset sequence (socket
  serialization prevents byte interleaving but not the incorrect hardware
  transition ordering).

### Fix

- Removed `_gk104_shutdown_clocks_power()` and its teardown call entirely.
  Shutdown no longer asserts PGOB `enable=true`, changes GPC clock selection,
  or disables the GPC PLL.  The clock/power state that successfully served the
  channel remains stable until FIFO/GR/PMU engine reset.
- Removed the `enable` argument from the Python PGOB helper and hard-coded the
  only locally used/Nouveau-backed operation (`false`).  A future teardown edit
  therefore cannot accidentally re-enable the unsupported reverse transition.
- `_quiesce_channel()` now sets the keepalive stop event and joins the thread
  before issuing any shutdown MMIO.  The socket's two-second timeout bounds an
  in-flight RPC; the join allows 2.5 seconds before reporting a teardown error.
- Updated offline tests to prove leaf interrupt masking leaves PGOB, GPC PLL
  selection, and PLL enable/sync bits untouched.  The earlier test asserting
  the destructive reverse transition has been removed.
- The separate Completion-Abort mitigation remains: GK104 close does not use
  FLR and does not disable BAR decode while DriverKit can still transact.

### Validation

- Python compilation: pass.
- Middle self-test: pass, `launch_words=39`.
- 256-element software add: pass.
- `git diff --check`: pass.
- No live device access was performed.  The exact faulty shutdown operations
  are removed, but live success still requires one explicitly acknowledged,
  freshly power-cycled run followed by the delayed-panic observation window.

## Session 2026-07-14 — 01:51 panic and last-known-good comparison

### New panic evidence

- The next explicitly acknowledged live run still panicked macOS at 01:51:58:
  `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-015158.0002.panic`.
- The Apple controller condition is unchanged: Completion Abort
  `0x200000 out of 0x220000` at `APCIECPort.cpp:2056`.
- Python (PID 4078) and TinyGPU (PID 4080) were both alive.  Python again had
  roughly 615 MB resident and only one thread, placing it after the FECS
  keepalive stop/join transition rather than in early initialization.
- Symbolication differs usefully from the prior crash: Python was in
  `_socket.sock_recv_into` via `sock_recv_guts`, waiting for a synchronous
  TinyGPU RPC response.  The prior panic caught `sock_sendall`.  Removing PGOB
  reversal, GPC PLL shutdown, and the keepalive race was therefore necessary
  cleanup but not sufficient; a later synchronous teardown request remained.

### Comparison with committed history

- Reviewed all commits touching `examples_kepler/add.py`.  Commit `6b04d4f`
  (`failure is isolated to A0C0 SET_OBJEC`) is the last committed diagnostic
  path that records a complete two-second hardware timeout without recording a
  macOS panic.  Its path had no layered channel/PCI close implementation.
- Current code had accumulated two PCI shutdown layers: `_quiesce_channel()`
  performed `_disable_pci_bus_master()` (config read/write/read), then
  `NVDevice.close()` unconditionally called `_shutdown_kepler_pci_for_close()`
  and performed a second config read/write/read.  Transport `fini()` had a
  third guard layer.
- The first successful shutdown did not set `_close_shutdown_done`, so the
  second synchronous transaction was guaranteed.  This exactly matches the
  newest panic location: keepalive absent, main thread waiting in
  `recv_into`, TinyGPU still alive.

### Fix after comparison

- Reduced `_quiesce_channel()` to the Nouveau channel lifecycle: stop channel,
  commit an empty runlist, unbind the instance, then revoke PCI bus mastering.
  Removed all post-unbind GR, FIFO, PMU, power, leaf-subdevice, and flush-page
  MMIO.  Once the channel is absent and PCI bus mastering is clear, these
  extra endpoint transactions add risk without permitting any further DMA.
- A successful bus-master read-back now immediately sets
  `_close_shutdown_done`.  `NVDevice.close()` checks that flag and sends no
  endpoint RPC; `APLRemotePCIDevice.fini()` already observes the same flag and
  only closes the socket/owned server.
- Added a native pthread name for each teardown stage.  If another macOS panic
  occurs, its Python thread entry should identify the exact operation (for
  example `kgpu:empty GR runlist` or `kgpu:PCI bus-master disable`) instead of
  exposing only the generic socket function.
- Added an offline one-shot regression test: when quiesce has marked PCI
  shutdown complete, `NVDevice.close()` plus transport `fini()` close the
  socket while emitting zero additional protocol frames.

### Offline validation

- Python compilation: pass.
- Middle self-test: pass, `launch_words=39`, including the new zero-frame
  one-shot-close assertion.
- 256-element software add: pass.
- `git diff --check`: pass.
- No additional live access was made after the 01:51 panic.  Live execution
  remains gated and unsafe until a fresh power-cycle and explicit test.

## Session 2026-07-14 — 02:20 panic removes all teardown BAR MMIO

### Result of next validation

- A user-started validation of the one-shot/minimal teardown still panicked at
  02:20:49.  Report:
  `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-022049.0002.panic`.
- Panic string remains AppleT8103PCIeC Completion Abort
  `0x200000 out of 0x220000`.
- Python (PID 3519, about 615 MB resident) and TinyGPU (PID 3521) were alive;
  each had one thread.  Python again symbolicates to
  `_socket.sock_recv_into`, blocked waiting for a synchronous RPC response.
  Both processes stopped making progress at approximately the same time for
  roughly three seconds before the kernel panic, indicating the DriverKit
  operation itself stalled.
- The native pthread stage name did not appear in Apple's panic snapshot for
  the main Python thread, so it did not disambiguate the remaining MMIO/config
  read.  Stack and thread-count evidence still places the request after the
  keepalive stop transition.

### Deeper conclusion and fix

- The prior minimal path still performed master-interrupt reads, channel
  stop read-modify-write, empty-runlist status reads, and unbind BAR writes
  before clearing PCI bus mastering.  Any one could be the synchronous request
  that received Completion Abort after the active command phase.
- Teardown now performs zero BAR0/BAR1 RPCs after stopping and joining the FECS
  keepalive thread.  Its first and only endpoint shutdown operation is PCI
  command-space bus-master disable with read-back.
- Once `PCI_COMMAND_MASTER` is clear, even a channel left bound in hardware
  cannot originate DMA into the DriverKit SYS/IOMMU mappings.  The successful
  read-back sets `_close_shutdown_done`; normal close and transport fini then
  issue no additional endpoint request and only close the socket/owned server.
- This deliberately prefers PCI-level DMA revocation over attempting to make a
  wedged GK104 scheduler orderly through additional BAR transactions.  It also
  matches the git-history lesson: the last committed non-panicking timeout path
  did not contain the accumulated BAR teardown machinery.

## Session 2026-07-14 — post-quiesce full-add access is the crash boundary

### Additional panic

- Another user-started hardware run panicked at 07:56:34 with the identical
  AppleT8103PCIeC Completion Abort.  Report:
  `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-075634.0002.panic`.
- Python (PID 2332, about 615 MB resident) and TinyGPU (PID 2335) were both
  active with one thread each.  This was another hardware process, not the
  concurrent offline compilation/self-test command.  Python was again waiting
  for a synchronous teardown/post-launch RPC.

### Exact full-add sequencing error

- Git-history comparison showed four semaphore-only live runs completed
  without a macOS crash before full compute was enabled.  The first crashes
  coincide with full launches whose completion semaphore reached 2 but whose
  256 output floats remained zero.
- The decisive code difference was after semaphore completion:
  `submit_launch()` called `_quiesce_channel("completion")` before returning.
  Semaphore-only then printed success and returned without more GPU access.
- Full `run_hardware_demo()` instead continued after that quiesce with dozens
  of BAR0 diagnostics, trap reads/clears, FECS/GPCCS reads, LTC operations, and
  the BAR1 output read.  Depending on the teardown revision, quiesce had
  already stopped FECS keepalive, reset/gated engines, or cleared PCI bus
  mastering.  The first subsequent synchronous RPC could therefore receive
  the observed PCIe Completion Abort.  This directly explains both the stable
  semaphore-only history and the full-add-only crash onset.

### Fix

- Successful `submit_launch()` now returns immediately when the semaphore
  reaches its done value.  It performs no quiescence.  The caller completes all
  post-launch diagnostics and reads/validates the GPU-written output while the
  device remains in the same responsive state used by the completion poll.
- `NVDevice.close()` is now the sole normal quiesce owner, reached from the
  existing `finally` only after output handling completes.  No caller may issue
  MMIO after it.
- The semaphore-timeout path no longer quiesces and then executes hundreds of
  diagnostic BAR reads.  It raises immediately; `NVDevice.close()` performs
  the one teardown.  The old deep block is unreachable and retained only as
  temporary register documentation.
- Quiesce itself stops/joins keepalive, performs no BAR MMIO, clears PCI bus
  mastering once, and marks shutdown complete so close/fini issue zero further
  endpoint frames.

### Offline validation

- Python compilation: pass.
- Middle self-test: pass, `launch_words=39`.
- 256-element software add: pass.
- `git diff --check`: pass.
- A live test must start from a physically power-cycled enclosure; rebooting
  macOS alone does not clear the persistent endpoint state demonstrated by the
  delayed panics.

## Session 2026-07-14 — 11:35 panic isolates the custom close RPC

### New crash evidence

- The next full run still panicked at 11:35:18.  The report is
  `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-113518.0002.panic`.
- The controller condition is unchanged: AppleT8103PCIeC reported Completion
  Abort `0x200000 out of 0x220000` at `APCIECPort.cpp:2056`.
- Python PID 6438 (about 615 MB resident) and TinyGPU PID 6440 were alive.
  Python had exactly one thread.  Its exact Python 3.14 images symbolize to
  `_socket.sock_recv_into` / `sock_recv_guts`, waiting for a synchronous
  TinyGPU response.
- The missing FECS helper thread is decisive for this revision: it exists
  throughout submission, semaphore polling, post-launch handling, and output
  read.  `_quiesce_channel()` stops and joins it before its former only
  endpoint operation, `_disable_pci_bus_master()`.  The rejected synchronous
  request is therefore in the added PCI config close sequence, not the bulk
  BAR1 output read.

### Git/known-good comparison

- The working `examples/add.py` RTX 3080 transport does not implement a custom
  PCI shutdown in `fini()`.  It reuses the stable `/tmp/tinygpu.sock` server
  and leaves the signed DriverKit server lifecycle alone.
- Commit `6b04d4f`, the last committed Kepler diagnostic path without a
  recorded panic, likewise only closed its client socket.  The current worktree
  had diverged by creating a unique temporary socket/server per invocation,
  issuing config read/write/read after the helper joined, and terminating the
  server during close.
- The panic process list also contains more than one
  `org.tinygrad.tinygpu.driver2` process, reinforcing that Kepler-specific
  server detach/rebind churn is the wrong lifecycle to add on top of the
  already-proven signed TinyGPU transport.

### Fix

- Restored the proven transport lifecycle: the Kepler client now defaults to
  `/tmp/tinygpu.sock`, reuses the shared server, closes only its own socket,
  and never terminates/relaunches the TinyGPU server during normal or partial
  cleanup.
- `_quiesce_channel()` now only stops and joins the host FECS helper.  It emits
  zero BAR, config-space, reset, or server-lifecycle RPCs.  `NVDevice.close()`
  likewise emits zero endpoint RPCs before client socket close.
- Normal successful add no longer runs the large post-launch trap/status/W1C
  sweep or changes FE power state.  That state-mutating capture is gated behind
  `KEPLER_UNSAFE_POSTLAUNCH_DIAGNOSTICS=1`.  The normal path goes directly from
  the serialized, WFI-completed semaphore to one bulk BAR1 output read.
- Removed the host-driven LTC flush/invalidate after completion.  Mesa's NVE4
  launch sequence uses `NV50_GRAPH_SERIALIZE`; this QMD also requests its
  release membar before the WFI semaphore.  The extra BAR0 flush was not part
  of the normal command/fence path and enlarged the post-completion failure
  surface.

### Safety status

- No hardware operation was made while applying this fix.  The enclosure must
  be physically power-cycled before any validation because this panic left the
  endpoint/bridge state stale.
- Offline validation passes: `py_compile`, `--middle-selftest` (including the
  zero-RPC client-only close and stable-socket assertions), the 256-element
  software demo, and `git diff --check`.
- A future test must be exactly one default full-add invocation.  Do not enable
  unsafe post-launch diagnostics and do not issue a follow-up probe.

## Session 2026-07-14 — P0 completion-abort isolation patch

### Source reconciliation

- The post-launch trap sweep, trap W1C writes, `FE_PWR=AUTO`, and `0x260=1`
  remain available only behind the explicitly dangerous
  `KEPLER_UNSAFE_POSTLAUNCH_DIAGNOSTICS=1` mode.  The normal `full-add` path
  performs none of them.
- After the completion semaphore, normal `full-add` now arms an exact RPC
  budget for one `MMIO_READ` of BAR1 at the output range.  Any BAR0 read/write,
  config request, reset, mapping request, wrong BAR1 range, retry, or second
  protocol frame raises in Python before `sendall()`.
- The transport freezes immediately after the successful output read.  Output
  validation, helper stop, optional hold-open sleep, and client close are local
  operations and cannot emit another endpoint request.

### Transport and lifecycle enforcement

- `_rpc()` records unbuffered `BEGIN` and `END` lines through an already-open
  `os.write()` file descriptor.  Records include sequence, monotonic timestamp,
  phase, thread ID, command name and ID, BAR, offset, size, config offset,
  payload length/hash, result, byte count, exception, and duration.
- `KEPLER_SINGLE_THREAD_RPC=1` is mandatory for hardware launch.  `_rpc()`
  asserts the owner thread, and both FECS helper threads are disabled in this
  mode.  The main semaphore loop performs the FE power keepalive until the
  semaphore completes.
- High-level phases now include `connect`, `map-bars`, `vbios-devinit`,
  `firmware-load`, `channel-build`, `golden-context`, `runlist-submit`,
  `semaphore-poll`, `output-read`, `host-helper-stop`, `hold-open`, and
  `client-close`.
- `KEPLER_HOLD_OPEN_SECONDS` freezes the transport, retains the Python process,
  socket, mappings, page tables, RAMIN, USERD, GPFIFO, and allocations, and
  sleeps without endpoint RPCs before client-only close.
- The client no longer auto-starts TinyGPU.  Exactly one external shared server
  must already own `/tmp/tinygpu.sock`; the script never launches or terminates
  it.  No TinyGPU server or socket was present during this offline patch, so a
  live run is currently forbidden by the test plan.

### Deterministic launch selection

- `KEPLER_SUBMIT_MODE` accepts `gpfifo`, `bypass`, or `dispatch`.  The normal
  `gpfifo` mode cannot automatically fall through to either experimental path.
  BYPASS stops if `SET_OBJECT` does not bind instead of injecting DISPATCH.
- `KEPLER_TEST_STAGE` currently implements `sem`, `set-object`, and `full-add`.
  Legacy `KEPLER_TEST_SEM_ONLY` and `KEPLER_TEST_SET_OBJECT` variables are
  rejected.  Constant-store and reduced-width add stages remain P1 work and
  are rejected before hardware initialization rather than silently mapped to
  a different test.
- Hardware launch also refuses `DEBUG != 0` or a missing `KEPLER_RPC_TRACE`.

### Offline enforcement and remaining external work

- `--middle-selftest` uses a fake TinyGPU socket to prove that a non-owner
  thread emits no frame, the final budget permits exactly one bulk BAR1 read,
  BAR0/config/reset/extra frames are rejected, freeze permits zero frames, and
  socket close is local-only.  It also verifies matching flight-recorder
  `BEGIN`/`END` and `FREEZE` records.
- Dormant `_disable_pci_bus_master()`, `_reset_pci_function_for_close()`, and
  `_shutdown_kepler_pci_for_close()` helpers were removed from the live script
  to prevent accidental reintroduction into close.
- Matching server-side DriverKit flight recording is not implemented here:
  the TinyGPU server source is not present in this repository.  That must be
  patched and rebuilt in its owning repository before server-side attribution
  is available.
- No live GPU command, probe, reset, or TinyGPU process was started during this
  session.

## Session 2026-07-14 — first corrected live run panicked during channel-build

### Exact run

- After replugging the enclosure, one invocation was run with
  `KEPLER_LIVE_ACK=completion-abort-risk`, `KEPLER_SUBMIT_MODE=gpfifo`,
  `KEPLER_TEST_STAGE=full-add`, `KEPLER_SINGLE_THREAD_RPC=1`,
  `KEPLER_HOLD_OPEN_SECONDS=120`, `DEBUG=0`, and a persistent RPC trace.
- The client connected to the shared `/tmp/tinygpu.sock` server.  This was the
  first corrected run after fixing `_temp_sock()` to use the literal `/tmp`
  path; the earlier failed attempt never connected to hardware.
- Source commit: `eec69db1d7e887bdb79dbf936159f02bc7e8d126`.
- `examples_kepler/add.py` SHA-256:
  `61acbb9f0a4f77d437b73527ae8d19f86fae3ad3fd3a53c3f9e19fcfa79a4a8a8`.
- Program log: `logs/add-20260714-124500.log`.
- Client trace: `logs/rpc-20260714-124500.log` (577,654,916 bytes).
- Panic report: `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-124607.0002.panic`.

### Result and plan comparison

- macOS panicked with the same AppleT8103PCIeC `0x200000` Completion Abort.
  The panic report contains Python PID 5626, TinyGPU PID 5445, and the
  `org.tinygrad.tinygpu.driver2` processes.
- The run did **not** reach the post-completion path that P0 was intended to
  isolate.  The program log stops after GR context/pagepool/bundle raw-zero
  stores.  The final trace phase is `channel-build`; it contains no
  `runlist-submit`, `semaphore-poll`, `output-read`, `FREEZE`, or hold-open
  phase.
- The trace has no unmatched `BEGIN`: the last recorded RPC completed before
  the delayed panic.  It ended with roughly 1,006,000 BAR0 RPCs in
  `channel-build`, followed by about 50 seconds before the 12:46:07 panic.
  This points to autonomous GPU/DriverKit activity or a delayed PCIe fault
  from channel construction, not a post-output close RPC.
- Therefore the run followed the §13 command envelope and the transport
  prerequisites, but it did not validate §8 Stage E or the P0 output-freeze
  boundary.  The next step must be the controlled ladder in §8, beginning with
  a fresh power-cycle and a smaller Stage B/C channel-free or semaphore-only
  invocation; do not repeat full-add merely because the process stopped.

## Session 2026-07-14 — semaphore-only Stage C survived hold and close

- After another physical replug, the shared server was restarted on
  `/tmp/tinygpu.sock` and exactly one `KEPLER_TEST_STAGE=sem` invocation was
  run with `DEBUG=0`, `KEPLER_SUBMIT_MODE=gpfifo`, single-thread RPC, and a
  120-second hold.
- The run reached `hardware_sem=ok value=2`, froze the transport, stopped the
  host helper, held all mappings alive for 120 seconds with zero endpoint RPCs,
  and closed the client socket cleanly with exit status 0.
- Trace phase order was `channel-build` → `golden-context` →
  `runlist-submit` → `semaphore-poll` → `FREEZE sem-complete` →
  `host-helper-stop` → `hold-open` → `client-close`.
- No new panic report appeared.  This is a successful §8 Stage C result and
  demonstrates that the post-semaphore freeze/hold/client-close boundary is
  stable for the semaphore-only path.
- Program log: `logs/add-20260714-125311-sem.log`.
- Client trace: `logs/rpc-20260714-125311-sem.log` (779,959,159 bytes).
- The trace is very large because the current channel/golden-context setup
  emits roughly 1.4 million BAR0 frames before the semaphore.  This is within
  the pre-completion scope of the plan, but should be reduced or sampled before
  repeated live testing.

## Session 2026-07-14 — SET_OBJECT stage panicked in channel-build

- After another physical replug, exactly one `KEPLER_TEST_STAGE=set-object`
  invocation was run with the shared `/tmp/tinygpu.sock` server,
  `KEPLER_SUBMIT_MODE=gpfifo`, `KEPLER_SINGLE_THREAD_RPC=1`,
  `KEPLER_HOLD_OPEN_SECONDS=120`, and `DEBUG=0`.
- macOS panicked again with Completion Abort `0x200000` at 13:24:40.
  Panic report: `/Library/Logs/DiagnosticReports/panic-full-2026-07-14-132440.0002.panic`.
  Python PID was 3023 and TinyGPU PID was 3014.
- The program stopped during the same GR buffer/channel construction area;
  it never reached SET_OBJECT submission, semaphore completion, freeze, or
  hold-open.  Program log: `logs/add-20260714-132301-set-object.log`.
- This run demonstrates the flight recorder's intended failure boundary.  The
  final unmatched request is:

  `BEGIN seq=902015 phase=channel-build cmd=MMIO_WRITE bar=0 offset=8088260 size=4`

  There is no matching `END`, so this BAR0 write is the operation that was in
  flight when the controller panic occurred.  Client trace:
  `logs/rpc-20260714-132301-set-object.log`.
- Therefore the controlled invocation procedure was followed, but the system
  is not yet safe for another live run.  The repeated fault is pre-semaphore
  channel construction, not the post-completion lifecycle.  Stop live testing
  and isolate or remove the GR/channel-build BAR0 write sequence before any
  further replugged invocation.

### Exact PRAMIN mapping for the SET_OBJECT-stage panic

- BAR0 offset `0x7b6ac4` is not a standalone register literal.  It is the
  `0x700000 + (pa & 0xfffff)` aperture used by `_gk104_pramin_write_literal()`;
  the preceding request writes the PRAMIN window selector at `0x1700`.
- The contiguous aperture-write run begins at `0x70d000`, which is the start
  of the VRAM-backed attrib allocation (`pa=0x50d000`).  The unmatched request
  is therefore attrib dword `0xa9ac4`, physical address `0x5b6ac4`, payload
  `0xffffffff`, during `repair_zero("attrib", ...)`.
- The process log ends immediately after the GR-context, pagepool, and bundle
  zero repairs; it never prints attrib completion.  `SET_OBJECT` was not
  reached.  The next code work should label and isolate this attrib PRAMIN
  zeroing sequence (or replace it with a bounded alternative) offline before
  another hardware invocation.

### Offline isolation patch

- `add.py` now labels each `repair_zero()` operation in the RPC phase field and
  accepts the explicit `KEPLER_ATTRIB_REPAIR=bar1` experiment.  The default
  remains `literal`; the opt-in mode bypasses only the attrib buffer's
  per-dword BAR0 PRAMIN stream and uses the existing page-bounded BAR1 writer.
- `py_compile`, `--middle-selftest`, the software backend demo, and scoped
  `git diff --check` pass.  No live invocation has been started after this
  patch.

## Session 2026-07-16 — cold GDDR5 / PRAMIN / `KEPLER_N=8`

Operator notes also live in `examples_kepler/reset_egpu.md` (night–night17).
Goal this arc: cold bring-up far enough for `hardware_demo=ok N=8`.
**Not yet achieved** — furthest point: **FECS ready + deferred bit0 + enabled
BAR1**, but pre-bit0 PRAMIN stores do not create valid physical VRAM roots.

### Proven on silicon

- **Host / MEMX-WR32 of `0x1620`/`0x26f0` kills TinyGPU BAR0** when clearing the
  full Nouveau pause masks (`0xaa2`).  MEMX ENTER is equally hostile.
- **bit0-only clear of `0x1620[0]`** keeps BAR0 alive on clean cold; turns
  PRAMIN `0xbad0fb` → virgin `0xffffffff` (night5).  No `0x26f0`, no unpause.
- **all-MEMX** reaches `MEMX WR32 totals: 78 execs, 210 words`.
- **Defer bit0** until after PGRAPH/FECS (`KEPLER_RAM_BIT0_DEFER=1`): full
  PGRAPH pack + **FECS IMEM match + FECS ready** (`FE_PWR=0x2`,
  `gpc=4 tpc=[2,2,2,2]`) on true cold (night13–17).
- **After deferred bit0, host `0x1700` alone kills BAR0** (night14) — cannot
  use host PRAMIN window setup.  **MEMX WR32** is the intended PRAMIN path
  (`KEPLER_PRAMIN_MEMX=1`), assuming virgin XOR (`0xffffffff ^ wanted`).
- **After deferred bit0, PMU host MMIO is inaccessible**, not merely wedged:
  night17 read `PMC_ENABLE`, `0x10a10c`, and the PMU ring as `0xffffffff`.
  The correct Nouveau MC-level PMU reset through `PMC_ENABLE[13]` did not
  restore `0x10a10c`.  Therefore all required roots must be staged before the
  bit0 crossing.
- **Night18 disproved post-bit0 work inside that same script:** the queued
  order was `0x1700`, bit0, then PRAMIN roots.  FECS was ready, but the first
  post-transition Falcon `nv_wr32(0x700000...)` collapsed all BAR0
  (`PMC_BOOT_0=0xffffffff`).  Local `ref/linux_drm/.../pmu/fuc/memx.fuc`
  confirms WR32 pairs execute serially; the PMU cannot service a following
  MMIO store once bit0 has hidden its aperture.
- **Night19 validated pre-staging:** the `0x5c` root script returned normally,
  host bit0 kept `PMC_BOOT_0=0x0e4040a2`, and PMU was no longer needed.  The
  next failure was our own ordering error: `0x070000` BAR flush was issued
  while `0x1704` was still disabled and its status reads became 43-ms
  timeouts.  Local Nouveau orders `gf100_bar_bar1_init()` (enable `0x1704`)
  before `gf100_bar_bar1_wait()` (two flushes); the atomic branch now matches
  that order and removes its unnecessary first-activation VMM invalidate.
- **Night20 proved the whole `0x070000` block is post-bit0-inaccessible:**
  `0x1704` enable completed in 9 us, but the immediately following BAR flush
  again produced 43-ms status reads and timed out.  The new `0xa4` pre-bit0
  MEMX script now stages roots, enables `0x1704`, and performs both BAR
  flush/wait cycles before the host bit0 clear.  After the crossing, the
  TinyGPU-only path uses exact BAR1 readback as its posting barrier and never
  touches `0x070000/0x070010/0x070004`; shared/Linux behavior is unchanged.
- **Night21 reached the first real BAR1 read:** the complete `0xa4` pre-bit0
  roots+enable+flush script returned, bit0 kept BAR0 live, and BAR1 returned
  structured walk-failure sentinels (`0xbad0fb03`, `...12`, `...13`, `...14`)
  instead of the two control PTEs.
- **Night22 eliminated PRAMIN encoding as the variable:** it staged two
  complete trees in separate bit19-safe banks—literal at
  `0x200000/0x210000/0x220000` and virgin-XOR at
  `0x100000/0x110000/0x120000`—and pre-enabled/flushed both.  Literal returned
  the same `0xbad0fbxx` walk-failure sentinel; switching `0x1704` to XOR
  returned all ones.  Because night21 already tested XOR while active and got
  the sentinel, neither encoding populated physical VRAM.  Pre-bit0 PRAMIN is
  therefore an acknowledging stub, not a usable storage path.
- **Patched for night23:** use the PMU Falcon's documented external transfer
  engine instead of PRAMIN.  MEMIF port 0 is set to direct VRAM (`TYPE=4`);
  the 40 root bytes are DMA-stored and independently DMA-loaded into a second
  DMEM scratch area for exact comparison.  Only after all comparisons pass do
  we enable/flush BAR1 pre-bit0 and perform the proven host bit0-only clear.
  `ref/cmpunlocker` confirms a fixed/aligned Falcon DMA target pattern but
  delegates transfer setup to GSP booter; the actual GK104 sequence comes from
  local `ref/envytools/docs/hw/falcon/xfer.rst` and `rnndb/falcon.xml`.
- **Night23 true-cold reached the first PMU DMA request:** FECS was ready and
  topology was again 4×2.  The store command was accepted (`CTRL=0x220`) but
  stayed at `STATUS=0x10012`—data transfer busy with exactly one store pending.
  This is an activation stall, not a rejected command or bad address.
  **Patched for night24:** set documented MEMIF.CTRL `ENABLE` (bit4) and
  `IGNORE_ACTIVATION` (bit7) before selecting port0 direct VRAM.  Nouveau also
  sets `IGNORE_ACTIVATION` before host-originated Falcon DMA on supported
  engines.  Readback of both control bits is mandatory, and the timeout is
  reduced from 500 ms/~25k reads to 50 ms with 500-us polling.
- **Night24 activated PMU MEMIF successfully:** control read back
  `0x110→0x190`, port0 read back `0x110→0x114`, and both the first DMA store
  and load completed.  Loadback was sixteen `ff` bytes, proving the engine and
  target selection work but pre-bit0 physical VRAM discards stores just like
  PRAMIN.  Bit0 was not crossed.
- **Night25 crossed bit0 with FECS-staged roots, then lost FECS host MMIO:**
  `XFER_CTRL`/`XFER_STATUS` became `0xffffffff` immediately after the proven
  host bit0-only clear — the same permanent aperture loss already seen on
  PMU.  Host-triggered Falcon DMA cannot complete the post-bit0 store on
  either engine.  Card left dirty; no retry.
- **Night26 ran autonomous FECS but BAR1 stayed virgin `ff`:** roots staged,
  bootstrap armed, bit0 kept `PMC_BOOT_0` live, then two-PTE readback was
  `ffffffffffffffff`.  Entry was at `0xc20`, outside the live
  `gk104_fecs_code.bin` IMEM window (`0xc00`; pad is `0xb0a..0xbff`).
- **Night27 aborted before bit0:** runtime IMEM store at `0xb20` read back
  `0xbadf5000` after FECS had already run.  Bit0 was not crossed.
- **Patched for night28:** embed bootstrap into `_fecs_code` before the
  initial `falcon_write_imem`; arming is halt→ENTRY=`0xb20`→START→PC-in-pad
  only (no runtime CODE write).
- Also avoid: early LTC/ZBC after RAM; BLCG `0x4041f0`; PRAMIN literal `0`
  fallback; dirty GPC-awake+PRAMIN-stub without `ALLOW_DIRTY`.
- **VRAM bit19 alias**: `KEPLER_VRAM_BIT19_SAFE=1`.

### Furthest live cold (night13–17)

1. all-MEMX 78/210 → bit0 deferred → skip LTC/BLCG
2. full PGRAPH pack OK → **FECS ready** (topology 4×2)
3. deferred bit0 OK (`PMC_BOOT_0` live)
4. **Blocker:** PRAMIN/BAR1 bootstrap — host `0x1700` is fatal and neither
   internal nor MC-level PMU reset restores host PMU MMIO after bit0.
5. Night18 `0x1700` + bit0 + roots ordering killed BAR0 before BAR1 enable.
6. **Patched for night19:** stage only the 10 required root dwords
   (instance `+0x200/+0x208`, PDE, two PTEs) in a `0x5c` MEMX script and wait
   for its reply while PMU is live; only then do the proven host bit0 clear.
   Expand the 16-MiB identity SPT through verified BAR1.
7. Night19 proved step 6 through the bit0 crossing; failed at a pre-enable BAR
   flush.  **Patched for night20:** enable `0x1704` first, then flush twice,
   matching Nouveau exactly.
8. Night20 enabled `0x1704`, but post-bit0 `0x070000` still timed out.
   **Patched for night21:** do roots + enable + both flush/waits in one
   pre-bit0 `0xa4` MEMX script, then use BAR1 readback only after bit0.
9. Night21 completed step 8; BAR1 walk returned `0xbad0fbxx` for the XOR tree.
10. Night22 tested complete literal and XOR roots; neither produced a valid
    walk.  **Patched for night23:** bypass PRAMIN with PMU direct-VRAM DMA,
    require DMA store→load equality for all 40 root bytes, then enable/flush
    BAR1 and cross bit0.
11. Night23 accepted the first 16-byte DMA store but left it pending because
    MEMIF was not activated.  **Patched for night24:** enable MEMIF and ignore
    channel activation before the transfer; require exact control readback.
12. Night24 completed PMU DMA but read back virgin all-ones: pre-bit0 VRAM is
    a discard stub independent of aperture.  **Patched for night25:** stage in
    FECS `xfer_data`, cross bit0, then FECS-DMA store/load roots while VRAM is
    live, before enabling BAR1.
13. Night25 staged FECS roots and crossed bit0 with live `PMC_BOOT_0`, then
    immediately saw FECS `XFER_CTRL/STATUS=0xffffffff`.  Host-triggered Falcon
    DMA is dead after bit0 on both PMU and FECS.  **Patched for night26:**
    autonomous FECS path — patch `fecs_bar1_bootstrap` into IMEM pad, arm it
    (halt→entry→start) before bit0, host-clear bit0, let the Falcon `xdst`
    the three root fragments, then enable `0x1704` and verify via BAR1.
14. Night26 ran the autonomous path end-to-end on a true-cold card: FECS ready,
    roots staged, bootstrap "armed", bit0 kept `PMC_BOOT_0` live, then BAR1
    read back `ffffffff` (virgin).  Root cause: entry was at `0xc20`, past the
    live `gk104_fecs_code.bin` IMEM window (`0xc00` bytes; last live byte
    `0xb09`; zeros `0xb0a..0xbff`).  **Patched for night27:** relocate to
    `0xb20`, require IMEM patch readback + PC in-pad before bit0, wait 100 ms
    after bit0 before `0x1704`.
15. Night27 reached FECS ready and staged roots, then aborted before bit0:
    runtime IMEM store at `0xb20` read back `0xbadf5000`.  **Patched for
    night28:** embed the bootstrap into `_fecs_code` before the initial
    `falcon_write_imem`; arming only halt→ENTRY=`0xb20`→START→PC-in-pad.
16. Night28 embedded+armed (`pc=0xb28` in delay), bit0 OK, BAR1 still
    virgin `ff`.  Fixed delay finished and `xdst` ran *before* host bit0
    (pre-bit0 VRAM discard).  **Patched for night29:** poll `0x1620` via
    `nv_rd32` until bit0 clears, then `xdst`.
17. Night29 abort before bit0: PC samples stayed `0x567` under a pad-only
    check.  Poll spends its time in `nv_rd32` at `0x68`, outside the pad.
    **Patched for night30:** accept PC in pad *or* `nv_rd32` range; poke
    `UC_PC` while halted before START.
18. Night30 true cold reached FECS ready, then TinyGPU `Broken pipe` during
    LTC sysmem map (before BAR1 arm).  Bit0 not crossed.
19. Night30b true cold: ENTRY=`0xb20` but after START CPUCTL=`0x20` (SLEEPING)
    and PC stuck at main `0x567` — `nv_rd32(0x1620)` poll leaves the pad and
    traps back to main.  **Patched for night31:** long in-pad delay only
    (`sethi 0x10000000`, no calls; host waits 5s after bit0).  Each of the
    three root stores is now serialized with its own `xdwait`; assembled code
    is 80 bytes at `0xb20..0xb70`.  Shared and wrapper middle selftests,
    software demos, and mmiotrace 24/24 pass.
20. Night31 true cold: bootstrap started in the local loop (`pc=0xb29`), host
    crossed bit0 with live `PMC_BOOT_0`, but after the 5-second wait BAR1 was
    still virgin `ff`.  No channel launch/retry followed.  At the cold
    FECS/HUB clock, `0x10000000` iterations can exceed the host wait.
    **Patched for night32:** `0x01000000` iterations and a 10-second host wait;
    keep the per-transfer `xdwait`s.  Observed START-to-bit0 latency is only
    about 1.4 ms, so this leaves a wide pre/post-bit0 timing bracket.
21. Night32 true cold: the shortened loop started at `pc=0xb29`, bit0 crossed
    cleanly, and the complete 10-second wait still ended with virgin `ff` BAR1
    roots.  Delay duration is not the blocker.  The remaining ordering bug is
    that nights26-32 wrote FECS `xfer_data` and `MEM_BASE/MEM_TARGET` while the
    firmware was running, then called `falcon_stop()` afterward.  **Patched
    for night33:** halt first; stage and exactly read back every DMEM root while
    stopped; configure/read back the transfer target; then START at `0xb20`
    without another stop.  A staging mismatch aborts before bit0.
22. Night33 true cold: halt-first guard worked and aborted before bit0.  The
    first two staged instance dwords read back exactly, but the next two were
    `0xbadf5000`: wanted
    `0000110000000000ffffff0000000000`, got
    `00001100000000000050dfba0050dfba`.  The cold FECS DMEM port cannot be
    trusted to autoincrement across this transfer.  **Patched for night34:**
    explicitly program `DATA_INDEX` for every staged and read-back dword,
    retaining halt-first ordering and exact pre-bit0 verification.
23. Night34 true cold: explicit `DATA_INDEX` still returned `0xbadf5000` for
    all four instance-root dwords while halted, and the guard again aborted
    before bit0.  This is post-init FECS host-DMEM aperture loss, not an
    autoincrement error.  **Patched for night35:** remove host DMEM staging
    entirely.  The embedded Falcon pad now constructs all ten fixed root
    dwords locally in `xfer_data`, then performs the three serialized `xdst`s.
    Host code only validates the fixed root/layout constants and configures
    `MEM_BASE/MEM_TARGET` after halt.  Bootstrap is 156 bytes at
    `0xb20..0xbbc`; assembly/Python sync, middle, software, and mmiotrace 24/24
    pass.
24. Night35 true cold: Falcon-local root construction ran, but arm reported
    `pc=0xb2f`—the first instruction *after* the `0xb29..0xb2e` delay loop—
    before host bit0.  The short loop had already expired, so local stores and
    `xdst` could reach the pre-bit0 discard stub; BAR1 remained `ff` after the
    crossing.  **Patched for night36:** restore `0x10000000` iterations,
    accept only PC in `0xb29..0xb2e` (anything later aborts before bit0), and
    wait 30 seconds after crossing.  Falcon-local root construction and the
    per-transfer `xdwait`s remain.
25. Night36 true cold: arm sampled `pc=0xb29` inside the guarded long delay,
    bit0 crossed cleanly, and BAR1 was still virgin `ff` after the complete
    30-second wait.  This rules out both pre-bit0 timing and host-DMEM root
    staging.  **Patched for night37:** the running Falcon now programs its own
    `MEM_BASE=0x1000` and `MEM_TARGET=0x80000002` immediately before `xdst`,
    matching Nouveau `hub.fuc` `ctx_load` ordering instead of relying on target
    state written through the host aperture before START.
26. Night37 true cold: Falcon-local target programming also armed in the long
    loop at `pc=0xb29`; bit0 stayed live, but the two control PTEs were still
    all `ff` after 30 seconds.  Recomparison with Nouveau found the host path
    omitted `gf100_bar_bar1_wait()` after enabling `0x1704`.  **Patched for
    night38:** reproduce the full golden-trace activation order after `xdst`:
    flush → HUB-only PDB invalidate (`0x100cb8/0x100cbc`) → flush → enable
    `0x1704` → flush twice.  Do not read the post-bit0-inaccessible status
    registers; night20 proves trigger writes complete while only reads stall.
27. Night38 true cold reached the complete post-`xdst` activation sequence,
    but the first following `PMC_BOOT_0` read took 43 ms and returned `ff`.
    TinyGPU reported every trigger write as accepted, so RPC completion does
    not imply the post-bit0 BAR/LTC/VMM block survived.  **Patched for night39:**
    while Falcon PC is still in the guarded delay and bit0 remains set, enable
    `0x1704` and complete Nouveau's two readable `0x070000` flushes.  Then clear
    bit0, let the serialized `xdst`s create the roots, and make BAR1 readback
    the first page-table lookup; issue no post-bit0 BAR0 writes.
28. FECS physical DMA still has a circular prerequisite: Nouveau only uses
    FECS direct-memory transfers after `ctx_load` activates a real channel, but
    BAR1 roots are required before that channel can exist.  Night24 already
    proved PMU MEMIF `ENABLE|IGNORE_ACTIVATION` completes channel-independent
    transfers.  **Patched for night40:** embed a PMU v4 pad routine at
    `0xb20` (`pmu_bar1_bootstrap.fuc`) that builds the same roots in local
    DMEM and `xdst`s through port0 direct VRAM after the long delay; host arms
    MEMIF, observes PC in the delay, pre-activates BAR1, clears bit0, then
    waits.  Offline middle/fake path now asserts the PMU image pad, not FECS.
29. Night40 true cold reached FECS ready + topology `4x2`, then aborted in
    `_gk104_pmu_dma_prepare`: soft `falcon_stop(PMU)` never posted
    `CPUSTAT.STOPPED` (1 s of `0x10a128` polls).  Bit0 was not crossed.
    **Patched for night40b:** on soft-stop timeout, MC-reset the PMU
    (`PMC_ENABLE[13]`), reload the embedded image with `start=False` at
    ENTRY=`0xb20`, then configure MEMIF and START the pad.
30. Night40b true cold: soft-stop missed (`CPUCTL=0x20` SLEEPING), MC reload
    + MEMIF OK (`port=0x114 ctrl=0x190`), then arm aborted because host
    TRACEPC `0x10aff0` stayed `0` (unlike FECS).  **Patched for night40c:**
    match `gt215_pmu_init` start (scrub clear → ENTRY → START, no HALT
    pulse); verify IMEM pad dword; accept TRACEPC-dead + non-halted/non-
    stopped CPUCTL as armed.
31. Night40c true cold: arm accepted with `TRACEPC=0`, pre-bit0 BAR1 + bit0
    kept `PMC_BOOT_0` live, but BAR1 control PTEs were `bad0fb` walk
    sentinels — roots never landed.  False arm likely (pad never ran).
    **Patched for night40d:** rebuild pad to construct roots and publish
    DMEM magic `0x40c0b002` at `0xd7c` *before* the long delay; host must
    read magic + exact root bytes before crossing bit0.
32. Night40d true cold: magic+roots proven, bit0 live, BAR1 still `bad0fb`.
    Pad ran but `xdst` still missed live VRAM (likely finished delay before
    bit0).  **Patched for night40e:** host GO at `0xd78` after BAR1 activate;
    falcon then delays and `xdst`s with idiomatic `$xdbase=(pa>>8)`.
33. Night40e: GO written, bit0 live, BAR1 still `bad0fb`.  Likely `cmpu`
    wait_go never exited.  **Patched for night40f:** `sub`-based wait_go,
    DELAY_ENTERED magic at `0xd74` (host must see it before bit0), pad at
    `0xb14` to fit.
34. Night40f true cold: `delay-entered=0x40c0b004` proven, bit0 live,
    BAR1 still `bad0fb`.  Timing/wait_go are ruled out — Falcon reaches the
    post-GO delay, but `xdst` still does not produce BAR1-visible roots.
35. **Night40g true cold:** host-staged DMEM roots + Falcon IO `XFER_*`
    (`CTRL=0x220`/`0x120`, pad end `0xbf4`).  Handshake identical
    (`magic`/`GO`/`delay-entered`, bit0 live `PMC_BOOT_0`), BAR1 still
    `bad0fb` (`wanted=01120000…01100000…` vs `actual=03fbd0ba…`).  So uc
    `xdst` vs IO XFER is not the differentiator — post-MC-reset PMU MEMIF
    stores are not BAR1-visible.
36. **Night40h true cold:** live IMEM pad patch verified (`0xb14..0xbf4`,
    no MC reset), MEMIF `0x114/0x190`, full handshake
    (`magic`/`GO`/`delay-entered`), bit0 live — BAR1 still `bad0fb`.
    Live-vs-MC-reload is ruled out; post-bit0 PMU MEMIF stores are not
    BAR1-visible on either engine state.
37. **Night40i true cold:** `ENABLE|IGNORE` validated then IGNORE cleared
    (`ctrl=0x110` ENABLE-only), live IMEM patch, full handshake, bit0 live —
    BAR1 still `bad0fb`.  IGNORE_ACTIVATION is not the missing FB-commit
    switch.
38. **Night40j true cold:** compact `xdst` + `call wr32@0x34` HUB
    `PAGE_ALL` PDB invalidate after stores; live IMEM patch; full handshake;
    bit0 live — BAR1 still identical `bad0fb`.  Negative-walk cache is
    ruled out (or wr32 invalidate is a no-op); roots are still not
    BAR1-visible.
39. **Night40k true cold (partial):** after Nouveau-motivated ENABLE-only
    arm, host MEMIF store hung `CTRL=0x220 STATUS=0x10012` (one store
    pending) — same signature as night23 without activation.  **Bit0 was
    not crossed.**  Conclusion: channel-less PMU MEMIF requires
    `IGNORE_ACTIVATION` to retire transfers (`gm200_flcn_fw_load` no-inst
    sets IGNORE).  Night40i ENABLE-only was therefore a DMA hang, not a
    silent FB commit.  Code restored to ENABLE|IGNORE; host verify +
    falcon fallback kept for the next cold run.
40. **Night40k power-cycle cold:** `ENABLE|IGNORE` host MEMIF completed;
    all three pre-bit0 loadbacks were `ff` (night24 discard reproduced).
    Falcon fallback: live IMEM patch, full handshake, bit0 live — BAR1
    still `bad0fb`.  Pre-bit0 physical VRAM discards PMU MEMIF stores even
    with IGNORE; post-bit0 falcon xdst still not BAR1-visible.
41. **Night40l true cold:** skipped host pre-bit0 `0x1704`; falcon
    `wr32` enable after xdst; handshake+bit0 OK; BAR1 still `bad0fb`
    (`03fbd0ba06…` — walk failure, so BAR1 was likely enabled).  Pre-bit0
    enable is not the sole cause; roots still never become BAR1-visible.
42. **Night40m true cold:** fire-and-forget MEMX `DELAY(100ms)` +
    virgin-XOR PRAMIN `WR32` roots + `0x1704` + two flushes; bit0 during
    DELAY; `finish=1096`; BAR1 still `bad0fb`.  Post-bit0 MEMX PRAMIN
    writes are also not BAR1-visible (same class of failure as MEMIF/xdst).
43. **Patched for night40n:** MEMX `WR32` roots are now literal.  Nouveau's
    `memx_func_wr32` directly calls `nv_wr32(addr, data)`; it does not perform
    the host-PRAMIN read/modify/XOR workaround used elsewhere in this script.
    `KEPLER_BAR1_MEMX_LITERAL=0` retains the night40m encoding for comparison.
44. **Night40n true cold:** literal MEMX roots produced the identical BAR1
    `bad0fb` walk-failure sentinel.  Literal vs XOR is ruled out.  More
    importantly, fire-and-forget only proves EXEC was queued; night18 already
    showed PMU `nv_wr32` cannot safely complete after the bit0 transition.
45. **Patched for night40o:** bootstrap through unpaged physical BAR1.  Local
    envytools `bars.rst` says G80+ BAR1 maps directly to VRAM with VM disabled,
    and `pbus.xml` defines GF100 `0x1704[31]=0` as VRAM mode.  Keep `0x1704=0`,
    host-clear bit0, write+verify the 40 root bytes at physical BAR1 offsets,
    then set `0x1704=0x80000100` and validate the first paged read.  This avoids
    the circular need for valid page tables before BAR1 can create them.
46. **Night40o true cold:** direct physical BAR1 after bit0 read virgin
    `0xffffffff`, but both XOR and literal stores at physical `0x100200` were
    discarded.  Failure is below BAR1 paging: no tested post-bit0 client can
    commit physical VRAM.  The skipped framebuffer-quiesce window is now the
    leading RAM-controller defect.
47. **Patched for night40p:** reproduce Nouveau's RAM transition atomically in
    one PMU MEMX EXEC: pre-transition writes + `ENTER` + GDDR5 programming,
    delays and waits + `LEAVE`.  Previously `KEPLER_RAM_BLOCK=bit0` skipped
    ENTER/LEAVE entirely, programming the controller while live.  Standalone
    ENTER stranded host MMIO; the combined script restores the aperture before
    its reply.  Offline serialization is 46 commands / `0x448` bytes within
    the PMU's `0x800`-byte data buffer, with exactly one ENTER and one LEAVE.
48. **Night40p true cold:** the atomic script was submitted (`31` WR32 groups,
    `208` words) but the PMU EXEC reply timed out after 15 seconds.  Later review
    found the script still contained target `prog0` stores after `LEAVE`, so the
    timeout did not prove that LEAVE itself was unreachable.
49. **Patched for night40q:** precede the RAM script with one minimal synchronous
    MEMX EXEC containing only `ENTER; LEAVE`.  Review also corrected the RAM
    training status address (`0x110974`, not `0x110000`), restored Nouveau's
    `64us`/`200us` WAIT values, required the patched ENTER firmware, and moved
    target `prog0` to a separate post-transition MEMX EXEC.
50. **Night40r true cold:** minimal `ENTER;LEAVE` preflight completed; corrected
    atomic RAM transition completed (`47` commands, `0x458` bytes), and separate
    target `prog0` completed (`2` commands, `0x30` bytes).  Night40p's MEMX hang
    is fixed.  FECS reached ready and channel construction continued, but direct
    physical BAR1 root `0x100200` rejected `0x00110000` and read back
    `0xbad0fb0d`.  Correct framebuffer quiesce/program/unquiesce is therefore
    not sufficient to make physical VRAM stores stick.  **Needs enclosure power
    cycle.**
51. **Night40s true cold:** review found `_gk104_bit0_unstub()` silently skipped
    atomic mode, so night40r never performed its claimed late clear.  Allowing
    it produced the expected live `0x1620 0xaab→0xaaa`, but direct BAR1 changed
    from `bad0fb` to `0xffffffff` and every store timed out/discarded.  The PMU
    firmware explains why: ENTER clears bit0 only inside framebuffer pause and
    LEAVE deliberately restores it.  A late clear is not a valid steady state.
52. **Night40t true cold:** root PRAMIN `nv_wr32`s were moved inside one
    synchronous `ENTER→WR32→LEAVE`; LEAVE restored `0x1620=0xaab`, BAR1 enable
    and Nouveau's two flushes completed, but the first VM access still returned
    identical `bad0fb`.  Correct ENTER scope does not make MEMX PRAMIN stores
    physically visible.
53. **Patched for night40u:** replace blind PRAMIN `WR32` with verified direct-
    VRAM PMU MEMIF under the same ownership boundary.  Roots are staged in
    DMEM, `ENTER` runs, each fragment is MEMIF-stored then MEMIF-loaded to a
    separate scratch address and compared exactly, and `LEAVE` always runs in
    `finally`.  BAR1 is enabled/flushed only after all physical loadbacks match.
    The fake now rejects MEMIF root transfers outside ENTER and verifies
    `ENTER < store/load < LEAVE < enable < first BAR1 access`.


54. **Night40ac true cold:** Nouveau comparison confirmed join/PDE/PTE encodings
    match `gf100_vmm_join_` / `gf100_vmm_pgd_pde` / `gf100_vmm_valid`.  Architectural
    gap is INST: Nouveau writes BAR instance via PRAMIN/BAR2 (`NVKM_MEM_TARGET_INST`);
    we cannot.  HOST-PDB experiment staged instance+PGD+SPT in sysmem at
    `bus_base+{0x100000,0x110000,0x120000}` with HOST join/PDE and pointed
    `0x1704` at `bus_base+inst` (`ctl=0x80080100`, inst=`0x80100000`).  First BAR1
    walk returned the same `bad0fb` sentinel — PBUS BAR1 CHAN has no aperture field,
    so instance fetch is VRAM-only.  HOST escape hatch closed.  `KEPLER_BAR1_HOST_PDB`
    kept for offline coverage, default off.
55. **Patched for night40ad:** Nouveau `memx.fuc` ENTER sets `NV_PPWR_OUTPUT_SET`
    (`0x07e0`) FB_PAUSE then **waits forever** on `OUTPUT` (`0x07c0`) bit2; LEAVE
    clears and waits bit2 clear.  Our pad previously SET then immediately `xdst`
    (host ENTER_NOWAIT patches the same infinite wait out of stock MEMX).  Pad now
    bounded-polls bit2 (≤255 iters), stores `0` or `4` to DMEM `0xd6c`, then `xdst`.
    Fits exactly in IMEM `0xb14..0xc00` (MEMIF re-arm removed).  Host prints status
    after DONE.  Live verdict: if status=`4` and loadback still `ff`, pause is not
    the discard cause; if status=`0`, FB never quiesces under our ENTER masks.
56. **Night40ad true cold:** pad published `FB_PAUSE_status=0x4 (acked)` then all three
    post-LEAVE MEMIF loadbacks returned virgin `ff`.  So either (a) pause ack is real
    and physical VRAM still discards `xdst`, or (b) bit2 was **sticky** after nowait
    MEMX LEAVE during ram_program (CLR issued, wait-for-clear patched out) and the
    "ack" was not a rising edge.  Night40ae distinguishes these.
57. **Patched for night40ae:** rising-edge pause — CLR→poll-clear (`0xd68`)→SET→
    poll-set (`0xd6c`) before `xdst`.  IMEM budget forces pad to only `xdst` the
    16-byte join; PGD/SPT skipped until pause edge is proven.
58. **Night40ae true cold:** `FB_PAUSE_clear=0x0 (ok)` and `FB_PAUSE_set=0x4 (acked)`
    — rising edge is real; night40ad ack was not sticky.  Join loadback still
    virgin `ff`.  **FB_PAUSE wait is closed as the discard cause.**  Physical VRAM
    rejects falcon `xdst` even under Nouveau-correct ENTER masks + rising-edge pause.
59. **Patched for night40af:** line-by-line Nouveau INST bootstrap is
    `nv50_instobj_wr32_slow` — `0x001700 = pa>>16` then literal `0x700000+(pa&0xfffff)`
    (`instmem/nv50.c:62-71`); BAR2 (`0x1714`) is only the fast CPU map after PTs exist
    (`bar/gf100.c:72-80,130-137`).  Pad PRAMIN-writes the join via `call 0x34`
    (kernel `wr32`) under rising-edge pause instead of `xdst`.
60. **Night40af bug:** `nv_iowr` clears `$r0` (`macros.fuc:193-197`); using `$r0` as the
    PRAMIN loop counter hangs after the first dword (DONE unread).
62. **Night40ag true cold:** Nouveau PRAMIN pad DONE (`0x40c0b005`) with
    `FB_PAUSE_set=0x4`, store=`PRAMIN(0x1700/0x700000) r5-loop`, join at
    `pa=0x30200` (first 512 KiB).  Post-LEAVE MEMIF loadback still virgin `ff`.
    PRAMIN client path is Nouveau-correct; physical VRAM still rejects stores
    after cold GDDR under **nowait** atomic `ram_program`.
63. **Patched for night40ah:** Nouveau `gk104_ram_prog` → `ram_exec(true)` runs
    GDDR WR32s inside MEMX ENTER that *waits* on `OUTPUT` FB_PAUSE
    (`memx.fuc:110-115`); LEAVE waits clear (`:131-136`).  Live still loads PMU
    with `ENTER_NOWAIT` for early MEMX, then reloads stock wait bras before
    atomic `ram_program` (`KEPLER_RAM_ENTER_WAIT=1`, preflight off).  On EXEC
    timeout, fall back to nowait once.  Dump MC regs after RAM.  Offline
    selftest covers refuse-without-opt-in and allow-with-`ENTER_WAIT`.
64. **Night40ah true cold:** stock ENTER wait during atomic `ram_program`
    **completed** (no timeout): `FB_PAUSE_wait=on`, `atomic MEMX RAM transition
    completed`.  MC dump after: `0x10f808=0x7aa00050` (offline golden
    `0x72a00000`), `0x1620=0xaab`, REFPLL `0x137390=0x30001` (lock bit17 set).
    Pad still DONE+PRAMIN+pause ack; MEMIF loadback still virgin `ff`.
    **Closed:** missing FB_PAUSE wait during GDDR `ram_program` is not why
    physical VRAM discards stores.
65. **Patched for night40ai:** two Nouveau deltas still open after night40ah:
    (a) `gk104_ram_prog` does `prog0(1000)` *before* `ram_exec` then
    `prog0(target)` after — we only applied target tables pre-ENTER;
    (b) cold `0x10f808` residuals outside Nouveau's mask survive (`xor
    0x08000050`).  Default `KEPLER_RAM_PROG0_PRE_MHZ=1000` and
    `KEPLER_RAM_808_ABSOLUTE=1`.
66. **Night40ai true cold:** `0x10f808 absolute 0x7aa00050->0x72a00000` and
    MC dump confirms `0x10f808=0x72a00000`; ENTER wait still completes; pad
    DONE+PRAMIN+pause ack; MEMIF loadback still virgin `ff`.  On this Palit
    ROM, prog0(1000) and prog0(648) select the **same** RAMMAP entry 2 — so
    (a) was a no-op.  **Closed:** cold `0x10f808` residuals are not why VRAM
    discards stores.
67. **Patched for night40aj:** Nouveau golden order after RAM is
    `fb_init_page(0x100c80)` then LTC/ZBC (`gf100.c:68-73`, mmiotrace
    @~0.393s).  Live had `KEPLER_POST_RAM_LTC=0` because night10 collapsed
    BAR0 when that ran *after* bit0.  With `KEPLER_RAM_BIT0_DEFER=1`, restore
    `KEPLER_POST_RAM_LTC=1` so post-RAM LTC runs before bit0.
68. **Night40aj true cold:** post-RAM LTC ran (`ltc_nr=4 parts=4 mask=0x0
    lpg128=True`) and BAR0 stayed live through FECS + pad.  Pad DONE+PRAMIN+
    pause ack; MEMIF loadback still virgin `ff`.  **Closed:** skipping
    Nouveau `fb_init_page`/LTC before deferred bit0 is not why VRAM discards
    stores.  (LTC `mask=0` with `ltc_nr=4` is Nouveau-correct: no disabled
    partitions.)
69. **Patched for night40ak:** pad does Nouveau PRAMIN store then **in-pause**
    `call 0x4` (`rd32`) of join dword0; DMEM `0xd6c` = expected−got (`0`=match).
    Fits IMEM exactly `0xb14..0xc00`.  Distinguishes H12 (LEAVE destroys) from
    “never visible under pause”.
70. **Night40ak true cold:** `DONE` published;
    `inpause_rb_delta=0x453004fd (PRAMIN MISMATCH under pause)`; post-LEAVE
    MEMIF still virgin `ff`.  **Closed H12:** a good PRAMIN store is **not**
    being destroyed by LEAVE — falcon PRAMIN readback already mismatches while
    FB_PAUSE is still held.  Physical/client visibility fails inside ENTER.
    (Delta decode: expected join dword0 `0x10000` − got `0xbad0fb03` =
    `0x453004fd` — same stub night40al later names absolutely.)
71. **Night40al true cold (H21):** pad writes known `0xa5a5a5a5` via Nouveau
    PRAMIN (`0x1700=0`, `0x730200=0x700000+0x30200`) under rising-edge pause;
    DMEM `0xd6c` = **absolute** got.
    `inpause_pramin_got=0xbad0fb03 (stub bad0fb; wrote 0xa5a5a5a5)`.
    **Closed H21:** in-pause PRAMIN is not returning VRAM (neither literal nor
    virgin `ff`) — it returns the same structured `bad0fb03` sentinel as a
    broken BAR1 walk.  Address math matches `nv50_instobj_*_slow`
    (`instmem/nv50.c:62-71`, `set_bar0_window_addr` → `0x1700=pa>>16`).
72. **Night40am true cold (H22/H23):** host PRAMIN literal at pa `0x30200` with
    `0x1620=0xaab` (bit0 still set), **no** pad ENTER — Nouveau INST shape
    (`nv50_instobj_wr32_slow`).
    `host_pramin_got=0xbad0fb03 (stub bad0fb; wrote 0xa5a5a5a5)`.
    **Closed H22:** ENTER-scoped PRAMIN is not the differentiator — host outside
    pause fails identically.  **Closed H23:** all tested FB clients still see the
    pre-RAM `bad0fb` stub after `ram_program`+LTC.  Stop iterating PRAMIN pads.
    Note: `0x10f210=0x80000404` this night (was `0x80000303` on al/ak); Nouveau
    writes absolute `0x80000000` (`ramgk104.c:583,908`).  `0x110974=0xa` still
    (Nouveau `ram_wait` wants low nibble **0**, `ramgk104.c:149-152`).
73. **Night40an true cold (H14/H24):** per-part train dump after `ram_program`:
    `part0..3` at `0x110974/111974/112974/113974` **all** `=0xa` (nibble `0xa`).
    Nouveau `gk104_ram_train` waits `(st & 0xf)==0` (`ramgk104.c:149-152`).
    **Closed H14 as present:** training status never clears on any partition.
    **Closed H24:** PMU `wait` (`kernel.fuc:118-134`) falls through on timeout
    without failing EXEC — so atomic MEMX_WAIT can “succeed” while nibble stays
    `0xa`.  Bug was also that H14 `RuntimeError` was swallowed as
    `MC dump skipped:` — fixed to re-raise.  Run continued into H21 pad (same
    `bad0fb03`); next cold must stop at H14.
74. **Night40ao true cold (H25):** H14 now **aborts** bring-up (no H21 pad).
    Trace: `train trigger mask=0xbc0f0000 data=0x980a0000 parts=4 pmask=0x0
    waits=4`.  Host 100ms re-poll: status **STUCK** at `0xa`.
    **Closed H25:** link-train never completes despite correct tables/waits.
75. **Patched + night40ap (H27):** atomic `set_gpio_level` compared a host
    re-read to `old` after a *buffered* MEMX write (almost never triggered).
    Fixed to Nouveau-equivalent `(old&0x3000) != want`.  Live: GPIO `0x18`
    pulsed; GPIO `0x2e` already at want `0x3000` (`old=0x700a`, `trig=no`).
    Train still stuck `0xa`.  **Closed H27:** 0x2e pulse not the blocker.
76. **Night40aq true cold (H29):** requested `KEPLER_TRAIN_WAIT_NS=50000000`
    to test a presumed 50 ms wait.  The stock-ENTER EXEC hit its exact 60 s
    host timeout; the nowait fallback hit its exact 15 s timeout.  Later
    cross-check against MEMX DELAY proves the raw TIMER_LOW unit remains ns;
    the 50 ms tight wait instead wedged the Falcon's repeated MMIO-read loop.
    ao/ap already showed `0xa` after Nouveau's 500 us plus host re-poll, so
    **H29 is closed** and waits above 5 ms/part are now rejected before MEMX.
    Card dirty; no PRAMIN/BAR1 ran.
77. **Patched for next cold (H31):** line-by-line audit found the port restored
    MR5/LP3 *after* the mode-1 `0x10f830[24]` set/clear pulse.  Nouveau does
    final train + wait, MR5/LP3 restore (+ optional 1 us), then the pulse
    (`ramgk104.c:642-657`).  Reordered the port and added an atomic-script
    ordering assertion.  This sits directly at the stuck train-completion
    boundary; live effect remains unverified until an enclosure power cycle.
78. **Patched for next cold (H32):** a broader execution-order audit found the
    port queued early MR1 termination before `prog0(1000)`, `0x10f808`, and
    MEMX ENTER.  Nouveau executes `prog0(1000)` first, then its transition is
    `0x10f808 → ENTER → early MR1` (`ramgk104.c:260-269,1231-1245`).  Moved
    MR1 after ENTER and added script-order assertions.  Unlike H31, this is
    before the failing train and is now the leading causal hypothesis.
79. **Night40ar true cold:** H32/H31 exact ordering completed, but all four
    train nibbles remained `0xa` through the stock wait plus 1 s host re-poll.
    **Closed H31/H32 as blockers.**  Strict H14 stopped before PRAMIN/BAR1.
80. **Patched for next cold (H33):** source audit found the active
    `0x10f670[31]` settling delay was 10,000,000 ns in the port versus exactly
    10,000 ns in Nouveau (`ramgk104.c:440-443`)—1000× too long immediately
    before PFB timing/MR setup.  Corrected to 10 us.
81. **Patched for next cold (H34):** the port unconditionally zeroed
    `0x10f698/69c`, then much later cleared all of `0x10f694` before applying
    its RAMCFG field.  Nouveau conditionally programs `0x10f698/69c` and
    immediately forces one *preserved* `0x10f694[ff00ff00]` update
    (`ramgk104.c:397-412`).  Strap 6 takes the zero branch for 698/69c, but
    the late destructive 694 sequence was active.  Replaced it with one early
    preserved write and asserted its exact command-stream position.
82. **Patched for next cold (H35):** Nouveau updates `0x100770`, performs the
    bit-2 transition wait if its selected value changed, and only then masks
    `0x100778` (`ramgk104.c:554-574`).  The port wrote 778 first and built the
    770 full value with unmasked selected fields.  Reordered and rebuilt the
    masked data exactly.  Also fixed selected-value handling for the
    `0x10f604/610/614` diff fields; the 610/614 differences are dormant on
    this ROM, while the 604/770 path is active.
83. **Patched for next cold (H36, leading):** all eleven PFB timing words were
    shifted by one register.  Nouveau writes `timing[10]` to `0x10f248`, then
    `timing[0..9]` to `0x10f290..0x10f2e8` (`ramgk104.c:461-471`); the port
    zipped `timing[0..10]` straight across those addresses.  Corrected the
    rotation and added an exact address/value assertion over the flattened
    atomic MEMX script.  This is pre-training and changes every timing register,
    making it the strongest current explanation for all parts sticking at
    `0xa`.
84. **Patched/instrumented (H37):** `nvkm_gddr5_calc()` saves the ordinary
    MR1 termination image for partition mirrors and selects the alternate
    image for the main controller only when `pnuts != 0`; the port had those
    images reversed.  Corrected it and made early zero `0x022438` use the same
    bounded four-part topology fallback as the train waits.  Next cold logs
    `0x110204` values and computed `pnuts`.  Both Palit timing fields equal 3,
    so the MR1 inversion itself is dormant for this ROM; topology fallback
    matters only if the live partition configurations differ.
85. **Night40as true cold:** the complete H33–H37 command image executed.
    Live topology was exact and homogeneous: `parts=4`, `pmask=0`,
    `0x110204..113204=0x0255a000`, `pnuts=0`.  All four train nibbles still
    stayed `0xa` through Nouveau's 500 us waits plus 1 s host re-poll; strict
    H14 aborted before PRAMIN/BAR1.  **Closed H33–H37 as blockers.**
86. **Closed H38:** Nouveau `nvbios_post()` executes a separate BIT-I `+14`
    script after the normal script table.  This Palit ROM points it at
    `0xacfa`, which is a single `DONE` opcode, so omitting it has no hardware
    effect.
87. **Patched for next cold (H39, leading):** dynamic strap-6 replay covered
    580 NVINIT opcode dispatches and found no executed PLL, COMPUTE_MEM,
    CONFIGURE_MEM, or CONFIGURE_CLK opcodes.  It did find one executed
    `INIT_GPIO` at ROM `0x8b2d`; the Python interpreter treated it as a no-op.
    Nouveau resets every non-unused DCB entry to its default, pulses
    `0x00d604`, writes each GPIO low-byte metadata field, and installs the
    `0x00d740` routing entries (`bios/init.c:1960-1973`,
    `gpio/gf119.c:27-50`).  Ported that behavior for all 23 active Palit GPIOs,
    implemented `GPIO_NE` too, and added exact offline state/trigger assertions.
88. **Opened H40/H41 (literal-port boundary):** `gk104_ram_calc/prog` is a
    reclocking transition built after devinit, GPIO, FB construction, topology,
    and a readable former memory clock; this userspace path invokes it against
    a cold un-POSTed card.  H40 asks whether missing/invalid former state—not a
    remaining line inside `gk104_ram_calc_gddr5()`—is the blocker.  H41 asks
    whether TinyGPU-required execution differences (all-MEMX prog0, ENTER wait
    patching, no Linux IRQ/subdev lifecycle) change semantics despite matching
    register order.  Prove both with a source-literal command/state oracle and
    stop at its first divergence; do not blindly translate unrelated Nouveau.
89. **Night40at true cold (H39):** live devinit selected `INIT_GPIO_NE`, reset
    18 non-excluded functions, and changed eight (`0x18,0x23,0x34,0x2e,0x79,
    0x07,0x51,0x52`).  The RAM transition then found memory GPIOs `0x18` and
    `0x2e` already at their requested levels.  Atomic RAMFUC completed, but
    all four train nibbles stayed `0xa` through 1 s host re-poll; strict H14
    stopped before PRAMIN/BAR1.  **Closed H39 as blocker.**
90. **Patched + night40au (H42):** the straight NVINIT port had stubbed all
    VGA indexed I/O.  Ported Nouveau's exact GK104 byte transactions at
    `0x601000+port` and added index/data address-order tests.  Palit's one
    executed read was CRTC index `0x8d`; live value was `0x00`, preserving the
    same `INIT_GPIO_NE` path.  All train nibbles still `0xa`.  **Closed H42 as
    blocker; fixed a real source-port defect.**  The former-memory selector
    was `0x1373f4[3:0]=1`, so H40's simplest “no readable clock path” form is
    also closed.
91. **Patched + night40av (H43):** line-by-line GDDR5 audit found the next
    omitted source lines: display-memory quiesce `0x62c000=0x0f0f0000` after
    ENTER and restore `0x0f0f0f00` after LEAVE.  Added both and asserted their
    position in the atomic stream.  Live transition completed; all four train
    nibbles remained `0xa`.  **Closed H43 as blocker.**  Post-train refresh
    low counters vary (`at=0x2d2d`, `au=0x4343`, `av=0x3737`) while bit31 and
    the stuck nibble are stable; treat them as observations, not a cause yet.
92. **Night40aw true cold (H41-A):** restored Nouveau's literal host
    `gk104_ram_prog_0(1000)` before PMU `ram_exec`.  Contrary to older runs,
    the current PMU reload path accepted the following MEMX smoke and the full
    atomic transition completed.  All four train nibbles still remained
    `0xa`; strict H14 stopped before PRAMIN/BAR1.  **Closed H41-A as blocker.**
    The RAM reload explicitly used unpatched stock FB_PAUSE waits and completed,
    so the ENTER/LEAVE part of H41-B is already closed by aw/av evidence.
93. **Patched + night40ax (H45):** `ramfuc_mask()` caches each
    register and queues WR32 only when the value changes or `ram_nuke` forced
    it.  Python queued every masked write.  Split source-like change-detect
    RAMFUC masks from always-write host `nvkm_mask`, retained the forced
    `0x10f694` update, and made `prog0` issue all seven source host masks.
    Corrected the test to identify the final `0x10f830[24]` pulse by value and
    order instead of counting all writes to that register.  Full offline suite
    green.  Live atomic payload dropped from aw's 188 WR32 words to **138**,
    proving 50 spurious words were removed, but all four train nibbles stayed
    `0xa`.  **Closed H45 as blocker; fixed a material source-port defect.**
94. **Patched + night40ay (H46):** `r1373f4_init()` rebuilds REFPLL only when
    `0x132024` or `0x132034[15:0]` differs from the calculated coefficient.
    Python rebuilt it unconditionally immediately before training.  Added the
    exact coefficient guard and live trace; mismatch and already-matching
    offline tests both pass.  Cold live found current `0x32301/0x1000` versus
    wanted `0x11701/0x1000`, so the source guard correctly chose
    `reprogram=yes`; the old unconditional port would have taken the same
    active path.  Atomic RAMFUC remained 138 WR32 words and all four train
    nibbles remained `0xa` through the 1 s host re-poll.  **Closed H46 as a
    blocker; retained the source-faithful fix.**
95. **Patched for next cold (H47):** after the conditional
    REFPLL block, Nouveau's mode-1 path ORs `0x00010100` into `0x1373f4`, then
    ORs `0x10`, before `r1373f4_fini()`.  Python jumped straight to fini,
    omitting those source-selection bits.  Added both source masks and an exact
    flattened-stream tail assertion:
    `0x10100, 0x10110, 0x10111, 0x00111`.  The full 24-checkpoint mmiotrace
    suite and software demo pass.  The next cold MC dump also captures
    `0x1373f4`, `0x132024`, and `0x132034` to prove the committed selector and
    coefficients rather than infer them from queued commands.
96. **Night40az true cold (H47):** the two added mode-1 masks were active:
    atomic RAMFUC grew from 138 to 142 WR32 words, completed, and committed
    `0x1373f4=0x00000111`.  REFPLL committed the requested
    `0x132024=0x00011701`, `0x132034[15:0]=0x1000`, with lock/status
    `0x137390=0x30001`.  All four train nibbles nevertheless remained `0xa`
    through the 1 s host re-poll, and strict H14 stopped before PRAMIN/BAR1.
    **Closed H47 as blocker; retained the source-literal fix.**
97. **Patched for Night41a (H48, leading):** line-by-line comparison moved
    outside `gk104_ram_calc_gddr5()` to its caller.  Night40az entered with
    selector 1 and REFPLL `0x32301/0x1000`; Nouveau's exact
    `gk104_clk_read_mem()` decoder yields **324000 kHz**.  The Palit ROM maps
    324 MHz to RAMMAP 1/timing 4 and 648 MHz to RAMMAP 2/timing 0.  For this
    upward transition, `gk104_ram_calc()` first copies target
    `timing_20_30_07=6` over former value 5, executes that 324-MHz `xition`,
    returns positive from `prog()`, then the clock core loops and executes the
    648-MHz target.  Python skipped the first pass.  Ported the exact former
    clock decoder and active two-pass contract.  Offline atomic evidence now
    contains two complete scripts with timing[10] values `0x031455a5` then
    `0x06086442`; full middle, 24-checkpoint mmiotrace, software demo, compile,
    and diff checks pass.
98. **Night41a true cold (H48):** both source caller passes executed.  The
    324-MHz xition kept matching REFPLL `0x32301/0x1000`, used train data
    `0x880a0000`, and completed with 34 commands / 120 WR32 words.  The
    following 648-MHz target rebuilt REFPLL to `0x11701/0x1000`, used
    `0x980a0000`, and completed with 37 commands / 134 words.  Final
    `0x10f210=0x80001717` differed materially from night40az's one-pass
    `0x80003939`, proving the intermediate transition affected silicon, but
    all four nibbles still remained `0xa` through 1 s.  Strict H14 stopped
    before PRAMIN/BAR1.  **Closed H48 as blocker; retained the caller fix.**
99. **Instrumented for Night41b (H49, leading fork):** Nouveau's normal
    `nvkm_fb_init()` calls `nvkm_ram_init()`; `gk104_ram_calc/prog()` belongs
    to the separate, later clock pstate loop.  Python always ran both but only
    sampled train status after reclocking.  Added a non-aborting MC/four-part
    status dump immediately after RAMMAP/training init and before PMU reload or
    ram_program.  If pre-reclock nibbles are zero, the extra cold reclock is
    destructive/unnecessary; if they are already `0xa`, the first divergence
    is POST/devinit or RAM init and further reclock edits are irrelevant.
100. **Night41b true cold (H49 proven):** the source-ordered VBIOS/RAM init
    completed with all four train status nibbles **zero**.  The pre-reclock
    snapshot was `0x10f210=0x80005656`, `0x10f808=0x7aa00050`,
    `0x10f910/914=0x08020000`, selector `0x1373f4=1`, and REFPLL
    `0x132024=0x32301` / fraction `0x1000`.  The immediate H48 two-pass
    324→648 MHz reclock then changed every train nibble to **`0xa`**, with
    `0x10f210=0x80005c5c`, `0x10f808=0x72a00000`, and
    `0x10f910/914=0x19080000`.  The nibbles remained `0xa` through the 1 s
    host re-poll; strict H14 stopped before PRAMIN/BAR1.  **Confirmed H49:**
    RAM init succeeds and the extra cold reclock destroys its trained state.
101. **Patched for Night41c (lifecycle fix):** cold bring-up now defaults
    `KEPLER_RAM_PROGRAM=0`, matching `nvkm_fb_init()` stopping after
    `nvkm_ram_init()`.  `KEPLER_RAM_PROGRAM=1` remains an explicit reclocking
    diagnostic.  When reclocking is skipped, the post-init four-part status
    check is strict, so PRAMIN/BAR1 cannot run unless all train nibbles remain
    zero.  Normalized the shared sequencer's environment fallback and updated
    the trace tests.  Compile, middle selftest, all 24 mmiotrace checkpoints
    (1636 exact cold writes), software demo, and diff checks pass.
102. **Night41c true cold (H50 classified):** the default skip preserved all
    four train nibbles at **zero** through FB page/LTC initialization.  The
    boundary snapshot was `0x10f210=0x80001c1c`,
    `0x10f808=0x7aa00050`, `0x10f910/914=0x08020000`, selector
    `0x1373f4=1`, and REFPLL `0x32301/0x1000`.  LTC settled with
    `0x100c80=0x208000`; FECS reached ready and topology 4×2, then launch
    reached BAR1 bootstrap.  The embedded PMU pad was still Night40al's closed
    H21 literal-only diagnostic: it wrote `0xa5a5a5a5` under ENTER, read
    `0xbad0fb03`, and deliberately raised before any root `xdst`.  Thus H50's
    preservation half is confirmed, but train nibble zero alone does **not**
    make direct PRAMIN usable; the first new blocker was stale experiment code.
103. **Patched for Night41d (H51 fixed / H52 instrumented):** replaced the
    H21 literal probe with the complete ENTER→three direct-VRAM `xdst`→LEAVE
    bootstrap for the Nouveau BAR1 instance root (`0x30200`), PDE (`0x10000`),
    and PTEs (`0x20000`).  Removed the intentional H21 abort and made
    post-LEAVE MEMIF loadback verify all 40 bytes before `0x1704` enable.
    Assembled Falcon source is `0xde` bytes, the padded embedded image ends at
    `0xbf4 < 0xc00`, and source/header/Python bytes match exactly.  Compile,
    middle selftest, all 24 mmiotrace checkpoints, software demo, and diff
    checks pass; the fake observes all three loadbacks before BAR1 enable.

### Nouveau line-by-line (PFB timing block) — patched after night40ar

| Source order | Nouveau | Old port | Current port |
|--------------|---------|----------|--------------|
| 1 | conditional `0x10f698/69c` | unconditional zero | conditional exact |
| 2 | forced preserved `0x10f694` | late full clear + mask | one early preserved mask |
| 3 | `0x10f670` settle 10,000 ns | 10,000,000 ns | 10,000 ns |
| 4 | `248=t[10]`, `290..2e8=t[0..9]` | `t[0..10]` straight | rotated exact |
| 5 | selected values gate 604/610/614 data | diff implied set | selected-value exact |
| 6 | `0x100770` → optional wait → `0x100778` | 778 before 770 | source order exact |

### Nouveau line-by-line (GPIO 0x2e) — night40ap

| Step | Nouveau | Ours | Match? |
|------|---------|------|--------|
| `vc = !ramcfg_11_02_08` | strap6 → vc=0 | same | yes |
| Pre-train GPIO | `func2E[0]` + trig if changed | `set_gpio_level(0x2e,0)` | yes |
| Change detect | `temp != ram_rd32` after mask | `old&0x3000 != want` (fixed) | yes |
| Live | — | already `0x3000`, skip trig | expected |

### Nouveau literal-port boundary — after night40av

| Layer/step | Nouveau | Python now | Classification |
|------------|---------|------------|----------------|
| NVINIT VGA byte I/O | `0x601000+port` | exact byte index/data | fixed; H42 closed |
| GPIO reset/defaults | DCB reset + trigger/routes | exact active Palit path | fixed; H39 closed |
| GDDR5 central body | `ramgk104.c:252-675` | register/order + RAMFUC cache tests | H45 fixed; audit continues |
| Former memory selector | must be 1 or 2 | live is 1 | prerequisite present |
| `prog0(1000)` | host before `ram_exec` | literal host mode validated | H41-A closed |
| ENTER/LEAVE firmware | stock FB_PAUSE waits | stock-wait RAM reload completes | H41-B closed |
| `0x10f808` final mask | preserve bits outside mask | deliberate absolute cold cleanup | **open H44** |
| Invocation context | reclock only in later pstate worker | default now stops after RAM init | **H40/H49 fixed** |

### Nouveau line-by-line (mode-1 REFPLL transition) — after night40ay

| Source order | Nouveau | Python now | Evidence |
|--------------|---------|------------|----------|
| 1 | compare `0x132024`, `0x132034[15:0]` | exact conditional guard | ay mismatch, rebuild taken |
| 2 | rebuild REFPLL only on mismatch | exact queued block | H46 closed as blocker |
| 3 | `0x1373f4 |= 0x00010100` | exact RAMFUC mask | H47 offline exact |
| 4 | `0x1373f4 |= 0x10` | exact RAMFUC mask | H47 offline exact |
| 5 | fini selector mode 1, clear bit 16 | exact `0x10111 → 0x00111` | H47 offline exact |

### Nouveau line-by-line (clock caller loop) — Night41a candidate

| Source order | Nouveau | Python now | Evidence |
|--------------|---------|------------|----------|
| 1 | `read_mem()` decodes former clock | exact PLL/divider port | `0x32301/0x1000 → 324000 kHz` |
| 2 | choose former/target RAMMAP entries | 324→entry 1, 648→entry 2 | Palit ROM decoded |
| 3 | build `xition`; copy three transition fields | exact upward copy | `timing_20_30_07: 5→6` active |
| 4 | `calc/prog(xition)` | complete 324-MHz atomic pass | offline script exact |
| 5 | positive return loops to target | complete 648-MHz atomic pass | offline script exact |

### Nouveau lifecycle boundary — Night41b result

| Boundary | Nouveau owner | Python action | Night41b proof |
|----------|---------------|---------------|----------------|
| POST/devinit | device preinit | BIT-I scripts | cold path reached valid training |
| RAMMAP/train tables | FB `nvkm_ram_init()` | source-locked 1636-write cold slice | all four nibbles became `0` |
| Reclock transition | later clock pstate worker | explicit diagnostic only | immediate run changed all to `0xa` |

### Nouveau line-by-line (BAR1 bootstrap) — Night41d result

| Source order | Nouveau | Python Night41d | Guard/evidence |
|--------------|---------|-----------------|----------------|
| 1 | allocate 4 KiB BAR1 INST object | fixed instance bank `0x30000` | layout assertion |
| 2 | `nvkm_vmm_join()` writes instance root | staged at DMEM `0xd80`, xdst→`0x30200` | 16-byte loadback |
| 3 | VMM writes PDE and PTEs | staged at `0xd90/0xda0`, xdst→`0x10000/0x20000` | 8+16-byte loadback |
| 4 | instmem mapping makes root writes durable | PMU direct-VRAM MEMIF under ENTER | **3/3, 40/40 bytes exact after LEAVE** |
| 5 | PT flush, HUB-only PDB invalidate, flush | exact `0x70000`; `0x100cb8/0x100cbc=0x80000005`; `0x70000` | matches golden trace order |
| 6 | `gf100_bar_bar1_init()` writes `0x1704=0x80000000|(inst>>12)` | exact `0x80000030` | only after 3/3 match |
| 7 | `gf100_bar_bar1_wait()` flushes twice | exact double readable flush | live completed; BAR0 remained live |
| 8 | first BAR1 root walk resolves temporary PTEs | expected `01020000…01030000…` | **failed: structured `bad0fb` walk sentinel** |

### Nouveau allocation boundary — Night41e result / Night41f fix

| Check | Nouveau | Night41e | Night41f |
|-------|---------|----------|----------|
| VRAM allocator floor | reserves `[0,0x40000)` for VGA | PGD/SPT/INST all below `0x40000` | PGD=`0x40000`, SPT=`0x50000`, INST=`0x60000` |
| Physical durability | INST objects live in VRAM | all 40 bytes durable | same 3/3 loadback guard retained |
| BAR activation | INST address `>>12` | `0x80000030`; walk returned `bad0fb` | `0x80000060`; cold proof pending |
| Size probe owner | `0x11020c + fbp*0x1000` | legacy `0x10020c` snapshot was a sentinel | corrected four-FBP snapshot |

### Hypothesis ledger (VRAM non-retention)

| ID | Hypothesis | Status | How to prove/disprove |
|----|------------|--------|------------------------|
| H01–H14, H19, H21–H26 | prior closed | **closed** | — |
| H15 | Refresh/MC image wrong despite train zero | **open↓** | PMU MEMIF retains exact bytes while only PBUS clients fail; revisit after PRAMIN ownership is solved |
| H16 | Missing VBIOS/GPIO beyond known | **open↓** | 0x2e/0x18 OK |
| H17 | Per-FBP size/decode wrong | **closed** | 41f: four enabled FBPs, each `0x400` MiB; total 4 GiB matches the card |
| H25 | Link-train HW never completes | **superseded by H40/H49** | 41b proved base RAM init reaches `0`; only the misplaced optional reclock creates `0xa` |
| H27 | Pre-train GPIO 0x2e broken | **closed** | ap: already correct |
| H28 | `0xa` sticky success/error not busy | **closed↑** | 41b: valid RAM init is `0`; destructive reclock creates `0xa` |
| H29 | MEMX_WAIT too short for cold | **closed** | aq: 50ms wedges PMU; stock 500µs + host repoll stuck |
| H30 | Early refresh/MR sequence wrong | **deferred** | revisit only after base-clock PRAMIN exists; it cannot explain the pre-runtime ownership boundary |
| H31 | MR5/LP3 restored after mode pulse | **closed, fixed** | ar still `0xa` |
| H32 | Early MR1 queued before prog0/ENTER | **closed, fixed** | ar still `0xa` |
| H33–H37 | Corrected PFB timing/order/topology defects | **closed, fixed** | as: exact path still `0xa` |
| H38 | Missing BIT-I unknown post script | **closed** | ROM target is only `DONE` |
| H39 | Executed `INIT_GPIO` was a no-op | **closed, fixed** | at: defaults applied; nibble still `0xa` |
| H40 | Cold card violates Nouveau reclocking prerequisites | **confirmed, fixed** | 41b: reclock is outside FB init and turns `0→a`; default removed |
| H41-A | Host `prog0(1000)` ordering | **closed** | aw: literal host path completed; nibble `0xa` |
| H41-B | Stock ENTER/LEAVE wait semantics | **closed** | av/aw stock RAM reload completed |
| H42 | NVINIT VGA indexed I/O was stubbed | **closed, fixed** | au: CR8d=0; nibble still `0xa` |
| H43 | Missing display-memory quiesce bracket | **closed, fixed** | av: bracket executed; nibble still `0xa` |
| H44 | Absolute `0x10f808` differs from source mask | **deferred** | A/B source-preserve only after base-clock PRAMIN exists and cold residual is safe |
| H45 | RAMFUC unchanged masks incorrectly emitted | **closed, fixed** | ax: 188→138 words; nibble still `0xa` |
| H46 | REFPLL rebuilt even when coefficients match | **closed, fixed** | ay: coefficients differed, rebuild required; nibble still `0xa` |
| H47 | Missing mode-1 `0x1373f4 |= 0x10100; |= 0x10` | **closed, fixed** | az: selector `0x111`; nibble still `0xa` |
| H48 | Missing active former→xition→target caller loop | **closed, fixed** | 41a: two passes changed MC state; nibble still `0xa` |
| H49 | Train status may fail before, or be broken by, cold reclock | **confirmed, fixed; merged with H40** | 41b: ram_init `0,0,0,0`; immediate reclock `a,a,a,a` |
| H50 | Preserved trained state yields usable PRAMIN/BAR1 | **disproven** | 41c/d/g preserved zero and PMU-visible data, while PRAMIN/physical BAR1 remained stubbed |
| H51 | Closed H21 literal-only diagnostic still owns default launch | **confirmed, fixed** | 41c aborted before xdst; restored production transfer pad |
| H52 | Three PMU xdst roots persist in the PMU MEMIF view with valid train state | **confirmed** | 41d: 3/3 fragments, 40/40 bytes exact after LEAVE; H57 proves this is not global PBUS visibility |
| H53 | PFB/PBUS BAR client state is incomplete despite durable roots | **superseded by H67/H70** | later boundary probes show runtime initialization never creates live PRAMIN at all |
| H54 | Root encoding or BAR activation order differs from Nouveau | **closed** | bytes match `vmmgf100.c`; flush/invalidate/enable order matches golden trace |
| H55 | Missing enabled `gk104_fb_clkgate_pack` causes the walk failure | **closed as golden-path blocker** | `NvPmEnableGating` defaults false; golden writes zero, never the pack's enable values |
| H56 | BAR roots inside Nouveau-reserved VGA VRAM are not PBUS-walkable | **closed** | 41f relocated all roots above `0x40000`; identical `bad0fb` remained |
| H57 | PMU MEMIF-visible roots are not visible through PBUS physical BAR1 | **confirmed** | 41g: 3/3 MEMIF exact, physical BAR1 returned incrementing `bad0fb` with VM disabled |
| H58 | PMU→PBUS handoff needs LTC flush/invalidate | **closed** | 41g: Nouveau flush `0x70010` + invalidate `0x70004` both completed; physical reread still `bad0fb` |
| H59 | Complete `NVKM_MEM_TARGET_INST` allocation semantics are required | **deprioritized** | Nouveau instobj writes use BAR2 or PRAMIN; both depend on the PBUS visibility already disproven by H57 |
| H60 | Python omitted a BIT-I POST script or memory opcode | **closed in that scope** | all 7 normal scripts are present and the unknown script is only `DONE`; later H76/H78 found executed non-memory opcode and invocation-state mismatches, so this is not a claim of whole-interpreter parity |
| H61 | Nouveau inherits firmware-POSTed PBUS/PRAMIN state that this cold eGPU lacks | **confirmed as an ownership boundary; causal sufficiency unproved** | golden trace reads PRAMIN `0xbeef` at +3 us, before RAMMAP; 41m found the exact Nouveau RAMIN-source predicates present in golden state and absent after enclosure replug |
| H62 | A full enclosure replug causes platform firmware to POST the external GTX 770 | **disproven** | 41h entry: marker `0`, topology `badf1200`, current PRAMIN word `ffffffff`; no GPU writes followed |
| H63 | Python `INIT_ZM_REG` omitted Nouveau's `PMC_ENABLE[0]` invariant | **confirmed, fixed; not the final blocker** | 41j: first cold ROM value `0x2020` became Nouveau-correct `0x2021`; DMEM immediately changed from `badf1200` to exact read/write, but after full devinit PBUS still returned the H57 `bad0fb` walk |
| H64 | Missing `nv50_devinit_init()` display-encoder scripts contain cold board/PBUS setup | **closed for this ROM** | four TMDS matches point to script `0x5a55` (`DONE`); the DP script `0x6247` suppresses its body because the connector is external, not eDP |
| H65 | Python omitted Nouveau's pre-NVINIT extended VGA CRTC unlock | **confirmed, fixed; not the final blocker** | 41k emitted `CR3f=0x57`, then read `CR8d=0`; train and MEMIF roots remained exact, but physical BAR1 still returned incrementing `bad0fb` before/after LTC handoff |
| H66 | Python establishes physical PRAMIN temporarily, then a later stage destroys it | **disproven** | 41l: cold was `ffffffff`; first full `PMC_ENABLE` exposed the `bad0fb` stub, which persisted through devinit, PGOB, zero-nibble RAM training, and FB/LTC init |
| H67 | The cold PBUS/PRAMIN client is never instantiated by the runtime bring-up | **confirmed observationally** | 41l fixed-PA reads were stubbed at every active boundary; golden Nouveau reads `beef` before its first init write, and its BAR2 bootstrap's first store already consumes PRAMIN |
| H68 | Platform/option-ROM POST leaves an enabled high-VRAM ROM window alongside live PRAMIN | **confirmed correlation; not causally sufficient** | golden: `0x619f04=0xfffe09` and PRAMIN `beef`; 41m cold lacks both; 41n reproduced the register state against PMU-visible bytes but PRAMIN stayed stubbed |
| H69 | Setting `0x619f04` and BAR0 mirror `0x088050[0]` alone will instantiate PBUS/PRAMIN | **disproven for those MMIO writes** | 41n exact-address A/B stayed sequential `bad0fb`; native PCI-config offset `0x50` remains a distinct untested transaction class |
| H70 | An additional pre-runtime producer establishes a hidden PBUS/PRAMIN prerequisite before Nouveau starts | **refined; no longer the Python blocker** | golden runtime inherits live PRAMIN, but Night41s proves corrected Nouveau-compatible VBIOS POST can create live fixed-PA PRAMIN from a virgin cold entry on this arm64 path |
| H71 | Replaying only the traced pre-runtime option-ROM producer slice can establish PRAMIN on arm64 | **open↓; contingency** | no longer required to explain PRAMIN activation; retain only if the corrected POST transition cannot be preserved through RAM/BAR bring-up |
| H72 | Another register in the golden runtime's pre-init readable MMIO set identifies the prerequisite | **disproven** | 41o extracted all seven unique non-aperture reads in that set; only the three Night41n values differ; this does not claim every possible MMIO register was sampled |
| H73 | The missing producer uses PCI config, native legacy VGA I/O, reset/power sequencing, or write-only/internal state absent from mmiotrace | **open↓; golden-platform explanation only** | still explains why the golden trace enters live, but Night41s proves these domains are not required for Python to activate PRAMIN through corrected NVINIT POST |
| H74 | Replaying the reachable ROM VGA-enable prefix through NVIDIA's BAR0 VGA alias is sufficient to instantiate PRAMIN | **disproven** | 41p exact ordered A/B (`W 3c3=1`, readback, `W 3c2=1`, readback, `CR3f=57`) left PRAMIN `ffffffff` and topology `badf1200`; native x86 I/O equivalence remains unproved |
| H75 | The reachable option-ROM native I/O-BAR prefix instantiates the missing PBUS/PRAMIN client | **open↓; secondary** | real ROM reachability and exact transaction shape are confirmed offline, but Night41s activates PRAMIN without native I/O; retain as a firmware-equivalence fallback, not the next cold experiment |
| H76 | Python skipped the executed Nouveau NV50+ `INIT_IO 0x3c3,0,1` activation sequence | **confirmed, fixed; closed as PBUS activator** | 41q ran the literal ten-write sequence and waits before all other GPU init; topology stayed `badf1200` and PRAMIN stayed `ffffffff` |
| H77 | Python's cold lifecycle order changes the meaning of otherwise-correct Nouveau operations | **confirmed at POST boundary** | 41s entered virgin (`PRAMIN=ffffffff`), ran the seven scripts in Nouveau order, and immediately read four non-stub data dwords at fixed PA `0xfffe0000`; RAM was intentionally not entered |
| H78 | Top-level NVINIT state leakage, active I2C stubs, fail-open MMIO, and clobbering `0x10f65c` made Python POST non-equivalent to Nouveau | **confirmed, fixed; cold-supported** | 41s crossed the PRAMIN boundary with the parity cluster enabled; the follow-up audit also resets the private RAMCFG cache per top-level invocation and implements literal `INIT_CR_INDEX_ADDRESS_LATCHED`; `0x10f65c` is downstream and raw transport errors did not occur, so the exact causal member remains unisolated |
| H79 | One corrected top-level POST script/opcode is the first PRAMIN activator | **open↑; leading discriminator** | Night41t samples fixed PA after scripts `87e5,8fe8,64ff,abb5,abb6,ac95,acfb` and stops at the first positive result; `ac95` is a strong candidate because leaked `execute=3` previously suppressed its `0xd628`/`0xd604` writes, but this is not yet live proof |

### Roadmap (ordered) — prove/disprove

1. **Night41t / H79:** on the next fresh replug run
   `--probe-nouveau-post-script-bisect`.  It preserves one Nouveau interpreter,
   samples fixed PA after every top-level script, restores `0x1700`, and stops
   at the first positive transition.  No RAM, PMU, BAR, or reset follows.
2. If one script activates PRAMIN, isolate its first causal nested script/opcode
   with ROM-offset checkpoints.  Prefer offline sequence alignment first; use
   at most one additional cold prefix bisection if the script contains several
   plausible irreversible groups.
3. **Preservation run:** execute the exact successful POST, require the same
   four-word PRAMIN-positive checkpoint, then run only base `gk104_ram_init`.
   Record all train nibbles and fixed-PA PRAMIN immediately afterward.  Stop on
   either nonzero training or loss of PRAMIN.
4. If RAM retains train `0` and live PRAMIN, reuse the already verified
   Nouveau root encodings and advance through LTC → MMU/BAR2 → BAR1.  Require a
   VM-disabled physical read to match PMU-visible bytes before enabling BAR1 VM;
   then run the existing channel/ADD path and stop at its first mismatch.
5. If RAM destroys the new PRAMIN state, bisect only the source-literal RAM
   boundaries; if BAR setup fails while physical PRAMIN remains live, compare
   the relevant `gf100_bar`/instmem slice line by line rather than porting the
   entire driver.
6. Keep H75/H71/H73 as a secondary x86/firmware branch.  Native I/O remains
   useful for explaining the golden inherited state, but is no longer required
   for Python PRAMIN activation.
7. Keep H69/H74/H76 closed for their tested MMIO/alias forms.  Continue
   executed-opcode parity; raw MMIO/VGA failures stay fatal and I2C protocol
   failures follow each Nouveau opcode's explicit contract.
8. Defer reclocking-only H30/H44 until ADD works at base clock; never restore
   the destructive optional reclock to cold FB init.
9. Every cold run has one predeclared early prediction and stops at its first
   discriminator—no unchanged full-path replay.

### Still missing for `hardware_demo=ok N=8`

1. Obtain one firmware/Nouveau-POSTed GTX 770 state without removing card power
2. Require POST ownership bit + live PRAMIN at process entry
3. Reuse the already exact BAR1 roots, expand BAR1 + channel
4. Launch → `hardware_demo=ok N=8`

### Night41g result and Nouveau POST boundary

Night41g preserved all four train nibbles at zero and again loaded back all
three PMU root fragments exactly.  With `0x1704=0`, however, PBUS physical BAR1
returned sequential `bad0fb` sentinels for every one of the 40 bytes.  Nouveau's
LTC flush and invalidate commands both completed, and the second physical read
returned the same sentinel class.  The guard stopped before BAR1 VM enable.

The follow-up source/ROM audit found no missing normal init script: Python runs
the same seven BIT-I scripts as `nvbios_post()`, the BIT-I unknown script points
to a single `DONE`, and this ROM executes no `INIT_COMPUTE_MEM`,
`INIT_CONFIGURE_MEM`, or `INIT_CONFIGURE_CLK`.  More importantly, the checked-in
Nouveau golden trace reads PRAMIN `0xbeef` only three microseconds after its
first MMIO and before RAMMAP/training.  `gf100_bar_oneinit()` and
`nv50_instobj` consume that existing PRAMIN/BAR2 path; they do not create it.
The ROM does set `0x2240c[1]` at script offset `0xac25`, and Night41g's RPC
trace confirms Python emitted that write.  Because physical PBUS remained
`bad0fb`, the bit is a completed-devinit marker rather than the missing client
activation.
This moves the boundary from BAR/VMM encoding to platform/VBIOS POST ownership.

### Night41h entry-only result

The dedicated five-field entry probe (after one initial `PMC_BOOT_0` validation)
classified the fresh enclosure replug without running NVINIT, RAMMAP, PMU,
BAR1, or a GPU reset:

- `PMC_BOOT_0=0x0e4040a2` — BAR0 and the GK104 identity are healthy;
- `0x2240c=0` — no inherited completed-devinit marker;
- `0x409604=0xbadf1200` — GPC remains power-gated;
- `0x1700=0`, current `0x700000=0xffffffff` — no positive inherited PRAMIN
  evidence.

Thus an electrical replug is not a firmware POST event for this external GPU.
The run stopped immediately and did not consume the cold state with another
known-failing H57 replay.  `NvForcePost` is not a separate implementation:
Nouveau routes it to the same `nvbios_post()` init-table interpreter already
ported here.  A literal runtime-driver port therefore cannot manufacture the
missing pre-driver state observed in the golden trace.

### Night41j result — `INIT_ZM_REG` fidelity

A line-by-line comparison of every opcode class parsed by this ROM found one
real semantic divergence.  Nouveau `init_zm_reg()` special-cases address
`0x000200` by ORing bit 0 before the write.  Python had written the ROM payload
verbatim, so the first cold instruction at ROM `0x87e6` emitted `0x2020`
instead of `0x2021`.  The handler now matches Nouveau, with focused tests for
both the special address and an ordinary register; compile, middle selftest,
software demo, and the 24-checkpoint/1636-write golden RAM path all pass.

Night41j proved that this was a functional defect: before devinit the DMEM
probe returned `badf1200`; immediately after the corrected master enable it
round-tripped the exact test word and PGRAPH status changed to `badf1000`.
The full ROM sequence subsequently programmed its intended sparse engine
state, and the run still ended at the same later boundary: train nibbles were
all zero, all 40 root bytes loaded back exactly through PMU MEMIF, but physical
VM-disabled BAR1 returned sequential `bad0fb` before and after completed LTC
flush/invalidate.  H63 is therefore fixed and experimentally confirmed, but
it does not supersede H57/H61.

The trace audit also explains why blindly translating the remaining Nouveau
driver is unlikely to close H61.  In the checked-in reference trace all 159
devinit operation lines carry `[ ]` (parsed with execution disabled); the only
executed `[0]` operation lines are six RAM restriction writes and two `DONE`s.
That Linux run inherited a card already POSTed before Nouveau's runtime path.
A literal Python port is useful for finding semantic drift such as H63, but it
cannot reproduce writes that the reference run itself never performed.

### Night41k result — devinit lifecycle parity

The next line-by-line layer found two omissions outside the opcode switch.
First, `nv50_devinit_init()` runs each matched DCB output's encoder script after
POST.  For this Palit ROM that path emits no relevant write: four TMDS outputs
select `0x5a55`, whose first byte is `DONE`, and the sole DP output selects
`0x6247`, whose body is gated to an internal eDP connector.  H64 is closed
without adding inert machinery.

Second, `nvkm_devinit_preinit()` always calls `nvkm_lockvgac(..., false)` before
the init tables.  On GK104 that is the indexed write `CR3f=0x57`.  Python now
performs that write once before running all BIT-I scripts in a shared
interpreter view.  Night41k observed the exact write and the following extended
`CR8d=0` read.  All four train nibbles stayed zero; PMU MEMIF again loaded all
40 root bytes back exactly.  Physical VM-disabled BAR1 nevertheless returned
the same sequential `bad0fb` signature before and after completed LTC flush and
invalidate, so H65 is fixed but closed as the PBUS blocker.

The instmem/BAR source walk sharpens the porting decision.  GK104 uses
`nv50_instobj_wr32_slow()` for bootstrap objects until BAR2 exists; that slow
path itself writes through `0x1700` plus the `0x700000` PRAMIN window.
`gf100_bar_oneinit()` then creates BAR2 first and BAR1 second.  Therefore a
literal BAR/instmem translation cannot bypass H57: its first page-table write
already depends on the physical PRAMIN visibility that cold Python lacks.
Lifecycle parity is still worth auditing, but each added slice must have a
hardware-visible prediction; translating allocator and object scaffolding
without restoring PRAMIN is circular.

### Night41l result — PRAMIN lifecycle timeline

The golden mmiotrace was searched backward from its first `0x700000` access.
It maps BAR0, reads boot/strap/board state, reads `0x1700=0xffb0`, writes only
the selector `0x1700=0xfffe`, and immediately reads live PRAMIN
`0x700000=0xbeef`.  There is no preceding Nouveau GPU-initialization write
that can be copied as a PBUS enable.

Night41l sampled four dwords at fixed physical address zero while restoring
`0x1700` after every probe:

| Boundary | Physical PRAMIN result |
|---|---|
| inherited cold ownership | `ffffffff` ×4 |
| full `PMC_ENABLE` | `bad0fb00,09,0a,0b` |
| VBIOS devinit | `bad0fb0c..0f` |
| PMU/PGOB | `bad0fb10..13` |
| RAM init, train `0,0,0,0` | `bad0fb14..17` |
| FB page + LTC/ZBC init | `bad0fb18..1b` |

The monotonically advancing low byte demonstrates one persistent PBUS stub
response, not framebuffer contents that become valid and are later lost.
`PMC_ENABLE` makes the client respond instead of returning all-ones, but none
of the runtime owners instantiate a physical mapping.  Late PMU MEMIF again
proved that trained VRAM and all 40 root bytes are durable behind the private
Falcon path.  H66 is disproven and H67 is confirmed observationally.

This is also the answer to a wholesale Python Nouveau port: it would improve
fidelity but would not, by itself, cross this boundary.  The golden runtime
has live PRAMIN before its first initialization write, and Nouveau's own
`nv50_instobj_wr32_slow()` requires that aperture to build BAR2.  The missing
owner is earlier than the runtime driver trace—platform/option-ROM POST or an
equivalent undocumented sequence.  Porting that proven pre-runtime slice may
work; mechanically translating the remaining allocator, BAR, FIFO, and GR
objects first will consume the same stub and reproduce Night41l.

### Night41m entry — inherited ROM-window ownership

Nouveau's BIOS source order exposes the missing owner more precisely than the
runtime FB/BAR paths.  `shadowramin.c:pramin_init()` first requires display to
be enabled (`0x22500[0]=0`), then requires `PDISPLAY.VGA.ROM_WINDOW`
(`0x619f04`) to have enable bit 3 set and target bits `1` (VRAM).  It derives
the high-VRAM shadow address from that register, changes only `0x1700`, and
reads the ROM through `0x700000`.  It does not initialise display, VRAM, PBUS,
or the ROM window.  Separately, `shadowrom.c` calls `nvkm_pci_rom_shadow()` to
toggle PCI config offset `0x50`; the golden BAR0 mirror is `0x088050`.

The new `--probe-rom-shadow-ownership` path mirrors those predicates with five
BAR0 reads and no GPU write.  On the fresh Night41m enclosure replug it found:

```
0x22500=0x00000100 display_enabled=True
0x619f04=0x00000001 window_enabled=False target=1 base=0
0x88050=0x00000000 shadow_enabled=False
0x1700=0x00000000 PRAMIN[0]=0xffffffff
ramin_eligible=False firmware_shadow_ready=False
```

The golden Nouveau entry instead has `0x619f04=0x00fffe09`: enabled, VRAM
target, base `0xfffe0000`.  It temporarily selects that base through `0x1700`
and immediately reads live data (`0xbeef`) before any init-table write.  That
word is not a valid `0xaa55` ROM signature; Nouveau subsequently tries PROM, so
the trace proves an inherited live aperture and ROM-window configuration, not
that this VRAM page contains a valid VBIOS copy.  Thus H68 is a correlation at
the ownership boundary rather than a proven activator.

The card ROM explains the host mismatch.  It contains a legacy x86 image at
raw offset `0x600` (size `0xf600`) and a compressed EFI image at `0xfc00`
(machine `0x8664`, x86-64; compression type `1`).  The current host is arm64,
so neither image is natively executable by its firmware.  A literal runtime
Nouveau port therefore remains downstream of the blocker.  The technically
relevant port is the pre-runtime option-ROM slice, preferably obtained first
as an x86 hardware MMIO trace rather than guessed register writes.

### Night41n result — exact ROM-window/PRAMIN A/B

Night41n converted H69 from a tempting register replay into a direct causal
test.  The run preserved valid RAM training (`0,0,0,0`), completed the
autonomous ENTER/XFER/LEAVE root transfer, and loaded all 40 root bytes back
exactly through PMU MEMIF.  Before any physical BAR1 or VM enable, PMU then
read 16 bytes at the exact golden physical address `0xfffe0000`:

```
PMU physical: f3ccff5bff64ffc7ffefbf23f7beff4f
PRAMIN before:            04fbd0ba06fbd0ba07fbd0ba08fbd0ba
PRAMIN after 0x619f04=0x00fffe09:
                          09fbd0ba0afbd0ba0bfbd0ba0cfbd0ba
PRAMIN after 0x088050[0]=1:
                          0dfbd0ba0efbd0ba0ffbd0ba10fbd0ba
```

All three PRAMIN samples are the same monotonically advancing `bad0fb` PBUS
stub class and none matches the PMU physical bytes.  The helper restored
`0x619f04=1`, `0x088050=2`, and `0x1700=0x10`; an independent read-only probe
observed those exact values afterward.  The intentional H69 stop occurred
before physical BAR1/VM activation.

H69 is therefore disproven, H68 is demoted from cause to correlation, and H70
is the leading boundary: firmware establishes additional state not represented
by the two visible final registers.  A whole runtime Nouveau port still starts
too late.  The next useful artifact is an x86 option-ROM/platform trace that
includes PCI configuration and VGA I/O as well as MMIO, ending at the first
live PRAMIN read.

### Night41o result — complete readable golden-entry diff

The golden mmiotrace was cut mechanically at line 42907, its first real device
initialisation write (`0x140=0`, interrupt unarm).  Excluding the sequential
PRAMIN/PROM aperture reads and their temporary selector/shadow writes leaves
exactly seven unique read-only MMIO registers.  The new
`--probe-golden-preinit` path read those seven in trace order on a fresh card:

| Register | Golden | Night41o cold | Result |
|---|---:|---:|---|
| `PMC_BOOT_1 0x000004` | `0` | `0` | exact |
| `PMC_BOOT_0 0x000000` | `0x0e4040a2` | `0x0e4040a2` | exact |
| `PSTRAPS 0x101000` | `0x8040509a` | `0x8040509a` | exact |
| `PDISPLAY 0x022500` | `0x100` | `0x100` | exact |
| `ROM_WINDOW 0x619f04` | `0x00fffe09` | `1` | known difference |
| `PRAMIN selector 0x001700` | `0xffb0` | `0` | known difference |
| `PCI shadow mirror 0x088050` | `1` | `0` | known difference |

Thus H72 is disproven: the runtime trace exposes no fourth inherited readable
register to test.  The only mismatches are the exact three states Night41n
already reproduced and disproved as sufficient.  The replug remained otherwise
untouched; this probe performed no GPU write.

Static inspection also explains the blind spot.  The legacy ROM's 16-bit entry
at image offset `0x50` jumps to `0x2caa` and enters an environment-dependent
routine with native VGA and device-I/O-BAR transactions.  The separate
`0x2f0a` graphics/sequencer routine belongs to INT 10h font service
`AH=0x11, AL=0x04`; it is not on the proven pre-NVINIT path.  The ROM's
`0x619f04` occurrence is embedded NVINIT bytecode, which Python already
interprets, not a standalone x86 MMIO activation write.  H73 therefore points
to the proven native-I/O prefix and other pre-runtime domains, not to the font
service.

A whole runtime Nouveau port cannot recover operations that precede the
runtime trace.  A literal option-ROM translation might, but it would also have
to reproduce its 16-bit firmware environment and port-I/O semantics.  The
smaller reliable route remains a three-space trace on x86 followed by replay of
only the producer prefix ending at first live PRAMIN.

### Night41p result — reachable option-ROM VGA prefix

Static reachability is now established for one concrete pre-NVINIT side
effect.  The legacy image follows
`0x0050 -> 0x2caa -> far 0x657e -> call 0x67ae`.  That function performs this
ordered native-I/O prefix before the NVINIT calls:

```
OUT 0x3c3, 0x01; IN 0x3c3
OUT 0x3c2, 0x01; IN 0x3c3
extended CRTC unlock (CR3f=0x57)
```

Python now has an explicit, offline-asserted diagnostic for those six accesses
through NVIDIA's BAR0 VGA alias.  The Night41p cold A/B recorded exactly four
byte accesses at `0x6013c3/0x6013c2` followed by the two CRTC bytes at
`0x6013d4/0x6013d5`.  Immediate entry probes were unchanged:

```
before: topology=0xbadf1200 PRAMIN[0]=0xffffffff
after:  topology=0xbadf1200 PRAMIN[0]=0xffffffff
```

H74 is therefore disproven for the BAR0 alias, and the normal devinit path
still performs only Nouveau's source-backed CRTC unlock.  This does not close
native PCI I/O: the ROM's x86 `OUT/IN` transactions and BAR0's VGA alias are
not proven to have identical bridge/device side effects.  The next causal test
must capture or execute native PCI I/O, native PCI config, and BAR0 MMIO in one
ordered timeline.

The same proven entry path continues through image `0x484d` and `0x3406`.  It
uses `0xcf8/0xcfc` to discover PCI config dword `0x24`, writes
`0x2469fdb9` and `1` to the resulting I/O BAR at `+0/+4`, then uses `+8/+12`
to read-modify-write indexed register `0x200` with bit `0x10000`.  TinyGPU has
no native PCI-I/O RPC, so Night41p did not test these operations; they are H75,
the leading bounded replay candidate.  Porting all runtime Nouveau remains downstream of this
boundary; porting additional ROM routines without reachability and an
immediate PRAMIN prediction would be another unfalsifiable bulk translation.

### Night41q result — literal Nouveau INIT_IO special case

The line-by-line interpreter audit found two real bugs.  `_reg_mask()` clipped
the value to the clear-mask, unlike Nouveau's `(old & ~mask) | val`; that made
`INIT_OR_REG` a no-op.  The Palit script also executes `INIT_IO` at `0x85bb`
with `port=0x3c3, mask=0, data=1`, while Python skipped it.  Nouveau gives this
exact NV50+ case a special ten-write sequence over `0x614100`, `0xe18c`,
`0x614900`, and `PMC_ENABLE[30]`, separated by two 10 ms waits.  Python now
ports that sequence literally, plus the executed `INIT_CR`, `INIT_ZM_CR`, and
`INIT_ZM_CR_GROUP` semantics; offline tests assert every write and port access.

Night41q ran only that one `INIT_IO` opcode on a fresh card, before RAM, PMU,
PGOB, GR reset, or global engine enable:

```
before: topology=0xbadf1200 PRAMIN[0]=0xffffffff
after:  topology=0xbadf1200 PRAMIN[0]=0xffffffff
```

Thus H76 is confirmed as a code defect and fixed, but closed as the missing
PBUS activator.  H77 remains important because the normal Python path still
orders engine reset/enable and PMU/PGOB differently from Nouveau.  H75 remains
the strongest concrete pre-runtime candidate that has not been executed: the
legacy ROM's native device-I/O-BAR writes cannot be issued by the current
TinyGPU protocol.

### Night41r result — active I2C exposed the last NVINIT error-contract gap

Night41r entered with live BAR0, marker clear, topology `0xbadf1200`, selector
zero, and `PRAMIN[0]=ffffffff`.  The guarded lifecycle probe forbade PCI reset
and strap override, unlocked VGA state once, and began the literal seven-script
POST.  It reached two real `I2C_IF` conditions on DCB primary bus `0x80` mapped
through CCB2 to GF119 register `0xd054`.  The first read completed; the second
NACKed, and Python aborted before the POST boundary.

Nouveau does not abort that opcode on a protocol failure: `nvkm_i2c_rd()`
returns `-EIO`, the result is assigned to an unsigned byte (`0xfb`), and the
condition simply becomes false.  Python now follows that per-opcode contract.
Raw TinyGPU/MMIO failures remain fatal; only device-level I2C absence/NACK uses
Nouveau's branch semantics.  Night41r therefore classified no PRAMIN boundary
and consumed no RAM experiment, but it removed the last observed divergence.

### Night41s result — corrected POST activates physical PRAMIN

Night41s repeated the same guarded cold entry and completed all seven top-level
scripts.  The two I2C conditions followed the live board outcomes (`0xff`, then
NACK→false), and `INIT_GPIO_NE` completed.  Before any RAM, LTC, PMU, BAR, GR,
or FIFO operation, the fixed physical snapshot changed from a virgin aperture
to four non-stub data dwords:

```
entry current PRAMIN[0] = ffffffff
fixed PA 0xfffe0000 after POST =
  fffd7e7f 0800e100 efff55ff 01001000
```

The probe intentionally stopped there.  This is the first cold causal win over
the old `ffffffff`/`bad0fb` boundary.  It confirms H77 at the POST boundary and
cold-supports the H78 parity cluster.  It also refines H70: external x86 option
ROM/native-I/O state is sufficient to explain the golden inherited state, but
it is not necessary for this Python path because corrected Nouveau-compatible
NVINIT POST itself can instantiate readable PRAMIN.

The successful trace contains 803 MMIO writes and ends with the POST marker,
then `0xd628`/`0xd604`, followed only by the selector snapshot.  Offline replay
shows why top-level script `0xac95` is a strong candidate: script `0xabb6`
leaves `execute=3`; the former Python state leak suppressed `0xac95`, while a
fresh Nouveau invocation state lets it emit `0xd628` and the GPIO trigger.
That correlation is H79, not yet proof; an earlier large script may already
have activated the aperture.

### Night41t instrumentation — top-level POST boundary bisector

`--probe-nouveau-post-script-bisect` is now armed offline.  It enforces the
same live/unposted/PRAMIN-negative entry, forbids reset and strap override,
keeps one interpreter alive for nested-call fidelity, and executes these
top-level scripts in order:

```
87e5 8fe8 64ff abb5 abb6 ac95 acfb
```

After each script it samples four dwords at fixed PA `0xfffe0000`, restores
`0x1700`, and stops immediately at the first positive boundary.  It never
enters RAM.  Both middle selftests, the 24-checkpoint mmiotrace suite, Python
compilation, 78 Falcon-oracle tests, 33 x86-ROM tests, and `git diff --check`
pass.  The source-parity audit additionally fixed the per-top-level RAMCFG
cache lifetime and literal `INIT_CR_INDEX_ADDRESS_LATCHED`; neither opcode
explains Night41s, but both are now covered offline.  The next live run requires a fresh
enclosure replug and should be logged as
`logs/night41t-nouveau-post-script-bisect.rpc`.
