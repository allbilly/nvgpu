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
