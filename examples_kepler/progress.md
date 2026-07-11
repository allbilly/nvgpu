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
