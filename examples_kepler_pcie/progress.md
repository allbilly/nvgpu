# Progress — GTX 770 / GK104 PCIe bring-up (`examples_kepler_pcie/add.py`)

Linux port of `examples_kepler/add.py` using raw MMIO via sysfs resourceN mmap
instead of macOS TinyGPU socket transport. Target: `out[i]=a[i]+b[i]` compute
kernel on the GTX 770 (GK104) at PCI 09:00.0 (unbound, no driver).

Parent progress: `examples_kepler/progress.md` (TinyGPU/macOS path).

## Hardware

- RTX 3080 Ti at 04:00.0 (bound to nvidia 595, /dev/nvidia0)
- GTX 770 (GK104) at 09:00.0 — UNBOUND, no driver
- GK104 falcon firmware in `firmware/gk104/` (extracted via `firmware/gk104/extract_fw.py`)
- VBIOS: `examples_kepler/Palit.GTX770.4096.131216.rom`

## Verified working

1. **PCIe transport works.** `--probe` reads `PMC_BOOT_0 = 0x0e4040a2` → GK104.
2. **VBIOS devinit executes.** GPC PLL locks, `0x409604=0x40004`, GPC falcons
   un-gated.
3. **FECS firmware loads and starts.** `UC_PC=0x567`, `CPUCTL=0x20` (running),
   `0x409800 bit31` set (ready).
4. **FECS ctx_chan (cmd 1) works.** CC_SCRATCH0 bit31 set, CHAN_ADDR/CHAN_NEXT
   populated with inst_tag.
5. **CHSW golden context save works** (on POSTed card). CHAN_ADDR clears,
   golden context saved to VRAM.
6. **PBDMA bind + runlist commit** reaches the scheduler (runlist pending bit
   clears on POSTed card).

## Current blocker — FECS EFI overlay jump

The GTX 770 has EFI firmware in separate overlay memory (PC > 0x10000). When
a non-FIFO interrupt fires, the FECS jumps to the overlay and gets stuck
(PC=0x1097a or 0x10137). The overlay persists across PCI secondary bus resets
and falcon resets.

### Fix applied: disable non-FIFO interrupts before ctx_chan

The EFI overlay has interrupt vectors pointing to overlay code. Disabling all
FECS interrupts except FIFO (bit 2) via IRQMCLR (0x409014) before ctx_chan
prevents the overlay jump. Re-enabling bits 2,5,8 before CHSW allows the
context switch to proceed.

### Fix applied: CHSW approach (non-firmware path)

Replaced the `wfi_golden_save` (FIFO cmd 0x9, nouveau firmware path) with the
CHSW approach from the non-firmware path (`gf100_grctx_generate`):
1. Clear CHAN_NEXT.VALID (bit 31)
2. Fake CHSW interrupt via INTR_SET (0x409000, bit 8)
3. Wait for CHAN_ADDR.VALID to clear (context unload complete)

### Fix applied: PFIFO init before PBDMA bind

Moved `gk104_fifo_init()` (SUBFIFO_ENABLE, runq init, interrupt clear) before
the early PBDMA bind. Without SUBFIFO_ENABLE, the runlist DMA never completes.

## Remaining issues

### BIND_ERROR 0x01 (BIND_NOT_UNBOUND)

After the inst pointer write, PFIFO reports `BIND_ERROR code=0x01`
(BIND_NOT_UNBOUND). The channel was already bound from a previous run. Need
to unbind (write 0 to CHAN_INST_REG) before binding, per
`gk104_chan_unbind()`.

### Runlist DMA timeout

The runlist pending bit (0x2284 + runl_id * 8) never clears on the un-POSTed
card. On the POSTed card (test66), the first runlist commit also timed out
but the second succeeded. The SCHED_ERROR (code 0x01) after bind may be
related.

### Card requires reboot after secondary bus reset

The secondary bus reset clears the POSTed state (EFI firmware). The overlay
memory persists but the card becomes un-POSTed. A system reboot is needed to
restore the POSTed state. Without a POSTed state, the FECS overlay still
causes issues (though the interrupt-masking fix helps).

## Key register reference

- FECS FALCON_BASE = 0x409000
- FALCON INTR (status) = base + 0x008
- FALCON INTR_CLR (ACK) = base + 0x004
- FALCON IRQMSET (INTR_EN_SET) = base + 0x010
- FALCON IRQMCLR (INTR_EN_CLR) = base + 0x014
- FALCON IRQMASK = base + 0x018
- FALCON CPUCTL = base + 0x100 (START=0x2, HALT=0x10, RESET=0x80)
- FALCON UC_PC = base + 0x1c0 (via CPUSTAT at 0x128 for status)
- CC_SCRATCH(0) = 0x409800 (bit31 = FECS ready/done)
- CC_SCRATCH_CLR(0) = 0x409840
- FIFO data = 0x409500, FIFO cmd = 0x409504
- CHAN_ADDR = 0x409b00, CHAN_NEXT = 0x409b04
- INTR_SET (software trigger) = 0x409000
- PFIFO INTR = 0x2100, PFIFO INTR_EN = 0x2140
- BIND_ERROR = 0x252c (code in bits[7:0])
- SCHED_ERROR = 0x254c (code in bits[7:0])
- CHAN_START_REG = 0x800004 + chan_id * 8 (ENABLE_TRIGGER = bit10)
- CHAN_SUBMIT_REG = 0x800000 + chan_id * 8 (inst pointer)
- RUNLIST_BASE = 0x2270, RUNLIST_SUBMIT = 0x2274
- RUNLIST_PENDING = 0x2284 + runl_id * 8 (bit20 = pending)
- SUBFIFO_ENABLE = 0x000204
- USERD BAR1 = 0x2254

## Test log index

- test66: POSTed card, CHSW golden save worked, runlist timeout at second commit
- test67: POSTed card, userd_mmio_base NameError fixed, runlist timeout
- test68: un-POSTed card (after secondary bus reset), FECS overlay jump
- test69: un-POSTed card, IRQMCLR before ctx_chan, FECS stuck (disabled FIFO intr)
- test70: un-POSTed card, IRQMCLR keeping bit2, ctx_chan worked, CHSW timeout
- test71: un-POSTed card, CHSW with bit8 re-enabled, golden save worked, runlist timeout
- test72: un-POSTed card, BIND_ERROR 0x01 detected, sleep caused overlay jump

## 2026-07-15 — full progress and golden-reference audit

- Read both repository progress logs in full: this PCIe log and all 2,595
  lines of `examples_kepler/progress.md`.  The PCIe implementation must retain
  the Linux raw-MMIO transport while reconciling the mature GR/FIFO work from
  the TinyGPU path; previously ruled-out experiments are not being replayed.
- Classified the reference captures.  `opencl_{ioctl,strace,rm_sequence}` is
  the proprietary RTX 3080 Ti control-plane trace and contains no raw GK104
  register sequence.  The actionable golden reference is the GTX 770 capture
  produced by nouveau while running the OpenCL add:
  `nouveau_gk104_mmiotrace.txt`, with `nouveau_gk104_trace.txt` and
  `nouveau_gk104_fecs_writes.txt` as condensed companions.
- Preserved the existing uncommitted PCIe fixes (BIND_ERROR diagnostics and
  FECS interrupt masking/re-enable).  The next implementation step is a
  phase-by-phase comparison of the golden trace against `submit_launch()`,
  beginning with channel unbind/bind, runlist submission, FECS context setup,
  USERD/GPFIFO publication, and compute launch ordering.

## 2026-07-15 — golden FIFO/channel lifecycle applied

- Extracted the authoritative channel-2 sequence from the nouveau/OpenCL MMIO
  capture.  Nouveau binds `0x800ff8a3`, sets channel control `START=0x400`,
  then commits the runlist.  During GR context replacement it writes
  `STOP=0x800`, kicks `0x2634` and waits for bit 20 to clear, changes RAMIN
  `0x210/0x214`, then starts the channel again.  It does not rebuild the
  runlist for ordinary GP_PUT submission.
- Fixed the pre-golden runlist deadlock: the old PCIe path set runlist 0's
  block bit at `0x2630` immediately before submitting it.  The new shared
  commit helper waits for any prior DMA, explicitly allows the runlist,
  commits `0x2270/0x2274`, waits for pending bit 20 to clear, and acknowledges
  only the matching `0x2a00` completion bit.
- Replaced the unsafe stale-channel cleanup with the traced lifecycle:
  stop/preempt through KICK_CHID, commit an empty list, unbind and verify zero,
  clear any old bind interrupt, bind the new instance, hard-fail on
  BIND_ERROR/readback mismatch, start, and commit the populated list.  A
  BIND_NOT_UNBOUND error can no longer be logged and ignored.
- Context pointer changes are now serialized like nouveau.  The temporary
  golden context is detached only after stop/preempt, and the runtime context
  is installed only after a second stop/preempt.  The normal path reuses the
  already-admitted runlist; the former duplicate commit is available only via
  `KEPLER_RECOMMIT_RUNLIST=1`.
- Preserved the PCIe-specific FECS interrupt masking that was required by
  tests 68-71.  The golden nouveau trace uses freshly loaded internal firmware
  and therefore has no corresponding IRQMCLR writes; that difference does not
  invalidate the observed FIFO/runlist ordering.
- Offline verification passes with `PYTHONPATH=ref`: `py_compile`,
  `--middle-selftest`, the 256-element software demo, and `git diff --check`.
  The self-test now asserts runlist allow/encoding/ack and channel
  stop/KICK ordering.  No live raw-MMIO run was attempted because opening the
  real sysfs BAR resources requires sudo/password; silicon validation remains
  outstanding and `hardware_demo=ok` is not claimed.

## 2026-07-15 — live golden-lifecycle validation

- Re-ran `--probe` as root against `09:00.0`; BAR0 access remains healthy and
  `PMC_BOOT_0=0x0e4040a2` identifies GK104.  All live add attempts used the
  required `KEPLER_LIVE_ACK=raw-mmio-risk`, fresh VRAM heap ranges, and exited
  through the client-only close path without a host hang.
- Corrected the captured engine-context-pointer lifetime.  RAMIN
  `0x210/0x214` now remains zero while the channel is bound and admitted, then
  receives `(gr_ctx.va + 0x80000) | 4` immediately before FECS `ctx_chan`.
  This matches the golden writes at trace lines 4,076,674 and 4,076,718; the
  former implementation published the pointer before PFIFO bind.
- Removed the hard-coded VRAM SPT diagnostic address.  It now selects the SPT
  cloned for the context VA's actual PGD index, so fresh
  `KEPLER_VRAM_HEAP_BASE` tests no longer report a PTE from an old heap.
- Made runlist membership reflect post-reset hardware state.  Nouveau's
  capture commits resident channel 0 plus application channel 2 (count 2),
  but the diagnostic FIFO reset removes channel 0.  The code now preserves it
  only if `CHAN_INST[0].VALID` is actually set; reset tests correctly submitted
  a single `(chid, 0)` entry and reported `0x2284=0x00100001`.
- Moved the opt-in PFIFO MC reset into the early GR reset boundary.  Resetting
  PFIFO after FECS boot can desynchronise FIFO and PGRAPH current-channel
  state.  A diagnostic settled explicit unbind of a reset-created zero slot
  did not change the following bind error; it ruled out a merely delayed
  UNBOUND transition and was removed because Nouveau does not do it for a
  fresh channel.
- Live tests on CHIDs 13, 2 (the golden CHID), and 0 all produce the same
  result: the `CHAN_INST` value latches and `CHAN_CTRL` becomes `0x11000000`,
  but PFIFO raises `BIND_ERROR=0x01` (`BIND_NOT_UNBOUND`).  Temporarily
  continuing past that error proved the latch is not success: runlist pending
  bit 20 remains set indefinitely, PFIFO raises `INTR=0x100`,
  `SCHED_ERROR=0x0d`, and `SCHED_STATUS=0x5`.  A delayed read-only snapshot
  confirmed the DMA never completes; USERD `GP_GET` and `GP_PUT` were both
  zero, so premature work submission is not the trigger.
- Restored strict fail-closed handling: any bind interrupt is terminal even
  when INST/CTRL look latched.  The remaining live blocker is therefore the
  PFIFO internal bind state established during bare-card initialization, not
  runlist encoding, the application CHID, USERD publication, or FECS context
  pointer timing.  `hardware_demo=ok` is still not claimed.
- Audited Nouveau's FIFO `oneinit` and channel-object initialization after the
  live tests.  `oneinit` constructs software CHID/runqueue objects and the
  global USERD BAR1 mapping but adds no missing PFIFO register sequence; the
  hardware order in `nvkm_uchan_init()` is bind, allow/start, then runlist
  insert, matching this port.  The remaining discrepancy is below that public
  initialization sequence or in the bare-card reset/instance-memory state.

## 2026-07-15 — native-width BAR0 fix; PFIFO bind/runlist now works

- Found the PFIFO `BIND_NOT_UNBOUND` root cause in `LinuxPCIDevice`, below the
  channel lifecycle itself.  `mmio_read32()` and `mmio_write32()` used generic
  Python memoryview slices, which do not guarantee a single 32-bit MMIO
  transaction.  A split/bytewise `CHAN_INST` command can trigger once on a
  partial value and then retrigger while already bound, exactly matching the
  impossible-looking clean pre-bind state followed by `BIND_ERROR=0x01`.
- BAR register access now uses one aligned native `ctypes.c_uint32` load/store
  with explicit range checks.  Generic BAR1 buffer transfers remain byte
  copies.  This immediately eliminated the bind interrupt on silicon:
  channel 2 binds at `0x80007621`, START is accepted, the populated runlist
  pending bit clears, and channel control reaches the scheduler-owned `0x1`
  state seen later in the golden trace.
- Re-audited the exact FIFO initialization.  Removed the old `PBDMA + 0xc0`
  write because Nouveau uses it only during PBDMA interrupt recovery, not
  init.  Added the captured MC leaf routing enable at `0x640[8]`, kept USERD's
  BAR1 VA at zero (`0x2254=0x10000000`), split GR and PFIFO MC resets, and
  reproduced the two isolated PFIFO reset toggles visible before Nouveau's
  FIFO init.
- High-placement diagnostics ruled out low aperture placement: RAMIN at
  `0xff000000` and USERD at `0xfe000000` produced the same bind failure before
  the transport fix.  Those temporary overrides were removed.  An explicit
  failed-bind stop/preempt/empty-runlist/unbind/rebind also returned the same
  error and was removed; it confirmed that accepting or retrying the malformed
  first MMIO command was not a safe workaround.
- The pre-bind RAMFC prefix is now flushed, invalidated, delayed, and compared
  byte-for-byte before `CHAN_INST`.  The FECS context-header page is similarly
  stabilized through PRAMIN, rather than trusting the known alternating
  `ffffffff/00000000` BAR1 sample.  Both checks pass on the live card.
- Live execution now advances beyond the former blocker into FECS
  `ctx_chan`.  It consistently stops after the second `ctx_load` transfer at
  firmware PC `0x097a` (reported by the diagnostic register as `0x1097a`),
  with `CHAN_ADDR=CHAN_NEXT=0x80007621`, `MEM_CHAN=0x7621`, and VM target 1.
  The channel-VMM invalidate is now issued at the exact golden point directly
  before RAMIN `0x210`, and FE power is returned to AUTO after FECS reset as
  in `gf100_grctx_generate()`; neither changes this later DMA wait.
- A low context VA (`0x00480000`, PGD slot 0) produced the same stop while
  changing `MEM_BASE` from `0x00080800` to `0x00004800`, ruling out the
  hard-coded PGD-slot-1 placement.  The override was removed.  Current live
  blocker: determine why FECS does not retire the second context-load DMA even
  though the cloned VRAM VMM PTE, RAMFC, and context header all verify and
  PFIFO scheduling is now healthy.  `hardware_demo=ok` is still not claimed.

## 2026-07-15 — golden GR context generation and runtime submission advance

- Stabilized every FECS-visible page-table leaf used by golden generation:
  the context header, pagepool, bundle, attrib, and their PDEs are rewritten
  and verified through PRAMIN immediately before `ctx_chan`, followed by the
  channel-PDB invalidate.  This removed the second context-load wait described
  above; its captured MMU record was `INVALID_STORAGE_TYPE` on the bundle VA.
- Audited `gf100_grctx_generate_main()` against local Nouveau and the OpenCL
  mmiotrace.  The 554 writes before ICMD now match the capture exactly apart
  from allocated buffer VAs: SM-ID setup precedes GPC `0xc10`, DS words all
  precede PD words, the ROP-map writes use captured order, and a duplicate
  `0x419f78` write is gone.  All 831 ICMD addresses and all 1,032 ICMD
  address/data writes match as well.  ICMD GO_IDLE now waits for both ICMD and
  whole-GR idle and fails closed after two seconds.
- Stock FECS firmware is now the default; the former `ctx_4170s` FORCE_ON
  firmware mutation is opt-in.  With stock firmware and the exact write
  stream, live `grctx_main` completes through ICMD and method packs, and FECS
  completes the golden context save.
- The runtime channel now receives an unconditional second runlist commit,
  matching the golden trace after the runtime context pointer is installed.
  Host-side `ctx_chan`, direct GR patch replay, and a redundant post-commit
  START were removed from the normal path so GP_PUT owns the automatic FECS
  load transition.
- USERD GET and PUT are published and reverified through physical PRAMIN as a
  pair.  The full runtime PTE/PDE set is restabilized immediately before
  GP_PUT.  BAR1 and PRAMIN can observe GPU-owned GET at different instants, so
  doorbell validation accepts either valid ring index without rolling a
  consumed GET back to zero.
- The latest live full-add run advances `GP_GET=1`, PBDMA `IB_GET=IB_PUT=1`,
  and starts FECS automatic context load.  It then raises PGRAPH
  `INTR=0x00200000`, `TRAP=0x01000000` (GPC), trapped address `0x80070000`,
  while VM fault source bit 0 is active.  FECS remains at PC `0x109d4`, the
  completion word remains its cold-attach drift value `0xfffffffe`, and
  `hardware_demo=ok` is still not claimed.  The next live pass captures the
  active slot-0 VM record and concise GPC subunit state; the previously printed
  slot-7 record was stale diagnostic output from the earlier PBDMA fault.
- That follow-up run identifies slot 0 as instance `0x621`, read fault VA
  `0x00100000`, type `0x442` (hub client 4, reason 2), while all four GPCs
  report TPC1.  PBDMA has fetched to push offset `0x20`; Mesa's primary NVE4
  source confirms the TEMP packet encoding and per-MP size calculation already
  match, so no speculative TEMP expansion was applied.
- A local Nouveau audit then found the runtime-context construction error.
  `CB_RESERVED+0x80000` is only the temporary golden-generation address;
  `gf100_gr_chan_bind()` copies `gr->size` bytes to a channel context at offset
  zero and RAMIN points to that base.  For external firmware its header is
  count at `0x10`, full MMIO-list VA at `0x14/0x18`, control word 1 at `0x1c`,
  and zeroes at `0x20/0x28/0x2c/0xf4/0xf8`.  The port had repaired the offset-
  zero copy but pointed FECS back at `+0x80000` and wrote the legacy non-
  firmware header there, bypassing every repair.  The pointer and header now
  match Nouveau; live validation of this correction is the next step.
- Live validation shows the base pointer and firmware header stick exactly but
  do not change the trap.  A `KEPLER_TEST_STAGE=set-object` run, containing no
  TEMP setup, reproduces the same all-GPC/TPC1 trap and VA `0x00100000`; TEMP
  backing is therefore ruled out.
- The remaining runtime copy still trusted BAR1 reads of the FECS-saved golden
  image even though that aperture is known to return complemented dwords.  It
  now captures the golden image through physical PRAMIN, copies and verifies
  the runtime image through PRAMIN, updates the firmware header in the captured
  host image, and re-publishes both context and MMIO patch list immediately
  before GP_PUT.  The next run validates this authoritative-copy correction
  and records the selected TPC subunit status.
- PRAMIN capture eliminates the thousands of apparent BAR1-copy mismatches
  (`mismatches=0` on the first physical verification) and narrows the selected
  trap to GPC2/TPC0 plus GPC3/TPC1; both report MP global `0x4`
  (`MULTIPLE_WARP_ERRORS`).  The slot-0 VA/type remain unchanged.
- A bounded one-page writable guard at `0x100000` did not alter the fault, so
  no wider mapping was added.  The next diagnostic records that leaf PTE and
  the loaded pagepool/bundle/attrib base registers at timeout to determine
  whether the leaf vanished, was not selected by the GR client, or the reason
  code is not a missing-leaf condition.
- The leaf did vanish: timeout readback is `0xffff8fee`, the exact complemented
  XOR delta of intended PTE `0x00007011`, while loaded GR buffer registers are
  correct (`pagepool=0x200`, `bundle=0x280`, attrib bases `0x80000200` and
  `0x10000200`).  Immediate compensated PRAMIN readback was acknowledging an
  XOR update before its delayed literal value settled.  Final context,
  per-channel MMIO list, and all PTE/PDE entries now use literal stores plus a
  delayed physical verification immediately before GP_PUT.
- Publishing the `0x100000` guard leaf as the final framebuffer store before
  the BAR1 GP_PUT doorbell makes it remain `0x7011` both when GP_GET advances
  and at timeout.  The previously active slot-0 MMU record is gone
  (`FAULTS=0`), while PBDMA still consumes the set-object pushbuffer and the
  same GPC3/TPC1 MP `MULTIPLE_WARP_ERRORS` trap occurs.  The translation fault
  is therefore fixed and is no longer the explanation for the GPC trap.  The
  next check treats the trap as context state: verify that the FECS-saved
  golden image is truly idle and that its runtime copy/MMIO patch header agrees
  with Nouveau before expanding the command stream or mappings.
- That audit found the runtime header had been switched to the wrong Nouveau
  branch.  The locally extracted `gk104_fecs_code.bin` is built from
  `hubgk104.fuc3.h`, so it uses the internal `!gr->firmware` protocol (FIFO
  commands 1/2).  Its channel header is the MMIO pair count at offset `0x00`
  and `patch_list_va >> 8` at `0x04`; offsets `0x10..0x2c` belong to the
  proprietary/external firmware path.  Restoring the internal header changes
  FECS from stuck PC `0x109d4` to a completed automatic load and idle PC
  `0x567`; PBDMA consumes the entire 36-byte set-object test stream.  GR still
  raises the MP-only GPC trap and remains busy after the exact Nouveau trap-ISR
  acknowledgement, so the current check records all post-ack trap levels and
  the subchannel class binding to distinguish a reasserted exception from a
  successfully bound object with invalid restored SM state.
- The post-load interrupt audit confirms that the exact Nouveau MP-trap ISR
  sequence clears every TPC, GPC, top-level trap, and PGRAPH interrupt without
  reassertion.  Nevertheless `SUBCH1` remains zero and GR stays busy, proving
  SET_OBJECT never executes and the MP exception originates in runtime-context
  restore.
- A full delayed PRAMIN audit of the nominally zero golden allocation finds a
  deterministic `0xffffffff/0x00000000` pattern: 131,072 of 262,144 dwords are
  nonzero, always the low dword of each qword.  Native 64-bit BAR1 stores,
  `resource1_wc`, inverse payloads, and page-sized compensated PRAMIN updates
  did not make those low lanes durable after changing the PRAMIN window.
- The ordering audit then found that `submit_launch()` populated all VRAM
  objects through BAR1 before enabling BAR1's identity VMM at `0x1704`.
  Initialization was moved immediately after the USERD/heap physical addresses
  are allocated and before the first framebuffer store, with the later
  duplicate bootstrap suppressed.  The first attempt to validate this ordering
  did not complete: following the BAR3 diagnostic, BAR0 returned all ones, and
  the subsequent live attempt crashed and rebooted the host.  Treat the early-
  BAR1 change and the experimental full-allocation zero proof as unvalidated.
  Do not issue further live MMIO until the BAR bootstrap/page-table ordering is
  audited offline and a minimal read-only probe is normal after reboot.
- After the host reboot, the guarded read-only probe succeeds again:
  `PMC_BOOT_0=0x0e4040a2` (`chip_id=0xe4`, GK104).  The all-ones BAR0 state was
  transient across the crash/reboot rather than a persistent PCI failure.
  Live debugging is explicitly resumed, with each risky experiment recorded
  here before proceeding; the BAR3 diagnostic is excluded from the next run.
- Fresh-reboot validation of the early-BAR1 ordering ran on a cold card through
  VBIOS/PMU/FECS bring-up and stopped fail-closed before `ctx_chan`.  Even with
  BAR1 identity enabled before the first object write, golden-context page zero
  does not stabilize after four write/flush/invalidate attempts.  Nonzero
  dwords begin at `0x0,0x8,0x10,...`, with a small discontinuity around
  `0x50..0x78`.  Thus late BAR1 enable was a real ordering error but is not the
  complete cause of the framebuffer corruption.  The next step audits the
  active BAR1 instance/PDE/PTE for the 4-MiB heap mapping and its VRAM target;
  no FECS context load or GPFIFO submission occurred in this run.
- The read-only BAR1 audit proves the translation hierarchy is correct:
  `BAR1_CTL=0x80000100`, PDB `0x110000`, PDE0 selects SPT `0x120000`, and heap
  leaves are exactly `PTE[0x400]=0x4001` and `PTE[0x402]=0x4021`.  Immediately
  after the failed clear, both BAR1 and the correctly selected PRAMIN 4-MiB
  window settle to all `0xffffffff`.  The failure is below BAR1/GMMU.
- This fresh boot also began from `badf1200` cold-domain state and the script
  ran only VBIOS devinit script 0 before clocks/PMU/GR bring-up.  The durable
  all-ones framebuffer strongly indicates that the secondary GK104's VRAM
  controller/training sequence was never executed after reboot.  Stop treating
  the symptom as a PRAMIN XOR quirk; audit and implement the VBIOS/Nouveau
  memory-init path before another golden-context allocation test.
- Per the repository bring-up rule, `examples_pcie/add_opencl` is now used as
  the health check before risky GTX 770 tests.  On this reboot it selects the
  bound RTX 3080 Ti, produces `0 3 6 9 12`, and reports `PASS`; together with
  the valid GK104 `--probe`, this confirms the host OpenCL/NVIDIA stack and the
  raw GTX 770 PCI endpoint are healthy before continuing.  It does not validate
  GTX 770 VRAM because that card remains unbound.
- The local Nouveau v7.2-rc2 source confirms the missing cold-memory phase.
  `gk104_ram_init()` runs every script pointer in the BIT-M v2 RAM-map table
  at offset `M+0x05` while selecting each mode through `0x10f65c[7:4]`, then
  programs `0x10f584`, `0x10ecc0`, and `0x10f160`.  It subsequently decodes
  BIT-M/M0205 and BIT-M/M0209 training data and fills the GK104 GDDR5 training
  ports at `0x10f900..0x10f96c`.  The current Python cold path executes only
  BIT-I devinit scripts, so neither part of Nouveau's RAM initialization is
  present.  The next step is to decode this ROM's exact BIT-M tables and port
  the bounded Nouveau sequence before another live VRAM write test.
- Re-ran the OpenCL-add health check from the Kepler PCIe directory using the
  exact path documented by its README, `../examples_pcie/add_opencl`.  It exits
  0, selects PCI device `10de:2208` / `NVIDIA GeForce RTX 3080 Ti`, prints
  `0 3 6 9 12`, and reports `PASS`.  There is no separate OpenCL binary inside
  `examples_kepler_pcie`; this confirms the host/OpenCL baseline only.  It is
  not yet a GTX 770 health pass: `09:00.0` remains unbound and NVIDIA 595 has
  no Kepler support, so the raw-MMIO add path still has to establish durable
  GTX 770 VRAM before that card can execute the kernel.
- Corrected the cold-RAM source attribution before implementation: Nouveau's
  `gk104_ram_init()` gets its three early RAM-mode script pointers from the
  BIT-P v2 RAM-map (`P+0x04`), not from BIT-M.  This Palit ROM's RAM-map is at
  `0x6f9d`, its pointer array is at `0x715e`, and the scripts are `0xa760`,
  `0xa876`, and `0xa98c`.  BIT-M separately points to M0205 at `0x824f` and
  M0209 at `0x8296` for the training data written after those scripts.  No
  live write was made from the earlier mistaken table label.
- The requested health criterion is now stricter: OpenCL add must select and
  pass on GTX 770 `10de:1184`, not merely pass on the RTX 3080 Ti.  The
  documented route is Nouveau plus Rusticl.  Current audit: `09:00.0` has no
  bound driver, both the Nouveau kernel module and Rusticl ICD are installed,
  and the proprietary NVIDIA modules remain loaded for the RTX 3080 Ti.  The
  next bounded steps are to verify/install the local GK104 firmware, attempt a
  Nouveau-only bind of `09:00.0`, and run the OpenCL binary with only
  `rusticl.icd`; each result will be recorded before resuming raw MMIO.
- Installed the four repository-local GK104 GR firmware blobs at Nouveau's
  required `/lib/firmware/nvidia/gk104/gr/` names (`fecs_{inst,data}.bin` and
  `gpccs_{inst,data}.bin`).  SHA-256 comparison confirms every installed file
  is byte-identical to its `firmware/gk104/gk104_*` source.  No driver is bound
  yet; the next step explicitly loads Nouveau with `modeset=1` despite the
  system's proprietary-driver blacklist, then checks that the RTX 3080 Ti
  remains on NVIDIA before binding only GTX 770 `09:00.0`.
- Loaded Nouveau explicitly with `modeset=1`; the RTX 3080 Ti remains correctly
  bound to NVIDIA.  Nouveau automatically probed the unbound GTX 770 but could
  not attach, logging `unknown chipset (ffffffff)`.  This is earlier than any
  firmware or Rusticl step: the kernel read all ones from GK104 `PMC_BOOT_0`.
  Thus the requested GTX 770 OpenCL health check cannot run in the device's
  present PCI/MMIO state.  Before any bind retry, audit PCI COMMAND/BAR state
  and available reset controls; do not misreport the 3080 OpenCL PASS as a
  Kepler PASS.
- PCI configuration itself remains valid after Nouveau's failed probe:
  COMMAND is `0x0007` (I/O, memory, and bus mastering enabled), BAR0/BAR1/BAR3
  retain their assigned apertures, sysfs enable is `1`, and runtime power is
  `active`.  A function reset control is available at
  `/sys/bus/pci/devices/0000:09:00.0/reset`.  Because BAR0 alone is returning
  all ones, the next recovery is the bounded sysfs function reset followed by
  a four-byte read-only BAR0 probe before asking Nouveau to bind again.
- The device-level sysfs reset is not usable in this wedged state: writing `1`
  returns `I/O error`, and a subsequent four-byte BAR0 read also returns
  `Input/output error`; PCI COMMAND and sysfs enable remain unchanged.  No
  Nouveau bind or OpenCL launch was attempted after that failed reset.  Next
  inspect the upstream PCIe bridge topology and only consider a secondary-bus
  reset if the bridge contains no other endpoint; otherwise a host reboot is
  required to recover the GTX 770.
- Topology audit shows the GTX 770 has a dedicated root-port secondary bus:
  AMD bridge `00:03.1` owns bus 09 and its only functions are GTX 770 VGA
  `09:00.0` and the same card's HDMI audio `09:00.1`.  The VGA function's
  advertised reset method is `bus`; HDMI audio is currently bound to
  `snd_hda_intel`.  The prior bus-reset failure may therefore be the bound
  sibling preventing a coordinated reset.  Next temporarily unbind only
  `09:00.1`, retry the VGA bus reset and four-byte probe, then rebind audio
  after the GPU/Nouveau health-check attempt.
- Unbound only GTX 770 HDMI audio `09:00.1` and retried the VGA function's
  advertised `bus` reset.  It still returned `I/O error`, and BAR0 remains
  inaccessible; both card functions are now unbound and PCI COMMAND is still
  `0x0007`.  Because bridge `00:03.1` is dedicated to this physical card, the
  final non-reboot recovery attempt will explicitly pulse its standard PCI
  Secondary Bus Reset bit, then restore the bit and perform a four-byte BAR0
  read before any driver bind.
- The explicit Secondary Bus Reset pulse completed and the GTX 770 still
  enumerates as `10de:1184`, but reset cleared its PCI COMMAND and BAR config
  registers.  Restoring COMMAND alone is insufficient: config BAR0/BAR1/BAR3
  now contain only zero/type bits while sysfs retains the kernel's assigned
  resource ranges, so BAR0 correctly returns `EIO`.  Before judging the GPU
  unrecovered, restore both card functions' BAR registers from their existing
  sysfs resource allocations, then enable decode and retry the four-byte probe.
- Restored VGA BAR0/BAR1/BAR3/BAR5 and audio BAR0 from the unchanged sysfs
  allocations, then restored VGA COMMAND `0x0007` and audio COMMAND `0x0006`.
  Config-space readback matches those assignments, but the sysfs BAR0 file
  still returns `EIO`; the bus reset therefore has not yet produced a userspace
  MMIO health read.  The next discriminating check is a Nouveau bind retry:
  its kernel probe maps the restored physical BAR directly.  Success will
  allow Rusticl/OpenCL; another `ffffffff` probe means only a host/slot power
  cycle can recover this card.
- The immediate Nouveau bind retry did not reach a new chipset read; sysfs
  returned `EIO` and the kernel logged probe error `-17`, while the only
  `unknown chipset (ffffffff)` line remains the earlier probe.  Nouveau has no
  bound device and module use count is zero, so unload/reload it once to clear
  any failed-probe bookkeeping and trigger a fresh kernel probe against the
  restored config.  Keep the RTX 3080 Ti on NVIDIA throughout.
- Unloading and reloading only Nouveau cleared the failed-probe state and
  succeeded: GTX 770 `09:00.0` is now bound to Nouveau while RTX 3080 Ti
  `04:00.0` remains bound to NVIDIA.  Nouveau detects the full `4096 MiB
  GDDR5`, completes clock/thermal/DRM setup, acquires both FECS and GPCCS
  falcons from the installed firmware, and initializes GR without a reported
  fault.  This independently proves the card, VRAM, firmware, and kernel
  Nouveau cold-init path are healthy after the bridge reset.  Next run the
  reference OpenCL add with `OCL_ICD_VENDORS` restricted to `rusticl.icd` and
  require its device name/PCI identity to be GTX 770 before accepting PASS.
- The first Rusticl-only launch does not yet expose a platform:
  `OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd ../examples_pcie/add_opencl`
  exits 1 with `get platform failed: -1001` (`CL_PLATFORM_NOT_FOUND_KHR`).
  This is now an OpenCL/Rusticl enumeration issue rather than a PCI, VRAM, or
  Nouveau-bind failure.  Next inspect the installed Mesa/Rusticl enable policy
  and retry with its explicit Nouveau driver opt-in if supported.
- Official Mesa documentation confirms `RUSTICL_ENABLE=nouveau` is the correct
  opt-in and lists Nouveau as a supported Rusticl driver.  Both
  `RUSTICL_ENABLE=nouveau` and `nouveau:0` still enumerate no OpenCL platform.
  The render-node/device mapping is correct (`renderD129 -> 09:00.0`) and a
  Mesa-only EGL probe on the same node succeeds as renderer `NVE4` with OpenGL
  4.3, proving the installed classic Nouveau Gallium userspace can open and
  accelerate the GTX 770.  Next audit which `libOpenCL.so` the test binary is
  actually loading and isolate the distro ICD loader before investigating a
  Rusticl compute-cap rejection.
- **GTX 770 OpenCL health check now passes.**  `ldd`/loader tracing found the
  binary was resolving CUDA 12.6's `/usr/local/cuda-12.6/.../libOpenCL.so.1`;
  that loader returned `CL_PLATFORM_NOT_FOUND_KHR` for the Rusticl-only ICD.
  Running from `examples_kepler_pcie/` with
  `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu`,
  `OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd`, and
  `RUSTICL_ENABLE=nouveau` selects `device: NVE4`, prints `0 3 6 9 12`, and
  reports `PASS`.  Since Rusticl is the only ICD and its active render node is
  `renderD129 -> 0000:09:00.0`, this is a real GTX 770 add, not the RTX 3080 Ti.
  Use this exact environment as the Kepler health preflight from now on.
- Added executable `examples_kepler_pcie/opencl_add_health.sh` and documented
  it in the local README.  The wrapper refuses to run unless `09:00.0` is bound
  to Nouveau, forces Ubuntu's ICD loader plus the Rusticl-only ICD, and launches
  the existing `examples_pcie/add_opencl`.  After correcting an initial
  working-directory typo in the `chmod` command, the wrapper itself exits 0 on
  GTX 770/NVE4 with `0 3 6 9 12` and `PASS`.  This wrapper is now the mandatory
  GTX 770 health check before returning the card to the raw-MMIO path.
- After the passing GTX 770 health check, unbound Nouveau without a reset so
  its verified memory training would remain as the baseline for raw MMIO.  The
  sysfs unbind completed (the writing `sudo` process was reported `Killed`
  during teardown, but the host stayed up and the device is unbound).  The
  immediately following guarded raw probe succeeds with
  `PMC_BOOT_0=0x0e4040a2`.  This is the desired known-good/post-initialized
  state for the next `KEPLER_TEST_STAGE=set-object` run.
- The bounded post-Nouveau `KEPLER_TEST_STAGE=set-object` run exits 0 and ends
  `hardware_set-object=ok value=2`.  Unlike the cold raw run, every GR context,
  pagepool, bundle, and attribute allocation passes delayed physical-zero
  verification; FECS `ctx_chan`, GR context generation/golden save, and the
  `0x2c400` runtime-context copy complete with `mismatches=0`; the final guard
  PTE remains literal `0x7011`.  The 36-byte SET_OBJECT test is accepted with
  no MMU fault or host crash.  This isolates the former all-ones/MP-trap path
  to missing cold VRAM initialization and validates the corrected BAR1/FECS/
  runtime-context ordering once Nouveau has initialized the card.  Before the
  full kernel launch, run the GTX 770 OpenCL wrapper again through a fresh
  Nouveau bind/unbind cycle, then preserve that initialized state for raw add.
- The attempted second Nouveau bind did **not** yield another health check.
  During teardown/re-probe, the kernel page-faulted in Nouveau
  `nve0_bo_move_copy()` via `nouveau_bo_move_m2mf()`/
  `nouveau_drm_device_fini()`.  The sysfs bind writer is stuck in uninterruptible
  sleep, `/sys/.../09:00.0/driver` remains absent, and no OpenCL wrapper was
  launched.  Do not retry Nouveau in this boot.  Close the waiting shell and
  use only a read-only raw probe to decide whether the already verified/trained
  card can proceed to full raw add; a bad probe requires reboot.
- After interrupting the waiting session, the bind child remains in kernel
  `D` state, but a separate guarded raw probe still reads the correct
  `PMC_BOOT_0=0x0e4040a2`.  The device is unbound and BAR0 remains healthy.
  The valid GTX 770 OpenCL PASS immediately preceding the successful
  set-object run remains the health baseline; avoid all further Nouveau binds
  this boot and proceed directly to one full raw add attempt.
- The full raw add makes substantially more progress and exits normally with a
  bounded timeout rather than crashing.  PBDMA consumes the GPFIFO/IB
  (`IB_GET=IB_PUT=1`, `DMA_GET=0x900009c`), FECS loads the runtime channel
  (`CHAN_ADDR=CHAN_NEXT=0x80000621`), MMU fault count remains zero, the final
  guard PTE is still `0x7011`, and all loaded GR buffer bases are correct.  GR
  raises trap `0x100` at `ADDR=0x800102bc`, `DATA=0x3`, `STATUS=0`; the output
  semaphore remains its initialized value `1`, so the kernel never launches.
  This is now a specific subchannel-1 method `0x2bc` rejection after successful
  context restore, not the previous cold-VRAM or MP trap.  Next map method
  `0x2bc` to the generated launch stream/class and compare its ordering/value
  with the Nouveau/OpenCL golden push.
- Nouveau's `gf100_gr_trap_intr()` gives an exact bounded action for this new
  shape.  Trap bit `0x100` is SKED; the driver reads `0x407020`, writes the
  SKED status register only when that status is nonzero, but always W1C-acks
  `0x400108=0x100`.  Its parent ISR then W1C-acks
  `0x400100=0x00200000` and restores `0x400500=0x00010001`.  Our observed SKED
  status is exactly zero, and an unbound raw device has no kernel ISR to perform
  those acknowledgements.  Add only an exact `INTR.TRAP + TRAP==SKED +
  SKED_STATUS==0 + no GPC trap` handler in the post-GP_GET poll; leave every
  nonzero/combined trap fail-closed.
- Implemented the exact empty-SKED acknowledgement in the one-shot post-GP_GET
  poll.  It records `0x407020`, activates only for top-level TRAP with
  `PGRAPH_TRAP==0x100`, zero SKED status, and zero GPC bitmap, then performs the
  three Nouveau-order writes; all other trap shapes remain untouched.  Offline
  gates pass: `py_compile`, `--middle-selftest`, software add, and
  `git diff --check`.  Next confirm the raw GK104 probe is still healthy and
  run one full add to see whether the semaphore advances or a more specific
  SKED/GPC fault replaces the empty latch.
- Pre-live probe after the patch still returns
  `PMC_BOOT_0=0x0e4040a2`.  The orphaned Nouveau sysfs-bind child remains in
  kernel `D` state but holds no driver link and has not prevented raw BAR0
  access.  Proceeding with one full launch; no further driver/module action is
  allowed in this boot.
- The patched full run correctly leaves the trap untouched because the stable
  SKED status is nonzero: `0x1000`, decoded by Nouveau as
  `TOTAL_TEMP_SIZE`.  PBDMA/MMU/context evidence otherwise matches the prior
  run and the semaphore remains `1`.  The earlier zero SKED read was a timing
  sample before the detailed scheduler status settled, not a benign interrupt
  to clear.  The narrow zero-status ISR handler remains dormant and the real
  fault stays fail-closed.  Next compare `TEMP_ADDRESS`, both `MP_TEMP_SIZE`
  method banks, and the QMD local-memory low/high/CRS fields against Mesa's
  NVE4 implementation and class headers.
- Mesa's NVC0 TLS sizing rule explains `TOTAL_TEMP_SIZE`: GK104 supports 64
  resident warps per MP, and this QMD requests `0x800` bytes of CRS local memory
  per warp while low/high local memory are zero.  The minimum is therefore
  `0x800 * 64 = 0x20000` per MP (already 32-KiB aligned), or `0x100000` for the
  GTX 770's eight MPs.  The current setup allocates/programs only `0x40000`
  total (`0x8000` per MP), one quarter of the scheduler requirement.  Replace
  that guessed size with constants derived from the QMD and GK104 occupancy,
  preserving Mesa's two identical MP_TEMP_SIZE banks.
- Implemented the derived TEMP constants (`64` warps/MP, `0x800` CRS bytes per
  warp, `0x20000` per MP, `0x100000` total) and use them for both allocation
  and launch-method programming.  The builder now rejects undersized or
  non-eight-way-divisible TEMP sizes.  `py_compile`, middle selftest, software
  add, and `git diff --check` pass.  Next raw-probe the still-unbound card and
  run one full launch; expected progress is disappearance of SKED
  `TOTAL_TEMP_SIZE`.
- Pre-live probe for the TEMP-size validation remains healthy at
  `PMC_BOOT_0=0x0e4040a2`; proceeding with one full raw launch and no Nouveau
  action.
- **Full raw GTX 770 add passes on hardware.**  With `0x100000` total TEMP,
  SKED `TOTAL_TEMP_SIZE` disappears, the launch and serialized completion
  semaphore retire, and the output is read from GPU VRAM at `pa=0x705000`.
  All 256 GPU-written floats match the CPU reference (`mismatches=0/256`), the
  process exits 0, and it reports `hardware_demo=ok N=256` followed by the
  zero-RPC client-only close.  The host did not crash.  This validates the
  complete post-Nouveau workflow: GTX 770 Rusticl/OpenCL health PASS, unbind
  without reset, guarded raw probe, then raw-MMIO add PASS.  Next port the
  proven TEMP sizing (and only shared source-backed fixes) to
  `examples_kepler/add.py`, update its progress log, and run both offline
  verification suites before claiming the port complete.
- Final offline verification passes for both implementations: `py_compile`,
  each `--middle-selftest`, each `NV_BACKEND=software` vector add, and scoped
  `git diff --check`.  Both selftests report 39 launch words and both software
  demos report `software_demo=ok N=256`.  The Linux hardware result remains
  `hardware_demo=ok N=256` with `mismatches=0/256`.
- Final workspace audit: the health wrapper passes `sh -n` and is executable;
  repository-wide `git diff --check` is clean.  One orphaned kernel `D`-state
  process remains from Nouveau's `nve0_bo_move_copy()` page fault during the
  attempted second bind (`pid 8857` in this boot).  Do not perform another
  Nouveau/module transition; reboot before the next bind-based health cycle.
## 2026-07-15 — standalone launcher work

- Reproduced the reported `sudo python3` import failure: the script imports
  `tinygrad` before adding any checkout path, while the package is available at
  `ref/tinygrad` and only the virtualenv currently knows that location.
- Located all remaining environment-only launch requirements.  The live ACK
  gates are in `main()`, and the built-in ROM/cubin fallbacks incorrectly point
  into `examples_kepler_pcie/` instead of the existing assets in
  `examples_kepler/`.
- Implementation plan: resolve imports/assets from `__file__`, default the FIFO
  reset used by the proven GTX 770 path, remove the explicit ACK requirement,
  and automatically re-exec through sudo only when raw sysfs MMIO permissions
  require it.  Offline/self-test paths will remain unprivileged.

- Confirmed these launcher changes are applied in `add.py`: `/usr/bin/python3`
  resolves `ref/tinygrad`, ROM/cubin defaults point at `examples_kepler/`, and
  non-root live invocation re-execs through `sudo`.
- Added `mul.py`. Its default command is a dependency-free 256-element CPU
  multiply smoke test. Hardware multiply requires an explicit `KEPLER_CUBIN`,
  because only the add sm_30 cubin is validated on this card.
- Made the shared Kepler close diagnostic conditional on `DEBUG`; normal
  successful launches no longer print that internal teardown note.
- Syntax and offline checks pass for both Kepler add files and `mul.py`; the
  normal PCIe self-test output is now only the `kepler_selftest=ok` result.
- Added quiet live-mode output filtering: the normal hardware command keeps
  only the output summary and `hardware_demo=ok`; the register/firmware trace
  remains available through the existing debug path.

## 2026-07-15 — restore post-filter-repo cleanup

- Reapplied the standalone layout: `examples_kepler_pcie/add.py` now contains
  the complete shared Linux implementation (8,180 lines); the macOS entrypoint
  is a 377-line wrapper with the TinyGPU transport embedded directly and imports
  the shared implementation with `from examples_kepler_pcie import add as shared`.
- Restored the transport-factory hook so the wrapper can inject its TinyGPU
  socket device without a second copy of the RM/GMMU/launch implementation.
- Restored local CUDA 10.2 PTX assembly for both operations, byte comparison
  against the add reference and the verified multiply digest, plus operation-
  aware hardware expectations. `mul.py` now assembles and launches through the
  shared implementation instead of requiring a precompiled cubin.
- Verification step 1: all three files compile with `python3 -m py_compile`.
- Verification step 2: Linux and macOS `--middle-selftest` both pass.
- Verification step 3: Linux and macOS `NV_BACKEND=software` demos pass
  (`software_demo=ok N=256`); the standalone multiply CPU smoke test passes.
- Verification step 4: add and mul assembly both report
  `cubin_compare=byte-identical` (1,768 bytes); mul SHA-256 is
  `edde9272ed2b6e5a98c47cd52c18cfdfec40670af77383ab14d59762bb77fbd8`.
- Final layout check: `examples_kepler_pcie/add.py` is 8,196 lines and
  `examples_kepler/add.py` is 389 lines, matching the recovered cleanup target.
- Final repeat after the documentation/count adjustment: py_compile, both
  middle self-tests, the software demo, both cubin comparisons, and
  `git diff --check` all pass.
- Wrapper smoke step: macOS add `--compare-cubin` reports byte-identical for
  both add and mul, and the standalone multiply CPU smoke test reports
  `software_mul=ok N=256`.
- Tightened the cubin gate: an available assembler that produces bytes differing
  from either reference now fails closed; only an unavailable assembler permits
  the checked-in add fallback (mul still requires local CUDA 10.2 ptxas).
- Post-gate regression check passes again: line counts remain 8,196/389,
  add and mul byte comparisons are identical, both self-tests/software paths
  pass, the mul smoke test passes, and `git diff --check` is clean.

## 2026-07-15 — MMIO trace availability

- Checked the working tree, Git refs, and unreachable Git objects for the
  historical `nouveau_gk104_mmiotrace.txt` capture and compressed variants.
  No copy is present or recoverable; only source-code/progress references
  remain. A fresh Nouveau MMIO capture must therefore be made and compressed
  (for example, `gzip -9 nouveau_gk104_mmiotrace.txt`).

