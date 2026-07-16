# reset_egpu.md — GK104 eGPU BAR0 / PCI recovery

**Status (2026-07-17):** Procedure implemented in userspace
(`_gk104_ensure_bar0_mmio`). Offline FakeHW cases pass.

**Live status:**

| When | Result |
| --- | --- |
| 2026-07-16 pre-replug | Probe: `PCI_ID=0xffffffff` (tier 1 config/link loss) |
| 2026-07-16 post-replug | Probe: `PCI_ID=0x118410de`, `COMMAND=0x0007→0x0403`, `mse_was=True`, `PMC_BOOT_0=0x0e4040a2` |
| 2026-07-16 cold add after replug | Passed `map-bars` / `vbios-devinit` / `firmware-load` / `channel-build`; failed **PRAMIN** store at `0x100000` (`wanted=0x0 actual=0xffffffff`) — FB/PRAMIN path, not dead BAR |
| 2026-07-16 evening debug | **Root cause:** host `0x1620`/`0x26f0` pause collapses TinyGPU BAR0 to all-ones (proven: after pause `boot0=0xffffffff`). MEMX ENTER hangs on FB_PAUSE; MEMX WR32/DELAY EXEC work (74 execs / 208 words observed). Defaults: `KEPLER_RAM_BLOCK=0`, `KEPLER_RAM_MEMX_WR=1`. False POSTed fixed. After `0x1620` experiments the card stays GPC-awake with dead PRAMIN — **physical replug**, then cold `add.py` only (no probe, no `RAM_BLOCK=direct`). |
| 2026-07-16 night | After replug: PRAMIN **writeback** works (do not treat `0xffffffff` as stub — virgin GDDR). FECS reaches ready. **VRAM address space only stable for 512 KiB**: writing the upper half of any 1 MiB region destroys the lower half (bit19 / incomplete GDDR address train). `repair_zero(GR ctx 1MiB)` therefore fails. MEMX ENTER still times out and can kill BAR0. **Need clean replug**; next work is deeper RAM train (or pack all instmem into one 512 KiB bank). |
| 2026-07-16 night2 | Clean cold: host or MEMX WR32 **without** pause leave `0xbad0fb` stub. Applying ENTER side-effect masks **via MEMX WR32** (not host). Bug: MEMX flush fallback was replaying `0x1620` on host → killed link; fallback now **skips** 0x1620/0x26f0. Also MEMX_WAIT for train(). **Replug** then cold `add.py`. |
| 2026-07-16 night4 | Patched MEMX ENTER (skip FB_PAUSE wait) still kills TinyGPU — falcon `nv_wr32(0x1620)` is as hostile as host/MEMX-WR32. Default stays `KEPLER_RAM_BLOCK=0`. Stub→live PRAMIN still unsolved without pause. **bit19 fix:** split GR runtime/golden into separate 512 KiB banks; shrink attrib CB to fit one bank (`KEPLER_VRAM_BIT19_SAFE=1`). Need clean **physical replug**, then cold `add.py` (hope virgin PRAMIN writeback like night note). |
| 2026-07-16 night5 | Host **read** of `0x1620` is fine (`0xaab`). **No-op write** and **bit0-only clear** keep BAR0 alive; clearing `0x1620[0xaa2]` kills the link. After bit0 clear, PRAMIN moved `0xbad0fb`→`0xffffffff` (virgin). Aggressive writeback probe then dropped the link. macOS default `KEPLER_RAM_BLOCK=bit0`; softened PRAMIN live probe. **Need replug**, then cold `KEPLER_N=8` add. |
| 2026-07-16 night6 | Cold `KEPLER_N=8`: first MEMX WR32 (`0x10f468`) hung PMU; subsequent loads stay deaf (ring full). Early bit0 + soft PRAMIN live coded. On wedged PMU, bit0 drops `PMC_BOOT_0` to `0xffffffff` (unlike clean night5). Refuse host GDDR5 **and** bit0 when MEMX dead. |
| 2026-07-16 night7 | After replug: MEMX ready, but **mid-MEMX bit0 kills BAR0**. Fixed: host `prog0`/`0x10f808`, MEMX for the rest, **post-MEMX bit0** unstub; boot POSTed check uses `soft=False`. Residual `GPC topo=0` (not `0xbadf`) leaves PMU deaf after WR32 timeout — FLR clears ring sometimes but not always. |
| 2026-07-16 night8 | True cold + `KEPLER_RAM_PROGRAM=bit0-only` with **both** `0x1620`/`0x26f0` pause+unpause **killed BAR0**. Next: only `0x1620[0]` clear (night5), no `0x26f0`, no unpause; `KEPLER_RAM_INIT=0` with bit0-only. **Need enclosure power cycle.** |
| 2026-07-16 night9 | Dirty (GPC-awake + PRAMIN stub) + all-MEMX reached 78 execs + post-MEMX bit0 with live BAR0; then soft PRAMIN **read** via `0x1700`/`0x700000` killed BAR0. WindowServer watchdog (~42s, display OFF) around hybrid attempts. Soft live now **skips** PRAMIN poke; macOS defaults `KEPLER_REFUSE_DIRTY=1`. **Enclosure power cycle**, then cold `KEPLER_N=8` only (no probe). |
| 2026-07-16 night10 | True cold: all-MEMX 78 execs + bit0 + soft-accept OK; then `fb_init_page`/`LTC`/`ZBC` (`0x100c80`/`0x17ea*`) collapsed BAR0 → continued with all-ones topology → PRAMIN store fail. macOS defaults `KEPLER_POST_RAM_LTC=0`; abort on dead BAR0. **Need another enclosure power cycle**, then cold `KEPLER_N=8`. |
| 2026-07-16 night11 | True cold + LTC skip: reached live clock/PGRAPH diag; **host write `0x4041f0` (BLCG, even no-op 0→0) killed BAR0**. macOS defaults `KEPLER_PGRAPH_BLCG=0`; more `require_bar0` checkpoints. Link down after run — **enclosure power cycle**, then cold `KEPLER_N=8`. |
| 2026-07-16 night12 | BLCG skipped; entire `GK104_PGRAPH_PACK_MMIO` after post-MEMX bit0 completed then `PMC_BOOT_0=0xffffffff`. **Defer bit0** (`KEPLER_RAM_BIT0_DEFER=1`) until first PRAMIN store so PGRAPH/FECS run before unstub. **Need enclosure power cycle.** |
| 2026-07-16 night13 | **Deferred bit0 worked:** PGRAPH pack + FECS IMEM match + **FECS ready** (`FE_PWR=0x2`, gpc=4 tpc=2×4). Then deferred bit0 OK; PRAMIN XOR+**literal 0** fallback left `actual=0xffffffff` and hung MMIO. Keep `KEPLER_RAM_BIT0_DEFER=1`; default `KEPLER_PRAMIN_LITERAL=0` (XOR-only). **Need enclosure power cycle.** |
| 2026-07-16 night14 | Same FECS-ready path; XOR-only still died on **host `0x1700` alone** after deferred bit0 (before any `0x700000` access). Default `KEPLER_PRAMIN_MEMX=1` (MEMX WR32 for PRAMIN window+stores, virgin XOR). **Need enclosure power cycle.** |
| 2026-07-16 night15 | FECS ready again; deferred bit0 OK; MEMX-PRAMIN failed on **PMU data-segment acquire** (`0x10a580=0xffffffff`) — PMU wedged after FECS path. Fix: reload PMU falcon + MEMX rediscovery before PRAMIN MEMX WR32. **Need enclosure power cycle.** |
| 2026-07-16 night16 | True cold reproduced night15 exactly: FECS ready, topology `4x2`, deferred bit0 left `PMC_BOOT_0=0x0e4040a2`, but the internal-CPUCTL PMU reload still ended at `0x10a580=0xffffffff`. Root cause from local Nouveau: GK104 `gf100_pmu_reset()` resets the **whole PMU subdevice through `PMC_ENABLE[13]` (`0x2000`)**, not through `0x10a100`. Recovery now pulses that MC bit, waits for `0x10a10c[2:1]` scrub completion, then reloads. Offline middle + 24/24 mmiotrace pass. **Needs enclosure power cycle for silicon validation.** |
| 2026-07-16 night17 | True cold again reached FECS ready + topology `4x2`; deferred bit0 kept `PMC_BOOT_0=0x0e4040a2` but made **all PMU host MMIO inaccessible** (`PMC_ENABLE`, `0x10a10c`, and ring all read `0xffffffff`). The correct MC-level `PMC_ENABLE[13]` reset did not restore the aperture (`0x10a10c` stayed all-ones), proving this is not a recoverable Falcon wedge. New macOS-only plan: one pre-bit0 MEMX EXEC programs `0x1700`, clears bit0, and writes a minimal BAR1 instance/PGD/two-PTE bootstrap (`0x484` bytes within the `0x800` MEMX segment); expand and verify the remaining SPT through BAR1. Gated by `KEPLER_TINYGPU_ATOMIC_BAR1=1`; shared/Linux does not enable it. Fake XOR-BAR1 end-to-end test + all offline gates pass. **Needs enclosure power cycle for night18.** |
| 2026-07-16 night18 | True cold again reached FECS ready + topology `4x2`, then the atomic order `0x1700`, bit0, PRAMIN roots collapsed all BAR0 (`PMC_BOOT_0=0xffffffff`). Local MEMX Falcon source confirms WR32 pairs execute serially: after bit0 hides PMU MMIO, the following `nv_wr32(0x700000...)` cannot complete safely. Fix: before bit0, stage only the 10 consumed root dwords in a `0x5c` MEMX script and require its reply; then use the night5/13–17-proven host bit0-only clear and expand through BAR1. Still macOS-wrapper-only; shared/Linux default is unchanged. Offline middle, fake XOR-BAR1 end-to-end, direct shared software demo, and diff checks pass. **Needs enclosure power cycle for night19.** |
| 2026-07-16 night19 | Pre-bit0 root staging succeeded with a real PMU reply (`bytes=0x5c`); the host bit0-only clear kept `PMC_BOOT_0=0x0e4040a2`. The next operation, `0x070000` BAR flush, was incorrectly issued while `0x1704` was disabled; trace status reads then took ~43 ms until timeout, and the enable store was never reached. Local Nouveau does `gf100_bar_bar1_init()` first and `gf100_bar_bar1_wait()` second. Fix: remove the pre-enable BAR/VMM flushes, enable `0x1704`, then flush twice. Offline middle, fake XOR-BAR1 end-to-end, direct shared software demo, and diff checks pass. **Needs enclosure power cycle for night20.** |
| 2026-07-16 night20 | Root staging and host bit0 succeeded again. Host `0x1704` enable completed in 9 us, proving night19's order was corrected, but the following post-bit0 `0x070000` flush still became repeated ~43-ms reads and timed out. Thus the BAR/LTC flush block itself is inaccessible after bit0, independent of BAR1 enable state. Fix: one `0xa4` pre-bit0 MEMX script now writes roots, enables `0x1704`, performs both `0x070000` flushes and Falcon-side waits, and requires the PMU reply before host bit0. After bit0 the TinyGPU-only path uses exact BAR1 readback instead of `0x070000/0x070010/0x070004`; shared/Linux defaults remain unchanged. Offline middle, fake five-command MEMX/XOR-BAR1 end-to-end, direct shared software demo, and diff checks pass. **Needs enclosure power cycle for night21.** |
| 2026-07-17 night21 | The full `0xa4` pre-bit0 roots+enable+two-flush script returned successfully; host bit0 again kept BAR0 live. The first actual BAR1 read was reached and returned `0xbad0fb03/12/13/14` instead of the staged control PTEs, proving BAR1 is enabled but its root walk is invalid. Remaining variable: pre-bit0 PRAMIN may use literal rather than virgin-XOR stores. Fix: a `0x148` script stages two complete trees in separate bit19-safe banks (literal at 2 MiB, XOR at 1 MiB), enables/flushes both before bit0, then tests literal first and safely switches `0x1704` to XOR if needed; exact two-PTE equality selects the tree. Shared/Linux remains unchanged. Fake XOR model selects XOR and expands the full BAR1 identity; all offline gates pass. **Needs enclosure power cycle for night22.** |
| 2026-07-17 night22 | Both complete pre-bit0 PRAMIN trees were invalid: literal active returned the same structured `0xbad0fbxx` walk-failure sentinel and XOR returned all ones after the safe `0x1704` switch (night21 had already produced the sentinel with XOR active). Thus pre-bit0 PRAMIN acknowledges stores but does not populate physical VRAM, regardless of encoding. Fix for night23: PMU Falcon direct-VRAM DMA (`MEMIF.TYPE=4`) stores the 40 root bytes and DMA-loads each fragment into separate DMEM scratch for exact verification, then enables/flushes BAR1 before the proven host bit0 clear. The macOS wrapper alone enables this path; shared/Linux defaults remain unchanged. End-to-end fake DMA/BAR1 expansion, middle selftest, shared software demo, compile, and diff checks pass. **Needs full enclosure power removal for night23.** |
| 2026-07-17 night23 preflight | The attempted cold run stopped at the dirty-state guard before PMU DMA: `PMC_BOOT_0=0x0e4040a2`, GPC topology was already `0x40004`, and PRAMIN/DMEM returned `badf` sentinels. The RPC trace contains only 22 map/boot-state operations; bit0 was not crossed and no FLR, probe, forced-dirty run, or retry followed. A cable replug did not remove GPU auxiliary power. **Fully power off the enclosure until PSU/GPU lights and fans are dark, then power on for the real night23 run.** |
| 2026-07-17 night23 | Full power removal produced a true-cold card. The run again reached FECS ready with topology 4×2, then submitted the first 16-byte PMU direct-VRAM DMA store. Hardware accepted `XFER_CTRL=0x220`, but `XFER_STATUS=0x10012` remained busy with one store pending for the full timeout; bit0 was never crossed. Root cause: port0 type was configured, but Falcon MEMIF itself was not activated for a host-originated transfer. Fix for night24: read-modify-write MEMIF.CTRL `ENABLE` (bit4) and `IGNORE_ACTIVATION` (bit7), require both bits to read back, then select direct VRAM. Timeout traffic is bounded to 50 ms with 500-us polling. Fake DMA now requires the same activation state; compile, middle selftest, shared software demo, and diff checks pass. **Needs full enclosure power removal for night24.** |
| 2026-07-17 night24 | True cold again reached FECS ready/topology 4×2. PMU MEMIF activation and direct-VRAM port selection read back exactly (`CTRL 0x110→0x190`, port0 `0x110→0x114`); the first DMA store and load both completed, but loadback remained 16 bytes of `ff`. Thus pre-bit0 physical VRAM discards writes independently of PRAMIN or DMA encoding; bit0 was not crossed. Fix for night25: stage roots in FECS firmware's reserved `xfer_data` DMEM (`0x200..0x2ff`), preconfigure/read back FECS physical-VRAM `MEM_BASE/MEM_TARGET`, cross bit0, then use FECS DMA store→load verification before enabling BAR1. The fake requires post-bit0 ordering and models the FECS target; compile, middle selftest, shared software demo, and diff checks pass. **Needs full enclosure power removal for night25.** |

**MSE-only recovery** (config ID live, MSE clear → boot0 restored *without* replug) has **not** yet been demonstrated on silicon.

---

## Do not assume replug from `PMC_BOOT_0` alone

| Observation | Means | Does **not** mean |
| --- | --- | --- |
| `PMC_BOOT_0 == 0xffffffff` | BAR0 MMIO decode failed / not mapped | GPU silicon is hard-dead |
| UT3G USB4 link up | Enclosure / upstream link present | GPU BAR0 is healthy |
| IOPCIDevice still present | Provider may still exist | Memory Space Enable is on |

Apple PCIDriverKit: `IOPCIDevice::Close()` clears **Memory Space Enable** and
**Bus Master Enable**. After a DEXT crash, unload, or Close, a new process must
re-Open and re-arm those bits before BAR MMIO.

Plausible false-dead path:

```text
old test hangs
  → TinyGPU / DEXT Close or restart
  → macOS clears PCI_COMMAND.MSE (+ often Bus Master)
  → new server maps BAR0 without restoring MSE
  → PMC_BOOT_0 reads 0xffffffff
  → misdiagnosed as “GPU hard dead / must replug”
```

---

## Code entry (this repo)

| Piece | Location |
| --- | --- |
| Recovery helper | `examples_kepler_pcie/add.py` → `_gk104_ensure_bar0_mmio()` |
| Cold bring-up | `NVDevice._init_hardware()` (phase `map-bars`) |
| macOS probe | `examples_kepler/add.py` → `_probe()` |
| TinyGPU CFG/MAP/RESET RPC | `APLRemotePCIDevice.read_config` / `write_config` / `bar_info` / `reset` |
| Offline FakeHW | `kepler_selftest()` — MSE-off, config-lost, reset recovery |

TinyGPU.app DEXT sources are **not** in this tree (signed binary). Client RPC
already exposes `CFG_READ` / `CFG_WRITE` / `MAP_BAR` / `RESET`. Ideal DEXT
startup remains: Open → restore COMMAND → map BARs; userspace recovery is the
safety net until that lands in TinyGPU.

---

## Operator playbook

Prefer **one** shared server; do not thrash-restart TinyGPU hoping BAR0 heals
without reading config first.

```bash
# 1) Exactly one server (GUI TinyGPU ≠ server)
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU server /tmp/tinygpu.sock

# 2) Diagnose (uses MSE recovery)
python3 examples_kepler/add.py --probe

# 3) If PMC_BOOT_0 chip_id is 0xe4, cold add:
export KEPLER_LIVE_ACK=completion-abort-risk
export KEPLER_RPC_TRACE=logs/rpc-$(date +%Y%m%d-%H%M%S).log
python3 -u examples_kepler/add.py
```

### How to read logs

| Log | Meaning | Action |
| --- | --- | --- |
| `PCI_ID=0x118410de`, `mse_was=False`, live `PMC_BOOT_0` | MSE was cleared; software restore worked | **No replug** — TinyGPU reopen should always do this |
| `PCI_ID` valid, `MSE` on, `reset=True`, then live boot0 | MMIO hung; FLR/hot reset worked | Prefer fixing DEXT reopen; replug not required that time |
| `PCI_ID` valid, MSE on, reset done, boot0 still dead | Hung MMIO/link | Enclosure power / link cycle |
| `PCI_ID=0xffffffff` | Config space gone | Enclosure power / Thunderbolt link cycle |

Healthy GK104: `PCI_ID≈0x118410de`, `PMC_BOOT_0` chip_id `0xe4` (e.g. `0x0e4040a2`).

---

## If MSE is on but BAR0 stays all-ones

Preferred DEXT / client order:

1. Stop submit / disable IRQs / tear down DMA mappings  
2. Clear Bus Master Enable  
3. Release old BAR mapping  
4. `Reset(FunctionReset)` then HotReset if unsupported  
5. Re-Open → restore MSE → remap BARs → full GPU init → enable Bus Master **last**

---

## Offline gate

```bash
python3 examples_kepler/add.py --mmiotrace-selftest
# FakeHW MSE / config-lost / reset cases also run under --middle-selftest
```

**Live MSE-only recovery (ID live, MSE clear → boot0 restored without replug)
has not been demonstrated on silicon.** Live observation after a hung session
was tier-1 config `0xffffffff`.
EOF
