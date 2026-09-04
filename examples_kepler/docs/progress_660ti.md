# Progress — GTX 660 Ti / GK104 add_660ti.py bring-up

Live status of getting a Kepler `sm_30` `out[i]=a[i]+b[i]` (and `mul`) kernel
running on the seven-TPC GTX 660 Ti (GK104) eGPU over Chestnut USB3.

## Current 660 Ti state (2026-09-04)

**Add and multiply now pass on the physical `10de:1183` GTX 660 Ti over
Chestnut USB3.** Both verification runs started from an ATX/PCIe power cycle,
used the native `[2,2,2,1]` seven-TPC topology, retired one GPFIFO entry with
completion semaphore 2, and reported no GR/GPC/SKED trap:

- add, `N=8`: 8/8 results matched
- mul, `N=256`: 256/256 results matched

The final initialization defect was a paired divergence from Nouveau. The
Python `nvkm_mask()` returned the post-write value instead of the original
register value, so `nvkm_fuse_read_31c()` restored its temporary fuse-access
gates incorrectly. In addition, `gk104_pmu_pgob()` ignored Nouveau's
`0x02141c[0]` board-fuse gate. This 660 Ti reads zero there, as does the
same-board mmiotrace, so Nouveau skips PGOB. Matching both behaviors produced
the first retained numerical passes on device `10de:1183`.

## Investigation record (before the fix)

A history audit found no retained successful GTX 660 Ti run. Commit `bc12d25`
claims “660 add works” in its subject, but its only checked-in S10 log identifies
PCI device `10de:1184`, topology `[2,2,2,2]`, and is therefore the GTX 770. The
newly added `progress_660ti.md` at that revision also says the live device was a
GTX 770 over USB4. No reachable log combines `10de:1183`, topology `[2,2,2,1]`,
and `mismatches=0`, so `bc12d25` is an unverified 660-labeled A/B candidate,
not a working oracle.

The first exact direct-USB rerun of detached `bc12d25` on a freshly replugged
660 Ti was manually interrupted after 90 seconds. It had completed S1-S8
(`S6=65.0 s`) and was still reading the entire `0x29b00`-byte runtime context
one dword per USB request before exposing `GP_PUT`; therefore it produced no
compute result and is inconclusive, rather than a demonstrated failure. It also
reported the power-gated sentinel `0xbadf1200` as an S2-successful topology,
confirming that the historical posted-state check was unsound. The current
stack rejects that sentinel. Live `KEPLER_TRACE=1` output now bypasses the
health-check buffer so long S6/S9 operations stream instead of appearing hung.
Any second historical run must allow the full verification to finish and set
`KEPLER_AUTO_WARM_CONTINUE=0` so one requested run cannot silently become two.

The complete 175,274,772-byte `mmiotrace_660ti.txt` has now been parsed: all
4,372,128 recorded events are present (`97,450` reads, `4,274,552` writes,
63 maps, and 63 unmaps; no dropped entries). The trace is not a successful
compute reference. Its accompanying Nouveau/OpenCL run failed while creating
the M2MF PGRAPH context with a HUB/FE PDE fault at VA `0x34d000`, killed
channel 2, and never published a compute `GP_PUT`. It is nevertheless an
authoritative reference through cold init, Falcon load, golden-context build,
RAMFC construction, and the attempted runtime-context transition.

The comparison rules out the large bring-up components:

- The unmodified stock PMU/FECS/GPCCS instruction images reconstruct the trace
  at every loaded dword (768/768, 768/768, and 448/448 respectively); PMU data
  is 896/896. The direct-USB live images intentionally differ from stock at 59
  PMU and 48 FECS dwords (wait-branch and autonomous BAR1 helper patches in
  stock-zero padding); live GPCCS remains 448/448 exact. Those USB patches were
  already present in commit `bc12d25`, so they predate the later WIP revision;
  history does not independently prove them on a successful 660 Ti run.
- Both FECS DMEM sessions match all 243 reconstructed dwords, including all
  50 hub csdata words. Both GPCCS sessions match all 79 dwords, including GPC,
  TPC, PPC csdata, and pointer words.
- The 140-write GK104 PGRAPH pack is exact. The generated golden-context
  sequence has 3,033 aligned exact writes; the trace alone has an empty ICMD
  enable/disable pair and this stack alone has a trailing `0x419cb8=0`.
- The 24-entry runtime MMIO list, native seven-TPC attribute values, GPC MMU
  setup, RAMFC field meanings, FECS mailbox1 size `0x29b00`, and context-pointer
  lifecycle all match.
- Replaying the onboard ROM with trace RAMCFG group 6 matches 1,635 of 1,636
  cold writes; the sole intentional difference is compression-tag backing at
  `0x17e8d4` (Nouveau allocated tag RAM, this uncompressed stack uses zero).

The checked-in `runtime/golden_gk104_cold_slice.json` and its 23-checkpoint
`--mmiotrace-selftest` are the earlier Palit/GTX 770 cold baseline. They remain
useful regression coverage, but a pass must not be interpreted as validation
against this newly supplied 660 Ti capture; the exact 660 comparison above was
performed independently against all relevant records in the full trace.

The full trace exposed three incorrect assumptions in the current 660 path.
Healthy runlist commits now issue only `0x2270/0x2274`, rather than preempting
and unblocking runlist 0 first. The failed Nouveau capture reuses its admitted
list after STOP/KICK and runtime-context-pointer replacement, but it never
reaches compute. The retained S10-passing GTX 770 log and the unverified
660-labeled `bc12d25` candidate both perform a second runtime commit, while the
latest no-recommit 660 run consumed the IB without producing shader stores. The
sanitized `0x2270/0x2274` recommit remains the next direct-USB experiment;
`KEPLER_RECOMMIT_RUNLIST=0` preserves the failed trace's ordering for A/B.
The offline runtime-context header was also corrected to the live/trace format:
pair count at dword 0 and MMIO-list VA shifted by eight at dword 1.

The add/mul cubins contain only CB0 loads and global LD/ST, but commit
`543b4f3` replaced the 660-specific 47-word TIC/TSC+CB7 launch state with the
770's 39-word stream and changed the tree's stated status from working to not
working. The earlier 660-labeled candidate also used a privileged FECS guard leaf,
pre-launch floorsweep re-arm, no LTC entries in the slow USB FECS context
walk, and FE-power reinforcement during submit. Those defaults are restored;
the 39-word stream and trace-exact LTC list remain explicit A/B controls.
An offline byte comparison against `bc12d25` now also confirms that the current
add cubin, 512-byte CUDA parameter buffer, 256-byte CWD, and complete 47-word
method stream are exact; the method stream includes the same no-WFI semaphore.
The cleaned-up direct 2-MiB TLS alignment had nevertheless changed the actual
default address map. The historical candidate's allocator sequence is restored
and locked offline: TLS/TXC/push/GR-context are again
`0x280000/0x360000/0x160000/0x400000`; the native segmented attribute buffer
retains its trace-derived contents at the same `0x500000` VA.
BAR1 L2 warming is no longer followed by a default invalidate. Since the
no-WFI completion value only proves PBDMA progress, the path preserves the
historical candidate's 500 ms output-settle budget as a local sleep, then takes
one result snapshot. Small results follow its BAR0
PRAMIN path; `KEPLER_MIRROR_COPY=bar1` and results above the configured PRAMIN
cap retain the single BAR1 read. It performs no speculative output polling or
post-snapshot MMIO. The USB transport's PMU nowait patch and long-context-build
power keepalive remain in place because a fast Linux PCIe trace cannot validate
those transport-specific requirements.

One board-state discrepancy was pending live verification: this trace read
`0x101000=0x80404c9a` (RAMCFG group 6), whereas the latest direct-Mac run read
`0x80405096` (group 5). Group 5 and group 6 differ at 196 RAM-init writes.
The live path now samples and caches `0x101000` immediately after BAR0 becomes
readable and before BIT-I POST can rewrite strap state, matching Nouveau's
ordering. A valid sample selects its own group for both POST and RAM training;
zero/`bad*` samples retain group 5 as the direct-Mac fallback, and an explicit
`KEPLER_RAMCFG_STRAP` still takes precedence for controlled A/B tests. The
successful Orange Pi runs sampled `0x80404c9a`, selected group 6, and therefore
verified this path physically.

The lossless Nouveau capture `mmiotrace_660ti.txt.gz` records native attribute
state `0x405830=0x02180648`, `0x4064c4=0x0192ffff`, and the four PPC pairs. The
old raw path instead shrank the seven-TPC geometry to fit one 512-KiB bank.
`add_660ti.py` now keeps the native 0x9c000-byte logical attribute buffer and
maps it over `0x80000 + 0x1c000` bit-19-safe physical segments. All golden,
critical, and live PTE construction uses the same segmented-offset helper.

Seven MPs require `0xe0000` TLS in total, still exactly `0x20000` per MP. TLS
is kept above a VA floor of `0x200000`; preserving the historical allocator
sequence places the default workload at `0x280000` and leaves the separately
backed FECS guard leaf at VA `0x100000`. Compilation, add/mul offline selftests, add/mul software runs,
the trace-format test, exact captured attribute-register assertions,
segmented-PTE tests, and the final BAR1-read transport guard pass offline.
Physical S10 result validation has now passed for add and multiply as recorded
above.

## GTX 770 baseline (2026-07-20)

**add and mul work for N = 1 to 1048576 with 0 mismatches.** N=1048576 (64
channel windows) previously crashed macOS on the second run due to a sysmem
mmap leak. Root causes found and fixed:
- **H25**: sysmem mmap leak — each windowed NVDevice reopen leaked 256 MB of
  mmap'd VM; 64 windows accumulated 16 GB and exhausted the process VM limit.
  Fixed: `fini()` now munmaps + closes all sysmem allocations.
- **H27**: atexit handler accumulation — each window registered
  `_quiesce_channel` without unregistering; 64 stale closures would hang at
  exit. Fixed: `close()` now unregisters.
- **H28**: atexit.unregister skipped if teardown raises — `teardown()` and
  `atexit.unregister()` were in the same try block. Fixed: teardown now in its
  own try/except, unregister always runs.
- **H29**: temp directory leak in `compile_kepler_cubin_docker()` —
  `tempfile.mkdtemp()` was never cleaned up. Fixed: function now reads cubin
  into memory and `shutil.rmtree`s the temp dir in a `finally` block.
- **H30**: socket not closed in error path in `_tinygpu_sock_reachable()` —
  if `s.connect()` raised a non-OSError exception, the socket leaked. Fixed:
  socket now created outside try, closed in `finally`.
- **examples_kepler_pcie/mul.py indentation bug** (pre-existing, commit
  c6ea9239): lines 50-52 were at module level instead of inside `main()`,
  causing `shared.main()` to run at import time. Fixed: added 2-space indent.
**GPU is currently online** — replugged and TinyGPU server restarted. All
fixes verified on live hardware:
- N=32768 (2 windows): PASS, 0 mismatches
- N=131072 (8 windows): PASS, 0 mismatches, no crash, no exit hang
- N=524288 (32 windows): PASS, 0 mismatches, no crash, no exit hang
- N=1048576 (64 windows): PASS, 0 mismatches, no crash, no exit hang
  (this is the scale that previously crashed macOS via the H25 sysmem leak)
- mul N=256: PASS, 0 mismatches
- mul N=4096: PASS, 0 mismatches
- mul N=131072 (8 windows): PASS, 0 mismatches, no crash, no exit hang
- mul N=524288 (32 windows): PASS, 0 mismatches, no crash, no exit hang
- mul N=1048576 (64 windows): PASS, 0 mismatches, no crash, no exit hang

### Test matrix (all 0 mismatches, seed=42)

| N | add | mul | Path | Windows |
|---|---|---|---|---|
| 1 | OK | — | single-CTA | 1 |
| 8 | OK | OK | single-CTA | 1 |
| 256 | OK | OK | single-CTA | 1 |
| 1024 | OK | OK | single-CTA | 1 |
| 4096 | OK | OK | multi-CTA | 1 |
| 8192 | OK | OK | multi-CTA | 1 |
| 16384 | OK | OK | multi-CTA | 1 |
| 32768 | OK | OK | windowed | 2 |
| 65536 | OK | OK | windowed | 4 |
| 131072 | OK | OK | windowed | 8 |
| 262144 | OK | — | windowed | 16 |
| 524288 | OK | OK | windowed | 32 |
| 1048576 | OK | OK | windowed | 64 |

N=1048576 now passes reliably after H25/H27/H28 fixes (previously crashed
macOS on the second run due to sysmem mmap leak).

Offline `--middle-selftest`: green.

## Root cause: L2 cache coherency on un-POSTed card

The GK104 L2 cache is 128 KiB. On this un-POSTed card (no VBIOS full POST),
the LTC invalidate register (0x070004) does NOT evict stale L2 lines — it
accepts the write and clears the busy bit, but stale lines remain. The SM
reads stale L2 data instead of fetching fresh VRAM.

### Fix: L2 warming via BAR1 bulk reads

Before each kernel launch, the code reads all read-only compute mirrors (a,
b, code, cbuf, cwd) through BAR1. BAR1 reads go through L2, replacing stale
lines with correct VRAM data. This "warms" L2 so the SM hits correct data.

- **Cap**: 128 KiB per mirror (= L2 size). Warming more is pointless.
- **Windowing**: a+b = N*8 bytes must fit in L2 (N <= 16384). For N > 16384,
  the work is split into channel windows of 16384 elements each. Each window
  reopens the device, re-initializes context, re-warms L2, and runs its slice.
- **BAR1 vs PRAMIN**: BAR1 bulk read (1 RPC) is ~16000x faster than per-dword
  PRAMIN reads (16384 RPCs) and equally effective for L2 warming. BAR1 bulk
  reads of <=128 KiB are reliable; larger transfers corrupt (bit15 flips).

### Key code locations

- `_gk104_ltc_invalidate` (line ~9372): flush + invalidate, with `flush`
  parameter to control writeback.
- Single LTC flush before all mirror writes (line ~11879): prevents L2 set
  aliasing where per-mirror flush writes back stale lines over earlier mirrors.
- BAR1 bulk read L2 warming (line ~13002): `KEPLER_L2_WARM_VIA_BAR1=1`
  (default ON).
- L2-constrained windowing (line ~13712): `KEPLER_L2_MAX_ELEMENTS=16384`
  controls multi-CTA vs windowed threshold.
- Pre-GP_PUT final LTC invalidate is skipped by default so BAR1-warmed lines
  survive to execution; `KEPLER_SKIP_FINAL_LTC_INV=0` restores the experiment.

## Hypothesis board

### Proven (fixes in production code)

| ID | Hypothesis | Evidence |
|----|-----------|----------|
| H1 | FECS hub MMIO dump while stuck | Diagnostic in SET_OBJECT hang path |
| H2 | SET_OBJECT hang clears GPC1/2 TPC_NR | Floorsweep re-arm logic |
| H9 | FECS image size is ready-time mailbox1 | Removed unsupported host command 0x10; trace and 770 read 0x409804 directly |
| H10 | SET_OBJECT hang: FECS_MMIO_CTRL WRITE to LTC | LTC mmio-list omitted by default |
| H14 | train-status strict must abort | Raises on failure |
| H16 | eng-ctx hang: GPC3 PPC+0xe4 shrunk | PPC mmio-list omitted |
| H17 | eng-ctx leaves FE_PWR=0 | Re-assert FORCE_ON |
| H18 | eng-ctx auto-load races FE power-gate | FE pwr force-on in poll loop |
| H19 | per-GPC GPCCS falcon state | Per-GPC diagnostic dump |
| H21 | BAR1 golden→runtime copy false-passes | PRAMIN copy used instead |
| H22 | Bit-flip drift in mirrors | Settle rewrites via PRAMIN |
| H23a | Per-mirror LTC flush causes set aliasing | 20484 mismatches → 0 with single flush |
| H23b | BAR1 bulk reads corrupt >64 KiB | 21889 mismatches in 256 KiB BAR1 read |
| H23c | LTC invalidate ineffective post-bit0 | 278 mismatches without warming → 0 with |
| H23d | L2 warming works for <=128 KiB | N=16384 passes; N=32768 fails without windowing |
| H23e | Channel windowing scales to large N | N=524288 (32 windows) passes |
| H23f | Final LTC invalidate drops warmed lines | Skipping it + no warming = 2736 mismatches |
| H23g | BAR1 bulk read warms L2 (1 RPC) | N=16384 passes with BAR1 warm, 0 mismatches |
| H25 | Sysmem mmap leak crashes macOS at 64 windows | 64×256MB=16GB leaked VM; fini() now munmaps+close |
| H25-twin | LinuxPCIDevice mlock without munlock (twin of H25) | fini() now munlocks+close sysmem_buf |
| H25-port | H25/H25-twin/FileIOInterface fixes ported to add.py (GTX 770) | Linux port (examples_kepler_pcie) now safe for windowed runs |
| H26 | LTC invalidate (0x70004) is not desktop coherency mechanism | Nouveau only calls it on GK20a; desktop uses 0x070000 BAR flush |
| H27 | atexit handler accumulation in windowed path | Each window registered _quiesce_channel without unregistering; close() now unregisters |
| H28 | atexit.unregister skipped if teardown raises | teardown("device close") and atexit.unregister were in same try block; if teardown raised, unregister was skipped leaving stale handler. Fixed: teardown now in its own try/except, unregister always runs |
| H29 | temp directory leak in compile_kepler_cubin_docker() | tempfile.mkdtemp() was never cleaned up. Fixed: function reads cubin into memory and shutil.rmtree the temp dir in a finally block |
| H30 | socket not closed in error path in _tinygpu_sock_reachable() | if s.connect() raised a non-OSError exception, the socket leaked. Fixed: socket created outside try, closed in finally |
| H31 | socket and file descriptor leaks in benchmark _ensure_sock() | sockets created without finally blocks, log file opened without with. Fixed: all sockets have finally: s.close(), log file uses with statement |
| H31-pcie | examples_kepler_pcie/mul.py indentation bug (commit c6ea9239) | lines 50-52 were at module level instead of inside main(), causing shared.main() to run at import time. Fixed: added 2-space indent |
| H32 | KEPLER_SKIP_LTC=1 optimization for LTC invalidate | Safe up to N=524288 (32 windows) but hangs at N=1048576 (64 windows). LTC invalidate prevents cache state accumulation across many windows. Env var added as opt-in for smaller workloads. |

### Disproven

| ID | Hypothesis | Evidence |
|----|-----------|----------|
| H20 | Channel preempt + ctxctl IDLE clears sticky GPC | Live shot wedged FECS; disabled |

### Open

| ID | Hypothesis | Status |
|----|-----------|--------|
| ~~OQ1~~ | ~~Why is LTC invalidate ineffective?~~ | **Resolved (H26):** not the desktop coherency mechanism |
| OQ2 | BAR1 bulk read corruption threshold | Known >128 KiB; exact boundary unmapped |
| OQ3 | Can LTC invalidate calls be removed for performance? | **Partially resolved:** KEPLER_SKIP_LTC=1 skips all hot-path LTC invalidate calls. Safe up to N=524288 (32 windows) but hangs at N=1048576 (64 windows). LTC invalidate is needed at higher window counts to prevent cache state accumulation. |
| OQ4 | atexit handler accumulation (H27/H28) | Fixed: close() now unregisters (H27), and unregister always runs even if teardown raises (H28). |

## Roadmap

### OQ1: Why is LTC invalidate ineffective? — ROOT CAUSE FOUND (H26)

**Root cause: LTC invalidate (0x70004) is not the desktop coherency mechanism.**

Nouveau source analysis (`ref/linux/drivers/gpu/drm/nouveau/nvkm/subdev/ltc/`):
- `gf100_ltc_invalidate()` writes 0x70004 and waits for bits[1:0] to clear.
- `gf100_ltc_flush()` writes 0x70010 and waits for bits[1:0] to clear.
- **On desktop GK104, `nvkm_ltc_invalidate()` is never called.** It's only
  used from `gk20a.c` (Tegra K1 embedded). Desktop coherency is maintained
  through `g84_bar_flush()` which writes **0x070000** (BAR flush register),
  a completely different mechanism.

This means 0x70004 may only affect compression tag state, not data cache
lines. On our un-POSTed card over TinyGPU:
- 0x070000 (BAR flush) times out after 43ms (TinyGPU transport limitation)
- 0x70004 (LTC invalidate) accepts the write and clears busy but doesn't
  evict stale data lines (it was never the right mechanism for this)
- 0x70010 (LTC flush) similarly doesn't flush data lines to VRAM

**Conclusion:** Our L2 warming workaround (BAR1 bulk reads to replace stale
L2 lines with correct VRAM data) is the correct approach. Neither the BAR
flush nor the LTC invalidate/flush registers can solve this on our platform.
The windowing strategy (limiting a+b to 128 KiB = L2 size per window) is
also correct — it ensures all compute data fits in L2 after warming.

**Nouveau reference files:**
- `ref/linux/drivers/gpu/drm/nouveau/nvkm/subdev/ltc/gf100.c:126-149`
  (invalidate + flush implementations)
- `ref/linux/drivers/gpu/drm/nouveau/nvkm/subdev/ltc/gk104.c:38-51`
  (gk104 uses gf100 invalidate/flush, but they're never called on desktop)
- `ref/linux/drivers/gpu/drm/nouveau/nvkm/subdev/bar/g84.c:28-40`
  (g84_bar_flush — the actual desktop coherency mechanism, writes 0x070000)
- `ref/linux/drivers/gpu/drm/nouveau/nvkm/subdev/instmem/gk20a.c`
  (only caller of nvkm_ltc_invalidate — Tegra only)

### OQ2: BAR1 bulk read corruption threshold

Known: <=128 KiB reliable, 256 KiB corrupts (bit15 flips, same pattern as L2
staleness). The corruption likely comes from L2 set aliasing within the bulk
read itself.

To prove/disprove:
1. Binary search: test 128, 160, 192, 224, 256 KiB.
2. Check if corruption starts at 128 KiB + 1 byte or at 256 KiB.

### OQ5: N=1048576 macOS crash — ROOT CAUSE FOUND (H25)

**Root cause: sysmem mmap leak.** Each `NVDevice` reopen calls
`alloc_sysmem(256 MB)` which mmaps a 256 MB GPU-visible host buffer via a
file descriptor received from TinyGPU (`MAP_SYSMEM_FD` + recvmsg SCM_RIGHTS).
The old `fini()` only closed the Unix socket — it never munmapped the sysmem
region or closed the received fd. After 64 windows: 64 × 256 MB = **16 GB
of leaked mmap'd virtual memory** + 64 leaked file descriptors. This
exhausted the process VM limit and crashed macOS.

**Fix (H25):** `APLRemotePCIDevice` now tracks all sysmem allocations in
`self._sysmem_maps = [(addr, nbytes, fio), ...]`. `fini()` munmaps each
region and closes each fd (clearing `fio.fd = None` to prevent
`FileIOInterface.__del__` from double-closing the same fd number after it
may have been reused by another allocation). This should allow N=1048576
(64 windows) to run without crashing.

**Twin fix (H25-twin):** `LinuxPCIDevice.alloc_sysmem` calls `libc.mlock()`
to pin DMA pages but `fini()` never called `munlock()` or explicitly closed
the Python `mmap.mmap` object. Fixed: `fini()` now munlocks + closes
`self._sysmem_buf`. This matters for the Linux port if it ever does
windowed runs.

**Safety guards also added:**
- `KEPLER_MAX_WINDOWS=32` (default): rejects N requiring >32 windows.
- `KEPLER_INTER_WINDOW_SLEEP=0.5s` (default): sleep between window reopen
  cycles to let USB4 transport recover.

**Other audited paths:**
- `_kepler_emergency_teardown` / atexit handlers: each closure has its own
  `_teardown_done` flag; atexit calls are no-ops if device already closed.
  **H27 fix:** `NVDevice.close()` now calls `atexit.unregister(teardown)` after
  running the teardown, so accumulated windowed closures don't all fire at
  process exit. Without this, 64 windows would register 64 atexit handlers
  that could each wait 2.5s on thread join if close() wasn't called.
- FECS keepalive thread: properly joined in `_quiesce_channel` via
  `NVDevice.close()`.
- BAR mappings (LinuxPCIDevice): properly munmap'd + fd closed in `fini()`.
- trace_fd: properly closed in `fini()`.
- Golden context `_ka_thread2`: initialized to None, never assigned — dead
  code, no actual leak. Cleanup guard `if _ka_thread2 is not None` prevents
  any TypeError.
- `open()` without `with` (7 sites in PMU reload/firmware read): NOT real
  leaks in CPython — file objects are immediately collected by reference
  counting after `open(p, "rb").read()`. Style issue only.
- Thread lifecycle: only one thread (FECS keepalive, daemon=True); properly
  joined in `_quiesce_channel` via `NVDevice.close()`. H27 fix also
  unregisters the atexit handler.
- `LinuxPCIDevice.alloc_sysmem` single-buffer: only called once per device
  init (line 2698); windowed path creates fresh devices, so no leak.

**Status: GPU OFFLINE (2026-07-20 10:18 HKT).** The crash left the GPU in a
dirty state (BAR1 returns `0xbad0ac33` power-gated sentinel, DMEM probes
fail with `0xbadf1200`). TinyGPU server was restarted but GPU RPCs still
fail with "unknown error" — the GPU itself needs a physical eGPU power
cycle (not just USB replug or server restart).

### Linux PCIe path status

The Linux path (`examples_kepler_pcie/add.py` re-exporting
`examples_kepler/add_770.py`) is **working**: `hardware_demo=ok N=256` with
`mismatches=0/256` on the GTX 770 at 09:00.0 (2026-07-15). VBIOS devinit
executes, GPC PLL locks, FECS posts ready, ctx_chan works, golden context
saves, and the full add kernel runs with correct results. The blocker was
TEMP size (needed 0x100000 total for GK104's 8 MPs × 64 warps), now fixed.

The H25/H27 fixes ported to `add.py` in this session will benefit the Linux
path when it scales beyond N=256 to windowed runs (N>16384). The
`KEPLER_MAX_WINDOWS` and `KEPLER_INTER_WINDOW_SLEEP` safety guards are in
place, and the atexit handler cleanup (H27) will prevent stale closure
accumulation.

### H25-port: fixes backported to add.py (GTX 770 / Linux PCIe path)

The H25 fixes were originally applied only to `add_660ti.py` (GTX 660 Ti /
macOS TinyGPU). The Linux port (`examples_kepler_pcie/add.py`) re-exports
`examples_kepler/add_770.py` (GTX 770), which had the same three bugs:

1. **APLRemotePCIDevice sysmem mmap leak**: `alloc_sysmem` created a
   `FileIOInterface(fd=fd)` that was immediately discarded — the fd got
   closed by `__del__` but the mmap was never munmapped. Now tracks
   `_sysmem_maps` and munmaps + closes in `fini()`.
2. **LinuxPCIDevice mlock leak**: `alloc_sysmem` mlocked `_sysmem_buf` but
   `fini()` never munlocked or closed it. Now munlocks + closes in `fini()`.
3. **FileIOInterface.__del__ double-close**: `hasattr(self, 'fd')` didn't
   handle `fd=None` after explicit close. Now uses `getattr` + `is not None`.

Safety guards also ported:
- `KEPLER_MAX_WINDOWS=32` (default): rejects N requiring >32 windows.
- `KEPLER_INTER_WINDOW_SLEEP=0.5s` (default): sleep between window reopen
  cycles to let transport recover.
- H27: `NVDevice.close()` now calls `atexit.unregister(teardown)` after
  running the teardown, preventing accumulated windowed closures from
  firing at process exit.

Verified offline: `add.py` compiles, `--middle-selftest` passes
(`kepler_selftest=ok`).
