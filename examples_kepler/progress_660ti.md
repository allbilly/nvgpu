# Progress — GTX 770 / GK104 add_660ti.py bring-up

Live status of getting a Kepler `sm_30` `out[i]=a[i]+b[i]` (and `mul`) kernel
running on the GTX 770 (GK104) eGPU over USB4 via TinyGPU.

## Current state (2026-07-20)

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
- Pre-GP_PUT final LTC invalidate (line ~13068): `KEPLER_SKIP_FINAL_LTC_INV=1`
  to skip (for testing).

## Hypothesis board

### Proven (fixes in production code)

| ID | Hypothesis | Evidence |
|----|-----------|----------|
| H1 | FECS hub MMIO dump while stuck | Diagnostic in SET_OBJECT hang path |
| H2 | SET_OBJECT hang clears GPC1/2 TPC_NR | Floorsweep re-arm logic |
| H9 | FECS discover_image_size via method 0x10 | Mailbox 0x409804 can be stale |
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
`examples_kepler/add.py`) is **working**: `hardware_demo=ok N=256` with
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
`examples_kepler/add.py` (GTX 770), which had the same three bugs:

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
