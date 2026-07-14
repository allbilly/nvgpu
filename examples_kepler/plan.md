# GK104 `add.py` Crash Debug-and-Fix Plan

## 1. Current diagnosis

The macOS crashes are not ordinary Python failures or GPU MMU faults. The panic bit `0x200000` corresponds to Apple PCIe port interrupt bit 21, `PORT_INT_CPL_ABORT`: a PCIe transaction received a Completion Abort.

The recent panic symbolication repeatedly caught Python inside `APLRemotePCIDevice._rpc()`, either sending a TinyGPU request or waiting for its response. The latest investigations progressively removed:

* Function reset and post-reset config access
* PCI decode disable
* GPC PLL and PGOB shutdown
* Repeated PCI shutdown layers
* BAR teardown MMIO
* Post-quiesce output access
* TinyGPU server termination and relaunch

The current intended lifecycle is now: reuse `/tmp/tinygpu.sock`, leave the shared TinyGPU server running, stop the host helper, and close only the client socket.

There is, however, a critical source/documentation discrepancy to resolve before another live run. `progress.md` says normal execution no longer performs post-launch trap clearing, FE power changes, or other diagnostic BAR0 operations before reading output. The current raw `add.py` still appears to contain trap W1C writes, `FE_PWR=AUTO`, and `0x260=1` before the BAR1 output read.

That discrepancy is the first code issue to fix.

---

## 2. Mandatory safety rules

Until the crash is isolated:

1. Perform **one live invocation per physical enclosure power-cycle**.
2. Never run `--probe` immediately before or after the full test.
3. Do not run chained shell commands that start another hardware process.
4. Keep `DEBUG=0`; the debug path performs many additional BAR0 reads.
5. Never issue `CFG_READ`, `CFG_WRITE`, `RESET`, BAR decode disable, PLL shutdown, PGOB reversal, or engine-reset operations during close.
6. Reuse the existing shared TinyGPU server and `/tmp/tinygpu.sock`.
7. Do not run a second test merely because the first process exited successfully. Previous panics were delayed.
8. Save the exact Git commit, worktree diff, program log, RPC trace, and panic timestamp for every test.

---

## 3. Phase 0 — Establish a reproducible offline baseline

Before changing or touching hardware, record the exact source state:

```bash
git rev-parse HEAD
git status --short
git diff -- examples_kepler/add.py examples_kepler/progress.md
shasum -a 256 examples_kepler/add.py
```

Run all offline gates:

```bash
python3 -m py_compile examples_kepler/add.py
python3 examples_kepler/add.py --middle-selftest
NV_BACKEND=software python3 examples_kepler/add.py
git diff --check
```

Also inspect TinyGPU lifecycle state:

```bash
pgrep -alf TinyGPU
ls -l /tmp/tinygpu.sock
lsof -U 2>/dev/null | grep tinygpu.sock
```

### Required result

* All offline tests pass.
* Exactly one intended TinyGPU server owns the stable socket.
* No unique per-run socket paths remain.
* No uncommitted lifecycle experiment is hidden in the worktree.

Do not proceed if the local `add.py` differs from the version described by `progress.md` without documenting the difference.

---

## 4. Phase 1 — Reconcile the normal post-completion path

Create one deliberately minimal success path.

After the completion semaphore reaches its expected value, the only allowed sequence should be:

```text
completion semaphore observed
        │
        ├── one bulk BAR1 output read
        ├── validate output entirely on CPU
        ├── stop/join host helper
        └── close client socket
```

The normal path must not perform any of the following after semaphore completion:

* `snapshot_gr_traps()`
* Trap W1C clears
* PGRAPH interrupt clears
* FECS or GPCCS register reads
* GPC/TPC trap reads
* LTC flush or invalidate
* `FE_PWR` mode change
* `nvkm_mc_unk260()` or `0x260` write
* PFIFO channel stop or unbind
* Empty-runlist commit
* PCI configuration access
* Function reset
* TinyGPU server termination

Move all state-mutating diagnostics into a dedicated, explicitly dangerous diagnostic mode. Do not merely hide them behind `DEBUG`; ordinary debugging should remain safe.

Recommended structure:

```python
result = submit_launch(...)

# No BAR0 operations here.
out = read_output_bar1_once(...)

validate_output(out)

# No endpoint RPC from this point onward.
transport.freeze("output-read-complete")
```

Add an invariant to `_rpc()`:

```python
if self._endpoint_frozen:
    raise RuntimeError(
        f"endpoint RPC after freeze: cmd={cmd} bar={bar} args={args}"
    )
```

This converts accidental post-output hardware access into a Python failure instead of another PCIe transaction.

---

## 5. Phase 2 — Add a persistent RPC flight recorder

The next panic must identify the exact request, not merely `_socket.recv_into`.

Instrument `_rpc()` with a monotonically increasing request number and two records:

```text
BEGIN seq=184 phase=output-read cmd=MMIO_READ bar=1 offset=... size=1024
END   seq=184 status=ok bytes=1024 duration_us=...
```

Record at least:

* Sequence number
* Monotonic timestamp
* Current high-level phase
* Thread ID
* Command name and numeric ID
* BAR number
* Offset
* Size
* Config-space offset where applicable
* Payload length and hash for writes
* Start and completion status
* Exception or timeout
* Duration

Write `BEGIN` **before** `sendall()`. Write `END` only after the complete response or write transmission has finished. The final unmatched `BEGIN` identifies the operation that stalled or received Completion Abort.

Use an already-open unbuffered file descriptor:

```python
os.write(trace_fd, record.encode())
```

Avoid ordinary buffered logging. A kernel panic may prevent buffers from being flushed.

Add matching server-side instrumentation to TinyGPU around each DriverKit operation. Client logging tells which RPC was sent; server logging tells which DriverKit request began but did not return.

### High-level phase names

Use explicit phases such as:

```text
connect
map-bars
vbios-devinit
firmware-load
golden-context
channel-build
runlist-submit
semaphore-poll
output-read
host-helper-stop
client-close
```

---

## 6. Phase 3 — Remove concurrent socket ownership

The FECS keepalive helper currently shares the socket with the main thread. A lock prevents byte interleaving, but concurrency still complicates crash attribution and lifecycle ordering.

For crash isolation, introduce a strict single-thread transport mode:

* Do not start the FECS keepalive thread.
* Perform any necessary FE power keepalive from the main semaphore polling loop.
* Stop issuing keepalive operations once the semaphore completes.
* Ensure only the main thread calls `_rpc()`.

Longer-term, either retain single-threaded transport or give one dedicated transport thread exclusive socket ownership and send it requests through a queue. Multiple threads should not independently perform endpoint operations.

Add an assertion:

```python
assert threading.get_ident() == self._rpc_owner_thread
```

### Required result

Every live run has exactly one RPC-producing thread. The final trace ordering is therefore unambiguous.

---

## 7. Phase 4 — Add a no-RPC hold-open experiment

The current lifecycle mixes three possible failure sources:

1. Active GK104 hardware state
2. Release of GPU-visible system mappings
3. Client/server close lifecycle

Separate them with a hold-open mode.

After the output read—or after the selected test stage—keep all of these alive:

* Python process
* TinyGPU socket
* GPU-visible mmap
* Page tables
* RAMIN
* USERD
* GPFIFO
* Output allocation

During the hold period, issue **zero endpoint RPCs**.

Proposed option:

```text
KEPLER_HOLD_OPEN_SECONDS=120
```

Expected flow:

```text
last intended RPC completes
RPC transport frozen
sleep 120 seconds with mappings alive
close client socket
continue observing
```

Interpretation:

| Result                                | Likely boundary                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| Panic during the zero-RPC hold        | Autonomous GPU DMA, active engine state, or DriverKit activity |
| Stable hold, panic when client closes | Client detach or DriverKit lifecycle                           |
| Panic during BAR1 output read         | BAR1 address, size, mapping, or endpoint decode                |
| Stable hold and stable close          | Prior extra teardown/diagnostic RPC was the cause              |
| Stable system but wrong output        | Host crash fixed; continue with compute debugging              |

This experiment is substantially more informative than repeatedly altering teardown.

---

## 8. Phase 5 — Controlled live-test ladder

Each stage gets its own physical enclosure power-cycle and one process invocation.

### Stage A — Transport and one harmless read

Operations:

```text
connect shared socket
MAP_BAR
read PMC_BOOT_0
client socket close
```

Purpose: establish whether the shared transport lifecycle alone is stable.

### Stage B — Hardware initialization only

Stop after:

```text
VBIOS devinit
GPC PLL setup
PMU/PGOB setup
FECS and GPCCS ready
golden context generation
```

Enter the no-RPC hold before close. No FIFO channel or runlist.

### Stage C — Semaphore-only submission

Use the already proven semaphore command path:

```text
channel bind
runlist submit
PBDMA consumes GPFIFO
WFI semaphore reaches done value
zero-RPC hold
client-only close
```

Do not read traps or output.

### Stage D — Full launch without output read

Launch the QMD and wait for the serialized completion semaphore, but freeze endpoint access before BAR1 output read.

Interpretation:

* Crash here means the compute launch or resulting autonomous GPU activity is enough.
* Stability means the output-read or later lifecycle remains suspect.

### Stage E — Full launch plus exactly one BAR1 output read

Perform one bulk read of the output buffer. No preceding BAR0 diagnostics and no subsequent endpoint access.

This is the decisive normal-path test.

---

## 9. Phase 6 — Make every launch mode deterministic

The present code contains multiple experimental submission routes, including normal GPFIFO, BYPASS submission, and direct DISPATCH injection when `SET_OBJECT` appears not to bind. Automatic fallback makes a run impossible to interpret because one invocation can exercise several distinct hardware paths.

Replace automatic fallback with explicit modes:

```text
KEPLER_SUBMIT_MODE=gpfifo
KEPLER_SUBMIT_MODE=bypass
KEPLER_SUBMIT_MODE=dispatch
```

For the normal validation run:

```text
KEPLER_SUBMIT_MODE=gpfifo
```

If `SET_OBJECT` fails, stop and report the failure. Do not silently disable/re-enable PGRAPH pull and inject methods through another interface.

Likewise, separate:

```text
KEPLER_TEST_STAGE=sem
KEPLER_TEST_STAGE=set-object
KEPLER_TEST_STAGE=constant-store
KEPLER_TEST_STAGE=one-element-add
KEPLER_TEST_STAGE=full-add
```

Each stage should have a known RPC and method budget.

---

## 10. Phase 7 — Fix compute correctness only after host stability

The completion semaphore reaching `2` proves the FIFO command stream reached the fence, but it does not by itself prove that every SM executed the expected store correctly. Earlier full launches completed their semaphore while leaving all output values zero.

Use this kernel ladder:

### Test 1 — Constant store

One thread writes a fixed bit pattern:

```c
out[0] = 0x3f800000;  // 1.0f
```

This removes input loads, indexing, and arithmetic.

### Test 2 — Single-element add

```c
out[0] = a[0] + b[0];
```

One block, one thread.

### Test 3 — Warp-sized add

One block, 32 threads.

### Test 4 — Full 256-element add

Only attempt after the first three produce correct output.

For each test, verify offline before hardware:

* Genuine `sm_30` cubin
* Entry-point name
* Register count
* Code offset and size
* Constant-buffer layout
* QMD program address
* QMD grid/block dimensions
* QMD register count
* Input and output virtual addresses
* Corresponding live VRAM PTEs
* Output buffer initialized to a NaN or fixed sentinel
* Release membar and WFI ordering

If the constant-store kernel fails but no trap is visible, focus on:

* `SET_OBJECT` binding
* QMD launch method
* Program address units
* Code PTE target and privilege bits
* Output PTE target
* GR context patch list
* SM power and context residency

If constant store works but add fails, investigate:

* Parameter ABI
* Constant-buffer index
* Pointer width and ordering
* Global load address spaces
* Register count and launch dimensions
* SASS semantics

---

## 11. Phase 8 — Add an offline RPC-budget test

Extend `--middle-selftest` with a fake TinyGPU socket and assert the exact operation class allowed in each phase.

For a successful full add, after the completion semaphore:

```text
Allowed:
  one MMIO_READ, bar=1, output range

Forbidden:
  all MMIO_READ bar=0
  all MMIO_WRITE bar=0
  CFG_READ
  CFG_WRITE
  RESET
  MAP_BAR
  server launch/termination
```

After `transport.freeze()`:

```text
Allowed:
  zero protocol frames
  local thread join
  socket.close()
```

The test should fail if one extra frame appears.

Also remove or relocate dormant live helpers such as:

* `_disable_pci_bus_master()`
* `_reset_pci_function_for_close()`
* `_shutdown_kepler_pci_for_close()`

Even if currently unused, leaving them beside the normal lifecycle makes accidental reintroduction likely.

---

## 12. Recommended immediate patch order

### P0 — Before any further live test

1. Reconcile `progress.md` and the actual post-completion code.
2. Remove all default post-completion BAR0 diagnostics and writes.
3. Add the `_rpc()` BEGIN/END flight recorder.
4. Add transport freeze after the final intended RPC.
5. Force single-threaded RPC ownership.
6. Retain the shared TinyGPU server and client-only socket close.
7. Add the no-RPC hold-open mode.
8. Disable automatic BYPASS/DISPATCH fallbacks.
9. Extend the fake-socket test to enforce the post-semaphore RPC budget.

### P1 — After one stable cold run

1. Run the constant-store kernel.
2. Validate one-element add.
3. Validate 32 elements.
4. Validate 256 elements.
5. Add only narrowly selected failure diagnostics.

### P2 — After stability and correctness

Split the 4,000-line experimental script into:

```text
kepler_transport.py
kepler_init.py
kepler_vmm.py
kepler_fecs.py
kepler_fifo.py
kepler_grctx.py
kepler_launch.py
kepler_diagnostics.py
add.py
```

Model the live path as an explicit state machine so that invalid transitions—especially endpoint access after close/freeze—are rejected automatically.

---

## 13. First live command after the P0 patch

After a physical enclosure power-cycle, run exactly one command:

```bash
mkdir -p logs

env \
  KEPLER_LIVE_ACK=completion-abort-risk \
  KEPLER_SUBMIT_MODE=gpfifo \
  KEPLER_TEST_STAGE=full-add \
  KEPLER_SINGLE_THREAD_RPC=1 \
  KEPLER_HOLD_OPEN_SECONDS=120 \
  KEPLER_RPC_TRACE="$PWD/logs/rpc-$(date +%Y%m%d-%H%M%S).log" \
  PYTHONUNBUFFERED=1 \
  DEBUG=0 \
  python3 examples_kepler/add.py \
  2>&1 | tee "logs/add-$(date +%Y%m%d-%H%M%S).log"
```

These proposed environment options must first be implemented; do not assume current `main` supports them.

After the command:

* Do not run `--probe`.
* Do not start another hardware process.
* Check whether the TinyGPU server count changed.
* Preserve the final RPC trace.
* Observe the machine beyond the prior delayed-panic interval.
* If a panic occurs, match its timestamp to the final unmatched RPC `BEGIN`.

---

## 14. Definition of fixed

The host-stability problem is fixed only when:

1. One cold full-add run completes and survives at least five minutes.
2. The RPC trace shows no config, reset, teardown BAR, or server-lifecycle request.
3. There are zero endpoint requests after transport freeze.
4. No extra TinyGPU server is launched or terminated.
5. Three separate cold-start validations complete without a macOS panic.
6. The output buffer contains the correct 256 floating-point sums.
7. A deliberate kernel-error test fails in userspace with bounded diagnostics rather than destabilizing the PCIe controller.

The highest-priority suspected defect is currently the remaining post-completion BAR0 activity visible in `add.py`, despite `progress.md` stating that the normal path had been reduced to a single BAR1 output read followed by client-only close.

