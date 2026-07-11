# Full Code Review: Standalone GK104 / GTX 770 eGPU Stack

## Executive summary

The code is a useful bring-up notebook packaged as a Python module, but it is **not yet a viable userspace driver or kernel-launch implementation**. The software-only path is internally consistent enough to exercise allocations, copies, descriptor construction, and a CPU-simulated add. The live path, however, fails at several independent layers before a Kepler kernel could execute.

**Overall assessment:** prototype / research scaffold.

**Live-path readiness:** not runnable.

**Safety recommendation:** do not run the full hardware path with `DEBUG=1`; it performs arbitrary writes to GR and FECS control registers.

The highest-priority blockers are:

1. Lower-level GMMU page-directory entries use the VRAM target even though the page tables reside in sysmem.
2. The FALCON context-data helper is dead code and does not complete the FECS initialization sequence.
3. The GPFIFO contains raw pushbuffer words rather than GP entries that point to a separate pushbuffer.
4. The GPFIFO base, complete RAMFC, PBDMA state, runlist, engine bindings, and valid GR context are absent.
5. The hardware demo uploads a complete ELF cubin as executable instructions and ignores cubin metadata and relocations.
6. The live path can silently use placeholder, non-executable SASS.
7. The initial semaphore wait can block forever because the semaphore begins at zero but the command stream waits for one.

---

## Scope and validation performed

The reviewed file contains 1,584 lines.

I performed the following checks:

| Check | Result |
|---|---|
| Python syntax compilation | Pass |
| Direct `--middle-selftest` execution in the current environment | Fails at import: `ModuleNotFoundError: tinygrad` |
| Offline self-test with tinygrad autogen imports stubbed | Pass |
| Software demo with the same stubs | Pass |
| `HCQBuffer.offset()` targeted test | Fails with `AttributeError` |
| Software `NVDevice.read32()` targeted test | Fails with `TypeError` |
| Hardware-style sysmem page-table target inspection | Lower PDE targets are VRAM; leaf target is SYS |
| Cubin fallback inspection | Falls back to the known placeholder cubin |
| `falcon_csdata_write()` call-site count | One occurrence: its definition; no call sites |

Observed test output:

```text
kepler_selftest=ok cubin_sha=cdf45fde3229835b343a301f6bfa52aa45e0f760ba7221ea1155938a42927bd0 launch_words=26 sections=12
software_demo=ok N=256 launch_words=26 cwd_bytes=256

offset_error: AttributeError 'memoryview' object has no attribute 'view'
read32_error: TypeError SoftwarePCIDevice.mmio_read32() takes 2 positional arguments but 3 were given

sys_pde_targets: 0x0 0x0
leaf_target:      0x20
expected_sys:     0x20

fallback_is_placeholder: True
falcon_csdata_write references: 1
```

The passing software demo does **not** execute SASS or submit work to the emulated FIFO. It performs the vector addition in a Python loop.

---

## What is good

The project has several useful qualities:

- It states clearly that the hardware path is incomplete.
- Software and hardware backends are separated at a high level.
- The code is organized by subsystem: allocators, GMMU, transport, FALCON, cubin, launch descriptor, and demos.
- The software path gives deterministic coverage for basic memory allocation and copy behavior.
- The self-test checks ELF identity, method-stream shape, helper arithmetic, and a basic three-level page-table walk.
- The firmware locator and timeout diagnostics are practical for incremental bring-up.
- Using GPU-visible coherent sysmem is a reasonable initial strategy when local VRAM initialization is unavailable.

These strengths make the file useful as an experimental harness. They do not compensate for the missing hardware state required by Host, PBDMA, GR, and the compute engine.

---

# Critical findings

## C1. Hardware sysmem page tables are linked through VRAM-target PDEs

**Location:** `GK104MemoryManager.map_range()`, approximately lines 285–303.

The leaf mapping receives the caller's address space:

```python
pt.set_entry(lv0, p, table=False, aspace=aspace)
```

But newly allocated page-table levels do not:

```python
pd2.set_entry(lv2, pd1_pa, table=True)
pd1.set_entry(lv1, pt_pa, table=True)
```

`set_entry()` defaults to `AddrSpace.PHYS`, which encodes `PTE_APER_VRAM`. On the hardware path, the root, PD1, PT, and mapped buffers all live in the mmap-backed sysmem allocation. The root entry in `_gk104_pgd_entry()` is marked SYS, but its child PDEs are marked VRAM.

The resulting walk is conceptually:

```text
RAMIN root pointer  --SYS--> root PD
root PD entry       --VRAM-> PD1       incorrect
PD1 entry           --VRAM-> PT        incorrect
PT leaf             --SYS--> data page
```

The GPU cannot reach PD1 or PT through those targets.

### Required fix

Page-table allocations need an explicit table address space:

```python
table_aspace = AddrSpace.SYS if self.bus_base else AddrSpace.PHYS

pd2.set_entry(lv2, pd1_pa, table=True, aspace=table_aspace)
pd1.set_entry(lv1, pt_pa, table=True, aspace=table_aspace)
```

A cleaner design stores `page_table_aspace` on the memory manager and never relies on the leaf mapping's address space for table pages.

Add a hardware-style test that checks all three target fields, not merely the presence bits.

---

## C2. The PTE frame mask contradicts the claimed 40-bit address format

**Location:** `GK104PageTableEntry.PTE_FRAME`, approximately line 212.

The code claims address bits `[39:12]`, but defines:

```python
PTE_FRAME = 0xFFFFFF000
```

That mask has a bit length of 36, so it preserves only through bit 35. A mask for bits `[39:12]` would require ten hexadecimal address digits above the page offset, for example:

```python
0xFFFFFFFFF000
```

The exact silicon encoding must still be verified, but the present constant is internally inconsistent with the code's own 40-bit VA/PA model. High I/O virtual addresses can be silently truncated.

### Required fix

- Verify the precise GK104 PTE/PDE address field.
- Derive the mask from named bit positions rather than a hand-written literal.
- Reject addresses that cannot be represented instead of truncating them.

Example:

```python
PTE_ADDR_LO = 12
PTE_ADDR_HI = 39
PTE_FRAME = ((1 << (PTE_ADDR_HI + 1)) - 1) & ~((1 << PTE_ADDR_LO) - 1)

if (bus + paddr) & ~PTE_FRAME:
    raise ValueError("physical address is not representable by GK104 PTE")
```

---

## C3. FECS context-data initialization is still absent

**Locations:** `_init_hardware()` around lines 761–770 and `falcon_csdata_write()` around lines 1146–1167.

The hardware path performs:

```text
load FECS DMEM/IMEM
load GPCCS DMEM/IMEM
release gate
start FECS
wait for ready
```

The newly added `falcon_csdata_write()` is never called. There are no embedded or parsed GK104 hub/GPC/TPC/PPC register-init packs. Consequently, this revision does not change the actual FECS startup sequence.

The helper also cannot be considered validated:

- Its only input is a flat list of words.
- It has no distinct register-base parameter for hub, GPC, TPC, and PPC streams.
- Its docstring claims values are already in FUC DMEM, but the required register initialization data includes explicit address/count/pitch/value information.
- It mixes special selector values with ordinary raw DMEM `WRITE` indexing without a transcript test against a known implementation.
- It reads two words from the selected data port, chooses the maximum, streams arbitrary words, and writes an end pointer, but no test establishes that this is the required protocol.

### Required fix

Represent context initialization data explicitly:

```python
@dataclasses.dataclass(frozen=True)
class GRInit:
    addr: int
    count: int
    pitch: int
    value: int
```

Then port the exact hub, GPC0, GPC1, TPC, and PPC sequences and generate the expected MMIO transaction transcript. Call all required loaders before `falcon_start()`.

The immediate acceptance test should be read-only:

```text
firmware loaded
context-data streams installed
FECS starts
ready mailbox reaches expected state
context image size is plausible
no arbitrary post-start control-register writes
```

---

## C4. GPFIFO and pushbuffer are conflated

**Location:** `submit_launch()`, approximately lines 1432–1479.

The function allocates `gpfifo` and writes the method stream directly into it:

```python
ring = bytearray(len(words) * 4)
for i, w in enumerate(words):
    struct.pack_into("<I", ring, i * 4, w)
vram[gpfifo_pa:gpfifo_pa + len(ring)] = bytes(ring)
```

That memory is a pushbuffer segment, not a GPFIFO. A GPFIFO should contain GP entries describing pushbuffer segment addresses and lengths.

The required relationship is:

```text
GPFIFO ring
  entry 0 = {pushbuffer GPU VA, pushbuffer length, flags}
                       |
                       v
pushbuffer allocation
  method header
  method data
  method header
  method data
  ...
```

The comment stating that “GPFIFO entries are 8-byte METHOD headers” is incorrect. Method headers are 32-bit pushbuffer entries. GP entries are a separate host structure.

### Required fix

Create two allocations:

```python
pushbuf = alloc.alloc(round_up(len(words) * 4, 0x100))
gpfifo = alloc.alloc(GPFIFO_ENTRY_SIZE * GPFIFO_ENTRY_COUNT)
```

Write raw method words to `pushbuf`. Encode one valid GP entry in `gpfifo` using `pushbuf.va_addr` and `len(words)`.

---

## C5. `GP_PUT` is written as a physical byte address instead of a ring index

**Location:** `submit_launch()`, approximately lines 1475–1477.

The code writes:

```python
put = gpfifo_pa + len(ring)
```

`GP_PUT` is a producer position in the GP-entry circular buffer. It is not the physical address of the end of the written data. After publishing one GP entry at index zero, a typical initial value is one, subject to the exact ring format.

### Required fix

After writing one GP entry and performing the required memory ordering:

```python
userd_gp_put = 1
```

Also initialize the GPFIFO base and limit fields in the appropriate RAMFC state. The current code never records `gpfifo_pa` or `gpfifo.va_addr` in RAMFC, so Host has no way to locate it.

---

## C6. The channel, RAMFC, PBDMA, and runlist setup is incomplete

**Location:** `submit_launch()`.

The function creates a mostly zero-filled RAMIN image and writes a few guessed fields. It does not establish a complete executable Host channel.

Missing or unvalidated state includes:

- GPFIFO base and ring-size encoding.
- GP fetch configuration.
- PBDMA assignment and state.
- Channel engine bindings.
- Compute object/context binding.
- Runlist storage and runlist entries.
- Runlist base programming.
- Required channel configuration/authentication fields.
- USERD layout validation.
- TLB/cache invalidation after page-table updates.
- A memory-ordering barrier before publishing `GP_PUT` and ringing the channel.

This line is not a runlist implementation:

```python
dev.write32(PFIFO_RUNLIST_SUBMIT, chan_id)
```

A runlist submission trigger must refer to a previously constructed and programmed runlist.

The channel-start write also contradicts its comment:

```python
# Channel start: ... |= 0x400
dev.write32(CHAN_START_REG + chan_id * 8, 0x400)
```

If the register requires preservation of other fields, this clobbers them.

### Required fix

Do not proceed directly to a compute kernel. Implement and validate in this order:

1. Channel instance and RAMFC.
2. GPFIFO base/limit and GP GET/PUT.
3. Runlist allocation, entry encoding, base programming, and commit.
4. PBDMA assignment.
5. A minimal pushbuffer that performs no engine work or only a host semaphore release.
6. Channel completion and fault inspection.
7. Compute-class binding.

---

## C7. The GR context is an invalid zero-filled buffer

**Location:** `submit_launch()`, approximately lines 1444–1448.

The code allocates one MiB and leaves it zeroed:

```python
gr_ctx = alloc.alloc(0x100000)
```

A GR context is generated from architecture-specific context initialization, bundle, pagepool, attribute, and context-switch data. A zero-filled allocation is not a valid substitute.

The code acknowledges this as a TODO but still continues to bind and start the channel.

### Required fix

Fail closed until GR-context generation exists:

```python
raise NotImplementedError("GK104 GR context generation is not implemented")
```

Then port the GK104 generation procedure and verify the generated image against a Nouveau trace or a known-good dump.

---

## C8. The entire cubin ELF is uploaded as executable code

**Location:** `run_hardware_demo()`, approximately lines 1495–1516.

The hardware demo performs:

```python
cubin = get_kepler_cubin()
code_dev = allocator.alloc(len(cubin))
allocator._copyin(code_dev, cubin)
...
build_cwd(code_addr=0, ...)
```

This places the ELF header at the shader code base. The GPU executes machine instructions; it does not load ELF sections, resolve symbols, apply relocations, or parse `.nv.info` metadata.

`NVProgram` currently stores bytes only:

```python
class NVProgram:
    def __init__(...):
        self.cubin = lib
```

It does not parse:

- `.text.E_4`.
- The `E_4` symbol and entry offset.
- `.nv.info` and `.nv.info.E_4`.
- Register count.
- Constant-bank size and parameter layout.
- Shared/local memory requirements.
- Relocations.

### Required fix

Implement a real cubin loader that returns a structured program object:

```python
@dataclasses.dataclass
class KeplerProgramImage:
    image: bytes
    entry_va_offset: int
    text_size: int
    regs: int
    shared_bytes: int
    local_bytes: int
    cbuf0_size: int
    parameters: list[KernelParameter]
```

The loader should build a relocated load image, upload it, and set `code_va` and `code_addr` to the actual function entry.

---

## C9. Placeholder SASS can silently reach the live path

**Location:** `get_kepler_cubin()`, approximately lines 1073–1081.

When no prebuilt cubin is supplied and Docker compilation fails, the function returns `build_cubin()`. The code itself labels that cubin structural and non-executable.

This is dangerous because the hardware demo cannot distinguish a verified cubin from placeholder bytes.

### Required fix

Use separate APIs:

```python
def get_verified_kepler_cubin() -> bytes:
    ...
    raise RuntimeError("No verified sm_30 cubin is available")


def build_placeholder_cubin_for_tests() -> bytes:
    ...
```

The live path must never call the placeholder builder.

Also validate:

- ELF magic and class.
- `EM_CUDA`.
- sm_30 flags.
- Presence of the requested function.
- Nonempty executable section.
- Parameter and register metadata.

---

## C10. The kernel ABI and descriptor metadata are hard-coded

**Locations:** `build_cwd()` and `run_hardware_demo()`.

The code assumes:

```python
regs=4
cbuf_size=256
pointer offsets = 0x00, 0x08, 0x10
```

Those values are not derived from the selected cubin. Even a valid nvcc-generated `E_4` may use a different register count, constant-bank size, parameter base, or parameter offsets.

The placeholder SASS comments refer to offsets around `0x160`–`0x174`, which already contradict the demo's `0x00`–`0x10` writes.

### Required fix

Populate the CWD and parameter buffer from cubin metadata. Add descriptor validation for:

- Grid and block dimensions.
- Nonzero sizes.
- Field-width limits.
- CWD address alignment.
- Constant-buffer address alignment.
- Shared-memory alignment.
- Register allocation bounds.

---

## C11. The initial semaphore wait can deadlock the queue

**Location:** `run_hardware_demo()` and `build_launch_words()`.

`signal` is allocated from zeroed memory. The first semaphore operation waits for `wait_value=1`:

```python
words = build_launch_words(signal.va_addr, 1, 2, ...)
```

Nothing writes one before submission. Under the intended semantics, the queue blocks before reaching cache invalidation or launch.

### Required fix

For the first channel test, remove the acquire entirely. After the host path works, either initialize the semaphore:

```python
allocator._copyin(signal, struct.pack("<I", 1))
```

or use a wait value consistent with its initial state. Confirm the exact semaphore operation flags and payload width.

---

# High-severity findings

## H1. `HCQBuffer.offset()` is broken for the software backend

**Location:** lines 377–385.

`cpu_view()` returns a built-in `memoryview`, but `offset()` calls:

```python
self.cpu_view().view(...)
```

Python `memoryview` has no `.view()` method.

### Fix

Keep the original HCQ ownership/view model or support both types explicitly:

```python
base = self.cpu_view()
size = self.size - off if size is None else size
if isinstance(base, memoryview):
    view = base[off:off + size]
else:
    view = base.view(offset=off, size=size)
```

Validate `off >= 0`, `size >= 0`, and `off + size <= self.size`.

---

## H2. MMIO method signatures are inconsistent

**Locations:** `SoftwarePCIDevice` and `NVDevice`.

`NVDevice.read32()` calls:

```python
mmio_read32(0, off)
```

But the software backend defines:

```python
mmio_read32(self, offset)
```

The same mismatch exists for writes. The live backend lacks `mmio_read64()` and `mmio_write64()` entirely, while `NVDevice` exposes them.

### Fix

Define one interface everywhere:

```python
mmio_read32(bar: int, offset: int) -> int
mmio_write32(bar: int, offset: int, value: int) -> None
mmio_read64(bar: int, offset: int) -> int
mmio_write64(bar: int, offset: int, value: int) -> None
```

Implement 64-bit access as two little-endian 32-bit operations if the transport has no native 64-bit RPC.

---

## H3. Remote MMIO subviews ignore offset and size

**Locations:** `RemoteMMIOInterface.view()` and `APLRemotePCIDevice.map_bar()`.

Both discard the requested `offset` and `size`. Every derived view still addresses the whole BAR from zero.

### Fix

Track a base offset:

```python
class RemoteMMIOInterface:
    def __init__(self, pci_dev, bar, fmt="B", offset=0, size=None):
        _, bar_size = pci_dev.bar_info(bar)
        self.offset = offset
        self.nbytes = bar_size - offset if size is None else size
```

Every read/write must add `self.offset`. Slice handling must account for element size, step, bounds, and negative indices or explicitly reject unsupported cases.

---

## H4. The socket FD receive path assumes a complete message

**Location:** `_rpc()`, approximately lines 511–525.

For FD-bearing responses, one `recvmsg(17, ...)` call is assumed to return all 17 bytes and a valid first ancillary record. Unix stream sockets can return a partial payload.

The protocol comment says the response is `<QQB>`, while the implementation parses `<BQQ>`. That contradiction must be resolved against the pinned server version.

### Fix

- Validate `len(msg)` and receive the remainder.
- Validate `ancdata` is nonempty.
- Check `SOL_SOCKET` and `SCM_RIGHTS`.
- Validate the received FD.
- Define request/response structs once with named `struct.Struct` instances.
- Add protocol fixture tests using `socket.socketpair()`.

---

## H5. Sysmem contiguity and length are assumed, not verified

**Location:** `_init_hardware()` and `alloc_sysmem()`.

The code requests contiguous memory and then uses only:

```python
dev.bus_base = paddrs[0]
```

It does not check:

- `paddrs` is nonempty.
- Enough pages were returned.
- Each page is contiguous with the first.
- `mapped_size` covers the requested allocation.
- Returned segment sizes are page-aligned.

If the allocation is scattered, every page after the first receives the wrong bus address.

### Fix

Prefer mapping the actual physical-page list into the GMMU. If the implementation requires contiguity, verify it explicitly and fail with a clear error.

---

## H6. Page-table writes are not followed by a GPU TLB/cache invalidation

The CPU modifies page tables in coherent sysmem and immediately proceeds toward channel submission. Coherent CPU visibility does not automatically establish that the GPU's translation caches have consumed the new mappings.

### Fix

Implement the appropriate GK104 VM flush/invalidation sequence after page-table updates and before the channel can reference newly mapped VAs. Encapsulate it in the memory manager so callers cannot forget it.

---

## H7. Debug mode performs unsafe control-register writes

**Location:** `_init_hardware()`, approximately lines 793–798.

`DEBUG=1` writes:

```python
self.write32(0x400100, 0x12345678)
self.write32(0x409100, 0x2)
```

The latter is the FECS start/control path used elsewhere in the same file. Neither is a harmless transport test.

### Fix

Use read-only diagnostics after FECS starts. If a write/read test is required, use a documented scratch register that is known to be safe for the exact chip and state.

---

## H8. Firmware inputs are silently truncated

`falcon_write_dmem()` and `falcon_write_imem()` drop trailing bytes that are not a multiple of four:

```python
data = data[:len(data) // 4 * 4]
```

A malformed firmware image should be rejected, not silently changed. The code also does not check IMEM/DMEM capacity or verify the complete loaded image.

### Fix

Validate size and alignment before any MMIO write. Hash firmware blobs, check expected sizes, and optionally read back sampled words before starting the FALCON.

The timeout diagnostic should derive its expected first word from:

```python
struct.unpack_from("<I", fecs_code, 0)[0]
```

rather than embedding a manually byte-swapped literal.

---

## H9. `--probe-falcon` creates two independent device connections

The command first calls `APLRemotePCIDevice.probe()`, then constructs `NVDevice(backend="hardware")`, which creates another `APLRemotePCIDevice` and another temporary socket path. The first connection is not reused. The `NVDevice` connection is not finalized in this branch.

### Fix

Allow dependency injection:

```python
dev = NVDevice(backend="hardware", pci_dev=d)
```

Use `try/finally` or context managers to close every socket and clean temporary paths.

---

# Medium-severity findings

## M1. The file is not actually standalone

The header calls it standalone, but import fails unless tinygrad is installed because autogen modules are imported at module load time. This prevents even the offline self-test from running in a clean Python environment.

Possible fixes:

- Vendor the required generated constants.
- Move hardware-only imports behind the hardware backend.
- Provide a clear dependency check with an actionable error.
- Rename the claim from “standalone” to “single-file, requires tinygrad autogen modules.”

---

## M2. `FileIOInterface` mishandles file descriptor zero

```python
self.fd = fd or os.open(path, flags)
```

If `fd == 0`, it is treated as absent.

Use:

```python
self.fd = os.open(path, flags) if fd is None else fd
```

---

## M3. `wait_cond()` is a hot busy loop

The function repeatedly invokes the callback without sleeping. Over a remote MMIO transport, that can flood the socket. If `timeout_ms <= 0`, `val` can also be uninitialized in the error message.

Use `time.monotonic_ns()`, initialize `val`, and add a configurable poll interval.

---

## M4. The VA allocator is global mutable class state

```python
GK104MemoryManager.va_allocator = TLSFAllocator(...)
```

Every new memory manager resets the allocator for all devices. Multiple devices or repeated initialization can produce overlapping virtual addresses.

Make it an instance field.

---

## M5. Several API parameters are accepted but ignored

Examples:

- `valid`, `uncached`, `snooped`, and `frag` in `set_entry()`.
- `boot` in `map_range()`.
- `contiguous` in `valloc()`.
- `host`, `cpu_access`, and `force_devmem` in `PCIIfaceBase.alloc()`.
- `off` and `size` in `map_bar()`.

Ignored parameters make call sites appear more capable than they are. Either implement them or remove them until supported.

---

## M6. No meaningful deallocation or teardown exists

`free()`, `device_fini()`, `NVDev.fini()`, and `synchronize()` are no-ops. Every submission leaks RAMIN, USERD, GPFIFO, GR context, code, descriptor, and data allocations. Channels are not stopped or unbound.

Add deterministic lifecycle management before repeated hardware experiments.

---

## M7. Assertions are used as production validation

Most self-test and output checks use `assert`. Running Python with `-O` removes them.

Use explicit exceptions in runtime code and a real test framework for tests.

---

## M8. Cubin compilation is repeated and leaks temporary directories

`run_hardware_demo()` calls `get_kepler_cubin()` twice. Without `KEPLER_CUBIN`, this may launch Docker twice. `compile_kepler_cubin_docker()` leaves its temporary directory behind and suppresses useful compiler diagnostics unless debug output happens to expose the exception.

Compile or load once, cache the result, and clean temporary files.

---

## M9. Backend selection is not validated

Any backend string other than exactly `"software"` selects the hardware path. A typo can initiate live MMIO operations.

Use an enum or explicit validation:

```python
if backend not in {"software", "hardware"}:
    raise ValueError(...)
```

---

## M10. Main error handling is incomplete

The main constructor handler catches only `NotImplementedError` and `OSError`. Hardware startup can raise `RuntimeError`, `TimeoutError`, `ConnectionError`, `IndexError`, or protocol exceptions, which bypass the intended diagnostic path.

Catch expected bring-up exceptions at the CLI boundary while preserving tracebacks under a debug option.

---

## M11. Input and bounds validation is sparse

Examples:

- `nvm()` does not validate method alignment, subchannel range, or argument count.
- `decode_words()` accepts truncated streams without reporting a malformed packet.
- `build_cwd()` does not validate dimension width, nonzero block sizes, address alignment, shared-memory alignment, or register count.
- Allocators permit unusual or invalid sizes without a consistent contract.
- Buffer copies do not provide explicit size diagnostics.

Add validation near construction points rather than debugging resulting GPU faults.

---

## M12. Comments and implementation contradict each other

Examples:

- The code says to avoid `0xffffffff`, then writes it immediately.
- The GPFIFO allocation comment says “256 entries x 16 bytes,” while the following comment calls entries eight-byte method headers and then stores four-byte words.
- The header claims the offline gate validates the cubin builder, but it validates only internal structure, not executable Kepler instructions.

For low-level hardware work, stale comments are especially costly. Treat comments as part of the correctness surface.

---

# Test-suite review

## What the current tests establish

The tests show that:

- The placeholder ELF has the expected fixed byte length.
- Header offsets and counts match the same constants used to build it.
- The method encoder and decoder agree with one another.
- A software-only page-table walk can resolve a low VRAM address.
- Basic allocation and copy operations work in a host bytearray.
- A Python loop computes vector addition correctly.

## What they do not establish

They do not validate:

- Real sm_30 SASS.
- ELF section loading, symbols, relocations, or metadata.
- Sysmem page-table targets.
- High physical addresses.
- TLB invalidation.
- FALCON context-data installation.
- FECS/GPCCS behavior.
- RAMFC or USERD layout.
- GPFIFO GP-entry encoding.
- Runlist programming.
- PBDMA assignment.
- GR-context generation.
- Semaphore method semantics.
- Compute descriptor field correctness.
- Actual hardware execution.

## Recommended test layers

### Layer 1: pure unit tests

Test:

- Bitfield encoders and representable-address checks.
- PTE/PDE target and frame encoding.
- Pushbuffer method encoding.
- GP-entry encoding.
- RAMFC field packing.
- Runlist entry packing.
- CWD fields and validation.
- Cubin ELF parsing and relocation.
- Socket protocol framing and FD transfer.

### Layer 2: recorded-transcript tests

Record known-good register and memory traces and compare generated transactions for:

- FALCON firmware load.
- Context-data installation.
- Channel construction.
- Runlist commit.
- A semaphore-only pushbuffer.

### Layer 3: hardware milestones

Each milestone should stop immediately after one newly validated behavior:

1. Read stable identity registers.
2. Enable engines and read expected status.
3. Load and verify FECS/GPCCS firmware without starting.
4. Install context data and reach FECS ready.
5. Allocate sysmem and validate GMMU with a minimal engine access.
6. Start a Host channel and execute a semaphore release.
7. Bind the compute class without launching.
8. Upload a parsed, verified cubin.
9. Launch a one-thread kernel.
10. Launch the vector-add kernel.

Do not combine multiple unvalidated milestones into `run_hardware_demo()`.

---

# Recommended architecture

Split the file into focused modules:

```text
kepler/
  transport.py      TinyGPU framing, BAR access, sysmem allocation
  registers.py      typed constants and bitfields
  memory.py         allocators, GMMU, VM flush
  falcon.py         firmware loading and context-data installation
  fifo.py           RAMFC, USERD, GP entries, pushbuffer, runlist
  gr.py             GR initialization and context generation
  cubin.py          ELF parser, metadata, relocations
  compute.py        CWD construction and compute methods
  device.py         staged initialization and lifecycle
  tests/
```

Use explicit state transitions:

```python
class BringupState(enum.IntEnum):
    CONNECTED = 1
    ENGINES_ENABLED = 2
    FIRMWARE_LOADED = 3
    FECS_READY = 4
    VM_READY = 5
    CHANNEL_READY = 6
    COMPUTE_READY = 7
```

Every operation should assert the required state and fail closed.

---

# Prioritized remediation plan

## Phase 0: make the harness safe and deterministic

- Remove arbitrary debug writes.
- Validate backend names.
- Fix resource cleanup and duplicate connections.
- Fix MMIO signatures and remote subviews.
- Fix `HCQBuffer.offset()`.
- Make the offline test runnable without a full tinygrad install, or correct the documentation.
- Separate placeholder and verified cubins.

## Phase 1: repair and validate GMMU

- Correct page-table target apertures.
- Verify the PTE/PDE frame width and bit positions.
- Map actual sysmem segments rather than assuming contiguity.
- Add representability and alignment checks.
- Implement VM flush/invalidation.
- Test high bus addresses and multi-segment mappings.

## Phase 2: finish FECS bring-up

- Port actual GK104 register-init packs.
- Implement the exact context-data protocol.
- Add transaction-level tests.
- Start FECS only after context data is installed.
- Remove post-start destructive diagnostics.

## Phase 3: build a real Host channel

- Implement complete RAMFC and USERD state.
- Encode GPFIFO GP entries.
- Store methods in a separate pushbuffer.
- Program GPFIFO base and size.
- Build and commit a runlist.
- Assign PBDMA and engine contexts.
- Execute a semaphore-only submission.

## Phase 4: generate a valid GR context

- Port GK104 GR-context generation.
- Bind the GR context to the channel.
- Verify context-switch behavior and fault registers.

## Phase 5: load and launch a real program

- Parse cubin ELF sections and symbols.
- Apply relocations.
- Parse register, shared/local-memory, and parameter metadata.
- Upload the load image with prefetch padding.
- Construct the parameter constant bank from metadata.
- Populate the CWD from the parsed program.
- Begin with one thread and one output store.

---

# Acceptance criteria for “first real kernel”

The project should not claim a successful Kepler compute path until all of the following are true:

- FECS reaches ready after the complete initialization sequence.
- Every page-table level has the correct target and address encoding.
- A VM flush is performed after mappings change.
- The channel appears in a valid committed runlist.
- Host consumes a GP entry and its referenced pushbuffer.
- `GP_GET` advances from zero to one.
- A semaphore-only pushbuffer reaches completion without Host/PBDMA/GR faults.
- The cubin loader identifies `.text.E_4`, entry offset, register count, and parameters.
- The uploaded program begins with actual SASS rather than ELF bytes.
- The CWD uses metadata-derived fields.
- A one-thread kernel writes a known value.
- The vector-add result is produced by the GPU, not by the Python simulator.

---

# Final assessment

The code is valuable as a **bring-up planning document and offline structural harness**, but the live path currently crosses too many unimplemented boundaries at once. The most important newly confirmed defect is the sysmem GMMU hierarchy: the root points to sysmem, while lower page-table links point to VRAM. Even a perfect FIFO and compute descriptor could not work through that translation tree.

After fixing the GMMU, the next milestone should be complete FECS initialization. After FECS is stable, implement a proper Host channel and semaphore-only pushbuffer. Cubin parsing and compute launch should come only after those layers work independently.

**Recommended status label:** `experimental / non-executable hardware skeleton`.
