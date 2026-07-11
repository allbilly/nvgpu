# Revised GTX 770 / GK104 `add.py` Bring-Up Plan

## Goal

Run one verified Kepler `sm_30` kernel on the GTX 770 over TinyGPU:

```text
out[0] = 0x1234abcd
```

Only after the one-thread sentinel kernel works should the project attempt vector addition.

The project is divided into two separate targets:

```text
Target A: command submission on an already initialized GPU
Target B: cold initialization from power-on
```

Target A must be completed first.

---

## Phase 0 — Disable unsafe live execution

### Required changes

Remove or gate all live writes from `NVDevice.__init__()`.

Specifically, do not execute:

```python
PMC_ENABLE = 0xffffffff
falcon_load(...)
submit_launch(...)
```

Hardware construction should perform only:

```text
connect transport
read PCI identity
read BAR information
read selected status registers
stop
```

Add an explicit write gate:

```bash
NV_ALLOW_WRITES=1
```

and require an individual phase:

```bash
python add.py --hardware --phase fifo
```

### Pass condition

Running:

```bash
python add.py --probe
```

performs no MMIO or PCI configuration writes.

---

## Phase 1 — Make the TinyGPU transport exact

### Implement one canonical API

```python
class TinyGPUTransport:
    def pci_read32(self, offset): ...
    def pci_write32(self, offset, value): ...

    def bar_info(self, bar): ...
    def bar_read32(self, bar, offset): ...
    def bar_write32(self, bar, offset, value): ...
    def bar_read(self, bar, offset, size): ...
    def bar_write(self, bar, offset, data): ...

    def alloc_sysmem(self, size): ...
    def close(self): ...
```

Do not expose inconsistent signatures such as:

```text
software: mmio_read32(offset)
hardware: mmio_read32(bar, offset)
```

### Protocol requirements

For every RPC:

* verify the response structure against the server source;
* check status for reads and writes;
* handle error payloads;
* validate response lengths;
* serialize access with a lock;
* propagate the selected device ID;
* close the socket cleanly;
* reject out-of-range BAR accesses.

Do not implement `SYSMEM_READ` or `SYSMEM_WRITE` until their numeric commands and payload formats are confirmed on the TinyGPU server.

Prefer a shared-memory file descriptor or real mmap if `MAP_SYSMEM_FD` provides one.

### Tests

Use a fake socket server to test:

* partial socket reads;
* nonzero status;
* short payload;
* wrong device ID;
* timeout;
* concurrent calls;
* 32-bit and 64-bit access;
* offset and size validation.

### Pass condition

Repeated read-only probing returns stable:

```text
PCI vendor/device
BAR0 size
BAR1 size
PMC_BOOT_0
```

with no writes and no socket framing errors.

---

## Phase 2 — Classify the actual GPU state

Do not assume the GTX 770 is cold, posted, or has unusable VRAM.

Read:

```text
PCI command register
BAR assignments
PMC_BOOT_0
engine-enable/status registers
PFB/VRAM status
VBIOS ROM signature
FIFO status
GR status
Falcon status
```

Classify the device as:

```text
STATE 0: PCI device inaccessible
STATE 1: PCI/BAR accessible, engines reset
STATE 2: GPU posted, VRAM available
STATE 3: GPU partly initialized
STATE 4: existing FIFO/GR state active
```

### Reference capture

Boot the same GTX 770 under Linux/Nouveau and capture the corresponding state after initialization. A second card is not required.

Capture:

* PCI configuration;
* selected BAR0 registers;
* VBIOS image;
* VRAM size and memory type;
* page-directory format;
* RAMFC;
* USERD;
* runlist;
* GPFIFO;
* GR context;
* compute pushbuffer;
* launch descriptor.

### Pass condition

The TinyGPU state is documented before any attempt to initialize engines.

---

## Phase 3 — Validate GPU-visible system memory

Allocate one small block, initially 64 KiB rather than 256 MiB.

The allocation object must contain:

```python
@dataclass
class RemoteAllocation:
    gpu_bus_address: int
    size: int
    host_mapping: memoryview | None
```

Provide:

```python
read(offset, size)
write(offset, data)
flush(offset, size)
```

### Tests

Run:

* zeros;
* ones;
* walking bits;
* address pattern;
* random data;
* boundary writes;
* repeated retained-write tests.

At this phase, only the host and TinyGPU transport need to access the allocation. No GPU engine is involved.

### Pass condition

Host writes and reads are coherent and repeatable across the exact allocation returned by TinyGPU.

---

## Phase 4 — Port GK104 GMMU exactly

Delete the current generic three-level page-table implementation.

Port the 4 KiB GK104 mapping path directly from Nouveau:

```text
small-page table
page directory
GF100-family PTE encoding
GK104 descriptor sizes
PDB attachment
TLB/PDB invalidation
```

Do not implement large pages, compression, peers or VRAM mappings yet.

### Required objects

```python
class GK104PTEEncoder
class GK104PageDirectory
class GK104AddressSpace
class GK104MMUInvalidator
```

### Required details

Implement:

* physical address shifted as required by the hardware;
* valid bit;
* privilege bit;
* read-only bit;
* volatile bit;
* host aperture;
* storage kind zero;
* exact PDE encoding;
* PDB address and limit in channel instance memory;
* MMU flush through the appropriate registers.

Every page-table write must modify the real GPU-visible sysmem allocation, not merely a local bytearray.

### Golden tests

For selected virtual-to-physical mappings, compare emitted 64-bit PDE/PTE values with a Nouveau capture.

### Pass condition

A manually inspected page-table dump is byte-identical to the Linux reference for equivalent mappings.

---

## Phase 5 — Implement FIFO without GR or compute

Build separate allocations for:

```text
channel instance/RAMFC
USERD
runlist
GPFIFO descriptor ring
pushbuffer
completion storage
```

The data flow must be:

```text
method words
    ↓
pushbuffer allocation

pushbuffer address + length
    ↓
GPFIFO descriptor

channel ID + scheduling data
    ↓
runlist entry
```

Never place method words directly in the GPFIFO ring.

### Implement from Nouveau

Port:

* channel-instance initialization;
* RAMFC fields;
* USERD pointer;
* GPFIFO base and ring size;
* PDB attachment;
* channel bind;
* channel start;
* runlist-entry format;
* runlist commit;
* PUT/GET accounting.

### Diagnostics

Before each submission save:

```text
RAMFC dump
USERD dump
GPFIFO dump
pushbuffer decode
runlist dump
PFIFO status
PBDMA status
MMU fault registers
```

### Pass condition

PFIFO consumes a minimal valid pushbuffer and GET advances to PUT without:

```text
MMU fault
PBDMA fault
illegal method
runlist timeout
```

---

## Phase 6 — Prove memory movement before compute

Bind the GK104 copy class or another simple non-GR memory engine.

Test:

```text
source sysmem buffer
        ↓ GPU copy
destination sysmem buffer
        ↓
host verification
```

Use a small 4 KiB pattern.

Add a completion mechanism derived from the Kepler channel/engine interface. Do not reuse modern `NVC56F` semaphore methods unless their equivalence with `A06F` is proven.

### Pass condition

The GPU copies 4 KiB correctly for at least 100 runs with no faults.

A successful copy proves:

```text
GMMU
RAMFC
runlist
GPFIFO
pushbuffer
GPU reads
GPU writes
```

---

## Phase 7 — Port GK104 GR initialization

Only now introduce:

```text
PGRAPH/GR initialization
FECS firmware
GPCCS firmware
GR context generation
context-switch buffers
per-GPC/TPC state
channel GR-context binding
```

Extract the precise firmware arrays and upload sequence from one selected Nouveau kernel version.

Do not:

* split arbitrary firmware files;
* load PMU first;
* enable every PMC engine bit;
* mix firmware from different Nouveau versions.

The context generator and firmware must come from the same reference version.

### Pass condition

The GR engine initializes and a channel context can be loaded and switched without FECS, GPCCS or PGRAPH faults.

---

## Phase 8 — Load a verified Kepler cubin

Require a real cubin produced by CUDA 10.2 or another verified `sm_30` compiler.

Hardware mode must reject the synthetic cubin.

Parse:

```text
ELF header
.text.E_4
symbol entry offset
register count
barrier count
constant-memory size
parameter metadata
shared-memory requirement
```

Upload `.text.E_4` into executable GPU memory and program the Kepler compute object’s code-base methods.

### Initial kernel

Use:

```cuda
extern "C" __global__
void sentinel(unsigned *out) {
    out[0] = 0x1234abcd;
}
```

This avoids indexing, multiple parameters and floating-point validation.

### Pass condition

The loaded SASS matches `nvdisasm`, and the metadata parser agrees with `cuobjdump`.

---

## Phase 9 — Implement the GK104 launch descriptor

Build the descriptor from the Envytools/Mesa layout, not from guessed offsets.

Required fields include:

```text
program entry offset
grid dimensions
block dimensions
register allocation
barrier allocation
shared-memory allocation
local-memory state
constant-buffer validity
constant-buffer address
constant-buffer size
cache configuration
required fixed/default fields
```

Bind the compute class to one subchannel, program its code/local/shared state, flush required caches, then issue:

```text
LAUNCH_DESC_ADDRESS = descriptor_va >> 8
LAUNCH              = verified trigger value
```

Use a golden descriptor captured from Linux for the same sentinel kernel.

### Pass condition

The generated 256-byte descriptor is byte-identical to the known-good descriptor except for expected addresses.

---

## Phase 10 — Run the sentinel kernel

Initialize:

```text
output = 0xdeadbeef
completion = 0
```

Submit one thread:

```text
grid  = 1 × 1 × 1
block = 1 × 1 × 1
```

Wait for completion while polling:

* completion memory;
* USERD GET;
* PFIFO faults;
* PBDMA faults;
* MMU faults;
* PGRAPH traps.

### Pass condition

```text
output == 0x1234abcd
completion reached
no fault registers set
```

Repeat 100 times with different virtual and physical placements.

---

## Phase 11 — Implement vector addition

Only after the sentinel passes:

1. Pack all three pointers into the kernel parameter constant buffer.
2. Use four threads and four values.
3. Increase to one warp.
4. Increase to one block.
5. Add multiple blocks.
6. Add bounds checking.
7. Test page-boundary crossings.
8. Test repeated launches and ring wraparound.

### Pass condition

At least 100 consecutive vector-add runs pass bitwise or within the chosen floating-point tolerance, with untouched guard regions.

---

## Phase 12 — Cold initialization and performance

After compute works on an initialized GPU, implement the cold path:

```text
PCI command/BAR setup
VBIOS/devinit
memory-controller setup
GDDR5 initialization
engine resets
GR firmware
thermal setup
PMU
clock selection
```

Reclocking comes last. The initial driver should operate at the safe boot performance state.

### Final definition of working

```text
[ ] Read-only probe is reliable
[ ] TinyGPU RPC protocol is verified
[ ] Remote sysmem is coherent
[ ] Exact GK104 GMMU works
[ ] FIFO and runlist work
[ ] Copy engine works
[ ] GR context switching works
[ ] Verified sm_30 cubin loads
[ ] Golden launch descriptor matches
[ ] Sentinel kernel passes repeatedly
[ ] Vector addition passes repeatedly
[ ] Fault capture and recovery work
[ ] Cold initialization is reproducible
```
