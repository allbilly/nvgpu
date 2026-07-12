Do **not** “fix” this by mapping page `0xd8` yet. Nouveau’s loader deliberately assigns sequential virtual tags starting from the supplied initial tag; for the normal GK104 FECS upload that means `0x00, 0x01, …`. Your TLB dump therefore looks structurally correct. Nouveau writes one new tag every `0x100` bytes and pads only the final page; it does not create an extra `0xd8` mapping. ([Code Browser][1])

The stronger conclusion is:

> FECS is executing the wrong byte stream, starting at the wrong byte offset, or being disassembled with the wrong Falcon ISA configuration.

A genuine GK104 FECS entry sequence should not immediately branch into a virtual page that the same firmware image never maps.

## Most likely issue: firmware format or entry offset

The first four bytes are:

```text
03 9b 0e f5
```

You previously treated the first three bytes as one unknown instruction and began the branch at byte 3:

```text
0000: 03 9b 0e      unknown
0003: f5 98 00 d8   bra ae, 0xd803
```

That interpretation is suspect for two reasons:

1. The very first instruction is unknown.
2. The resulting second instruction branches outside the mapped image.

That combination usually means **instruction-boundary desynchronization**, not a deliberate firmware path.

The `f5` byte may be part of the first valid instruction, or the blob may include a header that must not be loaded as executable code.

## Check these before modifying the loader

### 1. Confirm exactly which file is being loaded

At runtime, print:

```python
print("FECS code path:", fecs_code_path)
print("FECS code size:", hex(len(fecs_code)))
print("FECS code sha256:", hashlib.sha256(fecs_code).hexdigest())
print("FECS first 64:", fecs_code[:64].hex(" "))
```

Also print the DMEM file’s path, size and hash. This catches common errors such as:

* FECS data loaded as FECS code
* GPCCS code loaded into FECS
* an extracted container rather than the raw code segment
* a stale file from another GPU generation
* a symlink resolving to an unexpected firmware file

A code size around `0xc00` is plausible. It does not itself prove the file is correct.

### 2. Compare the file bytes with the words actually read back from IMEM

Do not compare only `imem[0]`. Dump at least the first `0x40` bytes from both sources:

```python
def dump_words(data, count=16):
    for i in range(count):
        w = int.from_bytes(data[i * 4:i * 4 + 4], "little")
        print(f"{i * 4:04x}: {w:08x}")

print("FILE:")
dump_words(fecs_code)

print("IMEM:")
for off in range(0, 0x40, 4):
    falcon_set_imem_index(off)
    print(f"{off:04x}: {rd32(FECS + 0x184):08x}")
```

The Nouveau implementation writes each host `u32` directly to the IMEM data port. On a normal little-endian host, the raw bytes `03 9b 0e f5` correspond to the MMIO word `0xf50e9b03`. ([Code Browser][1])

Test all four possible transformations explicitly:

```text
file bytes:          03 9b 0e f5
32-bit byte-swapped: f5 0e 9b 03
16-bit swapped:      0e f5 03 9b
halfword order:      9b 03 f5 0e
```

Do not apply any transformation unless one matches the known-good Nouveau image or produces a coherent disassembly.

### 3. Verify `envydis` architecture and variant options

An incorrect Falcon ISA version can decode valid opcodes as unknown and then lose synchronization.

Run:

```bash
envydis --help
```

Then test the firmware using the Falcon generation/variant appropriate for Kepler FECS. Do not rely on a bare invocation if `envydis` supports chipset or Falcon-version selection.

The acceptance criterion is not merely “no unknown instruction.” A correct decode should have:

* a valid instruction at address zero
* sensible instruction boundaries
* branches landing inside mapped code
* repeated valid code over several dozen instructions

Try decoding from offsets `0`, `1`, `2`, `3`, and `4` as a diagnostic:

```bash
for n in 0 1 2 3 4; do
    echo "=== offset $n ==="
    dd if=gk104_fecs_code.bin bs=1 skip=$n status=none |
        envydis <correct falcon options> - | head -30
done
```

Only one offset should produce sustained coherent code. If offset `4` or a larger fixed offset works, the file likely contains a header.

## Inspect the firmware file for a header

Dump the first `0x100` bytes:

```bash
xxd -g1 -l 256 firmware/gk104/gk104_fecs_code.bin
xxd -g4 -e -l 256 firmware/gk104/gk104_fecs_code.bin
```

Look for fields resembling:

* magic/version
* code size
* data size
* entry offset
* code offset
* bootloader offset

Also check whether the file was extracted from a larger NVIDIA firmware package and whether the extraction tool already produced separate files such as:

```text
fecs_inst.bin
fecs_data.bin
fecs_code.bin
hub.fuc
```

The filename alone is not sufficient evidence that byte zero is the executable entry.

## Add an entry-point sweep—not a page-tag hack

As a controlled diagnostic, set `UC_ENTRY` to candidate offsets within the existing image and observe the first PC sequence:

```python
for entry in (0x0, 0x1, 0x2, 0x3, 0x4, 0x8, 0x10, 0x20, 0x40, 0x100):
    falcon_reset()
    falcon_load_same_image()

    wr32(FECS + 0x104, entry)
    wr32(FECS + 0x100, 0x2)

    pcs = []
    for _ in range(32):
        pcs.append(rd32(FECS + 0xff0))

    print(f"entry={entry:#x}:",
          " ".join(f"{pc:#x}" for pc in pcs[:12]),
          f"ctrl={rd32(FECS + 0x100):#x}",
          f"intr={rd32(FECS + 0x008):#x}")
```

This is only a discriminator. The correct entry must come from the firmware format or Nouveau’s firmware descriptor, not from whichever offset happens to run longest.

## Patch direction for `add.py`

At the loader, preserve sequential tags:

```python
def falcon_write_imem(base, code, start=0, tag=0, port=0):
    index = start | (1 << 24)
    wr32(base + 0x180 + port * 0x10, index)

    word_count = (len(code) + 3) // 4

    for i in range(word_count):
        if (i & 0x3f) == 0:
            wr32(base + 0x188 + port * 0x10, tag)
            tag += 1

        chunk = code[i * 4:i * 4 + 4]
        word = int.from_bytes(chunk.ljust(4, b"\x00"), "little")
        wr32(base + 0x184 + port * 0x10, word)

    while word_count & 0x3f:
        wr32(base + 0x184 + port * 0x10, 0)
        word_count += 1
```

This mirrors Nouveau’s essential behavior: direct 32-bit writes, one sequential tag per 256-byte page, and final-page padding. ([Code Browser][1])

Add assertions immediately afterward:

```python
assert len(code) <= physical_imem_size
assert verify_imem_bytes(code)
assert verify_identity_tlb((len(code) + 0xff) // 0x100)
```

Do **not** introduce:

```python
tag = 0xd8
```

That could turn the current deterministic no-hit into execution of arbitrary physical bytes while hiding the actual image/entry problem.

## Current ranking

1. **Wrong or incorrectly extracted firmware file**
2. **Wrong `envydis` Falcon version causing false instruction boundaries**
3. **A header or nonzero code entry offset is being loaded as instruction zero**
4. **Byte/word transformation in `falcon_write_imem`**
5. Wrong GPU firmware variant
6. Actual firmware intentionally branching to `0xd8` — very unlikely

The decisive next artifact is a side-by-side dump of:

```text
filename + SHA-256 + length
first 64 file bytes
first 16 IMEM words
exact envydis command and version
disassembly from offsets 0–4
```

That will identify whether the breakage is extraction, upload byte ordering, or disassembler configuration before any risky TLB changes.

[1]: https://codebrowser.dev/linux/linux/drivers/gpu/drm/nouveau/nvkm/falcon/v1.c.html "v1.c source code [linux/drivers/gpu/drm/nouveau/nvkm/falcon/v1.c] - Codebrowser "


====

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
