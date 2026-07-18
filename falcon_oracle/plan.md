# Falcon State Oracle Development Plan

**Project:** GTX 770 / GK104 TinyGPU bring-up  
**Target:** NVIDIA Falcon firmware used by the local GK104 PMU/FECS path  
**Plan version:** 2026-07-18  
**Primary implementation language:** Python  
**Primary references:** local Nouveau source under `ref/linux/`, local envytools documentation/rnndb, existing `examples_kepler/add.py` fake and trace tests

---

## 1. Objective

Develop a small, deterministic Falcon state oracle that executes selected real Falcon firmware images and explains their state transitions.

Given:

- an IMEM image;
- a DMEM image;
- an entry PC;
- initial Falcon register and flag state;
- an initial MMIO/device state;
- a scripted set of asynchronous hardware events;

the oracle must produce:

- the final Falcon CPU state;
- all DMEM changes;
- all externally visible MMIO reads and writes;
- all Falcon XFER requests and completions;
- sleep, wait, halt and fault transitions;
- a deterministic execution trace;
- the first unsupported instruction or first divergence from the semantic reference path.

The first useful oracle is **instruction-accurate and event-driven**, not cycle-accurate.

```text
IMEM + DMEM + initial state + external events
                    |
                    v
             Falcon executor
                    |
        +-----------+------------+
        |                        |
        v                        v
 internal CPU trace       shared GK104 fake/bus
                                 |
                                 v
                       normalized external trace
```

The oracle should answer questions such as:

- Did the actual firmware execute the expected `ENTER -> XFER -> xdwait -> LEAVE` sequence?
- Which register or flag caused a branch to be taken?
- Did a helper clobber `$r0`?
- Did a transfer fail to start, fail to complete, or complete with the wrong bytes?
- Does actual `memx.fuc` produce the same MMIO effects as the existing semantic MEMX implementation?
- At which PC does the firmware first diverge from expected source-level behavior?

---

## 2. Current project evidence to preserve

The implementation must build on the existing verified infrastructure rather than duplicate it.

### 2.1 Existing authorities

Use these references in order:

1. Local Nouveau Falcon source and GK104 implementation under `ref/linux/`.
2. Local envytools Falcon ISA documentation and rnndb.
3. Existing assembled `.fuc`, `.fuc3.h`, extracted binary and disassembly artifacts.
4. Live hardware observations recorded in the project progress log.
5. The existing Python fake and exact trace tests.

Do not infer ISA behavior from guessed opcode names or comments when envytools or Nouveau source is available.

### 2.2 Existing test infrastructure

The project already has:

- `--middle-selftest`;
- a Python software demo;
- 24 exact `mmiotrace` checkpoints;
- a source-locked cold path containing 1,636 exact writes;
- fake-device assertions for ordering and transfer prerequisites;
- source/header/Python byte comparisons for embedded Falcon code;
- live hardware evidence for PMU MEMIF and autonomous XFER behavior.

The oracle must integrate with these tests rather than create an isolated test universe.

### 2.3 First firmware corpus target

The first firmware image is the custom PMU BAR1 bootstrap pad:

- assembled size: `0xde` bytes;
- embedded below the `0xc00` IMEM boundary;
- source, generated header and Python bytes already checked for exact equality;
- behavior includes `ENTER`, three direct-VRAM transfers, transfer waits and `LEAVE`;
- live hardware has verified all three fragments and all 40 bytes after `LEAVE`.

This is the smallest useful image and the best MVP target.

### 2.4 Historical regression facts

The oracle must preserve tests for known Falcon-specific failures:

- `nv_iowr` clears `$r0`;
- an earlier loop using `$r0` failed after its first write;
- `xdst` operations need serialization with `xdwait`;
- instruction placement beyond live IMEM must fail;
- `ENTER` and `LEAVE` have real wait semantics, not simple fixed delays;
- MEMX WR32 pairs execute serially;
- a queued EXEC is not proof that every later firmware-side operation completed.

---

## 3. Scope

### 3.1 Phase-one scope

The first implementation supports only what is required to execute the PMU bootstrap pad:

- Falcon instruction fetch and decode;
- general-purpose registers and required flags;
- PC changes, branches, calls and returns;
- bounded IMEM and DMEM;
- the instruction forms present in the pad;
- Falcon MMIO reads and writes;
- direct-VRAM XFER store/load operations;
- `xdwait`;
- scripted asynchronous device events;
- deterministic trace output;
- semantic-versus-Falcon differential comparison.

### 3.2 Phase-two scope

After the pad passes:

- execute stock `memx.fuc`;
- model its command buffer in DMEM;
- support its complete executed instruction subset;
- compare actual firmware effects with the existing semantic MEMX path;
- run minimal and full RAMFUC/MEMX fixtures.

### 3.3 Phase-three scope

Optional FECS expansion:

- execute the internal Nouveau `hubgk104.fuc3` path;
- model command FIFO and scratch/mailbox registers;
- model interrupt entry and return;
- model `ctx_chan`, `ctx_save`, and internal CHSW command flow;
- support VM-mode Falcon XFER only after the direct-VRAM XFER model is stable.

### 3.4 Non-goals

The initial oracle must not model:

- analog GDDR5 training;
- PLL analog lock behavior;
- PCIe link negotiation;
- platform or option-ROM POST;
- PBUS/PRAMIN hidden initialization;
- cycle-accurate instruction timing;
- every Falcon generation;
- SEC2 Heavy Secure mode;
- WPR, PLMs or signature verification;
- GPU-wide interrupt routing;
- full PGRAPH, PFIFO or GR simulation;
- arbitrary unknown firmware without explicit instruction coverage.

Unknown state must remain unknown. Unsupported behavior must raise a structured failure instead of being approximated silently.

---

## 4. Core architecture

Use four independent layers.

```text
Layer A: decode
  raw IMEM bytes -> DecodedInstruction

Layer B: CPU execution
  DecodedInstruction + FalconState -> state changes and bus requests

Layer C: device/bus model
  MMIO and XFER requests -> deterministic device effects

Layer D: trace and differential analysis
  CPU/device events -> normalized traces and first-divergence reports
```

### 4.1 Reuse the current fake

The current fake should become the shared device model used by both:

- the existing semantic Python path;
- the new instruction-level Falcon executor.

```text
semantic bootstrap implementation ----+
                                      |
                                      v
                              shared GK104 fake
                                      ^
                                      |
actual Falcon interpreter ------------+
```

Do not implement a second set of pause, XFER, BAR1-ordering or MEMIF rules in the oracle. Extract the existing rules behind a reusable adapter.

### 4.2 Execution modes

Support three modes:

1. **CPU-only mode**
   - Executes registers, flags, branches and DMEM behavior.
   - Stops at the first external MMIO or XFER operation.

2. **Scripted-device mode**
   - Uses deterministic YAML/JSON scenarios for MMIO and XFER completion.
   - Required for unit and regression tests.

3. **Existing-fake mode**
   - Connects to the project’s existing GK104 fake.
   - Required for semantic-versus-Falcon comparison.

A live-hardware backend is not required for the oracle MVP. Live traces are evidence and test fixtures, not the normal execution backend.

---

## 5. Repository layout

```text
tools/
└── falcon_oracle/
    ├── __init__.py
    ├── errors.py
    ├── values.py
    ├── state.py
    ├── instruction.py
    ├── decoder.py
    ├── executor.py
    ├── trace.py
    ├── diff.py
    ├── runner.py
    ├── coverage.py
    ├── manifests.py
    │
    ├── bus/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── scripted.py
    │   ├── sparse_memory.py
    │   └── gk104_fake_adapter.py
    │
    ├── devices/
    │   ├── __init__.py
    │   ├── pmu_output.py
    │   ├── falcon_xfer.py
    │   └── scripted_mmio.py
    │
    ├── corpus/
    │   ├── README.md
    │   ├── pmu_bar1_bootstrap/
    │   │   ├── image.bin
    │   │   ├── image.dis
    │   │   ├── image.manifest.json
    │   │   ├── symbols.json
    │   │   ├── initial_dmem.bin
    │   │   └── expected_external_trace.jsonl
    │   ├── memx/
    │   │   ├── image.bin
    │   │   ├── image.dis
    │   │   ├── image.manifest.json
    │   │   └── fixtures/
    │   └── fecs/
    │       └── README.md
    │
    └── tests/
        ├── test_values.py
        ├── test_memory.py
        ├── test_decoder.py
        ├── test_instruction_semantics.py
        ├── test_control_flow.py
        ├── test_scripted_bus.py
        ├── test_xfer.py
        ├── test_bootstrap_pad.py
        ├── test_nv_iowr_r0_clobber.py
        ├── test_memx.py
        ├── test_trace_diff.py
        └── fixtures/
```

Avoid putting the core interpreter inside `examples_kepler/add.py`. That file should import the oracle or adapters, not own them.

---

## 6. Data model

### 6.1 Falcon CPU state

```python
from dataclasses import dataclass, field


@dataclass
class FalconState:
    pc: int
    registers: list[int]

    carry: bool = False
    overflow: bool = False
    sign: bool = False
    zero: bool = False

    imem: bytearray = field(default_factory=bytearray)
    dmem: bytearray = field(default_factory=bytearray)

    instruction_count: int = 0
    logical_step: int = 0

    status: str = "running"
    fault_reason: str | None = None
```

Requirements:

- derive the actual register count from the selected Falcon ISA definition;
- mask all general-purpose register writes to 32 bits;
- make flags explicit;
- make PC units explicit;
- keep CPU state independent of device/MMIO state;
- include a maximum instruction count;
- support state snapshot and restore.

### 6.2 IMEM and DMEM

Use bounds-checked memory classes.

```python
class FalconMemory:
    def read_u8(self, offset: int) -> int: ...
    def read_u16(self, offset: int) -> int: ...
    def read_u32(self, offset: int) -> int: ...

    def write_u8(self, offset: int, value: int) -> None: ...
    def write_u16(self, offset: int, value: int) -> None: ...
    def write_u32(self, offset: int, value: int) -> None: ...
```

Define explicitly:

- byte order;
- alignment rules;
- valid address ranges;
- behavior on out-of-range access;
- whether fetch and data addresses are byte- or word-based.

Every access outside the configured region must fault.

### 6.3 Decoded instruction

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedInstruction:
    pc: int
    size: int
    raw: bytes

    mnemonic: str
    operands: tuple[object, ...]

    branch_target: int | None = None
    metadata: dict[str, object] | None = None
```

Decoder and executor must remain separate. The decoder must not modify CPU state.

### 6.4 Structured errors

```text
FalconOracleError
├── DecodeError
├── UnsupportedInstruction
├── InvalidInstructionLength
├── InvalidIMEMAccess
├── InvalidDMEMAccess
├── InvalidAlignment
├── InvalidRegister
├── InvalidBranchTarget
├── DeviceModelError
├── UnsupportedMMIO
├── UnsupportedXferMode
├── XferTimeout
├── WaitTimeout
└── DivergenceError
```

Each failure should contain:

- PC;
- raw instruction bytes when applicable;
- decoded mnemonic when available;
- relevant registers;
- last external event;
- source symbol if available.

---

## 7. Normalized event trace

Define one event format for the semantic path and Falcon path.

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventKind(Enum):
    INSTRUCTION = auto()
    REGISTER_WRITE = auto()
    FLAG_WRITE = auto()
    DMEM_READ = auto()
    DMEM_WRITE = auto()
    MMIO_READ = auto()
    MMIO_WRITE = auto()
    XFER_START = auto()
    XFER_COMPLETE = auto()
    WAIT_BEGIN = auto()
    WAIT_END = auto()
    CALL = auto()
    RETURN = auto()
    SLEEP = auto()
    WAKE = auto()
    HALT = auto()
    FAULT = auto()
    MARKER = auto()


@dataclass(frozen=True)
class OracleEvent:
    sequence: int
    kind: EventKind

    pc: int | None = None
    raw_instruction: bytes | None = None
    mnemonic: str | None = None

    address: int | None = None
    value: int | None = None
    size: int | None = None

    source_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 7.1 Trace levels

Support:

- `external`: MMIO, XFER, sleep/halt/fault and markers;
- `effects`: external events plus register/DMEM writes;
- `instructions`: every instruction and all effects.

Default to `external`.

### 7.2 Deterministic output

For identical inputs, these must be identical:

- event sequence;
- final CPU state;
- final DMEM;
- final sparse external memory;
- stop reason;
- trace hash.

Never include wall-clock timestamps in the canonical trace.

---

## 8. Firmware corpus and manifests

Every executable image must have a manifest.

```json
{
  "schema": 1,
  "name": "pmu_bar1_bootstrap",
  "architecture": "falcon",
  "engine": "pmu",
  "base_address": "0x00000b14",
  "entry_address": "0x00000b14",
  "image_size": 222,
  "sha256": "...",
  "source": "firmware/gk104/pmu_bar1_bootstrap.fuc",
  "expected_stop": "done",
  "required_features": [
    "dmem",
    "mmio",
    "xfer_direct_vram",
    "xdwait"
  ]
}
```

The corpus-generation tool must verify:

- source/header/Python binary equality where all forms exist;
- image SHA-256;
- image size;
- entry point inside the image;
- no overlap with reserved IMEM;
- exact final decoded address;
- no undecoded bytes.

### 8.1 Disassembly manifest

Convert trusted envydis output into a stable fixture containing:

- PC;
- raw bytes;
- instruction length;
- mnemonic;
- operand forms;
- branch target;
- optional source symbol.

Do not invoke envydis during every unit test. Generate and review the manifest once, then check it into the corpus.

### 8.2 Symbol map

When possible, create a source map:

```json
{
  "0x00000b14": {
    "symbol": "bootstrap_entry",
    "source": "pmu_bar1_bootstrap.fuc:1"
  },
  "0x00000b68": {
    "symbol": "xfer_instance",
    "source": "pmu_bar1_bootstrap.fuc:28"
  }
}
```

Source mapping is diagnostic metadata, not execution state.

---

## 9. Decoder implementation plan

### 9.1 Coverage-driven decoding

Do not implement the entire Falcon ISA first.

For the selected image:

1. disassemble the exact image with envydis;
2. enumerate unique instruction encoding forms;
3. implement those forms only;
4. fail on every unrecognized form;
5. add one decoder test per form;
6. require 100% byte and instruction-instance coverage before execution work proceeds.

Coverage report:

```text
pmu_bar1_bootstrap
  image bytes:                  222 / 222 decoded
  instruction instances:         N / N decoded
  unique encoding forms:         M / M supported
  unknown opcodes:                0
  overlapping instructions:       0
  trailing undecoded bytes:        0
```

`N` and `M` must be generated from the actual image.

### 9.2 Decoder acceptance tests

Required tests:

- every manifest instruction decodes to expected mnemonic and length;
- every branch target matches the manifest;
- decoding begins and ends on exact image boundaries;
- a one-bit mutation either decodes differently or fails;
- truncated instructions fail;
- strict decoding from a non-instruction boundary fails;
- the decoder never reads past IMEM.

---

## 10. CPU execution plan

### 10.1 Step function

```python
class FalconExecutor:
    def step(self) -> list[OracleEvent]:
        instruction = self.decoder.decode_one(
            self.state.imem,
            self.state.pc,
        )
        return self.execute_instruction(instruction)

    def run(
        self,
        *,
        max_instructions: int,
        breakpoints: set[int] | None = None,
    ) -> "ExecutionResult":
        ...
```

Each instruction handler must:

1. validate operands;
2. read input registers/memory;
3. compute the result;
4. update flags;
5. update destination state;
6. update the PC;
7. emit events.

### 10.2 Instruction unit tests

For every form, test:

- normal values;
- zero;
- all-ones;
- carry or overflow boundaries where relevant;
- source and destination aliasing;
- PC advancement;
- flags;
- immediate sign/zero extension;
- branch taken and not taken.

### 10.3 Control flow

Implement and test:

- unconditional branch;
- conditional branch;
- direct call;
- return;
- helper-call side effects;
- branch to first and last valid instruction;
- invalid branch target;
- self-loop with instruction cap;
- nested calls used by the corpus.

Do not substitute a hidden Python call stack for architectural return behavior unless the ISA explicitly uses one.

---

## 11. Shared bus interface

```python
from typing import Protocol


class FalconBus(Protocol):
    def mmio_read32(self, address: int) -> int:
        ...

    def mmio_write32(self, address: int, value: int) -> None:
        ...

    def xfer_start(self, request: "XferRequest") -> int:
        ...

    def xfer_poll(self, token: int) -> "XferStatus":
        ...

    def advance(self, logical_steps: int = 1) -> None:
        ...
```

The bus owns:

- device MMIO state;
- scheduled external events;
- sparse VRAM/sysmem state;
- XFER tokens and completion;
- device-specific assertions.

The CPU owns:

- registers;
- flags;
- PC;
- IMEM;
- DMEM.

---

## 12. Scripted device model

External hardware changes must be explicit.

```yaml
schema: 1
name: pause-rising-edge-success

initial_mmio:
  "0x000007c0": "0x00000000"

events:
  - trigger:
      kind: mmio_write
      address: "0x000007e0"
      value: "0x00000004"
    effect:
      after_steps: 5
      set_mmio:
        "0x000007c0": "0x00000004"

  - trigger:
      kind: mmio_write
      address: "0x000007e4"
      value: "0x00000004"
    effect:
      after_steps: 3
      set_mmio:
        "0x000007c0": "0x00000000"
```

Required scenarios:

- pause initially clear, then set;
- pause initially sticky-high;
- pause clears before a new rising edge;
- pause never sets;
- pause never clears;
- XFER completes immediately;
- XFER completes after N polls;
- XFER never completes;
- loadback bytes differ from stored bytes;
- unsupported MMIO access.

The scenario engine must be driven by logical steps or observed events, never wall-clock sleeps.

---

## 13. XFER engine model

### 13.1 Request model

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class XferRequest:
    source_space: str
    source_address: int

    destination_space: str
    destination_address: int

    size: int
    direction: str
    port: int
    target_mode: str
```

MVP modes:

- `dmem_to_direct_vram`;
- `direct_vram_to_dmem`.

Reject VM-mode requests initially.

### 13.2 Sparse external memory

```python
class SparseMemory:
    def read(self, address: int, size: int) -> bytes: ...
    def write(self, address: int, data: bytes) -> None: ...
```

Support explicit initialization patterns:

- zero;
- `0xff`;
- known fixture bytes;
- unmapped fault.

Do not guess uninitialized VRAM contents.

### 13.3 Transfer state

```text
CREATED -> PENDING -> COMPLETE

or

CREATED -> PENDING -> FAULT
```

`xdwait` must block until the relevant token reaches a terminal state.

Required assertions:

- one XFER cannot silently overwrite another pending transfer;
- transfer size is exact;
- source and destination ranges are bounds-checked;
- loadback uses bytes stored in modeled memory;
- completion becomes visible only after the scripted completion point.

---

## 14. Existing fake adapter

Extract the current fake’s relevant rules behind `GK104FakeBus`.

The adapter must preserve assertions including:

- root transfers only inside the permitted `ENTER`/`LEAVE` ownership region;
- store/load ordering;
- all loadback checks before BAR1 enable;
- first BAR1 access after the required activation sequence;
- no hidden device mutation during reads;
- strict rejection of unsupported operations.

Integration sequence:

1. identify fake methods used by the semantic bootstrap;
2. move shared state transitions into a reusable object;
3. keep compatibility wrappers so current tests remain unchanged;
4. add `FalconBus` methods that delegate to the shared object;
5. run all existing tests before adding interpreter behavior.

The first refactor must be behavior-preserving.

---

## 15. PMU bootstrap-pad MVP

### 15.1 Inputs

Fixture must include:

- exact IMEM bytes;
- actual entry point;
- exact initial DMEM layout;
- 16-byte instance-root fragment;
- 8-byte PDE fragment;
- 16-byte PTE fragment;
- initial MMIO values;
- scripted pause clear/set behavior;
- scripted XFER completion;
- sparse VRAM targets.

### 15.2 Required sequence

```text
ENTER request
-> wait for FB_PAUSE acknowledgement
-> XFER instance fragment
-> xdwait
-> XFER PDE fragment
-> xdwait
-> XFER PTE fragment
-> xdwait
-> LEAVE request
-> wait for FB_PAUSE clear
-> final status/DONE behavior
```

Where loadback occurs in Falcon firmware, include it in the instruction-level trace. Where verification is host-side, keep it in the semantic/fake comparison rather than attributing it to Falcon instructions.

### 15.3 Acceptance criteria

MVP passes only when:

- 100% of image bytes decode;
- every executed instruction is supported;
- no instruction fetch leaves the image;
- all three transfers have exact source, destination and size;
- each required `xdwait` follows the correct XFER;
- pause acknowledgement occurs before the first transfer;
- pause clear occurs before completion;
- final VRAM bytes equal the 40 expected bytes;
- final DMEM/status fields match the fixture;
- externally visible events match the semantic implementation;
- the canonical trace is deterministic.

---

## 16. Historical regression suite

### 16.1 `$r0` clobber

Create bad and fixed corpus variants.

Bad:

```text
loop counter in r0
-> call nv_iowr
-> r0 cleared
-> loop state corrupted
-> DONE not reached
```

Fixed:

```text
loop counter in non-clobbered register
-> all writes complete
-> DONE reached
```

Assertions:

- bad image shows `$r0` becoming zero at the helper;
- first divergence is reported at the helper/return boundary;
- bad image stops by instruction limit or modeled wait, not false success;
- fixed image reaches completion.

### 16.2 IMEM placement

Test an entry beyond configured live IMEM.

Expected:

```text
InvalidIMEMAccess
entry=<address>
image_end=<address>
```

### 16.3 Missing `xdwait`

Mutate the pad by removing or bypassing one wait. The fake/scripted bus should report:

- second XFER while first is pending;
- wrong token waited on;
- loadback before completion.

### 16.4 Sticky pause

Run a scenario where pause status is already high before `ENTER`. Distinguish an old sticky value from a real clear-to-set edge.

### 16.5 Transfer corruption

Inject one changed byte before loadback. Report physical destination, byte offset, expected value, actual value and initiating PC.

---

## 17. Semantic-versus-Falcon differential execution

Provide two paths:

```python
def execute_semantic_bootstrap(...) -> list[OracleEvent]:
    ...


def execute_falcon_bootstrap(...) -> list[OracleEvent]:
    ...
```

Normalize traces before comparison.

```python
def externally_visible(events: list[OracleEvent]) -> list[OracleEvent]:
    return [
        event
        for event in events
        if event.kind in {
            EventKind.MMIO_READ,
            EventKind.MMIO_WRITE,
            EventKind.XFER_START,
            EventKind.XFER_COMPLETE,
            EventKind.SLEEP,
            EventKind.WAKE,
            EventKind.HALT,
            EventKind.FAULT,
            EventKind.MARKER,
        }
    ]
```

### 17.1 Divergence classes

```text
MISSING_EVENT
EXTRA_EVENT
EVENT_KIND_MISMATCH
ADDRESS_MISMATCH
VALUE_MISMATCH
SIZE_MISMATCH
ORDER_MISMATCH
WAIT_CONDITION_MISMATCH
XFER_MODE_MISMATCH
BRANCH_MISMATCH
FINAL_STATE_MISMATCH
STOP_REASON_MISMATCH
```

### 17.2 First-divergence report

```text
FIRST DIVERGENCE

Corpus:
  pmu_bar1_bootstrap

Semantic event:
  XFER_START DMEM 0x0d90 -> VRAM 0x010000 size=8

Falcon event:
  MMIO_WRITE 0x001700 <- 0x00000003

Falcon state:
  pc=0x0b82
  instruction=...
  r0=0x00000000
  r5=0x00000001
  flags=...

Source:
  pmu_bar1_bootstrap.fuc:...
```

Use phase markers for resynchronization so one missing event does not make every later event appear unrelated.

---

## 18. `memx.fuc` phase

After the bootstrap pad passes, move to stock `memx.fuc`.

### 18.1 Required behavior

Verify:

- WR32 group parsing;
- serial WR32 behavior;
- WAIT semantics;
- DELAY semantics;
- `ENTER`;
- `LEAVE`;
- command-buffer termination;
- reply/completion behavior;
- failure on malformed command streams.

### 18.2 Fixture ladder

Implement in order:

1. empty/terminator-only script;
2. one WR32;
3. two WR32 commands;
4. one WR32 group;
5. one successful WAIT;
6. one WAIT timeout;
7. one DELAY;
8. `ENTER; LEAVE`;
9. `ENTER; WR32; LEAVE`;
10. complete atomic RAM transition;
11. former-clock transition script;
12. target-clock transition script;
13. two-pass 324 MHz to 648 MHz sequence.

### 18.3 Differential acceptance

For each fixture:

```text
semantic MEMX external trace
==
Falcon-executed memx.fuc external trace
```

Compare MMIO address, value, order, wait condition, completion point and final DMEM command/reply state.

### 18.4 Mutation tests

Reintroduce historical defects:

- wrong training-status address;
- wrong wait duration;
- target commands inside wrong ownership region;
- unchanged RAMFUC masks emitted;
- missing mode-selection masks;
- wrong timing-word rotation;
- wrong write order.

The oracle should identify the first firmware-visible divergence.

---

## 19. Optional FECS phase

Do not begin FECS until PMU pad and MEMX phases are stable.

### 19.1 FECS device model

Add:

- scratch and mailbox registers;
- command FIFO;
- interrupt enable/status/set/clear semantics;
- CHSW interrupt injection;
- channel/current-context state;
- VM XFER engine;
- sleep/wake behavior;
- command completion markers.

### 19.2 Initial fixtures

1. firmware entry to ready;
2. unsupported command and `E_BAD_COMMAND`;
3. `ctx_chan`;
4. `ctx_save`;
5. internal CHSW command `0x4001`;
6. one VM XFER success;
7. one VM XFER fault;
8. `xdwait` completion.

### 19.3 FECS acceptance boundary

The FECS phase is useful when it can answer:

- which instruction initiates a VM transfer;
- what MEM_BASE and MEM_TARGET were active;
- why `xdwait` did or did not complete;
- which command ID reached the main loop;
- which scratch/mailbox bit is expected at completion.

It need not simulate the whole GR engine.

---

## 20. CLI

```bash
# Verify corpus integrity and decode coverage
python -m tools.falcon_oracle.runner coverage \
  --corpus tools/falcon_oracle/corpus/pmu_bar1_bootstrap

# Disassemble with the local decoder
python -m tools.falcon_oracle.runner disasm \
  --image tools/falcon_oracle/corpus/pmu_bar1_bootstrap/image.bin \
  --base 0xb14

# Execute with a scripted bus
python -m tools.falcon_oracle.runner run \
  --corpus tools/falcon_oracle/corpus/pmu_bar1_bootstrap \
  --scenario tools/falcon_oracle/tests/fixtures/pause_xfer_success.yaml \
  --trace-level effects \
  --output logs/falcon-bootstrap.jsonl

# Compare with semantic execution
python -m tools.falcon_oracle.runner diff \
  --reference logs/semantic-bootstrap.jsonl \
  --candidate logs/falcon-bootstrap.jsonl \
  --external-only \
  --stop-at-first

# Run MEMX firmware
python -m tools.falcon_oracle.runner run-memx \
  --corpus tools/falcon_oracle/corpus/memx \
  --commands tools/falcon_oracle/corpus/memx/fixtures/enter-leave.bin \
  --scenario tools/falcon_oracle/tests/fixtures/memx_pause.yaml
```

Every command must return nonzero on corpus failure, unknown instruction, device-model failure, timeout, divergence or final-state mismatch.

---

## 21. Test strategy

### Level 0: pure data structures

- 32-bit normalization;
- memory bounds;
- event serialization;
- state snapshot/restore;
- manifest parsing.

### Level 1: decoder

- exact manifest match;
- every instruction form;
- malformed/truncated instruction;
- strict boundary behavior.

### Level 2: CPU execution

- ALU;
- flags;
- branches;
- calls/returns;
- DMEM accesses;
- instruction cap.

### Level 3: scripted bus

- MMIO read/write;
- scheduled events;
- wait success/timeout;
- XFER state machine;
- sparse memory.

### Level 4: bootstrap pad

- complete real image;
- success scenario;
- pause failures;
- XFER failures;
- historical mutations.

### Level 5: differential execution

- semantic and actual Falcon trace equivalence;
- first-divergence classification.

### Level 6: MEMX

- minimal commands;
- full streams;
- historical RAMFUC mutations.

### Level 7: optional FECS

- ready path;
- command paths;
- VM XFER.

### 21.1 Confidence labels

Each fixture should state its evidence level:

```text
SOURCE_ONLY
OFFLINE_EXACT
FAKE_EXACT
LIVE_OBSERVED
LIVE_BYTE_VERIFIED
```

### 21.2 CI gates

```bash
python3 -m py_compile tools/falcon_oracle/*.py
python3 -m pytest tools/falcon_oracle/tests
python3 examples_kepler/add.py --middle-selftest
NV_BACKEND=software python3 examples_kepler/add.py
git diff --check
```

Also require corpus hashes, 100% decode coverage for the current MVP corpus, deterministic trace hashes and no change to current fake behavior.

---

## 22. Determinism and reproducibility

Record:

```text
oracle version
Python version
corpus manifest hash
IMEM SHA-256
initial DMEM SHA-256
scenario SHA-256
initial device-state SHA-256
trace SHA-256
final CPU-state SHA-256
final DMEM SHA-256
final external-memory SHA-256
```

Canonical JSON must use sorted keys, fixed integer formatting, stable byte encoding and no wall-clock timestamps.

---

## 23. Performance targets

Correctness is primary, but target:

- bootstrap pad under 100 ms in instruction-trace mode;
- minimal MEMX fixture under 100 ms;
- full MEMX transition fixture under 1 second;
- no unbounded loops;
- no JIT until interpreter semantics are stable.

---

## 24. Development milestones

### Milestone 0 — Freeze interfaces

Deliverables:

- plan checked into repository;
- corpus paths;
- event schema;
- bus protocol;
- error taxonomy;
- MVP acceptance criteria.

Exit gate:

```text
No implementation begins until the event format and shared fake boundary are fixed.
```

### Milestone 1 — Shared fake extraction

Deliverables:

- reusable fake state object;
- `GK104FakeBus` adapter;
- semantic path emitting normalized events;
- compatibility wrappers.

Exit gate:

```text
All existing tests pass without behavior changes.
```

### Milestone 2 — Corpus and manifests

Deliverables:

- exact PMU pad binary;
- SHA-256 manifest;
- envydis instruction manifest;
- symbol map;
- initial DMEM fixture;
- expected external trace.

Exit gate:

```text
Source/header/Python/binary equality and image boundaries verified.
```

### Milestone 3 — Decoder

Deliverables:

- strict decoder;
- one test per encoding form;
- coverage report;
- malformed-image tests.

Exit gate:

```text
100% byte and instruction-instance decode coverage for the 0xde-byte pad.
```

### Milestone 4 — CPU core

Deliverables:

- registers and flags;
- PC and control flow;
- IMEM and DMEM;
- instruction subset;
- snapshots;
- instruction trace.

Exit gate:

```text
The pad executes deterministically until its first external operation.
```

### Milestone 5 — Scripted MMIO

Deliverables:

- scripted bus;
- logical scheduler;
- MMIO handlers;
- wait success/timeout;
- pause scenarios.

Exit gate:

```text
The pad passes through ENTER acknowledgement in the success scenario.
```

### Milestone 6 — XFER engine

Deliverables:

- direct-VRAM XFER;
- sparse memory;
- XFER tokens;
- `xdwait`;
- loadback;
- corruption and timeout scenarios.

Exit gate:

```text
The real pad produces three exact transfers and 40 correct destination bytes.
```

### Milestone 7 — Historical regressions

Deliverables:

- `$r0` clobber fixture;
- invalid-IMEM fixture;
- missing-`xdwait` fixture;
- sticky-pause fixture;
- transfer-corruption fixture.

Exit gate:

```text
Every historical defect fails at expected PC with a structured report.
```

### Milestone 8 — Differential oracle

Deliverables:

- semantic external trace;
- Falcon external trace;
- normalization;
- divergence classifier;
- first-divergence report.

Exit gate:

```text
Real PMU pad and semantic implementation produce identical external effects.
```

### Milestone 9 — MEMX firmware

Deliverables:

- MEMX corpus and manifest;
- remaining instruction forms;
- command-buffer model;
- fixture ladder;
- differential tests.

Exit gate:

```text
Actual memx.fuc and semantic MEMX agree on every supported fixture.
```

### Milestone 10 — Hardening

Deliverables:

- deterministic hashes;
- CI integration;
- documentation;
- corpus-generation script;
- stable CLI.

Exit gate:

```text
A clean checkout reproduces every MVP result without hardware.
```

### Milestone 11 — Optional FECS expansion

Deliverables:

- FECS corpus;
- interrupt/mailbox model;
- command FIFO;
- VM XFER;
- focused command fixtures.

Exit gate:

```text
The oracle explains one complete FECS command path and one VM-XFER wait path.
```

---

## 25. Recommended commit sequence

```text
Commit 1  Add plan, event schema, errors and manifests.
Commit 2  Extract shared fake state and add GK104FakeBus.
Commit 3  Add PMU pad corpus, hashes, disassembly and symbols.
Commit 4  Implement strict IMEM/DMEM and decoder skeleton.
Commit 5  Reach 100% decode coverage for PMU pad.
Commit 6  Implement CPU instruction subset and control flow.
Commit 7  Add scripted MMIO/event scheduler and ENTER/LEAVE waits.
Commit 8  Add direct-VRAM XFER, xdwait and sparse memory.
Commit 9  Execute real PMU pad end-to-end.
Commit 10 Add historical regression fixtures and divergence reports.
Commit 11 Add semantic-versus-Falcon differential test.
Commit 12 Add memx.fuc corpus and minimal command fixtures.
Commit 13+ Expand MEMX, harden CI, then consider FECS.
```

Each commit keeps `py_compile`, oracle tests, middle selftest, software demo and `git diff --check` green.

---

## 26. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Incorrect opcode semantics | Convincing but false traces | Compare decoder with envydis; test every form; fail closed |
| Duplicate fake/device rules | Both paths agree because of copied bugs | Share one device model; keep execution backends separate |
| Wall-clock dependency | Nondeterministic tests | Logical-step scheduler only |
| Unknown MMIO treated as zero | False branches and false success | Raise `UnsupportedMMIO` unless scenario defines it |
| Overbuilding full ISA | Long project before useful output | Coverage-driven subset from the `0xde` pad |
| Queued XFER treated as complete | Repeats prior debugging errors | Explicit token lifecycle and `xdwait` |
| Invalid padding executed | Hides bad entry/image bugs | Strict manifest and IMEM bounds |
| Analog hardware simulated deterministically | Wrong RAM/PLL conclusions | Script observations; do not predict analog behavior |
| FECS expands scope too early | MVP stalls | FECS only after PMU and MEMX acceptance |
| Fake refactor changes behavior | Existing evidence invalidated | Compatibility wrappers and behavior-preserving milestone |
| Corpus drift | Tests use different firmware silently | SHA-256 manifests and CI checks |

---

## 27. Definition of done

The MVP is complete when a clean checkout can:

1. verify the exact PMU bootstrap corpus;
2. decode 100% of its bytes;
3. execute it from the real entry PC;
4. model required MMIO and direct-VRAM XFER behavior;
5. reproduce the three-fragment, 40-byte external effect;
6. detect `$r0`, IMEM-boundary, sticky-pause and missing-`xdwait` failures;
7. produce a deterministic canonical trace;
8. compare that trace with the semantic implementation;
9. report first divergence with PC, instruction, registers and source symbol;
10. run all of this without live hardware.

The second release is complete when stock `memx.fuc` produces the same normalized external effects as the existing semantic MEMX implementation for the full fixture ladder.

FECS support is a later extension, not part of the MVP.

---

## 28. Immediate implementation checklist

```text
[ ] Create tools/falcon_oracle/
[ ] Define OracleEvent and canonical JSONL serialization
[ ] Define FalconBus protocol
[ ] Extract current fake into reusable state object
[ ] Add GK104FakeBus without changing behavior
[ ] Freeze the 0xde-byte PMU pad binary and SHA-256
[ ] Generate envydis instruction manifest
[ ] Generate symbol/source map
[ ] Add strict IMEM and DMEM
[ ] Implement decoder forms used by pad
[ ] Reach 100% decode coverage
[ ] Implement CPU state and instruction subset
[ ] Add logical-step scripted MMIO
[ ] Add pause clear/set scenarios
[ ] Add direct-VRAM XFER and xdwait
[ ] Execute real pad end-to-end
[ ] Add semantic-versus-Falcon comparison
[ ] Add historical mutation tests
[ ] Add deterministic trace hashes
[ ] Integrate into CI
[ ] Begin memx.fuc only after pad MVP passes
```
