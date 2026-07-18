 # x86rom_py/plan.md — GK104 Legacy Option-ROM Producer Investigation

  ## Summary

  Build a dependency-free, trace-guided Python executor for the Palit GTX 770 legacy x86 option ROM. Its initial purpose is
  narrow: identify and reproduce the pre-Nouveau operation that changes physical PRAMIN from 0xffffffff/0xbad0fbxx into usable
  VRAM.

  This implementation is justified by Night41l–o:

  - RAM training succeeds when cold reclocking is skipped.
  - PMU MEMIF can store and reload VRAM exactly.
  - PRAMIN remains one persistent PBUS stub throughout Python’s runtime initialization.
  - Golden Nouveau consumes working PRAMIN before its first initialization write.
  - Reproducing 0x619f04, 0x1700, and PCI shadow state is insufficient.
  - Every other readable pre-init MMIO register already matches.
  - The remaining producer is therefore likely option-ROM/platform activity involving PCI configuration, legacy I/O, write-only
    MMIO, ordering, or hidden state.

  Do not port the remaining Nouveau runtime driver line by line as the next step. Its BAR/instmem bootstrap already depends on
  working PRAMIN. Do not build a general-purpose x86 emulator. Implement only the Palit ROM path needed to reach the first
  live-PRAMIN boundary.

  Canonical input:

  - ROM container: examples_kepler/Palit.GTX770.4096.131216.rom
  - Container SHA-256: d574f587406c107e963c702a7980c4773226d4815d86afc77f9a60d96df7d02d
  - Legacy image: offset 0x600, size 0xf600
  - Legacy image SHA-256: fe64395721e8acf52474d5473d1fbde8727466669741b2900f9bc939cab55e33
  - Legacy checksum: zero modulo 256
  - Entry: C000:0003; startup reaches image offset 0x50, then 0x2caa
  - EFI x86-64 image at 0xfc00: out of scope

  ## Architecture

  ### ROM and execution model

  Implement these modules under x86rom_py/:

  - rom.py
      - Parse the container and PCI expansion-ROM structures.
      - Select the legacy image structurally, never through an ambiguous 55 aa scan.
      - Validate hashes, checksum, PCIR pointer, code type, NVIDIA vendor/device identity, image length, and final-image
        indicator.

      - Expose immutable RomImage metadata and bytes.

  - cpu.py
      - Implement an interpreter for dynamically reached 16-bit real-mode instructions.
      - Model 32-bit general registers, segment registers, IP/EIP, EFLAGS, prefixes, ModR/M, stack behavior, string operations,
        near/far control flow, interrupts, and IN/OUT.

      - Preserve exact carry, overflow, auxiliary-carry, parity, sign, zero, direction, and interrupt flags.
      - Add opcodes only when reached by the pinned ROM.
      - Abort with CS:IP, bytes, and register state on every unsupported opcode.
      - Do not manually translate ROM routines into ad hoc Python functions.

  - memory.py
      - Provide conventional memory, IVT, BIOS Data Area, stack, scratch memory, and read-only ROM mapping at 0xc0000.
      - Model real-mode address wrapping and controlled 32-bit physical accesses.
      - Translate emulated PCI BAR physical ranges into bus accesses.
      - Abort on unsupported physical apertures or writes to ROM.

  - firmware.py
      - Define a versioned FirmwareProfile containing entry registers, IVT/BDA state, PCI topology, BAR assignments, and
        deterministic BIOS-service responses.

      - Implement only BIOS interrupts reached by this ROM.
      - Route PCI BIOS INT 1Ah/B1xx operations through the bus.
      - Implement reached timing, equipment, video, and memory services.
      - Abort on unknown interrupt functions instead of inventing successful responses.

  ### Three-space hardware bus

  Define a common bus protocol in bus.py:

  pci_read(offset, width) -> int
  pci_write(offset, width, value) -> None
  io_read(port, width) -> int
  io_write(port, width, value) -> None
  mmio_read(bar, offset, width) -> int
  mmio_write(bar, offset, width, value) -> None
  delay_us(duration) -> None
  checkpoint(name) -> ProbeResult

  Provide:

  - ModelBus: deterministic register/config model for unit tests.
  - ReplayBus: consumes expected reads and checks emitted operations against a golden three-space trace.
  - RecordingBus: records all activity from another bus.
  - LiveBus: wraps APLRemotePCIDevice or LinuxPCIDevice without duplicating their transports.

  Routing rules:

  - Emulate PCI mechanism #1 accesses through 0xcf8/0xcfc; never touch host chipset ports directly.
  - Resolve guest BAR addresses dynamically to BDF + BAR + offset.
  - Route proven NVIDIA VGA ports through byte-width BAR0 aliases at 0x601000 + port.
  - Do not assume all legacy ports have a BAR0 equivalent. An unclassified port aborts live execution.
  - Preserve transaction width, ordering, repeated reads, delays, and posting reads exactly.
  - Never combine adjacent byte writes into wider transactions.

  ### Trace and producer slicing

  Use append-only JSONL events in trace.py:

  sequence
  timestamp_ns
  phase
  cs
  ip
  instruction_bytes
  operation
  address_space
  direction
  width
  bdf
  bar
  raw_address
  canonical_offset
  value
  read_result
  firmware_service
  posted_flush
  dependency_tag

  Implement:

  - Exact trace comparison with first-divergence reporting.
  - Explicit declarations for volatile read fields and poll-loop repetition.
  - No implicit width, address, or ordering normalization.
  - Cross-run normalization of assigned BAR addresses while retaining raw addresses.
  - Backward slicing from the first successful PRAMIN checkpoint.
  - Reports separating stable producer operations, polling, volatile reads, address-allocation differences, and genuine
    control-flow divergence.

  ## Evidence and Hypotheses

   Hypothesis                                     Status                          Proof or disproof
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   H70: pre-runtime firmware creates hidden       Leading                         Find the first external operation after
   PBUS/PRAMIN state                                                              which PRAMIN becomes real.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   H71: a bounded option-ROM producer prefix      Open                            Replay the complete captured prefix on one
   can reproduce that state                                                       fresh arm64 replug.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   H73: the missing producer is PCI config,       Leading                         Capture all three spaces in one ordered
   legacy I/O, write-only MMIO, or internal                                       timeline.
   state absent from mmiotrace
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-1: the known VGA preamble is sufficient    Mostly disproven                Retain as a fixture, but require an
                                                                                  immediate PRAMIN checkpoint after it.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-2: environment-dependent wrapper code      Primary open                    Compare full ROM external effects with
   performs missing operations around NVINIT                                      existing NvbiosInit effects.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-3: PCI BIOS responses select a branch      Open                            Replay recorded x86 firmware responses and
   absent from the Python path                                                    locate the first branch divergence.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-4: platform firmware, rather than the      Open                            Compare ROM execution with and without the
   card ROM, owns the producer                                                    platform’s enumeration/bridge prefix.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-5: exact write width, posting read, or     Open                            Preserve the trace exactly, then vary one
   timing creates hidden state                                                    property per later cold experiment.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-6: the ROM needs operations unavailable    Open                            Abort and classify the first unsupported
   through TinyGPU                                                                port, BAR remap, bridge, reset, or unsafe
                                                                                  write.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-7: the ROM returns while PRAMIN remains    Open                            With CPU and firmware conformance proven,
   stubbed                                                                        attribute ownership to platform firmware
                                                                                  outside the ROM.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-8: a general x86 emulator is required      Deprioritized                   Promote only if the exact ROM path cannot be
                                                                                  represented with reached instructions and
                                                                                  recorded services.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-9: corrected Nouveau-compatible VBIOS      Confirmed for NVINIT POST        Night41s changed fixed-PA PRAMIN from
   POST activates physical PRAMIN                                                  virgin to real data before RAM; later
                                                                                  runtime stages remain downstream.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-10: 0x619f04 and PCI config 0x50 are       Disproven                       Night41n reproduced both and retained the
   sufficient                                                                     stub.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-11: another readable pre-init MMIO         Disproven                       Night41o exhausted that set.
   register explains the difference
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-12: BAR roots, placement, flush,           Closed/deprioritized            H54, H56, H57, and H58 already isolate
   invalidation, or encoding are the current                                      failure below the BAR walk.
   blocker
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-13: RAM training or retention is still     Closed at boot clock            Night41b–d produced zero train nibbles and
   the current blocker                                                            exact PMU loadback.
  ─────────────────────────────────────────────  ──────────────────────────────  ──────────────────────────────────────────────
   X86-14: cold reclocking should remain in       Disproven                       It changes valid train status from zero to
   initialization                                                                 0xa; keep disabled.

  ## Implementation Roadmap

  ### Phase 1 — Structured ROM and static census

  1. Implement structured ROM extraction and pinned identity checks.
  2. Produce a static disassembly/census using installed LLVM tooling.
  3. Record entry-point control flow and all statically visible IN, OUT, interrupt, far-call, PCI mechanism, and physical-
     memory instructions.

  4. Generate a machine-readable reached/unknown opcode ledger.
  5. Verify the known 0x50 → 0x2caa path and direct VGA operations around image offset 0x2f0a.

  Exit gate: the ROM image and entry path are deterministic, and no EFI image can be selected accidentally.

  ### Phase 2 — Three-space x86 oracle

  1. POST the exact GTX 770 on an x86 firmware host, preferably as the primary legacy VGA adapter.
  2. Verify the posted baseline has stable non-stub PRAMIN before Nouveau initialization.
  3. Capture PCI config, every legacy port access, and BAR MMIO in one timeline.
  4. Preferred software capture is instrumented QEMU TCG/SeaBIOS with the physical card assigned through VFIO and direct VFIO
     mmap acceleration disabled.

  5. Capture CPU CS:IP and instruction bytes for each device transaction and BIOS call.
  6. Take three genuine cold captures.
  7. If instrumented VFIO cannot cover an address space, use a PCIe protocol analyzer or firmware/QEMU instrumentation for that
     space; do not declare the trace complete with a blind region.

  Exit gate: three captures agree after documented normalization and identify the first stable non-sentinel PRAMIN observation.

  ### Phase 3 — Trace replay before CPU live execution

  1. Implement trace parsing, canonicalization, comparison, and backward slicing.
  2. Start from the first live-PRAMIN event and retain every causal predecessor:
      - Writes.
      - Branch-controlling reads.
      - PCI/BAR setup.
      - Bridge VGA routing.
      - Posting reads.
      - Required delays and poll completion.

  3. Keep the first producer prefix complete; do not minimize it before it works.
  4. Implement read-only replay preview.
  5. Implement the prefix against ModelBus, then ReplayBus.
  6. Classify every operation by address space, width, safety, and expected consequence.

  Exit gate: replay matches every stable oracle event and explains every permitted variance.

  ### Phase 4 — Minimal real-mode executor

  1. Implement CPU instructions in reached order, with one fixture per opcode/form.
  2. Implement firmware services only when execution reaches them.
  3. Run until the next unsupported opcode/service, add its semantics and tests, and repeat.
  4. Reach and reproduce the known VGA preamble event-for-event.
  5. Continue until ROM return, first live-PRAMIN checkpoint, or an explicitly classified platform-only boundary.
  6. Compare embedded NVINIT operations against NvbiosInit.
  7. Report wrapper operations separately; these are the primary candidates for H70/H73.
  8. Keep QEMU/LLVM as development oracles only; Python runtime remains standard-library-only.

  Exit gate: the Python execution matches the x86 oracle through the producer boundary, including reads and control flow.

  ### Phase 5 — Guarded live experiment

  Add a standalone CLI:

  python3 -m x86rom_py inspect
  python3 -m x86rom_py analyze
  python3 -m x86rom_py compare TRACE_A TRACE_B
  python3 -m x86rom_py replay GOLDEN_TRACE
  python3 -m x86rom_py live \
    --backend tinygpu \
    --live-ack x86rom-producer-risk \
    --trace-out PATH \
    --stop-at pramin-live

  Live execution requirements:

  1. Exact 10de:1184, subsystem identity, ROM hashes, straps, topology, and expected BAR layout.
  2. Fresh enclosure power cycle and untouched cold-entry snapshot.
  3. GPU and audio function unbound from competing drivers.
  4. Mandatory output trace opened before the first write.
  5. Explicit instruction, operation, and wall-time budgets.
  6. One attempt only; no automatic reset, retry, fallback, or alternate branch.
  7. Abort on unsupported operation, entry mismatch, BAR0 loss, unexpected read, trace divergence, or unsafe write.
  8. Stop immediately when PRAMIN first becomes stable and non-stub.
  9. Close the transport without reset.
  10. Verify preservation from a second read-only process using the existing golden-preinit probe.

  Do not proceed into BAR1, VM, FIFO, GR, compute, or reclocking during this experiment.

  ### Phase 6 — Integration decision

  If PRAMIN becomes live:

  1. Confirm PRAMIN bytes match PMU physical bytes at the same VRAM address.
  2. Confirm all four RAM-training nibbles remain zero.
  3. Preserve the proven producer prefix as a standalone initialization phase.
  4. Continue with the existing Nouveau-derived BAR2/BAR1 path without repeating cold RAM initialization.
  5. Run compute at the boot memory clock.
  6. Address reclocking separately only after hardware_demo=ok.

  If the complete conformant ROM returns with PRAMIN still stubbed:

  1. Record this as a successful negative result.
  2. Disprove the option ROM alone as sufficient.
  3. Move the producer boundary to platform firmware, bridge setup, or reset sequencing.
  4. Reuse the trace tooling to capture that earlier prefix.
  5. Do not expand the runtime Nouveau port or generalize the x86 emulator.

  ## Safety Policy

  Implement safety.py with these mandatory rules:

  - Read-only operation is the default.
  - Live writes require the exact acknowledgement token and pinned identities.
  - Reject known TinyGPU-hostile writes to 0x1620 and 0x26f0; report the ROM location rather than executing them.
  - Reject reset, power, clock, RAM-training, bus-master/DMA, BAR remap, BAR1, VM, FIFO, and GR operations unless the producer
    trace proves they are required and a later reviewed policy explicitly allows them.

  - Never claim rollback for write-only, edge-triggered, read-to-clear, or hidden state.
  - Preserve exact access width.
  - Preserve posting-read and delay boundaries.
  - Check PMC_BOOT_0 and endpoint identity at every major checkpoint.
  - Never continue after BAR0 becomes all ones.
  - Never automatically retry after a timeout or transport failure.
  - Treat a warm PCI reset as different from an enclosure power cycle.
  - Keep bus mastering disabled unless the oracle proves it necessary and the device is IOMMU-contained.
  - Do not modify the current examples_kepler runtime path until standalone PRAMIN activation succeeds.

  ## Tests and Acceptance Criteria

  ### Offline tests

  - Validate both ROM hashes, offsets, lengths, PCIR metadata, checksum, and malformed-image rejection.
  - Test flags, segment/address wrapping, operand/address-size prefixes, ModR/M, stack wrapping, far calls/returns, interrupts,
    REP strings, and all reached I/O widths.

  - Differentially compare instruction fixtures with QEMU.
  - Test every reached BIOS service, including PCI success/error flag behavior.
  - Verify 0xcf8/0xcfc config emulation and dynamic BAR translation.
  - Verify VGA port accesses remain byte-width transactions.
  - Detect missing, extra, reordered, resized, or value-divergent trace events.
  - Reproduce entry control flow and the known VGA preamble.
  - Differentially compare embedded NVINIT effects with NvbiosInit.
  - Test all stop reasons:
      - pramin-live
      - rom-return
      - unsupported-opcode
      - unsupported-firmware-service
      - unsafe-operation
      - trace-divergence
      - instruction-budget
      - operation-budget
      - wall-time-budget
      - loop-detected
      - bar0-lost

  - Keep the existing middle selftest, software demo, compile checks, and all mmiotrace checkpoints green.

  ### Live success

  Success requires all of the following:

  - One fresh-replug attempt.
  - Exact device and ROM identity.
  - Complete trace persisted.
  - PRAMIN becomes stable and does not match 0xffffffff or the advancing 0xbad0fbxx class.
  - PRAMIN data equals PMU physical data at the same address.
  - PMC_BOOT_0 remains valid.
  - All four training nibbles remain zero.
  - A separate read-only process confirms PRAMIN is still live.
  - No runtime Nouveau initialization was needed to produce the transition.

  A conformant rom-return with stubbed PRAMIN is also a valid result: it disproves the ROM-alone hypothesis and prevents
  further work in the wrong layer.

  ## Assumptions

  - Python standard library only.
  - LLVM and QEMU are development/test oracles, not runtime dependencies.
  - Scope is the pinned Palit legacy ROM and its producer prefix.
  - EFI and unrelated option ROMs are excluded.
  - Existing PCI config/MMIO transports and NvbiosInit are reused.
  - TinyGPU cannot perform arbitrary host legacy I/O; only trace-proven BAR0 aliases may be used.
  - The golden three-space trace is the behavioral specification.
  - Reclocking remains disabled until compute works at the boot clock.
  - The next cold hardware run occurs only after a complete trace yields a bounded sequence and a precise predicted PRAMIN
    transition.
