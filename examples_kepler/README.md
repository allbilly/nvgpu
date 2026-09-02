# Kepler GK104 examples

`add_770.py` and `add_660ti.py` are the board-specific entry points. The GTX
770 path contains the userspace Kepler
RM path: PCI transport, VBIOS/devinit, clock and GDDR5 setup, PMU plus
FECS/GPCCS Falcon loading, GMMU/FIFO/GR context creation, and compute submit.
Kepler has no GSP; only S5 is skipped for that reason.

## Usage

```sh
# Offline, no GPU access
python3 -S examples_kepler/add_770.py --offline-selftest
python3 -S examples_kepler/add_770.py --trace-selftest
python3 -S examples_kepler/add_660ti.py --middle-selftest
python3 -S examples_kepler/add_660ti.py --trace-selftest
NV_BACKEND=software python3 -S examples_kepler/add_770.py add
NV_BACKEND=software python3 -S examples_kepler/add_770.py mul

# Live, only while the intended GK104 eGPU is connected
python3 examples_kepler/add_770.py
python3 examples_kepler/add_770.py mul
python3 examples_kepler/add_660ti.py
python3 examples_kepler/add_660ti.py mul

# Transport overrides (AUTO prefers direct Chestnut USB3 for either board)
KEPLER_IFACE=USB python3 examples_kepler/add_770.py
KEPLER_IFACE=SOCKET python3 examples_kepler/add_770.py mul
KEPLER_IFACE=USB python3 examples_kepler/add_660ti.py
KEPLER_IFACE=SOCKET python3 examples_kepler/add_660ti.py mul
```

`--middle-selftest` remains an alias for `--offline-selftest` in the GTX 770
entry point. Multiplication is selected by passing `mul` to either board entry
point, matching the EN210 interface.

The GTX 770 macOS path prefers direct Chestnut USB3 and falls back to
TinyGPU.app's Unix socket; its Linux path uses raw PCI sysfs. `KEPLER_IFACE` accepts
`AUTO`, `USB`, or `SOCKET`, and `KEPLER_USBDEV=vendor:product` overrides the
default Chestnut IDs. Direct USB uses a CPU-only staging arena: all GPU-visible
instance, runlist, GPFIFO, push, semaphore, page-table, context, cubin, and data
objects are copied and remapped into physical VRAM before submission.

The GTX 660 Ti entry point uses the same macOS selection contract: direct
Chestnut USB3 when present, with TinyGPU.app's socket as the fallback. Both
board entry points accept `KEPLER_USBDEV=VID:PID` for a non-default controller.

## S1-S10 trace

1. PCI transport and BAR mapping
2. GK104 topology, PRAMIN, and GMMU preparation
3. VBIOS devinit plus clock, power, and memory initialization
4. legacy PMU/FECS/GPCCS Falcon loading and boot
5. GSP/RM (N/A: no GSP; the selected board entry point is the userspace RM)
6. golden and runtime GR context construction
7. RAMIN, USERD, channel, runlist, and GPFIFO setup
8. cubin, CWD, constants, and push-buffer preparation
9. runlist commit, GP_PUT, and completion wait
10. output validation

The default `KEPLER_TRACE=1` reports the semantic bring-up checkpoints, wait
outcomes, phase timings, and S1-S10 lifecycle using the same visual hierarchy
as `examples/add.py`: numbered subsections, indented child details, and
caller-attributed `CALL`, `RETURN`, `CTX`, and wait records. It streams while
the GPU is running, including during the long S6 context build. Set
`KEPLER_TRACE=2` for changed register reads plus every register write,
annotated with stage, transport phase, and caller. Set `KEPLER_TRACE=0` to hide
the stage and semantic trace. `--trace-selftest` renders and verifies the
format without probing PCI or USB.

Live traces also report Chestnut LTSSM/retrain state, each PCIe hop's maximum
and negotiated speed/width, direct-USB phase durations, and the GDDR5 policy
plus controller/partition training status. These observations do not trigger a
PCIe retrain or change the RAM initialization policy. Set `KEPLER_PCIE_TRACE=0`,
`KEPLER_PHASE_TRACE=0`, `KEPLER_HW_TRACE=0`, or `KEPLER_MEM_TRACE=0` to hide
the corresponding detail. The S1-S10 summary includes elapsed time for each
completed stage.

## Layout

- top level: `add_770.py`, `add_660ti.py`, and this README
- `runtime/`: live helper modules, cubins, VBIOS images, golden fixture, and Falcon sources/headers
- `diagnostics/`: benchmark, VRAM, and mmiotrace checks
- `tools/`: CUDA/PTX sources, disassembly, generators, and setup scripts
- `docs/`: bring-up history, plans, and reset notes

The raw historical `nouveau_gk104_mmiotrace.txt.gz` is optional. When absent,
the offline gate uses the checked-in equivalent 1,636-write JSON fixture and
omits only the gzip-to-JSON duplication checks.
