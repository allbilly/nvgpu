# nvgpu

Running a simple add CUDA kernel on RTX 3080 eGPU (ADT-ut3g ASM2464PD) with
hand-written Python — no `tinygrad` runtime import on the live path.

## What "not even tinygrad" means

| Layer | Source |
|-------|--------|
| GSP boot, RM RPC, channels, GPFIFO, cubin launch | **`examples/middle_nv.py`** = `examples/add.py` (userspace stack reimplemented in Python, ~2 k lines, fully vendored) |
| PCIe config, BAR map, MMIO, sysmem FD (macOS) | **TinyGPU.app** — Apple-signed helper; required unless you disable SIP and provide another driver path |
| GSP RM, FECS, PMU, compute firmware | **NVIDIA** (on-GPU; unchanged) |

The live path (`main()`) imports only `from tinygrad.runtime.autogen import
nv, nv_570, pci, nv_regs, libc` — ctypes constants only. `Device["NV"]`,
`_load_tinygrad`, and `from tinygrad.device/runtime/ops` are gone from the
live path. The cubin is hand-assembled SM86 SASS (4× FADD, 2× LDG, 1× STG),
embedded in `middle_nv.py` and shipped with the file.

On Linux the hardware boundary is the open NVIDIA KMD (`/dev/nvidiactl`,
BAR mmap) instead of TinyGPU. We are **not** porting the Linux kernel
driver to macOS.

## Quick start

```bash
# Tier 1 (offline, no eGPU required) — verifies the cubin and the
# launch-words builder without touching the GPU.
python3 examples/middle_nv.py --middle-selftest
#   middle_selftest=ok cubin_sha=54f9606... launch_words=20 rpc_checksum=0xc040404

# Tier 2 (eGPU required) — boots the GSP, runs the kernel, prints the result.
python3 examples/add.py
#   cubin_bytes=2856 expected_result=[11.0, 22.0, 33.0, 44.0]
#   user compute_gpfifo ring_size=65536 token=0x1
#   device=NV iface=PCIIface
#   submit #manual: NVComputeQueue words=12
#   ...
#   result=[11.0, 22.0, 33.0, 44.0]
```

`examples/add.py` and `examples/middle_nv.py` are byte-identical
(`diff -q` returns no output). The `add.py` name is kept for compatibility
with the original tinygrad-based example that lived at that path; the
canonical standalone file is `middle_nv.py`.

## Stage timing (real-time)

```
NV_ADD_TRACE_STAGES=1 python3 examples/middle_nv.py
#   stage t= 0.000s  cubin built
#   stage t= 3.873s  device ready (boot+GSP+golden+user-channel+gpfifo)
#   stage t= 3.877s  3 input buffers allocated (sysmem via BAR1 mmap)
#   stage t= 3.877s  3 copyin done (H2D)
#   stage t= 3.888s  program built (cubin uploaded to VRAM, NVProgram ready)
#   stage t= 3.890s  manual_launch done (kernel executed on eGPU)
#   stage t= 3.890s  copyout done (D2H)
#   stage t= 3.890s  final result decoded: [11.0, 22.0, 33.0, 44.0]
#   ~4 s wall time total; 3.87 s in GSP boot, ~2 ms in the add kernel.
```

## Health test

```bash
python3 examples/add_tiny.py   # tinygrad path — always green; if not, the eGPU is broken
```

## Files

```
nvgpu/
├── AGENT.md                 # agent context, status, verify recipe
├── README.md                # this file
├── TODO.md                  # milestone log (now closed)
├── add.cu                   # (removed — cubin is embedded in middle_nv.py)
├── add.cubin                # (removed — see above)
├── dump/                    # (removed)
├── experimental/            # (removed)
├── examples/
│   ├── add.py              # the standalone live path (== middle_nv.py)
│   ├── middle_nv.py        # vendored NV stack + cubin builder + live driver
│   ├── add_tiny.py         # frozen tinygrad health reference
│   └── mul.py              # multiply kernel (separate; same NV stack)
├── firmware/ga102           # GSP firmware blobs (loaded by middle_nv)
└── ref/                    # tinygrad source for reference
```

## cuda tools on macos

macOS cannot run NVIDIA CUDA tools natively. Use Docker for the tools; no
GPU passthrough is needed for `nvcc`, `ptxas`, or `nvdisasm`.

To compare our hand-built cubin against `nvcc`-generated output, dump the
cubin from inside Python:

```bash
python3 -c "import examples.middle_nv as a; open('add.cubin','wb').write(a.build_cubin())"
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  nvdisasm add.cubin
```

## tinygrad llm (reference only)

```bash
echo "1+1=" | DEV=NV python3 -m tinygrad.llm
```
