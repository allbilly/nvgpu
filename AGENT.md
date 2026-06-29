when in doubt, ask deepwiki first
use ~/nvgpu

## Goal

`python3 examples/add.py` prints `result=[11.0, 22.0, 33.0, 44.0]`, no
`tinygrad` runtime import on the live path. `examples/mul.py` likewise.

## Status (2026-06-23)

**GREEN.** `python3 examples/add.py` and `python3 examples/middle_nv.py`
are identical and both print `result=[11.0, 22.0, 33.0, 44.0]` on the
RTX 3080 eGPU (ADT-Link UT3G over USB4). `python3 examples/add_tiny.py`
remains the tinygrad health reference and also prints
`result=[11.0, 22.0, 33.0, 44.0]`.

The standalone NV stack (GSP boot, golden-image init, user-mode compute
channel, GPFIFO submit, semaphore release) is fully vendored in
`examples/middle_nv.py` (a single 2 k-line file). The cubin is hand-
assembled SM86 SASS (4× FADD, 2× LDG, 1× STG). Live-path imports are
limited to `from tinygrad.runtime.autogen import nv, nv_570, pci,
nv_regs, libc` (ctypes constants only) — `Device["NV"]`, `_load_tinygrad`,
and `from tinygrad.device/runtime/ops` are gone from the live path.

## Verify

```
# Tier 1 (offline, no eGPU required)
python3 examples/middle_nv.py --middle-selftest
#   -> middle_selftest=ok cubin_sha=54f9606... launch_words=20 rpc_checksum=0xc040404

# Tier 2 (eGPU required)
python3 examples/add.py
#   -> result=[11.0, 22.0, 33.0, 44.0]
python3 examples/add_tiny.py
#   -> result=[11.0, 22.0, 33.0, 44.0]

# Stage timing (optional)
NV_ADD_TRACE_STAGES=1 python3 examples/middle_nv.py
#   -> stage t=0.000s cubin built
#      stage t=3.873s device ready (boot+GSP+golden+user-channel+gpfifo)
#      stage t=3.888s program built (cubin uploaded to VRAM, NVProgram ready)
#      stage t=3.890s manual_launch done (kernel executed on eGPU, result on device)
#      stage t=3.890s copyout done (D2H)
#      stage t=3.890s final result decoded: [11.0, 22.0, 33.0, 44.0]
#   ~4 s wall time total; 3.87 s in GSP boot, ~2 ms in the add kernel.
```

## Health test

`python3 examples/add_tiny.py` should always print
`result=[11.0, 22.0, 33.0, 44.0]`. If it doesn't, the eGPU / GSP / tinygrad
stack is broken and the standalone path is also likely broken.

## Reference repos (deepwiki MCP)

- allbilly/ane: examples/
- allbilly/rk3588: examples/
- florianmattana/sass-king
- mikex86/LibreCuda
- cloudcores/CuAssembler
- vectorch-ai/ScaleLLM
- SzymonOzog/GPU_Programming
- gpuasm.com
- redplait/denvdis
- daadaada/turingas
- hkust-adsl/gass
- gpgpu-sim/gpgpu-sim_distribution
- Tim453/ClusterSim
- nvidia/open-gpu-kernel-modules

- https://blog.doubleword.ai/what-happens-when-you-run-a-cuda-kernel

Revealing NVIDIA Closed-Source Driver Command Streams for CPU–GPU Runtime Behavior Insight
- https://arxiv.org/html/2604.26889
