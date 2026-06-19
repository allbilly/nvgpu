# nvgpu

running a simple add cuda kerenl on rtx3080 egpu with ADT-ut3g ASM2464PD
with simple python, not even tinygrad

# examples

Run the no-hardware validation suite for the hand-built add cubin and standalone backend:

```bash
python3 examples/add.py --validation-suite
python3 examples/mul.py --validation-suite
```

Before a live eGPU run, use the strict transport preflight gate. It stops before GSP/RM work if TinyGPU or PCIe visibility is not ready:

```bash
NV_ADD_TRANSPORT=mac-egpu python3 examples/add.py --transport-preflight-plan --require-ready
```

When the preflight reports `ready-for-gsp`, run the hand-built add cubin through the local example helper:

```bash
python3 examples/add.py
# result=[11.0, 22.0, 33.0, 44.0]
```

The default live run enables GSP boot plus golden-context preparation. Set `NV_ADD_BOOT_GSP=0` and
`NV_ADD_PREPARE_GOLDEN_CTX=0` only when intentionally testing the narrower non-booted path.

# cuda tools on macos

macOS cannot run NVIDIA CUDA tools natively. Use Docker only for the tools; no GPU passthrough is needed for `nvcc`, `ptxas`, or `nvdisasm`.

Generate the cubin from our Python byte builder:

```bash
python3 -c 'import examples.add as a; open("add.cubin", "wb").write(a.build_cubin())'
```

Disassemble it with CUDA tools in Docker:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work nvidia/cuda:12.4.1-devel-ubuntu22.04 nvdisasm add.cubin
```

The first run pulls a large CUDA image and can take a while.

Normal CUDA flow for comparison:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash -lc 'nvcc -arch=sm_86 -cubin add.cu -o add_nvcc.cubin && nvdisasm add_nvcc.cubin'
```

# tinygrad llm
echo "1+1=" | DEV=NV python3 -m tinygrad.llm
