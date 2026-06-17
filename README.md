# nvgpu

running a simple add cuda kerenl on rtx3080 egpu with ADT-ut3g ASM2464PD
with simple python, not even tinygrad

# examples

Run the hand-built add cubin through the local example helper:

```bash
python3 examples/add.py
# result=[11.0, 22.0, 33.0, 44.0]
```

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
