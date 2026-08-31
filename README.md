# nvgpu

Running simple add and multiply CUDA kernels on an RTX 3080 eGPU over USB4/PCIe
or USB 3.1 with hand-written Python — no `tinygrad` runtime import but with some
autogen stuffs.

## Quick start 

`examples/add.py` automatically uses a supported Chestnut USB device when one
is connected; otherwise it uses the TinyGPU/PCI transport. Set `NV_IFACE=USB`
or `NV_IFACE=PCI` to select a transport explicitly.

### USB4/PCIe

On macOS, install TinyGPU from https://docs.tinygrad.org/tinygpu/

```bash
python3 examples/add.py
#   tested with adt ut3g 3080
#   result=[11.0, 22.0, 33.0, 44.0]

python3 examples/add.py mul
#   result=[10.0, 40.0, 90.0, 160.0]
```

### USB3.1

On macOS, tinyGPU not required, but ensure the usb cable is at least usb3.1 10G speed

```bash
NV_IFACE=USB python3 examples/add.py
#   tested with chestnut USB3 tinygrad/asm2464pd-firmware@ed4e39b7e0794e19ba193477067c48757a5cf9ef
#   works on m1 mac usb3.1, failed on orangepi5 usb3.0, because of speed assuming the gsp firmware uploading, could be solved by firmware phased upload time matching
#   result=[11.0, 22.0, 33.0, 44.0]
```

## cuda tools on macos

macOS cannot run NVIDIA CUDA tools natively. Use Docker for the tools; no
GPU passthrough is needed for `nvcc`, `ptxas`, or `nvdisasm`.

To compare our hand-built cubin against `nvcc`-generated output, dump the
cubin from inside Python:

```bash
python3 -c "import examples.add as a; open('add.cubin','wb').write(a.build_cubin())"
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  nvdisasm add.cubin
```

## llm

```bash
echo "1+1=" | DEV=NV python3 -m tinygrad.llm
```

# Kepler
examples_kepler_pcie/add.py
- Linux port of examples_kepler/add.py: macOS TinyGPU.app socket transport
  replaced by LinuxPCIDevice (raw MMIO via sysfs resourceN mmap). Reuses
  nvbios_init / pgraph_mmio_gk104 from examples_kepler/ via sys.path insert.
- live path WORKING: hardware_demo=ok N=256 mismatches=0/256 (2026-07-15).
  KEPLER_LIVE_ACK gating is macOS-only; Linux only needs root (auto-sudo).
- `--middle-selftest` and `NV_BACKEND=software` pass offline (no hardware/root).
- live `--probe` works: reads PMC_BOOT_0=0x0e4040a2 (GK104) from 09:00.0.
- live add op needs: root (sudo), KEPLER_CUBIN=../examples_kepler/add_kepler.cubin,
  KEPLER_VBIOS=../examples_kepler/Palit.GTX770.4096.131216.rom, and
  ref/linux/ (torvalds/linux v7.2-rc2 sparse-checkout of
  drivers/gpu/drm/nouveau/nvkm/engine/gr) for grctx_gk104.py to parse csdata.
- VBIOS devinit executes, GPC PLL locks, FECS posts ready, ctx_chan works,
  golden context saves, full add kernel runs with correct results.
- KEPLER_SKIP_LTC=1 skips hot-path LTC invalidate calls (H26: Nouveau never
  calls them on desktop GK104). Safe up to N=524288 (32 windows) but hangs
  at N=1048576 (64 windows) due to cache state accumulation.
- See examples_kepler_pcie/progress.md for Linux-specific bring-up history.
