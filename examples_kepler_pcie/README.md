# examples_kepler_pcie

Linux raw-MMIO userspace driver for GTX 770 (Kepler GK104, sm_30) over PCIe.
Port of `examples_kepler/add.py` (macOS TinyGPU socket transport) to direct
BAR0/BAR1 mmap via sysfs (`/sys/bus/pci/devices/<bdf>/resourceN`).

## Running the OpenCL add (reference working path)

A minimal OpenCL vector add is at `../examples_pcie/add_opencl.c`. It runs on
the RTX 3080 Ti (Ampere GA102) via the NVIDIA proprietary ICD and serves as the
**known-good reference** for what a successful compute kernel launch looks like.

### Prerequisites

```sh
# GCC + OpenCL ICD loader + NVIDIA OpenCL runtime (already on this box)
sudo apt install gcc ocl-icd-opencl-dev mesa-opencl-icd
# nvidia-opencl-dev + libnvidia-opencl.so.1 come from the NVIDIA driver package
```

Verify the ICD is visible:
```sh
cat /etc/OpenCL/vendors/nvidia.icd   # -> libnvidia-opencl.so.1
ls /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
```

### Build and run

```sh
cd examples_pcie
gcc -O2 -o add_opencl add_opencl.c -lOpenCL
./add_opencl
```

Expected output:
```
device: NVIDIA GeForce RTX 3080 Ti
first 5: 0 3 6 9 12
PASS
```

The `pci id for fd ... driver (null)` lines are Mesa's rusticl ICD probing the
fd before the NVIDIA ICD wins device selection — cosmetic, not an error.

### Capturing a trace

```sh
# strace: ioctl sequence to /dev/nvidiactl + /dev/nvidia0
strace -f -v -e trace=ioctl -x -o /tmp/opencl_ioctl_trace.txt ./add_opencl

# nsys: OS runtime + timeline profile
nsys profile --trace=osrt --output=/tmp/opencl_add_nsys ./add_opencl
nsys stats /tmp/opencl_add_nsys.nsys-rep --report osrt_sum
```

The strace captures 559 ioctls (type `0x46` = NVIDIA driver). These are the
proprietary RM ioctls, not raw MMIO — they show the high-level sequence
(context create, memory alloc, kernel launch, fence) but not the register
writes that `add.py` needs to replicate.

### Getting a Kepler-specific MMIO trace via nouveau

The most useful trace for fixing `add.py` is from **nouveau** running on the
GTX 770 itself, since nouveau does the same raw MMIO bring-up. To capture it:

```sh
# 1. Install GK104 firmware where nouveau expects it
sudo mkdir -p /lib/firmware/nvidia/gk104/gr
sudo cp firmware/gk104/gk104_fecs_code.bin  /lib/firmware/nvidia/gk104/gr/fecs_inst.bin
sudo cp firmware/gk104/gk104_fecs_data.bin  /lib/firmware/nvidia/gk104/gr/fecs_data.bin
sudo cp firmware/gk104/gk104_gpccs_code.bin /lib/firmware/nvidia/gk104/gr/gpccs_inst.bin
sudo cp firmware/gk104/gk104_gpccs_data.bin /lib/firmware/nvidia/gk104/gr/gpccs_data.bin

# 2. Load nouveau with debug tracing
sudo modprobe nouveau debug=TRACE

# 3. Bind nouveau to the GTX 770 (09:00.0)
echo "10de 1184" | sudo tee /sys/bus/pci/drivers/nouveau/new_id
# or: echo 0000:09:00.0 | sudo tee /sys/bus/pci/drivers/nouveau/bind

# 4. Run OpenCL via rusticl on the GTX 770
OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd ./add_opencl

# 5. Capture the MMIO trace from dmesg
sudo dmesg | grep nouveau > /tmp/nouveau_gk104_trace.txt
```

The nouveau trace shows the exact register write sequence for Kepler GR
context generation, FIFO channel setup, and compute launch — the sequence
`add.py` is trying to replicate.

## Running the low-level add.py

### Offline selftest (no hardware/root needed)

```sh
PYTHONPATH=../ref:$PYTHONPATH NV_BACKEND=software python3 add.py --middle-selftest
```

### Live probe (root + KEPLER_LIVE_ACK required)

```sh
# Probe: read PMC_BOOT_0 chip ID
sudo KEPLER_LIVE_ACK=raw-mmio-risk PYTHONPATH=../ref:$PYTHONPATH python3 add.py --probe

# FECS falcon bring-up
sudo KEPLER_LIVE_ACK=raw-mmio-risk PYTHONPATH=../ref:$PYTHONPATH python3 add.py --probe-falcon

# Full hardware add (needs cubin + vbios + firmware)
sudo KEPLER_LIVE_ACK=raw-mmio-risk \
     KEPLER_CUBIN=../examples_kepler/add_kepler.cubin \
     KEPLER_VBIOS=../examples_kepler/Palit.GTX770.4096.131216.rom \
     PYTHONPATH=../ref:$PYTHONPATH python3 add.py
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `NV_BACKEND` | `kepler` | `software` for offline, `kepler` for hardware |
| `KEPLER_LIVE_ACK` | (unset) | Must be `raw-mmio-risk` to allow live MMIO |
| `KEPLER_CUBIN` | (auto) | Path to sm_30 cubin |
| `KEPLER_VBIOS` | `Palit.GTX770.4096.131216.rom` | VBIOS ROM for devinit |
| `NV_PCIBDF` | (auto-detect) | PCI BDF of the GK104 |
| `NV_FIRMWARE_DIR` | `../firmware/gk104` | Path to falcon firmware |
| `KEPLER_TEST_STAGE` | `full-add` | `sem`, `set-object`, or `full-add` |
| `KEPLER_SUBMIT_MODE` | `gpfifo` | `gpfifo`, `bypass`, or `dispatch` |
| `DEBUG` | `0` | Verbose register traces |

## Current blocker

FECS posts ready, FIFO channel is set up, GPFIFO is consumed (GP_GET=1), but
the compute kernel does NOT execute — the completion semaphore stays at its
initial value. Even an immediate-exit kernel stalls. The GR golden context
image is generated via `_gk104_grctx_main()` but the compute launch path
(SET_OBJECT → LAUNCH → semaphore) does not produce a semaphore update.

See `../examples_kepler/progress.md` for the full bring-up history.
