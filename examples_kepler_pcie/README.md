# examples_kepler_pcie

Linux raw-MMIO userspace driver for Kepler GK104 cards (GTX 660 Ti/770,
sm_30) over PCIe.  The PCI scanner accepts both `10de:1183` (660 Ti) and
`10de:1184` (770), plus the other desktop GK104 IDs.
Port of `examples_kepler/add.py` (macOS TinyGPU socket transport) to direct
BAR0/BAR1 mmap via sysfs (`/sys/bus/pci/devices/<bdf>/resourceN`).

## Kepler simple-add status (this box)

| Path | GTX 660 Ti (`10de:1183`) | Notes |
|------|--------------------------|--------|
| Rusticl OpenCL (`nouveau`) | FAIL | `HOST0` PDE @ `0x11000` even headless + `iommu=pt` |
| EGL/GL compute (`compute_add_egl`) | FAIL | PGRAPH context `-16` / GR idle timeout |
| **NVK Vulkan compute** (`vk_add_compute`) | **works if GPU not wedged** | Confirmed PASS once; fails after raw-MMIO / bad Nouveau state until **reboot** |
| Raw `add.py --probe` / `--probe-falcon` | OK | Chip ID + FECS/GPCCS start |
| Raw `add.py` full add | FAIL | GR ctx page did not stabilise |

**GTX 770** previously had a reliable Rusticl OpenCL PASS on this machine; that remains the OpenCL health baseline when that card is installed.

Boot with `iommu=pt` (this kernel ignores `amd_iommu=pt`).  Keep default target `multi-user` (headless) while testing Kepler.  Bind Nouveau only after the 3080 Ti is on `nvidia`.  Do not `rmmod` Kepler (`nve0_bo_move_copy` oops).  PCI FLR alone often **does not** clear a wedged GK104 — use a reboot.

## How to stably run simple add on GTX 660 Ti (NVK)

**Use NVK Vulkan compute, not OpenCL.**  Instability we hit was almost always **GPU channel/GR wedge** from a prior Nouveau OpenCL/GL/`add.py` session — not a random NVK bug on a clean card.

### Prerequisites (once per boot)

1. Kernel cmdline includes `iommu=pt`.
2. Headless: `systemctl get-default` → `multi-user.target` (no GNOME on boot-VGA Kepler).
3. `0000:04:00.0` (3080 Ti) already bound to **nvidia**.
4. `0000:09:00.0` (660 Ti) **unbound** at boot (nouveau blacklisted by nvidia-installer is fine).

### Build (once)

```sh
cd examples_kepler_pcie
sudo apt install -y libvulkan-dev glslang-tools
glslangValidator -V -S comp -o add.spv add.comp
gcc -O2 -o vk_add_compute vk_add_compute.c -lvulkan
```

### Run (clean Nouveau → NVK add)

```sh
# Only after 3080 is on nvidia — never load nouveau earlier in the boot.
sudo modprobe nouveau modeset=2 runpm=0
# If not auto-bound:
#   echo 10de 1183 | sudo tee /sys/bus/pci/drivers/nouveau/new_id
#   echo 0000:09:00.0 | sudo tee /sys/bus/pci/drivers/nouveau/bind

sudo sh ./nvk_add_health.sh
# or manually:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nouveau_icd.json ./vk_add_compute ./add.spv
```

Expected:

```
physdev[0]: GeForce GTX 660 Ti (NVK GK104) vendor=0x10de device=0x1183
using: GeForce GTX 660 Ti (NVK GK104)
first 5: 0 3 6 9 12
PASS
```

### Stability rules

| Do | Don't |
|----|--------|
| Run NVK add on a **fresh** Nouveau bind after boot | Run OpenCL / `compute_add_egl` / raw `add.py` **before** NVK on the same bind |
| Keep headless | Let GNOME/Xorg open Kepler `card*` |
| If submit returns `-4` / channel kill / job timeout → **reboot**, then bind+NVK again | Expect PCI `reset`/unbind alone to heal a wedged GK104 |
| Leave Nouveau bound after a PASS (optional) | `rmmod nouveau` (oops risk) |

`nvk_add_health.sh` will try one FLR+rebind retry; if that still fails, reboot is required.

### Why it looked “flaky”

1. First PASS was on a clean headless Nouveau bind.
2. Later `add.py` raw MMIO + unbind/rebind left GR/FIFO wedged (`PGRAPH` busy, PDE/PTE faults).
3. Retests then saw all-zero `FAIL` or `VK_ERROR_DEVICE_LOST` — same silicon, bad driver state.
4. Barriers/fence in `vk_add_compute.c` help host↔shader sync; they do **not** unwedge a dead channel.

## Running the OpenCL add (reference / 770 health)

A minimal OpenCL vector add is at `../examples_pcie/add_opencl.c`. On the
RTX 3080 Ti it uses the NVIDIA ICD.  For Kepler, use the wrapper below with
Rusticl after binding `09:00.0` to Nouveau:

```sh
./opencl_add_health.sh
```

Expected on a **GTX 770** (prior PASS):

```
device: NVE4
first 5: 0 3 6 9 12
PASS
```

**GTX 660 Ti** OpenCL does **not** PASS today (PDE fault).  See `progress.md`.

```sh
# Build OpenCL binary if needed
cd ../examples_pcie && gcc -O2 -o add_opencl add_opencl.c -lOpenCL
```

The wrapper restricts ICDs to Rusticl and puts distro `libOpenCL` before CUDA's
bundled copy.  `pci id for fd ... driver (null)` lines from Mesa probing the
3080 DRM node are cosmetic.

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
