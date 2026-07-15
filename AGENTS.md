- DO NOT attempt to remove files outside pwd
- DO NOT run commands or search that is too long
- working examples/add.py already but for another gpu model

- if the repo is not in ref/, ask deepwiki on ref repo, else read it locally
- allbilly/nvgpu : this repo
- envytools/envytools
- allbilly/linux_drm : Nouveau driver
- intel-lgci-fdo-gitlab-mirror/mesa.mesa
- daadaada/turingas
- xiuxiazhang/KeplerAs
- allbilly/amdgpu4
- TheTom/pascal-egpu

reclocking
- drivers/gpu/drm/nouveau/nvkm/subdev/clk/gk104.c

hardware (this box, Linux x86_64)
- RTX 3080 Ti at 04:00.0 (bound to nvidia 595.71.05, /dev/nvidia0)
- GTX 770 (GK104) at 09:00.0 — UNBOUND, no driver. nvidia 595 dropped Kepler (last
  supporting branch is 470 legacy), so it cannot bind to the proprietary driver.
- GK104 falcon firmware (fecs/gpccs/pmu) lives in firmware/gk104/ (extracted via
  firmware/gk104/extract_fw.py). NOT in /lib/firmware. nouveau expects them at
  /lib/firmware/nvidia/gk104/gr/{fecs_inst,fecs_data,gpccs_inst,gpccs_data}.bin
- no passwordless sudo; user is in the sudo group (sudo commands need a password)

examples_kepler_pcie/add.py
- Linux port of examples_kepler/add.py: macOS TinyGPU.app socket transport
  replaced by LinuxPCIDevice (raw MMIO via sysfs resourceN mmap). Reuses
  nvbios_init / pgraph_mmio_gk104 from examples_kepler/ via sys.path insert.
- live path still KEPLER-TODO skeleton, gated behind KEPLER_LIVE_ACK=raw-mmio-risk.
- `--middle-selftest` and `NV_BACKEND=software` pass offline (no hardware/root).
- live `--probe` works: reads PMC_BOOT_0=0x0e4040a2 (GK104) from 09:00.0.
- live add op needs: root (sudo), KEPLER_CUBIN=../examples_kepler/add_kepler.cubin,
  KEPLER_VBIOS=../examples_kepler/Palit.GTX770.4096.131216.rom, and
  ref/linux/ (torvalds/linux v7.2-rc2 sparse-checkout of
  drivers/gpu/drm/nouveau/nvkm/engine/gr) for grctx_gk104.py to parse csdata.
- FECS firmware loads+starts but does NOT post ready (0x409800 bit31) — GPC
  clock/power domain gated; same blocker as examples_kepler/ (see progress.md).

ref/linux
- sparse-checkout of torvalds/linux v7.2-rc2 at drivers/gpu/drm/nouveau/nvkm/
  engine/gr (needed by examples_kepler/grctx_gk104.py at import time).
- clone: git clone --depth 1 --filter=blob:none --sparse --branch v7.2-rc2
  https://github.com/torvalds/linux ref/linux && cd ref/linux &&
  git sparse-checkout set drivers/gpu/drm/nouveau/nvkm/engine/gr
