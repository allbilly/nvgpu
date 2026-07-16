#!/bin/sh
# Stable-ish GTX 660 Ti simple add via NVK (not OpenCL).
# Requires: iommu=pt, 3080 on nvidia, headless preferred, root for bind/FLR.
# Usage: sudo sh examples_kepler_pcie/nvk_add_health.sh
set -eu

BDF=${KEPLER_PCI_BDF:-0000:09:00.0}
KEEP=${KEPLER_KEEP_NVIDIA_BDF:-0000:04:00.0}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BIN=$SCRIPT_DIR/vk_add_compute
SPV=$SCRIPT_DIR/add.spv
DEVICE=/sys/bus/pci/devices/$BDF
KEEP_DEV=/sys/bus/pci/devices/$KEEP

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo sh $0" >&2
  exit 1
fi

if ! grep -Eq '(^| )iommu=pt( |$)' /proc/cmdline; then
  echo "warning: boot with iommu=pt (this AMD kernel ignores amd_iommu=pt)" >&2
fi

keep=$(basename "$(readlink "$KEEP_DEV/driver" 2>/dev/null || true)")
if [ "$keep" != nvidia ]; then
  echo "FAIL: $KEEP must be on nvidia first (now: ${keep:-unbound})" >&2
  exit 1
fi

if [ ! -x "$BIN" ] || [ ! -f "$SPV" ]; then
  echo "build first: glslangValidator -V -S comp -o add.spv add.comp && gcc -O2 -o vk_add_compute vk_add_compute.c -lvulkan" >&2
  exit 1
fi

# FLR + rebind clears wedged GR/channel state without a full reboot (usually).
flr_rebind() {
  echo "FLR+rebind $BDF ..."
  if [ -e "$DEVICE/driver" ]; then
    timeout --signal=KILL 5s sh -c "echo $BDF > $DEVICE/driver/unbind" || true
  fi
  sleep 0.5
  if [ -e "$DEVICE/reset" ]; then
    echo 1 > "$DEVICE/reset" || true
    sleep 1
  fi
  if ! lsmod | grep -q '^nouveau'; then
    modprobe nouveau modeset=2 runpm=0
  fi
  echo "10de 1183" > /sys/bus/pci/drivers/nouveau/new_id 2>/dev/null || true
  if [ ! -e "$DEVICE/driver" ]; then
    echo "$BDF" > /sys/bus/pci/drivers/nouveau/bind
  fi
  sleep 2
  keep=$(basename "$(readlink "$KEEP_DEV/driver" 2>/dev/null || true)")
  if [ "$keep" != nvidia ]; then
    echo "FAIL: nouveau stole $KEEP — reboot to recover; do not continue" >&2
    exit 2
  fi
}

drv=$(basename "$(readlink "$DEVICE/driver" 2>/dev/null || true)")
if [ "$drv" != nouveau ]; then
  flr_rebind
fi

run_once() {
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nouveau_icd.json \
    timeout --signal=KILL 30s "$BIN" "$SPV"
}

echo "=== NVK add attempt 1 ==="
set +e
run_once
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "stable_path=ok (no FLR needed)"
  exit 0
fi

echo "attempt 1 failed (rc=$rc); recovering with FLR+rebind"
flr_rebind
echo "=== NVK add attempt 2 (after FLR) ==="
set +e
run_once
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "stable_path=FLR+rebind then vk_add_compute"
  exit 0
fi

echo "FAIL: NVK add still broken after FLR (rc=$rc)." >&2
echo "GPU may need a full reboot before Nouveau/NVK is healthy again." >&2
exit "$rc"
