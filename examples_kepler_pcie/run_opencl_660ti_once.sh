#!/bin/sh
# OpenCL add on GTX 660 Ti (09:00.0) via Nouveau+Rusticl.
# Prerequisites: iommu=pt; RTX 3080 Ti already on nvidia; do NOT load nouveau early at boot.
# Never rmmod/unbind Kepler afterward (nve0_bo_move_copy oops). Never let nouveau claim 04:00.0.
set -eu

BDF=${KEPLER_PCI_BDF:-0000:09:00.0}
KEEP=${KEPLER_KEEP_NVIDIA_BDF:-0000:04:00.0}
LOG=${KEPLER_OPENCL_LOG:-/home/a/nvgpu/examples_kepler_pcie/last_opencl_health.log}
DEVICE=/sys/bus/pci/devices/$BDF
KEEP_DEV=/sys/bus/pci/devices/$KEEP

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo sh $0" >&2
  exit 1
fi

# ponytail: tee so interactive + file both get the result
exec > >(tee "$LOG") 2>&1
echo "=== $(date -Is) kepler opencl ==="
echo "cmdline: $(cat /proc/cmdline)"

if ! grep -Eq '(^| )iommu=pt( |$)' /proc/cmdline; then
  echo "FAIL: boot with iommu=pt (amd_iommu=pt alone is ignored on this kernel)"
  exit 1
fi

keep_drv=$(basename "$(readlink "$KEEP_DEV/driver" 2>/dev/null || true)")
if [ "$keep_drv" != nvidia ]; then
  echo "FAIL: $KEEP must already be bound to nvidia (now: ${keep_drv:-unbound})"
  echo "      loading nouveau first lets it steal the 3080 Ti"
  exit 1
fi

if ! lsmod | grep -q '^nouveau'; then
  # Explicit modeset=2 overrides nvidia-installer's modeset=0.
  modprobe nouveau modeset=2 runpm=0
fi
echo "modeset=$(cat /sys/module/nouveau/parameters/modeset)"

# If nouveau stole the 3080 Ti, abort (do not unbind here — recover with reboot).
keep_drv=$(basename "$(readlink "$KEEP_DEV/driver" 2>/dev/null || true)")
if [ "$keep_drv" != nvidia ]; then
  echo "FAIL: nouveau claimed $KEEP (driver=${keep_drv:-unbound}); reboot to recover"
  exit 1
fi

if [ ! -e "$DEVICE/driver" ]; then
  echo "10de 1183" > /sys/bus/pci/drivers/nouveau/new_id 2>/dev/null || true
  echo "$BDF" > /sys/bus/pci/drivers/nouveau/bind
  sleep 2
fi

DRIVER=$(basename "$(readlink "$DEVICE/driver" 2>/dev/null || true)")
echo "kepler_driver=$DRIVER keep_driver=$(basename "$(readlink "$KEEP_DEV/driver")")"
if [ "$DRIVER" != nouveau ]; then
  echo "FAIL: $BDF not on nouveau"
  exit 1
fi

# Strip seat ACL on any Kepler KMS node
for n in /dev/dri/card* /dev/dri/by-path/pci-0000:09:00.0-card; do
  [ -e "$n" ] || continue
  case $(readlink -f "$n" 2>/dev/null) in
    *) ;;
  esac
  # only touch nodes whose sysfs device is 09:00.0
  dev=$(cd /sys/class/drm/$(basename "$n")/device 2>/dev/null && pwd -P) || continue
  case "$dev" in
    */0000:09:00.0) setfacl -b "$n" 2>/dev/null || true; chmod 600 "$n"; chown root:root "$n" ;;
  esac
done

echo "--- OpenCL ---"
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu
export OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd
export RUSTICL_ENABLE=nouveau
set +e
timeout --signal=KILL 30s /home/a/nvgpu/examples_pcie/add_opencl
rc=$?
set -e
echo "rc=$rc"
dmesg | grep -iE "nouveau.*fault|construct context|AMD-Vi.*fault|killed!" | tail -20 || true
exit "$rc"
