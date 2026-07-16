#!/bin/sh
set -eu

BDF=${KEPLER_PCI_BDF:-0000:09:00.0}
DEVICE=/sys/bus/pci/devices/$BDF
DRIVER=$(basename "$(readlink "$DEVICE/driver" 2>/dev/null || true)")

if [ "$DRIVER" != nouveau ]; then
  echo "Kepler OpenCL health check requires $BDF bound to nouveau (current: ${DRIVER:-unbound})" >&2
  exit 1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Kepler GART DMA needs IOMMU pass-through.  On this AMD host only iommu=pt
# counts (amd_iommu=pt is ignored: "AMD-Vi: Unknown option - 'pt'").
if [ -r /proc/cmdline ] && ! grep -Eq '(^| )iommu=pt( |$)' /proc/cmdline; then
  echo "warning: need iommu=pt on cmdline; Kepler DMA may fault" >&2
fi
# 660 Ti (10de:1183): even headless + iommu=pt, Rusticl channel setup hits
# HOST0 PDE @ 0x11000 (known fail vs prior GTX 770 PASS).  See progress.md.
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd \
RUSTICL_ENABLE=nouveau \
exec timeout --signal=KILL 30s "$SCRIPT_DIR/../examples_pcie/add_opencl"
