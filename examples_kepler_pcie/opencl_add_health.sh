#!/bin/sh
set -eu

BDF=${KEPLER_PCI_BDF:-0000:09:00.0}
DEVICE=/sys/bus/pci/devices/$BDF
DRIVER=$(basename "$(readlink "$DEVICE/driver" 2>/dev/null || true)")

if [ "$DRIVER" != nouveau ]; then
  echo "GTX 770 health check requires $BDF bound to nouveau (current: ${DRIVER:-unbound})" >&2
  exit 1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd \
RUSTICL_ENABLE=nouveau \
exec "$SCRIPT_DIR/../examples_pcie/add_opencl"
