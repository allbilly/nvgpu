#!/bin/sh
# Capture mmiotrace around a successful NVK Vulkan add on GTX 660 Ti.
# Must run only after 3080 Ti is on nvidia. Prefer post-reboot clean state.
# Usage: sudo sh examples_kepler_pcie/capture_nvk_mmiotrace.sh
set -eu

BDF=${KEPLER_PCI_BDF:-0000:09:00.0}
KEEP=${KEPLER_KEEP_NVIDIA_BDF:-0000:04:00.0}
OUT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RAW=${KEPLER_MMIO_RAW:-/tmp/mmiotrace_660ti_nvk_pass.txt}
GZ=$OUT_DIR/mmiotrace_660ti_nvk_pass.txt.gz
BIN=$OUT_DIR/vk_add_compute
SPV=$OUT_DIR/add.spv
TRACE=/sys/kernel/debug/tracing
LOG=$OUT_DIR/last_nvk_mmiotrace.log

exec >"$LOG" 2>&1
echo "=== $(date -Is) capture_nvk_mmiotrace ==="

if [ "$(id -u)" -ne 0 ]; then
  echo "need root" >&2
  exit 1
fi

# Wait for 3080 on nvidia (avoid nouveau stealing it at early boot).
i=0
while [ "$i" -lt 90 ]; do
  keep=$(basename "$(readlink /sys/bus/pci/devices/$KEEP/driver 2>/dev/null || true)")
  if [ "$keep" = nvidia ]; then
    break
  fi
  sleep 1
  i=$((i + 1))
done
keep=$(basename "$(readlink /sys/bus/pci/devices/$KEEP/driver 2>/dev/null || true)")
if [ "$keep" != nvidia ]; then
  echo "FAIL: $KEEP not on nvidia after wait (now=${keep:-unbound})"
  exit 2
fi
echo "keep_driver=nvidia"

if [ ! -x "$BIN" ] || [ ! -f "$SPV" ]; then
  echo "FAIL: build vk_add_compute + add.spv first"
  exit 1
fi

mount -t debugfs debugfs /sys/kernel/debug 2>/dev/null || true
if [ ! -d "$TRACE" ]; then
  echo "FAIL: no $TRACE"
  exit 1
fi

# Stop any prior tracer.
echo 0 > "$TRACE/tracing_on" 2>/dev/null || true
echo nop > "$TRACE/current_tracer" 2>/dev/null || true
# Large buffer (same order as prior 660 Ti capture).
echo 262144 > "$TRACE/buffer_size_kb" 2>/dev/null || echo 131072 > "$TRACE/buffer_size_kb" || true
echo "buffer_size_kb=$(cat $TRACE/buffer_size_kb)"

echo mmiotrace > "$TRACE/current_tracer"
echo 1 > "$TRACE/tracing_on"
# Drain pipe in background.
rm -f "$RAW"
: > "$RAW"
cat "$TRACE/trace_pipe" >> "$RAW" &
CATPID=$!
sleep 1

# Bind Nouveau headless if needed.
if ! lsmod | grep -q '^nouveau'; then
  modprobe nouveau modeset=2 runpm=0
fi
if [ ! -e /sys/bus/pci/devices/$BDF/driver ]; then
  echo "10de 1183" > /sys/bus/pci/drivers/nouveau/new_id 2>/dev/null || true
  echo "$BDF" > /sys/bus/pci/drivers/nouveau/bind
  sleep 2
fi
echo "09=$(basename $(readlink /sys/bus/pci/devices/$BDF/driver))"
keep=$(basename "$(readlink /sys/bus/pci/devices/$KEEP/driver)")
if [ "$keep" != nvidia ]; then
  echo "FAIL: nouveau stole $KEEP"
  echo 0 > "$TRACE/tracing_on" || true
  kill "$CATPID" 2>/dev/null || true
  exit 2
fi

echo "=== NVK add under mmiotrace ==="
set +e
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nouveau_icd.json \
  timeout --signal=KILL 30s "$BIN" "$SPV"
rc=$?
set -e
echo "vk_rc=$rc"

# Stop trace cleanly.
sleep 1
echo 0 > "$TRACE/tracing_on"
echo nop > "$TRACE/current_tracer"
sleep 1
kill "$CATPID" 2>/dev/null || true
wait "$CATPID" 2>/dev/null || true

lines=$(wc -l < "$RAW" | tr -d ' ')
echo "raw_lines=$lines raw_bytes=$(wc -c < "$RAW" | tr -d ' ')"
if [ "$rc" -ne 0 ]; then
  echo "FAIL: NVK add did not PASS; leaving raw at $RAW (not promoting to golden)"
  gzip -kf -9 "$RAW" || true
  exit "$rc"
fi

gzip -9 -c "$RAW" > "$GZ"
sha=$(sha256sum "$GZ" | awk '{print $1}')
echo "wrote $GZ sha256=$sha lines=$lines"
# Also keep a short nouveau dmesg companion.
dmesg | grep -i nouveau | tail -200 > "$OUT_DIR/nvk_add_dmesg_tail.txt" || true
echo "PASS capture complete"
exit 0
