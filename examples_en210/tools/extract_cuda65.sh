#!/usr/bin/env bash
# Extract ptxas/nvcc/nvdisasm/cuobjdump + nvvm + open64 from the CUDA 6.5 runfile.
# Run this once before setup_ptxas_6_5_osx.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNFILE="${1:-$SCRIPT_DIR/cuda_6.5.14_linux_64.run}"

if [ ! -f "$RUNFILE" ]; then
  echo "Downloading CUDA 6.5.14 runfile to $RUNFILE..."
  curl -L -o "$RUNFILE" \
    "https://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run"
fi

chmod +x "$RUNFILE"

TMPDIR_EXTRACT="$(mktemp -d)"
TMPDIR_UNPACK="$(mktemp -d)"
cleanup() { rm -rf "$TMPDIR_EXTRACT" "$TMPDIR_UNPACK"; }
trap cleanup EXIT

echo "Extracting outer runfile..."
"$RUNFILE" --extract="$TMPDIR_EXTRACT" --silent

echo "Unpacking inner toolkit runfile..."
sh "$TMPDIR_EXTRACT/cuda-linux64-rel-6.5.14-18749181.run" \
  --noexec --keep --target "$TMPDIR_UNPACK"

echo "Copying binaries to $SCRIPT_DIR/cuda65-bin/..."
mkdir -p "$SCRIPT_DIR/cuda65-bin"
cp "$TMPDIR_UNPACK/bin/ptxas"      "$SCRIPT_DIR/cuda65-bin/"
cp "$TMPDIR_UNPACK/bin/nvcc"       "$SCRIPT_DIR/cuda65-bin/"
cp "$TMPDIR_UNPACK/bin/nvcc.profile" "$SCRIPT_DIR/cuda65-bin/"
cp "$TMPDIR_UNPACK/bin/nvdisasm"   "$SCRIPT_DIR/cuda65-bin/"
cp "$TMPDIR_UNPACK/bin/cuobjdump"  "$SCRIPT_DIR/cuda65-bin/"
cp -r "$TMPDIR_UNPACK/nvvm"        "$SCRIPT_DIR/cuda65-bin/"
cp -r "$TMPDIR_UNPACK/open64"      "$SCRIPT_DIR/cuda65-bin/"

echo "Done. $(du -sh "$SCRIPT_DIR/cuda65-bin" | cut -f1) in cuda65-bin/"
echo "Now run: bash setup_ptxas_6_5_osx.sh"
