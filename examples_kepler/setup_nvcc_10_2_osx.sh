#!/usr/bin/env bash
# Lightweight macOS shim for CUDA 10.2.89 (sm_30), modeled on
# tinygrad/extra/setup_nvcc_osx.sh. There is no CUDA 10.3 — NVIDIA jumped
# 10.2 → 11.0 and dropped sm_30 in 11.0's ptxas.
#
# On Apple Silicon the image is linux/amd64 (QEMU). Packages are x86_64 only.
set -euo pipefail

IMAGE="${CUDA102_IMAGE:-cuda-nvcc:10.2-light}"
CONTAINER="${CUDA102_CNAME:-cuda-nvcc-10-2-persistent}"
BIN_DIR="${INSTALL_LOC:-$HOME/.local/bin}"
SHIM="$BIN_DIR/nvcc10shim"
DOCKER_HOST="${DOCKER_HOST:-unix://$HOME/.colima/docker.sock}"
export DOCKER_HOST

command -v docker >/dev/null 2>&1 || {
  echo "error: Docker is not installed or not in PATH" >&2
  exit 1
}

mkdir -p "$BIN_DIR"

# Compiler + disasm only (no cudart samples / full toolkit). In-image smoke
# test proves sm_30 still assembles before we ship the tag.
docker build --platform=linux/amd64 -t "$IMAGE" - <<'DOCKERFILE'
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/cuda-10.2/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget gnupg2 gcc g++ make \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
 && dpkg -i cuda-keyring_1.0-1_all.deb \
 && apt-get update && apt-get install -y --no-install-recommends \
      cuda-nvcc-10-2 \
      cuda-nvdisasm-10-2 \
      cuda-cuobjdump-10-2 \
 && rm -rf /var/lib/apt/lists/* /tmp/*

# In-image smoke: PTX → sm_30 cubin → disasm must see CTAID / TID.
# Store the index so ptxas cannot DCE the ctaid/tid math (empty kernels → NOPs).
RUN set -eux; \
    PTXAS=/usr/local/cuda-10.2/bin/ptxas; \
    CUOBJ=/usr/local/cuda-10.2/bin/cuobjdump; \
    NVCC=/usr/local/cuda-10.2/bin/nvcc; \
    printf '%s\n' \
      '.version 6.4' \
      '.target sm_30' \
      '.address_size 64' \
      '.visible .entry E_4(.param .u64 E_4_param_0)' \
      '{' \
      '    .reg .b32 %r<4>;' \
      '    .reg .b64 %rd<2>;' \
      '    ld.param.u64 %rd1, [E_4_param_0];' \
      '    mov.u32 %r1, %ctaid.x;' \
      '    mov.u32 %r2, %ntid.x;' \
      '    mov.u32 %r3, %tid.x;' \
      '    mad.lo.u32 %r1, %r1, %r2, %r3;' \
      '    st.global.u32 [%rd1], %r1;' \
      '    ret;' \
      '}' \
      > /tmp/smoke.ptx; \
    "$PTXAS" -arch=sm_30 /tmp/smoke.ptx -o /tmp/smoke.cubin; \
    "$CUOBJ" -sass /tmp/smoke.cubin | tee /tmp/smoke.sass; \
    grep -E 'CTAID\.X|SR_CTAID' /tmp/smoke.sass; \
    grep -E 'TID\.X|SR_TID' /tmp/smoke.sass; \
    "$NVCC" --version | tee /tmp/nvcc.ver; \
    grep -E 'release 10\.2|V10\.2' /tmp/nvcc.ver; \
    echo 'in-image sm_30 smoke: PASS'; \
    rm -f /tmp/smoke.ptx /tmp/smoke.cubin /tmp/smoke.sass /tmp/nvcc.ver

CMD ["sleep", "300"]
DOCKERFILE

cat > "$SHIM" <<'SHIM'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CUDA102_IMAGE:-cuda-nvcc:10.2-light}"
CONTAINER="${CUDA102_CNAME:-cuda-nvcc-10-2-persistent}"
TOOL="$(basename "$0")"
export DOCKER_HOST="${DOCKER_HOST:-unix://$HOME/.colima/docker.sock}"

case "$TOOL" in
  nvcc10) TOOL=nvcc ;;
  ptxas10) TOOL=ptxas ;;
  nvdisasm10) TOOL=nvdisasm ;;
  cuobjdump10) TOOL=cuobjdump ;;
  nvcc|ptxas|nvdisasm|cuobjdump) ;;
  *)
    echo "error: unsupported CUDA tool name: $TOOL" >&2
    exit 2
    ;;
esac

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  mounts=(-v "$HOME:$HOME")
  # macOS TMPDIR lives under /var/folders → /private/var/folders; mount both.
  [[ -d /var/folders ]] && mounts+=(-v /var/folders:/var/folders)
  [[ -d /private/var/folders ]] && mounts+=(-v /private/var/folders:/private/var/folders)
  [[ -d /tmp ]] && mounts+=(-v /tmp:/tmp)

  docker run -d \
    --platform=linux/amd64 \
    --name "$CONTAINER" \
    "${mounts[@]}" \
    -e "HOME=$HOME" \
    -w "$HOME" \
    "$IMAGE" >/dev/null
fi

# Resolve relative paths against the host cwd (docker exec does not inherit it).
args=()
for a in "$@"; do
  case "$a" in
    -*|/*) args+=("$a") ;;
    *) args+=("$(cd "$PWD" && python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$a")") ;;
  esac
done

exec docker exec \
  -e "HOME=$HOME" \
  -w "$PWD" \
  "$CONTAINER" "/usr/local/cuda-10.2/bin/$TOOL" "${args[@]}"
SHIM

chmod +x "$SHIM"

# Versioned names avoid replacing a newer (e.g. 12.8) nvcc.
for tool in nvcc10 ptxas10 nvdisasm10 cuobjdump10; do
  ln -sfn "$SHIM" "$BIN_DIR/$tool"
done

# Host-side smoke through the shim + mounts (keep under $HOME — /var/folders
# TMPDIR is not always visible inside Colima/Docker mounts).
mkdir -p "$HOME/.cache"
SMOKE_DIR="$(mktemp -d "$HOME/.cache/cuda102-smoke.XXXXXX")"
cleanup() { rm -rf "$SMOKE_DIR"; }
trap cleanup EXIT

cat > "$SMOKE_DIR/smoke.ptx" <<'PTX'
.version 6.4
.target sm_30
.address_size 64
.visible .entry E_4(.param .u64 E_4_param_0)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [E_4_param_0];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;
    st.global.u32 [%rd1], %r1;
    ret;
}
PTX

"$BIN_DIR/ptxas10" -arch=sm_30 "$SMOKE_DIR/smoke.ptx" -o "$SMOKE_DIR/smoke.cubin"
sass="$("$BIN_DIR/cuobjdump10" -sass "$SMOKE_DIR/smoke.cubin")"
echo "$sass" | grep -E 'CTAID\.X|SR_CTAID' >/dev/null
echo "$sass" | grep -E 'TID\.X|SR_TID' >/dev/null
"$BIN_DIR/nvcc10" --version | grep -E 'release 10\.2|V10\.2' >/dev/null

cat <<EOF
Installed CUDA 10.2.89 wrappers in:
  $BIN_DIR/nvcc10
  $BIN_DIR/ptxas10
  $BIN_DIR/nvdisasm10
  $BIN_DIR/cuobjdump10

CUDA 10.2.89 sm_30 smoke test: PASS

Compile GK104 / GTX 770 device code:
  $BIN_DIR/nvcc10 -arch=sm_30 --cubin kernel.cu -o kernel.cubin

Assemble PTX:
  $BIN_DIR/ptxas10 -arch=sm_30 kernel.ptx -o kernel.cubin
EOF
