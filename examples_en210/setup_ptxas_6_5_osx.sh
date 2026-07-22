#!/usr/bin/env bash
# Lightweight macOS shim for CUDA 6.5.14 (sm_10/sm_11/sm_12/sm_13), modeled on
# examples_kepler/setup_nvcc_10_2_osx.sh. CUDA 6.5 is the LAST version that
# supports Tesla-era sm_12 (GT218 / GeForce 210). It was dropped in CUDA 7.0.
#
# No split .deb packages exist for 6.5, so we extract binaries from the runfile
# installer locally (cuda65-bin/) and COPY them into the image.
#
# On Apple Silicon the image is linux/amd64 (QEMU). Packages are x86_64 only.
set -euo pipefail

IMAGE="${CUDA65_IMAGE:-cuda-ptxas:6.5-light}"
CONTAINER="${CUDA65_CNAME:-cuda-ptxas-6-5-persistent}"
BIN_DIR="${INSTALL_LOC:-$HOME/.local/bin}"
SHIM="$BIN_DIR/ptxas65shim"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_HOST="${DOCKER_HOST:-unix://$HOME/.colima/docker.sock}"
export DOCKER_HOST

command -v docker >/dev/null 2>&1 || {
  echo "error: Docker is not installed or not in PATH" >&2
  exit 1
}

if [ ! -d "$SCRIPT_DIR/cuda65-bin" ]; then
  echo "error: $SCRIPT_DIR/cuda65-bin not found." >&2
  echo "  Run: cd $SCRIPT_DIR && bash extract_cuda65.sh" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"

docker build --platform=linux/amd64 -t "$IMAGE" "$SCRIPT_DIR" -f - <<'DOCKERFILE'
FROM ubuntu:14.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/cuda-6.5/bin:${PATH}

# Copy pre-extracted binaries from the build context (cuda65-bin/).
COPY cuda65-bin/ptxas      /usr/local/cuda-6.5/bin/ptxas
COPY cuda65-bin/nvcc       /usr/local/cuda-6.5/bin/nvcc
COPY cuda65-bin/nvcc.profile /usr/local/cuda-6.5/bin/nvcc.profile
COPY cuda65-bin/nvdisasm   /usr/local/cuda-6.5/bin/nvdisasm
COPY cuda65-bin/cuobjdump  /usr/local/cuda-6.5/bin/cuobjdump
COPY cuda65-bin/nvvm       /usr/local/cuda-6.5/nvvm
COPY cuda65-bin/open64     /usr/local/cuda-6.5/open64

RUN chmod +x /usr/local/cuda-6.5/bin/ptxas /usr/local/cuda-6.5/bin/nvcc /usr/local/cuda-6.5/bin/nvdisasm /usr/local/cuda-6.5/bin/cuobjdump

# nvcc needs to find nvvm and open64 relative to its install dir.
# Fix nvcc.profile paths to point to /usr/local/cuda-6.5.
RUN sed -i 's|TOP=.*|TOP=/usr/local/cuda-6.5|' /usr/local/cuda-6.5/bin/nvcc.profile

# In-image smoke: PTX → sm_12 cubin → disasm.
RUN set -eux; \
    PTXAS=/usr/local/cuda-6.5/bin/ptxas; \
    CUOBJ=/usr/local/cuda-6.5/bin/cuobjdump; \
    NVCC=/usr/local/cuda-6.5/bin/nvcc; \
    printf '%s\n' \
      '.version 4.1' \
      '.target sm_12' \
      '.address_size 32' \
      '.visible .entry add(.param .u32 a, .param .u32 b, .param .u32 out)' \
      '{' \
      '    .reg .u32 %r<6>;' \
      '    ld.param.u32 %r1, [a];' \
      '    ld.param.u32 %r2, [b];' \
      '    ld.param.u32 %r3, [out];' \
      '    ld.global.u32 %r4, [%r1];' \
      '    ld.global.u32 %r5, [%r2];' \
      '    add.u32 %r4, %r4, %r5;' \
      '    st.global.u32 [%r3], %r4;' \
      '    ret;' \
      '}' \
      > /tmp/smoke.ptx; \
    "$PTXAS" -arch=sm_12 /tmp/smoke.ptx -o /tmp/smoke.cubin; \
    test -s /tmp/smoke.cubin; \
    "$CUOBJ" -sass /tmp/smoke.cubin | tee /tmp/smoke.sass; \
    "$NVCC" --version | tee /tmp/nvcc.ver; \
    grep -E 'release 6\.5|V6\.5' /tmp/nvcc.ver; \
    echo 'in-image sm_12 smoke: PASS'; \
    rm -f /tmp/smoke.ptx /tmp/smoke.cubin /tmp/smoke.sass /tmp/nvcc.ver

CMD ["sleep", "300"]
DOCKERFILE

cat > "$SHIM" <<'SHIM'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CUDA65_IMAGE:-cuda-ptxas:6.5-light}"
CONTAINER="${CUDA65_CNAME:-cuda-ptxas-6-5-persistent}"
TOOL="$(basename "$0")"
export DOCKER_HOST="${DOCKER_HOST:-unix://$HOME/.colima/docker.sock}"

case "$TOOL" in
  nvcc65) TOOL=nvcc ;;
  ptxas65) TOOL=ptxas ;;
  nvdisasm65) TOOL=nvdisasm ;;
  cuobjdump65) TOOL=cuobjdump ;;
  nvcc|ptxas|nvdisasm|cuobjdump) ;;
  *)
    echo "error: unsupported CUDA tool name: $TOOL" >&2
    exit 2
    ;;
esac

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  mounts=(-v "$HOME:$HOME")
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
  "$CONTAINER" "/usr/local/cuda-6.5/bin/$TOOL" "${args[@]}"
SHIM

chmod +x "$SHIM"

for tool in nvcc65 ptxas65 nvdisasm65 cuobjdump65; do
  ln -sfn "$SHIM" "$BIN_DIR/$tool"
done

# Host-side smoke through the shim.
mkdir -p "$HOME/.cache"
SMOKE_DIR="$(mktemp -d "$HOME/.cache/cuda65-smoke.XXXXXX")"
cleanup() { rm -rf "$SMOKE_DIR"; }
trap cleanup EXIT

cat > "$SMOKE_DIR/smoke.ptx" <<'PTX'
.version 4.1
.target sm_12
.address_size 32
.visible .entry add(.param .u32 a, .param .u32 b, .param .u32 out)
{
    .reg .u32 %r<6>;
    ld.param.u32 %r1, [a];
    ld.param.u32 %r2, [b];
    ld.param.u32 %r3, [out];
    ld.global.u32 %r4, [%r1];
    ld.global.u32 %r5, [%r2];
    add.u32 %r4, %r4, %r5;
    st.global.u32 [%r3], %r4;
    ret;
}
PTX

"$BIN_DIR/ptxas65" -arch=sm_12 "$SMOKE_DIR/smoke.ptx" -o "$SMOKE_DIR/smoke.cubin"
"$BIN_DIR/cuobjdump65" -sass "$SMOKE_DIR/smoke.cubin" > "$SMOKE_DIR/smoke.sass"
test -s "$SMOKE_DIR/smoke.cubin"
"$BIN_DIR/nvcc65" --version | grep -E 'release 6\.5|V6\.5' >/dev/null

cat <<EOF
Installed CUDA 6.5.14 wrappers in:
  $BIN_DIR/nvcc65
  $BIN_DIR/ptxas65
  $BIN_DIR/nvdisasm65
  $BIN_DIR/cuobjdump65

CUDA 6.5.14 sm_12 smoke test: PASS

Compile Tesla / GT218 (GeForce 210) device code:
  $BIN_DIR/nvcc65 -arch=sm_12 --cubin kernel.cu -o kernel.cubin

Assemble PTX for sm_12:
  $BIN_DIR/ptxas65 -arch=sm_12 kernel.ptx -o kernel.cubin

Disassemble cubin:
  $BIN_DIR/cuobjdump65 -sass kernel.cubin
EOF
