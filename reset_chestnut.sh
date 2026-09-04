cd /home/orangepi/asm2464pd-firmware

UV_BIN="$(command -v uv)"

test -x "$UV_BIN" || { echo "uv not found"; exit 1; }

sudo "$UV_BIN" python install 3.11

sudo "$UV_BIN" run \
  --python 3.11 \
  --with pyftdi \
  --with pyusb \
  --with tinygrad==0.14.0 \
  python3 ftdi_debug.py -bn

sudo "$UV_BIN" run \
  --python 3.11 \
  --with pyftdi \
  --with pyusb \
  --with tinygrad==0.14.0 \
  python3 flash.py firmware/AS_USB4_231204_85_00_00.bin

sudo "$UV_BIN" run \
  --python 3.11 \
  --with pyftdi \
  --with pyusb \
  --with tinygrad==0.14.0 \
  python3 ftdi_debug.py -rn
