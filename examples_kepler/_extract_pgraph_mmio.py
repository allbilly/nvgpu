#!/usr/bin/env python3
"""Generate pgraph_mmio_gk104.py from the Linux kernel gk104.c / gf100*.c init lists."""
import os, re, glob

REPO = "/Users/yeren/Desktop/nvgpu"
SRC_DIR = os.path.join(REPO, "ref/linux/drivers/gpu/drm/nouveau/nvkm/engine/gr")
OUT = os.path.join(REPO, "examples_kepler/pgraph_mmio_gk104.py")

# Parse the gk104_gr_pack_mmio array to get ordered list of init array names.
with open(os.path.join(SRC_DIR, "gk104.c")) as f:
    gk104 = f.read()

m = re.search(r"gk104_gr_pack_mmio\[\]\s*=\s*\{(.*?)\}\s*;", gk104, re.S)
if not m:
    raise RuntimeError("gk104_gr_pack_mmio not found")
pack_names = re.findall(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}", m.group(1))

# Remove trailing sentinel empty pack if present
pack_names = [n for n in pack_names if n != "0"]

# Parse every gr_init array in the source .c files.
arrays = {}
for cpath in sorted(glob.glob(os.path.join(SRC_DIR, "*.c"))):
    text = open(cpath).read()
    # remove C comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for m in re.finditer(r"(?:static\s+)?const\s+struct\s+gf100_gr_init\s+([A-Za-z_][A-Za-z0-9_]*)\[\]\s*=\s*\{(.*?)\}\s*;", text, re.S):
        name = m.group(1)
        body = m.group(2)
        flat = []
        for entry in re.finditer(r"\{\s*(0x[0-9a-fA-F]+)\s*,\s*(\d+)\s*,\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*\}", body):
            addr = int(entry.group(1), 16)
            count = int(entry.group(2))
            pitch = int(entry.group(3), 16)
            value = int(entry.group(4), 16)
            for i in range(count):
                flat.append((addr + i * pitch, value))
        arrays[name] = flat

# Expand in order
all_writes = []
missing = []
for name in pack_names:
    if name in arrays:
        all_writes.extend(arrays[name])
    else:
        missing.append(name)

if missing:
    raise RuntimeError("Missing arrays: " + ", ".join(missing))

# Deduplicate consecutive writes to the same address (keep last) because the same
# register may appear in multiple packs with different values.
seen = {}
ordered = []
for addr, value in all_writes:
    key = addr
    if key in seen:
        ordered[seen[key]] = (addr, value)
    else:
        seen[key] = len(ordered)
        ordered.append((addr, value))

# Write Python source
with open(OUT, "w") as f:
    f.write("# Auto-generated from gk104.c / gf100.c / gf108.c / gf117.c / gf119.c\n")
    f.write("# Flattened gk104_gr_pack_mmio with later writes overriding earlier ones.\n")
    f.write("GK104_PGRAPH_PACK_MMIO = [\n")
    for addr, value in ordered:
        f.write(f"  (0x{addr:06x}, 0x{value:08x}),\n")
    f.write("]\n")

print(f"Wrote {len(ordered)} register writes to {OUT}")
