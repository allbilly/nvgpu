"""
grctx_gk104.py — extract the GK104 GR context-init (csdata) register lists from
the nouveau linux source and resolve them to flat method-stream entries.

nouveau's gf100_gr_init_csdata() (engine/gr/gf100.c) uploads a *method stream*
into the FECS/GPCCS FALCON DMEM. The stream only encodes (register_addr, count,
pitch) — the actual init VALUES live in the FUC data file already loaded to
DMEM. So we only need addr/count/pitch from each gf100_gr_init entry.

The five csdata packs (hub/gpc_0/gpc_1/tpc/ppc) and their init arrays are static
const tables in gf100.c / ctxgk104.c / gf119.c. We parse them out of the local
ref/linux tree at import time.
"""

import os
import re

_REF = os.path.join(os.path.dirname(__file__), "..", "ref", "linux",
                    "drivers", "gpu", "drm", "nouveau", "nvkm", "engine", "gr")

# (pack_name, falcon_base, starstar, base) — from gf100_gr_init_ctxctl_int
CSDATA_LOADS = [
    ("gk104_grctx_pack_hub",   0x409000, 0x000, 0x000000),
    ("gk104_grctx_pack_gpc_0", 0x41a000, 0x000, 0x418000),
    ("gk104_grctx_pack_gpc_1", 0x41a000, 0x000, 0x418000),
    ("gk104_grctx_pack_tpc",   0x41a000, 0x004, 0x419800),
    ("gk104_grctx_pack_ppc",   0x41a000, 0x008, 0x41be00),
]


def _parse_init_arrays(text):
    """Return {name: [(addr,count,pitch), ...]} for every gf100_gr_init[] table.
    Stops at the first entry with count==0 (pack_for_each_init terminator)."""
    out = {}
    for m in re.finditer(
            r"const\s+struct\s+gf100_gr_init\s+(\w+)\[\]\s*=\s*\{(.*?)\};",
            text, re.S):
        name, body = m.group(1), m.group(2)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        entries = []
        for ent in re.findall(r"\{(.*?)\}", body, re.S):
            nums = re.findall(r"0x[0-9a-fA-F]+|0|[1-9]\d*", ent)
            if len(nums) < 3:
                continue
            addr = int(nums[0], 0)
            count = int(nums[1], 0)
            pitch = int(nums[2], 0)
            if count == 0:
                break  # terminator
            entries.append((addr, count, pitch))
        out[name] = entries
    return out


def _parse_packs(text):
    """Return {pack_name: [init_array_name, ...]} for every gf100_gr_pack[] table."""
    out = {}
    for m in re.finditer(
            r"const\s+struct\s+gf100_gr_pack\s+(\w+)\[\]\s*=\s*\{(.*?)\};",
            text, re.S):
        name, body = m.group(1), m.group(2)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        refs = re.findall(r"\{(.*?)\}", body, re.S)
        names = [r.strip() for r in refs if r.strip()]
        out[name] = names
    return out


def _load_source():
    text = ""
    for fn in sorted(os.listdir(_REF)):
        if fn.endswith(".c"):
            text += "\n" + open(os.path.join(_REF, fn)).read()
    return text


def _build():
    text = _load_source()
    inits = _parse_init_arrays(text)
    packs = _parse_packs(text)
    resolved = {}
    for pack_name, falcon, starstar, base in CSDATA_LOADS:
        if pack_name not in packs:
            raise KeyError(f"csdata pack {pack_name} not found in source")
        flat = []
        for init_name in packs[pack_name]:
            if init_name not in inits:
                raise KeyError(f"init array {init_name} (referenced by "
                               f"{pack_name}) not found in source")
            flat.extend(inits[init_name])
        resolved[pack_name] = {
            "falcon": falcon,
            "starstar": starstar,
            "base": base,
            "entries": flat,        # list of (addr, count, pitch)
        }
    return resolved


CSDATA = _build()


def method_stream(entries, base):
    """Replicate nouveau gf100_gr_init_csdata() packing: produce the list of
    32-bit method words (xfer<<26 | addr) exactly as the driver would write
    them to falcon+0x1c4. `base` is subtracted from each addr (see csdata)."""
    words = []
    addr = ~0 & 0xffffffff
    prev = ~0 & 0xffffffff
    xfer = 0
    for (eaddr, count, pitch) in entries:
        head = (eaddr - base) & 0xffffffff
        tail = head + count * pitch
        while head < tail:
            if head != (prev + 4) or xfer >= 32:
                if xfer:
                    words.append(((xfer - 1) << 26) | addr)
                    xfer = 0
                addr = head
            prev = head
            xfer += 1
            head += pitch
    if xfer:
        words.append(((xfer - 1) << 26) | addr)
    return words


if __name__ == "__main__":
    for pack_name, _, _, _ in CSDATA_LOADS:
        d = CSDATA[pack_name]
        ws = method_stream(d["entries"], d["base"])
        print(f"{pack_name}: {len(d['entries'])} entries -> "
              f"{len(ws)} method words "
              f"(falcon={d['falcon']:#x} starstar={d['starstar']:#x} "
              f"base={d['base']:#x})")
