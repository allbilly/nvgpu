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

# Packs used by gf100_grctx_generate_main() via gf100_gr_mmio() but NOT by
# csdata (they have no falcon/base — they're direct MMIO writes).
# Also includes icmd and mthd packs.
MMIO_PACKS = [
    "gk104_grctx_pack_hub",
    "gk104_grctx_pack_gpc_0",
    "gf100_grctx_pack_zcull",
    "gk104_grctx_pack_gpc_1",
    "gk104_grctx_pack_tpc",
    "gk104_grctx_pack_ppc",
]

ICMD_PACKS = [
    "gk104_grctx_pack_icmd",
]

MTHD_PACKS = [
    "gk104_grctx_pack_mthd",
]

# GK104 grctx constants (from ctxgk104.c gk104_grctx)
GK104_GRCTX_CONSTS = {
    "bundle_size": 0x3000,
    "bundle_min_gpm_fifo_depth": 0x180,
    "bundle_token_limit": 0x600,
    "pagepool_size": 0x8000,
    "attrib_nr_max": 0x324,
    "attrib_nr": 0x218,
    "alpha_nr_max": 0x7ff,
    "alpha_nr": 0x648,
}


def _parse_init_arrays(text):
    """Return {name: [(addr,count,pitch,data), ...]} for every gf100_gr_init[] table.
    Stops at the first entry with count==0 (pack_for_each_init terminator).
    The struct is {u32 addr, u8 count, u32 pitch, u64 data}."""
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
            data = int(nums[3], 0) if len(nums) > 3 else 0
            if count == 0:
                break  # terminator
            entries.append((addr, count, pitch, data))
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
    _inits = _parse_init_arrays(text)
    _packs = _parse_packs(text)
    resolved = {}
    for pack_name, falcon, starstar, base in CSDATA_LOADS:
        if pack_name not in _packs:
            raise KeyError(f"csdata pack {pack_name} not found in source")
        flat = []
        for init_name in _packs[pack_name]:
            if init_name not in _inits:
                raise KeyError(f"init array {init_name} (referenced by "
                               f"{pack_name}) not found in source")
            flat.extend(_inits[init_name])
        resolved[pack_name] = {
            "falcon": falcon,
            "starstar": starstar,
            "base": base,
            "entries": flat,        # list of (addr, count, pitch, data)
        }
    return resolved, _inits, _packs


CSDATA, inits, packs = _build()


def _resolve_pack(pack_name):
    """Return the flat list of (addr, count, pitch, data) entries for a pack."""
    if pack_name not in packs:
        raise KeyError(f"pack {pack_name} not found")
    flat = []
    for init_name in packs[pack_name]:
        if init_name not in inits:
            raise KeyError(f"init array {init_name} (referenced by "
                           f"{pack_name}) not found")
        flat.extend(inits[init_name])
    return flat


def mmio_pack(pack_name):
    """Return [(addr, data), ...] for a pack, expanding count/pitch.
    Matches gf100_gr_mmio(): direct register writes."""
    return mmio_writes(_resolve_pack(pack_name))


def icmd_pack(pack_name):
    """Return [(addr, data), ...] for an icmd pack, expanding count/pitch.
    Matches gf100_gr_icmd(): writes via 0x400200/0x400204 interface."""
    return mmio_writes(_resolve_pack(pack_name))


def mthd_pack(pack_name):
    """Return [(type, addr, data), ...] for a mthd pack, expanding count/pitch.
    Matches gf100_gr_mthd(): writes via 0x404488/0x40448c interface.
    Each pack entry has a type from the pack struct."""
    out = []
    raw = _parse_packs_with_types(pack_name)
    for entry_info in raw:
        init_name = entry_info["init"]
        ptype = entry_info["type"]
        if init_name not in inits:
            raise KeyError(f"init array {init_name} not found")
        for entry in inits[init_name]:
            addr, count, pitch, data = entry[0], entry[1], entry[2], entry[3]
            for _ in range(count):
                out.append((ptype, addr, data & 0xffffffff))
                addr += pitch
    return out


def _parse_packs_with_types(pack_name):
    """Return list of {init, type} for a pack, parsing the {init, type} format."""
    out = []
    for m in re.finditer(
            r"const\s+struct\s+gf100_gr_pack\s+(\w+)\[\]\s*=\s*\{(.*?)\};",
            _load_source(), re.S):
        if m.group(1) != pack_name:
            continue
        body = re.sub(r"/\*.*?\*/", "", m.group(2), flags=re.S)
        for ent in re.findall(r"\{(.*?)\}", body, re.S):
            # Parse { init_name, type } — skip empty terminator entries
            parts = re.findall(r'(\w+|0x[0-9a-fA-F]+|\d+)', ent)
            if not parts:
                continue
            init_name = parts[0]
            ptype = int(parts[1], 0) if len(parts) > 1 else 0
            out.append({"init": init_name, "type": ptype})
    return out


def method_stream(entries, base):
    """Replicate nouveau gf100_gr_init_csdata() packing: produce the list of
    32-bit method words (xfer<<26 | addr) exactly as the driver would write
    them to falcon+0x1c4. `base` is subtracted from each addr (see csdata).
    Entries are (addr, count, pitch, data) tuples; data is ignored here as
    the actual values live in the FUC DMEM."""
    words = []
    addr = ~0 & 0xffffffff
    prev = ~0 & 0xffffffff
    xfer = 0
    for entry in entries:
        eaddr, count, pitch = entry[0], entry[1], entry[2]
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


def mmio_writes(entries, base=0):
    """Replicate nouveau gf100_gr_mmio(): produce a list of (addr, value) pairs
    by expanding each (addr, count, pitch, data) entry.  `base` is NOT subtracted
    here (unlike csdata) — gf100_gr_mmio writes absolute register addresses."""
    out = []
    for entry in entries:
        addr, count, pitch, data = entry[0], entry[1], entry[2], entry[3]
        for _ in range(count):
            out.append((addr, data & 0xffffffff))
            addr += pitch
    return out


if __name__ == "__main__":
    for pack_name, _, _, _ in CSDATA_LOADS:
        d = CSDATA[pack_name]
        ws = method_stream(d["entries"], d["base"])
        print(f"{pack_name}: {len(d['entries'])} entries -> "
              f"{len(ws)} method words "
              f"(falcon={d['falcon']:#x} starstar={d['starstar']:#x} "
              f"base={d['base']:#x})")
