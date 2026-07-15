#!/usr/bin/env python3
"""Standalone NV stack for a GTX 770 (Kepler, GK104, sm_30) over raw PCIe on Linux.

Linux port of examples_kepler/add.py: the macOS TinyGPU.app DriverKit socket
transport from the upstream file is replaced here by direct BAR0/BAR1 mmap via
sysfs (/sys/bus/pci/devices/<bdf>/resourceN).  Requires the GK104 to be unbound
from the proprietary nvidia driver (e.g. bound to vfio-pci or no driver) and
root (or CAP_SYS_RAWIO) to open the resource files.

================================================================================
IMPORTANT — KEPLER BRING-UP STATUS (read before running the live path)
================================================================================
Kepler GK104 has **NO GSP**.  There is no firmware that runs the RM for us.
The entire RM must be reimplemented in userspace as raw MMIO / register
programming — this is exactly what nouveau's `nvkm` does in ~50k lines
(FALCON PMU bring-up, GR context-switch bundles, FIFO channels, GMMU page
tables, the compute work descriptor launch).

Consequence: the *live* path in this file is a **best-effort skeleton**.  Every
step that requires real hardware interaction is marked with `# KEPLER-TODO` and
a short note on what register/sequence to implement.  It will NOT run a kernel
on first try; it needs the same iterative, on-hardware bring-up the GA102 path
went through (23 debug milestones there).  The offline gate below, however, is
fully implemented and runnable — it validates the Kepler cubin builder, the
GK104 GMMU page-table helpers, the CWD launch-word builder, and the shared
platform scaffolding.

The only external dependencies this module has are:
  - tinygrad.runtime.autogen.{nv, nv_570, nv_regs, pci, libc}  (ctypes constants only)
  - Python standard library
  - nvbios_init / pgraph_mmio_gk104 (reused from examples_kepler/ via sys.path)

NO imports from tinygrad.runtime.support / ops / device / renderer / uop / helpers
are permitted on the live path — those have been vendored inline below.
"""
from __future__ import annotations
import os, sys, ctypes, ctypes.util, time, mmap, struct, math, array as _array_mod, socket, subprocess, contextlib, functools, itertools, enum, atexit, select, dataclasses, collections, urllib.request, hashlib, tempfile, gzip, pathlib, json, threading
from typing import cast, Any, ClassVar, Generic, TypeVar

# --- autogen ctypes (allowed: "ctypes constants only") ---
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu, pci
from tinygrad.runtime.autogen import nv_regs
from tinygrad.runtime.autogen import libc
# Reuse the Kepler bring-up modules from examples_kepler/ rather than duplicating
# them here (ponytail: fewest files).  NV_KEPLER_PATH overrides for non-standard layouts.
sys.path.insert(0, os.environ.get("NV_KEPLER_PATH",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples_kepler")))
import nvbios_init
from pgraph_mmio_gk104 import GK104_PGRAPH_PACK_MMIO

# ============================================================================
# Helpers (slimmed from tinygrad/helpers.py — Kepler-agnostic, reused verbatim)
# ============================================================================
DEBUG = int(os.environ.get("DEBUG", "0"))
def getenv(k: str, default=0):
  v = os.environ.get(k)
  if v is None: return default
  try: return int(v)
  except: return v
def getbits(value: int, start: int, end: int) -> int: return (value >> start) & ((1 << (end - start + 1)) - 1)
def i2u(dtype: int, val: int) -> int: return val & ((1 << (dtype * 8)) - 1)
def round_up(num: int, amt: int) -> int: return ((num + amt - 1) // amt) * amt
def round_down(num: int, amt: int) -> int: return -round_up(-num, amt)
def ceildiv(num: int, amt: int) -> int: return -(num // -amt)
def lo32(x: int) -> int: return x & 0xFFFFFFFF
def hi32(x: int) -> int: return x >> 32
def data64(x: int) -> tuple: return ((x >> 32) & 0xFFFFFFFF, x & 0xFFFFFFFF)  # (hi, lo) — matches tinygrad helpers.data64
def data64_le(x: int) -> tuple: return (x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF)  # (lo, hi) — matches tinygrad helpers.data64_le
def unwrap(x): return x

def nv_flags(reg, **kwargs) -> int:
  return functools.reduce(int.__or__, ((getattr(nv_gpu, f"{reg}_{k}_{v}".upper()) if isinstance(v, str) else v) <<
    getattr(nv_gpu, f"{reg}_{k}".upper())[1] for k, v in kwargs.items()), 0)
OSX = sys.platform == "darwin"

# Tinygrad uses this from `array.array` but our stubs above use the stdlib module.
array = _array_mod

def to_mv(ptr: int, sz: int) -> memoryview: return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")
def mv_address(mv) -> int: return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def from_mv(mv: memoryview, to_type=ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents

def wait_cond(cb, *args, value=True, timeout_ms=10000, msg=""):
  start_time = int(time.perf_counter() * 1000)
  while int(time.perf_counter() * 1000) - start_time < timeout_ms:
    if (val := cb(*args)) == value: return val
  raise TimeoutError(f"{msg}. Timed out after {timeout_ms} ms, condition not met: {val} != {value}")

def _ensure_downloads_dir() -> pathlib.Path:
  d = pathlib.Path(os.path.expanduser("~")) / ".cache" / "tinygrad"
  d.mkdir(parents=True, exist_ok=True)
  return d

def temp(name: str) -> str:
  return os.path.join(tempfile.gettempdir(), name)

def pluralize(n, s, p=None):
  if p is None: p = s + "s"
  return f"{n} {p}" if n != 1 else f"1 {s}"


# ============================================================================
# memory.py (vendored; Kepler-agnostic allocators reused verbatim)
# ============================================================================
class BumpAllocator:
  def __init__(self, size: int, base: int = 0, wrap: bool = True):
    self.size, self.ptr, self.base, self.wrap = size, 0, base, wrap
  def alloc(self, size: int, alignment: int = 1) -> int:
    if round_up(self.ptr, alignment) + size > self.size:
      if not self.wrap: raise RuntimeError("Out of memory")
      self.ptr = 0
    self.ptr = (res := round_up(self.ptr, alignment)) + size
    return res + self.base

class TLSFAllocator:
  def __init__(self, size: int, base: int = 0, block_size: int = 16, lv2_cnt: int = 16):
    self.size, self.base, self.block_size, self.l2_cnt = size, base, block_size, lv2_cnt.bit_length()
    self.storage = [collections.defaultdict(list) for _ in range(size.bit_length() + 1)]
    self.lv1_entries = [0] * len(self.storage)
    self.blocks = {0: (size, None, None, True)}
    if size > 0: self._insert_block(0, size)

  @functools.cache
  def lv1(self, size): return size.bit_length()
  @functools.cache
  def lv2(self, size): return (size - (1 << (size.bit_length() - 1))) // (1 << max(0, size.bit_length() - self.l2_cnt))

  def _insert_block(self, start: int, size: int, prev=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].append(start)
    self.lv1_entries[self.lv1(size)] += 1
    self.blocks[start] = (size, start + size, prev, True)
    return self
  def _remove_block(self, start: int, size: int, prev=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].remove(start)
    self.lv1_entries[self.lv1(size)] -= 1
    self.blocks[start] = (size, start + size, prev, False)
    return self
  def _split_block(self, start: int, size: int, new_size: int):
    nxt = self.blocks[start][1]
    assert self.blocks[start][3], "block must be free"
    self._remove_block(start, size)._insert_block(start, new_size)._insert_block(start + new_size, size - new_size, prev=start)
    if nxt in self.blocks:
      self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start + new_size, self.blocks[nxt][3])
    return self
  def _merge_right(self, start: int):
    size, nxt, _, is_free = self.blocks[start]
    assert is_free, "block must be free"
    while is_free and nxt in self.blocks:
      if (blk := self.blocks[nxt])[3] is False: break
      self._remove_block(start, size)._remove_block(nxt, blk[0])._insert_block(start, size := size + blk[0])
      assert self.blocks[start][1] == blk[1]
      _, nxt, _, _ = self.blocks.pop(nxt)
    if nxt in self.blocks: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start, self.blocks[nxt][3])
  def _merge_block(self, start: int):
    while (x := self.blocks[start][2]) is not None and self.blocks[x][3] is True: start = x
    self._merge_right(start)
  def alloc(self, req_size: int, align: int = 1) -> int:
    req_size = max(self.block_size, req_size)
    size = max(self.block_size, req_size + align - 1)
    size = round_up(size, (1 << size.bit_length() - self.l2_cnt))
    for l1 in range(self.lv1(size), len(self.storage)):
      if self.lv1_entries[l1] == 0: continue
      for l2 in range(self.lv2(size) if l1 == size.bit_length() else 0, (1 << self.l2_cnt)):
        if len(self.storage[l1][l2]) > 0:
          start = self.storage[l1][l2][0]
          nsize = self.blocks[start][0]
          assert nsize >= size, "block must be larger"
          if (new_start := round_up(start, align)) != start:
            self._split_block(start, nsize, new_start - start)
            start, nsize = new_start, self.blocks[new_start][0]
          if nsize > req_size: self._split_block(start, nsize, req_size)
          self._remove_block(start, req_size)
          return start + self.base
    raise MemoryError(f"Can't allocate {req_size} bytes")
  def free(self, start: int):
    self._insert_block(start - self.base, self.blocks[start - self.base][0])._merge_block(start - self.base)

class AddrSpace(enum.Enum):
  PHYS = enum.auto(); SYS = enum.auto(); NCOH = enum.auto(); PEER = enum.auto()

@dataclasses.dataclass(frozen=True)
class VirtMapping:
  va_addr: int; size: int; paddrs: list; aspace: AddrSpace; uncached: bool = False; snooped: bool = False


# ============================================================================
# GK104 GMMU page tables
# ----------------------------------------------------------------------------
# Kepler GK104 uses the GF100 2-level GMMU layout: a PGD covering VA bits
# 39:27 and a 4-KiB small-page table covering bits 26:12. Entries are 8 bytes.
# Big-page / 4KB-page support is present; we implement the 4KB (and 64KB big)
# path here.  The exact PTE/PDE bit layout below follows the nouveau gk104_mmu
# definitions and MUST be validated against the silicon during bring-up.
#
#   GK104_PTE_VALID  (1 << 0)
#   GK104_PTE_WRITE  (1 << 1)
#   GK104_PTE_READ   (1 << 2)
#   GF100_PTE target is at bits 33:34, storage kind at 36:43, and the
#   physical address is stored as (paddr >> 8) in bits 4:31.
#
# KEPLER-TODO: confirm PTE/PDE bit positions against running nouveau / an
# nvkm register dump on the GTX 770 (especially the aperture bits and the
# big-page encoding for 64KB pages).
# ============================================================================
class GK104PageTableEntry:
  # GF100_VM/GK104 uses the same PDE/PTE format as the GF100 Nouveau VMM.
  PTE_VALID = 1 << 0
  PTE_READ_ONLY = 1 << 2
  PTE_PDE_4K = 2 << 32         # SPT target (HOST) in PDE high-half bits [32:33]
  PTE_APER_VRAM = 0 << 33
  PTE_APER_SYS  = 2 << 33
  PTE_FRAME = 0xFFFFFFF0       # address field bits [4:31], paddr >> 8
  # Convenience flags retained for the offline self-test.
  PTE_READ  = 1 << 1
  PTE_WRITE = 1 << 2

  def __init__(self, dev, paddr, lv):
    self.dev, self.paddr, self.lv, self.addr = dev, paddr, lv, paddr
  def _read64(self, idx):
    return struct.unpack_from("<Q", self.dev.vram, self.paddr + idx * 8)[0]
  def _write64(self, idx, val):
    struct.pack_into("<Q", self.dev.vram, self.paddr + idx * 8, val)
  def entry(self, idx):
    return self._read64(idx)
  def valid(self, idx):
    present = (0x3 << 32) if self.lv == 1 else self.PTE_VALID
    return (self.entry(idx) & present) != 0
  def is_page(self, idx):
    # A level-0 entry is a leaf page.
    return self.lv == 0
  def supports_huge_page(self, paddr):
    return (paddr & 0xFFFF) == 0  # 64KB alignment for big pages
  def address(self, idx):
    # Target frame of the entry at `idx` (next table for PDEs, page for PTEs),
    # returned as the *local* allocator offset (bus_base stripped) so the walk
    # can index the next level with the same offsets the allocator uses.
    base = getattr(self.dev, "mm", None)
    bus = base.bus_base if base else 0
    entry = self.entry(idx)
    if self.lv == 1:
      return ((entry >> 24) & ~0xfff) - bus  # SPT address is PDE bits [63:24]
    return ((entry & self.PTE_FRAME) << 8) - bus
  def set_entry(self, idx, paddr, table=False, valid=True, aspace=AddrSpace.PHYS, uncached=False, snooped=False, frag=0):
    # Build the GF100/GK104 8-byte leaf or small-page PDE.
    base = getattr(self.dev, "mm", None)
    bus = base.bus_base if base and aspace is not AddrSpace.PHYS else 0
    if table:
      val = self.PTE_PDE_4K | (1 << 34) | ((bus + paddr) << 24)
    else:
      aper = (self.PTE_APER_SYS if aspace is AddrSpace.SYS else
              (3 << 33) if aspace is AddrSpace.NCOH else self.PTE_APER_VRAM)
      vol = (1 << 32) if aspace is AddrSpace.SYS else 0
      val = ((bus + paddr) >> 8) | self.PTE_VALID | vol | aper
    self._write64(idx, val)
  def palloc(self, size, zero=False, boot=False, ptable=False):
    return self.dev.mm.palloc(size, zero=zero, boot=boot, ptable=ptable)


class GK104MemoryManager:
  """Kepler GK104 GMMU memory manager using Nouveau's GF100 layout."""
  va_allocator = None
  va_bits = 40
  # PGD index is VA[39:27], SPT index is VA[26:12].
  va_shifts = [15, 13, 12]
  pte_cnt = [1 << 15, 1 << 13]
  pt_t = GK104PageTableEntry

  def __init__(self, dev, vram_size, boot_size, bus_base=0):
    self.dev, self.vram_size = dev, vram_size
    self.bus_base = bus_base  # GPU-visible base of `vram` (sysmem bus addr)
    self.boot_allocator = TLSFAllocator(boot_size, base=0)
    self.pa_allocator = TLSFAllocator(vram_size - boot_size, base=boot_size)
    # Keep VA zero reserved, but use an absolute allocator base so hardware
    # ring/context alignment is not skewed by a 0x1000 allocator offset.
    GK104MemoryManager.va_allocator = TLSFAllocator(1 << 36, base=0)
    GK104MemoryManager.va_allocator.alloc(0x1000, 0x1000)
    self.root_pa = self.palloc(0x10000, zero=True, boot=True)
    self.root_page_table = self.pt_t(self.dev, self.root_pa, lv=1)
    self.vram = dev.vram
  def alloc_vaddr(self, size, align=0x1000):
    return GK104MemoryManager.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))
  def map_range(self, vaddr, size, paddrs, aspace, uncached=False, snooped=False, boot=False):
    # GF100/GK104 2-level walk for 4-KiB pages:
    #   [11:0]   page offset
    #   [26:12]  small-page table index (15 bits)
    #   [39:27]  PGD index (13 bits)
    # `paddrs` is a list of (paddr, seg_size) physical segments; we walk VA and
    # PA together and populate PDE/PTE entries (allocated from the boot region
    # as page-table pages).
    def pa_at(off):
      o = off
      for p, sz in paddrs:
        if o < sz: return p + o
        o -= sz
      raise IndexError("pa out of range")
    pages = ceildiv(size, 0x1000)
    for i in range(pages):
      v = vaddr + i * 0x1000
      p = pa_at(i * 0x1000)
      pgd_idx = (v >> 27) & 0x1FFF
      spt_idx = (v >> 12) & 0x7FFF
      pgd = self.root_page_table
      if not pgd.valid(pgd_idx):
        spt_pa = self.palloc(0x40000, zero=True, boot=True, ptable=True)
        # GF100_PDE: the 4-KiB SPT is encoded in the high half.  HOST target
        # occupies bits [32:33], VOL is bit 34, and address is shifted by 24.
        # The SPT itself is allocated from the boot/system pool; this target
        # describes the page-table storage, not the mapped leaf pages.
        pde_target = (2 << 32) | (1 << 34)
        struct.pack_into("<Q", self.dev.vram, pgd.paddr + pgd_idx * 8,
                         pde_target |
                         ((self.bus_base + spt_pa) << 24))
      spt = self.pt_t(self.dev, pgd.address(pgd_idx), lv=0)
      spt.set_entry(spt_idx, p, table=False, aspace=aspace)
    return VirtMapping(vaddr, size, [p for p, _ in paddrs], aspace, uncached, snooped)
  def palloc(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    allocator = self.boot_allocator if boot else self.pa_allocator
    paddr = allocator.alloc(round_up(size, 0x1000), align)
    if zero: self.dev.vram[paddr:paddr + size] = bytes(size)
    return paddr
  def valloc(self, size, align=0x1000, uncached=False, contiguous=False, aspace=AddrSpace.PHYS):
    va = self.alloc_vaddr(size := round_up(size, 0x1000), align)
    paddrs = [(self.palloc(size, zero=True), size)]
    return self.map_range(va, size, paddrs, aspace=aspace, uncached=uncached)


# ============================================================================
# hcq.py (vendored; MMIO + file IO + HCQBuffer — Kepler-agnostic)
# ============================================================================
class MMIOInterface:
  def __init__(self, addr, nbytes, fmt='B'):
    self.mv, self.addr, self.nbytes, self.fmt = to_mv(addr, nbytes).cast(fmt), addr, nbytes, fmt
  def __len__(self): return self.nbytes // struct.calcsize(self.fmt)
  def __getitem__(self, k): return (self.mv[k] if self.fmt == 'B' else self.mv[k].tolist()) if isinstance(k, slice) else self.mv[k]
  def __setitem__(self, k, v):
    if self.fmt != 'B' and isinstance(v, (list, tuple)):
      self.mv[k] = array.array(self.fmt, v)
    else:
      self.mv[k] = v
  def view(self, offset=0, size=None, fmt=None):
    return MMIOInterface(self.addr + offset, (self.nbytes - offset) if size is None else size, fmt=fmt or self.fmt)

class FileIOInterface:
  def __init__(self, path="", flags=os.O_RDONLY, fd=None):
    self.path = path
    self.fd = fd or os.open(path, flags)
  def __del__(self):
    if hasattr(self, 'fd'):
      try: os.close(self.fd)
      except: pass
  def ioctl(self, request, arg):
    import fcntl
    return fcntl.ioctl(self.fd, request, arg)
  def mmap(self, start, sz, prot, flags, offset):
    return FileIOInterface._mmap(start, sz, prot, flags, self.fd, offset)
  def read(self, size=None, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as f: return f.read(size)
  def write(self, content, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as f: f.write(content)
  def seek(self, offset): os.lseek(self.fd, offset, os.SEEK_SET)
  @staticmethod
  def _mmap(start, sz, prot, flags, fd, offset):
    x = libc.mmap(start, sz, prot, flags, fd, offset)
    if x == 0xffffffffffffffff: raise OSError(f"Failed to mmap {sz} bytes at {hex(start)}: {os.strerror(ctypes.get_errno())}")
    return x
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, -1, offset)
  @staticmethod
  def munmap(buf, sz): return libc.munmap(buf, sz)
  @staticmethod
  def exists(path): return os.path.exists(path)
  @staticmethod
  def readlink(path): return os.readlink(path)
  @staticmethod
  def eventfd(initval, flags=None):
    import fcntl as _f
    return FileIOInterface(fd=os.eventfd(initval, flags))

def hcq_filter_visible_devices(devs, device):
  return devs

class HCQBuffer:
  def __init__(self, va_addr, size, meta=None, _base=None, view=None, owner=None):
    self.va_addr, self.size, self.meta, self._base, self.owner = va_addr, size, meta, _base, owner
    self._view = view
  def cpu_view(self):
    return self._base if self._view is None else self._view
  def offset(self, off, size=None):
    v = self.cpu_view().view(off, (self.size - off) if size is None else size)
    return HCQBuffer(self.va_addr + off, (self.size - off) if size is None else size, meta=self.meta, _base=self._base, view=v, owner=self.owner)


# ============================================================================
# ELF loader (vendored; generic — Kepler-agnostic, used to parse our cubin)
# ============================================================================
class ElfSection: pass
def _elf_strtab(blob, idx): return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')
def link_sym(sym: str, libs: list[str]) -> int:
  raise NotImplementedError("link_sym: Kepler stub")
def elf_loader(blob, force_section_align=1, link_libs=None):
  # Minimal ELF64 loader used only to inspect our own cubin after build.
  e_phoff = struct.unpack_from("<Q", blob, 0x20)[0]
  e_shoff = struct.unpack_from("<Q", blob, 0x28)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", blob, 0x36)
  e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3a)
  return {"phoff": e_phoff, "shoff": e_shoff, "phnum": e_phnum, "shnum": e_shnum}

def elf_section(blob, wanted):
  shoff = struct.unpack_from("<Q", blob, 0x28)[0]
  entsz, count = struct.unpack_from("<HH", blob, 0x3a)
  shstrndx = struct.unpack_from("<H", blob, 0x3e)[0]
  def sh(i): return struct.unpack_from("<IIQQQQIIQQ", blob, shoff + i * entsz)
  shstr = sh(shstrndx)
  names = blob[shstr[4]:shstr[4] + shstr[5]]
  for i in range(count):
    h = sh(i)
    end = names.find(b"\0", h[0])
    if names[h[0]:end].decode(errors="replace") == wanted:
      return h, bytes(blob[h[4]:h[4] + h[5]])
  raise KeyError(wanted)

def elf_section_bytes(blob, wanted):
  """Return one ELF64 section payload by name (used to upload SASS, not ELF)."""
  return elf_section(blob, wanted)[1]

def cubin_register_count(blob, kernel):
  regs = (elf_section(blob, f".text.{kernel}")[0][7] >> 24) & 0x3f
  if not regs: raise ValueError(f"cubin has no register count for {kernel}")
  return regs


# ============================================================================
# PCI interface — unlike GA102 (which talks to the GSP through /dev/nvidiactl),
# Kepler GK104 has no GSP and is driven directly through BAR0/BAR1 mmap'd from
# sysfs resource files.  There is no socket, no RPC, no per-access round-trip;
# every register access is a plain load/store on the mmap'd window.  The device
# must be unbound from the proprietary nvidia driver (the 595+ driver dropped
# Kepler anyway) — bind it to vfio-pci or leave it driverless, then run as root.
# ============================================================================
PAGESIZE = 0x1000

# GK104 PCI device id (Palit GTX 770) — used by probe() to pick the right BDF
# when NV_PCIBDF is not set.  Kepler desktop ids: 0x1180 (GK104), 0x1184/0x1185
# (GK104 respin), 0x1188 (GK104), 0x1189 (GK104), 0x118e (GK104).
GK104_PCI_IDS = {0x1180, 0x1184, 0x1185, 0x1187, 0x1188, 0x1189, 0x118e}


class RemotePCIDevice:
  """Abstract transport for a raw PCIe GPU (Linux sysfs mmap, vfio, ...)."""
  def __init__(self, name, transport):
    self.name, self.transport = name, transport
  def bar_info(self, bar):
    raise NotImplementedError
  def map_bar(self, bar, fmt='B', off=0, size=None):
    raise NotImplementedError
  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    raise NotImplementedError
  def mmio_read(self, bar, offset, size):
    raise NotImplementedError
  def mmio_write(self, bar, offset, data):
    raise NotImplementedError


def _detect_gk104_bdf():
  """Scan /sys/bus/pci/devices for an unbound NVIDIA GK104 and return its BDF.
  A device bound to any driver (nvidia, nouveau, snd_hda_intel, ...) is skipped:
  opening its resourceN file requires the driver to not be holding the BAR.
  Returns None if none is found (caller falls back to NV_PCIBDF or errors)."""
  base = pathlib.Path("/sys/bus/pci/devices")
  if not base.is_dir():
    return None
  for d in sorted(base.iterdir()):
    ven_path = d / "vendor"
    if not ven_path.exists():
      continue
    try:
      ven = int(ven_path.read_text().strip(), 16)
      dev = int((d / "device").read_text().strip(), 16)
    except (OSError, ValueError):
      continue
    if ven == 0x10de and dev in GK104_PCI_IDS and not (d / "driver").is_symlink():
      return d.name
  return None


class LinuxPCIDevice(RemotePCIDevice):
  """Linux: raw PCIe BAR access via /sys/bus/pci/devices/<bdf>/resourceN.
  _sysfs_root is overridable (e.g. by the offline selftest) to point at a fake
  sysfs tree.

  Each resourceN file is a mmap'd window onto BAR N.  We mmap it once and serve
  every register access as a direct load/store on the resulting memoryview — no
  socket, no RPC, no per-access round-trip.  PCI config space is read/written
  through the `config` sysfs file.  Opening resourceN requires root (or
  CAP_SYS_RAWIO); the device must NOT be bound to the proprietary nvidia driver.

  The freeze / arm_final_output_read helpers keep the same state-machine shape
  as the macOS transport in examples_kepler/add.py so the (gated) live bring-up
  path in run_hardware_demo can call them unchanged; on Linux a freeze simply
  rejects further MMIO since there is no socket to drain.  KEPLER-TODO: real
  sysmem DMA mapping needs a vfio group / IOMMU mapping, not an anonymous host
  buffer."""
  _sysfs_root = "/sys/bus/pci/devices"
  def __init__(self, bdf=None, dev_id=0):
    super().__init__("NV", "pcie")
    self.dev_id = dev_id
    self.bdf = bdf or os.environ.get("NV_PCIBDF") or _detect_gk104_bdf()
    if not self.bdf:
      raise RuntimeError("no GK104 found in /sys/bus/pci/devices; set NV_PCIBDF=<bdf>")
    self.sysfs = f"{self._sysfs_root}/{self.bdf}"
    if not pathlib.Path(self.sysfs).is_dir():
      raise RuntimeError(f"PCI device {self.bdf} not present at {self.sysfs}")
    self._bars = {}          # bar -> (fd, mmap_ptr, size, memoryview)
    self._config_fd = None
    self._fini_done = False
    self._phase = "init"
    self._frozen = False
    self._freeze_reason = None
    self._final_read_budget = None  # (bar, offset, size) or "consumed" or None
    self._res = self._read_resource()

  # --- sysfs parsing ---
  def _read_resource(self):
    # /sys/bus/pci/devices/<bdf>/resource: one line per BAR, "start end flags".
    res = []
    with open(f"{self.sysfs}/resource", "r") as f:
      for line in f:
        line = line.strip()
        if not line:
          res.append((0, 0, 0)); continue
        parts = line.split()
        res.append((int(parts[0], 0), int(parts[1], 0), int(parts[2], 0)))
    return res

  def bar_info(self, bar):
    if bar >= len(self._res):
      return (0, 0)
    start, end, _ = self._res[bar]
    return (start, end - start + 1) if end else (0, 0)

  def _map_bar(self, bar):
    if bar in self._bars:
      return self._bars[bar]
    start, size = self.bar_info(bar)
    if not size:
      raise OSError(f"BAR{bar} not present on {self.bdf}")
    fd = os.open(f"{self.sysfs}/resource{bar}", os.O_RDWR | os.O_SYNC)
    ptr = libc.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, fd, 0)
    if ptr == ctypes.c_void_p(-1).value:
      os.close(fd)
      raise OSError(f"mmap BAR{bar} ({size} bytes) failed on {self.bdf}: {os.strerror(ctypes.get_errno())}")
    mv = to_mv(ptr, size)
    self._bars[bar] = (fd, ptr, size, mv)
    return self._bars[bar]

  # --- MMIO (direct load/store on the mmap'd BAR) ---
  def _guard(self):
    if self._frozen:
      raise RuntimeError(f"MMIO after freeze: {self._freeze_reason}")
    if self._final_read_budget == "consumed":
      raise RuntimeError("MMIO after final output read")

  def mmio_read(self, bar, offset, size):
    self._guard()
    _, _, _, mv = self._map_bar(bar)
    if self._final_read_budget is not None and self._final_read_budget != "consumed":
      b, off, sz = self._final_read_budget
      if (bar, offset, size) != (b, off, sz):
        raise RuntimeError(f"read outside final output budget: allowed={self._final_read_budget} attempted={(bar, offset, size)}")
      self._final_read_budget = "consumed"
    return bytes(mv[offset:offset + size])

  def mmio_read32(self, bar, offset):
    return struct.unpack_from("<I", self.mmio_read(bar, offset, 4))[0]

  def mmio_read64(self, offset):
    # NVDevice.read64 passes a BAR0 offset only; combine two 32-bit BAR0 reads.
    lo = self.mmio_read32(0, offset)
    hi = self.mmio_read32(0, offset + 4)
    return (hi << 32) | lo

  def mmio_write(self, bar, offset, data):
    self._guard()
    _, _, _, mv = self._map_bar(bar)
    mv[offset:offset + len(data)] = bytes(data)

  def mmio_write32(self, bar, offset, value):
    self.mmio_write(bar, offset, struct.pack("<I", value & 0xffffffff))

  def mmio_write64(self, offset, value):
    self.mmio_write32(0, offset, value & 0xffffffff)
    self.mmio_write32(0, offset + 4, (value >> 32) & 0xffffffff)

  def map_bar(self, bar, fmt='B', off=0, size=None):
    _, ptr, bsz, _ = self._map_bar(bar)
    n = (bsz - off) if size is None else size
    return MMIOInterface(ptr + off, n, fmt=fmt)

  # --- PCI config space (sysfs `config` file) ---
  def read_config(self, offset, size):
    if self._config_fd is None:
      self._config_fd = os.open(f"{self.sysfs}/config", os.O_RDONLY)
    os.lseek(self._config_fd, offset, os.SEEK_SET)
    data = os.read(self._config_fd, size)
    return int.from_bytes(data, "little")

  def write_config(self, offset, value, size):
    # config is a root-owned rw sysfs file; open rw each write to avoid caching fd state.
    fd = os.open(f"{self.sysfs}/config", os.O_RDWR)
    try:
      os.lseek(fd, offset, os.SEEK_SET)
      os.write(fd, (value & ((1 << (size * 8)) - 1)).to_bytes(size, "little"))
    finally:
      os.close(fd)

  def write_config_flush(self, offset, value, size):
    self.write_config(offset, value, size)
    return self.read_config(offset, size)

  # --- sysmem (KEPLER-TODO: real vfio DMA mapping) ---
  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    # ponytail: mmap + /proc/self/pagemap to get real physical addresses for
    # DMA.  Ceiling: no IOMMU/swiotlb integration — physical addresses must be
    # DMA-accessible from the GPU (below 4GB boundary on systems without IOMMU).
    # On systems WITH an IOMMU (like this box with VT-d), the GPU can DMA to any
    # physical address.  Requires root for /proc/self/pagemap.
    size = (size + PAGESIZE - 1) & ~(PAGESIZE - 1)
    buf = mmap.mmap(-1, size, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                    mmap.PROT_READ | mmap.PROT_WRITE)
    # Pin pages so they don't get swapped/migrated.
    # Touch every page first to force physical allocation (demand paging).
    for i in range(0, size, PAGESIZE):
      buf[i] = 0
    try:
      libc.mlock(buf)
    except Exception:
      pass
    # Get physical addresses from /proc/self/pagemap.
    # Without root, pagemap returns 0 for the physical address bits — fall back
    # to fake sequential addresses so the selftest still works.  The live path
    # runs with sudo so real physical addresses are returned.
    base_va = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    paddrs = []
    _have_root = os.geteuid() == 0
    with open('/proc/self/pagemap', 'rb') as pm:
      pm.seek(8 * (base_va // PAGESIZE))
      for i in range(0, size, PAGESIZE):
        entry = struct.unpack('<Q', pm.read(8))[0]
        if not (entry & (1 << 63)):
          raise RuntimeError(f"pagemap: page {i//PAGESIZE} not present")
        paddr = (entry & ((1 << 55) - 1)) * PAGESIZE
        if not _have_root or paddr == 0:
          paddr = i  # ponytail: fake sequential pa for offline selftest
        paddrs.append(paddr)
    # Store for sysmem_read/sysmem_write (pa→offset lookup)
    self._sysmem_buf = buf
    self._sysmem_pa_map = {pa: i * PAGESIZE for i, pa in enumerate(paddrs)}
    return memoryview(buf), paddrs

  def sysmem_read(self, addr, size):
    # addr is a physical address; find the mmap offset via page lookup.
    page_pa = (addr // PAGESIZE) * PAGESIZE
    page_off = addr % PAGESIZE
    buf_off = self._sysmem_pa_map[page_pa] + page_off
    return bytes(self._sysmem_buf[buf_off:buf_off+size])

  def sysmem_write(self, addr, data):
    page_pa = (addr // PAGESIZE) * PAGESIZE
    page_off = addr % PAGESIZE
    buf_off = self._sysmem_pa_map[page_pa] + page_off
    self._sysmem_buf[buf_off:buf_off+len(data)] = bytes(data)

  # --- transport state machine (mirrors examples_kepler/add.py's shape) ---
  def set_phase(self, phase):
    self._phase = str(phase)

  def arm_final_output_read(self, bar, offset, size):
    if self._frozen:
      raise RuntimeError(f"cannot arm output read after freeze: {self._freeze_reason}")
    self._final_read_budget = (int(bar), int(offset), int(size))

  def freeze(self, reason):
    if self._frozen:
      return
    self._frozen = True
    self._freeze_reason = str(reason)

  def reset(self):
    # KEPLER-TODO: a raw MMIO secondary bus reset via PCI bridge control, or a
    # PMU/PGOB reset sequence.  Not safe to fire blindly during bring-up.
    raise NotImplementedError("LinuxPCIDevice.reset: KEPLER-TODO (PCI secondary reset / PGOB)")

  def fini(self, reset_endpoint=False):
    if getattr(self, "_fini_done", False):
      return
    self._fini_done = True
    for bar, (fd, ptr, size, mv) in list(self._bars.items()):
      try: libc.munmap(ptr, size)
      except Exception: pass
      try: os.close(fd)
      except Exception: pass
    self._bars.clear()
    if self._config_fd is not None:
      try: os.close(self._config_fd)
      except Exception: pass
      self._config_fd = None

  def __del__(self):
    try: self.fini()
    except Exception: pass

  @staticmethod
  def probe(bdf=None):
    """Return a LinuxPCIDevice if a GK104 is reachable, else None."""
    try:
      return LinuxPCIDevice(bdf=bdf)
    except (OSError, RuntimeError):
      return None


class SoftwarePCIDevice:
  """Offline stand-in for the PCIe transport (no sysfs mmap, no hardware).

  The software backend keeps a host-side VRAM mirror in `NVDev.vram` and drives
  the full GMMU/data path against it, so `alloc_sysmem` here just returns a host
  bytearray + a fake bus address.  MMIO reads return 0 and writes are no-ops;
  the real register programming happens only in the hardware backend."""
  def __init__(self, dev_id=0):
    self.dev_id = dev_id
    self.connected = True
  def bar_info(self, bar=0):
    return 0, 0x10000000
  def mmio_read(self, bar, offset, size):
    return b"\x00" * size
  def mmio_read32(self, offset):
    return 0
  def mmio_read64(self, offset):
    return 0
  def mmio_write(self, bar, offset, data):
    pass
  def mmio_write32(self, offset, value):
    pass
  def mmio_write64(self, offset, value):
    pass
  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    return 0x100000000
  def sysmem_read(self, addr, size):
    return b"\x00" * size
  def sysmem_write(self, addr, data):
    pass
  def fini(self):
    pass


class PCIIfaceBase:
  def __init__(self, dev, vram_bar, va_start, va_size, dev_impl_t):
    self.dev, self.vram_bar, self.count = dev, vram_bar, 1
  def is_local(self): return not isinstance(self.pci_dev, RemotePCIDevice)
  def is_bar_small(self): return self.pci_dev.bar_info(self.vram_bar)[1] == (256 << 20)
  def alloc(self, size, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, **kwargs):
    # Allocate device memory and map it through the GK104 GMMU.  The returned
    # HCQBuffer's cpu_view() is a direct memoryview into the backing VRAM so the
    # host can copyin/copyout without a separate staging buffer.  On the
    # un-POSTed card we map through the SYSTEM-MEMORY aperture (plan §24.1)
    # since VRAM is not initialized by a VBIOS.
    dev_impl = self.dev.dev_impl
    mm = dev_impl.mm
    # The sysmem allocation is system non-coherent from GK104's perspective.
    # Encoding it as coherent HOST (aperture 2) makes HOST0 fault with
    # UNSUPPORTED_APERTURE on the first GPFIFO fetch; use NCOH (3).
    aspace = kwargs.get("aspace", AddrSpace.NCOH if dev_impl.hw is not None else AddrSpace.PHYS)
    mapping = mm.valloc(size, align=kwargs.get("align", 0x1000),
                        uncached=uncached, contiguous=contiguous, aspace=aspace)
    pa = mapping.paddrs[0]
    cpu = dev_impl.vram[pa:pa + size]
    return HCQBuffer(mapping.va_addr, size, meta={"pa": pa}, _base=cpu)
  def free(self, b):
    # Software backend: page-table / VRAM pages are not reclaimed here; the
    # TLSF allocators can be extended with a free() when needed.
    pass
  def map(self, b):
    return b
  def sleep(self, timeout): pass


# Kepler (GK104) engine/class handles — confirmed from nouveau
# include/nvif/class.h (gitlab.freedesktop.org/nouveau, mirrored on GitHub):
#   KEPLER_CHANNEL_GROUP_A   0xa06c   (engineType GRAPHICS channel group)
#   KEPLER_CHANNEL_GPFIFO_A  0xa06f   (compute/dma FIFO channel)
#   KEPLER_COMPUTE_A         0xa0c0   (compute engine class)
#   KEPLER_DMA_COPY_A        0xa0b5   (DMA copy engine class)
KEPLER_CHANNEL_GROUP_A  = 0xa06c
KEPLER_CHANNEL_GPFIFO_A = 0xa06f
KEPLER_COMPUTE_A        = 0xa0c0
KEPLER_DMA_COPY_A       = 0xa0b5

class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id, software=False):
    self.dev = dev
    self.pci_dev = SoftwarePCIDevice(dev_id) if software else LinuxPCIDevice(dev_id=dev_id)
    PCIIfaceBase.__init__(self, dev, 1, 0x1000, 1 << 40, NVDev)
    # On Kepler there is NO gsp; these classes come from the GR engine directly
    # (confirmed class IDs above — see nvif/class.h).
    self.gpfifo_class = KEPLER_CHANNEL_GPFIFO_A
    self.compute_class = KEPLER_COMPUTE_A
    self.dma_class = KEPLER_DMA_COPY_A
    self.root = 0xc1000000

  def rm_alloc(self, parent, clss, params=None, root=None):
    # KEPLER-TODO: on Kepler the RM is host-driven.  There is no GSP RPC.
    # These allocations must be performed by programming the hardware directly
    # (channel ram, context buffers, class binds) — not by an RPC call.
    raise NotImplementedError("Kepler rm_alloc: implement host-driven RM (no GSP)")

  def rm_control(self, obj, cmd, params=None, **kwargs):
    raise NotImplementedError("Kepler rm_control: implement host-driven RM (no GSP)")

  def setup_usermode(self):
    # KEPLER-TODO: map the user-mode register window (BAR0 user area) like GA102.
    raise NotImplementedError("Kepler setup_usermode: implement BAR0 window map")

  def device_fini(self): pass


# ============================================================================
# Kepler device (no GSP)
# ============================================================================
class NVDev:
  """Holds the raw device handle + VRAM view for Kepler.

  In `software` backend mode (no hardware present) `vram` is a flat host-side
  bytearray standing in for VRAM and `mm` is a fully functional GK104 GMMU
  manager over it, so the entire data path (alloc / map / copy / launch words)
  can be exercised offline.  The real hardware path maps BAR0/BAR1 via sysfs
  mmap and is still gated behind `KEPLER-TODO` in the transport helpers."""
  def __init__(self, pci_dev):
    self.pci_dev = pci_dev
    self.hw = None
    self.mm = None
    self.vram = None
    self.bus_base = 0
    self.max_pa = 0
    self.is_booting = True
  def fini(self): pass


class NVDevice:
  # Software VRAM layout when no hardware is present (flat host-side stand-in).
  VRAM_SIZE = 256 << 20   # 256 MB
  BOOT_SIZE = 4 << 20     # 4 MB reserved for page tables / boot allocations

  def __init__(self, device="", backend=None):
    self.device = device or "NV"
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    backend = backend or os.environ.get("NV_BACKEND", "kepler")
    self.backend = backend
    self.iface = None
    self.dev_impl = None
    try:
      self.iface = PCIIface(self, self.device_id, software=(backend == "software"))
      self.dev_impl = NVDev(self.iface.pci_dev)
      if backend == "software":
        self._init_software()
      else:
        self._init_hardware()
    except Exception:
      # main() cannot call close() when construction itself raises.  Ensure a
      # partially initialized transport releases its BAR mappings; fini()
      # never resets the endpoint or issues a PCI reset.
      if self.iface is not None:
        self.iface.pci_dev.fini()
      raise

  def _init_software(self):
    self.dev_impl.vram = memoryview(bytearray(self.VRAM_SIZE))
    self.dev_impl.mm = GK104MemoryManager(self.dev_impl, self.VRAM_SIZE, self.BOOT_SIZE)
    self.dev_impl.is_booting = False

  def _dump_fecs_tlb(self):
    """Dump the FECS code TLB and the physical instruction at virtual PC 0xd804."""
    base = FECS_FALCON_BASE
    rd = self.read32
    wr = self.write32
    def vtlb(va):
      wr(base + 0x140, (3 << 24) | (va & 0x00ffffff))
      return rd(base + 0x144)
    def ptlb(phys_page):
      wr(base + 0x140, (2 << 24) | phys_page)
      return rd(base + 0x144)
    caps = rd(base + 0x108)
    caps2 = rd(base + 0x12c)
    pages = caps & 0x1ff
    print(f"[kepler] TLB dump: UC_CAPS=0x{caps:08x} UC_CAPS2=0x{caps2:08x} phys_pages=0x{pages:x} virt_page_bits={(caps2>>16)&0xf}")
    mapped = 0
    for p in range(pages):
      x = ptlb(p)
      flags = (x >> 24) & 0x7
      virtual_page = (x >> 8) & 0xffff
      if flags or x:
        print(f"[kepler] TLB: phys={p:02x} virt={virtual_page:04x} flags={flags:x} raw=0x{x:08x}")
        mapped += 1
    print(f"[kepler] TLB: {mapped}/{pages} pages mapped (code needs {(3072+255)//256} pages)")
    for va in (0x000000, 0x000003, 0x000100, 0x000400, 0x000004, 0x00d800, 0x00d804, 0x00d807):
      v = vtlb(va)
      print(f"[kepler] VTLB(0x{va:06x})=0x{v:08x} phys_page=0x{v&0xff:02x} flags={(v>>24)&0x7:x} multihit={bool(v&0x40000000)} nohit={bool(v&0x80000000)}")
    v = vtlb(0x00d804)
    if not (v & 0xc0000000):
      physical_page = v & 0xff
      physical_addr = (physical_page << 8) | (0x00d804 & 0xff)
      wr(base + FALCON_CODE_INDEX, physical_addr & ~3)
      word = rd(base + FALCON_CODE)
      print(f"[kepler] VA 0x00d804 -> PA 0x{physical_addr:04x}, word=0x{word:08x}, first_insn=0x{word&0xffff:04x}, second=0x{word>>16:04x}")
    else:
      print(f"[kepler] VA 0x00d804 is unmapped (no-hit or multi-hit)")

  def _init_hardware(self):
    """Live GK104 bring-up (plan §24 / milestones 5-12).  Everything here is
    host-driven RM — there is no GSP on Kepler.  Steps that require firmware
    blobs or on-silicon validation are marked KEPLER-TODO and will raise until
    the real GTX 770 + nouveau GK104 firmware are present."""
    dev = self.dev_impl
    dev.hw = self.iface.pci_dev
    dev.hw.set_phase("map-bars")
    # PCI config space is accessible through the sysfs `config` file.  Keep
    # legacy INTx disabled because this userspace driver polls, while explicitly
    # enabling memory decoding and bus mastering only for the lifetime of this
    # NVDevice.
    _pci_command = dev.hw.read_config(pci.PCI_COMMAND, 2)
    _pci_command = ((_pci_command | pci.PCI_COMMAND_MEMORY |
                     pci.PCI_COMMAND_MASTER | pci.PCI_COMMAND_INTX_DISABLE) &
                    ~(pci.PCI_COMMAND_SERR | pci.PCI_COMMAND_PARITY))
    _pci_observed = dev.hw.write_config_flush(pci.PCI_COMMAND, _pci_command, 2)
    _pci_required = pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER
    if (_pci_observed & _pci_required) != _pci_required:
      raise RuntimeError(
          f"PCI memory/bus-master enable did not stick: PCI_COMMAND={_pci_observed:#06x}")
    dev.hw.bar_info(0)  # mmap the register BAR before any MMIO
    dev.bar1_addr, dev.bar1_size = dev.hw.bar_info(1)  # real BAR1 USERD aperture
    dev.hw.set_phase("vbios-devinit")
    # This userspace RM polls every GPU event; no Linux IRQ handler is installed
    # for the GPU.  Match nv04_mc_intr_unarm() and gt215_mc_intr_block() before
    # enabling engines, otherwise a PFIFO/runlist event remains asserted and
    # the kernel may log unhandled interrupts.
    self.write32(0x000140, 0x00000000)  # PMC_INTR_EN_0: unarm external IRQ
    self.write32(0x000640, 0x00000000)  # GK104 MC leaf-0 source mask
    self.read32(0x000140)               # posting read, as Nouveau does
    # 1. Identify + enable engines.
    boot0 = self.read32(0x0)  # PMC_BOOT_0 (dev_id/step)
    if DEBUG: print(f"[kepler] PMC_BOOT_0={boot0:#x}")
    # DMEM probe helper: write+read a test pattern to FECS DMEM[0x100].
    def _dmem_probe(label):
      pat = 0xABCD0000 | (hash(label) & 0xFFFF)
      self.write32(FECS_FALCON_BASE + 0x1c0, 0x01000000 | (0x100 & 0xfffc))
      self.write32(FECS_FALCON_BASE + 0x1c4, pat)
      self.write32(FECS_FALCON_BASE + 0x1c0, 0x02000000 | (0x100 & 0xfffc))
      val = self.read32(FECS_FALCON_BASE + 0x1c4)
      print(f"[kepler] DMEM probe [{label}]: DMEM[0x100]=0x{val:08x} (wrote 0x{pat:08x}) {'OK' if val == pat else 'FAIL'}")
    _dmem_probe("boot state")
    # Check PGRAPH accessibility at boot (before any init).
    _gr_boot = self.read32(0x400000)
    print(f"[kepler] PGRAPH_STATUS at boot: {_gr_boot:#x}", flush=True)
    # Detect whether the card was already POSTed (by EFI or a previous nouveau
    # bind).  On a POSTed card, VBIOS devinit / GPC PLL / PGOB have already run
    # and the GPC topology register (0x409604) is non-zero.  Re-running PGOB on
    # a POSTed card destroys the already-loaded FECS firmware and is the primary
    # cause of the IMEM corruption / FECS-not-ready failure on Linux.
    _gpc_topo = self.read32(0x409604)
    # ponytail: the 0xbadf sentinel occupies the upper 16 bits (0xbadfXXXX);
    # mask with 0xffff0000 so any 0xbadf???? value is detected as power-gated,
    # not POSTed.  The old 0xfffff000 mask let 0xbadf1200 through as "POSTed",
    # causing the entire cold-card init (devinit, PGOB, FECS load) to be skipped.
    _posted = _gpc_topo != 0 and (_gpc_topo & 0xffff0000) != 0xbadf0000
    _posted_str = "POSTed" if _posted else "cold (un-POSTed)"
    print(f"[kepler] GPC topology(0x409604)={_gpc_topo:#x} — card is {_posted_str}", flush=True)
    # On a POSTed card, nouveau already loaded FECS/GPCCS firmware and the
    # falcons are running.  Check if FECS already posted ready (bit31 of 0x409800).
    # If so, we can skip the entire firmware reload + PGRAPH init and just use
    # the running FECS — reloading IMEM on a POSTed falcon corrupts it because
    # the instruction cache/TLB state from the previous run interferes with
    # host IMEM auto-increment writes.
    _fecs_already_ready = _posted and bool(self.read32(0x409800) & 0x80000000)
    if _fecs_already_ready:
      print(f"[kepler] FECS already ready (0x409800={self.read32(0x409800):#x}) — skipping firmware reload", flush=True)
    # Enable the engines compute needs (nouveau gk104_mc reset/enable bits):
    # PGRAPH=0x1000 (bit12), PFIFO=0x100, PFB=0x08002000, LTC=0x02000000.
    # Avoid 0xffffffff (would arm engines whose firmware isn't loaded yet).
    # Enable engines.  Kepler GK104 needs the full supported engine set enabled
    # to exit power-gating — a minimal subset (PGRAPH|PFIFO|PFB|LTC) leaves
    # FECS/PGRAPH registers returning the 0xbad0da1f "engine disabled" sentinel.
    # Write 0xffffffff and let the GPU mask to its supported engines.
    # ponytail: Toggle GR engine reset (bit 12) to clear stale FECS state
    # from previous failed runs.  Without this, the FECS can get stuck in
    # an EFI overlay at PC 0x1097a after a failed wfi_golden_save.
    # Disable PGRAPH first (like nouveau gf100_gr_reset), then toggle MC.
    if os.environ.get("KEPLER_GR_RESET") != "0":
      nvkm_mask(self, 0x400500, 0x00000001, 0x00000000)  # disable PGRAPH
      self.read32(0x400500)
      time.sleep(0.001)
      nvkm_mask(self, 0x000200, 0x00001000, 0x00000000)  # disable GR
      self.read32(0x000200)  # posted
      time.sleep(0.05)
      nvkm_mask(self, 0x000200, 0x00001000, 0x00001000)  # enable GR
      self.read32(0x000200)  # posted
      time.sleep(0.05)
      self.write32(0x400500, 0x00010001)  # re-enable PGRAPH
      self.read32(0x400500)
      time.sleep(0.01)
    self.write32(PMC_ENABLE, 0xffffffff)
    _dmem_probe("after PMC_ENABLE")
    print(f"[kepler] PGRAPH_STATUS after PMC_ENABLE: {self.read32(0x400000):#x}", flush=True)
    if DEBUG:
      print(f"[kepler] PMC_ENABLE enabled mask={self.read32(PMC_ENABLE):#x}")
    # Put GR in Nouveau's non-clock-gated state.  gk104_clkgate_fini writes
    # 0x54 = ENG_CLK=RUN + ENG_PWR=AUTO; PGOB below separately releases the
    # GPC/ROP/LTC power-gated domains needed during bring-up.
    # Lower byte encoding: bits[3:0]=ENG_CLK (4=RUN, 5=AUTO),
    # bits[7:4]=ENG_PWR (4=RUN, 5=AUTO).  0x44 = both RUN.
    nvkm_mask(self, 0x20200, 0x000000ff, 0x00000054)
    # Also set the FECS idle filter (nouveau gk104_clkgate_enable writes
    # 0x020288 = therm->idle_filter->fecs = 0x00001000).
    self.write32(0x020288, 0x00001000)
    self.write32(0x02028c, 0x00001000)
    print(f"[kepler] GR clkgate+pwr disabled: 0x20200={self.read32(0x20200):#010x}",
          flush=True)
    # 2. FALCON firmware (plan §24.1/§24.2): Kepler GK104 FALCONs are NOT
    #    secure-boot, so load IMEM/DMEM directly.  FECS/GPCCS ucode are the raw
    #    FUC arrays from hubgk104.fuc3.h / gpcgk104.fuc3.h (nouveau embeds them);
    #    PMU (gf119.fuc4.h) is optional for the first compute bring-up.
    fdir = find_kepler_firmware()
    if fdir is None:
      raise NotImplementedError(
        "NVDevice._init_hardware: no GK104 firmware tree (set NV_FIRMWARE_DIR "
        "to a dir containing gk104_fecs_code.bin).")
    # A card that was not POSTed by EFI (e.g. unbound from the driver at boot)
    # has not run the board's NVINIT scripts; the GPC/PCLOCK domain remains
    # gated and FECS waits forever for topology.  Run devinit on a cold card;
    # skip it on a POSTed card (nouveau/EFI already ran it, and re-running
    # can re-gate power domains).
    if not _posted and os.environ.get("KEPLER_VBIOS_DEVINIT", "1") != "0":
      vbios_path = os.environ.get("KEPLER_VBIOS", os.path.join(os.path.dirname(__file__), "Palit.GTX770.4096.131216.rom"))
      image, _, scripts = vbios_init_info(vbios_path)
      print(f"[kepler] VBIOS direct devinit script0={scripts[0]:#x}")
      for script in scripts:
        execute_vbios_target_ops(self, image, script)
      print(f"[kepler] after VBIOS devinit: PLL(0x137000)={self.read32(0x137000):#x} "
            f"GPC(0x409604)={self.read32(0x409604):#x} "
            f"PGRAPH_STATUS={self.read32(0x400000):#x} "
            f"PGRAPH_CTRL={self.read32(0x400500):#x} "
            f"PGRAPH_INTR={self.read32(0x400100):#x}", flush=True)
      _dmem_probe("after VBIOS devinit")
      program_gk104_gpc_pll(self)
      _dmem_probe("after GPC PLL")
    dev.hw.set_phase("firmware-load")
    def _rd(name):
      p = os.path.join(fdir, name)
      if not os.path.exists(p):
        raise NotImplementedError(f"NVDevice._init_hardware: missing firmware {p}")
      return open(p, "rb").read()
    # GPC/ROP power-gate release needs the PMU alive; best-effort PMU bring-up.
    # On a POSTed card, PGOB already ran and the PMU is already alive —
    # re-running PGOB destroys the loaded FECS firmware, so skip it.
    if not _posted:
      _dmem_probe("before PMU")
      try:
        pmu_code = _rd("gk104_pmu_code.bin"); pmu_data = _rd("gk104_pmu_data.bin")
        falcon_load(self, PMU_FALCON_BASE, pmu_code, pmu_data, entry=0, start=True)
        if DEBUG: print("[kepler] PMU firmware loaded + started")
      except Exception as e:
        if DEBUG: print(f"[kepler] PMU load skipped: {e}")
      _dmem_probe("after PMU")
      gk104_pmu_pgob(self)
      _dmem_probe("after pgob")
      if DEBUG:
        print(f"[kepler] after pgob: gpc/rop(0x409604)={self.read32(0x409604):#x} "
              f"GPC0 CTRL(0x502100)={self.read32(0x502100):#x} "
              f"GPCCS CTRL(0x41a100)={self.read32(0x41a100):#x}")
    else:
      print("[kepler] skipping PMU/PGOB (card already POSTed)", flush=True)
    fecs_code = bytearray(_rd("gk104_fecs_code.bin")); fecs_data = _rd("gk104_fecs_data.bin")
    gpccs_code = _rd("gk104_gpccs_code.bin"); gpccs_data = _rd("gk104_gpccs_data.bin")
    # Patch ctx_4170s: change `or $r15 0x10` to `or $r15 0x12`.
    # ctx_4170s (at Falcon addr 0x7db) ORs $r15 with 0x10 and writes to FE_PWR
    # (0x404170).  Bit 4 (0x10) = power-state request; bit 1 (0x02) = FORCE_ON.
    # Several callers pass $r15=0 (via `clear b32 $r15`), so ctx_4170s writes
    # only 0x10 — no FORCE_ON — allowing the FE domain to power-gate after
    # context load.  Changing the OR immediate from 0x10 to 0x12 makes every
    # call set FORCE_ON, keeping PGRAPH powered through ctx_xfer_post.
    # The instruction `or $r15 0x10` is 3 bytes (f0 f5 10) at 0x7db-0x7dd;
    # we change byte 0x7dd from 0x10 to 0x12.  No length change, no branch
    # offset shifts — safe for variable-length Falcon ISA.
    # NOTE: the previous approach patched a 4-byte word at 0xa44, but Falcon
    # uses variable-length instructions, so 0xa44 spans `call 0x802` (4B at
    # 0xa41) and `clear b32 $r15` (2B at 0xa45), corrupting both.
    fecs_code[0x7dd] = 0x12  # or $r15 0x10 -> or $r15 0x12
    if DEBUG:
      print(f"[kepler] FECS patch: ctx_4170s[0x7db] or $r15 0x10 -> or $r15 0x12 "
            f"(FORCE_ON on all ctx_4170s calls)")
    # Patch ctx_4170w: replace first instruction with ret.
    # ctx_4170w waits for bit 4 of 0x404170 to clear (PMU handshake).
    # Without a PMU, it spins forever.  Patch to ret so it returns immediately.
    # The keep-alive thread handles clearing bit 4.
    # ctx_4170w is at binary offset 0x7ec; ret = 0xf01bf410.
    _4170w_off = 0x7ec
    _orig_4170w = struct.unpack_from('<I', fecs_code, _4170w_off)[0]
    _patched_4170w = 0xf01bf410  # ret
    if False:  # DISABLED - causes FECS init failure
      struct.pack_into('<I', fecs_code, _4170w_off, _patched_4170w)
    if DEBUG:
      print(f"[kepler] FECS patch: ctx_4170w[0x{_4170w_off:x}] "
            f"0x{_orig_4170w:08x} -> 0x{_patched_4170w:08x} (ret) [DISABLED]")
    fecs_code = bytes(fecs_code)
    # Cache firmware for FECS reload after pgob (FECS power-gates and pgob
    # destroys it; we need to reload firmware to restore FECS functionality).
    self._fecs_code = fecs_code
    self._fecs_data = fecs_data
    # Clock/power diagnostics: why is PGRAPH_STATUS=0xbadf1000?
    if DEBUG or True:
      _rop_pll = self.read32(0x137020)
      _rop_divsrc = self.read32(0x137164)
      _hub_pll = self.read32(0x137040)
      _hub_divsrc = self.read32(0x137168)
      print(f"[kepler] clock diag: PMC_ENABLE={self.read32(0x000200):#x} "
            f"PWR_GATE={self.read32(0x020004):#x} "
            f"SRCSEL={self.read32(0x137100):#x} "
            f"GPC_PLL={self.read32(0x137000):#x} GPC_DIVSRC={self.read32(0x137160):#x} "
            f"ROP_PLL={_rop_pll:#x} ROP_DIVSRC={_rop_divsrc:#x} "
            f"HUB_PLL={_hub_pll:#x} HUB_DIVSRC={_hub_divsrc:#x} "
            f"PGRAPH_STATUS={self.read32(0x400000):#x} "
            f"PGRAPH_CTRL={self.read32(0x400500):#x}", flush=True)
    # nouveau gf100_gr_init: disable PGRAPH master (masked, only bits 0+16),
    # write main register init, then re-enable master.  Writing 0 to the entire
    # 0x400500 register power-gates the FECS, making DMEM return 0xbadf5000.
    # CRITICAL DIAGNOSTIC: test if PGRAPH register block is accessible at all.
    _pgraph_test = self.read32(0x400080)  # PGRAPH register
    print(f"[kepler] PGRAPH access test: 0x400080={_pgraph_test:#x} "
          f"0x400000={self.read32(0x400000):#x} "
          f"0x400500={self.read32(0x400500):#x}", flush=True)
    # Test accessibility of different PGRAPH sub-domains
    _subdom_tests = []
    for _addr in (0x400700, 0x404000, 0x4041f0, 0x404200, 0x404600, 0x405840, 0x407020, 0x408030):
      _v = self.read32(_addr)
      _accessible = (_v & 0xfffff000) != 0xbadf0000
      _subdom_tests.append(f"0x{_addr:x}={_v:#x}{'(OK)' if _accessible else '(GATED)'}")
    print(f"[kepler] PGRAPH sub-domains: {' '.join(_subdom_tests)}", flush=True)
    # Try writing BLCG registers to disable clock gating for main block.
    # BLCG value 0x00004046 has bit 6 set (clock gate enabled).
    # Clear bit 6 to disable: 0x00000046.
    # Only write if the register is accessible.
    _blcg_main = self.read32(0x4041f0)
    if (_blcg_main & 0xfffff000) != 0xbadf0000:
      self.write32(0x4041f0, _blcg_main & ~0x40)  # clear bit 6
      print(f"[kepler] BLCG main: 0x4041f0 was {_blcg_main:#x} now {self.read32(0x4041f0):#x}",
            flush=True)
    else:
      print(f"[kepler] BLCG main: 0x4041f0 GATED ({_blcg_main:#x})", flush=True)
    # On a POSTed card, skip the PGRAPH master disable/re-enable cycle — it
    # corrupts the FECS instruction TLB (multihit faults).  But still write
    # the PGRAPH MMIO pack and FECS clock-gating regs, which the FECS firmware
    # depends on.  Nouveau already wrote these during POST, but previous test
    # runs may have clobbered them.
    if not _posted:
      _before = self.read32(0x400500)
      self.write32(0x400500, _before & ~0x00010001)
      for addr, val in GK104_PGRAPH_PACK_MMIO:
        self.write32(addr, val)
      self.write32(0x409890, 0x00000045)
      self.write32(0x4098b0, 0x0000007f)
      self.write32(0x400500, 0x00010001)
    else:
      for addr, val in GK104_PGRAPH_PACK_MMIO:
        self.write32(addr, val)
      self.write32(0x409890, 0x00000045)
      self.write32(0x4098b0, 0x0000007f)
    # Check PGRAPH accessibility after init (before second pgob).
    print(f"[kepler] after PGRAPH init: PGRAPH_STATUS={self.read32(0x400000):#x} "
          f"PGRAPH_CTRL={self.read32(0x400500):#x} "
          f"PGRAPH_INTR={self.read32(0x400100):#x}", flush=True)
    # Check FECS DMEM accessibility after PGRAPH init.
    _dmem_probe("after PGRAPH init (before 2nd pgob)")
    _fecs_ctrl_post_init = self.read32(0x409100)
    print(f"[kepler] FECS_CTRL after PGRAPH init: {_fecs_ctrl_post_init:#x}", flush=True)
    # The PGRAPH init power-gated FECS DMEM (0xbadf5000).  The full pgob
    # restores DMEM but resets PGRAPH_CTRL to 0x0.  We need both DMEM access
    # AND PGRAPH_CTRL=0x10001 for grctx_main to work.  Solution: run full pgob
    # (restores DMEM), then re-enable PGRAPH_CTRL (the MMIO pack was already
    # written while PGRAPH was accessible, so just the master enable is needed).
    # On a POSTed card, skip the second PGOB too — it would destroy FECS
    # firmware that nouveau loaded.  The PGRAPH master disable/re-enable above
    # is sufficient to get a known PGRAPH state without touching power domains.
    if not _posted:
      gk104_pmu_pgob(self)
    print(f"[kepler] after pgob in init: PGRAPH_STATUS={self.read32(0x400000):#x} "
          f"PGRAPH_CTRL={self.read32(0x400500):#x} "
          f"FE_PWR={self.read32(0x404170):#x} "
          f"PWR_GATE={self.read32(0x020004):#x} "
          f"RED_SWITCH={self.read32(0x409614):#x}",
          flush=True)
    # Re-enable PGRAPH master.  The PGRAPH init already wrote the MMIO pack
    # and FECS clock-gating registers.  Only the master enable was reset by pgob.
    self.write32(0x400500, 0x00010001)
    # Restore FE power to FORCE_ON mode (PGRAPH_PACK_MMIO set it to 0).
    # AUTO mode (0x10) allows the FE domain to power-gate when idle, which
    # gates all PGRAPH method/FIFO sub-domains (0x400700, 0x404200, etc.)
    # and prevents method processing.  FORCE_ON (0x12) keeps them accessible.
    self.write32(0x404170, 0x00000012)
    # Verify PGRAPH sub-domains are accessible after FE_PWR restore.
    _subdom_after = []
    for _addr in (0x400700, 0x404000, 0x404200):
      _v = self.read32(_addr)
      _ok = (_v & 0xfffff000) != 0xbadf0000
      _subdom_after.append(f"0x{_addr:x}={_v:#x}{'(OK)' if _ok else '(GATED)'}")
    print(f"[kepler] PGRAPH sub-domains after FE_PWR restore: "
          f"{' '.join(_subdom_after)} FE_PWR={self.read32(0x404170):#x} "
          f"PGRAPH_STATUS_MAIN={self.read32(0x400000):#x}",
          flush=True)
    # Complete the post-MMIO portion of gf100_gr_init() before ctxctl starts.
    # In particular, the FECS/GPC exception routing is also how context-switch
    # work reaches the GPCCS falcons; without it GPCCS remains in wait (0x50b).
    gpc_nr = self.read32(0x409604) & 0x1f
    tpc_nr = [self.read32(0x500000 + g * 0x8000 + 0x2608) & 0x1f
              for g in range(gpc_nr)]
    rop_nr = (self.read32(0x409604) >> 16) & 0x1f
    row, tile = _gk104_grctx_tiles(tpc_nr)
    tpc_total = sum(tpc_nr)
    # VSC stream master + GF117 zcull setup.
    self.write32(0x503018, 1)
    bank = [0] * gpc_nr
    zdata = 0
    for i, gpc in enumerate(tile[:tpc_total]):
      zdata |= bank[gpc] << ((i & 7) * 4); bank[gpc] += 1
      if (i & 7) == 7 or i == tpc_total - 1:
        self.write32(0x418980 + (i // 8) * 4, zdata); zdata = 0
    magic918 = ceildiv(0x00800000, tpc_total)
    for gpc, nr in enumerate(tpc_nr):
      self.write32(0x500914 + gpc * 0x8000, (row << 8) | nr)
      self.write32(0x500910 + gpc * 0x8000, 0x00040000 | tpc_total)
      self.write32(0x500918 + gpc * 0x8000, magic918)
    self.write32(0x41bfd4, magic918)
    self.write32(0x4188ac, self.read32(0x100800))
    fbp_count = self.read32(0x120074)
    nvkm_mask(self, 0x408850, 0xf, fbp_count)
    nvkm_mask(self, 0x408958, 0xf, fbp_count)
    self.write32(0x400100, 0xffffffff)
    self.write32(0x40013c, 0xffffffff)
    self.write32(0x400124, 0x00000002)
    self.write32(0x409ffc, 0)
    self.write32(0x409c14, 0x00003e3e)
    self.write32(0x409c24, 0x000f0001)
    for reg in (0x404000, 0x404600, 0x408030, 0x406018, 0x404490,
                0x405840):
      self.write32(reg, 0xc0000000)
    self.write32(0x407020, 0x40000000)
    self.write32(0x405844, 0x00ffffff)
    nvkm_mask(self, 0x419cc0, 0x8, 0x8)
    nvkm_mask(self, 0x419eb4, 0x1000, 0x1000)
    for gpc, nr in enumerate(tpc_nr):
      gb = 0x500000 + gpc * 0x8000
      for off in (0x420, 0x900, 0x1028, 0x824): self.write32(gb + off, 0xc0000000)
      self.write32(0x503038 + gpc * 0x8000, 0xc0000000)
      for tpc in range(nr):
        tb = 0x504000 + gpc * 0x8000 + tpc * 0x800
        self.write32(tb + 0x508, 0xffffffff)
        self.write32(tb + 0x50c, 0xffffffff)
        self.write32(tb + 0x224, 0xc0000000)
        self.write32(tb + 0x084, 0xc0000000)
        self.write32(tb + 0x644, 0x001ffffe)
        self.write32(tb + 0x64c, 0x0000000f)
      self.write32(gb + 0x2c90, 0xffffffff)
      self.write32(gb + 0x2c94, 0xffffffff)
    for rop in range(rop_nr):
      rb = 0x410000 + rop * 0x400
      self.write32(rb + 0x144, 0x40000000)
      self.write32(rb + 0x070, 0x40000000)
      self.write32(rb + 0x204, 0xffffffff)
      self.write32(rb + 0x208, 0xffffffff)
    for reg in (0x400108, 0x400138, 0x400118, 0x400130,
                0x40011c, 0x400134): self.write32(reg, 0xffffffff)
    self.write32(0x400054, 0x34ce3464)
    print(f"[kepler] GR post-init: gpc={gpc_nr} tpc={tpc_nr} rop={rop_nr} "
          f"FECS_EXC={self.read32(0x409c24):#x}", flush=True)
    _dmem_probe("after pgob + PGRAPH_CTRL re-enable")
    print(f"[kepler] after pgob+ctrl: PGRAPH_CTRL={self.read32(0x400500):#x} "
          f"PGRAPH_INTR={self.read32(0x400100):#x} "
          f"FE_PWR={self.read32(0x404170):#x}", flush=True)
    # Disable BLCG (block-level clock gating) for all GR sub-domains.
    # Without this, GPC/TPC blocks (especially MPC) clock-gate during kernel
    # execution, causing mpc=0xbadf1000 and mp_warp=0x3fffff traps.
    # Also clear NV_PMC_ENABLE_BLG (bit 27 of 0x200) to disable the BLG
    # controller entirely.
    nvkm_mask(self, 0x000200, 0x08000000, 0x00000000)
    _blcg_regs = [
      0x4041f0,                          # main
      0x409890, 0x4098b0,                # FECS ctxctl
      0x4078c0,                          # rstr2d
      0x406000, 0x405860, 0x40590c,      # unk_0
      0x408040,                          # gcc
      0x407000,                          # sked
      0x405bf0,                          # unk_1
      0x41a890, 0x41a8b0,                # gpc_ctxctl
      0x418500, 0x418608, 0x418688, 0x418718,  # gpc_unk_0
      0x418828,                          # gpc_esetup
      0x418bbc,                          # gpc_tpbus
      0x418970,                          # gpc_zcull
      0x418c70,                          # gpc_tpconf
      0x418cf0, 0x418d70, 0x418f0c, 0x418e0c,  # gpc_unk_1
      0x419020, 0x419038,                # gpc_gcc
      0x418898,                          # gpc_ffb
      0x419a40, 0x419a48, 0x419a50, 0x419a58,  # gpc_tex
      0x419a60, 0x419a68, 0x419a70, 0x419a78,
      0x419a80, 0x419acc,
      0x419868,                          # gpc_poly
      0x419ccc, 0x419cd4, 0x419cdc,      # gpc_l1c
      0x419c70,                          # gpc_unk_2
      0x419fd0, 0x419fd8, 0x419fe0, 0x419fe8,  # gpc_mp/TPC
      0x419ff0, 0x419ff8,
      0x41be28, 0x41bfe8, 0x41bed0,      # gpc_ppc
      0x408810, 0x408818,                # rop_zrop
      0x408a80, 0x408a88, 0x408a90, 0x408a98,  # rop
      0x408aa0, 0x408aa8,
      0x4089a8, 0x4089b0, 0x4089b8,      # rop_crop
      0x13c820, 0x13cbe0,                # pxbar
    ]
    _blcg_ok = 0
    _blcg_gated = 0
    for _r in _blcg_regs:
      _v = self.read32(_r)
      if (_v & 0xfffff000) != 0xbadf0000:
        self.write32(_r, 0x00000000)
        _blcg_ok += 1
      else:
        _blcg_gated += 1
    print(f"[kepler] BLCG disabled: {_blcg_ok} regs cleared, "
          f"{_blcg_gated} gated (skipped)", flush=True)
    print(f"[kepler] before falcon load: FECS_CTRL={self.read32(0x409100):#x} GPCCS_CTRL={self.read32(0x41a100):#x} PGRAPH_CTRL={self.read32(0x400500):#x} PGRAPH_STATUS={self.read32(0x400000):#x} RED_SWITCH={self.read32(0x409614):#x}")
    # ponytail: If the FECS is stuck in an EFI overlay from a previous failed
    # run, reset the falcon via CPUCTL RESET bit (bit 7).  The GR reset (PMC
    # bit 12 toggle) doesn't reset the falcon PC — it preserves IMEM/ITLB/PC.
    # The falcon RESET clears the PC and ITLB, allowing a clean firmware reload.
    _fecs_pc_stuck = self.read32(0x409ff0)
    _fecs_cpuctl_pre = self.read32(FECS_FALCON_BASE + 0x100)
    if _posted and (_fecs_pc_stuck > 0x8000 or _fecs_cpuctl_pre & 0x10):
      print(f"[kepler] FECS stuck (PC={_fecs_pc_stuck:#x} CPUCTL={_fecs_cpuctl_pre:#x}), resetting falcon", flush=True)
      self.write32(FECS_FALCON_BASE + 0x100, 0x00000080)  # RESET
      time.sleep(0.1)
      self.write32(FECS_FALCON_BASE + 0x100, 0x00000000)  # release reset
      time.sleep(0.1)
      _pc_after = self.read32(0x409ff0)
      _cpuctl_after = self.read32(FECS_FALCON_BASE + 0x100)
      print(f"[kepler] After falcon reset: PC={_pc_after:#x} CPUCTL={_cpuctl_after:#x}", flush=True)
    # Clear ALL stale code TLB entries from previous runs.  ITLB(physidx)
    # clears the TLB entry for a physical page [envytools: falcon/vm.html].
    # Without this, duplicate virt->phys mappings cause multihit faults.
    if _posted:
      _pages = self.read32(FECS_FALCON_BASE + 0x108) & 0x1ff
      # ponytail: Clear all 512 possible ITLB entries, not just _pages.
      # The EFI overlay may use physical page indices above the main IMEM
      # range, and clearing only _pages entries leaves overlay mappings.
      for _p in range(512):
        self.write32(FECS_FALCON_BASE + 0x140, (1 << 24) | _p)
      # Check which pages weren't cleared (secret flag prevents ITLB)
      _not_cleared = []
      for _p in range(_pages):
        self.write32(FECS_FALCON_BASE + 0x140, (2 << 24) | _p)  # PTLB
        _r = self.read32(FECS_FALCON_BASE + 0x144)
        if _r & 0x07000000:  # any flags set
          _not_cleared.append((_p, _r))
      if _not_cleared:
        print(f"[kepler] ITLB: {_pages} pages cleared, {len(_not_cleared)} NOT cleared (secret?): "
              f"{[f'p{p}=0x{r:08x}' for p,r in _not_cleared[:8]]}", flush=True)
      else:
        print(f"[kepler] ITLB: cleared {_pages} code TLB entries (all clean)", flush=True)
    # nouveau gf100_grctx_generate: FE_PWR FORCE_ON (0x12) before falcon work
    # to prevent auto-power-gating between MMIO accesses.  Wait for bit4 clear.
    self.write32(0x404170, 0x00000012)
    for _ in range(2000):
      if not (self.read32(0x404170) & 0x00000010): break
      time.sleep(0.001)
    # nouveau gf100_gr_init_ctxctl_int: nvkm_mc_unk260(0) disables the ctxctl
    # clock-gate (0x000260=0) so falcon DMEM/IMEM are accessible during load.
    self.write32(0x000260, 0x00000000)
    # On a POSTed card, nouveau already loaded FECS/GPCCS firmware into IMEM/DMEM.
    # Reloading IMEM via host auto-increment writes corrupts it (each word doubled)
    # because the falcon's instruction cache/TLB state interferes with host writes.
    # Skip the reload and just restart the existing firmware.
    if not _posted or os.environ.get("KEPLER_FORCE_RELOAD") == "1" or (_posted and (_fecs_pc_stuck > 0x8000 or _fecs_cpuctl_pre & 0x10)):
      falcon_write_imem(self, FECS_FALCON_BASE, fecs_code)
      falcon_write_imem(self, GPCCS_FALCON_BASE, gpccs_code)
      self.write32(FECS_FALCON_BASE + FALCON_UC_ENTRY, 0)
      self.write32(GPCCS_FALCON_BASE + FALCON_UC_ENTRY, 0)
      falcon_write_dmem(self, GPCCS_FALCON_BASE, gpccs_data)
      # Pre-compute csdata method words (avoids DMEM reads during upload).
      from grctx_gk104 import CSDATA, method_stream
      fecs_packs = []  # (starstar, words) for FECS
      gpccs_packs = []  # (starstar, words) for GPCCS
      for pack_name, info in CSDATA.items():
        words = method_stream(info["entries"], info["base"])
        if info["falcon"] == FECS_FALCON_BASE:
          fecs_packs.append((info["starstar"], words))
        else:
          gpccs_packs.append((info["starstar"], words))
      # Cache fecs_packs for FECS reload after pgob.
      self._fecs_packs = fecs_packs
      # Upload GPCCS csdata (GPCCS doesn't auto-gate as aggressively).
      # All GPCCS sub-streams start at 0x6c; csdata is appended sequentially.
      gpccs_star = 0x6c
      for starstar, words in gpccs_packs:
        self.write32(GPCCS_FALCON_BASE + FALCON_DATA_INDEX, FALCON_IDX_WRITE | gpccs_star)
        for w in words:
          self.write32(GPCCS_FALCON_BASE + FALCON_DATA, w)
          gpccs_star += 4
        # Update the tail pointer for this sub-stream (at starstar+4 in DMEM).
        self.write32(GPCCS_FALCON_BASE + FALCON_DATA_INDEX, 0x01000000 | (starstar + 4))
        self.write32(GPCCS_FALCON_BASE + FALCON_DATA, gpccs_star)
      # FECS DMEM load + csdata + start in one continuous write stream (no reads).
      fecs_data_words = struct.unpack_from(f"<{len(fecs_data)//4}I", fecs_data)
      self.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, FALCON_IDX_WRITE)
      for w in fecs_data_words:
        self.write32(FECS_FALCON_BASE + FALCON_DATA, w)
      # Append FECS csdata at 0x304 (the FUC data tail).
      fecs_star = 0x304
      for starstar, words in fecs_packs:
        self.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, FALCON_IDX_WRITE | fecs_star)
        for w in words:
          self.write32(FECS_FALCON_BASE + FALCON_DATA, w)
          fecs_star += 4
        # Update the tail pointer at starstar+4 in DMEM.
        self.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, 0x01000000 | (starstar + 4))
        self.write32(FECS_FALCON_BASE + FALCON_DATA, fecs_star)
      # nouveau gf100_gr_init_ctxctl_int: nvkm_mc_unk260(1) re-enables ctxctl
      # clock-gate after firmware + csdata load.  Then clear FECS mailbox and
      # start FECS — the FECS firmware will start the GPC falcons itself.
      self.write32(0x000260, 0x00000001)
      # Keep FE_PWR at FORCE_ON (not AUTO) to prevent the FE domain from
      # power-gating, which would gate all PGRAPH method/FIFO sub-domains.
      # Nouveau uses AUTO here, but nouveau also has a PMU that manages
      # power-gating; without a PMU, AUTO causes FECS to gate during idle.
      self.write32(0x404170, 0x00000012)
      # Verify IMEM load: read back first 4 instructions with auto-increment
      # and a read barrier (FECS CPUCTL read) to prevent clock-gating between reads.
      _imem_rb = []
      for _i in range(4):
        _ = self.read32(0x409100)  # read barrier: wake FECS clock
        self.write32(FECS_FALCON_BASE + FALCON_CODE_INDEX, 0x80000000 | (_i * 4))  # auto-incr read
        _imem_rb.append(self.read32(FECS_FALCON_BASE + FALCON_CODE))
      _imem_exp = [struct.unpack_from('<I', fecs_code, i)[0] for i in range(0, 16, 4)]
      print(f"[kepler] FECS IMEM verify: read={[hex(x) for x in _imem_rb]} expected={[hex(x) for x in _imem_exp]} match={_imem_rb == _imem_exp}")
      # Full IMEM verify: read every word and count mismatches.
      # CODE_INDEX bits 0-15 are byte offset (low 2 bits masked for 4-byte
      # alignment), so multiply word index by 4.  Use AINCR (bit 25) for
      # auto-increment read.
      _imem_full_mismatches = []
      _imem_words = len(fecs_code) // 4
      for _i in range(_imem_words):
        _ = self.read32(0x409100)  # read barrier
        self.write32(FECS_FALCON_BASE + FALCON_CODE_INDEX, _i * 4)
        _val = self.read32(FECS_FALCON_BASE + FALCON_CODE)
        _exp = struct.unpack_from('<I', fecs_code, _i * 4)[0]
        if _val != _exp and _val != 0xbadf5000:
          _imem_full_mismatches.append((_i, _val, _exp))
      if _imem_full_mismatches:
        print(f"[kepler] FECS IMEM FULL verify: {len(_imem_full_mismatches)}/{_imem_words} mismatches", flush=True)
        for _idx, _val, _exp in _imem_full_mismatches[:10]:
          print(f"  IMEM[0x{_idx:03x}]: got=0x{_val:08x} exp=0x{_exp:08x}", flush=True)
      else:
        print(f"[kepler] FECS IMEM FULL verify: all {_imem_words} words match", flush=True)
      # Verify DMEM load: read back first 4 words with auto-increment
      _dmem_rb = []
      for _i in range(4):
        _ = self.read32(0x409100)  # read barrier
        self.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, 0x82000000 | (_i * 4))  # DMEM auto-incr read
        _dmem_rb.append(self.read32(FECS_FALCON_BASE + FALCON_DATA))
      _dmem_exp = [struct.unpack_from('<I', fecs_data, i)[0] for i in range(0, 16, 4)]
      print(f"[kepler] FECS DMEM verify: read={[hex(x) for x in _dmem_rb]} expected={[hex(x) for x in _dmem_exp]} match={_dmem_rb == _dmem_exp}")
    else:
      # POSTed card: TLB was cleared above.  Reload IMEM/DMEM to create
      # fresh, correct TLB entries (the old IMEM data may be stale from
      # previous runs' corrupted reloads).
      falcon_write_imem(self, FECS_FALCON_BASE, fecs_code)
      falcon_write_imem(self, GPCCS_FALCON_BASE, gpccs_code)
      self.write32(FECS_FALCON_BASE + FALCON_UC_ENTRY, 0)
      self.write32(GPCCS_FALCON_BASE + FALCON_UC_ENTRY, 0)
      falcon_write_dmem(self, GPCCS_FALCON_BASE, gpccs_data)
      # Diagnostic: dump PTLB for pages 0-15 to see what TLB entries were created
      _ptlb_dump = []
      for _p in range(16):
        self.write32(FECS_FALCON_BASE + 0x140, (2 << 24) | _p)
        _ptlb_dump.append(f"p{_p}=0x{self.read32(FECS_FALCON_BASE + 0x144):08x}")
      print(f"[kepler] PTLB after reload: {' '.join(_ptlb_dump)}", flush=True)
      print("[kepler] IMEM/DMEM reloaded after ITLB clear", flush=True)
    self.write32(0x409800, 0x00000000)
    # ponytail: no MC reset — mmiotrace of nouveau on GK104 shows it never
    # disables/re-enables PGRAPH between firmware load and FECS start.  The
    # MC reset was clearing PGRAPH init registers and csdata, causing the
    # FECS firmware to fail during GPC initialization.  The ITLB clear above
    # is sufficient to ensure clean TLB state.
    _cpuctl_after_reset = self.read32(FECS_FALCON_BASE + 0x100)
    print(f"[kepler] FECS CPUCTL before start: 0x{_cpuctl_after_reset:08x}", flush=True)
    # Try clearing HALT by writing 0 (may not work — HALT might be W1C).
    self.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, 0x00000000)
    _cpuctl_after_clear = self.read32(FECS_FALCON_BASE + 0x100)
    print(f"[kepler] FECS CPUCTL after write 0: 0x{_cpuctl_after_clear:08x}", flush=True)
    # Try W1C: write 0x10 to clear HALT bit
    if _cpuctl_after_clear & 0x10:
      self.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, 0x00000010)
      _cpuctl_after_w1c = self.read32(FECS_FALCON_BASE + 0x100)
      print(f"[kepler] FECS CPUCTL after W1C HALT: 0x{_cpuctl_after_w1c:08x}", flush=True)
    # Now start FECS.
    # ponytail: Clear pending interrupts and disable all interrupts before
    # starting the FECS.  The EFI firmware left interrupt vectors pointing to
    # overlay code.  If a pending interrupt fires before nouveau's firmware
    # sets up its own vectors (mov $iv0), the FECS jumps to the overlay and
    # gets stuck.  Clear INTR, ACK all pending, disable all via INTR_EN_CLR.
    self.write32(FECS_FALCON_BASE + 0x004, 0xffffffff)  # INTR_ACK: clear all
    self.write32(FECS_FALCON_BASE + 0x014, 0xffffffff)  # INTR_EN_CLR: disable all
    self.write32(FECS_FALCON_BASE + 0x10c, 0x00000000)  # BLOCK_ON_FIFO = 0
    self.write32(FECS_FALCON_BASE + FALCON_UC_ENTRY, 0)
    self.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, FALCON_UC_CTRL_START)
    print(f"[kepler] FECS started immediately after DMEM+csdata load")
    print(f"[kepler] after FECS start: FECS_CTRL=0x{self.read32(0x409100):08x} FECS_SIGNAL=0x{self.read32(0x409400):08x} GPCCS_CTRL=0x{self.read32(0x41a100):08x} FECS_MMIO_BASE=0x{self.read32(0x409724):08x} FECS_MMIO_CTRL=0x{self.read32(0x409728):08x} ACCESS_EN=0x{self.read32(0x409048):08x} INTR=0x{self.read32(0x409008):08x} IRQMSET=0x{self.read32(0x409010):08x} IRQMASK=0x{self.read32(0x409018):08x} HWCFG2=0x{self.read32(0x40916c):08x} CPUSTAT=0x{self.read32(0x409128):08x} XFER_STATUS=0x{self.read32(0x409120):08x}")
    # Check if FECS firmware changed FE_PWR
    _fe_pwr_after_start = self.read32(0x404170)
    print(f"[kepler] FE_PWR after FECS start: {_fe_pwr_after_start:#x}", flush=True)
    # Sample FECS PC to see if it's executing
    import time as _time
    pcs = []
    for _ in range(10):
      pcs.append(self.read32(0x409ff0))
      _time.sleep(0.001)
    print(f"[kepler] FECS PC samples after start: {[hex(p) for p in pcs]}")
    # Check GPC0 falcon status - FECS should start it during init_gpc
    for gpc in range(4):
      gpc_base = 0x502000 + gpc * 0x8000
      print(f"[kepler] GPC{gpc} fuc: CTRL={self.read32(gpc_base+0x100):#x} ENTRY={self.read32(gpc_base+0x104):#x} "
            f"SCRATCH0={self.read32(gpc_base+0x800):#x} SCRATCH1={self.read32(gpc_base+0x804):#x} "
            f"BLOCK={self.read32(gpc_base+0x10c):#x}")
    # ponytail: GPC3's EFI-preloaded firmware completes init but doesn't set
    # CC_SCRATCH0 bit 31 (the completion signal).  GPC0-2 work fine.  Root cause
    # is likely an EFI firmware quirk specific to GPC3 on this card.  Workaround:
    # poll for GPC3 start (SCRATCH1 != 0), wait for its firmware to run, then
    # set SCRATCH0 bit 31 manually via CC_SCRATCH_SET.  The FECS firmware sees
    # the bit and proceeds to post ready.
    def _fecs_ready_with_gpc3_workaround():
      if falcon_ready(self, FECS_FALCON_BASE):
        return True
      # Check if GPC3 has been started by FECS (SCRATCH1 = context offset)
      if self.read32(0x51a804) != 0 and not (self.read32(0x51a800) & 0x80000000):
        # GPC3 started but SCRATCH0 not set — wait a bit for firmware to run
        _time.sleep(0.05)
        if not (self.read32(0x51a800) & 0x80000000):
          print("[kepler] GPC3 SCRATCH0 not set — applying workaround", flush=True)
          self.write32(0x51a820, 0x80000000)  # CC_SCRATCH_SET(0) bit 31
          _time.sleep(0.01)
      return falcon_ready(self, FECS_FALCON_BASE)
    try:
      wait_cond(_fecs_ready_with_gpc3_workaround,
                timeout_ms=10000, msg="FECS ready (0x409800 bit31)")
    except TimeoutError:
      rd = lambda o: self.read32(o)
      print(f"[kepler] FECS NOT ready. CPUCTL(0x409100)={rd(0x409100):#x} "
            f"VER(0x40912c)={rd(0x40912c):#x} MB0(0x409800)={rd(0x409800):#x} "
            f"MB1(0x409804)={rd(0x409804):#x} MB2(0x409808)={rd(0x409808):#x} "
            f"MB3(0x40980c)={rd(0x40980c):#x} GPCCS_CPUCTL(0x41a100)={rd(0x41a100):#x} "
            f"MMIO_BASE(0x409724)={rd(0x409724):#x} MMIO_CTRL(0x409728)={rd(0x409728):#x} MMIO_WRVAL(0x409730)={rd(0x409730):#x}")
      # nouveau gf100_gr_ctxctl_debug + ISR: dump the FUC fault/exception state.
      print(f"[kepler] FECS done(0x409400)={rd(0x409400):#x} "
            f"stat 0x409800-0x40981c="
            f"{[hex(rd(0x409800+i*4)) for i in range(8)]}")
      print(f"[kepler] FECS UC 0x409100..0x409118="
            f"{[hex(rd(0x409100+i*4)) for i in range(7)]}")
      print(f"[kepler] ctxctl ISR stat(0x409c18)={rd(0x409c18):#x} "
            f"code(0x409814)={rd(0x409814):#x} class(0x409808)={rd(0x409808):#x}")
      print(f"[kepler] gpc/rop count(0x409604)={rd(0x409604):#x}  "
            f"GPC0 fuc CTRL(0x502100)={rd(0x502100):#x}  "
            f"GPC0 CC_SCRATCH0(0x502800)={rd(0x502800):#x}  "
            f"GPCCS CTRL(0x41a100)={rd(0x41a100):#x}")
      # Extended MMIO/BAR diagnostics
      print(f"[kepler] IDLE_STATUS(0x409420)={rd(0x409420):#x} "
            f"BAR(0x409414)={rd(0x409414):#x} "
            f"BAR_SET0(0x409418)={rd(0x409418):#x} "
            f"BAR_SET1(0x40941c)={rd(0x40941c):#x} "
            f"BAR_MASK0(0x40940c)={rd(0x40940c):#x} "
            f"BAR_MASK1(0x409410)={rd(0x409410):#x}")
      # Sample FECS PC several times to see if it's moving
      pcs = []
      for _ in range(10):
        pcs.append(rd(0x409ff0))
        time.sleep(0.001)
      print(f"[kepler] FECS PC samples: {[hex(p) for p in pcs]}")
      # Try host-side write to GPC0 CTRL to see if the register is accessible
      print(f"[kepler] GPC0 CTRL before host write: {rd(0x502100):#x}")
      self.write32(0x502100, 0x02)
      time.sleep(0.01)
      print(f"[kepler] GPC0 CTRL after host write 2: {rd(0x502100):#x}")
      # Check all GPC falcon CTRL registers
      for gpc in range(4):
        gb = 0x502000 + gpc * 0x8000
        print(f"[kepler] GPC{gpc}: CTRL={rd(gb+0x100):#x} ENTRY={rd(gb+0x104):#x} "
              f"SCRATCH0={rd(gb+0x800):#x} SCRATCH1={rd(gb+0x804):#x} "
              f"CPUSTAT={rd(gb+0x128):#x} INTR={rd(gb+0x008):#x} "
              f"IRQMSET={rd(gb+0x010):#x} IRQMASK={rd(gb+0x018):#x} "
              f"BLOCK={rd(gb+0x10c):#x} HWCFG2={rd(gb+0x16c):#x}")
      # Sample GPC0 and GPC1 fuc PC to see if they're executing
      _gpc0_pcs = []
      _gpc1_pcs = []
      for _ in range(10):
        _gpc0_pcs.append(rd(0x502ff0))
        _gpc1_pcs.append(rd(0x50aff0))
        time.sleep(0.001)
      print(f"[kepler] GPC0 fuc PC samples: {[hex(p) for p in _gpc0_pcs]}")
      print(f"[kepler] GPC1 fuc PC samples: {[hex(p) for p in _gpc1_pcs]}")
      # Check if GPC1's register space is accessible (power/clock gated?)
      print(f"[kepler] GPC1 reg access: 0x508c30={rd(0x508c30):#x} "
            f"0x50a608={rd(0x50a608):#x} 0x50a100={rd(0x50a100):#x} "
            f"0x50a128={rd(0x50a128):#x} 0x50a724={rd(0x50a724):#x} "
            f"0x50a728={rd(0x50a728):#x} 0x50a72c={rd(0x50a72c):#x}")
      # Check GPC2 (the one that's stuck)
      print(f"[kepler] GPC2 reg access: 0x510c30={rd(0x510c30):#x} "
            f"0x512608={rd(0x512608):#x} 0x512100={rd(0x512100):#x} "
            f"0x512128={rd(0x512128):#x} 0x512724={rd(0x512724):#x} "
            f"0x512728={rd(0x512728):#x} 0x51272c={rd(0x51272c):#x}")
      _gpc2_pcs = []
      for _ in range(10):
        _gpc2_pcs.append(rd(0x512ff0))
        time.sleep(0.001)
      print(f"[kepler] GPC2 fuc PC samples: {[hex(p) for p in _gpc2_pcs]}")
      # Check GPC3 (the one that's stuck now)
      print(f"[kepler] GPC3 reg access: 0x518c30={rd(0x518c30):#x} "
            f"0x51a608={rd(0x51a608):#x} 0x51a100={rd(0x51a100):#x} "
            f"0x51a128={rd(0x51a128):#x} 0x51a724={rd(0x51a724):#x} "
            f"0x51a728={rd(0x51a728):#x} 0x51a72c={rd(0x51a72c):#x}")
      _gpc3_pcs = []
      for _ in range(10):
        _gpc3_pcs.append(rd(0x51aff0))
        time.sleep(0.001)
      print(f"[kepler] GPC3 fuc PC samples: {[hex(p) for p in _gpc3_pcs]}")
      # Test: manually set GPC3's SCRATCH0 bit 31 via CC_SCRATCH_SET
      # If FECS proceeds, GPC3's firmware completed but didn't set the bit
      print(f"[kepler] GPC3 SCRATCH0 before host set: {rd(0x51a800):#x}")
      self.write32(0x51a820, 0x80000000)  # CC_SCRATCH_SET(0) bit 31
      time.sleep(0.1)
      print(f"[kepler] GPC3 SCRATCH0 after host set: {rd(0x51a800):#x}")
      print(f"[kepler] FECS SCRATCH0 after GPC3 set: {rd(0x409800):#x}")
      # Check GPC1 DMEM mmio list pointers (offset 0x00-0x10)
      _gpc1_dmem = []
      for _a in range(0, 32, 4):
        self.write32(0x50a1c0, 0x82000000 | _a)  # DMEM auto-incr read
        _gpc1_dmem.append(self.read32(0x50a1c8))
      print(f"[kepler] GPC1 DMEM[0:8]: {['0x'+format(x,'08x') for x in _gpc1_dmem]}")
      # Compare with GPC0 DMEM
      _gpc0_dmem = []
      for _a in range(0, 32, 4):
        self.write32(0x5021c0, 0x82000000 | _a)
        _gpc0_dmem.append(self.read32(0x5021c8))
      print(f"[kepler] GPC0 DMEM[0:8]: {['0x'+format(x,'08x') for x in _gpc0_dmem]}")
      # Check GPC0 IMEM first 4 words
      _gpc0_imem = []
      for _a in range(0, 16, 4):
        self.write32(0x502180, _a)
        _gpc0_imem.append(self.read32(0x502188))
      print(f"[kepler] GPC0 fuc IMEM[0:4]: {['0x'+format(x,'08x') for x in _gpc0_imem]}")
      # Verify the FECS firmware image actually loaded into imem/dmem.
      # Read first 8 words WITHOUT autoincrement to check for gaps.
      _imem_words = []
      for _a in range(0, 32, 4):
        self.write32(FECS_FALCON_BASE + FALCON_CODE_INDEX, _a)  # no WRITE bit = read
        _imem_words.append(self.read32(FECS_FALCON_BASE + FALCON_CODE))
      _imem_exp = [struct.unpack_from('<I', fecs_code, i)[0] for i in range(0, 32, 4)]
      print(f"[kepler] FECS IMEM[0:8] (no autoincr): {['0x'+format(x,'08x') for x in _imem_words]}")
      print(f"[kepler] FECS IMEM[0:8] expected:      {['0x'+format(x,'08x') for x in _imem_exp]}")
      print(f"[kepler] FECS IMEM match={_imem_words == _imem_exp}")
      self.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, 0)
      dmem0 = rd(FECS_FALCON_BASE + FALCON_DATA)
      print(f"[kepler] dmem[0]={dmem0:#x}")
      # Inspect the IO address the PC is pointing to (0xd804 -> 0x360/IDX1)
      orig_idx = self.read32(0x409ffc)
      self.write32(0x409ffc, 0)
      v0 = self.read32(0x409360)
      self.write32(0x409ffc, 1)
      v1 = self.read32(0x409360)
      self.write32(0x409ffc, orig_idx)
      print(f"[kepler] FECS IO 0x409360 HOST_IO_INDEX=0 -> {v0:#x} (0xd800 block)")
      print(f"[kepler] FECS IO 0x409360 HOST_IO_INDEX=1 -> {v1:#x} (0xd804 block, PC target)")
      # Falcon v3+ has per-page code TLB.  Dump it to see whether the PC is
      # fetching an unmapped page or a real exit instruction.
      self._dump_fecs_tlb()
      raise
    # FECS firmware sets FE_PWR=0 during its init, gating PGRAPH sub-domains.
    # Re-assert FORCE_ON now that FECS is ready and idle.  This keeps the FE
    # domain powered so PGRAPH method/FIFO sub-domains (0x400700, 0x404200)
    # remain accessible for channel dispatch and method processing.
    self.write32(0x404170, 0x00000012)  # FE_PWR FORCE_ON
    # Disable the FECS idle filter (0x020288) to prevent auto clock-gating.
    # nouveau's gk104_clkgate_enable writes 0x00001000 to this register, which
    # tells the therm subsystem to clock-gate FECS after an idle period.
    # Without a PMU managing power state, this causes FECS to power-gate
    # when idle, making all FECS registers read 0xbadf1000.  Setting it to 0
    # disables the idle filter and keeps FECS clocked.
    self.write32(0x020288, 0x00000000)  # FECS idle filter disabled
    self.write32(0x02028c, 0x00000000)  # HUBMMU idle filter disabled
    _fe_pwr_ready = self.read32(0x404170)
    print(f"[kepler] FE_PWR after FECS ready: {_fe_pwr_ready:#x} "
          f"idle_filter={self.read32(0x020288):#x}", flush=True)
    self.write32(FECS_FALCON_BASE + 0x048, 0x00000003)
    ctx_size = self.read32(0x409804)
    # Save this immediately.  FECS mailbox 1 is reused by the context
    # generator, so reading 0x409804 again after grctx_main returns command
    # data (observed 0x8c000), not the engine context-image size.
    self.gr_ctx_size = ctx_size
    if DEBUG:
      rd = lambda o: self.read32(o)
      print(f"[kepler] PMC_ENABLE(0x200)={rd(0x200):#x}  PGRAPH_STATUS(0x400000)={rd(0x400000):#x}")
      print(f"[kepler] FECS CPUCTL(0x409100)={rd(0x409100):#x}  FALCON_VER(0x40912c)={rd(0x40912c):#x}")
      for o in (0x400100, 0x409000, 0x101000, 0x010000, 0x000004):
        print(f"[kepler]   0x{o:06x} = {rd(o):#x}")
      mb = [rd(0x409800 + i * 4) for i in range(8)]
      print(f"[kepler] FECS mailboxes 0x409800..0x40981c = {[hex(x) for x in mb]}")
      # write/read roundtrip: does the dext forward GR/FECS MMIO, or is the
      # engine simply clock-gated?  If readback == written, the path works.
      self.write32(0x400100, 0x12345678)
      print(f"[kepler] GR write/read 0x400100: wrote 0x12345678, read {rd(0x400100):#x}")
      self.write32(0x409100, 0x2)
      print(f"[kepler] FECS write/read 0x409100: wrote 0x2, read {rd(0x409100):#x}")
      print(f"[kepler] GR ctx image size={ctx_size:#x}  (0xbad0da1f = register-block not accessible)")
    # Complete the GK104 subdev initialization that was not performed at boot.
    # On a properly POSTed card, the VBIOS/nouveau driver init sequence runs:
    #   fb_preinit  → sysmem flush page (0x100c10)
    #   fb_init     → big-page mode (0x100c80) + sysmem flush page
    #   ltc_init    → L2 cache topology (0x17e8d8/0x17e000/0x17e8d4/0x17e8c0)
    #   bar_init    → BAR1 VMM enable (0x1704)
    # The un-POSTed card has none of these, so CPU framebuffer writes are not
    # visible to GPU internal clients (PBDMA reads zero GP entries).
    #
    # LTC init and FB init_page need no sysmem, so run them first.
    _gk104_ltc_init(self)
    _gk104_fb_init_page(self)
    # Clear any bootstrap BAR1 mapping left by an earlier diagnostic process
    # so the direct PCI BAR mapping is used until we set up the VMM aperture.
    if (self.read32(0x001704) & 0x3fffffff) == 0x100:
      self.write32(0x001704, 0)
    # 3. Sysmem aperture: allocate GPU-visible host memory and mmap it as a
    #    CPU-coherent buffer.  Its bus base (physical address) becomes the GMMU
    #    bus_base.  alloc_sysmem uses mmap + /proc/self/pagemap to get real
    #    physical addresses for DMA.
    sysmem_size = self.VRAM_SIZE
    memview, paddrs = self.iface.pci_dev.alloc_sysmem(sysmem_size, contiguous=True)
    dev.vram = memview.mv if hasattr(memview, "mv") else memview
    dev.bus_base = paddrs[0]
    dev.max_pa = sysmem_size
    # gf100_fb_sysmem_flush_page_init(): program the sysmem flush page address
    # so the GPU can flush dirty cache lines to sysmem.  Without this, the L2
    # cache may retain stale zeros and PBDMA/GR read incorrect data.
    self.write32(0x100c10, dev.bus_base >> 8)
    # GK104 PFIFO polls USERD through BAR1's VMM, not through the CPU's direct
    # PCI BAR mapping.  Nouveau creates this mapping during BAR/FIFO oneinit.
    # Use an identity VA->VRAM mapping so USERD BAR1 VAs equal the framebuffer
    # offsets allocated by submit_launch().
    if os.environ.get("KEPLER_INIT_BAR1_EARLY") == "1":
      _gk104_init_bar1_identity(self, bus_base=dev.bus_base, map_vram=True)
    dev.mm = GK104MemoryManager(dev, sysmem_size, self.BOOT_SIZE, bus_base=dev.bus_base)
    # 4-12. FIFO channel / GPFIFO / USERD / GR context / launch: KEPLER-TODO.
    dev.is_booting = False

  def runtime(self, name, lib):
    return NVProgram(self, name, lib)
  def synchronize(self): pass

  def close(self):
    teardown = getattr(self, "_kepler_emergency_teardown", None)
    try:
      if teardown is not None:
        teardown("device close")
      if self.backend != "software":
        # No teardown MMIO is safe after the successful command/output phase.
        # The transport simply unmaps the BARs and closes the resource fds.
        print("[kepler] client-only close: no teardown MMIO/config/reset",
              flush=True)
    finally:
      pci_dev = self.iface.pci_dev
      if hasattr(pci_dev, "set_phase"):
        pci_dev.set_phase("client-close")
      self.iface.pci_dev.fini()

  # MMIO convenience helpers used by the host-driven hardware bring-up.
  def read32(self, off): return self.dev_impl.pci_dev.mmio_read32(0, off)
  def write32(self, off, v): self.dev_impl.pci_dev.mmio_write32(0, off, v)
  def read64(self, off): return self.dev_impl.pci_dev.mmio_read64(off)
  def write64(self, off, v): self.dev_impl.pci_dev.mmio_write64(off, v)


# ============================================================================
# Kepler (sm_30) cubin builder
# ----------------------------------------------------------------------------
# Kepler instructions are 8 bytes (single-issue; no Maxwell/Pascal dual-issue
# bundle word).  The opcode lives in the low byte(s).  The encodings below are
# derived from ref/denvdis/data/sm3_1.txt (the envytools Kepler decode spec):
#   FADD  pipe = 0b01010_0_000  (envytools "FADD")
#   LD/ST global classes "LD" / "ST" (envytools data/sm3_1.txt:12607 / 12803)
#
# KEPLER-TODO (cubin): the hand-assembled SASS below is a STRUCTURAL best effort
# and has NOT been verified with nvdisasm against an sm_30 cubin.  To get the
# exact, verified bytes, build a reference on a machine with a CUDA toolkit
# that still supports sm_30 (CUDA <= 11.x, since CUDA 12 dropped sm_30):
#
#   cat > add_kepler.cu <<'EOF'
#   extern "C" __global__ void E_4(const float* a, const float* b, float* out){
#     int i = blockIdx.x*blockDim.x + threadIdx.x; out[i] = a[i] + b[i];
#   }
#   EOF
#   nvcc -arch=sm_30 -cubin -o add_kepler.cubin add_kepler.cu
#   nvdisasm -fun add_kepler.cubin > add_kepler.sass
#
# then paste the SASS into the bundles list.  The offline self-test only checks
# structure/stability, NOT that the SASS is semantically correct — that is what
# the nvdisasm comparison is for.
# ============================================================================
class CubinHelper:
  class Reg:
    RZ = 255
    R0 = 0; R1 = 1; R2 = 2; R3 = 3; R4 = 4; R5 = 5; R6 = 6; R7 = 7
    R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15

  class UReg:
    URZ = 63
    UR4 = 4

  class Op:
    # envytools sm3_1 opcodes (10-bit pipe field; placed at bits[15:6] style)
    LDC    = 0x7a02
    FADD   = 0x7280   # KEPLER-TODO: confirm exact FADD opcode from sm3_1
    LDG    = 0x7981   # KEPLER-TODO: confirm exact LD.G opcode from sm3_1
    STG    = 0x7986   # KEPLER-TODO: confirm exact ST.G opcode from sm3_1
    EXIT   = 0x794d
    BRA    = 0x7947
    NOP    = 0x7918

  SECTION_NAMES = (
    ".shstrtab", ".strtab", ".symtab", ".symtab_shndx", ".nv.info", ".text.E_4", ".nv.info.E_4", ".nv.shared.E_4",
    ".nv.constant0.E_4", ".rel.nv.constant0.E_4", ".debug_frame", ".rel.debug_frame", ".rela.debug_frame", ".nv.callgraph",
    ".nv.prototype", ".nv.rel.action"
  )
  SYMBOL_NAMES = (
    ".shstrtab", ".strtab", ".symtab", ".symtab_shndx", ".nv.info", ".text.E_4", ".nv.info.E_4", ".nv.shared.E_4",
    ".rel.nv.constant0.E_4", ".nv.constant0.E_4", ".debug_frame", ".rel.debug_frame", ".rela.debug_frame", ".nv.callgraph",
    ".nv.prototype", ".nv.rel.action", "E_4"
  )
  SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_REL = 1, 2, 3, 9
  SHT_CUDA_INFO, SHT_CUDA_CALLGRAPH, SHT_CUDA_RELOCINFO = 0x70000000, 0x70000001, 0x7000000b
  SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR, SHF_INFO_LINK = 1, 2, 4, 0x40
  STB_GLOBAL, STT_SECTION, STT_FUNC = 1, 3, 2
  PT_LOAD, PT_PHDR = 1, 6
  PF_X, PF_R = 1, 4
  ET_EXEC, EM_CUDA = 2, 190
  EV_CURRENT, ELF_ABIVERSION, ELF_VERSION = 1, 7, 128
  ELFOSABI_CUDA = 0x33
  ELFCLASS64, ELFDATA2LSB = 2, 1
  EF_CUDA_SM30 = 0x001e001e   # sm_30: SM=0x1e(30), PTX_SM=0x1e(30)
  ELF_HEADER_SIZE = 64
  SECTION_HEADER_SIZE = 64
  PROGRAM_HEADER_SIZE = 56
  SECTION_HEADERS_OFF = 1920
  PROGRAM_HEADERS_OFF = 2688
  SHSTRTAB_OFF = 64
  STRTAB_OFF = 283
  SYMTAB_OFF = 512
  DEBUG_FRAME_OFF = 680
  NV_INFO_OFF = 792
  NV_INFO_E4_OFF = 828
  NV_CALLGRAPH_OFF = 932
  NV_REL_ACTION_OFF = 968
  REL_DEBUG_FRAME_OFF = 984
  NV_CONSTANT0_OFF = 1000
  NV_CONSTANT0_SIZE = 376
  TEXT_OFF = 1408

  @staticmethod
  def string_table(names):
    table, offsets = bytearray(b"\0"), {}
    for name in names:
      offsets[name] = len(table)
      table += name.encode() + b"\0"
    return bytes(table), offsets

  @staticmethod
  def words_blob(words): return b"".join(struct.pack("<I", w) for w in (words if isinstance(words, (list, tuple)) else (words,)))

  @staticmethod
  def header(phoff, shoff, phnum, shnum, shstrndx):
    ident = b"\x7fELF" + bytes((CubinHelper.ELFCLASS64, CubinHelper.ELFDATA2LSB, CubinHelper.EV_CURRENT, CubinHelper.ELFOSABI_CUDA, CubinHelper.ELF_ABIVERSION)) + bytes(7)
    return struct.pack("<16sHHIQQQIHHHHHH", ident, CubinHelper.ET_EXEC, CubinHelper.EM_CUDA, CubinHelper.ELF_VERSION, 0, phoff, shoff, CubinHelper.EF_CUDA_SM30, 64, 56, phnum, 64, shnum, shstrndx)

  @staticmethod
  def symtab_entry(name, bind, typ, other, shndx, value=0, size=0): return struct.pack("<IBBHQQ", name, (bind << 4) | typ, other, shndx, value, size)

  @staticmethod
  def dwarf64_record(payload): return struct.pack("<IQ", 0xffffffff, len(payload)) + payload

  def cie_record(self):
    cie_id, version, augmentation, address_size, segment_size = 0xffffffffffffffff, 3, 0, 4, 0x7c
    code_align, data_align, return_register = 0xffffffff, 0x0f, 0x0c
    frame_instructions = bytes((0x81, 0x80, 0x80, 0x28, 0x00, 0x08, 0xff, 0x81, 0x80, 0x28, 0x08, 0x81, 0x80, 0x80, 0x28, 0, 0, 0))
    return self.dwarf64_record(struct.pack("<QBBBBIBB", cie_id, version, augmentation, address_size, segment_size, code_align, data_align, return_register) + frame_instructions)

  def fde_record(self):
    cie_pointer, initial_location, address_range = 0, 0, 512
    frame_instructions = self.words_blob((0x404, 0x3c0400, 0x810c0000, 0x288080, 0xfffffc04, 0x3f, 0))
    return self.dwarf64_record(struct.pack("<QQQ", cie_pointer, initial_location, address_range) + frame_instructions)

  def nv_info_attr(self, kind, selector, payload_words, format_byte=4): return self.words_blob(((kind << 12) | (selector << 8) | format_byte, *payload_words))

  def section_header(self, name, typ, flags, addr, offset, size, link=0, info=0, align=1, entsize=0): return (self.SHN[name] if name else 0, typ, flags, addr, offset, size, link, info, align, entsize)

  def program_header(self, typ, flags, offset, filesz, memsz=None, vaddr=0, paddr=0, align=8): return (typ, flags, offset, vaddr, paddr, filesz, filesz if memsz is None else memsz, align)

  def __init__(self):
    self.SHSTRTAB, self.SHN = self.string_table(self.SECTION_NAMES)
    self.STRTAB, self.STN = self.string_table(self.SYMBOL_NAMES)
    self.SYMTAB = b"".join((
      self.symtab_entry(0, 0, 0, 0, 0),
      self.symtab_entry(self.STN[".text.E_4"], 0, self.STT_SECTION, 0, 11),
      self.symtab_entry(self.STN[".nv.constant0.E_4"], 0, self.STT_SECTION, 0, 10),
      self.symtab_entry(self.STN[".debug_frame"], 0, self.STT_SECTION, 0, 4),
      self.symtab_entry(self.STN[".nv.callgraph"], 0, self.STT_SECTION, 0, 7),
      self.symtab_entry(self.STN[".nv.rel.action"], 0, self.STT_SECTION, 0, 8),
      self.symtab_entry(self.STN["E_4"], self.STB_GLOBAL, self.STT_FUNC, 0x10, 11, size=512),
    ))
    self.DEBUG_FRAME = self.cie_record() + self.fde_record()
    self.NV_INFO = b"".join((
      self.nv_info_attr(0x82, 0xf, (6, 14)),
      self.nv_info_attr(0x81, 0x1, (6, 0)),
      self.nv_info_attr(0x81, 0x2, (6, 0)),
    ))
    self.NV_INFO_E4 = b"".join((
      self.nv_info_attr(0x43, 0x7, (128, 0x3501)),
      self.nv_info_attr(0x80, 0xa, (2, 0x180160, 0x181903)),
      self.nv_info_attr(0xc1, 0x7, (0, 0x100002, 0x21f000)),
      self.nv_info_attr(0xc1, 0x7, (0, 0x80001, 0x21f000)),
      self.nv_info_attr(0xc1, 0x7, (0, 0, 0x21f000)),
      self.nv_info_attr(0xff1, 0xb, ((0x41 << 12) | (0xc << 8) | 4, 240), format_byte=3),
      self.nv_info_attr(0xc0, 0x5, (1, 1, 1)),
    ))
    self.NV_CALLGRAPH = b"".join(struct.pack("<II", 0, target) for target in (0xffffffff, 0xfffffffe, 0xfffffffd, 0xfffffffc))
    self.NV_REL_ACTION = struct.pack("<IIHHHH", 115, 0, 0, 0x1100, 0x0025, 0x3605)
    self.REL_DEBUG_FRAME = struct.pack("<QQ", 68, (6 << 32) | 2)
    self.SECTION_HEADERS = (
      self.section_header("", 0, 0, 0, 0, 0, align=0),
      self.section_header(".shstrtab", self.SHT_STRTAB, 0, 0, self.SHSTRTAB_OFF, len(self.SHSTRTAB)),
      self.section_header(".strtab", self.SHT_STRTAB, 0, 0, self.STRTAB_OFF, len(self.STRTAB)),
      self.section_header(".symtab", self.SHT_SYMTAB, 0, 0, self.SYMTAB_OFF, len(self.SYMTAB), link=2, info=6, align=8, entsize=24),
      self.section_header(".debug_frame", self.SHT_PROGBITS, 0, 0, self.DEBUG_FRAME_OFF, len(self.DEBUG_FRAME)),
      self.section_header(".nv.info", self.SHT_CUDA_INFO, 0, 0, self.NV_INFO_OFF, len(self.NV_INFO), link=3, align=4),
      self.section_header(".nv.info.E_4", self.SHT_CUDA_INFO, self.SHF_INFO_LINK, 0, self.NV_INFO_E4_OFF, len(self.NV_INFO_E4), link=3, info=11, align=4),
      self.section_header(".nv.callgraph", self.SHT_CUDA_CALLGRAPH, 0, 0, self.NV_CALLGRAPH_OFF, len(self.NV_CALLGRAPH), link=3, align=4, entsize=8),
      self.section_header(".nv.rel.action", self.SHT_CUDA_RELOCINFO, 0, 0, self.NV_REL_ACTION_OFF, len(self.NV_REL_ACTION), align=8, entsize=8),
      self.section_header(".rel.debug_frame", self.SHT_REL, self.SHF_INFO_LINK, 0, self.REL_DEBUG_FRAME_OFF, len(self.REL_DEBUG_FRAME), link=3, info=4, align=8, entsize=16),
      self.section_header(".nv.constant0.E_4", self.SHT_PROGBITS, self.SHF_ALLOC | self.SHF_INFO_LINK, 0, self.NV_CONSTANT0_OFF, self.NV_CONSTANT0_SIZE, info=11, align=4),
      self.section_header(".text.E_4", self.SHT_PROGBITS, self.SHF_ALLOC | self.SHF_EXECINSTR, 0, self.TEXT_OFF, 512, link=3, info=0x0e000006, align=128),
    )
    self.PROGRAM_HEADERS = (
      self.program_header(self.PT_PHDR, self.PF_R | self.PF_X, self.PROGRAM_HEADERS_OFF, 168),
      self.program_header(self.PT_LOAD, self.PF_R | self.PF_X, self.NV_CONSTANT0_OFF, 920),
      self.program_header(self.PT_LOAD, self.PF_R | self.PF_X, self.PROGRAM_HEADERS_OFF, 168),
    )

ch = CubinHelper()

def build_cubin():
  # KEPLER-TODO (SASS): Kepler is 8-byte instructions, single issue.  Each
  # bundle below is (opcode_hi, imm24, imm24, imm24) packed into 4 uint32 words
  # = 16 bytes here only as a placeholder container; the REAL encoding is one
  # 64-bit instruction per slot.  Replace with verified sm_30 SASS from nvdisasm
  # (see the build recipe in the CubinHelper docstring above).
  bundles = [
    # ---- prologue: load kernarg pointers from c[0x0] (param constant bank) ----
    ((ch.Reg.R4 << 16) | ch.Op.LDC, 0x00005c00, 0x00000f00, 0x000fe200),  # MOV R4, c[0x0][0x170]  (a)
    ((ch.Reg.R5 << 16) | ch.Op.LDC, 0x00005d00, 0x00000f00, 0x000fe400),  # MOV R5, c[0x0][0x174]  (b)
    ((ch.Reg.R6 << 16) | ch.Op.LDC, 0x00005800, 0x00000f00, 0x000fc400),  # MOV R6, c[0x0][0x160]  (out)
    # ---- global loads ----
    ((ch.Reg.R4 << 24) | (ch.Reg.R8 << 16) | ch.Op.LDG, 0x00000004, 0x0c1e1d00, 0x000ea800),  # LDG.E R8, [R4]
    ((ch.Reg.R5 << 24) | (ch.Reg.R9 << 16) | ch.Op.LDG, 0x00000004, 0x0c1e1d00, 0x000ea400),  # LDG.E R9, [R5]
    # ---- arithmetic ----
    ((ch.Reg.R10 << 24) | (ch.Reg.R10 << 16) | ch.Op.FADD, 0x00000008, 0x00000000, 0x000fe200),  # FADD R10, R9, R8
    # ---- global store ----
    ((ch.Reg.R6 << 24) | ch.Op.STG, 0x00000008, 0x0c101d04, 0x000fe200),  # STG.E [R6], R10
    (ch.Op.EXIT, 0x00000000, 0x03800000, 0x000fea00),  # EXIT
    (ch.Op.BRA, 0xfffffff0, 0x0383ffff, 0x000fc000),  # BRA .
  ]
  text = b"".join(ch.words_blob(bundle) for bundle in bundles)

  SECTIONS = {
    ch.SHSTRTAB_OFF: ch.SHSTRTAB, ch.STRTAB_OFF: ch.STRTAB, ch.SYMTAB_OFF: ch.SYMTAB,
    ch.DEBUG_FRAME_OFF: ch.DEBUG_FRAME,
    ch.NV_INFO_OFF: ch.NV_INFO, ch.NV_INFO_E4_OFF: ch.NV_INFO_E4, ch.NV_CALLGRAPH_OFF: ch.NV_CALLGRAPH, ch.NV_REL_ACTION_OFF: ch.NV_REL_ACTION,
    ch.REL_DEBUG_FRAME_OFF: ch.REL_DEBUG_FRAME,
    ch.NV_CONSTANT0_OFF: bytes(ch.NV_CONSTANT0_SIZE), ch.TEXT_OFF: text,
  }

  cubin = bytearray(2856)
  cubin[:ch.ELF_HEADER_SIZE] = ch.header(phoff=ch.PROGRAM_HEADERS_OFF, shoff=ch.SECTION_HEADERS_OFF, phnum=len(ch.PROGRAM_HEADERS), shnum=len(ch.SECTION_HEADERS), shstrndx=1)
  for offset, data in SECTIONS.items():
    cubin[offset:offset+len(data)] = data
  for index, header in enumerate(ch.SECTION_HEADERS):
    cubin[ch.SECTION_HEADERS_OFF + index * ch.SECTION_HEADER_SIZE:ch.SECTION_HEADERS_OFF + (index + 1) * ch.SECTION_HEADER_SIZE] = struct.pack("<IIQQQQIIQQ", *header)
  for index, header in enumerate(ch.PROGRAM_HEADERS):
    cubin[ch.PROGRAM_HEADERS_OFF + index * ch.PROGRAM_HEADER_SIZE:ch.PROGRAM_HEADERS_OFF + (index + 1) * ch.PROGRAM_HEADER_SIZE] = struct.pack("<IIQQQQQQ", *header)
  return bytes(cubin)


# ============================================================================
# Real Kepler cubin (plan Milestone 13): a CUDA 10.2 `nvcc -arch=sm_30` cubin is
# the only verified path.  We compile it in Docker if available, or load a
# prebuilt one from $KEPLER_CUBIN.  The offline build_cubin() above remains only
# for the self-test (it is NOT executable SASS).
# ============================================================================
KEPLER_CU_SRC = """
extern "C" __global__ void E_4(const float* a, const float* b, float* out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = a[i] + b[i];
}
"""

def compile_kepler_cubin_docker(tag="nvidia/cuda:11.0.3-devel-ubuntu20.04"):
  """Compile a genuine sm_30 cubin with the locally cached CUDA 11.0 image."""
  try:
    import tempfile, subprocess
    d = tempfile.mkdtemp(prefix="kepler_cubin_")
    cu = os.path.join(d, "add.cu")
    with open(cu, "w") as f: f.write(KEPLER_CU_SRC)
    out = os.path.join(d, "add.cubin")
    subprocess.run(
      ["docker", "run", "--rm", "-v", f"{d}:/work", "-w", "/work", tag,
       "bash", "-c", "nvcc -arch=sm_30 -cubin add.cu -o add.cubin"],
      check=True, capture_output=True, timeout=600)
    return out if os.path.exists(out) else None
  except Exception as e:
    if DEBUG: print(f"[kepler] docker nvcc unavailable: {e}")
    return None

def get_kepler_cubin():
  path = os.environ.get("KEPLER_CUBIN")
  # The repository carries the verified nvcc-built sm_30 image used by this
  # example.  Prefer it so `python3 examples_kepler/add.py` is self-contained;
  # Docker remains a fallback for source-only checkouts.
  bundled = os.path.join(os.path.dirname(__file__), "add_kepler.cubin")
  if not path and os.path.exists(bundled): path = bundled
  if not path or not os.path.exists(path): path = compile_kepler_cubin_docker()
  if not path or not os.path.exists(path):
    raise RuntimeError("live hardware requires a real sm_30 add cubin; set KEPLER_CUBIN or start Docker")
  with open(path, "rb") as f: cubin = f.read()
  if len(cubin) < 0x40 or cubin[:4] != b"\x7fELF" or struct.unpack_from("<I", cubin, 0x30)[0] & 0xff != 0x1e:
    raise ValueError(f"KEPLER_CUBIN must be an sm_30 ELF cubin: {path}")
  elf_section_bytes(cubin, ".text.E_4")
  cubin_register_count(cubin, "E_4")
  return cubin


# ============================================================================
# Kepler FALCON microcontrollers (plan §24.2).  GK104 FALCONs are NOT in
# secure-boot (HS) mode, so IMEM/DMEM load + start works directly (no ACR).
# ============================================================================
FALCON_UC_CTRL      = 0x100
FALCON_UC_ENTRY     = 0x104
FALCON_CODE_INDEX   = 0x180
FALCON_CODE         = 0x184
FALCON_CODE_TAG     = 0x188   # ITAG: written every 256B (64 words) during imem load
FALCON_DATA_INDEX   = 0x1c0
FALCON_DATA         = 0x1c4
FALCON_STATUS       = 0x800   # bit31 (0x80000000) = FECS init done / ready
FALCON_UC_CTRL_START = 1 << 1   # UC_CTRL START_TRIGGER bit (bit 1 = 0x2)
FALCON_IDX_WRITE    = 1 << 24   # CODE/DATA_INDEX WRITE bit (nouveau falcon_v1: only this)
PMU_FALCON_BASE  = 0x10a000
FECS_FALCON_BASE = 0x409000
GPCCS_FALCON_BASE = 0x41a000
PMC_ENABLE       = 0x200

def falcon_write_dmem(dev, base, data):
  """nouveau nvkm_falcon_v1_load_dmem: DATA_INDEX = start|WRITE, then DATA words."""
  data = data[:len(data) // 4 * 4]
  dev.write32(base + FALCON_DATA_INDEX, FALCON_IDX_WRITE)
  for i in range(0, len(data), 4):
    dev.write32(base + FALCON_DATA, struct.unpack_from("<I", data, i)[0])

def falcon_write_imem(dev, base, code):
  """Load code into falcon IMEM.  On GK104 FECS, the CODE write autoincrement
  advances by 8 bytes (not 4), doubling each word at two consecutive addresses.
  Work around this by writing each word with an explicit CODE_INDEX (no
  autoincrement).  ITAG is written at the start of each 256-byte page.
  ponytail: explicit indexing is slower (2 writes per word) but correct.
  Ceiling: if autoincrement is fixed on a future variant, switch back to
  the single-write autoincrement path for speed."""
  code = code[:len(code) // 4 * 4]
  nwords = len(code) // 4
  tag = 0
  for i in range(nwords):
    addr = i * 4
    if (addr & 0xff) == 0:
      dev.write32(base + FALCON_CODE_TAG, tag)
      tag += 1
    dev.write32(base + FALCON_CODE_INDEX, addr)  # no WRITE bit = no autoincr
    dev.write32(base + FALCON_CODE, struct.unpack_from("<I", code, i * 4)[0])
  # Pad to page boundary with zeros
  i = nwords
  while (i * 4) & 0xff:
    addr = i * 4
    dev.write32(base + FALCON_CODE_INDEX, addr)
    dev.write32(base + FALCON_CODE, 0)
    i += 1

def falcon_stop(dev, base, timeout_ms=500):
  """Stop a running FALCON: set HALT, clear START, wait for STOPPED status.
  Needed before reloading IMEM/DMEM on a card that was already POSTed by
  nouveau/EFI — the running falcon corrupts auto-increment IMEM writes."""
  dev.write32(base + FALCON_UC_CTRL, 0x00000010)  # HALT
  dev.write32(base + FALCON_UC_CTRL, 0x00000000)  # clear START
  wait_cond(lambda: bool(dev.read32(base + 0x128) & 0x10),  # CPUSTAT STOPPED
            timeout_ms=timeout_ms, msg=f"falcon stop at {base:#x}")

def falcon_start(dev, base):
  """nouveau gf100_gr_init_ctxctl_int FECS start: clear 0x10c, pulse CPUCTL start."""
  dev.write32(base + 0x10c, 0x00000000)
  dev.write32(base + FALCON_UC_CTRL, FALCON_UC_CTRL_START)

def falcon_ready(dev, base):
  """FECS ready flag: 0x409800 & 0x80000000 (init done)."""
  return bool(dev.read32(base + FALCON_STATUS) & 0x80000000)

# -----------------------------------------------------------------------------
# Minimal VBIOS inspection.  This deliberately does not execute init scripts
# yet: a wrong script can change board power/clock state.  It is enough to
# validate a dump, enumerate its PCI ROM images, and expose the BIT directory
# that the devinit port will consume.
# -----------------------------------------------------------------------------
def inspect_vbios(path):
  data = pathlib.Path(path).read_bytes()
  if len(data) < 0x20:
    raise ValueError(f"VBIOS is too short: {len(data)} bytes")

  pcirs = []
  pos = 0
  while True:
    pos = data.find(b"PCIR", pos)
    if pos < 0: break
    if pos + 0x18 <= len(data):
      vendor, device = struct.unpack_from("<HH", data, pos + 4)
      image_len = struct.unpack_from("<H", data, pos + 0x10)[0] * 512
      pcirs.append((pos, vendor, device, image_len))
    pos += 4
  if not pcirs:
    raise ValueError("VBIOS has no PCIR structure")
  matches = [x for x in pcirs if x[1:3] == (0x10de, 0x1184)]
  if not matches:
    raise ValueError("VBIOS does not contain a GK104/GTX 770 (10DE:1184) image")

  # BIT is the modern Kepler table directory.  Nouveau searches for the
  # extended signature ff b8 "BIT"; the header fields are documented by
  # nvkm/subdev/bios/bit.c.
  bit = data.find(b"\xff\xb8BIT")
  entries = []
  image_base = data.rfind(b"\x55\xaa", 0, bit + 1)
  if bit >= 0 and bit + 12 <= len(data):
    stride, count = data[bit + 9], data[bit + 10]
    if stride >= 6 and bit + 12 + stride * count <= len(data):
      for i in range(count):
        p = bit + 12 + i * stride
        ident, version = data[p], data[p + 1]
        length, offset = struct.unpack_from("<HH", data, p + 2)
        entries.append((ident, version, length, offset))
  print(f"vbios: path={path} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}")
  for p, vendor, device, image_len in matches:
    rom_base = data.rfind(b"\x55\xaa", max(0, p - 0x1000), p + 1)
    print(f"vbios: PCIR@{p:#x} id={vendor:04X}:{device:04X} "
          f"image_base={(rom_base if rom_base >= 0 else 'unknown')} "
          f"image_bytes={image_len}")
  if bit < 0:
    print("vbios: BIT directory not found")
  else:
    # BIT table offsets are relative to the selected ROM image in NVGI dumps;
    # retain both coordinates because MMIO/devinit code consumes image-local
    # offsets while the file inspector operates on the container.
    print(f"vbios: BIT@{bit:#x} image_relative={bit-image_base if image_base else 'unknown'} "
          f"entries={len(entries)} stride={data[bit+9]:#x}")
    for ident, version, length, offset in entries:
      name = chr(ident) if 32 <= ident < 127 else f"0x{ident:02x}"
      raw_offset = image_base + offset if image_base else None
      print(f"  BIT {name} v{version} len={length} offset={offset:#x} "
            f"raw={(raw_offset if raw_offset is not None else 'unknown')}")
  return data

def _vbios_first_image(data, device=0x1184):
  """Return the clean PCI ROM image for a matching GK104 image in an NVGI dump."""
  for pcir in range(len(data)):
    if data[pcir:pcir + 4] != b"PCIR" or pcir + 0x12 > len(data): continue
    vendor, dev = struct.unpack_from("<HH", data, pcir + 4)
    if vendor != 0x10de or dev != device: continue
    size = struct.unpack_from("<H", data, pcir + 0x10)[0] * 512
    base = data.rfind(b"\x55\xaa", max(0, pcir - 0x1000), pcir + 1)
    if base >= 0 and base + size <= len(data): return data[base:base + size]
  raise ValueError("no complete matching PCI ROM image")

def _vbios_find_init_table(image):
  """Find the NVINIT pointer table without assuming a container offset.

  GK104's BIT-I record is present in this dump but is not decoded by the
  vendored EnvyTools release.  The table itself is unambiguous: a count byte
  followed by 16-bit pointers whose first script starts with RESET_BEGUN (8c).
  """
  for off in range(1, len(image) - 32):
    count = image[off - 1]
    if not 1 <= count <= 16 or off + count * 2 > len(image): continue
    ptrs = list(struct.unpack_from("<" + "H" * count, image, off))
    if ptrs[0] >= len(image) or image[ptrs[0]] != 0x8c: continue
    if all(0 < p < len(image) for p in ptrs): return off, ptrs
  raise ValueError("NVINIT script pointer table not found")

def _vbios_condition_table(image):
  """Return the BIT-I generic condition table pointer for this image."""
  bit = image.find(b"BIT")
  if bit < 2 or bit + 12 > len(image): return 0
  hlen, rlen, count = image[bit + 6], image[bit + 7], image[bit + 8]
  if hlen < 12 or rlen < 6: return 0
  for i in range(count):
    p = bit - 2 + hlen + i * rlen
    if p + 6 > len(image): break
    if image[p] == ord("I") and image[p + 1] == 1:
      t = struct.unpack_from("<H", image, p + 4)[0]
      if t + 18 <= len(image): return struct.unpack_from("<H", image, t + 6)[0]
  return 0

def _vbios_script_table(image):
  """Return the BIT-I indexed init-script table pointer."""
  bit = image.find(b"BIT")
  if bit < 2 or bit + 12 > len(image): return 0
  hlen, rlen, count = image[bit + 6], image[bit + 7], image[bit + 8]
  for i in range(count):
    p = bit - 2 + hlen + i * rlen
    if p + 6 > len(image): break
    if image[p] == ord("I") and image[p + 1] == 1:
      t = struct.unpack_from("<H", image, p + 4)[0]
      return struct.unpack_from("<H", image, t)[0] if t + 2 <= len(image) else 0
  return 0

def _vbios_target_ops(image, start, limit=0x4000, _seen=None, _enabled=True):
  """Decode only simple direct writes while inspecting an init script.

  This is intentionally read-only and incomplete; it is a guardrail for
  selecting the power/clock operations before the full conditional executor
  is enabled.
  """
  if _seen is None: _seen = set()
  if start in _seen: return []
  _seen.add(start)
  end = min(len(image), start + limit)
  pos, ops = start, []
  cond_table = _vbios_condition_table(image)
  while pos < end:
    op = image[pos]
    if op == 0x75 and pos + 2 <= end:       # CONDITION (evaluated at execution)
      cond = image[pos + 1]
      if cond_table and cond_table + cond * 12 + 12 <= len(image):
        reg, mask, val = struct.unpack_from("<III", image, cond_table + cond * 12)
        ops.append((pos, "condition", reg, mask, val))
      pos += 2
      continue
    if op == 0x72:                         # RESUME
      ops.append((pos, "resume"))
      pos += 1; continue
    if op == 0x7a and pos + 9 <= end:       # ZM_REG
      addr, val = struct.unpack_from("<II", image, pos + 1)
      if 0x020000 <= addr < 0x021000 or 0x137000 <= addr < 0x137400:
        ops.append((pos, "write", addr, val))
      pos += 9; continue
    if op == 0x6e and pos + 13 <= end:      # NV_REG mask/or
      addr, mask, val = struct.unpack_from("<III", image, pos + 1)
      if 0x020000 <= addr < 0x021000 or 0x137000 <= addr < 0x137400:
        ops.append((pos, "mask", addr, mask, val))
      pos += 13; continue
    if op == 0x91 and pos + 6 <= end:      # ZM_REG_GROUP
      addr = struct.unpack_from("<I", image, pos + 1)[0]
      count = image[pos + 5]
      for i in range(count):
        value = struct.unpack_from("<I", image, pos + 6 + i * 4)[0]
        if 0x020000 <= addr < 0x021000 or 0x137000 <= addr < 0x137400:
          ops.append((pos + 6 + i * 4, "write", addr, value))
      pos += 6 + count * 4; continue
    if op in (0x47, 0x48) and pos + 9 <= end:
      addr, mask = struct.unpack_from("<II", image, pos + 1)
      if 0x020000 <= addr < 0x021000 or 0x137000 <= addr < 0x137400:
        ops.append((pos, "andn" if op == 0x47 else "or", addr, mask))
      pos += 9; continue
    if op == 0x65 and pos + 13 <= end:
      addr, first, second = struct.unpack_from("<III", image, pos + 1)
      if 0x020000 <= addr < 0x021000 or 0x137000 <= addr < 0x137400:
        ops.extend(((pos, "write", addr, first), (pos + 4, "write", addr, second)))
      pos += 13; continue
    if op == 0x90 and pos + 9 <= end:        # COPY_ZM_REG
      src, dst = struct.unpack_from("<II", image, pos + 1)
      if ((0x020000 <= src < 0x021000 or 0x137000 <= src < 0x137400) and
          (0x020000 <= dst < 0x021000 or 0x137000 <= dst < 0x137400)):
        ops.append((pos, "copy", dst, src))
      pos += 9; continue
    if op == 0x97 and pos + 13 <= end:       # ZM_MASK_ADD
      addr, mask, add = struct.unpack_from("<III", image, pos + 1)
      if 0x020000 <= addr < 0x021000 or 0x137000 <= addr < 0x137400:
        ops.append((pos, "maskadd", addr, mask, add))
      pos += 13; continue
    if op == 0x5b and pos + 3 <= end:        # SUB_DIRECT
      target = struct.unpack_from("<H", image, pos + 1)[0]
      ops.extend(_vbios_target_ops(image, target, limit, _seen, _enabled))
      pos += 3; continue
    if op == 0x5c and pos + 3 <= end:        # JUMP
      target = struct.unpack_from("<H", image, pos + 1)[0]
      ops.extend(_vbios_target_ops(image, target, limit, _seen, _enabled))
      break
    if op == 0x6b and pos + 2 <= end:        # SUB (indexed script)
      index = image[pos + 1]
      table = _vbios_script_table(image)
      if table and table + index * 2 + 2 <= len(image):
        target = struct.unpack_from("<H", image, table + index * 2)[0]
        if target: ops.extend(_vbios_target_ops(image, target, limit, _seen, _enabled))
      pos += 2; continue
    if op == 0x58 and pos + 6 <= end:       # ZM_REG_SEQUENCE
      pos += 6 + image[pos + 5] * 4; continue
    if op in (0x33, 0x5b, 0x74, 0x75): pos += 2 if op in (0x33, 0x74, 0x75) else 3; continue
    pos += 1
    if op == 0x71: break                    # END
  return ops

def vbios_init_info(path):
  image = nvbios_init.find_vbios_image(pathlib.Path(path).read_bytes())
  scripts = nvbios_init.find_vbios_scripts(image)
  print(f"vbios-init: image_bytes={len(image)} scripts={len(scripts)}")
  for i, script in enumerate(scripts):
    print(f"  script[{i}]={script:#x}")
  return image, 0, scripts

def execute_vbios_target_ops(dev, image, script, dry_run=False):
  """Execute one NVINIT script using the full nvbios interpreter."""
  if not dry_run:
    nvbios_init.run_vbios_init(dev, image, [script],
                               debug=bool(DEBUG and getenv("KEPLER_VBIOS_TRACE", 0)))
  return 0

def program_gk104_gpc_pll(dev, target_khz=300000, ref_khz=810000):
  """Program CLK0 using the GK104 PLL layout from nouveau gk104.c."""
  best = None
  for p in range(1, 64):
    for m in range(17, 33):
      for n in range(8, 256):
        vco = ref_khz * n // m
        if not 1_100_000 <= vco <= 2_404_000: continue
        out = vco // p
        candidate = (abs(out - target_khz), out, p, n, m)
        if best is None or candidate < best: best = candidate
  if best is None: raise ValueError("no GK104 GPC PLL coefficient satisfies VBIOS limits")
  _, actual, p, n, m = best
  coef = (p << 16) | (n << 8) | m
  before = dev.read32(0x137000)
  if (before & 0xfffff000) == 0xbadf3000:
    raise RuntimeError("GPC PLL remains power/clock gated after VBIOS init")
  dev.write32(0x137000, before & ~0x5)
  dev.write32(0x137004, coef)
  dev.write32(0x137000, (before & ~0x5) | 0x1)
  time.sleep(0.01)
  after = dev.read32(0x137000)
  locked = (after & 0xfffff000) != 0xbadf3000 and bool(after & 0x00020000)
  print(f"gpc-pll: target={target_khz} actual={actual} ref={ref_khz} "
        f"P={p} N={n} M={m} coef={coef:#x} lock={'YES' if locked else 'NO'}")
  if not locked: return actual, False
  dev.write32(0x137000, after | 0x4)
  return actual, locked

def nvkm_mask(dev, addr, mask, val):
  """nouveau nvkm_mask: (r & ~mask) | (val & mask)."""
  r = dev.read32(addr)
  r = (r & ~mask) | (val & mask)
  dev.write32(addr, r)
  return r


def _set_native_thread_name(name):
  """Expose the active teardown stage in a process snapshot (macOS-only helper
  retained from the upstream file; a no-op on Linux)."""
  if sys.platform != "darwin": return
  try:
    pthread_setname_np = ctypes.CDLL(None).pthread_setname_np
    pthread_setname_np.argtypes = [ctypes.c_char_p]
    pthread_setname_np.restype = ctypes.c_int
    pthread_setname_np(name.encode("ascii", "replace")[:63])
  except Exception:
    pass

def _pmu_magic_(dev, ctrl, size):
  """nouveau gk104.c magic_(): poke the PMU 0xc800 sequencer (War00C800_0).
  Fully instrumented: reports the command written, whether the ready bit was
  observed, how many 0xc804 words were written, and the final 0xc800 value."""
  dev.write32(0x00c800, 0x00000000)
  dev.write32(0x00c808, 0x00000000)
  dev.write32(0x00c800, ctrl)
  ready = False
  deadline = time.time() + 2.0
  nwords = 0
  while time.time() < deadline:
    if dev.read32(0x00c800) & 0x40000000:
      for _ in range(size):
        dev.write32(0x00c804, 0x00000000)
        nwords += 1
      ready = True
      break
    time.sleep(0.0005)
  final = dev.read32(0x00c800)
  dev.write32(0x00c800, 0x00000000)
  print(f"  c800 cmd={ctrl:#010x} ready={'YES' if ready else 'TIMEOUT'} "
        f"c804_words={nwords} final_c800={final:#010x}")
  return ready

def _pmu_magic(dev, ctrl):
  _pmu_magic_(dev, 0x8000a41f | ctrl, 6)
  _pmu_magic_(dev, 0x80000421 | ctrl, 1)

def gk104_pmu_pgob(dev, war00c800=True, settle_s=0.05):
  """Mirror Nouveau's GK104 GR-init PGOB operation.

  This releases GPC/ROP/LTC power-gating so FECS can read the topology during
  bring-up.  Nouveau's GK104 GR code only calls the underlying operation with
  false (during oneinit/init); it has no shutdown caller with true.  Do not
  expose that unsupported reverse transition in this userspace API.

  The un-gate (0x020004 bit30) plus the 0x10a78c / 0x200 handshake require the
  PMU falcon to be alive (nouveau runs pgob after PMU subdev init).  The
  War00C800_0 0xc800 pokes also need the PMU; if the PMU is not running they
  simply time out (2s each) and proceed.  Runs TWICE in nouveau (oneinit +
  gr_init_); the un-gate is sticky, so once before ctxctl suffices for bring-up."""
  # fuse 0x31c bit0 gate skipped (set on real GK104).
  nvkm_mask(dev, 0x000200, 0x00001000, 0x00000000)   # clear GR reset (bit12)
  dev.read32(0x000200)                                # posted
  nvkm_mask(dev, 0x000200, 0x08000000, 0x08000000)   # set bit27
  time.sleep(settle_s)
  nvkm_mask(dev, 0x10a78c, 0x00000002, 0x00000002)
  nvkm_mask(dev, 0x10a78c, 0x00000001, 0x00000001)
  nvkm_mask(dev, 0x10a78c, 0x00000001, 0x00000000)
  nvkm_mask(dev, 0x020004, 0xc0000000, 0x40000000)
  time.sleep(settle_s)
  nvkm_mask(dev, 0x10a78c, 0x00000002, 0x00000000)
  nvkm_mask(dev, 0x10a78c, 0x00000001, 0x00000001)
  nvkm_mask(dev, 0x10a78c, 0x00000001, 0x00000000)
  nvkm_mask(dev, 0x000200, 0x08000000, 0x00000000)   # clear bit27
  nvkm_mask(dev, 0x000200, 0x00001000, 0x00001000)   # release GR reset (bit12)
  dev.read32(0x000200)                                # posted
  if war00c800:
    _pmu_magic(dev, 0x04000000)
    _pmu_magic(dev, 0x06000000)
    _pmu_magic(dev, 0x0c000000)
    _pmu_magic(dev, 0x0e000000)


def _gk104_shutdown_leaf_irqs(dev):
  """Mask and acknowledge GK104 interrupt producers before PCI teardown."""
  # First prevent any leaf source below from reaching the PCIe interrupt.
  dev.write32(0x000140, 0x00000000)  # PMC master interrupt mask
  dev.write32(0x000640, 0x00000000)  # MSI mask

  # PFIFO plus all three GK104 PBDMAs.
  dev.write32(0x002140, 0x00000000)
  for pbdma in range(3):
    q = 0x040000 + pbdma * 0x2000
    dev.write32(q + 0x10c, 0x00000000)
    dev.write32(q + 0x14c, 0x00000000)
    intr0, intr1 = dev.read32(q + 0x108), dev.read32(q + 0x148)
    if intr0: dev.write32(q + 0x108, intr0)
    if intr1: dev.write32(q + 0x148, intr1)

  # Exact g84_therm_fini(): PTherm mask/ACK and its PBUS parent ACK.
  dev.write32(0x020000, 0x00000000)
  dev.write32(0x020100, 0xffffffff)
  dev.write32(0x001100, 0x00010000)

  # GK104 GPIO and AUX/I2C masks are separate from their W1C status words.
  gpio0, gpio1 = dev.read32(0x00dc00), dev.read32(0x00dc80)
  dev.write32(0x00dc08, 0x00000000)
  dev.write32(0x00dc88, 0x00000000)
  if gpio0: dev.write32(0x00dc00, gpio0)
  if gpio1: dev.write32(0x00dc80, gpio1)
  aux = dev.read32(0x00dc60)
  dev.write32(0x00dc68, 0x00000000)
  if aux: dev.write32(0x00dc60, aux)

  # Exact gt215_pmu_fini mask, followed by ACK of any already-latched source.
  dev.write32(0x10a014, 0x00000060)
  pmu = dev.read32(0x10a004)
  if pmu: dev.write32(0x10a004, pmu)
  dev.read32(0x000140)  # flush posted writes


def mask32(dev, reg, mask, value):
  """Traced nvkm_mask: prints before/written/after so we can prove the central
  PGOB write actually retains 0x020004[31:30]==01b (or not)."""
  before = dev.read32(reg)
  written = (before & ~mask) | (value & mask)
  dev.write32(reg, written)
  _ = dev.read32(0x000200)        # flush PCIe posted write
  after = dev.read32(reg)
  print(f"mask32 reg={reg:#08x} before={before:#010x} "
        f"written={written:#010x} after={after:#010x}")
  return before, written, after


def nvkm_fuse_read_31c(dev):
  """Exact nouveau gf100_fuse_read(0x31c): gating fuse access then reading
  0x021100 + 0x31c = 0x02141c.  Returns the full 0x02141c value (bit0 gates PGOB)."""
  fuse_enable = nvkm_mask(dev, 0x022400, 0x800, 0x800)
  unk = nvkm_mask(dev, 0x021000, 0x1, 0x1)
  val = dev.read32(0x021100 + 0x31c)   # 0x02141c
  dev.write32(0x021000, unk)
  dev.write32(0x022400, fuse_enable)
  return val


GPC_CLK_REGS = (0x137100, 0x137160, 0x1371d0, 0x137250,
                0x137000, 0x137004, 0x00e800, 0x00e804,
                0x00e820, 0x00e824)

def probe_pgob_power_on(dev):
  """Standalone PGOB bring-up trace (no PMU/FECS firmware involved).  Run AFTER
  PMC_ENABLE, then read 0x409604 / 0x41a100 / 0x502100 to see if the GPC domain
  came online.  Mirrors nouveau gk104_pmu_pgob(enable=False) exactly."""
  print("=== PGOB pre-state ===")
  for reg in (0x000200, 0x020004, 0x10a78c, 0x00c800, 0x409604,
              0x400700, 0x409100, 0x41a100, 0x502100):
    print(f"pre  {reg:#08x} = {dev.read32(reg):#010x}")
  fuse = nvkm_fuse_read_31c(dev)
  print(f"pgob_fuse_31c (0x02141c) = {fuse:#010x}  bit0={'SET' if fuse & 1 else 'CLEAR'}")

  # Exact Nouveau gk104_pmu_pgob(..., false) core sequence.
  mask32(dev, 0x000200, 0x00001000, 0x00000000)
  _ = dev.read32(0x000200)
  mask32(dev, 0x000200, 0x08000000, 0x08000000)
  time.sleep(0.050)
  mask32(dev, 0x10a78c, 0x00000002, 0x00000002)
  mask32(dev, 0x10a78c, 0x00000001, 0x00000001)
  mask32(dev, 0x10a78c, 0x00000001, 0x00000000)
  _, wanted, observed_immediate = mask32(dev, 0x020004, 0xc0000000, 0x40000000)
  time.sleep(0.050)
  observed_delayed = dev.read32(0x020004)
  print(f"PGOB control wanted={wanted:#010x} "
        f"immediate={observed_immediate:#010x} delayed={observed_delayed:#010x}")
  mask32(dev, 0x10a78c, 0x00000002, 0x00000000)
  mask32(dev, 0x10a78c, 0x00000001, 0x00000001)
  mask32(dev, 0x10a78c, 0x00000001, 0x00000000)
  mask32(dev, 0x000200, 0x08000000, 0x00000000)
  mask32(dev, 0x000200, 0x00001000, 0x00001000)
  _ = dev.read32(0x000200)

  # chipset-0xe4 0xc800 workaround (do not omit).
  print("=== PGOB c800 workaround (enable=False) ===")
  for c in (0x04000000, 0x06000000, 0x0c000000, 0x0e000000):
    _pmu_magic(dev, c)

  print("=== PGOB post-state ===")
  for reg in (0x000200, 0x020004, 0x10a78c, 0x409604,
              0x400700, 0x409100, 0x41a100, 0x502100):
    print(f"post {reg:#08x} = {dev.read32(reg):#010x}")
  print("=== GPC clock registers (read-only) ===")
  for reg in GPC_CLK_REGS:
    print(f"clk  {reg:#08x} = {dev.read32(reg):#010x}")


def probe_gpc_fixed_100mhz(dev):
  """User-specified discriminator: select Nouveau's board-independent fixed
  100 MHz bypass source (0x137160=2) and clear the 0x137100 PLL-select bit,
  then re-read 0x409604 / 0x41a100 / 0x502100.  Needs NO GPC PLL, NO VBIOS,
  NO VCO.  If the GPC domain wakes (0x409604 != 0, GPC regs != 0xbadf3000)
  the mux was the blocker; if not, the blocker is missing VBIOS/devinit GR-domain
  init (deeper than a PLL)."""
  def rd(reg): return dev.read32(reg)
  def mask(reg, bits, value):
    before = rd(reg)
    written = (before & ~bits) | (value & bits)
    dev.write32(reg, written)
    _ = rd(0x000200)        # flush posted PCIe write
    after = rd(reg)
    print(f"{reg:#08x}: before={before:#010x} write={written:#010x} after={after:#010x}")
    return after
  print("=== GPC clock before ===")
  for reg in (0x20200, 0x137100, 0x137160, 0x1371d0, 0x137250, 0x137000,
              0x409604, 0x41a100, 0x502100):
    print(f"{reg:#08x} = {rd(reg):#010x}")
  # PTHERM engine clock-gating: GR engine (0x20200+0x00) is in AUTO mode, so the
  # idle GPC clock is gated (-> 0xbadf3000 sentinel).  Force ENG_CLK=RUN
  # (nouveau gk104_clkgate_fini low byte 0x54) so the GPC clock domain stays on.
  print("=== PTHERM clock-gate: force GR ENG_CLK=RUN ===")
  mask(0x20200, 0x000000ff, 0x00000054)
  print(f"0x20200 = {rd(0x20200):#010x}")
  # GR/GPC domain power+clock release via RED_SWITCH (nouveau gf100_gr_fecs_reset:
  # write 0x409614=0x70 then enable 0x700 => POWER+ENABLE for MAIN/GPC/ROP).
  # Board-independent; no VBIOS, no FECS firmware needed.  Do this FIRST so the
  # GPC clock domain is powered when we select a clock source.
  print("=== GR RED_SWITCH (0x409614) domain release ===")
  dev.write32(0x409614, 0x00000070)   # POWER_MAIN|GPC|ROP
  time.sleep(0.00001)
  mask(0x409614, 0x00000700, 0x00000700)  # ENABLE_MAIN|GPC|ROP
  time.sleep(0.00001)
  print(f"0x409614 = {rd(0x409614):#010x}")
  # GPC clock: DIV (bypass) mode, source from a RUNNING VCO / fixed ref -- no GPC PLL
  # (0x137000 is power/clock-gated, can't be written).  gf100_div_src encoding:
  # SRC[3:0]=3 (SRC3), VCO[8]=0/1 (RPLL_e800/e820), SRC0[17:16]=3 (108MHz).
  mask(0x137100, 0x00000001, 0x00000000)  # SRC_SEL CLK0 = DIV (not GPC PLL)
  for src, name in (
      (0x00000002, "fixed 100MHz (SRC=2)"),
      (0x00000003, "RPLL_e800 (SRC=3,VCO=0)"),
      (0x00000103, "RPLL_e820 (SRC=3,VCO=1)"),
      (0x00030000, "fixed 108MHz (SRC0=3)"),
  ):
    dev.write32(0x137160, src)
    _ = rd(0x000200)
    time.sleep(0.010)
    print(f"0x137160={src:#010x} ({name}): 0x409604={rd(0x409604):#010x} "
          f"0x41a100={rd(0x41a100):#010x} 0x502100={rd(0x502100):#010x} "
          f"0x137000={rd(0x137000):#010x}")
  print("=== final state ===")
  for reg in (0x409604, 0x41a100, 0x502100, 0x137000, 0x137160, 0x409614):
    print(f"{reg:#08x} = {rd(reg):#010x}")
  # Decisive: is the GPC PLL block (0x137000) even writable?  If a write sticks,
  # the block is accessible and we can program the PLL (needs VBIOS coef); if it
  # keeps returning 0xbadf3000, the block is truly power/clock-gated and requires
  # VBIOS devinit to release.  gf100_pll_ctrl: bit0=ENABLE, bit1=PWROFF,
  # bit16=PLL_PWR.
  print("=== try direct 0x137000 write ===")
  for val in (0x00000000, 0x00000010, 0x00010011):
    dev.write32(0x137000, val)
    _ = rd(0x000200)
    print(f"write 0x137000={val:#010x} -> readback={rd(0x137000):#010x}")


def falcon_load(dev, base, imem, dmem, entry=0, start=True):
  """Load `imem`/`dmem` into the FALCON at MMIO `base` (raw, no bin-header) and
  optionally start it.  For GK104 GR we load FECS+GPCCS first, then start FECS."""
  falcon_write_dmem(dev, base, dmem)
  falcon_write_imem(dev, base, imem)
  dev.write32(base + FALCON_UC_ENTRY, entry)
  if start:
    falcon_start(dev, base)
  return

def falcon_csdata_write(dev, base, starstar, words):
  """nouveau gf100_gr_init_csdata(): upload a GR register-init *method stream*
  into the falcon DMEM via the 0x1c0/0x1c4 host->falcon method interface.

  The method stream only encodes (reg_addr, count, pitch); the actual init
  values live in the FUC data file already loaded to DMEM.  `words` is the flat
  list of 32-bit method words (xfer<<26 | addr) — see grctx_gk104.method_stream.
  `starstar` is the csdata sub-stream selector (0 hub, 0 gpc, 4 tpc, 8 ppc)."""
  # 1. query the current DMEM write pointer so we append after the FUC data.
  #    The FUC data stores (head, tail) at offset starstar.  We read them
  #    separately (explicit index per read) because some hardware does not
  #    auto-increment the DATA register reliably.
  #    NOTE: reads can return 0xbadfXXXX (power-gate sentinel) on some hardware.
  star = 0xbadf5000
  temp = 0xbadf5000
  for attempt in range(20):
    dev.write32(base + FALCON_DATA_INDEX, 0x02000000 + starstar)
    star = dev.read32(base + FALCON_DATA)
    dev.write32(base + FALCON_DATA_INDEX, 0x02000000 + starstar + 4)
    temp = dev.read32(base + FALCON_DATA)
    if (star & 0xffff0000) != 0xbadf0000 and (temp & 0xffff0000) != 0xbadf0000:
      break
    time.sleep(0.002)
  if (star & 0xffff0000) == 0xbadf0000:
    star = temp
  if (temp & 0xffff0000) == 0xbadf0000:
    temp = star
  if (star & 0xffff0000) == 0xbadf0000:
    # All retries failed; compute fallback from FUC data size.
    star = 0x304 if base == 0x409000 else 0x6c
    print(f"[kepler] csdata star: FALLBACK to {star:#x} for base={base:#x} starstar={starstar:#x}")
  else:
    if DEBUG:
      print(f"[kepler] csdata star: star={star:#x} temp={temp:#x} base={base:#x} starstar={starstar:#x} (attempt {attempt})")
  if temp > star:
    star = temp
  # 2. set DMEM write offset = star, then stream the method words.
  dev.write32(base + FALCON_DATA_INDEX, FALCON_IDX_WRITE | star)
  for w in words:
    dev.write32(base + FALCON_DATA, w)
    star += 4
  # 3. finalize the sub-stream (tell the falcon where it ends).
  #    nouveau does NOT increment star for the last word, then writes star+4.
  #    We increment star for every word, so the tail is just star (no +4).
  dev.write32(base + FALCON_DATA_INDEX, 0x01000004 + starstar)
  dev.write32(base + FALCON_DATA, star)

def find_kepler_firmware():
  """Locate a GK104 firmware tree (plan §24.1).  Returns the dir containing the
  FECS/ GPCCS ucode .bin files extracted from the nouveau embedded FUC arrays
  (hubgk104.fuc3.h / gpcgk104.fuc3.h), or None."""
  here = os.path.dirname(os.path.abspath(__file__))
  for d in (os.environ.get("NV_FIRMWARE_DIR"),
            os.path.join(here, "..", "firmware", "gk104"),
            os.path.join(here, "firmware", "gk104"),
            os.path.expanduser("~/nvidia/gk104"),
            "/usr/lib/firmware/nvidia/gk104",
            "/lib/firmware/nvidia/gk104"):
    if d and os.path.isfile(os.path.join(d, "gk104_fecs_code.bin")):
      return os.path.abspath(d)
  return None


# ============================================================================
# Kepler compute launch — Compute Work Descriptor (CWD), method 0x0910
# ----------------------------------------------------------------------------
# Kepler does NOT use a QMD (that is Maxwell+).  The launch is driven by writing
# a "Compute Work Descriptor" into a buffer and pointing the KEPLER_COMPUTE
# method LAUNCH_DESC_ADDRESS (0x02b4, value is the CWD VA shifted right by 8) at
# it, then issuing LAUNCH (0x02bc) to trigger the kernel.
# KEPLER-TODO: the exact CWD field layout (grid/block dims, program counter /
# shader VA, register count, shared-mem size, parameter buffer VA) must be
# matched to the GK104 compute class.  The launch-word builder below mirrors the
# GA102 semaphore+invalidate+launch structure but substitutes the Kepler method
# numbers; verify against nouveau's nvc0_compute.c launch path.
# ============================================================================
METHOD_NAMES = {
  0x0000: "SET_OBJECT",
  0x0010: "NV906F_SEMAPHORE_ADDRESS_HIGH",
  0x0014: "NV906F_SEMAPHORE_ADDRESS_LOW",
  0x0018: "NV906F_SEMAPHORE_SEQUENCE",
  0x001c: "NV906F_SEMAPHORE_TRIGGER",
  0x02b4: "KEPLER_COMPUTE_LAUNCH_DESC_ADDRESS",
  0x02bc: "KEPLER_COMPUTE_LAUNCH",
  0x160c: "KEPLER_COMPUTE_CODE_ADDRESS_LO",
  0x1608: "KEPLER_COMPUTE_CODE_ADDRESS_HI",
  0x1698: "INVALIDATE_SHADER_CACHES",
  0x0020: "NON_STALL_INTERRUPT",
}

def nvm(subchannel, method, *args, typ=2):
  return [(typ << 28) | (len(args) << 16) | (subchannel << 13) | (method >> 2), *args]

def gk104_semaphore(addr, value, operation):
  """Emit the GK104/NV906F subchannel semaphore sequence."""
  return [*nvm(0, 0x0010, (addr >> 32) & 0xffffffff,
              addr & 0xffffffff, value, operation)]

def build_launch_words(timeline_addr, wait_value, done_value, launch_desc_addr,
                       code_va=0, temp_va=0, temp_size=0x40000):
  # Kepler compute launch (envytools gk104_compute.xml, plan §24.3): bind the
  # compute class via SET_OBJECT, set the shader PC via CODE_ADDRESS_LO/HI
  # (0x160c/0x1608), point at the CWD via LAUNCH_DESC_ADDRESS (0x02b4, VA>>8),
  # then LAUNCH (0x02bc, value=3) to trigger.  Semaphore + cache-invalidate wrap.
  # Setup methods match Mesa's nve4_compute_setup_state() for Kepler.
  # GTX 770 has eight MPs.  NVE4 exposes two identical per-MP TEMP size banks;
  # Mesa programs both and rounds the low word down to a 32-KiB granule.
  temp_per_mp = temp_size // 8
  return [
    *nvm(1, 0x0000, KEPLER_COMPUTE_A),            # SET_OBJECT: bind compute class
    *nvm(1, 0x0790, temp_va >> 32, temp_va & 0xffffffff), # TEMP_ADDRESS HIGH/LOW
    *nvm(1, 0x02e4, temp_per_mp >> 32, temp_per_mp & ~0x7fff, 0xff), # MP_TEMP_SIZE[0]
    *nvm(1, 0x02f0, temp_per_mp >> 32, temp_per_mp & ~0x7fff, 0xff), # MP_TEMP_SIZE[1]
    *nvm(1, 0x077c, 0xff << 24),                  # LOCAL_BASE
    *nvm(1, 0x0214, 0xfe << 24),                  # SHARED_BASE
    *nvm(1, 0x0310, 0x300),                       # SASS_VERSION (Kepler)
    *nvm(1, 0x2608, 7),                           # TEX_CB_INDEX (Mesa NVE4 setup)
    *nvm(1, 0x1698, 0x00001011),                  # INVALIDATE_SHADER_CACHES
    *nvm(1, 0x1608, code_va >> 32, code_va & 0xffffffff), # CODE_ADDRESS HIGH/LOW
    *nvm(1, 0x02b4, launch_desc_addr >> 8),       # LAUNCH_DESC_ADDRESS (VA<<8 by HW)
    *nvm(1, 0x02bc, 0x3),                          # LAUNCH (trigger, value=3)
    *nvm(1, 0x0110, 0),                            # NV50_GRAPH_SERIALIZE
    *gk104_semaphore(timeline_addr, done_value, 0x00000002), # RELEASE, WFI enabled
    *nvm(0, 0x0020, 0),
  ]

def decode_words(words):
  index = 0
  while index < len(words):
    header = words[index]
    typ, size, subc, method = (header >> 28) & 0xf, (header >> 16) & 0xfff, (header >> 13) & 0x7, (header << 2) & 0x7fff
    args = words[index + 1:index + 1 + size]
    yield index, typ, subc, method, METHOD_NAMES.get(method, f"UNKNOWN_0x{method:x}"), args
    index += size + 1

def describe_args(method, args):
  if method == 0x0010 and len(args) == 4:
    sem_addr = (args[0] << 32) | args[1]
    return [f"sem_addr=0x{sem_addr:x}", f"sequence={args[2]}", f"trigger=0x{args[3]:08x}"]
  if method == 0x02b4 and len(args) == 1:
    return [f"cwd_addr=0x{args[0] << 8:x}"]
  if method == 0x02bc and len(args) == 1:
    return [f"launch_trigger=0x{args[0]:x}"]
  if method == 0x1698 and len(args) == 1:
    return [f"invalidate_flags=0x{args[0]:08x}"]
  return [f"arg{i}=0x{arg:08x}" for i, arg in enumerate(args)]

def build_cuda_param_cbuf(*ptrs):
  cbuf = bytearray(0x200)
  struct.pack_into(f"<{len(ptrs)}Q", cbuf, 0x140, *ptrs)
  return cbuf

def build_cwd(code_addr, grid, block, shared=0, cbuf_addr=0, cbuf_size=0x200, regs=4):
  """Build a GK104 Compute Work Descriptor (QMD) using the NVA0C0_QMDV00_06
  field layout from NVIDIA's cla0c0qmd.h, matching Mesa's
  nve4_compute_setup_launch_desc().

  The QMD is 256 bytes (64 words).  Field bit positions use MW(X:Y) notation
  where X/Y are absolute bit indices across the entire structure.  Key fields:
    word 7  (0x1c): INVALIDATE_* caches (bits 250-255)
    word 8  (0x20): PROGRAM_OFFSET (bits 287:256)
    word 11 (0x2c): RELEASE_MEMBAR, CWD_MEMBAR, API_VISIBLE_CALL_LIMIT
    word 12 (0x30): CTA_RASTER_WIDTH  (grid X, 32-bit)
    word 13 (0x34): CTA_RASTER_HEIGHT (grid Y, bits 15:0) | CTA_RASTER_DEPTH (grid Z, bits 31:16)
    word 17 (0x44): SHARED_MEMORY_SIZE
    word 18 (0x48): CTA_THREAD_DIMENSION0 (block X, bits 31:16)
    word 19 (0x4c): CTA_THREAD_DIMENSION1/2 (block Y bits 15:0, Z bits 31:16)
    word 20 (0x50): CONSTANT_BUFFER_VALID (bits 0:7), L1_CONFIGURATION (bits 31:29)
    word 29 (0x74): CONSTANT_BUFFER_ADDR_LOWER(0)
    word 30 (0x78): CB_ADDR_UPPER(0) + CB_SIZE(0) (bits 31:15)
    word 45 (0xb4): SHADER_LOCAL_MEMORY_LOW_SIZE (bits 23:0), BARRIER_COUNT (bits 31:27)
    word 46 (0xb8): SHADER_LOCAL_MEMORY_HIGH_SIZE (bits 23:0), REGISTER_COUNT (bits 31:24)
    word 47 (0xbc): SHADER_LOCAL_MEMORY_CRS_SIZE (bits 23:0), SASS_VERSION (bits 31:24)
  """
  grid_x, grid_y, grid_z = grid
  block_x, block_y, block_z = block
  qmd = 0

  def field(lo, hi, value):
    """Insert one cla0c0qmd.h MW(hi:lo) field into the little-endian QMD."""
    nonlocal qmd
    width = hi - lo + 1
    assert 0 <= value < (1 << width), (lo, hi, value)
    qmd |= value << lo

  # Word 7 (0x1c): invalidate texture/sampler/data/shader/constant caches.
  # Bits 250-255 = INVALIDATE_{TEXTURE_HEADER,TEXTURE_SAMPLER,TEXTURE_DATA,
  # SHADER_DATA,INSTRUCTION,SHADER_CONSTANT}_CACHE.  Mesa sets all except
  # INSTRUCTION_CACHE.  0xbc = 1011_1100 → bits 255:250 = 101110.
  field(250, 255, 0x2f)
  # Word 8 (0x20): PROGRAM_OFFSET = code_addr (offset from CODE_ADDRESS method).
  field(256, 287, code_addr & 0xffffffff)
  # Word 11 (0x2c): RELEASE_MEMBAR=FE_SYSMEMBAR(bit14), CWD_MEMBAR=L1_SYSMEMBAR
  # (bits17:16=01), API_VISIBLE_CALL_LIMIT=NO_CHECK(bit26).
  field(366, 366, 1)
  field(368, 369, 1)
  field(378, 378, 1)
  # Words 12-13 (0x30-0x34): CTA_RASTER_WIDTH (32-bit) and
  # CTA_RASTER_HEIGHT (bits 15:0) | CTA_RASTER_DEPTH (bits 31:16) packed.
  # Confirmed by Mesa's indirect launch: grid_z written to desc+0x36.
  field(384, 415, grid_x)
  field(416, 431, grid_y)
  field(432, 447, grid_z)
  # Word 17 (0x44): SHARED_MEMORY_SIZE (align 0x100).
  field(544, 561, round_up(shared, 0x100))
  # Word 18 (0x48): CTA_THREAD_DIMENSION0 in bits 31:16.
  field(592, 607, block_x)
  # Word 19 (0x4c): CTA_THREAD_DIMENSION1 (bits 15:0) / DIMENSION2 (bits 31:16).
  field(608, 623, block_y)
  field(624, 639, block_z)
  # Word 20 (0x50): CB0 valid (bit 0) + L1_CONFIG=16KB (bits 31:29 = 001).
  field(640, 640, 1)
  field(669, 671, 1)
  # Word 29 (0x74): CONSTANT_BUFFER_ADDR_LOWER(0).
  field(928, 959, cbuf_addr & 0xffffffff)
  # Word 30 (0x78): CB_ADDR_UPPER(0) bits 7:0 + CB_SIZE(0) bits 31:15.
  field(960, 967, (cbuf_addr >> 32) & 0xff)
  field(975, 991, cbuf_size)
  # Word 45 (0xb4): SHADER_LOCAL_MEMORY_LOW_SIZE=0 (bits 23:0), BARRIER_COUNT=0.
  field(1440, 1463, 0)
  field(1467, 1471, 0)
  # Word 46 (0xb8): SHADER_LOCAL_MEMORY_HIGH_SIZE=0 + REGISTER_COUNT (bits 31:24).
  field(1472, 1495, 0)
  field(1496, 1503, regs & 0xff)
  # Word 47 (0xbc): SHADER_LOCAL_MEMORY_CRS_SIZE=0x800 + SASS_VERSION=0x30.
  field(1504, 1527, 0x800)
  field(1528, 1535, 0x30)
  return qmd.to_bytes(0x100, "little")


# NVProgram / NVAllocator / NVSignal are placeholders until the Kepler RM path
# exists; they mirror the GA102 structure.
class NVSignal:
  def __init__(self, owner=None): self.owner = owner; self.value_addr = 0; self.value = 0
class NVProgram:
  def __init__(self, dev, name, lib): self.dev, self.name = dev, name; self.cubin = lib
  kernargs_alloc_size = 256
class NVAllocator:
  def __init__(self, dev): self.dev = dev
  def alloc(self, size, **kwargs): return self.dev.iface.alloc(size, **kwargs)
  def _copyin(self, dst, src):
    # dev.vram is a CPU-coherent buffer on both backends (host bytearray for
    # software, mmap'd sysmem for hardware), so a plain slice write is enough.
    mv = dst.cpu_view()
    mv[:len(src)] = src
  def _copyout(self, dst, src):
    mv = src.cpu_view()
    dst[:] = mv[:len(dst)]


MIDDLE_CUBIN_BYTES = 2856
MIDDLE_LAUNCH_WORDS = 39

def kepler_selftest():
  """Tier 1 offline gate (no hardware required): cubin structure + GMMU helpers
  + launch-word builder + shared scaffolding sanity."""
  cubin = build_cubin()
  assert len(cubin) == MIDDLE_CUBIN_BYTES, f"cubin size {len(cubin)} != {MIDDLE_CUBIN_BYTES}"
  sha = hashlib.sha256(cubin).hexdigest()
  # ELF header sanity: must be an ELF, EM_CUDA, with our section/program tables.
  assert cubin[:4] == b"\x7fELF", "cubin is not an ELF"
  assert struct.unpack_from("<H", cubin, 0x12)[0] == CubinHelper.EM_CUDA, "not EM_CUDA"
  eh = elf_loader(cubin)
  assert eh["shnum"] == len(ch.SECTION_HEADERS), "section header count mismatch"
  assert eh["phnum"] == len(ch.PROGRAM_HEADERS), "program header count mismatch"

  words = build_launch_words(0xdeadbeef00001000, 3, 7, 0x2000, 0x3000)
  decoded = list(decode_words(words))
  assert len(words) == MIDDLE_LAUNCH_WORDS, f"launch word count {len(words)} != {MIDDLE_LAUNCH_WORDS}"
  assert any(m == 0x02bc for _, _, _, m, _, _ in decoded), "expected KEPLER_COMPUTE_LAUNCH method"
  assert any(m == 0x02b4 for _, _, _, m, _, _ in decoded), "expected KEPLER_COMPUTE_LAUNCH_DESC_ADDRESS method"
  sem_methods = [m for _, _, _, m, _, _ in decoded if m == 0x0010]
  assert len(sem_methods) == 1, "expected one GK104 completion semaphore sequence"
  assert gk104_mp_trap_addrs(0, 0) == (0x504648, 0x504650)
  assert gk104_mp_trap_addrs(1, 1) == (0x50ce48, 0x50ce50)
  params = build_cuda_param_cbuf(0x1122334455667788, 2, 3)
  assert len(params) == 0x200 and struct.unpack_from("<3Q", params, 0x140) == (0x1122334455667788, 2, 3)
  regs = cubin_register_count(cubin, "E_4")
  assert regs > 0
  cwd = build_cwd(0, (1, 1, 1), (256, 1, 1), cbuf_addr=0x123400, regs=regs)
  # Verify QMD field offsets match NVA0C0_QMDV00_06 spec.
  assert struct.unpack_from("<I", cwd, 0x18)[0] == 0, "reserved word 6 must remain zero"
  assert struct.unpack_from("<I", cwd, 0x1c)[0] == 0xbc000000, "invalidate bits at wrong offset"
  assert struct.unpack_from("<I", cwd, 0x20)[0] == 0, "PROGRAM_OFFSET should be 0"
  assert struct.unpack_from("<I", cwd, 0x30)[0] == 1, "CTA_RASTER_WIDTH"
  assert struct.unpack_from("<I", cwd, 0x34)[0] == 1 | (1 << 16), "CTA_RASTER_HEIGHT|DEPTH"
  assert struct.unpack_from("<I", cwd, 0x78)[0] == (0x200 << 15)
  assert struct.unpack_from("<I", cwd, 0xb8)[0] == (regs << 24)

  # helpers sanity
  assert lo32(0x123456789abcdef0) == 0x9abcdef0
  assert hi32(0x123456789abcdef0) == 0x12345678
  assert round_up(17, 16) == 32
  assert ceildiv(17, 16) == 2
  assert wait_cond(lambda: 1, value=1, timeout_ms=100)
  arr = array.array('I', [0, 1, 2, 3]); arr[1] = 0x42
  assert arr[1] == 0x42 and arr[2] == 2

  # GK104 GMMU helper sanity: PTE bit construction (no device needed)
  pte = GK104PageTableEntry(None, 0, 0)
  # Writable VRAM leaf: PRESENT set, READ_ONLY (bit 3) clear, TARGET=VRAM.
  leaf = pte.PTE_VALID | (0x1000 >> 8) | pte.PTE_APER_VRAM
  assert leaf & pte.PTE_VALID, "Kepler PTE must be valid"
  assert not (leaf & pte.PTE_READ_ONLY), "Kepler leaf PTE must be writable (READ_ONLY clear)"
  assert (leaf & (0x3 << 33)) == pte.PTE_APER_VRAM, "Kepler PTE aperture must be VRAM"

  # GK104/GF100 2-level walk sanity (software VRAM stand-in, no hardware needed)
  class _FakeDev:
    def __init__(self, vram): self.vram = vram; self.mm = None
  fd = _FakeDev(memoryview(bytearray(1 << 20)))
  fd.mm = GK104MemoryManager(fd, 1 << 20, 1 << 19)
  mp = fd.mm.map_range(0x1000, 0x3000, [(0x2000, 0x3000)], AddrSpace.PHYS)
  pgd = fd.mm.root_page_table
  pgd_idx, spt_idx = (0x1000 >> 27) & 0x1FFF, (0x1000 >> 12) & 0x7FFF
  assert pgd.valid(pgd_idx), "PGD entry must be present"
  spt = GK104PageTableEntry(fd, pgd.address(pgd_idx), lv=0)
  assert spt.valid(spt_idx), "PTE must be present"
  # leaf frame should resolve back to the mapped physical page
  assert spt.address(spt_idx) == 0x2000, "PTE frame must match paddr"
  assert mp.size == 0x3000 and mp.paddrs == [0x2000]

  # LinuxPCIDevice raw-MMIO transport sanity: build a fake sysfs tree in tmpfs
  # and exercise BAR mmap, config-space R/W, the freeze / final-output-budget
  # state machine, and fini cleanup — no real hardware, no root needed.
  import tempfile as _tf, shutil as _shutil
  _fake_root = _tf.mkdtemp(prefix="kepler_pcie_selftest_")
  _fake_bdf = "0000:09:00.0"
  _fake_dev_dir = pathlib.Path(_fake_root) / _fake_bdf
  _fake_dev_dir.mkdir(parents=True)
  # resource file: BAR0 = 16 MiB at 0xf0000000, BAR1 = 4 KiB at 0xe0000000,
  # remaining bars/rom 0.  Lines: "start end flags".
  (_fake_dev_dir / "resource").write_text(
      "0xf0000000 0xf0ffffff 0x00000000\n"   # BAR0 16 MiB
      "0xe0000000 0xe0000fff 0x00000000\n"   # BAR1 4 KiB
      "0x00000000 0x00000000 0x00000000\n"   # BAR2
      "0x00000000 0x00000000 0x00000000\n"   # BAR3
      "0x00000000 0x00000000 0x00000000\n"   # BAR4
      "0x00000000 0x00000000 0x00000000\n"   # BAR5
      "0x00000000 0x00000000 0x00000000\n")  # ROM
  # config: 256 bytes; vendor=0x10de (offset 0), device=0x1180 (offset 2).
  _cfg = bytearray(256)
  _cfg[0:2] = (0x10de).to_bytes(2, "little")
  _cfg[2:4] = (0x1180).to_bytes(2, "little")
  _cfg[4:6] = (0x0200).to_bytes(2, "little")  # class code (VGA)
  (_fake_dev_dir / "config").write_bytes(bytes(_cfg))
  # BAR0 backing: 16 MiB tmpfs file; BAR1 backing: 4 KiB.
  with open(_fake_dev_dir / "resource0", "wb") as _f:
    _f.truncate(16 << 20)
  with open(_fake_dev_dir / "resource1", "wb") as _f:
    _f.truncate(4096)
  _saved_root = LinuxPCIDevice._sysfs_root
  LinuxPCIDevice._sysfs_root = _fake_root
  try:
    hw = LinuxPCIDevice(bdf=_fake_bdf, dev_id=7)
    # bar_info parsed from the resource file.
    b0_base, b0_size = hw.bar_info(0)
    assert b0_base == 0xf0000000 and b0_size == (16 << 20), f"BAR0 info {hw.bar_info(0)}"
    assert hw.bar_info(1) == (0xe0000000, 4096), f"BAR1 info {hw.bar_info(1)}"
    assert hw.bar_info(5) == (0, 0), "absent BAR must be (0,0)"
    # MMIO read/write through the mmap'd BAR0.
    hw.mmio_write32(0, 0x1000, 0xdeadbeef)
    assert hw.mmio_read32(0, 0x1000) == 0xdeadbeef, "BAR0 32-bit round-trip"
    hw.mmio_write(0, 0x2000, struct.pack("<2I", 0x11223344, 0x55667788))
    assert hw.mmio_read(0, 0x2000, 8) == struct.pack("<2I", 0x11223344, 0x55667788), "BAR0 bulk read"
    # 64-bit BAR0 read (NVDevice.read64 uses mmio_read64(offset)).
    assert hw.mmio_read64(0x2000) == (0x55667788 << 32) | 0x11223344, "mmio_read64"
    # map_bar returns a direct MMIOInterface over the mmap.
    bar0 = hw.map_bar(0, fmt='I')
    assert bar0[0x1000 // 4] == 0xdeadbeef, "map_bar view read"
    # PCI config space via the sysfs `config` file.
    assert hw.read_config(0, 2) == 0x10de, "config vendor read"
    assert hw.read_config(2, 2) == 0x1180, "config device read"
    # write_config_flush: set PCI_COMMAND bit0 and read it back.
    _obs = hw.write_config_flush(4, 0x0007, 2)
    assert _obs == 0x0007, f"config writeback {_obs:#x}"
    # alloc_sysmem returns a host buffer + page bus paddrs.
    _mv, _paddrs = hw.alloc_sysmem(0x3000, contiguous=True)
    assert len(_mv) == 0x3000 and len(_paddrs) == 3, f"alloc_sysmem {len(_paddrs)} pages"
    _mv[0:4] = struct.pack("<I", 0xcafef00d)
    assert hw.sysmem_read(_paddrs[0], 4) == b"\x0d\xf0\xfe\xca", "sysmem_read round-trip"
    hw.sysmem_write(_paddrs[1], b"\xaa\xbb\xcc\xdd")
    assert _mv[0x1000:0x1004] == b"\xaa\xbb\xcc\xdd", "sysmem_write round-trip"
    # arm_final_output_read allows exactly one matching BAR read, then consumed.
    _out = bytes(range(16))
    hw.mmio_write(1, 0, _out)
    hw.arm_final_output_read(1, 0, len(_out))
    try:
      hw.mmio_read(0, 0x1000, 4)
      raise AssertionError("BAR0 read escaped final output budget")
    except RuntimeError as _e:
      assert "outside final output budget" in str(_e)
    assert hw.mmio_read(1, 0, len(_out)) == _out, "final output read"
    try:
      hw.mmio_write32(0, 0x1000, 0x1)
      raise AssertionError("BAR0 write escaped consumed final output budget")
    except RuntimeError as _e:
      assert "after final output read" in str(_e)
    # freeze rejects all further MMIO; set_phase still allowed.
    hw.freeze("output-read-complete")
    hw.set_phase("client-close")
    try:
      hw.mmio_read(1, 0, 4)
      raise AssertionError("MMIO escaped transport freeze")
    except RuntimeError as _e:
      assert "after freeze" in str(_e)
    hw.fini()
    # fini is idempotent and clears the BAR mappings.
    hw.fini()
    assert hw._bars == {}, "fini did not release BAR mappings"
    # probe() against the fake tree returns the device; a bad bdf returns None.
    assert LinuxPCIDevice.probe(bdf=_fake_bdf) is not None, "probe should succeed on fake tree"
    LinuxPCIDevice._sysfs_root = "/sys/bus/pci/devices"
    assert LinuxPCIDevice.probe(bdf="0000:ff:00.0") is None, "probe on missing bdf must be None"
    LinuxPCIDevice._sysfs_root = _fake_root
  finally:
    LinuxPCIDevice._sysfs_root = _saved_root
    _shutil.rmtree(_fake_root, ignore_errors=True)

  # NVDevice.close() must call fini on its LinuxPCIDevice without raising.
  _close_hw = object.__new__(LinuxPCIDevice)
  _close_hw._bars = {}; _close_hw._config_fd = None; _close_hw._fini_done = False
  _close_dev = object.__new__(NVDevice)
  _close_dev.backend = "kepler"
  _close_dev.iface = type("_FakeIface", (), {"pci_dev": _close_hw})()
  _close_dev.close()  # must not raise; client-only close, no reset
  assert _close_hw._fini_done, "NVDevice.close did not finalize the transport"


  # GK104 shutdown must silence leaf interrupts without inventing a reverse
  # PGOB transition or turning off the active PLL.  Nouveau has neither
  # operation in GK104 device fini.
  class _FakeRegs:
    def __init__(self):
      self.regs = {
        0x000200: 0xffffffff, 0x020004: 0x40000000,
        0x137000: 0x00000005, 0x137100: 0x00000001,
        0x00dc00: 0x12, 0x00dc80: 0x34, 0x00dc60: 0x56,
        0x10a004: 0x80,
      }
      for pbdma in range(3):
        q = 0x040000 + pbdma * 0x2000
        self.regs[q + 0x108] = 1 << pbdma
        self.regs[q + 0x148] = 0x10 << pbdma
      self.writes = []
    def read32(self, reg): return self.regs.get(reg, 0)
    def write32(self, reg, value):
      self.regs[reg] = value & 0xffffffff
      self.writes.append((reg, value & 0xffffffff))
  fake_regs = _FakeRegs()
  _gk104_shutdown_leaf_irqs(fake_regs)
  assert fake_regs.regs[0x020004] == 0x40000000
  assert fake_regs.regs[0x137100] == 0x00000001
  assert fake_regs.regs[0x137000] == 0x00000005
  for reg in (0x000140, 0x000640, 0x002140, 0x020000,
              0x00dc08, 0x00dc88, 0x00dc68):
    assert fake_regs.regs[reg] == 0, f"shutdown mask {reg:#x} left enabled"
  for pbdma in range(3):
    q = 0x040000 + pbdma * 0x2000
    assert fake_regs.regs[q + 0x10c] == fake_regs.regs[q + 0x14c] == 0

  print(f"kepler_selftest=ok cubin_sha={sha} launch_words={len(words)} sections={eh['shnum']}")
  return sha

def run_software_demo(dev):
  """End-to-end data path on the software VRAM stand-in (no real Kepler SASS
  executes — the add is performed host-side to validate alloc/map/copy/CWD)."""
  import random
  N = 256
  cubin = build_cubin()
  prog = dev.runtime("E_4", cubin)
  allocator = NVAllocator(dev)

  a_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
  b_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])

  a_dev = allocator.alloc(N * 4)
  b_dev = allocator.alloc(N * 4)
  out_dev = allocator.alloc(N * 4)
  # Code + constant (param) buffers for the launch descriptor.
  code_dev = allocator.alloc(len(cubin))
  allocator._copyin(code_dev, cubin)
  cbuf = build_cuda_param_cbuf(a_dev.va_addr, b_dev.va_addr, out_dev.va_addr)
  cbuf_dev = allocator.alloc(len(cbuf))
  allocator._copyin(cbuf_dev, cbuf)

  allocator._copyin(a_dev, a_host.tobytes())
  allocator._copyin(b_dev, b_host.tobytes())
  # A GPU result must overwrite every lane.  Use a non-finite bit-pattern so
  # untouched/partially-written output cannot accidentally equal the reference.
  allocator._copyin(out_dev, struct.pack("<I", 0x7fc00001) * N)

  # Build + map a CWD, then emit the Kepler compute launch words.
  cwd = build_cwd(code_addr=0, grid=(1, 1, 1), block=(N, 1, 1), cbuf_addr=cbuf_dev.va_addr,
                  regs=cubin_register_count(cubin, "E_4"))
  cwd_dev = allocator.alloc(len(cwd))
  allocator._copyin(cwd_dev, cwd)
  words = build_launch_words(0x1000, 1, 2, cwd_dev.va_addr, code_dev.va_addr)
  decoded = list(decode_words(words))
  assert any(m == 0x02bc for _, _, _, m, _, _ in decoded), "launch words must include KEPLER_COMPUTE_LAUNCH"

  # Simulate the kernel: out = a + b, reading/writing the mapped VRAM buffers.
  a_mv = a_dev.cpu_view()
  b_mv = b_dev.cpu_view()
  out_mv = out_dev.cpu_view()
  for i in range(N):
    av = struct.unpack_from("<f", a_mv, i * 4)[0]
    bv = struct.unpack_from("<f", b_mv, i * 4)[0]
    struct.pack_into("<f", out_mv, i * 4, av + bv)

  out_host = bytearray(N * 4)
  allocator._copyout(out_host, out_dev)
  out_arr = array.array('f'); out_arr.frombytes(bytes(out_host))

  expected = [a_host[i] + b_host[i] for i in range(N)]
  assert all(abs(out_arr[i] - expected[i]) < 1e-5 for i in range(N)), "software add mismatch"
  print(f"software_demo=ok N={N} launch_words={len(words)} cwd_bytes={len(cwd)}")

# ----------------------------------------------------------------------------
# Live GK104 submit (plan milestones 5-12).  GK104 FIFO registers confirmed from
# nouveau (allbilly/linux_drm) + pascal-egpu:
#   RAMIN instance: pgd base @0x00/0x04, USERD base @0x08/0x0c,
#     GR ctx ptr @0x0210/0x0214, GP_PUT @0x48, GP_GET @0x4c.
#   USERD GP_GET=0x88, GP_PUT=0x8c (nvif/chan506f.c).
#   Channel bind: 0x800000 + id*8  = 0x80000000 | (inst_addr >> 12)
#   Channel start: 0x800004 + id*8 |= 0x400
#   PFIFO_RUNLIST_SUBMIT = 0x2274
# ----------------------------------------------------------------------------
CHAN_SUBMIT_REG = 0x800000
CHAN_START_REG  = 0x800004
PFIFO_RUNLIST_SUBMIT = 0x2274
# GR engine runlist ID, decoded from TOP scan (0x22700) exactly as Nouveau's
# gk104_top_oneinit():
#   TOP[00] ENUM 0x8006183e: engine=0, runlist=(data & 0x01e00000)>>21 = 0
#   TOP[01] ENGINE_TYPE 0x00000003: type=0 -> NVKM_ENGINE_GR
GR_RUNLIST_ID = 0
USERD_GP_GET = 0x88
USERD_GP_PUT = 0x8c

def gk104_mp_trap_addrs(gpc, tpc):
  base = 0x504000 + gpc * 0x8000 + tpc * 0x800
  return base + 0x648, base + 0x650

# ----------------------------------------------------------------------------
# Phase 1: Register-accurate trap snapshot (per the GK104 A0C0 debug plan).
# Captures every trap/status register at four points: before runlist, after
# GP_GET consumed, after first trap observed, and after W1C clear.
# Returns a structured dict that can be printed or emitted as JSON.
# ----------------------------------------------------------------------------
#
# Register map (from nouveau nvkm/engine/gr/gf100.c):
#   0x400100  PGRAPH_INTR          (bit0=NOTIFY, bit4=ILLEGAL_MTHD,
#                                   bit5=ILLEGAL_CLASS, bit20=DATA_ERROR,
#                                   bit21=TRAP, bit19=CTXCTL)
#   0x400108  PGRAPH_TRAP          (bit0=DISPATCH, bit1=M2MF, bit3=CCACHE,
#                                   bit4=SHADER, bit6=UNK6, bit7=MACRO,
#                                   bit8=SKED, bit24=GPC, bit25=ROP)
#   0x400110  PGRAPH_DATA_ERROR_CODE
#   0x400118  PGRAPH_GPC_TRAP      (bitmap of faulted GPCs)
#   0x400120  PGRAPH_ROP_TRAP      (bitmap of faulted ROPs)
#   0x400700  PGRAPH_STATUS        (engine idle/fifo status)
#   0x400704  PGRAPH_TRAPPED_ADDR  (subc[18:16], mthd[13:2])
#   0x400708  PGRAPH_TRAPPED_DATA
#   0x404000  DISPATCH trap status
#   0x404200+N GR_SUBCH_CLASS      (class bound to subchannel N)
#   0x405840  SHADER trap status
#   0x407020  SKED trap status
#   0x408030  CCACHE trap status
#   0x409b00  FECS CHAN_ADDR       (current channel instance addr)
#   0x409b04  FECS CHAN_NEXT       (next channel instance addr)
#
# Per-GPC (GPC_UNIT(gpc, r) = 0x500000 + gpc*0x8000 + r):
#   +0x2c90   GPC trap summary     (bit0=ROP, bit1=ZCULL, bit2=CCACHE,
#                                   bit3=ESETUP, bit16+N=TPC_N)
#   +0x0420   GPC/PROP trap status
#   +0x0434   GPC/PROP trap x
#   +0x0438   GPC/PROP trap y
#   +0x043c   GPC/PROP trap format/storage
#   +0x0900   GPC/ZCULL trap
#   +0x1028   GPC/CCACHE trap
#   +0x0824   GPC/ESETUP trap
#
# Per-TPC (TPC_UNIT(gpc, tpc, r) = 0x504000 + gpc*0x8000 + tpc*0x800 + r):
#   +0x0508   TPC trap summary     (bit0=TEX, bit1=MP, bit2=POLY,
#                                   bit3=L1C, bit4=MPC)
#   +0x0224   TPC/TEX trap
#   +0x0084   TPC/POLY trap
#   +0x048c   TPC/L1C trap
#   +0x0430   TPC/MPC trap
#   +0x0648   MP warp error
#   +0x0650   MP global error
#
# Per-ROP (ROP_UNIT(rop, r) = 0x410000 + rop*0x400 + r):
#   +0x070    ROP trap status z
#   +0x144    ROP trap status c

# Nouveau error decoding tables (for human-readable snapshot output).
_GK104_DISPATCH_ERRORS = {
  0x1: "INJECTED_BUNDLE_ERROR", 0x2: "CLASS_SUBCH_MISMATCH",
  0x4: "SUBCHSW_DURING_NOTIFY",
}
_GK104_CCACHE_ERRORS = {0x1: "INTR", 0x2: "LDCONST_OOB"}
_GK104_SKED_ERRORS = {
  0x40: "CTA_RESUME", 0x80: "CONSTANT_BUFFER_SIZE",
  0x200: "LOCAL_MEMORY_SIZE_POS", 0x400: "LOCAL_MEMORY_SIZE_NEG",
  0x800: "WARP_CSTACK_SIZE", 0x1000: "TOTAL_TEMP_SIZE",
  0x2000: "REGISTER_COUNT", 0x40000: "TOTAL_THREADS",
  0x100000: "PROGRAM_OFFSET", 0x200000: "SHARED_MEMORY_SIZE",
  0x800000: "CTA_THREAD_DIMENSION_ZERO", 0x1000000: "MEMORY_WINDOW_OVERLAP",
  0x2000000: "SHARED_CONFIG_TOO_SMALL", 0x4000000: "TOTAL_REGISTER_COUNT",
}
_GK104_MP_WARP_ERRORS = {
  0x01: "STACK_ERROR", 0x02: "API_STACK_ERROR", 0x03: "RET_EMPTY_STACK_ERROR",
  0x04: "PC_WRAP", 0x05: "MISALIGNED_PC", 0x06: "PC_OVERFLOW",
  0x07: "MISALIGNED_IMMC_ADDR", 0x08: "MISALIGNED_REG",
  0x09: "ILLEGAL_INSTR_ENCODING", 0x0a: "ILLEGAL_SPH_INSTR_COMBO",
  0x0b: "ILLEGAL_INSTR_PARAM", 0x0c: "INVALID_CONST_ADDR",
  0x0d: "OOR_REG", 0x0e: "OOR_ADDR", 0x0f: "MISALIGNED_ADDR",
  0x10: "INVALID_ADDR_SPACE", 0x11: "ILLEGAL_INSTR_PARAM2",
  0x12: "INVALID_CONST_ADDR_LDC", 0x13: "GEOMETRY_SM_ERROR",
  0x14: "DIVERGENT", 0x15: "WARP_EXIT",
}
_GK104_MP_GLOBAL_ERRORS = {
  0x1: "SM_TO_SM_FAULT", 0x2: "L1_ERROR", 0x4: "MULTIPLE_WARP_ERRORS",
  0x8: "PHYSICAL_STACK_OVERFLOW", 0x10: "BPT_INT", 0x20: "BPT_PAUSE",
  0x40: "SINGLE_STEP_COMPLETE", 0x20000000: "ECC_SEC_ERROR",
  0x40000000: "ECC_DED_ERROR", 0x80000000: "TIMEOUT",
}
_GK104_PGRAPH_INTR_BITS = {
  0x1: "NOTIFY", 0x10: "ILLEGAL_MTHD", 0x20: "ILLEGAL_CLASS",
  0x80000: "CTXCTL", 0x100000: "DATA_ERROR", 0x200000: "TRAP",
}
_GK104_PGRAPH_TRAP_BITS = {
  0x1: "DISPATCH", 0x2: "M2MF", 0x8: "CCACHE", 0x10: "SHADER",
  0x40: "UNK6", 0x80: "MACRO", 0x100: "SKED",
  0x1000000: "GPC", 0x2000000: "ROP",
}

def _decode_bits(value, table):
  return [name for bit, name in table.items() if value & bit]

def _decode_enum(value, table):
  return table.get(value, "")

def snapshot_fecs_gpccs(dev):
  """Capture FECS and GPCCS falcon state."""
  fecs = {
    "cpuctl": dev.read32(0x409100),
    "cpustat": dev.read32(0x409128),
    "intr": dev.read32(0x409008),
    "iren": dev.read32(0x409010),
    "scratch0": dev.read32(0x409800),
    "scratch1": dev.read32(0x409804),
    "scratch5_err": dev.read32(0x409814),
    "chan_addr": dev.read32(0x409b00),
    "chan_next": dev.read32(0x409b04),
    "fifo_cmd": dev.read32(0x409504),
    "fifo_data": dev.read32(0x409500),
    "mem_cmd": dev.read32(0x409a1c),
    "mem_target": dev.read32(0x409acc),
    "mem_base": dev.read32(0x409a20),
    "pc": dev.read32(0x409ff0),
  }
  gpccs = {
    "cpuctl": dev.read32(0x41a100),
    "cpustat": dev.read32(0x41a128),
    "scratch0": dev.read32(0x41a800),
    "pc": dev.read32(0x41aff0),
  }
  return {"fecs": fecs, "gpccs": gpccs}

def snapshot_gr_traps(dev, label, *, emit_json=False):
  """Capture a register-accurate PGRAPH/FECS/GPCCS/GPC/TPC trap snapshot.

  Returns a structured dict.  When emit_json is True, also prints a JSON
  blob tagged with ``label`` so the output can be parsed by tooling.
  """
  _ka = lambda: dev.read32(0x409100)  # FECS keep-alive (read to prevent clock-gating)
  _ka()
  gpc_nr = dev.read32(0x409604) & 0x1f
  tpc_nr = [dev.read32(0x500000 + g * 0x8000 + 0x2608) & 0x1f
            for g in range(gpc_nr)]
  rop_nr = 0  # GK104 ROP count — read from topology if available
  try:
    rop_nr = (dev.read32(0x409604) >> 16) & 0x1f
  except Exception:
    pass

  # Top-level PGRAPH registers.
  _ka()
  pgraph_intr = dev.read32(0x400100)
  pgraph_trap = dev.read32(0x400108)
  pgraph_data_error = dev.read32(0x400110)
  pgraph_gpc_trap = dev.read32(0x400118)
  pgraph_rop_trap = dev.read32(0x400120)
  pgraph_status = dev.read32(0x400700)
  pgraph_trapped_addr = dev.read32(0x400704)
  pgraph_trapped_data = dev.read32(0x400708)
  pgraph_status_main = dev.read32(0x400000)
  pgraph_ctrl = dev.read32(0x400500)
  fe_pwr = dev.read32(0x404170)

  trapped_subc = (pgraph_trapped_addr >> 16) & 0x7
  trapped_mthd = pgraph_trapped_addr & 0x3ffc

  # Sub-unit trap status registers.
  _ka()
  dispatch_stat = dev.read32(0x404000)
  ccache_stat = dev.read32(0x408030)
  shader_stat = dev.read32(0x405840)
  sked_stat = dev.read32(0x407020) & 0x3fffffff
  m2mf_stat = dev.read32(0x404600)
  unk6_stat = dev.read32(0x40601c)
  macro_stat = dev.read32(0x404490)
  macro_pc = dev.read32(0x404494)
  macro_op = dev.read32(0x40449c)

  # Bound classes for subchannels 0..7 (0x404200 + subc*4).
  subch_classes = {}
  for s in range(8):
    subch_classes[s] = dev.read32(0x404200 + s * 4)

  # HUB fault units (0x409c00 + unit*0x20, 16 units).
  hub_faults = []
  _ka()
  for u in range(16):
    stat = dev.read32(0x409c00 + u * 0x20)
    if stat & 0x80000000:
      hub_faults.append({
        "unit": u, "stat": stat,
        "addr": dev.read32(0x409c04 + u * 0x20),
      })

  # Per-GPC trap state.
  gpcs = []
  _ka()
  for g in range(gpc_nr):
    gb = 0x500000 + g * 0x8000
    gpc_summary = dev.read32(gb + 0x2c90)
    gpc_rop_stat = dev.read32(gb + 0x0420) & 0x3fffffff
    gpc_rop_x = dev.read32(gb + 0x0434)
    gpc_rop_y = dev.read32(gb + 0x0438)
    gpc_rop_fmt = dev.read32(gb + 0x043c)
    gpc_zcull = dev.read32(gb + 0x0900)
    gpc_ccache = dev.read32(gb + 0x1028)
    gpc_esetup = dev.read32(gb + 0x0824)
    tpcs = []
    _ka()
    for t in range(tpc_nr[g]):
      tb = 0x504000 + g * 0x8000 + t * 0x800
      tpc_summary = dev.read32(tb + 0x0508)
      tex = dev.read32(tb + 0x0224)
      poly = dev.read32(tb + 0x0084)
      l1c = dev.read32(tb + 0x048c)
      mpc = dev.read32(tb + 0x0430)
      mp_warp = dev.read32(tb + 0x0648)
      mp_global = dev.read32(tb + 0x0650)
      tpcs.append({
        "tpc": t,
        "summary": tpc_summary,
        "tex": tex, "poly": poly, "l1c": l1c, "mpc": mpc,
        "mp_warp": mp_warp, "mp_global": mp_global,
        "mp_warp_name": _decode_enum(mp_warp & 0xffff, _GK104_MP_WARP_ERRORS),
        "mp_global_names": _decode_bits(mp_global, _GK104_MP_GLOBAL_ERRORS),
      })
    gpcs.append({
      "gpc": g, "summary": gpc_summary,
      "rop_stat": gpc_rop_stat, "rop_x": gpc_rop_x, "rop_y": gpc_rop_y,
      "rop_fmt": gpc_rop_fmt,
      "zcull": gpc_zcull, "ccache": gpc_ccache, "esetup": gpc_esetup,
      "tpcs": tpcs,
    })

  # Per-ROP trap state.
  rops = []
  _ka()
  for r in range(max(rop_nr, 1)):
    rb = 0x410000 + r * 0x400
    rops.append({
      "rop": r,
      "stat_z": dev.read32(rb + 0x070),
      "stat_c": dev.read32(rb + 0x144),
    })

  fecs_gpccs = snapshot_fecs_gpccs(dev)

  snap = {
    "label": label,
    "pgraph": {
      "intr": pgraph_intr,
      "intr_names": _decode_bits(pgraph_intr, _GK104_PGRAPH_INTR_BITS),
      "trap": pgraph_trap,
      "trap_names": _decode_bits(pgraph_trap, _GK104_PGRAPH_TRAP_BITS),
      "data_error": pgraph_data_error,
      "gpc_trap": pgraph_gpc_trap,
      "rop_trap": pgraph_rop_trap,
      "status": pgraph_status,
      "status_main": pgraph_status_main,
      "ctrl": pgraph_ctrl,
      "fe_pwr": fe_pwr,
      "trapped_addr": pgraph_trapped_addr,
      "trapped_subc": trapped_subc,
      "trapped_mthd": trapped_mthd,
      "trapped_data": pgraph_trapped_data,
      "dispatch_stat": dispatch_stat,
      "dispatch_names": _decode_bits(dispatch_stat & 0x3fffffff,
                                     _GK104_DISPATCH_ERRORS),
      "ccache_stat": ccache_stat,
      "ccache_names": _decode_bits(ccache_stat & 0x3fffffff,
                                   _GK104_CCACHE_ERRORS),
      "shader_stat": shader_stat,
      "sked_stat": sked_stat,
      "sked_names": _decode_bits(sked_stat, _GK104_SKED_ERRORS),
      "m2mf_stat": m2mf_stat,
      "unk6_stat": unk6_stat,
      "macro_stat": macro_stat, "macro_pc": macro_pc, "macro_op": macro_op,
      "subch_classes": subch_classes,
    },
    "hub_faults": hub_faults,
    "gpcs": gpcs,
    "rops": rops,
    "fecs": fecs_gpccs["fecs"],
    "gpccs": fecs_gpccs["gpccs"],
    "topology": {"gpc_nr": gpc_nr, "tpc_nr": tpc_nr, "rop_nr": rop_nr},
  }

  # Human-readable summary (always printed).
  _print_trap_snapshot(snap)

  # Optional JSON output for tooling.
  if emit_json or os.environ.get("KEPLER_TRAP_JSON") == "1":
    print(f"[kepler-trap-json] {json.dumps(snap, default=str)}", flush=True)

  return snap

def _print_trap_snapshot(snap):
  """Print a concise human-readable summary of a trap snapshot."""
  p = snap["pgraph"]
  print(f"[kepler-trap] {snap['label']}: "
        f"INTR={p['intr']:#x}[{','.join(p['intr_names']) or '-'}] "
        f"TRAP={p['trap']:#x}[{','.join(p['trap_names']) or '-'}] "
        f"GPC={p['gpc_trap']:#x} ROP={p['rop_trap']:#x} "
        f"ADDR={p['trapped_addr']:#x}(subc={p['trapped_subc']},"
        f"mthd={p['trapped_mthd']:#x}) DATA={p['trapped_data']:#x} "
        f"STATUS={p['status_main']:#x} FE_PWR={p['fe_pwr']:#x}", flush=True)
  if p["trap"] & 0x1:
    print(f"[kepler-trap]   DISPATCH={p['dispatch_stat']:#x}"
          f"[{','.join(p['dispatch_names']) or '-'}]", flush=True)
  if p["trap"] & 0x8:
    print(f"[kepler-trap]   CCACHE={p['ccache_stat']:#x}"
          f"[{','.join(p['ccache_names']) or '-'}]", flush=True)
  if p["trap"] & 0x10:
    print(f"[kepler-trap]   SHADER={p['shader_stat']:#x} "
          f"sph={p['shader_stat'] & 0xffffff:#x} "
          f"stage={(p['shader_stat'] >> 24) & 0x3f:#x}", flush=True)
  if p["trap"] & 0x100:
    print(f"[kepler-trap]   SKED={p['sked_stat']:#x}"
          f"[{','.join(p['sked_names']) or '-'}]", flush=True)
  for s, cls in p["subch_classes"].items():
    if cls:
      print(f"[kepler-trap]   subch{s}={cls:#x}", flush=True)
  for hf in snap["hub_faults"]:
    print(f"[kepler-trap]   HUB fault unit={hf['unit']} "
          f"stat={hf['stat']:#x} addr={hf['addr']:#x}", flush=True)
  for g in snap["gpcs"]:
    if g["summary"] or any(t["summary"] for t in g["tpcs"]):
      print(f"[kepler-trap]   GPC{g['gpc']}: summary={g['summary']:#x} "
            f"zcull={g['zcull']:#x} ccache={g['ccache']:#x} "
            f"esetup={g['esetup']:#x}", flush=True)
      for t in g["tpcs"]:
        if t["summary"]:
          print(f"[kepler-trap]     TPC{t['tpc']}: summary={t['summary']:#x} "
                f"tex={t['tex']:#x} poly={t['poly']:#x} l1c={t['l1c']:#x} "
                f"mpc={t['mpc']:#x} mp_warp={t['mp_warp']:#x}"
                f"[{t['mp_warp_name'] or '-'}] "
                f"mp_global={t['mp_global']:#x}"
                f"[{','.join(t['mp_global_names']) or '-'}]", flush=True)
  f = snap["fecs"]
  g = snap["gpccs"]
  print(f"[kepler-trap]   FECS: CPUCTL={f['cpuctl']:#x} "
        f"PC={f['pc']:#x} SCRATCH0={f['scratch0']:#x} "
        f"SCRATCH1={f['scratch1']:#x} SCRATCH5={f['scratch5_err']:#x} "
        f"CHAN_ADDR={f['chan_addr']:#x} CHAN_NEXT={f['chan_next']:#x} "
        f"MEM_CMD={f['mem_cmd']:#x} MEM_TGT={f['mem_target']:#x} "
        f"MEM_BASE={f['mem_base']:#x}", flush=True)
  print(f"[kepler-trap]   GPCCS: CPUCTL={g['cpuctl']:#x} "
        f"PC={g['pc']:#x} SCRATCH0={g['scratch0']:#x}", flush=True)

# GK104 GPFIFO entries are 8 bytes: a push-buffer address followed by a
# length/flags word. GP_PUT/GP_GET are ring-entry indices, not byte addresses.
GPFIFO_ENTRY_BYTES = 8

def _gk104_pgd_entry(dev, table_pa):
  """Build the VMM join value for a system-memory page directory."""
  mm = dev.dev_impl.mm
  return (mm.bus_base + table_pa) | 0x6  # HOST target + VOL

def _gk104_gr_wait_idle(dev, timeout_s=2.0):
  """Wait for GR to go idle (gf100_gr_wait_idle)."""
  deadline = time.time() + timeout_s
  while time.time() < deadline:
    dev.read32(0x400700)  # required to update FIFO_ENGINE_STATUS
    gr_enabled = dev.read32(0x200) & 0x1000
    gr_busy = dev.read32(0x40060c) & 0x1
    if not gr_enabled or not gr_busy:
      return True
    time.sleep(0.001)
  return False

def _gk104_gr_fecs_reset(dev):
  """Release the FECS MAIN/GPC/ROP domains as gf100_gr_fecs_reset().

  This reset belongs to golden-context generation, after ctxctl firmware has
  booted.  Doing it during the initial PGRAPH MMIO pack can hide IMEM/DMEM;
  omitting it here leaves the method engine unable to load the saved context.
  """
  dev.write32(0x409614, 0x00000070)
  time.sleep(0.00001)
  nvkm_mask(dev, 0x409614, 0x00000700, 0x00000700)
  time.sleep(0.00001)
  dev.read32(0x409614)

def _gk104_gr_mmio(dev, writes):
  """Write a list of (addr, value) pairs via direct MMIO (gf100_gr_mmio)."""
  for addr, data in writes:
    dev.write32(addr, data & 0xffffffff)

def _gk104_gr_icmd(dev, writes):
  """Write icmd entries via the 0x400200/0x400204 interface (gf100_gr_icmd).
  writes is a list of (addr, data) pairs.  Optimizes by only writing data
  when it changes."""
  dev.write32(0x400208, 0x80000000)  # enable icmd mode
  prev_data = None
  for addr, data in writes:
    data = data & 0xffffffff
    if data != prev_data:
      dev.write32(0x400204, data)
      prev_data = data
    dev.write32(0x400200, addr)
    # Wait for busy bit (0x400700 bit2) to clear
    deadline = time.time() + 0.2
    while dev.read32(0x400700) & 0x4:
      if time.time() >= deadline:
        break
      time.sleep(0.001)
  dev.write32(0x400208, 0x00000000)  # disable icmd mode

def _gk104_gr_mthd(dev, entries):
  """Write mthd entries via the 0x404488/0x40448c interface (gf100_gr_mthd).
  entries is a list of (type, addr, data) tuples."""
  prev_data = None
  for ptype, addr, data in entries:
    data = data & 0xffffffff
    if data != prev_data:
      dev.write32(0x40448c, data)
      prev_data = data
    ctrl = 0x80000000 | ptype
    dev.write32(0x404488, ctrl | (addr << 14))

def _gk104_grctx_tiles(tpc_nr):
  """Return (screen-tile-row-offset, tile map) from gf100_gr_oneinit_tiles."""
  gpc_nr, tpc_total, tpc_max = len(tpc_nr), sum(tpc_nr), max(tpc_nr)
  fixed = {15: 6, 14: 5, 13: 2, 11: 7, 10: 6, 7: 1, 5: 1,
           3: 2, 2: 1, 1: 1}
  row = fixed.get(tpc_total)
  if row is None:
    row = next((p for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                            37, 41, 43, 47, 53, 59, 61) if tpc_total % p), 3)
  gpc_map = sorted(range(gpc_nr), key=lambda x: -tpc_nr[x])
  mul = 2 if (gpc_nr * tpc_max) & 1 else 1
  denom = gpc_nr * tpc_max * mul
  frac = [tpc_nr[g] * gpc_nr * mul for g in gpc_map]
  err = [i * tpc_max * mul - denom // 2 for i in range(gpc_nr)]
  run = [frac[i] + err[i] for i in range(gpc_nr)]
  tile = []
  while len(tile) < tpc_total:
    for j in range(gpc_nr):
      if run[j] * 2 >= denom and len(tile) < tpc_total:
        tile.append(gpc_map[j])
        run[j] += frac[j] - denom
      else:
        run[j] += frac[j]
  return row, tile + [0xff] * (32 - len(tile))

def _gk104_grctx_floorsweep(dev, tpc_nr, ppc_tpc_mask):
  """Port GK104's topology-dependent gf100_grctx_generate_floorsweep()."""
  gpc_nr, tpc_total = len(tpc_nr), sum(tpc_nr)
  row, tile = _gk104_grctx_tiles(tpc_nr)

  # gf100_gr_oneinit_sm_id orders SMs by TPC index, then GPC index.
  sm = 0
  for tpc in range(max(tpc_nr)):
    for gpc in range(gpc_nr):
      if tpc >= tpc_nr[gpc]: continue
      for reg in (0x504698, 0x5044e8, 0x504088):
        dev.write32(reg + gpc * 0x8000 + tpc * 0x800, sm)
      dev.write32(0x500c10 + gpc * 0x8000 + tpc * 4, sm)
      dev.write32(0x500c08 + gpc * 0x8000, tpc_nr[gpc])
      dev.write32(0x500c8c + gpc * 0x8000, tpc_nr[gpc])
      sm += 1

  # DS and PD NUM_TPC_PER_GPC tables.
  for block in range(4):
    data = 0
    for j in range(8):
      gpc = block * 8 + j
      if gpc < gpc_nr: data |= tpc_nr[gpc] << (j * 4)
    dev.write32(0x405870 + block * 4, data)
    dev.write32(0x406028 + block * 4, data)

  # gf117 ROP mapping with GK104 tile distribution.
  packed = [0] * 6
  for i in range(32): packed[i // 6] |= (tile[i] & 7) << ((i % 6) * 5)
  shift, ntpcv = 0, tpc_total
  while not (ntpcv & 16): ntpcv, shift = ntpcv << 1, shift + 1
  magic0 = (ntpcv << 16) | (shift << 21) | (((1 << 5) % ntpcv) << 24)
  magic1 = sum(((1 << (i + 5)) % ntpcv) << ((i - 1) * 5) for i in range(1, 7))
  for base in (0x418b08, 0x41bf00, 0x40780c):
    for i, data in enumerate(packed): dev.write32(base + i * 4, data)
  dev.write32(0x418bb8, (tpc_total << 8) | row)
  dev.write32(0x41bfd0, (tpc_total << 8) | row | magic0)
  dev.write32(0x41bfe4, magic1)
  dev.write32(0x4078bc, (tpc_total << 8) | row)

  # GK104 alpha/beta distribution.  GK104 has one PPC per GPC.
  ppc_counts = [m.bit_count() for m in ppc_tpc_mask]
  for i in range(32):
    atarget, btarget = max(tpc_total * i // 32, 1), 0
    btarget = tpc_total - atarget
    alpha, amask, bmask = atarget < btarget, 0, 0
    for gpc, count in enumerate(ppc_counts):
      abits = (count if atarget else 0) if alpha else count - (count if btarget else 0)
      bbits = count - abits
      pmask = ppc_tpc_mask[gpc]
      trim = count
      while trim > abits: pmask, trim = pmask & (pmask - 1), trim - 1
      amask |= pmask << (gpc * 8)
      bmask |= (pmask ^ ppc_tpc_mask[gpc]) << (gpc * 8)
      atarget -= min(abits, atarget); btarget -= min(bbits, btarget)
      if abits or bbits: alpha = not alpha
    dev.write32(0x406800 + i * 0x20, amask & 0xffffffff)
    dev.write32(0x406c00 + i * 0x20, bmask & 0xffffffff)

  for i in range(8): dev.write32(0x4064d0 + i * 4, 0)
  dev.write32(0x405b00, (tpc_total << 8) | gpc_nr)
  nvkm_mask(dev, 0x419f78, 0x00000009, 0)
  return row, tile[:tpc_total]

def _gk104_grctx_main(dev, pagepool_pa, bundle_pa, attrib_cb_pa, tpc_nr):
  """Replicate gf100_grctx_generate_main() for GK104.
  Writes GR register init lists, pagepool/bundle/attrib_cb addresses,
  icmd and mthd bundles, and various grctx fixups."""
  from grctx_gk104 import (mmio_pack, icmd_pack, mthd_pack,
                            MMIO_PACKS, ICMD_PACKS, MTHD_PACKS,
                            GK104_GRCTX_CONSTS)
  C = GK104_GRCTX_CONSTS
  gpc_nr, tpc_total = len(tpc_nr), sum(tpc_nr)
  ppc_tpc_mask = [dev.read32(0x500c30 + gpc * 0x8000) & 0xff
                  for gpc in range(gpc_nr)]
  # A zero mask is not a valid active PPC; topology registers occasionally
  # read late on this cold-attached card, so fall back to the known TPC mask.
  ppc_tpc_mask = [m or ((1 << tpc_nr[gpc]) - 1)
                  for gpc, m in enumerate(ppc_tpc_mask)]
  dev.write32(0x000260, 0)
  # Diagnostic: check if GR registers are accessible before writing.
  _test_read = dev.read32(0x404154)
  _test_intr = dev.read32(0x400100)
  print(f"[kepler] grctx_main pre-check: 0x404154={_test_read:#x} "
        f"0x400100={_test_intr:#x} 0x405b00={dev.read32(0x405b00):#x} "
        f"0x40060c={dev.read32(0x40060c):#x}", flush=True)
  # Clear pending PGRAPH interrupts before writing mmio packs.
  # The ctxctl interrupt (0x80000) from FECS ctx_chan may keep GR busy.
  if _test_intr:
    dev.write32(0x400100, _test_intr)
  # Clear FECS exceptions
  _fecs_exc_pre = dev.read32(0x409c18)
  if _fecs_exc_pre:
    dev.write32(0x409c20, _fecs_exc_pre)
  # 1. Write mmio packs (hub, gpc_0, zcull, gpc_1, tpc, ppc)
  for pack_name in MMIO_PACKS:
    _gk104_gr_mmio(dev, mmio_pack(pack_name))
  _idle1 = _gk104_gr_wait_idle(dev)
  print(f"[kepler] grctx after mmio_packs: 0x40060c={dev.read32(0x40060c):#x} idle={_idle1} ctrl={dev.read32(0x400500):#x}", flush=True)
  # 2. Save and zero idle timeout (0x404154)
  idle_timeout = dev.read32(0x404154)
  dev.write32(0x404154, 0)
  # 3. Pagepool address (gf100_grctx_generate_pagepool + gk104 override)
  dev.write32(0x40800c, pagepool_pa >> 8)
  dev.write32(0x408010, 0x80000000)
  dev.write32(0x419004, pagepool_pa >> 8)
  dev.write32(0x419008, 0x00000000)
  dev.write32(0x4064cc, 0x80000000)  # GK104-specific
  # 4. Bundle address (gf100_grctx_generate_bundle + gk104 override)
  bundle_size = C["bundle_size"]
  dev.write32(0x408004, bundle_pa >> 8)
  dev.write32(0x408008, 0x80000000 | (bundle_size >> 8))
  dev.write32(0x418808, bundle_pa >> 8)
  dev.write32(0x41880c, 0x80000000 | (bundle_size >> 8))
  state_limit = min(C["bundle_min_gpm_fifo_depth"], bundle_size // 0x20)
  token_limit = C["bundle_token_limit"]
  dev.write32(0x4064c8, (state_limit << 16) | token_limit)
  # 5. Attrib CB address (gf100_grctx_generate_attrib_cb)
  dev.write32(0x418810, 0x80000000 | (attrib_cb_pa >> 12))
  dev.write32(0x419848, 0x10000000 | (attrib_cb_pa >> 12))
  print(f"[kepler] grctx buffer regs: pagepool={dev.read32(0x40800c):#x}/"
        f"{dev.read32(0x419004):#x} expected={pagepool_pa >> 8:#x} "
        f"bundle={dev.read32(0x408004):#x}/{dev.read32(0x418808):#x} "
        f"expected={bundle_pa >> 8:#x}", flush=True)
  # 6. Attrib configuration (gf117_grctx_generate_attrib)
  alpha = C["alpha_nr"]
  beta = C["attrib_nr"]
  dev.write32(0x405830, (beta << 16) | alpha)
  dev.write32(0x4064c4, ((alpha // 4) << 16) | 0xffff)
  bo, ao = 0, C["attrib_nr_max"] * tpc_total
  for gpc, mask in enumerate(ppc_tpc_mask):
    count = mask.bit_count()
    ppc = 0x503000 + gpc * 0x8000
    dev.write32(ppc + 0xc0, (1 << 28) | (beta * count << 16) | bo)
    bo += C["attrib_nr_max"] * count
    dev.write32(ppc + 0xe4, (alpha * count << 16) | ao)
    ao += C["alpha_nr_max"] * count
  # Preserve the LTC context registers in the generated image.
  dev.write32(0x17e91c, dev.read32(0x17e91c))
  dev.write32(0x17e920, dev.read32(0x17e920))
  # 7. unkn (gk104_grctx_generate_unkn)
  nvkm_mask(dev, 0x418c6c, 0x00000001, 0x00000001)
  nvkm_mask(dev, 0x41980c, 0x00000010, 0x00000010)
  nvkm_mask(dev, 0x41be08, 0x00000004, 0x00000004)
  nvkm_mask(dev, 0x4064c0, 0x80000000, 0x80000000)
  nvkm_mask(dev, 0x405800, 0x08000000, 0x08000000)
  nvkm_mask(dev, 0x419c00, 0x00000008, 0x00000008)
  # 8. Topology-dependent floorsweep state.
  row, tile = _gk104_grctx_floorsweep(dev, tpc_nr, ppc_tpc_mask)
  _gpc_tpc_readback = dev.read32(0x405b00)
  print(f"[kepler] grctx_main GPC/TPC: wrote {(tpc_total << 8) | gpc_nr:#x} "
        f"readback {_gpc_tpc_readback:#x}", flush=True)
  # 9. r419f78 (gk104_grctx_generate_r419f78)
  nvkm_mask(dev, 0x419f78, 0x00000009, 0x00000000)
  _gk104_gr_wait_idle(dev)
  print(f"[kepler] grctx after step9: 0x40060c={dev.read32(0x40060c):#x} ctrl={dev.read32(0x400500):#x}", flush=True)
  # 10. icmd bundles
  for pack_name in ICMD_PACKS:
    _gk104_gr_icmd(dev, icmd_pack(pack_name))
  print(f"[kepler] grctx after icmd: 0x40060c={dev.read32(0x40060c):#x} ctrl={dev.read32(0x400500):#x}", flush=True)
  # 11. Restore idle timeout
  dev.write32(0x404154, idle_timeout)
  # 12. mthd bundles
  for pack_name in MTHD_PACKS:
    _gk104_gr_mthd(dev, mthd_pack(pack_name))
  dev.write32(0x000260, 1)
  print(f"[kepler] grctx after mthd: 0x40060c={dev.read32(0x40060c):#x} ctrl={dev.read32(0x400500):#x}", flush=True)
  # 13. r419cb8 (gf100_grctx_generate_r419cb8)
  nvkm_mask(dev, 0x419cb8, 0x00007c00, 0x00000000)
  _idle_ok = _gk104_gr_wait_idle(dev)
  print(f"[kepler] grctx after step13: 0x40060c={dev.read32(0x40060c):#x} idle={_idle_ok} ctrl={dev.read32(0x400500):#x}", flush=True)
  print(f"[kepler] grctx topology: tpc_nr={tpc_nr} ppc_masks="
        f"{[hex(x) for x in ppc_tpc_mask]} row={row} tile={tile}", flush=True)

def _gk104_vmm_flush(dev):
  """Flush the GF100/GK104 page tables after populating new mappings."""
  mm = dev.dev_impl.mm
  pdb = mm.bus_base + mm.root_pa
  # gf100_vmm_invalidate(): HOST PDB target (2), PDB address in bits 4:.
  dev.write32(0x100cb8, 2 | ((pdb >> 12) << 4))
  dev.write32(0x100cbc, 0x80000001)  # PAGE_ALL
  deadline = time.time() + 0.2
  while time.time() < deadline:
    if dev.read32(0x100c80) & 0x00008000:
      return True
    time.sleep(0.001)
  return False

def _gk104_pramin_write(dev, pa, data):
  """Store framebuffer words through BAR0 PRAMIN and verify each result.

  On the un-POSTed card, writes through 0x700000 sometimes behave as
  XOR updates against the old framebuffer word (an uncleared 0xffffffff plus
  value V becomes ~V).  Use current^wanted first, which is the correct delta
  for that path, then fall back to a literal store if the aperture is behaving
  normally.  Never return with silently inverted channel data.
  """
  data = memoryview(data).cast("B")
  if len(data) & 3:
    raise ValueError("PRAMIN write must be 4-byte aligned")
  window = None
  for off in range(0, len(data), 4):
    addr = pa + off
    base = addr & 0xffffff00000
    if base != window:
      dev.write32(0x001700, base >> 16)
      window = base
    reg = 0x700000 + (addr & 0xfffff)
    wanted = struct.unpack_from("<I", data, off)[0]
    current = dev.read32(reg)
    if current == wanted:
      continue
    dev.write32(reg, current ^ wanted)
    actual = dev.read32(reg)
    if actual != wanted:
      dev.write32(reg, wanted)
      actual = dev.read32(reg)
    if actual != wanted:
      raise RuntimeError(f"PRAMIN store failed at {addr:#x}: "
                         f"wanted={wanted:#x} actual={actual:#x}")

def _gk104_pramin_read32(dev, pa):
  base = pa & 0xffffff00000
  dev.write32(0x001700, base >> 16)
  return dev.read32(0x700000 + (pa & 0xfffff))

def _gk104_pramin_write_literal(dev, pa, data):
  """Write raw dwords without the readback/XOR compensation."""
  data = memoryview(data).cast("B")
  if len(data) & 3:
    raise ValueError("literal PRAMIN write must be 4-byte aligned")
  for off in range(0, len(data), 4):
    addr = pa + off
    dev.write32(0x001700, (addr & 0xffffff00000) >> 16)
    dev.write32(0x700000 + (addr & 0xfffff),
                struct.unpack_from("<I", data, off)[0])

def _gk104_bar_flush(dev):
  """Flush pending framebuffer/BAR writes (g84_bar_flush)."""
  for _ in range(2):  # gf100_bar_bar1_wait() deliberately flushes twice.
    dev.write32(0x070000, 0x00000001)
    deadline = time.time() + 0.2
    while dev.read32(0x070000) & 0x00000002:
      if time.time() >= deadline:
        return False
      time.sleep(0.001)
  return True

def _gk104_ltc_init(dev):
  """Initialize GK104 LTC (L2 cache) — gk104_ltc_init() + gf100_ltc_oneinit().

  Without this, the L2 cache is not properly configured and GPU internal
  clients (PBDMA, GR, etc.) may read stale/zero data from VRAM.  This is
  a critical missing init step on the un-POSTed card.

  gf100_ltc_oneinit() reads the LTC topology (ltc_nr, lts_nr) from hardware
  and calls gf100_ltc_oneinit_tag_ram() to allocate compression tag RAM.
  On this card there is no VBIOS-initialized VRAM, so num_tags=0 and
  tag_base=0 (matching the no-ram path in gf100_ltc_oneinit_tag_ram()).
  gk104_ltc_init() then programs the LTC registers.
  """
  # gf100_ltc_oneinit(): read LTC topology from hardware.
  parts = dev.read32(0x022438)
  mask = dev.read32(0x022554)
  lts_nr = dev.read32(0x17e8dc) >> 28
  ltc_nr = 0
  for i in range(parts):
    if not (mask & (1 << i)):
      ltc_nr += 1
  # gf100_ltc_oneinit_tag_ram(): no VRAM → num_tags=0, tag_base=0.
  tag_base = 0
  # gk104_ltc_init(): program LTC registers.
  lpg128 = not (dev.read32(0x100c80) & 0x00000001)
  dev.write32(0x17e8d8, ltc_nr)
  dev.write32(0x17e000, ltc_nr)
  dev.write32(0x17e8d4, tag_base)
  nvkm_mask(dev, 0x17e8c0, 0x00000002, 0x00000002 if lpg128 else 0x00000000)
  if DEBUG:
    print(f"[kepler] LTC init: ltc_nr={ltc_nr} lts_nr={lts_nr} parts={parts} "
          f"mask={mask:#x} lpg128={lpg128} tag_base={tag_base}", flush=True)

def _gk104_fb_init_page(dev):
  """Set GF100/GK104 big-page mode — gf100_fb_init_page().

  GK104 default is 17-bit (128 KiB) big pages: 0x100c80 bit0 = 0.
  """
  nvkm_mask(dev, 0x100c80, 0x00000001, 0x00000000)

def _gk104_init_bar1_identity(dev, mapped_size=0x08000000, bus_base=0,
                              map_vram=True, userd_alias_pa=None):
  """Bootstrap an identity BAR1 mapping for GK104 VRAM.

  Maps the low ``mapped_size`` bytes of the sysmem-backed "VRAM" 1:1 through
  the BAR1 VMM, matching what gf100_bar_oneinit()+gf100_bar_bar1_init()
  create on a properly POSTed card.  The default 128 MiB covers the full
  BAR1 aperture on the GTX 770, including all channel VMM page tables
  and FIFO buffers allocated from the 0x400000+ heap.

  By default this maps BAR1 virtual addresses to the same VRAM offsets.  That
  is the mapping PFIFO's GK104 USERD polling path requires.  ``map_vram=False``
  retains the older diagnostic mapping to GPU-visible host sysmem.

      pte = (pa >> 8) | BIT(0) | BIT(32) | (2ULL << 33)

  The page directory and instance block live in PRAMIN (VRAM aperture 0),
  matching gf100_vmm_pgd_pde() for NVKM_MEM_TARGET_VRAM.
  """
  dev.write32(0x001704, dev.read32(0x001704) & ~0x80000000)
  inst_pa = 0x00100000
  pgd_pa = 0x00110000
  spt_pa = 0x00120000
  pages = mapped_size // 0x1000
  spt_bytes = pages * 8

  # Instance block: PDB points to the PRAMIN-resident page directory.
  # gf100_vmm_join_() for VRAM: base |= (0 << 0) | pd->addr.
  inst = bytearray(0x220)
  struct.pack_into("<Q", inst, 0x200, pgd_pa)
  struct.pack_into("<Q", inst, 0x208, mapped_size - 1)
  _gk104_pramin_write(dev, inst_pa, inst)
  # PDE: 4-KiB SPT in PRAMIN (VRAM target).  In gf100_vmm_pgd_pde() the SPT
  # is pt[1] (desc->type == SPT → type=1), encoded in the high 32 bits:
  #   data |= 1ULL << 32; data |= pt->addr << 24;
  # (The low 32 bits are for pt[0] = LPT, which we do not use.)
  _gk104_pramin_write(dev, pgd_pa,
                      struct.pack("<Q", (1 << 32) | (spt_pa << 24)))

  # PTEs: map each 4 KiB BAR1 VA to the corresponding VRAM page, or to host
  # sysmem for the legacy diagnostic mode.  gf100_vmm_valid() encodes VRAM as
  # aperture 0 without VOL, and HOST as aperture 2 with VOL.
  #   data = (addr >> 8) | map->type
  #   map->type = BIT(0) | (vol << 32) | (aper << 33)
  #   vol=1, aper=2 for NVKM_MEM_TARGET_HOST.
  pte_data = bytearray(spt_bytes)
  for page in range(pages):
    if map_vram:
      # Nouveau allocates a BAR1 VMA for the global USERD object.  Alias its
      # first pages at VA 0 while retaining identity mapping elsewhere.
      pa = ((userd_alias_pa + page * 0x1000)
            if userd_alias_pa is not None and page < 2 else page * 0x1000)
      pte = (pa >> 8) | 0x1
    else:
      pa = bus_base + page * 0x1000
      pte = (pa >> 8) | 0x1 | (0x1 << 32) | (0x2 << 33)
    struct.pack_into("<Q", pte_data, page * 8, pte)
  # Write PTEs in 4 KiB pages to stay within the PRAMIN window.
  for off in range(0, spt_bytes, 0x1000):
    end = min(off + 0x1000, spt_bytes)
    _gk104_pramin_write(dev, spt_pa + off, pte_data[off:end])
  _gk104_bar_flush(dev)
  if not _gk104_ltc_invalidate(dev):
    raise TimeoutError("GK104 LTC flush before BAR1 enable did not complete")
  # The framebuffer stores can read back correctly immediately and then
  # settle to their XOR-inverted value a few milliseconds later.  Stabilise
  # the translation root and the two USERD alias leaves before HUB clients
  # are allowed to use them.
  critical = [
      (inst_pa + 0x200, pgd_pa),
      (pgd_pa, (1 << 32) | (spt_pa << 24)),
  ]
  for page in range(2):
    pa = ((userd_alias_pa + page * 0x1000)
          if userd_alias_pa is not None else page * 0x1000)
    critical.append((spt_pa + page * 8, (pa >> 8) | 0x1))
    if userd_alias_pa is not None:
      userd_page = userd_alias_pa // 0x1000 + page
      critical.append((spt_pa + userd_page * 8, (pa >> 8) | 0x1))
  for attempt in range(4):
    for addr, wanted in critical:
      _gk104_pramin_write(dev, addr, struct.pack("<Q", wanted))
    _gk104_bar_flush(dev)
    if not _gk104_ltc_invalidate(dev):
      raise TimeoutError("GK104 LTC invalidate for BAR1 roots did not complete")
    time.sleep(0.005)
    actual = [(_gk104_pramin_read32(dev, addr) |
               (_gk104_pramin_read32(dev, addr + 4) << 32))
              for addr, _ in critical]
    if actual == [wanted for _, wanted in critical]:
      break
  else:
    raise RuntimeError(f"GK104 BAR1 roots did not stabilise: {actual}")
  _gk104_vmm_flush_pdb(dev, pgd_pa, target=0, hub_only=True)
  if DEBUG:
    print(f"[kepler] BAR1 bootstrap PRAMIN pdb="
          f"{_gk104_pramin_read32(dev, inst_pa + 0x200):#x}/"
          f"{_gk104_pramin_read32(dev, inst_pa + 0x204):#x} "
          f"pde0={_gk104_pramin_read32(dev, pgd_pa):#x}/"
          f"{_gk104_pramin_read32(dev, pgd_pa + 4):#x} "
          f"pte100={_gk104_pramin_read32(dev, spt_pa + 0x100 * 8):#x}/"
          f"{_gk104_pramin_read32(dev, spt_pa + 0x100 * 8 + 4):#x}", flush=True)
  dev.write32(0x001704, 0x80000000 | (inst_pa >> 12))
  _gk104_bar_flush(dev)
  if DEBUG:
    print(f"[kepler] BAR1 identity enabled inst={inst_pa:#x} pgd={pgd_pa:#x} "
          f"spt={spt_pa:#x} size={mapped_size:#x} "
          f"target={'VRAM' if map_vram else 'HOST'} "
          f"userd_alias={userd_alias_pa if userd_alias_pa is not None else 0:#x} "
          f"bus_base={bus_base:#x}",
          flush=True)

def _gk104_ltc_invalidate(dev):
  """Invalidate GK104's L2 after CPU BAR1/PRAMIN framebuffer stores."""
  # gf100_ltc_flush(): commit BAR/PRAMIN writes held by LTC first.
  dev.write32(0x070010, 0x00000001)
  deadline = time.time() + 0.2
  while dev.read32(0x070010) & 0x00000003:
    if time.time() >= deadline:
      return False
    time.sleep(0.001)
  dev.write32(0x070004, 0x00000001)
  deadline = time.time() + 0.2
  while dev.read32(0x070004) & 0x00000003:
    if time.time() >= deadline:
      return False
    time.sleep(0.001)
  return True

def _gk104_clone_vmm_to_vram(dev, bar1_alloc, bar1_write):
  """Clone the current GK104 4-KiB page tables into framebuffer memory.

  FECS VM DMA uses the channel VMM while constructing/saving a context.  On
  this machine it cannot reliably fetch the page-directory hierarchy from the
  host allocation used by ``GK104MemoryManager``.  Keep the already-validated
  leaf PTEs (including their SYS/VRAM aperture bits), but place the PGD and one
  distinct SPT per populated PGD entry in BAR1-visible VRAM.

  This deliberately mirrors gf100_vmm_pgd_pde() and gf100_vmm_join_(): a VRAM
  SPT PDE is ``1 | (spt_addr >> 8)``, while the instance's PDB value is the
  *full* VRAM PGD address (target bits are zero).  The ``addr>>12<<4`` form is
  only for 0x100cb8 TLB invalidation and is not valid in instance memory.
  """
  mm, backing = dev.dev_impl.mm, dev.dev_impl.vram
  pgd_size, spt_size = 0x10000, 0x40000
  pgd_pa = bar1_alloc(pgd_size, align=0x1000)
  pgd_image = bytearray(pgd_size)
  populated = []
  for pgd_idx in range(1 << 13):
    src_pde = mm.root_page_table.entry(pgd_idx)
    if not ((src_pde >> 32) & 0x3):
      continue
    src_spt = mm.root_page_table.address(pgd_idx)
    if src_spt < 0 or src_spt + spt_size > len(backing):
      raise RuntimeError(f"invalid host SPT {src_spt:#x} for PGD[{pgd_idx}]")
    dst_spt = bar1_alloc(spt_size, align=0x1000)
    spt_image = bytes(backing[src_spt:src_spt + spt_size])
    bar1_write(dst_spt, spt_image)
    # BAR1 bulk writes can acknowledge a page transfer while leaving portions
    # of framebuffer memory unchanged.  Re-issue every live
    # PTE as the native 8-byte transaction used by Nouveau's VMM writer.  We do
    # not issue 32768 separate clears; only populated entries can be reached by
    # the VAs this process submits.
    for ptei in range(spt_size // 8):
      pte = struct.unpack_from("<Q", spt_image, ptei * 8)[0]
      if pte:
        _gk104_pramin_write(dev, dst_spt + ptei * 8, struct.pack("<Q", pte))
    struct.pack_into("<Q", pgd_image, pgd_idx * 8,
                     (1 << 32) | (dst_spt << 24))
    populated.append((pgd_idx, src_spt, dst_spt))
  bar1_write(pgd_pa, pgd_image)
  for pgd_idx, _src_spt, dst_spt in populated:
    pde = (1 << 32) | (dst_spt << 24)
    _gk104_pramin_write(dev, pgd_pa + pgd_idx * 8, struct.pack("<Q", pde))
  _gk104_bar_flush(dev)
  return pgd_pa, populated

def _gk104_vmm_flush_pdb(dev, pdb, target=0, hub_only=False):
  """Invalidate translations for one GF100/GK104 page directory.

  ``target`` is the PDB aperture (0=VRAM, 2=HOST, 3=NCOH) written to 0x100cb8.
  ``hub_only`` adds HUB_ONLY (0x4) to the flush type, matching
  gf100_vmm_flush() when engref[NVKM_SUBDEV_BAR] > 0 (i.e. BAR1's VMM).
  """
  deadline = time.time() + 0.2
  while time.time() < deadline:
    if dev.read32(0x100c80) & 0x00ff0000:
      break
    time.sleep(0.001)
  dev.write32(0x100cb8, target | ((pdb >> 12) << 4))
  flush_type = 0x80000001  # PAGE_ALL
  if hub_only:
    flush_type |= 0x00000004  # HUB_ONLY
  dev.write32(0x100cbc, flush_type)
  deadline = time.time() + 0.2
  while time.time() < deadline:
    if dev.read32(0x100c80) & 0x00008000:
      return True
    time.sleep(0.001)
  return False

def _reload_fecs_after_pgob(dev):
  """Reload FECS firmware after pgob has reset the GR engine.
  FECS power-gates when idle (hardware auto power-gate that can't be disabled
  without a PMU).  The pgob un-gates the GR engine but destroys FECS (PC=0).
  This function reloads FECS IMEM + DMEM + csdata and restarts it, so it can
  process channel context switches.  The golden context is already in memory
  and referenced by the channel's instance pointer, so no golden save is needed.
  Returns True if FECS became ready, False otherwise."""
  fecs_code = dev._fecs_code
  fecs_data = dev._fecs_data
  fecs_packs = dev._fecs_packs
  # Disable ctxctl clock-gate for FECS DMEM/IMEM access.
  dev.write32(0x000260, 0x00000000)
  # Set FE_PWR FORCE_ON before falcon work.
  dev.write32(0x404170, 0x00000012)
  for _ in range(2000):
    if not (dev.read32(0x404170) & 0x00000010): break
    time.sleep(0.001)
  # Load FECS IMEM.
  falcon_write_imem(dev, FECS_FALCON_BASE, fecs_code)
  dev.write32(FECS_FALCON_BASE + FALCON_UC_ENTRY, 0)
  # Load FECS DMEM + csdata.
  fecs_data_words = struct.unpack_from(f"<{len(fecs_data)//4}I", fecs_data)
  dev.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, FALCON_IDX_WRITE)
  for w in fecs_data_words:
    dev.write32(FECS_FALCON_BASE + FALCON_DATA, w)
  fecs_star = 0x304
  for starstar, words in fecs_packs:
    dev.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, FALCON_IDX_WRITE | fecs_star)
    for w in words:
      dev.write32(FECS_FALCON_BASE + FALCON_DATA, w)
      fecs_star += 4
    dev.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, 0x01000000 | (starstar + 4))
    dev.write32(FECS_FALCON_BASE + FALCON_DATA, fecs_star)
  # Re-enable ctxctl clock-gate, then start FECS.
  dev.write32(0x000260, 0x00000001)
  # Keep FE_PWR at FORCE_ON (not AUTO).
  dev.write32(0x404170, 0x00000012)
  # Disable FECS idle filter to prevent auto clock-gating.
  dev.write32(0x020288, 0x00000000)
  dev.write32(0x02028c, 0x00000000)
  # Clear RED_SWITCH ENABLE bits to disable power-gating.
  _red_switch = dev.read32(0x409614)
  dev.write32(0x409614, _red_switch & ~0x700)
  # Ensure PWR_GATE has bit30=1 (un-gate), bit31=0 (disable gating).
  nvkm_mask(dev, 0x020004, 0xc0000000, 0x40000000)
  # Clear FECS mailbox and start FECS.
  dev.write32(0x409800, 0x00000000)
  # Read and clear the halt/reset register before starting.
  _halt_reg = dev.read32(FECS_FALCON_BASE + 0x10c)
  dev.write32(FECS_FALCON_BASE + 0x10c, 0x00000000)
  # Clear CPUCTL completely first, then start.
  _pre_start = dev.read32(0x409100)
  dev.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, 0x00000000)  # clear all
  time.sleep(0.001)
  _mid_start = dev.read32(0x409100)
  dev.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, FALCON_UC_CTRL_START)
  time.sleep(0.01)
  _post_start = dev.read32(0x409100)
  print(f"[kepler] FECS reload: halt_reg={_halt_reg:#x} "
        f"pre={_pre_start:#x} mid={_mid_start:#x} post={_post_start:#x} "
        f"PC={dev.read32(0x409ff0):#x}", flush=True)
  # Verify IMEM was loaded
  dev.write32(FECS_FALCON_BASE + FALCON_CODE_INDEX, 0)
  _imem_check = dev.read32(FECS_FALCON_BASE + FALCON_CODE)
  print(f"[kepler] FECS reload IMEM verify: IMEM[0]={_imem_check:#x} "
        f"(expect 0x039b0ef5)", flush=True)
  print(f"[kepler] FECS reloaded after pgob, waiting for ready...", flush=True)
  try:
    wait_cond(lambda: falcon_ready(dev, FECS_FALCON_BASE),
              timeout_ms=2000, msg="FECS ready after reload")
  except TimeoutError:
    print(f"[kepler] FECS NOT ready after reload: CPUCTL={dev.read32(0x409100):#x} "
          f"SCRATCH0={dev.read32(0x409800):#x} PC={dev.read32(0x409ff0):#x}",
          flush=True)
    return False
  # Set FE_PWR FORCE_ON and disable idle filter after FECS is ready.
  dev.write32(0x404170, 0x00000012)
  dev.write32(0x020288, 0x00000000)
  dev.write32(0x02028c, 0x00000000)
  _red_switch = dev.read32(0x409614)
  dev.write32(0x409614, _red_switch & ~0x700)
  nvkm_mask(dev, 0x020004, 0xc0000000, 0x40000000)
  print(f"[kepler] FECS reloaded and ready: CPUCTL={dev.read32(0x409100):#x} "
        f"SCRATCH0={dev.read32(0x409800):#x} FE_PWR={dev.read32(0x404170):#x}",
        flush=True)
  return True

def submit_launch(dev, words, signal_va, signal_pa, wait_value, done_value):
  """Set up a GK104 compute channel (RAMIN + USERD + GPFIFO), push `words` into
  the GPFIFO ring, kick the channel, and poll the host semaphore until it
  reaches `done_value`.  Writes go straight into the CPU-coherent sysmem mmap
  (`dev.vram`); `signal_pa` is the mmap offset of the semaphore page.  The GR
  context-buffer content (RAMIN 0x0210) and the exact GPFIFO ring base wiring
  are KEPLER-TODO pending a nouveau GK104 channel trace on silicon."""
  hw = dev.dev_impl.hw
  if hw is not None:
    hw.set_phase("channel-build")
  single_thread_rpc = (hw is None or getattr(hw, "_single_thread_rpc", True))
  submit_mode = os.environ.get("KEPLER_SUBMIT_MODE", "gpfifo")
  if submit_mode not in ("gpfifo", "bypass", "dispatch"):
    raise ValueError(
        f"invalid KEPLER_SUBMIT_MODE={submit_mode!r}; expected gpfifo, bypass, or dispatch")
  if submit_mode != "gpfifo" and os.environ.get("KEPLER_PUT_BEFORE_RUNLIST", "0") != "0":
    raise ValueError("KEPLER_PUT_BEFORE_RUNLIST is only valid in gpfifo mode")
  # Reassert the polling-only interrupt policy after VBIOS/engine init and
  # before any operation that can raise a PFIFO or PGRAPH source.
  dev.write32(0x000140, 0x00000000)
  dev.write32(0x000640, 0x00000000)
  dev.read32(0x000140)
  vram, base, mm = dev.dev_impl.vram, dev.dev_impl.bus_base, dev.dev_impl.mm
  signal_initial = struct.unpack_from("<I", vram, signal_pa)[0]
  # The old bring-up path fell back to undocumented/manual dispatch when the
  # normal scheduler did not become resident.  That turned a diagnosable MMU
  # fault into GPU DMA through an invalid context and could crash the host.
  # Keep those experiments available only as an explicit
  # developer opt-in; the normal add demo must fail closed.
  unsafe_experiments = os.environ.get("KEPLER_UNSAFE_EXPERIMENTS") == "1"
  alloc = NVAllocator(dev)
  ramin = alloc.alloc(0x1000)
  userd = alloc.alloc(0x200)
  gpfifo = alloc.alloc(0x2000, align=0x2000)   # 512 entries x 8 bytes, limit2=10
  runlist = alloc.alloc(0x1000, aspace=AddrSpace.NCOH)
  push = alloc.alloc(0x10000, align=0x10000)  # Nouveau main push buffer
  # GR engine context buffer (RAMIN 0x0210).  nouveau allocates
  # CB_RESERVED(0x80000) + gr->size and fills it via gf100_grctx_generate_main
  # (bundle/pagepool/attrib_cb ctxsw bundles).
  gr_ctx = alloc.alloc(0x100000)
  # Global GR buffers needed by grctx->main() (ctxgf100.c / ctxgk104.c).
  # pagepool: 0x8000 bytes, bundle_cb: 0x3000 bytes,
  # attrib_cb: 0x20 * (attrib_nr_max + alpha_nr_max) * tpc_total
  pagepool = alloc.alloc(0x8000, align=0x1000)
  bundle_cb = alloc.alloc(0x3000, align=0x1000)
  # attrib_cb size depends on tpc_total; use a safe default for GK104
  # (8 TPCs max on GTX 770): 0x20 * (0x324 + 0x7ff) * 8 = 0xb7c00
  attrib_cb = alloc.alloc(0x100000, align=0x1000)
  mmio_list = alloc.alloc(0x1000)
  chan_id = int(os.environ.get("KEPLER_CHAN_ID", "1"))
  userd_base_off = chan_id * 0x200
  use_vram_inst = (os.environ.get("KEPLER_VRAM_INST") != "0" and dev.dev_impl.hw is not None)
  # nv50_runl_alloc() uses NVKM_MEM_TARGET_INST; on discrete GK104 that is
  # framebuffer memory and gk104_runl_commit() emits TARGET=VRAM.  Host NCOH
  # is accepted by the descriptor format but is not Nouveau's normal path.
  use_vram_runlist = (os.environ.get("KEPLER_VRAM_RUNLIST", "1") != "0" and
                      dev.dev_impl.hw is not None)
  use_vram_gpfifo = (os.environ.get("KEPLER_VRAM_GPFIFO", "1") != "0" and
                      use_vram_inst)
  use_vram_push = (os.environ.get("KEPLER_VRAM_PUSH", "1") != "0" and
                    use_vram_inst)
  use_vram_signal = (os.environ.get("KEPLER_VRAM_SEMAPHORE", "1") != "0" and
                      use_vram_inst)
  from grctx_gk104 import GK104_GRCTX_CONSTS as _GC
  _tpc_nr = [dev.read32(0x500000 + gpc * 0x8000 + 0x2608) & 0x1f
             for gpc in range(dev.read32(0x409604) & 0x1f)]
  _ppc_masks = [(dev.read32(0x500c30 + gpc * 0x8000) & 0xff) or
                ((1 << _tpc_nr[gpc]) - 1) for gpc in range(len(_tpc_nr))]
  _bundle_size = _GC["bundle_size"]
  _state_limit = min(_GC["bundle_min_gpm_fifo_depth"], _bundle_size // 0x20)
  _alpha, _beta = _GC["alpha_nr"], _GC["attrib_nr"]
  runtime_mmio_entries = [
    (0x40800c, pagepool.va_addr >> 8), (0x408010, 0x80000000),
    (0x419004, pagepool.va_addr >> 8), (0x419008, 0),
    (0x4064cc, 0x80000000),
    (0x408004, bundle_cb.va_addr >> 8),
    (0x408008, 0x80000000 | (_bundle_size >> 8)),
    (0x418808, bundle_cb.va_addr >> 8),
    (0x41880c, 0x80000000 | (_bundle_size >> 8)),
    (0x4064c8, (_state_limit << 16) | _GC["bundle_token_limit"]),
    (0x418810, 0x80000000 | (attrib_cb.va_addr >> 12)),
    (0x419848, 0x10000000 | (attrib_cb.va_addr >> 12)),
    (0x405830, (_beta << 16) | _alpha),
    (0x4064c4, ((_alpha // 4) << 16) | 0xffff),
  ]
  print(f"[kepler] GR buffer VAs: pagepool={pagepool.va_addr:#x}+{pagepool.size:#x} "
        f"bundle={bundle_cb.va_addr:#x}+{bundle_cb.size:#x} "
        f"attrib={attrib_cb.va_addr:#x}+{attrib_cb.size:#x} "
        f"mmio={mmio_list.va_addr:#x}+{mmio_list.size:#x}", flush=True)
  _bo, _ao = 0, _GC["attrib_nr_max"] * sum(_tpc_nr)
  for _gpc, _mask in enumerate(_ppc_masks):
    _count = _mask.bit_count()
    _ppc = 0x503000 + _gpc * 0x8000
    runtime_mmio_entries.append(
      (_ppc + 0xc0, (1 << 28) | (_beta * _count << 16) | _bo))
    _bo += _GC["attrib_nr_max"] * _count
    runtime_mmio_entries.append((_ppc + 0xe4, (_alpha * _count << 16) | _ao))
    _ao += _GC["alpha_nr_max"] * _count
  runtime_mmio_entries.extend(((0x17e91c, dev.read32(0x17e91c)),
                               (0x17e920, dev.read32(0x17e920))))
  mmio_blob = b"".join(struct.pack("<II", reg, value & 0xffffffff)
                       for reg, value in runtime_mmio_entries)
  vram[mmio_list.meta["pa"]:mmio_list.meta["pa"] + len(mmio_blob)] = mmio_blob
  mmio_list.meta["priv"] = True
  if os.environ.get("KEPLER_MMIO_LIST_VRAM") == "1":
    dev._kepler_vram_mirrors = list(
        getattr(dev, "_kepler_vram_mirrors", ())) + [mmio_list]
  if DEBUG and dev.dev_impl.hw is not None:
    bar1_ctl = dev.read32(0x1704)
    bar1_inst_pa = (bar1_ctl & 0x3fffffff) << 12
    try:
      bar1_head = dev.dev_impl.hw.mmio_read(1, bar1_inst_pa + 0x200, 16).hex()
    except Exception as e:
      bar1_head = f"read-error:{type(e).__name__}"
    print(f"[kepler] existing_bar1 ctl={bar1_ctl:#x} inst={bar1_inst_pa:#x} "
          f"pdb={bar1_head}")
    # Decode BAR1 VMM PDB: bits [1:0]=target, [2]=VOL, [63:3]=PD address
    try:
      pdb_bytes = dev.dev_impl.hw.mmio_read(1, bar1_inst_pa + 0x200, 8)
      pdb_lo = struct.unpack("<I", pdb_bytes[0:4])[0]
      pdb_hi = struct.unpack("<I", pdb_bytes[4:8])[0]
      pdb_val = (pdb_hi << 32) | pdb_lo
      bar1_target = pdb_val & 0x3
      bar1_vol = (pdb_val >> 2) & 0x1
      bar1_pd_addr = pdb_val & ~0x7
      bar1_limit_bytes = dev.dev_impl.hw.mmio_read(1, bar1_inst_pa + 0x208, 8)
      bar1_limit = struct.unpack("<Q", bar1_limit_bytes)[0]
      print(f"[kepler] bar1_vmm pdb={pdb_val:#x} target={bar1_target} vol={bar1_vol} "
            f"pd_addr={bar1_pd_addr:#x} limit={bar1_limit:#x}")
      # Read first few PGD entries to see what's mapped
      if bar1_pd_addr and bar1_pd_addr < dev.dev_impl.bar1_size:
        for i in range(4):
          pte_bytes = dev.dev_impl.hw.mmio_read(1, bar1_pd_addr + i * 8, 8)
          pte_lo = struct.unpack("<I", pte_bytes[0:4])[0]
          pte_hi = struct.unpack("<I", pte_bytes[4:8])[0]
          pte_val = (pte_hi << 32) | pte_lo
          if pte_val & 0x1:  # valid bit
            print(f"[kepler] bar1_pgd[{i}]={pte_val:#x} -> maps VA "
                  f"[{i * (1<<27):#x}, {(i+1) * (1<<27):#x})")
    except Exception as e:
      print(f"[kepler] bar1_vmm decode error: {type(e).__name__}: {e}")
  # Do not allocate from low framebuffer memory: VBIOS/instmem/display reserve
  # portions of it, and writes at the old 4-MiB cursor returned alternating
  # stale dwords on this card.  Keep the bring-up heap inside the 128-MiB BAR1
  # aperture but above those bootstrap objects.
  bar1_cursor = int(os.environ.get("KEPLER_VRAM_HEAP_BASE", "0x400000"), 0)
  def bar1_alloc(size, align=0x1000):
    nonlocal bar1_cursor
    bar1_cursor = round_up(bar1_cursor, align)
    pa = bar1_cursor
    bar1_cursor += round_up(size, 0x1000)
    if bar1_cursor > dev.dev_impl.bar1_size:
      raise MemoryError("BAR1-backed instance allocation exceeds aperture")
    return pa
  def bar1_write(pa, data):
    # The BAR1 mmap can acknowledge a large write while only applying the
    # first portion on this un-POSTed card.  Keep each transfer within one
    # 4-KiB page; otherwise page-table tails remain uninitialised and FECS
    # DMA walks garbage.
    data = memoryview(data).cast("B")
    for off in range(0, len(data), 0x1000):
      dev.dev_impl.hw.mmio_write(1, pa + off, data[off:off + 0x1000].tobytes())
  ctx_alias_ptes = []
  if use_vram_inst:
    # Match gf100_fb_init_page() for GK104's default 128-KiB big-page mode.
    # The bit also selects the PGD/SPT split used for 4-KiB mappings.
    nvkm_mask(dev, 0x100c80, 0x1, 0x0)
    # gk104_chan_ramfc_write() puts the USERD memory object's VRAM address in
    # RAMFC 0x08.  BAR1 is then the CPU mapping used to access that VRAM.  Keep
    # those two address domains distinct; the PCI BAR resource address is not
    # a GPU USERD address.
    userd_vram_pa = bar1_alloc(0x2000, align=0x2000)
    # Nouveau's nvkm_gpuobj_new() uses NVKM_MEM_TARGET_INST for the channel
    # instance and GR context. Keep their direct physical addresses in the
    # framebuffer aperture, while ordinary buffers remain in the VMM SYS path.
    grctx_vram_pa = bar1_alloc(0x100000)
    pagepool_vram_pa = bar1_alloc(0x8000)
    bundle_vram_pa = bar1_alloc(0x3000)
    attrib_vram_pa = bar1_alloc(0x100000)
    mmio_list_vram_pa = bar1_alloc(0x1000)
    gpfifo_vram_pa = bar1_alloc(0x2000, align=0x2000) if use_vram_gpfifo else None
    if gpfifo_vram_pa is not None and os.environ.get("KEPLER_GPFIFO_IN_USERD") == "1":
      # Diagnostic: USERD page 0 offset 0 is unused by CHID 1 (its USERD is at
      # 0x200) and is already proven visible to PBDMA through 0x2254.
      gpfifo_vram_pa = userd_vram_pa
    push_vram_pa = bar1_alloc(0x10000, align=0x10000) if use_vram_push else None
    push_in_gpfifo = (push_vram_pa is not None and gpfifo_vram_pa is not None and
                      os.environ.get("KEPLER_PUSH_IN_GPFIFO") == "1")
    if push_in_gpfifo:
      push_vram_pa = gpfifo_vram_pa
    signal_vram_pa = bar1_alloc(0x1000) if use_vram_signal else None
    # Allocate RAMIN after the large scratch objects.  The first portion of the
    # raw BAR aperture is not retaining every dword on this un-POSTed card,
    # whereas later instance-memory pages (used by the VMM tables) are stable.
    ramin_vram_pa = bar1_alloc(0x1000)
    for pa, size in ((userd_vram_pa, 0x2000), (ramin_vram_pa, 0x1000),
                     (grctx_vram_pa, 0x100000), (pagepool_vram_pa, 0x8000),
                     (bundle_vram_pa, 0x3000), (attrib_vram_pa, 0x100000)):
      bar1_write(pa, bytes(size))
    # FECS saves only defined context words; all gaps must start at zero.
    # BAR1 bulk stores can leave inverted/stale dwords, so repair the
    # golden-context allocation before it becomes the source for every channel.
    attrib_repair_mode = os.environ.get("KEPLER_ATTRIB_REPAIR", "literal")
    if attrib_repair_mode not in ("literal", "bar1"):
      raise ValueError(
          "invalid KEPLER_ATTRIB_REPAIR; expected literal or bar1")
    def repair_zero(label, pa, size):
      # The compensated writer cannot be used for zero here: its transformed
      # immediate readback makes a raw 0xffffffff operand look like zero while
      # the GPU still observes all ones.  Store literal zero dwords instead.
      phase_label = label.lower().replace(" ", "-")
      if hw is not None:
        hw.set_phase(f"channel-build-repair-zero-{phase_label}")
      if label == "attrib" and attrib_repair_mode == "bar1":
        # Isolation mode: avoid the per-dword BAR0 PRAMIN aperture stream.
        # bar1_write() remains page-bounded, matching the already-used bulk
        # initialization path.  This is deliberately opt-in because the
        # un-POSTed card has shown stale/inverted BAR1 readback in places.
        bar1_write(pa, bytes(size))
      else:
        _gk104_pramin_write_literal(dev, pa, bytes(size))
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError(f"GK104 LTC invalidate while zeroing {label} failed")
      time.sleep(0.005)
      sample = dev.dev_impl.hw.mmio_read(1, pa, min(size, 0x20)).hex()
      print(f"[kepler] {label} raw-zero store complete sample={sample}", flush=True)
    repair_zero("GR ctx", grctx_vram_pa, 0x100000)
    repair_zero("pagepool", pagepool_vram_pa, 0x8000)
    repair_zero("bundle", bundle_vram_pa, 0x3000)
    repair_zero("attrib", attrib_vram_pa, 0x100000)
    if gpfifo_vram_pa is not None:
      bar1_write(gpfifo_vram_pa, bytes(0x2000))
    if push_vram_pa is not None:
      bar1_write(push_vram_pa, bytes(0x10000))
    if signal_vram_pa is not None:
      signal_page = bytearray(0x1000)
      struct.pack_into("<I", signal_page, 0, struct.unpack_from("<I", vram, signal_pa)[0])
      bar1_write(signal_vram_pa, signal_page)
      _gk104_pramin_write(dev, signal_vram_pa, signal_page[:4])
      _gk104_bar_flush(dev)
      if DEBUG:
        print(f"[kepler] VRAM semaphore pa={signal_vram_pa:#x} initial="
              f"{_gk104_pramin_read32(dev, signal_vram_pa):#x}", flush=True)
    gr_ctx = HCQBuffer(0x08000000, 0x100000, meta={"pa": grctx_vram_pa})
    dev.dev_impl.mm.map_range(gr_ctx.va_addr, gr_ctx.size,
                              [(grctx_vram_pa, gr_ctx.size)], AddrSpace.PHYS)
    # Map the MMIO list and other GR buffers in the channel's VMM so the
    # FECS firmware can read them via MEM_TARGET=VM (xdld).  Without these
    # mappings, the FECS hangs at xdld in ctx_mmio_loop because the MMIO
    # list VA has no page table entry.
    dev.dev_impl.mm.map_range(mmio_list.va_addr, 0x1000,
                              [(mmio_list_vram_pa, 0x1000)], AddrSpace.PHYS)
    dev.dev_impl.mm.map_range(pagepool.va_addr, 0x8000,
                              [(pagepool_vram_pa, 0x8000)], AddrSpace.PHYS)
    dev.dev_impl.mm.map_range(bundle_cb.va_addr, 0x3000,
                              [(bundle_vram_pa, 0x3000)], AddrSpace.PHYS)
    dev.dev_impl.mm.map_range(attrib_cb.va_addr, 0x100000,
                              [(attrib_vram_pa, 0x100000)], AddrSpace.PHYS)
    print(f"[kepler] VMM map: mmio_list va={mmio_list.va_addr:#x} "
          f"pa={mmio_list_vram_pa:#x}", flush=True)
    # Write the MMIO list data to VRAM so the FECS can read it via VMM.
    bar1_write(mmio_list_vram_pa, mmio_blob)
    _gk104_pramin_write(dev, mmio_list_vram_pa, mmio_blob)
    _gk104_bar_flush(dev)
  else:
    ramin_vram_pa = None
    userd_vram_pa = None
    pagepool_vram_pa = bundle_vram_pa = attrib_vram_pa = None
    mmio_list_vram_pa = None
    gpfifo_vram_pa = None
    push_vram_pa = None
    signal_vram_pa = None
  if use_vram_runlist:
    runlist_vram_pa = bar1_alloc(0x1000)
    bar1_write(runlist_vram_pa, bytes(0x1000))
  else:
    runlist_vram_pa = None
  # Kepler's main GPFIFO entry is laid out as push base + 0x10000. Mirror the
  # Nouveau virtual layout even though the two backing allocations are local.
  # Keep the FIFO beside (but non-overlapping with) the proven GR-context range
  # in PGD slot 1.  FECS already validates this PGD/SPT path on silicon.
  push_va = int(os.environ.get("KEPLER_PUSH_VA", "0x09000000"), 0)
  gpfifo_va = push_va + 0x10000
  mm.map_range(push_va, push.size, [(push.meta['pa'], push.size)], AddrSpace.NCOH)
  mm.map_range(gpfifo_va, gpfifo.size, [(gpfifo.meta['pa'], gpfifo.size)], AddrSpace.NCOH)
  push.va_addr, gpfifo.va_addr = push_va, gpfifo_va
  if DEBUG:
    print(f"[kepler] fifo_vas ramin={ramin.va_addr:#x} userd={userd.va_addr:#x} "
          f"gpfifo={gpfifo.va_addr:#x} runlist={runlist.va_addr:#x} push={push.va_addr:#x} "
          f"gr_ctx={gr_ctx.va_addr:#x}")
  if DEBUG:
    print(f"[kepler] vmm_flush={'done' if _gk104_vmm_flush(dev) else 'timeout'} "
          f"pdb={dev.dev_impl.mm.bus_base + dev.dev_impl.mm.root_pa:#x}")
  else:
    _gk104_vmm_flush(dev)

  ramin_pa, userd_pa, gpfifo_pa = ramin.meta['pa'], userd.meta['pa'], gpfifo.meta['pa']
  grctx_pa = gr_ctx.meta['pa']
  # ctxgf100.c allocates the golden context from INST/VRAM, not sysmem. Use a
  # BAR1-backed 1 MiB region for the hardware path; the sysmem allocation is
  # retained only as the software/offline backing object.
  # NOTE: The _int path (!gr->firmware) does NOT write data[0x1c/0x20/0x28/0x2c]
  # — those writes are _ext path only (ctxgf100.c lines 1499-1504).  The context
  # buffer starts zeroed, and grctx->main() populates the GR registers before
  # the context unload saves the golden image.
  ctx_header = gr_ctx.meta['pa'] + 0x80000
  inst = bytearray(0x1000)
  # FECS VM DMA needs a VRAM-resident page-table hierarchy.  Clone the complete
  # software VMM so context, push, semaphore, code, CWD and parameter mappings
  # all share exactly the same leaf encodings.
  if use_vram_inst:
    if DEBUG:
      probe_va = gr_ctx.va_addr + 0x80000
      probe_pgd_i = (probe_va >> 27) & 0x1fff
      probe_spt_i = (probe_va >> 12) & 0x7fff
      probe_src_pde = mm.root_page_table.entry(probe_pgd_i)
      probe_src_spt = mm.root_page_table.address(probe_pgd_i)
      probe_src_pte = struct.unpack_from("<Q", vram, probe_src_spt + probe_spt_i * 8)[0]
      print(f"[kepler] host VMM walk: pgd[{probe_pgd_i}]={probe_src_pde:#x} "
            f"spt={probe_src_spt:#x} pte[{probe_spt_i:#x}]={probe_src_pte:#x}", flush=True)
    vmm_pgd_pa, cloned_spts = _gk104_clone_vmm_to_vram(
      dev, bar1_alloc, bar1_write)
    # Some GR internal clients on the un-POSTed GK104 cannot fetch launch
    # descriptors through the NCOH host aperture even though PBDMA can.
    # Allow the caller to mirror selected buffers into framebuffer memory and
    # replace only their leaf PTEs in the channel's cloned VMM.
    cloned_by_pgd = {idx: dst for idx, _src, dst in cloned_spts}
    _alias_pgd_idx = 0x1fff
    _alias_spt_pa = bar1_alloc(0x40000)
    _alias_mem_pa = bar1_alloc(0x10000)
    bar1_write(_alias_spt_pa, bytes(0x40000))
    bar1_write(_alias_mem_pa, bytes(0x10000))
    cloned_by_pgd[_alias_pgd_idx] = _alias_spt_pa
    cloned_spts.append((_alias_pgd_idx, -1, _alias_spt_pa))
    _alias_cursor = _alias_mem_pa
    for _bad_base, _bad_size in (
        ((((~(bundle_cb.va_addr >> 8)) & 0xffffffff) << 8) & ((1 << 40) - 1),
         bundle_cb.size),
        ((((~(pagepool.va_addr >> 8)) & 0xffffffff) << 8) & ((1 << 40) - 1),
         pagepool.size)):
      _first = _bad_base & ~0xfff
      _pages = round_up((_bad_base & 0xfff) + _bad_size, 0x1000) // 0x1000
      for _page in range(_pages):
        _va = _first + _page * 0x1000
        _spti = (_va >> 12) & 0x7fff
        ctx_alias_ptes.append(
            (_alias_spt_pa + _spti * 8,
             ((_alias_cursor + _page * 0x1000) >> 8) | 1))
      _alias_cursor += _pages * 0x1000
    # The cold BAR path leaves unused SPT dwords as all ones.  SCC context
    # load touches this otherwise-unallocated scratch VA; give it a dedicated
    # writable guard page instead of accepting the stale RO PTE.
    ctx_alias_ptes.append(
        (cloned_by_pgd[0] + 0x67 * 8, (_alias_cursor >> 8) | 1))
    _alias_cursor += 0x1000
    for mirror in getattr(dev, "_kepler_vram_mirrors", ()):
      mirror_pa = bar1_alloc(round_up(mirror.size, 0x1000))
      src_pa = mirror.meta["pa"]
      image = bytes(vram[src_pa:src_pa + mirror.size])
      bar1_write(mirror_pa, image + bytes(round_up(mirror.size, 0x1000) - mirror.size))
      for page in range(round_up(mirror.size, 0x1000) // 0x1000):
        va = mirror.va_addr + page * 0x1000
        pgdi, spti = (va >> 27) & 0x1fff, (va >> 12) & 0x7fff
        pte = ((mirror_pa + page * 0x1000) >> 8) | 1
        if mirror.meta.get("priv"):
          pte |= 2
        _gk104_pramin_write(dev, cloned_by_pgd[pgdi] + spti * 8,
                            struct.pack("<Q", pte))
      mirror.meta["vram_pa"] = mirror_pa
      print(f"[kepler] VRAM mirror: va={mirror.va_addr:#x} pa={mirror_pa:#x} "
            f"size={mirror.size:#x}", flush=True)
    if getattr(dev, "_kepler_vram_mirrors", ()):
      _gk104_bar_flush(dev)
    # The GR scratch objects are INST/VRAM memory in Nouveau but all register
    # programming uses their channel virtual addresses.  Point those VAs at
    # our BAR1-backed allocations in the cloned channel VMM.
    for scratch, scratch_pa, privileged in ((pagepool, pagepool_vram_pa, True),
                                            (bundle_cb, bundle_vram_pa, True),
                                            (attrib_cb, attrib_vram_pa, False)):
      for page in range(round_up(scratch.size, 0x1000) // 0x1000):
        va = scratch.va_addr + page * 0x1000
        pgdi, spti = (va >> 27) & 0x1fff, (va >> 12) & 0x7fff
        pte = ((scratch_pa + page * 0x1000) >> 8) | 1
        if privileged:
          pte |= 2
        _gk104_pramin_write(dev, cloned_by_pgd[pgdi] + spti * 8,
                            struct.pack("<Q", pte))
    # gk104_ectx_ctor() maps the engine context with gf100_vmm_map_v0.priv=1.
    for page in range(gr_ctx.size // 0x1000):
      va = gr_ctx.va_addr + page * 0x1000
      pgdi, spti = (va >> 27) & 0x1fff, (va >> 12) & 0x7fff
      pte = ((grctx_vram_pa + page * 0x1000) >> 8) | 3
      _gk104_pramin_write(dev, cloned_by_pgd[pgdi] + spti * 8,
                          struct.pack("<Q", pte))
    _gk104_bar_flush(dev)
    if gpfifo_vram_pa is not None or push_vram_pa is not None or signal_vram_pa is not None:
      signal_buf = HCQBuffer(signal_va, 0x1000, meta={"pa": signal_vram_pa})
      for buf, vram_pa in ((gpfifo, gpfifo_vram_pa), (push, push_vram_pa),
                           (signal_buf, signal_vram_pa)):
        if vram_pa is None:
          continue
        pgdi = (buf.va_addr >> 27) & 0x1fff
        spti = (buf.va_addr >> 12) & 0x7fff
        for page in range(buf.size // 0x1000):
          pte = ((vram_pa + page * 0x1000) >> 8) | 1
          _gk104_pramin_write(dev, cloned_by_pgd[pgdi] + (spti + page) * 8,
                              struct.pack("<Q", pte))
      _gk104_bar_flush(dev)
    pgd = vmm_pgd_pa
    vram_vmm_flushed = _gk104_vmm_flush_pdb(dev, vmm_pgd_pa, target=0)
    print(f"[kepler] VRAM VMM: pgd_pa={vmm_pgd_pa:#x} "
          f"spts={[(i, hex(dst)) for i, _src, dst in cloned_spts]} "
          f"join={pgd:#x} flush={'done' if vram_vmm_flushed else 'timeout'}", flush=True)
  else:
    pgd = _gk104_pgd_entry(dev, mm.root_pa)
  # gf100_vmm_join(): the channel VMM instance stores the page-directory
  # pointer at 0x200 and the VMM limit-1 at 0x208 (vmmgf100.c gf100_vmm_join_).
  struct.pack_into("<I", inst, 0x0200, pgd & 0xffffffff)
  struct.pack_into("<I", inst, 0x0204, pgd >> 32)
  # VMM limit: 40-bit VA space -> limit = (1<<40) - 1 = 0xFFFFFFFFFF
  struct.pack_into("<Q", inst, 0x0208, (1 << 40) - 1)
  userd_addr = ((userd_vram_pa + userd_base_off) if use_vram_inst else
                (dev.dev_impl.bar1_addr + userd_base_off))
  struct.pack_into("<I", inst, 0x08, userd_addr & 0xffffffff)   # USERD lo
  struct.pack_into("<I", inst, 0x0c, userd_addr >> 32)          # USERD hi
  # ctxgf100.c: RAMIN points at ctx->addr + CB_RESERVED (0x80000), not the
  # allocation base; bit 2 marks the engine context pointer valid.
  grctx_va = (gr_ctx.va_addr + 0x80000) | 4
  struct.pack_into("<I", inst, 0x0210, grctx_va & 0xffffffff)  # GR ctx ptr lo
  struct.pack_into("<I", inst, 0x0214, grctx_va >> 32)        # GR ctx ptr hi
  # Diagnostic: verify context buffer is mapped in the channel's page table.
  if dev.dev_impl.hw is not None:
    _ctx_va = gr_ctx.va_addr + 0x80000
    _pgd_idx = (_ctx_va >> 27) & 0x1fff
    _spt_idx = (_ctx_va >> 12) & 0x7fff
    _pgd_entry = dev.dev_impl.mm.root_page_table.entry(_pgd_idx)
    _spt = GK104PageTableEntry(dev.dev_impl, dev.dev_impl.mm.root_page_table.address(_pgd_idx), 0)
    _pte = _spt.entry(_spt_idx)
    print(f"[kepler] GR ctx buffer: va={_ctx_va:#x} pa={gr_ctx.meta['pa']:#x} "
          f"pgd[{_pgd_idx}]={_pgd_entry:#x} pte[{_spt_idx}]={_pte:#x}", flush=True)
    # Also check the PTE in the VRAM VMM's SPT (the one the GR engine uses)
    if 'vmm_pgd_pa' in dir() or 'vmm_pgd_pa' in locals():
      _vram_spt_pa = 0x673000  # from VRAM VMM diagnostic
      _vram_pte = _gk104_pramin_read32(dev, _vram_spt_pa + _spt_idx * 8)
      _vram_pte_hi = _gk104_pramin_read32(dev, _vram_spt_pa + _spt_idx * 8 + 4)
      print(f"[kepler] VRAM VMM PTE[{_spt_idx}]: lo={_vram_pte:#x} hi={_vram_pte_hi:#x} "
            f"(expected lo={((gr_ctx.meta['pa'] >> 8) | 1):#x})", flush=True)
  struct.pack_into("<I", inst, 0x48, 0)   # GP_PUT (relative)
  struct.pack_into("<I", inst, 0x4c, 0)   # GP_GET (relative)
  if use_vram_inst:
    bar1_write(ramin_vram_pa, inst)
    # RAMFC contains many meaningful zero fields as well as the nonzero
    # GPFIFO/VMM words.  A bulk BAR1 upload can leave old 0xffffffff dwords;
    # verify the complete live RAMFC/VMM prefix before allowing PBDMA to load.
    _gk104_pramin_write(dev, ramin_vram_pa, inst[:0x220])
    _gk104_bar_flush(dev)
  else:
    vram[ramin_pa:ramin_pa + len(inst)] = bytes(inst)
  if DEBUG and use_vram_inst:
    inst_read = dev.dev_impl.hw.mmio_read(1, ramin_vram_pa + 0x48, 8)
    print(f"[kepler] ramfc48={inst_read.hex()} userd={userd_addr:#x} "
          f"gpfifo_va={gpfifo.va_addr:#x} ring_pa={gpfifo.meta['pa']:#x}")

  # The methods live in a push buffer.  A GK104 GP entry is 8 bytes: the
  # push-buffer GPU VA followed by its word count in bits 10..29.
  push_bytes = bytearray(len(words) * 4)
  for i, w in enumerate(words):
    struct.pack_into("<I", push_bytes, i * 4, w)
  vram[push.meta['pa']:push.meta['pa'] + len(push_bytes)] = bytes(push_bytes)
  push_phys_offset = 0x100 if (use_vram_inst and
                               os.environ.get("KEPLER_PUSH_IN_GPFIFO") == "1") else 0
  if push_vram_pa is not None:
    bar1_write(push_vram_pa + push_phys_offset, push_bytes)
    _gk104_pramin_write(dev, push_vram_pa + push_phys_offset, push_bytes)
    _gk104_bar_flush(dev)
    if DEBUG:
      print(f"[kepler] VRAM push pa={push_vram_pa:#x} words="
            f"{[hex(x) for x in struct.unpack_from('<' + 'I' * len(words), push_bytes)]}",
            flush=True)
  ring = bytearray(GPFIFO_ENTRY_BYTES)
  push_addr = push.va_addr + push_phys_offset
  # This is the channel's main push buffer, matching nvif_chan_gpfifo_push_kick
  # (main=true), so BIT(9) remains clear. The env override is a diagnostic for
  # the alternate external-entry encoding used by nvif_chan_gpfifo_push().
  gpfifo_external = os.environ.get("KEPLER_GPFIFO_EXTERNAL") == "1"
  gpfifo_no_prefetch = os.environ.get("KEPLER_GPFIFO_NO_PREFETCH") == "1"
  struct.pack_into("<II", ring, 0, push_addr & 0xffffffff,
                   (push_addr >> 32) | ((1 << 9) if gpfifo_external else 0) |
                   (len(words) << 10) | ((1 << 31) if gpfifo_no_prefetch else 0))
  ring_store = bytearray(ring)
  if os.environ.get("KEPLER_GPU_XOR_RING") == "1":
    struct.pack_into("<I", ring_store, 0,
                     (~struct.unpack_from("<I", ring, 0)[0]) & 0xffffffff)
  vram[gpfifo_pa:gpfifo_pa + len(ring)] = ring
  if gpfifo_vram_pa is not None:
    ring_page = bytearray(0x1000)
    ring_page[:len(ring)] = ring
    bar1_write(gpfifo_vram_pa, ring_page)
    _gk104_pramin_write(dev, gpfifo_vram_pa, ring)
    if push_vram_pa == gpfifo_vram_pa and push_phys_offset:
      bar1_write(push_vram_pa + push_phys_offset, push_bytes)
      _gk104_pramin_write(dev, push_vram_pa + push_phys_offset, push_bytes)
    _gk104_bar_flush(dev)
    if DEBUG:
      bar_ring = dev.dev_impl.hw.mmio_read(1, gpfifo_vram_pa, 8).hex()
      pri_ring = (_gk104_pramin_read32(dev, gpfifo_vram_pa),
                  _gk104_pramin_read32(dev, gpfifo_vram_pa + 4))
      print(f"[kepler] VRAM ring expected={ring.hex()} bar={bar_ring} "
            f"pramin={[hex(x) for x in pri_ring]}", flush=True)

  # RAMFC GPFIFO base and ring-size log2, matching gk104_chan_ramfc_write().
  gpfifo_va = gpfifo.va_addr
  struct.pack_into("<II", inst, 0x48, gpfifo_va & 0xffffffff,
                   (gpfifo_va >> 32) | (10 << 16))
  # Remaining fields from gk104_chan_ramfc_write().
  struct.pack_into("<I", inst, 0x10, 0x0000face)
  struct.pack_into("<I", inst, 0x30, 0xfffff902)
  struct.pack_into("<I", inst, 0x84, 0x20400000)
  # gk104_chan_ramfc's 0xfff is the class capability default; channel
  # creation passes the selected engine mask to ramfc_write().  This channel
  # belongs to GR only, so advertise BIT(GR engine id 0).  Advertising absent
  # video/copy engines raises PBDMA DEVICE on this topology.
  struct.pack_into("<I", inst, 0x94, 0x30000001)
  struct.pack_into("<I", inst, 0x9c, 0x00000100)
  struct.pack_into("<I", inst, 0xac, 0x0000001f)
  # gk104_chan_ramfc.priv advertises support for the constructor's `priv`
  # argument; an ordinary userspace channel passes false.
  struct.pack_into("<I", inst, 0xe4, 0x00000000)
  struct.pack_into("<I", inst, 0xe8, chan_id)
  struct.pack_into("<I", inst, 0xb8, 0xf8000000)
  struct.pack_into("<II", inst, 0xf8, 0x10003080, 0x10000010)
  if use_vram_inst:
    bar1_write(ramin_vram_pa, inst)
    _gk104_pramin_write(dev, ramin_vram_pa, inst[:0x220])
    _gk104_bar_flush(dev)
  else:
    vram[ramin_pa:ramin_pa + len(inst)] = bytes(inst)
  if DEBUG and use_vram_inst:
    # Read the instance block PDB and channel PDE through individual byte BAR1
    # reads for diagnostics only.  Never use untrusted BAR readback to drive
    # the live path; PRAMIN is authoritative for page-table validation.
    def _bar1_read_u32(pa):
      return int.from_bytes(b"".join(dev.dev_impl.hw.mmio_read(1, pa + i, 1)
                                     for i in range(4)), "little")
    # Read PDB from instance block
    _pdb_lo = _bar1_read_u32(ramin_vram_pa + 0x200)
    _pdb_hi = _bar1_read_u32(ramin_vram_pa + 0x204)
    # Read PDE for signal VA (pgd_idx=0) from cloned page directory
    _pde_lo = _bar1_read_u32(vmm_pgd_pa + 0)
    _pde_hi = _bar1_read_u32(vmm_pgd_pa + 4)
    # Read PTE for signal VA (spt_idx=4) from SPT
    _spt_pa = ((_pde_lo | (_pde_hi << 32)) >> 24) & ~0xfff
    _pte_lo = _bar1_read_u32(_spt_pa + 4 * 8)
    _pte_hi = _bar1_read_u32(_spt_pa + 4 * 8 + 4)

  # gk104_mc_reset[] maps NVKM_ENGINE_FIFO to PMC_ENABLE bit 0x100.  A
  # previous process can leave the PBDMA context cache alive even
  # though the RAMFC and USERD backing stores have been rewritten.  Keep this
  # diagnostic opt-in until the live path is stable; it must happen before the
  # PBDMA/runlist programming below.
  if os.environ.get("KEPLER_FIFO_RESET") == "1":
    before_fifo = dev.read32(PMC_ENABLE)
    nvkm_mask(dev, PMC_ENABLE, 0x00000100, 0)
    time.sleep(0.01)
    nvkm_mask(dev, PMC_ENABLE, 0x00000100, 0x00000100)
    if DEBUG:
      print(f"[kepler] fifo_mc_reset PMC_ENABLE {before_fifo:#x} -> "
            f"{dev.read32(PMC_ENABLE):#x}")

  # gk104_runl_insert_chan(): runlist entries are (chid, 0), followed by a
  # commit of the runlist GPU address and entry count.
  runlist_pa = runlist_vram_pa if use_vram_runlist else runlist.meta['pa']
  if use_vram_runlist:
    runlist_entry = struct.pack("<II", chan_id, 0)
    bar1_write(runlist_pa, runlist_entry)
    _gk104_pramin_write(dev, runlist_pa, runlist_entry)
    _gk104_bar_flush(dev)
  else:
    struct.pack_into("<II", vram, runlist_pa, chan_id, 0)

  # Golden context initialization (ctxgf100.c, !gr->firmware path).
  #
  # The Nouveau FECS firmware (hubgk104.fuc3) only supports FIFO commands
  # 1 (ctx_chan: set current channel) and 2 (ctx_save: save context).
  # Commands 3 (bind_pointer) and 9 (golden_save) are from the secure-boot
  # firmware path (gr->firmware) and are NOT supported by the internal
  # firmware loaded here.  Using them causes E_BAD_COMMAND and a timeout.
  #
  # The correct _int path is:
  # 1. Clear CC_SCRATCH(0) bit31 (the "done" flag) via CC_SCRATCH_CLR(0)
  # 2. Send FIFO cmd 1 (set channel) with inst_tag as data
  # 3. Wait for CC_SCRATCH(0) bit31 to be set again (FECS signals done)
  # 4. (Context population via grctx->main() happens here in Nouveau)
  # 5. Trigger a context unload: clear CHAN_NEXT bit31, set IRQMSET bit8
  # 6. Wait for CHAN_ADDR bit31 to clear (context switch complete)
  # 7. Clear CHAN_ADDR bit31
  if dev.dev_impl.hw is not None:
    hw.set_phase("golden-context")
    # ponytail: gk104_fifo_init must run BEFORE the PBDMA bind (which submits
    # a runlist).  Without SUBFIFO_ENABLE and runq init, the runlist DMA
    # never completes — the pending bit stays set forever.
    if os.environ.get("KEPLER_USERD_BAR1_ZERO") == "1":
      userd_bar1_base = 0
    else:
      userd_bar1_base = (dev.dev_impl.bar1_addr
                         if os.environ.get("KEPLER_USERD_BAR1_RESOURCE") == "1"
                         else (userd_vram_pa if use_vram_inst else dev.dev_impl.bar1_addr))
    dev.write32(0x2254, 0x10000000 | (userd_bar1_base >> 12))
    dev.write32(0x000204, 0x7)  # SUBFIFO_ENABLE: 3 PBDMAs
    nvkm_mask(dev, 0x2a04, 0xbfffffff, 0xbfffffff)
    for pbdma in range(3):
      dev.read32(0x409100)  # FECS keep-alive
      q = 0x40000 + pbdma * 0x2000
      dev.write32(q + 0xc0, 0x80600008)
      dev.write32(q + 0x13c, dev.read32(q + 0x13c) & ~0x10000100)
      dev.write32(q + 0x108, 0xffffffff)
      dev.write32(q + 0x10c, 0xfffffeff)
      dev.write32(q + 0x148, 0xffffffff)
      dev.write32(q + 0x14c, 0xffffffff)
    dev.write32(0x2100, 0xffffffff)
    dev.write32(0x2140, 0x7fffffff)
    dev.write32(0x259c, 0xffffffff)
    dev.write32(0x2a00, 0xffffffff)
    # Check GR busy status BEFORE any PBDMA/bind operations
    print(f"[kepler] GR status before PBDMA bind: 0x40060c={dev.read32(0x40060c):#x} "
          f"0x2640={dev.read32(0x2640):#x} 0x400100={dev.read32(0x400100):#x} "
          f"0x409c18={dev.read32(0x409c18):#x}", flush=True)
    # ponytail: Clear stale GR busy state from previous failed runs.
    # The FECS exception (0x409c18 bit 4) keeps GR busy (0x40060c=0x1).
    # Clear the exception and PGRAPH interrupt, then re-enable PGRAPH.
    if dev.read32(0x40060c) & 0x1:
      _exc = dev.read32(0x409c18)
      _intr = dev.read32(0x400100)
      print(f"[kepler] GR busy from stale state: exc={_exc:#x} intr={_intr:#x}", flush=True)
      if _exc: dev.write32(0x409c20, _exc)
      if _intr: dev.write32(0x400100, _intr)
      dev.write32(0x400500, 0x00010001)
      time.sleep(0.05)
      print(f"[kepler] After stale clear: 0x40060c={dev.read32(0x40060c):#x} "
            f"0x409c18={dev.read32(0x409c18):#x} 0x400100={dev.read32(0x400100):#x}", flush=True)
    # ponytail: Bind the PBDMA channel BEFORE ctx_chan so the FIFO knows about
    # the channel.  Without this, ctx_chan makes an unknown channel current,
    # the FIFO marks GR as busy (0x40060c=0x1), and the FECS ctx_xfer_idle
    # loop never completes.  Nouveau does this at t=2144.494 (before grctx).
    _chan_ctrl_reg = CHAN_START_REG + chan_id * 8
    _chan_inst_reg = CHAN_SUBMIT_REG + chan_id * 8
    _rl_addr = runlist_pa if use_vram_runlist else (base + runlist_pa)
    _rl_target = 0 if use_vram_runlist else 3
    _pending_reg = 0x2284 + GR_RUNLIST_ID * 8
    # Clear stale runlist
    nvkm_mask(dev, 0x2630, 1 << GR_RUNLIST_ID, 1 << GR_RUNLIST_ID)
    _deadline = time.time() + 0.1
    while dev.read32(_pending_reg) & 0x00100000 and time.time() < _deadline:
      time.sleep(0.001)
    # Stop channel + unbind old inst
    nvkm_mask(dev, _chan_ctrl_reg, 0x800, 0x800)
    dev.write32(_chan_inst_reg, 0)
    # Set runlist ID (bits 16..19)
    nvkm_mask(dev, _chan_ctrl_reg, 0x000f0000, GR_RUNLIST_ID << 16)
    # Clear USERD GP_GET/GP_PUT and other pointers
    _userd_mmio_base = ((userd_vram_pa + userd_base_off) if use_vram_inst else
                        userd_base_off)
    for _uo in (0x40, 0x44, 0x48, 0x4c, 0x50, 0x58, 0x5c, 0x60,
                USERD_GP_GET, USERD_GP_PUT):
      dev.dev_impl.hw.mmio_write(1, _userd_mmio_base + _uo, struct.pack("<I", 0))
      if use_vram_inst:
        _gk104_pramin_write(dev, _userd_mmio_base + _uo, struct.pack("<I", 0))
    # Write inst pointer (triggers PBDMA bind)
    _bind_addr = ramin_vram_pa if use_vram_inst else (base + ramin_pa)
    dev.write32(_chan_inst_reg, 0x80000000 | (_bind_addr >> 12))
    # ponytail: Do NOT set ENABLE_TRIGGER yet — the channel has no work.
    # Setting it before ctx_chan may cause the PBDMA to spin and keep GR busy.
    # Submit runlist with 1 entry (the golden-context channel)
    _pfifo_intr = dev.read32(0x2a00)
    if _pfifo_intr:
      dev.write32(0x2a00, _pfifo_intr)
    dev.write32(0x2270, (_rl_target << 28) | (_rl_addr >> 12))
    dev.write32(0x2274, (GR_RUNLIST_ID << 20) | 1)  # count=1
    _deadline = time.time() + 0.2
    while dev.read32(_pending_reg) & 0x00100000:
      if time.time() >= _deadline:
        print(f"[kepler] runlist commit timeout: pending={dev.read32(_pending_reg):#x}", flush=True)
        break
      time.sleep(0.001)
    _pfifo_intr = dev.read32(0x2a00)
    if _pfifo_intr:
      dev.write32(0x2a00, _pfifo_intr)
    print(f"[kepler] PBDMA bind+runlist: 0x2640={dev.read32(0x2640):#x} "
          f"ctrl=0x{dev.read32(_chan_ctrl_reg):08x} "
          f"inst=0x{dev.read32(_chan_inst_reg):08x}", flush=True)
    inst_phys = ramin_vram_pa if use_vram_inst else (base + ramin_pa)
    inst_tag = 0x80000000 | (inst_phys >> 12)
    print(f"[kepler] FECS golden ctx: inst_tag={inst_tag:#x} inst_phys={inst_phys:#x} "
          f"ramin_pa={ramin_pa:#x} ramin_vram_pa={ramin_vram_pa if ramin_vram_pa is not None else 0:#x} base={base:#x}", flush=True)
    # gf100_grctx_generate() ordering is significant: hold FE power on while
    # releasing the FECS domains, then init SCC RAM before making the
    # temporary golden-context channel current.  Keep FE_PWR at FORCE_ON
    # (not AUTO) to prevent PGRAPH sub-domains from power-gating.
    dev.write32(0x404170, 0x00000012)  # NV_PGRAPH_FE_PWR_MODE_FORCE_ON
    wait_cond(lambda: not bool(dev.read32(0x404170) & 0x00000010),
              timeout_ms=2000, msg="FE power FORCE_ON")
    _gk104_gr_fecs_reset(dev)
    # Do NOT restore AUTO — keep FORCE_ON to prevent PGRAPH sub-domain gating.
    dev.write32(0x40802c, 0x00000001)  # initialise SCC RAM
    print(f"[kepler] grctx reset: RED_SWITCH={dev.read32(0x409614):#x} "
          f"FE_PWR={dev.read32(0x404170):#x} SCC={dev.read32(0x40802c):#x}", flush=True)
    # GPC MMU setup (gf100_gr_init_gpc_mmu): the GPC MMU needs buffer
    # addresses for virtual address translation.  Without this, compute
    # kernels can't access buffers via virtual addresses.
    # This must be done BEFORE ctx_chan while the initialized GPC_BCAST domain
    # is accessible.  Do not reset RED_SWITCH here: gf100_gr_fecs_reset already
    # did that during engine init, and repeating it with a live context is unsafe.
    # Allocate mmu_rd and mmu_wr buffers (1 page each, 0x1000 bytes).
    if use_vram_inst:
      mmu_rd_pa = bar1_alloc(0x1000)
      mmu_wr_pa = bar1_alloc(0x1000)
      bar1_write(mmu_rd_pa, bytes(0x1000))
      bar1_write(mmu_wr_pa, bytes(0x1000))
      _gk104_bar_flush(dev)
    else:
      mmu_rd = alloc.alloc(0x1000, align=0x1000)
      mmu_wr = alloc.alloc(0x1000, align=0x1000)
      mmu_rd_pa = base + mmu_rd.meta['pa']
      mmu_wr_pa = base + mmu_wr.meta['pa']
    # gf100_gr_init_gpc_mmu():
    #   0x418880 = 0x100c80 & 1  (big page size bit)
    #   0x4188a4 = 0x03000000
    #   0x418888..0x418894 = 0
    #   0x4188b4 = mmu_wr >> 8
    #   0x4188b8 = mmu_rd >> 8
    _bigpage = dev.read32(0x100c80) & 0x1
    dev.write32(0x418880, _bigpage)
    dev.write32(0x4188a4, 0x03000000)
    dev.write32(0x418888, 0x00000000)
    dev.write32(0x41888c, 0x00000000)
    dev.write32(0x418890, 0x00000000)
    dev.write32(0x418894, 0x00000000)
    dev.write32(0x4188b4, mmu_wr_pa >> 8)
    dev.write32(0x4188b8, mmu_rd_pa >> 8)
    # Verify all GPC MMU writes stuck
    _mmu_readback = {f"0x{r:x}": dev.read32(r) for r in
                     (0x418880, 0x4188a4, 0x418888, 0x41888c,
                      0x418890, 0x418894, 0x4188b4, 0x4188b8)}
    print(f"[kepler] GPC MMU: bigpage={_bigpage} mmu_wr={mmu_wr_pa:#x} "
          f"mmu_rd={mmu_rd_pa:#x} readback={_mmu_readback}", flush=True)
    # Check if GPC_BCAST domain is accessible
    _gpc_bcast_test = dev.read32(0x418804)
    _pwr_gate_pre = dev.read32(0x020004)
    _pmc_enable_pre = dev.read32(0x000200)
    _pgraph_ctrl_pre = dev.read32(0x400500)
    print(f"[kepler] GPC_BCAST test: 0x418804={_gpc_bcast_test:#x} "
          f"PWR_GATE={_pwr_gate_pre:#x} PMC_ENABLE={_pmc_enable_pre:#x} "
          f"PGRAPH_CTRL={_pgraph_ctrl_pre:#x}", flush=True)
    # Diagnostic: dump FECS and GR state before golden context init.
    fecs_stat0 = dev.read32(0x409800)
    fecs_cpustat = dev.read32(0x409128)   # CPUSTAT
    fecs_intr = dev.read32(0x409008)      # INTR (interrupt status)
    fecs_iren = dev.read32(0x409010)      # INTR_EN_SET
    print(f"[kepler] FECS before golden ctx: CC_SCRATCH0={fecs_stat0:#x} "
          f"CPUCTL={dev.read32(0x409100):#x} CPUSTAT={fecs_cpustat:#x} "
          f"INTR={fecs_intr:#x} IREN={fecs_iren:#x} "
          f"PGRAPH_STATUS={dev.read32(0x400000):#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x} "
          f"GPC_TOPOLOGY={dev.read32(0x409604):#x} "
          f"CHAN_ADDR={dev.read32(0x409b00):#x} CHAN_NEXT={dev.read32(0x409b04):#x}", flush=True)
    # 1. Make channel current: clear done flag, send cmd 1 (ctx_chan).
    dev.write32(0x409840, 0x80000000)   # CC_SCRATCH_CLR(0): clear bit31
    dev.write32(0x409500, inst_tag)      # FIFO data = inst tag
    dev.write32(0x409504, 0x00000001)    # FIFO cmd = 1 (set channel)
    # 2. Wait for CC_SCRATCH(0) bit31 (FECS sets this in main_done).
    try:
      wait_cond(lambda: bool(dev.read32(0x409800) & 0x80000000),
                timeout_ms=2000, msg="FECS ctx_chan (cmd 1)")
    except TimeoutError:
      print(f"[kepler] FECS ctx_chan TIMEOUT: CC_SCRATCH0={dev.read32(0x409800):#x} "
            f"CPUCTL={dev.read32(0x409100):#x} CPUSTAT={dev.read32(0x409128):#x} "
            f"INTR={dev.read32(0x409008):#x} CHAN_ADDR={dev.read32(0x409b00):#x} "
            f"CHAN_NEXT={dev.read32(0x409b04):#x}", flush=True)
      # Extended diagnostics: FECS PC, MMIO, GPC status
      import time as _t
      pcs = []
      for _ in range(10):
        pcs.append(dev.read32(0x409ff0))
        _t.sleep(0.001)
      print(f"[kepler] FECS PC after timeout: {[hex(p) for p in pcs]}")
      print(f"[kepler] FECS MMIO_CTRL=0x{dev.read32(0x409728):08x} "
            f"MMIO_RDVAL=0x{dev.read32(0x40972c):08x} "
            f"SIGNAL=0x{dev.read32(0x409400):08x} "
            f"IDLE=0x{dev.read32(0x409420):08x}")
      print(f"[kepler] FECS DMA: MEM_BASE=0x{dev.read32(0x409a04):08x} "
            f"MEM_CHAN=0x{dev.read32(0x409a0c):08x} "
            f"MEM_CMD=0x{dev.read32(0x409a10):08x} "
            f"MEM_TARGET=0x{dev.read32(0x409a20):08x}")
      for gpc in range(4):
        gpc_base = 0x502000 + gpc * 0x8000
        gpccs_base = 0x41a000 + gpc * 0x2000
        print(f"[kepler] GPC{gpc}: CTRL=0x{dev.read32(gpc_base+0x100):08x} "
              f"SCRATCH0=0x{dev.read32(gpc_base+0x800):08x} "
              f"GPCCS_CTRL=0x{dev.read32(gpccs_base+0x100):08x} "
              f"GPCCS_SCRATCH0=0x{dev.read32(gpccs_base+0x800):08x}")
      raise
    print(f"[kepler] FECS ctx_chan done: CC_SCRATCH0={dev.read32(0x409800):#x} "
          f"CHAN_ADDR={dev.read32(0x409b00):#x} CHAN_NEXT={dev.read32(0x409b04):#x}", flush=True)
    # Check GR busy status right after ctx_chan
    print(f"[kepler] GR status after ctx_chan: 0x40060c={dev.read32(0x40060c):#x} "
          f"0x400700={dev.read32(0x400700):#x} 0x400100={dev.read32(0x400100):#x} "
          f"0x409c18={dev.read32(0x409c18):#x} "
          f"0x2640={dev.read32(0x2640):#x} 0x2644={dev.read32(0x2644):#x} "
          f"FECS_ERR=0x{dev.read32(0x409814):08x} "
          f"FECS_PC=0x{dev.read32(0x409ff0):08x}", flush=True)
    # ponytail: The FECS sets an exception (0x409c18 bit 4) during ctx_chan
    # that keeps GR busy (0x40060c=0x1) and may cause the FECS to jump to an
    # EFI overlay handler.  Clear it IMMEDIATELY (no delay) to prevent the
    # FECS from getting stuck in the overlay.  Re-enable PGRAPH like nouveau's
    # gf100_gr_intr epilogue.
    _fecs_exc = dev.read32(0x409c18)
    _pgraph_intr = dev.read32(0x400100)
    if _fecs_exc or _pgraph_intr:
      dev.write32(0x409c20, _fecs_exc)
      dev.write32(0x400100, _pgraph_intr)
      dev.write32(0x400500, 0x00010001)
      print(f"[kepler] Cleared exc={_fecs_exc:#x} intr={_pgraph_intr:#x}: "
            f"0x40060c={dev.read32(0x40060c):#x} ctrl={dev.read32(0x400500):#x} "
            f"PC={dev.read32(0x409ff0):#x}", flush=True)
    # Start a keep-alive thread to prevent FE domain power-gating during
    # grctx_main and the golden context save.  The FECS sleeps after ctx_chan
    # and the FE domain power-gates without FORCE_ON.
    _ka_stop2 = threading.Event()
    _ka_thread2 = None
    def _ka2():
      while not _ka_stop2.is_set():
        try:
          _pwr = dev.read32(0x404170)
          if _pwr & 0x10:
            dev.write32(0x404170, (_pwr & ~0x10) | 0x02)
          elif _pwr != 0x02:
            dev.write32(0x404170, 0x00000002)
        except Exception:
          pass
        time.sleep(0.0002)
    # ponytail: Always run the keep-alive thread during grctx_main + golden
    # save.  Without it, the FE domain power-gates and FECS registers return
    # 0xbadf1000, making the wfi_golden_save FIFO command unreachable.
    dev.write32(0x404170, 0x00000002)
    _ka_thread2 = threading.Thread(target=_ka2, daemon=True)
    _ka_thread2.start()
    # 3. Populate GR context via grctx->main() (gf100_grctx_generate_main).
    #    This writes the GR register init lists, pagepool/bundle/attrib_cb
    #    addresses, icmd and mthd bundles.  The channel is current so GR
    #    registers are directly accessible.
    # Read GPC/TPC topology (gf100.c gr_0x409604):
    #   gpc_nr = 0x409604 & 0x1f
    #   tpc_nr[i] = GPC_UNIT(i, 0x2608) = 0x500000 + i*0x8000 + 0x2608
    #   tpc_total = sum of tpc_nr[i]
    gpc_nr = dev.read32(0x409604) & 0x1f
    tpc_nr = [dev.read32(0x500000 + i * 0x8000 + 0x2608) & 0x1f
              for i in range(gpc_nr)]
    tpc_total = sum(tpc_nr)
    print(f"[kepler] grctx_main: gpc_nr={gpc_nr} tpc_total={tpc_total} "
          f"PGRAPH_STATUS={dev.read32(0x400000):#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x}", flush=True)
    # Nouveau programs the mapped VMA, not the physical INST-memory address.
    pagepool_gpu_pa = pagepool.va_addr
    bundle_gpu_pa = bundle_cb.va_addr
    attrib_cb_gpu_pa = attrib_cb.va_addr
    try:
      _gk104_grctx_main(dev, pagepool_gpu_pa, bundle_gpu_pa,
                        attrib_cb_gpu_pa, tpc_nr)
      print(f"[kepler] grctx_main done", flush=True)
    except Exception as e:
      print(f"[kepler] grctx_main ERROR: {type(e).__name__}: {e}", flush=True)
      raise
    # Verify GR is still accessible after grctx_main.
    _gr_post = dev.read32(0x400000)
    print(f"[kepler] post-grctx_main: PGRAPH_STATUS={_gr_post:#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x}", flush=True)
    if _gr_post & 0xffff0000 == 0xbadf0000:
      print(f"[kepler] WARNING: GR inaccessible after grctx_main (false positive: 0xbadf1000 is normal idle)", flush=True)
    # Wait for GR to be truly idle before triggering context switch.
    # ctx_xfer_idle in the FECS firmware waits for 0x409c00 bit 13 to clear,
    # which requires 0x40060c bit 0 (GR busy) to be 0.
    # ponytail: grctx_main MMIO writes can re-trigger the FECS exception
    # (0x409c18) which keeps GR busy.  Clear it again before the golden save.
    _fecs_exc_post = dev.read32(0x409c18)
    _pgraph_intr_post = dev.read32(0x400100)
    if _fecs_exc_post or _pgraph_intr_post:
      print(f"[kepler] Clearing post-grctx exc={_fecs_exc_post:#x} intr={_pgraph_intr_post:#x}", flush=True)
      dev.write32(0x409c20, _fecs_exc_post)
      dev.write32(0x400100, _pgraph_intr_post)
      dev.write32(0x400500, 0x00010001)  # re-enable PGRAPH
      time.sleep(0.01)
    _gr_busy = dev.read32(0x40060c) & 0x1
    if _gr_busy:
      print(f"[kepler] GR busy after grctx_main (0x40060c={dev.read32(0x40060c):#x}), waiting for idle", flush=True)
      _idle_ok = _gk104_gr_wait_idle(dev, timeout_s=2.0)
      print(f"[kepler] GR idle wait: {'OK' if _idle_ok else 'TIMEOUT'} 0x40060c={dev.read32(0x40060c):#x}", flush=True)
    # 4. Save the generated golden context exactly as gf100_grctx_generate().
    #    The non-firmware path (which we use with EFI firmware) clears
    #    CHAN_NEXT.VALID, fakes a CHSW interrupt, and waits for FECS to
    #    unload the current context.  PFIFO will load this saved image when
    #    it schedules the channel; bypassing CHSW leaves GR without engine state.
    # ponytail: The FECS sleeps after ctx_chan and won't wake for FIFO commands
    # because the EFI firmware doesn't enable FIFO interrupts (IREN=0x0).
    # Enable the FIFO interrupt (bit 2) in INTR_EN_SET before sending the
    # CHSW trigger, so the FECS wakes up to process the context switch.
    _iren_pre = dev.read32(0x409010)
    print(f"[kepler] golden save: IREN={_iren_pre:#x} PC={dev.read32(0x409ff0):#x} "
          f"CPUCTL={dev.read32(0x409100):#x} CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x}", flush=True)
    # Enable FIFO interrupt (bit 2) + ctxsw (bit 5) to wake FECS from sleep
    dev.write32(0x409010, 0x00000024)  # INTR_EN_SET: bits 2,5
    # Clear CHAN_NEXT valid bit (gf100_grctx_generate non-firmware path)
    nvkm_mask(dev, 0x409b04, 0x80000000, 0x00000000)
    # Fake CHSW interrupt: set HUB_CHSW_PULSE (bit 8) via INTR_SET
    dev.write32(0x409000, 0x00000100)
    print(f"[kepler] CHSW trigger: INTR={dev.read32(0x409008):#x} "
          f"CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x} "
          f"PC={dev.read32(0x409ff0):#x}", flush=True)
    # Wait for CHAN_ADDR valid bit to clear (FECS unloads the context)
    _deadline_save = time.time() + 2.0
    while time.time() < _deadline_save:
      if not (dev.read32(0x409b00) & 0x80000000):
        break
      time.sleep(0.001)
    else:
      print(f"[kepler] CHSW TIMEOUT: CHAN_ADDR={dev.read32(0x409b00):#x} "
            f"INTR={dev.read32(0x409008):#x} PC={dev.read32(0x409ff0):#x} "
            f"CPUCTL={dev.read32(0x409100):#x} "
            f"0x40060c={dev.read32(0x40060c):#x}", flush=True)
      raise RuntimeError("FECS golden context save (CHSW) timed out")
    # Clear CHAN_ADDR valid bit (gf100_grctx_generate: nvkm_mask 0x409b00)
    nvkm_mask(dev, 0x409b00, 0x80000000, 0x00000000)
    # The saved image begins at CB_RESERVED and its leading words are FECS
    # context metadata.  Do not clear them: Nouveau copies gr->size bytes from
    # this offset verbatim into each runtime engine context.
    if use_vram_inst:
      saved_head = [_gk104_pramin_read32(dev, grctx_vram_pa + 0x80000 + i)
                    for i in range(0, 16, 4)]
    else:
      saved_head = list(struct.unpack_from("<4I", vram, grctx_pa + 0x80000))
    print(f"[kepler] FECS golden save done: CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x} "
          f"IREN={dev.read32(0x409010):#x} head={[hex(x) for x in saved_head]}", flush=True)
    # Stop the golden-save keep-alive thread.
    _ka_stop2.set()
    if _ka_thread2 is not None:
      _ka_thread2.join(timeout=2.5)
      if _ka_thread2.is_alive():
        raise RuntimeError("golden-context keepalive did not stop")
    # Verify RAMFC is intact after golden context save
    if use_vram_inst:
      _ramfc_94 = _gk104_pramin_read32(dev, ramin_vram_pa + 0x94)
      _ramfc_84 = _gk104_pramin_read32(dev, ramin_vram_pa + 0x84)
      _ramfc_9c = _gk104_pramin_read32(dev, ramin_vram_pa + 0x9c)
      _ramfc_ac = _gk104_pramin_read32(dev, ramin_vram_pa + 0xac)
      _ramfc_e8 = _gk104_pramin_read32(dev, ramin_vram_pa + 0xe8)
    else:
      _ramfc_94 = struct.unpack_from('<I', vram, ramin_pa + 0x94)[0]
      _ramfc_84 = struct.unpack_from('<I', vram, ramin_pa + 0x84)[0]
      _ramfc_9c = struct.unpack_from('<I', vram, ramin_pa + 0x9c)[0]
      _ramfc_ac = struct.unpack_from('<I', vram, ramin_pa + 0xac)[0]
      _ramfc_e8 = struct.unpack_from('<I', vram, ramin_pa + 0xe8)[0]
    print(f"[kepler] RAMFC after golden save: 0x84={_ramfc_84:#x} "
          f"0x94={_ramfc_94:#x} 0x9c={_ramfc_9c:#x} 0xac={_ramfc_ac:#x} "
          f"0xe8={_ramfc_e8:#x}", flush=True)
    print(f"[kepler] FECS ready after golden save: SCRATCH0={dev.read32(0x409800):#x} "
          f"CPUCTL={dev.read32(0x409100):#x}", flush=True)
    # ponytail: no cmd 0x39 (start_ctxsw) needed — mmiotrace of nouveau on GK104
    # shows it never sends cmd 0x39.  The FECS firmware starts context switching
    # on its own after the cmd 1 (ctx_chan) channel bind.  The old cmd 0x39 code
    # here was based on source analysis of gf100_gr_fecs_start_ctxsw which is
    # only called on Fermi (GF100-GF110), not Kepler (GK104).
    # FE_PWR FORCE_ON: keep the FE domain powered through the runtime context
    # setup, bind, runlist submission, and launch.  In AUTO mode, FE power-gates
    # during the multi-second runtime context copy/repair, and once gated,
    # FECS registers read 0xbadf1000 and the channel can never dispatch to GR.
    # The FORCE_ON write must happen while FE is still awake (right after the
    # golden save, which is the last FE activity).
    dev.write32(0x404170, 0x00000012)  # NV_PGRAPH_FE_PWR_MODE_FORCE_ON
    wait_cond(lambda: not bool(dev.read32(0x404170) & 0x00000010),
              timeout_ms=2000, msg="FE power FORCE_ON after golden save")
    # Also disable the ctxctl clock-gate (0x000260=0, nouveau nvkm_mc_unk260(0))
    # to keep FECS clocked.  FE_PWR=FORCE_ON controls the FE (front-end) domain
    # but does NOT prevent the FECS falcon itself from clock-gating.  The
    # ctxctl clock-gate at 0x000260 was re-enabled at the end of firmware load
    # (line ~1045), allowing FECS to auto-gate when idle.  During the multi-
    # second runtime context setup, FECS gates and cannot be woken by the
    # channel bind.  Disabling the clock-gate keeps FECS always clocked.
    dev.write32(0x000260, 0x00000000)
    # Disable FECS idle filter to prevent therm-managed auto clock-gating.
    dev.write32(0x020288, 0x00000000)
    dev.write32(0x02028c, 0x00000000)
    # Clear RED_SWITCH ENABLE bits to disable power-gating for ROP/GPC/MAIN.
    # The FECS firmware sets ENABLE_ROP|ENABLE_GPC|ENABLE_MAIN (0x700) during
    # ctx_redswitch.  Clearing these bits prevents the domains from
    # power-gating when idle.  Keep POWER bits (0x70) to keep domains powered.
    _red_switch = dev.read32(0x409614)
    dev.write32(0x409614, _red_switch & ~0x700)  # clear ENABLE bits
    # Ensure 0x020004 has bit30=1 (un-gate) and bit31=0 (disable gating).
    nvkm_mask(dev, 0x020004, 0xc0000000, 0x40000000)
    print(f"[kepler] FE_PWR+ctxctl after golden save: "
          f"FE_PWR={dev.read32(0x404170):#x} "
          f"ctxctl={dev.read32(0x000260):#x} "
          f"FECS_CTRL={dev.read32(0x409100):#x} "
          f"RED_SWITCH={dev.read32(0x409614):#x} "
          f"PWR_GATE={dev.read32(0x020004):#x} "
          f"idle_filter={dev.read32(0x020288):#x}",
          flush=True)
    # Tight poll: sample FECS CPUCTL every 1ms to find exact gate time.
    _gate_samples = []
    for _ in range(50):
      _v = dev.read32(0x409100)
      _gate_samples.append((_v, _))
      if _v == 0xbadf1000 or (_v & 0xfffff000) == 0xbadf0000:
        break
      time.sleep(0.001)
    print(f"[kepler] FECS gate poll after golden save: "
          f"{[(i, hex(v)) for v, i in _gate_samples[:10]]}",
          flush=True)
    # Nouveau keeps the CB_RESERVED allocation only while generating the
    # global golden image.  A real channel owns a gr->size context beginning at
    # offset zero, populated by copying data[CB_RESERVED:CB_RESERVED+size].
    ctx_size = dev.gr_ctx_size
    if not 0 < ctx_size <= 0x80000:
      raise RuntimeError(f"invalid GK104 context size {ctx_size:#x}")
    if use_vram_inst:
      _copy_start = time.time()
      for off in range(0, ctx_size, 0x1000):
        # Keep FECS alive by reading CPUCTL during the copy.
        # FECS power-gates after ~10ms of idle; the copy can take longer.
        dev.read32(0x409100)
        chunk = dev.dev_impl.hw.mmio_read(
          1, grctx_vram_pa + 0x80000 + off, min(0x1000, ctx_size - off))
        bar1_write(grctx_vram_pa + off, chunk)
      _copy_elapsed = time.time() - _copy_start
      _fecs_after_copy = dev.read32(0x409100)
      print(f"[kepler] ctx copy: {_copy_elapsed:.3f}s FECS={_fecs_after_copy:#x}",
            flush=True)
      # Track FECS state through each subsequent step
      _fecs_check_points = []
      _gk104_bar_flush(dev)
      _fecs_check_points.append(("after_bar_flush", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      ctx_mismatches = []
      for off in range(0, ctx_size, 0x1000):
        dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
        size = min(0x1000, ctx_size - off)
        src = dev.dev_impl.hw.mmio_read(1, grctx_vram_pa + 0x80000 + off, size)
        dst = dev.dev_impl.hw.mmio_read(1, grctx_vram_pa + off, size)
        for word in range(0, size, 4):
          if src[word:word + 4] != dst[word:word + 4]:
            ctx_mismatches.append((off + word, src[word:word + 4], dst[word:word + 4]))
      _fecs_check_points.append(("after_verify", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      print(f"[kepler] runtime GR ctx verify: mismatches={len(ctx_mismatches)} "
            f"sample={[(hex(o), s.hex(), d.hex()) for o, s, d in ctx_mismatches[:8]]}",
            flush=True)
      remaining = len(ctx_mismatches)
      for repair_pass in range(4):
        if not ctx_mismatches:
          break
        for off, wanted, _actual in ctx_mismatches:
          dev.read32(0x409100)
          _gk104_pramin_write(dev, grctx_vram_pa + off, wanted)
        _gk104_bar_flush(dev)
        ctx_mismatches = []
        for off in range(0, ctx_size, 0x1000):
          dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
          size = min(0x1000, ctx_size - off)
          src = dev.dev_impl.hw.mmio_read(1, grctx_vram_pa + 0x80000 + off, size)
          dst = dev.dev_impl.hw.mmio_read(1, grctx_vram_pa + off, size)
          for word in range(0, size, 4):
            if src[word:word + 4] != dst[word:word + 4]:
              ctx_mismatches.append((off + word, src[word:word + 4],
                                     dst[word:word + 4]))
        remaining = len(ctx_mismatches)
        print(f"[kepler] runtime GR ctx repair pass={repair_pass + 1} "
              f"remaining={remaining}", flush=True)
      if remaining:
        raise RuntimeError(f"runtime GR context copy remains corrupt ({remaining} dwords)")
      # gf100_gr_chan_bind(): number of address/value pairs and MMIO-list VA/256.
      dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
      _gk104_pramin_write(dev, grctx_vram_pa + 0x80000,
                          struct.pack("<II", len(runtime_mmio_entries),
                                      mmio_list.va_addr >> 8))
      _gk104_bar_flush(dev)
      _fecs_check_points.append(("after_mmio_list_write", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
      runtime_ctx_va = (gr_ctx.va_addr + 0x80000) | 4
      _gk104_pramin_write(dev, ramin_vram_pa + 0x210,
                          struct.pack("<Q", runtime_ctx_va))
      _gk104_bar_flush(dev)
      _fecs_check_points.append(("after_ramin_ctx_write", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
      runtime_head = [_gk104_pramin_read32(dev, grctx_vram_pa + 0x80000 + i)
                      for i in range(0, 16, 4)]
      _fecs_check_points.append(("after_head_read", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      print(f"[kepler] FECS check points: "
            f"{[(n, hex(v), hex(p), hex(g)) for n, v, p, g in _fecs_check_points]}",
            flush=True)
    else:
      vram[grctx_pa:grctx_pa + ctx_size] = \
        vram[grctx_pa + 0x80000:grctx_pa + 0x80000 + ctx_size]
      runtime_ctx_va = (gr_ctx.va_addr + 0x80000) | 4
      struct.pack_into("<Q", vram, ramin_pa + 0x210, runtime_ctx_va)
      runtime_head = list(struct.unpack_from("<4I", vram, grctx_pa))
    print(f"[kepler] runtime GR ctx: size={ctx_size:#x} va={runtime_ctx_va:#x} "
          f"head={[hex(x) for x in runtime_head]}", flush=True)
    dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
    _fecs_check_points2 = [("after_runtime_ctx_print", dev.read32(0x409100))]

  # Final GR accessibility diagnostic before committing the runlist.
  if dev.dev_impl.hw is not None:
    stale_trap = dev.read32(0x400108)
    _fecs_check_points2.append(("after_stale_trap_read", dev.read32(0x409100)))
    stale_gpcs = dev.read32(0x400118)
    for gpc in range(len(_tpc_nr)):
      if not (stale_gpcs & (1 << gpc)):
        continue
      for tpc in range(_tpc_nr[gpc]):
        tb = 0x504000 + gpc * 0x8000 + tpc * 0x800
        stat = dev.read32(tb + 0x508)
        if stat & 0x2:
          dev.write32(tb + 0x648, 0)
          dev.write32(tb + 0x650, dev.read32(tb + 0x650))
        if stat:
          dev.write32(0x500000 + gpc * 0x8000 + 0x2c90,
                      0x00010000 << tpc)
      dev.write32(0x400118, 1 << gpc)
    _fecs_check_points2.append(("after_gpc_clear", dev.read32(0x409100)))
    if stale_trap:
      dev.write32(0x400108, stale_trap)
    stale_intr = dev.read32(0x400100)
    if stale_intr:
      dev.write32(0x400100, stale_intr)
    dev.write32(0x400500, 0x00010001)
    _fecs_check_points2.append(("after_pgraph_ctrl", dev.read32(0x409100)))
    print(f"[kepler] cleared stale GR traps: trap={stale_trap:#x} "
          f"gpcs={stale_gpcs:#x} intr={stale_intr:#x} -> "
          f"{dev.read32(0x400108):#x}/{dev.read32(0x400118):#x}/"
          f"{dev.read32(0x400100):#x}", flush=True)
    dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
    _gr_final = dev.read32(0x400000)
    _gpc_topo = dev.read32(0x409604)
    dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
    _fecs_ctrl_pre = dev.read32(0x409100)
    _fecs_check_points2.append(("after_pre_runlist_read", dev.read32(0x409100)))
    print(f"[kepler] pre-runlist GR: PGRAPH_STATUS={_gr_final:#x} "
          f"GPC_TOPOLOGY={_gpc_topo:#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x} "
          f"FECS_CTRL={_fecs_ctrl_pre:#x}", flush=True)
    # Start a background FECS keep-alive thread before snapshot.
    # Continuously re-assert FE_PWR FORCE_ON to prevent the FE domain
    # from power-gating during/after the context switch.
    _fecs_keepalive_stop = threading.Event()
    _ka_thread = None
    def _fecs_keepalive():
      while not _fecs_keepalive_stop.is_set():
        try:
          # Simulate the PMU's response to the ctx_4170s/ctx_4170w handshake.
          # The FECS firmware sets bit 4 (0x10) of 0x404170 to request a power
          # state change, then spins in ctx_4170w waiting for bit 4 to clear.
          # Without a PMU, bit 4 never clears and the FE domain power-gates.
          # We clear bit 4 and keep FORCE_ON (bit 1) set.
          _pwr = dev.read32(0x404170)
          if _pwr & 0x10:
            # FECS requested a power state change — acknowledge it.
            dev.write32(0x404170, (_pwr & ~0x10) | 0x02)
          elif _pwr != 0x02:
            dev.write32(0x404170, 0x00000002)  # FE_PWR FORCE_ON only
        except Exception:
          pass
        time.sleep(0.0001)  # 0.1ms — fast enough to catch the handshake
    if single_thread_rpc:
      print("[kepler] strict single-thread RPC mode: FECS helper disabled", flush=True)
    else:
      _ka_thread = threading.Thread(target=_fecs_keepalive, daemon=True)
      _ka_thread.start()
      print("[kepler] FECS keep-alive thread started (FE_PWR FORCE_ON)", flush=True)
    # Phase 1 capture point 1: before runlist submission.
    # Skip snapshot_gr_traps for now — it reads GPC/TPC registers which
    # may trigger FECS power-gating.  Test if FECS stays alive without it.
    # snapshot_gr_traps(dev, "before_runlist")
    dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
    _fecs_check_points2.append(("after_snapshot", dev.read32(0x409100)))
    print(f"[kepler] FECS check points2: "
          f"{[(n, hex(v)) for n, v in _fecs_check_points2]}", flush=True)

  # 0x2200 programming belongs to the older GF100 path, not gk104_fifo_init().
  # Retain it solely for explicit register experiments.
  _pfifo_ctrl = dev.read32(0x2200)
  if DEBUG:
    print(f"[kepler] PFIFO_CTRL before enable: 0x{_pfifo_ctrl:08x}", flush=True)
  if unsafe_experiments:
    nvkm_mask(dev, 0x2200, 0x00000001, 0x00000001)
    nvkm_mask(dev, 0x2200, 0x00000100, 0x00000100)
  if DEBUG:
    print(f"[kepler] PFIFO_CTRL after enable: 0x{dev.read32(0x2200):08x}", flush=True)
  # Verify USERD data is readable through PRAMIN (the PBDMA's access path)
  if dev.dev_impl.hw is not None and use_vram_inst:
    _userd_gp_get = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, userd_vram_pa + userd_base_off + USERD_GP_GET, 4))[0]
    _userd_gp_put = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, userd_vram_pa + userd_base_off + USERD_GP_PUT, 4))[0]
    if DEBUG:
      print(f"[kepler] USERD[{chan_id}] at pa=0x{userd_vram_pa + userd_base_off:x}: "
            f"GP_GET={_userd_gp_get} GP_PUT={_userd_gp_put}", flush=True)

  # Bind + start the channel (nouveau gk104_chan_bind_inst / gk104_chan_start).
  # FE_PWR was set to FORCE_ON after the golden save and should still be on.
  # Verify it here; if FE gated during setup, report it.
  if dev.dev_impl.hw is not None:
    _fe_pwr_pre_bind = dev.read32(0x404170)
    if _fe_pwr_pre_bind & 0x2 == 0:
      # Re-assert FORCE_ON if it was lost (e.g. by a power-gating transition).
      dev.write32(0x404170, 0x00000012)
      wait_cond(lambda: not bool(dev.read32(0x404170) & 0x00000010),
                timeout_ms=2000, msg="FE power FORCE_ON re-assert before bind")
      _fe_pwr_pre_bind = dev.read32(0x404170)
    # Tight diagnostic: sample FECS CPUCTL every 1ms for 20ms to see exactly
    # when it gates.  This helps identify which operation triggers the gate.
    dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
    _fecs_samples = []
    for _ in range(20):
      _v = dev.read32(0x409100)
      if not _fecs_samples or _fecs_samples[-1][1] != _v:
        _fecs_samples.append((_, _v))
      dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
      time.sleep(0.001)
    if any(v == 0xbadf1000 for _, v in _fecs_samples):
      print(f"[kepler] FECS gate trace: {[(t, hex(v)) for t, v in _fecs_samples]}",
            flush=True)
    # Check if clkgate settings survived
    _clkgate = dev.read32(0x20200)
    _pwr_gate = dev.read32(0x020004)
    print(f"[kepler] pre-bind power: 0x20200={_clkgate:#010x} "
          f"0x020004={_pwr_gate:#010x} "
          f"FE_PWR={dev.read32(0x404170):#x} "
          f"ctxctl={dev.read32(0x000260):#x}", flush=True)
    # Disable GPC/TPC power-gating: nouveau gk104_pmu_pgob(enable=false)
    # sets 0x020004 bits[31:30] = 01 (bit30=1 un-gate, bit31=0 disable gating).
    # Previously tried 0x00000000 (both bits clear) but that didn't prevent
    # GPC power-gating during kernel execution.
    nvkm_mask(dev, 0x020004, 0xc0000000, 0x40000000)
    _pwr_gate2 = dev.read32(0x020004)
    print(f"[kepler] PWR_GATE cleared: 0x020004={_pwr_gate2:#010x}", flush=True)
    # FECS is power-gated (0xbadf1000).  In nouveau, the PMU manages FECS
    # power state.  Without a PMU, FECS power-gates when idle and can't be
    # woken by simple register writes.  The pgob un-gates the GR engine but
    # destroys FECS (PC=0).  Solution: run pgob to un-gate, then reload FECS
    # firmware so it can process channel context switches.
    _fecs_ctrl = dev.read32(0x409100)
    print(f"[kepler] FECS before bind: CPUCTL={_fecs_ctrl:#x} "
          f"(gated={'yes' if _fecs_ctrl == 0xbadf1000 else 'no'})", flush=True)
    if _fecs_ctrl == 0xbadf1000 or (_fecs_ctrl & 0xfffff000) == 0xbadf0000:
      print("[kepler] FECS power-gated, running pgob + FECS reload...", flush=True)
      gk104_pmu_pgob(dev, war00c800=False)
      dev.write32(0x400500, 0x00010001)  # PGRAPH master enable
      _reload_fecs_after_pgob(dev)
      _fecs_ctrl = dev.read32(0x409100)
    print(f"[kepler] after un-gate attempt: FECS_CTRL={_fecs_ctrl:#x} "
          f"PGRAPH_STATUS={dev.read32(0x400000):#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x} "
          f"FE_PWR={dev.read32(0x404170):#x} "
          f"PWR_GATE={dev.read32(0x020004):#x}", flush=True)
    # Check PGRAPH sub-domains after pgob re-un-gate
    _subdom_pgob = []
    for _addr in (0x400700, 0x404000, 0x404200):
      _v = dev.read32(_addr)
      _ok = (_v & 0xfffff000) != 0xbadf0000
      _subdom_pgob.append(f"0x{_addr:x}={_v:#x}{'(OK)' if _ok else '(GATED)'}")
    print(f"[kepler] PGRAPH sub-domains after pgob: {' '.join(_subdom_pgob)}",
          flush=True)
    # Try to wake FECS up by writing to SCRATCH0 and polling CPUCTL.
    # If FECS is power-gated, writing to a FECS register should request
    # a power-up.  Poll for up to 100ms.
    _fecs_ctrl = dev.read32(0x409100)
    if _fecs_ctrl == 0xbadf1000:
      print("[kepler] FECS still gated after pgob!", flush=True)
    print(f"[kepler] FE_PWR before bind: {dev.read32(0x404170):#x} "
          f"PGRAPH_STATUS={dev.read32(0x400000):#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x} "
          f"FECS_CTRL={dev.read32(0x409100):#x}", flush=True)
  # gk104_chan_bind_inst() is exactly VALID | (inst_addr >> 12).  Unlike the
  # runlist descriptor, this register has no TARGET field on GK104.  BAR1 is
  # merely the CPU mapping used to populate the VRAM instance object; it does
  # not turn the PFIFO instance pointer into a HUB-MMU virtual address.
  if use_vram_inst:
    _live_pdes = [(vmm_pgd_pa + _idx * 8,
                   (1 << 32) | (_dst_spt << 24))
                  for _idx, _src_spt, _dst_spt in cloned_spts]
    _live_pte_map = {}
    # Preserve and stabilise every populated source mapping (inputs, output,
    # code, constants, CWD, and allocator metadata).  Bulk cloning alone is
    # insufficient on this path because individual PTE dwords can settle to an
    # all-ones entry without producing an immediate MMU fault.
    for _pgdi, _src_spt, _dst_spt in cloned_spts:
      if _src_spt < 0:
        continue
      for _spti in range(0x8000):
        _pte = struct.unpack_from("<Q", vram, _src_spt + _spti * 8)[0]
        if _pte:
          _live_pte_map[_dst_spt + _spti * 8] = _pte
    for _buf, _buf_pa, _priv in ((pagepool, pagepool_vram_pa, True),
                                 (bundle_cb, bundle_vram_pa, True),
                                 (attrib_cb, attrib_vram_pa, False)):
      for _page in range(round_up(_buf.size, 0x1000) // 0x1000):
        _va = _buf.va_addr + _page * 0x1000
        _pgdi, _spti = (_va >> 27) & 0x1fff, (_va >> 12) & 0x7fff
        _pte = ((_buf_pa + _page * 0x1000) >> 8) | 1 | (2 if _priv else 0)
        _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = _pte
    for _page in range(gr_ctx.size // 0x1000):
      _va = gr_ctx.va_addr + _page * 0x1000
      _pgdi, _spti = (_va >> 27) & 0x1fff, (_va >> 12) & 0x7fff
      _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = \
          ((grctx_vram_pa + _page * 0x1000) >> 8) | 3
    for _mirror in getattr(dev, "_kepler_vram_mirrors", ()):
      _mirror_pa = _mirror.meta.get("vram_pa")
      if _mirror_pa is None:
        continue
      for _page in range(round_up(_mirror.size, 0x1000) // 0x1000):
        _va = _mirror.va_addr + _page * 0x1000
        _pgdi, _spti = (_va >> 27) & 0x1fff, (_va >> 12) & 0x7fff
        _pte = ((_mirror_pa + _page * 0x1000) >> 8) | 1
        if _mirror.meta.get("priv"):
          _pte |= 2
        _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = _pte
    if mmio_list.meta.get("vram_pa") is None:
      _va = mmio_list.va_addr
      _pgdi, _spti = (_va >> 27) & 0x1fff, (_va >> 12) & 0x7fff
      _src_spt = mm.root_page_table.address(_pgdi)
      _src_pte = struct.unpack_from("<Q", vram, _src_spt + _spti * 8)[0]
      _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = _src_pte
    for _pte_addr, _pte in ctx_alias_ptes:
      _live_pte_map[_pte_addr] = _pte
    _signal_buf = HCQBuffer(signal_va, 0x1000, meta={"pa": signal_vram_pa})
    for _buf, _buf_pa in ((gpfifo, gpfifo_vram_pa),
                          (push, push_vram_pa),
                          (_signal_buf, signal_vram_pa)):
      if _buf_pa is None:
        continue
      for _page in range(_buf.size // 0x1000):
        _va = _buf.va_addr + _page * 0x1000
        _pgdi, _spti = (_va >> 27) & 0x1fff, (_va >> 12) & 0x7fff
        _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = \
            ((_buf_pa + _page * 0x1000) >> 8) | 1
    _live_ptes = list(_live_pte_map.items())
    for _attempt in range(4):
      for _entry_addr, _entry_wanted in (*_live_pdes, *_live_ptes):
        _gk104_pramin_write(dev, _entry_addr, struct.pack("<Q", _entry_wanted))
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for channel PTEs did not complete")
      time.sleep(0.005)
      _live_entries = [*_live_pdes, *_live_ptes]
      _pte_actual = [(_gk104_pramin_read32(dev, addr) |
                      (_gk104_pramin_read32(dev, addr + 4) << 32))
                     for addr, _ in _live_entries]
      if _pte_actual == [wanted for _, wanted in _live_entries]:
        break
    else:
      raise RuntimeError(f"GK104 channel PTEs did not stabilise: {_pte_actual}")
    if _live_ptes and not _gk104_vmm_flush_pdb(dev, vmm_pgd_pa, target=0):
      raise TimeoutError("GK104 channel VMM flush did not complete")
    if signal_vram_pa is not None:
      _signal_wanted = struct.pack("<I", signal_initial)
      for _attempt in range(4):
        bar1_write(signal_vram_pa, _signal_wanted)
        _gk104_pramin_write(dev, signal_vram_pa, _signal_wanted)
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate for signal did not complete")
        time.sleep(0.005)
        _signal_bar = struct.unpack(
            "<I", dev.dev_impl.hw.mmio_read(1, signal_vram_pa, 4))[0]
        _signal_pramin = _gk104_pramin_read32(dev, signal_vram_pa)
        if _signal_bar == _signal_pramin == signal_initial:
          break
      else:
        raise RuntimeError(
            f"GK104 signal did not stabilise: bar={_signal_bar:#x} "
            f"pramin={_signal_pramin:#x}")
    for _mirror in getattr(dev, "_kepler_vram_mirrors", ()):
      _mirror_pa = _mirror.meta.get("vram_pa")
      if _mirror_pa is None:
        continue
      _mirror_wanted = bytes(vram[_mirror.meta["pa"]:
                                  _mirror.meta["pa"] + _mirror.size])
      if _mirror is mmio_list:
        _mmio_encoded = b"".join(
            struct.pack("<I", (~struct.unpack_from("<I", _mirror_wanted, off)[0]) &
                        0xffffffff)
            for off in range(0, len(_mirror_wanted), 4))
        _gk104_pramin_write_literal(dev, _mirror_pa, _mmio_encoded)
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate for MMIO list failed")
        continue
      for _attempt in range(4):
        bar1_write(_mirror_pa, _mirror_wanted)
        _gk104_pramin_write(dev, _mirror_pa, _mirror_wanted)
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate for VRAM mirror failed")
        time.sleep(0.005)
        _mirror_bar = dev.dev_impl.hw.mmio_read(
            1, _mirror_pa, len(_mirror_wanted))
        _mirror_pramin = b"".join(
            struct.pack("<I", _gk104_pramin_read32(dev, _mirror_pa + off))
            for off in range(0, len(_mirror_wanted), 4))
        if _mirror_bar == _mirror_pramin == _mirror_wanted:
          break
      else:
        raise RuntimeError(
            f"GK104 VRAM mirror did not stabilise: va={_mirror.va_addr:#x}")
    _ramin_expected = bytearray(inst[:0x220])
    struct.pack_into("<Q", _ramin_expected, 0x210, runtime_ctx_va)
    for _attempt in range(4):
      _gk104_pramin_write(dev, ramin_vram_pa, _ramin_expected)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for RAMFC did not complete")
      time.sleep(0.005)
      _ramin_actual = b"".join(
          struct.pack("<I", _gk104_pramin_read32(dev, ramin_vram_pa + off))
          for off in range(0, len(_ramin_expected), 4))
      if _ramin_actual == bytes(_ramin_expected):
        break
    else:
      _bad = [off for off in range(0, len(_ramin_expected), 4)
              if _ramin_actual[off:off + 4] != _ramin_expected[off:off + 4]]
      raise RuntimeError(f"GK104 RAMFC did not stabilise: offsets={_bad[:16]}")
    ramin_bind_addr = ramin_vram_pa
  else:
    raise RuntimeError("GK104 live channels require a VRAM instance block")
  _chan_ctrl_reg = CHAN_START_REG + chan_id * 8
  _chan_inst_reg = CHAN_SUBMIT_REG + chan_id * 8
  runlist_addr = runlist_pa if use_vram_runlist else (base + runlist_pa)
  runlist_target = 0 if use_vram_runlist else 3

  def _empty_gr_runlist():
    """Remove every channel from GR runlist 0 and wait for scheduler DMA."""
    dev.write32(0x000140, 0x00000000)
    dev.write32(0x000640, 0x00000000)
    dev.read32(0x000140)
    pending_reg = 0x2284 + GR_RUNLIST_ID * 8
    # Prevent a stale entry from being selected while the replacement list is
    # being committed.  A zero count is Nouveau's normal empty-runlist form.
    nvkm_mask(dev, 0x2630, 1 << GR_RUNLIST_ID, 1 << GR_RUNLIST_ID)
    deadline = time.time() + 0.1
    while dev.read32(pending_reg) & 0x00100000 and time.time() < deadline:
      time.sleep(0.001)
    dev.write32(0x2270, (runlist_target << 28) | (runlist_addr >> 12))
    dev.write32(PFIFO_RUNLIST_SUBMIT, GR_RUNLIST_ID << 20)
    deadline = time.time() + 0.2
    while dev.read32(pending_reg) & 0x00100000:
      if time.time() >= deadline:
        raise TimeoutError("GK104 empty-runlist update did not complete")
      time.sleep(0.001)
    intr = dev.read32(0x2a00)
    if intr:
      dev.write32(0x2a00, intr)

  # ponytail: PBDMA channel bind and runlist commit already done before
  # ctx_chan (above).  Skip the duplicate bind here; just verify the channel
  # is still bound and not in an error state.
  if DEBUG:
    print(f"[kepler] chan bind (already done): ctrl=0x{dev.read32(_chan_ctrl_reg):08x} "
          f"inst=0x{dev.read32(_chan_inst_reg):08x}", flush=True)
  # USERD and inst pointer already written in the early PBDMA bind above.
  # Start the channel BEFORE the runlist commit.  Nouveau's sequence is:
  # gk104_chan_bind → gk104_chan_start (nvkm_chan_allow) → nvkm_chan_insert
  # (runlist commit).  The ENABLE_TRIGGER must be set before the scheduler
  # processes the runlist entry, otherwise the channel is skipped.
  _teardown_done = False
  def _quiesce_channel(reason="teardown"):
    """Stop the host helper thread without touching the endpoint."""
    nonlocal _teardown_done
    if _teardown_done:
      return
    errors = []
    def _attempt(label, op):
      _set_native_thread_name(f"kgpu:{label}")
      try:
        return op()
      except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return None

    _fecs_keepalive_stop.set()
    # Do not let the helper retain the socket while the client is closing.
    if _ka_thread is not None:
      _ka_thread.join(timeout=2.5)
      if _ka_thread.is_alive():
        errors.append("FECS keepalive did not stop before client close")
    # Do not issue BAR, config-space, FLR, or server-lifecycle operations here.
    # The working examples/add.py transport has no such close protocol.  The
    # latest panic caught this thread in recv_into only after keepalive joined,
    # leaving the former PCI_COMMAND read/write/read as the live rejected RPC.
    _teardown_done = True
    _set_native_thread_name("kgpu:teardown done")
    print(f"[kepler] host helper stopped ({reason}): zero endpoint teardown RPC",
          flush=True)
    for error in errors:
      print(f"[kepler] WARNING: teardown step failed ({reason}): {error}",
            file=sys.stderr, flush=True)

  dev._kepler_emergency_teardown = _quiesce_channel
  atexit.register(_quiesce_channel, "atexit")
  nvkm_mask(dev, _chan_ctrl_reg, 0x400, 0x400)  # start (ENABLE_TRIGGER)
  if DEBUG:
    print(f"[kepler] chan bind step4 start: ctrl=0x{dev.read32(_chan_ctrl_reg):08x}", flush=True)
    print(f"[kepler] channel bind={dev.read32(_chan_inst_reg):#x} "
          f"ctrl={dev.read32(_chan_ctrl_reg):#x} "
          f"PGRAPH_STATUS={dev.read32(0x400000):#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x}", flush=True)
  # FECS state after channel bind
  _fecs_ctrl_post_bind = dev.read32(0x409100)
  _subch_post_bind = dev.read32(0x404200)
  # Check for SCHED_ERROR right after channel bind (before runlist commit)
  _post_bind_intr = dev.read32(0x2100)
  if _post_bind_intr & 0x100:
    _post_bind_sched = dev.read32(0x254c)
    print(f"[kepler] SCHED_ERROR after bind (before runlist): "
          f"INTR=0x{_post_bind_intr:08x} SCHED=0x{_post_bind_sched:08x}", flush=True)
    # ponytail: Properly clear SCHED_ERROR: mask interrupt, ACK, re-enable
    # (like tu102_fifo_intr_sched_ctxsw).  Just writing 0x2100 doesn't
    # prevent re-assertion; the scheduler keeps the error latched.
    nvkm_mask(dev, 0x2140, 0x00000100, 0x00000000)  # mask SCHED_ERROR intr
    dev.write32(0x2100, 0x00000100)                  # ACK the interrupt
    dev.write32(0x254c, 0x00000001)                  # clear scheduler error
    nvkm_mask(dev, 0x2140, 0x00000100, 0x00000100)  # re-enable intr
  print(f"[kepler] post-bind FECS: CPUCTL={_fecs_ctrl_post_bind:#x} "
        f"SCRATCH0={dev.read32(0x409800):#x} "
        f"CHAN_ADDR={dev.read32(0x409b00):#x} "
        f"CHAN_NEXT={dev.read32(0x409b04):#x} "
        f"subch4={_subch_post_bind:#x}", flush=True)
  # Pre-set RED_SWITCH to all-enabled state before the CHSW.
  # After the golden save, RED_SWITCH=0x70 (powered but not enabled).
  # The FECS ctx_redswitch() expects domains to be in a safe state;
  # without ENABLE bits set, the toggle can crash the FECS.
  if dev.dev_impl.hw is not None:
    _rs = dev.read32(0x409614)
    if _rs != 0x770:
      dev.write32(0x409614, 0x770)
      print(f"[kepler] RED_SWITCH pre-set: 0x{_rs:x} -> 0x770", flush=True)
  # Context switching starts automatically when the FECS firmware processes
  # the cmd 1 (ctx_chan) channel bind.  No explicit cmd 0x39 is needed on GK104
  # (confirmed via mmiotrace of nouveau).  The hardware will perform a proper
  # context switch when PBDMA forwards the first methods to GR.  Do NOT pre-load via cmd 1 here — that
  # prevents the proper CHSW from firing.
  # Make the channel visible to its runq before advancing USERD GP_PUT.
  # Diagnostic: read TOP device list (0x22700+) to verify GR engine is
  # enumerated and associated with the correct runlist.  GK104's PBDMA
  # needs the TOP device list to know which engine to dispatch methods to.
  _top_regs = []
  for _ti in range(64):
    _tv = dev.read32(0x22700 + _ti * 4)
    if _tv:
      _top_regs.append((_ti, _tv))
  print(f"[kepler] TOP device list: {len(_top_regs)} non-zero entries", flush=True)
  for _ti, _tv in _top_regs:
    _kind = _tv & 3
    _kind_name = {0: "NOT_VALID", 1: "DATA", 2: "ENUM", 3: "ENGINE_TYPE"}.get(_kind, "?")
    print(f"  TOP[{_ti:02x}]: 0x{_tv:08x} ({_kind_name})", flush=True)
  # Also read PBDMA engine assignment registers
  _pbdma_eng = [dev.read32(0x400a4 + _i * 0x40) for _i in range(3)]
  print(f"[kepler] PBDMA_ENGINES pre-runlist: "
        f"{[hex(_e) for _e in _pbdma_eng]}", flush=True)
  # Check PBDMA enable and engine assignment registers
  _pbdma_enable_lo = dev.read32(0x000204)
  _pbdma_enable_hi = dev.read32(0x002204)
  _eng_assign = [dev.read32(0x002208 + _i * 4) for _i in range(6)]
  print(f"[kepler] PBDMA enable: lo=0x{_pbdma_enable_lo:08x} hi=0x{_pbdma_enable_hi:08x} "
        f"eng_assign={[hex(_e) for _e in _eng_assign]}", flush=True)
  # Check PBDMA interrupts and status
  for _pi in range(3):
    _pbdma_intr_0_stat = dev.read32(0x40108 + _pi * 0x2000)
    _pbdma_intr_0_mask = dev.read32(0x4010c + _pi * 0x2000)
    _pbdma_intr_1_stat = dev.read32(0x40148 + _pi * 0x2000)
    _pbdma_ctrl = dev.read32(0x4013c + _pi * 0x2000)
    _pbdma_chid = dev.read32(0x40120 + _pi * 0x2000)
    print(f"[kepler] PBDMA{_pi}: INTR0=0x{_pbdma_intr_0_stat:08x} INTR0_MASK=0x{_pbdma_intr_0_mask:08x} "
          f"INTR1=0x{_pbdma_intr_1_stat:08x} CTRL=0x{_pbdma_ctrl:08x} CHID=0x{_pbdma_chid:08x}", flush=True)
  # Check USERD BAR1 address register
  _userd_bar1 = dev.read32(0x2254)
  print(f"[kepler] USERD BAR1 reg 0x2254=0x{_userd_bar1:08x}", flush=True)
  # Check CHAN_TABLE for channel 0 (at PFIFO offset 0x1000, stride 8)
  _chan_table_chan = dev.read32(0x3000 + chan_id * 8)
  _chan_table_state = dev.read32(0x3004 + chan_id * 8)
  print(f"[kepler] CHAN_TABLE[{chan_id}]: CHAN=0x{_chan_table_chan:08x} "
        f"STATE=0x{_chan_table_state:08x} "
        f"(RUNNABLE={bool(_chan_table_state & 1)} LOADED={bool(_chan_table_state & 0x1000)})", flush=True)
  # Check runlist allow/block register
  _runl_block = dev.read32(0x2630)
  print(f"[kepler] runlist block/allow 0x2630=0x{_runl_block:08x}", flush=True)
  # ponytail: Diagnostics for runlist DMA failure
  print(f"[kepler] PFIFO state: CTRL=0x{dev.read32(0x2200):08x} "
        f"SUBFIFO=0x{dev.read32(0x000204):08x} "
        f"INTR=0x{dev.read32(0x2100):08x} "
        f"RUNLIST_BASE=0x{dev.read32(0x2270):08x} "
        f"RUNLISTSubmit=0x{dev.read32(0x2274):08x} "
        f"pending=0x{dev.read32(0x2284 + GR_RUNLIST_ID * 8):08x}", flush=True)
  # Verify GR ctx pointer in instance block is still valid before runlist commit
  if dev.dev_impl.hw is not None and use_vram_inst:
    _inst_gr_lo = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, ramin_vram_pa + 0x210, 4))[0]
    _inst_gr_hi = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, ramin_vram_pa + 0x214, 4))[0]
    _inst_gr_val = (_inst_gr_hi << 32) | _inst_gr_lo
    print(f"[kepler] inst GR ctx ptr: 0x{_inst_gr_val:010x} valid={bool(_inst_gr_val & 4)} "
          f"va=0x{(_inst_gr_val & ~4):x}", flush=True)
    # Also check other engine ctx ptrs for conflicts
    for _eng_name, _eng_off in [("SEC", 0x220), ("MSPDEC", 0x250), ("MSPPP", 0x260),
                                 ("MSVLD", 0x270), ("VIC", 0x280), ("MSENC", 0x290)]:
      _e_lo = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, ramin_vram_pa + _eng_off, 4))[0]
      if _e_lo:
        print(f"[kepler] inst {_eng_name} ctx ptr at 0x{_eng_off:x}: lo=0x{_e_lo:08x} (CONFLICT!)", flush=True)
  if DEBUG and use_vram_runlist:
    print(f"[kepler] runlist_precommit="
          f"{struct.unpack('<II', dev.dev_impl.hw.mmio_read(1, runlist_pa, 8))}")
  if DEBUG:
    print(f"[kepler] runlist_addr={runlist_addr:#x} target={runlist_target} "
          f"use_vram_runlist={use_vram_runlist} runlist_pa={runlist_pa:#x} "
          f"base={base:#x}", flush=True)
    if not use_vram_runlist:
      _rl_off = runlist_pa
      _rl_words = struct.unpack_from('<II', vram, _rl_off)
      print(f"[kepler] runlist buffer @vram[{_rl_off:#x}]: "
            f"words={_rl_words}", flush=True)
  # Match nvkm_runl_update_locked(): clear the pending runlist fault and
  # unblock scheduler processing before submitting the new list.
  # Nouveau inserts an idle channel first; userspace advances GP_PUT only
  # after its ring and pushbuffer stores are complete.  Publishing PUT before
  # the runlist races PBDMA against the delayed framebuffer writes.
  # Late GP_PUT is the validated compute ordering: admit and load the channel
  # context first, then expose the already-stabilised entry to PBDMA.  The old
  # early-PUT ordering remains available only as an explicit diagnostic.
  put_before_runlist = os.environ.get("KEPLER_PUT_BEFORE_RUNLIST", "0") != "0"
  if put_before_runlist:
    if use_vram_inst:
      _gk104_pramin_write(dev, _userd_mmio_base + USERD_GP_PUT,
                          struct.pack("<I", 1))
    else:
      dev.dev_impl.hw.mmio_write(1, _userd_mmio_base + USERD_GP_PUT,
                                 struct.pack("<I", 1))
    _gk104_bar_flush(dev)
  dev.write32(0x262c, 1 << GR_RUNLIST_ID)
  nvkm_mask(dev, 0x2630, 1 << GR_RUNLIST_ID, 0)
  # Check engine state BEFORE runlist commit to see if engine is already faulted
  _engn_stat_pre = dev.read32(0x2640 + 0 * 8)  # engn 0 = GR
  _sched_stat_pre = dev.read32(0x263c)
  _sched_err_pre = dev.read32(0x254c)
  _userd_bar1_pre = dev.read32(0x2254)
  if DEBUG:
    print(f"[kepler] pre-runlist engn0=0x{_engn_stat_pre:08x} "
          f"sched=0x{_sched_stat_pre:08x} sched_err=0x{_sched_err_pre:08x} "
          f"userd_bar1=0x{_userd_bar1_pre:08x} "
          f"BUSY={bool(_engn_stat_pre&0x80000000)} "
          f"FAULTED={bool(_engn_stat_pre&0x40000000)}",
          flush=True)
  # If engine is faulted, try to clear it by writing to the fault clear register
  # and also clearing any PGRAPH interrupts
  if _engn_stat_pre & 0x40000000:
    if DEBUG:
      print("[kepler] engine faulted pre-runlist — attempting clear", flush=True)
    # Clear all runlist faults
    dev.write32(0x262c, 0xffffffff)
    # Clear PGRAPH interrupts
    _pgraph_intr = dev.read32(0x400100)
    if _pgraph_intr:
      dev.write32(0x400100, _pgraph_intr)
    # Clear PFIFO interrupts
    _pfifo_intr = dev.read32(0x2100)
    if _pfifo_intr:
      dev.write32(0x2100, _pfifo_intr)
    time.sleep(0.001)
    _engn_stat_post_clear = dev.read32(0x2640)
    if DEBUG:
      print(f"[kepler] after fault clear: engn0=0x{_engn_stat_post_clear:08x} "
            f"FAULTED={bool(_engn_stat_post_clear&0x40000000)}",
            flush=True)
  # Unfreeze the scheduler (FREEZE at 0x2638).  On a properly VBIOS-initialized
  # GPU this is already 0, but on an un-POSTed card it may be undefined.
  if unsafe_experiments:
    dev.write32(0x2638, 0)
  if DEBUG:
    _freeze_after = dev.read32(0x2638)
    print(f"[kepler] FREEZE after write 0: 0x{_freeze_after:08x}", flush=True)
  # Acknowledge stale MMU faults through VM_FAULT_SOURCE.  The 0x2800 records
  # are payload registers, not W1C status registers (gf100_fifo_intr_mmu_fault).
  _stale_fault_source = dev.read32(0x259c)
  if _stale_fault_source:
    dev.write32(0x259c, _stale_fault_source)
  # Clear PFIFO interrupts
  dev.write32(0x2100, 0xffffffff)
  # All CPU BAR1 preparation is complete.  Enable the BAR1 VMM only now:
  # direct BAR writes alias VRAM while the VMM is enabled, whereas PFIFO
  # requires the VMM for USERD polling.  Subsequent stores use PRAMIN.
  if use_vram_inst and os.environ.get("KEPLER_INIT_BAR1", "1") != "0":
    _gk104_init_bar1_identity(dev, bus_base=base, map_vram=True,
                              userd_alias_pa=userd_vram_pa)
    # gk104_fifo_init() runs after BAR1 init in Nouveau.  Re-latch the USERD
    # polling VMA now that 0x1704 points at the live BAR1 page directory.
    dev.write32(0x2254, 0x10000000 | (userd_vram_pa >> 12))
    _userd_alias_put = struct.unpack(
        "<I", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
    _userd_phys_put = _gk104_pramin_read32(
        dev, userd_vram_pa + userd_base_off + USERD_GP_PUT)
    _expected_put = 1 if put_before_runlist else 0
    if (_userd_alias_put != _userd_phys_put or
        _userd_alias_put != _expected_put):
      raise RuntimeError(
          f"GK104 BAR1 USERD alias mismatch before runlist: "
          f"alias={_userd_alias_put:#x} physical={_userd_phys_put:#x}")
    if DEBUG:
      print(f"[kepler] BAR1 USERD alias stable: GP_PUT={_userd_alias_put}",
            flush=True)
  if use_vram_runlist:
    # PRAMIN stores can acknowledge with the desired immediate readback and
    # settle to the XOR-inverted value later.  Stabilise the tiny runlist only
    # after every other framebuffer write, and verify after flush/invalidate.
    runlist_entry = struct.pack("<II", chan_id, 0)
    for _attempt in range(4):
      _gk104_pramin_write(dev, runlist_pa, runlist_entry)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for runlist did not complete")
      time.sleep(0.005)
      _runlist_actual = (_gk104_pramin_read32(dev, runlist_pa),
                         _gk104_pramin_read32(dev, runlist_pa + 4))
      if _runlist_actual == (chan_id, 0):
        break
    else:
      raise RuntimeError(f"GK104 runlist store did not stabilise: {_runlist_actual}")
  if use_vram_inst and os.environ.get("KEPLER_INIT_BAR1", "1") != "0":
    # Recheck after the last PRAMIN transaction and a delayed settle.  Repair
    # the live roots and invalidate HUB translations if the alias drifted.
    for _attempt in range(4):
      time.sleep(0.005)
      _userd_alias_put = struct.unpack(
          "<I", dev.dev_impl.hw.mmio_read(
              1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
      _userd_phys_put = _gk104_pramin_read32(
          dev, userd_vram_pa + userd_base_off + USERD_GP_PUT)
      _expected_put = 1 if put_before_runlist else 0
      if _userd_alias_put == _userd_phys_put == _expected_put:
        break
      _gk104_pramin_write(dev, 0x00100000 + 0x200,
                          struct.pack("<Q", 0x00110000))
      _gk104_pramin_write(
          dev, 0x00110000,
          struct.pack("<Q", (1 << 32) | (0x00120000 << 24)))
      for _page in range(2):
        _pa = userd_vram_pa + _page * 0x1000
        _gk104_pramin_write(
            dev, 0x00120000 + _page * 8,
            struct.pack("<Q", (_pa >> 8) | 1))
        _userd_page = userd_vram_pa // 0x1000 + _page
        _gk104_pramin_write(
            dev, 0x00120000 + _userd_page * 8,
            struct.pack("<Q", (_pa >> 8) | 1))
      _gk104_pramin_write(
          dev, userd_vram_pa + userd_base_off + USERD_GP_PUT,
          struct.pack("<I", _expected_put))
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for USERD alias did not complete")
      _gk104_vmm_flush_pdb(dev, 0x00110000, target=0, hub_only=True)
      dev.write32(0x001704, 0x80000000 | (0x00100000 >> 12))
      dev.write32(0x2254, 0x10000000 | (userd_vram_pa >> 12))
    else:
      raise RuntimeError(
          f"GK104 BAR1 USERD alias did not remain stable: "
          f"alias={_userd_alias_put:#x} physical={_userd_phys_put:#x}")
    print(f"[kepler] BAR1 USERD delayed check: GP_PUT={_userd_alias_put}",
          flush=True)
  if put_before_runlist and use_vram_inst:
    # GP_PUT was published while the channel was still absent from the
    # scheduler.  Repair and verify the GPU-visible command pages after that
    # doorbell, then make the channel runnable with the runlist commit below.
    _literal_ring = os.environ.get("KEPLER_GPU_LITERAL_RING") == "1"
    for _attempt in range(4):
      if gpfifo_vram_pa is not None:
        if _literal_ring:
          _gk104_pramin_write_literal(dev, gpfifo_vram_pa, ring)
        else:
          bar1_write(gpfifo_vram_pa, ring_store)
          _gk104_pramin_write(dev, gpfifo_vram_pa, ring_store)
      if push_vram_pa is not None:
        bar1_write(push_vram_pa + push_phys_offset, push_bytes)
        _gk104_pramin_write(dev, push_vram_pa + push_phys_offset, push_bytes)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for precommit commands failed")
      time.sleep(0.005)
      if _literal_ring:
        # Raw literal stores intentionally bypass the transformed CPU
        # readback; PBDMA is the authoritative consumer for this diagnostic.
        break
      _ring_actual = (dev.dev_impl.hw.mmio_read(1, gpfifo_vram_pa, len(ring_store))
                      if gpfifo_vram_pa is not None else ring_store)
      _push_actual = (dev.dev_impl.hw.mmio_read(
          1, push_vram_pa + push_phys_offset, len(push_bytes))
          if push_vram_pa is not None else bytes(push_bytes))
      _ring_pramin = (b"".join(struct.pack("<I", _gk104_pramin_read32(
          dev, gpfifo_vram_pa + off)) for off in range(0, len(ring_store), 4))
          if gpfifo_vram_pa is not None else ring_store)
      _push_pramin = (b"".join(struct.pack("<I", _gk104_pramin_read32(
          dev, push_vram_pa + push_phys_offset + off))
          for off in range(0, len(push_bytes), 4))
          if push_vram_pa is not None else bytes(push_bytes))
      if (_ring_actual == _ring_pramin == ring_store and
          _push_actual == _push_pramin == bytes(push_bytes)):
        break
    else:
      raise RuntimeError("GK104 precommit command buffers did not stabilise")
  if DEBUG:
    print(f"[kepler] stale MMU faults after clear: source={dev.read32(0x259c):#x}",
          flush=True)
  # Re-assert the runlist ID in the channel control register right before
  # the runlist commit.  The hardware clears bits [19:16] when the inst
  # pointer is written, but the scheduler needs the runlist ID to match
  # the runlist being committed.
  nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x000f0000, GR_RUNLIST_ID << 16)
  if DEBUG:
    print(f"[kepler] pre-runlist chan_ctrl=0x{dev.read32(CHAN_START_REG + chan_id * 8):08x}", flush=True)
  dev.write32(0x2270, (runlist_target << 28) | (runlist_addr >> 12))
  if hw is not None:
    hw.set_phase("runlist-submit")
  dev.write32(PFIFO_RUNLIST_SUBMIT, (GR_RUNLIST_ID << 20) | 1)
  if DEBUG and use_vram_runlist:
    print(f"[kepler] runlist_vram pa={runlist_pa:#x} words="
          f"{struct.unpack('<II', dev.dev_impl.hw.mmio_read(1, runlist_pa, 8))} "
          f"runq_masks={[hex(dev.read32(0x2390 + i * 4)) for i in range(3)]}")
  if DEBUG:
    print(f"[kepler] runlist submit: id={GR_RUNLIST_ID} "
          f"0x2270=0x{dev.read32(0x2270):08x} "
          f"0x2274=0x{dev.read32(0x2274):08x}", flush=True)
  # gk104_runl_pending() is the per-runlist bit at 0x2284 + runl_id * 8;
  # nv50_runl_wait() waits for it to clear before a channel kick can rely
  # on the new list.
  _runl_pending_reg = 0x2284 + GR_RUNLIST_ID * 8
  if DEBUG:
    print(f"[kepler] runlist pending reg=0x{_runl_pending_reg:x} "
          f"val=0x{dev.read32(_runl_pending_reg):08x}", flush=True)
  deadline = time.time() + 0.2
  while dev.read32(_runl_pending_reg) & 0x00100000:
    if time.time() >= deadline:
      raise TimeoutError("GK104 runlist update did not complete")
    time.sleep(0.001)
  # gk104_fifo_intr_runlist(): committing the list raises the per-runlist
  # completion interrupt at 0x2a00.  Nouveau acknowledges that source before
  # letting normal PFIFO dispatch continue.  Leaving it asserted keeps the
  # PFIFO master RUNLIST bit high even though 0x2284 says the DMA completed.
  runlist_intr = dev.read32(0x2a00)
  if DEBUG:
    print(f"[kepler] runlist intr 0x2a00=0x{runlist_intr:08x}", flush=True)
  if runlist_intr:
    dev.write32(0x2a00, runlist_intr)
  # The un-POSTed card needs the channel's start trigger re-issued after the
  # runlist DMA is visible.  This is the single masked SET used by the older
  # working path (which reached IB_GET=1); clearing the trigger first is not
  # part of gk104_chan_start() and can halt an already-enabled channel.
  # Keep this independent of the unsafe manual scheduler/dispatch fallbacks.
  if os.environ.get("KEPLER_POST_RUNLIST_START", "1") != "0":
    nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x00000400, 0x00000400)
  if DEBUG:
    print(f"[kepler] post-runlist chan_ctrl=0x{dev.read32(CHAN_START_REG + chan_id * 8):08x}", flush=True)
  # Give the scheduler time to process the runlist entries and populate
  # the CHAN_TABLE.  The DMA completion (PLAYLIST_RD BUSY=0) doesn't mean
  # the scheduler has finished processing the entries.
  time.sleep(0.01)
  if DEBUG:
    _ct_chan = dev.read32(0x3000 + chan_id * 8)
    _ct_state = dev.read32(0x3004 + chan_id * 8)
    print(f"[kepler] CHAN_TABLE after delay: CHAN=0x{_ct_chan:08x} STATE=0x{_ct_state:08x}", flush=True)
  # Kick the PBDMA to start processing the channel.  KICK_CHID (0x2634)
  # tells the scheduler to immediately dispatch the specified channel.
  if unsafe_experiments:
    dev.write32(0x2634, chan_id)
  time.sleep(0.01)
  if DEBUG:
    _ct_chan2 = dev.read32(0x3000 + chan_id * 8)
    _ct_state2 = dev.read32(0x3004 + chan_id * 8)
    print(f"[kepler] CHAN_TABLE after kick: CHAN=0x{_ct_chan2:08x} STATE=0x{_ct_state2:08x}", flush=True)
  # On GK104, the hardware scheduler should populate CHAN_TABLE automatically
  # from the runlist.  But if the scheduler is not functional (un-POSTed card
  # VBIOS PFIFO init), manually write the CHAN_TABLE like GF100 does:
  # gf100_chan_bind: 0x3000 + id*8 = 0xc0000000 | (inst_addr >> 12)
  # gf100_chan_start: 0x3004 + id*8 = 0x001f0001
  _chan_table_chan_reg = 0x3000 + chan_id * 8
  _chan_table_state_reg = 0x3004 + chan_id * 8
  if unsafe_experiments and dev.read32(_chan_table_chan_reg) == 0:
    dev.write32(_chan_table_chan_reg, 0xc0000000 | (ramin_bind_addr >> 12))
    dev.write32(_chan_table_state_reg, 0x001f0001)
    if DEBUG:
      print(f"[kepler] manual CHAN_TABLE write: "
            f"CHAN=0x{dev.read32(_chan_table_chan_reg):08x} "
            f"STATE=0x{dev.read32(_chan_table_state_reg):08x}", flush=True)
  # Check PFIFO and PBDMA state after runlist commit
  if dev.dev_impl.hw is not None:
    _pfifo_intr = dev.read32(0x2100)
    _chan_table_err = dev.read32(0x252c)
    _sched_err = dev.read32(0x254c) if _pfifo_intr & 0x100 else 0
    print(f"[kepler] CHAN_TABLE_ERROR=0x{_chan_table_err:08x}", flush=True)
    _bind_err = dev.read32(0x252c) if _pfifo_intr & 0x1 else 0
    # Also check SCHED_ERROR right after channel bind (before runlist commit)
    _pre_runlist_intr = dev.read32(0x2100)
    _pre_sched_err = dev.read32(0x254c) if _pre_runlist_intr & 0x100 else 0
    _chsw_stat = dev.read32(0x2630)
    _runq_masks = [dev.read32(0x2390 + i * 4) for i in range(3)]
    _pbdma_ch = [dev.read32(0x40120 + i * 0x2000) for i in range(3)]
    _pbdma_engines = [dev.read32(0x400a4 + i * 0x2000) for i in range(3)]
    _engn_status = [dev.read32(0x2640 + i * 8) for i in range(8)]
    _chan_ctrl = dev.read32(CHAN_START_REG + chan_id * 8)
    _chan_inst = dev.read32(CHAN_SUBMIT_REG + chan_id * 8)
    print(f"[kepler] PFIFO after runlist: INTR=0x{_pfifo_intr:08x} "
          f"SCHED_ERR=0x{_sched_err:02x} BIND_ERR=0x{_bind_err:02x} "
          f"CHSW=0x{_chsw_stat:08x} runq_masks={[hex(x) for x in _runq_masks]} "
          f"pbdma_ch={[hex(x) for x in _pbdma_ch]} "
          f"pbdma_engines={[hex(x) for x in _pbdma_engines]}", flush=True)
    # Read SCHED_ERROR and SCHED_STATUS directly (not gated on INTR)
    _sched_err_direct = dev.read32(0x254c)
    _sched_status_direct = dev.read32(0x263c)
    _engn0_direct = dev.read32(0x2640)
    print(f"[kepler] SCHED direct: ERR=0x{_sched_err_direct:08x} "
          f"STATUS=0x{_sched_status_direct:08x} ENGN0=0x{_engn0_direct:08x}",
          flush=True)
    print(f"[kepler] engn_status={[hex(x) for x in _engn_status if x]} "
          f"chan_ctrl=0x{_chan_ctrl:08x} chan_inst=0x{_chan_inst:08x}", flush=True)
    # Check CHAN_TABLE after runlist commit
    _post_chan_table_chan = dev.read32(0x3000 + chan_id * 8)
    _post_chan_table_state = dev.read32(0x3004 + chan_id * 8)
    print(f"[kepler] CHAN_TABLE[{chan_id}] after runlist: CHAN=0x{_post_chan_table_chan:08x} "
          f"STATE=0x{_post_chan_table_state:08x} "
          f"(RUNNABLE={bool(_post_chan_table_state & 1)} LOADED={bool(_post_chan_table_state & 0x1000)})", flush=True)
    # Check PFIFO_CHAN state (GK104 uses 0x800000, not the GF100 CHAN_TABLE at 0x3000)
    _pfifo_chan = dev.read32(0x800000 + chan_id * 8)
    _pfifo_chan_state = dev.read32(0x800004 + chan_id * 8)
    print(f"[kepler] PFIFO_CHAN[{chan_id}]: CHAN=0x{_pfifo_chan:08x} "
          f"STATE=0x{_pfifo_chan_state:08x} "
          f"(ENABLED={bool(_pfifo_chan_state & 1)} "
          f"ENABLE_TRIGGER={bool(_pfifo_chan_state & 0x400)} "
          f"ENGINE={(_pfifo_chan_state >> 16) & 0xf} "
          f"UNK24_RO={(_pfifo_chan_state >> 24) & 0x7} "
          f"UNK28_RO={bool(_pfifo_chan_state & 0x10000000)})",
          flush=True)
    # Check MMU fault registers (0x2800 + unit * 0x10, 8 units on GK104)
    _mmu_faults = []
    _fault_source = dev.read32(0x259c)
    for _unit in range(8):
      if not (_fault_source & (1 << _unit)):
        continue
      _fault_inst = dev.read32(0x2800 + _unit * 0x10)
      _fault_valo = dev.read32(0x2804 + _unit * 0x10)
      _fault_vahi = dev.read32(0x2808 + _unit * 0x10)
      _fault_type = dev.read32(0x280c + _unit * 0x10)
      if _fault_inst or _fault_type:
        _mmu_faults.append((_unit, _fault_inst, _fault_valo, _fault_vahi, _fault_type))
    if _mmu_faults:
      for _unit, _inst, _valo, _vahi, _type in _mmu_faults:
        _fault_addr = (_vahi << 32) | _valo
        print(f"[kepler] MMU_FAULT unit={_unit} inst=0x{_inst:08x} "
              f"addr=0x{_fault_addr:010x} "
              f"type=0x{_type:08x} reason={_type & 0xf} "
              f"client={(_type >> 8) & 0x1f:#x} "
              f"hub={bool(_type & 0x40)} write={bool(_type & 0x80)}",
              flush=True)
    else:
      print("[kepler] MMU_FAULT: none", flush=True)
    # Check USERD after runlist commit (through BAR1)
    if dev.dev_impl.hw is not None and use_vram_inst:
      _post_userd_gp_get = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, userd_vram_pa + userd_base_off + USERD_GP_GET, 4))[0]
      _post_userd_gp_put = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, userd_vram_pa + userd_base_off + USERD_GP_PUT, 4))[0]
      print(f"[kepler] USERD[{chan_id}] after runlist: GP_GET={_post_userd_gp_get} GP_PUT={_post_userd_gp_put}", flush=True)
    # Check PLAYLIST_RD for the GR runlist.
    _playlist_rd = dev.read32(0x2280 + GR_RUNLIST_ID * 8)
    _playlist_rd_len = dev.read32(0x2284 + GR_RUNLIST_ID * 8)
    print(f"[kepler] PLAYLIST_RD[{GR_RUNLIST_ID}]: addr=0x{_playlist_rd:08x} "
          f"len=0x{_playlist_rd_len:08x} (LEN={_playlist_rd_len & 0xfff} "
          f"BUSY={bool(_playlist_rd_len & 0x100000)})", flush=True)
    # Check the PBDMA whose hardware runlist mask contains the GR runlist.
    _gr_pbdma = next((i for i, mask in enumerate(_runq_masks)
                      if mask & (1 << GR_RUNLIST_ID)), 0)
    _p1_base = 0x40000 + _gr_pbdma * 0x2000
    _p1_ib_put = dev.read32(_p1_base + 0x00)
    _p1_ib_get = dev.read32(_p1_base + 0x14)
    _p1_dma_get = dev.read32(_p1_base + 0x18)
    _p1_ctrl = dev.read32(_p1_base + 0x40)
    _p1_chid = dev.read32(_p1_base + 0x120)
    _p1_eng = dev.read32(_p1_base + 0xa4)
    print(f"[kepler] PBDMA{_gr_pbdma} detail: IB_PUT=0x{_p1_ib_put:08x} IB_GET=0x{_p1_ib_get:08x} "
          f"DMA_GET=0x{_p1_dma_get:08x} CTRL=0x{_p1_ctrl:08x} CHID=0x{_p1_chid:08x} "
          f"ENG=0x{_p1_eng:08x}", flush=True)
    # Check ENGINE_CHANNEL_INST for engine 0 (PGRAPH) at 0x2680
    _eng_chan_inst = [dev.read32(0x2680 + _i * 4) for _i in range(8)]
    print(f"[kepler] ENGINE_CHANNEL_INST={[hex(x) for x in _eng_chan_inst]}", flush=True)
    # Check PBDMA_STATUS at 0x26c0
    _pbdma_status = dev.read32(0x26c0)
    print(f"[kepler] PBDMA_STATUS=0x{_pbdma_status:08x}", flush=True)
    # Check FREEZE at 0x2638
    _freeze = dev.read32(0x2638)
    print(f"[kepler] FREEZE=0x{_freeze:08x}", flush=True)
    # Verify instance pointer and read USERD addr from instance block
    _chan_inst_val = dev.read32(CHAN_SUBMIT_REG + chan_id * 8)
    _inst_pa = (_chan_inst_val & 0x0fffffff) << 12
    print(f"[kepler] chan_inst=0x{_chan_inst_val:08x} inst_pa=0x{_inst_pa:08x}", flush=True)
    # Check PBDMA idle status (0x3080 + pbdma_id * 4, bits 13-15)
    _pbdma_idle = [dev.read32(0x3080 + _i * 4) for _i in range(3)]
    print(f"[kepler] PBDMA idle status: {[hex(x) for x in _pbdma_idle]} "
          f"(idle={[not (x & 0xe000) for x in _pbdma_idle]})", flush=True)
    # Check UNK1080 (PBDMA status) more carefully
    for _pi in range(3):
      _val = dev.read32(0x3080 + _pi * 4)
      print(f"[kepler] PBDMA{_pi} status 0x{0x3080+_pi*4:x}=0x{_val:08x} "
            f"bits13-15={( _val >> 13) & 7}", flush=True)
    # Check SCHED_STATUS and ENG_STATE
    _sched_status = dev.read32(0x263c)
    print(f"[kepler] SCHED_STATUS=0x{_sched_status:08x}", flush=True)
    # Check BYPASS registers
    _bypass_config = dev.read32(0x26c4)
    _bypass_status = dev.read32(0x5000)
    print(f"[kepler] BYPASS_CONFIG=0x{_bypass_config:08x} "
          f"BYPASS_STATUS=0x{_bypass_status:08x}", flush=True)
    for _ei in range(7):
      _es = dev.read32(0x2640 + _ei * 4)
      if _es:
        print(f"[kepler] ENG_STATE[{_ei}]=0x{_es:08x}", flush=True)
    # Dump runlist entry
    if use_vram_runlist:
      _rl = struct.unpack('<II', dev.dev_impl.hw.mmio_read(1, runlist_pa, 8))
    else:
      _rl = struct.unpack_from('<II', vram, runlist_pa)
    print(f"[kepler] runlist entry: chan_id={_rl[0]:#x} word1={_rl[1]:#x} "
          f"(expected chan_id={chan_id:#x})", flush=True)
  # FECS state after runlist commit
  _fecs_ctrl_post_rl = dev.read32(0x409100)
  _subch_post_rl = dev.read32(0x404200)
  print(f"[kepler] post-runlist FECS: CPUCTL={_fecs_ctrl_post_rl:#x} "
        f"SCRATCH0={dev.read32(0x409800):#x} "
        f"CHAN_ADDR={dev.read32(0x409b00):#x} "
        f"CHAN_NEXT={dev.read32(0x409b04):#x} "
        f"subch4={_subch_post_rl:#x}", flush=True)
  # Manually trigger a context switch to load the channel context.
  # On GK104, the PBDMA should automatically generate a CHSW interrupt when
  # it schedules a new channel, but on this un-POSTed card without a PMU, the PBDMA
  # forwards methods without waiting for the FECS to load the context.
  # Manually set CHAN_NEXT with bit 31 (new channel to load) and trigger
  # the CHSW interrupt (bit 8 of INTR_SET at 0x409000), exactly like the
  # golden ctx unload does in reverse.
  _chan_next_pre = dev.read32(0x409b04)
  _chan_addr_pre = dev.read32(0x409b00)
  print(f"[kepler] manual CHSW: CHAN_ADDR={_chan_addr_pre:#x} "
        f"CHAN_NEXT={_chan_next_pre:#x} -> setting CHAN_NEXT bit31",
        flush=True)
  # Set CHAN_NEXT with bit 31 to indicate a new channel to load
  if unsafe_experiments:
    dev.write32(0x409b04, _chan_next_pre | 0x80000000)
    # Trigger the CHSW interrupt (bit 8 of INTR_SET)
    dev.write32(0x409000, 0x00000100)
  # Wait for the FECS to process the context switch
  _chsw_start = time.time()
  _chsw_done = False
  for _ in range(2000 if unsafe_experiments else 0):
    _ca = dev.read32(0x409b00)
    _cn = dev.read32(0x409b04)
    _scratch0 = dev.read32(0x409800)
    # Context switch is done when CHAN_ADDR gets bit 31 (context loaded)
    # and CHAN_NEXT loses bit 31 (FECS consumed the switch request)
    if _ca & 0x80000000:
      _chsw_done = True
      break
    if (_ca & 0xffff0000) == 0xbadf0000:
      print(f"[kepler] manual CHSW: FECS power-gated during switch!",
            flush=True)
      break
    time.sleep(0.001)
  _chsw_elapsed = time.time() - _chsw_start
  print(f"[kepler] manual CHSW result: done={_chsw_done} "
        f"elapsed={_chsw_elapsed:.3f}s "
        f"CHAN_ADDR={dev.read32(0x409b00):#x} "
        f"CHAN_NEXT={dev.read32(0x409b04):#x} "
        f"SCRATCH0={dev.read32(0x409800):#x} "
        f"PC={dev.read32(0x409ff0):#x}", flush=True)
  # Read CTXCTL ENGINE_STATUS (0x409c00) to understand context switch state.
  # Bits: 0=CHSW_PENDING, 1=CHAN_VALID, 3=CHSW_PULSE,
  #        7=DAEMON2CTXCTL_REQ, 8=DAEMON2CTXCTL_ACK,
  #        9=CTXCTL2DAEMON_REQ, 10=CTXCTL2DAEMON_ACK,
  #        13=IDLE_BUSY, 15=PAUSE_BUSY
  _ctxctl_eng_stat = dev.read32(0x409c00)
  _ctxctl_eng_trig = dev.read32(0x409c08)
  print(f"[kepler] CTXCTL ENGINE_STATUS=0x{_ctxctl_eng_stat:08x} "
        f"TRIGGER=0x{_ctxctl_eng_trig:08x} "
        f"CHSW_PEND={bool(_ctxctl_eng_stat&1)} CHAN_VALID={bool(_ctxctl_eng_stat&2)} "
        f"D2C_REQ={bool(_ctxctl_eng_stat&0x80)} D2C_ACK={bool(_ctxctl_eng_stat&0x100)} "
        f"C2D_REQ={bool(_ctxctl_eng_stat&0x200)} C2D_ACK={bool(_ctxctl_eng_stat&0x400)} "
        f"IDLE_BUSY={bool(_ctxctl_eng_stat&0x2000)} PAUSE_BUSY={bool(_ctxctl_eng_stat&0x8000)}",
        flush=True)
  # Call gf100_gr_fecs_bind_pointer: send WRCMD_CMD=0x03 (BIND_POINTER) with
  # the channel instance address.  This tells PGRAPH CTXCTL which context to
  # use, setting CHAN_VALID.  Without this, PGRAPH CTXCTL has no valid channel
  # and methods sit in the FIFO unprocessed.
  # gf100_gr_fecs_bind_pointer(gr, 0x80000000 | addr) where addr = inst->addr >> 12
  _bind_inst = 0x80000000 | ((ramin_bind_addr >> 12) & 0x0FFFFFFF)
  # Check FECS interrupt/falcon state before sending WRCMD
  _fecs_cpuctl = dev.read32(0x409100)
  _fecs_intr = dev.read32(0x409008)
  _fecs_intr_en = dev.read32(0x409018)
  _fecs_iren = dev.read32(0x409010)
  print(f"[kepler] FECS pre-bind: CPUCTL=0x{_fecs_cpuctl:08x} "
        f"INTR=0x{_fecs_intr:08x} INTR_EN=0x{_fecs_intr_en:08x} "
        f"IREN=0x{_fecs_iren:08x} "
        f"SLEEPING={bool(_fecs_cpuctl&0x20)} HALT={bool(_fecs_cpuctl&0x2)}",
        flush=True)
  # If FECS is sleeping, ensure FIFO_DATA interrupt (bit 2) is enabled
  # and trigger it to wake the falcon
  if unsafe_experiments and _fecs_cpuctl & 0x20:  # SLEEPING
    # Enable FIFO_DATA interrupt (bit 2) in INTR_EN_SET
    dev.write32(0x409010, 0xffffffff)  # IREN: enable all external interrupts
    dev.write32(0x409010 + 0x4, 0xff)  # INTR_EN_SET: enable all falcon interrupts
    # Actually use INTR_EN_SET at 0x409010 + 4 = 0x409014
    # Wait, the registers are: 0x010 = INTR_EN_SET, 0x014 = INTR_EN_CLR, 0x018 = INTR_EN
    # Let me re-check: INTR_EN_SET is at offset 0x010
    dev.write32(0x409010, 0xff)  # INTR_EN_SET: enable all interrupts
    time.sleep(0.001)
    _fecs_intr_en2 = dev.read32(0x409018)
    print(f"[kepler] FECS wake: INTR_EN=0x{_fecs_intr_en2:08x}", flush=True)
  if unsafe_experiments:
    nvkm_mask(dev, 0x409800, 0x00000030, 0x00000000)
    dev.write32(0x409500, _bind_inst)
    dev.write32(0x409504, 0x00000003)  # BIND_POINTER
  _bp_done = False
  _bp_err = False
  for _ in range(2000 if unsafe_experiments else 0):
    _scratch0 = dev.read32(0x409800)
    if _scratch0 & 0x00000020:  # error
      _bp_err = True
      break
    if _scratch0 & 0x00000010:  # done
      _bp_done = True
      break
    time.sleep(0.001)
  _ctxctl_eng_stat2 = dev.read32(0x409c00)
  print(f"[kepler] FECS bind_pointer: done={_bp_done} err={_bp_err} "
        f"inst=0x{_bind_inst:08x} SCRATCH0=0x{dev.read32(0x409800):08x} "
        f"CTXCTL_STATUS=0x{_ctxctl_eng_stat2:08x} "
        f"CHAN_VALID={bool(_ctxctl_eng_stat2&2)}",
        flush=True)
  # If bind_pointer failed, try the non-firmware ctx_chan path
  if unsafe_experiments and not _bp_done:
    print("[kepler] bind_pointer failed — trying non-firmware ctx_chan path", flush=True)
    # Clear SCRATCH0 bit 31 (init done) via CC_SCRATCH_CLEAR
    dev.write32(0x409840, 0x80000000)
    time.sleep(0.001)
    # Send ctx_chan command (0x01) with instance address
    dev.write32(0x409500, _bind_inst)
    dev.write32(0x409504, 0x00000001)  # ctx_chan
    for _ in range(2000):
      _scratch0 = dev.read32(0x409800)
      if _scratch0 & 0x80000000:
        break
      time.sleep(0.001)
    _ctxctl_eng_stat3 = dev.read32(0x409c00)
    print(f"[kepler] ctx_chan path: SCRATCH0=0x{dev.read32(0x409800):08x} "
          f"CTXCTL_STATUS=0x{_ctxctl_eng_stat3:08x} "
          f"CHAN_VALID={bool(_ctxctl_eng_stat3&2)}",
          flush=True)
  # Rapid poll of FECS state to catch the CHSW processing
  if dev.dev_impl.hw is not None:
    _poll_start = time.time()
    _poll_data = []
    for _i in range(100):
      _ctrl = dev.read32(0x409100)
      _pc = dev.read32(0x409ff0)
      _ca = dev.read32(0x409b00)
      _cn = dev.read32(0x409b04)
      _poll_data.append((_ctrl, _pc, _ca, _cn))
      if _ctrl == 0x0 or (_ctrl & 0xffff0000) == 0xbadf0000:
        break
      time.sleep(0.001)
    _poll_elapsed = time.time() - _poll_start
    print(f"[kepler] FECS CHSW poll ({_poll_elapsed:.3f}s, {_i+1} samples):", flush=True)
    for _j, (_c, _p, _a, _n) in enumerate(_poll_data[:20]):
      print(f"  [{_j:3d}] CPUCTL=0x{_c:08x} PC=0x{_p:08x} "
            f"CHAN_ADDR=0x{_a:08x} CHAN_NEXT=0x{_n:08x}", flush=True)
  # Advance USERD GP_PUT past the written entry.
  if DEBUG and use_vram_inst:
    late_ring = (dev.dev_impl.hw.mmio_read(1, gpfifo_vram_pa, 8).hex()
                 if gpfifo_vram_pa is not None else "host")
    late_push = (dev.dev_impl.hw.mmio_read(1, push_vram_pa + push_phys_offset,
                                          len(push_bytes)).hex()
                 if push_vram_pa is not None else "host")
    late_signal = (dev.dev_impl.hw.mmio_read(1, signal_vram_pa, 4).hex()
                   if signal_vram_pa is not None else "host")
    late_userd = dev.dev_impl.hw.mmio_read(
      1, _userd_mmio_base + USERD_GP_GET, 8).hex()
    print(f"[kepler] pre-kick VRAM ring={late_ring} push={late_push} "
          f"signal={late_signal} userd_get_put={late_userd}", flush=True)
  if use_vram_inst:
    # Re-store command data only after the channel is resident.  This avoids a
    # zero GP entry prefetched while the freshly-created RAMFC still had PUT=0.
    for _attempt in range(4):
      if gpfifo_vram_pa is not None:
        bar1_write(gpfifo_vram_pa, ring_store)
        _gk104_pramin_write(dev, gpfifo_vram_pa, ring_store)
      if push_vram_pa is not None:
        bar1_write(push_vram_pa + push_phys_offset, push_bytes)
        _gk104_pramin_write(dev, push_vram_pa + push_phys_offset, push_bytes)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate did not complete")
      time.sleep(0.005)
      _ring_actual = (dev.dev_impl.hw.mmio_read(
          1, gpfifo_vram_pa, len(ring_store))
          if gpfifo_vram_pa is not None else ring_store)
      _push_actual = (dev.dev_impl.hw.mmio_read(
          1, push_vram_pa + push_phys_offset, len(push_bytes))
          if push_vram_pa is not None else bytes(push_bytes))
      _ring_pramin = (b"".join(struct.pack("<I", _gk104_pramin_read32(
          dev, gpfifo_vram_pa + off)) for off in range(0, len(ring_store), 4))
          if gpfifo_vram_pa is not None else ring_store)
      _push_pramin = (b"".join(struct.pack("<I", _gk104_pramin_read32(
          dev, push_vram_pa + push_phys_offset + off))
          for off in range(0, len(push_bytes), 4))
          if push_vram_pa is not None else bytes(push_bytes))
      if (_ring_actual == _ring_pramin == ring_store and
          _push_actual == _push_pramin == bytes(push_bytes)):
        break
    else:
      raise RuntimeError("GK104 command buffers did not stabilise before GP_PUT")
  # Fail closed before making any command visible to the PBDMA.  A fresh MMU
  # or HCE fault means the instance/VMM context is not safe for GPU DMA.
  _prelaunch_faults = []
  _prelaunch_fault_source = dev.read32(0x259c)
  for _unit in range(8):
    if not (_prelaunch_fault_source & (1 << _unit)):
      continue
    _inst = dev.read32(0x2800 + _unit * 0x10)
    _type = dev.read32(0x280c + _unit * 0x10)
    if _inst or _type:
      _prelaunch_faults.append((_unit, _inst, _type))
  _prelaunch_hce = []
  for _pbdma in range(3):
    _hce = dev.read32(0x40148 + _pbdma * 0x2000)
    if _hce:
      _prelaunch_hce.append((_pbdma, _hce))
  if _prelaunch_faults or _prelaunch_hce:
    _probe_va = 0x67000
    _probe_pgdi, _probe_spti = (_probe_va >> 27) & 0x1fff, (_probe_va >> 12) & 0x7fff
    _probe_pte_addr = cloned_by_pgd[_probe_pgdi] + _probe_spti * 8
    _probe_pte = (_gk104_pramin_read32(dev, _probe_pte_addr) |
                  (_gk104_pramin_read32(dev, _probe_pte_addr + 4) << 32))
    print(f"[kepler] faulted GR buffer regs: "
          f"bundle={dev.read32(0x408004):#x}/{dev.read32(0x418808):#x} "
          f"pagepool={dev.read32(0x40800c):#x}/{dev.read32(0x419004):#x} "
          f"pte[0x67000]={_probe_pte:#x}",
          flush=True)
    _quiesce_channel("prelaunch fault")
    raise RuntimeError(
      f"unsafe GK104 channel state before GP_PUT: "
      f"mmu_faults={_prelaunch_faults} hce={_prelaunch_hce}")
  if not put_before_runlist and submit_mode == "gpfifo":
    # The un-POSTed card's saved context restores complemented scratch-buffer
    # bases.  With PUT still zero the channel is resident and idle here, so
    # load it with the documented internal-FECS ctx_chan command, then replay
    # Nouveau's per-channel GR patch list before exposing any methods.
    dev.write32(0x409840, 0x80000000)
    dev.write32(0x409500, 0x80000000 | (ramin_bind_addr >> 12))
    dev.write32(0x409504, 0x00000001)
    wait_cond(lambda: bool(dev.read32(0x409800) & 0x80000000),
              timeout_ms=2000, msg="FECS runtime ctx_chan")
    print(f"[kepler] runtime ctx_chan: CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x}", flush=True)
    for _reg, _value in runtime_mmio_entries:
      dev.write32(_reg, _value & 0xffffffff)
    _gk104_gr_wait_idle(dev)
    print(f"[kepler] late GR patch: bundle={dev.read32(0x408004):#x}/"
          f"{dev.read32(0x418808):#x} pagepool={dev.read32(0x40800c):#x}/"
          f"{dev.read32(0x419004):#x}", flush=True)
    if use_vram_inst:
      # BAR1 readback is CPU-visible but, on this cold-attached card, does not
      # prove that PFIFO observes the store.  Mirror the doorbell through
      # PRAMIN and invalidate LTC before waiting for PBDMA.
      for _attempt in range(4):
        # This is the normal userspace notification path: USERD is mapped in
        # BAR1 precisely so GP_PUT can be written through the CPU aperture.
        dev.dev_impl.hw.mmio_write(
            1, _userd_mmio_base + USERD_GP_PUT, struct.pack("<I", 1))
        _gk104_pramin_write(dev, _userd_mmio_base + USERD_GP_PUT,
                            struct.pack("<I", 1))
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate after GP_PUT did not complete")
        time.sleep(0.005)
        _put_phys = _gk104_pramin_read32(
            dev, _userd_mmio_base + USERD_GP_PUT)
        _put_bar = struct.unpack(
            "<I", dev.dev_impl.hw.mmio_read(
                1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
        if _put_phys == _put_bar == 1:
          break
      else:
        raise RuntimeError(
            f"GK104 GP_PUT did not stabilise: physical={_put_phys:#x} "
            f"bar1={_put_bar:#x}")
      nvkm_mask(dev, _chan_ctrl_reg, 0x00000400, 0x00000400)
    else:
      dev.dev_impl.hw.mmio_write(1, _userd_mmio_base + USERD_GP_PUT,
                                 struct.pack("<I", 1))
      _gk104_bar_flush(dev)
  userd_put_readback = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
  # Submission paths are selected before launch.  Never fall through from the
  # normal GPFIFO path into BYPASS or direct DISPATCH in the same invocation.
  _ct_check = dev.read32(0x3000 + chan_id * 8)
  if submit_mode == "bypass" and dev.dev_impl.hw is not None:
    if DEBUG:
      print("[kepler] CHAN_TABLE empty — using BYPASS path", flush=True)
    # Enable BYPASS for PGRAPH (engine 0).
    # The FECS context switch has already been done (manual CHSW above),
    # so the GR context and page table are already loaded.  Set the channel
    # instance pointer in BYPASS_CHANNEL_SWITCH for the bypass context.
    dev.write32(0x26c4, 0x00000100)  # ENGINE=0, ENABLE=1 (bit 8)
    # Do NOT write BYPASS_CHANNEL_SWITCH — the manual CHSW already loaded
    # the GR context.  Writing CHSW triggers a FECS context switch that
    # gets stuck (CTXCTL busy forever).  Instead, just enable BYPASS and
    # PGRAPH FIFO PULL to submit methods directly.
    time.sleep(0.01)
    _bypass_status = dev.read32(0x5000)
    if DEBUG:
      _fifo_ctrl = dev.read32(0x400500)
      _fifo_stat = dev.read32(0x400504)
      print(f"[kepler] BYPASS: CONFIG=0x{dev.read32(0x26c4):08x} "
            f"STATUS=0x{_bypass_status:08x} "
            f"FIFO_CTRL=0x{_fifo_ctrl:08x} FIFO_STAT=0x{_fifo_stat:08x}", flush=True)
    # Enable PGRAPH FIFO PULL bit (0x400500 bit 0) to process incoming methods
    _fifo_ctrl = dev.read32(0x400500)
    if not (_fifo_ctrl & 0x1):
      dev.write32(0x400500, _fifo_ctrl | 0x1)
      if DEBUG:
        print(f"[kepler] PGRAPH FIFO PULL enabled: 0x{dev.read32(0x400500):08x}", flush=True)
    # Read CTX_SWITCH and DISPATCH state
    if DEBUG:
      _ctx_sw = dev.read32(0x404010)
      _disp_cmd_addr = dev.read32(0x404004)
      _disp_subch = dev.read32(0x404024)
      print(f"[kepler] DISPATCH: CTX_SWITCH=0x{_ctx_sw:08x} "
            f"CMD_ADDR=0x{_disp_cmd_addr:08x} SUBCH=0x{_disp_subch:08x}", flush=True)
    # Try to clear stuck CTXCTL by clearing the CTX_SWITCH register
    dev.write32(0x404010, 0x00000000)
    time.sleep(0.001)
    # Submit methods via BYPASS.  The words list is flat: header, arg0, arg1, ...
    # BYPASS_ADDR (0x5004): bits 2-13 = MTHD>>2, bits 16-18 = SUBCH
    # BYPASS_DATA (0x5008): poke to submit one word
    _wi = 0
    while _wi < len(words):
      _hdr = words[_wi]
      _size = (_hdr >> 16) & 0xfff
      _subc = (_hdr >> 13) & 0x7
      _mthd = (words[_wi] & 0x1fff) << 2  # method address
      _args = words[_wi + 1:_wi + 1 + _size]
      # Write ADDR with method and subchannel.
      # BYPASS_ADDR: bits 2-13 = MTHD (method address, already 4-byte aligned),
      # bits 16-18 = SUBCH.  The field value is method_addr >> 2, placed at
      # bit 2 in the register, so we write (mthd & 0x3ffc) | (subch << 16).
      dev.write32(0x5004, (_subc << 16) | (_mthd & 0x3ffc))
      if DEBUG and _wi < 40:
        _addr_rb = dev.read32(0x5004)
        _st = dev.read32(0x5000)
        print(f"[kepler] BYPASS submit: subc={_subc} mthd=0x{_mthd:x} "
              f"args={[hex(_a) for _a in _args]} "
              f"ADDR_rb=0x{_addr_rb:08x} STATUS=0x{_st:08x}", flush=True)
      for _arg in _args:
        dev.write32(0x5008, _arg)
        # Check for errors after each data write
        _st = dev.read32(0x5000)
        if _st & 0x1e00:  # error bits 8-12
          if DEBUG:
            print(f"[kepler] BYPASS error after data: STATUS=0x{_st:08x}", flush=True)
          break
      _wi += 1 + _size
    if DEBUG:
      _fifo_stat_post = dev.read32(0x400504)
      _fifo_ctrl_post = dev.read32(0x400500)
      _disp_trap = dev.read32(0x404000)
      _pgraph_stat = dev.read32(0x400700)
      _subch1_post = dev.read32(0x404200 + 4)
      print(f"[kepler] BYPASS done, STATUS=0x{dev.read32(0x5000):08x} "
            f"FIFO_CTRL=0x{_fifo_ctrl_post:08x} FIFO_STAT=0x{_fifo_stat_post:08x} "
            f"DISP_TRAP=0x{_disp_trap:08x} PGRAPH_STAT=0x{_pgraph_stat:08x} "
            f"subch1=0x{_subch1_post:08x}", flush=True)
    # SET_OBJECT failure is terminal for this deterministic invocation.
    _subch1_check = dev.read32(0x404200 + 4)
    if _subch1_check == 0:
      raise RuntimeError(
          "SET_OBJECT did not bind in bypass mode; direct DISPATCH fallback disabled")
  elif submit_mode == "dispatch" and dev.dev_impl.hw is not None:
    # Explicitly dangerous diagnostic path.  It is never reached from gpfifo
    # or bypass mode as an automatic recovery action.
    _wi2 = 0
    while _wi2 < len(words):
      _hdr2 = words[_wi2]
      _size2 = (_hdr2 >> 16) & 0xfff
      _subc2 = (_hdr2 >> 13) & 0x7
      _mthd2 = (words[_wi2] & 0x1fff) << 2
      _args2 = words[_wi2 + 1:_wi2 + 1 + _size2]
      _addr_val = (1 << 31) | (_subc2 << 16) | ((_mthd2 >> 2) << 2)
      for _ai2, _arg2 in enumerate(_args2):
        dev.write32(0x404004, _addr_val)
        dev.write32(0x404008, _arg2)
        if _ai2 + 1 < len(_args2):
          _addr_val += 4
      if DEBUG and _wi2 < 40:
        _sc = dev.read32(0x404200 + 4)
        print(f"[kepler] DISPATCH inject: subc={_subc2} mthd=0x{_mthd2:x} "
              f"args={[hex(_a) for _a in _args2]} subch1=0x{_sc:08x}", flush=True)
      _wi2 += 1 + _size2
  if DEBUG:
    print(f"[kepler] BAR1 USERD GP_PUT write=1 readback={userd_put_readback:#x} "
          f"userd_pa={userd_addr:#x} bar1_off={_userd_mmio_base:#x} "
          f"bar1={dev.dev_impl.bar1_addr:#x} size={dev.dev_impl.bar1_size:#x}")
    q0 = 0x40000
    trace = []
    for _ in range(8):
      snap = tuple(dev.read32(q0 + off) for off in
                   (0x00, 0x14, 0x18, 0x48, 0x54, 0x58, 0x5c, 0x64, 0x84, 0x88))
      if not trace or snap != trace[-1]:
        trace.append(snap)
    print(f"[kepler] pbdma0 submit trace={[[hex(x) for x in row] for row in trace]}",
          flush=True)
  # Commit the GR runlist (ID=3 from TOP scan).  Target 3 is non-coherent
  # system memory in Nouveau's gf100_runl_commit(), and the count is separate
  # from the channel ID.  Poll the host-visible semaphore page.
  if DEBUG and dev.dev_impl.hw is not None:
    # Rapid PGRAPH busy poll: check if the engine actually starts processing
    # the LAUNCH method.  0x40060c bit0 = gr_busy, 0x400700 = status.
    _busy_samples = []
    for _bi in range(20):
      _all_eng = [dev.read32(0x400a4 + _p * 0x2000) for _p in range(3)]
      _busy_samples.append((
        dev.read32(0x40060c),  # GR_BUSY
        dev.read32(0x400700),  # PGRAPH_STATUS
        dev.read32(0x400704),  # TRAPPED_ADDR
        _all_eng,
        dev.read32(0x404170),  # FE_PWR
        dev.read32(0x409100),  # FECS_CTRL
        dev.read32(0x404200 + 4),  # subch1
      ))
      time.sleep(0.001)
    print(f"[kepler] PGRAPH busy poll after kick:", flush=True)
    for _bj, (_b, _s, _a, _engs, _fp, _fc, _sc1) in enumerate(_busy_samples[:10]):
      print(f"  [{_bj:2d}] GR_BUSY=0x{_b:08x} STATUS=0x{_s:08x} "
            f"ADDR=0x{_a:08x} ENGINES={[hex(_e) for _e in _engs]} "
            f"FE_PWR=0x{_fp:08x} FECS=0x{_fc:08x} subch1=0x{_sc1:08x}", flush=True)
    # PBDMA error check: scan all PBDMAs to find the one serving our channel.
    for _pbdma_idx in range(3):
      _q = 0x40000 + _pbdma_idx * 0x2000
      _pbdma_intr = dev.read32(_q + 0x108)
      _pbdma_intr_mask = dev.read32(_q + 0x10c)
      _pbdma_hce_intr = dev.read32(_q + 0x148)
      _pbdma_hce_mask = dev.read32(_q + 0x14c)
      _pbdma_trap_addr = dev.read32(_q + 0xc0)
      _pbdma_trap_data = dev.read32(_q + 0xc4)
      _pbdma_chid = dev.read32(_q + 0x120)
      _pbdma_state0 = dev.read32(_q + 0x84)
      _pbdma_state1 = dev.read32(_q + 0x88)
      _pbdma_ib_get = dev.read32(_q + 0x14)
      _pbdma_ib_put = dev.read32(_q + 0x00)
      _pbdma_engines = dev.read32(_q + 0xa4)
      if _pbdma_chid or _pbdma_intr or _pbdma_engines:
        print(f"[kepler] PBDMA{_pbdma_idx}: INTR=0x{_pbdma_intr:08x} MASK=0x{_pbdma_intr_mask:08x} "
              f"HCE_INTR=0x{_pbdma_hce_intr:08x} HCE_MASK=0x{_pbdma_hce_mask:08x} "
              f"TRAP_ADDR=0x{_pbdma_trap_addr:08x} TRAP_DATA=0x{_pbdma_trap_data:08x} "
              f"CHID=0x{_pbdma_chid:08x} ENGINES=0x{_pbdma_engines:08x} "
              f"STATE0=0x{_pbdma_state0:08x} STATE1=0x{_pbdma_state1:08x} "
              f"IB_GET=0x{_pbdma_ib_get:08x} IB_PUT=0x{_pbdma_ib_put:08x}", flush=True)
    # Check PGRAPH subch registers and FECS FIFO
    _subch0 = dev.read32(0x400740)
    _subch1 = dev.read32(0x400744)
    _fecs_fifo_cmd = dev.read32(0x409504)
    _fecs_fifo_data = dev.read32(0x409500)
    _fecs_iren = dev.read32(0x409010)
    print(f"[kepler] PGRAPH subch: 0=0x{_subch0:08x} 1=0x{_subch1:08x} "
          f"FECS_FIFO_CMD=0x{_fecs_fifo_cmd:08x} "
          f"FECS_FIFO_DATA=0x{_fecs_fifo_data:08x} "
          f"IREN=0x{_fecs_iren:08x}", flush=True)
  _fecs_poll_count = 0
  _gp_get_snapshot_taken = False
  if hw is not None:
    hw.set_phase("semaphore-poll")
  for _ in range(2000):
    # FECS keep-alive: re-assert FE_PWR FORCE_ON (bit 1 only, NOT bit 4).
    # The FECS firmware uses bit 4 as a ctx_4170s/ctx_4170w handshake.
    dev.write32(0x404170, 0x00000002)  # FE_PWR FORCE_ON only
    if signal_vram_pa is not None:
      val = _gk104_pramin_read32(dev, signal_vram_pa)
    else:
      val = struct.unpack_from("<I", vram, signal_pa)[0]
    if val == done_value:
      # The caller still needs BAR0 diagnostics and (for compute) BAR1 output.
      # Quiescing here used to disable PCI DMA and stop keepalive before those
      # reads, which made the first post-launch RPC receive Completion Abort.
      # NVDevice.close() owns the sole quiesce point after all output access.
      return
    # Phase 1 capture point 2: after GP_GET shows entry consumed (one-shot).
    # Skip snapshot_gr_traps — it reads GPC/TPC registers which triggers FECS
    # power-gating.  Just record minimal info inline.
    if not _gp_get_snapshot_taken and dev.dev_impl.hw is not None:
      try:
        _gp_get_now = struct.unpack("<I", dev.dev_impl.hw.mmio_read(
            1, _userd_mmio_base + USERD_GP_GET, 4))[0]
        if _gp_get_now:
          _fecs_ctrl_snap = dev.read32(0x409100)
          _pgraph_status_snap = dev.read32(0x400700)
          _trapped_addr = dev.read32(0x400704)
          _trapped_data = dev.read32(0x400708)
          _intr = dev.read32(0x400100)
          _trap = dev.read32(0x400108)
          print(f"[kepler-trap] after_gp_get_consumed: INTR={_intr:#x} "
                f"TRAP={_trap:#x} ADDR={_trapped_addr:#x} "
                f"DATA={_trapped_data:#x} STATUS={_pgraph_status_snap:#x} "
                f"FE_PWR={dev.read32(0x404170):#x} FECS={_fecs_ctrl_snap:#x}",
                flush=True)
          _gp_get_snapshot_taken = True
      except Exception:
        pass
    # Poll FECS state every 10ms to catch when it stops
    _fecs_poll_count += 1
    if _fecs_poll_count % 10 == 0:
      _fecs_ctrl_poll = dev.read32(0x409100)
      _fecs_chan_next_poll = dev.read32(0x409b04)
      _fe_pwr_poll = dev.read32(0x404170)
      _pgraph_status_poll = dev.read32(0x400700)
      _pwr_gate_poll = dev.read32(0x020004)
      if _fecs_ctrl_poll != 0x20 or _fe_pwr_poll != 0x2:
        _fecs_pc_poll = dev.read32(0x409ff0)
        _fecs_scr0_poll = dev.read32(0x409800)
        _fecs_scr5_poll = dev.read32(0x409814)  # CC_SCRATCH(5) = error code
        _fecs_cpustat_poll = dev.read32(0x409128)
        _fecs_intr_poll = dev.read32(0x409008)
        _fecs_ca_poll = dev.read32(0x409b00)
        print(f"[kepler] poll {_fecs_poll_count}ms: FECS_CTRL={_fecs_ctrl_poll:#x} "
              f"PC={_fecs_pc_poll:#x} SCRATCH0={_fecs_scr0_poll:#x} "
              f"SCRATCH5={_fecs_scr5_poll:#x} "
              f"CPUSTAT={_fecs_cpustat_poll:#x} "
              f"INTR={_fecs_intr_poll:#x} "
              f"CHAN_ADDR={_fecs_ca_poll:#x} CHAN_NEXT={_fecs_chan_next_poll:#x} "
              f"FE_PWR={_fe_pwr_poll:#x} "
              f"PGRAPH_STATUS={_pgraph_status_poll:#x} "
              f"PWR_GATE={_pwr_gate_poll:#x} sem={val}", flush=True)
    time.sleep(0.001)
  # Do not quiesce and then continue issuing diagnostic BAR reads.  The caller's
  # finally block owns teardown, and a timeout path must fail without hundreds
  # of extra endpoint transactions against a potentially wedged channel.
  raise TimeoutError(f"semaphore did not reach {done_value} (last={val})")

  # Legacy deep diagnostics below are intentionally unreachable on the safe
  # live path.  Retained temporarily as register documentation for offline
  # analysis; they must never run after quiesce.
  # Phase 1 capture point 3: after timeout / first trap observed.
  if dev.dev_impl.hw is not None:
    snapshot_gr_traps(dev, "after_timeout_trap")
  diag = {r: dev.read32(r) for r in (0x2254, 0x800000, 0x800004, 0x800008, 0x2270, 0x2274,
                                     0x2100, 0x252c, 0x256c, 0x259c,
                                     0x400000, 0x400004, 0x400014, 0x400048,
                                     0x400500, 0x409100, 0x409800, 0x409128,
                                     0x2800, 0x2804, 0x2808, 0x280c,
                                     0x2810, 0x2814, 0x2818, 0x281c)}
  for pbdma in range(3):
    q = 0x40000 + pbdma * 0x2000
    diag.update({q + off: dev.read32(q + off) for off in (0x108, 0x10c, 0x120, 0x13c,
                                                            0x148, 0x14c, 0x150, 0x154,
                                                            0x000, 0x004, 0x008, 0x010,
                                                            0x014, 0x018, 0x048, 0x04c,
                                                            0x054, 0x058, 0x05c, 0x064,
                                                            0x0c0, 0x0c4,
                                                            0x100, 0x104, 0x110, 0x114,
                                                            0x118, 0x700, 0x704, 0x708,
                                                            0x70c, 0x740, 0x744, 0x748,
                                                            0x74c, 0x780, 0x784, 0x790,
                                                            0x044, 0x084, 0x088, 0x09c,
                                                            0x0a4, 0x0ac, 0x0e0, 0x0e4,
                                                            0x0e8, 0x038, 0x03c, 0x040,
                                                            0x028, 0x074, 0x098)})
  if DEBUG:
    q0 = 0x40000
    print("[kepler] pbdma0 pointers " + " ".join(
      f"{name}=0x{dev.read32(q0 + off):08x}" for name, off in (
        ("IB_PUT", 0x00), ("CTRL_LO", 0x08), ("SIG", 0x10),
        ("IB_GET", 0x14), ("DMA_GET", 0x18), ("IB_ADDR", 0x48),
        ("REF", 0x28),
        ("SEM_HI", 0x38), ("SEM_LO", 0x3c), ("SEM_SEQ", 0x40),
        ("SEM_STATE", 0x44),
        ("IB_CFG", 0x4c), ("IB_ENTRY", 0x54), ("DMA_PUT", 0x5c),
        ("IB_CRC", 0x74), ("PB_CRC", 0x98),
        ("STATE0", 0x84), ("STATE1", 0x88), ("INTR", 0x108),
        ("FEATURE", 0x9c), ("ENGINES", 0xa4), ("CUR_ENGINE", 0xac),
        ("IB_FLAGS", 0xe0), ("CTX_E4", 0xe4), ("CTX_E8", 0xe8),
        ("CH", 0x120))))
    print("[kepler] scheduler " + " ".join(
      f"{reg:#06x}=0x{dev.read32(reg):08x}" for reg in
      (0x2284, 0x2390, 0x2394, 0x2398, 0x262c, 0x2630, 0x2634,
       0x2638, 0x263c, 0x2640, 0x2644, 0x26c0,
       0x3080, 0x3084, 0x3088)), flush=True)
    vm_source = dev.read32(0x259c)
    for unit in range(32):
      if vm_source & (1 << unit):
        inst_fault = dev.read32(0x2800 + unit * 0x10)
        addr_lo = dev.read32(0x2804 + unit * 0x10)
        addr_hi = dev.read32(0x2808 + unit * 0x10)
        fault_type = dev.read32(0x280c + unit * 0x10)
        print(f"[kepler] VM_FAULT unit={unit} inst={inst_fault << 12:#x} "
              f"addr={(addr_hi << 32) | addr_lo:#x} type={fault_type:#010x} "
              f"reason={fault_type & 0xf} client={(fault_type >> 8) & 0x1f:#x} "
              f"hub={bool(fault_type & 0x40)} write={bool(fault_type & 0x80)}",
              flush=True)
    print(f"[kepler] PFIFO faults master={dev.read32(0x2100):#x} "
          f"sched={dev.read32(0x254c):#x} runlist={dev.read32(0x2a00):#x}", flush=True)
    # PGRAPH error diagnostics
    _pgraph_intr = dev.read32(0x400100)
    _pgraph_addr = dev.read32(0x400704)
    _pgraph_data = dev.read32(0x400708)
    _pgraph_status2 = dev.read32(0x400700)
    print(f"[kepler] PGRAPH error: INTR={_pgraph_intr:#x} ADDR={_pgraph_addr:#x} "
          f"DATA={_pgraph_data:#x} STATUS2={_pgraph_status2:#x} "
          f"FE_PWR={dev.read32(0x404170):#x} "
          f"PGRAPH_CTRL={dev.read32(0x400500):#x}", flush=True)
    # HUB fault diagnostics
    for _hub_unit in range(16):
      _hub_stat = dev.read32(0x409c00 + _hub_unit * 0x20)
      if _hub_stat & 0x80000000:
        _hub_addr = dev.read32(0x409c04 + _hub_unit * 0x20)
        print(f"[kepler] HUB fault unit={_hub_unit} stat={_hub_stat:#x} "
              f"addr={_hub_addr:#x}", flush=True)
  # PGRAPH error diagnostics (unconditional)
  _pgraph_intr = dev.read32(0x400100)
  _pgraph_addr = dev.read32(0x400704)
  _pgraph_data = dev.read32(0x400708)
  _pgraph_status2 = dev.read32(0x400700)
  _pgraph_trap = dev.read32(0x400108)
  _pwr_gate = dev.read32(0x020004)
  _pmc_enable = dev.read32(0x000200)
  print(f"[kepler] PGRAPH error: INTR={_pgraph_intr:#x} TRAP={_pgraph_trap:#x} ADDR={_pgraph_addr:#x} "
        f"DATA={_pgraph_data:#x} STATUS2={_pgraph_status2:#x} "
        f"FE_PWR={dev.read32(0x404170):#x} "
        f"PGRAPH_CTRL={dev.read32(0x400500):#x} "
        f"PWR_GATE={_pwr_gate:#x} PMC_ENABLE={_pmc_enable:#x}", flush=True)
  if _pgraph_trap:
    print(f"[kepler] GR trap units: DISPATCH={dev.read32(0x404000):#x} "
          f"CCACHE={dev.read32(0x408030):#x} SHADER={dev.read32(0x405840):#x} "
          f"SKED={dev.read32(0x407020):#x} GPC={dev.read32(0x400118):#x}", flush=True)
    for gpc in range(dev.read32(0x409604) & 0x1f):
      gb = 0x500000 + gpc * 0x8000
      print(f"[kepler] GPC{gpc} trap: summary={dev.read32(gb + 0x2c90):#x} "
            f"zcull={dev.read32(gb + 0x900):#x} ccache={dev.read32(gb + 0x1028):#x} "
            f"esetup={dev.read32(gb + 0x824):#x}", flush=True)
      for tpc in range(dev.read32(gb + 0x2608) & 0x1f):
        tb = 0x504000 + gpc * 0x8000 + tpc * 0x800
        stat = dev.read32(tb + 0x508)
        if stat & 0x1f:
          print(f"[kepler] GPC{gpc}/TPC{tpc} trap: summary={stat:#x} "
                f"tex={dev.read32(tb + 0x224):#x} poly={dev.read32(tb + 0x084):#x} "
                f"l1c={dev.read32(tb + 0x48c):#x} mpc={dev.read32(tb + 0x430):#x} "
                f"mp_warp={dev.read32(tb + 0x648):#x} "
                f"mp_global={dev.read32(tb + 0x650):#x}", flush=True)
  # FECS state diagnostics (is FECS processing context switches?)
  _fecs_cpuctl = dev.read32(0x409100)
  _fecs_cpustat = dev.read32(0x409128)
  _fecs_intr = dev.read32(0x409008)
  _fecs_iren = dev.read32(0x409010)
  _fecs_scratch0 = dev.read32(0x409800)
  _fecs_scratch1 = dev.read32(0x409804)
  _fecs_scratch5 = dev.read32(0x409814)  # error code register
  _fecs_chan_addr = dev.read32(0x409b00)
  _fecs_chan_next = dev.read32(0x409b04)
  _fecs_fifo_cmd = dev.read32(0x409504)
  _fecs_fifo_data = dev.read32(0x409500)
  _fecs_mem_cmd = dev.read32(0x409a1c)  # MEM_CMD
  _fecs_mem_target = dev.read32(0x409acc)  # MEM_TARGET
  _fecs_mem_base = dev.read32(0x409a20)  # MEM_BASE
  _fecs_pc = dev.read32(0x409ff0)
  print(f"[kepler] FECS state: CPUCTL={_fecs_cpuctl:#x} CPUSTAT={_fecs_cpustat:#x} "
        f"INTR={_fecs_intr:#x} IREN={_fecs_iren:#x} "
        f"SCRATCH0={_fecs_scratch0:#x} SCRATCH1={_fecs_scratch1:#x} "
        f"SCRATCH5(err)={_fecs_scratch5:#x} "
        f"CHAN_ADDR={_fecs_chan_addr:#x} CHAN_NEXT={_fecs_chan_next:#x} "
        f"FIFO_CMD={_fecs_fifo_cmd:#x} FIFO_DATA={_fecs_fifo_data:#x} "
        f"MEM_CMD={_fecs_mem_cmd:#x} MEM_TARGET={_fecs_mem_target:#x} "
        f"MEM_BASE={_fecs_mem_base:#x} PC={_fecs_pc:#x}", flush=True)
  # GPCCS state
  _gpccs_cpuctl = dev.read32(0x41a100)
  _gpccs_cpustat = dev.read32(0x41a128)
  _gpccs_scratch0 = dev.read32(0x41a800)
  _gpccs_pc = dev.read32(0x41aff0)
  print(f"[kepler] GPCCS state: CPUCTL={_gpccs_cpuctl:#x} CPUSTAT={_gpccs_cpustat:#x} "
        f"SCRATCH0={_gpccs_scratch0:#x} PC={_gpccs_pc:#x}", flush=True)
  top = [dev.read32(0x22700 + i * 4) for i in range(64)]
  gp_get = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + USERD_GP_GET, 4))[0]
  push_get = struct.unpack("<II", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + 0x58, 8))
  remote_sem = None
  if dev.dev_impl.hw is not None:
    try:
      remote_sem = dev.dev_impl.hw.sysmem_read(base + signal_pa, 4).hex()
    except Exception as e:
      remote_sem = f"read-error:{type(e).__name__}"
  raise TimeoutError(f"semaphore did not reach {done_value} (last={val}, "
                     f"remote_sem={remote_sem}, userd_gp_get={gp_get:#x}, "
                     f"userd_push_get={[hex(x) for x in push_get]}, regs={diag}, "
                     f"top={[hex(x) for x in top if x]})")

def _freeze_stop_and_hold(dev, reason):
  """Freeze protocol traffic, stop local helpers, and retain live mappings."""
  hw = dev.dev_impl.hw
  if hw is None:
    return
  hw.freeze(reason)
  hw.set_phase("host-helper-stop")
  teardown = getattr(dev, "_kepler_emergency_teardown", None)
  if teardown is not None:
    teardown(reason)
  hold_seconds = float(os.environ.get("KEPLER_HOLD_OPEN_SECONDS", "0"))
  if not math.isfinite(hold_seconds) or hold_seconds < 0:
    raise ValueError("KEPLER_HOLD_OPEN_SECONDS must be a finite non-negative number")
  if hold_seconds:
    hw.set_phase("hold-open")
    print(f"[kepler] zero-RPC hold with mappings alive: {hold_seconds:g}s", flush=True)
    time.sleep(hold_seconds)


def run_hardware_demo(dev, cubin=None):
  """End-to-end add on the real GTX 770 over raw PCIe MMIO (sysmem compute
  path, plan §24.1).  Requires root, a GK104 firmware tree ($NV_FIRMWARE_DIR),
  and on-silicon FIFO validation."""
  import random
  N = 256
  test_stage = os.environ.get("KEPLER_TEST_STAGE", "full-add")
  if test_stage not in ("sem", "set-object", "full-add"):
    raise ValueError(
        f"unsupported KEPLER_TEST_STAGE={test_stage!r}; "
        "implemented stages are sem, set-object, and full-add")
  sem_only = test_stage == "sem"
  set_object_only = test_stage == "set-object"
  if cubin is None: cubin = get_kepler_cubin()
  prog = dev.runtime("E_4", cubin)
  allocator = NVAllocator(dev)

  a_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
  b_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])

  a_dev = allocator.alloc(N * 4)
  b_dev = allocator.alloc(N * 4)
  out_dev = allocator.alloc(N * 4)
  signal = allocator.alloc(16)
  # Mesa programs TEMP size per MP in 32-KiB units.  GK104/GTX 770 has eight
  # MPs, so the smallest non-zero legal backing allocation is 8 * 0x8000.
  temp_dev = allocator.alloc(0x40000)
  # Code + constant (param) buffers for the launch descriptor.
  sass = elf_section_bytes(cubin, ".text.E_4")
  code_dev = allocator.alloc(round_up(len(sass), 0x100))
  allocator._copyin(code_dev, sass)
  cbuf = build_cuda_param_cbuf(a_dev.va_addr, b_dev.va_addr, out_dev.va_addr)
  cbuf_dev = allocator.alloc(len(cbuf))
  allocator._copyin(cbuf_dev, cbuf)
  # The one-shot demo has no prior GPU producer to acquire from.  Seed a
  # sentinel and require the GPU completion packet to replace it with 2.
  allocator._copyin(signal, struct.pack("<I", 1))
  cwd = build_cwd(code_addr=0, grid=(1, 1, 1), block=(N, 1, 1), cbuf_addr=cbuf_dev.va_addr,
                  regs=cubin_register_count(cubin, "E_4"))
  cwd_dev = allocator.alloc(len(cwd))
  allocator._copyin(cwd_dev, cwd)
  # GPC instruction/data clients on this un-POSTed card do not reliably see
  # the SYS aperture even though HUB/PBDMA semaphore traffic does.  Put the
  # complete compute working set behind the already-validated VRAM VMM.  The
  # output mirror is read back after WFI; no host arithmetic is involved.
  dev._kepler_vram_mirrors = ([] if sem_only else
      [a_dev, b_dev, out_dev, temp_dev, code_dev, cbuf_dev, cwd_dev])
  allocator._copyin(a_dev, a_host.tobytes())
  allocator._copyin(b_dev, b_host.tobytes())
  # A GPU result must overwrite every lane; untouched VRAM stays non-finite.
  allocator._copyin(out_dev, struct.pack("<I", 0x7fc00001) * N)

  words = build_launch_words(signal.va_addr, 1, 2, cwd_dev.va_addr,
                             code_dev.va_addr, temp_dev.va_addr, 0x40000)
  if set_object_only:
    words = [*nvm(1, 0x0000, KEPLER_COMPUTE_A),
             *gk104_semaphore(signal.va_addr, 2, 0x00000002),
             *nvm(0, 0x0020, 0)]
  # Verify compute buffers are mapped in the channel's VMM
  if dev.dev_impl.hw is not None:
    for _name, _buf in [("a_dev", a_dev), ("b_dev", b_dev), ("out_dev", out_dev),
                        ("code_dev", code_dev), ("cbuf_dev", cbuf_dev),
                        ("cwd_dev", cwd_dev), ("signal", signal),
                        ("temp_dev", temp_dev)]:
      _va = _buf.va_addr
      _pgd_idx = (_va >> 27) & 0x1fff
      _spt_idx = (_va >> 12) & 0x7fff
      _pgd_entry = dev.dev_impl.mm.root_page_table.entry(_pgd_idx)
      try:
        _spt = GK104PageTableEntry(dev.dev_impl, dev.dev_impl.mm.root_page_table.address(_pgd_idx), 0)
        _pte = _spt.entry(_spt_idx)
      except Exception:
        _pte = 0
      print(f"[kepler] VMM map: {_name} va={_va:#x} pa={_buf.meta['pa']:#x} "
            f"pgd[{_pgd_idx}]={_pgd_entry:#x} pte={_pte:#x}", flush=True)
  # FE_PWR was already set to FORCE_ON before the channel bind above.
  # Just verify it's still on; if FE power-gated during setup, report it.
  if dev.dev_impl.hw is not None:
    _fe_pwr_final = dev.read32(0x404170)
    _pgraph_status_pre = dev.read32(0x400000)
    print(f"[kepler] FE_PWR before launch: {_fe_pwr_final:#x} "
          f"RED_SWITCH={dev.read32(0x409614):#x} "
          f"PGRAPH_STATUS={_pgraph_status_pre:#x} "
          f"FECS_CTRL={dev.read32(0x409100):#x}", flush=True)
  if DEBUG:
    pgd_idx = (signal.va_addr >> 27) & 0x1fff
    spt_idx = (signal.va_addr >> 12) & 0x7fff
    pgd_entry = dev.dev_impl.mm.root_page_table.entry(pgd_idx)
    spt = GK104PageTableEntry(dev.dev_impl, dev.dev_impl.mm.root_page_table.address(pgd_idx), 0)
    print(f"[kepler] vmm signal_va={signal.va_addr:#x} signal_pa={signal.meta['pa']:#x} "
          f"bus={dev.dev_impl.mm.bus_base:#x} pgd={pgd_entry:#x} pte={spt.entry(spt_idx):#x}")
  if sem_only:
    allocator._copyin(signal, struct.pack("<I", 0))
    # Host-only semaphore RELEASE without WFI: no engine methods precede this,
    # so a WFI would stall forever waiting for ENGINES!=0.  RELEASE_WFI=DIS
    # (bit20=1) lets the PBDMA write the semaphore through the HUB VMM directly.
    # Keep the scheduler gate entirely inside PBDMA.  SET_REFERENCE on an
    # unbound subchannel raises EMPTY_SUBC/DEVICE before the semaphore packet.
    words = [*gk104_semaphore(signal.va_addr, 2, 0x01000002)]
  submit_launch(dev, words, signal.va_addr, signal.meta['pa'], 1, 2)

  if sem_only or set_object_only:
    print(f"hardware_{test_stage}=ok value=2")
    _freeze_stop_and_hold(dev, f"{test_stage}-complete")
    return

  # Deep trap/status capture changes live GR state and emits hundreds of RPCs.
  # Keep it available only as an explicit unsafe diagnostic; a normal add run
  # goes directly from the completed, serialized semaphore to one bulk output
  # read, matching the smallest proven command path.
  if (dev.dev_impl.hw is not None and
      os.environ.get("KEPLER_UNSAFE_POSTLAUNCH_DIAGNOSTICS") == "1"):
    _pgraph_intr_post = dev.read32(0x400100)
    _pgraph_addr_post = dev.read32(0x400704)
    _pgraph_data_post = dev.read32(0x400708)
    _pgraph_status2_post = dev.read32(0x400700)
    _pgraph_status_post = dev.read32(0x400000)
    # HUB fault status
    _hub_stat_post = dev.read32(0x409c00)
    # Check FECS state
    _fecs_ctrl_post = dev.read32(0x409100)
    _fecs_scratch0_post = dev.read32(0x409800)
    # Check PBDMA state
    _pbdma0_ctrl = dev.read32(0x40000)
    _pbdma0_stat = dev.read32(0x40008)
    # Check PGRAPH exception registers
    _pgraph_exc = dev.read32(0x400108)
    # Check MP trap registers (per-GPC)
    _mp_trap = []
    for _gpc in range(dev.read32(0x409604) & 0x1f):
      _tpc_nr = dev.read32(0x500000 + _gpc * 0x8000 + 0x2608) & 0x1f
      for _tpc in range(_tpc_nr):
        _warp_addr, _global_addr = gk104_mp_trap_addrs(_gpc, _tpc)
        _warp, _global = dev.read32(_warp_addr), dev.read32(_global_addr)
        if _warp or _global:
          _mp_trap.append(f"GPC{_gpc}.TPC{_tpc}: warp={_warp:#x} global={_global:#x}")
    print(f"[kepler] post-launch: PGRAPH_INTR={_pgraph_intr_post:#x} "
          f"ADDR={_pgraph_addr_post:#x} DATA={_pgraph_data_post:#x} "
          f"STATUS={_pgraph_status_post:#x} STATUS2={_pgraph_status2_post:#x} "
          f"EXC={_pgraph_exc:#x} "
          f"HUB_STAT={_hub_stat_post:#x} "
          f"FECS_CTRL={_fecs_ctrl_post:#x} SCRATCH0={_fecs_scratch0_post:#x} "
          f"PBDMA0_CTRL={_pbdma0_ctrl:#x} PBDMA0_STAT={_pbdma0_stat:#x}", flush=True)
    if _mp_trap:
      print(f"[kepler] MP traps: {_mp_trap}", flush=True)
    # Check HUB faults in detail
    if _hub_stat_post:
      for _hub_unit in range(8):
        _unit_stat = dev.read32(0x409c00 + _hub_unit * 0x20)
        if _unit_stat & 0x80000000:
          _unit_addr = dev.read32(0x409c04 + _hub_unit * 0x20)
          print(f"[kepler] HUB fault unit={_hub_unit} stat={_unit_stat:#x} "
                f"addr={_unit_addr:#x}", flush=True)
    # Phase 1 capture point 4: after W1C clear (nouveau gf100_gr_trap_intr).
    # Clear traps in the same order nouveau does, then take a final snapshot.
    _trap = dev.read32(0x400108)
    if _trap & 0x1:
      dev.write32(0x404000, 0xc0000000)
      dev.write32(0x400108, 0x1)
    if _trap & 0x2:
      dev.write32(0x404600, 0xc0000000)
      dev.write32(0x400108, 0x2)
    if _trap & 0x8:
      dev.write32(0x408030, 0xc0000000)
      dev.write32(0x400108, 0x8)
    if _trap & 0x10:
      dev.write32(0x405840, 0xc0000000)
      dev.write32(0x400108, 0x10)
    if _trap & 0x40:
      dev.write32(0x40601c, 0xc0000000)
      dev.write32(0x400108, 0x40)
    if _trap & 0x80:
      dev.write32(0x404490, 0xc0000000)
      dev.write32(0x400108, 0x80)
    if _trap & 0x100:
      stat = dev.read32(0x407020) & 0x3fffffff
      if stat:
        dev.write32(0x407020, 0x40000000)
      dev.write32(0x400108, 0x100)
    if _trap & 0x01000000:
      _gpc_stat = dev.read32(0x400118)
      for _g in range(dev.read32(0x409604) & 0x1f):
        if _gpc_stat & (1 << _g):
          # Clear per-GPC TPC traps then the GPC summary.
          _gb = 0x500000 + _g * 0x8000
          _gpc_sum = dev.read32(_gb + 0x2c90)
          _tpc_n = dev.read32(_gb + 0x2608) & 0x1f
          for _t in range(_tpc_n):
            _mask = 0x00010000 << _t
            if _gpc_sum & _mask:
              _tb = 0x504000 + _g * 0x8000 + _t * 0x800
              _ts = dev.read32(_tb + 0x508)
              if _ts & 0x1:
                dev.write32(_tb + 0x224, 0xc0000000)
              if _ts & 0x2:
                dev.write32(_tb + 0x648, 0)
                dev.write32(_tb + 0x650, dev.read32(_tb + 0x650))
              if _ts & 0x4:
                dev.write32(_tb + 0x84, 0xc0000000)
              if _ts & 0x8:
                dev.write32(_tb + 0x48c, 0xc0000000)
              if _ts & 0x10:
                dev.write32(_tb + 0x430, 0xc0000000)
              dev.write32(_gb + 0x2c90, _mask)
          if _gpc_sum & 0x1:
            dev.write32(_gb + 0x420, 0xc0000000)
          if _gpc_sum & 0x2:
            dev.write32(_gb + 0x900, 0xc0000000)
          if _gpc_sum & 0x4:
            dev.write32(_gb + 0x1028, 0xc0000000)
          if _gpc_sum & 0x8:
            dev.write32(_gb + 0x824, 0xc0000000)
          dev.write32(0x400118, 1 << _g)
      dev.write32(0x400108, 0x01000000)
    if _trap & 0x02000000:
      _rop_nr = (dev.read32(0x409604) >> 16) & 0x1f
      for _r in range(max(_rop_nr, 1)):
        dev.write32(0x410000 + _r * 0x400 + 0x070, 0xc0000000)
        dev.write32(0x410000 + _r * 0x400 + 0x144, 0xc0000000)
      dev.write32(0x400108, 0x02000000)
    if _trap:
      dev.write32(0x400108, _trap)
    _intr = dev.read32(0x400100)
    if _intr:
      dev.write32(0x400100, _intr)
    snapshot_gr_traps(dev, "after_w1c_clear")
    # Restore FE_PWR to AUTO and re-enable ctxctl clock-gate now that the
    # launch and diagnostics are done.
    dev.write32(0x404170, 0x00000010)  # NV_PGRAPH_FE_PWR_MODE_AUTO
    dev.write32(0x000260, 0x00000001)  # nvkm_mc_unk260(1): re-enable ctxctl

  out_host = bytearray(N * 4)
  if dev.dev_impl.hw is not None and out_dev.meta.get("vram_pa") is None:
    dev.dev_impl.hw.freeze("missing-output-vram-mapping")
    raise RuntimeError("full-add output has no VRAM BAR1 mapping")
  if dev.dev_impl.hw is not None:
    # LAUNCH is followed by GRAPH_SERIALIZE and a WFI-enabled semaphore; the
    # QMD also requests FE_SYSMEMBAR release ordering.  Read BAR1 in one RPC.
    # A host-driven LTC flush/invalidate here adds more BAR0 transactions after
    # completion and is not part of Mesa's NVE4 launch/fence sequence.
    dev.dev_impl.hw.arm_final_output_read(
        1, out_dev.meta["vram_pa"], len(out_host))
    out_host[:] = dev.dev_impl.hw.mmio_read(
        1, out_dev.meta["vram_pa"], len(out_host))
    dev.dev_impl.hw.freeze("output-read-complete")
    print(f"[kepler] GPU output read from VRAM pa={out_dev.meta['vram_pa']:#x}",
          flush=True)
  else:
    allocator._copyout(out_host, out_dev)
  out_arr = array.array('f'); out_arr.frombytes(bytes(out_host))
  expected = [a_host[i] + b_host[i] for i in range(N)]
  # Debug: show first few values
  _mismatches = sum(1 for i in range(N)
                    if not math.isfinite(out_arr[i]) or
                    abs(out_arr[i] - expected[i]) >= 1e-5)
  print(f"[kepler] output: first 8 actual={[round(out_arr[i],4) for i in range(8)]} "
        f"expected={[round(expected[i],4) for i in range(8)]} "
        f"mismatches={_mismatches}/{N}", flush=True)
  if _mismatches > 0:
    print(f"[kepler] raw output hex: {out_host[:32].hex()}", flush=True)
    print(f"[kepler] raw a_host hex: {a_host.tobytes()[:32].hex()}", flush=True)
    print(f"[kepler] raw b_host hex: {b_host.tobytes()[:32].hex()}", flush=True)
  _freeze_stop_and_hold(dev, "output-read-complete")
  assert _mismatches == 0, f"hardware add mismatch ({_mismatches}/{N} wrong)"
  print(f"hardware_demo=ok N={N}")

def main():
  if "--middle-selftest" in sys.argv:
    kepler_selftest()
    return
  if "--vbios-info" in sys.argv:
    i = sys.argv.index("--vbios-info")
    path = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else \
      os.path.join(os.path.dirname(__file__), "Palit.GTX770.4096.131216.rom")
    try:
      inspect_vbios(path)
    except (OSError, ValueError) as e:
      print(f"vbios-info: {e}", file=sys.stderr)
      sys.exit(1)
    return
  if "--vbios-init-info" in sys.argv:
    i = sys.argv.index("--vbios-init-info")
    path = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else \
      os.path.join(os.path.dirname(__file__), "Palit.GTX770.4096.131216.rom")
    try:
      vbios_init_info(path)
    except (OSError, ValueError, struct.error) as e:
      print(f"vbios-init-info: {e}", file=sys.stderr)
      sys.exit(1)
    return
  live_probe_flags = {
    "--probe", "--probe-pgob", "--probe-gpc-clock",
    "--probe-vbios-devinit", "--probe-falcon",
  }
  if (live_probe_flags.intersection(sys.argv)
      and os.environ.get("KEPLER_LIVE_ACK") != "raw-mmio-risk"):
    print("live Kepler probe refused: this path drives GK104 raw MMIO from "
          "userspace with no validated RM; a bad sequence can hang the card or "
          "the PCIe link.  Set KEPLER_LIVE_ACK=raw-mmio-risk only for one "
          "explicitly authorized test on a card unbound from the nvidia driver "
          "(root required).",
          file=sys.stderr)
    sys.exit(2)
  if "--probe" in sys.argv:
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe: GK104 not reachable (is the card unbound from nvidia? set NV_PCIBDF)")
      sys.exit(1)
    d.bar_info(0)  # mmap the register BAR before any MMIO
    boot0 = d.mmio_read32(0, 0x0)  # PMC_BOOT_0: chip id + stepping
    print(f"probe: PMC_BOOT_0=0x{boot0:08x} (chip_id={(boot0>>20)&0xfff}, "
          f"revision=0x{(boot0>>4)&0xff}, fab=0x{(boot0>>4)&0xf})")
    d.fini()
    return
  if "--probe-pgob" in sys.argv:
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe-pgob: GK104 not reachable (is the card unbound from nvidia?)")
      sys.exit(1)
    d.bar_info(0)
    boot0 = d.mmio_read32(0, 0x0)
    print(f"probe-pgob: PMC_BOOT_0=0x{boot0:08x} (chip_id={(boot0>>20)&0xfff})")
    class BAR0Dev:
      def read32(self, r): return d.mmio_read32(0, r)
      def write32(self, r, v): return d.mmio_write32(0, r, v)
    dev = BAR0Dev()
    dev.write32(0x000200, 0xffffffff)   # PMC_ENABLE = full mask
    print("probe-pgob: PMC_ENABLE=0xffffffff")
    probe_pgob_power_on(dev)
    d.fini()
    return
  if "--probe-gpc-clock" in sys.argv:
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe-gpc-clock: GK104 not reachable (is the card unbound from nvidia?)")
      sys.exit(1)
    d.bar_info(0)
    boot0 = d.mmio_read32(0, 0x0)
    print(f"probe-gpc-clock: PMC_BOOT_0=0x{boot0:08x} (chip_id={(boot0>>20)&0xfff})")
    class BAR0Dev:
      def read32(self, r): return d.mmio_read32(0, r)
      def write32(self, r, v): return d.mmio_write32(0, r, v)
    dev = BAR0Dev()
    dev.write32(0x000200, 0xffffffff)   # PMC_ENABLE = full mask
    print("probe-gpc-clock: PMC_ENABLE=0xffffffff")
    probe_gpc_fixed_100mhz(dev)
    d.fini()
    return
  if "--probe-vbios-devinit" in sys.argv:
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe-vbios-devinit: GK104 not reachable (is the card unbound from nvidia?)")
      sys.exit(1)
    d.bar_info(0)
    class BAR0Dev:
      def read32(self, r): return d.mmio_read32(0, r)
      def write32(self, r, v): return d.mmio_write32(0, r, v)
    dev = BAR0Dev()
    dev.write32(PMC_ENABLE, 0xffffffff)
    i = sys.argv.index("--probe-vbios-devinit")
    path = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else \
      os.path.join(os.path.dirname(__file__), "Palit.GTX770.4096.131216.rom")
    image, _, scripts = vbios_init_info(path)
    print(f"probe-vbios-devinit: executing script0={scripts[0]:#x}")
    for script in scripts:
      execute_vbios_target_ops(dev, image, script)
    try:
      program_gk104_gpc_pll(dev)
    except RuntimeError as e:
      print(f"probe-vbios-devinit: {e}")
    for reg in (0x137000, 0x137100, 0x137160, 0x409604, 0x41a100, 0x502100):
      print(f"probe-vbios-devinit: {reg:#x}={dev.read32(reg):#010x}")
    d.fini()
    return
  if "--probe-falcon" in sys.argv:
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe-falcon: GK104 not reachable (is the card unbound from nvidia?)")
      sys.exit(1)
    d.bar_info(0)
    boot0 = d.mmio_read32(0, 0x0)
    print(f"probe-falcon: PMC_BOOT_0=0x{boot0:08x} (chip_id={(boot0>>20)&0xfff})")
    # Minimal FALCON bring-up via NVDevice hardware init (loads FECS+GPCCS,
    # starts FECS, waits for 0x409800 bit31).
    try:
      dev = NVDevice(backend="hardware")
      print("probe-falcon: FECS+GPCCS loaded and started OK")
    except Exception as e:
      print(f"probe-falcon: FALCON bring-up failed: {type(e).__name__}: {e}")
      sys.exit(1)
    d.fini()
    return
  # Allow --cubin PATH to supply a real sm_30 cubin.
  for i, a in enumerate(sys.argv):
    if a == "--cubin" and i + 1 < len(sys.argv):
      os.environ["KEPLER_CUBIN"] = sys.argv[i + 1]
  backend = os.environ.get("NV_BACKEND", "kepler")
  cubin = None
  if backend != "software":
    if os.environ.get("KEPLER_LIVE_ACK") != "raw-mmio-risk":
      print("hardware launch refused: this GK104 path drives raw MMIO from "
            "userspace with no validated RM and can hang the card or the PCIe "
            "link; set KEPLER_LIVE_ACK=raw-mmio-risk only for one explicitly "
            "authorized test on a card unbound from the nvidia driver (root)",
            file=sys.stderr)
      sys.exit(2)
    if DEBUG != 0:
      print("hardware launch refused: DEBUG must remain 0 during crash isolation",
            file=sys.stderr)
      sys.exit(2)
    submit_mode = os.environ.get("KEPLER_SUBMIT_MODE", "gpfifo")
    if submit_mode not in ("gpfifo", "bypass", "dispatch"):
      print(f"hardware launch refused: invalid KEPLER_SUBMIT_MODE={submit_mode!r}",
            file=sys.stderr)
      sys.exit(2)
    test_stage = os.environ.get("KEPLER_TEST_STAGE", "full-add")
    if test_stage not in ("sem", "set-object", "full-add"):
      print(f"hardware launch refused: unsupported KEPLER_TEST_STAGE={test_stage!r}; "
            "implemented stages are sem, set-object, and full-add",
            file=sys.stderr)
      sys.exit(2)
    if ("KEPLER_TEST_SEM_ONLY" in os.environ or
        "KEPLER_TEST_SET_OBJECT" in os.environ):
      print("hardware launch refused: legacy test-stage variables are disabled; "
            "use KEPLER_TEST_STAGE", file=sys.stderr)
      sys.exit(2)
    try: cubin = get_kepler_cubin()
    except (RuntimeError, ValueError, KeyError, struct.error) as e:
      print(f"hardware launch refused: {e}", file=sys.stderr)
      sys.exit(2)
  try:
    dev = NVDevice("NV", backend=backend)
  except (NotImplementedError, OSError, RuntimeError) as e:
    print(f"hardware initialization failed safely: {e}", file=sys.stderr)
    print("No automatic probe/retry was made; confirm the GK104 is unbound from "
          "the nvidia driver and that you are root before another isolated run.",
          file=sys.stderr)
    sys.exit(2)
  try:
    if backend == "software":
      run_software_demo(dev)
    else:
      run_hardware_demo(dev, cubin)
  finally:
    dev.close()

if __name__ == "__main__":
  main()
