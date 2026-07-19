#!/usr/bin/env python3
"""Standalone NV stack for a GTX 660 Ti (Kepler GK104 / NVE4, sm_30).

Fork of ``examples_kepler/add.py`` (GTX 770 / Palit strap-6 golden) with the
board-specific defaults this eGPU actually needs:

  - live ``0x101000`` RAMCFG group **5** (not the Palit 770 pin of 6)
  - 7-TPC fuse map (``[2,2,2,1]``) is read from HW; do not assume 8 SM
  - PCI id ``0x1183`` (660 Ti); some runs may show ``0x1184`` after a foreign
    (Palit 770) BIT-I POST rewrote identity — refuse that ROM on this board
  - this enclosure's card is Gigabyte ``1458:3556`` (GV-N66TWF2-2GD class);
    cold POST must use the onboard PROM (dumped) or a matching 0x1183 VBIOS,
    never the Palit GTX770 image

Self-contained like ``examples/add.py``.  Transport is platform-selected:

  - macOS: TinyGPU.app DriverKit Unix socket (``/tmp/tinygpu.sock``)
  - Linux: raw BAR0/BAR1 mmap via sysfs ``resourceN`` (root / CAP_SYS_RAWIO;
    GK104 unbound from nvidia)


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
  - nvbios_init / pgraph_mmio_gk104 (sibling modules in this directory)

NO imports from tinygrad.runtime.support / ops / device / renderer / uop / helpers
are permitted on the live path — those have been vendored inline below.

``python3 examples_kepler/add.py --middle-selftest`` exercises the offline
builders without touching the card.  ``--mmiotrace-selftest`` is the golden
Nouveau mmiotrace gate (no hardware/pagemap).  ``NV_BACKEND=software`` runs the
complete allocator and launch-word path against host memory.  On Linux a normal
live invocation selects the unbound GK104 from sysfs and requests sudo when
necessary; on macOS it talks to TinyGPU.app at ``/tmp/tinygpu.sock``.
``KEPLER_OPERATION=mul`` selects the multiply image and expected vector result.
``examples_kepler_pcie/add.py`` is a thin re-export of this file for older
callers.  DEBUG=1 enables the detailed register trace; normal launches keep
only summary lines so a successful health check is easy to spot in logs.
Progress and crash-safety notes live beside this file in `progress.md`.
"""
from __future__ import annotations
import os, sys, ctypes, ctypes.util, time, mmap, struct, math, array as _array_mod, socket, subprocess, contextlib, functools, itertools, enum, atexit, select, dataclasses, collections, urllib.request, hashlib, tempfile, gzip, pathlib, json, threading, io
from typing import cast, Any, ClassVar, Generic, TypeVar

# Resolve everything from this checkout before importing tinygrad.  This keeps
# both `/usr/bin/python3` (including sudo's clean environment) and a virtualenv
# working without installing anything system-wide.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SHARED_KEPLER_DIR = SCRIPT_DIR  # assets + nvbios_init live beside this file
TINYGRAD_ROOT = os.path.join(REPO_ROOT, "ref")
for _path in (TINYGRAD_ROOT, SHARED_KEPLER_DIR):
  if _path not in sys.path: sys.path.insert(0, _path)
# Palit 770 is the add.py / offline golden only.  Live 660 Ti cold POST must
# use onboard PROM (dumped to ONBOARD_VBIOS_CACHE) or an explicit KEPLER_VBIOS.
PALIT_770_VBIOS = os.path.join(SHARED_KEPLER_DIR, "Palit.GTX770.4096.131216.rom")
ONBOARD_VBIOS_CACHE = os.path.join(SHARED_KEPLER_DIR, "onboard_gk104.rom")
DEFAULT_VBIOS = PALIT_770_VBIOS
# Gigabyte GTX 660 Ti WindForce (TechPowerUp #128422) observed on this enclosure.
GIGABYTE_660TI_SUBSYS = 0x35561458  # little-endian config dword: vend|dev
DEFAULT_CUBIN = os.path.join(SHARED_KEPLER_DIR, "add_kepler.cubin")
DEFAULT_MUL_CUBIN = os.path.join(SHARED_KEPLER_DIR, "mul_kepler.cubin")
# Reference produced by CUDA 10.2 `ptxas -arch=sm_30` for the multiply
# variant assembled from the checked-in Kepler PTX.  Keep this independent of
# the add cubin so a stale/precompiled add image cannot silently run a mul test.
MUL_CUBIN_BYTES = 1768
MUL_CUBIN_SHA256 = "7f0e019e5fe8dc5e68e1b14a70c3841391b63062df2daba26d01f8e6d0ac2a52"

# Optional override for PCI transport (tests / alternate backends).
# Default: APLRemotePCIDevice on macOS, LinuxPCIDevice elsewhere.
_PCI_TRANSPORT_FACTORY = None
def set_pci_transport_factory(factory):
  global _PCI_TRANSPORT_FACTORY
  _PCI_TRANSPORT_FACTORY = factory

# --- autogen ctypes (allowed: "ctypes constants only") ---
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu, pci
from tinygrad.runtime.autogen import nv_regs
from tinygrad.runtime.autogen import libc
# Reuse the Kepler bring-up modules from examples_kepler/ rather than duplicating
# them here (ponytail: fewest files).  NV_KEPLER_PATH overrides for non-standard layouts.
sys.path.insert(0, os.environ.get("NV_KEPLER_PATH", SHARED_KEPLER_DIR))
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
  val = None
  while int(time.perf_counter() * 1000) - start_time < timeout_ms:
    if (val := cb(*args)) == value: return val
    time.sleep(0.0001)
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

# GK104 PCI device ids — used by Linux probe() when NV_PCIBDF is unset.
# 0x1183 = GTX 660 Ti; 0x1184 = GTX 770 (some fused 7-TPC boards still report it).
GK104_PCI_IDS = {
    0x1180, 0x1183, 0x1184, 0x1185, 0x1187, 0x1188, 0x1189, 0x118e, 0x1195,
}


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



# ============================================================================
# macOS TinyGPU.app DriverKit transport (Unix socket /tmp/tinygpu.sock)
# ============================================================================
class RemoteCmd(enum.IntEnum):
  # Wire command IDs spoken by TinyGPU.app's DriverKit extension, as decoded by
  # TheTom/pascal-egpu's TinyGPUClient (which successfully reads PMC_BOOT_0 over
  # this protocol on a real eGPU).  Request: struct.pack('<BIIQQQ', cmd, dev_id,
  # bar, *args3); response: struct.unpack('<QQB', ...) = (value1, value2, status)
  # followed by `readout` payload bytes when present.
  MAP_BAR       = 1
  MAP_SYSMEM_FD = 2
  CFG_READ      = 3
  CFG_WRITE     = 4
  RESET         = 5
  SYSMEM_READ   = 9
  SYSMEM_WRITE  = 10
  MMIO_READ     = 6
  MMIO_WRITE    = 7
  MAP_SYSMEM    = 8

class RemoteMMIOInterface(MMIOInterface):
  """MMIO register window that routes reads/writes through the transport
  (no local memoryview — every access is a TinyGPU RPC)."""
  def __init__(self, pci_dev, bar, fmt='B'):
    self.pci_dev, self.bar, self.fmt = pci_dev, bar, fmt
    self.addr, self.nbytes = pci_dev.bar_info(bar)
  def __len__(self): return self.nbytes // struct.calcsize(self.fmt)
  def __getitem__(self, k):
    if isinstance(k, slice):
      start = k.start or 0
      n = (k.stop or self.nbytes) - start
      return self.pci_dev.mmio_read(self.bar, start, n)
    sz = struct.calcsize(self.fmt)
    return struct.unpack_from(self.fmt, self.pci_dev.mmio_read(self.bar, k * sz, sz))[0]
  def __setitem__(self, k, v):
    if self.fmt != 'B' and isinstance(v, (list, tuple)):
      v = b"".join(struct.pack(self.fmt, x) for x in v)
    if isinstance(k, slice):
      start = k.start or 0
      self.pci_dev.mmio_write(self.bar, start, bytes(v) if isinstance(v, (bytes, bytearray)) else v)
    else:
      sz = struct.calcsize(self.fmt)
      self.pci_dev.mmio_write(self.bar, k * sz, struct.pack(self.fmt, v))
  def view(self, offset=0, size=None, fmt=None):
    return RemoteMMIOInterface(self.pci_dev, self.bar, fmt=fmt or self.fmt)
  def read32(self, off): return self.pci_dev.mmio_read32(self.bar, off)
  def write32(self, off, val): return self.pci_dev.mmio_write32(self.bar, off, val)

def _temp_sock():
  # Match the proven examples/add.py client: one stable server/socket is reused
  # across invocations.  Spawning a new DriverKit server on every Kepler run
  # and terminating it during close caused avoidable service detach/rebind
  # churn on the Apple PCIe path.
  # Keep this literal: macOS tempfile.gettempdir() is often a per-user
  # /var/folders path, while the shared TinyGPU server contract is /tmp.
  return "/tmp/tinygpu.sock"

class APLRemotePCIDevice(RemotePCIDevice):
  """macOS: TinyGPU.app signed DriverKit extension exposes raw PCIe BAR access
  for an eGPU over a local Unix socket.  This is a faithful port of the proven
  client in examples/add.py (which drives this same hardware), adapted for
  Kepler bring-up.  The shared server must already own the stable socket; this
  crash-isolation client never starts or terminates the DriverKit service."""
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"
  APP_COMMIT = "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"

  def __init__(self, name="NV", transport=None, dev_id=0, sock_path=None, timeout_ms=2000):
    super().__init__(name, transport or "usb4")
    self.dev_id = dev_id
    self.sock_path = sock_path or os.environ.get("APL_REMOTE_SOCK", _temp_sock())
    self._sock = None
    self._server_proc = None
    self._pci_config_available = False
    self._fini_done = False
    self._sock_lock = threading.Lock()
    self._rpc_seq = 0
    self._rpc_phase = "connect"
    self._rpc_owner_thread = threading.get_ident()
    self._single_thread_rpc = os.environ.get("KEPLER_SINGLE_THREAD_RPC", "1") != "0"
    self._endpoint_frozen = False
    self._endpoint_freeze_reason = None
    self._final_rpc_budget = None
    self._trace_fd = None
    self._trace_light = os.environ.get("KEPLER_RPC_LIGHT", "1") != "0"
    self._trace_mmio_count = 0
    self._trace_mmio_logged = 0
    # Light mode keeps PHASE/FREEZE + non-MMIO RPCs; skips per-dword MMIO
    # BEGIN/END (those were ~2 GiB/run and dominated wall time).
    self._trace_mmio_sample = int(os.environ.get("KEPLER_RPC_MMIO_SAMPLE", "0") or "0")
    trace_path = os.environ.get("KEPLER_RPC_TRACE")
    if trace_path:
      flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
      if hasattr(os, "O_CLOEXEC"): flags |= os.O_CLOEXEC
      self._trace_fd = os.open(trace_path, flags, 0o600)
      mode = "light" if self._trace_light else "full"
      self._trace_record(
          f"TRACE_MODE mode={mode} mmio_sample={self._trace_mmio_sample}")
    try:
      self.set_phase("connect")
      self._connect(timeout_ms)
    except Exception:
      # __init__ failures never reach NVDevice.close().  Roll back this client
      # socket, but leave the shared signed server lifecycle alone.
      self.fini(reset_endpoint=False)
      raise

  def _connect(self, timeout_ms):
    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self._sock.settimeout(timeout_ms / 1000.0)
    connected = False
    for i in range(100):
      try:
        self._sock.connect(self.sock_path); connected = True; break
      except (ConnectionRefusedError, FileNotFoundError):
        time.sleep(0.05)
    if not connected:
      raise RuntimeError(
          f"shared TinyGPU server is not reachable at {self.sock_path}; "
          "start exactly one intended server before the cold live run")

  def _trace_record(self, record):
    fd = getattr(self, "_trace_fd", None)
    if fd is not None:
      data = (record.rstrip("\n") + "\n").encode("utf-8", "backslashreplace")
      while data:
        written = os.write(fd, data)
        if written <= 0:
          raise OSError("RPC flight recorder made no write progress")
        data = data[written:]

  @staticmethod
  def _trace_atom(value):
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace(" ", "\\x20")

  def set_phase(self, phase):
    self._rpc_phase = str(phase)
    self._trace_record(
        f"PHASE monotonic_ns={time.monotonic_ns()} thread={threading.get_ident()} "
        f"phase={self._trace_atom(self._rpc_phase)}")

  def arm_final_output_read(self, bar, offset, size):
    """Allow exactly one final BAR read after the completion semaphore."""
    if self._endpoint_frozen:
      raise RuntimeError(f"cannot arm output read after freeze: {self._endpoint_freeze_reason}")
    self._final_rpc_budget = (int(RemoteCmd.MMIO_READ), int(bar), int(offset), int(size))
    self.set_phase("output-read")

  def freeze(self, reason):
    """Reject every future protocol frame while still allowing local close."""
    if self._endpoint_frozen:
      return
    self._endpoint_frozen = True
    self._endpoint_freeze_reason = str(reason)
    self._trace_record(
        f"FREEZE monotonic_ns={time.monotonic_ns()} thread={threading.get_ident()} "
        f"phase={self._trace_atom(self._rpc_phase)} "
        f"reason={self._trace_atom(self._endpoint_freeze_reason)} "
        f"mmio_count={getattr(self, '_trace_mmio_count', 0)} "
        f"mmio_logged={getattr(self, '_trace_mmio_logged', 0)}")

  def _recvall(self, n):
    buf = bytearray(n)
    got = 0
    while got < n:
      cnt = self._sock.recv_into(memoryview(buf)[got:])
      if cnt == 0: raise ConnectionError("TinyGPU socket closed")
      got += cnt
    return bytes(buf)

  def _rpc(self, cmd, bar, *args, readout=0, payload=b'', has_fd=False):
    if getattr(self, "_endpoint_frozen", False):
      raise RuntimeError(
          f"endpoint RPC after freeze: cmd={cmd} bar={bar} args={args}")
    owner = getattr(self, "_rpc_owner_thread", threading.get_ident())
    if getattr(self, "_single_thread_rpc", False):
      assert threading.get_ident() == owner, (
          f"endpoint RPC from non-owner thread: owner={owner} "
          f"current={threading.get_ident()} cmd={cmd} bar={bar} args={args}")
    cmd_id = int(cmd)
    padded_args = (tuple(args) + (0, 0, 0))[:3]
    offset = int(padded_args[0])
    size = int(padded_args[1])
    budget = getattr(self, "_final_rpc_budget", None)
    if budget == "consumed":
      raise RuntimeError(
          f"endpoint RPC after final output read: cmd={cmd} bar={bar} args={args}")
    if budget is not None:
      attempted = (cmd_id, int(bar), offset, size)
      if attempted != budget:
        raise RuntimeError(
            f"RPC outside final output budget: allowed={budget} attempted={attempted}")
      # Consume before sendall: a failed final read must never be retried.
      self._final_rpc_budget = "consumed"
    with self._sock_lock:
      self._rpc_seq = getattr(self, "_rpc_seq", 0) + 1
      seq = self._rpc_seq
      start_ns = time.monotonic_ns()
      phase = getattr(self, "_rpc_phase", "unknown")
      cmd_name = cmd.name if isinstance(cmd, RemoteCmd) else RemoteCmd(cmd_id).name
      payload_hash = hashlib.sha256(payload).hexdigest() if payload else "-"
      config_offset = offset if cmd_id in (int(RemoteCmd.CFG_READ), int(RemoteCmd.CFG_WRITE)) else "-"
      common = (
          f"seq={seq} monotonic_ns={start_ns} phase={self._trace_atom(phase)} "
          f"thread={threading.get_ident()} cmd={cmd_name} cmd_id={cmd_id} "
          f"bar={bar} offset={offset} size={size} config_offset={config_offset} "
          f"args={self._trace_atom(repr(tuple(args)))} "
          f"readout={readout} payload_len={len(payload)} payload_sha256={payload_hash}")
      is_mmio = cmd_id in (int(RemoteCmd.MMIO_READ), int(RemoteCmd.MMIO_WRITE))
      log_rpc = True
      if is_mmio:
        self._trace_mmio_count = getattr(self, "_trace_mmio_count", 0) + 1
        if getattr(self, "_trace_light", False):
          sample = getattr(self, "_trace_mmio_sample", 0)
          log_rpc = bool(sample and (self._trace_mmio_count % sample) == 0)
          if log_rpc:
            self._trace_mmio_logged = getattr(self, "_trace_mmio_logged", 0) + 1
      if log_rpc:
        self._trace_record(f"BEGIN {common}")
      try:
        self._sock.sendall(struct.pack("<BIIQQQ", cmd_id, self.dev_id, bar,
                                       *padded_args) + payload)
        if payload:  # writes: server sends no response (matches examples/add.py)
          if log_rpc:
            self._trace_record(
                f"END {common} status=ok bytes={len(payload)} "
                f"duration_us={(time.monotonic_ns() - start_ns) // 1000}")
          return None
        if has_fd:
          msg, anc, _, _ = self._sock.recvmsg(17, socket.CMSG_LEN(4))
          fd = struct.unpack('<i', anc[0][2][:4])[0]
        else:
          msg = self._recvall(17); fd = None
        status, value1, value2 = struct.unpack("<BQQ", msg)
        if status != 0:
          err = self._recvall(value1).decode('utf-8') if value1 > 0 else 'unknown error'
          raise RuntimeError(f"TinyGPU RPC cmd={cmd_id} bar={bar} args={args} failed: {err}")
        data = self._recvall(readout) if readout else b""
        if log_rpc:
          self._trace_record(
              f"END {common} status=ok bytes={len(data)} "
              f"duration_us={(time.monotonic_ns() - start_ns) // 1000}")
        return value1, value2, data, fd
      except BaseException as exc:
        # Always record exceptions even in light mode.
        self._trace_record(
            f"END {common} status=exception "
            f"exception={self._trace_atom(type(exc).__name__ + ':' + str(exc))} "
            f"duration_us={(time.monotonic_ns() - start_ns) // 1000}")
        raise

  def bar_info(self, bar):
    v1, v2, _, _ = self._rpc(RemoteCmd.MAP_BAR, bar)
    return (v1, v2)

  def mmio_read(self, bar, offset, size):
    _, _, data, _ = self._rpc(RemoteCmd.MMIO_READ, bar, offset, size, readout=size)
    return data

  def mmio_read32(self, bar, offset):
    return struct.unpack_from("<I", self.mmio_read(bar, offset, 4))[0]

  def mmio_write(self, bar, offset, data):
    self._rpc(RemoteCmd.MMIO_WRITE, bar, offset, len(data), payload=bytes(data))

  def mmio_write32(self, bar, offset, value):
    self.mmio_write(bar, offset, struct.pack("<I", value))

  def map_bar(self, bar, fmt='B', off=0, size=None):
    return RemoteMMIOInterface(self, bar, fmt=fmt)

  def read_config(self, offset, size):
    value = self._rpc(RemoteCmd.CFG_READ, 0, offset, size)[0]
    self._pci_config_available = True
    return value

  def write_config(self, offset, value, size):
    self._rpc(RemoteCmd.CFG_WRITE, 0, offset, size, value)

  def write_config_flush(self, offset, value, size):
    self.write_config(offset, value, size)
    return self.read_config(offset, size)

  def reset(self):
    self._rpc(RemoteCmd.RESET, 0)

  def fini(self, reset_endpoint=False):
    if getattr(self, "_fini_done", False):
      return
    self._fini_done = True
    sock = getattr(self, "_sock", None)
    # Deliberately match the working RTX 3080 client in examples/add.py: close
    # only this client socket.  Do not issue Kepler-only CFG/RESET transactions
    # and do not terminate/relaunch the signed TinyGPU DriverKit server.  The
    # 11:35 panic caught the sole Python thread in recv_into after the FECS
    # thread had been joined, which isolates the rejected request to the extra
    # synchronous PCI shutdown path removed here.
    if sock:
      try:
        sock.close()
      except Exception:
        pass
    self._sock = None
    self._server_proc = None
    trace_fd = getattr(self, "_trace_fd", None)
    if trace_fd is not None:
      try:
        os.close(trace_fd)
      finally:
        self._trace_fd = None

  def __del__(self):
    # Fallback only; normal and partial-constructor paths call fini directly.
    try: self.fini()
    except Exception: pass

  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    """Allocate GPU-visible host memory.  Returns (memoryview, [bus_paddrs])
    via MAP_SYSMEM_FD + recvmsg fd + mmap — CPU-coherent, so copies go straight
    to the mmap (no SYSMEM_READ/WRITE RPCs).  Matches examples/add.py."""
    mapped_size, _, _, fd = self._rpc(RemoteCmd.MAP_SYSMEM_FD, 0, size, int(contiguous), has_fd=True)
    memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt='B')
    paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
    paddrs = [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]
    return memview, paddrs

  def sysmem_read(self, addr, size):
    _, _, data, _ = self._rpc(RemoteCmd.SYSMEM_READ, 0, addr, size, readout=size)
    return data

  def sysmem_write(self, addr, data):
    self._rpc(RemoteCmd.SYSMEM_WRITE, 0, addr, len(data), payload=bytes(data))

  @staticmethod
  def probe(sock_path=None, timeout_ms=500):
    """Return an APLRemotePCIDevice if TinyGPU is reachable, else None."""
    try:
      return APLRemotePCIDevice(sock_path=sock_path, timeout_ms=timeout_ms)
    except (OSError, RuntimeError):
      return None


class _MacPCIDeviceFactory:
  def __call__(self, dev_id=0, **kwargs): return APLRemotePCIDevice(dev_id=dev_id)


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
    resource_path = f"{self.sysfs}/resource{bar}"
    wc_path = f"{resource_path}_wc"
    if (bar == 1 and os.environ.get("KEPLER_BAR1_WC", "1") != "0" and
        os.path.exists(wc_path)):
      resource_path = wc_path
    fd = os.open(resource_path, os.O_RDWR | os.O_SYNC)
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
    self._guard()
    _, ptr, size, _ = self._map_bar(bar)
    if offset & 3 or offset < 0 or offset + 4 > size:
      raise ValueError(f"unaligned/out-of-range BAR{bar} read32 at {offset:#x}")
    if self._final_read_budget is not None and self._final_read_budget != "consumed":
      b, off, sz = self._final_read_budget
      if (bar, offset, 4) != (b, off, sz):
        raise RuntimeError(
            f"read outside final output budget: allowed={self._final_read_budget} "
            f"attempted={(bar, offset, 4)}")
      self._final_read_budget = "consumed"
    # A register access must be one native-width MMIO transaction.  A generic
    # memoryview slice assignment/read may lower to byte copies; split writes
    # are invalid for command registers such as GK104 CHAN_INST.
    return ctypes.c_uint32.from_address(ptr + offset).value

  def mmio_read64(self, offset):
    # NVDevice.read64 passes a BAR0 offset only; combine two 32-bit BAR0 reads.
    lo = self.mmio_read32(0, offset)
    hi = self.mmio_read32(0, offset + 4)
    return (hi << 32) | lo

  def mmio_write(self, bar, offset, data):
    self._guard()
    _, ptr, size, _ = self._map_bar(bar)
    data = memoryview(data).cast("B")
    if offset < 0 or offset + len(data) > size:
      raise ValueError(f"out-of-range BAR{bar} write at {offset:#x}")
    # PCI MMIO is not ordinary RAM: slice assignment may lower to byte stores,
    # which this GK104 demonstrably does not merge into a correct framebuffer
    # qword.  Emit the widest naturally aligned native transaction, while
    # preserving a single 32-bit store for register-like BAR1 doorbells.
    pos = 0
    while pos < len(data):
      addr, remaining = offset + pos, len(data) - pos
      if not (addr & 7) and remaining >= 8:
        ctypes.c_uint64.from_address(ptr + addr).value = \
            struct.unpack_from("<Q", data, pos)[0]
        pos += 8
      elif not (addr & 3) and remaining >= 4:
        ctypes.c_uint32.from_address(ptr + addr).value = \
            struct.unpack_from("<I", data, pos)[0]
        pos += 4
      elif not (addr & 1) and remaining >= 2:
        ctypes.c_uint16.from_address(ptr + addr).value = \
            struct.unpack_from("<H", data, pos)[0]
        pos += 2
      else:
        ctypes.c_uint8.from_address(ptr + addr).value = data[pos]
        pos += 1

  def mmio_write32(self, bar, offset, value):
    self._guard()
    _, ptr, size, _ = self._map_bar(bar)
    if offset & 3 or offset < 0 or offset + 4 > size:
      raise ValueError(f"unaligned/out-of-range BAR{bar} write32 at {offset:#x}")
    ctypes.c_uint32.from_address(ptr + offset).value = value & 0xffffffff

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
# Default matches full GK104 GTX 770 (8 SM).  660 Ti fuse maps are often 7;
# live bring-up overwrites via NVDevice._gk104_mp_count after GR topology read.
GK104_MP_COUNT           = 8
GK104_MAX_WARPS_PER_MP   = 64
GK104_CRS_BYTES_PER_WARP = 0x800
GK104_TEMP_PER_MP = round_up(
    GK104_MAX_WARPS_PER_MP * GK104_CRS_BYTES_PER_WARP, 0x8000)
GK104_TEMP_SIZE = GK104_MP_COUNT * GK104_TEMP_PER_MP
KEPLER_DMA_COPY_A       = 0xa0b5


def _gk104_mp_count_for(dev=None, default=GK104_MP_COUNT) -> int:
  """Return the live SM/MP count (sum of per-GPC TPC enables).

  Mesa's nve4_compute_setup uses ``screen->mp_count`` from GRAPH_UNITS
  (``(mp_count << 8) | gpc_count``), not a hard-coded 8.  GTX 660 Ti fuse
  maps commonly report 7 TPCs; programming TEMP as if there were 8 SMs
  mismatches SKED's view of the floorswept topology.
  """
  env = os.environ.get("KEPLER_MP_COUNT", "").strip()
  if env:
    return max(1, int(env, 0))
  if dev is not None:
    cached = getattr(dev, "_gk104_mp_count", None)
    if cached:
      return int(cached)
    try:
      gpc_nr = dev.read32(0x409604) & 0x1f
      if gpc_nr:
        total = sum(dev.read32(0x500000 + g * 0x8000 + 0x2608) & 0x1f
                    for g in range(gpc_nr))
        if total:
          return total
    except Exception:
      pass
  return default

class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id, software=False):
    self.dev = dev
    if software:
      self.pci_dev = SoftwarePCIDevice(dev_id)
    elif _PCI_TRANSPORT_FACTORY is not None:
      self.pci_dev = _PCI_TRANSPORT_FACTORY(dev_id=dev_id)
    elif OSX:
      self.pci_dev = APLRemotePCIDevice(dev_id=dev_id)
    else:
      self.pci_dev = LinuxPCIDevice(dev_id=dev_id)
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
    self._pmu_memx_data = None
    self._pmu_memx_nowait = False
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

  def _init_pmu_memx(self):
    """Discover the PMU MEMX data segment after the PMU Falcon starts."""
    if self.backend == "software":
      return False
    # gt215_pmu_init() waits for both ring descriptors after starting the
    # Falcon.  Do the same here; an immediate zero is normal while the PMU
    # boot code is still publishing its queue layout.
    deadline = time.monotonic() + 2.0
    while not (self.read32(0x10a4d0) and self.read32(0x10a4dc)):
      if time.monotonic() >= deadline:
        return False
      time.sleep(0.00001)
    data_base, data_size = pmu_send(
        self, PMU_PROC_MEMX, PMU_MEMX_INFO, PMU_MEMX_INFO_DATA, 0)
    if not data_base or not data_size:
      raise RuntimeError(f"invalid GK104 MEMX data segment {data_base:#x}/{data_size:#x}")
    self._pmu_memx_data = (data_base, data_size)
    return True

  def pmu_memx_block(self):
    """FB pause via real MEMX ENTER (requires FB_PAUSE wait patched out).

    Host and MEMX-WR32 stores to 0x1620/0x26f0 kill TinyGPU BAR0.  Stock ENTER
    hangs forever on FB_PAUSE.  With ``KEPLER_PMU_ENTER_NOWAIT``, ENTER applies
    the falcon-side 0x1620 masks and returns.
    """
    if self._pmu_memx_data is None:
      raise RuntimeError("PMU MEMX is unavailable")
    if not getattr(self, "_pmu_memx_nowait", False):
      raise RuntimeError(
          "MEMX ENTER requires KEPLER_PMU_ENTER_NOWAIT (stock FB_PAUSE hang)")
    self.pmu_memx_exec_commands([(PMU_MEMX_ENTER, ())], timeout_s=3.0)

  def pmu_memx_unblock(self):
    """FB unpause via real MEMX LEAVE (requires FB_PAUSE wait patched out)."""
    if self._pmu_memx_data is None:
      raise RuntimeError("PMU MEMX is unavailable")
    if not getattr(self, "_pmu_memx_nowait", False):
      raise RuntimeError(
          "MEMX LEAVE requires KEPLER_PMU_ENTER_NOWAIT (stock FB_PAUSE hang)")
    self.pmu_memx_exec_commands([(PMU_MEMX_LEAVE, ())], timeout_s=3.0)

  def pmu_memx_exec_commands(self, commands, timeout_s=5.0):
    """Run a MEMX script (WR32/DELAY/WAIT/…).  Do not use ENTER on cold eGPU."""
    if self._pmu_memx_data is None:
      raise RuntimeError("PMU MEMX is unavailable")
    return pmu_memx_exec(self, self._pmu_memx_data[0], commands,
                         timeout_s=timeout_s)

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
    # Do NOT treat PMC_BOOT_0=0xffffffff as an immediate physical-replug
    # requirement.  After TinyGPU/DEXT Close(), macOS clears PCI COMMAND
    # Memory Space Enable; a new server that maps BAR0 before restoring MSE
    # will read all-ones even though config space (and the UT3G link) is fine.
    boot0, bar0_meta = _gk104_ensure_bar0_mmio(dev.hw)
    print(
        f"[kepler] BAR0 recover: id={bar0_meta['id32']:#010x} "
        f"cmd={bar0_meta['command_before']:#06x}->{bar0_meta['command_after']:#06x} "
        f"mse_was={bar0_meta['mse_before']} reset={bar0_meta['did_reset']} "
        f"PMC_BOOT_0={boot0:#010x}",
        flush=True)
    # Session lifetime: memory decode + bus master (DMA) + INTx disabled
    # (this userspace path polls).  Recovery above deliberately left MASTER
    # clear until BAR0 MMIO was proven live.
    _pci_command = dev.hw.read_config(pci.PCI_COMMAND, 2)
    _pci_command = ((_pci_command | pci.PCI_COMMAND_MEMORY |
                     pci.PCI_COMMAND_MASTER | pci.PCI_COMMAND_INTX_DISABLE) &
                    ~(pci.PCI_COMMAND_SERR | pci.PCI_COMMAND_PARITY))
    _pci_observed = dev.hw.write_config_flush(pci.PCI_COMMAND, _pci_command, 2)
    _pci_required = pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER
    if (_pci_observed & _pci_required) != _pci_required:
      raise RuntimeError(
          f"PCI memory/bus-master enable did not stick: PCI_COMMAND={_pci_observed:#06x}")
    # Refresh BAR1 after COMMAND is fully armed for the session.
    dev.bar1_addr, dev.bar1_size = dev.hw.bar_info(1)
    dev.hw.set_phase("vbios-devinit")
    # This userspace RM polls every GPU event; no Linux IRQ handler is installed
    # for the GPU.  Match nv04_mc_intr_unarm() and gt215_mc_intr_block() before
    # enabling engines, otherwise a PFIFO/runlist event remains asserted and
    # the kernel may log unhandled interrupts.
    self.write32(0x000140, 0x00000000)  # PMC_INTR_EN_0: unarm external IRQ
    self.write32(0x000640, 0x00000000)  # GK104 MC leaf-0 source mask
    self.read32(0x000140)               # posting read, as Nouveau does
    # 1. Identify + enable engines (boot0 already validated by ensure_bar0).
    if DEBUG: print(f"[kepler] PMC_BOOT_0={boot0:#x}")
    # DMEM probe helper: write+read a test pattern to FECS DMEM[0x100].
    def _dmem_probe(label):
      pat = 0xABCD0000 | (hash(label) & 0xFFFF)
      self.write32(FECS_FALCON_BASE + 0x1c0, 0x01000000 | (0x100 & 0xfffc))
      self.write32(FECS_FALCON_BASE + 0x1c4, pat)
      self.write32(FECS_FALCON_BASE + 0x1c0, 0x02000000 | (0x100 & 0xfffc))
      val = self.read32(FECS_FALCON_BASE + 0x1c4)
      print(f"[kepler] DMEM probe [{label}]: DMEM[0x100]=0x{val:08x} (wrote 0x{pat:08x}) {'OK' if val == pat else 'FAIL'}")
    def _stage_pramin(label):
      if os.environ.get("KEPLER_PRAMIN_STAGE_TRACE", "0") == "1":
        _gk104_pramin_stage_snapshot(self, label)
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
    # Also reject all-ones / 0xbad0…. bus-error sentinels seen when the eGPU
    # link is down or BAR0 is not yet decoded — those must not skip cold init.
    #
    # PGOB alone publishes a live GPC topology (e.g. 0x40004) while PRAMIN still
    # returns the 0xbad0fb stub on a cold eGPU that never ran EFI GDDR training.
    # Treat that as un-POSTed for RAM/devinit so we do not skip VRAM bring-up.
    _gpc_awake = _gk104_topo_is_posted(_gpc_topo)
    # Night41ax/H100: hard PRAMIN probe (0x1700 retarget) *before* BIT-I POST
    # prevents Night41u fixed-PA activation (same class as H80 mid-POST).
    # Soft-live on true-cold (GPC not awake) skips the poke; dirty half-POST
    # (GPC awake) still hard-probes so REFUSE_DIRTY stays honest.
    _pramin_live = _gk104_pramin_looks_live(self, soft=not _gpc_awake)
    _post_owner = _gk104_post_ownership_snapshot(self, _pramin_live)
    _posted = _gpc_awake and _pramin_live
    if _gpc_awake and not _pramin_live:
      _posted_str = "GPC-awake but PRAMIN stub (forcing cold RAM)"
    elif _posted:
      _posted_str = "POSTed"
    else:
      _posted_str = "cold (un-POSTed)"
    print(f"[kepler] GPC topology(0x409604)={_gpc_topo:#x} "
          f"PRAMIN_live={_pramin_live} — card is {_posted_str}", flush=True)
    _stage_pramin("cold ownership")
    # TinyGPU: half-POSTed (GPC awake, PRAMIN stub) after a failed RAM attempt
    # is hostile — cold RAM on that state has hung USB4 / WindowServer.
    # Refuse unless the operator opts in after a real enclosure power cycle.
    if (_gpc_awake and not _pramin_live and
        os.environ.get("KEPLER_REFUSE_DIRTY", "0") == "1" and
        os.environ.get("KEPLER_ALLOW_DIRTY", "0") != "1"):
      raise RuntimeError(
          "GK104 is GPC-awake with dead/stub PRAMIN (dirty after prior MMIO). "
          "Power-cycle the eGPU enclosure (not just USB replug), restart "
          "TinyGPU server, then cold-run add.py once with no probe.  "
          "Set KEPLER_ALLOW_DIRTY=1 only to force cold RAM on this state.")
    # After soft BAR recovery / incomplete power-cycle the topology often
    # reads 0 (not the cold 0xbadf… gate).  One FLR clears a wedged PMU ring
    # so MEMX INFO works again — cheaper than another enclosure cycle.
    if (not _posted and _gpc_topo == 0 and
        os.environ.get("KEPLER_COLD_FLR", "0") != "0" and
        hasattr(dev.hw, "reset")):
      print("[kepler] residual cold (GPC topo=0); PCI FLR before bring-up",
            flush=True)
      try:
        dev.hw.reset()
        time.sleep(1.0)
        boot0, _meta = _gk104_ensure_bar0_mmio(dev.hw)
        self.write32(0x000140, 0x00000000)
        self.write32(0x000640, 0x00000000)
        _gpc_topo = self.read32(0x409604)
        _dmem_probe("after cold FLR")
        print(f"[kepler] after FLR: PMC_BOOT_0={boot0:#x} "
              f"GPC(0x409604)={_gpc_topo:#x}", flush=True)
      except Exception as e:
        print(f"[kepler] cold FLR skipped: {e}", flush=True)
    # On a POSTed card, nouveau already loaded FECS/GPCCS firmware and the
    # falcons are running.  Check if FECS already posted ready (bit31 of 0x409800).
    # If so, we can skip the entire firmware reload + PGRAPH init and just use
    # the running FECS — reloading IMEM on a POSTed falcon corrupts it because
    # the instruction cache/TLB state from the previous run interferes with
    # host IMEM auto-increment writes.
    _fecs_already_ready = _posted and bool(self.read32(0x409800) & 0x80000000)
    if _fecs_already_ready:
      print(f"[kepler] FECS already ready (0x409800={self.read32(0x409800):#x}) — skipping firmware reload", flush=True)

    # Resolve VBIOS / cold-POST intent before PMC enable.  Night41u/H99:
    # fixed-PA PRAMIN activates only when BIT-I POST runs on a virgin card
    # *before* PMC_ENABLE/GR reset (lifecycle order).  Full add used to enable
    # engines first; after-POST fixed-PA then stayed bad0fb and BAR1-after-POST
    # never armed.
    _ram_init_mode = os.environ.get("KEPLER_RAM_INIT", "1")
    if _ram_init_mode not in ("0", "1", "force"):
      raise ValueError("invalid KEPLER_RAM_INIT; expected 0, 1, or force")
    _run_devinit = (not _posted and
                    os.environ.get("KEPLER_VBIOS_DEVINIT", "1") != "0")
    _ram_program_mode = os.environ.get("KEPLER_RAM_PROGRAM", "0")
    _bit0_only = _ram_program_mode == "bit0-only"
    _need_vbios_image = (
        _run_devinit or _ram_init_mode == "force" or _bit0_only or
        (_ram_program_mode != "0" and not _posted))
    image = None
    scripts = None
    if _need_vbios_image:
      # Prefer onboard PROM whenever the cache is missing (SPI flash is still
      # the Gigabyte image even after a foreign BIT-I flipped PCI id to 0x1184).
      vbios_path = _gk104_resolve_vbios_path(
          self, allow_dump=not os.path.exists(ONBOARD_VBIOS_CACHE))
      os.environ["KEPLER_VBIOS"] = vbios_path
      print(f"[kepler] VBIOS image: {vbios_path}", flush=True)
      image, _, scripts = vbios_init_info(vbios_path)

    def _gk104_bar1_after_post_if_live():
      if os.environ.get("KEPLER_BAR1_AFTER_POST", "1") == "0":
        return
      try:
        _pa_word = _gk104_pramin_read32(self, 0xfffe0000) & 0xffffffff
      except Exception as e:
        _pa_word = 0xbad0fb00
        print(f"[kepler] post-POST fixed-PA probe skipped: {e}", flush=True)
      if (not _gk104_pramin_word_is_stub(_pa_word) and
          _pa_word not in (0, 0xffffffff)):
        _map = int(os.environ.get("KEPLER_BAR1_MAP_SIZE", "0x8000000"), 0)
        print(f"[kepler] Nouveau-order BAR1 after POST "
              f"(fixed-PA={_pa_word:#x}, map={_map:#x})", flush=True)
        _saved = {k: os.environ.get(k) for k in (
            "KEPLER_TINYGPU_ATOMIC_BAR1", "KEPLER_PRAMIN_MEMX",
            "KEPLER_PRAMIN_LITERAL", "KEPLER_PRAMIN_LITERAL_FIRST")}
        os.environ["KEPLER_TINYGPU_ATOMIC_BAR1"] = "0"
        os.environ["KEPLER_PRAMIN_MEMX"] = "0"
        # Night41ay: after H79 fixed-PA activation the host aperture is
        # literal; XOR-first on unread 0xffffffff with macOS
        # KEPLER_PRAMIN_LITERAL=0 leaves 0xffffdffe (wanted^ffffffff).
        os.environ["KEPLER_PRAMIN_LITERAL"] = "1"
        os.environ["KEPLER_PRAMIN_LITERAL_FIRST"] = "1"
        try:
          _gk104_init_bar1_identity(
              self, mapped_size=_map, map_vram=True, userd_alias_pa=None)
          print(f"[kepler] BAR1 after POST: 0x1704="
                f"{self.read32(0x001704):#x}", flush=True)
          # Keep classic host-PRAMIN stores (do NOT set _bar1_identity_ready —
          # that diverts through BAR1 and Night41ap wedged USB4).  Still mark
          # so channel-build skips the atomic MEMX pad re-init.
          try:
            setattr(self, "_bar1_classic_posted", True)
            setattr(self, "_bar1_identity_size", _map)
            setattr(self, "_bar1_identity_userd", None)
          except Exception:
            pass
        except Exception as e:
          print(f"[kepler] BAR1 after POST failed (continuing): {e}",
                flush=True)
        finally:
          for k, v in _saved.items():
            if v is None:
              os.environ.pop(k, None)
            else:
              os.environ[k] = v
      else:
        print(f"[kepler] BAR1 after POST skipped "
              f"(fixed-PA={_pa_word:#x})", flush=True)

    _early_post_done = False
    if (_run_devinit and scripts is not None and
        os.environ.get("KEPLER_POST_BEFORE_PMC", "1") != "0"):
      print(f"[kepler] VBIOS POST before PMC (script0={scripts[0]:#x})",
            flush=True)
      nvbios_init.run_vbios_init(
          self, image, scripts,
          debug=bool(DEBUG and getenv("KEPLER_VBIOS_TRACE", 0)))
      print(f"[kepler] after VBIOS POST: PLL(0x137000)={self.read32(0x137000):#x} "
            f"GPC(0x409604)={self.read32(0x409604):#x} "
            f"0x2240c={self.read32(0x2240c):#x} "
            f"0x11e338={self.read32(0x11e338):#x}", flush=True)
      _dmem_probe("after VBIOS POST (pre-PMC)")
      _stage_pramin("after VBIOS POST (pre-PMC)")
      _gk104_bar1_after_post_if_live()
      program_gk104_gpc_pll(self)
      _dmem_probe("after GPC PLL (pre-PMC)")
      _early_post_done = True

    # Enable the engines compute needs (nouveau gk104_mc reset/enable bits):
    # PGRAPH=0x1000 (bit12), PFIFO=0x100, PFB=0x08002000, LTC=0x02000000.
    # Avoid 0xffffffff (would arm engines whose firmware isn't loaded yet).
    # Enable engines.  Kepler GK104 needs the full supported engine set enabled
    # to exit power-gating — a minimal subset (PGRAPH|PFIFO|PFB|LTC) leaves
    # FECS/PGRAPH registers returning the 0xbad0da1f "engine disabled" sentinel.
    # Write 0xffffffff and let the GPU mask to its supported engines.
    # nvkm_fifo_preinit() gives PFIFO its own posted MC reset.  Keep it separate
    # from GR: the golden trace toggles 0x100 alone, and combining both reset
    # domains into a single 0x1100 transition is not equivalent internally.
    if os.environ.get("KEPLER_FIFO_RESET") == "1":
      nvkm_mask(self, 0x000200, 0x00000100, 0x00000000)
      self.read32(0x000200)
      nvkm_mask(self, 0x000200, 0x00000100, 0x00000100)
      self.read32(0x000200)
    # Toggle GR engine reset (bit 12) to clear stale FECS state from previous
    # failed runs.  Disable PGRAPH first (like nouveau gf100_gr_reset), then
    # toggle the GR MC domain independently.
    if os.environ.get("KEPLER_GR_RESET") != "0":
      nvkm_mask(self, 0x400500, 0x00000001, 0x00000000)  # disable PGRAPH
      self.read32(0x400500)
      time.sleep(0.001)
      nvkm_mask(self, 0x000200, 0x00001000, 0x00000000)
      self.read32(0x000200)  # posted
      time.sleep(0.05)
      nvkm_mask(self, 0x000200, 0x00001000, 0x00001000)
      self.read32(0x000200)  # posted
      time.sleep(0.05)
      self.write32(0x400500, 0x00010001)  # re-enable PGRAPH
      self.read32(0x400500)
      time.sleep(0.01)
    # Nouveau's general engine-reset pass resets PFIFO again after the GR/TOP
    # domains.  Its trace contains two distinct 0x000200 bit-8 toggles before
    # FIFO init; retain that ordering rather than resetting a live FIFO later.
    if os.environ.get("KEPLER_FIFO_RESET") == "1":
      nvkm_mask(self, 0x000200, 0x00000100, 0x00000000)
      self.read32(0x000200)
      nvkm_mask(self, 0x000200, 0x00000100, 0x00000100)
      self.read32(0x000200)
    self.write32(PMC_ENABLE, 0xffffffff)
    _dmem_probe("after PMC_ENABLE")
    _stage_pramin("after PMC_ENABLE")
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
    # gated and FECS waits forever for topology.  Cold path already ran BIT-I
    # POST before PMC when KEPLER_POST_BEFORE_PMC=1 (_early_post_done).
    if _run_devinit and not _early_post_done:
      print(f"[kepler] VBIOS direct devinit script0={scripts[0]:#x}")
      nvbios_init.run_vbios_init(
          self, image, scripts,
          debug=bool(DEBUG and getenv("KEPLER_VBIOS_TRACE", 0)))
      print(f"[kepler] after VBIOS devinit: PLL(0x137000)={self.read32(0x137000):#x} "
            f"GPC(0x409604)={self.read32(0x409604):#x} "
            f"PGRAPH_STATUS={self.read32(0x400000):#x} "
            f"PGRAPH_CTRL={self.read32(0x400500):#x} "
            f"PGRAPH_INTR={self.read32(0x400100):#x}", flush=True)
      _dmem_probe("after VBIOS devinit")
      _stage_pramin("after VBIOS devinit")
      _gk104_bar1_after_post_if_live()
      program_gk104_gpc_pll(self)
      _dmem_probe("after GPC PLL")
    elif _early_post_done:
      print("[kepler] skipping post-PMC VBIOS POST (already ran pre-PMC)",
            flush=True)
    dev.hw.set_phase("firmware-load")
    def _rd(name):
      p = os.path.join(fdir, name)
      if not os.path.exists(p):
        raise NotImplementedError(f"NVDevice._init_hardware: missing firmware {p}")
      return open(p, "rb").read()
    # GPC/ROP power-gate release needs the PMU alive; best-effort PMU bring-up.
    # Skip PMU/PGOB only when FECS already posted ready (true firmware POST).
    # Night41ao: PMC-only ungate can make topo+PRAMIN look "POSTed" without
    # PMU/PGOB; skipping then leaves FECS in wait_donez and GPCs STOPPED.
    if not _fecs_already_ready:
      _dmem_probe("before PMU")
      _pmu_started = False
      try:
        pmu_code = _rd("gk104_pmu_code.bin"); pmu_data = _rd("gk104_pmu_data.bin")
        # Cold TinyGPU never raises FB_PAUSE; stock ENTER/LEAVE hang forever.
        # Default: patch the wait loops so real MEMX ENTER can apply 0x1620
        # pause from the falcon (host/MEMX-WR32 of those regs kill the link).
        if os.environ.get("KEPLER_PMU_ENTER_NOWAIT", "1") != "0":
          pmu_code = _patch_pmu_memx_nowait(pmu_code)
          self._pmu_memx_nowait = True
          print("[kepler] PMU MEMX ENTER/LEAVE FB_PAUSE wait patched out",
                flush=True)
        else:
          self._pmu_memx_nowait = False
        pmu_code = _gk104_pmu_embed_bar1_bootstrap(pmu_code)
        self._pmu_code = bytes(pmu_code)
        # Only hard-reset the falcon when bringing up a residual/wedged PMU.
        # On true-cold (0xbadf…) a fresh load after power-cycle does not need
        # it; resetting first has been correlated with flaky MEMX INFO.
        if (_gpc_topo == 0 or
            os.environ.get("KEPLER_PMU_FORCE_RESET", "0") != "0"):
          falcon_reset(self, PMU_FALCON_BASE)
        falcon_load(self, PMU_FALCON_BASE, pmu_code, pmu_data, entry=0, start=True)
        _pmu_started = True
        # Cold PMU host rings need a moment after START before MEMX INFO.
        time.sleep(0.05)
        if DEBUG: print("[kepler] PMU firmware loaded + started")
      except Exception as e:
        if DEBUG: print(f"[kepler] PMU load skipped: {e}")
      _dmem_probe("after PMU")
      gk104_pmu_pgob(self)
      _dmem_probe("after pgob")
      _stage_pramin("after PMU/PGOB")
      if DEBUG:
        print(f"[kepler] after pgob: gpc/rop(0x409604)={self.read32(0x409604):#x} "
              f"GPC0 CTRL(0x502100)={self.read32(0x502100):#x} "
              f"GPCCS CTRL(0x41a100)={self.read32(0x41a100):#x}")
      if _pmu_started and os.environ.get("KEPLER_PMU_MEMX", "1") != "0":
        try:
          if self._init_pmu_memx():
            print("[kepler] PMU MEMX transport ready", flush=True)
        except Exception as e:
          print(f"[kepler] PMU MEMX unavailable after load: {e}; retrying once",
                flush=True)
          time.sleep(0.1)
          try:
            self._pmu_memx_data = None
            if self._init_pmu_memx():
              print("[kepler] PMU MEMX transport ready (retry)", flush=True)
            else:
              print("[kepler] PMU MEMX still unavailable after retry", flush=True)
          except Exception as e2:
            print(f"[kepler] PMU MEMX unavailable after load: {e2}", flush=True)
      # Half-POST (topo awake, FECS not ready): still need GPC PLL if VBIOS
      # clock programming was skipped with _run_devinit.
      if _posted and not _run_devinit:
        try:
          program_gk104_gpc_pll(self)
          _dmem_probe("after GPC PLL (half-POST)")
        except Exception as e:
          print(f"[kepler] GPC PLL on half-POST skipped: {e}", flush=True)
    else:
      print("[kepler] skipping PMU/PGOB (FECS already ready)", flush=True)
    # Nouveau's gk104_ram_init() and first memory-clock transition run only
    # after PMU/PGOB has made the framebuffer domains usable.  On a cold eGPU
    # this ordering is essential: programming 0x10f*/0x132* while the PMU is
    # still absent can acknowledge the RPC yet leave GDDR5 electrically
    # untrained.  Keep the phase before FECS/PGRAPH allocations, but after the
    # PMU power-domain bring-up.
    _ram_program_mode = os.environ.get("KEPLER_RAM_PROGRAM", "0")
    _bit0_only = _ram_program_mode == "bit0-only"
    _want_ram = (
        image is not None and (
            _bit0_only or
            (_ram_init_mode != "0" and (_run_devinit or _ram_init_mode == "force"))))
    if _want_ram:
      _ram_debug = bool(DEBUG and getenv("KEPLER_VBIOS_TRACE", 0))
      # getenv() int-parses numeric env values, so compare as strings via
      # os.environ — otherwise KEPLER_RAM_PROGRAM=0 becomes int 0 and
      # ``0 != "0"`` is True, silently forcing ram_program back on.
      _ram_program = _ram_program_mode != "0"
      # Nouveau's known-good cold baseline for this Palit ROM is 648 MHz
      # memory (see nouveau_gk104_trace.txt); stay in RAMMAP entry 2 until a
      # durable BAR1 read proves a higher-frequency transition.
      _ram_freq = int(os.environ.get("KEPLER_RAM_FREQ", "648"))
      _skip_ram_init = (
          _bit0_only and os.environ.get("KEPLER_RAM_AFTER_BIT0") != "memx")
      if _ram_init_mode != "0" and not _skip_ram_init:
        print("[kepler] running GK104 VBIOS RAMMAP/GDDR5 initialization",
              flush=True)
        nvbios_init.run_vbios_ram_init(self, image, debug=_ram_debug)
        # Night41b/H49: Nouveau performs this FB ram_init phase independently
        # of the later clock pstate calc/prog loop.  Sample the exact boundary
        # without aborting so one cold run can classify whether training was
        # already stuck before our optional reclock transition.
        if not hasattr(self, "ops"):
          _gk104_dump_ram_mc_regs(
              self, label="after ram_init, before ram_program",
              strict=not _ram_program)
          _stage_pramin("after RAM init")
        # Re-discover MEMX after RAMMAP: heavy 0x10f* traffic can leave the
        # host-command path needing a fresh INFO before WR32 buffering.
        if os.environ.get("KEPLER_PMU_MEMX", "1") != "0":
          try:
            self._pmu_memx_data = None
            if self._init_pmu_memx():
              print("[kepler] PMU MEMX re-ready before ram_program", flush=True)
          except Exception as e:
            print(f"[kepler] PMU MEMX re-init failed: {e}; trying PMU reload",
                  flush=True)
            try:
              pmu_code = _rd("gk104_pmu_code.bin")
              pmu_data = _rd("gk104_pmu_data.bin")
              if os.environ.get("KEPLER_PMU_ENTER_NOWAIT", "1") != "0":
                pmu_code = _patch_pmu_memx_nowait(pmu_code)
                self._pmu_memx_nowait = True
              pmu_code = _gk104_pmu_embed_bar1_bootstrap(pmu_code)
              self._pmu_code = bytes(pmu_code)
              falcon_reset(self, PMU_FALCON_BASE)
              falcon_load(self, PMU_FALCON_BASE, pmu_code, pmu_data,
                          entry=0, start=True)
              self._pmu_memx_data = None
              if self._init_pmu_memx():
                print("[kepler] PMU MEMX recovered after reload", flush=True)
            except Exception as e2:
              print(f"[kepler] PMU MEMX recovery failed: {e2}", flush=True)
      if _ram_program:
        # Night40ah: Nouveau memx.fuc ENTER waits forever on OUTPUT FB_PAUSE
        # before any GDDR WR32.  Live default keeps ENTER_NOWAIT for early MEMX,
        # then reloads stock waits for the atomic ram_program script.
        _enter_wait = (
            os.environ.get("KEPLER_RAM_ENTER_WAIT", "0") != "0" and
            not hasattr(self, "ops"))
        _ram_ok = False
        if _enter_wait and getattr(self, "_pmu_memx_nowait", False):
          try:
            # Preflight ENTER+LEAVE under stock wait can hang before MC train;
            # the indivisible RAM script itself is the Nouveau experiment.
            os.environ["KEPLER_RAM_ATOMIC_PREFLIGHT"] = "0"
            os.environ.setdefault("KEPLER_RAM_MEMX_ATOMIC", "1")
            os.environ.setdefault("KEPLER_RAM_BLOCK", "atomic")
            _gk104_pmu_reload_for_ram(
                self, nowait=False, reason="night40ah Nouveau ENTER wait")
          except Exception as e:
            print(f"[kepler] ENTER-wait PMU reload failed ({e}); "
                  f"keeping nowait for ram_program", flush=True)
            _enter_wait = False
        print(f"[kepler] programming GK104 cold GDDR5 controller at {_ram_freq} MHz"
              f"{' (atomic + Nouveau FB_PAUSE wait)' if _enter_wait else ''}",
              flush=True)
        try:
          nvbios_init.run_vbios_ram_program(
            self, image, freq_mhz=_ram_freq, debug=_ram_debug)
          _ram_ok = True
        except (TimeoutError, RuntimeError) as e:
          if not _enter_wait:
            raise
          print(f"[kepler] ram_program with ENTER wait failed ({e}); "
                f"reloading nowait and retrying once", flush=True)
          _gk104_pmu_reload_for_ram(
              self, nowait=True, reason="night40ah ENTER-wait timeout fallback")
          os.environ["KEPLER_RAM_BLOCK"] = "atomic"
          os.environ["KEPLER_RAM_MEMX_ATOMIC"] = "1"
          nvbios_init.run_vbios_ram_program(
            self, image, freq_mhz=_ram_freq, debug=_ram_debug)
          _ram_ok = True
        if _ram_ok and not hasattr(self, "ops"):
          try:
            _gk104_dump_ram_mc_regs(self)
          except RuntimeError:
            # H14 train-status strict (and similar) must abort bring-up.
            raise
          except Exception as e:
            print(f"[kepler] MC dump skipped: {e}", flush=True)
      if not _gk104_pramin_looks_live(self):
        # Soft live never pokes PRAMIN; only report boot0 here (a 0x1700
        # window read after bit0 has killed TinyGPU BAR0).
        try:
          boot0 = self.read32(0) & 0xffffffff
          boot0_s = f"{boot0:#x}"
        except Exception as e:
          boot0_s = f"<read err {e}>"
        sample = "<skipped>"
        if os.environ.get("KEPLER_PRAMIN_SOFT_LIVE", "1") == "0":
          try:
            sample = hex(_gk104_pramin_read32(self, 0x100000) & 0xffffffff)
          except Exception as e:
            sample = f"<pramin err {e}>"
        # Offline FakeMMIO recorders used by mmiotrace_selftest have no FB.
        if hasattr(self, "ops"):
          print(f"[kepler] PRAMIN stub after cold RAM on recorder "
                f"(sample={sample}); continuing offline", flush=True)
        else:
          raise RuntimeError(
              f"PRAMIN not usable after cold RAM (sample={sample} "
              f"boot0={boot0_s} soft_live="
              f"{os.environ.get('KEPLER_PRAMIN_SOFT_LIVE', '1')!r}); "
              "VRAM aperture is not usable for channel instance stores.  "
              "Replug for a clean cold POST with working PMU MEMX; "
              "refusing to continue into FECS/channel bring-up")
      else:
        soft = os.environ.get("KEPLER_PRAMIN_SOFT_LIVE", "1") != "0"
        if soft:
          if os.environ.get("KEPLER_RAM_BIT0_DEFER", "0") == "1":
            print("[kepler] PRAMIN soft-accept after cold RAM "
                  "(boot0 live; bit0 deferred until first PRAMIN store)",
                  flush=True)
          else:
            print("[kepler] PRAMIN soft-accept after cold RAM "
                  "(boot0 live; skipped PRAMIN window poke)", flush=True)
        else:
          print("[kepler] PRAMIN live after cold RAM (writeback ok)", flush=True)
    # Golden mmiotrace: after RAMMAP/training, Nouveau does fb_init_page
    # (0x100c80) then LTC/ZBC (0x17ea*/0x17e8*), ~20s before FECS.  Night10
    # collapsed TinyGPU BAR0 when this ran *after* bit0; with BIT0_DEFER the
    # macOS live default is KEPLER_POST_RAM_LTC=1 (before bit0).  Set 0 to skip.
    if os.environ.get("KEPLER_POST_RAM_LTC", "1") != "0":
      _gk104_post_ram_fb_ltc(self)
      _gk104_require_bar0_live(self, "after post-RAM LTC/ZBC")
      _stage_pramin("after FB/LTC init")
    else:
      print("[kepler] skipping post-RAM fb_init_page/LTC/ZBC "
            "(KEPLER_POST_RAM_LTC=0; TinyGPU-hostile after bit0)", flush=True)
      _gk104_require_bar0_live(self, "after cold RAM (LTC skipped)")
    fecs_code = bytearray(_rd("gk104_fecs_code.bin")); fecs_data = _rd("gk104_fecs_data.bin")
    gpccs_code = _rd("gk104_gpccs_code.bin"); gpccs_data = _rd("gk104_gpccs_data.bin")
    # Historical diagnostic: patch ctx_4170s from `or $r15 0x10` to
    # `or $r15 0x12`.
    # ctx_4170s (at Falcon addr 0x7db) ORs $r15 with 0x10 and writes to FE_PWR
    # (0x404170).  Bit 4 (0x10) = power-state request; bit 1 (0x02) = FORCE_ON.
    # Several callers pass $r15=0 (via `clear b32 $r15`), so the diagnostic
    # made every firmware transition request FORCE_ON.  That diverges from
    # Nouveau and the golden trace, which run the GR ICMD sequence with the
    # stock ctx_4170s helper and FE power in AUTO.  Keep the patch opt-in now
    # that Linux BAR0 accesses use native-width transactions.
    # The instruction `or $r15 0x10` is 3 bytes (f0 f5 10) at 0x7db-0x7dd;
    # we change byte 0x7dd from 0x10 to 0x12.  No length change, no branch
    # offset shifts — safe for variable-length Falcon ISA.
    # NOTE: the previous approach patched a 4-byte word at 0xa44, but Falcon
    # uses variable-length instructions, so 0xa44 spans `call 0x802` (4B at
    # 0xa41) and `clear b32 $r15` (2B at 0xa45), corrupting both.
    if os.environ.get("KEPLER_FECS_FORCE_ON_PATCH") == "1":
      fecs_code[0x7dd] = 0x12  # or $r15 0x10 -> or $r15 0x12
    if DEBUG:
      print(f"[kepler] FECS ctx_4170s FORCE_ON patch: "
            f"{'enabled' if fecs_code[0x7dd] == 0x12 else 'disabled'}")
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
    # Pre-bake the autonomous BAR1 bootstrap into the IMEM zero pad.  Night27
    # showed that runtime CODE writes at 0xb20 read back 0xbadf5000 once FECS
    # has already run, so the routine must ride along with the initial load.
    fecs_code = _gk104_fecs_embed_bar1_bootstrap(fecs_code)
    fecs_code = bytes(fecs_code)
    # Cache firmware for FECS reload after pgob (FECS power-gates and pgob
    # destroys it; we need to reload firmware to restore FECS functionality).
    self._fecs_code = fecs_code
    self._fecs_data = fecs_data
    _gk104_require_bar0_live(self, "before clock/PGRAPH diag")
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
    # TinyGPU (2026-07-16 careful N=8): after post-MEMX bit0, *any* host
    # write to 0x4041f0 (even a no-op clear of bit6 when already 0) collapses
    # BAR0 to all-ones.  Skip BLCG poke unless KEPLER_PGRAPH_BLCG=1.
    _blcg_main = self.read32(0x4041f0)
    if os.environ.get("KEPLER_PGRAPH_BLCG", "1") == "0":
      print(f"[kepler] BLCG main: skipping 0x4041f0 write "
            f"(was {_blcg_main:#x}; KEPLER_PGRAPH_BLCG=0)", flush=True)
    elif (_blcg_main & 0xfffff000) != 0xbadf0000:
      self.write32(0x4041f0, _blcg_main & ~0x40)  # clear bit 6
      _gk104_require_bar0_live(self, "after BLCG main 0x4041f0 write")
      print(f"[kepler] BLCG main: 0x4041f0 was {_blcg_main:#x} now {self.read32(0x4041f0):#x}",
            flush=True)
    else:
      print(f"[kepler] BLCG main: 0x4041f0 GATED ({_blcg_main:#x})", flush=True)
    _gk104_require_bar0_live(self, "before PGRAPH pack MMIO")
    # TinyGPU: full GK104_PGRAPH_PACK_MMIO after bit0 collapses BAR0.  Even
    # without bit0 the pack is heavy; KEPLER_PGRAPH_PACK=0 keeps only the
    # FECS clock-gate + master enable that this cold path needs.
    _do_pack = os.environ.get("KEPLER_PGRAPH_PACK", "1") != "0"
    if not _posted:
      _before = self.read32(0x400500)
      self.write32(0x400500, _before & ~0x00010001)
      if _do_pack:
        for addr, val in GK104_PGRAPH_PACK_MMIO:
          self.write32(addr, val)
      else:
        print("[kepler] PGRAPH pack: minimal (KEPLER_PGRAPH_PACK=0)", flush=True)
      self.write32(0x409890, 0x00000045)
      self.write32(0x4098b0, 0x0000007f)
      self.write32(0x400500, 0x00010001)
    else:
      if _do_pack:
        for addr, val in GK104_PGRAPH_PACK_MMIO:
          self.write32(addr, val)
      else:
        print("[kepler] PGRAPH pack: minimal (KEPLER_PGRAPH_PACK=0)", flush=True)
      self.write32(0x409890, 0x00000045)
      self.write32(0x4098b0, 0x0000007f)
    _gk104_require_bar0_live(self, "after PGRAPH pack MMIO")
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
    tpc_total = sum(tpc_nr)
    if gpc_nr == 0 or tpc_total == 0:
      raise RuntimeError(
          f"GK104 GR topology unusable after init: gpc_nr={gpc_nr} tpc_nr={tpc_nr} "
          f"topo={self.read32(0x409604):#x}")
    try:
      setattr(self, "_gk104_mp_count", tpc_total)
    except Exception:
      pass
    print(f"[kepler] live MP/TPC count={tpc_total} (gpc={gpc_nr} tpc_nr={tpc_nr})",
          flush=True)
    row, tile = _gk104_grctx_tiles(tpc_nr)
    # VSC stream master + GF117 zcull setup.
    self.write32(0x503018, 1)
    bank = [0] * gpc_nr
    zdata = 0
    for i, gpc in enumerate(tile[:tpc_total]):
      if not (0 <= gpc < gpc_nr):
        raise RuntimeError(f"invalid GK104 zcull tile gpc={gpc} gpc_nr={gpc_nr}")
      zdata |= (bank[gpc] & 0xf) << ((i & 7) * 4); bank[gpc] += 1
      if (i & 7) == 7 or i == tpc_total - 1:
        self.write32(0x418980 + (i // 8) * 4, zdata & 0xffffffff); zdata = 0
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
    # TinyGPU: host writes to 0x4041f0 (and the bulk BLCG list) after bit0
    # collapse BAR0 — skip unless KEPLER_PGRAPH_BLCG=1.
    if os.environ.get("KEPLER_PGRAPH_BLCG", "1") == "0":
      print("[kepler] BLCG disabled: skipped (KEPLER_PGRAPH_BLCG=0)", flush=True)
    else:
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
      _gk104_require_bar0_live(self, "after BLCG bulk clear")
    _gk104_require_bar0_live(self, "before falcon load")
    print(f"[kepler] before falcon load: FECS_CTRL={self.read32(0x409100):#x} GPCCS_CTRL={self.read32(0x41a100):#x} PGRAPH_CTRL={self.read32(0x400500):#x} PGRAPH_STATUS={self.read32(0x400000):#x} RED_SWITCH={self.read32(0x409614):#x}")
    # ponytail: If the FECS is stuck in an EFI overlay from a previous failed
    # run, reset the falcon via CPUCTL RESET bit (bit 7).  The GR reset (PMC
    # bit 12 toggle) doesn't reset the falcon PC — it preserves IMEM/ITLB/PC.
    # The falcon RESET clears the PC and ITLB, allowing a clean firmware reload.
    #
    # envytools falcon.xml uc_cpuctl: bit4=STOPPED, bit5=SLEEPING.  STOPPED is
    # the normal park state after FECS posts ready — NOT a wedge.  The old
    # `CPUCTL & 0x10` check forced IMEM reload+restart on every warm open and
    # left FECS_MMIO_CTRL=0x40404170 (FE_PWR address leaked) with sticky GPC
    # busy on SET_OBJECT.  Healthy warm keeps GPCCS_CTRL=0x10 and
    # FECS_MMIO_CTRL in the 0x005xxxxx GPC window.
    _fecs_pc = self.read32(0x409ff0)
    _fecs_cpuctl_pre = self.read32(FECS_FALCON_BASE + 0x100)
    _fecs_mmio_ctrl = self.read32(0x409728)
    _fecs_scratch0_now = self.read32(0x409800)
    _fecs_still_ready = bool(_fecs_scratch0_now & 0x80000000)
    _fecs_efi_overlay = _fecs_pc > 0x8000
    # Healthy FECS MMIO window points into GPC space (0x005xxxxx), e.g.
    # 0x00502800 / 0x0051a800.  0 and 0x40404170 (FE_PWR address leaked) both
    # mean ctxctl cannot talk to GPCs — must reload+start.
    _fecs_mmio_sane = ((_fecs_mmio_ctrl & 0xfff00000) == 0x00500000)
    _force_reload = os.environ.get("KEPLER_FORCE_RELOAD") == "1"
    # Re-check ready at falcon-load time: early _fecs_already_ready can go
    # stale after the second PGRAPH/pgob path clears SCRATCH0 while leaving
    # CPUCTL=STOPPED, which previously warm-kept a dead FECS.
    _need_falcon_reload = (
        (not _posted) or (not _fecs_still_ready) or _force_reload
        or _fecs_efi_overlay or (not _fecs_mmio_sane))
    if _posted and _fecs_efi_overlay:
      print(f"[kepler] FECS EFI-overlay wedge (PC={_fecs_pc:#x} CPUCTL={_fecs_cpuctl_pre:#x}), resetting falcon", flush=True)
      self.write32(FECS_FALCON_BASE + 0x100, 0x00000080)  # RESET
      time.sleep(0.1)
      self.write32(FECS_FALCON_BASE + 0x100, 0x00000000)  # release reset
      time.sleep(0.1)
      _pc_after = self.read32(0x409ff0)
      _cpuctl_after = self.read32(FECS_FALCON_BASE + 0x100)
      print(f"[kepler] After falcon reset: PC={_pc_after:#x} CPUCTL={_cpuctl_after:#x}", flush=True)
    elif not _fecs_mmio_sane:
      print(f"[kepler] FECS_MMIO_CTRL not sane ({_fecs_mmio_ctrl:#x}); forcing falcon reload",
            flush=True)
    elif not _fecs_still_ready:
      print(f"[kepler] FECS not ready (SCRATCH0={_fecs_scratch0_now:#x} "
            f"CPUCTL={_fecs_cpuctl_pre:#x} MMIO_CTRL={_fecs_mmio_ctrl:#x}); "
            f"loading+starting falcons", flush=True)
    else:
      print(f"[kepler] FECS warm-keep (PC={_fecs_pc:#x} CPUCTL={_fecs_cpuctl_pre:#x} "
            f"MMIO_CTRL={_fecs_mmio_ctrl:#x} GPCCS={self.read32(0x41a100):#x})",
            flush=True)
    # Clear ALL stale code TLB entries from previous runs.  ITLB(physidx)
    # clears the TLB entry for a physical page [envytools: falcon/vm.html].
    # Without this, duplicate virt->phys mappings cause multihit faults.
    # Clear ITLB only when we are about to reload IMEM.  Blasting TLB entries
    # on a warm-kept FECS that is already STOPPED parks the wrong mappings.
    if _posted and _need_falcon_reload:
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
    # Skip the reload when FECS is already ready with a sane MMIO window.
    if _need_falcon_reload:
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
      self.write32(0x409800, 0x00000000)
      # FECS-only start after reload (matches add-660ti-013040 which got
      # FECS_MMIO_CTRL=0x0051a800).  Host-starting GPCCS here made
      # discover_image_size return 0 with mailbox stuck at 0x8c000.
      _cpuctl_after_reset = self.read32(FECS_FALCON_BASE + 0x100)
      print(f"[kepler] FECS CPUCTL before start: 0x{_cpuctl_after_reset:08x}", flush=True)
      self.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, 0x00000000)
      self.write32(FECS_FALCON_BASE + 0x004, 0xffffffff)
      self.write32(FECS_FALCON_BASE + 0x014, 0xffffffff)
      self.write32(FECS_FALCON_BASE + 0x10c, 0x00000000)
      self.write32(FECS_FALCON_BASE + FALCON_UC_ENTRY, 0)
      self.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, FALCON_UC_CTRL_START)
      print(f"[kepler] FECS started immediately after DMEM+csdata load")
      print(f"[kepler] after FECS start: FECS_CTRL=0x{self.read32(0x409100):08x} FECS_SIGNAL=0x{self.read32(0x409400):08x} GPCCS_CTRL=0x{self.read32(0x41a100):08x} FECS_MMIO_BASE=0x{self.read32(0x409724):08x} FECS_MMIO_CTRL=0x{self.read32(0x409728):08x} ACCESS_EN=0x{self.read32(0x409048):08x} INTR=0x{self.read32(0x409008):08x} IRQMSET=0x{self.read32(0x409010):08x} IRQMASK=0x{self.read32(0x409018):08x} HWCFG2=0x{self.read32(0x40916c):08x} CPUSTAT=0x{self.read32(0x409128):08x} XFER_STATUS=0x{self.read32(0x409120):08x}")
      _fe_pwr_after_start = self.read32(0x404170)
      print(f"[kepler] FE_PWR after FECS start: {_fe_pwr_after_start:#x}", flush=True)
      import time as _time
      pcs = []
      for _ in range(10):
        pcs.append(self.read32(0x409ff0))
        _time.sleep(0.001)
      print(f"[kepler] FECS PC samples after start: {[hex(p) for p in pcs]}")
      for gpc in range(4):
        gpc_base = 0x502000 + gpc * 0x8000
        print(f"[kepler] GPC{gpc} fuc: CTRL={self.read32(gpc_base+0x100):#x} ENTRY={self.read32(gpc_base+0x104):#x} "
              f"SCRATCH0={self.read32(gpc_base+0x800):#x} SCRATCH1={self.read32(gpc_base+0x804):#x} "
              f"BLOCK={self.read32(gpc_base+0x10c):#x}")
      def _fecs_ready_with_gpc3_workaround():
        if falcon_ready(self, FECS_FALCON_BASE):
          return True
        if self.read32(0x51a804) != 0 and not (self.read32(0x51a800) & 0x80000000):
          _time.sleep(0.05)
          if not (self.read32(0x51a800) & 0x80000000):
            print("[kepler] GPC3 SCRATCH0 not set — applying workaround", flush=True)
            self.write32(0x51a820, 0x80000000)
            _time.sleep(0.01)
        return falcon_ready(self, FECS_FALCON_BASE)
      try:
        wait_cond(_fecs_ready_with_gpc3_workaround,
                  timeout_ms=10000, msg="FECS ready (0x409800 bit31)")
      except TimeoutError:
        rd = lambda o: self.read32(o)
        print(f"[kepler] FECS NOT ready. CPUCTL(0x409100)={rd(0x409100):#x} "
              f"MB0(0x409800)={rd(0x409800):#x} MMIO_CTRL(0x409728)={rd(0x409728):#x} "
              f"GPCCS={rd(0x41a100):#x}", flush=True)
        raise
    else:
      # Warm-keep: FECS already ready with sane MMIO_CTRL; do not reload/restart.
      pass
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
    # Wait out transient FECS MMIO (e.g. WRITE to FE_PWR → 0x40404170) before
    # discover; method 0x10 returns 0 while the window is not in GPC space.
    for _ in range(50):
      _mc = self.read32(0x409728)
      if (_mc & 0xfff00000) == 0x00500000 or _mc == 0:
        break
      time.sleep(0.002)
    _mailbox_size = self.read32(0x409804)
    # H9: always run Nouveau gf100_gr_fecs_discover_image_size (method 0x10 →
    # size in 0x409800).  Mailbox 0x409804 alone can be stale on warm-keep or
    # wrong for this die's floorsweep; GTX 770 notes used 0x2c400 while 660 Ti
    # often reports 0x29b00 (7 SM).  Override with KEPLER_GR_CTX_SIZE=0x….
    _ready = self.read32(0x409800)
    self.write32(0x409800, 0x00000000)
    self.write32(0x409500, 0x00000000)
    self.write32(0x409504, 0x00000010)
    ctx_size = 0
    for _ in range(2000):
      ctx_size = self.read32(0x409800)
      if ctx_size:
        break
      time.sleep(0.001)
    if not (0 < ctx_size <= 0x80000):
      # Do NOT forge SCRATCH0 bit31 — that made warm-skip look "ready" and
      # then hung ctx_chan.  Prefer a sane mailbox, else known 660 Ti size.
      _fallback = (_mailbox_size if 0 < _mailbox_size <= 0x80000 else 0x29b00)
      if not (_ready & 0x80000000):
        raise RuntimeError(
            f"FECS discover_image_size failed (got {ctx_size:#x}) and FECS "
            f"was not ready (SCRATCH0 was {_ready:#x})")
      print(f"[kepler] FECS discover_image_size failed "
            f"(got {ctx_size:#x}, mailbox={_mailbox_size:#x}); "
            f"using fallback {_fallback:#x}", flush=True)
      ctx_size = _fallback
    else:
      print(f"[kepler] FECS discover_image_size={ctx_size:#x} "
            f"(mailbox={_mailbox_size:#x}; ready was {_ready:#x})", flush=True)
    # discover overwrites 0x409800 with the size; restore ready bit.
    self.write32(0x409800, 0x80000000)
    _ov = os.environ.get("KEPLER_GR_CTX_SIZE", "").strip()
    if _ov:
      _forced = int(_ov, 0)
      if not (0 < _forced <= 0x80000):
        raise RuntimeError(f"invalid KEPLER_GR_CTX_SIZE={_ov!r}")
      print(f"[kepler] GR ctx size override: discover={ctx_size:#x} "
            f"→ forced={_forced:#x}", flush=True)
      ctx_size = _forced
    self.gr_ctx_size = ctx_size
    print(f"[kepler] GR ctx image size={ctx_size:#x}", flush=True)
    # Complete the GK104 subdev initialization that was not performed at boot.
    # On a properly POSTed card, the VBIOS/nouveau driver init sequence runs:
    #   fb_preinit  → sysmem flush page (0x100c10)
    #   fb_init     → big-page mode (0x100c80) + sysmem flush page
    #   ltc_init    → L2 cache topology (0x17e8d8/0x17e000/0x17e8d4/0x17e8c0)
    #   bar_init    → BAR1 VMM enable (0x1704)
    # The un-POSTed card has none of these, so CPU framebuffer writes are not
    # visible to GPU internal clients (PBDMA reads zero GP entries).
    #
    # Late safety: if a POSTed path skipped cold RAM (and therefore the early
    # LTC call above), still bring up L2 before sysmem/BAR1 work.
    _gk104_post_ram_fb_ltc(self)
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
    if getattr(self, "_closed", False):
      return
    self._closed = True
    teardown = getattr(self, "_kepler_emergency_teardown", None)
    try:
      if teardown is not None:
        teardown("device close")
      if self.backend != "software":
        # No teardown MMIO is safe after the successful command/output phase.
        # The transport simply unmaps the BARs and closes the resource fds.
        if DEBUG:
          print("[kepler] client-only close: no teardown MMIO/config/reset",
                flush=True)
    finally:
      pci_dev = self.iface.pci_dev
      if hasattr(pci_dev, "set_phase"):
        pci_dev.set_phase("client-close")
      self.iface.pci_dev.fini()

  # MMIO convenience helpers used by the host-driven hardware bring-up.
  def read8(self, off):
    return self.dev_impl.pci_dev.mmio_read(0, off, 1)[0]
  def write8(self, off, v):
    self.dev_impl.pci_dev.mmio_write(0, off, bytes((v & 0xff,)))
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

def assemble_kepler_cubin(operation="add"):
  """Assemble the checked-in sm_30 PTX locally and return its cubin bytes.

  CUDA 10.2 is required for Kepler; newer toolkits intentionally reject sm_30.
  The source is tiny and is transformed for mul before invoking ptxas, so no
  precompiled operation-specific image is needed.
  """
  if operation not in ("add", "mul"):
    raise ValueError(f"unsupported Kepler operation: {operation}")
  ptx_path = os.path.join(SHARED_KEPLER_DIR, "add_kepler.ptx")
  with open(ptx_path, "r", encoding="utf-8") as f:
    ptx = f.read()
  if operation == "mul":
    ptx = ptx.replace("add.f32", "mul.f32")
  candidates = [os.environ.get("KEPLER_PTXAS"),
                "/usr/local/cuda-10.2/bin/ptxas",
                "/tmp/cuda102-nvcc/usr/local/cuda-10.2/bin/ptxas"]
  ptxas = next((p for p in candidates if p and os.path.exists(p)), None)
  if ptxas is None:
    raise RuntimeError("CUDA 10.2 ptxas not found (set KEPLER_PTXAS)")
  with tempfile.TemporaryDirectory(prefix="kepler_ptxas_") as d:
    src, out = os.path.join(d, f"{operation}.ptx"), os.path.join(d, f"{operation}.cubin")
    with open(src, "w", encoding="utf-8") as f: f.write(ptx)
    subprocess.run([ptxas, "-arch=sm_30", src, "-o", out],
                   check=True, capture_output=True, timeout=60)
    with open(out, "rb") as f: return f.read()

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
  operation = os.environ.get("KEPLER_OPERATION", "add")
  path = os.environ.get("KEPLER_CUBIN")
  # Assemble from PTX first.  This makes the normal path self-contained and
  # verifies that the local assembler reproduces the checked-in add image (or
  # the known-good mul digest) before any MMIO launch.
  if not path:
    assembled = None
    try:
      assembled = assemble_kepler_cubin(operation)
    except (OSError, RuntimeError, subprocess.SubprocessError):
      pass
    if assembled is not None:
      if operation == "add":
        with open(DEFAULT_CUBIN, "rb") as f: expected = f.read()
        if assembled != expected:
          raise RuntimeError("local sm_30 add cubin differs from checked-in reference")
      elif (len(assembled) != MUL_CUBIN_BYTES or
            hashlib.sha256(assembled).hexdigest() != MUL_CUBIN_SHA256):
        raise RuntimeError("local sm_30 mul cubin differs from verified reference")
      return assembled
  # Explicit cubins are accepted for bring-up, then validated as sm_30 ELF.
  bundled = DEFAULT_MUL_CUBIN if operation == "mul" else DEFAULT_CUBIN
  if not path and os.path.exists(bundled): path = bundled
  if not path or not os.path.exists(path):
    if operation == "add":
      path = compile_kepler_cubin_docker()
  if not path or not os.path.exists(path):
    raise RuntimeError(
        f"live hardware requires a real sm_30 {operation} cubin; "
        "set KEPLER_CUBIN or install CUDA 10.2 ptxas")
  with open(path, "rb") as f: cubin = f.read()
  if len(cubin) < 0x40 or cubin[:4] != b"\x7fELF" or struct.unpack_from("<I", cubin, 0x30)[0] & 0xff != 0x1e:
    raise ValueError(f"KEPLER_CUBIN must be an sm_30 ELF cubin: {path}")
  if operation == "mul":
    digest = hashlib.sha256(cubin).hexdigest()
    if len(cubin) != MUL_CUBIN_BYTES or digest != MUL_CUBIN_SHA256:
      raise ValueError(
          f"KEPLER_CUBIN mul digest mismatch: {len(cubin)} bytes sha256={digest}")
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
# Autonomous BAR1-root bootstrap: lives in the trailing zero pad of the live
# gk104_fecs_code.bin image (last live byte at 0xb09; zeros 0xb0a..0xbff).
# Assembled from examples_kepler/fecs_bar1_bootstrap.fuc (envyas -m falcon -V fuc3).
# night26 placed this at 0xc20 (past the 0xc00 IMEM window) and BAR1 stayed
# virgin 0xff — entry must stay inside the loaded image.
FECS_BAR1_BOOTSTRAP_IMEM = 0xb20
FECS_BAR1_BOOTSTRAP_DELAY_START = 0xb29
FECS_BAR1_BOOTSTRAP_DELAY_END = 0xb2f
FECS_BAR1_BOOTSTRAP_CODE = bytes(bytearray().join(
    struct.pack("<I", w) for w in (
        0x17f004bd, 0x0013f100, 0x0112b610, 0xfefd1bf4, 0x0bfe0007,
        0x0047f100, 0x0043f002, 0x000037f1, 0x801133f0, 0x34bd0043,
        0xf1014380, 0xf0ffff37, 0x4380ff33, 0x8034bd02, 0x43800343,
        0x0137f104, 0x0033f012, 0x80054380, 0x34bd0843, 0xf1094380,
        0xf0100137, 0x43800033, 0x8034bd0a, 0x07f10b43, 0x03f08100,
        0x0037f102, 0x0033f010, 0xbd0003d0, 0x0007f104, 0x0203f088,
        0x000237f1, 0x800033f1, 0xbd0003d0, 0x0017f104, 0x0213f002,
        0x020027f1, 0xfa0023f0, 0x03f80621, 0x021017f1, 0xf00113f0,
        0x23f00027, 0x0621fa01, 0x17f103f8, 0x13f00220, 0x0027f002,
        0xfa0223f0, 0x03f80621, 0xf98e0ef5,
    )))
# Hardcoded xdst layout baked into the Falcon routine (matches
# _gk104_init_bar1_identity's first-512KiB instance bank — bit19-safe).
FECS_BAR1_BOOTSTRAP_XFER = (
    (0x200, 0x30200, 16),   # DMEM, ext offset, size — instance +0x200
    (0x210, 0x10000, 8),    # PDE
    (0x220, 0x20000, 16),   # first two SPT entries
)
FECS_BAR1_BOOTSTRAP_ROOTS = (
    struct.pack("<QQ", 0x00010000, 0x00ffffff),
    struct.pack("<Q", (1 << 32) | (0x00020000 << 24)),
    struct.pack("<QQ", 0x00000201, 0x00000301),
)
FECS_BAR1_BOOTSTRAP_END = FECS_BAR1_BOOTSTRAP_IMEM + len(FECS_BAR1_BOOTSTRAP_CODE)
# Night40l: host-staged roots + xdst, then wr32@0x34 enables 0x1704
# (no pre-bit0 BAR1 init; no PDB invalidate).
PMU_BAR1_BOOTSTRAP_IMEM = 0xb14
PMU_BAR1_BOOTSTRAP_DELAY_START = 0xb57
PMU_BAR1_BOOTSTRAP_DELAY_END = 0xb5d
PMU_BAR1_BOOTSTRAP_MAGIC_DMEM = 0xd7c
PMU_BAR1_BOOTSTRAP_MAGIC = 0x40c0b002
PMU_BAR1_BOOTSTRAP_GO_DMEM = 0xd78
PMU_BAR1_BOOTSTRAP_GO = 0x40c0b003
PMU_BAR1_BOOTSTRAP_DELAY_ENTERED_DMEM = 0xd74
PMU_BAR1_BOOTSTRAP_DELAY_ENTERED = 0x40c0b004
PMU_BAR1_BOOTSTRAP_DONE_DMEM = 0xd70
PMU_BAR1_BOOTSTRAP_DONE = 0x40c0b005
# Retained for explicit H21/H22 PRAMIN diagnostics; the default pad uses xdst.
PMU_BAR1_BOOTSTRAP_PAUSE_STATUS_DMEM = 0xd6c
PMU_BAR1_BOOTSTRAP_INPAUSE_RB_DMEM = 0xd6c
PMU_BAR1_BOOTSTRAP_LITERAL = 0xa5a5a5a5
# Night40ae clear-status retained for offline fake; af+ pad skips clear poll.
PMU_BAR1_BOOTSTRAP_PAUSE_CLEAR_DMEM = 0xd68
PMU_BAR1_BOOTSTRAP_CODE = bytes(bytearray().join(
    struct.pack("<I", w) for w in (
        0xb00237f1, 0x40c033f1, 0x0d7c47f1, 0x800043f0,
        0x27f10043, 0x23f1b003, 0x47f140c0, 0x43f00d78,
        0x00439800, 0xf40232bb, 0xe7f1fa1b, 0xe3f01620,
        0x08d7f100, 0x00d3f000, 0xf13421f4, 0xf126f0e7,
        0xf40000d7, 0x07f13421, 0x37f007e4, 0x0003d004,
        0x07e007f1, 0xd00437f0, 0x57f10003, 0x07f100ff,
        0x06cf07c0, 0x0464f000, 0xb6091bf4, 0x1bf40152,
        0xf124bdf0, 0xfe060237, 0x17f10037, 0x13f00d80,
        0x0621fa02, 0x37f103f8, 0x37fe0400, 0x9017f100,
        0x0113f00d, 0xf80621fa, 0x0037f103, 0x0037fe05,
        0x0da017f1, 0xfa0213f0, 0x03f80621, 0x07e407f1,
        0xd00437f0, 0xe7f10003, 0xd7f126f0, 0x21f40001,
        0x20e7f134, 0xabd7f116, 0x3421f40a, 0xb00537f1,
        0x40c033f1, 0x0d7047f1, 0xf4004380,
        0x0000000e,
    )))
PMU_BAR1_BOOTSTRAP_END = PMU_BAR1_BOOTSTRAP_IMEM + len(PMU_BAR1_BOOTSTRAP_CODE)
# Host stages all three; the default pad xdsts each root under ENTER.
PMU_BAR1_BOOTSTRAP_XFER = (
    (0x0d80, 0x00060200, 16),
    (0x0d90, 0x00040000, 8),
    (0x0da0, 0x00050000, 16),
)
PMU_BAR1_BOOTSTRAP_XFER_PAD = PMU_BAR1_BOOTSTRAP_XFER
PMU_BAR1_BOOTSTRAP_ROOTS = (
    struct.pack("<QQ", 0x00040000, 0x00ffffff),
    struct.pack("<Q", (1 << 32) | (0x00050000 << 24)),
    struct.pack("<QQ", 0x00000501, 0x00000601),
)
PMC_ENABLE       = 0x200

# GK104 PMU host-command / MEMX protocol (ref/linux/.../pmu/gt215.c and
# pmu/memx.c).  The PMU is not a GSP: these are the ordinary Falcon rings
# initialized by the PMU firmware itself.
PMU_PROC_MEMX = 0x584d454d
PMU_MEMX_INFO = 0
PMU_MEMX_EXEC = 1
PMU_MEMX_INFO_DATA = 0
PMU_MEMX_ENTER = 1
PMU_MEMX_LEAVE = 2
PMU_MEMX_WR32 = 3
PMU_MEMX_WAIT = 4
PMU_MEMX_DELAY = 5
PMU_MEMX_TRAIN = 7

# gf119.fuc4.h: memx_func_enter_wait @ 0x0534 / leave_wait @ 0x0561.
# Both spin on PMU OUTPUT FB_PAUSE; cold eGPU never acks → EXEC hangs.
# bra z/nz imm8 at these offsets; rewrite to unconditional bra +3 (fall through).
_PMU_ENTER_WAIT_BRA = 0x53e
_PMU_LEAVE_WAIT_BRA = 0x56b
_PMU_BRA_Z_BACK = bytes((0xf4, 0x0b, 0xf6))   # bra z, -10
_PMU_BRA_NZ_BACK = bytes((0xf4, 0x1b, 0xf6))  # bra nz, -10
_PMU_BRA_FALLTHROUGH = bytes((0xf4, 0x0e, 0x03))  # bra (always), +3


def _patch_pmu_memx_nowait(code: bytearray | bytes) -> bytearray:
  """Skip FB_PAUSE spin in MEMX ENTER/LEAVE (keep 0x1620/0x26f0 side effects)."""
  out = bytearray(code)
  if out[_PMU_ENTER_WAIT_BRA:_PMU_ENTER_WAIT_BRA + 3] != _PMU_BRA_Z_BACK:
    raise RuntimeError(
        f"unexpected PMU ENTER wait bra at {_PMU_ENTER_WAIT_BRA:#x}: "
        f"{out[_PMU_ENTER_WAIT_BRA:_PMU_ENTER_WAIT_BRA + 3].hex()}")
  if out[_PMU_LEAVE_WAIT_BRA:_PMU_LEAVE_WAIT_BRA + 3] != _PMU_BRA_NZ_BACK:
    raise RuntimeError(
        f"unexpected PMU LEAVE wait bra at {_PMU_LEAVE_WAIT_BRA:#x}: "
        f"{out[_PMU_LEAVE_WAIT_BRA:_PMU_LEAVE_WAIT_BRA + 3].hex()}")
  out[_PMU_ENTER_WAIT_BRA:_PMU_ENTER_WAIT_BRA + 3] = _PMU_BRA_FALLTHROUGH
  out[_PMU_LEAVE_WAIT_BRA:_PMU_LEAVE_WAIT_BRA + 3] = _PMU_BRA_FALLTHROUGH
  return out


def _gk104_pmu_reload_for_ram(dev, nowait: bool, reason: str) -> None:
  """Reload PMU (+ BAR1 pad) with or without Nouveau FB_PAUSE waits.

  Night40ag: pad SET+poll proves OUTPUT bit2 can rise after cold RAM.  Nouveau
  ``gk104_ram_prog`` runs the GDDR5 body inside MEMX ENTER that *waits* for
  that ack (``memx.fuc``).  Our default nowait patch therefore reprograms the
  MC without real pause — night40ah restores the wait for ``ram_program``.
  """
  fw = find_kepler_firmware()
  if not fw:
    raise RuntimeError(f"no GK104 firmware tree for PMU reload ({reason})")
  pmu_code = bytearray(open(os.path.join(fw, "gk104_pmu_code.bin"), "rb").read())
  pmu_data = open(os.path.join(fw, "gk104_pmu_data.bin"), "rb").read()
  if nowait:
    pmu_code = _patch_pmu_memx_nowait(pmu_code)
  pmu_code = _gk104_pmu_embed_bar1_bootstrap(pmu_code)
  try:
    falcon_stop(dev, PMU_FALCON_BASE, timeout_ms=200)
  except Exception:
    pass
  falcon_reset(dev, PMU_FALCON_BASE)
  falcon_load(dev, PMU_FALCON_BASE, pmu_code, pmu_data, entry=0, start=True)
  time.sleep(0.05)
  try:
    setattr(dev, "_pmu_code", bytes(pmu_code))
    setattr(dev, "_pmu_memx_nowait", nowait)
    setattr(dev, "_pmu_memx_data", None)
  except Exception:
    pass
  init = getattr(dev, "_init_pmu_memx", None)
  if init is None or not init():
    raise RuntimeError(f"PMU MEMX unavailable after reload ({reason})")
  print(f"[kepler] PMU reloaded for ram_program ({reason}): "
        f"FB_PAUSE_wait={'off' if nowait else 'on (Nouveau)'}", flush=True)


def _gk104_dump_ram_mc_regs(dev, label: str = "after ram_program", *,
                            strict: bool | None = None) -> None:
  """Sample memory-controller regs Nouveau waits on / programs around RAM."""
  regs = (0x10f200, 0x10f210, 0x10f808, 0x10f160, 0x10f584, 0x10ecc0,
          0x100c80, 0x110974, 0x111974, 0x137390, 0x001620, 0x0026f0,
          0x10f910, 0x10f914, 0x022438, 0x022554,
          0x1373f4, 0x132024, 0x132034,  # memory selector / REFPLL coefficients
          0x00d61c, 0x00d638, 0x00d604)  # GPIO 0x18 / 0x2e / trig
  parts = []
  for r in regs:
    try:
      parts.append(f"{r:#x}={dev.read32(r) & 0xffffffff:#x}")
    except Exception as e:
      parts.append(f"{r:#x}=<{e}>")
  print(f"[kepler] MC dump ({label}): " + " ".join(parts), flush=True)
  # Nouveau gk104_ram_train: ram_wait(0x110974+i*0x1000, 0xf, 0) per part
  # (ramgk104.c:149-152).  Live still shows 0xa after EXEC — fail loud (H14).
  try:
    nparts = min(dev.read32(0x022438) & 0xff, 16)
    pmask = dev.read32(0x022554) & 0xffffffff
  except Exception as e:
    print(f"[kepler] train-status partition probe skipped: {e}", flush=True)
    return
  if nparts == 0:
    nparts = int(os.environ.get("KEPLER_RAM_PARTS", "4"), 0)
    print(f"[kepler] train-status: 0x022438 was 0; probing parts={nparts}",
          flush=True)
  bad = []
  first = []
  for i in range(nparts):
    if pmask & (1 << i):
      continue
    addr = 0x110974 + i * 0x1000
    try:
      st = dev.read32(addr) & 0xffffffff
    except Exception as e:
      bad.append(f"{addr:#x}=<{e}>")
      continue
    print(f"[kepler] train status part{i} {addr:#x}={st:#x} "
          f"nibble={st & 0xf:#x}", flush=True)
    first.append((addr, st))
    if st & 0xf:
      bad.append(f"{addr:#x}={st:#x}")
  # Night40ao: host re-poll — distinguish stuck-forever from slow train
  # after silent MEMX_WAIT timeout (kernel.fuc:118-134).
  repoll_ms = int(os.environ.get("KEPLER_TRAIN_HOST_REPOLL_MS", "100"), 0)
  if bad and repoll_ms > 0:
    deadline = time.monotonic() + repoll_ms / 1000.0
    moved = False
    while time.monotonic() < deadline:
      for addr, st0 in list(first):
        st1 = dev.read32(addr) & 0xffffffff
        if st1 != st0:
          print(f"[kepler] train status CHANGED {addr:#x} "
                f"{st0:#x}->{st1:#x} during {repoll_ms}ms host repoll",
                flush=True)
          moved = True
          first = [(a, (st1 if a == addr else s)) for a, s in first]
      if all((dev.read32(a) & 0xf) == 0 for a, _ in first):
        print("[kepler] train status cleared during host repoll", flush=True)
        break
      time.sleep(0.001)
    if not moved:
      print(f"[kepler] train status STUCK through {repoll_ms}ms host repoll "
            f"(H25: HW never completes)", flush=True)
    bad = []
    for addr, _st0 in first:
      st = dev.read32(addr) & 0xffffffff
      print(f"[kepler] train status after repoll {addr:#x}={st:#x} "
            f"nibble={st & 0xf:#x}", flush=True)
      if st & 0xf:
        bad.append(f"{addr:#x}={st:#x}")
  if strict is None:
    strict = os.environ.get("KEPLER_TRAIN_STATUS_STRICT", "1") != "0"
  if bad and strict:
    raise RuntimeError(
        f"H14: RAM train status nibble not clear {label} "
        f"(Nouveau waits 0xf==0): {', '.join(bad)}; "
        "VRAM clients will keep returning bad0fb — fix train/MC before PRAMIN")


def _gk104_train_nibbles_clear(dev) -> bool:
  """True when every active RAM partition train nibble is 0 (Nouveau wait)."""
  try:
    nparts = min(dev.read32(0x022438) & 0xff, 16)
    pmask = dev.read32(0x022554) & 0xffffffff
  except Exception:
    return False
  if nparts == 0:
    nparts = int(os.environ.get("KEPLER_RAM_PARTS", "4"), 0)
  for i in range(nparts):
    if pmask & (1 << i):
      continue
    if (dev.read32(0x110974 + i * 0x1000) & 0xf) != 0:
      return False
  return True


def _gk104_sample_clk_regs(dev) -> dict:
  out = {}
  for reg in GPC_CLK_REGS:
    try:
      out[reg] = dev.read32(reg) & 0xffffffff
    except Exception as e:
      out[reg] = e
  return out


def _gk104_bar1_verify_top(dev, mapped_size=None) -> None:
  """R/W a marker at the last mapped BAR1 page (128 MiB done criterion)."""
  if mapped_size is None:
    mapped_size = getattr(dev, "_bar1_identity_size", None)
  if not mapped_size or mapped_size < 0x2000:
    return
  if os.environ.get("KEPLER_BAR1_VERIFY_TOP", "1") == "0":
    return
  top = mapped_size - 0x1000
  marker = (0xa5a55a5a ^ (top & 0xffffffff)) & 0xffffffff
  blob = struct.pack("<I", marker)
  hw = getattr(getattr(dev, "dev_impl", None), "hw", None)
  used_bar1 = False
  try:
    if hw is not None and hasattr(hw, "mmio_write") and hasattr(hw, "mmio_read"):
      hw.mmio_write(1, top, blob)
      _gk104_bar_flush(dev)
      got = struct.unpack("<I", bytes(hw.mmio_read(1, top, 4)))[0]
      used_bar1 = True
    else:
      _gk104_pramin_write(dev, top, blob)
      _gk104_bar_flush(dev)
      got = _gk104_pramin_read32(dev, top) & 0xffffffff
  except Exception as e:
    raise RuntimeError(
        f"BAR1 top R/W probe failed at {top:#x} (size={mapped_size:#x}): {e}")
  if got != marker:
    raise RuntimeError(
        f"BAR1 top mismatch at {top:#x}: wanted={marker:#x} got={got:#x} "
        f"via={'BAR1' if used_bar1 else 'PRAMIN'} size={mapped_size:#x}")
  if used_bar1:
    try:
      via_pramin = _gk104_pramin_read32(dev, top) & 0xffffffff
      if via_pramin != marker:
        print(f"[kepler] BAR1 top BAR1={got:#x} PRAMIN={via_pramin:#x} "
              f"(BAR1 match; PRAMIN diverge ok if window)", flush=True)
    except Exception:
      pass
  print(f"[kepler] BAR1 top R/W ok va={top:#x} size={mapped_size:#x} "
        f"via={'BAR1' if used_bar1 else 'PRAMIN'}", flush=True)


def _pmu_wait_data_access(dev, value, timeout_s=0.25):
  """Acquire the PMU data-segment semaphore used by gt215_pmu_send/memx."""
  deadline = time.monotonic() + timeout_s
  # A previous aborted MEMX/host op can leave 0x10a580 stuck non-zero; clear
  # before requesting the new lock value (nouveau always writes then spins).
  cur = dev.read32(0x10a580) & 0xffffffff
  if cur not in (0, value & 0xffffffff):
    dev.write32(0x10a580, 0)
  dev.write32(0x10a580, value)
  while dev.read32(0x10a580) != value:
    if time.monotonic() >= deadline:
      # One recovery attempt: force unlock, re-request.
      dev.write32(0x10a580, 0)
      time.sleep(0.0001)
      dev.write32(0x10a580, value)
      if (dev.read32(0x10a580) & 0xffffffff) == (value & 0xffffffff):
        return
      raise TimeoutError(f"PMU data-segment acquire timeout value={value:#x} "
                         f"stuck={dev.read32(0x10a580):#x}")
    time.sleep(0.00001)


def pmu_send(dev, process, message, data0, data1, timeout_s=2.0):
  """Send one synchronous GK104 PMU command and return its two-word reply."""
  deadline = time.monotonic() + timeout_s
  ring = dev.read32(0x10a4d0)
  if not ring:
    raise RuntimeError("GK104 PMU host-command ring is not initialized")
  send_base = ring & 0x0000ffff
  addr = dev.read32(0x10a4a0) & 0x0f
  while dev.read32(0x10a4b0) == (addr ^ 0x08):
    if time.monotonic() >= deadline:
      raise TimeoutError("GK104 PMU host-command ring is full")
    time.sleep(0.00001)

  _pmu_wait_data_access(dev, 0x00000001)
  try:
    dev.write32(0x10a1c0,
                0x01000000 | (((addr & 0x07) << 4) + send_base))
    for word in (process, message, data0, data1):
      dev.write32(0x10a1c4, word & 0xffffffff)
    dev.write32(0x10a4a0, (addr + 1) & 0x0f)
  finally:
    dev.write32(0x10a580, 0)

  recv_ring = dev.read32(0x10a4dc)
  if not recv_ring:
    raise RuntimeError("GK104 PMU host-reply ring is not initialized")
  recv_base = recv_ring & 0x0000ffff
  while time.monotonic() < deadline:
    get = dev.read32(0x10a4cc) & 0x0f
    put = dev.read32(0x10a4c8) & 0x0f
    if get != put:
      _pmu_wait_data_access(dev, 0x00000002)
      try:
        dev.write32(0x10a1c0,
                    0x02000000 | (((get & 0x07) << 4) + recv_base))
        reply = tuple(dev.read32(0x10a1c4) & 0xffffffff for _ in range(4))
        dev.write32(0x10a4cc, (get + 1) & 0x0f)
      finally:
        dev.write32(0x10a580, 0)
      if reply[0] == (process & 0xffffffff) and reply[1] == (message & 0xffffffff):
        return reply[2], reply[3]
      # An unrelated asynchronous PMU message is not expected on this path;
      # keep draining until the synchronous response arrives.
      continue
    time.sleep(0.00001)
  raise TimeoutError(f"GK104 PMU reply timeout process={process:#x} message={message:#x}")


def pmu_memx_exec(dev, data_base, commands, timeout_s=5.0):
  """Execute a compact PMU MEMX script.

  ``commands`` contains ``(opcode, words)`` tuples.  Each command is encoded
  exactly as memx.c's ``memx_out``: a 16-bit payload length in the high half
  and the MEMX opcode in the low half, followed by its payload words.
  """
  words = []
  for opcode, payload in commands:
    payload = tuple(int(x) & 0xffffffff for x in payload)
    words.append(((len(payload) << 16) | (int(opcode) & 0xffff)) & 0xffffffff)
    words.extend(payload)
  _pmu_wait_data_access(dev, 0x00000003)
  try:
    dev.write32(0x10a1c0, 0x01000000 | (data_base & 0x00ffffff))
    for word in words:
      dev.write32(0x10a1c4, word)
    finish = dev.read32(0x10a1c0) & 0x00ffffff
  finally:
    dev.write32(0x10a580, 0)
  return pmu_send(dev, PMU_PROC_MEMX, PMU_MEMX_EXEC,
                  data_base, finish, timeout_s=timeout_s)


def pmu_memx_exec_fire(dev, data_base, commands) -> int:
  """Submit a MEMX EXEC without waiting for the PMU reply.

  Night40m: the script's DELAY covers host bit0; afterward PMU host MMIO is
  dead so a synchronous reply cannot be collected.  Return the script end
  address that was handed to EXEC.
  """
  words = []
  for opcode, payload in commands:
    payload = tuple(int(x) & 0xffffffff for x in payload)
    words.append(((len(payload) << 16) | (int(opcode) & 0xffff)) & 0xffffffff)
    words.extend(payload)
  _pmu_wait_data_access(dev, 0x00000003)
  try:
    dev.write32(0x10a1c0, 0x01000000 | (data_base & 0x00ffffff))
    for word in words:
      dev.write32(0x10a1c4, word)
    finish = dev.read32(0x10a1c0) & 0x00ffffff
  finally:
    dev.write32(0x10a580, 0)
  # One-shot ring push of EXEC; ignore the reply path entirely.
  process, message, data0, data1 = (
      PMU_PROC_MEMX, PMU_MEMX_EXEC, data_base, finish)
  ring = dev.read32(0x10a4d0)
  if not ring:
    raise RuntimeError("GK104 PMU host-command ring is not initialized")
  send_base = ring & 0x0000ffff
  addr = dev.read32(0x10a4a0) & 0x0f
  deadline = time.monotonic() + 0.2
  while dev.read32(0x10a4b0) == (addr ^ 0x08):
    if time.monotonic() >= deadline:
      raise TimeoutError("GK104 PMU host-command ring is full (fire)")
    time.sleep(0.00001)
  _pmu_wait_data_access(dev, 0x00000001)
  try:
    dev.write32(0x10a1c0,
                0x01000000 | (((addr & 0x07) << 4) + send_base))
    for word in (process, message, data0, data1):
      dev.write32(0x10a1c4, word & 0xffffffff)
    dev.write32(0x10a4a0, (addr + 1) & 0x0f)
  finally:
    dev.write32(0x10a580, 0)
  return finish


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
  raw = pathlib.Path(path).read_bytes()
  # Onboard 660 Ti PROM images use PCIR device 0x1183; Palit golden is 0x1184.
  device = 0x1184
  if raw[:2] == b"\x55\xaa" and len(raw) >= 0x1c:
    try:
      pcir = struct.unpack_from("<H", raw, 0x18)[0]
      if raw[pcir:pcir + 4] == b"PCIR":
        device = struct.unpack_from("<H", raw, pcir + 6)[0]
    except Exception:
      pass
  image = nvbios_init.find_vbios_image(raw, device=device)
  scripts = nvbios_init.find_vbios_scripts(image)
  print(f"vbios-init: image_bytes={len(image)} scripts={len(scripts)} "
        f"pcir_device={device:#x}")
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


def _gk104_clk_calc_gpc_info(dev, target_khz: int, crystal_khz: int = 27_000):
  """Nouveau ``calc_clk`` for GPC (idx 0): pick closest divider vs PLL path."""
  # Fixed sources (calc_src).
  if target_khz in (27_000, 108_000):
    dsrc = 0x00030000 if target_khz == 108_000 else 0
    return {"freq": target_khz, "dsrc": dsrc, "ddiv": 0, "mdiv": 0,
            "ssel": 0, "coef": 0}
  if target_khz == 100_000:
    return {"freq": target_khz, "dsrc": 2, "ddiv": 0, "mdiv": 0,
            "ssel": 0, "coef": 0}

  def read_pll(pll: int) -> int:
    ctrl = dev.read32(pll) & 0xffffffff
    coef = dev.read32(pll + 4) & 0xffffffff
    if not (ctrl & 1):
      return 0
    p = (coef >> 16) & 0x3f
    n = (coef >> 8) & 0xff
    m = coef & 0xff or 1
    if pll in (0x00e800, 0x00e820):
      return crystal_khz * n // (m * (p or 1))
    return 0

  def read_vco() -> int:
    ssrc = dev.read32(0x137160) & 0xffffffff
    return read_pll(0x00e820 if (ssrc & 0x100) else 0x00e800)

  def calc_div(ref: int, freq: int) -> tuple[int, int]:
    div = min(max((ref * 2) // max(freq, 1), 2), 65)
    return (ref * 2) // div, div - 2

  vco = read_vco() or crystal_khz
  clk0, div0 = calc_div(vco, target_khz)
  clk0, div1d = calc_div(clk0, target_khz)
  dsrc = 3
  ddiv = (0x80000000 | div0) if div0 else 0
  mdiv_div = (0x80000000 | div1d) if div1d else 0

  # PLL path (calc_pll / program_gk104_gpc_pll search).
  best = None
  ref = vco
  for p in range(1, 64):
    for m in range(17, 33):
      for n in range(8, 256):
        vco_n = ref * n // m
        if not 1_100_000 <= vco_n <= 2_404_000:
          continue
        out = vco_n // p
        cand = (abs(out - target_khz), out, p, n, m)
        if best is None or cand < best:
          best = cand
  clk1 = coef = 0
  div1p = 0
  if best is not None and best[1]:
    _, clk1_raw, p, n, m = best
    coef = (p << 16) | (n << 8) | m
    clk1, div1p = calc_div(clk1_raw, target_khz)

  if abs(target_khz - clk0) <= abs(target_khz - clk1):
    return {"freq": clk0, "dsrc": dsrc, "ddiv": ddiv, "mdiv": mdiv_div,
            "ssel": 0, "coef": 0}
  mdiv_pll = (0x80000000 | (div1p << 8)) if div1p else 0
  return {"freq": clk1, "dsrc": 0x40000100, "ddiv": 0, "mdiv": mdiv_pll,
          "ssel": 1, "coef": coef}


def gk104_clk_prog_gpc(dev, target_khz: int) -> dict:
  """Program GPC domain via Nouveau ``gk104_clk_prog`` stages (idx 0 only)."""
  info = _gk104_clk_calc_gpc_info(dev, int(target_khz))
  idx = 0
  # stage0: div programming
  if not info["ssel"]:
    nvkm_mask(dev, 0x1371d0 + idx * 4, 0x8000003f, info["ddiv"])
    dev.write32(0x137160 + idx * 4, info["dsrc"])
  # stage1_0: select div mode
  nvkm_mask(dev, 0x137100, 1 << idx, 0)
  t0 = time.time()
  while time.time() - t0 < 2.0:
    if not (dev.read32(0x137100) & (1 << idx)):
      break
    time.sleep(0.001)
  # stage2: maybe program pll
  addr = 0x137000 + idx * 0x20
  nvkm_mask(dev, addr, 0x00000004, 0)
  nvkm_mask(dev, addr, 0x00000001, 0)
  if info["coef"]:
    dev.write32(addr + 4, info["coef"])
    nvkm_mask(dev, addr, 0x00000001, 0x00000001)
    nvkm_mask(dev, addr, 0x00000010, 0)
    t0 = time.time()
    while time.time() - t0 < 2.0:
      if dev.read32(addr) & 0x00020000:
        break
      time.sleep(0.001)
    nvkm_mask(dev, addr, 0x00000010, 0x00000010)
    nvkm_mask(dev, addr, 0x00000004, 0x00000004)
  # stage3: final divider
  if info["ssel"]:
    nvkm_mask(dev, 0x137250 + idx * 4, 0x00003f00, info["mdiv"])
  else:
    nvkm_mask(dev, 0x137250 + idx * 4, 0x0000003f, info["mdiv"])
  # stage4_0: maybe select pll mode
  if info["ssel"]:
    nvkm_mask(dev, 0x137100, 1 << idx, info["ssel"])
    t0 = time.time()
    while time.time() - t0 < 2.0:
      if (dev.read32(0x137100) & (1 << idx)) == info["ssel"]:
        break
      time.sleep(0.001)
  print(f"[kepler] gk104_clk_prog GPC target={target_khz} "
        f"actual={info['freq']} ssel={info['ssel']} coef={info['coef']:#x}",
        flush=True)
  return info


def _gk104_experimental_pstate(dev) -> None:
  """Nouveau ``nvkm_pstate_prog`` order: ram calc/prog, then GPC clk_prog.

  Experimental only.  Voltage/fan/PCIe and full multi-domain clk are out of
  scope.  Aborts if GDDR train nibbles leave the clear state.
  """
  vbios_path = os.environ.get("KEPLER_VBIOS", DEFAULT_VBIOS)
  image, _, _ = vbios_init_info(vbios_path)
  pstates = nvbios_init.parse_perf_pstates(image)
  if not pstates:
    raise RuntimeError("experimental pstate: PERF table missing/unparsed")
  idx = int(os.environ.get("KEPLER_PSTATE_IDX", "0"), 0)
  if not 0 <= idx < len(pstates):
    raise ValueError(
        f"KEPLER_PSTATE_IDX={idx} out of range 0..{len(pstates) - 1}")
  ps = pstates[idx]
  mem_khz = int(os.environ.get("KEPLER_PSTATE_MEM_KHZ", "0"), 0) or ps["mem_khz"]
  gpc_khz = int(os.environ.get("KEPLER_PSTATE_GPC_KHZ", "0"), 0) or ps["gpc_khz"]
  do_mem = os.environ.get("KEPLER_EXPERIMENTAL_PSTATE_MEM", "1") != "0"
  do_clk = os.environ.get("KEPLER_EXPERIMENTAL_PSTATE_CLK", "1") != "0"
  print(f"[kepler] experimental pstate idx={idx} id={ps['pstate']:#x} "
        f"mem={mem_khz}kHz gpc={gpc_khz}kHz mem_prog={do_mem} "
        f"clk_prog={do_clk}", flush=True)
  before = _gk104_sample_clk_regs(dev)
  print("[kepler] clk before experimental-pstate: " + " ".join(
      f"{r:#x}={v if isinstance(v, Exception) else f'{v:#x}'}"
      for r, v in before.items()), flush=True)
  if do_mem:
    if not mem_khz:
      raise RuntimeError("experimental pstate: mem domain frequency is 0")
    freq_mhz = max(1, (mem_khz + 500) // 1000)
    debug = bool(DEBUG and getenv("KEPLER_VBIOS_TRACE", 0))
    print(f"[kepler] experimental pstate: ram_program freq={freq_mhz}",
          flush=True)
    nvbios_init.run_vbios_ram_program(
        dev, image, freq_mhz=freq_mhz, debug=debug)
    if not _gk104_train_nibbles_clear(dev):
      raise RuntimeError(
          "experimental pstate: train nibbles non-zero after ram_program")
  if do_clk:
    if not gpc_khz:
      raise RuntimeError("experimental pstate: gpc domain frequency is 0")
    gk104_clk_prog_gpc(dev, gpc_khz)
  after = _gk104_sample_clk_regs(dev)
  print("[kepler] clk after experimental-pstate: " + " ".join(
      f"{r:#x}={v if isinstance(v, Exception) else f'{v:#x}'}"
      for r, v in after.items()), flush=True)
  if do_mem and not _gk104_train_nibbles_clear(dev):
    raise RuntimeError(
        "experimental pstate: train nibbles non-zero after clk_prog")
  print("[kepler] experimental pstate: done"
        + (" (train clear)" if do_mem else " (clk-only)"), flush=True)


def _gk104_reclock_after_ok(dev) -> None:
  """Optional pre-LAUNCH reclock — never during FB init (Night41b H40/H49).

  Gates (all default off):
    KEPLER_EXPERIMENTAL_PSTATE=1 — Nouveau-ordered pstate (ram then GPC clk)
    KEPLER_RECLOCK_AFTER_OK=1    — legacy mem-only ``run_vbios_ram_program``
  Experimental wins when both are set.  Called after channel setup and before
  ``submit_launch`` so a GPC boost covers the compute kernel.  Mem reclock
  requires clear train nibbles; clk-only experimental may run without that.
  """
  experimental = os.environ.get("KEPLER_EXPERIMENTAL_PSTATE", "0") != "0"
  legacy = os.environ.get("KEPLER_RECLOCK_AFTER_OK", "0") != "0"
  if not experimental and not legacy:
    return
  train_ok = _gk104_train_nibbles_clear(dev)
  if experimental:
    do_mem = os.environ.get("KEPLER_EXPERIMENTAL_PSTATE_MEM", "1") != "0"
    if do_mem and not train_ok:
      print("[kepler] experimental pstate skipped: train nibbles not clear "
            "(set KEPLER_EXPERIMENTAL_PSTATE_MEM=0 for clk-only)",
            flush=True)
      return
    if not do_mem and not train_ok:
      print("[kepler] experimental pstate: clk-only despite non-clear train "
            "nibbles", flush=True)
    _gk104_dump_ram_mc_regs(dev, label="before experimental-pstate",
                            strict=False)
    _gk104_experimental_pstate(dev)
    return
  if not train_ok:
    print("[kepler] reclock-after-ok skipped: train nibbles not clear",
          flush=True)
    return
  _gk104_dump_ram_mc_regs(dev, label="before reclock-after-ok", strict=True)
  before = _gk104_sample_clk_regs(dev)
  print("[kepler] clk before reclock-after-ok: " + " ".join(
      f"{r:#x}={v if isinstance(v, Exception) else f'{v:#x}'}"
      for r, v in before.items()), flush=True)
  vbios_path = os.environ.get("KEPLER_VBIOS", DEFAULT_VBIOS)
  image, _, _ = vbios_init_info(vbios_path)
  freq = int(os.environ.get("KEPLER_RAM_FREQ", "648"), 0)
  debug = bool(DEBUG and getenv("KEPLER_VBIOS_TRACE", 0))
  print(f"[kepler] reclock-after-ok: run_vbios_ram_program freq={freq}",
        flush=True)
  nvbios_init.run_vbios_ram_program(dev, image, freq_mhz=freq, debug=debug)
  after = _gk104_sample_clk_regs(dev)
  print("[kepler] clk after reclock-after-ok: " + " ".join(
      f"{r:#x}={v if isinstance(v, Exception) else f'{v:#x}'}"
      for r, v in after.items()), flush=True)
  _gk104_dump_ram_mc_regs(dev, label="after reclock-after-ok", strict=True)
  if not _gk104_train_nibbles_clear(dev):
    raise RuntimeError(
        "reclock-after-ok left RAM train nibbles non-zero (abort)")
  print("[kepler] reclock-after-ok: train still clear", flush=True)


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


def falcon_reset(dev, base, settle_s=0.05):
  """Reset a Falcon so hung firmware can be reloaded cleanly.

  A prior MEMX WR32 timeout can leave the PMU Falcon alive-looking (rings
  still programmed) but deaf to host commands.  GK104's PMU is special:
  Nouveau's ``gf100_pmu_reset()`` resets the whole PMU subdevice through
  ``PMC_ENABLE[13]``.  Pulsing the internal CPUCTL cannot recover an already
  inaccessible PMU aperture (night16 returned 0xffffffff at 0x10a580).
  Other Falcons retain the internal CPUCTL reset used by the FECS path.
  """
  if (base & 0xffffffff) == (PMU_FALCON_BASE & 0xffffffff):
    pmc = dev.read32(PMC_ENABLE) & 0xffffffff
    # gk104_mc_reset[] maps NVKM_SUBDEV_PMU to bit 0x00002000 and
    # gf100_pmu_reset() calls nvkm_mc_disable()/nvkm_mc_enable().  Do not
    # reject 0xffffffff here: full PMC_ENABLE is a valid cold-GK104 state.
    # A dead BAR is diagnosed by the scrub/liveness checks, not by this value.
    dev.write32(PMC_ENABLE, pmc & ~0x00002000)
    dev.read32(PMC_ENABLE)  # flush the disable before re-enabling the engine
    time.sleep(settle_s)
    dev.write32(PMC_ENABLE, pmc | 0x00002000)
    dev.read32(PMC_ENABLE)
    time.sleep(settle_s)
    # gt215_pmu_init(): do not touch IMEM/DMEM until HW scrubbing completes.
    deadline = time.monotonic() + 2.0
    while True:
      scrub = dev.read32(PMU_FALCON_BASE + 0x10c) & 0xffffffff
      if scrub != 0xffffffff and not (scrub & 0x00000006):
        break
      if time.monotonic() >= deadline:
        raise TimeoutError(
            f"PMU MC reset did not restore/scrub aperture: 0x10a10c={scrub:#x}")
      time.sleep(0.001)
    dev.write32(0x10a580, 0)
    return
  try:
    falcon_stop(dev, base, timeout_ms=100)
  except Exception:
    # Already wedged / power-gated — still pulse RESET.
    pass
  dev.write32(base + FALCON_UC_CTRL, 0x00000080)
  time.sleep(settle_s)
  dev.write32(base + FALCON_UC_CTRL, 0x00000000)
  time.sleep(settle_s)


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

# NV906F_SEMAPHORED: OPERATION_RELEASE | RELEASE_WFI_DIS (bit20).
# NOT 0x01000002 — that sets RELEASE_SIZE_4BYTE (bit24) and leaves WFI enabled.
# After SET_OBJECT/LAUNCH leaves PGRAPH sticky-busy, a WFI RELEASE never stores.
GK104_SEM_RELEASE_NO_WFI = 0x00100002

def gk104_semaphore(addr, value, operation):
  """Emit the GK104/NV906F subchannel semaphore sequence."""
  return [*nvm(0, 0x0010, (addr >> 32) & 0xffffffff,
              addr & 0xffffffff, value, operation)]

def build_launch_words(timeline_addr, wait_value, done_value, launch_desc_addr,
                       code_va=0, temp_va=0, temp_size=GK104_TEMP_SIZE):
  # Kepler compute launch (envytools gk104_compute.xml, plan §24.3): bind the
  # compute class via SET_OBJECT, set the shader PC via CODE_ADDRESS_LO/HI
  # (0x160c/0x1608), point at the CWD via LAUNCH_DESC_ADDRESS (0x02b4, VA>>8),
  # then LAUNCH (0x02bc, value=3) to trigger.  Semaphore + cache-invalidate wrap.
  # Setup methods match Mesa's nve4_compute_setup_state() for Kepler.
  # GTX 770 has eight MPs.  NVE4 exposes two identical per-MP TEMP size banks;
  # Mesa programs both and rounds the low word down to a 32-KiB granule.  The
  # QMD requests 0x800 bytes of CRS per warp, and GK104 can host 64 warps/MP,
  # so anything below 0x20000 per MP raises SKED TOTAL_TEMP_SIZE.
  words, _final = build_multi_launch_words(
      timeline_addr, done_value, [launch_desc_addr],
      code_va=code_va, temp_va=temp_va, temp_size=temp_size)
  return words


def _gk104_compute_setup_words(code_va, temp_va, temp_size, mp_count=None,
                                tic_va=0):
  """Mesa nve4_screen_compute_setup()-shaped methods shared by launch builders."""
  mp_count = int(mp_count or GK104_MP_COUNT)
  min_temp = mp_count * GK104_TEMP_PER_MP
  assert temp_size >= min_temp and temp_size % mp_count == 0, (
      f"temp_size={temp_size:#x} mp_count={mp_count} min={min_temp:#x}")
  temp_per_mp = temp_size // mp_count
  # Mesa programs the enable mask as 0xff; size is what tracks mp_count.
  # TIC/TSC: Mesa always points these at a real VRAM heap even for pure
  # global LD/ST kernels (TEX_CB_INDEX=7).  Leaving them 0 left SKED quiet
  # but SMs never retired stores on this 660 Ti path.
  words = [
    *nvm(1, 0x0000, KEPLER_COMPUTE_A),            # SET_OBJECT: bind compute class
    *nvm(1, 0x0790, temp_va >> 32, temp_va & 0xffffffff), # TEMP_ADDRESS HIGH/LOW
    *nvm(1, 0x02e4, temp_per_mp >> 32, temp_per_mp & ~0x7fff, 0xff), # MP_TEMP_SIZE[0]
    *nvm(1, 0x02f0, temp_per_mp >> 32, temp_per_mp & ~0x7fff, 0xff), # MP_TEMP_SIZE[1]
    *nvm(1, 0x077c, 0xff << 24),                  # LOCAL_BASE
    *nvm(1, 0x0214, 0xfe << 24),                  # SHARED_BASE
    *nvm(1, 0x0310, 0x300),                       # SASS_VERSION (Kepler)
  ]
  if tic_va:
    tsc_va = tic_va + 0x10000  # Mesa: TSC follows 2048*32B TIC entries
    words.extend([
      *nvm(1, 0x1574, tic_va >> 32, tic_va & 0xffffffff, 2047),  # TIC_ADDRESS
      *nvm(1, 0x155c, tsc_va >> 32, tsc_va & 0xffffffff, 2047),  # TSC_ADDRESS
    ])
  words.extend([
    *nvm(1, 0x2608, 7),                           # TEX_CB_INDEX (Mesa NVE4 setup)
    *nvm(1, 0x1698, 0x00001011),                  # FLUSH CODE|GLOBAL|CB
    *nvm(1, 0x1608, code_va >> 32, code_va & 0xffffffff), # CODE_ADDRESS HIGH/LOW
  ])
  return words


def build_multi_launch_words(timeline_addr, done_value, launch_desc_addrs,
                             code_va=0, temp_va=0, temp_size=GK104_TEMP_SIZE,
                             batch=None, mp_count=None, tic_va=0):
  """One compute setup + many LAUNCH_DESC/LAUNCH pairs + one WFI semaphore.

  # Live GK104 retires at most ~19 LAUNCHes per channel lifetime (20th leaves
  # work as NaN even when a later GPFIFO entry is consumed).  Mid-stream WFI
  # then more LAUNCHes also hangs.  Use channel-windowed reopen for large N
  # (see _run_hardware_demo_windows) or KEPLER_MULTI_CTA=auto (N≤20480).
  """
  assert launch_desc_addrs, "need at least one CWD"
  # batch reserved for experiments; default = all launches then one WFI.
  if batch is None:
    batch = int(os.environ.get("KEPLER_LAUNCH_BATCH", "0"), 0)
  words = _gk104_compute_setup_words(code_va, temp_va, temp_size, mp_count=mp_count,
                                      tic_va=tic_va)
  done = done_value
  n = len(launch_desc_addrs)
  for i, launch_desc_addr in enumerate(launch_desc_addrs):
    words.extend([
      *nvm(1, 0x02b4, launch_desc_addr >> 8),     # LAUNCH_DESC_ADDRESS
      *nvm(1, 0x02bc, 0x3),                        # LAUNCH
      *nvm(1, 0x0110, 0),                          # NV50_GRAPH_SERIALIZE
    ])
    if batch > 0 and ((i + 1) % batch == 0 or (i + 1) == n):
      words.extend([
        *gk104_semaphore(timeline_addr, done, GK104_SEM_RELEASE_NO_WFI),
      ])
      if (i + 1) < n:
        done += 1
  if batch <= 0:
    # RELEASE_WFI=DIS: SET_OBJECT leaves PGRAPH_STATUS sticky-busy on this
    # 7-TPC GK104, so a WFI RELEASE never stores even after SERIALIZE.  Rely
    # on NV50_GRAPH_SERIALIZE above for method retirement; verify compute by
    # reading the GPU-written output, not by WFI alone.
    words.extend([
      *gk104_semaphore(timeline_addr, done, GK104_SEM_RELEASE_NO_WFI),
    ])
  words.extend([*nvm(0, 0x0020, 0)])
  return words, done


def build_multi_launch_ibs(timeline_addr, done_value, launch_desc_addrs,
                           code_va=0, temp_va=0, temp_size=GK104_TEMP_SIZE,
                           ib_max=None, mp_count=None, tic_va=0):
  """Split chunk launches across GPFIFO IBs of ≤ib_max LAUNCHes each.

  First IB includes compute setup; every IB ends with its own WFI semaphore
  (done_value, done_value+1, ...).  Host must stage GP_PUT one entry at a
  time and wait between IBs — publishing all entries up front leaves the
  second IB hung after the first WFI.  Returns (ib_word_lists, final_done).
  """
  assert launch_desc_addrs, "need at least one CWD"
  if ib_max is None:
    ib_max = int(os.environ.get("KEPLER_LAUNCH_IB_MAX", "16"), 0)
  if ib_max <= 0:
    words, done = build_multi_launch_words(
        timeline_addr, done_value, launch_desc_addrs,
        code_va=code_va, temp_va=temp_va, temp_size=temp_size,
        mp_count=mp_count, tic_va=tic_va)
    return [words], done
  batches = []
  done = done_value
  n = len(launch_desc_addrs)
  for start in range(0, n, ib_max):
    addrs = launch_desc_addrs[start:start + ib_max]
    words = []
    # Every IB re-binds compute state for same-channel staged GP_PUT.
    words.extend(_gk104_compute_setup_words(
        code_va, temp_va, temp_size, mp_count=mp_count, tic_va=tic_va))
    for launch_desc_addr in addrs:
      words.extend([
        *nvm(1, 0x02b4, launch_desc_addr >> 8),
        *nvm(1, 0x02bc, 0x3),
        *nvm(1, 0x0110, 0),
      ])
    last = (start + ib_max) >= n
    # Live: any WFI after the first IB's completion leaves the next IB's
    # semaphore unretired (GET advances, DMA_GET moves, sem stuck).  All
    # multi-IB semaphores use RELEASE_WFI=DIS; host still stages on the value.
    words.extend([
      *gk104_semaphore(timeline_addr, done, GK104_SEM_RELEASE_NO_WFI),
      *nvm(0, 0x0020, 0),
    ])
    batches.append(words)
    if not last:
      done += 1
  return batches, done

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

def build_cuda_param_cbuf(*ptrs, grid=(1, 1, 1), block=(1, 1, 1)):
  """CUDA user const bank 0: driver ABI dims at 0x28.., kernel params at 0x140.

  sm_30 ptxas emits `IMAD idx, ctaid, c[0][0x28], tid` for blockIdx*blockDim+tid,
  so bare launches must plant blockDim/gridDim themselves (no CUDA driver).
  """
  cbuf = bytearray(0x200)
  gx, gy, gz = grid
  bx, by, bz = block
  # KeplerAs / Maxas ABI: blockDim.{x,y,z} @ 0x28/0x2c/0x30, gridDim @ 0x34/0x38/0x3c.
  struct.pack_into("<6I", cbuf, 0x28, bx, by, bz, gx, gy, gz)
  struct.pack_into(f"<{len(ptrs)}Q", cbuf, 0x140, *ptrs)
  return cbuf

def build_cwd(code_addr, grid, block, shared=0, cbuf_addr=0, cbuf_size=0x200, regs=4,
              cb7_addr=0, cb7_size=0x800):
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
  # Mesa always binds CB7 (driver aux) because TEX_CB_INDEX=7.
  if cb7_addr:
    field(640 + 7, 640 + 7, 1)  # CONSTANT_BUFFER_VALID(7)
    # CB index i uses bits at +64*i from CB0's fields.
    field(928 + 7 * 64, 959 + 7 * 64, cb7_addr & 0xffffffff)
    field(960 + 7 * 64, 967 + 7 * 64, (cb7_addr >> 32) & 0xff)
    field(975 + 7 * 64, 991 + 7 * 64, cb7_size)
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
  # Multi-IB splitter: 20 CWDs with ib_max=16 → 2 IBs, done values 2 then 3.
  _ibs, _done = build_multi_launch_ibs(
      0x1000, 2, list(range(0x1000, 0x1000 + 20 * 0x100, 0x100)),
      code_va=0x3000, temp_va=0x4000, ib_max=16)
  assert len(_ibs) == 2 and _done == 3, (_ibs, _done)
  _launch0 = sum(1 for _, _, _, m, _, _ in decode_words(_ibs[0]) if m == 0x02bc)
  _launch1 = sum(1 for _, _, _, m, _, _ in decode_words(_ibs[1]) if m == 0x02bc)
  assert (_launch0, _launch1) == (16, 4), (_launch0, _launch1)
  assert sum(1 for _, _, _, m, _, _ in decode_words(_ibs[0]) if m == 0x0010) == 1
  assert sum(1 for _, _, _, m, _, _ in decode_words(_ibs[1]) if m == 0x0010) == 1
  # Re-kick IBs re-bind SET_OBJECT (setup on every IB).
  assert any(m == 0x0000 for _, _, _, m, _, _ in decode_words(_ibs[1]))
  assert gk104_mp_trap_addrs(0, 0) == (0x504648, 0x504650)
  assert gk104_mp_trap_addrs(1, 1) == (0x50ce48, 0x50ce50)
  params = build_cuda_param_cbuf(0x1122334455667788, 2, 3, grid=(2, 1, 1), block=(256, 1, 1))
  assert len(params) == 0x200 and struct.unpack_from("<3Q", params, 0x140) == (0x1122334455667788, 2, 3)
  assert struct.unpack_from("<6I", params, 0x28) == (256, 1, 1, 2, 1, 1)
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

  # The cold GDDR5 controller port must retain Nouveau's fractional reference
  # PLL accumulator and the Palit strap-6 RAMCFG selection.  Seed 0x101000 with
  # the golden mmiotrace value (0x8040509a → strap 6).
  class _FakeRamRegs:
    def __init__(self):
      self.regs = {0x101000: 0x8040509a, 0x022438: 0, 0x022554: 0,
                   0x100710: 0x80000000, 0x137390: 0x00020000,
                   0x10f65c: 0, 0x10f584: 0x15004000, 0x10f160: 0x3}
      self.writes = []
    def read32(self, reg): return self.regs.get(reg, 0)
    def write32(self, reg, value):
      value &= 0xffffffff
      self.regs[reg] = value
      self.writes.append((reg, value))
  _ram_image = nvbios_init.find_vbios_image(pathlib.Path(PALIT_770_VBIOS).read_bytes())
  _or_regs = _FakeRamRegs()
  _or_addr = 0x123400
  _or_regs.regs[_or_addr] = 0x00000100
  _or_image = bytearray(9)
  struct.pack_into("<II", _or_image, 1, _or_addr, 0x00000024)
  _or_init = nvbios_init.NvbiosInit(_or_regs, bytes(_or_image))
  _or_init._op_or_reg()
  assert _or_regs.regs[_or_addr] == 0x00000124
  assert _or_init.offset == 9
  # Nouveau creates fresh execute/selector state for every top-level script.
  # A false condition in one script must not suppress the next script.
  _state_image = bytearray(0x50)
  _state_image[0x20:0x23] = bytes((0x75, 0x00, 0x71))
  struct.pack_into("<BII", _state_image, 0x30,
                   0x7a, 0x00123400, 0x89abcdef)
  _state_image[0x39] = 0x71
  _state_regs = _FakeRamRegs()
  _state_init = nvbios_init.NvbiosInit(_state_regs, bytes(_state_image))
  _state_init.run_script(0x20)
  assert _state_init.execute == 3
  _state_init._ramcfg_value = 6
  _state_init.run_script(0x30)
  assert _state_regs.regs[0x00123400] == 0x89abcdef
  assert _state_init.execute == 1
  assert _state_init._ramcfg_value is None
  # Night41x nested bisect: stop_before halts top-level progress inside 0x8fe8.
  _stop_image = nvbios_init.find_vbios_image(
      pathlib.Path(PALIT_770_VBIOS).read_bytes())
  _stop_regs = _FakeRamRegs()
  _stop_init = nvbios_init.NvbiosInit(_stop_regs, _stop_image)
  _stop_init.run_script(0x87e5)
  _n_after_87e5 = len(_stop_regs.writes)
  _stop_init.run_script(0x8fe8, stop_before=0x9e34)
  assert _stop_init.offset >= 0x9e34, _stop_init.offset
  assert len(_stop_regs.writes) > _n_after_87e5
  class _FailingNvinitRegs:
    def read32(self, reg): raise OSError("transport read failed")
    def write32(self, reg, value): raise OSError("transport write failed")
  _failing_init = nvbios_init.NvbiosInit(_FailingNvinitRegs(), b"")
  for _fn in (lambda: _failing_init._reg_rd32(0x1234),
              lambda: _failing_init._reg_wr32(0x1234, 0x55),
              lambda: _failing_init._reg_mask(0x1234, 0xff, 0x55)):
    try:
      _fn()
      raise AssertionError("NVINIT transport failure was swallowed")
    except RuntimeError as _exc:
      assert "0x00001234" in str(_exc)
  class _FakeVgaRegs:
    def __init__(self):
      self.data = {0x6013d5: 0x20}
      self.ops = []
    def read8(self, reg):
      self.ops.append(("R8", reg, self.data.get(reg, 0)))
      return self.data.get(reg, 0)
    def write8(self, reg, value):
      self.ops.append(("W8", reg, value & 0xff))
      self.data[reg] = value & 0xff
  _vga_regs = _FakeVgaRegs()
  _vga_init = nvbios_init.NvbiosInit(_vga_regs, _ram_image)
  assert _vga_init._vgai_rd(0x03d4, 0x8d) == 0x20
  assert _vga_regs.ops == [
    ("W8", 0x6013d4, 0x8d), ("R8", 0x6013d5, 0x20),
  ], f"Nouveau VGA indexed read mismatch: {_vga_regs.ops}"
  _vga_regs.ops.clear()
  _vga_init._vgai_wr(0x03c4, 0x01, 0x20)
  assert _vga_regs.ops == [
    ("W8", 0x6013c4, 0x01), ("W8", 0x6013c5, 0x20),
  ], f"Nouveau VGA indexed write mismatch: {_vga_regs.ops}"
  _vga_regs.ops.clear()
  _latched_image = bytes((0x51, 0x10, 0x11, 0x20, 0x02, 0xaa, 0xbb))
  _vga_regs.data[0x6013d5] = 0x5a
  _latched_init = nvbios_init.NvbiosInit(_vga_regs, _latched_image)
  _latched_init._op_cr_idx_adr_latch()
  assert _latched_init.offset == len(_latched_image)
  assert _vga_regs.ops == [
    ("W8", 0x6013d4, 0x10), ("R8", 0x6013d5, 0x5a),
    ("W8", 0x6013d4, 0x10), ("W8", 0x6013d5, 0x20),
    ("W8", 0x6013d4, 0x11), ("W8", 0x6013d5, 0xaa),
    ("W8", 0x6013d4, 0x10), ("W8", 0x6013d5, 0x21),
    ("W8", 0x6013d4, 0x11), ("W8", 0x6013d5, 0xbb),
    ("W8", 0x6013d4, 0x10), ("W8", 0x6013d5, 0x5a),
  ], f"Nouveau CR index/address latch mismatch: {_vga_regs.ops}"
  _vga_regs.ops.clear()
  _vga_init.unlock_vga_crtc()
  assert _vga_regs.ops == [
    ("W8", 0x6013d4, 0x3f), ("W8", 0x6013d5, 0x57),
  ], f"Nouveau VGA CRTC unlock mismatch: {_vga_regs.ops}"
  _vga_regs.ops.clear()
  _vga_init.option_rom_vga_enable_prefix()
  assert _vga_regs.ops == [
    ("W8", 0x6013c3, 0x01), ("R8", 0x6013c3, 0x01),
    ("W8", 0x6013c2, 0x01), ("R8", 0x6013c3, 0x01),
    ("W8", 0x6013d4, 0x3f), ("W8", 0x6013d5, 0x57),
  ], f"option-ROM VGA preamble mismatch: {_vga_regs.ops}"
  class _FakeInitIORegs(_FakeVgaRegs):
    def __init__(self):
      super().__init__()
      self.regs = {0x000200: 0x00002021}
      self.mmio_writes = []
    def read32(self, reg): return self.regs.get(reg, 0)
    def write32(self, reg, value):
      value &= 0xffffffff
      self.regs[reg] = value
      self.mmio_writes.append((reg, value))
  _io_regs = _FakeInitIORegs()
  _io_image = bytes((0x69, 0xc3, 0x03, 0x00, 0x01))
  _io_init = nvbios_init.NvbiosInit(_io_regs, _io_image)
  _io_init._op_io()
  assert _io_init.offset == 5
  assert _io_regs.mmio_writes == [
    (0x614100, 0x00800000), (0x00e18c, 0x00020000),
    (0x614900, 0x00800000), (0x000200, 0x00002021),
    (0x00e18c, 0x00000000), (0x000200, 0x40002021),
    (0x614100, 0x00800018), (0x614900, 0x00800018),
    (0x614100, 0x10000018), (0x614900, 0x10000018),
  ], f"Nouveau INIT_IO special sequence mismatch: {_io_regs.mmio_writes}"
  assert _io_regs.ops == [
    ("R8", 0x6013c3, 0), ("W8", 0x6013c3, 1),
  ], f"Nouveau INIT_IO port sequence mismatch: {_io_regs.ops}"
  _i2c_init = nvbios_init.NvbiosInit(_FakeRamRegs(), _ram_image)
  assert _i2c_init._i2c_bus_reg(0x80) == 0x00d054, \
      "Palit DCB primary I2C must resolve CCB2/drive2 to 0xd054"
  _i2c_if = nvbios_init.NvbiosInit(
      _FakeRamRegs(), bytes((0x5e, 0x80, 0x40, 0x99, 0xff, 0x41)))
  _i2c_reads = []
  def _fake_i2c_read(index, addr, reg):
    _i2c_reads.append((index, addr, reg))
    return 0x41
  _i2c_if._i2c_read_reg = _fake_i2c_read
  _i2c_if._op_i2c_if()
  assert _i2c_reads == [(0x80, 0x40, 0x99)]
  assert _i2c_if.offset == 6 and _i2c_if.execute == 1
  _i2c_if_fail = nvbios_init.NvbiosInit(
      _FakeRamRegs(), bytes((0x5e, 0x80, 0x40, 0x9a, 0x0f, 0x08)))
  def _fake_i2c_fail(index, addr, reg):
    raise nvbios_init.NvbiosI2CError("offline NACK")
  _i2c_if_fail._i2c_read_reg = _fake_i2c_fail
  _i2c_if_fail._op_i2c_if()
  assert _i2c_if_fail.offset == 6 and _i2c_if_fail.execute == 3, \
      "Nouveau I2C_IF -EIO must convert to 0xfb and select false branch"
  _i2c_zm = nvbios_init.NvbiosInit(
      _FakeRamRegs(), bytes((0x4d, 0x80, 0x40, 0x01, 0xdd, 0x03)))
  _i2c_writes = []
  _i2c_zm._i2c_write_reg = lambda index, addr, reg, value: \
      _i2c_writes.append((index, addr, reg, value))
  _i2c_zm._op_zm_i2c_byte()
  assert _i2c_writes == [(0x80, 0x20, 0xdd, 0x03)]
  assert _i2c_zm.offset == 6
  _cr_regs = _FakeVgaRegs()
  _cr_init = nvbios_init.NvbiosInit(
      _cr_regs, bytes((0x52, 0x8d, 0xf0, 0x05,
                       0x53, 0x85, 0xff,
                       0x54, 0x02, 0x80, 0x01, 0x81, 0x02)))
  _cr_init._op_cr()
  _cr_init._op_zm_cr()
  _cr_init._op_zm_cr_group()
  assert _cr_init.offset == 13
  assert _cr_regs.ops == [
    ("W8", 0x6013d4, 0x8d), ("R8", 0x6013d5, 0x20),
    ("W8", 0x6013d4, 0x8d), ("W8", 0x6013d5, 0x25),
    ("W8", 0x6013d4, 0x85), ("W8", 0x6013d5, 0xff),
    ("W8", 0x6013d4, 0x80), ("W8", 0x6013d5, 0x01),
    ("W8", 0x6013d4, 0x81), ("W8", 0x6013d5, 0x02),
  ], f"Nouveau CRTC opcode sequence mismatch: {_cr_regs.ops}"
  _gpio_regs = _FakeRamRegs()
  _gpio_init = nvbios_init.NvbiosInit(_gpio_regs, _ram_image)
  assert _gpio_init.rd08(0x8b2d) == 0x8e
  _gpio_init.offset = 0x8b2d
  _gpio_init._op_gpio()
  _active_gpios = [gpio for gpio in _gpio_init.dcb_gpios()
                   if gpio["func"] != 0xff]
  assert len(_active_gpios) == 23
  assert sum(addr == 0x00d604 for addr, _value in _gpio_regs.writes) == 23
  _expected_routes = {}
  for _gpio in _active_gpios:
    _level = _gpio["log1"] if _gpio["defs"] else _gpio["log0"]
    _got = _gpio_regs.regs[_gpio["reg"]]
    assert (_got & 0x00003000) == ((_level ^ 2) << 12)
    assert (_got & 0xff) == _gpio["unk0"]
    if _gpio["unk1"]:
      _expected_routes[0x00d740 + (_gpio["unk1"] - 1) * 4] = _gpio["line"]
  for _reg, _line in _expected_routes.items():
    assert (_gpio_regs.regs[_reg] & 0xff) == _line
  _ram_regs = _FakeRamRegs()
  _ram_regs.regs[0x10f65c] = 0xa5a50000
  # RAMMAP/training prefix must match the Nouveau golden mmiotrace.
  nvbios_init.run_vbios_ram_init(_ram_regs, _ram_image)
  _early = [(r, v) for r, v in _ram_regs.writes
            if r in (0x10f65c, 0x11e67c, 0x11e708, 0x11e6a0, 0x11e6a4,
                     0x11e6a8, 0x11e6ac, 0x11e6b0, 0x11e6b4)]
  assert _early[:9] == [
    (0x10f65c, 0xa5a50010), (0x11e67c, 0xfff10000), (0x11e708, 0x00030222),
    (0x11e6a0, 0x04040404), (0x11e6a4, 0x04040404), (0x11e6a8, 0x0f0f0f0f),
    (0x11e6ac, 0x0f0f0f0f), (0x11e6b0, 0x06060606), (0x11e6b4, 0x06060606),
  ], f"RAMMAP prefix mismatch vs golden: {_early[:9]}"
  assert _ram_regs.regs[0x10f65c] == 0xa5a50000, \
      f"RAMMAP selector clobbered unrelated bits: {_ram_regs.regs[0x10f65c]:#x}"
  assert any(r == 0x10f918 and v == 0x55555555 for r, v in _ram_regs.writes), \
      "strap-6 training type00[0] must be 0x55555555"
  _ram_regs.writes.clear()
  _cfg = nvbios_init.run_vbios_ram_program(_ram_regs, _ram_image, freq_mhz=648)
  assert _cfg["ramcfg_index"] == 6 and _cfg["timing_index"] == 0
  assert _ram_regs.regs[0x132024] == 0x00011701, \
      f"unexpected GK104 648MHz PLL coefficient {_ram_regs.regs.get(0x132024, 0):#x}"
  assert _ram_regs.regs[0x132030] == 0x10000000
  assert _ram_regs.regs[0x132034] == 0x00001000
  _pll_match = _FakeRamRegs()
  _pll_match.regs.update({0x132024: 0x00011701, 0x132034: 0x00001000})
  nvbios_init.run_vbios_ram_program(_pll_match, _ram_image, freq_mhz=648)
  _refpll_program_regs = {
      0x132000, 0x132020, 0x137320, 0x132030,
      0x132034, 0x132024, 0x132028,
  }
  assert not any(reg in _refpll_program_regs for reg, _value in _pll_match.writes), \
      "matching Nouveau REFPLL coefficients must skip reprogramming"
  # Strap-6 GDDR5 path: !01_04 and !07_80 → data 0x32a00000 on top of the
  # early 0x40000000 enter bit, matching gk104_ram_calc_gddr5().
  assert _ram_regs.regs[0x10f808] == 0x72a00000, \
      f"unexpected GK104 0x10f808 {_ram_regs.regs.get(0x10f808, 0):#x}"
  # Default KEPLER_RAM_BLOCK=enter without PMU falls back to skip (no host 0x1620).
  assert not any(r == 0x1620 for r, _ in _ram_regs.writes), \
      "default cold RAM program must not emit host 0x1620 pause masks"
  _direct = _FakeRamRegs()
  os.environ["KEPLER_RAM_BLOCK"] = "direct"
  try:
    nvbios_init.run_vbios_ram_program(_direct, _ram_image, freq_mhz=648)
  finally:
    os.environ.pop("KEPLER_RAM_BLOCK", None)
  assert any(r == 0x1620 for r, _ in _direct.writes), \
      "KEPLER_RAM_BLOCK=direct must emit host 0x1620 pause masks"

  class _FakeAtomicRamRegs(_FakeRamRegs):
    def __init__(self):
      super().__init__()
      self.regs.update({0x1620: 0xaab, 0x26f0: 1, 0x022438: 2,
                        0x110974: 0, 0x111974: 0})
      self._pmu_memx_data = (0x3cc, 0x800)
      self._pmu_memx_nowait = True
      self.exec_calls = []
    def pmu_memx_exec_commands(self, commands, timeout_s=5.0):
      commands = [(op, tuple(payload)) for op, payload in commands]
      self.exec_calls.append(commands)
      for opcode, payload in commands:
        if opcode == PMU_MEMX_ENTER:
          self.regs[0x1620] = self.regs[0x1620] & ~0xaa3
          self.regs[0x26f0] = self.regs[0x26f0] & ~1
        elif opcode == PMU_MEMX_LEAVE:
          self.regs[0x26f0] = self.regs[0x26f0] | 1
          self.regs[0x1620] = self.regs[0x1620] | 0xaa3
        elif opcode == PMU_MEMX_WR32:
          assert len(payload) % 2 == 0
          for i in range(0, len(payload), 2):
            self.write32(payload[i], payload[i + 1])
        elif opcode == PMU_MEMX_WAIT:
          reg, mask, value, _nsec = payload
          assert (self.read32(reg) & mask) == value
        else:
          assert opcode == PMU_MEMX_DELAY
      return (0, 0)

  _atomic_ram = _FakeAtomicRamRegs()
  _atomic_ram_env = {key: os.environ.get(key) for key in (
      "KEPLER_RAM_MEMX_ATOMIC", "KEPLER_RAM_BLOCK",
      "KEPLER_TRAIN_WAIT_NS")}
  try:
    os.environ["KEPLER_RAM_MEMX_ATOMIC"] = "1"
    os.environ["KEPLER_RAM_BLOCK"] = "atomic"
    os.environ.pop("KEPLER_TRAIN_WAIT_NS", None)
    nvbios_init.run_vbios_ram_program(
        _atomic_ram, _ram_image, freq_mhz=648)
  finally:
    for _key, _value in _atomic_ram_env.items():
      if _value is None:
        os.environ.pop(_key, None)
      else:
        os.environ[_key] = _value
  # One smoke DELAY, one minimal ENTER+LEAVE preflight, one indivisible RAM
  # transition, then a separate smoke/target-prog0 phase.
  assert len(_atomic_ram.exec_calls) == 5, _atomic_ram.exec_calls
  assert [op for op, _payload in _atomic_ram.exec_calls[1]] == [
      PMU_MEMX_ENTER, PMU_MEMX_LEAVE]
  _atomic_script = _atomic_ram.exec_calls[2]
  _atomic_ops = [op for op, _payload in _atomic_script]
  assert _atomic_ops.count(PMU_MEMX_ENTER) == 1
  assert _atomic_ops.count(PMU_MEMX_LEAVE) == 1
  assert _atomic_ops.index(PMU_MEMX_ENTER) < _atomic_ops.index(PMU_MEMX_LEAVE)
  _enter_index = _atomic_ops.index(PMU_MEMX_ENTER)
  _leave_index = _atomic_ops.index(PMU_MEMX_LEAVE)
  _display_commands = [
      (i, value) for i, (op, payload) in enumerate(_atomic_script)
      if op == PMU_MEMX_WR32
      for addr, value in zip(payload[0::2], payload[1::2])
      if addr == 0x62c000]
  assert len(_display_commands) == 2, _display_commands
  assert _display_commands == [
      (_display_commands[0][0], 0x0f0f0000),
      (_display_commands[1][0], 0x0f0f0f00),
  ], _display_commands
  assert (_enter_index < _display_commands[0][0] < _leave_index <
          _display_commands[1][0]), (
      "Nouveau brackets GDDR5 RAMFUC with display-memory quiesce")
  _mr1_command_indices = [
      i for i, (op, payload) in enumerate(_atomic_script)
      if op == PMU_MEMX_WR32 and 0x10f330 in payload[0::2]]
  _prog0_command_indices = [
      i for i, (op, payload) in enumerate(_atomic_script)
      if op == PMU_MEMX_WR32 and
      ({0x10f468, 0x10f420, 0x10f430, 0x10f400,
        0x10f410, 0x10f440, 0x10f444} & set(payload[0::2]))]
  assert _mr1_command_indices and _mr1_command_indices[0] > _enter_index, (
      "Nouveau queues early MR1 termination after MEMX ENTER")
  assert _prog0_command_indices and max(_prog0_command_indices) < _enter_index, (
      "Nouveau applies prog0(1000) before executing the ENTER transition")
  _atomic_waits = [payload for op, payload in _atomic_script
                   if op == PMU_MEMX_WAIT]
  assert (0x137390, 0x00020000, 0x00020000, 64_000) in _atomic_waits
  assert (0x110974, 0x0000000f, 0x00000000, 500_000) in _atomic_waits
  assert (0x111974, 0x0000000f, 0x00000000, 500_000) in _atomic_waits
  for _wait in _atomic_waits:
    if _wait[0] == 0x100710:
      assert _wait[3] == 200_000, _wait
  _atomic_writes = [(addr, value)
                    for op, payload in _atomic_script if op == PMU_MEMX_WR32
                    for addr, value in zip(payload[0::2], payload[1::2])]
  _r1373f4_values = [value for addr, value in _atomic_writes
                     if addr == 0x1373f4]
  assert _r1373f4_values[-4:] == [
      0x00010100, 0x00010110, 0x00010111, 0x00000111,
  ], f"Nouveau mode-1 r1373f4 init/fini changed: {_r1373f4_values}"
  _timing_regs = (0x10f248, 0x10f290, 0x10f294, 0x10f298,
                  0x10f29c, 0x10f2a0, 0x10f2a4, 0x10f2a8,
                  0x10f2ac, 0x10f2cc, 0x10f2e8)
  _timing_values = (_cfg["timing"][10], *_cfg["timing"][:10])
  _last_atomic_value = {
      reg: [value for addr, value in _atomic_writes if addr == reg][-1]
      for reg in _timing_regs
  }
  assert tuple(_last_atomic_value[reg] for reg in _timing_regs) == \
      _timing_values, "Nouveau PFB timing[10], timing[0..9] mapping changed"
  _index_694 = [i for i, (addr, _value) in enumerate(_atomic_writes)
                if addr == 0x10f694]
  assert len(_index_694) == 1
  assert (_atomic_writes.index((0x10f69c, 0)) < _index_694[0] <
          next(i for i, (addr, _value) in enumerate(_atomic_writes)
               if addr == 0x10f248)), \
      "Nouveau forces one preserved 0x10f694 update before PFB timing"
  assert not any(addr == 0x10f60c for addr, _value in _atomic_writes), \
      "unchanged ramfuc_mask(0x10f60c) must not queue a write"
  _last_100770 = max(i for i, (addr, _value) in enumerate(_atomic_writes)
                     if addr == 0x100770)
  _last_100778 = max(i for i, (addr, _value) in enumerate(_atomic_writes)
                     if addr == 0x100778)
  assert _last_100770 < _last_100778, \
      "Nouveau completes the 0x100770 transition before 0x100778"
  assert (PMU_MEMX_DELAY, (10_000,)) in _atomic_script
  assert (PMU_MEMX_DELAY, (10_000_000,)) not in _atomic_script
  # Both Palit termination fields are 3, so the source-correct main/nuts MR1
  # split is intentionally dormant for this board image.
  assert _cfg["timing_20_2e_30"] == _cfg["timing_20_2e_c0"] == 3
  _mr5_indices = [i for i, (addr, _value) in enumerate(_atomic_writes)
                  if addr == 0x10f340]
  # Fake MR5 starts at the final source value for this RAMCFG, so both MR5
  # masks are unchanged and ramfuc correctly suppresses them.  0x10f830 is
  # also touched earlier; identify the final mode pulse by its last two values.
  assert _cfg["ramcfg_11_07_02"] and not _mr5_indices, _mr5_indices
  _mode_writes = [(i, value) for i, (addr, value) in enumerate(_atomic_writes)
                  if addr == 0x10f830]
  assert len(_mode_writes) >= 2, _mode_writes
  (_mode_set_i, _mode_set), (_mode_clear_i, _mode_clear) = _mode_writes[-2:]
  assert (_mode_set_i < _mode_clear_i and
          (_mode_set & 0x01000000) == 0x01000000 and
          (_mode_clear & 0x01000000) == 0 and
          (_mode_set ^ _mode_clear) == 0x01000000), (
      "Nouveau final 0x10f830[24] set/clear pulse changed", _mode_writes)
  _prog0_regs = {0x10f468, 0x10f420, 0x10f430, 0x10f400,
                 0x10f410, 0x10f440, 0x10f444}
  _after_leave = False
  for _op, _payload in _atomic_script:
    if _op == PMU_MEMX_LEAVE:
      _after_leave = True
    elif _after_leave and _op == PMU_MEMX_WR32:
      assert not (_prog0_regs & set(_payload[0::2])), _payload
  assert _atomic_ram.exec_calls[3] == [(PMU_MEMX_DELAY, (1000,))]
  _prog0_script = _atomic_ram.exec_calls[4]
  assert _prog0_script and all(op == PMU_MEMX_WR32
                               for op, _payload in _prog0_script)
  _prog0_addrs = {addr for _op, _payload in _prog0_script
                  for addr in _payload[0::2]}
  assert _prog0_addrs and _prog0_addrs <= _prog0_regs, _prog0_addrs
  assert _atomic_ram.regs[0x1620] == 0xaab
  assert _atomic_ram.regs[0x26f0] & 1

  # Night41a/H48: live night40az entered with selector 1 and the source
  # REFPLL image 0x32301/0x1000, which gk104_clk_read() decodes as 324 MHz.
  # Nouveau performs an active xition pass before the 648-MHz target because
  # timing_20_30_07 differs between those two RAMMAP entries.
  _xition_ram = _FakeAtomicRamRegs()
  _xition_ram.regs.update({
      0x1373f4: 0x1, 0x132020: 0x1, 0x132024: 0x00032301,
      0x132030: 0x10000000, 0x132034: 0x00001000, 0x137320: 0,
  })
  assert nvbios_init._gk104_read_mem_clock_khz(_xition_ram) == 324_000
  _xition_env = {key: os.environ.get(key) for key in (
      "KEPLER_RAM_MEMX_ATOMIC", "KEPLER_RAM_BLOCK")}
  try:
    os.environ["KEPLER_RAM_MEMX_ATOMIC"] = "1"
    os.environ["KEPLER_RAM_BLOCK"] = "atomic"
    nvbios_init.run_vbios_ram_program(
        _xition_ram, _ram_image, freq_mhz=648)
  finally:
    for _key, _value in _xition_env.items():
      if _value is None:
        os.environ.pop(_key, None)
      else:
        os.environ[_key] = _value
  _xition_scripts = [
      script for script in _xition_ram.exec_calls
      if len(script) > 2 and any(op == PMU_MEMX_ENTER for op, _ in script)]
  assert len(_xition_scripts) == 2, _xition_ram.exec_calls
  _xition_timing10 = []
  for _script in _xition_scripts:
    _writes = [(addr, value) for op, payload in _script
               if op == PMU_MEMX_WR32
               for addr, value in zip(payload[0::2], payload[1::2])]
    _xition_timing10.append(
        [value for addr, value in _writes if addr == 0x10f248][-1])
  assert _xition_timing10 == [0x031455a5, 0x06086442], _xition_timing10

  # Night40aq: a 50 ms tight MEMX_WAIT wedged the PMU MMIO-read loop.  Reject
  # oversized diagnostics before they can queue another unrecoverable script.
  _oversize_wait = _FakeAtomicRamRegs()
  _oversize_error = None
  _oversize_env = {key: os.environ.get(key) for key in (
      "KEPLER_RAM_MEMX_ATOMIC", "KEPLER_RAM_BLOCK",
      "KEPLER_TRAIN_WAIT_NS")}
  try:
    os.environ["KEPLER_RAM_MEMX_ATOMIC"] = "1"
    os.environ["KEPLER_RAM_BLOCK"] = "atomic"
    os.environ["KEPLER_TRAIN_WAIT_NS"] = "50000000"
    nvbios_init.run_vbios_ram_program(
        _oversize_wait, _ram_image, freq_mhz=648)
  except ValueError as _error:
    _oversize_error = str(_error)
  finally:
    for _key, _value in _oversize_env.items():
      if _value is None:
        os.environ.pop(_key, None)
      else:
        os.environ[_key] = _value
  assert _oversize_error and "nanoseconds per partition" in _oversize_error
  assert not _oversize_wait.exec_calls

  _unsafe_atomic = _FakeAtomicRamRegs()
  _unsafe_atomic._pmu_memx_nowait = False
  _unsafe_error = None
  _unsafe_env = {key: os.environ.get(key) for key in (
      "KEPLER_RAM_MEMX_ATOMIC", "KEPLER_RAM_BLOCK", "KEPLER_RAM_ENTER_WAIT")}
  try:
    os.environ["KEPLER_RAM_MEMX_ATOMIC"] = "1"
    os.environ["KEPLER_RAM_BLOCK"] = "atomic"
    os.environ.pop("KEPLER_RAM_ENTER_WAIT", None)
    nvbios_init.run_vbios_ram_program(
        _unsafe_atomic, _ram_image, freq_mhz=648)
  except RuntimeError as _error:
    _unsafe_error = str(_error)
  finally:
    for _key, _value in _unsafe_env.items():
      if _value is None:
        os.environ.pop(_key, None)
      else:
        os.environ[_key] = _value
  assert _unsafe_error and (
      "_pmu_memx_nowait is false" in _unsafe_error or
      "KEPLER_RAM_ENTER_WAIT=1" in _unsafe_error), _unsafe_error
  assert not _unsafe_atomic.exec_calls

  # Night40ah: stock ENTER wait is allowed when explicitly opted in.
  _wait_atomic = _FakeAtomicRamRegs()
  _wait_atomic._pmu_memx_nowait = False
  _wait_env = {key: os.environ.get(key) for key in (
      "KEPLER_RAM_MEMX_ATOMIC", "KEPLER_RAM_BLOCK",
      "KEPLER_RAM_ENTER_WAIT", "KEPLER_RAM_ATOMIC_PREFLIGHT")}
  try:
    os.environ["KEPLER_RAM_MEMX_ATOMIC"] = "1"
    os.environ["KEPLER_RAM_BLOCK"] = "atomic"
    os.environ["KEPLER_RAM_ENTER_WAIT"] = "1"
    os.environ["KEPLER_RAM_ATOMIC_PREFLIGHT"] = "0"
    nvbios_init.run_vbios_ram_program(
        _wait_atomic, _ram_image, freq_mhz=648)
  finally:
    for _key, _value in _wait_env.items():
      if _value is None:
        os.environ.pop(_key, None)
      else:
        os.environ[_key] = _value
  assert _wait_atomic.exec_calls, "ENTER_WAIT atomic must EXEC the RAM script"
  _wait_ops = [op for call in _wait_atomic.exec_calls for op, _ in call]
  assert PMU_MEMX_ENTER in _wait_ops and PMU_MEMX_LEAVE in _wait_ops

  # Full golden-mmiotrace checkpoint suite (also available as
  # --mmiotrace-selftest; must pass on macOS before the next eGPU replug).
  import mmiotrace_selftest as _mmio_st
  _hooks = _mmio_st.build_hooks_from_add_module(sys.modules[__name__])
  assert _mmio_st.run_mmiotrace_selftest(_hooks, verbose=False) == 0
  assert not _gk104_topo_is_posted(0xffffffff)
  assert not _gk104_topo_is_posted(0xbadf1200)
  assert not _gk104_topo_is_posted(0)
  assert _gk104_topo_is_posted(0x00010004)
  assert _gk104_topo_is_posted(0x00040004)  # golden FECS-era topology
  assert not _gk104_pramin_word_is_stub(0xffffffff)  # virgin VRAM, not a stub
  assert not _gk104_pramin_word_is_stub(0)
  assert _gk104_pramin_word_is_stub(0xbad0fb14)
  assert _gk104_pramin_word_is_stub(0xbadf3010)
  assert not _gk104_pramin_word_is_stub(0x0000beef)
  assert not _gk104_pramin_word_is_stub(0xa5a5a5a5)
  class _FakeZmReg:
    def __init__(self): self.writes = []
    def read32(self, reg): return 0
    def write32(self, reg, value): self.writes.append((reg, value))
  _zm_image = bytearray(0x80)
  struct.pack_into("<BII", _zm_image, 0x20, 0x7a, 0x000200, 0x00002020)
  _zm_image[0x29] = 0x71
  _zm_dev = _FakeZmReg()
  nvbios_init.NvbiosInit(_zm_dev, bytes(_zm_image)).run_script(0x20)
  assert _zm_dev.writes == [(0x000200, 0x00002021)]
  struct.pack_into("<BII", _zm_image, 0x20, 0x7a, 0x000204, 0x00002020)
  _zm_dev = _FakeZmReg()
  nvbios_init.NvbiosInit(_zm_dev, bytes(_zm_image)).run_script(0x20)
  assert _zm_dev.writes == [(0x000204, 0x00002020)]
  class _FakePostEntry:
    def __init__(self, pramin):
      self.regs = {
          0x000000: 0x0e4040a2,
          0x02240c: 0x00000002,
          0x409604: 0x00040004,
          0x001700: 0x0000fffe,
          0x700000: pramin,
      }
      self.reads = []
    def read32(self, reg):
      self.reads.append(reg)
      return self.regs[reg]
  _posted_entry = _FakePostEntry(0x0000beef)
  assert _gk104_post_entry_probe(_posted_entry)["night41h_ready"]
  assert _posted_entry.reads == [
      0x000000, 0x02240c, 0x409604, 0x001700, 0x700000]
  assert not _gk104_post_entry_probe(
      _FakePostEntry(0xbad0fb03))["night41h_ready"]
  class _FakeRomShadowEntry:
    def __init__(self, rom_window=0x00fffe09, pramin=0x0000beef):
      self.regs = {
          0x022500: 0x00000100,
          0x619f04: rom_window,
          0x088050: 0x00000001,
          0x001700: 0x0000ffb0,
          0x700000: pramin,
      }
      self.reads = []
    def read32(self, reg):
      self.reads.append(reg)
      return self.regs[reg]
  _rom_shadow = _FakeRomShadowEntry()
  _rom_snap = _gk104_rom_shadow_entry_probe(_rom_shadow)
  assert _rom_snap["ramin_source_eligible"]
  assert _rom_snap["firmware_shadow_ready"]
  assert _rom_snap["rom_window_base"] == 0xfffe0000
  assert _rom_shadow.reads == [
      0x022500, 0x619f04, 0x088050, 0x001700, 0x700000]
  assert not _gk104_rom_shadow_entry_probe(
      _FakeRomShadowEntry(rom_window=0x00fffe08))["ramin_source_eligible"]
  assert not _gk104_rom_shadow_entry_probe(
      _FakeRomShadowEntry(pramin=0xbad0fb03))["firmware_shadow_ready"]
  class _FakeGoldenPreinit:
    def __init__(self, overrides=None):
      self.regs = dict(GK104_GOLDEN_PREINIT_READS)
      self.regs.update(overrides or {})
      self.reads = []
    def read32(self, reg):
      self.reads.append(reg)
      return self.regs[reg]
  _golden_preinit = _FakeGoldenPreinit()
  _golden_snap = _gk104_golden_preinit_entry_probe(_golden_preinit)
  assert _golden_snap["exact_match"]
  assert _golden_preinit.reads == [reg for reg, _value in
                                   GK104_GOLDEN_PREINIT_READS]
  _cold_preinit = _gk104_golden_preinit_entry_probe(_FakeGoldenPreinit({
      0x101000: 0x00000000,
      0x619f04: 0x00000001,
      0x001700: 0x00000000,
      0x088050: 0x00000000,
  }))
  assert _cold_preinit["mismatch_regs"] == (
      0x101000, 0x619f04, 0x001700, 0x088050)
  class _FakeRomWindowAB:
    def __init__(self, activation):
      self.activation = activation
      self.regs = {
          0x022500: 0x00000100,
          0x619f04: 0x00000001,
          0x088050: 0x00000000,
          0x001700: 0x00000000,
      }
      self.physical = b"N41N-GOLDEN-VRAM"
      self.stub_count = 0
      self.writes = []
    def read32(self, reg):
      if 0x700000 <= reg < 0x800000:
        window = self.regs[0x619f04]
        selected = (self.regs[0x001700] << 16) + (reg - 0x700000)
        base = (window & 0xffffff00) << 8
        visible = (bool(window & 0x8) and (window & 0x3) == 1 and
                   base <= selected < base + len(self.physical))
        if self.activation == "pci-shadow":
          visible = visible and bool(self.regs[0x088050] & 1)
        elif self.activation == "none":
          visible = False
        if visible:
          off = selected - 0xfffe0000
          return struct.unpack_from("<I", self.physical, off)[0]
        word = 0xbad0fb00 | (self.stub_count & 0xff)
        self.stub_count += 1
        return word
      return self.regs.get(reg, 0)
    def write32(self, reg, value):
      self.regs[reg] = value & 0xffffffff
      self.writes.append((reg, value & 0xffffffff))
  for _activation, _classification in (
      ("rom-window", "rom-window-sufficient"),
      ("pci-shadow", "rom-window-plus-pci-shadow"),
      ("none", "still-stubbed")):
    _ab_dev = _FakeRomWindowAB(_activation)
    _ab_old = dict(_ab_dev.regs)
    _ab = _gk104_rom_window_pramin_ab_probe(
        _ab_dev, 0xfffe0000, _ab_dev.physical)
    assert _ab["classification"] == _classification, _ab
    assert _ab_dev.regs == _ab_old
  class _FakePraminStage:
    def __init__(self):
      self.selector = 0x0000fffe
      self.writes = []
    def read32(self, reg):
      if reg == 0x001700: return self.selector
      return 0x12340000 | ((reg - 0x700000) >> 2)
    def write32(self, reg, value):
      assert reg == 0x001700
      self.selector = value & 0xffffffff
      self.writes.append((reg, self.selector))
  _stage = _FakePraminStage()
  assert _gk104_pramin_stage_snapshot(
      _stage, "selftest", pa=0x12300000) == (
          0x12340000, 0x12340001, 0x12340002, 0x12340003)
  assert _stage.writes == [
      (0x001700, 0x00001230), (0x001700, 0x0000fffe)]
  assert _stage.selector == 0x0000fffe

  # GK104 PMU recovery must use the MC-level PMC_ENABLE bit, not CPUCTL inside
  # the PMU aperture.  The latter was the night15/16 reload bug: the transport
  # stayed at 0xffffffff because an inaccessible engine cannot self-reset.
  class _FakePmuResetRegs:
    def __init__(self):
      self.regs = {PMC_ENABLE: 0xe011312d,
                   PMU_FALCON_BASE + 0x10c: 0}
      self.writes = []
    def read32(self, reg):
      return self.regs.get(reg, 0)
    def write32(self, reg, value):
      value &= 0xffffffff
      self.regs[reg] = value
      self.writes.append((reg, value))
  _pmu_reset_regs = _FakePmuResetRegs()
  falcon_reset(_pmu_reset_regs, PMU_FALCON_BASE, settle_s=0)
  _pmc_writes = [v for r, v in _pmu_reset_regs.writes if r == PMC_ENABLE]
  assert _pmc_writes == [0xe011112d, 0xe011312d], _pmc_writes
  assert not any(r == PMU_FALCON_BASE + FALCON_UC_CTRL
                 for r, _ in _pmu_reset_regs.writes)
  assert _pmu_reset_regs.writes[-1] == (0x10a580, 0)
  _pmu_reset_full = _FakePmuResetRegs()
  _pmu_reset_full.regs[PMC_ENABLE] = 0xffffffff
  falcon_reset(_pmu_reset_full, PMU_FALCON_BASE, settle_s=0)
  assert [v for r, v in _pmu_reset_full.writes if r == PMC_ENABLE] == [
      0xffffdfff, 0xffffffff]

  # Exercise the macOS-only crossing end-to-end against an XOR-like BAR1
  # model.  Night25: host FECS XFER MMIO dies after bit0, so the fake arms an
  # autonomous Falcon bootstrap and commits the staged DMEM roots during the
  # bit0 write itself (standing in for the Falcon delay+IO XFER).
  class _AtomicBar1Hw:
    def __init__(self, owner):
      self.owner = owner
    def _xlate(self, va, size):
      o = self.owner
      ctl = o.regs.get(0x1704, 0)
      if not (ctl & 0x80000000):
        # GF100 PBUS BAR1 VRAM mode: VM disabled maps BAR1 directly to VRAM.
        pa = ((ctl & 0x0fffffff) << 12) + va
        assert 0 <= pa <= len(o.vram) - size, (pa, size)
        return pa
      inst_pa = (ctl & 0x3fffffff) << 12
      limit = struct.unpack_from("<Q", o.vram, inst_pa + 0x208)[0]
      assert va + size - 1 <= limit, (va, size, limit)
      # gf100_vmm_join_: low 3 bits are target/VOL (HOST=2|VOL).
      pgd_pa = struct.unpack_from("<Q", o.vram, inst_pa + 0x200)[0] & ~0x7
      assert 0 <= pgd_pa <= len(o.vram) - 8, pgd_pa
      pde = struct.unpack_from("<Q", o.vram, pgd_pa)[0]
      # GF100_PDE SPT_ADDRESS is bits [63:36], value is PA>>12 (envytools).
      # Matches Nouveau's `pt->addr << 24` with target/VOL below bit 36.
      spt_pa = ((pde >> 36) & 0xfffffff) << 12
      assert 0 <= spt_pa <= len(o.vram) - 8, (pde, spt_pa)
      pte = struct.unpack_from("<Q", o.vram, spt_pa + (va >> 12) * 8)[0]
      assert pte & 1, f"invalid fake BAR1 PTE va={va:#x} pte={pte:#x}"
      # GF100_PTE ADDRESS is bits [31:4] = PA>>12; ignore VOL/aper in [63:32].
      pa = (((pte >> 4) & 0x0fffffff) << 12) | (va & 0xfff)
      assert 0 <= pa <= len(o.vram) - size, (pte, pa, size)
      return pa
    def mmio_read(self, bar, va, size):
      assert bar == 1
      ctl = self.owner.regs.get(0x1704, 0)
      access = "paged BAR1 access" if ctl & 0x80000000 else \
          "physical BAR1 access"
      self.owner.bar1_semantic_ops.append((access, va, size))
      try:
        pa = self._xlate(va, size)
        return bytes(self.owner.vram[pa:pa + size])
      except (AssertionError, struct.error):
        bad = b"".join(struct.pack("<I", 0xbad0fb00 | (i & 0xff))
                       for i in range((size + 3) // 4))
        return bad[:size]
    def mmio_write(self, bar, va, data):
      assert bar == 1 and len(data) == 4
      pa = self._xlate(va, len(data))
      cur = struct.unpack_from("<I", self.owner.vram, pa)[0]
      val = struct.unpack("<I", data)[0]
      struct.pack_into("<I", self.owner.vram, pa, cur ^ val)

  class _AtomicBar1Dev:
    def __init__(self):
      self.regs = {
          0: 0x0e4040a2, 0x1620: 0xaab, 0x1704: 0,
          0x409100: 0, 0x409104: 0, 0x409128: 0x10, 0x409ff0: 0,
          0x409118: 0, 0x409120: 0x10, 0x10a580: 0,
          0x100c80: 0x00018000,
          PMU_FALCON_BASE + FALCON_UC_CTRL: 0,
          PMU_FALCON_BASE + FALCON_UC_ENTRY: 0,
          PMU_FALCON_BASE + 0x128: 0x10,
          PMU_FALCON_BASE + 0xff0: 0,
          PMU_FALCON_BASE + 0x600: 0x110,
          PMU_FALCON_BASE + 0x624: 0x110,
      }
      self.vram = bytearray(b"\xff") * 0x1000000
      self.dmem = bytearray(0x2000)
      self.imem = bytearray(0x2000)
      self._dmem_addr = 0
      self._dmem_write = False
      self._imem_addr = 0
      self._pmu_memx_data = (0x3cc, 0x800)
      self._pmu_code = bytes(_gk104_pmu_embed_bar1_bootstrap(bytes(0xc00)))
      self.pmu_imem = bytearray(0x2000)
      self.pmu_imem[:len(self._pmu_code)] = self._pmu_code
      self._pmu_imem_addr = 0
      self._pmu_dmem_addr = 0
      self._pmu_dmem_write = False
      self._pmu_auto_bar1_armed = False
      self._pmu_auto_bar1_applied = False
      self._memx_inside_enter = False
      self._fecs_stopped_for_staging = False
      self.bar_flushes = 0
      self.bar1_semantic_ops = []
      impl = type("_AtomicImpl", (), {})()
      impl.hw = _AtomicBar1Hw(self)
      impl.bar1_size = len(self.vram)
      self.dev_impl = impl
    def _apply_autonomous_roots(self):
      assert (self.regs[PMU_FALCON_BASE + 0x600] & 0x7) == 0x4
      assert (self.regs[PMU_FALCON_BASE + 0x624] & 0x90) == 0x90
      assert (self.regs[0x1620] & 1) == 0, "autonomous xdst before bit0"
      assert self.regs.get(PMU_FALCON_BASE + FALCON_UC_ENTRY, 0) == \
          PMU_BAR1_BOOTSTRAP_IMEM
      # Model local PMU DMEM construction and absolute direct-VRAM stores.
      for (dmem, _ext, size), blob in zip(
          PMU_BAR1_BOOTSTRAP_XFER, PMU_BAR1_BOOTSTRAP_ROOTS):
        assert len(blob) == size
        self.dmem[dmem:dmem + size] = blob
      for dmem, ext, size in PMU_BAR1_BOOTSTRAP_XFER:
        pa = ext
        assert pa + size <= len(self.vram)
        self.vram[pa:pa + size] = self.dmem[dmem:dmem + size]
        self.bar1_semantic_ops.append(("PMU xdst", ext, size))
      self._pmu_auto_bar1_applied = True
    def read32(self, reg):
      if reg == FECS_FALCON_BASE + FALCON_DATA:
        assert not self._dmem_write
        value = struct.unpack_from("<I", self.dmem, self._dmem_addr)[0]
        self._dmem_addr += 4
        return value
      if reg == FECS_FALCON_BASE + FALCON_CODE:
        return struct.unpack_from("<I", self.imem, self._imem_addr)[0]
      if reg == PMU_FALCON_BASE + FALCON_CODE:
        return struct.unpack_from("<I", self.pmu_imem, self._pmu_imem_addr)[0]
      if reg == PMU_FALCON_BASE + FALCON_DATA:
        assert not self._pmu_dmem_write
        value = struct.unpack_from("<I", self.dmem, self._pmu_dmem_addr)[0]
        self._pmu_dmem_addr += 4
        return value
      return self.regs.get(reg, 0)
    def write32(self, reg, value):
      value &= 0xffffffff
      if not (self.regs.get(0x1620, 1) & 1) and reg in (
          0x070000, 0x100cb8, 0x100cbc):
        raise AssertionError(
            f"post-bit0 host access to proven-dead BAR/LTC/VMM reg {reg:#x}")
      if reg == PMU_FALCON_BASE + FALCON_CODE_INDEX:
        self._pmu_imem_addr = value & 0x00ffffff
        return
      if reg == PMU_FALCON_BASE + FALCON_CODE:
        struct.pack_into("<I", self.pmu_imem, self._pmu_imem_addr, value)
        return
      if reg == PMU_FALCON_BASE + FALCON_CODE_TAG:
        self.regs[reg] = value
        return
      if reg == PMU_FALCON_BASE + FALCON_DATA_INDEX:
        self._pmu_dmem_addr = value & 0x00ffffff
        self._pmu_dmem_write = bool(value & FALCON_IDX_WRITE)
        return
      if reg == PMU_FALCON_BASE + FALCON_DATA:
        assert self._pmu_dmem_write
        struct.pack_into("<I", self.dmem, self._pmu_dmem_addr, value)
        # Host GO write releases the pad into its delay (model DELAY_ENTERED).
        if (self._pmu_dmem_addr == PMU_BAR1_BOOTSTRAP_GO_DMEM and
            value == PMU_BAR1_BOOTSTRAP_GO):
          struct.pack_into("<I", self.dmem,
                           PMU_BAR1_BOOTSTRAP_DELAY_ENTERED_DMEM,
                           PMU_BAR1_BOOTSTRAP_DELAY_ENTERED)
          self.regs[PMU_FALCON_BASE + 0xff0] = PMU_BAR1_BOOTSTRAP_DELAY_START
          self.regs[0x1620] = 0x8
          self.regs[0x26f0] = 0
          self.bar1_semantic_ops.append(("Falcon ENTER", 0x8, 0))
          self.bar1_semantic_ops.append(("FB_PAUSE SET", 4, 0))
          # Night40ae: rising-edge — clear then set (Nouveau LEAVE then ENTER).
          struct.pack_into("<I", self.dmem,
                           PMU_BAR1_BOOTSTRAP_PAUSE_CLEAR_DMEM, 0)
          self._apply_autonomous_roots()
          self.bar1_semantic_ops.append(("FB_PAUSE CLR", 4, 0))
          self.regs[0x26f0] = 1
          self.regs[0x1620] = 0xaab
          self.bar1_semantic_ops.append(("Falcon LEAVE", 0xaab, 0))
          struct.pack_into("<I", self.dmem, PMU_BAR1_BOOTSTRAP_DONE_DMEM,
                           PMU_BAR1_BOOTSTRAP_DONE)
        self._pmu_dmem_addr += 4
        return
      if reg == PMU_FALCON_BASE + FALCON_UC_CTRL:
        self.regs[reg] = value
        if value & FALCON_UC_CTRL_START:
          self.regs[PMU_FALCON_BASE + 0x128] = 0
          entry = self.regs.get(PMU_FALCON_BASE + FALCON_UC_ENTRY, 0)
          if entry == PMU_BAR1_BOOTSTRAP_IMEM:
            self._pmu_auto_bar1_armed = True
            self.regs[PMU_FALCON_BASE + 0xff0] = \
                PMU_BAR1_BOOTSTRAP_DELAY_START
            # Roots are host-staged in dma_prepare; pad only publishes magic.
            struct.pack_into("<I", self.dmem, PMU_BAR1_BOOTSTRAP_MAGIC_DMEM,
                             PMU_BAR1_BOOTSTRAP_MAGIC)
            struct.pack_into("<I", self.dmem, PMU_BAR1_BOOTSTRAP_GO_DMEM, 0)
            struct.pack_into("<I", self.dmem,
                             PMU_BAR1_BOOTSTRAP_DELAY_ENTERED_DMEM, 0)
        if value & 0x10:
          self.regs[PMU_FALCON_BASE + 0x128] = 0x10
        return
      if reg == FECS_FALCON_BASE + FALCON_DATA_INDEX:
        self._dmem_addr = value & 0x00ffffff
        self._dmem_write = bool(value & FALCON_IDX_WRITE)
        return
      if reg == FECS_FALCON_BASE + FALCON_DATA:
        assert self._dmem_write
        if 0x200 <= self._dmem_addr < 0x300:
          assert self._fecs_stopped_for_staging, \
              "FECS xfer_data must be staged only after halt"
        struct.pack_into("<I", self.dmem, self._dmem_addr, value)
        self._dmem_addr += 4
        return
      if reg == FECS_FALCON_BASE + FALCON_CODE_INDEX:
        self._imem_addr = value & 0x00ffffff
        return
      if reg == FECS_FALCON_BASE + FALCON_CODE:
        struct.pack_into("<I", self.imem, self._imem_addr, value)
        return
      if reg == FECS_FALCON_BASE + FALCON_CODE_TAG:
        self.regs[reg] = value
        return
      if reg == FECS_FALCON_BASE + FALCON_UC_CTRL:
        self.regs[reg] = value
        if value & FALCON_UC_CTRL_START:
          self._fecs_stopped_for_staging = False
          self.regs[FECS_FALCON_BASE + 0x128] = 0  # running
          entry = self.regs.get(FECS_FALCON_BASE + FALCON_UC_ENTRY, 0)
          if entry == FECS_BAR1_BOOTSTRAP_IMEM:
            self._fecs_auto_bar1_armed = True
            # Sit in the long delay loop so the pre-bit0 PC check passes.
            self.regs[FECS_FALCON_BASE + 0xff0] = FECS_BAR1_BOOTSTRAP_IMEM + 0x9
        if value & 0x10:  # HALT
          self._fecs_stopped_for_staging = True
          self.regs[FECS_FALCON_BASE + 0x128] = 0x10  # STOPPED
        return
      if reg == FECS_FALCON_BASE + FALCON_UC_ENTRY:
        self.regs[reg] = value
        return
      if reg == FECS_FALCON_BASE + 0x118:
        raise AssertionError(
            "host must not submit FECS XFER after night25; use autonomous xdst")
      if reg == PMU_FALCON_BASE + 0x118:
        # Night40k: stores require ENTER.  Night40ab: post-LEAVE loadback is
        # allowed once the autonomous pad has published roots into VRAM.
        mode = (value >> 4) & 3
        if mode == 2:
          assert self._memx_inside_enter, \
              "direct-VRAM MEMIF roots must be transferred inside ENTER/LEAVE"
        elif mode == 0:
          assert self._memx_inside_enter or self._pmu_auto_bar1_applied, \
              "MEMIF loadback only inside ENTER or after autonomous DONE"
        code = (value >> 8) & 7
        size = 4 << code
        ext = ((self.regs.get(PMU_FALCON_BASE + 0x110, 0) & 0xffffffff) << 8) + (
            self.regs.get(PMU_FALCON_BASE + 0x11c, 0) & 0xffffffff)
        local = self.regs.get(PMU_FALCON_BASE + 0x114, 0) & 0xffffffff
        if mode == 2:
          self.vram[ext:ext + size] = bytes(self.dmem[local:local + size])
          self.bar1_semantic_ops.append(("MEMIF root", ext, size))
        elif mode == 0:
          self.dmem[local:local + size] = bytes(self.vram[ext:ext + size])
          self.bar1_semantic_ops.append(("MEMIF loadback", ext, size))
        self.regs[reg] = value & ~0x3
        self.regs[PMU_FALCON_BASE + 0x120] = 0x10
        return
      if reg in (0x409a04, 0x409a20):
        assert self._fecs_stopped_for_staging, \
            "FECS transfer target must be configured only after halt"
        self.regs[reg] = value
        return
      if reg == 0x1620:
        prev = self.regs.get(reg, 0)
        self.regs[reg] = value
        if (prev & 1) and not (value & 1) and self._pmu_auto_bar1_armed:
          go = struct.unpack_from("<I", self.dmem, PMU_BAR1_BOOTSTRAP_GO_DMEM)[0]
          assert go == PMU_BAR1_BOOTSTRAP_GO, "PMU IO XFER requires host GO before bit0"
          self.bar1_semantic_ops.append(("bit0 unstub", value, 0))
          self._apply_autonomous_roots()
        return
      if reg == 0x070000:
        assert value == 1
        self.bar_flushes += 1
        self.bar1_semantic_ops.append(("g84_bar_flush", value, 0))
        self.regs[reg] = 0
        return
      if reg == 0x001704 and value & 0x80000000:
        self.bar1_semantic_ops.append(("gf100_bar_bar1_init", value, 0))
      self.regs[reg] = value
    def pmu_memx_exec_commands(self, commands, timeout_s=5.0):
      if commands[0][0] == PMU_MEMX_DELAY:
        return (0, 0)
      window = 0
      for opcode, payload in commands:
        if opcode == PMU_MEMX_ENTER:
          assert not self._memx_inside_enter
          self._memx_inside_enter = True
          self.regs[0x1620] &= ~0xaa3
          self.regs[0x26f0] = self.regs.get(0x26f0, 1) & ~1
          self.bar1_semantic_ops.append(("MEMX ENTER", 0, 0))
          continue
        if opcode == PMU_MEMX_LEAVE:
          assert self._memx_inside_enter
          self.regs[0x26f0] = self.regs.get(0x26f0, 0) | 1
          self.regs[0x1620] |= 0xaa3
          self._memx_inside_enter = False
          self.bar1_semantic_ops.append(("MEMX LEAVE", 0, 0))
          continue
        if opcode == PMU_MEMX_WAIT:
          reg, mask, value, _nsec = payload
          assert (self.regs.get(reg, 0) & mask) == value
          continue
        assert opcode == PMU_MEMX_WR32 and len(payload) % 2 == 0
        for i in range(0, len(payload), 2):
          reg, value = payload[i:i + 2]
          if reg == 0x1700:
            window = value << 16
          elif reg == 0x1620:
            self.write32(reg, value)
          elif 0x700000 <= reg < 0x800000:
            assert self._memx_inside_enter, \
                "framebuffer roots must be stored inside MEMX ENTER/LEAVE"
            pa = window + reg - 0x700000
            struct.pack_into("<I", self.vram, pa, value)
            self.bar1_semantic_ops.append(("MEMX root", pa, 4))
          else:
            self.regs[reg] = value
      return (0, 0)

  _atomic_env = {k: os.environ.get(k) for k in (
      "KEPLER_TINYGPU_ATOMIC_BAR1", "KEPLER_PRAMIN_MEMX",
      "KEPLER_RAM_BIT0_DEFER", "KEPLER_RAM_BLOCK",
      "KEPLER_BAR1_MAP_SIZE", "KEPLER_BAR1_DIRECT_PHYS",
      "KEPLER_BAR1_HOST_PDB", "KEPLER_ROM_WINDOW_AB")}
  try:
    os.environ["KEPLER_TINYGPU_ATOMIC_BAR1"] = "1"
    os.environ["KEPLER_PRAMIN_MEMX"] = "1"
    os.environ["KEPLER_RAM_BIT0_DEFER"] = "1"
    os.environ["KEPLER_RAM_BLOCK"] = "atomic"
    os.environ["KEPLER_BAR1_MAP_SIZE"] = "0x1000000"
    os.environ["KEPLER_BAR1_DIRECT_PHYS"] = "1"
    os.environ["KEPLER_BAR1_HOST_PDB"] = "0"
    os.environ["KEPLER_ROM_WINDOW_AB"] = "0"
    _atomic_dev = _AtomicBar1Dev()
    _gk104_init_bar1_identity(
        _atomic_dev, map_vram=True, userd_alias_pa=0x400000)
    assert _atomic_dev._bar1_identity_ready
    assert _atomic_dev.regs[0x1620] == 0xaab
    assert _atomic_dev._pmu_auto_bar1_applied
    assert _atomic_dev.bar_flushes >= 2
    _enter = _atomic_dev.bar1_semantic_ops.index(("Falcon ENTER", 0x8, 0))
    _pause = _atomic_dev.bar1_semantic_ops.index(("FB_PAUSE SET", 4, 0))
    _root = _atomic_dev.bar1_semantic_ops.index(("PMU xdst", 0x60200, 16))
    _unpause = _atomic_dev.bar1_semantic_ops.index(("FB_PAUSE CLR", 4, 0))
    _leave = _atomic_dev.bar1_semantic_ops.index(("Falcon LEAVE", 0xaab, 0))
    _loadback = _atomic_dev.bar1_semantic_ops.index(
        ("MEMIF loadback", 0x60200, 16))
    _vm_switch = _atomic_dev.bar1_semantic_ops.index(
        ("gf100_bar_bar1_init", 0x80000060, 0))
    _physical_first = _atomic_dev.bar1_semantic_ops.index(
        ("physical BAR1 access", 0x60200, 16))
    _paged_first = _atomic_dev.bar1_semantic_ops.index(
        ("paged BAR1 access", 0, 16))
    assert (_enter < _pause < _root < _unpause < _leave < _loadback <
            _physical_first < _vm_switch < _paged_first), \
        _atomic_dev.bar1_semantic_ops[:32]
    assert struct.unpack_from("<Q", _atomic_dev.vram, 0x60208)[0] == 0xffffff
    assert struct.unpack_from("<Q", _atomic_dev.vram, 0x50000)[0] == 0x4001
    assert struct.unpack_from("<Q", _atomic_dev.vram, 0x50008)[0] == 0x4011
    # PMU firmware image must carry the autonomous pad routine.
    assert _atomic_dev._pmu_code[
        PMU_BAR1_BOOTSTRAP_IMEM:PMU_BAR1_BOOTSTRAP_IMEM + 4] == \
        PMU_BAR1_BOOTSTRAP_CODE[:4]
    # Night41c: H21 is closed; production pad must carry all three direct-VRAM
    # xdsts and must not regress to the literal-only PRAMIN diagnostic.
    assert PMU_BAR1_BOOTSTRAP_CODE.count(b"\xfa\x21\x06") == 3
    assert bytes.fromhex("f1e70017f1d70000") not in PMU_BAR1_BOOTSTRAP_CODE
    assert b"\xa5\xa5" not in PMU_BAR1_BOOTSTRAP_CODE
    assert PMU_BAR1_BOOTSTRAP_END <= 0xc00

    # Night40ac: HOST-PDB path (Nouveau join/PDE HOST; experimental 0x1704).
    os.environ["KEPLER_BAR1_HOST_PDB"] = "1"
    _host_dev = _AtomicBar1Dev()
    _gk104_init_bar1_identity(
        _host_dev, bus_base=0, map_vram=False, userd_alias_pa=0x400000)
    assert _host_dev._bar1_identity_ready
    assert getattr(_host_dev, "_bar1_host_pdb", False)
    assert _host_dev.regs[0x1704] == 0x80000040
    _host_join = struct.unpack_from("<Q", _host_dev.vram, 0x40200)[0]
    assert _host_join == _gk104_host_join(0x41000), hex(_host_join)
    _host_pde = struct.unpack_from("<Q", _host_dev.vram, 0x41000)[0]
    assert _host_pde == _gk104_host_pde_spt(0x100000), hex(_host_pde)
    assert struct.unpack_from("<Q", _host_dev.vram, 0x100000)[0] == \
        _gk104_host_pte(0x400000)
    assert ("gf100_bar_bar1_init", 0x80000040, 0) in _host_dev.bar1_semantic_ops
  finally:
    for _key, _value in _atomic_env.items():
      if _value is None:
        os.environ.pop(_key, None)
      else:
        os.environ[_key] = _value

  # BAR0 MSE recovery: PMC_BOOT_0=0xffffffff must not imply physical replug
  # when config space is alive and COMMAND.MSE was cleared (DEXT Close()).
  class _FakeBarHw:
    def __init__(self, command=0x0000, boot0=0x0e4040a2, id32=0x118410de):
      self.command = command & 0xffff
      self.boot0 = boot0 & 0xffffffff
      self.id32 = id32 & 0xffffffff
      self.mapped = 0
      self.resets = 0
    def read_config(self, offset, size):
      if offset == 0 and size == 4: return self.id32
      if offset == pci.PCI_VENDOR_ID and size == 4: return self.id32
      if offset == pci.PCI_COMMAND and size == 2: return self.command
      raise AssertionError(f"unexpected config read {offset:#x}/{size}")
    def write_config(self, offset, value, size):
      if offset == pci.PCI_COMMAND and size == 2:
        self.command = value & 0xffff
        return
      raise AssertionError(f"unexpected config write {offset:#x}")
    def write_config_flush(self, offset, value, size):
      self.write_config(offset, value, size)
      return self.read_config(offset, size)
    def bar_info(self, bar):
      assert bar == 0
      self.mapped += 1
      return (0xf0000000, 16 << 20)
    def mmio_read32(self, bar, offset):
      assert bar == 0 and offset == 0
      # BAR MMIO only decodes when MSE is set (Apple Close() clears it).
      if not (self.command & pci.PCI_COMMAND_MEMORY):
        return 0xffffffff
      return self.boot0
    def reset(self):
      self.resets += 1
      self.command = 0  # Close()-like clear after reset
      self.boot0 = 0x0e4040a2
  # MSE was off → restore → live boot0; MASTER stays clear during probe.
  _hw = _FakeBarHw(command=0x0000)
  _boot, _meta = _gk104_ensure_bar0_mmio(_hw)
  assert _boot == 0x0e4040a2 and _meta["mse_before"] is False
  assert (_hw.command & pci.PCI_COMMAND_MEMORY) and not (_hw.command & pci.PCI_COMMAND_MASTER)
  assert _hw.mapped >= 1 and _hw.resets == 0
  # Config lost → physical cycle message (not "BAR0 looks dead" alone).
  try:
    _gk104_ensure_bar0_mmio(_FakeBarHw(id32=0xffffffff))
    raise AssertionError("expected config-lost error")
  except RuntimeError as _e:
    assert "config space lost" in str(_e).lower() or "endpoint/link" in str(_e).lower()
  # MSE already on but BAR dead → reset path recovers.
  class _FakeBarHwReset(_FakeBarHw):
    def mmio_read32(self, bar, offset):
      assert bar == 0 and offset == 0
      if not (self.command & pci.PCI_COMMAND_MEMORY):
        return 0xffffffff
      return 0xffffffff if self.resets == 0 else self.boot0
  _hw3 = _FakeBarHwReset(command=pci.PCI_COMMAND_MEMORY, boot0=0x0e4040a2)
  _boot3, _meta3 = _gk104_ensure_bar0_mmio(_hw3)
  assert _boot3 == 0x0e4040a2 and _meta3["did_reset"] and _hw3.resets == 1
  _hw4 = _FakeBarHwReset(command=pci.PCI_COMMAND_MEMORY, boot0=0x0e4040a2)
  try:
    _gk104_ensure_bar0_mmio(_hw4, allow_reset=False)
    raise AssertionError("cold-boundary probe unexpectedly recovered BAR0")
  except RuntimeError as _e:
    assert "inaccessible" in str(_e).lower()
  assert _hw4.resets == 0

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

  # Golden-trace FIFO lifecycle sanity: a commit must explicitly allow the
  # runlist, encode target/address/count, acknowledge completion, and a context
  # pointer update must first stop and KICK/preempt the channel.
  class _FakeFifoRegs:
    def __init__(self):
      self.regs = {0x2630: 1 << GR_RUNLIST_ID,
                   0x2284 + GR_RUNLIST_ID * 8: 0,
                   0x2a00: 1 << GR_RUNLIST_ID,
                   CHAN_START_REG + 2 * 8: 0}
      self.writes = []
    def read32(self, reg): return self.regs.get(reg, 0)
    def write32(self, reg, value):
      value &= 0xffffffff
      self.regs[reg] = value
      self.writes.append((reg, value))
  _fifo = _FakeFifoRegs()
  _gk104_commit_runlist(_fifo, 0x12345000, 1, target=3)
  assert not (_fifo.regs[0x2630] & (1 << GR_RUNLIST_ID)), \
      "runlist remained blocked at commit"
  assert _fifo.regs[0x2270] == 0x30012345
  assert _fifo.regs[PFIFO_RUNLIST_SUBMIT] == 1
  _gk104_stop_preempt_channel(_fifo, 2)
  assert _fifo.regs[CHAN_START_REG + 2 * 8] & 0x800
  assert _fifo.regs[0x2634] == 2

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
    # alloc_sysmem resolves Linux PFNs through /proc/self/pagemap.  Keep the
    # rest of this fake-sysfs transport test portable on macOS, where /proc is
    # absent and the TinyGPU wrapper is the real transport.
    if os.path.exists("/proc/self/pagemap"):
      _mv, _paddrs = hw.alloc_sysmem(0x3000, contiguous=True)
      assert len(_mv) == 0x3000 and len(_paddrs) == 3, \
          f"alloc_sysmem {len(_paddrs)} pages"
      _mv[0:4] = struct.pack("<I", 0xcafef00d)
      assert hw.sysmem_read(_paddrs[0], 4) == b"\x0d\xf0\xfe\xca", \
          "sysmem_read round-trip"
      hw.sysmem_write(_paddrs[1], b"\xaa\xbb\xcc\xdd")
      assert _mv[0x1000:0x1004] == b"\xaa\xbb\xcc\xdd", \
          "sysmem_write round-trip"
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

  if _hosts is not None:
    a_host, b_host = _hosts
    if len(a_host) != N or len(b_host) != N:
      raise ValueError(f"_hosts length {len(a_host)}/{len(b_host)} != KEPLER_N={N}")
  else:
    a_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
    b_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
  if os.environ.get("KEPLER_PRINT_IO", "0") != "0":
    _show = min(N, 16)
    print(f"[kepler] inputs a[0:{_show}]={[round(a_host[i], 4) for i in range(_show)]}",
          flush=True)
    print(f"[kepler] inputs b[0:{_show}]={[round(b_host[i], 4) for i in range(_show)]}",
          flush=True)

  a_dev = allocator.alloc(N * 4)
  b_dev = allocator.alloc(N * 4)
  out_dev = allocator.alloc(N * 4)
  # Code + constant (param) buffers for the launch descriptor.
  code_dev = allocator.alloc(len(cubin))
  allocator._copyin(code_dev, cubin)
  cbuf = build_cuda_param_cbuf(
      a_dev.va_addr, b_dev.va_addr, out_dev.va_addr,
      grid=(1, 1, 1), block=(N, 1, 1))
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

  operation = os.environ.get("KEPLER_OPERATION", "add")
  expected = ([a_host[i] * b_host[i] for i in range(N)] if operation == "mul"
              else [a_host[i] + b_host[i] for i in range(N)])
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

def _gk104_wait_runlist_idle(dev, runl_id=GR_RUNLIST_ID, timeout_s=0.2,
                             label="runlist update"):
  """Wait for GK104's asynchronous runlist DMA to finish."""
  pending_reg = 0x2284 + runl_id * 8
  deadline = time.time() + timeout_s
  while dev.read32(pending_reg) & 0x00100000:
    if time.time() >= deadline:
      raise TimeoutError(
          f"GK104 {label} did not complete: {pending_reg:#x}="
          f"{dev.read32(pending_reg):#x} 2270={dev.read32(0x2270):#x} "
          f"2274={dev.read32(0x2274):#x} 262c={dev.read32(0x262c):#x} "
          f"2630={dev.read32(0x2630):#x} "
          f"runq={[hex(dev.read32(0x2390 + i * 4)) for i in range(3)]} "
          f"intr={dev.read32(0x2100):#x}/{dev.read32(0x2a00):#x}")
    time.sleep(0.001)

def _gk104_commit_runlist(dev, runlist_addr, count, target=0,
                          runl_id=GR_RUNLIST_ID, timeout_s=0.2):
  """Commit a GK104 runlist using the ordering captured from nouveau."""
  _gk104_wait_runlist_idle(dev, runl_id, timeout_s, "previous runlist update")
  dev.write32(0x262c, 1 << runl_id)
  # gk104_runl_allow(): a blocked runlist deliberately leaves the update
  # pending.  The old PCIe path set this bit immediately before committing.
  nvkm_mask(dev, 0x2630, 1 << runl_id, 0)
  dev.write32(0x2270, (target << 28) | (runlist_addr >> 12))
  dev.write32(PFIFO_RUNLIST_SUBMIT, (runl_id << 20) | count)
  _gk104_wait_runlist_idle(dev, runl_id, timeout_s, "runlist update")
  # gk104_fifo_intr_runlist(): acknowledge the completed list, exactly as the
  # golden trace does before scheduling continues.
  intr = dev.read32(0x2a00)
  if intr & (1 << runl_id):
    dev.write32(0x2a00, 1 << runl_id)

def _gk104_stop_preempt_channel(dev, chan_id, timeout_s=0.2,
                                label="channel preempt"):
  """Stop a GK104 channel and wait for KICK_CHID preemption to complete."""
  nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x00000800, 0x00000800)
  dev.write32(0x2634, chan_id)
  deadline = time.time() + timeout_s
  while dev.read32(0x2634) & 0x00100000:
    if time.time() >= deadline:
      raise TimeoutError(
          f"GK104 {label} did not complete: KICK_CHID={dev.read32(0x2634):#x}")
    time.sleep(0.001)

def _gk104_clear_sched_error(dev):
  """Acknowledge a latched GK104 scheduler error before a new runlist DMA."""
  intr = dev.read32(0x2100)
  if not (intr & 0x00000100):
    return 0
  code = dev.read32(0x254c)
  nvkm_mask(dev, 0x2140, 0x00000100, 0x00000000)
  dev.write32(0x2100, 0x00000100)
  dev.write32(0x254c, 0x00000001)
  nvkm_mask(dev, 0x2140, 0x00000100, 0x00000100)
  return code

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
    # A GO_IDLE bundle has stronger completion semantics than an ordinary
    # ICMD.  Nouveau waits for the whole GR engine here before polling the
    # command-interface busy bit.
    if (addr & 0xffff) == 0xe100 and not _gk104_gr_wait_idle(dev, 2.0):
      raise TimeoutError(
          f"GK104 ICMD GO_IDLE timed out at {addr:#x}: "
          f"status={dev.read32(0x400700):#x} busy={dev.read32(0x40060c):#x} "
          f"intr={dev.read32(0x400100):#x} fecs_exc={dev.read32(0x409c18):#x} "
          f"pfifo={dev.read32(0x2100):#x} faults={dev.read32(0x259c):#x} "
          f"fault0={dev.read32(0x2800):#x}/"
          f"{dev.read32(0x2804):#x}/{dev.read32(0x2808):#x}/"
          f"{dev.read32(0x280c):#x}")
    # Match nvkm_msec(..., 2000) around 0x400700 bit 2.  The previous
    # 200-ms, fail-open loop hid the first failing bundle and allowed method
    # initialization to run while ICMD was still active.
    deadline = time.time() + 2.0
    while dev.read32(0x400700) & 0x4:
      if time.time() >= deadline:
        raise TimeoutError(
            f"GK104 ICMD timed out at {addr:#x}: "
            f"status={dev.read32(0x400700):#x} busy={dev.read32(0x40060c):#x} "
            f"intr={dev.read32(0x400100):#x} fecs_exc={dev.read32(0x409c18):#x} "
            f"pfifo={dev.read32(0x2100):#x} faults={dev.read32(0x259c):#x}")
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

def _gk104_read_tpc_nr(dev, gpc_nr=None):
  """Silicon TPC_NR per GPC, optionally overridden by KEPLER_FORCE_TPC_NR.

  H2 discriminator: e.g. ``KEPLER_FORCE_TPC_NR=2`` or ``2,0,0,0`` forces a
  reduced floorsweep.  If SET_OBJECT then idles (GPC_STATUS=0), the 7-SM
  asymmetric map is causal; if it still hangs, look elsewhere (H1/H15).
  """
  if gpc_nr is None:
    gpc_nr = max(1, dev.read32(0x409604) & 0x1f)
  tpc_nr = [dev.read32(0x500000 + g * 0x8000 + 0x2608) & 0x1f
            for g in range(gpc_nr)]
  _ov = os.environ.get("KEPLER_FORCE_TPC_NR", "").strip()
  if not _ov:
    return tpc_nr
  forced = [int(x, 0) for x in _ov.split(",") if x.strip() != ""]
  if not forced or any(n < 0 or n > 8 for n in forced):
    raise RuntimeError(f"invalid KEPLER_FORCE_TPC_NR={_ov!r}")
  if len(forced) < gpc_nr:
    forced = forced + [0] * (gpc_nr - len(forced))
  elif len(forced) > gpc_nr:
    forced = forced[:gpc_nr]
  print(f"[kepler] FORCE_TPC_NR: silicon={tpc_nr} → forced={forced}", flush=True)
  return forced


def _gk104_grctx_floorsweep(dev, tpc_nr, ppc_tpc_mask):
  """Port GK104's topology-dependent gf100_grctx_generate_floorsweep()."""
  gpc_nr, tpc_total = len(tpc_nr), sum(tpc_nr)
  row, tile = _gk104_grctx_tiles(tpc_nr)

  # gf100_gr_oneinit_sm_id orders SMs by TPC index, then GPC index.
  sm = 0
  for gpc in range(gpc_nr):
    # Explicitly clear disabled GPCs (H2 FORCE_TPC_NR zeros).
    if tpc_nr[gpc] == 0:
      dev.write32(0x500c08 + gpc * 0x8000, 0)
      dev.write32(0x500c8c + gpc * 0x8000, 0)
  for tpc in range(max(tpc_nr) if tpc_nr else 0):
    for gpc in range(gpc_nr):
      if tpc >= tpc_nr[gpc]: continue
      dev.write32(0x504698 + gpc * 0x8000 + tpc * 0x800, sm)
      dev.write32(0x5044e8 + gpc * 0x8000 + tpc * 0x800, sm)
      dev.write32(0x500c10 + gpc * 0x8000 + tpc * 4, sm)
      dev.write32(0x504088 + gpc * 0x8000 + tpc * 0x800, sm)
      dev.write32(0x500c08 + gpc * 0x8000, tpc_nr[gpc])
      dev.write32(0x500c8c + gpc * 0x8000, tpc_nr[gpc])
      sm += 1

  # DS and PD NUM_TPC_PER_GPC tables.
  _num_tpc_words = []
  for block in range(4):
    data = 0
    for j in range(8):
      gpc = block * 8 + j
      if gpc < gpc_nr: data |= tpc_nr[gpc] << (j * 4)
    _num_tpc_words.append(data)
  for block, data in enumerate(_num_tpc_words):
    dev.write32(0x405870 + block * 4, data)
  for block, data in enumerate(_num_tpc_words):
    dev.write32(0x406028 + block * 4, data)

  # gf117 ROP mapping with GK104 tile distribution.
  packed = [0] * 6
  for i in range(32): packed[i // 6] |= (tile[i] & 7) << ((i % 6) * 5)
  shift, ntpcv = 0, tpc_total
  while not (ntpcv & 16): ntpcv, shift = ntpcv << 1, shift + 1
  magic0 = (ntpcv << 16) | (shift << 21) | (((1 << 5) % ntpcv) << 24)
  magic1 = sum(((1 << (i + 5)) % ntpcv) << ((i - 1) * 5) for i in range(1, 7))
  dev.write32(0x418bb8, (tpc_total << 8) | row)
  for i, data in enumerate(packed): dev.write32(0x418b08 + i * 4, data)
  dev.write32(0x41bfd0, (tpc_total << 8) | row | magic0)
  dev.write32(0x41bfe4, magic1)
  for i, data in enumerate(packed): dev.write32(0x41bf00 + i * 4, data)
  dev.write32(0x4078bc, (tpc_total << 8) | row)
  for i, data in enumerate(packed): dev.write32(0x40780c + i * 4, data)

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
  # Prefer silicon PPC mask when non-zero; if TPC_NR was forced to 0 for a
  # GPC (H2 discriminator), keep that GPC's PPC mask cleared.
  ppc_tpc_mask = [
      (0 if tpc_nr[gpc] == 0 else
       (m or ((1 << tpc_nr[gpc]) - 1)))
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
  # 6. Attrib configuration (gf117_grctx_generate_attrib).
  # Prefer channel-build attrib set (may be bit19-shrunk) so golden save and
  # the per-channel mmio list write identical PPC+0xc0/0xe4 values (H16).
  _attr = getattr(dev, "_kepler_attrib", None) or {}
  alpha = int(_attr.get("alpha_nr", C["alpha_nr"]))
  beta = int(_attr.get("attrib_nr", C["attrib_nr"]))
  _anmax = int(_attr.get("attrib_nr_max", C["attrib_nr_max"]))
  _almax = int(_attr.get("alpha_nr_max", C["alpha_nr_max"]))
  if _attr:
    print(f"[kepler] grctx attrib consts: alpha={alpha:#x} beta={beta:#x} "
          f"anmax={_anmax:#x} almax={_almax:#x} (channel-build)", flush=True)
  dev.write32(0x405830, (beta << 16) | alpha)
  dev.write32(0x4064c4, ((alpha // 4) << 16) | 0xffff)
  bo, ao = 0, _anmax * tpc_total
  for gpc, mask in enumerate(ppc_tpc_mask):
    count = mask.bit_count()
    ppc = 0x503000 + gpc * 0x8000
    dev.write32(ppc + 0xc0, (1 << 28) | (beta * count << 16) | bo)
    bo += _anmax * count
    dev.write32(ppc + 0xe4, (alpha * count << 16) | ao)
    ao += _almax * count
  _ppc_e4 = [dev.read32(0x503000 + g * 0x8000 + 0xe4) for g in range(gpc_nr)]
  print(f"[kepler] grctx PPC+0xe4: {[hex(x) for x in _ppc_e4]}", flush=True)
  # Preserve the LTC context registers in the generated image
  # (Nouveau gk104_grctx_generate_patch_ltc).  After SET_OBJECT hang,
  # FECS_MMIO_CTRL often shows a WRITE to 0x17e920; that is the last
  # eng-ctx MMIO, not proof these self-writes are optional.
  print(f"[kepler] LTC ctx preserve: 0x17e91c={dev.read32(0x17e91c):#x} "
        f"0x17e920={dev.read32(0x17e920):#x}", flush=True)
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
  # 9. floorsweep already performed gk104_grctx_generate_r419f78(); wait for
  # those topology writes to settle before entering ICMD mode.
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
  _c08 = [dev.read32(0x500c08 + g * 0x8000) for g in range(gpc_nr)]
  _c8c = [dev.read32(0x500c8c + g * 0x8000) for g in range(gpc_nr)]
  _c30 = [dev.read32(0x500c30 + g * 0x8000) for g in range(gpc_nr)]
  print(f"[kepler] grctx GPC TPC_NR: c08={[hex(x) for x in _c08]} "
        f"c8c={[hex(x) for x in _c8c]} c30={[hex(x) for x in _c30]}",
        flush=True)

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

def _gk104_ensure_pmu_memx_ready(dev) -> None:
  """Re-discover or reload PMU MEMX after FECS (semaphore can be wedged)."""
  _gk104_require_bar0_live(dev, "before PMU MEMX ensure")
  def _smoke() -> bool:
    try:
      if getattr(dev, "_pmu_memx_data", None) is None:
        return False
      sem = dev.read32(0x10a580) & 0xffffffff
      if sem == 0xffffffff:
        return False
      if sem != 0:
        dev.write32(0x10a580, 0)
      dev.pmu_memx_exec_commands([(PMU_MEMX_DELAY, (1000,))], timeout_s=2.0)
      return True
    except Exception:
      return False
  if _smoke():
    return
  print("[kepler] PMU MEMX not ready for BAR1 bootstrap/PRAMIN; "
        "reloading PMU falcon",
        flush=True)
  fw = find_kepler_firmware()
  if not fw:
    raise RuntimeError("PMU MEMX dead and no GK104 firmware tree for reload")
  pmu_code = open(os.path.join(fw, "gk104_pmu_code.bin"), "rb").read()
  pmu_data = open(os.path.join(fw, "gk104_pmu_data.bin"), "rb").read()
  if os.environ.get("KEPLER_PMU_ENTER_NOWAIT", "1") != "0":
    pmu_code = _patch_pmu_memx_nowait(pmu_code)
    try:
      setattr(dev, "_pmu_memx_nowait", True)
    except Exception:
      pass
  pmu_code = _gk104_pmu_embed_bar1_bootstrap(pmu_code)
  try:
    setattr(dev, "_pmu_code", bytes(pmu_code))
  except Exception:
    pass
  print(f"[kepler] PMU MC reset pre: PMC_ENABLE={dev.read32(PMC_ENABLE) & 0xffffffff:#x} "
        f"SCRUB={dev.read32(PMU_FALCON_BASE + 0x10c) & 0xffffffff:#x} "
        f"RING={dev.read32(0x10a4d0) & 0xffffffff:#x}", flush=True)
  falcon_reset(dev, PMU_FALCON_BASE)
  print(f"[kepler] PMU MC reset post: PMC_ENABLE={dev.read32(PMC_ENABLE) & 0xffffffff:#x} "
        f"SCRUB={dev.read32(PMU_FALCON_BASE + 0x10c) & 0xffffffff:#x}",
        flush=True)
  falcon_load(dev, PMU_FALCON_BASE, pmu_code, pmu_data, entry=0, start=True)
  try:
    setattr(dev, "_pmu_memx_data", None)
  except Exception:
    pass
  init = getattr(dev, "_init_pmu_memx", None)
  if init is None or not init():
    raise RuntimeError("PMU MEMX rediscovery failed after falcon reload")
  if not _smoke():
    raise RuntimeError("PMU MEMX DELAY smoke failed after falcon reload")
  print("[kepler] PMU MEMX ready after reload", flush=True)


def _gk104_pramin_write_memx(dev, pa, data) -> None:
  """Store PRAMIN words via PMU MEMX WR32 (host 0x1700 after bit0 kills TinyGPU).

  After deferred bit0, night14 showed host ``write32(0x1700, …)`` alone drops
  BAR0.  MEMX already programs ``0x10f*`` safely; use it for the PRAMIN
  window + XOR aperture stores.  Assume virgin ``0xffffffff`` after bit0
  (night5) so each store is ``0xffffffff ^ wanted`` without a host readback.
  """
  _gk104_ensure_pmu_memx_ready(dev)
  data = memoryview(data).cast("B")
  if len(data) & 3:
    raise ValueError("PRAMIN write must be 4-byte aligned")
  pairs: list[tuple[int, int]] = []
  window = None
  for off in range(0, len(data), 4):
    addr = pa + off
    base = addr & 0xffffff00000
    if base != window:
      pairs.append((0x001700, (base >> 16) & 0xffffffff))
      window = base
    reg = 0x700000 + (addr & 0xfffff)
    wanted = struct.unpack_from("<I", data, off)[0]
    # Virgin XOR aperture after bit0 unstub.
    pairs.append((reg, (0xffffffff ^ wanted) & 0xffffffff))
  # Same small chunks as nvbios _MemxBus.flush (cold PMU timeouts on large).
  chunk_n = 4
  execs = 0
  for i in range(0, len(pairs), chunk_n):
    chunk = pairs[i:i + chunk_n]
    payload: list[int] = []
    for addr, val in chunk:
      payload.extend([addr & 0xffffffff, val & 0xffffffff])
    try:
      if (dev.read32(0x10a580) & 0xffffffff) != 0:
        dev.write32(0x10a580, 0)
    except Exception:
      pass
    dev.pmu_memx_exec_commands([(PMU_MEMX_WR32, tuple(payload))], timeout_s=8.0)
    execs += 1
    if execs == 1 or execs % 64 == 0:
      _gk104_require_bar0_live(dev, f"during MEMX PRAMIN exec={execs}")
  _gk104_require_bar0_live(dev, f"after MEMX PRAMIN @{pa:#x}")
  if not hasattr(dev, "ops"):
    print(f"[kepler] PRAMIN via MEMX: pa={pa:#x} bytes={len(data)} "
          f"execs={execs}", flush=True)


def _gk104_pramin_write(dev, pa, data, *, force_bar0=False):
  """Store framebuffer words through BAR0 PRAMIN and verify each result.

  On the un-POSTed card, writes through 0x700000 sometimes behave as
  XOR updates against the old framebuffer word (an uncleared 0xffffffff plus
  value V becomes ~V).  Use current^wanted first, which is the correct delta
  for that path, then fall back to a literal store if the aperture is behaving
  normally.  Never return with silently inverted channel data.

  When BAR1 identity is up, the default path substitutes verified BAR1 stores
  (faster on TinyGPU).  H22: that substitute left mantissa bit15 sticky on
  compute mirror `b` (pre-GP_PUT xor=0x8000).  Pass force_bar0=True for
  eng-ctx / compute mirrors that must use the real 0x1700/0x700000 aperture.
  """
  # TinyGPU: once BAR0 is all-ones, 0x1700/0x700000 pokes hang the USB4 path.
  # Deferred bit0 (after PGRAPH/FECS) must run before the first PRAMIN store.
  if os.environ.get("KEPLER_RAM_BIT0_DEFER", "0") == "1":
    _gk104_bit0_unstub(dev)
  _gk104_require_bar0_live(dev, f"before PRAMIN store @{pa:#x}")
  if (not force_bar0 and
      getattr(dev, "_bar1_identity_ready", False)):
    _gk104_bar1_write_verified(
        dev, pa, data, label=f"BAR1 PRAMIN substitute @{pa:#x}")
    return
  # Host 0x1700 after bit0 kills TinyGPU (night14); prefer MEMX WR32.
  if (os.environ.get("KEPLER_PRAMIN_MEMX", "0") == "1" and
      hasattr(dev, "pmu_memx_exec_commands")):
    _gk104_pramin_write_memx(dev, pa, data)
    return
  data = memoryview(data).cast("B")
  if len(data) & 3:
    raise ValueError("PRAMIN write must be 4-byte aligned")
  window = None
  allow_literal = os.environ.get("KEPLER_PRAMIN_LITERAL", "1") != "0"
  # Night41ap: warm POSTed FB is a literal aperture. XOR-first then corrupts
  # (e.g. 0x4001→0x4000) and, with macOS KEPLER_PRAMIN_LITERAL=0, never
  # recovers. Virgin GDDR still reads 0xffffffff and needs XOR compensation.
  force_literal_first = os.environ.get("KEPLER_PRAMIN_LITERAL_FIRST", "") == "1"
  force_xor_first = os.environ.get("KEPLER_PRAMIN_LITERAL_FIRST", "") == "0"
  for off in range(0, len(data), 4):
    addr = pa + off
    base = addr & 0xffffff00000
    if base != window:
      dev.write32(0x001700, base >> 16)
      window = base
      _gk104_require_bar0_live(dev, f"after PRAMIN window @{base:#x}")
    reg = 0x700000 + (addr & 0xfffff)
    wanted = struct.unpack_from("<I", data, off)[0]
    current = dev.read32(reg) & 0xffffffff
    _gk104_require_bar0_live(dev, f"after PRAMIN read @{addr:#x}")
    if current == wanted:
      continue
    virgin = (current == 0xffffffff)
    # Warm/dirty dwords always prefer literal (Nouveau instmem).  KEPLER_PRAMIN_LITERAL
    # only gates the virgin-XOR → literal fallback (night13: literal 0 on XOR hung).
    literal_first = (
        force_literal_first or
        (not force_xor_first and not virgin))
    if literal_first:
      # Warm/dirty dword: Nouveau-shaped literal store (instmem wr32_slow).
      dev.write32(reg, wanted)
      actual = dev.read32(reg) & 0xffffffff
      _gk104_require_bar0_live(dev, f"after PRAMIN literal @{addr:#x}")
      if actual != wanted:
        # Fallback: treat as XOR aperture if literal did not stick.
        dev.write32(reg, (actual ^ wanted) & 0xffffffff)
        actual = dev.read32(reg) & 0xffffffff
        _gk104_require_bar0_live(dev, f"after PRAMIN XOR @{addr:#x}")
    else:
      # XOR aperture: virgin 0xffffffff + wanted 0 needs write of 0xffffffff,
      # not a literal 0 (literal 0 on XOR leaves all-ones and has killed TinyGPU).
      dev.write32(reg, (current ^ wanted) & 0xffffffff)
      actual = dev.read32(reg) & 0xffffffff
      _gk104_require_bar0_live(dev, f"after PRAMIN XOR @{addr:#x}")
      if actual != wanted and allow_literal:
        dev.write32(reg, wanted)
        actual = dev.read32(reg) & 0xffffffff
        _gk104_require_bar0_live(dev, f"after PRAMIN literal @{addr:#x}")
    if actual != wanted:
      stub = (" (FB aperture stub/untrained — cold RAM/MEMX did not enable "
              "PRAMIN)" if _gk104_pramin_word_is_stub(actual) else "")
      raise RuntimeError(f"PRAMIN store failed at {addr:#x}: "
                         f"wanted={wanted:#x} actual={actual:#x}{stub}")

def _gk104_pramin_read32(dev, pa, *, force_bar0=False):
  if (not force_bar0 and
      getattr(dev, "_bar1_identity_ready", False)):
    return _gk104_bar1_read32(dev, pa)
  base = pa & 0xffffff00000
  dev.write32(0x001700, base >> 16)
  return dev.read32(0x700000 + (pa & 0xfffff))


def _gk104_host_pramin_literal_probe(dev, pa: int = 0x30200,
                                     lit: int = None) -> tuple[int, str, int]:
  """Nouveau-shaped host PRAMIN outside FB_PAUSE (night40am H22/H23).

  Does not enter MEMX pause and does not clear bit0.  Returns (got, tag, r1620).
  """
  if lit is None:
    lit = PMU_BAR1_BOOTSTRAP_LITERAL
  r1620 = dev.read32(0x001620) & 0xffffffff
  base = pa & 0xffffff00000
  addr = pa & 0xfffff
  # nv50_instobj_wr32_slow / rd32_slow (instmem/nv50.c:62-91).
  dev.write32(0x001700, base >> 16)
  dev.write32(0x700000 + addr, lit & 0xffffffff)
  got = dev.read32(0x700000 + addr) & 0xffffffff
  tag = (
      "MATCH literal" if got == (lit & 0xffffffff) else
      "virgin ff" if got == 0xffffffff else
      "stub bad0fb" if (got & 0xffffff00) == 0xbad0fb00 else
      "stub badf" if (got & 0xffff0000) == 0xbadf0000 else
      "OTHER")
  print(f"[kepler] H22 host PRAMIN probe pa={pa:#x} "
        f"got={got:#x} ({tag}; wrote {lit:#x}) 0x1620={r1620:#x} "
        f"(no ENTER, no bit0 clear)", flush=True)
  return got, tag, r1620


def _gk104_pramin_write_literal(dev, pa, data):
  """Write raw dwords without the readback/XOR compensation."""
  if os.environ.get("KEPLER_RAM_BIT0_DEFER", "0") == "1":
    _gk104_bit0_unstub(dev)
  if getattr(dev, "_bar1_identity_ready", False):
    _gk104_bar1_write_verified(
        dev, pa, data, label=f"BAR1 literal substitute @{pa:#x}")
    return
  if (os.environ.get("KEPLER_PRAMIN_MEMX", "0") == "1" and
      hasattr(dev, "pmu_memx_exec_commands")):
    # Still use virgin-XOR encoding via MEMX — literal host path is TinyGPU-hostile.
    _gk104_pramin_write_memx(dev, pa, data)
    return
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
  if getattr(dev, "_tinygpu_post_bit0_bar_flush_unsafe", False):
    # Nights19/20 proved 0x070000 becomes a 43-ms timeout source after the
    # host bit0-only crossing, even with BAR1 enabled.  A BAR1 read is the only
    # safe posting barrier left on TinyGPU; verified writers additionally
    # require exact readback for every dword.
    try:
      raw = dev.dev_impl.hw.mmio_read(1, 0, 4)
      return len(raw) == 4
    except Exception:
      return False
  for _ in range(2):  # gf100_bar_bar1_wait() deliberately flushes twice.
    dev.write32(0x070000, 0x00000001)
    deadline = time.time() + 0.2
    while dev.read32(0x070000) & 0x00000002:
      if time.time() >= deadline:
        return False
      time.sleep(0.001)
  return True


def _gk104_ltc_cache_op(dev, reg: int, label: str) -> None:
  """Run one GF100/GK104 LTC cache operation and require completion."""
  dev.write32(reg, 0x00000001)
  deadline = time.monotonic() + 2.0
  status = dev.read32(reg) & 0xffffffff
  while status & 0x00000003:
    if time.monotonic() >= deadline:
      raise TimeoutError(f"LTC {label} timed out: {reg:#x}={status:#x}")
    time.sleep(0.001)
    status = dev.read32(reg) & 0xffffffff
  print(f"[kepler] LTC {label} complete: {reg:#x}={status:#x}", flush=True)


def _gk104_rom_window_pramin_ab_probe(dev, pa: int, physical: bytes) -> dict:
  """A/B Nouveau's inherited ROM-window predicates against PMU ground truth.

  ``physical`` must already have been read from ``pa`` through PMU direct-VRAM
  MEMIF.  This routine performs no framebuffer store.  It mirrors
  shadowramin.c's exact ``0x619f04`` encoding, checks PRAMIN before/after, then
  tests the independently sourced PCI ROM-shadow bit only if ROM_WINDOW alone
  was insufficient.  All modified registers are restored in ``finally``.
  """
  pa = int(pa)
  physical = bytes(physical)
  if pa & 0xffff:
    raise ValueError("ROM-window A/B physical address must be 64-KiB aligned")
  if not physical or len(physical) & 3 or len(physical) > 0x100000:
    raise ValueError("ROM-window A/B ground truth must be aligned and non-empty")
  display = dev.read32(0x022500) & 0xffffffff
  if display & 1:
    raise RuntimeError(
        f"ROM-window A/B refused: Nouveau says display disabled (0x22500={display:#x})")
  old_window = dev.read32(0x619f04) & 0xffffffff
  old_shadow = dev.read32(0x088050) & 0xffffffff
  old_selector = dev.read32(0x001700) & 0xffffffff
  wanted_window = ((pa >> 8) & 0xffffff00) | 0x00000009

  def read_pramin() -> bytes:
    dev.write32(0x001700, pa >> 16)
    return b"".join(struct.pack("<I", dev.read32(0x700000 + off) & 0xffffffff)
                    for off in range(0, len(physical), 4))

  baseline = after_window = after_shadow = b""
  classification = "unclassified"
  try:
    baseline = read_pramin()
    if baseline == physical:
      classification = "already-visible"
    else:
      dev.write32(0x619f04, wanted_window)
      window_readback = dev.read32(0x619f04) & 0xffffffff
      decoded = (window_readback & 0xffffff00) << 8
      if not (window_readback & 0x8) or (window_readback & 0x3) != 1 or decoded != pa:
        raise RuntimeError(
            f"ROM-window A/B write rejected: wanted={wanted_window:#x} "
            f"readback={window_readback:#x} decoded={decoded:#x}")
      after_window = read_pramin()
      if after_window == physical:
        classification = "rom-window-sufficient"
      else:
        dev.write32(0x088050, old_shadow | 1)
        shadow_readback = dev.read32(0x088050) & 0xffffffff
        if not (shadow_readback & 1):
          raise RuntimeError(
              f"ROM-window A/B PCI shadow bit rejected: {shadow_readback:#x}")
        after_shadow = read_pramin()
        classification = ("rom-window-plus-pci-shadow" if
                          after_shadow == physical else "still-stubbed")
  finally:
    dev.write32(0x088050, old_shadow)
    dev.write32(0x619f04, old_window)
    dev.write32(0x001700, old_selector)

  restored = (
      dev.read32(0x619f04) & 0xffffffff,
      dev.read32(0x088050) & 0xffffffff,
      dev.read32(0x001700) & 0xffffffff,
  )
  expected_restored = (old_window, old_shadow, old_selector)
  if restored != expected_restored:
    raise RuntimeError(
        f"ROM-window A/B restore mismatch: wanted={expected_restored} got={restored}")

  snap = {
      "pa": pa,
      "physical": physical,
      "baseline": baseline,
      "after_window": after_window,
      "after_shadow": after_shadow,
      "classification": classification,
      "old_window": old_window,
      "wanted_window": wanted_window,
      "old_shadow": old_shadow,
      "old_selector": old_selector,
      "restored": restored,
  }
  print(
      "[kepler] ROM_WINDOW/PRAMIN A/B: "
      f"pa={pa:#x} PMU={physical.hex()} baseline={baseline.hex()} "
      f"window={after_window.hex() or '-'} shadow={after_shadow.hex() or '-'} "
      f"classification={classification} restore_verified=True restored="
      f"0x619f04:{old_window:#x}/0x88050:{old_shadow:#x}/0x1700:{old_selector:#x}",
      flush=True)
  return snap


def _gk104_verify_physical_bar1_roots(dev, segments) -> None:
  """Prove PBUS sees PMU roots before asking its VM walker to consume them."""
  segments = tuple((int(pa), bytes(blob)) for pa, blob in segments)

  def observe(label):
    ok = True
    for pa, wanted in segments:
      got = dev.dev_impl.hw.mmio_read(1, pa, len(wanted))
      match = got == wanted
      ok = ok and match
      print(f"[kepler] physical BAR1 root {label} "
            f"{'OK' if match else 'FAIL'} pa={pa:#x} size={len(wanted):#x} "
            f"got={got.hex()}", flush=True)
    return ok

  # GF100 PBUS BAR1 with VM disabled and CHAN=0 exposes physical VRAM at VA=PA.
  dev.write32(0x001704, 0)
  if not _gk104_bar_flush(dev):
    raise TimeoutError("bar flush before physical BAR1 root proof timed out")
  if observe("before-LTC-handoff"):
    return

  # PMU MEMIF and PBUS are distinct clients.  Perform the standard Nouveau
  # writeback then invalidate once before declaring the bytes client-private.
  _gk104_ltc_cache_op(dev, 0x00070010, "flush")
  _gk104_ltc_cache_op(dev, 0x00070004, "invalidate")
  if not _gk104_bar_flush(dev):
    raise TimeoutError("bar flush after LTC handoff timed out")
  if not observe("after-LTC-handoff"):
    raise RuntimeError(
        "PMU roots pass MEMIF loadback but fail physical BAR1/PBUS readback; "
        "BAR1 VM enable skipped (client visibility/instmem lifecycle mismatch)")


# Keep the Nouveau function names at the adaptation boundary.  These wrappers
# deliberately correspond one-to-one with subdev/bar/gf100.c; platform-specific
# ordering belongs in the caller, not in renamed/reimplemented register code.
def _gf100_bar_bar1_init(dev, inst: int) -> None:
  addr = inst >> 12
  dev.write32(0x001704, 0x80000000 | addr)


def _gf100_bar_bar1_wait(dev) -> None:
  # gf100_bar_bar1_wait(): "NFI why it's twice" -- _gk104_bar_flush already
  # implements both g84_bar_flush() trigger/wait cycles.
  if not _gk104_bar_flush(dev):
    raise TimeoutError("Nouveau gf100_bar_bar1_wait did not complete")


# A line-by-line semantic map, not a claim that TinyGPU can preserve Nouveau's
# host ordering across bit0.  Every divergence is named and backed by a live
# observation so future comparisons cannot silently omit a phase.
GK104_BAR1_NOUVEAU_FLOW = (
    "gf100_vmm page-table stores",
    "g84_bar_flush",
    "gf100_vmm_flush PAGE_ALL|HUB_ONLY",
    "gf100_vmm_join instance stores",
    "g84_bar_flush",
    "gf100_bar_bar1_init",
    "gf100_bar_bar1_wait (g84_bar_flush x2)",
    "first BAR1 access",
)
GK104_BAR1_TINYGPU_FLOW = (
    "PMU direct-VRAM MEMIF ENABLE|IGNORE_ACTIVATION",
    "PMU host MEMIF store+load verify (pre-bit0)",
    "PMU magic+roots in DMEM (wait_go) [falcon fallback]",
    "gf100_bar_bar1_init",
    "gf100_bar_bar1_wait (g84_bar_flush x2)",
    "PMU GO [falcon fallback]",
    "bit0 unstub",
    "PMU delay + xdst + wr32 HUB PDB invalidate [falcon fallback]",
    "first BAR1 access",
)
GK104_BAR1_DIVERGENCES = (
    ("root stores", "post-bit0 PMU direct-VRAM xdst",
     "pre-bit0 VRAM discards PRAMIN and DMA stores (nights22/24)"),
    ("BAR1 init/wait", "pre-bit0 while FECS is delayed",
     "post-bit0 BAR/LTC/VMM host writes collapse BAR0 (night38)"),
    ("VMM invalidate", "omitted before first-ever lookup",
     "root stores occur after host VMM aperture disappears; no prior lookup/cache"),
)

def _gk104_topo_is_posted(gpc_topo: int) -> bool:
  """True when 0x409604 looks like a live POSTed GPC topology, not a sentinel."""
  topo_hi = gpc_topo & 0xffff0000
  return (gpc_topo != 0 and
          gpc_topo != 0xffffffff and
          topo_hi != 0xbadf0000 and
          topo_hi != 0xbad00000 and
          topo_hi != 0xbad0d000)


def _gk104_post_ownership_snapshot(dev, pramin_live: bool) -> dict:
  """Read the GF100 devinit/PRAMIN boundary without changing GPU state.

  Nouveau reads ``0x2240c[1]`` before POST, and this ROM's NVINIT script sets it
  at 0xac25 when devinit completes.  Capture it before Python runs that script
  so inherited platform state remains distinguishable from our own marker.
  Keep this checkpoint read-only: Night41g proved that guessing an activation
  write above the PMU-private framebuffer view is not a safe substitute for a
  POSTed PBUS client.
  """
  snap = {
      "devinit_posted": dev.read32(0x02240c) & 0xffffffff,
      "pramin_window": dev.read32(0x001700) & 0xffffffff,
      "pramin_live": bool(pramin_live),
  }
  print(
      "[kepler] POST ownership: "
      f"0x2240c={snap['devinit_posted']:#010x} "
      f"firmware_posted_bit={bool(snap['devinit_posted'] & 0x2)} "
      f"0x1700={snap['pramin_window']:#010x} "
      f"PRAMIN_live={snap['pramin_live']}",
      flush=True)
  return snap


def _gk104_pramin_stage_snapshot(dev, label: str, pa: int = 0) -> tuple:
  """Read one fixed physical PRAMIN window and restore its selector.

  The Nouveau golden trace has live PRAMIN before its first init write.  This
  diagnostic localises whether cold Python ever reaches that inherited state;
  it never writes framebuffer data or enables either BAR VM.
  """
  old = dev.read32(0x001700) & 0xffffffff
  dev.write32(0x001700, (int(pa) & 0xffffff00000) >> 16)
  try:
    words = tuple(dev.read32(0x700000 + i * 4) & 0xffffffff
                  for i in range(4))
  finally:
    dev.write32(0x001700, old)
  kinds = tuple("stub" if _gk104_pramin_word_is_stub(word) else
                "virgin" if word in (0, 0xffffffff) else "data"
                for word in words)
  print(f"[kepler] PRAMIN stage [{label}] pa={pa:#x} "
        f"words={[f'{word:#010x}' for word in words]} kinds={list(kinds)}",
        flush=True)
  return words


def _gk104_post_entry_probe(dev) -> dict:
  """Classify inherited POST state using only BAR0 reads.

  This deliberately does not retarget PRAMIN through 0x1700: the existing
  selector and first aperture dword are evidence about the state inherited at
  process entry.  A non-sentinel/non-virgin dword is positive evidence; zero
  or all-ones remains inconclusive rather than being promoted to "live".
  """
  snap = {
      "boot0": dev.read32(0x000000) & 0xffffffff,
      "devinit_posted": dev.read32(0x02240c) & 0xffffffff,
      "gpc_topology": dev.read32(0x409604) & 0xffffffff,
      "pramin_window": dev.read32(0x001700) & 0xffffffff,
      "pramin_word": dev.read32(0x700000) & 0xffffffff,
  }
  snap["posted_marker"] = bool(snap["devinit_posted"] & 0x2)
  snap["gpc_awake"] = _gk104_topo_is_posted(snap["gpc_topology"])
  snap["pramin_positive"] = (
      not _gk104_pramin_word_is_stub(snap["pramin_word"]) and
      snap["pramin_word"] not in (0x00000000, 0xffffffff))
  snap["night41h_ready"] = bool(
      _gk104_boot0_looks_live(snap["boot0"]) and
      snap["posted_marker"] and snap["gpc_awake"] and
      snap["pramin_positive"])
  print(
      "[kepler] POST entry probe: "
      f"boot0={snap['boot0']:#010x} 0x2240c={snap['devinit_posted']:#010x} "
      f"marker={snap['posted_marker']} topo={snap['gpc_topology']:#010x} "
      f"0x1700={snap['pramin_window']:#010x} "
      f"PRAMIN[0]={snap['pramin_word']:#010x} "
      f"night41h_ready={snap['night41h_ready']}",
      flush=True)
  return snap


def _gk104_rom_shadow_entry_probe(dev) -> dict:
  """Classify Nouveau's inherited PRAMIN VBIOS source using BAR0 reads only.

  This mirrors ``nvbios_ramin.pramin_init()`` in Nouveau shadowramin.c.  The
  routine treats PDISPLAY and ROM_WINDOW as inherited safety predicates; it
  does not write them or claim that setting them would initialise PBUS.
  ``0x088050`` is the GK104 BAR0 mirror of PCI config offset 0x50 observed in
  the golden trace and is reported separately from the RAMIN predicates.
  """
  snap = {
      "display_state": dev.read32(0x022500) & 0xffffffff,
      "rom_window": dev.read32(0x619f04) & 0xffffffff,
      "pci_rom_shadow": dev.read32(0x088050) & 0xffffffff,
      "pramin_window": dev.read32(0x001700) & 0xffffffff,
      "pramin_word": dev.read32(0x700000) & 0xffffffff,
  }
  snap["display_enabled"] = not bool(snap["display_state"] & 0x1)
  snap["rom_window_enabled"] = bool(snap["rom_window"] & 0x8)
  snap["rom_window_target"] = snap["rom_window"] & 0x3
  snap["rom_window_vram"] = snap["rom_window_target"] == 1
  snap["rom_window_base"] = (snap["rom_window"] & 0xffffff00) << 8
  snap["pci_rom_shadow_enabled"] = bool(snap["pci_rom_shadow"] & 0x1)
  snap["pramin_positive"] = (
      not _gk104_pramin_word_is_stub(snap["pramin_word"]) and
      snap["pramin_word"] not in (0x00000000, 0xffffffff))
  snap["ramin_source_eligible"] = bool(
      snap["display_enabled"] and snap["rom_window_enabled"] and
      snap["rom_window_vram"])
  snap["firmware_shadow_ready"] = bool(
      snap["ramin_source_eligible"] and snap["pramin_positive"])
  print(
      "[kepler] ROM shadow entry probe: "
      f"0x22500={snap['display_state']:#010x} "
      f"display_enabled={snap['display_enabled']} "
      f"0x619f04={snap['rom_window']:#010x} "
      f"window_enabled={snap['rom_window_enabled']} "
      f"target={snap['rom_window_target']} "
      f"base={snap['rom_window_base']:#x} "
      f"0x88050={snap['pci_rom_shadow']:#010x} "
      f"shadow_enabled={snap['pci_rom_shadow_enabled']} "
      f"0x1700={snap['pramin_window']:#010x} "
      f"PRAMIN[0]={snap['pramin_word']:#010x} "
      f"ramin_eligible={snap['ramin_source_eligible']} "
      f"firmware_shadow_ready={snap['firmware_shadow_ready']}",
      flush=True)
  return snap


def _gk104_pci_hw(dev):
  """Return the live PCI transport for config/PROM access."""
  hw = getattr(getattr(dev, "dev_impl", None), "hw", None)
  if hw is None:
    hw = getattr(dev, "hw", None)
  if hw is None:
    hw = getattr(getattr(dev, "iface", None), "pci_dev", None)
  if hw is None:
    hw = getattr(getattr(dev, "dev_impl", None), "pci_dev", None)
  return hw


def _gk104_pci_id32(dev) -> int:
  hw = _gk104_pci_hw(dev)
  if hw is None or not hasattr(hw, "read_config"):
    return 0
  try:
    return hw.read_config(0, 4) & 0xffffffff
  except Exception:
    return 0


def _gk104_pci_subsys(dev) -> int:
  hw = _gk104_pci_hw(dev)
  if hw is None or not hasattr(hw, "read_config"):
    return 0
  try:
    return hw.read_config(0x2c, 4) & 0xffffffff
  except Exception:
    return 0


def _gk104_dump_prom_rom(dev, out_path=None, max_size=0x80000) -> bytes:
  """Dump the SPI PROM via BAR0 0x300000 with PCI ROM-shadow disabled.

  Matches Nouveau ``nvbios_prom_*`` (clear ``0x088050`` bit0, read PROM window).
  Dumps up to 512 KiB, then trims to the end of the last complete PCI ROM
  image (PCIR walk).  Restores the shadow bit in ``finally``.
  """
  out_path = out_path or ONBOARD_VBIOS_CACHE
  old_shadow = dev.read32(0x088050) & 0xffffffff
  data = bytearray()
  try:
    # Nouveau: nvkm_pci_rom_shadow(false) clears bit0 to expose PROM.
    dev.write32(0x088050, old_shadow & ~0x1)
    hw = _gk104_pci_hw(dev)
    if hw is not None and hasattr(hw, "mmio_read"):
      for off in range(0, max_size, 0x1000):
        data.extend(bytes(hw.mmio_read(0, 0x300000 + off, 0x1000)))
    else:
      for off in range(0, max_size, 4):
        data.extend(struct.pack("<I", dev.read32(0x300000 + off) & 0xffffffff))
    raw = bytes(data)
    # Trim to the last complete PCI Expansion ROM image.
    end = 0
    off = 0
    while off + 0x20 <= len(raw) and raw[off:off + 2] == b"\x55\xaa":
      pcir_rel = struct.unpack_from("<H", raw, off + 0x18)[0]
      pcir = off + pcir_rel
      if pcir + 0x18 > len(raw) or raw[pcir:pcir + 4] != b"PCIR":
        break
      img_len = struct.unpack_from("<H", raw, pcir + 0x10)[0] * 512
      indi = raw[pcir + 0x15]
      nxt = off + img_len
      if img_len <= 0 or nxt > len(raw):
        break
      end = nxt
      if indi & 0x80:
        break
      off = nxt
    if end >= 0x200:
      raw = raw[:end]
    if len(raw) < 0x200:
      raise RuntimeError(
          f"PROM dump too small ({len(raw):#x} B); head={raw[:16].hex()}")
    # Prove BIT-I parse works for 0x1183/0x1184 before caching.
    image = nvbios_init.find_vbios_image(raw)
    scripts = nvbios_init.find_vbios_scripts(image)
    if not scripts:
      raise RuntimeError("PROM dump has no BIT-I scripts")
    pathlib.Path(out_path).write_bytes(raw)
    print(f"[kepler] dumped onboard PROM → {out_path} "
          f"({len(raw):#x} B, scripts={[hex(s) for s in scripts]})",
          flush=True)
    return raw
  finally:
    try:
      dev.write32(0x088050, old_shadow)
    except Exception:
      pass


def _gk104_resolve_vbios_path(dev=None, *, allow_dump=False) -> str:
  """Pick a VBIOS image safe for this 660 Ti board.

  Order: explicit ``KEPLER_VBIOS`` (not the Palit golden) → cached onboard dump
  → cold PROM dump → refuse Palit on Gigabyte/0x1183 unless
  ``KEPLER_ALLOW_FOREIGN_VBIOS=1``.
  """
  env = os.environ.get("KEPLER_VBIOS", "").strip()
  # Ignore a Palit path left in the environment — that is not an operator
  # choice for this board (older launchers setdefault'd it).
  if env and os.path.abspath(env) != os.path.abspath(PALIT_770_VBIOS):
    return env
  if os.path.exists(ONBOARD_VBIOS_CACHE):
    return ONBOARD_VBIOS_CACHE
  pci_id = (_gk104_pci_id32(dev) >> 16) & 0xffff if dev is not None else 0
  subsys = _gk104_pci_subsys(dev) if dev is not None else 0
  looks_660ti = (pci_id == 0x1183 or subsys == GIGABYTE_660TI_SUBSYS)
  foreign_ok = os.environ.get("KEPLER_ALLOW_FOREIGN_VBIOS", "0") == "1"
  if allow_dump and dev is not None:
    try:
      _gk104_dump_prom_rom(dev, ONBOARD_VBIOS_CACHE)
      return ONBOARD_VBIOS_CACHE
    except Exception as e:
      print(f"[kepler] onboard PROM dump failed: {e}", flush=True)
      if looks_660ti and not foreign_ok:
        raise RuntimeError(
            f"cold PROM dump failed on 660 Ti (pci_id={pci_id:#x} "
            f"subsys={subsys:#010x}): {e}. Power-cycle and retry, or set "
            "KEPLER_VBIOS= to a matching Gigabyte 0x1183 ROM "
            "(TechPowerUp #128422). Do not use the Palit GTX770 image.") from e
  if looks_660ti and not foreign_ok:
    raise RuntimeError(
        "add_660ti refuses Palit GTX770 VBIOS on this board "
        f"(pci_id={pci_id:#x} subsys={subsys:#010x}). "
        "Power-cycle the eGPU, then either: "
        "(1) re-run so cold PROM dump writes examples_kepler/onboard_gk104.rom, "
        "or (2) set KEPLER_VBIOS= to a Gigabyte 0x1183 ROM matching "
        "1458:3556 (TechPowerUp #128422). "
        "Set KEPLER_ALLOW_FOREIGN_VBIOS=1 only to force the Palit image.")
  return PALIT_770_VBIOS


GK104_GOLDEN_PREINIT_READS = (
    # Every unique non-PRAMIN/non-PROM MMIO read before trace line 42907,
    # Nouveau's first actual initialisation write (interrupt unarm at 0x140).
    (0x000004, 0x00000000),  # NV_PMC_BOOT_1: endian-switch state
    (0x000000, 0x0e4040a2),  # NV_PMC_BOOT_0: GK104 identity
    (0x101000, 0x8040509a),  # PSTRAPS / RAMCFG group 6
    (0x022500, 0x00000100),  # PDISPLAY enabled predicate
    (0x619f04, 0x00fffe09),  # PDISPLAY.VGA.ROM_WINDOW
    (0x001700, 0x0000ffb0),  # inherited PRAMIN selector
    (0x088050, 0x00000001),  # PCI config 0x50 BAR0 mirror
)


def _gk104_golden_preinit_entry_probe(dev) -> dict:
  """Compare every safe non-aperture golden read before device preinit.

  The table is mechanically extracted from the checked-in mmiotrace before
  line 42907.  PRAMIN/PROM aperture reads and their selector/shadow writes are
  deliberately excluded, leaving seven read-only registers.
  """
  actual = tuple((reg, dev.read32(reg) & 0xffffffff)
                 for reg, _expected in GK104_GOLDEN_PREINIT_READS)
  expected = dict(GK104_GOLDEN_PREINIT_READS)
  mismatch_regs = tuple(reg for reg, value in actual
                        if value != expected[reg])
  snap = {
      "actual": actual,
      "mismatch_regs": mismatch_regs,
      "exact_match": not mismatch_regs,
  }
  print(
      "[kepler] golden preinit entry diff: " + " ".join(
          f"{reg:#08x}={value:#010x}/golden={expected[reg]:#010x}"
          f"{'*' if value != expected[reg] else ''}"
          for reg, value in actual) +
      f" mismatches={[f'{reg:#x}' for reg in mismatch_regs]}",
      flush=True)
  return snap


def _gk104_pramin_word_is_stub(word: int) -> bool:
  """True when a PRAMIN dword is a dead/stub sentinel, not framebuffer data.

  Cold eGPUs that have not completed EFI/Nouveau memory training return the
  ``0xbad0fbXX`` aperture stub (low byte counts host touches).  ``0xbadf…``
  power-gate patterns are also unusable.  Do **not** treat ``0`` / ``0xffffffff``
  as stubs: virgin GDDR often reads all-ones, and the PRAMIN store path already
  compensates XOR-vs-``0xffffffff`` updates.
  """
  if (word & 0xffffff00) == 0xbad0fb00:
    return True
  if (word & 0xffff0000) in (0xbadf0000, 0xbad00000, 0xbad0d000):
    return True
  return False


def _gk104_pramin_looks_live(dev, pa=0x100000, soft=None) -> bool:
  """True when PRAMIN looks usable for channel instance stores.

  On TinyGPU, even a soft ``0x1700``/``0x700000`` PRAMIN *read* after
  post-MEMX ``0x1620[0]`` clear has collapsed BAR0 to all-ones and hung the
  USB4 path long enough for WindowServer watchdog (2026-07-16).  Default
  ``KEPLER_PRAMIN_SOFT_LIVE=1`` therefore accepts a live ``PMC_BOOT_0``
  without touching the aperture; channel-build stores remain the hard proof.

  ``soft=False`` forces the writeback probe (boot POSTed detection only —
  virgin/all-ones must not be mistaken for a POSTed card).
  """
  if soft is None:
    soft = os.environ.get("KEPLER_PRAMIN_SOFT_LIVE", "1") != "0"
  try:
    boot0 = dev.read32(0) & 0xffffffff
    if not _gk104_boot0_looks_live(boot0):
      return False
  except Exception:
    return False
  if soft:
    # Do not poke PRAMIN window — proven TinyGPU-hostile after bit0 unstub.
    return True
  try:
    word = _gk104_pramin_read32(dev, pa) & 0xffffffff
  except Exception:
    return False
  if _gk104_pramin_word_is_stub(word):
    return False
  pat = 0xa5a55a5a
  try:
    # Match _gk104_pramin_write: try XOR compensation first, then literal.
    base = pa & 0xffffff00000
    reg = 0x700000 + (pa & 0xfffff)
    dev.write32(0x001700, base >> 16)
    current = dev.read32(reg) & 0xffffffff
    if _gk104_pramin_word_is_stub(current):
      return False
    if current == pat:
      return True
    # Single XOR attempt; if BAR0 dies mid-probe, abort as not-live.
    dev.write32(reg, current ^ pat)
    actual = dev.read32(reg) & 0xffffffff
    boot0 = dev.read32(0) & 0xffffffff
    if not _gk104_boot0_looks_live(boot0):
      return False
    if actual != pat:
      dev.write32(reg, pat)
      actual = dev.read32(reg) & 0xffffffff
      boot0 = dev.read32(0) & 0xffffffff
      if not _gk104_boot0_looks_live(boot0):
        return False
    if actual != pat:
      return False
    # Best-effort restore (instance region is rewritten later anyway).
    now = dev.read32(reg) & 0xffffffff
    if now != current:
      dev.write32(reg, current)  # literal
      if (dev.read32(reg) & 0xffffffff) != current:
        dev.write32(reg, now ^ current)  # XOR aperture
    return True
  except Exception:
    return False

# Palit GTX 770 (GK104): NVIDIA 0x10de / 0x1184.  chip_id in PMC_BOOT_0 is 0xe4.
GK104_PCI_VENDOR_DEVICE = 0x118410DE
GK104_PMC_BOOT0_CHIP = 0xE4


def _gk104_boot0_looks_live(boot0: int) -> bool:
  if boot0 in (0, 0xffffffff):
    return False
  if (boot0 & 0xfffff000) == 0xbad00000:
    return False
  return ((boot0 >> 20) & 0xfff) == GK104_PMC_BOOT0_CHIP


def _gk104_require_bar0_live(dev, label: str) -> int:
  """Abort before spewing all-ones MMIO once TinyGPU BAR0 has collapsed."""
  try:
    boot0 = dev.read32(0) & 0xffffffff
  except Exception as e:
    raise RuntimeError(f"BAR0 unreadable {label}: {e}") from e
  if not _gk104_boot0_looks_live(boot0):
    raise RuntimeError(
        f"BAR0 dead {label} (PMC_BOOT_0={boot0:#x}); refusing further MMIO.  "
        "Power-cycle the eGPU enclosure and cold-run once.")
  return boot0


def _gk104_bit0_unstub(dev) -> None:
  """Legacy-only late framebuffer bit0 clear; once per device.

  MEMX ENTER owns this bit and LEAVE deliberately restores it.  Atomic mode
  must put framebuffer stores inside ENTER/LEAVE instead of clearing it late.
  """
  if getattr(dev, "_bit0_unstub_done", False):
    return
  if os.environ.get("KEPLER_RAM_BLOCK", "0") != "bit0":
    return
  _gk104_require_bar0_live(dev, "before deferred bit0 unstub")
  r1620 = dev.read32(0x001620) & 0xffffffff
  print(f"[kepler] deferred bit0 unstub: 0x1620 {r1620:#x}->{r1620 & ~1:#x}",
        flush=True)
  if os.environ.get("KEPLER_RAM_BIT0_FULL", "0") == "1":
    def _host_mask(reg: int, m: int, val: int) -> None:
      old = dev.read32(reg) & 0xffffffff
      dev.write32(reg, ((old & ~m) | val) & 0xffffffff)
    _host_mask(0x001620, 0x00000001, 0)
    _host_mask(0x0026f0, 0x00000001, 0)
    _host_mask(0x0026f0, 0x00000001, 0x00000001)
    _host_mask(0x001620, 0x00000001, 0x00000001)
  else:
    dev.write32(0x001620, r1620 & ~0x1)
  boot0 = _gk104_require_bar0_live(dev, "after deferred bit0 unstub")
  print(f"[kepler] deferred bit0 done; PMC_BOOT_0={boot0:#x}", flush=True)
  try:
    setattr(dev, "_bit0_unstub_done", True)
  except Exception:
    pass


def _gk104_ensure_bar0_mmio(hw, *, allow_reset: bool = True) -> tuple:
  """Recover BAR0 MMIO without assuming a physical replug is required.

  Tiered check (Apple PCIDriverKit: Close() clears COMMAND Memory Space /
  Bus Master; reopen must restore them before BAR access):

    1. ConfigurationRead32(0x00) — vendor/device still on the link?
    2. ConfigurationRead16(0x04) — restore MSE=1, keep MASTER=0 for probe
    3. Remap BAR0, read PMC_BOOT_0
    4. If still dead with valid ID+MSE → optional Function/Hot Reset
    5. Only then report a physical power/link cycle

  Returns ``(boot0, meta_dict)``.
  """
  meta = {
    "id32": None,
    "command_before": None,
    "command_after": None,
    "mse_before": None,
    "did_reset": False,
  }

  def _read_id():
    return hw.read_config(pci.PCI_VENDOR_ID, 4) & 0xffffffff

  def _restore_mse_only():
    """Open memory decode for BAR probe; leave Bus Master off until boot0 OK."""
    before = hw.read_config(pci.PCI_COMMAND, 2) & 0xffff
    if meta["command_before"] is None:
      meta["command_before"] = before
      meta["mse_before"] = bool(before & pci.PCI_COMMAND_MEMORY)
    want = ((before | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_INTX_DISABLE) &
            ~(pci.PCI_COMMAND_MASTER | pci.PCI_COMMAND_SERR | pci.PCI_COMMAND_PARITY))
    after = hw.write_config_flush(pci.PCI_COMMAND, want, 2) & 0xffff
    meta["command_after"] = after
    if not (after & pci.PCI_COMMAND_MEMORY):
      raise RuntimeError(
          f"PCI COMMAND.MSE did not stick (before={before:#06x} after={after:#06x}); "
          "TinyGPU reopen/init failed to restore memory decode — not a proven "
          "physical GPU death yet")
    return before, after

  def _remap_and_read_boot0():
    hw.bar_info(0)  # refresh BAR0 mapping after COMMAND change / reset
    return hw.mmio_read32(0, 0) & 0xffffffff

  id32 = _read_id()
  meta["id32"] = id32
  if id32 in (0, 0xffffffff):
    raise RuntimeError(
        f"PCI config space lost (vendor/device={id32:#010x}); "
        "endpoint/link is down — physical power or Thunderbolt link cycle required")

  vend = id32 & 0xffff
  devid = (id32 >> 16) & 0xffff
  if vend != 0x10de:
    raise RuntimeError(
        f"unexpected PCI vendor/device={id32:#010x} (want NVIDIA 0x10de/GK104)")

  _restore_mse_only()
  boot0 = _remap_and_read_boot0()
  if _gk104_boot0_looks_live(boot0):
    return boot0, meta

  # Config OK + MSE set, but BAR0 still all-ones / wrong chip → try PCI reset
  # before declaring a physical cycle.  TinyGPU exposes RemoteCmd.RESET;
  # Linux sysfs path may no-op or raise — treat failure as soft.
  if allow_reset and hasattr(hw, "reset") and callable(hw.reset):
    try:
      print(f"[kepler] BAR0 still dead (PMC_BOOT_0={boot0:#010x}) with "
            f"id={id32:#010x} cmd={meta['command_after']:#06x}; trying PCI reset",
            flush=True)
      hw.reset()
      meta["did_reset"] = True
      time.sleep(0.05)
      id32 = _read_id()
      meta["id32"] = id32
      if id32 in (0, 0xffffffff):
        raise RuntimeError(
            f"PCI config lost after Reset() (id={id32:#010x}); "
            "physical power/link cycle required")
      _restore_mse_only()
      boot0 = _remap_and_read_boot0()
      if _gk104_boot0_looks_live(boot0):
        return boot0, meta
    except RuntimeError:
      raise
    except Exception as e:
      print(f"[kepler] PCI reset unavailable or failed: {e}", flush=True)

  mse = bool((meta.get("command_after") or 0) & pci.PCI_COMMAND_MEMORY)
  raise RuntimeError(
      f"GK104 BAR0 MMIO still inaccessible after MSE restore"
      f"{' + Reset' if meta['did_reset'] else ''} "
      f"(PMC_BOOT_0={boot0:#010x}, PCI_ID={id32:#010x}/{vend:04x}:{devid:04x}, "
      f"COMMAND={meta.get('command_before'):#06x}->{meta.get('command_after'):#06x}, "
      f"MSE_was={meta.get('mse_before')}, MSE_now={mse}). "
      f"Config space is alive — this is a hung MMIO/link state, not a missing "
      f"device.  Try PCI reset path / enclosure power cycle; do not assume "
      f"replug solely from PMC_BOOT_0=0xffffffff.")


def _gk104_ltc_init(dev):
  """Initialize GK104 LTC (L2 cache) — gk104_ltc_init() + gf100_ltc_oneinit().

  Golden mmiotrace order (Nouveau on this Palit GTX 770):
    RAMMAP/training → LTC topology + ZBC clear → 0x17e8d8/0x17e000/0x17e8d4
    … much later → FECS.  Call this immediately after cold RAM init, before FECS.

  Compression tag RAM (0x17e8d4) is left at 0: the golden trace programs a
  VRAM-backed tag_base (0x7fddf) after Nouveau allocates tag memory; our
  userspace path has no fb tags heap yet.  Uncompressed compute does not
  require tags (gf100_ltc_oneinit_tag_ram's no-ram path).
  """
  # gf100_ltc_oneinit(): read LTC topology from hardware.
  # Nouveau reads 0x022438 unmasked; on this card it is 4.  Bound it anyway —
  # gated/sentinel reads previously hung bring-up in range(parts).
  parts = dev.read32(0x022438) & 0xff
  mask = dev.read32(0x022554)
  lts_nr = (dev.read32(0x17e8dc) >> 28) & 0xf
  ltc_nr = 0
  for i in range(min(parts, 32)):
    if not (mask & (1 << i)):
      ltc_nr += 1
  # Nouveau clears ZBC colour/depth slots 1..15 during LTC bring-up
  # (ltc/base.c: zbc_*_min=1 reserves index 0 for disabled; golden writes
  # 0x17ea44 = 1..15 then depth, never index 0).
  for i in range(1, 16):
    nvkm_mask(dev, 0x17ea44, 0x0000000f, i)
    for off in (0x17ea48, 0x17ea4c, 0x17ea50, 0x17ea54):
      dev.write32(off, 0)
  for i in range(1, 16):
    nvkm_mask(dev, 0x17ea44, 0x0000000f, i)
    dev.write32(0x17ea58, 0)
  tag_base = 0
  # gk104_ltc_init(): program LTC registers.
  lpg128 = not (dev.read32(0x100c80) & 0x00000001)
  dev.write32(0x17e8d8, ltc_nr)
  dev.write32(0x17e000, ltc_nr)
  dev.write32(0x17e8d4, tag_base)
  nvkm_mask(dev, 0x17e8c0, 0x00000002, 0x00000002 if lpg128 else 0x00000000)
  # Skip chatter on FakeMMIO recorders used by mmiotrace_selftest.
  if not hasattr(dev, "ops"):
    print(f"[kepler] LTC init: ltc_nr={ltc_nr} lts_nr={lts_nr} parts={parts} "
          f"mask={mask:#x} lpg128={lpg128} tag_base={tag_base:#x}", flush=True)

def _gk104_fb_init_page(dev):
  """Set GF100/GK104 big-page mode — gf100_fb_init_page().

  GK104 default is 17-bit (128 KiB) big pages: 0x100c80 bit0 = 0.
  """
  nvkm_mask(dev, 0x100c80, 0x00000001, 0x00000000)

def _gk104_post_ram_fb_ltc(dev):
  """Golden post-RAM cold step: fb_init_page then LTC/ZBC (before FECS).

  Nouveau order on this Palit GTX 770 (mmiotrace @~0.393s): 0x100c80 →
  ZBC 1..15 → 0x17e8d8/0x17e000/0x17e8d4/0x17e8c0.  Idempotent via
  ``_ltc_inited``.
  """
  if getattr(dev, "_ltc_inited", False):
    return
  _gk104_fb_init_page(dev)
  _gk104_ltc_init(dev)
  if not hasattr(dev, "ops"):
    try:
      c80 = dev.read32(0x100c80) & 0xffffffff
      t0 = dev.read32(0x110974) & 0xffffffff
      t1 = dev.read32(0x111974) & 0xffffffff
      print(f"[kepler] post-RAM LTC settle: 0x100c80={c80:#x} "
            f"(bigpage16={bool(c80 & 1)}) 0x110974={t0:#x} 0x111974={t1:#x}",
            flush=True)
    except Exception as e:
      print(f"[kepler] post-RAM LTC settle dump skipped: {e}", flush=True)
  try:
    setattr(dev, "_ltc_inited", True)
  except Exception:
    pass


def _gk104_bar1_read32(dev, va: int) -> int:
  """Read one dword through the CPU BAR1 aperture."""
  raw = dev.dev_impl.hw.mmio_read(1, va, 4)
  if len(raw) != 4:
    raise RuntimeError(f"short BAR1 read at {va:#x}: {len(raw)} bytes")
  return struct.unpack("<I", raw)[0]


_GK104_BAR1_CLIENT_REGS = (
    # GK104 RAM amount is per-FBP (gf100_ram_probe_fbpa_amount), not the
    # legacy NV50-era 0x10020c register.  Keep topology beside all four sizes.
    0x022438, 0x022554,
    0x11020c, 0x11120c, 0x11220c, 0x11320c,
    # VM state (H15/H17).
    0x100c10, 0x100c80,
    # HUB invalidate command/status plus the FB/LTC client image.
    0x100cb8, 0x100cbc, 0x10f000,
    0x17e030, 0x17e040, 0x17e8c0, 0x17e8d4, 0x17e8d8,
    # MEMX ownership and PBUS BAR aperture controls.
    0x001620, 0x0026f0, 0x001700, 0x001704,
)


def _gk104_dump_bar1_client_regs(dev, label: str) -> dict[int, int]:
  """Snapshot BAR0-visible PFB/PBUS state around BAR1 root activation.

  These are read-only observations; notably this avoids indexed PFB trap
  registers whose selector writes would perturb the first-failure state.
  """
  values = {}
  for reg in _GK104_BAR1_CLIENT_REGS:
    try:
      values[reg] = dev.read32(reg) & 0xffffffff
    except Exception as err:
      print(f"[kepler] BAR1 client snapshot {label}: {reg:#08x}=ERROR({err})",
            flush=True)
  rendered = " ".join(f"{reg:#08x}={value:#010x}"
                      for reg, value in values.items())
  print(f"[kepler] BAR1 client snapshot {label}: {rendered}", flush=True)
  return values


def _gk104_bar1_write_verified(dev, va: int, data, *, label="BAR1") -> None:
  """Write framebuffer dwords through BAR1 with XOR/literal compensation.

  Cold TinyGPU BAR1 has exhibited both ordinary stores and XOR-like/stale
  stores.  Read the current dword, try the exact XOR delta first, then a
  literal store, and require readback before advancing.  This replaces the
  old post-bit0 PRAMIN repair path, whose 0x1700 access is fatal.
  """
  data = memoryview(data).cast("B")
  if (va & 3) or (len(data) & 3):
    raise ValueError("verified BAR1 write must be dword aligned")
  for off in range(0, len(data), 4):
    addr = va + off
    wanted = struct.unpack_from("<I", data, off)[0]
    actual = _gk104_bar1_read32(dev, addr)
    if actual == wanted:
      continue
    for _attempt in range(4):
      delta = (actual ^ wanted) & 0xffffffff
      dev.dev_impl.hw.mmio_write(1, addr, struct.pack("<I", delta))
      actual = _gk104_bar1_read32(dev, addr)
      if actual == wanted:
        break
      dev.dev_impl.hw.mmio_write(1, addr, struct.pack("<I", wanted))
      actual = _gk104_bar1_read32(dev, addr)
      if actual == wanted:
        break
    if actual != wanted:
      raise RuntimeError(
          f"{label} store failed at {addr:#x}: wanted={wanted:#x} "
          f"actual={actual:#x}")


def _gk104_fecs_embed_bar1_bootstrap(fecs_code):
  """Splice the autonomous BAR1 bootstrap into the FECS IMEM zero pad.

  The live ``gk104_fecs_code.bin`` is 0xc00 bytes with zeros at 0xb0a..0xbff.
  Runtime IMEM stores after FECS has run read back 0xbadf5000 (night27), so
  the routine must be present in the image that ``falcon_write_imem`` loads.
  """
  code = bytearray(fecs_code)
  end = FECS_BAR1_BOOTSTRAP_END
  if end > len(code):
    raise RuntimeError(
        f"FECS bootstrap end {end:#x} exceeds firmware image {len(code):#x}")
  pad = memoryview(code)[FECS_BAR1_BOOTSTRAP_IMEM:end]
  if any(pad) and bytes(pad) != FECS_BAR1_BOOTSTRAP_CODE:
    raise RuntimeError(
        f"FECS IMEM pad at {FECS_BAR1_BOOTSTRAP_IMEM:#x} is not empty "
        f"(found {bytes(pad)[:16].hex()}...)")
  code[FECS_BAR1_BOOTSTRAP_IMEM:end] = FECS_BAR1_BOOTSTRAP_CODE
  print(f"[kepler] FECS BAR1 bootstrap embedded at "
        f"{FECS_BAR1_BOOTSTRAP_IMEM:#x}..{end:#x}", flush=True)
  return code


def _gk104_pmu_embed_bar1_bootstrap(pmu_code):
  """Embed the v4 autonomous PMU routine in the stock firmware zero pad."""
  code = bytearray(pmu_code)
  end = PMU_BAR1_BOOTSTRAP_END
  if end > len(code):
    raise RuntimeError(
        f"PMU bootstrap end {end:#x} exceeds firmware image {len(code):#x}")
  pad = memoryview(code)[PMU_BAR1_BOOTSTRAP_IMEM:end]
  if any(pad) and bytes(pad) != PMU_BAR1_BOOTSTRAP_CODE:
    raise RuntimeError(
        f"PMU IMEM pad at {PMU_BAR1_BOOTSTRAP_IMEM:#x} is not empty "
        f"(found {bytes(pad)[:16].hex()}...)")
  code[PMU_BAR1_BOOTSTRAP_IMEM:end] = PMU_BAR1_BOOTSTRAP_CODE
  print(f"[kepler] PMU BAR1 bootstrap embedded at "
        f"{PMU_BAR1_BOOTSTRAP_IMEM:#x}..{end:#x}", flush=True)
  return code


def _gk104_pmu_patch_live_imem_pad(dev) -> None:
  """Night40h: patch the pad into the live MEMX PMU without MC reset.

  Night40g proved Falcon IO XFER after MC-reload still yields BAR1 `bad0fb`.
  Night24's completing MEMIF path ran on the live post-FECS PMU.  Soft-stop
  leaves CPUCTL SLEEPING (never STOPPED); still try an IMEM pad write +
  readback there so MEMIF keeps the live engine's FB binding.
  """
  code = getattr(dev, "_pmu_code", None)
  if not code or len(code) < PMU_BAR1_BOOTSTRAP_END:
    raise RuntimeError("PMU image missing for live IMEM pad patch")
  if code[PMU_BAR1_BOOTSTRAP_IMEM:PMU_BAR1_BOOTSTRAP_END] != \
      PMU_BAR1_BOOTSTRAP_CODE:
    code = _gk104_pmu_embed_bar1_bootstrap(code)
    try:
      setattr(dev, "_pmu_code", bytes(code))
    except Exception:
      pass
  pad = PMU_BAR1_BOOTSTRAP_CODE
  print("[kepler] PMU live IMEM pad patch (no MC reset)", flush=True)
  tag_page = -1
  for off in range(0, len(pad), 4):
    addr = PMU_BAR1_BOOTSTRAP_IMEM + off
    page = addr >> 8
    if page != tag_page:
      dev.write32(PMU_FALCON_BASE + FALCON_CODE_TAG, page)
      tag_page = page
    dev.write32(PMU_FALCON_BASE + FALCON_CODE_INDEX, addr & 0x00ffffff)
    dev.write32(PMU_FALCON_BASE + FALCON_CODE,
                struct.unpack_from("<I", pad, off)[0])
  for off in range(0, len(pad), 4):
    addr = PMU_BAR1_BOOTSTRAP_IMEM + off
    want = struct.unpack_from("<I", pad, off)[0]
    got = _gk104_pmu_imem_read32(dev, addr)
    if got != want:
      raise RuntimeError(
          f"PMU live IMEM pad mismatch at {addr:#x}: "
          f"wanted={want:#x} actual={got:#x}")
  print(f"[kepler] PMU live IMEM pad verified "
        f"{PMU_BAR1_BOOTSTRAP_IMEM:#x}..{PMU_BAR1_BOOTSTRAP_END:#x}",
        flush=True)


def _gk104_pmu_reload_halted_for_bar1(dev) -> None:
  """MC-reset the PMU and reload the embedded image without START.

  Night40: after FECS ready, soft CPUCTL HALT never posts CPUSTAT.STOPPED on
  the live MEMX PMU, so falcon_stop() timed out before MEMIF arm.  Nouveau
  recovers this class of PMU wedging with gf100_pmu_reset() (PMC_ENABLE[13]).
  Reload leaves ENTRY at the pad and CPU halted so arm only needs START.
  Night40h prefers live IMEM patch; this remains the fallback when that
  readback fails.
  """
  code = getattr(dev, "_pmu_code", None)
  if not code or len(code) < PMU_BAR1_BOOTSTRAP_END:
    fw = find_kepler_firmware()
    if not fw:
      raise RuntimeError("PMU image missing for autonomous BAR1 reload")
    code = open(os.path.join(fw, "gk104_pmu_code.bin"), "rb").read()
    if os.environ.get("KEPLER_PMU_ENTER_NOWAIT", "1") != "0":
      code = _patch_pmu_memx_nowait(code)
    code = _gk104_pmu_embed_bar1_bootstrap(code)
    try:
      setattr(dev, "_pmu_code", bytes(code))
    except Exception:
      pass
  elif code[PMU_BAR1_BOOTSTRAP_IMEM:PMU_BAR1_BOOTSTRAP_END] != \
      PMU_BAR1_BOOTSTRAP_CODE:
    code = _gk104_pmu_embed_bar1_bootstrap(code)
    try:
      setattr(dev, "_pmu_code", bytes(code))
    except Exception:
      pass
  fw = find_kepler_firmware()
  if not fw:
    raise RuntimeError("GK104 firmware tree missing for PMU data reload")
  pmu_data = open(os.path.join(fw, "gk104_pmu_data.bin"), "rb").read()
  print("[kepler] PMU MC reset + halted reload for autonomous BAR1", flush=True)
  falcon_reset(dev, PMU_FALCON_BASE)
  falcon_load(dev, PMU_FALCON_BASE, code, pmu_data,
              entry=PMU_BAR1_BOOTSTRAP_IMEM, start=False)
  try:
    setattr(dev, "_pmu_memx_data", None)
  except Exception:
    pass


def _gk104_pmu_dma_prepare(dev, segments):
  """Validate fixed roots, halt PMU, and enable channel-independent MEMIF."""
  segments = tuple((int(pa), bytes(blob)) for pa, blob in segments)
  actual = tuple((pa, len(blob)) for pa, blob in segments)
  expected = tuple((ext, size) for _dmem, ext, size in PMU_BAR1_BOOTSTRAP_XFER)
  if actual != expected:
    raise RuntimeError(
        f"PMU autonomous bootstrap layout mismatch: actual={actual} "
        f"expected={expected}")
  blobs = tuple(blob for _pa, blob in segments)
  if blobs != PMU_BAR1_BOOTSTRAP_ROOTS:
    raise RuntimeError(
        "PMU autonomous bootstrap root constants mismatch: "
        f"actual={[b.hex() for b in blobs]} "
        f"expected={[b.hex() for b in PMU_BAR1_BOOTSTRAP_ROOTS]}")

  _gk104_ensure_pmu_memx_ready(dev)
  # Prefer a cheap soft halt when CPUSTAT.STOPPED posts (fake + idle falcons).
  # On the live post-FECS MEMX PMU (night40) it never does — try live IMEM
  # pad patch first (night40h); MC-reload only if that readback fails.
  try:
    falcon_stop(dev, PMU_FALCON_BASE, timeout_ms=200)
  except TimeoutError as err:
    cpuctl = dev.read32(PMU_FALCON_BASE + FALCON_UC_CTRL) & 0xffffffff
    cpustat = dev.read32(PMU_FALCON_BASE + 0x128) & 0xffffffff
    print(f"[kepler] PMU soft-stop missed STOPPED "
          f"(CPUCTL={cpuctl:#x} CPUSTAT={cpustat:#x}); {err}", flush=True)
    try:
      _gk104_pmu_patch_live_imem_pad(dev)
    except Exception as patch_err:
      print(f"[kepler] PMU live IMEM pad patch failed ({patch_err}); "
            f"falling back to MC reset", flush=True)
      _gk104_pmu_reload_halted_for_bar1(dev)
  # Standard Falcon MEMIF: port0 TYPE=VRAM(4), CTRL ENABLE(4)|
  # IGNORE_ACTIVATION(7).  Night24 read these exact transitions back as
  # PORT 0x110->0x114 and CTRL 0x110->0x190 before a completed store/load.
  port_reg = PMU_FALCON_BASE + 0x600
  ctrl_reg = PMU_FALCON_BASE + 0x624
  port = (dev.read32(port_reg) & ~0x7) | 0x4
  ctrl = dev.read32(ctrl_reg) | 0x00000090
  dev.write32(port_reg, port)
  dev.write32(ctrl_reg, ctrl)
  port_actual = dev.read32(port_reg) & 0xffffffff
  ctrl_actual = dev.read32(ctrl_reg) & 0xffffffff
  if (port_actual & 0x7) != 0x4 or (ctrl_actual & 0x90) != 0x90:
    raise RuntimeError(
        f"PMU direct-VRAM MEMIF rejected: port={port_actual:#x} "
        f"ctrl={ctrl_actual:#x}")
  print(f"[kepler] PMU local-root constants + MEMIF validated: "
        f"port={port_actual:#x} ctrl={ctrl_actual:#x}", flush=True)
  # Night40k live: clearing IGNORE left CTRL=0x220 STATUS=0x10012 (one store
  # pending forever) — channel-less MEMIF *requires* IGNORE_ACTIVATION to
  # retire transfers (matches Nouveau gm200_flcn_fw_load no-inst path).
  # Keep ENABLE|IGNORE for both host verify and falcon xdst.
  # Night40g: stage fixed roots into DMEM while halted; the pad no longer
  # constructs them (IO XFER path needs the instruction budget).
  for (dmem, _ext, size), blob in zip(
      PMU_BAR1_BOOTSTRAP_XFER, PMU_BAR1_BOOTSTRAP_ROOTS):
    assert len(blob) == size
    _gk104_pmu_dmem_write(dev, dmem, blob)
  for (dmem, _ext, size), blob in zip(
      PMU_BAR1_BOOTSTRAP_XFER, PMU_BAR1_BOOTSTRAP_ROOTS):
    actual = _gk104_pmu_dmem_read(dev, dmem, size)
    if actual != blob:
      raise RuntimeError(
          f"PMU DMEM root stage mismatch at {dmem:#x}: "
          f"wanted={blob.hex()} actual={actual.hex()}")
  print("[kepler] PMU DMEM roots staged for xdst+invalidate bootstrap", flush=True)


def _gk104_pmu_memif_wait_idle(dev, timeout_ms=50) -> None:
  """Wait for PMU XFER queue idle (CTRL pending clear, STATUS busy clear)."""
  ctrl_reg = PMU_FALCON_BASE + 0x118
  stat_reg = PMU_FALCON_BASE + 0x120
  deadline = time.monotonic() + timeout_ms / 1000.0
  while time.monotonic() < deadline:
    ctrl = dev.read32(ctrl_reg) & 0xffffffff
    stat = dev.read32(stat_reg) & 0xffffffff
    if not (ctrl & 1) and not (stat & 2):
      return
    time.sleep(0.0005)
  raise TimeoutError(
      f"PMU MEMIF XFER idle timed out: CTRL={dev.read32(ctrl_reg):#x} "
      f"STATUS={dev.read32(stat_reg):#x}")


def _gk104_pmu_memif_xfer(dev, *, local: int, ext_pa: int, size: int,
                          store: bool) -> None:
  """Host-originated PMU MEMIF DMA (night24 register path).

  EXT_BASE is PA>>8 with EXT_OFFSET 0 (envytools xfer.rst).  CTRL mode:
  data load=0, data store=2; size code s where bytes = 4<<s.
  """
  if size not in (4, 8, 16, 32, 64, 128, 256):
    raise ValueError(f"unsupported MEMIF xfer size {size}")
  # bytes = 4 << code
  code = size.bit_length() - 3
  if code < 0 or (4 << code) != size:
    raise ValueError(f"size {size} is not 4<<code")
  mode = 2 if store else 0
  ctrl = (mode << 4) | (code << 8)
  _gk104_pmu_memif_wait_idle(dev)
  dev.write32(PMU_FALCON_BASE + 0x110, (ext_pa >> 8) & 0xffffffff)
  dev.write32(PMU_FALCON_BASE + 0x114, local & 0xffffffff)
  dev.write32(PMU_FALCON_BASE + 0x11c, 0)
  dev.write32(PMU_FALCON_BASE + 0x118, ctrl)
  _gk104_pmu_memif_wait_idle(dev)


def _gk104_pmu_memif_verify_roots(dev) -> bool:
  """Night40k: pre-bit0 host MEMIF store+load of staged roots (ENABLE-only).

  Night24 completed DMA with IGNORE but loadback was ff.  Nouveau's no-inst
  falcon path sets IGNORE (gm200_flcn_fw_load); we already cleared it in
  prepare.  Exact loadback here proves whether physical VRAM accepts PMU
  stores before bit0.
  """
  scratch = 0x0dc0
  ok = True
  for (dmem, ext, size), blob in zip(
      PMU_BAR1_BOOTSTRAP_XFER, PMU_BAR1_BOOTSTRAP_ROOTS):
    assert len(blob) == size
    _gk104_pmu_memif_xfer(dev, local=dmem, ext_pa=ext, size=size, store=True)
    _gk104_pmu_dmem_write(dev, scratch, b"\xff" * size)
    _gk104_pmu_memif_xfer(dev, local=scratch, ext_pa=ext, size=size, store=False)
    got = _gk104_pmu_dmem_read(dev, scratch, size)
    match = got == blob
    print(f"[kepler] PMU MEMIF pre-bit0 {'store+load OK' if match else 'LOADBACK FAIL'} "
          f"pa={ext:#x} size={size:#x} got={got.hex()}", flush=True)
    ok = ok and match
  return ok


def _gk104_pmu_imem_read32(dev, addr: int) -> int:
  """Read one PMU IMEM dword through the host CODE port (explicit index)."""
  dev.write32(PMU_FALCON_BASE + FALCON_CODE_INDEX, addr & 0x00ffffff)
  return dev.read32(PMU_FALCON_BASE + FALCON_CODE) & 0xffffffff


def _gk104_pmu_dmem_read(dev, addr: int, size: int) -> bytes:
  """Read PMU DMEM bytes through the host DATA port (explicit index)."""
  if (addr & 3) or (size & 3) or size <= 0:
    raise ValueError("PMU DMEM read must be dword-aligned and non-empty")
  out = bytearray(size)
  for off in range(0, size, 4):
    dev.write32(PMU_FALCON_BASE + FALCON_DATA_INDEX, (addr + off) & 0x00ffffff)
    struct.pack_into("<I", out, off,
                     dev.read32(PMU_FALCON_BASE + FALCON_DATA) & 0xffffffff)
  return bytes(out)


def _gk104_pmu_dmem_write(dev, addr: int, data) -> None:
  """Write PMU DMEM bytes through the host DATA port (explicit index)."""
  data = memoryview(data).cast("B")
  if (addr & 3) or (len(data) & 3) or not data:
    raise ValueError("PMU DMEM write must be dword-aligned and non-empty")
  for off in range(0, len(data), 4):
    dev.write32(PMU_FALCON_BASE + FALCON_DATA_INDEX,
                ((addr + off) & 0x00ffffff) | FALCON_IDX_WRITE)
    dev.write32(PMU_FALCON_BASE + FALCON_DATA,
                struct.unpack_from("<I", data, off)[0])


def _gk104_pmu_arm_autonomous_bootstrap(dev) -> None:
  """Start the PMU pad; prove roots+magic, leave it spinning on host GO."""
  code = getattr(dev, "_pmu_code", None)
  if not code or len(code) < PMU_BAR1_BOOTSTRAP_END:
    raise RuntimeError("PMU firmware image missing; cannot arm BAR1 bootstrap")
  embedded = code[PMU_BAR1_BOOTSTRAP_IMEM:PMU_BAR1_BOOTSTRAP_END]
  if embedded != PMU_BAR1_BOOTSTRAP_CODE:
    raise RuntimeError(
        f"PMU bootstrap not embedded at {PMU_BAR1_BOOTSTRAP_IMEM:#x}: "
        f"wanted={PMU_BAR1_BOOTSTRAP_CODE[:4].hex()} "
        f"actual={embedded[:4].hex()}")
  imem0 = _gk104_pmu_imem_read32(dev, PMU_BAR1_BOOTSTRAP_IMEM)
  want0 = struct.unpack_from("<I", PMU_BAR1_BOOTSTRAP_CODE, 0)[0]
  if imem0 != want0:
    raise RuntimeError(
        f"PMU IMEM pad mismatch at {PMU_BAR1_BOOTSTRAP_IMEM:#x}: "
        f"wanted={want0:#x} actual={imem0:#x}")
  # Clear stale handshake state before START.
  _gk104_pmu_dmem_write(dev, PMU_BAR1_BOOTSTRAP_GO_DMEM,
                        struct.pack("<I", 0))
  _gk104_pmu_dmem_write(dev, PMU_BAR1_BOOTSTRAP_DELAY_ENTERED_DMEM,
                        struct.pack("<I", 0))
  _gk104_pmu_dmem_write(dev, PMU_BAR1_BOOTSTRAP_DONE_DMEM,
                        struct.pack("<I", 0))
  dev.write32(PMU_FALCON_BASE + 0x10c, 0x00000000)
  dev.write32(PMU_FALCON_BASE + FALCON_UC_ENTRY, PMU_BAR1_BOOTSTRAP_IMEM)
  falcon_start(dev, PMU_FALCON_BASE)
  deadline = time.monotonic() + 0.5
  magic = 0
  while time.monotonic() < deadline:
    magic = struct.unpack_from(
        "<I", _gk104_pmu_dmem_read(dev, PMU_BAR1_BOOTSTRAP_MAGIC_DMEM, 4))[0]
    if magic == PMU_BAR1_BOOTSTRAP_MAGIC:
      break
    time.sleep(0.001)
  else:
    cpuctl = dev.read32(PMU_FALCON_BASE + FALCON_UC_CTRL) & 0xffffffff
    cpustat = dev.read32(PMU_FALCON_BASE + 0x128) & 0xffffffff
    pc = dev.read32(PMU_FALCON_BASE + 0xff0) & 0xffffffff
    raise RuntimeError(
        f"PMU bootstrap did not publish DMEM magic "
        f"{PMU_BAR1_BOOTSTRAP_MAGIC:#x} at {PMU_BAR1_BOOTSTRAP_MAGIC_DMEM:#x}: "
        f"got={magic:#x} CPUCTL={cpuctl:#x} CPUSTAT={cpustat:#x} TRACEPC={pc:#x}")
  actual_roots = tuple(
      _gk104_pmu_dmem_read(dev, dmem, size)
      for dmem, _ext, size in PMU_BAR1_BOOTSTRAP_XFER)
  if actual_roots != PMU_BAR1_BOOTSTRAP_ROOTS:
    raise RuntimeError(
        "PMU bootstrap local roots mismatch before GO: "
        f"actual={[b.hex() for b in actual_roots]} "
        f"expected={[b.hex() for b in PMU_BAR1_BOOTSTRAP_ROOTS]}")
  # Night40z: MEMIF was programmed while halted; re-apply and prove it still
  # reads back while the pad spins on GO (before ENTER hides host MMIO).
  port_reg = PMU_FALCON_BASE + 0x600
  ctrl_reg = PMU_FALCON_BASE + 0x624
  port = (dev.read32(port_reg) & ~0x7) | 0x4
  ctrl = dev.read32(ctrl_reg) | 0x00000090
  dev.write32(port_reg, port)
  dev.write32(ctrl_reg, ctrl)
  port_actual = dev.read32(port_reg) & 0xffffffff
  ctrl_actual = dev.read32(ctrl_reg) & 0xffffffff
  if (port_actual & 0x7) != 0x4 or (ctrl_actual & 0x90) != 0x90:
    raise RuntimeError(
        f"PMU MEMIF lost before GO: port={port_actual:#x} ctrl={ctrl_actual:#x}")
  cpuctl = dev.read32(PMU_FALCON_BASE + FALCON_UC_CTRL) & 0xffffffff
  cpustat = dev.read32(PMU_FALCON_BASE + 0x128) & 0xffffffff
  pc = dev.read32(PMU_FALCON_BASE + 0xff0) & 0xffffffff
  try:
    setattr(dev, "_pmu_auto_bar1_armed", True)
  except Exception:
    pass
  print(f"[kepler] PMU autonomous BAR1 bootstrap armed at "
        f"{PMU_BAR1_BOOTSTRAP_IMEM:#x} magic={magic:#x} (waiting for GO) "
        f"TRACEPC={pc:#x} CPUCTL={cpuctl:#x} CPUSTAT={cpustat:#x} "
        f"MEMIF port={port_actual:#x} ctrl={ctrl_actual:#x}", flush=True)


def _gk104_pmu_signal_go(dev) -> None:
  """Release the autonomous ENTER→XFER→LEAVE pad without racing host MMIO."""
  _gk104_pmu_dmem_write(dev, PMU_BAR1_BOOTSTRAP_DELAY_ENTERED_DMEM,
                        struct.pack("<I", 0))
  _gk104_pmu_dmem_write(dev, PMU_BAR1_BOOTSTRAP_GO_DMEM,
                        struct.pack("<I", PMU_BAR1_BOOTSTRAP_GO))
  # Do not read PMU MMIO here: the pad may already have cleared 0x1620[0].
  # It restores the bit and publishes DONE before host polling resumes.
  print(f"[kepler] PMU BAR1 bootstrap GO={PMU_BAR1_BOOTSTRAP_GO:#x}; "
        "awaiting autonomous ENTER/XFER/LEAVE", flush=True)


def _gk104_fecs_dmem_write(dev, addr: int, data) -> None:
  """Write the FECS firmware's reserved 0x200..0x2ff xfer_data area."""
  data = memoryview(data).cast("B")
  if (addr & 3) or (len(data) & 3) or addr < 0x200 or addr + len(data) > 0x300:
    raise ValueError("FECS DMA scratch must be dword aligned within 0x200..0x2ff")
  # Night33: sequential access returned the first two root dwords followed by
  # the aperture sentinel 0xbadf5000 while FECS was halted.  Avoid relying on
  # the cold Falcon port's autoincrement, just as falcon_write_imem() already
  # does for GK104 IMEM.
  for off in range(0, len(data), 4):
    dev.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX,
                FALCON_IDX_WRITE | (addr + off))
    dev.write32(FECS_FALCON_BASE + FALCON_DATA,
                struct.unpack_from("<I", data, off)[0])


def _gk104_fecs_dmem_read(dev, addr: int, size: int) -> bytes:
  if (addr & 3) or (size & 3) or addr < 0x200 or addr + size > 0x300:
    raise ValueError("FECS DMA scratch must be dword aligned within 0x200..0x2ff")
  out = bytearray()
  for off in range(0, size, 4):
    dev.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX,
                0x02000000 | (addr + off))
    out += struct.pack("<I", dev.read32(
        FECS_FALCON_BASE + FALCON_DATA) & 0xffffffff)
  return bytes(out)


def _gk104_fecs_imem_patch(dev, addr: int, code) -> None:
  """Patch FECS IMEM at ``addr`` while the falcon is halted (not reset)."""
  code = memoryview(code).tobytes()
  if (addr & 3) or (len(code) & 3):
    raise ValueError(f"FECS IMEM patch must be dword-aligned ({addr:#x}+{len(code):#x})")
  # ITAG is per 256-byte page; reuse the page index the original load used.
  page = addr & ~0xff
  if (addr + len(code) - 1) & ~0xff != page:
    raise ValueError("FECS IMEM patch must stay inside one 256-byte page")
  dev.write32(FECS_FALCON_BASE + FALCON_CODE_TAG, page >> 8)
  for off in range(0, len(code), 4):
    dev.write32(FECS_FALCON_BASE + FALCON_CODE_INDEX, addr + off)
    dev.write32(FECS_FALCON_BASE + FALCON_CODE,
                struct.unpack_from("<I", code, off)[0])


def _gk104_fecs_dma_prepare(dev, segments, aperture_base: int):
  """Validate the pad's fixed roots and configure FECS before bit0."""
  segments = tuple((int(pa), bytes(blob)) for pa, blob in segments)
  staged = []
  local = 0x200
  for pa, blob in segments:
    if len(blob) not in (4, 8, 16, 32, 64, 128, 256):
      raise ValueError(f"unsupported FECS DMA root fragment size {len(blob):#x}")
    if pa < aperture_base:
      raise ValueError(f"FECS DMA root {pa:#x} precedes base {aperture_base:#x}")
    local = round_up(local, len(blob))
    staged.append((pa, blob, local, len(blob)))
    local += len(blob)

  # The assembled bootstrap hardcodes DMEM/ext offsets for the standard
  # 1 MiB instance bank.  Refuse any other layout rather than silently
  # DMA-ing the wrong VRAM addresses after bit0.
  expected = []
  for dmem, ext, size in FECS_BAR1_BOOTSTRAP_XFER:
    expected.append((aperture_base + ext, dmem, size))
  actual = [(pa, local, size) for pa, _blob, local, size in staged]
  if actual != expected:
    raise RuntimeError(
        f"FECS autonomous bootstrap layout mismatch: "
        f"actual={actual} expected={expected}")
  actual_blobs = tuple(blob for _pa, blob, _local, _size in staged)
  if actual_blobs != FECS_BAR1_BOOTSTRAP_ROOTS:
    raise RuntimeError(
        "FECS autonomous bootstrap root constants mismatch: "
        f"actual={[b.hex() for b in actual_blobs]} "
        f"expected={[b.hex() for b in FECS_BAR1_BOOTSTRAP_ROOTS]}")

  # hub.fuc uses MEM_BASE/MEM_TARGET plus Falcon port0 for physical VRAM
  # channel-header transfers.  Configure the same target while FECS MMIO is
  # still reachable; the Falcon routine itself issues xdst after bit0.
  dev.write32(0x409a04, aperture_base >> 8)
  dev.write32(0x409a20, 0x80000002)
  base_actual = dev.read32(0x409a04) & 0xffffffff
  target_actual = dev.read32(0x409a20) & 0xffffffff
  if base_actual != aperture_base >> 8 or target_actual != 0x80000002:
    raise RuntimeError(
        f"FECS physical-VRAM target rejected: base={base_actual:#x} "
        f"target={target_actual:#x}")
  print(f"[kepler] FECS local-root constants validated: count={len(staged)} "
        f"base={aperture_base:#x}", flush=True)
  return tuple(staged)


def _gk104_fecs_arm_autonomous_bootstrap(
    dev, *, already_stopped: bool = False) -> None:
  """Halt FECS and restart it at the pre-embedded BAR1 bootstrap delay loop.

  Night25: after bit0, FECS host MMIO (including XFER_CTRL) becomes
  0xffffffff, so the host cannot submit DMA.  The Falcon must issue xdst
  itself.  Halt (not reset) keeps stack/DMEM/ready-state intact.

  Night27: runtime IMEM stores at 0xb20 read back 0xbadf5000 once FECS has
  already run, so the bootstrap is embedded into ``_fecs_code`` before the
  initial ``falcon_write_imem`` instead of being patched here.

  Night28: short in-pad delay finished before host bit0 (discard stub).
  Night29/30: polling 0x1620 via nv_rd32 trapped back to main (PC=0x567,
  CPUCTL SLEEPING).  Night31 proved the no-call loop starts at 0xb29, but
  0x10000000 iterations had not produced roots by the 5-second check.
  Night35 exposed a timing race directly: arm sampled PC=0xb2f, already past
  the short delay.  Night36 restores 0x10000000 iterations, requires the PC
  specifically inside 0xb29..0xb2e before bit0, and waits 30 seconds after.
  """
  if FECS_BAR1_BOOTSTRAP_END > 0xc00:
    raise RuntimeError(
        f"FECS bootstrap {FECS_BAR1_BOOTSTRAP_IMEM:#x}..{FECS_BAR1_BOOTSTRAP_END:#x} "
        f"exceeds the live 0xc00-byte IMEM image")
  fecs_code = getattr(dev, "_fecs_code", None)
  if not fecs_code or len(fecs_code) < FECS_BAR1_BOOTSTRAP_END:
    raise RuntimeError("FECS firmware image missing; cannot arm BAR1 bootstrap")
  embedded = fecs_code[FECS_BAR1_BOOTSTRAP_IMEM:FECS_BAR1_BOOTSTRAP_END]
  if embedded != FECS_BAR1_BOOTSTRAP_CODE:
    raise RuntimeError(
        f"FECS bootstrap not embedded at {FECS_BAR1_BOOTSTRAP_IMEM:#x}: "
        f"wanted={FECS_BAR1_BOOTSTRAP_CODE[:4].hex()} "
        f"actual={embedded[:4].hex()}")
  if not already_stopped:
    falcon_stop(dev, FECS_FALCON_BASE, timeout_ms=1000)
  # Software fake still needs an explicit IMEM poke; silicon already has the
  # bytes from the initial load.
  if getattr(dev, "imem", None) is not None or type(dev).__name__ == "_AtomicBar1Dev":
    _gk104_fecs_imem_patch(dev, FECS_BAR1_BOOTSTRAP_IMEM, FECS_BAR1_BOOTSTRAP_CODE)
  # Clear HALT, set entry+PC, start.  Require PC in the in-pad delay loop
  # (no nv_rd32 calls — those trapped on night29/30).
  dev.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, 0x00000010)  # W1C HALT
  dev.write32(FECS_FALCON_BASE + 0x10c, 0x00000000)
  dev.write32(FECS_FALCON_BASE + FALCON_UC_ENTRY, FECS_BAR1_BOOTSTRAP_IMEM)
  dev.write32(FECS_FALCON_BASE + 0xff0, FECS_BAR1_BOOTSTRAP_IMEM)
  dev.write32(FECS_FALCON_BASE + FALCON_UC_CTRL, FALCON_UC_CTRL_START)
  cpuctl = dev.read32(FECS_FALCON_BASE + FALCON_UC_CTRL) & 0xffffffff
  entry = dev.read32(FECS_FALCON_BASE + FALCON_UC_ENTRY) & 0xffffffff
  pcs = []
  deadline = time.monotonic() + 0.1
  while time.monotonic() < deadline:
    pc = dev.read32(FECS_FALCON_BASE + 0xff0) & 0xffffffff
    pcs.append(pc)
    if FECS_BAR1_BOOTSTRAP_DELAY_START <= pc < FECS_BAR1_BOOTSTRAP_DELAY_END:
      break
    time.sleep(0.0005)
  else:
    raise RuntimeError(
        f"FECS bootstrap not in delay loop after ENTRY={FECS_BAR1_BOOTSTRAP_IMEM:#x}: "
        f"CPUCTL={cpuctl:#x} ENTRY={entry:#x} "
        f"pcs={[hex(p) for p in pcs[-8:]]}")
  try:
    setattr(dev, "_fecs_auto_bar1_armed", True)
  except Exception:
    pass
  print(f"[kepler] FECS autonomous BAR1 bootstrap armed at "
        f"{FECS_BAR1_BOOTSTRAP_IMEM:#x} pc={pc:#x}", flush=True)


def _gk104_wait_autonomous_roots(dev) -> int:
  """Require DONE after the PMU's autonomous ENTER→XFER→LEAVE transaction."""
  transferred = sum(size for _d, _e, size in PMU_BAR1_BOOTSTRAP_XFER)
  if getattr(dev, "_pmu_auto_bar1_applied", False):
    return transferred
  # Avoid touching PMU while the short critical section has host MMIO hidden.
  time.sleep(0.25)
  deadline = time.monotonic() + 2.0
  done = 0
  while time.monotonic() < deadline:
    done = struct.unpack_from(
        "<I", _gk104_pmu_dmem_read(dev, PMU_BAR1_BOOTSTRAP_DONE_DMEM, 4))[0]
    if done == PMU_BAR1_BOOTSTRAP_DONE:
      print(f"[kepler] PMU autonomous BAR1 roots DONE={done:#x} "
            f"xdst_bytes={transferred:#x}",
            flush=True)
      return transferred
    time.sleep(0.005)
  raise TimeoutError(
      f"PMU autonomous BAR1 roots did not restore/publish DONE: got={done:#x}")


def _gk104_pre_bit0_bar1_activate(dev, bar1_ctl: int) -> None:
  """Enable BAR1 and complete Nouveau's two waits before host MMIO disappears."""
  # The Falcon is still inside its verified long delay and bit0 is still set.
  # Arm BAR1 now, then use the ordinary readable g84_bar_flush path exactly as
  # gf100_bar_bar1_init()+gf100_bar_bar1_wait() do.  Night38 proved that moving
  # these writes after bit0 kills BAR0 even if their RPC requests return OK.
  inst = (bar1_ctl & 0x3fffffff) << 12
  _gf100_bar_bar1_init(dev, inst)
  _gf100_bar_bar1_wait(dev)
  _gk104_require_bar0_live(dev, "after pre-bit0 BAR1 activation")
  print("[kepler] pre-bit0 BAR1 activation: enable + two Nouveau flushes done",
        flush=True)


def _gk104_resolve_sysmem(dev):
  """Return the CPU-visible sysmem buffer (NVDev.vram or AtomicBar1Dev.vram)."""
  vram = getattr(dev, "vram", None)
  if vram is not None:
    return vram
  impl = getattr(dev, "dev_impl", None)
  if impl is not None:
    return getattr(impl, "vram", None)
  return None


def _gk104_host_pte(bus_pa: int) -> int:
  """gf100_vmm_valid() leaf for NVKM_MEM_TARGET_HOST: VOL|aper=2|VALID."""
  return (bus_pa >> 8) | 0x1 | (1 << 32) | (2 << 33)


def _gk104_host_join(pgd_bus_pa: int) -> int:
  """gf100_vmm_join_() PDB for HOST: target=2 | VOL | addr."""
  return (pgd_bus_pa | 0x6) & ((1 << 40) - 1)


def _gk104_host_pde_spt(spt_bus_pa: int) -> int:
  """gf100_vmm_pgd_pde() SPT (pt[1]) for HOST: target=2 | VOL | addr<<24."""
  return (2 << 32) | (1 << 34) | (spt_bus_pa << 24)


def _gk104_sysmem_write(dev, offset: int, data: bytes) -> None:
  """CPU store into the GPU-visible sysmem buffer (dev.vram on TinyGPU)."""
  blob = bytes(data)
  vram = _gk104_resolve_sysmem(dev)
  if vram is None:
    raise RuntimeError("sysmem BAR1 bootstrap needs dev.vram / dev_impl.vram")
  end = offset + len(blob)
  if end > len(vram):
    raise RuntimeError(
        f"sysmem write [{offset:#x},{end:#x}) exceeds buffer {len(vram):#x}")
  vram[offset:end] = blob


def _gk104_sysmem_read(dev, offset: int, size: int) -> bytes:
  vram = _gk104_resolve_sysmem(dev)
  if vram is None:
    raise RuntimeError("sysmem BAR1 bootstrap needs dev.vram / dev_impl.vram")
  return bytes(vram[offset:offset + size])


def _gk104_host_bar1_bootstrap(
    dev, bus_base: int, inst_pa: int, pgd_pa: int, spt_pa: int,
    join: bytes, pde: bytes, bootstrap_pte: bytes) -> int:
  """Publish BAR1 roots in CPU-writable HOST sysmem and enable 0x1704.

  Nouveau keeps ``bar->inst`` in ``NVKM_MEM_TARGET_INST`` (VRAM) and only
  allows HOST for the joined PGD/SPT (``gf100_vmm_join_`` /
  ``gf100_vmm_pgd_pde``).  Night40ab proved ENTER-scoped PMU ``xdst`` into
  VRAM still loadbacks as virgin ``ff``, so the 4 KiB instance cannot be
  staged the Nouveau way on this cold eGPU.

  This path is the remaining experiment: stage instance+PGD+SPT in the same
  sysmem buffer the channel VMM already uses, encode join/PDE/PTE as HOST
  (matching ``_gk104_pgd_entry`` / ``GK104MemoryManager``), and point
  ``0x1704`` at ``bus_base+inst`` — which Nouveau never does (PBUS CHAN has
  no aperture field).  Live success/failure of the first BAR1 walk is the
  verdict.
  """
  _gk104_require_bar0_live(dev, "before HOST BAR1 bootstrap")
  inst_addr = bus_base + inst_pa
  chan = inst_addr >> 12
  if chan > 0x3fffffff:
    raise RuntimeError(
        f"HOST BAR1 inst {inst_addr:#x} exceeds PBUS 0x1704 CHAN width")
  bar1_ctl = 0x80000000 | chan

  # Zero the instance page then install join at +0x200 (gf100_vmm_join_).
  _gk104_sysmem_write(dev, inst_pa, bytes(0x1000))
  _gk104_sysmem_write(dev, inst_pa + 0x200, join)
  _gk104_sysmem_write(dev, pgd_pa, pde)
  _gk104_sysmem_write(dev, spt_pa, bootstrap_pte)
  # Posting: CPU store → GPU-visible sysmem.  0x100c10 was programmed at
  # device init (gf100_fb_sysmem_flush_page_init).
  got_join = _gk104_sysmem_read(dev, inst_pa + 0x200, len(join))
  got_pde = _gk104_sysmem_read(dev, pgd_pa, len(pde))
  got_pte = _gk104_sysmem_read(dev, spt_pa, len(bootstrap_pte))
  if got_join != join or got_pde != pde or got_pte != bootstrap_pte:
    raise RuntimeError(
        "HOST BAR1 roots failed CPU readback "
        f"join={got_join.hex()} pde={got_pde.hex()} pte={got_pte.hex()}")

  if not _gk104_bar_flush(dev):
    raise TimeoutError("bar flush before HOST PDB invalidate timed out")
  if not _gk104_vmm_flush_pdb(
      dev, bus_base + pgd_pa, target=2, hub_only=True):
    print("[kepler] HOST HUB_ONLY PDB invalidate timed out; continuing",
          flush=True)
  if not _gk104_bar_flush(dev):
    raise TimeoutError("bar flush after HOST PDB invalidate timed out")
  _gk104_dump_bar1_client_regs(dev, "HOST before 0x1704")
  _gf100_bar_bar1_init(dev, inst_addr)
  _gf100_bar_bar1_wait(dev)
  _gk104_dump_bar1_client_regs(dev, "HOST after 0x1704+double-flush")
  try:
    setattr(dev, "_tinygpu_bar1_ctl", bar1_ctl)
    setattr(dev, "_pmu_auto_bar1_applied", True)
    setattr(dev, "_bar1_host_pdb", True)
  except Exception:
    pass
  print(f"[kepler] HOST BAR1 roots staged in sysmem: "
        f"inst={inst_addr:#x} pgd={bus_base + pgd_pa:#x} "
        f"spt={bus_base + spt_pa:#x} ctl={bar1_ctl:#x}",
        flush=True)
  return bar1_ctl


def _gk104_atomic_bar1_bootstrap(
    dev, segments, inst_pa: int) -> None:
  """Cross bit0 and establish the minimal BAR1 VM translation roots."""
  _gk104_require_bar0_live(dev, "before atomic BAR1 bootstrap")
  bar1_ctl = 0x80000000 | (inst_pa >> 12)
  segments = tuple((int(pa), bytes(blob)) for pa, blob in segments)

  # Night40s established that clearing 0x1620[0] after MEMX LEAVE is not a
  # usable steady state: BAR1 changes from bad0fb to all-ones.  Nouveau's
  # memx.fuc owns the bit strictly inside ENTER/LEAVE.  Night40t further
  # showed that PMU nv_wr32(PRAMIN) in that scope is not physically visible.
  # Night40u then proved a standalone ENTER cannot reply because that clear
  # hides PMU host MMIO.  The embedded Falcon pad therefore owns the entire
  # ENTER→direct-VRAM XFER→LEAVE transaction and publishes DONE only after
  # restoring host visibility.  BAR1 enable+flush follows DONE.
  if os.environ.get("KEPLER_BAR1_DIRECT_PHYS", "1") != "0":
    # Night40am (H22/H23): Nouveau writes INST via host PRAMIN outside pause.
    # Opt-in probe before the ENTER-scoped pad; stops with a classified got.
    if (os.environ.get("KEPLER_HOST_PRAMIN_PROBE", "0") == "1" and
        not getattr(dev, "_pmu_auto_bar1_applied", False)):
      got, tag, r1620 = _gk104_host_pramin_literal_probe(dev)
      raise RuntimeError(
          f"H22 night40am: host_pramin_got={got:#x} ({tag}); "
          f"0x1620={r1620:#x}; pad ENTER skipped")
    _gk104_pmu_dma_prepare(dev, segments)
    _gk104_pmu_arm_autonomous_bootstrap(dev)
    _gk104_pmu_signal_go(dev)
    transferred = _gk104_wait_autonomous_roots(dev)
    # Night40y: Falcon xdld inside ENTER hung forever (DONE unread as
    # 0xffffffff).  Night40ab: after LEAVE, host MEMIF loadback separates
    # "xdst did not stick in VRAM" from "VRAM ok but BAR1 walk fails".
    boot0 = _gk104_require_bar0_live(dev, "after autonomous ENTER/XFER/LEAVE")
    r1620 = dev.read32(0x001620) & 0xffffffff
    if not (r1620 & 1):
      raise RuntimeError(
          f"MEMX LEAVE did not restore framebuffer state: 0x1620={r1620:#x}")
    port_reg = PMU_FALCON_BASE + 0x600
    ctrl_reg = PMU_FALCON_BASE + 0x624
    dev.write32(port_reg, (dev.read32(port_reg) & ~0x7) | 0x4)
    dev.write32(ctrl_reg, dev.read32(ctrl_reg) | 0x00000090)
    scratch = 0x0dc0
    loadback_ok = True
    # The pad xdsts all staged roots; post-LEAVE loadback verifies each one
    # before BAR1 is enabled.
    pad_ext = {ext for _d, ext, _s in PMU_BAR1_BOOTSTRAP_XFER_PAD}
    for (dmem, ext, size), (_pa, wanted) in zip(
        PMU_BAR1_BOOTSTRAP_XFER, segments):
      if ext not in pad_ext:
        print(f"[kepler] PMU post-LEAVE MEMIF loadback SKIP pa={ext:#x} "
              f"(not in night40ag pad XFER)", flush=True)
        continue
      _gk104_pmu_dmem_write(dev, scratch, b"\xff" * size)
      try:
        _gk104_pmu_memif_xfer(
            dev, local=scratch, ext_pa=ext, size=size, store=False)
      except TimeoutError as err:
        loadback_ok = False
        print(f"[kepler] PMU post-LEAVE MEMIF loadback HUNG pa={ext:#x}: {err}",
              flush=True)
        continue
      got = _gk104_pmu_dmem_read(dev, scratch, size)
      match = got == wanted
      loadback_ok = loadback_ok and match
      print(f"[kepler] PMU post-LEAVE MEMIF loadback "
            f"{'OK' if match else 'FAIL'} pa={ext:#x} size={size:#x} "
            f"got={got.hex()}", flush=True)
    if not loadback_ok:
      raise RuntimeError(
          "autonomous PMU PRAMIN/xdst not visible to post-LEAVE MEMIF loadback; "
          "BAR1 enable skipped (physical store path still broken)")
    if os.environ.get("KEPLER_ROM_WINDOW_AB", "0") == "1":
      # Night41n/H69: use the PMU's exact physical high-VRAM bytes as an oracle,
      # then A/B the two inherited registers seen at the start of the golden
      # Nouveau trace.  Stop after classification; do not conflate it with the
      # later physical-BAR1 or VM paths.
      shadow_pa = 0xfffe0000
      _gk104_pmu_dmem_write(dev, scratch, b"\xff" * 16)
      _gk104_pmu_memif_xfer(
          dev, local=scratch, ext_pa=shadow_pa, size=16, store=False)
      physical = _gk104_pmu_dmem_read(dev, scratch, 16)
      ab = _gk104_rom_window_pramin_ab_probe(dev, shadow_pa, physical)
      raise RuntimeError(
          f"H69 Night41n classified: {ab['classification']}; "
          "intentional stop before physical BAR1/VM")
    # Night41ar H57: host BAR1/PBUS still returns bad0fb while post-LEAVE MEMIF
    # loadback proves the roots.  Nouveau enables 0x1704 from INST/VRAM without
    # a host physical BAR1 prove.  Default: trust MEMIF and arm VM; set
    # KEPLER_BAR1_REQUIRE_PHYSICAL=1 to keep the old hard gate.
    if os.environ.get("KEPLER_BAR1_REQUIRE_PHYSICAL", "0") == "1":
      _gk104_verify_physical_bar1_roots(dev, segments)
    else:
      try:
        _gk104_verify_physical_bar1_roots(dev, segments)
      except RuntimeError as err:
        print(f"[kepler] H57: skipping host physical BAR1 proof ({err}); "
              f"arming 0x1704 from MEMIF-proven roots", flush=True)
    # Nouveau stores PTs then flush → HUB_ONLY PDB invalidate → flush before
    # BAR1 enable.  Night40aa still saw bad0fb without this; try it now that
    # loadback proved the roots are in the MEMIF VRAM view.
    pgd_pa = segments[1][0]
    if not _gk104_bar_flush(dev):
      raise TimeoutError("bar flush before PDB invalidate timed out")
    if not _gk104_vmm_flush_pdb(dev, pgd_pa, target=0, hub_only=True):
      print("[kepler] HUB_ONLY PDB invalidate timed out; continuing", flush=True)
    if not _gk104_bar_flush(dev):
      raise TimeoutError("bar flush after PDB invalidate timed out")
    _gk104_dump_bar1_client_regs(dev, "PMU before 0x1704")
    _gf100_bar_bar1_init(dev, inst_pa)
    _gf100_bar_bar1_wait(dev)
    _gk104_dump_bar1_client_regs(dev, "PMU after 0x1704+double-flush")
    _gk104_require_bar0_live(dev, "after ENTER-scoped BAR1 enable/flush")
    try:
      setattr(dev, "_tinygpu_bar1_ctl", bar1_ctl)
      setattr(dev, "_pmu_auto_bar1_applied", True)
    except Exception:
      pass
    transferred = sum(len(blob) for _pa, blob in segments)
    print(f"[kepler] autonomous ENTER/XFER/LEAVE roots DONE and VM armed: "
          f"bytes={transferred:#x} ctl={bar1_ctl:#x} "
          f"0x1620={r1620:#x} PMC_BOOT_0={boot0:#x}",
          flush=True)
    return

  # Night40m/n: keep the live MEMX PMU (do not steal it with the BAR1 pad).
  # Fire DELAY then literal PRAMIN WR32 of roots + 0x1704 enable + two
  # 0x070000 flushes.  Host bit0 during DELAY; no PMU reply after crossing.
  #
  # MEMX's memx_func_wr32 uses nv_wr32(addr, data), so its payload is literal.
  # The XOR compensation used by host PRAMIN is a workaround for the observed
  # broken/stale host aperture and must not be carried into this Falcon path.
  if os.environ.get("KEPLER_BAR1_MEMX_POSTBIT0", "1") != "0":
    _gk104_ensure_pmu_memx_ready(dev)
    memx_data = getattr(dev, "_pmu_memx_data", None)
    if memx_data is None:
      raise RuntimeError("PMU MEMX data base missing for post-bit0 root script")
    data_base = memx_data[0] if isinstance(memx_data, tuple) else int(memx_data)
    pairs: list[int] = []
    literal = os.environ.get("KEPLER_BAR1_MEMX_LITERAL", "1") != "0"
    window = None
    for pa, blob in segments:
      for off in range(0, len(blob), 4):
        addr = pa + off
        base = addr & 0xffffff00000
        if base != window:
          pairs.extend([0x001700, (base >> 16) & 0xffffffff])
          window = base
        wanted = struct.unpack_from("<I", blob, off)[0]
        pairs.extend([0x700000 + (addr & 0xfffff),
                      wanted if literal else (0xffffffff ^ wanted) & 0xffffffff])
    pairs.extend([
        0x001704, bar1_ctl & 0xffffffff,
        0x00070000, 1,
        0x00070000, 1,
    ])
    # One WR32 cmd; payload length is pair count (addr/data words).
    commands = [
        (PMU_MEMX_DELAY, (100_000_000,)),  # 100 ms — covers host bit0
        (PMU_MEMX_WR32, tuple(pairs)),
    ]
    finish = None
    if type(dev).__name__ == "_AtomicBar1Dev":
      # Fake: run DELAY(no-op)+WR32 synchronously so XOR roots land in vram.
      window = 0
      for opcode, payload in commands:
        if opcode == PMU_MEMX_DELAY:
          continue
        assert opcode == PMU_MEMX_WR32
        for i in range(0, len(payload), 2):
          reg, value = payload[i], payload[i + 1]
          if reg == 0x1700:
            window = value << 16
          elif 0x700000 <= reg < 0x800000:
            pa = window + reg - 0x700000
            if literal:
              struct.pack_into("<I", dev.vram, pa, value)
            else:
              cur = struct.unpack_from("<I", dev.vram, pa)[0]
              struct.pack_into("<I", dev.vram, pa, cur ^ value)
          else:
            dev.write32(reg, value)
      setattr(dev, "_pmu_auto_bar1_applied", True)
    else:
      finish = pmu_memx_exec_fire(dev, data_base, commands)
    encoding = "literal" if literal else "virgin-XOR"
    print(f"[kepler] PMU MEMX post-bit0 root script fired "
          f"(DELAY 100ms + {encoding} WR32 roots/enable/flush); finish={finish}",
          flush=True)
    _gk104_bit0_unstub(dev)
    boot0 = _gk104_require_bar0_live(dev, "after MEMX-script BAR1 bit0 crossing")
    time.sleep(0.25)  # DELAY 100ms + WR32 margin at cold HUB clock
    try:
      setattr(dev, "_pmu_memx_data", None)
      setattr(dev, "_tinygpu_post_bit0_bar_flush_unsafe", True)
      setattr(dev, "_tinygpu_bar1_ctl", bar1_ctl)
      setattr(dev, "_pmu_auto_bar1_applied", True)
    except Exception:
      pass
    transferred = sum(len(blob) for _pa, blob in segments)
    print(f"[kepler] post-bit0 MEMX BAR1 roots armed: "
          f"bytes={transferred:#x} PMC_BOOT_0={boot0:#x}", flush=True)
    return

  # Legacy falcon-pad path (KEPLER_BAR1_MEMX_POSTBIT0=0).
  _gk104_pmu_dma_prepare(dev, segments)
  try:
    host_roots_ok = _gk104_pmu_memif_verify_roots(dev)
  except TimeoutError as err:
    print(f"[kepler] PMU host MEMIF verify hung ({err}); treating as FAIL",
          flush=True)
    host_roots_ok = False
  if host_roots_ok:
    print("[kepler] PMU host MEMIF roots committed pre-bit0; "
          "skipping falcon autonomous xdst", flush=True)
    _gk104_pre_bit0_bar1_activate(dev, bar1_ctl)
    _gk104_bit0_unstub(dev)
    boot0 = _gk104_require_bar0_live(dev, "after host-MEMIF BAR1 bit0 crossing")
    try:
      setattr(dev, "_pmu_memx_data", None)
      setattr(dev, "_tinygpu_post_bit0_bar_flush_unsafe", True)
      setattr(dev, "_tinygpu_bar1_ctl", bar1_ctl)
      setattr(dev, "_pmu_auto_bar1_applied", True)
    except Exception:
      pass
    transferred = sum(size for _d, _e, size in PMU_BAR1_BOOTSTRAP_XFER)
    print(f"[kepler] post-bit0 host-MEMIF BAR1 roots armed: "
          f"bytes={transferred:#x} PMC_BOOT_0={boot0:#x}", flush=True)
    return
  print("[kepler] PMU host MEMIF pre-bit0 loadback failed; "
        "falling back to falcon post-bit0 xdst+BAR1 enable", flush=True)
  _gk104_pmu_arm_autonomous_bootstrap(dev)
  _gk104_pmu_signal_go(dev)
  _gk104_bit0_unstub(dev)
  boot0 = _gk104_require_bar0_live(dev, "after staged BAR1 bit0 crossing")
  try:
    setattr(dev, "_pmu_memx_data", None)
    setattr(dev, "_tinygpu_post_bit0_bar_flush_unsafe", True)
    setattr(dev, "_tinygpu_bar1_ctl", bar1_ctl)
  except Exception:
    pass
  transferred = _gk104_wait_autonomous_roots(dev)
  _gk104_require_bar0_live(dev, "after autonomous PMU root wait")
  print(f"[kepler] post-bit0 PMU autonomous BAR1 roots armed: "
        f"bytes={transferred:#x} PMC_BOOT_0={boot0:#x}", flush=True)

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
  # Full PCI BAR1 is 128 MiB.  SPT for 4 KiB leaves needs pages*8 bytes
  # (gf100_vmm_desc_17_12 SPT span 15 → 0x40000 for 128 MiB).  The old
  # spt@0x50000/inst@0x60000 bank only held 16 MiB; SPT now lives at 1 MiB
  # (still below the 0x400000 heap, bit19-safe low half of that bank).
  if os.environ.get("KEPLER_BAR1_MAP_SIZE"):
    mapped_size = int(os.environ["KEPLER_BAR1_MAP_SIZE"], 0)
  elif os.environ.get("KEPLER_PRAMIN_MEMX", "0") == "1":
    mapped_size = int(os.environ.get("KEPLER_BAR1_MAP_SIZE", "0x8000000"), 0)
  if mapped_size > 0x8000000:
    mapped_size = 0x8000000
    print(f"[kepler] BAR1 identity clamped to {mapped_size:#x} (PCI BAR1)",
          flush=True)
  if mapped_size < 0x1000 or (mapped_size & 0xfff):
    raise ValueError(f"BAR1 mapped_size must be 4 KiB aligned, got {mapped_size:#x}")
  if getattr(dev, "_bar1_identity_ready", False):
    prior_size = getattr(dev, "_bar1_identity_size", mapped_size)
    prior_alias = getattr(dev, "_bar1_identity_userd", userd_alias_pa)
    if prior_size != mapped_size or prior_alias != userd_alias_pa:
      raise RuntimeError(
          f"BAR1 identity already active with size={prior_size:#x} "
          f"alias={prior_alias}, requested size={mapped_size:#x} "
          f"alias={userd_alias_pa}")
    return
  # Night41aw: classic BAR1 installed during pre-PMC POST — do not re-enter
  # the atomic MEMX pad (H57) or rewrite roots through a stubbing host BAR1.
  if getattr(dev, "_bar1_classic_posted", False):
    prior_size = getattr(dev, "_bar1_identity_size", mapped_size)
    prior_alias = getattr(dev, "_bar1_identity_userd", None)
    if prior_size == mapped_size and prior_alias == userd_alias_pa:
      print(f"[kepler] BAR1 already classic-posted "
            f"(0x1704={dev.read32(0x001704):#x}); skipping re-init",
            flush=True)
      return
    raise RuntimeError(
        f"BAR1 classic-posted size={prior_size:#x} alias={prior_alias}, "
        f"requested size={mapped_size:#x} alias={userd_alias_pa}")
  dev.write32(0x001704, dev.read32(0x001704) & ~0x80000000)
  # Nouveau's GF100-family VRAM allocator reserves the first 256 KiB for VGA
  # memory (ramgf100.c:rsvd_head).  INST+PGD stay just above that; SPT for
  # up to 128 MiB (0x40000 bytes) sits at 1 MiB so it neither collides with
  # INST nor the 0x400000 channel heap, and stays in a bit19-safe low half.
  inst_pa = 0x00040000
  pgd_pa = 0x00041000
  spt_pa = 0x00100000
  pages = mapped_size // 0x1000
  spt_bytes = pages * 8
  if min(pgd_pa, spt_pa, inst_pa) < 0x40000:
    raise RuntimeError("BAR1 roots overlap Nouveau's reserved VGA VRAM")
  if spt_pa + spt_bytes > 0x400000:
    raise RuntimeError(
        f"BAR1 SPT {spt_pa:#x}+{spt_bytes:#x} overlaps VRAM heap @0x400000")
  # Refuse layouts that cross a bit19 bank boundary (this eGPU aliases
  # pa^=0x80000 inside each 1 MiB).
  if (spt_pa ^ (spt_pa + spt_bytes - 1)) & 0x80000:
    raise RuntimeError(
        f"BAR1 SPT crosses bit19 bank: {spt_pa:#x}+{spt_bytes:#x}")
  # Night40ac live: HOST 0x1704 inst (bus_base+off) still walks bad0fb — PBUS
  # CHAN has no aperture; instance fetch is INST/VRAM only (gf100_bar_bar1_init).
  # Default off; KEPLER_BAR1_HOST_PDB=1 retains the experiment / offline coverage.
  use_host_pdb = (
      os.environ.get("KEPLER_BAR1_HOST_PDB", "0") == "1" and
      _gk104_resolve_sysmem(dev) is not None)

  # Instance block: PDB points at the page directory.
  # VRAM join (Nouveau BAR default): base = pd->addr.
  # HOST join (gf100_vmm_join_ / channel VMM): base = pd->addr | 2 | VOL.
  inst = bytearray(0x220)
  if use_host_pdb:
    struct.pack_into("<Q", inst, 0x200, _gk104_host_join(bus_base + pgd_pa))
  else:
    struct.pack_into("<Q", inst, 0x200, pgd_pa)
  struct.pack_into("<Q", inst, 0x208, mapped_size - 1)
  # PDE: 4-KiB SPT in high half (gf100_vmm_pgd_pde pt[1]).
  if use_host_pdb:
    pde = struct.pack("<Q", _gk104_host_pde_spt(bus_base + spt_pa))
  else:
    pde = struct.pack("<Q", (1 << 32) | (spt_pa << 24))

  # PTEs: VRAM identity (PFIFO USERD path) or HOST identity into sysmem.
  # HOST-PDB forces HOST leaves — the sysmem buffer is the only store that
  # sticks on this cold eGPU (night40ab).
  pte_data = bytearray(spt_bytes)
  for page in range(pages):
    if use_host_pdb or not map_vram:
      off = ((userd_alias_pa + page * 0x1000)
             if userd_alias_pa is not None and page < 2 else page * 0x1000)
      pte = _gk104_host_pte(bus_base + off)
    else:
      # Nouveau allocates a BAR1 VMA for the global USERD object.  Alias its
      # first pages at VA 0 while retaining identity mapping elsewhere.
      pa = ((userd_alias_pa + page * 0x1000)
            if userd_alias_pa is not None and page < 2 else page * 0x1000)
      pte = (pa >> 8) | 0x1
    struct.pack_into("<Q", pte_data, page * 8, pte)

  atomic_bootstrap = (
      os.environ.get("KEPLER_TINYGPU_ATOMIC_BAR1", "0") == "1" and
      os.environ.get("KEPLER_PRAMIN_MEMX", "0") == "1" and
      os.environ.get("KEPLER_RAM_BIT0_DEFER", "0") == "1" and
      not getattr(dev, "_bit0_unstub_done", False) and
      hasattr(dev, "pmu_memx_exec_commands"))
  if use_host_pdb:
    # Temporary control aliases: VA0 → SPT page, VA0x1000 → instance page.
    bootstrap_pte = struct.pack(
        "<QQ",
        _gk104_host_pte(bus_base + spt_pa),
        _gk104_host_pte(bus_base + inst_pa))
    _gk104_host_bar1_bootstrap(
        dev, bus_base, inst_pa, pgd_pa, spt_pa,
        bytes(inst[0x200:0x210]), pde, bootstrap_pte)
    actual = dev.dev_impl.hw.mmio_read(1, 0, len(bootstrap_pte))
    if actual != bootstrap_pte:
      _gk104_dump_bar1_client_regs(dev, "HOST control-walk mismatch")
      raise RuntimeError(
          f"HOST BAR1 control mapping mismatch: "
          f"wanted={bootstrap_pte.hex()} actual={actual.hex()} "
          f"(0x1704 CHAN has no HOST aperture; Nouveau keeps bar->inst in "
          f"NVKM_MEM_TARGET_INST / VRAM — night40ac)")
    print("[kepler] BAR1 root mechanism selected: HOST sysmem PDB "
          "(Nouveau join/PDE HOST encodings; 0x1704 inst=bus_base+off)",
          flush=True)
  elif atomic_bootstrap:
    # Atomic MEMX pad still uses the legacy 16 MiB first-512KiB root bank
    # baked into FECS/PMU falcon xdst layouts.  Refuse larger maps here.
    if mapped_size > 0x1000000:
      raise RuntimeError(
          "atomic BAR1 bootstrap only supports ≤16 MiB; "
          "use classic PRAMIN path (KEPLER_TINYGPU_ATOMIC_BAR1=0) for 128 MiB")
    # The PMU aperture becomes permanently host-inaccessible after bit0
    # (night17), so stage just enough BAR1 state before the host-only bit0
    # clear.  VA0 maps the first SPT page (control); VA0x1000 maps the instance
    # page for verification.  The full aperture limit is staged up front.
    # Remap to legacy PAs expected by the falcon pad (0x40000/0x50000/0x60000).
    pgd_pa = 0x00040000
    spt_pa = 0x00050000
    inst_pa = 0x00060000
    if use_host_pdb:
      struct.pack_into("<Q", inst, 0x200, _gk104_host_join(bus_base + pgd_pa))
    else:
      struct.pack_into("<Q", inst, 0x200, pgd_pa)
    struct.pack_into("<Q", inst, 0x208, mapped_size - 1)
    if use_host_pdb:
      pde = struct.pack("<Q", _gk104_host_pde_spt(bus_base + spt_pa))
    else:
      pde = struct.pack("<Q", (1 << 32) | (spt_pa << 24))
    bootstrap_inst = bytearray(inst)
    bootstrap_pte = struct.pack(
        "<QQ", (spt_pa >> 8) | 0x1, (inst_pa >> 8) | 0x1)
    # gf100_vmm_join() consumes only instance +0x200/+0x208.  Store those four
    # dwords plus the PDE and two temporary PTEs with direct-VRAM PMU MEMIF
    # while MEMX ENTER owns framebuffer access, and load them back exactly
    # before LEAVE.  Only the proven roots are then exposed through BAR1 VM.
    _gk104_atomic_bar1_bootstrap(
        dev, ((inst_pa + 0x200, bootstrap_inst[0x200:0x210]),
              (pgd_pa, pde), (spt_pa, bootstrap_pte)), inst_pa)

    actual = dev.dev_impl.hw.mmio_read(1, 0, len(bootstrap_pte))
    if actual != bootstrap_pte:
      _gk104_dump_bar1_client_regs(dev, "PMU control-walk mismatch")
      raise RuntimeError(
          f"verified-PMU BAR1 control mapping mismatch: "
          f"wanted={bootstrap_pte.hex()} actual={actual.hex()}")
    mechanism = ("autonomous PMU ENTER/XFER/LEAVE"
                 if os.environ.get("KEPLER_BAR1_DIRECT_PHYS", "1") != "0"
                 else "MEMX DELAY+WR32 after bit0")
    print(f"[kepler] BAR1 root mechanism selected: {mechanism}",
          flush=True)

  if use_host_pdb or atomic_bootstrap:
    # Populate the first SPT page through VA0 while it maps the SPT itself.
    # Entries 0/1 remain the control/instance aliases until expansion ends.
    _gk104_bar1_write_verified(
        dev, 0x10, pte_data[0x10:0x1000], label="BAR1 SPT page0")
    first_page_checks = {
        inst_pa // 0x1000: dev.dev_impl.hw.mmio_read(
            1, (inst_pa // 0x1000) * 8, 8),
        spt_pa // 0x1000: dev.dev_impl.hw.mmio_read(
            1, (spt_pa // 0x1000) * 8, 8),
    }
    for index, actual in first_page_checks.items():
      wanted = pte_data[index * 8:index * 8 + 8]
      if actual != wanted:
        raise RuntimeError(
            f"BAR1 SPT page0 identity[{index:#x}] mismatch: "
            f"wanted={wanted.hex()} actual={actual.hex()}")
    # First SPT page installed identity leaves for the rest of the SPT
    # (16 MiB → 8 pages; 128 MiB → 64 pages). Fill the remainder through
    # those identity mappings.
    for off in range(0x1000, spt_bytes, 0x1000):
      end = min(off + 0x1000, spt_bytes)
      _gk104_bar1_write_verified(
          dev, spt_pa + off, pte_data[off:end],
          label=f"BAR1 SPT page {off // 0x1000}")
    # Now access the first SPT page by its identity VA, replace the temporary
    # roots with the intended USERD aliases, and keep the identity page usable
    # for later verified PRAMIN substitutions.
    _gk104_bar1_write_verified(
        dev, spt_pa, pte_data[:0x10], label="BAR1 final alias PTEs")
    # PTE0/1 were used as temporary control aliases and may be cached.  BAR1
    # fini/init invalidates that private walk without touching the inaccessible
    # post-bit0 0x070000/0x100cbc paths.  Exact BAR1 readback below is the
    # posting barrier.
    bar1_ctl = getattr(dev, "_tinygpu_bar1_ctl",
                       0x80000000 | ((bus_base + inst_pa) >> 12
                                     if use_host_pdb else inst_pa >> 12))
    dev.write32(0x001704, bar1_ctl & ~0x80000000)
    dev.write32(0x001704, bar1_ctl)
    final_roots = dev.dev_impl.hw.mmio_read(1, spt_pa, 0x10)
    final_limit = dev.dev_impl.hw.mmio_read(1, inst_pa + 0x208, 8)
    if final_roots != pte_data[:0x10] or final_limit != inst[0x208:0x210]:
      inst_identity_pte = dev.dev_impl.hw.mmio_read(
          1, spt_pa + (inst_pa // 0x1000) * 8, 8)
      raise RuntimeError(
          f"expanded BAR1 roots mismatch roots={final_roots.hex()} "
          f"limit={final_limit.hex()} inst_pte={inst_identity_pte.hex()}")
    try:
      setattr(dev, "_bar1_identity_ready", True)
      setattr(dev, "_bar1_identity_size", mapped_size)
      setattr(dev, "_bar1_identity_userd", userd_alias_pa)
    except Exception:
      pass
    print(f"[kepler] {'HOST' if use_host_pdb else 'atomic'} BAR1 identity "
          f"ready: inst={inst_pa:#x} pgd={pgd_pa:#x} spt={spt_pa:#x} "
          f"size={mapped_size:#x}",
          flush=True)
    return

  _gk104_pramin_write(dev, inst_pa, inst)
  _gk104_pramin_write(dev, pgd_pa, pde)
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
    if userd_alias_pa is not None and userd_alias_pa < mapped_size:
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
  # Do NOT set _bar1_identity_ready here.  That flag diverts PRAMIN stores
  # through BAR1 self-walks; Night41ap showed BAR1@0x400000 → 0xbad0ac7f and
  # the subsequent BAR1 hammering wedged TinyGPU/USB4 (macOS hard hang).
  # Nouveau keeps instmem on BAR2/PRAMIN, not BAR1-into-BAR1.  Atomic/HOST
  # bootstrap paths may set the flag only after their control-walk proof.
  #
  # Mark classic-posted so channel-build FAST_ZERO/ctx-copy can use BAR1 page
  # bulk instead of per-dword PRAMIN (~4.5M TinyGPU RPCs / ~35s otherwise).
  # Warm reopen never re-runs post-POST BAR1, so this flag must be set here.
  try:
    setattr(dev, "_bar1_classic_posted", True)
    setattr(dev, "_bar1_identity_size", mapped_size)
    setattr(dev, "_bar1_identity_userd", userd_alias_pa)
  except Exception:
    pass
  if not hasattr(dev, "ops"):
    _gk104_bar1_verify_top(dev, mapped_size)
  if DEBUG:
    print(f"[kepler] BAR1 identity enabled inst={inst_pa:#x} pgd={pgd_pa:#x} "
          f"spt={spt_pa:#x} size={mapped_size:#x} "
          f"target={'VRAM' if map_vram else 'HOST'} "
          f"userd_alias={userd_alias_pa if userd_alias_pa is not None else 0:#x} "
          f"bus_base={bus_base:#x}",
          flush=True)

def _gk104_ltc_invalidate(dev):
  """Invalidate GK104's L2 after CPU BAR1/PRAMIN framebuffer stores."""
  if getattr(dev, "_tinygpu_post_bit0_bar_flush_unsafe", False):
    # 0x070010/0x070004 share the post-bit0-inaccessible BAR/LTC block with
    # 0x070000.  On the cold first-use path no GPU client has cached these
    # freshly written objects yet; use the safe BAR1 posting read instead.
    return _gk104_bar_flush(dev)
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
    # Prefer BAR1 page verify when classic literal BAR1 is up (same as
    # FAST_ZERO).  The old path re-issued every live PTE via per-dword
    # PRAMIN — dominant cost inside golden-context on TinyGPU.
    fast_spt = (os.environ.get("KEPLER_FAST_ZERO", "1") != "0" and
                bool(getattr(dev, "_bar1_classic_posted", False) or
                     getattr(dev, "_bar1_identity_ready", False)))
    if fast_spt:
      _gk104_bar_flush(dev)
      _xfer = int(os.environ.get("KEPLER_BAR1_CHUNK", "0x40000"), 0)
      if _xfer < 0x1000 or (_xfer & 0xfff):
        _xfer = 0x1000
      _sample = os.environ.get("KEPLER_ZERO_VERIFY", "sample") != "full"
      _bad_pages = []
      if _sample:
        for _poff in (0, spt_size // 2, max(0, spt_size - 8)):
          _n = min(8, spt_size - _poff)
          _got = bytes(dev.dev_impl.hw.mmio_read(1, dst_spt + _poff, _n))
          if _got != spt_image[_poff:_poff + _n]:
            _bad_pages.append(_poff)
            break
      if _bad_pages or not _sample:
        _bad_pages = []
        for _off in range(0, spt_size, _xfer):
          _n = min(_xfer, spt_size - _off)
          _got = bytes(dev.dev_impl.hw.mmio_read(1, dst_spt + _off, _n))
          if _got != spt_image[_off:_off + _n]:
            for _p in range(_off, _off + _n, 0x1000):
              _pn = min(0x1000, _off + _n - _p)
              if (bytes(dev.dev_impl.hw.mmio_read(1, dst_spt + _p, _pn)) !=
                  spt_image[_p:_p + _pn]):
                _bad_pages.append(_p)
      for _off in _bad_pages:
        _n = min(0x1000, spt_size - _off)
        _gk104_pramin_write(dev, dst_spt + _off, spt_image[_off:_off + _n])
    else:
      # BAR1 bulk writes can acknowledge a page transfer while leaving portions
      # of framebuffer memory unchanged.  Re-issue every live PTE as the native
      # 8-byte transaction used by Nouveau's VMM writer.
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

def _gk104_fe_pwr_force_on(dev):
  """Ack FECS ctx_4170s handshake and keep FE domain FORCE_ON.

  Without a PMU, FECS sets 0x404170 bit4 then waits for clear; if bit4
  sticks or FE_PWR drops to 0, eng-ctx load parks with sticky GPC busy.
  H18: FECS returns to main wait@0x567 (done) while FE_PWR=0 / GPC_STATUS=0x6.
  """
  try:
    _pwr = dev.read32(0x404170) & 0xffffffff
  except Exception:
    return
  # Prefer 0x12 (FORCE_ON|request-ack style) so bit1 stays set after FECS
  # clears bit4; fall back to 0x02 if the card only latches FORCE_ON.
  if _pwr & 0x10:
    dev.write32(0x404170, (_pwr & ~0x10) | 0x02)
  elif _pwr not in (0x02, 0x12):
    dev.write32(0x404170, 0x00000012)


def submit_launch(dev, words, signal_va, signal_pa, wait_value, done_value,
                  ib_batches=None):
  """Set up a GK104 compute channel (RAMIN + USERD + GPFIFO), push `words` into
  the GPFIFO ring, kick the channel, and poll the host semaphore until it
  reaches `done_value`.  Writes go straight into the CPU-coherent sysmem mmap
  (`dev.vram`); `signal_pa` is the mmap offset of the semaphore page.  The GR
  context-buffer content (RAMIN 0x0210) and the exact GPFIFO ring base wiring
  are KEPLER-TODO pending a nouveau GK104 channel trace on silicon.

  `ib_batches`, when set, is a list of push-word lists packed as consecutive
  GPFIFO entries (GP_PUT=len).  Use this to stay under the ~19-LAUNCH/IB
  live ceiling; `words` is ignored when `ib_batches` is provided.
  """
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
  if ib_batches is not None:
    if not ib_batches:
      raise ValueError("ib_batches must be non-empty")
    words = [w for batch in ib_batches for w in batch]
  else:
    ib_batches = [words]
  n_gp_entries = len(ib_batches)
  if n_gp_entries > 512:
    raise ValueError(f"too many GPFIFO entries ({n_gp_entries}); max 512")
  total_push_bytes = sum(len(b) for b in ib_batches) * 4
  push_alloc_size = max(0x10000, (total_push_bytes + 0xffff) & ~0xffff)
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
  push = alloc.alloc(push_alloc_size, align=0x10000)  # Nouveau main push buffer
  # GR engine context buffer (RAMIN 0x0210).  nouveau allocates
  # CB_RESERVED(0x80000) + gr->size and fills it via gf100_grctx_generate_main
  # (bundle/pagepool/attrib_cb ctxsw bundles).
  gr_ctx = alloc.alloc(0x100000)
  # Global GR buffers needed by grctx->main() (ctxgf100.c / ctxgk104.c).
  # pagepool: 0x8000 bytes, bundle_cb: 0x3000 bytes,
  # attrib_cb: 0x20 * (attrib_nr_max + alpha_nr_max) * tpc_total
  pagepool = alloc.alloc(0x8000, align=0x1000)
  bundle_cb = alloc.alloc(0x3000, align=0x1000)
  from grctx_gk104 import GK104_GRCTX_CONSTS as _GC
  _gpc_n = max(1, dev.read32(0x409604) & 0x1f)
  _tpc_nr = _gk104_read_tpc_nr(dev, _gpc_n)
  _tpc_total = sum(_tpc_nr) or 8
  _attrib_nr_max = _GC["attrib_nr_max"]
  _alpha_nr_max = _GC["alpha_nr_max"]
  _attrib_nr = _GC["attrib_nr"]
  _alpha_nr = _GC["alpha_nr"]
  # Full Nouveau attrib CB is ~0xb2300 for 8 TPCs.  With bit19 aliasing only
  # 512 KiB per 1 MiB bank is unique — shrink counts so the CB fits.
  if os.environ.get("KEPLER_VRAM_BIT19_SAFE", "1") != "0":
    _sum_max = max(1, 0x80000 // (0x20 * _tpc_total))
    if _attrib_nr_max + _alpha_nr_max > _sum_max:
      _attrib_nr_max = min(_attrib_nr_max, max(1, _sum_max // 3))
      _alpha_nr_max = max(1, _sum_max - _attrib_nr_max)
      _attrib_nr = min(_attrib_nr, _attrib_nr_max)
      _alpha_nr = min(_alpha_nr, _alpha_nr_max)
      print(f"[kepler] bit19-safe attrib shrink: "
            f"attrib_nr_max={_attrib_nr_max:#x} alpha_nr_max={_alpha_nr_max:#x} "
            f"attrib_nr={_attrib_nr:#x} alpha_nr={_alpha_nr:#x} "
            f"tpc={_tpc_total}", flush=True)
  # H16: eng-ctx hang's last MMIO is GPC3 PPC+0xe4 = 0x51b0e4 with the
  # *shrunk* e4 value (0x61839e4).  Golden/grctx_main previously programmed
  # full Nouveau alpha_nr=0x648 — mmio-list then overwrote with shrunk
  # numbers and left GPC1+2 sticky.  Keep one consistent set on the device.
  try:
    setattr(dev, "_kepler_attrib", {
        "attrib_nr_max": _attrib_nr_max,
        "alpha_nr_max": _alpha_nr_max,
        "attrib_nr": _attrib_nr,
        "alpha_nr": _alpha_nr,
    })
  except Exception:
    pass
  _attrib_size = round_up(
      0x20 * (_attrib_nr_max + _alpha_nr_max) * _tpc_total, 0x1000)
  attrib_cb = alloc.alloc(_attrib_size, align=0x1000)
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
  _ppc_masks = [(dev.read32(0x500c30 + gpc * 0x8000) & 0xff) or
                ((1 << _tpc_nr[gpc]) - 1) for gpc in range(len(_tpc_nr))]
  _bundle_size = _GC["bundle_size"]
  _state_limit = min(_GC["bundle_min_gpm_fifo_depth"], _bundle_size // 0x20)
  _alpha, _beta = _alpha_nr, _attrib_nr
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
  _bo, _ao = 0, _attrib_nr_max * sum(_tpc_nr)
  # H16b: last eng-ctx MMIO is GPC3 PPC+0xe4 (0x51b0e4).  Default keep PPC
  # entries (Nouveau-shaped).  Set KEPLER_PPC_MMIO_LIST=0 to omit and test
  # whether mmio-list PPC restores alone cause sticky GPC1+2.
  _ppc_mmio = os.environ.get("KEPLER_PPC_MMIO_LIST", "1") != "0"
  for _gpc, _mask in enumerate(_ppc_masks):
    _count = _mask.bit_count()
    _ppc = 0x503000 + _gpc * 0x8000
    _c0 = (1 << 28) | (_beta * _count << 16) | _bo
    _e4 = (_alpha * _count << 16) | _ao
    if _ppc_mmio:
      runtime_mmio_entries.append((_ppc + 0xc0, _c0))
      runtime_mmio_entries.append((_ppc + 0xe4, _e4))
    _bo += _attrib_nr_max * _count
    _ao += _alpha_nr_max * _count
  if _ppc_mmio:
    print(f"[kepler] PPC mmio-list: "
          f"{[(hex(r), hex(v)) for r, v in runtime_mmio_entries if (r & 0xff) in (0xc0, 0xe4) and 0x503000 <= r <= 0x51b0e4]}",
          flush=True)
  else:
    print("[kepler] PPC mmio-list: omitted (H16b; set KEPLER_PPC_MMIO_LIST=1 "
          "to include)", flush=True)
  # H10 discriminator: SET_OBJECT hang leaves FECS_MMIO_CTRL as WRITE to
  # 0x17e920.  Nouveau patch_ltc puts these in the channel mmio list; default
  # omit them so we can prove/disprove mmio-list LTC as the sticky-GPC cause.
  # Golden FECS save may still restore LTC via strands.  Set
  # KEPLER_LTC_MMIO_LIST=1 to restore Nouveau-shaped entries.
  if os.environ.get("KEPLER_LTC_MMIO_LIST", "0") == "1":
    _ltc_a, _ltc_b = dev.read32(0x17e91c), dev.read32(0x17e920)
    runtime_mmio_entries.extend(((0x17e91c, _ltc_a), (0x17e920, _ltc_b)))
    print(f"[kepler] LTC mmio-list: included 0x17e91c={_ltc_a:#x} "
          f"0x17e920={_ltc_b:#x}", flush=True)
  else:
    print(f"[kepler] LTC mmio-list: omitted (H10; set KEPLER_LTC_MMIO_LIST=1 "
          f"to include) live=0x17e91c={dev.read32(0x17e91c):#x} "
          f"0x17e920={dev.read32(0x17e920):#x}", flush=True)
  mmio_blob = b"".join(struct.pack("<II", reg, value & 0xffffffff)
                       for reg, value in runtime_mmio_entries)
  # H16c: empty the channel mmio list so eng-ctx load skips FECS MMIO walk.
  # If GPC_STATUS still sticks, strands (not mmio-list) are the hang site.
  if os.environ.get("KEPLER_MMIO_LIST_EMPTY", "0") == "1":
    print(f"[kepler] MMIO list emptied (H16c; had {len(runtime_mmio_entries)} "
          f"entries)", flush=True)
    runtime_mmio_entries = []
    mmio_blob = b""
  vram[mmio_list.meta["pa"]:mmio_list.meta["pa"] + max(len(mmio_blob), 8)] = (
      mmio_blob + bytes(max(0, 8 - len(mmio_blob))))
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
  # Incomplete GDDR address train on this eGPU aliases pa ^= 0x80000 (bit19).
  # Default: only allocate in the low half of each 1 MiB bank.
  bit19_safe = os.environ.get("KEPLER_VRAM_BIT19_SAFE", "1") != "0"
  def bar1_alloc(size, align=0x1000):
    nonlocal bar1_cursor
    size = round_up(size, 0x1000)
    if bit19_safe:
      if size > 0x80000:
        raise MemoryError(
            f"bit19-safe BAR1 alloc size {size:#x} exceeds 512 KiB bank")
      while True:
        bar1_cursor = round_up(bar1_cursor, align)
        bank = bar1_cursor & ~0xfffff
        lo_end = bank + 0x80000
        if bar1_cursor >= lo_end or bar1_cursor + size > lo_end:
          bar1_cursor = bank + 0x100000
          continue
        break
    else:
      bar1_cursor = round_up(bar1_cursor, align)
    pa = bar1_cursor
    bar1_cursor += size
    if bar1_cursor > dev.dev_impl.bar1_size:
      raise MemoryError("BAR1-backed instance allocation exceeds aperture")
    return pa
  def bar1_write(pa, data):
    # Un-POSTed cards can acknowledge large BAR1 writes while only applying the
    # first page.  After classic BAR1 POST we raise the chunk (KEPLER_BAR1_CHUNK,
    # default 256 KiB) to cut TinyGPU RPC count on bulk zeros/mirrors/ctx copy.
    data = memoryview(data).cast("B")
    chunk = _bar1_xfer["chunk"]
    for off in range(0, len(data), chunk):
      dev.dev_impl.hw.mmio_write(1, pa + off, data[off:off + chunk].tobytes())
  def vram_store(pa, data, *, label="vram"):
    """Store into framebuffer for GPU clients.

    H21/H22: BAR1 bulk stores can false-pass host readback while FECS/SM see
    different bits (eng-ctx hang; float mantissa flips on add).  Default
    PRAMIN for small compute mirrors; large scratch (TLS/TXC) stays BAR1
    unless KEPLER_MIRROR_COPY=pramin-all.  KEPLER_MIRROR_COPY=bar1 forces BAR1.
    """
    mode = os.environ.get("KEPLER_MIRROR_COPY", "pramin").strip().lower()
    if mode not in ("pramin", "pramin-all", "bar1"):
      raise ValueError(
          f"invalid KEPLER_MIRROR_COPY={mode!r}; "
          "expected pramin|pramin-all|bar1")
    data = memoryview(data).cast("B").tobytes()
    _pramin_max = int(os.environ.get("KEPLER_MIRROR_PRAMIN_MAX", "0x10000"), 0)
    use_pramin = (
        mode == "pramin-all" or
        (mode == "pramin" and len(data) <= _pramin_max))
    if not use_pramin:
      bar1_write(pa, data)
      return "bar1"
    _gk104_pramin_write(dev, pa, data, force_bar0=True)
    return "pramin"
  def vram_load(pa, n):
    mode = os.environ.get("KEPLER_MIRROR_COPY", "pramin").strip().lower()
    _pramin_max = int(os.environ.get("KEPLER_MIRROR_PRAMIN_MAX", "0x10000"), 0)
    use_pramin = (
        mode == "pramin-all" or
        (mode == "pramin" and n <= _pramin_max))
    if not use_pramin:
      return bytes(dev.dev_impl.hw.mmio_read(1, pa, n))
    out = bytearray(n)
    for off in range(0, n, 4):
      struct.pack_into("<I", out, off,
                      _gk104_pramin_read32(dev, pa + off, force_bar0=True))
    return bytes(out)
  _bar1_xfer = {"chunk": 0x1000}
  ctx_alias_ptes = []
  fault_guard_pte_addr = None
  fault_guard_pte_wanted = None
  bar1_identity_ready = False
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
    # PRAMIN is a 1-MiB window (0x1700 selects bits[36:16]).  Split the GR
    # context into two bit19-safe 512 KiB banks: runtime at +0 and golden
    # (CB_RESERVED) in a separate bank — a single 1 MiB object aliases on this
    # eGPU and destroys itself during zero/repair.
    grctx_vram_pa = bar1_alloc(0x80000)
    grctx_golden_vram_pa = bar1_alloc(0x80000)
    pagepool_vram_pa = bar1_alloc(0x8000)
    bundle_vram_pa = bar1_alloc(0x3000)
    attrib_vram_pa = bar1_alloc(attrib_cb.size)
    mmio_list_vram_pa = bar1_alloc(0x1000)
    print(f"[kepler] VRAM inst: grctx_runtime={grctx_vram_pa:#x} "
          f"grctx_golden={grctx_golden_vram_pa:#x} attrib={attrib_vram_pa:#x}+"
          f"{attrib_cb.size:#x} bit19_safe={bit19_safe}", flush=True)
    gpfifo_vram_pa = bar1_alloc(0x2000, align=0x2000) if use_vram_gpfifo else None
    if gpfifo_vram_pa is not None and os.environ.get("KEPLER_GPFIFO_IN_USERD") == "1":
      # Diagnostic: USERD page 0 offset 0 is unused by CHID 1 (its USERD is at
      # 0x200) and is already proven visible to PBDMA through 0x2254.
      gpfifo_vram_pa = userd_vram_pa
    push_vram_pa = bar1_alloc(push_alloc_size, align=0x10000) if use_vram_push else None
    push_in_gpfifo = (push_vram_pa is not None and gpfifo_vram_pa is not None and
                      os.environ.get("KEPLER_PUSH_IN_GPFIFO") == "1")
    if push_in_gpfifo:
      push_vram_pa = gpfifo_vram_pa
    signal_vram_pa = bar1_alloc(0x1000) if use_vram_signal else None
    ramin_vram_pa = bar1_alloc(0x1000)
    # BAR1 must be backed by its VMM before the first framebuffer store.  The
    # old ordering populated every object while 0x1704 was still disabled and
    # only created the identity mapping much later during FIFO setup.
    if os.environ.get("KEPLER_INIT_BAR1", "1") != "0":
      # USERD VA0 alias (PTE=pa>>8|1) can fail to stick PRESENT on cold PRAMIN
      # XOR aperture (Night41ao: wanted 0x4001 actual 0x4000). Default: pure
      # identity; USERD is reached as VA==PA within the mapped window.
      _alias = (None if os.environ.get("KEPLER_USERD_ALIAS", "0") == "0"
                else userd_vram_pa)
      _gk104_init_bar1_identity(dev, bus_base=base, map_vram=True,
                                userd_alias_pa=_alias)
      bar1_identity_ready = True
    # Classic BAR1 is live for bulk transfers (POST path or warm reopen).
    if (bar1_identity_ready or
        getattr(dev, "_bar1_classic_posted", False)):
      _chunk = int(os.environ.get("KEPLER_BAR1_CHUNK", "0x40000"), 0)
      if _chunk >= 0x1000 and (_chunk & 0xfff) == 0:
        _bar1_xfer["chunk"] = _chunk
    # Soft PRAMIN_live can be true from PMC_BOOT_0 alone while BAR1 still
    # returns 0xbad0**** stubs after the first store (dirty / half-POST).  Fail
    # closed before thrashing repair_zero — needs enclosure power-cycle.
    try:
      if (dev.read32(0x409800) & 0x80000000) or (dev.read32(0x409100) & 0x10):
        falcon_stop(dev, FECS_FALCON_BASE, timeout_ms=1000)
        print("[kepler] halted FECS before GR/instance zero", flush=True)
    except Exception as e:
      print(f"[kepler] FECS halt before zero skipped: {e}", flush=True)
    for pa, size in ((grctx_vram_pa, 0x80000), (grctx_golden_vram_pa, 0x80000),
                     (pagepool_vram_pa, 0x8000), (bundle_vram_pa, 0x3000),
                     (attrib_vram_pa, attrib_cb.size)):
      bar1_write(pa, bytes(size))
    try:
      _bar1_word = struct.unpack(
          "<I", bytes(dev.dev_impl.hw.mmio_read(1, grctx_vram_pa, 4)))[0]
      if _gk104_pramin_word_is_stub(_bar1_word):
        raise RuntimeError(
            f"BAR1 VRAM stub at grctx pa={grctx_vram_pa:#x} "
            f"word={_bar1_word:#x} after bulk zero: framebuffer aperture is "
            "dead/dirty. Power-cycle the eGPU enclosure (not just USB replug), "
            "restart TinyGPU server, then re-run add_660ti.py once.")
    except RuntimeError:
      raise
    except Exception as e:
      print(f"[kepler] BAR1 grctx post-zero check skipped: {e}", flush=True)
    bar1_write(userd_vram_pa, bytes(0x2000))
    _gk104_pramin_write_literal(dev, userd_vram_pa, bytes(0x2000))
    bar1_write(ramin_vram_pa, bytes(0x1000))
    _gk104_pramin_write_literal(dev, ramin_vram_pa, bytes(0x1000))
    _gk104_bar_flush(dev)
    # FECS saves only defined context words; all gaps must start at zero.
    # BAR1 bulk stores can leave inverted/stale dwords, so repair the
    # golden-context allocation before it becomes the source for every channel.
    attrib_repair_mode = os.environ.get("KEPLER_ATTRIB_REPAIR", "literal")
    if attrib_repair_mode not in ("literal", "bar1"):
      raise ValueError(
          "invalid KEPLER_ATTRIB_REPAIR; expected literal or bar1")
    # Classic+literal BAR1 (Night41ay+): bulk BAR1 zeros above, then verify
    # (rewrite only dirty chunks).  Old path: per-dword PRAMIN RMW (~35s) —
    # keep behind KEPLER_FAST_ZERO=0.  KEPLER_ZERO_CHUNK (default 256 KiB)
    # bounds BAR1 verify RPCs on TinyGPU.
    fast_zero = os.environ.get("KEPLER_FAST_ZERO", "1") != "0"
    # Prefer device flags; also accept the local bar1_identity_ready set just
    # above when classic init completed in this channel-build (warm reopen).
    classic_bar1 = bool(getattr(dev, "_bar1_classic_posted", False) or
                        getattr(dev, "_bar1_identity_ready", False) or
                        bar1_identity_ready)
    zero_chunk = int(os.environ.get("KEPLER_ZERO_CHUNK",
                                    os.environ.get("KEPLER_BAR1_CHUNK", "0x40000")), 0)
    if zero_chunk < 0x1000 or (zero_chunk & 0xfff):
      zero_chunk = 0x1000
    _zero_need_ltc = False

    def repair_zero(label, pa, size):
      # Virgin GDDR often reads as 0xffffffff.  On the XOR aperture a literal
      # BAR1 store of zeros is a no-op (old^0 == old).  After H101 classic
      # literal BAR1, bulk BAR1 zeros stick; fall back to compensated PRAMIN
      # only if a chunk still fails.  Verify-first avoids rewriting clean
      # buffers (~2–3s wall on TinyGPU for the three 512 KiB GR regions).
      phase_label = label.lower().replace(" ", "-")
      if hw is not None:
        hw.set_phase(f"channel-build-repair-zero-{phase_label}")
      use_bar1 = (attrib_repair_mode == "bar1" and label == "attrib")
      zero_page = bytes(0x1000)
      t0 = time.monotonic()
      rewrote = 0
      step = zero_chunk if (fast_zero and classic_bar1) else 0x1000
      verify_mode = os.environ.get("KEPLER_ZERO_VERIFY", "sample")
      nonlocal _zero_need_ltc
      # Sample mode: trust the bulk BAR1 zero unless a few probes fail, then
      # fall through to the full rewrite path.  Avoids re-reading ~1.5 MiB.
      if fast_zero and classic_bar1 and verify_mode != "full":
        _probes = sorted({0, size // 2, max(0, size - 4),
                          0x1000 if size > 0x1000 else 0,
                          0x10000 if size > 0x10000 else 0})
        _dirty = False
        for _poff in _probes:
          if _poff + 4 > size:
            continue
          got = bytes(dev.dev_impl.hw.mmio_read(1, pa + _poff, 4))
          if got != b"\x00\x00\x00\x00":
            _dirty = True
            break
        if not _dirty:
          sample = dev.dev_impl.hw.mmio_read(1, pa, min(size, 0x20)).hex()
          print(f"[kepler] {label} physical-zero verified; "
                f"BAR1 sample={sample} "
                f"({time.monotonic() - t0:.2f}s fast_zero={fast_zero} "
                f"rewrote=0 chunk={step:#x} verify=sample)",
                flush=True)
          return
      for _page in range(0, size, step):
        _page_size = min(step, size - _page)
        _want = (zero_page if _page_size == 0x1000 else bytes(_page_size))
        if fast_zero and classic_bar1:
          got = bytes(dev.dev_impl.hw.mmio_read(1, pa + _page, _page_size))
          if got == _want:
            continue
          rewrote += 1
          for _attempt in range(6):
            bar1_write(pa + _page, _want)
            _gk104_bar_flush(dev)
            got = bytes(dev.dev_impl.hw.mmio_read(1, pa + _page, _page_size))
            if got == _want:
              break
            # Compensated PRAMIN in 4 KiB slices, then re-check via BAR1.
            for _sub in range(0, _page_size, 0x1000):
              _sz = min(0x1000, _page_size - _sub)
              _gk104_pramin_write(dev, pa + _page + _sub, bytes(_sz))
            _gk104_bar_flush(dev)
            got = bytes(dev.dev_impl.hw.mmio_read(1, pa + _page, _page_size))
            if got == _want:
              break
          else:
            _page_bad = [i for i in range(0, _page_size, 4)
                         if got[i:i + 4] != b"\x00\x00\x00\x00"]
            _sample = [struct.unpack_from("<I", got, x)[0]
                       for x in _page_bad[:4]]
            raise RuntimeError(
                f"GK104 {label} chunk {_page:#x} did not stabilise: "
                f"offsets={[hex(_page + x) for x in _page_bad[:16]]} "
                f"samples={[hex(x) for x in _sample]}")
          continue
        for _attempt in range(6):
          if use_bar1 and _attempt < 2:
            bar1_write(pa + _page, _want)
          _gk104_pramin_write(dev, pa + _page, _want)
          _gk104_bar_flush(dev)
          time.sleep(0.002)
          _page_bad = [_off for _off in range(0, _page_size, 4)
                       if _gk104_pramin_read32(dev, pa + _page + _off)]
          if not _page_bad:
            break
        else:
          _page_bad = [_off for _off in range(0, _page_size, 4)
                       if _gk104_pramin_read32(dev, pa + _page + _off)]
          _sample = [_gk104_pramin_read32(dev, pa + _page + _off) & 0xffffffff
                     for _off in _page_bad[:4]]
          raise RuntimeError(
              f"GK104 {label} page {_page:#x} did not stabilise: "
              f"offsets={[hex(_page + x) for x in _page_bad[:16]]} "
              f"samples={[hex(x) for x in _sample]}")
        rewrote += 1
      if rewrote or not (fast_zero and classic_bar1):
        _zero_need_ltc = True
      # Full-buffer path only when sample found dirt or verify=full.
      if fast_zero and classic_bar1 and rewrote == 0:
        sample = dev.dev_impl.hw.mmio_read(1, pa, min(size, 0x20)).hex()
        print(f"[kepler] {label} physical-zero verified; "
              f"BAR1 sample={sample} "
              f"({time.monotonic() - t0:.2f}s fast_zero={fast_zero} "
              f"rewrote=0 chunk={step:#x} verify=full)",
              flush=True)
        return
      _gk104_bar_flush(dev)
      if fast_zero and classic_bar1:
        _bad = []
        for _page in range(0, size, step):
          _page_size = min(step, size - _page)
          got = bytes(dev.dev_impl.hw.mmio_read(1, pa + _page, _page_size))
          _bad.extend(_page + i for i in range(0, _page_size, 4)
                      if got[i:i + 4] != b"\x00\x00\x00\x00")
        if _bad:
          for _page in range(0, size, 0x1000):
            _gk104_pramin_write(dev, pa + _page,
                                bytes(min(0x1000, size - _page)))
          _gk104_bar_flush(dev)
          _bad = []
          for _page in range(0, size, step):
            _page_size = min(step, size - _page)
            got = bytes(dev.dev_impl.hw.mmio_read(1, pa + _page, _page_size))
            _bad.extend(_page + i for i in range(0, _page_size, 4)
                        if got[i:i + 4] != b"\x00\x00\x00\x00")
      else:
        _bad = [_off for _off in range(0, size, 4)
                if _gk104_pramin_read32(dev, pa + _off)]
        if _bad:
          for _page in range(0, size, 0x1000):
            _gk104_pramin_write(dev, pa + _page,
                                bytes(min(0x1000, size - _page)))
          _gk104_bar_flush(dev)
          _bad = [_off for _off in range(0, size, 4)
                  if _gk104_pramin_read32(dev, pa + _off)]
      if _bad:
        if fast_zero and classic_bar1:
          _sample = [
              struct.unpack("<I",
                            bytes(dev.dev_impl.hw.mmio_read(1, pa + _off, 4)))[0]
              for _off in _bad[:4]]
        else:
          _sample = [_gk104_pramin_read32(dev, pa + _off) & 0xffffffff
                     for _off in _bad[:4]]
        raise RuntimeError(
            f"GK104 {label} final zero proof failed: "
            f"nonzero={len(_bad)} offsets={[hex(x) for x in _bad[:16]]} "
            f"samples={[hex(x) for x in _sample]}")
      sample = dev.dev_impl.hw.mmio_read(1, pa, min(size, 0x20)).hex()
      print(f"[kepler] {label} physical-zero verified; "
            f"BAR1 sample={sample} "
            f"({time.monotonic() - t0:.2f}s fast_zero={fast_zero} "
            f"rewrote={rewrote} chunk={step:#x})",
            flush=True)

    repair_zero("GR ctx runtime", grctx_vram_pa, 0x80000)
    repair_zero("GR ctx golden", grctx_golden_vram_pa, 0x80000)
    repair_zero("pagepool", pagepool_vram_pa, 0x8000)
    repair_zero("bundle", bundle_vram_pa, 0x3000)
    repair_zero("attrib", attrib_vram_pa, attrib_cb.size)
    # One LTC flush after any rewrite (not once per buffer).
    if _zero_need_ltc:
      try:
        dev.write32(0x070010, 0x00000001)
        _deadline = time.time() + 0.2
        while dev.read32(0x070010) & 0x00000003:
          if time.time() >= _deadline:
            break
          time.sleep(0.001)
      except Exception:
        pass
      time.sleep(0.005)
    # ctx_chan starts at CB_RESERVED (separate golden bank).  Prove that first
    # context-header page remains zero after LTC settling.
    _ctx_header_pa = grctx_golden_vram_pa
    _zero_page = bytes(0x1000)
    for _attempt in range(4):
      _gk104_pramin_write(dev, _ctx_header_pa, _zero_page)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for context header failed")
      time.sleep(0.005)
      _ctx_header_actual = b"".join(
          struct.pack("<I", _gk104_pramin_read32(dev, _ctx_header_pa + off))
          for off in range(0, 0x1000, 4))
      if _ctx_header_actual == _zero_page:
        break
    else:
      _bad = [off for off in range(0, 0x1000, 4)
              if _ctx_header_actual[off:off + 4] != b"\x00\x00\x00\x00"]
      raise RuntimeError(
          f"GK104 context header did not stabilise: offsets={_bad[:16]}")
    if gpfifo_vram_pa is not None:
      bar1_write(gpfifo_vram_pa, bytes(0x2000))
    if push_vram_pa is not None:
      bar1_write(push_vram_pa, bytes(push_alloc_size))
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
    # VA is still contiguous CB_RESERVED+size; PA splits across two lo512 banks.
    dev.dev_impl.mm.map_range(gr_ctx.va_addr, gr_ctx.size,
                              [(grctx_vram_pa, 0x80000),
                               (grctx_golden_vram_pa, 0x80000)],
                              AddrSpace.PHYS)
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
    dev.dev_impl.mm.map_range(attrib_cb.va_addr, attrib_cb.size,
                              [(attrib_vram_pa, attrib_cb.size)], AddrSpace.PHYS)
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
  # Kepler's main GPFIFO sits immediately after the push buffer. Mirror the
  # Nouveau virtual layout even though the two backing allocations are local.
  # Keep the FIFO beside (but non-overlapping with) the proven GR-context range
  # in PGD slot 1.  FECS already validates this PGD/SPT path on silicon.
  push_va = int(os.environ.get("KEPLER_PUSH_VA", "0x09000000"), 0)
  gpfifo_va = push_va + push.size
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
  # BAR1-backed region for the hardware path; the sysmem allocation is
  # retained only as the software/offline backing object.
  # NOTE: The _int path (!gr->firmware) does NOT write data[0x1c/0x20/0x28/0x2c]
  # — those writes are _ext path only (ctxgf100.c lines 1499-1504).  The context
  # buffer starts zeroed, and grctx->main() populates the GR registers before
  # the context unload saves the golden image.
  if use_vram_inst:
    ctx_header = grctx_golden_vram_pa
  else:
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
    if hw is not None:
      hw.set_phase("golden-context-vmm-clone")
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
    # VA 0x67000 sits inside large out[] mirrors (e.g. N>=24576 with out at
    # 0x50000).  Guards are fill-ins for holes only — live_pte_map must apply
    # them *before* mirrors so compute buffers keep their real PTEs.
    ctx_alias_ptes.append(
        (cloned_by_pgd[0] + 0x67 * 8, (_alias_cursor >> 8) | 1))
    _alias_cursor += 0x1000
    # SET_OBJECT's first automatic firmware context load performs a selected
    # GPC read at VA 0x100000 on this GK104.  Nouveau's fully populated channel
    # VM satisfies it; our sparse userspace VM did not.  Back only that observed
    # leaf with a dedicated writable guard page, preserving faults everywhere
    # else instead of adding a broad identity/aperture mapping.  temp_dev also
    # lives at 0x100000; mirrors must win over this guard in live_pte_map.
    fault_guard_pte_addr = cloned_by_pgd[0] + 0x100 * 8
    # priv=1: SET_OBJECT firmware/GPC walk matches eng-ctx mapping style.
    fault_guard_pte_wanted = (_alias_cursor >> 8) | 3
    ctx_alias_ptes.append(
        (fault_guard_pte_addr, fault_guard_pte_wanted))
    _alias_cursor += 0x1000
    for mirror in getattr(dev, "_kepler_vram_mirrors", ()):
      mirror_size = round_up(mirror.size, 0x1000)
      # Bit19-safe banks are 512 KiB; mirrors (e.g. 1 MiB mmio list) must be
      # striped across banks.  Map still uses contiguous VA; PA need not be.
      chunks = []
      rem = mirror_size
      while rem:
        chunk = min(rem, 0x80000 if bit19_safe else rem)
        chunks.append((bar1_alloc(chunk), chunk))
        rem -= chunk
      src_pa = mirror.meta["pa"]
      image = bytes(vram[src_pa:src_pa + mirror.size])
      image = image + bytes(mirror_size - len(image))
      off = 0
      _store_modes = set()
      for chunk_pa, chunk_sz in chunks:
        _store_modes.add(vram_store(chunk_pa, image[off:off + chunk_sz],
                                    label=f"mirror@{mirror.va_addr:#x}"))
        off += chunk_sz
      page = 0
      for chunk_pa, chunk_sz in chunks:
        for local in range(0, chunk_sz, 0x1000):
          va = mirror.va_addr + page * 0x1000
          pgdi, spti = (va >> 27) & 0x1fff, (va >> 12) & 0x7fff
          pte = ((chunk_pa + local) >> 8) | 1
          if mirror.meta.get("priv"):
            pte |= 2
          _gk104_pramin_write(dev, cloned_by_pgd[pgdi] + spti * 8,
                              struct.pack("<Q", pte))
          page += 1
      mirror_pa = chunks[0][0]
      mirror.meta["vram_pa"] = mirror_pa
      mirror.meta["vram_chunks"] = chunks
      print(f"[kepler] VRAM mirror: va={mirror.va_addr:#x} pa={mirror_pa:#x} "
            f"size={mirror.size:#x} chunks={len(chunks)} "
            f"via={'+'.join(sorted(_store_modes))}", flush=True)
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
    # Runtime (+0) and golden (+0x80000) land in separate bit19-safe PA banks.
    for page in range(gr_ctx.size // 0x1000):
      va = gr_ctx.va_addr + page * 0x1000
      pgdi, spti = (va >> 27) & 0x1fff, (va >> 12) & 0x7fff
      off = page * 0x1000
      pa = (grctx_vram_pa + off) if off < 0x80000 else (
          grctx_golden_vram_pa + (off - 0x80000))
      pte = (pa >> 8) | 3
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
    # Stash for post-launch channel PTE dumps (SM vs host aperture checks).
    dev._kepler_vmm_pgd_pa = vmm_pgd_pa
    dev._kepler_cloned_by_pgd = dict(cloned_by_pgd)
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
  # ctxgf100.c eventually points RAMIN at ctx->addr + CB_RESERVED (0x80000),
  # with bit 2 marking the engine-context pointer valid.  The golden trace
  # keeps this field zero through channel bind/runlist admission and installs
  # it only immediately before FECS ctx_chan.
  grctx_va = (gr_ctx.va_addr + 0x80000) | 4
  struct.pack_into("<Q", inst, 0x0210, 0)
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
      _vram_spt_pa = cloned_by_pgd[_pgd_idx]
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
  push_addr = push.va_addr + push_phys_offset
  # This is the channel's main push buffer, matching nvif_chan_gpfifo_push_kick
  # (main=true), so BIT(9) remains clear. The env override is a diagnostic for
  # the alternate external-entry encoding used by nvif_chan_gpfifo_push().
  gpfifo_external = os.environ.get("KEPLER_GPFIFO_EXTERNAL") == "1"
  gpfifo_no_prefetch = os.environ.get("KEPLER_GPFIFO_NO_PREFETCH") == "1"
  ring = bytearray(n_gp_entries * GPFIFO_ENTRY_BYTES)
  _off_words = 0
  for _ei, _batch in enumerate(ib_batches):
    _entry_addr = push_addr + _off_words * 4
    struct.pack_into(
        "<II", ring, _ei * GPFIFO_ENTRY_BYTES,
        _entry_addr & 0xffffffff,
        (_entry_addr >> 32) | ((1 << 9) if gpfifo_external else 0) |
        (len(_batch) << 10) | ((1 << 31) if gpfifo_no_prefetch else 0))
    _off_words += len(_batch)
  ring_store = bytearray(ring)
  if os.environ.get("KEPLER_GPU_XOR_RING") == "1":
    struct.pack_into("<I", ring_store, 0,
                     (~struct.unpack_from("<I", ring, 0)[0]) & 0xffffffff)
  if n_gp_entries > 1:
    print(f"[kepler] multi-IB: entries={n_gp_entries} words={_off_words} "
          f"push_size={push_alloc_size:#x}", flush=True)
  vram[gpfifo_pa:gpfifo_pa + len(ring)] = ring
  if gpfifo_vram_pa is not None:
    ring_page = bytearray(0x1000)
    ring_page[:len(ring)] = ring
    bar1_write(gpfifo_vram_pa, ring_page)
    if push_vram_pa == gpfifo_vram_pa and push_phys_offset:
      bar1_write(push_vram_pa + push_phys_offset, push_bytes)
    _gk104_bar_flush(dev)
    _ring_ok = True
    if push_vram_pa == gpfifo_vram_pa and push_phys_offset:
      _got_push = bytes(dev.dev_impl.hw.mmio_read(
          1, push_vram_pa + push_phys_offset, len(push_bytes)))
      _ring_ok = (_got_push == bytes(push_bytes) and
                  bytes(dev.dev_impl.hw.mmio_read(1, gpfifo_vram_pa, len(ring)))
                  == bytes(ring))
    else:
      _ring_ok = (bytes(dev.dev_impl.hw.mmio_read(1, gpfifo_vram_pa, len(ring)))
                  == bytes(ring))
    if not _ring_ok:
      _gk104_pramin_write(dev, gpfifo_vram_pa, ring)
      if push_vram_pa == gpfifo_vram_pa and push_phys_offset:
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

  # Both traced PFIFO resets are applied during LinuxPCIDevice.init_hw().  Do
  # not reset PFIFO here after FECS has booted: that destroys scheduler state
  # without resetting GR's current context.

  # The capture retains Nouveau's persistent kernel channel 0, but an opt-in
  # FIFO reset removes it.  Preserve channel 0 only when its post-reset bind
  # state says that it is still resident.
  _kernel_chan0_bound = bool(
      dev.read32(CHAN_SUBMIT_REG) & 0x80000000) if dev.dev_impl.hw is not None else False
  runlist_chids = ([0] if _kernel_chan0_bound and chan_id != 0 else []) + [chan_id]
  runlist_words = tuple(word for chid in runlist_chids for word in (chid, 0))
  runlist_resident_count = len(runlist_chids) - 1
  runlist_active_count = len(runlist_chids)
  runlist_pa = runlist_vram_pa if use_vram_runlist else runlist.meta['pa']
  if use_vram_runlist:
    runlist_entry = struct.pack("<" + "I" * len(runlist_words), *runlist_words)
    # The first commit consumes this entry immediately.  A single PRAMIN store
    # is insufficient on this card (writes can settle to the complemented
    # value), so use the same flush/invalidate/readback proof as the runtime
    # recommit path before exposing its address to scheduler DMA.
    for _attempt in range(4):
      bar1_write(runlist_pa, runlist_entry)
      _gk104_pramin_write(dev, runlist_pa, runlist_entry)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for initial runlist failed")
      time.sleep(0.005)
      _initial_runlist = tuple(
          _gk104_pramin_read32(dev, runlist_pa + off)
          for off in range(0, len(runlist_entry), 4))
      if _initial_runlist == runlist_words:
        break
    else:
      raise RuntimeError(
          f"GK104 initial runlist store did not stabilise: {_initial_runlist}")
  else:
    struct.pack_into("<" + "I" * len(runlist_words), vram, runlist_pa,
                     *runlist_words)

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
    if (use_vram_inst and not bar1_identity_ready and
        os.environ.get("KEPLER_INIT_BAR1", "1") != "0"):
      # Nouveau maps USERD into a BAR1 VMA (allocator-chosen VA), then programs
      # 0x2254 with that VA.  Optional KEPLER_USERD_ALIAS=1 keeps the old VA0
      # shortcut; default uses identity VA==PA (Nouveau-shaped 0x2254).
      _alias = (userd_vram_pa
                if os.environ.get("KEPLER_USERD_ALIAS", "0") == "1"
                else None)
      _gk104_init_bar1_identity(dev, bus_base=base, map_vram=True,
                                userd_alias_pa=_alias)
      userd_bar1_base = 0 if _alias is not None else userd_vram_pa
      bar1_identity_ready = True
    elif os.environ.get("KEPLER_USERD_BAR1_ZERO") == "1":
      userd_bar1_base = 0
    elif bar1_identity_ready and os.environ.get("KEPLER_USERD_ALIAS", "0") != "1":
      # Identity BAR1: PFIFO polls USERD at VA==PA.
      userd_bar1_base = userd_vram_pa if use_vram_inst else 0
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
      dev.write32(q + 0x13c, dev.read32(q + 0x13c) & ~0x10000100)
      dev.write32(q + 0x108, 0xffffffff)
      dev.write32(q + 0x10c, 0xfffffeff)
      dev.write32(q + 0x148, 0xffffffff)
      dev.write32(q + 0x14c, 0xffffffff)
    dev.write32(0x2100, 0xffffffff)
    dev.write32(0x2140, 0x7fffffff)
    nvkm_mask(dev, 0x000640, 0x00000100, 0x00000100)
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
    # A previous failed process can leave this CHID bound and present in the
    # persistent hardware runlist.  Follow nouveau's traced destruction order:
    # stop/preempt it, publish an empty replacement list, then unbind.  Merely
    # writing STOP followed immediately by INST=0 raises BIND_NOT_UNBOUND.
    _old_inst = dev.read32(_chan_inst_reg)
    _old_ctrl = dev.read32(_chan_ctrl_reg)
    _stopped_ctrl = _old_ctrl
    _unbound, _unbound_ctrl = _old_inst, _old_ctrl
    # An unbound CHID must go directly to bind.  Sending CHAN_INST=0 to a
    # clean slot leaves an asynchronous bind-state transaction that races the
    # immediately following bind.  Only recover a slot that advertises a live
    # instance owner.
    if ((_old_inst & 0x80000000) or
        (_old_ctrl & 0x11000000) == 0x11000000):
      # The trace has two preempts here.  The first belongs to engine-context
      # detachment: stop/kick, clear RAMIN's GR pointer, then allow the channel.
      # Channel destruction subsequently stops/kicks it again, removes it from
      # the runlist, and only then writes CHAN_INST=0.
      _gk104_stop_preempt_channel(dev, chan_id,
                                  label="stale-context detach preempt")
      if use_vram_inst:
        _gk104_pramin_write(dev, ramin_vram_pa + 0x210, bytes(8))
        _gk104_bar_flush(dev)
      else:
        struct.pack_into("<Q", vram, ramin_pa + 0x210, 0)
      nvkm_mask(dev, _chan_ctrl_reg, 0x00000400, 0x00000400)
      _gk104_stop_preempt_channel(dev, chan_id,
                                  label="stale-channel removal preempt")
      _stopped_ctrl = dev.read32(_chan_ctrl_reg)
      _gk104_commit_runlist(dev, _rl_addr, runlist_resident_count,
                            target=_rl_target)
      dev.write32(_chan_inst_reg, 0)
      _unbind_deadline = time.time() + 0.2
      while True:
        _unbound = dev.read32(_chan_inst_reg)
        _unbound_ctrl = dev.read32(_chan_ctrl_reg)
        if _unbound == 0 and _unbound_ctrl == 0:
          break
        if time.time() >= _unbind_deadline:
          raise TimeoutError(
              f"GK104 channel {chan_id} did not finish unbinding: "
              f"INST={_unbound:#x} CTRL={_unbound_ctrl:#x}")
        time.sleep(0.001)
    # Discard a stale bind interrupt before attempting the new bind; any bind
    # interrupt observed afterwards belongs to the new operation.
    _stale_bind_intr = dev.read32(0x2100)
    _stale_bind_err = dev.read32(0x252c) if _stale_bind_intr & 0x1 else 0
    if _stale_bind_intr & 0x1:
      dev.write32(0x2100, 0x1)
    # The detach phase deliberately zeroed the physical context pointer.  With
    # no stale owner left, publish the complete new RAMFC before rebinding it.
    if use_vram_inst:
      _ramfc_expected = bytes(inst[:0x220])
      for _attempt in range(4):
        _gk104_pramin_write(dev, ramin_vram_pa, _ramfc_expected)
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate for pre-bind RAMFC failed")
        time.sleep(0.005)
        _ramfc_actual = b"".join(
            struct.pack("<I", _gk104_pramin_read32(dev, ramin_vram_pa + off))
            for off in range(0, len(_ramfc_expected), 4))
        if _ramfc_actual == _ramfc_expected:
          break
      else:
        _bad = [off for off in range(0, len(_ramfc_expected), 4)
                if _ramfc_actual[off:off + 4] !=
                _ramfc_expected[off:off + 4]]
        raise RuntimeError(
            f"GK104 pre-bind RAMFC did not stabilise: offsets={_bad[:16]}")
    else:
      vram[ramin_pa:ramin_pa + 0x220] = inst[:0x220]
    # gk104_chan_bind() reaches a freshly allocated channel whose complete
    # control word is zero.  The golden trace therefore writes CHAN_CTRL=0
    # immediately before CHAN_INST.  A masked update preserves hardware-owned
    # 0x11000000 state and the following bind fails BIND_NOT_UNBOUND.
    dev.write32(_chan_ctrl_reg, GR_RUNLIST_ID << 16)
    # Clear USERD GP_GET/GP_PUT and other pointers
    _userd_phys_base = ((userd_vram_pa + userd_base_off) if use_vram_inst else
                        userd_base_off)
    # VA0 USERD alias → BAR1 offset is just the CHID USERD slice.  Identity
    # BAR1 (default) maps USERD at VA==PA, matching Nouveau's 0x2254 VMA base.
    if (use_vram_inst and os.environ.get("KEPLER_INIT_BAR1", "1") != "0" and
        os.environ.get("KEPLER_USERD_ALIAS", "0") == "1"):
      _userd_mmio_base = userd_base_off
    else:
      _userd_mmio_base = _userd_phys_base
    for _uo in (0x40, 0x44, 0x48, 0x4c, 0x50, 0x58, 0x5c, 0x60,
                USERD_GP_GET, USERD_GP_PUT):
      dev.dev_impl.hw.mmio_write(1, _userd_mmio_base + _uo, struct.pack("<I", 0))
      if use_vram_inst:
        _gk104_pramin_write(dev, _userd_phys_base + _uo, struct.pack("<I", 0))
    # Write inst pointer (triggers PBDMA bind)
    _bind_addr = ramin_vram_pa if use_vram_inst else (base + ramin_pa)
    # CTRL/USERD preparation can complete asynchronously and report an older
    # bind-state transition.  Establish the attribution boundary immediately
    # before CHAN_INST so the interrupt sampled below belongs to this bind.
    _preinst_intr = dev.read32(0x2100)
    _preinst_bind_err = dev.read32(0x252c) if _preinst_intr & 0x1 else 0
    if _preinst_intr & 0x1:
      dev.write32(0x2100, 0x1)
    _wanted_bound = 0x80000000 | (_bind_addr >> 12)
    dev.write32(_chan_inst_reg, _wanted_bound)
    _bound = dev.read32(_chan_inst_reg)
    _bound_ctrl = dev.read32(_chan_ctrl_reg)
    # Check BIND_ERROR immediately (no sleep — sleeping lets interrupts fire
    # and the FECS jumps to the overlay).  A failed bind is terminal: allowing
    # FECS or PBDMA to consume this channel would turn a lifecycle error into
    # an invalid DMA/context operation.
    _bind_intr = dev.read32(0x2100)
    if _bind_intr & 0x1:
      _bind_err_code = dev.read32(0x252c)
      dev.write32(0x2100, 0x1)  # clear BIND_ERROR
      # Live tests prove that INST latching and CTRL gaining 0x11000000 do not
      # make BIND_NOT_UNBOUND recoverable: every such admission deadlocks the
      # following runlist with SCHED_ERROR 0x0d.  Any bind interrupt is fatal.
      if _bind_intr & 0x1:
        raise RuntimeError(
            f"GK104 channel bind failed: INTR={_bind_intr:#x} "
            f"BIND_ERROR={_bind_err_code:#x} bind_addr={_bind_addr:#x} "
            f"bound={_bound:#x}/{_bound_ctrl:#x} old_inst={_old_inst:#x} "
            f"old_ctrl={_old_ctrl:#x} stopped_ctrl={_stopped_ctrl:#x} "
            f"unbound={_unbound:#x}/{_unbound_ctrl:#x} "
            f"prebind_intr={_stale_bind_intr:#x}/err={_stale_bind_err:#x} "
            f"preinst_intr={_preinst_intr:#x}/err={_preinst_bind_err:#x}")
    if _bound != _wanted_bound:
      raise RuntimeError(
          f"GK104 channel bind did not stick: {_bound:#x} != {_wanted_bound:#x}")
    # Golden nouveau ordering is bind -> START(0x400) -> runlist commit.  GP_PUT
    # is still zero, so admitting the channel cannot consume unfinished work.
    nvkm_mask(dev, _chan_ctrl_reg, 0x00000400, 0x00000400)
    _precommit_sched = _gk104_clear_sched_error(dev)
    if _precommit_sched:
      print(f"[kepler] cleared precommit SCHED_ERROR={_precommit_sched:#x}",
            flush=True)
    if hw is not None:
      hw.set_phase("runlist-submit-golden-channel")
    _gk104_commit_runlist(dev, _rl_addr, runlist_active_count,
                          target=_rl_target)
    if hw is not None:
      hw.set_phase("golden-context")
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
    # Match gf100_grctx_generate() and the golden trace: return FE power to
    # AUTO after the FECS-domain reset and wait for the request bit to clear.
    # Leaving FORCE_ON asserted deadlocks the firmware's own MMIO read of
    # 0x404170 during ctx_chan on this GK104.
    dev.write32(0x404170, 0x00000010)
    wait_cond(lambda: not bool(dev.read32(0x404170) & 0x00000010),
              timeout_ms=2000, msg="FE power AUTO")
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
    # ponytail: Disable all FECS interrupts except FIFO (bit 2) before
    # ctx_chan.  The EFI overlay has interrupt vectors pointing to overlay
    # code.  If a non-FIFO interrupt fires during ctx_chan processing, the
    # FECS jumps to the overlay and gets stuck.  The FIFO interrupt (bit 2)
    # is needed for the FECS to process the FIFO command.  Re-enable before
    # CHSW — and restore the full post-firmware mask after golden save so
    # SET_OBJECT context-load can see engine-specific lines (bits 8-15).
    _fecs_irqmask_before_ctx_chan = dev.read32(0x409018)
    print(f"[kepler] FECS IRQMASK before ctx_chan CLR: {_fecs_irqmask_before_ctx_chan:#x}",
          flush=True)
    dev.write32(FECS_FALCON_BASE + 0x004, 0xfffffffb)  # INTR_CLR: clear all except FIFO
    dev.write32(FECS_FALCON_BASE + 0x014, 0xfffffffb)  # IRQMCLR: disable all except bit 2
    # Prove the exact PDE/PTEs used by FECS and GR at the point of use.
    # Early readback is insufficient on this card: framebuffer dwords can
    # settle to a different value after several milliseconds, which surfaces
    # as CTXCTL INVALID_STORAGE_TYPE rather than PAGE_NOT_PRESENT.
    if use_vram_inst:
      _ctx_va = gr_ctx.va_addr + 0x80000
      _ctx_pgdi = (_ctx_va >> 27) & 0x1fff
      _ctx_spti = (_ctx_va >> 12) & 0x7fff
      _ctx_spt_pa = cloned_by_pgd[_ctx_pgdi]
      _ctx_critical = []
      _critical_pgdis = {_ctx_pgdi}
      for _scratch, _scratch_pa, _privileged in (
          (pagepool, pagepool_vram_pa, True),
          (bundle_cb, bundle_vram_pa, True),
          (attrib_cb, attrib_vram_pa, False)):
        for _page in range(round_up(_scratch.size, 0x1000) // 0x1000):
          _va = _scratch.va_addr + _page * 0x1000
          _pgdi = (_va >> 27) & 0x1fff
          _spti = (_va >> 12) & 0x7fff
          _critical_pgdis.add(_pgdi)
          _pte = ((_scratch_pa + _page * 0x1000) >> 8) | 1
          if _privileged:
            _pte |= 2
          _ctx_critical.append(
              (cloned_by_pgd[_pgdi] + _spti * 8, _pte))
      # Page-directory entries precede leaves, matching the hardware walk.
      _ctx_critical[:0] = [
          (vmm_pgd_pa + _pgdi * 8,
           (1 << 32) | (cloned_by_pgd[_pgdi] << 24))
          for _pgdi in sorted(_critical_pgdis)]
      _ctx_critical.append(
          (_ctx_spt_pa + _ctx_spti * 8,
           ((grctx_golden_vram_pa) >> 8) | 3))
      for _attempt in range(4):
        for _addr, _wanted in _ctx_critical:
          _gk104_pramin_write(dev, _addr, struct.pack("<Q", _wanted))
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate for context VMM failed")
        time.sleep(0.005)
        _ctx_actual = tuple(
            _gk104_pramin_read32(dev, _addr) |
            (_gk104_pramin_read32(dev, _addr + 4) << 32)
            for _addr, _wanted in _ctx_critical)
        if _ctx_actual == tuple(_wanted for _addr, _wanted in _ctx_critical):
          break
      else:
        raise RuntimeError(
            f"GK104 GR scratch VMM did not stabilise: "
            f"mismatches={sum(a != w for a, (_p, w) in zip(_ctx_actual, _ctx_critical))}")
      # gf100_grctx_generate() invalidates the channel VMM immediately before
      # publishing RAMIN[0x210].
      if not _gk104_vmm_flush_pdb(dev, vmm_pgd_pa, target=0):
        raise TimeoutError("GK104 channel VMM invalidate before ctx_chan failed")
    _gk104_pramin_write(dev, ramin_vram_pa + 0x210,
                        struct.pack("<Q", grctx_va))
    _gk104_bar_flush(dev)
    _installed_grctx = (
        _gk104_pramin_read32(dev, ramin_vram_pa + 0x210) |
        (_gk104_pramin_read32(dev, ramin_vram_pa + 0x214) << 32))
    if _installed_grctx != grctx_va:
      raise RuntimeError(
          f"GK104 GR context pointer did not stick: "
          f"{_installed_grctx:#x} != {grctx_va:#x}")
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
      _fifo_intr = dev.read32(0x2100)
      _fault_source = dev.read32(0x259c)
      print(f"[kepler] FECS timeout PFIFO_INTR=0x{_fifo_intr:08x} "
            f"FAULT_SOURCE=0x{_fault_source:08x}", flush=True)
      for _unit in range(16):
        if not (_fault_source & (1 << _unit)):
          continue
        _fr = 0x2800 + _unit * 0x10
        _fi = dev.read32(_fr)
        _ft = dev.read32(_fr + 0x0c)
        print(f"[kepler] FECS timeout MMU[{_unit}]: "
              f"INST=0x{_fi:08x} VA=0x{dev.read32(_fr + 4):08x}/"
              f"0x{dev.read32(_fr + 8):08x} TYPE=0x{_ft:08x}", flush=True)
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
    # Keep FE power in AUTO while grctx_main and the golden save run.  This is
    # the state Nouveau leaves after gf100_gr_fecs_reset(), and the OpenCL
    # golden trace performs the entire ICMD/method sequence without host-side
    # 0x404170 traffic.  Concurrent FORCE_ON writes can interleave with ICMD
    # MMIO and were based on observations made before BAR0 accesses were fixed
    # to native-width transactions.
    _ka_stop2 = threading.Event()
    _ka_thread2 = None
    # 3. Populate GR context via grctx->main() (gf100_grctx_generate_main).
    #    This writes the GR register init lists, pagepool/bundle/attrib_cb
    #    addresses, icmd and mthd bundles.  The channel is current so GR
    #    registers are directly accessible.
    # Read GPC/TPC topology (gf100.c gr_0x409604):
    #   gpc_nr = 0x409604 & 0x1f
    #   tpc_nr[i] = GPC_UNIT(i, 0x2608) = 0x500000 + i*0x8000 + 0x2608
    #   tpc_total = sum of tpc_nr[i]
    gpc_nr = dev.read32(0x409604) & 0x1f
    tpc_nr = _gk104_read_tpc_nr(dev, gpc_nr)
    tpc_total = sum(tpc_nr)
    print(f"[kepler] grctx_main: gpc_nr={gpc_nr} tpc_total={tpc_total} "
          f"tpc_nr={tpc_nr} "
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
    if hw is not None:
      hw.set_phase("golden-context-save")
    # ponytail: The FECS sleeps after ctx_chan and won't wake for FIFO commands
    # because the EFI firmware doesn't enable FIFO interrupts (IREN=0x0).
    # Enable the FIFO interrupt (bit 2) in INTR_EN_SET before sending the
    # CHSW trigger, so the FECS wakes up to process the context switch.
    _iren_pre = dev.read32(0x409010)
    print(f"[kepler] golden save: IREN={_iren_pre:#x} PC={dev.read32(0x409ff0):#x} "
          f"CPUCTL={dev.read32(0x409100):#x} CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x}", flush=True)
    # Enable FIFO interrupt (bit 2) + ctxsw (bit 5) + HUB_CHSW_PULSE (bit 8)
    # to wake FECS from sleep.  ponytail: bit 8 must be re-enabled here
    # because we disabled it (via IRQMCLR) before ctx_chan to prevent
    # overlay jumps.  Without bit 8, the CHSW pulse is never seen by FECS.
    dev.write32(0x409010, 0x00000124)  # INTR_EN_SET: bits 2,5,8
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
      saved_head = [_gk104_pramin_read32(dev, grctx_golden_vram_pa + i)
                    for i in range(0, 16, 4)]
    else:
      saved_head = list(struct.unpack_from("<4I", vram, grctx_pa + 0x80000))
    print(f"[kepler] FECS golden save done: CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x} "
          f"IRQMASK={dev.read32(0x409018):#x} head={[hex(x) for x in saved_head]}", flush=True)
    # Restore FECS interrupt enables wiped before ctx_chan.  Only bits 2/5/8
    # (0x124) were re-enabled for golden CHSW; that left IRQMASK=0x124 and
    # dropped engine-specific lines from the post-start mask (observed
    # 0x8704: bits 9/10/15).  Without those, SET_OBJECT's eng-ctx load
    # parks FECS (CPUCTL=SLEEPING@0x567) with GPC1+2 sticky-busy and
    # GPC TPC_NR cleared on the hung GPCs.
    _fecs_irq_restore = (_fecs_irqmask_before_ctx_chan | 0x00000124) & 0xffff
    if _fecs_irq_restore:
      dev.write32(0x409010, _fecs_irq_restore)  # INTR_EN_SET
    print(f"[kepler] FECS IRQMASK restored after golden: "
          f"want={_fecs_irq_restore:#x} actual={dev.read32(0x409018):#x}",
          flush=True)
    # Discriminator: did FECS actually write context bytes?
    if use_vram_inst:
      _nz = 0
      _sample = []
      _scan = min(ctx_size if "ctx_size" in dir() else 0x29b00, 0x8000)
      # ctx_size is on NVDevice; use discovered size from mailbox stash if present
      _scan_sz = getattr(dev, "gr_ctx_size", 0x29b00) or 0x29b00
      _scan_sz = min(_scan_sz, 0x10000)
      for _off in range(0, _scan_sz, 4):
        _w = _gk104_pramin_read32(dev, grctx_golden_vram_pa + _off)
        if _w:
          _nz += 1
          if len(_sample) < 8:
            _sample.append((_off, _w))
      print(f"[kepler] golden buf scan: size={_scan_sz:#x} nonzero_dwords={_nz} "
            f"first_hits={[ (hex(o), hex(v)) for o,v in _sample ]} "
            f"FECS_PC={dev.read32(0x409ff0):#x} CPUCTL={dev.read32(0x409100):#x} "
            f"INTR={dev.read32(0x409008):#x} IRQMASK={dev.read32(0x409018):#x}",
            flush=True)
    # Stop the golden-save keep-alive thread.
    _ka_stop2.set()
    if _ka_thread2 is not None:
      _ka_thread2.join(timeout=2.5)
      if _ka_thread2.is_alive():
        raise RuntimeError("golden-context keepalive did not stop")
    # Nouveau detaches the temporary golden engine context only while the
    # channel is stopped and KICK_CHID has completed.  The golden trace writes
    # RAMIN[0x210:0x218]=0 at this point; updating a live RAMFC lets PBDMA retain
    # a stale engine-context pointer in its loaded state.
    _gk104_stop_preempt_channel(dev, chan_id,
                                label="golden-context detach preempt")
    if use_vram_inst:
      _gk104_pramin_write(dev, ramin_vram_pa + 0x210, bytes(8))
      _gk104_bar_flush(dev)
      if (_gk104_pramin_read32(dev, ramin_vram_pa + 0x210) or
          _gk104_pramin_read32(dev, ramin_vram_pa + 0x214)):
        raise RuntimeError("GK104 golden engine-context pointer did not detach")
    else:
      struct.pack_into("<Q", vram, ramin_pa + 0x210, 0)
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
    # Tight poll: sample FECS CPUCTL to find exact gate time.
    _gate_samples = []
    _gate_iters = 50 if DEBUG else 5
    for _ in range(_gate_iters):
      _v = dev.read32(0x409100)
      _gate_samples.append((_v, _))
      if _v == 0xbadf1000 or (_v & 0xfffff000) == 0xbadf0000:
        break
      time.sleep(0.001)
    if DEBUG:
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
      if hw is not None:
        hw.set_phase("golden-context-ctx-copy")
      _copy_start = time.time()
      # H21: BAR1 golden→runtime eng-ctx copy false-passes the BAR1 sample
      # verify (both sides read the same corrupt image) and leaves sticky
      # GPC_STATUS=0x6 / FE_BUSY=1 after SET_OBJECT.  PRAMIN copy idles
      # (GPC_STATUS=0) with a correct mmio-list header.  Opt into BAR1 with
      # KEPLER_CTX_COPY=bar1; KEPLER_ENG_CTX=golden skips the copy entirely.
      _ctx_copy = os.environ.get("KEPLER_CTX_COPY", "pramin").strip().lower()
      if _ctx_copy not in ("pramin", "bar1"):
        raise ValueError(
            f"invalid KEPLER_CTX_COPY={_ctx_copy!r}; expected pramin|bar1")
      fast_ctx = (
          _ctx_copy == "bar1" and
          os.environ.get("KEPLER_FAST_ZERO", "1") != "0" and
          bool(getattr(dev, "_bar1_classic_posted", False) or
               getattr(dev, "_bar1_identity_ready", False) or
               bar1_identity_ready))
      runtime_ctx_expected = bytearray(ctx_size)
      if fast_ctx:
        _rd = _bar1_xfer["chunk"]
        for off in range(0, ctx_size, _rd):
          n = min(_rd, ctx_size - off)
          runtime_ctx_expected[off:off + n] = bytes(
              dev.dev_impl.hw.mmio_read(1, grctx_golden_vram_pa + off, n))
        # Sanity: reject stub/all-ones BAR1 images.  Do NOT reject all-zero
        # head — FECS golden save on this GK104 starts with zeros
        # (Night41bb: head=['0x0','0x0','0x0','0x0']) while later words are live.
        _head = struct.unpack_from("<4I", runtime_ctx_expected, 0)
        _sample_offs = (0x100, 0x200, 0x1000, 0x2000, max(0, ctx_size // 2))
        _sample = [struct.unpack_from("<I", runtime_ctx_expected, o)[0]
                   for o in _sample_offs if o + 4 <= ctx_size]
        if (all(w == 0xffffffff for w in _head) or
            any(_gk104_pramin_word_is_stub(w) for w in _head) or
            (any(_gk104_pramin_word_is_stub(w) for w in _sample))):
          fast_ctx = False
        elif all(w == 0 for w in _head) and all(w == 0 for w in _sample):
          # Entire sampled image is zero — FECS did not populate; use PRAMIN.
          fast_ctx = False
      if not fast_ctx:
        # BAR1 readback on some cold attaches returns complemented dwords.
        # Capture the FECS-saved image through authoritative BAR0 PRAMIN
        # (force_bar0: identity BAR1 substitute left sticky bit flips — H22).
        for off in range(0, ctx_size, 4):
          if not (off & 0xfff):
            dev.read32(0x409100)
          struct.pack_into(
              "<I", runtime_ctx_expected, off,
              _gk104_pramin_read32(dev, grctx_golden_vram_pa + off,
                                   force_bar0=True))
      if fast_ctx:
        bar1_write(grctx_vram_pa, runtime_ctx_expected)
      else:
        _gk104_pramin_write(dev, grctx_vram_pa, runtime_ctx_expected,
                            force_bar0=True)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate after runtime context copy failed")
      _copy_elapsed = time.time() - _copy_start
      _fecs_after_copy = dev.read32(0x409100)
      print(f"[kepler] ctx copy via {'BAR1' if fast_ctx else 'PRAMIN'}: "
            f"{_copy_elapsed:.3f}s FECS={_fecs_after_copy:#x}",
            flush=True)
      # Track FECS state through each subsequent step
      _fecs_check_points = []
      _fecs_check_points.append(("after_bar_flush", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      ctx_mismatches = []
      if fast_ctx:
        _rd = _bar1_xfer["chunk"]
        _sample_ctx = os.environ.get("KEPLER_ZERO_VERIFY", "sample") != "full"
        if _sample_ctx:
          for off in (0, ctx_size // 2, max(0, ctx_size - 4),
                      0x1000 if ctx_size > 0x1000 else 0):
            if off + 4 > ctx_size:
              continue
            got = bytes(dev.dev_impl.hw.mmio_read(1, grctx_vram_pa + off, 4))
            want = runtime_ctx_expected[off:off + 4]
            if got != want:
              ctx_mismatches.append((off, want, got))
        else:
          for off in range(0, ctx_size, _rd):
            n = min(_rd, ctx_size - off)
            got = bytes(dev.dev_impl.hw.mmio_read(1, grctx_vram_pa + off, n))
            want = runtime_ctx_expected[off:off + n]
            if got != want:
              for i in range(0, n, 4):
                if got[i:i + 4] != want[i:i + 4]:
                  ctx_mismatches.append(
                      (off + i, want[i:i + 4], got[i:i + 4]))
      else:
        for off in range(0, ctx_size, 4):
          actual = _gk104_pramin_read32(dev, grctx_vram_pa + off)
          wanted = struct.unpack_from("<I", runtime_ctx_expected, off)[0]
          if actual != wanted:
            ctx_mismatches.append(
                (off, struct.pack("<I", wanted), struct.pack("<I", actual)))
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
        if fast_ctx:
          for off in range(0, ctx_size, 0x1000):
            n = min(0x1000, ctx_size - off)
            got = bytes(dev.dev_impl.hw.mmio_read(1, grctx_vram_pa + off, n))
            want = runtime_ctx_expected[off:off + n]
            if got != want:
              for i in range(0, n, 4):
                if got[i:i + 4] != want[i:i + 4]:
                  ctx_mismatches.append(
                      (off + i, want[i:i + 4], got[i:i + 4]))
        else:
          for off in range(0, ctx_size, 4):
            actual = _gk104_pramin_read32(dev, grctx_vram_pa + off)
            wanted = struct.unpack_from("<I", runtime_ctx_expected, off)[0]
            if actual != wanted:
              ctx_mismatches.append(
                  (off, struct.pack("<I", wanted), struct.pack("<I", actual)))
        remaining = len(ctx_mismatches)
        print(f"[kepler] runtime GR ctx repair pass={repair_pass + 1} "
              f"remaining={remaining}", flush=True)
      if remaining:
        raise RuntimeError(f"runtime GR context copy remains corrupt ({remaining} dwords)")
      # These blobs are Nouveau's embedded hubgk104.fuc3 firmware, selected by
      # the !gr->firmware path (commands 1/2, not proprietary commands 3/9).
      # gf100_gr_chan_bind() therefore replaces the first two golden-image
      # words with the MMIO pair count and patch-list VA shifted by eight.
      #
      # H21 discriminator: KEPLER_ENG_CTX=golden points RAMIN 0x0210 at the
      # FECS-saved CB_RESERVED image (VA+0x80000) and skips the mmio-list
      # header rewrite.  Golden SAVE leaves GPC_STATUS=0; if SET_OBJECT then
      # also idles, the hang is in the runtime copy/header path, not a
      # phantom FE/GPC latch.
      _eng_ctx_mode = os.environ.get("KEPLER_ENG_CTX", "runtime").strip().lower()
      if _eng_ctx_mode not in ("runtime", "golden", "runtime-nohdr"):
        raise ValueError(
            f"invalid KEPLER_ENG_CTX={_eng_ctx_mode!r}; "
            "expected runtime|golden|runtime-nohdr")
      dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
      if _eng_ctx_mode == "golden":
        print(f"[kepler] ENG_CTX=golden: skip mmio-list header rewrite; "
              f"RAMIN→CB_RESERVED va={gr_ctx.va_addr + 0x80000:#x}",
              flush=True)
        runtime_ctx_va = (gr_ctx.va_addr + 0x80000) | 4
      else:
        if _eng_ctx_mode == "runtime":
          _runtime_header = struct.pack(
              "<II", len(runtime_mmio_entries), mmio_list.va_addr >> 8)
          for _off in (0x00, 0x04):
            runtime_ctx_expected[_off:_off + 4] = \
                _runtime_header[_off:_off + 4]
            _gk104_pramin_write(
                dev, grctx_vram_pa + _off, _runtime_header[_off:_off + 4])
          _gk104_bar_flush(dev)
        else:
          print(f"[kepler] ENG_CTX=runtime-nohdr: keep golden head on "
                f"runtime copy; mmio_entries={len(runtime_mmio_entries)}",
                flush=True)
        runtime_ctx_va = gr_ctx.va_addr | 4
      _fecs_check_points.append(("after_mmio_list_write", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
      _gk104_pramin_write(dev, ramin_vram_pa + 0x210,
                          struct.pack("<Q", runtime_ctx_va))
      _gk104_bar_flush(dev)
      _fecs_check_points.append(("after_ramin_ctx_write", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      dev.read32(0x409100)  # FECS keep-alive (read, not write, to preserve SCRATCH0)
      _head_pa = (grctx_golden_vram_pa if _eng_ctx_mode == "golden"
                  else grctx_vram_pa)
      runtime_head = [_gk104_pramin_read32(dev, _head_pa + i)
                      for i in range(0, 0x30, 4)]
      _fecs_check_points.append(("after_head_read", dev.read32(0x409100),
                                 dev.read32(0x400000), dev.read32(0x020004)))
      print(f"[kepler] FECS check points: "
            f"{[(n, hex(v), hex(p), hex(g)) for n, v, p, g in _fecs_check_points]}",
            flush=True)
    else:
      vram[grctx_pa:grctx_pa + ctx_size] = \
        vram[grctx_pa + 0x80000:grctx_pa + 0x80000 + ctx_size]
      struct.pack_into("<I", vram, grctx_pa + 0x10,
                       len(runtime_mmio_entries))
      struct.pack_into("<Q", vram, grctx_pa + 0x14, mmio_list.va_addr)
      for _off, _value in ((0x1c, 1), (0x20, 0), (0x28, 0), (0x2c, 0),
                           (0xf4, 0), (0xf8, 0)):
        struct.pack_into("<I", vram, grctx_pa + _off, _value)
      runtime_ctx_va = gr_ctx.va_addr | 4
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
          _gk104_fe_pwr_force_on(dev)
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
    _userd_gp_get = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + USERD_GP_GET, 4))[0]
    _userd_gp_put = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
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
    _fast_pte = (os.environ.get("KEPLER_FAST_ZERO", "1") != "0" and
                 bool(getattr(dev, "_bar1_classic_posted", False) or
                      getattr(dev, "_bar1_identity_ready", False) or
                      bar1_identity_ready))
    # Preserve and stabilise every populated source mapping (inputs, output,
    # code, constants, CWD, and allocator metadata).  Bulk cloning alone is
    # insufficient on this path because individual PTE dwords can settle to an
    # all-ones entry without producing an immediate MMU fault.
    #
    # FAST_ZERO/classic BAR1: skip the 0x8000-entry host SPT rescan.  Clone
    # already bulk-copied those leaves; re-issuing every non-zero host PTE via
    # TinyGPU dominated wall time (~16s).  Only re-issue explicit channel
    # overrides below, then BAR1-verify.
    if not _fast_pte:
      for _pgdi, _src_spt, _dst_spt in cloned_spts:
        if _src_spt < 0:
          continue
        for _spti in range(0x8000):
          _pte = struct.unpack_from("<Q", vram, _src_spt + _spti * 8)[0]
          if _pte:
            _live_pte_map[_dst_spt + _spti * 8] = _pte
    # Scratch/fault guards fill unallocated VAs only.  Apply them before real
    # buffer maps so a/b/out/temp mirrors are not stomped (the old order left a
    # 4 KiB hole at VA 0x67000 whenever out[] covered that page).
    for _pte_addr, _pte in ctx_alias_ptes:
      _live_pte_map[_pte_addr] = _pte
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
      _off = _page * 0x1000
      _pa = (grctx_vram_pa + _off) if _off < 0x80000 else (
          grctx_golden_vram_pa + (_off - 0x80000))
      _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = (_pa >> 8) | 3
    for _mirror in getattr(dev, "_kepler_vram_mirrors", ()):
      _chunks = _mirror.meta.get("vram_chunks")
      _mirror_pa = _mirror.meta.get("vram_pa")
      if _mirror_pa is None and not _chunks:
        continue
      if not _chunks:
        _chunks = [(_mirror_pa, round_up(_mirror.size, 0x1000))]
      _page = 0
      for _chunk_pa, _chunk_sz in _chunks:
        for _local in range(0, _chunk_sz, 0x1000):
          if _page * 0x1000 >= round_up(_mirror.size, 0x1000):
            break
          _va = _mirror.va_addr + _page * 0x1000
          _pgdi, _spti = (_va >> 27) & 0x1fff, (_va >> 12) & 0x7fff
          _pte = ((_chunk_pa + _local) >> 8) | 1
          if _mirror.meta.get("priv"):
            _pte |= 2
          _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = _pte
          _page += 1
    if mmio_list.meta.get("vram_pa") is None:
      _va = mmio_list.va_addr
      _pgdi, _spti = (_va >> 27) & 0x1fff, (_va >> 12) & 0x7fff
      _src_spt = mm.root_page_table.address(_pgdi)
      _src_pte = struct.unpack_from("<Q", vram, _src_spt + _spti * 8)[0]
      _live_pte_map[cloned_by_pgd[_pgdi] + _spti * 8] = _src_pte
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
    # Fast path trusts the BAR1 bulk SPT clone for host-backed demo buffers
    # (a/b/out/code/…).  Only the explicit overrides above are re-issued.
    _live_ptes = list(_live_pte_map.items())
    _live_entries = [*_live_pdes, *_live_ptes]
    # Classic+literal BAR1: each PTE is one 8-byte BAR1 store + BAR1 readback.
    # The old path used _gk104_pramin_write (per-dword RMW+verify) and two
    # _gk104_pramin_read32s per entry — dominant remaining TinyGPU cost after
    # FAST_ZERO (~16s here with hundreds of live PTEs).
    if hw is not None:
      hw.set_phase("golden-context-pte-stabilize")
    # Group PTE/PDE updates into 4 KiB BAR1 page RMW cycles.  Per-entry
    # TinyGPU RPCs for ~700 leaves were still ~16s; page bulk is tens of RPCs.
    _page_updates = {}
    for _entry_addr, _entry_wanted in _live_entries:
      _page = _entry_addr & ~0xfff
      _off = _entry_addr & 0xfff
      _page_updates.setdefault(_page, {})[_off] = _entry_wanted
    print(f"[kepler] PTE stabilize: entries={len(_live_entries)} "
          f"pages={len(_page_updates)} fast_pte={_fast_pte}", flush=True)
    for _attempt in range(4):
      if _fast_pte:
        for _page, _updates in _page_updates.items():
          _img = bytearray(
              bytes(dev.dev_impl.hw.mmio_read(1, _page, 0x1000)))
          for _off, _wanted in _updates.items():
            struct.pack_into("<Q", _img, _off, _wanted)
          bar1_write(_page, bytes(_img))
      else:
        for _entry_addr, _entry_wanted in _live_entries:
          _gk104_pramin_write(dev, _entry_addr, struct.pack("<Q", _entry_wanted))
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for channel PTEs did not complete")
      time.sleep(0.005)
      if _fast_pte:
        _ok = True
        _pte_actual = []
        for _page, _updates in _page_updates.items():
          _img = bytes(dev.dev_impl.hw.mmio_read(1, _page, 0x1000))
          for _off, _wanted in _updates.items():
            _got = struct.unpack_from("<Q", _img, _off)[0]
            _pte_actual.append(_got)
            if _got != _wanted:
              _ok = False
        if _ok:
          break
      else:
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
        # The semaphore is consumed through the channel VMM and its physical
        # VRAM page.  BAR1 VA identity leaves are a separate CPU aperture and
        # can independently settle to an invalid entry on this card; do not
        # reject a proven physical store because that optional alias is bad.
        _gk104_pramin_write(dev, signal_vram_pa, _signal_wanted)
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate for signal did not complete")
        time.sleep(0.005)
        _signal_bar = struct.unpack(
            "<I", dev.dev_impl.hw.mmio_read(1, signal_vram_pa, 4))[0]
        _signal_pramin = _gk104_pramin_read32(dev, signal_vram_pa)
        if _signal_pramin == signal_initial:
          break
      else:
        raise RuntimeError(
            f"GK104 signal did not stabilise: bar={_signal_bar:#x} "
            f"pramin={_signal_pramin:#x}")
    if hw is not None:
      hw.set_phase("golden-context-mirror-stabilize")
    for _mirror in getattr(dev, "_kepler_vram_mirrors", ()):
      _chunks = _mirror.meta.get("vram_chunks")
      _mirror_pa = _mirror.meta.get("vram_pa")
      if _mirror_pa is None and not _chunks:
        continue
      if not _chunks:
        _chunks = [(_mirror_pa, round_up(_mirror.size, 0x1000))]
      _mirror_wanted = bytes(vram[_mirror.meta["pa"]:
                                  _mirror.meta["pa"] + _mirror.size])
      if _mirror is mmio_list:
        _mmio_encoded = b"".join(
            struct.pack("<I", (~struct.unpack_from("<I", _mirror_wanted, off)[0]) &
                        0xffffffff)
            for off in range(0, len(_mirror_wanted), 4))
        if _fast_pte:
          _off = 0
          for _chunk_pa, _chunk_sz in _chunks:
            _take = min(_chunk_sz, len(_mmio_encoded) - _off)
            if _take <= 0:
              break
            bar1_write(_chunk_pa, _mmio_encoded[_off:_off + _take])
            _off += _take
        else:
          _gk104_pramin_write_literal(dev, _mirror_pa, _mmio_encoded)
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError("GK104 LTC invalidate for MMIO list failed")
        continue
      # temp_dev alone is 1 MiB (GK104_TEMP_SIZE).  Per-dword PRAMIN write +
      # verify here was the remaining ~16s / ~1.3M TinyGPU RPCs after FAST_ZERO.
      # Bit19-safe mirrors may be striped across 512 KiB banks (vram_chunks).
      for _attempt in range(4):
        if _fast_pte:
          _off = 0
          _via = set()
          for _chunk_pa, _chunk_sz in _chunks:
            _take = min(_chunk_sz, len(_mirror_wanted) - _off)
            if _take <= 0:
              break
            _via.add(vram_store(_chunk_pa, _mirror_wanted[_off:_off + _take],
                                label="mirror-stabilize"))
            _off += _take
          _gk104_bar_flush(dev)
          if not _gk104_ltc_invalidate(dev):
            raise TimeoutError("GK104 LTC invalidate for VRAM mirror failed")
          time.sleep(0.005)
          # H22: sample offsets (0, mid, end) missed bit15 flips at +0x4/+0x18
          # on the 32-byte `b` mirror.  Always full-verify small compute mirrors.
          _mirror_ok = True
          _sample_mode = (
              os.environ.get("KEPLER_ZERO_VERIFY", "sample") != "full" and
              len(_mirror_wanted) > 0x1000)
          if _sample_mode and len(_mirror_wanted) >= 16:
            for _poff in (0, len(_mirror_wanted) // 2,
                          max(0, len(_mirror_wanted) - 4)):
              # Map linear offset into striped chunks.
              _left, _base = _poff, None
              for _cpa, _csz in _chunks:
                if _left < _csz:
                  _base = _cpa + _left
                  break
                _left -= _csz
              if _base is None:
                continue
              got = vram_load(_base, 4)
              if got != _mirror_wanted[_poff:_poff + 4]:
                _mirror_ok = False
                break
            if _mirror_ok:
              break
          _mirror_got = bytearray()
          _off = 0
          for _chunk_pa, _chunk_sz in _chunks:
            _take = min(_chunk_sz, len(_mirror_wanted) - _off)
            if _take <= 0:
              break
            _mirror_got.extend(vram_load(_chunk_pa, _take))
            _off += _take
          if bytes(_mirror_got) == _mirror_wanted:
            break
          # Fall through to rewrite on next attempt when small mirrors drift.
        else:
          # As with the semaphore above, validate the physical VRAM image that
          # the channel PTE names.  The identity BAR1 alias is not part of that
          # GPU mapping and may have an independently corrupted leaf.
          _gk104_pramin_write(dev, _mirror_pa, _mirror_wanted)
          _gk104_bar_flush(dev)
          if not _gk104_ltc_invalidate(dev):
            raise TimeoutError("GK104 LTC invalidate for VRAM mirror failed")
          time.sleep(0.005)
          _mirror_pramin = b"".join(
              struct.pack("<I", _gk104_pramin_read32(dev, _mirror_pa + off))
              for off in range(0, len(_mirror_wanted), 4))
          if _mirror_pramin == _mirror_wanted:
            break
      else:
        raise RuntimeError(
            f"GK104 VRAM mirror did not stabilise: va={_mirror.va_addr:#x}")
    # Match the second stop/preempt in the golden trace before binding the
    # freshly copied runtime context, then leave the channel stopped until the
    # START write below.
    _gk104_stop_preempt_channel(dev, chan_id,
                                label="runtime-context bind preempt")
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
    _inst_gr_lo = _gk104_pramin_read32(dev, ramin_vram_pa + 0x210)
    _inst_gr_hi = _gk104_pramin_read32(dev, ramin_vram_pa + 0x214)
    _inst_gr_val = (_inst_gr_hi << 32) | _inst_gr_lo
    print(f"[kepler] inst GR ctx ptr: 0x{_inst_gr_val:010x} valid={bool(_inst_gr_val & 4)} "
          f"va=0x{(_inst_gr_val & ~4):x}", flush=True)
    # Also check other engine ctx ptrs for conflicts
    for _eng_name, _eng_off in [("SEC", 0x220), ("MSPDEC", 0x250), ("MSPPP", 0x260),
                                 ("MSVLD", 0x270), ("VIC", 0x280), ("MSENC", 0x290)]:
      _e_lo = _gk104_pramin_read32(dev, ramin_vram_pa + _eng_off)
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
  # Multi-IB: publish GP_PUT one entry at a time (staged).  Entry 0 first.
  _gp_put_initial = 1
  if put_before_runlist:
    if use_vram_inst:
      _gk104_pramin_write(dev, _userd_phys_base + USERD_GP_PUT,
                          struct.pack("<I", _gp_put_initial))
    else:
      dev.dev_impl.hw.mmio_write(1, _userd_mmio_base + USERD_GP_PUT,
                                 struct.pack("<I", _gp_put_initial))
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
    _alias = (userd_vram_pa
              if os.environ.get("KEPLER_USERD_ALIAS", "0") == "1"
              else None)
    _gk104_init_bar1_identity(dev, bus_base=base, map_vram=True,
                              userd_alias_pa=_alias)
    # gk104_fifo_init() runs after BAR1 init in Nouveau.  Re-latch the USERD
    # polling VMA now that 0x1704 points at the live BAR1 page directory.
    _userd_bar1 = 0 if _alias is not None else userd_vram_pa
    dev.write32(0x2254, 0x10000000 | (_userd_bar1 >> 12))
    _expected_put = 1 if put_before_runlist else 0
    for _attempt in range(4):
      _gk104_pramin_write(
          dev, userd_vram_pa + userd_base_off + USERD_GP_GET,
          struct.pack("<I", 0))
      _gk104_pramin_write(
          dev, userd_vram_pa + userd_base_off + USERD_GP_PUT,
          struct.pack("<I", _expected_put))
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for USERD data failed")
      time.sleep(0.005)
      _userd_alias_put = struct.unpack(
          "<I", dev.dev_impl.hw.mmio_read(
              1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
      _userd_alias_get = struct.unpack(
          "<I", dev.dev_impl.hw.mmio_read(
              1, _userd_mmio_base + USERD_GP_GET, 4))[0]
      _userd_phys_put = _gk104_pramin_read32(
          dev, userd_vram_pa + userd_base_off + USERD_GP_PUT)
      _userd_phys_get = _gk104_pramin_read32(
          dev, userd_vram_pa + userd_base_off + USERD_GP_GET)
      if (_userd_alias_get == _userd_phys_get == 0 and
          _userd_alias_put == _userd_phys_put == _expected_put):
        break
    else:
      raise RuntimeError(
          f"GK104 BAR1 USERD data did not stabilise before runlist: "
          f"get={_userd_alias_get:#x}/{_userd_phys_get:#x} "
          f"put={_userd_alias_put:#x}/{_userd_phys_put:#x}")
    if DEBUG:
      print(f"[kepler] BAR1 USERD alias stable: GP_PUT={_userd_alias_put}",
            flush=True)
  if use_vram_runlist:
    # PRAMIN stores can acknowledge with the desired immediate readback and
    # settle to the XOR-inverted value later.  Stabilise the tiny runlist only
    # after every other framebuffer write, and verify after flush/invalidate.
    runlist_entry = struct.pack("<" + "I" * len(runlist_words), *runlist_words)
    for _attempt in range(4):
      _gk104_pramin_write(dev, runlist_pa, runlist_entry)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for runlist did not complete")
      time.sleep(0.005)
      _runlist_actual = tuple(
          _gk104_pramin_read32(dev, runlist_pa + off)
          for off in range(0, len(runlist_entry), 4))
      if _runlist_actual == runlist_words:
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
      _userd_alias_get = struct.unpack(
          "<I", dev.dev_impl.hw.mmio_read(
              1, _userd_mmio_base + USERD_GP_GET, 4))[0]
      _userd_phys_put = _gk104_pramin_read32(
          dev, userd_vram_pa + userd_base_off + USERD_GP_PUT)
      _userd_phys_get = _gk104_pramin_read32(
          dev, userd_vram_pa + userd_base_off + USERD_GP_GET)
      _expected_put = 1 if put_before_runlist else 0
      if (_userd_alias_get == _userd_phys_get == 0 and
          _userd_alias_put == _userd_phys_put == _expected_put):
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
        if userd_vram_pa < 0x08000000:
          _userd_page = userd_vram_pa // 0x1000 + _page
          _gk104_pramin_write(
              dev, 0x00120000 + _userd_page * 8,
              struct.pack("<Q", (_pa >> 8) | 1))
      _gk104_pramin_write(
          dev, userd_vram_pa + userd_base_off + USERD_GP_GET,
          struct.pack("<I", 0))
      _gk104_pramin_write(
          dev, userd_vram_pa + userd_base_off + USERD_GP_PUT,
          struct.pack("<I", _expected_put))
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for USERD alias did not complete")
      _gk104_vmm_flush_pdb(dev, 0x00110000, target=0, hub_only=True)
      dev.write32(0x001704, 0x80000000 | (0x00100000 >> 12))
      dev.write32(0x2254, 0x10000000)
    else:
      raise RuntimeError(
          f"GK104 BAR1 USERD alias did not remain stable: "
          f"get={_userd_alias_get:#x}/{_userd_phys_get:#x} "
          f"put={_userd_alias_put:#x}/{_userd_phys_put:#x}")
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
      if push_vram_pa is not None:
        bar1_write(push_vram_pa + push_phys_offset, push_bytes)
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
      if (os.environ.get("KEPLER_ZERO_VERIFY", "sample") != "full" and
          _ring_actual == ring_store and
          _push_actual == bytes(push_bytes)):
        break
      # BAR1 mismatch (or full verify): dual-write via PRAMIN then re-check.
      if gpfifo_vram_pa is not None and not _literal_ring:
        _gk104_pramin_write(dev, gpfifo_vram_pa, ring_store)
      if push_vram_pa is not None:
        _gk104_pramin_write(dev, push_vram_pa + push_phys_offset, push_bytes)
      _gk104_bar_flush(dev)
      if not _gk104_ltc_invalidate(dev):
        raise TimeoutError("GK104 LTC invalidate for precommit commands failed")
      time.sleep(0.005)
      _ring_actual = (dev.dev_impl.hw.mmio_read(1, gpfifo_vram_pa, len(ring_store))
                      if gpfifo_vram_pa is not None else ring_store)
      _push_actual = (dev.dev_impl.hw.mmio_read(
          1, push_vram_pa + push_phys_offset, len(push_bytes))
          if push_vram_pa is not None else bytes(push_bytes))
      if (os.environ.get("KEPLER_ZERO_VERIFY", "sample") != "full" and
          _ring_actual == ring_store and
          _push_actual == bytes(push_bytes)):
        break
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
  # Re-assert the runlist ID after the stop/start context-pointer transition,
  # then submit the populated list a second time.  The golden trace contains
  # this distinct runtime commit after the temporary golden context has been
  # detached; without it, PFIFO retains the earlier stopped channel state and
  # never generates the FECS runtime context switch.
  nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x000f0000, GR_RUNLIST_ID << 16)
  if DEBUG:
    print(f"[kepler] pre-runlist chan_ctrl=0x{dev.read32(CHAN_START_REG + chan_id * 8):08x}", flush=True)
  if hw is not None:
    hw.set_phase("runlist-submit")
  _gk104_commit_runlist(dev, runlist_addr, runlist_active_count,
                        target=runlist_target)
  if DEBUG and use_vram_runlist:
    print(f"[kepler] runlist_vram pa={runlist_pa:#x} words="
          f"{struct.unpack('<II', dev.dev_impl.hw.mmio_read(1, runlist_pa, 8))} "
          f"runq_masks={[hex(dev.read32(0x2390 + i * 4)) for i in range(3)]}")
  if DEBUG:
    print(f"[kepler] runlist submit: id={GR_RUNLIST_ID} "
          f"0x2270=0x{dev.read32(0x2270):08x} "
          f"0x2274=0x{dev.read32(0x2274):08x}", flush=True)
  # gk104_runl_pending() is the per-runlist bit at 0x2284 + runl_id * 8.
  _runl_pending_reg = 0x2284 + GR_RUNLIST_ID * 8
  if DEBUG:
    print(f"[kepler] runlist pending reg=0x{_runl_pending_reg:x} "
          f"val=0x{dev.read32(_runl_pending_reg):08x}", flush=True)
  _gk104_wait_runlist_idle(dev, label="runtime channel runlist")
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
  if os.environ.get("KEPLER_POST_RUNLIST_START") == "1":
    nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x00000400, 0x00000400)
  if DEBUG:
    print(f"[kepler] post-runlist chan_ctrl=0x{dev.read32(CHAN_START_REG + chan_id * 8):08x}", flush=True)
  # Give the scheduler a brief window to populate CHAN_TABLE after runlist DMA.
  time.sleep(0.002 if not DEBUG else 0.01)
  if DEBUG:
    _ct_chan = dev.read32(0x3000 + chan_id * 8)
    _ct_state = dev.read32(0x3004 + chan_id * 8)
    print(f"[kepler] CHAN_TABLE after delay: CHAN=0x{_ct_chan:08x} STATE=0x{_ct_state:08x}", flush=True)
  # Kick the PBDMA to start processing the channel.  KICK_CHID (0x2634)
  # tells the scheduler to immediately dispatch the specified channel.
  if unsafe_experiments:
    dev.write32(0x2634, chan_id)
  time.sleep(0.002 if not DEBUG else 0.01)
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
  if DEBUG and dev.dev_impl.hw is not None:
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
      _post_userd_gp_get = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + USERD_GP_GET, 4))[0]
      _post_userd_gp_put = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
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
  # FECS / CHSW diagnostics (DEBUG only — dozens of BAR0 RPCs on the hot path).
  if DEBUG:
    _fecs_ctrl_post_rl = dev.read32(0x409100)
    _subch_post_rl = dev.read32(0x404200)
    print(f"[kepler] post-runlist FECS: CPUCTL={_fecs_ctrl_post_rl:#x} "
          f"SCRATCH0={dev.read32(0x409800):#x} "
          f"CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x} "
          f"subch4={_subch_post_rl:#x}", flush=True)
  _chan_next_pre = dev.read32(0x409b04)
  _chan_addr_pre = dev.read32(0x409b00)
  if DEBUG:
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
  if DEBUG:
    _chsw_elapsed = time.time() - _chsw_start
    print(f"[kepler] manual CHSW result: done={_chsw_done} "
          f"elapsed={_chsw_elapsed:.3f}s "
          f"CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x} "
          f"SCRATCH0={dev.read32(0x409800):#x} "
          f"PC={dev.read32(0x409ff0):#x}", flush=True)
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
  if DEBUG:
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
  if unsafe_experiments and (dev.read32(0x409100) & 0x20):  # SLEEPING
    # Enable FIFO_DATA interrupt (bit 2) in INTR_EN_SET
    dev.write32(0x409010, 0xffffffff)  # IREN: enable all external interrupts
    dev.write32(0x409010 + 0x4, 0xff)  # INTR_EN_SET: enable all falcon interrupts
    # Actually use INTR_EN_SET at 0x409010 + 4 = 0x409014
    # Wait, the registers are: 0x010 = INTR_EN_SET, 0x014 = INTR_EN_CLR, 0x018 = INTR_EN
    # Let me re-check: INTR_EN_SET is at offset 0x010
    dev.write32(0x409010, 0xff)  # INTR_EN_SET: enable all interrupts
    time.sleep(0.001)
    if DEBUG:
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
  if DEBUG:
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
  # Rapid poll of FECS state to catch the CHSW processing (DEBUG only —
  # 100×4 BAR0 RPCs + sleeps burned ~0.8–1s on the live TinyGPU path).
  if DEBUG and dev.dev_impl.hw is not None:
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
      if push_vram_pa is not None:
        bar1_write(push_vram_pa + push_phys_offset, push_bytes)
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
      # Classic BAR1: trust BAR1 image; skip per-dword PRAMIN (was thousands
      # of TinyGPU RPCs for the push buffer).
      if (os.environ.get("KEPLER_ZERO_VERIFY", "sample") != "full" and
          _ring_actual == ring_store and
          _push_actual == bytes(push_bytes)):
        break
      if gpfifo_vram_pa is not None:
        _gk104_pramin_write(dev, gpfifo_vram_pa, ring_store)
      if push_vram_pa is not None:
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
      if (os.environ.get("KEPLER_ZERO_VERIFY", "sample") != "full" and
          _ring_actual == ring_store and
          _push_actual == bytes(push_bytes)):
        break
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
    # Nouveau does not issue a host FECS command here.  Advancing GP_PUT lets
    # the scheduler request the runtime context switch; FECS then consumes the
    # saved context and its per-channel MMIO patch list atomically.
    if use_vram_inst:
      # The context image and patch list are as translation-critical as their
      # PTEs.  Re-publish both through PRAMIN at the point of use so a late
      # complemented framebuffer dword cannot poison FECS context load.
      for _attempt in range(4):
        _gk104_pramin_write_literal(dev, grctx_vram_pa,
                                    runtime_ctx_expected)
        _gk104_pramin_write_literal(dev, mmio_list_vram_pa, mmio_blob)
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError(
              "GK104 LTC invalidate for pre-GP_PUT context did not complete")
        time.sleep(0.005)
        _ctx_actual = bytearray(ctx_size)
        for _off in range(0, ctx_size, 4):
          struct.pack_into("<I", _ctx_actual, _off,
                           _gk104_pramin_read32(dev, grctx_vram_pa + _off))
        _mmio_actual = bytearray(len(mmio_blob))
        for _off in range(0, len(mmio_blob), 4):
          struct.pack_into("<I", _mmio_actual, _off,
                           _gk104_pramin_read32(
                               dev, mmio_list_vram_pa + _off))
        if (_ctx_actual == runtime_ctx_expected and
            _mmio_actual == mmio_blob):
          break
      else:
        raise RuntimeError(
            "GK104 pre-GP_PUT context/MMIO list did not stabilise")
      # The initial proof happens before context copying and the second
      # runlist commit.  This card's framebuffer dwords can drift much later;
      # re-prove every live translation immediately before the first PBDMA
      # fetch so GPFIFO and pushbuffer leaves cannot become PAGE_NOT_PRESENT in
      # that gap.
      for _attempt in range(4):
        for _entry_addr, _entry_wanted in _live_entries:
          _gk104_pramin_write_literal(
              dev, _entry_addr, struct.pack("<Q", _entry_wanted))
        _gk104_bar_flush(dev)
        if not _gk104_ltc_invalidate(dev):
          raise TimeoutError(
              "GK104 LTC invalidate for pre-GP_PUT PTEs did not complete")
        time.sleep(0.005)
        _preput_actual = [
            _gk104_pramin_read32(dev, _addr) |
            (_gk104_pramin_read32(dev, _addr + 4) << 32)
            for _addr, _wanted in _live_entries]
        if _preput_actual == [_wanted for _addr, _wanted in _live_entries]:
          break
      else:
        raise RuntimeError("GK104 pre-GP_PUT PTEs did not stabilise")
      if not _gk104_vmm_flush_pdb(dev, vmm_pgd_pa, target=0):
        raise TimeoutError("GK104 pre-GP_PUT VMM flush did not complete")
      _guard_after_vmm = (
          _gk104_pramin_read32(dev, fault_guard_pte_addr) |
          (_gk104_pramin_read32(dev, fault_guard_pte_addr + 4) << 32))
      print(f"[kepler] guard PTE after VMM flush: "
            f"wanted={fault_guard_pte_wanted:#x} "
            f"actual={_guard_after_vmm:#x}", flush=True)
    print(f"[kepler] runtime context armed: CHAN_ADDR={dev.read32(0x409b00):#x} "
          f"CHAN_NEXT={dev.read32(0x409b04):#x}", flush=True)
    # H22: bad add lanes matched a+(b with mantissa bit15 flipped).  Prove
    # whether VRAM already holds the flipped bits before GP_PUT.
    if (use_vram_inst and
        os.environ.get("KEPLER_DUMP_MIRROR_AB", "1") != "0"):
      for _m in getattr(dev, "_kepler_vram_mirrors", ()) or ():
        _mpa = _m.meta.get("vram_pa")
        if _mpa is None or _m.size > 0x100:
          continue
        try:
          _want = bytes(_m.cpu_view()[:_m.size])
        except Exception:
          continue
        _got = bytearray(_m.size)
        for _off in range(0, _m.size, 4):
          struct.pack_into("<I", _got, _off,
                          _gk104_pramin_read32(dev, _mpa + _off,
                                               force_bar0=True))
        _xors = []
        for _off in range(0, _m.size, 4):
          _w = struct.unpack_from("<I", _want, _off)[0]
          _g = struct.unpack_from("<I", _got, _off)[0]
          if _w != _g:
            _xors.append((_off, _w, _g, _w ^ _g))
        if _xors:
          # H22 settle: rewrite drifted dwords through BAR0 PRAMIN and recheck.
          for _off, _w, _g, _x in _xors:
            _gk104_pramin_write(dev, _mpa + _off, struct.pack("<I", _w),
                                force_bar0=True)
          _gk104_bar_flush(dev)
          time.sleep(0.002)
          _xors2 = []
          for _off in range(0, _m.size, 4):
            _w = struct.unpack_from("<I", _want, _off)[0]
            _g = _gk104_pramin_read32(dev, _mpa + _off, force_bar0=True)
            struct.pack_into("<I", _got, _off, _g)
            if _w != _g:
              _xors2.append((_off, _w, _g, _w ^ _g))
          _xors = _xors2
        print(f"[kepler] pre-GP_PUT mirror va={_m.va_addr:#x} pa={_mpa:#x} "
              f"size={_m.size:#x} mismatches={len(_xors)} "
              f"xors={[ (hex(o), hex(w), hex(g), hex(x)) for o,w,g,x in _xors[:8] ]} "
              f"head_want={_want[:16].hex()} head_got={bytes(_got[:16]).hex()}",
              flush=True)
        if _xors:
          raise RuntimeError(
              f"GK104 compute mirror unsettled before GP_PUT: "
              f"va={_m.va_addr:#x} xors={[(hex(o), hex(x)) for o,_,_,x in _xors]}")
    if use_vram_inst:
      # Make the observed GR guard leaf the final framebuffer store before the
      # USERD notification.  The complete hierarchy was already invalidated
      # above; this last literal publication closes the window in which the
      # cold PRAMIN path can settle a dword to its complemented write payload.
      _gk104_pramin_write_literal(
          dev, fault_guard_pte_addr,
          struct.pack("<Q", fault_guard_pte_wanted))
      _gk104_bar_flush(dev)
      _guard_before_doorbell = (
          _gk104_pramin_read32(dev, fault_guard_pte_addr) |
          (_gk104_pramin_read32(dev, fault_guard_pte_addr + 4) << 32))
      print(f"[kepler] guard PTE immediately before GP_PUT: "
            f"{_guard_before_doorbell:#x}", flush=True)
      # Ring through BAR1 last.  Do not mirror GP_PUT through PRAMIN or issue
      # another LTC invalidate after notification: either operation widens the
      # interval before PBDMA/GR reads the just-published leaf.
      for _attempt in range(4):
        _get_before = _gk104_pramin_read32(
            dev, _userd_phys_base + USERD_GP_GET)
        # GP_GET may already have advanced to one between attempts.  That is
        # successful PBDMA consumption, not framebuffer drift; never roll a
        # consumed entry back to zero.  Staged multi-IB starts at PUT=1.
        if _get_before not in (0, 1):
          _gk104_pramin_write(dev, _userd_phys_base + USERD_GP_GET,
                              struct.pack("<I", 0))
        # This is the normal userspace notification path: USERD is mapped in
        # BAR1 precisely so GP_PUT can be written through the CPU aperture.
        if os.environ.get("KEPLER_SUBMIT_FE_PWR", "1") != "0":
          _gk104_fe_pwr_force_on(dev)
        dev.dev_impl.hw.mmio_write(
            1, _userd_mmio_base + USERD_GP_PUT, struct.pack("<I", _gp_put_initial))
        _gk104_bar_flush(dev)
        time.sleep(0.001)
        _put_phys = _gk104_pramin_read32(
            dev, _userd_phys_base + USERD_GP_PUT)
        _get_phys = _gk104_pramin_read32(
            dev, _userd_phys_base + USERD_GP_GET)
        _put_bar = struct.unpack(
            "<I", dev.dev_impl.hw.mmio_read(
                1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
        _get_bar = struct.unpack(
            "<I", dev.dev_impl.hw.mmio_read(
                1, _userd_mmio_base + USERD_GP_GET, 4))[0]
        # GET is GPU-owned and can advance through one aperture before the
        # other CPU view becomes coherent.  Staged kick: PUT starts at 1.
        if (_get_phys in (0, 1) and _get_bar in (0, 1) and
            _put_phys == _put_bar == _gp_put_initial):
          break
      else:
        raise RuntimeError(
            f"GK104 USERD doorbell did not stabilise: "
            f"get={_get_phys:#x}/{_get_bar:#x} "
            f"put={_put_phys:#x}/{_put_bar:#x}")
    else:
      dev.dev_impl.hw.mmio_write(1, _userd_mmio_base + USERD_GP_PUT,
                                 struct.pack("<I", _gp_put_initial))
      _gk104_bar_flush(dev)
  # Host kernel window starts at doorbell (GP_PUT), not at poll entry —
  # small N retires before the first semaphore RPC otherwise (~0ms).
  _kernel_t0 = time.perf_counter()
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
    print(f"[kepler] BAR1 USERD GP_PUT write={_gp_put_initial}/{n_gp_entries} readback={userd_put_readback:#x} "
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
  _last_fecs_poll = None
  if hw is not None:
    hw.set_phase("semaphore-poll")
  # Staged multi-IB: GP_PUT starts at 1; after each WFI bump PUT to publish
  # the next pre-packed ring entry.  Do not write GP_GET (GPU-owned); resetting
  # it left PUT==GET and an empty ring on re-kick attempts.
  _sem_targets = list(range(done_value - n_gp_entries + 1, done_value + 1))
  assert _sem_targets[-1] == done_value, (_sem_targets, done_value)
  val = signal_initial
  # Host-visible "kernel time": GP_PUT → done_value.  Includes post-kick
  # diagnostics + TinyGPU poll RPCs; compare baseline vs boost, not FLOPs.
  if "_kernel_t0" not in locals():
    _kernel_t0 = time.perf_counter()
  for _ti, _target in enumerate(_sem_targets):
    if _ti > 0:
      _next_put = _ti + 1
      print(f"[kepler] staged GP_PUT={_next_put}/{n_gp_entries} "
            f"after sem={_sem_targets[_ti - 1]}", flush=True)
      if dev.dev_impl.hw is not None:
        dev.dev_impl.hw.mmio_write(
            1, _userd_mmio_base + USERD_GP_PUT, struct.pack("<I", _next_put))
        _gk104_bar_flush(dev)
    _gp_get_snapshot_taken = False
    for _ in range(2000):
      # H18: eng-ctx auto-load on SET_OBJECT races FE power-gate; keep FE alive
      # in the same RPC thread (works with KEPLER_SINGLE_THREAD_RPC=1).
      if (dev.dev_impl.hw is not None and
          os.environ.get("KEPLER_SUBMIT_FE_PWR", "1") != "0"):
        _gk104_fe_pwr_force_on(dev)
      if signal_vram_pa is not None:
        val = _gk104_pramin_read32(dev, signal_vram_pa)
      else:
        val = struct.unpack_from("<I", vram, signal_pa)[0]
      if val == _target:
        if _target == done_value:
          _kernel_ms = (time.perf_counter() - _kernel_t0) * 1000.0
          _prev = float(getattr(dev, "_kepler_kernel_time_ms", 0.0) or 0.0)
          setattr(dev, "_kepler_kernel_time_ms", _prev + _kernel_ms)
          print(f"[kepler] kernel_time_ms={_kernel_ms:.3f} "
                f"total_ms={_prev + _kernel_ms:.3f} "
                f"sem={done_value} ibs={n_gp_entries}", flush=True)
          return
        break
      if not _gp_get_snapshot_taken and dev.dev_impl.hw is not None:
        try:
          _gp_get_now = struct.unpack("<I", dev.dev_impl.hw.mmio_read(
              1, _userd_mmio_base + USERD_GP_GET, 4))[0]
          _intr = dev.read32(0x400100)
          _trap = dev.read32(0x400108)
          _gpc_trap = dev.read32(0x400118)
          _sked_status = dev.read32(0x407020) & 0x3fffffff
          # Service SKED even when GP_GET is still 0 — on 7-TPC boards the
          # entry can trap before USERD advances, and waiting for GET leaves
          # the channel wedged forever.
          _empty_sked_trap = bool(
              (_intr & 0x00200000) and _trap == 0x00000100 and
              _gpc_trap == 0 and _sked_status == 0)
          _any_sked = bool(_sked_status) or bool(_trap & 0x100)
          if _empty_sked_trap or (_any_sked and not _gp_get_snapshot_taken):
            if _empty_sked_trap:
              dev.write32(0x400108, 0x00000100)
              dev.write32(0x400100, 0x00200000)
              dev.write32(0x400500, 0x00010001)
              print(f"[kepler-trap] serviced empty SKED: "
                    f"INTR={dev.read32(0x400100):#x} "
                    f"TRAP={dev.read32(0x400108):#x} "
                    f"GP_GET={_gp_get_now}", flush=True)
            elif _any_sked:
              print(f"[kepler-trap] SKED live during poll: "
                    f"INTR={_intr:#x} TRAP={_trap:#x} "
                    f"SKED={_sked_status:#x}[{','.join(_decode_bits(_sked_status, _GK104_SKED_ERRORS)) or '-'}] "
                    f"GPC_TRAP={_gpc_trap:#x} GP_GET={_gp_get_now}",
                    flush=True)
            _gp_get_snapshot_taken = True
        except Exception:
          pass
      time.sleep(0.001)
    else:
      done_value = _target
      break
  else:
    return
  # Do not quiesce and then continue issuing diagnostic BAR reads.  The caller's
  # finally block owns teardown, and a timeout path must fail without hundreds
  # of extra endpoint transactions against a potentially wedged channel.
  if use_vram_inst and dev.dev_impl.hw is not None:
    try:
      _intr = dev.read32(0x400100)
      _trap = dev.read32(0x400108)
      _sked = dev.read32(0x407020) & 0x3fffffff
      _stat = dev.read32(0x400700)
      _taddr = dev.read32(0x400704)
      _tdata = dev.read32(0x400708)
      _subch = [dev.read32(0x404200 + i * 4) for i in range(8)]
      print(f"[kepler] timeout GR: INTR={_intr:#x} TRAP={_trap:#x}"
            f"[{','.join(_decode_bits(_trap, _GK104_PGRAPH_TRAP_BITS)) or '-'}] "
            f"STATUS={_stat:#x} TRAPPED={_taddr:#x}/{_tdata:#x} "
            f"SKED={_sked:#x}[{','.join(_decode_bits(_sked, _GK104_SKED_ERRORS)) or '-'}] "
            f"SUBCH={[hex(x) for x in _subch]} "
            f"FECS={dev.read32(0x409100):#x}@{dev.read32(0x409ff0):#x} "
            f"CHAN={dev.read32(0x409b00):#x}",
            flush=True)
      _gs = [_gk104_pramin_read32(dev, grctx_golden_vram_pa + o)
             for o in (0, 0x100, 0x200, 0x1000, 0x2000)]
      _rs = [_gk104_pramin_read32(dev, grctx_vram_pa + o)
             for o in (0, 0x100, 0x200, 0x1000, 0x2000)]
      print(f"[kepler] timeout ctx sample golden={[hex(x) for x in _gs]} "
            f"runtime={[hex(x) for x in _rs]}", flush=True)
    except Exception as _trap_e:
      print(f"[kepler] timeout GR dump failed: {_trap_e}", flush=True)
  if use_vram_inst:
    _timeout_fault_source = dev.read32(0x259c)
    _timeout_faults = []
    for _unit in range(8):
      if _timeout_fault_source & (1 << _unit):
        _base = 0x2800 + _unit * 0x10
        _timeout_faults.append(
            (_unit, dev.read32(_base), dev.read32(_base + 4),
             dev.read32(_base + 8), dev.read32(_base + 12)))
    _fault_va_pte_addr = cloned_by_pgd[0] + 0x100 * 8
    _fault_va_pte = (_gk104_pramin_read32(dev, _fault_va_pte_addr) |
                     (_gk104_pramin_read32(dev, _fault_va_pte_addr + 4) << 32))
    _timeout_get_phys = _gk104_pramin_read32(
        dev, _userd_phys_base + USERD_GP_GET)
    _timeout_put_phys = _gk104_pramin_read32(
        dev, _userd_phys_base + USERD_GP_PUT)
    _timeout_get_bar = struct.unpack(
        "<I", dev.dev_impl.hw.mmio_read(
            1, _userd_mmio_base + USERD_GP_GET, 4))[0]
    _timeout_put_bar = struct.unpack(
        "<I", dev.dev_impl.hw.mmio_read(
            1, _userd_mmio_base + USERD_GP_PUT, 4))[0]
    print(f"[kepler] timeout USERD get={_timeout_get_phys:#x}/"
          f"{_timeout_get_bar:#x} put={_timeout_put_phys:#x}/"
          f"{_timeout_put_bar:#x} PBDMA0_IB="
          f"{dev.read32(0x40000):#x}/{dev.read32(0x40014):#x} "
          f"DMA_GET={dev.read32(0x40018):#x} CHID={dev.read32(0x40120):#x} "
          f"ENG={dev.read32(0x400a4):#x} PFIFO={dev.read32(0x2100):#x} "
          f"FAULTS={_timeout_fault_source:#x} "
          f"FAULT_RECORDS={[(u, hex(i), hex(lo), hex(hi), hex(t)) for u, i, lo, hi, t in _timeout_faults]} "
          f"PTE[0x100000]={_fault_va_pte:#x} "
          f"GR_BUFS={dev.read32(0x40800c):#x}/"
          f"{dev.read32(0x408004):#x}/"
          f"{dev.read32(0x418810):#x}/"
          f"{dev.read32(0x419848):#x} "
          f"CHAN={dev.read32(0x409b00):#x}/{dev.read32(0x409b04):#x} "
          f"FECS={dev.read32(0x409100):#x}@{dev.read32(0x409ff0):#x}",
          flush=True)
  # Large-N: did later chunk launches retire even if the final sem stuck?
  try:
    import os as _os
    _n = int(_os.environ.get("KEPLER_N", "0") or "0")
    if _n > 0 and use_vram_inst and dev.dev_impl.hw is not None:
      _out_pa = None
      for _m in (getattr(dev, "_kepler_vram_mirrors", None) or []):
        if getattr(_m, "size", 0) == _n * 4 and getattr(_m, "va_addr", 0) == 0x50000:
          _out_pa = _m.meta.get("vram_pa") or _m.meta.get("pa")
          break
      if _out_pa is None:
        # Fall back to the classic out mirror PA from recent brings-up.
        _out_pa = 0xb38000
      _chunk = 1024
      _words = []
      for _off in (0, 16 * _chunk, 19 * _chunk, _n - 1):
        if 0 <= _off < _n:
          _w = _gk104_pramin_read32(dev, _out_pa + _off * 4)
          _words.append((_off, _w))
      _sent = 0
      for _i in range(0, _n, max(1, _n // 64)):
        if _gk104_pramin_read32(dev, _out_pa + _i * 4) == 0x7fc00001:
          _sent += 1
      print(f"[kepler] timeout out sample pa={_out_pa:#x} "
            f"words={[(o, hex(w)) for o, w in _words]} "
            f"sentinel_stride≈{_sent}/64", flush=True)
  except Exception as _e:
    print(f"[kepler] timeout out sample failed: {_e}", flush=True)
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


def _run_hardware_demo_windows(dev, cubin, a_host, b_host, chunk, window_chunks):
  """Split large N across fresh channels: live GK104 retires ≤~19 LAUNCHes
  per channel lifetime (multi-IB on the same channel still dies after the
  20th LAUNCH even though GET/DMA advance)."""
  N = len(a_host)
  assert len(b_host) == N
  win = window_chunks * chunk
  backend = os.environ.get("NV_BACKEND", "kepler")
  orig_n = os.environ.get("KEPLER_N")
  cur = dev
  offset = 0
  first = True
  operation = os.environ.get("KEPLER_OPERATION", "add")
  try:
    while offset < N:
      n_win = min(win, N - offset)
      print(f"[kepler] channel window offset={offset} n={n_win} "
            f"(≤{window_chunks} LAUNCHes/channel)", flush=True)
      if not first:
        try:
          cur.close()
        except Exception as e:
          print(f"[kepler] window close: {e}", flush=True)
        os.environ["KEPLER_ALLOW_DIRTY"] = "1"
        cur = NVDevice("NV", backend=backend)
      os.environ["KEPLER_N"] = str(n_win)
      run_hardware_demo(
          cur, cubin,
          _hosts=(a_host[offset:offset + n_win],
                  b_host[offset:offset + n_win]),
          _no_window=True)
      offset += n_win
      first = False
  finally:
    if orig_n is None:
      os.environ.pop("KEPLER_N", None)
    else:
      os.environ["KEPLER_N"] = orig_n
    if cur is not dev:
      try:
        cur.close()
      except Exception:
        pass
  print(f"hardware_demo=ok N={N} operation={operation} (channel-windowed)",
        flush=True)


def run_hardware_demo(dev, cubin=None, *, _hosts=None, _no_window=False):
  """End-to-end add on the real GTX 770 over raw PCIe MMIO (sysmem compute
  path, plan §24.1).  Requires root, a GK104 firmware tree ($NV_FIRMWARE_DIR),
  and on-silicon FIFO validation."""
  import random
  setattr(dev, "_kepler_kernel_time_ms", 0.0)
  N = int(os.environ.get("KEPLER_N", "256"))
  if N <= 0:
    raise ValueError(f"invalid KEPLER_N={N}; expected > 0")
  # a+b+out in BAR1-backed VRAM; leave headroom for ctx/temp/heap (~1 MiB TLS).
  if N * 12 > 0x4000000:
    raise ValueError(
        f"KEPLER_N={N} needs {N * 12:#x} bytes for a+b+out; "
        f"max ~64 MiB under 128 MiB BAR1")
  # Soft cap: a+b+out fits under ~64 MiB of 128 MiB BAR1; multi-IB keeps each
  # GPFIFO entry ≤KEPLER_LAUNCH_IB_MAX (default 16) LAUNCHes.
  _n_max = int(os.environ.get("KEPLER_N_MAX", "1048576"), 0)
  if N > _n_max and os.environ.get("KEPLER_N_FORCE", "0") == "0":
    raise ValueError(
        f"KEPLER_N={N} exceeds KEPLER_N_MAX={_n_max} "
        f"(set KEPLER_N_FORCE=1 to override)")
  # Multi-CTA (one LAUNCH, grid_x=n_chunks).  Default auto max tracks N_MAX;
  # the old 20480 cap was only to dodge a 4 KiB PTE hole at VA 0x67000.
  if os.environ.get("KEPLER_BLOCK"):
    chunk = int(os.environ["KEPLER_BLOCK"], 0)
  elif N <= 1024:
    chunk = N
  else:
    chunk = 1024
  if chunk <= 0 or chunk > 1024:
    raise ValueError(f"invalid KEPLER_BLOCK={chunk}; expected 1..1024")
  n_chunks = (N + chunk - 1) // chunk
  _one_ib_max = int(os.environ.get("KEPLER_LAUNCH_ONE_IB_MAX", "19"), 0)
  _ib_max = int(os.environ.get("KEPLER_LAUNCH_IB_MAX", "16"), 0)
  if _ib_max <= 0:
    _ib_max = max(_one_ib_max, 1)
  _mcta_auto_max = int(os.environ.get("KEPLER_MULTI_CTA_AUTO_MAX",
                                      str(_n_max)), 0)
  _mcta = os.environ.get("KEPLER_MULTI_CTA", "auto").lower()
  if _mcta in ("1", "true", "yes", "on"):
    use_multi_cta = True
  elif _mcta in ("0", "false", "no", "off"):
    use_multi_cta = False
  else:
    # auto: one multi-CTA LAUNCH when N fits the proven range and divides
    # evenly by chunk (cubin has no i<N guard).  Else chunked / windows.
    use_multi_cta = (
        N <= _mcta_auto_max and n_chunks > 1 and (N % chunk) == 0)
  if use_multi_cta and (N % chunk) != 0:
    raise ValueError(
        f"KEPLER_MULTI_CTA requires N%{chunk}==0 (N={N}); "
        f"cubin has no bounds check")
  _seed = os.environ.get("KEPLER_SEED")
  if _seed is not None and _seed != "":
    random.seed(int(_seed, 0))
    print(f"[kepler] RNG seed={int(_seed, 0)}", flush=True)
  if use_multi_cta:
    print(f"[kepler] launch N={N} multi-CTA grid=({n_chunks},1,1) "
          f"block=({chunk},1,1)", flush=True)
  else:
    _n_wins = (n_chunks + _one_ib_max - 1) // _one_ib_max
    print(f"[kepler] launch N={N} chunks={n_chunks} chunk={chunk} "
          f"channel_windows={_n_wins} window_chunks={_one_ib_max} "
          f"(single-CTA each)", flush=True)
  test_stage = os.environ.get("KEPLER_TEST_STAGE", "full-add")
  if (not _no_window and not use_multi_cta and n_chunks > _one_ib_max
      and test_stage == "full-add"):
    # Build hosts once (same RNG as a single-shot run), then reopen the
    # device between ≤19-LAUNCH windows.
    if _hosts is not None:
      a_host, b_host = _hosts
    else:
      a_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
      b_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
    return _run_hardware_demo_windows(
        dev, cubin, a_host, b_host, chunk, _one_ib_max)
  if test_stage not in ("sem", "set-object", "full-add"):
    raise ValueError(
        f"unsupported KEPLER_TEST_STAGE={test_stage!r}; "
        "implemented stages are sem, set-object, and full-add")
  sem_only = test_stage == "sem"
  set_object_only = test_stage == "set-object"
  if cubin is None: cubin = get_kepler_cubin()
  prog = dev.runtime("E_4", cubin)
  allocator = NVAllocator(dev)

  if _hosts is not None:
    a_host, b_host = _hosts
    if len(a_host) != N or len(b_host) != N:
      raise ValueError(
          f"_hosts length {len(a_host)}/{len(b_host)} != KEPLER_N={N}")
  else:
    a_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
    b_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
  if os.environ.get("KEPLER_PRINT_IO", "0") != "0":
    _show = min(N, 16)
    print(f"[kepler] inputs a[0:{_show}]={[round(a_host[i], 4) for i in range(_show)]}",
          flush=True)
    print(f"[kepler] inputs b[0:{_show}]={[round(b_host[i], 4) for i in range(_show)]}",
          flush=True)

  a_dev = allocator.alloc(N * 4)
  b_dev = allocator.alloc(N * 4)
  out_dev = allocator.alloc(N * 4)
  signal = allocator.alloc(16)
  # Mesa TLS: CRS for 64 warps × live MP count (GTX 770=8, this 660 Ti=7).
  _mp_count = _gk104_mp_count_for(dev)
  _temp_size = _mp_count * GK104_TEMP_PER_MP
  print(f"[kepler] compute TLS: mp_count={_mp_count} temp_size={_temp_size:#x}",
        flush=True)
  # SET_OBJECT's firmware probe reads channel VA 0x100000.  A dedicated guard
  # PTE backs that leaf; if TLS lands at 0x80000 with size 0xe0000 it covers
  # 0x100000, live_pte_map fights the guard (wanted=guard actual=temp-mirror),
  # and pre-GP_PUT guard restore punches a hole in TLS — GPC1+GPC2 then stick
  # busy forever with NaN output.  Do not trust small-pad freelist cursor alone:
  # keep allocating TLS-sized buffers until the chosen VA is past 0x200000.
  _tls_pads = []
  _probe_lo, _probe_hi = 0x100000, 0x101000
  _tls_floor = 0x200000
  while True:
    temp_dev = allocator.alloc(_temp_size, align=0x10000)
    _t0, _t1 = temp_dev.va_addr, temp_dev.va_addr + temp_dev.size
    _covers_probe = _t0 < _probe_hi and _t1 > _probe_lo
    if not _covers_probe and _t0 >= _tls_floor:
      break
    _tls_pads.append(temp_dev)
    if len(_tls_pads) > 32:
      raise RuntimeError(
          f"failed to place TLS past SET_OBJECT probe: "
          f"last={_t0:#x}+{_temp_size:#x} pads={len(_tls_pads)}")
  print(f"[kepler] TLS va={temp_dev.va_addr:#x} "
        f"(discarded {len(_tls_pads)} overlapping/low TLS allocs)", flush=True)
  # Mesa txc heap: 2048 TIC + 2048 TSC entries = 128 KiB.  Must NOT land in the
  # GR scratch hole (pagepool @0x20000, bundle @0x28000) — a low hole alloc
  # would stomp those PTEs when mirrors are applied.
  _txc_spacers = []
  while True:
    txc_dev = allocator.alloc(0x20000, align=0x10000)
    if txc_dev.va_addr >= 0x100000:
      break
    _txc_spacers.append(txc_dev)
  allocator._copyin(txc_dev, bytes(0x20000))
  print(f"[kepler] TIC/TSC heap va={txc_dev.va_addr:#x} "
        f"(skipped {len(_txc_spacers)} low hole allocs)", flush=True)
  # Code + constant (param) buffers for the launch descriptor.
  sass = elf_section_bytes(cubin, ".text.E_4")
  code_dev = allocator.alloc(round_up(len(sass), 0x100))
  allocator._copyin(code_dev, sass)
  regs = cubin_register_count(cubin, "E_4")
  if os.environ.get("KEPLER_FORCE_REGS"):
    regs = int(os.environ["KEPLER_FORCE_REGS"], 0)
  cwd_devs = []
  cbuf_devs = []
  if use_multi_cta:
    cbuf = build_cuda_param_cbuf(
        a_dev.va_addr, b_dev.va_addr, out_dev.va_addr,
        grid=(n_chunks, 1, 1), block=(chunk, 1, 1))
    cbuf_dev = allocator.alloc(len(cbuf))
    allocator._copyin(cbuf_dev, cbuf)
    cwd = build_cwd(code_addr=0, grid=(n_chunks, 1, 1), block=(chunk, 1, 1),
                    cbuf_addr=cbuf_dev.va_addr, regs=regs,
                    cb7_addr=txc_dev.va_addr, cb7_size=0x800)
    cwd_dev = allocator.alloc(len(cwd))
    allocator._copyin(cwd_dev, cwd)
    cbuf_devs.append(cbuf_dev)
    cwd_devs.append(cwd_dev)
  else:
    for off in range(0, N, chunk):
      n = min(chunk, N - off)
      cbuf = build_cuda_param_cbuf(
          a_dev.va_addr + off * 4, b_dev.va_addr + off * 4,
          out_dev.va_addr + off * 4,
          grid=(1, 1, 1), block=(n, 1, 1))
      cbuf_dev = allocator.alloc(len(cbuf))
      allocator._copyin(cbuf_dev, cbuf)
      cwd = build_cwd(code_addr=0, grid=(1, 1, 1), block=(n, 1, 1),
                      cbuf_addr=cbuf_dev.va_addr, regs=regs,
                      cb7_addr=txc_dev.va_addr, cb7_size=0x800)
      cwd_dev = allocator.alloc(len(cwd))
      allocator._copyin(cwd_dev, cwd)
      cbuf_devs.append(cbuf_dev)
      cwd_devs.append(cwd_dev)
  # The one-shot demo has no prior GPU producer to acquire from.  Seed a
  # sentinel and require the GPU completion packet to replace it with 2.
  allocator._copyin(signal, struct.pack("<I", 1))
  # GPC instruction/data clients on this un-POSTed card do not reliably see
  # the SYS aperture even though HUB/PBDMA semaphore traffic does.  Put the
  # complete compute working set behind the already-validated VRAM VMM.  The
  # output mirror is read back after WFI; no host arithmetic is involved.
  dev._kepler_vram_mirrors = ([] if sem_only else
      [a_dev, b_dev, out_dev, temp_dev, txc_dev, code_dev, *cbuf_devs, *cwd_devs])
  allocator._copyin(a_dev, a_host.tobytes())
  allocator._copyin(b_dev, b_host.tobytes())
  # A GPU result must overwrite every lane; untouched VRAM stays non-finite.
  allocator._copyin(out_dev, struct.pack("<I", 0x7fc00001) * N)

  # Channel-windowed path guarantees ≤_one_ib_max CWDs per submit.  Same-channel
  # multi-IB does not lift the ~19 LAUNCH/channel ceiling.
  cwd_addrs = [c.va_addr for c in cwd_devs]
  if set_object_only:
    # SET_OBJECT leaves PGRAPH sticky-busy on this card; completion must use
    # RELEASE_WFI=DIS (bit20 → 0x00100002), not RELEASE_SIZE_4BYTE (bit24 →
    # 0x01000002) which was mislabeled as WFI=DIS and hung forever.
    words = [*nvm(1, 0x0000, KEPLER_COMPUTE_A),
             *gk104_semaphore(signal.va_addr, 2, GK104_SEM_RELEASE_NO_WFI),
             *nvm(0, 0x0020, 0)]
    launch_batches = [(words, 2)]
  else:
    # Channel-windowed path guarantees ≤_one_ib_max CWDs per submit.
    if len(cwd_addrs) > _one_ib_max:
      raise ValueError(
          f"internal: {len(cwd_addrs)} CWDs exceeds one-channel ceiling "
          f"{_one_ib_max}")
    words, launch_done = build_multi_launch_words(
        signal.va_addr, 2, cwd_addrs,
        code_dev.va_addr, temp_dev.va_addr, _temp_size,
        mp_count=_mp_count, tic_va=txc_dev.va_addr)
    launch_batches = [(words, launch_done)]
  # Verify compute buffers are mapped in the channel's VMM
  if dev.dev_impl.hw is not None:
    for _name, _buf in [("a_dev", a_dev), ("b_dev", b_dev), ("out_dev", out_dev),
                        ("code_dev", code_dev), ("cbuf0", cbuf_devs[0]),
                        ("cwd0", cwd_devs[0]), ("signal", signal),
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
    # H2 guard: SET_OBJECT hang clears GPC1/2 TPC_NR (c08→0) while sticky
    # GPC_STATUS=0x6.  Re-arm floorsweep TPC_NR from silicon 0x2608 right
    # before the push so eng-ctx load does not start with all-ones counts.
    # Disable with KEPLER_PRE_LAUNCH_FLOORSWEEP=0.
    if os.environ.get("KEPLER_PRE_LAUNCH_FLOORSWEEP", "1") != "0":
      _gpc_nr = dev.read32(0x409604) & 0x1f
      _fuse_tpc = _gk104_read_tpc_nr(dev, _gpc_nr)
      _ppc_masks = [
          ((1 << _fuse_tpc[g]) - 1) if _fuse_tpc[g] else 0
          for g in range(_gpc_nr)]
      # Prefer live PPC mask when forcing is off and silicon reports one.
      if not os.environ.get("KEPLER_FORCE_TPC_NR", "").strip():
        _ppc_masks = [(dev.read32(0x500c30 + g * 0x8000) & 0xff) or
                      _ppc_masks[g] for g in range(_gpc_nr)]
      _gk104_grctx_floorsweep(dev, _fuse_tpc, _ppc_masks)
      print(f"[kepler] pre-launch floorsweep re-armed: tpc={_fuse_tpc} "
            f"ppc={[hex(m) for m in _ppc_masks]}", flush=True)
    _pre_c08 = [dev.read32(0x500c08 + g * 0x8000) for g in range(4)]
    print(f"[kepler] pre-launch GPC TPC_NR: c08={[hex(x) for x in _pre_c08]} "
          f"405b00={dev.read32(0x405b00):#x} "
          f"GPC_STATUS={dev.read32(0x400604):#x}", flush=True)
  if DEBUG:
    pgd_idx = (signal.va_addr >> 27) & 0x1fff
    spt_idx = (signal.va_addr >> 12) & 0x7fff
    pgd_entry = dev.dev_impl.mm.root_page_table.entry(pgd_idx)
    spt = GK104PageTableEntry(dev.dev_impl, dev.dev_impl.mm.root_page_table.address(pgd_idx), 0)
    print(f"[kepler] vmm signal_va={signal.va_addr:#x} signal_pa={signal.meta['pa']:#x} "
          f"bus={dev.dev_impl.mm.bus_base:#x} pgd={pgd_entry:#x} pte={spt.entry(spt_idx):#x}")
  # Opt-in reclock before LAUNCH so GPC boost actually covers the kernel.
  # TinyGPU still allows BAR0 here; the armed final BAR1 read later does not.
  if not _no_window:
    _gk104_reclock_after_ok(dev)
  if sem_only:
    allocator._copyin(signal, struct.pack("<I", 0))
    # Host-only semaphore RELEASE without WFI: no engine methods precede this,
    # so a WFI would stall forever waiting for ENGINES!=0.  RELEASE_WFI=DIS
    # (NV906F bit20) lets the PBDMA write through the HUB VMM directly.
    # Keep the scheduler gate entirely inside PBDMA.  SET_REFERENCE on an
    # unbound subchannel raises EMPTY_SUBC/DEVICE before the semaphore packet.
    words = [*gk104_semaphore(signal.va_addr, 2, GK104_SEM_RELEASE_NO_WFI)]
    launch_batches = [(words, 2)]
  for _bi, (_words_or_ibs, launch_done) in enumerate(launch_batches):
    if isinstance(_words_or_ibs[0], list):
      ib_batches = _words_or_ibs
      words = ib_batches[0]
    else:
      words = _words_or_ibs
      ib_batches = [words]
    if _bi > 0:
      allocator._copyin(signal, struct.pack("<I", 1))
    submit_launch(dev, words, signal.va_addr, signal.meta['pa'], 1, launch_done,
                  ib_batches=ib_batches)

  if sem_only or set_object_only:
    # Prefix [kepler] so quiet-stdout filtering in main() keeps the line.
    print(f"[kepler] hardware_{test_stage}=ok value=2", flush=True)
    if dev.dev_impl.hw is not None:
      _s=dev.read32(0x400700); _b=dev.read32(0x40060c); _k=dev.read32(0x407020)
      _eng=dev.read32(0x2640); _mthd=dev.read32(0x409808); _tr=dev.read32(0x40981c)
      _ctxctl=dev.read32(0x409c00)
      _pgd=getattr(dev, "_kepler_cloned_by_pgd", None) or {}
      _pte100=0
      if 0 in _pgd:
        _pte100=_gk104_pramin_read32(dev, _pgd[0]+0x100*8)|(
            _gk104_pramin_read32(dev, _pgd[0]+0x100*8+4)<<32)
      print(f"[kepler] post-{test_stage}: STATUS2={_s:#x} ALL={bool(_s&1)} "
            f"GPC={bool(_s&0x1000000)} FE_BUSY={_b:#x} SKED={_k:#x} "
            f"GPC_STATUS={dev.read32(0x400604):#x} "
            f"FECS_EXC={dev.read32(0x409c18):#x} "
            f"INTR={dev.read32(0x400100):#x} TRAP={dev.read32(0x400108):#x}",
            flush=True)
      print(f"[kepler] post-{test_stage}-ctxsw: ENG={_eng:#x} "
            f"busy={bool(_eng&0x80000000)} chsw={bool(_eng&0x8000)} "
            f"FECS_MTHD={_mthd:#x} chsw_load={bool(_mthd&0x80000)} "
            f"TRACE={_tr:#x} CTXCTL={_ctxctl:#x} "
            f"CHAN={dev.read32(0x409b00):#x}/{dev.read32(0x409b04):#x} "
            f"PC={dev.read32(0x409ff0):#x} subch1={dev.read32(0x404204):#x} "
            f"PTE100000={_pte100:#x}", flush=True)
      # Discriminator: does GPC_STATUS ever drop after SET_OBJECT?
      _gpc_samples=[]
      for _i in range(20):
        _gpc_samples.append(dev.read32(0x400604))
        if _gpc_samples[-1]==0 and not (dev.read32(0x40060c)&1):
          break
        time.sleep(0.05)
      print(f"[kepler] post-{test_stage}-gpc-poll: samples={[hex(x) for x in _gpc_samples]} "
            f"FECS={dev.read32(0x409100):#x}@{dev.read32(0x409ff0):#x} "
            f"GPCCS={dev.read32(0x41a100):#x} MMIO_CTRL={dev.read32(0x409728):#x} "
            f"STATUS2={dev.read32(0x400700):#x}", flush=True)
      # H1 dump: decode FECS hub MMIO window + strand/mmctx while stuck.
      _mc = dev.read32(0x409728)
      _mmio_addr = _mc & 0x03fffffc
      _tgt = dev.read32(_mmio_addr) if _mmio_addr else 0
      print(f"[kepler] post-{test_stage}-mmio-strand: "
            f"CTRL={_mc:#x} ADDR={_mmio_addr:#x} "
            f"WR={'Y' if _mc & 0x40000000 else 'N'} "
            f"TRIG={'Y' if _mc & 0x80000000 else 'N'} "
            f"RDVAL={dev.read32(0x40972c):#x} "
            f"WRVAL={dev.read32(0x409730):#x} "
            f"BASE={dev.read32(0x409724):#x} "
            f"STRAND_STAT={dev.read32(0x409924):#x} "
            f"STRAND_CMD={dev.read32(0x409928):#x} "
            f"STRAND_WORDS={dev.read32(0x409910):#x} "
            f"STRAND_SEL={dev.read32(0x40991c):#x} "
            f"MMCTX={dev.read32(0x409710):#x} "
            f"MMCTX_Q={dev.read32(0x409720):#x} "
            f"LOAD_CNT={dev.read32(0x40974c):#x} "
            f"HUBSTAT={dev.read32(0x409c00):#x} "
            f"XFER_BIT={bool(dev.read32(0x409c00) & 0x2000)} "
            f"SIGNAL={dev.read32(0x409400):#x} "
            f"FE_PWR={dev.read32(0x404170):#x} "
            f"target_live={_tgt:#x}", flush=True)
      # H21: falcons idle but FE_BUSY/GPC_STATUS sticky — dump FE/BAR/RED and
      # try a host GO_IDLE ICMD.  If GPC drops, FE method pipeline was stuck.
      _bar_mask0 = dev.read32(0x40940c)
      _bar_mask1 = dev.read32(0x409410)
      _bar = dev.read32(0x409414)
      _hub_red = dev.read32(0x409614)
      print(f"[kepler] post-{test_stage}-fe-bar: "
            f"FE_BUSY={dev.read32(0x40060c):#x} "
            f"STATUS2={dev.read32(0x400700):#x} "
            f"PGRAPH_CTRL={dev.read32(0x400500):#x} "
            f"BAR_MASK0={_bar_mask0:#x} BAR_MASK1={_bar_mask1:#x} "
            f"BAR={_bar:#x} "
            f"HUB_RED={_hub_red:#x} "
            f"(EN_MAIN={bool(_hub_red & 0x100)} EN_GPC={bool(_hub_red & 0x200)} "
            f"EN_ROP={bool(_hub_red & 0x400)} "
            f"PWR_MAIN={bool(_hub_red & 0x10)} PWR_GPC={bool(_hub_red & 0x20)} "
            f"PWR_ROP={bool(_hub_red & 0x40)}) "
            f"UNK86C={dev.read32(0x40986c):#x} "
            f"408a14={dev.read32(0x408a14):#x}", flush=True)
      if (os.environ.get("KEPLER_MID_HANG_GO_IDLE", "1") != "0" and
          (_gpc_samples[-1] if _gpc_samples else 0) != 0):
        _gpc_before_gi = dev.read32(0x400604)
        _fe_before_gi = dev.read32(0x40060c)
        try:
          # Nouveau gf100_gr_icmd GO_IDLE bundle address ends in 0xe100.
          _gk104_gr_icmd(dev, [(0x400e100, 0)])
          _gi_err = None
        except Exception as _gie:
          _gi_err = f"{type(_gie).__name__}:{_gie}"
        _gpc_after_gi = []
        for _i in range(10):
          _gpc_after_gi.append(dev.read32(0x400604))
          if _gpc_after_gi[-1] == 0 and not (dev.read32(0x40060c) & 1):
            break
          time.sleep(0.05)
        print(f"[kepler] post-{test_stage}-go-idle: "
              f"GPC={_gpc_before_gi:#x}->{_gpc_after_gi[-1]:#x} "
              f"FE_BUSY={_fe_before_gi:#x}->{dev.read32(0x40060c):#x} "
              f"STATUS2={dev.read32(0x400700):#x} "
              f"ENG={dev.read32(0x2640):#x} "
              f"err={_gi_err} samples={[hex(x) for x in _gpc_after_gi]}",
              flush=True)
      # H2 mid-hang: eng-ctx load clears GPC1/2 TPC_NR (unk→0) while FECS parks
      # at SLEEPING@0x567 with MMIO_CTRL pending (LTC or GPC3+0x30e4).  Attempt
      # host floorsweep re-arm + FECS method nudge; if GPC_STATUS drops, H2 is
      # causal.  KEPLER_MID_HANG_GPC_REPAIR=0 disables.
      if (os.environ.get("KEPLER_MID_HANG_GPC_REPAIR", "1") != "0" and
          (_gpc_samples[-1] if _gpc_samples else 0) != 0):
        _gpc_nr = dev.read32(0x409604) & 0x1f
        _fuse = _gk104_read_tpc_nr(dev, _gpc_nr)
        _ppc = [
            ((1 << _fuse[g]) - 1) if _fuse[g] else 0
            for g in range(_gpc_nr)]
        if not os.environ.get("KEPLER_FORCE_TPC_NR", "").strip():
          _ppc = [(dev.read32(0x500c30 + g * 0x8000) & 0xff) or _ppc[g]
                  for g in range(_gpc_nr)]
        _mmio_before = dev.read32(0x409728)
        _fe_before = dev.read32(0x404170)
        # H17: eng-ctx can leave FE_PWR=0 (empty-mmio hang saw WRVAL=0x10 to
        # 0x404170 with live=0).  Re-assert FORCE_ON before floorsweep nudge.
        dev.write32(0x404170, 0x00000012)
        _gk104_grctx_floorsweep(dev, _fuse, _ppc)
        # Host-restore v0 only on GPCs that remain in the forced map.
        for _g in range(_gpc_nr):
          if not _fuse[_g]:
            continue
          _gb = 0x500000 + _g * 0x8000
          if (dev.read32(_gb + 0xc80) == 0 and
              (dev.read32(0x400604) & (1 << _g))):
            dev.write32(_gb + 0xc80, 0x20200004)
        _post_c08 = [dev.read32(0x500c08 + g * 0x8000) for g in range(4)]
        _gpc_after = []
        for _i in range(20):
          _gpc_after.append(dev.read32(0x400604))
          if _gpc_after[-1] == 0:
            break
          time.sleep(0.05)
        print(f"[kepler] post-{test_stage}-tpcnr-repair: "
              f"GPC_STATUS={_gpc_after[-1]:#x} c08={[hex(x) for x in _post_c08]} "
              f"ENG={dev.read32(0x2640):#x} "
              f"FECS={dev.read32(0x409100):#x}@{dev.read32(0x409ff0):#x} "
              f"CHAN={dev.read32(0x409b00):#x} "
              f"FE_PWR={_fe_before:#x}->{dev.read32(0x404170):#x} "
              f"MMIO_before={_mmio_before:#x} "
              f"MMIO_after={dev.read32(0x409728):#x} "
              f"samples={[hex(x) for x in _gpc_after]}", flush=True)
      _gpc_detail=[]
      for _g in range(4):
        _gb=0x500000+_g*0x8000
        _gpc_detail.append(
          f"GPC{_g}:fs={dev.read32(_gb+0xc30):#x} "
          f"unk={dev.read32(_gb+0xc08):#x} "
          f"red={dev.read32(0x502614+_g*0x8000):#x} "
          f"v0={dev.read32(_gb+0xc80):#x} "
          f"ctrl={dev.read32(0x502100+_g*0x8000):#x} "
          f"sc0={dev.read32(0x502800+_g*0x8000):#x}")
      print(f"[kepler] post-{test_stage}-gpc-detail: {' | '.join(_gpc_detail)} "
            f"HUB_RED={dev.read32(0x409614):#x} 405b00={dev.read32(0x405b00):#x} "
            f"ENG={dev.read32(0x2640):#x} IRQMASK={dev.read32(0x409018):#x} "
            f"ACCESS_EN={dev.read32(0x409048):#x}", flush=True)
      # H19: per-GPC GPCCS falcon state (0x502000+g*0x8000).  FECS is already
      # back in main wait@0x567; if GPC1/2 GPCCS are wedged mid-ctx_xfer while
      # GPC0/3 are idle, that is the sticky-GPC cause.
      _gpccs_lines = []
      for _g in range(4):
        _gb = 0x502000 + _g * 0x8000
        _gpccs_lines.append(
            f"GPC{_g}:CTRL={dev.read32(_gb+0x100):#x} "
            f"PC={dev.read32(_gb+0xff0):#x} "
            f"CPUSTAT={dev.read32(_gb+0x128):#x} "
            f"SCR0={dev.read32(_gb+0x800):#x} "
            f"SCR1={dev.read32(_gb+0x804):#x} "
            f"ENGSTAT={dev.read32(_gb+0xc00):#x} "
            f"MMIO={dev.read32(_gb+0x728):#x} "
            f"SIGNAL={dev.read32(_gb+0x400):#x} "
            f"INTR={dev.read32(_gb+0x008):#x} "
            f"RED={dev.read32(0x502614+_g*0x8000):#x}")
      print(f"[kepler] post-{test_stage}-gpccs: "
            f"HUB_GPCCS={dev.read32(0x41a100):#x}@{dev.read32(0x41aff0):#x} "
            f"{' || '.join(_gpccs_lines)}", flush=True)
      # H19b: TPC unit VSTATUS on busy GPCs (falcons were idle@0x50b).
      _tpc_lines = []
      for _g in range(4):
        if not (dev.read32(0x400604) & (1 << _g)):
          continue
        _nr = dev.read32(0x500000 + _g * 0x8000 + 0x2608) & 0x1f
        for _t in range(max(_nr, 1)):
          _tb = 0x504000 + _g * 0x8000 + _t * 0x800
          _tpc_lines.append(
              f"GPC{_g}.TPC{_t}:V0={dev.read32(_tb+0x600):#x} "
              f"V1={dev.read32(_tb+0x638):#x} "
              f"TRAP={dev.read32(_tb+0x648):#x}/"
              f"{dev.read32(_tb+0x650):#x}")
      if _tpc_lines:
        print(f"[kepler] post-{test_stage}-tpc-vstatus: "
              f"{' | '.join(_tpc_lines)}", flush=True)
      # H20: FECS/GPCCS idle but GPC_STATUS sticky — try channel preempt +
      # ctxctl IDLE trigger; if GPC_STATUS drops, eng-load latch is clearable.
      # Default OFF: live shot wedged FECS into ctx_xfer_idle (HUBSTAT IDLE_BUSY)
      # with KICK_CHID timeout; enable KEPLER_MID_HANG_PREEMPT=1 to retest.
      if (os.environ.get("KEPLER_MID_HANG_PREEMPT", "0") == "1" and
          (_gpc_samples[-1] if _gpc_samples else 0) != 0):
        _eng_before = dev.read32(0x2640)
        _gpc_before = dev.read32(0x400604)
        try:
          _chan = int(os.environ.get("KEPLER_CHAN_ID", "1"))
          _gk104_stop_preempt_channel(dev, _chan, timeout_s=0.5,
                                      label="set-object hang preempt")
          _preempt_err = None
        except Exception as _pe:
          _preempt_err = f"{type(_pe).__name__}:{_pe}"
        # ctxctl ENGINE_TRIGGER.IDLE (bit2)
        dev.write32(0x409c08, 0x00000004)
        time.sleep(0.05)
        _gpc_after_p = []
        for _i in range(10):
          _gpc_after_p.append(dev.read32(0x400604))
          if _gpc_after_p[-1] == 0:
            break
          time.sleep(0.05)
        print(f"[kepler] post-{test_stage}-preempt: "
              f"GPC={_gpc_before:#x}->{_gpc_after_p[-1]:#x} "
              f"ENG={_eng_before:#x}->{dev.read32(0x2640):#x} "
              f"HUBSTAT={dev.read32(0x409c00):#x} "
              f"FECS={dev.read32(0x409100):#x}@{dev.read32(0x409ff0):#x} "
              f"err={_preempt_err} samples={[hex(x) for x in _gpc_after_p]}",
              flush=True)
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
    # Host RELEASE is WFI=DIS (sticky PGRAPH busy after SET_OBJECT).  Poll the
    # VRAM output until the NaN sentinel clears or a settle budget expires so
    # we do not declare failure while SMs are still in flight.
    _out_pa = out_dev.meta["vram_pa"]
    _sent = struct.pack("<I", 0x7fc00001) * min(N, 8)
    _t0 = time.perf_counter()
    _settle_ms = float(os.environ.get("KEPLER_OUT_SETTLE_MS", "500"))
    _mirror_mode = os.environ.get("KEPLER_MIRROR_COPY", "pramin").strip().lower()
    def _vram_rd(pa, n):
      # Match mirror store path: real BAR0 PRAMIN for small buffers (H22).
      if (_mirror_mode == "pramin-all" or
          (_mirror_mode == "pramin" and
           n <= int(os.environ.get("KEPLER_MIRROR_PRAMIN_MAX", "0x10000"), 0))):
        buf = bytearray(n)
        for off in range(0, n, 4):
          struct.pack_into(
              "<I", buf, off,
              _gk104_pramin_read32(dev, pa + off, force_bar0=True))
        return bytes(buf)
      return bytes(dev.dev_impl.hw.mmio_read(1, pa, n))
    while True:
      _chunk = _vram_rd(_out_pa, min(len(out_host), 32))
      if _chunk[:len(_sent)] != _sent:
        break
      if (time.perf_counter() - _t0) * 1000.0 >= _settle_ms:
        break
      time.sleep(0.001)
    print(f"[kepler] output settle: waited_ms={(time.perf_counter()-_t0)*1000:.1f} "
          f"head={_chunk[:16].hex()} via={_mirror_mode}", flush=True)
    # Pre-freeze mirror/SKED dump (arm_final_output_read rejects further BAR1).
    try:
      def _b1(pa, n=32):
        return _vram_rd(pa, n)
      _code_pa = code_dev.meta.get("vram_pa")
      _cwd_pa = cwd_devs[0].meta.get("vram_pa")
      _a_pa = a_dev.meta.get("vram_pa")
      _cbuf_pa = cbuf_devs[0].meta.get("vram_pa")
      _sass = elf_section_bytes(cubin if cubin is not None else get_kepler_cubin(), ".text.E_4")[:16]
      _code_h = _b1(_code_pa)
      print(f"[kepler] mirror verify code_pa={_code_pa:#x} head={_code_h.hex()} "
            f"sass_match={_code_h[:16]==_sass}", flush=True)
      _cwd = _b1(_cwd_pa, 0x40)
      print(f"[kepler] mirror verify cwd_pa={_cwd_pa:#x} "
            f"w7={struct.unpack_from('<I', _cwd, 0x1c)[0]:#x} "
            f"w8={struct.unpack_from('<I', _cwd, 0x20)[0]:#x} "
            f"gridxy={struct.unpack_from('<II', _cwd, 0x30)}", flush=True)
      print(f"[kepler] mirror verify a_pa={_a_pa:#x} head={_b1(_a_pa).hex()}",
            flush=True)
      print(f"[kepler] mirror verify cbuf_pa={_cbuf_pa:#x} "
            f"params140={_b1(_cbuf_pa, 0x160)[0x140:0x158].hex()}", flush=True)
      print(f"[kepler] SKED=0x{dev.read32(0x407020):08x} "
            f"STATUS2=0x{dev.read32(0x400700):08x} "
            f"BUSY=0x{dev.read32(0x40060c):08x} "
            f"CTXCTL=0x{dev.read32(0x409c00):08x}", flush=True)
      # Channel PTE vs mirror PA: if SM used a stale SYS PTE, mirror stays sentinel.
      _pgd = getattr(dev, "_kepler_cloned_by_pgd", None) or {}
      for _nm, _buf in (("out", out_dev), ("code", code_dev), ("cbuf", cbuf_devs[0]),
                        ("cwd", cwd_devs[0])):
        _va = _buf.va_addr
        _spt = _pgd.get((_va >> 27) & 0x1fff)
        if _spt is None:
          print(f"[kepler] PTE {_nm}: no SPT", flush=True)
          continue
        _pte = struct.unpack_from("<Q", bytes(dev.dev_impl.hw.mmio_read(
            1, _spt + ((_va >> 12) & 0x7fff) * 8, 8)))[0]
        # GK104 leaf: ((pa >> 8) | valid|priv|...). Recover PA from bits 31:3.
        _pa = ((_pte & ~0x7) << 8) if (_pte & 1) else 0
        print(f"[kepler] PTE {_nm}: va={_va:#x} pte={_pte:#x} pa={_pa:#x} "
              f"mirror={_buf.meta.get('vram_pa'):#x} "
              f"match={_pa == (_buf.meta.get('vram_pa') or -1)}", flush=True)
      try:
        _sys = bytes(out_dev.cpu_view()[:32])
      except Exception:
        _sys = b""
      print(f"[kepler] out sysmem head={_sys[:16].hex() if _sys else 'n/a'} "
            f"mirror={_chunk[:16].hex()} same={_sys[:16]==_chunk[:16] if _sys else None}",
            flush=True)
      _mp=[]
      for _g in range(4):
        _tn=dev.read32(0x500000+_g*0x8000+0x2608)&0x1f
        if _tn > 8: continue
        for _ti in range(_tn):
          _wa,_ga=gk104_mp_trap_addrs(_g,_ti)
          _w,_gl=dev.read32(_wa),dev.read32(_ga)
          if _w or _gl:
            _mp.append(f"GPC{_g}.TPC{_ti}:w={_w:#x}/g={_gl:#x}")
      print(f"[kepler] MP traps: {_mp or 'none'}", flush=True)
    except Exception as _e:
      print(f"[kepler] mirror verify failed: {_e}", flush=True)
    # LAUNCH is followed by GRAPH_SERIALIZE; final host RELEASE is WFI=DIS.
    # QMD requests FE_SYSMEMBAR release ordering.  Prefer PRAMIN for the
    # output blob when mirrors used PRAMIN (H22); BAR1 path keeps TinyGPU's
    # arm_final_output_read one-shot.
    _out_n = len(out_host)
    _use_pramin_out = (
        _mirror_mode == "pramin-all" or
        (_mirror_mode == "pramin" and
         _out_n <= int(os.environ.get("KEPLER_MIRROR_PRAMIN_MAX", "0x10000"), 0)))
    if _use_pramin_out:
      out_host[:] = _vram_rd(out_dev.meta["vram_pa"], _out_n)
      print(f"[kepler] GPU output read from VRAM pa={out_dev.meta['vram_pa']:#x} "
            f"via=pramin", flush=True)
    else:
      dev.dev_impl.hw.arm_final_output_read(
          1, out_dev.meta["vram_pa"], len(out_host))
      out_host[:] = dev.dev_impl.hw.mmio_read(
          1, out_dev.meta["vram_pa"], len(out_host))
      print(f"[kepler] GPU output read from VRAM pa={out_dev.meta['vram_pa']:#x} "
            f"via=bar1", flush=True)
  else:
    allocator._copyout(out_host, out_dev)
  out_arr = array.array('f'); out_arr.frombytes(bytes(out_host))
  operation = os.environ.get("KEPLER_OPERATION", "add")
  expected = ([a_host[i] * b_host[i] for i in range(N)] if operation == "mul"
              else [a_host[i] + b_host[i] for i in range(N)])
  # Debug: show first few values
  _mismatches = sum(1 for i in range(N)
                    if not math.isfinite(out_arr[i]) or
                    abs(out_arr[i] - expected[i]) >= 1e-5)
  print(f"[kepler] output: first 8 actual={[round(out_arr[i],4) for i in range(min(8, N))]} "
        f"expected={[round(expected[i],4) for i in range(min(8, N))]} "
        f"mismatches={_mismatches}/{N}", flush=True)
  if _mismatches and os.environ.get("KEPLER_DUMP_MISMATCH", "0") != "0":
    bad = [i for i in range(N)
           if not math.isfinite(out_arr[i]) or
           abs(out_arr[i] - expected[i]) >= 1e-5]
    # Collapse to inclusive ranges.
    ranges = []
    lo = hi = bad[0]
    for i in bad[1:]:
      if i == hi + 1:
        hi = i
      else:
        ranges.append((lo, hi))
        lo = hi = i
    ranges.append((lo, hi))
    sent = struct.pack("<I", 0x7fc00001)
    n_sent = sum(1 for i in bad if struct.pack("<f", out_arr[i]) == sent)
    print(f"[kepler] mismatch ranges ({len(ranges)}): "
          f"{[(hex(a), hex(b), b - a + 1) for a, b in ranges[:16]]}"
          f"{'...' if len(ranges) > 16 else ''} "
          f"sentinel={n_sent}/{len(bad)}", flush=True)
  if os.environ.get("KEPLER_PRINT_IO", "0") != "0":
    _show = min(N, 16)
    print(f"[kepler] out[0:{_show}]={[round(out_arr[i], 4) for i in range(_show)]}",
          flush=True)
    print(f"[kepler] exp[0:{_show}]={[round(expected[i], 4) for i in range(_show)]}",
          flush=True)
  if _mismatches > 0:
    print(f"[kepler] raw output hex: {out_host[:32].hex()}", flush=True)
    print(f"[kepler] raw a_host hex: {a_host.tobytes()[:32].hex()}", flush=True)
    print(f"[kepler] raw b_host hex: {b_host.tobytes()[:32].hex()}", flush=True)
  if dev.dev_impl.hw is not None:
    dev.dev_impl.hw.freeze("output-read-complete")
  _freeze_stop_and_hold(dev, "output-read-complete")
  assert _mismatches == 0, f"hardware {operation} mismatch ({_mismatches}/{N} wrong)"
  print(f"hardware_demo=ok N={N} operation={operation}")
  _kt = getattr(dev, "_kepler_kernel_time_ms", None)
  if _kt is not None:
    print(f"[kepler] kernel_time_total_ms={float(_kt):.3f}", flush=True)

def _probe_hw_device():
  """Return a live PCI transport for probe flags (TinyGPU on macOS, sysfs on Linux)."""
  if OSX:
    return APLRemotePCIDevice.probe()
  return LinuxPCIDevice.probe()

# --- macOS TinyGPU probe / server helpers ---
def _probe():
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe: TinyGPU.app socket is not reachable (is the eGPU connected?)")
    raise SystemExit(1)
  try:
    boot0, meta = _gk104_ensure_bar0_mmio(dev)
    print(f"probe: PCI_ID={meta['id32']:#010x} "
          f"COMMAND={meta['command_before']:#06x}->{meta['command_after']:#06x} "
          f"mse_was={meta['mse_before']} reset={meta['did_reset']}")
    print(f"probe: PMC_BOOT_0=0x{boot0:08x} (chip_id={(boot0 >> 20) & 0xfff})")
  finally:
    dev.fini()

def _probe_post_ownership():
  """Read the inherited Nouveau POST boundary and exit without GPU writes."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-post-ownership: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    boot0, _meta = _gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
    snap = _gk104_post_entry_probe(BAR0Dev())
    if snap["boot0"] != boot0:
      raise RuntimeError("PMC_BOOT_0 changed during POST ownership probe")
    print("probe-post-ownership: " +
          ("READY for posted Night41h" if snap["night41h_ready"] else
           "NOT posted/PRAMIN-visible; do not repeat cold BAR1 run"))
  finally:
    dev.fini()

def _probe_rom_shadow_ownership():
  """Read Nouveau's inherited PRAMIN VBIOS-source predicates; write nothing."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-rom-shadow-ownership: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    _gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
    snap = _gk104_rom_shadow_entry_probe(BAR0Dev())
    print("probe-rom-shadow-ownership: " +
          ("READY: inherited Nouveau RAMIN source is present" if
           snap["firmware_shadow_ready"] else
           "MISSING: cold firmware RAMIN source is not present"))
  finally:
    dev.fini()

def _probe_golden_preinit():
  """Compare all seven safe pre-init golden reads without GPU writes."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-golden-preinit: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    _gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
    snap = _gk104_golden_preinit_entry_probe(BAR0Dev())
    print(f"probe-golden-preinit: mismatches="
          f"{[hex(reg) for reg in snap['mismatch_regs']]}")
  finally:
    dev.fini()

def _probe_option_rom_vga_preamble():
  """A/B the proven x86-ROM VGA prefix against immediate PRAMIN state."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-option-rom-vga-preamble: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    _gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
      def write32(self, r, v): dev.mmio_write32(0, r, v)
      def read8(self, r): return dev.mmio_read(0, r, 1)[0]
      def write8(self, r, v): dev.mmio_write(0, r, bytes((v & 0xff,)))
    bar0 = BAR0Dev()
    before = _gk104_post_entry_probe(bar0)
    image = nvbios_init.find_vbios_image(
        pathlib.Path(DEFAULT_VBIOS).read_bytes())
    nvbios_init.NvbiosInit(bar0, image).option_rom_vga_enable_prefix()
    after = _gk104_post_entry_probe(bar0)
    activated = bool(after["pramin_positive"] and
                     not before["pramin_positive"])
    print("probe-option-rom-vga-preamble: "
          f"before={before['pramin_word']:#010x} "
          f"after={after['pramin_word']:#010x} activated={activated}")
  finally:
    dev.fini()

def _probe_nouveau_init_io():
  """A/B the executed Palit INIT_IO special case before any other init."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-nouveau-init-io: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    _gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
      def write32(self, r, v): dev.mmio_write32(0, r, v)
      def read8(self, r): return dev.mmio_read(0, r, 1)[0]
      def write8(self, r, v): dev.mmio_write(0, r, bytes((v & 0xff,)))
    bar0 = BAR0Dev()
    before = _gk104_post_entry_probe(bar0)
    image = nvbios_init.find_vbios_image(
        pathlib.Path(DEFAULT_VBIOS).read_bytes())
    init = nvbios_init.NvbiosInit(bar0, image)
    if init.rd08(0x85bb) != 0x69:
      raise RuntimeError("Palit VBIOS INIT_IO opcode moved from 0x85bb")
    init.offset = 0x85bb
    init._op_io()
    if init.offset != 0x85c0:
      raise RuntimeError(f"INIT_IO ended at unexpected {init.offset:#x}")
    after = _gk104_post_entry_probe(bar0)
    activated = bool(after["pramin_positive"] and
                     not before["pramin_positive"])
    print("probe-nouveau-init-io: "
          f"before={before['pramin_word']:#010x} "
          f"after={after['pramin_word']:#010x} activated={activated}")
  finally:
    dev.fini()

def _probe_nouveau_base_lifecycle(*, bisect_post_scripts=False):
  """Bisect Nouveau's POST and base FB/RAM boundaries on one cold entry.

  ``bisect_post_scripts`` runs a *prefix* of BIT-I top-level scripts, then
  takes exactly one fixed-PA PRAMIN sample.  Night41t proved that retargeting
  ``0x1700`` between scripts leaves the same core NVINIT MMIO stream as
  Night41s but keeps fixed-PA ``0xfffe0000`` virgin; Night41s only sampled
  once at end-of-POST and saw data.  Mid-POST selector traffic is therefore
  forbidden.  Set ``KEPLER_POST_SCRIPT_PREFIX`` to the 1-based prefix length
  (1..N).  Optionally set ``KEPLER_NVINIT_STOP_OFFSET`` so the *last* script
  of the prefix stops before that top-level ROM offset (Night41x nested
  bisect of ``0x8fe8``).  One cold cycle tests one cut.

  Full lifecycle (no bisect): runs all BIT-I scripts, samples fixed-PA, then
  by default stops if POST already activated (H79 causal stop).  Set
  ``KEPLER_LIFECYCLE_THROUGH_RAM=1`` to continue into ``run_vbios_ram_init``
  and sample again (Night41ah preservation discriminator).  Set
  ``KEPLER_LIFECYCLE_THROUGH_LTC=1`` to also run ``_gk104_post_ram_fb_ltc``
  and sample again (Night41ai; implies through-ram).  Set
  ``KEPLER_LIFECYCLE_THROUGH_BAR=1`` to also run a one-page
  ``_gk104_init_bar1_identity`` and sample again (Night41aj; implies
  through-ltc). Optional ``KEPLER_LIFECYCLE_BAR_MAP_SIZE`` (default
  ``0x1000``).
  """
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-nouveau-base-lifecycle: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    _boot0, ensure_meta = _gk104_ensure_bar0_mmio(
        dev, allow_reset=False)
    if ensure_meta.get("did_reset"):
      raise RuntimeError("lifecycle probe forbids PCI reset")
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
      def write32(self, r, v): dev.mmio_write32(0, r, v)
      def read8(self, r): return dev.mmio_read(0, r, 1)[0]
      def write8(self, r, v): dev.mmio_write(0, r, bytes((v & 0xff,)))
    bar0 = BAR0Dev()
    before = _gk104_post_entry_probe(bar0)
    if (not _gk104_boot0_looks_live(before["boot0"]) or
        before["posted_marker"] or before["pramin_positive"]):
      raise RuntimeError(
          "lifecycle probe requires a live, unposted, PRAMIN-negative cold "
          f"entry (boot0={before['boot0']:#010x} "
          f"posted={before['posted_marker']} "
          f"pramin={before['pramin_word']:#010x}); replug without retry")
    # Night41ah+: KEPLER_LIFECYCLE_THROUGH_RAM=1 continues into RAMMAP even
    # when after-POST fixed-PA PRAMIN is already positive (H79 closed).  Strap
    # override stays forbidden on the bisect/causal-stop path; through-ram may
    # pin Palit strap 6 only after POST if 0x101000 is still unread.
    # Night41ai+: KEPLER_LIFECYCLE_THROUGH_LTC=1 continues past RAM into
    # Nouveau's post-RAM FB page + LTC init (implies through-ram).
    # Night41aj+: KEPLER_LIFECYCLE_THROUGH_BAR=1 continues past LTC into a
    # minimal BAR1 identity bootstrap (implies through-ltc).
    # Night41am+: KEPLER_LIFECYCLE_THROUGH_PMC=1 continues past BAR into
    # Nouveau nv50_mc_init-only PMC_ENABLE=0xffffffff (implies through-bar;
    # no PGOB — H93a discriminator).
    through_pmc = os.environ.get("KEPLER_LIFECYCLE_THROUGH_PMC", "0") == "1"
    through_bar = (os.environ.get("KEPLER_LIFECYCLE_THROUGH_BAR", "0") == "1" or
                   through_pmc)
    through_ltc = (os.environ.get("KEPLER_LIFECYCLE_THROUGH_LTC", "0") == "1" or
                   through_bar)
    through_ram = (os.environ.get("KEPLER_LIFECYCLE_THROUGH_RAM", "0") == "1" or
                   through_ltc)
    if (os.environ.get("KEPLER_RAMCFG_STRAP") not in (None, "") and
        not through_ram):
      raise RuntimeError(
          "lifecycle probe forbids KEPLER_RAMCFG_STRAP override; "
          "the post-POST live strap must select RAMCFG "
          "(set KEPLER_LIFECYCLE_THROUGH_RAM=1 to allow a post-POST pin)")
    def pramin_positive(words):
      return any(not _gk104_pramin_word_is_stub(word) and
                 word not in (0x00000000, 0xffffffff) for word in words)
    image, _bit_i, scripts = vbios_init_info(DEFAULT_VBIOS)
    os.environ.setdefault("KEPLER_VBIOS_I2C_TRACE", "1")
    if bisect_post_scripts:
      raw_prefix = os.environ.get("KEPLER_POST_SCRIPT_PREFIX", "").strip()
      if not raw_prefix:
        raise RuntimeError(
            "Night41t retired mid-POST 0x1700 sampling; set "
            "KEPLER_POST_SCRIPT_PREFIX to a 1-based script count "
            f"(1..{len(scripts)}) for one end-only fixed-PA sample")
      prefix = int(raw_prefix, 0)
      if prefix < 1 or prefix > len(scripts):
        raise RuntimeError(
            f"KEPLER_POST_SCRIPT_PREFIX={prefix} out of range 1..{len(scripts)}")
      raw_stop = os.environ.get("KEPLER_NVINIT_STOP_OFFSET", "").strip()
      stop_before = int(raw_stop, 0) if raw_stop else None
      chosen = scripts[:prefix]
      init = nvbios_init.NvbiosInit(bar0, image, debug=False)
      init.unlock_vga_crtc()
      for index, script in enumerate(chosen):
        is_last = index + 1 == len(chosen)
        if is_last and stop_before is not None:
          init.run_script(script, stop_before=stop_before)
          print(f"probe-nouveau-post-script-bisect: "
                f"ran script[{index}]={script:#06x} "
                f"stop_before={stop_before:#x} "
                f"(ended@{init.offset:#x}; no mid-POST PRAMIN sample)",
                flush=True)
        else:
          init.run_script(script)
          print(f"probe-nouveau-post-script-bisect: "
                f"ran script[{index}]={script:#06x} "
                f"(no mid-POST PRAMIN sample)", flush=True)
      label = (f"POST prefix={prefix} last={chosen[-1]:#06x}"
               + (f" stop_before={stop_before:#x}" if stop_before is not None
                  else ""))
      words = _gk104_pramin_stage_snapshot(
          bar0, label, pa=0xfffe0000)
      activated = pramin_positive(words)
      stop_s = (f"{stop_before:#x}" if stop_before is not None else "none")
      print("probe-nouveau-post-script-bisect: "
            f"prefix={prefix} last={chosen[-1]:#06x} "
            f"stop_before={stop_s} "
            f"activated={activated}; "
            "intentional causal stop before RAM "
            "(single end-of-prefix 0x1700 sample only)")
      return
    nvbios_init.run_vbios_init(bar0, image, scripts, debug=False)
    after_post = _gk104_pramin_stage_snapshot(
        bar0, "nouveau-measure after POST", pa=0xfffe0000)
    post_ok = pramin_positive(after_post)
    if post_ok and not through_ram:
      print("probe-nouveau-base-lifecycle: activated=after-post; "
            "intentional causal stop before RAM "
            "(set KEPLER_LIFECYCLE_THROUGH_RAM=1 to continue into RAMMAP)")
      return
    if through_ram:
      strap_reg = bar0.read32(0x101000)
      print(f"probe-nouveau-base-lifecycle: after-post "
            f"activated={post_ok}; through-ram; "
            f"0x101000={strap_reg:#010x}", flush=True)
      if ((strap_reg & 0x0000003c) == 0 and
          os.environ.get("KEPLER_RAMCFG_STRAP") in (None, "")):
        # Cold unread strap would select the wrong M0205/M0209 tables;
        # pin Palit golden strap 6 only for the RAMMAP phase.
        os.environ["KEPLER_RAMCFG_STRAP"] = "5"
        print("probe-nouveau-base-lifecycle: pinned "
              "KEPLER_RAMCFG_STRAP=5 for RAMMAP (0x101000 unread; 660 Ti)",
              flush=True)
    nvbios_init.run_vbios_ram_init(bar0, image, debug=False)
    after_ram = _gk104_pramin_stage_snapshot(
        bar0, "nouveau-measure after RAM", pa=0xfffe0000)
    ram_ok = pramin_positive(after_ram)
    after_ltc = None
    ltc_ok = False
    after_bar = None
    bar_ok = False
    bar1_dword = None
    bar1_ctl = None
    if through_ltc:
      if not ram_ok:
        print("probe-nouveau-base-lifecycle: through-ltc skipped; "
              "after-RAM fixed-PA not positive", flush=True)
      else:
        print("probe-nouveau-base-lifecycle: through-ltc; "
              "running post-RAM FB page + LTC", flush=True)
        _gk104_post_ram_fb_ltc(bar0)
        after_ltc = _gk104_pramin_stage_snapshot(
            bar0, "nouveau-measure after LTC", pa=0xfffe0000)
        ltc_ok = pramin_positive(after_ltc)
    if through_bar:
      if not ltc_ok:
        print("probe-nouveau-base-lifecycle: through-bar skipped; "
              "after-LTC fixed-PA not positive", flush=True)
      else:
        # Nouveau-shaped BAR1 identity: PRAMIN roots + 0x1704 enable.
        # Default one page (H89/H90); H91+ use KEPLER_LIFECYCLE_BAR_MAP_SIZE
        # up to 0x8000000 (128 MiB; SPT@1MiB).
        map_size = int(os.environ.get("KEPLER_LIFECYCLE_BAR_MAP_SIZE",
                                      "0x1000"), 0)
        print(f"probe-nouveau-base-lifecycle: through-bar; "
              f"BAR1 identity map_size={map_size:#x}", flush=True)
        _gk104_init_bar1_identity(
            bar0, mapped_size=map_size, map_vram=True)
        after_bar = _gk104_pramin_stage_snapshot(
            bar0, "nouveau-measure after BAR1", pa=0xfffe0000)
        bar_ok = pramin_positive(after_bar)
        bar1_ctl = bar0.read32(0x001704) & 0xffffffff
        # Identity VA→PA; sample page 0, mid, and last mapped page (H91).
        n_pages = max(map_size // 0x1000, 1)
        sample_pages = [0]
        if n_pages > 1:
          sample_pages.append(n_pages // 2)
        if n_pages > 2:
          sample_pages.append(n_pages - 1)
        sample_pages = sorted(set(sample_pages))
        # TinyGPU MMIO_READ needs prior MAP_BAR (bar_info); BAR0 already mapped
        # in _gk104_ensure_bar0_mmio — Night41aj skipped this for BAR1 (H90a).
        bar1_dword = None
        page0_dword = None
        multi_ok = True
        try:
          bar1_addr, bar1_size = dev.bar_info(1)
          print(f"probe-nouveau-base-lifecycle: MAP_BAR1 "
                f"addr={bar1_addr:#x} size={bar1_size:#x}", flush=True)
          for page in sample_pages:
            pa = page * 0x1000
            # Use offset-correct PRAMIN read (stage_snapshot only hits window base).
            pr = _gk104_pramin_read32(bar0, pa) & 0xffffffff
            raw = bytes(dev.mmio_read(1, pa, 4))
            b1 = struct.unpack("<I", raw)[0] if len(raw) == 4 else None
            hit = b1 is not None and b1 == pr
            multi_ok = multi_ok and hit
            if page == 0:
              page0_dword, bar1_dword = pr, b1
            print(f"probe-nouveau-base-lifecycle: page{page} "
                  f"PRAMIN={pr:#010x} BAR1={b1:#010x} match={hit}"
                  if b1 is not None else
                  f"probe-nouveau-base-lifecycle: page{page} "
                  f"PRAMIN={pr:#010x} BAR1=unreadable match=False",
                  flush=True)
        except Exception as e:
          multi_ok = False
          print(f"probe-nouveau-base-lifecycle: physical BAR1 "
                f"read failed: {e}", flush=True)
        page0_s = (f"{page0_dword:#010x}" if page0_dword is not None
                   else "None")
        bar1_s = (f"{bar1_dword:#010x}" if bar1_dword is not None
                  else "unreadable")
        print(f"probe-nouveau-base-lifecycle: 0x1704={bar1_ctl:#010x} "
              f"PRAMIN[PA0]={page0_s} BAR1[0]={bar1_s} "
              f"match={multi_ok} pages={sample_pages}",
              flush=True)
    topo_before_pmc = None
    topo_after_pmc = None
    pmc_before = None
    pmc_after = None
    if through_pmc:
      if not bar_ok:
        print("probe-nouveau-base-lifecycle: through-pmc skipped; "
              "after-BAR fixed-PA not positive", flush=True)
      else:
        # Nouveau nv50_mc_init / gk104_mc.init: wr32(0x000200, 0xffffffff).
        # H93a: isolate MC full enable from PGOB (gf100_gr_oneinit).
        pmc_before = bar0.read32(0x000200) & 0xffffffff
        topo_before_pmc = bar0.read32(0x409604) & 0xffffffff
        pgraph_before = bar0.read32(0x400000) & 0xffffffff
        print(f"probe-nouveau-base-lifecycle: through-pmc; "
              f"before PMC_ENABLE={pmc_before:#010x} "
              f"topo={topo_before_pmc:#010x} "
              f"PGRAPH={pgraph_before:#010x}", flush=True)
        bar0.write32(0x000200, 0xffffffff)
        _ = bar0.read32(0x000200)  # posting read
        pmc_after = bar0.read32(0x000200) & 0xffffffff
        topo_after_pmc = bar0.read32(0x409604) & 0xffffffff
        pgraph_after = bar0.read32(0x400000) & 0xffffffff
        fecs_scratch = bar0.read32(0x409800) & 0xffffffff
        ungated = (topo_after_pmc != 0xbadf1200 and
                   (topo_after_pmc & 0xffff0000) != 0xbadf0000)
        print(f"probe-nouveau-base-lifecycle: after PMC_ENABLE="
              f"{pmc_after:#010x} topo={topo_after_pmc:#010x} "
              f"PGRAPH={pgraph_after:#010x} "
              f"0x409800={fecs_scratch:#010x} ungated={ungated}",
              flush=True)
        after_pmc_fixed = _gk104_pramin_stage_snapshot(
            bar0, "nouveau-measure after PMC", pa=0xfffe0000)
        print(f"probe-nouveau-base-lifecycle: after-pmc fixed-PA "
              f"preserved={pramin_positive(after_pmc_fixed)}",
              flush=True)
    # Summarize furthest stage reached.
    if through_pmc and topo_after_pmc is not None:
      ungated = (topo_after_pmc != 0xbadf1200 and
                 (topo_after_pmc & 0xffff0000) != 0xbadf0000)
      stage = ("after-pmc-ungated" if ungated else "after-pmc-still-gated")
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} "
            f"ltc={[hex(word) for word in after_ltc]} "
            f"bar={[hex(word) for word in after_bar]} "
            f"pmc={pmc_after:#010x} topo={topo_after_pmc:#010x} "
            f"activated={stage}")
    elif through_bar and after_bar is not None:
      if post_ok and ram_ok and ltc_ok and bar_ok:
        stage = "after-bar-preserved"
      elif post_ok and ram_ok and ltc_ok and not bar_ok:
        stage = "after-bar-clobbered"
      elif bar_ok:
        stage = "after-bar"
      else:
        stage = "none"
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} "
            f"ltc={[hex(word) for word in after_ltc]} "
            f"bar={[hex(word) for word in after_bar]} activated={stage}")
    elif through_ltc and after_ltc is not None:
      if post_ok and ram_ok and ltc_ok:
        stage = "after-ltc-preserved"
      elif post_ok and ram_ok and not ltc_ok:
        stage = "after-ltc-clobbered"
      elif ltc_ok:
        stage = "after-ltc"
      elif post_ok and ram_ok:
        stage = "after-ram-preserved"
      elif ram_ok:
        stage = "after-ram"
      elif post_ok:
        stage = "after-post-clobbered"
      else:
        stage = "none"
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} "
            f"ltc={[hex(word) for word in after_ltc]} activated={stage}")
    else:
      if post_ok and ram_ok:
        stage = "after-ram-preserved"
      elif ram_ok:
        stage = "after-ram"
      elif post_ok:
        stage = "after-post-clobbered"
      else:
        stage = "none"
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} activated={stage}")
  finally:
    dev.fini()

def _tinygpu_sock_reachable(sock_path=None, timeout_s=0.2):
  sock_path = sock_path or _temp_sock()
  try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    s.connect(sock_path)
    s.close()
    return True
  except OSError:
    return False

def _ensure_tinygpu_server():
  """Start TinyGPU.app's shared server if /tmp/tinygpu.sock is missing.

  Opt out with KEPLER_ENSURE_TINYGPU=0.  The server is left running for later
  invocations (matches the shared-socket contract in plan.md).
  """
  if os.environ.get("KEPLER_ENSURE_TINYGPU", "1") == "0":
    return
  sock_path = os.environ.get("APL_REMOTE_SOCK", _temp_sock())
  if _tinygpu_sock_reachable(sock_path):
    return
  app = APLRemotePCIDevice.APP_PATH
  if not os.path.isfile(app):
    print(f"[kepler] TinyGPU binary missing at {app}; start the server manually",
          flush=True)
    return
  # Stale socket file with no listener → remove so bind succeeds.
  try:
    if os.path.exists(sock_path) and not _tinygpu_sock_reachable(sock_path):
      os.unlink(sock_path)
  except OSError:
    pass
  log_path = "/tmp/tinygpu-server.log"
  with open(log_path, "ab", buffering=0) as logf:
    subprocess.Popen(
        [app, "server", sock_path],
        stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True)
  for _ in range(50):
    if _tinygpu_sock_reachable(sock_path):
      print(f"[kepler] started TinyGPU server at {sock_path}", flush=True)
      return
    time.sleep(0.1)
  print(f"[kepler] TinyGPU server did not become ready at {sock_path}; "
        f"see {log_path}", flush=True)

def main():
  # The verified GTX 770 sequence needs FIFO reset; retain an environment
  # override for diagnosis, but make the known-good behavior the default.
  os.environ.setdefault("KEPLER_FIFO_RESET", "1")
  if OSX:
    set_pci_transport_factory(_MacPCIDeviceFactory())
    os.environ.setdefault("KEPLER_NO_AUTO_SUDO", "1")
    # Dispatch macOS-only cold-lifecycle probes before the shared launcher.
    if "--probe-nouveau-base-lifecycle" in sys.argv:
      _probe_nouveau_base_lifecycle(); return
    if "--probe-nouveau-post-script-bisect" in sys.argv:
      _probe_nouveau_base_lifecycle(bisect_post_scripts=True); return
    if "--probe-option-rom-vga-preamble" in sys.argv:
      _probe_option_rom_vga_preamble(); return
    if "--probe-nouveau-init-io" in sys.argv:
      _probe_nouveau_init_io(); return
  # Offline goldens still pin Palit GTX770 strap-6 RAMMAP tables.  Live 660 Ti
  # on this enclosure reads 0x101000=0x80405096 → RAMCFG group 5; pinning strap
  # 6 there trains the wrong M0205/M0209 tables and shows up as GP_PUT-consumed
  # / semaphore-never-done timeouts (same signature as wrong mp_count TEMP).
  _offline = ("--middle-selftest" in sys.argv or
              "--mmiotrace-selftest" in sys.argv)
  if _offline:
    # Offline goldens are Palit strap-6; ignore any live strap leftover in env.
    os.environ["KEPLER_RAMCFG_STRAP"] = "6"
  else:
    os.environ.setdefault("KEPLER_RAMCFG_STRAP", "5")
    print("[kepler] add_660ti: KEPLER_RAMCFG_STRAP="
          f"{os.environ.get('KEPLER_RAMCFG_STRAP')} "
          "(660 Ti live group; override if your 0x101000 differs)",
          flush=True)
  # MEMX INFO+WR32 work.  Shared default skips pause (golden mmiotrace / Linux).
  # macOS TinyGPU wrapper overrides to bit0-only host pause.
  os.environ.setdefault("KEPLER_PMU_MEMX", "1")
  os.environ.setdefault("KEPLER_PMU_ENTER_NOWAIT", "1")
  os.environ.setdefault("KEPLER_RAM_BLOCK", "0")
  os.environ.setdefault("KEPLER_RAM_MEMX_WR", "1")
  if "--middle-selftest" in sys.argv:
    # Offline fake register buses intentionally have no PMU Falcon.  Keep the
    # live default strict, but allow the documented selftest command to use
    # the host-write golden model (same policy as the macOS wrapper).
    os.environ.setdefault("KEPLER_RAM_REQUIRE_MEMX", "0")
    kepler_selftest()
    return
  if "--mmiotrace-selftest" in sys.argv:
    # Offline golden-mmiotrace gate — no hardware / pagemap.  Run this on
    # macOS before the next eGPU replug.
    os.environ.setdefault("KEPLER_RAM_REQUIRE_MEMX", "0")
    sys.path.insert(0, SHARED_KEPLER_DIR)
    import mmiotrace_selftest as _mmio_st
    raise SystemExit(_mmio_st.run_mmiotrace_selftest(
        _mmio_st.build_hooks_from_add_module(sys.modules[__name__])))
  if "--compare-cubin" in sys.argv:
    operation = os.environ.get("KEPLER_OPERATION", "add")
    try:
      assembled = assemble_kepler_cubin(operation)
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
      print(f"cubin_compare=assembler-unavailable operation={operation} reason={e}")
      return
    if operation == "add":
      with open(DEFAULT_CUBIN, "rb") as f: expected = f.read()
      same = assembled == expected
      ref_bytes = len(expected)
    else:
      same = (len(assembled) == MUL_CUBIN_BYTES and
              hashlib.sha256(assembled).hexdigest() == MUL_CUBIN_SHA256)
      ref_bytes = MUL_CUBIN_BYTES
    print(f"cubin_compare={'byte-identical' if same else 'mismatch'} "
          f"operation={operation} assembled_bytes={len(assembled)} reference_bytes={ref_bytes}")
    return
  if "--vbios-info" in sys.argv:
    i = sys.argv.index("--vbios-info")
    path = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else \
      DEFAULT_VBIOS
    try:
      inspect_vbios(path)
    except (OSError, ValueError) as e:
      print(f"vbios-info: {e}", file=sys.stderr)
      sys.exit(1)
    return
  if "--vbios-init-info" in sys.argv:
    i = sys.argv.index("--vbios-init-info")
    path = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else \
      DEFAULT_VBIOS
    try:
      vbios_init_info(path)
    except (OSError, ValueError, struct.error) as e:
      print(f"vbios-init-info: {e}", file=sys.stderr)
      sys.exit(1)
    return
  live_probe_flags = {
    "--probe", "--probe-post-ownership", "--probe-rom-shadow-ownership",
    "--probe-golden-preinit", "--probe-pgob", "--probe-gpc-clock",
    "--probe-vbios-devinit", "--probe-falcon",
  }
  backend = os.environ.get("NV_BACKEND", "kepler")
  offline = any(x in sys.argv for x in (
      "--middle-selftest", "--mmiotrace-selftest",
      "--vbios-info", "--vbios-init-info", "--compare-cubin"))
  if OSX and not offline:
    # Proven TinyGPU classic BAR1 + literal PRAMIN path (Night41ay/bc).
    os.environ.setdefault("KEPLER_POST_BEFORE_PMC", "1")
    os.environ.setdefault("KEPLER_TINYGPU_ATOMIC_BAR1", "0")
    os.environ.setdefault("KEPLER_PRAMIN_MEMX", "0")
    os.environ.setdefault("KEPLER_PRAMIN_LITERAL", "1")
    os.environ.setdefault("KEPLER_PRAMIN_LITERAL_FIRST", "1")
    os.environ.setdefault("KEPLER_RAM_BIT0_DEFER", "0")
    os.environ.setdefault("KEPLER_PMU_MEMX", "0")
    os.environ.setdefault("KEPLER_RAM_PROGRAM", "0")
    os.environ.setdefault("KEPLER_USERD_ALIAS", "0")
    os.environ.setdefault("KEPLER_RAM_MEMX_ATOMIC", "0")
    os.environ.setdefault("KEPLER_RAM_ENTER_WAIT", "1")
    os.environ.setdefault("KEPLER_RAM_ATOMIC_PREFLIGHT", "0")
    os.environ.setdefault("KEPLER_RAM_BLOCK", "0")
    os.environ.setdefault("KEPLER_RAM_REQUIRE_MEMX", "0")
    os.environ.setdefault("KEPLER_PRAMIN_SOFT_LIVE", "1")
    os.environ.setdefault("KEPLER_REFUSE_DIRTY", "1")
    os.environ.setdefault("KEPLER_POST_RAM_LTC", "1")
    os.environ.setdefault("KEPLER_PGRAPH_BLCG", "0")
    os.environ.setdefault("KEPLER_PGRAPH_PACK", "1")
    os.environ.setdefault("KEPLER_BAR1_MEMX_LITERAL", "1")
    os.environ.setdefault("KEPLER_BAR1_DIRECT_PHYS", "1")
    os.environ.setdefault("KEPLER_BAR1_MAP_SIZE", "0x8000000")
    os.environ.setdefault("KEPLER_RAM_HOST_PROG0", "0")
    os.environ.setdefault("KEPLER_AUTO_WARM_CONTINUE", "1")
    os.environ.setdefault("KEPLER_RPC_LIGHT", "1")
    os.environ.setdefault("KEPLER_FAST_ZERO", "1")
    os.environ.setdefault("KEPLER_LIVE_ACK", "completion-abort-risk")
    os.environ.setdefault("KEPLER_RPC_TRACE",
                          os.path.join(REPO_ROOT, "logs", "kepler_rpc.jsonl"))
    # Do NOT setdefault KEPLER_VBIOS to Palit — cold path must dump/select
    # onboard PROM via _gk104_resolve_vbios_path.
    os.environ.setdefault("KEPLER_N", "8")
    os.environ.setdefault("KEPLER_SEED", "42")
    if os.environ.get("KEPLER_RAM_PROGRAM") == "bit0-only":
      if os.environ.get("KEPLER_RAM_AFTER_BIT0") != "memx":
        os.environ.setdefault("KEPLER_RAM_INIT", "0")
    if backend != "software":
      if os.environ.get("KEPLER_LIVE_ACK") not in (
          "completion-abort-risk", "1", "yes", "true"):
        raise SystemExit(
            "hardware launch refused: set KEPLER_LIVE_ACK=completion-abort-risk "
            "(or unset it — that is now the live default) for an authorized "
            "TinyGPU test; set KEPLER_LIVE_ACK=0 to keep refusing")
      if not os.environ.get("KEPLER_RPC_TRACE"):
        raise SystemExit("hardware launch refused: KEPLER_RPC_TRACE is required")
      os.makedirs(os.path.dirname(os.path.abspath(
          os.environ["KEPLER_RPC_TRACE"])) or ".", exist_ok=True)
      _ensure_tinygpu_server()
  needs_hardware = backend != "software" and (bool(live_probe_flags.intersection(sys.argv)) or
                                                "--middle-selftest" not in sys.argv)
  if (needs_hardware and os.environ.get("KEPLER_NO_AUTO_SUDO") != "1" and
      hasattr(os, "geteuid") and os.geteuid() != 0):
    # Raw sysfs BAR mmap normally requires root.  Re-exec the exact interpreter
    # so `python3 file.py` is the only command the user has to remember.
    print("[kepler] raw PCIe access needs privilege; requesting sudo...", flush=True)
    os.execvp("sudo", ["sudo", "--", sys.executable, os.path.abspath(__file__), *sys.argv[1:]])
  if "--probe" in sys.argv:
    if OSX:
      _probe(); return
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
  if "--probe-post-ownership" in sys.argv:
    if OSX:
      _probe_post_ownership(); return
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe-post-ownership: GK104 not reachable (is the card unbound?)")
      sys.exit(1)
    d.bar_info(0)
    class BAR0Dev:
      def read32(self, r): return d.mmio_read32(0, r)
    try:
      snap = _gk104_post_entry_probe(BAR0Dev())
      print("probe-post-ownership: " +
            ("READY for posted Night41h" if snap["night41h_ready"] else
             "NOT posted/PRAMIN-visible; do not repeat cold BAR1 run"))
    finally:
      d.fini()
    return
  if "--probe-rom-shadow-ownership" in sys.argv:
    if OSX:
      _probe_rom_shadow_ownership(); return
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe-rom-shadow-ownership: GK104 not reachable (is the card unbound?)")
      sys.exit(1)
    d.bar_info(0)
    class BAR0Dev:
      def read32(self, r): return d.mmio_read32(0, r)
    try:
      snap = _gk104_rom_shadow_entry_probe(BAR0Dev())
      print("probe-rom-shadow-ownership: " +
            ("READY: inherited Nouveau RAMIN source is present" if
             snap["firmware_shadow_ready"] else
             "MISSING: cold firmware RAMIN source is not present"))
    finally:
      d.fini()
    return
  if "--probe-golden-preinit" in sys.argv:
    if OSX:
      _probe_golden_preinit(); return
    d = LinuxPCIDevice.probe()
    if d is None:
      print("probe-golden-preinit: GK104 not reachable (is the card unbound?)")
      sys.exit(1)
    d.bar_info(0)
    class BAR0Dev:
      def read32(self, r): return d.mmio_read32(0, r)
    try:
      snap = _gk104_golden_preinit_entry_probe(BAR0Dev())
      print(f"probe-golden-preinit: mismatches="
            f"{[hex(reg) for reg in snap['mismatch_regs']]}")
    finally:
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
      DEFAULT_VBIOS
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
  cubin = None
  if backend != "software":
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
  # The bring-up contains intentionally verbose register diagnostics. Keep
  # normal live launches readable; DEBUG=1 remains the opt-in trace mode (and
  # is rejected above for hardware crash-isolation runs).
  quiet_buf = io.StringIO() if backend != "software" else None
  quiet_ctx = contextlib.redirect_stdout(quiet_buf) if quiet_buf is not None else contextlib.nullcontext()
  init_err = None
  run_err = None
  dev = None

  def _emit_quiet_diagnostics(prefix="[kepler]", also=("hardware_demo=", "[nvbios]")):
    if quiet_buf is None:
      return
    for line in quiet_buf.getvalue().splitlines():
      if line.startswith(prefix) or any(line.startswith(a) for a in also):
        print(line, flush=True)

  def _one_live_pass():
    nonlocal init_err, run_err, dev, quiet_buf, quiet_ctx
    init_err = None
    run_err = None
    dev = None
    if quiet_buf is not None:
      quiet_buf = io.StringIO()
      quiet_ctx = contextlib.redirect_stdout(quiet_buf)
    with quiet_ctx:
      try:
        dev = NVDevice("NV", backend=backend)
      except (NotImplementedError, OSError, RuntimeError) as e:
        init_err = e
      if init_err is None and dev is not None:
        try:
          if backend == "software":
            run_software_demo(dev)
          else:
            run_hardware_demo(dev, cubin)
        except Exception as e:
          run_err = e
        finally:
          try:
            dev.close()
          except Exception:
            pass

  _one_live_pass()

  # Night41ay/bc: cold first demo often returns NaN after FECS is already
  # ready; a second open with ALLOW_DIRTY finishes add. One user-visible shot.
  # Do NOT warm-continue after BAR1 aperture death — that needs a power cycle.
  if (run_err is not None and backend != "software" and
      os.environ.get("KEPLER_AUTO_WARM_CONTINUE", "1") != "0"):
    _err_s = str(run_err)
    _bar1_dead = any(s in _err_s for s in (
        "BAR1 VRAM stub", "BAR1 top mismatch", "framebuffer aperture is dead",
        "refuses Palit GTX770 VBIOS"))
    if _bar1_dead:
      print("[kepler] skipping warm-continue after BAR1/VBIOS failure "
            f"({type(run_err).__name__}: {run_err})", flush=True)
    else:
      _emit_quiet_diagnostics()
      print("[kepler] cold demo failed; auto warm-continue once "
            f"({type(run_err).__name__}: {run_err})", flush=True)
      os.environ["KEPLER_ALLOW_DIRTY"] = "1"
      os.environ.pop("KEPLER_FORCE_RELOAD", None)
      _one_live_pass()

  # Emit outside redirect_stdout so diagnostics reach the real terminal/log.
  if init_err is not None:
    _emit_quiet_diagnostics()
    print(f"hardware initialization failed safely: {init_err}", file=sys.stderr)
    print("No automatic probe/retry was made; confirm the GK104 is unbound from "
          "the nvidia driver and that you are root before another isolated run.",
          file=sys.stderr)
    sys.exit(2)
  if run_err is not None:
    _emit_quiet_diagnostics()
    raise run_err
  if quiet_buf is not None:
    _keep = (
        "[kepler] output:",
        "hardware_demo=",
        "hardware_sem=",
        "hardware_set-object=",
        "[kepler] GR ctx runtime physical-zero",
        "[kepler] GR ctx golden physical-zero",
        "[kepler] attrib physical-zero",
        "[kepler] BAR1 top R/W",
        "[kepler] Nouveau-order BAR1",
        "[kepler] reclock-after-ok",
        "[kepler] experimental pstate",
        "[kepler] gk104_clk_prog",
        "[kepler] clk before experimental",
        "[kepler] clk after experimental",
        "[kepler] BAR1 after POST",
        "[kepler] BAR1 identity clamped",
        "[kepler] ctx copy via",
        "[kepler] PTE stabilize:",
        "[kepler] launch N=",
        "[kepler] channel window",
        "[kepler] kernel_time_ms=",
        "[kepler] kernel_time_total_ms=",
        "[kepler] hardware_sem=",
        "[kepler] hardware_set-object=",
        "[kepler] post-sem:",
        "[kepler] post-set-object:",
        "[kepler] post-set-object-ctxsw:",
        "[kepler] post-set-object-gpc-poll:",
        "[kepler] post-set-object-gpc-detail:",
        "[kepler] post-sem-ctxsw:",
        "[kepler] post-sem-gpc-poll:",
        "[kepler] post-sem-gpc-detail:",
        "[kepler] output settle:",
        "[kepler] mirror verify",
        "[kepler] PTE ",
        "[kepler] out sysmem",
        "[kepler] MP traps:",
        "[kepler] SKED=",
        "[kepler] raw output hex:",
        "[kepler] compute TLS:",
        "[kepler] TLS va=",
        "[kepler] TIC/TSC heap",
        "[kepler] VRAM mirror:",
        "[kepler] live MP/TPC",
        "[kepler] GPCCS started",
        "[kepler] FECS not ready",
        "[kepler] FECS warm-keep",
        "[kepler] FECS_MMIO_CTRL",
        "[kepler] FECS discover_image_size",
        "[kepler] GR ctx image size",
        "[kepler] after FECS start",
        "[kepler] loading+starting",
        "[kepler] FECS golden save done",
        "[kepler] FECS IRQMASK",
        "[kepler] golden buf scan:",
        "[kepler] runtime GR ctx:",
        "[kepler] grctx topology:",
        "[kepler] grctx GPC TPC_NR:",
        "[kepler] LTC ctx preserve",
        "[kepler] bit19-safe attrib shrink:",
        "[kepler] grctx attrib consts:",
        "[kepler] grctx PPC+0xe4:",
        "[kepler] PPC mmio-list:",
        "[kepler] MMIO list emptied",
        "[kepler] LTC mmio-list:",
        "[kepler] FECS discover_image_size",
        "[kepler] GR ctx size override:",
        "[kepler] pre-launch floorsweep re-armed:",
        "[kepler] FORCE_TPC_NR:",
        "[kepler] pre-launch GPC TPC_NR:",
        "[kepler] post-set-object-tpcnr-repair:",
        "[kepler] post-set-object-mmio-strand:",
        "[kepler] post-set-object-gpccs:",
        "[kepler] post-set-object-tpc-vstatus:",
        "[kepler] post-set-object-preempt:",
        "[kepler] grctx_main GPC/TPC:",
        "[kepler] grctx_main:",
    )
    for line in quiet_buf.getvalue().splitlines():
      if any(line.startswith(p) for p in _keep):
        print(line, flush=True)

if __name__ == "__main__":
  main()
