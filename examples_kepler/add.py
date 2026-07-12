#!/usr/bin/env python3
"""Standalone NV stack for a GTX 770 eGPU (Kepler, GK104, sm_30) over PCIe/USB4.

================================================================================
IMPORTANT — KEPLER BRING-UP STATUS (read before running the live path)
================================================================================
The GA102 `examples/add.py` (RTX 3080) works *only* because that GPU has a GSP
(GPU System Processor): GSP firmware runs NVIDIA's Resource Manager and we drive
it through an RPC queue.  Kepler GK104 has **NO GSP**.  There is no firmware that
runs the RM for us.  The entire RM must be reimplemented in userspace as raw
MMIO / register programming — this is exactly what nouveau's `nvkm` does in
~50k lines (FALCON PMU bring-up, GR context-switch bundles, FIFO channels,
GMMU page tables, the compute work descriptor launch).

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

NO imports from tinygrad.runtime.support / ops / device / renderer / uop / helpers
are permitted on the live path — those have been vendored inline below.
"""
from __future__ import annotations
import os, sys, ctypes, ctypes.util, time, mmap, struct, array as _array_mod, socket, subprocess, contextlib, functools, itertools, enum, atexit, select, dataclasses, collections, urllib.request, hashlib, tempfile, gzip, pathlib
from typing import cast, Any, ClassVar, Generic, TypeVar

# --- autogen ctypes (allowed: "ctypes constants only") ---
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu, pci
from tinygrad.runtime.autogen import nv_regs
from tinygrad.runtime.autogen import libc
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
  PTE_PDE_4K = 2 << 32         # HOST small-page table in a GF100 PDE
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
      return ((entry >> 36) << 12) - bus
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
        # GF100_PDE: SPT present at bit 32, HOST target at bits 33:34,
        # and SPT address at bits 36:63 (physical address << 24).
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

def elf_section_bytes(blob, wanted):
  """Return one ELF64 section payload by name (used to upload SASS, not ELF)."""
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
      return bytes(blob[h[4]:h[4] + h[5]])
  raise KeyError(wanted)


# ============================================================================
# PCI interface — Kepler uses the SAME PCIe transport as GA102 (TinyGPU.app on
# macOS, /dev/nvidiactl on Linux).  The difference is entirely in what we do
# with the BARs afterwards: no GSP, direct MMIO + FALCON.
# ============================================================================
PAGESIZE = 0x1000

class RemoteCmd(enum.IntEnum):
  # Wire command IDs spoken by TinyGPU.app's DriverKit extension, as decoded by
  # TheTom/pascal-egpu's TinyGPUClient (which successfully reads PMC_BOOT_0 over
  # this protocol on a real eGPU).  Request: struct.pack('<BIIQQQ', cmd, dev_id,
  # bar, *args3); response: struct.unpack('<QQB', ...) = (value1, value2, status)
  # followed by `readout` payload bytes when present.
  MAP_BAR       = 1
  MAP_SYSMEM_FD = 2
  SYSMEM_READ   = 9
  SYSMEM_WRITE  = 10
  MMIO_READ     = 6
  MMIO_WRITE    = 7
  MAP_SYSMEM    = 8

class RemotePCIDevice:
  """Abstract transport for a remote PCIe GPU (TinyGPU socket, vfio, ...)."""
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
  return os.path.join(tempfile.mkdtemp(prefix="tinygpu_"), "tinygpu.sock")

class APLRemotePCIDevice(RemotePCIDevice):
  """macOS: TinyGPU.app signed DriverKit extension exposes raw PCIe BAR access
  for an eGPU over a local Unix socket.  This is a faithful port of the proven
  client in examples/add.py (which drives this same hardware), adapted for
  Kepler bring-up.  The server is auto-started from /Applications/TinyGPU.app
  if not already running, exactly like examples/add.py."""
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"
  APP_COMMIT = "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"

  def __init__(self, name="NV", transport=None, dev_id=0, sock_path=None, timeout_ms=2000):
    super().__init__(name, transport or "usb4")
    self.dev_id = dev_id
    self.sock_path = sock_path or os.environ.get("APL_REMOTE_SOCK", _temp_sock())
    self._sock = None
    self._connect(timeout_ms)

  def _connect(self, timeout_ms):
    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self._sock.settimeout(timeout_ms / 1000.0)
    connected = False
    for i in range(100):
      try:
        self._sock.connect(self.sock_path); connected = True; break
      except (ConnectionRefusedError, FileNotFoundError):
        if i == 0:
          subprocess.Popen([self.APP_PATH, "server", self.sock_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
    if not connected:
      raise RuntimeError(f"Failed to connect to TinyGPU server at {self.sock_path}")

  def _recvall(self, n):
    buf = bytearray(n)
    got = 0
    while got < n:
      cnt = self._sock.recv_into(memoryview(buf)[got:])
      if cnt == 0: raise ConnectionError("TinyGPU socket closed")
      got += cnt
    return bytes(buf)

  def _rpc(self, cmd, bar, *args, readout=0, payload=b'', has_fd=False):
    self._sock.sendall(struct.pack("<BIIQQQ", int(cmd), self.dev_id, bar, *((tuple(args) + (0, 0, 0))[:3])) + payload)
    if payload:  # writes: server sends no response (matches examples/add.py)
      return None
    if has_fd:
      msg, anc, _, _ = self._sock.recvmsg(17, socket.CMSG_LEN(4))
      fd = struct.unpack('<i', anc[0][2][:4])[0]
    else:
      msg = self._recvall(17); fd = None
    status, value1, value2 = struct.unpack("<BQQ", msg)
    if status != 0:
      err = self._recvall(value1).decode('utf-8') if value1 > 0 else 'unknown error'
      raise RuntimeError(f"TinyGPU RPC cmd={int(cmd)} bar={bar} args={args} failed: {err}")
    data = self._recvall(readout) if readout else b""
    return value1, value2, data, fd

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

  def fini(self):
    try:
      if self._sock: self._sock.close()
    except Exception:
      pass
    self._sock = None

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


class SoftwarePCIDevice:
  """Offline stand-in for the eGPU transport (no socket, no TinyGPU.app).

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
    # host can copyin/copyout without a separate staging buffer.  On the eGPU we
    # map through the SYSTEM-MEMORY aperture (plan §24.1) since VRAM is not
    # initialized by a VBIOS on an eGPU.
    dev_impl = self.dev.dev_impl
    mm = dev_impl.mm
    aspace = kwargs.get("aspace", AddrSpace.SYS if dev_impl.hw is not None else AddrSpace.PHYS)
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
    self.pci_dev = SoftwarePCIDevice(dev_id) if software else APLRemotePCIDevice("NV", "usb4")
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

  In `software` backend mode (no eGPU present) `vram` is a flat host-side
  bytearray standing in for VRAM and `mm` is a fully functional GK104 GMMU
  manager over it, so the entire data path (alloc / map / copy / launch words)
  can be exercised offline.  The real hardware path maps BAR0/BAR1 via TinyGPU
  and is still gated behind `KEPLER-TODO` in the transport helpers."""
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
  # Software VRAM layout when no eGPU is present (flat host-side stand-in).
  VRAM_SIZE = 256 << 20   # 256 MB
  BOOT_SIZE = 4 << 20     # 4 MB reserved for page tables / boot allocations

  def __init__(self, device="", backend=None):
    self.device = device or "NV"
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    backend = backend or os.environ.get("NV_BACKEND", "software")
    self.backend = backend
    self.iface = PCIIface(self, self.device_id, software=(backend == "software"))
    self.dev_impl = NVDev(self.iface.pci_dev)
    if backend == "software":
      self._init_software()
    else:
      self._init_hardware()

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
    for p in range(pages):
      x = ptlb(p)
      flags = (x >> 24) & 0x7
      virtual_page = (x >> 8) & 0xffff
      if flags or x:
        print(f"[kepler] TLB: phys={p:02x} virt={virtual_page:04x} flags={flags:x} raw=0x{x:08x}")
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
    """Live eGPU bring-up (plan §24 / milestones 5-12).  Everything here is
    host-driven RM — there is no GSP on Kepler.  Steps that require firmware
    blobs or on-silicon validation are marked KEPLER-TODO and will raise until
    the real GTX 770 + TinyGPU.app + nouveau GK104 firmware are present."""
    dev = self.dev_impl
    dev.hw = self.iface.pci_dev
    dev.hw.bar_info(0)  # MAP_BAR: map the register BAR before any MMIO
    dev.bar1_addr, dev.bar1_size = dev.hw.bar_info(1)  # real BAR1 USERD aperture
    # 1. Identify + enable engines.
    boot0 = self.read32(0x0)  # PMC_BOOT_0 (dev_id/step)
    if DEBUG: print(f"[kepler] PMC_BOOT_0={boot0:#x}")
    # Enable the engines compute needs (nouveau gk104_mc reset/enable bits):
    # PGRAPH=0x1000 (bit12), PFIFO=0x100, PFB=0x08002000, LTC=0x02000000.
    # Avoid 0xffffffff (would arm engines whose firmware isn't loaded yet).
    # Enable engines.  Kepler GK104 needs the full supported engine set enabled
    # to exit power-gating — a minimal subset (PGRAPH|PFIFO|PFB|LTC) leaves
    # FECS/PGRAPH registers returning the 0xbad0da1f "engine disabled" sentinel.
    # Write 0xffffffff and let the GPU mask to its supported engines.
    self.write32(PMC_ENABLE, 0xffffffff)
    if DEBUG:
      print(f"[kepler] PMC_ENABLE enabled mask={self.read32(PMC_ENABLE):#x}")
    # 2. FALCON firmware (plan §24.1/§24.2): Kepler GK104 FALCONs are NOT
    #    secure-boot, so load IMEM/DMEM directly.  FECS/GPCCS ucode are the raw
    #    FUC arrays from hubgk104.fuc3.h / gpcgk104.fuc3.h (nouveau embeds them);
    #    PMU (gf119.fuc4.h) is optional for the first compute bring-up.
    fdir = find_kepler_firmware()
    if fdir is None:
      raise NotImplementedError(
        "NVDevice._init_hardware: no GK104 firmware tree (set NV_FIRMWARE_DIR "
        "to a dir containing gk104_fecs_code.bin).")
    if os.environ.get("KEPLER_VBIOS_DEVINIT", "0") == "1":
      vbios_path = os.environ.get("KEPLER_VBIOS", os.path.join(os.path.dirname(__file__), "Palit.GTX770.4096.131216.rom"))
      image, _, scripts = vbios_init_info(vbios_path)
      print(f"[kepler] VBIOS direct devinit script0={scripts[0]:#x}")
      for script in scripts:
        execute_vbios_target_ops(self, image, script)
      print(f"[kepler] after VBIOS devinit: PLL(0x137000)={self.read32(0x137000):#x} "
            f"GPC(0x409604)={self.read32(0x409604):#x}")
      program_gk104_gpc_pll(self)
    def _rd(name):
      p = os.path.join(fdir, name)
      if not os.path.exists(p):
        raise NotImplementedError(f"NVDevice._init_hardware: missing firmware {p}")
      return open(p, "rb").read()
    # GPC/ROP power-gate release needs the PMU alive; best-effort PMU bring-up.
    try:
      pmu_code = _rd("gk104_pmu_code.bin"); pmu_data = _rd("gk104_pmu_data.bin")
      falcon_load(self, PMU_FALCON_BASE, pmu_code, pmu_data, entry=0, start=True)
      if DEBUG: print("[kepler] PMU firmware loaded + started")
    except Exception as e:
      if DEBUG: print(f"[kepler] PMU load skipped: {e}")
    gk104_pmu_pgob(self)
    if DEBUG:
      print(f"[kepler] after pgob: gpc/rop(0x409604)={self.read32(0x409604):#x} "
            f"GPC0 CTRL(0x502100)={self.read32(0x502100):#x} "
            f"GPCCS CTRL(0x41a100)={self.read32(0x41a100):#x}")
    fecs_code = _rd("gk104_fecs_code.bin"); fecs_data = _rd("gk104_fecs_data.bin")
    gpccs_code = _rd("gk104_gpccs_code.bin"); gpccs_data = _rd("gk104_gpccs_data.bin")
    # nouveau gf100_gr_init: disable PGRAPH master, write main register init,
    # then re-enable master.  Extra FECS power/reset and clock-gating are needed
    # before the FALCON can start.
    self.write32(0x400500, 0x00000000)
    for addr, val in GK104_PGRAPH_PACK_MMIO:
      self.write32(addr, val)
    # GK104 FECS clock-gating and power/enable (from gf100_gr_fecs_reset).
    self.write32(0x409890, 0x00000045)
    self.write32(0x4098b0, 0x0000007f)
    self.write32(0x409614, 0x00000070)
    time.sleep(0.00001)
    before = self.read32(0x409614)
    self.write32(0x409614, (before & ~0x00000700) | 0x00000700)
    time.sleep(0.00001)
    _ = self.read32(0x409614)
    self.write32(0x400500, 0x00010001)
    print(f"[kepler] before ctxctl gate: FECS_CTRL={self.read32(0x409100):#x} GPCCS_CTRL={self.read32(0x41a100):#x} PGRAPH_CTRL={self.read32(0x400500):#x} RED_SWITCH={self.read32(0x409614):#x}")
    self.write32(0x260, 0)
    print(f"[kepler] after ctxctl gate (0x260=0): FECS_CTRL={self.read32(0x409100):#x} GPCCS_CTRL={self.read32(0x41a100):#x}")
    falcon_load(self, FECS_FALCON_BASE, fecs_code, fecs_data, entry=0, start=False)
    falcon_load(self, GPCCS_FALCON_BASE, gpccs_code, gpccs_data, entry=0, start=False)
    self.write32(0x260, 1)
    print(f"[kepler] after ctxctl ungate (0x260=1): FECS_CTRL={self.read32(0x409100):#x} GPCCS_CTRL={self.read32(0x41a100):#x}")
    # nouveau gf100_gr_init_ctxctl_int: upload the GR context-init (csdata)
    # register lists into the falcons via the 0x1c0/0x1c4 method interface
    # BEFORE starting FECS.  Without these the FECS firmware has no context
    # program to run and never posts "ready" (0x409800 bit31).
    from grctx_gk104 import CSDATA, method_stream
    for pack_name, info in CSDATA.items():
      words = method_stream(info["entries"], info["base"])
      falcon_csdata_write(self, info["falcon"], info["starstar"], words)
      if DEBUG:
        print(f"[kepler] csdata {pack_name}: {len(words)} method words -> "
              f"falcon={info['falcon']:#x} starstar={info['starstar']:#x}")
    print(f"[kepler] before FECS start: FECS_CTRL=0x{self.read32(0x409100):08x} FECS_SIGNAL=0x{self.read32(0x409400):08x} GPCCS_CTRL=0x{self.read32(0x41a100):08x} FECS_MMIO_BASE=0x{self.read32(0x409724):08x} FECS_MMIO_CTRL=0x{self.read32(0x409728):08x} ACCESS_EN=0x{self.read32(0x409048):08x} INTR=0x{self.read32(0x409008):08x} HWCFG2=0x{self.read32(0x40916c):08x} UC_STATUS=0x{self.read32(0x409128):08x} XFER_STATUS=0x{self.read32(0x409120):08x}")
    # tight transient trace around FECS start to see if it ever executes
    FECS_TRACE_REGS = {
      "INTR":          0x409008,
      "STATUS":        0x40904c,
      "UC_CTRL":       0x409100,
      "UC_ENTRY":      0x409104,
      "BLOCK_ON_FIFO": 0x40910c,
      "UC_STATUS":     0x409128,
      "BH_CTRL":       0x409148,
      "BH_PC":         0x40914c,
      "UC_PC":         0x409ff0,
      "HOST_IO_INDEX": 0x409ffc,
      "FECS_RESET":    0x409614,
    }
    def snapshot():
      return {n: self.read32(a) for n, a in FECS_TRACE_REGS.items()}
    before = snapshot()
    self.write32(0x40910c, 0x00000000)
    _ = self.read32(0x40910c)
    self.write32(0x409100, 0x00000002)
    _ = self.read32(0x409100)
    trace = []
    for _ in range(256):
      trace.append((
        self.read32(0x409100),
        self.read32(0x409128),
        self.read32(0x40904c),
        self.read32(0x409008),
        self.read32(0x409ff0),
        self.read32(0x40914c),
      ))
    after = snapshot()
    # print the trace compactly
    print(f"[kepler] FECS trace before start: {before}")
    print(f"[kepler] FECS trace after  start: {after}")
    # find the first sample where any register changed from the before snapshot
    base_vals = (before["UC_CTRL"], before["UC_STATUS"], before["STATUS"], before["INTR"], before["UC_PC"], before["BH_PC"])
    for i, s in enumerate(trace):
      if s != base_vals:
        print(f"[kepler] FECS trace first delta at sample {i}: {s}")
        break
    else:
      print(f"[kepler] FECS trace no change over 256 samples")
    # find any sample where UC_PC or STATUS changed
    for i, s in enumerate(trace):
      if s[4] != base_vals[4] or s[2] != base_vals[2]:
        print(f"[kepler] FECS trace first movement at sample {i}: UC_PC={s[4]:#x} STATUS={s[2]:#x} UC_CTRL={s[0]:#x} UC_STATUS={s[1]:#x} INTR={s[3]:#x} BH_PC={s[5]:#x}")
        break
    print(f"[kepler] after FECS start: FECS_CTRL=0x{self.read32(0x409100):08x} ACCESS_EN=0x{self.read32(0x409048):08x}")
    try:
      wait_cond(lambda: falcon_ready(self, FECS_FALCON_BASE),
                timeout_ms=2000, msg="FECS ready (0x409800 bit31)")
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
      # Verify the FECS firmware image actually loaded into imem/dmem.
      self.write32(FECS_FALCON_BASE + FALCON_CODE_INDEX, 0)   # read mode, addr 0
      imem0 = rd(FECS_FALCON_BASE + FALCON_CODE)
      self.write32(FECS_FALCON_BASE + FALCON_DATA_INDEX, 0)
      dmem0 = rd(FECS_FALCON_BASE + FALCON_DATA)
      print(f"[kepler] FECS imem[0]={imem0:#x} (expect 0x039b0ef5)  dmem[0]={dmem0:#x}")
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
    ctx_size = self.read32(0x409804)
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
    # 3. Sysmem aperture: allocate GPU-visible host memory via MAP_SYSMEM_FD and
    #    mmap it as a CPU-coherent buffer.  Its bus base becomes the GMMU
    #    bus_base (plan §24.1: eGPU has no VBIOS-init VRAM).
    sysmem_size = self.VRAM_SIZE
    memview, paddrs = self.iface.pci_dev.alloc_sysmem(sysmem_size, contiguous=True)
    dev.vram = memview.mv if hasattr(memview, "mv") else memview
    dev.bus_base = paddrs[0]
    dev.max_pa = sysmem_size
    dev.mm = GK104MemoryManager(dev, sysmem_size, self.BOOT_SIZE, bus_base=dev.bus_base)
    # 4-12. FIFO channel / GPFIFO / USERD / GR context / launch: KEPLER-TODO.
    dev.is_booting = False

  def runtime(self, name, lib):
    return NVProgram(self, name, lib)
  def synchronize(self): pass

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
  EF_CUDA_SM30 = 0x300030   # KEPLER-TODO: confirm exact Kepler EF_CUDA flags
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

def compile_kepler_cubin_docker(tag="nvidia/cuda:10.2-devel-ubuntu18.04"):
  """Compile a real sm_30 cubin with nvcc inside Docker. Returns the .cubin
  path, or None if Docker/nvcc is unavailable (plan §24.2: CUDA 10.2 still
  supports sm_30; CUDA 11+ dropped it)."""
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
  env_path = os.environ.get("KEPLER_CUBIN")
  if env_path and os.path.exists(env_path):
    with open(env_path, "rb") as f: return f.read()
  compiled = compile_kepler_cubin_docker()
  if compiled and os.path.exists(compiled):
    with open(compiled, "rb") as f: return f.read()
  # Offline placeholder — structure only, not executable SASS.
  return build_cubin()


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
  """nouveau nvkm_falcon_v1_load_imem: CODE_INDEX = start|WRITE, write a fresh
  ITAG every 64 words, then pad the code to a 0x40-word boundary with zeros."""
  code = code[:len(code) // 4 * 4]
  nwords = len(code) // 4
  dev.write32(base + FALCON_CODE_INDEX, FALCON_IDX_WRITE)
  tag = 0
  for i in range(nwords):
    if (i & 0x3f) == 0:
      dev.write32(base + FALCON_CODE_TAG, tag)
      tag += 1
    dev.write32(base + FALCON_CODE, struct.unpack_from("<I", code, i * 4)[0])
  i = nwords
  while i & 0x3f:
    dev.write32(base + FALCON_CODE, 0)
    i += 1

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
    nvbios_init.run_vbios_init(dev, image, [script], debug=DEBUG)
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

def gk104_pmu_pgob(dev, war00c800=True):
  """nouveau gk104_pmu_pgob(pmu, enable=false): release GPC/ROP/LTC power-gating
  so the FECS firmware can read the GPC/ROP topology (0x409604) and the per-GPC
  falcons (0x502000+, 0x41a000) become accessible.  Without this, 0x409604 = 0
  and the GR ctxctl init never posts "ready".

  The un-gate (0x020004 bit30) plus the 0x10a78c / 0x200 handshake require the
  PMU falcon to be alive (nouveau runs pgob after PMU subdev init).  The
  War00C800_0 0xc800 pokes also need the PMU; if the PMU is not running they
  simply time out (2s each) and proceed.  Runs TWICE in nouveau (oneinit +
  gr_init_); the un-gate is sticky, so once before ctxctl suffices for bring-up."""
  # fuse 0x31c bit0 gate skipped (set on real GK104).
  nvkm_mask(dev, 0x000200, 0x00001000, 0x00000000)   # clear GR reset (bit12)
  dev.read32(0x000200)                                # posted
  nvkm_mask(dev, 0x000200, 0x08000000, 0x08000000)   # set bit27
  time.sleep(0.05)
  nvkm_mask(dev, 0x10a78c, 0x00000002, 0x00000002)
  nvkm_mask(dev, 0x10a78c, 0x00000001, 0x00000001)
  nvkm_mask(dev, 0x10a78c, 0x00000001, 0x00000000)
  nvkm_mask(dev, 0x020004, 0xc0000000, 0x40000000)   # UN-GATE GR/GPC/ROP (bit30=1)
  time.sleep(0.05)
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
  dev.write32(base + FALCON_DATA_INDEX, 0x02000000 + starstar)
  star = dev.read32(base + FALCON_DATA)
  temp = dev.read32(base + FALCON_DATA)
  if temp > star:
    star = temp
  # 2. set DMEM write offset = star, then stream the method words.
  dev.write32(base + FALCON_DATA_INDEX, FALCON_IDX_WRITE | star)
  for w in words:
    dev.write32(base + FALCON_DATA, w)
    star += 4
  # 3. finalize the sub-stream (tell the falcon where it ends).
  dev.write32(base + FALCON_DATA_INDEX, 0x01000004 + starstar)
  dev.write32(base + FALCON_DATA, star + 4)

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

def build_launch_words(timeline_addr, wait_value, done_value, launch_desc_addr, code_va=0):
  # Kepler compute launch (envytools gk104_compute.xml, plan §24.3): bind the
  # compute class via SET_OBJECT, set the shader PC via CODE_ADDRESS_LO/HI
  # (0x160c/0x1608), point at the CWD via LAUNCH_DESC_ADDRESS (0x02b4, VA>>8),
  # then LAUNCH (0x02bc, value=3) to trigger.  Semaphore + cache-invalidate wrap.
  return [
    *nvm(1, 0x0000, KEPLER_COMPUTE_A),            # SET_OBJECT: bind compute class
    *gk104_semaphore(timeline_addr, wait_value, 0x00000004), # ACQUIRE_GEQUAL
    *nvm(1, 0x1698, 0x00001011),                  # INVALIDATE_SHADER_CACHES
    *nvm(1, 0x160c, code_va & 0xffffffff),        # CODE_ADDRESS_LO (shader PC base)
    *nvm(1, 0x1608, code_va >> 32),               # CODE_ADDRESS_HI
    *nvm(1, 0x02b4, launch_desc_addr >> 8),       # LAUNCH_DESC_ADDRESS (VA<<8 by HW)
    *nvm(1, 0x02bc, 0x3),                          # LAUNCH (trigger, value=3)
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

def build_cwd(code_addr, grid, block, shared=0, cbuf_addr=0, cbuf_size=256, regs=4):
  """Build a GK104 Compute Work Descriptor (0x100 bytes) per envytools
  gk104_compute.xml GK104_COMPUTE_LAUNCH_DESC decode (plan §24.3):
    0x20 PROG_START (offset from CODE_ADDRESS), 0x30 GRIDDIM_X, 0x34 GRIDDIM_YZ
    (Y bits0-15, Z bits16-31), 0x44 SHARED_ALLOC (align 0x100),
    0x48 BLOCKDIM_X (bits16-31!), 0x4c BLOCKDIM_YZ (Y lo, Z hi),
    0x50 CB_VALID (c0), 0x74/0x78 CB_CONFIG (cbuf addr/size),
    0xb8 GPR_ALLOC (register count, bits24-29)."""
  grid_x, grid_y, grid_z = grid
  block_x, block_y, block_z = block
  desc = bytearray(0x100)
  struct.pack_into("<I", desc, 0x20, code_addr & 0xffffffff)   # PROG_START (offset from CODE_ADDRESS)
  struct.pack_into("<I", desc, 0x30, grid_x)                   # GRIDDIM_X
  struct.pack_into("<I", desc, 0x34, grid_y | (grid_z << 16))  # GRIDDIM_YZ
  struct.pack_into("<I", desc, 0x44, shared)                   # SHARED_ALLOC
  struct.pack_into("<I", desc, 0x48, block_x << 16)            # BLOCKDIM_X (bits 16-31)
  struct.pack_into("<I", desc, 0x4c, block_y | (block_z << 16))  # BLOCKDIM_YZ
  struct.pack_into("<I", desc, 0x50, 0x1)                      # CB_VALID: c0 enabled
  struct.pack_into("<I", desc, 0x74, cbuf_addr & 0xffffffff)   # CB_CONFIG_0 (c0 addr lo)
  struct.pack_into("<I", desc, 0x78, ((cbuf_addr >> 32) & 0xff) | (cbuf_size << 15))  # CB_CONFIG_1: SIZE[31:15]
  struct.pack_into("<I", desc, 0xb8, (regs & 0x3f) << 24)      # GPR_ALLOC (bits 24-29)
  return bytes(desc)


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
MIDDLE_LAUNCH_WORDS = 24   # SET_OBJECT(2)+SEM(5)+INV(2)+CODE_LO(2)+CODE_HI(2)+LDESC(2)+LAUNCH(2)+SEM(5)+INT(2)

def kepler_selftest():
  """Tier 1 offline gate (no eGPU required): cubin structure + GMMU helpers
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
  assert len(sem_methods) == 2, "expected two GK104 semaphore sequences"

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

  # GK104/GF100 2-level walk sanity (software VRAM stand-in, no eGPU needed)
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

  print(f"kepler_selftest=ok cubin_sha={sha} launch_words={len(words)} sections={eh['shnum']}")
  return sha

def run_software_demo(dev):
  """End-to-end data path on the software VRAM stand-in (no real Kepler SASS
  executes — the add is performed host-side to validate alloc/map/copy/CWD)."""
  import random
  N = 256
  prog = dev.runtime("E_4", build_cubin())
  allocator = NVAllocator(dev)

  a_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
  b_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])

  a_dev = allocator.alloc(N * 4)
  b_dev = allocator.alloc(N * 4)
  out_dev = allocator.alloc(N * 4)
  # Code + constant (param) buffers for the launch descriptor.
  code = build_cubin()
  code_dev = allocator.alloc(len(code))
  allocator._copyin(code_dev, code)
  cbuf = bytearray(0x100)
  struct.pack_into("<Q", cbuf, 0x00, a_dev.va_addr)   # c0[0]: a
  struct.pack_into("<Q", cbuf, 0x08, b_dev.va_addr)   # c0[8]: b
  struct.pack_into("<Q", cbuf, 0x10, out_dev.va_addr)  # c0[16]: out
  cbuf_dev = allocator.alloc(len(cbuf))
  allocator._copyin(cbuf_dev, cbuf)

  allocator._copyin(a_dev, a_host.tobytes())
  allocator._copyin(b_dev, b_host.tobytes())

  # Build + map a CWD, then emit the Kepler compute launch words.
  cwd = build_cwd(code_addr=0, grid=(1, 1, 1), block=(N, 1, 1), cbuf_addr=cbuf_dev.va_addr)
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
# Live eGPU submit (plan milestones 5-12).  GK104 FIFO registers confirmed from
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
USERD_GP_GET = 0x88
USERD_GP_PUT = 0x8c

# GK104 GPFIFO entries are 8 bytes: a push-buffer address followed by a
# length/flags word. GP_PUT/GP_GET are ring-entry indices, not byte addresses.
GPFIFO_ENTRY_BYTES = 8

def _gk104_pgd_entry(dev, table_pa):
  """Build the VMM join value for a system-memory page directory."""
  mm = dev.dev_impl.mm
  return (mm.bus_base + table_pa) | 0x6  # HOST target + VOL

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

def submit_launch(dev, words, signal_pa, wait_value, done_value):
  """Set up a GK104 compute channel (RAMIN + USERD + GPFIFO), push `words` into
  the GPFIFO ring, kick the channel, and poll the host semaphore until it
  reaches `done_value`.  Writes go straight into the CPU-coherent sysmem mmap
  (`dev.vram`); `signal_pa` is the mmap offset of the semaphore page.  The GR
  context-buffer content (RAMIN 0x0210) and the exact GPFIFO ring base wiring
  are KEPLER-TODO pending a nouveau GK104 channel trace on silicon."""
  vram, base, mm = dev.dev_impl.vram, dev.dev_impl.bus_base, dev.dev_impl.mm
  alloc = NVAllocator(dev)
  ramin = alloc.alloc(0x1000)
  userd = alloc.alloc(0x200)
  gpfifo = alloc.alloc(0x2000, align=0x2000)   # 512 entries x 8 bytes, limit2=10
  runlist = alloc.alloc(0x1000, aspace=AddrSpace.NCOH)
  push = alloc.alloc(0x10000, align=0x10000)  # Nouveau main push buffer
  # GR engine context buffer (RAMIN 0x0210).  nouveau allocates
  # CB_RESERVED(0x80000) + gr->size and fills it via gf100_grctx_generate_main
  # (bundle/pagepool/attrib_cb ctxsw bundles).  We allocate + zero it; the
  # ctxsw content is KEPLER-TODO (needs the gk104 grctx init data on silicon).
  gr_ctx = alloc.alloc(0x100000)
  chan_id = int(os.environ.get("KEPLER_CHAN_ID", "1"))
  userd_base_off = chan_id * 0x200
  use_vram_inst = os.environ.get("KEPLER_VRAM_INST") == "1" and dev.dev_impl.hw is not None
  use_vram_runlist = os.environ.get("KEPLER_VRAM_RUNLIST") == "1" and dev.dev_impl.hw is not None
  if DEBUG and dev.dev_impl.hw is not None:
    bar1_ctl = dev.read32(0x1704)
    bar1_inst_pa = (bar1_ctl & 0x3fffffff) << 12
    try:
      bar1_head = dev.dev_impl.hw.mmio_read(1, bar1_inst_pa + 0x200, 16).hex()
    except Exception as e:
      bar1_head = f"read-error:{type(e).__name__}"
    print(f"[kepler] existing_bar1 ctl={bar1_ctl:#x} inst={bar1_inst_pa:#x} "
          f"pdb={bar1_head}")
  # TinyGPU's pre-existing BAR1 VMM instance is at 0x102000 on this board;
  # never allocate channel RAMIN/USERD over that live page-table object.
  bar1_cursor = 0x400000
  def bar1_alloc(size, align=0x1000):
    nonlocal bar1_cursor
    bar1_cursor = round_up(bar1_cursor, align)
    pa = bar1_cursor
    bar1_cursor += round_up(size, 0x1000)
    if bar1_cursor > dev.dev_impl.bar1_size:
      raise MemoryError("BAR1-backed instance allocation exceeds aperture")
    return pa
  def bar1_write(pa, data):
    dev.dev_impl.hw.mmio_write(1, pa, bytes(data))
  if use_vram_inst:
    # gk104_chan_ramfc_write() puts the USERD memory object's VRAM address in
    # RAMFC 0x08.  BAR1 is then the CPU mapping used to access that VRAM.  Keep
    # those two address domains distinct; the PCI BAR resource address is not
    # a GPU USERD address.
    userd_vram_pa = bar1_alloc(0x2000, align=0x2000)
    # Nouveau's nvkm_gpuobj_new() uses NVKM_MEM_TARGET_INST for the channel
    # instance and GR context. Keep their direct physical addresses in the
    # framebuffer aperture, while ordinary buffers remain in the VMM SYS path.
    ramin_vram_pa = bar1_alloc(0x1000)
    grctx_vram_pa = bar1_alloc(0x100000)
    gr_ctx = HCQBuffer(0x08000000, 0x100000, meta={"pa": grctx_vram_pa})
    dev.dev_impl.mm.map_range(gr_ctx.va_addr, gr_ctx.size,
                              [(grctx_vram_pa, gr_ctx.size)], AddrSpace.PHYS)
  else:
    ramin_vram_pa = None
    userd_vram_pa = None
  if use_vram_runlist:
    runlist_vram_pa = bar1_alloc(0x1000)
  else:
    runlist_vram_pa = None
  # Kepler's main GPFIFO entry is laid out as push base + 0x10000. Mirror the
  # Nouveau virtual layout even though the two backing allocations are local.
  push_va = 0x10000000
  gpfifo_va = push_va + 0x10000
  mm.map_range(push_va, push.size, [(push.meta['pa'], push.size)], AddrSpace.SYS)
  mm.map_range(gpfifo_va, gpfifo.size, [(gpfifo.meta['pa'], gpfifo.size)], AddrSpace.SYS)
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
  ctx_header = gr_ctx.meta['pa'] + 0x80000
  if use_vram_inst:
    header = bytearray(0x2c)
    struct.pack_into("<IIII", header, 0x1c, 1, 0, 0, 0)
    bar1_write(ctx_header, header)
  else:
    struct.pack_into("<IIII", vram, ctx_header + 0x1c, 1, 0, 0, 0)
  inst = bytearray(0x1000)
  pgd = _gk104_pgd_entry(dev, mm.root_pa)
  # gf100_vmm_join(): the channel VMM instance stores the page-directory
  # pointer at 0x200, not in the RAMFC's 0x00 area.
  struct.pack_into("<I", inst, 0x0200, pgd & 0xffffffff)
  struct.pack_into("<I", inst, 0x0204, pgd >> 32)
  userd_addr = ((userd_vram_pa + userd_base_off) if use_vram_inst else
                (dev.dev_impl.bar1_addr + userd_base_off))
  struct.pack_into("<I", inst, 0x08, userd_addr & 0xffffffff)   # USERD lo
  struct.pack_into("<I", inst, 0x0c, userd_addr >> 32)          # USERD hi
  # ctxgf100.c: RAMIN points at ctx->addr + CB_RESERVED (0x80000), not the
  # allocation base; bit 2 marks the engine context pointer valid.
  grctx_va = (gr_ctx.va_addr + 0x80000) | 4
  struct.pack_into("<I", inst, 0x0210, grctx_va & 0xffffffff)  # GR ctx ptr lo
  struct.pack_into("<I", inst, 0x0214, grctx_va >> 32)        # GR ctx ptr hi
  struct.pack_into("<I", inst, 0x48, 0)   # GP_PUT (relative)
  struct.pack_into("<I", inst, 0x4c, 0)   # GP_GET (relative)
  if use_vram_inst:
    bar1_write(ramin_vram_pa, inst)
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
  ring = bytearray(GPFIFO_ENTRY_BYTES)
  push_addr = push.va_addr
  # This is the channel's main push buffer, matching nvif_chan_gpfifo_push_kick
  # (main=true), so BIT(9) remains clear. The env override is a diagnostic for
  # the alternate external-entry encoding used by nvif_chan_gpfifo_push().
  gpfifo_external = os.environ.get("KEPLER_GPFIFO_EXTERNAL") == "1"
  struct.pack_into("<II", ring, 0, push_addr & 0xffffffff,
                   (push_addr >> 32) | ((1 << 9) if gpfifo_external else 0) |
                   (len(words) << 10))
  vram[gpfifo_pa:gpfifo_pa + len(ring)] = ring

  # RAMFC GPFIFO base and ring-size log2, matching gk104_chan_ramfc_write().
  gpfifo_va = gpfifo.va_addr
  struct.pack_into("<II", inst, 0x48, gpfifo_va & 0xffffffff,
                   (gpfifo_va >> 32) | (10 << 16))
  # Remaining fields from gk104_chan_ramfc_write().
  struct.pack_into("<I", inst, 0x10, 0x0000face)
  struct.pack_into("<I", inst, 0x30, 0xfffff902)
  struct.pack_into("<I", inst, 0x84, 0x20400000)
  struct.pack_into("<I", inst, 0x94, 0x30000fff)
  struct.pack_into("<I", inst, 0x9c, 0x00000100)
  struct.pack_into("<I", inst, 0xac, 0x0000001f)
  struct.pack_into("<I", inst, 0xe4, 0x00000000)
  struct.pack_into("<I", inst, 0xe8, chan_id)
  struct.pack_into("<I", inst, 0xb8, 0xf8000000)
  struct.pack_into("<II", inst, 0xf8, 0x10003080, 0x10000010)
  if use_vram_inst:
    bar1_write(ramin_vram_pa, inst)
  else:
    vram[ramin_pa:ramin_pa + len(inst)] = bytes(inst)
  if DEBUG and use_vram_inst:
    inst_read = dev.dev_impl.hw.mmio_read(1, ramin_vram_pa + 0x48, 8)
    print(f"[kepler] ramfc48_final={inst_read.hex()} ring_words="
          f"{struct.unpack_from('<II', vram, gpfifo_pa)} userd_ramfc="
          f"{dev.dev_impl.hw.mmio_read(1, ramin_vram_pa + 0x08, 8).hex()}")

  # gk104_mc_reset[] maps NVKM_ENGINE_FIFO to PMC_ENABLE bit 0x100.  A
  # previous TinyGPU process can leave the PBDMA context cache alive even
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
    bar1_write(runlist_pa, struct.pack("<II", chan_id, 0))
  else:
    struct.pack_into("<II", vram, runlist_pa, chan_id, 0)

  # gf100_gr_fecs_bind_pointer() followed by gf100_gr_fecs_wfi_golden_save().
  # These are required to make FECS materialize the golden GR image into the
  # context buffer before PBDMA dispatches a compute method stream.
  if dev.dev_impl.hw is not None:
    inst_tag = 0x80000000 | ((base + ramin_pa) >> 12)

  # gk104_fifo_init(): USERD is a BAR1-backed aperture on this generation.
  # TinyGPU exposes the coherent allocation at the same GPU-visible base, so
  # point PFIFO's BAR1 window at that base before submitting the runlist.
  if os.environ.get("KEPLER_USERD_BAR1_ZERO") == "1":
    userd_bar1_base = 0
  else:
    userd_bar1_base = (dev.dev_impl.bar1_addr
                       if os.environ.get("KEPLER_USERD_BAR1_RESOURCE") == "1"
                       else (userd_vram_pa if use_vram_inst else dev.dev_impl.bar1_addr))
  dev.write32(0x2254, 0x10000000 | (userd_bar1_base >> 12))
  # gk104_fifo_init_pbdmas(): enable the three GK104 PBDMAs and release the
  # scheduler's error-disable latch before committing a runlist.
  dev.write32(0x204, 0x7)
  nvkm_mask(dev, 0x2a04, 0xbfffffff, 0xbfffffff)
  # gf100_runq_init()/gk104_runq_init() for PBDMAs 0..2.  The 0x2390 values
  # are hardware-provided runlist masks (gk104_runq_runm() only reads them),
  # so do not overwrite them here.
  for pbdma in range(3):
    q = 0x40000 + pbdma * 0x2000
    dev.write32(q + 0x13c, dev.read32(q + 0x13c) & ~0x10000100)
    dev.write32(q + 0x108, 0xffffffff)
    dev.write32(q + 0x10c, 0xfffffeff)
    dev.write32(q + 0x148, 0xffffffff)
    dev.write32(q + 0x14c, 0xffffffff)
  dev.write32(0x2100, 0xffffffff)
  dev.write32(0x2140, 0x7fffffff)

  # Bind + start the channel (nouveau gk104_chan_bind_inst / gk104_chan_start).
  ramin_bind_addr = ramin_vram_pa if use_vram_inst else (base + ramin_pa)
  # Tear down a channel left halted by an earlier diagnostic run before
  # reusing CHID 0 and its BAR1 USERD state.
  nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x800, 0x800)
  # gk104_chan_bind() also selects the runlist in bits 16..19 of the channel
  # control word.  Do this explicitly for runlist 0; leaving the old value in
  # place can bind the channel to a different scheduler than 0x2274 commits.
  nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x000f0000, 0)
  dev.write32(CHAN_SUBMIT_REG + chan_id * 8, 0)
  # gf100_chan_userd_clear(): a reused BAR1 USERD must start with matching
  # GET/PUT and empty top-level pointers, otherwise PBDMA reports GPPTR.
  userd_mmio_base = ((userd_vram_pa + userd_base_off) if use_vram_inst else
                     userd_base_off)
  for userd_off in (0x40, 0x44, 0x48, 0x4c, 0x50, 0x58, 0x5c, 0x60,
                    USERD_GP_GET, USERD_GP_PUT):
    dev.dev_impl.hw.mmio_write(1, userd_mmio_base + userd_off, struct.pack("<I", 0))
  dev.write32(CHAN_SUBMIT_REG + chan_id * 8, 0x80000000 | (ramin_bind_addr >> 12))
  nvkm_mask(dev, CHAN_START_REG + chan_id * 8, 0x400, 0x400)
  # Make the channel visible to its runq before advancing USERD GP_PUT.
  runlist_addr = runlist_pa if use_vram_runlist else (base + runlist_pa)
  runlist_target = 0 if use_vram_runlist else 3
  if DEBUG and use_vram_runlist:
    print(f"[kepler] runlist_precommit="
          f"{struct.unpack('<II', dev.dev_impl.hw.mmio_read(1, runlist_pa, 8))}")
  # Match nvkm_runl_update_locked(): clear the pending runlist fault and
  # unblock scheduler processing before submitting the new list.
  dev.write32(0x262c, 1)
  nvkm_mask(dev, 0x2630, 1, 0)
  dev.write32(0x2270, (runlist_target << 28) | (runlist_addr >> 12))
  dev.write32(PFIFO_RUNLIST_SUBMIT, 1)
  if DEBUG and use_vram_runlist:
    print(f"[kepler] runlist_vram pa={runlist_pa:#x} words="
          f"{struct.unpack('<II', dev.dev_impl.hw.mmio_read(1, runlist_pa, 8))} "
          f"runq_masks={[hex(dev.read32(0x2390 + i * 4)) for i in range(3)]}")
  # gk104_runl_pending() is the per-runlist bit at 0x2284; nv50_runl_wait()
  # waits for it to clear before a channel kick can rely on the new list.
  deadline = time.time() + 0.2
  while dev.read32(0x2284) & 0x00100000:
    if time.time() >= deadline:
      raise TimeoutError("GK104 runlist update did not complete")
    time.sleep(0.001)
  # Advance USERD GP_PUT past the written entry.
  dev.dev_impl.hw.mmio_write(1, userd_mmio_base + USERD_GP_PUT, struct.pack("<I", 1))
  userd_put_readback = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, userd_mmio_base + USERD_GP_PUT, 4))[0]
  if DEBUG:
    print(f"[kepler] BAR1 USERD GP_PUT write=1 readback={userd_put_readback:#x} "
          f"userd_pa={userd_addr:#x} bar1_off={userd_mmio_base:#x} "
          f"bar1={dev.dev_impl.bar1_addr:#x} size={dev.dev_impl.bar1_size:#x}")
  # Commit runlist 0.  Target 3 is non-coherent system memory in Nouveau's
  # gf100_runl_commit(), and the count is separate from the channel ID.
  # Poll the host-visible semaphore page.
  for _ in range(2000):
    val = struct.unpack_from("<I", vram, signal_pa)[0]
    if val == done_value:
      return
    time.sleep(0.001)
  diag = {r: dev.read32(r) for r in (0x2254, 0x800000, 0x800004, 0x800008, 0x2270, 0x2274,
                                     0x2100, 0x252c, 0x256c, 0x259c,
                                     0x400000, 0x400004, 0x400014, 0x400048,
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
                                                            0x74c, 0x780, 0x784, 0x790)})
  if DEBUG:
    q0 = 0x40000
    print("[kepler] pbdma0 pointers " + " ".join(
      f"{name}=0x{dev.read32(q0 + off):08x}" for name, off in (
        ("IB_PUT", 0x00), ("CTRL_LO", 0x08), ("SIG", 0x10),
        ("IB_GET", 0x14), ("DMA_GET", 0x18), ("IB_ADDR", 0x48),
        ("IB_CFG", 0x4c), ("IB_ENTRY", 0x54), ("DMA_PUT", 0x5c),
        ("STATE0", 0x84), ("STATE1", 0x88), ("INTR", 0x108),
        ("CH", 0x120))))
  top = [dev.read32(0x22700 + i * 4) for i in range(64)]
  gp_get = struct.unpack("<I", dev.dev_impl.hw.mmio_read(1, userd_mmio_base + USERD_GP_GET, 4))[0]
  push_get = struct.unpack("<II", dev.dev_impl.hw.mmio_read(1, userd_mmio_base + 0x58, 8))
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

def run_hardware_demo(dev):
  """End-to-end add on the real GTX 770 over TinyGPU (sysmem compute path,
  plan §24.1).  Requires TinyGPU.app, a GK104 firmware tree ($NV_FIRMWARE_DIR),
  and on-silicon FIFO validation."""
  import random
  N = 256
  prog = dev.runtime("E_4", get_kepler_cubin())
  allocator = NVAllocator(dev)

  a_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])
  b_host = array.array('f', [random.uniform(-1, 1) for _ in range(N)])

  a_dev = allocator.alloc(N * 4)
  b_dev = allocator.alloc(N * 4)
  out_dev = allocator.alloc(N * 4)
  signal = allocator.alloc(16)
  # Code + constant (param) buffers for the launch descriptor.
  cubin = get_kepler_cubin()
  sass = elf_section_bytes(cubin, ".text.E_4")
  code_dev = allocator.alloc(round_up(len(sass), 0x100))
  allocator._copyin(code_dev, sass)
  cbuf = bytearray(0x100)
  struct.pack_into("<Q", cbuf, 0x00, a_dev.va_addr)   # c0[0]: a
  struct.pack_into("<Q", cbuf, 0x08, b_dev.va_addr)   # c0[8]: b
  struct.pack_into("<Q", cbuf, 0x10, out_dev.va_addr)  # c0[16]: out
  cbuf_dev = allocator.alloc(len(cbuf))
  allocator._copyin(cbuf_dev, cbuf)
  # build_launch_words() begins with a semaphore wait for wait_value=1.
  # Seed the coherent semaphore before submitting the push buffer.
  allocator._copyin(signal, struct.pack("<I", 1))
  cwd = build_cwd(code_addr=0, grid=(1, 1, 1), block=(N, 1, 1), cbuf_addr=cbuf_dev.va_addr)
  cwd_dev = allocator.alloc(len(cwd))
  allocator._copyin(cwd_dev, cwd)
  allocator._copyin(a_dev, a_host.tobytes())
  allocator._copyin(b_dev, b_host.tobytes())

  words = build_launch_words(signal.va_addr, 1, 2, cwd_dev.va_addr, code_dev.va_addr)
  if DEBUG:
    pgd_idx = (signal.va_addr >> 27) & 0x1fff
    spt_idx = (signal.va_addr >> 12) & 0x7fff
    pgd_entry = dev.dev_impl.mm.root_page_table.entry(pgd_idx)
    spt = GK104PageTableEntry(dev.dev_impl, dev.dev_impl.mm.root_page_table.address(pgd_idx), 0)
    print(f"[kepler] vmm signal_va={signal.va_addr:#x} signal_pa={signal.meta['pa']:#x} "
          f"bus={dev.dev_impl.mm.bus_base:#x} pgd={pgd_entry:#x} pte={spt.entry(spt_idx):#x}")
  if os.environ.get("KEPLER_TEST_SEM_ONLY") == "1":
    allocator._copyin(signal, struct.pack("<I", 0))
    words = gk104_semaphore(signal.va_addr, 2, 0x00000002)
  submit_launch(dev, words, signal.meta['pa'], 1, 2)

  out_host = bytearray(N * 4)
  allocator._copyout(out_host, out_dev)
  out_arr = array.array('f'); out_arr.frombytes(bytes(out_host))
  expected = [a_host[i] + b_host[i] for i in range(N)]
  assert all(abs(out_arr[i] - expected[i]) < 1e-5 for i in range(N)), "hardware add mismatch"
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
  if "--probe" in sys.argv:
    d = APLRemotePCIDevice.probe()
    if d is None:
      print("probe: TinyGPU.app not reachable (is the eGPU connected / app running?)")
      sys.exit(1)
    d.bar_info(0)  # MAP_BAR: map the register BAR before any MMIO
    boot0 = d.mmio_read32(0, 0x0)  # PMC_BOOT_0: chip id + stepping
    print(f"probe: PMC_BOOT_0=0x{boot0:08x} (chip_id={(boot0>>20)&0xfff}, "
          f"revision=0x{(boot0>>4)&0xff}, fab=0x{(boot0>>4)&0xf})")
    d.fini()
    return
  if "--probe-pgob" in sys.argv:
    d = APLRemotePCIDevice.probe()
    if d is None:
      print("probe-pgob: TinyGPU.app not reachable (is the eGPU connected?)")
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
    d = APLRemotePCIDevice.probe()
    if d is None:
      print("probe-gpc-clock: TinyGPU.app not reachable (is the eGPU connected?)")
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
    d = APLRemotePCIDevice.probe()
    if d is None:
      print("probe-vbios-devinit: TinyGPU.app not reachable (is the eGPU connected?)")
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
    d = APLRemotePCIDevice.probe()
    if d is None:
      print("probe-falcon: TinyGPU.app not reachable (is the eGPU connected?)")
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
  backend = os.environ.get("NV_BACKEND", "software")
  try:
    dev = NVDevice("NV", backend=backend)
  except (NotImplementedError, OSError) as e:
    print("LIVE PATH NOT YET IMPLEMENTED (Kepler bring-up pending):")
    print(f"  {e}")
    print("Run `python3 examples_kepler/add.py --middle-selftest` for the offline gate,")
    print("or `NV_BACKEND=software python3 examples_kepler/add.py` for the software demo,")
    print("or `python3 examples_kepler/add.py --probe` to identify the eGPU.")
    sys.exit(2)
  if backend == "software":
    run_software_demo(dev)
  else:
    run_hardware_demo(dev)

if __name__ == "__main__":
  main()
