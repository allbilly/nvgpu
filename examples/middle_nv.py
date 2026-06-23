#!/usr/bin/env python3
"""Vendored NV stack for add_middle.py — single RTX 3080 eGPU over PCIe/USB4.

Vendored from ref/tinygrad/tinygrad/runtime/support/{nv,system,hcq,memory,elf}.py
and ref/tinygrad/tinygrad/runtime/ops_nv.py (slimmed to the PCIe path only).

The only external dependencies this module has are:
  - tinygrad.runtime.autogen.{nv, nv_570, nv_regs, pci, libc}  (ctypes constants only)
  - Python standard library

NO imports from tinygrad.runtime.support, tinygrad.runtime.ops, tinygrad.device,
tinygrad.renderer, tinygrad.uop, tinygrad.helpers are permitted in this module —
those have been vendored inline below.
"""
from __future__ import annotations
import os, sys, ctypes, ctypes.util, time, mmap, struct, array as _array_mod, socket, subprocess, contextlib, functools, itertools, enum, atexit, select, dataclasses, collections, urllib.request, hashlib, tempfile, gzip, pathlib
from typing import cast, Any, ClassVar, Generic, TypeVar
from dataclasses import dataclass, replace

# --- autogen ctypes (allowed by goal: "ctypes constants only") ---
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu, pci
from tinygrad.runtime.autogen import nv_regs
from tinygrad.runtime.autogen import libc

# ============================================================================
# Helpers (slimmed from tinygrad/helpers.py — only what we actually use)
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

def fetch_fw(path: str, name: str, sha256: str) -> bytes:
  cache_dir = _ensure_downloads_dir() / "fw"
  cache_dir.mkdir(parents=True, exist_ok=True)
  fp = cache_dir / name
  if fp.is_file() and hashlib.sha256(fp.read_bytes()).hexdigest() == sha256:
    return fp.read_bytes()
  url = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/1e2c15348485939baf1b6d1f5a7a3b799d80703d/{path}/{name}"
  with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "middle_nv"}), timeout=10) as r:
    data = r.read()
  if hashlib.sha256(data).hexdigest() != sha256:
    raise RuntimeError(f"fetch_fw sha mismatch for {name}")
  fp.write_bytes(data)
  return data

def pluralize(n, s, p=None):
  if p is None: p = s + "s"
  return f"{n} {p}" if n != 1 else f"1 {s}"


# ============================================================================
# memory.py (vendored from ref/tinygrad/tinygrad/runtime/support/memory.py)
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
  PHYS = enum.auto(); SYS = enum.auto(); PEER = enum.auto()

@dataclasses.dataclass(frozen=True)
class VirtMapping:
  va_addr: int; size: int; paddrs: list; aspace: AddrSpace; uncached: bool = False; snooped: bool = False

class PageTableTraverseContext:
  def __init__(self, dev, pt, vaddr, create_pts=False, free_pts=False, inspect=False, boot=False):
    self.dev, self.vaddr, self.create_pts, self.free_pts, self.inspect, self.boot = dev, vaddr - dev.mm.va_base, create_pts, free_pts, inspect, boot
    self.pt_stack = [(pt, self._pt_pte_idx(pt, self.vaddr), self._pt_pte_size(pt))]
  def _pt_pte_cnt(self, lv): return self.dev.mm.pte_cnt[lv]
  def _pt_pte_size(self, pt): return self.dev.mm.pte_covers[pt.lv]
  def _pt_pte_idx(self, pt, va): return (va // self._pt_pte_size(pt)) % self._pt_pte_cnt(pt.lv)
  def level_down(self):
    pt, pte_idx, _ = self.pt_stack[-1]
    if not pt.valid(pte_idx):
      assert self.create_pts, "Not allowed to create new page table"
      pt.set_entry(pte_idx, self.dev.mm.palloc(0x1000, zero=True, boot=self.boot, ptable=True), table=True, valid=True)
    assert not pt.is_page(pte_idx), f"Must be table pt={pt.paddr:#x}, {pt.lv=} {pte_idx=} {pt.entry(pte_idx)=:#x}"
    child_page_table = self.dev.mm.pt_t(self.dev, pt.address(pte_idx), lv=pt.lv + 1)
    self.pt_stack.append((child_page_table, self._pt_pte_idx(child_page_table, self.vaddr), self._pt_pte_size(child_page_table)))
    return self.pt_stack[-1]
  def _try_free_pt(self) -> bool:
    pt, _, _ = self.pt_stack[-1]
    if self.free_pts and pt != self.dev.mm.root_page_table and all(not pt.valid(i) for i in range(self._pt_pte_cnt(self.pt_stack[-1][0].lv))):
      self.dev.mm.pfree(pt.paddr, ptable=True)
      parent_pt, parent_pte_idx, _ = self.pt_stack[-2]
      parent_pt.set_entry(parent_pte_idx, 0x0, valid=False)
      return True
    return False
  def level_up(self):
    while self._try_free_pt() or self.pt_stack[-1][1] == self._pt_pte_cnt(self.pt_stack[-1][0].lv):
      pt, pt_cnt, _ = self.pt_stack.pop()
      if pt_cnt == self._pt_pte_cnt(pt.lv): self.pt_stack[-1] = (self.pt_stack[-1][0], self.pt_stack[-1][1] + 1, self.pt_stack[-1][2])
  def next(self, size, paddr=None, off=0):
    while size > 0:
      pt, pte_idx, pte_covers = self.pt_stack[-1]
      if self.create_pts:
        assert paddr is not None, "paddr must be provided when allocating new page tables"
        while pte_covers > size or not pt.supports_huge_page(paddr + off) or self.vaddr & (pte_covers - 1) != 0:
          pt, pte_idx, pte_covers = self.level_down()
      else:
        while not pt.is_page(pte_idx) and (self.free_pts or pt.valid(pte_idx)):
          pt, pte_idx, pte_covers = self.level_down()
      entries = max(min(size // pte_covers, self._pt_pte_cnt(pt.lv) - pte_idx), 1 if self.inspect else 0)
      assert entries > 0, f"Invalid entries {size=:#x}, {pte_covers=:#x}"
      yield off, pt, pte_idx, entries, pte_covers
      size, off, self.vaddr = size - entries * pte_covers, off + entries * pte_covers, self.vaddr + entries * pte_covers
      self.pt_stack[-1] = (pt, pte_idx + entries, pte_covers)
      self.level_up()

class MemoryManager:
  va_allocator = None
  def __init__(self, dev, vram_size, boot_size, pt_t, va_bits, va_shifts, va_base, palloc_ranges, first_lv=0, reserve_ptable=False):
    self.dev, self.vram_size, self.va_shifts, self.va_base, lvl_msb = dev, vram_size, va_shifts, va_base, va_shifts + [va_bits + 1]
    self.pte_covers, self.pte_cnt = [1 << x for x in va_shifts][::-1], [1 << (lvl_msb[i+1] - lvl_msb[i]) for i in range(len(lvl_msb) - 1)][::-1]
    self.pt_t, self.palloc_ranges, self.level_cnt, self.va_bits, self.reserve_ptable = pt_t, palloc_ranges, len(va_shifts), va_bits, reserve_ptable
    self.boot_allocator = TLSFAllocator(boot_size, base=0)
    self.ptable_allocator = TLSFAllocator(round_up(vram_size // 512, 1 << 20) if self.reserve_ptable else 0, base=self.boot_allocator.size)
    self.pa_allocator = TLSFAllocator(vram_size - (off_sz := self.boot_allocator.size + self.ptable_allocator.size), base=off_sz)
    self.root_page_table = pt_t(self.dev, self.palloc(0x1000, zero=not self.dev.smi_dev, boot=True), lv=first_lv)
  def _frag_size(self, va, sz, must_cover=True):
    va_pwr2_div, sz_pwr2_div, sz_pwr2_max = va & -(va) if va > 0 else (1 << 63), sz & -(sz), (1 << (sz.bit_length() - 1))
    return (min(va_pwr2_div, sz_pwr2_div) if must_cover else min(va_pwr2_div, sz_pwr2_max)).bit_length() - 1 - 12
  def page_tables(self, vaddr, size):
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True)
    for _ in ctx.next(size, paddr=0): return [pt for pt, _, _ in ctx.pt_stack]
  def map_range(self, vaddr, size, paddrs, aspace, uncached=False, snooped=False, boot=False):
    assert size == sum(p[1] for p in paddrs), f"Size mismatch {size=} {sum(p[1] for p in paddrs)=}"
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, boot=boot, inspect=True)
    for _, pt, pte_idx, pte_cnt, _ in ctx.next(size):
      for pte_off in range(pte_cnt): assert not pt.valid(pte_idx + pte_off), f"PTE already mapped: {pt.entry(pte_idx + pte_off):#x}"
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True, boot=boot)
    for paddr, psize in paddrs:
      for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(psize, paddr=paddr):
        for pte_off in range(pte_cnt):
          pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers, uncached=uncached, aspace=aspace, snooped=snooped,
                       frag=self._frag_size(ctx.vaddr + off, pte_cnt * pte_covers), valid=True)
    self.on_range_mapped()
    return VirtMapping(vaddr, size, paddrs, aspace=aspace, uncached=uncached, snooped=snooped)
  def unmap_range(self, vaddr, size):
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, free_pts=True)
    for _, pt, pte_idx, pte_cnt, _ in ctx.next(size):
      for pte_id in range(pte_idx, pte_idx + pte_cnt):
        assert pt.valid(pte_id), f"PTE not mapped: {pt.entry(pte_id):#x}"
        pt.set_entry(pte_id, paddr=0x0, valid=False)
  def on_range_mapped(self): pass
  @classmethod
  def alloc_vaddr(cls, size, align=0x1000):
    assert cls.va_allocator is not None, "must be set"
    return cls.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))
  @functools.cache
  def identity_va(self, uncached):
    self.map_range(va := self.alloc_vaddr(self.vram_size, self.vram_size), self.vram_size, [(0, self.vram_size)], AddrSpace.PHYS, uncached=uncached)
    return va
  def valloc(self, size, align=0x1000, uncached=False, contiguous=False):
    if not getenv("GMMU", 1):
      paddr = self.palloc(size := round_up(size, 0x1000), align, zero=False)
      return VirtMapping(self.identity_va(uncached) + paddr, size, [(paddr, size)], aspace=AddrSpace.PHYS, uncached=uncached)
    va = self.alloc_vaddr(size := round_up(size, 0x1000), align)
    if contiguous: paddrs = [(self.palloc(size, zero=True), size)]
    else:
      nxt_range, rem_size, paddrs = 0, size, []
      while rem_size > 0:
        while self.palloc_ranges[nxt_range][0] > rem_size: nxt_range += 1
        try: paddrs += [(self.palloc(try_sz := self.palloc_ranges[nxt_range][0], self.palloc_ranges[nxt_range][1], zero=False), try_sz)]
        except MemoryError:
          nxt_range += 1
          if nxt_range == len(self.palloc_ranges):
            for paddr, _ in paddrs: self.pfree(paddr)
            raise MemoryError(f"Failed to allocate memory (OOM). Request size={size:#x}")
          continue
        rem_size -= self.palloc_ranges[nxt_range][0]
    return self.map_range(va, size, paddrs, aspace=AddrSpace.PHYS, uncached=uncached)
  def vfree(self, vm):
    if not getenv("GMMU", 1): return self.pfree(vm.paddrs[0][0])
    assert self.va_allocator is not None, "must be set"
    self.unmap_range(vm.va_addr, vm.size)
    self.va_allocator.free(vm.va_addr)
    for paddr, _ in vm.paddrs: self.pfree(paddr)
  def palloc(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    assert self.dev.is_booting == boot, "During booting, only boot memory can be allocated"
    allocator = self.boot_allocator if boot else (self.ptable_allocator if self.reserve_ptable and ptable else self.pa_allocator)
    paddr = allocator.alloc(round_up(size, 0x1000), align)
    if zero: self.dev.vram[paddr:paddr + size] = bytes(size)
    return paddr
  def pfree(self, paddr, ptable=False):
    (self.ptable_allocator if self.reserve_ptable and ptable else self.pa_allocator).free(paddr)


# ============================================================================
# hcq.py (vendored from ref/tinygrad/tinygrad/runtime/support/hcq.py)
# Slimmed: only MMIOInterface, FileIOInterface, HCQBuffer, hcq_filter_visible_devices.
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
  def listdir(self): return os.listdir(self.path)
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
    self.va_addr, self.size, self.meta, self._base, self.view = va_addr, size, meta, _base, view
    self._devs = [owner] if owner is not None else []
    self.owner = owner
    self._mappings = {}
  def offset(self, offset=0, size=None):
    return HCQBuffer(self.va_addr + offset, size or (self.size - offset), owner=self.owner, meta=self.meta,
                     _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))
  def cpu_view(self):
    assert self.view is not None, "buffer has no cpu_view"
    return self.view
  @property
  def base(self): return self._base or self

# ============================================================================
# elf.py (vendored from ref/tinygrad/tinygrad/runtime/support/elf.py)
# ============================================================================
@dataclass(frozen=True)
class ElfSection:
  name: str
  header: Any
  content: bytes

def _elf_strtab(blob, idx): return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

def link_sym(sym: str, libs: list[str]) -> int:
  for lib in libs:
    try:
      return unwrap(ctypes.cast(getattr(ctypes.CDLL(ctypes.util.find_library(lib)), sym), ctypes.c_void_p).value)
    except (OSError, AttributeError):
      pass
  raise RuntimeError(f'Attempting to relocate against an undefined symbol {sym}')

def elf_loader(blob, force_section_align=1, link_libs=None):
  assert blob[:4] == libc.ELFMAG.encode(), "blob is not an ELF, missing magic bytes"
  ecls = {libc.ELFCLASS32: "Elf32", libc.ELFCLASS64: "Elf64"}[blob[libc.EI_CLASS]]
  header = getattr(libc, f"{ecls}_Ehdr").from_buffer_copy(blob)
  section_headers = (getattr(libc, f"{ecls}_Shdr") * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  sh_strtab = blob[(shstrst := section_headers[header.e_shstrndx].sh_offset):shstrst + section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_elf_strtab(sh_strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset + sh.sh_size]) for sh in section_headers]

  def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)
  rel = [(sh, sh.name[4:], _to_carray(sh, getattr(libc, f"{ecls}_Rel"))) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, sh.name[5:], _to_carray(sh, getattr(libc, f"{ecls}_Rela"))) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = next((_to_carray(sh, getattr(libc, f"{ecls}_Sym")) for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB), None)
  progbits = [sh for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS]

  image = bytearray(max([sh.header.sh_addr + sh.header.sh_size for sh in progbits if sh.header.sh_addr != 0] + [0]))
  for sh in progbits:
    if sh.header.sh_addr != 0:
      image[sh.header.sh_addr:sh.header.sh_addr + sh.header.sh_size] = sh.content
    else:
      image += b'\0' * (((align := max(sh.header.sh_addralign, force_section_align)) - len(image) % align) % align) + sh.content
      sh.header.sh_addr = len(image) - len(sh.content)

  relocs = []
  for sh, trgt_sh_name, c_rels in rel + rela:
    if trgt_sh_name == ".eh_frame":
      continue
    target_image_off = next(tsh for tsh in sections if tsh.name == trgt_sh_name).header.sh_addr
    rels = [(r.r_offset, unwrap(symtab)[getattr(libc, f"{ecls.upper()}_R_SYM")(r.r_info)],
             getattr(libc, f"{ecls.upper()}_R_TYPE")(r.r_info), getattr(r, "r_addend", 0)) for r in c_rels]
    relocs += [(target_image_off + roff,
                 link_sym(_elf_strtab(sh_strtab, sym.st_name), link_libs or []) if sym.st_shndx == 0 else
                 sections[sym.st_shndx].header.sh_addr + sym.st_value,
                 rtype, raddend) for roff, sym, rtype, raddend in rels]
  return memoryview(image), sections, relocs


# ============================================================================
# system.py (vendored from ref/tinygrad/tinygrad/runtime/support/system.py)
# Slimmed: only the APLRemotePCIDevice path (Mac eGPU via TinyGPU.app unix socket).
# ============================================================================
MAP_FIXED, MAP_FIXED_NOREPLACE = 0x10, 0x100000
MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400
PAGESIZE = mmap.PAGESIZE

class _System:
  @functools.cached_property
  def libsys(self): return ctypes.CDLL(ctypes.util.find_library("System"))
  @functools.cached_property
  def atomic_lib(self): return ctypes.CDLL(ctypes.util.find_library('atomic')) if not OSX else None
  def memory_barrier(self):
    lib = self.libsys if OSX else self.atomic_lib
    if lib is not None: lib.atomic_thread_fence(5)

  def reserve_va(self, va_start, va_size):
    FileIOInterface.anon_mmap(va_start, va_size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, 0)

  def flock_acquire(self, name):
    import fcntl as _f
    lock_name = temp(name)
    if os.path.exists(lock_name):
      lock_fd = os.open(lock_name, os.O_RDWR)
    else:
      os.umask(0)
      lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try: _f.flock(lock_fd, _f.LOCK_EX | _f.LOCK_NB)
    except OSError: raise RuntimeError(
      f"Failed to acquire lock file {name}. Only one eGPU client (add_tiny/add_middle/TinyGPU) may run at a time. "
      f"`lsof {lock_name}` shows which process holds it.")
    return lock_fd

System = _System()

class RemoteCmd(enum.IntEnum):
  PROBE, MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE, MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE, RESIZE_BAR, PING = range(13)

class RemoteMMIOInterface(MMIOInterface):
  def __init__(self, dev, residx, nbytes, fmt='B', off=0, rd_cmd=RemoteCmd.MMIO_READ, wr_cmd=RemoteCmd.MMIO_WRITE):
    self.dev, self.residx, self.nbytes, self.fmt, self.off, self.el_sz = dev, residx, nbytes, fmt, off, struct.calcsize(fmt)
    self.rd_cmd, self.wr_cmd = rd_cmd, wr_cmd
  def __getitem__(self, index):
    sl = index if isinstance(index, slice) else slice(index, index + 1)
    start, stop = (sl.start or 0) * self.el_sz, (sl.stop or len(self)) * self.el_sz
    data = self.dev._bulk_read(self.rd_cmd, self.residx, self.off + start, stop - start)
    result = data if self.fmt == 'B' else list(struct.unpack(f'<{(stop - start) // self.el_sz}{self.fmt}', data))
    return result if isinstance(index, slice) else result[0]
  def __setitem__(self, index, val):
    start = (index.start or 0) * self.el_sz if isinstance(index, slice) else index * self.el_sz
    if self.fmt == 'B':
      data = bytes(val) if isinstance(val, (bytes, bytearray, memoryview)) else (bytes(val) if isinstance(val, (list, tuple)) else bytes([val]))
      if not isinstance(index, slice): data = data[:self.el_sz]
    elif isinstance(index, slice):
      data = struct.pack(f'<{len(val)}{self.fmt}', *val)
    else:
      data = struct.pack(f'<{self.fmt}', val)
    self.dev._bulk_write(self.wr_cmd, self.residx, self.off + start, data)
  def view(self, offset=0, size=None, fmt=None):
    return RemoteMMIOInterface(self.dev, self.residx, size or (self.nbytes - offset), fmt or self.fmt,
      self.off + offset, self.rd_cmd, self.wr_cmd)

class RemotePCIDevice:
  def __init__(self, devpref, pcibus, sock):
    self.sock, self.pcibus, self.dev_id = sock, pcibus, int(pcibus.split(':')[-1]) if ':' in pcibus else 0
    for buft in [socket.SO_SNDBUF, socket.SO_RCVBUF]: self.sock.setsockopt(socket.SOL_SOCKET, buft, 64 << 20)
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")

  @staticmethod
  def _recvall(sock, n):
    data = b''
    while len(data) < n and (chunk := sock.recv(n - len(data))): data += chunk
    if len(data) < n: raise RuntimeError("Connection closed")
    return data
  @staticmethod
  def _rpc(sock, dev_id, cmd, *args, bar=0, readout_size=0, payload=b'', has_fd=False):
    sock.sendall(struct.pack('<BIIQQQ', cmd, dev_id, bar, *(*args, 0, 0, 0)[:3]) + payload)
    if has_fd:
      msg, anc, _, _ = sock.recvmsg(17, socket.CMSG_LEN(4))
      fd = struct.unpack('<i', anc[0][2][:4])[0]
    else: msg, fd = RemotePCIDevice._recvall(sock, 17), None
    if (resp := struct.unpack('<BQQ', msg))[0] != 0:
      raise RuntimeError(f"RPC failed: {RemotePCIDevice._recvall(sock, resp[1]).decode('utf-8') if resp[1] > 0 else 'unknown error'}")
    return (resp[1], resp[2]) + ((RemotePCIDevice._recvall(sock, readout_size) if readout_size > 0 else None),) + (fd,)

  def _bulk_read(self, cmd, idx, offset, size):
    return unwrap(self._rpc(self.sock, self.dev_id, cmd, offset, size, bar=idx, readout_size=size)[2])
  def _bulk_write(self, cmd, idx, offset, data):
    self.sock.sendall(struct.pack('<BIIQQQ', cmd, self.dev_id, idx, offset, len(data), 0) + data)

  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    mapped_size, _, _, fd = self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
    memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt='B')
    paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
    return memview, [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]

  def reset(self): self._rpc(self.sock, self.dev_id, RemoteCmd.RESET)
  def read_config(self, offset, size): return self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_READ, offset, size)[0]
  def write_config(self, offset, value, size): self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_WRITE, offset, size, value)
  def write_config_flush(self, offset, value, size): self.write_config(offset, value, size); self.read_config(offset, size)
  @functools.cache
  def bar_info(self, bar_idx): return self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_BAR, bar=bar_idx)[:2]
  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'):
    return RemoteMMIOInterface(self, bar, size or self.bar_info(bar)[1], fmt).view(off, size, fmt)
  def resize_bar(self, bar_idx): self._rpc(self.sock, self.dev_id, RemoteCmd.RESIZE_BAR, bar=bar_idx)

class APLRemotePCIDevice(RemotePCIDevice):
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"
  APP_COMMIT = "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"  # pinned commit

  def __init__(self, devpref, pcibus):
    sock_path = os.environ.get("APL_REMOTE_SOCK", temp("tinygpu.sock"))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connected = False
    for i in range(100):
      try:
        sock.connect(sock_path); connected = True; break
      except (ConnectionRefusedError, FileNotFoundError):
        if i == 0:
          subprocess.Popen([self.APP_PATH, "server", sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
    if not connected: raise RuntimeError(f"Failed to connect to TinyGPU server at {sock_path}")
    super().__init__(devpref, "usb4", sock=sock)


# ============================================================================
# nv/nvdev.py (vendored from ref/tinygrad/tinygrad/runtime/support/nv/nvdev.py)
# Slimmed: NVReg, NVPageTableEntry, NVMemoryManager, NVDev (boot shell only).
# ============================================================================
NV_DEBUG = getenv("NV_DEBUG", 0)

class NVReg:
  def __init__(self, nvdev, base, off, fields=None):
    self.nvdev, self.base, self.off, self.fields = nvdev, base, off, fields or {}
  def __getitem__(self, idx): return NVReg(self.nvdev, self.base, self.off(idx), fields=self.fields)
  def with_base(self, base): return NVReg(self.nvdev, base + self.base, self.off, self.fields)
  def add_field(self, name, start, end): self.fields[name] = (start, end)
  def read(self): return self.nvdev.rreg(self.base + self.off)
  def read_bitfields(self): return self.decode(self.read())
  def write(self, _ini_val=0, **kwargs): self.nvdev.wreg(self.base + self.off, _ini_val | self.encode(**kwargs))
  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)
  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1] - self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)
  def encode(self, **kwargs):
    return functools.reduce(int.__or__, (value << self.fields[name][0] for name, value in kwargs.items()), 0)
  def decode(self, val):
    return {name: getbits(val, start, end) for name, (start, end) in self.fields.items()}

class NVPageTableEntry:
  def __init__(self, nvdev, paddr, lv):
    self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')

  def _is_dual_pde(self): return self.lv == self.nvdev.mm.level_cnt - 2

  def set_entry(self, entry_id, paddr, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    if not table:
      x = self.nvdev.pte_t.encode(valid=valid, address_sys=paddr >> 12,
        aperture=2 if aspace is AddrSpace.SYS else 0, kind=6,
        **({'pcf': int(uncached)} if self.nvdev.mmu_ver == 3 else {'vol': uncached}))
    else:
      pde = self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t
      small, sys = ("_small" if self._is_dual_pde() else ""), "" if self.nvdev.mmu_ver == 3 else "_sys"
      x = pde.encode(is_pte=False, **{f'aperture{small}': 1 if valid else 0, f'address{small}{sys}': paddr >> 12},
        **({f'pcf{small}': 0b10} if self.nvdev.mmu_ver == 3 else {'no_ats': 1}))
    if self._is_dual_pde(): self.entries[2 * entry_id], self.entries[2 * entry_id + 1] = x & 0xffffffffffffffff, x >> 64
    else: self.entries[entry_id] = x

  def entry(self, entry_id):
    return (self.entries[2 * entry_id + 1] << 64) | self.entries[2 * entry_id] if self._is_dual_pde() else self.entries[entry_id]

  def read_fields(self, entry_id):
    if self.is_page(entry_id): return self.nvdev.pte_t.decode(self.entry(entry_id))
    return (self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t).decode(self.entry(entry_id))

  def is_page(self, entry_id): return (self.entry(entry_id) & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True

  def supports_huge_page(self, paddr): return self.lv >= self.nvdev.mm.level_cnt - 3 and paddr % self.nvdev.mm.pte_covers[self.lv] == 0

  def valid(self, entry_id):
    if self.is_page(entry_id): return self.read_fields(entry_id)['valid']
    return self.read_fields(entry_id)['aperture_small' if self._is_dual_pde() else 'aperture'] != 0

  def address(self, entry_id):
    small, sys = ("_small" if self._is_dual_pde() else ""), "_sys" if self.nvdev.mmu_ver == 2 or self.lv == self.nvdev.mm.level_cnt - 1 else ""
    return self.read_fields(entry_id)[f'address{small}{sys}'] << 12

class NVMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000)
  def on_range_mapped(self):
    self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev:
  def __init__(self, pci_dev):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')
    self.smi_dev, self.is_booting, self.is_err_state = False, True, False
    self._early_ip_init()
    self._early_mmu_init()
    self.is_booting = False  # past boot phase: alloc_boot_mem can use palloc(boot=False)
    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()


  def fini(self):
    for ip in [self.gsp, self.flcn]: ip.fini_hw()

  def reg(self, reg): return self.__dict__[reg]
  def wreg(self, addr, value):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr): return self.mmio[addr // 4]

  def _early_ip_init(self):
    self.reg_names = set()
    self.reg_offsets = {}
    self.include("nv_ref", "")
    self.include("dev_fb", "tu102")
    self.include("dev_gc6_island", "ga102")
    needs_reset = self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0
    if needs_reset:
      self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
      self.pci_dev.reset()
      time.sleep(0.1)
    self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.chip_id = self.reg("NV_PMC_BOOT_0").read()
    self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
    self.fw_name = {"GB2": "gb202", "AD1": "ad102", "GA1": "ga102"}[self.chip_name[:3]]
    self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)
    # Construct falcon/gsp IPs (NV_FLCN_COT path is GB2xx-only — RTX 3080 is GA102)
    self.flcn = NV_FLCN(self)
    self.gsp = NV_GSP(self)
    if needs_reset: self.flcn.wait_for_reset()

  def _early_mmu_init(self):
    self.include("dev_vm", "tu102")
    self.include("dev_mmu", "gh100" if self.mmu_ver == 3 else "tu102")
    self.pte_t, self.pde_t, self.dual_pde_t = [self.__dict__[name] for name in [f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE', f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE']]
    self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20
    self.vram, self.mmio = self.pci_dev.map_bar(1), self.pci_dev.map_bar(0, fmt='I')
    self.large_bar = self.vram.nbytes >= self.vram_size
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])
    self.mm = NVMemoryManager(self, self.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry,
                              va_bits=bits, va_shifts=shifts, va_base=0,
                              palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]],
                              reserve_ptable=not self.large_bar)

  def _alloc_boot_mem(self, size, data=None, contiguous=False, sysmem=None):
    sz = round_up(size, 0x1000)
    if sysmem is True or (sysmem is None and not self.large_bar):
      view, sysaddr = self.pci_dev.alloc_sysmem(size, 0, contiguous=contiguous)
      paddr = None
    else:
      paddr = self.mm.palloc(sz, boot=False)
      view = self.vram.view(paddr, sz)
      sysaddr = [self.pci_dev.bar_info(1)[0] + paddr + i * 0x1000 for i in range(sz // 0x1000)]
    if data is not None: view[:size] = data
    return view, paddr, sysaddr

  def include(self, name, arch):
    # nv_ref.regs and dev_*.regs are dicts of {reg_name: (base, off_lambda)} tuples in tinygrad.
    src_mod = getattr(nv_regs, name)
    regs = getattr(src_mod, arch or "regs")
    for k, v in regs.items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v


# ============================================================================
# nv/ip.py (vendored from ref/tinygrad/tinygrad/runtime/support/nv/ip.py)
# Slimmed: NV_FLCN (skip NV_FLCN_COT — we run GA102 not GB2xx), NV_GSP,
#          NVRpcQueue (with run_cpu_seq handling).
# ============================================================================
@dataclasses.dataclass(frozen=True)
class GRBufDesc:
  size: int; virt: bool; phys: bool; local: bool = False

class NV_IP:
  def __init__(self, nvdev): self.nvdev = nvdev
  def init_sw(self): pass
  def init_hw(self): pass
  def fini_hw(self): pass

class NVRpcQueue:
  def __init__(self, gsp, view, completion_q_view=None):
    self.tx_view = view.view(fmt='I')
    wait_cond(lambda: self.tx_view[getattr(nv.msgqTxHeader, 'entryOff').offset // 4], value=0x1000, msg="RPC queue not initialized")
    self.tx = nv.msgqTxHeader.from_buffer_copy(bytes(view[:ctypes.sizeof(nv.msgqTxHeader)]))
    if completion_q_view is not None:
      comp_tx = nv.msgqTxHeader.from_buffer_copy(bytes(completion_q_view[:ctypes.sizeof(nv.msgqTxHeader)]))
      self.rx_view = completion_q_view.view(comp_tx.rxHdrOff, fmt='I')
    self.gsp, self.view, self.seq = gsp, view, 0
    self.queue_mv = view.view(self.tx.entryOff, self.tx.msgSize * self.tx.msgCount)

  def _checksum(self, data):
    if (pad_len := (-len(data)) % 8): data += b'\x00' * pad_len
    checksum = 0
    for offset in range(0, len(data), 8): checksum ^= struct.unpack_from('Q', data, offset)[0]
    return hi32(checksum) ^ lo32(checksum)

  def _send_rpc_record(self, func, msg):
    header = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, rpc_result=nv.NV_VGPU_MSG_RESULT_RPC_PENDING,
      rpc_result_private=nv.NV_VGPU_MSG_RESULT_RPC_PENDING, header_version=(3 << 24), function=func, length=len(msg) + 0x20)
    msg = bytes(header) + msg
    phdr = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=ceildiv(len(msg) + ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT), self.tx.msgSize), seqNum=self.seq)
    phdr.checkSum = self._checksum(bytes(phdr) + msg)
    msg = (bytes(phdr) + msg).ljust(phdr.elemCount * self.tx.msgSize, b'\x00')
    wp = self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4]
    off, first = wp * self.tx.msgSize, min(len(msg), len(self.queue_mv) - wp * self.tx.msgSize)
    self.queue_mv[off:off + first] = msg[:first]
    if first < len(msg): self.queue_mv[:len(msg) - first] = msg[first:]
    self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4] = (wp + phdr.elemCount) % self.tx.msgCount
    System.memory_barrier()
    self.seq += 1
    self.gsp.nvdev.NV_PGSP_QUEUE_HEAD[0].write(0x0)

  def send_rpc(self, func, msg):
    max_payload = self.tx.msgSize * 16 - ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT) - ctypes.sizeof(nv.rpc_message_header_v)
    self._send_rpc_record(func, msg[:max_payload])
    for off in range(max_payload, len(msg), max_payload):
      self._send_rpc_record(nv.NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD, msg[off:off + max_payload])

  def read_resp(self):
    System.memory_barrier()
    while self.rx_view[0] != self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4]:
      off = self.rx_view[0] * self.tx.msgSize
      hdr = nv.rpc_message_header_v.from_buffer_copy(bytes(self.queue_mv[off + 0x30:off + 0x30 + ctypes.sizeof(nv.rpc_message_header_v)]))
      msg = bytes(self.queue_mv[off + 0x50:off + 0x50 + hdr.length])
      if hdr.function == nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: self.gsp.run_cpu_seq(msg)
      elif hdr.function == nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG:
        print(f"nv {self.gsp.nvdev.devfmt}: GSP LOG: {msg[12:].rstrip(bytes([0])).decode('utf-8')}")
      self.gsp.nvdev.is_err_state |= hdr.function in {nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG, nv.NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED}
      self.rx_view[0] = (self.rx_view[0] + round_up(hdr.length, self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
      System.memory_barrier()
      if hdr.rpc_result != 0: raise RuntimeError(f"RPC call {hdr.function} failed with result {hdr.rpc_result}")
      yield hdr.function, msg

  def wait_resp(self, cmd, timeout=10000):
    start_time = int(time.perf_counter() * 1000)
    while (int(time.perf_counter() * 1000) - start_time) < timeout:
      if (msg := next((message for func, message in self.read_resp() if func == cmd), None)) is not None: return msg
    raise RuntimeError(f"Timeout waiting for RPC response for command {cmd}")


class NV_FLCN(NV_IP):
  def wait_for_reset(self):
    wait_cond(lambda _: self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.read_bitfields()['read_protection_level0'] == 1
                        and self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0].read() & 0xff == 0xff,
              "waiting for reset")

  def init_sw(self):
    self.nvdev.include("dev_gsp", "ga102")
    self.nvdev.include("dev_falcon_v4", "ga102")
    self.nvdev.include("dev_riscv_pri", "ga102")
    self.nvdev.include("dev_fbif_v4", "ga102")
    self.nvdev.include("dev_falcon_second_pri", "ga102")
    self.nvdev.include("dev_sec_pri", "ga102")
    self.nvdev.include("dev_bus", "tu102")
    self.prep_ucode()
    self.prep_booter()

  def prep_ucode(self):
    vbios_bytes, vbios_off = memoryview(bytes(_array_mod.array('I', self.nvdev.mmio[0x00300000 // 4:(0x00300000 + 0x100000) // 4]))), 0
    while True:
      pci_blck = vbios_bytes[vbios_off + nv.OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR:].cast('H')[0]
      imglen = vbios_bytes[vbios_off + pci_blck + nv.OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN:].cast('H')[0] * nv.PCI_ROM_IMAGE_BLOCK_SIZE
      match vbios_bytes[vbios_off + pci_blck + nv.OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE]:
        case nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE: block_size = imglen
        case nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT:
          expansion_rom_off = vbios_off - block_size
          break
      vbios_off += imglen
    bit_header = nv.BIT_HEADER_V1_00.from_buffer_copy(vbios_bytes[(bit_addr := 0x1b0):bit_addr + ctypes.sizeof(nv.BIT_HEADER_V1_00)])
    assert bit_header.Signature == 0x00544942, f"Invalid BIT header signature {hex(bit_header.Signature)}"
    for i in range(bit_header.TokenEntries):
      bit = nv.BIT_TOKEN_V1_00.from_buffer_copy(vbios_bytes[bit_addr + bit_header.HeaderSize + i * bit_header.TokenSize:])
      if bit.TokenId != nv.BIT_TOKEN_FALCON_DATA or bit.DataVersion != 2 or bit.DataSize < nv.BIT_DATA_FALCON_DATA_V2_SIZE_4: continue
      falcon_data = nv.BIT_DATA_FALCON_DATA_V2.from_buffer_copy(vbios_bytes[bit.DataPtr & 0xffff:])
      ucode_hdr = nv.FALCON_UCODE_TABLE_HDR_V1.from_buffer_copy(vbios_bytes[(table_ptr := expansion_rom_off + falcon_data.FalconUcodeTablePtr):])
      for j in range(ucode_hdr.EntryCount):
        ucode_entry = nv.FALCON_UCODE_TABLE_ENTRY_V1.from_buffer_copy(vbios_bytes[table_ptr + ucode_hdr.HeaderSize + j * ucode_hdr.EntrySize:])
        if ucode_entry.ApplicationID != nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: continue
        ucode_desc_hdr = nv.FALCON_UCODE_DESC_HEADER.from_buffer_copy(vbios_bytes[expansion_rom_off + ucode_entry.DescPtr:])
        ucode_desc_off = expansion_rom_off + ucode_entry.DescPtr
        ucode_desc_size = ucode_desc_hdr.vDesc >> 16
    self.desc_v3 = nv.FALCON_UCODE_DESC_V3.from_buffer_copy(vbios_bytes[ucode_desc_off:ucode_desc_off + ucode_desc_size])
    sig_total_size = ucode_desc_size - nv.FALCON_UCODE_DESC_V3_SIZE_44
    signature = vbios_bytes[ucode_desc_off + nv.FALCON_UCODE_DESC_V3_SIZE_44:][:sig_total_size]
    image = vbios_bytes[ucode_desc_off + ucode_desc_size:][:round_up(self.desc_v3.StoredSize, 256)]
    self.frts_offset = self.nvdev.vram_size - 0x100000 - 0x100000
    read_vbios_desc = nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = nv.FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_FRTS_REGION_DESC),
      frtsRegionOffset4K=self.frts_offset >> 12, frtsRegionSize=0x100, frtsRegionMediaType=2)
    frts_cmd = nv.FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)

    def __patch(cmd_id, cmd):
      patched_image = bytearray(image)
      dmem_offset = 0
      hdr = nv.FALCON_APPLICATION_INTERFACE_HEADER_V1.from_buffer_copy(image[(app_hdr_off := self.desc_v3.IMEMLoadSize + self.desc_v3.InterfaceOffset):])
      ents = (nv.FALCON_APPLICATION_INTERFACE_ENTRY_V1 * hdr.entryCount).from_buffer_copy(image[app_hdr_off + ctypes.sizeof(hdr):])
      for i in range(hdr.entryCount):
        if ents[i].id == nv.FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER: dmem_offset = ents[i].dmemOffset
      dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[(dmem_mapper_offset := self.desc_v3.IMEMLoadSize + dmem_offset):])
      dmem.init_cmd = cmd_id
      patched_image[dmem_mapper_offset:dmem_mapper_offset + len(bytes(dmem))] = bytes(dmem)
      patched_image[(cmd_off := self.desc_v3.IMEMLoadSize + dmem.cmd_in_buffer_offset):cmd_off + len(cmd)] = cmd
      patched_image[(sig_off := self.desc_v3.IMEMLoadSize + self.desc_v3.PKCDataOffset):sig_off + 0x180] = signature[-0x180:]
      return self.nvdev._alloc_boot_mem(len(patched_image), data=patched_image, sysmem=False)

    _, self.frts_image_paddr, _ = __patch(0x15, bytes(frts_cmd))

  def prep_booter(self):
    sha = {"ga102": "4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b",
           "ad102": "8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a"}[self.nvdev.fw_name]
    h = nv.struct_nvfw_bin_hdr.from_buffer_copy(b := fetch_fw(f"nvidia/{self.nvdev.fw_name}/gsp", "booter_load-570.144.bin", sha))
    lh = nv.struct_nvfw_hs_load_header_v2.from_buffer_copy(b, (hs := nv.struct_nvfw_hs_header_v2.from_buffer_copy(b, h.header_offset)).header_offset)
    app = nv.struct_nvfw_hs_load_header_v2_app.from_buffer_copy(b, hs.header_offset + ctypes.sizeof(nv.struct_nvfw_hs_load_header_v2))
    patch_loc, patch_sig = struct.unpack_from("<I", b, hs.patch_loc)[0], struct.unpack_from("<I", b, hs.patch_sig)[0]
    sig = b[(sig_off := hs.sig_prod_offset + patch_sig):sig_off + (sig_len := hs.sig_prod_size // struct.unpack_from("<I", b, hs.num_sig)[0])]
    (patched_image := bytearray(b[h.data_offset:h.data_offset + h.data_size]))[patch_loc:patch_loc + sig_len] = sig
    _, self.booter_image_paddr, _ = self.nvdev._alloc_boot_mem(len(patched_image), data=patched_image, sysmem=False)
    self.booter_data_off, self.booter_data_sz, self.booter_code_off, self.booter_code_sz = lh.os_data_offset, lh.os_data_size, app.offset, app.size

  def init_hw(self):
    self.falcon, self.sec2 = 0x00110000, 0x00840000
    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_paddr, code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
      imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
      dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
      pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)
    assert self.nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() != 0, "WPR2 is not initialized"
    self.reset(self.falcon, riscv=True)
    self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.nvdev.gsp.libos_args_sysmem))
    self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.nvdev.gsp.libos_args_sysmem))
    self.reset(self.sec2)
    mbx = self.execute_hs(self.sec2, self.booter_image_paddr, code_off=self.booter_code_off, data_off=self.booter_data_off,
      imemPa=0x0, imemVa=self.booter_code_off, imemSz=self.booter_code_sz, dmemPa=0x0, dmemVa=0x0, dmemSz=self.booter_data_sz,
      pkc_off=0x10, engid=1, ucodeid=3, mailbox=self.nvdev.gsp.wpr_meta_sysmem)
    assert mbx[0] == 0x0, f"Booter failed to execute, mailbox is {mbx[0]:08x}, {mbx[1]:08x}"
    self.nvdev.NV_PFALCON_FALCON_OS.with_base(self.falcon).write(0x0)
    assert self.nvdev.NV_PRISCV_RISCV_CPUCTL.with_base(self.falcon).read_bitfields()['active_stat'] == 1, "GSP Core is not active"

  def execute_dma(self, base, cmd, dest, mem_off, src, size):
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA does not progress")
    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE.with_base(base).write(lo32(src >> 8))
    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE1.with_base(base).write(hi32(src >> 8) & 0x1ff)
    xfered = 0
    while xfered < size:
      wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA does not progress")
      self.nvdev.NV_PFALCON_FALCON_DMATRFMOFFS.with_base(base).write(dest + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFFBOFFS.with_base(base).write(mem_off + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).write(cmd)
      xfered += 256
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['idle'], msg="DMA does not complete")

  def start_cpu(self, base):
    if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['alias_en'] == 1:
      self.nvdev.wreg(base + self.nvdev.NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2)
    else:
      self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).write(startcpu=1)

  def wait_cpu_halted(self, base):
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['halted'], msg="not halted")

  def execute_hs(self, base, img_paddr, code_off, data_off, imemPa, imemVa, imemSz, dmemPa, dmemVa, dmemSz, pkc_off, engid, ucodeid, mailbox=None):
    self.disable_ctx_req(base)
    self.nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base(base)[ctx_dma := 0].update(target=0, mem_type=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL)
    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=1, sec=1)
    self.execute_dma(base, cmd, dest=imemPa, mem_off=imemVa, src=img_paddr + code_off - imemVa, size=imemSz)
    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=0, sec=0)
    self.execute_dma(base, cmd, dest=dmemPa, mem_off=dmemVa, src=img_paddr + data_off - dmemVa, size=dmemSz)
    self.nvdev.NV_PFALCON2_FALCON_BROM_PARAADDR.with_base(base)[0].write(pkc_off)
    self.nvdev.NV_PFALCON2_FALCON_BROM_ENGIDMASK.with_base(base).write(engid)
    self.nvdev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID.with_base(base).write(val=ucodeid)
    self.nvdev.NV_PFALCON2_FALCON_MOD_SEL.with_base(base).write(algo=self.nvdev.NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K)
    self.nvdev.NV_PFALCON_FALCON_BOOTVEC.with_base(base).write(imemVa)
    if mailbox is not None:
      self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).write(lo32(mailbox))
      self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).write(hi32(mailbox))
    self.start_cpu(base)
    self.wait_cpu_halted(base)
    if mailbox is not None:
      return self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).read(), self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).read()

  def disable_ctx_req(self, base):
    self.nvdev.NV_PFALCON_FBIF_CTL.with_base(base).update(allow_phys_no_ctx=1)
    self.nvdev.NV_PFALCON_FALCON_DMACTL.with_base(base).write(0x0)

  def reset(self, base, riscv=False):
    engine_reg = self.nvdev.NV_PGSP_FALCON_ENGINE if base == self.falcon else self.nvdev.NV_PSEC_FALCON_ENGINE
    engine_reg.write(reset=1)
    time.sleep(0.1)
    engine_reg.write(reset=0)
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['mem_scrubbing'], value=0, msg="Scrubbing not completed")
    if riscv:
      self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=1, valid=0, brfetch=1)
    elif self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['riscv'] == 1:
      self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=0)
      wait_cond(lambda: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).read_bitfields()['valid'], msg="RISCV core not booted")
      self.nvdev.NV_PFALCON_FALCON_RM.with_base(base).write(self.nvdev.chip_id)


class NV_GSP(NV_IP):
  def init_sw(self):
    self.handle_gen = itertools.count(0xcf000000)
    self.init_rm_args()
    self.init_libos_args()
    self.init_wpr_meta()
    self.rpc_set_gsp_system_info()
    self.rpc_set_registry_table()
    self.gpfifo_class, self.compute_class, self.dma_class = nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_B, nv_gpu.AMPERE_DMA_COPY_B
    match self.nvdev.chip_name[:2]:
      case "AD": self.compute_class = nv_gpu.ADA_COMPUTE_A
      case "GB": self.gpfifo_class, self.compute_class, self.dma_class = nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.BLACKWELL_DMA_COPY_B

  def init_rm_args(self, queue_size=0x40000):
    pte_cnt = ((queue_pte_cnt := (queue_size * 2) // 0x1000)) + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
    pt_size = round_up(pte_cnt * 8, 0x1000)
    queues_view, _, queues_sysmem = self.nvdev._alloc_boot_mem(pt_size + queue_size * 2, sysmem=True)
    for i, sysmem in enumerate(queues_sysmem): queues_view.view(i * 0x8, 0x8, fmt='Q')[0] = sysmem
    queue_args = nv.MESSAGE_QUEUE_INIT_ARGUMENTS(sharedMemPhysAddr=queues_sysmem[0], pageTableEntryCount=pte_cnt,
      cmdQueueOffset=pt_size, statQueueOffset=pt_size + queue_size)
    _, _, rm_args_addrs = self.nvdev._alloc_boot_mem(ctypes.sizeof(nv.GSP_ARGUMENTS_CACHED),
      data=bytes(nv.GSP_ARGUMENTS_CACHED(bDmemStack=True, messageQueueInitArguments=queue_args)))
    self.rm_args_sysmem = rm_args_addrs[0]
    self.cmd_q_view, self.stat_q_view = queues_view.view(pt_size), queues_view.view(pt_size + queue_size)
    self.cmd_q_view[:ctypes.sizeof(nv.msgqTxHeader)] = bytes(nv.msgqTxHeader(version=0, size=queue_size, entryOff=0x1000, msgSize=0x1000,
      msgCount=(queue_size - 0x1000) // 0x1000, writePtr=0, flags=1, rxHdrOff=ctypes.sizeof(nv.msgqTxHeader)))
    self.cmd_q = NVRpcQueue(self, self.cmd_q_view, None)

  def init_libos_args(self):
    _, _, logbuf_addrs = self.nvdev._alloc_boot_mem(2 << 20)
    libos_args_view, _, libos_addrs = self.nvdev._alloc_boot_mem(0x1000)
    self.libos_args_sysmem = libos_addrs[0]
    libos_structs = [nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x10000,
        id8=int.from_bytes(bytes(f"LOG{name}", 'utf-8'), 'big'), pa=logbuf_addrs[0] + 0x10000 * i)
        for i, name in enumerate(["INIT", "INTR", "RM", "MNOC", "KRNL"])]
    libos_structs.append(nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM,
      size=0x1000, id8=int.from_bytes(bytes("RMARGS", 'utf-8'), 'big'), pa=self.rm_args_sysmem))
    libos_args_view[:sum(ctypes.sizeof(s) for s in libos_structs)] = b''.join(bytes(s) for s in libos_structs)

  def init_gsp_image(self):
    _, sections, _ = elf_loader(fetch_fw("nvidia/ga102/gsp", "gsp-570.144.bin", "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f"))
    self.gsp_image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    signature = next((sh.content for sh in sections if sh.name == (f".fwsignature_{self.nvdev.chip_name[:4].lower()}x")))
    npages = [0, 0, 0, round_up(len(self.gsp_image), 0x1000) // 0x1000]
    for i in range(3, 0, -1): npages[i - 1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1
    offsets = [sum(npages[:i]) * 0x1000 for i in range(4)]
    radix_view, _, self.gsp_radix3_addrs = self.nvdev._alloc_boot_mem(offsets[-1] + len(self.gsp_image))
    radix_view.view(offsets[-1], len(self.gsp_image))[:] = self.gsp_image
    for i in range(0, 3):
      cur_offset = sum(npages[:i + 1])
      radix_view.view(offsets[i], npages[i + 1] * 8, fmt='Q')[:] = _array_mod.array('Q', self.gsp_radix3_addrs[cur_offset:cur_offset + npages[i + 1]])
    _, _, gsp_sig_addrs = self.nvdev._alloc_boot_mem(len(signature), data=signature)
    self.gsp_signature_bar1 = gsp_sig_addrs[0]

  def init_boot_binary_image(self):
    sha = {"ga102": "82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad",
           "ad102": "65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8",
           "gb202": "d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071"}[self.nvdev.fw_name]
    h = nv.struct_nvfw_bin_hdr.from_buffer_copy(b := fetch_fw(f"nvidia/{self.nvdev.fw_name}/gsp", "bootloader-570.144.bin", sha))
    self.booter_image, self.booter_desc = b[h.data_offset:h.data_offset + h.data_size], nv.RM_RISCV_UCODE_DESC.from_buffer_copy(b, h.header_offset)
    _, _, booter_addrs = self.nvdev._alloc_boot_mem(len(self.booter_image), data=self.booter_image)
    self.booter_bar1 = booter_addrs[0]

  def init_wpr_meta(self):
    self.init_gsp_image()
    self.init_boot_binary_image()
    common = {'sizeOfBootloader': (boot_sz := len(self.booter_image)), 'sysmemAddrOfBootloader': self.booter_bar1,
      'sizeOfRadix3Elf': (radix3_sz := len(self.gsp_image)), 'sysmemAddrOfRadix3Elf': self.gsp_radix3_addrs[0],
      'sizeOfSignature': 0x1000, 'sysmemAddrOfSignature': self.gsp_signature_bar1,
      'bootloaderCodeOffset': self.booter_desc.monitorCodeOffset, 'bootloaderDataOffset': self.booter_desc.monitorDataOffset,
      'bootloaderManifestOffset': self.booter_desc.manifestOffset, 'revision': nv.GSP_FW_WPR_META_REVISION, 'magic': nv.GSP_FW_WPR_META_MAGIC}
    if self.nvdev.fmc_boot:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=0x20000, pmuReservedSize=0x1820000, nonWprHeapSize=0x220000,
        gspFwHeapSize=0x8700000, frtsSize=0x100000)
    else:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=(vga_sz := 0x100000), vgaWorkspaceOffset=(vga_off := self.nvdev.vram_size - vga_sz),
        gspFwWprEnd=vga_off, frtsSize=(frts_sz := 0x100000), frtsOffset=(frts_off := vga_off - frts_sz),
        bootBinOffset=(boot_off := frts_off - boot_sz),
        gspFwOffset=(gsp_off := round_down(boot_off - radix3_sz, 0x10000)), gspFwHeapSize=(gsp_heap_sz := 0x8100000),
        fbSize=self.nvdev.vram_size,
        gspFwHeapOffset=(gsp_heap_off := round_down(gsp_off - gsp_heap_sz, 0x100000)),
        gspFwWprStart=(wpr_st := round_down(gsp_heap_off - 0x1000, 0x100000)),
        nonWprHeapSize=(non_wpr_sz := 0x100000), nonWprHeapOffset=(non_wpr_off := round_down(wpr_st - non_wpr_sz, 0x100000)),
        gspFwRsvdStart=non_wpr_off)
      assert self.nvdev.flcn.frts_offset == m.frtsOffset, f"FRTS mismatch: {self.nvdev.flcn.frts_offset} != {m.frtsOffset}"
    self.wpr_meta, _, wpr_meta_addrs = self.nvdev._alloc_boot_mem(ctypes.sizeof(type(m)), data=bytes(m))
    self.wpr_meta_sysmem = wpr_meta_addrs[0]

  def promote_ctx(self, client, subdevice, obj, ctxbufs, bufs=None, virt=None, phys=None):
    res, prom = {}, nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(ctxbufs), engineType=0x1, hChanClient=client, hObject=obj)
    for i, (buf, desc) in enumerate(ctxbufs.items()):
      use_v, use_p = (desc.virt if virt is None else virt), (desc.phys if phys is None else phys)
      x = (bufs or {}).get(buf, self.nvdev.mm.valloc(desc.size, contiguous=True))
      prom.promoteEntry[i] = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY(bufferId=buf,
        gpuVirtAddr=x.va_addr if use_v else 0, bInitialize=use_p,
        gpuPhysAddr=x.paddrs[0][0] if use_p else 0, size=desc.size if use_p else 0,
        physAttr=0x4 if use_p else 0, bNonmapped=(use_p and not use_v))
      res[buf] = x
    self.rpc_rm_control(hObject=subdevice, cmd=nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX, params=prom, client=client)
    return res

  def init_golden_image(self):
    self.rpc_rm_alloc(hParent=0x0, hClass=0x0, params=nv_gpu.NV0000_ALLOC_PARAMETERS())
    dev = self.rpc_rm_alloc(hParent=self.priv_root, hClass=nv_gpu.NV01_DEVICE_0,
      params=nv_gpu.NV0080_ALLOC_PARAMETERS(hClientShare=self.priv_root))
    subdev = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.NV20_SUBDEVICE_0, params=nv_gpu.NV2080_ALLOC_PARAMETERS())
    vaspace = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.FERMI_VASPACE_A, params=nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS())
    self.vaspace = vaspace  # exposed for NVDevice.__init__ reuse
    res_va = self.nvdev.mm.alloc_vaddr(res_sz := (512 << 20))
    bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(
      pageSize=res_sz, numLevelsToCopy=3, virtAddrLo=res_va, virtAddrHi=res_va + res_sz - 1)
    for i, pt in enumerate(self.nvdev.mm.page_tables(res_va, size=res_sz)):
      bufs_p.levels[i] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_level(
        physAddress=pt.paddr, size=self.nvdev.mm.pte_cnt[0] * 8 if i == 0 else 0x1000,
        pageShift=self.nvdev.mm.pte_covers[i].bit_length() - 1, aperture=1)
    self.rpc_rm_control(hObject=vaspace, cmd=nv_gpu.NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, params=bufs_p)
    gpfifo_area = self.nvdev.mm.valloc(4 << 10, contiguous=True)
    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.paddrs[0][0] + 0x20 * 8, size=0x20, addressSpace=2, cacheAttrib=0)
    gg_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr, gpFifoEntries=32,
      engineType=0x1, cid=3, hVASpace=vaspace, userdOffset=(ctypes.c_uint64 * 8)(0x20 * 8),
      userdMem=userd, internalFlags=0x1a, flags=0x200320)
    ch_gpfifo = self.rpc_rm_alloc(hParent=dev, hClass=self.gpfifo_class, params=gg_params)
    gr_ctx_bufs_info = self.rpc_rm_control(hObject=subdev,
      cmd=nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO,
      params=nv_gpu.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS()).engineContextBuffersInfo[0]
    def _ctx_info(idx, add=0, align=None):
      return round_up(gr_ctx_bufs_info.engine[idx].size + add, align or gr_ctx_bufs_info.engine[idx].alignment)
    gr_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS, add=0x40000)
    patch_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH)
    cfgs_sizes = {x: _ctx_info(x + 14, align=(2 << 20) if x == 5 else None) for x in range(3, 11)}
    self.grctx_bufs = {0: GRBufDesc(gr_size, phys=True, virt=True),
      1: GRBufDesc(patch_size, phys=True, virt=True, local=True),
      2: GRBufDesc(patch_size, phys=True, virt=True),
      **{x: GRBufDesc(cfgs_sizes[x], phys=False, virt=True) for x in range(3, 7)},
      9: GRBufDesc(cfgs_sizes[9], phys=True, virt=True), 10: GRBufDesc(cfgs_sizes[10], phys=True, virt=False),
      11: GRBufDesc(cfgs_sizes[10], phys=True, virt=True)}
    self.promote_ctx(self.priv_root, subdev, ch_gpfifo, {k: v for k, v in self.grctx_bufs.items() if not v.local})
    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.compute_class, params=None)
    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.dma_class, params=None)
    return ch_gpfifo, subdev, dev, vaspace, gpfifo_area

  def init_hw(self):
    self.stat_q = NVRpcQueue(self, self.stat_q_view, self.cmd_q_view)
    self.cmd_q.rx_view = self.stat_q_view.view(self.stat_q.tx.rxHdrOff, fmt='I')
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE)
    self.nvdev.NV_PBUS_BAR1_BLOCK.write(mode=0, target=0, ptr=0)
    if self.nvdev.fmc_boot: self.nvdev.NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR.write(mode=0, target=0, ptr=0)
    self.priv_root = 0xc1e00004
    return self.init_golden_image()

  def fini_hw(self): self.rpc_unloading_guest_driver()

  def rpc_alloc_memory(self, hDevice, hClass, paddrs, length, flags, client=None):
    assert all(sz == 0x1000 for _, sz in paddrs)
    rpc = nv.rpc_alloc_memory_v(hClient=(client := client or self.priv_root), hDevice=hDevice, hMemory=(handle := next(self.handle_gen)),
      hClass=hClass, flags=flags, pteAdjust=0, format=6, length=length, pageCount=len(paddrs))
    rpc.pteDesc.idr, rpc.pteDesc.length = nv.NV_VGPU_PTEDESC_IDR_NONE, (len(paddrs) & 0xffff)
    payload = bytes(rpc) + b''.join(bytes(nv.struct_pte_desc_pte_pde(pte=(paddr >> 12))) for paddr, _ in paddrs)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY, bytes(payload))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY)
    return handle

  def rpc_rm_alloc(self, hParent, hClass, params, client=None):
    if hClass == self.gpfifo_class:
      ramfc_alloc = self.nvdev.mm.valloc(0x1000, contiguous=True)
      params.ramfcMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x200, addressSpace=2, cacheAttrib=0)
      params.instanceMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x1000, addressSpace=2, cacheAttrib=0)
      _, method_paddr, _ = self.nvdev._alloc_boot_mem(0x5000, sysmem=False)
      params.mthdbufMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=method_paddr, size=0x5000, addressSpace=2, cacheAttrib=0)
      if client is not None and client != self.priv_root and params.hObjectError != 0:
        params.errorNotifierMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=0, size=0xecc, addressSpace=0, cacheAttrib=0)
        params.userdMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=params.hUserdMemory[0] + params.userdOffset[0], size=0x400, addressSpace=2, cacheAttrib=0)
    alloc_args = nv.rpc_gsp_rm_alloc_v(hClient=(client := client or self.priv_root), hParent=hParent,
      hObject=(obj := next(self.handle_gen)), hClass=hClass, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, bytes(alloc_args) + (bytes(params) if params is not None else b''))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC)
    if hClass == nv_gpu.FERMI_VASPACE_A and client != self.priv_root:
      self.rpc_set_page_directory(device=hParent, hVASpace=obj, pdir_paddr=self.nvdev.mm.root_page_table.paddr, client=client)
    if hClass == nv_gpu.NV01_DEVICE_0 and client != self.priv_root: self.device = obj
    if hClass == nv_gpu.NV20_SUBDEVICE_0: self.subdevice = obj
    if hClass == self.compute_class and client != self.priv_root:
      phys_gr_ctx = self.promote_ctx(client, self.subdevice, hParent, {k: v for k, v in self.grctx_bufs.items() if k in [0, 1, 2]}, virt=False)
      self.promote_ctx(client, self.subdevice, hParent, {k: v for k, v in self.grctx_bufs.items() if k in [0, 1, 2]}, phys_gr_ctx, phys=False)
    return obj if hClass != nv_gpu.NV1_ROOT else client

  def rpc_rm_control(self, hObject, cmd, params, client=None, extra=None):
    if cmd == nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES:
      self.rpc_rm_control(hObject, nv_gpu.NVB0CC_CTRL_CMD_INTERNAL_PERMISSIONS_INIT,
        nv_gpu.NVB0CC_CTRL_INTERNAL_PERMISSIONS_INIT_PARAMS(
          bAdminProfilingPermitted=1, bDevProfilingPermitted=1, bCtxProfilingPermitted=1,
          bVideoMemoryProfilingPermitted=1, bSysMemoryProfilingPermitted=1), client=client)
    elif cmd == nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM:
      params.hMemPmaBuffer = self.rpc_alloc_memory(self.device, nv_gpu.NV01_MEMORY_LIST_SYSTEM, extra[0].meta.mapping.paddrs, extra[0].size,
        pma_flags := (nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4 | nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30), client=client)
      params.hMemPmaBytesAvailable = self.rpc_alloc_memory(self.device, nv_gpu.NV01_MEMORY_LIST_SYSTEM, extra[1].meta.mapping.paddrs, extra[1].size,
        pma_flags | nv_gpu.NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES << 21, client=client)
    control_args = nv.rpc_gsp_rm_control_v(hClient=(client := client or self.priv_root), hObject=hObject, cmd=cmd, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, bytes(control_args) + (bytes(params) if params is not None else b''))
    res = self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL)
    st = type(params).from_buffer_copy(res[len(bytes(control_args)):]) if params is not None else None
    if self.nvdev.chip_name.startswith("GB2") and cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      cast(nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS, st).workSubmitToken |= (1 << 30)
    return st

  def rpc_set_page_directory(self, device, hVASpace, pdir_paddr, client=None, pasid=0xffffffff):
    params = nv.struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05(physAddress=pdir_paddr,
      numEntries=self.nvdev.mm.pte_cnt[0], flags=0x8, hVASpace=hVASpace, pasid=pasid, subDeviceId=1, chId=0)
    alloc_args = nv.rpc_set_page_directory_v(hClient=client or self.priv_root, hDevice=device, pasid=pasid, params=params)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, bytes(alloc_args))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY)

  def rpc_set_gsp_system_info(self):
    def bdf_as_int(s):
      if s.startswith("usb") or s.startswith("remote"): return 0x000
      return (int(s[5:7], 16) << 8) | (int(s[8:10], 16) << 3) | int(s[-1], 16)
    pcidev = self.nvdev.pci_dev
    data = nv.GspSystemInfo(gpuPhysAddr=pcidev.bar_info(0)[0], gpuPhysFbAddr=pcidev.bar_info(1)[0],
      gpuPhysInstAddr=pcidev.bar_info(3)[0],
      pciConfigMirrorBase=[0x88000, 0x92000][self.nvdev.fmc_boot], pciConfigMirrorSize=0x1000,
      nvDomainBusDeviceFunc=bdf_as_int(self.nvdev.devfmt), bIsPassthru=1,
      PCIDeviceID=pcidev.read_config(pci.PCI_VENDOR_ID, 4), PCISubDeviceID=pcidev.read_config(pci.PCI_SUBSYSTEM_VENDOR_ID, 4),
      PCIRevisionID=pcidev.read_config(pci.PCI_REVISION_ID, 1), maxUserVa=0x7ffffffff000)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, bytes(data))

  def rpc_unloading_guest_driver(self):
    data = nv.rpc_unloading_guest_driver_v(bInPMTransition=0, bGc6Entering=0, newLevel=1 << 6)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, bytes(data))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER)

  def rpc_set_registry_table(self):
    table = {'RMForcePcieConfigSave': 0x1, 'RMSecBusResetEnable': 0x1}
    entries_bytes, data_bytes = bytes(), bytes()
    hdr_size, entries_size = ctypes.sizeof(nv.PACKED_REGISTRY_TABLE), ctypes.sizeof(nv.PACKED_REGISTRY_ENTRY) * len(table)
    for k, v in table.items():
      entries_bytes += bytes(nv.PACKED_REGISTRY_ENTRY(nameOffset=hdr_size + entries_size + len(data_bytes),
        type=nv.REGISTRY_TABLE_ENTRY_TYPE_DWORD, data=v, length=4))
      data_bytes += k.encode('utf-8') + b'\x00'
    header = nv.PACKED_REGISTRY_TABLE(size=hdr_size + len(entries_bytes) + len(data_bytes), numEntries=len(table))
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_REGISTRY, bytes(header) + entries_bytes + data_bytes)

  def run_cpu_seq(self, seq_buf):
    hdr = nv.rpc_run_cpu_sequencer_v17_00.from_buffer_copy(seq_buf[:(hdr_sz := ctypes.sizeof(nv.rpc_run_cpu_sequencer_v17_00))])
    cmd_iter = iter(memoryview(seq_buf[hdr_sz:]).cast('I')[:hdr.cmdIndex])
    for op in cmd_iter:
      if op == 0x0: self.nvdev.wreg(next(cmd_iter), next(cmd_iter))
      elif op == 0x1:
        addr, val, mask = next(cmd_iter), next(cmd_iter), next(cmd_iter)
        self.nvdev.wreg(addr, (self.nvdev.rreg(addr) & ~mask) | (val & mask))
      elif op == 0x2:
        addr, mask, val, _, _ = next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter)
        wait_cond(lambda a, m: (self.nvdev.rreg(a) & m), addr, mask, value=val, msg=f"Register {addr:#x} not equal to {val:#x}")
      elif op == 0x3: time.sleep(next(cmd_iter) / 1e6)
      elif op == 0x4:
        addr, index = next(cmd_iter), next(cmd_iter)
        hdr.regSaveArea[index] = self.nvdev.rreg(addr)
      elif op == 0x5:
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon)
        self.nvdev.flcn.disable_ctx_req(self.nvdev.flcn.falcon)
      elif op == 0x6: self.nvdev.flcn.start_cpu(self.nvdev.flcn.falcon)
      elif op == 0x7: self.nvdev.flcn.wait_cpu_halted(self.nvdev.flcn.falcon)
      elif op == 0x8:
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon, riscv=True)
        self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.libos_args_sysmem))
        self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.libos_args_sysmem))
        self.nvdev.flcn.start_cpu(self.nvdev.flcn.sec2)
        wait_cond(lambda: self.nvdev.NV_PGC6_BSI_SECURE_SCRATCH_14.read_bitfields()['boot_stage_3_handoff'], msg="SEC2 didn't hand off")
        mailbox = self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(self.nvdev.flcn.sec2).read()
        assert mailbox == 0x0, f"Falcon SEC2 failed to execute, mailbox is {mailbox:08x}"
      else: raise ValueError(f"Unknown op code {op} in run_cpu_seq")


# ============================================================================
# ops_nv.py slices (vendored from ref/tinygrad/tinygrad/runtime/ops_nv.py)
# Slimmed: GPFifo, PCIIface, NVProgram, NVAllocator, NVDevice, NVSignal,
#          NVComputeQueue (minimal), QMD, NVArgsState.
# ============================================================================
SignalType = TypeVar('SignalType', bound='HCQSignal')

class HCQCompiled:
  peer_groups: ClassVar[dict] = {}
  signal_t: ClassVar[type] = None  # set by NVSignal below

class NVSignal:
  def __init__(self, value=0, owner=None):
    self.value_addr = 0  # assigned by NVDevice.signal_page
    self._value = value
    self.owner = owner
  @property
  def value(self):
    if self.owner is not None and hasattr(self.owner, '_signal_page') and self.value_addr:
      return int(self.owner._signal_page.cpu_view().view(0, 8, 'Q')[0])
    return self._value
  @value.setter
  def value(self, v):
    self._value = v
    if self.owner is not None and hasattr(self.owner, '_signal_page') and self.value_addr:
      self.owner._signal_page.cpu_view().view(0, 8, 'Q')[0] = v
  def _sleep(self, time_ms: int):
    if time_ms > 200 and self.owner is not None: self.owner.iface.sleep(200)

@dataclasses.dataclass
class GPFifo:
  ring: MMIOInterface
  gpput: MMIOInterface
  entries_count: int
  token: int
  put_value: int = 0

# -------- NV command queues (from ref/tinygrad ops_nv.py) --------
class NVCommandQueue:
  def __init__(self):
    self._q: list[int] = []
    self.active_qmd = None

  def nvm(self, subchannel, mthd, *args, typ=2):
    self._q.extend([(typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2), *args])
    return self

  def setup(self, compute_class=None, copy_class=None, local_mem_window=None, shared_mem_window=None,
            local_mem=None, local_mem_tpc_bytes=None):
    if compute_class: self.nvm(1, nv_gpu.NVC6C0_SET_OBJECT, compute_class)
    if copy_class: self.nvm(4, nv_gpu.NVC6C0_SET_OBJECT, copy_class)
    if local_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(local_mem_window))
    if shared_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(shared_mem_window))
    if local_mem: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(local_mem))
    if local_mem_tpc_bytes: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(local_mem_tpc_bytes), 0xff)
    return self

  def wait(self, signal, value=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             nv_flags("NVC56F_SEM_EXECUTE", operation="acq_circ_geq", payload_size="64bit"))
    self.active_qmd = None
    return self

  def signal(self, signal, value=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit", release_timestamp="en"))
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    self.active_qmd = None
    return self

  def _submit_to_gpfifo(self, dev, gpfifo: GPFifo):
    cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4, 16)
    cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
    dev.cmdq[cmdq_wptr:cmdq_wptr + len(self._q)] = _array_mod.array('I', [w & 0xffffffff for w in self._q])
    dev.cmdq[cmdq_wptr:cmdq_wptr + len(self._q)] = _array_mod.array('I', [w & 0xffffffff for w in self._q])
    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr // 4 << 2) | (len(self._q) << 42) | (1 << 41)
    gpfifo.gpput[0] = (gpfifo.put_value + 1) % gpfifo.entries_count
    System.memory_barrier()
    dev.gpu_mmio[0x90 // 4] = gpfifo.token
    gpfifo.put_value += 1
    if not dev.iface.is_local():
      dev.iface.sleep(200)
class NVComputeQueue(NVCommandQueue):
  def submit(self, dev):
    self._submit_to_gpfifo(dev, dev.compute_gpfifo)
    return self

class NVCopyQueue(NVCommandQueue):
  def submit(self, dev):
    self._submit_to_gpfifo(dev, dev.dma_gpfifo)
    return self

def submit_gpfifo(dev, words, fifo=None):
  """Push a pre-built method stream to compute (default) or copy GPFIFO."""
  q = NVComputeQueue()
  q._q = list(words)
  q._submit_to_gpfifo(dev, fifo or dev.compute_gpfifo)

@dataclasses.dataclass(frozen=True)
class PCIAllocationMeta:
  mapping: Any
  has_cpu_mapping: bool
  hMemory: int = 0

# -------- QMD ----------
class QMD:
  fields: dict = {}

  def __init__(self, dev, view=None, **kwargs):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)
    if (pref := "NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00") not in QMD.fields:
      QMD.fields[pref] = {**{name[len(pref) + 1:]: dt for name, dt in nv_gpu.__dict__.items() if name.startswith(pref) and isinstance(dt, tuple)},
        **{name[len(pref) + 1:] + f"_{i}": dt(i) for name, dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith(pref) and callable(dt)}}
    self.mv, self.pref = (memoryview(bytearray(self.sz * 4)) if view is None else view), pref
    if kwargs: self.write(**kwargs)

  def _rw_bits(self, hi, lo, value=None):
    mask = ((1 << (width := hi - lo + 1)) - 1) << (lo % 8)
    num = int.from_bytes(self.mv[lo // 8:hi // 8 + 1], "little")
    if value is None: return (num & mask) >> (lo % 8)
    if value >= (1 << width): raise ValueError(f"{value:#x} does not fit.")
    self.mv[lo // 8:hi // 8 + 1] = int((num & ~mask) | ((value << (lo % 8)) & mask)).to_bytes((hi // 8 - lo // 8 + 1), "little")

  def write(self, **kwargs):
    for k, val in kwargs.items(): self._rw_bits(*QMD.fields[self.pref][k.upper()], value=val)
  def read(self, k, val=0): return self._rw_bits(*QMD.fields[self.pref][k.upper()])
  def field_offset(self, k): return QMD.fields[self.pref][k.upper()][1] // 8
  def set_constant_buf_addr(self, i, addr):
    if self.ver < 4:
      self.write(**{f'constant_buffer_addr_upper_{i}': hi32(addr), f'constant_buffer_addr_lower_{i}': lo32(addr)})
    else:
      self.write(**{f'constant_buffer_addr_upper_shifted6_{i}': hi32(addr >> 6), f'constant_buffer_addr_lower_shifted6_{i}': lo32(addr >> 6)})


# -------- NVProgram ----------
class NVArgsState:
  def __init__(self, buf, prg, bufs, vals=()):
    self.buf, self.prg, self.bufs, self.vals = buf, prg, bufs, vals

class NVProgram:
  def __init__(self, dev, name, lib):
    self.dev, self.name, self.lib = dev, name, lib
    self.constbufs = {0: (0, 0x160)}
    image, sections, relocs = elf_loader(self.lib, force_section_align=128)
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000) + 0x1000)
    prog_addr = self.lib_gpu.va_addr
    self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0x160
    prog_sz = image.nbytes
    for sh in sections:
      if sh.name == f".nv.shared.{name}":
        self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
      if sh.name == f".text.{name}":
        prog_addr, prog_sz = self.lib_gpu.va_addr + sh.header.sh_addr, sh.header.sh_size
      elif sh.name.startswith(".nv.info"):
        for typ, param, data in self._parse_elf_info(sh):
          if sh.name == f".nv.info.{name}" and param == 0xa:
            cbuf0_size = struct.unpack_from("IH", data)[1]
          elif sh.name == ".nv.info" and param == 0x12:
            self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240
          elif sh.name == ".nv.info" and param == 0x2f:
            self.regs_usage = struct.unpack_from("II", data)[1]
    for apply_image_offset, rel_sym_offset, typ, _ in relocs:
      if typ == 2:
        image[apply_image_offset:apply_image_offset + 8] = struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset)
      elif typ == 0x38:
        image[apply_image_offset + 4:apply_image_offset + 8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
      elif typ == 0x39:
        image[apply_image_offset + 4:apply_image_offset + 8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
      else:
        raise RuntimeError(f"unknown NV reloc {typ}")
    min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
    self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)
    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
    else:
      self.cbuf_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
    self.dev._ensure_has_local_memory(self.lcmem_usage)
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()
    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1
    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      qmd = {
        'qmd_major_version': 5,
        'qmd_type': nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA,
        'program_address_upper_shifted4': hi32(prog_addr >> 4),
        'program_address_lower_shifted4': lo32(prog_addr >> 4),
        'register_count': self.regs_usage,
        'shared_memory_size_shifted7': self.shmem_usage >> 7,
        'shader_local_memory_high_size_shifted4': self.dev.slm_per_thread >> 4,
      }
    else:
      qmd = {
        'qmd_major_version': 3,
        'sm_global_caching_enable': 1,
        'program_address_upper': hi32(prog_addr),
        'program_address_lower': lo32(prog_addr),
        'shared_memory_size': self.shmem_usage,
        'register_count_v': self.regs_usage,
        'shader_local_memory_high_size': self.dev.slm_per_thread,
      }
    self.qmd = QMD(
      dev, **qmd, qmd_group_id=0x3f,
      invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1,
      api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR,
      constant_buffer_invalidate_0=1,
      min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg,
      max_sm_config_shared_mem_size=0x1a,
      program_prefetch_size=min(prog_sz >> 8, 0x1ff),
      sass_version=dev.sass_version,
      program_prefetch_addr_upper_shifted=prog_addr >> 40,
      program_prefetch_addr_lower_shifted=prog_addr >> 8,
    )
    for i, (addr, sz) in self.constbufs.items():
      self.qmd.set_constant_buf_addr(i, addr)
      self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})
    self.kernargs_alloc_size = round_up(self.constbufs[0][1], 1 << 8) + (8 << 8)

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off + 4:start_off + sz + 4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4


# -------- NVAllocator ----------
class NVAllocator:
  def __init__(self, dev): self.dev = dev
  def alloc(self, size, host=False, uncached=False, contiguous=False, cpu_access=True):
    return self.dev.iface.alloc(size, host=host, uncached=uncached, contiguous=contiguous, cpu_access=cpu_access)
  def free(self, b): self.dev.iface.free(b)
  def _copyin(self, dest, src: memoryview):
    if dest.view is None: raise RuntimeError("buffer has no cpu mapping")
    dest.cpu_view().view(fmt='B')[:len(src)] = bytes(src)
  def _copyout(self, dest: memoryview, src):
    if src.view is None: raise RuntimeError("buffer has no cpu mapping")
    # Use int indexing to avoid memoryview struct mismatch when the dest is
    # memoryview(bytearray) and src.view is an MMIOInterface (non-buffer-protocol).
    n = len(dest)
    for i in range(n):
      dest[i] = src.cpu_view()[i]

# -------- Module-level NVDevice helpers --------
def wait_signal(signal, value, timeout_ms=30000):
  start = time.perf_counter()
  while signal.value < value:
    if (time.perf_counter() - start) * 1000 > timeout_ms:
      print(f"wait_signal timeout: signal={signal.value} want>={value}", flush=True)
      return False
    time.sleep(0.001)
  return True

# -------- PCIIfaceBase ----------
class PCIIfaceBase:
  def __init__(self, dev, vram_bar, va_start, va_size, dev_impl_t):
    self.dev, self.vram_bar, self.count = dev, vram_bar, 1
    self.dev_impl = dev_impl_t(self.pci_dev) if isinstance(getattr(self, 'pci_dev', None), object) else None
  @property
  def peer_group(self): return getattr(self.pci_dev, 'peer_group', type(self.pci_dev).__name__)
  def is_local(self): return not isinstance(self.pci_dev, RemotePCIDevice)
  def is_bar_small(self): return self.pci_dev.bar_info(self.vram_bar)[1] == (256 << 20)
  def alloc(self, size, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, **kwargs):
    should_use_sysmem = host or ((cpu_access if self.is_bar_small() else (uncached and cpu_access)) and not force_devmem)
    if should_use_sysmem:
      va = self.dev_impl.mm.alloc_vaddr(size := round_up(size, PAGESIZE))
      memview, paddrs = self.pci_dev.alloc_sysmem(size, vaddr=va, contiguous=contiguous)
      mapping = self.dev_impl.mm.map_range(va, size, [(p, 0x1000) for p in paddrs], aspace=AddrSpace.SYS, snooped=True, uncached=True)
      return HCQBuffer(va, size, meta=PCIAllocationMeta(mapping, True, paddrs[0]), view=memview, owner=self.dev)
    size = round_up(size, (2 << 20) if size >= (8 << 20) else (4 << 10))
    mapping = self.dev_impl.mm.valloc(size, uncached=uncached, contiguous=cpu_access)
    barview = self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], size=mapping.size) if cpu_access else None
    return HCQBuffer(mapping.va_addr, size, view=barview, meta=PCIAllocationMeta(mapping, cpu_access, mapping.paddrs[0][0]), owner=self.dev)
  def free(self, b):
    if b.owner != self.dev and self.is_local() and b.meta.has_cpu_mapping: FileIOInterface.munmap(b.va_addr, b.size)
    if b.owner == self.dev and b.meta.mapping.aspace is AddrSpace.PHYS: self.dev_impl.mm.vfree(b.meta.mapping)
  def map(self, b):
    # Mirrors tinygrad PCIIfaceBase.map: re-map an existing buffer (e.g. host/sysmem signal page)
    # into this device's vaspace page table. Required for the user channel's GPU to write to a
    # host-allocated signal buffer.
    paddrs, aspace = b.meta.mapping.paddrs, b.meta.mapping.aspace
    self.dev_impl.mm.map_range(int(b.va_addr), round_up(b.size, 0x1000), paddrs, aspace=aspace,
                                snooped=True, uncached=b.meta.mapping.uncached)
    return HCQBuffer(b.va_addr, b.size, meta=b.meta, owner=b.owner)
  def sleep(self, timeout): pass

# -------- PCIIface ----------
class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id):
    self.dev = dev
    self.pci_dev = APLRemotePCIDevice("NV", "usb4")
    PCIIfaceBase.__init__(self, dev, 1, NVMemoryManager.va_allocator.base, NVMemoryManager.va_allocator.size, NVDev)
    gsp = self.dev_impl.gsp
    self.gpfifo_class, self.compute_class, self.dma_class = gsp.gpfifo_class, gsp.compute_class, gsp.dma_class
    self.root, self.gpu_instance = 0xc1000000, 0


  def rm_alloc(self, parent, clss, params=None, root=None):
    return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)

  def rm_control(self, obj, cmd, params=None, **kwargs):
    return self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root, **kwargs)

  def setup_usermode(self):
    return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)

  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass
  def device_fini(self): self.dev_impl.fini()
  def sleep(self, timeout):
    for _ in self.dev_impl.gsp.stat_q.read_resp(): pass
    if self.dev_impl.is_err_state: raise RuntimeError("Device fault detected")


# -------- NVDevice ----------
class NVDevice:
  def __init__(self, device=""):
    self.device = device or "NV"
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = PCIIface(self, self.device_id)

    # tinygrad PCIIface line 561: create NV01_ROOT first
    self.iface.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # GSP init_hw ran in NVDev.__init__ — golden channel + subdev + dev allocated there.
    # The user-mode NVDevice creates its own NV01_DEVICE_0/SUBDEVICE/vaspace/channel_group.
    gsp = self.iface.dev_impl.gsp

    # NVDevice lines 593-595: device, subdevice, virtmem
    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root,
      vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST,
      nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff,
        flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) |
               (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) |
               (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))))
    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL,
      nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(limit=0x1ffffffffffff))
    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info(
      'num_gpcs', 'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')
    self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)
    self.vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A,
      nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
        flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED))

    # Channel group
    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    self.channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)

    # Allocate compute gpfifo area + ctxshare (mirrors tinygrad lines 611-617)
    self.gpfifo_area = self.iface.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True)
    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=self.vaspace,
      flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    self.ctxshare = self.iface.rm_alloc(self.channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A, ctxshare_params)

    # usermode + gpu_mmio must exist before any GPFIFO submit (tinygrad NVDevice lines 608-609)
    self.usermode, self.gpu_mmio = self.iface.setup_usermode()
    self.compute_gpfifo, self.compute_channel = self._new_gpu_fifo(self.gpfifo_area, self.ctxshare, self.channel_group,
      offset=0, entries=0x10000, compute=True)
    self.dma_gpfifo, self.dma_channel = self._new_gpu_fifo(self.gpfifo_area, self.ctxshare, self.channel_group,
      offset=0x100000, entries=0x10000, compute=False)
    self.iface.rm_control(self.channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
      nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
    print(f"user compute_gpfifo ring_size={len(self.compute_gpfifo.ring)} token=0x{self.compute_gpfifo.token:x}", flush=True)
    # Command-queue buffer for method pushes (sysmem-backed counter)
    self.cmdq_page = self.iface.alloc(0x200000, cpu_access=True)
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=int(self.cmdq_page.va_addr), wrap=True)
    self.cmdq = self.cmdq_page.cpu_view().view(fmt='I')
    # Timeline signal. tinygrad uses allocator.alloc(host=True, uncached=True, cpu_access=True)
    # and then allocator.map() to add a PTE; for our vidmem fallback, the original alloc
    # with cpu_access=True already maps a PTE in the user vaspace (proven: PTE is valid).
    self._signal_page = self.iface.alloc(0x1000, cpu_access=True, uncached=True)
    # Skip the destructive 0-init write: racing with the GPU's first release can lose the write.
    # The first next_timeline() release will be value 1, which is what we wait for.
    self._signal_page_view = self._signal_page.cpu_view().view(fmt='Q')
    self.timeline_signal = NVSignal(owner=self)
    self.timeline_signal.value_addr = self._signal_page.va_addr
    self.timeline_value = 1  # tinygrad convention: first next_timeline() returns 1, 2, 3...

    # Kernargs scratch buffer (sysmem so manual_launch can read/write it directly via mmap)
    self.kernargs_buf = self.iface.alloc(0x400000, cpu_access=True, uncached=True)
    self.kernargs_offset_allocator = BumpAllocator(size=self.kernargs_buf.size, wrap=True)

    # Setup shader shared/local memory windows (mirrors tinygrad NVDevice._setup_gpfifos).
    # Without these windows, the compute engine faults when the kernel accesses
    # the local/shared memory space.
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
    self.allocator = NVAllocator(self)
    self._setup_gpfifos()

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False):
    notifier = self.iface.alloc(48 << 20, uncached=True)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
      gpFifoOffset=gpfifo_area.va_addr + offset, gpFifoEntries=entries, hContextShare=ctxshare,
      hObjectError=notifier.meta.hMemory, hObjectBuffer=self.virtmem if video else gpfifo_area.meta.hMemory,
      hUserdMemory=(ctypes.c_uint32 * 8)(gpfifo_area.meta.hMemory),
      userdOffset=(ctypes.c_uint64 * 8)(entries * 8 + offset), engineType=19 if video else 0)
    gpfifo = self.iface.rm_alloc(channel_group, self.iface.gpfifo_class, params)
    if compute:
      self.debug_compute_obj = self.iface.rm_alloc(gpfifo, self.iface.compute_class)
      self.debug_channel = gpfifo
    elif not video:
      self.iface.rm_alloc(gpfifo, self.iface.dma_class)
    if channel_group == self.nvdevice:
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_BIND, nv_gpu.NVA06F_CTRL_BIND_PARAMS(engineType=params.engineType))
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
    ws_token_params = self.iface.rm_control(gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1))
    if self.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      ws_token_params.workSubmitToken |= (1 << 30)
    gpput_off = offset + entries * 8 + getattr(nv_gpu.AmpereAControlGPFifo, 'GPPut').offset
    fifo = GPFifo(ring=gpfifo_area.cpu_view().view(offset, entries * 8, fmt='Q'),
                  gpput=gpfifo_area.cpu_view().view(gpput_off, 4, fmt='I'),
                  entries_count=entries, token=ws_token_params.workSubmitToken)
    return fifo, gpfifo

  def _query_gpu_info(self, *reqs):
    nvrs = [getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_' + r.upper(),
                    getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_LITTER_' + r.upper(), None)) for r in reqs]
    x = self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO,
                                nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS())
    return [x.engineInfo[0].infoList[nvr].data for nvr in nvrs]

  def _setup_gpfifos(self):
    """Submit SET_OBJECT + shader memory windows + timeline signals (mirrors tinygrad NVDevice._setup_gpfifos)."""
    self.slm_per_thread, self.shader_local_mem = 0, None
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
    NVComputeQueue().setup(compute_class=self.iface.compute_class,
                           local_mem_window=self.local_mem_window,
                           shared_mem_window=self.shared_mem_window) \
                    .signal(self.timeline_signal, self.next_timeline()).submit(self)
    NVCopyQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                 .setup(copy_class=self.iface.dma_class) \
                 .signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required:
      return
    self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    total = round_up(bytes_per_tpc * self.num_tpc_per_gpc * self.num_gpcs, 0x20000)
    if self.shader_local_mem is None or self.shader_local_mem.size < total:
      self.shader_local_mem = self.allocator.alloc(total, cpu_access=False)
    if self.shader_local_mem.size < total:
      print(f"WARN: shader_local_mem alloc too small got=0x{self.shader_local_mem.size:x} need=0x{total:x}", flush=True)
      self.slm_per_thread = old_slm_per_thread
      return
    NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                     .setup(local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc) \
                     .signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()
  def is_nvd(self): return isinstance(self.iface, PCIIface)
  def runtime(self, name, lib):
    return NVProgram(self, name, lib)

  def next_timeline(self):
    self.timeline_value += 1
    return self.timeline_value - 1

  def synchronize(self):
    # next_timeline() returns the value just stored in timeline_value (post-increment).
    # The GPU writes that value into the semaphore, so the latest release is
    # timeline_value - 1. Wait for the previous-release to land.
    target = self.timeline_value - 1
    start = time.perf_counter()
    while self.timeline_signal.value < target:
      elapsed_ms = (time.perf_counter() - start) * 1000
      if elapsed_ms > 5000:
        # Diagnostic: ring/gpput/signal snapshot when we give up
        gpput = self.compute_gpfifo.gpput[0]
        try:
          last_entry = int(self.compute_gpfifo.ring[(self.compute_gpfifo.put_value - 1) % self.compute_gpfifo.entries_count])
          last_va = ((last_entry & ((1 << 40) - 1)) >> 2) << 2
          last_pkts = (last_entry >> 42) & ((1 << 20) - 1)
        except Exception:
          last_va, last_pkts = -1, -1
        print(f"WARN: synchronize timeout target={target} got={self.timeline_signal.value} "
              f"gpput={gpput} put_value={self.compute_gpfifo.put_value} "
              f"last_ring_va=0x{last_va:x} last_pkts={last_pkts} sig_va=0x{self.timeline_signal.value_addr:x}",
              flush=True)
        return False
      if self.iface.is_local():
        for _ in self.iface.dev_impl.gsp.stat_q.read_resp(): pass
      else:
        self.iface.sleep(10)
      time.sleep(0.001)
    return True


# ============================================================================
# Demo driver: standalone NV add kernel (live path)
# Run with: python3 examples/middle_nv.py
# Optional:  python3 examples/middle_nv.py --middle-selftest (offline gate)
# ============================================================================

METHOD_NAMES = {
  0x005c: "NVC56F_SEM_ADDR_LO",
  0x02b4: "NVC6C0_SEND_PCAS_A",
  0x02c0: "NVC6C0_SEND_SIGNALING_PCAS2_B",
  0x1698: "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
  0x0020: "NVC56F_NON_STALL_INTERRUPT",
}

def nvm(subchannel, method, *args, typ=2):
  return [(typ << 28) | (len(args) << 16) | (subchannel << 13) | (method >> 2), *args]

def build_launch_words(timeline_addr, wait_value, done_value, qmd_addr):
  lo, hi = timeline_addr & 0xffffffff, timeline_addr >> 32
  return [
    *nvm(0, 0x005c, lo, hi, wait_value, 0, 0x01000003),
    *nvm(1, 0x1698, 0x00001011),
    *nvm(1, 0x02b4, qmd_addr >> 8),
    *nvm(1, 0x02c0, 0x00000009),
    *nvm(0, 0x005c, lo, hi, done_value, 0, 0x03100001),
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
  if method == 0x005c and len(args) == 5:
    sem_addr = (args[1] << 32) | args[0]
    payload = (args[3] << 32) | args[2]
    return [f"sem_addr=0x{sem_addr:x}", f"payload={payload}", f"execute=0x{args[4]:08x}"]
  if method == 0x1698 and len(args) == 1:
    return [f"invalidate_flags=0x{args[0]:08x}"]
  if method == 0x02b4 and len(args) == 1:
    return [f"qmd_addr=0x{args[0] << 8:x}", f"qmd_addr_shifted8=0x{args[0]:x}"]
  if method == 0x02c0 and len(args) == 1:
    return [f"pcas2_action=0x{args[0]:x}"]
  return [f"arg{i}=0x{arg:08x}" for i, arg in enumerate(args)]

class CubinHelper:
  class Reg:
    RZ = 255
    R0 = 0; R1 = 1; R2 = 2; R3 = 3; R4 = 4; R5 = 5; R6 = 6; R7 = 7
    R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15

  class UReg:
    URZ = 63
    UR4 = 4  # only UR4 is used in our cubin

  class Op:
    LDC     = 0x7a02
    LDCU64  = 0x7ab9
    FADD    = 0x7221
    FMUL    = 0x7220
    LDG     = 0x7981
    STG     = 0x7986
    EXIT    = 0x794d
    BRA     = 0x7947
    NOP     = 0x7918

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
  EF_CUDA_SM86 = 0x560556
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
    return struct.pack("<16sHHIQQQIHHHHHH", ident, CubinHelper.ET_EXEC, CubinHelper.EM_CUDA, CubinHelper.ELF_VERSION, 0, phoff, shoff, CubinHelper.EF_CUDA_SM86, 64, 56, phnum, 64, shnum, shstrndx)

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
  bundles = [
    # SASS_COMMON_PREFIX
    ((ch.Reg.R1 << 16) | ch.Op.LDC,    0x00000a00, 0x00000f00, 0x000fe400),  # MOV R1, c[0x0][0x28]
    ((ch.Reg.R4 << 16) | ch.Op.LDC,    0x00005c00, 0x00000f00, 0x000fe200),  # MOV R4, c[0x0][0x170]
    ((ch.UReg.UR4 << 16) | ch.Op.LDCU64, 0x00004600, 0x00000a00, 0x000fe200),  # ULDC.64 UR4, c[0x0][0x118]
    ((ch.Reg.R5 << 16) | ch.Op.LDC,    0x00005d00, 0x00000f00, 0x000fe400),  # MOV R5, c[0x0][0x174]
    ((ch.Reg.R2 << 16) | ch.Op.LDC,    0x00005a00, 0x00000f00, 0x000fe400),  # MOV R2, c[0x0][0x168]
    ((ch.Reg.R3 << 16) | ch.Op.LDC,    0x00005b00, 0x00000f00, 0x000fe400),  # MOV R3, c[0x0][0x16c]
    ((ch.Reg.R4 << 24) | (ch.Reg.R4 << 16) | ch.Op.LDG, 0x00000004, 0x0c1e1d00, 0x000ea800),  # LDG.E.128 R4, [R4.64]
    ((ch.Reg.R2 << 24) | (ch.Reg.R8 << 16) | ch.Op.LDG, 0x00000004, 0x0c1e1d00, 0x000ea400),  # LDG.E.128 R8, [R2.64]

    # SASS_ARITHMETIC
    ((ch.Reg.R11 << 24) | (ch.Reg.R11 << 16) | ch.Op.FADD, 0x00000007, 0x00000000, 0x004fe200),  # FADD R11, R11, R7
    ((ch.Reg.R10 << 24) | (ch.Reg.R10 << 16) | ch.Op.FADD, 0x00000006, 0x00000000, 0x000fe200),  # FADD R10, R10, R6
    ((ch.Reg.R9  << 24) | (ch.Reg.R9  << 16) | ch.Op.FADD, 0x00000005, 0x00000000, 0x000fe200),  # FADD R9, R9, R5
    ((ch.Reg.R8  << 24) | (ch.Reg.R8 << 16) | ch.Op.FADD, 0x00000004, 0x00000000, 0x000fe200),    # FADD R8, R8, R4

    # SASS_COMMON_SUFFIX
    ((ch.Reg.R6 << 16) | ch.Op.LDC,    0x00005800, 0x00000f00, 0x000fc400),  # MOV R6, c[0x0][0x160]
    ((ch.Reg.R7 << 16) | ch.Op.LDC,    0x00005900, 0x00000f00, 0x000fca00),  # MOV R7, c[0x0][0x164]
    ((ch.Reg.R6 << 24) | ch.Op.STG,    0x00000008, 0x0c101d04, 0x000fe200),  # STG.E.128 [R6.64], R8
    (ch.Op.EXIT,                    0x00000000, 0x03800000, 0x000fea00),  # EXIT
    (ch.Op.BRA,                     0xfffffff0, 0x0383ffff, 0x000fc000),  # BRA .
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

def manual_launch(dev, program, out, a, b):
  kernargs = dev.kernargs_buf.offset(dev.kernargs_offset_allocator.alloc(program.kernargs_alloc_size, 8), program.kernargs_alloc_size)
  cbuf_words = program.cbuf_0 or []
  kernargs.cpu_view().view(size=len(cbuf_words) * 4, fmt='I')[:] = array.array('I', cbuf_words)
  kernargs.cpu_view().view(offset=len(cbuf_words) * 4, size=3 * 8, fmt='Q')[:] = array.array('Q', [out.va_addr, a.va_addr, b.va_addr])
  qmd_buf = kernargs.offset(round_up(program.constbufs[0][1], 1 << 8))
  qmd_buf.cpu_view().view(size=program.qmd.mv.nbytes, fmt='B')[:] = program.qmd.mv
  qmd = type(program.qmd)(dev=dev, view=qmd_buf.cpu_view())
  qmd.write(cta_raster_width=1, cta_raster_height=1, cta_raster_depth=1,
            cta_thread_dimension0=1, cta_thread_dimension1=1, cta_thread_dimension2=1)
  qmd.set_constant_buf_addr(0, kernargs.va_addr)
  wait_value = dev.timeline_value - 1
  done_value = dev.next_timeline()
  signal_addr = dev.timeline_signal.value_addr
  qmd.write(release0_enable=1, release0_address_lower=signal_addr & 0xffffffff, release0_address_upper=(signal_addr >> 32) & 0xff,
            release0_payload_lower=done_value & 0xffffffff, release0_payload_upper=done_value >> 32)
  words = build_launch_words(signal_addr, wait_value, done_value, qmd_buf.va_addr)[:12]
  print(f"submit #manual: NVComputeQueue words={len(words)}")
  for index, typ, subc, method, name, args in decode_words(words):
    print(f"  method[{index}] {name}: typ={typ} subc={subc} mthd=0x{method:x} args=[{', '.join(describe_args(method, args))}]")
  submit_gpfifo(dev, words)
  wait_signal(dev.timeline_signal, done_value)

MIDDLE_CUBIN_SHA256 = "54f9606fe6b03d6cc98186358c68a74cebe8275137c1e98723967f9a14c67324"
MIDDLE_CUBIN_BYTES = 2856
MIDDLE_LAUNCH_WORDS = 20

def middle_selftest():
  """Tier 1 offline gate: cubin + launch words + helper sanity."""
  cubin = build_cubin()
  assert len(cubin) == MIDDLE_CUBIN_BYTES, f"cubin size {len(cubin)} != {MIDDLE_CUBIN_BYTES}"
  sha = hashlib.sha256(cubin).hexdigest()
  assert sha == MIDDLE_CUBIN_SHA256, f"cubin sha {sha} != {MIDDLE_CUBIN_SHA256}"
  words = build_launch_words(0xdeadbeef00001000, 3, 7, 0x2000)
  assert len(words) == MIDDLE_LAUNCH_WORDS, f"launch words {len(words)} != {MIDDLE_LAUNCH_WORDS}"
  decoded = list(decode_words(words))
  assert len(decoded) == 6, f"decode_words count {len(decoded)} != 6"
  sem_methods = [m for _, _, _, m, _, _ in decoded if m == 0x005c]
  assert len(sem_methods) == 2, "expected two semaphore methods"
  # helpers sanity
  assert lo32(0x123456789abcdef0) == 0x9abcdef0
  assert hi32(0x123456789abcdef0) == 0x12345678
  assert round_up(17, 16) == 32
  assert ceildiv(17, 16) == 2
  assert wait_cond(lambda: 1, value=1, timeout_ms=100)
  # mmio roundtrip (use array directly; MMIOInterface is the autogen, not the test slim wrapper)
  arr = array.array('I', [0, 1, 2, 3])
  arr[1] = 0x42
  assert arr[1] == 0x42
  assert arr[2] == 2
  # grbuf
  desc = GRBufDesc(size=4096, virt=True, phys=False)
  assert desc.size == 4096 and desc.virt and not desc.phys
  # nvrpcqueue checksum
  data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
  if (pad_len := (-len(data)) % 8): data += b"\x00" * pad_len
  cs = 0
  for off in range(0, len(data), 8): cs ^= struct.unpack_from("Q", data, off)[0]
  cs = hi32(cs) ^ lo32(cs)
  assert isinstance(cs, int) and 0 <= cs <= 0xffffffff
  print(f"middle_selftest=ok cubin_sha={sha} launch_words={len(words)} rpc_checksum=0x{cs:x}")

def main():
  if "--middle-selftest" in sys.argv:
    middle_selftest()
    return
  # live path: do the actual add kernel
  t0 = time.perf_counter()
  def _ts(label):
    if os.environ.get("NV_ADD_TRACE_STAGES") == "1":
      print(f"  stage t={time.perf_counter()-t0:6.3f}s  {label}", flush=True)
  a = (1.0, 2.0, 3.0, 4.0)
  b = (10.0, 20.0, 30.0, 40.0)
  cubin = build_cubin()
  _ts("cubin built")
  print(f"cubin_bytes={len(cubin)} expected_result={[x + y for x, y in zip(a, b)]}")
  dev = NVDevice("NV")
  _ts("device ready (boot+GSP+golden+user-channel+gpfifo)")
  print(f"device={dev.device} iface={type(dev.iface).__name__}", flush=True)
  a_buf = dev.allocator.alloc(16)
  b_buf = dev.allocator.alloc(16)
  out_buf = dev.allocator.alloc(16)
  _ts("3 input buffers allocated (sysmem via BAR1 mmap)")
  dev.allocator._copyin(a_buf, memoryview(struct.pack("4f", *a)))
  dev.allocator._copyin(b_buf, memoryview(struct.pack("4f", *b)))
  dev.allocator._copyin(out_buf, memoryview(bytes(16)))
  _ts("3 copyin done (H2D)")
  program = dev.runtime("E_4", cubin)
  _ts("program built (cubin uploaded to VRAM, NVProgram ready)")
  manual_launch(dev, program, out_buf, a_buf, b_buf)
  _ts("manual_launch done (kernel executed on eGPU, result on device)")
  result_bytes = bytearray(16)
  dev.allocator._copyout(memoryview(result_bytes), out_buf)
  _ts("copyout done (D2H)")
  result = list(struct.unpack("4f", result_bytes))
  _ts(f"final result decoded: {result}")
  print(f"result={result}")

if __name__ == "__main__":
  main()
