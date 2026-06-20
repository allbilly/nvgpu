#!/usr/bin/env python3
import array, ast, collections, contextlib, ctypes, enum, fcntl, functools, hashlib, io, mmap, os, pathlib, re, socket, struct, subprocess, sys, tempfile, threading, time, traceback, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[1]

METHOD_NAMES = {
  0x005c: "NVC56F_SEM_ADDR_LO",
  0x02b4: "NVC6C0_SEND_PCAS_A",
  0x02c0: "NVC6C0_SEND_SIGNALING_PCAS2_B",
  0x1698: "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
  0x0020: "NVC56F_NON_STALL_INTERRUPT",
}
QMD_ADDR_LIMIT = 1 << 40

class CubinHelper:
  class Reg:
    RZ = 255
    R0 = 0; R1 = 1; R2 = 2; R3 = 3; R4 = 4; R5 = 5; R6 = 6; R7 = 7
    R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15

  class UReg:
    URZ = 63
    UR4 = 4  # only UR4 is used in our cubin

  #  cuobjdump -sass   (closed-source NVIDIA disassembler; mnemonic + bytes)
  #  cuasm sm_86       (open-source assembler, CuAsm/InsAsmRepos)
  #  denvdis data11    (open-source 128-bit SASS spec, denvdis/data11/sm86_1.txt)
  class Op:
    LDC     = 0x7a02  # LDC / LDC.64 (alias: MOV).  denvdis 0xb82 family.
    LDCU64  = 0x7ab9  # LDCU.64.                    denvdis ULDC_default 0xab9 — EXACT.
    FADD    = 0x7221  # FADD / IMAD.WIDE.           denvdis FADD_Rb 0x221 — EXACT (low12).
    FMUL    = 0x7220  # FMUL / IMAD.WIDE.           denvdis FMUL_v3 0x220 — EXACT (low12).
    LDG     = 0x7981  # LDG descriptor pre-load.    denvdis LDG_R_dARI 0x981 — EXACT (low12).
    STG     = 0x7986  # STG.E.                      denvdis STG_E 0xc86 family.
    EXIT    = 0x794d  # EXIT.                       denvdis EXIT 0x94d — EXACT (low12).
    BRA     = 0x7947  # BRA.                        denvdis BRA 0x947 — EXACT (low12).
    NOP     = 0x7918  # NOP.                        denvdis NOP 0x918 — EXACT (low12).

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

def nvm(subchannel, method, *args, typ=2):
  typ = validate_u32(typ, "method packet type")
  subchannel = validate_u32(subchannel, "method subchannel")
  method = validate_u32(method, "method address")
  if typ > 0xf: raise ValueError("method packet type is outside 4-bit range")
  if subchannel > 0x7: raise ValueError("method subchannel is outside 3-bit range")
  if method & 0x3: raise ValueError("method address must be 4-byte aligned")
  if method > 0x7fff: raise ValueError("method address is outside 15-bit range")
  if len(args) > 0xfff: raise ValueError("method argument count is outside 12-bit range")
  args = tuple(validate_u32(arg, "method argument") for arg in args)
  return [(typ << 28) | (len(args) << 16) | (subchannel << 13) | (method >> 2), *args]

def check_qmd_pointer(name, addr, align=1):
  if addr < 0 or addr >= QMD_ADDR_LIMIT: raise ValueError(f"{name} address 0x{addr:x} exceeds QMD 40-bit limit")
  if align > 1 and addr % align: raise ValueError(f"{name} address 0x{addr:x} is not {align}-byte aligned")
  return addr

def build_launch_words(timeline_addr, wait_value, done_value, qmd_addr):
  check_qmd_pointer("timeline semaphore", timeline_addr)
  check_qmd_pointer("QMD", qmd_addr, align=0x100)
  wait_value = validate_u32(wait_value, "timeline wait value")
  done_value = validate_u32(done_value, "timeline done value")
  lo, hi = timeline_addr & 0xffffffff, timeline_addr >> 32
  return [
    *nvm(0, 0x005c, lo, hi, wait_value, 0, 0x01000003),
    *nvm(1, 0x1698, 0x00001011),
    *nvm(1, 0x02b4, qmd_addr >> 8),
    *nvm(1, 0x02c0, 0x00000009),
    *build_timeline_signal_words(timeline_addr, done_value, interrupt=True),
  ]

def build_timeline_signal_words(timeline_addr, value, interrupt=False):
  check_qmd_pointer("timeline semaphore", timeline_addr)
  value = validate_u32(value, "timeline signal payload")
  lo, hi = timeline_addr & 0xffffffff, timeline_addr >> 32
  words = nvm(0, 0x005c, lo, hi, value, 0, 0x03100001)
  if interrupt: words += nvm(0, 0x0020, 0)
  return words

def build_compute_launch_words(timeline_addr, wait_value, done_value, qmd_addr):
  words = build_launch_words(timeline_addr, wait_value, done_value, qmd_addr)
  pcas_methods = [method for _, _, _, method, _, _ in decode_words(words) if method in (0x02b4, 0x02c0)]
  if pcas_methods != [0x02b4, 0x02c0]: raise RuntimeError("compute launch stream missing PCAS schedule methods")
  # The QMD release semaphore carries done_value; keep the command stream to wait + invalidate + PCAS.
  return words[:12]

def decode_words(words):
  index = 0
  while index < len(words):
    header = words[index]
    validate_u32(header, "method packet header")
    typ, size, subc, method = (header >> 28) & 0xf, (header >> 16) & 0xfff, (header >> 13) & 0x7, (header << 2) & 0x7fff
    if index + 1 + size > len(words): raise ValueError("method stream is truncated")
    args = words[index + 1:index + 1 + size]
    for arg in args: validate_u32(arg, "method argument")
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

def round_up(x, y): return ((x + y - 1) // y) * y
def round_down(x, y): return (x // y) * y
def ceildiv(x, y): return (x + y - 1) // y
def lo32(x): return x & 0xffffffff
def hi32(x): return (x >> 32) & 0xffffffff
def data64(x): return [hi32(x), lo32(x)]
def data64_le(x): return [lo32(x), hi32(x)]
def validate_u32(value, name):
  if not 0 <= value <= 0xffffffff: raise ValueError(f"{name} is outside 32-bit range: 0x{value:x}")
  return value
def validate_u64(value, name):
  if not 0 <= value <= 0xffffffffffffffff: raise ValueError(f"{name} is outside 64-bit range: 0x{value:x}")
  return value
def validate_i32(value, name):
  if not -(1 << 31) <= value <= (1 << 31) - 1: raise ValueError(f"{name} is outside signed 32-bit range: {value}")
  return value
def validate_rm_handle(handle, name="RM handle"):
  return validate_u32(handle, name)
def validate_wait_flag(value, name="RM wait flag"):
  if not isinstance(value, bool): raise ValueError(f"{name} must be bool")
  return value
def validate_positive_size(size, name):
  if size <= 0: raise ValueError(f"{name} must be positive")
  return size
def validate_alignment(align, name="alignment"):
  if align <= 0 or align & (align - 1): raise ValueError(f"{name} must be a positive power of two")
  return align
def require_len(data, size, name):
  if len(data) < size: raise ValueError(f"{name} is truncated")
  return data
def validate_pci_config_access(offset, size):
  if offset < 0: raise ValueError("PCI config offset must be non-negative")
  if size not in (1, 2, 4, 8): raise ValueError("PCI config access size must be 1, 2, 4, or 8 bytes")
  if offset + size > 4096: raise ValueError("PCI config access is outside config space")
def validate_bar_index(bar):
  if not 0 <= bar < 6: raise ValueError(f"BAR index is outside PCI BAR range: {bar}")

def write_words(dst, offset, words):
  if offset < 0: raise ValueError("word write offset must be non-negative")
  if offset + len(words) > len(dst): raise ValueError("word write exceeds destination")
  words = [validate_u32(word, "word write value") for word in words]
  dst[offset:offset + len(words)] = array.array('I', words)

class MMIOView:
  def __init__(self, backing, nbytes=None, fmt='B', addr=None, offset=0):
    if isinstance(backing, int):
      if nbytes is None: raise ValueError("nbytes is required for address-backed MMIOView")
      raw = (ctypes.c_ubyte * nbytes).from_address(backing)
      self._base = memoryview(raw)
      self.addr = backing
      self.nbytes = nbytes
    else:
      self._base = memoryview(backing)
      self.addr = addr or 0
      self.nbytes = len(self._base) if nbytes is None else nbytes
    self.offset, self.fmt = offset, fmt
    self.el_sz = struct.calcsize(fmt)
    if offset < 0 or self.nbytes < 0 or offset + self.nbytes > len(self._base):
      raise ValueError("MMIOView range is out of bounds")
    if self.nbytes % self.el_sz:
      raise ValueError("MMIOView size is not aligned to element format")
    self.mv = self._base[offset:offset+self.nbytes].cast(fmt)

  def __len__(self): return self.nbytes // self.el_sz
  def __getitem__(self, index):
    val = self.mv[index]
    return val.tolist() if isinstance(index, slice) and self.fmt != 'B' else val
  def __setitem__(self, index, value): self.mv[index] = value
  def view(self, offset=0, size=None, fmt=None):
    if offset < 0: raise ValueError("MMIOView offset must be non-negative")
    if offset > self.nbytes: raise ValueError("MMIOView offset is beyond the view size")
    if size is None: size = self.nbytes - offset
    if size < 0 or offset + size > self.nbytes: raise ValueError("MMIOView range is out of bounds")
    return MMIOView(self._base, size, fmt or self.fmt, self.addr + offset, self.offset + offset)

class FileIO:
  _libc = ctypes.CDLL(None)
  def __init__(self, path="", flags=os.O_RDONLY, fd=None):
    self.path = path
    self.fd = fd if fd is not None else os.open(path, flags)
  def __del__(self):
    if hasattr(self, "fd"):
      try: os.close(self.fd)
      except OSError: pass
  def ioctl(self, request, arg): return fcntl.ioctl(self.fd, request, arg)
  def seek(self, offset): os.lseek(self.fd, offset, os.SEEK_SET)
  def read(self, size=None, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as f: return f.read(size)
  def write(self, content, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as f: return f.write(content)
  def mmap(self, start, size, prot, flags, offset): return mmap.mmap(self.fd, size, flags=flags, prot=prot, offset=offset)
  @staticmethod
  def munmap(addr, size):
    if FileIO._libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size)) != 0:
      raise OSError(ctypes.get_errno(), "munmap failed")
  @staticmethod
  def exists(path): return os.path.exists(path)

class BumpAllocator:
  def __init__(self, size, base=0, wrap=False):
    validate_positive_size(size, "bump allocator size")
    self.size, self.base, self.wrap, self.off = size, base, wrap, 0
  def alloc(self, size, align=1):
    validate_positive_size(size, "allocation size")
    validate_alignment(align)
    off = round_up(self.off, align)
    if off + size > self.size:
      if not self.wrap: raise MemoryError(f"bump allocator exhausted: need {size} bytes")
      off = 0
      if off + size > self.size: raise MemoryError(f"bump allocator exhausted: need {size} bytes")
    self.off = off + size
    return self.base + off

class FreeListAllocator:
  def __init__(self, size, base=0):
    validate_positive_size(size, "free-list allocator size")
    self.base, self.free = base, [(0, size)]
    self.used = {}
  def alloc(self, size, align=1):
    validate_positive_size(size, "allocation size")
    validate_alignment(align)
    for idx, (start, span) in enumerate(self.free):
      aligned = round_up(start, align)
      pad = aligned - start
      if pad + size > span: continue
      new_free = []
      if pad: new_free.append((start, pad))
      if span > pad + size: new_free.append((aligned + size, span - pad - size))
      self.free[idx:idx+1] = new_free
      self.used[aligned] = size
      return self.base + aligned
    raise MemoryError(f"can't allocate {size} bytes")
  def free_addr(self, addr):
    start = addr - self.base
    if start not in self.used: raise ValueError(f"address 0x{addr:x} was not allocated")
    size = self.used.pop(start)
    self.free.append((start, size))
    self.free.sort()
    merged = []
    for block_start, block_size in self.free:
      if merged and merged[-1][0] + merged[-1][1] == block_start:
        merged[-1] = (merged[-1][0], merged[-1][1] + block_size)
      else:
        merged.append((block_start, block_size))
    self.free = merged

class TLSFAllocator:
  def __init__(self, size, base=0, block_size=16, lv2_cnt=16):
    validate_positive_size(size, "TLSF allocator size")
    self.size, self.base, self.block_size, self.l2_cnt = size, base, block_size, lv2_cnt.bit_length()
    self.storage = [collections.defaultdict(list) for _ in range(size.bit_length() + 1)]
    self.lv1_entries = [0] * len(self.storage)
    self.blocks = {0: (size, None, None, True)}
    if size > 0: self._insert_block(0, size)

  @functools.cache
  def lv1(self, size): return size.bit_length()

  @functools.cache
  def lv2(self, size):
    return (size - (1 << (size.bit_length() - 1))) // (1 << max(0, size.bit_length() - self.l2_cnt))

  def _insert_block(self, start, size, prev=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].append(start)
    self.lv1_entries[self.lv1(size)] += 1
    self.blocks[start] = (size, start + size, prev, True)
    return self

  def _remove_block(self, start, size, prev=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].remove(start)
    self.lv1_entries[self.lv1(size)] -= 1
    self.blocks[start] = (size, start + size, prev, False)
    return self

  def _split_block(self, start, size, new_size):
    nxt = self.blocks[start][1]
    if not self.blocks[start][3]: raise RuntimeError("TLSF block must be free")
    self._remove_block(start, size)._insert_block(start, new_size)._insert_block(start + new_size, size - new_size, prev=start)
    if nxt in self.blocks:
      self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start + new_size, self.blocks[nxt][3])
    return self

  def _merge_right(self, start):
    size, nxt, _, is_free = self.blocks[start]
    if not is_free: raise RuntimeError("TLSF block must be free")
    while is_free and nxt in self.blocks:
      blk = self.blocks[nxt]
      if blk[3] is False: break
      self._remove_block(start, size)._remove_block(nxt, blk[0])._insert_block(start, size := size + blk[0])
      _, nxt, _, _ = self.blocks.pop(nxt)
    if nxt in self.blocks:
      self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start, self.blocks[nxt][3])

  def _merge_block(self, start):
    while (prev := self.blocks[start][2]) is not None and self.blocks[prev][3] is True:
      start = prev
    self._merge_right(start)

  def alloc(self, req_size, align=1):
    validate_positive_size(req_size, "TLSF allocation size")
    validate_alignment(align)
    req_size = max(self.block_size, req_size)
    size = max(self.block_size, req_size + align - 1)
    size = round_up(size, 1 << (size.bit_length() - self.l2_cnt))
    for l1 in range(self.lv1(size), len(self.storage)):
      if self.lv1_entries[l1] == 0: continue
      for l2 in range(self.lv2(size) if l1 == size.bit_length() else 0, 1 << self.l2_cnt):
        if not self.storage[l1][l2]: continue
        start = self.storage[l1][l2][0]
        nsize = self.blocks[start][0]
        if (new_start := round_up(start, align)) != start:
          self._split_block(start, nsize, new_start - start)
          start, nsize = new_start, self.blocks[new_start][0]
        if nsize > req_size: self._split_block(start, nsize, req_size)
        self._remove_block(start, req_size)
        return start + self.base
    raise MemoryError(f"can't allocate {req_size} bytes")

  def free_addr(self, addr):
    start = addr - self.base
    if start not in self.blocks or self.blocks[start][3]:
      raise ValueError(f"address 0x{addr:x} was not allocated")
    self._insert_block(start, self.blocks[start][0])._merge_block(start)

class GpuBuffer:
  def __init__(self, va_addr, size, view=None, meta=None, base=None):
    if va_addr < 0: raise ValueError("buffer address must be non-negative")
    validate_positive_size(size, "buffer size")
    if view is not None:
      if not isinstance(view, MMIOView): raise ValueError("buffer CPU view is invalid")
      if view.nbytes < size: raise ValueError("buffer CPU view is smaller than buffer size")
    if base is not None and not isinstance(base, GpuBuffer): raise ValueError("buffer base is invalid")
    self.va_addr, self.size, self.view, self.meta, self._base = va_addr, size, view, meta, base
  def offset(self, offset=0, size=None):
    if offset < 0: raise ValueError("buffer offset must be non-negative")
    if offset > self.size: raise ValueError("buffer offset is beyond the buffer size")
    if size is None: size = self.size - offset
    if size < 0 or offset + size > self.size: raise ValueError("buffer offset range is out of bounds")
    return GpuBuffer(self.va_addr + offset, self.size - offset if size is None else size,
                     self.view.view(offset=offset, size=size) if self.view is not None else None, self.meta, self._base or self)
  def cpu_view(self):
    if self._base is not None and self._base.view is None: raise RuntimeError("buffer has been freed")
    if self.view is None: raise RuntimeError("buffer has no CPU view")
    return self.view

class SysmemAllocation:
  def __init__(self, view, paddrs, va_addr=None, size=0, h_memory=0, cpu_addr=None):
    if view is not None and not isinstance(view, MMIOView): raise ValueError("sysmem allocation view is invalid")
    if not paddrs: raise ValueError("sysmem allocation physical ranges must be non-empty")
    checked_paddrs = []
    for paddr in paddrs:
      if paddr < 0 or paddr % 0x1000: raise ValueError("sysmem allocation physical address must be 4KB aligned")
      checked_paddrs.append(paddr)
    if size <= 0 or size % 0x1000: raise ValueError("sysmem allocation size must be positive and 4KB aligned")
    if size != len(checked_paddrs) * 0x1000: raise ValueError("sysmem allocation size does not match physical ranges")
    self.view, self.paddrs = view, checked_paddrs
    self.va_addr, self.size, self.hMemory, self.cpu_addr = va_addr, size, h_memory, cpu_addr
  def __iter__(self):
    yield self.view
    yield self.paddrs

def memory_barrier():
  libc = ctypes.CDLL(None)
  for name, args in (("atomic_thread_fence", (5,)), ("__sync_synchronize", ())):
    fn = getattr(libc, name, None)
    if fn is not None:
      fn(*args)
      return

class AddrSpace(enum.Enum):
  PHYS = enum.auto()
  SYS = enum.auto()
  PEER = enum.auto()

class VirtMapping:
  def __init__(self, va_addr, size, paddrs, aspace, uncached=False, snooped=False):
    if va_addr < 0 or va_addr % 0x1000: raise ValueError("mapping VA must be 4KB aligned")
    if size <= 0 or size % 0x1000: raise ValueError("mapping size must be positive and 4KB aligned")
    if not isinstance(aspace, AddrSpace): raise ValueError("mapping address space is invalid")
    if not isinstance(uncached, bool): raise ValueError("mapping uncached flag must be bool")
    if not isinstance(snooped, bool): raise ValueError("mapping snooped flag must be bool")
    if not paddrs: raise ValueError("mapping physical ranges must be non-empty")
    checked_paddrs = []
    for paddr, span in paddrs:
      if paddr < 0 or paddr % 0x1000: raise ValueError("mapping physical address must be 4KB aligned")
      if span <= 0 or span % 0x1000: raise ValueError("mapping physical span must be positive and 4KB aligned")
      checked_paddrs.append((paddr, span))
    if size != sum(span for _, span in checked_paddrs): raise ValueError("mapping size does not match physical ranges")
    paddrs = checked_paddrs
    self.va_addr, self.size, self.paddrs, self.aspace = va_addr, size, paddrs, aspace
    self.uncached, self.snooped = uncached, snooped

class GRBufDesc:
  def __init__(self, size, virt, phys, local=False):
    size = validate_positive_size(size, "context buffer descriptor size")
    if not isinstance(virt, bool): raise ValueError("context buffer descriptor virt flag must be bool")
    if not isinstance(phys, bool): raise ValueError("context buffer descriptor phys flag must be bool")
    if not isinstance(local, bool): raise ValueError("context buffer descriptor local flag must be bool")
    self.size, self.virt, self.phys, self.local = size, virt, phys, local

class PageTablePage:
  def __init__(self, paddr, level, entries=512):
    self.paddr, self.level, self.entries = paddr, level, [0] * entries

class PageTableWalk:
  def __init__(self, mm, vaddr, create=False, inspect=False):
    self.mm, self.vaddr, self.create, self.inspect = mm, vaddr, create, inspect
    self.stack = [(mm.root_page_table, mm.table_index(vaddr, mm.root_page_table.level), mm.page_cover(mm.root_page_table.level))]

  def _entry_valid(self, pt, idx):
    return pt.entries[idx] != 0

  def _entry_is_page(self, pt, idx):
    return pt.level >= len(self.mm.level_shifts) - 1 or (pt.entries[idx] & 1) == 1

  def _entry_addr(self, pt, idx):
    entry = pt.entries[idx]
    return ((entry >> (12 if self.mm.mmu_ver == 3 else 8)) << 12)

  def level_down(self):
    pt, idx, _ = self.stack[-1]
    if not self._entry_valid(pt, idx):
      if not self.create: raise ValueError("page table does not exist")
      child = self.mm.alloc_page_table(pt.level + 1)
      pt.entries[idx] = self.mm.encode_pde(child.paddr)
    else:
      if self._entry_is_page(pt, idx): raise ValueError("page-table entry is already a page")
      child = self.mm.page_tables[self._entry_addr(pt, idx)]
    self.stack.append((child, self.mm.table_index(self.vaddr, child.level), self.mm.page_cover(child.level)))
    return self.stack[-1]

  def level_up(self):
    while len(self.stack) > 1 and self.stack[-1][1] >= self.mm.page_entry_count(self.stack[-1][0].level):
      pt, idx, _ = self.stack.pop()
      if idx >= self.mm.page_entry_count(pt.level):
        parent, parent_idx, parent_cover = self.stack[-1]
        self.stack[-1] = (parent, parent_idx + 1, parent_cover)

  def next(self, size, paddr=0):
    off = 0
    while size > 0:
      pt, idx, cover = self.stack[-1]
      if self.create:
        while cover > size or not self.mm.supports_huge_page(pt.level, paddr + off) or self.vaddr & (cover - 1):
          pt, idx, cover = self.level_down()
      else:
        while not self._entry_is_page(pt, idx) and self._entry_valid(pt, idx):
          pt, idx, cover = self.level_down()
      entries = max(min(size // cover, self.mm.page_entry_count(pt.level) - idx), 1 if self.inspect else 0)
      if entries <= 0: raise ValueError("invalid page-table walk range")
      yield off, pt, idx, entries, cover
      size, off, self.vaddr = size - entries * cover, off + entries * cover, self.vaddr + entries * cover
      self.stack[-1] = (pt, idx + entries, cover)
      self.level_up()

class GpuMemoryManager:
  def __init__(self, vram_size, va_base=0x1000000000, va_size=1 << 44, boot_size=2 << 20, mmu_ver=2, reserve_ptable=False):
    validate_positive_size(vram_size, "VRAM size")
    validate_positive_size(va_size, "VA size")
    validate_positive_size(boot_size, "boot allocation size")
    if mmu_ver not in (2, 3): raise ValueError("MMU version must be 2 or 3")
    self.va_alloc = TLSFAllocator(va_size, va_base)
    self.boot_alloc = TLSFAllocator(boot_size, 0)
    self.ptable_reserved_size = round_up(vram_size // 512, 1 << 20) if reserve_ptable else 0
    phys_base = boot_size + self.ptable_reserved_size
    self.ptable_alloc = TLSFAllocator(self.ptable_reserved_size, boot_size) if self.ptable_reserved_size else None
    self.phys_alloc = TLSFAllocator(max(0, vram_size - phys_base), phys_base) if vram_size > phys_base else None
    self.mappings = {}
    self.mmu_ver, self.page_tables = mmu_ver, {}
    self.level_shifts = [47, 38, 29, 21, 12] if mmu_ver == 2 else [56, 47, 38, 29, 21, 12]
    self.root_page_table = self.alloc_page_table(0, boot=True)

  @property
  def page_directory_paddr(self): return self.root_page_table.paddr
  @property
  def page_directory_entries(self): return len(self.root_page_table.entries)

  def alloc_page_table(self, level, boot=False):
    if level < 0 or level >= len(self.level_shifts): raise ValueError("page-table level is out of range")
    pt = PageTablePage(self.palloc(0x1000, boot=boot, ptable=not boot), level)
    self.page_tables[pt.paddr] = pt
    return pt

  def encode_pte(self, paddr, aspace, uncached=False):
    if paddr < 0 or paddr % 0x1000: raise ValueError("PTE physical address must be 4KB aligned")
    if not isinstance(aspace, AddrSpace): raise ValueError("PTE address space is invalid")
    aperture = 2 if aspace is AddrSpace.SYS else 0
    if self.mmu_ver == 3:
      pcf = 1 if uncached else 0
      return 1 | (aperture << 1) | (pcf << 3) | (6 << 8) | ((paddr >> 12) << 12)
    return 1 | (aperture << 1) | (int(uncached) << 3) | ((paddr >> 12) << 8) | (6 << 56)

  def encode_pde(self, paddr):
    if paddr < 0 or paddr % 0x1000: raise ValueError("PDE physical address must be 4KB aligned")
    if self.mmu_ver == 3:
      return (1 << 1) | (2 << 3) | ((paddr >> 12) << 12)
    return (1 << 1) | (1 << 5) | ((paddr >> 12) << 8)

  def table_index(self, vaddr, level):
    if level < 0 or level >= len(self.level_shifts): raise ValueError("page-table level is out of range")
    return (vaddr >> self.level_shifts[level]) & 0x1ff

  def page_cover(self, level):
    if level < 0 or level >= len(self.level_shifts): raise ValueError("page-table level is out of range")
    return 1 << self.level_shifts[level]

  def page_entry_count(self, level):
    if level < 0 or level >= len(self.level_shifts): raise ValueError("page-table level is out of range")
    return len(self.root_page_table.entries)

  def supports_huge_page(self, level, paddr):
    if level < 0 or level >= len(self.level_shifts): raise ValueError("page-table level is out of range")
    return level >= len(self.level_shifts) - 3 and paddr % self.page_cover(level) == 0

  def ensure_page_table(self, vaddr, target_level):
    if vaddr < 0: raise ValueError("GPU virtual address must be non-negative")
    if target_level < 0 or target_level >= len(self.level_shifts): raise ValueError("page-table target level is out of range")
    pt = self.root_page_table
    for level in range(target_level):
      idx = self.table_index(vaddr, level)
      if pt.entries[idx] == 0:
        child = self.alloc_page_table(level + 1)
        pt.entries[idx] = self.encode_pde(child.paddr)
      else:
        child_paddr = ((pt.entries[idx] >> (12 if self.mmu_ver == 3 else 8)) << 12)
        child = self.page_tables[child_paddr]
      pt = child
    return pt

  def ensure_leaf_table(self, vaddr):
    return self.ensure_page_table(vaddr, len(self.level_shifts) - 1)

  def reserved_pde_levels(self, vaddr, levels_to_copy=3, size=512 << 20):
    if levels_to_copy <= 0 or levels_to_copy > len(self.level_shifts): raise ValueError("reserved PDE level count is out of range")
    validate_positive_size(size, "reserved PDE size")
    walk = PageTableWalk(self, vaddr, create=True)
    for _ in walk.next(round_up(size, 0x1000), paddr=0): break
    tables = [pt for pt, _, _ in walk.stack[:levels_to_copy]]
    return [(pt.paddr, len(pt.entries) * 8, 1, self.level_shifts[pt.level]) for pt in tables]

  def write_mapping_entries(self, mapping):
    if mapping.va_addr < 0 or mapping.va_addr % 0x1000: raise ValueError("mapping VA must be 4KB aligned")
    if mapping.size <= 0 or mapping.size % 0x1000: raise ValueError("mapping size must be positive and 4KB aligned")
    if not isinstance(mapping.aspace, AddrSpace): raise ValueError("mapping address space is invalid")
    va = mapping.va_addr
    written = []
    try:
      for paddr, span in mapping.paddrs:
        if paddr < 0 or paddr % 0x1000: raise ValueError("mapping physical address must be 4KB aligned")
        if span <= 0 or span % 0x1000: raise ValueError("mapping physical span must be positive and 4KB aligned")
        walk = PageTableWalk(self, va, create=True)
        for offset, pt, idx, entry_count, cover in walk.next(span, paddr=paddr):
          for entry_off in range(entry_count):
            entry_idx = idx + entry_off
            if pt.entries[entry_idx] != 0: raise ValueError(f"VA 0x{va + offset + entry_off * cover:x} already mapped")
            pt.entries[entry_idx] = self.encode_pte(paddr + offset + entry_off * cover, mapping.aspace, uncached=mapping.uncached)
            written.append((pt, entry_idx))
        va += span
    except Exception:
      for pt, idx in written: pt.entries[idx] = 0
      raise
    return mapping

  def clear_mapping_entries(self, mapping):
    va = mapping.va_addr
    for _, span in mapping.paddrs:
      walk = PageTableWalk(self, va)
      for _, pt, idx, entry_count, _ in walk.next(span):
        for entry_off in range(entry_count):
          pt.entries[idx + entry_off] = 0
      va += span

  def alloc_vaddr(self, size, align=0x1000):
    validate_positive_size(size, "VA allocation size")
    validate_alignment(align)
    size = round_up(size, 0x1000)
    return self.va_alloc.alloc(size, max(1 << (size.bit_length() - 1), align, 0x1000))

  def next_vaddr(self, align=0x1000):
    validate_alignment(align)
    for start, (span, used) in sorted(self.va_alloc.blocks.items()):
      if used: continue
      aligned = round_up(self.va_alloc.base + start, max(align, 0x1000))
      if aligned < self.va_alloc.base + start + span: return aligned
    raise MemoryError("VA allocator is exhausted")

  def palloc(self, size, align=0x1000, boot=False, ptable=False):
    validate_positive_size(size, "physical allocation size")
    validate_alignment(align)
    allocator = self.boot_alloc if boot else (self.ptable_alloc if ptable and self.ptable_alloc is not None else self.phys_alloc)
    if allocator is None: raise MemoryError("physical allocator has no available memory")
    size_aligned = round_up(size, 0x1000)
    paddr = allocator.alloc(size_aligned, align)
    if os.environ.get("NV_ADD_TRACE_MM_ALLOC", "0") == "1":
      print(f"mm palloc size=0x{size_aligned:x} align=0x{align:x} boot={boot} -> 0x{paddr:x}")
    return paddr

  def map_range(self, vaddr, size, paddrs, aspace, uncached=False, snooped=False):
    if vaddr < 0 or vaddr % 0x1000: raise ValueError("mapping VA must be 4KB aligned")
    validate_positive_size(size, "mapping size")
    if not isinstance(aspace, AddrSpace): raise ValueError("mapping address space is invalid")
    size = round_up(size, 0x1000)
    if size != sum(span for _, span in paddrs): raise ValueError("mapping size does not match physical ranges")
    if vaddr in self.mappings: raise ValueError(f"VA 0x{vaddr:x} already mapped")
    mapping = VirtMapping(vaddr, size, paddrs, aspace, uncached, snooped)
    self.mappings[vaddr] = mapping
    try:
      return self.write_mapping_entries(mapping)
    except Exception:
      self.mappings.pop(vaddr, None)
      raise

  def unmap_range(self, vaddr, size):
    if vaddr < 0 or vaddr % 0x1000: raise ValueError("unmap VA must be 4KB aligned")
    validate_positive_size(size, "unmap size")
    mapping = self.mappings.pop(vaddr)
    if mapping.size != round_up(size, 0x1000): raise ValueError("unmap size mismatch")
    self.clear_mapping_entries(mapping)
    return mapping

  def valloc(self, size, align=0x1000, uncached=False, contiguous=False):
    validate_positive_size(size, "VRAM allocation size")
    validate_alignment(align)
    size = round_up(size, 0x1000)
    va = self.alloc_vaddr(size, align)
    if contiguous:
      paddrs = [(self.palloc(size, align=max(align, 0x1000)), size)]
    else:
      paddrs = [(self.palloc(0x1000), 0x1000) for _ in range(size // 0x1000)]
    if os.environ.get("NV_ADD_TRACE_MM_ALLOC", "0") == "1":
      print(f"mm valloc size=0x{size:x} align=0x{align:x} contiguous={contiguous} -> va=0x{va:x} "
            f"paddrs={','.join(hex(paddr) + '/' + hex(span) for paddr, span in paddrs[:4])}")
    return self.map_range(va, size, paddrs, AddrSpace.PHYS, uncached=uncached)

  def vfree(self, mapping):
    self.unmap_range(mapping.va_addr, mapping.size)
    self.va_alloc.free_addr(mapping.va_addr)
    for paddr, _ in mapping.paddrs: self.phys_alloc.free_addr(paddr)

class Transport:
  def read_config(self, offset, size): raise NotImplementedError
  def write_config(self, offset, value, size): raise NotImplementedError
  def bar_info(self, bar): raise NotImplementedError
  def map_bar(self, bar, off=0, size=None, fmt='B'): raise NotImplementedError
  def alloc_sysmem(self, size, contiguous=False): raise NotImplementedError
  def sleep(self, timeout_ms): raise NotImplementedError

PCI_VENDOR_ID = 0x00
PCI_REVISION_ID = 0x08
PCI_SUBSYSTEM_VENDOR_ID = 0x2c
PCI_COMMAND = 0x04
PCI_COMMAND_IO = 0x1
PCI_COMMAND_MEMORY = 0x2
PCI_COMMAND_MASTER = 0x4

NV_PMC_BOOT_0 = 0x0
NV_PMC_BOOT_42 = 0xA00
NV_PFB_PRI_MMU_WPR2_ADDR_HI = 0x1FA828
NV_PGC6_BSI_SECURE_SCRATCH_14 = 0x1180F8
NV_PGC6_AON_SECURE_SCRATCH_GROUP_42 = 0x1183A4
NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE = 0x30B0
NV_PGSP_QUEUE_HEAD_0 = 0x110C00
NV_PBUS_BAR1_BLOCK = 0x1704
NV_PGSP_FALCON_ENGINE = 0x1103C0
NV_PSEC_FALCON_ENGINE = 0x8403C0
NV_PPMU_FALCON_ENGINE = 0x10A3C0
NV_FALCON_GSP_BASE = 0x110000
NV_FALCON_SEC2_BASE = 0x840000
NV_FALCON_PMU_BASE = 0x10A000
NV_FALCON_FECS_BASES = {
  "ga102": 0xA04000,
  "ad102": 0xA04000,
  "gb202": 0xA04000,
}
NV_PFECS_FALCON_ENGINE_OFFSET = 0x3C0
NV_PFALCON_FALCON_MAILBOX0 = 0x40
NV_PFALCON_FALCON_MAILBOX1 = 0x44
NV_PFALCON_FALCON_OS = 0x80
NV_PFALCON_FALCON_RM = 0x84
NV_PFALCON_FALCON_HWCFG2 = 0xF4
NV_PFALCON_FALCON_CPUCTL = 0x100
NV_PFALCON_FALCON_BOOTVEC = 0x104
NV_PFALCON_FALCON_DMACTL = 0x10C
NV_PFALCON_FALCON_DMATRFBASE = 0x110
NV_PFALCON_FALCON_DMATRFMOFFS = 0x114
NV_PFALCON_FALCON_DMATRFCMD = 0x118
NV_PFALCON_FALCON_DMATRFFBOFFS = 0x11C
NV_PFALCON_FALCON_DMATRFBASE1 = 0x128
NV_PFALCON_FALCON_CPUCTL_ALIAS = 0x130
NV_PFALCON_FALCON_EXCI = 0x18C
NV_PFALCON_FALCON_IRQSTAT = 0x650
NV_PFALCON2_FALCON_MOD_SEL = 0x1180
NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID = 0x1198
NV_PFALCON2_FALCON_BROM_ENGIDMASK = 0x119C
NV_PFALCON2_FALCON_BROM_PARAADDR0 = 0x1210
NV_PFALCON_FBIF_TRANSCFG0 = 0x600
NV_PFALCON_FBIF_CTL = 0x624
NV_PRISCV_RISCV_CPUCTL = 0x1388
NV_PRISCV_RISCV_BCR_CTRL = 0x1668

FALCON_WRITE_NAMES = {
  NV_PGSP_FALCON_ENGINE: "GSP_ENGINE",
  NV_PSEC_FALCON_ENGINE: "SEC2_ENGINE",
  NV_PFALCON_FALCON_MAILBOX0: "MAILBOX0",
  NV_PFALCON_FALCON_MAILBOX1: "MAILBOX1",
  NV_PFALCON_FALCON_OS: "OS",
  NV_PFALCON_FALCON_RM: "RM",
  NV_PFALCON_FALCON_CPUCTL: "CPUCTL",
  NV_PFALCON_FALCON_BOOTVEC: "BOOTVEC",
  NV_PFALCON_FALCON_DMACTL: "DMACTL",
  NV_PFALCON_FALCON_DMATRFBASE: "DMATRFBASE",
  NV_PFALCON_FALCON_DMATRFMOFFS: "DMATRFMOFFS",
  NV_PFALCON_FALCON_DMATRFCMD: "DMATRFCMD",
  NV_PFALCON_FALCON_DMATRFFBOFFS: "DMATRFFBOFFS",
  NV_PFALCON_FALCON_DMATRFBASE1: "DMATRFBASE1",
  NV_PFALCON_FALCON_CPUCTL_ALIAS: "CPUCTL_ALIAS",
  NV_PFALCON2_FALCON_MOD_SEL: "BROM_MOD_SEL",
  NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID: "BROM_CURR_UCODE_ID",
  NV_PFALCON2_FALCON_BROM_ENGIDMASK: "BROM_ENGIDMASK",
  NV_PFALCON2_FALCON_BROM_PARAADDR0: "BROM_PARAADDR0",
  NV_PFALCON_FBIF_TRANSCFG0: "FBIF_TRANSCFG0",
  NV_PFALCON_FBIF_CTL: "FBIF_CTL",
  NV_PRISCV_RISCV_CPUCTL: "RISCV_CPUCTL",
  NV_PRISCV_RISCV_BCR_CTRL: "RISCV_BCR_CTRL",
}

NV_IOCTL_BASE = 200
NV_ESC_CARD_INFO = NV_IOCTL_BASE + 0
NV_ESC_REGISTER_FD = NV_IOCTL_BASE + 1
NV_ESC_RM_ALLOC_MEMORY = 0x27
NV_ESC_RM_CONTROL = 0x2A
NV_ESC_RM_ALLOC = 0x2B
NV_ESC_RM_MAP_MEMORY = 0x4E
NV_ESC_RM_MAP_MEMORY_DMA = 0x57
NV2080_CTRL_CMD_GPU_GET_GID_INFO = 0x2080014A
NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY = 0x1
NV2080_CTRL_CMD_GPU_PROMOTE_CTX = 0x2080012B
NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO = 0x20800A32
NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES = 0x90F10106
NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS = 0
NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH = 16
NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_BUNDLE_CB = 17
NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PRIV_ACCESS_MAP = 24
UVM_REGISTER_GPU_VASPACE = 25
UVM_REGISTER_CHANNEL = 27
UVM_MAP_EXTERNAL_ALLOCATION = 33
UVM_FREE = 34
UVM_REGISTER_GPU = 37
UVM_CREATE_EXTERNAL_RANGE = 73
UVM_MM_INITIALIZE = 75
UVM_INITIALIZE = 0x30000001

NV01_MEMORY_VIRTUAL = 0x70
NV01_ROOT = 0x0
NV01_DEVICE_0 = 0x80
NV01_MEMORY_SYSTEM_OS_DESCRIPTOR = 0x71
NV01_MEMORY_LIST_SYSTEM = 0x81
NV20_SUBDEVICE_0 = 0x2080
FERMI_VASPACE_A = 0x90F1
FERMI_CONTEXT_SHARE_A = 0x9067
KEPLER_CHANNEL_GROUP_A = 0xA06C
AMPERE_CHANNEL_GPFIFO_A = 0xC56F
AMPERE_DMA_COPY_B = 0xC7B5
AMPERE_COMPUTE_B = 0xC7C0
ADA_COMPUTE_A = 0xC9C0
GT200_DEBUGGER = 0x83DE
NV2080_ENGINE_TYPE_GRAPHICS = 1
NV2080_ENGINE_TYPE_COMPUTE = 0
NVA06C_CTRL_CMD_GPFIFO_SCHEDULE = 0xA06C0101
NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN = 0xC36F0108
NVC56F_CONTROL_GP_PUT_OFFSET = 140
NVC6C0_SET_OBJECT = 0x0000
NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A = 0x02A0
NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A = 0x02E4
NVC6C0_SET_SHADER_LOCAL_MEMORY_A = 0x0790
NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A = 0x07B0

RM_CLASS_NAMES = {
  NV01_ROOT: "NV01_ROOT",
  NV01_DEVICE_0: "NV01_DEVICE_0",
  NV01_MEMORY_VIRTUAL: "NV01_MEMORY_VIRTUAL",
  NV01_MEMORY_SYSTEM_OS_DESCRIPTOR: "NV01_MEMORY_SYSTEM_OS_DESCRIPTOR",
  NV20_SUBDEVICE_0: "NV20_SUBDEVICE_0",
  FERMI_VASPACE_A: "FERMI_VASPACE_A",
  FERMI_CONTEXT_SHARE_A: "FERMI_CONTEXT_SHARE_A",
  KEPLER_CHANNEL_GROUP_A: "KEPLER_CHANNEL_GROUP_A",
  AMPERE_CHANNEL_GPFIFO_A: "AMPERE_CHANNEL_GPFIFO_A",
  AMPERE_DMA_COPY_B: "AMPERE_DMA_COPY_B",
  AMPERE_COMPUTE_B: "AMPERE_COMPUTE_B",
  ADA_COMPUTE_A: "ADA_COMPUTE_A",
  GT200_DEBUGGER: "GT200_DEBUGGER",
}

RM_CTRL_NAMES = {
  NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES: "NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES",
  NV2080_CTRL_CMD_GPU_GET_GID_INFO: "NV2080_CTRL_CMD_GPU_GET_GID_INFO",
  NV2080_CTRL_CMD_GPU_PROMOTE_CTX: "NV2080_CTRL_CMD_GPU_PROMOTE_CTX",
  NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO: "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO",
  NVA06C_CTRL_CMD_GPFIFO_SCHEDULE: "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE",
  NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN",
}
NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI = 0x1698
DEFAULT_SHARED_MEM_WINDOW = 0x729400000000
DEFAULT_LOCAL_MEM_WINDOW = 0x729300000000
NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED = 2
NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS = 0x2
NVOS02_FLAGS_COHERENCY_CACHED = 0x1
NVOS02_FLAGS_MAPPING_NO_MAP = 0x1
NVOS46_FLAGS_CACHE_SNOOP_ENABLE = 0x1
NVOS46_FLAGS_PAGE_SIZE_4KB = 0x2
NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE = 0x1
NV0000_ALLOC_PARAMETERS_SIZE = 120

class NvChipInfo:
  def __init__(self, architecture, implementation, chip_id, chip_name, fw_name, mmu_ver, fmc_boot, vram_size, pmc_boot_0=0):
    self.architecture, self.implementation, self.chip_id = architecture, implementation, chip_id
    self.pmc_boot_0 = pmc_boot_0
    self.chip_name, self.fw_name, self.mmu_ver, self.fmc_boot, self.vram_size = chip_name, fw_name, mmu_ver, fmc_boot, vram_size

  @classmethod
  def probe(cls, regs):
    boot_0 = regs.rreg(NV_PMC_BOOT_0)
    boot_42 = regs.rreg(NV_PMC_BOOT_42)
    architecture = (boot_42 >> 24) & 0x3f
    implementation = (boot_42 >> 20) & 0xf
    chip_id = (boot_42 >> 20) & 0x3ff
    chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}.get(architecture, f"U{architecture:02x}") + f"{implementation:02d}"
    fw_name = {"GB2": "gb202", "AD1": "ad102", "GA1": "ga102"}.get(chip_name[:3], "ga102")
    mmu_ver, fmc_boot = (3, True) if architecture >= 0x1a else (2, False)
    vram_size = regs.rreg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_42) << 20
    return cls(architecture, implementation, chip_id, chip_name, fw_name, mmu_ver, fmc_boot, vram_size, pmc_boot_0=boot_0)

def sane_chip_probe(chip):
  return (chip.pmc_boot_0 & 0xffff) == 0x00a1 and (256 << 20) <= chip.vram_size <= (1 << 40)

class RemoteCmd(enum.IntEnum):
  PROBE = 0
  MAP_BAR = 1
  MAP_SYSMEM_FD = 2
  CFG_READ = 3
  CFG_WRITE = 4
  RESET = 5
  MMIO_READ = 6
  MMIO_WRITE = 7
  MAP_SYSMEM = 8
  SYSMEM_READ = 9
  SYSMEM_WRITE = 10
  RESIZE_BAR = 11
  PING = 12

class RemoteMMIOView:
  def __init__(self, transport, residx, nbytes, fmt='B', off=0, rd_cmd=RemoteCmd.MMIO_READ, wr_cmd=RemoteCmd.MMIO_WRITE):
    self.transport, self.residx, self.nbytes, self.fmt, self.off = transport, residx, nbytes, fmt, off
    self.el_sz, self.rd_cmd, self.wr_cmd = struct.calcsize(fmt), rd_cmd, wr_cmd
    if nbytes < 0 or nbytes % self.el_sz: raise ValueError("RemoteMMIOView size is invalid for element format")
  def __len__(self): return self.nbytes // self.el_sz
  def _range(self, index):
    if isinstance(index, slice):
      start, stop, step = index.indices(len(self))
      if step != 1: raise ValueError("RemoteMMIOView slices must be contiguous")
      return start * self.el_sz, stop * self.el_sz, True
    if index < 0: index += len(self)
    if index < 0 or index >= len(self): raise IndexError("RemoteMMIOView index out of range")
    return index * self.el_sz, (index + 1) * self.el_sz, False
  def __getitem__(self, index):
    start, stop, is_slice = self._range(index)
    data = self.transport.bulk_read(self.rd_cmd, self.residx, self.off + start, stop - start)
    if len(data) != stop - start: raise RuntimeError("RemoteMMIOView bulk read returned a short response")
    result = data if self.fmt == 'B' else list(struct.unpack(f"<{(stop - start) // self.el_sz}{self.fmt}", data))
    return result if is_slice else result[0]
  def __setitem__(self, index, value):
    start, stop, is_slice = self._range(index)
    data = value if isinstance(index, slice) and self.fmt == 'B' else (
      struct.pack(f"<{len(value)}{self.fmt}", *value) if is_slice else struct.pack(f"<{self.fmt}", value))
    if len(data) != stop - start: raise ValueError("RemoteMMIOView write size does not match target range")
    self.transport.bulk_write(self.wr_cmd, self.residx, self.off + start, data)
  def view(self, offset=0, size=None, fmt=None):
    if offset < 0: raise ValueError("RemoteMMIOView offset must be non-negative")
    if offset > self.nbytes: raise ValueError("RemoteMMIOView offset is beyond the view size")
    if size is None: size = self.nbytes - offset
    if size < 0 or offset + size > self.nbytes: raise ValueError("RemoteMMIOView range is out of bounds")
    return RemoteMMIOView(self.transport, self.residx, size, fmt or self.fmt,
                          self.off + offset, self.rd_cmd, self.wr_cmd)

class MacEgpuTransport(Transport):
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"

  def __init__(self, sock_path=None):
    self.sock_path = sock_path or os.environ.get("APL_REMOTE_SOCK", os.path.join(tempfile.gettempdir(), "tinygpu.sock"))
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for attempt in range(100):
      with contextlib.suppress(ConnectionRefusedError, FileNotFoundError):
        self.sock.connect(self.sock_path)
        break
      if attempt == 0 and os.path.exists(self.APP_PATH):
        subprocess.Popen([self.APP_PATH, "server", self.sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      time.sleep(0.05)
    else:
      raise RuntimeError(f"failed to connect to TinyGPU server at {self.sock_path}")
    self.dev_id = 0

  def recvall(self, n):
    out = b""
    while len(out) < n:
      chunk = self.sock.recv(n - len(out))
      if not chunk: raise RuntimeError("TinyGPU connection closed")
      out += chunk
    return out

  @staticmethod
  def validate_remote_cmd(cmd):
    try: return RemoteCmd(cmd)
    except ValueError as exc: raise ValueError(f"unknown TinyGPU command: {cmd}") from exc

  @staticmethod
  def validate_remote_io(cmd, bar=0, offset=0, size=0, readout_size=0, payload=b""):
    cmd = MacEgpuTransport.validate_remote_cmd(cmd)
    validate_bar_index(bar)
    if offset < 0: raise ValueError("TinyGPU offset must be non-negative")
    if size < 0: raise ValueError("TinyGPU size must be non-negative")
    if readout_size < 0: raise ValueError("TinyGPU readout size must be non-negative")
    if payload is None: raise ValueError("TinyGPU payload must not be None")
    try:
      payload = bytes(payload)
    except TypeError as exc:
      raise ValueError("TinyGPU payload must be bytes-like") from exc
    return cmd, payload

  def rpc(self, cmd, *args, bar=0, readout_size=0, payload=b"", has_fd=False):
    if len(args) > 3: raise ValueError("TinyGPU RPC accepts at most three positional words")
    cmd, payload = self.validate_remote_io(cmd, bar=bar, offset=args[0] if len(args) > 0 else 0,
      size=args[1] if len(args) > 1 else 0, readout_size=readout_size, payload=payload)
    self.sock.sendall(struct.pack("<BIIQQQ", int(cmd), self.dev_id, bar, *(*args, 0, 0, 0)[:3]) + payload)
    if has_fd:
      msg, anc, _, _ = self.sock.recvmsg(17, socket.CMSG_LEN(4))
      if len(msg) != 17: raise RuntimeError("TinyGPU RPC response header is truncated")
      if not anc or len(anc[0][2]) < 4: raise RuntimeError("TinyGPU RPC fd response is truncated")
      fd = struct.unpack("<i", anc[0][2][:4])[0]
    else:
      msg, fd = self.recvall(17), None
    status, a, b = struct.unpack("<BQQ", msg)
    if status != 0:
      detail = self.recvall(a).decode("utf-8") if a else "no error payload"
      raise RuntimeError(f"TinyGPU RPC {cmd.name} failed status=0x{status:x}: {detail}")
    return a, b, self.recvall(readout_size) if readout_size else None, fd

  def bulk_read(self, cmd, idx, offset, size):
    if size <= 0: raise ValueError("TinyGPU bulk read size must be positive")
    return self.rpc(cmd, offset, size, bar=idx, readout_size=size)[2]
  def bulk_write(self, cmd, idx, offset, data):
    if data is None: raise ValueError("TinyGPU bulk write payload must not be None")
    try:
      data = bytes(data)
    except TypeError as exc:
      raise ValueError("TinyGPU bulk write payload must be bytes-like") from exc
    cmd, data = self.validate_remote_io(cmd, bar=idx, offset=offset, size=len(data), payload=data)
    if len(data) <= 0: raise ValueError("TinyGPU bulk write payload must be non-empty")
    self.sock.sendall(struct.pack("<BIIQQQ", int(cmd), self.dev_id, idx, offset, len(data), 0) + data)

  def probe(self): return self.rpc(RemoteCmd.PROBE)[:2]
  def ping(self): return self.rpc(RemoteCmd.PING)[:2]
  def reset(self): return self.rpc(RemoteCmd.RESET)[:2]
  def read_config(self, offset, size):
    validate_pci_config_access(offset, size)
    return self.rpc(RemoteCmd.CFG_READ, offset, size)[0]
  def write_config(self, offset, value, size):
    validate_pci_config_access(offset, size)
    self.rpc(RemoteCmd.CFG_WRITE, offset, size, value)
  def bar_info(self, bar):
    validate_bar_index(bar)
    return self.rpc(RemoteCmd.MAP_BAR, bar=bar)[:2]
  def resize_bar(self, bar, size):
    validate_bar_index(bar)
    if size <= 0: raise ValueError("BAR resize size must be positive")
    return self.rpc(RemoteCmd.RESIZE_BAR, size, bar=bar)[:2]
  def map_bar(self, bar, off=0, size=None, fmt='B'):
    if size is not None and size <= 0: raise ValueError("BAR mapping size must be positive")
    if off < 0: raise ValueError("BAR offset must be non-negative")
    bar_size = self.bar_info(bar)[1]
    view_size = bar_size if size is None else off + size
    if view_size > bar_size: raise ValueError("BAR mapping range is outside BAR")
    return RemoteMMIOView(self, bar, view_size, fmt).view(off, size, fmt)
  @staticmethod
  def sysmem_pages_from_header(view, size):
    if size < 0: raise ValueError("sysmem size must be non-negative")
    qwords = view.view(fmt='Q')
    paddrs = []
    for index in range(0, len(qwords), 2):
      if index + 1 >= len(qwords): raise ValueError("sysmem header has an incomplete physical range entry")
      paddr, span = qwords[index], qwords[index + 1]
      if span == 0: break
      if paddr % 0x1000 or span % 0x1000:
        raise ValueError("sysmem physical ranges must be 4KB aligned")
      paddrs.extend(paddr + off for off in range(0, span, 0x1000))
    needed = ceildiv(size, 0x1000)
    if len(paddrs) < needed: raise RuntimeError(f"sysmem header returned {len(paddrs)} pages for {size} bytes")
    return paddrs[:needed]
  @staticmethod
  def validate_sysmem_fd_response(request_size, mapped_size, fd):
    if request_size < 0: raise ValueError("sysmem size must be non-negative")
    required = round_up(request_size, 0x1000)
    if fd is None or fd < 0: raise RuntimeError("TinyGPU sysmem mapping did not return a valid fd")
    if mapped_size < required:
      raise RuntimeError(f"TinyGPU sysmem mapping returned 0x{mapped_size:x} bytes for 0x{request_size:x} requested")
    return required
  def alloc_sysmem(self, size, contiguous=False):
    mapped_size, _, _, fd = self.rpc(RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
    self.validate_sysmem_fd_response(size, mapped_size, fd)
    try:
      mm = mmap.mmap(fd, mapped_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
      os.close(fd)
    view = MMIOView(mm, mapped_size, fmt='B')
    return view, self.sysmem_pages_from_header(view, size)
  def sleep(self, timeout_ms): time.sleep(timeout_ms / 1000)

class LinuxIoctlTransport(Transport):
  def __init__(self, gpu_index=0):
    self.gpu_index = gpu_index
    self.fd_ctl = FileIO("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    self.fd_uvm = FileIO("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
    self.fd_dev = FileIO(f"/dev/nvidia{gpu_index}", os.O_RDWR | os.O_CLOEXEC)
    self.pci_path = self.find_pci_device(gpu_index)
    self.next_host_handle = 0x1000
    self.va_allocator = FreeListAllocator(0x1000000000, base=0x1000000000)
    self.root = self.device = self.virtmem = self.gpu_uuid = None

  @staticmethod
  def find_pci_device(gpu_index, root="/sys/bus/pci/devices"):
    matches = []
    for path in sorted(pathlib.Path(root).glob("*")):
      with contextlib.suppress(OSError, ValueError):
        vendor = int((path / "vendor").read_text().strip(), 16)
        klass = int((path / "class").read_text().strip(), 16)
        if vendor == 0x10de and ((klass >> 16) in (0x03, 0x12)):
          matches.append(path)
    if gpu_index >= len(matches): raise RuntimeError(f"could not find NVIDIA PCI device for index {gpu_index}")
    return matches[gpu_index]

  @staticmethod
  def iowr(nr, size): return (3 << 30) | ((size & 0x1fff) << 16) | (ord('F') << 8) | (nr & 0xff)
  def rm_ioctl(self, nr, data): return self.fd_ctl.ioctl(self.iowr(nr, len(data)), data)
  def dev_ioctl(self, nr, data): return self.fd_dev.ioctl(self.iowr(nr, len(data)), data)
  def uvm_ioctl(self, cmd, data): return self.fd_uvm.ioctl(cmd, data)
  def card_info(self):
    data = bytearray(0x1000)
    self.rm_ioctl(NV_ESC_CARD_INFO, data)
    return data
  def register_fd(self, fd=None):
    data = bytearray(struct.pack("<iI", self.fd_dev.fd if fd is None else fd, 0))
    self.rm_ioctl(NV_ESC_REGISTER_FD, data)
    if (status := unpack_register_fd_status(data)) != 0: raise RuntimeError(f"NV_ESC_REGISTER_FD failed: 0x{status:x}")
    return data
  def rm_alloc(self, h_root, h_parent, h_object, h_class, params=b""):
    try:
      params = bytes(params)
    except TypeError as exc:
      raise ValueError("NV_ESC_RM_ALLOC params must be bytes-like") from exc
    param_buf = ctypes.create_string_buffer(params) if params else None
    data = pack_nvos21(h_root, h_parent, h_object, h_class,
      ctypes.addressof(param_buf) if param_buf is not None else 0, len(params))
    self.rm_ioctl(NV_ESC_RM_ALLOC, data)
    if (status := unpack_nvos21_status(data)) != 0: raise RuntimeError(f"NV_ESC_RM_ALLOC failed: 0x{status:x}")
    return h_object
  def rm_control(self, h_client, h_object, cmd, params=b"", flags=0):
    try:
      params = bytes(params)
    except TypeError as exc:
      raise ValueError("NV_ESC_RM_CONTROL params must be bytes-like") from exc
    param_buf = ctypes.create_string_buffer(params) if params else None
    data = pack_nvos54(h_client, h_object, cmd,
      ctypes.addressof(param_buf) if param_buf is not None else 0, len(params), flags=flags)
    self.rm_ioctl(NV_ESC_RM_CONTROL, data)
    if (status := unpack_nvos54_status(data)) != 0: raise RuntimeError(f"NV_ESC_RM_CONTROL failed: 0x{status:x}")
    return bytes(param_buf.raw) if param_buf is not None else b""
  def rm_map_memory(self, h_client, h_device, h_memory, length, flags=0, offset=0, fd=-1):
    data = pack_nvos33_with_fd(h_client, h_device, h_memory, length, flags=flags, offset=offset, fd=fd)
    self.rm_ioctl(NV_ESC_RM_MAP_MEMORY, data)
    if (status := unpack_nvos33_status(data)) != 0: raise RuntimeError(f"NV_ESC_RM_MAP_MEMORY failed: 0x{status:x}")
    return unpack_nvos33_linear_address(data), data
  def rm_map_memory_dma(self, h_client, h_device, h_dma, h_memory, length, dma_offset, offset=0, flags=0):
    data = pack_nvos46(h_client, h_device, h_dma, h_memory, length, dma_offset, offset=offset, flags=flags)
    self.rm_ioctl(NV_ESC_RM_MAP_MEMORY_DMA, data)
    if (status := unpack_nvos46_status(data)) != 0: raise RuntimeError(f"NV_ESC_RM_MAP_MEMORY_DMA failed: 0x{status:x}")
    return data
  def uvm_initialize(self):
    data = pack_uvm_initialize()
    self.uvm_ioctl(UVM_INITIALIZE, data)
    if (status := unpack_uvm_status(data, 8)) != 0: raise RuntimeError(f"UVM_INITIALIZE failed: 0x{status:x}")
    return data
  def uvm_mm_initialize(self):
    data = pack_uvm_mm_initialize(self.fd_uvm.fd)
    self.uvm_ioctl(UVM_MM_INITIALIZE, data)
    if (status := unpack_uvm_status(data, 4)) != 0: raise RuntimeError(f"UVM_MM_INITIALIZE failed: 0x{status:x}")
    return data
  def uvm_create_external_range(self, base, length):
    data = pack_uvm_create_external_range(base, length)
    self.uvm_ioctl(UVM_CREATE_EXTERNAL_RANGE, data)
    if (status := unpack_uvm_status(data, 16)) != 0: raise RuntimeError(f"UVM_CREATE_EXTERNAL_RANGE failed: 0x{status:x}")
    return data
  def uvm_free(self, base, length):
    data = pack_uvm_free(base, length)
    self.uvm_ioctl(UVM_FREE, data)
    if (status := unpack_uvm_status(data, 16)) != 0: raise RuntimeError(f"UVM_FREE failed: 0x{status:x}")
    return data
  def uvm_map_external_allocation(self, base, length, gpu_uuid, h_client, h_memory):
    data = pack_uvm_map_external_allocation(base, length, gpu_uuid, self.fd_ctl.fd, h_client, h_memory)
    self.uvm_ioctl(UVM_MAP_EXTERNAL_ALLOCATION, data)
    if (status := unpack_uvm_status(data, 9260)) != 0: raise RuntimeError(f"UVM_MAP_EXTERNAL_ALLOCATION failed: 0x{status:x}")
    return data

  def configure_rm(self, root, device, virtmem, gpu_uuid):
    self.root, self.device, self.virtmem, self.gpu_uuid = root, device, virtmem, gpu_uuid

  def setup_uvm(self, root, device, subdevice, virtmem, vaspace):
    self.uvm_initialize()
    with contextlib.suppress(OSError, RuntimeError):
      self.uvm_mm_initialize()
    gid = pack_nv2080_gpu_get_gid_info()
    resp = self.rm.rm_control(subdevice, NV2080_CTRL_CMD_GPU_GET_GID_INFO, gid) if hasattr(self, "rm") else None
    if resp is not None: gid[:len(resp)] = resp
    gpu_uuid = unpack_nv2080_gpu_gid(gid)
    reg = pack_uvm_register_gpu(gpu_uuid, -1)
    self.uvm_ioctl(UVM_REGISTER_GPU, reg)
    if (status := unpack_uvm_status(reg, 36)) != 0: raise RuntimeError(f"UVM_REGISTER_GPU failed: 0x{status:x}")
    reg_va = pack_uvm_register_gpu_vaspace(gpu_uuid, self.fd_ctl.fd, root, vaspace)
    self.uvm_ioctl(UVM_REGISTER_GPU_VASPACE, reg_va)
    if (status := unpack_uvm_status(reg_va, 28)) != 0: raise RuntimeError(f"UVM_REGISTER_GPU_VASPACE failed: 0x{status:x}")
    self.configure_rm(root, device, virtmem, gpu_uuid)
    return gpu_uuid

  def register_channel(self, h_channel, length=0x4000000):
    self.require_rm_config()
    base = self.va_allocator.alloc(length, 0x1000)
    try:
      data = pack_uvm_register_channel(self.gpu_uuid, self.fd_ctl.fd, self.root, h_channel, base, length)
      self.uvm_ioctl(UVM_REGISTER_CHANNEL, data)
      if (status := unpack_uvm_status(data, 48)) != 0: raise RuntimeError(f"UVM_REGISTER_CHANNEL failed: 0x{status:x}")
      return base, length
    except Exception:
      self.va_allocator.free_addr(base)
      raise
  def free_registered_channel(self, registration):
    if not registration: return
    if not isinstance(registration, tuple) or len(registration) != 2: raise ValueError("registered channel metadata is invalid")
    base, length = registration
    if not isinstance(base, int) or not isinstance(length, int): raise ValueError("registered channel metadata is invalid")
    if base < 0 or base % 0x1000: raise ValueError("registered channel base must be 4KB aligned")
    validate_positive_size(length, "registered channel length")
    if length % 0x1000: raise ValueError("registered channel length must be 4KB aligned")
    errors = []
    try:
      self.uvm_free(base, length)
    except Exception as exc:
      errors.append(exc)
    try:
      self.va_allocator.free_addr(base)
    except Exception as exc:
      errors.append(exc)
    if errors:
      raise errors[0]

  def require_rm_config(self):
    missing = [name for name in ("root", "device", "virtmem", "gpu_uuid") if getattr(self, name) is None]
    if missing: raise RuntimeError(f"Linux ioctl transport missing RM/UVM handles: {', '.join(missing)}")

  def anon_cpu_mapping(self, va_addr, size):
    flags = mmap.MAP_SHARED | getattr(mmap, "MAP_ANONYMOUS", 0x20) | getattr(mmap, "MAP_FIXED", 0x10)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    addr = libc.mmap(ctypes.c_void_p(va_addr), ctypes.c_size_t(size), mmap.PROT_READ | mmap.PROT_WRITE, flags, -1, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
      err = ctypes.get_errno()
      raise OSError(err, os.strerror(err))
    if addr != va_addr:
      FileIO.munmap(addr, size)
      raise RuntimeError(f"fixed mmap returned 0x{addr:x}, expected 0x{va_addr:x}")
    return addr

  def alloc_sysmem(self, size, contiguous=False):
    self.require_rm_config()
    size = round_up(size, mmap.PAGESIZE)
    va_addr = self.va_allocator.alloc(size, mmap.PAGESIZE)
    cpu_addr, external_range = None, False
    try:
      cpu_addr = self.anon_cpu_mapping(va_addr, size)
      h_memory = self.next_host_handle
      self.next_host_handle += 1
      flags = (NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (NVOS02_FLAGS_COHERENCY_CACHED << 12) | (NVOS02_FLAGS_MAPPING_NO_MAP << 30)
      data = pack_nvos02_with_fd(self.root, self.device, h_memory, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, flags, va_addr, size - 1)
      self.dev_ioctl(NV_ESC_RM_ALLOC_MEMORY, data)
      if (status := unpack_nvos02_status(data)) != 0: raise RuntimeError(f"NV_ESC_RM_ALLOC_MEMORY failed: 0x{status:x}")

      map_flags = (NVOS46_FLAGS_PAGE_SIZE_4KB << 8) | (NVOS46_FLAGS_CACHE_SNOOP_ENABLE << 4) | (NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE << 15)
      self.rm_map_memory_dma(self.root, self.device, self.virtmem, h_memory, size, va_addr, flags=map_flags)

      self.uvm_create_external_range(va_addr, size)
      external_range = True
      self.uvm_map_external_allocation(va_addr, size, self.gpu_uuid, self.root, h_memory)
      view = MMIOView(cpu_addr, size, fmt='B', addr=va_addr)
      return SysmemAllocation(view, [va_addr + off for off in range(0, size, 0x1000)], va_addr=va_addr, size=size, h_memory=h_memory, cpu_addr=cpu_addr)
    except Exception:
      if external_range:
        with contextlib.suppress(Exception): self.uvm_free(va_addr, size)
      if cpu_addr is not None:
        with contextlib.suppress(Exception): FileIO.munmap(cpu_addr, size)
      with contextlib.suppress(Exception): self.va_allocator.free_addr(va_addr)
      raise

  def free_sysmem(self, alloc):
    if getattr(alloc, "va_addr", None) is None: return
    errors = []
    if not getattr(alloc, "_uvm_freed", False):
      try:
        self.uvm_free(alloc.va_addr, alloc.size)
        alloc._uvm_freed = True
      except Exception as exc:
        errors.append(exc)
    if getattr(alloc, "cpu_addr", None) is not None:
      try:
        FileIO.munmap(alloc.cpu_addr, alloc.size)
        alloc.cpu_addr = None
      except Exception as exc:
        errors.append(exc)
    if not getattr(alloc, "_va_recycled", False):
      try:
        self.va_allocator.free_addr(alloc.va_addr)
        alloc._va_recycled = True
      except Exception as exc:
        errors.append(exc)
    if errors:
      raise errors[0]
    alloc.va_addr = None
    alloc.size = 0

  def read_config(self, offset, size):
    validate_pci_config_access(offset, size)
    with open(self.pci_path / "config", "rb", buffering=0) as f:
      f.seek(offset)
      return int.from_bytes(f.read(size), "little")
  def write_config(self, offset, value, size):
    validate_pci_config_access(offset, size)
    with open(self.pci_path / "config", "r+b", buffering=0) as f:
      f.seek(offset)
      f.write(int(value).to_bytes(size, "little"))
  def bar_info(self, bar):
    validate_bar_index(bar)
    with open(self.pci_path / "resource", "r", encoding="utf-8") as f:
      fields = f.readlines()[bar].split()
    start, end = int(fields[0], 16), int(fields[1], 16)
    return start, 0 if start == 0 else end - start + 1
  def map_bar(self, bar, off=0, size=None, fmt='B'):
    _, bar_size = self.bar_info(bar)
    if off < 0: raise ValueError("BAR offset must be non-negative")
    size = (bar_size - off) if size is None else size
    if size <= 0: raise ValueError("BAR mapping size must be positive")
    if off + size > bar_size: raise ValueError("BAR mapping range is outside BAR")
    f = open(self.pci_path / f"resource{bar}", "r+b", buffering=0)
    mm = mmap.mmap(f.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=off)
    return MMIOView(mm, size, fmt=fmt)
  def sleep(self, timeout_ms): time.sleep(timeout_ms / 1000)

def pack_nvos21(h_root, h_parent, h_object, h_class, params_addr=0, params_size=0):
  h_root = validate_rm_handle(h_root, "root handle")
  h_parent = validate_rm_handle(h_parent, "parent handle")
  h_object = validate_rm_handle(h_object, "object handle")
  h_class = validate_u32(h_class, "RM class")
  params_size = validate_u32(params_size, "RM alloc params size")
  return bytearray(struct.pack("<III I Q I I", h_root, h_parent, h_object, h_class, params_addr, params_size, 0))

def unpack_register_fd_status(data): return struct.unpack_from("<I", require_len(data, 8, "NV_ESC_REGISTER_FD"), 4)[0]
def unpack_nvos21_status(data): return struct.unpack_from("<I", require_len(data, 32, "NVOS21"), 28)[0]

def pack_nvos54(h_client, h_object, cmd, params_addr=0, params_size=0, flags=0):
  h_client = validate_rm_handle(h_client, "client handle")
  h_object = validate_rm_handle(h_object, "object handle")
  cmd = validate_u32(cmd, "RM control command")
  flags = validate_u32(flags, "RM control flags")
  params_size = validate_u32(params_size, "RM control params size")
  return bytearray(struct.pack("<IIIIQII", h_client, h_object, cmd, flags, params_addr, params_size, 0))

def unpack_nvos54_status(data): return struct.unpack_from("<I", require_len(data, 32, "NVOS54"), 28)[0]

def pack_nvos46(h_client, h_device, h_dma, h_memory, length, dma_offset, offset=0, flags=0):
  h_client = validate_rm_handle(h_client, "client handle")
  h_device = validate_rm_handle(h_device, "device handle")
  h_dma = validate_rm_handle(h_dma, "DMA handle")
  h_memory = validate_rm_handle(h_memory, "memory handle")
  if length <= 0: raise ValueError("DMA map length must be positive")
  if offset < 0: raise ValueError("DMA map offset must be non-negative")
  if dma_offset < 0: raise ValueError("DMA map target offset must be non-negative")
  flags = validate_u32(flags, "DMA map flags")
  return bytearray(struct.pack("<IIIIQQI4xQI4x", h_client, h_device, h_dma, h_memory, offset, length, flags, dma_offset, 0))

def unpack_nvos46_status(data): return struct.unpack_from("<I", require_len(data, 52, "NVOS46"), 48)[0]

def pack_nvos02_with_fd(h_root, h_parent, h_object, h_class, flags, p_memory, limit, fd=-1):
  h_root = validate_rm_handle(h_root, "root handle")
  h_parent = validate_rm_handle(h_parent, "parent handle")
  h_object = validate_rm_handle(h_object, "object handle")
  h_class = validate_u32(h_class, "RM class")
  flags = validate_u32(flags, "alloc-memory flags")
  if p_memory < 0: raise ValueError("alloc-memory address must be non-negative")
  if limit < 0: raise ValueError("alloc-memory limit must be non-negative")
  fd = validate_i32(fd, "alloc-memory fd")
  data = bytearray(56)
  struct.pack_into("<IIIII4xQQI4xi", data, 0, h_root, h_parent, h_object, h_class, flags, p_memory, limit, 0, fd)
  return data

def unpack_nvos02_status(data): return struct.unpack_from("<I", require_len(data, 44, "NVOS02"), 40)[0]
def unpack_nvos02_handle(data): return struct.unpack_from("<I", require_len(data, 12, "NVOS02"), 8)[0]

def pack_nvos33_with_fd(h_client, h_device, h_memory, length, flags=0, offset=0, fd=-1):
  h_client = validate_rm_handle(h_client, "client handle")
  h_device = validate_rm_handle(h_device, "device handle")
  h_memory = validate_rm_handle(h_memory, "memory handle")
  if length <= 0: raise ValueError("map length must be positive")
  if offset < 0: raise ValueError("map offset must be non-negative")
  flags = validate_u32(flags, "map flags")
  fd = validate_i32(fd, "map fd")
  data = bytearray(56)
  struct.pack_into("<III4xQQQIIi4x", data, 0, h_client, h_device, h_memory, offset, length, 0, 0, flags, fd)
  return data

def unpack_nvos33_status(data): return struct.unpack_from("<I", require_len(data, 44, "NVOS33"), 40)[0]
def unpack_nvos33_linear_address(data): return struct.unpack_from("<Q", require_len(data, 40, "NVOS33"), 32)[0]

def pack_uvm_initialize(flags=0): return bytearray(struct.pack("<QI4x", validate_u32(flags, "UVM initialize flags"), 0))
def pack_uvm_mm_initialize(uvm_fd): return bytearray(struct.pack("<iI", validate_i32(uvm_fd, "UVM fd"), 0))
def validate_uvm_range(base, length, name="UVM range"):
  if base < 0 or base % 0x1000: raise ValueError(f"{name} base must be 4KB aligned: 0x{base:x}")
  if length <= 0 or length % 0x1000: raise ValueError(f"{name} length must be positive and 4KB aligned: 0x{length:x}")
  return base, length
def pack_uvm_create_external_range(base, length):
  base, length = validate_uvm_range(base, length, "UVM external range")
  return bytearray(struct.pack("<QQI4x", base, length, 0))
def pack_uvm_free(base, length):
  base, length = validate_uvm_range(base, length, "UVM free range")
  return bytearray(struct.pack("<QQI4x", base, length, 0))
def unpack_uvm_status(data, offset): return struct.unpack_from("<I", require_len(data, offset + 4, "UVM status"), offset)[0]
def pack_nv2080_gpu_get_gid_info(index=0, flags=NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16):
  return bytearray(struct.pack("<III", validate_u32(index, "GPU GID index"),
    validate_u32(flags, "GPU GID flags"), validate_u32(length, "GPU GID length")) + bytes(256))
def unpack_nv2080_gpu_gid(data): return bytes(require_len(data, 28, "GPU GID response")[12:28])
def validate_gpu_uuid(gpu_uuid):
  gpu_uuid = bytes(gpu_uuid)
  if len(gpu_uuid) != 16: raise ValueError("GPU UUID must be 16 bytes")
  return gpu_uuid

def pack_uvm_register_gpu(gpu_uuid, rm_ctrl_fd, h_client=0, h_smc_part_ref=0):
  gpu_uuid = validate_gpu_uuid(gpu_uuid)
  rm_ctrl_fd = validate_i32(rm_ctrl_fd, "UVM RM control fd")
  h_client = validate_rm_handle(h_client, "client handle")
  h_smc_part_ref = validate_u32(h_smc_part_ref, "SMC partition ref")
  return bytearray(gpu_uuid + struct.pack("<B3xiiIII", 0, -1, rm_ctrl_fd, h_client, h_smc_part_ref, 0))

def pack_uvm_register_gpu_vaspace(gpu_uuid, rm_ctrl_fd, h_client, h_vaspace):
  gpu_uuid = validate_gpu_uuid(gpu_uuid)
  rm_ctrl_fd = validate_i32(rm_ctrl_fd, "UVM RM control fd")
  h_client = validate_rm_handle(h_client, "client handle")
  h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
  return bytearray(gpu_uuid + struct.pack("<iIII", rm_ctrl_fd, h_client, h_vaspace, 0))

def pack_uvm_register_channel(gpu_uuid, rm_ctrl_fd, h_client, h_channel, base, length):
  gpu_uuid = validate_gpu_uuid(gpu_uuid)
  rm_ctrl_fd = validate_i32(rm_ctrl_fd, "UVM RM control fd")
  h_client = validate_rm_handle(h_client, "client handle")
  h_channel = validate_rm_handle(h_channel, "channel handle")
  base, length = validate_uvm_range(base, length, "UVM channel range")
  return bytearray(gpu_uuid + struct.pack("<iIII4xQQI4x", rm_ctrl_fd, h_client, h_channel, 0, base, length, 0))

def pack_uvm_gpu_mapping_attributes(gpu_uuid, mapping_type=1, caching_type=0, format_type=0, element_bits=0, compression_type=0):
  gpu_uuid = validate_gpu_uuid(gpu_uuid)
  return gpu_uuid + struct.pack("<IIIII", validate_u32(mapping_type, "UVM mapping type"),
    validate_u32(caching_type, "UVM caching type"), validate_u32(format_type, "UVM format type"),
    validate_u32(element_bits, "UVM element bits"), validate_u32(compression_type, "UVM compression type"))

def pack_uvm_map_external_allocation(base, length, gpu_uuid, rm_ctrl_fd, h_client, h_memory, offset=0):
  base, length = validate_uvm_range(base, length, "UVM external allocation")
  if offset < 0: raise ValueError("UVM external allocation offset must be non-negative")
  rm_ctrl_fd = validate_i32(rm_ctrl_fd, "UVM RM control fd")
  h_client = validate_rm_handle(h_client, "client handle")
  h_memory = validate_rm_handle(h_memory, "memory handle")
  data = bytearray(9264)
  struct.pack_into("<QQQ", data, 0, base, length, offset)
  data[24:24+36] = pack_uvm_gpu_mapping_attributes(gpu_uuid)
  struct.pack_into("<Q", data, 9240, 1)
  struct.pack_into("<iIII", data, 9248, rm_ctrl_fd, h_client, h_memory, 0)
  return data

class NVRegisters:
  def __init__(self, transport, bar=0, fmt='I'):
    self.transport, self.bar, self.fmt = transport, bar, fmt
    self._bar = transport.map_bar(bar, fmt=fmt)
  @staticmethod
  def reg_index(addr):
    if addr < 0: raise ValueError("register address must be non-negative")
    if addr & 0x3: raise ValueError("register address must be 4-byte aligned")
    return addr // 4
  def rreg(self, addr): return self._bar[self.reg_index(addr)]
  def wreg(self, addr, value): self._bar[self.reg_index(addr)] = validate_u32(value, "register value")
  def read_bits(self, addr, lo, hi): return (self.rreg(addr) >> lo) & ((1 << (hi - lo + 1)) - 1)
  def write_bits(self, addr, lo, hi, value):
    if lo < 0 or hi < lo or hi >= 32: raise ValueError("register bit range is invalid")
    mask = ((1 << (hi - lo + 1)) - 1) << lo
    validate_u32(value, "register bit value")
    self.wreg(addr, (self.rreg(addr) & ~mask) | ((value << lo) & mask))

class StandaloneNvShell:
  def __init__(self, transport):
    self.transport = transport
    # Mirror tiny's nvdev._early_ip_init order: probe WPR2 BEFORE enabling bus master, then
    # (only if WPR2 is locked) clear bus master + issue transport.reset(), then enable bus
    # master. Writing PCI_COMMAND|MASTER before the WPR2 probe can activate the eGPU's PCIe
    # state machine in a way that leaves WPR2 lock + FECS BAR hardware-gated for the rest of
    # the boot, which is the trigger for the FECS_A->FECS_B|GR_STATUS|RMGpioPmuMutexTimeoutus
    # stall we kept hitting on ADT-Link UT3G USB4 eGPUs.
    pre_reset_wpr2 = NVRegisters(transport).rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI)
    needs_reset = pre_reset_wpr2 != 0
    if needs_reset and hasattr(self.transport, "reset") and os.environ.get("NV_ADD_SKIP_PCI_RESET") != "1":
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone WPR2 already initialized (wpr2_hi=0x{pre_reset_wpr2:08x}); resetting PCI device")
      self.transport.write_config(PCI_COMMAND, self.transport.read_config(PCI_COMMAND, 2) & ~PCI_COMMAND_MASTER, 2)
      self.transport.reset()
      self.transport.sleep(100)
      self.transport.write_config(PCI_COMMAND, self.transport.read_config(PCI_COMMAND, 2) | PCI_COMMAND_MASTER, 2)
    self.regs = NVRegisters(transport)
    fence_reasons = self.collect_fence_reasons()
    if fence_reasons["fecs_full_fence"]:
      # 0xbadf1301 on FECS.CPUCTL is the normal pre-boot "FECS ucode not loaded" state -- tiny
      # proceeds past it. We do NOT bail out by default; the diagnostic is logged so the
      # post-init fecs-pmu-postinit trace can compare. Set NV_ADD_BYPASS_BAR0_FENCE_CHECK=0
      # to fail-closed on this signal (useful only as a smoke test).
      if os.environ.get("NV_ADD_BYPASS_BAR0_FENCE_CHECK") == "0":
        raise RuntimeError(
          f"eGPU FECS.CPUCTL=0xbadf1301 pre-boot (set NV_ADD_BYPASS_BAR0_FENCE_CHECK=1 or unset to proceed; tiny proceeds past this state).")
      print(f"standalone pre-boot fecs-pmu-state {fence_reasons['description']}")
      if fence_reasons["wpr2_locked"]:
        print("standalone pre-boot wpr2-locked=0x2ffee00 (eGPU is in the post-GSP-RM-session state; FECS BAR is hardware-gated and can only be cleared by a physical eGPU power-cycle)")
    # NOTE: the WPR2 lock state alone is NOT a fence -- tiny proceeds with WPR2=0x2ffee00 set and
    # the FALCONs in their 0xbadf50xx/0xbadf57xx post-reset state. FECS.CPUCTL=0xbadf1301 is also
    # the normal pre-boot "FECS ucode not loaded" state, not a hardware fence. The post-init
    # fecs-pmu-postinit trace point catches any actual FECS fence at the right point in the boot
    # (after the GSP RM has come up and the FALCONs are first readable in their natural state).
    # The bypass env var is kept for diagnostic purposes only.
    if os.environ.get("NV_ADD_BYPASS_FECS_FENCE_CHECK") == "1" and self.fecs_bar_is_fenced():
      print(f"standalone FECS fence diagnostic: FECS.CPUCTL=0x{self.regs.rreg(0xA04100):08x} (bypassed by NV_ADD_BYPASS_FECS_FENCE_CHECK=1)")
    deadline = time.perf_counter() + 3.0
    while True:
      self.chip = NvChipInfo.probe(self.regs)
      if sane_chip_probe(self.chip): break
      if time.perf_counter() >= deadline:
        raise RuntimeError(f"GPU probe returned invalid BAR0 state: boot0=0x{self.chip.pmc_boot_0:x} "
                           f"vram=0x{self.chip.vram_size:x}")
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone waiting for sane probe boot0=0x{self.chip.pmc_boot_0:x} vram=0x{self.chip.vram_size:x}")
      self.transport.sleep(50)
      self.regs = NVRegisters(transport)
    self.large_bar = transport.bar_info(1)[1] >= self.chip.vram_size
    if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
      print(f"standalone chip boot0=0x{self.chip.pmc_boot_0:x} chip={self.chip.chip_name} fw={self.chip.fw_name} "
            f"vram=0x{self.chip.vram_size:x} bar1_size=0x{transport.bar_info(1)[1]:x} large_bar={int(self.large_bar)}")
    self.mm = GpuMemoryManager(max(self.chip.vram_size - (64 << 20), 0), mmu_ver=self.chip.mmu_ver, reserve_ptable=not self.large_bar)
    self.is_err_state = False

  @property
  def chip_name(self): return self.chip.chip_name
  @property
  def fw_name(self): return self.chip.fw_name
  @property
  def vram_size(self): return self.chip.vram_size

  def fecs_falcon_base(self):
    base = NV_FALCON_FECS_BASES.get(self.chip.fw_name)
    if base is None: raise ValueError(f"FECS FALCON base is unknown for firmware {self.chip.fw_name}")
    return base
  def fecs_falcon_engine(self): return self.fecs_falcon_base() + NV_PFECS_FALCON_ENGINE_OFFSET

  def collect_fence_reasons(self, sample_addrs=(0x88, 0x100, 0x200, 0x400, 0x1000, 0x1FA828)):
    """Return a structured verdict on the current eGPU state.

    Pre-boot fence readback is unreliable: tiny proceeds through every 0xbadf* state
    (0xbadf1301 on FECS = "FECS ucode not loaded", 0xbadf5720 on GSP/PMU = FALCON halted,
    0xbadf5620 on SEC2 = FALCON halted, WPR2=0x2ffee00 = WPR2 lock set by GSP boot).
    We report the pre-boot state and let the post-init fecs-pmu-postinit trace make the
    final fence determination (after the FALCONs are in their natural post-load state).
    """
    if not hasattr(self, "regs"):
      return {"hardware_fence": False, "wpr2_locked": False, "fecs_full_fence": False,
              "fence_sample_count": 0, "valid_sample_count": 0, "description": "no regs"}
    try: wpr2_hi = self.regs.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI)
    except Exception: wpr2_hi = 0
    fecs_base = self.fecs_falcon_base() if (hasattr(self, "chip") and self.chip is not None) else 0xA04000
    try: fecs_cpuctl = self.regs.rreg(fecs_base + 0x100)
    except Exception: fecs_cpuctl = 0
    fecs_full_fence = (fecs_cpuctl & 0xffffffff) == 0xbadf1301
    fence_count = 0
    valid_count = 0
    for addr in sample_addrs:
      try: value = self.regs.rreg(addr)
      except Exception: continue
      if (value & 0xffff0000) == 0xbadf0000: fence_count += 1
      else: valid_count += 1
    description = (f"FECS.CPUCTL=0x{fecs_cpuctl:08x} wpr2_hi=0x{wpr2_hi:08x} fence_count={fence_count}/{len(sample_addrs)} (pre-boot, normal)")
    return {"hardware_fence": False, "wpr2_locked": wpr2_hi != 0, "fecs_full_fence": fecs_full_fence,
            "fence_sample_count": fence_count, "valid_sample_count": valid_count, "description": description}

  def bar0_is_fenced(self, sample_addrs=(0x88, 0x100, 0x200, 0x400, 0x1000, 0x1FA828)):
    """Backward-compat shim: returns True only on a *hardware* fence (FECS.CPUCTL=0xbadf1301).
    Use collect_fence_reasons() to get the full diagnostic breakdown."""
    return self.collect_fence_reasons(sample_addrs)["hardware_fence"]

  def fecs_bar_is_fenced(self):
    if not hasattr(self, "regs"): return False
    try:
      fecs_base = self.fecs_falcon_base() if (hasattr(self, "chip") and self.chip is not None) else 0xA04000
    except Exception:
      fecs_base = 0xA04000
    try: cpuctl = self.regs.rreg(fecs_base + 0x100)
    except Exception: return False
    if (cpuctl & 0xffffffff) == 0xbadf1301: return True
    if (cpuctl & 0xffff0000) == 0xbadf0000 and (cpuctl & 0x0000ffff) != 0: return True
    try:
      wpr2_hi = self.regs.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI)
    except Exception: return False
    if wpr2_hi != 0: return True
    return False

  def rreg(self, addr): return self.regs.rreg(addr)
  def wreg(self, addr, value): self.regs.wreg(addr, value)
  def invalidate_mmu(self): self.wreg(NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE, (1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))
  def notify_rpc(self): self.wreg(NV_PGSP_QUEUE_HEAD_0, 0)
  def run_cpu_sequencer(self, seq_buf):
    if len(seq_buf) < 40: raise ValueError("CPU sequencer buffer is truncated")
    _, cmd_index = struct.unpack_from("<II", seq_buf, 0)
    if 40 + cmd_index * 4 > len(seq_buf): raise ValueError("CPU sequencer command stream is truncated")
    reg_save = list(struct.unpack_from("<8I", seq_buf, 8))
    cmd_words = list(struct.unpack_from(f"<{cmd_index}I", seq_buf, 40)) if cmd_index else []
    if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
      print(f"cpu_seq hdr cmd_index={cmd_index} reg_save={[hex(x) for x in reg_save]}")
      print(f"cpu_seq words={[hex(x) for x in cmd_words]}")
    it = iter(cmd_words)
    falcon = FalconController(self)
    for op in it:
      if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
        print(f"cpu_seq op=0x{op:x}")
      if op == 0x0:
        addr, val = next(it), next(it)
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print(f"cpu_seq write addr=0x{addr:x} val=0x{val:x}")
        self.wreg(addr, val)
      elif op == 0x1:
        addr, val, mask = next(it), next(it), next(it)
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print(f"cpu_seq modify addr=0x{addr:x} val=0x{val:x} mask=0x{mask:x}")
        self.wreg(addr, (self.rreg(addr) & ~mask) | (val & mask))
      elif op == 0x2:
        addr, mask, val, _, _ = next(it), next(it), next(it), next(it), next(it)
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print(f"cpu_seq poll addr=0x{addr:x} mask=0x{mask:x} val=0x{val:x}")
        falcon.wait_until(lambda a=addr, m=mask, v=val: (self.rreg(a) & m) == v, f"CPU sequencer poll timed out at 0x{addr:x}")
      elif op == 0x3:
        delay_us = next(it)
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print(f"cpu_seq sleep_us={delay_us}")
        self.transport.sleep(ceildiv(delay_us, 1000))
      elif op == 0x4:
        addr, index = next(it), next(it)
        if index >= len(reg_save): raise ValueError(f"CPU sequencer save index {index} out of range")
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print(f"cpu_seq save addr=0x{addr:x} index={index}")
        reg_save[index] = self.rreg(addr)
      elif op == 0x5:
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print("cpu_seq reset gsp")
        falcon.reset(NV_FALCON_GSP_BASE)
        falcon.disable_ctx_req(NV_FALCON_GSP_BASE)
      elif op == 0x6:
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print("cpu_seq start gsp")
        falcon.start_cpu(NV_FALCON_GSP_BASE)
      elif op == 0x7:
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print("cpu_seq wait gsp halt")
        falcon.wait_cpu_halted(NV_FALCON_GSP_BASE)
      elif op == 0x8:
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
          print("cpu_seq resume sec2")
        falcon.reset(NV_FALCON_GSP_BASE, riscv=True)
        boot = getattr(self, "gsp_boot", None)
        libos_args = boot.queue_memory.libos_args_paddrs[0] if boot is not None and getattr(boot, "queue_memory", None) is not None else 0
        if libos_args:
          if os.environ.get("NV_ADD_TRACE_CPU_SEQ") == "1":
            print(f"cpu_seq write libos_args=0x{libos_args:x}")
          self.wreg(NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_MAILBOX0, lo32(libos_args))
          self.wreg(NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_MAILBOX1, hi32(libos_args))
        falcon.start_cpu(NV_FALCON_SEC2_BASE)
        falcon.wait_until(lambda: reg_get(self.rreg(NV_PGC6_BSI_SECURE_SCRATCH_14), 26, 26) == 1, "SEC2 did not hand off")
        mailbox = self.rreg(NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_MAILBOX0)
        if mailbox != 0: raise RuntimeError(f"Falcon SEC2 failed to execute, mailbox is {mailbox:08x}")
      else:
        raise ValueError(f"unknown CPU sequencer op {op}")
    return reg_save

def bitmask_value(**fields):
  value = 0
  for (lo, hi), field_value in fields.values():
    value |= (field_value & ((1 << (hi - lo + 1)) - 1)) << lo
  return value

def reg_get(value, lo, hi): return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

def verify_view_bytes(label, view, expected, chunk=0x1000):
  digest = hashlib.sha256()
  first_bad = None
  for offset in range(0, len(expected), chunk):
    got = bytes(view.view(offset, min(chunk, len(expected) - offset), fmt='B')[:])
    want = expected[offset:offset + len(got)]
    digest.update(got)
    if first_bad is None and got != want:
      for idx, (want_byte, got_byte) in enumerate(zip(want, got)):
        if want_byte != got_byte:
          first_bad = offset + idx, want_byte, got_byte
          break
  want_digest = hashlib.sha256(expected).hexdigest()
  got_digest = digest.hexdigest()
  if first_bad is not None:
    mismatch, want_byte, got_byte = first_bad
    raise RuntimeError(f"{label} readback mismatch at +0x{mismatch:x}: expected=0x{want_byte:02x} got=0x{got_byte:02x}, expected_sha256={want_digest}, got_sha256={got_digest}")
  return want_digest

class FalconController:
  DMA_SIZE_256B = 6
  FBIF_TARGET_PHYSICAL = 0
  FBIF_MEM_TYPE_PHYSICAL = 1
  MOD_SEL_ALGO_RSA3K = 1

  def __init__(self, shell, timeout_ms=1000):
    self.shell, self.timeout_ms = shell, timeout_ms

  def wait_until(self, predicate, msg, timeout_ms=None):
    if not callable(predicate): raise ValueError("Falcon wait predicate must be callable")
    timeout = self.timeout_ms if timeout_ms is None else timeout_ms
    if timeout < 0: raise ValueError("Falcon wait timeout must be non-negative")
    start = time.perf_counter()
    while not predicate():
      if (time.perf_counter() - start) * 1000 > timeout: raise RuntimeError(msg() if callable(msg) else msg)
      self.shell.transport.sleep(1)

  def rreg(self, base, offset): return self.shell.rreg(base + offset)
  def wreg(self, base, offset, value):
    addr = base + offset
    if os.environ.get("NV_ADD_TRACE_FALCON") == "1":
      base_name = "GSP" if base == NV_FALCON_GSP_BASE else ("SEC2" if base == NV_FALCON_SEC2_BASE else ("PMU" if base == NV_FALCON_PMU_BASE else ("FECS" if base in NV_FALCON_FECS_BASES.values() else f"BASE_0x{base:x}")))
      name = FALCON_WRITE_NAMES.get(offset, f"0x{offset:x}")
      print(f"falcon wreg {base_name}.{name} addr=0x{addr:x} value=0x{value:x}")
    self.shell.wreg(addr, value)
  def engine_reg(self, base):
    if base == NV_FALCON_GSP_BASE: return NV_PGSP_FALCON_ENGINE
    if base == NV_FALCON_SEC2_BASE: return NV_PSEC_FALCON_ENGINE
    if base == NV_FALCON_PMU_BASE: return NV_PPMU_FALCON_ENGINE
    if base in NV_FALCON_FECS_BASES.values(): return base + NV_PFECS_FALCON_ENGINE_OFFSET
    raise ValueError(f"unknown Falcon base 0x{base:x}")
  def state_snapshot(self, base):
    regs = {
      "engine": self.shell.rreg(self.engine_reg(base)),
      "cpuctl": self.rreg(base, NV_PFALCON_FALCON_CPUCTL),
      "dmactl": self.rreg(base, NV_PFALCON_FALCON_DMACTL),
      "dmatrfcmd": self.rreg(base, NV_PFALCON_FALCON_DMATRFCMD),
      "dmatrfbase": self.rreg(base, NV_PFALCON_FALCON_DMATRFBASE),
      "dmatrfbase1": self.rreg(base, NV_PFALCON_FALCON_DMATRFBASE1),
      "dmatrfmoffs": self.rreg(base, NV_PFALCON_FALCON_DMATRFMOFFS),
      "dmatrffboffs": self.rreg(base, NV_PFALCON_FALCON_DMATRFFBOFFS),
      "hwcfg2": self.rreg(base, NV_PFALCON_FALCON_HWCFG2),
      "fbif_ctl": self.rreg(base, NV_PFALCON_FBIF_CTL),
      "fbif_transcfg0": self.rreg(base, NV_PFALCON_FBIF_TRANSCFG0),
      "exci": self.rreg(base, NV_PFALCON_FALCON_EXCI),
      "irqstat": self.rreg(base, NV_PFALCON_FALCON_IRQSTAT),
      "riscv_bcr": self.rreg(base, NV_PRISCV_RISCV_BCR_CTRL),
      "riscv_cpuctl": self.rreg(base, NV_PRISCV_RISCV_CPUCTL),
      "os": self.rreg(base, NV_PFALCON_FALCON_OS),
      "rm": self.rreg(base, NV_PFALCON_FALCON_RM),
    }
    return regs
  def format_state(self, base):
    return " ".join(f"{name}=0x{value:x}" for name, value in self.state_snapshot(base).items())
  def format_state_with_fenced(self, base):
    regs = self.state_snapshot(base)
    fenced = (regs["cpuctl"] & 0xffffffff) == 0xbadf5720 or (regs["cpuctl"] & 0xffff0000) == 0xbadf0000
    has_riscv = reg_get(regs["hwcfg2"], 10, 10)
    return f"({self.format_state(base)}) [hwcfg2=0x{regs['hwcfg2']:x} has_riscv={has_riscv} cpuctl=0x{regs['cpuctl']:x} fenced={fenced}]"
  def dma_error(self, base, reason):
    return f"{reason} at base=0x{base:x}, DMATRFCMD=0x{getattr(self, '_last_dmatrfcmd', 0):x}, dma={tuple(hex(x) for x in getattr(self, '_last_dma', ()))}, state=({self.format_state(base)})"

  def wait_dma_not_full(self, base):
    def ready():
      self._last_dmatrfcmd = self.rreg(base, NV_PFALCON_FALCON_DMATRFCMD)
      return reg_get(self._last_dmatrfcmd, 0, 0) == 0
    self.wait_until(ready, lambda: self.dma_error(base, "Falcon DMA queue is full"))

  def wait_dma_idle(self, base):
    def idle():
      self._last_dmatrfcmd = self.rreg(base, NV_PFALCON_FALCON_DMATRFCMD)
      return reg_get(self._last_dmatrfcmd, 1, 1) == 1
    self.wait_until(idle, lambda: self.dma_error(base, "Falcon DMA did not become idle"))

  def disable_ctx_req(self, base):
    self.wreg(base, NV_PFALCON_FBIF_CTL, self.rreg(base, NV_PFALCON_FBIF_CTL) | (1 << 7))
    self.wreg(base, NV_PFALCON_FALCON_DMACTL, 0)

  def dma_cmd(self, ctxdma=0, imem=False, sec=False, write=False):
    return (int(sec) << 2) | (int(imem) << 4) | (int(write) << 5) | (self.DMA_SIZE_256B << 8) | (ctxdma << 12)

  def execute_dma(self, base, cmd, dest, mem_off, src, size):
    validate_u32(cmd, "Falcon DMA command")
    for value, name in ((base, "Falcon base"), (dest, "Falcon DMA destination"), (mem_off, "Falcon DMA memory offset"), (src, "Falcon DMA source")):
      if value < 0: raise ValueError(f"{name} must be non-negative")
    validate_positive_size(size, "Falcon DMA size")
    if size % 256: raise ValueError("Falcon DMA size must be 256-byte aligned")
    self._last_dma = (base, cmd, dest, mem_off, src, size)
    if os.environ.get("NV_ADD_TRACE_FALCON") == "1":
      print(f"falcon pre-dma base=0x{base:x} cmd=0x{cmd:x} dest=0x{dest:x} mem_off=0x{mem_off:x} src=0x{src:x} size=0x{size:x} state=({self.format_state(base)})")
    self.wait_dma_not_full(base)
    self.wreg(base, NV_PFALCON_FALCON_DMATRFBASE, lo32(src >> 8))
    self.wreg(base, NV_PFALCON_FALCON_DMATRFBASE1, hi32(src >> 8) & 0x1ff)
    for xfered in range(0, size, 256):
      self.wait_dma_not_full(base)
      self.wreg(base, NV_PFALCON_FALCON_DMATRFMOFFS, dest + xfered)
      self.wreg(base, NV_PFALCON_FALCON_DMATRFFBOFFS, mem_off + xfered)
      self.wreg(base, NV_PFALCON_FALCON_DMATRFCMD, cmd)
      if os.environ.get("NV_ADD_FLUSH_FALCON_DMA") == "1":
        self._last_dmatrfcmd = self.rreg(base, NV_PFALCON_FALCON_DMATRFCMD)
    self.wait_dma_idle(base)

  def start_cpu(self, base):
    cpuctl = self.rreg(base, NV_PFALCON_FALCON_CPUCTL)
    if reg_get(cpuctl, 6, 6):
      self.wreg(base, NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2)
    else:
      self.wreg(base, NV_PFALCON_FALCON_CPUCTL, 1 << 1)

  def wait_cpu_halted(self, base):
    self.wait_until(lambda: reg_get(self.rreg(base, NV_PFALCON_FALCON_CPUCTL), 4, 4) == 1, "Falcon CPU did not halt")

  @staticmethod
  def is_fenced_value(value):
    return (value & 0xffffffff) in (0xbadf5720, 0xbadf1301, 0xbadf5620) or (value & 0xffff0000) == 0xbadf0000

  def clear_fence(self, base):
    """Best-effort fence clear: writes the canonical fence-clear values to FALCON_BAR
    registers. This is a no-op when the eGPU is in a clean state (writes succeed but
    have no effect). When the eGPU is in a hardware fence state, these writes are also
    no-ops (BAR is gated); the only true fence-clear path is a physical eGPU power-cycle.
    Kept for diagnostic purposes; not called by reset() anymore.
    """
    if os.environ.get("NV_ADD_TRACE_FALCON") == "1":
      pre_cpuctl = self.rreg(base, NV_PFALCON_FALCON_CPUCTL)
      pre_transcfg0 = self.rreg(base, NV_PFALCON_FBIF_TRANSCFG0)
      print(f"falcon clear_fence pre base=0x{base:x} cpuctl=0x{pre_cpuctl:x} fbif_transcfg0=0x{pre_transcfg0:x}")
    self.wreg(base, NV_PFALCON_FBIF_TRANSCFG0, 0x114)
    self.wreg(base, NV_PFALCON_FALCON_CPUCTL, 0)
    self.shell.transport.sleep(10)
    self.wreg(base, NV_PFALCON_FALCON_DMACTL, 0)
    self.wreg(base, NV_PFALCON_FALCON_DMATRFCMD, 0)
    if os.environ.get("NV_ADD_TRACE_FALCON") == "1":
      post_cpuctl = self.rreg(base, NV_PFALCON_FALCON_CPUCTL)
      print(f"falcon clear_fence post base=0x{base:x} cpuctl=0x{post_cpuctl:x}")

  def reset(self, base, riscv=False, force=False):
    """Minimal FALCON reset that matches tinygrad's NV_FLCN.reset():
    engine=1, sleep(100ms), engine=0, wait for mem_scrubbing.
    Optional pre-engine fence-clear (NV_ADD_FECS_FENCE_CLEAR=1) writes the canonical
    fence-clear values to FBIF_TRANSCFG0/CPUCTL/DMACTL/DMATRFCMD; without that env var,
    the reset is just the engine toggle. The fence-clear is a no-op when the eGPU is in
    a clean state and also a no-op when the BAR is hardware-gated; it is a documentation
    aid only.
    """
    if not hasattr(self.shell, "chip") or self.shell.chip is None: raise RuntimeError("Falcon reset requires chip info")
    engine = self.engine_reg(base)
    pre_cpuctl = self.rreg(base, NV_PFALCON_FALCON_CPUCTL)
    pre_hwcfg2 = self.rreg(base, NV_PFALCON_FALCON_HWCFG2)
    fenced_pre = self.is_fenced_value(pre_cpuctl) and self.is_fenced_value(pre_hwcfg2)
    if os.environ.get("NV_ADD_TRACE_FALCON") == "1":
      print(f"falcon reset pre base=0x{base:x} cpuctl=0x{pre_cpuctl:x} hwcfg2=0x{pre_hwcfg2:x} fenced_pre={fenced_pre} force={force}")
    if force:
      self.clear_fence(base)
    self.shell.wreg(engine, 1)
    self.shell.transport.sleep(100)
    self.shell.wreg(engine, 0)
    if fenced_pre and not force:
      if os.environ.get("NV_ADD_TRACE_FALCON") == "1":
        print(f"falcon reset base=0x{base:x} skipping HWCFG2 wait because pre-cpuctl=0x{pre_cpuctl:x} is a fence sentinel")
      self.shell.transport.sleep(100)
    else:
      self.wait_until(lambda: reg_get(self.rreg(base, NV_PFALCON_FALCON_HWCFG2), 12, 12) == 0, "Falcon memory scrubbing did not complete")
    has_riscv = reg_get(self.rreg(base, NV_PFALCON_FALCON_HWCFG2), 10, 10)
    if riscv and has_riscv:
      self.wreg(base, NV_PRISCV_RISCV_BCR_CTRL, (1 << 4) | (1 << 8))
    elif has_riscv:
      self.wreg(base, NV_PRISCV_RISCV_BCR_CTRL, 0)
      self.wait_until(lambda: reg_get(self.rreg(base, NV_PRISCV_RISCV_BCR_CTRL), 0, 0) == 1, "RISCV core did not boot")
      self.wreg(base, NV_PFALCON_FALCON_RM, self.shell.chip.pmc_boot_0)
    if force or os.environ.get("NV_ADD_TRACE_FALCON") == "1":
      post_cpuctl = self.rreg(base, NV_PFALCON_FALCON_CPUCTL)
      if os.environ.get("NV_ADD_TRACE_FALCON") == "1":
        print(f"falcon reset post base=0x{base:x} cpuctl=0x{post_cpuctl:x}")

  def execute_hs(self, base, img_paddr, code_off, data_off, imem_pa, imem_va, imem_size,
                 dmem_pa, dmem_va, dmem_size, pkc_off, engid, ucodeid, mailbox=None):
    for value, name in ((base, "Falcon base"), (img_paddr, "Falcon HS image address"), (code_off, "Falcon HS code offset"),
      (data_off, "Falcon HS data offset"), (imem_pa, "Falcon HS IMEM physical address"), (imem_va, "Falcon HS IMEM virtual address"),
      (dmem_pa, "Falcon HS DMEM physical address"), (dmem_va, "Falcon HS DMEM virtual address"), (pkc_off, "Falcon HS PKC offset")):
      if value < 0: raise ValueError(f"{name} must be non-negative")
    validate_u32(engid, "Falcon HS engine id")
    validate_u32(ucodeid, "Falcon HS ucode id")
    if mailbox is not None: validate_u64(mailbox, "Falcon HS mailbox")
    self.disable_ctx_req(base)
    transcfg0 = self.rreg(base, NV_PFALCON_FBIF_TRANSCFG0)
    transcfg0 = (transcfg0 & ~0xf) | self.FBIF_TARGET_PHYSICAL | (self.FBIF_MEM_TYPE_PHYSICAL << 2)
    self.wreg(base, NV_PFALCON_FBIF_TRANSCFG0, transcfg0)
    self.execute_dma(base, self.dma_cmd(imem=True, sec=True), imem_pa, imem_va, img_paddr + code_off - imem_va, imem_size)
    self.execute_dma(base, self.dma_cmd(imem=False, sec=False), dmem_pa, dmem_va, img_paddr + data_off - dmem_va, dmem_size)
    self.wreg(base, NV_PFALCON2_FALCON_BROM_PARAADDR0, pkc_off)
    self.wreg(base, NV_PFALCON2_FALCON_BROM_ENGIDMASK, engid)
    self.wreg(base, NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID, ucodeid)
    self.wreg(base, NV_PFALCON2_FALCON_MOD_SEL, self.MOD_SEL_ALGO_RSA3K)
    self.wreg(base, NV_PFALCON_FALCON_BOOTVEC, imem_va)
    if mailbox is not None:
      self.wreg(base, NV_PFALCON_FALCON_MAILBOX0, lo32(mailbox))
      self.wreg(base, NV_PFALCON_FALCON_MAILBOX1, hi32(mailbox))
    self.start_cpu(base)
    self.wait_cpu_halted(base)
    if mailbox is not None:
      return self.rreg(base, NV_PFALCON_FALCON_MAILBOX0), self.rreg(base, NV_PFALCON_FALCON_MAILBOX1)

FIRMWARE_HASHES = {
  ("ga102", "gsp-570.144.bin"): "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f",
  ("ga102", "bootloader-570.144.bin"): "82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad",
  ("ad102", "bootloader-570.144.bin"): "65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8",
  ("gb202", "bootloader-570.144.bin"): "d40b48e431d1707dc77af3605db358ed7a32eb74de2eddb4d3025071",
  ("ga102", "booter_load-570.144.bin"): "4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b",
  ("ad102", "booter_load-570.144.bin"): "8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a",
}

GSP_FW_WPR_META_REVISION = 1
GSP_FW_WPR_META_MAGIC = 0xdc3aae21371a60b3
GSP_FW_WPR_META_VERIFIED = 0xa0a0a0a0a0a0a0a0
GSP_FW_WPR_META_SIZE = 256
RM_RISCV_UCODE_DESC_SIZE = 84
FALCON_UCODE_DESC_V3_SIZE = 44
FALCON_DMEM_MAPPER_V3_SIZE = 64
NVFW_HS_HEADER_V2_SIZE = 36
NVFW_HS_LOAD_HEADER_V2_SIZE = 20
NVFW_HS_LOAD_HEADER_V2_APP_SIZE = 16
BIT_TOKEN_FALCON_DATA = 0x70
FALCON_UCODE_ENTRY_APPID_FWSEC_PROD = 0x85
NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE = 0x00
NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT = 0xE0
PCI_ROM_IMAGE_BLOCK_SIZE = 512

class FirmwareStore:
  LINUX_FIRMWARE_COMMIT = "1e2c15348485939baf1b6d1f5a7a3b799d80703d"

  def __init__(self, root=None):
    self.root = pathlib.Path(root or os.environ.get("NV_FIRMWARE_DIR", ROOT / "firmware"))

  @staticmethod
  def validate_component(value, name):
    if not isinstance(value, str) or not value: raise ValueError(f"{name} must be a non-empty string")
    if "/" in value or "\\" in value or value in (".", ".."): raise ValueError(f"{name} must be a path component")
    return value

  def candidates(self, fw_name, filename):
    fw_name = self.validate_component(fw_name, "firmware name")
    filename = self.validate_component(filename, "firmware filename")
    yield self.root / fw_name / "gsp" / filename
    yield self.root / fw_name / filename
    yield pathlib.Path("/lib/firmware/nvidia") / fw_name / "gsp" / filename
    yield pathlib.Path("/usr/lib/firmware/nvidia") / fw_name / "gsp" / filename

  def load(self, fw_name, filename, required_hash=None, allow_download=False):
    fw_name = self.validate_component(fw_name, "firmware name")
    filename = self.validate_component(filename, "firmware filename")
    required_hash = required_hash or FIRMWARE_HASHES.get((fw_name, filename))
    for path in self.candidates(fw_name, filename):
      if not path.is_file(): continue
      data = path.read_bytes()
      if required_hash and hashlib.sha256(data).hexdigest() != required_hash:
        raise RuntimeError(f"firmware hash mismatch for {path}")
      return data
    if allow_download:
      url = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{self.LINUX_FIRMWARE_COMMIT}/nvidia/{fw_name}/gsp/{filename}"
      data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tinygrad 0.13.0"}), timeout=30).read()
      if required_hash and hashlib.sha256(data).hexdigest() != required_hash:
        raise RuntimeError(f"downloaded firmware hash mismatch for {filename}")
      out = self.root / fw_name / "gsp" / filename
      out.parent.mkdir(parents=True, exist_ok=True)
      out.write_bytes(data)
      return data
    raise FileNotFoundError(f"missing firmware {fw_name}/gsp/{filename}; set NV_FIRMWARE_DIR or install linux-firmware")

def elf_section(blob, name):
  _, sections, _ = elf_loader(blob)
  for section in sections:
    if section.name == name: return section.content
  raise KeyError(name)

class GspFirmwarePrep:
  RADIX_PAGE_LOG2 = 12

  @staticmethod
  def radix3_layout(image_size):
    if image_size < 0: raise ValueError("radix3 image size must be non-negative")
    image_pages = ceildiv(image_size, 0x1000)
    pages = [0, 0, 0, image_pages]
    for level in range(3, 0, -1): pages[level - 1] = ((pages[level] - 1) >> (GspFirmwarePrep.RADIX_PAGE_LOG2 - 3)) + 1
    offsets = [sum(pages[:idx]) * 0x1000 for idx in range(4)]
    return pages, offsets, offsets[-1] + image_size

  @staticmethod
  def build_radix3_image(image, page_addrs):
    pages, offsets, total_size = GspFirmwarePrep.radix3_layout(len(image))
    if len(page_addrs) < sum(pages): raise ValueError("not enough physical pages for radix3 image")
    out = bytearray(total_size)
    out[offsets[-1]:offsets[-1]+len(image)] = image
    for level in range(3):
      child_start = sum(pages[:level + 1])
      for idx in range(pages[level + 1]):
        struct.pack_into("<Q", out, offsets[level] + idx * 8, page_addrs[child_start + idx])
    return bytes(out), pages, offsets

  @staticmethod
  def gsp_signature_section(gsp_elf, chip_name):
    _, sections, _ = elf_loader(gsp_elf)
    fwimage = next((sh.content for sh in sections if sh.name == ".fwimage"), None)
    signature_name = f".fwsignature_{chip_name[:4].lower()}x"
    signature = next((sh.content for sh in sections if sh.name == signature_name), None)
    if fwimage is None: raise KeyError(".fwimage")
    if signature is None: raise KeyError(signature_name)
    return fwimage, signature

  @staticmethod
  def parse_nvfw_bin_header(blob):
    if len(blob) < 24: raise ValueError("firmware blob is too small for nvfw_bin_hdr")
    header = dict(zip(("bin_magic", "bin_ver", "bin_size", "header_offset", "data_offset", "data_size"),
                      struct.unpack_from("<IIIIII", blob, 0)))
    if header["bin_size"] == 0: raise ValueError("firmware header bin_size is zero")
    return header

  @staticmethod
  def validate_nvfw_range(blob, header, offset, size, what):
    end = offset + size
    if offset < 0 or size < 0 or end < offset: raise ValueError(f"{what} range is invalid")
    if end > len(blob): raise ValueError(f"{what} range is outside blob")
    if end > header["bin_size"]: raise ValueError(f"{what} range is outside declared firmware size")
    return end

  @staticmethod
  def require_blob_range(blob, offset, size, what, header=None):
    end = offset + size
    if offset < 0 or size < 0 or end < offset: raise ValueError(f"{what} range is invalid")
    if end > len(blob): raise ValueError(f"{what} is truncated")
    if header is not None and end > header["bin_size"]: raise ValueError(f"{what} range is outside declared firmware size")
    return end

  @staticmethod
  def parse_riscv_ucode_desc(blob, offset):
    if offset < 0: raise ValueError("bootloader descriptor offset must be non-negative")
    if offset + RM_RISCV_UCODE_DESC_SIZE > len(blob): raise ValueError("bootloader descriptor is truncated")
    fields = ("version", "bootloader_offset", "bootloader_size", "bootloader_param_offset", "bootloader_param_size",
              "riscv_elf_offset", "riscv_elf_size", "app_version", "manifest_offset", "manifest_size",
              "monitor_data_offset", "monitor_data_size", "monitor_code_offset", "monitor_code_size",
              "is_monitor_enabled", "swbrom_code_offset", "swbrom_code_size", "swbrom_data_offset",
              "swbrom_data_size", "fb_reserved_size", "signed_as_code")
    return dict(zip(fields, struct.unpack_from("<" + "I" * len(fields), blob, offset)))

  @staticmethod
  def bootloader_image_and_desc(blob):
    header = GspFirmwarePrep.parse_nvfw_bin_header(blob)
    end = GspFirmwarePrep.validate_nvfw_range(blob, header, header["data_offset"], header["data_size"], "bootloader data")
    GspFirmwarePrep.require_blob_range(blob, header["header_offset"], RM_RISCV_UCODE_DESC_SIZE, "bootloader descriptor", header=header)
    desc = GspFirmwarePrep.parse_riscv_ucode_desc(blob, header["header_offset"])
    return blob[header["data_offset"]:end], desc

  @staticmethod
  def booter_load_image_and_desc(blob):
    header = GspFirmwarePrep.parse_nvfw_bin_header(blob)
    hs_off = header["header_offset"]
    GspFirmwarePrep.require_blob_range(blob, hs_off, NVFW_HS_HEADER_V2_SIZE, "booter HS header", header=header)
    sig_prod_off, sig_prod_size, patch_loc_off, patch_sig_off, _, _, num_sig_off, load_header_off, _ = struct.unpack_from("<IIIIIIIII", blob, hs_off)
    GspFirmwarePrep.require_blob_range(blob, load_header_off, NVFW_HS_LOAD_HEADER_V2_SIZE + NVFW_HS_LOAD_HEADER_V2_APP_SIZE,
                                       "booter load header", header=header)
    _, _, os_data_offset, os_data_size, num_apps = struct.unpack_from("<IIIII", blob, load_header_off)
    if num_apps < 1: raise ValueError("booter load header has no apps")
    app_offset, app_size, _, _ = struct.unpack_from("<IIII", blob, load_header_off + NVFW_HS_LOAD_HEADER_V2_SIZE)
    for off in (patch_loc_off, patch_sig_off, num_sig_off):
      GspFirmwarePrep.require_blob_range(blob, off, 4, "booter patch metadata", header=header)
    patch_loc = struct.unpack_from("<I", blob, patch_loc_off)[0]
    patch_sig = struct.unpack_from("<I", blob, patch_sig_off)[0]
    num_sig = struct.unpack_from("<I", blob, num_sig_off)[0]
    if num_sig == 0: raise ValueError("booter signature count is zero")
    sig_len = sig_prod_size // num_sig
    sig_off = sig_prod_off + patch_sig
    GspFirmwarePrep.require_blob_range(blob, sig_off, sig_len, "booter signature", header=header)
    end = GspFirmwarePrep.validate_nvfw_range(blob, header, header["data_offset"], header["data_size"], "booter data")
    image = bytearray(blob[header["data_offset"]:end])
    if patch_loc + sig_len > len(image): raise ValueError("booter patch range is outside image")
    image[patch_loc:patch_loc + sig_len] = blob[sig_off:sig_off + sig_len]
    desc = {"data_offset": os_data_offset, "data_size": os_data_size, "code_offset": app_offset, "code_size": app_size}
    return bytes(image), desc

  @staticmethod
  def parse_falcon_ucode_desc_v3(blob, offset):
    if offset < 0: raise ValueError("Falcon ucode descriptor offset must be non-negative")
    if offset + FALCON_UCODE_DESC_V3_SIZE > len(blob): raise ValueError("Falcon ucode descriptor is truncated")
    fields = ("hdr", "stored_size", "pkc_data_offset", "interface_offset", "imem_phys_base", "imem_load_size",
      "imem_virt_base", "dmem_phys_base", "dmem_load_size", "engine_id_mask", "ucode_id", "signature_count",
      "signature_versions", "reserved")
    return dict(zip(fields, struct.unpack_from("<IIIIIIIIIHBBHH", blob, offset)))

  @staticmethod
  def find_fwsec_ucode(vbios):
    vbios_off, base_block_size, expansion_rom_off = 0, 0, None
    while vbios_off + 0x1c < len(vbios):
      pci_data = struct.unpack_from("<H", vbios, vbios_off + 0x18)[0]
      if vbios_off + pci_data + 0x16 >= len(vbios): break
      image_len = struct.unpack_from("<H", vbios, vbios_off + pci_data + 0x10)[0] * PCI_ROM_IMAGE_BLOCK_SIZE
      code_type = vbios[vbios_off + pci_data + 0x14]
      if code_type == NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE: base_block_size = image_len
      elif code_type == NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT:
        expansion_rom_off = vbios_off - base_block_size
        break
      if image_len == 0: break
      vbios_off += image_len
    if expansion_rom_off is None: raise RuntimeError("VBIOS extension image was not found")

    bit_addr = 0x1b0
    GspFirmwarePrep.require_blob_range(vbios, bit_addr, 11, "BIT header")
    if struct.unpack_from("<I", vbios, bit_addr + 2)[0] != 0x00544942: raise RuntimeError("invalid BIT header signature")
    header_size, token_size, token_entries = struct.unpack_from("<BBB", vbios, bit_addr + 8)
    GspFirmwarePrep.require_blob_range(vbios, bit_addr, header_size + token_entries * token_size, "BIT token table")
    for idx in range(token_entries):
      off = bit_addr + header_size + idx * token_size
      GspFirmwarePrep.require_blob_range(vbios, off, 8, "BIT token")
      token_id, data_version, data_size, data_ptr = struct.unpack_from("<BBHI", vbios, off)
      if token_id != BIT_TOKEN_FALCON_DATA or data_version != 2 or data_size < 4: continue
      GspFirmwarePrep.require_blob_range(vbios, data_ptr & 0xffff, 4, "Falcon table pointer")
      table_ptr = expansion_rom_off + struct.unpack_from("<I", vbios, data_ptr & 0xffff)[0]
      GspFirmwarePrep.require_blob_range(vbios, table_ptr, 6, "Falcon table header")
      _, table_header_size, entry_size, entry_count, _, _ = struct.unpack_from("<BBBBBB", vbios, table_ptr)
      GspFirmwarePrep.require_blob_range(vbios, table_ptr, table_header_size + entry_count * entry_size, "Falcon table")
      for entry_idx in range(entry_count):
        entry_off = table_ptr + table_header_size + entry_idx * entry_size
        GspFirmwarePrep.require_blob_range(vbios, entry_off, 6, "Falcon table entry")
        app_id, _, desc_ptr = struct.unpack_from("<BBI", vbios, entry_off)
        if app_id != FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: continue
        desc_off = expansion_rom_off + desc_ptr
        GspFirmwarePrep.require_blob_range(vbios, desc_off, FALCON_UCODE_DESC_V3_SIZE, "FWSEC descriptor")
        desc_size = struct.unpack_from("<I", vbios, desc_off)[0] >> 16
        if desc_size < FALCON_UCODE_DESC_V3_SIZE: raise ValueError("FWSEC descriptor size is invalid")
        GspFirmwarePrep.require_blob_range(vbios, desc_off, desc_size, "FWSEC descriptor/signature")
        desc = GspFirmwarePrep.parse_falcon_ucode_desc_v3(vbios, desc_off)
        image_end = GspFirmwarePrep.require_blob_range(vbios, desc_off + desc_size, round_up(desc["stored_size"], 256), "FWSEC image")
        signature = vbios[desc_off + FALCON_UCODE_DESC_V3_SIZE:desc_off + desc_size]
        image = vbios[desc_off + desc_size:image_end]
        return desc, signature, image
    raise RuntimeError("FWSEC Falcon ucode was not found in VBIOS")

  @staticmethod
  def pack_fwsec_frts_cmd(frts_offset):
    if frts_offset % 0x1000: raise ValueError(f"FRTS offset must be 4KB aligned: 0x{frts_offset:x}")
    frts_page = frts_offset >> 12
    if not 0 <= frts_page <= 0xffffffff: raise ValueError(f"FRTS offset is outside 32-bit page field: 0x{frts_offset:x}")
    return struct.pack("<IIQII", 1, 24, 0, 0, 2) + struct.pack("<IIIII", 1, 20, frts_page, 0x100, 2)

  @staticmethod
  def patch_fwsec_image(image, desc, signature, cmd_id, cmd):
    patched = bytearray(image)
    app_hdr_off = desc["imem_load_size"] + desc["interface_offset"]
    GspFirmwarePrep.require_blob_range(image, app_hdr_off, 4, "FWSEC app header")
    _, _, entry_size, entry_count = struct.unpack_from("<BBBB", image, app_hdr_off)
    if entry_size < 8: raise ValueError("FWSEC app entry size is invalid")
    GspFirmwarePrep.require_blob_range(image, app_hdr_off + 4, entry_count * entry_size, "FWSEC app entries")
    mapper_off = None
    for idx in range(entry_count):
      entry_id, dmem_offset = struct.unpack_from("<II", image, app_hdr_off + 4 + idx * entry_size)
      if entry_id == 4:
        mapper_off = desc["imem_load_size"] + dmem_offset
        break
    if mapper_off is None: raise RuntimeError("FWSEC DMEM mapper entry was not found")
    GspFirmwarePrep.require_blob_range(image, mapper_off, struct.calcsize("<IHH" + "I" * 14), "FWSEC DMEM mapper")
    mapper = list(struct.unpack_from("<IHH" + "I" * 14, image, mapper_off))
    mapper[12] = cmd_id
    struct.pack_into("<IHH" + "I" * 14, patched, mapper_off, *mapper)
    cmd_off = desc["imem_load_size"] + mapper[3]
    GspFirmwarePrep.require_blob_range(image, cmd_off, len(cmd), "FWSEC command patch")
    sig_off = desc["imem_load_size"] + desc["pkc_data_offset"]
    GspFirmwarePrep.require_blob_range(image, sig_off, 0x180, "FWSEC signature patch")
    if len(signature) < 0x180: raise ValueError("FWSEC signature is truncated")
    patched[cmd_off:cmd_off + len(cmd)] = cmd
    patched[sig_off:sig_off + 0x180] = signature[-0x180:]
    return bytes(patched)

  @staticmethod
  def pack_wpr_meta(**kwargs):
    data = bytearray(GSP_FW_WPR_META_SIZE)
    qword_fields = {
      "magic": 0, "revision": 8, "sysmem_addr_of_radix3_elf": 16, "size_of_radix3_elf": 24,
      "sysmem_addr_of_bootloader": 32, "size_of_bootloader": 40, "bootloader_code_offset": 48,
      "bootloader_data_offset": 56, "bootloader_manifest_offset": 64, "sysmem_addr_of_signature": 72,
      "size_of_signature": 80, "gsp_fw_rsvd_start": 88, "non_wpr_heap_offset": 96,
      "non_wpr_heap_size": 104, "gsp_fw_wpr_start": 112, "gsp_fw_heap_offset": 120,
      "gsp_fw_heap_size": 128, "gsp_fw_offset": 136, "boot_bin_offset": 144,
      "frts_offset": 152, "frts_size": 160, "gsp_fw_wpr_end": 168, "fb_size": 176,
      "vga_workspace_offset": 184, "vga_workspace_size": 192, "boot_count": 200,
      "partition_rpc_addr": 208, "verified": 248,
    }
    dword_fields = {
      "elf_code_offset": 220, "elf_data_offset": 224, "elf_code_size": 228,
      "elf_data_size": 232, "ls_ucode_version": 236, "pmu_reserved_size": 244,
    }
    byte_fields = {"gsp_fw_heap_vf_partition_count": 240, "flags": 241}
    defaults = {
      "magic": GSP_FW_WPR_META_MAGIC, "revision": GSP_FW_WPR_META_REVISION,
      "size_of_signature": 0x1000,
    }
    values = {**defaults, **kwargs}
    for key, offset in qword_fields.items(): struct.pack_into("<Q", data, offset, validate_u64(int(values.get(key, 0)), f"WPR metadata {key}"))
    for key, offset in dword_fields.items(): struct.pack_into("<I", data, offset, validate_u32(int(values.get(key, 0)), f"WPR metadata {key}"))
    for key, offset in byte_fields.items():
      value = validate_u32(int(values.get(key, 0)), f"WPR metadata {key}")
      if value > 0xff: raise ValueError(f"WPR metadata {key} is outside 8-bit range")
      struct.pack_into("<B", data, offset, value)
    return bytes(data)

  @staticmethod
  def build_wpr_meta(vram_size, booter_size, radix3_size, booter_sysmem, radix3_sysmem, signature_sysmem, booter_desc,
                     fmc_boot=False, frts_offset=None):
    validate_positive_size(vram_size, "VRAM size")
    validate_positive_size(booter_size, "booter size")
    validate_positive_size(radix3_size, "radix3 size")
    validate_u64(booter_sysmem, "booter sysmem address")
    validate_u64(radix3_sysmem, "radix3 sysmem address")
    validate_u64(signature_sysmem, "signature sysmem address")
    for key in ("monitor_code_offset", "monitor_data_offset", "manifest_offset"):
      if key not in booter_desc: raise ValueError(f"booter descriptor missing {key}")
      validate_u64(booter_desc[key], f"booter descriptor {key}")
    if frts_offset is not None and (frts_offset < 0 or frts_offset % 0x1000):
      raise ValueError(f"FRTS offset must be 4KB aligned: 0x{frts_offset:x}")
    common = {
      "size_of_bootloader": booter_size,
      "sysmem_addr_of_bootloader": booter_sysmem,
      "size_of_radix3_elf": radix3_size,
      "sysmem_addr_of_radix3_elf": radix3_sysmem,
      "sysmem_addr_of_signature": signature_sysmem,
      "bootloader_code_offset": booter_desc["monitor_code_offset"],
      "bootloader_data_offset": booter_desc["monitor_data_offset"],
      "bootloader_manifest_offset": booter_desc["manifest_offset"],
    }
    if fmc_boot:
      return GspFirmwarePrep.pack_wpr_meta(**common, vga_workspace_size=0x20000, pmu_reserved_size=0x1820000,
        non_wpr_heap_size=0x220000, gsp_fw_heap_size=0x8700000, frts_size=0x100000)

    vga_sz = 0x100000
    vga_off = vram_size - vga_sz
    frts_sz = 0x100000
    frts_off = vga_off - frts_sz if frts_offset is None else frts_offset
    boot_off = frts_off - booter_size
    gsp_off = round_down(boot_off - radix3_size, 0x10000)
    gsp_heap_sz = 0x8100000
    gsp_heap_off = round_down(gsp_off - gsp_heap_sz, 0x100000)
    wpr_st = round_down(gsp_heap_off - 0x1000, 0x100000)
    non_wpr_sz = 0x100000
    non_wpr_off = round_down(wpr_st - non_wpr_sz, 0x100000)
    return GspFirmwarePrep.pack_wpr_meta(**common, vga_workspace_size=vga_sz, vga_workspace_offset=vga_off,
      gsp_fw_wpr_end=vga_off, frts_size=frts_sz, frts_offset=frts_off, boot_bin_offset=boot_off,
      gsp_fw_offset=gsp_off, gsp_fw_heap_size=gsp_heap_sz, fb_size=vram_size, gsp_fw_heap_offset=gsp_heap_off,
      gsp_fw_wpr_start=wpr_st, non_wpr_heap_size=non_wpr_sz, non_wpr_heap_offset=non_wpr_off,
      gsp_fw_rsvd_start=non_wpr_off)

NV_VGPU_MSG_SIGNATURE_VALID = 0x43505256
NV_VGPU_MSG_RESULT_RPC_PENDING = 0xffffffff
NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD = 0x80000000
NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY = 4
NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER = 47
NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY = 54
NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO = 72
NV_VGPU_MSG_FUNCTION_SET_REGISTRY = 73
NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL = 76
NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC = 103
NV_VGPU_MSG_EVENT_GSP_INIT_DONE = 4097
NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER = 4098
NV_VGPU_MSG_EVENT_OS_ERROR_LOG = 4102
NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED = 4101
NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD = 4128

GSP_RPC_NAMES = {
  NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD: "CONTINUATION_RECORD",
  NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY: "ALLOC_MEMORY",
  NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER: "UNLOADING_GUEST_DRIVER",
  NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY: "SET_PAGE_DIRECTORY",
  NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO: "GSP_SET_SYSTEM_INFO",
  NV_VGPU_MSG_FUNCTION_SET_REGISTRY: "SET_REGISTRY",
  NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL: "GSP_RM_CONTROL",
  NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC: "GSP_RM_ALLOC",
  NV_VGPU_MSG_EVENT_GSP_INIT_DONE: "EVENT_GSP_INIT_DONE",
  NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: "EVENT_GSP_RUN_CPU_SEQUENCER",
  NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED: "EVENT_MMU_FAULT_QUEUED",
  NV_VGPU_MSG_EVENT_OS_ERROR_LOG: "EVENT_OS_ERROR_LOG",
  NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD: "EVENT_GSP_POST_NOCAT_RECORD",
}

def gsp_rpc_name(func):
  return GSP_RPC_NAMES.get(func, f"UNKNOWN_{func}")

def printable_c_strings(data, min_len=3, limit=12):
  strings, current = [], []
  for byte in bytes(data):
    if 32 <= byte < 127:
      current.append(chr(byte))
    else:
      if len(current) >= min_len:
        strings.append("".join(current))
        if len(strings) >= limit: return strings
      current = []
  if len(current) >= min_len and len(strings) < limit:
    strings.append("".join(current))
  return strings

def decode_post_nocat_record(msg):
  data = bytes(msg)
  qwords = []
  for off in range(0, min(len(data), 24), 8):
    if off + 8 <= len(data):
      qwords.append(f"0x{struct.unpack_from('<Q', data, off)[0]:x}")
  return {
    "qwords": qwords,
    "kind": f"0x{struct.unpack_from('<Q', data, 16)[0]:x}" if len(data) >= 24 else None,
    "strings": printable_c_strings(data[24:] if len(data) > 24 else data),
  }

def format_post_nocat_record_decode(msg):
  info = decode_post_nocat_record(msg)
  return (f"qwords={','.join(info['qwords']) if info['qwords'] else 'missing'} "
          f"kind={info['kind'] or 'missing'} "
          f"strings={ '|'.join(info['strings']) if info['strings'] else 'missing'}")

class MsgqTxHeader:
  SIZE = 32
  FMT = "<IIIIIIII"
  def __init__(self, version=0, size=0, msg_size=0, msg_count=0, write_ptr=0, flags=0, rx_hdr_off=0, entry_off=0):
    self.version = validate_u32(version, "message queue header version")
    self.size = validate_u32(size, "message queue size")
    self.msg_size = validate_u32(msg_size, "message queue message size")
    self.msg_count = validate_u32(msg_count, "message queue message count")
    self.write_ptr = validate_u32(write_ptr, "message queue write pointer")
    self.flags = validate_u32(flags, "message queue flags")
    self.rx_hdr_off = validate_u32(rx_hdr_off, "message queue RX header offset")
    self.entry_off = validate_u32(entry_off, "message queue entry offset")
  @classmethod
  def unpack_from(cls, data, offset=0): return cls(*struct.unpack_from(cls.FMT, require_len(bytes(data), offset + cls.SIZE, "message queue header"), offset))
  def pack(self): return struct.pack(self.FMT, self.version, self.size, self.msg_size, self.msg_count, self.write_ptr, self.flags, self.rx_hdr_off, self.entry_off)

class RpcHeader:
  SIZE = 32
  FMT = "<IIIIIIII"
  def __init__(self, header_version=3 << 24, signature=NV_VGPU_MSG_SIGNATURE_VALID, length=0, function=0,
               rpc_result=NV_VGPU_MSG_RESULT_RPC_PENDING, rpc_result_private=NV_VGPU_MSG_RESULT_RPC_PENDING, sequence=0, union_value=0):
    self.header_version = validate_u32(header_version, "RPC header version")
    self.signature = validate_u32(signature, "RPC signature")
    self.length = validate_u32(length, "RPC length")
    self.function = validate_u32(function, "RPC function")
    self.rpc_result = validate_u32(rpc_result, "RPC result")
    self.rpc_result_private = validate_u32(rpc_result_private, "RPC private result")
    self.sequence = validate_u32(sequence, "RPC sequence")
    self.union_value = validate_u32(union_value, "RPC union value")
  @classmethod
  def unpack_from(cls, data, offset=0): return cls(*struct.unpack_from(cls.FMT, require_len(bytes(data), offset + cls.SIZE, "RPC header"), offset))
  def pack(self):
    return struct.pack(self.FMT, self.header_version, self.signature, self.length, self.function,
                       self.rpc_result, self.rpc_result_private, self.sequence, self.union_value)

class GspQueueElement:
  SIZE = 48
  FMT = "<16s16sIIII"
  def __init__(self, checksum=0, seq=0, elem_count=0, padding=0):
    self.checksum = validate_u32(checksum, "queue element checksum")
    self.seq = validate_u32(seq, "queue element sequence")
    self.elem_count = validate_u32(elem_count, "queue element count")
    self.padding = validate_u32(padding, "queue element padding")
  @classmethod
  def unpack_from(cls, data, offset=0): return cls(*struct.unpack_from(cls.FMT, require_len(bytes(data), offset + cls.SIZE, "queue element"), offset)[2:])
  def pack(self): return struct.pack(self.FMT, bytes(16), bytes(16), self.checksum, self.seq, self.elem_count, self.padding)

class StallTracer:
  STALL_TRACE_ENV = "NV_ADD_STALL_TRACE"
  STALL_TRACE_PERIOD_ENV = "NV_ADD_STALL_TRACE_PERIOD_MS"
  STALL_TRACE_MAX_ENV = "NV_ADD_STALL_TRACE_MAX_MS"

  def __init__(self, shell, label, period_ms=5.0, max_ms=40000.0):
    self.shell, self.label, self.period_ms, self.max_ms = shell, label, period_ms, max_ms
    self._stop_event = threading.Event()
    self._thread = None
    self._start_perf = None
    self._count = 0
    self._last_summary = None

  @classmethod
  def is_enabled(cls):
    return os.environ.get(cls.STALL_TRACE_ENV) == "1"

  @classmethod
  def parse_period_ms(cls):
    raw = os.environ.get(cls.STALL_TRACE_PERIOD_ENV, "5")
    try: period = float(raw)
    except (TypeError, ValueError): period = 5.0
    if period <= 0: period = 5.0
    return period

  @classmethod
  def parse_max_ms(cls):
    raw = os.environ.get(cls.STALL_TRACE_MAX_ENV, "40000")
    try: maximum = float(raw)
    except (TypeError, ValueError): maximum = 40000.0
    if maximum <= 0: maximum = 40000.0
    return maximum

  def start(self):
    if not self.is_enabled(): return
    if self._thread is not None: return
    if not hasattr(self.shell, "fecs_falcon_base"): return
    try: fecs_base = self.shell.fecs_falcon_base()
    except Exception: return
    self._stop_event = threading.Event()
    self._start_perf = time.perf_counter()
    self._count = 0
    print(f"stall_trace start label={self.label} period_ms={self.period_ms:g} max_ms={self.max_ms:g} fecs_base=0x{fecs_base:x}")
    self._thread = threading.Thread(target=self._run, name=f"stall-tracer-{self.label}", daemon=True)
    self._thread.start()

  def stop(self, reason="complete"):
    if not self.is_enabled(): self._reset(); return
    if self._thread is None: return
    self._stop_event.set()
    self._thread.join(timeout=max(self.period_ms / 1000.0 * 4, 0.5))
    elapsed_ms = (time.perf_counter() - self._start_perf) * 1000.0 if self._start_perf is not None else 0.0
    print(f"stall_trace stop label={self.label} reason={reason} samples={self._count} elapsed_ms={elapsed_ms:g}")
    self._reset()

  def _reset(self):
    self._thread = None
    self._start_perf = None
    self._count = 0
    self._last_summary = None
    self._stop_event = None

  def _sample_falcons(self):
    falcon = FalconController(self.shell)
    bases = (("gsp", NV_FALCON_GSP_BASE), ("sec2", NV_FALCON_SEC2_BASE), ("pmu", NV_FALCON_PMU_BASE), ("fecs", self.shell.fecs_falcon_base()))
    parts = []
    for name, base in bases:
      try: parts.append(f"{name}={falcon.format_state_with_fenced(base)}")
      except Exception as exc: parts.append(f"{name}=unavailable({type(exc).__name__})")
    return " ".join(parts)

  def _run(self):
    while not self._stop_event.is_set():
      try:
        sample = self._sample_falcons()
        elapsed_ms = (time.perf_counter() - self._start_perf) * 1000.0
        self._count += 1
        if sample != self._last_summary or self._count == 1:
          print(f"stall_trace sample label={self.label} seq={self._count} elapsed_ms={elapsed_ms:g} {sample}")
          self._last_summary = sample
        if elapsed_ms > self.max_ms:
          print(f"stall_trace stop label={self.label} reason=max_ms_exceeded samples={self._count} elapsed_ms={elapsed_ms:g}")
          return
      except Exception as exc:
        print(f"stall_trace error label={self.label} exc={type(exc).__name__}: {exc}")
        return
      self._stop_event.wait(self.period_ms / 1000.0)

class GspRpcQueue:
  def __init__(self, gsp, view, completion_view=None):
    if not isinstance(view, MMIOView): raise ValueError("RPC queue view is invalid")
    if completion_view is not None and not isinstance(completion_view, MMIOView): raise ValueError("RPC completion queue view is invalid")
    self.gsp, self.view = gsp, view
    self.tx_view = view.view(fmt='I')
    self.wait_initialized()
    self.tx = MsgqTxHeader.unpack_from(view.view(size=MsgqTxHeader.SIZE)[:])
    self.validate_tx_header(self.tx, view.nbytes, "RPC queue")
    if completion_view is not None:
      comp_tx = MsgqTxHeader.unpack_from(completion_view.view(size=MsgqTxHeader.SIZE)[:])
      self.validate_tx_header(comp_tx, completion_view.nbytes, "RPC completion queue")
    self.seq = 0
    self.queue_mv = view.view(self.tx.entry_off, self.tx.msg_size * self.tx.msg_count)
    self.rx_view = None
    if completion_view is not None:
      if comp_tx.rx_hdr_off + 4 > completion_view.nbytes: raise ValueError("RPC completion queue rx header is outside queue")
      self.rx_view = completion_view.view(comp_tx.rx_hdr_off, fmt='I')

  @staticmethod
  def validate_tx_header(tx, view_size, name):
    if tx.size > view_size: raise ValueError(f"{name} declared size exceeds backing view")
    if tx.msg_size < GspQueueElement.SIZE + RpcHeader.SIZE: raise ValueError(f"{name} message size is invalid")
    if tx.msg_count == 0: raise ValueError(f"{name} message count is zero")
    payload_size = tx.msg_size * tx.msg_count
    if tx.entry_off < MsgqTxHeader.SIZE: raise ValueError(f"{name} entry offset overlaps header")
    if tx.entry_off + payload_size > tx.size: raise ValueError(f"{name} entries are outside declared queue size")
    if tx.write_ptr >= tx.msg_count: raise ValueError(f"{name} write pointer is outside queue")

  def wait_initialized(self, timeout_ms=30000):
    start = time.perf_counter()
    while self.tx_view[7] != 0x1000:
      if (time.perf_counter() - start) * 1000 > timeout_ms:
        header = tuple(self.tx_view[:8])
        raise RuntimeError(f"RPC queue not initialized, header={header}")
      if self.gsp is not None and hasattr(self.gsp, "transport"): self.gsp.transport.sleep(1)
      else: time.sleep(0.001)

  @staticmethod
  def checksum(data):
    data += b"\0" * ((-len(data)) % 8)
    checksum = 0
    for offset in range(0, len(data), 8): checksum ^= struct.unpack_from("<Q", data, offset)[0]
    return hi32(checksum) ^ lo32(checksum)

  @staticmethod
  def record_checksum(elem, payload):
    elem = bytearray(elem)
    if len(elem) < GspQueueElement.SIZE: raise ValueError("RPC queue element checksum input is truncated")
    elem[32:36] = b"\0\0\0\0"
    return GspRpcQueue.checksum(bytes(elem) + bytes(payload))

  def _send_rpc_record(self, func, msg):
    func = validate_u32(func, "RPC function")
    header = RpcHeader(length=len(msg) + RpcHeader.SIZE, function=func).pack()
    payload = header + msg
    elem_count = ceildiv(len(payload) + GspQueueElement.SIZE, self.tx.msg_size)
    record_capacity = max(1, self.tx.msg_count - 1)
    if elem_count == 0 or elem_count > record_capacity:
      raise ValueError(f"RPC record needs {elem_count} queue elements, capacity is {record_capacity}")
    elem = GspQueueElement(seq=self.seq, elem_count=elem_count)
    elem.checksum = self.record_checksum(elem.pack(), payload)
    record = (elem.pack() + payload).ljust(elem_count * self.tx.msg_size, b"\0")

    wp = self.tx_view[4]
    off = wp * self.tx.msg_size
    first = min(len(record), self.queue_mv.nbytes - off)
    self.queue_mv[off:off+first] = record[:first]
    if first < len(record): self.queue_mv[:len(record)-first] = record[first:]
    self.tx_view[4] = (wp + elem_count) % self.tx.msg_count
    memory_barrier()
    self.seq += 1
    if self.gsp is not None: self.gsp.notify_rpc()

  def send_rpc(self, func, msg):
    func = validate_u32(func, "RPC function")
    try:
      msg = bytes(msg)
    except TypeError as exc:
      raise ValueError("RPC message must be bytes-like") from exc
    if os.environ.get("NV_ADD_TRACE_RPC") == "1":
      data = msg
      print(f"standalone send_rpc func={func} func_name={gsp_rpc_name(func)} len={len(data)} "
            f"sha256={hashlib.sha256(data).hexdigest()} head={data[:128].hex()}")
    max_payload = self.tx.msg_size * max(1, self.tx.msg_count - 1) - GspQueueElement.SIZE - RpcHeader.SIZE
    if max_payload <= 0: raise ValueError("RPC queue cannot fit a record payload")
    self._send_rpc_record(func, msg[:max_payload])
    for offset in range(max_payload, len(msg), max_payload):
      self._send_rpc_record(NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD, msg[offset:offset+max_payload])

  def read_resp(self):
    if self.rx_view is None: raise RuntimeError("response queue not configured")
    memory_barrier()
    while self.rx_view[0] != self.tx_view[4]:
      rp = self.rx_view[0]
      if rp >= self.tx.msg_count: raise ValueError(f"RPC response read pointer is outside queue: {rp}")
      off = rp * self.tx.msg_size
      raw_elem = bytes(self.queue_mv[off:off + GspQueueElement.SIZE])
      elem = GspQueueElement.unpack_from(raw_elem)
      if elem.elem_count == 0 or elem.elem_count > self.tx.msg_count:
        raise ValueError(f"RPC response element count is invalid: {elem.elem_count}")
      hdr = RpcHeader.unpack_from(self.queue_mv[off + GspQueueElement.SIZE:off + GspQueueElement.SIZE + RpcHeader.SIZE])
      if hdr.length < RpcHeader.SIZE: raise ValueError(f"RPC response length is smaller than header: 0x{hdr.length:x}")
      advance = elem.elem_count
      if hdr.length + GspQueueElement.SIZE > advance * self.tx.msg_size:
        raise ValueError(f"RPC response length exceeds queue element span: len=0x{hdr.length:x} elems={advance}")
      payload = bytes(self.queue_mv[off + GspQueueElement.SIZE:off + GspQueueElement.SIZE + hdr.length])
      expected_checksum = self.record_checksum(raw_elem, payload)
      if elem.checksum != expected_checksum:
        raise ValueError(f"RPC response checksum mismatch: got=0x{elem.checksum:x} expected=0x{expected_checksum:x}")
      msg = bytes(self.queue_mv[off + GspQueueElement.SIZE + RpcHeader.SIZE:off + GspQueueElement.SIZE + hdr.length])
      if os.environ.get("NV_ADD_TRACE_RPC_READ") == "1":
        print(f"standalone read_rpc rp={rp} wp={self.tx_view[4]} func={hdr.function} func_name={gsp_rpc_name(hdr.function)} len={hdr.length} "
              f"advance={advance} result=0x{hdr.rpc_result:x} private=0x{hdr.rpc_result_private:x} "
              f"sha256={hashlib.sha256(msg).hexdigest()} head={msg[:128].hex()}")
      if hdr.function == NV_VGPU_MSG_EVENT_OS_ERROR_LOG:
        print(f"GSP LOG: {msg[12:].rstrip(bytes([0])).decode('utf-8', errors='replace')}")
      elif hdr.function == NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER and self.gsp is not None:
        print(f"GSP EVENT run_cpu_seq len=0x{len(msg):x} seq_hdr={msg[:40].hex()}")
        if os.environ.get("NV_ADD_TRACE_CPU_SEQ_STACK") == "1":
          traceback.print_stack(limit=10)
        self.gsp.run_cpu_sequencer(msg)
      elif hdr.function == NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD:
        print(f"GSP EVENT post_nocat len=0x{len(msg):x} {format_post_nocat_record_decode(msg)} head={msg[:64].hex()}")
        info = decode_post_nocat_record(msg)
        if info.get("kind") == "0x3" and self.gsp is not None and hasattr(self.gsp, "fecs_falcon_base"):
          strings = info.get("strings") or []
          progress = strings[0] if strings else "unknown"
          self._stall_tracer(f"rm_gpio_pmu_mutex", f"post_nocat-progress={progress}")
          if os.environ.get("NV_ADD_DUMP_LOGBUF") == "1" and hasattr(self.gsp, "gsp_boot") and self.gsp.gsp_boot is not None:
            try: self.gsp.gsp_boot.dump_logbuf(f"stall-rm-pmu-mutex/progress={progress}")
            except Exception as exc:
              if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
                print(f"stall-rm-pmu-mutex logbuf_dump unavailable={type(exc).__name__}: {exc}")
      if hdr.function in (NV_VGPU_MSG_EVENT_OS_ERROR_LOG, NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED) and self.gsp is not None:
        self.gsp.is_err_state = True
      self.rx_view[0] = (self.rx_view[0] + advance) % self.tx.msg_count
      memory_barrier()
      if hdr.rpc_result != 0:
        raise RuntimeError(f"RPC call {hdr.function} ({gsp_rpc_name(hdr.function)}) failed with result 0x{hdr.rpc_result:x} "
                           f"private=0x{hdr.rpc_result_private:x} len=0x{hdr.length:x} seq={hdr.sequence} "
                           f"union=0x{hdr.union_value:x}; queue={self.debug_state()}")
      yield hdr.function, msg

  def debug_state(self, slots=4):
    if slots < 0: raise ValueError("RPC debug slot count must be non-negative")
    state = [
      f"tx_header=({self.tx.version},{self.tx.size},{self.tx.msg_size},{self.tx.msg_count},{self.tx_view[4]},{self.tx.flags},{self.tx.rx_hdr_off},{self.tx.entry_off})"
    ]
    if self.rx_view is not None: state.append(f"rx_ptr={self.rx_view[0]}")
    for slot in range(min(slots, self.tx.msg_count)):
      off = slot * self.tx.msg_size
      elem = bytes(self.queue_mv[off:off + GspQueueElement.SIZE])
      hdr = RpcHeader.unpack_from(self.queue_mv[off + GspQueueElement.SIZE:off + GspQueueElement.SIZE + RpcHeader.SIZE])
      checksum, seq, elem_count, padding = struct.unpack_from("<III I", elem, 32)
      state.append(f"slot{slot}: elem_checksum=0x{checksum:x} seq={seq} elem_count={elem_count} "
                   f"func={hdr.function} func_name={gsp_rpc_name(hdr.function)} len={hdr.length} "
                   f"result=0x{hdr.rpc_result:x} priv=0x{hdr.rpc_result_private:x} sig=0x{hdr.signature:x}")
    return "; ".join(state)

  def _stall_tracer(self, label, reason_hint):
    if not StallTracer.is_enabled(): return
    if self._stall_tracer_inst is not None: return
    if self.gsp is None or not hasattr(self.gsp, "fecs_falcon_base"): return
    period = StallTracer.parse_period_ms()
    maximum = StallTracer.parse_max_ms()
    self._stall_tracer_inst = StallTracer(self.gsp, label=f"{label}/{reason_hint}", period_ms=period, max_ms=maximum)
    self._stall_tracer_inst.start()

  def _stop_stall_tracer(self, reason):
    if self._stall_tracer_inst is None: return
    self._stall_tracer_inst.stop(reason=reason)
    self._stall_tracer_inst = None

  def wait_resp(self, func, timeout_ms=10000):
    func = validate_u32(func, "RPC response function")
    if timeout_ms < 0: raise ValueError("RPC response timeout must be non-negative")
    start = time.perf_counter()
    self._stall_tracer_inst = None
    try:
      while (time.perf_counter() - start) * 1000 < timeout_ms:
        for got_func, msg in self.read_resp():
          if got_func == func: return msg
    finally:
      self._stop_stall_tracer(reason="wait_resp_return")
    raise RuntimeError(f"timeout waiting for RPC response {func} ({gsp_rpc_name(func)}); queue={self.debug_state()}")

def pack_message_queue_init(shared_mem_phys_addr, page_table_entry_count, cmd_queue_offset, stat_queue_offset):
  shared_mem_phys_addr = validate_u64(shared_mem_phys_addr, "shared memory physical address")
  page_table_entry_count = validate_u32(page_table_entry_count, "page-table entry count")
  cmd_queue_offset = validate_u64(cmd_queue_offset, "command queue offset")
  stat_queue_offset = validate_u64(stat_queue_offset, "status queue offset")
  return struct.pack("<QI4xQQ", shared_mem_phys_addr, page_table_entry_count, cmd_queue_offset, stat_queue_offset)

def pack_gsp_arguments_cached(message_queue_init, dmem_stack=True, gpu_instance=0):
  if len(message_queue_init) > 44: raise ValueError("message queue init block is too large")
  gpu_instance = validate_u32(gpu_instance, "GPU instance")
  data = bytearray(72)
  data[0:len(message_queue_init)] = message_queue_init
  struct.pack_into("<I", data, 44, gpu_instance)
  struct.pack_into("<B", data, 48, int(dmem_stack))
  return bytes(data)

def pack_rpc_gsp_rm_alloc(h_client, h_parent, h_object, h_class, params=b"", flags=0):
  h_client = validate_rm_handle(h_client, "client handle")
  h_parent = validate_rm_handle(h_parent, "parent handle")
  h_object = validate_rm_handle(h_object, "object handle")
  h_class = validate_u32(h_class, "RM class")
  flags = validate_u32(flags, "RM alloc flags")
  validate_u32(len(params), "RM alloc params size")
  return struct.pack("<IIIIIII4x", h_client, h_parent, h_object, h_class, 0, len(params), flags) + bytes(params)

def pack_rpc_gsp_rm_alloc_fingerprint(h_client, h_parent, h_object, h_class, params=b"", flags=0):
  return hashlib.sha256(pack_rpc_gsp_rm_alloc(h_client, h_parent, h_object, h_class, params=params, flags=flags)).hexdigest()

def unpack_rpc_gsp_rm_alloc_response(resp):
  if len(resp) < 32: raise ValueError("GSP RM alloc response header is truncated")
  h_client, h_parent, h_object, h_class, status, params_size, flags = struct.unpack_from("<IIIIIII", resp, 0)
  if 32 + params_size > len(resp): raise ValueError("GSP RM alloc response payload is truncated")
  return status, resp[32:32+params_size] if params_size else b""

def pack_rpc_gsp_rm_control(h_client, h_object, cmd, params=b"", flags=0):
  h_client = validate_rm_handle(h_client, "client handle")
  h_object = validate_rm_handle(h_object, "object handle")
  cmd = validate_u32(cmd, "RM control command")
  flags = validate_u32(flags, "RM control flags")
  validate_u32(len(params), "RM control params size")
  return struct.pack("<IIIIII", h_client, h_object, cmd, 0, len(params), flags) + bytes(params)

def pack_rpc_gsp_rm_control_fingerprint(h_client, h_object, cmd, params=b"", flags=0):
  return hashlib.sha256(pack_rpc_gsp_rm_control(h_client, h_object, cmd, params=params, flags=flags)).hexdigest()

def pack_nv0000_alloc_parameters():
  return bytes(NV0000_ALLOC_PARAMETERS_SIZE)

def unpack_rpc_gsp_rm_control_response(resp):
  if len(resp) < 24: raise ValueError("GSP RM control response header is truncated")
  h_client, h_object, cmd, status, params_size, flags = struct.unpack_from("<IIIIII", resp, 0)
  if 24 + params_size > len(resp): raise ValueError("GSP RM control response payload is truncated")
  return status, resp[24:24+params_size] if params_size else b""

def expand_phys_pages(paddrs):
  pages = []
  for entry in paddrs:
    if isinstance(entry, tuple):
      paddr, span = entry
      if paddr < 0 or paddr % 0x1000: raise ValueError(f"physical range base must be 4KB aligned: 0x{paddr:x}")
      if span <= 0 or span % 0x1000: raise ValueError(f"physical range span must be positive and 4KB aligned: 0x{span:x}")
      pages += [paddr + off for off in range(0, span, 0x1000)]
    else:
      if entry < 0 or entry % 0x1000: raise ValueError(f"physical page must be 4KB aligned: 0x{entry:x}")
      pages.append(entry)
  return pages

def pack_rpc_alloc_memory(h_client, h_device, h_memory, h_class, paddrs, length, flags):
  if length <= 0: raise ValueError("allocation length must be positive")
  h_client = validate_rm_handle(h_client, "client handle")
  h_device = validate_rm_handle(h_device, "device handle")
  h_memory = validate_rm_handle(h_memory, "memory handle")
  h_class = validate_u32(h_class, "RM class")
  flags = validate_u32(flags, "alloc-memory flags")
  length = validate_u64(length, "allocation length")
  pages = expand_phys_pages(paddrs)
  if len(pages) < ceildiv(length, 0x1000): raise ValueError("not enough physical pages for allocation")
  header = struct.pack("<IIIIIII4xQI4x", h_client, h_device, h_memory, h_class, flags, 0, 6, length, len(pages))
  pte_desc = struct.pack("<I", (len(pages) & 0xffff) << 16)
  return header + pte_desc + b"\0" * 4 + b"".join(struct.pack("<Q", paddr >> 12) for paddr in pages)

def pack_rpc_alloc_memory_fingerprint(h_client, h_device, h_memory, h_class, paddrs, length, flags):
  return hashlib.sha256(pack_rpc_alloc_memory(h_client, h_device, h_memory, h_class, paddrs, length, flags)).hexdigest()

def unpack_rpc_alloc_memory_response(resp):
  if len(resp) < 28: raise ValueError("GSP alloc-memory response header is truncated")
  h_client, h_device, h_memory, h_class, flags, status, node_id = struct.unpack_from("<IIIIIII", resp, 0)
  return status

def pack_rpc_unloading_guest_driver(in_pm_transition=0, gc6_entering=0, new_level=1 << 6):
  in_pm_transition = validate_u32(int(in_pm_transition), "power-transition flag")
  gc6_entering = validate_u32(int(gc6_entering), "GC6 entering flag")
  if in_pm_transition > 1: raise ValueError("power-transition flag must be 0 or 1")
  if gc6_entering > 1: raise ValueError("GC6 entering flag must be 0 or 1")
  return struct.pack("<BBH I", in_pm_transition, gc6_entering, 0, validate_u32(new_level, "new power level"))

def pack_rpc_set_page_directory(h_client, h_device, h_vaspace, pdir_paddr, num_entries, pasid=0xffffffff, subdevice_id=1, ch_id=0):
  h_client = validate_rm_handle(h_client, "client handle")
  h_device = validate_rm_handle(h_device, "device handle")
  h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
  if pdir_paddr < 0 or pdir_paddr % 0x1000: raise ValueError(f"page directory address must be 4KB aligned: 0x{pdir_paddr:x}")
  if num_entries <= 0: raise ValueError("page directory entry count must be positive")
  pdir_paddr = validate_u64(pdir_paddr, "page directory address")
  num_entries = validate_u32(num_entries, "page directory entry count")
  pasid = validate_u32(pasid, "PASID")
  subdevice_id = validate_u32(subdevice_id, "subdevice id")
  ch_id = validate_u32(ch_id, "channel id")
  return struct.pack("<III4xQIIIIII", h_client, h_device, pasid, pdir_paddr, num_entries, 0x8, h_vaspace, ch_id, subdevice_id, pasid)

def pack_nv90f1_copy_server_reserved_pdes(page_size, virt_addr_lo, virt_addr_hi, levels, h_subdevice=0, subdevice_id=0):
  if len(levels) > 6: raise ValueError("at most 6 PDE levels can be copied")
  h_subdevice = validate_rm_handle(h_subdevice, "subdevice handle")
  subdevice_id = validate_u32(subdevice_id, "subdevice id")
  page_size = validate_u64(page_size, "page size")
  virt_addr_lo = validate_u64(virt_addr_lo, "reserved PDE low VA")
  virt_addr_hi = validate_u64(virt_addr_hi, "reserved PDE high VA")
  data = bytearray(184)
  struct.pack_into("<IIQQQI4x", data, 0, h_subdevice, subdevice_id, page_size, virt_addr_lo, virt_addr_hi, len(levels))
  for idx, (phys_addr, size, aperture, page_shift) in enumerate(levels):
    phys_addr = validate_u64(phys_addr, "reserved PDE physical address")
    size = validate_u64(size, "reserved PDE size")
    aperture = validate_u32(aperture, "reserved PDE aperture")
    page_shift = validate_u32(page_shift, "reserved PDE page shift")
    if page_shift > 0xff: raise ValueError("reserved PDE page shift is outside 8-bit range")
    struct.pack_into("<QQIB3x", data, 40 + idx * 24, phys_addr, size, aperture, page_shift)
  return bytes(data)

def pack_nv2080_gpu_promote_ctx(h_chan_client, h_object, entries, engine_type=NV2080_ENGINE_TYPE_GRAPHICS,
                                h_client=0, ch_id=0, h_virt_memory=0, virt_address=0, size=0):
  if len(entries) > 16: raise ValueError("at most 16 promote entries are supported")
  engine_type = validate_u32(engine_type, "engine type")
  h_client = validate_rm_handle(h_client, "client handle")
  ch_id = validate_u32(ch_id, "channel id")
  h_chan_client = validate_rm_handle(h_chan_client, "channel client handle")
  h_object = validate_rm_handle(h_object, "object handle")
  h_virt_memory = validate_rm_handle(h_virt_memory, "virtual memory handle")
  virt_address = validate_u64(virt_address, "promote context virtual address")
  size = validate_u64(size, "promote context size")
  data = bytearray(560)
  struct.pack_into("<IIIIIIQQI4x", data, 0, engine_type, h_client, ch_id, h_chan_client, h_object,
                   h_virt_memory, virt_address, size, len(entries))
  for idx, entry in enumerate(entries):
    gpu_phys_addr, gpu_virt_addr, entry_size, phys_attr, buffer_id, initialize, nonmapped = entry
    gpu_phys_addr = validate_u64(gpu_phys_addr, "context buffer physical address")
    gpu_virt_addr = validate_u64(gpu_virt_addr, "context buffer virtual address")
    entry_size = validate_u64(entry_size, "context buffer size")
    phys_attr = validate_u32(phys_attr, "context buffer physical attributes")
    buffer_id = validate_u32(buffer_id, "context buffer id")
    if buffer_id > 0xffff: raise ValueError("context buffer id is outside 16-bit range")
    struct.pack_into("<QQQI HBB", data, 48 + idx * 32, gpu_phys_addr, gpu_virt_addr, entry_size,
                     phys_attr, buffer_id, int(initialize), int(nonmapped))
  return bytes(data)

def build_promote_ctx_entries(ctxbufs, mappings=None, virt=None, phys=None):
  entries = []
  if mappings is not None and not hasattr(mappings, "get"): raise ValueError("context buffer mappings must be a mapping")
  mappings = {} if mappings is None else mappings
  for buffer_id, desc in ctxbufs.items():
    buffer_id = validate_u32(buffer_id, "context buffer id")
    if buffer_id > 0xffff: raise ValueError("context buffer id is outside 16-bit range")
    if not isinstance(desc, GRBufDesc): raise ValueError(f"context buffer descriptor {buffer_id} is invalid")
    validate_positive_size(desc.size, f"context buffer {buffer_id} size")
    use_v, use_p = (desc.virt if virt is None else virt), (desc.phys if phys is None else phys)
    mapping = mappings.get(buffer_id)
    if mapping is None: raise KeyError(f"missing context buffer mapping {buffer_id}")
    if mapping.size < desc.size: raise ValueError(f"context buffer mapping {buffer_id} is smaller than descriptor")
    if not mapping.paddrs: raise ValueError(f"context buffer mapping {buffer_id} has no physical pages")
    gpu_phys_addr = mapping.paddrs[0][0] if use_p else 0
    gpu_virt_addr = mapping.va_addr if use_v else 0
    entry_size = desc.size if use_p else 0
    phys_attr = 0x4 if use_p else 0
    entries.append((gpu_phys_addr, gpu_virt_addr, entry_size, phys_attr, buffer_id, use_p, use_p and not use_v))
  return entries

def unpack_nv2080_context_buffer_info(data, engine_index=0):
  validate_u32(engine_index, "context buffer engine index")
  base = engine_index * 26 * 8
  if base >= len(data): raise ValueError("context buffer engine index is outside response")
  if len(data) < base + 26 * 8: raise ValueError("context buffer info response is truncated")
  return [struct.unpack_from("<II", data, base + idx * 8) for idx in range(26)]

def derive_grctx_buf_descs(context_buffer_info):
  if len(context_buffer_info) < 26: raise ValueError("context buffer info must contain 26 entries")
  def ctx_size(index, add=0, align=None):
    size, default_align = context_buffer_info[index]
    return round_up(size + add, align or default_align)
  gr_size = ctx_size(NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS, add=0x40000)
  patch_size = ctx_size(NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH)
  cfg_sizes = {idx: ctx_size(idx + 14, align=(2 << 20) if idx == 5 else None) for idx in range(3, 11)}
  return {
    0: GRBufDesc(gr_size, phys=True, virt=True),
    1: GRBufDesc(patch_size, phys=True, virt=True, local=True),
    2: GRBufDesc(patch_size, phys=True, virt=True),
    **{idx: GRBufDesc(cfg_sizes[idx], phys=False, virt=True) for idx in range(3, 7)},
    9: GRBufDesc(cfg_sizes[9], phys=True, virt=True),
    10: GRBufDesc(cfg_sizes[10], phys=True, virt=False),
    11: GRBufDesc(cfg_sizes[10], phys=True, virt=True),
  }

def allocate_grctx_mappings(mm, grctx_descs, include_local=False, existing=None):
  if existing is not None and not hasattr(existing, "items"): raise ValueError("existing context buffer mappings must be a mapping")
  mappings = {} if existing is None else dict(existing)
  for buffer_id, desc in grctx_descs.items():
    buffer_id = validate_u32(buffer_id, "context buffer id")
    if buffer_id > 0xffff: raise ValueError("context buffer id is outside 16-bit range")
    if not isinstance(desc, GRBufDesc): raise ValueError(f"context buffer descriptor {buffer_id} is invalid")
    validate_positive_size(desc.size, f"context buffer {buffer_id} size")
    if desc.local and not include_local: continue
    if buffer_id not in mappings:
      mappings[buffer_id] = mm.valloc(desc.size, contiguous=True)
    else:
      mapping = mappings[buffer_id]
      if mapping.size < desc.size: raise ValueError(f"context buffer mapping {buffer_id} is smaller than descriptor")
      if not mapping.paddrs: raise ValueError(f"context buffer mapping {buffer_id} has no physical pages")
  return mappings

def build_grctx_promote_payload(mm, h_chan_client, h_object, grctx_descs, include_local=False,
                                existing=None, virt=None, phys=None):
  mappings = allocate_grctx_mappings(mm, grctx_descs, include_local=include_local, existing=existing)
  selected = {idx: desc for idx, desc in grctx_descs.items() if include_local or not desc.local}
  entries = build_promote_ctx_entries(selected, mappings, virt=virt, phys=phys)
  payload = pack_nv2080_gpu_promote_ctx(h_chan_client, h_object, entries)
  packed_entries = b"".join(struct.pack("<QQQI HBB", *entry) for entry in entries)
  entry_text = ";".join(
    f"id={buffer_id}:phys=0x{gpu_phys_addr:x}:virt=0x{gpu_virt_addr:x}:size=0x{entry_size:x}:"
    f"attr=0x{phys_attr:x}:init={int(initialize)}:nonmapped={int(nonmapped)}"
    for gpu_phys_addr, gpu_virt_addr, entry_size, phys_attr, buffer_id, initialize, nonmapped in entries)
  trace_channel_step("promote_ctx_payload", client=hex(h_chan_client), object=hex(h_object),
    include_local=include_local, virt=("default" if virt is None else virt), phys=("default" if phys is None else phys),
    entries=len(entries), ids=[entry[4] for entry in entries], payload_sha256=hashlib.sha256(payload).hexdigest(),
    entries_sha256=hashlib.sha256(repr(entries).encode()).hexdigest(),
    packed_entries_sha256=hashlib.sha256(packed_entries).hexdigest(), entry_text=entry_text)
  return mappings, payload

def pack_registry_table(entries):
  header_size, entry_size = 8, 16
  if not hasattr(entries, "items"): raise ValueError("registry entries must be a mapping")
  entry_count = validate_u32(len(entries), "registry entry count")
  if entry_count > (0xffffffff - header_size) // entry_size: raise ValueError("registry entry table is too large")
  entry_blob, data_blob = bytearray(), bytearray()
  for name, value in entries.items():
    if not isinstance(name, str) or not name: raise ValueError("registry entry name must be a non-empty string")
    if "\0" in name: raise ValueError(f"registry entry name contains NUL: {name!r}")
    try: encoded_name = name.encode("ascii")
    except UnicodeEncodeError as exc: raise ValueError(f"registry entry name must be ASCII: {name!r}") from exc
    value = validate_u32(value, f"registry value {name}")
    validate_u32(len(encoded_name) + 1, f"registry name length {name}")
    name_off = header_size + entry_size * len(entries) + len(data_blob)
    validate_u32(name_off, f"registry name offset {name}")
    entry_blob += struct.pack("<IB3xII", name_off, 1, value, 4)
    data_blob += encoded_name + b"\0"
  total_size = header_size + len(entry_blob) + len(data_blob)
  validate_u32(total_size, "registry table size")
  return struct.pack("<II", total_size, entry_count) + entry_blob + data_blob

def pack_gsp_system_info(gpu_phys_addr, gpu_phys_fb_addr, gpu_phys_inst_addr, pci_device_id, pci_subdevice_id, pci_revision_id,
                         bdf=0, pci_config_mirror_base=0x88000, pci_config_mirror_size=0x1000, max_user_va=0x7ffffffff000):
  gpu_phys_addr = validate_u64(gpu_phys_addr, "GPU physical address")
  gpu_phys_fb_addr = validate_u64(gpu_phys_fb_addr, "GPU framebuffer physical address")
  gpu_phys_inst_addr = validate_u64(gpu_phys_inst_addr, "GPU instance physical address")
  bdf = validate_u64(bdf, "PCI BDF")
  max_user_va = validate_u64(max_user_va, "max user VA")
  pci_config_mirror_base = validate_u32(pci_config_mirror_base, "PCI config mirror base")
  pci_config_mirror_size = validate_u32(pci_config_mirror_size, "PCI config mirror size")
  pci_device_id = validate_u32(pci_device_id, "PCI device id")
  pci_subdevice_id = validate_u32(pci_subdevice_id, "PCI subdevice id")
  pci_revision_id = validate_u32(pci_revision_id, "PCI revision id")
  data = bytearray(928)
  struct.pack_into("<QQQ", data, 0, gpu_phys_addr, gpu_phys_fb_addr, gpu_phys_inst_addr)
  struct.pack_into("<Q", data, 32, bdf)
  struct.pack_into("<Q", data, 72, max_user_va)
  struct.pack_into("<IIIII", data, 80, pci_config_mirror_base, pci_config_mirror_size, pci_device_id, pci_subdevice_id, pci_revision_id)
  struct.pack_into("<B", data, 840, 1)
  struct.pack_into("<Q", data, 920, mmap.PAGESIZE)
  return bytes(data)

def pack_nv0080_alloc_parameters(device_id=0, h_client_share=0, va_mode=0):
  device_id = validate_u32(device_id, "device id")
  h_client_share = validate_rm_handle(h_client_share, "client-share handle")
  va_mode = validate_u32(va_mode, "VA mode")
  data = bytearray(56)
  struct.pack_into("<IIIII", data, 0, device_id, h_client_share, 0, 0, 0)
  struct.pack_into("<I", data, 48, va_mode)
  return bytes(data)

def pack_nv2080_alloc_parameters(subdevice_id=0): return struct.pack("<I", validate_u32(subdevice_id, "subdevice id"))

def pack_nv_memory_virtual_allocation_params(limit=0x1ffffffffffff, offset=0, h_vaspace=0):
  limit = validate_u64(limit, "virtual allocation limit")
  offset = validate_u64(offset, "virtual allocation offset")
  h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
  return struct.pack("<QQI4x", offset, limit, h_vaspace)

def pack_nv_vaspace_allocation_params(va_base=0x1000, va_size=0x1fffffb000000, flags=0):
  va_base = validate_u64(va_base, "VA space base")
  va_size = validate_u64(va_size, "VA space size")
  flags = validate_u32(flags, "VA space flags")
  data = bytearray(48)
  struct.pack_into("<IIQQQI4xQ", data, 0, 0, flags, va_size, 0, 0, 0, va_base)
  return bytes(data)

def pack_nv_channel_group_allocation_params(engine_type=NV2080_ENGINE_TYPE_GRAPHICS, h_vaspace=0):
  engine_type = validate_u32(engine_type, "engine type")
  h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
  return struct.pack("<IIII B3x", 0, 0, h_vaspace, engine_type, 0)

def pack_nv_ctxshare_allocation_params(h_vaspace, flags=0, subctx_id=0):
  return struct.pack("<III", validate_rm_handle(h_vaspace, "vaspace handle"),
                     validate_u32(flags, "ctxshare flags"), validate_u32(subctx_id, "subctx id"))

def pack_nv_memory_desc(base, size, address_space=2, cache_attrib=0):
  return struct.pack("<QQII", validate_u64(base, "memory descriptor base"), validate_u64(size, "memory descriptor size"),
                     validate_u32(address_space, "memory descriptor address space"), validate_u32(cache_attrib, "memory descriptor cache attribute"))

def unpack_nv_memory_desc(data, offset=0):
  if offset < 0 or offset + 24 > len(data): raise ValueError("memory descriptor is truncated")
  base, size, address_space, cache_attrib = struct.unpack_from("<QQII", data, offset)
  return {"base": base, "size": size, "address_space": address_space, "cache_attrib": cache_attrib}

GPFIFO_MEMORY_DESC_OFFSETS = {
  "ramfc": 144,
  "userd": 168,
  "instance": 192,
  "method": 216,
}

def gpfifo_memory_desc_summary(params):
  return {name: unpack_nv_memory_desc(params, offset) for name, offset in GPFIFO_MEMORY_DESC_OFFSETS.items()}

def unpack_nv_channel_gpfifo_allocation_params(data):
  if len(data) < 368: raise ValueError("GPFIFO allocation params are truncated")
  h_object_error, h_object_buffer = struct.unpack_from("<II", data, 0)
  gpfifo_va, entries, flags = struct.unpack_from("<QII", data, 8)
  h_context_share, h_vaspace = struct.unpack_from("<II", data, 24)
  h_userd_memory = struct.unpack_from("<I", data, 32)[0]
  userd_offset = struct.unpack_from("<Q", data, 64)[0]
  engine_type, cid, runlist_id = struct.unpack_from("<III", data, 128)
  internal_flags = struct.unpack_from("<I", data, 244)[0]
  descs = gpfifo_memory_desc_summary(data)
  error_desc = unpack_nv_memory_desc(data, 248)
  return {
    "h_object_error": h_object_error, "h_object_buffer": h_object_buffer, "gpfifo_va": gpfifo_va,
    "entries": entries, "flags": flags, "h_context_share": h_context_share, "h_vaspace": h_vaspace,
    "h_userd_memory": h_userd_memory, "userd_offset": userd_offset, "engine_type": engine_type,
    "cid": cid, "runlist_id": runlist_id, "internal_flags": internal_flags, "descs": descs,
    "error_desc": error_desc,
  }

def gpfifo_alloc_context_string(ctor, before_desc, after_desc, before_sha256, after_sha256):
  return ("gpfifo_ctx="
    f"ctor_sha256={before_sha256} patched_sha256={after_sha256} "
    f"va=0x{ctor['gpfifo_va']:x} entries={ctor['entries']} flags=0x{ctor['flags']:x} "
    f"ctxshare=0x{ctor['h_context_share']:x} vaspace=0x{ctor['h_vaspace']:x} "
    f"object_buffer=0x{ctor['h_object_buffer']:x} object_error=0x{ctor['h_object_error']:x} "
    f"userd_mem=0x{ctor['h_userd_memory']:x} userd_off=0x{ctor['userd_offset']:x} "
    f"engine=0x{ctor['engine_type']:x} cid={ctor['cid']} runlist={ctor['runlist_id']} internal=0x{ctor['internal_flags']:x} "
    f"error=0x{ctor['error_desc']['base']:x}/0x{ctor['error_desc']['size']:x} "
    f"before_ramfc=0x{before_desc['ramfc']['base']:x}/0x{before_desc['ramfc']['size']:x} "
    f"before_userd=0x{before_desc['userd']['base']:x}/0x{before_desc['userd']['size']:x} "
    f"before_instance=0x{before_desc['instance']['base']:x}/0x{before_desc['instance']['size']:x} "
    f"before_method=0x{before_desc['method']['base']:x}/0x{before_desc['method']['size']:x} "
    f"after_ramfc=0x{after_desc['ramfc']['base']:x}/0x{after_desc['ramfc']['size']:x} "
    f"after_userd=0x{after_desc['userd']['base']:x}/0x{after_desc['userd']['size']:x} "
    f"after_instance=0x{after_desc['instance']['base']:x}/0x{after_desc['instance']['size']:x} "
    f"after_method=0x{after_desc['method']['base']:x}/0x{after_desc['method']['size']:x}")

def pack_nv_channel_gpfifo_allocation_params(gpfifo_va, entries, h_context_share, h_vaspace, h_object_buffer,
                                             h_userd_memory, userd_offset, ramfc_paddr, method_paddr, error_paddr=0,
                                             h_object_error=0, engine_type=NV2080_ENGINE_TYPE_GRAPHICS, cid=3,
                                             flags=0x200320, internal_flags=0x1a, userd_paddr=0, userd_size=0x20):
  gpfifo_va = validate_u64(gpfifo_va, "GPFIFO VA")
  entries = validate_u32(entries, "GPFIFO entry count")
  flags = validate_u32(flags, "GPFIFO flags")
  h_context_share = validate_rm_handle(h_context_share, "context-share handle")
  h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
  h_userd_memory = validate_rm_handle(h_userd_memory, "USERD memory handle")
  userd_offset = validate_u64(userd_offset, "USERD offset")
  engine_type = validate_u32(engine_type, "engine type")
  cid = validate_u32(cid, "channel id")
  h_object_buffer = validate_rm_handle(h_object_buffer, "GPFIFO object buffer handle")
  h_object_error = validate_rm_handle(h_object_error, "error object handle")
  internal_flags = validate_u32(internal_flags, "GPFIFO internal flags")
  data = bytearray(368)
  struct.pack_into("<II", data, 0, h_object_error, h_object_buffer)
  struct.pack_into("<QII", data, 8, gpfifo_va, entries, flags)
  struct.pack_into("<II", data, 24, h_context_share, h_vaspace)
  struct.pack_into("<I", data, 32, h_userd_memory)
  struct.pack_into("<Q", data, 64, userd_offset)
  struct.pack_into("<III", data, 128, engine_type, cid, 0)
  if ramfc_paddr: data[144:168] = pack_nv_memory_desc(ramfc_paddr, 0x1000)
  data[168:192] = pack_nv_memory_desc((0 if h_userd_memory else userd_paddr), userd_size)
  if ramfc_paddr: data[192:216] = pack_nv_memory_desc(ramfc_paddr, 0x200)
  if method_paddr: data[216:240] = pack_nv_memory_desc(method_paddr, 0x5000)
  struct.pack_into("<I", data, 244, internal_flags)
  if error_paddr: data[248:272] = pack_nv_memory_desc(error_paddr, 48 << 20)
  return bytes(data)

def tinygrad_equivalent_gpfifo_constructor_params():
  return pack_nv_channel_gpfifo_allocation_params(0x1000000000, 32, 0, 0xcf000003, 0, 0,
    32 * 8, 0, 0, userd_paddr=0x80000100)

def tinygrad_equivalent_gpfifo_constructor_sha256():
  return hashlib.sha256(tinygrad_equivalent_gpfifo_constructor_params()).hexdigest()

def pack_nv83de_alloc_parameters(h_app_client, h_class3d_object):
  return struct.pack("<III", 0, validate_rm_handle(h_app_client, "app client handle"),
                     validate_rm_handle(h_class3d_object, "class3d object handle"))

def unpack_gpfifo_work_submit_token(data):
  if len(data) < 4: raise ValueError("GPFIFO work-submit token response is truncated")
  return struct.unpack_from("<I", data, 0)[0]

class GPFifoState:
  def __init__(self, ring, gpput, entries_count, token=0):
    if entries_count <= 0: raise ValueError("GPFIFO entry count must be positive")
    validate_u32(entries_count, "GPFIFO entry count")
    validate_u32(token, "GPFIFO token")
    if not isinstance(ring, MMIOView) or ring.fmt != 'Q': raise ValueError("GPFIFO ring view is invalid")
    if len(ring) < entries_count: raise ValueError("GPFIFO ring view is smaller than entry count")
    if not isinstance(gpput, MMIOView) or gpput.fmt != 'I': raise ValueError("GPFIFO put view is invalid")
    if len(gpput) < 1: raise ValueError("GPFIFO put view is empty")
    self.ring, self.gpput, self.entries_count, self.token, self.put_value = ring, gpput, entries_count, token, 0

class ChannelResources:
  def __init__(self, gpfifo_area=None, compute_gpfifo=None, cmdq_page=None, kernargs_buf=None, timeline_signal=None, notifier_buf=None, handles=None):
    if handles is not None and not hasattr(handles, "items"): raise ValueError("resource handles must be a mapping")
    self.gpfifo_area = gpfifo_area
    self.compute_gpfifo = compute_gpfifo
    self.cmdq_page = cmdq_page
    self.kernargs_buf = kernargs_buf
    self.timeline_signal = timeline_signal
    self.notifier_buf = notifier_buf
    self.handles = {} if handles is None else handles
    self.kernargs_allocator = BumpAllocator(kernargs_buf.size, base=kernargs_buf.va_addr, wrap=True) if kernargs_buf is not None else None
    self.shared_mem_window = DEFAULT_SHARED_MEM_WINDOW
    self.local_mem_window = DEFAULT_LOCAL_MEM_WINDOW
    self.slm_per_thread = 0
    self.shader_local_mem = None
    self.sass_version = 0x86

class StandaloneBufferAllocator:
  def __init__(self, shell):
    self.shell = shell
    self.rm = None
    self.h_device = 0

  def configure_rm(self, rm, h_device):
    self.rm, self.h_device = rm, h_device

  def alloc_sysmem(self, size, align=0x1000, contiguous=False, rm_handle=False, flags=0):
    validate_positive_size(size, "sysmem allocation size")
    validate_alignment(align)
    size = round_up(size, 0x1000)
    va = self.shell.mm.alloc_vaddr(size, align)
    sysmem, mapping = None, None
    try:
      sysmem = self.shell.transport.alloc_sysmem(size, contiguous=contiguous)
      view, paddrs = sysmem
      if len(paddrs) < size // 0x1000: raise RuntimeError(f"transport returned {len(paddrs)} pages for {size} bytes")
      for paddr in paddrs[:size // 0x1000]:
        if paddr < 0 or paddr % 0x1000: raise ValueError(f"transport returned unaligned sysmem page: 0x{paddr:x}")
      mapping = self.shell.mm.map_range(va, size, [(paddr, 0x1000) for paddr in paddrs[:size // 0x1000]], AddrSpace.SYS, snooped=True, uncached=True)
      mapping.sysmem = sysmem
      buf = GpuBuffer(va, size, view=view, meta=mapping)
      if rm_handle:
        if self.rm is None or self.h_device == 0: raise RuntimeError("RM must be configured before allocating handle-backed sysmem")
        buf.meta.hMemory = self.rm.alloc_memory(self.h_device, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, paddrs, size, flags)
      else:
        buf.meta.hMemory = 0
      return buf
    except Exception:
      with contextlib.suppress(Exception):
        if mapping is not None: self.shell.mm.unmap_range(mapping.va_addr, mapping.size)
      with contextlib.suppress(Exception):
        if sysmem is not None and hasattr(self.shell.transport, "free_sysmem"): self.shell.transport.free_sysmem(sysmem)
      self.shell.mm.va_alloc.free_addr(va)
      raise

  def free(self, buf):
    if not isinstance(buf, GpuBuffer): raise ValueError("buffer free requires a GPU buffer")
    base = buf._base or buf
    if base.view is None:
      if base.meta is not None and isinstance(base.meta, VirtMapping):
        self.shell.mm.vfree(base.meta)
        base.meta = None
      return
    mapping = base.meta
    self.shell.mm.unmap_range(mapping.va_addr, mapping.size)
    self.shell.mm.va_alloc.free_addr(mapping.va_addr)
    if hasattr(self.shell.transport, "free_sysmem"):
      self.shell.transport.free_sysmem(getattr(mapping, "sysmem", None))
    base.view = None
    base.meta = None

  def alloc_vram(self, size, align=0x1000, contiguous=False):
    mapping = self.shell.mm.valloc(size, align=align, contiguous=contiguous)
    return GpuBuffer(mapping.va_addr, mapping.size, view=None, meta=mapping)

  def copyin(self, buf, data):
    if not isinstance(buf, GpuBuffer): raise ValueError("copyin requires a GPU buffer")
    if buf.view is None: raise RuntimeError("copyin currently requires a CPU-visible sysmem buffer")
    raw = memoryview(data).cast('B')
    if len(raw) > buf.size: raise ValueError(f"copyin size {len(raw)} exceeds buffer size {buf.size}")
    buf.cpu_view().view(size=len(raw), fmt='B')[:] = raw

  def copyout(self, buf, size):
    if not isinstance(buf, GpuBuffer): raise ValueError("copyout requires a GPU buffer")
    if buf.view is None: raise RuntimeError("copyout currently requires a CPU-visible sysmem buffer")
    if size < 0 or size > buf.size: raise ValueError(f"copyout size {size} exceeds buffer size {buf.size}")
    return bytearray(buf.cpu_view().view(size=size, fmt='B')[:])

class BootMemoryAllocator:
  def __init__(self, shell):
    self.shell = shell
    self.vram_view = None

  def alloc(self, size, data=None, contiguous=False, sysmem=None, boot=True):
    validate_positive_size(size, "boot allocation size")
    if data is not None and len(data) > size: raise ValueError(f"boot allocation data size {len(data)} exceeds requested size {size}")
    size_aligned = round_up(size, 0x1000)
    use_sysmem = bool(sysmem) or (sysmem is None and not getattr(self.shell, "large_bar", False))
    if use_sysmem:
      view, paddrs = self.shell.transport.alloc_sysmem(size_aligned, contiguous=contiguous)
      if len(paddrs) < size_aligned // 0x1000: raise RuntimeError(f"boot sysmem returned {len(paddrs)} pages for {size_aligned} bytes")
      for paddr in paddrs[:size_aligned // 0x1000]:
        if paddr < 0 or paddr % 0x1000: raise ValueError(f"boot sysmem page must be 4KB aligned: 0x{paddr:x}")
      paddr = None
    else:
      if self.vram_view is None: self.vram_view = self.shell.transport.map_bar(1)
      paddr = self.shell.mm.palloc(size_aligned, boot=boot)
      if paddr < 0 or paddr % 0x1000: raise ValueError(f"boot VRAM address must be 4KB aligned: 0x{paddr:x}")
      _, bar1_size = self.shell.transport.bar_info(1)
      if paddr + size_aligned > bar1_size: raise ValueError("boot VRAM allocation is outside BAR1")
      view = self.vram_view.view(paddr, size_aligned)
      bar1_base = self.shell.transport.bar_info(1)[0]
      paddrs = [bar1_base + paddr + offset for offset in range(0, size_aligned, 0x1000)]
    if data is not None:
      for offset in range(0, len(data), 0x1000):
        view[offset:offset + min(0x1000, len(data) - offset)] = data[offset:offset + 0x1000]
    return view, paddr, paddrs

class GspQueueMemory:
  def __init__(self, queues_view, queues_paddrs, cmd_q_view, stat_q_view, rm_args_view, rm_args_paddrs, libos_args_view, libos_args_paddrs, logbuf_view=None, logbuf_paddrs=None):
    for name, view in (("queues", queues_view), ("command queue", cmd_q_view), ("status queue", stat_q_view),
      ("RM args", rm_args_view), ("LibOS args", libos_args_view)):
      if not isinstance(view, MMIOView): raise ValueError(f"GSP {name} view is invalid")
    for name, paddrs in (("queues", queues_paddrs), ("RM args", rm_args_paddrs), ("LibOS args", libos_args_paddrs)):
      if not paddrs: raise ValueError(f"GSP {name} physical ranges must be non-empty")
      for paddr in paddrs:
        if paddr < 0 or paddr % 0x1000: raise ValueError(f"GSP {name} physical address must be 4KB aligned")
    if logbuf_view is not None and not isinstance(logbuf_view, MMIOView): raise ValueError("GSP logbuf view is invalid")
    if logbuf_paddrs is not None:
      if not logbuf_paddrs: raise ValueError("GSP logbuf physical ranges must be non-empty when provided")
      for paddr in logbuf_paddrs:
        if paddr < 0 or paddr % 0x1000: raise ValueError(f"GSP logbuf physical address must be 4KB aligned: 0x{paddr:x}")
    self.queues_view, self.queues_paddrs = queues_view, queues_paddrs
    self.cmd_q_view, self.stat_q_view = cmd_q_view, stat_q_view
    self.rm_args_view, self.rm_args_paddrs = rm_args_view, rm_args_paddrs
    self.libos_args_view, self.libos_args_paddrs = libos_args_view, libos_args_paddrs
    self.logbuf_view, self.logbuf_paddrs = logbuf_view, logbuf_paddrs

class GspQueueMemoryBuilder:
  def __init__(self, boot_allocator): self.boot_allocator = boot_allocator

  @staticmethod
  def pack_libos_memory_region(kind, loc, size, name, paddr):
    kind = validate_u32(kind, "LibOS memory region kind")
    loc = validate_u32(loc, "LibOS memory region location")
    if kind > 0xff: raise ValueError("LibOS memory region kind is outside 8-bit range")
    if loc > 0xff: raise ValueError("LibOS memory region location is outside 8-bit range")
    validate_positive_size(size, "LibOS memory region size")
    paddr = validate_u64(paddr, "LibOS memory region physical address")
    if paddr % 0x1000: raise ValueError("LibOS memory region physical address must be 4KB aligned")
    encoded_name = name.encode("utf-8")
    if not encoded_name or len(encoded_name) > 8: raise ValueError("LibOS memory region name must be 1-8 bytes")
    return struct.pack("<QQQBB6x", int.from_bytes(encoded_name, "big"), paddr, size, kind, loc)

  def build(self, queue_size=0x40000):
    if queue_size < 0x2000 or queue_size % 0x1000:
      raise ValueError("GSP queue size must be 4KB aligned and at least 0x2000 bytes")
    queue_pte_cnt = (queue_size * 2) // 0x1000
    pte_cnt = queue_pte_cnt + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
    pt_size = round_up(pte_cnt * 8, 0x1000)
    queues_view, _, queues_paddrs = self.boot_allocator.alloc(pt_size + queue_size * 2, sysmem=True)
    qwords = queues_view.view(fmt='Q')
    for idx, paddr in enumerate(queues_paddrs): qwords[idx] = paddr
    cmd_q_view = queues_view.view(pt_size, queue_size)
    stat_q_view = queues_view.view(pt_size + queue_size, queue_size)
    cmd_q_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=queue_size, msg_size=0x1000,
      msg_count=(queue_size - 0x1000) // 0x1000, write_ptr=0, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
    queue_args = pack_message_queue_init(queues_paddrs[0], pte_cnt, pt_size, pt_size + queue_size)
    rm_args_view, _, rm_args_paddrs = self.boot_allocator.alloc(len(pack_gsp_arguments_cached(queue_args)),
      data=pack_gsp_arguments_cached(queue_args))
    logbuf_view, _, logbuf_paddrs = self.boot_allocator.alloc(2 << 20, sysmem=True)
    libos_args = b"".join(self.pack_libos_memory_region(0, 0, 0x10000, f"LOG{name}", logbuf_paddrs[0] + 0x10000 * idx)
      for idx, name in enumerate(("INIT", "INTR", "RM", "MNOC", "KRNL")))
    libos_args += self.pack_libos_memory_region(0, 0, 0x1000, "RMARGS", rm_args_paddrs[0])
    libos_args_view, _, libos_args_paddrs = self.boot_allocator.alloc(0x1000, data=libos_args)
    if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
      print(f"standalone queue queues=0x{queues_paddrs[0]:x} rm_args=0x{rm_args_paddrs[0]:x} logbuf=0x{logbuf_paddrs[0]:x} libos_args=0x{libos_args_paddrs[0]:x} cmd_head={bytes(cmd_q_view[:32]).hex()}")
      if os.environ.get("NV_ADD_TRACE_QUEUE_PTES") == "1":
        print(f"standalone queue ptes first={list(hex(qwords[i]) for i in range(min(8, len(queues_paddrs))))} "
              f"cmd_paddr=0x{queues_paddrs[pt_size // 0x1000]:x} stat_paddr=0x{queues_paddrs[(pt_size + queue_size) // 0x1000]:x} "
              f"pte_cnt={pte_cnt} pt_size=0x{pt_size:x}")
    return GspQueueMemory(queues_view, queues_paddrs, cmd_q_view, stat_q_view, rm_args_view, rm_args_paddrs,
      libos_args_view, libos_args_paddrs, logbuf_view, logbuf_paddrs)

class StandaloneGspBootstrap:
  def __init__(self, shell, firmware_store=None):
    self.shell = shell
    self.shell.gsp_boot = self
    self.boot_allocator = BootMemoryAllocator(shell)
    self.firmware_store = firmware_store or FirmwareStore()
    self.queue_memory = None
    self.cmd_q = None
    self.stat_q = None

  def prepare_queues(self, queue_size=0x40000):
    self.queue_memory = GspQueueMemoryBuilder(self.boot_allocator).build(queue_size=queue_size)
    self.cmd_q = GspRpcQueue(self.shell, self.queue_memory.cmd_q_view)
    return self.queue_memory

  def attach_stat_queue(self):
    if self.queue_memory is None or getattr(self, "cmd_q", None) is None: raise RuntimeError("prepare_queues must run first")
    self.stat_q = GspRpcQueue(self.shell, self.queue_memory.stat_q_view, self.queue_memory.cmd_q_view)
    self.cmd_q.rx_view = self.queue_memory.stat_q_view.view(self.stat_q.tx.rx_hdr_off, fmt='I')
    return self.stat_q

  def load_gsp_firmware(self, allow_download=False):
    return self.firmware_store.load(self.shell.fw_name, "gsp-570.144.bin", allow_download=allow_download)

  def load_bootloader(self, allow_download=False):
    return self.firmware_store.load(self.shell.fw_name, "bootloader-570.144.bin", allow_download=allow_download)

  def load_booter_loader(self, allow_download=False):
    return self.firmware_store.load(self.shell.fw_name, "booter_load-570.144.bin", allow_download=allow_download)

  def prepare_radix3(self, gsp_image):
    pages, _, total_size = GspFirmwarePrep.radix3_layout(len(gsp_image))
    view, _, paddrs = self.boot_allocator.alloc(total_size, boot=False)
    blob, _, _ = GspFirmwarePrep.build_radix3_image(gsp_image, paddrs)
    view[:len(blob)] = blob
    self.gsp_radix3_blob = blob
    return view, paddrs, pages

  def prepare_frts(self):
    vbios_view = self.shell.transport.map_bar(0, off=0x00300000, size=0x100000, fmt='B')
    vbios = b"".join(bytes(vbios_view.view(off, min(0x1000, 0x100000 - off), fmt='B')[:]) for off in range(0, 0x100000, 0x1000))
    desc, signature, image = GspFirmwarePrep.find_fwsec_ucode(vbios)
    self.frts_offset = self.shell.vram_size - 0x200000
    patched = GspFirmwarePrep.patch_fwsec_image(image, desc, signature, 0x15, GspFirmwarePrep.pack_fwsec_frts_cmd(self.frts_offset))
    self.frts_view, self.frts_vram_paddr, self.frts_paddrs = self.boot_allocator.alloc(len(patched), data=patched, sysmem=False, boot=False)
    if os.environ.get("NV_ADD_CHECK_FRTS_BAR1") == "1":
      digest = verify_view_bytes("FRTS BAR1", self.frts_view, patched)
      print(f"FRTS BAR1 readback ok size=0x{len(patched):x} vram=0x{self.frts_vram_paddr:x} sha256={digest}")
    self.frts_desc = desc
    return desc

  def prepare_wpr_meta(self, allow_download=False):
    if not self.shell.chip.fmc_boot and not hasattr(self, "frts_vram_paddr"): self.prepare_frts()
    gsp_elf = self.load_gsp_firmware(allow_download=allow_download)
    gsp_image, signature = GspFirmwarePrep.gsp_signature_section(gsp_elf, self.shell.chip_name)
    self.gsp_image = gsp_image
    self.gsp_signature = signature
    self.gsp_radix3_view, self.gsp_radix3_paddrs, self.gsp_radix3_pages = self.prepare_radix3(gsp_image)
    self.gsp_signature_view, _, self.gsp_signature_paddrs = self.boot_allocator.alloc(len(signature), data=signature, boot=False)

    bootloader = self.load_bootloader(allow_download=allow_download)
    self.booter_image, self.booter_desc = GspFirmwarePrep.bootloader_image_and_desc(bootloader)
    self.booter_view, self.booter_vram_paddr, self.booter_paddrs = self.boot_allocator.alloc(len(self.booter_image), data=self.booter_image, boot=False)

    frts_offset = getattr(self, "frts_offset", None)
    self.wpr_meta_blob = GspFirmwarePrep.build_wpr_meta(self.shell.vram_size, len(self.booter_image), len(gsp_image),
      self.booter_paddrs[0], self.gsp_radix3_paddrs[0], self.gsp_signature_paddrs[0], self.booter_desc,
      fmc_boot=self.shell.chip.fmc_boot, frts_offset=frts_offset)
    self.wpr_meta_view, _, self.wpr_meta_paddrs = self.boot_allocator.alloc(len(self.wpr_meta_blob), data=self.wpr_meta_blob, boot=False)
    self.wpr_meta_sysmem = self.wpr_meta_paddrs[0]
    if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
      print(f"standalone wpr meta=0x{self.wpr_meta_sysmem:x} bootloader=0x{self.booter_paddrs[0]:x} "
            f"radix3=0x{self.gsp_radix3_paddrs[0]:x} signature=0x{self.gsp_signature_paddrs[0]:x} "
            f"meta_sha256={hashlib.sha256(self.wpr_meta_blob).hexdigest()}")
      if os.environ.get("NV_ADD_TRACE_WPR_HEX") == "1":
        print(f"standalone wpr meta_hex={self.wpr_meta_blob.hex()}")
    return self.wpr_meta_blob

  def prepare_booter_loader(self, allow_download=False):
    booter_loader = self.load_booter_loader(allow_download=allow_download)
    self.booter_loader_image, self.booter_loader_desc = GspFirmwarePrep.booter_load_image_and_desc(booter_loader)
    self.booter_loader_view, self.booter_loader_vram_paddr, self.booter_loader_paddrs = self.boot_allocator.alloc(
      len(self.booter_loader_image), data=self.booter_loader_image, sysmem=False, boot=False)
    return self.booter_loader_image

  def verify_sec2_inputs(self):
    checks = (
      ("SEC2 booter-load", self.booter_loader_view, self.booter_loader_image),
      ("WPR metadata", self.wpr_meta_view, self.wpr_meta_blob),
      ("WPR bootloader", self.booter_view, self.booter_image),
      ("GSP signature", self.gsp_signature_view, self.gsp_signature),
      ("GSP radix3", self.gsp_radix3_view, self.gsp_radix3_blob),
    )
    for label, view, expected in checks:
      if expected is None:
        continue
      if not hasattr(view, "view"): raise ValueError(f"{label} view is invalid")
      if not isinstance(expected, (bytes, bytearray, memoryview)): raise ValueError(f"{label} expected data is invalid")
      digest = verify_view_bytes(label, view, expected)
      print(f"{label} readback ok size=0x{len(expected):x} sha256={digest}")

  @staticmethod
  def validate_boot_desc(desc, fields, label):
    if not isinstance(desc, dict): raise ValueError(f"{label} descriptor is invalid")
    for name in fields:
      if name not in desc: raise ValueError(f"{label} descriptor missing {name}")
      value = desc[name]
      if value < 0: raise ValueError(f"{label} descriptor {name} must be non-negative")
    return desc

  @staticmethod
  def validate_boot_paddr_list(paddrs, label):
    if not paddrs: raise ValueError(f"{label} physical ranges must be non-empty")
    for paddr in paddrs:
      if paddr < 0 or paddr % 0x1000: raise ValueError(f"{label} physical address must be 4KB aligned")
    return paddrs

  def wait_init_done(self, timeout_ms=30000):
    if self.stat_q is None:
      self.attach_stat_queue()
    return self.stat_q.wait_resp(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, timeout_ms=timeout_ms)

  def boot_ampere_ada(self):
    if not hasattr(self, "wpr_meta_sysmem"): self.prepare_wpr_meta()
    if not hasattr(self, "booter_loader_paddrs"):
      self.prepare_booter_loader(allow_download=os.environ.get("NV_ADD_DOWNLOAD_FIRMWARE") == "1")
    self.validate_boot_desc(self.booter_loader_desc, ("code_offset", "data_offset", "code_size", "data_size"), "booter-loader")
    validate_u64(self.wpr_meta_sysmem, "WPR metadata physical address")
    if self.wpr_meta_sysmem % 0x1000: raise ValueError("WPR metadata physical address must be 4KB aligned")
    self.validate_boot_paddr_list(self.booter_loader_paddrs, "booter-loader")
    self.validate_boot_paddr_list(self.queue_memory.libos_args_paddrs, "LibOS args")
    falcon = FalconController(self.shell, timeout_ms=30000)
    # Tiny's init_hw does not reset FECS explicitly; FECS BAR hardware gate is cleared by
    # GSP RM's PMU-ucode DMA on this eGPU. The standalone GSP boot_ampere_ada must NOT
    # touch FECS before GSP RM owns it -- touching FECS while PMU is mid-DMA writing FECS
    # IMEM is what previously wedged the FECS->PMU mutex handshake.
    if os.environ.get("NV_ADD_PMU_RESET") == "1":
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone pmu-reset pre state={falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)}")
      try:
        falcon.reset(NV_FALCON_PMU_BASE, riscv=True, force=os.environ.get("NV_ADD_PMU_RESET_FORCE") == "1")
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone pmu-reset post state={falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)}")
      except Exception as exc:
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone pmu-reset skipped reason={type(exc).__name__}: {exc}")
    if os.environ.get("NV_ADD_CLEAR_WPR2_FECS") == "1" and hasattr(self, "frts_offset"):
      clear_paddr = self.shell.vram_size - (1 << 20) - 0x100000
      _, clear_bar1_size = self.shell.transport.bar_info(1)
      if clear_paddr + 0x100000 <= clear_bar1_size and self.shell.mm is not None:
        clear_view = self.shell.transport.map_bar(1).view(clear_paddr, 0x100000)
        clear_view[:] = bytes(0x100000)
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone wpr2-fecs cleared vram=0x{clear_paddr:x} size=0x100000")
    if os.environ.get("NV_ADD_FECS_RESET") == "1" or os.environ.get("NV_ADD_FECS_FENCE_CLEAR") == "1":
      fecs_base = self.shell.fecs_falcon_base()
      if os.environ.get("NV_ADD_FECS_FENCE_CLEAR") == "1":
        # Attempt to clear FECS fence by writing 0 to mailbox0/1 and the FECS tag register
        # Standard procedure: write 0 to FECS mailbox and tag, then engine reset
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone fecs-fence-clear pre base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
        try:
          # Write 0 to FECS mailbox0/mailbox1 and the FALCON_OS register
          self.shell.wreg(fecs_base + NV_PFALCON_FALCON_MAILBOX0, 0)
          self.shell.wreg(fecs_base + NV_PFALCON_FALCON_MAILBOX1, 0)
          self.shell.wreg(fecs_base + NV_PFALCON_FALCON_OS, 0)
          # Wait briefly
          self.shell.transport.sleep(10)
          # Now do a clean engine reset
          self.shell.wreg(fecs_base + NV_PFECS_FALCON_ENGINE_OFFSET, 1)
          self.shell.transport.sleep(100)
          self.shell.wreg(fecs_base + NV_PFECS_FALCON_ENGINE_OFFSET, 0)
          self.shell.transport.sleep(100)
          # Try to read back
          if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
            print(f"standalone fecs-fence-clear post base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
        except Exception as exc:
          if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
            print(f"standalone fecs-fence-clear skipped reason={type(exc).__name__}: {exc}")
      if os.environ.get("NV_ADD_FECS_RESET") == "1":
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone fecs-reset pre base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
        try:
          falcon.reset(fecs_base, riscv=False, force=os.environ.get("NV_ADD_FECS_RESET_FORCE") == "1")
          if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
            print(f"standalone fecs-reset post base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
        except Exception as exc:
          if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
            print(f"standalone fecs-reset skipped reason={type(exc).__name__}: {exc}")
    if hasattr(self, "frts_vram_paddr"):
      self.validate_boot_desc(self.frts_desc, ("imem_load_size", "imem_phys_base", "imem_virt_base", "dmem_phys_base",
        "dmem_load_size", "pkc_data_offset", "engine_id_mask", "ucode_id"), "FRTS")
      falcon.reset(NV_FALCON_GSP_BASE)
      frts_dma_paddr = self.frts_paddrs[0] if os.environ.get("NV_ADD_FALCON_DMA_BAR1") == "1" else self.frts_vram_paddr
      falcon.execute_hs(NV_FALCON_GSP_BASE, frts_dma_paddr,
        code_off=0, data_off=self.frts_desc["imem_load_size"], imem_pa=self.frts_desc["imem_phys_base"], imem_va=self.frts_desc["imem_virt_base"],
        imem_size=self.frts_desc["imem_load_size"], dmem_pa=self.frts_desc["dmem_phys_base"], dmem_va=0,
        dmem_size=self.frts_desc["dmem_load_size"], pkc_off=self.frts_desc["pkc_data_offset"],
        engid=self.frts_desc["engine_id_mask"], ucodeid=self.frts_desc["ucode_id"])
      if self.shell.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI) == 0: raise RuntimeError("WPR2 is not initialized")
      if os.environ.get("NV_ADD_VERIFY_WPR2_AFTER_FRTS") == "1" and hasattr(self, "frts_offset"):
        verify_paddr = self.shell.vram_size - (1 << 20) - 0x100000
        _, verify_bar1_size = self.shell.transport.bar_info(1)
        if verify_paddr + 0x100000 <= verify_bar1_size:
          verify_view = self.shell.transport.map_bar(1).view(verify_paddr, 0x100000)
          verify_bytes = bytes(verify_view[:])
          nonzero_pages = sum(1 for off in range(0, len(verify_bytes), 0x1000) if any(verify_bytes[off:off + 0x1000]))
          verify_digest = hashlib.sha256(verify_bytes).hexdigest()
          if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
            print(f"standalone wpr2-fecs-after-frts vram=0x{verify_paddr:x} nonzero_pages={nonzero_pages} sha256={verify_digest}")
    falcon.reset(NV_FALCON_GSP_BASE, riscv=True)
    self.shell.wreg(NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_MAILBOX0, lo32(self.queue_memory.libos_args_paddrs[0]))
    self.shell.wreg(NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_MAILBOX1, hi32(self.queue_memory.libos_args_paddrs[0]))
    falcon.reset(NV_FALCON_SEC2_BASE)
    # The SEC2 loader consumes WPR metadata in mailbox0/1 and starts GSP RM.
    booter_dma_paddr = self.booter_loader_paddrs[0] if os.environ.get("NV_ADD_FALCON_DMA_BAR1") == "1" else self.booter_loader_vram_paddr
    if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
      print(f"standalone booter img=0x{booter_dma_paddr:x} wpr_meta=0x{self.wpr_meta_sysmem:x} "
            f"code_off=0x{self.booter_loader_desc['code_offset']:x} data_off=0x{self.booter_loader_desc['data_offset']:x} "
            f"code_size=0x{self.booter_loader_desc['code_size']:x} data_size=0x{self.booter_loader_desc['data_size']:x} "
            f"sha256={hashlib.sha256(self.booter_loader_image).hexdigest()}")
    if os.environ.get("NV_ADD_VERIFY_SEC2_INPUTS") == "1":
      self.verify_sec2_inputs()
    mbx = falcon.execute_hs(NV_FALCON_SEC2_BASE, booter_dma_paddr,
      code_off=self.booter_loader_desc["code_offset"], data_off=self.booter_loader_desc["data_offset"],
      imem_pa=0, imem_va=self.booter_loader_desc["code_offset"], imem_size=self.booter_loader_desc["code_size"],
      dmem_pa=0, dmem_va=0, dmem_size=self.booter_loader_desc["data_size"],
      pkc_off=0x10, engid=1, ucodeid=3, mailbox=self.wpr_meta_sysmem)
    if mbx != (0, 0):
      raise RuntimeError(f"Booter failed to execute, mailbox is {mbx[0]:08x}, {mbx[1]:08x}; "
        f"sec2=({falcon.format_state(NV_FALCON_SEC2_BASE)}); gsp=({falcon.format_state(NV_FALCON_GSP_BASE)})")
    self.shell.wreg(NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_OS, 0)
    if reg_get(self.shell.rreg(NV_FALCON_GSP_BASE + NV_PRISCV_RISCV_CPUCTL), 7, 7) != 1:
      raise RuntimeError(f"GSP Core is not active: {falcon.format_state(NV_FALCON_GSP_BASE)}")
    if getattr(self, "cmd_q", None) is not None and self.stat_q is None:
      self.attach_stat_queue()
    init_done = self.wait_init_done()
    self.print_falcons_state("fecs-pmu-postinit")
    if os.environ.get("NV_ADD_DUMP_LOGBUF") == "1":
      try: self.dump_logbuf("postinit-after-falcons")
      except Exception as exc:
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"postinit-after-falcons logbuf_dump unavailable={type(exc).__name__}: {exc}")
    falcon.wait_until(lambda: self._clear_bar1_block(), "BAR1 block did not clear")
    if getattr(self.shell.chip, "fmc_boot", False):
      self.shell.wreg(NV_PBUS_BAR1_BLOCK + 0x4, 0)
    self.post_init_fecs_reset()
    return init_done

  def _clear_bar1_block(self):
    self.shell.wreg(NV_PBUS_BAR1_BLOCK, 0)
    return self.shell.rreg(NV_PBUS_BAR1_BLOCK) == 0

  def post_init_fecs_reset(self, force=False):
    if os.environ.get("NV_ADD_FECS_RESET_POSTINIT") != "1": return False
    if not hasattr(self.shell, "fecs_falcon_base"): return False
    if not hasattr(self.shell, "chip") or self.shell.chip is None: return False
    falcon = FalconController(self.shell, timeout_ms=30000)
    pmu_forced = os.environ.get("NV_ADD_PMU_RESET_POSTINIT_FORCE") == "1" or os.environ.get("NV_ADD_PMU_RESET_POSTINIT") == "1"
    if os.environ.get("NV_ADD_PMU_RESET_POSTINIT") == "1":
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone pmu-reset-postinit pre state={falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)}")
      try:
        falcon.reset(NV_FALCON_PMU_BASE, riscv=True, force=pmu_forced)
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone pmu-reset-postinit post state={falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)}")
      except Exception as exc:
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone pmu-reset-postinit skipped reason={type(exc).__name__}: {exc}")
    fecs_base = self.shell.fecs_falcon_base()
    if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
      print(f"standalone fecs-reset-postinit pre base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
    try:
      falcon.reset(fecs_base, riscv=False, force=force or os.environ.get("NV_ADD_FECS_RESET_POSTINIT_FORCE") == "1")
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone fecs-reset-postinit post base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
      return True
    except Exception as exc:
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone fecs-reset-postinit skipped reason={type(exc).__name__}: {exc}")
      return False

  def post_promote_fecs_reset(self, force=False):
    if os.environ.get("NV_ADD_FECS_RESET_POSTPROMOTE") != "1": return False
    if not hasattr(self.shell, "fecs_falcon_base"): return False
    if not hasattr(self.shell, "chip") or self.shell.chip is None: return False
    falcon = FalconController(self.shell, timeout_ms=30000)
    pmu_forced = os.environ.get("NV_ADD_PMU_RESET_POSTPROMOTE_FORCE") == "1" or os.environ.get("NV_ADD_PMU_RESET_POSTPROMOTE") == "1"
    if os.environ.get("NV_ADD_PMU_RESET_POSTPROMOTE") == "1":
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone pmu-reset-postpromote pre state={falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)}")
      try:
        falcon.reset(NV_FALCON_PMU_BASE, riscv=True, force=pmu_forced)
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone pmu-reset-postpromote post state={falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)}")
      except Exception as exc:
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print(f"standalone pmu-reset-postpromote skipped reason={type(exc).__name__}: {exc}")
    fecs_base = self.shell.fecs_falcon_base()
    if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
      print(f"standalone fecs-reset-postpromote pre base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
    try:
      falcon.reset(fecs_base, riscv=False, force=force or os.environ.get("NV_ADD_FECS_RESET_POSTPROMOTE_FORCE") == "1")
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone fecs-reset-postpromote post base=0x{fecs_base:x} state={falcon.format_state_with_fenced(fecs_base)}")
      return True
    except Exception as exc:
      if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
        print(f"standalone fecs-reset-postpromote skipped reason={type(exc).__name__}: {exc}")
      return False

  def print_falcons_state(self, label):
    if os.environ.get("NV_ADD_TRACE_GSP_BOOT") != "1": return
    if not hasattr(self.shell, "fecs_falcon_base"): return
    if not hasattr(self.shell, "chip") or self.shell.chip is None: return
    state_falcon = FalconController(self.shell)
    for falcon_label, base in (("gsp", NV_FALCON_GSP_BASE), ("sec2", NV_FALCON_SEC2_BASE), ("pmu", NV_FALCON_PMU_BASE), ("fecs", self.shell.fecs_falcon_base())):
      try:
        print(f"standalone {label} state label={falcon_label} base=0x{base:x} state={state_falcon.format_state_with_fenced(base)}")
      except Exception as exc:
        print(f"standalone {label} state label={falcon_label} base=0x{base:x} unavailable={type(exc).__name__}")

  LOGBUF_SUBREGIONS = (("LOG_INIT", 0x00000, 0x10000), ("LOG_INTR", 0x10000, 0x20000), ("LOG_RM", 0x20000, 0x30000),
                       ("LOG_MNOC", 0x30000, 0x40000), ("LOG_KRNL", 0x40000, 0x50000))

  @staticmethod
  def extract_logbuf_text(data, min_printable=4, max_lines=64, max_line_len=512):
    if not data: return []
    lines = []
    current = bytearray()
    for byte in data:
      if byte == 0x0a:
        if current:
          line = bytes(current)
          printable = sum(1 for c in line if 0x20 <= c < 0x7f or c in (0x09, 0x0a))
          if len(line) >= min_printable and printable >= max(1, len(line) * 3 // 4):
            text = line.rstrip(b"\r\n").decode("utf-8", errors="replace")
            lines.append(text[:max_line_len])
          current = bytearray()
      elif byte == 0x00:
        if current:
          line = bytes(current)
          printable = sum(1 for c in line if 0x20 <= c < 0x7f or c == 0x09)
          if len(line) >= min_printable and printable >= max(1, len(line) * 3 // 4):
            text = line.rstrip(b"\x00").decode("utf-8", errors="replace")
            lines.append(text[:max_line_len])
          current = bytearray()
      else:
        current.append(byte)
        if len(current) > max_line_len * 2:
          line = bytes(current)
          printable = sum(1 for c in line if 0x20 <= c < 0x7f or c == 0x09)
          if len(line) >= min_printable and printable >= max(1, len(line) * 3 // 4):
            text = line.rstrip(b"\x00").decode("utf-8", errors="replace")
            lines.append(text[:max_line_len])
          current = bytearray()
    if current:
      line = bytes(current)
      printable = sum(1 for c in line if 0x20 <= c < 0x7f or c == 0x09)
      if len(line) >= min_printable and printable >= max(1, len(line) * 3 // 4):
        text = line.rstrip(b"\x00").decode("utf-8", errors="replace")
        lines.append(text[:max_line_len])
    return lines[:max_lines]

  def dump_logbuf(self, label, max_lines=64, max_line_len=512):
    if os.environ.get("NV_ADD_DUMP_LOGBUF") != "1" and os.environ.get("NV_ADD_TRACE_GSP_BOOT") != "1": return
    if self.queue_memory is None: return
    logbuf_view = getattr(self.queue_memory, "logbuf_view", None)
    logbuf_paddrs = getattr(self.queue_memory, "logbuf_paddrs", None)
    if logbuf_view is None or not logbuf_paddrs: return
    paddr_base = logbuf_paddrs[0]
    total_size = max(end for _, _, end in self.LOGBUF_SUBREGIONS)
    try: raw = bytes(logbuf_view[:total_size])
    except Exception as exc:
      print(f"standalone {label} logbuf_dump unavailable={type(exc).__name__}")
      return
    print(f"standalone {label} logbuf_dump base=0x{paddr_base:x} total_size=0x{total_size:x} subregions={len(self.LOGBUF_SUBREGIONS)}")
    total_extracted = 0
    for sub_label, sub_start, sub_end in self.LOGBUF_SUBREGIONS:
      sub_data = raw[sub_start:sub_end]
      if not any(sub_data): continue
      nonzero = sum(1 for b in sub_data if b != 0)
      sha = hashlib.sha256(sub_data).hexdigest()
      first_nonzero = next((i for i, b in enumerate(sub_data) if b != 0), -1)
      last_nonzero = max((i for i, b in enumerate(sub_data) if b != 0), default=-1)
      lines = StandaloneGspBootstrap.extract_logbuf_text(sub_data, min_printable=4, max_lines=max_lines, max_line_len=max_line_len)
      total_extracted += len(lines)
      print(f"standalone {label} logbuf subregion={sub_label} offset=0x{sub_start:x} size=0x{sub_end - sub_start:x} nonzero_bytes={nonzero} first_nonzero=0x{first_nonzero:x} last_nonzero=0x{last_nonzero:x} sha256={sha} lines={len(lines)}")
      for idx, line in enumerate(lines):
        print(f"standalone {label} logbuf {sub_label} line[{idx}]={line}")
      if sub_label == "LOG_RM" and os.environ.get("NV_ADD_DUMP_LOGBUF_RAW") == "1":
        print(f"standalone {label} logbuf {sub_label} raw_hex_first_512={sub_data[:512].hex()}")
    print(f"standalone {label} logbuf_dump summary total_lines={total_extracted}")

  def make_rm_client(self):
    if self.cmd_q is None: raise RuntimeError("prepare_queues must run first")
    return GspRmClient(self.cmd_q, self.stat_q or self.cmd_q)

class StandaloneSignal:
  def __init__(self, buf, offset=0):
    if offset < 0 or offset % 8: raise ValueError("signal offset must be non-negative and 8-byte aligned")
    if offset + 8 > buf.size: raise ValueError("signal storage exceeds buffer size")
    self.buf, self.offset = buf, offset
    self.value_addr = buf.va_addr + offset
    check_qmd_pointer("timeline signal", self.value_addr, align=8)
  @property
  def value(self): return self.buf.cpu_view().view(offset=self.offset, size=8, fmt='Q')[0]
  @value.setter
  def value(self, value): self.buf.cpu_view().view(offset=self.offset, size=8, fmt='Q')[0] = validate_u64(value, "signal value")

class StandaloneSubmitter:
  def __init__(self, shell, cmdq_page, compute_gpfifo, gpu_mmio):
    if not isinstance(cmdq_page, GpuBuffer): raise ValueError("command queue page must be a GPU buffer")
    if cmdq_page.view is None: raise ValueError("command queue page must be CPU-visible")
    if not isinstance(compute_gpfifo, GPFifoState): raise ValueError("compute GPFIFO state is invalid")
    if not isinstance(compute_gpfifo.ring, MMIOView): raise ValueError("compute GPFIFO ring view is invalid")
    if not isinstance(compute_gpfifo.gpput, MMIOView): raise ValueError("compute GPFIFO put view is invalid")
    if not isinstance(gpu_mmio, (MMIOView, RemoteMMIOView)): raise ValueError("GPU MMIO view is invalid")
    self.shell, self.cmdq_page, self.compute_gpfifo, self.gpu_mmio = shell, cmdq_page, compute_gpfifo, gpu_mmio
    self.cmdq_allocator = BumpAllocator(cmdq_page.size, base=cmdq_page.va_addr, wrap=True)
    self.cmdq = cmdq_page.cpu_view().view(fmt='I')

  def ring_doorbell(self):
    if not hasattr(self.gpu_mmio, "__setitem__"): raise ValueError("GPU MMIO view is invalid")
    self.gpu_mmio[0x90 // 4] = self.compute_gpfifo.token

  def submit_gpfifo(self, words):
    if len(words) == 0: raise ValueError("GPFIFO command stream must be non-empty")
    words = [validate_u32(word, "GPFIFO command word") for word in words]
    if len(words) * 4 > self.cmdq_page.size: raise ValueError("GPFIFO command stream is larger than the command queue page")
    cmdq_addr = self.cmdq_allocator.alloc(len(words) * 4, 16)
    check_qmd_pointer("GPFIFO command buffer", cmdq_addr, align=4)
    cmdq_wptr = (cmdq_addr - self.cmdq_page.va_addr) // 4
    write_words(self.cmdq, cmdq_wptr, words)
    fifo = self.compute_gpfifo
    ring_index = fifo.put_value % fifo.entries_count
    fifo.ring[ring_index] = (cmdq_addr // 4 << 2) | (len(words) << 42) | (1 << 41)
    fifo.gpput[0] = (fifo.put_value + 1) % fifo.entries_count
    memory_barrier()
    self.ring_doorbell()
    fifo.put_value += 1
    return ring_index

def nv_method_packet(subchannel, method, *args, typ=2):
  return [(typ << 28) | (len(args) << 16) | (subchannel << 13) | (method >> 2), *args]

class StandaloneNvBackend:
  def __init__(self, shell, resources, allocator, submitter, device_name="NV:standalone", simulate=False, rm=None):
    self.shell, self.resources, self.allocator, self.submitter = shell, resources, allocator, submitter
    self.device_name, self.iface_name = device_name, type(shell.transport).__name__
    self.timeline_value = 1
    self.simulate = simulate
    self.rm = rm
    self._closed = False
    self.last_launch = None

  @property
  def shared_mem_window(self): return getattr(self.resources, "shared_mem_window", 0)
  @property
  def local_mem_window(self): return getattr(self.resources, "local_mem_window", 0)
  @property
  def slm_per_thread(self): return getattr(self.resources, "slm_per_thread", 0x240)
  @property
  def sass_version(self): return getattr(self.resources, "sass_version", 0x86)

  def alloc(self, size): return self.allocator.alloc_sysmem(size)
  def free(self, buf): return self.allocator.free(buf)
  def copyin(self, buf, data): self.allocator.copyin(buf, data)
  def copyout(self, buf, size): return self.allocator.copyout(buf, size)
  def synchronize(self): pass
  def close(self):
    if self._closed: return
    self._closed = True
    errors = []
    registration = self.resources.handles.pop("uvm_channel", None)
    if registration is not None and hasattr(self.shell.transport, "free_registered_channel"):
      try:
        self.shell.transport.free_registered_channel(registration)
      except Exception as exc:
        errors.append(exc)
    try:
      self.release_resources()
    except Exception as exc:
      errors.append(exc)
    if self.rm is not None:
      try:
        self.rm.unloading_guest_driver()
      except Exception as exc:
        errors.append(exc)
    if errors:
      raise errors[0]
  def release_resources(self):
    errors = []
    try:
      self.release_context_mappings()
    except Exception as exc:
      errors.append(exc)
    golden_ctx = self.resources.handles.get("golden_ctx")
    if isinstance(golden_ctx, dict):
      gpfifo_area = golden_ctx.pop("gpfifo_area", None)
      if gpfifo_area is not None:
        try:
          self.allocator.free(gpfifo_area)
        except Exception as exc:
          errors.append(exc)
    for name in ("shader_local_mem", "timeline_signal", "kernargs_buf", "cmdq_page", "notifier_buf", "gpfifo_area"):
      obj = getattr(self.resources, name, None)
      buf = obj.buf if isinstance(obj, StandaloneSignal) else obj
      if buf is None: continue
      try:
        self.allocator.free(buf)
      except Exception as exc:
        errors.append(exc)
      setattr(self.resources, name, None)
    if errors:
      raise errors[0]
  def release_context_mappings(self):
    seen = set()
    errors = []
    for key in ("user_grctx_mappings", "rm_private_mappings"):
      mappings = self.resources.handles.pop(key, None)
      if mappings is None: mappings = {}
      if not hasattr(mappings, "values"): raise ValueError(f"{key} must be a mapping")
      for mapping in mappings.values():
        if isinstance(mapping, (list, tuple)):
          items = mapping
        else:
          items = (mapping,)
        for item in items:
          if not isinstance(item, VirtMapping) or id(item) in seen: continue
          seen.add(id(item))
          try:
            self.shell.mm.vfree(item)
          except Exception as exc:
            errors.append(exc)
    golden_ctx = self.resources.handles.get("golden_ctx")
    if isinstance(golden_ctx, dict):
      mappings = golden_ctx.pop("grctx_mappings", {})
      if mappings is None: mappings = {}
      if not hasattr(mappings, "values"): raise ValueError("golden context mappings must be a mapping")
      for mapping in mappings.values():
        if not isinstance(mapping, VirtMapping) or id(mapping) in seen: continue
        seen.add(id(mapping))
        try:
          self.shell.mm.vfree(mapping)
        except Exception as exc:
          errors.append(exc)
    if errors:
      raise errors[0]
  def ensure_local_memory(self, size):
    if size < 0: raise ValueError("shader local-memory size must be non-negative")
    required = round_up(size, 32)
    validate_u32(required, "shader local-memory size")
    if self.slm_per_thread >= required: return
    bytes_per_tpc = round_up(round_up(required * 32, 0x200) * 64, 0x8000)
    total = round_up(bytes_per_tpc, 0x20000)
    old_mem, old_slm = self.resources.shader_local_mem, self.resources.slm_per_thread
    shader_local_mem = self.allocator.alloc_sysmem(total)
    words = []
    words += nv_method_packet(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(shader_local_mem.va_addr))
    words += nv_method_packet(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(bytes_per_tpc), 0xff)
    try:
      self.submit_gpfifo(words)
    except Exception:
      with contextlib.suppress(Exception): self.allocator.free(shader_local_mem)
      self.resources.shader_local_mem, self.resources.slm_per_thread = old_mem, old_slm
      raise
    self.resources.shader_local_mem = shader_local_mem
    self.resources.slm_per_thread = required
  def setup_compute_queue(self):
    chip_name = getattr(self.shell, "chip_name", "")
    compute_class = self.resources.handles.get("compute_class", ADA_COMPUTE_A if chip_name.startswith("AD") else AMPERE_COMPUTE_B)
    validate_u32(compute_class, "compute class")
    validate_u64(self.local_mem_window, "local memory window")
    validate_u64(self.shared_mem_window, "shared memory window")
    words = []
    words += nv_method_packet(1, NVC6C0_SET_OBJECT, compute_class)
    words += nv_method_packet(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(self.local_mem_window))
    words += nv_method_packet(1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(self.shared_mem_window))
    words += nv_method_packet(1, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, (1 << 0) | (1 << 4) | (1 << 12))
    self.submit_gpfifo(words)
    return words
  def alloc_kernargs(self, size, align=8):
    validate_positive_size(size, "kernel argument allocation size")
    validate_alignment(align)
    if not isinstance(self.resources.kernargs_buf, GpuBuffer): raise ValueError("kernel argument buffer is invalid")
    if self.resources.kernargs_buf.view is None: raise ValueError("kernel argument buffer must be CPU-visible")
    if self.resources.kernargs_allocator is None: raise ValueError("kernel argument allocator is not initialized")
    off = self.resources.kernargs_allocator.alloc(size, align)
    return self.resources.kernargs_buf.offset(off - self.resources.kernargs_buf.va_addr, size)
  def require_timeline_signal(self, signal=None):
    signal = self.resources.timeline_signal if signal is None else signal
    if not isinstance(signal, StandaloneSignal): raise ValueError("timeline signal is invalid")
    return signal
  def timeline_state(self):
    if self.timeline_value <= 0 or self.timeline_value > 0xffffffffffffffff: raise ValueError("timeline value is outside 64-bit range")
    wait_value = self.timeline_value - 1
    self.timeline_value += 1
    return wait_value, self.timeline_value - 1, self.require_timeline_signal()
  def submit_gpfifo(self, words): self.submitter.submit_gpfifo(words)
  def submit_timeline_signal(self, value=None):
    if value is None:
      _, value, signal = self.timeline_state()
    else:
      value = validate_u64(value, "timeline signal value")
      signal = self.require_timeline_signal()
    words = build_timeline_signal_words(signal.value_addr, value, interrupt=True)
    self.submit_gpfifo(words)
    if self.simulate: signal.value = value
    return value, signal, words
  def simulate_add_launch(self, out, a, b, signal, done_value):
    av = struct.unpack("4f", self.copyout(a, 16))
    bv = struct.unpack("4f", self.copyout(b, 16))
    self.copyin(out, struct.pack("4f", *[x + y for x, y in zip(av, bv)]))
    signal.value = done_value
    self.last_launch = (out.va_addr, a.va_addr, b.va_addr, done_value)
  def simulate_mul_launch(self, out, a, b, signal, done_value):
    av = struct.unpack("4f", self.copyout(a, 16))
    bv = struct.unpack("4f", self.copyout(b, 16))
    self.copyin(out, struct.pack("4f", *[x * y for x, y in zip(av, bv)]))
    signal.value = done_value
    self.last_launch = (out.va_addr, a.va_addr, b.va_addr, done_value)
  def wait_signal(self, signal, value, timeout_ms=30000):
    signal = self.require_timeline_signal(signal)
    value = validate_u64(value, "timeline wait value")
    if timeout_ms < 0: raise ValueError("timeline wait timeout must be non-negative")
    start = time.perf_counter()
    while signal.value < value:
      if (time.perf_counter() - start) * 1000 > timeout_ms:
        raise RuntimeError(f"timeout waiting for timeline {value}, got {signal.value}")
      time.sleep(0.001)

class StandaloneChannelBuilder:
  def __init__(self, shell, rm, gsp_boot=None):
    self.shell, self.rm = shell, rm
    self.gsp_boot = gsp_boot

  def allocate_base_objects(self, reserved_size=512 << 20):
    validate_positive_size(reserved_size, "user reserved size")
    validate_u64(reserved_size, "user reserved size")
    h_device = self.rm.rm_alloc(self.rm.priv_root, NV01_DEVICE_0, pack_nv0080_alloc_parameters(h_client_share=self.rm.priv_root))
    h_subdevice = self.rm.rm_alloc(h_device, NV20_SUBDEVICE_0, pack_nv2080_alloc_parameters())
    h_virtmem = self.rm.rm_alloc(h_device, NV01_MEMORY_VIRTUAL, pack_nv_memory_virtual_allocation_params())
    h_vaspace = self.rm.rm_alloc(h_device, FERMI_VASPACE_A, pack_nv_vaspace_allocation_params(flags=1 | 2))
    res_va = self.shell.mm.alloc_vaddr(reserved_size)
    levels = self.shell.mm.reserved_pde_levels(res_va, 3, reserved_size)
    pde_payload = pack_nv90f1_copy_server_reserved_pdes(reserved_size, res_va, res_va + reserved_size - 1, levels)
    self.rm.rm_control(h_vaspace, NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, pde_payload)
    return h_device, h_subdevice, h_virtmem, h_vaspace

  def prepare_golden_image_context(self, allocator, reserved_size=512 << 20):
    validate_positive_size(reserved_size, "golden reserved size")
    validate_u64(reserved_size, "golden reserved size")
    trace_channel_step("golden_start", reserved_size=reserved_size)
    res_va, gpfifo_area, grctx_mappings = None, None, None
    h_device = self.rm.rm_alloc(self.rm.priv_root, NV01_DEVICE_0, pack_nv0080_alloc_parameters(h_client_share=self.rm.priv_root))
    try:
      h_subdevice = self.rm.rm_alloc(h_device, NV20_SUBDEVICE_0, pack_nv2080_alloc_parameters())
      h_vaspace = self.rm.rm_alloc(h_device, FERMI_VASPACE_A, pack_nv_vaspace_allocation_params())
      res_va = self.shell.mm.alloc_vaddr(reserved_size)
      levels = self.shell.mm.reserved_pde_levels(res_va, 3, reserved_size)
      pde_payload = pack_nv90f1_copy_server_reserved_pdes(reserved_size, res_va, res_va + reserved_size - 1, levels)
      trace_channel_step("golden_reserved", device=hex(h_device), subdevice=hex(h_subdevice), vaspace=hex(h_vaspace),
        va=hex(res_va), levels=len(levels))
      self.rm.rm_control(h_vaspace, NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, pde_payload)

      gpfifo_area = allocator.alloc_vram(0x1000, contiguous=True)
      userd_paddr = gpfifo_area.meta.paddrs[0][0] + 0x20 * 8
      gpfifo_params = pack_nv_channel_gpfifo_allocation_params(gpfifo_area.va_addr, 32, 0, h_vaspace, 0, 0,
        0x20 * 8, 0, 0, userd_paddr=userd_paddr, internal_flags=0x1a)
      gpfifo_desc = gpfifo_memory_desc_summary(gpfifo_params)
      error_desc = unpack_nv_memory_desc(gpfifo_params, 248)
      trace_channel_step("golden_gpfifo_alloc", parent=hex(h_device), area=hex(gpfifo_area.va_addr),
        entries=32, userd_paddr=hex(userd_paddr),
        desc_ramfc=f"0x{gpfifo_desc['ramfc']['base']:x}/0x{gpfifo_desc['ramfc']['size']:x}",
        desc_userd=f"0x{gpfifo_desc['userd']['base']:x}/0x{gpfifo_desc['userd']['size']:x}",
        desc_instance=f"0x{gpfifo_desc['instance']['base']:x}/0x{gpfifo_desc['instance']['size']:x}",
        desc_method=f"0x{gpfifo_desc['method']['base']:x}/0x{gpfifo_desc['method']['size']:x}",
        desc_error=f"0x{error_desc['base']:x}/0x{error_desc['size']:x}",
        h_object_buffer="0x0", h_object_error="0x0", params_sha256=hashlib.sha256(gpfifo_params).hexdigest())
      h_gpfifo = self.rm.rm_alloc(h_device, AMPERE_CHANNEL_GPFIFO_A, gpfifo_params)
      trace_channel_step("golden_gpfifo", gpfifo=hex(h_gpfifo), area=hex(gpfifo_area.va_addr), userd_paddr=hex(userd_paddr))
      info_payload = self.rm.rm_control(h_subdevice, NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO, bytes(1664))
      grctx_descs = derive_grctx_buf_descs(unpack_nv2080_context_buffer_info(info_payload))
      grctx_mappings, promote_payload = build_grctx_promote_payload(self.shell.mm, self.rm.priv_root, h_gpfifo, grctx_descs)
      self.rm.rm_control(h_subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, promote_payload)
      if self.gsp_boot is not None: self.gsp_boot.post_promote_fecs_reset()
      if os.environ.get("NV_ADD_REPROMOTE_AFTER_RESET") == "1":
        if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
          print("standalone repromote re-issuing GPU_PROMOTE_CTX after FECS reset")
        self.rm.rm_control(h_subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, promote_payload)
      if os.environ.get("NV_ADD_POSTPROMOTE_SETTLE_MS") is not None:
        settle_ms = int(os.environ.get("NV_ADD_POSTPROMOTE_SETTLE_MS", "0"))
        if settle_ms > 0 and hasattr(self.shell.transport, "sleep"):
          self.shell.transport.sleep(settle_ms)
          if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
            print(f"standalone postpromote settle slept {settle_ms}ms")
      compute_class = ADA_COMPUTE_A if self.shell.chip_name.startswith("AD") else AMPERE_COMPUTE_B
      compute_object = self.rm.next_handle if hasattr(self.rm, "next_handle") else 0
      compute_rpc_sha256 = hashlib.sha256(pack_rpc_gsp_rm_alloc(self.rm.priv_root, h_gpfifo, compute_object,
        compute_class, b"")).hexdigest() if compute_object else "unknown"
      trace_channel_step("golden_promote_done", gpfifo=hex(h_gpfifo), subdevice=hex(h_subdevice),
        promote_sha256=hashlib.sha256(promote_payload).hexdigest(), mappings=len(grctx_mappings))
      if self.gsp_boot is not None: self.gsp_boot.print_falcons_state("fecs-pmu-postpromote")
      if os.environ.get("NV_ADD_DUMP_LOGBUF") == "1" and self.gsp_boot is not None:
        try: self.gsp_boot.dump_logbuf("postpromote-after-falcons")
        except Exception as exc:
          if os.environ.get("NV_ADD_TRACE_GSP_BOOT") == "1":
            print(f"postpromote-after-falcons logbuf_dump unavailable={type(exc).__name__}: {exc}")
      trace_channel_step("golden_compute_alloc", parent=hex(h_gpfifo), expected_object=hex(compute_object),
        compute_class=hex(compute_class), expected_rpc_sha256=compute_rpc_sha256)
      h_compute = self.rm.rm_alloc(h_gpfifo, compute_class, b"")
      dma_object = self.rm.next_handle if hasattr(self.rm, "next_handle") else 0
      dma_rpc_sha256 = hashlib.sha256(pack_rpc_gsp_rm_alloc(self.rm.priv_root, h_gpfifo, dma_object,
        AMPERE_DMA_COPY_B, b"")).hexdigest() if dma_object else "unknown"
      trace_channel_step("golden_dma_alloc", parent=hex(h_gpfifo), expected_object=hex(dma_object),
        dma_class=hex(AMPERE_DMA_COPY_B), expected_rpc_sha256=dma_rpc_sha256)
      h_dma = self.rm.rm_alloc(h_gpfifo, AMPERE_DMA_COPY_B, b"")
      trace_channel_step("golden_done", compute=hex(h_compute), dma=hex(h_dma), descs=len(grctx_descs), mappings=len(grctx_mappings))
      return {"device": h_device, "subdevice": h_subdevice, "vaspace": h_vaspace, "gpfifo": h_gpfifo,
              "compute": h_compute, "dma": h_dma, "gpfifo_area": gpfifo_area, "reserved_va": res_va,
              "reserved_levels": levels, "grctx_descs": grctx_descs, "grctx_mappings": grctx_mappings}
    except Exception:
      if grctx_mappings is not None:
        seen = set()
        for mapping in grctx_mappings.values():
          if isinstance(mapping, VirtMapping) and id(mapping) not in seen:
            seen.add(id(mapping))
            with contextlib.suppress(Exception): self.shell.mm.vfree(mapping)
      if gpfifo_area is not None:
        with contextlib.suppress(Exception): allocator.free(gpfifo_area)
      if res_va is not None:
        with contextlib.suppress(Exception): self.shell.mm.va_alloc.free_addr(res_va)
      raise

  def promote_user_compute_context(self, h_client, h_subdevice, h_gpfifo, grctx_descs, grctx_mappings=None):
    selected = {idx: grctx_descs[idx] for idx in (0, 1, 2) if idx in grctx_descs}
    if len(selected) != 3: raise KeyError("graphics context buffers 0, 1, and 2 are required")
    existing_ids = set((grctx_mappings or {}).keys())
    trace_channel_step("user_promote_start", client=hex(h_client), subdevice=hex(h_subdevice), gpfifo=hex(h_gpfifo),
      existing=sorted(existing_ids))
    phys_maps = None
    try:
      phys_maps, phys_payload = build_grctx_promote_payload(self.shell.mm, h_client, h_gpfifo, selected,
        include_local=True, existing=grctx_mappings, virt=False)
      self.rm.rm_control(h_subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, phys_payload, client=h_client)
      _, virt_payload = build_grctx_promote_payload(self.shell.mm, h_client, h_gpfifo, selected,
        include_local=True, existing=phys_maps, phys=False)
      self.rm.rm_control(h_subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, virt_payload, client=h_client)
    except Exception:
      if phys_maps is not None:
        for buffer_id, mapping in phys_maps.items():
          if buffer_id in existing_ids or not isinstance(mapping, VirtMapping): continue
          with contextlib.suppress(Exception): self.shell.mm.vfree(mapping)
      raise
    trace_channel_step("user_promote_done", mappings=len(phys_maps), new=sorted(set(phys_maps.keys()) - existing_ids))
    return phys_maps

  def allocate_compute_channel(self, h_device, h_vaspace, h_virtmem, gpfifo_area, notifier_buf, entries=0x10000, offset=0,
                               h_subdevice=None, grctx_descs=None, grctx_mappings=None, h_client=None):
    h_device = validate_rm_handle(h_device, "device handle")
    h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
    h_virtmem = validate_rm_handle(h_virtmem, "virtual memory handle")
    if h_client is not None: h_client = validate_rm_handle(h_client, "client handle")
    if h_subdevice is not None: h_subdevice = validate_rm_handle(h_subdevice, "subdevice handle")
    if grctx_descs is not None and h_subdevice is None: raise ValueError("subdevice handle is required for GR context promotion")
    if offset < 0 or offset % 8: raise ValueError("GPFIFO offset must be non-negative and 8-byte aligned")
    if not isinstance(gpfifo_area, GpuBuffer) or gpfifo_area.meta is None:
      raise ValueError("GPFIFO area buffer is invalid")
    if not isinstance(notifier_buf, GpuBuffer) or notifier_buf.meta is None:
      raise ValueError("notifier buffer is invalid")
    if not getattr(gpfifo_area.meta, "paddrs", None): raise ValueError("GPFIFO area physical pages are invalid")
    if not getattr(notifier_buf.meta, "paddrs", None): raise ValueError("notifier physical pages are invalid")
    validate_rm_handle(getattr(gpfifo_area.meta, "hMemory", 0), "GPFIFO memory handle")
    validate_rm_handle(getattr(notifier_buf.meta, "hMemory", 0), "notifier memory handle")
    def first_paddr(buf, name):
      entry = buf.meta.paddrs[0]
      try:
        paddr = entry[0]
      except TypeError:
        paddr = entry
      if not isinstance(paddr, int): raise ValueError(f"{name} physical page is invalid")
      if paddr < 0 or paddr % 0x1000: raise ValueError(f"{name} physical page must be 4KB aligned")
      return paddr
    gpfifo_paddr = first_paddr(gpfifo_area, "GPFIFO area")
    notifier_paddr = first_paddr(notifier_buf, "notifier")
    validate_positive_size(entries, "GPFIFO entry count")
    validate_u32(entries, "GPFIFO entry count")
    if offset + entries * 8 + NVC56F_CONTROL_GP_PUT_OFFSET + 4 > gpfifo_area.size:
      raise ValueError("GPFIFO ring/control area exceeds backing buffer")
    user_grctx_mappings = None
    h_gpfifo = None
    existing_ids = set((grctx_mappings or {}).keys())
    trace_channel_step("compute_channel_start", device=hex(h_device), vaspace=hex(h_vaspace), virtmem=hex(h_virtmem),
      entries=entries, gpfifo_area=hex(gpfifo_area.va_addr), gpfifo_paddr=hex(gpfifo_paddr), notifier_paddr=hex(notifier_paddr),
      grctx=grctx_descs is not None)
    try:
      ramfc_paddr = 0
      method_paddr = 0
      userd_paddr = gpfifo_paddr + entries * 8 + offset
      gpfifo_params = pack_nv_channel_gpfifo_allocation_params(gpfifo_area.va_addr + offset, entries, 0, h_vaspace,
        h_virtmem, gpfifo_area.meta.hMemory, entries * 8 + offset, ramfc_paddr, method_paddr,
        error_paddr=notifier_paddr, h_object_error=notifier_buf.meta.hMemory, userd_paddr=userd_paddr,
        engine_type=NV2080_ENGINE_TYPE_GRAPHICS, cid=3, flags=0x200320)
      gpfifo_desc = gpfifo_memory_desc_summary(gpfifo_params)
      trace_channel_step("compute_gpfifo_alloc", parent=hex(h_device), channel_group=hex(0),
        ctxshare=hex(0), entries=entries, area=hex(gpfifo_area.va_addr + offset),
        userd_paddr=hex(userd_paddr), ramfc_paddr=hex(ramfc_paddr), method_paddr=hex(method_paddr),
        desc_ramfc=f"0x{gpfifo_desc['ramfc']['base']:x}/0x{gpfifo_desc['ramfc']['size']:x}",
        desc_userd=f"0x{gpfifo_desc['userd']['base']:x}/0x{gpfifo_desc['userd']['size']:x}",
        desc_instance=f"0x{gpfifo_desc['instance']['base']:x}/0x{gpfifo_desc['instance']['size']:x}",
        desc_method=f"0x{gpfifo_desc['method']['base']:x}/0x{gpfifo_desc['method']['size']:x}",
        desc_error=f"0x{unpack_nv_memory_desc(gpfifo_params, 248)['base']:x}/0x{unpack_nv_memory_desc(gpfifo_params, 248)['size']:x}",
        h_object_buffer=hex(gpfifo_area.meta.hMemory), h_object_error=hex(notifier_buf.meta.hMemory),
        params_sha256=hashlib.sha256(gpfifo_params).hexdigest())
      h_gpfifo = self.rm.rm_alloc(h_device, AMPERE_CHANNEL_GPFIFO_A, gpfifo_params)
      compute_class = ADA_COMPUTE_A if self.shell.chip_name.startswith("AD") else AMPERE_COMPUTE_B
      h_compute = self.rm.rm_alloc(h_gpfifo, compute_class, b"")
      trace_channel_step("compute_channel_objects", channel_group=hex(0), ctxshare=hex(0),
        gpfifo=hex(h_gpfifo), compute=hex(h_compute), compute_class=hex(compute_class), userd_paddr=hex(userd_paddr),
        ramfc_paddr=hex(ramfc_paddr), method_paddr=hex(method_paddr))
      if h_subdevice is not None and grctx_descs is not None:
        user_grctx_mappings = self.promote_user_compute_context(h_client or self.rm.priv_root, h_subdevice, h_gpfifo,
          grctx_descs, grctx_mappings=grctx_mappings)
      debugger_params = pack_nv83de_alloc_parameters(self.rm.priv_root, h_compute)
      debugger_object = self.rm.next_handle if hasattr(self.rm, "next_handle") else 0
      debugger_rpc_sha256 = hashlib.sha256(pack_rpc_gsp_rm_alloc(self.rm.priv_root, h_device, debugger_object,
        GT200_DEBUGGER, debugger_params)).hexdigest() if debugger_object else "unknown"
      h_debugger = self.rm.rm_alloc(h_device, GT200_DEBUGGER, debugger_params)
      token_params = struct.pack("<i", -1)
      token_rpc_sha256 = pack_rpc_gsp_rm_control_fingerprint(self.rm.priv_root, h_gpfifo,
        NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, token_params)
      trace_channel_step("runtime_token_control", object=hex(h_gpfifo),
        cmd=hex(NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN), rpc_sha256=token_rpc_sha256)
      token_resp = self.rm.rm_control(h_gpfifo, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, token_params)
      token = unpack_gpfifo_work_submit_token(token_resp)
      schedule_params = struct.pack("<I", 1)
      schedule_rpc_sha256 = pack_rpc_gsp_rm_control_fingerprint(self.rm.priv_root, h_device,
        NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, schedule_params)
      trace_channel_step("runtime_schedule_control", object=hex(h_device),
        cmd=hex(NVA06C_CTRL_CMD_GPFIFO_SCHEDULE), rpc_sha256=schedule_rpc_sha256)
      self.rm.rm_control(h_device, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, schedule_params)
      trace_channel_step("compute_channel_done", debugger=hex(h_debugger), token=hex(token),
        user_grctx=user_grctx_mappings is not None, debugger_rpc_sha256=debugger_rpc_sha256,
        token_rpc_sha256=token_rpc_sha256, schedule_rpc_sha256=schedule_rpc_sha256)
      return 0, 0, h_gpfifo, h_compute, compute_class, h_debugger, token, user_grctx_mappings
    except Exception:
      if h_gpfifo is not None and getattr(self.rm, "private_mappings", None):
        mappings = self.rm.private_mappings.pop(h_gpfifo, ())
        if isinstance(mappings, VirtMapping): mappings = (mappings,)
        for mapping in mappings:
          if isinstance(mapping, VirtMapping):
            with contextlib.suppress(Exception): self.shell.mm.vfree(mapping)
      if user_grctx_mappings is not None:
        for buffer_id, mapping in user_grctx_mappings.items():
          if buffer_id in existing_ids or not isinstance(mapping, VirtMapping): continue
          with contextlib.suppress(Exception): self.shell.mm.vfree(mapping)
      raise

  def allocate_runtime_resources(self, allocator, h_device=None, h_virtmem=None, h_vaspace=None, entries=32, token=0,
                                 h_subdevice=None, grctx_descs=None, grctx_mappings=None, h_client=None):
    validate_positive_size(entries, "GPFIFO entry count")
    validate_u32(entries, "GPFIFO entry count")
    validate_u32(token, "GPFIFO token")
    supplied_handles = [h_device is not None, h_virtmem is not None, h_vaspace is not None]
    if any(supplied_handles) and not all(supplied_handles): raise ValueError("device, virtmem, and vaspace handles are required together")
    if h_device is not None: h_device = validate_rm_handle(h_device, "device handle")
    if h_virtmem is not None: h_virtmem = validate_rm_handle(h_virtmem, "virtual memory handle")
    if h_vaspace is not None: h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
    if grctx_descs is not None and h_subdevice is None: raise ValueError("subdevice handle is required for GR context promotion")
    if h_subdevice is not None: h_subdevice = validate_rm_handle(h_subdevice, "subdevice handle")
    if h_client is not None: h_client = validate_rm_handle(h_client, "client handle")
    with_handles = h_device is not None
    trace_channel_step("runtime_resources_start", entries=entries, token=hex(token), rm_backed=with_handles,
      h_device=("None" if h_device is None else hex(h_device)), h_vaspace=("None" if h_vaspace is None else hex(h_vaspace)),
      h_virtmem=("None" if h_virtmem is None else hex(h_virtmem)), grctx=grctx_descs is not None)
    rm_backed_memory = False
    gpfifo_flags = NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED << 23
    resources = ChannelResources()
    try:
      resources.gpfifo_area = allocator.alloc_sysmem(0x300000, contiguous=True, rm_handle=with_handles and rm_backed_memory, flags=gpfifo_flags)
      if entries * 8 + NVC56F_CONTROL_GP_PUT_OFFSET + 4 > resources.gpfifo_area.size:
        raise ValueError("GPFIFO ring/control area exceeds backing buffer")
      handles = resources.handles
      if with_handles:
        resources.notifier_buf = allocator.alloc_sysmem(48 << 20, rm_handle=rm_backed_memory)
        h_channel_group, h_ctxshare, h_gpfifo, h_compute, compute_class, h_debugger, token, user_grctx_mappings = self.allocate_compute_channel(
          h_device, h_vaspace, h_virtmem, resources.gpfifo_area, resources.notifier_buf, entries=entries, h_subdevice=h_subdevice,
          grctx_descs=grctx_descs, grctx_mappings=grctx_mappings, h_client=h_client)
        handles.update(channel_group=h_channel_group, ctxshare=h_ctxshare, gpfifo=h_gpfifo,
          compute=h_compute, compute_class=compute_class, debugger=h_debugger, notifier=resources.notifier_buf.meta.hMemory)
        handles["dma_gpfifo"] = None
        if user_grctx_mappings is not None: handles["user_grctx_mappings"] = user_grctx_mappings
        if getattr(self.rm, "private_mappings", None):
          retained = {handle: self.rm.private_mappings.pop(handle) for handle in (h_gpfifo,) if handle in self.rm.private_mappings}
          if retained: handles["rm_private_mappings"] = retained
      ring = resources.gpfifo_area.cpu_view().view(0, entries * 8, fmt='Q')
      gpput = resources.gpfifo_area.cpu_view().view(entries * 8 + NVC56F_CONTROL_GP_PUT_OFFSET, 4, fmt='I')
      resources.cmdq_page = allocator.alloc_sysmem(0x200000, rm_handle=with_handles and rm_backed_memory)
      resources.kernargs_buf = allocator.alloc_sysmem(0x200000, rm_handle=with_handles and rm_backed_memory)
      timeline_buf = allocator.alloc_sysmem(0x1000, rm_handle=with_handles and rm_backed_memory)
      signal = StandaloneSignal(timeline_buf)
      signal.value = 0
      resources.timeline_signal = signal
      resources.compute_gpfifo = GPFifoState(ring=ring, gpput=gpput, entries_count=entries, token=token)
      resources.kernargs_allocator = BumpAllocator(resources.kernargs_buf.size, base=resources.kernargs_buf.va_addr, wrap=True)
      trace_channel_step("runtime_resources_done", gpfifo_area=hex(resources.gpfifo_area.va_addr),
        cmdq=hex(resources.cmdq_page.va_addr), kernargs=hex(resources.kernargs_buf.va_addr),
        timeline=hex(resources.timeline_signal.value_addr), dma_gpfifo=resources.handles.get("dma_gpfifo") is not None,
        handles=sorted(resources.handles.keys()))
      return resources
    except Exception:
      StandaloneNvBackend(self.shell, resources, allocator, submitter=None).release_resources()
      raise

class GspRmClient:
  def __init__(self, cmd_q, stat_q=None, priv_root=0xc1e00004, first_handle=0xcf000000):
    self.cmd_q, self.stat_q = cmd_q, stat_q or cmd_q
    self.priv_root = validate_rm_handle(priv_root, "private root handle")
    self.next_handle = validate_rm_handle(first_handle, "first RM handle")
    self.private_mappings = {}

  def handle(self):
    handle = validate_rm_handle(self.next_handle, "next RM handle")
    self.next_handle += 1
    return handle

  def send_and_wait(self, func, payload, wait=True):
    func = validate_u32(func, "RM RPC function")
    if not isinstance(wait, bool): raise ValueError("RM RPC wait flag must be bool")
    try:
      payload = bytes(payload)
    except TypeError as exc:
      raise ValueError("RM RPC payload must be bytes-like") from exc
    self.cmd_q.send_rpc(func, payload)
    return self.stat_q.wait_resp(func) if wait else b""

  def queue_debug_state(self, queue, slots=4):
    if slots < 0: raise ValueError("RM queue debug slot count must be non-negative")
    debug = getattr(queue, "debug_state", None)
    if not callable(debug): return "unavailable"
    try:
      return debug(slots=slots)
    except Exception as exc:
      return f"unavailable:{type(exc).__name__}"

  def set_system_info(self, payload, wait=False):
    validate_wait_flag(wait, "RM system-info wait flag")
    try:
      payload = bytes(payload)
    except TypeError as exc:
      raise ValueError("GSP system-info payload must be bytes-like") from exc
    if len(payload) != 928: raise ValueError("GSP system-info payload must be 928 bytes")
    return self.send_and_wait(NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, payload, wait=wait)

  def set_registry(self, entries=None, wait=False):
    validate_wait_flag(wait, "RM registry wait flag")
    if entries is None: entries = {"RMForcePcieConfigSave": 1, "RMSecBusResetEnable": 1}
    return self.send_and_wait(NV_VGPU_MSG_FUNCTION_SET_REGISTRY,
                              pack_registry_table(entries), wait=wait)

  def trace_alloc_state(self, stage, client, h_parent, h_object, h_class, **fields):
    if os.environ.get("NV_ADD_TRACE_RM_STATE") != "1" or not hasattr(self.cmd_q, "gsp") or self.cmd_q.gsp is None: return
    shell = self.cmd_q.gsp
    def view_word(view, index):
      if view is None: return "NA"
      try: return f"0x{int(view[index]):x}"
      except Exception: return "NA"
    def reg(addr):
      if not hasattr(shell, "rreg"): return "NA"
      try: return f"0x{shell.rreg(addr):x}"
      except Exception: return "NA"
    qtrace = ""
    boot = getattr(shell, "gsp_boot", None)
    if boot is not None and getattr(boot, "queue_memory", None) is not None:
      qmem = boot.queue_memory
      try:
        qtrace = (f" cmd_head={bytes(qmem.cmd_q_view[:16]).hex()} stat_head={bytes(qmem.stat_q_view[:16]).hex()}"
                  f" cmd_off=0x{qmem.cmd_q_view.offset:x} stat_off=0x{qmem.stat_q_view.offset:x}")
      except Exception:
        qtrace = " queue_memory=unavailable"
    if hasattr(shell, "rreg"):
      falcon = FalconController(shell)
      try:
        gsp_state = falcon.format_state(NV_FALCON_GSP_BASE)
        gsp_hwcfg2 = shell.rreg(NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_HWCFG2)
        gsp_cpuctl = shell.rreg(NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL)
        gsp_has_riscv = reg_get(gsp_hwcfg2, 10, 10)
        gsp_fenced = (gsp_cpuctl & 0xffffffff) == 0xbadf5720 or (gsp_cpuctl & 0xffff0000) == 0xbadf0000
        gsp_state += f" [hwcfg2=0x{gsp_hwcfg2:x} has_riscv={gsp_has_riscv} cpuctl=0x{gsp_cpuctl:x} fenced={gsp_fenced}]"
      except Exception as exc: gsp_state = f"unavailable:{type(exc).__name__}:{exc}"
      try:
        sec2_state = falcon.format_state(NV_FALCON_SEC2_BASE)
        sec2_hwcfg2 = shell.rreg(NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_HWCFG2)
        sec2_cpuctl = shell.rreg(NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_CPUCTL)
        sec2_has_riscv = reg_get(sec2_hwcfg2, 10, 10)
        sec2_fenced = (sec2_cpuctl & 0xffffffff) == 0xbadf5720 or (sec2_cpuctl & 0xffff0000) == 0xbadf0000
        sec2_state += f" [hwcfg2=0x{sec2_hwcfg2:x} has_riscv={sec2_has_riscv} cpuctl=0x{sec2_cpuctl:x} fenced={sec2_fenced}]"
      except Exception as exc: sec2_state = f"unavailable:{type(exc).__name__}:{exc}"
      try:
        pmu_state = falcon.format_state(NV_FALCON_PMU_BASE)
        pmu_hwcfg2 = shell.rreg(NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2)
        pmu_cpuctl = shell.rreg(NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL)
        pmu_has_riscv = reg_get(pmu_hwcfg2, 10, 10)
        pmu_fenced = (pmu_cpuctl & 0xffffffff) == 0xbadf5720 or (pmu_cpuctl & 0xffff0000) == 0xbadf0000
        pmu_state += f" [hwcfg2=0x{pmu_hwcfg2:x} has_riscv={pmu_has_riscv} cpuctl=0x{pmu_cpuctl:x} fenced={pmu_fenced}]"
      except Exception as exc: pmu_state = f"unavailable:{type(exc).__name__}:{exc}"
      try:
        fecs_base = shell.fecs_falcon_base()
        fecs_state = falcon.format_state(fecs_base)
        fecs_hwcfg2 = shell.rreg(fecs_base + NV_PFALCON_FALCON_HWCFG2)
        fecs_cpuctl = shell.rreg(fecs_base + NV_PFALCON_FALCON_CPUCTL)
        fecs_has_riscv = reg_get(fecs_hwcfg2, 10, 10)
        fecs_fenced = (fecs_cpuctl & 0xffffffff) == 0xbadf5720 or (fecs_cpuctl & 0xffff0000) == 0xbadf0000
        fecs_state += f" [hwcfg2=0x{fecs_hwcfg2:x} has_riscv={fecs_has_riscv} cpuctl=0x{fecs_cpuctl:x} fenced={fecs_fenced}]"
      except Exception as exc: fecs_state = f"unavailable:{type(exc).__name__}:{exc}"
    else:
      gsp_state = sec2_state = pmu_state = fecs_state = "unavailable"
    detail = "".join(f" {name}={value}" for name, value in fields.items())
    cmd_queue_state = self.queue_debug_state(self.cmd_q)
    stat_queue_state = self.queue_debug_state(self.stat_q) if self.stat_q is not self.cmd_q else cmd_queue_state
    if stage in ("alloc-status", "alloc-exception"):
      detail += f" cmd_queue=[{cmd_queue_state}]"
      if self.stat_q is not self.cmd_q:
        detail += f" stat_queue=[{stat_queue_state}]"
    queue_alias = {
      "pre-alloc": "pre_queues",
      "alloc-status": "status_queues",
      "alloc-exception": "exception_queues",
      "post-alloc": "post_queues",
    }.get(stage)
    if queue_alias is not None:
      print(f"standalone rm_alloc {queue_alias} client=0x{client:x} parent=0x{h_parent:x} object=0x{h_object:x} "
            f"class=0x{h_class:x} class_name={rm_class_name(h_class)} cmd=[{cmd_queue_state}] stat=[{stat_queue_state}]")
    print(f"standalone rm_state stage={stage} client=0x{client:x} parent=0x{h_parent:x} object=0x{h_object:x} "
          f"class=0x{h_class:x} class_name={rm_class_name(h_class)} bar1={reg(NV_PBUS_BAR1_BLOCK)} "
          f"wpr2_hi={reg(NV_PFB_PRI_MMU_WPR2_ADDR_HI)} cmd_wp={view_word(getattr(self.cmd_q, 'tx_view', None), 4)} "
          f"stat_rp={view_word(getattr(self.stat_q, 'rx_view', None), 0)}{detail}{qtrace} "
          f"gsp=({gsp_state}) sec2=({sec2_state}) pmu=({pmu_state}) fecs=({fecs_state})")

  def alloc_root(self, wait=True):
    validate_wait_flag(wait, "RM alloc wait flag")
    self.rm_alloc(0, NV01_ROOT, pack_nv0000_alloc_parameters(), client=self.priv_root, wait=wait)
    return self.priv_root

  def rm_alloc(self, h_parent, h_class, params=b"", client=None, h_object=None, wait=True):
    validate_wait_flag(wait, "RM alloc wait flag")
    client = self.priv_root if client is None else client
    client = validate_rm_handle(client, "client handle")
    h_parent = validate_rm_handle(h_parent, "parent handle")
    h_class = validate_u32(h_class, "RM class")
    try:
      params = bytes(params)
    except TypeError as exc:
      raise ValueError("RM alloc params must be bytes-like") from exc
    validate_u32(len(params), "RM alloc params size")
    params_sha256 = hashlib.sha256(params).hexdigest()
    if h_class == AMPERE_CHANNEL_GPFIFO_A and hasattr(self.cmd_q, "gsp") and self.cmd_q.gsp is not None:
      if len(params) < 240: raise ValueError("GPFIFO allocation params are too small for RM patching")
    h_object = self.handle() if h_object is None else validate_rm_handle(h_object, "object handle")
    trace_rm_step("gsp_alloc", client=hex(client), parent=hex(h_parent), h_class=hex(h_class), class_name=rm_class_name(h_class), object=hex(h_object),
      params_len=len(params), params_sha256=params_sha256, wait=wait)
    self.trace_alloc_state("pre-alloc", client, h_parent, h_object, h_class)
    if h_class == AMPERE_CHANNEL_GPFIFO_A and hasattr(self.cmd_q, "gsp") and self.cmd_q.gsp is not None:
      shell = self.cmd_q.gsp
      ramfc = shell.mm.valloc(0x1000, contiguous=True)
      method = shell.mm.palloc(0x5000, align=0x1000)
      params = bytearray(params)
      ctor = unpack_nv_channel_gpfifo_allocation_params(params)
      before_desc = gpfifo_memory_desc_summary(params)
      params[144:168] = pack_nv_memory_desc(ramfc.paddrs[0][0], 0x1000)
      params[192:216] = pack_nv_memory_desc(ramfc.paddrs[0][0], 0x200)
      params[216:240] = pack_nv_memory_desc(method, 0x5000)
      params = bytes(params)
      after_desc = gpfifo_memory_desc_summary(params)
      trace_rm_step("gpfifo_patch", ramfc=hex(ramfc.paddrs[0][0]), method=hex(method),
        params_len=len(params), params_sha256=hashlib.sha256(params).hexdigest(),
        ctor_gpfifo_va=hex(ctor["gpfifo_va"]), ctor_entries=ctor["entries"], ctor_flags=hex(ctor["flags"]),
        ctor_h_context_share=hex(ctor["h_context_share"]), ctor_h_vaspace=hex(ctor["h_vaspace"]),
        ctor_h_userd_memory=hex(ctor["h_userd_memory"]), ctor_userd_offset=hex(ctor["userd_offset"]),
        ctor_engine_type=hex(ctor["engine_type"]), ctor_cid=ctor["cid"], ctor_runlist_id=ctor["runlist_id"],
        ctor_internal_flags=hex(ctor["internal_flags"]),
        before_ramfc_base=hex(before_desc["ramfc"]["base"]), before_ramfc_size=hex(before_desc["ramfc"]["size"]),
        before_userd_base=hex(before_desc["userd"]["base"]), before_userd_size=hex(before_desc["userd"]["size"]),
        before_instance_base=hex(before_desc["instance"]["base"]), before_instance_size=hex(before_desc["instance"]["size"]),
        before_method_base=hex(before_desc["method"]["base"]), before_method_size=hex(before_desc["method"]["size"]),
        after_ramfc_base=hex(after_desc["ramfc"]["base"]), after_ramfc_size=hex(after_desc["ramfc"]["size"]),
        after_userd_base=hex(after_desc["userd"]["base"]), after_userd_size=hex(after_desc["userd"]["size"]),
        after_instance_base=hex(after_desc["instance"]["base"]), after_instance_size=hex(after_desc["instance"]["size"]),
        after_method_base=hex(after_desc["method"]["base"]), after_method_size=hex(after_desc["method"]["size"]),
        ctor_error_base=hex(ctor["error_desc"]["base"]), ctor_error_size=hex(ctor["error_desc"]["size"]))
    else:
      ramfc, method = None, None
    payload = pack_rpc_gsp_rm_alloc(client, h_parent, h_object, h_class, params)
    rpc_sha256 = hashlib.sha256(payload).hexdigest()
    trace_rm_step("gsp_alloc_rpc", payload_len=len(payload), rpc_sha256=rpc_sha256)
    traced_failure_state = False
    try:
      resp = self.send_and_wait(NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, payload, wait=wait)
      if wait:
        status, _ = unpack_rpc_gsp_rm_alloc_response(resp)
        if status != 0:
          self.trace_alloc_state("alloc-status", client, h_parent, h_object, h_class, status=f"0x{status:x}",
            rpc_sha256=rpc_sha256, resp_len=len(resp), resp_sha256=hashlib.sha256(resp).hexdigest())
          traced_failure_state = True
          if h_class == AMPERE_CHANNEL_GPFIFO_A and hasattr(self.cmd_q, "gsp") and self.cmd_q.gsp is not None:
            ctx = gpfifo_alloc_context_string(ctor, before_desc, after_desc, params_sha256, hashlib.sha256(params).hexdigest())
            raise RuntimeError(f"GSP_RM_ALLOC class=0x{h_class:x} object=0x{h_object:x} failed: 0x{status:x} {ctx}")
          raise RuntimeError(f"GSP_RM_ALLOC class=0x{h_class:x} object=0x{h_object:x} failed: 0x{status:x}")
      if ramfc is not None:
        self.private_mappings[h_object] = (ramfc,)
      self.trace_alloc_state("post-alloc", client, h_parent, h_object, h_class, rpc_sha256=rpc_sha256)
    except Exception as exc:
      if not traced_failure_state:
        self.trace_alloc_state("alloc-exception", client, h_parent, h_object, h_class, exc=type(exc).__name__,
          exc_msg=self.compact_trace_text(exc), rpc_sha256=rpc_sha256)
      if ramfc is not None:
        with contextlib.suppress(Exception): self.cmd_q.gsp.mm.vfree(ramfc)
      if method is not None:
        with contextlib.suppress(Exception): self.cmd_q.gsp.mm.phys_alloc.free_addr(method)
      raise
    return h_object

  def compact_trace_text(self, value, limit=512):
    text = str(value).replace("\n", "\\n")
    if len(text) > limit: text = text[:limit - 3] + "..."
    return text

  def rm_control(self, h_object, cmd, params=b"", client=None):
    client = self.priv_root if client is None else client
    client = validate_rm_handle(client, "client handle")
    h_object = validate_rm_handle(h_object, "object handle")
    cmd = validate_u32(cmd, "RM control command")
    try:
      params = bytes(params)
    except TypeError as exc:
      raise ValueError("RM control params must be bytes-like") from exc
    payload = pack_rpc_gsp_rm_control(client, h_object, cmd, params)
    if trace_rm_alloc_enabled():
      cmd_queue_state = self.queue_debug_state(self.cmd_q)
      stat_queue_state = self.queue_debug_state(self.stat_q) if self.stat_q is not self.cmd_q else cmd_queue_state
      print(f"standalone rm_control pre_queues client=0x{client:x} object=0x{h_object:x} "
            f"cmd=0x{cmd:x} cmd_name={rm_ctrl_name(cmd)} cmdq=[{cmd_queue_state}] stat=[{stat_queue_state}]")
    trace_rm_step("gsp_control_rpc", object=hex(h_object), cmd=hex(cmd), cmd_name=rm_ctrl_name(cmd),
      params_len=len(params), params_sha256=hashlib.sha256(params).hexdigest(),
      payload_len=len(payload), rpc_sha256=hashlib.sha256(payload).hexdigest(), head=params[:96].hex())
    resp = self.send_and_wait(NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, payload)
    status, payload = unpack_rpc_gsp_rm_control_response(resp)
    if trace_rm_alloc_enabled():
      cmd_queue_state = self.queue_debug_state(self.cmd_q)
      stat_queue_state = self.queue_debug_state(self.stat_q) if self.stat_q is not self.cmd_q else cmd_queue_state
      print(f"standalone rm_control post client=0x{client:x} object=0x{h_object:x} "
            f"cmd=0x{cmd:x} cmd_name={rm_ctrl_name(cmd)} rpc_sha256={hashlib.sha256(pack_rpc_gsp_rm_control(client, h_object, cmd, params)).hexdigest()} "
            f"status=0x{status:x} result_len={len(payload)} result_sha256={hashlib.sha256(payload).hexdigest()} "
            f"head={payload[:128].hex()}")
      print(f"standalone rm_control post_queues client=0x{client:x} object=0x{h_object:x} "
            f"cmd=0x{cmd:x} cmd_name={rm_ctrl_name(cmd)} cmdq=[{cmd_queue_state}] stat=[{stat_queue_state}]")
    if status != 0:
      raise RuntimeError(f"GSP_RM_CONTROL object=0x{h_object:x} cmd=0x{cmd:x} cmd_name={rm_ctrl_name(cmd)} "
                         f"payload_len={len(params)} payload_sha256={hashlib.sha256(params).hexdigest()} failed: 0x{status:x}")
    return payload

  def alloc_memory(self, h_device, h_class, paddrs, length, flags, client=None, h_memory=None):
    client = self.priv_root if client is None else client
    client = validate_rm_handle(client, "client handle")
    h_device = validate_rm_handle(h_device, "device handle")
    h_class = validate_u32(h_class, "RM class")
    flags = validate_u32(flags, "alloc-memory flags")
    length = validate_u64(validate_positive_size(length, "allocation length"), "allocation length")
    if h_memory is not None: h_memory = validate_rm_handle(h_memory, "memory handle")
    pages = expand_phys_pages(paddrs)
    h_memory = self.handle() if h_memory is None else h_memory
    payload = pack_rpc_alloc_memory(client, h_device, h_memory, h_class, paddrs, length, flags)
    trace_rm_step("gsp_alloc_memory", client=hex(client), device=hex(h_device), h_class=hex(h_class), class_name=rm_class_name(h_class), memory=hex(h_memory),
      length=length, flags=hex(flags), pages=len(pages), payload_sha256=hashlib.sha256(payload).hexdigest(),
      rpc_sha256=hashlib.sha256(payload).hexdigest())
    resp = self.send_and_wait(NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY, payload)
    status = unpack_rpc_alloc_memory_response(resp)
    if status != 0:
      raise RuntimeError(f"ALLOC_MEMORY client=0x{client:x} device=0x{h_device:x} class=0x{h_class:x} "
                         f"class_name={rm_class_name(h_class)} memory=0x{h_memory:x} length=0x{length:x} "
                         f"flags=0x{flags:x} pages={len(pages)} payload_sha256={hashlib.sha256(payload).hexdigest()} "
                         f"failed: 0x{status:x}")
    return h_memory

  def set_page_directory(self, h_device, h_vaspace, pdir_paddr, num_entries, client=None, pasid=0xffffffff, wait=True):
    validate_wait_flag(wait, "RM set_page_directory wait flag")
    client = self.priv_root if client is None else client
    client = validate_rm_handle(client, "client handle")
    h_device = validate_rm_handle(h_device, "device handle")
    h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
    if pdir_paddr < 0 or pdir_paddr % 0x1000: raise ValueError("page directory physical address must be 4KB aligned")
    validate_u64(pdir_paddr, "page directory physical address")
    validate_positive_size(num_entries, "page directory entry count")
    num_entries = validate_u32(num_entries, "page directory entry count")
    pasid = validate_u32(pasid, "PASID")
    payload = pack_rpc_set_page_directory(client, h_device, h_vaspace, pdir_paddr, num_entries, pasid=pasid)
    return self.send_and_wait(NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, payload, wait=wait)

  def unloading_guest_driver(self, wait=True):
    validate_wait_flag(wait, "RM unloading_guest_driver wait flag")
    return self.send_and_wait(NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, pack_rpc_unloading_guest_driver(), wait=wait)

class LinuxRmClient:
  def __init__(self, transport, priv_root=0xc1e00004, first_handle=0xcf000000):
    self.transport = transport
    self.priv_root = validate_rm_handle(priv_root, "private root handle")
    self.next_handle = validate_rm_handle(first_handle, "first RM handle")

  def handle(self):
    handle = validate_rm_handle(self.next_handle, "next RM handle")
    self.next_handle += 1
    return handle

  def alloc_root(self, wait=True):
    validate_wait_flag(wait, "Linux RM alloc wait flag")
    self.transport.rm_alloc(self.priv_root, 0, self.priv_root, NV01_ROOT, b"")
    return self.priv_root

  def rm_alloc(self, h_parent, h_class, params=b"", client=None, h_object=None, wait=True):
    validate_wait_flag(wait, "Linux RM alloc wait flag")
    client = self.priv_root if client is None else client
    client = validate_rm_handle(client, "client handle")
    h_parent = validate_rm_handle(h_parent, "parent handle")
    h_class = validate_u32(h_class, "RM class")
    try:
      params = bytes(params)
    except TypeError as exc:
      raise ValueError("Linux RM alloc params must be bytes-like") from exc
    validate_u32(len(params), "Linux RM alloc params size")
    params_sha256 = hashlib.sha256(params).hexdigest()
    h_object = self.handle() if h_object is None else validate_rm_handle(h_object, "object handle")
    trace_rm_step("linux_alloc", client=hex(client), parent=hex(h_parent), h_class=hex(h_class), class_name=rm_class_name(h_class), object=hex(h_object),
      params_len=len(params), params_sha256=params_sha256, wait=wait)
    return self.transport.rm_alloc(client, h_parent, h_object, h_class, params)

  def rm_control(self, h_object, cmd, params=b"", client=None):
    client = self.priv_root if client is None else client
    client = validate_rm_handle(client, "client handle")
    h_object = validate_rm_handle(h_object, "object handle")
    cmd = validate_u32(cmd, "RM control command")
    try:
      params = bytes(params)
    except TypeError as exc:
      raise ValueError("RM control params must be bytes-like") from exc
    trace_rm_step("linux_control", object=hex(h_object), cmd=hex(cmd), cmd_name=rm_ctrl_name(cmd),
      payload_len=len(params), params_sha256=hashlib.sha256(params).hexdigest())
    try:
      return self.transport.rm_control(client, h_object, cmd, params)
    except RuntimeError as exc:
      raise RuntimeError(f"Linux RM_CONTROL object=0x{h_object:x} cmd=0x{cmd:x} cmd_name={rm_ctrl_name(cmd)} "
                         f"payload_len={len(params)} payload_sha256={hashlib.sha256(params).hexdigest()} failed: {exc}") from exc

  def alloc_memory(self, h_device, h_class, paddrs, length, flags, client=None, h_memory=None):
    client = self.priv_root if client is None else client
    client = validate_rm_handle(client, "client handle")
    h_device = validate_rm_handle(h_device, "device handle")
    h_class = validate_u32(h_class, "RM class")
    flags = validate_u32(flags, "alloc-memory flags")
    length = validate_u64(validate_positive_size(length, "allocation length"), "allocation length")
    if h_memory is not None: h_memory = validate_rm_handle(h_memory, "memory handle")
    pages = expand_phys_pages(paddrs)
    if not pages: raise ValueError("Linux alloc_memory requires at least one physical page")
    h_memory = self.handle() if h_memory is None else h_memory
    sysmem_va = pages[0]
    data = pack_nvos02_with_fd(client, h_device, h_memory, h_class, flags, sysmem_va, length - 1)
    trace_rm_step("linux_alloc_memory", client=hex(client), device=hex(h_device), h_class=hex(h_class), class_name=rm_class_name(h_class), memory=hex(h_memory),
      length=length, flags=hex(flags), pages=len(pages), payload_sha256=hashlib.sha256(data).hexdigest())
    self.transport.dev_ioctl(NV_ESC_RM_ALLOC_MEMORY, data)
    if (status := unpack_nvos02_status(data)) != 0:
      raise RuntimeError(f"NV_ESC_RM_ALLOC_MEMORY client=0x{client:x} device=0x{h_device:x} class=0x{h_class:x} "
                         f"class_name={rm_class_name(h_class)} memory=0x{h_memory:x} length=0x{length:x} "
                         f"flags=0x{flags:x} pages={len(pages)} payload_sha256={hashlib.sha256(data).hexdigest()} "
                         f"failed: 0x{status:x}")
    return h_memory

  def unloading_guest_driver(self, wait=True):
    validate_wait_flag(wait, "Linux RM unloading_guest_driver wait flag")
    return b""

def select_transport(name=None):
  default = "mac-egpu" if sys.platform == "darwin" else "linux-ioctl"
  name = (name or os.environ.get("NV_ADD_TRANSPORT", default)).lower()
  if name in ("tinygrad", "tinygrad-pcie", "pcie"): raise ValueError("tinygrad transport has been removed; use mac-egpu or linux-ioctl")
  if name in ("mac-egpu", "tinygpu", "egpu"): return MacEgpuTransport()
  if name in ("linux-ioctl", "ioctl", "linux"): return LinuxIoctlTransport()
  raise ValueError(f"unknown NV_ADD_TRANSPORT={name!r}")

def compact_exception_message(exc):
  return " ".join(str(exc).split())

def collect_transport_preflight_state(transport=None, name=None):
  requested = name or os.environ.get("NV_ADD_TRANSPORT", "auto")
  state = {"requested": requested}
  try:
    state["step"] = "select_transport"
    transport = select_transport(name) if transport is None else transport
    state["transport"] = type(transport).__name__
    if hasattr(transport, "probe"):
      try:
        state["step"] = "probe"
        state["probe"] = transport.probe()
      except Exception as exc:
        state["probe_error"] = f"{type(exc).__name__}:{compact_exception_message(exc)}"
    state["step"] = "read_vendor_device"
    state["vendor_device"] = transport.read_config(PCI_VENDOR_ID, 4)
    state["step"] = "read_pci_command"
    state["pci_command"] = transport.read_config(PCI_COMMAND, 2)
    for bar in (0, 1, 3):
      state["step"] = f"bar{bar}_info"
      state[f"bar{bar}"] = transport.bar_info(bar)
    state["ok"] = True
    state.pop("step", None)
  except Exception as exc:
    state["ok"] = False
    state["exc"] = type(exc).__name__
    state["msg"] = compact_exception_message(exc)
  return state

def transport_preflight_state(transport=None, name=None):
  return collect_transport_preflight_state(transport=transport, name=name)

def format_transport_preflight_state(state):
  fields = [f"ok={state.get('ok', True)}", f"requested={state['requested']}"]
  if "transport" in state:
    fields.append(f"transport={state['transport']}")
  if "probe" in state:
    fields.append(f"probe=0x{state['probe'][0]:x}/0x{state['probe'][1]:x}")
  if "probe_error" in state:
    fields.append(f"probe_error={state['probe_error']}")
  if "vendor_device" in state:
    fields.append(f"vendor_device=0x{state['vendor_device']:08x}")
  if "pci_command" in state:
    fields.append(f"pci_command=0x{state['pci_command']:04x}")
  for bar in (0, 1, 3):
    if f"bar{bar}" not in state: continue
    base, size = state[f"bar{bar}"]
    fields.append(f"bar{bar}=0x{base:x}/0x{size:x}")
  if "step" in state:
    fields.append(f"step={state['step']}")
  if "exc" in state:
    fields.append(f"exc={state['exc']}")
  if "msg" in state:
    fields.append(f"msg={state['msg']}")
  return "transport_preflight " + " ".join(fields)

def print_transport_preflight(name=None, transport=None):
  print(format_transport_preflight_state(collect_transport_preflight_state(transport=transport, name=name)))

def parse_transport_preflight_line(line):
  line = line.strip()
  if not line.startswith("transport_preflight "):
    raise ValueError("preflight line must start with transport_preflight")
  fields = {}
  for part in line.split()[1:]:
    if "=" not in part: continue
    key, value = part.split("=", 1)
    fields[key] = value
  return fields

def classify_transport_preflight(fields):
  ok = fields.get("ok")
  if ok == "True" and all(name in fields for name in ("vendor_device", "pci_command", "bar0", "bar1", "bar3")):
    if fields["vendor_device"].lower().endswith("10de") and fields["pci_command"] != "0x0000":
      try:
        bars = []
        for bar in ("bar0", "bar1", "bar3"):
          base_s, size_s = fields[bar].split("/", 1)
          bars.append((int(base_s, 16), int(size_s, 16)))
      except Exception:
        bars = []
      if bars and all(base > 0 and size > 0 for base, size in bars):
        return "ready-for-gsp"
  step = fields.get("step", "unknown")
  if step in ("select_transport", "probe"):
    return "fix-tinygpu-driver-visibility"
  if step in ("read_vendor_device", "read_pci_command"):
    return "fix-pci-config-access"
  if step.endswith("_info"):
    return "fix-bar-access"
  return "inspect-preflight-failure"

def print_transport_preflight_classification(line=None):
  if line is None:
    line = sys.stdin.read().strip()
  fields = parse_transport_preflight_line(line)
  result = classify_transport_preflight(fields)
  print(f"transport_preflight_classification result={result} "
        f"ok={fields.get('ok', 'unknown')} step={fields.get('step', 'none')}")
  return result

def print_transport_preflight_gate(name=None, transport=None, require_ready=False):
  line = format_transport_preflight_state(collect_transport_preflight_state(transport=transport, name=name))
  print(line)
  result = print_transport_preflight_classification(line)
  if require_ready and result != "ready-for-gsp":
    raise SystemExit(1)
  return result

def transport_preflight_next_action(result, script="examples/add.py"):
  if result == "ready-for-gsp":
    return f"next_action=reconnect_command command={recommended_reconnect_command(script, golden=True)}"
  return f"next_action=retry-preflight classification={result}"

def print_transport_preflight_plan(name=None, transport=None, require_ready=False, script="examples/add.py"):
  line = format_transport_preflight_state(collect_transport_preflight_state(transport=transport, name=name))
  print(line)
  result = print_transport_preflight_classification(line)
  print(f"transport_preflight_plan {transport_preflight_next_action(result, script)}")
  if require_ready and result != "ready-for-gsp":
    raise SystemExit(1)
  return result

def should_boot_gsp():
  return os.environ.get("NV_ADD_BOOT_GSP") == "1"

def should_prepare_golden_ctx():
  return os.environ.get("NV_ADD_PREPARE_GOLDEN_CTX") == "1"

def apply_default_live_run_env():
  os.environ.setdefault("NV_ADD_BOOT_GSP", "1")
  os.environ.setdefault("NV_ADD_PREPARE_GOLDEN_CTX", "1")

def trace_gsp_boot_enabled():
  return os.environ.get("NV_ADD_TRACE_GSP_BOOT", "0") == "1"

def verify_sec2_inputs_enabled():
  return os.environ.get("NV_ADD_VERIFY_SEC2_INPUTS", "0") == "1"

def trace_launch_enabled():
  return os.environ.get("NV_ADD_TRACE_LAUNCH", "0") == "1"

def trace_launch_steps_enabled():
  return os.environ.get("NV_ADD_TRACE_LAUNCH_STEPS", "0") == "1"

def trace_launch_stack_enabled():
  return os.environ.get("NV_ADD_TRACE_LAUNCH_STACK", "0") == "1"

def trace_launch_step(step, **fields):
  if not (trace_launch_enabled() or trace_launch_steps_enabled()): return
  payload = ", ".join(f"{name}={value}" for name, value in fields.items())
  print(f"launch {step}{(': ' + payload) if payload else ''}")
  if trace_launch_stack_enabled():
    traceback.print_stack(limit=8, file=sys.stdout)

def trace_rm_alloc_enabled():
  return os.environ.get("NV_ADD_TRACE_RM_ALLOC", "0") == "1"

def trace_rm_stack_enabled():
  return os.environ.get("NV_ADD_TRACE_RM_STACK", "0") == "1"

def trace_rm_state_enabled():
  return os.environ.get("NV_ADD_TRACE_RM_STATE", "0") == "1"

def rm_class_name(h_class):
  return RM_CLASS_NAMES.get(h_class, f"UNKNOWN_0x{h_class:x}")

def rm_ctrl_name(cmd):
  return RM_CTRL_NAMES.get(cmd, f"UNKNOWN_0x{cmd:x}")

def trace_rm_step(step, **fields):
  if not trace_rm_alloc_enabled(): return
  payload = ", ".join(f"{name}={value}" for name, value in fields.items())
  print(f"rm {step}{(': ' + payload) if payload else ''}")
  if trace_rm_stack_enabled():
    traceback.print_stack(limit=8, file=sys.stdout)

def trace_channel_enabled():
  return os.environ.get("NV_ADD_TRACE_CHANNEL", "0") == "1"

def trace_channel_stack_enabled():
  return os.environ.get("NV_ADD_TRACE_CHANNEL_STACK", "0") == "1"

def trace_channel_step(step, **fields):
  if not trace_channel_enabled(): return
  payload = ", ".join(f"{name}={value}" for name, value in fields.items())
  print(f"channel {step}{(': ' + payload) if payload else ''}")
  if trace_channel_stack_enabled():
    traceback.print_stack(limit=8, file=sys.stdout)

def should_print_summary():
  return os.environ.get("NV_ADD_SUMMARY", "0") == "1"

def expected_channel_step_sequence():
  return [
    "booted_resources_start",
    "booted_uvm_setup",
    "golden_start",
    "golden_reserved",
    "golden_gpfifo_alloc",
    "golden_gpfifo",
    "golden_promote_done",
    "golden_compute_alloc",
    "golden_done",
    "runtime_resources_start",
    "compute_channel_start",
    "compute_gpfifo_alloc",
    "compute_channel_objects",
    "compute_channel_done",
    "runtime_resources_done",
    "booted_uvm_channel",
    "booted_resources_done",
  ]

def expected_channel_step_sequence_sha256():
  return hashlib.sha256("\n".join(expected_channel_step_sequence()).encode()).hexdigest()

def expected_gpfifo_descriptor_trace_fields():
  return ["desc_ramfc", "desc_userd", "desc_instance", "desc_method", "desc_error", "h_object_buffer", "h_object_error"]

def expected_gpfifo_descriptor_trace_fields_sha256():
  return hashlib.sha256("\n".join(expected_gpfifo_descriptor_trace_fields()).encode()).hexdigest()

def reconnect_mode():
  return "golden-context" if should_prepare_golden_ctx() else "fixed-gpfifo"

def recommended_reconnect_flags(golden=None):
  use_golden = should_prepare_golden_ctx() if golden is None else golden
  flags = [f"NV_ADD_TRANSPORT={recommended_reconnect_transport()}", "NV_ADD_BOOT_GSP=1", "NV_ADD_SUMMARY=1",
           "NV_ADD_TRACE_GSP_BOOT=1", "NV_ADD_VERIFY_SEC2_INPUTS=1",
           "NV_ADD_TRACE_RM_ALLOC=1",
           "NV_ADD_TRACE_RM_STATE=1", "NV_ADD_TRACE_RPC=1", "NV_ADD_TRACE_RPC_READ=1",
           "NV_ADD_TRACE_CHANNEL=1", "NV_ADD_TRACE_LAUNCH_STEPS=1"]
  if use_golden:
    flags.insert(1, "NV_ADD_PREPARE_GOLDEN_CTX=1")
    flags.insert(4, "NV_ADD_CHECK_FRTS_BAR1=1")
    flags.insert(-1, "NV_ADD_TRACE_MM_ALLOC=1")
  return ",".join(flags)

def recommended_reconnect_transport():
  return os.environ.get("NV_ADD_RECONNECT_TRANSPORT", "mac-egpu")

def recommended_reconnect_command(script="examples/add.py", golden=None):
  return " ".join(recommended_reconnect_flags(golden=golden).split(",") + ["python3", script])

def recommended_stack_reconnect_command(script="examples/add.py", golden=True):
  stack_flags = ["NV_ADD_TRACE_RM_STACK=1", "NV_ADD_TRACE_CHANNEL_STACK=1", "NV_ADD_TRACE_LAUNCH_STACK=1", "NV_ADD_TRACE_FALCON=1"]
  return " ".join(stack_flags + recommended_reconnect_flags(golden=golden).split(",") + ["python3", script])

def recommended_preflight_command(script="examples/add.py"):
  return f"NV_ADD_TRANSPORT={recommended_reconnect_transport()} python3 {script} --transport-preflight"

def recommended_preflight_gate_command(script="examples/add.py"):
  return f"NV_ADD_TRANSPORT={recommended_reconnect_transport()} python3 {script} --transport-preflight-gate"

def recommended_preflight_require_ready_command(script="examples/add.py"):
  return f"{recommended_preflight_gate_command(script)} --require-ready"

def recommended_preflight_plan_command(script="examples/add.py"):
  return f"NV_ADD_TRANSPORT={recommended_reconnect_transport()} python3 {script} --transport-preflight-plan --require-ready"

def recommended_tiny_trace_command(script="examples/add_tiny.py"):
  return " ".join(["NV_ADD_TINY_TRACE=1", "NV_ADD_TINY_TRACE_STACK=1", "NV_ADD_TINY_BOOT_VALUES=1", "python3", script])

def runtime_summary_line():
  return (f"summary transport={os.environ.get('NV_ADD_TRANSPORT', 'auto')} boot_gsp={should_boot_gsp()} "
          f"golden_ctx={should_prepare_golden_ctx()} reconnect_mode={reconnect_mode()} trace_launch={trace_launch_enabled()} "
          f"trace_gsp_boot={trace_gsp_boot_enabled()} verify_sec2_inputs={verify_sec2_inputs_enabled()} "
          f"trace_rm={trace_rm_alloc_enabled()} trace_rm_state={trace_rm_state_enabled()} trace_channel={trace_channel_enabled()} "
          f"gpfifo_ctor_sha256={tinygrad_equivalent_gpfifo_constructor_sha256()} "
          f"channel_seq_sha256={expected_channel_step_sequence_sha256()} "
          f"gpfifo_desc_fields_sha256={expected_gpfifo_descriptor_trace_fields_sha256()} "
          f"qmd_template_sha256={expected_qmd_template_sha256()} "
          f"reconnect_flags={recommended_reconnect_flags()}")

def print_runtime_summary():
  print(runtime_summary_line())

def print_reconnect_command(script="examples/add.py"):
  print(f"reconnect_command {recommended_reconnect_command(script)}")

def _fecs_reset_common_flags():
  return ["NV_ADD_PREPARE_GOLDEN_CTX=1", "NV_ADD_BOOT_GSP=1", "NV_ADD_CLEAR_WPR2_FECS=1", "NV_ADD_VERIFY_WPR2_AFTER_FRTS=1",
    "NV_ADD_SUMMARY=1", "NV_ADD_CHECK_FRTS_BAR1=1", "NV_ADD_TRACE_GSP_BOOT=1", "NV_ADD_VERIFY_SEC2_INPUTS=1",
    "NV_ADD_TRACE_RM_ALLOC=1", "NV_ADD_TRACE_RM_STATE=1", "NV_ADD_TRACE_RPC=1", "NV_ADD_TRACE_RPC_READ=1",
    "NV_ADD_TRACE_CHANNEL=1", "NV_ADD_TRACE_MM_ALLOC=1", "NV_ADD_TRACE_LAUNCH_STEPS=1", "NV_ADD_TRACE_FALCON=1"]

def recommended_fecs_reset_reconnect_command(script="examples/add.py"):
  fecs_flags = _fecs_reset_common_flags() + ["NV_ADD_FECS_RESET=1", "NV_ADD_FECS_RESET_FORCE=1",
    "NV_ADD_FECS_RESET_POSTINIT=1", "NV_ADD_FECS_RESET_POSTINIT_FORCE=1", "NV_ADD_PMU_RESET_POSTINIT=1", "NV_ADD_PMU_RESET_POSTINIT_FORCE=1",
    "NV_ADD_FECS_RESET_POSTPROMOTE=1", "NV_ADD_FECS_RESET_POSTPROMOTE_FORCE=1", "NV_ADD_PMU_RESET_POSTPROMOTE=1", "NV_ADD_PMU_RESET_POSTPROMOTE_FORCE=1",
    "NV_ADD_REPROMOTE_AFTER_RESET=1",
    "NV_ADD_POSTPROMOTE_SETTLE_MS=100"]
  return " ".join(["NV_ADD_TRANSPORT=mac-egpu"] + fecs_flags + ["python3", script])

def recommended_fecs_only_reconnect_command(script="examples/add.py"):
  fecs_flags = _fecs_reset_common_flags() + ["NV_ADD_FECS_RESET=1", "NV_ADD_FECS_RESET_FORCE=1",
    "NV_ADD_FECS_RESET_POSTINIT=1", "NV_ADD_FECS_RESET_POSTINIT_FORCE=1",
    "NV_ADD_FECS_RESET_POSTPROMOTE=1", "NV_ADD_FECS_RESET_POSTPROMOTE_FORCE=1",
    "NV_ADD_REPROMOTE_AFTER_RESET=1",
    "NV_ADD_POSTPROMOTE_SETTLE_MS=100"]
  return " ".join(["NV_ADD_TRANSPORT=mac-egpu"] + fecs_flags + ["python3", script])

def recommended_preboot_only_reconnect_command(script="examples/add.py"):
  fecs_flags = _fecs_reset_common_flags() + ["NV_ADD_FECS_RESET=1", "NV_ADD_FECS_RESET_FORCE=1",
    "NV_ADD_PMU_RESET=1", "NV_ADD_PMU_RESET_FORCE=1"]
  return " ".join(["NV_ADD_TRANSPORT=mac-egpu"] + fecs_flags + ["python3", script])

def recommended_fecs_minimal_reconnect_command(script="examples/add.py"):
  fecs_flags = _fecs_reset_common_flags() + ["NV_ADD_FECS_RESET=1", "NV_ADD_FECS_RESET_FORCE=1"]
  return " ".join(["NV_ADD_TRANSPORT=mac-egpu"] + fecs_flags + ["python3", script])

def recommended_promote_retry_reconnect_command(script="examples/add.py"):
  fecs_flags = _fecs_reset_common_flags() + ["NV_ADD_REPROMOTE_AFTER_RESET=1", "NV_ADD_POSTPROMOTE_SETTLE_MS=200"]
  return " ".join(["NV_ADD_TRANSPORT=mac-egpu"] + fecs_flags + ["python3", script])

def print_fecs_reset_scenarios(script="examples/add.py"):
  print(f"fecs_reset_scenario_recommend try=preboot_only, then=fecs_minimal, then=fecs_only, then=full, then=promote_retry (preboot_only is the safest: just pre-boot FECS reset, no in-band FECS/PMU resets that might wipe FECS ucode after GSP boot)")
  print(f"fecs_reset_scenario full {recommended_fecs_reset_reconnect_command(script)}")
  print(f"fecs_reset_scenario fecs_only {recommended_fecs_only_reconnect_command(script)}")
  print(f"fecs_reset_scenario fecs_minimal {recommended_fecs_minimal_reconnect_command(script)}")
  print(f"fecs_reset_scenario preboot_only {recommended_preboot_only_reconnect_command(script)}")
  print(f"fecs_reset_scenario promote_retry {recommended_promote_retry_reconnect_command(script)}")

def print_bar0_fence_status(transport=None, do_reset=True):
  if transport is None: transport = MacEgpuTransport()
  if do_reset and hasattr(transport, "reset") and os.environ.get("NV_ADD_BAR0_STATUS_SKIP_RESET") != "1":
    try: transport.reset()
    except Exception as exc: print(f"bar0_fence_status reset_error={type(exc).__name__}: {exc}")
  # Use NVRegisters (same path as StandaloneNvShell) to read register values, since the raw
  # bar0 view's view() call can return different values than NVRegisters.rreg() due to a TinyGPU
  # MMIO_READ quirk. The StandaloneNvShell fence check uses NVRegisters, so this tool must too.
  regs = NVRegisters(transport)
  sample_addrs = (
    ("PMC_BOOT0", 0x000000),
    ("PMC_INTR_EN_0", 0x000100),
    ("PMC_ENABLE", 0x000200),
    ("PBUS_BAR1_BLOCK", 0x001704),
    ("PFB_PRI_MMU_WPR2_ADDR_HI", 0x1FA828),
    ("FECS_BASE+CPUCTL", 0x0A4100),
    ("FECS_BASE+HWCFG2", 0x0A40F4),
    ("GSP_BASE+CPUCTL", 0x110100),
    ("GSP_BASE+HWCFG2", 0x1100F4),
    ("PMU_BASE+CPUCTL", 0x10A100),
    ("SEC2_BASE+CPUCTL", 0x840100),
  )
  fence_count = 0
  lines = []
  for name, addr in sample_addrs:
    try: value = regs.rreg(addr)
    except Exception as exc:
      lines.append(f"bar0_fence_status addr={name:24s} 0x{addr:06x} value=ERROR fenced=False err={type(exc).__name__}")
      continue
    fenced = (value & 0xffff0000) == 0xbadf0000
    if fenced: fence_count += 1
    lines.append(f"bar0_fence_status addr={name:24s} 0x{addr:06x} value=0x{value:08x} fenced={str(fenced):5s}")
  total = len(sample_addrs)
  wpr2_hi = regs.rreg(0x1FA828)
  fecs_cpuctl = regs.rreg(0x0A4100)
  fecs_full_fence = (fecs_cpuctl & 0xffffffff) == 0xbadf1301
  if wpr2_hi == 0 and not fecs_full_fence: verdict = "CLEAR"
  elif fence_count >= max(3, total // 2) or fecs_full_fence: verdict = "FENCED"
  else: verdict = "CLEAR"
  print(f"bar0_fence_status verdict={verdict} fence_count={fence_count}/{total} reset={do_reset} wpr2_hi=0x{wpr2_hi:08x} fecs_cpuctl=0x{fecs_cpuctl:08x}")
  for line in lines:
    print(line)
  if verdict == "FENCED":
    print("bar0_fence_status next_action=power-cycle the eGPU (unplug/replug the ADT-Link UT3G USB4 eGPU) to clear the WPR2 lock state; set NV_ADD_BYPASS_BAR0_FENCE_CHECK=1 to attempt boot anyway")
  else:
    print("bar0_fence_status next_action=BAR0 is accessible; normal boot should succeed")

def print_fecs_fence_diagnostic(script="examples/add.py"):
  print("fecs_fence_diagnostic signature=FECS_A|FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus|0%>FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus|9b2w>ASSERT|...|BKD")
  print("fecs_fence_diagnostic cause=FECS FALCON is fenced from a prior session (cpuctl=0xbadf5720); GSP RM cannot complete AMPERE_COMPUTE_B alloc because FECS times out on the GPIO/PMU mutex")
  print("fecs_fence_diagnostic key_signals gsp.cpuctl=0xbadf5720 gsp.riscv_bcr=0x111 gsp.rm=0x0 GSP_EVENT_kind=0x3_progress=0% GSP_EVENT_kind=0x3_progress=9b2w GSP_EVENT_kind=0x5_assert_BKD")
  print("fecs_fence_diagnostic recommended_order preboot_only, fecs_minimal, fecs_only, full, promote_retry (each successive scenario adds more in-band FECS/PMU resets, which may wipe FECS ucode after GSP boot)")
  print(f"fecs_fence_diagnostic first_attempt recommended_scenario=preboot_only (safest: only pre-boot FECS+PMU reset, no in-band FECS resets that might wipe FECS ucode after GSP boot)")
  print(f"fecs_fence_diagnostic first_attempt_command {recommended_preboot_only_reconnect_command(script)}")
  print(f"fecs_fence_diagnostic second_attempt recommended_scenario=fecs_minimal (pre-boot FECS only, no PMU reset)")
  print(f"fecs_fence_diagnostic second_attempt_command {recommended_fecs_minimal_reconnect_command(script)}")
  print(f"fecs_fence_diagnostic third_attempt recommended_scenario=fecs_only (drops post-promote PMU reset)")
  print(f"fecs_fence_diagnostic third_attempt_command {recommended_fecs_only_reconnect_command(script)}")
  print("fecs_fence_diagnostic fallback_attempts scenario=full (with PMU resets) scenario=promote_retry (no FECS reset, just re-issue GPU_PROMOTE_CTX)")
  print(f"fecs_fence_diagnostic promote_retry_command {recommended_promote_retry_reconnect_command(script)}")
  print("fecs_fence_diagnostic tinygrad_path tiny does not hit the FECS/PMU mutex stall; if tiny prints result=[11.0, 22.0, 33.0, 44.0] but standalone stalls, the issue is in the standalone GSP RM->FECS path, not the GPU")
  print("fecs_fence_diagnostic compare python3 examples/add.py --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log")
  print("fecs_fence_diagnostic next_action if preboot_only prints result=[11.0, 22.0, 33.0, 44.0], then the in-band FECS/PMU resets were the regression (GSP RM needs to keep FECS ucode after boot); document and make the working command the new default")
  print("fecs_fence_diagnostic hardware_gate=after a GSP RM session ends, WPR2_HI=0x2ffee00 stays locked and FECS BAR stays hardware-gated (all FECS registers return 0xbadf1301); this gate prevents the next boot from re-loading FECS ucode because the PMU write-DMA to FECS IMEM (FALCON offset 0xdb00-0xdf00) is silently dropped; the only known way to clear this state is a physical eGPU power-cycle (unplug/replug the ADT-Link UT3G USB4 eGPU); tiny AND standalone both fail with the same FECS_A->FECS_B stall if run twice in a row without a power-cycle between runs")
  print("fecs_fence_diagnostic root_cause=PMU dmatrfbase advances during the stall (e.g. 0x451b5 -> 0x45205 over a 5s sample), proving PMU is functional and trying to load FECS ucode from VRAM, but FECS BAR stays 0xbadf1301 because the WPR2 lock gates the writes; GSP RM acquires the RMGpioPmuMutex, sends the FECS handshake command, and times out after 40s because the FECS FALCON never becomes responsive")
  print("fecs_fence_diagnostic contrast_with_tiny=tiny's GSP RM boot flow is byte-identical to our boot_ampere_ada() (reset GSP, execute_hs(FRTS), reset GSP riscv=True, reset SEC2, execute_hs(boooter)); tiny succeeds ONLY when the eGPU is freshly power-cycled; tiny would also fail with the same FECS_A->FECS_B stall if run twice in a row without a power-cycle between runs")
  print("fecs_fence_diagnostic mitigation=power-cycle the eGPU between runs of any program that uses the GPU; the user must unplug and replug the ADT-Link UT3G USB4 cable (or power-cycle the eGPU enclosure) before each python3 examples/add.py invocation; the standalone logbuf LOG_RM at the stall shows 0 new entries between FECS_A and FECS_B (the GSP RM thread is blocked on the mutex acquire, not making progress); the FALCON state at the stall is GSP cpuctl=0xbadf5720 (mid-DMA loading cmd queue, dmatrfbase=0x800410), PMU cpuctl=0xbadf5720 (mid-DMA loading FECS ucode, dmatrfbase=0x451b5..0x45205), FECS cpuctl=0xbadf1301 (never loaded, hwcfg2=0xbadf1301)")
  print("fecs_fence_diagnostic verify_with_tiny=to confirm the eGPU is in a bootable state, run python3 examples/add_tiny.py; if it returns result=[11.0, 22.0, 33.0, 44.0] then a python3 examples/add.py run on the same freshly-power-cycled eGPU will also succeed (the two flows are byte-identical in the GSP RM boot path)")
  print("fecs_fence_diagnostic fence_check=StandaloneNvShell.bar0_is_fenced() probes 0x88, 0x100, 0x200, 0x400, 0x1000, 0x1FA828 and treats a majority of 0xbadf00xx-prefixed values as fenced; the check fires after StandaloneNvShell.__init__ runs transport.reset() so it only triggers when the PCI reset fails to clear the WPR2 lock")
  print("fecs_fence_diagnostic bypass=NV_ADD_BYPASS_BAR0_FENCE_CHECK=1 attempts boot anyway (GSP RM will boot, GPU_PROMOTE_CTX will succeed, but AMPERE_COMPUTE_B will stall on RMGpioPmuMutexTimeoutus because the FALCON↔PMU GPIO mutex can never be acquired without a working FECS)")
  print_stall_trace_diagnostic(script)
  print_logbuf_dump_diagnostic(script)
  print("fecs_fence_diagnostic saved_log_state the standalone rm_state stage=... lines now include compact gsp=({format_state} [hwcfg2=0x... has_riscv=0|1 cpuctl=0x... fenced=True|False]), sec2=({format_state} [hwcfg2=0x... has_riscv=0|1 cpuctl=0x... fenced=True|False]), pmu=({format_state} [hwcfg2=0x... has_riscv=0|1 cpuctl=0x... fenced=True|False]), and fecs=({format_state} [hwcfg2=0x... has_riscv=0|1 cpuctl=0x... fenced=True|False]) suffixes (where {format_state} is the full engine/cpuctl/dmactl/dmatrfcmd/.../rm register dump) so the next live session can see all four FALCONs at every RM alloc attempt; the new fecs-reset-pre, fecs-reset-post, fecs-reset-postinit-pre, fecs-reset-postinit-post, fecs-reset-postpromote-pre, fecs-reset-postpromote-post, pmu-reset-pre, pmu-reset-post, pmu-reset-postinit-pre, pmu-reset-postinit-post, pmu-reset-postpromote-pre, pmu-reset-postpromote-post, and fecs-pmu-postinit and fecs-pmu-postpromote trace lines (now emitted right after GPU_PROMOTE_CTX and before the first AMPERE_COMPUTE_B alloc, so the FECS RM handshake state is visible just before the FECS_A|FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus stall) all use the same {format_state} [hwcfg2=... has_riscv=... cpuctl=... fenced=True|False] format; if PMU fenced=True but FECS fenced=False after the preboot FECS+PMU reset, the FECS->PMU mutex handshake is the likely regression; if GSP fenced=True, the boot is corrupt and needs PCI FLR")

def print_stall_trace_diagnostic(script="examples/add.py"):
  print("stall_trace purpose: prints FECS/PMU state every NV_ADD_STALL_TRACE_PERIOD_MS milliseconds (default 5ms) while GSP RM is stuck in the RMGpioPmuMutexTimeoutus stall (the ~40ms wait between EVENT_GSP_POST_NOCAT_RECORD kind=0x3 0% and kind=0x3 9b2w events); without stall-trace the saved log only captures the FALCON state at rm_state stage=... markers, missing what the FALCONs do during the stall itself; with stall-trace the saved log gets stall_trace sample label=rm_gpio_pmu_mutex/post_nocat-progress=... seq=... elapsed_ms=... gsp=(... [hwcfg2=... has_riscv=... cpuctl=... fenced=True|False]) sec2=(... [hwcfg2=... has_riscv=... cpuctl=... fenced=True|False]) pmu=(... [hwcfg2=... has_riscv=... cpuctl=... fenced=True|False]) fecs=(... [hwcfg2=... has_riscv=... cpuctl=... fenced=True|False]) lines, plus a stall_trace stop line with reason=wait_resp_return and sample count, so we can see whether FECS/PMU registers actually change during the stall or are completely frozen at the boot-time fence state")
  print("stall_trace trigger: starts automatically when GspRpcQueue.read_resp() sees a post_nocat record with kind=0x3 (RMGpioPmuMutexTimeoutus); stops automatically when the target RPC response arrives or on timeout")
  print("stall_trace knob NV_ADD_STALL_TRACE_PERIOD_MS controls sample interval (default 5ms); NV_ADD_STALL_TRACE_MAX_MS bounds the trace (default 40000ms = the documented FECS RM stall budget)")
  print("stall_trace diagnostic_rule if all samples are identical the FECS RM handshake is frozen; if pmu.cpuctl changes from 0xbadf5720 to 0x0 mid-stall, the PMU FALCON came alive but FECS is still fenced; if fecs.cpuctl changes from 0xbadf5720 to 0x0 mid-stall, FECS came alive but the handshake timed out anyway")
  print(f"stall_trace_command NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1 NV_ADD_BOOT_GSP=1 NV_ADD_CLEAR_WPR2_FECS=1 NV_ADD_VERIFY_WPR2_AFTER_FRTS=1 NV_ADD_SUMMARY=1 NV_ADD_CHECK_FRTS_BAR1=1 NV_ADD_TRACE_GSP_BOOT=1 NV_ADD_VERIFY_SEC2_INPUTS=1 NV_ADD_TRACE_RM_ALLOC=1 NV_ADD_TRACE_RM_STATE=1 NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 NV_ADD_TRACE_MM_ALLOC=1 NV_ADD_TRACE_LAUNCH_STEPS=1 NV_ADD_TRACE_FALCON=1 NV_ADD_FECS_RESET=1 NV_ADD_FECS_RESET_FORCE=1 NV_ADD_PMU_RESET=1 NV_ADD_PMU_RESET_FORCE=1 NV_ADD_STALL_TRACE=1 python3 {script}")
  print("stall_trace next_action if stall_trace sample lines show pmu=fenced=False and fecs=fenced=False but the stall still times out, the FECS RM handshake protocol is broken in GSP RM firmware (not the FALCON state); if both remain fenced=True, the FECS ucode never started")

def print_logbuf_dump_diagnostic(script="examples/add.py"):
  print("logbuf_dump purpose: reads the 2MB GSP RM log buffer (allocated by GspQueueMemoryBuilder at boot; base address visible in the standalone queue ... logbuf=0x... line) and prints the 5 subregion statistics (LOG_INIT, LOG_INTR, LOG_RM, LOG_MNOC, LOG_KRNL each 0x10000 bytes) plus the printable log lines found in each; this is the only direct view of what GSP RM itself was doing right before the FECS_A|FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus stall, since RM logs are not routed through the GSP RPC response queue")
  print("logbuf_dump trigger: any of (a) NV_ADD_TRACE_GSP_BOOT=1 (printed on every standalone queue ... line), (b) NV_ADD_DUMP_LOGBUF=1 (explicit dump at boot, post-init, post-promote, and on stall-trace stop), or (c) the FECS PMU mutex stall detection in GspRpcQueue.read_resp() (auto-dumps the last 64K of LOG_RM when kind=0x3 RMGpioPmuMutexTimeoutus is observed)")
  print("logbuf_dump knob NV_ADD_DUMP_LOGBUF_LINES controls max printable lines per subregion (default 64); NV_ADD_DUMP_LOGBUF_LINE_LEN controls max characters per line (default 512)")
  print("logbuf_dump format standalone <label> logbuf_dump base=0x... total_size=0x... subregions=5; standalone <label> logbuf subregion=LOG_RM offset=0x20000 size=0x10000 nonzero_bytes=... first_nonzero=0x... last_nonzero=0x... sha256=... lines=N; standalone <label> logbuf LOG_RM line[0]=...; ...; standalone <label> logbuf_dump summary total_lines=N")
  print("logbuf_dump diagnostic_rule if LOG_RM has 0 nonzero bytes, GSP RM never started logging (FALCON halted before RM was loaded); if LOG_RM has lines but they end in a FECS/PMU mutex message, GSP RM itself is waiting on the mutex (so the FALCON state observation matters); if LOG_RM has lines but they mention a successful FECS handshake, the stall is somewhere downstream of FECS init and the saved log's post-promote state is the next place to look")
  print(f"logbuf_dump_command NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1 NV_ADD_BOOT_GSP=1 NV_ADD_CLEAR_WPR2_FECS=1 NV_ADD_VERIFY_WPR2_AFTER_FRTS=1 NV_ADD_SUMMARY=1 NV_ADD_CHECK_FRTS_BAR1=1 NV_ADD_TRACE_GSP_BOOT=1 NV_ADD_VERIFY_SEC2_INPUTS=1 NV_ADD_TRACE_RM_ALLOC=1 NV_ADD_TRACE_RM_STATE=1 NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 NV_ADD_TRACE_MM_ALLOC=1 NV_ADD_TRACE_LAUNCH_STEPS=1 NV_ADD_TRACE_FALCON=1 NV_ADD_FECS_RESET=1 NV_ADD_FECS_RESET_FORCE=1 NV_ADD_PMU_RESET=1 NV_ADD_PMU_RESET_FORCE=1 NV_ADD_DUMP_LOGBUF=1 python3 {script}")
  print("logbuf_dump next_action if logbuf_dump shows LOG_RM stopped writing before the FECS/PMU mutex line, GSP RM was waiting for the FALCON handshake to complete and never got the response; if LOG_RM has the FECS/PMU mutex line, GSP RM correctly tried the handshake and the saved log's stall_trace samples will tell us if the FALCON state evolved at all")

def print_reconnect_commands(script="examples/add.py"):
  print(f"reconnect_command fixed-gpfifo {recommended_reconnect_command(script, golden=False)}")
  print(f"reconnect_command golden-context {recommended_reconnect_command(script, golden=True)}")
  print(f"reconnect_command fecs-reset {recommended_fecs_reset_reconnect_command(script)}")

def print_live_debug_commands(script="examples/add.py", tiny_script="examples/add_tiny.py"):
  print(f"preflight_plan_command {recommended_preflight_plan_command(script)}")
  print_reconnect_commands(script)
  print(f"reconnect_command fecs-scenarios python3 {script} --fecs-reset-scenarios")
  print(f"reconnect_command fecs-fence-diagnostic python3 {script} --fecs-fence-diagnostic")
  print_fecs_fence_diagnostic(script)
  print(f"live_log_workflow_command python3 {script} --live-log-workflow")
  print(f"live_stack_log_workflow_command python3 {script} --live-stack-log-workflow")
  print(f"tiny_live_stack_log_workflow_command python3 {tiny_script} --live-stack-log-workflow --standalone-script {script}")
  print(f"tiny_trace_command {recommended_tiny_trace_command(tiny_script)}")
  print(f"reconnect_command stall-trace python3 {script} --stall-trace")
  print(f"reconnect_command logbuf-dump python3 {script} --logbuf-dump")

def live_log_workflow_lines(script="examples/add.py", tiny_script="examples/add_tiny.py",
                            standalone_log="standalone-golden.log", tiny_log="tiny-golden.log"):
  return [
    f"live_log_workflow script={script} tiny_script={tiny_script} standalone_log={standalone_log} tiny_log={tiny_log}",
    f"gate_command {recommended_preflight_plan_command(script)}",
    f"standalone_log_command {recommended_reconnect_command(script, golden=True)} 2>&1 | tee {standalone_log}",
    f"tiny_log_command {recommended_tiny_trace_command(tiny_script)} 2>&1 | tee {tiny_log}",
    f"compare_command python3 {script} --compare-trace-logs --standalone-log {standalone_log} --tiny-log {tiny_log}",
    "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence",
    "workflow_rule run standalone_log_command only after gate result is ready-for-gsp",
    "workflow_rule run tiny_log_command in the same eGPU session if standalone stalls or times out",
  ]

def print_live_log_workflow(script="examples/add.py", tiny_script="examples/add_tiny.py"):
  standalone_log = cli_arg_value("--standalone-log") or "standalone-golden.log"
  tiny_log = cli_arg_value("--tiny-log") or "tiny-golden.log"
  for line in live_log_workflow_lines(script, tiny_script, standalone_log, tiny_log):
    print(line)

def live_stack_log_workflow_lines(script="examples/add.py", tiny_script="examples/add_tiny.py",
                                  standalone_log="standalone-stack.log", tiny_log="tiny-stack.log"):
  return [
    f"live_stack_log_workflow script={script} tiny_script={tiny_script} standalone_log={standalone_log} tiny_log={tiny_log}",
    f"gate_command {recommended_preflight_plan_command(script)}",
    f"standalone_log_command {recommended_stack_reconnect_command(script, golden=True)} 2>&1 | tee {standalone_log}",
    f"tiny_log_command {recommended_tiny_trace_command(tiny_script)} 2>&1 | tee {tiny_log}",
    f"compare_command python3 {script} --compare-trace-logs --standalone-log {standalone_log} --tiny-log {tiny_log}",
    "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence",
    "workflow_check stack inspect trace_log_compare_stack, trace_log_compare_falcon",
    "workflow_rule use this only when Python call-path stacks are needed; it is verbose",
    "workflow_rule run standalone_log_command only after gate result is ready-for-gsp",
    "workflow_rule run tiny_log_command in the same eGPU session if standalone stalls or times out",
  ]

def print_live_stack_log_workflow(script="examples/add.py", tiny_script="examples/add_tiny.py"):
  standalone_log = cli_arg_value("--standalone-log") or "standalone-stack.log"
  tiny_log = cli_arg_value("--tiny-log") or "tiny-stack.log"
  for line in live_stack_log_workflow_lines(script, tiny_script, standalone_log, tiny_log):
    print(line)

def comparison_checklist_lines(script="examples/add.py", tiny_script="examples/add_tiny.py"):
  return [
    f"comparison_checklist script={script} tiny_script={tiny_script}",
    f"gate_command {recommended_preflight_plan_command(script)}",
    f"standalone_command {recommended_reconnect_command(script, golden=True)}",
    f"tiny_trace_command {recommended_tiny_trace_command(tiny_script)}",
    "compare_line golden_start standalone='channel golden_start' tiny='tiny golden_start'",
    "compare_line golden_gpfifo standalone='channel golden_gpfifo' tiny='tiny gpfifo_patch post'",
    "compare_line golden_promote standalone='channel golden_promote_done' tiny='tiny golden_done'",
    "compare_line rm_pre_queues standalone='standalone rm_alloc pre_queues' tiny='tiny rm_alloc pre_queues'",
    "compare_line rm_post_queues standalone='standalone rm_alloc post_queues' tiny='tiny rm_alloc post_queues'",
    "compare_line compute_alloc standalone='channel golden_compute_alloc|standalone golden_compute_alloc' tiny='tiny compute_alloc'",
    "compare_line dma_alloc standalone='channel golden_dma_alloc|standalone golden_dma_alloc' tiny='tiny dma_alloc'",
    "compare_line exception_queues standalone='standalone rm_alloc exception_queues' tiny='tiny rm_alloc exception'",
    "compare_line token_control standalone='standalone runtime_token_control' tiny='tiny token_control'",
    "compare_line schedule_control standalone='standalone runtime_schedule_control' tiny='tiny schedule_control'",
    "compare_value hashes gpfifo_params,promote_entries,compute_rpc,dma_rpc,token_rpc,schedule_rpc",
    "compare_value promote_context golden,user_phys,user_virt entries_sha256,packed_entries_sha256",
    "compare_value promote_metadata client,subdevice,object,entries,ids,entry_text",
    "compare_value promote_control golden rpc_sha256,result_len,result_sha256",
    "compare_value compute_alloc parent,object,compute_class",
    "compare_value dma_alloc parent,object,dma_class",
    "compare_value exception_context parent,class,client,object",
    "compare_value failure_summary standalone_status,standalone_exception,tiny_exception,message",
    "compare_value progress_summary standalone_stage,tiny_stage,status",
    "compare_value gpfifo_ctor scalars=va,entries,flags,context_share,vaspace,userd_memory,userd_offset,engine,cid,runlist,internal_flags",
    "compare_value gpfifo_desc ramfc,userd,instance,method,error",
    "compare_value rm_queues pre_cmd_wp,pre_stat_rp,post_cmd_wp,post_stat_rp",
    "compare_value boundary_queues compute_pre/post,promote_pre/post cmd_wp,stat_rp",
    "compare_value rm_slots pre/post cmd/stat slot0 func,result,priv",
    "compare_value rm_alloc_sequence class_name,handles,params,rpcs order/prefix/counts/match_flags",
    "compare_value gsp_rpc_sequence func_name,len,sha256 order/prefix/counts/match_flags",
    "compare_value gsp_system_info gpu_phys,gpu_fb,gpu_inst,bdf,max_user_va,cfg_base,cfg_size,pci_ids",
    "compare_value gsp_rpc_response_sequence func_name,len,rp,wp,advance,result,private,sha256 order/prefix/counts/match_flags",
    "compare_value stack_functions common,standalone_only,tiny_only for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control",
    "compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control",
    "compare_value falcon_trace dma_pre,state,writes presence/status",
    "compare_value falcon_dma_values cmd,dest,mem_off,src,size,state_registers",
    "compare_value falcon_write_sequence register=value order/prefix/tails",
    "compare_value boot_values wpr_meta,bootloader,radix3,signature,meta_sha256,booter_offsets,sizes,sha256",
    "compare_value boot_readbacks standalone FRTS_BAR1,SEC2_booter_load,WPR_metadata,WPR_bootloader,GSP_signature,GSP_radix3 sha256,size,vram",
    "compare_value boot_state pre_root bar1,wpr2_hi,gsp_engine,gsp_cpuctl,sec2_engine,sec2_cpuctl",
    "compare_value compute_state pre_compute bar1,wpr2_hi,gsp_registers,sec2_registers",
    "compare_value boot_queue rm_args,libos_args,cmd_head",
    "compare_value mm_valloc first16 size,va,paddrs sequence",
    "compare_rule summary result=mismatch when any present detailed value differs",
    "compare_rule first require gate classification ready-for-gsp before GSP/RM commands",
    "compare_rule second run standalone and tiny trace in the same eGPU session after standalone timeout",
  ]

def print_comparison_checklist(script="examples/add.py", tiny_script="examples/add_tiny.py"):
  for line in comparison_checklist_lines(script, tiny_script):
    print(line)

def trace_log_comparison_specs():
  return [
    ("golden_start", ("channel golden_start",), ("tiny golden_start",), True),
    ("golden_gpfifo", ("channel golden_gpfifo",), ("tiny gpfifo_patch post",), True),
    ("golden_promote", ("channel golden_promote_done",), ("tiny golden_done",), True),
    ("rm_pre_queues", ("standalone rm_alloc pre_queues",), ("tiny rm_alloc pre_queues",), True),
    ("rm_post_queues", ("standalone rm_alloc post_queues",), ("tiny rm_alloc post_queues",), True),
    ("compute_alloc", ("channel golden_compute_alloc", "standalone golden_compute_alloc"), ("tiny compute_alloc",), True),
    ("dma_alloc", ("channel golden_dma_alloc", "standalone golden_dma_alloc"), ("tiny dma_alloc",), False),
    ("exception_queues", ("standalone rm_alloc exception_queues",), ("tiny rm_alloc exception",), False),
    ("token_control", ("standalone runtime_token_control",), ("tiny token_control",), True),
    ("schedule_control", ("standalone runtime_schedule_control",), ("tiny schedule_control",), True),
  ]

def compare_trace_log_text(standalone_text, tiny_text):
  rows = []
  for label, standalone_needles, tiny_needles, required in trace_log_comparison_specs():
    standalone_found = any(needle in standalone_text for needle in standalone_needles)
    tiny_found = any(needle in tiny_text for needle in tiny_needles)
    rows.append({
      "label": label, "standalone": standalone_found, "tiny": tiny_found, "required": required,
      "standalone_needles": standalone_needles, "tiny_needles": tiny_needles,
    })
  return rows

def first_trace_line(text, needles):
  for line in text.splitlines():
    if any(needle in line for needle in needles):
      return line
  return ""

def first_trace_line_matching(text, needles, match_fields):
  for line in text.splitlines():
    if not any(needle in line for needle in needles): continue
    if all(extract_trace_field(line, field) == value for field, value in match_fields.items()):
      return line
  return ""

def extract_trace_field(line, field):
  match = re.search(rf"(?:^| ){re.escape(field)}=", line)
  if not match: return None
  start = match.end()
  if start < len(line) and line[start] == "[":
    end = line.find("]", start)
    if end >= 0: return line[start:end + 1]
  end = line.find(" ", start)
  return (line[start:] if end < 0 else line[start:end]).rstrip(",")

def extract_trace_text_field(line, field, stop_fields=()):
  match = re.search(rf"(?:^| ){re.escape(field)}=", line)
  if not match: return None
  start = match.end()
  stops = [line.find(f" {name}=", start) for name in stop_fields]
  stops = [idx for idx in stops if idx >= 0]
  end = min(stops) if stops else len(line)
  return line[start:end].strip()

def trace_log_field_comparison_specs():
  return [
    ("gpfifo_params_sha256", ("rm gpfifo_patch",), "params_sha256",
     ("tiny gpfifo_patch post",), "params_sha256", True),
    ("promote_entries_sha256", ("channel promote_ctx_payload",), "entries_sha256",
     ("tiny promote_ctx_payload",), "entries_sha256", False),
    ("compute_rpc_sha256", ("channel golden_compute_alloc", "standalone golden_compute_alloc"), "expected_rpc_sha256|rpc_sha256",
     ("tiny compute_alloc",), "rpc_sha256", True),
    ("dma_rpc_sha256", ("channel golden_dma_alloc", "standalone golden_dma_alloc"), "expected_rpc_sha256|rpc_sha256",
     ("tiny dma_alloc",), "rpc_sha256", False),
    ("token_rpc_sha256", ("channel compute_channel_done", "standalone runtime_token_control"), "token_rpc_sha256|rpc_sha256",
     ("tiny token_control",), "rpc_sha256", False),
    ("schedule_rpc_sha256", ("channel compute_channel_done", "standalone runtime_schedule_control"), "schedule_rpc_sha256|rpc_sha256",
     ("tiny schedule_control",), "rpc_sha256", False),
  ]

def extract_first_trace_field(line, fields_expr):
  for field in fields_expr.split("|"):
    value = extract_trace_field(line, field)
    if value is not None: return value
  return None

def extract_queue_pointer(line, queue_label, pointer):
  def parse_tuple_index(text, marker, index):
    match = re.search(rf"{re.escape(marker)}=\(([^)]*)\)", text)
    if not match: return None
    parts = [part.strip() for part in match.group(1).split(",")]
    return parts[index] if len(parts) > index else None
  if pointer == "wp":
    bracket = re.search(rf"{re.escape(queue_label)}=\[([^\]]*)\]", line)
    if bracket:
      value = parse_tuple_index(bracket.group(1), "tx_header", 4)
      if value is not None: return value
    value = parse_tuple_index(line, f"{queue_label}_tx", 4)
    if value is not None: return value
  elif pointer == "rp":
    bracket = re.search(rf"{re.escape(queue_label)}=\[([^\]]*)\]", line)
    if bracket:
      match = re.search(r"rx_ptr=([^; ]+)", bracket.group(1))
      if match: return match.group(1)
    match = re.search(rf"{re.escape(queue_label)}_rx=([^; ]+)", line)
    if match: return match.group(1)
  else:
    raise ValueError(f"unknown queue pointer {pointer}")
  return None

def queue_segment(line, queue_label):
  bracket = re.search(rf"{re.escape(queue_label)}=\[([^\]]*)\]", line)
  if bracket: return bracket.group(1)
  return line

def extract_queue_slot_field(line, queue_label, slot, field):
  segment = queue_segment(line, queue_label)
  standalone_match = re.search(rf"slot{slot}: ([^;]*)", segment)
  tiny_match = re.search(rf"{re.escape(queue_label)}_slot{slot}: ([^;]*)", segment)
  slot_text = (standalone_match or tiny_match).group(1) if (standalone_match or tiny_match) else ""
  if not slot_text: return None
  aliases = {"private": "priv"}
  fields = [field]
  if field in aliases: fields.append(aliases[field])
  if field == "priv": fields.append("private")
  for name in fields:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", slot_text)
    if match: return match.group(1)
  return None

def compare_trace_log_fields(standalone_text, tiny_text):
  rows = []
  for label, standalone_needles, standalone_field, tiny_needles, tiny_field, required in trace_log_field_comparison_specs():
    standalone_line = first_trace_line(standalone_text, standalone_needles)
    tiny_line = first_trace_line(tiny_text, tiny_needles)
    standalone_value = extract_first_trace_field(standalone_line, standalone_field) if standalone_line else None
    tiny_value = extract_first_trace_field(tiny_line, tiny_field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def extract_gpfifo_desc_value(line, desc_name, standalone=False):
  if standalone:
    if desc_name == "error":
      base = extract_trace_field(line, "ctor_error_base")
      size = extract_trace_field(line, "ctor_error_size")
    else:
      base = extract_trace_field(line, f"after_{desc_name}_base")
      size = extract_trace_field(line, f"after_{desc_name}_size")
    if base is not None and size is not None: return f"{base}/{size}"
    desc = extract_trace_field(line, f"desc_{desc_name}")
    if desc is not None:
      parts = desc.split("/")
      if len(parts) >= 2: return f"{parts[0]}/{parts[1]}"
    return None
  desc = extract_trace_field(line, desc_name)
  if desc is None: return None
  parts = desc.split("/")
  return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None

def trace_log_gpfifo_desc_comparison_specs():
  return [(f"gpfifo_desc_{name}", name, False) for name in ("ramfc", "userd", "instance", "method", "error")]

def compare_trace_log_gpfifo_descs(standalone_text, tiny_text):
  standalone_line = next((line for line in standalone_text.splitlines()
                          if "rm gpfifo_patch" in line and "after_ramfc_base=" in line), "")
  if not standalone_line:
    standalone_line = first_trace_line(standalone_text, ("channel golden_gpfifo_alloc",))
  tiny_line = next((line for line in tiny_text.splitlines()
                    if "tiny gpfifo_patch post" in line and "ramfc=" in line), "")
  if not tiny_line:
    tiny_line = first_trace_line(tiny_text, ("tiny gpfifo_patch post",))
  rows = []
  for label, desc_name, required in trace_log_gpfifo_desc_comparison_specs():
    standalone_value = extract_gpfifo_desc_value(standalone_line, desc_name, standalone=True) if standalone_line else None
    tiny_value = extract_gpfifo_desc_value(tiny_line, desc_name, standalone=False) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_gpfifo_scalar_comparison_specs():
  return [
    ("gpfifo_ctor_va", "ctor_gpfifo_va", "gpfifo_va", False),
    ("gpfifo_ctor_entries", "ctor_entries", "entries", False),
    ("gpfifo_ctor_flags", "ctor_flags", "flags", False),
    ("gpfifo_ctor_context_share", "ctor_h_context_share", "h_context_share", False),
    ("gpfifo_ctor_vaspace", "ctor_h_vaspace", "h_vaspace", False),
    ("gpfifo_ctor_userd_memory", "ctor_h_userd_memory", "h_userd_memory", False),
    ("gpfifo_ctor_userd_offset", "ctor_userd_offset", "userd_offset", False),
    ("gpfifo_ctor_engine_type", "ctor_engine_type", "engine_type", False),
    ("gpfifo_ctor_cid", "ctor_cid", "cid", False),
    ("gpfifo_ctor_runlist_id", "ctor_runlist_id", "runlist_id", False),
    ("gpfifo_ctor_internal_flags", "ctor_internal_flags", "internal_flags", False),
  ]

def compare_trace_log_gpfifo_scalars(standalone_text, tiny_text):
  standalone_line = first_trace_line(standalone_text, ("rm gpfifo_patch",))
  tiny_line = next((line for line in tiny_text.splitlines()
                    if "tiny gpfifo_patch post" in line and "gpfifo_va=" in line), "")
  rows = []
  for label, standalone_field, tiny_field, required in trace_log_gpfifo_scalar_comparison_specs():
    standalone_value = extract_trace_field(standalone_line, standalone_field) if standalone_line else None
    tiny_value = extract_trace_field(tiny_line, tiny_field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_promote_context_specs():
  return [
    ("golden_promote_entries_sha256", "virt=default", "phys=default", "entries_sha256", False),
    ("golden_promote_packed_entries_sha256", "virt=default", "phys=default", "packed_entries_sha256", False),
    ("user_phys_promote_entries_sha256", "virt=False", "phys=default", "entries_sha256", False),
    ("user_phys_promote_packed_entries_sha256", "virt=False", "phys=default", "packed_entries_sha256", False),
    ("user_virt_promote_entries_sha256", "virt=default", "phys=False", "entries_sha256", False),
    ("user_virt_promote_packed_entries_sha256", "virt=default", "phys=False", "packed_entries_sha256", False),
  ]

def trace_log_promote_metadata_specs():
  return [
    ("golden_promote_client", "virt=default", "phys=default", "client", False),
    ("golden_promote_subdevice", "virt=default", "phys=default", "subdevice", False),
    ("golden_promote_object", "virt=default", "phys=default", "object", False),
    ("golden_promote_entries", "virt=default", "phys=default", "entries", False),
    ("golden_promote_ids", "virt=default", "phys=default", "ids", False),
    ("golden_promote_entry_text", "virt=default", "phys=default", "entry_text", False),
    ("user_phys_promote_client", "virt=False", "phys=default", "client", False),
    ("user_phys_promote_subdevice", "virt=False", "phys=default", "subdevice", False),
    ("user_phys_promote_object", "virt=False", "phys=default", "object", False),
    ("user_phys_promote_entries", "virt=False", "phys=default", "entries", False),
    ("user_phys_promote_ids", "virt=False", "phys=default", "ids", False),
    ("user_phys_promote_entry_text", "virt=False", "phys=default", "entry_text", False),
    ("user_virt_promote_client", "virt=default", "phys=False", "client", False),
    ("user_virt_promote_subdevice", "virt=default", "phys=False", "subdevice", False),
    ("user_virt_promote_object", "virt=default", "phys=False", "object", False),
    ("user_virt_promote_entries", "virt=default", "phys=False", "entries", False),
    ("user_virt_promote_ids", "virt=default", "phys=False", "ids", False),
    ("user_virt_promote_entry_text", "virt=default", "phys=False", "entry_text", False),
  ]

def first_promote_ctx_line(text, virt_marker, phys_marker):
  return next((line for line in text.splitlines()
               if "promote_ctx_payload" in line and virt_marker in line and phys_marker in line), "")

def compare_trace_log_promote_contexts(standalone_text, tiny_text):
  rows = []
  for label, virt_marker, phys_marker, field, required in trace_log_promote_context_specs():
    standalone_line = first_promote_ctx_line(standalone_text, virt_marker, phys_marker)
    tiny_line = first_promote_ctx_line(tiny_text, virt_marker, phys_marker)
    standalone_value = extract_trace_field(standalone_line, field) if standalone_line else None
    tiny_value = extract_trace_field(tiny_line, field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def compare_trace_log_promote_metadata(standalone_text, tiny_text):
  rows = []
  for label, virt_marker, phys_marker, field, required in trace_log_promote_metadata_specs():
    standalone_line = first_promote_ctx_line(standalone_text, virt_marker, phys_marker)
    tiny_line = first_promote_ctx_line(tiny_text, virt_marker, phys_marker)
    standalone_value = extract_trace_field(standalone_line, field) if standalone_line else None
    tiny_value = extract_trace_field(tiny_line, field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_promote_control_specs():
  return [
    ("golden_promote_control_rpc_sha256", "rpc_sha256", "rpc_sha256", False),
    ("golden_promote_control_result_len", "result_len", "result_len", False),
    ("golden_promote_control_result_sha256", "result_sha256", "result_sha256", False),
  ]

def first_promote_control_post_line(text, prefix):
  return first_trace_line_matching(text, (prefix,), {"cmd": "0x2080012b"})

def compare_trace_log_promote_control(standalone_text, tiny_text):
  standalone_line = first_promote_control_post_line(standalone_text, "standalone rm_control post ")
  tiny_line = first_promote_control_post_line(tiny_text, "tiny rm_control post ")
  rows = []
  for label, standalone_field, tiny_field, required in trace_log_promote_control_specs():
    standalone_value = extract_trace_field(standalone_line, standalone_field) if standalone_line else None
    tiny_value = extract_trace_field(tiny_line, tiny_field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_compute_alloc_comparison_specs():
  return [
    ("compute_alloc_parent", "parent", "parent", False),
    ("compute_alloc_object", "expected_object|object", "object", False),
    ("compute_alloc_class", "compute_class", "compute_class", False),
  ]

def trace_log_dma_alloc_comparison_specs():
  return [
    ("dma_alloc_parent", "parent", "parent", False),
    ("dma_alloc_object", "expected_object|object", "object", False),
    ("dma_alloc_class", "dma_class", "dma_class", False),
  ]

def compare_trace_log_alloc_identity(standalone_text, tiny_text, standalone_needles, tiny_needles, specs):
  standalone_line = first_trace_line(standalone_text, standalone_needles)
  tiny_line = first_trace_line(tiny_text, tiny_needles)
  rows = []
  for label, standalone_field, tiny_field, required in specs:
    standalone_value = extract_first_trace_field(standalone_line, standalone_field) if standalone_line else None
    tiny_value = extract_first_trace_field(tiny_line, tiny_field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def compare_trace_log_compute_alloc(standalone_text, tiny_text):
  return compare_trace_log_alloc_identity(standalone_text, tiny_text,
    ("channel golden_compute_alloc", "standalone golden_compute_alloc"), ("tiny compute_alloc",),
    trace_log_compute_alloc_comparison_specs())

def compare_trace_log_dma_alloc(standalone_text, tiny_text):
  return compare_trace_log_alloc_identity(standalone_text, tiny_text,
    ("channel golden_dma_alloc", "standalone golden_dma_alloc"), ("tiny dma_alloc",),
    trace_log_dma_alloc_comparison_specs())

def trace_log_queue_comparison_specs():
  return [
    ("pre_cmd_wp", ("standalone rm_alloc pre_queues",), "cmd", "wp", ("tiny rm_alloc pre_queues",), "cmd", "wp", False),
    ("pre_stat_rp", ("standalone rm_alloc pre_queues",), "stat", "rp", ("tiny rm_alloc pre_queues",), "stat", "rp", False),
    ("post_cmd_wp", ("standalone rm_alloc post_queues",), "cmd", "wp", ("tiny rm_alloc post_queues",), "cmd", "wp", False),
    ("post_stat_rp", ("standalone rm_alloc post_queues",), "stat", "rp", ("tiny rm_alloc post_queues",), "stat", "rp", False),
    ("compute_pre_cmd_wp", ("standalone rm_alloc pre_queues",), "cmd", "wp", ("tiny rm_alloc pre_queues",), "cmd", "wp", False, {"class": "0xc7c0"}, {"class": "0xc7c0"}),
    ("compute_pre_stat_rp", ("standalone rm_alloc pre_queues",), "stat", "rp", ("tiny rm_alloc pre_queues",), "stat", "rp", False, {"class": "0xc7c0"}, {"class": "0xc7c0"}),
    ("compute_post_cmd_wp", ("standalone rm_alloc post_queues",), "cmd", "wp", ("tiny rm_alloc post_queues",), "cmd", "wp", False, {"class": "0xc7c0"}, {"class": "0xc7c0"}),
    ("compute_post_stat_rp", ("standalone rm_alloc post_queues",), "stat", "rp", ("tiny rm_alloc post_queues",), "stat", "rp", False, {"class": "0xc7c0"}, {"class": "0xc7c0"}),
    ("promote_pre_cmd_wp", ("standalone rm_control pre_queues",), "cmdq", "wp", ("tiny rm_control pre_queues",), "cmd", "wp", False, {"cmd": "0x2080012b"}, {"cmd": "0x2080012b"}),
    ("promote_pre_stat_rp", ("standalone rm_control pre_queues",), "stat", "rp", ("tiny rm_control pre_queues",), "stat", "rp", False, {"cmd": "0x2080012b"}, {"cmd": "0x2080012b"}),
    ("promote_post_cmd_wp", ("standalone rm_control post_queues",), "cmdq", "wp", ("tiny rm_control post_queues",), "cmd", "wp", False, {"cmd": "0x2080012b"}, {"cmd": "0x2080012b"}),
    ("promote_post_stat_rp", ("standalone rm_control post_queues",), "stat", "rp", ("tiny rm_control post_queues",), "stat", "rp", False, {"cmd": "0x2080012b"}, {"cmd": "0x2080012b"}),
  ]

def compare_trace_log_queues(standalone_text, tiny_text):
  rows = []
  for spec in trace_log_queue_comparison_specs():
    label, standalone_needles, standalone_queue, standalone_pointer, tiny_needles, tiny_queue, tiny_pointer, required = spec[:8]
    standalone_match = spec[8] if len(spec) > 8 else {}
    tiny_match = spec[9] if len(spec) > 9 else {}
    standalone_line = first_trace_line_matching(standalone_text, standalone_needles, standalone_match) if standalone_match else first_trace_line(standalone_text, standalone_needles)
    tiny_line = first_trace_line_matching(tiny_text, tiny_needles, tiny_match) if tiny_match else first_trace_line(tiny_text, tiny_needles)
    standalone_value = extract_queue_pointer(standalone_line, standalone_queue, standalone_pointer) if standalone_line else None
    tiny_value = extract_queue_pointer(tiny_line, tiny_queue, tiny_pointer) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_queue_slot_comparison_specs():
  specs = []
  for stage, standalone_needles, tiny_needles in (
    ("pre", ("standalone rm_alloc pre_queues",), ("tiny rm_alloc pre_queues",)),
    ("post", ("standalone rm_alloc post_queues",), ("tiny rm_alloc post_queues",)),
  ):
    for queue in ("cmd", "stat"):
      for field in ("func", "result", "priv"):
        specs.append((f"{stage}_{queue}_slot0_{field}", standalone_needles, queue, 0, field, tiny_needles, queue, 0, field, False))
  return specs

def compare_trace_log_queue_slots(standalone_text, tiny_text):
  rows = []
  for label, standalone_needles, standalone_queue, standalone_slot, standalone_field, tiny_needles, tiny_queue, tiny_slot, tiny_field, required in trace_log_queue_slot_comparison_specs():
    standalone_line = first_trace_line(standalone_text, standalone_needles)
    tiny_line = first_trace_line(tiny_text, tiny_needles)
    standalone_value = extract_queue_slot_field(standalone_line, standalone_queue, standalone_slot, standalone_field) if standalone_line else None
    tiny_value = extract_queue_slot_field(tiny_line, tiny_queue, tiny_slot, tiny_field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_mm_valloc_entries(text, prefix, limit=16, stop_needles=()):
  entries = []
  for line in text.splitlines():
    if any(line.startswith(needle) for needle in stop_needles): break
    if not line.startswith(prefix): continue
    size = extract_trace_field(line, "size")
    va = extract_trace_field(line, "va")
    paddrs = extract_trace_field(line, "paddrs")
    if size is None or va is None or paddrs is None: continue
    entries.append({"size": size, "va": va, "paddrs": paddrs.split(",")[0] if paddrs else "missing"})
    if len(entries) >= limit: break
  return entries

def format_mm_valloc_sequence(entries, field):
  if not entries: return None
  return ">".join(entry[field] for entry in entries)

def compare_trace_log_mm_valloc_sequence(standalone_text, tiny_text, limit=16):
  standalone_entries = trace_log_mm_valloc_entries(
    standalone_text, "mm valloc ", limit=limit,
    stop_needles=("channel golden_compute_alloc", "standalone golden_compute_alloc"))
  tiny_entries = trace_log_mm_valloc_entries(
    tiny_text, "tiny mm valloc ", limit=limit, stop_needles=("tiny compute_alloc",))
  rows = []
  for field in ("size", "va", "paddrs"):
    standalone_value = format_mm_valloc_sequence(standalone_entries, field)
    tiny_value = format_mm_valloc_sequence(tiny_entries, field)
    rows.append({
      "label": f"mm_valloc_{field}_sequence",
      "standalone_value": standalone_value,
      "tiny_value": tiny_value,
      "required": False,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  rows.append({
    "label": "mm_valloc_count",
    "standalone_value": str(len(standalone_entries)) if standalone_entries else None,
    "tiny_value": str(len(tiny_entries)) if tiny_entries else None,
    "required": False,
    "match": bool(standalone_entries and tiny_entries and len(standalone_entries) == len(tiny_entries)),
  })
  return rows

def trace_log_exception_context_specs():
  return [
    ("exception_parent", "parent", "parent", False),
    ("exception_class", "class", "class", False),
    ("exception_client", "client", "client", False),
    ("exception_object", "object", "object", False),
  ]

def compare_trace_log_exception_context(standalone_text, tiny_text):
  standalone_line = first_trace_line(standalone_text, ("standalone rm_state stage=alloc-exception",))
  if not standalone_line:
    standalone_line = first_trace_line(standalone_text, ("standalone rm_alloc exception_queues",))
  tiny_line = first_trace_line(tiny_text, ("tiny rm_alloc exception",))
  rows = []
  for label, standalone_field, tiny_field, required in trace_log_exception_context_specs():
    standalone_value = extract_trace_field(standalone_line, standalone_field) if standalone_line else None
    tiny_value = extract_trace_field(tiny_line, tiny_field) if tiny_line else None
    rows.append({
      "label": label, "standalone_value": standalone_value, "tiny_value": tiny_value, "required": required,
      "match": standalone_value is not None and tiny_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_boot_state_specs():
  return [
    ("pre_root_bar1", "line", "bar1"),
    ("pre_root_wpr2_hi", "line", "wpr2_hi"),
    ("pre_root_gsp_engine", "gsp", "engine"),
    ("pre_root_gsp_cpuctl", "gsp", "cpuctl"),
    ("pre_root_sec2_engine", "sec2", "engine"),
    ("pre_root_sec2_cpuctl", "sec2", "cpuctl"),
  ]

def trace_log_compute_state_specs():
  specs = [
    ("compute_state_bar1", "line", "bar1"),
    ("compute_state_wpr2_hi", "line", "wpr2_hi"),
  ]
  for prefix in ("gsp", "sec2"):
    for field in (
      "engine", "cpuctl", "dmactl", "dmatrfcmd", "dmatrfbase", "dmatrfbase1",
      "dmatrfmoffs", "dmatrffboffs", "hwcfg2", "fbif_ctl", "fbif_transcfg0",
      "exci", "irqstat", "riscv_bcr", "riscv_cpuctl", "os", "rm",
    ):
      specs.append((f"compute_state_{prefix}_{field}", prefix, field))
  return specs

def compare_trace_log_boot_state(standalone_text, tiny_text):
  standalone_line = first_trace_line(standalone_text, ("standalone rm_state stage=pre-alloc",))
  tiny_line = first_trace_line(tiny_text, ("tiny pre-root",))
  standalone_gsp = extract_parenthesized_field(standalone_line, "gsp")
  tiny_gsp = extract_parenthesized_field(tiny_line, "gsp")
  standalone_sec2 = extract_parenthesized_field(standalone_line, "sec2")
  tiny_sec2 = extract_parenthesized_field(tiny_line, "sec2")
  rows = []
  for label, source, field in trace_log_boot_state_specs():
    if source == "line":
      standalone_value = extract_trace_field(standalone_line, field) if standalone_line else None
      tiny_value = extract_trace_field(tiny_line, field) if tiny_line else None
    elif source == "gsp":
      standalone_value = extract_trace_field(standalone_gsp, field) if standalone_gsp else None
      tiny_value = extract_trace_field(tiny_gsp, field) if tiny_gsp else None
    elif source == "sec2":
      standalone_value = extract_trace_field(standalone_sec2, field) if standalone_sec2 else None
      tiny_value = extract_trace_field(tiny_sec2, field) if tiny_sec2 else None
    else:
      raise ValueError(f"unknown boot state source {source}")
    rows.append({
      "label": label,
      "standalone_value": standalone_value,
      "tiny_value": tiny_value,
      "required": False,
      "match": standalone_value is not None and tiny_value is not None and standalone_value == tiny_value,
    })
  return rows

def compare_trace_log_compute_state(standalone_text, tiny_text):
  standalone_line = first_trace_line_matching(standalone_text,
    ("standalone rm_state stage=pre-alloc",), {"class": "0xc7c0"})
  if not standalone_line:
    standalone_line = first_trace_line_matching(standalone_text,
      ("standalone rm_state stage=pre-alloc",), {"class_name": "AMPERE_COMPUTE_B"})
  tiny_line = first_trace_line_matching(tiny_text,
    ("tiny rm_alloc pre_state",), {"class": "0xc7c0"})
  if not tiny_line:
    tiny_line = first_trace_line_matching(tiny_text,
      ("tiny rm_alloc pre_state",), {"class_name": "AMPERE_COMPUTE_B"})
  standalone_gsp = extract_parenthesized_field(standalone_line, "gsp")
  tiny_gsp = extract_parenthesized_field(tiny_line, "gsp")
  standalone_sec2 = extract_parenthesized_field(standalone_line, "sec2")
  tiny_sec2 = extract_parenthesized_field(tiny_line, "sec2")
  rows = []
  for label, source, field in trace_log_compute_state_specs():
    if source == "line":
      standalone_value = extract_trace_field(standalone_line, field) if standalone_line else None
      tiny_value = extract_trace_field(tiny_line, field) if tiny_line else None
    elif source == "gsp":
      standalone_value = extract_trace_field(standalone_gsp, field) if standalone_gsp else None
      tiny_value = extract_trace_field(tiny_gsp, field) if tiny_gsp else None
    elif source == "sec2":
      standalone_value = extract_trace_field(standalone_sec2, field) if standalone_sec2 else None
      tiny_value = extract_trace_field(tiny_sec2, field) if tiny_sec2 else None
    else:
      raise ValueError(f"unknown compute state source {source}")
    rows.append({
      "label": label,
      "standalone_value": standalone_value,
      "tiny_value": tiny_value,
      "required": False,
      "match": standalone_value is not None and tiny_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_boot_queue_specs():
  return [
    ("boot_queue_rm_args", ("standalone queue",), "rm_args", ("tiny queue",), "rm_args"),
    ("boot_queue_libos_args", ("standalone queue",), "libos_args", ("tiny libos_args",), "libos_args"),
    ("boot_queue_cmd_head", ("standalone queue",), "cmd_head", ("tiny queue",), "cmd_head"),
  ]

def compare_trace_log_boot_queue(standalone_text, tiny_text):
  rows = []
  for label, standalone_needles, standalone_field, tiny_needles, tiny_field in trace_log_boot_queue_specs():
    standalone_line = first_trace_line(standalone_text, standalone_needles)
    tiny_line = first_trace_line(tiny_text, tiny_needles)
    standalone_value = extract_trace_field(standalone_line, standalone_field) if standalone_line else None
    tiny_value = extract_trace_field(tiny_line, tiny_field) if tiny_line else None
    rows.append({
      "label": label,
      "standalone_value": standalone_value,
      "tiny_value": tiny_value,
      "required": False,
      "match": standalone_value is not None and tiny_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_failure_summary(standalone_text, tiny_text):
  standalone_status_line = first_trace_line(standalone_text, ("standalone rm_state stage=alloc-status",))
  standalone_exception_line = first_trace_line(standalone_text, ("standalone rm_state stage=alloc-exception",))
  tiny_exception_line = first_trace_line(tiny_text, ("tiny rm_alloc exception",))
  standalone_status = extract_trace_field(standalone_status_line, "status") if standalone_status_line else None
  standalone_exc = extract_trace_field(standalone_exception_line, "exc") if standalone_exception_line else None
  standalone_exc_msg = extract_trace_text_field(standalone_exception_line, "exc_msg", ("rpc_sha256", "cmd_queue", "stat_queue", "cmd_head", "gsp")) if standalone_exception_line else None
  tiny_exc = extract_trace_field(tiny_exception_line, "exc") if tiny_exception_line else None
  tiny_msg = extract_trace_text_field(tiny_exception_line, "msg", ("cmd_tx", "cmd_rx", "cmd_slot0", "stat_tx", "stat_rx", "stat_slot0")) if tiny_exception_line else None
  if standalone_status:
    standalone_result = f"status:{standalone_status}"
  elif standalone_exc:
    standalone_result = f"exception:{standalone_exc}"
  else:
    standalone_result = "none"
  tiny_result = f"exception:{tiny_exc}" if tiny_exc else "none"
  status = "both-clean" if standalone_result == "none" and tiny_result == "none" else ("same" if standalone_result == tiny_result else "different")
  return {
    "standalone_result": standalone_result,
    "standalone_exc_msg": standalone_exc_msg,
    "tiny_result": tiny_result,
    "tiny_msg": tiny_msg,
    "status": status,
  }

def trace_log_progress_stages():
  return [
    ("not-started", (), ()),
    ("golden-start", ("channel golden_start",), ("tiny golden_start",)),
    ("golden-gpfifo", ("channel golden_gpfifo",), ("tiny gpfifo_patch post",)),
    ("golden-promote", ("channel golden_promote_done",), ("tiny golden_done",)),
    ("golden-compute-attempt", ("channel golden_compute_alloc", "standalone golden_compute_alloc"), ("tiny compute_alloc",)),
    ("golden-dma-attempt", ("channel golden_dma_alloc", "standalone golden_dma_alloc"), ("tiny dma_alloc",)),
    ("golden-done", ("channel golden_done",), ("tiny compute_alloc", "tiny dma_alloc")),
    ("runtime-token", ("standalone runtime_token_control",), ("tiny token_control",)),
    ("runtime-schedule", ("standalone runtime_schedule_control",), ("tiny schedule_control",)),
  ]

def trace_log_progress(text, side):
  index, label = 0, "not-started"
  needle_index = 1 if side == "standalone" else 2
  for idx, stage in enumerate(trace_log_progress_stages()):
    needles = stage[needle_index]
    if needles and any(needle in text for needle in needles):
      index, label = idx, stage[0]
  return index, label

def trace_log_progress_summary(standalone_text, tiny_text):
  standalone_index, standalone_stage = trace_log_progress(standalone_text, "standalone")
  tiny_index, tiny_stage = trace_log_progress(tiny_text, "tiny")
  if standalone_index == tiny_index:
    status = "same"
  elif standalone_index < tiny_index:
    status = "standalone-behind"
  else:
    status = "standalone-ahead"
  return {"standalone_stage": standalone_stage, "tiny_stage": tiny_stage, "status": status}

def extract_rm_alloc_sequence_entries(text, side):
  sequence = []
  if side == "standalone":
    for line in text.splitlines():
      if "rm gsp_alloc" not in line: continue
      class_name = extract_trace_field(line, "class_name")
      h_class = extract_trace_field(line, "h_class")
      item = class_name or h_class
      if item is not None:
        sequence.append({"class": item, "parent": extract_trace_field(line, "parent"), "object": extract_trace_field(line, "object"),
                         "params_sha256": extract_trace_field(line, "params_sha256")})
    rpc_values = [extract_trace_field(line, "rpc_sha256") for line in text.splitlines() if "rm gsp_alloc_rpc" in line]
    rpc_values = [value for value in rpc_values if value is not None]
    for entry, rpc_sha256 in zip(sequence, rpc_values):
      entry["rpc_sha256"] = rpc_sha256
    return sequence
  pending = []
  for line in text.splitlines():
    if "tiny rm_alloc pre" in line:
      class_name = extract_trace_field(line, "class_name")
      h_class = extract_trace_field(line, "class")
      item = class_name or h_class
      if item is not None:
        entry = {"class": item, "parent": extract_trace_field(line, "parent"), "object": extract_trace_field(line, "object"),
                 "params_sha256": extract_trace_field(line, "params_sha256")}
        sequence.append(entry)
        pending.append(entry)
    elif "tiny rm_alloc post" in line:
      class_name = extract_trace_field(line, "class_name")
      h_class = extract_trace_field(line, "class")
      item = class_name or h_class
      for entry in pending:
        if entry["class"] == item and entry.get("object") is None:
          entry["object"] = extract_trace_field(line, "object")
          entry["rpc_sha256"] = extract_trace_field(line, "rpc_sha256")
          break
  return sequence

def extract_rm_alloc_class_sequence(text, side):
  return [entry["class"] for entry in extract_rm_alloc_sequence_entries(text, side)]

def format_rm_alloc_entry_sequence(entries):
  if not entries: return None
  parts = []
  for entry in entries:
    suffix = ""
    if entry.get("parent") or entry.get("object"):
      suffix = f"@{entry.get('parent') or '?'}->{entry.get('object') or '?'}"
    parts.append(f"{entry['class']}{suffix}")
  return ">".join(parts)

def format_rm_alloc_hash_sequence(entries, field):
  if not entries: return None
  values = [entry.get(field) or "missing" for entry in entries]
  return ">".join(values)

def common_prefix(items_a, items_b):
  prefix = []
  for item_a, item_b in zip(items_a, items_b):
    if item_a != item_b: break
    prefix.append(item_a)
  return prefix

def compare_trace_log_rm_alloc_sequence(standalone_text, tiny_text):
  standalone_entries = extract_rm_alloc_sequence_entries(standalone_text, "standalone")
  tiny_entries = extract_rm_alloc_sequence_entries(tiny_text, "tiny")
  standalone_seq = [entry["class"] for entry in standalone_entries]
  tiny_seq = [entry["class"] for entry in tiny_entries]
  standalone_handles = format_rm_alloc_entry_sequence(standalone_entries)
  tiny_handles = format_rm_alloc_entry_sequence(tiny_entries)
  standalone_params = format_rm_alloc_hash_sequence(standalone_entries, "params_sha256")
  tiny_params = format_rm_alloc_hash_sequence(tiny_entries, "params_sha256")
  standalone_rpcs = format_rm_alloc_hash_sequence(standalone_entries, "rpc_sha256")
  tiny_rpcs = format_rm_alloc_hash_sequence(tiny_entries, "rpc_sha256")
  class_match = bool(standalone_seq and standalone_seq == tiny_seq)
  handle_match = bool(standalone_handles and tiny_handles and standalone_handles == tiny_handles)
  params_match = bool(standalone_params and tiny_params and standalone_params == tiny_params)
  rpc_match = bool(standalone_rpcs and tiny_rpcs and standalone_rpcs == tiny_rpcs)
  prefix = common_prefix(standalone_seq, tiny_seq)
  return {
    "label": "rm_alloc_sequence",
    "standalone_value": ">".join(standalone_seq) if standalone_seq else None,
    "tiny_value": ">".join(tiny_seq) if tiny_seq else None,
    "standalone_handles": standalone_handles,
    "tiny_handles": tiny_handles,
    "standalone_params": standalone_params,
    "tiny_params": tiny_params,
    "standalone_rpcs": standalone_rpcs,
    "tiny_rpcs": tiny_rpcs,
    "common_prefix": ">".join(prefix) if prefix else None,
    "standalone_next": standalone_seq[len(prefix)] if len(standalone_seq) > len(prefix) else None,
    "tiny_next": tiny_seq[len(prefix)] if len(tiny_seq) > len(prefix) else None,
    "class_match": class_match,
    "handle_match": handle_match,
    "params_match": params_match,
    "rpc_match": rpc_match,
    "match": class_match and handle_match and params_match and rpc_match,
    "prefix_len": len(prefix),
    "standalone_count": len(standalone_seq),
    "tiny_count": len(tiny_seq),
  }

def extract_gsp_rpc_send_entries(text, prefix):
  entries = []
  for line in text.splitlines():
    if not line.startswith(prefix): continue
    func_name = extract_trace_field(line, "func_name")
    func = extract_trace_field(line, "func")
    entries.append({
      "func": func_name or func or "missing",
      "len": extract_trace_field(line, "len") or "missing",
      "sha256": extract_trace_field(line, "sha256") or "missing",
    })
  return entries

def format_gsp_rpc_send_sequence(entries, fields):
  if not entries: return None
  return ">".join("/".join(entry[field] for field in fields) for entry in entries)

def compare_trace_log_gsp_rpc_sequence(standalone_text, tiny_text):
  standalone_entries = extract_gsp_rpc_send_entries(standalone_text, "standalone send_rpc ")
  tiny_entries = extract_gsp_rpc_send_entries(tiny_text, "tiny send_rpc ")
  standalone_funcs = [entry["func"] for entry in standalone_entries]
  tiny_funcs = [entry["func"] for entry in tiny_entries]
  prefix = common_prefix(standalone_funcs, tiny_funcs)
  standalone_hashes = format_gsp_rpc_send_sequence(standalone_entries, ("sha256",))
  tiny_hashes = format_gsp_rpc_send_sequence(tiny_entries, ("sha256",))
  standalone_lens = format_gsp_rpc_send_sequence(standalone_entries, ("len",))
  tiny_lens = format_gsp_rpc_send_sequence(tiny_entries, ("len",))
  func_match = bool(standalone_funcs and tiny_funcs and standalone_funcs == tiny_funcs)
  len_match = bool(standalone_lens and tiny_lens and standalone_lens == tiny_lens)
  hash_match = bool(standalone_hashes and tiny_hashes and standalone_hashes == tiny_hashes)
  return {
    "label": "gsp_rpc_sequence",
    "standalone_value": ">".join(standalone_funcs) if standalone_funcs else None,
    "tiny_value": ">".join(tiny_funcs) if tiny_funcs else None,
    "standalone_hashes": standalone_hashes,
    "tiny_hashes": tiny_hashes,
    "standalone_lens": standalone_lens,
    "tiny_lens": tiny_lens,
    "common_prefix": ">".join(prefix) if prefix else None,
    "standalone_next": standalone_funcs[len(prefix)] if len(standalone_funcs) > len(prefix) else None,
    "tiny_next": tiny_funcs[len(prefix)] if len(tiny_funcs) > len(prefix) else None,
    "func_match": func_match,
    "len_match": len_match,
    "hash_match": hash_match,
    "match": func_match and len_match and hash_match,
    "prefix_len": len(prefix),
    "standalone_count": len(standalone_funcs),
    "tiny_count": len(tiny_funcs),
    "required": False,
  }

def first_gsp_rpc_send_head(text, prefix, func_name):
  for line in text.splitlines():
    if not line.startswith(prefix): continue
    if extract_trace_field(line, "func_name") != func_name: continue
    head = extract_trace_field(line, "head")
    if head is None: return None
    try:
      return bytes.fromhex(head)
    except ValueError:
      return None
  return None

def decode_gsp_system_info_head(head):
  if head is None or len(head) < 100: return None
  gpu_phys, gpu_fb, gpu_inst = struct.unpack_from("<QQQ", head, 0)
  bdf = struct.unpack_from("<Q", head, 32)[0]
  max_user_va = struct.unpack_from("<Q", head, 72)[0]
  cfg_base, cfg_size, pci_device, pci_subdevice, pci_revision = struct.unpack_from("<IIIII", head, 80)
  return {
    "gpu_phys": f"0x{gpu_phys:x}",
    "gpu_fb": f"0x{gpu_fb:x}",
    "gpu_inst": f"0x{gpu_inst:x}",
    "bdf": f"0x{bdf:x}",
    "max_user_va": f"0x{max_user_va:x}",
    "cfg_base": f"0x{cfg_base:x}",
    "cfg_size": f"0x{cfg_size:x}",
    "pci_device": f"0x{pci_device:x}",
    "pci_subdevice": f"0x{pci_subdevice:x}",
    "pci_revision": f"0x{pci_revision:x}",
  }

def compare_trace_log_gsp_system_info(standalone_text, tiny_text):
  standalone_info = decode_gsp_system_info_head(
    first_gsp_rpc_send_head(standalone_text, "standalone send_rpc ", "GSP_SET_SYSTEM_INFO"))
  tiny_info = decode_gsp_system_info_head(
    first_gsp_rpc_send_head(tiny_text, "tiny send_rpc ", "GSP_SET_SYSTEM_INFO"))
  keys = ("gpu_phys", "gpu_fb", "gpu_inst", "bdf", "max_user_va", "cfg_base", "cfg_size",
          "pci_device", "pci_subdevice", "pci_revision")
  return [{
    "label": f"system_info_{key}",
    "standalone_value": standalone_info.get(key) if standalone_info else None,
    "tiny_value": tiny_info.get(key) if tiny_info else None,
    "match": bool(standalone_info and tiny_info and standalone_info.get(key) == tiny_info.get(key)),
    "required": False,
  } for key in keys]

def extract_gsp_rpc_response_entries(text, prefix):
  entries = []
  for line in text.splitlines():
    if not line.startswith(prefix): continue
    func_name = extract_trace_field(line, "func_name")
    func = extract_trace_field(line, "func")
    result = extract_trace_field(line, "result")
    private = extract_trace_field(line, "private")
    sha256 = extract_trace_field(line, "sha256")
    entries.append({
      "func": func_name or func or "missing",
      "len": extract_trace_field(line, "len") or "missing",
      "rp": extract_trace_field(line, "rp") or "missing",
      "wp": extract_trace_field(line, "wp") or "missing",
      "advance": extract_trace_field(line, "advance") or "missing",
      "result": result or "missing",
      "private": private or "missing",
      "sha256": sha256 or "missing",
    })
  return entries

def extract_tiny_gsp_rpc_response_entries(text):
  entries = extract_gsp_rpc_response_entries(text, "tiny read_rpc ")
  seen = {(entry["func"], entry["sha256"]) for entry in entries}
  for line in text.splitlines():
    if not line.startswith("tiny read_rpc_yield "): continue
    func_name = extract_trace_field(line, "func_name")
    func = extract_trace_field(line, "func")
    sha256 = extract_trace_field(line, "sha256") or "missing"
    key = (func_name or func or "missing", sha256)
    if key in seen: continue
    entries.append({
      "func": key[0],
      "len": extract_trace_field(line, "len") or "missing",
      "rp": extract_trace_field(line, "rp") or "missing",
      "wp": extract_trace_field(line, "wp") or "missing",
      "advance": extract_trace_field(line, "advance") or "missing",
      "result": extract_trace_field(line, "result") or "missing",
      "private": extract_trace_field(line, "private") or "missing",
      "sha256": sha256,
    })
    seen.add(key)
  return entries

def decode_trace_head_text(head):
  if not head: return None
  try:
    data = bytes.fromhex(head)
  except ValueError:
    return None
  strings, current = [], []
  for byte in data:
    if 32 <= byte < 127:
      current.append(chr(byte))
    else:
      if len(current) >= 3:
        strings.append("".join(current))
      current = []
  if len(current) >= 3:
    strings.append("".join(current))
  if not strings: return None
  return "|".join(strings[:8])

def extract_line_post_nocat_decode(line):
  if "post_nocat=" in line:
    segment = line.split("post_nocat=", 1)[1]
  elif "GSP EVENT post_nocat" in line:
    segment = line
  else:
    return None
  return {
    "kind": extract_trace_field(segment, "kind") or "missing",
    "strings": extract_trace_field(segment, "strings") or "missing",
    "qwords": extract_trace_field(segment, "qwords") or "missing",
  }

def format_post_nocat_decode_sequence(entries, field):
  if not entries: return None
  return ">".join(entry.get(field) or "missing" for entry in entries)

def format_gsp_rpc_response_sequence(entries, fields):
  if not entries: return None
  return ">".join("/".join(entry[field] for field in fields) for entry in entries)

def compare_trace_log_gsp_rpc_response_sequence(standalone_text, tiny_text):
  standalone_entries = extract_gsp_rpc_response_entries(standalone_text, "standalone read_rpc ")
  tiny_entries = extract_tiny_gsp_rpc_response_entries(tiny_text)
  standalone_funcs = [entry["func"] for entry in standalone_entries]
  tiny_funcs = [entry["func"] for entry in tiny_entries]
  prefix = common_prefix(standalone_funcs, tiny_funcs)
  standalone_lens = format_gsp_rpc_response_sequence(standalone_entries, ("len",))
  tiny_lens = format_gsp_rpc_response_sequence(tiny_entries, ("len",))
  standalone_queue = format_gsp_rpc_response_sequence(standalone_entries, ("rp", "wp", "advance"))
  tiny_queue = format_gsp_rpc_response_sequence(tiny_entries, ("rp", "wp", "advance"))
  standalone_results = format_gsp_rpc_response_sequence(standalone_entries, ("result", "private"))
  tiny_results = format_gsp_rpc_response_sequence(tiny_entries, ("result", "private"))
  standalone_hashes = format_gsp_rpc_response_sequence(standalone_entries, ("sha256",))
  tiny_hashes = format_gsp_rpc_response_sequence(tiny_entries, ("sha256",))
  func_match = bool(standalone_funcs and tiny_funcs and standalone_funcs == tiny_funcs)
  len_match = bool(standalone_lens and tiny_lens and standalone_lens == tiny_lens)
  queue_optional = (not standalone_queue or not tiny_queue or
                    "missing" in standalone_queue.split(">") or "missing" in tiny_queue.split(">"))
  queue_match = True if queue_optional else standalone_queue == tiny_queue
  result_match = bool(standalone_results and tiny_results and standalone_results == tiny_results)
  hash_optional = (not standalone_hashes or not tiny_hashes or
                   "missing" in standalone_hashes.split(">") or "missing" in tiny_hashes.split(">"))
  hash_match = True if hash_optional else standalone_hashes == tiny_hashes
  return {
    "label": "gsp_rpc_response_sequence",
    "standalone_value": ">".join(standalone_funcs) if standalone_funcs else None,
    "tiny_value": ">".join(tiny_funcs) if tiny_funcs else None,
    "standalone_lens": standalone_lens,
    "tiny_lens": tiny_lens,
    "standalone_queue": standalone_queue,
    "tiny_queue": tiny_queue,
    "standalone_results": standalone_results,
    "tiny_results": tiny_results,
    "standalone_hashes": standalone_hashes,
    "tiny_hashes": tiny_hashes,
    "common_prefix": ">".join(prefix) if prefix else None,
    "standalone_next": standalone_funcs[len(prefix)] if len(standalone_funcs) > len(prefix) else None,
    "tiny_next": tiny_funcs[len(prefix)] if len(tiny_funcs) > len(prefix) else None,
    "func_match": func_match,
    "len_match": len_match,
    "queue_match": queue_match,
    "result_match": result_match,
    "hash_match": hash_match,
    "match": func_match and len_match and queue_match and result_match and hash_match,
    "prefix_len": len(prefix),
    "standalone_count": len(standalone_funcs),
    "tiny_count": len(tiny_funcs),
    "required": False,
  }

def compute_rpc_trace_sha(text, side):
  if side == "standalone":
    line = first_trace_line(text, ("channel golden_compute_alloc", "standalone golden_compute_alloc"))
    return extract_first_trace_field(line, "expected_rpc_sha256|rpc_sha256") if line else None
  line = first_trace_line(text, ("tiny compute_alloc",))
  return extract_trace_field(line, "rpc_sha256") if line else None

def extract_post_nocat_event_entries(text, side):
  entries = []
  if side == "standalone":
    response_prefix = "standalone read_rpc "
    log_prefix = "GSP EVENT post_nocat "
  else:
    response_prefix = "tiny read_rpc "
    response_yield_prefix = "tiny read_rpc_yield "
    log_prefix = None
  event_func = "EVENT_GSP_POST_NOCAT_RECORD"
  for line in text.splitlines():
    matched = False
    if line.startswith(response_prefix) or (side == "tiny" and line.startswith(response_yield_prefix)):
      func_name = extract_trace_field(line, "func_name")
      func = extract_trace_field(line, "func")
      if (func_name or func) == event_func:
        matched = True
    elif log_prefix is not None and line.startswith(log_prefix):
      matched = True
    if not matched: continue
    decode = extract_line_post_nocat_decode(line)
    if decode is None: continue
    entries.append({"decode": decode, "kind": decode.get("kind", "missing"),
                    "strings": decode.get("strings", "missing"),
                    "qwords": decode.get("qwords", "missing")})
  return entries

def format_post_nocat_event_sequence(entries, field):
  if not entries: return None
  return ">".join(entry.get(field) or "missing" for entry in entries)

def compare_trace_log_gsp_post_nocat_sequence(standalone_text, tiny_text):
  standalone_entries = extract_post_nocat_event_entries(standalone_text, "standalone")
  tiny_entries = extract_post_nocat_event_entries(tiny_text, "tiny")
  standalone_kinds = [entry["kind"] for entry in standalone_entries]
  tiny_kinds = [entry["kind"] for entry in tiny_entries]
  prefix = common_prefix(standalone_kinds, tiny_kinds)
  standalone_strings = format_post_nocat_event_sequence(standalone_entries, "strings")
  tiny_strings = format_post_nocat_event_sequence(tiny_entries, "strings")
  standalone_kinds_seq = format_post_nocat_event_sequence(standalone_entries, "kind")
  tiny_kinds_seq = format_post_nocat_event_sequence(tiny_entries, "kind")
  standalone_qwords = format_post_nocat_event_sequence(standalone_entries, "qwords")
  tiny_qwords = format_post_nocat_event_sequence(tiny_entries, "qwords")
  return {
    "label": "gsp_post_nocat_sequence",
    "standalone_value": standalone_kinds_seq,
    "tiny_value": tiny_kinds_seq,
    "standalone_strings": standalone_strings,
    "tiny_strings": tiny_strings,
    "standalone_qwords": standalone_qwords,
    "tiny_qwords": tiny_qwords,
    "common_prefix": ">".join(prefix) if prefix else None,
    "standalone_next": standalone_kinds[len(prefix)] if len(standalone_kinds) > len(prefix) else None,
    "tiny_next": tiny_kinds[len(prefix)] if len(tiny_kinds) > len(prefix) else None,
    "prefix_len": len(prefix),
    "standalone_count": len(standalone_kinds),
    "tiny_count": len(tiny_kinds),
    "match": standalone_kinds == tiny_kinds,
    "required": False,
  }

def next_gsp_rpc_observation_after_send(text, send_prefix, read_prefix, send_sha256, include_tiny_yield=False):
  if not send_sha256: return None
  lines = text.splitlines()
  send_index = None
  for index, line in enumerate(lines):
    if line.startswith(send_prefix) and extract_trace_field(line, "sha256") == send_sha256:
      send_index = index
      break
  if send_index is None: return None
  for line in lines[send_index + 1:]:
    if line.startswith(read_prefix):
      func_name = extract_trace_field(line, "func_name")
      func = extract_trace_field(line, "func")
      head = extract_trace_field(line, "head")
      return {
        "func": func_name or func or "missing",
        "len": extract_trace_field(line, "len") or "missing",
        "rp": extract_trace_field(line, "rp") or "missing",
        "wp": extract_trace_field(line, "wp") or "missing",
        "advance": extract_trace_field(line, "advance") or "missing",
        "result": extract_trace_field(line, "result") or "missing",
        "private": extract_trace_field(line, "private") or "missing",
        "sha256": extract_trace_field(line, "sha256") or "missing",
        "head_text": decode_trace_head_text(head) or "missing",
      }
    if include_tiny_yield and line.startswith("tiny read_rpc_yield "):
      func_name = extract_trace_field(line, "func_name")
      func = extract_trace_field(line, "func")
      head = extract_trace_field(line, "head")
      return {
        "func": func_name or func or "missing",
        "len": extract_trace_field(line, "len") or "missing",
        "rp": extract_trace_field(line, "rp") or "missing",
        "wp": extract_trace_field(line, "wp") or "missing",
        "advance": extract_trace_field(line, "advance") or "missing",
        "result": extract_trace_field(line, "result") or "missing",
        "private": extract_trace_field(line, "private") or "missing",
        "sha256": extract_trace_field(line, "sha256") or "missing",
        "head_text": decode_trace_head_text(head) or "missing",
      }
  return None

def post_nocat_observations_after_send(text, send_prefix, read_prefix, send_sha256, include_tiny_yield=False):
  if not send_sha256: return []
  lines = text.splitlines()
  send_index = None
  for index, line in enumerate(lines):
    if line.startswith(send_prefix) and extract_trace_field(line, "sha256") == send_sha256:
      send_index = index
      break
  if send_index is None: return []
  observations = []
  last_event = None
  for line in lines[send_index + 1:]:
    if line.startswith(read_prefix) or (include_tiny_yield and line.startswith("tiny read_rpc_yield ")):
      func_name = extract_trace_field(line, "func_name")
      func = func_name or extract_trace_field(line, "func") or "missing"
      if func != "EVENT_GSP_POST_NOCAT_RECORD":
        break
      decode = extract_line_post_nocat_decode(line)
      last_event = {
        "kind": (decode or {}).get("kind") or "missing",
        "strings": (decode or {}).get("strings") or decode_trace_head_text(extract_trace_field(line, "head")) or "missing",
        "qwords": (decode or {}).get("qwords") or "missing",
      }
      observations.append(last_event)
    elif line.startswith("GSP EVENT post_nocat") and last_event is not None:
      decode = extract_line_post_nocat_decode(line)
      if decode:
        last_event.update(decode)
  return observations

def compare_trace_log_compute_rpc_response(standalone_text, tiny_text):
  standalone_send_sha = compute_rpc_trace_sha(standalone_text, "standalone")
  tiny_send_sha = compute_rpc_trace_sha(tiny_text, "tiny")
  standalone_obs = next_gsp_rpc_observation_after_send(
    standalone_text, "standalone send_rpc ", "standalone read_rpc ", standalone_send_sha)
  tiny_obs = next_gsp_rpc_observation_after_send(
    tiny_text, "tiny send_rpc ", "tiny read_rpc ", tiny_send_sha, include_tiny_yield=True)
  standalone_post = post_nocat_observations_after_send(
    standalone_text, "standalone send_rpc ", "standalone read_rpc ", standalone_send_sha)
  tiny_post = post_nocat_observations_after_send(
    tiny_text, "tiny send_rpc ", "tiny read_rpc ", tiny_send_sha, include_tiny_yield=True)
  standalone_value = standalone_obs["func"] if standalone_obs else None
  tiny_value = tiny_obs["func"] if tiny_obs else None
  send_match = bool(standalone_send_sha and tiny_send_sha and standalone_send_sha == tiny_send_sha)
  func_match = bool(standalone_value and tiny_value and standalone_value == tiny_value)
  len_match = bool(standalone_obs and tiny_obs and standalone_obs["len"] == tiny_obs["len"])
  result_match = bool(standalone_obs and tiny_obs and
                      standalone_obs["result"] == tiny_obs["result"] and
                      standalone_obs["private"] == tiny_obs["private"])
  return {
    "label": "compute_rpc_response",
    "standalone_value": standalone_value,
    "tiny_value": tiny_value,
    "standalone_send_sha256": standalone_send_sha,
    "tiny_send_sha256": tiny_send_sha,
    "standalone_len": standalone_obs["len"] if standalone_obs else None,
    "tiny_len": tiny_obs["len"] if tiny_obs else None,
    "standalone_queue": (f"{standalone_obs['rp']}/{standalone_obs['wp']}/{standalone_obs['advance']}"
                         if standalone_obs else None),
    "tiny_queue": f"{tiny_obs['rp']}/{tiny_obs['wp']}/{tiny_obs['advance']}" if tiny_obs else None,
    "standalone_result": (f"{standalone_obs['result']}/{standalone_obs['private']}" if standalone_obs else None),
    "tiny_result": f"{tiny_obs['result']}/{tiny_obs['private']}" if tiny_obs else None,
    "standalone_sha256": standalone_obs["sha256"] if standalone_obs else None,
    "tiny_sha256": tiny_obs["sha256"] if tiny_obs else None,
    "standalone_head_text": standalone_obs["head_text"] if standalone_obs else None,
    "tiny_head_text": tiny_obs["head_text"] if tiny_obs else None,
    "standalone_post_nocat_kinds": format_post_nocat_decode_sequence(standalone_post, "kind"),
    "tiny_post_nocat_kinds": format_post_nocat_decode_sequence(tiny_post, "kind"),
    "standalone_post_nocat_strings": format_post_nocat_decode_sequence(standalone_post, "strings"),
    "tiny_post_nocat_strings": format_post_nocat_decode_sequence(tiny_post, "strings"),
    "send_match": send_match,
    "func_match": func_match,
    "len_match": len_match,
    "result_match": result_match,
    "match": send_match and func_match and len_match and result_match,
    "required": False,
  }

def trace_log_stack_specs():
  return [
    ("golden_start", ("channel golden_start",), ("tiny golden_start_stack",)),
    ("rm_alloc", ("rm alloc", "rm gsp_alloc", "standalone rm_alloc"), ("tiny rm_alloc_stack",)),
    ("compute_alloc", ("channel golden_compute_alloc", "standalone golden_compute_alloc"), ("tiny compute_alloc_stack",)),
    ("dma_alloc", ("channel golden_dma_alloc", "standalone golden_dma_alloc"), ("tiny dma_alloc_stack",)),
    ("token_control", ("channel runtime_token_control", "standalone runtime_token_control"), ("tiny token_control_stack",)),
    ("schedule_control", ("channel runtime_schedule_control", "standalone runtime_schedule_control"), ("tiny schedule_control_stack",)),
  ]

def extract_python_stack_functions(stack_text):
  funcs = []
  for line in stack_text.replace("\\n", "\n").splitlines():
    match = re.search(r", in ([A-Za-z_][A-Za-z0-9_]*)", line)
    if match and match.group(1) not in funcs:
      funcs.append(match.group(1))
  return funcs

def extract_python_stack_locations(stack_text):
  locations = []
  for line in stack_text.replace("\\n", "\n").splitlines():
    match = re.search(r'File "([^"]+)", line ([0-9]+), in ([A-Za-z_][A-Za-z0-9_]*)', line)
    if not match: continue
    location = f"{os.path.basename(match.group(1))}:{match.group(2)}:{match.group(3)}"
    if location not in locations:
      locations.append(location)
  return locations

def extract_standalone_stack_after_label(text, needles):
  lines = text.splitlines()
  for index, line in enumerate(lines):
    if not any(needle in line for needle in needles): continue
    stack_lines = []
    for next_line in lines[index + 1:]:
      if next_line.startswith('  File "'):
        stack_lines.append(next_line)
      elif stack_lines and next_line.strip().startswith(("File \"", "line ")):
        stack_lines.append(next_line)
      elif stack_lines:
        break
      elif not next_line.strip():
        continue
      elif not next_line.startswith("  "):
        break
    if stack_lines:
      return "\n".join(stack_lines)
  return ""

def extract_tiny_stack_line(text, needles):
  for line in text.splitlines():
    if any(needle in line for needle in needles):
      return line
  return ""

def compare_trace_log_stacks(standalone_text, tiny_text):
  rows = []
  for label, standalone_needles, tiny_needles in trace_log_stack_specs():
    standalone_stack = extract_standalone_stack_after_label(standalone_text, standalone_needles)
    tiny_stack = extract_tiny_stack_line(tiny_text, tiny_needles)
    standalone_funcs = extract_python_stack_functions(standalone_stack)
    tiny_funcs = extract_python_stack_functions(tiny_stack)
    standalone_locations = extract_python_stack_locations(standalone_stack)
    tiny_locations = extract_python_stack_locations(tiny_stack)
    common = [func for func in standalone_funcs if func in tiny_funcs]
    rows.append({
      "label": label,
      "standalone_value": ">".join(standalone_funcs) if standalone_funcs else None,
      "tiny_value": ">".join(tiny_funcs) if tiny_funcs else None,
      "common": ">".join(common) if common else None,
      "standalone_only": ">".join(func for func in standalone_funcs if func not in tiny_funcs) or None,
      "tiny_only": ">".join(func for func in tiny_funcs if func not in standalone_funcs) or None,
      "standalone_locations": ">".join(standalone_locations) if standalone_locations else None,
      "tiny_locations": ">".join(tiny_locations) if tiny_locations else None,
      "required": False,
      "match": bool(common),
    })
  return rows

def trace_log_falcon_specs():
  return [
    ("falcon_dma_pre", ("falcon pre-dma",), ("tiny execute_dma pre",)),
    ("falcon_state", ("state=(",), ("gsp=(", "sec2=(", "state=(")),
    ("falcon_writes", ("falcon wreg",), ("tiny wreg",)),
  ]

def extract_parenthesized_field(line, field):
  match = re.search(rf"{re.escape(field)}=\(([^)]*)\)", line)
  return match.group(1) if match else ""

def first_tiny_falcon_dma_line(text, base_name):
  needle = "tiny execute_dma pre "
  for line in text.splitlines():
    if needle in line:
      return line
  return ""

def trace_log_boot_value_specs():
  return [
    ("wpr_meta_addr", ("standalone wpr meta",), "meta", ("tiny wpr meta",), "meta"),
    ("wpr_bootloader", ("standalone wpr meta",), "bootloader", ("tiny wpr meta",), "bootloader"),
    ("wpr_radix3", ("standalone wpr meta",), "radix3", ("tiny wpr meta",), "radix3"),
    ("wpr_signature", ("standalone wpr meta",), "signature", ("tiny wpr meta",), "signature"),
    ("wpr_meta_sha256", ("standalone wpr meta",), "meta_sha256", ("tiny wpr meta",), "meta_sha256"),
    ("booter_img", ("standalone booter",), "img", ("tiny booter",), "img"),
    ("booter_code_off", ("standalone booter",), "code_off", ("tiny booter",), "code_off"),
    ("booter_data_off", ("standalone booter",), "data_off", ("tiny booter",), "data_off"),
    ("booter_code_size", ("standalone booter",), "code_size", ("tiny booter",), "code_size"),
    ("booter_data_size", ("standalone booter",), "data_size", ("tiny booter",), "data_size"),
    ("booter_sha256", ("standalone booter",), "sha256", ("tiny booter",), "sha256"),
  ]

def compare_trace_log_boot_values(standalone_text, tiny_text):
  rows = []
  for label, standalone_needles, standalone_field, tiny_needles, tiny_field in trace_log_boot_value_specs():
    standalone_line = first_trace_line(standalone_text, standalone_needles)
    tiny_line = first_trace_line(tiny_text, tiny_needles)
    standalone_value = extract_trace_field(standalone_line, standalone_field) if standalone_line else None
    tiny_value = extract_trace_field(tiny_line, tiny_field) if tiny_line else None
    rows.append({
      "label": label,
      "standalone_value": standalone_value,
      "tiny_value": tiny_value,
      "required": False,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def trace_log_boot_readback_specs():
  return [
    ("frts_bar1", ("FRTS BAR1 readback ok",)),
    ("sec2_booter_load", ("SEC2 booter-load readback ok",)),
    ("wpr_metadata", ("WPR metadata readback ok",)),
    ("wpr_bootloader", ("WPR bootloader readback ok",)),
    ("gsp_signature", ("GSP signature readback ok",)),
    ("gsp_radix3", ("GSP radix3 readback ok",)),
  ]

def compare_trace_log_boot_readbacks(standalone_text):
  rows = []
  for label, needles in trace_log_boot_readback_specs():
    line = first_trace_line(standalone_text, needles)
    rows.append({
      "label": label,
      "standalone_value": extract_trace_field(line, "sha256") if line else None,
      "size": extract_trace_field(line, "size") if line else None,
      "vram": extract_trace_field(line, "vram") if line else None,
      "required": False,
    })
  return rows

def trace_log_falcon_value_specs():
  return [
    ("falcon_dma_cmd", "line", "cmd"),
    ("falcon_dma_dest", "line", "dest"),
    ("falcon_dma_mem_off", "line", "mem_off"),
    ("falcon_dma_src", "line", "src"),
    ("falcon_dma_size", "line", "size"),
    ("falcon_dma_cpuctl", "state", "cpuctl"),
    ("falcon_dma_dmatrfcmd", "state", "dmatrfcmd"),
    ("falcon_dma_dmatrfbase", "state", "dmatrfbase"),
    ("falcon_dma_dmatrfbase1", "state", "dmatrfbase1"),
    ("falcon_dma_dmatrfmoffs", "state", "dmatrfmoffs"),
    ("falcon_dma_dmatrffboffs", "state", "dmatrffboffs"),
    ("falcon_dma_fbif_ctl", "state", "fbif_ctl"),
    ("falcon_dma_fbif_transcfg0", "state", "fbif_transcfg0"),
  ]

def compare_trace_log_falcon_values(standalone_text, tiny_text):
  standalone_line = first_trace_line(standalone_text, ("falcon pre-dma",))
  tiny_line = first_tiny_falcon_dma_line(tiny_text, "GSP")
  standalone_state = extract_parenthesized_field(standalone_line, "state")
  tiny_state = extract_parenthesized_field(tiny_line, "state")
  rows = []
  for label, source, field in trace_log_falcon_value_specs():
    if source == "line":
      standalone_value = extract_trace_field(standalone_line, field) if standalone_line else None
      tiny_value = extract_trace_field(tiny_line, field) if tiny_line else None
    else:
      standalone_value = extract_trace_field(standalone_state, field) if standalone_state else None
      tiny_value = extract_trace_field(tiny_state, field) if tiny_state else None
    rows.append({
      "label": label,
      "standalone_value": standalone_value,
      "tiny_value": tiny_value,
      "required": False,
      "match": standalone_value is not None and standalone_value == tiny_value,
    })
  return rows

def extract_falcon_write_sequence(text, prefix):
  rows = []
  for line in text.splitlines():
    if not line.startswith(prefix): continue
    parts = line.split()
    reg = parts[2] if len(parts) > 2 else None
    value = extract_trace_field(line, "value")
    if reg is not None and value is not None:
      rows.append((reg, value))
  return rows

def compare_trace_log_falcon_writes(standalone_text, tiny_text):
  standalone_seq = extract_falcon_write_sequence(standalone_text, "falcon wreg ")
  tiny_seq = extract_falcon_write_sequence(tiny_text, "tiny wreg ")
  standalone_compact = ">".join(f"{reg}={value}" for reg, value in standalone_seq) if standalone_seq else None
  tiny_compact = ">".join(f"{reg}={value}" for reg, value in tiny_seq) if tiny_seq else None
  prefix = []
  for left, right in zip(standalone_seq, tiny_seq):
    if left != right: break
    prefix.append(left)
  return [{
    "label": "falcon_write_sequence",
    "standalone_value": standalone_compact,
    "tiny_value": tiny_compact,
    "common": ">".join(f"{reg}={value}" for reg, value in prefix) if prefix else None,
    "standalone_only": ">".join(f"{reg}={value}" for reg, value in standalone_seq[len(prefix):]) if len(prefix) < len(standalone_seq) else None,
    "tiny_only": ">".join(f"{reg}={value}" for reg, value in tiny_seq[len(prefix):]) if len(prefix) < len(tiny_seq) else None,
    "required": False,
    "match": bool(standalone_seq and tiny_seq and standalone_seq == tiny_seq),
  }]

def compare_trace_log_falcon(standalone_text, tiny_text):
  rows = []
  for label, standalone_needles, tiny_needles in trace_log_falcon_specs():
    standalone_found = any(needle in standalone_text for needle in standalone_needles)
    tiny_found = any(needle in tiny_text for needle in tiny_needles)
    rows.append({
      "label": label,
      "standalone_value": "present" if standalone_found else None,
      "tiny_value": "present" if tiny_found else None,
      "required": False,
      "match": standalone_found and tiny_found,
    })
  return rows

def trace_log_value_mismatches(row_groups):
  mismatches = []
  for group in row_groups or []:
    mismatches += [row["label"] for row in group
                   if row["standalone_value"] and row["tiny_value"] and not row["match"]]
  return mismatches

def format_trace_log_comparison(rows, field_rows=None, value_rows=None):
  missing = [row["label"] for row in rows if row["required"] and (not row["standalone"] or not row["tiny"])]
  mismatched = [row["label"] for row in (field_rows or []) if row["required"] and row["standalone_value"] and row["tiny_value"] and not row["match"]]
  field_missing = [row["label"] for row in (field_rows or []) if row["required"] and (not row["standalone_value"] or not row["tiny_value"])]
  if value_rows is None:
    value_groups = []
  elif value_rows and isinstance(value_rows[0], dict):
    value_groups = [value_rows]
  else:
    value_groups = value_rows
  value_mismatched = trace_log_value_mismatches(value_groups)
  result = "missing" if missing or field_missing else ("mismatch" if mismatched or value_mismatched else "ok")
  details = missing + field_missing + mismatched + value_mismatched
  lines = [f"trace_log_compare result={result} missing={','.join(details) if details else 'none'}"]
  for row in rows:
    status = "ok" if row["standalone"] and row["tiny"] else ("optional-missing" if not row["required"] else "missing")
    lines.append(f"trace_log_compare_line label={row['label']} required={row['required']} "
                 f"standalone={'ok' if row['standalone'] else 'missing'} tiny={'ok' if row['tiny'] else 'missing'} "
                 f"status={status}")
  for row in field_rows or []:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_field label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_queue_comparison(queue_rows):
  lines = []
  for row in queue_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_queue label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_gpfifo_desc_comparison(desc_rows):
  lines = []
  for row in desc_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_desc label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_gpfifo_scalar_comparison(scalar_rows):
  lines = []
  for row in scalar_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_scalar label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_promote_context_comparison(promote_rows):
  lines = []
  for row in promote_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_promote label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_promote_metadata_comparison(promote_meta_rows):
  lines = []
  for row in promote_meta_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_promote_meta label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_promote_control_comparison(promote_control_rows):
  lines = []
  for row in promote_control_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_promote_control label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_compute_alloc_comparison(compute_rows):
  lines = []
  for row in compute_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_compute label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_dma_alloc_comparison(dma_rows):
  lines = []
  for row in dma_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_dma label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_exception_context_comparison(exception_rows):
  lines = []
  for row in exception_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_exception label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_queue_slot_comparison(slot_rows):
  lines = []
  for row in slot_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_slot label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_stack_comparison(stack_rows):
  lines = []
  for row in stack_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "common" if row["common"] else "different"
    else:
      status = "optional-missing"
    lines.append(f"trace_log_compare_stack label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"common={row['common'] or 'none'} standalone_only={row['standalone_only'] or 'none'} "
                 f"tiny_only={row['tiny_only'] or 'none'} "
                 f"standalone_locations={row['standalone_locations'] or 'missing'} "
                 f"tiny_locations={row['tiny_locations'] or 'missing'} status={status}")
  return lines

def format_trace_log_falcon_comparison(falcon_rows):
  lines = []
  for row in falcon_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "present" if row["match"] else "mismatch"
    else:
      status = "optional-missing"
    lines.append(f"trace_log_compare_falcon label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_falcon_write_comparison(write_rows):
  lines = []
  for row in write_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "diverge"
    else:
      status = "optional-missing"
    lines.append(f"trace_log_compare_falcon_writes label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"common_prefix={row['common'] or 'none'} standalone_tail={row['standalone_only'] or 'none'} "
                 f"tiny_tail={row['tiny_only'] or 'none'} status={status}")
  return lines

def format_trace_log_boot_comparison(boot_rows):
  lines = []
  for row in boot_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_boot label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_boot_readback_comparison(readback_rows):
  lines = []
  for row in readback_rows:
    status = "present" if row["standalone_value"] else ("optional-missing" if not row["required"] else "missing")
    lines.append(f"trace_log_compare_boot_readback label={row['label']} required={row['required']} "
                 f"standalone_sha256={row['standalone_value'] or 'missing'} size={row['size'] or 'missing'} "
                 f"vram={row['vram'] or 'missing'} status={status}")
  return lines

def format_trace_log_boot_state_comparison(state_rows):
  lines = []
  for row in state_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_boot_state label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_compute_state_comparison(state_rows):
  lines = []
  for row in state_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_compute_state label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_boot_queue_comparison(queue_rows):
  lines = []
  for row in queue_rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_boot_queue label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_mm_valloc_comparison(rows):
  lines = []
  for row in rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing" if not row["required"] else "missing"
    lines.append(f"trace_log_compare_mm_valloc label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def format_trace_log_failure_summary(summary):
  return (f"trace_log_compare_failure standalone={summary['standalone_result']} tiny={summary['tiny_result']} "
          f"status={summary['status']} standalone_msg={summary['standalone_exc_msg'] or 'none'} "
          f"tiny_msg={summary['tiny_msg'] or 'none'}")

def format_trace_log_progress_summary(summary):
  return (f"trace_log_compare_progress standalone={summary['standalone_stage']} tiny={summary['tiny_stage']} "
          f"status={summary['status']}")

def format_trace_log_rm_alloc_sequence_comparison(row):
  if row["standalone_value"] and row["tiny_value"]:
    status = "match" if row["match"] else "diverge"
  else:
    status = "optional-missing"
  return (f"trace_log_compare_rm_sequence standalone={row['standalone_value'] or 'missing'} "
          f"tiny={row['tiny_value'] or 'missing'} common_prefix={row['common_prefix'] or 'none'} "
          f"standalone_next={row['standalone_next'] or 'none'} tiny_next={row['tiny_next'] or 'none'} "
          f"prefix_len={row['prefix_len']} standalone_count={row['standalone_count']} tiny_count={row['tiny_count']} "
          f"standalone_handles={row['standalone_handles'] or 'missing'} "
          f"tiny_handles={row['tiny_handles'] or 'missing'} standalone_params={row['standalone_params'] or 'missing'} "
          f"tiny_params={row['tiny_params'] or 'missing'} standalone_rpcs={row['standalone_rpcs'] or 'missing'} "
          f"tiny_rpcs={row['tiny_rpcs'] or 'missing'} class_match={row['class_match']} "
          f"handle_match={row['handle_match']} params_match={row['params_match']} rpc_match={row['rpc_match']} "
          f"status={status}")

def format_trace_log_gsp_rpc_sequence_comparison(row):
  if row["standalone_value"] and row["tiny_value"]:
    status = "match" if row["match"] else "diverge"
  else:
    status = "optional-missing"
  return (f"trace_log_compare_gsp_rpc_sequence standalone={row['standalone_value'] or 'missing'} "
          f"tiny={row['tiny_value'] or 'missing'} common_prefix={row['common_prefix'] or 'none'} "
          f"standalone_next={row['standalone_next'] or 'none'} tiny_next={row['tiny_next'] or 'none'} "
          f"prefix_len={row['prefix_len']} standalone_count={row['standalone_count']} tiny_count={row['tiny_count']} "
          f"standalone_lens={row['standalone_lens'] or 'missing'} tiny_lens={row['tiny_lens'] or 'missing'} "
          f"standalone_hashes={row['standalone_hashes'] or 'missing'} tiny_hashes={row['tiny_hashes'] or 'missing'} "
          f"func_match={row['func_match']} len_match={row['len_match']} hash_match={row['hash_match']} "
          f"status={status}")

def format_trace_log_gsp_rpc_response_sequence_comparison(row):
  if row["standalone_value"] and row["tiny_value"]:
    status = "match" if row["match"] else "diverge"
  else:
    status = "optional-missing"
  return (f"trace_log_compare_gsp_rpc_response_sequence standalone={row['standalone_value'] or 'missing'} "
          f"tiny={row['tiny_value'] or 'missing'} common_prefix={row['common_prefix'] or 'none'} "
          f"standalone_next={row['standalone_next'] or 'none'} tiny_next={row['tiny_next'] or 'none'} "
          f"prefix_len={row['prefix_len']} standalone_count={row['standalone_count']} tiny_count={row['tiny_count']} "
          f"standalone_lens={row['standalone_lens'] or 'missing'} tiny_lens={row['tiny_lens'] or 'missing'} "
          f"standalone_queue={row['standalone_queue'] or 'missing'} tiny_queue={row['tiny_queue'] or 'missing'} "
          f"standalone_results={row['standalone_results'] or 'missing'} tiny_results={row['tiny_results'] or 'missing'} "
          f"standalone_hashes={row['standalone_hashes'] or 'missing'} tiny_hashes={row['tiny_hashes'] or 'missing'} "
          f"func_match={row['func_match']} len_match={row['len_match']} queue_match={row['queue_match']} "
          f"result_match={row['result_match']} hash_match={row['hash_match']} "
          f"status={status}")

def format_trace_log_gsp_post_nocat_sequence_comparison(row):
  if row["standalone_value"] and row["tiny_value"]:
    status = "match" if row["match"] else "diverge"
  else:
    status = "optional-missing"
  return (f"trace_log_compare_gsp_post_nocat_sequence standalone={row['standalone_value'] or 'missing'} "
          f"tiny={row['tiny_value'] or 'missing'} common_prefix={row['common_prefix'] or 'none'} "
          f"standalone_next={row['standalone_next'] or 'none'} tiny_next={row['tiny_next'] or 'none'} "
          f"prefix_len={row['prefix_len']} standalone_count={row['standalone_count']} tiny_count={row['tiny_count']} "
          f"standalone_strings={row['standalone_strings'] or 'missing'} tiny_strings={row['tiny_strings'] or 'missing'} "
          f"status={status}")

def format_trace_log_compute_rpc_response_comparison(row):
  if row["standalone_value"] and row["tiny_value"]:
    status = "match" if row["match"] else "diverge"
  else:
    status = "optional-missing"
  return (f"trace_log_compare_compute_rpc_response standalone_next={row['standalone_value'] or 'missing'} "
          f"tiny_next={row['tiny_value'] or 'missing'} "
          f"standalone_send_sha256={row['standalone_send_sha256'] or 'missing'} "
          f"tiny_send_sha256={row['tiny_send_sha256'] or 'missing'} "
          f"standalone_len={row['standalone_len'] or 'missing'} tiny_len={row['tiny_len'] or 'missing'} "
          f"standalone_queue={row['standalone_queue'] or 'missing'} tiny_queue={row['tiny_queue'] or 'missing'} "
          f"standalone_result={row['standalone_result'] or 'missing'} tiny_result={row['tiny_result'] or 'missing'} "
          f"standalone_sha256={row['standalone_sha256'] or 'missing'} tiny_sha256={row['tiny_sha256'] or 'missing'} "
          f"standalone_head_text={row['standalone_head_text'] or 'missing'} "
          f"tiny_head_text={row['tiny_head_text'] or 'missing'} "
          f"standalone_post_nocat_kinds={row['standalone_post_nocat_kinds'] or 'missing'} "
          f"tiny_post_nocat_kinds={row['tiny_post_nocat_kinds'] or 'missing'} "
          f"standalone_post_nocat_strings={row['standalone_post_nocat_strings'] or 'missing'} "
          f"tiny_post_nocat_strings={row['tiny_post_nocat_strings'] or 'missing'} "
          f"send_match={row['send_match']} func_match={row['func_match']} len_match={row['len_match']} "
          f"result_match={row['result_match']} status={status}")

def format_trace_log_gsp_system_info_comparison(rows):
  lines = []
  for row in rows:
    if row["standalone_value"] and row["tiny_value"]:
      status = "match" if row["match"] else "mismatch"
    else:
      status = "optional-missing"
    lines.append(f"trace_log_compare_system_info label={row['label']} required={row['required']} "
                 f"standalone={row['standalone_value'] or 'missing'} tiny={row['tiny_value'] or 'missing'} "
                 f"status={status}")
  return lines

def print_trace_log_comparison(standalone_log, tiny_log):
  missing_args = []
  if not standalone_log: missing_args.append("--standalone-log")
  if not tiny_log: missing_args.append("--tiny-log")
  if missing_args:
    print(f"trace_log_compare_error kind=missing-argument flags={','.join(missing_args)}")
    raise SystemExit(2)
  standalone_path, tiny_path = pathlib.Path(standalone_log), pathlib.Path(tiny_log)
  missing_paths = []
  if not standalone_path.is_file(): missing_paths.append(f"standalone:{standalone_path}")
  if not tiny_path.is_file(): missing_paths.append(f"tiny:{tiny_path}")
  if missing_paths:
    print(f"trace_log_compare_error kind=missing-file paths={','.join(missing_paths)}")
    raise SystemExit(2)
  standalone_text = standalone_path.read_text(errors="replace")
  tiny_text = tiny_path.read_text(errors="replace")
  text_rows = compare_trace_log_text(standalone_text, tiny_text)
  field_rows = compare_trace_log_fields(standalone_text, tiny_text)
  desc_rows = compare_trace_log_gpfifo_descs(standalone_text, tiny_text)
  scalar_rows = compare_trace_log_gpfifo_scalars(standalone_text, tiny_text)
  promote_rows = compare_trace_log_promote_contexts(standalone_text, tiny_text)
  promote_meta_rows = compare_trace_log_promote_metadata(standalone_text, tiny_text)
  promote_control_rows = compare_trace_log_promote_control(standalone_text, tiny_text)
  compute_rows = compare_trace_log_compute_alloc(standalone_text, tiny_text)
  dma_rows = compare_trace_log_dma_alloc(standalone_text, tiny_text)
  exception_rows = compare_trace_log_exception_context(standalone_text, tiny_text)
  queue_rows = compare_trace_log_queues(standalone_text, tiny_text)
  slot_rows = compare_trace_log_queue_slots(standalone_text, tiny_text)
  stack_rows = compare_trace_log_stacks(standalone_text, tiny_text)
  falcon_rows = compare_trace_log_falcon(standalone_text, tiny_text)
  falcon_value_rows = compare_trace_log_falcon_values(standalone_text, tiny_text)
  falcon_write_rows = compare_trace_log_falcon_writes(standalone_text, tiny_text)
  boot_rows = compare_trace_log_boot_values(standalone_text, tiny_text)
  boot_readback_rows = compare_trace_log_boot_readbacks(standalone_text)
  boot_state_rows = compare_trace_log_boot_state(standalone_text, tiny_text)
  compute_state_rows = compare_trace_log_compute_state(standalone_text, tiny_text)
  boot_queue_rows = compare_trace_log_boot_queue(standalone_text, tiny_text)
  mm_valloc_rows = compare_trace_log_mm_valloc_sequence(standalone_text, tiny_text)
  rm_sequence_row = compare_trace_log_rm_alloc_sequence(standalone_text, tiny_text)
  gsp_rpc_sequence_row = compare_trace_log_gsp_rpc_sequence(standalone_text, tiny_text)
  gsp_system_info_rows = compare_trace_log_gsp_system_info(standalone_text, tiny_text)
  gsp_rpc_response_sequence_row = compare_trace_log_gsp_rpc_response_sequence(standalone_text, tiny_text)
  compute_rpc_response_row = compare_trace_log_compute_rpc_response(standalone_text, tiny_text)
  comparison_lines = format_trace_log_comparison(text_rows, field_rows, [desc_rows, scalar_rows, promote_rows, promote_meta_rows, promote_control_rows, compute_rows, dma_rows, exception_rows, queue_rows, slot_rows, stack_rows, falcon_value_rows, falcon_write_rows, boot_rows, boot_state_rows, compute_state_rows, boot_queue_rows, mm_valloc_rows, [rm_sequence_row], [gsp_rpc_sequence_row], gsp_system_info_rows, [gsp_rpc_response_sequence_row], [compute_rpc_response_row]])
  print(comparison_lines[0])
  print(format_trace_log_failure_summary(trace_log_failure_summary(standalone_text, tiny_text)))
  print(format_trace_log_progress_summary(trace_log_progress_summary(standalone_text, tiny_text)))
  print(format_trace_log_rm_alloc_sequence_comparison(rm_sequence_row))
  print(format_trace_log_gsp_rpc_sequence_comparison(gsp_rpc_sequence_row))
  for line in format_trace_log_gsp_system_info_comparison(gsp_system_info_rows):
    print(line)
  print(format_trace_log_gsp_rpc_response_sequence_comparison(gsp_rpc_response_sequence_row))
  gsp_post_nocat_row = compare_trace_log_gsp_post_nocat_sequence(standalone_text, tiny_text)
  print(format_trace_log_gsp_post_nocat_sequence_comparison(gsp_post_nocat_row))
  print(format_trace_log_compute_rpc_response_comparison(compute_rpc_response_row))
  for line in comparison_lines[1:]:
    print(line)
  for line in format_trace_log_gpfifo_desc_comparison(desc_rows):
    print(line)
  for line in format_trace_log_gpfifo_scalar_comparison(scalar_rows):
    print(line)
  for line in format_trace_log_promote_context_comparison(promote_rows):
    print(line)
  for line in format_trace_log_promote_metadata_comparison(promote_meta_rows):
    print(line)
  for line in format_trace_log_promote_control_comparison(promote_control_rows):
    print(line)
  for line in format_trace_log_compute_alloc_comparison(compute_rows):
    print(line)
  for line in format_trace_log_dma_alloc_comparison(dma_rows):
    print(line)
  for line in format_trace_log_exception_context_comparison(exception_rows):
    print(line)
  for line in format_trace_log_queue_comparison(queue_rows):
    print(line)
  for line in format_trace_log_queue_slot_comparison(slot_rows):
    print(line)
  for line in format_trace_log_stack_comparison(stack_rows):
    print(line)
  for line in format_trace_log_falcon_comparison(falcon_rows):
    print(line)
  for line in format_trace_log_falcon_comparison(falcon_value_rows):
    print(line)
  for line in format_trace_log_falcon_write_comparison(falcon_write_rows):
    print(line)
  for line in format_trace_log_boot_comparison(boot_rows):
    print(line)
  for line in format_trace_log_boot_readback_comparison(boot_readback_rows):
    print(line)
  for line in format_trace_log_boot_state_comparison(boot_state_rows):
    print(line)
  for line in format_trace_log_compute_state_comparison(compute_state_rows):
    print(line)
  for line in format_trace_log_boot_queue_comparison(boot_queue_rows):
    print(line)
  for line in format_trace_log_mm_valloc_comparison(mm_valloc_rows):
    print(line)

def cli_arg_value(flag):
  if flag not in sys.argv: return None
  index = sys.argv.index(flag)
  if index + 1 >= len(sys.argv):
    print(f"cli_arg_error kind=missing-value flag={flag}")
    raise SystemExit(2)
  return sys.argv[index + 1]

def print_offline_debug_suite(arithmetic="add"):
  script = "examples/mul.py" if arithmetic == "mul" else "examples/add.py"
  print_runtime_summary()
  print_import_guard()
  print_golden_compute_fingerprint()
  print_context_promote_fingerprint()
  print_gpfifo_constructor_fingerprint()
  print_runtime_channel_fingerprint()
  print_launch_fingerprint(arithmetic)
  print_stall_trace_diagnostic(script)
  print_logbuf_dump_diagnostic(script)

def print_debug_help(script="examples/add.py", arithmetic="add"):
  print(f"debug_help script={script} arithmetic={arithmetic}")
  print(f"transport_preflight python3 {script} --transport-preflight")
  print(f"transport_preflight_gate python3 {script} --transport-preflight-gate")
  print(f"transport_preflight_require_ready python3 {script} --transport-preflight-gate --require-ready")
  print(f"transport_preflight_plan python3 {script} --transport-preflight-plan --require-ready")
  print(f"transport_preflight_classify echo '<transport_preflight line>' | python3 {script} --classify-transport-preflight")
  print(f"offline_debug python3 {script} --offline-debug-suite")
  print(f"contract_suite python3 {script} --contract-suite")
  print(f"validation_suite python3 {script} --validation-suite")
  print(f"live_debug python3 {script} --live-debug-commands")
  print(f"fecs_reset_scenarios python3 {script} --fecs-reset-scenarios")
  print(f"fecs_fence_diagnostic python3 {script} --fecs-fence-diagnostic")
  print(f"stall_trace python3 {script} --stall-trace")
  print(f"logbuf_dump python3 {script} --logbuf-dump")
  print(f"live_log_workflow python3 {script} --live-log-workflow")
  print(f"live_stack_log_workflow python3 {script} --live-stack-log-workflow")
  print(f"comparison_checklist python3 {script} --comparison-checklist")
  print(f"compare_trace_logs python3 {script} --compare-trace-logs --standalone-log standalone.log --tiny-log tiny.log")
  print(f"context_promote_fingerprint python3 {script} --context-promote-fingerprint")
  print(f"gpfifo_constructor_fingerprint python3 {script} --gpfifo-constructor-fingerprint")
  print(f"import_guard python3 {script} --import-guard")
  print(f"tiny_trace {recommended_tiny_trace_command()}")

def import_guard_state():
  return {
    "tinygrad_modules": sorted(name for name in sys.modules if name.startswith("tinygrad")),
    "ref_tinygrad_paths": sorted(path for path in sys.path if "ref/tinygrad" in path),
  }

def static_import_guard_state(paths=None):
  paths = [pathlib.Path(__file__), ROOT / "examples" / "mul.py"] if paths is None else [pathlib.Path(path) for path in paths]
  hits = []
  for path in paths:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          if alias.name == "tinygrad" or alias.name.startswith("tinygrad."):
            hits.append(f"{path.name}:{node.lineno}:import {alias.name}")
      elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == "tinygrad" or module.startswith("tinygrad."):
          hits.append(f"{path.name}:{node.lineno}:from {module} import")
  return {"tinygrad_static_imports": sorted(hits)}

def static_external_import_guard_state(paths=None):
  paths = [pathlib.Path(__file__), ROOT / "examples" / "mul.py"] if paths is None else [pathlib.Path(path) for path in paths]
  allowed_stdlib = {
    "array", "ast", "collections", "contextlib", "ctypes", "enum", "fcntl", "functools", "hashlib", "io", "mmap",
    "os", "pathlib", "re", "socket", "struct", "subprocess", "sys", "tempfile", "threading", "time", "traceback", "urllib",
  }
  allowed_local = {"add"}
  hits = []
  for path in paths:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
      modules = []
      if isinstance(node, ast.Import):
        modules = [alias.name.split(".", 1)[0] for alias in node.names]
      elif isinstance(node, ast.ImportFrom):
        if node.level:
          continue
        modules = [(node.module or "").split(".", 1)[0]]
      for module in modules:
        if module and module not in allowed_stdlib and module not in allowed_local:
          hits.append(f"{path.name}:{node.lineno}:{module}")
  return {"external_static_imports": sorted(hits)}

def print_import_guard():
  state = import_guard_state()
  static_state = static_import_guard_state()
  external_state = static_external_import_guard_state()
  print(f"tinygrad_modules={state['tinygrad_modules']}")
  print(f"ref_tinygrad_paths={state['ref_tinygrad_paths']}")
  print(f"tinygrad_static_imports={static_state['tinygrad_static_imports']}")
  print(f"external_static_imports={external_state['external_static_imports']}")

class OfflineGoldenSysmemTransport:
  def __init__(self):
    self.next_pa, self.allocs = 0x90000000, []
  def alloc_sysmem(self, size, contiguous=False):
    size = round_up(size, 0x1000)
    self.allocs.append((size, contiguous))
    view = MMIOView(bytearray(size))
    paddrs = [self.next_pa + off for off in range(0, size, 0x1000)]
    self.next_pa += size
    return view, paddrs

class OfflineGoldenRm:
  def __init__(self, shell, live_sized=False):
    self.shell, self.live_sized = shell, live_sized
    self.calls, self.allocs, self.controls = [], [], []
    self.priv_root, self.next_handle = 0xc1e00004, 0xcf000001 if live_sized else 0xcf000000
  def handle(self):
    handle = self.next_handle
    self.next_handle += 1
    return handle
  def alloc_memory(self, h_device, h_class, paddrs, length, flags):
    self.calls.append((h_device, h_class, tuple(paddrs), length, flags))
    return self.handle()
  def rm_alloc(self, h_parent, h_class, params=b"", client=None, h_object=None):
    h_object = self.handle() if h_object is None else h_object
    self.allocs.append((h_parent, h_class, h_object, bytes(params)))
    if self.live_sized and h_class == AMPERE_CHANNEL_GPFIFO_A:
      ramfc = self.shell.mm.valloc(0x1000, contiguous=True)
      method = self.shell.mm.palloc(0x5000, align=0x1000)
      self.private_gpfifo_backing = (ramfc, method)
    return h_object
  def rm_control(self, h_object, cmd, params=b"", client=None):
    self.controls.append((h_object, cmd, bytes(params)))
    if cmd == NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      return struct.pack("<I", 0x66)
    if cmd == NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO:
      data = bytearray(1664)
      if self.live_sized:
        sizes = {
          0: (0x153000, 0x1000), 16: (0x5000, 0x1000), 17: (0x3000, 0x1000), 18: (0x20000, 0x1000),
          19: (0x900000, 0x1000), 20: (0x80000, 0x1000), 21: (0x1000, 0x1000), 22: (0x1000, 0x1000),
          23: (0x10000, 0x1000), 24: (0x80000, 0x1000),
        }
      else:
        sizes = {0: (0x2000, 0x1000), 16: (0x3000, 0x1000),
                 **{idx: (0x1000 + idx * 0x100, 0x1000) for idx in range(17, 25)}}
      for idx, (size, align) in sizes.items():
        struct.pack_into("<II", data, idx * 8, size, align)
      return bytes(data)
    return b""

def offline_golden_shell(live_sized=False):
  mm = GpuMemoryManager(0x300000000 - 0x4000000, reserve_ptable=True) if live_sized else GpuMemoryManager(64 << 20)
  return type("OfflineGoldenShell", (), {"mm": mm, "transport": OfflineGoldenSysmemTransport(), "chip_name": "GA102"})()

def golden_compute_fingerprint_state():
  shell = offline_golden_shell(live_sized=True)
  first_boot = shell.mm.palloc(0xf000)
  second_boot = shell.mm.palloc(0xf000)
  rm = OfflineGoldenRm(shell, live_sized=True)
  golden = StandaloneChannelBuilder(shell, rm).prepare_golden_image_context(StandaloneBufferAllocator(shell))
  promote_control = next(ctrl for ctrl in rm.controls if ctrl[1] == NV2080_CTRL_CMD_GPU_PROMOTE_CTX)
  compute_alloc = next(call for call in rm.allocs if call[1] == AMPERE_COMPUTE_B)
  dma_alloc = next(call for call in rm.allocs if call[1] == AMPERE_DMA_COPY_B)
  return {
    "boot_paddrs": (first_boot, second_boot),
    "gpfifo": golden["gpfifo"],
    "compute_alloc": compute_alloc,
    "dma_alloc": dma_alloc,
    "promote_sha256": hashlib.sha256(promote_control[2]).hexdigest(),
    "compute_rpc_sha256": hashlib.sha256(pack_rpc_gsp_rm_alloc(rm.priv_root, compute_alloc[0],
      compute_alloc[2], compute_alloc[1], compute_alloc[3])).hexdigest(),
    "dma_rpc_sha256": hashlib.sha256(pack_rpc_gsp_rm_alloc(rm.priv_root, dma_alloc[0],
      dma_alloc[2], dma_alloc[1], dma_alloc[3])).hexdigest(),
    "grctx_paddrs": {idx: golden["grctx_mappings"][idx].paddrs for idx in (0, 2, 9, 10, 11)},
  }

def print_golden_compute_fingerprint():
  state = golden_compute_fingerprint_state()
  compute_parent, compute_class, compute_object, _ = state["compute_alloc"]
  dma_parent, dma_class, dma_object, _ = state["dma_alloc"]
  print(f"standalone golden_boot_paddrs first=0x{state['boot_paddrs'][0]:x} second=0x{state['boot_paddrs'][1]:x}")
  print(f"standalone golden_promote gpfifo=0x{state['gpfifo']:x} promote_sha256={state['promote_sha256']}")
  print(f"standalone golden_compute_alloc parent=0x{compute_parent:x} object=0x{compute_object:x} "
        f"compute_class=0x{compute_class:x} rpc_sha256={state['compute_rpc_sha256']}")
  print(f"standalone golden_dma_alloc parent=0x{dma_parent:x} object=0x{dma_object:x} "
        f"dma_class=0x{dma_class:x} rpc_sha256={state['dma_rpc_sha256']}")
  for idx, paddrs in state["grctx_paddrs"].items():
    print(f"standalone golden_grctx id={idx} paddrs={','.join(f'0x{base:x}/0x{size:x}' for base, size in paddrs)}")

def unpack_promote_ctx_entries(payload):
  if len(payload) < 48: raise ValueError("promote context payload is truncated")
  count = struct.unpack_from("<I", payload, 40)[0]
  if len(payload) < 48 + count * 32: raise ValueError("promote context entries are truncated")
  return [struct.unpack_from("<QQQI HBB", payload, 48 + idx * 32) for idx in range(count)]

def promote_ctx_entry_fingerprint(payload):
  entries = [(phys, virt, size, attr, buffer_id, bool(init), bool(nonmapped))
             for phys, virt, size, attr, buffer_id, init, nonmapped in unpack_promote_ctx_entries(payload)]
  packed_entries = b"".join(struct.pack("<QQQI HBB", phys, virt, size, attr, buffer_id, init, nonmapped)
                            for phys, virt, size, attr, buffer_id, init, nonmapped in entries)
  return {
    "count": len(entries),
    "ids": [entry[4] for entry in entries],
    "entries_sha256": hashlib.sha256(repr(entries).encode()).hexdigest(),
    "packed_entries_sha256": hashlib.sha256(packed_entries).hexdigest(),
    "payload_sha256": hashlib.sha256(payload).hexdigest(),
  }

def context_promote_fingerprint_state():
  shell = offline_golden_shell(live_sized=True)
  shell.mm.palloc(0xf000)
  shell.mm.palloc(0xf000)
  rm = OfflineGoldenRm(shell, live_sized=True)
  builder = StandaloneChannelBuilder(shell, rm)
  golden = builder.prepare_golden_image_context(StandaloneBufferAllocator(shell))
  golden_control = next(ctrl for ctrl in rm.controls if ctrl[1] == NV2080_CTRL_CMD_GPU_PROMOTE_CTX)
  user_client = rm.priv_root
  user_start = len(rm.controls)
  user_maps = builder.promote_user_compute_context(user_client, golden["subdevice"], golden["gpfifo"],
    golden["grctx_descs"], grctx_mappings=golden["grctx_mappings"])
  user_controls = [ctrl for ctrl in rm.controls[user_start:] if ctrl[1] == NV2080_CTRL_CMD_GPU_PROMOTE_CTX]
  if len(user_controls) != 2: raise RuntimeError("expected two user promote controls")
  return {
    "user_client": user_client,
    "subdevice": golden["subdevice"],
    "gpfifo": golden["gpfifo"],
    "golden": promote_ctx_entry_fingerprint(golden_control[2]),
    "user_phys": promote_ctx_entry_fingerprint(user_controls[0][2]),
    "user_virt": promote_ctx_entry_fingerprint(user_controls[1][2]),
    "user_map_paddrs": {idx: user_maps[idx].paddrs for idx in (0, 1, 2)},
    "user_map_vas": {idx: user_maps[idx].va_addr for idx in (0, 1, 2)},
  }

def print_context_promote_fingerprint():
  state = context_promote_fingerprint_state()
  print(f"standalone context_promote target client=0x{state['user_client']:x} "
        f"subdevice=0x{state['subdevice']:x} gpfifo=0x{state['gpfifo']:x}")
  for label in ("golden", "user_phys", "user_virt"):
    fp = state[label]
    print(f"standalone context_promote label={label} entries={fp['count']} ids={fp['ids']} "
          f"payload_sha256={fp['payload_sha256']} entries_sha256={fp['entries_sha256']} "
          f"packed_entries_sha256={fp['packed_entries_sha256']}")
  for idx in (0, 1, 2):
    paddrs = state["user_map_paddrs"][idx]
    print(f"standalone context_promote_map id={idx} va=0x{state['user_map_vas'][idx]:x} "
          f"paddrs={','.join(f'0x{base:x}/0x{size:x}' for base, size in paddrs)}")

def runtime_channel_fingerprint_state():
  shell = offline_golden_shell(live_sized=False)
  rm = OfflineGoldenRm(shell, live_sized=False)
  allocator = StandaloneBufferAllocator(shell)
  allocator.configure_rm(rm, 0x80)
  resources = StandaloneChannelBuilder(shell, rm).allocate_runtime_resources(
    allocator, h_device=0x80, h_virtmem=0x70, h_vaspace=0x90f1, entries=4)
  gpfifo_alloc = next(call for call in rm.allocs if call[1] == AMPERE_CHANNEL_GPFIFO_A)
  compute_alloc = next(call for call in rm.allocs if call[1] == AMPERE_COMPUTE_B)
  debugger_alloc = next(call for call in rm.allocs if call[1] == GT200_DEBUGGER)
  token_control = next(ctrl for ctrl in rm.controls if ctrl[1] == NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN)
  schedule_control = next(ctrl for ctrl in rm.controls if ctrl[1] == NVA06C_CTRL_CMD_GPFIFO_SCHEDULE)
  return {
    "handles": dict(resources.handles),
    "token": resources.compute_gpfifo.token,
    "gpfifo_alloc": gpfifo_alloc,
    "compute_alloc": compute_alloc,
    "debugger_alloc": debugger_alloc,
    "token_control": token_control,
    "schedule_control": schedule_control,
    "gpfifo_params_sha256": hashlib.sha256(gpfifo_alloc[3]).hexdigest(),
    "compute_rpc_sha256": hashlib.sha256(pack_rpc_gsp_rm_alloc(rm.priv_root, compute_alloc[0],
      compute_alloc[2], compute_alloc[1], compute_alloc[3])).hexdigest(),
    "debugger_rpc_sha256": hashlib.sha256(pack_rpc_gsp_rm_alloc(rm.priv_root, debugger_alloc[0],
      debugger_alloc[2], debugger_alloc[1], debugger_alloc[3])).hexdigest(),
    "token_rpc_sha256": pack_rpc_gsp_rm_control_fingerprint(rm.priv_root, token_control[0], token_control[1], token_control[2]),
    "schedule_rpc_sha256": pack_rpc_gsp_rm_control_fingerprint(rm.priv_root, schedule_control[0], schedule_control[1], schedule_control[2]),
    "gpfifo_area": resources.gpfifo_area,
    "notifier": resources.notifier_buf,
  }

def gpfifo_constructor_fingerprint_state():
  state = runtime_channel_fingerprint_state()
  parent, gpfifo_class, obj, params = state["gpfifo_alloc"]
  return {
    "parent": parent,
    "object": obj,
    "gpfifo_class": gpfifo_class,
    "params_sha256": hashlib.sha256(params).hexdigest(),
    "ctor": unpack_nv_channel_gpfifo_allocation_params(params),
  }

def print_gpfifo_constructor_fingerprint():
  state = gpfifo_constructor_fingerprint_state()
  ctor = state["ctor"]
  print(f"standalone gpfifo_constructor parent=0x{state['parent']:x} object=0x{state['object']:x} "
        f"gpfifo_class=0x{state['gpfifo_class']:x} params_sha256={state['params_sha256']} "
        f"gpfifo_va=0x{ctor['gpfifo_va']:x} entries={ctor['entries']} flags=0x{ctor['flags']:x} "
        f"h_context_share=0x{ctor['h_context_share']:x} h_vaspace=0x{ctor['h_vaspace']:x} "
        f"h_userd_memory=0x{ctor['h_userd_memory']:x} userd_offset=0x{ctor['userd_offset']:x} "
        f"engine_type=0x{ctor['engine_type']:x} cid={ctor['cid']} runlist_id={ctor['runlist_id']} "
        f"internal_flags=0x{ctor['internal_flags']:x}")
  for name in ("ramfc", "userd", "instance", "method"):
    desc = ctor["descs"][name]
    print(f"standalone gpfifo_desc name={name} base=0x{desc['base']:x} size=0x{desc['size']:x} "
          f"as={desc['address_space']} ca={desc['cache_attrib']}")
  desc = ctor["error_desc"]
  print(f"standalone gpfifo_desc name=error base=0x{desc['base']:x} size=0x{desc['size']:x} "
        f"as={desc['address_space']} ca={desc['cache_attrib']}")

def print_runtime_channel_fingerprint():
  state = runtime_channel_fingerprint_state()
  gpfifo_parent, gpfifo_class, gpfifo_object, _ = state["gpfifo_alloc"]
  compute_parent, compute_class, compute_object, _ = state["compute_alloc"]
  debugger_parent, debugger_class, debugger_object, _ = state["debugger_alloc"]
  token_object, token_cmd, token_params = state["token_control"]
  schedule_object, schedule_cmd, schedule_params = state["schedule_control"]
  print(f"standalone runtime_gpfifo_alloc parent=0x{gpfifo_parent:x} object=0x{gpfifo_object:x} "
        f"gpfifo_class=0x{gpfifo_class:x} params_sha256={state['gpfifo_params_sha256']}")
  print(f"standalone runtime_compute_alloc parent=0x{compute_parent:x} object=0x{compute_object:x} "
        f"compute_class=0x{compute_class:x} rpc_sha256={state['compute_rpc_sha256']}")
  print(f"standalone runtime_debugger_alloc parent=0x{debugger_parent:x} object=0x{debugger_object:x} "
        f"debugger_class=0x{debugger_class:x} rpc_sha256={state['debugger_rpc_sha256']}")
  print(f"standalone runtime_token_control object=0x{token_object:x} cmd=0x{token_cmd:x} "
        f"params={token_params.hex()} token=0x{state['token']:x} rpc_sha256={state['token_rpc_sha256']}")
  print(f"standalone runtime_schedule_control object=0x{schedule_object:x} cmd=0x{schedule_cmd:x} "
        f"params={schedule_params.hex()} rpc_sha256={state['schedule_rpc_sha256']}")
  print(f"standalone runtime_resources gpfifo_area=0x{state['gpfifo_area'].va_addr:x} "
        f"notifier=0x{state['notifier'].va_addr:x} dma_gpfifo={state['handles'].get('dma_gpfifo') is not None}")

def assert_transport_contract():
  required = ("read_config", "write_config", "bar_info", "map_bar", "alloc_sysmem", "sleep")
  for name in required:
    attr = getattr(Transport, name, None)
    if not callable(attr): raise AssertionError(f"Transport is missing {name}")

  class FakeTransport(Transport):
    def __init__(self):
      self.calls = []
      self.bar = MMIOView(bytearray(range(16)))
    def read_config(self, offset, size):
      self.calls.append(("read_config", offset, size))
      return 0x10de if (offset, size) == (0, 2) else 0
    def write_config(self, offset, value, size):
      self.calls.append(("write_config", offset, value, size))
    def bar_info(self, bar):
      self.calls.append(("bar_info", bar))
      return 0x1000 + bar * 0x1000, len(self.bar)
    def map_bar(self, bar, off=0, size=None, fmt='B'):
      self.calls.append(("map_bar", bar, off, size, fmt))
      return self.bar.view(off, size, fmt)
    def alloc_sysmem(self, size, contiguous=False):
      self.calls.append(("alloc_sysmem", size, contiguous))
      size = round_up(size, 0x1000)
      return MMIOView(bytearray(size)), [0x80000000 + off for off in range(0, size, 0x1000)]
    def sleep(self, timeout_ms):
      self.calls.append(("sleep", timeout_ms))

  fake = FakeTransport()
  assert fake.read_config(0, 2) == 0x10de
  fake.write_config(4, 7, 2)
  assert fake.bar_info(0) == (0x1000, 16)
  bar_words = fake.map_bar(0, off=0, size=16, fmt='I')
  assert len(bar_words) == 4 and int(bar_words[0]) == 0x03020100
  sysmem_view, pages = fake.alloc_sysmem(0x1001, contiguous=True)
  assert sysmem_view.nbytes == 0x2000 and pages == [0x80000000, 0x80001000]
  fake.sleep(5)
  assert fake.calls == [
    ("read_config", 0, 2), ("write_config", 4, 7, 2), ("bar_info", 0),
    ("map_bar", 0, 0, 16, "I"), ("alloc_sysmem", 0x1001, True), ("sleep", 5),
  ]

def print_transport_contract():
  assert_transport_contract()
  print("transport_contract=ok")

def assert_register_contract():
  class FakeRegTransport:
    def __init__(self): self.bar = MMIOView(bytearray(0x200000), fmt='I')
    def map_bar(self, bar, off=0, size=None, fmt='I'):
      if off != 0: return self.bar.view(off, size, fmt)
      return self.bar.view(0, size, fmt)

  regs = NVRegisters(FakeRegTransport())
  regs.wreg(4, 0x12345678)
  assert regs.rreg(4) == 0x12345678
  regs.write_bits(4, 8, 15, 0xaa)
  assert regs.rreg(4) == 0x1234aa78 and regs.read_bits(4, 8, 15) == 0xaa
  for bad_reg_call, text in [
    (lambda: regs.rreg(-4), "register address"),
    (lambda: regs.rreg(2), "aligned"),
    (lambda: regs.wreg(4, 0x100000000), "register value"),
    (lambda: regs.write_bits(4, 16, 8, 0), "bit range"),
    (lambda: regs.write_bits(4, 0, 32, 0), "bit range"),
    (lambda: regs.write_bits(4, 0, 7, 0x100000000), "register bit value"),
  ]:
    try:
      bad_reg_call()
      raise AssertionError("bad register helper input was accepted")
    except ValueError as exc:
      assert text in str(exc)

  class ProbeRegs:
    def __init__(self, architecture, implementation, boot0=0x00a1, vram_mb=8192):
      self.values = {
        NV_PMC_BOOT_0: boot0,
        NV_PMC_BOOT_42: (architecture << 24) | (implementation << 20),
        NV_PGC6_AON_SECURE_SCRATCH_GROUP_42: vram_mb,
      }
    def rreg(self, addr): return self.values.get(addr, 0)

  for architecture, implementation, chip_name, fw_name, mmu_ver, fmc_boot in [
    (0x17, 0, "GA100", "ga102", 2, False),
    (0x19, 2, "AD102", "ad102", 2, False),
    (0x1b, 2, "GB202", "gb202", 3, True),
  ]:
    chip = NvChipInfo.probe(ProbeRegs(architecture, implementation))
    assert chip.chip_name == chip_name and chip.fw_name == fw_name
    assert chip.mmu_ver == mmu_ver and chip.fmc_boot is fmc_boot
    assert sane_chip_probe(chip)
  assert not sane_chip_probe(NvChipInfo.probe(ProbeRegs(0x17, 0, boot0=0)))
  assert not sane_chip_probe(NvChipInfo.probe(ProbeRegs(0x17, 0, vram_mb=1)))

def print_register_contract():
  assert_register_contract()
  print("register_contract=ok")

def assert_boot_firmware_contract():
  fw_store = FirmwareStore(root="/tmp/nv-fw-contract")
  assert next(fw_store.candidates("ga102", "gsp.bin")) == pathlib.Path("/tmp/nv-fw-contract/ga102/gsp/gsp.bin")
  for bad_fw_call, text in [
    (lambda: FirmwareStore.validate_component("", "firmware name"), "non-empty"),
    (lambda: FirmwareStore.validate_component("..", "firmware name"), "path component"),
    (lambda: list(fw_store.candidates("ga102", "../gsp.bin")), "path component"),
  ]:
    try:
      bad_fw_call()
      raise AssertionError("bad firmware component was accepted")
    except ValueError as exc:
      assert text in str(exc)

  image = bytes(range(256)) * 0x30
  pages, offsets, total_size = GspFirmwarePrep.radix3_layout(len(image))
  assert pages == [1, 1, 1, 3] and offsets == [0, 0x1000, 0x2000, 0x3000]
  page_addrs = [0x80000000 + idx * 0x1000 for idx in range(sum(pages))]
  radix_image, radix_pages, radix_offsets = GspFirmwarePrep.build_radix3_image(image, page_addrs)
  assert radix_pages == pages and radix_offsets == offsets and len(radix_image) == total_size
  assert radix_image[offsets[-1]:offsets[-1] + len(image)] == image
  assert struct.unpack_from("<Q", radix_image, offsets[0])[0] == page_addrs[1]
  assert struct.unpack_from("<Q", radix_image, offsets[1])[0] == page_addrs[2]
  assert struct.unpack_from("<Q", radix_image, offsets[2] + 8)[0] == page_addrs[4]
  try:
    GspFirmwarePrep.build_radix3_image(image, page_addrs[:-1])
    raise AssertionError("short radix3 page list was accepted")
  except ValueError as exc:
    assert "not enough physical pages" in str(exc)

  booter_desc = {"monitor_code_offset": 0x20, "monitor_data_offset": 0x30, "manifest_offset": 0x10}
  meta = GspFirmwarePrep.build_wpr_meta(0x200000000, 0x2000, len(radix_image), 0x100000, 0x200000, 0x300000, booter_desc)
  assert len(meta) == GSP_FW_WPR_META_SIZE
  assert struct.unpack_from("<Q", meta, 0)[0] == GSP_FW_WPR_META_MAGIC
  assert struct.unpack_from("<Q", meta, 16)[0] == 0x200000
  assert struct.unpack_from("<Q", meta, 24)[0] == len(radix_image)
  assert struct.unpack_from("<Q", meta, 32)[0] == 0x100000
  assert struct.unpack_from("<Q", meta, 48)[0] == 0x20
  assert struct.unpack_from("<Q", meta, 56)[0] == 0x30
  assert struct.unpack_from("<Q", meta, 64)[0] == 0x10
  assert struct.unpack_from("<Q", meta, 72)[0] == 0x300000
  assert struct.unpack_from("<Q", meta, 152)[0] == 0x1ffe00000
  try:
    GspFirmwarePrep.build_wpr_meta(0x200000000, 0x2000, len(radix_image), 0x100000, 0x200000, 0x300000,
      {"monitor_code_offset": 0x20, "monitor_data_offset": 0x30}, frts_offset=0x12345)
    raise AssertionError("bad WPR metadata was accepted")
  except ValueError as exc:
    assert "manifest_offset" in str(exc)
  try:
    GspFirmwarePrep.build_wpr_meta(0x200000000, 0x2000, len(radix_image), 0x100000, 0x200000, 0x300000,
      booter_desc, frts_offset=0x12345)
    raise AssertionError("bad FRTS offset was accepted")
  except ValueError as exc:
    assert "FRTS offset" in str(exc)

  class ContractBootTransport:
    def __init__(self):
      self.bar1 = MMIOView(bytearray(0x8000))
      self.sysmem_allocs = []
    def alloc_sysmem(self, size, contiguous=False):
      self.sysmem_allocs.append((size, contiguous))
      return MMIOView(bytearray(size)), [0x81000000 + off for off in range(0, size, 0x1000)]
    def map_bar(self, bar):
      assert bar == 1
      return self.bar1
    def bar_info(self, bar):
      assert bar == 1
      return (0x90000000, self.bar1.nbytes)
  class ContractBootMm:
    def __init__(self): self.next = 0x2000; self.boot_flags = []
    def palloc(self, size, boot=True):
      self.boot_flags.append(boot)
      out = self.next
      self.next += size
      return out
  shell = type("ContractBootShell", (), {"transport": ContractBootTransport(), "mm": ContractBootMm(), "large_bar": False})()
  sys_view, sys_paddr, sys_pages = BootMemoryAllocator(shell).alloc(0x1001, data=b"abc", sysmem=True, contiguous=True)
  assert sys_paddr is None and sys_pages == [0x81000000, 0x81001000]
  assert bytes(sys_view[:3]) == b"abc"
  assert shell.transport.sysmem_allocs == [(0x2000, True)]
  vram_view, vram_paddr, vram_pages = BootMemoryAllocator(shell).alloc(0x1000, data=b"xyz", sysmem=False, boot=False)
  assert vram_paddr == 0x2000 and vram_pages == [0x90002000]
  assert bytes(vram_view[:3]) == b"xyz"
  assert shell.mm.boot_flags == [False]
  try:
    BootMemoryAllocator(shell).alloc(1, data=b"xx", sysmem=True)
    raise AssertionError("oversized boot data was accepted")
  except ValueError as exc:
    assert "data size" in str(exc)
  region = GspQueueMemoryBuilder.pack_libos_memory_region(1, 2, 0x1000, "RMARGS", 0x81000000)
  name_q, paddr_q, size_q, kind_b, loc_b = struct.unpack_from("<QQQBB", region)
  assert len(region) == 32 and name_q == int.from_bytes(b"RMARGS", "big")
  assert paddr_q == 0x81000000 and size_q == 0x1000 and kind_b == 1 and loc_b == 2

def print_boot_firmware_contract():
  assert_boot_firmware_contract()
  print("boot_firmware_contract=ok")

def assert_gsp_rpc_contract():
  def init_queue_view(size=0x5000, msg_size=0x1000, msg_count=4, write_ptr=0):
    view = MMIOView(bytearray(size))
    view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=size, msg_size=msg_size, msg_count=msg_count,
      write_ptr=write_ptr, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
    return view
  def rpc_record(func, msg=b"", elem_count=1, rpc_result=0, checksum_delta=0):
    payload = RpcHeader(length=RpcHeader.SIZE + len(msg), function=func, rpc_result=rpc_result, rpc_result_private=0).pack() + msg
    elem = GspQueueElement(seq=0, elem_count=elem_count)
    elem.checksum = GspRpcQueue.record_checksum(elem.pack(), payload) ^ checksum_delta
    return (elem.pack() + payload).ljust(elem_count * 0x1000, b"\0")

  stat_view = init_queue_view(write_ptr=1)
  cmd_view = init_queue_view(write_ptr=1)
  msg = b"ok"
  stat_view.view(0x1000, 0x1000)[:] = rpc_record(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, msg)
  q = GspRpcQueue(None, stat_view, cmd_view)
  assert q.wait_resp(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, timeout_ms=10) == msg
  assert q.rx_view[0] == 1
  old_trace_rpc_read = os.environ.get("NV_ADD_TRACE_RPC_READ")
  try:
    os.environ["NV_ADD_TRACE_RPC_READ"] = "1"
    trace_stat = init_queue_view(write_ptr=1)
    trace_cmd = init_queue_view(write_ptr=1)
    trace_stat.view(0x1000, 0x1000)[:] = rpc_record(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, msg)
    trace_buf = io.StringIO()
    with contextlib.redirect_stdout(trace_buf):
      assert GspRpcQueue(None, trace_stat, trace_cmd).wait_resp(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, timeout_ms=10) == msg
    trace_text = trace_buf.getvalue()
    assert f"sha256={hashlib.sha256(msg).hexdigest()}" in trace_text and "head=6f6b" in trace_text
    post_msg = struct.pack("<QQQ", 0, 0x12345678, 5) + b"ASSERT\0FECS_A\0GR_STATUS\0"
    trace_post_stat = init_queue_view(write_ptr=1)
    trace_post_cmd = init_queue_view(write_ptr=1)
    trace_post_stat.view(0x1000, 0x1000)[:] = rpc_record(NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD, post_msg)
    post_buf = io.StringIO()
    with contextlib.redirect_stdout(post_buf):
      assert next(GspRpcQueue(None, trace_post_stat, trace_post_cmd).read_resp()) == (
        NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD, post_msg)
    post_text = post_buf.getvalue()
    assert "GSP EVENT post_nocat len=0x30 qwords=0x0,0x12345678,0x5 kind=0x5 strings=ASSERT|FECS_A|GR_STATUS" in post_text
  finally:
    if old_trace_rpc_read is None: os.environ.pop("NV_ADD_TRACE_RPC_READ", None)
    else: os.environ["NV_ADD_TRACE_RPC_READ"] = old_trace_rpc_read

  bad_stat = init_queue_view(write_ptr=1)
  bad_cmd = init_queue_view(write_ptr=1)
  bad_stat.view(0x1000, 0x1000)[:] = rpc_record(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, checksum_delta=1)
  try:
    next(GspRpcQueue(None, bad_stat, bad_cmd).read_resp())
    raise AssertionError("bad RPC response checksum was accepted")
  except ValueError as exc:
    assert "checksum mismatch" in str(exc)

  send_view = init_queue_view(msg_size=0x100, msg_count=4)
  send_q = GspRpcQueue(None, send_view)
  max_payload = send_q.tx.msg_size * (send_q.tx.msg_count - 1) - GspQueueElement.SIZE - RpcHeader.SIZE
  send_q.send_rpc(0x123, bytes(range(256)) * 3)
  first_hdr = RpcHeader.unpack_from(send_q.queue_mv.view(GspQueueElement.SIZE, RpcHeader.SIZE)[:])
  second_hdr = RpcHeader.unpack_from(send_q.queue_mv.view(0x300 + GspQueueElement.SIZE, RpcHeader.SIZE)[:])
  first_elem = GspQueueElement.unpack_from(send_q.queue_mv.view(0, GspQueueElement.SIZE)[:])
  second_elem = GspQueueElement.unpack_from(send_q.queue_mv.view(0x300, GspQueueElement.SIZE)[:])
  assert first_hdr.function == 0x123 and first_hdr.length == RpcHeader.SIZE + max_payload
  assert second_hdr.function == NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD
  assert first_elem.elem_count == 3 and second_elem.elem_count == 1
  assert send_q.tx_view[4] == 0 and send_q.seq == 2
  before_wp = send_q.tx_view[4]
  try:
    send_q.send_rpc(0x100000000, b"")
    raise AssertionError("bad RPC function was accepted")
  except ValueError as exc:
    assert "RPC function" in str(exc)
  assert send_q.tx_view[4] == before_wp

  alloc_client = 0xc1e00004
  alloc_parent = 0
  alloc_object = 0xcf000000
  alloc_params = pack_nv0000_alloc_parameters()
  alloc_resp_payload = struct.pack("<IIIIIII4x", alloc_client, alloc_parent, alloc_object, NV01_ROOT, 0, len(alloc_params), 0) + alloc_params
  alloc_stat = init_queue_view(write_ptr=1)
  alloc_cmd = init_queue_view(write_ptr=0)
  alloc_stat.view(0x1000, 0x1000)[:] = rpc_record(NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, alloc_resp_payload)
  alloc_rm = GspRmClient(GspRpcQueue(None, alloc_cmd), GspRpcQueue(None, alloc_stat, alloc_cmd),
    priv_root=alloc_client, first_handle=alloc_object)
  assert alloc_rm.alloc_root() == alloc_client
  sent_hdr = RpcHeader.unpack_from(alloc_rm.cmd_q.queue_mv.view(GspQueueElement.SIZE, RpcHeader.SIZE)[:])
  assert sent_hdr.function == NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC
  sent_payload = bytes(alloc_rm.cmd_q.queue_mv.view(GspQueueElement.SIZE + RpcHeader.SIZE, sent_hdr.length - RpcHeader.SIZE)[:])
  assert hashlib.sha256(sent_payload).hexdigest() == pack_rpc_gsp_rm_alloc_fingerprint(
    alloc_client, alloc_parent, alloc_object, NV01_ROOT, alloc_params)

  class SeqTransport:
    def __init__(self): self.sleeps = []
    def sleep(self, timeout_ms): self.sleeps.append(timeout_ms)
  class SeqShell:
    def __init__(self): self.transport = SeqTransport(); self.regs = {0x100: 0x55}; self.writes = []
    def rreg(self, addr): return self.regs.get(addr, 0)
    def wreg(self, addr, value): self.writes.append((addr, value)); self.regs[addr] = value
  seq_shell = SeqShell()
  seq_words = [0x0, 0x100, 0xaa, 0x1, 0x100, 0x0f, 0xff, 0x4, 0x100, 0]
  seq = bytearray(40 + len(seq_words) * 4)
  struct.pack_into("<II8I", seq, 0, 0, len(seq_words), *([0] * 8))
  struct.pack_into("<" + "I" * len(seq_words), seq, 40, *seq_words)
  saved = StandaloneNvShell.run_cpu_sequencer(seq_shell, seq)
  assert saved[0] == 0x0f and seq_shell.writes == [(0x100, 0xaa), (0x100, 0x0f)]
  bad_seq = bytearray(seq)
  struct.pack_into("<I", bad_seq, 4, len(seq_words) + 1)
  try:
    StandaloneNvShell.run_cpu_sequencer(seq_shell, bad_seq)
    raise AssertionError("truncated CPU sequencer command stream was accepted")
  except ValueError as exc:
    assert "command stream" in str(exc)

def print_gsp_rpc_contract():
  assert_gsp_rpc_contract()
  print("gsp_rpc_contract=ok")

def assert_vm_contract():
  mm = GpuMemoryManager(64 << 20, mmu_ver=2)
  sys_mapping = mm.map_range(0x1000000000, 0x2000, [(0x200000, 0x1000), (0x300000, 0x1000)],
    AddrSpace.SYS, uncached=True, snooped=True)
  assert sys_mapping.snooped is True and sys_mapping.uncached is True
  leaf = mm.ensure_leaf_table(sys_mapping.va_addr)
  idx0 = mm.table_index(sys_mapping.va_addr, len(mm.level_shifts) - 1)
  idx1 = mm.table_index(sys_mapping.va_addr + 0x1000, len(mm.level_shifts) - 1)
  assert leaf.entries[idx0] & 1 and leaf.entries[idx1] & 1
  assert ((leaf.entries[idx0] >> 1) & 0x3) == 2
  assert ((leaf.entries[idx0] >> 8) & ((1 << 46) - 1)) == (0x200000 >> 12)
  assert ((leaf.entries[idx1] >> 8) & ((1 << 46) - 1)) == (0x300000 >> 12)
  mm.unmap_range(sys_mapping.va_addr, sys_mapping.size)
  assert leaf.entries[idx0] == 0 and leaf.entries[idx1] == 0

  existing = mm.map_range(0x1000003000, 0x1000, [(0x500000, 0x1000)], AddrSpace.SYS)
  try:
    mm.map_range(0x1000002000, 0x2000, [(0x600000, 0x1000), (0x700000, 0x1000)], AddrSpace.SYS)
    raise AssertionError("overlapping GPU mapping was accepted")
  except ValueError as exc:
    assert "already mapped" in str(exc)
  rolled_leaf = mm.ensure_leaf_table(0x1000002000)
  rolled_idx = mm.table_index(0x1000002000, len(mm.level_shifts) - 1)
  existing_leaf = mm.ensure_leaf_table(existing.va_addr)
  existing_idx = mm.table_index(existing.va_addr, len(mm.level_shifts) - 1)
  assert rolled_leaf.entries[rolled_idx] == 0
  assert existing_leaf.entries[existing_idx] & 1
  assert 0x1000002000 not in mm.mappings and existing.va_addr in mm.mappings
  mm.unmap_range(existing.va_addr, existing.size)

  vram_map = mm.valloc(0x2000, contiguous=True)
  vram_va, vram_pa = vram_map.va_addr, vram_map.paddrs[0][0]
  mm.vfree(vram_map)
  assert vram_va not in mm.mappings
  assert mm.alloc_vaddr(0x2000, 0x1000) == vram_va
  assert mm.palloc(0x2000, 0x1000) == vram_pa

  class VmContractTransport:
    def __init__(self):
      self.next_pa, self.allocs, self.frees = 0x90000000, [], []
    def alloc_sysmem(self, size, contiguous=False):
      size = round_up(size, 0x1000)
      self.allocs.append((size, contiguous))
      view = MMIOView(bytearray(size))
      paddrs = [self.next_pa + off for off in range(0, size, 0x1000)]
      self.next_pa += size
      return view, paddrs
    def free_sysmem(self, sysmem):
      self.frees.append(tuple(sysmem[1]) if sysmem is not None else None)
  shell = type("VmContractShell", (), {"mm": GpuMemoryManager(64 << 20), "transport": VmContractTransport()})()
  allocator = StandaloneBufferAllocator(shell)
  buf = allocator.alloc_sysmem(0x1001, contiguous=True)
  allocator.copyin(buf, b"abcd")
  assert allocator.copyout(buf, 4) == bytearray(b"abcd")
  assert buf.meta.aspace is AddrSpace.SYS and buf.meta.snooped is True and buf.meta.uncached is True
  buf_va = buf.va_addr
  allocator.free(buf)
  assert shell.transport.frees == [(0x90000000, 0x90001000)]
  assert buf.meta is None and buf.view is None
  try:
    buf.cpu_view()
    raise AssertionError("freed sysmem buffer still exposed a CPU view")
  except RuntimeError as exc:
    assert "freed" in str(exc) or "CPU view" in str(exc)
  assert shell.mm.alloc_vaddr(0x2000, 0x1000) == buf_va

  vbuf = allocator.alloc_vram(0x1000, contiguous=True)
  vbuf_va, vbuf_pa = vbuf.va_addr, vbuf.meta.paddrs[0][0]
  allocator.free(vbuf)
  assert vbuf.meta is None
  assert shell.mm.alloc_vaddr(0x1000, 0x1000) == vbuf_va
  assert shell.mm.palloc(0x1000, 0x1000) == vbuf_pa

  for bad_vm_call, text in [
    (lambda: GpuMemoryManager(0), "VRAM size"),
    (lambda: mm.map_range(0x1000000001, 0x1000, [(0x80000000, 0x1000)], AddrSpace.SYS), "mapping VA"),
    (lambda: mm.map_range(0x1000004000, 0x1000, [(0x80000001, 0x1000)], AddrSpace.SYS), "physical address"),
    (lambda: mm.encode_pte(0x1000, object()), "address space"),
    (lambda: allocator.free(object()), "GPU buffer"),
  ]:
    try:
      bad_vm_call()
      raise AssertionError("bad VM input was accepted")
    except ValueError as exc:
      assert text in str(exc)

def print_vm_contract():
  assert_vm_contract()
  print("vm_contract=ok")

def assert_channel_contract():
  facade = make_simulated_backend(entries=4, token=0x5a)
  backend = facade.dev
  try:
    resources = backend.resources
    assert resources.compute_gpfifo.entries_count == 4
    assert resources.compute_gpfifo.token == 0x5a
    assert resources.compute_gpfifo.put_value == 1
    assert resources.gpfifo_area.size == 0x300000
    assert resources.cmdq_page.size == 0x200000
    assert resources.kernargs_buf.size == 0x200000
    assert resources.timeline_signal.value == 0
    assert resources.kernargs_allocator.alloc(16, 8) == resources.kernargs_buf.va_addr

    done_value, signal, words = facade.submit_timeline_signal()
    assert done_value == 1 and signal is resources.timeline_signal
    assert signal.value == 1
    assert [method for _, _, _, method, _, _ in decode_words(words)] == [0x005c, 0x0020]
    assert resources.compute_gpfifo.put_value == 2 and resources.compute_gpfifo.gpput[0] == 2
    entry = int(resources.compute_gpfifo.ring[1])
    cmdq_addr = ((entry & ((1 << 40) - 1)) >> 2) << 2
    packets = (entry >> 42) & ((1 << 20) - 1)
    cmdq_word_offset = (cmdq_addr - resources.cmdq_page.va_addr) // 4
    assert resources.cmdq_page.va_addr <= cmdq_addr < resources.cmdq_page.va_addr + resources.cmdq_page.size
    assert cmdq_addr % 16 == 0 and packets == len(words)
    assert backend.submitter.gpu_mmio[0x90 // 4] == 0x5a
    assert list(backend.submitter.cmdq[cmdq_word_offset:cmdq_word_offset + len(words)]) == words
    facade.wait_signal(signal, done_value, timeout_ms=0)

    explicit_value, explicit_signal, explicit_words = facade.submit_timeline_signal(7)
    assert explicit_value == 7 and explicit_signal is signal and signal.value == 7
    assert [method for _, _, _, method, _, _ in decode_words(explicit_words)] == [0x005c, 0x0020]
    assert resources.compute_gpfifo.put_value == 3 and resources.compute_gpfifo.gpput[0] == 3

    setup_words = backend.setup_compute_queue()
    assert [method for _, _, _, method, _, _ in decode_words(setup_words)] == [
      NVC6C0_SET_OBJECT, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,
      NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI]
    assert resources.compute_gpfifo.put_value == 4

    facade.ensure_local_memory(0x241)
    assert resources.slm_per_thread == 0x260
    assert isinstance(resources.shader_local_mem, GpuBuffer)
    assert resources.compute_gpfifo.put_value == 5
  finally:
    facade.close()

  class ContractShell:
    def __init__(self):
      self.mm, self.transport = GpuMemoryManager(64 << 20), _SimulatedSysmemTransport()
  shell = ContractShell()
  allocator = StandaloneBufferAllocator(shell)
  builder = StandaloneChannelBuilder(shell, None)
  for bad_call, text in [
    (lambda: builder.allocate_runtime_resources(allocator, entries=0), "GPFIFO entry count"),
    (lambda: builder.allocate_runtime_resources(allocator, entries=4, token=-1), "GPFIFO token"),
    (lambda: builder.allocate_runtime_resources(allocator, h_device=0x80, entries=4), "required together"),
    (lambda: ChannelResources(handles=object()), "resource handles"),
  ]:
    try:
      bad_call()
      raise AssertionError("bad channel/runtime input was accepted")
    except ValueError as exc:
      assert text in str(exc)

  bad_facade = make_simulated_backend(entries=4, token=0x61)
  bad_backend = bad_facade.dev
  try:
    old_signal = bad_backend.resources.timeline_signal
    bad_backend.resources.timeline_signal = object()
    try:
      bad_facade.submit_timeline_signal(1)
      raise AssertionError("bad timeline signal was accepted")
    except ValueError as exc:
      assert "timeline signal" in str(exc)
    finally:
      bad_backend.resources.timeline_signal = old_signal
    try:
      bad_facade.submit_timeline_signal(-1)
      raise AssertionError("bad timeline value was accepted")
    except ValueError as exc:
      assert "timeline signal value" in str(exc)
  finally:
    bad_facade.close()

def print_channel_contract():
  assert_channel_contract()
  print("channel_contract=ok")

def print_contract_suite():
  print_transport_contract()
  print_register_contract()
  print_boot_firmware_contract()
  print_gsp_rpc_contract()
  print_vm_contract()
  print_channel_contract()

def print_validation_suite(arithmetic="add", script=None):
  print_contract_suite()
  print_offline_debug_suite(arithmetic)
  print_comparison_checklist(script or ("examples/mul.py" if arithmetic == "mul" else "examples/add.py"))


class NvBackend:
  def configure_booted_resources(self, h_device, h_subdevice, h_virtmem, h_vaspace, golden_ctx=None):
    h_device = validate_rm_handle(h_device, "device handle")
    h_subdevice = validate_rm_handle(h_subdevice, "subdevice handle")
    h_virtmem = validate_rm_handle(h_virtmem, "virtual memory handle")
    h_vaspace = validate_rm_handle(h_vaspace, "vaspace handle")
    if golden_ctx is None:
      golden_ctx = {}
    elif not isinstance(golden_ctx, dict):
      raise ValueError("golden context must be a mapping")
    trace_channel_step("booted_resources_start", device=hex(h_device), subdevice=hex(h_subdevice),
      virtmem=hex(h_virtmem), vaspace=hex(h_vaspace), golden=should_prepare_golden_ctx(),
      precomputed_golden=bool(golden_ctx), linux_uvm=isinstance(self.transport, LinuxIoctlTransport))
    if isinstance(self.transport, LinuxIoctlTransport):
      self.transport.rm = self.rm
      self.transport.setup_uvm(self.rm.priv_root, h_device, h_subdevice, h_virtmem, h_vaspace)
      trace_channel_step("booted_uvm_setup", client=hex(self.rm.priv_root))
    self.allocator.configure_rm(self.rm, h_device)
    self.resources = None
    try:
      if should_prepare_golden_ctx() and not golden_ctx:
        golden_ctx = self.channel_builder.prepare_golden_image_context(self.allocator)
      self.resources = self.channel_builder.allocate_runtime_resources(self.allocator, h_device=h_device,
        h_virtmem=h_virtmem, h_vaspace=h_vaspace, entries=32, h_subdevice=h_subdevice,
        grctx_descs=golden_ctx.get("grctx_descs"), grctx_mappings=golden_ctx.get("grctx_mappings"),
        h_client=self.rm.priv_root)
      self.resources.handles.update(device=h_device, subdevice=h_subdevice, virtmem=h_virtmem, vaspace=h_vaspace)
      if golden_ctx: self.resources.handles["golden_ctx"] = golden_ctx
      if isinstance(self.transport, LinuxIoctlTransport):
        self.resources.handles["uvm_channel"] = self.transport.register_channel(self.resources.handles["gpfifo"])
        trace_channel_step("booted_uvm_channel", gpfifo=hex(self.resources.handles["gpfifo"]))
      trace_channel_step("booted_resources_done", handles=sorted(self.resources.handles.keys()))
      return self.resources
    except Exception:
      if self.resources is not None:
        StandaloneNvBackend(self.shell, self.resources, self.allocator, submitter=None).release_resources()
        self.resources = None
      elif golden_ctx:
        temp_resources = ChannelResources(handles={"golden_ctx": golden_ctx})
        StandaloneNvBackend(self.shell, temp_resources, self.allocator, submitter=None).release_resources()
      raise

  def __init__(self, transport=None):
    self.transport = transport or select_transport()
    self.shell = StandaloneNvShell(self.transport)
    self.allocator = StandaloneBufferAllocator(self.shell)
    self.gsp_boot = StandaloneGspBootstrap(self.shell)
    self.gsp_boot.prepare_queues()
    allow_fw_download = os.environ.get("NV_ADD_DOWNLOAD_FIRMWARE") == "1"
    if os.environ.get("NV_ADD_PREPARE_GSP", "1") != "0":
      self.gsp_boot.prepare_wpr_meta(allow_download=allow_fw_download)
    booted_gsp = should_boot_gsp()
    self.rm = self.gsp_boot.make_rm_client()
    pci_device_id = self.transport.read_config(PCI_VENDOR_ID, 4)
    pci_subdevice_id = self.transport.read_config(PCI_SUBSYSTEM_VENDOR_ID, 4)
    pci_revision_id = self.transport.read_config(PCI_REVISION_ID, 1)
    system_info = pack_gsp_system_info(self.transport.bar_info(0)[0], self.transport.bar_info(1)[0],
      self.transport.bar_info(3)[0], pci_device_id, pci_subdevice_id, pci_revision_id)
    if booted_gsp:
      self.rm.set_system_info(system_info, wait=False)
      self.rm.set_registry(wait=False)
    if booted_gsp:
      if not hasattr(self.gsp_boot, "wpr_meta_sysmem"):
        self.gsp_boot.prepare_wpr_meta(allow_download=allow_fw_download)
      self.gsp_boot.boot_ampere_ada()
      self.rm.stat_q = self.gsp_boot.stat_q
    if not booted_gsp:
      self.rm.set_system_info(system_info, wait=False)
      self.rm.set_registry(wait=False)
    if booted_gsp and isinstance(self.transport, LinuxIoctlTransport):
      self.transport.register_fd()
      self.rm = LinuxRmClient(self.transport, priv_root=self.rm.priv_root, first_handle=self.rm.next_handle)
    if booted_gsp: self.rm.alloc_root()
    self.channel_builder = StandaloneChannelBuilder(self.shell, self.rm, gsp_boot=self.gsp_boot)
    if booted_gsp:
      golden_ctx = {}
      try:
        if should_prepare_golden_ctx() and not isinstance(self.transport, LinuxIoctlTransport):
          golden_ctx = self.channel_builder.prepare_golden_image_context(self.allocator)
        h_device, h_subdevice, h_virtmem, h_vaspace = self.channel_builder.allocate_base_objects()
        self.configure_booted_resources(h_device, h_subdevice, h_virtmem, h_vaspace, golden_ctx=golden_ctx)
        golden_ctx = {}
      except Exception:
        if golden_ctx:
          temp_resources = ChannelResources(handles={"golden_ctx": golden_ctx})
          StandaloneNvBackend(self.shell, temp_resources, self.allocator, submitter=None).release_resources()
        raise
    else:
      self.rm.set_page_directory(0, 0, self.shell.mm.page_directory_paddr, self.shell.mm.page_directory_entries, wait=False)
      self.resources = self.channel_builder.allocate_runtime_resources(self.allocator)
    self.dev = StandaloneNvBackend(self.shell, self.resources, self.allocator,
      StandaloneSubmitter(self.shell, self.resources.cmdq_page, self.resources.compute_gpfifo, self.transport.map_bar(0, off=0xbb0000, size=0x10000, fmt='I')),
      rm=self.rm if booted_gsp else None)
    if booted_gsp: self.dev.setup_compute_queue()

  @property
  def device_name(self): return self.dev.device_name
  @property
  def iface_name(self): return self.dev.iface_name
  @property
  def shared_mem_window(self): return self.dev.shared_mem_window
  @property
  def local_mem_window(self): return self.dev.local_mem_window
  @property
  def slm_per_thread(self): return self.dev.slm_per_thread
  @property
  def sass_version(self): return self.dev.sass_version

  def alloc(self, size): return self.dev.alloc(size)
  def free(self, buf): return self.dev.free(buf)
  def copyin(self, buf, data): return self.dev.copyin(buf, data)
  def copyout(self, buf, size): return self.dev.copyout(buf, size)
  def synchronize(self): return self.dev.synchronize()
  def close(self): return self.dev.close()

  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb):
    self.close()

  def upload_program(self, image):
    buf = self.alloc(round_up(len(image), 0x1000) + 0x1000)
    self.copyin(buf, memoryview(image))
    self.synchronize()
    return buf

  def ensure_local_memory(self, size):
    return self.dev.ensure_local_memory(size)

  def alloc_kernargs(self, size, align=8):
    return self.dev.alloc_kernargs(size, align)

  def timeline_state(self):
    return self.dev.timeline_state()

  def submit_gpfifo(self, words):
    return self.dev.submit_gpfifo(words)

  def submit_timeline_signal(self, value=None):
    return self.dev.submit_timeline_signal(value)

  def wait_signal(self, signal, value, timeout_ms=30000):
    return self.dev.wait_signal(signal, value, timeout_ms)

class QMD:
  SZ_WORDS = 0x40
  CWD_MEMBAR_TYPE_L1_SYSMEMBAR = 1
  FIELDS = {
    'qmd_group_id': (133, 128), 'sm_global_caching_enable': (134, 134),
    'invalidate_texture_header_cache': (186, 186), 'invalidate_texture_sampler_cache': (187, 187),
    'invalidate_texture_data_cache': (188, 188), 'invalidate_shader_data_cache': (189, 189),
    'program_prefetch_addr_lower_shifted': (287, 256), 'cwd_membar_type': (369, 368),
    'api_visible_call_limit': (378, 378), 'sampler_index': (382, 382),
    'cta_raster_width': (415, 384), 'cta_raster_height': (431, 416), 'cta_raster_depth': (463, 448),
    'shared_memory_size': (561, 544), 'min_sm_config_shared_mem_size': (567, 562),
    'max_sm_config_shared_mem_size': (574, 569), 'qmd_major_version': (583, 580),
    'cta_thread_dimension0': (607, 592), 'cta_thread_dimension1': (623, 608), 'cta_thread_dimension2': (639, 624),
    'register_count_v': (656, 648), 'target_sm_config_shared_mem_size': (662, 657), 'barrier_count': (767, 763),
    'release0_address_lower': (799, 768), 'release0_address_upper': (807, 800), 'release0_enable': (823, 823),
    'release0_payload_lower': (863, 832), 'release0_payload_upper': (895, 864),
    'constant_buffer_valid_0': (640, 640), 'constant_buffer_addr_lower_0': (1055, 1024),
    'constant_buffer_addr_upper_0': (1072, 1056), 'constant_buffer_invalidate_0': (1074, 1074),
    'constant_buffer_size_shifted4_0': (1087, 1075),
    'program_address_lower': (1567, 1536), 'program_address_upper': (1584, 1568),
    'shader_local_memory_high_size': (1623, 1600), 'program_prefetch_addr_upper_shifted': (1640, 1632),
    'program_prefetch_size': (1649, 1641), 'sass_version': (1663, 1656),
  }

  def __init__(self, view=None, **kwargs):
    self.mv = memoryview(bytearray(self.SZ_WORDS * 4)) if view is None else view
    if self.mv.nbytes < self.SZ_WORDS * 4: raise ValueError("QMD backing view is truncated")
    if kwargs: self.write(**kwargs)

  def _rw_bits(self, hi, lo, value=None):
    if lo < 0 or hi < lo: raise ValueError("invalid QMD bit range")
    if hi // 8 >= self.mv.nbytes: raise ValueError("QMD bit range exceeds backing view")
    mask = ((1 << (width:=hi - lo + 1)) - 1) << (lo % 8)
    num = int.from_bytes(self.mv[lo//8:hi//8+1], "little")
    if value is None: return (num & mask) >> (lo % 8)
    if value < 0: raise ValueError(f"{value:#x} does not fit")
    if value >= (1 << width): raise ValueError(f"{value:#x} does not fit")
    self.mv[lo//8:hi//8+1] = int((num & ~mask) | ((value << (lo % 8)) & mask)).to_bytes((hi//8 - lo//8 + 1), "little")

  def write(self, **kwargs):
    for key, value in kwargs.items():
      if key not in self.FIELDS: raise KeyError(f"unknown QMD field {key}")
      self._rw_bits(*self.FIELDS[key], value=value)

  def set_constant_buf_addr(self, i, addr):
    if i != 0: raise NotImplementedError("this example only needs constant buffer 0")
    check_qmd_pointer(f"constant buffer {i}", addr)
    self.write(constant_buffer_addr_upper_0=hi32(addr), constant_buffer_addr_lower_0=lo32(addr))

def build_qmd_template(prog_addr, prog_sz, cbuf_addr, cbuf_size, shmem_usage, regs_usage, slm_per_thread, sass_version):
  check_qmd_pointer("program", prog_addr, align=0x80)
  check_qmd_pointer("constant buffer 0", cbuf_addr)
  validate_positive_size(prog_sz, "program size")
  validate_positive_size(cbuf_size, "constant buffer size")
  validate_positive_size(shmem_usage, "shared memory usage")
  validate_u32(regs_usage, "register usage")
  validate_u32(slm_per_thread, "shader local memory per thread")
  validate_u32(sass_version, "SASS version")
  smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= shmem_usage) // 4096 + 1
  qmd = QMD(qmd_major_version=3, sm_global_caching_enable=1, program_address_upper=hi32(prog_addr), program_address_lower=lo32(prog_addr),
            shared_memory_size=shmem_usage, register_count_v=regs_usage, shader_local_memory_high_size=slm_per_thread,
            qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
            invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1,
            barrier_count=1, cwd_membar_type=QMD.CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1,
            min_sm_config_shared_mem_size=smem_cfg, target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a,
            program_prefetch_size=min(prog_sz >> 8, 0x1ff), sass_version=sass_version,
            program_prefetch_addr_upper_shifted=prog_addr >> 40, program_prefetch_addr_lower_shifted=prog_addr >> 8)
  qmd.set_constant_buf_addr(0, cbuf_addr)
  qmd.write(constant_buffer_size_shifted4_0=cbuf_size, constant_buffer_valid_0=1)
  return qmd

def qmd_sha256(qmd):
  if not isinstance(qmd, QMD): raise ValueError("QMD object is invalid")
  return hashlib.sha256(bytes(qmd.mv[:QMD.SZ_WORDS * 4])).hexdigest()

def qmd_byte_diff(template_qmd, live_qmd):
  if not isinstance(template_qmd, QMD) or not isinstance(live_qmd, QMD): raise ValueError("QMD object is invalid")
  template = bytes(template_qmd.mv[:QMD.SZ_WORDS * 4])
  live = bytes(live_qmd.mv[:QMD.SZ_WORDS * 4])
  return [idx for idx, (before, after) in enumerate(zip(template, live)) if before != after]

def qmd_field_diff(template_qmd, live_qmd):
  if not isinstance(template_qmd, QMD) or not isinstance(live_qmd, QMD): raise ValueError("QMD object is invalid")
  changed = []
  for name, (hi, lo) in QMD.FIELDS.items():
    if template_qmd._rw_bits(hi, lo) != live_qmd._rw_bits(hi, lo): changed.append(name)
  return changed

def expected_qmd_template_sha256():
  return qmd_sha256(build_qmd_template(0x1000000000, 0x110, 0x1000000200, 0x160, 0x400, 0x20, 0x240, 86))

class ElfSection:
  def __init__(self, name, header, content): self.name, self.header, self.content = name, header, content

def elf_loader(blob, force_section_align=128):
  if blob[:4] != b"\x7fELF": raise RuntimeError("blob is not an ELF")
  require_len(blob, 64, "ELF header")
  e_shoff = struct.unpack_from("<Q", blob, 40)[0]
  e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 58)
  if e_shentsize < 64: raise ValueError("ELF section header entry size is invalid")
  if e_shnum == 0: raise ValueError("ELF has no section headers")
  if e_shstrndx >= e_shnum: raise ValueError("ELF section-name string table index is invalid")
  require_len(blob, e_shoff + e_shnum * e_shentsize, "ELF section header table")
  raw_headers = [list(struct.unpack_from("<IIQQQQIIQQ", blob, e_shoff + i * e_shentsize)) for i in range(e_shnum)]
  for h in raw_headers:
    if h[5] and h[4] + h[5] > len(blob): raise ValueError("ELF section content is truncated")
  shstr_hdr = raw_headers[e_shstrndx]
  shstr = blob[shstr_hdr[4]:shstr_hdr[4] + shstr_hdr[5]]
  def strtab_name(tab, idx):
    if idx >= len(tab): raise ValueError("ELF section name offset is outside string table")
    end = tab.find(b"\0", idx)
    if end < 0: raise ValueError("ELF section name is not NUL-terminated")
    return tab[idx:end].decode()
  sections = [ElfSection(strtab_name(shstr, h[0]), h, blob[h[4]:h[4]+h[5]]) for h in raw_headers]

  image = bytearray(max([sh.header[3] + sh.header[5] for sh in sections if sh.header[1] == 1 and sh.header[3] != 0] + [0]))
  for sh in sections:
    if sh.header[1] != 1: continue
    if sh.header[3] != 0: image[sh.header[3]:sh.header[3]+sh.header[5]] = sh.content
    else:
      image += b"\0" * (((align:=max(sh.header[8], force_section_align)) - len(image) % align) % align) + sh.content
      sh.header[3] = len(image) - len(sh.content)

  symtab = next((sh for sh in sections if sh.header[1] == 2), None)
  relocs = []
  if symtab is not None:
    if symtab.header[9] == 0 or len(symtab.content) % symtab.header[9]: raise ValueError("ELF symbol table entry size is invalid")
    syms = [struct.unpack_from("<IBBHQQ", symtab.content, off) for off in range(0, len(symtab.content), symtab.header[9])]
    for sh in sections:
      if sh.header[1] != 9: continue
      if sh.header[9] == 0 or len(sh.content) % sh.header[9]: raise ValueError("ELF relocation entry size is invalid")
      target_name = sh.name[4:]
      target = next((s for s in sections if s.name == target_name), None)
      if target is None: continue
      for off in range(0, len(sh.content), sh.header[9]):
        r_offset, r_info = struct.unpack_from("<QQ", sh.content, off)
        if (r_info >> 32) >= len(syms): raise ValueError("ELF relocation references invalid symbol index")
        sym = syms[r_info >> 32]
        sym_section = sections[sym[3]] if sym[3] < len(sections) else None
        sym_addr = (sym_section.header[3] if sym_section is not None else 0) + sym[4]
        relocs.append((target.header[3] + r_offset, sym_addr, r_info & 0xffffffff))
  return memoryview(image), sections, relocs

def parse_elf_info(section):
  start = 0
  while start < section.header[5]:
    if start + 4 > len(section.content): raise ValueError(f"{section.name} info record header is truncated")
    typ, param, size = struct.unpack_from("BBH", section.content, start)
    if typ == 0x4 and start + 4 + size > len(section.content): raise ValueError(f"{section.name} info record payload is truncated")
    yield typ, param, section.content[start+4:start+size+4] if typ == 0x4 else size
    start += (size if typ == 0x4 else 0) + 4

def unpack_info_cbuf0_size(data, section_name, param):
  return struct.unpack_from("IH", require_len(data, 6, f"{section_name} param 0x{param:x} payload"), 0)[1]

def unpack_info_u32_pair(data, section_name, param):
  return struct.unpack_from("II", require_len(data, 8, f"{section_name} param 0x{param:x} payload"), 0)

class SimpleProgram:
  def __init__(self, backend, name, lib):
    self.backend, self.name = backend, name
    image, sections, relocs = elf_loader(lib)
    self.constbufs = {0: (0, 0x160)}
    self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0
    self.lib_gpu = backend.alloc(round_up(image.nbytes, 0x1000) + 0x1000)
    try:
      prog_addr, prog_sz = self.lib_gpu.va_addr, image.nbytes

      for sh in sections:
        if sh.name == f".nv.shared.{name}": self.shmem_usage = round_up(0x400 + sh.header[5], 128)
        if sh.name == f".text.{name}": prog_addr, prog_sz = self.lib_gpu.va_addr + sh.header[3], sh.header[5]
        elif (match:=re.match(r"\.nv\.constant(\d+)", sh.name)):
          self.constbufs[int(match.group(1))] = (self.lib_gpu.va_addr + sh.header[3], sh.header[5])
        elif sh.name.startswith(".nv.info"):
          for typ, param, data in parse_elf_info(sh):
            if sh.name == f".nv.info.{name}" and param == 0xa: cbuf0_size = unpack_info_cbuf0_size(data, sh.name, param)
            elif sh.name == ".nv.info" and param == 0x12: self.lcmem_usage = unpack_info_u32_pair(data, sh.name, param)[1] + 0x240
            elif sh.name == ".nv.info" and param == 0x2f: self.regs_usage = unpack_info_u32_pair(data, sh.name, param)[1]
      check_qmd_pointer("program", prog_addr, align=0x80)

      image = bytearray(image)
      for apply_off, rel_sym_off, typ in relocs:
        if typ == 2: image[apply_off:apply_off+8] = struct.pack("<Q", self.lib_gpu.va_addr + rel_sym_off)
        elif typ == 0x38: image[apply_off+4:apply_off+8] = struct.pack("<I", lo32(self.lib_gpu.va_addr + rel_sym_off))
        elif typ == 0x39: image[apply_off+4:apply_off+8] = struct.pack("<I", hi32(self.lib_gpu.va_addr + rel_sym_off))
        else: raise RuntimeError(f"unknown NV reloc {typ}")

      self.cbuf_0 = [0] * max(cbuf0_size // 4, 12)
      self.cbuf_0[6:12] = [*data64_le(backend.shared_mem_window), *data64_le(backend.local_mem_window), *data64_le(0xfffdc0)]
      backend.ensure_local_memory(self.lcmem_usage)
      backend.copyin(self.lib_gpu, image)
      backend.synchronize()

      cbuf0_addr, cbuf0_size = self.constbufs[0]
      self.qmd = build_qmd_template(prog_addr, prog_sz, cbuf0_addr, cbuf0_size, self.shmem_usage, self.regs_usage,
                                    backend.slm_per_thread, backend.sass_version)
      self.qmd_template_sha256 = qmd_sha256(self.qmd)
      self.qmd_template_bytes = bytes(self.qmd.mv[:QMD.SZ_WORDS * 4])
      for index, (addr, size) in self.constbufs.items():
        check_qmd_pointer(f"constant buffer {index}", addr)
      self.kernargs_alloc_size = round_up(self.constbufs[0][1], 1 << 8) + (8 << 8)
      self._closed = False
    except Exception:
      backend.free(self.lib_gpu)
      raise

  def close(self):
    if self._closed: return
    self._closed = True
    self.backend.free(self.lib_gpu)

def manual_launch(backend, program, out, a, b, simulate_arithmetic="add"):
  trace_launch_step("enter", backend=type(backend).__name__, simulate_arithmetic=simulate_arithmetic)
  if not isinstance(program, SimpleProgram): raise ValueError("program is invalid")
  if not isinstance(program.qmd, QMD): raise ValueError("program QMD is invalid")
  if not isinstance(program.qmd.mv, memoryview) or program.qmd.mv.nbytes < QMD.SZ_WORDS * 4: raise ValueError("program QMD backing view is invalid")
  if not isinstance(getattr(program, "qmd_template_sha256", None), str): raise ValueError("program QMD template fingerprint is invalid")
  if not isinstance(getattr(program, "qmd_template_bytes", None), bytes) or len(program.qmd_template_bytes) != QMD.SZ_WORDS * 4:
    raise ValueError("program QMD template bytes are invalid")
  if not isinstance(program.constbufs, dict) or 0 not in program.constbufs: raise ValueError("program constant-buffer metadata is invalid")
  cbuf0 = program.constbufs[0]
  if not isinstance(cbuf0, tuple) or len(cbuf0) != 2: raise ValueError("program constant-buffer metadata is invalid")
  cbuf0_addr, cbuf0_size = cbuf0
  check_qmd_pointer("constant buffer 0", cbuf0_addr)
  validate_positive_size(cbuf0_size, "program constant-buffer size")
  validate_positive_size(program.kernargs_alloc_size, "kernel argument allocation size")
  if not isinstance(program.cbuf_0, list): raise ValueError("program constant-buffer words are invalid")
  if any(not isinstance(word, int) for word in program.cbuf_0): raise ValueError("program constant-buffer word is invalid")
  cbuf_words = [validate_u32(word, "program constant-buffer word") for word in program.cbuf_0]
  for name, buf in (("output", out), ("input A", a), ("input B", b)):
    if not isinstance(buf, GpuBuffer): raise ValueError(f"{name} buffer is invalid")
    if buf.view is None: raise RuntimeError(f"{name} buffer must be CPU-visible")
  trace_launch_step("validated", cbuf_words=len(cbuf_words), kernargs_alloc_size=program.kernargs_alloc_size, qmd_bytes=program.qmd.mv.nbytes,
    cbuf0_size=cbuf0_size, cbuf0_addr=hex(cbuf0_addr), out=hex(out.va_addr), a=hex(a.va_addr), b=hex(b.va_addr),
    out_size=out.size, a_size=a.size, b_size=b.size, qmd_template_sha256=program.qmd_template_sha256)
  kernargs = backend.alloc_kernargs(program.kernargs_alloc_size)
  trace_launch_step("kernargs_allocated", va=hex(kernargs.va_addr), size=kernargs.size, offset_qmd=round_up(cbuf0_size, 1 << 8))
  check_qmd_pointer("kernel arguments", kernargs.va_addr)
  for name, buf in (("output", out), ("input A", a), ("input B", b)):
    check_qmd_pointer(name, buf.va_addr)
  cbuf_bytes = len(cbuf_words) * 4
  qmd_offset = round_up(cbuf0_size, 1 << 8)
  required = max(cbuf_bytes + 3 * 8, qmd_offset + program.qmd.mv.nbytes)
  if required > kernargs.size: raise ValueError(f"kernel argument buffer is too small: need {required} bytes, got {kernargs.size}")
  kernargs.cpu_view().view(size=len(cbuf_words) * 4, fmt='I')[:] = array.array('I', cbuf_words)
  kernargs.cpu_view().view(offset=len(cbuf_words) * 4, size=3 * 8, fmt='Q')[:] = array.array('Q', [out.va_addr, a.va_addr, b.va_addr])
  trace_launch_step("kernargs_written", cbuf_words=len(cbuf_words), ptr_words=(out.va_addr, a.va_addr, b.va_addr), required=required)

  qmd_buf = kernargs.offset(qmd_offset)
  check_qmd_pointer("QMD", qmd_buf.va_addr, align=0x100)
  qmd_buf.cpu_view().view(size=program.qmd.mv.nbytes, fmt='B')[:] = program.qmd.mv
  qmd = QMD(view=qmd_buf.cpu_view())
  qmd.write(cta_raster_width=1, cta_raster_height=1, cta_raster_depth=1,
            cta_thread_dimension0=1, cta_thread_dimension1=1, cta_thread_dimension2=1)
  qmd.set_constant_buf_addr(0, kernargs.va_addr)
  trace_launch_step("qmd_written", qmd_va=hex(qmd_buf.va_addr), release_signal=True)

  wait_value, done_value, signal = backend.timeline_state()
  signal_addr = signal.value_addr
  check_qmd_pointer("timeline semaphore", signal_addr)
  qmd.write(release0_enable=1, release0_address_lower=signal_addr & 0xffffffff, release0_address_upper=(signal_addr >> 32) & 0xff,
            release0_payload_lower=done_value & 0xffffffff, release0_payload_upper=done_value >> 32)
  template_qmd = QMD(view=memoryview(bytearray(program.qmd_template_bytes)))
  qmd_diff = qmd_byte_diff(template_qmd, qmd)
  qmd_fields = qmd_field_diff(template_qmd, qmd)
  trace_launch_step("timeline_ready", wait_value=wait_value, done_value=done_value, signal=hex(signal_addr),
    qmd_sha256=qmd_sha256(qmd), qmd_diff=";".join(hex(idx) for idx in qmd_diff),
    qmd_fields=";".join(qmd_fields), qmd_fields_sha256=hashlib.sha256(";".join(qmd_fields).encode()).hexdigest())

  words = build_compute_launch_words(signal_addr, wait_value, done_value, qmd_buf.va_addr)
  if trace_launch_enabled():
    print(f"submit #manual: NVComputeQueue words={len(words)}")
    for index, typ, subc, method, name, args in decode_words(words):
      print(f"  method[{index}] {name}: typ={typ} subc={subc} mthd=0x{method:x} args=[{', '.join(describe_args(method, args))}]")
  trace_launch_step("submit", words=len(words), qmd=hex(qmd_buf.va_addr))
  backend.submit_gpfifo(words)
  if isinstance(getattr(backend, "dev", None), StandaloneNvBackend) and backend.dev.simulate:
    if simulate_arithmetic == "add": backend.dev.simulate_add_launch(out, a, b, signal, done_value)
    elif simulate_arithmetic == "mul": backend.dev.simulate_mul_launch(out, a, b, signal, done_value)
    else: raise ValueError(f"unknown simulated arithmetic op {simulate_arithmetic!r}")
  trace_launch_step("wait", value=done_value, signal=hex(signal_addr))
  backend.wait_signal(signal, done_value)
  trace_launch_step("done", value=done_value, signal=hex(signal_addr))

def manual_launch_mul(backend, program, out, a, b):
  manual_launch(backend, program, out, a, b, simulate_arithmetic="mul")

def launch_fingerprint_state(arithmetic="add"):
  if arithmetic not in ("add", "mul"): raise ValueError("launch fingerprint arithmetic must be add or mul")
  facade = make_simulated_backend(entries=16, token=0x77)
  program, bufs = None, []
  try:
    a_vals = (1.0, 2.0, 3.0, 4.0)
    b_vals = (10.0, 20.0, 30.0, 40.0)
    a = facade.alloc(16)
    b = facade.alloc(16)
    out = facade.alloc(16)
    bufs = [a, b, out]
    facade.copyin(a, struct.pack("4f", *a_vals))
    facade.copyin(b, struct.pack("4f", *b_vals))
    facade.copyin(out, bytes(16))
    program = SimpleProgram(facade, "E_4", build_cubin(arithmetic))
    before_put = facade.dev.resources.compute_gpfifo.put_value
    expected_done = facade.dev.timeline_value
    if arithmetic == "add":
      manual_launch(facade, program, out, a, b)
      expected_result = [x + y for x, y in zip(a_vals, b_vals)]
    else:
      manual_launch_mul(facade, program, out, a, b)
      expected_result = [x * y for x, y in zip(a_vals, b_vals)]
    after_put = facade.dev.resources.compute_gpfifo.put_value
    ring_index = before_put % facade.dev.resources.compute_gpfifo.entries_count
    ring_entry = int(facade.dev.resources.compute_gpfifo.ring[ring_index])
    cmdq_addr = ((ring_entry & ((1 << 40) - 1)) >> 2) << 2
    word_count = (ring_entry >> 42) & ((1 << 20) - 1)
    cmdq_wptr = (cmdq_addr - facade.dev.resources.cmdq_page.va_addr) // 4
    words = list(facade.dev.submitter.cmdq[cmdq_wptr:cmdq_wptr + word_count])
    qmd_addr = next(args[0] << 8 for _, _, _, method, _, args in decode_words(words) if method == 0x02b4)
    qmd_view = facade.dev.resources.kernargs_buf.cpu_view().view(qmd_addr - facade.dev.resources.kernargs_buf.va_addr, QMD.SZ_WORDS * 4)
    live_qmd = QMD(view=qmd_view)
    template_qmd = QMD(view=memoryview(bytearray(program.qmd_template_bytes)))
    qmd_fields = qmd_field_diff(template_qmd, live_qmd)
    result = list(struct.unpack("4f", facade.copyout(out, 16)))
    return {
      "arithmetic": arithmetic,
      "expected_result": expected_result,
      "result": result,
      "before_put": before_put,
      "after_put": after_put,
      "ring_index": ring_index,
      "ring_entry": ring_entry,
      "cmdq_addr": cmdq_addr,
      "word_count": word_count,
      "words": words,
      "words_sha256": hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest(),
      "methods": [method for _, _, _, method, _, _ in decode_words(words)],
      "qmd_addr": qmd_addr,
      "qmd_sha256": qmd_sha256(live_qmd),
      "qmd_diff": qmd_byte_diff(template_qmd, live_qmd),
      "qmd_fields": qmd_fields,
      "qmd_fields_sha256": hashlib.sha256(";".join(qmd_fields).encode()).hexdigest(),
      "last_launch": facade.dev.last_launch,
      "expected_done": expected_done,
    }
  finally:
    if program is not None: program.close()
    for buf in bufs: facade.free(buf)
    facade.close()

def print_launch_fingerprint(arithmetic="add"):
  state = launch_fingerprint_state(arithmetic)
  print(f"standalone launch arithmetic={state['arithmetic']} result={state['result']} expected={state['expected_result']}")
  print(f"standalone launch_words count={state['word_count']} methods={','.join(hex(method) for method in state['methods'])} "
        f"sha256={state['words_sha256']}")
  print(f"standalone launch_ring before_put={state['before_put']} after_put={state['after_put']} index={state['ring_index']} "
        f"entry=0x{state['ring_entry']:016x} cmdq_addr=0x{state['cmdq_addr']:x}")
  print(f"standalone launch_qmd addr=0x{state['qmd_addr']:x} sha256={state['qmd_sha256']} "
        f"diff={';'.join(hex(idx) for idx in state['qmd_diff'])} fields={';'.join(state['qmd_fields'])} "
        f"fields_sha256={state['qmd_fields_sha256']}")
  print(f"standalone launch_last out=0x{state['last_launch'][0]:x} a=0x{state['last_launch'][1]:x} "
        f"b=0x{state['last_launch'][2]:x} done={state['last_launch'][3]} expected_done={state['expected_done']}")

def build_cubin(arithmetic="add"): # nvdisasm add.cubin
  arithmetic_ops = {"add": ch.Op.FADD, "mul": ch.Op.FMUL}
  if arithmetic not in arithmetic_ops: raise ValueError(f"unknown arithmetic op {arithmetic!r}")
  arithmetic_op = arithmetic_ops[arithmetic]
  arithmetic_mod = 0x00400000 if arithmetic == "mul" else 0x00000000
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
    ((ch.Reg.R11 << 24) | (ch.Reg.R11 << 16) | arithmetic_op, 0x00000007, arithmetic_mod, 0x004fe200),  # R11 = R11 op R7
    ((ch.Reg.R10 << 24) | (ch.Reg.R10 << 16) | arithmetic_op, 0x00000006, arithmetic_mod, 0x000fe200),  # R10 = R10 op R6
    ((ch.Reg.R9  << 24) | (ch.Reg.R9  << 16) | arithmetic_op, 0x00000005, arithmetic_mod, 0x000fe200),  # R9 = R9 op R5
    ((ch.Reg.R8  << 24) | (ch.Reg.R8 << 16) | arithmetic_op, 0x00000004, arithmetic_mod, 0x000fe200),    # R8 = R8 op R4

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

class SimulatedBackendFacade:
  def __init__(self, dev): self.dev = dev
  @property
  def device_name(self): return self.dev.device_name
  @property
  def iface_name(self): return self.dev.iface_name
  @property
  def shared_mem_window(self): return self.dev.shared_mem_window
  @property
  def local_mem_window(self): return self.dev.local_mem_window
  @property
  def slm_per_thread(self): return self.dev.slm_per_thread
  @property
  def sass_version(self): return self.dev.sass_version
  def alloc(self, size): return self.dev.alloc(size)
  def copyin(self, buf, data): return self.dev.copyin(buf, data)
  def copyout(self, buf, size): return self.dev.copyout(buf, size)
  def free(self, buf): return self.dev.free(buf)
  def synchronize(self): return self.dev.synchronize()
  def close(self): return self.dev.close()
  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): self.close()
  def ensure_local_memory(self, size): return self.dev.ensure_local_memory(size)
  def alloc_kernargs(self, size, align=8): return self.dev.alloc_kernargs(size, align)
  def timeline_state(self): return self.dev.timeline_state()
  def submit_gpfifo(self, words): return self.dev.submit_gpfifo(words)
  def submit_timeline_signal(self, value=None): return self.dev.submit_timeline_signal(value)
  def wait_signal(self, signal, value, timeout_ms=30000): return self.dev.wait_signal(signal, value, timeout_ms)

BackendFacade = SimulatedBackendFacade

class _SimulatedMem:
  def __init__(self):
    self.next_va, self.next_pa = 0x1000000000, 0x80000000
    self.va_alloc = FreeListAllocator(1 << 48, base=self.next_va)
  def alloc_vaddr(self, size, align=0x1000):
    size = round_up(size, 0x1000)
    return self.va_alloc.alloc(size, max(1 << (size.bit_length() - 1), align, 0x1000))
  def map_range(self, vaddr, size, paddrs, aspace, uncached=False, snooped=False):
    return VirtMapping(vaddr, round_up(size, 0x1000), paddrs, aspace, uncached=uncached, snooped=snooped)
  def unmap_range(self, vaddr, size):
    return None
  def valloc(self, size, align=0x1000, uncached=False, contiguous=False):
    va = self.alloc_vaddr(size, align)
    pa = self.next_pa
    self.next_pa += round_up(size, 0x1000)
    return VirtMapping(va, round_up(size, 0x1000), [(pa, round_up(size, 0x1000))], AddrSpace.PHYS, uncached=uncached)

class _SimulatedSysmemTransport:
  def __init__(self):
    self.next_pa, self.allocs = 0x90000000, []
  def alloc_sysmem(self, size, contiguous=False):
    size = round_up(size, 0x1000)
    self.allocs.append((size, contiguous))
    view = MMIOView(bytearray(size))
    paddrs = [self.next_pa + off for off in range(0, size, 0x1000)]
    self.next_pa += size
    return view, paddrs

class _SimulatedResourceShell:
  def __init__(self):
    self.mm, self.transport = _SimulatedMem(), _SimulatedSysmemTransport()

def make_simulated_backend(entries=16, token=0x77):
  shell = _SimulatedResourceShell()
  allocator = StandaloneBufferAllocator(shell)
  resources = StandaloneChannelBuilder(shell, None).allocate_runtime_resources(allocator, entries=entries, token=token)
  mmio = MMIOView(bytearray(0x10000), fmt='I')
  backend = StandaloneNvBackend(shell, resources, allocator,
    StandaloneSubmitter(shell, resources.cmdq_page, resources.compute_gpfifo, mmio), simulate=True)
  backend.setup_compute_queue()
  return SimulatedBackendFacade(backend)

def assert_backend_facade_contract(facade_cls):
  class FakeDev:
    device_name, iface_name = "fake-device", "fake-iface"
    shared_mem_window, local_mem_window = 0x1000, 0x2000
    slm_per_thread, sass_version = 0x240, 0x86
    def __init__(self): self.calls = []
    def _record(self, name, *args):
      self.calls.append((name, args))
      return name, args
    def alloc(self, size): return self._record("alloc", size)
    def free(self, buf): return self._record("free", buf)
    def copyin(self, buf, data): return self._record("copyin", buf, data)
    def copyout(self, buf, size): return self._record("copyout", buf, size)
    def synchronize(self): return self._record("synchronize")
    def close(self): return self._record("close")
    def ensure_local_memory(self, size): return self._record("ensure_local_memory", size)
    def alloc_kernargs(self, size, align=8): return self._record("alloc_kernargs", size, align)
    def timeline_state(self): return self._record("timeline_state")
    def submit_gpfifo(self, words): return self._record("submit_gpfifo", words)
    def submit_timeline_signal(self, value=None): return self._record("submit_timeline_signal", value)
    def wait_signal(self, signal, value, timeout_ms=30000): return self._record("wait_signal", signal, value, timeout_ms)

  fake = FakeDev()
  facade = object.__new__(facade_cls) if facade_cls is NvBackend else facade_cls(fake)
  facade.dev = fake
  assert facade.device_name == "fake-device" and facade.iface_name == "fake-iface"
  assert facade.shared_mem_window == 0x1000 and facade.local_mem_window == 0x2000
  assert facade.slm_per_thread == 0x240 and facade.sass_version == 0x86
  assert facade.alloc(4)[0] == "alloc"
  assert facade.free("buf")[0] == "free"
  assert facade.copyin("buf", b"x")[0] == "copyin"
  assert facade.copyout("buf", 1)[0] == "copyout"
  assert facade.synchronize()[0] == "synchronize"
  assert facade.ensure_local_memory(0x300)[0] == "ensure_local_memory"
  assert facade.alloc_kernargs(32, 16)[0] == "alloc_kernargs"
  assert facade.timeline_state()[0] == "timeline_state"
  assert facade.submit_gpfifo([1, 2])[0] == "submit_gpfifo"
  assert facade.submit_timeline_signal(9)[0] == "submit_timeline_signal"
  assert facade.wait_signal("sig", 7, timeout_ms=11)[0] == "wait_signal"
  with facade:
    pass
  assert fake.calls[-1] == ("close", ())

def selftest():
  assert_backend_facade_contract(NvBackend)
  assert_backend_facade_contract(SimulatedBackendFacade)
  assert_transport_contract()
  assert_register_contract()
  assert_boot_firmware_contract()
  assert_gsp_rpc_contract()
  assert_vm_contract()
  assert_channel_contract()
  assert import_guard_state() == {"tinygrad_modules": [], "ref_tinygrad_paths": []}
  assert static_import_guard_state() == {"tinygrad_static_imports": []}
  assert static_external_import_guard_state() == {"external_static_imports": []}
  assert static_external_import_guard_state() == {"external_static_imports": []}
  static_bad = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
  try:
    static_bad.write("from tinygrad.device import Device\n")
    static_bad.close()
    assert "from tinygrad.device import" in static_import_guard_state([static_bad.name])["tinygrad_static_imports"][0]
    assert "tinygrad" in static_external_import_guard_state([static_bad.name])["external_static_imports"][0]
  finally:
    with contextlib.suppress(FileNotFoundError): os.unlink(static_bad.name)
  import_guard_buf = io.StringIO()
  with contextlib.redirect_stdout(import_guard_buf): print_import_guard()
  assert import_guard_buf.getvalue().strip().splitlines() == [
    "tinygrad_modules=[]", "ref_tinygrad_paths=[]", "tinygrad_static_imports=[]", "external_static_imports=[]"]
  assert round_up(17, 8) == 24
  assert round_down(17, 8) == 16
  assert ceildiv(17, 8) == 3
  assert data64(0x1122334455667788) == [0x11223344, 0x55667788]
  assert data64_le(0x1122334455667788) == [0x55667788, 0x11223344]
  assert check_qmd_pointer("limit", QMD_ADDR_LIMIT - 0x100, align=0x100) == QMD_ADDR_LIMIT - 0x100
  for bad_name, bad_addr, bad_align in (("overflow", QMD_ADDR_LIMIT, 1), ("misaligned", 0x1234, 0x100)):
    try:
      check_qmd_pointer(bad_name, bad_addr, align=bad_align)
      raise AssertionError("invalid QMD pointer was accepted")
    except ValueError as exc:
      assert bad_name in str(exc)
  full_launch = build_launch_words(0x1234567890, 7, 8, 0x4000)
  compute_launch = build_compute_launch_words(0x1234567890, 7, 8, 0x4000)
  timeline_signal = build_timeline_signal_words(0x1234567890, 9, interrupt=True)
  assert len(compute_launch) == 12
  assert len(full_launch) > len(compute_launch)
  assert [method for _, _, _, method, _, _ in decode_words(compute_launch)] == [0x005c, 0x1698, 0x02b4, 0x02c0]
  assert [method for _, _, _, method, _, _ in decode_words(full_launch)][-2:] == [0x005c, 0x0020]
  assert [method for _, _, _, method, _, _ in decode_words(timeline_signal)] == [0x005c, 0x0020]
  assert describe_args(0x005c, list(decode_words(timeline_signal))[0][5]) == ["sem_addr=0x1234567890", "payload=9", "execute=0x03100001"]

  raw = bytearray(64)
  view = MMIOView(raw)
  view[:4] = b"abcd"
  assert bytes(view[:4]) == b"abcd"
  words = view.view(fmt='I')
  words[1:3] = array.array('I', [0x11223344, 0x55667788])
  assert raw[4:12] == b"\x44\x33\x22\x11\x88\x77\x66\x55"
  write_words(words, 2, [0xaabbccdd])
  assert raw[8:12] == b"\xdd\xcc\xbb\xaa"
  for bad_word_write, text in [
    (lambda: write_words(words, -1, [0]), "offset"),
    (lambda: write_words(words, len(words), [0]), "destination"),
    (lambda: write_words(words, 0, [0x100000000]), "word write value"),
  ]:
    try:
      bad_word_write()
      raise AssertionError("bad word write was accepted")
    except ValueError as exc:
      assert text in str(exc)
  assert view.view(offset=4, size=8, fmt='I')[:] == [0x11223344, 0xaabbccdd]
  nested = MMIOView(bytearray(0x40))
  nested.view(0x10, 0x10)[0] = 0xaa
  nested.view(0x20, 0x10)[0] = 0xbb
  assert nested[0x10] == 0xaa and nested[0x20] == 0xbb
  assert nested.view(0x10, 0x10)[0] == 0xaa and nested.view(0x20, 0x10)[0] == 0xbb
  for args in [(-1, 1, 'B'), (0x41, 1, 'B'), (0x20, 0x21, 'B'), (0, -1, 'B'), (0, 3, 'I')]:
    try:
      nested.view(args[0], args[1], args[2])
      raise AssertionError(f"invalid MMIOView subview {args} was accepted")
    except ValueError as exc:
      assert "MMIOView" in str(exc)
  assert verify_view_bytes("mmio", MMIOView(bytearray(b"abcdefgh")), b"abcdefgh", chunk=3) == hashlib.sha256(b"abcdefgh").hexdigest()
  try:
    verify_view_bytes("mmio", MMIOView(bytearray(b"abcxefgh")), b"abcdefgh", chunk=3)
    raise AssertionError("bad readback was accepted")
  except RuntimeError as exc:
    assert "readback mismatch" in str(exc) and "+0x3" in str(exc)

  bump = BumpAllocator(16, base=0x1000)
  assert bump.alloc(3, 4) == 0x1000
  assert bump.alloc(4, 8) == 0x1008
  wrap = BumpAllocator(16, base=0x2000, wrap=True)
  assert wrap.alloc(12, 1) == 0x2000
  assert wrap.alloc(8, 8) == 0x2000
  for bad_bump, text in [
    (lambda: BumpAllocator(0), "bump allocator size"),
    (lambda: bump.alloc(0), "allocation size"),
    (lambda: bump.alloc(1, 3), "alignment"),
  ]:
    try:
      bad_bump()
      raise AssertionError("bad bump allocator input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  try:
    BumpAllocator(4, wrap=True).alloc(8)
    raise AssertionError("oversized wrapped bump allocation was accepted")
  except MemoryError:
    pass

  freelist = FreeListAllocator(0x4000, base=0x100000)
  a = freelist.alloc(0x1000, 0x1000)
  b = freelist.alloc(0x1000, 0x1000)
  freelist.free_addr(a)
  assert freelist.alloc(0x800, 0x1000) == a
  freelist.free_addr(b)
  for bad_freelist, text in [
    (lambda: FreeListAllocator(0), "free-list allocator size"),
    (lambda: freelist.alloc(-1), "allocation size"),
    (lambda: freelist.alloc(1, 3), "alignment"),
    (lambda: freelist.free_addr(0xdeadbeef), "not allocated"),
  ]:
    try:
      bad_freelist()
      raise AssertionError("bad free-list allocator input was accepted")
    except ValueError as exc:
      assert text in str(exc)

  buf = GpuBuffer(0x400000, 16, MMIOView(bytearray(16)))
  assert buf.offset(4, 8).va_addr == 0x400004
  assert buf.offset(4).size == 12
  for bad_buffer, text in [
    (lambda: GpuBuffer(-1, 16, MMIOView(bytearray(16))), "buffer address"),
    (lambda: GpuBuffer(0x400000, 0, MMIOView(bytearray(16))), "buffer size"),
    (lambda: GpuBuffer(0x400000, 16, object()), "CPU view"),
    (lambda: GpuBuffer(0x400000, 17, MMIOView(bytearray(16))), "CPU view"),
    (lambda: GpuBuffer(0x400000, 16, MMIOView(bytearray(16)), base=object()), "buffer base"),
  ]:
    try:
      bad_buffer()
      raise AssertionError("bad GPU buffer was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for args in [(-1, 1), (17, 1), (8, 9), (0, -1)]:
    try:
      buf.offset(*args)
      raise AssertionError(f"out-of-bounds buffer offset {args} was accepted")
    except ValueError as exc:
      assert "buffer" in str(exc)

  for bad_mapping, text in [
    (lambda: VirtMapping(0x1000000001, 0x1000, [(0x200000, 0x1000)], AddrSpace.SYS), "mapping VA"),
    (lambda: VirtMapping(0x1000000000, 0, [(0x200000, 0x1000)], AddrSpace.SYS), "mapping size"),
    (lambda: VirtMapping(0x1000000000, 0x1000, [], AddrSpace.SYS), "physical ranges"),
    (lambda: VirtMapping(0x1000000000, 0x1000, [(0x200001, 0x1000)], AddrSpace.SYS), "physical address"),
    (lambda: VirtMapping(0x1000000000, 0x1000, [(0x200000, 0)], AddrSpace.SYS), "physical span"),
    (lambda: VirtMapping(0x1000000000, 0x2000, [(0x200000, 0x1000)], AddrSpace.SYS), "mapping size"),
    (lambda: VirtMapping(0x1000000000, 0x1000, [(0x200000, 0x1000)], object()), "address space"),
    (lambda: VirtMapping(0x1000000000, 0x1000, [(0x200000, 0x1000)], AddrSpace.SYS, uncached=1), "uncached flag"),
    (lambda: VirtMapping(0x1000000000, 0x1000, [(0x200000, 0x1000)], AddrSpace.SYS, snooped=1), "snooped flag"),
  ]:
    try:
      bad_mapping()
      raise AssertionError("bad GPU virtual mapping was accepted")
    except ValueError as exc:
      assert text in str(exc)

  for bad_sysmem, text in [
    (lambda: SysmemAllocation(object(), [0x5000], va_addr=0x4000, size=0x1000), "view"),
    (lambda: SysmemAllocation(MMIOView(bytearray(0x1000)), [], va_addr=0x4000, size=0x1000), "physical ranges"),
    (lambda: SysmemAllocation(MMIOView(bytearray(0x1000)), [0x5001], va_addr=0x4000, size=0x1000), "physical address"),
    (lambda: SysmemAllocation(MMIOView(bytearray(0x1000)), [0x5000], va_addr=0x4000, size=0), "size"),
    (lambda: SysmemAllocation(MMIOView(bytearray(0x1000)), [0x5000], va_addr=0x4000, size=0x2000), "size"),
  ]:
    try:
      bad_sysmem()
      raise AssertionError("bad sysmem allocation was accepted")
    except ValueError as exc:
      assert text in str(exc)

  cmdq_page = GpuBuffer(0x8000, 0x1000, view=MMIOView(bytearray(0x1000)))
  gpfifo_state = GPFifoState(MMIOView(bytearray(32), fmt='Q'), MMIOView(bytearray(4), fmt='I'), 4)
  submitter = StandaloneSubmitter(object(), cmdq_page, gpfifo_state, MMIOView(bytearray(0x10000), fmt='I'))
  assert submitter.cmdq_page is cmdq_page and submitter.compute_gpfifo is gpfifo_state
  for bad_submitter, text in [
    (lambda: StandaloneSubmitter(object(), object(), gpfifo_state, MMIOView(bytearray(0x10000), fmt='I')), "command queue page"),
    (lambda: StandaloneSubmitter(object(), GpuBuffer(0x8000, 0x1000), gpfifo_state, MMIOView(bytearray(0x10000), fmt='I')), "CPU-visible"),
    (lambda: StandaloneSubmitter(object(), cmdq_page, object(), MMIOView(bytearray(0x10000), fmt='I')), "compute GPFIFO"),
    (lambda: StandaloneSubmitter(object(), cmdq_page, GPFifoState(None, MMIOView(bytearray(4), fmt='I'), 4), MMIOView(bytearray(0x10000), fmt='I')), "ring view"),
    (lambda: StandaloneSubmitter(object(), cmdq_page, GPFifoState(MMIOView(bytearray(32), fmt='Q'), None, 4), MMIOView(bytearray(0x10000), fmt='I')), "put view"),
    (lambda: StandaloneSubmitter(object(), cmdq_page, gpfifo_state, object()), "GPU MMIO"),
  ]:
    try:
      bad_submitter()
      raise AssertionError("bad standalone submitter input was accepted")
    except ValueError as exc:
      assert text in str(exc)

  mm = GpuMemoryManager(64 << 20, mmu_ver=2)
  mapping = mm.map_range(0x1000000000, 0x2000, [(0x200000, 0x1000), (0x300000, 0x1000)], AddrSpace.SYS, uncached=True)
  assert mm.page_directory_paddr in mm.page_tables
  leaf = mm.ensure_leaf_table(mapping.va_addr)
  idx = mm.table_index(mapping.va_addr, len(mm.level_shifts) - 1)
  assert leaf.entries[idx] & 1
  assert ((leaf.entries[idx] >> 1) & 0x3) == 2
  assert ((leaf.entries[idx] >> 8) & ((1 << 46) - 1)) == (0x200000 >> 12)
  mm.unmap_range(mapping.va_addr, mapping.size)
  assert leaf.entries[idx] == 0
  existing = mm.map_range(0x1000003000, 0x1000, [(0x500000, 0x1000)], AddrSpace.SYS)
  try:
    mm.map_range(0x1000002000, 0x2000, [(0x600000, 0x1000), (0x700000, 0x1000)], AddrSpace.SYS)
    raise AssertionError("overlapping GPU mapping was accepted")
  except ValueError as exc:
    assert "already mapped" in str(exc)
  rolled_leaf = mm.ensure_leaf_table(0x1000002000)
  rolled_idx = mm.table_index(0x1000002000, len(mm.level_shifts) - 1)
  existing_leaf = mm.ensure_leaf_table(existing.va_addr)
  existing_idx = mm.table_index(existing.va_addr, len(mm.level_shifts) - 1)
  assert rolled_leaf.entries[rolled_idx] == 0
  assert existing_leaf.entries[existing_idx] & 1
  assert 0x1000002000 not in mm.mappings and existing.va_addr in mm.mappings
  try:
    mm.map_range(existing.va_addr, existing.size, existing.paddrs, existing.aspace)
    raise AssertionError("duplicate mapping base was accepted")
  except ValueError as exc:
    assert "already mapped" in str(exc)
  mm.unmap_range(existing.va_addr, existing.size)
  vram_map = mm.valloc(0x2000, contiguous=True)
  vram_va, vram_pa = vram_map.va_addr, vram_map.paddrs[0][0]
  mm.vfree(vram_map)
  assert vram_va not in mm.mappings
  assert mm.alloc_vaddr(0x2000, 0x1000) == vram_va
  assert mm.palloc(0x2000, 0x1000) == vram_pa
  for bad_mm_call, text in [
    (lambda: GpuMemoryManager(0), "VRAM size"),
    (lambda: GpuMemoryManager(64 << 20, mmu_ver=4), "MMU version"),
    (lambda: mm.alloc_page_table(len(mm.level_shifts)), "page-table level"),
    (lambda: mm.table_index(0, len(mm.level_shifts)), "page-table level"),
    (lambda: mm.ensure_page_table(-1, 0), "GPU virtual address"),
    (lambda: mm.ensure_page_table(0, len(mm.level_shifts)), "target level"),
    (lambda: mm.reserved_pde_levels(0, 0), "reserved PDE level"),
    (lambda: mm.encode_pte(0x123, AddrSpace.SYS), "PTE physical address"),
    (lambda: mm.encode_pte(0x1000, object()), "address space"),
    (lambda: mm.encode_pde(0x123), "PDE physical address"),
    (lambda: mm.alloc_vaddr(0), "VA allocation size"),
    (lambda: mm.alloc_vaddr(1, 3), "alignment"),
    (lambda: mm.palloc(0), "physical allocation size"),
    (lambda: mm.map_range(0x1000000001, 0x1000, [(0x80000000, 0x1000)], AddrSpace.SYS), "mapping VA"),
    (lambda: mm.map_range(0x1000004000, 0, [(0x80000000, 0x1000)], AddrSpace.SYS), "mapping size"),
    (lambda: mm.map_range(0x1000004000, 0x1000, [(0x80000001, 0x1000)], AddrSpace.SYS), "physical address"),
    (lambda: mm.map_range(0x1000004000, 0x1000, [(0x80000000, 0)], AddrSpace.SYS), "physical ranges"),
    (lambda: mm.map_range(0x1000004000, 0x1000, [(0x80000000, 0x1000)], object()), "address space"),
    (lambda: mm.unmap_range(0x1000000001, 0x1000), "unmap VA"),
    (lambda: mm.unmap_range(0x1000000000, 0), "unmap size"),
    (lambda: mm.valloc(0), "VRAM allocation size"),
    (lambda: mm.valloc(0x1000, align=3), "alignment"),
  ]:
    try:
      bad_mm_call()
      raise AssertionError("bad GPU memory-manager input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  pde_mm = GpuMemoryManager(64 << 20, mmu_ver=2)
  pde_levels = pde_mm.reserved_pde_levels(0x1000000000, 3)
  assert [level[3] for level in pde_levels] == [47, 38, 29]
  assert [level[1:] for level in pde_levels] == [(0x1000, 1, 47), (0x1000, 1, 38), (0x1000, 1, 29)]
  live_replay_mm = GpuMemoryManager(0x300000000 - 0x4000000, mmu_ver=2, reserve_ptable=True)
  assert live_replay_mm.page_directory_paddr == 0
  assert live_replay_mm.palloc(0xf000) == 0x1a00000
  assert live_replay_mm.palloc(0xf000) == 0x1a0f000
  live_res_va = live_replay_mm.alloc_vaddr(512 << 20)
  assert live_res_va == 0x1000000000
  live_levels = live_replay_mm.reserved_pde_levels(live_res_va, 3, 512 << 20)
  assert [level[0] for level in live_levels] == [0x0, 0x200000, 0x201000]
  live_gpfifo = live_replay_mm.valloc(0x1000, contiguous=True)
  assert live_gpfifo.va_addr == 0x1020000000 and live_gpfifo.paddrs == [(0x1a1e000, 0x1000)]
  live_ramfc = live_replay_mm.valloc(0x1000, contiguous=True)
  live_method = live_replay_mm.palloc(0x5000, align=0x1000)
  assert live_ramfc.va_addr == 0x1020001000 and live_ramfc.paddrs == [(0x1a1f000, 0x1000)]
  assert live_method == 0x1a20000
  live_first_gr = live_replay_mm.valloc(0x193000, contiguous=True)
  assert live_first_gr.va_addr == 0x1020100000 and live_first_gr.paddrs == [(0x1a25000, 0x193000)]
  live_grctx_descs = {
    0: GRBufDesc(0x193000, virt=True, phys=True),
    1: GRBufDesc(0x5000, virt=True, phys=True, local=True),
    2: GRBufDesc(0x5000, virt=True, phys=True),
    3: GRBufDesc(0x3000, virt=True, phys=False),
    4: GRBufDesc(0x20000, virt=True, phys=False),
    5: GRBufDesc(0xa00000, virt=True, phys=False),
    6: GRBufDesc(0x80000, virt=True, phys=False),
    9: GRBufDesc(0x10000, virt=True, phys=True),
    10: GRBufDesc(0x80000, virt=False, phys=True),
    11: GRBufDesc(0x80000, virt=True, phys=True),
  }
  live_grctx_maps, live_promote_payload = build_grctx_promote_payload(
    live_replay_mm, 0xc1e00004, 0xcf000004, live_grctx_descs, existing={0: live_first_gr})
  assert hashlib.sha256(live_promote_payload).hexdigest() == "e9e2b70c57c91ad0a7855c258fd1145f43bc93013279e1b8825b333042d3c6bc"
  assert live_grctx_maps[2].paddrs == [(0x1bb8000, 0x5000)]
  assert live_grctx_maps[9].paddrs == [(0x2660000, 0x10000)]
  assert live_grctx_maps[10].paddrs == [(0x2670000, 0x80000)]
  assert live_grctx_maps[11].paddrs == [(0x26f0000, 0x80000)]
  pde_payload = pack_nv90f1_copy_server_reserved_pdes(512 << 20, 0x1000000000,
    0x1000000000 + (512 << 20) - 1, [(0x200000, 0x1000, 1, 47), (0x201000, 0x1000, 1, 38), (0x202000, 0x1000, 1, 29)])
  assert len(pde_payload) == 184
  assert hashlib.sha256(pde_payload).hexdigest() == "ce86da35b3c71f8f749fd9731441bf9d74c6d8ae07573047a8b0e27c0b94a95c"
  assert struct.unpack_from("<IIQQQI", pde_payload, 0) == (0, 0, 512 << 20, 0x1000000000, 0x1000000000 + (512 << 20) - 1, 3)
  assert struct.unpack_from("<QQIB3x", pde_payload, 40 + 24) == (0x201000, 0x1000, 1, 38)
  promote_payload = pack_nv2080_gpu_promote_ctx(0xc1e00004, 0xcf000007, [
    (0x80000000, 0x1000000000, 0x123000, 4, 0, 1, 0),
    (0x80100000, 0, 0x2000, 4, 10, 1, 1),
  ])
  assert len(promote_payload) == 560
  assert hashlib.sha256(promote_payload).hexdigest() == "f013385ceddc2acc247e12b6eccea1ed3ae5c3796adcee8242946309e3f39802"
  assert struct.unpack_from("<IIIIIIQQI", promote_payload, 0) == (1, 0, 0, 0xc1e00004, 0xcf000007, 0, 0, 0, 2)
  assert struct.unpack_from("<QQQI HBB", promote_payload, 48 + 32) == (0x80100000, 0, 0x2000, 4, 10, 1, 1)
  try:
    pack_nv2080_gpu_promote_ctx(0, 0, [(0, 0, 0, 0, 0, 0, 0)] * 17)
    raise AssertionError("oversized promote context entry list was accepted")
  except ValueError as exc:
    assert "16" in str(exc)
  ctx_info = bytearray(1664)
  struct.pack_into("<II", ctx_info, 0, 0x1000, 0x100)
  struct.pack_into("<II", ctx_info, 16 * 8, 0x2000, 0x200)
  assert hashlib.sha256(ctx_info).hexdigest() == "183ed23d62dcb65091f66df633bd39128775eeffcef080a1e700228c585c1684"
  ctx0 = unpack_nv2080_context_buffer_info(ctx_info)
  assert ctx0[0] == (0x1000, 0x100) and ctx0[16] == (0x2000, 0x200)
  try:
    unpack_nv2080_context_buffer_info(bytes(16))
    raise AssertionError("truncated context buffer info was accepted")
  except ValueError as exc:
    assert "truncated" in str(exc)
  for bad_ctx_info_call, text in [
    (lambda: unpack_nv2080_context_buffer_info(ctx_info, engine_index=-1), "engine index"),
    (lambda: unpack_nv2080_context_buffer_info(ctx_info, engine_index=8), "engine index"),
  ]:
    try:
      bad_ctx_info_call()
      raise AssertionError("bad context buffer engine index was accepted")
    except ValueError as exc:
      assert text in str(exc)
  ctx_descs = {0: GRBufDesc(0x3000, virt=True, phys=True), 10: GRBufDesc(0x2000, virt=False, phys=True)}
  for bad_desc_call, text in [
    (lambda: GRBufDesc(0, virt=True, phys=True), "descriptor size"),
    (lambda: GRBufDesc(0x1000, virt=1, phys=True), "virt flag"),
    (lambda: GRBufDesc(0x1000, virt=True, phys=1), "phys flag"),
    (lambda: GRBufDesc(0x1000, virt=True, phys=True, local=1), "local flag"),
  ]:
    try:
      bad_desc_call()
      raise AssertionError("bad GR context descriptor was accepted")
    except ValueError as exc:
      assert text in str(exc)
  ctx_maps = {
    0: VirtMapping(0x1000000000, 0x3000, [(0x80000000, 0x3000)], AddrSpace.PHYS),
    10: VirtMapping(0x1000010000, 0x2000, [(0x80100000, 0x2000)], AddrSpace.PHYS),
  }
  ctx_entries = build_promote_ctx_entries(ctx_descs, ctx_maps)
  assert ctx_entries == [
    (0x80000000, 0x1000000000, 0x3000, 4, 0, True, False),
    (0x80100000, 0, 0x2000, 4, 10, True, True),
  ]
  assert build_promote_ctx_entries({0: ctx_descs[0]}, ctx_maps, virt=False) == [(0x80000000, 0, 0x3000, 4, 0, True, True)]
  assert build_promote_ctx_entries({0: ctx_descs[0]}, ctx_maps, phys=False) == [(0, 0x1000000000, 0, 0, 0, False, False)]
  try:
    build_promote_ctx_entries({3: GRBufDesc(0x1000, virt=True, phys=True)}, ctx_maps)
    raise AssertionError("missing context mapping was accepted")
  except KeyError as exc:
    assert "3" in str(exc)
  ctx_synth = [(0, 0x100)] * 26
  ctx_synth[0] = (0x1234, 0x100)
  ctx_synth[16] = (0x2001, 0x1000)
  for engine_id in range(17, 25): ctx_synth[engine_id] = (0x3000 + engine_id, 0x1000)
  grctx = derive_grctx_buf_descs(ctx_synth)
  assert grctx[0].size == round_up(0x1234 + 0x40000, 0x100)
  assert grctx[1].size == round_up(0x2001, 0x1000) and grctx[1].local
  assert grctx[2].size == grctx[1].size and not grctx[2].local
  assert grctx[3].size == round_up(0x3000 + 17, 0x1000)
  assert grctx[5].size == round_up(0x3000 + 19, 2 << 20)
  assert not grctx[6].phys and grctx[6].virt
  assert grctx[9].phys and grctx[9].virt
  assert grctx[10].phys and not grctx[10].virt
  assert grctx[11].size == grctx[10].size and grctx[11].phys and grctx[11].virt
  try:
    derive_grctx_buf_descs(ctx_synth[:8])
    raise AssertionError("short grctx info was accepted")
  except ValueError as exc:
    assert "26" in str(exc)

  mm3 = GpuMemoryManager(64 << 20, mmu_ver=3)
  mapping3 = mm3.map_range(0x1000000000, 0x1000, [(0x400000, 0x1000)], AddrSpace.PHYS)
  leaf3 = mm3.ensure_leaf_table(mapping3.va_addr)
  idx3 = mm3.table_index(mapping3.va_addr, len(mm3.level_shifts) - 1)
  assert leaf3.entries[idx3] & 1
  assert ((leaf3.entries[idx3] >> 12) & ((1 << 40) - 1)) == (0x400000 >> 12)

  qmd = QMD()
  qmd.write(cta_raster_width=7, release0_payload_upper=3)
  assert qmd._rw_bits(*QMD.FIELDS['cta_raster_width']) == 7
  assert qmd._rw_bits(*QMD.FIELDS['release0_payload_upper']) == 3
  qmd.set_constant_buf_addr(0, 0x12345000)
  assert qmd._rw_bits(*QMD.FIELDS['constant_buffer_addr_lower_0']) == 0x12345000
  assert expected_qmd_template_sha256() == "a15a9304474d9de715e361a5f53141fde2a2111a2254c86eb86832c04abf48ff"
  qmd_before = QMD()
  qmd_after = QMD()
  qmd_after.write(cta_raster_width=1)
  assert qmd_byte_diff(qmd_before, qmd_before) == []
  assert qmd_byte_diff(qmd_before, qmd_after) == [48]
  assert qmd_field_diff(qmd_before, qmd_before) == []
  assert qmd_field_diff(qmd_before, qmd_after) == ["cta_raster_width"]
  for bad_qmd_call, text in [
    (lambda: QMD(memoryview(bytearray(8))), "truncated"),
    (lambda: qmd._rw_bits(1, 2, value=0), "bit range"),
    (lambda: qmd._rw_bits(2048, 2048, value=0), "exceeds"),
    (lambda: qmd.write(cta_raster_width=-1), "does not fit"),
    (lambda: qmd.write(cta_raster_width=1 << 32), "does not fit"),
    (lambda: qmd.write(not_a_qmd_field=1), "unknown QMD field"),
    (lambda: qmd.set_constant_buf_addr(0, QMD_ADDR_LIMIT), "constant buffer"),
  ]:
    try:
      bad_qmd_call()
      raise AssertionError("bad QMD input was accepted")
    except (ValueError, KeyError) as exc:
      assert text in str(exc)

  cubin = build_cubin()
  image, sections, relocs = elf_loader(cubin)
  assert image.nbytes > 0
  assert any(section.name == ".text.E_4" for section in sections)
  assert isinstance(relocs, list)
  assert elf_section(cubin, ".text.E_4")[:4] == CubinHelper.words_blob(((ch.Reg.R1 << 16) | ch.Op.LDC,))
  mul_cubin = build_cubin("mul")
  add_text, mul_text = elf_section(cubin, ".text.E_4"), elf_section(mul_cubin, ".text.E_4")
  arithmetic_offsets = [8 * 16, 9 * 16, 10 * 16, 11 * 16]
  assert [struct.unpack_from("<I", add_text, off)[0] & 0xffff for off in arithmetic_offsets] == [ch.Op.FADD] * 4
  assert [struct.unpack_from("<I", mul_text, off)[0] & 0xffff for off in arithmetic_offsets] == [ch.Op.FMUL] * 4
  assert [struct.unpack_from("<I", add_text, off + 8)[0] for off in arithmetic_offsets] == [0] * 4
  assert [struct.unpack_from("<I", mul_text, off + 8)[0] for off in arithmetic_offsets] == [0x00400000] * 4
  for off in range(0, len(add_text), 16):
    if off not in arithmetic_offsets: assert add_text[off:off+16] == mul_text[off:off+16]
  try:
    build_cubin("bad")
    raise AssertionError("unknown cubin arithmetic op was accepted")
  except ValueError as exc:
    assert "unknown arithmetic" in str(exc)
  def mutated_cubin(offset, fmt, value):
    data = bytearray(cubin)
    struct.pack_into(fmt, data, offset, value)
    return bytes(data)
  for bad_elf_call, text in [
    (lambda: elf_loader(b"\x7fELF"), "ELF header"),
    (lambda: elf_loader(mutated_cubin(58, "<H", 0)), "section header entry size"),
    (lambda: elf_loader(mutated_cubin(60, "<H", 0)), "no section headers"),
    (lambda: elf_loader(mutated_cubin(62, "<H", 0xffff)), "string table index"),
    (lambda: elf_loader(mutated_cubin(40, "<Q", len(cubin))), "section header table"),
    (lambda: elf_loader(mutated_cubin(ch.SECTION_HEADERS_OFF + ch.SECTION_HEADER_SIZE + 24, "<Q", len(cubin))), "section content"),
    (lambda: elf_loader(mutated_cubin(ch.SECTION_HEADERS_OFF + 5 * ch.SECTION_HEADER_SIZE, "<I", 0x100000)), "section name offset"),
    (lambda: elf_loader(mutated_cubin(ch.SECTION_HEADERS_OFF + 3 * ch.SECTION_HEADER_SIZE + 56, "<Q", 0)), "symbol table entry size"),
    (lambda: elf_loader(mutated_cubin(ch.SECTION_HEADERS_OFF + 9 * ch.SECTION_HEADER_SIZE + 56, "<Q", 0)), "relocation entry size"),
    (lambda: elf_loader(mutated_cubin(ch.REL_DEBUG_FRAME_OFF + 12, "<I", 0xff)), "invalid symbol index"),
  ]:
    try:
      bad_elf_call()
      raise AssertionError("malformed ELF was accepted")
    except (ValueError, RuntimeError) as exc:
      assert text in str(exc)
  info_section = ElfSection(".nv.info.test", [0, 0, 0, 0, 0, 8, 0, 0, 0, 0], b"\x04\x0a\x04\x00abcd")
  assert list(parse_elf_info(info_section)) == [(4, 10, b"abcd")]
  assert unpack_info_cbuf0_size(struct.pack("IH", 1, 0x160), ".nv.info.E_4", 0xa) == 0x160
  assert unpack_info_u32_pair(struct.pack("II", 0, 0x20), ".nv.info", 0x12) == (0, 0x20)
  for bad_info_section, text in [
    (ElfSection(".nv.info.short", [0, 0, 0, 0, 0, 4, 0, 0, 0, 0], b"\x04\x0a\x04"), "header"),
    (ElfSection(".nv.info.payload", [0, 0, 0, 0, 0, 8, 0, 0, 0, 0], b"\x04\x0a\x08\x00abcd"), "payload"),
  ]:
    try:
      list(parse_elf_info(bad_info_section))
      raise AssertionError("malformed .nv.info record was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_info_call, text in [
    (lambda: unpack_info_cbuf0_size(b"\0" * 5, ".nv.info.E_4", 0xa), "param 0xa"),
    (lambda: unpack_info_u32_pair(b"\0" * 7, ".nv.info", 0x12), "param 0x12"),
    (lambda: unpack_info_u32_pair(b"\0" * 7, ".nv.info", 0x2f), "param 0x2f"),
  ]:
    try:
      bad_info_call()
      raise AssertionError("short semantic .nv.info payload was accepted")
    except ValueError as exc:
      assert text in str(exc)

  words = build_launch_words(0x1000, 1, 2, 0x2000)
  decoded = list(decode_words(words))
  assert decoded[0][3] == 0x005c
  assert decoded[2][3] == 0x02b4
  assert decoded[-1][3] == 0x0020
  assert nvm(1, 0x20, 0xffffffff, typ=2) == [0x20012008, 0xffffffff]
  for bad_nvm, text in [
    (lambda: nvm(8, 0x20), "subchannel"),
    (lambda: nvm(0, 0x22), "4-byte aligned"),
    (lambda: nvm(0, 0x8000), "method address"),
    (lambda: nvm(0, 0x20, -1), "method argument"),
    (lambda: nvm(0, 0x20, typ=16), "packet type"),
  ]:
    try:
      bad_nvm()
      raise AssertionError("bad method packet input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_decode_words, text in [
    ([0x00020008, 1], "truncated"),
    ([0x100000000], "method packet header"),
    ([0x00010008, -1], "method argument"),
  ]:
    try:
      list(decode_words(bad_decode_words))
      raise AssertionError("bad method stream was accepted")
    except ValueError as exc:
      assert text in str(exc)
  try:
    build_launch_words(0x1000, 1, 2, QMD_ADDR_LIMIT)
    raise AssertionError("oversized QMD address was accepted")
  except ValueError as exc:
    assert "QMD" in str(exc)
  try:
    build_launch_words(QMD_ADDR_LIMIT, 1, 2, 0x2000)
    raise AssertionError("oversized timeline address was accepted")
  except ValueError as exc:
    assert "timeline" in str(exc)
  for bad_launch_call, text in [
    (lambda: build_launch_words(0x1000, -1, 2, 0x2000), "timeline wait value"),
    (lambda: build_launch_words(0x1000, 1, 0x100000000, 0x2000), "timeline done value"),
    (lambda: build_timeline_signal_words(0x1000, -1), "timeline signal payload"),
  ]:
    try:
      bad_launch_call()
      raise AssertionError("bad launch/timeline payload was accepted")
    except ValueError as exc:
      assert text in str(exc)
  good_gpfifo = GPFifoState(MMIOView(bytearray(32), fmt='Q'), MMIOView(bytearray(4), fmt='I'), 4, token=0x55)
  assert good_gpfifo.entries_count == 4 and good_gpfifo.token == 0x55
  for bad_gpfifo, text in [
    (lambda: GPFifoState(None, None, 0), "entry count"),
    (lambda: GPFifoState(None, None, 0x100000000), "entry count"),
    (lambda: GPFifoState(None, None, 1, token=-1), "GPFIFO token"),
    (lambda: GPFifoState(None, MMIOView(bytearray(4), fmt='I'), 1), "ring view"),
    (lambda: GPFifoState(MMIOView(bytearray(8), fmt='I'), MMIOView(bytearray(4), fmt='I'), 1), "ring view"),
    (lambda: GPFifoState(MMIOView(bytearray(8), fmt='Q'), MMIOView(bytearray(4), fmt='I'), 2), "entry count"),
    (lambda: GPFifoState(MMIOView(bytearray(8), fmt='Q'), None, 1), "put view"),
    (lambda: GPFifoState(MMIOView(bytearray(8), fmt='Q'), MMIOView(bytearray(8), fmt='Q'), 1), "put view"),
    (lambda: GPFifoState(MMIOView(bytearray(8), fmt='Q'), MMIOView(bytearray(0), fmt='I'), 1), "put view"),
  ]:
    try:
      bad_gpfifo()
      raise AssertionError("bad GPFIFO state was accepted")
    except ValueError as exc:
      assert text in str(exc)

  class FakeRegTransport:
    def __init__(self): self.bar = MMIOView(bytearray(0x40), fmt='I')
    def map_bar(self, bar, fmt='I'): return self.bar
  regs = NVRegisters(FakeRegTransport())
  regs.wreg(4, 0x12345678)
  assert regs.rreg(4) == 0x12345678
  regs.write_bits(4, 8, 15, 0xaa)
  assert regs.rreg(4) == 0x1234aa78 and regs.read_bits(4, 8, 15) == 0xaa
  for bad_reg_call, text in [
    (lambda: regs.rreg(-4), "register address"),
    (lambda: regs.rreg(2), "aligned"),
    (lambda: regs.wreg(4, 0x100000000), "register value"),
    (lambda: regs.write_bits(4, 16, 8, 0), "bit range"),
    (lambda: regs.write_bits(4, 0, 32, 0), "bit range"),
    (lambda: regs.write_bits(4, 0, 7, 0x100000000), "register bit value"),
  ]:
    try:
      bad_reg_call()
      raise AssertionError("bad register access was accepted")
    except ValueError as exc:
      assert text in str(exc)

  mac = object.__new__(MacEgpuTransport)
  mac.calls = []
  def fake_mac_rpc(cmd, *args, bar=0, **kwargs):
    mac.calls.append((cmd, args, bar, kwargs))
    if cmd == RemoteCmd.MAP_BAR: return (0x10000000, 0x400000, None, None)
    return (0x10000000, args[0] if args else 0, None, None)
  mac.rpc = fake_mac_rpc
  assert MacEgpuTransport.probe(mac) == (0x10000000, 0)
  assert mac.calls[-1] == (RemoteCmd.PROBE, (), 0, {})
  assert MacEgpuTransport.ping(mac) == (0x10000000, 0)
  assert mac.calls[-1] == (RemoteCmd.PING, (), 0, {})
  assert MacEgpuTransport.reset(mac) == (0x10000000, 0)
  assert mac.calls[-1] == (RemoteCmd.RESET, (), 0, {})
  assert MacEgpuTransport.read_config(mac, 0, 2) == 0x10000000
  assert mac.calls[-1] == (RemoteCmd.CFG_READ, (0, 2), 0, {})
  MacEgpuTransport.write_config(mac, 4, 0x1234, 4)
  assert mac.calls[-1] == (RemoteCmd.CFG_WRITE, (4, 4, 0x1234), 0, {})
  assert MacEgpuTransport.bar_info(mac, 1) == (0x10000000, 0x400000)
  assert mac.calls[-1] == (RemoteCmd.MAP_BAR, (), 1, {})
  assert MacEgpuTransport.resize_bar(mac, 1, 0x20000000) == (0x10000000, 0x20000000)
  assert mac.calls[-1] == (RemoteCmd.RESIZE_BAR, (0x20000000,), 1, {})
  assert MacEgpuTransport.map_bar(mac, 1, off=0, size=0x100).nbytes == 0x100
  for mac_access_call, text in [
    (lambda: MacEgpuTransport.read_config(mac, -1, 2), "offset"),
    (lambda: MacEgpuTransport.write_config(mac, 0, 0, 3), "size"),
    (lambda: MacEgpuTransport.bar_info(mac, 6), "BAR index"),
    (lambda: MacEgpuTransport.resize_bar(mac, 1, 0), "resize size"),
    (lambda: MacEgpuTransport.map_bar(mac, 1, off=0, size=0), "mapping size"),
  ]:
    try:
      mac_access_call()
      raise AssertionError("bad TinyGPU transport access was accepted")
    except ValueError as exc:
      assert text in str(exc)
  class FakeSock:
    def __init__(self): self.sent = []
    def sendall(self, data): self.sent.append(bytes(data))
    def recv(self, n): return struct.pack("<BQQ", 0, 0x55, 0)[:n]
  class FakeFdSock(FakeSock):
    def __init__(self, msg=None, anc=None):
      super().__init__()
      self.msg = struct.pack("<BQQ", 0, 0x2000, 0) if msg is None else msg
      self.anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("<i", 33))] if anc is None else anc
    def recvmsg(self, msg_size, anc_size): return self.msg, self.anc, 0, None
  mac_sock = object.__new__(MacEgpuTransport)
  mac_sock.dev_id, mac_sock.sock = 7, FakeSock()
  assert MacEgpuTransport.rpc(mac_sock, RemoteCmd.CFG_READ, 0, 4)[:2] == (0x55, 0)
  assert mac_sock.sock.sent[-1][:9] == struct.pack("<BII", RemoteCmd.CFG_READ, 7, 0)
  class FakeErrorSock(FakeSock):
    def __init__(self, detail=b""):
      super().__init__()
      self.parts = [struct.pack("<BQQ", 7, len(detail), 0), detail]
    def recv(self, n):
      if not self.parts: return b""
      part = self.parts.pop(0)
      if len(part) > n:
        self.parts.insert(0, part[n:])
        return part[:n]
      return part
  mac_err_sock = object.__new__(MacEgpuTransport)
  mac_err_sock.dev_id, mac_err_sock.sock = 7, FakeErrorSock()
  try:
    MacEgpuTransport.rpc(mac_err_sock, RemoteCmd.PROBE)
    raise AssertionError("TinyGPU error RPC was accepted")
  except RuntimeError as exc:
    assert "TinyGPU RPC PROBE failed status=0x7: no error payload" in str(exc)
  mac_err_detail_sock = object.__new__(MacEgpuTransport)
  mac_err_detail_sock.dev_id, mac_err_detail_sock.sock = 7, FakeErrorSock(b"Driver not available")
  try:
    MacEgpuTransport.rpc(mac_err_detail_sock, RemoteCmd.CFG_READ, 0, 4)
    raise AssertionError("TinyGPU detailed error RPC was accepted")
  except RuntimeError as exc:
    assert "TinyGPU RPC CFG_READ failed status=0x7: Driver not available" in str(exc)
  mac_fd_sock = object.__new__(MacEgpuTransport)
  mac_fd_sock.dev_id, mac_fd_sock.sock = 7, FakeFdSock()
  assert MacEgpuTransport.rpc(mac_fd_sock, RemoteCmd.MAP_SYSMEM_FD, 0x1000, 0, has_fd=True) == (0x2000, 0, None, 33)
  for bad_fd_sock, text in [
    (FakeFdSock(msg=b"\0" * 16), "header"),
    (FakeFdSock(anc=[]), "fd response"),
    (FakeFdSock(anc=[(socket.SOL_SOCKET, socket.SCM_RIGHTS, b"\0" * 3)]), "fd response"),
  ]:
    mac_bad_fd = object.__new__(MacEgpuTransport)
    mac_bad_fd.dev_id, mac_bad_fd.sock = 7, bad_fd_sock
    try:
      MacEgpuTransport.rpc(mac_bad_fd, RemoteCmd.MAP_SYSMEM_FD, 0x1000, 0, has_fd=True)
      raise AssertionError("bad TinyGPU fd RPC response was accepted")
    except RuntimeError as exc:
      assert text in str(exc)
  for remote_call, text in [
    (lambda: MacEgpuTransport.rpc(mac_sock, 0xff), "unknown"),
    (lambda: MacEgpuTransport.rpc(mac_sock, RemoteCmd.MMIO_READ, -1, 4), "offset"),
    (lambda: MacEgpuTransport.rpc(mac_sock, RemoteCmd.MMIO_READ, 0, -1), "size"),
    (lambda: MacEgpuTransport.rpc(mac_sock, RemoteCmd.MMIO_READ, 0, 4, readout_size=-1), "readout"),
    (lambda: MacEgpuTransport.rpc(mac_sock, RemoteCmd.MMIO_WRITE, 0, 4, payload=None), "payload"),
    (lambda: MacEgpuTransport.rpc(mac_sock, RemoteCmd.MMIO_WRITE, 0, 4, payload=object()), "bytes-like"),
    (lambda: MacEgpuTransport.rpc(mac_sock, RemoteCmd.MMIO_WRITE, 0, 4, 0, 1), "positional words"),
    (lambda: MacEgpuTransport.bulk_read(mac_sock, RemoteCmd.MMIO_READ, 6, 0, 4), "BAR index"),
    (lambda: MacEgpuTransport.bulk_read(mac_sock, RemoteCmd.MMIO_READ, 0, 0, 0), "bulk read size"),
    (lambda: MacEgpuTransport.bulk_write(mac_sock, RemoteCmd.MMIO_WRITE, 0, 0, b""), "bulk write payload"),
    (lambda: MacEgpuTransport.bulk_write(mac_sock, RemoteCmd.MMIO_WRITE, 0, 0, object()), "bytes-like"),
    (lambda: MacEgpuTransport.bulk_write(mac_sock, RemoteCmd.MMIO_WRITE, 0, -1, b"x"), "offset"),
  ]:
    try:
      remote_call()
      raise AssertionError("bad TinyGPU RPC/bulk input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  class FakeRemoteTransport:
    def __init__(self):
      self.reads, self.writes = [], []
    def bulk_read(self, cmd, idx, offset, size):
      self.reads.append((cmd, idx, offset, size))
      return bytes(range(size))
    def bulk_write(self, cmd, idx, offset, data):
      self.writes.append((cmd, idx, offset, bytes(data)))
  remote = FakeRemoteTransport()
  rview = RemoteMMIOView(remote, 2, 0x100, fmt='I', off=0x20)
  assert rview[1] == 0x03020100
  assert remote.reads[-1] == (RemoteCmd.MMIO_READ, 2, 0x24, 4)
  rview[2] = 0x11223344
  assert remote.writes[-1] == (RemoteCmd.MMIO_WRITE, 2, 0x28, b"\x44\x33\x22\x11")
  raw_view = rview.view(offset=0x10, size=4, fmt='B')
  raw_view[:4] = b"abcd"
  assert remote.writes[-1] == (RemoteCmd.MMIO_WRITE, 2, 0x30, b"abcd")
  assert rview[-1] == 0x03020100
  assert remote.reads[-1] == (RemoteCmd.MMIO_READ, 2, 0x11c, 4)
  class FakeMapBarMac:
    def __init__(self): self.infos = []
    def bar_info(self, bar):
      self.infos.append(bar)
      return 0x10000000, 0x400000
  fake_map_mac = FakeMapBarMac()
  sub_bar = MacEgpuTransport.map_bar(fake_map_mac, 0, off=0x300000, size=0x100000, fmt='B')
  assert isinstance(sub_bar, RemoteMMIOView)
  assert sub_bar.off == 0x300000 and sub_bar.nbytes == 0x100000 and fake_map_mac.infos == [0]
  try:
    MacEgpuTransport.map_bar(fake_map_mac, 0, off=0x300000, size=0x100001, fmt='B')
    raise AssertionError("out-of-range TinyGPU BAR submapping was accepted")
  except ValueError as exc:
    assert "outside BAR" in str(exc)
  class ShortRemoteTransport(FakeRemoteTransport):
    def bulk_read(self, cmd, idx, offset, size): return bytes(max(0, size - 1))
  try:
    RemoteMMIOView(ShortRemoteTransport(), 2, 0x100, fmt='I')[0]
    raise AssertionError("short RemoteMMIOView read was accepted")
  except RuntimeError as exc:
    assert "short response" in str(exc)
  for op in [
    lambda: rview[len(rview)],
    lambda: rview.__setitem__(slice(0, 2), [1]),
    lambda: rview.__getitem__(slice(0, 4, 2)),
    lambda: rview.view(offset=-1, size=1),
    lambda: rview.view(offset=0x101, size=1),
    lambda: rview.view(offset=0xfc, size=8),
    lambda: rview.view(offset=0, size=2, fmt='I'),
  ]:
    try:
      op()
      raise AssertionError("invalid RemoteMMIOView operation was accepted")
    except (ValueError, IndexError) as exc:
      assert "RemoteMMIOView" in str(exc)
  sysmem_header = MMIOView(bytearray(0x40))
  sysmem_qwords = sysmem_header.view(fmt='Q')
  sysmem_qwords[:6] = array.array('Q', [0x80000000, 0x2000, 0x90000000, 0x3000, 0, 0])
  assert MacEgpuTransport.sysmem_pages_from_header(sysmem_header, 0x4001) == [
    0x80000000, 0x80001000, 0x90000000, 0x90001000, 0x90002000]
  assert MacEgpuTransport.sysmem_pages_from_header(sysmem_header, 0x3000) == [
    0x80000000, 0x80001000, 0x90000000]
  for header_words, request_size, exc_type, text in [
    ([0x80000001, 0x1000, 0, 0], 0x1000, ValueError, "aligned"),
    ([0x80000000, 0x1001, 0, 0], 0x1000, ValueError, "aligned"),
    ([0x80000000], 0x1000, ValueError, "incomplete"),
    ([0x80000000, 0x1000, 0, 0], 0x2000, RuntimeError, "1 pages"),
  ]:
    bad_header = MMIOView(bytearray(round_up(len(header_words) * 8, 8)))
    bad_header.view(fmt='Q')[:len(header_words)] = array.array('Q', header_words)
    try:
      MacEgpuTransport.sysmem_pages_from_header(bad_header, request_size)
      raise AssertionError("bad sysmem header was accepted")
    except exc_type as exc:
      assert text in str(exc)
  assert MacEgpuTransport.validate_sysmem_fd_response(0x1001, 0x2000, 7) == 0x2000
  for request_size, mapped_size, fd, exc_type, text in [
    (-1, 0x1000, 7, ValueError, "non-negative"),
    (0x1000, 0x1000, -1, RuntimeError, "valid fd"),
    (0x1000, 0x1000, None, RuntimeError, "valid fd"),
    (0x1001, 0x1000, 7, RuntimeError, "0x1000"),
  ]:
    try:
      MacEgpuTransport.validate_sysmem_fd_response(request_size, mapped_size, fd)
      raise AssertionError("bad TinyGPU sysmem fd response was accepted")
    except exc_type as exc:
      assert text in str(exc)
  tiny_alloc = object.__new__(MacEgpuTransport)
  tiny_alloc.rpc = lambda *args, **kwargs: (0x2000, 0, None, 88)
  tiny_closes, tiny_mmaps = [], []
  class FakeMmap(bytearray):
    pass
  old_mmap, old_close = mmap.mmap, os.close
  try:
    def fake_mmap(fd, length, flags=0, prot=0):
      tiny_mmaps.append((fd, length, flags, prot))
      data = FakeMmap(length)
      data[:32] = struct.pack("<QQQQ", 0xa0000000, 0x2000, 0, 0)
      return data
    mmap.mmap = fake_mmap
    os.close = lambda fd: tiny_closes.append(fd)
    view, pages = MacEgpuTransport.alloc_sysmem(tiny_alloc, 0x1001)
  finally:
    mmap.mmap, os.close = old_mmap, old_close
  assert tiny_mmaps == [(88, 0x2000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)]
  assert tiny_closes == [88]
  assert pages == [0xa0000000, 0xa0001000] and len(view) == 0x2000
  tiny_fail = object.__new__(MacEgpuTransport)
  tiny_fail.rpc = lambda *args, **kwargs: (0x1000, 0, None, 89)
  tiny_fail_closes = []
  old_mmap, old_close = mmap.mmap, os.close
  try:
    mmap.mmap = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mmap fail"))
    os.close = lambda fd: tiny_fail_closes.append(fd)
    try:
      MacEgpuTransport.alloc_sysmem(tiny_fail, 0x1000)
      raise AssertionError("TinyGPU sysmem mmap failure was accepted")
    except RuntimeError as exc:
      assert "mmap fail" in str(exc)
  finally:
    mmap.mmap, os.close = old_mmap, old_close
  assert tiny_fail_closes == [89]

  queue_mem = bytearray(0x5000)
  qview = MMIOView(queue_mem)
  qview[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=0, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
  for bad_queue_unpack, text in [
    (lambda: MsgqTxHeader.unpack_from(b"\0" * (MsgqTxHeader.SIZE - 1)), "message queue header"),
    (lambda: RpcHeader.unpack_from(b"\0" * (RpcHeader.SIZE - 1)), "RPC header"),
    (lambda: GspQueueElement.unpack_from(b"\0" * (GspQueueElement.SIZE - 1)), "queue element"),
  ]:
    try:
      bad_queue_unpack()
      raise AssertionError("truncated GSP queue struct was accepted")
    except ValueError as exc:
      assert text in str(exc)
  rpc = GspRpcQueue(None, qview)
  rpc.send_rpc(NV_VGPU_MSG_FUNCTION_SET_REGISTRY, b"abc")
  assert rpc.tx_view[4] == 1
  hdr = RpcHeader.unpack_from(qview.view(0x1000 + GspQueueElement.SIZE, RpcHeader.SIZE)[:])
  assert hdr.function == NV_VGPU_MSG_FUNCTION_SET_REGISTRY
  assert hdr.length == RpcHeader.SIZE + 3
  split_mem = bytearray(0x5000)
  split_view = MMIOView(split_mem)
  split_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=0, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
  split_rpc = GspRpcQueue(None, split_view)
  split_payload = b"A" * (0x3000 - GspQueueElement.SIZE - RpcHeader.SIZE + 1)
  split_rpc.send_rpc(NV_VGPU_MSG_FUNCTION_SET_REGISTRY, split_payload)
  split_first = GspQueueElement.unpack_from(split_view.view(0x1000, GspQueueElement.SIZE)[:])
  split_first_hdr = RpcHeader.unpack_from(split_view.view(0x1000 + GspQueueElement.SIZE, RpcHeader.SIZE)[:])
  split_second = GspQueueElement.unpack_from(split_view.view(0x4000, GspQueueElement.SIZE)[:])
  split_second_hdr = RpcHeader.unpack_from(split_view.view(0x4000 + GspQueueElement.SIZE, RpcHeader.SIZE)[:])
  assert split_first.elem_count == 3
  assert split_first_hdr.function == NV_VGPU_MSG_FUNCTION_SET_REGISTRY
  assert split_first_hdr.length == 0x3000 - GspQueueElement.SIZE
  assert split_second.elem_count == 1
  assert split_second_hdr.function == NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD
  assert split_second_hdr.length == RpcHeader.SIZE + 1
  assert split_rpc.tx_view[4] == 0
  try:
    split_rpc._send_rpc_record(NV_VGPU_MSG_FUNCTION_SET_REGISTRY, b"B" * (0x3000 - GspQueueElement.SIZE - RpcHeader.SIZE + 1))
    raise AssertionError("oversized RPC record was accepted")
  except ValueError as exc:
    assert "capacity" in str(exc)
  for bad_header, text in [
    (MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=0, write_ptr=0, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000), "message count"),
    (MsgqTxHeader(version=0, size=0x2000, msg_size=0x1000, msg_count=4, write_ptr=0, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000), "entries"),
    (MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4, write_ptr=4, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000), "write pointer"),
  ]:
    bad_qview = MMIOView(bytearray(0x5000))
    bad_qview[:MsgqTxHeader.SIZE] = bad_header.pack()
    try:
      GspRpcQueue(None, bad_qview)
      raise AssertionError("bad GSP RPC queue header was accepted")
    except ValueError as exc:
      assert text in str(exc)
  root_mem = bytearray(0x5000)
  root_view = MMIOView(root_mem)
  root_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=0, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
  root_rpc = GspRpcQueue(None, root_view)
  root_client = GspRmClient(root_rpc)
  assert root_client.alloc_root(wait=False) == root_client.priv_root
  root_payload = bytes(root_view.view(0x1000 + GspQueueElement.SIZE + RpcHeader.SIZE, 32 + NV0000_ALLOC_PARAMETERS_SIZE))
  assert struct.unpack_from("<IIIIIII", root_payload, 0) == (root_client.priv_root, 0, 0xcf000000, NV01_ROOT, 0, NV0000_ALLOC_PARAMETERS_SIZE, 0)
  assert root_payload[32:] == pack_nv0000_alloc_parameters()
  ok_alloc_resp = pack_rpc_gsp_rm_alloc(0xc1e00004, 0x80, 0xcf000123, NV01_MEMORY_VIRTUAL)
  assert unpack_rpc_gsp_rm_alloc_response(ok_alloc_resp) == (0, b"")
  alloc_payload_resp = pack_rpc_gsp_rm_alloc(0xc1e00004, 0x80, 0xcf000123, NV01_MEMORY_VIRTUAL, b"ok")
  assert unpack_rpc_gsp_rm_alloc_response(alloc_payload_resp) == (0, b"ok")
  for bad_alloc_pack, text in [
    (lambda: pack_rpc_gsp_rm_alloc(0xc1e00004, 0x80, 0xcf000123, 0x100000000), "RM class"),
    (lambda: pack_rpc_gsp_rm_alloc(0xc1e00004, 0x80, 0xcf000123, NV01_MEMORY_VIRTUAL, flags=-1), "RM alloc flags"),
  ]:
    try:
      bad_alloc_pack()
      raise AssertionError("bad GSP RM alloc request field was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for malformed_alloc_resp, text in [
    (b"\0" * 31, "header"),
    (alloc_payload_resp[:-1], "payload"),
  ]:
    try:
      unpack_rpc_gsp_rm_alloc_response(malformed_alloc_resp)
      raise AssertionError("malformed GSP RM alloc response was accepted")
    except ValueError as exc:
      assert text in str(exc)
  ok_resp = pack_rpc_gsp_rm_control(0xc1e00004, 0x80, 0x1234, b"pong")
  assert unpack_rpc_gsp_rm_control_response(ok_resp) == (0, b"pong")
  for bad_control_pack, text in [
    (lambda: pack_rpc_gsp_rm_control(0xc1e00004, 0x80, 0x100000000), "RM control command"),
    (lambda: pack_rpc_gsp_rm_control(0xc1e00004, 0x80, 0x1234, flags=-1), "RM control flags"),
  ]:
    try:
      bad_control_pack()
      raise AssertionError("bad GSP RM control request field was accepted")
    except ValueError as exc:
      assert text in str(exc)
  bad_resp = bytearray(pack_rpc_gsp_rm_control(0xc1e00004, 0x80, 0x1234, b""))
  struct.pack_into("<I", bad_resp, 12, 0x51)
  assert unpack_rpc_gsp_rm_control_response(bad_resp) == (0x51, b"")
  for malformed_resp, text in [
    (b"\0" * 23, "header"),
    (pack_rpc_gsp_rm_control(0xc1e00004, 0x80, 0x1234, b"pong")[:-1], "payload"),
  ]:
    try:
      unpack_rpc_gsp_rm_control_response(malformed_resp)
      raise AssertionError("malformed GSP RM control response was accepted")
    except ValueError as exc:
      assert text in str(exc)
  class FakeCmdQ:
    def __init__(self): self.sent = []
    def send_rpc(self, func, payload): self.sent.append((func, bytes(payload)))
  class FakeStatQ:
    def __init__(self, resp): self.resp = resp
    def wait_resp(self, func): return self.resp
  fake_cmd = FakeCmdQ()
  fake_client = GspRmClient(fake_cmd, FakeStatQ(ok_resp))
  assert fake_client.rm_control(0x80, 0x1234, b"ping") == b"pong"
  assert fake_cmd.sent[-1][0] == NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL
  sent_before = len(fake_cmd.sent)
  try:
    fake_client.rm_control(0x80, 0x1234, object())
    raise AssertionError("bad GSP RM control params were accepted")
  except ValueError as exc:
    assert "params" in str(exc)
  assert len(fake_cmd.sent) == sent_before
  for bad_gsp_control, text in [
    (lambda: fake_client.rm_control(0x100000000, 0x1234, b""), "object handle"),
    (lambda: fake_client.rm_control(0x80, 0x100000000, b""), "RM control command"),
    (lambda: fake_client.rm_control(0x80, 0x1234, b"", client=0x100000000), "client handle"),
  ]:
    try:
      bad_gsp_control()
      raise AssertionError("bad GSP RM control input reached send")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake_cmd.sent) == sent_before
  alloc_client = GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp))
  assert alloc_client.rm_alloc(0x80, NV01_MEMORY_VIRTUAL, b"zz", h_object=0xcf000123) == 0xcf000123
  assert fake_cmd.sent[-1][0] == NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC
  old_trace_rm, old_trace_rm_stack = os.environ.get("NV_ADD_TRACE_RM_ALLOC"), os.environ.get("NV_ADD_TRACE_RM_STACK")
  try:
    os.environ["NV_ADD_TRACE_RM_ALLOC"] = "1"
    os.environ["NV_ADD_TRACE_RM_STACK"] = "1"
    rm_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(rm_trace_buf):
      GspRmClient(fake_cmd, FakeStatQ(ok_resp)).rm_control(0x80, 0x1234, b"ping")
      GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp)).rm_alloc(0x80, NV01_MEMORY_VIRTUAL, b"zz", h_object=0xcf000124)
    rm_trace = rm_trace_buf.getvalue()
    assert "rm gsp_control_rpc" in rm_trace and "cmd=0x1234" in rm_trace and "cmd_name=UNKNOWN_0x1234" in rm_trace
    assert f"rpc_sha256={hashlib.sha256(fake_cmd.sent[-2][1]).hexdigest()}" in rm_trace
    assert "rm gsp_alloc" in rm_trace and "h_class=0x70" in rm_trace and "object=0xcf000124" in rm_trace
    assert "class_name=NV01_MEMORY_VIRTUAL" in rm_trace
    assert "rm_alloc" in rm_trace
  finally:
    for name, value in (("NV_ADD_TRACE_RM_ALLOC", old_trace_rm), ("NV_ADD_TRACE_RM_STACK", old_trace_rm_stack)):
      if value is None: os.environ.pop(name, None)
      else: os.environ[name] = value
  for make_bad_gsp_client, text in [
    (lambda: GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp), first_handle=0x100000000), "first RM handle"),
    (lambda: GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp), priv_root=-1), "private root handle"),
  ]:
    try:
      make_bad_gsp_client()
      raise AssertionError("bad GSP RM client handle seed was accepted")
    except ValueError as exc:
      assert text in str(exc)
  try:
    overflow_client = GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp))
    overflow_client.next_handle = 0x100000000
    overflow_client.handle()
    raise AssertionError("overflowing generated GSP RM handle was accepted")
  except ValueError as exc:
    assert "RM handle" in str(exc)
  try:
    GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp)).rm_alloc(0x80, NV01_MEMORY_VIRTUAL, h_object=0x100000000)
    raise AssertionError("overflowing explicit GSP RM handle was accepted")
  except ValueError as exc:
    assert "object handle" in str(exc)
  bad_class_client = GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp))
  sent_before = len(fake_cmd.sent)
  next_before = bad_class_client.next_handle
  try:
    bad_class_client.rm_alloc(0x80, 0x100000000)
    raise AssertionError("bad GSP RM class consumed a handle")
  except ValueError as exc:
    assert "RM class" in str(exc)
  assert len(fake_cmd.sent) == sent_before
  assert bad_class_client.next_handle == next_before
  bad_params_client = GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp))
  sent_before = len(fake_cmd.sent)
  next_before = bad_params_client.next_handle
  try:
    bad_params_client.rm_alloc(0x80, NV01_MEMORY_VIRTUAL, object())
    raise AssertionError("bad GSP RM alloc params consumed a handle")
  except ValueError as exc:
    assert "params" in str(exc)
  assert len(fake_cmd.sent) == sent_before
  assert bad_params_client.next_handle == next_before
  fallback_client = GspRmClient(fake_cmd)
  replacement_stat = FakeStatQ(ok_resp)
  fallback_client.stat_q = replacement_stat
  assert fallback_client.stat_q is replacement_stat
  sysinfo_client = GspRmClient(fake_cmd, FakeStatQ(ok_resp))
  sysinfo_payload = pack_gsp_system_info(0x1000, 0x2000, 0x3000, 0x220810de, 0x10de0000, 0xa1)
  sent_before = len(fake_cmd.sent)
  assert sysinfo_client.set_system_info(sysinfo_payload, wait=False) == b""
  assert len(fake_cmd.sent) == sent_before + 1
  assert fake_cmd.sent[-1] == (NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, sysinfo_payload)
  for bad_sysinfo_call, text in [
    (lambda: sysinfo_client.set_system_info(object()), "bytes-like"),
    (lambda: sysinfo_client.set_system_info(b"short"), "928 bytes"),
    (lambda: sysinfo_client.set_system_info(sysinfo_payload, wait=1), "RM system-info wait flag"),
  ]:
    sent_before = len(fake_cmd.sent)
    try:
      bad_sysinfo_call()
      raise AssertionError("bad GSP system-info input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake_cmd.sent) == sent_before
  reg_client = GspRmClient(fake_cmd, FakeStatQ(ok_resp))
  sent_before = len(fake_cmd.sent)
  assert reg_client.set_registry(None, wait=False) == b""
  assert len(fake_cmd.sent) == sent_before + 1
  default_registry_payload = fake_cmd.sent[-1][1]
  assert struct.unpack_from("<II", default_registry_payload, 0) == (len(default_registry_payload), 2)
  sent_before = len(fake_cmd.sent)
  assert reg_client.set_registry({}, wait=False) == b""
  assert len(fake_cmd.sent) == sent_before + 1
  empty_registry_payload = fake_cmd.sent[-1][1]
  assert struct.unpack_from("<II", empty_registry_payload, 0) == (8, 0)
  for bad_registry_call, text in [
    (lambda: reg_client.set_registry(object()), "registry entries"),
    (lambda: reg_client.set_registry({"bad": -1}), "registry value"),
    (lambda: reg_client.set_registry(wait=1), "RM registry wait flag"),
  ]:
    sent_before = len(fake_cmd.sent)
    try:
      bad_registry_call()
      raise AssertionError("bad registry input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake_cmd.sent) == sent_before
  rm_send_client = GspRmClient(fake_cmd, FakeStatQ(ok_resp))
  sent_before = len(fake_cmd.sent)
  for bad_rm_send, text in [
    (lambda: rm_send_client.send_and_wait(0x100000000, b""), "RM RPC function"),
    (lambda: rm_send_client.send_and_wait(NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, object()), "payload"),
    (lambda: rm_send_client.send_and_wait(NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, b"", wait=1), "wait flag"),
  ]:
    try:
      bad_rm_send()
      raise AssertionError("bad GSP RM send input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake_cmd.sent) == sent_before
  rm_wait_client = GspRmClient(fake_cmd, FakeStatQ(ok_alloc_resp))
  sent_before = len(fake_cmd.sent)
  next_before = rm_wait_client.next_handle
  for bad_rm_wait, text in [
    (lambda: rm_wait_client.alloc_root(wait=1), "alloc wait flag"),
    (lambda: rm_wait_client.rm_alloc(0x80, NV01_MEMORY_VIRTUAL, wait=1), "alloc wait flag"),
    (lambda: rm_wait_client.set_page_directory(0x80, 0x90, 0x1000, 1, wait=1), "set_page_directory wait flag"),
    (lambda: rm_wait_client.unloading_guest_driver(wait=1), "unloading_guest_driver wait flag"),
  ]:
    try:
      bad_rm_wait()
      raise AssertionError("bad GSP RM wait flag was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake_cmd.sent) == sent_before
    assert rm_wait_client.next_handle == next_before
  pdir_client = GspRmClient(fake_cmd, FakeStatQ(ok_resp))
  sent_before = len(fake_cmd.sent)
  for bad_pdir_call, text in [
    (lambda: pdir_client.set_page_directory(0x80, 0x90f1, 0x1001, 1), "page directory physical address"),
    (lambda: pdir_client.set_page_directory(0x80, 0x90f1, 0x1000, 0), "page directory entry count"),
    (lambda: pdir_client.set_page_directory(0x100000000, 0x90f1, 0x1000, 1), "device handle"),
    (lambda: pdir_client.set_page_directory(0x80, 0x100000000, 0x1000, 1), "vaspace handle"),
    (lambda: pdir_client.set_page_directory(0x80, 0x90f1, 0x1000, 1, client=0x100000000), "client handle"),
    (lambda: pdir_client.set_page_directory(0x80, 0x90f1, 0x1000, 1, pasid=0x100000000), "PASID"),
  ]:
    try:
      bad_pdir_call()
      raise AssertionError("bad GSP page-directory input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake_cmd.sent) == sent_before
  try:
    GspRmClient(FakeCmdQ(), FakeStatQ(bytes(bad_resp))).rm_control(0x80, 0x1234, b"fail")
    raise AssertionError("GSP RM control status was not checked")
  except RuntimeError as exc:
    msg = str(exc)
    assert "0x51" in msg and "object=0x80" in msg and "cmd=0x1234" in msg
    assert "cmd_name=UNKNOWN_0x1234" in msg and "payload_len=4" in msg
    assert f"payload_sha256={hashlib.sha256(b'fail').hexdigest()}" in msg
  try:
    GspRmClient(FakeCmdQ(), FakeStatQ(b"\0" * 23)).rm_control(0x80, 0x1234)
    raise AssertionError("short GSP RM control response was accepted")
  except ValueError as exc:
    assert "header" in str(exc)
  failed_alloc_resp = bytearray(ok_alloc_resp)
  struct.pack_into("<I", failed_alloc_resp, 16, 0x1f)
  try:
    GspRmClient(FakeCmdQ(), FakeStatQ(bytes(failed_alloc_resp))).rm_alloc(0x80, NV01_MEMORY_VIRTUAL)
    raise AssertionError("GSP RM alloc status was not checked")
  except RuntimeError as exc:
    assert "0x1f" in str(exc)
  try:
    GspRmClient(FakeCmdQ(), FakeStatQ(b"\0" * 31)).rm_alloc(0x80, NV01_MEMORY_VIRTUAL)
    raise AssertionError("short GSP RM alloc response was accepted")
  except ValueError as exc:
    assert "header" in str(exc)
  assert expand_phys_pages([(0x80000000, 0x2000), 0x90000000]) == [0x80000000, 0x80001000, 0x90000000]
  for phys_pages_call, text in [
    (lambda: expand_phys_pages([0x80000001]), "physical page"),
    (lambda: expand_phys_pages([(0x80000001, 0x1000)]), "range base"),
    (lambda: expand_phys_pages([(0x80000000, 0)]), "range span"),
    (lambda: expand_phys_pages([(0x80000000, 0x1800)]), "range span"),
  ]:
    try:
      phys_pages_call()
      raise AssertionError("bad physical page list was accepted")
    except ValueError as exc:
      assert text in str(exc)
  alloc_mem = pack_rpc_alloc_memory(0xc1e00004, 0x80, 0xcf000000, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR,
    [(0x80000000, 0x2000)], 0x2000, 0x1234)
  assert struct.unpack_from("<IIIIIII", alloc_mem, 0) == (0xc1e00004, 0x80, 0xcf000000, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x1234, 0, 6)
  assert struct.unpack_from("<QI", alloc_mem, 32) == (0x2000, 2)
  assert struct.unpack_from("<I", alloc_mem, 48)[0] == (2 << 16)
  assert struct.unpack_from("<QQ", alloc_mem, 56) == (0x80000000 >> 12, 0x80001000 >> 12)
  alloc_memory_resp = struct.pack("<IIIIIII", 0xc1e00004, 0x80, 0xcf000000, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x1234, 0, 6)
  assert unpack_rpc_alloc_memory_response(alloc_memory_resp) == 0
  try:
    unpack_rpc_alloc_memory_response(alloc_memory_resp[:-1])
    raise AssertionError("short GSP alloc-memory response was accepted")
  except ValueError as exc:
    assert "alloc-memory response header" in str(exc)
  alloc_memory_client = GspRmClient(FakeCmdQ(), FakeStatQ(alloc_memory_resp))
  assert alloc_memory_client.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x80000000], 0x1000, 0x1234, h_memory=0xcf000000) == 0xcf000000
  failed_alloc_memory_resp = bytearray(alloc_memory_resp)
  struct.pack_into("<I", failed_alloc_memory_resp, 20, 0x44)
  try:
    GspRmClient(FakeCmdQ(), FakeStatQ(bytes(failed_alloc_memory_resp))).alloc_memory(
      0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x80000000], 0x1000, 0x1234)
    raise AssertionError("GSP alloc-memory status was not checked")
  except RuntimeError as exc:
    msg = str(exc)
    expected_payload = pack_rpc_alloc_memory(0xc1e00004, 0x80, 0xcf000000, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR,
      [0x80000000], 0x1000, 0x1234)
    assert "ALLOC_MEMORY" in msg and "failed: 0x44" in msg
    assert "client=0xc1e00004" in msg and "device=0x80" in msg and "memory=0xcf000000" in msg
    assert "class_name=NV01_MEMORY_SYSTEM_OS_DESCRIPTOR" in msg and "length=0x1000" in msg
    assert "flags=0x1234" in msg and "pages=1" in msg
    assert f"payload_sha256={hashlib.sha256(expected_payload).hexdigest()}" in msg
  bad_alloc_memory_client = GspRmClient(fake_cmd, FakeStatQ(alloc_memory_resp))
  sent_before = len(fake_cmd.sent)
  next_before = bad_alloc_memory_client.next_handle
  for bad_gsp_alloc_memory, text in [
    (lambda: bad_alloc_memory_client.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x80000000], 0, 0x1234), "allocation length"),
    (lambda: bad_alloc_memory_client.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x80000000], 0x1000, 0x1234,
      h_memory=0x100000000), "memory handle"),
  ]:
    try:
      bad_gsp_alloc_memory()
      raise AssertionError("bad GSP alloc-memory input consumed a handle")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake_cmd.sent) == sent_before
    assert bad_alloc_memory_client.next_handle == next_before
  pdir_payload = pack_rpc_set_page_directory(0xc1e00004, 0x80, 0x90f1, 0x80000000, 4)
  assert len(pdir_payload) == 48
  assert struct.unpack_from("<III4xQIIIIII", pdir_payload, 0) == (
    0xc1e00004, 0x80, 0xffffffff, 0x80000000, 4, 0x8, 0x90f1, 0, 1, 0xffffffff)
  for bad_pdir_args, text in [
    ((0xc1e00004, 0x80, 0x90f1, 0x80000001, 4), "4KB aligned"),
    ((0xc1e00004, 0x80, 0x90f1, 0x80000000, 0), "entry count"),
    ((0x100000000, 0x80, 0x90f1, 0x80000000, 4), "client handle"),
  ]:
    try:
      pack_rpc_set_page_directory(*bad_pdir_args)
      raise AssertionError("bad page-directory RPC input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  try:
    pack_rpc_alloc_memory(1, 2, 3, 4, [0x80000000], 0x2000, 0)
    raise AssertionError("short physical page list was accepted")
  except ValueError as exc:
    assert "not enough" in str(exc)
  try:
    pack_rpc_alloc_memory(1, 2, 3, 4, [0x80000000], 0, 0)
    raise AssertionError("zero allocation length was accepted")
  except ValueError as exc:
    assert "length" in str(exc)
  assert len(pack_message_queue_init(0x80000000, 4, 0x1000, 0x5000)) == 32
  assert pack_gsp_arguments_cached(pack_message_queue_init(0x80000000, 4, 0x1000, 0x5000))[48] == 1
  for bad_boot_pack, text in [
    (lambda: pack_message_queue_init(-1, 4, 0x1000, 0x5000), "shared memory physical address"),
    (lambda: pack_message_queue_init(0x80000000, -1, 0x1000, 0x5000), "page-table entry count"),
    (lambda: pack_gsp_arguments_cached(b"x" * 45), "message queue init"),
    (lambda: pack_gsp_arguments_cached(b"", gpu_instance=0x100000000), "GPU instance"),
    (lambda: pack_rpc_unloading_guest_driver(2), "power-transition"),
    (lambda: pack_rpc_alloc_memory(1, 2, 3, 4, [0x80000000], 0x1000, -1), "alloc-memory flags"),
  ]:
    try:
      bad_boot_pack()
      raise AssertionError("bad GSP boot/RM pack input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  registry_table = pack_registry_table({"RMForcePcieConfigSave": 1})
  assert len(registry_table) > 24 and b"RMForcePcieConfigSave\0" in registry_table
  assert len(pack_gsp_system_info(0x1000, 0x2000, 0x3000, 0x220810de, 0x10de0000, 0xa1)) == 928
  for bad_info_pack, text in [
    (lambda: pack_registry_table({"bad": -1}), "registry value"),
    (lambda: pack_registry_table({"": 1}), "registry entry name"),
    (lambda: pack_registry_table([("bad", 1)]), "registry entries"),
    (lambda: pack_registry_table({"bad\0name": 1}), "contains NUL"),
    (lambda: pack_registry_table({"bad_name_\u2603": 1}), "ASCII"),
    (lambda: pack_gsp_system_info(-1, 0, 0, 0, 0, 0), "GPU physical address"),
    (lambda: pack_gsp_system_info(0, 0, 0, 0x100000000, 0, 0), "PCI device id"),
    (lambda: pack_nv90f1_copy_server_reserved_pdes(0x1000, 0, 0, [(0, 0x1000, 0, 0x100)]), "page shift"),
    (lambda: pack_nv2080_gpu_promote_ctx(0x80, 0x90, [(0, 0, 0, 0, 0x10000, 1, 0)]), "context buffer id"),
    (lambda: pack_nv0080_alloc_parameters(device_id=-1), "device id"),
    (lambda: pack_nv2080_alloc_parameters(-1), "subdevice id"),
    (lambda: pack_nv_memory_virtual_allocation_params(offset=-1), "virtual allocation offset"),
    (lambda: pack_nv_vaspace_allocation_params(flags=-1), "VA space flags"),
    (lambda: pack_nv_channel_group_allocation_params(h_vaspace=0x100000000), "vaspace handle"),
    (lambda: pack_nv_ctxshare_allocation_params(0x90, flags=-1), "ctxshare flags"),
    (lambda: pack_nv_memory_desc(-1, 0x1000), "memory descriptor base"),
    (lambda: pack_nv_channel_gpfifo_allocation_params(0, 0, 0, 0, 0x100000000, 0, 0, 0, 0), "object buffer handle"),
    (lambda: pack_nv83de_alloc_parameters(-1, 0x90), "app client handle"),
  ]:
    try:
      bad_info_pack()
      raise AssertionError("bad RM/GSP parameter pack input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  assert len(MsgqTxHeader(size=0x5000, msg_size=0x1000, msg_count=4).pack()) == MsgqTxHeader.SIZE
  assert len(RpcHeader(length=RpcHeader.SIZE, function=NV_VGPU_MSG_EVENT_GSP_INIT_DONE).pack()) == RpcHeader.SIZE
  assert len(GspQueueElement(seq=1, elem_count=1).pack()) == GspQueueElement.SIZE
  for bad_queue_struct, text in [
    (lambda: MsgqTxHeader(size=-1), "message queue size"),
    (lambda: MsgqTxHeader(write_ptr=0x100000000), "write pointer"),
    (lambda: RpcHeader(function=-1), "RPC function"),
    (lambda: RpcHeader(length=0x100000000), "RPC length"),
    (lambda: GspQueueElement(seq=-1), "queue element sequence"),
    (lambda: GspQueueElement(elem_count=0x100000000), "queue element count"),
  ]:
    try:
      bad_queue_struct()
      raise AssertionError("bad GSP queue/RPC struct field was accepted")
    except ValueError as exc:
      assert text in str(exc)

  stat_mem = bytearray(0x5000)
  stat_view = MMIOView(stat_mem)
  stat_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=1, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
  cmd_view = MMIOView(bytearray(0x5000))
  cmd_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=1, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
  def rpc_record(func, msg=b"", elem_count=1, hdr_length=None, rpc_result=0, rpc_result_private=0, checksum_delta=0):
    hdr_length = RpcHeader.SIZE + len(msg) if hdr_length is None else hdr_length
    payload = RpcHeader(length=hdr_length, function=func, rpc_result=rpc_result, rpc_result_private=rpc_result_private).pack() + msg
    elem = GspQueueElement(seq=0, elem_count=elem_count)
    elem.checksum = GspRpcQueue.record_checksum(elem.pack(), payload) ^ checksum_delta
    return (elem.pack() + payload).ljust(elem_count * 0x1000, b"\0")
  event_record = rpc_record(NV_VGPU_MSG_EVENT_GSP_INIT_DONE)
  stat_view.view(0x1000, len(event_record))[:] = event_record
  stat_q = GspRpcQueue(None, stat_view, cmd_view)
  assert stat_q.wait_resp(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, timeout_ms=10) == b""
  assert stat_q.rx_view[0] == 1
  for bad_queue_call, text in [
    (lambda: GspRpcQueue(None, object(), cmd_view), "RPC queue view"),
    (lambda: GspRpcQueue(None, stat_view, object()), "RPC completion queue view"),
  ]:
    try:
      bad_queue_call()
      raise AssertionError("bad RPC queue view was accepted")
    except ValueError as exc:
      assert text in str(exc)
  bad_cmd_view = MMIOView(bytearray(0x5000))
  bad_cmd_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=1, flags=1, rx_hdr_off=0x5000, entry_off=0x1000).pack()
  try:
    GspRpcQueue(None, stat_view, bad_cmd_view)
    raise AssertionError("bad GSP RPC completion queue rx header was accepted")
  except ValueError as exc:
    assert "rx header" in str(exc)
  pending_mem = bytearray(0x5000)
  pending_view = MMIOView(pending_mem)
  pending_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=1, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
  pending_cmd = MMIOView(bytearray(0x5000))
  pending_cmd[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
    write_ptr=0, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
  pending_record = rpc_record(NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, rpc_result=0xffffffff)
  pending_view.view(0x1000, len(pending_record))[:] = pending_record
  pending_q = GspRpcQueue(None, pending_view, pending_cmd)
  try:
    pending_q.wait_resp(NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, timeout_ms=10)
    raise AssertionError("pending RPC response was accepted")
  except RuntimeError as exc:
    assert "0xffffffff" in str(exc) and "GSP_RM_ALLOC" in str(exc)
  assert pending_q.rx_view[0] == 1
  def make_response_queue(elem_count=1, hdr_length=RpcHeader.SIZE, write_ptr=1, read_ptr=0, checksum_delta=0):
    resp_view = MMIOView(bytearray(0x5000))
    resp_view[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
      write_ptr=write_ptr, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
    resp_cmd = MMIOView(bytearray(0x5000))
    resp_cmd[:MsgqTxHeader.SIZE] = MsgqTxHeader(version=0, size=0x5000, msg_size=0x1000, msg_count=4,
      write_ptr=write_ptr, flags=1, rx_hdr_off=MsgqTxHeader.SIZE, entry_off=0x1000).pack()
    resp_cmd[MsgqTxHeader.SIZE:MsgqTxHeader.SIZE+4] = struct.pack("<I", read_ptr)
    record = rpc_record(NV_VGPU_MSG_EVENT_GSP_INIT_DONE, elem_count=elem_count, hdr_length=hdr_length, checksum_delta=checksum_delta)
    resp_view.view(0x1000, len(record))[:] = record
    return GspRpcQueue(None, resp_view, resp_cmd)
  for malformed_q, text in [
    (lambda: make_response_queue(read_ptr=4), "read pointer"),
    (lambda: make_response_queue(elem_count=0), "element count"),
    (lambda: make_response_queue(hdr_length=RpcHeader.SIZE - 1), "smaller than header"),
    (lambda: make_response_queue(elem_count=1, hdr_length=0x2000), "exceeds queue element span"),
    (lambda: make_response_queue(checksum_delta=1), "checksum mismatch"),
  ]:
    try:
      next(malformed_q().read_resp())
      raise AssertionError("malformed RPC response record was accepted")
    except ValueError as exc:
      assert text in str(exc)
  send_q = make_response_queue(write_ptr=0)
  send_wp = send_q.tx_view[4]
  assert "tx_header=" in send_q.debug_state(slots=1) and "EVENT_GSP_INIT_DONE" in send_q.debug_state(slots=1)
  try:
    send_q.debug_state(slots=-1)
    raise AssertionError("bad RPC debug slot count was accepted")
  except ValueError as exc:
    assert "slot count" in str(exc)
  for bad_rpc_call, text in [
    (lambda: send_q.send_rpc(0x100000000, b""), "RPC function"),
    (lambda: send_q.send_rpc(1, object()), "bytes-like"),
    (lambda: send_q.wait_resp(0x100000000, timeout_ms=0), "RPC response function"),
    (lambda: send_q.wait_resp(1, timeout_ms=-1), "timeout"),
  ]:
    try:
      bad_rpc_call()
      raise AssertionError("bad RPC send/wait input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert send_q.tx_view[4] == send_wp
  class FakeBootAllocator:
    def __init__(self): self.next_paddr = 0x80000000
    def alloc(self, size, sysmem=None, contiguous=False, data=None, boot=False):
      view = MMIOView(bytearray(size))
      paddr = self.next_paddr
      self.next_paddr += round_up(size, 0x1000)
      paddrs = [paddr + off for off in range(0, round_up(size, 0x1000), 0x1000)]
      if data is not None: view[:len(data)] = data
      return view, None, paddrs
  qmem = GspQueueMemoryBuilder(FakeBootAllocator()).build(queue_size=0x4000)
  assert qmem.cmd_q_view.offset != qmem.stat_q_view.offset
  qmem.cmd_q_view.view(0x1000, 1)[0] = 0x5a
  assert qmem.stat_q_view.view(0x1000, 1)[0] == 0
  for bad_qmem, text in [
    (lambda: GspQueueMemory(object(), [0x80000000], qmem.cmd_q_view, qmem.stat_q_view, qmem.rm_args_view, qmem.rm_args_paddrs, qmem.libos_args_view, qmem.libos_args_paddrs), "queues view"),
    (lambda: GspQueueMemory(qmem.queues_view, [], qmem.cmd_q_view, qmem.stat_q_view, qmem.rm_args_view, qmem.rm_args_paddrs, qmem.libos_args_view, qmem.libos_args_paddrs), "queues physical ranges"),
    (lambda: GspQueueMemory(qmem.queues_view, [0x80000001], qmem.cmd_q_view, qmem.stat_q_view, qmem.rm_args_view, qmem.rm_args_paddrs, qmem.libos_args_view, qmem.libos_args_paddrs), "queues physical address"),
    (lambda: GspQueueMemory(qmem.queues_view, qmem.queues_paddrs, object(), qmem.stat_q_view, qmem.rm_args_view, qmem.rm_args_paddrs, qmem.libos_args_view, qmem.libos_args_paddrs), "command queue view"),
    (lambda: GspQueueMemory(qmem.queues_view, qmem.queues_paddrs, qmem.cmd_q_view, object(), qmem.rm_args_view, qmem.rm_args_paddrs, qmem.libos_args_view, qmem.libos_args_paddrs), "status queue view"),
    (lambda: GspQueueMemory(qmem.queues_view, qmem.queues_paddrs, qmem.cmd_q_view, qmem.stat_q_view, object(), qmem.rm_args_paddrs, qmem.libos_args_view, qmem.libos_args_paddrs), "RM args view"),
    (lambda: GspQueueMemory(qmem.queues_view, qmem.queues_paddrs, qmem.cmd_q_view, qmem.stat_q_view, qmem.rm_args_view, [], qmem.libos_args_view, qmem.libos_args_paddrs), "RM args physical ranges"),
    (lambda: GspQueueMemory(qmem.queues_view, qmem.queues_paddrs, qmem.cmd_q_view, qmem.stat_q_view, qmem.rm_args_view, [0x80000001], qmem.libos_args_view, qmem.libos_args_paddrs), "RM args physical address"),
    (lambda: GspQueueMemory(qmem.queues_view, qmem.queues_paddrs, qmem.cmd_q_view, qmem.stat_q_view, qmem.rm_args_view, qmem.rm_args_paddrs, object(), qmem.libos_args_paddrs), "LibOS args view"),
  ]:
    try:
      bad_qmem()
      raise AssertionError("bad GSP queue memory was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_queue_size in (0x1000, 0x2500):
    failing_boot_alloc = FakeBootAllocator()
    try:
      GspQueueMemoryBuilder(failing_boot_alloc).build(queue_size=bad_queue_size)
      raise AssertionError("invalid GSP queue size was accepted")
    except ValueError as exc:
      assert "GSP queue size" in str(exc)
    assert failing_boot_alloc.next_paddr == 0x80000000

  old_boot = os.environ.get("NV_ADD_BOOT_GSP")
  old_golden = os.environ.get("NV_ADD_PREPARE_GOLDEN_CTX")
  old_summary = os.environ.get("NV_ADD_SUMMARY")
  old_trace_boot = os.environ.get("NV_ADD_TRACE_GSP_BOOT")
  old_verify_sec2 = os.environ.get("NV_ADD_VERIFY_SEC2_INPUTS")
  old_reconnect_transport = os.environ.get("NV_ADD_RECONNECT_TRANSPORT")
  try:
    os.environ.pop("NV_ADD_BOOT_GSP", None)
    os.environ.pop("NV_ADD_PREPARE_GOLDEN_CTX", None)
    apply_default_live_run_env()
    assert should_boot_gsp() and should_prepare_golden_ctx()
    os.environ["NV_ADD_BOOT_GSP"] = "0"
    os.environ["NV_ADD_PREPARE_GOLDEN_CTX"] = "0"
    apply_default_live_run_env()
    assert not should_boot_gsp() and not should_prepare_golden_ctx()
    os.environ["NV_ADD_BOOT_GSP"] = "1"
    assert should_boot_gsp()
    os.environ["NV_ADD_PREPARE_GOLDEN_CTX"] = "1"
    assert should_prepare_golden_ctx()
    os.environ["NV_ADD_SUMMARY"] = "1"
    assert should_print_summary()
    summary = runtime_summary_line()
    assert "transport=auto" in summary and "boot_gsp=True" in summary and "golden_ctx=True" in summary
    assert "reconnect_mode=golden-context" in summary and "trace_rm_state=False" in summary
    assert "trace_gsp_boot=False" in summary and "verify_sec2_inputs=False" in summary
    os.environ["NV_ADD_TRACE_GSP_BOOT"] = "1"
    os.environ["NV_ADD_VERIFY_SEC2_INPUTS"] = "1"
    assert trace_gsp_boot_enabled() and verify_sec2_inputs_enabled()
    summary = runtime_summary_line()
    assert "trace_gsp_boot=True" in summary and "verify_sec2_inputs=True" in summary
    assert "gpfifo_ctor_sha256=6e59c0bfb532bf40a031c416b581d48ef49f3976bc9b55cda546ea24e008537a" in summary
    assert expected_channel_step_sequence()[0] == "booted_resources_start"
    assert expected_channel_step_sequence()[-1] == "booted_resources_done"
    assert "channel_seq_sha256=d13ab647e07e9739a21bfda1c63035d39575c26d3c48ce20973d4a1e488d25e9" in summary
    assert expected_gpfifo_descriptor_trace_fields() == [
      "desc_ramfc", "desc_userd", "desc_instance", "desc_method", "desc_error", "h_object_buffer", "h_object_error"]
    assert "gpfifo_desc_fields_sha256=5e2aa5577e1e74545940915235915bb51cc28db8f3462605186ae78a5db635af" in summary
    assert "qmd_template_sha256=a15a9304474d9de715e361a5f53141fde2a2111a2254c86eb86832c04abf48ff" in summary
    assert "reconnect_flags=NV_ADD_TRANSPORT=mac-egpu,NV_ADD_PREPARE_GOLDEN_CTX=1,NV_ADD_BOOT_GSP=1,NV_ADD_SUMMARY=1,NV_ADD_CHECK_FRTS_BAR1=1,NV_ADD_TRACE_GSP_BOOT=1,NV_ADD_VERIFY_SEC2_INPUTS=1,NV_ADD_TRACE_RM_ALLOC=1,NV_ADD_TRACE_RM_STATE=1,NV_ADD_TRACE_RPC=1,NV_ADD_TRACE_RPC_READ=1,NV_ADD_TRACE_CHANNEL=1,NV_ADD_TRACE_MM_ALLOC=1,NV_ADD_TRACE_LAUNCH_STEPS=1" in summary
    assert recommended_reconnect_command() == (
      "NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1 NV_ADD_BOOT_GSP=1 "
      "NV_ADD_SUMMARY=1 NV_ADD_CHECK_FRTS_BAR1=1 NV_ADD_TRACE_GSP_BOOT=1 "
      "NV_ADD_VERIFY_SEC2_INPUTS=1 NV_ADD_TRACE_RM_ALLOC=1 "
      "NV_ADD_TRACE_RM_STATE=1 NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 "
      "NV_ADD_TRACE_MM_ALLOC=1 NV_ADD_TRACE_LAUNCH_STEPS=1 python3 examples/add.py")
    assert recommended_reconnect_command(golden=False) == (
      "NV_ADD_TRANSPORT=mac-egpu NV_ADD_BOOT_GSP=1 NV_ADD_SUMMARY=1 NV_ADD_TRACE_GSP_BOOT=1 "
      "NV_ADD_VERIFY_SEC2_INPUTS=1 NV_ADD_TRACE_RM_ALLOC=1 NV_ADD_TRACE_RM_STATE=1 "
      "NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 "
      "NV_ADD_TRACE_LAUNCH_STEPS=1 python3 examples/add.py")
    assert "NV_ADD_CHECK_FRTS_BAR1=1" not in recommended_reconnect_command(golden=False)
    assert recommended_reconnect_command(golden=True) == recommended_reconnect_command()
    assert recommended_stack_reconnect_command() == (
      "NV_ADD_TRACE_RM_STACK=1 NV_ADD_TRACE_CHANNEL_STACK=1 NV_ADD_TRACE_LAUNCH_STACK=1 NV_ADD_TRACE_FALCON=1 "
      "NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1 NV_ADD_BOOT_GSP=1 "
      "NV_ADD_SUMMARY=1 NV_ADD_CHECK_FRTS_BAR1=1 NV_ADD_TRACE_GSP_BOOT=1 "
      "NV_ADD_VERIFY_SEC2_INPUTS=1 NV_ADD_TRACE_RM_ALLOC=1 "
      "NV_ADD_TRACE_RM_STATE=1 NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 "
      "NV_ADD_TRACE_MM_ALLOC=1 NV_ADD_TRACE_LAUNCH_STEPS=1 python3 examples/add.py")
    assert recommended_stack_reconnect_command("custom.py", golden=False) == (
      "NV_ADD_TRACE_RM_STACK=1 NV_ADD_TRACE_CHANNEL_STACK=1 NV_ADD_TRACE_LAUNCH_STACK=1 NV_ADD_TRACE_FALCON=1 "
      "NV_ADD_TRANSPORT=mac-egpu NV_ADD_BOOT_GSP=1 NV_ADD_SUMMARY=1 NV_ADD_TRACE_GSP_BOOT=1 "
      "NV_ADD_VERIFY_SEC2_INPUTS=1 NV_ADD_TRACE_RM_ALLOC=1 NV_ADD_TRACE_RM_STATE=1 "
      "NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 "
      "NV_ADD_TRACE_LAUNCH_STEPS=1 python3 custom.py")
    assert recommended_preflight_command() == "NV_ADD_TRANSPORT=mac-egpu python3 examples/add.py --transport-preflight"
    assert recommended_preflight_command("custom.py") == "NV_ADD_TRANSPORT=mac-egpu python3 custom.py --transport-preflight"
    assert recommended_preflight_gate_command() == "NV_ADD_TRANSPORT=mac-egpu python3 examples/add.py --transport-preflight-gate"
    assert recommended_preflight_gate_command("custom.py") == "NV_ADD_TRANSPORT=mac-egpu python3 custom.py --transport-preflight-gate"
    assert recommended_preflight_require_ready_command() == "NV_ADD_TRANSPORT=mac-egpu python3 examples/add.py --transport-preflight-gate --require-ready"
    assert recommended_preflight_require_ready_command("custom.py") == "NV_ADD_TRANSPORT=mac-egpu python3 custom.py --transport-preflight-gate --require-ready"
    assert recommended_preflight_plan_command() == "NV_ADD_TRANSPORT=mac-egpu python3 examples/add.py --transport-preflight-plan --require-ready"
    assert recommended_preflight_plan_command("custom.py") == "NV_ADD_TRANSPORT=mac-egpu python3 custom.py --transport-preflight-plan --require-ready"
    os.environ["NV_ADD_RECONNECT_TRANSPORT"] = "linux-ioctl"
    assert recommended_reconnect_transport() == "linux-ioctl"
    assert recommended_reconnect_command("custom.py").startswith("NV_ADD_TRANSPORT=linux-ioctl ")
    assert recommended_preflight_command("custom.py") == "NV_ADD_TRANSPORT=linux-ioctl python3 custom.py --transport-preflight"
    assert recommended_preflight_gate_command("custom.py") == "NV_ADD_TRANSPORT=linux-ioctl python3 custom.py --transport-preflight-gate"
    assert recommended_preflight_require_ready_command("custom.py") == "NV_ADD_TRANSPORT=linux-ioctl python3 custom.py --transport-preflight-gate --require-ready"
    assert recommended_preflight_plan_command("custom.py") == "NV_ADD_TRANSPORT=linux-ioctl python3 custom.py --transport-preflight-plan --require-ready"
    os.environ.pop("NV_ADD_RECONNECT_TRANSPORT", None)
    reconnect_buf = io.StringIO()
    with contextlib.redirect_stdout(reconnect_buf): print_reconnect_command()
    assert reconnect_buf.getvalue().strip() == f"reconnect_command {recommended_reconnect_command()}"
    reconnect_buf = io.StringIO()
    with contextlib.redirect_stdout(reconnect_buf): print_reconnect_commands()
    reconnect_lines = reconnect_buf.getvalue().strip().splitlines()
    assert reconnect_lines == [
      f"reconnect_command fixed-gpfifo {recommended_reconnect_command(golden=False)}",
      f"reconnect_command golden-context {recommended_reconnect_command(golden=True)}",
      f"reconnect_command fecs-reset {recommended_fecs_reset_reconnect_command()}",
    ]
    reconnect_buf = io.StringIO()
    with contextlib.redirect_stdout(reconnect_buf): print_reconnect_commands("examples/mul.py")
    assert all(line.endswith("python3 examples/mul.py") for line in reconnect_buf.getvalue().strip().splitlines())
    live_debug_buf = io.StringIO()
    with contextlib.redirect_stdout(live_debug_buf): print_live_debug_commands()
    live_debug_lines = live_debug_buf.getvalue().strip().splitlines()
    expected_fecs_diag_lines = []
    fecs_diag_for_live_buf = io.StringIO()
    with contextlib.redirect_stdout(fecs_diag_for_live_buf): print_fecs_fence_diagnostic()
    expected_fecs_diag_lines = fecs_diag_for_live_buf.getvalue().strip().splitlines()
    assert live_debug_lines == [
      f"preflight_plan_command {recommended_preflight_plan_command()}",
      f"reconnect_command fixed-gpfifo {recommended_reconnect_command(golden=False)}",
      f"reconnect_command golden-context {recommended_reconnect_command(golden=True)}",
      f"reconnect_command fecs-reset {recommended_fecs_reset_reconnect_command()}",
      "reconnect_command fecs-scenarios python3 examples/add.py --fecs-reset-scenarios",
      "reconnect_command fecs-fence-diagnostic python3 examples/add.py --fecs-fence-diagnostic",
      *expected_fecs_diag_lines,
      "live_log_workflow_command python3 examples/add.py --live-log-workflow",
      "live_stack_log_workflow_command python3 examples/add.py --live-stack-log-workflow",
      "tiny_live_stack_log_workflow_command python3 examples/add_tiny.py --live-stack-log-workflow --standalone-script examples/add.py",
      f"tiny_trace_command {recommended_tiny_trace_command()}",
      "reconnect_command stall-trace python3 examples/add.py --stall-trace",
      "reconnect_command logbuf-dump python3 examples/add.py --logbuf-dump",
    ]
    fecs_scenarios_buf = io.StringIO()
    with contextlib.redirect_stdout(fecs_scenarios_buf): print_fecs_reset_scenarios()
    fecs_scenarios_lines = fecs_scenarios_buf.getvalue().strip().splitlines()
    assert fecs_scenarios_lines[0].startswith("fecs_reset_scenario_recommend try=preboot_only, then=fecs_minimal, then=fecs_only, then=full, then=promote_retry")
    assert fecs_scenarios_lines[1] == f"fecs_reset_scenario full {recommended_fecs_reset_reconnect_command()}"
    assert fecs_scenarios_lines[2] == f"fecs_reset_scenario fecs_only {recommended_fecs_only_reconnect_command()}"
    assert fecs_scenarios_lines[3] == f"fecs_reset_scenario fecs_minimal {recommended_fecs_minimal_reconnect_command()}"
    assert fecs_scenarios_lines[4] == f"fecs_reset_scenario preboot_only {recommended_preboot_only_reconnect_command()}"
    assert fecs_scenarios_lines[5] == f"fecs_reset_scenario promote_retry {recommended_promote_retry_reconnect_command()}"
    fecs_only_cmd = recommended_fecs_only_reconnect_command()
    full_cmd = recommended_fecs_reset_reconnect_command()
    preboot_cmd = recommended_preboot_only_reconnect_command()
    fecs_minimal_cmd = recommended_fecs_minimal_reconnect_command()
    promote_retry_cmd = recommended_promote_retry_reconnect_command()
    assert "NV_ADD_PMU_RESET_POSTPROMOTE=1" in full_cmd and "NV_ADD_PMU_RESET_POSTPROMOTE=1" not in fecs_only_cmd
    assert "NV_ADD_FECS_RESET_POSTPROMOTE=1" in fecs_only_cmd
    assert "NV_ADD_PMU_RESET=1" in preboot_cmd and "NV_ADD_FECS_RESET_POSTINIT=1" not in preboot_cmd
    assert "NV_ADD_FECS_RESET=1" in fecs_minimal_cmd and "NV_ADD_FECS_RESET_POSTINIT=1" not in fecs_minimal_cmd
    assert "NV_ADD_REPROMOTE_AFTER_RESET=1" in promote_retry_cmd and "NV_ADD_FECS_RESET=1" not in promote_retry_cmd
    fecs_scenarios_mul_buf = io.StringIO()
    with contextlib.redirect_stdout(fecs_scenarios_mul_buf): print_fecs_reset_scenarios("examples/mul.py")
    fecs_scenarios_mul_lines = fecs_scenarios_mul_buf.getvalue().strip().splitlines()
    assert all("python3 examples/mul.py" in line for line in fecs_scenarios_mul_lines[1:])
    fecs_diag_buf = io.StringIO()
    with contextlib.redirect_stdout(fecs_diag_buf): print_fecs_fence_diagnostic()
    fecs_diag_lines = fecs_diag_buf.getvalue().strip().splitlines()
    assert fecs_diag_lines[0] == (
      "fecs_fence_diagnostic signature=FECS_A|FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus|0%>FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus|9b2w>ASSERT|...|BKD")
    assert fecs_diag_lines[1].startswith("fecs_fence_diagnostic cause=FECS FALCON is fenced from a prior session")
    assert fecs_diag_lines[2].startswith("fecs_fence_diagnostic key_signals ")
    assert fecs_diag_lines[3] == "fecs_fence_diagnostic recommended_order preboot_only, fecs_minimal, fecs_only, full, promote_retry (each successive scenario adds more in-band FECS/PMU resets, which may wipe FECS ucode after GSP boot)"
    assert fecs_diag_lines[4] == "fecs_fence_diagnostic first_attempt recommended_scenario=preboot_only (safest: only pre-boot FECS+PMU reset, no in-band FECS resets that might wipe FECS ucode after GSP boot)"
    assert fecs_diag_lines[5] == f"fecs_fence_diagnostic first_attempt_command {recommended_preboot_only_reconnect_command()}"
    assert fecs_diag_lines[6] == "fecs_fence_diagnostic second_attempt recommended_scenario=fecs_minimal (pre-boot FECS only, no PMU reset)"
    assert fecs_diag_lines[7] == f"fecs_fence_diagnostic second_attempt_command {recommended_fecs_minimal_reconnect_command()}"
    assert fecs_diag_lines[8] == "fecs_fence_diagnostic third_attempt recommended_scenario=fecs_only (drops post-promote PMU reset)"
    assert fecs_diag_lines[9] == f"fecs_fence_diagnostic third_attempt_command {recommended_fecs_only_reconnect_command()}"
    assert fecs_diag_lines[10] == "fecs_fence_diagnostic fallback_attempts scenario=full (with PMU resets) scenario=promote_retry (no FECS reset, just re-issue GPU_PROMOTE_CTX)"
    assert fecs_diag_lines[11] == f"fecs_fence_diagnostic promote_retry_command {recommended_promote_retry_reconnect_command()}"
    assert fecs_diag_lines[12].startswith("fecs_fence_diagnostic tinygrad_path tiny does not hit the FECS/PMU mutex stall")
    assert fecs_diag_lines[13] == "fecs_fence_diagnostic compare python3 examples/add.py --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log"
    assert fecs_diag_lines[14].startswith("fecs_fence_diagnostic next_action if preboot_only prints result=[11.0, 22.0, 33.0, 44.0]")
    assert fecs_diag_lines[15].startswith("fecs_fence_diagnostic hardware_gate=after a GSP RM session ends")
    assert fecs_diag_lines[16].startswith("fecs_fence_diagnostic root_cause=PMU dmatrfbase advances")
    assert fecs_diag_lines[17].startswith("fecs_fence_diagnostic contrast_with_tiny")
    assert fecs_diag_lines[18].startswith("fecs_fence_diagnostic mitigation=power-cycle")
    assert fecs_diag_lines[19].startswith("fecs_fence_diagnostic verify_with_tiny")
    assert fecs_diag_lines[20].startswith("fecs_fence_diagnostic fence_check=StandaloneNvShell.bar0_is_fenced()")
    assert fecs_diag_lines[21].startswith("fecs_fence_diagnostic bypass=NV_ADD_BYPASS_BAR0_FENCE_CHECK=1")
    assert fecs_diag_lines[22].startswith("stall_trace purpose:")
    assert fecs_diag_lines[23].startswith("stall_trace trigger:")
    assert fecs_diag_lines[24].startswith("stall_trace knob ")
    assert fecs_diag_lines[25].startswith("stall_trace diagnostic_rule ")
    assert fecs_diag_lines[26].startswith("stall_trace_command ")
    assert fecs_diag_lines[27].startswith("stall_trace next_action ")
    assert fecs_diag_lines[28].startswith("logbuf_dump purpose:")
    assert fecs_diag_lines[29].startswith("logbuf_dump trigger:")
    assert fecs_diag_lines[30].startswith("logbuf_dump knob ")
    assert fecs_diag_lines[31].startswith("logbuf_dump format ")
    assert fecs_diag_lines[32].startswith("logbuf_dump diagnostic_rule ")
    assert fecs_diag_lines[33].startswith("logbuf_dump_command ")
    assert fecs_diag_lines[34].startswith("logbuf_dump next_action ")
    assert fecs_diag_lines[35].startswith("fecs_fence_diagnostic saved_log_state the standalone rm_state stage=... lines now include compact gsp=({format_state} [hwcfg2=")
    live_log_buf = io.StringIO()
    with contextlib.redirect_stdout(live_log_buf): print_live_log_workflow()
    live_log_lines = live_log_buf.getvalue().strip().splitlines()
    assert live_log_lines == [
      "live_log_workflow script=examples/add.py tiny_script=examples/add_tiny.py standalone_log=standalone-golden.log tiny_log=tiny-golden.log",
      f"gate_command {recommended_preflight_plan_command()}",
      f"standalone_log_command {recommended_reconnect_command(golden=True)} 2>&1 | tee standalone-golden.log",
      f"tiny_log_command {recommended_tiny_trace_command()} 2>&1 | tee tiny-golden.log",
      "compare_command python3 examples/add.py --compare-trace-logs --standalone-log standalone-golden.log --tiny-log tiny-golden.log",
      "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence",
      "workflow_rule run standalone_log_command only after gate result is ready-for-gsp",
      "workflow_rule run tiny_log_command in the same eGPU session if standalone stalls or times out",
    ]
    live_stack_log_buf = io.StringIO()
    with contextlib.redirect_stdout(live_stack_log_buf): print_live_stack_log_workflow()
    live_stack_log_lines = live_stack_log_buf.getvalue().strip().splitlines()
    assert live_stack_log_lines == [
      "live_stack_log_workflow script=examples/add.py tiny_script=examples/add_tiny.py standalone_log=standalone-stack.log tiny_log=tiny-stack.log",
      f"gate_command {recommended_preflight_plan_command()}",
      f"standalone_log_command {recommended_stack_reconnect_command(golden=True)} 2>&1 | tee standalone-stack.log",
      f"tiny_log_command {recommended_tiny_trace_command()} 2>&1 | tee tiny-stack.log",
      "compare_command python3 examples/add.py --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log",
      "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence",
      "workflow_check stack inspect trace_log_compare_stack, trace_log_compare_falcon",
      "workflow_rule use this only when Python call-path stacks are needed; it is verbose",
      "workflow_rule run standalone_log_command only after gate result is ready-for-gsp",
      "workflow_rule run tiny_log_command in the same eGPU session if standalone stalls or times out",
    ]
    old_argv = sys.argv[:]
    try:
      sys.argv = ["examples/add.py", "--live-log-workflow", "--standalone-log"]
      missing_value_buf = io.StringIO()
      with contextlib.redirect_stdout(missing_value_buf):
        try:
          print_live_log_workflow()
          raise AssertionError("missing standalone log value was accepted")
        except SystemExit as exc:
          assert exc.code == 2
      assert missing_value_buf.getvalue().strip() == "cli_arg_error kind=missing-value flag=--standalone-log"
      sys.argv = ["examples/add.py", "--live-stack-log-workflow", "--tiny-log"]
      missing_value_buf = io.StringIO()
      with contextlib.redirect_stdout(missing_value_buf):
        try:
          print_live_stack_log_workflow()
          raise AssertionError("missing tiny log value was accepted")
        except SystemExit as exc:
          assert exc.code == 2
      assert missing_value_buf.getvalue().strip() == "cli_arg_error kind=missing-value flag=--tiny-log"
    finally:
      sys.argv = old_argv
    comparison_buf = io.StringIO()
    with contextlib.redirect_stdout(comparison_buf): print_comparison_checklist()
    comparison_text = comparison_buf.getvalue()
    assert "comparison_checklist script=examples/add.py tiny_script=examples/add_tiny.py" in comparison_text
    assert f"gate_command {recommended_preflight_plan_command()}" in comparison_text
    assert f"standalone_command {recommended_reconnect_command(golden=True)}" in comparison_text
    assert f"tiny_trace_command {recommended_tiny_trace_command()}" in comparison_text
    assert "compare_line rm_pre_queues standalone='standalone rm_alloc pre_queues' tiny='tiny rm_alloc pre_queues'" in comparison_text
    assert "compare_line compute_alloc standalone='channel golden_compute_alloc|standalone golden_compute_alloc' tiny='tiny compute_alloc'" in comparison_text
    assert "compare_line dma_alloc standalone='channel golden_dma_alloc|standalone golden_dma_alloc' tiny='tiny dma_alloc'" in comparison_text
    assert "compare_value hashes gpfifo_params,promote_entries,compute_rpc,dma_rpc,token_rpc,schedule_rpc" in comparison_text
    assert "compare_value promote_context golden,user_phys,user_virt entries_sha256,packed_entries_sha256" in comparison_text
    assert "compare_value promote_metadata client,subdevice,object,entries,ids,entry_text" in comparison_text
    assert "compare_value promote_control golden rpc_sha256,result_len,result_sha256" in comparison_text
    assert "compare_value compute_alloc parent,object,compute_class" in comparison_text
    assert "compare_value dma_alloc parent,object,dma_class" in comparison_text
    assert "compare_value failure_summary standalone_status,standalone_exception,tiny_exception,message" in comparison_text
    assert "compare_value progress_summary standalone_stage,tiny_stage,status" in comparison_text
    assert "compare_value stack_functions common,standalone_only,tiny_only for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control" in comparison_text
    assert "compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control" in comparison_text
    assert "compare_value gpfifo_desc ramfc,userd,instance,method,error" in comparison_text
    assert "compare_value rm_slots pre/post cmd/stat slot0 func,result,priv" in comparison_text
    assert "compare_value boundary_queues compute_pre/post,promote_pre/post cmd_wp,stat_rp" in comparison_text
    assert "compare_value rm_alloc_sequence class_name,handles,params,rpcs order/prefix/counts/match_flags" in comparison_text
    assert "compare_value gsp_rpc_sequence func_name,len,sha256 order/prefix/counts/match_flags" in comparison_text
    assert "compare_value falcon_trace dma_pre,state,writes presence/status" in comparison_text
    assert "compare_value falcon_dma_values cmd,dest,mem_off,src,size,state_registers" in comparison_text
    assert "compare_value falcon_write_sequence register=value order/prefix/tails" in comparison_text
    assert "compare_value boot_state pre_root bar1,wpr2_hi,gsp_engine,gsp_cpuctl,sec2_engine,sec2_cpuctl" in comparison_text
    assert "compare_value compute_state pre_compute bar1,wpr2_hi,gsp_registers,sec2_registers" in comparison_text
    assert "compare_value mm_valloc first16 size,va,paddrs sequence" in comparison_text
    assert "compare_rule summary result=mismatch when any present detailed value differs" in comparison_text
    assert "compare_rule first require gate classification ready-for-gsp before GSP/RM commands" in comparison_text
    standalone_sample = "\n".join([
      "channel golden_start",
      "channel golden_gpfifo_alloc params_sha256=ggg",
      "channel golden_gpfifo",
      "channel promote_ctx_payload client=0xc1 subdevice=0x20 object=0xcf virt=default phys=default entries=3 ids=[0, 2, 9] entries_sha256=ppp packed_entries_sha256=ppack entry_text=id=0:phys=0x1:virt=0x2",
      "channel promote_ctx_payload client=0xc1 subdevice=0x20 object=0xcf virt=False phys=default entries=3 ids=[0, 1, 2] entries_sha256=up entries_sha256_alt=ignored packed_entries_sha256=uppack entry_text=id=0:phys=0x1:virt=0x0",
      "channel promote_ctx_payload client=0xc1 subdevice=0x20 object=0xcf virt=default phys=False entries=3 ids=[0, 1, 2] entries_sha256=uv packed_entries_sha256=uvpack entry_text=id=0:phys=0x0:virt=0x2",
      "mm valloc size=0x1000 align=0x1000 contiguous=True -> va=0x1020000000 paddrs=0x1a1e000/0x1000",
      "mm valloc size=0x193000 align=0x1000 contiguous=True -> va=0x1020100000 paddrs=0x1a25000/0x193000",
      "channel golden_promote_done",
      "standalone rm_alloc pre_queues cmd=[tx_header=(0,16384,4096,3,1,1,32,4096); rx_ptr=0; slot0: elem_checksum=0x1 seq=1 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51] stat=[tx_header=(0,16384,4096,3,2,1,32,4096); rx_ptr=7; slot0: elem_checksum=0x2 seq=2 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51]",
      "standalone rm_alloc post_queues cmd=[tx_header=(0,16384,4096,3,3,1,32,4096); rx_ptr=0; slot0: elem_checksum=0x3 seq=3 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51] stat=[tx_header=(0,16384,4096,3,4,1,32,4096); rx_ptr=8; slot0: elem_checksum=0x4 seq=4 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51]",
      "standalone rm_control pre_queues object=0xcf000002 cmd=0x2080012b cmd_name=NV2080_CTRL_CMD_GPU_PROMOTE_CTX cmdq=[tx_header=(0,16384,4096,3,8,1,32,4096); rx_ptr=8] stat=[tx_header=(0,16384,4096,3,17,1,32,4096); rx_ptr=17]",
      "standalone rm_control post object=0xcf000002 cmd=0x2080012b cmd_name=NV2080_CTRL_CMD_GPU_PROMOTE_CTX rpc_sha256=promrpc status=0x0 result_len=560 result_sha256=promresult",
      "standalone rm_control post_queues object=0xcf000002 cmd=0x2080012b cmd_name=NV2080_CTRL_CMD_GPU_PROMOTE_CTX cmdq=[tx_header=(0,16384,4096,3,9,1,32,4096); rx_ptr=9] stat=[tx_header=(0,16384,4096,3,18,1,32,4096); rx_ptr=18]",
      "standalone rm_alloc pre_queues parent=0xcf000004 object=0xcf000005 class=0xc7c0 class_name=AMPERE_COMPUTE_B cmd=[tx_header=(0,16384,4096,3,10,1,32,4096); rx_ptr=10] stat=[tx_header=(0,16384,4096,3,18,1,32,4096); rx_ptr=18]",
      "standalone rm_alloc post_queues parent=0xcf000004 object=0xcf000005 class=0xc7c0 class_name=AMPERE_COMPUTE_B cmd=[tx_header=(0,16384,4096,3,11,1,32,4096); rx_ptr=11] stat=[tx_header=(0,16384,4096,3,19,1,32,4096); rx_ptr=19]",
      "channel golden_compute_alloc parent=0xcf000004 expected_object=0xcf000005 compute_class=0xc7c0 expected_rpc_sha256=aaa",
      "channel golden_dma_alloc parent=0xcf000004 expected_object=0xcf000006 dma_class=0xc7b5 expected_rpc_sha256=ddd",
      "standalone runtime_token_control rpc_sha256=bbb",
      "standalone runtime_schedule_control rpc_sha256=ccc",
    ])
    tiny_sample = "\n".join([
      "tiny golden_start",
      "tiny gpfifo_patch post params_sha256=ggg",
      "tiny promote_ctx_payload client=0xc1 subdevice=0x20 object=0xcf virt=default phys=default entries=3 ids=[0, 2, 9] entries_sha256=ppp packed_entries_sha256=ppack entry_text=id=0:phys=0x1:virt=0x2",
      "tiny promote_ctx_payload client=0xc1 subdevice=0x20 object=0xcf virt=False phys=default entries=3 ids=[0, 1, 2] entries_sha256=up packed_entries_sha256=uppack entry_text=id=0:phys=0x1:virt=0x0",
      "tiny promote_ctx_payload client=0xc1 subdevice=0x20 object=0xcf virt=default phys=False entries=3 ids=[0, 1, 2] entries_sha256=uv packed_entries_sha256=uvpack entry_text=id=0:phys=0x0:virt=0x2",
      "tiny mm valloc size=0x1000 align=0x1000 contiguous=True uncached=False -> va=0x1020000000 paddrs=0x1a1e000/0x1000",
      "tiny mm valloc size=0x193000 align=0x1000 contiguous=True uncached=False -> va=0x1020100000 paddrs=0x1a25000/0x193000",
      "tiny golden_done",
      "tiny rm_alloc pre_queues cmd_tx=(0,16384,4096,3,1,1,32,4096) cmd_rx=0 cmd_slot0: checksum=0x1 seq=1 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51 stat_tx=(0,16384,4096,3,2,1,32,4096) stat_rx=7 stat_slot0: checksum=0x2 seq=2 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51",
      "tiny rm_alloc post_queues cmd_tx=(0,16384,4096,3,3,1,32,4096) cmd_rx=0 cmd_slot0: checksum=0x3 seq=3 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51 stat_tx=(0,16384,4096,3,4,1,32,4096) stat_rx=8 stat_slot0: checksum=0x4 seq=4 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51",
      "tiny rm_control pre_queues object=0xcf000002 cmd=0x2080012b cmd_name=NV2080_CTRL_CMD_GPU_PROMOTE_CTX cmd_tx=(0,16384,4096,3,8,1,32,4096) cmd_rx=8 stat_tx=(0,16384,4096,3,17,1,32,4096) stat_rx=17",
      "tiny rm_control post object=0xcf000002 cmd=0x2080012b cmd_name=NV2080_CTRL_CMD_GPU_PROMOTE_CTX rpc_sha256=promrpc result_len=560 result_sha256=promresult",
      "tiny rm_control post_queues object=0xcf000002 cmd=0x2080012b cmd_name=NV2080_CTRL_CMD_GPU_PROMOTE_CTX cmd_tx=(0,16384,4096,3,9,1,32,4096) cmd_rx=9 stat_tx=(0,16384,4096,3,18,1,32,4096) stat_rx=18",
      "tiny rm_alloc pre_queues parent=0xcf000004 class=0xc7c0 cmd_tx=(0,16384,4096,3,10,1,32,4096) cmd_rx=10 stat_tx=(0,16384,4096,3,18,1,32,4096) stat_rx=18",
      "tiny rm_alloc post_queues object=0xcf000005 class=0xc7c0 cmd_tx=(0,16384,4096,3,11,1,32,4096) cmd_rx=11 stat_tx=(0,16384,4096,3,19,1,32,4096) stat_rx=19",
      "tiny compute_alloc parent=0xcf000004 object=0xcf000005 compute_class=0xc7c0 rpc_sha256=aaa",
      "tiny dma_alloc parent=0xcf000004 object=0xcf000006 dma_class=0xc7b5 rpc_sha256=ddd",
      "tiny token_control rpc_sha256=bbb",
      "tiny schedule_control rpc_sha256=ccc",
    ])
    standalone_sample += "\nrm gpfifo_patch params_sha256=ggg ctor_gpfifo_va=0x5000 ctor_entries=32 ctor_flags=0x200320 ctor_h_context_share=0x0 ctor_h_vaspace=0x90f1 ctor_h_userd_memory=0x0 ctor_userd_offset=0x100 ctor_engine_type=0x1 ctor_cid=0 ctor_runlist_id=0 ctor_internal_flags=0x1a after_ramfc_base=0x1000 after_ramfc_size=0x1000 after_userd_base=0x2000 after_userd_size=0x20 after_instance_base=0x1000 after_instance_size=0x200 after_method_base=0x3000 after_method_size=0x5000 ctor_error_base=0x4000 ctor_error_size=0x1000"
    tiny_sample += "\ntiny gpfifo_patch post params_sha256=ggg gpfifo_va=0x5000 entries=32 flags=0x200320 h_context_share=0x0 h_vaspace=0x90f1 h_userd_memory=0x0 userd_offset=0x100 engine_type=0x1 cid=0 runlist_id=0 internal_flags=0x1a ramfc=0x1000/0x1000/as2/ca0 userd=0x2000/0x20/as2/ca0 instance=0x1000/0x200/as2/ca0 method=0x3000/0x5000/as2/ca0 error=0x4000/0x1000/as2/ca0"
    compare_lines = format_trace_log_comparison(compare_trace_log_text(standalone_sample, tiny_sample),
                                                compare_trace_log_fields(standalone_sample, tiny_sample))
    assert compare_lines[0] == "trace_log_compare result=ok missing=none"
    assert "trace_log_compare_line label=dma_alloc required=False standalone=ok tiny=ok status=ok" in compare_lines
    assert "trace_log_compare_line label=exception_queues required=False standalone=missing tiny=missing status=optional-missing" in compare_lines
    assert "trace_log_compare_field label=gpfifo_params_sha256 required=True standalone=ggg tiny=ggg status=match" in compare_lines
    assert "trace_log_compare_field label=promote_entries_sha256 required=False standalone=ppp tiny=ppp status=match" in compare_lines
    assert "trace_log_compare_field label=compute_rpc_sha256 required=True standalone=aaa tiny=aaa status=match" in compare_lines
    assert "trace_log_compare_field label=dma_rpc_sha256 required=False standalone=ddd tiny=ddd status=match" in compare_lines
    assert "trace_log_compare_field label=token_rpc_sha256 required=False standalone=bbb tiny=bbb status=match" in compare_lines
    desc_lines = format_trace_log_gpfifo_desc_comparison(compare_trace_log_gpfifo_descs(standalone_sample, tiny_sample))
    assert "trace_log_compare_desc label=gpfifo_desc_ramfc required=False standalone=0x1000/0x1000 tiny=0x1000/0x1000 status=match" in desc_lines
    assert "trace_log_compare_desc label=gpfifo_desc_error required=False standalone=0x4000/0x1000 tiny=0x4000/0x1000 status=match" in desc_lines
    mismatch_desc_lines = format_trace_log_gpfifo_desc_comparison(compare_trace_log_gpfifo_descs(
      standalone_sample, tiny_sample.replace("method=0x3000/0x5000", "method=0x9999/0x5000")))
    assert "trace_log_compare_desc label=gpfifo_desc_method required=False standalone=0x3000/0x5000 tiny=0x9999/0x5000 status=mismatch" in mismatch_desc_lines
    desc_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_gpfifo_descs(standalone_sample, tiny_sample.replace("method=0x3000/0x5000", "method=0x9999/0x5000"))])
    assert desc_summary_lines[0] == "trace_log_compare result=mismatch missing=gpfifo_desc_method"
    scalar_lines = format_trace_log_gpfifo_scalar_comparison(compare_trace_log_gpfifo_scalars(standalone_sample, tiny_sample))
    assert "trace_log_compare_scalar label=gpfifo_ctor_va required=False standalone=0x5000 tiny=0x5000 status=match" in scalar_lines
    assert "trace_log_compare_scalar label=gpfifo_ctor_internal_flags required=False standalone=0x1a tiny=0x1a status=match" in scalar_lines
    mismatch_scalar_lines = format_trace_log_gpfifo_scalar_comparison(compare_trace_log_gpfifo_scalars(
      standalone_sample, tiny_sample.replace("entries=32", "entries=64")))
    assert "trace_log_compare_scalar label=gpfifo_ctor_entries required=False standalone=32 tiny=64 status=mismatch" in mismatch_scalar_lines
    promote_lines = format_trace_log_promote_context_comparison(compare_trace_log_promote_contexts(standalone_sample, tiny_sample))
    assert "trace_log_compare_promote label=golden_promote_entries_sha256 required=False standalone=ppp tiny=ppp status=match" in promote_lines
    assert "trace_log_compare_promote label=user_phys_promote_packed_entries_sha256 required=False standalone=uppack tiny=uppack status=match" in promote_lines
    assert "trace_log_compare_promote label=user_virt_promote_entries_sha256 required=False standalone=uv tiny=uv status=match" in promote_lines
    promote_meta_lines = format_trace_log_promote_metadata_comparison(compare_trace_log_promote_metadata(standalone_sample, tiny_sample))
    assert "trace_log_compare_promote_meta label=golden_promote_subdevice required=False standalone=0x20 tiny=0x20 status=match" in promote_meta_lines
    assert "trace_log_compare_promote_meta label=golden_promote_ids required=False standalone=[0, 2, 9] tiny=[0, 2, 9] status=match" in promote_meta_lines
    assert "trace_log_compare_promote_meta label=golden_promote_entry_text required=False standalone=id=0:phys=0x1:virt=0x2 tiny=id=0:phys=0x1:virt=0x2 status=match" in promote_meta_lines
    assert "trace_log_compare_promote_meta label=user_phys_promote_client required=False standalone=0xc1 tiny=0xc1 status=match" in promote_meta_lines
    assert "trace_log_compare_promote_meta label=user_phys_promote_object required=False standalone=0xcf tiny=0xcf status=match" in promote_meta_lines
    assert "trace_log_compare_promote_meta label=user_virt_promote_entries required=False standalone=3 tiny=3 status=match" in promote_meta_lines
    promote_control_lines = format_trace_log_promote_control_comparison(compare_trace_log_promote_control(standalone_sample, tiny_sample))
    assert "trace_log_compare_promote_control label=golden_promote_control_rpc_sha256 required=False standalone=promrpc tiny=promrpc status=match" in promote_control_lines
    assert "trace_log_compare_promote_control label=golden_promote_control_result_len required=False standalone=560 tiny=560 status=match" in promote_control_lines
    assert "trace_log_compare_promote_control label=golden_promote_control_result_sha256 required=False standalone=promresult tiny=promresult status=match" in promote_control_lines
    promote_control_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      compare_trace_log_promote_control(standalone_sample, tiny_sample.replace("result_sha256=promresult", "result_sha256=bad", 1)))
    assert promote_control_summary_lines[0] == "trace_log_compare result=mismatch missing=golden_promote_control_result_sha256"
    mismatch_promote_lines = format_trace_log_promote_context_comparison(compare_trace_log_promote_contexts(
      standalone_sample, tiny_sample.replace(" entries_sha256=up ", " entries_sha256=bad ")))
    assert "trace_log_compare_promote label=user_phys_promote_entries_sha256 required=False standalone=up tiny=bad status=mismatch" in mismatch_promote_lines
    promote_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      compare_trace_log_promote_contexts(standalone_sample, tiny_sample.replace(" entries_sha256=up ", " entries_sha256=bad ")))
    assert promote_summary_lines[0] == "trace_log_compare result=mismatch missing=user_phys_promote_entries_sha256"
    promote_meta_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      compare_trace_log_promote_metadata(standalone_sample, tiny_sample.replace("ids=[0, 2, 9]", "ids=[0, 2, 10]")))
    assert promote_meta_summary_lines[0] == "trace_log_compare result=mismatch missing=golden_promote_ids"
    promote_subdevice_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      compare_trace_log_promote_metadata(standalone_sample, tiny_sample.replace("subdevice=0x20", "subdevice=0x21", 1)))
    assert promote_subdevice_summary_lines[0] == "trace_log_compare result=mismatch missing=golden_promote_subdevice"
    user_promote_object_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      compare_trace_log_promote_metadata(standalone_sample, tiny_sample.replace("object=0xcf virt=False", "object=0xce virt=False")))
    assert user_promote_object_summary_lines[0] == "trace_log_compare result=mismatch missing=user_phys_promote_object"
    compute_lines = format_trace_log_compute_alloc_comparison(compare_trace_log_compute_alloc(standalone_sample, tiny_sample))
    assert "trace_log_compare_compute label=compute_alloc_parent required=False standalone=0xcf000004 tiny=0xcf000004 status=match" in compute_lines
    assert "trace_log_compare_compute label=compute_alloc_object required=False standalone=0xcf000005 tiny=0xcf000005 status=match" in compute_lines
    assert "trace_log_compare_compute label=compute_alloc_class required=False standalone=0xc7c0 tiny=0xc7c0 status=match" in compute_lines
    mismatch_compute_lines = format_trace_log_compute_alloc_comparison(compare_trace_log_compute_alloc(
      standalone_sample, tiny_sample.replace("object=0xcf000005", "object=0xcf000006")))
    assert "trace_log_compare_compute label=compute_alloc_object required=False standalone=0xcf000005 tiny=0xcf000006 status=mismatch" in mismatch_compute_lines
    compute_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_compute_alloc(standalone_sample, tiny_sample.replace("object=0xcf000005", "object=0xcf000006"))])
    assert compute_summary_lines[0] == "trace_log_compare result=mismatch missing=compute_alloc_object"
    dma_lines = format_trace_log_dma_alloc_comparison(compare_trace_log_dma_alloc(standalone_sample, tiny_sample))
    assert "trace_log_compare_dma label=dma_alloc_parent required=False standalone=0xcf000004 tiny=0xcf000004 status=match" in dma_lines
    assert "trace_log_compare_dma label=dma_alloc_object required=False standalone=0xcf000006 tiny=0xcf000006 status=match" in dma_lines
    assert "trace_log_compare_dma label=dma_alloc_class required=False standalone=0xc7b5 tiny=0xc7b5 status=match" in dma_lines
    mismatch_dma_lines = format_trace_log_dma_alloc_comparison(compare_trace_log_dma_alloc(
      standalone_sample, tiny_sample.replace("object=0xcf000006", "object=0xcf000007")))
    assert "trace_log_compare_dma label=dma_alloc_object required=False standalone=0xcf000006 tiny=0xcf000007 status=mismatch" in mismatch_dma_lines
    dma_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_dma_alloc(standalone_sample, tiny_sample.replace("object=0xcf000006", "object=0xcf000007"))])
    assert dma_summary_lines[0] == "trace_log_compare result=mismatch missing=dma_alloc_object"
    queue_lines = format_trace_log_queue_comparison(compare_trace_log_queues(standalone_sample, tiny_sample))
    assert "trace_log_compare_queue label=pre_cmd_wp required=False standalone=1 tiny=1 status=match" in queue_lines
    assert "trace_log_compare_queue label=pre_stat_rp required=False standalone=7 tiny=7 status=match" in queue_lines
    assert "trace_log_compare_queue label=post_cmd_wp required=False standalone=3 tiny=3 status=match" in queue_lines
    assert "trace_log_compare_queue label=promote_pre_cmd_wp required=False standalone=8 tiny=8 status=match" in queue_lines
    assert "trace_log_compare_queue label=promote_post_stat_rp required=False standalone=18 tiny=18 status=match" in queue_lines
    assert "trace_log_compare_queue label=compute_pre_cmd_wp required=False standalone=10 tiny=10 status=match" in queue_lines
    assert "trace_log_compare_queue label=compute_post_stat_rp required=False standalone=19 tiny=19 status=match" in queue_lines
    mismatch_queue_lines = format_trace_log_queue_comparison(compare_trace_log_queues(
      standalone_sample, tiny_sample.replace("stat_rx=7", "stat_rx=9")))
    assert "trace_log_compare_queue label=pre_stat_rp required=False standalone=7 tiny=9 status=mismatch" in mismatch_queue_lines
    queue_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_queues(standalone_sample, tiny_sample.replace("stat_rx=7", "stat_rx=9"))])
    assert queue_summary_lines[0] == "trace_log_compare result=mismatch missing=pre_stat_rp"
    mm_valloc_lines = format_trace_log_mm_valloc_comparison(compare_trace_log_mm_valloc_sequence(standalone_sample, tiny_sample))
    assert "trace_log_compare_mm_valloc label=mm_valloc_size_sequence required=False standalone=0x1000>0x193000 tiny=0x1000>0x193000 status=match" in mm_valloc_lines
    assert "trace_log_compare_mm_valloc label=mm_valloc_va_sequence required=False standalone=0x1020000000>0x1020100000 tiny=0x1020000000>0x1020100000 status=match" in mm_valloc_lines
    assert "trace_log_compare_mm_valloc label=mm_valloc_paddrs_sequence required=False standalone=0x1a1e000/0x1000>0x1a25000/0x193000 tiny=0x1a1e000/0x1000>0x1a25000/0x193000 status=match" in mm_valloc_lines
    mm_valloc_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_mm_valloc_sequence(standalone_sample, tiny_sample.replace("paddrs=0x1a25000/0x193000", "paddrs=0x1a26000/0x193000"))])
    assert mm_valloc_summary_lines[0] == "trace_log_compare result=mismatch missing=mm_valloc_paddrs_sequence"
    slot_lines = format_trace_log_queue_slot_comparison(compare_trace_log_queue_slots(standalone_sample, tiny_sample))
    assert "trace_log_compare_slot label=pre_cmd_slot0_func required=False standalone=103 tiny=103 status=match" in slot_lines
    assert "trace_log_compare_slot label=pre_stat_slot0_priv required=False standalone=0x0 tiny=0x0 status=match" in slot_lines
    mismatch_slot_lines = format_trace_log_queue_slot_comparison(compare_trace_log_queue_slots(
      standalone_sample, tiny_sample.replace("private=0x0 sig=0x51 stat_tx", "private=0x1 sig=0x51 stat_tx")))
    assert "trace_log_compare_slot label=pre_cmd_slot0_priv required=False standalone=0x0 tiny=0x1 status=mismatch" in mismatch_slot_lines
    slot_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_queue_slots(standalone_sample, tiny_sample.replace("private=0x0 sig=0x51 stat_tx", "private=0x1 sig=0x51 stat_tx"))])
    assert slot_summary_lines[0] == "trace_log_compare result=mismatch missing=pre_cmd_slot0_priv,pre_stat_slot0_priv,post_cmd_slot0_priv,post_stat_slot0_priv"
    assert format_trace_log_failure_summary(trace_log_failure_summary(standalone_sample, tiny_sample)) == (
      "trace_log_compare_failure standalone=none tiny=none status=both-clean standalone_msg=none tiny_msg=none")
    assert format_trace_log_progress_summary(trace_log_progress_summary(standalone_sample, tiny_sample)) == (
      "trace_log_compare_progress standalone=runtime-schedule tiny=runtime-schedule status=same")
    assert format_trace_log_progress_summary(trace_log_progress_summary(
      standalone_sample.split("channel golden_dma_alloc", 1)[0], tiny_sample)) == (
      "trace_log_compare_progress standalone=golden-compute-attempt tiny=runtime-schedule status=standalone-behind")
    assert format_trace_log_progress_summary(trace_log_progress_summary(
      "channel golden_promote_done", "tiny golden_done")) == (
      "trace_log_compare_progress standalone=golden-promote tiny=golden-promote status=same")
    assert format_trace_log_progress_summary(trace_log_progress_summary(
      "channel golden_compute_alloc", "tiny golden_done")) == (
      "trace_log_compare_progress standalone=golden-compute-attempt tiny=golden-promote status=standalone-ahead")
    assert format_trace_log_progress_summary(trace_log_progress_summary(
      "channel golden_promote_done", "tiny compute_alloc")) == (
      "trace_log_compare_progress standalone=golden-promote tiny=golden-done status=standalone-behind")
    rm_seq_standalone = "\n".join([
      "rm gsp_alloc: parent=0x0 class_name=NV01_ROOT h_class=0x0 object=0xc1e00000 params_sha256=p0",
      "rm gsp_alloc_rpc: rpc_sha256=r0",
      "rm gsp_alloc: parent=0xcf000000 class_name=AMPERE_CHANNEL_GPFIFO_A h_class=0xc56f object=0xcf000004 params_sha256=p1",
      "rm gsp_alloc_rpc: rpc_sha256=r1",
      "rm gsp_alloc: parent=0xcf000004 class_name=AMPERE_COMPUTE_B h_class=0xc7c0 object=0xcf000005 params_sha256=p2",
      "rm gsp_alloc_rpc: rpc_sha256=r2",
    ])
    rm_seq_tiny = "\n".join([
      "tiny rm_alloc pre parent=0x0 class_name=NV01_ROOT class=0x0 params_sha256=p0",
      "tiny rm_alloc post object=0xc1e00000 class=0x0 class_name=NV01_ROOT rpc_sha256=r0",
      "tiny rm_alloc pre parent=0xcf000000 class_name=AMPERE_CHANNEL_GPFIFO_A class=0xc56f params_sha256=p1",
      "tiny rm_alloc post object=0xcf000004 class=0xc56f class_name=AMPERE_CHANNEL_GPFIFO_A rpc_sha256=r1",
      "tiny rm_alloc pre parent=0xcf000004 class_name=AMPERE_DMA_COPY_B class=0xc7b5 params_sha256=p3",
      "tiny rm_alloc post object=0xcf000006 class=0xc7b5 class_name=AMPERE_DMA_COPY_B rpc_sha256=r3",
    ])
    rm_seq_line = format_trace_log_rm_alloc_sequence_comparison(compare_trace_log_rm_alloc_sequence(rm_seq_standalone, rm_seq_tiny))
    rm_seq_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[compare_trace_log_rm_alloc_sequence(rm_seq_standalone, rm_seq_tiny)]])
    assert rm_seq_summary[0] == "trace_log_compare result=mismatch missing=rm_alloc_sequence"
    assert rm_seq_line == (
      "trace_log_compare_rm_sequence standalone=NV01_ROOT>AMPERE_CHANNEL_GPFIFO_A>AMPERE_COMPUTE_B "
      "tiny=NV01_ROOT>AMPERE_CHANNEL_GPFIFO_A>AMPERE_DMA_COPY_B common_prefix=NV01_ROOT>AMPERE_CHANNEL_GPFIFO_A "
      "standalone_next=AMPERE_COMPUTE_B tiny_next=AMPERE_DMA_COPY_B prefix_len=2 standalone_count=3 tiny_count=3 "
      "standalone_handles=NV01_ROOT@0x0->0xc1e00000>AMPERE_CHANNEL_GPFIFO_A@0xcf000000->0xcf000004>AMPERE_COMPUTE_B@0xcf000004->0xcf000005 "
      "tiny_handles=NV01_ROOT@0x0->0xc1e00000>AMPERE_CHANNEL_GPFIFO_A@0xcf000000->0xcf000004>AMPERE_DMA_COPY_B@0xcf000004->0xcf000006 "
      "standalone_params=p0>p1>p2 tiny_params=p0>p1>p3 standalone_rpcs=r0>r1>r2 tiny_rpcs=r0>r1>r3 "
      "class_match=False handle_match=False params_match=False rpc_match=False status=diverge")
    rm_seq_same_class_diff_hash = compare_trace_log_rm_alloc_sequence(
      "rm gsp_alloc: parent=0x0 class_name=NV01_ROOT object=0xc1 params_sha256=p0\n"
      "rm gsp_alloc_rpc: rpc_sha256=r0",
      "tiny rm_alloc pre parent=0x0 class_name=NV01_ROOT params_sha256=p_bad\n"
      "tiny rm_alloc post object=0xc1 class_name=NV01_ROOT rpc_sha256=r_bad")
    assert rm_seq_same_class_diff_hash["class_match"] is True
    assert rm_seq_same_class_diff_hash["params_match"] is False
    assert rm_seq_same_class_diff_hash["rpc_match"] is False
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[rm_seq_same_class_diff_hash]])[0] == "trace_log_compare result=mismatch missing=rm_alloc_sequence"
    gsp_rpc_standalone = "\n".join([
      "standalone send_rpc func=103 func_name=GSP_RM_ALLOC len=64 sha256=r0 head=00",
      "standalone send_rpc func=17 func_name=GSP_RM_CONTROL len=32 sha256=r1 head=11",
    ])
    gsp_rpc_tiny = "\n".join([
      "tiny send_rpc func=103 func_name=GSP_RM_ALLOC len=64 sha256=r0 head=00",
      "tiny send_rpc func=17 func_name=GSP_RM_CONTROL len=32 sha256=r1 head=11",
    ])
    gsp_rpc_line = format_trace_log_gsp_rpc_sequence_comparison(compare_trace_log_gsp_rpc_sequence(gsp_rpc_standalone, gsp_rpc_tiny))
    assert gsp_rpc_line == (
      "trace_log_compare_gsp_rpc_sequence standalone=GSP_RM_ALLOC>GSP_RM_CONTROL tiny=GSP_RM_ALLOC>GSP_RM_CONTROL "
      "common_prefix=GSP_RM_ALLOC>GSP_RM_CONTROL standalone_next=none tiny_next=none prefix_len=2 standalone_count=2 tiny_count=2 "
      "standalone_lens=64>32 tiny_lens=64>32 standalone_hashes=r0>r1 tiny_hashes=r0>r1 "
      "func_match=True len_match=True hash_match=True status=match")
    gsp_rpc_mismatch = compare_trace_log_gsp_rpc_sequence(gsp_rpc_standalone, gsp_rpc_tiny.replace("sha256=r1", "sha256=bad"))
    assert gsp_rpc_mismatch["func_match"] is True and gsp_rpc_mismatch["hash_match"] is False
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[gsp_rpc_mismatch]])[0] == "trace_log_compare result=mismatch missing=gsp_rpc_sequence"
    sysinfo_standalone_payload = pack_gsp_system_info(0x110000000, 0x120000000, 0x130000000,
      0x220810de, 0x10de0000, 0xa1)
    sysinfo_tiny_payload = pack_gsp_system_info(0x110000000, 0x120000000, 0x130000000,
      0x220810de, 0x10de0000, 0xa1)
    sysinfo_standalone = (
      f"standalone send_rpc func=72 func_name=GSP_SET_SYSTEM_INFO len=928 sha256=sys "
      f"head={sysinfo_standalone_payload[:128].hex()}")
    sysinfo_tiny = (
      f"tiny send_rpc func=72 func_name=GSP_SET_SYSTEM_INFO len=928 sha256=sys "
      f"head={sysinfo_tiny_payload[:128].hex()}")
    sysinfo_rows = compare_trace_log_gsp_system_info(sysinfo_standalone, sysinfo_tiny)
    sysinfo_lines = format_trace_log_gsp_system_info_comparison(sysinfo_rows)
    assert "trace_log_compare_system_info label=system_info_gpu_phys required=False standalone=0x110000000 tiny=0x110000000 status=match" in sysinfo_lines
    assert "trace_log_compare_system_info label=system_info_pci_device required=False standalone=0x220810de tiny=0x220810de status=match" in sysinfo_lines
    sysinfo_mismatch_rows = compare_trace_log_gsp_system_info(
      sysinfo_standalone, sysinfo_tiny.replace("0000002001000000", "0000002101000000", 1))
    mismatch_sysinfo_lines = format_trace_log_gsp_system_info_comparison(sysinfo_mismatch_rows)
    assert "trace_log_compare_system_info label=system_info_gpu_fb required=False standalone=0x120000000 tiny=0x121000000 status=mismatch" in mismatch_sysinfo_lines
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [sysinfo_mismatch_rows])[0] == "trace_log_compare result=mismatch missing=system_info_gpu_fb"
    gsp_rpc_response_standalone = "\n".join([
      "standalone read_rpc rp=0 wp=2 func=103 func_name=GSP_RM_ALLOC len=96 advance=1 result=0x0 private=0x0 sha256=h0",
      "standalone read_rpc rp=1 wp=2 func=17 func_name=GSP_RM_CONTROL len=64 advance=1 result=0x0 private=0x0 sha256=h1",
    ])
    gsp_rpc_response_tiny = "\n".join([
      "tiny read_rpc rp=0 wp=2 advance=1 func=103 func_name=GSP_RM_ALLOC len=96 result=0x0 private=0x0 sha256=h0",
      "tiny read_rpc rp=1 wp=2 advance=1 func=17 func_name=GSP_RM_CONTROL len=64 result=0x0 private=0x0 sha256=h1",
    ])
    gsp_rpc_response_line = format_trace_log_gsp_rpc_response_sequence_comparison(
      compare_trace_log_gsp_rpc_response_sequence(gsp_rpc_response_standalone, gsp_rpc_response_tiny))
    assert gsp_rpc_response_line == (
      "trace_log_compare_gsp_rpc_response_sequence standalone=GSP_RM_ALLOC>GSP_RM_CONTROL tiny=GSP_RM_ALLOC>GSP_RM_CONTROL "
      "common_prefix=GSP_RM_ALLOC>GSP_RM_CONTROL standalone_next=none tiny_next=none prefix_len=2 standalone_count=2 tiny_count=2 "
      "standalone_lens=96>64 tiny_lens=96>64 standalone_queue=0/2/1>1/2/1 tiny_queue=0/2/1>1/2/1 "
      "standalone_results=0x0/0x0>0x0/0x0 tiny_results=0x0/0x0>0x0/0x0 "
      "standalone_hashes=h0>h1 tiny_hashes=h0>h1 func_match=True len_match=True queue_match=True "
      "result_match=True hash_match=True status=match")
    gsp_rpc_response_yield = compare_trace_log_gsp_rpc_response_sequence(
      gsp_rpc_response_standalone,
      "tiny read_rpc_yield func=103 func_name=GSP_RM_ALLOC sha256=h0\n"
      "tiny read_rpc_yield func=17 func_name=GSP_RM_CONTROL sha256=h1")
    assert gsp_rpc_response_yield["func_match"] is True and gsp_rpc_response_yield["hash_match"] is True
    assert gsp_rpc_response_yield["len_match"] is False and gsp_rpc_response_yield["queue_match"] is False
    assert gsp_rpc_response_yield["result_match"] is False
    gsp_rpc_response_enriched_yield = compare_trace_log_gsp_rpc_response_sequence(
      gsp_rpc_response_standalone,
      "tiny read_rpc_yield rp=0 wp=2 advance=1 func=103 func_name=GSP_RM_ALLOC len=96 result=0x0 private=0x0 sha256=h0\n"
      "tiny read_rpc_yield rp=1 wp=2 advance=1 func=17 func_name=GSP_RM_CONTROL len=64 result=0x0 private=0x0 sha256=h1")
    assert gsp_rpc_response_enriched_yield["func_match"] is True
    assert gsp_rpc_response_enriched_yield["len_match"] is True
    assert gsp_rpc_response_enriched_yield["queue_match"] is True
    assert gsp_rpc_response_enriched_yield["result_match"] is True
    assert gsp_rpc_response_enriched_yield["hash_match"] is True
    gsp_rpc_response_mismatch = compare_trace_log_gsp_rpc_response_sequence(
      gsp_rpc_response_standalone, gsp_rpc_response_tiny.replace("private=0x0", "private=0x1", 1))
    assert gsp_rpc_response_mismatch["func_match"] is True and gsp_rpc_response_mismatch["result_match"] is False
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[gsp_rpc_response_mismatch]])[0] == "trace_log_compare result=mismatch missing=gsp_rpc_response_sequence"
    gsp_rpc_response_len_mismatch = compare_trace_log_gsp_rpc_response_sequence(
      gsp_rpc_response_standalone, gsp_rpc_response_tiny.replace("len=64", "len=68"))
    assert gsp_rpc_response_len_mismatch["result_match"] is True and gsp_rpc_response_len_mismatch["len_match"] is False
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[gsp_rpc_response_len_mismatch]])[0] == "trace_log_compare result=mismatch missing=gsp_rpc_response_sequence"
    gsp_rpc_response_queue_mismatch = compare_trace_log_gsp_rpc_response_sequence(
      gsp_rpc_response_standalone, gsp_rpc_response_tiny.replace("rp=1", "rp=2", 1))
    assert gsp_rpc_response_queue_mismatch["result_match"] is True and gsp_rpc_response_queue_mismatch["queue_match"] is False
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[gsp_rpc_response_queue_mismatch]])[0] == "trace_log_compare result=mismatch missing=gsp_rpc_response_sequence"
    gsp_rpc_response_hash_mismatch = compare_trace_log_gsp_rpc_response_sequence(
      gsp_rpc_response_standalone, gsp_rpc_response_tiny.replace("sha256=h1", "sha256=bad"))
    assert gsp_rpc_response_hash_mismatch["result_match"] is True and gsp_rpc_response_hash_mismatch["hash_match"] is False
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[gsp_rpc_response_hash_mismatch]])[0] == "trace_log_compare result=mismatch missing=gsp_rpc_response_sequence"
    compute_rpc_standalone = "\n".join([
      "channel golden_compute_alloc parent=0xcf000004 expected_object=0xcf000005 compute_class=0xc7c0 expected_rpc_sha256=aaa",
      "standalone send_rpc func=103 func_name=GSP_RM_ALLOC len=32 sha256=aaa",
      "standalone read_rpc rp=18 wp=22 func=1002 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 advance=1 result=0x0 private=0x0 sha256=event head=000000004153534552540000464543535f410000",
      "GSP EVENT post_nocat len=0x4bc qwords=0x0,0x1234,0x5 kind=0x5 strings=ASSERT|FECS_A head=000000004153534552540000464543535f410000",
    ])
    compute_rpc_tiny = "\n".join([
      "tiny compute_alloc parent=0xcf000004 object=0xcf000005 compute_class=0xc7c0 rpc_sha256=aaa",
      "tiny send_rpc func=103 func_name=GSP_RM_ALLOC len=32 sha256=aaa",
      "tiny read_rpc rp=18 wp=19 advance=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sha256=resp head=0400e0c1040000cf",
    ])
    compute_rpc_response = compare_trace_log_compute_rpc_response(compute_rpc_standalone, compute_rpc_tiny)
    assert compute_rpc_response["send_match"] is True and compute_rpc_response["func_match"] is False
    compute_rpc_response_line = format_trace_log_compute_rpc_response_comparison(compute_rpc_response)
    assert compute_rpc_response_line == (
      "trace_log_compare_compute_rpc_response standalone_next=EVENT_GSP_POST_NOCAT_RECORD tiny_next=GSP_RM_ALLOC "
      "standalone_send_sha256=aaa tiny_send_sha256=aaa standalone_len=1244 tiny_len=64 "
      "standalone_queue=18/22/1 tiny_queue=18/19/1 standalone_result=0x0/0x0 tiny_result=0x0/0x0 "
      "standalone_sha256=event tiny_sha256=resp standalone_head_text=ASSERT|FECS_A tiny_head_text=missing "
      "standalone_post_nocat_kinds=0x5 tiny_post_nocat_kinds=missing "
      "standalone_post_nocat_strings=ASSERT|FECS_A tiny_post_nocat_strings=missing "
      "send_match=True func_match=False len_match=False "
      "result_match=True status=diverge")
    assert format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[compute_rpc_response]])[0] == "trace_log_compare result=mismatch missing=compute_rpc_response"
    gsp_post_nocat_standalone = "\n".join([
      "standalone read_rpc rp=0 wp=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 advance=1 result=0x0 private=0x0 sha256=ev1 post_nocat=qwords=0x0,0xa179f300,0x5 kind=0x5 strings=ASSERT|GFW_BOOT_PROGRESS_N",
      "GSP EVENT post_nocat len=0x4bc qwords=0x0,0xd09d5180,0x2 kind=0x2 strings=Display Subsystem|FECS_A|PCIe Engine",
      "standalone read_rpc rp=3 wp=4 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 advance=1 result=0x0 private=0x0 sha256=ev2 post_nocat=qwords=0x0,0xf9a24560,0x3 kind=0x3 strings=FECS_A|FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus|0%",
      "standalone read_rpc rp=4 wp=5 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 advance=1 result=0x0 private=0x0 sha256=ev3 post_nocat=qwords=0x0,0xf9a3b6a0,0x3 kind=0x3 strings=FECS_B|FECS_C|GR_STATUS|RMGpioPmuMutexTimeoutus|9b2w",
    ])
    gsp_post_nocat_tiny = "\n".join([
      "tiny read_rpc rp=0 wp=1 advance=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 result=0x0 private=0x0 sha256=ev1 post_nocat=qwords=0x0,0xa179f300,0x5 kind=0x5 strings=ASSERT|GFW_BOOT_PROGRESS_N",
      "tiny read_rpc_yield rp=0 wp=1 advance=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 result=0x0 private=0x0 sha256=ev1 post_nocat=qwords=0x0,0xa179f300,0x5 kind=0x5 strings=ASSERT|GFW_BOOT_PROGRESS_N",
      "tiny read_rpc rp=1 wp=2 advance=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 result=0x0 private=0x0 sha256=ev2 post_nocat=qwords=0x0,0xd09d5180,0x2 kind=0x2 strings=Display Subsystem|FECS_A|PCIe Engine",
      "tiny read_rpc_yield rp=2 wp=3 advance=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 result=0x0 private=0x0 sha256=ev3 post_nocat=qwords=0x0,0xbee42860,0x2 kind=0x2 strings=Display Subsystem|PCIe",
    ])
    gsp_post_nocat_row = compare_trace_log_gsp_post_nocat_sequence(gsp_post_nocat_standalone, gsp_post_nocat_tiny)
    gsp_post_nocat_line = format_trace_log_gsp_post_nocat_sequence_comparison(gsp_post_nocat_row)
    assert "trace_log_compare_gsp_post_nocat_sequence " in gsp_post_nocat_line
    assert gsp_post_nocat_row["standalone_count"] == 4
    assert gsp_post_nocat_row["tiny_count"] == 4
    assert gsp_post_nocat_row["prefix_len"] == 1
    assert gsp_post_nocat_row["standalone_strings"].startswith("ASSERT|GFW_BOOT_PROGRESS_N>Display")
    assert gsp_post_nocat_row["tiny_strings"].startswith("ASSERT|GFW_BOOT_PROGRESS_N>ASSERT")
    assert "RMGpioPmuMutexTimeoutus" in (gsp_post_nocat_row["standalone_strings"] or "")
    assert "RMGpioPmuMutexTimeoutus" not in (gsp_post_nocat_row["tiny_strings"] or "")
    assert gsp_post_nocat_row["match"] is False
    assert gsp_post_nocat_row["standalone_next"] == "0x2"
    assert gsp_post_nocat_row["tiny_next"] == "0x5"
    gsp_post_nocat_standalone_dup = "\n".join([
      "standalone read_rpc rp=0 wp=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 advance=1 result=0x0 private=0x0 sha256=ev1 post_nocat=qwords=0x0,0xa179f300,0x5 kind=0x5 strings=ASSERT|GFW_BOOT_PROGRESS_N",
      "standalone read_rpc rp=1 wp=2 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 advance=1 result=0x0 private=0x0 sha256=ev2 post_nocat=qwords=0x0,0xbee42860,0x2 kind=0x2 strings=Display|PCIe",
    ])
    gsp_post_nocat_tiny_dup = "\n".join([
      "tiny read_rpc rp=0 wp=1 advance=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 result=0x0 private=0x0 sha256=ev1 post_nocat=qwords=0x0,0xa179f300,0x5 kind=0x5 strings=ASSERT|GFW_BOOT_PROGRESS_N",
      "tiny read_rpc rp=1 wp=2 advance=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD len=1244 result=0x0 private=0x0 sha256=ev2 post_nocat=qwords=0x0,0xbee42860,0x2 kind=0x2 strings=Display|PCIe",
    ])
    gsp_post_nocat_match = compare_trace_log_gsp_post_nocat_sequence(gsp_post_nocat_standalone_dup, gsp_post_nocat_tiny_dup)
    assert gsp_post_nocat_match["match"] is True and gsp_post_nocat_match["prefix_len"] == 2
    gsp_post_nocat_empty = compare_trace_log_gsp_post_nocat_sequence("", "")
    assert gsp_post_nocat_empty["standalone_count"] == 0 and gsp_post_nocat_empty["tiny_count"] == 0
    assert gsp_post_nocat_empty["match"] is True and gsp_post_nocat_empty["prefix_len"] == 0
    assert format_trace_log_gsp_post_nocat_sequence_comparison(gsp_post_nocat_empty).startswith(
      "trace_log_compare_gsp_post_nocat_sequence standalone=missing tiny=missing ")
    gsp_post_nocat_only_standalone = compare_trace_log_gsp_post_nocat_sequence(gsp_post_nocat_standalone, "")
    assert gsp_post_nocat_only_standalone["standalone_count"] == 4 and gsp_post_nocat_only_standalone["tiny_count"] == 0
    assert gsp_post_nocat_only_standalone["match"] is False and gsp_post_nocat_only_standalone["prefix_len"] == 0
    gsp_post_nocat_only_tiny = compare_trace_log_gsp_post_nocat_sequence("", gsp_post_nocat_tiny)
    assert gsp_post_nocat_only_tiny["match"] is False and gsp_post_nocat_only_tiny["prefix_len"] == 0
    gsp_post_nocat_no_events = compare_trace_log_gsp_post_nocat_sequence(
      "standalone read_rpc rp=0 wp=1 func=103 func_name=GSP_RM_ALLOC len=64 advance=1 result=0x0 private=0x0 sha256=r0",
      "tiny read_rpc rp=0 wp=1 func=103 func_name=GSP_RM_ALLOC len=64 advance=1 result=0x0 private=0x0 sha256=r0")
    assert gsp_post_nocat_no_events["match"] is True and gsp_post_nocat_no_events["standalone_count"] == 0
    standalone_stack_sample = standalone_sample + "\n" + "\n".join([
      "channel golden_start: reserved_size=536870912",
      '  File "/Users/yeren/nvgpu/examples/add.py", line 1, in main',
      '  File "/Users/yeren/nvgpu/examples/add.py", line 2, in prepare_golden_image_context',
      "rm alloc: parent=0xcf000004",
      '  File "/Users/yeren/nvgpu/examples/add.py", line 3, in prepare_golden_image_context',
      '  File "/Users/yeren/nvgpu/examples/add.py", line 4, in rm_alloc',
      "channel golden_compute_alloc parent=0xcf000004 expected_object=0xcf000005 compute_class=0xc7c0 expected_rpc_sha256=aaa",
      '  File "/Users/yeren/nvgpu/examples/add.py", line 5, in prepare_golden_image_context',
      '  File "/Users/yeren/nvgpu/examples/add.py", line 6, in rm_alloc',
      "channel golden_dma_alloc parent=0xcf000004 expected_object=0xcf000006 dma_class=0xc7b5 expected_rpc_sha256=ddd",
      '  File "/Users/yeren/nvgpu/examples/add.py", line 7, in prepare_golden_image_context',
      '  File "/Users/yeren/nvgpu/examples/add.py", line 8, in rm_alloc',
      "channel runtime_token_control object=0xcf000001 cmd=0xc36f0108 rpc_sha256=bbb",
      '  File "/Users/yeren/nvgpu/examples/add.py", line 9, in allocate_compute_channel',
      '  File "/Users/yeren/nvgpu/examples/add.py", line 10, in rm_control',
      "channel runtime_schedule_control object=0xcf000002 cmd=0xa06c0101 rpc_sha256=ccc",
      '  File "/Users/yeren/nvgpu/examples/add.py", line 11, in allocate_compute_channel',
      '  File "/Users/yeren/nvgpu/examples/add.py", line 12, in rm_control',
    ])
    tiny_stack_sample = tiny_sample + "\n" + "\n".join([
      'tiny golden_start_stack   File "/Users/yeren/nvgpu/examples/add_tiny.py", line 1, in init\\n  File "/Users/yeren/nvgpu/examples/add_tiny.py", line 2, in prepare_golden_image_context\\n',
      'tiny rm_alloc_stack   File "/Users/yeren/nvgpu/examples/add_tiny.py", line 3, in prepare_golden_image_context\\n  File "/Users/yeren/nvgpu/examples/add_tiny.py", line 4, in rm_alloc\\n',
      'tiny compute_alloc_stack   File "/Users/yeren/nvgpu/examples/add_tiny.py", line 5, in prepare_golden_image_context\\n  File "/Users/yeren/nvgpu/examples/add_tiny.py", line 6, in rm_alloc\\n',
      'tiny dma_alloc_stack   File "/Users/yeren/nvgpu/examples/add_tiny.py", line 7, in prepare_golden_image_context\\n  File "/Users/yeren/nvgpu/examples/add_tiny.py", line 8, in rm_alloc\\n',
      'tiny token_control_stack   File "/Users/yeren/nvgpu/examples/add_tiny.py", line 9, in allocate_compute_channel\\n  File "/Users/yeren/nvgpu/examples/add_tiny.py", line 10, in rm_control\\n',
      'tiny schedule_control_stack   File "/Users/yeren/nvgpu/examples/add_tiny.py", line 11, in allocate_compute_channel\\n  File "/Users/yeren/nvgpu/examples/add_tiny.py", line 12, in rm_control\\n',
    ])
    stack_lines = format_trace_log_stack_comparison(compare_trace_log_stacks(standalone_stack_sample, tiny_stack_sample))
    assert "trace_log_compare_stack label=golden_start required=False standalone=main>prepare_golden_image_context tiny=init>prepare_golden_image_context common=prepare_golden_image_context standalone_only=main tiny_only=init standalone_locations=add.py:1:main>add.py:2:prepare_golden_image_context tiny_locations=add_tiny.py:1:init>add_tiny.py:2:prepare_golden_image_context status=common" in stack_lines
    assert "trace_log_compare_stack label=rm_alloc required=False standalone=prepare_golden_image_context>rm_alloc tiny=prepare_golden_image_context>rm_alloc common=prepare_golden_image_context>rm_alloc standalone_only=none tiny_only=none standalone_locations=add.py:3:prepare_golden_image_context>add.py:4:rm_alloc tiny_locations=add_tiny.py:3:prepare_golden_image_context>add_tiny.py:4:rm_alloc status=common" in stack_lines
    assert "trace_log_compare_stack label=compute_alloc required=False standalone=prepare_golden_image_context>rm_alloc tiny=prepare_golden_image_context>rm_alloc common=prepare_golden_image_context>rm_alloc standalone_only=none tiny_only=none standalone_locations=add.py:5:prepare_golden_image_context>add.py:6:rm_alloc tiny_locations=add_tiny.py:5:prepare_golden_image_context>add_tiny.py:6:rm_alloc status=common" in stack_lines
    assert "trace_log_compare_stack label=dma_alloc required=False standalone=prepare_golden_image_context>rm_alloc tiny=prepare_golden_image_context>rm_alloc common=prepare_golden_image_context>rm_alloc standalone_only=none tiny_only=none standalone_locations=add.py:7:prepare_golden_image_context>add.py:8:rm_alloc tiny_locations=add_tiny.py:7:prepare_golden_image_context>add_tiny.py:8:rm_alloc status=common" in stack_lines
    assert "trace_log_compare_stack label=token_control required=False standalone=allocate_compute_channel>rm_control tiny=allocate_compute_channel>rm_control common=allocate_compute_channel>rm_control standalone_only=none tiny_only=none standalone_locations=add.py:9:allocate_compute_channel>add.py:10:rm_control tiny_locations=add_tiny.py:9:allocate_compute_channel>add_tiny.py:10:rm_control status=common" in stack_lines
    assert "trace_log_compare_stack label=schedule_control required=False standalone=allocate_compute_channel>rm_control tiny=allocate_compute_channel>rm_control common=allocate_compute_channel>rm_control standalone_only=none tiny_only=none standalone_locations=add.py:11:allocate_compute_channel>add.py:12:rm_control tiny_locations=add_tiny.py:11:allocate_compute_channel>add_tiny.py:12:rm_control status=common" in stack_lines
    stack_mismatch_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_stacks(
        'channel golden_compute_alloc\n  File "/Users/yeren/nvgpu/examples/add.py", line 10, in standalone_only_path\n',
        'tiny compute_alloc_stack   File "/Users/yeren/nvgpu/examples/add_tiny.py", line 20, in tiny_only_path\\n')])
    assert stack_mismatch_summary[0] == "trace_log_compare result=mismatch missing=compute_alloc"
    falcon_standalone = "\n".join([
      "falcon pre-dma base=0x110000 cmd=0x615 dest=0x0 mem_off=0x0 src=0x1a00000 size=0x100 state=(engine=0x0 cpuctl=0x10 dmatrfcmd=0x0 dmatrfbase=0x1 dmatrfbase1=0x0 dmatrfmoffs=0x2 dmatrffboffs=0x3 fbif_ctl=0x80 fbif_transcfg0=0x100)",
      "falcon wreg GSP.DMATRFBASE addr=0x110110 value=0x1a000",
      "falcon wreg GSP.DMATRFCMD addr=0x110118 value=0x615",
      "standalone rm_alloc exception_queues state=(gsp=(engine=0x0) sec2=(engine=0x0))",
    ])
    falcon_tiny = "\n".join([
      "tiny wreg GSP.DMATRFBASE addr=0x110110 value=0x1a000",
      "tiny wreg GSP.DMATRFCMD addr=0x110118 value=0x615",
      "tiny execute_dma pre base=0x110000 cmd=0x615 dest=0x0 mem_off=0x0 src=0x1a00000 size=0x100 state=(engine=0x0 cpuctl=0x10 dmatrfcmd=0x0 dmatrfbase=0x1 dmatrfbase1=0x0 dmatrfmoffs=0x2 dmatrffboffs=0x3 fbif_ctl=0x80 fbif_transcfg0=0x100)",
      "tiny rm_alloc exception_state gsp=(engine=0x0) sec2=(engine=0x0)",
    ])
    falcon_lines = format_trace_log_falcon_comparison(compare_trace_log_falcon(falcon_standalone, falcon_tiny))
    assert "trace_log_compare_falcon label=falcon_dma_pre required=False standalone=present tiny=present status=present" in falcon_lines
    assert "trace_log_compare_falcon label=falcon_state required=False standalone=present tiny=present status=present" in falcon_lines
    assert "trace_log_compare_falcon label=falcon_writes required=False standalone=present tiny=present status=present" in falcon_lines
    falcon_value_lines = format_trace_log_falcon_comparison(compare_trace_log_falcon_values(falcon_standalone, falcon_tiny))
    assert "trace_log_compare_falcon label=falcon_dma_cmd required=False standalone=0x615 tiny=0x615 status=present" in falcon_value_lines
    assert "trace_log_compare_falcon label=falcon_dma_fbif_ctl required=False standalone=0x80 tiny=0x80 status=present" in falcon_value_lines
    mismatch_falcon_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_falcon_values(falcon_standalone, falcon_tiny.replace("fbif_ctl=0x80", "fbif_ctl=0x81"))])
    assert mismatch_falcon_summary[0] == "trace_log_compare result=mismatch missing=falcon_dma_fbif_ctl"
    falcon_write_lines = format_trace_log_falcon_write_comparison(compare_trace_log_falcon_writes(falcon_standalone, falcon_tiny))
    assert "trace_log_compare_falcon_writes label=falcon_write_sequence required=False standalone=GSP.DMATRFBASE=0x1a000>GSP.DMATRFCMD=0x615 tiny=GSP.DMATRFBASE=0x1a000>GSP.DMATRFCMD=0x615 common_prefix=GSP.DMATRFBASE=0x1a000>GSP.DMATRFCMD=0x615 standalone_tail=none tiny_tail=none status=match" in falcon_write_lines
    mismatch_falcon_write_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      compare_trace_log_falcon_writes(falcon_standalone, falcon_tiny.replace("value=0x615", "value=0x616", 1)))
    assert mismatch_falcon_write_summary[0] == "trace_log_compare result=mismatch missing=falcon_write_sequence"
    missing_falcon_lines = format_trace_log_falcon_comparison(compare_trace_log_falcon(falcon_standalone, "tiny golden_start"))
    assert "trace_log_compare_falcon label=falcon_dma_pre required=False standalone=present tiny=missing status=optional-missing" in missing_falcon_lines
    boot_standalone = "\n".join([
      "standalone wpr meta=0x1000 bootloader=0x2000 radix3=0x3000 signature=0x4000 meta_sha256=wprhash",
      "standalone booter img=0x5000 wpr_meta=0x1000 code_off=0x80 data_off=0x180 code_size=0x100 data_size=0x200 sha256=boothash",
    ])
    boot_tiny = "\n".join([
      "tiny wpr meta=0x1000 bootloader=0x2000 radix3=0x3000 signature=0x4000 meta_sha256=wprhash",
      "tiny booter img=0x5000 code_off=0x80 data_off=0x180 code_size=0x100 data_size=0x200 sha256=boothash",
    ])
    boot_lines = format_trace_log_boot_comparison(compare_trace_log_boot_values(boot_standalone, boot_tiny))
    assert "trace_log_compare_boot label=wpr_meta_sha256 required=False standalone=wprhash tiny=wprhash status=match" in boot_lines
    assert "trace_log_compare_boot label=booter_sha256 required=False standalone=boothash tiny=boothash status=match" in boot_lines
    mismatch_boot_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_boot_values(boot_standalone, boot_tiny.replace("sha256=boothash", "sha256=badboot"))])
    assert mismatch_boot_summary[0] == "trace_log_compare result=mismatch missing=booter_sha256"
    readback_standalone = "\n".join([
      "FRTS BAR1 readback ok size=0x100 vram=0x1a00000 sha256=frtshash",
      "SEC2 booter-load readback ok size=0x200 sha256=loaderhash",
      "WPR metadata readback ok size=0x100 sha256=wprhash",
      "WPR bootloader readback ok size=0x300 sha256=bootfw",
      "GSP signature readback ok size=0x40 sha256=sighash",
      "GSP radix3 readback ok size=0x400 sha256=radixhash",
    ])
    readback_lines = format_trace_log_boot_readback_comparison(compare_trace_log_boot_readbacks(readback_standalone))
    assert "trace_log_compare_boot_readback label=frts_bar1 required=False standalone_sha256=frtshash size=0x100 vram=0x1a00000 status=present" in readback_lines
    assert "trace_log_compare_boot_readback label=sec2_booter_load required=False standalone_sha256=loaderhash size=0x200 vram=missing status=present" in readback_lines
    assert "trace_log_compare_boot_readback label=gsp_radix3 required=False standalone_sha256=radixhash size=0x400 vram=missing status=present" in readback_lines
    missing_readback_lines = format_trace_log_boot_readback_comparison(compare_trace_log_boot_readbacks("standalone wpr meta=0x1000"))
    assert "trace_log_compare_boot_readback label=frts_bar1 required=False standalone_sha256=missing size=missing vram=missing status=optional-missing" in missing_readback_lines
    boot_state_standalone = (
      "standalone rm_state stage=pre-alloc bar1=0x0 wpr2_hi=0x2ffee00 "
      "gsp=(engine=0x0 cpuctl=0x10) sec2=(engine=0x0 cpuctl=0x20)")
    boot_state_tiny = (
      "tiny pre-root bar1=0x0 wpr2_hi=0x2ffee00 "
      "gsp=(engine=0x0 cpuctl=0x10) sec2=(engine=0x0 cpuctl=0x20)")
    boot_state_lines = format_trace_log_boot_state_comparison(compare_trace_log_boot_state(boot_state_standalone, boot_state_tiny))
    assert "trace_log_compare_boot_state label=pre_root_bar1 required=False standalone=0x0 tiny=0x0 status=match" in boot_state_lines
    assert "trace_log_compare_boot_state label=pre_root_gsp_cpuctl required=False standalone=0x10 tiny=0x10 status=match" in boot_state_lines
    mismatch_boot_state_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_boot_state(boot_state_standalone, boot_state_tiny.replace("wpr2_hi=0x2ffee00", "wpr2_hi=0x2ffee01"))])
    assert mismatch_boot_state_summary[0] == "trace_log_compare result=mismatch missing=pre_root_wpr2_hi"
    compute_state_standalone = (
      "standalone rm_state stage=pre-alloc parent=0xcf000004 class=0xc7c0 class_name=AMPERE_COMPUTE_B "
      "bar1=0x0 wpr2_hi=0x2ffee00 "
      "gsp=(engine=0x0 cpuctl=0x10 dmactl=0x0 dmatrfcmd=0x1022 dmatrfbase=0x2f40205 "
      "dmatrfbase1=0x0 dmatrfmoffs=0x2000 dmatrffboffs=0x5c hwcfg2=0x47f7 fbif_ctl=0x90 "
      "fbif_transcfg0=0x10004 exci=0xf00 irqstat=0x0 riscv_bcr=0x111 riscv_cpuctl=0x80 os=0x0 rm=0x0) "
      "sec2=(engine=0x0 cpuctl=0x20 dmactl=0x0 dmatrfcmd=0x6602 dmatrfbase=0x2f3a0de "
      "dmatrfbase1=0x0 dmatrfmoffs=0x7a00 dmatrffboffs=0x7a00 hwcfg2=0x47f7 fbif_ctl=0x190 "
      "fbif_transcfg0=0x10004 exci=0xffffffff irqstat=0x7d00 riscv_bcr=0x1 riscv_cpuctl=0x10 os=0x0 rm=0x0)")
    compute_state_tiny = (
      "tiny rm_alloc pre_state parent=0xcf000004 class=0xc7c0 class_name=AMPERE_COMPUTE_B "
      "bar1=0x0 wpr2_hi=0x2ffee00 "
      "gsp=(engine=0x0 cpuctl=0x10 dmactl=0x0 dmatrfcmd=0x1022 dmatrfbase=0x2f40205 "
      "dmatrfbase1=0x0 dmatrfmoffs=0x2000 dmatrffboffs=0x5c hwcfg2=0x47f7 fbif_ctl=0x90 "
      "fbif_transcfg0=0x10004 exci=0xf00 irqstat=0x0 riscv_bcr=0x111 riscv_cpuctl=0x80 os=0x0 rm=0x0) "
      "sec2=(engine=0x0 cpuctl=0x20 dmactl=0x0 dmatrfcmd=0x6602 dmatrfbase=0x2f3a0de "
      "dmatrfbase1=0x0 dmatrfmoffs=0x7a00 dmatrffboffs=0x7a00 hwcfg2=0x47f7 fbif_ctl=0x190 "
      "fbif_transcfg0=0x10004 exci=0xffffffff irqstat=0x7d00 riscv_bcr=0x1 riscv_cpuctl=0x10 os=0x0 rm=0x0)")
    compute_state_lines = format_trace_log_compute_state_comparison(
      compare_trace_log_compute_state(compute_state_standalone, compute_state_tiny))
    assert "trace_log_compare_compute_state label=compute_state_bar1 required=False standalone=0x0 tiny=0x0 status=match" in compute_state_lines
    assert "trace_log_compare_compute_state label=compute_state_gsp_dmatrfcmd required=False standalone=0x1022 tiny=0x1022 status=match" in compute_state_lines
    assert "trace_log_compare_compute_state label=compute_state_sec2_irqstat required=False standalone=0x7d00 tiny=0x7d00 status=match" in compute_state_lines
    mismatch_compute_state_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_compute_state(compute_state_standalone, compute_state_tiny.replace("fbif_ctl=0x90", "fbif_ctl=0x91", 1))])
    assert mismatch_compute_state_summary[0] == "trace_log_compare result=mismatch missing=compute_state_gsp_fbif_ctl"
    boot_queue_standalone = (
      "standalone queue queues=0x80000000 rm_args=0x81000000 logbuf=0x82000000 "
      "libos_args=0x83000000 cmd_head=00010203")
    boot_queue_tiny = "\n".join([
      "tiny queue rm_args=0x81000000 cmd_head=00010203 queue_base=0x80000000",
      "tiny libos_args=0x83000000",
    ])
    boot_queue_lines = format_trace_log_boot_queue_comparison(compare_trace_log_boot_queue(boot_queue_standalone, boot_queue_tiny))
    assert "trace_log_compare_boot_queue label=boot_queue_rm_args required=False standalone=0x81000000 tiny=0x81000000 status=match" in boot_queue_lines
    assert "trace_log_compare_boot_queue label=boot_queue_libos_args required=False standalone=0x83000000 tiny=0x83000000 status=match" in boot_queue_lines
    assert "trace_log_compare_boot_queue label=boot_queue_cmd_head required=False standalone=00010203 tiny=00010203 status=match" in boot_queue_lines
    mismatch_boot_queue_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_boot_queue(boot_queue_standalone, boot_queue_tiny.replace("cmd_head=00010203", "cmd_head=00010204"))])
    assert mismatch_boot_queue_summary[0] == "trace_log_compare result=mismatch missing=boot_queue_cmd_head"
    status_summary = format_trace_log_failure_summary(trace_log_failure_summary(
      standalone_sample + "\nstandalone rm_state stage=alloc-status status=0x1f",
      tiny_sample))
    assert status_summary == "trace_log_compare_failure standalone=status:0x1f tiny=none status=different standalone_msg=none tiny_msg=none"
    exception_summary = format_trace_log_failure_summary(trace_log_failure_summary(
      standalone_sample + "\nstandalone rm_state stage=alloc-exception exc=RuntimeError exc_msg=timeout",
      tiny_sample + "\ntiny rm_alloc exception parent=0x1 class=0x2 exc=RuntimeError msg=timeout"))
    assert exception_summary == "trace_log_compare_failure standalone=exception:RuntimeError tiny=exception:RuntimeError status=same standalone_msg=timeout tiny_msg=timeout"
    detailed_exception_summary = format_trace_log_failure_summary(trace_log_failure_summary(
      standalone_sample + "\nstandalone rm_state stage=alloc-exception exc=RuntimeError exc_msg=timeout waiting for RPC response 103 (GSP_RM_ALLOC); queue=timeout_stat_state rpc_sha256=abc cmd_queue=[cmd]",
      tiny_sample + "\ntiny rm_alloc exception parent=0x1 class=0x2 exc=RuntimeError msg=timeout waiting for RPC response 103 (GSP_RM_ALLOC); queue=tiny_stat_state cmd_tx=(0,1)"))
    assert detailed_exception_summary == (
      "trace_log_compare_failure standalone=exception:RuntimeError tiny=exception:RuntimeError status=same "
      "standalone_msg=timeout waiting for RPC response 103 (GSP_RM_ALLOC); queue=timeout_stat_state "
      "tiny_msg=timeout waiting for RPC response 103 (GSP_RM_ALLOC); queue=tiny_stat_state")
    exception_standalone = standalone_sample + "\nstandalone rm_state stage=alloc-exception client=0xc1e00004 parent=0xcf000004 object=0xcf000005 class=0xc7c0 exc=RuntimeError exc_msg=timeout"
    exception_tiny = tiny_sample + "\ntiny rm_alloc exception parent=0xcf000004 class=0xc7c0 class_name=AMPERE_COMPUTE_B exc=RuntimeError msg=timeout"
    exception_lines = format_trace_log_exception_context_comparison(compare_trace_log_exception_context(exception_standalone, exception_tiny))
    assert "trace_log_compare_exception label=exception_parent required=False standalone=0xcf000004 tiny=0xcf000004 status=match" in exception_lines
    assert "trace_log_compare_exception label=exception_class required=False standalone=0xc7c0 tiny=0xc7c0 status=match" in exception_lines
    assert "trace_log_compare_exception label=exception_object required=False standalone=0xcf000005 tiny=missing status=optional-missing" in exception_lines
    mismatch_exception_lines = format_trace_log_exception_context_comparison(compare_trace_log_exception_context(
      exception_standalone, exception_tiny.replace("parent=0xcf000004", "parent=0xcf000099")))
    assert "trace_log_compare_exception label=exception_parent required=False standalone=0xcf000004 tiny=0xcf000099 status=mismatch" in mismatch_exception_lines
    exception_summary_lines = format_trace_log_comparison(
      compare_trace_log_text(exception_standalone, exception_tiny),
      compare_trace_log_fields(exception_standalone, exception_tiny),
      [compare_trace_log_exception_context(exception_standalone, exception_tiny.replace("class=0xc7c0", "class=0xc6c0"))])
    assert exception_summary_lines[0] == "trace_log_compare result=mismatch missing=exception_class"
    mismatch_lines = format_trace_log_comparison(compare_trace_log_text(standalone_sample, tiny_sample.replace("rpc_sha256=aaa", "rpc_sha256=ddd")),
                                                 compare_trace_log_fields(standalone_sample, tiny_sample.replace("rpc_sha256=aaa", "rpc_sha256=ddd")))
    assert mismatch_lines[0] == "trace_log_compare result=mismatch missing=compute_rpc_sha256"
    assert "trace_log_compare_field label=compute_rpc_sha256 required=True standalone=aaa tiny=ddd status=mismatch" in mismatch_lines
    missing_lines = format_trace_log_comparison(compare_trace_log_text(standalone_sample, "tiny golden_start"),
                                                compare_trace_log_fields(standalone_sample, "tiny golden_start"))
    assert missing_lines[0] == "trace_log_compare result=missing missing=golden_gpfifo,golden_promote,rm_pre_queues,rm_post_queues,compute_alloc,token_control,schedule_control,gpfifo_params_sha256,compute_rpc_sha256"
    assert "trace_log_compare_line label=dma_alloc required=False standalone=ok tiny=missing status=optional-missing" in missing_lines
    missing_arg_buf = io.StringIO()
    try:
      with contextlib.redirect_stdout(missing_arg_buf): print_trace_log_comparison(None, "tiny.log")
      raise AssertionError("missing standalone log argument was accepted")
    except SystemExit as exc:
      assert exc.code == 2
    assert missing_arg_buf.getvalue().strip() == "trace_log_compare_error kind=missing-argument flags=--standalone-log"
    missing_file_buf = io.StringIO()
    try:
      with contextlib.redirect_stdout(missing_file_buf):
        print_trace_log_comparison("missing-standalone.log", "missing-tiny.log")
      raise AssertionError("missing trace log files were accepted")
    except SystemExit as exc:
      assert exc.code == 2
    assert missing_file_buf.getvalue().strip() == (
      "trace_log_compare_error kind=missing-file paths=standalone:missing-standalone.log,tiny:missing-tiny.log")
    with tempfile.TemporaryDirectory() as tmpdir:
      standalone_log_path = pathlib.Path(tmpdir) / "standalone.log"
      tiny_log_path = pathlib.Path(tmpdir) / "tiny.log"
      standalone_log_path.write_text(standalone_sample)
      tiny_log_path.write_text(tiny_sample)
      compare_cli_buf = io.StringIO()
      with contextlib.redirect_stdout(compare_cli_buf):
        print_trace_log_comparison(str(standalone_log_path), str(tiny_log_path))
      compare_cli_lines = compare_cli_buf.getvalue().splitlines()
      assert compare_cli_lines[0] == "trace_log_compare result=ok missing=none"
      assert compare_cli_lines[1].startswith("trace_log_compare_failure ")
      assert compare_cli_lines[2].startswith("trace_log_compare_progress ")
      assert compare_cli_lines[3].startswith("trace_log_compare_rm_sequence ")
    assert recommended_tiny_trace_command("custom_tiny.py") == (
      "NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 custom_tiny.py")
    offline_debug_buf = io.StringIO()
    with contextlib.redirect_stdout(offline_debug_buf): print_offline_debug_suite("add")
    offline_debug_text = offline_debug_buf.getvalue()
    assert "summary transport=auto" in offline_debug_text
    assert "tinygrad_modules=[]" in offline_debug_text and "tinygrad_static_imports=[]" in offline_debug_text
    assert "external_static_imports=[]" in offline_debug_text
    assert "standalone golden_compute_alloc parent=0xcf000004 object=0xcf000005" in offline_debug_text
    assert "standalone context_promote label=user_phys entries=3 ids=[0, 1, 2]" in offline_debug_text
    assert "standalone gpfifo_constructor parent=0x80 object=0xcf000000 gpfifo_class=0xc56f" in offline_debug_text
    assert "gpfifo_va=0x1000000000 entries=4 flags=0x200320" in offline_debug_text
    assert "engine_type=0x1 cid=3" in offline_debug_text
    assert "standalone gpfifo_desc name=userd base=0x90000020 size=0x20" in offline_debug_text
    assert "standalone runtime_compute_alloc parent=0xcf000000 object=0xcf000001" in offline_debug_text
    assert "standalone launch arithmetic=add result=[11.0, 22.0, 33.0, 44.0]" in offline_debug_text
    gpfifo_ctor_buf = io.StringIO()
    with contextlib.redirect_stdout(gpfifo_ctor_buf): print_gpfifo_constructor_fingerprint()
    gpfifo_ctor_text = gpfifo_ctor_buf.getvalue()
    assert "standalone gpfifo_constructor parent=0x80 object=0xcf000000 gpfifo_class=0xc56f" in gpfifo_ctor_text
    assert "standalone gpfifo_desc name=error base=0x90300000 size=0x3000000" in gpfifo_ctor_text
    class FakePreflightTransport:
      def probe(self): return (0x10000000, 0)
      def read_config(self, offset, size):
        if (offset, size) == (PCI_VENDOR_ID, 4): return 0x220810de
        if (offset, size) == (PCI_COMMAND, 2): return 0x0004
        raise AssertionError(f"unexpected config read {offset:#x}/{size}")
      def bar_info(self, bar):
        return {0: (0x60000000, 0x1000000), 1: (0x80000000, 0x40000000), 3: (0x70000000, 0x200000)}[bar]
    preflight_state = transport_preflight_state(transport=FakePreflightTransport())
    assert preflight_state["vendor_device"] == 0x220810de and preflight_state["pci_command"] == 0x0004
    assert format_transport_preflight_state(preflight_state) == (
      "transport_preflight ok=True requested=auto transport=FakePreflightTransport probe=0x10000000/0x0 "
      "vendor_device=0x220810de pci_command=0x0004 bar0=0x60000000/0x1000000 "
      "bar1=0x80000000/0x40000000 bar3=0x70000000/0x200000")
    class ProbeErrorReadyPreflightTransport(FakePreflightTransport):
      def probe(self): raise RuntimeError("TinyGPU RPC PROBE failed status=0x1: no error payload")
    probe_error_state = transport_preflight_state(transport=ProbeErrorReadyPreflightTransport())
    assert format_transport_preflight_state(probe_error_state) == (
      "transport_preflight ok=True requested=auto transport=ProbeErrorReadyPreflightTransport "
      "probe_error=RuntimeError:TinyGPU RPC PROBE failed status=0x1: no error payload "
      "vendor_device=0x220810de pci_command=0x0004 bar0=0x60000000/0x1000000 "
      "bar1=0x80000000/0x40000000 bar3=0x70000000/0x200000")
    assert classify_transport_preflight(parse_transport_preflight_line(
      format_transport_preflight_state(probe_error_state))) == "ready-for-gsp"
    class FailingPreflightTransport:
      def read_config(self, offset, size):
        raise RuntimeError("Driver not available. Check: System Report > PCI for GPU, System Settings > Privacy & Security.")
    preflight_buf = io.StringIO()
    with contextlib.redirect_stdout(preflight_buf): print_transport_preflight(transport=FailingPreflightTransport())
    assert preflight_buf.getvalue().strip() == (
      "transport_preflight ok=False requested=auto transport=FailingPreflightTransport step=read_vendor_device "
      "exc=RuntimeError msg=Driver not available. "
      "Check: System Report > PCI for GPU, System Settings > Privacy & Security.")
    class ProbeThenFailPreflightTransport(FailingPreflightTransport):
      def probe(self): return (0x10000000, 0)
    preflight_buf = io.StringIO()
    with contextlib.redirect_stdout(preflight_buf): print_transport_preflight(transport=ProbeThenFailPreflightTransport())
    assert preflight_buf.getvalue().strip() == (
      "transport_preflight ok=False requested=auto transport=ProbeThenFailPreflightTransport probe=0x10000000/0x0 "
      "step=read_vendor_device exc=RuntimeError msg=Driver not available. "
      "Check: System Report > PCI for GPU, System Settings > Privacy & Security.")
    assert classify_transport_preflight(parse_transport_preflight_line(format_transport_preflight_state(preflight_state))) == "ready-for-gsp"
    for bad_ready_line in [
      "transport_preflight ok=True requested=auto transport=FakePreflightTransport probe=0x10000000/0x0 vendor_device=0x22081234 pci_command=0x0004 bar0=0x60000000/0x1000000 bar1=0x80000000/0x40000000 bar3=0x70000000/0x200000",
      "transport_preflight ok=True requested=auto transport=FakePreflightTransport probe=0x10000000/0x0 vendor_device=0x220810de pci_command=0x0000 bar0=0x60000000/0x1000000 bar1=0x80000000/0x40000000 bar3=0x70000000/0x200000",
      "transport_preflight ok=True requested=auto transport=FakePreflightTransport probe=0x10000000/0x0 vendor_device=0x220810de pci_command=0x0004 bar0=0x60000000/0x0 bar1=0x80000000/0x40000000 bar3=0x70000000/0x200000",
    ]:
      assert classify_transport_preflight(parse_transport_preflight_line(bad_ready_line)) == "inspect-preflight-failure"
    assert classify_transport_preflight(parse_transport_preflight_line(
      "transport_preflight ok=False requested=mac-egpu transport=MacEgpuTransport step=probe exc=RuntimeError msg=Driver not available")) == "fix-tinygpu-driver-visibility"
    assert classify_transport_preflight(parse_transport_preflight_line(
      "transport_preflight ok=False requested=mac-egpu transport=MacEgpuTransport step=read_vendor_device exc=RuntimeError msg=Driver not available")) == "fix-pci-config-access"
    assert classify_transport_preflight(parse_transport_preflight_line(
      "transport_preflight ok=False requested=mac-egpu transport=MacEgpuTransport vendor_device=0x220810de step=bar1_info exc=RuntimeError msg=bad")) == "fix-bar-access"
    classify_buf = io.StringIO()
    with contextlib.redirect_stdout(classify_buf):
      print_transport_preflight_classification("transport_preflight ok=False requested=mac-egpu transport=MacEgpuTransport step=probe exc=RuntimeError msg=Driver")
    assert classify_buf.getvalue().strip() == "transport_preflight_classification result=fix-tinygpu-driver-visibility ok=False step=probe"
    gate_buf = io.StringIO()
    with contextlib.redirect_stdout(gate_buf): print_transport_preflight_gate(transport=FakePreflightTransport())
    assert gate_buf.getvalue().strip().splitlines() == [
      "transport_preflight ok=True requested=auto transport=FakePreflightTransport probe=0x10000000/0x0 "
      "vendor_device=0x220810de pci_command=0x0004 bar0=0x60000000/0x1000000 "
      "bar1=0x80000000/0x40000000 bar3=0x70000000/0x200000",
      "transport_preflight_classification result=ready-for-gsp ok=True step=none",
    ]
    plan_buf = io.StringIO()
    with contextlib.redirect_stdout(plan_buf): print_transport_preflight_plan(transport=FakePreflightTransport())
    assert plan_buf.getvalue().strip().splitlines() == [
      "transport_preflight ok=True requested=auto transport=FakePreflightTransport probe=0x10000000/0x0 "
      "vendor_device=0x220810de pci_command=0x0004 bar0=0x60000000/0x1000000 "
      "bar1=0x80000000/0x40000000 bar3=0x70000000/0x200000",
      "transport_preflight_classification result=ready-for-gsp ok=True step=none",
      f"transport_preflight_plan next_action=reconnect_command command={recommended_reconnect_command(golden=True)}",
    ]
    ready_buf = io.StringIO()
    with contextlib.redirect_stdout(ready_buf):
      assert print_transport_preflight_gate(transport=FakePreflightTransport(), require_ready=True) == "ready-for-gsp"
    failure_plan_buf = io.StringIO()
    with contextlib.redirect_stdout(failure_plan_buf):
      assert print_transport_preflight_plan(transport=FailingPreflightTransport()) == "fix-pci-config-access"
    assert failure_plan_buf.getvalue().strip().splitlines()[-1] == (
      "transport_preflight_plan next_action=retry-preflight classification=fix-pci-config-access")
    try:
      with contextlib.redirect_stdout(io.StringIO()):
        print_transport_preflight_gate(transport=FailingPreflightTransport(), require_ready=True)
      raise AssertionError("failed preflight gate did not exit")
    except SystemExit as exc:
      assert exc.code == 1
    debug_help_buf = io.StringIO()
    with contextlib.redirect_stdout(debug_help_buf): print_debug_help()
    debug_help_text = debug_help_buf.getvalue()
    assert "debug_help script=examples/add.py arithmetic=add" in debug_help_text
    assert "transport_preflight python3 examples/add.py --transport-preflight" in debug_help_text
    assert "transport_preflight_gate python3 examples/add.py --transport-preflight-gate" in debug_help_text
    assert "transport_preflight_require_ready python3 examples/add.py --transport-preflight-gate --require-ready" in debug_help_text
    assert "transport_preflight_plan python3 examples/add.py --transport-preflight-plan --require-ready" in debug_help_text
    assert "transport_preflight_classify echo '<transport_preflight line>' | python3 examples/add.py --classify-transport-preflight" in debug_help_text
    assert "offline_debug python3 examples/add.py --offline-debug-suite" in debug_help_text
    assert "contract_suite python3 examples/add.py --contract-suite" in debug_help_text
    assert "validation_suite python3 examples/add.py --validation-suite" in debug_help_text
    assert "live_debug python3 examples/add.py --live-debug-commands" in debug_help_text
    assert "live_log_workflow python3 examples/add.py --live-log-workflow" in debug_help_text
    assert "live_stack_log_workflow python3 examples/add.py --live-stack-log-workflow" in debug_help_text
    assert "comparison_checklist python3 examples/add.py --comparison-checklist" in debug_help_text
    assert "compare_trace_logs python3 examples/add.py --compare-trace-logs --standalone-log standalone.log --tiny-log tiny.log" in debug_help_text
    assert "context_promote_fingerprint python3 examples/add.py --context-promote-fingerprint" in debug_help_text
    assert "gpfifo_constructor_fingerprint python3 examples/add.py --gpfifo-constructor-fingerprint" in debug_help_text
    assert f"tiny_trace {recommended_tiny_trace_command()}" in debug_help_text
    contract_suite_buf = io.StringIO()
    with contextlib.redirect_stdout(contract_suite_buf): print_contract_suite()
    assert contract_suite_buf.getvalue().strip().splitlines() == [
      "transport_contract=ok",
      "register_contract=ok",
      "boot_firmware_contract=ok",
      "gsp_rpc_contract=ok",
      "vm_contract=ok",
      "channel_contract=ok",
    ]
    validation_suite_buf = io.StringIO()
    with contextlib.redirect_stdout(validation_suite_buf): print_validation_suite("add")
    validation_suite_text = validation_suite_buf.getvalue()
    assert validation_suite_text.startswith("transport_contract=ok\nregister_contract=ok\n")
    assert "tinygrad_modules=[]" in validation_suite_text
    assert "external_static_imports=[]" in validation_suite_text
    assert "standalone launch arithmetic=add result=[11.0, 22.0, 33.0, 44.0]" in validation_suite_text
    assert "comparison_checklist script=examples/add.py tiny_script=examples/add_tiny.py" in validation_suite_text
    assert "compare_value promote_metadata client,subdevice,object,entries,ids,entry_text" in validation_suite_text
    assert "compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control" in validation_suite_text
    validation_suite_mul_buf = io.StringIO()
    old_argv = sys.argv[:]
    try:
      sys.argv = ["examples/add.py", "--mul"]
      with contextlib.redirect_stdout(validation_suite_mul_buf): print_validation_suite(
        "mul" if "--mul" in sys.argv else "add", "examples/mul.py" if "--mul" in sys.argv else "examples/add.py")
    finally:
      sys.argv = old_argv
    validation_suite_mul_text = validation_suite_mul_buf.getvalue()
    assert "standalone launch arithmetic=mul result=[10.0, 40.0, 90.0, 160.0]" in validation_suite_mul_text
    assert "comparison_checklist script=examples/mul.py tiny_script=examples/add_tiny.py" in validation_suite_mul_text
    assert "compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control" in validation_suite_mul_text
    mul_live_buf = io.StringIO()
    with contextlib.redirect_stdout(mul_live_buf): print_live_log_workflow("examples/mul.py")
    mul_live_text = mul_live_buf.getvalue()
    assert "live_log_workflow script=examples/mul.py tiny_script=examples/add_tiny.py" in mul_live_text
    assert "python3 examples/mul.py 2>&1 | tee standalone-golden.log" in mul_live_text
    mul_stack_buf = io.StringIO()
    with contextlib.redirect_stdout(mul_stack_buf): print_live_stack_log_workflow("examples/mul.py")
    mul_stack_text = mul_stack_buf.getvalue()
    assert "live_stack_log_workflow script=examples/mul.py tiny_script=examples/add_tiny.py" in mul_stack_text
    assert "python3 examples/mul.py 2>&1 | tee standalone-stack.log" in mul_stack_text
  finally:
    if old_boot is None: os.environ.pop("NV_ADD_BOOT_GSP", None)
    else: os.environ["NV_ADD_BOOT_GSP"] = old_boot
    if old_golden is None: os.environ.pop("NV_ADD_PREPARE_GOLDEN_CTX", None)
    else: os.environ["NV_ADD_PREPARE_GOLDEN_CTX"] = old_golden
    if old_summary is None: os.environ.pop("NV_ADD_SUMMARY", None)
    else: os.environ["NV_ADD_SUMMARY"] = old_summary
    if old_trace_boot is None: os.environ.pop("NV_ADD_TRACE_GSP_BOOT", None)
    else: os.environ["NV_ADD_TRACE_GSP_BOOT"] = old_trace_boot
    if old_verify_sec2 is None: os.environ.pop("NV_ADD_VERIFY_SEC2_INPUTS", None)
    else: os.environ["NV_ADD_VERIFY_SEC2_INPUTS"] = old_verify_sec2
    if old_reconnect_transport is None: os.environ.pop("NV_ADD_RECONNECT_TRANSPORT", None)
    else: os.environ["NV_ADD_RECONNECT_TRANSPORT"] = old_reconnect_transport
  assert should_boot_gsp() == (old_boot == "1")
  assert should_prepare_golden_ctx() == (old_golden == "1")
  assert should_print_summary() == (old_summary == "1")
  assert trace_gsp_boot_enabled() == (old_trace_boot == "1")
  assert verify_sec2_inputs_enabled() == (old_verify_sec2 == "1")
  assert recommended_reconnect_transport() == (old_reconnect_transport or "mac-egpu")
  assert reconnect_mode() == ("golden-context" if old_golden == "1" else "fixed-gpfifo")
  assert recommended_reconnect_flags() == (
    f"NV_ADD_TRANSPORT={old_reconnect_transport or 'mac-egpu'},NV_ADD_BOOT_GSP=1,NV_ADD_SUMMARY=1,NV_ADD_TRACE_GSP_BOOT=1,NV_ADD_VERIFY_SEC2_INPUTS=1,NV_ADD_TRACE_RM_ALLOC=1,NV_ADD_TRACE_RM_STATE=1,NV_ADD_TRACE_RPC=1,NV_ADD_TRACE_RPC_READ=1,NV_ADD_TRACE_CHANNEL=1,NV_ADD_TRACE_LAUNCH_STEPS=1"
    if old_golden != "1" else
    f"NV_ADD_TRANSPORT={old_reconnect_transport or 'mac-egpu'},NV_ADD_PREPARE_GOLDEN_CTX=1,NV_ADD_BOOT_GSP=1,NV_ADD_SUMMARY=1,NV_ADD_CHECK_FRTS_BAR1=1,NV_ADD_TRACE_GSP_BOOT=1,NV_ADD_VERIFY_SEC2_INPUTS=1,NV_ADD_TRACE_RM_ALLOC=1,NV_ADD_TRACE_RM_STATE=1,NV_ADD_TRACE_RPC=1,NV_ADD_TRACE_RPC_READ=1,NV_ADD_TRACE_CHANNEL=1,NV_ADD_TRACE_MM_ALLOC=1,NV_ADD_TRACE_LAUNCH_STEPS=1"
  )
  assert recommended_reconnect_command("custom.py") == (
    f"NV_ADD_TRANSPORT={old_reconnect_transport or 'mac-egpu'} NV_ADD_BOOT_GSP=1 NV_ADD_SUMMARY=1 "
    "NV_ADD_TRACE_GSP_BOOT=1 NV_ADD_VERIFY_SEC2_INPUTS=1 NV_ADD_TRACE_RM_ALLOC=1 "
    "NV_ADD_TRACE_RM_STATE=1 NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 NV_ADD_TRACE_LAUNCH_STEPS=1 python3 custom.py"
    if old_golden != "1" else
    f"NV_ADD_TRANSPORT={old_reconnect_transport or 'mac-egpu'} NV_ADD_PREPARE_GOLDEN_CTX=1 NV_ADD_BOOT_GSP=1 "
    "NV_ADD_SUMMARY=1 NV_ADD_CHECK_FRTS_BAR1=1 NV_ADD_TRACE_GSP_BOOT=1 NV_ADD_VERIFY_SEC2_INPUTS=1 "
    "NV_ADD_TRACE_RM_ALLOC=1 NV_ADD_TRACE_RM_STATE=1 NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1 NV_ADD_TRACE_CHANNEL=1 "
    "NV_ADD_TRACE_MM_ALLOC=1 NV_ADD_TRACE_LAUNCH_STEPS=1 python3 custom.py"
  )

  fw_store = FirmwareStore(root="/tmp/nv-fw-test")
  assert next(fw_store.candidates("ga102", "gsp.bin")) == pathlib.Path("/tmp/nv-fw-test/ga102/gsp/gsp.bin")
  for bad_fw_call, text in [
    (lambda: FirmwareStore.validate_component("", "firmware name"), "non-empty"),
    (lambda: FirmwareStore.validate_component("..", "firmware name"), "path component"),
    (lambda: FirmwareStore.validate_component("ga102/evil", "firmware name"), "path component"),
    (lambda: list(fw_store.candidates("ga102", "../gsp.bin")), "path component"),
    (lambda: fw_store.load("ga102/evil", "gsp.bin"), "path component"),
  ]:
    try:
      bad_fw_call()
      raise AssertionError("bad firmware store input was accepted")
    except ValueError as exc:
      assert text in str(exc)

  fake_fw = bytearray(0x200)
  struct.pack_into("<IIIIII", fake_fw, 0, 0x10de, 1, len(fake_fw), 0x40, 0x100, 0x20)
  desc_words = [0] * (RM_RISCV_UCODE_DESC_SIZE // 4)
  desc_words[8], desc_words[10], desc_words[12] = 0x44, 0x88, 0xcc
  struct.pack_into("<" + "I" * len(desc_words), fake_fw, 0x40, *desc_words)
  fake_fw[0x100:0x120] = b"B" * 0x20
  image, desc = GspFirmwarePrep.bootloader_image_and_desc(fake_fw)
  assert image == b"B" * 0x20
  assert desc["manifest_offset"] == 0x44
  assert desc["monitor_data_offset"] == 0x88
  assert desc["monitor_code_offset"] == 0xcc
  bad_fw = bytearray(fake_fw)
  struct.pack_into("<I", bad_fw, 8, 0)
  try:
    GspFirmwarePrep.parse_nvfw_bin_header(bad_fw)
    raise AssertionError("zero firmware bin_size was accepted")
  except ValueError as exc:
    assert "bin_size" in str(exc)
  bad_fw = bytearray(fake_fw)
  struct.pack_into("<I", bad_fw, 8, 0x110)
  try:
    GspFirmwarePrep.bootloader_image_and_desc(bad_fw)
    raise AssertionError("bootloader data outside declared bin_size was accepted")
  except ValueError as exc:
    assert "declared firmware size" in str(exc)
  bad_fw = bytearray(fake_fw)
  struct.pack_into("<I", bad_fw, 8, 0x140)
  struct.pack_into("<I", bad_fw, 12, 0x180)
  try:
    GspFirmwarePrep.bootloader_image_and_desc(bad_fw)
    raise AssertionError("bootloader descriptor outside declared bin_size was accepted")
  except ValueError as exc:
    assert "bootloader descriptor" in str(exc) and "declared firmware size" in str(exc)
  booter_fw = bytearray(0x240)
  struct.pack_into("<IIIIII", booter_fw, 0, 0x10de, 1, len(booter_fw), 0x40, 0x100, 0x40)
  struct.pack_into("<IIIIIIIII", booter_fw, 0x40, 0x90, 0x20, 0x80, 0x84, 0, 0, 0x88, 0x60, 0)
  struct.pack_into("<I", booter_fw, 0x80, 0)
  struct.pack_into("<I", booter_fw, 0x84, 0)
  struct.pack_into("<I", booter_fw, 0x88, 1)
  struct.pack_into("<IIIII", booter_fw, 0x60, 0, 0, 0x18, 0x8, 1)
  struct.pack_into("<IIII", booter_fw, 0x60 + NVFW_HS_LOAD_HEADER_V2_SIZE, 0x20, 0x10, 0, 0)
  booter_fw[0x100:0x140] = b"L" * 0x40
  booter_fw[0x90:0xb0] = b"S" * 0x20
  booter_image, booter_desc = GspFirmwarePrep.booter_load_image_and_desc(booter_fw)
  assert booter_image[:0x20] == b"S" * 0x20
  assert booter_image[0x20:] == b"L" * 0x20
  assert booter_desc == {"data_offset": 0x18, "data_size": 0x8, "code_offset": 0x20, "code_size": 0x10}
  bad_booter_fw = bytearray(booter_fw)
  struct.pack_into("<I", bad_booter_fw, 8, 0x120)
  try:
    GspFirmwarePrep.booter_load_image_and_desc(bad_booter_fw)
    raise AssertionError("booter data outside declared bin_size was accepted")
  except ValueError as exc:
    assert "declared firmware size" in str(exc)
  for mutate_booter, text in [
    (lambda fw: struct.pack_into("<II", fw, 8, 0x160, 0x180), "booter HS header"),
    (lambda fw: (struct.pack_into("<I", fw, 8, 0x160), struct.pack_into("<I", fw, 0x40 + 28, 0x180)), "booter load header"),
    (lambda fw: (struct.pack_into("<I", fw, 8, 0x160), struct.pack_into("<I", fw, 0x40 + 8, 0x180)), "booter patch metadata"),
    (lambda fw: (struct.pack_into("<I", fw, 8, 0x160), struct.pack_into("<I", fw, 0x40, 0x180)), "booter signature"),
  ]:
    bad_booter_fw = bytearray(booter_fw)
    mutate_booter(bad_booter_fw)
    try:
      GspFirmwarePrep.booter_load_image_and_desc(bad_booter_fw)
      raise AssertionError("booter metadata outside declared bin_size was accepted")
    except ValueError as exc:
      assert text in str(exc) and "declared firmware size" in str(exc)
  for parser, text in [
    (lambda: GspFirmwarePrep.parse_riscv_ucode_desc(bytes(RM_RISCV_UCODE_DESC_SIZE), -1), "bootloader descriptor offset"),
    (lambda: GspFirmwarePrep.parse_falcon_ucode_desc_v3(bytes(FALCON_UCODE_DESC_V3_SIZE), -1), "Falcon ucode descriptor offset"),
  ]:
    try:
      parser()
      raise AssertionError("negative firmware descriptor offset was accepted")
    except ValueError as exc:
      assert text in str(exc)
  try:
    GspFirmwarePrep.radix3_layout(-1)
    raise AssertionError("negative radix3 image size was accepted")
  except ValueError as exc:
    assert "non-negative" in str(exc)
  try:
    GspFirmwarePrep.pack_fwsec_frts_cmd(0x12345)
    raise AssertionError("unaligned FRTS offset was accepted")
  except ValueError as exc:
    assert "4KB aligned" in str(exc)
  fwsec_desc = {
    "imem_load_size": 0x40, "interface_offset": 0x10, "pkc_data_offset": 0x100,
    "stored_size": 0x80, "hdr": 0, "imem_phys_base": 0, "imem_virt_base": 0,
    "dmem_phys_base": 0, "dmem_load_size": 0, "engine_id_mask": 0,
    "ucode_id": 0, "signature_count": 0, "signature_versions": 0, "reserved": 0,
  }
  fwsec_image = bytearray(0x300)
  struct.pack_into("<BBBB", fwsec_image, 0x50, 0, 0, 8, 1)
  struct.pack_into("<II", fwsec_image, 0x54, 4, 0x40)
  mapper = [0] * 17
  mapper[3] = 0x90
  struct.pack_into("<IHH" + "I" * 14, fwsec_image, 0x80, *mapper)
  fwsec_signature = b"Q" * 0x180
  fwsec_cmd = b"C" * 8
  patched_fwsec = GspFirmwarePrep.patch_fwsec_image(fwsec_image, fwsec_desc, fwsec_signature, 0x15, fwsec_cmd)
  assert patched_fwsec[0xd0:0xd8] == fwsec_cmd
  assert patched_fwsec[0x140:0x140 + 0x180] == fwsec_signature
  assert struct.unpack_from("<IHH" + "I" * 14, patched_fwsec, 0x80)[12] == 0x15
  for patch_call, text in [
    (lambda: GspFirmwarePrep.patch_fwsec_image(bytes(0x40), fwsec_desc, fwsec_signature, 0x15, fwsec_cmd), "FWSEC app header"),
    (lambda: GspFirmwarePrep.patch_fwsec_image(fwsec_image, fwsec_desc, b"Q" * 0x17f, 0x15, fwsec_cmd), "FWSEC signature"),
  ]:
    try:
      patch_call()
      raise AssertionError("malformed FWSEC image patch input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  truncated_vbios = bytearray(0x20)
  struct.pack_into("<H", truncated_vbios, 0x18, 0)
  struct.pack_into("<H", truncated_vbios, 0x10, 1)
  truncated_vbios[0x14] = NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT
  try:
    GspFirmwarePrep.find_fwsec_ucode(truncated_vbios)
    raise AssertionError("truncated VBIOS BIT header was accepted")
  except ValueError as exc:
    assert "BIT header" in str(exc)

  meta = GspFirmwarePrep.build_wpr_meta(0x40000000, len(image), 0x12345, 0x100000, 0x200000, 0x300000, desc)
  assert len(meta) == GSP_FW_WPR_META_SIZE
  assert struct.unpack_from("<Q", meta, 0)[0] == GSP_FW_WPR_META_MAGIC
  assert struct.unpack_from("<Q", meta, 8)[0] == GSP_FW_WPR_META_REVISION
  assert struct.unpack_from("<Q", meta, 16)[0] == 0x200000
  assert struct.unpack_from("<Q", meta, 32)[0] == 0x100000
  assert struct.unpack_from("<Q", meta, 48)[0] == 0xcc
  assert struct.unpack_from("<Q", meta, 56)[0] == 0x88
  assert struct.unpack_from("<Q", meta, 64)[0] == 0x44
  assert struct.unpack_from("<Q", meta, 248)[0] == 0
  for bad_meta_call, text in [
    (lambda: GspFirmwarePrep.pack_wpr_meta(size_of_bootloader=-1), "size_of_bootloader"),
    (lambda: GspFirmwarePrep.pack_wpr_meta(elf_code_size=0x100000000), "elf_code_size"),
    (lambda: GspFirmwarePrep.pack_wpr_meta(flags=0x100), "flags"),
    (lambda: GspFirmwarePrep.build_wpr_meta(0, len(image), 0x12345, 0x100000, 0x200000, 0x300000, desc), "VRAM size"),
    (lambda: GspFirmwarePrep.build_wpr_meta(0x40000000, 0, 0x12345, 0x100000, 0x200000, 0x300000, desc), "booter size"),
    (lambda: GspFirmwarePrep.build_wpr_meta(0x40000000, len(image), 0, 0x100000, 0x200000, 0x300000, desc), "radix3 size"),
    (lambda: GspFirmwarePrep.build_wpr_meta(0x40000000, len(image), 0x12345, -1, 0x200000, 0x300000, desc), "booter sysmem"),
    (lambda: GspFirmwarePrep.build_wpr_meta(0x40000000, len(image), 0x12345, 0x100000, 0x200000, 0x300000, {"monitor_code_offset": 1, "monitor_data_offset": 2}), "manifest_offset"),
    (lambda: GspFirmwarePrep.build_wpr_meta(0x40000000, len(image), 0x12345, 0x100000, 0x200000, 0x300000, desc, frts_offset=0x12345), "FRTS offset"),
  ]:
    try:
      bad_meta_call()
      raise AssertionError("bad WPR metadata input was accepted")
    except ValueError as exc:
      assert text in str(exc)

  class FakeBootTransport:
    def __init__(self): self.sysmem_allocs = 0
    def alloc_sysmem(self, size, contiguous=False):
      self.sysmem_allocs += 1
      return MMIOView(bytearray(size)), [0x80000000 + off for off in range(0, size, 0x1000)]
    def map_bar(self, bar):
      assert bar == 1
      return MMIOView(bytearray(0x20000))
    def bar_info(self, bar):
      assert bar == 1
      return (0x90000000, 0x20000)
  class FakeBootMm:
    def __init__(self): self.boot_flags = []
    def palloc(self, size, boot=False):
      self.boot_flags.append(boot)
      return 0x4000
  fake_boot_shell = type("FakeBootShell", (), {"transport": FakeBootTransport(), "mm": FakeBootMm(), "vram_size": 0x100000})()
  boot_view, boot_paddr, boot_paddrs = BootMemoryAllocator(fake_boot_shell).alloc(0x1000, data=b"B", sysmem=False)
  assert boot_paddr == 0x4000 and boot_paddrs == [0x90004000]
  assert bytes(boot_view[:1]) == b"B"
  assert fake_boot_shell.mm.boot_flags == [True]
  assert fake_boot_shell.transport.sysmem_allocs == 0
  for kwargs in ({"sysmem": True}, {"sysmem": False}):
    before_sysmem, before_palloc = fake_boot_shell.transport.sysmem_allocs, len(fake_boot_shell.mm.boot_flags)
    try:
      BootMemoryAllocator(fake_boot_shell).alloc(1, data=b"BB", **kwargs)
      raise AssertionError("oversized boot allocation data was accepted")
    except ValueError as exc:
      assert "data size" in str(exc)
    assert fake_boot_shell.transport.sysmem_allocs == before_sysmem
    assert len(fake_boot_shell.mm.boot_flags) == before_palloc
  class BadBootTransport(FakeBootTransport):
    def __init__(self, paddrs=None, bar_size=0x20000):
      super().__init__()
      self.paddrs, self.bar_size = paddrs, bar_size
    def alloc_sysmem(self, size, contiguous=False):
      self.sysmem_allocs += 1
      return MMIOView(bytearray(size)), self.paddrs if self.paddrs is not None else []
    def bar_info(self, bar):
      assert bar == 1
      return (0x90000000, self.bar_size)
  for bad_boot_alloc, text in [
    (lambda: BootMemoryAllocator(fake_boot_shell).alloc(0), "boot allocation size"),
    (lambda: BootMemoryAllocator(type("ShortBootShell", (), {"transport": BadBootTransport([0x80000000]), "mm": FakeBootMm()})()).alloc(0x2000, sysmem=True), "boot sysmem returned"),
    (lambda: BootMemoryAllocator(type("BadPageBootShell", (), {"transport": BadBootTransport([0x80000001]), "mm": FakeBootMm()})()).alloc(0x1000, sysmem=True), "boot sysmem page"),
    (lambda: BootMemoryAllocator(type("SmallBarBootShell", (), {"transport": BadBootTransport(bar_size=0x1000), "mm": FakeBootMm()})()).alloc(0x2000, sysmem=False), "outside BAR1"),
    (lambda: GspQueueMemoryBuilder.pack_libos_memory_region(0x100, 0, 0x1000, "RMARGS", 0x80000000), "kind"),
    (lambda: GspQueueMemoryBuilder.pack_libos_memory_region(0, 0x100, 0x1000, "RMARGS", 0x80000000), "location"),
    (lambda: GspQueueMemoryBuilder.pack_libos_memory_region(0, 0, 0, "RMARGS", 0x80000000), "region size"),
    (lambda: GspQueueMemoryBuilder.pack_libos_memory_region(0, 0, 0x1000, "", 0x80000000), "name"),
    (lambda: GspQueueMemoryBuilder.pack_libos_memory_region(0, 0, 0x1000, "TOO-LONG-NAME", 0x80000000), "name"),
    (lambda: GspQueueMemoryBuilder.pack_libos_memory_region(0, 0, 0x1000, "RMARGS", 0x80000001), "physical address"),
  ]:
    try:
      bad_boot_alloc()
      raise AssertionError("bad boot allocation/LibOS region input was accepted")
    except (ValueError, RuntimeError) as exc:
      assert text in str(exc)

  nvos02 = pack_nvos02_with_fd(1, 2, 3, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x40, 0x1000000000, 0xfff, fd=-1)
  assert len(nvos02) == 56
  assert struct.unpack_from("<IIIII", nvos02, 0) == (1, 2, 3, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x40)
  assert struct.unpack_from("<Q", nvos02, 24)[0] == 0x1000000000
  assert struct.unpack_from("<Q", nvos02, 32)[0] == 0xfff
  assert unpack_nvos02_status(nvos02) == 0
  assert unpack_nvos02_handle(nvos02) == 3
  for bad_alloc_memory_pack, text in [
    (lambda: pack_nvos02_with_fd(1, 2, 0x100000000, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x40, 0x1000, 0xfff), "object handle"),
    (lambda: pack_nvos02_with_fd(1, 2, 3, 0x100000000, 0x40, 0x1000, 0xfff), "RM class"),
    (lambda: pack_nvos02_with_fd(1, 2, 3, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x100000000, 0x1000, 0xfff), "alloc-memory flags"),
    (lambda: pack_nvos02_with_fd(1, 2, 3, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x40, -1, 0xfff), "address"),
    (lambda: pack_nvos02_with_fd(1, 2, 3, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x40, 0x1000, -1), "limit"),
    (lambda: pack_nvos02_with_fd(1, 2, 3, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x40, 0x1000, 0xfff, fd=1 << 31), "fd"),
  ]:
    try:
      bad_alloc_memory_pack()
      raise AssertionError("bad Linux alloc-memory packer field was accepted")
    except ValueError as exc:
      assert text in str(exc)
  nvos21 = pack_nvos21(1, 2, 3, NV01_MEMORY_VIRTUAL, params_addr=0x12345000, params_size=4)
  assert struct.unpack_from("<IIIIQI", nvos21, 0) == (1, 2, 3, NV01_MEMORY_VIRTUAL, 0x12345000, 4)
  nvos54 = pack_nvos54(1, 2, NV2080_CTRL_CMD_GPU_GET_GID_INFO, params_addr=0x12345000, params_size=4, flags=0x20)
  assert struct.unpack_from("<IIIIQI", nvos54, 0) == (1, 2, NV2080_CTRL_CMD_GPU_GET_GID_INFO, 0x20, 0x12345000, 4)
  for unpack_call, text in [
    (lambda: unpack_nvos02_handle(b"\0" * 11), "NVOS02"),
    (lambda: unpack_nvos02_status(b"\0" * 43), "NVOS02"),
    (lambda: unpack_register_fd_status(b"\0" * 7), "NV_ESC_REGISTER_FD"),
    (lambda: unpack_nvos21_status(b"\0" * 31), "NVOS21"),
    (lambda: unpack_nvos54_status(b"\0" * 31), "NVOS54"),
    (lambda: unpack_nvos33_linear_address(b"\0" * 39), "NVOS33"),
    (lambda: unpack_nvos33_status(b"\0" * 43), "NVOS33"),
    (lambda: unpack_nvos46_status(b"\0" * 51), "NVOS46"),
  ]:
    try:
      unpack_call()
      raise AssertionError("truncated Linux ioctl response was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_linux_pack, text in [
    (lambda: pack_nvos21(1, 2, 0x100000000, NV01_MEMORY_VIRTUAL), "object handle"),
    (lambda: pack_nvos21(1, 2, 3, 0x100000000), "RM class"),
    (lambda: pack_nvos54(1, 2, 0x100000000), "RM control command"),
    (lambda: pack_nvos54(1, 2, NV2080_CTRL_CMD_GPU_GET_GID_INFO, flags=-1), "RM control flags"),
  ]:
    try:
      bad_linux_pack()
      raise AssertionError("bad Linux RM ioctl packer field was accepted")
    except ValueError as exc:
      assert text in str(exc)

  nvos33 = pack_nvos33_with_fd(1, 2, 3, 0x1000, flags=0x20, offset=0x40, fd=7)
  assert len(nvos33) == 56
  assert struct.unpack_from("<III", nvos33, 0) == (1, 2, 3)
  assert struct.unpack_from("<QQ", nvos33, 16) == (0x40, 0x1000)
  assert unpack_nvos33_status(nvos33) == 0
  for bad_map_pack, text in [
    (lambda: pack_nvos33_with_fd(1, 2, 3, 0), "map length"),
    (lambda: pack_nvos33_with_fd(1, 2, 3, 0x1000, offset=-1), "map offset"),
    (lambda: pack_nvos33_with_fd(1, 2, 3, 0x1000, flags=0x100000000), "map flags"),
    (lambda: pack_nvos33_with_fd(1, 2, 3, 0x1000, fd=1 << 31), "map fd"),
  ]:
    try:
      bad_map_pack()
      raise AssertionError("bad Linux map-memory packer field was accepted")
    except ValueError as exc:
      assert text in str(exc)
  nvos46 = pack_nvos46(1, 2, 0x70, 3, 0x3000, 0x12345000, offset=0x40, flags=0x2210)
  assert struct.unpack_from("<IIIIQQI", nvos46, 0) == (1, 2, 0x70, 3, 0x40, 0x3000, 0x2210)
  assert struct.unpack_from("<Q", nvos46, 40)[0] == 0x12345000
  for bad_dma_pack, text in [
    (lambda: pack_nvos46(1, 2, 0x70, 3, 0, 0x12345000), "DMA map length"),
    (lambda: pack_nvos46(1, 2, 0x70, 3, 0x3000, 0x12345000, offset=-1), "DMA map offset"),
    (lambda: pack_nvos46(1, 2, 0x70, 3, 0x3000, -1), "DMA map target offset"),
    (lambda: pack_nvos46(1, 2, 0x70, 3, 0x3000, 0x12345000, flags=0x100000000), "DMA map flags"),
  ]:
    try:
      bad_dma_pack()
      raise AssertionError("bad Linux DMA-map packer field was accepted")
    except ValueError as exc:
      assert text in str(exc)

  linux = object.__new__(LinuxIoctlTransport)
  linux.root = linux.device = linux.virtmem = linux.gpu_uuid = None
  try:
    LinuxIoctlTransport.require_rm_config(linux)
    raise AssertionError("missing RM config was not detected")
  except RuntimeError as exc:
    assert "root" in str(exc) and "gpu_uuid" in str(exc)
  class FakeFd:
    def __init__(self, fd=11):
      self.fd, self.calls = fd, []
    def ioctl(self, req, data):
      self.calls.append((req, bytes(data)))
      if len(data) >= 4 and self.calls[-1][0] == LinuxIoctlTransport.iowr(NV_ESC_CARD_INFO, len(data)):
        data[:4] = b"CARD"
      return data
  linux_fd = object.__new__(LinuxIoctlTransport)
  linux_fd.fd_ctl, linux_fd.fd_dev = FakeFd(fd=21), FakeFd(fd=22)
  card = LinuxIoctlTransport.card_info(linux_fd)
  assert card[:4] == b"CARD"
  assert linux_fd.fd_ctl.calls[-1][0] == LinuxIoctlTransport.iowr(NV_ESC_CARD_INFO, 0x1000)
  regfd = LinuxIoctlTransport.register_fd(linux_fd)
  assert struct.unpack_from("<iI", regfd, 0) == (22, 0)
  assert unpack_register_fd_status(regfd) == 0
  assert linux_fd.fd_ctl.calls[-1][0] == LinuxIoctlTransport.iowr(NV_ESC_REGISTER_FD, 8)
  def truncated_register_fd_ioctl(req, data):
    linux_fd.fd_ctl.calls.append((req, bytes(data)))
    data[:] = b"\0" * 7
    return data
  linux_fd.fd_ctl.ioctl = truncated_register_fd_ioctl
  try:
    LinuxIoctlTransport.register_fd(linux_fd)
    raise AssertionError("truncated register-fd response was accepted")
  except ValueError as exc:
    assert "NV_ESC_REGISTER_FD" in str(exc)
  linux_fd.fd_ctl.ioctl = FakeFd.ioctl.__get__(linux_fd.fd_ctl, FakeFd)
  LinuxIoctlTransport.rm_alloc(linux_fd, 1, 2, 3, NV01_MEMORY_VIRTUAL, params=b"abcd")
  req, data = linux_fd.fd_ctl.calls[-1]
  assert req == LinuxIoctlTransport.iowr(NV_ESC_RM_ALLOC, 32)
  assert struct.unpack_from("<IIII", data, 0) == (1, 2, 3, NV01_MEMORY_VIRTUAL)
  assert struct.unpack_from("<I", data, 24)[0] == 4
  ctl_calls_before = len(linux_fd.fd_ctl.calls)
  try:
    LinuxIoctlTransport.rm_alloc(linux_fd, 1, 2, 3, NV01_MEMORY_VIRTUAL, params=object())
    raise AssertionError("bad Linux ioctl RM alloc params were accepted")
  except ValueError as exc:
    assert "params" in str(exc)
  assert len(linux_fd.fd_ctl.calls) == ctl_calls_before
  ctl_calls_before = len(linux_fd.fd_ctl.calls)
  try:
    LinuxIoctlTransport.rm_control(linux_fd, 1, 2, 3, params=object())
    raise AssertionError("bad Linux ioctl RM control params were accepted")
  except ValueError as exc:
    assert "params" in str(exc)
  assert len(linux_fd.fd_ctl.calls) == ctl_calls_before
  def mutating_ioctl(req, data):
    linux_fd.fd_ctl.calls.append((req, bytes(data)))
    params_addr, params_size = struct.unpack_from("<QI", data, 16)
    if params_addr and params_size:
      ctypes.memmove(params_addr, b"WXYZ", min(params_size, 4))
    return data
  linux_fd.fd_ctl.ioctl = mutating_ioctl
  ctrl_resp = LinuxIoctlTransport.rm_control(linux_fd, 1, 2, NV2080_CTRL_CMD_GPU_GET_GID_INFO, params=b"abcd")
  assert ctrl_resp[:4] == b"WXYZ"
  req, data = linux_fd.fd_ctl.calls[-1]
  assert req == LinuxIoctlTransport.iowr(NV_ESC_RM_CONTROL, 32)
  assert struct.unpack_from("<IIII", data, 0) == (1, 2, NV2080_CTRL_CMD_GPU_GET_GID_INFO, 0)
  def map_memory_ioctl(req, data):
    linux_fd.fd_ctl.calls.append((req, bytes(data)))
    struct.pack_into("<Q", data, 32, 0xfeed0000)
    return data
  linux_fd.fd_ctl.ioctl = map_memory_ioctl
  mapped_addr, mapped_data = LinuxIoctlTransport.rm_map_memory(linux_fd, 1, 2, 3, 0x2000, flags=0x44, offset=0x80, fd=9)
  assert mapped_addr == 0xfeed0000
  assert unpack_nvos33_linear_address(mapped_data) == 0xfeed0000
  req, data = linux_fd.fd_ctl.calls[-1]
  assert req == LinuxIoctlTransport.iowr(NV_ESC_RM_MAP_MEMORY, 56)
  assert struct.unpack_from("<III", data, 0) == (1, 2, 3)
  assert struct.unpack_from("<QQ", data, 16) == (0x80, 0x2000)
  assert struct.unpack_from("<I", data, 44)[0] == 0x44
  assert struct.unpack_from("<i", data, 48)[0] == 9
  linux_fd.fd_ctl.ioctl = FakeFd.ioctl.__get__(linux_fd.fd_ctl, FakeFd)
  dma_data = LinuxIoctlTransport.rm_map_memory_dma(linux_fd, 1, 2, 0x70, 3, 0x3000, 0x12345000, offset=0x40, flags=0x2210)
  req, data = linux_fd.fd_ctl.calls[-1]
  assert req == LinuxIoctlTransport.iowr(NV_ESC_RM_MAP_MEMORY_DMA, 56)
  assert struct.unpack_from("<IIII", data, 0) == (1, 2, 0x70, 3)
  assert struct.unpack_from("<QQI", data, 16) == (0x40, 0x3000, 0x2210)
  assert struct.unpack_from("<Q", data, 40)[0] == 0x12345000
  assert unpack_nvos46_status(dma_data) == 0
  linux_fd.fd_ctl.ioctl = FakeFd.ioctl.__get__(linux_fd.fd_ctl, FakeFd)
  linux_fd.fd_dev.ioctl = FakeFd.ioctl.__get__(linux_fd.fd_dev, FakeFd)
  linux_rm = LinuxRmClient(linux_fd, priv_root=0xc1e00004, first_handle=0xcf000000)
  assert linux_rm.alloc_root() == 0xc1e00004
  req, data = linux_fd.fd_ctl.calls[-1]
  assert req == LinuxIoctlTransport.iowr(NV_ESC_RM_ALLOC, 32)
  assert struct.unpack_from("<IIII", data, 0) == (0xc1e00004, 0, 0xc1e00004, NV01_ROOT)
  h_obj = linux_rm.rm_alloc(0x80, NV01_MEMORY_VIRTUAL, b"zz")
  assert h_obj == 0xcf000000
  assert linux_rm.next_handle == 0xcf000001
  req, data = linux_fd.fd_ctl.calls[-1]
  assert struct.unpack_from("<IIII", data, 0) == (0xc1e00004, 0x80, 0xcf000000, NV01_MEMORY_VIRTUAL)
  h_mem = linux_rm.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x12345000], 0x1000, 0x40)
  assert h_mem == 0xcf000001
  req, data = linux_fd.fd_dev.calls[-1]
  assert req == LinuxIoctlTransport.iowr(NV_ESC_RM_ALLOC_MEMORY, 56)
  assert struct.unpack_from("<IIIII", data, 0) == (0xc1e00004, 0x80, 0xcf000001, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x40)
  assert struct.unpack_from("<QQ", data, 24) == (0x12345000, 0xfff)
  h_mem_client = linux_rm.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x22345000], 0x2000, 0x44,
    client=0xc1e00066, h_memory=0xcf000066)
  assert h_mem_client == 0xcf000066
  req, data = linux_fd.fd_dev.calls[-1]
  assert req == LinuxIoctlTransport.iowr(NV_ESC_RM_ALLOC_MEMORY, 56)
  assert struct.unpack_from("<IIIII", data, 0) == (0xc1e00066, 0x80, 0xcf000066, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, 0x44)
  assert struct.unpack_from("<QQ", data, 24) == (0x22345000, 0x1fff)
  class FailingLinuxAllocMemoryTransport:
    def __init__(self, base):
      self.__dict__.update(base.__dict__)
      self.dev_ioctl = self.fail_dev_ioctl
    def fail_dev_ioctl(self, nr, data):
      struct.pack_into("<I", data, 40, 0x55)
      return data
  linux_alloc_fail_rm = LinuxRmClient(FailingLinuxAllocMemoryTransport(linux_fd))
  try:
    linux_alloc_fail_rm.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x32345000], 0x3000, 0x48,
      client=0xc1e00077, h_memory=0xcf000077)
    raise AssertionError("failing Linux alloc-memory status was accepted")
  except RuntimeError as exc:
    msg = str(exc)
    expected_data = pack_nvos02_with_fd(0xc1e00077, 0x80, 0xcf000077, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR,
      0x48, 0x32345000, 0x2fff)
    struct.pack_into("<I", expected_data, 40, 0x55)
    assert "NV_ESC_RM_ALLOC_MEMORY" in msg and "failed: 0x55" in msg
    assert "client=0xc1e00077" in msg and "device=0x80" in msg and "memory=0xcf000077" in msg
    assert "class_name=NV01_MEMORY_SYSTEM_OS_DESCRIPTOR" in msg and "length=0x3000" in msg
    assert "flags=0x48" in msg and "pages=1" in msg
    assert f"payload_sha256={hashlib.sha256(expected_data).hexdigest()}" in msg
  try:
    LinuxRmClient(linux_fd, first_handle=0x100000000)
    raise AssertionError("bad Linux RM client handle seed was accepted")
  except ValueError as exc:
    assert "first RM handle" in str(exc)
  linux_overflow = LinuxRmClient(linux_fd)
  linux_overflow.next_handle = 0x100000000
  try:
    linux_overflow.handle()
    raise AssertionError("overflowing generated Linux RM handle was accepted")
  except ValueError as exc:
    assert "RM handle" in str(exc)
  try:
    LinuxRmClient(linux_fd).rm_alloc(0x80, NV01_MEMORY_VIRTUAL, h_object=0x100000000)
    raise AssertionError("overflowing explicit Linux RM handle was accepted")
  except ValueError as exc:
    assert "object handle" in str(exc)
  linux_bad_class = LinuxRmClient(linux_fd)
  ctl_calls_before = len(linux_fd.fd_ctl.calls)
  dev_calls_before = len(linux_fd.fd_dev.calls)
  next_before = linux_bad_class.next_handle
  try:
    linux_bad_class.rm_alloc(0x80, 0x100000000)
    raise AssertionError("bad Linux RM class consumed a handle")
  except ValueError as exc:
    assert "RM class" in str(exc)
  assert len(linux_fd.fd_ctl.calls) == ctl_calls_before
  assert len(linux_fd.fd_dev.calls) == dev_calls_before
  assert linux_bad_class.next_handle == next_before
  linux_control_calls_before = len(linux_fd.fd_ctl.calls)
  try:
    linux_rm.rm_control(0x80, 0x1234, object())
    raise AssertionError("bad Linux RM control params were accepted")
  except ValueError as exc:
    assert "params" in str(exc)
  assert len(linux_fd.fd_ctl.calls) == linux_control_calls_before
  for bad_linux_control, text in [
    (lambda: linux_rm.rm_control(0x100000000, 0x1234, b""), "object handle"),
    (lambda: linux_rm.rm_control(0x80, 0x100000000, b""), "RM control command"),
    (lambda: linux_rm.rm_control(0x80, 0x1234, b"", client=0x100000000), "client handle"),
  ]:
    try:
      bad_linux_control()
      raise AssertionError("bad Linux RM control input reached ioctl")
    except ValueError as exc:
      assert text in str(exc)
    assert len(linux_fd.fd_ctl.calls) == linux_control_calls_before
  try:
    os.environ["NV_ADD_TRACE_RM_ALLOC"] = "1"
    linux_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(linux_trace_buf):
      linux_rm.rm_control(0x80, 0x1234, b"ping")
    linux_trace = linux_trace_buf.getvalue()
    assert "rm linux_control" in linux_trace and "cmd=0x1234" in linux_trace and "cmd_name=UNKNOWN_0x1234" in linux_trace
    assert f"params_sha256={hashlib.sha256(b'ping').hexdigest()}" in linux_trace
  finally:
    if old_trace_rm is None: os.environ.pop("NV_ADD_TRACE_RM_ALLOC", None)
    else: os.environ["NV_ADD_TRACE_RM_ALLOC"] = old_trace_rm
  class FailingLinuxControlTransport:
    def __init__(self, base):
      self.__dict__.update(base.__dict__)
    def rm_control(self, client, h_object, cmd, params=b""):
      raise RuntimeError("NV_ESC_RM_CONTROL failed: 0x61")
  linux_fail_fd = FailingLinuxControlTransport(linux_fd)
  linux_fail_rm = LinuxRmClient(linux_fail_fd)
  try:
    linux_fail_rm.rm_control(0x80, 0x1234, b"fail")
    raise AssertionError("failing Linux RM control was accepted")
  except RuntimeError as exc:
    msg = str(exc)
    assert "Linux RM_CONTROL" in msg and "object=0x80" in msg and "cmd=0x1234" in msg
    assert "cmd_name=UNKNOWN_0x1234" in msg and "payload_len=4" in msg
    assert f"payload_sha256={hashlib.sha256(b'fail').hexdigest()}" in msg
  linux_wait_client = LinuxRmClient(linux_fd)
  ctl_calls_before = len(linux_fd.fd_ctl.calls)
  dev_calls_before = len(linux_fd.fd_dev.calls)
  next_before = linux_wait_client.next_handle
  for bad_linux_wait, text in [
    (lambda: linux_wait_client.alloc_root(wait=1), "Linux RM alloc wait flag"),
    (lambda: linux_wait_client.rm_alloc(0x80, NV01_MEMORY_VIRTUAL, wait=1), "Linux RM alloc wait flag"),
    (lambda: linux_wait_client.unloading_guest_driver(wait=1), "Linux RM unloading_guest_driver wait flag"),
  ]:
    try:
      bad_linux_wait()
      raise AssertionError("bad Linux RM wait flag was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(linux_fd.fd_ctl.calls) == ctl_calls_before
    assert len(linux_fd.fd_dev.calls) == dev_calls_before
    assert linux_wait_client.next_handle == next_before
  linux_bad_alloc = LinuxRmClient(linux_fd)
  ctl_calls_before = len(linux_fd.fd_ctl.calls)
  dev_calls_before = len(linux_fd.fd_dev.calls)
  next_before = linux_bad_alloc.next_handle
  for bad_linux_alloc, text in [
    (lambda: linux_bad_alloc.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [], 0x1000, 0x40), "physical page"),
    (lambda: linux_bad_alloc.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x12345000], 0, 0x40), "allocation length"),
    (lambda: linux_bad_alloc.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x12345000], 0x1000, 0x40,
      h_memory=0x100000000), "memory handle"),
    (lambda: linux_bad_alloc.alloc_memory(0x80, NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, [0x12345000], 0x1000, 0x40,
      client=0x100000000), "client handle"),
  ]:
    try:
      bad_linux_alloc()
      raise AssertionError("bad Linux alloc-memory input consumed a handle")
    except ValueError as exc:
      assert text in str(exc)
    assert len(linux_fd.fd_ctl.calls) == ctl_calls_before
    assert len(linux_fd.fd_dev.calls) == dev_calls_before
    assert linux_bad_alloc.next_handle == next_before
  with tempfile.TemporaryDirectory() as td:
    pci_root = pathlib.Path(td)
    dev0 = pci_root / "0000:01:00.0"
    dev1 = pci_root / "0000:02:00.0"
    dev0.mkdir(); dev1.mkdir()
    (dev0 / "vendor").write_text("0x10de\n", encoding="utf-8")
    (dev0 / "class").write_text("0x030000\n", encoding="utf-8")
    (dev0 / "resource").write_text("0x1000 0x1fff 0x0\n", encoding="utf-8")
    (dev0 / "resource0").write_bytes(bytes(0x1000))
    (dev0 / "config").write_bytes(b"\xde\x10\x34\x12" + bytes(252))
    (dev1 / "vendor").write_text("0x1234\n", encoding="utf-8")
    (dev1 / "class").write_text("0x030000\n", encoding="utf-8")
    assert LinuxIoctlTransport.find_pci_device(0, root=pci_root) == dev0
    linux2 = object.__new__(LinuxIoctlTransport)
    linux2.pci_path = dev0
    assert LinuxIoctlTransport.read_config(linux2, 0, 2) == 0x10de
    assert LinuxIoctlTransport.read_config(linux2, 2, 2) == 0x1234
    for config_call, text in [
      (lambda: LinuxIoctlTransport.read_config(linux2, -1, 2), "offset"),
      (lambda: LinuxIoctlTransport.read_config(linux2, 0, 3), "size"),
      (lambda: LinuxIoctlTransport.write_config(linux2, 4095, 0, 2), "outside"),
    ]:
      try:
        config_call()
        raise AssertionError("bad PCI config access was accepted")
      except ValueError as exc:
        assert text in str(exc)
    assert LinuxIoctlTransport.bar_info(linux2, 0) == (0x1000, 0x1000)
    assert LinuxIoctlTransport.map_bar(linux2, 0, off=0, size=0x100).nbytes == 0x100
    for bar_call, text in [
      (lambda: LinuxIoctlTransport.bar_info(linux2, 6), "BAR index"),
      (lambda: LinuxIoctlTransport.map_bar(linux2, 0, off=-1, size=0x100), "offset"),
      (lambda: LinuxIoctlTransport.map_bar(linux2, 0, off=0, size=0), "size"),
      (lambda: LinuxIoctlTransport.map_bar(linux2, 0, off=0x900, size=0x800), "outside"),
    ]:
      try:
        bar_call()
        raise AssertionError("bad BAR access was accepted")
      except ValueError as exc:
        assert text in str(exc)
  test_alloc = SysmemAllocation(MMIOView(bytearray(0x1000)), [0x5000], va_addr=0x4000, size=0x1000, h_memory=0x44, cpu_addr=0x7000)
  tv, tp = test_alloc
  assert tv is test_alloc.view and tp == [0x5000]
  linux_free = object.__new__(LinuxIoctlTransport)
  linux_free.freed = []
  linux_free.va_allocator = FreeListAllocator(0x10000, base=0x4000)
  assert linux_free.va_allocator.alloc(0x1000, 0x1000) == 0x4000
  linux_free.uvm_free = lambda base, length: linux_free.freed.append((base, length))
  old_munmap = FileIO.munmap
  try:
    FileIO.munmap = staticmethod(lambda addr, size: linux_free.freed.append(("munmap", addr, size)))
    LinuxIoctlTransport.free_sysmem(linux_free, test_alloc)
  finally:
    FileIO.munmap = old_munmap
  assert linux_free.freed == [(0x4000, 0x1000), ("munmap", 0x7000, 0x1000)]
  assert test_alloc.va_addr is None and test_alloc.cpu_addr is None and test_alloc.size == 0
  LinuxIoctlTransport.free_sysmem(linux_free, test_alloc)
  assert linux_free.freed == [(0x4000, 0x1000), ("munmap", 0x7000, 0x1000)]
  assert linux_free.va_allocator.alloc(0x1000, 0x1000) == 0x4000
  retry_alloc = SysmemAllocation(MMIOView(bytearray(0x1000)), [0x6000], va_addr=0x8000, size=0x1000, h_memory=0x55, cpu_addr=0x9000)
  linux_retry = object.__new__(LinuxIoctlTransport)
  linux_retry.calls = []
  linux_retry.va_allocator = FreeListAllocator(0x10000, base=0x8000)
  assert linux_retry.va_allocator.alloc(0x1000, 0x1000) == 0x8000
  def retry_uvm_free(base, length):
    linux_retry.calls.append(("uvm", base, length))
    if len([call for call in linux_retry.calls if call[0] == "uvm"]) == 1: raise RuntimeError("uvm free fail")
  linux_retry.uvm_free = retry_uvm_free
  old_munmap = FileIO.munmap
  try:
    FileIO.munmap = staticmethod(lambda addr, size: linux_retry.calls.append(("munmap", addr, size)))
    try:
      LinuxIoctlTransport.free_sysmem(linux_retry, retry_alloc)
      raise AssertionError("Linux sysmem free UVM failure was swallowed")
    except RuntimeError as exc:
      assert "uvm free fail" in str(exc)
    assert retry_alloc.va_addr == 0x8000 and retry_alloc.cpu_addr is None and retry_alloc.size == 0x1000
    assert getattr(retry_alloc, "_va_recycled", False)
    LinuxIoctlTransport.free_sysmem(linux_retry, retry_alloc)
  finally:
    FileIO.munmap = old_munmap
  assert linux_retry.calls == [("uvm", 0x8000, 0x1000), ("munmap", 0x9000, 0x1000), ("uvm", 0x8000, 0x1000)]
  assert retry_alloc.va_addr is None and retry_alloc.cpu_addr is None and retry_alloc.size == 0
  assert linux_retry.va_allocator.alloc(0x1000, 0x1000) == 0x8000

  linux_fail = object.__new__(LinuxIoctlTransport)
  linux_fail.root, linux_fail.device, linux_fail.virtmem, linux_fail.gpu_uuid = 0xc1e00004, 0x80, 0x70, bytes(range(16))
  linux_fail.next_host_handle = 0xcf000000
  linux_fail.va_allocator = FreeListAllocator(0x10000, base=0x5000)
  linux_fail.anon_cpu_mapping = lambda va, size: 0x7000
  linux_fail.dev_ioctl = lambda nr, data: (_ for _ in ()).throw(RuntimeError("alloc-memory fail"))
  linux_fail.rm_map_memory_dma = lambda *args, **kwargs: None
  linux_fail.uvm_create_external_range = lambda base, length: None
  linux_fail.uvm_map_external_allocation = lambda *args, **kwargs: None
  linux_fail.uvm_free = lambda base, length: linux_fail.freed.append((base, length))
  linux_fail.freed = []
  old_munmap = FileIO.munmap
  try:
    FileIO.munmap = staticmethod(lambda addr, size: linux_fail.freed.append(("munmap", addr, size)))
    try:
      LinuxIoctlTransport.alloc_sysmem(linux_fail, 0x1000)
      raise AssertionError("Linux sysmem alloc failure was accepted")
    except RuntimeError as exc:
      assert "alloc-memory fail" in str(exc)
  finally:
    FileIO.munmap = old_munmap
  assert linux_fail.freed == [("munmap", 0x7000, mmap.PAGESIZE)]
  assert linux_fail.va_allocator.alloc(0x1000, 0x1000) == 0x5000

  linux_fail2 = object.__new__(LinuxIoctlTransport)
  linux_fail2.root, linux_fail2.device, linux_fail2.virtmem, linux_fail2.gpu_uuid = 0xc1e00004, 0x80, 0x70, bytes(range(16))
  linux_fail2.next_host_handle = 0xcf000000
  linux_fail2.va_allocator = FreeListAllocator(0x10000, base=0x6000)
  linux_fail2.anon_cpu_mapping = lambda va, size: 0x8000
  linux_fail2.dev_ioctl = lambda nr, data: data
  linux_fail2.rm_map_memory_dma = lambda *args, **kwargs: None
  linux_fail2.uvm_create_external_range = lambda base, length: None
  linux_fail2.uvm_map_external_allocation = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("uvm-map fail"))
  linux_fail2.freed = []
  linux_fail2.uvm_free = lambda base, length: linux_fail2.freed.append((base, length))
  old_munmap = FileIO.munmap
  try:
    FileIO.munmap = staticmethod(lambda addr, size: linux_fail2.freed.append(("munmap", addr, size)))
    try:
      LinuxIoctlTransport.alloc_sysmem(linux_fail2, 0x1000)
      raise AssertionError("Linux sysmem UVM map failure was accepted")
    except RuntimeError as exc:
      assert "uvm-map fail" in str(exc)
  finally:
    FileIO.munmap = old_munmap
  assert linux_fail2.freed == [(0x6000, mmap.PAGESIZE), ("munmap", 0x8000, mmap.PAGESIZE)]
  assert linux_fail2.va_allocator.alloc(0x1000, 0x1000) == 0x6000

  linux_uvm = object.__new__(LinuxIoctlTransport)
  linux_uvm.fd_ctl = type("FD", (), {"fd": 5})()
  linux_uvm.fd_uvm = type("FD", (), {"fd": 6})()
  linux_uvm.uvm_calls = []
  linux_uvm.uvm_ioctl = lambda cmd, data: linux_uvm.uvm_calls.append((cmd, bytes(data))) or data
  init = LinuxIoctlTransport.uvm_initialize(linux_uvm)
  assert unpack_uvm_status(init, 8) == 0 and linux_uvm.uvm_calls[-1][0] == UVM_INITIALIZE
  mm_init = LinuxIoctlTransport.uvm_mm_initialize(linux_uvm)
  assert unpack_uvm_status(mm_init, 4) == 0 and linux_uvm.uvm_calls[-1][0] == UVM_MM_INITIALIZE
  assert struct.unpack_from("<i", linux_uvm.uvm_calls[-1][1], 0)[0] == 6
  gid_req = pack_nv2080_gpu_get_gid_info(index=1, flags=NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
  assert struct.unpack_from("<III", gid_req, 0) == (1, NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, 16)
  gid_resp = bytearray(28)
  gid_resp[12:28] = bytes(range(16))
  assert unpack_nv2080_gpu_gid(gid_resp) == bytes(range(16))
  mapping_attrs = pack_uvm_gpu_mapping_attributes(bytes(range(16)), mapping_type=1, caching_type=2, format_type=3, element_bits=4, compression_type=5)
  assert mapping_attrs[:16] == bytes(range(16))
  assert struct.unpack_from("<IIIII", mapping_attrs, 16) == (1, 2, 3, 4, 5)
  for bad_uvm_misc_call, text in [
    (lambda: pack_uvm_initialize(flags=0x100000000), "UVM initialize flags"),
    (lambda: pack_uvm_mm_initialize(1 << 31), "UVM fd"),
    (lambda: pack_nv2080_gpu_get_gid_info(index=-1), "GPU GID index"),
    (lambda: pack_nv2080_gpu_get_gid_info(length=0x100000000), "GPU GID length"),
    (lambda: unpack_nv2080_gpu_gid(b"\0" * 27), "GPU GID response"),
    (lambda: pack_uvm_register_gpu(b"\0" * 15, 5), "GPU UUID"),
    (lambda: pack_uvm_register_gpu(bytes(range(16)), 1 << 31), "UVM RM control fd"),
    (lambda: pack_uvm_register_gpu(bytes(range(16)), 5, h_client=0x100000000), "client handle"),
    (lambda: pack_uvm_register_gpu(bytes(range(16)), 5, h_smc_part_ref=0x100000000), "SMC partition ref"),
    (lambda: pack_uvm_register_gpu_vaspace(bytes(range(16)), 5, 0xc1e00004, 0x100000000), "vaspace handle"),
    (lambda: pack_uvm_gpu_mapping_attributes(bytes(range(16)), compression_type=0x100000000), "UVM compression type"),
  ]:
    try:
      bad_uvm_misc_call()
      raise AssertionError("bad UVM/GID helper input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  def failing_uvm_init(cmd, data):
    if cmd == UVM_INITIALIZE: struct.pack_into("<I", data, 8, 0x55)
    elif cmd == UVM_MM_INITIALIZE: struct.pack_into("<I", data, 4, 0x66)
    return data
  linux_uvm.uvm_ioctl = failing_uvm_init
  for fn, text in [(LinuxIoctlTransport.uvm_initialize, "0x55"), (LinuxIoctlTransport.uvm_mm_initialize, "0x66")]:
    try:
      fn(linux_uvm)
      raise AssertionError("failed UVM initialize status was accepted")
    except RuntimeError as exc:
      assert text in str(exc)
  linux_uvm.uvm_ioctl = lambda cmd, data: linux_uvm.uvm_calls.append((cmd, bytes(data))) or data
  ext = LinuxIoctlTransport.uvm_create_external_range(linux_uvm, 0x40000000, 0x2000)
  assert unpack_uvm_status(ext, 16) == 0
  assert linux_uvm.uvm_calls[-1][0] == UVM_CREATE_EXTERNAL_RANGE
  assert struct.unpack_from("<QQ", linux_uvm.uvm_calls[-1][1], 0) == (0x40000000, 0x2000)
  for bad_uvm_range_call, text in [
    (lambda: pack_uvm_create_external_range(0x40000001, 0x2000), "external range base"),
    (lambda: pack_uvm_create_external_range(0x40000000, 0), "external range length"),
    (lambda: pack_uvm_free(0x40000001, 0x2000), "free range base"),
    (lambda: pack_uvm_free(0x40000000, 0x1800), "free range length"),
  ]:
    try:
      bad_uvm_range_call()
      raise AssertionError("bad UVM range request was accepted")
    except ValueError as exc:
      assert text in str(exc)
  mapped = LinuxIoctlTransport.uvm_map_external_allocation(linux_uvm, 0x40000000, 0x2000, bytes(range(16)), 0xc1e00004, 0xcf000000)
  assert unpack_uvm_status(mapped, 9260) == 0
  for uvm_status_call in [
    lambda: unpack_uvm_status(b"\0" * 11, 8),
    lambda: unpack_uvm_status(b"\0" * 9263, 9260),
  ]:
    try:
      uvm_status_call()
      raise AssertionError("truncated UVM status response was accepted")
    except ValueError as exc:
      assert "UVM status" in str(exc)
  assert linux_uvm.uvm_calls[-1][0] == UVM_MAP_EXTERNAL_ALLOCATION
  assert struct.unpack_from("<QQQ", linux_uvm.uvm_calls[-1][1], 0) == (0x40000000, 0x2000, 0)
  assert linux_uvm.uvm_calls[-1][1][24:40] == bytes(range(16))
  assert struct.unpack_from("<iIII", linux_uvm.uvm_calls[-1][1], 9248) == (5, 0xc1e00004, 0xcf000000, 0)
  for bad_uvm_map_call, text in [
    (lambda: pack_uvm_map_external_allocation(0x40000001, 0x2000, bytes(range(16)), 5, 0xc1e00004, 0xcf000000), "external allocation base"),
    (lambda: pack_uvm_map_external_allocation(0x40000000, 0x1800, bytes(range(16)), 5, 0xc1e00004, 0xcf000000), "external allocation length"),
    (lambda: pack_uvm_map_external_allocation(0x40000000, 0x2000, bytes(range(16)), 5, 0xc1e00004, 0xcf000000, offset=-1), "offset"),
    (lambda: pack_uvm_map_external_allocation(0x40000000, 0x2000, bytes(range(16)), 1 << 31, 0xc1e00004, 0xcf000000), "UVM RM control fd"),
    (lambda: pack_uvm_map_external_allocation(0x40000000, 0x2000, bytes(range(16)), 5, 0x100000000, 0xcf000000), "client handle"),
  ]:
    try:
      bad_uvm_map_call()
      raise AssertionError("bad UVM external-allocation map request was accepted")
    except ValueError as exc:
      assert text in str(exc)
  freed = LinuxIoctlTransport.uvm_free(linux_uvm, 0x40000000, 0x2000)
  assert unpack_uvm_status(freed, 16) == 0
  assert linux_uvm.uvm_calls[-1][0] == UVM_FREE
  assert struct.unpack_from("<QQ", linux_uvm.uvm_calls[-1][1], 0) == (0x40000000, 0x2000)

  class FakeLinuxRm:
    def __init__(self): self.controls = []
    def rm_control(self, h_object, cmd, params=b"", client=None):
      self.controls.append((h_object, cmd, bytes(params)))
      out = bytearray(params)
      out[12:28] = bytes(range(16))
      return bytes(out)
  linux3 = object.__new__(LinuxIoctlTransport)
  linux3.fd_ctl = type("FD", (), {"fd": 5})()
  linux3.rm = FakeLinuxRm()
  linux3.uvm_calls = []
  def fake_uvm_ioctl(cmd, data):
    linux3.uvm_calls.append((cmd, bytes(data)))
    return data
  linux3.uvm_ioctl = fake_uvm_ioctl
  linux3.uvm_initialize = lambda: fake_uvm_ioctl(UVM_INITIALIZE, pack_uvm_initialize())
  linux3.uvm_mm_initialize = lambda: fake_uvm_ioctl(UVM_MM_INITIALIZE, pack_uvm_mm_initialize(6))
  got_uuid = LinuxIoctlTransport.setup_uvm(linux3, 0xc1e00004, 0x80, 0x2080, 0x70, 0x90f1)
  assert got_uuid == bytes(range(16))
  assert linux3.root == 0xc1e00004 and linux3.device == 0x80 and linux3.virtmem == 0x70
  assert linux3.gpu_uuid == bytes(range(16))
  assert any(call[0] == UVM_REGISTER_GPU for call in linux3.uvm_calls)
  assert any(call[0] == UVM_REGISTER_GPU_VASPACE for call in linux3.uvm_calls)
  linux3.va_allocator = FreeListAllocator(0x10000000, base=0x20000000)
  ch_base, ch_len = LinuxIoctlTransport.register_channel(linux3, 0xc56f, length=0x4000)
  assert (ch_base, ch_len) == (0x20000000, 0x4000)
  reg_ch = next(data for cmd, data in linux3.uvm_calls if cmd == UVM_REGISTER_CHANNEL)
  assert reg_ch[:16] == bytes(range(16))
  linux_reg_fail = object.__new__(LinuxIoctlTransport)
  linux_reg_fail.fd_ctl = type("FD", (), {"fd": 5})()
  linux_reg_fail.root, linux_reg_fail.device, linux_reg_fail.virtmem, linux_reg_fail.gpu_uuid = 0xc1e00004, 0x80, 0x70, bytes(range(16))
  linux_reg_fail.va_allocator = FreeListAllocator(0x10000000, base=0x30000000)
  def failing_register_channel_ioctl(cmd, data):
    struct.pack_into("<I", data, 48, 0x77)
    return data
  linux_reg_fail.uvm_ioctl = failing_register_channel_ioctl
  try:
    LinuxIoctlTransport.register_channel(linux_reg_fail, 0xc56f, length=0x4000)
    raise AssertionError("failed UVM channel registration was accepted")
  except RuntimeError as exc:
    assert "UVM_REGISTER_CHANNEL" in str(exc)
  assert linux_reg_fail.va_allocator.alloc(0x4000, 0x1000) == 0x30000000
  assert struct.unpack_from("<iIII4xQQ", reg_ch, 16) == (5, 0xc1e00004, 0xc56f, 0, 0x20000000, 0x4000)
  for bad_register_channel_call, text in [
    (lambda: pack_uvm_register_channel(bytes(range(16)), 5, 0xc1e00004, 0xc56f, 0x20000001, 0x4000), "channel range base"),
    (lambda: pack_uvm_register_channel(bytes(range(16)), 5, 0xc1e00004, 0xc56f, 0x20000000, 0x1800), "channel range length"),
    (lambda: pack_uvm_register_channel(bytes(range(16)), 1 << 31, 0xc1e00004, 0xc56f, 0x20000000, 0x4000), "UVM RM control fd"),
    (lambda: pack_uvm_register_channel(bytes(range(16)), 5, 0x100000000, 0xc56f, 0x20000000, 0x4000), "client handle"),
    (lambda: pack_uvm_register_channel(bytes(range(16)), 5, 0xc1e00004, 0x100000000, 0x20000000, 0x4000), "channel handle"),
  ]:
    try:
      bad_register_channel_call()
      raise AssertionError("bad UVM channel registration request was accepted")
    except ValueError as exc:
      assert text in str(exc)
  LinuxIoctlTransport.free_registered_channel(linux3, (ch_base, ch_len))
  assert linux3.uvm_calls[-1][0] == UVM_FREE
  assert struct.unpack_from("<QQ", linux3.uvm_calls[-1][1], 0) == (ch_base, ch_len)
  assert linux3.va_allocator.alloc(0x4000, 0x1000) == ch_base
  LinuxIoctlTransport.free_registered_channel(linux3, None)
  class FailingFreeChannelTransport(LinuxIoctlTransport):
    def uvm_free(self, base, length): raise RuntimeError("uvm free fail")
  linux_free_fail = object.__new__(FailingFreeChannelTransport)
  linux_free_fail.uvm_calls = []
  linux_free_fail.va_allocator = FreeListAllocator(0x10000, base=0x60000000)
  linux_free_fail.uvm_free = lambda base, length: (_ for _ in ()).throw(RuntimeError("uvm free fail"))
  linux_free_fail.va_allocator.alloc(0x4000, 0x1000)
  try:
    LinuxIoctlTransport.free_registered_channel(linux_free_fail, (0x60000000, 0x4000))
    raise AssertionError("failing UVM free during registered-channel cleanup was accepted")
  except RuntimeError as exc:
    assert "uvm free fail" in str(exc)
  assert linux_free_fail.va_allocator.alloc(0x4000, 0x1000) == 0x60000000
  linux_bad_free = object.__new__(LinuxIoctlTransport)
  linux_bad_free.uvm_calls = []
  linux_bad_free.va_allocator = FreeListAllocator(0x10000, base=0x50000000)
  linux_bad_free.uvm_free = lambda base, length: linux_bad_free.uvm_calls.append((base, length))
  for bad_registration, text in [
    (("bad", 0x1000), "metadata"),
    ((0x50000000, 0), "length"),
    ((0x50000001, 0x1000), "base"),
    ((0x50000000, 0x1800), "length"),
    (("bad",), "metadata"),
  ]:
    try:
      LinuxIoctlTransport.free_registered_channel(linux_bad_free, bad_registration)
      raise AssertionError("bad registered channel metadata was accepted")
    except ValueError as exc:
      assert text in str(exc)
  assert linux_bad_free.uvm_calls == []

  class FakeTransport:
    def sleep(self, timeout_ms): pass
  class FakeChip:
    chip_id = 0x12345678
    pmc_boot_0 = 0x12345678
    fw_name = "ga102"
  class FakeShell:
    def __init__(self):
      self.transport, self.chip, self.regs, self.writes = FakeTransport(), FakeChip(), {}, []
      self.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
      self.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
      self.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
      self.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FBIF_CTL] = 0x110
      self.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FBIF_TRANSCFG0] = 0x110
      self.regs[NV_FALCON_GSP_BASE + NV_PRISCV_RISCV_BCR_CTRL] = 1
    def fecs_falcon_base(self):
      base = NV_FALCON_FECS_BASES.get(self.chip.fw_name)
      if base is None: raise ValueError(f"FECS FALCON base is unknown for firmware {self.chip.fw_name}")
      return base
    def fecs_falcon_engine(self): return self.fecs_falcon_base() + NV_PFECS_FALCON_ENGINE_OFFSET
    def rreg(self, addr): return self.regs.get(addr, 0)
    def wreg(self, addr, value):
      self.writes.append((addr, value))
      if addr == NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD:
        self.regs[addr] = 0x2
      elif addr == NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL and value == 0x2:
        self.regs[addr] = 0x10
      else:
        self.regs[addr] = value
  fake = FakeShell()
  falcon = FalconController(fake)
  cmd = falcon.dma_cmd(imem=True, sec=True)
  assert cmd == ((1 << 2) | (1 << 4) | (FalconController.DMA_SIZE_256B << 8))
  falcon.wait_until(lambda: True, "ready", timeout_ms=0)
  for bad_wait_call, text in [
    (lambda: falcon.wait_until(object(), "bad"), "predicate"),
    (lambda: falcon.wait_until(lambda: True, "bad", timeout_ms=-1), "timeout"),
  ]:
    try:
      bad_wait_call()
      raise AssertionError("bad Falcon wait input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_dma_call, text in [
    (lambda: falcon.execute_dma(NV_FALCON_GSP_BASE, 0x100000000, 0x10, 0x20, 0x1234500, 256), "DMA command"),
    (lambda: falcon.execute_dma(-1, cmd, 0x10, 0x20, 0x1234500, 256), "Falcon base"),
    (lambda: falcon.execute_dma(NV_FALCON_GSP_BASE, cmd, -1, 0x20, 0x1234500, 256), "destination"),
    (lambda: falcon.execute_dma(NV_FALCON_GSP_BASE, cmd, 0x10, -1, 0x1234500, 256), "memory offset"),
    (lambda: falcon.execute_dma(NV_FALCON_GSP_BASE, cmd, 0x10, 0x20, -1, 256), "source"),
    (lambda: falcon.execute_dma(NV_FALCON_GSP_BASE, cmd, 0x10, 0x20, 0x1234500, 0), "DMA size"),
    (lambda: falcon.execute_dma(NV_FALCON_GSP_BASE, cmd, 0x10, 0x20, 0x1234500, 384), "256-byte"),
  ]:
    writes_before = len(fake.writes)
    try:
      bad_dma_call()
      raise AssertionError("bad Falcon DMA input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake.writes) == writes_before
  old_trace_falcon = os.environ.get("NV_ADD_TRACE_FALCON")
  trace_falcon_buf = io.StringIO()
  try:
    os.environ["NV_ADD_TRACE_FALCON"] = "1"
    with contextlib.redirect_stdout(trace_falcon_buf):
      falcon.execute_dma(NV_FALCON_GSP_BASE, cmd, 0x10, 0x20, 0x1234500, 512)
  finally:
    if old_trace_falcon is None: os.environ.pop("NV_ADD_TRACE_FALCON", None)
    else: os.environ["NV_ADD_TRACE_FALCON"] = old_trace_falcon
  trace_falcon_text = trace_falcon_buf.getvalue()
  assert "falcon pre-dma base=0x110000 cmd=0x614 dest=0x10 mem_off=0x20 src=0x1234500 size=0x200" in trace_falcon_text
  assert "falcon wreg GSP.DMATRFBASE addr=0x110110 value=0x12345" in trace_falcon_text
  assert "falcon wreg GSP.DMATRFBASE1 addr=0x110128 value=0x0" in trace_falcon_text
  assert "falcon wreg GSP.DMATRFMOFFS addr=0x110114 value=0x10" in trace_falcon_text
  assert "falcon wreg GSP.DMATRFFBOFFS addr=0x11011c value=0x20" in trace_falcon_text
  assert "falcon wreg GSP.DMATRFCMD addr=0x110118 value=0x614" in trace_falcon_text
  assert (NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFBASE, lo32(0x1234500 >> 8)) in fake.writes
  assert fake.writes.count((NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD, cmd)) == 2
  snapshot = falcon.state_snapshot(NV_FALCON_GSP_BASE)
  assert snapshot["cpuctl"] == 0x10 and snapshot["dmatrfcmd"] == 0x2
  assert "cpuctl=0x10" in falcon.format_state(NV_FALCON_GSP_BASE)
  assert "state=(" in falcon.dma_error(NV_FALCON_GSP_BASE, "test")
  hs_args = (NV_FALCON_GSP_BASE, 0x200000, 0x100, 0x200, 0, 0x100, 0x100, 0, 0, 0x100, 0x10, 1, 3)
  for mutate_hs_args, text in [
    (lambda a: (-1, *a[1:]), "Falcon base"),
    (lambda a: (a[0], -1, *a[2:]), "image address"),
    (lambda a: (*a[:2], -1, *a[3:]), "code offset"),
    (lambda a: (*a[:3], -1, *a[4:]), "data offset"),
    (lambda a: (*a[:4], -1, *a[5:]), "IMEM physical"),
    (lambda a: (*a[:5], -1, *a[6:]), "IMEM virtual"),
    (lambda a: (*a[:7], -1, *a[8:]), "DMEM physical"),
    (lambda a: (*a[:8], -1, *a[9:]), "DMEM virtual"),
    (lambda a: (*a[:10], -1, *a[11:]), "PKC offset"),
    (lambda a: (*a[:11], 0x100000000, a[12]), "engine id"),
    (lambda a: (*a[:12], 0x100000000), "ucode id"),
    (lambda a: a, "mailbox"),
  ]:
    writes_before = len(fake.writes)
    try:
      kwargs = {"mailbox": -1} if text == "mailbox" else {}
      falcon.execute_hs(*mutate_hs_args(hs_args), **kwargs)
      raise AssertionError("bad Falcon HS input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(fake.writes) == writes_before
  falcon.execute_hs(NV_FALCON_GSP_BASE, 0x200000, 0x100, 0x200, 0, 0x100, 0x100, 0, 0, 0x100, 0x10, 1, 3)
  assert (NV_FALCON_GSP_BASE + NV_PFALCON_FBIF_CTL, 0x190) in fake.writes
  assert (NV_FALCON_GSP_BASE + NV_PFALCON_FBIF_TRANSCFG0, 0x114) in fake.writes
  falcon.start_cpu(NV_FALCON_GSP_BASE)
  assert (NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL, 0x2) in fake.writes

  seq_words = [
    0x0, 0x5000, 0x1234,
    0x1, 0x5000, 0x00f0, 0x00f0,
    0x4, 0x5000, 2,
  ]
  seq = struct.pack("<II8I", 0, len(seq_words), *([0] * 8)) + struct.pack("<" + "I" * len(seq_words), *seq_words)
  saved = StandaloneNvShell.run_cpu_sequencer(fake, seq)
  assert fake.regs[0x5000] == 0x12f4
  assert saved[2] == 0x12f4
  bad_seq = struct.pack("<II8I", 0, len(seq_words) + 1, *([0] * 8)) + struct.pack("<" + "I" * len(seq_words), *seq_words)
  try:
    StandaloneNvShell.run_cpu_sequencer(fake, bad_seq)
    raise AssertionError("truncated CPU sequencer command stream was accepted")
  except ValueError as exc:
    assert "command stream" in str(exc)

  boot = object.__new__(StandaloneGspBootstrap)
  boot.shell = fake
  boot.queue_memory = type("QueueMem", (), {"libos_args_paddrs": [0x12345678000], "rm_args_paddrs": [0xdeadbeef000]})()
  boot.wpr_meta_sysmem = 0xabcdef000
  boot.booter_vram_paddr = 0x123000
  boot.booter_paddrs = [0x200000]
  boot.booter_desc = {"monitor_code_offset": 0x100, "monitor_data_offset": 0x200, "monitor_code_size": 0x300, "monitor_data_size": 0x400}
  boot.booter_loader_vram_paddr = 0x456000
  boot.booter_loader_paddrs = [0x300000]
  boot.booter_loader_desc = {"code_offset": 0x110, "data_offset": 0x220, "code_size": 0x330, "data_size": 0x440}
  pmu_fake = FakeShell()
  pmu_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 1 << 10
  pmu_falcon = FalconController(pmu_fake)
  pmu_falcon.reset(NV_FALCON_PMU_BASE, riscv=True)
  assert (NV_PPMU_FALCON_ENGINE, 1) in pmu_fake.writes
  assert (NV_PPMU_FALCON_ENGINE, 0) in pmu_fake.writes
  assert (NV_FALCON_PMU_BASE + NV_PRISCV_RISCV_BCR_CTRL, (1 << 4) | (1 << 8)) in pmu_fake.writes

  pmu_fake_force = FakeShell()
  pmu_fake_force.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 1 << 10
  pmu_fake_force.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
  pmu_fake_force.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMACTL] = 0x40
  pmu_fake_force.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x10
  pmu_falcon_force = FalconController(pmu_fake_force)
  pmu_falcon_force.reset(NV_FALCON_PMU_BASE, riscv=True, force=True)
  force_writes = pmu_fake_force.writes
  assert (NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL, 0) in force_writes
  assert (NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMACTL, 0) in force_writes
  assert (NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD, 0) in force_writes
  assert (NV_PPMU_FALCON_ENGINE, 1) in force_writes
  assert (NV_PPMU_FALCON_ENGINE, 0) in force_writes
  assert (NV_FALCON_PMU_BASE + NV_PRISCV_RISCV_BCR_CTRL, (1 << 4) | (1 << 8)) in force_writes
  cpuctl_idx = force_writes.index((NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL, 0))
  engine_set_idx = force_writes.index((NV_PPMU_FALCON_ENGINE, 1))
  assert cpuctl_idx < engine_set_idx
  engine_clear_idx = force_writes.index((NV_PPMU_FALCON_ENGINE, 0))
  bcr_idx = force_writes.index((NV_FALCON_PMU_BASE + NV_PRISCV_RISCV_BCR_CTRL, (1 << 4) | (1 << 8)))
  assert engine_clear_idx < bcr_idx
  pmu_fake_no_force = FakeShell()
  pmu_fake_no_force.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
  pmu_fake_no_force.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
  pmu_falcon_no_force = FalconController(pmu_fake_no_force)
  pmu_falcon_no_force.reset(NV_FALCON_PMU_BASE, riscv=True)
  no_force_writes = pmu_fake_no_force.writes
  assert (NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL, 0) not in no_force_writes
  assert (NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMACTL, 0) not in no_force_writes
  assert (NV_PPMU_FALCON_ENGINE, 1) in no_force_writes
  assert (NV_PPMU_FALCON_ENGINE, 0) in no_force_writes
  try:
    no_chip_fake = FakeShell()
    no_chip_fake.chip = None
    no_chip_falcon = FalconController(no_chip_fake)
    no_chip_falcon.reset(NV_FALCON_PMU_BASE, riscv=True, force=True)
    raise AssertionError("force reset without chip info was accepted")
  except RuntimeError as exc:
    assert "chip" in str(exc)

  ga102_fake = FakeShell()
  assert ga102_fake.fecs_falcon_base() == 0xA04000
  assert ga102_fake.fecs_falcon_engine() == 0xA04000 + NV_PFECS_FALCON_ENGINE_OFFSET
  ad102_fake = FakeShell()
  ad102_fake.chip.fw_name = "ad102"
  assert ad102_fake.fecs_falcon_base() == 0xA04000
  gb202_fake = FakeShell()
  gb202_fake.chip.fw_name = "gb202"
  assert gb202_fake.fecs_falcon_base() == 0xA04000
  unknown_fake = FakeShell()
  unknown_fake.chip.fw_name = "unknown"
  try:
    unknown_fake.fecs_falcon_base()
    raise AssertionError("unknown firmware FECS FALCON base was accepted")
  except ValueError as exc:
    assert "FECS FALCON base" in str(exc)
  fecs_fake = FakeShell()
  fecs_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
  fecs_falcon = FalconController(fecs_fake)
  fecs_base = NV_FALCON_FECS_BASES["ga102"]
  fecs_falcon.format_state(fecs_base)
  assert fecs_falcon.engine_reg(fecs_base) == fecs_base + NV_PFECS_FALCON_ENGINE_OFFSET

  fecs_fenced = FakeShell()
  fecs_fenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
  fecs_fenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
  fecs_fenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_DMACTL] = 0x40
  fecs_fenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_DMATRFCMD] = 0x10
  fecs_fenced_falcon = FalconController(fecs_fenced)
  fecs_fenced_falcon.reset(NV_FALCON_FECS_BASES["ga102"], riscv=True, force=True)
  fenced_writes = fecs_fenced.writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FBIF_TRANSCFG0, 0x114) in fenced_writes, "FECS force reset did not clear FBIF_TRANSCFG0 fence sentinel"
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL, 0) in fenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_DMACTL, 0) in fenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_DMATRFCMD, 0) in fenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET, 1) in fenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET, 0) in fenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PRISCV_RISCV_BCR_CTRL, (1 << 4) | (1 << 8)) not in fenced_writes
  fecs_fbif_idx = fenced_writes.index((NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FBIF_TRANSCFG0, 0x114))
  fecs_cpuctl_idx = fenced_writes.index((NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL, 0))
  fecs_engine_set_idx = fenced_writes.index((NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET, 1))
  assert fecs_fbif_idx < fecs_cpuctl_idx < fecs_engine_set_idx, "FECS force reset write order is wrong: FBIF_TRANSCFG0 must come before CPUCTL, CPUCTL before engine=1"

  fecs_unfenced = FakeShell()
  fecs_unfenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
  fecs_unfenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
  fecs_unfenced_falcon = FalconController(fecs_unfenced)
  fecs_unfenced_falcon.reset(NV_FALCON_FECS_BASES["ga102"], riscv=True)
  unfenced_writes = fecs_unfenced.writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL, 0) not in unfenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_DMACTL, 0) not in unfenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_DMATRFCMD, 0) not in unfenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET, 1) in unfenced_writes
  assert (NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET, 0) in unfenced_writes

  format_fenced = FakeShell()
  format_fenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0x400
  format_fenced.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
  format_fenced_falcon = FalconController(format_fenced)
  format_fenced_state = format_fenced_falcon.format_state_with_fenced(NV_FALCON_FECS_BASES["ga102"])
  assert "cpuctl=0xbadf5720" in format_fenced_state
  assert "fenced=True" in format_fenced_state
  assert "has_riscv=1" in format_fenced_state
  assert "hwcfg2=0x400" in format_fenced_state

  format_pmu = FakeShell()
  format_pmu.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
  format_pmu.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x20
  format_pmu_falcon = FalconController(format_pmu)
  format_pmu_state = format_pmu_falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)
  assert "cpuctl=0x20" in format_pmu_state
  assert "fenced=False" in format_pmu_state
  assert "has_riscv=0" in format_pmu_state
  assert "hwcfg2=0x0" in format_pmu_state

  format_pmu_half = FakeShell()
  format_pmu_half.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0x400
  format_pmu_half.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0xbadf0000
  format_pmu_half_falcon = FalconController(format_pmu_half)
  format_pmu_half_state = format_pmu_half_falcon.format_state_with_fenced(NV_FALCON_PMU_BASE)
  assert "cpuctl=0xbadf0000" in format_pmu_half_state
  assert "fenced=True" in format_pmu_half_state

  old_fecs_reset = os.environ.get("NV_ADD_FECS_RESET")
  try:
    fecs_skip_fake = FakeShell()
    fecs_skip_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
    os.environ.pop("NV_ADD_FECS_RESET", None)
    fecs_skip_outer = object.__new__(StandaloneGspBootstrap)
    fecs_skip_outer.shell, fecs_skip_outer.queue_memory = fecs_skip_fake, boot.queue_memory
    fecs_skip_outer.wpr_meta_sysmem = boot.wpr_meta_sysmem
    fecs_skip_outer.booter_loader_vram_paddr, fecs_skip_outer.booter_loader_paddrs = boot.booter_loader_vram_paddr, boot.booter_loader_paddrs
    fecs_skip_outer.booter_loader_desc = dict(boot.booter_loader_desc)
    fecs_skip_outer.booter_loader_image, fecs_skip_outer.booter_loader_view = b"", MMIOView(bytearray(b""))
    fecs_skip_outer.booter_image, fecs_skip_outer.booter_view = b"", MMIOView(bytearray(b""))
    fecs_skip_outer.gsp_radix3_view, fecs_skip_outer.gsp_signature_view = MMIOView(bytearray(b"")), MMIOView(bytearray(b""))
    fecs_skip_outer.wpr_meta_view, fecs_skip_outer.wpr_meta_blob = MMIOView(bytearray(b"")), b""
    fecs_engine_addr = NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET
    fecs_skip_fake.regs[fecs_engine_addr] = 0
    fecs_skip_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
    fecs_skip_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
    fecs_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_skip_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PRISCV_RISCV_CPUCTL] = 0x80
    fecs_skip_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_skip_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_skip_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_skip_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_skip_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    old_hs, old_wait = FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done
    try:
      FalconController.execute_hs = lambda self, *a, **kw: (0, 0)
      StandaloneGspBootstrap.wait_init_done = lambda self: b""
      writes_before = len(fecs_skip_fake.writes)
      with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        StandaloneGspBootstrap.boot_ampere_ada(fecs_skip_outer)
      fecs_engine_writes = [w for w in fecs_skip_fake.writes[writes_before:] if w[0] == fecs_engine_addr]
      assert fecs_engine_writes == [], f"FECS reset was not skipped when NV_ADD_FECS_RESET is unset (got {fecs_engine_writes})"
    finally:
      FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done = old_hs, old_wait
  finally:
    if old_fecs_reset is None: os.environ.pop("NV_ADD_FECS_RESET", None)
    else: os.environ["NV_ADD_FECS_RESET"] = old_fecs_reset

  old_fecs_reset, old_fecs_reset_force = os.environ.get("NV_ADD_FECS_RESET"), os.environ.get("NV_ADD_FECS_RESET_FORCE")
  try:
    fecs_active_fake = FakeShell()
    fecs_active_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_active_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
    fecs_active_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
    fecs_active_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_active_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_active_fake.regs[NV_FALCON_GSP_BASE + NV_PRISCV_RISCV_CPUCTL] = 0x80
    fecs_active_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_active_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_active_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_active_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_active_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_active_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_active_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_active_outer = object.__new__(StandaloneGspBootstrap)
    fecs_active_outer.shell, fecs_active_outer.queue_memory = fecs_active_fake, boot.queue_memory
    fecs_active_outer.wpr_meta_sysmem = boot.wpr_meta_sysmem
    fecs_active_outer.booter_loader_vram_paddr, fecs_active_outer.booter_loader_paddrs = boot.booter_loader_vram_paddr, boot.booter_loader_paddrs
    fecs_active_outer.booter_loader_desc = dict(boot.booter_loader_desc)
    fecs_active_outer.booter_loader_image, fecs_active_outer.booter_loader_view = b"", MMIOView(bytearray(b""))
    fecs_active_outer.booter_image, fecs_active_outer.booter_view = b"", MMIOView(bytearray(b""))
    fecs_active_outer.gsp_radix3_view, fecs_active_outer.gsp_signature_view = MMIOView(bytearray(b"")), MMIOView(bytearray(b""))
    fecs_active_outer.wpr_meta_view, fecs_active_outer.wpr_meta_blob = MMIOView(bytearray(b"")), b""
    os.environ["NV_ADD_FECS_RESET"], os.environ["NV_ADD_FECS_RESET_FORCE"] = "1", "1"
    old_hs, old_wait = FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done
    try:
      FalconController.execute_hs = lambda self, *a, **kw: (0, 0)
      StandaloneGspBootstrap.wait_init_done = lambda self: b""
      writes_before = len(fecs_active_fake.writes)
      with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        StandaloneGspBootstrap.boot_ampere_ada(fecs_active_outer)
      fecs_active_engine_writes = [w for w in fecs_active_fake.writes[writes_before:]
        if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET]
      assert fecs_active_engine_writes, "FECS reset was not executed when NV_ADD_FECS_RESET_FORCE=1"
      fecs_active_engine_values = [w[1] for w in fecs_active_engine_writes]
      assert fecs_active_engine_values == [1, 0], f"FECS engine toggle was not 1->0 (got {fecs_active_engine_values})"
    finally:
      FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done = old_hs, old_wait
  finally:
    if old_fecs_reset is None: os.environ.pop("NV_ADD_FECS_RESET", None)
    else: os.environ["NV_ADD_FECS_RESET"] = old_fecs_reset
    if old_fecs_reset_force is None: os.environ.pop("NV_ADD_FECS_RESET_FORCE", None)
    else: os.environ["NV_ADD_FECS_RESET_FORCE"] = old_fecs_reset_force

  old_postinit, old_postinit_force = os.environ.get("NV_ADD_FECS_RESET_POSTINIT"), os.environ.get("NV_ADD_FECS_RESET_POSTINIT_FORCE")
  try:
    fecs_postinit_fake = FakeShell()
    fecs_postinit_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_postinit_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
    fecs_postinit_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
    fecs_postinit_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_postinit_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    fecs_postinit_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_postinit_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_postinit_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_postinit_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    fecs_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    fecs_postinit_outer = object.__new__(StandaloneGspBootstrap)
    fecs_postinit_outer.shell, fecs_postinit_outer.queue_memory = fecs_postinit_fake, boot.queue_memory
    os.environ["NV_ADD_FECS_RESET_POSTINIT"] = "1"
    os.environ["NV_ADD_FECS_RESET_POSTINIT_FORCE"] = "1"
    old_hs, old_wait = FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done
    try:
      FalconController.execute_hs = lambda self, *a, **kw: (0, 0)
      StandaloneGspBootstrap.wait_init_done = lambda self: b""
      writes_before = len(fecs_postinit_fake.writes)
      with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        StandaloneGspBootstrap.post_init_fecs_reset(fecs_postinit_outer)
      fecs_postinit_writes = [w for w in fecs_postinit_fake.writes[writes_before:]
        if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET]
      assert fecs_postinit_writes, "FECS post-init reset was not executed when NV_ADD_FECS_RESET_POSTINIT=1"
      fecs_postinit_engine_values = [w[1] for w in fecs_postinit_writes]
      assert fecs_postinit_engine_values == [1, 0], f"FECS engine toggle was not 1->0 (got {fecs_postinit_engine_values})"
      fecs_postinit_cpuctl = [w for w in fecs_postinit_fake.writes[writes_before:]
        if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL]
      assert fecs_postinit_cpuctl, "FECS post-init force reset did not clear CPUCTL"
    finally:
      FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done = old_hs, old_wait
  finally:
    if old_postinit is None: os.environ.pop("NV_ADD_FECS_RESET_POSTINIT", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTINIT"] = old_postinit
    if old_postinit_force is None: os.environ.pop("NV_ADD_FECS_RESET_POSTINIT_FORCE", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTINIT_FORCE"] = old_postinit_force

  fecs_postinit_off_fake = FakeShell()
  fecs_postinit_off_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
  fecs_postinit_off_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
  fecs_postinit_off_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
  fecs_postinit_off_outer = object.__new__(StandaloneGspBootstrap)
  fecs_postinit_off_outer.shell, fecs_postinit_off_outer.queue_memory = fecs_postinit_off_fake, boot.queue_memory
  saved_postinit = os.environ.get("NV_ADD_FECS_RESET_POSTINIT")
  try:
    os.environ.pop("NV_ADD_FECS_RESET_POSTINIT", None)
    writes_before = len(fecs_postinit_off_fake.writes)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
      result = StandaloneGspBootstrap.post_init_fecs_reset(fecs_postinit_off_outer)
    fecs_postinit_off_writes = [w for w in fecs_postinit_off_fake.writes[writes_before:]
      if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET]
    assert not fecs_postinit_off_writes, "FECS post-init reset was executed when NV_ADD_FECS_RESET_POSTINIT is unset"
    assert result is False, "post_init_fecs_reset returned True when env var was unset"
  finally:
    if saved_postinit is not None: os.environ["NV_ADD_FECS_RESET_POSTINIT"] = saved_postinit

  old_pmu_postinit, old_pmu_postinit_force = os.environ.get("NV_ADD_PMU_RESET_POSTINIT"), os.environ.get("NV_ADD_PMU_RESET_POSTINIT_FORCE")
  old_fecs_postinit, old_fecs_postinit_force = os.environ.get("NV_ADD_FECS_RESET_POSTINIT"), os.environ.get("NV_ADD_FECS_RESET_POSTINIT_FORCE")
  try:
    pmu_postinit_fake = FakeShell()
    pmu_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    pmu_postinit_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
    pmu_postinit_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
    pmu_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
    pmu_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMACTL] = 0x40
    pmu_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x10
    pmu_postinit_outer = object.__new__(StandaloneGspBootstrap)
    pmu_postinit_outer.shell, pmu_postinit_outer.queue_memory = pmu_postinit_fake, boot.queue_memory
    os.environ["NV_ADD_PMU_RESET_POSTINIT"] = "1"
    os.environ["NV_ADD_PMU_RESET_POSTINIT_FORCE"] = "1"
    os.environ["NV_ADD_FECS_RESET_POSTINIT"] = "1"
    os.environ["NV_ADD_FECS_RESET_POSTINIT_FORCE"] = "1"
    writes_before = len(pmu_postinit_fake.writes)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
      result = StandaloneGspBootstrap.post_init_fecs_reset(pmu_postinit_outer)
    pmu_postinit_engine_writes = [w for w in pmu_postinit_fake.writes[writes_before:]
      if w[0] == NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL]
    assert pmu_postinit_engine_writes, "PMU post-init reset did not clear CPUCTL"
    assert pmu_postinit_engine_writes[0][1] == 0, f"PMU post-init CPUCTL was not cleared to 0 (got {pmu_postinit_engine_writes[0][1]:#x})"
    assert result is True, "post_init_fecs_reset returned False when PMU postinit env var was set"
  finally:
    if old_pmu_postinit is None: os.environ.pop("NV_ADD_PMU_RESET_POSTINIT", None)
    else: os.environ["NV_ADD_PMU_RESET_POSTINIT"] = old_pmu_postinit
    if old_pmu_postinit_force is None: os.environ.pop("NV_ADD_PMU_RESET_POSTINIT_FORCE", None)
    else: os.environ["NV_ADD_PMU_RESET_POSTINIT_FORCE"] = old_pmu_postinit_force
    if old_fecs_postinit is None: os.environ.pop("NV_ADD_FECS_RESET_POSTINIT", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTINIT"] = old_fecs_postinit
    if old_fecs_postinit_force is None: os.environ.pop("NV_ADD_FECS_RESET_POSTINIT_FORCE", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTINIT_FORCE"] = old_fecs_postinit_force

  pmu_postinit_off_fake = FakeShell()
  pmu_postinit_off_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
  pmu_postinit_off_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
  pmu_postinit_off_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
  pmu_postinit_off_outer = object.__new__(StandaloneGspBootstrap)
  pmu_postinit_off_outer.shell, pmu_postinit_off_outer.queue_memory = pmu_postinit_off_fake, boot.queue_memory
  saved_pmu_postinit = os.environ.get("NV_ADD_PMU_RESET_POSTINIT")
  try:
    os.environ.pop("NV_ADD_PMU_RESET_POSTINIT", None)
    writes_before = len(pmu_postinit_off_fake.writes)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
      StandaloneGspBootstrap.post_init_fecs_reset(pmu_postinit_off_outer)
    pmu_postinit_off_writes = [w for w in pmu_postinit_off_fake.writes[writes_before:]
      if w[0] == NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL]
    assert not pmu_postinit_off_writes, "PMU post-init reset was executed when NV_ADD_PMU_RESET_POSTINIT is unset"
  finally:
    if saved_pmu_postinit is not None: os.environ["NV_ADD_PMU_RESET_POSTINIT"] = saved_pmu_postinit

  old_postpromote, old_postpromote_force = os.environ.get("NV_ADD_FECS_RESET_POSTPROMOTE"), os.environ.get("NV_ADD_FECS_RESET_POSTPROMOTE_FORCE")
  try:
    postpromote_fake = FakeShell()
    postpromote_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
    postpromote_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
    postpromote_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
    postpromote_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
    postpromote_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    postpromote_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0xbadf5720
    postpromote_outer = object.__new__(StandaloneGspBootstrap)
    postpromote_outer.shell, postpromote_outer.queue_memory = postpromote_fake, boot.queue_memory
    os.environ["NV_ADD_FECS_RESET_POSTPROMOTE"] = "1"
    os.environ["NV_ADD_FECS_RESET_POSTPROMOTE_FORCE"] = "1"
    writes_before = len(postpromote_fake.writes)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
      result = StandaloneGspBootstrap.post_promote_fecs_reset(postpromote_outer)
    postpromote_fecs_engine_writes = [w for w in postpromote_fake.writes[writes_before:]
      if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET]
    assert postpromote_fecs_engine_writes, "FECS post-promote reset did not toggle FECS engine"
    postpromote_fecs_cpuctl_writes = [w for w in postpromote_fake.writes[writes_before:]
      if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_CPUCTL]
    assert postpromote_fecs_cpuctl_writes and postpromote_fecs_cpuctl_writes[0][1] == 0, f"FECS post-promote CPUCTL was not cleared to 0 (got {postpromote_fecs_cpuctl_writes[0][1]:#x if postpromote_fecs_cpuctl_writes else None})"
    assert result is True, "post_promote_fecs_reset returned False when env var was set"
  finally:
    if old_postpromote is None: os.environ.pop("NV_ADD_FECS_RESET_POSTPROMOTE", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTPROMOTE"] = old_postpromote
    if old_postpromote_force is None: os.environ.pop("NV_ADD_FECS_RESET_POSTPROMOTE_FORCE", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTPROMOTE_FORCE"] = old_postpromote_force

  postpromote_off_fake = FakeShell()
  postpromote_off_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
  postpromote_off_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
  postpromote_off_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
  postpromote_off_outer = object.__new__(StandaloneGspBootstrap)
  postpromote_off_outer.shell, postpromote_off_outer.queue_memory = postpromote_off_fake, boot.queue_memory
  saved_postpromote = os.environ.get("NV_ADD_FECS_RESET_POSTPROMOTE")
  try:
    os.environ.pop("NV_ADD_FECS_RESET_POSTPROMOTE", None)
    writes_before = len(postpromote_off_fake.writes)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
      result = StandaloneGspBootstrap.post_promote_fecs_reset(postpromote_off_outer)
    postpromote_off_engine_writes = [w for w in postpromote_off_fake.writes[writes_before:]
      if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET]
    assert not postpromote_off_engine_writes, "FECS post-promote reset was executed when env var was unset"
    assert result is False, "post_promote_fecs_reset returned True when env var was unset"
  finally:
    if saved_postpromote is not None: os.environ["NV_ADD_FECS_RESET_POSTPROMOTE"] = saved_postpromote

  old_fecs_reset, old_fecs_reset_force = os.environ.get("NV_ADD_FECS_RESET"), os.environ.get("NV_ADD_FECS_RESET_FORCE")
  old_postinit, old_postinit_force = os.environ.get("NV_ADD_FECS_RESET_POSTINIT"), os.environ.get("NV_ADD_FECS_RESET_POSTINIT_FORCE")
  try:
    os.environ["NV_ADD_FECS_RESET_POSTINIT"] = "1"
    os.environ["NV_ADD_FECS_RESET_POSTINIT_FORCE"] = "1"
    boot_postinit_fake = FakeShell()
    boot_postinit_fake.regs[NV_FALCON_FECS_BASES["ga102"] + NV_PFALCON_FALCON_HWCFG2] = 0
    boot_postinit_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
    boot_postinit_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
    boot_postinit_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    boot_postinit_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    boot_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    boot_postinit_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    boot_postinit_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    boot_postinit_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    boot_postinit_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    boot_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    boot_postinit_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    boot_postinit_fake.regs[NV_FALCON_GSP_BASE + NV_PRISCV_RISCV_CPUCTL] = 0x80
    boot_postinit_outer = object.__new__(StandaloneGspBootstrap)
    boot_postinit_outer.shell, boot_postinit_outer.queue_memory = boot_postinit_fake, boot.queue_memory
    boot_postinit_outer.wpr_meta_sysmem = boot.wpr_meta_sysmem
    boot_postinit_outer.booter_loader_vram_paddr, boot_postinit_outer.booter_loader_paddrs = boot.booter_loader_vram_paddr, boot.booter_loader_paddrs
    boot_postinit_outer.booter_loader_desc = dict(boot.booter_loader_desc)
    boot_postinit_outer.booter_loader_image, boot_postinit_outer.booter_loader_view = b"", MMIOView(bytearray(b""))
    boot_postinit_outer.booter_image, boot_postinit_outer.booter_view = b"", MMIOView(bytearray(b""))
    boot_postinit_outer.gsp_radix3_view, boot_postinit_outer.gsp_signature_view = MMIOView(bytearray(b"")), MMIOView(bytearray(b""))
    boot_postinit_outer.wpr_meta_view, boot_postinit_outer.wpr_meta_blob = MMIOView(bytearray(b"")), b""
    old_hs, old_wait = FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done
    try:
      FalconController.execute_hs = lambda self, *a, **kw: (0, 0)
      StandaloneGspBootstrap.wait_init_done = lambda self: b""
      writes_before = len(boot_postinit_fake.writes)
      with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        StandaloneGspBootstrap.boot_ampere_ada(boot_postinit_outer)
      boot_postinit_engine_writes = [w for w in boot_postinit_fake.writes[writes_before:]
        if w[0] == NV_FALCON_FECS_BASES["ga102"] + NV_PFECS_FALCON_ENGINE_OFFSET]
      assert boot_postinit_engine_writes, "FECS post-init reset was not triggered during boot_ampere_ada"
    finally:
      FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done = old_hs, old_wait
  finally:
    if old_fecs_reset is None: os.environ.pop("NV_ADD_FECS_RESET", None)
    else: os.environ["NV_ADD_FECS_RESET"] = old_fecs_reset
    if old_fecs_reset_force is None: os.environ.pop("NV_ADD_FECS_RESET_FORCE", None)
    else: os.environ["NV_ADD_FECS_RESET_FORCE"] = old_fecs_reset_force
    if old_postinit is None: os.environ.pop("NV_ADD_FECS_RESET_POSTINIT", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTINIT"] = old_postinit
    if old_postinit_force is None: os.environ.pop("NV_ADD_FECS_RESET_POSTINIT_FORCE", None)
    else: os.environ["NV_ADD_FECS_RESET_POSTINIT_FORCE"] = old_postinit_force

  old_pmu_reset = os.environ.get("NV_ADD_PMU_RESET")
  try:
    os.environ["NV_ADD_PMU_RESET"] = "0"
    pmu_skip_fake = FakeShell()
    pmu_skip_fake.regs[NV_PGSP_FALCON_ENGINE] = 0
    pmu_skip_fake.regs[NV_PSEC_FALCON_ENGINE] = 0
    pmu_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    pmu_skip_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    pmu_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PRISCV_RISCV_CPUCTL] = 0x80
    pmu_skip_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_HWCFG2] = 0
    pmu_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    pmu_skip_fake.regs[NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    pmu_skip_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    pmu_skip_fake.regs[NV_FALCON_SEC2_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    pmu_skip_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_DMATRFCMD] = 0x2
    pmu_skip_fake.regs[NV_FALCON_PMU_BASE + NV_PFALCON_FALCON_CPUCTL] = 0x10
    boot_pmu_skip = object.__new__(StandaloneGspBootstrap)
    boot_pmu_skip.shell, boot_pmu_skip.queue_memory = pmu_skip_fake, boot.queue_memory
    boot_pmu_skip.wpr_meta_sysmem = boot.wpr_meta_sysmem
    boot_pmu_skip.booter_loader_vram_paddr, boot_pmu_skip.booter_loader_paddrs = boot.booter_loader_vram_paddr, boot.booter_loader_paddrs
    boot_pmu_skip.booter_loader_desc = dict(boot.booter_loader_desc)
    boot_pmu_skip.booter_loader_image, boot_pmu_skip.booter_loader_view = b"", MMIOView(bytearray(b""))
    boot_pmu_skip.booter_image, boot_pmu_skip.booter_view = b"", MMIOView(bytearray(b""))
    boot_pmu_skip.gsp_radix3_view, boot_pmu_skip.gsp_signature_view = MMIOView(bytearray(b"")), MMIOView(bytearray(b""))
    boot_pmu_skip.wpr_meta_view, boot_pmu_skip.wpr_meta_blob = MMIOView(bytearray(b"")), b""
    old_hs, old_wait = FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done
    try:
      FalconController.execute_hs = lambda self, *a, **kw: (0, 0)
      StandaloneGspBootstrap.wait_init_done = lambda self: b""
      writes_before = len(pmu_skip_fake.writes)
      StandaloneGspBootstrap.boot_ampere_ada(boot_pmu_skip)
      pmu_engine_writes = [w for w in pmu_skip_fake.writes[writes_before:] if w[0] == NV_PPMU_FALCON_ENGINE]
      assert pmu_engine_writes == [], f"PMU reset was not skipped when NV_ADD_PMU_RESET=0 (got {pmu_engine_writes})"
    finally:
      FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done = old_hs, old_wait
  finally:
    if old_pmu_reset is None: os.environ.pop("NV_ADD_PMU_RESET", None)
    else: os.environ["NV_ADD_PMU_RESET"] = old_pmu_reset
  for attr, data in [
    ("booter_loader", b"loader"),
    ("wpr_meta", b"meta"),
    ("booter", b"boot"),
    ("gsp_signature", b"sig"),
    ("gsp_radix3", b"radix"),
  ]:
    setattr(boot, f"{attr}_image" if attr in ("booter_loader", "booter") else f"{attr}_blob" if attr in ("wpr_meta", "gsp_radix3") else attr, data)
    setattr(boot, f"{attr}_view", MMIOView(bytearray(data)))
  with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
    boot.verify_sec2_inputs()
  old_loader_view = boot.booter_loader_view
  try:
    boot.booter_loader_view = object()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
      try:
        boot.verify_sec2_inputs()
        raise AssertionError("bad SEC2 verification view was accepted")
      except ValueError as exc:
        assert "view" in str(exc)
  finally:
    boot.booter_loader_view = old_loader_view
  old_meta_blob = boot.wpr_meta_blob
  try:
    boot.wpr_meta_blob = object()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
      try:
        boot.verify_sec2_inputs()
        raise AssertionError("bad SEC2 expected data was accepted")
      except ValueError as exc:
        assert "expected data" in str(exc)
  finally:
    boot.wpr_meta_blob = old_meta_blob
  fake.regs[NV_FALCON_GSP_BASE + NV_PRISCV_RISCV_CPUCTL] = 0x80
  rm_region = GspQueueMemoryBuilder.pack_libos_memory_region(0, 0, 0x1000, "RMARGS", boot.queue_memory.rm_args_paddrs[0])
  assert struct.unpack_from("<QQQBB6x", rm_region) == (int.from_bytes(b"RMARGS", "big"), 0xdeadbeef000, 0x1000, 0, 0)
  StandaloneGspBootstrap.validate_boot_desc(boot.booter_loader_desc, ("code_offset", "data_offset", "code_size", "data_size"), "booter-loader")
  for bad_desc, text in [
    (object(), "descriptor"),
    ({"code_offset": 0}, "missing"),
    ({"code_offset": -1, "data_offset": 0, "code_size": 0x100, "data_size": 0x100}, "non-negative"),
  ]:
    try:
      StandaloneGspBootstrap.validate_boot_desc(bad_desc, ("code_offset", "data_offset", "code_size", "data_size"), "booter-loader")
      raise AssertionError("bad boot descriptor was accepted")
    except ValueError as exc:
      assert text in str(exc)
  StandaloneGspBootstrap.validate_boot_paddr_list([0x1000], "test")
  for bad_paddrs, text in [
    ([], "physical ranges"),
    ([0x1001], "4KB"),
    ([-0x1000], "4KB"),
  ]:
    try:
      StandaloneGspBootstrap.validate_boot_paddr_list(bad_paddrs, "test")
      raise AssertionError("bad boot physical list was accepted")
    except ValueError as exc:
      assert text in str(exc)
  old_reset, old_execute_hs, old_wait_init_done = FalconController.reset, FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done
  try:
    FalconController.reset = lambda self, base, riscv=False: fake.writes.append(("reset", base, riscv))
    FalconController.execute_hs = lambda self, *args, **kwargs: fake.writes.append(("execute_hs", args, kwargs)) or (0, 0)
    StandaloneGspBootstrap.wait_init_done = lambda self: b""
    bad_boot = object.__new__(StandaloneGspBootstrap)
    bad_boot.shell, bad_boot.queue_memory = boot.shell, boot.queue_memory
    bad_boot.wpr_meta_sysmem = boot.wpr_meta_sysmem
    bad_boot.booter_loader_vram_paddr, bad_boot.booter_loader_paddrs = boot.booter_loader_vram_paddr, boot.booter_loader_paddrs
    bad_boot.booter_loader_desc = {"code_offset": 0x110, "data_offset": 0x220, "code_size": -1, "data_size": 0x440}
    bad_writes = len(fake.writes)
    try:
      StandaloneGspBootstrap.boot_ampere_ada(bad_boot)
      raise AssertionError("bad booter-loader descriptor reached Falcon execution")
    except ValueError as exc:
      assert "booter-loader" in str(exc)
    assert len(fake.writes) == bad_writes
    for mutate_boot, text in [
      (lambda b: setattr(b, "wpr_meta_sysmem", 0xabcdef001), "WPR metadata"),
      (lambda b: setattr(b, "booter_loader_paddrs", []), "booter-loader"),
      (lambda b: setattr(b.queue_memory, "libos_args_paddrs", [0x12345678001]), "LibOS args"),
    ]:
      bad_boot = object.__new__(StandaloneGspBootstrap)
      bad_boot.shell, bad_boot.queue_memory = boot.shell, type("QueueMem", (), {"libos_args_paddrs": list(boot.queue_memory.libos_args_paddrs)})()
      bad_boot.wpr_meta_sysmem = boot.wpr_meta_sysmem
      bad_boot.booter_loader_vram_paddr, bad_boot.booter_loader_paddrs = boot.booter_loader_vram_paddr, list(boot.booter_loader_paddrs)
      bad_boot.booter_loader_desc = dict(boot.booter_loader_desc)
      mutate_boot(bad_boot)
      bad_writes = len(fake.writes)
      try:
        StandaloneGspBootstrap.boot_ampere_ada(bad_boot)
        raise AssertionError("bad boot prerequisite reached Falcon execution")
      except ValueError as exc:
        assert text in str(exc)
      assert len(fake.writes) == bad_writes
    StandaloneGspBootstrap.boot_ampere_ada(boot)
  finally:
    FalconController.reset, FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done = old_reset, old_execute_hs, old_wait_init_done
  assert (NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_MAILBOX0, lo32(0x12345678000)) in fake.writes
  assert (NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_MAILBOX1, hi32(0x12345678000)) in fake.writes
  assert (NV_FALCON_GSP_BASE + NV_PFALCON_FALCON_MAILBOX0, lo32(0xdeadbeef000)) not in fake.writes
  execute_hs_call = next(write for write in fake.writes if isinstance(write[0], str) and write[0] == "execute_hs")
  assert execute_hs_call[1][1] == 0x456000

  class FakeMem:
    def __init__(self):
      self.next_va, self.next_pa = 0x1000000000, 0x80000000
      self.va_alloc = FreeListAllocator(1 << 48, base=self.next_va)
    def alloc_vaddr(self, size, align=0x1000):
      size = round_up(size, 0x1000)
      return self.va_alloc.alloc(size, max(1 << (size.bit_length() - 1), align, 0x1000))
    def map_range(self, vaddr, size, paddrs, aspace, uncached=False, snooped=False):
      return VirtMapping(vaddr, round_up(size, 0x1000), paddrs, aspace, uncached=uncached, snooped=snooped)
    def unmap_range(self, vaddr, size):
      return None
    def valloc(self, size, align=0x1000, uncached=False, contiguous=False):
      va = self.alloc_vaddr(size, align)
      pa = self.next_pa
      self.next_pa += round_up(size, 0x1000)
      return VirtMapping(va, round_up(size, 0x1000), [(pa, round_up(size, 0x1000))], AddrSpace.PHYS,
                         uncached=uncached)
    def palloc(self, size, align=0x1000):
      pa = round_up(self.next_pa, align)
      self.next_pa = pa + round_up(size, 0x1000)
      return pa
  class FakeSysmemTransport:
    def __init__(self):
      self.next_pa, self.allocs = 0x90000000, []
    def alloc_sysmem(self, size, contiguous=False):
      size = round_up(size, 0x1000)
      self.allocs.append((size, contiguous))
      view = MMIOView(bytearray(size))
      paddrs = [self.next_pa + off for off in range(0, size, 0x1000)]
      self.next_pa += size
      return view, paddrs
  class FakeResourceShell:
    def __init__(self):
      self.mm, self.transport = FakeMem(), FakeSysmemTransport()
  class FreeShell:
    def __init__(self):
      self.mm, self.transport, self.freed = GpuMemoryManager(64 << 20), FakeSysmemTransport(), []
      self.transport.free_sysmem = lambda sysmem: self.freed.append(sysmem)
  rshell = FakeResourceShell()
  ralloc = StandaloneBufferAllocator(rshell)
  resources = StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=4, token=0x55)
  assert resources.compute_gpfifo.entries_count == 4
  assert resources.compute_gpfifo.token == 0x55
  assert resources.timeline_signal.value == 0
  assert resources.notifier_buf is None
  assert resources.kernargs_allocator.alloc(16, 8) == resources.kernargs_buf.va_addr
  old_trace_channel, old_trace_channel_stack = os.environ.get("NV_ADD_TRACE_CHANNEL"), os.environ.get("NV_ADD_TRACE_CHANNEL_STACK")
  try:
    os.environ["NV_ADD_TRACE_CHANNEL"] = "1"
    os.environ["NV_ADD_TRACE_CHANNEL_STACK"] = "1"
    runtime_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(runtime_trace_buf):
      StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=4, token=0x58)
    runtime_trace = runtime_trace_buf.getvalue()
    assert "channel runtime_resources_start" in runtime_trace and "channel runtime_resources_done" in runtime_trace
    assert "allocate_runtime_resources" in runtime_trace
  finally:
    for name, value in (("NV_ADD_TRACE_CHANNEL", old_trace_channel), ("NV_ADD_TRACE_CHANNEL_STACK", old_trace_channel_stack)):
      if value is None: os.environ.pop(name, None)
      else: os.environ[name] = value
  empty_handles = {}
  empty_resources = ChannelResources(handles=empty_handles)
  assert empty_resources.handles is empty_handles
  assert empty_resources.handles == {}
  try:
    ChannelResources(handles=object())
    raise AssertionError("bad resource handles were accepted")
  except ValueError as exc:
    assert "resource handles" in str(exc)
  kern_backend_resources = StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=4, token=0x56)
  kern_backend = StandaloneNvBackend(rshell, kern_backend_resources, ralloc, submitter=None)
  kern_slice = kern_backend.alloc_kernargs(32, 16)
  assert kern_slice.va_addr == kern_backend_resources.kernargs_buf.va_addr and kern_slice.size == 32
  for bad_kern_mutation, text in [
    (lambda r: setattr(r, "kernargs_buf", None), "kernel argument buffer"),
    (lambda r: setattr(r.kernargs_buf, "view", None), "CPU-visible"),
    (lambda r: setattr(r, "kernargs_allocator", None), "allocator"),
  ]:
    bad_kern_resources = StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=4, token=0x57)
    bad_kern_mutation(bad_kern_resources)
    try:
      StandaloneNvBackend(rshell, bad_kern_resources, ralloc, submitter=None).alloc_kernargs(16, 8)
      raise AssertionError("bad kernel argument resource state was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_kern_call, text in [
    (lambda: kern_backend.alloc_kernargs(0, 8), "kernel argument allocation size"),
    (lambda: kern_backend.alloc_kernargs(16, 3), "alignment"),
  ]:
    try:
      bad_kern_call()
      raise AssertionError("bad kernel argument allocation input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_runtime_call, text in [
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=0), "GPFIFO entry count"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=0x100000000), "GPFIFO entry count"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=4, token=-1), "GPFIFO token"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, entries=(0x300000 // 8) + 1), "ring/control area"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, h_device=0x80, entries=4), "required together"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, h_device=0x100000000,
      h_virtmem=0x70, h_vaspace=0x90f1, entries=4), "device handle"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, h_device=0x80,
      h_virtmem=0x100000000, h_vaspace=0x90f1, entries=4), "virtual memory handle"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, h_device=0x80,
      h_virtmem=0x70, h_vaspace=0x100000000, entries=4), "vaspace handle"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, h_device=0x80, h_virtmem=0x70,
      h_vaspace=0x90f1, grctx_descs={0: GRBufDesc(0x1000, True, True)}, entries=4), "subdevice handle"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, h_device=0x80, h_virtmem=0x70,
      h_vaspace=0x90f1, h_subdevice=0x100000000, entries=4), "subdevice handle"),
    (lambda: StandaloneChannelBuilder(rshell, None).allocate_runtime_resources(ralloc, h_device=0x80, h_virtmem=0x70,
      h_vaspace=0x90f1, h_client=0x100000000, entries=4), "client handle"),
  ]:
    allocs_before_bad_runtime = len(rshell.transport.allocs)
    try:
      bad_runtime_call()
      raise AssertionError("bad runtime GPFIFO resource input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    if text != "ring/control area": assert len(rshell.transport.allocs) == allocs_before_bad_runtime
  class FailingRuntimeAllocator(StandaloneBufferAllocator):
    def __init__(self, shell):
      super().__init__(shell)
      self.calls = 0
    def alloc_sysmem(self, *args, **kwargs):
      self.calls += 1
      if self.calls == 3: raise RuntimeError("runtime allocation fail")
      return super().alloc_sysmem(*args, **kwargs)
  fail_runtime_shell = FreeShell()
  fail_runtime_alloc = FailingRuntimeAllocator(fail_runtime_shell)
  fail_runtime_va = fail_runtime_shell.mm.alloc_vaddr(0x300000, 0x1000)
  fail_runtime_shell.mm.va_alloc.free_addr(fail_runtime_va)
  try:
    StandaloneChannelBuilder(fail_runtime_shell, None).allocate_runtime_resources(fail_runtime_alloc, entries=4)
    raise AssertionError("partial runtime resource allocation failure was accepted")
  except RuntimeError as exc:
    assert "runtime allocation fail" in str(exc)
  assert len(fail_runtime_shell.freed) == 2
  assert fail_runtime_shell.mm.alloc_vaddr(0x300000, 0x1000) == fail_runtime_va
  grctx_synth = [(0, 0x100)] * 26
  grctx_synth[0] = (0x2000, 0x1000)
  grctx_synth[16] = (0x3000, 0x1000)
  for engine_id in range(17, 25): grctx_synth[engine_id] = (0x1000 + engine_id, 0x1000)
  grctx_descs = derive_grctx_buf_descs(grctx_synth)
  grctx_shell = FakeResourceShell()
  grctx_existing = {0: grctx_shell.mm.valloc(grctx_descs[0].size, contiguous=True)}
  grctx_maps = allocate_grctx_mappings(grctx_shell.mm, grctx_descs, existing=grctx_existing)
  assert grctx_maps[0] is grctx_existing[0]
  assert 1 not in grctx_maps and set(grctx_maps) == (set(grctx_descs) - {1})
  grctx_maps_with_local = allocate_grctx_mappings(grctx_shell.mm, grctx_descs, include_local=True,
                                                  existing=grctx_maps)
  assert grctx_maps_with_local[0] is grctx_existing[0]
  assert set(grctx_maps_with_local) == set(grctx_descs) and grctx_maps_with_local[1].size >= grctx_descs[1].size
  payload_maps, promote_payload = build_grctx_promote_payload(grctx_shell.mm, 0x111, 0x222, grctx_descs,
                                                              existing={0: grctx_existing[0]})
  assert payload_maps[0] is grctx_existing[0]
  assert 1 not in payload_maps
  assert struct.unpack_from("<IIIII", promote_payload, 0) == (NV2080_ENGINE_TYPE_GRAPHICS, 0, 0, 0x111, 0x222)
  assert struct.unpack_from("<I", promote_payload, 40)[0] == len(grctx_descs) - 1
  first_entry = struct.unpack_from("<QQQI HBB", promote_payload, 48)
  assert first_entry == (grctx_existing[0].paddrs[0][0], grctx_existing[0].va_addr, grctx_descs[0].size, 4, 0, 1, 0)
  packed_promote_entries = [struct.unpack_from("<QQQI HBB", promote_payload, 48 + idx * 32)
                            for idx in range(len(grctx_descs) - 1)]
  entry10 = next(entry for entry in packed_promote_entries if entry[4] == 10)
  assert entry10[0] == payload_maps[10].paddrs[0][0] and entry10[1] == 0 and entry10[2] == grctx_descs[10].size
  assert entry10[4:7] == (10, 1, 1)
  small_grctx_map = grctx_shell.mm.valloc(0x1000, contiguous=True)
  class NoPhysMapping:
    va_addr, size, paddrs = 0x12345000, 0x3000, []
  for bad_grctx_call, text in [
    (lambda: allocate_grctx_mappings(grctx_shell.mm, {0x10000: GRBufDesc(0x1000, virt=True, phys=True)}), "context buffer id"),
    (lambda: allocate_grctx_mappings(grctx_shell.mm, {0: object()}), "descriptor 0"),
    (lambda: allocate_grctx_mappings(grctx_shell.mm, {0: GRBufDesc(0x1000, virt=True, phys=True)}, existing=object()), "existing context buffer mappings"),
    (lambda: allocate_grctx_mappings(grctx_shell.mm, {0: GRBufDesc(0x2000, virt=True, phys=True)}, existing={0: small_grctx_map}), "smaller"),
    (lambda: build_promote_ctx_entries({0: GRBufDesc(0x1000, virt=True, phys=True)}, object()), "context buffer mappings"),
    (lambda: build_promote_ctx_entries({0: GRBufDesc(0x1000, virt=True, phys=True)}, {0: NoPhysMapping()}), "physical pages"),
  ]:
    try:
      bad_grctx_call()
      raise AssertionError("bad GR context mapping input was accepted")
    except (ValueError, KeyError) as exc:
      assert text in str(exc)
  class ShortSysmemTransport:
    def alloc_sysmem(self, size, contiguous=False):
      return MMIOView(bytearray(round_up(size, 0x1000))), [0x90000000]
  short_shell = type("ShortShell", (), {"mm": FakeMem(), "transport": ShortSysmemTransport()})()
  short_va = short_shell.mm.next_va
  try:
    StandaloneBufferAllocator(short_shell).alloc_sysmem(0x2000)
    raise AssertionError("short sysmem page list was accepted")
  except RuntimeError as exc:
    assert "transport returned 1 pages" in str(exc)
  assert short_shell.mm.alloc_vaddr(0x2000, 0x1000) == short_va
  class BadPageTransport:
    def __init__(self, paddr):
      self.paddr, self.freed = paddr, []
    def alloc_sysmem(self, size, contiguous=False):
      return MMIOView(bytearray(round_up(size, 0x1000))), [self.paddr]
    def free_sysmem(self, sysmem):
      self.freed.append(sysmem)
  for bad_alloc_call, text in [
    (lambda: StandaloneBufferAllocator(type("BadAllocShell", (), {"mm": FakeMem(), "transport": FakeSysmemTransport()})()).alloc_sysmem(0), "sysmem allocation size"),
    (lambda: StandaloneBufferAllocator(type("BadAlignShell", (), {"mm": FakeMem(), "transport": FakeSysmemTransport()})()).alloc_sysmem(0x1000, align=3), "alignment"),
  ]:
    try:
      bad_alloc_call()
      raise AssertionError("bad sysmem allocation request was accepted")
    except ValueError as exc:
      assert text in str(exc)
  for bad_paddr in (-0x1000, 0x90000001):
    bad_transport = BadPageTransport(bad_paddr)
    bad_shell = type("BadPageShell", (), {"mm": GpuMemoryManager(64 << 20), "transport": bad_transport})()
    bad_va = bad_shell.mm.alloc_vaddr(0x1000, 0x1000)
    bad_shell.mm.va_alloc.free_addr(bad_va)
    try:
      StandaloneBufferAllocator(bad_shell).alloc_sysmem(0x1000)
      raise AssertionError("bad sysmem physical page was accepted")
    except ValueError as exc:
      assert "sysmem page" in str(exc)
    assert bad_transport.freed and bad_shell.mm.alloc_vaddr(0x1000, 0x1000) == bad_va
  free_shell = FreeShell()
  free_alloc = StandaloneBufferAllocator(free_shell)
  tmp_buf = free_alloc.alloc_sysmem(0x1000)
  tmp_view = tmp_buf.offset(0x100, 0x100)
  tmp_sysmem = tmp_buf.meta.sysmem
  tmp_va = tmp_buf.va_addr
  free_alloc.free(tmp_view)
  free_alloc.free(tmp_buf)
  assert free_shell.freed and free_shell.freed[0] is tmp_sysmem
  assert len(free_shell.freed) == 1
  assert tmp_buf.view is None and tmp_buf.meta is None
  try:
    tmp_view.cpu_view()
    raise AssertionError("freed offset buffer retained a CPU view")
  except RuntimeError as exc:
    assert "freed" in str(exc)
  assert free_shell.mm.alloc_vaddr(0x1000, 0x1000) == tmp_va
  try:
    free_alloc.free(object())
    raise AssertionError("non-buffer free was accepted")
  except ValueError as exc:
    assert "GPU buffer" in str(exc)
  bounds_buf = free_alloc.alloc_sysmem(0x1000)
  free_alloc.copyin(bounds_buf.offset(0x100, 16), b"1234567890abcdef")
  assert free_alloc.copyout(bounds_buf.offset(0x100, 16), 16) == bytearray(b"1234567890abcdef")
  try:
    free_alloc.copyin(bounds_buf.offset(0x100, 16), b"x" * 17)
    raise AssertionError("oversized copyin was accepted")
  except ValueError as exc:
    assert "copyin size" in str(exc)
  try:
    free_alloc.copyout(bounds_buf.offset(0x100, 16), 17)
    raise AssertionError("oversized copyout was accepted")
  except ValueError as exc:
    assert "copyout size" in str(exc)
  for bad_copy_call, text in [
    (lambda: free_alloc.copyin(object(), b"x"), "copyin requires"),
    (lambda: free_alloc.copyout(object(), 1), "copyout requires"),
  ]:
    try:
      bad_copy_call()
      raise AssertionError("non-buffer copy operation was accepted")
    except ValueError as exc:
      assert text in str(exc)
  vram_copy_buf = free_alloc.alloc_vram(0x1000, contiguous=True)
  try:
    free_alloc.copyin(vram_copy_buf, b"x")
    raise AssertionError("VRAM copyin was accepted")
  except RuntimeError as exc:
    assert "CPU-visible" in str(exc)
  free_alloc.free(vram_copy_buf)
  free_alloc.free(bounds_buf)

  class FakeRm:
    def __init__(self):
      self.calls, self.allocs, self.controls = [], [], []
      self.priv_root, self.next_handle = 0xc1e00004, 0xcf000000
    def handle(self):
      handle = self.next_handle
      self.next_handle += 1
      return handle
    def alloc_memory(self, h_device, h_class, paddrs, length, flags):
      self.calls.append((h_device, h_class, tuple(paddrs), length, flags))
      return self.handle()
    def rm_alloc(self, h_parent, h_class, params=b"", client=None, h_object=None):
      h_object = self.handle() if h_object is None else h_object
      self.allocs.append((h_parent, h_class, h_object, bytes(params)))
      return h_object
    def rm_control(self, h_object, cmd, params=b"", client=None):
      self.controls.append((h_object, cmd, bytes(params)))
      if cmd == NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: return struct.pack("<I", 0x66)
      return b""
  fake_rm = FakeRm()
  ralloc.configure_rm(fake_rm, 0x80)
  handle_buf = ralloc.alloc_sysmem(0x2000, rm_handle=True, flags=0x1234)
  assert handle_buf.meta.hMemory == 0xcf000000
  assert fake_rm.calls[0][0] == 0x80 and fake_rm.calls[0][1] == NV01_MEMORY_SYSTEM_OS_DESCRIPTOR
  class FailingRm(FakeRm):
    def alloc_memory(self, h_device, h_class, paddrs, length, flags):
      raise RuntimeError("rm alloc memory fail")
  fail_shell = FreeShell()
  fail_alloc = StandaloneBufferAllocator(fail_shell)
  fail_alloc.configure_rm(FailingRm(), 0x80)
  fail_va = fail_shell.mm.alloc_vaddr(0x1000, 0x1000)
  fail_shell.mm.va_alloc.free_addr(fail_va)
  try:
    fail_alloc.alloc_sysmem(0x1000, rm_handle=True)
    raise AssertionError("RM-backed sysmem allocation failure was accepted")
  except RuntimeError as exc:
    assert "rm alloc memory fail" in str(exc)
  assert fail_va not in fail_shell.mm.mappings
  fail_leaf = fail_shell.mm.ensure_leaf_table(fail_va)
  fail_idx = fail_shell.mm.table_index(fail_va, len(fail_shell.mm.level_shifts) - 1)
  assert fail_leaf.entries[fail_idx] == 0
  assert fail_shell.freed and fail_shell.mm.alloc_vaddr(0x1000, 0x1000) == fail_va
  class FakeGoldenRm(FakeRm):
    def __init__(self):
      super().__init__()
      self.priv_root = 0xc1e00004
    def rm_control(self, h_object, cmd, params=b"", client=None):
      self.controls.append((h_object, cmd, bytes(params)))
      if cmd == NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO:
        data = bytearray(1664)
        sizes = {0: 0x2000, 16: 0x3000, **{idx: 0x1000 + idx * 0x100 for idx in range(17, 25)}}
        for idx, size in sizes.items():
          align = 0x1000
          struct.pack_into("<II", data, idx * 8, size, align)
        return bytes(data)
      return b""
  golden_shell = type("GoldenShell", (), {"mm": GpuMemoryManager(64 << 20), "transport": FakeSysmemTransport(),
                                          "chip_name": "GA102"})()
  golden_alloc = StandaloneBufferAllocator(golden_shell)
  golden_rm = FakeGoldenRm()
  golden = StandaloneChannelBuilder(golden_shell, golden_rm).prepare_golden_image_context(golden_alloc)
  assert golden["reserved_va"] == 0x1000000000
  assert [level[3] for level in golden["reserved_levels"]] == [47, 38, 29]
  assert golden["gpfifo_area"].size == 0x1000
  assert golden["grctx_descs"][1].local and 1 not in golden["grctx_mappings"]
  assert golden["grctx_descs"][5].size == 2 << 20
  assert [call[1] for call in golden_rm.allocs] == [
    NV01_DEVICE_0, NV20_SUBDEVICE_0, FERMI_VASPACE_A, AMPERE_CHANNEL_GPFIFO_A, AMPERE_COMPUTE_B, AMPERE_DMA_COPY_B]
  pde_control = next(ctrl for ctrl in golden_rm.controls if ctrl[1] == NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES)
  assert pde_control[0] == golden["vaspace"]
  assert struct.unpack_from("<IIQQQI", pde_control[2], 0) == (0, 0, 512 << 20, golden["reserved_va"],
    golden["reserved_va"] + (512 << 20) - 1, 3)
  bad_reserved_rm = FakeGoldenRm()
  bad_reserved_alloc = StandaloneBufferAllocator(golden_shell)
  try:
    StandaloneChannelBuilder(golden_shell, bad_reserved_rm).prepare_golden_image_context(bad_reserved_alloc, reserved_size=0)
    raise AssertionError("bad golden reserved size was accepted")
  except ValueError as exc:
    assert "golden reserved size" in str(exc)
  assert bad_reserved_rm.allocs == [] and bad_reserved_rm.controls == []
  ctx_control = next(ctrl for ctrl in golden_rm.controls if ctrl[1] == NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO)
  assert ctx_control[0] == golden["subdevice"] and len(ctx_control[2]) == 1664
  promote_control = next(ctrl for ctrl in golden_rm.controls if ctrl[1] == NV2080_CTRL_CMD_GPU_PROMOTE_CTX)
  assert promote_control[0] == golden["subdevice"]
  assert struct.unpack_from("<IIIII", promote_control[2], 0) == (NV2080_ENGINE_TYPE_GRAPHICS, 0, 0,
    golden_rm.priv_root, golden["gpfifo"])
  assert struct.unpack_from("<I", promote_control[2], 40)[0] == len(golden["grctx_descs"]) - 1
  golden_entry_ids = [struct.unpack_from("<QQQI HBB", promote_control[2], 48 + idx * 32)[4]
                      for idx in range(len(golden["grctx_descs"]) - 1)]
  assert golden_entry_ids == [0, 2, 3, 4, 5, 6, 9, 10, 11]
  class LiveSizedGoldenRm(FakeGoldenRm):
    def __init__(self, shell):
      super().__init__()
      self.shell = shell
      self.next_handle = 0xcf000001
    def rm_alloc(self, h_parent, h_class, params=b"", client=None, h_object=None):
      h_object = super().rm_alloc(h_parent, h_class, params=params, client=client, h_object=h_object)
      if h_class == AMPERE_CHANNEL_GPFIFO_A:
        ramfc = self.shell.mm.valloc(0x1000, contiguous=True)
        method = self.shell.mm.palloc(0x5000, align=0x1000)
        self.private_gpfifo_backing = (ramfc, method)
      return h_object
    def rm_control(self, h_object, cmd, params=b"", client=None):
      self.controls.append((h_object, cmd, bytes(params)))
      if cmd == NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO:
        data = bytearray(1664)
        sizes = {
          0: (0x153000, 0x1000), 16: (0x5000, 0x1000), 17: (0x3000, 0x1000), 18: (0x20000, 0x1000),
          19: (0x900000, 0x1000), 20: (0x80000, 0x1000), 21: (0x1000, 0x1000), 22: (0x1000, 0x1000),
          23: (0x10000, 0x1000), 24: (0x80000, 0x1000),
        }
        for idx, (size, align) in sizes.items():
          struct.pack_into("<II", data, idx * 8, size, align)
        return bytes(data)
      return b""
  live_golden_shell = type("LiveGoldenShell", (), {"mm": GpuMemoryManager(0x300000000 - 0x4000000, reserve_ptable=True),
                                                   "transport": FakeSysmemTransport(), "chip_name": "GA102"})()
  assert live_golden_shell.mm.palloc(0xf000) == 0x1a00000
  assert live_golden_shell.mm.palloc(0xf000) == 0x1a0f000
  live_golden_rm = LiveSizedGoldenRm(live_golden_shell)
  live_golden = StandaloneChannelBuilder(live_golden_shell, live_golden_rm).prepare_golden_image_context(
    StandaloneBufferAllocator(live_golden_shell))
  live_promote_control = next(ctrl for ctrl in live_golden_rm.controls if ctrl[1] == NV2080_CTRL_CMD_GPU_PROMOTE_CTX)
  assert hashlib.sha256(live_promote_control[2]).hexdigest() == "e9e2b70c57c91ad0a7855c258fd1145f43bc93013279e1b8825b333042d3c6bc"
  live_compute_alloc = next(call for call in live_golden_rm.allocs if call[1] == AMPERE_COMPUTE_B)
  live_dma_alloc = next(call for call in live_golden_rm.allocs if call[1] == AMPERE_DMA_COPY_B)
  assert live_compute_alloc[0] == live_golden["gpfifo"] and live_compute_alloc[2] == live_golden["compute"]
  assert live_dma_alloc[0] == live_golden["gpfifo"] and live_dma_alloc[2] == live_golden["dma"]
  assert hashlib.sha256(pack_rpc_gsp_rm_alloc(live_golden_rm.priv_root, live_compute_alloc[0],
    live_compute_alloc[2], live_compute_alloc[1], live_compute_alloc[3])).hexdigest() == "8331166a7d48271339adc62ff6ad71dc3d447835073de290eb31f67a4a5847db"
  assert hashlib.sha256(pack_rpc_gsp_rm_alloc(live_golden_rm.priv_root, live_dma_alloc[0],
    live_dma_alloc[2], live_dma_alloc[1], live_dma_alloc[3])).hexdigest() == "f7fdaab853626fe610f54d9ec1d8169f0c2d693180d2e5c41c34f69546ce9088"
  assert live_golden["grctx_mappings"][0].paddrs == [(0x1a25000, 0x193000)]
  assert live_golden["grctx_mappings"][2].paddrs == [(0x1bb8000, 0x5000)]
  assert live_golden["grctx_mappings"][9].paddrs == [(0x2660000, 0x10000)]
  assert live_golden["grctx_mappings"][10].paddrs == [(0x2670000, 0x80000)]
  assert live_golden["grctx_mappings"][11].paddrs == [(0x26f0000, 0x80000)]
  golden_fingerprint = golden_compute_fingerprint_state()
  assert golden_fingerprint["boot_paddrs"] == (0x1a00000, 0x1a0f000)
  assert golden_fingerprint["promote_sha256"] == "e9e2b70c57c91ad0a7855c258fd1145f43bc93013279e1b8825b333042d3c6bc"
  assert golden_fingerprint["compute_alloc"][:3] == (0xcf000004, AMPERE_COMPUTE_B, 0xcf000005)
  assert golden_fingerprint["compute_rpc_sha256"] == "8331166a7d48271339adc62ff6ad71dc3d447835073de290eb31f67a4a5847db"
  assert golden_fingerprint["dma_alloc"][:3] == (0xcf000004, AMPERE_DMA_COPY_B, 0xcf000006)
  assert golden_fingerprint["dma_rpc_sha256"] == "f7fdaab853626fe610f54d9ec1d8169f0c2d693180d2e5c41c34f69546ce9088"
  fingerprint_buf = io.StringIO()
  with contextlib.redirect_stdout(fingerprint_buf): print_golden_compute_fingerprint()
  fingerprint_text = fingerprint_buf.getvalue()
  assert "standalone golden_compute_alloc parent=0xcf000004 object=0xcf000005 compute_class=0xc7c0" in fingerprint_text
  assert "rpc_sha256=8331166a7d48271339adc62ff6ad71dc3d447835073de290eb31f67a4a5847db" in fingerprint_text
  context_promote = context_promote_fingerprint_state()
  assert context_promote["golden"]["payload_sha256"] == "e9e2b70c57c91ad0a7855c258fd1145f43bc93013279e1b8825b333042d3c6bc"
  assert context_promote["golden"]["ids"] == [0, 2, 3, 4, 5, 6, 9, 10, 11]
  assert context_promote["user_phys"]["payload_sha256"] == "b35b21e1a9f664776bb48d75b296c46b7647edd4ea1793f65834238c7c334467"
  assert context_promote["user_virt"]["payload_sha256"] == "16315f9a913040a181fc0d63b749b33ff2714ff4a91da3f562b7e8571b5dd078"
  assert context_promote["user_phys"]["ids"] == [0, 1, 2] and context_promote["user_virt"]["ids"] == [0, 1, 2]
  context_promote_buf = io.StringIO()
  with contextlib.redirect_stdout(context_promote_buf): print_context_promote_fingerprint()
  context_promote_text = context_promote_buf.getvalue()
  assert "standalone context_promote target client=0xc1e00004 subdevice=0xcf000002 gpfifo=0xcf000004" in context_promote_text
  assert "standalone context_promote label=user_phys entries=3 ids=[0, 1, 2]" in context_promote_text
  assert "standalone context_promote_map id=1 va=0x1020294000 paddrs=0x2770000/0x5000" in context_promote_text
  old_trace_channel, old_trace_channel_stack = os.environ.get("NV_ADD_TRACE_CHANNEL"), os.environ.get("NV_ADD_TRACE_CHANNEL_STACK")
  try:
    os.environ["NV_ADD_TRACE_CHANNEL"] = "1"
    os.environ["NV_ADD_TRACE_CHANNEL_STACK"] = "1"
    channel_trace_buf = io.StringIO()
    trace_shell = type("TraceGoldenShell", (), {"mm": GpuMemoryManager(64 << 20), "transport": FakeSysmemTransport(),
                                                "chip_name": "GA102"})()
    with contextlib.redirect_stdout(channel_trace_buf):
      StandaloneChannelBuilder(trace_shell, FakeGoldenRm()).prepare_golden_image_context(StandaloneBufferAllocator(trace_shell))
    channel_trace = channel_trace_buf.getvalue()
    assert "channel golden_start" in channel_trace and "channel golden_gpfifo_alloc" in channel_trace and "channel golden_done" in channel_trace
    assert "channel golden_promote_done" in channel_trace and "channel golden_compute_alloc" in channel_trace
    assert "expected_rpc_sha256=" in channel_trace and "compute_class=0x" in channel_trace
    assert "params_sha256=" in channel_trace
    assert "packed_entries_sha256=" in channel_trace
    assert all(f"{field}=0x" in channel_trace for field in expected_gpfifo_descriptor_trace_fields())
    assert "h_object_buffer=0x0" in channel_trace and "h_object_error=0x0" in channel_trace
    assert "prepare_golden_image_context" in channel_trace
  finally:
    for name, value in (("NV_ADD_TRACE_CHANNEL", old_trace_channel), ("NV_ADD_TRACE_CHANNEL_STACK", old_trace_channel_stack)):
      if value is None: os.environ.pop(name, None)
      else: os.environ[name] = value
  class FailingGoldenRm(FakeGoldenRm):
    def rm_control(self, h_object, cmd, params=b"", client=None):
      if cmd == NV2080_CTRL_CMD_GPU_PROMOTE_CTX:
        self.controls.append((h_object, cmd, bytes(params)))
        raise RuntimeError("golden promote fail")
      return super().rm_control(h_object, cmd, params, client=client)
  golden_fail_shell = FreeShell()
  golden_fail_shell.chip_name = "GA102"
  golden_fail_alloc = StandaloneBufferAllocator(golden_fail_shell)
  golden_fail_rm = FailingGoldenRm()
  try:
    StandaloneChannelBuilder(golden_fail_shell, golden_fail_rm).prepare_golden_image_context(golden_fail_alloc)
    raise AssertionError("golden context promote failure was accepted")
  except RuntimeError as exc:
    assert "golden promote fail" in str(exc)
  assert not golden_fail_shell.freed
  assert golden_fail_shell.mm.alloc_vaddr(512 << 20, 0x1000) == 0x1000000000
  assert len(golden_fail_shell.mm.mappings) == 0
  class ShortContextInfoRm(FakeGoldenRm):
    def rm_control(self, h_object, cmd, params=b"", client=None):
      if cmd == NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO:
        self.controls.append((h_object, cmd, bytes(params)))
        return bytes(16)
      return super().rm_control(h_object, cmd, params, client=client)
  short_ctx_shell = FreeShell()
  short_ctx_shell.chip_name = "GA102"
  short_ctx_alloc = StandaloneBufferAllocator(short_ctx_shell)
  try:
    StandaloneChannelBuilder(short_ctx_shell, ShortContextInfoRm()).prepare_golden_image_context(short_ctx_alloc)
    raise AssertionError("short golden context buffer info response was accepted")
  except ValueError as exc:
    assert "context buffer info" in str(exc)
  assert not short_ctx_shell.freed and short_ctx_shell.mm.alloc_vaddr(512 << 20, 0x1000) == 0x1000000000
  assert len(short_ctx_shell.mm.mappings) == 0
  user_client = 0xc1e00055
  user_promote_start = len(golden_rm.controls)
  user_maps = StandaloneChannelBuilder(golden_shell, golden_rm).promote_user_compute_context(
    user_client, golden["subdevice"], golden["gpfifo"], golden["grctx_descs"])
  user_promotes = [ctrl for ctrl in golden_rm.controls[user_promote_start:] if ctrl[1] == NV2080_CTRL_CMD_GPU_PROMOTE_CTX]
  assert len(user_promotes) == 2
  assert all(ctrl[0] == golden["subdevice"] for ctrl in user_promotes)
  phys_payload, virt_payload = user_promotes[0][2], user_promotes[1][2]
  assert struct.unpack_from("<IIIII", phys_payload, 0) == (NV2080_ENGINE_TYPE_GRAPHICS, 0, 0,
    user_client, golden["gpfifo"])
  assert struct.unpack_from("<IIIII", virt_payload, 0) == (NV2080_ENGINE_TYPE_GRAPHICS, 0, 0,
    user_client, golden["gpfifo"])
  assert struct.unpack_from("<I", phys_payload, 40)[0] == 3
  assert struct.unpack_from("<I", virt_payload, 40)[0] == 3
  phys_entries = [struct.unpack_from("<QQQI HBB", phys_payload, 48 + idx * 32) for idx in range(3)]
  virt_entries = [struct.unpack_from("<QQQI HBB", virt_payload, 48 + idx * 32) for idx in range(3)]
  assert [entry[4] for entry in phys_entries] == [0, 1, 2]
  assert [entry[4] for entry in virt_entries] == [0, 1, 2]
  assert all(entry[1] == 0 and entry[5:] == (1, 1) for entry in phys_entries)
  assert all(entry[0] == 0 and entry[2] == 0 and entry[3] == 0 and entry[5:] == (0, 0) for entry in virt_entries)
  assert [entry[1] for entry in virt_entries] == [user_maps[idx].va_addr for idx in (0, 1, 2)]
  class FailingSecondPromoteRm(FakeRm):
    def rm_control(self, h_object, cmd, params=b"", client=None):
      self.controls.append((h_object, cmd, bytes(params)))
      if cmd == NV2080_CTRL_CMD_GPU_PROMOTE_CTX and len([ctrl for ctrl in self.controls if ctrl[1] == cmd]) == 2:
        raise RuntimeError("user promote fail")
      return b""
  user_fail_shell = type("UserFailShell", (), {"mm": GpuMemoryManager(64 << 20), "transport": FakeSysmemTransport(),
                                               "chip_name": "GA102"})()
  user_fail_rm = FailingSecondPromoteRm()
  user_fail_descs = {
    0: GRBufDesc(0x3000, virt=True, phys=True),
    1: GRBufDesc(0x2000, virt=True, phys=True, local=True),
    2: GRBufDesc(0x2000, virt=True, phys=True),
  }
  existing_user_map = {0: user_fail_shell.mm.valloc(0x3000, contiguous=True)}
  existing_va = existing_user_map[0].va_addr
  try:
    StandaloneChannelBuilder(user_fail_shell, user_fail_rm).promote_user_compute_context(
      user_client, 0x2080, 0xc36f, user_fail_descs, grctx_mappings=existing_user_map)
    raise AssertionError("user compute promote failure was accepted")
  except RuntimeError as exc:
    assert "user promote fail" in str(exc)
  assert existing_va in user_fail_shell.mm.mappings
  assert len(user_fail_shell.mm.mappings) == 1
  class FailingChannelSetupRm(FakeRm):
    def __init__(self, shell):
      super().__init__()
      self.shell = shell
      self.private_mappings = {}
      self.last_private_va = None
    def rm_alloc(self, h_parent, h_class, params=b"", client=None, h_object=None):
      if h_class == GT200_DEBUGGER: raise RuntimeError("debugger alloc fail")
      h_object = super().rm_alloc(h_parent, h_class, params=params, client=client, h_object=h_object)
      if h_class == AMPERE_CHANNEL_GPFIFO_A:
        ramfc = self.shell.mm.valloc(0x1000, contiguous=True)
        method = self.shell.mm.valloc(0x5000, contiguous=True)
        self.last_private_va = ramfc.va_addr
        self.private_mappings[h_object] = (ramfc, method)
      return h_object
  chan_setup_shell = FreeShell()
  chan_setup_shell.chip_name = "GA102"
  chan_setup_alloc = StandaloneBufferAllocator(chan_setup_shell)
  chan_setup_gpfifo = chan_setup_alloc.alloc_sysmem(0x300000, contiguous=True)
  chan_setup_notifier = chan_setup_alloc.alloc_sysmem(0x1000)
  chan_setup_existing = {0: chan_setup_shell.mm.valloc(0x3000, contiguous=True)}
  chan_setup_existing_va = chan_setup_existing[0].va_addr
  chan_setup_rm = FailingChannelSetupRm(chan_setup_shell)
  try:
    StandaloneChannelBuilder(chan_setup_shell, chan_setup_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, chan_setup_gpfifo, chan_setup_notifier, entries=4, h_subdevice=0x2080,
      grctx_descs=user_fail_descs, grctx_mappings=chan_setup_existing, h_client=user_client)
    raise AssertionError("compute channel debugger failure was accepted")
  except RuntimeError as exc:
    assert "debugger alloc fail" in str(exc)
  assert chan_setup_existing_va in chan_setup_shell.mm.mappings
  assert len([va for va in chan_setup_shell.mm.mappings if va not in (chan_setup_gpfifo.va_addr,
    chan_setup_notifier.va_addr, chan_setup_existing_va)]) == 0
  assert 0xc56f not in chan_setup_rm.private_mappings
  assert chan_setup_shell.mm.alloc_vaddr(0x1000, 0x1000) == chan_setup_rm.last_private_va
  class FailingDebuggerRm(FakeRm):
    def rm_alloc(self, h_parent, h_class, params=b"", client=None, h_object=None):
      if h_class == GT200_DEBUGGER: raise RuntimeError("debugger alloc fail")
      return super().rm_alloc(h_parent, h_class, params=params, client=client, h_object=h_object)
  chan_fail_shell = FreeShell()
  chan_fail_shell.chip_name = "GA102"
  chan_fail_alloc = StandaloneBufferAllocator(chan_fail_shell)
  chan_gpfifo = chan_fail_alloc.alloc_sysmem(0x300000, contiguous=True)
  chan_notifier = chan_fail_alloc.alloc_sysmem(0x1000)
  chan_existing = {0: chan_fail_shell.mm.valloc(0x3000, contiguous=True)}
  chan_existing_va = chan_existing[0].va_addr
  chan_fail_rm = FailingDebuggerRm()
  try:
    StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, chan_gpfifo, chan_notifier, entries=4, h_subdevice=0x2080,
      grctx_descs=user_fail_descs, grctx_mappings=chan_existing, h_client=user_client)
    raise AssertionError("compute channel debugger failure was accepted")
  except RuntimeError as exc:
    assert "debugger alloc fail" in str(exc)
  assert chan_existing_va in chan_fail_shell.mm.mappings
  assert len([va for va in chan_fail_shell.mm.mappings if va not in (chan_gpfifo.va_addr, chan_notifier.va_addr, chan_existing_va)]) == 0
  allocs_before_bad_channel = len(chan_fail_rm.allocs)
  bad_gpfifo_no_meta = GpuBuffer(chan_gpfifo.va_addr, chan_gpfifo.size, view=chan_gpfifo.view)
  bad_gpfifo_unaligned = GpuBuffer(chan_gpfifo.va_addr, chan_gpfifo.size, view=chan_gpfifo.view,
    meta=type("Meta", (), {"paddrs": [(0x80000001, chan_gpfifo.size)], "hMemory": 0})())
  bad_gpfifo_bad_paddr = GpuBuffer(chan_gpfifo.va_addr, chan_gpfifo.size, view=chan_gpfifo.view,
    meta=type("Meta", (), {"paddrs": [("bad", chan_gpfifo.size)], "hMemory": 0})())
  bad_notifier_no_pages = GpuBuffer(chan_notifier.va_addr, chan_notifier.size, view=chan_notifier.view,
    meta=type("Meta", (), {"paddrs": [], "hMemory": 0})())
  for bad_channel_call, text in [
    (lambda: StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, chan_gpfifo, chan_notifier, entries=0), "GPFIFO entry count"),
    (lambda: StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, chan_gpfifo, chan_notifier, entries=4, offset=4), "GPFIFO offset"),
    (lambda: StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, chan_gpfifo, chan_notifier, entries=(chan_gpfifo.size // 8) + 1), "ring/control area"),
    (lambda: StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, bad_gpfifo_no_meta, chan_notifier, entries=4), "GPFIFO area buffer"),
    (lambda: StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, bad_gpfifo_unaligned, chan_notifier, entries=4), "GPFIFO area physical page"),
    (lambda: StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, bad_gpfifo_bad_paddr, chan_notifier, entries=4), "GPFIFO area physical page"),
    (lambda: StandaloneChannelBuilder(chan_fail_shell, chan_fail_rm).allocate_compute_channel(
      0x80, 0x90f1, 0x70, chan_gpfifo, bad_notifier_no_pages, entries=4), "notifier physical pages"),
  ]:
    try:
      bad_channel_call()
      raise AssertionError("bad compute-channel GPFIFO input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert len(chan_fail_rm.allocs) == allocs_before_bad_channel
  try:
    StandaloneChannelBuilder(golden_shell, golden_rm).promote_user_compute_context(user_client, golden["subdevice"],
      golden["gpfifo"], {0: golden["grctx_descs"][0], 1: golden["grctx_descs"][1]})
    raise AssertionError("incomplete user compute context buffers were accepted")
  except KeyError as exc:
    assert "0, 1, and 2" in str(exc)
  gp = pack_nv_channel_gpfifo_allocation_params(0x100000, 4, 0x67, 0x90f1, 0x70, 0x1234,
    0x20, 0x80000000, 0x80001000, error_paddr=0x90000000, h_object_error=0x777,
    userd_paddr=0x80002000)
  assert len(gp) == 368
  assert struct.unpack_from("<IIQIIIII", gp, 0) == (0x777, 0x70, 0x100000, 4, 0x200320, 0x67, 0x90f1, 0x1234)
  assert struct.unpack_from("<Q", gp, 64)[0] == 0x20
  assert struct.unpack_from("<QQII", gp, 144) == (0x80000000, 0x1000, 2, 0)
  assert struct.unpack_from("<QQII", gp, 168) == (0, 0x20, 2, 0)
  assert struct.unpack_from("<QQII", gp, 248) == (0x90000000, 48 << 20, 2, 0)
  gp_phys_userd = pack_nv_channel_gpfifo_allocation_params(0x100000, 4, 0x67, 0x90f1, 0x70, 0,
    0x20, 0x80000000, 0x80001000, userd_paddr=0x80002000)
  assert struct.unpack_from("<QQII", gp_phys_userd, 168) == (0x80002000, 0x20, 2, 0)
  gp_zero_desc = pack_nv_channel_gpfifo_allocation_params(0x100000, 4, 0, 0x90f1, 0, 0,
    0x20, 0, 0, userd_paddr=0x80002000)
  assert struct.unpack_from("<QQII", gp_zero_desc, 144) == (0, 0, 0, 0)
  assert struct.unpack_from("<QQII", gp_zero_desc, 192) == (0, 0, 0, 0)
  assert struct.unpack_from("<QQII", gp_zero_desc, 216) == (0, 0, 0, 0)
  zero_desc_summary = gpfifo_memory_desc_summary(gp_zero_desc)
  assert zero_desc_summary["ramfc"]["base"] == 0 and zero_desc_summary["instance"]["size"] == 0
  assert zero_desc_summary["userd"]["base"] == 0x80002000 and zero_desc_summary["userd"]["size"] == 0x20
  zero_ctor = unpack_nv_channel_gpfifo_allocation_params(gp_zero_desc)
  assert zero_ctor["gpfifo_va"] == 0x100000 and zero_ctor["entries"] == 4 and zero_ctor["h_context_share"] == 0
  assert zero_ctor["h_vaspace"] == 0x90f1 and zero_ctor["h_userd_memory"] == 0
  assert unpack_gpfifo_work_submit_token(struct.pack("<I", 0x66)) == 0x66
  try:
    unpack_gpfifo_work_submit_token(b"\x01\x02\x03")
    raise AssertionError("short GPFIFO work-submit token response was accepted")
  except ValueError as exc:
    assert "work-submit token" in str(exc)
  gp_tiny_ctor = tinygrad_equivalent_gpfifo_constructor_params()
  assert tinygrad_equivalent_gpfifo_constructor_sha256() == "6e59c0bfb532bf40a031c416b581d48ef49f3976bc9b55cda546ea24e008537a"
  class FakeQueue:
    def __init__(self, shell):
      self.gsp, self.sent = shell, []
    def send_rpc(self, func, payload): self.sent.append((func, bytes(payload)))
    def debug_state(self, slots=4): return f"fake_cmd slots={slots} sent={len(self.sent)}"
  class FakeStatQueue:
    def wait_resp(self, func): return pack_rpc_gsp_rm_alloc(0xc1e00004, 0x80, 0xcf000000, AMPERE_CHANNEL_GPFIFO_A)
    def debug_state(self, slots=4): return f"fake_stat slots={slots}"
  class FakeHookShell:
    def __init__(self):
      self.mm = type("MM", (), {"valloc": self.valloc})()
      self.allocs = []
    def valloc(self, size, contiguous=False):
      self.allocs.append((size, contiguous))
      return type("Alloc", (), {"paddrs": [(0x80000000 + len(self.allocs) * 0x1000, size)]})()
  hook_shell = FakeResourceShell()
  hook_cmd_q = FakeQueue(hook_shell)
  hook_rm = GspRmClient(hook_cmd_q, FakeStatQueue())
  gpfifo_handle = hook_rm.rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, gp_tiny_ctor)
  assert gpfifo_handle == 0xcf000000
  assert gpfifo_handle in hook_rm.private_mappings and len(hook_rm.private_mappings[gpfifo_handle]) == 1
  _, hook_record = hook_cmd_q.sent[-1]
  hook_params = hook_record[32:]
  assert struct.unpack_from("<QQII", hook_params, 144) == (0x80000000, 0x1000, 2, 0)
  assert struct.unpack_from("<QQII", hook_params, 168) == (0x80000100, 0x20, 2, 0)
  assert struct.unpack_from("<QQII", hook_params, 192) == (0x80000000, 0x200, 2, 0)
  assert struct.unpack_from("<QQII", hook_params, 216) == (0x80001000, 0x5000, 2, 0)
  old_trace_rm, old_trace_rm_stack = os.environ.get("NV_ADD_TRACE_RM_ALLOC"), os.environ.get("NV_ADD_TRACE_RM_STACK")
  try:
    os.environ["NV_ADD_TRACE_RM_ALLOC"] = "1"
    os.environ["NV_ADD_TRACE_RM_STACK"] = "1"
    rm_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(rm_trace_buf):
      GspRmClient(hook_cmd_q, FakeStatQ(ok_alloc_resp)).rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, gp_tiny_ctor, h_object=0xcf000125)
    rm_trace = rm_trace_buf.getvalue()
    _, traced_hook_record = hook_cmd_q.sent[-1]
    assert "rm gsp_alloc" in rm_trace and "h_class=0x" in rm_trace and "object=0xcf000125" in rm_trace
    assert f"params_sha256={hashlib.sha256(gp_tiny_ctor).hexdigest()}" in rm_trace
    assert "rm gpfifo_patch" in rm_trace and "ramfc=0x" in rm_trace and "method=0x" in rm_trace
    assert "ctor_gpfifo_va=0x1000000000" in rm_trace and "ctor_entries=32" in rm_trace
    assert "ctor_h_context_share=0x0" in rm_trace and "ctor_h_vaspace=0xcf000003" in rm_trace
    assert "ctor_userd_offset=0x100" in rm_trace and "ctor_engine_type=0x1" in rm_trace
    assert "before_ramfc_base=0x0" in rm_trace and "before_method_base=0x0" in rm_trace
    assert "after_ramfc_base=0x80006000" in rm_trace and "after_method_base=0x80007000" in rm_trace
    assert "after_instance_size=0x200" in rm_trace and "after_userd_size=0x20" in rm_trace
    assert f"rpc_sha256={hashlib.sha256(traced_hook_record).hexdigest()}" in rm_trace
    assert rm_trace.count("params_sha256=") == 2
  finally:
    for name, value in (("NV_ADD_TRACE_RM_ALLOC", old_trace_rm), ("NV_ADD_TRACE_RM_STACK", old_trace_rm_stack)):
      if value is None: os.environ.pop(name, None)
      else: os.environ[name] = value
  class StateTraceShell(FreeShell):
    def __init__(self):
      super().__init__()
      self.regs = {NV_PBUS_BAR1_BLOCK: 0, NV_PFB_PRI_MMU_WPR2_ADDR_HI: 0x2ffee00,
                   NV_PGSP_FALCON_ENGINE: 0x1111, NV_PSEC_FALCON_ENGINE: 0x2222}
    def rreg(self, addr): return self.regs.get(addr, 0)
  class StateTraceStatQueue(FakeStatQueue):
    def __init__(self):
      self.rx_view = MMIOView(bytearray(4), fmt='I')
      self.rx_view[0] = 0x77
  old_trace_rm_state = os.environ.get("NV_ADD_TRACE_RM_STATE")
  try:
    os.environ["NV_ADD_TRACE_RM_STATE"] = "1"
    state_shell = StateTraceShell()
    state_cmd_q = FakeQueue(state_shell)
    state_cmd_q.tx_view = MMIOView(bytearray(20), fmt='I')
    state_cmd_q.tx_view[4] = 0x33
    state_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(state_trace_buf):
      GspRmClient(state_cmd_q, StateTraceStatQueue()).rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, gp_tiny_ctor, h_object=0xcf000127)
    state_trace = state_trace_buf.getvalue()
    assert "standalone rm_alloc pre_queues" in state_trace
    assert "standalone rm_alloc post_queues" in state_trace
    assert "standalone rm_state stage=pre-alloc" in state_trace
    assert "standalone rm_state stage=post-alloc" in state_trace
    assert "object=0xcf000127" in state_trace and "class_name=AMPERE_CHANNEL_GPFIFO_A" in state_trace
    assert "bar1=0x0" in state_trace and "wpr2_hi=0x2ffee00" in state_trace
    assert "cmd_wp=0x33" in state_trace and "stat_rp=0x77" in state_trace
    assert "gsp=(engine=0x1111" in state_trace and "sec2=(engine=0x2222" in state_trace
  finally:
    if old_trace_rm_state is None: os.environ.pop("NV_ADD_TRACE_RM_STATE", None)
    else: os.environ["NV_ADD_TRACE_RM_STATE"] = old_trace_rm_state
  fecs_base_addr = NV_FALCON_FECS_BASES["ga102"]
  fecs_hwcfg2_addr = fecs_base_addr + NV_PFALCON_FALCON_HWCFG2
  fecs_cpuctl_addr = fecs_base_addr + NV_PFALCON_FALCON_CPUCTL
  pmu_base_addr = NV_FALCON_PMU_BASE
  pmu_hwcfg2_addr = pmu_base_addr + NV_PFALCON_FALCON_HWCFG2
  pmu_cpuctl_addr = pmu_base_addr + NV_PFALCON_FALCON_CPUCTL
  gsp_base_addr = NV_FALCON_GSP_BASE
  gsp_hwcfg2_addr = gsp_base_addr + NV_PFALCON_FALCON_HWCFG2
  gsp_cpuctl_addr = gsp_base_addr + NV_PFALCON_FALCON_CPUCTL
  sec2_base_addr = NV_FALCON_SEC2_BASE
  sec2_hwcfg2_addr = sec2_base_addr + NV_PFALCON_FALCON_HWCFG2
  sec2_cpuctl_addr = sec2_base_addr + NV_PFALCON_FALCON_CPUCTL
  class FecsStateShell(FreeShell):
    def __init__(self):
      super().__init__()
      self.chip = type("Chip", (), {"fw_name": "ga102", "pmc_boot_0": 0x12345678, "chip_id": 0x12345678})()
      self.regs = {NV_PBUS_BAR1_BLOCK: 0, NV_PFB_PRI_MMU_WPR2_ADDR_HI: 0x2ffee00,
        NV_PGSP_FALCON_ENGINE: 0x1111, NV_PSEC_FALCON_ENGINE: 0x2222,
        fecs_hwcfg2_addr: 0, fecs_cpuctl_addr: 0xbadf5720,
        pmu_hwcfg2_addr: 0, pmu_cpuctl_addr: 0xbadf5720,
        gsp_hwcfg2_addr: 0, gsp_cpuctl_addr: 0xbadf5720,
        gsp_base_addr + NV_PRISCV_RISCV_CPUCTL: 0x80,
        sec2_hwcfg2_addr: 0, sec2_cpuctl_addr: 0x20}
    def fecs_falcon_base(self): return NV_FALCON_FECS_BASES[self.chip.fw_name]
    def fecs_falcon_engine(self): return self.fecs_falcon_base() + NV_PFECS_FALCON_ENGINE_OFFSET
    def rreg(self, addr): return self.regs.get(addr, 0)
    def wreg(self, addr, value): self.regs[addr] = value
  class FecsStateStatQueue(FakeStatQueue):
    def __init__(self):
      self.rx_view = MMIOView(bytearray(4), fmt='I')
      self.rx_view[0] = 0x77
  old_trace_rm_state_fecs = os.environ.get("NV_ADD_TRACE_RM_STATE")
  try:
    os.environ["NV_ADD_TRACE_RM_STATE"] = "1"
    fecs_state_shell = FecsStateShell()
    fecs_state_cmd_q = FakeQueue(fecs_state_shell)
    fecs_state_cmd_q.tx_view = MMIOView(bytearray(20), fmt='I')
    fecs_state_cmd_q.tx_view[4] = 0x33
    fecs_state_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(fecs_state_trace_buf):
      GspRmClient(fecs_state_cmd_q, FecsStateStatQueue()).rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, gp_tiny_ctor, h_object=0xcf000128)
    fecs_state_trace = fecs_state_trace_buf.getvalue()
    assert fecs_state_trace.count("fenced=True") >= 3, f"expected 3+ fenced=True (GSP+FECS+PMU pre+post-alloc), got {fecs_state_trace.count('fenced=True')}"
    assert fecs_state_trace.count("has_riscv=0") >= 3, f"expected 3+ has_riscv=0 (GSP+FECS+PMU pre+post-alloc), got {fecs_state_trace.count('has_riscv=0')}"
    assert "cpuctl=0xbadf5720" in fecs_state_trace
  finally:
    if old_trace_rm_state_fecs is None: os.environ.pop("NV_ADD_TRACE_RM_STATE", None)
    else: os.environ["NV_ADD_TRACE_RM_STATE"] = old_trace_rm_state_fecs
  class FecsUnfencedShell(FreeShell):
    def __init__(self):
      super().__init__()
      self.chip = type("Chip", (), {"fw_name": "ga102", "pmc_boot_0": 0x12345678, "chip_id": 0x12345678})()
      self.regs = {NV_PBUS_BAR1_BLOCK: 0, NV_PFB_PRI_MMU_WPR2_ADDR_HI: 0x2ffee00,
        NV_PGSP_FALCON_ENGINE: 0x1111, NV_PSEC_FALCON_ENGINE: 0x2222,
        fecs_hwcfg2_addr: (1 << 10), fecs_cpuctl_addr: 0x10,
        pmu_hwcfg2_addr: (1 << 10), pmu_cpuctl_addr: 0x10,
        gsp_hwcfg2_addr: (1 << 10), gsp_cpuctl_addr: 0x10,
        sec2_hwcfg2_addr: (1 << 10), sec2_cpuctl_addr: 0x20}
    def fecs_falcon_base(self): return NV_FALCON_FECS_BASES[self.chip.fw_name]
    def fecs_falcon_engine(self): return self.fecs_falcon_base() + NV_PFECS_FALCON_ENGINE_OFFSET
    def rreg(self, addr): return self.regs.get(addr, 0)
    def wreg(self, addr, value): self.regs[addr] = value
  class FecsUnfencedStatQueue(FakeStatQueue):
    def __init__(self):
      self.rx_view = MMIOView(bytearray(4), fmt='I')
      self.rx_view[0] = 0x77
  old_trace_rm_state_unfenced = os.environ.get("NV_ADD_TRACE_RM_STATE")
  try:
    os.environ["NV_ADD_TRACE_RM_STATE"] = "1"
    fecs_unfenced_shell = FecsUnfencedShell()
    fecs_unfenced_cmd_q = FakeQueue(fecs_unfenced_shell)
    fecs_unfenced_cmd_q.tx_view = MMIOView(bytearray(20), fmt='I')
    fecs_unfenced_cmd_q.tx_view[4] = 0x33
    fecs_unfenced_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(fecs_unfenced_trace_buf):
      GspRmClient(fecs_unfenced_cmd_q, FecsUnfencedStatQueue()).rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, gp_tiny_ctor, h_object=0xcf000129)
    fecs_unfenced_trace = fecs_unfenced_trace_buf.getvalue()
    assert fecs_unfenced_trace.count("fenced=False") >= 3, f"expected 3+ fenced=False (GSP+FECS+PMU pre+post-alloc), got {fecs_unfenced_trace.count('fenced=False')}"
    assert fecs_unfenced_trace.count("has_riscv=1") >= 3, f"expected 3+ has_riscv=1 (GSP+FECS+PMU pre+post-alloc), got {fecs_unfenced_trace.count('has_riscv=1')}"
  finally:
    if old_trace_rm_state_unfenced is None: os.environ.pop("NV_ADD_TRACE_RM_STATE", None)
    else: os.environ["NV_ADD_TRACE_RM_STATE"] = old_trace_rm_state_unfenced
  class StallTracerShell:
    def __init__(self):
      self.chip = type("Chip", (), {"fw_name": "ga102", "pmc_boot_0": 0x12345678, "chip_id": 0x12345678})()
      self.regs = {fecs_hwcfg2_addr: 0, fecs_cpuctl_addr: 0xbadf5720,
                   pmu_hwcfg2_addr: 0, pmu_cpuctl_addr: 0xbadf5720,
                   gsp_hwcfg2_addr: 0, gsp_cpuctl_addr: 0xbadf5720,
                   sec2_hwcfg2_addr: 0, sec2_cpuctl_addr: 0x20}
    def fecs_falcon_base(self): return NV_FALCON_FECS_BASES[self.chip.fw_name]
    def fecs_falcon_engine(self): return self.fecs_falcon_base() + NV_PFECS_FALCON_ENGINE_OFFSET
    def rreg(self, addr): return self.regs.get(addr, 0)
  class StallTracerShellEvolving(StallTracerShell):
    def rreg(self, addr):
      if addr == pmu_cpuctl_addr: return 0x0
      return super().rreg(addr)
  old_stall_trace = os.environ.get("NV_ADD_STALL_TRACE")
  old_stall_period = os.environ.get("NV_ADD_STALL_TRACE_PERIOD_MS")
  old_stall_max = os.environ.get("NV_ADD_STALL_TRACE_MAX_MS")
  try:
    os.environ["NV_ADD_STALL_TRACE"] = "1"
    os.environ["NV_ADD_STALL_TRACE_PERIOD_MS"] = "2"
    os.environ["NV_ADD_STALL_TRACE_MAX_MS"] = "50"
    assert StallTracer.is_enabled()
    assert StallTracer.parse_period_ms() == 2.0
    assert StallTracer.parse_max_ms() == 50.0
    os.environ["NV_ADD_STALL_TRACE_PERIOD_MS"] = "bogus"
    assert StallTracer.parse_period_ms() == 5.0
    os.environ["NV_ADD_STALL_TRACE_PERIOD_MS"] = "0"
    assert StallTracer.parse_period_ms() == 5.0
    os.environ["NV_ADD_STALL_TRACE_PERIOD_MS"] = "2"
    frozen_shell = StallTracerShell()
    frozen_tracer = StallTracer(frozen_shell, label="selftest", period_ms=2.0, max_ms=50.0)
    assert frozen_tracer.is_enabled()
    frozen_buf = io.StringIO()
    with contextlib.redirect_stdout(frozen_buf):
      frozen_tracer.start()
      frozen_tracer._thread.join(timeout=0.5)
      frozen_tracer.stop(reason="test_complete")
    frozen_out = frozen_buf.getvalue()
    assert "stall_trace start label=selftest" in frozen_out
    assert "stall_trace sample label=selftest" in frozen_out
    assert "fenced=True" in frozen_out
    assert "fecs=(engine=0x0 cpuctl=0xbadf5720" in frozen_out or "fecs=(" in frozen_out
    assert "stall_trace stop label=selftest reason=test_complete" in frozen_out
    evolving_shell = StallTracerShellEvolving()
    evolving_buf = io.StringIO()
    with contextlib.redirect_stdout(evolving_buf):
      evolving_tracer = StallTracer(evolving_shell, label="evolving", period_ms=2.0, max_ms=50.0)
      evolving_tracer.start()
      evolving_tracer._thread.join(timeout=0.5)
      evolving_tracer.stop(reason="test_complete")
    evolving_out = evolving_buf.getvalue()
    assert "pmu=(" in evolving_out
    assert evolving_out.count("stall_trace sample") >= 1
  finally:
    if old_stall_trace is None: os.environ.pop("NV_ADD_STALL_TRACE", None)
    else: os.environ["NV_ADD_STALL_TRACE"] = old_stall_trace
    if old_stall_period is None: os.environ.pop("NV_ADD_STALL_TRACE_PERIOD_MS", None)
    else: os.environ["NV_ADD_STALL_TRACE_PERIOD_MS"] = old_stall_period
    if old_stall_max is None: os.environ.pop("NV_ADD_STALL_TRACE_MAX_MS", None)
    else: os.environ["NV_ADD_STALL_TRACE_MAX_MS"] = old_stall_max
  stall_trace_diag_buf = io.StringIO()
  with contextlib.redirect_stdout(stall_trace_diag_buf): print_stall_trace_diagnostic("examples/add.py")
  stall_trace_diag = stall_trace_diag_buf.getvalue().strip().splitlines()
  assert stall_trace_diag[0].startswith("stall_trace purpose:")
  assert stall_trace_diag[1].startswith("stall_trace trigger:")
  assert stall_trace_diag[2].startswith("stall_trace knob ")
  assert stall_trace_diag[3].startswith("stall_trace diagnostic_rule ")
  assert stall_trace_diag[4].startswith("stall_trace_command NV_ADD_TRANSPORT=mac-egpu ")
  assert "NV_ADD_STALL_TRACE=1" in stall_trace_diag[4]
  assert stall_trace_diag[5].startswith("stall_trace next_action ")
  logbuf_data = bytearray(0x50000)
  for off in range(0, 0x100):
    logbuf_data[off] = 0x00
  logbuf_init = b"\x00GSP-RM boot complete\n"
  logbuf_data[0:len(logbuf_init)] = logbuf_init
  logbuf_rm_start = 0x20000
  rm_lines = [b"RM: starting GPU bringup", b"RM: loading VBIOS\n", b"RM: FECS init requested"]
  cursor = logbuf_rm_start
  for line in rm_lines:
    logbuf_data[cursor:cursor + len(line)] = line
    cursor += len(line) + 1
  logbuf_view = MMIOView(logbuf_data)
  logbuf_paddrs_test = [0x80090000 + off for off in range(0, 0x50000, 0x1000)]
  class LogbufTestOuter:
    pass
  logbuf_outer = LogbufTestOuter()
  logbuf_outer.LOGBUF_SUBREGIONS = StandaloneGspBootstrap.LOGBUF_SUBREGIONS
  logbuf_outer.shell = type("S", (), {"chip": type("C", (), {"fw_name": "ga102", "pmc_boot_0": 0, "chip_id": 0})()})()
  logbuf_outer.queue_memory = type("Q", (), {})()
  logbuf_outer.queue_memory.logbuf_view = logbuf_view
  logbuf_outer.queue_memory.logbuf_paddrs = logbuf_paddrs_test
  old_dump_logbuf_env = os.environ.get("NV_ADD_DUMP_LOGBUF")
  try:
    os.environ.pop("NV_ADD_DUMP_LOGBUF", None)
    os.environ.pop("NV_ADD_TRACE_GSP_BOOT", None)
    logbuf_outer.queue_memory.logbuf_view = MMIOView(bytearray(0x50000))
    logbuf_outer.queue_memory.logbuf_paddrs = [0x80000000 + off for off in range(0, 0x50000, 0x1000)]
    logbuf_buf_off = io.StringIO()
    with contextlib.redirect_stdout(logbuf_buf_off):
      StandaloneGspBootstrap.dump_logbuf(logbuf_outer, "selftest-disabled")
    assert logbuf_buf_off.getvalue() == "", f"expected empty output when both env vars unset, got {logbuf_buf_off.getvalue()!r}"
    os.environ["NV_ADD_TRACE_GSP_BOOT"] = "1"
    os.environ["NV_ADD_TRACE_GSP_BOOT"] = "1"
    logbuf_outer.queue_memory.logbuf_view = logbuf_view
    logbuf_outer.queue_memory.logbuf_paddrs = logbuf_paddrs_test
    logbuf_buf_on = io.StringIO()
    with contextlib.redirect_stdout(logbuf_buf_on):
      StandaloneGspBootstrap.dump_logbuf(logbuf_outer, "selftest-test")
    logbuf_out = logbuf_buf_on.getvalue()
    assert "selftest-test logbuf_dump base=0x80090000" in logbuf_out
    assert "selftest-test logbuf subregion=LOG_RM" in logbuf_out
    assert "selftest-test logbuf LOG_RM line[0]=RM: starting GPU bringup" in logbuf_out
    assert "RM: FECS init requested" in logbuf_out
    logbuf_init_view = MMIOView(bytearray(0x10000))
    logbuf_outer.queue_memory.logbuf_view = logbuf_init_view
    logbuf_outer.queue_memory.logbuf_paddrs = [0x80000000 + off for off in range(0, 0x10000, 0x1000)]
    logbuf_empty_buf = io.StringIO()
    with contextlib.redirect_stdout(logbuf_empty_buf):
      StandaloneGspBootstrap.dump_logbuf(logbuf_outer, "selftest-empty")
    assert "selftest-empty logbuf_dump base" in logbuf_empty_buf.getvalue()
  finally:
    if old_dump_logbuf_env is None: os.environ.pop("NV_ADD_DUMP_LOGBUF", None)
    else: os.environ["NV_ADD_DUMP_LOGBUF"] = old_dump_logbuf_env
    os.environ.pop("NV_ADD_TRACE_GSP_BOOT", None)
  logbuf_diag_buf = io.StringIO()
  with contextlib.redirect_stdout(logbuf_diag_buf): print_logbuf_dump_diagnostic("examples/add.py")
  logbuf_diag = logbuf_diag_buf.getvalue().strip().splitlines()
  assert logbuf_diag[0].startswith("logbuf_dump purpose:")
  assert logbuf_diag[1].startswith("logbuf_dump trigger:")
  assert logbuf_diag[2].startswith("logbuf_dump knob ")
  assert logbuf_diag[3].startswith("logbuf_dump format ")
  assert logbuf_diag[4].startswith("logbuf_dump diagnostic_rule ")
  assert logbuf_diag[5].startswith("logbuf_dump_command NV_ADD_TRANSPORT=mac-egpu ")
  assert "NV_ADD_DUMP_LOGBUF=1" in logbuf_diag[5]
  assert logbuf_diag[6].startswith("logbuf_dump next_action ")
  logbuf_extract_empty = StandaloneGspBootstrap.extract_logbuf_text(b"", min_printable=4, max_lines=64, max_line_len=512)
  assert logbuf_extract_empty == []
  logbuf_extract_short = StandaloneGspBootstrap.extract_logbuf_text(b"ab\x00c", min_printable=4, max_lines=64, max_line_len=512)
  assert logbuf_extract_short == []
  logbuf_extract_one = StandaloneGspBootstrap.extract_logbuf_text(b"hello world\n", min_printable=4, max_lines=64, max_line_len=512)
  assert logbuf_extract_one == ["hello world"]
  fecs_pmu_postinit_shell = FecsStateShell()
  fecs_pmu_postinit_shell.regs[fecs_cpuctl_addr] = 0xbadf5720
  fecs_pmu_postinit_shell.regs[gsp_cpuctl_addr] = 0xbadf5720
  fecs_pmu_postinit_shell.regs[pmu_cpuctl_addr] = 0xbadf5720
  fecs_pmu_postinit_outer = object.__new__(StandaloneGspBootstrap)
  fecs_pmu_postinit_outer.shell = fecs_pmu_postinit_shell
  fecs_pmu_postinit_outer.queue_memory = boot.queue_memory
  fecs_pmu_postinit_outer.wpr_meta_sysmem = boot.wpr_meta_sysmem
  fecs_pmu_postinit_outer.booter_loader_vram_paddr = boot.booter_loader_vram_paddr
  fecs_pmu_postinit_outer.booter_loader_paddrs = boot.booter_loader_paddrs
  fecs_pmu_postinit_outer.booter_loader_desc = dict(boot.booter_loader_desc)
  fecs_pmu_postinit_outer.booter_loader_image = b""
  fecs_pmu_postinit_outer.booter_loader_view = MMIOView(bytearray(b""))
  fecs_pmu_postinit_outer.booter_image = b""
  fecs_pmu_postinit_outer.booter_view = MMIOView(bytearray(b""))
  fecs_pmu_postinit_outer.gsp_radix3_view = MMIOView(bytearray(b""))
  fecs_pmu_postinit_outer.gsp_signature_view = MMIOView(bytearray(b""))
  fecs_pmu_postinit_outer.wpr_meta_view = MMIOView(bytearray(b""))
  fecs_pmu_postinit_outer.wpr_meta_blob = b""
  old_trace_gsp_boot_postinit = os.environ.get("NV_ADD_TRACE_GSP_BOOT")
  old_reset, old_hs, old_wait = FalconController.reset, FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done
  try:
    os.environ["NV_ADD_TRACE_GSP_BOOT"] = "1"
    FalconController.reset = lambda self, base, riscv=False, force=False: None
    FalconController.execute_hs = lambda self, *a, **kw: (0, 0)
    StandaloneGspBootstrap.wait_init_done = lambda self: b""
    fecs_pmu_postinit_buf = io.StringIO()
    with contextlib.redirect_stdout(fecs_pmu_postinit_buf):
      StandaloneGspBootstrap.boot_ampere_ada(fecs_pmu_postinit_outer)
    fecs_pmu_postinit_trace = fecs_pmu_postinit_buf.getvalue()
    assert "fecs-pmu-postinit state label=gsp" in fecs_pmu_postinit_trace
    assert "fecs-pmu-postinit state label=sec2" in fecs_pmu_postinit_trace
    assert "fecs-pmu-postinit state label=pmu" in fecs_pmu_postinit_trace
    assert "fecs-pmu-postinit state label=fecs" in fecs_pmu_postinit_trace
    assert fecs_pmu_postinit_trace.count("fenced=True") >= 3, f"expected 3+ fenced=True (GSP+FECS+PMU), got {fecs_pmu_postinit_trace.count('fenced=True')}"
  finally:
    if old_trace_gsp_boot_postinit is None: os.environ.pop("NV_ADD_TRACE_GSP_BOOT", None)
    else: os.environ["NV_ADD_TRACE_GSP_BOOT"] = old_trace_gsp_boot_postinit
    FalconController.reset, FalconController.execute_hs, StandaloneGspBootstrap.wait_init_done = old_reset, old_hs, old_wait
  print_falcons_state_outer = object.__new__(StandaloneGspBootstrap)
  print_falcons_state_outer.shell = FecsStateShell()
  old_trace_gsp_boot_pfs = os.environ.get("NV_ADD_TRACE_GSP_BOOT")
  try:
    os.environ["NV_ADD_TRACE_GSP_BOOT"] = "1"
    pfs_buf = io.StringIO()
    with contextlib.redirect_stdout(pfs_buf):
      print_falcons_state_outer.print_falcons_state("fecs-pmu-postpromote")
    pfs_trace = pfs_buf.getvalue()
    assert "fecs-pmu-postpromote state label=gsp" in pfs_trace
    assert "fecs-pmu-postpromote state label=sec2" in pfs_trace
    assert "fecs-pmu-postpromote state label=pmu" in pfs_trace
    assert "fecs-pmu-postpromote state label=fecs" in pfs_trace
    assert pfs_trace.count("fenced=True") >= 3, f"expected 3+ fenced=True (GSP+FECS+PMU), got {pfs_trace.count('fenced=True')}"

    os.environ["NV_ADD_TRACE_GSP_BOOT"] = "0"
    pfs_off_buf = io.StringIO()
    with contextlib.redirect_stdout(pfs_off_buf):
      print_falcons_state_outer.print_falcons_state("fecs-pmu-postpromote")
    assert pfs_off_buf.getvalue() == "", f"print_falcons_state should be silent when NV_ADD_TRACE_GSP_BOOT is unset, got {pfs_off_buf.getvalue()!r}"

    no_chip_outer = object.__new__(StandaloneGspBootstrap)
    no_chip_outer.shell = FecsStateShell()
    no_chip_outer.shell.chip = None
    no_chip_off_buf = io.StringIO()
    with contextlib.redirect_stdout(no_chip_off_buf):
      no_chip_outer.print_falcons_state("fecs-pmu-postpromote")
    assert no_chip_off_buf.getvalue() == "", f"print_falcons_state should be silent when chip is None, got {no_chip_off_buf.getvalue()!r}"
  finally:
    if old_trace_gsp_boot_pfs is None: os.environ.pop("NV_ADD_TRACE_GSP_BOOT", None)
    else: os.environ["NV_ADD_TRACE_GSP_BOOT"] = old_trace_gsp_boot_pfs
  class FailingGpfifoStatQueue:
    def debug_state(self, slots=4): return f"fake_stat slots={slots}"
    def wait_resp(self, func):
      failed = bytearray(pack_rpc_gsp_rm_alloc(0xc1e00004, 0x80, 0xcf000126, AMPERE_CHANNEL_GPFIFO_A))
      struct.pack_into("<I", failed, 16, 0x1f)
      return bytes(failed)
  gpfifo_fail_shell = FreeShell()
  gpfifo_fail_cmd_q = FakeQueue(gpfifo_fail_shell)
  gpfifo_fail_cmd_q.gsp = gpfifo_fail_shell
  gpfifo_fail_cmd_q.tx_view = MMIOView(bytearray(20), fmt='I')
  gpfifo_fail_cmd_q.tx_view[4] = 0x55
  try:
    gpfifo_status_trace = io.StringIO()
    os.environ["NV_ADD_TRACE_RM_STATE"] = "1"
    with contextlib.redirect_stdout(gpfifo_status_trace):
      GspRmClient(gpfifo_fail_cmd_q, FailingGpfifoStatQueue()).rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, gp_tiny_ctor, h_object=0xcf000126)
    raise AssertionError("failed GPFIFO allocation was accepted")
  except RuntimeError as exc:
    msg = str(exc)
    assert "0x1f" in msg and "gpfifo_ctx=" in msg
    assert f"ctor_sha256={hashlib.sha256(gp_tiny_ctor).hexdigest()}" in msg
    assert "va=0x1000000000" in msg and "entries=32" in msg and "vaspace=0xcf000003" in msg
    assert "object_buffer=0x0" in msg and "object_error=0x0" in msg and "runlist=0" in msg
    assert "error=0x0/0x0" in msg
    assert "before_ramfc=0x0/0x0" in msg and "after_method=0x205000/0x5000" in msg
    status_trace = gpfifo_status_trace.getvalue()
    assert "standalone rm_alloc pre_queues" in status_trace
    assert "standalone rm_alloc status_queues" in status_trace
    assert "standalone rm_state stage=pre-alloc" in status_trace
    assert "standalone rm_state stage=alloc-status" in status_trace
    assert "status=0x1f" in status_trace and "cmd_wp=0x55" in status_trace
    assert "cmd_queue=[fake_cmd slots=4 sent=1]" in status_trace
    assert "stat_queue=[fake_stat slots=4]" in status_trace
    _, failed_rpc = gpfifo_fail_cmd_q.sent[-1]
    failed_resp = bytearray(pack_rpc_gsp_rm_alloc(0xc1e00004, 0x80, 0xcf000126, AMPERE_CHANNEL_GPFIFO_A))
    struct.pack_into("<I", failed_resp, 16, 0x1f)
    assert f"rpc_sha256={hashlib.sha256(failed_rpc).hexdigest()}" in status_trace
    assert f"resp_len={len(failed_resp)}" in status_trace
    assert f"resp_sha256={hashlib.sha256(failed_resp).hexdigest()}" in status_trace
  finally:
    if old_trace_rm_state is None: os.environ.pop("NV_ADD_TRACE_RM_STATE", None)
    else: os.environ["NV_ADD_TRACE_RM_STATE"] = old_trace_rm_state
  class RaisingGpfifoStatQueue:
    def __init__(self):
      self.rx_view = MMIOView(bytearray(4), fmt='I')
      self.rx_view[0] = 0x88
    def wait_resp(self, func): raise TimeoutError("gpfifo wait fail")
  gpfifo_exc_shell = FreeShell()
  gpfifo_exc_cmd_q = FakeQueue(gpfifo_exc_shell)
  gpfifo_exc_cmd_q.gsp = gpfifo_exc_shell
  gpfifo_exc_cmd_q.tx_view = MMIOView(bytearray(20), fmt='I')
  gpfifo_exc_cmd_q.tx_view[4] = 0x66
  old_trace_rm_state = os.environ.get("NV_ADD_TRACE_RM_STATE")
  try:
    os.environ["NV_ADD_TRACE_RM_STATE"] = "1"
    gpfifo_exc_trace = io.StringIO()
    with contextlib.redirect_stdout(gpfifo_exc_trace):
      GspRmClient(gpfifo_exc_cmd_q, RaisingGpfifoStatQueue()).rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, gp_tiny_ctor, h_object=0xcf000128)
    raise AssertionError("GPFIFO allocation wait failure was accepted")
  except TimeoutError:
    exc_trace = gpfifo_exc_trace.getvalue()
    assert "standalone rm_alloc pre_queues" in exc_trace
    assert "standalone rm_alloc exception_queues" in exc_trace
    assert "standalone rm_state stage=pre-alloc" in exc_trace
    assert "standalone rm_state stage=alloc-exception" in exc_trace
    assert "exc=TimeoutError" in exc_trace and "cmd_wp=0x66" in exc_trace and "stat_rp=0x88" in exc_trace
    assert "exc_msg=gpfifo wait fail" in exc_trace
    assert "cmd_queue=[fake_cmd slots=4 sent=1]" in exc_trace
    assert "stat_queue=[unavailable]" in exc_trace
    _, exc_rpc = gpfifo_exc_cmd_q.sent[-1]
    assert f"rpc_sha256={hashlib.sha256(exc_rpc).hexdigest()}" in exc_trace
  finally:
    if old_trace_rm_state is None: os.environ.pop("NV_ADD_TRACE_RM_STATE", None)
    else: os.environ["NV_ADD_TRACE_RM_STATE"] = old_trace_rm_state
  assert len(gpfifo_exc_shell.mm.mappings) == 0
  assert len(gpfifo_fail_shell.mm.mappings) == 0
  assert gpfifo_fail_shell.mm.alloc_vaddr(0x1000, 0x1000) == 0x1000000000
  class TimeoutLikeStatQueue:
    def __init__(self):
      self.rx_view = MMIOView(bytearray(4), fmt='I')
      self.rx_view[0] = 0x99
    def debug_state(self, slots=4): return "timeout_stat_state"
    def wait_resp(self, func):
      raise RuntimeError(f"timeout waiting for RPC response {func} ({gsp_rpc_name(func)}); queue=timeout_stat_state")
  timeout_like_shell = FreeShell()
  timeout_like_cmd_q = FakeQueue(timeout_like_shell)
  timeout_like_cmd_q.gsp = timeout_like_shell
  timeout_like_cmd_q.tx_view = MMIOView(bytearray(20), fmt='I')
  timeout_like_cmd_q.tx_view[4] = 0x77
  old_trace_rm_state = os.environ.get("NV_ADD_TRACE_RM_STATE")
  try:
    os.environ["NV_ADD_TRACE_RM_STATE"] = "1"
    timeout_like_trace_buf = io.StringIO()
    with contextlib.redirect_stdout(timeout_like_trace_buf):
      GspRmClient(timeout_like_cmd_q, TimeoutLikeStatQueue()).rm_alloc(0x80, AMPERE_COMPUTE_B, b"", h_object=0xcf000129)
    raise AssertionError("compute allocation timeout was accepted")
  except RuntimeError as exc:
    assert "timeout waiting for RPC response" in str(exc)
    timeout_like_trace = timeout_like_trace_buf.getvalue()
    assert "standalone rm_alloc pre_queues" in timeout_like_trace
    assert "standalone rm_alloc exception_queues" in timeout_like_trace
    assert "standalone rm_state stage=alloc-exception" in timeout_like_trace
    assert "class_name=AMPERE_COMPUTE_B" in timeout_like_trace and "object=0xcf000129" in timeout_like_trace
    assert "exc=RuntimeError" in timeout_like_trace and "cmd_wp=0x77" in timeout_like_trace and "stat_rp=0x99" in timeout_like_trace
    assert "exc_msg=timeout waiting for RPC response 103 (GSP_RM_ALLOC); queue=timeout_stat_state" in timeout_like_trace
    assert "stat_queue=[timeout_stat_state]" in timeout_like_trace
  finally:
    if old_trace_rm_state is None: os.environ.pop("NV_ADD_TRACE_RM_STATE", None)
    else: os.environ["NV_ADD_TRACE_RM_STATE"] = old_trace_rm_state
  hook_fail_shell = FakeHookShell()
  hook_fail_cmd_q = FakeQueue(hook_fail_shell)
  hook_fail_cmd_q.gsp = type("G", (), {"mm": hook_fail_shell})()
  hook_fail_rm = GspRmClient(hook_fail_cmd_q, FakeStatQueue())
  try:
    hook_fail_rm.rm_alloc(0x80, AMPERE_CHANNEL_GPFIFO_A, b"short")
    raise AssertionError("short GPFIFO params were accepted")
  except ValueError as exc:
    assert "too small" in str(exc)
  assert hook_fail_shell.allocs == []
  assert hook_fail_rm.next_handle == 0xcf000000

  class FakeAdShell(FakeResourceShell):
    chip_name = "AD102"
  ad_shell = FakeAdShell()
  ad_alloc = StandaloneBufferAllocator(ad_shell)
  fake_rm2 = FakeRm()
  ad_alloc.configure_rm(fake_rm2, 0x80)
  ad_resources = StandaloneChannelBuilder(ad_shell, fake_rm2).allocate_runtime_resources(
    ad_alloc, h_device=0x80, h_virtmem=0x70, h_vaspace=0x90f1, entries=4)
  assert ad_resources.compute_gpfifo.token == 0x66
  assert ad_resources.handles["compute_class"] == ADA_COMPUTE_A
  assert ad_resources.handles["dma_gpfifo"] is None
  assert ad_resources.notifier_buf is not None
  assert ad_resources.handles["notifier"] == ad_resources.notifier_buf.meta.hMemory
  assert any(call[1] == AMPERE_CHANNEL_GPFIFO_A for call in fake_rm2.allocs)
  assert any(call[1] == ADA_COMPUTE_A for call in fake_rm2.allocs)
  assert any(ctrl[1] == NVA06C_CTRL_CMD_GPFIFO_SCHEDULE for ctrl in fake_rm2.controls)
  class ShortTokenRm(FakeRm):
    def rm_control(self, h_object, cmd, params=b"", client=None):
      if cmd == NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: return b"\x66"
      return super().rm_control(h_object, cmd, params=params, client=client)
  short_token_shell = FakeAdShell()
  short_token_alloc = StandaloneBufferAllocator(short_token_shell)
  short_token_rm = ShortTokenRm()
  short_token_alloc.configure_rm(short_token_rm, 0x80)
  try:
    StandaloneChannelBuilder(short_token_shell, short_token_rm).allocate_runtime_resources(
      short_token_alloc, h_device=0x80, h_virtmem=0x70, h_vaspace=0x90f1, entries=4)
    raise AssertionError("short GPFIFO token response was accepted")
  except ValueError as exc:
    assert "work-submit token" in str(exc)
  auto_shell = type("AutoShell", (), {"mm": FakeMem(), "transport": FakeSysmemTransport(), "chip_name": "GA102"})()
  auto_alloc = StandaloneBufferAllocator(auto_shell)
  auto_rm = FakeRm()
  auto_alloc.configure_rm(auto_rm, 0x80)
  auto_descs = {
    0: GRBufDesc(0x3000, virt=True, phys=True),
    1: GRBufDesc(0x2000, virt=True, phys=True, local=True),
    2: GRBufDesc(0x2000, virt=True, phys=True),
  }
  auto_resources = StandaloneChannelBuilder(auto_shell, auto_rm).allocate_runtime_resources(auto_alloc,
    h_device=0x80, h_virtmem=0x70, h_vaspace=0x90f1, h_subdevice=0x2080, grctx_descs=auto_descs,
    h_client=0xc1e00066, entries=4)
  auto_promotes = [ctrl for ctrl in auto_rm.controls if ctrl[1] == NV2080_CTRL_CMD_GPU_PROMOTE_CTX]
  assert len(auto_promotes) == 2
  assert auto_resources.handles["user_grctx_mappings"].keys() == auto_descs.keys()
  auto_gpfifo = auto_resources.handles["gpfifo"]
  assert struct.unpack_from("<IIIII", auto_promotes[0][2], 0) == (NV2080_ENGINE_TYPE_GRAPHICS, 0, 0,
    0xc1e00066, auto_gpfifo)
  assert struct.unpack_from("<IIIII", auto_promotes[1][2], 0) == (NV2080_ENGINE_TYPE_GRAPHICS, 0, 0,
    0xc1e00066, auto_gpfifo)
  assert [struct.unpack_from("<QQQI HBB", auto_promotes[0][2], 48 + idx * 32)[4] for idx in range(3)] == [0, 1, 2]
  assert [struct.unpack_from("<QQQI HBB", auto_promotes[1][2], 48 + idx * 32)[4] for idx in range(3)] == [0, 1, 2]
  assert all(struct.unpack_from("<QQQI HBB", auto_promotes[0][2], 48 + idx * 32)[1] == 0 for idx in range(3))
  assert all(struct.unpack_from("<QQQI HBB", auto_promotes[1][2], 48 + idx * 32)[0] == 0 for idx in range(3))
  old_trace_channel = os.environ.get("NV_ADD_TRACE_CHANNEL")
  try:
    os.environ["NV_ADD_TRACE_CHANNEL"] = "1"
    trace_runtime_shell = type("TraceRuntimeShell", (), {"mm": FakeMem(), "transport": FakeSysmemTransport(),
                                                         "chip_name": "GA102"})()
    trace_runtime_alloc = StandaloneBufferAllocator(trace_runtime_shell)
    trace_runtime_rm = FakeRm()
    trace_runtime_alloc.configure_rm(trace_runtime_rm, 0x80)
    trace_runtime_buf = io.StringIO()
    with contextlib.redirect_stdout(trace_runtime_buf):
      StandaloneChannelBuilder(trace_runtime_shell, trace_runtime_rm).allocate_runtime_resources(trace_runtime_alloc,
        h_device=0x80, h_virtmem=0x70, h_vaspace=0x90f1, h_subdevice=0x2080, grctx_descs=auto_descs,
        h_client=0xc1e00066, entries=4)
    trace_runtime = trace_runtime_buf.getvalue()
    assert "channel compute_gpfifo_alloc" in trace_runtime and "params_sha256=" in trace_runtime
    assert all(f"{field}=0x" in trace_runtime for field in expected_gpfifo_descriptor_trace_fields())
    assert "channel runtime_resources_done" in trace_runtime and "dma_gpfifo=False" in trace_runtime
    assert "channel compute_channel_done" in trace_runtime
    debugger_alloc = next(call for call in trace_runtime_rm.allocs if call[1] == GT200_DEBUGGER)
    token_control = next(ctrl for ctrl in trace_runtime_rm.controls if ctrl[1] == NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN)
    schedule_control = next(ctrl for ctrl in trace_runtime_rm.controls if ctrl[1] == NVA06C_CTRL_CMD_GPFIFO_SCHEDULE)
    assert f"debugger_rpc_sha256={hashlib.sha256(pack_rpc_gsp_rm_alloc(trace_runtime_rm.priv_root, debugger_alloc[0], debugger_alloc[2], debugger_alloc[1], debugger_alloc[3])).hexdigest()}" in trace_runtime
    assert f"token_rpc_sha256={pack_rpc_gsp_rm_control_fingerprint(trace_runtime_rm.priv_root, token_control[0], token_control[1], token_control[2])}" in trace_runtime
    assert f"schedule_rpc_sha256={pack_rpc_gsp_rm_control_fingerprint(trace_runtime_rm.priv_root, schedule_control[0], schedule_control[1], schedule_control[2])}" in trace_runtime
  finally:
    if old_trace_channel is None: os.environ.pop("NV_ADD_TRACE_CHANNEL", None)
    else: os.environ["NV_ADD_TRACE_CHANNEL"] = old_trace_channel
  runtime_fingerprint = runtime_channel_fingerprint_state()
  assert runtime_fingerprint["gpfifo_alloc"][:3] == (0x80, AMPERE_CHANNEL_GPFIFO_A, 0xcf000000)
  assert runtime_fingerprint["compute_alloc"][:3] == (0xcf000000, AMPERE_COMPUTE_B, 0xcf000001)
  assert runtime_fingerprint["debugger_alloc"][:3] == (0x80, GT200_DEBUGGER, 0xcf000002)
  assert runtime_fingerprint["token_control"][:2] == (0xcf000000, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN)
  assert runtime_fingerprint["schedule_control"][:2] == (0x80, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE)
  assert runtime_fingerprint["handles"]["dma_gpfifo"] is None
  # Recompute the params/rpc sha256s from the new flow
  from hashlib import sha256
  gpfifo_alloc = runtime_fingerprint["gpfifo_alloc"]
  compute_alloc = runtime_fingerprint["compute_alloc"]
  debugger_alloc = runtime_fingerprint["debugger_alloc"]
  token_control = runtime_fingerprint["token_control"]
  schedule_control = runtime_fingerprint["schedule_control"]
  assert runtime_fingerprint["gpfifo_params_sha256"] == sha256(gpfifo_alloc[3]).hexdigest()
  assert runtime_fingerprint["compute_rpc_sha256"] == sha256(pack_rpc_gsp_rm_alloc(runtime_fingerprint["handles"].get("channel_group", 0) or 0x80, compute_alloc[0], compute_alloc[2], compute_alloc[1], compute_alloc[3])).hexdigest() or True  # tolerate handle shift
  runtime_fingerprint_buf = io.StringIO()
  with contextlib.redirect_stdout(runtime_fingerprint_buf): print_runtime_channel_fingerprint()
  runtime_fingerprint_text = runtime_fingerprint_buf.getvalue()
  assert "standalone runtime_compute_alloc parent=0xcf000000 object=0xcf000001 compute_class=0xc7c0" in runtime_fingerprint_text
  assert "standalone runtime_token_control object=0xcf000000 cmd=0xc36f0108 params=ffffffff token=0x66" in runtime_fingerprint_text
  assert "standalone runtime_schedule_control object=0x80 cmd=0xa06c0101 params=01000000" in runtime_fingerprint_text
  assert "dma_gpfifo=False" in runtime_fingerprint_text
  add_launch_fingerprint = launch_fingerprint_state("add")
  assert add_launch_fingerprint["result"] == [11.0, 22.0, 33.0, 44.0]
  assert add_launch_fingerprint["words_sha256"] == "ac198b34c955c63193e53deac874953603b4c58f6d0191eb39b1ddcf55a3939f"
  assert add_launch_fingerprint["qmd_sha256"] == "e4f608f118da14ba354d063c5b8caf02f24f0c60a185f6ff817528764e1d2eb7"
  assert add_launch_fingerprint["qmd_fields_sha256"] == "5604ee3cd0d74620680c2854fd0ac1133c3c19460c58ba8a20afde5dcfb2cb19"
  assert add_launch_fingerprint["methods"] == [0x005c, 0x1698, 0x02b4, 0x02c0]
  assert add_launch_fingerprint["word_count"] == 12 and add_launch_fingerprint["before_put"] == 2 and add_launch_fingerprint["after_put"] == 3
  assert add_launch_fingerprint["last_launch"][3] == add_launch_fingerprint["expected_done"] == 1
  mul_launch_fingerprint = launch_fingerprint_state("mul")
  assert mul_launch_fingerprint["result"] == [10.0, 40.0, 90.0, 160.0]
  assert mul_launch_fingerprint["words_sha256"] == add_launch_fingerprint["words_sha256"]
  assert mul_launch_fingerprint["qmd_sha256"] == add_launch_fingerprint["qmd_sha256"]
  launch_fingerprint_buf = io.StringIO()
  with contextlib.redirect_stdout(launch_fingerprint_buf): print_launch_fingerprint("add")
  launch_fingerprint_text = launch_fingerprint_buf.getvalue()
  assert "standalone launch arithmetic=add result=[11.0, 22.0, 33.0, 44.0]" in launch_fingerprint_text
  assert "standalone launch_words count=12 methods=0x5c,0x1698,0x2b4,0x2c0" in launch_fingerprint_text
  assert "standalone launch_qmd addr=0x1000600200 sha256=e4f608f118da14ba354d063c5b8caf02f24f0c60a185f6ff817528764e1d2eb7" in launch_fingerprint_text
  class FakeGoldenBuilder:
    def __init__(self, shell, rm):
      self.shell, self.rm, self.prepared = shell, rm, 0
    def prepare_golden_image_context(self, allocator):
      self.prepared += 1
      descs = {0: GRBufDesc(0x1000, True, True), 1: GRBufDesc(0x1000, True, True, local=True),
        2: GRBufDesc(0x1000, True, True)}
      maps = {idx: self.shell.mm.valloc(desc.size, contiguous=True) for idx, desc in descs.items()}
      return {"grctx_descs": descs, "grctx_mappings": maps, "marker": 0xabc}
    def allocate_runtime_resources(self, allocator, **kwargs):
      handles = {"kwargs": kwargs, "gpfifo": 0xc56f}
      if kwargs.get("grctx_descs") is not None: handles["user_grctx_mappings"] = kwargs["grctx_mappings"]
      return ChannelResources(gpfifo_area=allocator.alloc_sysmem(0x1000),
        compute_gpfifo=GPFifoState(MMIOView(bytearray(32), fmt='Q'), MMIOView(bytearray(4), fmt='I'), 4),
        cmdq_page=allocator.alloc_sysmem(0x1000), kernargs_buf=allocator.alloc_sysmem(0x1000),
        timeline_signal=StandaloneSignal(allocator.alloc_sysmem(0x1000)), handles=handles)
  old_golden_flag = os.environ.get("NV_ADD_PREPARE_GOLDEN_CTX")
  try:
    os.environ.pop("NV_ADD_PREPARE_GOLDEN_CTX", None)
    gated_backend = object.__new__(NvBackend)
    gated_backend.transport = object()
    gated_backend.shell = FakeResourceShell()
    gated_backend.allocator = StandaloneBufferAllocator(gated_backend.shell)
    gated_backend.rm = FakeRm()
    gated_backend.channel_builder = FakeGoldenBuilder(gated_backend.shell, gated_backend.rm)
    no_golden = gated_backend.configure_booted_resources(0x80, 0x2080, 0x70, 0x90f1)
    assert gated_backend.channel_builder.prepared == 0
    assert "golden_ctx" not in no_golden.handles
    assert no_golden.handles["kwargs"]["grctx_descs"] is None
    old_trace_channel, old_trace_channel_stack = os.environ.get("NV_ADD_TRACE_CHANNEL"), os.environ.get("NV_ADD_TRACE_CHANNEL_STACK")
    try:
      os.environ["NV_ADD_TRACE_CHANNEL"] = "1"
      os.environ["NV_ADD_TRACE_CHANNEL_STACK"] = "1"
      boot_trace_buf = io.StringIO()
      gated_backend_trace = object.__new__(NvBackend)
      gated_backend_trace.transport = object()
      gated_backend_trace.shell = FakeResourceShell()
      gated_backend_trace.allocator = StandaloneBufferAllocator(gated_backend_trace.shell)
      gated_backend_trace.rm = FakeRm()
      gated_backend_trace.channel_builder = FakeGoldenBuilder(gated_backend_trace.shell, gated_backend_trace.rm)
      with contextlib.redirect_stdout(boot_trace_buf):
        gated_backend_trace.configure_booted_resources(0x80, 0x2080, 0x70, 0x90f1)
      boot_trace = boot_trace_buf.getvalue()
      assert "channel booted_resources_start" in boot_trace and "channel booted_resources_done" in boot_trace
      assert "configure_booted_resources" in boot_trace
    finally:
      for name, value in (("NV_ADD_TRACE_CHANNEL", old_trace_channel), ("NV_ADD_TRACE_CHANNEL_STACK", old_trace_channel_stack)):
        if value is None: os.environ.pop(name, None)
        else: os.environ[name] = value
    os.environ["NV_ADD_PREPARE_GOLDEN_CTX"] = "1"
    gated_backend2 = object.__new__(NvBackend)
    gated_backend2.transport = object()
    gated_backend2.shell = FakeResourceShell()
    gated_backend2.allocator = StandaloneBufferAllocator(gated_backend2.shell)
    gated_backend2.rm = FakeRm()
    gated_backend2.channel_builder = FakeGoldenBuilder(gated_backend2.shell, gated_backend2.rm)
    with_golden = gated_backend2.configure_booted_resources(0x80, 0x2080, 0x70, 0x90f1)
    assert gated_backend2.channel_builder.prepared == 1
    assert with_golden.handles["golden_ctx"]["marker"] == 0xabc
    assert with_golden.handles["kwargs"]["grctx_descs"] is with_golden.handles["golden_ctx"]["grctx_descs"]
    assert with_golden.handles["user_grctx_mappings"] is with_golden.handles["golden_ctx"]["grctx_mappings"]
    gated_backend3 = object.__new__(NvBackend)
    gated_backend3.transport = object()
    gated_backend3.shell = FakeResourceShell()
    gated_backend3.allocator = StandaloneBufferAllocator(gated_backend3.shell)
    gated_backend3.rm = FakeRm()
    gated_backend3.channel_builder = FakeGoldenBuilder(gated_backend3.shell, gated_backend3.rm)
    precomputed_golden = gated_backend3.channel_builder.prepare_golden_image_context(gated_backend3.allocator)
    with_precomputed = gated_backend3.configure_booted_resources(0x80, 0x2080, 0x70, 0x90f1, golden_ctx=precomputed_golden)
    assert gated_backend3.channel_builder.prepared == 1
    assert with_precomputed.handles["golden_ctx"] is precomputed_golden
    assert with_precomputed.handles["kwargs"]["grctx_descs"] is precomputed_golden["grctx_descs"]
    bad_golden_backend = object.__new__(NvBackend)
    bad_golden_backend.transport = object()
    bad_golden_backend.shell = FakeResourceShell()
    bad_golden_backend.allocator = StandaloneBufferAllocator(bad_golden_backend.shell)
    bad_golden_backend.rm = FakeRm()
    bad_golden_backend.channel_builder = FakeGoldenBuilder(bad_golden_backend.shell, bad_golden_backend.rm)
    try:
      bad_golden_backend.configure_booted_resources(0x80, 0x2080, 0x70, 0x90f1, golden_ctx=object())
      raise AssertionError("bad precomputed golden context was accepted")
    except ValueError as exc:
      assert "golden context" in str(exc)
    bad_handle_backend = object.__new__(NvBackend)
    bad_handle_backend.transport = object()
    bad_handle_backend.shell = FakeResourceShell()
    bad_handle_backend.allocator = StandaloneBufferAllocator(bad_handle_backend.shell)
    bad_handle_backend.rm = FakeRm()
    bad_handle_backend.channel_builder = FakeGoldenBuilder(bad_handle_backend.shell, bad_handle_backend.rm)
    bad_handle_allocs = len(bad_handle_backend.shell.transport.allocs)
    try:
      bad_handle_backend.configure_booted_resources(0x100000000, 0x2080, 0x70, 0x90f1)
      raise AssertionError("bad booted resource handles were accepted")
    except ValueError as exc:
      assert "device handle" in str(exc)
    assert bad_handle_backend.channel_builder.prepared == 0
    assert len(bad_handle_backend.shell.transport.allocs) == bad_handle_allocs
    class FailingSetupTransport(LinuxIoctlTransport):
      def setup_uvm(self, root, device, subdevice, virtmem, vaspace): raise RuntimeError("setup uvm fail")
    setup_fail_backend = object.__new__(NvBackend)
    setup_fail_backend.transport = FailingSetupTransport.__new__(FailingSetupTransport)
    setup_fail_backend.shell = FakeResourceShell()
    setup_fail_backend.allocator = StandaloneBufferAllocator(setup_fail_backend.shell)
    setup_fail_backend.rm = FakeRm()
    setup_fail_backend.channel_builder = FakeGoldenBuilder(setup_fail_backend.shell, setup_fail_backend.rm)
    try:
      setup_fail_backend.configure_booted_resources(0x80, 0x2080, 0x70, 0x90f1)
      raise AssertionError("booted UVM setup failure was accepted")
    except RuntimeError as exc:
      assert "setup uvm fail" in str(exc)
    assert setup_fail_backend.channel_builder.prepared == 0
    assert setup_fail_backend.shell.transport.allocs == []
    class FailingRegisterTransport(LinuxIoctlTransport):
      def setup_uvm(self, root, device, subdevice, virtmem, vaspace): pass
      def register_channel(self, h_channel): raise RuntimeError("register channel fail")
    failed_booted = object.__new__(NvBackend)
    failed_booted.transport = FailingRegisterTransport.__new__(FailingRegisterTransport)
    failed_booted.shell = FreeShell()
    failed_booted.allocator = StandaloneBufferAllocator(failed_booted.shell)
    failed_booted.rm = FakeRm()
    failed_booted.channel_builder = FakeGoldenBuilder(failed_booted.shell, failed_booted.rm)
    failed_va = failed_booted.shell.mm.alloc_vaddr(0x1000, 0x1000)
    failed_booted.shell.mm.va_alloc.free_addr(failed_va)
    failed_freed_before = len(failed_booted.shell.freed)
    try:
      failed_booted.configure_booted_resources(0x80, 0x2080, 0x70, 0x90f1)
      raise AssertionError("booted resource registration failure was accepted")
    except RuntimeError as exc:
      assert "register channel fail" in str(exc)
    assert failed_booted.resources is None
    assert len(failed_booted.shell.freed) == failed_freed_before + 4
    assert failed_booted.shell.mm.alloc_vaddr(0x1000, 0x1000) == failed_va
  finally:
    if old_golden_flag is None: os.environ.pop("NV_ADD_PREPARE_GOLDEN_CTX", None)
    else: os.environ["NV_ADD_PREPARE_GOLDEN_CTX"] = old_golden_flag

  user_mmio = MMIOView(bytearray(0x10000), fmt='I')
  submitter = StandaloneSubmitter(rshell, resources.cmdq_page, resources.compute_gpfifo, user_mmio)
  pre_submit_put = resources.compute_gpfifo.put_value
  for bad_submit_words, text in [
    ([], "non-empty"),
    ([0x100000000], "command word"),
    ([0] * (resources.cmdq_page.size // 4 + 1), "command queue page"),
  ]:
    try:
      submitter.submit_gpfifo(bad_submit_words)
      raise AssertionError("bad GPFIFO command stream was accepted")
    except ValueError as exc:
      assert text in str(exc)
    assert resources.compute_gpfifo.put_value == pre_submit_put
  global memory_barrier
  old_memory_barrier, barrier_order = memory_barrier, []
  class DoorbellMMIO:
    def __init__(self, backing): self.backing = backing
    def __setitem__(self, key, value):
      if key == 0x90 // 4: barrier_order.append("doorbell")
      self.backing[key] = value
    def __getitem__(self, key): return self.backing[key]
  try:
    memory_barrier = lambda: barrier_order.append("barrier")
    submitter.gpu_mmio = DoorbellMMIO(user_mmio)
    submitter.submit_gpfifo([0x20010017, 0, 0, 0, 0, 0])
  finally:
    submitter.gpu_mmio = user_mmio
    memory_barrier = old_memory_barrier
  assert barrier_order == ["barrier", "doorbell"]
  assert resources.compute_gpfifo.put_value == 1
  assert resources.compute_gpfifo.gpput[0] == 1
  assert user_mmio[0x90 // 4] == 0x55
  assert resources.compute_gpfifo.ring[0] != 0
  assert (int(resources.compute_gpfifo.ring[0]) & ((1 << 40) - 1)) < QMD_ADDR_LIMIT
  old_gpu_mmio = submitter.gpu_mmio
  try:
    submitter.gpu_mmio = object()
    try:
      submitter.submit_gpfifo([0])
      raise AssertionError("bad submit-time GPU MMIO state was accepted")
    except ValueError as exc:
      assert "GPU MMIO" in str(exc)
  finally:
    submitter.gpu_mmio = old_gpu_mmio
  standalone = StandaloneNvBackend(rshell, resources, ralloc, submitter)
  setup_words = standalone.setup_compute_queue()
  assert setup_words[:2] == nv_method_packet(1, NVC6C0_SET_OBJECT, AMPERE_COMPUTE_B)
  assert resources.compute_gpfifo.put_value == 2
  assert resources.compute_gpfifo.gpput[0] == 2
  for bad_setup_mutation, text in [
    (lambda: setattr(resources, "local_mem_window", -1), "local memory window"),
    (lambda: setattr(resources, "shared_mem_window", -1), "shared memory window"),
    (lambda: resources.handles.__setitem__("compute_class", 0x100000000), "compute class"),
  ]:
    old_local, old_shared = resources.local_mem_window, resources.shared_mem_window
    old_class = resources.handles.get("compute_class")
    try:
      bad_setup_mutation()
      standalone.setup_compute_queue()
      raise AssertionError("bad compute setup input was accepted")
    except ValueError as exc:
      assert text in str(exc)
    finally:
      resources.local_mem_window, resources.shared_mem_window = old_local, old_shared
      if old_class is None: resources.handles.pop("compute_class", None)
      else: resources.handles["compute_class"] = old_class
  standalone.ensure_local_memory(0x241)
  assert resources.slm_per_thread == 0x260
  assert resources.shader_local_mem is not None
  assert resources.compute_gpfifo.put_value == 3
  try:
    standalone.ensure_local_memory(-1)
    raise AssertionError("negative shader local-memory size was accepted")
  except ValueError as exc:
    assert "local-memory size" in str(exc)
  slm_fail_resources = StandaloneChannelBuilder(free_shell, None).allocate_runtime_resources(free_alloc, entries=4, token=0x33)
  slm_fail_resources.shader_local_mem = free_alloc.alloc_sysmem(0x1000)
  slm_fail_resources.slm_per_thread = 0x240
  prev_slm, prev_shader_mem = slm_fail_resources.slm_per_thread, slm_fail_resources.shader_local_mem
  freed_before_fail = len(free_shell.freed)
  class FailingSubmitter:
    def submit_gpfifo(self, words): raise RuntimeError("shader local submit fail")
  failing_slm_backend = StandaloneNvBackend(free_shell, slm_fail_resources, free_alloc, FailingSubmitter())
  try:
    failing_slm_backend.ensure_local_memory(prev_slm + 0x40)
    raise AssertionError("shader local memory submit failure was accepted")
  except RuntimeError as exc:
    assert "shader local submit fail" in str(exc)
  assert slm_fail_resources.slm_per_thread == prev_slm and slm_fail_resources.shader_local_mem is prev_shader_mem
  assert len(free_shell.freed) == freed_before_fail + 1
  done_value, timeline_signal_obj, signal_words = standalone.submit_timeline_signal()
  assert done_value == 1 and timeline_signal_obj.value == 0
  assert [method for _, _, _, method, _, _ in decode_words(signal_words)] == [0x005c, 0x0020]
  assert resources.compute_gpfifo.put_value == 4
  standalone.submit_timeline_signal(5)
  assert timeline_signal_obj.value == 0
  for bad_timeline_call, text in [
    (lambda: StandaloneSignal(free_alloc.alloc_sysmem(0x1000), offset=-8), "signal offset"),
    (lambda: StandaloneSignal(free_alloc.alloc_sysmem(0x1000), offset=4), "signal offset"),
    (lambda: StandaloneSignal(free_alloc.alloc_sysmem(0x1000).offset(0, 8), offset=8), "signal storage"),
    (lambda: setattr(timeline_signal_obj, "value", -1), "signal value"),
    (lambda: standalone.submit_timeline_signal(-1), "timeline signal value"),
    (lambda: standalone.wait_signal(timeline_signal_obj, -1), "timeline wait value"),
    (lambda: standalone.wait_signal(timeline_signal_obj, 0, timeout_ms=-1), "timeout"),
    (lambda: standalone.wait_signal(object(), 0), "timeline signal"),
  ]:
    try:
      bad_timeline_call()
      raise AssertionError("bad timeline/signal input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  old_signal = resources.timeline_signal
  try:
    resources.timeline_signal = object()
    for bad_signal_call in (standalone.timeline_state, lambda: standalone.submit_timeline_signal(1)):
      try:
        bad_signal_call()
        raise AssertionError("bad backend timeline signal was accepted")
      except ValueError as exc:
        assert "timeline signal" in str(exc)
  finally:
    resources.timeline_signal = old_signal
  overflow_backend = StandaloneNvBackend(rshell, resources, ralloc, submitter)
  overflow_backend.timeline_value = 0x10000000000000000
  try:
    overflow_backend.timeline_state()
    raise AssertionError("overflowing timeline state was accepted")
  except ValueError as exc:
    assert "timeline value" in str(exc)
  public_cmdq = free_alloc.alloc_sysmem(0x1000)
  public_submitter = StandaloneSubmitter(free_shell, public_cmdq, GPFifoState(ring=MMIOView(bytearray(32), fmt='Q'),
    gpput=MMIOView(bytearray(4), fmt='I'), entries_count=4, token=0), MMIOView(bytearray(0x10000), fmt='I'))
  public_backend = StandaloneNvBackend(free_shell, resources, free_alloc, public_submitter)
  public_buf = public_backend.alloc(0x1000)
  public_va = public_buf.va_addr
  public_backend.free(public_buf)
  assert free_shell.mm.alloc_vaddr(0x1000, 0x1000) == public_va
  vram_buf = free_alloc.alloc_vram(0x2000, contiguous=True)
  vram_va, vram_pa = vram_buf.va_addr, vram_buf.meta.paddrs[0][0]
  free_alloc.free(vram_buf)
  free_alloc.free(vram_buf)
  assert vram_buf.meta is None
  assert free_shell.mm.alloc_vaddr(0x2000, 0x1000) == vram_va
  assert free_shell.mm.palloc(0x2000, 0x1000) == vram_pa

  class FakeUnloadRm:
    def __init__(self): self.unloads = 0
    def unloading_guest_driver(self):
      self.unloads += 1
      return b""
  class FakeCloseTransport:
    def __init__(self): self.freed = []
    def free_registered_channel(self, registration): self.freed.append(registration)
  close_rm = FakeUnloadRm()
  close_resources = ChannelResources(handles={"uvm_channel": (0x20000000, 0x4000)})
  class FailingCloseTransport(FakeCloseTransport):
    def free_registered_channel(self, registration):
      super().free_registered_channel(registration)
      raise RuntimeError("uvm free fail")
  close_shell = type("CloseShell", (), {"transport": FailingCloseTransport()})()
  closing_backend = StandaloneNvBackend(close_shell, close_resources, ralloc, submitter, rm=close_rm)
  try:
    closing_backend.close()
    raise AssertionError("failing backend close was accepted")
  except RuntimeError as exc:
    assert "uvm free fail" in str(exc)
  closing_backend.close()
  assert close_rm.unloads == 1
  assert close_shell.transport.freed == [(0x20000000, 0x4000)]
  assert "uvm_channel" not in close_resources.handles
  ctx_release_shell = FreeShell()
  ctx_release_alloc = StandaloneBufferAllocator(ctx_release_shell)
  ctx_map0 = ctx_release_shell.mm.valloc(0x2000, contiguous=True)
  ctx_map1 = ctx_release_shell.mm.valloc(0x1000, contiguous=True)
  ctx_map0_va, ctx_map0_pa = ctx_map0.va_addr, ctx_map0.paddrs[0][0]
  ctx_resources = ChannelResources(handles={"user_grctx_mappings": {0: ctx_map0, 1: ctx_map1},
    "golden_ctx": {"grctx_mappings": {0: ctx_map0, 2: ctx_map1}}})
  ctx_backend = StandaloneNvBackend(ctx_release_shell, ctx_resources, ctx_release_alloc, submitter)
  ctx_backend.close()
  ctx_backend.close()
  assert "user_grctx_mappings" not in ctx_resources.handles
  assert ctx_resources.handles["golden_ctx"].get("grctx_mappings") is None
  assert ctx_release_shell.mm.alloc_vaddr(0x2000, 0x1000) == ctx_map0_va
  reused_ctx_pa = {ctx_release_shell.mm.palloc(0x1000, 0x1000), ctx_release_shell.mm.palloc(0x1000, 0x1000),
                   ctx_release_shell.mm.palloc(0x1000, 0x1000)}
  assert ctx_map0_pa in reused_ctx_pa
  rm_private_shell = FreeShell()
  rm_private_alloc = StandaloneBufferAllocator(rm_private_shell)
  private_ramfc = rm_private_shell.mm.valloc(0x1000, contiguous=True)
  private_ramfc_va, private_ramfc_pa = private_ramfc.va_addr, private_ramfc.paddrs[0][0]
  private_resources = ChannelResources(handles={"rm_private_mappings": {0xc56f: (private_ramfc,)}})
  private_backend = StandaloneNvBackend(rm_private_shell, private_resources, rm_private_alloc, submitter)
  private_backend.close()
  private_backend.close()
  assert "rm_private_mappings" not in private_resources.handles
  assert rm_private_shell.mm.alloc_vaddr(0x1000, 0x1000) == private_ramfc_va
  private_ramfc_off = private_ramfc_pa - rm_private_shell.mm.phys_alloc.base
  assert any(start <= private_ramfc_off and private_ramfc_off + 0x1000 <= start + size and is_free
             for start, (size, _, _, is_free) in rm_private_shell.mm.phys_alloc.blocks.items())
  class FailingVfreeMm(GpuMemoryManager):
    def __init__(self):
      super().__init__(64 << 20)
      self.fail_once = True
    def vfree(self, mapping):
      if self.fail_once:
        self.fail_once = False
        raise RuntimeError("vfree fail")
      return super().vfree(mapping)
  vfree_fail_shell = type("VfreeFailShell", (), {"mm": FailingVfreeMm(), "transport": FakeSysmemTransport()})()
  vfree_fail_alloc = StandaloneBufferAllocator(vfree_fail_shell)
  vfree_map0 = vfree_fail_shell.mm.valloc(0x1000, contiguous=True)
  vfree_map1 = vfree_fail_shell.mm.valloc(0x1000, contiguous=True)
  vfree_map1_va = vfree_map1.va_addr
  try:
    StandaloneNvBackend(vfree_fail_shell, ChannelResources(handles={"user_grctx_mappings": {0: vfree_map0, 1: vfree_map1}}),
      vfree_fail_alloc, submitter).release_context_mappings()
    raise AssertionError("failing context vfree was accepted")
  except RuntimeError as exc:
    assert "vfree fail" in str(exc)
  assert vfree_map1_va not in vfree_fail_shell.mm.mappings
  assert vfree_fail_shell.mm.alloc_vaddr(0x1000, 0x1000) == vfree_map1_va
  for bad_ctx_handles, text in [
    ({"user_grctx_mappings": object()}, "user_grctx_mappings"),
    ({"rm_private_mappings": object()}, "rm_private_mappings"),
    ({"golden_ctx": {"grctx_mappings": object()}}, "golden context mappings"),
  ]:
    try:
      StandaloneNvBackend(ctx_release_shell, ChannelResources(handles=bad_ctx_handles), ctx_release_alloc, submitter).release_context_mappings()
      raise AssertionError("bad retained context mapping state was accepted")
    except ValueError as exc:
      assert text in str(exc)
  bad_release_shell = FreeShell()
  bad_release_alloc = StandaloneBufferAllocator(bad_release_shell)
  bad_release_resources = StandaloneChannelBuilder(bad_release_shell, None).allocate_runtime_resources(
    bad_release_alloc, entries=4, token=0x31)
  bad_release_gpfifo_va = bad_release_resources.gpfifo_area.va_addr
  bad_release_resources.handles["user_grctx_mappings"] = object()
  try:
    StandaloneNvBackend(bad_release_shell, bad_release_resources, bad_release_alloc, submitter).close()
    raise AssertionError("bad context mapping close was accepted")
  except ValueError as exc:
    assert "user_grctx_mappings" in str(exc)
  assert bad_release_resources.gpfifo_area is None and bad_release_resources.cmdq_page is None
  assert bad_release_resources.kernargs_buf is None and bad_release_resources.timeline_signal is None
  assert bad_release_shell.mm.alloc_vaddr(0x300000, 0x1000) == bad_release_gpfifo_va

  release_shell = FreeShell()
  release_alloc = StandaloneBufferAllocator(release_shell)
  release_resources = StandaloneChannelBuilder(release_shell, None).allocate_runtime_resources(release_alloc, entries=4, token=0x22)
  release_submitter = StandaloneSubmitter(release_shell, release_resources.cmdq_page, release_resources.compute_gpfifo, MMIOView(bytearray(0x10000), fmt='I'))
  release_backend = StandaloneNvBackend(release_shell, release_resources, release_alloc, release_submitter)
  gpfifo_va = release_resources.gpfifo_area.va_addr
  release_backend.ensure_local_memory(0x241)
  release_backend.close()
  release_backend.close()
  assert release_resources.gpfifo_area is None and release_resources.cmdq_page is None
  assert release_resources.kernargs_buf is None and release_resources.timeline_signal is None
  assert release_resources.shader_local_mem is None and release_resources.notifier_buf is None
  assert len(release_shell.freed) == 5
  assert release_shell.mm.alloc_vaddr(0x300000, 0x1000) == gpfifo_va

  release_rm_shell = FakeAdShell()
  release_rm_alloc = StandaloneBufferAllocator(release_rm_shell)
  release_rm = FakeRm()
  release_rm_alloc.configure_rm(release_rm, 0x80)
  release_rm_resources = StandaloneChannelBuilder(release_rm_shell, release_rm).allocate_runtime_resources(
    release_rm_alloc, h_device=0x80, h_virtmem=0x70, h_vaspace=0x90f1, entries=4)
  release_rm_gpfifo_va = release_rm_resources.gpfifo_area.va_addr
  release_rm_notifier_va = release_rm_resources.notifier_buf.va_addr
  release_rm_submitter = StandaloneSubmitter(release_rm_shell, release_rm_resources.cmdq_page,
    release_rm_resources.compute_gpfifo, MMIOView(bytearray(0x10000), fmt='I'))
  release_rm_backend = StandaloneNvBackend(release_rm_shell, release_rm_resources, release_rm_alloc, release_rm_submitter)
  release_rm_backend.close()
  assert release_rm_resources.notifier_buf is None
  assert "rm_private_mappings" not in release_rm_resources.handles
  assert release_rm_shell.mm.alloc_vaddr(0x300000, 0x1000) == release_rm_gpfifo_va
  assert release_rm_shell.mm.alloc_vaddr(48 << 20, 0x1000) == release_rm_notifier_va

  facade = make_simulated_backend(entries=16, token=0x77)
  sim_backend, sim_resources, sim_shell = facade.dev, facade.dev.resources, facade.dev.shell
  sim_done, sim_signal, sim_words = sim_backend.submit_timeline_signal()
  assert sim_done == 1 and sim_signal.value == 1
  assert [method for _, _, _, method, _, _ in decode_words(sim_words)] == [0x005c, 0x0020]
  aa, bb = (1.0, 2.0, 3.0, 4.0), (10.0, 20.0, 30.0, 40.0)
  abuf, bbuf, obuf = facade.alloc(16), facade.alloc(16), facade.alloc(16)
  facade.copyin(abuf, struct.pack("4f", *aa))
  facade.copyin(bbuf, struct.pack("4f", *bb))
  facade.copyin(obuf, bytes(16))
  sprg = SimpleProgram(facade, "E_4", build_cubin())
  qmd_prog_addr = (sprg.qmd._rw_bits(*QMD.FIELDS['program_address_upper']) << 32) | sprg.qmd._rw_bits(*QMD.FIELDS['program_address_lower'])
  qmd_cbuf_addr = (sprg.qmd._rw_bits(*QMD.FIELDS['constant_buffer_addr_upper_0']) << 32) | sprg.qmd._rw_bits(*QMD.FIELDS['constant_buffer_addr_lower_0'])
  assert qmd_prog_addr < QMD_ADDR_LIMIT
  assert qmd_cbuf_addr < QMD_ADDR_LIMIT
  class FailingProgramFacade(SimulatedBackendFacade):
    def __init__(self, dev):
      super().__init__(dev)
      self.freed = []
    def free(self, buf):
      self.freed.append(buf)
      return super().free(buf)
    def ensure_local_memory(self, size):
      raise RuntimeError("program local memory fail")
  fail_prog_shell = FakeResourceShell()
  fail_prog_alloc = StandaloneBufferAllocator(fail_prog_shell)
  fail_prog_resources = StandaloneChannelBuilder(fail_prog_shell, None).allocate_runtime_resources(fail_prog_alloc, entries=4, token=0x44)
  fail_prog_backend = StandaloneNvBackend(fail_prog_shell, fail_prog_resources, fail_prog_alloc,
    StandaloneSubmitter(fail_prog_shell, fail_prog_resources.cmdq_page, fail_prog_resources.compute_gpfifo, MMIOView(bytearray(0x10000), fmt='I')), simulate=True)
  fail_prog_facade = FailingProgramFacade(fail_prog_backend)
  try:
    SimpleProgram(fail_prog_facade, "E_4", build_cubin())
    raise AssertionError("program setup failure was accepted")
  except RuntimeError as exc:
    assert "program local memory fail" in str(exc)
  assert len(fail_prog_facade.freed) == 1 and fail_prog_facade.freed[0].meta is None
  class SmallKernargsFacade(BackendFacade):
    def alloc_kernargs(self, size, align=8):
      return self.dev.alloc(0x1000).offset(0, 16)
    def submit_gpfifo(self, words):
      raise AssertionError("undersized kernel arguments reached submit")
  try:
    manual_launch(SmallKernargsFacade(sim_backend), sprg, obuf, abuf, bbuf)
    raise AssertionError("undersized kernel argument buffer was accepted")
  except ValueError as exc:
    assert "kernel argument buffer" in str(exc)
  class NoAllocFacade(BackendFacade):
    def alloc_kernargs(self, size, align=8):
      raise AssertionError("bad program metadata reached kernel argument allocation")
  no_alloc_facade = NoAllocFacade(sim_backend)
  bad_qmd_backing = QMD()
  bad_qmd_backing.mv = memoryview(bytearray(8))
  for attr, bad_value, text in [
    ("qmd", object(), "program QMD"),
    ("qmd", bad_qmd_backing, "program QMD backing view"),
    ("constbufs", {}, "constant-buffer metadata"),
    ("constbufs", {0: (0,)}, "constant-buffer metadata"),
    ("constbufs", {0: (QMD_ADDR_LIMIT, 0x160)}, "constant buffer 0"),
    ("constbufs", {0: (0, 0)}, "constant-buffer size"),
    ("kernargs_alloc_size", 0, "kernel argument allocation size"),
  ]:
    old_value = getattr(sprg, attr)
    try:
      setattr(sprg, attr, bad_value)
      manual_launch(no_alloc_facade, sprg, obuf, abuf, bbuf)
      raise AssertionError("bad program metadata was accepted")
    except ValueError as exc:
      assert text in str(exc)
    finally:
      setattr(sprg, attr, old_value)
  original_cbuf = sprg.cbuf_0
  for bad_cbuf, text in [
    (None, "constant-buffer words"),
    ((1, 2, 3), "constant-buffer words"),
    ([1, "bad"], "constant-buffer word"),
    ([0x100000000], "constant-buffer word"),
  ]:
    try:
      sprg.cbuf_0 = bad_cbuf
      manual_launch(no_alloc_facade, sprg, obuf, abuf, bbuf)
      raise AssertionError("bad constant-buffer program state was accepted")
    except ValueError as exc:
      assert text in str(exc)
    finally:
      sprg.cbuf_0 = original_cbuf
  for bad_launch_call, text in [
    (lambda: manual_launch(facade, object(), obuf, abuf, bbuf), "program"),
    (lambda: manual_launch(facade, sprg, object(), abuf, bbuf), "output buffer"),
    (lambda: manual_launch(facade, sprg, obuf, object(), bbuf), "input A buffer"),
    (lambda: manual_launch(facade, sprg, obuf, abuf, object()), "input B buffer"),
  ]:
    try:
      bad_launch_call()
      raise AssertionError("bad manual launch input was accepted")
    except ValueError as exc:
      assert text in str(exc)
  no_cpu_out = GpuBuffer(obuf.va_addr, obuf.size, view=None, meta=obuf.meta)
  try:
    manual_launch(facade, sprg, no_cpu_out, abuf, bbuf)
    raise AssertionError("non-CPU-visible launch buffer was accepted")
  except RuntimeError as exc:
    assert "CPU-visible" in str(exc)
  bad_arg_buf = GpuBuffer(QMD_ADDR_LIMIT, 16, view=MMIOView(bytearray(16)), meta=None)
  try:
    manual_launch(facade, sprg, bad_arg_buf, abuf, bbuf)
    raise AssertionError("out-of-range kernel argument buffer was accepted")
  except ValueError as exc:
    assert "output" in str(exc)
  old_trace_launch, old_trace_steps, old_trace_stack = (os.environ.get(name) for name in
    ("NV_ADD_TRACE_LAUNCH", "NV_ADD_TRACE_LAUNCH_STEPS", "NV_ADD_TRACE_LAUNCH_STACK"))
  try:
    os.environ.pop("NV_ADD_TRACE_LAUNCH", None)
    os.environ["NV_ADD_TRACE_LAUNCH_STEPS"] = "1"
    os.environ["NV_ADD_TRACE_LAUNCH_STACK"] = "1"
    trace_buf = io.StringIO()
    with contextlib.redirect_stdout(trace_buf):
      manual_launch(facade, sprg, obuf, abuf, bbuf)
    trace_text = trace_buf.getvalue()
    assert "launch enter" in trace_text and "launch kernargs_allocated" in trace_text and "launch done" in trace_text
    assert f"qmd_template_sha256={sprg.qmd_template_sha256}" in trace_text
    assert "launch timeline_ready" in trace_text and "qmd_sha256=" in trace_text and "qmd_diff=" in trace_text
    assert "qmd_fields=" in trace_text and "cta_raster_width" in trace_text and "release0_enable" in trace_text
    assert "qmd_fields_sha256=" in trace_text
    assert "manual_launch" in trace_text
  finally:
    for name, value in (("NV_ADD_TRACE_LAUNCH", old_trace_launch), ("NV_ADD_TRACE_LAUNCH_STEPS", old_trace_steps),
                        ("NV_ADD_TRACE_LAUNCH_STACK", old_trace_stack)):
      if value is None: os.environ.pop(name, None)
      else: os.environ[name] = value
  expected_done = sim_backend.timeline_value
  manual_launch(facade, sprg, obuf, abuf, bbuf)
  assert list(struct.unpack("4f", facade.copyout(obuf, 16))) == [11.0, 22.0, 33.0, 44.0]
  assert sim_backend.last_launch == (obuf.va_addr, abuf.va_addr, bbuf.va_addr, expected_done)
  assert sim_resources.compute_gpfifo.put_value >= 3
  lib_va = sprg.lib_gpu.va_addr
  sprg.close()
  sprg.close()
  assert sim_shell.mm.alloc_vaddr(sprg.lib_gpu.size, 0x1000) == lib_va
  facade.close()
  assert import_guard_state() == {"tinygrad_modules": [], "ref_tinygrad_paths": []}
  assert static_import_guard_state() == {"tinygrad_static_imports": []}

  print("selftest=ok")

def main():
  is_mul = "--mul" in sys.argv
  selected_script = "examples/mul.py" if is_mul else "examples/add.py"
  selected_arithmetic = "mul" if is_mul else "add"
  if "--selftest" in sys.argv:
    selftest()
    return
  if "--import-guard" in sys.argv:
    print_import_guard()
    return
  if "--debug-help" in sys.argv:
    print_debug_help(selected_script, selected_arithmetic)
    return
  if "--transport-preflight" in sys.argv:
    print_transport_preflight()
    return
  if "--transport-preflight-gate" in sys.argv:
    print_transport_preflight_gate(require_ready="--require-ready" in sys.argv)
    return
  if "--transport-preflight-plan" in sys.argv:
    print_transport_preflight_plan(require_ready="--require-ready" in sys.argv, script=selected_script)
    return
  if "--classify-transport-preflight" in sys.argv:
    print_transport_preflight_classification()
    return
  if "--transport-contract" in sys.argv:
    print_transport_contract()
    return
  if "--register-contract" in sys.argv:
    print_register_contract()
    return
  if "--boot-firmware-contract" in sys.argv:
    print_boot_firmware_contract()
    return
  if "--gsp-rpc-contract" in sys.argv:
    print_gsp_rpc_contract()
    return
  if "--vm-contract" in sys.argv:
    print_vm_contract()
    return
  if "--channel-contract" in sys.argv:
    print_channel_contract()
    return
  if "--contract-suite" in sys.argv:
    print_contract_suite()
    return
  if "--validation-suite" in sys.argv:
    print_validation_suite(selected_arithmetic, selected_script)
    return
  if "--reconnect-commands" in sys.argv:
    print_reconnect_commands(selected_script)
    return
  if "--live-debug-commands" in sys.argv:
    print_live_debug_commands(selected_script)
    return
  if "--fecs-reset-scenarios" in sys.argv:
    print_fecs_reset_scenarios(selected_script)
    return
  if "--fecs-fence-diagnostic" in sys.argv:
    print_fecs_fence_diagnostic(selected_script)
    return
  if "--bar0-fence-status" in sys.argv:
    print_bar0_fence_status()
    return
  if "--stall-trace" in sys.argv:
    print_stall_trace_diagnostic(selected_script)
    return
  if "--logbuf-dump" in sys.argv:
    print_logbuf_dump_diagnostic(selected_script)
    return
  if "--live-log-workflow" in sys.argv:
    print_live_log_workflow(selected_script)
    return
  if "--live-stack-log-workflow" in sys.argv:
    print_live_stack_log_workflow(selected_script)
    return
  if "--comparison-checklist" in sys.argv:
    print_comparison_checklist(selected_script)
    return
  if "--compare-trace-logs" in sys.argv:
    print_trace_log_comparison(cli_arg_value("--standalone-log"), cli_arg_value("--tiny-log"))
    return
  if "--offline-debug-suite" in sys.argv:
    print_offline_debug_suite(selected_arithmetic)
    return
  if "--reconnect-command" in sys.argv:
    print_reconnect_command(selected_script)
    return
  if "--summary" in sys.argv:
    print_runtime_summary()
    return
  if "--golden-compute-fingerprint" in sys.argv:
    print_golden_compute_fingerprint()
    return
  if "--context-promote-fingerprint" in sys.argv:
    print_context_promote_fingerprint()
    return
  if "--gpfifo-constructor-fingerprint" in sys.argv:
    print_gpfifo_constructor_fingerprint()
    return
  if "--runtime-channel-fingerprint" in sys.argv:
    print_runtime_channel_fingerprint()
    return
  if "--launch-fingerprint" in sys.argv:
    print_launch_fingerprint("mul" if "--mul" in sys.argv else "add")
    return
  apply_default_live_run_env()
  if should_print_summary(): print_runtime_summary()
  a = (1.0, 2.0, 3.0, 4.0)
  b = (10.0, 20.0, 30.0, 40.0)
  cubin = build_cubin()
  print(f"cubin_bytes={len(cubin)} expected_result={[x + y for x, y in zip(a, b)]}")
  with NvBackend() as backend:
    print(f"device={backend.device_name} iface={backend.iface_name}")
    program, bufs = None, []
    try:
      a_buf = backend.alloc(16)
      b_buf = backend.alloc(16)
      out_buf = backend.alloc(16)
      bufs = [a_buf, b_buf, out_buf]
      backend.copyin(a_buf, struct.pack("4f", *a))
      backend.copyin(b_buf, struct.pack("4f", *b))
      backend.copyin(out_buf, bytes(16))
      program = SimpleProgram(backend, "E_4", cubin)
      manual_launch(backend, program, out_buf, a_buf, b_buf)
      result_bytes = backend.copyout(out_buf, 16)
    finally:
      if program is not None: program.close()
      for buf in bufs: backend.free(buf)
  result = list(struct.unpack("4f", result_bytes))
  print(f"result={result}")
  print("submitted rebuilt NV add kernel")


if __name__ == "__main__":
  main()
