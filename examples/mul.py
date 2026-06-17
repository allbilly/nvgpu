#!/usr/bin/env python3
import array, pathlib, struct, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nv_pcie import open_pcie_device

METHOD_NAMES = {
  0x005c: "NVC56F_SEM_ADDR_LO",
  0x02b4: "NVC6C0_SEND_PCAS_A",
  0x02c0: "NVC6C0_SEND_SIGNALING_PCAS2_B",
  0x1698: "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
  0x0020: "NVC56F_NON_STALL_INTERRUPT",
}

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

def round_up(x, y): return ((x + y - 1) // y) * y

def write_words(dst, offset, words):
  dst[offset:offset + len(words)] = array.array('I', words)

def submit_gpfifo(dev, words):
  cmdq_addr = dev.cmdq_allocator.alloc(len(words) * 4, 16)
  cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
  write_words(dev.cmdq, cmdq_wptr, words)

  fifo = dev.compute_gpfifo
  ring_index = fifo.put_value % fifo.entries_count
  fifo.ring[ring_index] = (cmdq_addr // 4 << 2) | (len(words) << 42) | (1 << 41)
  fifo.gpput[0] = (fifo.put_value + 1) % fifo.entries_count
  dev.gpu_mmio[0x90 // 4] = fifo.token
  fifo.put_value += 1

  entry = int(fifo.ring[ring_index])
  addr = ((entry & ((1 << 40) - 1)) >> 2) << 2
  packets = (entry >> 42) & ((1 << 20) - 1)
  print(f"  GPFIFO[{ring_index}]=0x{entry:016x} addr=0x{addr:x} packets={packets} token=0x{fifo.token:x}")
  print(f"  doorbell gpput={fifo.gpput[0]} put_value={fifo.put_value}")

def wait_signal(signal, value, timeout_ms=30000):
  start = time.perf_counter()
  while signal.value < value:
    if (time.perf_counter() - start) * 1000 > timeout_ms:
      raise RuntimeError(f"timeout waiting for timeline {value}, got {signal.value}")
    time.sleep(0.001)

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

def build_cubin(): # nvdisasm mul.cubin
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
    ((ch.Reg.R11 << 24) | (ch.Reg.R11 << 16) | ch.Op.FMUL, 0x00000007, 0x00400000, 0x004fe200),  # FMUL R11, R11, R7
    ((ch.Reg.R10 << 24) | (ch.Reg.R10 << 16) | ch.Op.FMUL, 0x00000006, 0x00400000, 0x000fe200),  # FMUL R10, R10, R6
    ((ch.Reg.R9  << 24) | (ch.Reg.R9  << 16) | ch.Op.FMUL, 0x00000005, 0x00400000, 0x000fe200),  # FMUL R9, R9, R5
    ((ch.Reg.R8  << 24) | (ch.Reg.R8 << 16) | ch.Op.FMUL, 0x00000004, 0x00400000, 0x000fe200),    # FMUL R8, R8, R4

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

def main():
  a = (1.0, 2.0, 3.0, 4.0)
  b = (10.0, 20.0, 30.0, 40.0)
  cubin = build_cubin()
  print(f"cubin_bytes={len(cubin)} expected_result={[x * y for x, y in zip(a, b)]}")
  dev = open_pcie_device()
  print(f"device={dev.device} iface={type(dev.iface).__name__}")
  a_buf = dev.allocator.alloc(16)
  b_buf = dev.allocator.alloc(16)
  out_buf = dev.allocator.alloc(16)
  dev.allocator._copyin(a_buf, memoryview(struct.pack("4f", *a)))
  dev.allocator._copyin(b_buf, memoryview(struct.pack("4f", *b)))
  dev.allocator._copyin(out_buf, memoryview(bytes(16)))
  program = dev.runtime("E_4", cubin)
  manual_launch(dev, program, out_buf, a_buf, b_buf)
  result_bytes = bytearray(16)
  dev.allocator._copyout(memoryview(result_bytes), out_buf)
  result = list(struct.unpack("4f", result_bytes))
  print(f"result={result}")
  print("submitted rebuilt NV mul kernel")


if __name__ == "__main__":
  main()
