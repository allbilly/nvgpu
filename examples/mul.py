#!/usr/bin/env python3
import pathlib, struct, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "tinygrad"))

from tinygrad.device import Device
from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.support.hcq import HWQueue


METHOD_NAMES = {int(getattr(nv_gpu, name)): name for name in dir(nv_gpu)
                if name[:7] in {"NVC9B0_", "NVC6C0_", "NVC56F_", "NVC6B5_"} and isinstance(getattr(nv_gpu, name), int)}
METHOD_NAMES[0x0020] = "NVC56F_NON_STALL_INTERRUPT"


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


def decode_words(words):
  index = 0
  while index < len(words):
    header = words[index]
    typ, size, subc, method = (header >> 28) & 0xf, (header >> 16) & 0xfff, (header >> 13) & 0x7, (header << 2) & 0x7fff
    args = words[index + 1:index + 1 + size]
    yield index, typ, subc, method, METHOD_NAMES.get(method, f"UNKNOWN_0x{method:x}"), args
    index += size + 1


SHSTRTAB = b"\0.shstrtab\0.strtab\0.symtab\0.symtab_shndx\0.nv.info\0.text.E_4\0.nv.info.E_4\0.nv.shared.E_4\0.nv.constant0.E_4\0.rel.nv.constant0.E_4\0.debug_frame\0.rel.debug_frame\0.rela.debug_frame\0.nv.callgraph\0.nv.prototype\0.nv.rel.action\0"
STRTAB = b"\0.shstrtab\0.strtab\0.symtab\0.symtab_shndx\0.nv.info\0.text.E_4\0.nv.info.E_4\0.nv.shared.E_4\0.rel.nv.constant0.E_4\0.nv.constant0.E_4\0.debug_frame\0.rel.debug_frame\0.rela.debug_frame\0.nv.callgraph\0.nv.prototype\0.nv.rel.action\0E_4\0"
SYMTAB_WORDS = (0, 0, 0, 0, 0, 0, 50, 720899, 0, 0, 0, 0, 110, 655363, 0, 0, 0, 0, 128, 262147, 0, 0, 0, 0, 176, 458755, 0, 0, 0, 0, 204, 524291, 0, 0, 0, 0, 219, 725010, 0, 0, 512, 0)
DEBUG_FRAME_WORDS = (4294967295, 36, 0, 4294967295, 4294967295, 2080636931, 4294967295, 2155940879, 134228096, 679510527, 2155905288, 40, 4294967295, 52, 0, 0, 0, 0, 0, 512, 0, 1028, 3933184, 2165047296, 2654336, 4294966276, 63, 0)
NV_INFO_WORDS = (536324, 6, 14, 528644, 6, 0, 528900, 6, 0)
NV_INFO_E4_WORDS = (276228, 128, 13569, 526852, 2, 1573216, 1579267, 792324, 0, 1048578, 2224128, 792324, 0, 524289, 2224128, 792324, 0, 0, 2224128, 16718595, 269316, 240, 787716, 1, 1, 1)
NV_CALLGRAPH_WORDS = (0, 4294967295, 0, 4294967294, 0, 4294967293, 0, 4294967292)
NV_REL_ACTION_WORDS = (115, 0, 285212672, 906297381)
REL_DEBUG_FRAME_WORDS = (68, 0, 2, 6)
SECTION_HEADERS = (
  (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
  (1, 3, 0, 0, 64, 219, 0, 0, 1, 0),
  (11, 3, 0, 0, 283, 223, 0, 0, 1, 0),
  (19, 2, 0, 0, 512, 168, 2, 6, 8, 24),
  (128, 1, 0, 0, 680, 112, 0, 0, 1, 0),
  (41, 1879048192, 0, 0, 792, 36, 3, 0, 4, 0),
  (60, 1879048192, 64, 0, 828, 104, 3, 11, 4, 0),
  (176, 1879048193, 0, 0, 932, 32, 3, 0, 4, 8),
  (204, 1879048203, 0, 0, 968, 16, 0, 0, 8, 8),
  (141, 9, 64, 0, 984, 16, 3, 4, 8, 16),
  (88, 1, 66, 0, 1000, 376, 0, 11, 4, 0),
  (50, 1, 6, 0, 1408, 512, 3, 234881030, 128, 0),
)


# SASS bundles (4 little-endian u32 words each) for the .text.E_4 section of the
# E_4(out, a, b) kernel. Each bundle is a single SASS instruction; the names below
# are the cuobjdump -sass form.
#
# Spec (SASS source, for reference):
SASS_SOURCE = """
.visible .entry E_4(.param .u64 out, .param .u64 a, .param .u64 b) {
  .reg .u64 %out, %a, %b;
  .reg .f32 %a0,%a1,%a2,%a3, %b0,%b1,%b2,%b3, %r0,%r1,%r2,%r3;
  ld.param.u64 %out, [out];
  ld.param.u64 %a,   [a];
  ld.param.u64 %b,   [b];
  ld.global.v4.f32 {%a0,%a1,%a2,%a3}, [%a];
  ld.global.v4.f32 {%b0,%b1,%b2,%b3}, [%b];
  mul.rn.f32 %r0, %a0, %b0;   // for add, replace mul.rn.f32 with add.rn.f32
  mul.rn.f32 %r1, %a1, %b1;
  mul.rn.f32 %r2, %a2, %b2;
  mul.rn.f32 %r3, %a3, %b3;
  st.global.v4.f32 [%out], {%r0,%r1,%r2,%r3};
  ret;
}
"""

# SM86 SASS classes. Mirrors the Register/UniformRegister/Opcode tables in
# denvdis/data11/sm86_1.txt and the assembler's CuAsm/InsAsmRepos sm_86
# default repos. Each value is the 8-bit register index (or 16-bit opcode
# field) as it appears in the SASS bundle.
#
# Register class: R0=0, R1=1, ..., RZ=255 (denvdis line 1865).
class Reg:
  RZ = 255
  R0 = 0; R1 = 1; R2 = 2; R3 = 3; R4 = 4; R5 = 5; R6 = 6; R7 = 7
  R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15

# Uniform Register class: UR0=0, UR1=1, ..., URZ=63.
class UReg:
  URZ = 63
  UR4 = 4  # only UR4 is used in our cubin

# Opcode class: low 16 bits of word0. The high 4 bits of these (0x7000)
# are a wait/yield opex field; the low 12 bits are the actual opcode.
# Each value is cross-checked against:
#   - cuobjdump -sass   (closed-source NVIDIA disassembler; mnemonic + bytes)
#   - cuasm sm_86       (open-source assembler, CuAsm/InsAsmRepos)
#   - denvdis data11    (open-source 128-bit SASS spec, denvdis/data11/sm86_1.txt)
# Note: cuobjdump calls Op.LDC "MOV" (alias for the same SASS bytes on
# SM86 — LDC and MOV are the same instruction, only the name differs).
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

# Bundle layout: each tuple is 4 little-endian u32 words forming one
# 128-bit SASS instruction. Per-family field positions (per denvdis
# sm86_1.txt):
#   LDC / LDC.64 (Op.LDC):  word0 byte 2 = Rd, byte 3 = *Ra (offset reg)
#   LDCU.64  (Op.LDCU64):   word0 byte 2 = URd
#   FADD/FMUL R_R_R:        word0 byte 3 = Ra, byte 2 = Rd, word1 byte 0 = Rc
#   LDG.E.128 (Op.LDG):     word0 byte 3 = Ra, byte 2 = Rd
#   STG.E    (Op.STG):      word0 byte 3 = Rd
#   EXIT/BRA/NOP:           no register field
#
# The PREFIX/LDC and ARITH/FMUL comment names below reflect the
# nvdisasm mnemonic and register list; the actual register byte values
# are produced by the Reg.Rn / UReg.URn constants on the right side of
# the shift expressions, so a name like "PREFIX_LDC_R1_C28" matches the
# actual encoding (Rd = Reg.R1 = 1 at byte 2).
PREFIX_LDC_R1_C28          = ((Reg.R1 << 16) | Op.LDC,    0x00000a00, 0x00000f00, 0x000fe400)  # MOV R1, c[0x0][0x28]
PREFIX_LDC_R4_C170         = ((Reg.R4 << 16) | Op.LDC,    0x00005c00, 0x00000f00, 0x000fe200)  # MOV R4, c[0x0][0x170]
PREFIX_LDCU64_UR4_C118     = ((UReg.UR4 << 16) | Op.LDCU64, 0x00004600, 0x00000a00, 0x000fe200)  # ULDC.64 UR4, c[0x0][0x118]
PREFIX_LDC_R5_C174         = ((Reg.R5 << 16) | Op.LDC,    0x00005d00, 0x00000f00, 0x000fe400)  # MOV R5, c[0x0][0x174]
PREFIX_LDC_R2_C168         = ((Reg.R2 << 16) | Op.LDC,    0x00005a00, 0x00000f00, 0x000fe400)  # MOV R2, c[0x0][0x168]
PREFIX_LDC_R3_C16C         = ((Reg.R3 << 16) | Op.LDC,    0x00005b00, 0x00000f00, 0x000fe400)  # MOV R3, c[0x0][0x16c]
PREFIX_LDG_R4_R4           = ((Reg.R4 << 24) | (Reg.R4 << 16) | Op.LDG, 0x00000004, 0x0c1e1d00, 0x000ea800)  # LDG.E.128 R4, [R4.64]
PREFIX_LDG_R8_R2           = ((Reg.R2 << 24) | (Reg.R8 << 16) | Op.LDG, 0x00000004, 0x0c1e1d00, 0x000ea400)  # LDG.E.128 R8, [R2.64]
ARITH_FMUL_R11_R11_R7      = ((Reg.R11 << 24) | (Reg.R11 << 16) | Op.FMUL, 0x00000007, 0x00400000, 0x004fe200)  # FMUL R11, R11, R7
ARITH_FMUL_R10_R10_R6      = ((Reg.R10 << 24) | (Reg.R10 << 16) | Op.FMUL, 0x00000006, 0x00400000, 0x000fe200)  # FMUL R10, R10, R6
ARITH_FMUL_R9_R9_R5        = ((Reg.R9  << 24) | (Reg.R9  << 16) | Op.FMUL, 0x00000005, 0x00400000, 0x000fe200)  # FMUL R9, R9, R5
ARITH_FMUL_R8_R8_R4        = ((Reg.R8  << 24) | (Reg.R8 << 16) | Op.FMUL, 0x00000004, 0x00400000, 0x000fe200)  # FMUL R8, R8, R4              (the mul)
SUFFIX_LDC_R6_C160         = ((Reg.R6 << 16) | Op.LDC,    0x00005800, 0x00000f00, 0x000fc400)  # MOV R6, c[0x0][0x160]
SUFFIX_LDC_R7_C164         = ((Reg.R7 << 16) | Op.LDC,    0x00005900, 0x00000f00, 0x000fca00)  # MOV R7, c[0x0][0x164]
SUFFIX_STG_R6              = ((Reg.R6 << 24) | Op.STG,    0x00000008, 0x0c101d04, 0x000fe200)  # STG.E.128 [R6.64], R8
SUFFIX_EXIT                = (Op.EXIT,                    0x00000000, 0x03800000, 0x000fea00)  # EXIT
SUFFIX_BRA                 = (Op.BRA,                     0xfffffff0, 0x0383ffff, 0x000fc000)  # BRA .L_1
NOP                        = (Op.NOP,                     0x00000000, 0x00000000, 0x000fc000)  # NOP

SASS_COMMON_PREFIX = (
  PREFIX_LDC_R1_C28,
  PREFIX_LDC_R4_C170,
  PREFIX_LDCU64_UR4_C118,
  PREFIX_LDC_R5_C174,
  PREFIX_LDC_R2_C168,
  PREFIX_LDC_R3_C16C,
  PREFIX_LDG_R4_R4,
  PREFIX_LDG_R8_R2,
)
SASS_COMMON_SUFFIX = (
  SUFFIX_LDC_R6_C160,
  SUFFIX_LDC_R7_C164,
  SUFFIX_STG_R6,
  SUFFIX_EXIT,
  SUFFIX_BRA,
)
SASS_ARITHMETIC = (ARITH_FMUL_R11_R11_R7, ARITH_FMUL_R10_R10_R6, ARITH_FMUL_R9_R9_R5, ARITH_FMUL_R8_R8_R4)  # mul: FMUL


def words_blob(words): return b"".join(struct.pack("<I", w) for w in (words if isinstance(words, (list, tuple)) else (words,)))


def build_text(arithmetic_words):
  bundles = [*SASS_COMMON_PREFIX, *arithmetic_words, *SASS_COMMON_SUFFIX, *[NOP] * 15]
  return b"".join(words_blob(bundle) for bundle in bundles)


def build_cubin(arithmetic_words):
  cubin = bytearray(2856)
  cubin[:64] = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01\x33\x07" + bytes(7), 2, 190, 128, 0, 2688, 1920, 5637462, 64, 56, 3, 64, 12, 1)
  text = build_text(arithmetic_words) if not isinstance(arithmetic_words, (bytes, bytearray)) else arithmetic_words
  sections = {
    64: SHSTRTAB, 283: STRTAB, 512: words_blob(SYMTAB_WORDS), 680: words_blob(DEBUG_FRAME_WORDS), 792: words_blob(NV_INFO_WORDS),
    828: words_blob(NV_INFO_E4_WORDS), 932: words_blob(NV_CALLGRAPH_WORDS), 968: words_blob(NV_REL_ACTION_WORDS),
    984: words_blob(REL_DEBUG_FRAME_WORDS), 1000: bytes(376), 1408: text,
  }
  for offset, data in sections.items(): cubin[offset:offset+len(data)] = data
  for index, header in enumerate(SECTION_HEADERS): cubin[1920 + index * 64:1920 + (index + 1) * 64] = struct.pack("<IIQQQQIIQQ", *header)
  phdrs = ((6, 5, 2688, 0, 0, 168, 168, 8), (1, 5, 1000, 0, 0, 920, 920, 8), (1, 5, 2688, 0, 0, 168, 168, 8))
  for index, header in enumerate(phdrs): cubin[2688 + index * 56:2688 + (index + 1) * 56] = struct.pack("<IIQQQQQQ", *header)
  return bytes(cubin)


build_cubin(SASS_ARITHMETIC)


def trace_submits():
  real_submit = HWQueue.submit
  count = 0
  def traced_submit(self, dev, var_vals=None):
    nonlocal count
    if var_vals is not None: self._apply_var_vals(var_vals)
    count += 1
    words = [int(word) for word in self._q]
    queue_name = type(self).__name__
    fifo = dev.compute_gpfifo if queue_name == "NVComputeQueue" else dev.dma_gpfifo
    before_put = fifo.put_value
    print(f"submit #{count}: {type(self).__name__} words={len(words)}")
    for index, typ, subc, method, name, args in decode_words(words):
      decoded = ", ".join(describe_args(method, args))
      print(f"  method[{index}] {name}: typ={typ} subc={subc} mthd=0x{method:x} args=[{decoded}]")
    ret = real_submit(self, dev, None)
    if fifo.put_value != before_put:
      entry = int(fifo.ring[before_put % fifo.entries_count])
      addr = ((entry & ((1 << 40) - 1)) >> 2) << 2
      packets = (entry >> 42) & ((1 << 20) - 1)
      print(f"  GPFIFO[{before_put % fifo.entries_count}]=0x{entry:016x} addr=0x{addr:x} packets={packets} token=0x{fifo.token:x}")
      print(f"  doorbell gpput={fifo.gpput[0]} put_value={fifo.put_value}")
    return ret
  HWQueue.submit = traced_submit
  return real_submit


def main():
  dev = Device["NV"]
  print(f"device={dev.device} iface={type(dev.iface).__name__}")
  a = dev.allocator.alloc(16)
  b = dev.allocator.alloc(16)
  out = dev.allocator.alloc(16)
  dev.allocator._copyin(a, memoryview(struct.pack("4f", 1.0, 2.0, 3.0, 4.0)))
  dev.allocator._copyin(b, memoryview(struct.pack("4f", 10.0, 20.0, 30.0, 40.0)))
  dev.allocator._copyin(out, memoryview(bytes(16)))
  program = dev.runtime("E_4", build_cubin(SASS_ARITHMETIC))
  real_submit = trace_submits()
  try:
    program(out, a, b, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=True)
  finally:
    HWQueue.submit = real_submit
  result_bytes = bytearray(16)
  dev.allocator._copyout(memoryview(result_bytes), out)
  print(f"result={list(struct.unpack('4f', result_bytes))}")
  print("submitted rebuilt NV mul kernel")


if __name__ == "__main__":
  main()
