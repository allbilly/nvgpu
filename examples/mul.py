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
  (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 3, 0, 0, 64, 219, 0, 0, 1, 0), (11, 3, 0, 0, 283, 223, 0, 0, 1, 0),
  (19, 2, 0, 0, 512, 168, 2, 6, 8, 24), (128, 1, 0, 0, 680, 112, 0, 0, 1, 0), (41, 1879048192, 0, 0, 792, 36, 3, 0, 4, 0),
  (60, 1879048192, 64, 0, 828, 104, 3, 11, 4, 0), (176, 1879048193, 0, 0, 932, 32, 3, 0, 4, 8),
  (204, 1879048203, 0, 0, 968, 16, 0, 0, 8, 8), (141, 9, 64, 0, 984, 16, 3, 4, 8, 16),
  (88, 1, 66, 0, 1000, 376, 0, 11, 4, 0), (50, 1, 6, 0, 1408, 512, 3, 234881030, 128, 0),
)
SASS_COMMON_PREFIX = ((0x00017a02, 0x00000a00, 0x00000f00, 0x000fe400), (0x00047a02, 0x00005c00, 0x00000f00, 0x000fe200), (0x00047ab9, 0x00004600, 0x00000a00, 0x000fe200), (0x00057a02, 0x00005d00, 0x00000f00, 0x000fe400), (0x00027a02, 0x00005a00, 0x00000f00, 0x000fe400), (0x00037a02, 0x00005b00, 0x00000f00, 0x000fe400), (0x04047981, 0x00000004, 0x0c1e1d00, 0x000ea800), (0x02087981, 0x00000004, 0x0c1e1d00, 0x000ea400))
SASS_MUL = ((0x0b0b7220, 0x00000007, 0x00400000, 0x004fe200), (0x0a0a7220, 0x00000006, 0x00400000, 0x000fe200), (0x09097220, 0x00000005, 0x00400000, 0x000fe200), (0x08087220, 0x00000004, 0x00400000, 0x000fe200))
SASS_COMMON_SUFFIX = ((0x00067a02, 0x00005800, 0x00000f00, 0x000fc400), (0x00077a02, 0x00005900, 0x00000f00, 0x000fca00), (0x06007986, 0x00000008, 0x0c101d04, 0x000fe200), (0x0000794d, 0x00000000, 0x03800000, 0x000fea00), (0x00007947, 0xfffffff0, 0x0383ffff, 0x000fc000))
SASS_NOP = (0x00007918, 0x00000000, 0x00000000, 0x000fc000)


def words_blob(words): return struct.pack(f"<{len(words)}I", *words)
def build_text(arithmetic_words): return b"".join(words_blob(bundle) for bundle in [*SASS_COMMON_PREFIX, *arithmetic_words, *SASS_COMMON_SUFFIX, *([SASS_NOP] * 15)])


def build_cubin(arithmetic_words):
  cubin = bytearray(2856)
  cubin[:64] = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01\x33\x07" + bytes(7), 2, 190, 128, 0, 2688, 1920, 5637462, 64, 56, 3, 64, 12, 1)
  sections = {64: SHSTRTAB, 283: STRTAB, 512: words_blob(SYMTAB_WORDS), 680: words_blob(DEBUG_FRAME_WORDS), 792: words_blob(NV_INFO_WORDS), 828: words_blob(NV_INFO_E4_WORDS), 932: words_blob(NV_CALLGRAPH_WORDS), 968: words_blob(NV_REL_ACTION_WORDS), 984: words_blob(REL_DEBUG_FRAME_WORDS), 1000: bytes(376), 1408: build_text(arithmetic_words)}
  for offset, data in sections.items(): cubin[offset:offset+len(data)] = data
  for index, header in enumerate(SECTION_HEADERS): cubin[1920 + index * 64:1920 + (index + 1) * 64] = struct.pack("<IIQQQQIIQQ", *header)
  phdrs = ((6, 5, 2688, 0, 0, 168, 168, 8), (1, 5, 1000, 0, 0, 920, 920, 8), (1, 5, 2688, 0, 0, 168, 168, 8))
  for index, header in enumerate(phdrs): cubin[2688 + index * 56:2688 + (index + 1) * 56] = struct.pack("<IIQQQQQQ", *header)
  return bytes(cubin)


KERNEL_CUBIN = build_cubin(SASS_MUL)


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
  program = dev.runtime("E_4", KERNEL_CUBIN)
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
