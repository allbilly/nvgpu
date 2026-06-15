#!/usr/bin/env python3
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
TINYGRAD = ROOT / "ref" / "tinygrad"
sys.path.insert(0, str(TINYGRAD))

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.support.hcq import HWQueue


LOG = ROOT / "dump" / "nv_mul_dump.log"
nvqcmds = {int(getattr(nv_gpu, name)): name for name in dir(nv_gpu)
           if name[:7] in {"NVC9B0_", "NVC6C0_", "NVC56F_", "NVC6B5_"} and isinstance(getattr(nv_gpu, name), int)}
real_submit = HWQueue.submit
submit_count = 0


def decode(words):
  packets, index = [], 0
  while index < len(words):
    header = int(words[index])
    typ, size, subc, mthd = (header >> 28) & 0xF, (header >> 16) & 0xFFF, (header >> 13) & 7, (header << 2) & 0x7FFF
    if typ == 0: break
    args = [int(words[index + 1 + arg_index]) for arg_index in range(size)]
    packets.append((index, typ, size, subc, mthd, nvqcmds.get(mthd, f"UNKNOWN_0x{mthd:x}"), args))
    index += size + 1
  return packets


def traced_submit(self, dev, var_vals=None):
  global submit_count
  submit_count += 1
  if var_vals is not None:
    self._apply_var_vals(var_vals)
    var_vals = None

  words = [int(word) for word in self._q]
  queue_name = type(self).__name__
  fifo = dev.compute_gpfifo if queue_name == "NVComputeQueue" else dev.dma_gpfifo
  before_put = fifo.put_value

  with LOG.open("a") as log:
    log.write(f"\nSUBMIT #{submit_count} queue={queue_name} words={len(words)} iface={type(dev.iface).__name__}\n")
    log.write("  words_hex=" + " ".join(f"{word:08x}" for word in words) + "\n")
    for index, typ, size, subc, mthd, name, args in decode(words):
      log.write(f"  method[{index}] {name} typ={typ} subc={subc} mthd=0x{mthd:x} size={size}\n")
      for arg_index, arg in enumerate(args): log.write(f"    arg{arg_index}=0x{arg:08x} ({arg})\n")

  ret = real_submit(self, dev, var_vals)
  if fifo.put_value != before_put:
    entry = int(fifo.ring[before_put % fifo.entries_count])
    addr = ((entry & ((1 << 40) - 1)) >> 2) << 2
    packets = (entry >> 42) & ((1 << 20) - 1)
    with LOG.open("a") as log:
      log.write(f"  GPFIFO[{before_put % fifo.entries_count}]=0x{entry:016x} addr=0x{addr:x} packets={packets} token=0x{fifo.token:x}\n")
      log.write(f"  doorbell gpput={fifo.gpput[0]} put_value={fifo.put_value}\n")
  return ret


def main():
  LOG.parent.mkdir(exist_ok=True)
  LOG.write_text("")
  HWQueue.submit = traced_submit
  dev = Device["NV"]
  with LOG.open("a") as log: log.write(f"device {dev.device} iface {type(dev.iface).__name__}\n")
  a = Tensor([1.0, 2.0, 3.0, 4.0], device="NV").realize()
  b = Tensor([10.0, 20.0, 30.0, 40.0], device="NV").realize()
  c = (a * b).realize()
  dev.synchronize()
  result = c.tolist()
  with LOG.open("a") as log: log.write(f"\nRESULT {result}\n")
  print(result)
  print(f"wrote {LOG}")


if __name__ == "__main__":
  main()
