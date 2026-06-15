#!/usr/bin/env python3
from dataclasses import dataclass
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
TINYGRAD = ROOT / "ref" / "tinygrad"
sys.path.insert(0, str(TINYGRAD))

from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVComputeQueue, NVCopyQueue


@dataclass(frozen=True)
class Arg:
  name: str
  value: int
  meaning: str


def arg(name, value, meaning): return Arg(name, value, meaning)
def cls(name, value): return arg(name, value, "NVIDIA engine object class id")
def method_flag(name, value, meaning): return arg(name, value, meaning)
def word(name, value, meaning): return arg(name, value, meaning)
def va_lo(name, value): return arg(name, value, "low 32 bits of captured GPU virtual address")
def va_hi(name, value): return arg(name, value, "high 32 bits of captured GPU virtual address")
def timeline(value): return arg(f"timeline_{value}", value, "timeline semaphore payload")
def zero(name="zero"): return arg(name, 0, "reserved/zero")


TYPE_INCREASING = 2
SUBC_M2MF = 4
SUBC_COMPUTE = 1
SUBC_HOST = 0

MTHD_SET_OBJECT = 0x0000
MTHD_NON_STALL_INTERRUPT = 0x0020
MTHD_SEM_ADDR_LO = 0x005c
MTHD_COPY_SEMAPHORE_A = 0x0240
MTHD_COPY_LAUNCH_DMA = 0x0300
MTHD_COPY_OFFSET_IN_UPPER = 0x0400
MTHD_COPY_LINE_LENGTH_IN = 0x0418
MTHD_COMPUTE_SET_SHARED_MEMORY_WINDOW_A = 0x02a0
MTHD_COMPUTE_SEND_PCAS_A = 0x02b4
MTHD_COMPUTE_SEND_SIGNALING_PCAS2_B = 0x02c0
MTHD_COMPUTE_SET_LOCAL_MEMORY_NON_THROTTLED_A = 0x02e4
MTHD_COMPUTE_SET_LOCAL_MEMORY_A = 0x0790
MTHD_COMPUTE_SET_LOCAL_MEMORY_WINDOW_A = 0x07b0
MTHD_COMPUTE_INVALIDATE_SHADER_CACHES_NO_WFI = 0x1698

COMPUTE_CLASS = cls("AMPERE_COMPUTE_B", 0xc7c0)
COPY_CLASS = cls("AMPERE_DMA_COPY_B", 0xc7b5)
TIMELINE_VA_LO = va_lo("timeline_signal_addr_lo", 0x202bbff0)
TIMELINE_VA_HI = va_hi("timeline_signal_addr_hi", 0x00000010)
WAIT_GE_64 = method_flag("sem_wait_ge_64", 0x01000003, "wait until 64-bit semaphore >= payload")
RELEASE_64_TIMESTAMP = method_flag("sem_release_64_timestamp", 0x03100001, "release 64-bit semaphore and timestamp")
COPY_PITCH = method_flag("copy_pitch_non_pipelined", 0x00000182, "DMA pitch copy launch flags")
COPY_RELEASE_SEMAPHORE = method_flag("copy_release_four_word_semaphore", 0x00000014, "DMA semaphore release flags")
CACHE_INVALIDATE_FLAGS = method_flag("invalidate_instruction_global_constant", 0x00001011, "invalidate shader instruction/global/constant caches")
PCAS2_LAUNCH_FLAGS = method_flag("pcas2_launch", 0x00000009, "launch QMD through PCAS2")

ADDRS = {
  "a_cpu_src": (0x21c00000, 0x00000010),
  "a_gpu_dst": (0x2000e000, 0x00000010),
  "b_cpu_src": (0x25000000, 0x00000010),
  "b_gpu_dst": (0x20015000, 0x00000010),
  "kernel_src": (0x25200000, 0x00000010),
  "kernel_dst": (0x2029a000, 0x00000010),
  "result_gpu_src": (0x20016000, 0x00000010),
  "result_cpu_dst": (0x21a00000, 0x00000010),
}


def copy_addr_args(src_name, dst_name):
  src_lo, src_hi = ADDRS[src_name]
  dst_lo, dst_hi = ADDRS[dst_name]
  return [va_hi(f"{src_name}_hi", src_hi), va_lo(f"{src_name}_lo", src_lo), va_hi(f"{dst_name}_hi", dst_hi), va_lo(f"{dst_name}_lo", dst_lo)]


def host_sem_wait(value): return [TIMELINE_VA_LO, TIMELINE_VA_HI, timeline(value), zero("payload_hi"), WAIT_GE_64]
def host_sem_release(value): return [TIMELINE_VA_LO, TIMELINE_VA_HI, timeline(value), zero("payload_hi"), RELEASE_64_TIMESTAMP]
def copy_sem_release(value): return [TIMELINE_VA_HI, TIMELINE_VA_LO, timeline(value)]
def line_length(bytes_count): return [word("line_length_bytes", bytes_count, "copy byte count")]


CAPTURED_SUBMITS = [
  ("NVComputeQueue", [
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_SET_OBJECT, "SET_OBJECT_COMPUTE", [COMPUTE_CLASS]),
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_COMPUTE_SET_LOCAL_MEMORY_WINDOW_A, "SET_SHADER_LOCAL_MEMORY_WINDOW_A", [va_hi("local_mem_window_hi", 0x7293), va_lo("local_mem_window_lo", 0)]),
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_COMPUTE_SET_SHARED_MEMORY_WINDOW_A, "SET_SHADER_SHARED_MEMORY_WINDOW_A", [va_hi("shared_mem_window_hi", 0x7294), va_lo("shared_mem_window_lo", 0)]),
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO release timeline=1", host_sem_release(1)),
    (TYPE_INCREASING, SUBC_HOST, MTHD_NON_STALL_INTERRUPT, "NON_STALL_INTERRUPT", [zero()]),
  ]),
  ("NVCopyQueue", [
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO wait timeline=1", host_sem_wait(1)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_SET_OBJECT, "SET_OBJECT_COPY", [COPY_CLASS]),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_SEMAPHORE_A, "SEMAPHORE_A release timeline=2", copy_sem_release(2)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA semaphore release", [COPY_RELEASE_SEMAPHORE]),
  ]),
  ("NVCopyQueue", [
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO wait timeline=2", host_sem_wait(2)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_OFFSET_IN_UPPER, "OFFSET_IN/OUT copy a CPU->GPU", copy_addr_args("a_cpu_src", "a_gpu_dst")),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LINE_LENGTH_IN, "LINE_LENGTH_IN bytes=16", line_length(16)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA pitch copy", [COPY_PITCH]),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_SEMAPHORE_A, "SEMAPHORE_A release timeline=3", copy_sem_release(3)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA semaphore release", [COPY_RELEASE_SEMAPHORE]),
  ]),
  ("NVCopyQueue", [
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO wait timeline=3", host_sem_wait(3)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_OFFSET_IN_UPPER, "OFFSET_IN/OUT copy b CPU->GPU", copy_addr_args("b_cpu_src", "b_gpu_dst")),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LINE_LENGTH_IN, "LINE_LENGTH_IN bytes=16", line_length(16)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA pitch copy", [COPY_PITCH]),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_SEMAPHORE_A, "SEMAPHORE_A release timeline=4", copy_sem_release(4)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA semaphore release", [COPY_RELEASE_SEMAPHORE]),
  ]),
  ("NVComputeQueue", [
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO wait timeline=4", host_sem_wait(4)),
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_COMPUTE_SET_LOCAL_MEMORY_A, "SET_SHADER_LOCAL_MEMORY_A", [va_hi("shader_local_mem_hi", 0x10), va_lo("shader_local_mem_lo", 0x30000000)]),
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_COMPUTE_SET_LOCAL_MEMORY_NON_THROTTLED_A, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A", [va_hi("local_mem_tpc_hi", 0), va_lo("local_mem_tpc_bytes", 0x001b0000), word("max_tpc", 0xff, "TPC mask/count value")]),
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO release timeline=5", host_sem_release(5)),
    (TYPE_INCREASING, SUBC_HOST, MTHD_NON_STALL_INTERRUPT, "NON_STALL_INTERRUPT", [zero()]),
  ]),
  ("NVCopyQueue", [
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO wait timeline=5", host_sem_wait(5)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_OFFSET_IN_UPPER, "OFFSET_IN/OUT copy kernel args/code", copy_addr_args("kernel_src", "kernel_dst")),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LINE_LENGTH_IN, "LINE_LENGTH_IN bytes=1024", line_length(1024)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA pitch copy", [COPY_PITCH]),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_SEMAPHORE_A, "SEMAPHORE_A release timeline=6", copy_sem_release(6)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA semaphore release", [COPY_RELEASE_SEMAPHORE]),
  ]),
  ("NVComputeQueue", [
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO wait timeline=6", host_sem_wait(6)),
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_COMPUTE_INVALIDATE_SHADER_CACHES_NO_WFI, "INVALIDATE_SHADER_CACHES_NO_WFI", [CACHE_INVALIDATE_FLAGS]),
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_COMPUTE_SEND_PCAS_A, "SEND_PCAS_A qmd_addr_hi", [word("qmd_addr_shifted_8", 0x102c0002, "QMD GPU virtual address >> 8")]),
    (TYPE_INCREASING, SUBC_COMPUTE, MTHD_COMPUTE_SEND_SIGNALING_PCAS2_B, "SEND_SIGNALING_PCAS2_B launch", [PCAS2_LAUNCH_FLAGS]),
  ]),
  ("NVCopyQueue", [
    (TYPE_INCREASING, SUBC_HOST, MTHD_SEM_ADDR_LO, "SEM_ADDR_LO wait timeline=7", host_sem_wait(7)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_OFFSET_IN_UPPER, "OFFSET_IN/OUT copy result GPU->CPU", copy_addr_args("result_gpu_src", "result_cpu_dst")),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LINE_LENGTH_IN, "LINE_LENGTH_IN bytes=16", line_length(16)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA pitch copy", [COPY_PITCH]),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_SEMAPHORE_A, "SEMAPHORE_A release timeline=8", copy_sem_release(8)),
    (TYPE_INCREASING, SUBC_M2MF, MTHD_COPY_LAUNCH_DMA, "LAUNCH_DMA semaphore release", [COPY_RELEASE_SEMAPHORE]),
  ]),
]


def packet_header(typ, subchannel, method, arg_count):
  return (typ << 28) | (arg_count << 16) | (subchannel << 13) | (method >> 2)


def packet_words(packet):
  typ, subchannel, method, _name, args = packet
  return [packet_header(typ, subchannel, method, len(args)), *[a.value for a in args]]


def submit_words_from_packets(packets):
  return [word for packet in packets for word in packet_words(packet)]


def submit_words(dev, queue_name, queue_words):
  queue = NVComputeQueue() if queue_name == "NVComputeQueue" else NVCopyQueue()
  queue._q = queue_words
  queue.submit(dev)


def main():
  dev = Device["NV"]
  print(f"device={dev.device} iface={type(dev.iface).__name__}")
  for index, (queue_name, packets) in enumerate(CAPTURED_SUBMITS, 1):
    queue_words = submit_words_from_packets(packets)
    print(f"submit #{index}: {queue_name} words={len(queue_words)}")
    for typ, subchannel, method, name, args in packets:
      decoded = ", ".join(f"{a.name}=0x{a.value:x} ({a.meaning})" for a in args)
      print(f"  {name}: typ={typ} subc={subchannel} mthd=0x{method:x} args=[{decoded}]")
    submit_words(dev, queue_name, queue_words)
  dev.synchronize()
  print("submitted captured NV add command stream")


if __name__ == "__main__":
  main()
