#!/usr/bin/env python3
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
TINYGRAD = ROOT / "ref" / "tinygrad"
sys.path.insert(0, str(TINYGRAD))

from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVComputeQueue, NVCopyQueue


CAPTURED_SUBMITS = [
  ("NVComputeQueue", "20012000 0000c7c0 200221ec 00007293 00000000 200220a8 00007294 00000000 20050017 202bbff0 00000010 00000001 00000000 03100001 20010008 00000000"),
  ("NVCopyQueue", "20050017 202bbff0 00000010 00000001 00000000 01000003 20018000 0000c7b5 20038090 00000010 202bbff0 00000002 200180c0 00000014"),
  ("NVCopyQueue", "20050017 202bbff0 00000010 00000002 00000000 01000003 20048100 00000010 21c00000 00000010 2000e000 20018106 00000010 200180c0 00000182 20038090 00000010 202bbff0 00000003 200180c0 00000014"),
  ("NVCopyQueue", "20050017 202bbff0 00000010 00000003 00000000 01000003 20048100 00000010 25000000 00000010 20015000 20018106 00000010 200180c0 00000182 20038090 00000010 202bbff0 00000004 200180c0 00000014"),
  ("NVComputeQueue", "20050017 202bbff0 00000010 00000004 00000000 01000003 200221e4 00000010 30000000 200320b9 00000000 001b0000 000000ff 20050017 202bbff0 00000010 00000005 00000000 03100001 20010008 00000000"),
  ("NVCopyQueue", "20050017 202bbff0 00000010 00000005 00000000 01000003 20048100 00000010 25200000 00000010 2029a000 20018106 00000400 200180c0 00000182 20038090 00000010 202bbff0 00000006 200180c0 00000014"),
  ("NVComputeQueue", "20050017 202bbff0 00000010 00000006 00000000 01000003 200125a6 00001011 200120ad 102c0002 200120b0 00000009"),
  ("NVCopyQueue", "20050017 202bbff0 00000010 00000007 00000000 01000003 20048100 00000010 20016000 00000010 21a00000 20018106 00000010 200180c0 00000182 20038090 00000010 202bbff0 00000008 200180c0 00000014"),
]


def words(hex_words):
  return [int(word, 16) for word in hex_words.split()]


def submit_words(dev, queue_name, queue_words):
  queue = NVComputeQueue() if queue_name == "NVComputeQueue" else NVCopyQueue()
  queue._q = queue_words
  queue.submit(dev)


def main():
  dev = Device["NV"]
  print(f"device={dev.device} iface={type(dev.iface).__name__}")
  for index, (queue_name, hex_words) in enumerate(CAPTURED_SUBMITS, 1):
    queue_words = words(hex_words)
    print(f"submit #{index}: {queue_name} words={len(queue_words)}")
    submit_words(dev, queue_name, queue_words)
  dev.synchronize()
  print("submitted captured NV add command stream")


if __name__ == "__main__":
  main()
