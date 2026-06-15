#!/usr/bin/env python3
import argparse, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent
TINYGRAD = ROOT / "ref" / "tinygrad"
sys.path.insert(0, str(TINYGRAD))

from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVComputeQueue, NVCopyQueue


LOG = ROOT / "dump" / "nv_mul_dump.log"
SUBMIT_RE = re.compile(r"^SUBMIT #(\d+) queue=(\w+) words=(\d+) iface=(\w+)")
WORDS_RE = re.compile(r"^\s+words_hex=(.*)$")


def load_events():
  events, current = [], None
  for line in LOG.read_text().splitlines():
    if match := SUBMIT_RE.match(line):
      current = {"submit": int(match.group(1)), "queue": match.group(2), "word_count": int(match.group(3)), "iface": match.group(4), "words": []}
      events.append(current)
    elif current is not None and (match := WORDS_RE.match(line)):
      current["words"] = [int(word, 16) for word in match.group(1).split()]
  return events


def main():
  parser = argparse.ArgumentParser(description="Replay or inspect captured tinygrad NV mul command words from dump/nv_mul_dump.log.")
  parser.add_argument("--queue-submit", action="store_true", help="submit captured queue words through tinygrad NV queues")
  args = parser.parse_args()

  events = load_events()
  print(f"loaded {len(events)} submissions from {LOG}")
  for event in events:
    print(f"SUBMIT #{event['submit']} queue={event['queue']} words={len(event['words'])}")
    print("  words_hex=" + " ".join(f"{word:08x}" for word in event["words"]))

  if args.queue_submit:
    dev = Device["NV"]
    for event in events:
      queue = NVComputeQueue() if event["queue"] == "NVComputeQueue" else NVCopyQueue()
      queue._q = event["words"]
      queue.submit(dev)
    dev.synchronize()


if __name__ == "__main__":
  main()
