"""Static census of the pinned legacy ROM (IN/OUT/INT/far/PCI/phys)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .constants import ENTRY_OFFSET, ENTRY_PATH_TARGET, ENTRY_PATH_VIA
from .rom import RomImage, load_rom


@dataclass
class OpcodeHit:
  offset: int
  opcode: str
  mnemonic: str
  detail: str = ""


@dataclass
class CensusReport:
  image_sha256: str
  size: int
  entry_offset: int
  entry_path: List[int]
  io_ins: List[OpcodeHit] = field(default_factory=list)
  io_outs: List[OpcodeHit] = field(default_factory=list)
  interrupts: List[OpcodeHit] = field(default_factory=list)
  far_calls: List[OpcodeHit] = field(default_factory=list)
  far_jmps: List[OpcodeHit] = field(default_factory=list)
  pci_mech: List[OpcodeHit] = field(default_factory=list)
  phys_mem: List[OpcodeHit] = field(default_factory=list)
  opcode_histogram: Dict[str, int] = field(default_factory=dict)
  unknown_ledger: List[str] = field(default_factory=list)
  llvm_available: bool = False
  llvm_disasm_lines: int = 0
  vga_preamble_offsets: List[int] = field(default_factory=list)

  def to_dict(self) -> dict:
    d = asdict(self)
    return d


def verify_entry_path(image: bytes) -> List[int]:
  """Verify EB at 3 → 0x50 and E9 at 0x50 → 0x2caa."""
  path = [ENTRY_OFFSET]
  if image[ENTRY_OFFSET] != 0xEB:
    raise ValueError(f"entry@{ENTRY_OFFSET:#x}: expected EB, got {image[ENTRY_OFFSET]:02x}")
  rel = struct_s8(image[ENTRY_OFFSET + 1])
  land = (ENTRY_OFFSET + 2 + rel) & 0xFFFF
  if land != ENTRY_PATH_VIA:
    raise ValueError(f"entry short jmp lands at {land:#x}, expected {ENTRY_PATH_VIA:#x}")
  path.append(land)
  if image[land] != 0xE9:
    raise ValueError(f"@{land:#x}: expected E9 near jmp")
  rel16 = int.from_bytes(image[land + 1:land + 3], "little", signed=True)
  target = (land + 3 + rel16) & 0xFFFF
  if target != ENTRY_PATH_TARGET:
    raise ValueError(f"near jmp lands at {target:#x}, expected {ENTRY_PATH_TARGET:#x}")
  path.append(target)
  return path


def struct_s8(b: int) -> int:
  return b - 0x100 if b & 0x80 else b


def static_scan(image: bytes) -> CensusReport:
  """Linear scan for statically visible I/O, INT, far control, CF8/CFC immediates."""
  rom = None
  try:
    # caller may pass raw legacy bytes
    pass
  except Exception:
    pass
  report = CensusReport(
    image_sha256="",
    size=len(image),
    entry_offset=ENTRY_OFFSET,
    entry_path=verify_entry_path(image),
  )

  i = 0
  hist: Counter = Counter()
  while i < len(image):
    b = image[i]
    hist[f"{b:02x}"] += 1
    # INT imm8
    if b == 0xCD and i + 1 < len(image):
      report.interrupts.append(OpcodeHit(i, "CD", "int", f"vector={image[i+1]:#04x}"))
      i += 2
      continue
    # OUT/IN imm8
    if b == 0xE4 and i + 1 < len(image):
      report.io_ins.append(OpcodeHit(i, "E4", "in al,imm8", f"port={image[i+1]:#04x}"))
      i += 2
      continue
    if b == 0xE5 and i + 1 < len(image):
      report.io_ins.append(OpcodeHit(i, "E5", "in ax,imm8", f"port={image[i+1]:#04x}"))
      i += 2
      continue
    if b == 0xE6 and i + 1 < len(image):
      report.io_outs.append(OpcodeHit(i, "E6", "out imm8,al", f"port={image[i+1]:#04x}"))
      i += 2
      continue
    if b == 0xE7 and i + 1 < len(image):
      report.io_outs.append(OpcodeHit(i, "E7", "out imm8,ax", f"port={image[i+1]:#04x}"))
      i += 2
      continue
    if b == 0xEC:
      report.io_ins.append(OpcodeHit(i, "EC", "in al,dx"))
      i += 1
      continue
    if b == 0xED:
      report.io_ins.append(OpcodeHit(i, "ED", "in ax,dx"))
      i += 1
      continue
    if b == 0xEE:
      report.io_outs.append(OpcodeHit(i, "EE", "out dx,al"))
      i += 1
      continue
    if b == 0xEF:
      report.io_outs.append(OpcodeHit(i, "EF", "out dx,ax"))
      i += 1
      continue
    # CALL/JMP far
    if b == 0x9A and i + 5 <= len(image):
      ip = int.from_bytes(image[i + 1:i + 3], "little")
      cs = int.from_bytes(image[i + 3:i + 5], "little")
      report.far_calls.append(OpcodeHit(i, "9A", "call far", f"{cs:04x}:{ip:04x}"))
      i += 5
      continue
    if b == 0xEA and i + 5 <= len(image):
      ip = int.from_bytes(image[i + 1:i + 3], "little")
      cs = int.from_bytes(image[i + 3:i + 5], "little")
      report.far_jmps.append(OpcodeHit(i, "EA", "jmp far", f"{cs:04x}:{ip:04x}"))
      i += 5
      continue
    # RETF
    if b == 0xCB:
      report.far_calls.append(OpcodeHit(i, "CB", "retf"))
      i += 1
      continue
    # Immediate 0x0CF8 / 0x0CFC as little-endian words nearby OUT patterns
    if i + 2 <= len(image):
      w = int.from_bytes(image[i:i + 2], "little")
      if w in (0x0CF8, 0x0CFC):
        report.pci_mech.append(
          OpcodeHit(i, f"{w:04x}", "pci_mech_imm", f"port={w:#x}")
        )
    i += 1

  report.opcode_histogram = dict(hist.most_common(64))

  # Known VGA preamble site
  for off in (0x67AE, 0x2F0A, 0x657E):
    if off < len(image):
      report.vga_preamble_offsets.append(off)

  return report


def try_llvm_disassemble(image: bytes, start: int, stop: int) -> Tuple[bool, List[str]]:
  """Best-effort llvm-objdump/llvm-mc disassembly for development oracle use."""
  llvm_objdump = shutil.which("llvm-objdump")
  # Homebrew llvm may not be on PATH
  for candidate in (
    "/opt/homebrew/opt/llvm/bin/llvm-objdump",
    "/usr/local/opt/llvm/bin/llvm-objdump",
  ):
    if Path(candidate).is_file():
      llvm_objdump = candidate
      break
  if not llvm_objdump:
    return False, []

  with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
    tf.write(image)
    path = tf.name
  try:
    # Newer llvm-objdump: --triple + binary input via -D may vary.
    # Fall back to hex dump style decode via llvm-mc if needed.
    cmd = [
      llvm_objdump, "-D", "--triple=i386", "-M=i8086",
      f"--start-address={start}", f"--stop-address={stop}", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
      # Try without -M
      cmd2 = [
        llvm_objdump, "-D", "--triple=i8086",
        f"--start-address={start}", f"--stop-address={stop}", path,
      ]
      proc = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
      return True, []  # tool present but couldn't decode binary this way
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return True, lines
  except Exception:
    return True, []
  finally:
    Path(path).unlink(missing_ok=True)


def analyze_rom(rom: Optional[RomImage] = None) -> CensusReport:
  if rom is None:
    rom = load_rom()
  report = static_scan(rom.data)
  report.image_sha256 = rom.sha256
  ok, lines = try_llvm_disassemble(rom.data, ENTRY_OFFSET, ENTRY_PATH_VIA + 8)
  report.llvm_available = ok
  report.llvm_disasm_lines = len(lines)
  # Machine-readable reached/unknown ledger seed: opcodes we intentionally support.
  supported = {
    "90", "f4", "fa", "fb", "fc", "fd", "f8", "f9", "f5",
    "cd", "cf", "c3", "c2", "cb", "ca", "e8", "e9", "eb", "ea", "9a",
    "e4", "e5", "e6", "e7", "ec", "ed", "ee", "ef",
    "50", "51", "52", "53", "54", "55", "56", "57",
    "58", "59", "5a", "5b", "5c", "5d", "5e", "5f",
    "60", "61", "68", "6a", "9c", "9d",
    "b0", "b8", "a0", "a1", "a2", "a3", "a8", "a9",
    "70", "74", "75", "72", "73", "76", "77", "eb",
    "0e", "1e", "16", "06", "07", "1f", "17",
    "f6", "f7", "80", "81", "83", "88", "89", "8a", "8b",
    "30", "32", "08", "0a", "20", "22", "28", "2a", "38", "3a",
    "00", "01", "02", "03", "40", "48", "fe", "ff",
    "0f", "66", "67", "2e", "3e", "26", "36",
  }
  unknown = sorted(k for k in report.opcode_histogram if k not in supported)
  report.unknown_ledger = unknown[:128]
  return report


def write_census_json(report: CensusReport, path: Path | str) -> None:
  Path(path).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
