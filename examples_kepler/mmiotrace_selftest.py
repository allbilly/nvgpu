"""Offline golden-mmiotrace gate for GK104 cold bring-up.

Primary lock: ``golden_gk104_cold_slice.json`` — **1636** Nouveau host writes
from first RAMMAP ``10f65c=0x10`` through LTC ``17e8c0``, compared 1:1 against
``run_vbios_ram_init`` + ``_gk104_post_ram_fb_ltc`` (sole divergence:
``tag_base`` 0 vs ``0x7fddf``).

Additional checkpoints cover strap, hang regressions, live source order, and
one-shot env defaults.  No hardware / root / pagemap required.

Run::

  python3 examples_kepler/add.py --mmiotrace-selftest
"""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Golden fixtures (extracted from nouveau_gk104_mmiotrace.txt.gz, BAR0@0xfb…)
# ---------------------------------------------------------------------------

GOLDEN_MMIOTRACE = pathlib.Path(__file__).resolve().parent / "nouveau_gk104_mmiotrace.txt.gz"
GOLDEN_COLD_SLICE = pathlib.Path(__file__).resolve().parent / "golden_gk104_cold_slice.json"
GOLDEN_BAR0 = 0xFB000000

# R 0x101000 @ t+0.000s
GOLDEN_STRAP_101000 = 0x8040509A
GOLDEN_RAMCFG_GROUP = 6  # (strap >> 2) & 0xf for this ROM path → group 6
GOLDEN_PMC_BOOT0_CHIP = 0xE4  # GK104; golden R PMC_BOOT_0 = 0x0e4040a2

# First RAMMAP script iteration @ t+0.390s (before selected-script skip loop)
GOLDEN_RAMMAP_PREFIX9: List[Tuple[int, int]] = [
  (0x10F65C, 0x00000010),
  (0x11E67C, 0xFFF10000),
  (0x11E708, 0x00030222),
  (0x11E6A0, 0x04040404),
  (0x11E6A4, 0x04040404),
  (0x11E6A8, 0x0F0F0F0F),
  (0x11E6AC, 0x0F0F0F0F),
  (0x11E6B0, 0x06060606),
  (0x11E6B4, 0x06060606),
]

# Training markers @ t+0.391s (strap-6 type00[0])
GOLDEN_TRAIN_10ECC0 = 0xFFFFFFFF
GOLDEN_TRAIN_10F918_TYPE00_0 = 0x55555555

# LTC topology reads (early @0.169s and again @0.393s)
GOLDEN_LTC_PARTS = 0x4
GOLDEN_LTC_MASK = 0x0
GOLDEN_LTC_17E8DC = 0x401B0038
GOLDEN_LTC_NR = 4
GOLDEN_LTS_NR = (GOLDEN_LTC_17E8DC >> 28) & 0xF  # 4

# fb_init_page @ t+0.393s — bit0 already clear; write is a no-op keep
GOLDEN_FB_100C80 = 0x00208000

# ZBC clear @ t+0.393s: indices 1..15 colour, then 1..15 depth (slot 0 reserved)
GOLDEN_ZBC_INDICES = list(range(1, 16)) + list(range(1, 16))

# LTC program writes immediately after ZBC
GOLDEN_LTC_17E8D8 = GOLDEN_LTC_NR
GOLDEN_LTC_17E000 = GOLDEN_LTC_NR
GOLDEN_TAG_BASE = 0x0007FDDF  # VRAM-backed tags in Nouveau; userspace uses 0
GOLDEN_LTC_17E8C0 = 0x3F800FF3  # bit1 already set (lpg128)

# Host never emits these in the first 2s — PMU MEMX owns them in Nouveau.
GOLDEN_HOST_NEVER_WRITES_EARLY = (0x10F808, 0x132024, 0x001620, 0x0026F0)

# Strap-6 cold ram_program end-state (our intentional direct-host fallback)
OURS_RAM_PROGRAM_10F808 = 0x72A00000
OURS_RAM_PROGRAM_132024 = 0x00011701
OURS_RAM_PROGRAM_132030 = 0x10000000
OURS_RAM_PROGRAM_132034 = 0x00001000
OURS_TAG_BASE = 0  # no fb tags heap yet

# Exact cold-phase lengths locked against the golden gz (re-checked by loader).
# RAMMAP = scripts + 10f584 finalisation (stops before 10ecc0 training marker).
GOLDEN_RAMMAP_PHASE_LEN = 20
GOLDEN_TRAIN_PHASE_LEN = 1506
GOLDEN_ZBC_BODY_LEN = 105

MACOS_WRAPPER = pathlib.Path(__file__).resolve().parent / "add.py"


Write = Tuple[int, int]  # (reg, value)
Op = Tuple[str, int, int]  # (R|W, reg, value)


def _is_rammap_phase_reg(reg: int) -> bool:
  return (0x10F650 <= reg <= 0x10F660 or 0x11E600 <= reg <= 0x11E7FF or
          reg in (0x10ECC0, 0x10F584, 0x10F160))


def _is_train_phase_reg(reg: int) -> bool:
  return reg == 0x10ECC0 or 0x10F900 <= reg <= 0x10F9FF or reg == 0x10F160


def _is_zbc_reg(reg: int) -> bool:
  return reg in (0x17EA44, 0x17EA48, 0x17EA4C, 0x17EA50, 0x17EA54, 0x17EA58)


def _is_ltc_prog_reg(reg: int) -> bool:
  return reg in (0x17E8D8, 0x17E000, 0x17E8D4, 0x17E8C0)


@dataclass(frozen=True)
class GoldenColdPhases:
  """Exact write streams carved from the Nouveau mmiotrace cold window."""
  strap: int
  parts: int
  mask: int
  rammap: Tuple[Write, ...]
  train: Tuple[Write, ...]
  zbc: Tuple[Write, ...]
  ltc_prog: Tuple[Write, ...]  # includes golden tag_base
  host_forbidden_counts: Tuple[Tuple[int, int], ...]


_GOLDEN_CACHE: Optional[GoldenColdPhases] = None


def load_golden_cold_phases(
    mmiotrace_path: pathlib.Path = GOLDEN_MMIOTRACE) -> GoldenColdPhases:
  """Parse the checked-in mmiotrace once; return exact cold-phase write streams."""
  global _GOLDEN_CACHE
  if _GOLDEN_CACHE is not None and mmiotrace_path == GOLDEN_MMIOTRACE:
    return _GOLDEN_CACHE
  if not mmiotrace_path.is_file():
    raise AssertionError(f"missing golden mmiotrace: {mmiotrace_path}")
  pat = re.compile(
      r"^(R|W)\s+(\d+)\s+([\d.]+)\s+\d+\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)")
  t0 = None
  strap = parts = mask = None
  all_w: List[Tuple[float, int, int]] = []
  host_forbidden = {r: 0 for r in GOLDEN_HOST_NEVER_WRITES_EARLY}

  with gzip.open(mmiotrace_path, "rt", errors="replace") as f:
    for line in f:
      m = pat.match(line)
      if not m:
        continue
      op, ts, phys, val = (m.group(1), float(m.group(3)),
                           int(m.group(4), 16), int(m.group(5), 16))
      if t0 is None:
        t0 = ts
      rel = ts - t0
      if rel > 2.0:
        break
      if not (GOLDEN_BAR0 <= phys < GOLDEN_BAR0 + 0x02000000):
        continue
      reg = phys - GOLDEN_BAR0
      if op == "R" and reg == 0x101000 and strap is None:
        strap = val
      if op == "R" and reg == 0x022438 and parts is None:
        parts = val
      if op == "R" and reg == 0x022554 and mask is None:
        mask = val
      if op != "W":
        continue
      if reg in host_forbidden:
        host_forbidden[reg] += 1
      if 0.390 <= rel < 0.394:
        all_w.append((rel, reg, val))

  # Split by markers (not brittle sub-second cut-points):
  #   RAMMAP … → first 10ecc0 → training … → first 100c80/17ea44 → ZBC/LTC
  i_ecc = next(i for i, (_, r, _) in enumerate(all_w) if r == 0x10ECC0)
  i_ltc = next(i for i, (_, r, _) in enumerate(all_w)
               if i > i_ecc and (r == 0x100C80 or r == 0x17EA44))
  rammap = [(r, v) for _, r, v in all_w[:i_ecc] if _is_rammap_phase_reg(r)]
  train = [(r, v) for _, r, v in all_w[i_ecc:i_ltc] if _is_train_phase_reg(r)]
  zbc = [(r, v) for _, r, v in all_w[i_ltc:] if _is_zbc_reg(r)]
  ltc_prog = [(r, v) for _, r, v in all_w[i_ltc:] if _is_ltc_prog_reg(r)]

  phases = GoldenColdPhases(
      strap=strap if strap is not None else -1,
      parts=parts if parts is not None else -1,
      mask=mask if mask is not None else -1,
      rammap=tuple(rammap),
      train=tuple(train),
      zbc=tuple(zbc),
      ltc_prog=tuple(ltc_prog),
      host_forbidden_counts=tuple(sorted(host_forbidden.items())),
  )
  assert phases.strap == GOLDEN_STRAP_101000, phases.strap
  assert phases.parts == GOLDEN_LTC_PARTS and phases.mask == GOLDEN_LTC_MASK
  assert len(phases.rammap) == GOLDEN_RAMMAP_PHASE_LEN, len(phases.rammap)
  assert len(phases.train) == GOLDEN_TRAIN_PHASE_LEN, len(phases.train)
  assert len(phases.zbc) == GOLDEN_ZBC_BODY_LEN, len(phases.zbc)
  assert phases.rammap[:9] == tuple(GOLDEN_RAMMAP_PREFIX9)
  assert phases.train[0] == (0x10ECC0, GOLDEN_TRAIN_10ECC0)
  assert any(r == 0x10F918 and v == GOLDEN_TRAIN_10F918_TYPE00_0
             for r, v in phases.train)
  assert [v & 0xF for r, v in phases.zbc if r == 0x17EA44] == GOLDEN_ZBC_INDICES
  assert phases.ltc_prog[0] == (0x17E8D8, GOLDEN_LTC_NR)
  assert phases.ltc_prog[2] == (0x17E8D4, GOLDEN_TAG_BASE)
  assert all(n == 0 for _, n in phases.host_forbidden_counts)
  if mmiotrace_path == GOLDEN_MMIOTRACE:
    _GOLDEN_CACHE = phases
  return phases


def _ours_rammap_phase(writes: Sequence[Write]) -> List[Write]:
  out: List[Write] = []
  for r, v in writes:
    if r == 0x10ECC0:
      break
    if _is_rammap_phase_reg(r):
      out.append((r, v))
  return out


def _ours_train_phase(writes: Sequence[Write],
                      stop_at: Optional[Sequence[int]] = None) -> List[Write]:
  """Training stream from first 0x10ecc0.

  When ``stop_at`` is set (e.g. ram_program / LTC markers), collection ends
  before the first write to any of those registers so a full cold orchestrator
  run does not mix GDDR5 training with later controller programming.
  """
  stop = set(stop_at or ())
  out: List[Write] = []
  started = False
  for r, v in writes:
    if started and r in stop:
      break
    if r == 0x10ECC0:
      started = True
    if not started:
      continue
    if _is_train_phase_reg(r):
      out.append((r, v))
  return out


def _ours_zbc_body(writes: Sequence[Write]) -> List[Write]:
  return [(r, v) for r, v in writes if _is_zbc_reg(r)]


def _ours_ltc_prog(writes: Sequence[Write]) -> List[Write]:
  return [(r, v) for r, v in writes if _is_ltc_prog_reg(r)]


def _assert_exact_writes(got: Sequence[Write], exp: Sequence[Write],
                         label: str) -> None:
  if list(got) == list(exp):
    return
  for i, (a, b) in enumerate(zip(got, exp)):
    if a != b:
      raise AssertionError(
          f"{label}: mismatch at write #{i}: "
          f"got {a[0]:#x}={a[1]:#010x} expected {b[0]:#x}={b[1]:#010x} "
          f"(got_len={len(got)} exp_len={len(exp)})")
  raise AssertionError(
      f"{label}: length mismatch got={len(got)} expected={len(exp)}")


def _with_strap6(fn: Callable[[], None]) -> None:
  old = os.environ.get("KEPLER_RAMCFG_STRAP")
  os.environ["KEPLER_RAMCFG_STRAP"] = "6"
  try:
    fn()
  finally:
    if old is None:
      os.environ.pop("KEPLER_RAMCFG_STRAP", None)
    else:
      os.environ["KEPLER_RAMCFG_STRAP"] = old



@dataclass
class FakeMMIO:
  """Record every MMIO access for subsequence / order assertions."""
  regs: dict = field(default_factory=dict)
  ops: List[Op] = field(default_factory=list)
  _ltc_inited: bool = False

  def read32(self, reg: int) -> int:
    reg &= 0xffffffff
    val = self.regs.get(reg, 0) & 0xffffffff
    self.ops.append(("R", reg, val))
    return val

  def write32(self, reg: int, value: int) -> None:
    reg &= 0xffffffff
    value &= 0xffffffff
    self.regs[reg] = value
    self.ops.append(("W", reg, value))

  @property
  def writes(self) -> List[Write]:
    return [(r, v) for op, r, v in self.ops if op == "W"]

  def writes_of(self, *regs: int) -> List[Write]:
    want = set(regs)
    return [(r, v) for r, v in self.writes if r in want]

  def first_write_index(self, reg: int) -> Optional[int]:
    for i, (op, r, _) in enumerate(self.ops):
      if op == "W" and r == reg:
        return i
    return None

  def first_write_value(self, reg: int) -> Optional[int]:
    for op, r, v in self.ops:
      if op == "W" and r == reg:
        return v
    return None


def _seed_golden_topology(dev: FakeMMIO) -> None:
  """Seed registers Nouveau reads during RAM/LTC on this Palit GTX 770."""
  # Strap + PLL/lock bits ram_program may poll
  dev.regs.update({
    0x101000: GOLDEN_STRAP_101000,
    0x022438: GOLDEN_LTC_PARTS,
    0x022554: GOLDEN_LTC_MASK,
    0x17E8DC: GOLDEN_LTC_17E8DC,
    0x100C80: GOLDEN_FB_100C80,
    0x17E8C0: GOLDEN_LTC_17E8C0 & ~0x2,  # force bit1 clear so our mask sets it
    0x100710: 0x80000000,
    0x137390: 0x00020000,  # REFPLL lock
    0x10F65C: 0,
    0x10F584: 0x15004000,
    0x10F160: 0x3,
  })


def _assert_subsequence(haystack: Sequence[Write], needle: Sequence[Write],
                        label: str) -> None:
  """Require ``needle`` to appear as a contiguous subsequence of ``haystack``."""
  n = len(needle)
  if n == 0:
    return
  for i in range(len(haystack) - n + 1):
    if list(haystack[i:i + n]) == list(needle):
      return
  # Helpful near-miss: show first differing prefix from start
  got = list(haystack[:n])
  raise AssertionError(
      f"{label}: expected contiguous writes {list(needle)}, "
      f"got prefix {got} (haystack_len={len(haystack)})")


def _assert_ordered(ops: Sequence[Op], earlier_reg: int, later_reg: int,
                    label: str, earlier_op: str = "W", later_op: str = "W") -> None:
  i_e = next((i for i, (op, r, _) in enumerate(ops)
              if op == earlier_op and r == earlier_reg), None)
  i_l = next((i for i, (op, r, _) in enumerate(ops)
              if op == later_op and r == later_reg), None)
  assert i_e is not None, f"{label}: missing {earlier_op} {earlier_reg:#x}"
  assert i_l is not None, f"{label}: missing {later_op} {later_reg:#x}"
  assert i_e < i_l, (
      f"{label}: {earlier_op} {earlier_reg:#x} (op#{i_e}) must precede "
      f"{later_op} {later_reg:#x} (op#{i_l})")


# ---------------------------------------------------------------------------
# Individual checkpoints (one golden observation each)
# ---------------------------------------------------------------------------

def _is_our_cold_slice_reg(reg: int) -> bool:
  """Registers Nouveau programs in the cold GDDR5+LTC host slice we implement."""
  return (
      _is_rammap_phase_reg(reg) or _is_train_phase_reg(reg) or
      _is_zbc_reg(reg) or _is_ltc_prog_reg(reg) or reg == 0x100C80)


def _extract_cold_slice_writes(writes: Sequence[Write]) -> List[Write]:
  """From first RAMMAP ``10f65c=0x10`` through LTC ``17e8c0`` (inclusive)."""
  out: List[Write] = []
  started = False
  for r, v in writes:
    if not started and r == 0x10F65C and v == 0x10:
      started = True
    if not started:
      continue
    if _is_our_cold_slice_reg(r):
      out.append((r, v))
    if r == 0x17E8C0 and _is_our_cold_slice_reg(r):
      break
  return out


GOLDEN_COLD_SLICE_LEN = 1636


def load_golden_cold_slice(
    path: pathlib.Path = GOLDEN_COLD_SLICE) -> Tuple[List[Write], dict]:
  """Load the checked-in 1636-write golden cold slice."""
  meta = json.loads(path.read_text(encoding="utf-8"))
  writes = [(int(r), int(v)) for r, v in meta["writes"]]
  assert len(writes) == meta["write_count"] == GOLDEN_COLD_SLICE_LEN, (
      f"cold-slice fixture length {len(writes)} != {GOLDEN_COLD_SLICE_LEN}")
  return writes, meta


def test_00_cold_slice_one_to_one(
    run_ram_init: Callable[[FakeMMIO, bytes], None],
    post_ram_fb_ltc: Callable[[FakeMMIO], None],
    image: bytes) -> None:
  """Primary one-shot gate: every golden cold-slice write matches ours 1:1.

  Replays ``ram_init`` + ``_gk104_post_ram_fb_ltc`` (no direct ``ram_program`` —
  golden host uses PMU MEMX for controller/PLL).  The sole allowed divergence
  is ``0x17e8d4`` tag_base (Nouveau ``0x7fddf``, ours ``0``).
  """
  golden, meta = load_golden_cold_slice()
  assert meta["tag_base_golden"] == GOLDEN_TAG_BASE
  assert meta["tag_base_ours"] == OURS_TAG_BASE

  def _run():
    nonlocal dev
    dev = FakeMMIO()
    _seed_golden_topology(dev)
    run_ram_init(dev, image)
    post_ram_fb_ltc(dev)
  dev: FakeMMIO
  _with_strap6(_run)

  ours = _extract_cold_slice_writes(dev.writes)
  assert len(ours) == len(golden) == GOLDEN_COLD_SLICE_LEN, (
      f"cold-slice length ours={len(ours)} golden={len(golden)}")
  tag_i = meta["tag_base_index"]
  for i, (got, exp) in enumerate(zip(ours, golden)):
    if i == tag_i:
      assert got[0] == 0x17E8D4 and exp[0] == 0x17E8D4
      assert got[1] == OURS_TAG_BASE and exp[1] == GOLDEN_TAG_BASE, (
          f"tag_base: ours={got[1]:#x} golden={exp[1]:#x}")
      continue
    if got != exp:
      raise AssertionError(
          f"cold-slice mismatch at write #{i}/{len(golden)}: "
          f"got {got[0]:#x}={got[1]:#010x} expected {exp[0]:#x}={exp[1]:#010x}")


def test_00b_cold_slice_fixture_matches_gz() -> None:
  """Checked-in JSON fixture must stay byte-identical to the mmiotrace extract."""
  golden, meta = load_golden_cold_slice()
  # Re-extract from gz with the same carve rules.
  pat = re.compile(
      r"^(R|W)\s+(\d+)\s+([\d.]+)\s+\d+\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)")
  t0 = None
  extracted: List[Write] = []
  started = False
  with gzip.open(GOLDEN_MMIOTRACE, "rt", errors="replace") as f:
    for line in f:
      m = pat.match(line)
      if not m:
        continue
      op, ts, phys, val = (m.group(1), float(m.group(3)),
                           int(m.group(4), 16), int(m.group(5), 16))
      if t0 is None:
        t0 = ts
      if ts - t0 > 1.0:
        break
      if not (GOLDEN_BAR0 <= phys < GOLDEN_BAR0 + 0x02000000):
        continue
      reg = phys - GOLDEN_BAR0
      if op != "W":
        continue
      if not started and reg == 0x10F65C and val == 0x10:
        started = True
      if not started:
        continue
      if _is_our_cold_slice_reg(reg):
        extracted.append((reg, val))
      if reg == 0x17E8C0:
        break
  assert extracted == golden, (
      f"fixture drift vs gz: len fixture={len(golden)} gz={len(extracted)}")
  assert meta["strap_101000"] == GOLDEN_STRAP_101000


def test_00c_live_ram_program_preserves_golden_slice(
    run_cold: Callable[[FakeMMIO, bytes], None], image: bytes) -> None:
  """Live default inserts ``ram_program`` between train and LTC.

  One-shot replug always runs that path (``KEPLER_RAM_PROGRAM=1``).  Assert the
  golden 1636-write slice still appears as pre (RAMMAP+train) + tail (fb/ZBC/LTC)
  around the direct-host controller block — ram_program must not clobber tables.
  """
  golden, _meta = load_golden_cold_slice()
  split = next(i for i, (r, _) in enumerate(golden) if r in (0x100C80, 0x17EA44))
  pre, tail = golden[:split], golden[split:]

  def _run():
    nonlocal dev
    dev = FakeMMIO()
    _seed_golden_topology(dev)
    run_cold(dev, image)  # ram_init + ram_program + post_ram_fb_ltc
  dev: FakeMMIO
  _with_strap6(_run)

  # Pre: cold-slice regs from first RAMMAP until ram_program markers.
  stop = {0x10F808, 0x001620, 0x132024}
  pre_ours: List[Write] = []
  started = False
  for r, v in dev.writes:
    if started and r in stop:
      break
    if r == 0x10F65C and v == 0x10:
      started = True
    if not started:
      continue
    if _is_our_cold_slice_reg(r):
      pre_ours.append((r, v))
  _assert_exact_writes(pre_ours, pre, "live pre-slice (before ram_program)")

  # Tail: from first fb/ZBC through 17e8c0.
  i_tail = next(i for i, (r, _) in enumerate(dev.writes)
                if r in (0x100C80, 0x17EA44))
  tail_ours = [(r, v) for r, v in dev.writes[i_tail:] if _is_our_cold_slice_reg(r)]
  # Stop after first 17e8c0 in the LTC program.
  cut = next(i for i, (r, _) in enumerate(tail_ours) if r == 0x17E8C0)
  tail_ours = tail_ours[:cut + 1]
  assert len(tail_ours) == len(tail)
  for i, (got, exp) in enumerate(zip(tail_ours, tail)):
    if got[0] == 0x17E8D4:
      assert got[1] == OURS_TAG_BASE and exp[1] == GOLDEN_TAG_BASE
      continue
    assert got == exp, f"live tail-slice #{i}: got {got} expected {exp}"


def test_01_strap_selects_ramcfg_group6(read_ramcfg: Callable[[FakeMMIO], int]) -> None:
  """Golden: R 0x101000 = 0x8040509a → RAMCFG group 6."""
  dev = FakeMMIO()
  _seed_golden_topology(dev)
  # Clear override so the helper reads the seeded strap.
  old = os.environ.pop("KEPLER_RAMCFG_STRAP", None)
  try:
    group = read_ramcfg(dev)
  finally:
    if old is not None:
      os.environ["KEPLER_RAMCFG_STRAP"] = old
  assert dev.regs[0x101000] == GOLDEN_STRAP_101000
  assert group == GOLDEN_RAMCFG_GROUP, (
      f"strap {GOLDEN_STRAP_101000:#x} must select RAMCFG group "
      f"{GOLDEN_RAMCFG_GROUP}, got {group}")


def test_02_rammap_phase_exact(run_ram_init: Callable[[FakeMMIO, bytes], None],
                               image: bytes, golden: GoldenColdPhases) -> None:
  """Golden @0.390s: full RAMMAP/script write stream is byte-identical."""
  def _run():
    nonlocal dev
    dev = FakeMMIO()
    _seed_golden_topology(dev)
    run_ram_init(dev, image)
  dev: FakeMMIO
  _with_strap6(_run)
  _assert_exact_writes(_ours_rammap_phase(dev.writes), golden.rammap,
                       "RAMMAP phase")


def test_03_training_phase_exact(run_ram_init: Callable[[FakeMMIO, bytes], None],
                                 image: bytes, golden: GoldenColdPhases) -> None:
  """Golden @0.391s: full GDDR5 training stream (1506 writes) matches exactly."""
  def _run():
    nonlocal dev
    dev = FakeMMIO()
    _seed_golden_topology(dev)
    run_ram_init(dev, image)
  dev: FakeMMIO
  _with_strap6(_run)
  _assert_exact_writes(_ours_train_phase(dev.writes), golden.train,
                       "training phase")
  # Wrong strap (0) must not be silently accepted when 0x101000 is zero.
  bad = FakeMMIO()
  bad.regs[0x101000] = 0
  old = os.environ.pop("KEPLER_RAMCFG_STRAP", None)
  os.environ.pop("KEPLER_RAMCFG_ALLOW_ZERO", None)
  try:
    raised = False
    try:
      run_ram_init(bad, image)
    except RuntimeError as e:
      raised = True
      assert "0x101000" in str(e) or "strap" in str(e).lower()
    assert raised, "zero strap must refuse RAM init without KEPLER_RAMCFG_ALLOW_ZERO"
  finally:
    if old is not None:
      os.environ["KEPLER_RAMCFG_STRAP"] = old


def test_04_fb_init_page_clears_bigpage_bit(fb_init_page: Callable[[FakeMMIO], None]) -> None:
  """Golden @0.393s: 0x100c80 keeps bit0 clear (128 KiB big pages)."""
  dev = FakeMMIO()
  dev.regs[0x100C80] = GOLDEN_FB_100C80 | 0x1  # force wrong mode
  fb_init_page(dev)
  assert (dev.regs[0x100C80] & 1) == 0, "fb_init_page must clear 0x100c80 bit0"
  # Golden keep path: already clear → write same value
  dev2 = FakeMMIO()
  dev2.regs[0x100C80] = GOLDEN_FB_100C80
  fb_init_page(dev2)
  assert dev2.regs[0x100C80] == GOLDEN_FB_100C80


def test_05_ltc_topology_parts4(ltc_init: Callable[[FakeMMIO], None]) -> None:
  """Golden: parts=4, mask=0 → ltc_nr=4; sentinel parts must not hang."""
  # Golden topology
  dev = FakeMMIO()
  _seed_golden_topology(dev)
  # Restore 17e8c0 without bit1 so mask sets it (seed clears bit1)
  ltc_init(dev)
  assert dev.regs[0x17E8D8] == GOLDEN_LTC_NR
  assert dev.regs[0x17E000] == GOLDEN_LTC_NR
  # Hang guard: unmasked 0xffffffff previously spun in range(parts)
  hung = FakeMMIO()
  hung.regs.update({0x022438: 0xFFFFFFFF, 0x022554: 0, 0x17E8DC: GOLDEN_LTC_17E8DC,
                    0x100C80: GOLDEN_FB_100C80, 0x17E8C0: 0})
  ltc_init(hung)
  assert hung.regs[0x17E8D8] == 32, "parts must be masked/capped (≤32)"
  assert GOLDEN_LTS_NR == 4


def test_06_zbc_body_exact(ltc_init: Callable[[FakeMMIO], None],
                           golden: GoldenColdPhases) -> None:
  """Golden ZBC clear body (105 writes: idx 1..15 colour then depth) exact."""
  dev = FakeMMIO()
  _seed_golden_topology(dev)
  ltc_init(dev)
  _assert_exact_writes(_ours_zbc_body(dev.writes), golden.zbc, "ZBC body")
  assert 0 not in [v & 0xF for r, v in golden.zbc if r == 0x17EA44]


def test_07_ltc_program_after_zbc(ltc_init: Callable[[FakeMMIO], None],
                                  golden: GoldenColdPhases) -> None:
  """Golden: after ZBC → W 17e8d8=4, 17e000=4, 17e8d4=tag, 17e8c0 lpg128."""
  dev = FakeMMIO()
  _seed_golden_topology(dev)
  ltc_init(dev)
  _assert_ordered(dev.ops, 0x17EA44, 0x17E8D8, "ZBC before LTC program")
  ours = _ours_ltc_prog(dev.writes)
  assert len(ours) >= 4, ours
  assert ours[0] == (0x17E8D8, GOLDEN_LTC_NR)
  assert ours[1] == (0x17E000, GOLDEN_LTC_NR)
  assert ours[2] == (0x17E8D4, OURS_TAG_BASE), (
      "userspace tag_base must be 0 (golden "
      f"{GOLDEN_TAG_BASE:#x})")
  assert (ours[3][1] & 0x2) == 0x2, "lpg128 bit must be set in 0x17e8c0"
  # Same register order as golden; only tag_base value may differ.
  assert [r for r, _ in ours[:4]] == [r for r, _ in golden.ltc_prog[:4]]
  assert golden.ltc_prog[2] == (0x17E8D4, GOLDEN_TAG_BASE)


def test_08_cold_order_ram_before_fb_before_ltc(
    run_cold: Callable[[FakeMMIO, bytes], None], image: bytes) -> None:
  """Golden order: RAMMAP/train → 100c80 → ZBC/LTC.  No FECS regs in between."""
  old = os.environ.get("KEPLER_RAMCFG_STRAP")
  os.environ["KEPLER_RAMCFG_STRAP"] = "6"
  try:
    dev = FakeMMIO()
    _seed_golden_topology(dev)
    run_cold(dev, image)
  finally:
    if old is None:
      os.environ.pop("KEPLER_RAMCFG_STRAP", None)
    else:
      os.environ["KEPLER_RAMCFG_STRAP"] = old

  _assert_ordered(dev.ops, 0x10F65C, 0x100C80, "RAMMAP before fb_init_page")
  _assert_ordered(dev.ops, 0x10ECC0, 0x100C80, "training before fb_init_page")
  _assert_ordered(dev.ops, 0x100C80, 0x17EA44, "fb_init_page before ZBC")
  _assert_ordered(dev.ops, 0x17EA44, 0x17E8D8, "ZBC before LTC nr")
  # FECS must not appear in this cold helper (loaded later in bring-up).
  fecs = [r for op, r, _ in dev.ops if op == "W" and 0x409000 <= r <= 0x409FFF]
  assert not fecs, f"cold RAM+LTC helper must not touch FECS, got {fecs[:8]}"


def test_09_ram_program_strap6_endstate(
    run_ram_program: Callable[[FakeMMIO, bytes], dict], image: bytes) -> None:
  """Direct-host cold ram_program end-state for strap-6 @ 648 MHz.

  Golden host never writes these (MEMX); this asserts *our* fallback math so
  a strap/PLL regression cannot ship unnoticed.
  """
  old = os.environ.get("KEPLER_RAMCFG_STRAP")
  os.environ["KEPLER_RAMCFG_STRAP"] = "6"
  try:
    dev = FakeMMIO()
    _seed_golden_topology(dev)
    # Enter-bit and misc controller state ram_program expects
    cfg = run_ram_program(dev, image)
  finally:
    if old is None:
      os.environ.pop("KEPLER_RAMCFG_STRAP", None)
    else:
      os.environ["KEPLER_RAMCFG_STRAP"] = old
  assert cfg.get("ramcfg_index") == GOLDEN_RAMCFG_GROUP, cfg
  assert dev.regs[0x10F808] == OURS_RAM_PROGRAM_10F808, (
      f"0x10f808={dev.regs.get(0x10F808, 0):#x} != {OURS_RAM_PROGRAM_10F808:#x}")
  assert dev.regs[0x132024] == OURS_RAM_PROGRAM_132024
  assert dev.regs[0x132030] == OURS_RAM_PROGRAM_132030
  assert dev.regs[0x132034] == OURS_RAM_PROGRAM_132034
  assert not any(r == 0x1620 for r, _ in dev.writes), \
      "default KEPLER_RAM_BLOCK=0 must not touch host 0x1620"
  # Dangerous opt-in still emits the host pause masks.
  old_block = os.environ.get("KEPLER_RAM_BLOCK")
  os.environ["KEPLER_RAM_BLOCK"] = "direct"
  try:
    dev2 = FakeMMIO()
    _seed_golden_topology(dev2)
    run_ram_program(dev2, image)
  finally:
    if old_block is None:
      os.environ.pop("KEPLER_RAM_BLOCK", None)
    else:
      os.environ["KEPLER_RAM_BLOCK"] = old_block
  assert any(r == 0x1620 for r, _ in dev2.writes), \
      "KEPLER_RAM_BLOCK=direct must touch 0x1620"
  # bit0-only mode defers pause until after MEMX, then clears only bit0.
  os.environ["KEPLER_RAM_BLOCK"] = "bit0"
  try:
    dev3 = FakeMMIO()
    _seed_golden_topology(dev3)
    # Seed pause regs so masks are visible; seed boot0 so post-MEMX
    # health check does not treat an empty FakeMMIO as dead BAR.
    dev3.regs[0x1620] = 0xaab
    dev3.regs[0x26f0] = 0x11
    dev3.regs[0] = 0x0e4040a2
    run_ram_program(dev3, image)
  finally:
    if old_block is None:
      os.environ.pop("KEPLER_RAM_BLOCK", None)
    else:
      os.environ["KEPLER_RAM_BLOCK"] = old_block
  bit0_writes = [v for r, v in dev3.writes if r == 0x1620]
  assert bit0_writes, "KEPLER_RAM_BLOCK=bit0 must touch 0x1620"
  assert any((v & 1) == 0 for v in bit0_writes), \
      f"bit0 pause must clear 0x1620[0]: {bit0_writes}"
  # Night5 default: clear only 0x1620[0], no unpause / no 0x26f0.
  assert not any((v & 1) == 1 for v in bit0_writes), \
      f"default bit0 must not restore 0x1620[0]: {bit0_writes}"
  assert not any(r == 0x26f0 for r, _ in dev3.writes), \
      "default bit0 must not touch 0x26f0"
  assert all((v & 0xaa2) == (0xaab & 0xaa2) for v in bit0_writes), \
      f"bit0 mode must not clear 0x1620[0xaa2]: {bit0_writes}"


def test_10_golden_file_phases(golden: GoldenColdPhases) -> None:
  """Loader self-check: phase lengths and key markers stay locked."""
  assert len(golden.rammap) == GOLDEN_RAMMAP_PHASE_LEN
  assert len(golden.train) == GOLDEN_TRAIN_PHASE_LEN
  assert len(golden.zbc) == GOLDEN_ZBC_BODY_LEN
  assert golden.strap == GOLDEN_STRAP_101000
  assert all(n == 0 for _, n in golden.host_forbidden_counts)


def test_11_intentional_divergences_are_explicit() -> None:
  """Documented cold-path divergences must stay explicit constants."""
  assert OURS_TAG_BASE == 0
  assert GOLDEN_TAG_BASE == 0x7FDDF
  assert OURS_TAG_BASE != GOLDEN_TAG_BASE
  assert GOLDEN_HOST_NEVER_WRITES_EARLY == (0x10F808, 0x132024, 0x001620, 0x0026F0)


def test_12_macos_replug_preflight() -> None:
  """macOS TinyGPU wrapper must pin golden strap/MEMX and allow this gate offline."""
  src = MACOS_WRAPPER.read_text(encoding="utf-8")
  assert 'setdefault("KEPLER_RAMCFG_STRAP", "6")' in src, \
      "macOS wrapper must default KEPLER_RAMCFG_STRAP=6 for Palit golden"
  assert 'setdefault("KEPLER_PMU_MEMX", "1")' in src, \
      "macOS wrapper must default KEPLER_PMU_MEMX=1 (MEMX WR32 path)"
  assert 'setdefault("KEPLER_PMU_ENTER_NOWAIT", "1")' in src, \
      "macOS wrapper must default KEPLER_PMU_ENTER_NOWAIT=1 (patch FB_PAUSE hang)"
  assert 'setdefault("KEPLER_RAM_BLOCK", "bit0")' in src, \
      "macOS live path must default KEPLER_RAM_BLOCK=bit0 (safe TinyGPU pause)"
  assert 'setdefault("KEPLER_RAM_BLOCK", "0")' in src, \
      "macOS offline path must default KEPLER_RAM_BLOCK=0 (golden mmiotrace)"
  assert 'setdefault("KEPLER_RAM_MEMX_WR", "1")' in src, \
      "macOS wrapper must default KEPLER_RAM_MEMX_WR=1"
  assert 'setdefault("KEPLER_RAM_REQUIRE_MEMX", "1")' in src, \
      "macOS live path must refuse host GDDR5 without MEMX"
  assert 'setdefault("KEPLER_PRAMIN_SOFT_LIVE", "1")' in src, \
      "macOS live path must soft-accept virgin PRAMIN (no writeback probe)"
  assert 'setdefault("KEPLER_REFUSE_DIRTY", "1")' in src, \
      "macOS live path must refuse GPC-awake+PRAMIN-stub without power cycle"
  assert 'setdefault("KEPLER_POST_RAM_LTC", "0")' in src, \
      "macOS live path must skip post-RAM LTC/ZBC after bit0 (kills BAR0)"
  assert 'setdefault("KEPLER_PGRAPH_BLCG", "0")' in src, \
      "macOS live path must skip PGRAPH BLCG writes after bit0 (kills BAR0)"
  assert 'setdefault("KEPLER_RAM_BIT0_DEFER", "1")' in src, \
      "macOS live path must defer bit0 until first PRAMIN (pack before bit0)"
  assert 'setdefault("KEPLER_PRAMIN_LITERAL", "0")' in src, \
      "macOS live path must skip PRAMIN literal fallback on XOR virgin"
  assert 'setdefault("KEPLER_PRAMIN_MEMX", "1")' in src, \
      "macOS live path must PRAMIN-store via MEMX after bit0 (host 0x1700 kills)"
  assert 'setdefault("KEPLER_TINYGPU_ATOMIC_BAR1", "1")' in src, \
      "macOS live path must opt into pre-bit0 BAR1 root staging"
  assert 'setdefault("KEPLER_RAM_REQUIRE_MEMX", "0")' in src, \
      "macOS offline path must allow host ram_program for golden mmiotrace"
  assert "--mmiotrace-selftest" in src, \
      "macOS wrapper must treat --mmiotrace-selftest as an offline flag"
  pcie = pathlib.Path(__file__).resolve().parent.parent / "examples_kepler_pcie" / "add.py"
  pcie_src = pcie.read_text(encoding="utf-8")
  assert 'setdefault("KEPLER_TINYGPU_ATOMIC_BAR1"' not in pcie_src, \
      "shared/Linux entrypoint must not enable TinyGPU BAR1 root staging"
  assert "_gk104_post_ram_fb_ltc" in pcie_src, \
      "live cold path must call shared _gk104_post_ram_fb_ltc"


def test_13_live_source_cold_order() -> None:
  """Live bring-up source must keep golden order: RAM → ram_program → fb/LTC → FECS."""
  pcie = pathlib.Path(__file__).resolve().parent.parent / "examples_kepler_pcie" / "add.py"
  src = pcie.read_text(encoding="utf-8")
  # Use the cold-path block markers (unique enough in this file).
  i_ram = src.find("nvbios_init.run_vbios_ram_init(self, image")
  i_prog = src.find("nvbios_init.run_vbios_ram_program(\n          self, image")
  if i_prog < 0:
    i_prog = src.find("nvbios_init.run_vbios_ram_program(")
  i_ltc = src.find("_gk104_post_ram_fb_ltc(self)")
  i_fecs = src.find('fecs_code = bytearray(_rd("gk104_fecs_code.bin")')
  assert min(i_ram, i_prog, i_ltc, i_fecs) >= 0, \
      f"missing cold markers ram={i_ram} prog={i_prog} ltc={i_ltc} fecs={i_fecs}"
  assert i_ram < i_prog < i_ltc < i_fecs, (
      f"live cold order broken: ram@{i_ram} prog@{i_prog} ltc@{i_ltc} fecs@{i_fecs}")

  def _line_at(idx: int) -> str:
    start = src.rfind("\n", 0, idx) + 1
    end = src.find("\n", idx)
    return src[start:end if end >= 0 else None]

  line_ltc = _line_at(i_ltc)
  line_fecs = _line_at(i_fecs)
  # May be gated by KEPLER_POST_RAM_LTC (TinyGPU skips after bit0).
  assert "_gk104_post_ram_fb_ltc(self)" in line_ltc, \
      f"post_ram_fb_ltc call missing on line, got {line_ltc!r}"
  assert line_fecs.startswith("    fecs_code"), \
      f"fecs load should be method-body level, got {line_fecs!r}"
  assert 'get("KEPLER_POST_RAM_LTC"' in src or 'KEPLER_POST_RAM_LTC' in src, \
      "cold path must gate post-RAM LTC behind KEPLER_POST_RAM_LTC"


def test_14_oneshot_env_defaults() -> None:
  """Defaults required for first-shot cold Palit bring-up after replug."""
  pcie = pathlib.Path(__file__).resolve().parent.parent / "examples_kepler_pcie" / "add.py"
  src = pcie.read_text(encoding="utf-8")
  assert 'os.environ.get("KEPLER_RAM_PROGRAM", "1")' in src
  assert 'os.environ.get("KEPLER_RAM_FREQ", "648")' in src
  assert 'get("KEPLER_RAM_INIT", "1")' in src
  mac = MACOS_WRAPPER.read_text(encoding="utf-8")
  assert 'setdefault("KEPLER_RAMCFG_STRAP", "6")' in mac
  assert 'setdefault("KEPLER_PMU_MEMX", "1")' in mac
  assert 'setdefault("KEPLER_PMU_ENTER_NOWAIT", "1")' in mac
  assert 'setdefault("KEPLER_RAM_BLOCK", "bit0")' in mac
  assert 'setdefault("KEPLER_RAM_BLOCK", "0")' in mac
  assert 'setdefault("KEPLER_RAM_MEMX_WR", "1")' in mac
  assert 'setdefault("KEPLER_RAM_REQUIRE_MEMX", "1")' in mac
  assert 'setdefault("KEPLER_PRAMIN_SOFT_LIVE", "1")' in mac
  assert 'setdefault("KEPLER_RAM_REQUIRE_MEMX", "0")' in mac


def test_15_full_cold_orchestrator_exact(
    run_cold: Callable[[FakeMMIO, bytes], None],
    image: bytes, golden: GoldenColdPhases) -> None:
  """Single FakeMMIO run of the live cold sequence must match every golden phase."""
  def _run():
    nonlocal dev
    dev = FakeMMIO()
    _seed_golden_topology(dev)
    run_cold(dev, image)
  dev: FakeMMIO
  _with_strap6(_run)
  _assert_exact_writes(_ours_rammap_phase(dev.writes), golden.rammap,
                       "orchestrator RAMMAP")
  _assert_exact_writes(
      _ours_train_phase(dev.writes, stop_at=(0x10F808, 0x001620, 0x132024,
                                             0x100C80, 0x17EA44)),
      golden.train, "orchestrator training")
  _assert_exact_writes(_ours_zbc_body(dev.writes), golden.zbc,
                       "orchestrator ZBC")
  # Our intentional direct ram_program must still run (golden host uses MEMX).
  assert any(r == 0x10F808 for r, _ in dev.writes), \
      "cold orchestrator must emit direct 0x10f808 (host ram_program path)"
  assert any(r == 0x132024 for r, _ in dev.writes), \
      "cold orchestrator must program 0x132024 PLL"
  fecs = [r for op, r, _ in dev.ops if op == "W" and 0x409000 <= r <= 0x409FFF]
  assert not fecs, f"cold orchestrator must stop before FECS, got {fecs[:8]}"
  # Order across the full stream
  _assert_ordered(dev.ops, 0x10F65C, 0x10ECC0, "RAMMAP before train")
  _assert_ordered(dev.ops, 0x10ECC0, 0x100C80, "train before fb_page")
  _assert_ordered(dev.ops, 0x100C80, 0x17EA44, "fb_page before ZBC")
  _assert_ordered(dev.ops, 0x17EA44, 0x17E8D8, "ZBC before LTC")
  _assert_ordered(dev.ops, 0x10F808, 0x17E8D8, "ram_program before LTC")


def test_16_sysmem_flush_page_and_bar1_contracts() -> None:
  """Golden programs 0x100c10 early and BAR1 before FECS; we do both later.

  Lock the *formulas* so a one-shot run still programs them correctly once
  sysmem exists, and keep the intentional order divergence explicit.
  """
  pcie = pathlib.Path(__file__).resolve().parent.parent / "examples_kepler_pcie" / "add.py"
  src = pcie.read_text(encoding="utf-8")
  assert "self.write32(0x100c10, dev.bus_base >> 8)" in src, \
      "live path must program sysmem flush page at 0x100c10 = bus_base>>8"
  assert "_gk104_init_bar1" in src
  # Documented order divergence vs golden (BAR1 @0.87s, FECS @20s):
  # our FECS load is immediately after LTC; BAR1/100c10 follow sysmem alloc.
  i_ltc = src.find("_gk104_post_ram_fb_ltc(self)")
  i_fecs = src.find('fecs_code = bytearray(_rd("gk104_fecs_code.bin")')
  i_flush = src.find("self.write32(0x100c10, dev.bus_base >> 8)")
  assert i_ltc < i_fecs < i_flush, (
      "expected live order LTC → FECS load → (later) 0x100c10; "
      f"ltc@{i_ltc} fecs@{i_fecs} flush@{i_flush}")


def test_17_rammap_second_script_present(golden: GoldenColdPhases) -> None:
  """Golden RAMMAP runs both non-selected scripts (10f65c=0x10 then 0x20)."""
  vals = [v for r, v in golden.rammap if r == 0x10F65C]
  assert vals[:2] == [0x10, 0x20], f"expected script selects 0x10 then 0x20, got {vals}"
  assert golden.rammap[-1] == (0x10F584, 0x04004000) or \
         any(r == 0x10F584 and v == 0x04004000 for r, v in golden.rammap), \
      "mode finalisation 0x10f584 missing from golden RAMMAP phase"


def test_18_every_cold_phase_write_is_locked(golden: GoldenColdPhases) -> None:
  """Every golden W in RAMMAP/train/ZBC/LTC ranges must sit in a locked phase.

  Scans the mmiotrace cold cluster and asserts we did not leave a single
  relevant write unverified (PRAMIN 0x7a**** / early BAR1 / 0x100c10 are
  outside this cold GDDR5+LTC contract).
  """
  pat = re.compile(
      r"^(R|W)\s+(\d+)\s+([\d.]+)\s+\d+\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)")
  t0 = None
  locked = set(golden.rammap) | set(golden.train) | set(golden.zbc) | set(golden.ltc_prog)
  # 100c80 keep-write is part of fb_init_page; allow either value equality.
  missing: List[Write] = []
  with gzip.open(GOLDEN_MMIOTRACE, "rt", errors="replace") as f:
    for line in f:
      m = pat.match(line)
      if not m:
        continue
      op, ts, phys, val = (m.group(1), float(m.group(3)),
                           int(m.group(4), 16), int(m.group(5), 16))
      if t0 is None:
        t0 = ts
      rel = ts - t0
      if rel > 0.40:
        break
      if rel < 0.389 or op != "W":
        continue
      if not (GOLDEN_BAR0 <= phys < GOLDEN_BAR0 + 0x02000000):
        continue
      reg = phys - GOLDEN_BAR0
      if not (_is_rammap_phase_reg(reg) or _is_train_phase_reg(reg) or
              _is_zbc_reg(reg) or _is_ltc_prog_reg(reg) or reg == 0x100C80):
        continue
      if reg == 0x100C80:
        assert val == GOLDEN_FB_100C80, val
        continue
      if (reg, val) not in locked:
        missing.append((reg, val))
  assert not missing, (
      f"{len(missing)} golden cold-phase write(s) not in locked streams; "
      f"first={missing[0][0]:#x}={missing[0][1]:#x}")


def test_19_replug_hang_regressions(ltc_init: Callable[[FakeMMIO], None],
                                    post_ram_fb_ltc: Callable[[FakeMMIO], None],
                                    topo_is_posted: Callable[[int], bool]) -> None:
  """Lock the failure modes that previously burned a cold eGPU session."""
  # 1) Unmasked LTC parts hung bring-up in pure-Python range().
  hung = FakeMMIO()
  hung.regs.update({0x022438: 0xFFFFFFFF, 0x022554: 0, 0x17E8DC: GOLDEN_LTC_17E8DC,
                    0x100C80: GOLDEN_FB_100C80, 0x17E8C0: 0})
  ltc_init(hung)
  assert hung.regs[0x17E8D8] == 32
  # 2) All-ones / badf topology must NOT look POSTed (would skip cold RAM/LTC).
  assert not topo_is_posted(0xffffffff)
  assert not topo_is_posted(0xbadf1200)
  assert not topo_is_posted(0)
  assert topo_is_posted(0x00040004)
  # 3) post_ram_fb_ltc is idempotent (late safety call must not double-init).
  dev = FakeMMIO()
  _seed_golden_topology(dev)
  post_ram_fb_ltc(dev)
  n1 = len(dev.writes)
  post_ram_fb_ltc(dev)
  assert len(dev.writes) == n1, "second _gk104_post_ram_fb_ltc must be a no-op"
  assert getattr(dev, "_ltc_inited", False) is True


def test_20_gk104_boot0_and_dead_bar_guards(add_src: str) -> None:
  """Live path must refuse dead BAR / non-GK104 before touching RAM."""
  assert "GK104_PMC_BOOT0_CHIP" in add_src or "0xe4" in add_src.lower()
  assert "_gk104_boot0_looks_live" in add_src
  assert "_gk104_ensure_bar0_mmio" in add_src
  assert "physical power or Thunderbolt link cycle required" in add_src
  assert "_gk104_pramin_looks_live" in add_src
  assert "_gk104_pramin_word_is_stub" in add_src
  # Golden Nouveau on this card: PMC_BOOT_0 chip_id 0xe4 (GK104).
  assert GOLDEN_PMC_BOOT0_CHIP == 0xE4


def test_21_replug_runbook_in_wrappers() -> None:
  """Wrappers must expose the offline gate and not require LIVE_ACK for it."""
  mac = MACOS_WRAPPER.read_text(encoding="utf-8")
  # Offline list includes mmiotrace (no LIVE_ACK / RPC_TRACE).
  assert "--mmiotrace-selftest" in mac
  # Probe is a separate path; full-add after replug must not depend on probe.
  assert "def _probe" in mac or '"--probe"' in mac
  pcie = pathlib.Path(__file__).resolve().parent.parent / "examples_kepler_pcie" / "add.py"
  pcie_src = pcie.read_text(encoding="utf-8")
  assert "--mmiotrace-selftest" in pcie_src
  assert "no hardware / pagemap" in pcie_src or "mmiotrace-selftest" in pcie_src


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class KeplerMMIOHooks:
  """Callbacks into examples_kepler_pcie.add / nvbios_init (injected by caller)."""
  vbios_path: str
  find_image: Callable[[bytes], bytes]
  read_ramcfg: Callable[[FakeMMIO], int]
  run_ram_init: Callable[[FakeMMIO, bytes], None]
  run_ram_program: Callable[[FakeMMIO, bytes], dict]
  fb_init_page: Callable[[FakeMMIO], None]
  ltc_init: Callable[[FakeMMIO], None]
  run_cold_ram_fb_ltc: Callable[[FakeMMIO, bytes], None]
  post_ram_fb_ltc: Callable[[FakeMMIO], None]
  topo_is_posted: Callable[[int], bool]
  pcie_add_source: str


def run_mmiotrace_selftest(hooks: KeplerMMIOHooks, *, verbose: bool = True) -> int:
  """Run every golden checkpoint.  Returns 0 on success."""
  image = hooks.find_image(pathlib.Path(hooks.vbios_path).read_bytes())
  golden = load_golden_cold_phases()
  tests: List[Tuple[str, Callable[[], None]]] = [
    ("00_cold_slice_1to1",
     lambda: test_00_cold_slice_one_to_one(
         hooks.run_ram_init, hooks.post_ram_fb_ltc, image)),
    ("00b_cold_slice_fixture_gz",
     lambda: test_00b_cold_slice_fixture_matches_gz()),
    ("00c_live_ram_program_preserves_slice",
     lambda: test_00c_live_ram_program_preserves_golden_slice(
         hooks.run_cold_ram_fb_ltc, image)),
    ("01_strap_ramcfg6",
     lambda: test_01_strap_selects_ramcfg_group6(hooks.read_ramcfg)),
    ("02_rammap_phase_exact",
     lambda: test_02_rammap_phase_exact(hooks.run_ram_init, image, golden)),
    ("03_training_phase_exact",
     lambda: test_03_training_phase_exact(hooks.run_ram_init, image, golden)),
    ("04_fb_init_page",
     lambda: test_04_fb_init_page_clears_bigpage_bit(hooks.fb_init_page)),
    ("05_ltc_topology",
     lambda: test_05_ltc_topology_parts4(hooks.ltc_init)),
    ("06_zbc_body_exact",
     lambda: test_06_zbc_body_exact(hooks.ltc_init, golden)),
    ("07_ltc_program",
     lambda: test_07_ltc_program_after_zbc(hooks.ltc_init, golden)),
    ("08_cold_order",
     lambda: test_08_cold_order_ram_before_fb_before_ltc(
         hooks.run_cold_ram_fb_ltc, image)),
    ("09_ram_program_endstate",
     lambda: test_09_ram_program_strap6_endstate(hooks.run_ram_program, image)),
    ("10_golden_file_phases",
     lambda: test_10_golden_file_phases(golden)),
    ("11_divergences_explicit",
     lambda: test_11_intentional_divergences_are_explicit()),
    ("12_macos_replug_preflight",
     lambda: test_12_macos_replug_preflight()),
    ("13_live_source_cold_order",
     lambda: test_13_live_source_cold_order()),
    ("14_oneshot_env_defaults",
     lambda: test_14_oneshot_env_defaults()),
    ("15_full_cold_orchestrator",
     lambda: test_15_full_cold_orchestrator_exact(
         hooks.run_cold_ram_fb_ltc, image, golden)),
    ("16_flush_page_bar1_contracts",
     lambda: test_16_sysmem_flush_page_and_bar1_contracts()),
    ("17_rammap_second_script",
     lambda: test_17_rammap_second_script_present(golden)),
    ("18_every_cold_write_locked",
     lambda: test_18_every_cold_phase_write_is_locked(golden)),
    ("19_replug_hang_regressions",
     lambda: test_19_replug_hang_regressions(
         hooks.ltc_init, hooks.post_ram_fb_ltc, hooks.topo_is_posted)),
    ("20_gk104_boot0_guards",
     lambda: test_20_gk104_boot0_and_dead_bar_guards(hooks.pcie_add_source)),
    ("21_replug_runbook",
     lambda: test_21_replug_runbook_in_wrappers()),
  ]
  failed = 0
  for name, fn in tests:
    try:
      fn()
      if verbose:
        print(f"mmiotrace_selftest: PASS {name}", flush=True)
    except Exception as e:
      failed += 1
      print(f"mmiotrace_selftest: FAIL {name}: {e}", flush=True)
  if failed:
    print(f"mmiotrace_selftest: FAILED {failed}/{len(tests)}", flush=True)
    return 1
  print(f"mmiotrace_selftest: ok ({len(tests)} checkpoints; "
        f"cold_slice={GOLDEN_COLD_SLICE_LEN} writes 1:1; "
        f"rammap={len(golden.rammap)} train={len(golden.train)} "
        f"zbc={len(golden.zbc)})", flush=True)
  return 0


def build_hooks_from_add_module(add_mod, vbios_path: Optional[str] = None) -> KeplerMMIOHooks:
  """Wire hooks against the loaded ``examples_kepler_pcie.add`` module."""
  import nvbios_init  # shared kepler helper on sys.path

  path = vbios_path or getattr(add_mod, "DEFAULT_VBIOS")
  image = nvbios_init.find_vbios_image(pathlib.Path(path).read_bytes())

  def read_ramcfg(dev: FakeMMIO) -> int:
    return nvbios_init.NvbiosInit(dev, image)._ramcfg_index()

  def run_ram_init(dev: FakeMMIO, img: bytes) -> None:
    nvbios_init.run_vbios_ram_init(dev, img, debug=False)

  def run_ram_program(dev: FakeMMIO, img: bytes) -> dict:
    return nvbios_init.run_vbios_ram_program(dev, img, freq_mhz=648, debug=False)

  def run_cold(dev: FakeMMIO, img: bytes) -> None:
    # Match live cold path: RAMMAP/train → program → fb → LTC.
    nvbios_init.run_vbios_ram_init(dev, img, debug=False)
    nvbios_init.run_vbios_ram_program(dev, img, freq_mhz=648, debug=False)
    add_mod._gk104_post_ram_fb_ltc(dev)

  return KeplerMMIOHooks(
      vbios_path=path,
      find_image=nvbios_init.find_vbios_image,
      read_ramcfg=read_ramcfg,
      run_ram_init=run_ram_init,
      run_ram_program=run_ram_program,
      fb_init_page=add_mod._gk104_fb_init_page,
      ltc_init=add_mod._gk104_ltc_init,
      run_cold_ram_fb_ltc=run_cold,
      post_ram_fb_ltc=add_mod._gk104_post_ram_fb_ltc,
      topo_is_posted=add_mod._gk104_topo_is_posted,
      pcie_add_source=pathlib.Path(add_mod.__file__).read_text(encoding="utf-8"),
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  argv = list(argv if argv is not None else sys.argv[1:])
  # Prefer importing the Linux-owned bring-up module.
  here = pathlib.Path(__file__).resolve().parent
  pcie = here.parent / "examples_kepler_pcie"
  sys.path.insert(0, str(pcie))
  sys.path.insert(0, str(here))
  import add as add_mod  # type: ignore
  hooks = build_hooks_from_add_module(add_mod)
  return run_mmiotrace_selftest(hooks)


if __name__ == "__main__":
  sys.exit(main())
