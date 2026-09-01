#!/usr/bin/env python3
"""Vendored NV stack for add/multiply — single RTX 3080 eGPU over PCIe/USB4 or USB3.

Vendored from ref/tinygrad/tinygrad/runtime/support/{nv,system,hcq,memory,elf}.py
and ref/tinygrad/tinygrad/runtime/ops_nv.py (slimmed to the PCIe path only).

PCIe/GSP tracing is vendored inline and remains disabled unless NV_TRACE or
NV_TRACE_TLP requests instrumentation.

NO imports from tinygrad.runtime.support, tinygrad.runtime.ops, tinygrad.device,
tinygrad.renderer, tinygrad.uop, tinygrad.helpers are permitted in this module —
those have been vendored inline below.
"""
from __future__ import annotations
import os, sys, ctypes, ctypes.util, time, mmap, struct, array as _array_mod, socket, subprocess, contextlib, functools, itertools, enum, atexit, select, dataclasses, collections, threading, urllib.request, hashlib, tempfile, gzip, pathlib, types, importlib
from typing import cast, Any, ClassVar, Generic, TypeVar
from dataclasses import dataclass, replace

# --- repository-local autogen ctypes ("ctypes constants only") ---
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from autogen import nv, nv_570 as nv_gpu, pci
from autogen import libc, libusb

import traceback

# ============================================================================
# Helpers (slimmed from tinygrad/helpers.py — only what we actually use)
# ============================================================================
DEBUG = int(os.environ.get("DEBUG", "0"))
def getenv(k: str, default=0):
  v = os.environ.get(k)
  if v is None: return default
  try: return int(v)
  except: return v
TRACE_RAW = getenv("NV_TRACE_RAW", getenv("TRACE_RAW", 0))   # verbose raw CFG/MMIO lines (default: semantic-only)
def getbits(value: int, start: int, end: int) -> int: return (value >> start) & ((1 << (end - start + 1)) - 1)
def i2u(dtype: int, val: int) -> int: return val & ((1 << (dtype * 8)) - 1)
def round_up(num: int, amt: int) -> int: return ((num + amt - 1) // amt) * amt
def round_down(num: int, amt: int) -> int: return -round_up(-num, amt)
def ceildiv(num: int, amt: int) -> int: return -(num // -amt)
def lo32(x: int) -> int: return x & 0xFFFFFFFF
def hi32(x: int) -> int: return x >> 32
def data64(x: int) -> tuple: return ((x >> 32) & 0xFFFFFFFF, x & 0xFFFFFFFF)  # (hi, lo) — matches tinygrad helpers.data64
def data64_le(x: int) -> tuple: return (x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF)  # (lo, hi) — matches tinygrad helpers.data64_le
def unwrap(x): return x

def nv_flags(reg, **kwargs) -> int:
  return functools.reduce(int.__or__, ((getattr(nv_gpu, f"{reg}_{k}_{v}".upper()) if isinstance(v, str) else v) <<
    getattr(nv_gpu, f"{reg}_{k}".upper())[1] for k, v in kwargs.items()), 0)
OSX = sys.platform == "darwin"

# Tinygrad uses this from `array.array` but our stubs above use the stdlib module.
array = _array_mod

def to_mv(ptr: int, sz: int) -> memoryview: return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")
def mv_address(mv) -> int: return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def from_mv(mv: memoryview, to_type=ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents

def wait_cond(cb, *args, value=True, timeout_ms=10000, msg=""):
  global IO_TAG, LAST_WAIT_SAMPLES
  LAST_WAIT_SAMPLES = {}
  verbose = TRACE and not quiet_active()
  if TRACE: IO_TAG = "WAIT"   # aggregation tallies inside here label as WAIT_IO
  if verbose:
    _trace("WAIT", f"{msg or 'waiting for condition'} (timeout {timeout_ms}ms)")
    wait_sampling_begin()
  t0 = time.perf_counter()
  iters = 0
  with _tscope():
    start_time = int(time.perf_counter() * 1000)
    while int(time.perf_counter() * 1000) - start_time < timeout_ms:
      iters += 1
      if iters % 64 == 0: time.sleep(0.00005)   # gentle backoff: don't flood the falcon PRI
      if (val := cb(*args)) == value:
        if verbose:
          LAST_WAIT_SAMPLES = wait_sampling_end()
          trajs = fmt_trajectories(LAST_WAIT_SAMPLES, ok=True)
          msg = f"MET after {iters} polls / {(time.perf_counter() - t0) * 1e3:.1f}ms"
          if trajs: msg += "\n      " + trajs.replace("\n", "\n      ")
          _trace("WAIT", msg)
        IO_TAG = ""
        return val
    samples = wait_sampling_end() if TRACE else {}
    if TRACE:
      trajs = fmt_trajectories(samples)
      msg = f"TIMEOUT after {iters} polls / {(time.perf_counter() - t0) * 1e3:.1f}ms"
      if trajs: msg += "\n      " + trajs.replace("\n", "\n      ")
      _trace("WAIT", msg)
    IO_TAG = ""
    raise TimeoutError(f"{msg}. Timed out after {timeout_ms} ms, condition not met: {val} != {value}")

def _ensure_downloads_dir() -> pathlib.Path:
  d = pathlib.Path(os.path.expanduser("~")) / ".cache" / "tinygrad"
  d.mkdir(parents=True, exist_ok=True)
  return d

def temp(name: str) -> str:
  return os.path.join(tempfile.gettempdir(), name)

def fetch_fw(path: str, name: str, sha256: str) -> bytes:
  cache_dir = _ensure_downloads_dir() / "fw"
  cache_dir.mkdir(parents=True, exist_ok=True)
  fp = cache_dir / name
  if fp.is_file() and hashlib.sha256(fp.read_bytes()).hexdigest() == sha256:
    return fp.read_bytes()
  url = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/1e2c15348485939baf1b6d1f5a7a3b799d80703d/{path}/{name}"
  with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "middle_nv"}), timeout=10) as r:
    data = r.read()
  if hashlib.sha256(data).hexdigest() != sha256:
    raise RuntimeError(f"fetch_fw sha mismatch for {name}")
  fp.write_bytes(data)
  return data

def pluralize(n, s, p=None):
  if p is None: p = s + "s"
  return f"{n} {p}" if n != 1 else f"1 {s}"


# ============================================================================
# PCIe/GSP tracing (vendored inline) — the "usbmon for PCIe" tap.
#   NV_TRACE=1  semantic lifecycle, transport, GSP RPC, and USB stream detail
#   NV_TRACE=2  + named register/bitfield decode
# Blind spot: GPU-initiated DMA writes into sysmem never cross this path.
# ============================================================================
TRACE = int(os.environ.get("NV_TRACE", "1") or 1)
TRACE_MAXDUMP = int(os.environ.get("NV_TRACE_MAXDUMP", "512") or 512)
TRACE_NOZERO = int(os.environ.get("NV_TRACE_NOZERO", "1") or 0)
USB_TRACE = int(os.environ.get("NV_USB_TRACE", "2" if TRACE else "0") or 0)

# Replayable PCIe-transaction stream: NV_TRACE_TLP=<path> writes one numbered
# line per bus transaction (CFG/BAR0/BAR1), independent of NV_TRACE. Payloads
# <=8B are inlined little-endian hex; larger blobs are sha256-stamped so a
# USB3 implementation can diff/replay against this known-good capture.
TLP_PATH = os.environ.get("NV_TRACE_TLP", "")
_tlp_fp = None

def _fmt_data(b: bytes) -> str:
  import hashlib
  if len(b) == 0: return "-"
  if len(b) <= 8: return f"0x{int.from_bytes(b, 'little'):0{len(b) * 2}x}"
  return f"blob{len(b):#x}@{hashlib.sha256(b).hexdigest()[:16]}"

def tlp(line: str):
  global _tlp_fp
  if not TLP_PATH: return
  try:
    if _tlp_fp is None: _tlp_fp = open(TLP_PATH, "w")
  except OSError:
    return
  with _trace_lock:
    n = event_next()
    _tlp_fp.write(f"{n:06d} {line}\n")
    if n % 64 == 0: _tlp_fp.flush()

_trace_lock = threading.Lock()
_trace_reg_cache: dict = {}
_trace_reg_obj: dict = {}
_trace_last_reg_rd: dict = {}
_trace_last_wire_rd: dict = {}
_trace_last_wire_wr: dict = {}
_trace_regdevs: list = []

# Driver-side types bound by add.py (see bind()).
_NVReg = None
_RemoteCmd = None
_METHOD_NAMES: dict = {}

def bind(*, NVReg=None, RemoteCmd=None, method_names=None):
  global _NVReg, _RemoteCmd, _METHOD_NAMES
  if NVReg is not None: _NVReg = NVReg
  if RemoteCmd is not None: _RemoteCmd = RemoteCmd
  if method_names is not None: _METHOD_NAMES = dict(method_names)

def _trace_decode_vbios(data: bytes) -> str | None:
  # Large BAR0 ROM reads in prep_ucode are the card's VBIOS — summarize it.
  if len(data) < 0x200 or data[:2] != b"\x55\xaa": return None
  try:
    ver = ""
    if (i := data.find(b"Version ")) >= 0:
      end = next((j for j in (data.find(b"\r", i), data.find(b"\n", i), data.find(b"\0", i)) if j > 0), i + 40)
      ver = data[i + 8:min(end, i + 40)].decode('ascii', 'replace').strip()
    prod = ""
    import re
    runs = [m.group().decode('ascii') for m in re.finditer(rb"[\x20-\x7e]{12,}", data[0x40:0x100])]
    prod = next((s for s in runs if "RTX" in s or "GTX" in s or "GPU" in s), runs[0] if runs else "?")
    ids = ""
    if (j := data.find(b"PCIR")) >= 0:
      ven, dev = int.from_bytes(data[j + 4:j + 6], 'little'), int.from_bytes(data[j + 6:j + 8], 'little')
      ids = f" pci={ven:04x}:{dev:04x}"
    return f"VBIOS: ver={ver or '?'} board=\"{prod.strip()}\"{ids}"
  except Exception: return None

def _trace_decode_pte(data: bytes) -> str | None:
  # Decode a GMMU PTE written by NVPageTableEntry.set_entry (8-byte BAR1 write).
  for d in _trace_regdevs:
    if (t := getattr(d, "pte_t", None)) is None: continue
    try: f = t.decode(struct.unpack('<Q', data)[0])
    except Exception: return None
    apn = {0: "VRAM", 1: "PEER", 2: "SYSMEM"}[f.get("aperture", 0)]
    return f"PTE -> pa={(f.get('address_sys', 0) << 12):#x} aperture={apn} kind={f.get('kind')} valid={f.get('valid')} vol={f.get('vol')}"
  return None

_CMD_NAMES = None

# Frames to walk past when attributing a trace line to the driver-level
# function that caused it (transport/reg plumbing, poll helpers, lambdas).
_TRACE_PLUMBING = frozenset({"_trace", "_trace_caller", "_trace_hexdump", "_trace_cmd_name", "_trace_struct_fields",
  "_trace_rpc_func", "_rpc", "_recvall", "_bulk_read", "_bulk_write", "wreg", "rreg", "_resolve_reg_name",
  "send_rpc", "_send_rpc_record", "read_resp", "wait_resp", "wait_cond", "<lambda>",
  "__getitem__", "__setitem__", "view", "read", "write", "update",
  "<genexpr>", "<listcomp>", "<dictcomp>", "tprint", "read_bitfields",
  "usb_trace", "usb_retrain_report", "usb_stream_report"})

def _trace_caller() -> str:
  fr = sys._getframe(1)
  while fr is not None:
    nm = fr.f_code.co_name
    if nm in ("_run_seq_ops", "flush_dma"): return "run_cpu_seq"   # sequencer helper: keep parent label
    if nm not in _TRACE_PLUMBING:
      # Qualify with the owning class so methods are unambiguous
      # (e.g. NV_GSP.init_gsp_image vs some other init_*).
      slf = fr.f_locals.get("self")
      return f"{type(slf).__name__}.{nm}" if slf is not None and not isinstance(slf, types.ModuleType) else nm
    fr = fr.f_back
  return "?"

def _trace_cmd_name(cmd):
  global _CMD_NAMES
  if _CMD_NAMES is None:
    assert _RemoteCmd is not None, "bind(RemoteCmd=...) not called"
    _CMD_NAMES = {int(c): c.name for c in _RemoteCmd}
  return _CMD_NAMES.get(cmd, f"CMD?{cmd:#x}")

_tls = threading.local()
def _tdepth() -> int: return getattr(_tls, "d", 0)

# Poll sampling: while a wait_cond runs, record every changing register value
# so its completion can summarize the firmware-progress trajectory.
def wait_sampling_begin():
  setattr(_tls, "ps", {})

def wait_sampling_end() -> dict:
  ps = getattr(_tls, "ps", None) or {}
  setattr(_tls, "ps", None)
  return ps

def poll_sample(addr: int, val: int):
  ps = getattr(_tls, "ps", None)
  if ps is None: return
  seq = ps.setdefault(addr, [])
  if not seq or seq[-1] != val: seq.append(val)

# Short labels for polled registers whose low byte IS the condition:
# 0x118234 = NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0] (GFW boot progress).
_WAIT_SHORT_LABELS = {0x118234: "GFW"}
_FALCON_BASES = (0x110000, 0x840000)
for _fb in _FALCON_BASES: _WAIT_SHORT_LABELS[_fb + 0x100] = "CPUCTL"

def fmt_trajectories(samples: dict, max_vals: int = 8, ok: bool = False) -> str:
  # Register-value trajectories seen during a poll. When a register has a
  # known short label and its low byte is the meaningful part, lead with the
  # decoded byte progression ("GFW: 01 -> 02 -> ff") followed by the full
  # register values ("REG: 0101 -> 0302 -> 03ff") so the satisfaction of the
  # condition is visible. Constant-poll registers are omitted.
  out = []
  for addr, vals in samples.items():
    if len(set(vals)) < 2: continue
    shown, more = vals[:max_vals], len(vals) > max_vals
    ell = " -> ..." if more else ""
    lbl = _WAIT_SHORT_LABELS.get(addr)
    if lbl == "CPUCTL":
      # HALTED is bit 4: show full values plus the bit transition that satisfied the wait.
      reg = " -> ".join(f"0x{v:08x}" for v in shown) + ell
      first, last = vals[0], vals[-1]
      out.append(f"CPUCTL: {reg} | HALTED {(first >> 4) & 1} -> {(last >> 4) & 1}" + (" ✓" if ok and (last >> 4) & 1 else ""))
    elif lbl:
      lo = " -> ".join(f"{v & 0xff:02x}" for v in shown) + ell + (" ✓" if ok and not more else "")
      hi_bits = any(v & ~0xff for v in vals)
      if hi_bits:
        reg = " -> ".join(f"{v:04x}" for v in shown) + ell
        out.append(f"{lbl}: {lo}\nREG: {reg}")
      else:
        out.append(f"{lbl}: {lo}")
    else:
      s = " -> ".join(f"0x{v:x}" for v in shown) + ell
      out.append(f"{addr:#x}: {s}")
  return "\n".join(out)

def _stage_pre() -> str:
  return "  " * _tdepth()

# ---------------------------------------------------------------------------
# Stage model: 10 fixed lifecycle stages (S1..S10), sub-stages to depth 3 max
# (S4.1.1). Raw transactions are events (E######), never stages.
# ---------------------------------------------------------------------------
STAGES = ["PCIe/Transport", "GPU Discovery+MMU", "Firmware Prep", "Secure FW Boot",
          "GSP/RM Online", "Golden GR Context", "User Channels+GPFIFO",
          "Workload+QMD Prep", "Submit+Execute", "Result Validation"]
_stage_status: dict = {}      # n -> {"ok": bool, "note": str}
_max_top_reached = 0

def _excepthook(t, v, tb):
  # A crash inside stage Sn marks that stage FAIL with the exception summary.
  n = getattr(_tls, "stg", 0)
  if n:
    _stage_status.setdefault(n, {"ok": False, "note": ""})
    _stage_status[n].update(ok=False, failed=True, note=f"{t.__name__}: {v}")
  sys.__excepthook__(t, v, tb)
sys.excepthook = _excepthook

def stage_set(n: int, note: str = "") -> str:
  """Enter top-level stage Sn. Announces inside are stable subsections:
  Sn.1, Sn.2 ... (monotonic, never reused); scopes under an announce number
  its children Sn.k.1, Sn.k.2 ..."""
  assert 1 <= n <= len(STAGES)
  global _max_top_reached
  if TRACE and getattr(_tls, "stg", 0) != 0: print(flush=True)   # section break
  _max_top_reached = max(_max_top_reached, n)
  setattr(_tls, "stg", n)
  setattr(_tls, "ann", 0)
  setattr(_tls, "chld", 0)
  setattr(_tls, "just", True)
  setattr(_tls, "smode", [])
  _REQ_CACHE.clear()
  _stage_status.setdefault(n, {"ok": False, "note": ""})
  if note: _stage_status[n]["note"] = note
  return f"S{n}"

def stage_done(note: str = ""):
  """Mark the current top-level stage OK (with optional result note)."""
  n = getattr(_tls, "stg", 0)
  _stage_status.setdefault(n, {"ok": False, "note": ""})
  _stage_status[n]["ok"] = True
  if note: _stage_status[n]["note"] = note

def stage_now() -> str:
  n = getattr(_tls, "stg", 0)
  if not n: return ""
  a, c = getattr(_tls, "ann", 0), getattr(_tls, "chld", 0)
  smode = getattr(_tls, "smode", [])
  if a == 0:   # header announce not yet made / none exists: children act as subs
    return f"S{n}" + (f".{c}" if c and smode else "")
  lbl = f"S{n}.{a}"
  if c and smode and smode[-1] == "child": lbl += f".{c}"
  return lbl

def rm_tree_lines() -> list[str]:
  if not _RM_OBJS: return []
  kids: dict = {}
  for h, o in _RM_OBJS.items():
    kids.setdefault(o["parent"], []).append(h)
  def cls(h):
    return _RM_OBJS[h]["cls"]
  lines = [f"{h:08x} {cls(h)}" for h in sorted(kids.get(0, []))]
  def walk(h, depth):
    out = []
    for ch in sorted(kids.get(h, [])):
        out.append("   " * (depth + 1) + "\u2514\u2500 " + f"{ch:08x} {cls(ch)}")
        out.extend(walk(ch, depth + 1))
    return out
  for root in sorted(kids.get(0, [])):
    lines += walk(root, 0)
  return lines

def stage_summary():
  if not TRACE: return
  print("\n" + "=" * 62, flush=True)
  print(f"{'NVIDIA GPU RUN':^62}", flush=True)
  print("=" * 62, flush=True)
  for i, name in enumerate(STAGES, 1):
    st = _stage_status.get(i)
    if st and st.get("ok"): mark, extra = "OK  ", f"  {st['note']}" if st["note"] else ""
    elif st and st.get("failed"): mark, extra = "FAIL", f"  {st['note']}" if st["note"] else ""
    elif st and st.get("na"): mark, extra = "N/A ", f"  {st['note']}" if st["note"] else ""
    elif st and i == _max_top_reached: mark, extra = "....", "  in progress at exit"
    elif st: mark, extra = "....", ""
    else: mark, extra = "----", ""
    print(f"[S{i}] {name:<28} {mark}{extra}", flush=True)
  print("=" * 62, flush=True)
  tree = rm_tree_lines()
  if tree:
    print("RM object tree:", flush=True)
    for ln in tree: print("  " + ln, flush=True)
    print("=" * 62, flush=True)

import atexit as _atexit
_atexit.register(stage_summary)

def _stage_counters() -> list:
  # Legacy sibling counters retained for non-stage nesting indentation.
  c = getattr(_tls, "cnt", None)
  if c is None: _tls.cnt = c = [0]
  return c

@contextlib.contextmanager
def _tscope():
  # Nesting scope: indents output, extends the stage path one level
  # (up to three label levels: S4 -> S4.1 -> S4.1.1; deeper is indent only),
  # and buffers wire facts for the Level-2 aggregation flush on exit.
  setattr(_tls, "d", _tdepth() + 1)
  smode = getattr(_tls, "smode", [])
  mode = "grand" if smode else "child"   # scope under an announce = its children
  smode.append(mode)
  c = _stage_counters(); c.append(0)
  ev: list | None = []
  setattr(_tls, "ev", ev)
  setattr(_tls, "rpc", {})
  try: yield
  finally:
    _tls.ev = None
    _agg_flush(ev, _stage_pre())
    _rpc_flush(getattr(_tls, "rpc", {}), _stage_pre())
    setattr(_tls, "rpc", {})
    setattr(_tls, "chld", 0)   # leaving scope: children of that announce are done
    smode.pop()
    c.pop(); setattr(_tls, "d", _tdepth() - 1)

def set_wire_suppress(on: bool):
  """Toggle Level-1/2 wire emission for the CURRENT THREAD without touching
  the TLP sidecar. Used by detectors that summarize a whole loop (DMA_RUN):
  constituents vanish, the summary speaks for them."""
  cur = getattr(_tls, "q", 0)
  setattr(_tls, "q", cur + 1 if on else max(cur - 1, 0))

def quiet_active() -> bool:
  # Inside quiet(): repetitive loops whose semantics are summarized by the
  # caller — suppress per-event printing/buffering (TLP sidecar stays full).
  return getattr(_tls, "q", 0) > 0

@contextlib.contextmanager
def quiet():
  setattr(_tls, "q", getattr(_tls, "q", 0) + 1)
  try: yield
  finally: setattr(_tls, "q", getattr(_tls, "q", 0) - 1)

_REQ_CACHE: dict = {}          # func_id -> request payload bytes (cleared on stage_set)
_RM_OBJS: dict = {}            # hObject -> {"cls": str, "parent": int} (RM object tree)
_RPC_STATUS_SYMS: dict | None = None

def rpc_status_sym(code: int) -> str:
  global _RPC_STATUS_SYMS
  if _RPC_STATUS_SYMS is None:
    d = {}
    for k, v in nv.__dict__.items():
      if isinstance(v, int) and (k.startswith("NV_ERR_") or k == "NV_OK"): d.setdefault(v, k)
    _RPC_STATUS_SYMS = d
  return _RPC_STATUS_SYMS.get(code, f"{code:#x}")

def rpc_remember_request(func: int, data: bytes):
  _REQ_CACHE[func] = bytes(data)

def _int_fields(t, data: bytes) -> dict:
  try: st = t.from_buffer_copy(data[:ctypes.sizeof(t)])
  except Exception: return {}
  out = {}
  for f in getattr(t, "_real_fields_", None) or getattr(t, "_fields_", []):
    if isinstance(f, str): continue
    v = getattr(st, f[0])
    if isinstance(v, int): out[f[0]] = v
  return out

def has_cached_request(func: int) -> bool:
  return func in _REQ_CACHE

def rpc_delta_lines(func: int, resp: bytes) -> list[str]:
  """Field-level diff of an RPC response against its cached request payload."""
  _trace_structs_init()
  t = _RPC_STRUCTS.get(func)
  req = _REQ_CACHE.get(func)
  if t is None or req is None: return []
  rq, rs = _int_fields(t, req), _int_fields(t, resp)
  out = []
  for k, rv in rs.items():
    ov = rq.get(k)
    if ov != rv and not (ov is None and rv == 0):
      out.append(f"{k}: {ov if ov is not None else '-'} -> {rv:#x}" if isinstance(rv, int) else k)
  return out

def rpc_note(key: str, detail: str) -> bool:
  """Repeat-collapsible record (e.g. NOCAT): True only the first time; the
  latest detail is kept so the scope-flush can print 'key xN: last-detail'."""
  rpc = getattr(_tls, "rpc", None)
  if rpc is None: return True
  ent = rpc.get(key)
  if ent is None:
    rpc[key] = [1, detail]; return True
  ent[0] += 1; ent[1] = detail
  return False

def stage_na(n: int, note: str = ""):
  """Mark stage Sn as not-applicable (handled elsewhere / unsupported)."""
  _stage_status[n] = {"ok": False, "na": True, "note": note}

def rpc_first(key: str) -> bool:
  """Inside a scope: True only for the FIRST occurrence of this RPC pattern;
  later identical ones are counted and summarized at scope exit."""
  rpc = getattr(_tls, "rpc", None)
  if rpc is None: return True
  n = rpc.get(key, 0)
  rpc[key] = n + 1
  return n == 0

def _rpc_flush(rpc: dict, pre):
  if not TRACE or not rpc: return
  stg = stage_now()
  total = sum(v[0] if isinstance(v, list) else v for v in rpc.values())
  def cnt(k):
    v = rpc[k]
    return v[0] if isinstance(v, list) else v
  items = ", ".join(f"{k} x{cnt(k)}" for k in sorted(rpc, key=lambda k: -cnt(k))[:8])
  if len(rpc) > 8: items += f", +{len(rpc) - 8} more"
  print(f"{pre}[{stg}] RPC traffic ({total}): {items}", flush=True)
  for k, v in rpc.items():
    if isinstance(v, list) and v[0] > 1 and v[1]:
      print(f"{pre}[{stg}]   {k} x{v[0]}: {v[1][:100]}", flush=True)

# Event ids: every wire transaction gets one; the TLP sidecar shares the counter
# so aggregated spans can reference exact raw ranges.
_ev_n = 0
def event_next() -> int:
  global _ev_n
  _ev_n += 1
  return _ev_n

# Per-scope wire-event aggregation (Level 2): compresses repetitive MMIO into
# (address,value) spans; Level 1 exact stream remains in the TLP sidecar.
def agg_active() -> bool:
  return getattr(_tls, "ev", None) is not None

def wire_ev(kind: str, bar: int, off: int, size: int, data: bytes):
  ev = getattr(_tls, "ev", None)
  if ev is not None: ev.append((kind, bar, off, size, bytes(data), event_next()))

# Falcon/boot overlay: parameterized over both engine bases (PGSP falcon
# @0x110000, SEC2 @0x840000) plus absolute boot registers. Curated because
# this MMIO range is block-contextual and the generic table misaliases it.
def _dmatrfcmd(v: int) -> str:
  parts = []
  if v & 1: parts.append("full")
  if v >> 1 & 1: parts.append("idle")
  if v >> 2 & 3 == 3: parts.append("sec")
  if v >> 4 & 1: parts.append("imem")
  if v >> 5 & 1: parts.append("write")
  sz = v >> 8 & 7
  if sz == 6: parts.append("256B")
  elif sz: parts.append(f"size={sz}")
  return " ".join(parts) or f"{v:#x}"

def _pmc_boot_0(v: int) -> str:
  """NV_PMC_BOOT_0 meaning only (bytes/LE rendered by falcon_boot_row)."""
  return "chip_id"

def _pmc_boot_42(v: int) -> str:
  """NV_PMC_BOOT_42 field decode (bytes/LE rendered by falcon_boot_row)."""
  flds = {"architecture": (24, 29), "implementation": (20, 23),
    "major_revision": (16, 19), "minor_revision": (12, 15), "minor_extended_revision": (8, 11)}
  g = {k: (v >> lo) & ((1 << (hi - lo + 1)) - 1) for k, (lo, hi) in flds.items()}
  soc = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}.get(g["architecture"], "?")
  return (f"arch[{flds['architecture'][1]}:{flds['architecture'][0]}]={g['architecture']:#x} "
          f"impl[{flds['implementation'][1]}:{flds['implementation'][0]}]={g['implementation']:#x} "
          f"-> {soc}{g['implementation']:02d}, rev {g['major_revision']}.{g['minor_revision']}.{g['minor_extended_revision']}")

_FALCON_OFFS = {
  0x040: ("FALCON_MAILBOX0", None),
  0x044: ("FALCON_MAILBOX1", None),
  0x080: ("FALCON_OS", lambda v: "fw version/status cleared" if v == 0 else f"fw={v:#x}"),
  0x084: ("FALCON_RM", lambda v: f"chip_id={v:#x}"),
  0x0f4: ("HWCFG2", lambda v: f"{'RISCV ' if v >> 10 & 1 else ''}scrub={'done' if not (v >> 12 & 1) else 'busy'} reset_ready={v >> 31 & 1}"),
  0x100: ("CPUCTL", lambda v: {1: "startcpu", 2: "start via ALIAS"}.get(v, f"flags={v:#x}" + (" HALTED" if v >> 4 & 1 else ""))),
  0x1388: ("RISCV_CPUCTL", lambda v: f"ACTIVE_STAT={v >> 7 & 1} HALTED={v & 1}" +
           (" -> GSP RISC-V running ✓" if v >> 7 & 1 and not v & 1 else "")),
  0x104: ("BOOTVEC", lambda v: f"imem entry={v:#x}"),
  0x10c: ("DMACTL", lambda v: "disabled" if v == 0 else f"{v:#x}"),
  0x110: ("DMATRFBASE", lambda v: f"fb base={(v & 0x1ff) << 8:#x}"),
  0x114: ("DMATRFMOFFS", None),
  0x118: ("DMATRFCMD", _dmatrfcmd),
  0x11c: ("DMATRFFBOFFS", None),
  0x3c0: ("FALCON_ENGINE", lambda v: "reset=1" if v & 1 else "reset released"),
  0x624: ("FBIF_CTL", lambda v: f"ALLOW_PHYS_NO_CTX={v >> 7 & 1}" + (" ✓" if v >> 7 & 1 else "")),
  0x1668: ("RISCV_BCR_CTRL", lambda v: f"core_sel={v >> 4 & 1} valid={v & 1} brfetch={v >> 8 & 1}"),
  0x1180: ("FALCON_MOD_SEL", lambda v: "RSA3K" if v == 1 else f"algo={v:#x}"),
  0x1198: ("BROM_CURR_UCODE_ID", lambda v: f"ucodeid={v}"),
  0x119c: ("BROM_ENGIDMASK", lambda v: f"engine_mask={v:#x}"),
  0x1210: ("BROM_PARAADDR", lambda v: f"pkc_off={v:#x}"),
}
_FALCON_BOOT_REGS = {}
for _eb in (0x110000, 0x840000):
  for _off, (_nm, _fn) in _FALCON_OFFS.items():
    _FALCON_BOOT_REGS[_eb + _off] = (_nm, _fn)
_FALCON_BOOT_REGS.update({
  # The two registers that make up NV_FLCN.wait_for_reset()'s condition.
  # 0x118128 = PGC6_AON SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK (nouveau checks bit0)
  # 0x118234 = PGC6_AON SECURE_SCRATCH_GROUP_05[0]: GFW boot progress; NVIDIA's
  #            falcon DMA test calls this the boot-complete reg (mask 0x3ff).
  0x118128: ("AON_SCRATCH05_PRI_LEVEL_MASK",
             lambda v: f"PRI unlocked bit[0]={v & 1}" + (" ✓" if v & 1 else " (locked)")),
  0x118234: ("GFW_BOOT_PROGRESS",
             lambda v: (f"BOOT complete, progress[7:0]=0x{v & 0xff:02x} ✓" if v & 0xff == 0xff
                        else f"in progress, {v & 0xff:02x} -> ff") + (f" status={v >> 8:#x}" if v >> 8 else "")),
  # PGC6_AON SECURE_SCRATCH_GROUP_42: usable framebuffer size in MiB
  # (nova-core names it NV_USABLE_FB_SIZE_IN_MB; nouveau: vidmem_size = v << 20).
  # NOT an address — a size expressed directly in MiB.
  0x1183a4: ("NV_USABLE_FB_SIZE_IN_MB",
             lambda v: f"{v:#x} MiB -> usable VRAM {v >> 10} GiB ({v << 20:#x} bytes)"),
  0x110c00: ("PGSP_QUEUE_HEAD doorbell", lambda v: "kick GSP RPC queue"),
  0x1fa828: ("WPR2_ADDR_HI", lambda v: f"this IS an address (not data) = exclusive TOP of the WPR2 secure fence in VRAM; "
             f"computed={((v & ~0xF) << 8):#x}; programmed by firmware during secure boot and persists across driver reloads; "
             f"nonzero => prior boot still locked -> cold reset required"),
  0x1fa824: ("WPR2_ADDR_LO", lambda v: f"lower={(v & ~0xF) << 8:#x}"),
  0x1704:   ("PBUS_VBIOS_SCRATCH[193]", lambda v: f"fmc_boot={'set' if v else 'clear'}"),
  0xa00:    ("NV_PMC_BOOT_42 (chip identity)", _pmc_boot_42),
  0x0:      ("NV_PMC_BOOT_0 (chip id)", _pmc_boot_0),
})

def hsz(n: int) -> str:
  if n >= (1 << 20): return f" ({n / (1 << 20):.1f} MiB)"
  if n >= (1 << 10): return f" ({n / (1 << 10):.1f} KiB)"
  return ""

def falcon_boot_ann(bar: int, off: int, val: int | bytes | None) -> str | None:
  if bar != 0 or off not in _FALCON_BOOT_REGS: return None
  nm, vf = _FALCON_BOOT_REGS[off]
  if vf is None or val is None: return f"[{nm}]"
  if isinstance(val, bytes): val = int.from_bytes(val[:4].ljust(4, b"\0"), "little")
  hint = vf(val)
  return f"[{nm}: {hint}]"

def falcon_boot_row(bar: int, off: int, val: bytes) -> tuple[str, str] | None:
  """Side-by-side decode for known 4-byte BAR0 reads: (name, "bytes | LE | meaning")."""
  if bar != 0 or off not in _FALCON_BOOT_REGS or not isinstance(val, (bytes, bytearray)): return None
  nm, vf = _FALCON_BOOT_REGS[off]
  b = bytes(val[:4])
  v = int.from_bytes(b.ljust(4, b"\0"), "little")
  hint = vf(v) if vf else ""
  hint = " ".join(hint.split("\n"))          # flatten multi-line decodes onto the row
  return nm, f"{b.hex(' ')} | 0x{v:08x}" + (f" | {hint}" if hint else "")

_MBX0_VAL = None   # last FALCON_MAILBOX0 write, for 64-bit mailbox pairing

def falcon_boot_write_row(bar: int, off: int, val: bytes) -> tuple[str, str] | None:
  """Write variant: same row as falcon_boot_row, plus changed-bit transitions
  vs the most recent READ of the same address (read-modify-write visibility),
  e.g. "... | ALLOW_PHYS_NO_CTX=1 | bit7 0->1"."""
  global _MBX0_VAL
  r = falcon_boot_row(bar, off, val)
  if r is None: return None
  nm, ln = r
  v = int.from_bytes(bytes(val[:4]).ljust(4, b"\0"), "little")
  if nm == "FALCON_MAILBOX0":
    _MBX0_VAL = v
    return nm, f"{bytes(val[:4]).hex(' ')} | LO=0x{v:08x}"
  if nm == "FALCON_MAILBOX1":
    lo = _MBX0_VAL
    mb64 = ((v << 32) | lo) if lo is not None else None
    ln = f"{bytes(val[:4]).hex(' ')} | HI=0x{v:08x}" + (f" | mailbox64={mb64:#x} (boot args pointer)" if mb64 else "")
    return nm, ln
  prev = _trace_last_wire_rd.get((bar, off))
  if prev is not None:
    p = int.from_bytes(prev[:4].ljust(4, b"\0"), "little")
    if p != v:
      bits = " ".join(f"bit{i} {(p >> i) & 1}->{(v >> i) & 1}" for i in range(32) if (p ^ v) >> i & 1)
      ln += f" | {bits}" if len(bits) <= 40 else f" | {p:#x} -> {v:#x}"
  return nm, ln

IO_TAG = ""   # scope for aggregation tallies: "WAIT" while a wait_cond polls, else generic IO

def _agg_flush(ev, pre):
  if not TRACE or not ev: return
  from collections import OrderedDict
  facts = [(bar, off, size, data, eid) for kind, bar, off, size, data, eid in ev if kind != "RD"]
  nreads = len(ev) - len(facts)
  def val8(d): return int.from_bytes(d[:8].ljust(8, b"\0"), "little")

  # --- strided-run detection: consecutive write *windows* where every lane
  # (offset within the window) advances by a constant value stride collapse
  # into ONE span. Handles plain PTE fills (period 1) and interleaved patterns
  # like dual-PDE [entry, companion] stepping huge pages (period 2).
  def val8(d): return int.from_bytes(d[:8].ljust(8, b"\0"), "little")
  runs, covered, i, n = [], set(), 0, len(facts)
  while i < n:
    bar, off, size, data, eid = facts[i]
    matched = False
    for period in (1, 2, 3, 4):
      min_win = 3
      if i + period * min_win > n: continue
      win = [facts[i + w * period: i + (w + 1) * period] for w in range(min_win)]
      if any(len(w) != period or w[0][0] != bar or w[0][2] != size for w in win): continue
      doff = win[1][0][1] - win[0][0][1]
      if doff <= 0 or win[2][0][1] - win[1][0][1] != doff: continue
      lanes, ok = [], True
      for l in range(period):
        vs = [val8(win[w][l][3]) for w in range(min_win)]
        dv = vs[1] - vs[0]
        if vs[2] - vs[1] != dv: ok = False; break
        lanes.append([win[0][l][1], vs[0], dv])
      if not ok: continue
      j = i + period * min_win
      while j + period <= n:
        nxt = facts[j: j + period]
        if any(x[0] != bar or x[2] != size for x in nxt): break
        if nxt[0][1] - facts[j - period][1] != doff: break
        bad = False
        for l in range(period):
          if val8(nxt[l][3]) - val8(facts[j - period + l][3]) != lanes[l][2]: bad = True; break
        if bad: break
        j += period
      cnt = (j - i) // period
      lanes_out = [(lo, v0, dv) for lo, v0, dv in lanes]
      eids = [facts[k][4] for k in range(i, j)]
      runs.append((bar, off, facts[j - 1][1], cnt, period, doff, lanes_out, min(eids), max(eids)))
      covered.update(range(i, j)); i = j; matched = True
      break
    if not matched: i += 1

  seqs: "OrderedDict[tuple, list]" = OrderedDict()
  for idx, (bar, off, size, data, eid) in enumerate(facts):
    if idx in covered: continue
    seqs.setdefault((bar, off, size), []).append(data)
  def lbl(bar, off):
    # Register-name resolution is BAR0-only: BAR1 is memory/page-table space,
    # so BAR1 offsets must never inherit BAR0 register names (e.g. BOOT_0 @0x0).
    nm = next((_resolve_reg_name(d, off) for d in _trace_regdevs if _resolve_reg_name(d, off)), None) if bar == 0 else None
    return f"BAR{bar}+{off:#x}" + (f" [{nm}]" if nm else "")
  parts = []
  for bar, off, last_off, cnt, period, doff, lanes_out, re0, re1 in runs:
    big = period >= 2   # interleaved [entry, companion] windows = PDE pairs
    tag = ("BIG_PDE_RUN" if big else "PTE_RUN") + f"[E{re0:06d}..E{re1:06d}]"
    lines = [f"{tag} BAR{bar}+{off:#x}..{last_off:#x} count={cnt} stride={doff:#x}"
             + (" [dual-PDE]" if period == 2 else "")]
    for li, (lo, v0, dv) in enumerate(lanes_out):
      vl = v0 + dv * (cnt - 1)
      if dv:
        step = f"{dv:+#x}" + (f" ({dv >> 20:+d} MiB)" if abs(dv) >= (1 << 20) else "")
        lines.append(f"{pre}      lane{li} +{lo - off:#x}: {v0:#x} -> {vl:#x} step {step}")
      else:
        lines.append(f"{pre}      lane{li} +{lo - off:#x}: const 0x0 x{cnt}")
    parts.append("\n".join(lines))
  for (bar, off, size), datas in seqs.items():
    L = lbl(bar, off)
    if len(datas) == 1:
      d = datas[0]
      if len(d) <= 8: parts.append(f"{L}={int.from_bytes(d[:8].ljust(8, chr(0).encode()), 'little'):#x}")
      elif TRACE_NOZERO and not d.strip(b"\x00"): parts.append(f"{L} PT_INIT/zero-filled size={len(d):#x}")
      else: parts.append(f"{L} <- {len(d):#x}B blob")
      continue
    vals = [int.from_bytes(d[:8].ljust(8, b"\0"), "little") for d in datas]
    diffs = {b - a for a, b in zip(vals, vals[1:])}
    if len(diffs) == 1 and vals[-1] != vals[0]:
      parts.append(f"{L}: {vals[0]:#x}->{vals[-1]:#x} step {diffs.pop():#x} x{len(datas)}")
    elif len(set(datas)) == 1:
      parts.append(f"{L}={vals[0]:#x} x{len(datas)}")
    else:
      parts.append(f"{L}: {len(datas)} writes/{len(set(datas))} distinct")
  rng = f"[E{facts[0][4]:06d}] " if facts else ""
  stg = stage_now()
  stgp = f"[{stg}] " if stg else ""
  if not facts and nreads:
    head = f"WAIT_IO reads={nreads} writes=0" if IO_TAG == "WAIT" else f"omitted polling: reads={nreads}"
    print(f"{pre}{stgp}{rng}{head}", flush=True)
  else:
    print(f"{pre}{stgp}{rng}{'WAIT_IO' if IO_TAG == 'WAIT' else 'IO'} reads={nreads} writes={len(facts)}", flush=True)
  for p_ in parts[:16]:
    for ln in p_.split("\n"): print(f"{pre}{stgp}    {ln}", flush=True)
  if len(parts) > 16: print(f"{pre}{stgp}    ... +{len(parts) - 16} more groups", flush=True)

def _trace(tag: str, body: str, payload: bytes | None = None):
  with _trace_lock:
    pre = _stage_pre()
    stg = stage_now()
    stgp = f"[{stg}] " if stg else ""
    lines = body.split("\n")
    print(f"{pre}{stgp}{tag}[{_trace_caller()}] {lines[0]}", flush=True)
    # Continuation rows stay close to the stage tag (long caller names would
    # otherwise push decoded fields absurdly far right).
    pad = pre + stgp + "  "
    for cont in lines[1:]:
      print(f"{pad}{cont}", flush=True)
    if payload: print(_trace_hexdump(payload, indent=f"{pre}      "), flush=True)
    if len(lines) > 1: print(flush=True)   # blank line closes a multi-line section

def tprint(msg: str):
  # Narrative announce. Numbering model:
  #   top-level (outside scopes): advances the stage's ANNOUNCE counter -> Sn.1, Sn.2
  #   inside a scope: advances that announce's CHILD counter -> Sn.1.1, Sn.1.2
  #   first announce right after stage_set(): bare Sn
  if TRACE:
    smode = getattr(_tls, "smode", [])
    if getattr(_tls, "just", False) and not smode:
      setattr(_tls, "just", False)              # bare Sn header announce
    elif smode:
      if smode[-1] == "child":
        setattr(_tls, "chld", getattr(_tls, "chld", 0) + 1)   # Sn.a.1, Sn.a.2 ...
      # grandchild scopes keep their parent's label (indent shows depth)
    else:
      setattr(_tls, "ann", getattr(_tls, "ann", 0) + 1)       # Sn.1, Sn.2 ...
      setattr(_tls, "chld", 0)
    _trace("CTX", msg)

def usb_trace(msg: str, level: int = 1):
  """Emit a USB-specific semantic record without changing lifecycle numbering."""
  if TRACE and USB_TRACE >= level: _trace("USB", msg)

def _usb_link_status(status: int) -> str:
  speed = status & 0xf
  width = (status >> 4) & 0x3f
  flags = (" training" if status & (1 << 11) else "") + (" dll_active" if status & (1 << 13) else "")
  return f"Gen{speed} x{width}{flags}" if speed and width else f"status={status:#06x}"

def usb_retrain_report(generation: int, rows: list[tuple]):
  """Report PCIe target/link changes after retraining has completed."""
  if not TRACE or USB_TRACE < 1: return
  lines = [f"PCIe retrain target=Gen{generation}"]
  for role, bus, cap, old_target, before, after in rows:
    lines.append(f"{role:<6} bus={bus}:00.0 cap={cap:#x} target Gen{old_target}->Gen{generation} "
                 f"link {_usb_link_status(before)} -> {_usb_link_status(after)}")
  _trace("USB", "\n".join(lines))

def usb_stream_report(*, image_size: int, ring_size: int, batch_size: int, logical_bytes: int, wire_bytes: int,
                      batch_count: int, slots: list[int], launched_at: float, last_bulk_at: float, restored_at: float,
                      queue_bytes: int, arg_pages: int):
  """Flush GSP stream records after the timing-critical F2 loop and restoration."""
  if not TRACE or USB_TRACE < 1: return
  bulk_ms, restore_ms = (last_bulk_at - launched_at) * 1e3, (restored_at - launched_at) * 1e3
  rate = wire_bytes / max(last_bulk_at - launched_at, 1e-9) / 1e6
  lines = ["GSP SRAM stream complete",
    f"image={image_size:#x} initial_ring={ring_size:#x} remaining={logical_bytes:#x}",
    f"refills={batch_count} batch={batch_size:#x} wire={wire_bytes:#x} slots={','.join(map(str, slots))}",
    f"F2 refill sectors={batch_size // 512:#x} slot_span={batch_size // 0x4000} bulk_out=0x02",
    f"timing launch->last_bulk={bulk_ms:.3f}ms wire_rate={rate:.1f}MB/s launch->restore={restore_ms:.3f}ms",
    f"post_stream F2_queue_restore={queue_bytes:#x} E5_arg_pages={arg_pages}"]
  if USB_TRACE >= 2:
    lines.append("buffered refill schedule: idx image_off logical wire slot F2_index deadline_ms")
    slot_span = batch_size // 0x4000
    for idx in range(batch_count):
      off = ring_size + idx * batch_size
      logical = min(batch_size, logical_bytes - idx * batch_size)
      slot = slots[idx % len(slots)]
      deadline_ms = 3.0 + idx * batch_size / ring_size * 1.4
      lines.append(f"[{idx:03d}] {off:#010x} {logical:#07x} {batch_size:#07x} {slot:02d} "
                   f"{slot | slot_span << 8:#06x} {deadline_ms:9.3f}")
  _trace("USB", "\n".join(lines))

def _trace_hexdump(data: bytes, indent="      ") -> str:
  if TRACE_NOZERO and data and not data.strip(b"\x00"):
    return f"{indent}<all zeros, {len(data):#x} bytes>"
  show = data[:TRACE_MAXDUMP]
  lines, prev, skip = [], None, False
  for i in range(0, len(show), 16):
    chunk = bytes(show[i:i + 16])
    if TRACE_NOZERO and chunk == prev: skip = True; continue
    if skip: lines.append(f"{indent}*"); skip = False
    prev = chunk
    lines.append(f"{indent}{i:08x}  {chunk.hex(' ')}")
  if skip: lines.append(f"{indent}*")
  if len(data) > len(show): lines.append(f"{indent}... +{len(data) - len(show):#x} more bytes")
  return "\n".join(lines)

def _trace_rpc_func(func: int) -> str:
  for pref in ("NV_VGPU_MSG_FUNCTION_", "NV_VGPU_MSG_EVENT_"):
    for k, v in nv.__dict__.items():
      if k.startswith(pref) and v == func: return k.removeprefix(pref)
  return f"FUNC_{func:#x}"

_PCI_CFG_NAMES = {0x00: "PCI_VENDOR_ID", 0x02: "PCI_DEVICE_ID", 0x04: "PCI_COMMAND", 0x06: "PCI_STATUS",
  0x08: "PCI_CLASS_REV", 0x0c: "PCI_CACHE_LINE", 0x10: "PCI_BAR0", 0x14: "PCI_BAR1", 0x18: "PCI_BAR2",
  0x1c: "PCI_BAR3", 0x20: "PCI_BAR4", 0x24: "PCI_BAR5", 0x2c: "PCI_SUBSYS_VND", 0x2e: "PCI_SUBSYS_ID",
  0x30: "PCI_ROM_ADDR", 0x34: "PCI_CAP_PTR", 0x3c: "PCI_INT_PIN", 0x3d: "PCI_INT_LINE"}
def _trace_pci_name(off: int) -> str | None:
  return _PCI_CFG_NAMES.get(off)

_PCI_CMD_BITS = (("IO", 0x1), ("MEM", 0x2), ("MASTER", 0x4))
def fmt_pci_command(v: int) -> str:
  """Decode PCI_COMMAND set bits, scan-friendly: "[IO MEM MASTER]"."""
  return "[" + " ".join(n for n, m in _PCI_CMD_BITS if v & m) + "]"

# ---------------------------------------------------------------------------
# RM object records: one semantic line per GSP_RM_ALLOC / GSP_RM_CONTROL /
# SET_PAGE_DIRECTORY, with a handle -> class registry so parents render as
# cf000008<NV01_DEVICE_0>. Raw wire structs remain TRACE_RAW-only.
# ---------------------------------------------------------------------------
_HANDLE_CLS = {0xc1000000: "USER_ROOT", 0xc1e00004: "PRIV_ROOT"}
_CTRL_CMD_NAMES: dict = {}          # populated by add.py from the autogen header
RM_SEMANTIC_FUNCS = ("GSP_RM_ALLOC", "GSP_RM_CONTROL", "SET_PAGE_DIRECTORY", "ALLOC_MEMORY")

def hfmt(h: int) -> str:
  n = _HANDLE_CLS.get(h)
  return f"{h:08x}<{n}>" if n else f"{h:08x}"

def rm_record(op: str, cls: str, obj=None, parent=None, cmd=None, extras=(), ok=True):
  """One readable RM operation record: 'ALLOC <class> obj=.. parent=.. -> OK'."""
  if obj is not None and cls: _HANDLE_CLS[obj] = cls
  head = f"{op:<6} {cls:<24}"
  if obj is not None: head += f" obj={hfmt(obj)}"
  if parent is not None: head += f" parent={hfmt(parent)}"
  if cmd is not None:
    nm = _CTRL_CMD_NAMES.get(cmd)
    head += f" cmd={nm}({cmd:#x})" if nm else f" cmd={cmd:#010x}"
  body = [f"      {k}={v}" for k, v in extras if k is not None]
  st = "-> OK" if ok else "-> FAIL"
  if body: _trace("RM", head + "\n" + "\n".join(body) + f"\n      {st}")
  else: _trace("RM", head.rstrip() + "  " + st)

_MTHD_NAMES = None
def _trace_method_name(mthd: int) -> str | None:
  # Pushbuffer methods are encoded as mthd>>2; seed with the static table then
  # sweep the autogen class headers (NVC56F semaphore/fifo, NVC6C0 compute...).
  global _MTHD_NAMES
  if _MTHD_NAMES is None:
    d = dict(_METHOD_NAMES)
    for k, v in nv_gpu.__dict__.items():
      if isinstance(v, int) and v > 0 and k.startswith(("NVC56F_", "NVC6C0_", "NVC36F_", "NVA06F_")) and v & 3 == 0:
        d.setdefault(v >> 2, k)
    _MTHD_NAMES = d
  return _MTHD_NAMES.get(mthd)

_RPC_STRUCTS = None
def _trace_structs_init():
  global _RPC_STRUCTS
  if _RPC_STRUCTS is None:
    _RPC_STRUCTS = {}
    for suffix, t in [("GSP_RM_CONTROL", nv.rpc_gsp_rm_control_v), ("GSP_RM_ALLOC", nv.rpc_gsp_rm_alloc_v),
                      ("SET_PAGE_DIRECTORY", nv.rpc_set_page_directory_v), ("UNLOADING_GUEST_DRIVER", nv.rpc_unloading_guest_driver_v),
                      ("GSP_SET_SYSTEM_INFO", nv.GspSystemInfo), ("SET_REGISTRY", nv.PACKED_REGISTRY_TABLE),
                      ("ALLOC_MEMORY", nv.rpc_alloc_memory_v)]:
      if (fid := getattr(nv, f"NV_VGPU_MSG_FUNCTION_{suffix}", None)) is not None: _RPC_STRUCTS[fid] = t

_CLASS_NAMES = None
def _trace_class_name(cls: int) -> str | None:
  # Reverse-map an RM class id to a symbol name from the autogen headers.
  # Curated class names win value collisions against the generic sweep
  # (e.g. hClass=0 is NV01_ROOT, not some random zero-valued constant).
  global _CLASS_NAMES
  if _CLASS_NAMES is None:
    import re
    d = {}
    for cn, _ in _TRACE_CLASS_PAIRS:
      if (c := getattr(nv_gpu, cn, None)) is not None and isinstance(c, int): d.setdefault(c, cn)
    pat = re.compile(r"^(?:NV[0-9]{2}|FERMI|KEPLER|AMPERE|TURING|PASCAL|VOLTA|BLACKWELL)[0-9A-Za-z_]*$")
    for k, v in nv_gpu.__dict__.items():
      if isinstance(v, int) and "_CTRL_" not in k and pat.match(k): d.setdefault(v, k)
    _CLASS_NAMES = d
  return _CLASS_NAMES.get(cls)

# Fields shown even when zero — they carry meaning (hClass=0 IS NV01_ROOT,
# and the response's params.hClient is the allocated-handle echo).
_TRACE_ALWAYS_SHOW = frozenset({"status", "flags", "hClass", "hParent", "hClient"})

_CLASS_PARAMS = None
_TRACE_CLASS_PAIRS = [("NV01_ROOT", "NV0000_ALLOC_PARAMETERS"), ("NV01_DEVICE_0", "NV0080_ALLOC_PARAMETERS"),
  ("NV20_SUBDEVICE_0", "NV2080_ALLOC_PARAMETERS"), ("NV01_MEMORY_VIRTUAL", "NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS"),
  ("FERMI_VASPACE_A", "NV_VASPACE_ALLOCATION_PARAMETERS"), ("FERMI_CONTEXT_SHARE_A", "NV_CTXSHARE_ALLOCATION_PARAMETERS"),
  ("KEPLER_CHANNEL_GROUP_A", "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS"),
  ("AMPERE_CHANNEL_GPFIFO_A", "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS"),
  ("BLACKWELL_CHANNEL_GPFIFO_A", "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS")]

def _trace_class_params(cls: int):
  # Class id -> allocation-parameter struct, for decoding the params[] block
  # of GSP_RM_ALLOC requests AND responses (response params carry results).
  global _CLASS_PARAMS
  if _CLASS_PARAMS is None:
    _CLASS_PARAMS = {}
    for cn, pn in _TRACE_CLASS_PAIRS:
      if (c := getattr(nv_gpu, cn, None)) is not None and (p := getattr(nv_gpu, pn, None)) is not None: _CLASS_PARAMS[c] = p
  return _CLASS_PARAMS.get(cls)

_ENG_NAMES = None
def _trace_engine_name(idx: int) -> str:
  global _ENG_NAMES
  if _ENG_NAMES is None:
    d = {}
    for k, v in nv_gpu.__dict__.items():
      if k.startswith("NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_") and isinstance(v, int):
        d.setdefault(v, k.removeprefix("NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_"))
    _ENG_NAMES = d
  return _ENG_NAMES.get(idx, f"eng{idx}")

_BUFID_NAMES = None
def _trace_bufid_name(idx: int) -> str:
  global _BUFID_NAMES
  if _BUFID_NAMES is None:
    d = {}
    for k, v in nv_gpu.__dict__.items():
      if k.startswith("NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_") and isinstance(v, int):
        d.setdefault(v, k.removeprefix("NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_"))
    _BUFID_NAMES = d
  return _BUFID_NAMES.get(idx, f"buf{idx}")

_ENGTYPE_NAMES = None
def _trace_engtype_name(idx: int) -> str:
  global _ENGTYPE_NAMES
  if _ENGTYPE_NAMES is None:
    d = {}
    for k, v in nv_gpu.__dict__.items():
      if k.startswith("NV2080_ENGINE_TYPE_") and isinstance(v, int):
        d.setdefault(v, k.removeprefix("NV2080_ENGINE_TYPE_"))
    _ENGTYPE_NAMES = d
  return _ENGTYPE_NAMES.get(idx, f"engType{idx}")

def _trace_row(indent: str, off: int, raw: bytes, name: str, val: int, extra: str = "") -> str:
  # Three-column style: offset | raw LE bytes | decoded value.
  return f"{indent}+{off:#06x}  {raw.hex(' ').ljust(23)}  {name.ljust(20)}= {val:#x}{extra}"

def _trace_fmt_event(func: int, msg: bytes) -> str:
  # GSP event payloads that aren't plain RPC structs.
  if func == nv.NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD and len(msg) >= 0x68:
    # GSP-RM log stream record: [+0x10 sev][+0x18 tag(8)][payload strings...].
    try:
      sev = int.from_bytes(msg[0x10:0x14], 'little')
      tag = bytes(msg[0x18:0x20]).split(b'\0')[0].decode('ascii', 'replace').strip() or "?"
      ts = int.from_bytes(msg[8:16], 'little')
      import re
      strs = [m.decode('ascii') for m in re.findall(rb"[\x20-\x7e]{4,}", msg[0x18:])][:8]
      return f"    nocat sev={sev} ts={ts:#x} [{tag}] " + " | ".join(strs)
    except Exception: return ""
  if func == nv.NV_VGPU_MSG_EVENT_UCODE_LIBOS_PRINT and len(msg) >= 0x18:
    # rpc_ucode_libos_print wrapper: text itself lives in the LOG sysmem buffer.
    eng = int.from_bytes(msg[0:4], 'little'); idx = int.from_bytes(msg[8:12], 'little')
    sz = int.from_bytes(msg[4:8], 'little'); off = int.from_bytes(msg[0x10:0x18], 'little')
    return f"    libos-print eng={eng:#x} idx={idx} len={sz:#x} -> LOG buffer +{off:#x} (text in sysmem, not on wire)"
  if func == nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE:
    return "(signal-only event, no payload)"
  return ""

def _trace_fmt_struct(func: int, data: bytes, indent="    ") -> str:
  # Aligned field decode of a known RPC header, e.g. rpc_gsp_rm_alloc_v's
  # hClient/hParent/hObject/hClass/status/paramsSize/flags at +0x00..+0x1c.
  _trace_structs_init()
  if (t := _RPC_STRUCTS.get(func)) is None or len(data) < ctypes.sizeof(t): return ""
  try: s = t.from_buffer_copy(data[:ctypes.sizeof(t)])
  except ValueError: return ""
  fields = getattr(t, "_real_fields_", None) or getattr(t, "_fields_", [])
  rows = []
  for f in fields:
    if isinstance(f, str): continue
    fname, ctype, off = f[0], f[1], f[2]
    v = getattr(s, fname)
    if not isinstance(v, int): continue
    if v == 0 and fname not in _TRACE_ALWAYS_SHOW: continue
    raw = data[off:off + ctypes.sizeof(ctype)]
    extra = f"  ({cn})" if fname == "hClass" and (cn := _trace_class_name(v)) else ""
    rows.append(_trace_row(indent, off, raw, fname, v, extra))
  # Decode the params[] block of RM_ALLOCs (class-specific struct).
  if func == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC:
    if getattr(s, "hObject", 0):
      cn = _trace_class_name(s.hClass)
      _RM_OBJS[s.hObject] = {"cls": cn or f"class{s.hClass:#x}", "parent": s.hParent}
    pt = _trace_class_params(s.hClass) if (hasattr(s, "hClass")) else None
    if pt is not None and len(data) >= ctypes.sizeof(t) + ctypes.sizeof(pt):
      try: ps = pt.from_buffer_copy(data[ctypes.sizeof(t):ctypes.sizeof(t) + ctypes.sizeof(pt)])
      except ValueError: return "\n".join(rows)
      base = ctypes.sizeof(t)
      pfields = getattr(pt, "_real_fields_", None) or getattr(pt, "_fields_", [])
      n_before = len(rows)
      for f in pfields:
        if isinstance(f, str): continue
        fname, ctype, off = f[0], f[1], f[2]
        v = getattr(ps, fname)
        if isinstance(v, int) and (v or fname in _TRACE_ALWAYS_SHOW):
          rows.append(_trace_row(indent, base + off, data[base + off:base + off + ctypes.sizeof(ctype)], "params." + fname, v))
      # Array + memory-descriptor params (e.g. CHANNELGPFIFO hUserdMemory[],
      # userdOffset[], ramfcMem/userdMem/... NV_MEMORY_DESC_PARAMS).
      MEMDESC_SIG = {"base", "size", "addressSpace", "cacheAttrib"}
      for f in pfields:
        if isinstance(f, str): continue
        fname, ctype, off = f[0], f[1], f[2]
        arr_len, elem_t = getattr(ctype, "_length_", None), getattr(ctype, "_type_", None)
        try:
          if arr_len and elem_t is not None and "uint" in str(elem_t):
            vals = [(i, getattr(ps, fname)[i]) for i in range(arr_len) if getattr(ps, fname)[i]]
            if vals:
              body_s = " ".join(f"[{i}]={v:#x}" for i, v in vals[:6]) + (" ..." if len(vals) > 6 else "")
              rows.append(f"{indent}{('params.' + fname).ljust(19)} {body_s}")
              if fname == "userdOffset" and vals:
                u0 = getattr(ps, "hUserdMemory", [0] * 8)[0] if hasattr(ps, "hUserdMemory") else 0
                if u0: rows.append(f"{indent}{'  -> USERD @ BAR1'.ljust(19)} {u0 + vals[0][1]:#x} (GPPUT @+0x8c)")
          elif hasattr(ctype, "_real_fields_") and ctype._real_fields_ and \
               MEMDESC_SIG <= {f2[0] for f2 in ctype._real_fields_}:
            for i in range(getattr(ctype, "_length_", 0) or 1):
              e = getattr(ps, fname)[i]
              if e.base or e.size:
                rows.append(f"{indent}  params.{fname}[{i}]: base={e.base:#x} size={e.size:#x} addrSpace={e.addressSpace} cache={e.cacheAttrib}")
        except Exception: continue
      if len(rows) == n_before:
        rows.append(f"{indent}{('params.' + pt.__name__).ljust(20)} (all-zero echo)")
  # SET_REGISTRY: PACKED_REGISTRY_TABLE with name-offset string table.
  if func == nv.NV_VGPU_MSG_FUNCTION_SET_REGISTRY and len(data) >= 8:
    tbl_size, n_entries = int.from_bytes(data[0:4], 'little'), int.from_bytes(data[4:8], 'little')
    tnames = {v: k.removeprefix("REGISTRY_TABLE_ENTRY_TYPE_") for k, v in nv.__dict__.items()
              if k.startswith("REGISTRY_TABLE_ENTRY_TYPE_")}
    rows.append(f"{indent}{('registry entries=' + str(n_entries)).ljust(20)}= tableSize={tbl_size:#x}")
    for i in range(n_entries):
      base = 8 + i * 16
      if base + 16 > len(data): break
      name_off, rtype, rval, rlen = struct.unpack_from('<IIII', data, base)
      nm = "?"
      if 0 < name_off < len(data):
        end = data.find(b'\x00', name_off)
        nm = data[name_off:end].decode('ascii', 'replace') if end > 0 else "?"
      tname = tnames.get(rtype, f"type{rtype}")
      if rtype == nv.REGISTRY_TABLE_ENTRY_TYPE_STRING and 0 < rval < len(data):
        vend = data.find(b'\x00', rval)
        val_s = repr(data[rval:vend].decode('ascii', 'replace')) if vend > 0 else "?"
      else:
        val_s = f"{rval:#x}" + (f" ({rval})" if 0 < rval <= 99 else "")
      rows.append(f"{indent}  [{i}] {nm}")
      rows.append(f"{indent}      type={tname:<6} value={val_s} length={rlen} nameOffset={name_off:#x}")
  # RM_CONTROL completions with per-cmd params decoders.
  if func == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL:
    hdr_sz = ctypes.sizeof(t)
    if s.cmd == nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO and len(data) >= hdr_sz + (isz := ctypes.sizeof(nv_gpu.struct_NV2080_CTRL_INTERNAL_STATIC_GR_CONTEXT_BUFFERS_INFO)):
      info = nv_gpu.struct_NV2080_CTRL_INTERNAL_STATIC_GR_CONTEXT_BUFFERS_INFO.from_buffer_copy(data[hdr_sz:hdr_sz + isz])
      for i in range(len(info.engine)):
        e = info.engine[i]
        if e.size in (0, 0xFFFFFFFF): continue
        raw = data[hdr_sz + i * 8:hdr_sz + i * 8 + 8]
        nm = f"ctxbuf[{_trace_engine_name(i)}]"
        rows.append(f"{indent}+{hdr_sz + i * 8:#06x}  {raw.hex(' ').ljust(23)}  {nm.ljust(20)}= size={e.size:#010x} align={e.alignment:#x}")
    elif s.cmd == nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX and len(data) >= hdr_sz + (psz := ctypes.sizeof(nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS)):
      p = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS.from_buffer_copy(data[hdr_sz:hdr_sz + psz])
      rows.append(f"{indent}{('promote ' + _trace_engtype_name(p.engineType)).ljust(20)}= hChan={p.hChanClient:#x} obj={p.hObject:#x} entries={p.entryCount}")
      for i in range(min(p.entryCount, len(p.promoteEntry))):
        e = p.promoteEntry[i]
        fl = (" init" if e.bInitialize else "") + (" nonmapped" if e.bNonmapped else "")
        rows.append(f"{indent}  [{i:>2}] {_trace_bufid_name(e.bufferId):<24} pa={e.gpuPhysAddr:#x} va={e.gpuVirtAddr:#x} size={e.size:#x}{fl}")
    elif s.cmd == nv_gpu.NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES and len(data) >= hdr_sz + (qsz := ctypes.sizeof(qt := nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS)):
      q = qt.from_buffer_copy(data[hdr_sz:hdr_sz + qsz])
      rows.append(f"{indent}{f'pdes pageSize={q.pageSize:#x}'.ljust(20)} va=[{q.virtAddrLo:#x}..{q.virtAddrHi:#x}] levels={q.numLevelsToCopy}")
      for i in range(min(q.numLevelsToCopy, len(q.levels))):
        L = q.levels[i]
        rows.append(f"{indent}  L{i} pa={L.physAddress:#x} size={L.size:#x} aperture={L.aperture} pageShift={L.pageShift:#x}")
  return "\n".join(rows)

def _trace_decode_bits(reg, val: int) -> str:
  # Decode a register value through its autogen field table {(start,end)}.
  if not getattr(reg, "fields", None): return ""
  try: d = reg.decode(val)
  except Exception: return ""
  nz = [f"{k}={v}" for k, v in d.items() if v]
  return " {" + ", ".join(nz) + "}" if nz else ""

def _resolve_reg_name(nvdev, addr: int) -> str | None:
  # Brute-force the (base, off_lambda, fields) tuples loaded by include() to map an
  # absolute MMIO address back to a register name (+ reg object for bitfields). Cached.
  # Falcon-block defs are RELATIVE (base=0): try the known engine bases for those.
  if addr in _trace_reg_cache: return _trace_reg_cache[addr]
  cands = [(k, v) for k, v in nvdev.__dict__.items() if _NVReg is not None and isinstance(v, _NVReg)]
  engine_bases = (0x00110000, 0x00840000)   # PGSP/GSP falcon, SEC2
  def scan(k, v, extra):
    idxs = range(256) if callable(v.off) else (0,)
    for i in idxs:
      try: o = v.off(i) if callable(v.off) else v.off
      except Exception: return None
      if o is None or not isinstance(o, int): continue
      if v.base + o + extra == addr: return (k + (f"[{i}]" if i else ""), v)
    return None
  res = None
  def is_array(v) -> bool:
    # Indexed registers have index-dependent offsets; constant lambdas
    # (lambda: 0x624) are effectively scalars and must NOT be treated as
    # spanning aliases like INTR_RETRIGGER/PGSP_QUEUE_HEAD do.
    if not callable(v.off): return False
    try: return v.off(0) != v.off(5)
    except Exception: return True
  def match_one(arrays: bool, rel_only: bool):
    # Priority: engine-RELATIVE scalars beat far-absolutes; arrays last.
    # Ties (e.g. FALCON_MAILBOX0 vs PFB_..._REPLAYABLE, both (0,0x40)) go to
    # names carrying the matching engine's block hint.
    hints = ("PGSP", "PFALCON", "PRISCV", "PSEC")
    best = None
    for k, v in cands:
      if is_array(v) != arrays: continue
      if not isinstance(getattr(v, "base", None), int): continue
      rel = v.base < (1 << 17)
      if rel != rel_only: continue
      extras = (*engine_bases, 0) if rel else (0,)
      for extra in extras:
        if (r := scan(k, v, extra)):
          score = 2 if extra != 0 and any(h in r[0] for h in hints) else (1 if extra == 0 else 0)
          if best is None or score > best[0]: best = (score, r)
          break
    return best[1] if best else None
  res = (match_one(False, True) or match_one(False, False) or
         match_one(True, True) or match_one(True, False))
  name = res[0] if res else None
  _trace_reg_cache[addr] = name
  _trace_reg_obj[addr] = res[1] if res else None
  return name

_TRACE_BULK_CMDS = None   # MMIO_READ/WRITE traced in _bulk_read/_bulk_write instead of _rpc

# ============================================================================
# memory.py (vendored from ref/tinygrad/tinygrad/runtime/support/memory.py)
# ============================================================================
class BumpAllocator:
  """Boot-phase bump allocator: simple linear alloc with optional wraparound."""
  def __init__(self, size: int, base: int = 0, wrap: bool = True):
    self.size, self.ptr, self.base, self.wrap = size, 0, base, wrap
  def alloc(self, size: int, alignment: int = 1) -> int:
    if round_up(self.ptr, alignment) + size > self.size:
      if not self.wrap: raise RuntimeError("Out of memory")
      self.ptr = 0
    self.ptr = (res := round_up(self.ptr, alignment)) + size
    return res + self.base

class TLSFAllocator:
  """Two-Level Segregated Fit allocator for VRAM and sysmem pages."""
  def __init__(self, size: int, base: int = 0, block_size: int = 16, lv2_cnt: int = 16):
    self.size, self.base, self.block_size, self.l2_cnt = size, base, block_size, lv2_cnt.bit_length()
    self.storage = [collections.defaultdict(list) for _ in range(size.bit_length() + 1)]
    self.lv1_entries = [0] * len(self.storage)
    self.blocks = {0: (size, None, None, True)}
    if size > 0: self._insert_block(0, size)

  @functools.cache
  def lv1(self, size): return size.bit_length()
  @functools.cache
  def lv2(self, size): return (size - (1 << (size.bit_length() - 1))) // (1 << max(0, size.bit_length() - self.l2_cnt))

  def _insert_block(self, start: int, size: int, prev=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].append(start)
    self.lv1_entries[self.lv1(size)] += 1
    self.blocks[start] = (size, start + size, prev, True)
    return self
  def _remove_block(self, start: int, size: int, prev=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].remove(start)
    self.lv1_entries[self.lv1(size)] -= 1
    self.blocks[start] = (size, start + size, prev, False)
    return self
  def _split_block(self, start: int, size: int, new_size: int):
    nxt = self.blocks[start][1]
    assert self.blocks[start][3], "block must be free"
    self._remove_block(start, size)._insert_block(start, new_size)._insert_block(start + new_size, size - new_size, prev=start)
    if nxt in self.blocks:
      self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start + new_size, self.blocks[nxt][3])
    return self
  def _merge_right(self, start: int):
    size, nxt, _, is_free = self.blocks[start]
    assert is_free, "block must be free"
    while is_free and nxt in self.blocks:
      if (blk := self.blocks[nxt])[3] is False: break
      self._remove_block(start, size)._remove_block(nxt, blk[0])._insert_block(start, size := size + blk[0])
      assert self.blocks[start][1] == blk[1]
      _, nxt, _, _ = self.blocks.pop(nxt)
    if nxt in self.blocks: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start, self.blocks[nxt][3])
  def _merge_block(self, start: int):
    while (x := self.blocks[start][2]) is not None and self.blocks[x][3] is True: start = x
    self._merge_right(start)
  def alloc(self, req_size: int, align: int = 1) -> int:
    req_size = max(self.block_size, req_size)
    size = max(self.block_size, req_size + align - 1)
    size = round_up(size, (1 << size.bit_length() - self.l2_cnt))
    for l1 in range(self.lv1(size), len(self.storage)):
      if self.lv1_entries[l1] == 0: continue
      for l2 in range(self.lv2(size) if l1 == size.bit_length() else 0, (1 << self.l2_cnt)):
        if len(self.storage[l1][l2]) > 0:
          start = self.storage[l1][l2][0]
          nsize = self.blocks[start][0]
          assert nsize >= size, "block must be larger"
          if (new_start := round_up(start, align)) != start:
            self._split_block(start, nsize, new_start - start)
            start, nsize = new_start, self.blocks[new_start][0]
          if nsize > req_size: self._split_block(start, nsize, req_size)
          self._remove_block(start, req_size)
          return start + self.base
    raise MemoryError(f"Can't allocate {req_size} bytes")
  def free(self, start: int):
    self._insert_block(start - self.base, self.blocks[start - self.base][0])._merge_block(start - self.base)

class AddrSpace(enum.Enum):
  PHYS = enum.auto(); SYS = enum.auto(); PEER = enum.auto()

@dataclasses.dataclass(frozen=True)
class VirtMapping:
  va_addr: int; size: int; paddrs: list; aspace: AddrSpace; uncached: bool = False; snooped: bool = False

class PageTableTraverseContext:
  def __init__(self, dev, pt, vaddr, create_pts=False, free_pts=False, inspect=False, boot=False):
    self.dev, self.vaddr, self.create_pts, self.free_pts, self.inspect, self.boot = dev, vaddr - dev.mm.va_base, create_pts, free_pts, inspect, boot
    self.pt_stack = [(pt, self._pt_pte_idx(pt, self.vaddr), self._pt_pte_size(pt))]
  def _pt_pte_cnt(self, lv): return self.dev.mm.pte_cnt[lv]
  def _pt_pte_size(self, pt): return self.dev.mm.pte_covers[pt.lv]
  def _pt_pte_idx(self, pt, va): return (va // self._pt_pte_size(pt)) % self._pt_pte_cnt(pt.lv)
  def level_down(self):
    pt, pte_idx, _ = self.pt_stack[-1]
    if not pt.valid(pte_idx):
      assert self.create_pts, "Not allowed to create new page table"
      pt.set_entry(pte_idx, self.dev.mm.palloc(0x1000, zero=True, boot=self.boot, ptable=True), table=True, valid=True)
    assert not pt.is_page(pte_idx), f"Must be table pt={pt.paddr:#x}, {pt.lv=} {pte_idx=} {pt.entry(pte_idx)=:#x}"
    child_page_table = self.dev.mm.pt_t(self.dev, pt.address(pte_idx), lv=pt.lv + 1)
    self.pt_stack.append((child_page_table, self._pt_pte_idx(child_page_table, self.vaddr), self._pt_pte_size(child_page_table)))
    return self.pt_stack[-1]
  def _try_free_pt(self) -> bool:
    pt, _, _ = self.pt_stack[-1]
    if self.free_pts and pt != self.dev.mm.root_page_table and all(not pt.valid(i) for i in range(self._pt_pte_cnt(self.pt_stack[-1][0].lv))):
      self.dev.mm.pfree(pt.paddr, ptable=True)
      parent_pt, parent_pte_idx, _ = self.pt_stack[-2]
      parent_pt.set_entry(parent_pte_idx, 0x0, valid=False)
      return True
    return False
  def level_up(self):
    while self._try_free_pt() or self.pt_stack[-1][1] == self._pt_pte_cnt(self.pt_stack[-1][0].lv):
      pt, pt_cnt, _ = self.pt_stack.pop()
      if pt_cnt == self._pt_pte_cnt(pt.lv): self.pt_stack[-1] = (self.pt_stack[-1][0], self.pt_stack[-1][1] + 1, self.pt_stack[-1][2])
  def next(self, size, paddr=None, off=0):
    while size > 0:
      pt, pte_idx, pte_covers = self.pt_stack[-1]
      if self.create_pts:
        assert paddr is not None, "paddr must be provided when allocating new page tables"
        while pte_covers > size or not pt.supports_huge_page(paddr + off) or self.vaddr & (pte_covers - 1) != 0:
          pt, pte_idx, pte_covers = self.level_down()
      else:
        while not pt.is_page(pte_idx) and (self.free_pts or pt.valid(pte_idx)):
          pt, pte_idx, pte_covers = self.level_down()
      entries = max(min(size // pte_covers, self._pt_pte_cnt(pt.lv) - pte_idx), 1 if self.inspect else 0)
      assert entries > 0, f"Invalid entries {size=:#x}, {pte_covers=:#x}"
      yield off, pt, pte_idx, entries, pte_covers
      size, off, self.vaddr = size - entries * pte_covers, off + entries * pte_covers, self.vaddr + entries * pte_covers
      self.pt_stack[-1] = (pt, pte_idx + entries, pte_covers)
      self.level_up()

class MemoryManager:
  """GMMU virtual/physical memory manager: multi-level page tables, VA/PA allocators."""
  va_allocator = None
  def __init__(self, dev, vram_size, boot_size, pt_t, va_bits, va_shifts, va_base, palloc_ranges, first_lv=0, reserve_ptable=False):
    self.dev, self.vram_size, self.va_shifts, self.va_base, lvl_msb = dev, vram_size, va_shifts, va_base, va_shifts + [va_bits + 1]
    self.pte_covers, self.pte_cnt = [1 << x for x in va_shifts][::-1], [1 << (lvl_msb[i+1] - lvl_msb[i]) for i in range(len(lvl_msb) - 1)][::-1]
    self.pt_t, self.palloc_ranges, self.level_cnt, self.va_bits, self.reserve_ptable = pt_t, palloc_ranges, len(va_shifts), va_bits, reserve_ptable
    self.boot_allocator = TLSFAllocator(boot_size, base=0)
    self.ptable_allocator = TLSFAllocator(round_up(vram_size // 512, 1 << 20) if self.reserve_ptable else 0, base=self.boot_allocator.size)
    self.pa_allocator = TLSFAllocator(vram_size - (off_sz := self.boot_allocator.size + self.ptable_allocator.size), base=off_sz)
    self.root_page_table = pt_t(self.dev, self.palloc(0x1000, zero=not self.dev.smi_dev, boot=True), lv=first_lv)
  def _frag_size(self, va, sz, must_cover=True):
    va_pwr2_div, sz_pwr2_div, sz_pwr2_max = va & -(va) if va > 0 else (1 << 63), sz & -(sz), (1 << (sz.bit_length() - 1))
    return (min(va_pwr2_div, sz_pwr2_div) if must_cover else min(va_pwr2_div, sz_pwr2_max)).bit_length() - 1 - 12
  def page_tables(self, vaddr, size):
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True)
    for _ in ctx.next(size, paddr=0): return [pt for pt, _, _ in ctx.pt_stack]
  def map_range(self, vaddr, size, paddrs, aspace, uncached=False, snooped=False, boot=False):
    assert size == sum(p[1] for p in paddrs), f"Size mismatch {size=} {sum(p[1] for p in paddrs)=}"
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, boot=boot, inspect=True)
    for _, pt, pte_idx, pte_cnt, _ in ctx.next(size):
      for pte_off in range(pte_cnt): assert not pt.valid(pte_idx + pte_off), f"PTE already mapped: {pt.entry(pte_idx + pte_off):#x}"
    if TRACE: tprint(f"GMMU_MAP va={vaddr:#x} size={size:#x} aperture={aspace.name} pages={sum(sz // 0x1000 for _, sz in paddrs)}")
    with _tscope():
      ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True, boot=boot)
      for paddr, psize in paddrs:
        for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(psize, paddr=paddr):
          for pte_off in range(pte_cnt):
            pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers, uncached=uncached, aspace=aspace, snooped=snooped,
                         frag=self._frag_size(ctx.vaddr + off, pte_cnt * pte_covers), valid=True)
      self.on_range_mapped()
    return VirtMapping(vaddr, size, paddrs, aspace=aspace, uncached=uncached, snooped=snooped)
  def unmap_range(self, vaddr, size):
    if TRACE: tprint(f"GMMU unmap va={vaddr:#x} size={size:#x}")
    with _tscope():
      ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, free_pts=True)
      for _, pt, pte_idx, pte_cnt, _ in ctx.next(size):
        for pte_id in range(pte_idx, pte_idx + pte_cnt):
          assert pt.valid(pte_id), f"PTE not mapped: {pt.entry(pte_id):#x}"
          pt.set_entry(pte_id, paddr=0x0, valid=False)
  def on_range_mapped(self): pass
  @classmethod
  def alloc_vaddr(cls, size, align=0x1000):
    assert cls.va_allocator is not None, "must be set"
    return cls.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))
  @functools.cache
  def identity_va(self, uncached):
    self.map_range(va := self.alloc_vaddr(self.vram_size, self.vram_size), self.vram_size, [(0, self.vram_size)], AddrSpace.PHYS, uncached=uncached)
    return va
  def valloc(self, size, align=0x1000, uncached=False, contiguous=False, zero=False):
    if not getenv("GMMU", 1):
      paddr = self.palloc(size := round_up(size, 0x1000), align, zero=False)
      return VirtMapping(self.identity_va(uncached) + paddr, size, [(paddr, size)], aspace=AddrSpace.PHYS, uncached=uncached)
    va = self.alloc_vaddr(size := round_up(size, 0x1000), align)
    if contiguous: paddrs = [(self.palloc(size, zero=True), size)]
    else:
      nxt_range, rem_size, paddrs = 0, size, []
      while rem_size > 0:
        while self.palloc_ranges[nxt_range][0] > rem_size: nxt_range += 1
        try: paddrs += [(self.palloc(try_sz := self.palloc_ranges[nxt_range][0], self.palloc_ranges[nxt_range][1], zero=zero), try_sz)]
        except MemoryError:
          nxt_range += 1
          if nxt_range == len(self.palloc_ranges):
            for paddr, _ in paddrs: self.pfree(paddr)
            raise MemoryError(f"Failed to allocate memory (OOM). Request size={size:#x}")
          continue
        rem_size -= self.palloc_ranges[nxt_range][0]
    return self.map_range(va, size, paddrs, aspace=AddrSpace.PHYS, uncached=uncached)
  def vfree(self, vm):
    if not getenv("GMMU", 1): return self.pfree(vm.paddrs[0][0])
    assert self.va_allocator is not None, "must be set"
    self.unmap_range(vm.va_addr, vm.size)
    self.va_allocator.free(vm.va_addr)
    for paddr, _ in vm.paddrs: self.pfree(paddr)
  def palloc(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    assert self.dev.is_booting == boot, "During booting, only boot memory can be allocated"
    allocator = self.boot_allocator if boot else (self.ptable_allocator if self.reserve_ptable and ptable else self.pa_allocator)
    paddr = allocator.alloc(round_up(size, 0x1000), align)
    if zero: self.dev.vram[paddr:paddr + size] = bytes(size)
    return paddr
  def pfree(self, paddr, ptable=False):
    (self.ptable_allocator if self.reserve_ptable and ptable else self.pa_allocator).free(paddr)


# ============================================================================
# hcq.py (vendored from ref/tinygrad/tinygrad/runtime/support/hcq.py)
# Slimmed: only MMIOInterface, FileIOInterface, HCQBuffer, hcq_filter_visible_devices.
# ============================================================================
class MMIOInterface:
  """Direct memory-mapped I/O window over a host-mapped address range."""
  def __init__(self, addr, nbytes, fmt='B'):
    self.mv, self.addr, self.nbytes, self.fmt = to_mv(addr, nbytes).cast(fmt), addr, nbytes, fmt
  def __len__(self): return self.nbytes // struct.calcsize(self.fmt)
  def __getitem__(self, k): return (self.mv[k] if self.fmt == 'B' else self.mv[k].tolist()) if isinstance(k, slice) else self.mv[k]
  def __setitem__(self, k, v):
    if self.fmt != 'B' and isinstance(v, (list, tuple)):
      self.mv[k] = array.array(self.fmt, v)
    else:
      self.mv[k] = v
  def view(self, offset=0, size=None, fmt=None):
    return MMIOInterface(self.addr + offset, (self.nbytes - offset) if size is None else size, fmt=fmt or self.fmt)

class FileIOInterface:
  """File-descriptor I/O: ioctl, mmap, read/write, eventfd."""
  def __init__(self, path="", flags=os.O_RDONLY, fd=None):
    self.path = path
    self.fd = fd or os.open(path, flags)
  def __del__(self):
    if hasattr(self, 'fd'):
      try: os.close(self.fd)
      except: pass
  def ioctl(self, request, arg):
    import fcntl
    return fcntl.ioctl(self.fd, request, arg)
  def mmap(self, start, sz, prot, flags, offset):
    return FileIOInterface._mmap(start, sz, prot, flags, self.fd, offset)
  def read(self, size=None, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as f: return f.read(size)
  def write(self, content, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as f: f.write(content)
  def listdir(self): return os.listdir(self.path)
  def seek(self, offset): os.lseek(self.fd, offset, os.SEEK_SET)
  @staticmethod
  def _mmap(start, sz, prot, flags, fd, offset):
    x = libc.mmap(start, sz, prot, flags, fd, offset)
    if x == 0xffffffffffffffff: raise OSError(f"Failed to mmap {sz} bytes at {hex(start)}: {os.strerror(ctypes.get_errno())}")
    return x
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, -1, offset)
  @staticmethod
  def munmap(buf, sz): return libc.munmap(buf, sz)
  @staticmethod
  def exists(path): return os.path.exists(path)
  @staticmethod
  def readlink(path): return os.readlink(path)
  @staticmethod
  def eventfd(initval, flags=None):
    import fcntl as _f
    return FileIOInterface(fd=os.eventfd(initval, flags))

def hcq_filter_visible_devices(devs, device):
  return devs

class HCQBuffer:
  """Hardware Command Queue buffer: GPU-visible VA allocation with optional CPU mmap view."""
  def __init__(self, va_addr, size, meta=None, _base=None, view=None, owner=None):
    self.va_addr, self.size, self.meta, self._base, self.view = va_addr, size, meta, _base, view
    self._devs = [owner] if owner is not None else []
    self.owner = owner
    self._mappings = {}
  def offset(self, offset=0, size=None):
    return HCQBuffer(self.va_addr + offset, size or (self.size - offset), owner=self.owner, meta=self.meta,
                     _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))
  def cpu_view(self):
    assert self.view is not None, "buffer has no cpu_view"
    return self.view
  @property
  def base(self): return self._base or self

# ============================================================================
# elf.py (vendored from ref/tinygrad/tinygrad/runtime/support/elf.py)
# ============================================================================
@dataclass(frozen=True)
class ElfSection:
  name: str
  header: Any
  content: bytes

def _elf_strtab(blob, idx): return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

def link_sym(sym: str, libs: list[str]) -> int:
  for lib in libs:
    try:
      return unwrap(ctypes.cast(getattr(ctypes.CDLL(ctypes.util.find_library(lib)), sym), ctypes.c_void_p).value)
    except (OSError, AttributeError):
      pass
  raise RuntimeError(f'Attempting to relocate against an undefined symbol {sym}')

def elf_loader(blob, force_section_align=1, link_libs=None):
  assert blob[:4] == libc.ELFMAG.encode(), "blob is not an ELF, missing magic bytes"
  ecls = {libc.ELFCLASS32: "Elf32", libc.ELFCLASS64: "Elf64"}[blob[libc.EI_CLASS]]
  header = getattr(libc, f"{ecls}_Ehdr").from_buffer_copy(blob)
  section_headers = (getattr(libc, f"{ecls}_Shdr") * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  sh_strtab = blob[(shstrst := section_headers[header.e_shstrndx].sh_offset):shstrst + section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_elf_strtab(sh_strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset + sh.sh_size]) for sh in section_headers]

  def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)
  rel = [(sh, sh.name[4:], _to_carray(sh, getattr(libc, f"{ecls}_Rel"))) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, sh.name[5:], _to_carray(sh, getattr(libc, f"{ecls}_Rela"))) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = next((_to_carray(sh, getattr(libc, f"{ecls}_Sym")) for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB), None)
  progbits = [sh for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS]

  image = bytearray(max([sh.header.sh_addr + sh.header.sh_size for sh in progbits if sh.header.sh_addr != 0] + [0]))
  for sh in progbits:
    if sh.header.sh_addr != 0:
      image[sh.header.sh_addr:sh.header.sh_addr + sh.header.sh_size] = sh.content
    else:
      image += b'\0' * (((align := max(sh.header.sh_addralign, force_section_align)) - len(image) % align) % align) + sh.content
      sh.header.sh_addr = len(image) - len(sh.content)

  relocs = []
  for sh, trgt_sh_name, c_rels in rel + rela:
    if trgt_sh_name == ".eh_frame":
      continue
    target_image_off = next(tsh for tsh in sections if tsh.name == trgt_sh_name).header.sh_addr
    rels = [(r.r_offset, unwrap(symtab)[getattr(libc, f"{ecls.upper()}_R_SYM")(r.r_info)],
             getattr(libc, f"{ecls.upper()}_R_TYPE")(r.r_info), getattr(r, "r_addend", 0)) for r in c_rels]
    relocs += [(target_image_off + roff,
                 link_sym(_elf_strtab(sh_strtab, sym.st_name), link_libs or []) if sym.st_shndx == 0 else
                 sections[sym.st_shndx].header.sh_addr + sym.st_value,
                 rtype, raddend) for roff, sym, rtype, raddend in rels]
  return memoryview(image), sections, relocs


# ============================================================================
# system.py (vendored from ref/tinygrad/tinygrad/runtime/support/system.py)
# Slimmed: only the APLRemotePCIDevice path (Mac eGPU via TinyGPU.app unix socket).
# ============================================================================
MAP_FIXED, MAP_FIXED_NOREPLACE = 0x10, 0x100000
MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400
PAGESIZE = mmap.PAGESIZE

class _System:
  @functools.cached_property
  def libsys(self): return ctypes.CDLL(ctypes.util.find_library("System"))
  @functools.cached_property
  def atomic_lib(self): return ctypes.CDLL(ctypes.util.find_library('atomic')) if not OSX else None
  def memory_barrier(self):
    lib = self.libsys if OSX else self.atomic_lib
    if lib is not None: lib.atomic_thread_fence(5)

  @staticmethod
  def pci_set_usb_bridge_buses(usb, primary, secondary, subordinate):
    """Program bridge bus numbers as byte writes; ASM's F0 path drops the combined dword write."""
    buses = primary | secondary << 8 | subordinate << 16
    for attempt in range(3):
      for offset, value in ((0, primary), (1, secondary), (2, subordinate)):
        usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS + offset, bus=primary, value=value, size=1)
      readback = usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=primary, size=4)
      if readback & 0xffffff == buses: return readback
      if attempt != 2: time.sleep(0.001)
    raise RuntimeError(f"USB PCIe bridge {primary} bus-number readback {readback:#x} != {buses:#x}")

  def pci_discover_usb_gpu(self, usb, max_bus=8):
    """Number one bridge at a time and stop before writing endpoint config as bridge config."""
    for bus in range(max_bus):
      next_bus = bus + 1
      buses = bus | next_bus << 8 | max_bus << 16
      self.pci_set_usb_bridge_buses(usb, bus, next_bus, max_bus)
      vid_did = usb.pcie_cfg_req(pci.PCI_VENDOR_ID, bus=next_bus, size=4)
      vendor, device = vid_did & 0xffff, vid_did >> 16
      header_type = usb.pcie_cfg_req(pci.PCI_HEADER_TYPE, bus=next_bus, size=1) & 0x7f
      if DEBUG >= 1: print(f"USB PCIe {bus} -> {next_bus}: {vendor:04x}:{device:04x} header={header_type}")
      usb_trace(f"topology bridge={bus}:00.0 downstream={next_bus}:00.0 id={vendor:04x}:{device:04x} header={header_type}")
      if vendor == 0x10de: return next_bus
      if header_type != 1:
        raise RuntimeError(f"USB PCIe endpoint {next_bus}:00.0 is {vendor:04x}:{device:04x}, expected NVIDIA")
    raise RuntimeError(f"NVIDIA endpoint not found within USB PCIe buses 1..{max_bus}")

  def pci_setup_usb_bars(self, usb, gpu_bus, mem_base, pref_mem_base):
    for bus in range(gpu_bus):
      self.pci_set_usb_bridge_buses(usb, bus, bus + 1, gpu_bus)
      usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, value=(mem_base >> 16) & 0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, value=0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, value=(pref_mem_base >> 16) & 0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, value=0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_BASE_UPPER32, bus=bus, value=pref_mem_base >> 32, size=4)
      usb.pcie_cfg_req(pci.PCI_PREF_LIMIT_UPPER32, bus=bus, value=0xffffffff, size=4)
      usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus,
                       value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

    cap_ptr = 0x100
    while cap_ptr:
      hdr = usb.pcie_cfg_req(cap_ptr, bus=gpu_bus, size=4)
      if pci.PCI_EXT_CAP_ID(hdr) == pci.PCI_EXT_CAP_ID_REBAR:
        cap = usb.pcie_cfg_req(cap_ptr + 0x04, bus=gpu_bus, size=4)
        ctrl = usb.pcie_cfg_req(cap_ptr + 0x08, bus=gpu_bus, size=4)
        usb.pcie_cfg_req(cap_ptr + 0x08, bus=gpu_bus, value=(ctrl & ~0x1f00) | ((int(cap >> 4).bit_length() - 1) << 8), size=4)
      cap_ptr = pci.PCI_EXT_CAP_NEXT(hdr)

    mem_space_addr, bar_off, bars = [mem_base, pref_mem_base], 0, {}
    while bar_off < 24:
      cfg = usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, size=4)
      bar_mem, bar_64 = bool(cfg & pci.PCI_BASE_ADDRESS_MEM_PREFETCH), cfg & pci.PCI_BASE_ADDRESS_MEM_TYPE_64
      if (cfg & pci.PCI_BASE_ADDRESS_SPACE) == pci.PCI_BASE_ADDRESS_SPACE_MEMORY:
        usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, value=0xffffffff, size=4)
        lo = usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, size=4) & 0xfffffff0
        if bar_64: usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, value=0xffffffff, size=4)
        hi = usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, size=4) if bar_64 else 0
        bar_size = ((~(((hi << 32) | lo) & ~0xf)) + 1) & (0xffffffffffffffff if bar_64 else 0xffffffff)
        if bar_size:
          usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, value=mem_space_addr[bar_mem] & 0xffffffff, size=4)
          if bar_64: usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, value=mem_space_addr[bar_mem] >> 32, size=4)
          bars[bar_off // 4] = (mem_space_addr[bar_mem], bar_size)
          mem_space_addr[bar_mem] += round_up(bar_size, 2 << 20)
      bar_off += 8 if bar_64 else 4

    usb.pcie_cfg_req(pci.PCI_COMMAND, bus=gpu_bus,
                     value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)
    usb_trace(f"BAR setup gpu_bus={gpu_bus} " + " ".join(
      f"BAR{idx}={addr:#x}/{size:#x}" for idx, (addr, size) in sorted(bars.items())))
    return bars

  def reserve_va(self, va_start, va_size):
    FileIOInterface.anon_mmap(va_start, va_size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, 0)

  def flock_acquire(self, name):
    import fcntl as _f
    lock_name = temp(name)
    if os.path.exists(lock_name):
      lock_fd = os.open(lock_name, os.O_RDWR)
    else:
      os.umask(0)
      lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try: _f.flock(lock_fd, _f.LOCK_EX | _f.LOCK_NB)
    except OSError: raise RuntimeError(
      f"Failed to acquire lock file {name}. Only one eGPU client (add_tiny/add_middle/TinyGPU) may run at a time. "
      f"`lsof {lock_name}` shows which process holds it.")
    return lock_fd

System = _System()

def _usb_check(rc, name):
  if rc < 0:
    err = ctypes.string_at(libusb.libusb_strerror(rc)).decode("utf-8", errors="replace")
    raise RuntimeError(f"{name}: {err} ({rc})")
  return rc

class USB3:
  """Minimal libusb transport used by the Chestnut PCIe controller."""
  @staticmethod
  @functools.cache
  def ctx():
    ctx = ctypes.POINTER(libusb.struct_libusb_context)()
    _usb_check(libusb.libusb_init(ctypes.byref(ctx)), "libusb_init")
    return ctx

  @classmethod
  def list_devices(cls, vendor, product):
    devs = ctypes.POINTER(ctypes.POINTER(libusb.struct_libusb_device))()
    count, found = _usb_check(libusb.libusb_get_device_list(cls.ctx(), ctypes.byref(devs)), "libusb_get_device_list"), []
    try:
      for i in range(count):
        desc = libusb.struct_libusb_device_descriptor()
        _usb_check(libusb.libusb_get_device_descriptor(devs[i], ctypes.byref(desc)), "libusb_get_device_descriptor")
        if (desc.idVendor, desc.idProduct) == (vendor, product):
          dev = libusb.libusb_ref_device(devs[i])
          found.append((dev, f"usb:{libusb.libusb_get_bus_number(dev)}-{libusb.libusb_get_device_address(dev)}"))
    finally:
      libusb.libusb_free_device_list(devs, 1)
    return found

  def __init__(self, dev):
    self.dev, self.handle, self.claimed, self.closed = dev, ctypes.POINTER(libusb.struct_libusb_device_handle)(), False, False
    try:
      _usb_check(libusb.libusb_open(dev, ctypes.byref(self.handle)), "libusb_open")
      auto_detach = libusb.libusb_set_auto_detach_kernel_driver(self.handle, 1)
      if auto_detach not in (0, libusb.LIBUSB_ERROR_NOT_SUPPORTED): _usb_check(auto_detach, "libusb_set_auto_detach_kernel_driver")
      active = libusb.libusb_kernel_driver_active(self.handle, 0)
      if active > 0: _usb_check(libusb.libusb_detach_kernel_driver(self.handle, 0), "libusb_detach_kernel_driver")
      elif active < 0 and active != libusb.LIBUSB_ERROR_NOT_SUPPORTED: _usb_check(active, "libusb_kernel_driver_active")
      config = ctypes.c_int()
      _usb_check(libusb.libusb_get_configuration(self.handle, ctypes.byref(config)), "libusb_get_configuration")
      if config.value != 1: _usb_check(libusb.libusb_set_configuration(self.handle, 1), "libusb_set_configuration")
      _usb_check(libusb.libusb_claim_interface(self.handle, 0), "libusb_claim_interface")
      self.claimed = True
      _usb_check(libusb.libusb_set_interface_alt_setting(self.handle, 0, 0), "libusb_set_interface_alt_setting")
    except Exception:
      self.close()
      raise
    atexit.register(self.close)

  def describe(self):
    desc = libusb.struct_libusb_device_descriptor()
    _usb_check(libusb.libusb_get_device_descriptor(self.dev, ctypes.byref(desc)), "libusb_get_device_descriptor")
    speed = libusb.libusb_get_device_speed(self.dev)
    speed_name = {0: "unknown", 1: "1.5Mb/s", 2: "12Mb/s", 3: "480Mb/s", 4: "5Gb/s", 5: "10Gb/s"}.get(speed, f"code{speed}")
    return (f"{desc.idVendor:04x}:{desc.idProduct:04x} usb:{libusb.libusb_get_bus_number(self.dev)}-"
            f"{libusb.libusb_get_device_address(self.dev)} speed={speed_name} cfg=1 if=0 alt=0 ep_out=0x02 ep_in=0x81")

  def close(self):
    if self.closed: return
    self.closed = True
    if self.claimed and self.handle:
      with contextlib.suppress(Exception): libusb.libusb_release_interface(self.handle, 0)
    if self.handle:
      with contextlib.suppress(Exception): libusb.libusb_close(self.handle)
    if self.dev:
      with contextlib.suppress(Exception): libusb.libusb_unref_device(self.dev)

  def control_write(self, request, value=0, index=0, data=b'', timeout=1000):
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data) if data else None
    rc = _usb_check(libusb.libusb_control_transfer(self.handle, 0x40, request, value, index, buf, len(data), timeout),
                    f"control OUT 0x{request:02x}")
    if rc != len(data): raise RuntimeError(f"control OUT 0x{request:02x}: short write {rc}/{len(data)}")

  def control_read(self, request, length, value=0, index=0, timeout=1000):
    buf = (ctypes.c_ubyte * length)()
    rc = _usb_check(libusb.libusb_control_transfer(self.handle, 0xc0, request, value, index, buf, length, timeout),
                    f"control IN 0x{request:02x}")
    if rc != length: raise RuntimeError(f"control IN 0x{request:02x}: short read {rc}/{length}")
    return bytes(buf)

  def bulk_write(self, payload, timeout=30000):
    payload = bytes(payload)
    buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
    transferred = ctypes.c_int()
    _usb_check(libusb.libusb_bulk_transfer(self.handle, 0x02, buf, len(payload), ctypes.byref(transferred), timeout), "bulk OUT 0x02")
    if transferred.value != len(payload): raise RuntimeError(f"bulk OUT 0x02: short write {transferred.value}/{len(payload)}")

  def bulk_read(self, length, timeout=30000):
    buf, transferred = (ctypes.c_ubyte * length)(), ctypes.c_int()
    _usb_check(libusb.libusb_bulk_transfer(self.handle, 0x81, buf, length, ctypes.byref(transferred), timeout), "bulk IN 0x81")
    if transferred.value != length: raise RuntimeError(f"bulk IN 0x81: short read {transferred.value}/{length}")
    return bytes(buf)

class CustomASM24Controller:
  def __init__(self, usb, require_link=True):
    self.usb = usb
    self.stream_chunk = 512 if libusb.libusb_get_device_speed(usb.dev) <= libusb.LIBUSB_SPEED_HIGH else 1024
    if not require_link: return
    if self.read(0xb450, 1)[0] != 0x78: self.set_pcie_power(True)
    if (ltssm := self.read(0xb450, 1)[0]) != 0x78:
      raise RuntimeError(f"Chestnut PCIe link not up (LTSSM=0x{ltssm:02x})")

  def set_pcie_power(self, enabled, timeout=10000): self.usb.control_write(0xf3, value=int(enabled), timeout=timeout)

  def _f0_out(self, fmt_type, byte_en, address, value, mode=0):
    self.usb.control_write(0xf0, fmt_type | byte_en << 8, mode & 3,
                           struct.pack('<III', address & 0xffffffff, address >> 32, value), 5000)

  def _f0_in(self):
    data = self.usb.control_read(0xf0, 8, timeout=5000)
    return struct.unpack_from('<I', data)[0], (data[4] >> 5) & 7, data[7]

  def pcie_request(self, fmt_type, address, value=None, size=4, retries=10):
    assert 0 < size <= 4
    offset = address & 3
    self._f0_out(fmt_type, ((1 << size) - 1) << offset, address & ~3, (value << (8 * offset)) if value is not None else 0)
    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return None
    data, cpl_status, ret_status = self._f0_in()
    if ret_status:
      if retries:
        time.sleep(0.001)
        return self.pcie_request(fmt_type, address, value, size, retries - 1)
      raise RuntimeError(f"PCIe request failed: status={ret_status} address={address:#x}")
    if cpl_status:
      status = {1: "Unsupported Request", 2: "Config Retry", 4: "Completer Abort"}.get(cpl_status, f"reserved {cpl_status:#x}")
      raise RuntimeError(f"PCIe completion failed: {status}, address={address:#x}")
    return (data >> (8 * offset)) & ((1 << (8 * size)) - 1) if value is None else None

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    assert byte_addr >> 12 == 0 and bus >> 8 == 0 and dev >> 5 == 0 and fn >> 3 == 0
    return self.pcie_request((0x44 if value is not None else 0x04) | int(bus > 0),
                             bus << 24 | dev << 19 | fn << 16 | (byte_addr & 0xfff), value, size)

  def pcie_mem_write(self, address, data):
    if not data: return
    data = bytes(data)
    if len(data) % 4: raise ValueError(f"PCIe writes must be dword aligned, got {len(data)} bytes")
    for off in range(0, len(data), self.stream_chunk):
      chunk = data[off:off + self.stream_chunk]
      chunk_addr = address + off
      # ed4e39b7-CLEAN's streaming handler retains the last single-TLP
      # address/type. Prime it with an idempotent first-dword write, then
      # stream the complete chunk (including that dword) from the same address.
      self.pcie_request(0x60 if chunk_addr >> 32 else 0x40, chunk_addr,
                        value=int.from_bytes(chunk[:4], "little"), size=4)
      if len(chunk) == 4: continue
      self._f0_out(0x60 if chunk_addr >> 32 else 0x40, 0x0f, chunk_addr, len(chunk) // 4, mode=1)
      self.usb.bulk_write(chunk)

  def pcie_mem_read(self, address, nbytes):
    if nbytes % 4: raise ValueError(f"PCIe reads must be dword aligned, got {nbytes} bytes")
    data = bytearray()
    for off in range(0, nbytes, self.stream_chunk):
      chunk_size, chunk_addr = min(self.stream_chunk, nbytes - off), address + off
      # Prime both the address and TLP type with a completion-checked read.
      first = self.pcie_request(0x20 if chunk_addr >> 32 else 0x00, chunk_addr, size=4)
      if chunk_size == 4:
        data += int(first).to_bytes(4, "little")
        continue
      self._f0_out(0x20 if chunk_addr >> 32 else 0x00, 0x0f, chunk_addr, chunk_size // 4, mode=2)
      data += self.usb.bulk_read(chunk_size)
    return bytes(data)

  def read(self, base_addr, length):
    return b''.join(self.usb.control_read(0xe4, min(0xff, length - off), value=base_addr + off)
                    for off in range(0, length, 0xff))

  def write(self, base_addr, data):
    for off, val in enumerate(data): self.usb.control_write(0xe5, value=base_addr + off, index=val)

  def scsi_write(self, data, slot_start=0):
    data = bytes(data)
    padded = data + bytes(round_up(len(data), 512) - len(data))
    slots = ceildiv(len(padded), 0x4000)
    if slot_start + slots > 32: raise ValueError(f"SRAM write exceeds 32 slots: start={slot_start}, slots={slots}")
    self.usb.control_write(0xf2, value=len(padded) // 512, index=(slot_start & 0xff) | slots << 8)
    self.usb.bulk_write(padded)

  def scsi_read_arm(self, size):
    self.usb.control_write(0xf2, value=ceildiv(size, 512) | 0x8000, index=(ceildiv(size, 0x4000) & 0xff) << 8)

  def scsi_read(self, size): return self.usb.bulk_read(round_up(size, 512), timeout=10000)[:size]

class USBMMIOInterface:
  def __init__(self, usb, addr, nbytes, fmt='B', pcimem=True, sram_start_slot=0):
    self.usb, self.addr, self.nbytes, self.fmt, self.el_sz, self.pcimem = usb, addr, nbytes, fmt, struct.calcsize(fmt), pcimem
    self.sram_start_slot = sram_start_slot
  def __len__(self): return self.nbytes // self.el_sz
  def _range(self, index):
    if isinstance(index, slice):
      if index.step not in (None, 1): raise ValueError("strided USB MMIO slices are not supported")
      start, stop = index.start or 0, len(self) if index.stop is None else index.stop
      return start * self.el_sz, (stop - start) * self.el_sz
    return index * self.el_sz, self.el_sz
  def __getitem__(self, index):
    off, size = self._range(index)
    if self.pcimem:
      start, end = self.addr + off, self.addr + off + size
      aligned_start, aligned_end = start & ~0x3, round_up(end, 4)
      data = self.usb.pcie_mem_read(aligned_start, aligned_end - aligned_start)[start - aligned_start:end - aligned_start]
    elif self.addr == 0xf000: data = self.usb.scsi_read(size)
    else: data = self.usb.read(self.addr + off, size)
    if isinstance(index, slice): return data if self.fmt == 'B' else memoryview(data).cast(self.fmt).tolist()
    return int.from_bytes(data, "little")
  def __setitem__(self, index, data):
    off, size = self._range(index)
    if isinstance(data, int): raw = struct.pack(self.fmt, data)
    elif self.fmt != 'B' and isinstance(data, (list, tuple)): raw = struct.pack(f'<{len(data)}{self.fmt}', *data)
    else: raw = bytes(data)
    if len(raw) != size: raise ValueError(f"USB MMIO write size mismatch: {len(raw)} != {size}")
    if self.pcimem:
      start, end = self.addr + off, self.addr + off + len(raw)
      aligned_start, aligned_end = start & ~0x3, round_up(end, 4)
      if start == aligned_start and end == aligned_end: aligned = raw
      else:
        aligned = bytearray(self.usb.pcie_mem_read(aligned_start, aligned_end - aligned_start))
        aligned[start - aligned_start:end - aligned_start] = raw
      self.usb.pcie_mem_write(aligned_start, aligned)
    elif self.addr == 0xf000: self.usb.scsi_write(raw, slot_start=self.sram_start_slot)
    else: self.usb.write(self.addr + off, raw)
  def view(self, offset=0, size=None, fmt=None):
    return USBMMIOInterface(self.usb, self.addr + offset, self.nbytes - offset if size is None else size,
                            fmt or self.fmt, self.pcimem, self.sram_start_slot)

class ASM24GSPQueueInterface:
  """Logical GSP queue pages aliased onto Chestnut SRAM and controller XDATA."""
  PAGE_SIZE, SLOT_SIZE, SRAM_SIZE = 0x1000, 0x4000, 0x80000
  PAGE_PADDRS = (0x213000, 0x27F000, 0x27B000, 0x27C000, 0x27D000, 0x27E000,
                 0x828000, 0x820000, 0x200000, 0x820000, 0x200000)
  def __init__(self, usb, size, fmt='B', offset=0, root=None):
    self.usb, self.offset, self.nbytes, self.fmt, self.el_sz = usb, offset, size, fmt, struct.calcsize(fmt)
    self._mirror = bytearray(self.SRAM_SIZE) if root is None else root._mirror
  def __len__(self): return self.nbytes // self.el_sz
  def _range(self, index):
    if isinstance(index, slice):
      if index.step not in (None, 1): raise ValueError("strided GSP queue slices are not supported")
      start, stop = index.start or 0, len(self) if index.stop is None else index.stop
      return start * self.el_sz, (stop - start) * self.el_sz
    return index * self.el_sz, self.el_sz
  def _page_mapping(self, logical_page):
    paddr = self.PAGE_PADDRS[logical_page]
    if paddr == 0x200000: return "xdata", 0xF000
    if 0x200000 <= paddr < 0x280000: return "sram", paddr - 0x200000
    return "xdata", {0x820000: 0xA000, 0x828000: 0xB800}[paddr]
  def _pieces(self, offset, size):
    end = offset + size
    while offset < end:
      page, page_off = divmod(offset, self.PAGE_SIZE)
      chunk = min(end - offset, self.PAGE_SIZE - page_off)
      kind, mapped = self._page_mapping(page)
      yield kind, mapped + page_off, chunk
      offset += chunk
  def __getitem__(self, index):
    off, size = self._range(index)
    out = bytearray()
    for kind, mapped, chunk in self._pieces(self.offset + off, size):
      out += self.usb.read(mapped, chunk) if kind == "xdata" else self._mirror[mapped:mapped + chunk]
    if isinstance(index, slice): return bytes(out) if self.fmt == 'B' else memoryview(out).cast(self.fmt).tolist()
    return int.from_bytes(out, "little")
  def __setitem__(self, index, data):
    off, size = self._range(index)
    if isinstance(data, int): raw = struct.pack(self.fmt, data)
    elif self.fmt != 'B' and isinstance(data, (list, tuple, _array_mod.array)): raw = struct.pack(f'<{len(data)}{self.fmt}', *data)
    else: raw = bytes(data)
    if len(raw) != size: raise ValueError(f"GSP queue write size mismatch: {len(raw)} != {size}")
    dirty_slots, pos = {}, 0
    for kind, mapped, chunk in self._pieces(self.offset + off, size):
      if kind == "xdata": self.usb.write(mapped, raw[pos:pos + chunk])
      else:
        self._mirror[mapped:mapped + chunk] = raw[pos:pos + chunk]
        for slot in range(mapped // self.SLOT_SIZE, ceildiv(mapped + chunk, self.SLOT_SIZE)):
          dirty_slots[slot] = max(dirty_slots.get(slot, 0), min(mapped + chunk, (slot + 1) * self.SLOT_SIZE))
      pos += chunk
    for slot, hi in sorted(dirty_slots.items()):
      slot_base = slot * self.SLOT_SIZE
      self.usb.scsi_write(bytes(self._mirror[slot_base:round_up(hi, 512)]), slot_start=slot)
  def view(self, offset=0, size=None, fmt=None):
    return ASM24GSPQueueInterface(self.usb, self.nbytes - offset if size is None else size,
                                  fmt or self.fmt, self.offset + offset, self)

class USBPCIDevice:
  """Chestnut-backed PCI device using F0 for config/BAR traffic and F2 for SRAM."""
  def __init__(self, devpref, dev, pcibus):
    self.pcibus, self.peer_group = pcibus, f"USBPCIDevice_{pcibus}"
    stage_set(1, "Chestnut USB3 transport up")
    try: self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    except Exception:
      libusb.libusb_unref_device(dev)
      raise
    transport = None
    try:
      transport = USB3(dev)
      self.usb = CustomASM24Controller(transport)
      usb_trace(f"transport {transport.describe()} stream_chunk={self.usb.stream_chunk:#x}")
      # Expose the controller's two additional PCIe-visible XDATA DMA windows.
      self.usb.write(0xB267, b'\x08')
      self.usb.write(0xB26F, b'\x28')
      usb_trace("controller DMA windows SRAM=0x200000/0x80000 sys_buf=0x820000/0x1000 cq_buf=0x828000/0x1000")
      self._gsp_args = {}
      self.gpu_bus_override = os.environ.get("NV_USB_GPU_BUS")
      self._setup_usb_bars()
      stage_done(f"usb3 F0/F2, NVIDIA bus {self.gpu_bus}")
    except Exception:
      if transport is not None: transport.close()
      os.close(self.lock_fd)
      self.lock_fd = None
      raise
    atexit.register(self.close)

  def function_level_reset(self):
    """Clear stale NVIDIA function/WPR state without cycling Chestnut power."""
    cap = 0x78
    command = self.read_config(pci.PCI_COMMAND, 2)
    bars = [(off, self.read_config(off, 4)) for off in (0x10, 0x18, 0x1c, 0x20)]
    devctl = self.read_config(cap + pci.PCI_EXP_DEVCTL, 2)
    usb_trace(f"GPU FLR begin command={command:#06x} devctl={devctl:#06x} preserved_bars=" +
              ",".join(f"{off:#x}:{value:#010x}" for off, value in bars))
    self.write_config_flush(pci.PCI_COMMAND, command & ~pci.PCI_COMMAND_MASTER, 2)
    self.write_config(cap + pci.PCI_EXP_DEVCTL, devctl | pci.PCI_EXP_DEVCTL_BCR_FLR, 2)
    time.sleep(0.1)
    for off, value in bars: self.write_config(off, value, 4)
    self.write_config(cap + pci.PCI_EXP_DEVCTL, devctl, 2)
    self.write_config_flush(pci.PCI_COMMAND, command, 2)
    usb_trace("GPU FLR complete; BARs and PCI_COMMAND restored")

  def dma_view(self, ctrl_addr, size, start_slot=0):
    return USBMMIOInterface(self.usb, ctrl_addr, size, pcimem=False, sram_start_slot=start_slot)
  def alloc_gsp_queues(self, size):
    self.gsp_queues = ASM24GSPQueueInterface(self.usb, size)
    return self.gsp_queues, list(self.gsp_queues.PAGE_PADDRS)
  def stage_gsp_args(self, data, offset):
    page = bytes(data) + bytes(0x100 - len(data))
    self._gsp_args[offset] = page
    self.usb.write(0xB800 + offset, page)
    return 0x828000 + offset
  def retrain_pcie(self, generation):
    links = (("bridge", self.gpu_bus - 1, 0x80), ("gpu", self.gpu_bus, 0x78))
    before = []
    for role, bus, cap in links:
      ctl2 = self.usb.pcie_cfg_req(cap + 0x30, bus=bus, size=2)
      link_status = self.usb.pcie_cfg_req(cap + 0x12, bus=bus, size=2)
      before.append((role, bus, cap, ctl2 & 0xf, link_status))
      self.usb.pcie_cfg_req(cap + 0x30, bus=bus, value=(ctl2 & ~0xF) | generation, size=2)
      linkctl = self.usb.pcie_cfg_req(cap + 0x10, bus=bus, size=2)
      self.usb.pcie_cfg_req(cap + 0x10, bus=bus, value=linkctl & ~0x3, size=2)
    linkctl = self.usb.pcie_cfg_req(0x90, bus=self.gpu_bus - 1, size=2)
    self.usb.pcie_cfg_req(0x90, bus=self.gpu_bus - 1, value=linkctl | 0x20, size=2)
    time.sleep(0.1)
    usb_retrain_report(generation, [(*row, self.usb.pcie_cfg_req(cap + 0x12, bus=bus, size=2))
                                    for row, (_, bus, cap) in zip(before, links)])
  def stream_gsp_boot(self, image, launched_at):
    ring_page, ring_pages, batch_pages = 44, 84, 28
    ring_size, batch_size = ring_pages * 0x1000, batch_pages * 0x1000
    logical_bytes = max(0, len(image) - ring_size)
    batch_count = ceildiv(logical_bytes, batch_size)
    slots = [ring_page // 4 + i * batch_pages // 4 for i in range(ring_pages // batch_pages)]
    last_bulk_at = launched_at
    for i, off in enumerate(range(ring_size, len(image), batch_size)):
      deadline = launched_at + 0.003 + (off - ring_size) / ring_size * 0.0014
      if i == 0:
        while time.perf_counter() < deadline: pass
      slot = ring_page // 4 + i % (ring_pages // batch_pages) * batch_pages // 4
      self.usb.usb.control_write(0xF2, value=batch_size // 512, index=slot | (batch_pages // 4 << 8))
      while time.perf_counter() < deadline: pass
      payload = bytes(image[off:off + batch_size])
      self.usb.usb.bulk_write(payload.ljust(batch_size, b'\x00'))
    last_bulk_at = time.perf_counter()
    while time.perf_counter() < launched_at + 0.270: pass
    queue_bytes = len(self.gsp_queues._mirror)
    self.usb.scsi_write(bytes(self.gsp_queues._mirror))
    for offset, page in self._gsp_args.items(): self.usb.write(0xB800 + offset, page)
    restored_at = time.perf_counter()
    usb_stream_report(image_size=len(image), ring_size=ring_size, batch_size=batch_size,
      logical_bytes=logical_bytes, wire_bytes=batch_count * batch_size, batch_count=batch_count, slots=slots,
      launched_at=launched_at, last_bulk_at=last_bulk_at, restored_at=restored_at,
      queue_bytes=queue_bytes, arg_pages=len(self._gsp_args))
  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    raise RuntimeError("Chestnut has no general-purpose host sysmem; its SRAM aperture is reserved for staged transfers")
  def close(self):
    if hasattr(self, 'usb'): self.usb.usb.close()
    if self.lock_fd is not None:
      os.close(self.lock_fd)
      self.lock_fd = None
  def reset(self):
    self.usb.set_pcie_power(False)
    time.sleep(0.1)
    self.usb.set_pcie_power(True)
    self._setup_usb_bars()
  def _setup_usb_bars(self):
    self.gpu_bus = int(self.gpu_bus_override, 0) if self.gpu_bus_override is not None else System.pci_discover_usb_gpu(self.usb)
    usb_trace(f"GPU bus selection source={'NV_USB_GPU_BUS' if self.gpu_bus_override is not None else 'discovery'} bus={self.gpu_bus}")
    self._bar_info = System.pci_setup_usb_bars(self.usb, self.gpu_bus, 0x10000000, 32 << 30)
  def read_config(self, offset, size): return self.usb.pcie_cfg_req(offset, bus=self.gpu_bus, size=size)
  def write_config(self, offset, value, size): self.usb.pcie_cfg_req(offset, bus=self.gpu_bus, value=value, size=size)
  def write_config_flush(self, offset, value, size):
    self.write_config(offset, value, size)
    return self.read_config(offset, size)
  def bar_info(self, bar_idx): return self._bar_info[bar_idx]
  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'):
    return USBMMIOInterface(self.usb, self.bar_info(bar)[0] + off, size or (self.bar_info(bar)[1] - off), fmt)
  def resize_bar(self, bar_idx): pass

class RemoteCmd(enum.IntEnum):
  """TinyGPU RPC command IDs for remote PCIe BAR/config/MMIO operations."""
  PROBE, MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE, MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE, RESIZE_BAR, PING = range(13)

class RemoteMMIOInterface(MMIOInterface):
  def __init__(self, dev, residx, nbytes, fmt='B', off=0, rd_cmd=RemoteCmd.MMIO_READ, wr_cmd=RemoteCmd.MMIO_WRITE):
    self.dev, self.residx, self.nbytes, self.fmt, self.off, self.el_sz = dev, residx, nbytes, fmt, off, struct.calcsize(fmt)
    self.rd_cmd, self.wr_cmd = rd_cmd, wr_cmd
  def __getitem__(self, index):
    sl = index if isinstance(index, slice) else slice(index, index + 1)
    start, stop = (sl.start or 0) * self.el_sz, (sl.stop or len(self)) * self.el_sz
    data = self.dev._bulk_read(self.rd_cmd, self.residx, self.off + start, stop - start)
    result = data if self.fmt == 'B' else list(struct.unpack(f'<{(stop - start) // self.el_sz}{self.fmt}', data))
    return result if isinstance(index, slice) else result[0]
  def __setitem__(self, index, val):
    start = (index.start or 0) * self.el_sz if isinstance(index, slice) else index * self.el_sz
    if self.fmt == 'B':
      data = bytes(val) if isinstance(val, (bytes, bytearray, memoryview)) else (bytes(val) if isinstance(val, (list, tuple)) else bytes([val]))
      if not isinstance(index, slice): data = data[:self.el_sz]
    elif isinstance(index, slice):
      data = struct.pack(f'<{len(val)}{self.fmt}', *val)
    else:
      data = struct.pack(f'<{self.fmt}', val)
    self.dev._bulk_write(self.wr_cmd, self.residx, self.off + start, data)
  def view(self, offset=0, size=None, fmt=None):
    return RemoteMMIOInterface(self.dev, self.residx, size or (self.nbytes - offset), fmt or self.fmt,
      self.off + offset, self.rd_cmd, self.wr_cmd)

class RemotePCIDevice:
  """Remote PCI device over Unix socket: BAR/config/MMIO/sysmem RPC."""
  def __init__(self, devpref, pcibus, sock):
    self.sock, self.pcibus, self.dev_id = sock, pcibus, int(pcibus.split(':')[-1]) if ':' in pcibus else 0
    for buft in [socket.SO_SNDBUF, socket.SO_RCVBUF]: self.sock.setsockopt(socket.SOL_SOCKET, buft, 64 << 20)
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")

  @staticmethod
  def _recvall(sock, n):
    data = b''
    while len(data) < n and (chunk := sock.recv(n - len(data))): data += chunk
    if len(data) < n: raise RuntimeError("Connection closed")
    return data
  @staticmethod
  def _rpc(sock, dev_id, cmd, *args, bar=0, readout_size=0, payload=b'', has_fd=False):
    # MMIO_READ/WRITE are traced in _bulk_read/_bulk_write (they carry bar/off/data cleanly).
    global _TRACE_BULK_CMDS
    if _TRACE_BULK_CMDS is None: _TRACE_BULK_CMDS = {int(RemoteCmd.MMIO_READ), int(RemoteCmd.MMIO_WRITE)}
    if TRACE and cmd not in _TRACE_BULK_CMDS:
      if cmd == int(RemoteCmd.RESET):
        _trace("-->", f"RESET dev={dev_id}")
      elif cmd in (int(RemoteCmd.MAP_SYSMEM_FD), int(RemoteCmd.MAP_SYSMEM)):
        pass   # not a BAR access: semantic SYSMEM_ALLOC line printed after the response
      elif cmd in (int(RemoteCmd.CFG_READ), int(RemoteCmd.CFG_WRITE)):
        # Raw config traffic is verbose-only (NV_TRACE_RAW=1); semantic
        # transitions (before -> after, verified) print at the call site.
        if TRACE_RAW:
          off, sz = args[0], args[1]
          nm = _trace_pci_name(off)
          tgt = f"{nm}[{off:#04x}]" if nm else f"cfg[{off:#04x}] size={sz}"
          lbl = "CFG_READ " if cmd == int(RemoteCmd.CFG_READ) else "CFG_WRITE"
          if cmd == int(RemoteCmd.CFG_WRITE):
            val = args[2]
            dec = f" {fmt_pci_command(val)}" if off == pci.PCI_COMMAND else ""
            _trace("-->", f"{lbl} dev={dev_id} {tgt} <- 0x{val:0{2 * sz}x}{dec}")
          else:
            _trace("-->", f"{lbl} dev={dev_id} {tgt}")
      else:
        _trace("-->", f"{_trace_cmd_name(cmd)} dev={dev_id} bar={bar} args={[hex(a) for a in args]} len={len(payload):#x}", bytes(payload))
    sock.sendall(struct.pack('<BIIQQQ', cmd, dev_id, bar, *(*args, 0, 0, 0)[:3]) + payload)
    if has_fd:
      msg, anc, _, _ = sock.recvmsg(17, socket.CMSG_LEN(4))
      fd = struct.unpack('<i', anc[0][2][:4])[0]
    else: msg, fd = RemotePCIDevice._recvall(sock, 17), None
    if (resp := struct.unpack('<BQQ', msg))[0] != 0:
      raise RuntimeError(f"RPC failed: {RemotePCIDevice._recvall(sock, resp[1]).decode('utf-8') if resp[1] > 0 else 'unknown error'}")
    readout = RemotePCIDevice._recvall(sock, readout_size) if readout_size > 0 else None
    if TLP_PATH:
      if cmd == RemoteCmd.CFG_READ and readout is not None:
        tlp(f"CFG_RD off={args[0]:#x} size={readout_size} -> " + _fmt_data(readout))
      elif cmd == RemoteCmd.CFG_WRITE:
        tlp(f"CFG_WR off={args[0]:#x} size={args[1]} data=0x{args[2]:x}")
      elif cmd == RemoteCmd.RESET: tlp("RESET")
      elif cmd in (RemoteCmd.MAP_SYSMEM_FD, RemoteCmd.MAP_SYSMEM): tlp(f"SYSMEM_MAP size={args[0]:#x} (fd-passed, not replayable)")
    if TRACE and cmd == int(RemoteCmd.RESET):
      _trace("<--", "RESET done")
    if TRACE and cmd in (int(RemoteCmd.MAP_SYSMEM_FD), int(RemoteCmd.MAP_SYSMEM)):
      req, mapped = args[0], resp[1]
      cont = bool(args[1]) if len(args) > 1 else False
      ok = "✓" if mapped >= req else "✗"
      _trace("SYSMEM", f"ALLOC requested={req:#x}{hsz(req)} contiguous={'yes' if cont else 'no'} -> "
             f"mapped={mapped:#x}{hsz(mapped)} {ok}")
    if TRACE and TRACE_RAW and cmd == int(RemoteCmd.CFG_READ):
      # Config read value rides in the RPC header (resp[1]); decode PCI_COMMAND.
      off, sz, val = args[0], args[1], resp[1]
      nm = _trace_pci_name(off)
      tgt = f"{nm}[{off:#04x}]" if nm else f"cfg[{off:#04x}] size={sz}"
      dec = f" {fmt_pci_command(val)}" if off == pci.PCI_COMMAND else ""
      _trace("<--", f"CFG_READ  dev={dev_id} {tgt} -> 0x{val:0{2 * sz}x}{dec}")
    if TRACE and readout is not None and cmd not in _TRACE_BULK_CMDS:
      _trace("<--", f"{_trace_cmd_name(cmd)} got {len(readout):#x}", readout)
    return (resp[1], resp[2]) + (readout,) + (fd,)

  def _bulk_read(self, cmd, idx, offset, size):
    data = unwrap(self._rpc(self.sock, self.dev_id, cmd, offset, size, bar=idx, readout_size=size)[2])
    if TLP_PATH: tlp(f"{_trace_cmd_name(cmd)} bar{idx} off={offset:#x} size={size} -> " + _fmt_data(data))
    if TRACE and quiet_active():
      pass
    elif TRACE and agg_active():
      wire_ev("RD", idx, offset, len(data), data)
    elif TRACE and (not TRACE_NOZERO or data.strip(b"\x00")):
      # Unchanged poll reads (same addr, same data as last time) are noise too;
      # the value-change transition still prints.
      key = (idx, offset)
      prev_rd = _trace_last_wire_rd.get(key)
      prev_wr = _trace_last_wire_wr.get(key)
      if not (TRACE_NOZERO and (prev_rd == data or prev_wr == data)):
        # Large ROM reads get summarized (VBIOS) instead of hex-walls.
        vb = _trace_decode_vbios(data) if TRACE_NOZERO and len(data) >= (1 << 16) else None
        _trace_last_wire_rd[key] = data
        _trace("-->", f"{_trace_cmd_name(cmd)} bar{idx} off={offset:#x} len={size:#x}{hsz(size)}")
        # Known 4-byte registers: one side-by-side row (bytes | u32 | meaning),
        # no duplicate hexdump. Unknown regs keep the generic dump.
        row = falcon_boot_row(idx, offset, data[:4]) if idx == 0 and size == 4 else None
        if vb: _trace("<--", f"{_trace_cmd_name(cmd)} bar{idx} off={offset:#x} {vb} [{len(data):#x}{hsz(len(data))}]")
        elif row:
          nm, ln = row
          _trace("<--", f"{_trace_cmd_name(cmd)} bar{idx}+{offset:#08x} {nm}\n      {ln}", None)
        else: _trace("<--", f"{_trace_cmd_name(cmd)} bar{idx} off={offset:#x} len={len(data):#x}{hsz(len(data))}", data)
    return data
  def _bulk_write(self, cmd, idx, offset, data):
    if TLP_PATH: tlp(f"{_trace_cmd_name(cmd)} bar{idx} off={offset:#x} size={len(data)} data=" + _fmt_data(bytes(data)))
    if TRACE and quiet_active():
      pass   # loop semantics summarized by caller; raw stream lives in TLP
    elif TRACE and agg_active():
      wire_ev("WR", idx, offset, len(data), bytes(data))
    elif TRACE:
      blob = TRACE_NOZERO and len(data) >= (1 << 12)
      zfill = TRACE_NOZERO and not blob and len(data) >= (1 << 10) and not bytes(data).strip(b"\x00")
      if blob or zfill:
        kind = "zero-filled" if zfill else "binary upload"
        _trace("-->", f"{_trace_cmd_name(cmd)} bar{idx} off={offset:#x} len={len(data):#x}{hsz(len(data))} <{kind}>")
      elif idx == 0 and len(data) == 4 and (row := falcon_boot_write_row(idx, offset, bytes(data))):
        nm, ln = row
        if nm.startswith("PGSP_QUEUE_HEAD") and not TRACE_RAW:
          pass   # RM doorbell kick implied by the semantic RM record that follows
        else:
          # Known 4-byte registers: side-by-side row (bytes | u32 | meaning),
          # plus changed-bit transitions vs the last read of the same address.
          _trace("-->", f"{_trace_cmd_name(cmd)} bar{idx}+{offset:#08x} {nm}\n      {ln}", None)
      else:
        ann = ""
        if len(data) == 8 and _trace_caller() == "set_entry" and (pte := _trace_decode_pte(bytes(data))):
          ann = f"\n      {pte}"
        _trace("-->", f"{_trace_cmd_name(cmd)} bar{idx} off={offset:#x} len={len(data):#x}{hsz(len(data))}{ann}", bytes(data))
    if len(data) <= 8:
      _trace_last_wire_wr[(idx, offset)] = bytes(data)
    self.sock.sendall(struct.pack('<BIIQQQ', cmd, self.dev_id, idx, offset, len(data), 0) + data)

  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    mapped_size, _, _, fd = self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
    memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt='B')
    paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
    return memview, [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]

  def reset(self): self._rpc(self.sock, self.dev_id, RemoteCmd.RESET)
  def read_config(self, offset, size): return self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_READ, offset, size)[0]
  def write_config(self, offset, value, size): self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_WRITE, offset, size, value)
  def write_config_flush(self, offset, value, size):
    """Write then read back; returns the readback value so callers can verify."""
    self.write_config(offset, value, size)
    return self.read_config(offset, size)
  @functools.cache
  def bar_info(self, bar_idx): return self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_BAR, bar=bar_idx)[:2]
  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'):
    return RemoteMMIOInterface(self, bar, size or self.bar_info(bar)[1], fmt).view(off, size, fmt)
  def resize_bar(self, bar_idx): self._rpc(self.sock, self.dev_id, RemoteCmd.RESIZE_BAR, bar=bar_idx)

class APLRemotePCIDevice(RemotePCIDevice):
  """S1 — Apple Silicon eGPU transport: connects to TinyGPU.app via Unix socket."""
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"
  APP_COMMIT = "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"  # pinned commit

  def __init__(self, devpref, pcibus):
    sock_path = os.environ.get("APL_REMOTE_SOCK", temp("tinygpu.sock"))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connected = False
    for i in range(100):
      try:
        sock.connect(sock_path); connected = True; break
      except (ConnectionRefusedError, FileNotFoundError):
        if i == 0:
          subprocess.Popen([self.APP_PATH, "server", sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
    if not connected: raise RuntimeError(f"Failed to connect to TinyGPU server at {sock_path}")
    stage_set(1, "TinyGPU transport up")
    super().__init__(devpref, "usb4", sock=sock)
    stage_done("usb4 socket+lock")


# ============================================================================
# nv/nvdev.py (vendored from ref/tinygrad/tinygrad/runtime/support/nv/nvdev.py)
# Slimmed: NVReg, NVPageTableEntry, NVMemoryManager, NVDev (boot shell only).
# ============================================================================
NV_DEBUG = getenv("NV_DEBUG", 0)

class NVReg:
  """Named GPU register with base offset, lambda index, and bitfield decode."""
  def __init__(self, nvdev, base, off, fields=None):
    self.nvdev, self.base, self.off, self.fields = nvdev, base, off, fields or {}
  def __getitem__(self, idx): return NVReg(self.nvdev, self.base, self.off(idx), fields=self.fields)
  def with_base(self, base): return NVReg(self.nvdev, base + self.base, self.off, self.fields)
  def add_field(self, name, start, end): self.fields[name] = (start, end)
  def read(self): return self.nvdev.rreg(self.base + self.off)
  def read_bitfields(self): return self.decode(self.read())
  def write(self, _ini_val=0, **kwargs): self.nvdev.wreg(self.base + self.off, _ini_val | self.encode(**kwargs))
  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)
  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1] - self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)
  def encode(self, **kwargs):
    return functools.reduce(int.__or__, (value << self.fields[name][0] for name, value in kwargs.items()), 0)
  def decode(self, val):
    return {name: getbits(val, start, end) for name, (start, end) in self.fields.items()}

class NVPageTableEntry:
  """Single GMMU page table level: read/write PTE and dual-PDE entries."""
  def __init__(self, nvdev, paddr, lv):
    self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')

  def _is_dual_pde(self): return self.lv == self.nvdev.mm.level_cnt - 2

  def set_entry(self, entry_id, paddr, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    if not table:
      x = self.nvdev.pte_t.encode(valid=valid, address_sys=paddr >> 12,
        aperture=2 if aspace is AddrSpace.SYS else 0, kind=6,
        **({'pcf': int(uncached)} if self.nvdev.mmu_ver == 3 else {'vol': uncached}))
    else:
      pde = self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t
      small, sys = ("_small" if self._is_dual_pde() else ""), "" if self.nvdev.mmu_ver == 3 else "_sys"
      x = pde.encode(is_pte=False, **{f'aperture{small}': 1 if valid else 0, f'address{small}{sys}': paddr >> 12},
        **({f'pcf{small}': 0b10} if self.nvdev.mmu_ver == 3 else {'no_ats': 1}))
    if self._is_dual_pde(): self.entries[2 * entry_id], self.entries[2 * entry_id + 1] = x & 0xffffffffffffffff, x >> 64
    else: self.entries[entry_id] = x

  def entry(self, entry_id):
    return (self.entries[2 * entry_id + 1] << 64) | self.entries[2 * entry_id] if self._is_dual_pde() else self.entries[entry_id]

  def read_fields(self, entry_id):
    if self.is_page(entry_id): return self.nvdev.pte_t.decode(self.entry(entry_id))
    return (self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t).decode(self.entry(entry_id))

  def is_page(self, entry_id): return (self.entry(entry_id) & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True

  def supports_huge_page(self, paddr): return self.lv >= self.nvdev.mm.level_cnt - 3 and paddr % self.nvdev.mm.pte_covers[self.lv] == 0

  def valid(self, entry_id):
    if self.is_page(entry_id): return self.read_fields(entry_id)['valid']
    return self.read_fields(entry_id)['aperture_small' if self._is_dual_pde() else 'aperture'] != 0

  def address(self, entry_id):
    small, sys = ("_small" if self._is_dual_pde() else ""), "_sys" if self.nvdev.mmu_ver == 2 or self.lv == self.nvdev.mm.level_cnt - 1 else ""
    return self.read_fields(entry_id)[f'address{small}{sys}'] << 12

class NVMemoryManager(MemoryManager):
  """NV-specific MemoryManager: VA allocator + MMU invalidation on map."""
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000)
  def __init__(self, *args, cpu_visible_limit=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_visible_pa_allocator = None
    if cpu_visible_limit is not None:
      pa_base = self.pa_allocator.base
      if not pa_base < cpu_visible_limit < self.vram_size:
        raise ValueError(f"invalid CPU-visible VRAM range {pa_base:#x}..{cpu_visible_limit:#x}")
      self.cpu_visible_pa_allocator = TLSFAllocator(cpu_visible_limit - pa_base, base=pa_base)
      self.pa_allocator = TLSFAllocator(self.vram_size - cpu_visible_limit, base=cpu_visible_limit)
  def palloc_cpu_visible(self, size, align=0x1000, zero=True):
    if self.cpu_visible_pa_allocator is None: return self.palloc(size, align=align, zero=zero)
    paddr = self.cpu_visible_pa_allocator.alloc(round_up(size, 0x1000), align)
    if zero: self.dev.vram[paddr:paddr + size] = bytes(size)
    return paddr
  def valloc_cpu_visible(self, size, align=0x1000, uncached=False, zero=True):
    va = self.alloc_vaddr(size := round_up(size, 0x1000), align)
    paddr = self.palloc_cpu_visible(size, align=align, zero=zero)
    return self.map_range(va, size, [(paddr, size)], aspace=AddrSpace.PHYS, uncached=uncached)
  def pfree(self, paddr, ptable=False):
    alloc = self.cpu_visible_pa_allocator
    if alloc is not None and alloc.base <= paddr < alloc.base + alloc.size: alloc.free(paddr)
    else: super().pfree(paddr, ptable)
  def on_range_mapped(self):
    self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev:
  def __init__(self, pci_dev):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')
    self.is_usb = isinstance(pci_dev, USBPCIDevice)
    self.smi_dev, self.is_booting, self.is_err_state = False, True, False
    self._early_ip_init()
    self._early_mmu_init()
    self.is_booting = False  # past boot phase: alloc_boot_mem can use palloc(boot=False)
    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()


  def _recover_stale_wpr(self):
    usb_trace("stale WPR2 detected; recovering with GPU function-level reset")
    cast(USBPCIDevice, self.pci_dev).function_level_reset()
    self.mmio = self.pci_dev.map_bar(0, fmt='I')
    self.flcn.wait_for_reset()
    if (wpr2_hi := self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read()) != 0:
      raise RuntimeError(f"USB FLR did not clear WPR2 (HI={wpr2_hi:#x}). Need replug.")
    usb_trace("stale WPR2 cleared by GPU FLR")

  def fini(self):
    if not self.is_usb:
      self.gsp.fini_hw()
      return
    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() == 0: return
    usb_trace("clean shutdown: UNLOADING_GUEST_DRIVER then SEC2 booter_unload")
    self.gsp.fini_hw()
    cast(NV_FLCN, self.flcn).shutdown_booter()
    usb_trace(f"clean shutdown complete WPR2_HI={self.reg('NV_PFB_PRI_MMU_WPR2_ADDR_HI').read():#x}")

  def reg(self, reg): return self.__dict__[reg]
  def wreg(self, addr, value):
    self.mmio[addr // 4] = value
    if TRACE >= 2:
      nm = _resolve_reg_name(self, addr) or hex(addr)
      bits = _trace_decode_bits(_trace_reg_obj.get(addr), value)
      _trace("REG", f"w {nm} = {value:#010x}{bits}")
    elif NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr):
    val = self.mmio[addr // 4]
    if TRACE:
      poll_sample(addr, val)
      if TRACE >= 2:
        # Suppress poll-loop spam: only log when the read value changes.
        if _trace_last_reg_rd.get(addr) != val:
          _trace_last_reg_rd[addr] = val
          nm = _resolve_reg_name(self, addr) or hex(addr)
          bits = _trace_decode_bits(_trace_reg_obj.get(addr), val)
          _trace("REG", f"r {nm} -> {val:#010x}{bits}")
    return val

  # =========================================================================
  # S2: Chip detect — read PMC_BOOT_0/42, identify GA102, probe MMU version.
  #     If WPR2 is nonzero from a prior boot, trigger PCI reset to recover.
  # =========================================================================
  def _early_ip_init(self):
    stage_set(2, "chip detect / MMU probe")
    self.reg_names = set()
    self.reg_offsets = {}
    self.include("nv_ref", "")
    self.include("dev_fb", "tu102")
    self.include("dev_gc6_island", "ga102")
    with _tscope():
      needs_reset = self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0
    recover_wpr = needs_reset and self.is_usb
    def _pci_cmd_transition(old, new, action):
      tprint(f"PCI_COMMAND {old:#06x} {fmt_pci_command(old)} -> "
             f"{new:#06x} {fmt_pci_command(new)}  {action}")
      with _tscope():   # nested CFG_READ/WRITE transport lines indent under the semantic line
        back = self.pci_dev.write_config_flush(pci.PCI_COMMAND, new, 2)
      tprint("verified ✓" if back == new else f"MISMATCH readback={back:#06x} ✗")
    if needs_reset:
      if recover_wpr:
        tprint("USB stale WPR2 recovery required")
      else:
        tprint("PCI reset required")
        pci_cmd = self.pci_dev.read_config(pci.PCI_COMMAND, 2)
        _pci_cmd_transition(pci_cmd, pci_cmd & ~pci.PCI_COMMAND_MASTER, "disable MASTER")
        with _tscope(): self.pci_dev.reset()
        time.sleep(0.1)
        tprint("PCI RESET complete")

    # Restore bus mastering after (optional) reset.
    pci_cmd = self.pci_dev.read_config(pci.PCI_COMMAND, 2)
    _pci_cmd_transition(pci_cmd, pci_cmd | pci.PCI_COMMAND_MASTER, "enable MASTER")
    with _tscope():
      self.chip_id = self.reg("NV_PMC_BOOT_0").read()
      self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
    self.fw_name = {"GB2": "gb202", "AD1": "ad102", "GA1": "ga102"}[self.chip_name[:3]]
    self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)
    tprint(f"chip={self.chip_name} fw={self.fw_name} mmu_ver={self.mmu_ver} fmc_boot={self.fmc_boot}")
    # Construct falcon/gsp IPs (NV_FLCN_COT path is GB2xx-only — RTX 3080 is GA102)
    self.flcn = NV_FLCN(self)
    self.gsp = NV_GSP(self)
    if recover_wpr: self._recover_stale_wpr()
    elif needs_reset: self.flcn.wait_for_reset()

  # =========================================================================
  # S2.1: MMU init — read VRAM size from scratch register, construct
  #       NVMemoryManager with boot/ptable/PA allocators, map BAR1.
  # =========================================================================
  def _early_mmu_init(self):
    self.include("dev_vm", "tu102")
    self.include("dev_mmu", "gh100" if self.mmu_ver == 3 else "tu102")
    self.pte_t, self.pde_t, self.dual_pde_t = [self.__dict__[name] for name in [f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE', f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE']]
    if TRACE: _trace_regdevs.append(self)
    with _tscope():
      self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20
      tprint(f"vram_size={self.vram_size:#x} large_bar={self.pci_dev.bar_info(1)[1] >= self.vram_size}")
      self.vram, self.mmio = self.pci_dev.map_bar(1), self.pci_dev.map_bar(0, fmt='I')
      self.large_bar = self.vram.nbytes >= self.vram_size
      bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])
      self.mm = NVMemoryManager(self, self.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry,
                                va_bits=bits, va_shifts=shifts, va_base=0,
                                palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]],
                                reserve_ptable=not self.large_bar,
                                cpu_visible_limit=self.pci_dev.bar_info(1)[1] if self.is_usb else None)
    stage_done(f"{self.chip_name}, vram={self.vram_size >> 30}GiB")

  # =========================================================================
  # S2.2/S3.2: Allocate boot-phase memory (sysmem or VRAM).
  # =========================================================================
  def _alloc_boot_mem(self, size, data=None, contiguous=False, sysmem=None):
    sz = round_up(size, 0x1000)
    if sysmem is True or (sysmem is None and not self.large_bar and not self.is_usb):
      view, sysaddr = self.pci_dev.alloc_sysmem(size, 0, contiguous=contiguous)
      paddr = None
    else:
      paddr = self.mm.palloc_cpu_visible(sz) if self.is_usb else self.mm.palloc(sz, boot=False)
      view = self.vram.view(paddr, sz)
      # USB boot structures placed in VRAM are addressed by GPU physical
      # address; the BAR1 bus aperture is not coherent GSP SYS memory.
      base = paddr if self.is_usb else self.pci_dev.bar_info(1)[0] + paddr
      sysaddr = [base + i * 0x1000 for i in range(sz // 0x1000)]
    if data is not None: view[:size] = data
    return view, paddr, sysaddr

  # =========================================================================
  # S2–S3: Register include — pull nv_ref/dev_fb/dev_gc6_island/dev_gsp/etc.
  # =========================================================================
  def include(self, name, arch):
    # nv_ref.regs and dev_*.regs are dicts of {reg_name: (base, off_lambda)} tuples in tinygrad.
    src_mod = importlib.import_module(f"autogen.nv_regs.{name}")
    regs = getattr(src_mod, arch or "regs")
    for k, v in regs.items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v


# ============================================================================
# nv/ip.py (vendored from ref/tinygrad/tinygrad/runtime/support/nv/ip.py)
# Slimmed: NV_FLCN (skip NV_FLCN_COT — we run GA102 not GB2xx), NV_GSP,
#          NVRpcQueue (with run_cpu_seq handling).
# ============================================================================
@dataclasses.dataclass(frozen=True)
class GRBufDesc:
  size: int; virt: bool; phys: bool; local: bool = False

class NV_IP:
  """Base class for NV IP blocks (NV_FLCN, NV_GSP)."""
  def __init__(self, nvdev): self.nvdev = nvdev
  def init_sw(self): pass
  def init_hw(self): pass
  def fini_hw(self): pass

class NVRpcQueue:
  """GSP message queue: send RPC records, read response events, doorbell kick."""
  def __init__(self, gsp, view, completion_q_view=None):
    self.tx_view = view.view(fmt='I')
    wait_cond(lambda: self.tx_view[getattr(nv.msgqTxHeader, 'entryOff').offset // 4], value=0x1000, msg="GSP RPC queue header initialized (entryOff==0x1000)")
    self.tx = nv.msgqTxHeader.from_buffer_copy(bytes(view[:ctypes.sizeof(nv.msgqTxHeader)]))
    if completion_q_view is not None:
      comp_tx = nv.msgqTxHeader.from_buffer_copy(bytes(completion_q_view[:ctypes.sizeof(nv.msgqTxHeader)]))
      self.rx_view = completion_q_view.view(comp_tx.rxHdrOff, fmt='I')
    self.gsp, self.view, self.seq = gsp, view, 0
    self.queue_mv = view.view(self.tx.entryOff, self.tx.msgSize * self.tx.msgCount)

  def _checksum(self, data):
    if (pad_len := (-len(data)) % 8): data += b'\x00' * pad_len
    checksum = 0
    for offset in range(0, len(data), 8): checksum ^= struct.unpack_from('Q', data, offset)[0]
    return hi32(checksum) ^ lo32(checksum)

  def _send_rpc_record(self, func, msg):
    header = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, rpc_result=nv.NV_VGPU_MSG_RESULT_RPC_PENDING,
      rpc_result_private=nv.NV_VGPU_MSG_RESULT_RPC_PENDING, header_version=(3 << 24), function=func, length=len(msg) + 0x20)
    msg = bytes(header) + msg
    phdr = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=ceildiv(len(msg) + ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT), self.tx.msgSize), seqNum=self.seq)
    phdr.checkSum = self._checksum(bytes(phdr) + msg)
    msg = (bytes(phdr) + msg).ljust(phdr.elemCount * self.tx.msgSize, b'\x00')
    wp = self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4]
    off, first = wp * self.tx.msgSize, min(len(msg), len(self.queue_mv) - wp * self.tx.msgSize)
    self.queue_mv[off:off + first] = msg[:first]
    if first < len(msg): self.queue_mv[:len(msg) - first] = msg[first:]
    self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4] = (wp + phdr.elemCount) % self.tx.msgCount
    System.memory_barrier()
    self.seq += 1
    self.gsp.nvdev.NV_PGSP_QUEUE_HEAD[0].write(0x0)

  def send_rpc(self, func, msg):
    if TRACE:
      name = _trace_rpc_func(func)
      rpc_remember_request(func, bytes(msg))
      # RM ops print one semantic record from the rpc_rm_* wrapper instead of
      # wire structs; raw structs stay available under NV_TRACE_RAW=1.
      quiet_sem = not TRACE_RAW and name in RM_SEMANTIC_FUNCS
      show = False if quiet_sem else (rpc_first(f"-> {name} len={len(msg):#x}") or not agg_active())
      if show:
        blk = _trace_fmt_struct(func, msg)
        # Raw hex only for UNKNOWN / undecoded payloads (or TRACE_RAW=1).
        payload = None if (blk and TRACE_NOZERO and not TRACE_RAW) else bytes(msg)
        _trace("RPC -->", f"{name} len={len(msg):#x}" + (f"\n{blk}" if blk else ""), payload)
    max_payload = self.tx.msgSize * 16 - ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT) - ctypes.sizeof(nv.rpc_message_header_v)
    self._send_rpc_record(func, msg[:max_payload])
    for off in range(max_payload, len(msg), max_payload):
      self._send_rpc_record(nv.NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD, msg[off:off + max_payload])

  def read_resp(self):
    System.memory_barrier()
    while self.rx_view[0] != self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4]:
      off = self.rx_view[0] * self.tx.msgSize
      hdr = nv.rpc_message_header_v.from_buffer_copy(bytes(self.queue_mv[off + 0x30:off + 0x30 + ctypes.sizeof(nv.rpc_message_header_v)]))
      msg = bytes(self.queue_mv[off + 0x50:off + 0x50 + hdr.length])
      if hdr.function == nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: self.gsp.run_cpu_seq(msg)
      elif hdr.function == nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG:
        print(f"nv {self.gsp.nvdev.devfmt}: GSP LOG: {msg[12:].rstrip(bytes([0])).decode('utf-8')}")
      self.gsp.nvdev.is_err_state |= hdr.function in {nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG, nv.NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED}
      self.rx_view[0] = (self.rx_view[0] + round_up(hdr.length, self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
      System.memory_barrier()
      if hdr.rpc_result != 0: raise RuntimeError(f"RPC call {hdr.function} failed with result {hdr.rpc_result}")
      if TRACE and hdr.function != nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG:
        name = _trace_rpc_func(hdr.function)
        if not TRACE_RAW and name in RM_SEMANTIC_FUNCS:
          # Semantic RM record printed by the wrapper; skip the transport line
          # (incl. the per-call doorbell noise) entirely on success.
          yield hdr.function, msg
          continue
        transport = "OK" if hdr.rpc_result == 0 else f"{hdr.rpc_result:#x}"
        blk = _trace_fmt_struct(hdr.function, msg)
        mstat = None
        import re as _re
        if blk and (m := _re.search(r"\bstatus\s+= (0x[0-9a-f]+)", blk)):
          code = int(m.group(1), 16)
          mstat = "OK" if code == 0 else f"{m.group(1)}({rpc_status_sym(code)})" if code else "OK"
        head = f"{name} transport={transport}" + (f" rm_status={mstat}" if mstat else "")
        collapsible = name.startswith(("POST_NOCAT", "UCODE_LIBOS_PRINT"))
        if collapsible:
          detail_blk = _trace_fmt_event(hdr.function, msg)
          key = f"<- {name}"
          first = rpc_note(key, (detail_blk or "").strip())
          if not first and agg_active():
            yield hdr.function, msg
            continue
          _trace("RPC <--", f"{head}\n{(detail_blk or '').strip()}")   # headers are fully decoded; raw stays in TLP
          yield hdr.function, msg
          continue
        show = rpc_first(f"<- {name} len={len(msg):#x}") or not agg_active()
        if show:
          delta = rpc_delta_lines(hdr.function, msg) if blk else []
          had_request = has_cached_request(hdr.function)
          evblk = "" if blk else (_trace_fmt_event(hdr.function, msg) or "")
          if blk and had_request:
            # Request was decoded: reply collapses to transport/status + deltas.
            body = ""
            if mstat is None or mstat == "OK": pass
            if delta:
              body = "    <- " + "; ".join(delta[:8]) + (" ..." if len(delta) > 8 else "")
          elif blk:
            body = "\n".join(ln for ln in blk.split("\n") if ln.strip())
          else:
            body = ""
          full = head + ((" [doorbell BAR0+0x110c00]") if name in ("GSP_RM_CONTROL", "GSP_RM_ALLOC", "SET_PAGE_DIRECTORY") else "") \
                 + (("\n" + body) if body else "") + (("\n" + evblk) if evblk else "")
          skip_dump = hdr.function == nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER or (((body or evblk) or had_request) and TRACE_NOZERO)
          _trace("RPC <--", full, None if skip_dump else msg)
      yield hdr.function, msg

  def wait_resp(self, cmd, timeout=10000):
    start_time = int(time.perf_counter() * 1000)
    while (int(time.perf_counter() * 1000) - start_time) < timeout:
      if (msg := next((message for func, message in self.read_resp() if func == cmd), None)) is not None: return msg
    raise RuntimeError(f"Timeout waiting for RPC response for command {cmd}")


class NV_FLCN(NV_IP):
  """S3–S4 — Falcon security processor: VBIOS prep, FWSEC/SEC2 secure boot."""
  # =========================================================================
  # S2.1: Wait for falcon PRI unlock and GFW boot progress == 0xff.
  #       Polls PRIV_LEVEL_MASK and boot_progress; timeout = 10s.
  # =========================================================================
  def wait_for_reset(self):
    wait_cond(lambda: self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.read_bitfields()['read_protection_level0'] == 1
                        and self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0].read() & 0xff == 0xff,
              msg="falcon PRI unlocked (PRIV_LEVEL_MASK.level0==1) and GFW boot progress byte == 0xff")
    # Semantic milestone: the falcon security handshake. Raw per-poll traffic
    # remains in the wire layer / TLP sidecar.
    with _tscope():
      mask = self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.read()
      prog = self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0].read()
      lvl0 = self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.read_bitfields()['read_protection_level0']
    samples = LAST_WAIT_SAMPLES
    traj = [v & 0xff for a, vs in samples.items() if a == 0x118234 for v in vs] or [prog & 0xff]
    unlocked = lvl0 and prog & 0xff == 0xff
    tprint("Falcon security handshake\n"
           f"      PRI read permission : {'OK' if lvl0 else 'LOCKED'}\n"
           f"      GFW boot progress   : {' -> '.join(f'0x{v:02x}' for v in traj)}\n"
           f"      result              : {'UNLOCKED' if unlocked else 'NOT READY'}")

  # =========================================================================
  # S3: VBIOS shadow + FWSEC extraction + booter load.
  #     Reads 1MB VBIOS ROM @BAR0+0x300000, parses BIT header for FWSEC ucode.
  #     Fetches booter_load-570.144.bin, patches signature, allocs boot memory.
  # =========================================================================
  def init_sw(self):
    self.nvdev.include("dev_gsp", "ga102")
    stage_set(3, "VBIOS / FWSEC / booter / GSP image")
    self.nvdev.include("dev_falcon_v4", "ga102")
    self.nvdev.include("dev_riscv_pri", "ga102")
    self.nvdev.include("dev_fbif_v4", "ga102")
    self.nvdev.include("dev_falcon_second_pri", "ga102")
    self.nvdev.include("dev_sec_pri", "ga102")
    self.nvdev.include("dev_bus", "tu102")
    self.prep_ucode()
    self.prep_booter()

  # =========================================================================
  # S3.1–S3.2: Parse 1MB VBIOS ROM shadow to extract FWSEC falcon ucode.
  #     Walks PCI expansion ROM chain, BIT header, FALCON_UCODE_TABLE.
  # =========================================================================
  def prep_ucode(self):
    tprint(f"loading VBIOS ROM: reading 1MB shadow @BAR0+0x300000 to extract FWSEC falcon ucode")
    with _tscope():
      vbios_bytes, vbios_off = memoryview(bytes(_array_mod.array('I', self.nvdev.mmio[0x00300000 // 4:(0x00300000 + 0x100000) // 4]))), 0
      while True:
        pci_blck = vbios_bytes[vbios_off + nv.OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR:].cast('H')[0]
        imglen = vbios_bytes[vbios_off + pci_blck + nv.OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN:].cast('H')[0] * nv.PCI_ROM_IMAGE_BLOCK_SIZE
        match vbios_bytes[vbios_off + pci_blck + nv.OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE]:
          case nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE: block_size = imglen
          case nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT:
            expansion_rom_off = vbios_off - block_size
            break
        vbios_off += imglen
      tprint(f"VBIOS chain: base rom {block_size:#x}, FWSEC expansion image @+{expansion_rom_off:#x}")
      bit_header = nv.BIT_HEADER_V1_00.from_buffer_copy(vbios_bytes[(bit_addr := 0x1b0):bit_addr + ctypes.sizeof(nv.BIT_HEADER_V1_00)])
      assert bit_header.Signature == 0x00544942, f"Invalid BIT header signature {hex(bit_header.Signature)}"
      for i in range(bit_header.TokenEntries):
        bit = nv.BIT_TOKEN_V1_00.from_buffer_copy(vbios_bytes[bit_addr + bit_header.HeaderSize + i * bit_header.TokenSize:])
        if bit.TokenId != nv.BIT_TOKEN_FALCON_DATA or bit.DataVersion != 2 or bit.DataSize < nv.BIT_DATA_FALCON_DATA_V2_SIZE_4: continue
        falcon_data = nv.BIT_DATA_FALCON_DATA_V2.from_buffer_copy(vbios_bytes[bit.DataPtr & 0xffff:])
        ucode_hdr = nv.FALCON_UCODE_TABLE_HDR_V1.from_buffer_copy(vbios_bytes[(table_ptr := expansion_rom_off + falcon_data.FalconUcodeTablePtr):])
        for j in range(ucode_hdr.EntryCount):
          ucode_entry = nv.FALCON_UCODE_TABLE_ENTRY_V1.from_buffer_copy(vbios_bytes[table_ptr + ucode_hdr.HeaderSize + j * ucode_hdr.EntrySize:])
          if ucode_entry.ApplicationID != nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: continue
          ucode_desc_hdr = nv.FALCON_UCODE_DESC_HEADER.from_buffer_copy(vbios_bytes[expansion_rom_off + ucode_entry.DescPtr:])
          ucode_desc_off = expansion_rom_off + ucode_entry.DescPtr
          ucode_desc_size = ucode_desc_hdr.vDesc >> 16
      self.desc_v3 = nv.FALCON_UCODE_DESC_V3.from_buffer_copy(vbios_bytes[ucode_desc_off:ucode_desc_off + ucode_desc_size])
      sig_total_size = ucode_desc_size - nv.FALCON_UCODE_DESC_V3_SIZE_44
      signature = vbios_bytes[ucode_desc_off + nv.FALCON_UCODE_DESC_V3_SIZE_44:][:sig_total_size]
      image = vbios_bytes[ucode_desc_off + ucode_desc_size:][:round_up(self.desc_v3.StoredSize, 256)]
      self.frts_offset = self.nvdev.vram_size - 0x100000 - 0x100000
      read_vbios_desc = nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2)
      frst_reg_desc = nv.FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_FRTS_REGION_DESC),
        frtsRegionOffset4K=self.frts_offset >> 12, frtsRegionSize=0x100, frtsRegionMediaType=2)
      frts_cmd = nv.FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)

      def __patch(cmd_id, cmd):
        patched_image = bytearray(image)
        dmem_offset = 0
        hdr = nv.FALCON_APPLICATION_INTERFACE_HEADER_V1.from_buffer_copy(image[(app_hdr_off := self.desc_v3.IMEMLoadSize + self.desc_v3.InterfaceOffset):])
        ents = (nv.FALCON_APPLICATION_INTERFACE_ENTRY_V1 * hdr.entryCount).from_buffer_copy(image[app_hdr_off + ctypes.sizeof(hdr):])
        for i in range(hdr.entryCount):
          if ents[i].id == nv.FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER: dmem_offset = ents[i].dmemOffset
        dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[(dmem_mapper_offset := self.desc_v3.IMEMLoadSize + dmem_offset):])
        dmem.init_cmd = cmd_id
        patched_image[dmem_mapper_offset:dmem_mapper_offset + len(bytes(dmem))] = bytes(dmem)
        patched_image[(cmd_off := self.desc_v3.IMEMLoadSize + dmem.cmd_in_buffer_offset):cmd_off + len(cmd)] = cmd
        patched_image[(sig_off := self.desc_v3.IMEMLoadSize + self.desc_v3.PKCDataOffset):sig_off + 0x180] = signature[-0x180:]
        tprint(f"FWSEC image: cmd={cmd_id:#x} len={len(patched_image):#x} sig@{sig_off:#x} frts_paddr={self.nvdev.flcn.frts_offset:#x}")
        return self.nvdev._alloc_boot_mem(len(patched_image), data=patched_image, sysmem=False)

      _, self.frts_image_paddr, _ = __patch(0x15, bytes(frts_cmd))

  # =========================================================================
  # S3.3: Fetch and patch booter_load image (nvidia/<fw>/gsp/booter_load-570.144.bin).
  # =========================================================================
  def prep_booter(self):
    shas = {
      "ga102": ("4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b",
                "8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009"),
      "ad102": ("8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a",
                "975b85a14ded8e430d30f000c3c1afdd55c15dee04f35ff9dfd876acd7e67186")}[self.nvdev.fw_name]

    def __prep(name, sha):
      h = nv.struct_nvfw_bin_hdr.from_buffer_copy(b := fetch_fw(f"nvidia/{self.nvdev.fw_name}/gsp", name, sha))
      hs = nv.struct_nvfw_hs_header_v2.from_buffer_copy(b, h.header_offset)
      lh = nv.struct_nvfw_hs_load_header_v2.from_buffer_copy(b, hs.header_offset)
      app = nv.struct_nvfw_hs_load_header_v2_app.from_buffer_copy(b, hs.header_offset + ctypes.sizeof(nv.struct_nvfw_hs_load_header_v2))
      patch_loc, patch_sig = struct.unpack_from("<I", b, hs.patch_loc)[0], struct.unpack_from("<I", b, hs.patch_sig)[0]
      sig = b[(sig_off := hs.sig_prod_offset + patch_sig):sig_off + (sig_len := hs.sig_prod_size // struct.unpack_from("<I", b, hs.num_sig)[0])]
      (patched_image := bytearray(b[h.data_offset:h.data_offset + h.data_size]))[patch_loc:patch_loc + sig_len] = sig
      tprint(f"BOOTER image {name}: len={len(patched_image):#x} patch_loc={patch_loc:#x} sig_len={sig_len:#x}")
      return bytes(patched_image), lh.os_data_offset, lh.os_data_size, app.offset, app.size

    load_image, self.booter_data_off, self.booter_data_sz, self.booter_code_off, self.booter_code_sz = \
      __prep("booter_load-570.144.bin", shas[0])
    with _tscope():
      _, self.booter_image_paddr, _ = self.nvdev._alloc_boot_mem(len(load_image), data=load_image, sysmem=False)
    if self.nvdev.is_usb:
      unload_image, self.booter_unload_data_off, self.booter_unload_data_sz, \
        self.booter_unload_code_off, self.booter_unload_code_sz = __prep("booter_unload-570.144.bin", shas[1])
      with _tscope():
        _, self.booter_unload_image_paddr, _ = self.nvdev._alloc_boot_mem(len(unload_image), data=unload_image, sysmem=False)

  def init_hw(self):
    self.falcon, self.sec2 = 0x00110000, 0x00840000
    stage_set(4, "FWSEC -> SEC2 -> GSP RISCV")
    if self.nvdev.is_usb:
      tprint("USB GSP stream: retrain PCIe to Gen1 and stage initial 512 KiB SRAM image")
      self.nvdev.pci_dev.retrain_pcie(1)
      usb_trace("F2 initial SRAM upload bytes=0x80000 sectors=0x400 slots=0:32 "
                "layout=meta@0x200000 sig@0x201000 booter@0x202000 radix@0x208000 ring@0x22c000")
      self.nvdev.pci_dev.usb.scsi_write(self.nvdev.gsp._boot_sram)
    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_paddr, code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
      imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
      dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
      pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)
    assert self.nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() != 0, "WPR2 is not initialized"
    self.reset(self.falcon, riscv=True)
    self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.nvdev.gsp.libos_args_sysmem))
    self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.nvdev.gsp.libos_args_sysmem))
    tprint(f"GSP boot args: MAILBOX[1:0] = {self.nvdev.gsp.libos_args_sysmem:#x} -> libos_args")
    self.reset(self.sec2)
    mbx = self.execute_hs(self.sec2, self.booter_image_paddr, code_off=self.booter_code_off, data_off=self.booter_data_off,
      imemPa=0x0, imemVa=self.booter_code_off, imemSz=self.booter_code_sz, dmemPa=0x0, dmemVa=0x0, dmemSz=self.booter_data_sz,
      pkc_off=0x10, engid=1, ucodeid=3, mailbox=self.nvdev.gsp.wpr_meta_sysmem, stream_gsp=True)
    assert mbx[0] == 0x0, f"Booter failed to execute, mailbox is {mbx[0]:08x}, {mbx[1]:08x}"
    self.nvdev.NV_PFALCON_FALCON_OS.with_base(self.falcon).write(0x0)
    assert self.nvdev.NV_PRISCV_RISCV_CPUCTL.with_base(self.falcon).read_bitfields()['active_stat'] == 1, "GSP Core is not active"
    if self.nvdev.is_usb: self.nvdev.pci_dev.retrain_pcie(3)
    stage_done('GSP RISCV active')

  def shutdown_booter(self):
    self.reset(self.sec2)
    mbx = self.execute_hs(self.sec2, self.booter_unload_image_paddr, code_off=self.booter_unload_code_off,
      data_off=self.booter_unload_data_off, imemPa=0x0, imemVa=self.booter_unload_code_off, imemSz=self.booter_unload_code_sz,
      dmemPa=0x0, dmemVa=0x0, dmemSz=self.booter_unload_data_sz, pkc_off=0x10, engid=1, ucodeid=3,
      mailbox=(0xff << 32) | 0xff)
    if mbx[0] != 0: raise RuntimeError(f"Booter Unload failed with mailbox {mbx[0]:#x}, {mbx[1]:#x}")

  # =========================================================================
  # S4.1/S4.4: DMA transfer from VRAM to falcon IMEM/DMEM (256B chunks).
  # =========================================================================
  def execute_dma(self, base, cmd, dest, mem_off, src, size):
    tprint(f"FALCON DMA: base={base:#x} dest={dest:#x} src_fb={src:#x} fb_off={mem_off:#x} size={size:#x} ({size // 256} x 256B chunks)")
    with _tscope():
      wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA ready for next chunk (DMATRFCMD.full == 0)")
      self.nvdev.NV_PFALCON_FALCON_DMATRFBASE.with_base(base).write(lo32(src >> 8))
      self.nvdev.NV_PFALCON_FALCON_DMATRFBASE1.with_base(base).write(hi32(src >> 8) & 0x1ff)
      xfered = 0
      chunks = ceildiv(size, 0x100)
      with quiet():
        while xfered < size:
          wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA ready for next chunk (DMATRFCMD.full == 0)")
          self.nvdev.NV_PFALCON_FALCON_DMATRFMOFFS.with_base(base).write(dest + xfered)
          self.nvdev.NV_PFALCON_FALCON_DMATRFFBOFFS.with_base(base).write(mem_off + xfered)
          self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).write(cmd)
          xfered += 256
      tprint(f"DMA_LOOP chunks[0..{chunks - 1}] dst_off {dest:#x} -> {dest + size - 0x100:#x} | "
             f"src_off {mem_off:#x} -> {mem_off + size - 0x100:#x} | cmd={cmd:#x} | status=completed")
      wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['idle'], msg="DMA transfer complete (DMATRFCMD.idle == 1)")

  def start_cpu(self, base):
    if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['alias_en'] == 1:
      self.nvdev.wreg(base + self.nvdev.NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2)
    else:
      self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).write(startcpu=1)

  def wait_cpu_halted(self, base):
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['halted'], msg="CPUCTL.HALTED[4] == 1")

  # =========================================================================
  # S4.1/S4.4: Load IMEM+DMEM via DMA, set BROM parameters, start CPU.
  # =========================================================================
  def execute_hs(self, base, img_paddr, code_off, data_off, imemPa, imemVa, imemSz, dmemPa, dmemVa, dmemSz,
                 pkc_off, engid, ucodeid, mailbox=None, stream_gsp=False):
    tprint(f"falcon boot: base={base:#x} IMEM {imemSz:#x} @va={imemVa:#x}/pa={imemPa:#x} | DMEM {dmemSz:#x} | ucodeid={ucodeid} mailbox={'-' if mailbox is None else hex(mailbox)}")
    with _tscope():
      self.disable_ctx_req(base)
      self.nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base(base)[ctx_dma := 0].update(target=0, mem_type=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL)
      cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
        ctxdma=ctx_dma, imem=1, sec=1)
      self.execute_dma(base, cmd, dest=imemPa, mem_off=imemVa, src=img_paddr + code_off - imemVa, size=imemSz)
      cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
        ctxdma=ctx_dma, imem=0, sec=0)
      self.execute_dma(base, cmd, dest=dmemPa, mem_off=dmemVa, src=img_paddr + data_off - dmemVa, size=dmemSz)
      self.nvdev.NV_PFALCON2_FALCON_BROM_PARAADDR.with_base(base)[0].write(pkc_off)
      self.nvdev.NV_PFALCON2_FALCON_BROM_ENGIDMASK.with_base(base).write(engid)
      self.nvdev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID.with_base(base).write(val=ucodeid)
      self.nvdev.NV_PFALCON2_FALCON_MOD_SEL.with_base(base).write(algo=self.nvdev.NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K)
      self.nvdev.NV_PFALCON_FALCON_BOOTVEC.with_base(base).write(imemVa)
      if mailbox is not None:
        self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).write(lo32(mailbox))
        self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).write(hi32(mailbox))
      self.start_cpu(base)
      if stream_gsp and self.nvdev.is_usb:
        self.nvdev.pci_dev.stream_gsp_boot(self.nvdev.gsp.gsp_image, time.perf_counter())
      self.wait_cpu_halted(base)
      if mailbox is not None:
        return self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).read(), self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).read()

  def disable_ctx_req(self, base):
    self.nvdev.NV_PFALCON_FBIF_CTL.with_base(base).update(allow_phys_no_ctx=1)
    self.nvdev.NV_PFALCON_FALCON_DMACTL.with_base(base).write(0x0)

  def reset(self, base, riscv=False):
    tprint(f"falcon reset: base={base:#x} riscv={riscv}")
    with _tscope():
      engine_reg = self.nvdev.NV_PGSP_FALCON_ENGINE if base == self.falcon else self.nvdev.NV_PSEC_FALCON_ENGINE
      engine_reg.write(reset=1)
      time.sleep(0.1)
      engine_reg.write(reset=0)
      wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['mem_scrubbing'], value=0, msg="FALCON mem scrub complete")
      if riscv:
        self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=1, valid=0, brfetch=1)
      elif self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['riscv'] == 1:
        self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=0)
        wait_cond(lambda: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).read_bitfields()['valid'], msg="RISCV BCR valid (core booted)")
        self.nvdev.NV_PFALCON_FALCON_RM.with_base(base).write(self.nvdev.chip_id)


class NV_GSP(NV_IP):
  """S3–S5 — GSP-RM: firmware args, INIT_DONE, cpu_seq, golden channel, RPC."""
  # =========================================================================
  # S3: GSP-RM software init — RM cmd queue, libos args, WPR meta, system info.
  # =========================================================================
  def init_sw(self):
    self.handle_gen, self.chan_runlists = itertools.count(0xcf000000), {}
    self.init_rm_args()
    self.init_libos_args()
    self.init_wpr_meta()
    self.rpc_set_gsp_system_info()
    self.rpc_set_registry_table()
    self.gpfifo_class, self.compute_class, self.dma_class = nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_B, nv_gpu.AMPERE_DMA_COPY_B
    match self.nvdev.chip_name[:2]:
      case "AD": self.compute_class = nv_gpu.ADA_COMPUTE_A
      case "GB": self.gpfifo_class, self.compute_class, self.dma_class = nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.BLACKWELL_DMA_COPY_B

  def _stage_args(self, data, offset):
    if self.nvdev.is_usb: return self.nvdev.pci_dev.stage_gsp_args(data, offset)
    return self.nvdev._alloc_boot_mem(len(data), data=data)[2][0]

  # =========================================================================
  # S3.6: Allocate RM command queue in sysmem (page table + 2 message queues).
  # =========================================================================
  def init_rm_args(self, queue_size=0x40000):
    queue_size = 0x5000 if self.nvdev.is_usb else queue_size
    pte_cnt = ((queue_pte_cnt := (queue_size * 2) // 0x1000)) + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
    pt_size = round_up(pte_cnt * 8, 0x1000)
    tprint(f"RM cmd queue: q={queue_size:#x} entries={queue_size // 0x1000}")
    with _tscope():
      if self.nvdev.is_usb:
        queues_view, queues_sysmem = self.nvdev.pci_dev.alloc_gsp_queues(pt_size + queue_size * 2)
      else:
        queues_view, _, queues_sysmem = self.nvdev._alloc_boot_mem(pt_size + queue_size * 2, sysmem=True)
      for i, sysmem in enumerate(queues_sysmem): queues_view.view(i * 0x8, 0x8, fmt='Q')[0] = sysmem
      queue_args = nv.MESSAGE_QUEUE_INIT_ARGUMENTS(sharedMemPhysAddr=queues_sysmem[0], pageTableEntryCount=pte_cnt,
        cmdQueueOffset=pt_size, statQueueOffset=pt_size + queue_size)
      rm_args = bytes(nv.GSP_ARGUMENTS_CACHED(bDmemStack=True, messageQueueInitArguments=queue_args))
      self.rm_args_sysmem = self._stage_args(rm_args, 0x100)
      self.cmd_q_view, self.stat_q_view = queues_view.view(pt_size), queues_view.view(pt_size + queue_size)
      self.cmd_q_view[:ctypes.sizeof(nv.msgqTxHeader)] = bytes(nv.msgqTxHeader(version=0, size=queue_size, entryOff=0x1000, msgSize=0x1000,
        msgCount=(queue_size - 0x1000) // 0x1000, writePtr=0, flags=1, rxHdrOff=ctypes.sizeof(nv.msgqTxHeader)))
      self.cmd_q = NVRpcQueue(self, self.cmd_q_view, None)
      tprint(f"RM cmd queue ready: sharedMem={queues_sysmem[0]:#x} pt={pt_size:#x}")

  # =========================================================================
  # S3.9: Allocate libos memory regions (5x LOG + RMARGS) in sysmem.
  # =========================================================================
  def init_libos_args(self):
    tprint("libos args: 5x LOG regions (0x10000 each) + RMARGS")
    with _tscope():
      _, _, logbuf_addrs = self.nvdev._alloc_boot_mem(2 << 20)
      log_loc = nv.LIBOS_MEMORY_REGION_LOC_FB if self.nvdev.is_usb else nv.LIBOS_MEMORY_REGION_LOC_SYSMEM
      libos_structs = [nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=log_loc, size=0x10000,
          id8=int.from_bytes(bytes(f"LOG{name}", 'utf-8'), 'big'), pa=logbuf_addrs[0] + 0x10000 * i)
          for i, name in enumerate(["INIT", "INTR", "RM", "MNOC", "KRNL"])]
      libos_structs.append(nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM,
        size=0x1000, id8=int.from_bytes(bytes("RMARGS", 'utf-8'), 'big'), pa=self.rm_args_sysmem))
      self.libos_args_sysmem = self._stage_args(b''.join(bytes(s) for s in libos_structs), 0x200)
      tprint(f"libos args @{self.libos_args_sysmem:#x}: LOG@[{logbuf_addrs[0]:#x}..{logbuf_addrs[0] + (2 << 20):#x}] RMARGS@{self.rm_args_sysmem:#x}")

  # =========================================================================
  # S3.12: Load GSP ELF, extract .fwimage + signature, build radix3 page table.
  # =========================================================================
  def init_gsp_image(self):
    _, sections, _ = elf_loader(fetch_fw("nvidia/ga102/gsp", "gsp-570.144.bin", "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f"))
    self.gsp_image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    self.gsp_signature = next((sh.content for sh in sections if sh.name == (f".fwsignature_{self.nvdev.chip_name[:4].lower()}x")))
    if self.nvdev.is_usb:
      tprint(f"GSP image [{self.nvdev.chip_name}/{self.nvdev.fw_name}]: fw={len(self.gsp_image):#x} sig={len(self.gsp_signature):#x} (SRAM streamed)")
      return
    npages = [0, 0, 0, round_up(len(self.gsp_image), 0x1000) // 0x1000]
    for i in range(3, 0, -1): npages[i - 1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1
    offsets = [sum(npages[:i]) * 0x1000 for i in range(4)]
    tprint(f"GSP image [{self.nvdev.chip_name}/{self.nvdev.fw_name}]: sig_section=.fwsignature_{self.nvdev.chip_name[:4].lower()}x "
           f"fw={len(self.gsp_image):#x} sig={len(self.gsp_signature):#x} radix3_pages={npages[3]}")
    with _tscope():
      radix_view, _, self.gsp_radix3_addrs = self.nvdev._alloc_boot_mem(offsets[-1] + len(self.gsp_image))
      radix_view.view(offsets[-1], len(self.gsp_image))[:] = self.gsp_image
      for i in range(0, 3):
        cur_offset = sum(npages[:i + 1])
        radix_view.view(offsets[i], npages[i + 1] * 8, fmt='Q')[:] = _array_mod.array('Q', self.gsp_radix3_addrs[cur_offset:cur_offset + npages[i + 1]])
      _, _, gsp_sig_addrs = self.nvdev._alloc_boot_mem(len(self.gsp_signature), data=self.gsp_signature)
      self.gsp_signature_bar1 = gsp_sig_addrs[0]
      tprint(f"GSP image mapped: radix3@{self.gsp_radix3_addrs[0]:#x} sig_bar1={self.gsp_signature_bar1:#x}")

  # =========================================================================
  # S3.3: Fetch and patch bootloader-570.144.bin (RISCV boot binary).
  # =========================================================================
  def init_boot_binary_image(self):
    sha = {"ga102": "82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad",
           "ad102": "65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8",
           "gb202": "d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071"}[self.nvdev.fw_name]
    h = nv.struct_nvfw_bin_hdr.from_buffer_copy(b := fetch_fw(f"nvidia/{self.nvdev.fw_name}/gsp", "bootloader-570.144.bin", sha))
    self.booter_image, self.booter_desc = b[h.data_offset:h.data_offset + h.data_size], nv.RM_RISCV_UCODE_DESC.from_buffer_copy(b, h.header_offset)
    with _tscope():
      _, _, booter_addrs = self.nvdev._alloc_boot_mem(len(self.booter_image), data=self.booter_image)
      self.booter_bar1 = booter_addrs[0]

  def _build_sram_wpr(self, meta):
    page_size, sram_size = 0x1000, 0x80000
    table_page = 8
    ring_page, ring_pages = 44, 84
    npages = [0, 0, 0, round_up(len(self.gsp_image), page_size) // page_size]
    for i in range(3, 0, -1):
      npages[i - 1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1
    table_pages = sum(npages[:3])
    table_addrs = [0x200000 + (table_page + i) * page_size for i in range(table_pages)]
    image_addrs = [0x200000 + (ring_page + i % ring_pages) * page_size for i in range(npages[3])]
    radix_addrs = table_addrs + image_addrs
    radix = bytearray(table_pages * page_size)
    offsets = [sum(npages[:i]) * page_size for i in range(4)]
    for i in range(3):
      start = sum(npages[:i + 1])
      values = radix_addrs[start:start + npages[i + 1]]
      struct.pack_into(f"<{len(values)}Q", radix, offsets[i], *values)
    meta.sysmemAddrOfRadix3Elf = table_addrs[0]
    meta.sysmemAddrOfBootloader, meta.sysmemAddrOfSignature = 0x202000, 0x201000
    sram = bytearray(sram_size)
    sram[:ctypes.sizeof(type(meta))] = bytes(meta)
    sram[page_size:page_size + len(self.gsp_signature)] = self.gsp_signature
    sram[2 * page_size:2 * page_size + len(self.booter_image)] = self.booter_image
    sram[table_page * page_size:(table_page + table_pages) * page_size] = radix
    sram[ring_page * page_size:(ring_page + ring_pages) * page_size] = self.gsp_image[:ring_pages * page_size]
    tprint(f"USB SRAM WPR: tables={table_pages} pages @{table_addrs[0]:#x}, ring={ring_pages} pages @{image_addrs[0]:#x}")
    return bytes(sram)

  # =========================================================================
  # S3.15: Build WPR meta (bootloader+radix3+frts offsets/sizes for VRAM layout).
  # =========================================================================
  def init_wpr_meta(self):
    self.init_gsp_image()
    self.init_boot_binary_image()
    sram_boot = self.nvdev.is_usb
    common = {'sizeOfBootloader': (boot_sz := len(self.booter_image)), 'sysmemAddrOfBootloader': 0 if sram_boot else self.booter_bar1,
      'sizeOfRadix3Elf': (radix3_sz := len(self.gsp_image)), 'sysmemAddrOfRadix3Elf': 0 if sram_boot else self.gsp_radix3_addrs[0],
      'sizeOfSignature': 0x1000, 'sysmemAddrOfSignature': 0 if sram_boot else self.gsp_signature_bar1,
      'bootloaderCodeOffset': self.booter_desc.monitorCodeOffset, 'bootloaderDataOffset': self.booter_desc.monitorDataOffset,
      'bootloaderManifestOffset': self.booter_desc.manifestOffset, 'revision': nv.GSP_FW_WPR_META_REVISION, 'magic': nv.GSP_FW_WPR_META_MAGIC}
    if self.nvdev.fmc_boot:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=0x20000, pmuReservedSize=0x1820000, nonWprHeapSize=0x220000,
        gspFwHeapSize=0x8700000, frtsSize=0x100000)
    else:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=(vga_sz := 0x100000), vgaWorkspaceOffset=(vga_off := self.nvdev.vram_size - vga_sz),
        gspFwWprEnd=vga_off, frtsSize=(frts_sz := 0x100000), frtsOffset=(frts_off := vga_off - frts_sz),
        bootBinOffset=(boot_off := frts_off - boot_sz),
        gspFwOffset=(gsp_off := round_down(boot_off - radix3_sz, 0x10000)), gspFwHeapSize=(gsp_heap_sz := 0x8100000),
        fbSize=self.nvdev.vram_size,
        gspFwHeapOffset=(gsp_heap_off := round_down(gsp_off - gsp_heap_sz, 0x100000)),
        gspFwWprStart=(wpr_st := round_down(gsp_heap_off - 0x1000, 0x100000)),
        nonWprHeapSize=(non_wpr_sz := 0x100000), nonWprHeapOffset=(non_wpr_off := round_down(wpr_st - non_wpr_sz, 0x100000)),
        gspFwRsvdStart=non_wpr_off)
      assert self.nvdev.flcn.frts_offset == m.frtsOffset, f"FRTS mismatch: {self.nvdev.flcn.frts_offset} != {m.frtsOffset}"
    tprint(f"WPR meta layout: bootloader={boot_sz:#x} radix3={radix3_sz:#x} "
           f"frts={getattr(m, 'frtsSize', 0):#x} fmc={self.nvdev.fmc_boot} sram_stream={sram_boot}")
    with _tscope():
      if sram_boot:
        self._boot_sram, self.wpr_meta_sysmem = self._build_sram_wpr(m), 0x200000
      else:
        self.wpr_meta, _, wpr_meta_addrs = self.nvdev._alloc_boot_mem(ctypes.sizeof(type(m)), data=bytes(m))
        self.wpr_meta_sysmem = wpr_meta_addrs[0]
      tprint(f"WPR meta stored @{self.wpr_meta_sysmem:#x}")
    stage_done("images + WPR ready")

  def _pramin_write_vram(self, paddr, data, label):
    """Write physical VRAM through BAR0's 64 KiB PRAMIN window and verify each chunk."""
    data = bytes(data)
    if (paddr | len(data)) & 3: raise ValueError(f"PRAMIN {label} write must be dword aligned")
    old_window = self.nvdev.rreg(0x1700)
    try:
      for off in range(0, len(data), 0x10000):
        addr, chunk = paddr + off, data[off:off + 0x10000]
        self.nvdev.wreg(0x1700, (addr >> 16) & 0xffffff)  # TARGET=VID_MEM
        view = self.nvdev.mmio.view(0x700000 + (addr & 0xffff), len(chunk), fmt='B')
        view[:] = chunk
        observed = bytes(view[:])
        if observed != chunk:
          mismatch = next((i for i, (got, want) in enumerate(zip(observed, chunk)) if got != want), 0)
          raise RuntimeError(f"PRAMIN {label} verify failed at VRAM {addr + mismatch:#x}")
        if DEBUG >= 2 and (off == 0 or off + len(chunk) == len(data) or (off & 0x7fffff) == 0):
          print(f"PRAMIN {label}: {off + len(chunk):#x}/{len(data):#x}")
    finally:
      self.nvdev.wreg(0x1700, old_window)

  def preload_wpr_for_resume(self):
    """Diagnostic: preload a signed normal-boot image, then exercise SEC2's mailbox-zero resume path."""
    meta = bytearray(self.wpr_meta[:ctypes.sizeof(nv.GspFwWprMeta)])
    m = nv.GspFwWprMeta.from_buffer_copy(meta)
    struct.pack_into('<Q', meta, 200, 1)  # bootCount: image is already resident
    struct.pack_into('<Q', meta, 248, nv.GSP_FW_WPR_META_VERIFIED)
    tprint(f"USB WPR preload/resume: ELF {len(self.gsp_image):#x}->{m.gspFwOffset:#x}, "
           f"bootloader {len(self.booter_image):#x}->{m.bootBinOffset:#x}, meta->{m.gspFwWprStart:#x}")
    with _tscope():
      self._pramin_write_vram(m.gspFwOffset, self.gsp_image, "GSP ELF")
      self._pramin_write_vram(m.bootBinOffset, self.booter_image, "bootloader")
      self._pramin_write_vram(m.gspFwWprStart, meta, "WPR meta")
    tprint("USB WPR preload verified; SEC2 will receive mailbox 0 (GC6-resume contract)")

  def promote_ctx(self, client, subdevice, obj, ctxbufs, bufs=None, virt=None, phys=None):
    res, prom = {}, nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(ctxbufs), engineType=0x1, hChanClient=client, hObject=obj)
    for i, (buf, desc) in enumerate(ctxbufs.items()):
      use_v, use_p = (desc.virt if virt is None else virt), (desc.phys if phys is None else phys)
      x = (bufs or {}).get(buf, self.nvdev.mm.valloc(desc.size, contiguous=True))
      prom.promoteEntry[i] = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY(bufferId=buf,
        gpuVirtAddr=x.va_addr if use_v else 0, bInitialize=use_p,
        gpuPhysAddr=x.paddrs[0][0] if use_p else 0, size=desc.size if use_p else 0,
        physAttr=0x4 if use_p else 0, bNonmapped=(use_p and not use_v))
      res[buf] = x
    self.rpc_rm_control(hObject=subdevice, cmd=nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX, params=prom, client=client)
    return res

  # =========================================================================
  # S5.1/S6: Golden image — root/dev/subdev/VASpace + PROMOTE_CTX for golden channel.
  # =========================================================================
  def init_golden_image(self):
    stage_set(6, "root/dev/subdev/VASpace/PROMOTE_CTX")
    self.rpc_rm_alloc(hParent=0x0, hClass=0x0, params=nv_gpu.NV0000_ALLOC_PARAMETERS())
    dev = self.rpc_rm_alloc(hParent=self.priv_root, hClass=nv_gpu.NV01_DEVICE_0,
      params=nv_gpu.NV0080_ALLOC_PARAMETERS(hClientShare=self.priv_root))
    subdev = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.NV20_SUBDEVICE_0, params=nv_gpu.NV2080_ALLOC_PARAMETERS())
    vaspace = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.FERMI_VASPACE_A, params=nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS())
    self.vaspace = vaspace  # exposed for NVDevice.__init__ reuse
    di = self.rpc_rm_control(subdev, nv_gpu.NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE,
      nv_gpu.NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS())
    self.runlists = {di.entries[i].engineData[2]: di.entries[i].engineData[3] for i in range(di.numEntries)}
    res_va = self.nvdev.mm.alloc_vaddr(res_sz := (512 << 20))
    bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(
      pageSize=res_sz, numLevelsToCopy=3, virtAddrLo=res_va, virtAddrHi=res_va + res_sz - 1)
    for i, pt in enumerate(self.nvdev.mm.page_tables(res_va, size=res_sz)):
      bufs_p.levels[i] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_level(
        physAddress=pt.paddr, size=self.nvdev.mm.pte_cnt[0] * 8 if i == 0 else 0x1000,
        pageShift=self.nvdev.mm.pte_covers[i].bit_length() - 1, aperture=1)
    self.rpc_rm_control(hObject=vaspace, cmd=nv_gpu.NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, params=bufs_p)
    gpfifo_area = self.nvdev.mm.valloc(4 << 10, contiguous=True)
    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.paddrs[0][0] + 0x20 * 8, size=0x20, addressSpace=2, cacheAttrib=0)
    gg_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr, gpFifoEntries=32,
      engineType=0x1, cid=3, hVASpace=vaspace, userdOffset=(ctypes.c_uint64 * 8)(0x20 * 8),
      userdMem=userd, internalFlags=0x1a, flags=0x200320)
    ch_gpfifo = self.rpc_rm_alloc(hParent=dev, hClass=self.gpfifo_class, params=gg_params)
    gr_ctx_bufs_info = self.rpc_rm_control(hObject=subdev,
      cmd=nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO,
      params=nv_gpu.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS()).engineContextBuffersInfo[0]
    def _ctx_info(idx, add=0, align=None):
      return round_up(gr_ctx_bufs_info.engine[idx].size + add, align or gr_ctx_bufs_info.engine[idx].alignment)
    gr_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS, add=0x40000)
    patch_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH)
    cfgs_sizes = {x: _ctx_info(x + 14, align=(2 << 20) if x == 5 else None) for x in range(3, 11)}
    self.grctx_bufs = {0: GRBufDesc(gr_size, phys=True, virt=True),
      1: GRBufDesc(patch_size, phys=True, virt=True, local=True),
      2: GRBufDesc(patch_size, phys=True, virt=True),
      **{x: GRBufDesc(cfgs_sizes[x], phys=False, virt=True) for x in range(3, 7)},
      9: GRBufDesc(cfgs_sizes[9], phys=True, virt=True), 10: GRBufDesc(cfgs_sizes[10], phys=True, virt=False),
      11: GRBufDesc(cfgs_sizes[10], phys=True, virt=True)}
    self.promote_ctx(self.priv_root, subdev, ch_gpfifo, {k: v for k, v in self.grctx_bufs.items() if not v.local})
    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.compute_class, params=None)
    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.dma_class, params=None)
    stage_done("golden channel live")
    return ch_gpfifo, subdev, dev, vaspace, gpfifo_area

  # =========================================================================
  # S5: GSP hardware init — wait INIT_DONE, BAR1 block rebase, golden image.
  #     Also handles run_cpu_seq events (GSP_SCRIPTED_FALCON_DMA).
  # =========================================================================
  def init_hw(self):
    tprint("GSP init: waiting INIT_DONE, BAR1 block rebase, golden image")
    stage_set(5, "INIT_DONE + cpu sequencer")
    with _tscope():
      self.stat_q = NVRpcQueue(self, self.stat_q_view, self.cmd_q_view)
      self.cmd_q.rx_view = self.stat_q_view.view(self.stat_q.tx.rxHdrOff, fmt='I')
      self.stat_q.wait_resp(nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE)
      stage_done("INIT_DONE received")
      self.nvdev.NV_PBUS_BAR1_BLOCK.write(mode=0, target=0, ptr=0)
      if self.nvdev.fmc_boot: self.nvdev.NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR.write(mode=0, target=0, ptr=0)
      self.priv_root = 0xc1e00004
      return self.init_golden_image()

  def fini_hw(self): self.rpc_unloading_guest_driver()

  def rpc_alloc_memory(self, hDevice, hClass, paddrs, length, flags, client=None):
    assert all(sz == 0x1000 for _, sz in paddrs)
    rpc = nv.rpc_alloc_memory_v(hClient=(client := client or self.priv_root), hDevice=hDevice, hMemory=(handle := next(self.handle_gen)),
      hClass=hClass, flags=flags, pteAdjust=0, format=6, length=length, pageCount=len(paddrs))
    rpc.pteDesc.idr, rpc.pteDesc.length = nv.NV_VGPU_PTEDESC_IDR_NONE, (len(paddrs) & 0xffff)
    payload = bytes(rpc) + b''.join(bytes(nv.struct_pte_desc_pte_pde(pte=(paddr >> 12))) for paddr, _ in paddrs)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY, bytes(payload))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY)
    return handle

  def rpc_rm_alloc(self, hParent, hClass, params, client=None):
    if hClass == self.gpfifo_class:
      ramfc_alloc = self.nvdev.mm.valloc(0x1000, contiguous=True)
      params.ramfcMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x200, addressSpace=2, cacheAttrib=0)
      params.instanceMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x1000, addressSpace=2, cacheAttrib=0)
      _, method_paddr, _ = self.nvdev._alloc_boot_mem(0x5000, sysmem=False)
      params.mthdbufMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=method_paddr, size=0x5000, addressSpace=2, cacheAttrib=0)
      if client is not None and client != self.priv_root and params.hObjectError != 0:
        params.errorNotifierMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=0, size=0xecc, addressSpace=0, cacheAttrib=0)
        params.userdMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=params.hUserdMemory[0] + params.userdOffset[0], size=0x400, addressSpace=2, cacheAttrib=0)
    alloc_args = nv.rpc_gsp_rm_alloc_v(hClient=(client := client or self.priv_root), hParent=hParent,
      hObject=(obj := next(self.handle_gen)), hClass=hClass, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, bytes(alloc_args) + (bytes(params) if params is not None else b''))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC)
    if TRACE and not TRACE_RAW:
      cls_nm = _trace_class_name(hClass) or f"{hClass:#x}"
      ex = []
      for k in ("limit", "vaBase", "vaSize", "flags", "gpFifoOffset", "gpFifoEntries", "engineType", "cid"):
        v = getattr(params, k, None) if params is not None else None
        if isinstance(v, int) and v: ex.append((k, f"{v:#x}"))
      rm_record("ALLOC", cls_nm, obj=obj, parent=hParent, extras=ex)
    if hClass == self.gpfifo_class:
      engine = params.engineType
      self.chan_runlists[obj] = self.runlists.get(engine + 10 * (engine >= nv_gpu.NV2080_ENGINE_TYPE_NVDEC0), 0)
    if hClass == nv_gpu.FERMI_VASPACE_A and client != self.priv_root:
      self.rpc_set_page_directory(device=hParent, hVASpace=obj, pdir_paddr=self.nvdev.mm.root_page_table.paddr, client=client)
    if hClass == nv_gpu.NV01_DEVICE_0 and client != self.priv_root: self.device = obj
    if hClass == nv_gpu.NV20_SUBDEVICE_0: self.subdevice = obj
    if hClass == self.compute_class and client != self.priv_root:
      phys_gr_ctx = self.promote_ctx(client, self.subdevice, hParent, {k: v for k, v in self.grctx_bufs.items() if k in [0, 1, 2]}, virt=False)
      self.promote_ctx(client, self.subdevice, hParent, {k: v for k, v in self.grctx_bufs.items() if k in [0, 1, 2]}, phys_gr_ctx, phys=False)
    return obj if hClass != nv_gpu.NV1_ROOT else client

  def rpc_rm_control(self, hObject, cmd, params, client=None, extra=None):
    if cmd == nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES:
      self.rpc_rm_control(hObject, nv_gpu.NVB0CC_CTRL_CMD_INTERNAL_PERMISSIONS_INIT,
        nv_gpu.NVB0CC_CTRL_INTERNAL_PERMISSIONS_INIT_PARAMS(
          bAdminProfilingPermitted=1, bDevProfilingPermitted=1, bCtxProfilingPermitted=1,
          bVideoMemoryProfilingPermitted=1, bSysMemoryProfilingPermitted=1), client=client)
    elif cmd == nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM:
      params.hMemPmaBuffer = self.rpc_alloc_memory(self.device, nv_gpu.NV01_MEMORY_LIST_SYSTEM, extra[0].meta.mapping.paddrs, extra[0].size,
        pma_flags := (nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4 | nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30), client=client)
      params.hMemPmaBytesAvailable = self.rpc_alloc_memory(self.device, nv_gpu.NV01_MEMORY_LIST_SYSTEM, extra[1].meta.mapping.paddrs, extra[1].size,
        pma_flags | nv_gpu.NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES << 21, client=client)
    control_args = nv.rpc_gsp_rm_control_v(hClient=(client := client or self.priv_root), hObject=hObject, cmd=cmd, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, bytes(control_args) + (bytes(params) if params is not None else b''))
    res = self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL)
    if TRACE and not TRACE_RAW:
      obj_cls = _HANDLE_CLS.get(hObject, f"{hObject:#08x}")
      ex = [("params", f"{ctypes.sizeof(params):#x}")] if params is not None else []
      rm_record("CTRL", obj_cls, obj=hObject, cmd=cmd, extras=ex)
    st = type(params).from_buffer_copy(res[len(bytes(control_args)):]) if params is not None else None
    if cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      cast(nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS, st).workSubmitToken |= \
        (self.chan_runlists[hObject] << 16) | ((1 << 30) if self.nvdev.chip_name.startswith("GB2") else 0)
    return st

  def rpc_set_page_directory(self, device, hVASpace, pdir_paddr, client=None, pasid=0xffffffff):
    params = nv.struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05(physAddress=pdir_paddr,
      numEntries=self.nvdev.mm.pte_cnt[0], flags=0x8, hVASpace=hVASpace, pasid=pasid, subDeviceId=1, chId=0)
    alloc_args = nv.rpc_set_page_directory_v(hClient=client or self.priv_root, hDevice=device, pasid=pasid, params=params)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, bytes(alloc_args))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY)
    if TRACE and not TRACE_RAW:
      rm_record("SETPD", "PAGE_DIRECTORY", parent=device,
                        extras=[("pasid", f"{pasid:#x}"), ("pdir", f"{pdir_paddr:#x}"), ("entries", str(self.nvdev.mm.pte_cnt[0]))])

  def rpc_set_gsp_system_info(self):
    def bdf_as_int(s):
      if s.startswith("usb") or s.startswith("remote"): return 0x000
      return (int(s[5:7], 16) << 8) | (int(s[8:10], 16) << 3) | int(s[-1], 16)
    with _tscope():
      pcidev = self.nvdev.pci_dev
      data = nv.GspSystemInfo(gpuPhysAddr=pcidev.bar_info(0)[0], gpuPhysFbAddr=pcidev.bar_info(1)[0],
        gpuPhysInstAddr=pcidev.bar_info(3)[0],
        pciConfigMirrorBase=[0x88000, 0x92000][self.nvdev.fmc_boot], pciConfigMirrorSize=0x1000,
        nvDomainBusDeviceFunc=bdf_as_int(self.nvdev.devfmt), bIsPassthru=1,
        PCIDeviceID=pcidev.read_config(pci.PCI_VENDOR_ID, 4), PCISubDeviceID=pcidev.read_config(pci.PCI_SUBSYSTEM_VENDOR_ID, 4),
        PCIRevisionID=pcidev.read_config(pci.PCI_REVISION_ID, 1), maxUserVa=0x7ffffffff000)
      self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, bytes(data))

  def rpc_unloading_guest_driver(self):
    data = nv.rpc_unloading_guest_driver_v(bInPMTransition=0, bGc6Entering=0, newLevel=1 << 6)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, bytes(data))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER)

  def rpc_set_registry_table(self):
    table = {'RMForcePcieConfigSave': 0x1, 'RMSecBusResetEnable': 0x1}
    entries_bytes, data_bytes = bytes(), bytes()
    hdr_size, entries_size = ctypes.sizeof(nv.PACKED_REGISTRY_TABLE), ctypes.sizeof(nv.PACKED_REGISTRY_ENTRY) * len(table)
    with _tscope():
      for k, v in table.items():
        entries_bytes += bytes(nv.PACKED_REGISTRY_ENTRY(nameOffset=hdr_size + entries_size + len(data_bytes),
          type=nv.REGISTRY_TABLE_ENTRY_TYPE_DWORD, data=v, length=4))
        data_bytes += k.encode('utf-8') + b'\x00'
      header = nv.PACKED_REGISTRY_TABLE(size=hdr_size + len(entries_bytes) + len(data_bytes), numEntries=len(table))
      self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_REGISTRY, bytes(header) + entries_bytes + data_bytes)

  # =========================================================================
  # S5.1: GSP CPU sequencer — execute scripted falcon DMA + register ops.
  # =========================================================================
  def run_cpu_seq(self, seq_buf):
    hdr = nv.rpc_run_cpu_sequencer_v17_00.from_buffer_copy(seq_buf[:(hdr_sz := ctypes.sizeof(nv.rpc_run_cpu_sequencer_v17_00))])
    tprint(f"GSP CPU sequencer: {hdr.cmdIndex} cmds size={len(seq_buf):#x}")
    with _tscope():
      self._run_seq_ops(seq_buf[hdr_sz:], hdr)

  # =========================================================================
  # S5.1: Decode and execute run_cpu_seq opcodes (write/masked-write/wait/..).
  # =========================================================================
  def _run_seq_ops(self, seq_ops, hdr):
    cmd_iter = iter(memoryview(seq_ops).cast('I')[:hdr.cmdIndex])
    dma = None   # active DMA_RUN accumulator (falcon DMATRF register pattern)
    def dma_reg(a):
      nm = _resolve_reg_name(self.nvdev, a)
      return nm if nm and "DMATRF" in nm else None
    def flush_dma():
      # Fold consecutive same-cmd falcon DMA chunks into ONE summary:
      #   CPU_SEQ DMA_RUN cmd=0x614 x64 / src .. -> .. step .. / dst .. / waits=N
      nonlocal dma
      if dma is not None: set_wire_suppress(False)   # summary replaces constituents
      if not dma or dma["n"] == 0:
        dma = None
        return
      n, rows = dma["n"], []
      for lbl, a0, al in (("src", dma["fb0"], dma["fbl"]), ("dst", dma["m0"], dma["ml"])):
        if a0 is None or al is None: continue
        if n > 1:
          rows.append(f"{lbl} {a0:#x} -> {al:#x} step {(al - a0) // (n - 1):+#x}")
        else: rows.append(f"{lbl} {a0:#x}")
      rows.append(f"waits={dma['w']} status=completed")
      tprint(f"CPU_SEQ DMA_RUN cmd={dma['cmd']:#x} x{n}\n    " + "\n    ".join(rows))
      dma = None
    for op in cmd_iter:
      # Collapse the GSP-scripted falcon DMA pattern: MOFFS/FBOFFS/CMD writes
      # + DMATRFCMD polls repeat per 256B chunk; render ONE summary per run.
      if op == 0x0:
        a, v = next(cmd_iter), next(cmd_iter)
        dr = dma_reg(a)
        if dr:
          # Detector owns these writes: always send them, never print them.
          if dma is None:
            dma = {"cmd": None, "n": 0, "w": 0, "m0": None, "ml": None,
                   "fb0": None, "fbl": None, "pm": None, "pf": None}
            set_wire_suppress(True)
        self.nvdev.wreg(a, v)   # ALWAYS write: detection may never swallow traffic
        if dr and "MOFFS" in dr:
          dma["pm"] = v
          if dma["m0"] is None: dma["m0"] = v
        elif dr and "FBOFFS" in dr:
          dma["pf"] = v
          if dma["fb0"] is None: dma["fb0"] = v
        elif dr and "CMD" in dr:
          if dma["cmd"] is None:
            dma["cmd"] = v; dma["n"] = 1
            dma["ml"], dma["fbl"] = dma.get("pm"), dma.get("pf")
          elif v == dma["cmd"]:
            dma["n"] += 1; dma["ml"] = dma.get("pm"); dma["fbl"] = dma.get("pf")
          else:
            # new cmd: capture this chunk's pending values BEFORE flush clears state
            pm, pf = dma.get("pm"), dma.get("pf")
            set_wire_suppress(True)   # re-arm: new run stays suppressed too
            flush_dma()
            dma = {"cmd": v, "n": 1, "w": 0, "m0": pm, "ml": pm,
                   "fb0": pf, "fbl": pf, "pm": pm, "pf": pf}
      elif op == 0x1:
        addr, val, mask = next(cmd_iter), next(cmd_iter), next(cmd_iter)
        self.nvdev.wreg(addr, (self.nvdev.rreg(addr) & ~mask) | (val & mask))
      elif op == 0x2:
        addr, mask, val, _, _ = next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter)
        if dma and dma_reg(addr) and "CMD" in (dma_reg(addr) or ""): dma["w"] += 1
        _rnm = _resolve_reg_name(self.nvdev, addr) or f"{addr:#x}"
        with quiet():   # per-chunk polls are summarized by DMA_RUN waits=N
          wait_cond(lambda a, m: (self.nvdev.rreg(a) & m), addr, mask, value=val, msg=f"{_rnm} & {mask:#x} == {val:#x}")
      elif op == 0x4:
        flush_dma()
        addr, index = next(cmd_iter), next(cmd_iter)
        hdr.regSaveArea[index] = self.nvdev.rreg(addr)
      elif op == 0x3:
        flush_dma(); time.sleep(next(cmd_iter) / 1e6)
      elif op == 0x5:
        flush_dma()
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon)
        self.nvdev.flcn.disable_ctx_req(self.nvdev.flcn.falcon)
      elif op == 0x6: self.nvdev.flcn.start_cpu(self.nvdev.flcn.falcon)
      elif op == 0x7: self.nvdev.flcn.wait_cpu_halted(self.nvdev.flcn.falcon)
      elif op == 0x8:
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon, riscv=True)
        self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.libos_args_sysmem))
        self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.libos_args_sysmem))
        self.nvdev.flcn.start_cpu(self.nvdev.flcn.sec2)
        wait_cond(lambda: self.nvdev.NV_PGC6_BSI_SECURE_SCRATCH_14.read_bitfields()['boot_stage_3_handoff'], msg="SEC2 boot handoff complete")
        mailbox = self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(self.nvdev.flcn.sec2).read()
        assert mailbox == 0x0, f"Falcon SEC2 failed to execute, mailbox is {mailbox:08x}"
      else: raise ValueError(f"Unknown op code {op} in run_cpu_seq")
    flush_dma()


# ============================================================================
# ops_nv.py slices (vendored from ref/tinygrad/tinygrad/runtime/ops_nv.py)
# Slimmed: GPFifo, PCIIface, NVProgram, NVAllocator, NVDevice, NVSignal,
#          NVComputeQueue (minimal), QMD, NVArgsState.
# ============================================================================
SignalType = TypeVar('SignalType', bound='HCQSignal')

class HCQCompiled:
  """Base class for hardware command queue compiled targets."""
  peer_groups: ClassVar[dict] = {}
  signal_t: ClassVar[type] = None  # set by NVSignal below

class NVSignal:
  """Timeline signal: GPU writes value to signal_page, CPU polls."""
  def __init__(self, value=0, owner=None):
    self.value_addr = 0  # assigned by NVDevice.signal_page
    self._value = value
    self.owner = owner
  @property
  def value(self):
    if self.owner is not None and hasattr(self.owner, '_signal_page') and self.value_addr:
      return int(self.owner._signal_page.cpu_view().view(0, 8, 'Q')[0])
    return self._value
  @value.setter
  def value(self, v):
    self._value = v
    if self.owner is not None and hasattr(self.owner, '_signal_page') and self.value_addr:
      self.owner._signal_page.cpu_view().view(0, 8, 'Q')[0] = v
  def _sleep(self, time_ms: int):
    if time_ms > 200 and self.owner is not None: self.owner.iface.sleep(200)

@dataclasses.dataclass
class GPFifo:
  """GPU FIFO ring: entry descriptor + GPPUT doorbell register."""
  ring: MMIOInterface
  gpput: MMIOInterface
  entries_count: int
  token: int
  put_value: int = 0

# -------- NV command queues (from ref/tinygrad ops_nv.py) --------
class NVCommandQueue:
  """Command queue builder: nvm() method calls, wait/signal, GPFIFO submit."""
  def __init__(self):
    self._q: list[int] = []
    self.active_qmd = None

  def nvm(self, subchannel, mthd, *args, typ=2):
    self._q.extend([(typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2), *args])
    return self

  def setup(self, compute_class=None, copy_class=None, local_mem_window=None, shared_mem_window=None,
            local_mem=None, local_mem_tpc_bytes=None):
    if compute_class: self.nvm(1, nv_gpu.NVC6C0_SET_OBJECT, compute_class)
    if copy_class: self.nvm(4, nv_gpu.NVC6C0_SET_OBJECT, copy_class)
    if local_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(local_mem_window))
    if shared_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(shared_mem_window))
    if local_mem: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(local_mem))
    if local_mem_tpc_bytes: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(local_mem_tpc_bytes), 0xff)
    return self

  def wait(self, signal, value=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             nv_flags("NVC56F_SEM_EXECUTE", operation="acq_circ_geq", payload_size="64bit"))
    self.active_qmd = None
    return self

  def signal(self, signal, value=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit", release_timestamp="en"))
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    self.active_qmd = None
    return self

  def _submit_to_gpfifo(self, dev, gpfifo: GPFifo):
    mthd_rows = []
    dec = list(decode_words(self._q))
    i = 0
    while i < len(dec):
      index, typ, subc, method, name, args = dec[i]
      nm = _trace_method_name(method) or name
      consumed = 1
      if TRACE:
        # Semantic one-liners for the launch-critical methods; generic fallback
        # keeps every other method visible with its decoded args.
        if method == 0x0 and len(args) == 1:
          cls = _trace_class_name(args[0]) or f"{args[0]:#x}"
          txt = f"BIND      {cls} ({args[0]:#x})"
        elif tuple(M_SLM) == (method, dec[i + 1][3] if i + 1 < len(dec) else None):
          # SET_SHADER_LOCAL_MEMORY_A/B pair: 64-bit VA across two methods.
          base = (args[0] << 32) | dec[i + 1][5][0]
          txt = f"SET_SHADER_LOCAL_MEMORY base={base:#x}"
          consumed = 2
        elif tuple(M_SLNT) == (method, dec[i + 1][3] if i + 1 < len(dec) else None, dec[i + 2][3] if i + 2 < len(dec) else None):
          # NON_THROTTLED A/B/C: SIZE_UPPER(A), SIZE_LOWER(B), MAX_SM_COUNT(C).
          size = (args[0] << 32) | dec[i + 1][5][0]
          txt = f"SET_SHADER_LOCAL_MEMORY_NON_THROTTLED size={size:#x}{hsz(size)} max_sm_count={dec[i + 2][5][0]:#x}"
          consumed = 3
        elif method == 0x005c and len(args) == 5:
          sem, pay, ex = (args[1] << 32) | args[0], (args[3] << 32) | args[2], args[4]
          oname = _sem_op_name(ex & 0xf)
          if sem == dev.timeline_signal.value_addr:
            # Our timeline semaphore: show the value transition, not the address.
            if "release" in oname: txt = f"SIGNAL    timeline {max(pay - 1, 0)} -> {pay}"
            elif oname.startswith("acq"): txt = f"WAIT      timeline >= {pay} ({oname})"
            else: txt = f"SEM       timeline op={oname} payload={pay}"
          else:
            pl = "64b" if ex & (1 << 24) else "32b"
            if oname.startswith("acq"): txt = f"WAIT      semaphore[{sem:#x}] >= {pay} ({oname},{pl})"
            elif "release" in oname: txt = f"SIGNAL    semaphore[{sem:#x}] = {pay} ({pl})"
            else: txt = f"SEM       semaphore[{sem:#x}] op={oname} payload={pay}"
        elif method == 0x1698 and len(args) == 1:
          on = "|".join(n for bit, n in [(0x1, "icache"), (0x10, "global_data"), (0x1000, "constant")] if args[0] & bit)
          txt = f"INVALIDATE {on or 'none'}"
        elif method == 0x02b4 and len(args) == 1: txt = f"QMD       -> {args[0] << 8:#x}"
        elif method == 0x02c0 and len(args) == 1: txt = f"LAUNCH    pcas2 action={args[0]:#x}"
        elif "SHARED_MEMORY_WINDOW" in nm: txt = f"shader shared mem window = {(args[0] << 32) | args[1]:#x}"
        elif "LOCAL_MEMORY_WINDOW" in nm: txt = f"shader local mem window  = {(args[0] << 32) | args[1]:#x}"
        else: txt = f"{nm} subc={subc} [{' '.join(f'{a:08x}' for a in args)}]" if args else f"{nm} subc={subc}"
        mthd_rows.append((index, txt))
      i += consumed
    cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4, 16)
    cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
    entry = (cmdq_addr // 4 << 2) | (len(self._q) << 42) | (1 << 41)
    put = gpfifo.put_value % gpfifo.entries_count
    new_put = (gpfifo.put_value + 1) % gpfifo.entries_count
    # The ordering-sensitive submit trio, collapsed into one decoded block.
    # Raw transactions remain in the TLP sidecar.
    with quiet():
      dev.cmdq[cmdq_wptr:cmdq_wptr + len(self._q)] = _array_mod.array('I', [w & 0xffffffff for w in self._q])
      gpfifo.ring[put] = entry
      gpfifo.gpput[0] = new_put
      System.memory_barrier()
      dev.gpu_mmio[0x90 // 4] = gpfifo.token
    gpfifo.put_value += 1
    if TRACE:
      global _SUBMIT_COUNT
      _SUBMIT_COUNT += 1
      fifo_is_compute = gpfifo is dev.compute_gpfifo
      first = getattr(gpfifo, "_subs", 0) == 0
      gpfifo._subs = getattr(gpfifo, "_subs", 0) + 1
      txts = " ".join(t for _, t in mthd_rows)
      kind = "COMPUTE" if fifo_is_compute else "COPY"
      if "LAUNCH " in txts or "LAUNCH   " in txts: role = "KERNEL_LAUNCH"
      elif first: role = "INIT"
      elif "local mem" in txts or "mem window" in txts: role = "MEM_SETUP"
      else: role = "SETUP"
      rows = [f"  {txt}" for _, txt in mthd_rows]
      if TRACE_RAW: rows.append(f"  raw: {' '.join(f'{w:08x}' for w in self._q)}")
      rows.append(f"  GPFIFO[{put}] -> GPPUT={new_put} -> DOORBELL token={gpfifo.token:#x}")
      _trace("SUBMIT", f"#{_SUBMIT_COUNT} [{kind}_{role}] {type(self).__name__} PB={cmdq_addr:#x} words={len(self._q)}\n" + "\n".join(rows))
    if not dev.iface.is_local():
      dev.iface.sleep(200)
class NVComputeQueue(NVCommandQueue):
  def submit(self, dev):
    self._submit_to_gpfifo(dev, dev.compute_gpfifo)
    return self

class NVCopyQueue(NVCommandQueue):
  def copy(self, dest, src, copy_size):
    for off in range(0, copy_size, 1 << 31):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(src.va_addr + off), *data64(dest.va_addr + off))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(copy_size - off, 1 << 31))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA,
               nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))
    return self

  def signal(self, signal, value=0):
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA,
             nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_four_word_semaphore"))
    return self

  def submit(self, dev):
    self._submit_to_gpfifo(dev, dev.dma_gpfifo)
    return self

# =========================================================================
# S7.16/S9: Submit pre-built method stream to compute or copy GPFIFO.
# =========================================================================
def submit_gpfifo(dev, words, fifo=None):
  """Push a pre-built method stream to compute (default) or copy GPFIFO."""
  q = NVComputeQueue()
  q._q = list(words)
  q._submit_to_gpfifo(dev, fifo or dev.compute_gpfifo)

@dataclasses.dataclass(frozen=True)
class PCIAllocationMeta:
  """Metadata for a PCIe buffer allocation: mapping, CPU mapping flag, hMemory handle."""
  mapping: Any
  has_cpu_mapping: bool
  hMemory: int = 0

# -------- QMD ----------
_QMD_KEY_FIELDS = ["qmd_major_version", "qmd_type",
  "program_address_upper", "program_address_lower",
  "program_address_upper_shifted4", "program_address_lower_shifted4",
  "register_count", "register_count_v", "shared_memory_size",
  "shader_local_memory_high_size", "shader_local_memory_high_size_shifted4",
  "cta_raster_width", "cta_raster_height", "cta_raster_depth",
  "cta_thread_dimension0", "cta_thread_dimension1", "cta_thread_dimension2",
  "constant_buffer_valid_0", "constant_buffer_addr_upper_0", "constant_buffer_addr_lower_0",
  "constant_buffer_addr_upper_shifted6_0", "constant_buffer_addr_lower_shifted6_0",
  "constant_buffer_size_shifted4_0", "api_visible_call_limit", "barrier_count",
  "sass_version", "program_prefetch_size", "program_prefetch_addr_upper_shifted",
  "program_prefetch_addr_lower_shifted",
  "release0_enable", "release0_address_lower", "release0_address_upper",
  "release0_payload_lower", "release0_payload_upper"]

class QMD:
  """Queue Method Descriptor: GPU kernel dispatch configuration (version 3 or 5)."""
  fields: dict = {}

  def __init__(self, dev, view=None, **kwargs):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)
    if (pref := "NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00") not in QMD.fields:
      QMD.fields[pref] = {**{name[len(pref) + 1:]: dt for name, dt in nv_gpu.__dict__.items() if name.startswith(pref) and isinstance(dt, tuple)},
        **{name[len(pref) + 1:] + f"_{i}": dt(i) for name, dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith(pref) and callable(dt)}}
    self.mv, self.pref = (memoryview(bytearray(self.sz * 4)) if view is None else view), pref
    if kwargs: self.write(**kwargs)

  def _rw_bits(self, hi, lo, value=None):
    mask = ((1 << (width := hi - lo + 1)) - 1) << (lo % 8)
    num = int.from_bytes(self.mv[lo // 8:hi // 8 + 1], "little")
    if value is None: return (num & mask) >> (lo % 8)
    if value >= (1 << width): raise ValueError(f"{value:#x} does not fit.")
    self.mv[lo // 8:hi // 8 + 1] = int((num & ~mask) | ((value << (lo % 8)) & mask)).to_bytes((hi // 8 - lo // 8 + 1), "little")

  def write(self, **kwargs):
    # traceback.print_exc()
    # breakpoint()
    for k, val in kwargs.items(): self._rw_bits(*QMD.fields[self.pref][k.upper()], value=val)
  def read(self, k, val=0): return self._rw_bits(*QMD.fields[self.pref][k.upper()])
  def field_offset(self, k): return QMD.fields[self.pref][k.upper()][1] // 8
  def set_constant_buf_addr(self, i, addr):
    if self.ver < 4:
      self.write(**{f'constant_buffer_addr_upper_{i}': hi32(addr), f'constant_buffer_addr_lower_{i}': lo32(addr)})
    else:
      self.write(**{f'constant_buffer_addr_upper_shifted6_{i}': hi32(addr >> 6), f'constant_buffer_addr_lower_shifted6_{i}': lo32(addr >> 6)})


# -------- NVProgram ----------
class NVArgsState:
  """Program argument state: constant buffers + buffer bindings."""
  def __init__(self, buf, prg, bufs, vals=()):
    self.buf, self.prg, self.bufs, self.vals = buf, prg, bufs, vals

class NVProgram:
  """S8 — CUDA program: ELF load, QMD build, constant buffer setup."""
  def __init__(self, dev, name, lib):
    self.dev, self.name, self.lib = dev, name, lib
    self.constbufs = {0: (0, 0x160)}
    image, sections, relocs = elf_loader(self.lib, force_section_align=128)
    tprint(f"program {name}: image={len(image):#x} relocs={len(relocs)}")
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000) + 0x1000)
    prog_addr = self.lib_gpu.va_addr
    self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0x160
    prog_sz = image.nbytes
    for sh in sections:
      if sh.name == f".nv.shared.{name}":
        self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
      if sh.name == f".text.{name}":
        prog_addr, prog_sz = self.lib_gpu.va_addr + sh.header.sh_addr, sh.header.sh_size
      elif sh.name.startswith(".nv.info"):
        for typ, param, data in self._parse_elf_info(sh):
          if sh.name == f".nv.info.{name}" and param == 0xa:
            cbuf0_size = struct.unpack_from("IH", data)[1]
          elif sh.name == ".nv.info" and param == 0x12:
            self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240
          elif sh.name == ".nv.info" and param == 0x2f:
            self.regs_usage = struct.unpack_from("II", data)[1]
    for apply_image_offset, rel_sym_offset, typ, _ in relocs:
      if typ == 2:
        image[apply_image_offset:apply_image_offset + 8] = struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset)
      elif typ == 0x38:
        image[apply_image_offset + 4:apply_image_offset + 8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
      elif typ == 0x39:
        image[apply_image_offset + 4:apply_image_offset + 8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
      else:
        raise RuntimeError(f"unknown NV reloc {typ}")
    min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
    self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)
    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
    else:
      self.cbuf_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
    self.dev._ensure_has_local_memory(self.lcmem_usage)
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()
    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1
    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      qmd = {
        'qmd_major_version': 5,
        'qmd_type': nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA,
        'program_address_upper_shifted4': hi32(prog_addr >> 4),
        'program_address_lower_shifted4': lo32(prog_addr >> 4),
        'register_count': self.regs_usage,
        'shared_memory_size_shifted7': self.shmem_usage >> 7,
        'shader_local_memory_high_size_shifted4': self.dev.slm_per_thread >> 4,
      }
    else:
      qmd = {
        'qmd_major_version': 3,
        'sm_global_caching_enable': 1,
        'program_address_upper': hi32(prog_addr),
        'program_address_lower': lo32(prog_addr),
        'shared_memory_size': self.shmem_usage,
        'register_count_v': self.regs_usage,
        'shader_local_memory_high_size': self.dev.slm_per_thread,
      }
    self.qmd = QMD(
      dev, **qmd, qmd_group_id=0x3f,
      invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1,
      api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR,
      constant_buffer_invalidate_0=1,
      min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg,
      max_sm_config_shared_mem_size=0x1a,
      program_prefetch_size=min(prog_sz >> 8, 0x1ff),
      sass_version=dev.sass_version,
      program_prefetch_addr_upper_shifted=prog_addr >> 40,
      program_prefetch_addr_lower_shifted=prog_addr >> 8,
    )
    for i, (addr, sz) in self.constbufs.items():
      self.qmd.set_constant_buf_addr(i, addr)
      self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})
    self.kernargs_alloc_size = round_up(self.constbufs[0][1], 1 << 8) + (8 << 8)

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off + 4:start_off + sz + 4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4


# -------- NVAllocator ----------
class NVAllocator:
  """Buffer allocator: host/device alloc, copyin H2D, copyout D2H."""
  def __init__(self, dev): self.dev = dev
  def alloc(self, size, host=False, uncached=False, contiguous=False, cpu_access=True, zero=False):
    return self.dev.iface.alloc(size, host=host, uncached=uncached, contiguous=contiguous, cpu_access=cpu_access, zero=zero)
  def free(self, b): self.dev.iface.free(b)
  def _copyin(self, dest, src: memoryview):
    if dest.view is None: raise RuntimeError("buffer has no cpu mapping")
    if TRACE:
      mp = getattr(getattr(dest, "meta", None), "mapping", None)
      pd = getattr(mp, "paddrs", None)
      pa = f"pa0={pd[0][0]:#x}" if pd else "pa=?"
      tprint(f"COPYIN va={dest.va_addr:#x} len={len(src):#x} {pa} (H2D via local mmap; GPU reads it back by DMA)")
    dest.cpu_view().view(fmt='B')[:len(src)] = bytes(src)
  def _copyout(self, dest: memoryview, src):
    if src.view is None: raise RuntimeError("buffer has no cpu mapping")
    if TRACE:
      # Provenance for D2H results: the GPU wrote these bytes straight into
      # host RAM via DMA (STG) — that transfer never appears as a PCIe TLP.
      # The CPU side reads a local mmap of the same pages: zero bus traffic.
      mp = getattr(getattr(src, "meta", None), "mapping", None)
      pd = getattr(mp, "paddrs", None)
      backing = "sysmem-fd (GPU DMA target)" if getattr(mp, "aspace", None) is not AddrSpace.PHYS or (
        pd and src.meta.has_cpu_mapping) else "VRAM via BAR1 window"
      pa = f"pa0={pd[0][0]:#x}" + (f"..{pd[-1][0] + pd[-1][1]:#x}" if len(pd) > 1 else "") if pd else "pa=?"
      tprint(f"COPYOUT va={src.va_addr:#x} len={len(dest):#x} {pa} backing={backing}")
    dest[:] = bytes(src.cpu_view()[:len(dest)])

# -------- Module-level NVDevice helpers --------
def wait_signal(signal, value, timeout_ms=30000):
  start = time.perf_counter()
  while signal.value < value:
    if (time.perf_counter() - start) * 1000 > timeout_ms:
      print(f"wait_signal timeout: signal={signal.value} want>={value}", flush=True)
      return False
    time.sleep(0.001)
  return True

# -------- PCIIfaceBase ----------
class PCIIfaceBase:
  """PCIe interface: alloc/free with sysmem fallback, buffer map."""
  def __init__(self, dev, vram_bar, va_start, va_size, dev_impl_t):
    self.dev, self.vram_bar, self.count = dev, vram_bar, 1
    self.dev_impl = dev_impl_t(self.pci_dev) if isinstance(getattr(self, 'pci_dev', None), object) else None
  @property
  def peer_group(self): return getattr(self.pci_dev, 'peer_group', type(self.pci_dev).__name__)
  def is_local(self): return not isinstance(self.pci_dev, RemotePCIDevice)
  def is_bar_small(self): return self.pci_dev.bar_info(self.vram_bar)[1] == (256 << 20)
  def alloc(self, size, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, zero=False, **kwargs):
    should_use_sysmem = host or ((cpu_access if self.is_bar_small() else (uncached and cpu_access)) and not force_devmem)
    size = round_up(size, PAGESIZE if should_use_sysmem else ((2 << 20) if size >= (8 << 20) else (4 << 10)))
    if should_use_sysmem:
      va = self.dev_impl.mm.alloc_vaddr(size, align=PAGESIZE)
      memview, paddrs = self.pci_dev.alloc_sysmem(size, vaddr=va, contiguous=contiguous)
      mapping = self.dev_impl.mm.map_range(va, size, [(p, 0x1000) for p in paddrs], aspace=AddrSpace.SYS, snooped=True, uncached=True)
      return HCQBuffer(va, size, meta=PCIAllocationMeta(mapping, True, paddrs[0]), view=memview, owner=self.dev)
    mapping = self.dev_impl.mm.valloc(size, uncached=uncached, contiguous=cpu_access, zero=zero)
    barview = self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], size=mapping.size) if cpu_access else None
    return HCQBuffer(mapping.va_addr, size, view=barview, meta=PCIAllocationMeta(mapping, cpu_access, mapping.paddrs[0][0]), owner=self.dev)
  def free(self, b):
    if b.owner != self.dev: self.dev.iface.dev_impl.mm.unmap_range(b.va_addr, round_up(b.size, 0x1000))
    if b.owner == self.dev and b.meta.mapping.aspace is AddrSpace.PHYS: self.dev_impl.mm.vfree(b.meta.mapping)
    if b.owner == self.dev and self.is_local() and b.meta.has_cpu_mapping: FileIOInterface.munmap(b.va_addr, b.size)
  def map(self, b):
    # Mirrors tinygrad PCIIfaceBase.map: re-map an existing buffer (e.g. host/sysmem signal page)
    # into this device's vaspace page table. Required for the user channel's GPU to write to a
    # host-allocated signal buffer.
    paddrs, aspace = b.meta.mapping.paddrs, b.meta.mapping.aspace
    self.dev_impl.mm.map_range(int(b.va_addr), round_up(b.size, 0x1000), paddrs, aspace=aspace,
                                snooped=True, uncached=b.meta.mapping.uncached)
    return HCQBuffer(b.va_addr, b.size, meta=b.meta, owner=b.owner)
  def sleep(self, timeout): pass

# -------- PCIIface ----------
class PCIIface(PCIIfaceBase):
  """Remote PCIe interface via APLRemotePCIDevice (TinyGPU.app)."""
  def __init__(self, dev, dev_id):
    self.dev = dev
    self.pci_dev = APLRemotePCIDevice("NV", "usb4")
    PCIIfaceBase.__init__(self, dev, 1, NVMemoryManager.va_allocator.base, NVMemoryManager.va_allocator.size, NVDev)
    gsp = self.dev_impl.gsp
    self.gpfifo_class, self.compute_class, self.dma_class = gsp.gpfifo_class, gsp.compute_class, gsp.dma_class
    self.root, self.gpu_instance = 0xc1000000, 0


  def rm_alloc(self, parent, clss, params=None, root=None):
    return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)

  def rm_control(self, obj, cmd, params=None, **kwargs):
    return self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root, **kwargs)

  def setup_usermode(self):
    return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)

  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass
  def device_fini(self): self.dev_impl.fini()
  def sleep(self, timeout):
    for _ in self.dev_impl.gsp.stat_q.read_resp(): pass
    if self.dev_impl.is_err_state: raise RuntimeError("Device fault detected")

class USBIface(PCIIface):
  """NVIDIA PCIe interface through Chestnut's USB F0/F2 firmware API."""
  @staticmethod
  def candidate_ids():
    requested = str(getenv("USBDEV", ""))
    return list(dict.fromkeys(([tuple(int(x, 16) for x in requested.split(':'))] if requested else []) +
                              [(0xadd1, 0x0001), (0x3801, 0x0001)]))
  @classmethod
  def available(cls, dev_id=0):
    visible = [item for vendor, product in cls.candidate_ids() for item in USB3.list_devices(vendor, product)]
    try: return dev_id < len(visible)
    finally:
      for usb_dev, _ in visible: libusb.libusb_unref_device(usb_dev)
  def __init__(self, dev, dev_id):
    ids = self.candidate_ids()
    visible = [item for vendor, product in ids for item in USB3.list_devices(vendor, product)]
    if dev_id >= len(visible):
      for usb_dev, _ in visible: libusb.libusb_unref_device(usb_dev)
      choices = ", ".join(f"{vendor:04x}:{product:04x}" for vendor, product in ids)
      raise RuntimeError(f"NV USB device {dev_id} not found (looked for {choices})")
    for idx, (usb_dev, _) in enumerate(visible):
      if idx != dev_id: libusb.libusb_unref_device(usb_dev)
    self.dev, self.pci_dev = dev, USBPCIDevice("NV", *visible[dev_id])
    PCIIfaceBase.__init__(self, dev, 1, NVMemoryManager.va_allocator.base, NVMemoryManager.va_allocator.size, NVDev)
    gsp = self.dev_impl.gsp
    self.gpfifo_class, self.compute_class, self.dma_class = gsp.gpfifo_class, gsp.compute_class, gsp.dma_class
    self.root, self.gpu_instance = 0xc1000000, 0

  def alloc(self, size, host=False, uncached=False, cpu_access=False, contiguous=False,
            force_devmem=False, zero=False, **kwargs):
    if not host and not cpu_access:
      return super().alloc(size, host=False, uncached=uncached, cpu_access=False,
                           contiguous=contiguous, force_devmem=True, zero=zero, **kwargs)
    mapping = self.dev_impl.mm.valloc_cpu_visible(size := round_up(size, 0x1000), uncached=uncached, zero=zero)
    barview = self.pci_dev.map_bar(self.vram_bar, off=mapping.paddrs[0][0], size=mapping.size)
    return HCQBuffer(mapping.va_addr, size, view=barview,
                     meta=PCIAllocationMeta(mapping, False, mapping.paddrs[0][0]), owner=self.dev)


# -------- NVDevice ----------
class NVDevice:
  """S7–S9 — User-mode GPU device: channels, GPFIFO, signals, synchronize."""
  # =========================================================================
  # S7: User-mode GPU device — NV01_ROOT, device/subdevice/vaspace, channel
  #     group, GPFIFOs, signal page, cmdq, allocator.
  # =========================================================================
  def __init__(self, device=""):
    self.device = device or "NV"
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    iface_name = str(getenv("NV_IFACE", "")).upper()
    if iface_name == "USB": iface_t = USBIface
    elif iface_name in ("", "AUTO"): iface_t = USBIface if USBIface.available(self.device_id) else PCIIface
    else: iface_t = PCIIface  # NV_IFACE=PCI (and legacy non-USB values) force the proven default path.
    self.iface = iface_t(self, self.device_id)

    # tinygrad PCIIface line 561: create NV01_ROOT first
    stage_set(7, "user dev/channels/GPFIFO/timeline")
    tprint("RM user client/device/VA setup")
    with _tscope():
      self.iface.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # GSP init_hw ran in NVDev.__init__ — golden channel + subdev + dev allocated there.
    # The user-mode NVDevice creates its own NV01_DEVICE_0/SUBDEVICE/vaspace/channel_group.

    # NVDevice lines 593-595: device, subdevice, virtmem
    with _tscope():
      device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root,
        vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES)
      self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
      self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
      self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST,
        nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff,
          flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) |
                 (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) |
                 (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))))
      self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL,
        nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(limit=0x1ffffffffffff))
      self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info(
        'num_gpcs', 'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')
      self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)
      self.vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A,
        nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
          flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED))

    tprint("Channel group + context share")
    with _tscope():
      channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
      self.channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)
      self.gpfifo_area = self.iface.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True)
      ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=self.vaspace,
        flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
      self.ctxshare = self.iface.rm_alloc(self.channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A, ctxshare_params)

    # usermode + gpu_mmio must exist before any GPFIFO submit (tinygrad NVDevice lines 608-609)
    with _tscope():
      self.usermode, self.gpu_mmio = self.iface.setup_usermode()
      self.compute_gpfifo, self.compute_channel = self._new_gpu_fifo(self.gpfifo_area, self.ctxshare, self.channel_group,
        offset=0, entries=0x10000, compute=True)
      self.dma_gpfifo, self.dma_channel = self._new_gpu_fifo(self.gpfifo_area, self.ctxshare, self.channel_group,
        offset=0x100000, entries=0x10000, compute=False)
      self.iface.rm_control(self.channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
        nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
    print(f"user compute_gpfifo ring_size={len(self.compute_gpfifo.ring)} token=0x{self.compute_gpfifo.token:x}", flush=True)
    # GPFIFO/timeline memory setup: cmdq page, signal page, kernargs scratch.
    tprint("GPFIFO/timeline memory setup")
    with _tscope():
      self.cmdq_page = self.iface.alloc(0x200000, cpu_access=True)
      self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=int(self.cmdq_page.va_addr), wrap=True)
      self.cmdq = self.cmdq_page.cpu_view().view(fmt='I')
      # Skip the destructive 0-init write: racing with the GPU's first release can lose it;
      # first next_timeline() release is value 1, which is what we wait for.
      self._signal_page = self.iface.alloc(0x1000, cpu_access=True, uncached=True)
      self._signal_page_view = self._signal_page.cpu_view().view(fmt='Q')
      self.timeline_signal = NVSignal(owner=self)
      self.timeline_signal.value_addr = self._signal_page.va_addr
      self.timeline_value = 1  # first next_timeline() returns 1
      # Kernargs scratch buffer (sysmem so manual_launch can read/write it directly via mmap)
      self.kernargs_buf = self.iface.alloc(0x400000, cpu_access=True, uncached=True)
      self.kernargs_offset_allocator = BumpAllocator(size=self.kernargs_buf.size, wrap=True)

    # Setup shader shared/local memory windows (mirrors tinygrad NVDevice._setup_gpfifos).
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
    self.allocator = NVAllocator(self)
    self._setup_gpfifos()

  # =========================================================================
  # S7.2/S7.10: Allocate a new GPU FIFO channel (GPFIFO ring + UMD + token).
  # =========================================================================
  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False):
    tprint(f"new gpu fifo: off={offset:#x} entries={entries:#x} kind={'compute' if compute else 'video' if video else 'copy'}")
    with _tscope():
      notifier = self.iface.alloc(48 << 20, uncached=True)
      params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
        gpFifoOffset=gpfifo_area.va_addr + offset, gpFifoEntries=entries, hContextShare=ctxshare,
        hObjectError=notifier.meta.hMemory, hObjectBuffer=self.virtmem if video else gpfifo_area.meta.hMemory,
        hUserdMemory=(ctypes.c_uint32 * 8)(gpfifo_area.meta.hMemory),
        userdOffset=(ctypes.c_uint64 * 8)(entries * 8 + offset), engineType=19 if video else 0)

      gpfifo = self.iface.rm_alloc(channel_group, self.iface.gpfifo_class, params)
      if compute:
        self.debug_compute_obj = self.iface.rm_alloc(gpfifo, self.iface.compute_class)
        self.debug_channel = gpfifo
      elif not video:
        self.iface.rm_alloc(gpfifo, self.iface.dma_class)
      if channel_group == self.nvdevice:
        self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_BIND, nv_gpu.NVA06F_CTRL_BIND_PARAMS(engineType=params.engineType))
        self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
      ws_token_params = self.iface.rm_control(gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
        nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1))
      if self.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
        ws_token_params.workSubmitToken |= (1 << 30)
      gpput_off = offset + entries * 8 + getattr(nv_gpu.AmpereAControlGPFifo, 'GPPut').offset
      fifo = GPFifo(ring=gpfifo_area.cpu_view().view(offset, entries * 8, fmt='Q'),
                    gpput=gpfifo_area.cpu_view().view(gpput_off, 4, fmt='I'),
                    entries_count=entries, token=ws_token_params.workSubmitToken)
      return fifo, gpfifo

  def _query_gpu_info(self, *reqs):
    nvrs = [getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_' + r.upper(),
                    getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_LITTER_' + r.upper(), None)) for r in reqs]
    x = self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO,
                                nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS())
    return [x.engineInfo[0].infoList[nvr].data for nvr in nvrs]

  # =========================================================================
  # S7.16: Setup GPFIFOs — initial BIND + memory window submits.
  # =========================================================================
  def _setup_gpfifos(self):
    """Submit SET_OBJECT + shader memory windows + timeline signals (mirrors tinygrad NVDevice._setup_gpfifos)."""
    tprint("setup gpfifos: BIND + shader mem windows + timeline")
    with _tscope():
      self.slm_per_thread, self.shader_local_mem = 0, None
      self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
      NVComputeQueue().setup(compute_class=self.iface.compute_class,
                             local_mem_window=self.local_mem_window,
                             shared_mem_window=self.shared_mem_window) \
                      .signal(self.timeline_signal, self.next_timeline()).submit(self)
      NVCopyQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                   .setup(copy_class=self.iface.dma_class) \
                   .signal(self.timeline_signal, self.next_timeline()).submit(self)
      self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required:
      return
    self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    total = round_up(bytes_per_tpc * self.num_tpc_per_gpc * self.num_gpcs, 0x20000)
    if self.shader_local_mem is None or self.shader_local_mem.size < total:
      tprint("shader local-memory mapping")
      self.shader_local_mem = self.allocator.alloc(total, cpu_access=False)
    if self.shader_local_mem.size < total:
      print(f"WARN: shader_local_mem alloc too small got=0x{self.shader_local_mem.size:x} need=0x{total:x}", flush=True)
      self.slm_per_thread = old_slm_per_thread
      return
    NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                     .setup(local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc) \
                     .signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()
  def is_nvd(self): return isinstance(self.iface, PCIIface)
  def runtime(self, name, lib):
    return NVProgram(self, name, lib)

  def next_timeline(self):
    self.timeline_value += 1
    return self.timeline_value - 1

  def synchronize(self):
    # next_timeline() returns the value just stored in timeline_value (post-increment).
    # The GPU writes that value into the semaphore, so the latest release is
    # timeline_value - 1. Wait for the previous-release to land.
    target = self.timeline_value - 1
    start = time.perf_counter()
    while self.timeline_signal.value < target:
      elapsed_ms = (time.perf_counter() - start) * 1000
      if elapsed_ms > 5000:
        # Diagnostic: ring/gpput/signal snapshot when we give up
        gpput = self.compute_gpfifo.gpput[0]
        try:
          last_entry = int(self.compute_gpfifo.ring[(self.compute_gpfifo.put_value - 1) % self.compute_gpfifo.entries_count])
          last_va = ((last_entry & ((1 << 40) - 1)) >> 2) << 2
          last_pkts = (last_entry >> 42) & ((1 << 20) - 1)
        except Exception:
          last_va, last_pkts = -1, -1
        print(f"WARN: synchronize timeout target={target} got={self.timeline_signal.value} "
              f"gpput={gpput} put_value={self.compute_gpfifo.put_value} "
              f"last_ring_va=0x{last_va:x} last_pkts={last_pkts} sig_va=0x{self.timeline_signal.value_addr:x}",
              flush=True)
        return False
      if self.iface.is_local():
        for _ in self.iface.dev_impl.gsp.stat_q.read_resp(): pass
      else:
        self.iface.sleep(10)
      time.sleep(0.001)
    return True


# ============================================================================
# Demo driver: standalone NV add kernel (live path)
# Run with: python3 examples/add.py
# Optional:  python3 examples/add.py --middle-selftest (offline gate)
# ============================================================================

METHOD_NAMES = {
  0x005c: "NVC56F_SEM_ADDR_LO",
  0x02b4: "NVC6C0_SEND_PCAS_A",
  0x02c0: "NVC6C0_SEND_SIGNALING_PCAS2_B",
  0x1698: "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
  0x0020: "NVC56F_NON_STALL_INTERRUPT",
}
# Pull the shader-memory-window method IDs straight from the autogen header so
# pushbuffer decode shows real names instead of UNKNOWN_0x...
# AMPERE_COMPUTE_B (0xC7C0) uses the NVC7C0_* names; fall back to NVC6C0_*.
def _mid(name: str):
  for pfx in ("NVC7C0_", "NVC6C0_"):
    if hasattr(nv_gpu, pfx + name): return getattr(nv_gpu, pfx + name)
  return None

for _mn in ("SET_OBJECT", "SET_SHADER_LOCAL_MEMORY_WINDOW_A", "SET_SHADER_SHARED_MEMORY_WINDOW_A",
            "SET_SHADER_LOCAL_MEMORY_WINDOW_B", "SET_SHADER_SHARED_MEMORY_WINDOW_B",
            "SET_SHADER_LOCAL_MEMORY_A", "SET_SHADER_LOCAL_MEMORY_B",
            "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A", "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B",
            "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C"):
  if (_id := _mid(_mn)) is not None: METHOD_NAMES.setdefault(_id, "NVC6C0_" + _mn if hasattr(nv_gpu, "NVC6C0_" + _mn) else "NVC7C0_" + _mn)

# Paired 64-bit / triple methods for shader local-memory programming.
M_SLM = tuple(_mid(f"SET_SHADER_LOCAL_MEMORY_{s}") for s in "AB")
M_SLNT = tuple(_mid(f"SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_{s}") for s in "ABC")

_SUBMIT_COUNT = 0   # lifetime GPFIFO submission counter (semantic SUBMIT #N labels)

# Reverse map for RM control commands: value -> short name (e.g. 0x2080200a -> PERF_BOOST).
_CTRL_CMD_NAMES = {v: k.split("_CTRL_CMD_")[-1] for k, v in nv_gpu.__dict__.items()
                           if isinstance(v, int) and "_CTRL_CMD_" in k and v}
bind(NVReg=NVReg, RemoteCmd=RemoteCmd, method_names=METHOD_NAMES)

_SEM_OPS = None
def _sem_op_name(v: int) -> str:
  global _SEM_OPS
  if _SEM_OPS is None:
    _SEM_OPS = {vv: k.removeprefix("NVC56F_SEM_EXECUTE_OPERATION_").lower()
                for k, vv in nv_gpu.__dict__.items()
                if k.startswith("NVC56F_SEM_EXECUTE_OPERATION_") and isinstance(vv, int)}
  return _SEM_OPS.get(v, f"{v:#x}")

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
    oname = _sem_op_name(args[4] & 0xf)
    pl64 = "payload64" if args[4] & (1 << 24) else "payload32"
    return [f"sem_addr=0x{sem_addr:x}", f"payload={payload}", f"execute=0x{args[4]:08x} {{{oname},{pl64}}}"]
  if method == 0x1698 and len(args) == 1:
    on = "|".join(n for bit, n in [(0x1, "icache"), (0x10, "global_data"), (0x1000, "constant")] if args[0] & bit)
    return [f"invalidate_flags=0x{args[0]:08x} {{{on or 'none'}}}"]
  if method == 0x02b4 and len(args) == 1:
    return [f"qmd_addr=0x{args[0] << 8:x}", f"qmd_addr_shifted8=0x{args[0]:x}"]
  if method == 0x02c0 and len(args) == 1:
    return [f"pcas2_action=0x{args[0]:x}"]
  return [f"arg{i}=0x{arg:08x}" for i, arg in enumerate(args)]

class CubinHelper:
  """Synthetic cubin builder for the add kernel demo (no external CUDA toolchain)."""
  class Reg:
    RZ = 255
    R0 = 0; R1 = 1; R2 = 2; R3 = 3; R4 = 4; R5 = 5; R6 = 6; R7 = 7
    R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15

  class UReg:
    URZ = 63
    UR4 = 4  # only UR4 is used in our cubin

  class Op:
    LDC     = 0x7a02
    LDCU64  = 0x7ab9
    FADD    = 0x7221
    FMUL    = 0x7220
    LDG     = 0x7981
    STG     = 0x7986
    EXIT    = 0x794d
    BRA     = 0x7947
    NOP     = 0x7918

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

def build_cubin(operation="add"):
  if operation == "add":
    arithmetic = [
      ((ch.Reg.R11 << 24) | (ch.Reg.R11 << 16) | ch.Op.FADD, 0x00000007, 0x00000000, 0x004fe200),  # FADD R11, R11, R7
      ((ch.Reg.R10 << 24) | (ch.Reg.R10 << 16) | ch.Op.FADD, 0x00000006, 0x00000000, 0x000fe200),  # FADD R10, R10, R6
      ((ch.Reg.R9  << 24) | (ch.Reg.R9  << 16) | ch.Op.FADD, 0x00000005, 0x00000000, 0x000fe200),  # FADD R9, R9, R5
      ((ch.Reg.R8  << 24) | (ch.Reg.R8  << 16) | ch.Op.FADD, 0x00000004, 0x00000000, 0x000fe200),  # FADD R8, R8, R4
    ]
  elif operation == "mul":
    arithmetic = [
      # SM86 FMUL needs word2 bit 22. Without it nvdisasm reports FMUL.INVALID0
      # and the kernel can signal completion without producing valid output.
      ((ch.Reg.R11 << 24) | (ch.Reg.R11 << 16) | ch.Op.FMUL, 0x00000007, 0x00400000, 0x004fe200),  # FMUL R11, R11, R7
      ((ch.Reg.R10 << 24) | (ch.Reg.R10 << 16) | ch.Op.FMUL, 0x00000006, 0x00400000, 0x000fe200),  # FMUL R10, R10, R6
      ((ch.Reg.R9  << 24) | (ch.Reg.R9  << 16) | ch.Op.FMUL, 0x00000005, 0x00400000, 0x000fe200),  # FMUL R9, R9, R5
      ((ch.Reg.R8  << 24) | (ch.Reg.R8  << 16) | ch.Op.FMUL, 0x00000004, 0x00400000, 0x000fe200),  # FMUL R8, R8, R4
    ]
  else: raise ValueError(f"unsupported operation: {operation!r}")

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

  ] + arithmetic + [
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

# =========================================================================
# S8–S9: Build kernargs + QMD, push to GPFIFO, wait for semaphore.
# =========================================================================
def manual_launch(dev, program, out, a, b):
  stage_set(8, "kernargs/QMD build")
  kernargs = dev.kernargs_buf.offset(dev.kernargs_offset_allocator.alloc(program.kernargs_alloc_size, 8), program.kernargs_alloc_size)
  cbuf_words = program.cbuf_0 or []
  kernargs.cpu_view().view(size=len(cbuf_words) * 4, fmt='I')[:] = array.array('I', cbuf_words)
  kernargs.cpu_view().view(offset=len(cbuf_words) * 4, size=3 * 8, fmt='Q')[:] = array.array('Q', [out.va_addr, a.va_addr, b.va_addr])
  qmd_buf = kernargs.offset(round_up(program.constbufs[0][1], 1 << 8))
  # Finish the QMD in host memory and upload it once. Repeated read/modify/write
  # cycles against BAR1 can exhaust Chestnut's F0 request path before launch.
  if isinstance(dev.iface, USBIface): qmd_view = memoryview(bytearray(program.qmd.mv))
  else:
    qmd_buf.cpu_view().view(size=program.qmd.mv.nbytes, fmt='B')[:] = program.qmd.mv
    qmd_view = qmd_buf.cpu_view()
  qmd = type(program.qmd)(dev=dev, view=qmd_view)
  qmd.write(cta_raster_width=1, cta_raster_height=1, cta_raster_depth=1,
            cta_thread_dimension0=1, cta_thread_dimension1=1, cta_thread_dimension2=1)
  qmd.set_constant_buf_addr(0, kernargs.va_addr)
  wait_value = dev.timeline_value - 1
  done_value = dev.next_timeline()
  signal_addr = dev.timeline_signal.value_addr
  qmd.write(release0_enable=1, release0_address_lower=signal_addr & 0xffffffff, release0_address_upper=(signal_addr >> 32) & 0xff,
            release0_payload_lower=done_value & 0xffffffff, release0_payload_upper=done_value >> 32)
  if isinstance(dev.iface, USBIface):
    qmd_bytes = bytes(qmd.mv)
    qmd_buf.cpu_view().view(size=len(qmd_bytes), fmt='B')[:] = qmd_bytes
  else: qmd_bytes = bytes(qmd_buf.cpu_view().view(size=program.qmd.mv.nbytes, fmt='B'))
  stage_done(f"QMD @{qmd_buf.va_addr:#x}")
  tprint(f"QMD @va={qmd_buf.va_addr:#x} size={len(qmd_bytes):#x} sha256:{hashlib.sha256(qmd_bytes).hexdigest()[:16]}")
  words = build_launch_words(signal_addr, wait_value, done_value, qmd_buf.va_addr)[:12]
  pcas = [a[0] for _, _, _, m, _, a in decode_words(words) if m == 0x2b4]
  assert pcas and (pcas[0] << 8) == qmd_buf.va_addr, f"SEND_PCAS_A target {pcas[0] << 8:#x} != QMD VA {qmd_buf.va_addr:#x}"
  tprint(f"DECODE pending kernel pushbuffer: {len(words)} dwords")
  for index, typ, subc, method, name, args in decode_words(words):
    print(f"  method[{index}] {name}: typ={typ} subc={subc} mthd=0x{method:x} args=[{', '.join(describe_args(method, args))}]")
  stage_set(9, "pushbuffer->GPFIFO->doorbell")
  submit_gpfifo(dev, words)
  wait_signal(dev.timeline_signal, done_value)
  tprint(f"EXECUTION complete: {_SUBMIT_COUNT} GPFIFO submissions, 1 kernel launch, timeline {done_value - 1} -> {done_value}")
  stage_done(f"submit #{_SUBMIT_COUNT}, kernel launch, timeline {done_value - 1}->{done_value}")

MIDDLE_CUBIN_SHA256 = "54f9606fe6b03d6cc98186358c68a74cebe8275137c1e98723967f9a14c67324"
MUL_CUBIN_SHA256 = "4895dc5534c8b7714b33164110657cba55936addd0d3ba74eb99a0856323f733"
MIDDLE_CUBIN_BYTES = 2856
MIDDLE_LAUNCH_WORDS = 20

def middle_selftest():
  """Tier 1 offline gate: add/mul cubins + launch words + helper sanity."""
  cubins = {operation: build_cubin(operation) for operation in ("add", "mul")}
  expected_shas = {"add": MIDDLE_CUBIN_SHA256, "mul": MUL_CUBIN_SHA256}
  shas = {operation: hashlib.sha256(cubin).hexdigest() for operation, cubin in cubins.items()}
  for operation, cubin in cubins.items():
    assert len(cubin) == MIDDLE_CUBIN_BYTES, f"{operation} cubin size {len(cubin)} != {MIDDLE_CUBIN_BYTES}"
    assert shas[operation] == expected_shas[operation], f"{operation} cubin sha {shas[operation]} != {expected_shas[operation]}"
    text = cubin[ch.TEXT_OFF:ch.TEXT_OFF + 512]
    expected_opcode, other_opcode = (ch.Op.FADD, ch.Op.FMUL) if operation == "add" else (ch.Op.FMUL, ch.Op.FADD)
    expected_count = sum((struct.unpack_from("<I", text, offset)[0] & 0x7fff) == expected_opcode for offset in range(0, 512, 16))
    other_count = sum((struct.unpack_from("<I", text, offset)[0] & 0x7fff) == other_opcode for offset in range(0, 512, 16))
    mode_count = sum(bool(struct.unpack_from("<I", text, offset + 8)[0] & 0x00400000) for offset in range(0, 512, 16))
    controls = [struct.unpack_from("<I", text, offset + 12)[0] for offset in range(8 * 16, 12 * 16, 16)]
    assert expected_count == 4, f"expected 4 {operation} opcodes, got {expected_count}"
    assert other_count == 0, f"expected no opposite opcodes in {operation} cubin, got {other_count}"
    assert mode_count == (4 if operation == "mul" else 0), f"unexpected {operation} FMUL mode-bit count: {mode_count}"
    assert controls == [0x004fe200, 0x000fe200, 0x000fe200, 0x000fe200], f"unexpected {operation} control words: {controls}"
  words = build_launch_words(0xdeadbeef00001000, 3, 7, 0x2000)
  assert len(words) == MIDDLE_LAUNCH_WORDS, f"launch words {len(words)} != {MIDDLE_LAUNCH_WORDS}"
  decoded = list(decode_words(words))
  assert len(decoded) == 6, f"decode_words count {len(decoded)} != 6"
  sem_methods = [m for _, _, _, m, _, _ in decoded if m == 0x005c]
  assert len(sem_methods) == 2, "expected two semaphore methods"
  # helpers sanity
  assert lo32(0x123456789abcdef0) == 0x9abcdef0
  assert hi32(0x123456789abcdef0) == 0x12345678
  assert round_up(17, 16) == 32
  assert ceildiv(17, 16) == 2
  assert wait_cond(lambda: 1, value=1, timeout_ms=100)
  # mmio roundtrip (use array directly; MMIOInterface is the autogen, not the test slim wrapper)
  arr = array.array('I', [0, 1, 2, 3])
  arr[1] = 0x42
  assert arr[1] == 0x42
  assert arr[2] == 2
  # grbuf
  desc = GRBufDesc(size=4096, virt=True, phys=False)
  assert desc.size == 4096 and desc.virt and not desc.phys
  # nvrpcqueue checksum
  data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
  if (pad_len := (-len(data)) % 8): data += b"\x00" * pad_len
  cs = 0
  for off in range(0, len(data), 8): cs ^= struct.unpack_from("Q", data, off)[0]
  cs = hi32(cs) ^ lo32(cs)
  assert isinstance(cs, int) and 0 <= cs <= 0xffffffff
  validate_result("add", [11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0])
  try: validate_result("mul", [0.0, 0.0, 0.0, 0.0], [10.0, 40.0, 90.0, 160.0])
  except RuntimeError: pass
  else: raise AssertionError("result validation accepted incorrect multiplication output")
  print(f"middle_selftest=ok add_sha={shas['add']} mul_sha={shas['mul']} launch_words={len(words)} rpc_checksum=0x{cs:x}")

def operation_from_argv(argv):
  if not argv: return "add"
  if argv in (["-h"], ["--help"]):
    print("usage: python3 examples/add.py [add|mul]\n       python3 examples/add.py --middle-selftest")
    return None
  if len(argv) == 1 and argv[0] in ("add", "mul"): return argv[0]
  raise SystemExit("usage: python3 examples/add.py [add|mul]")

def validate_result(operation, result, expected_result):
  if result != expected_result:
    raise RuntimeError(f"{operation} result mismatch: expected {expected_result}, got {result}")

def main():
  if "--middle-selftest" in sys.argv:
    middle_selftest()
    return
  operation = operation_from_argv(sys.argv[1:])
  if operation is None: return
  # live path: run the selected arithmetic kernel
  t0 = time.perf_counter()
  def _ts(label):
    if os.environ.get("NV_ADD_TRACE_STAGES") == "1":
      print(f"  stage t={time.perf_counter()-t0:6.3f}s  {label}", flush=True)
  a = (1.0, 2.0, 3.0, 4.0)
  b = (10.0, 20.0, 30.0, 40.0)
  cubin = build_cubin(operation)
  expected_result = [x + y for x, y in zip(a, b)] if operation == "add" else [x * y for x, y in zip(a, b)]
  _ts(f"cubin built ({operation})")
  print(f"operation={operation} cubin_bytes={len(cubin)} expected_result={expected_result}")
  dev = NVDevice("NV")
  result_bytes = bytearray(16)
  try:
    _ts("device ready (boot+GSP+golden+user-channel+gpfifo)")
    print(f"device={dev.device} iface={type(dev.iface).__name__}", flush=True)
    tprint("INPUT_A alloc/map/copyin")
    a_buf = dev.allocator.alloc(16)
    dev.allocator._copyin(a_buf, memoryview(struct.pack("4f", *a)))
    tprint("INPUT_B alloc/map/copyin")
    b_buf = dev.allocator.alloc(16)
    dev.allocator._copyin(b_buf, memoryview(struct.pack("4f", *b)))
    tprint("OUTPUT alloc/map/init")
    out_buf = dev.allocator.alloc(16)
    dev.allocator._copyin(out_buf, memoryview(bytes(16)))
    program = dev.runtime("E_4", cubin)
    stage_done(f"{_SUBMIT_COUNT} setup submissions, timeline 0->{dev.timeline_value - 1}")
    _ts("program built (cubin uploaded to VRAM, NVProgram ready)")
    manual_launch(dev, program, out_buf, a_buf, b_buf)
    _ts("manual_launch done (kernel executed on eGPU, result on device)")
    dev.allocator._copyout(memoryview(result_bytes), out_buf)
    _ts("copyout done (D2H)")
  finally:
    # Gracefully unload GSP on PCI and perform the additional SEC2 teardown on USB.
    dev.iface.device_fini()
  print(f"raw_result={bytes(result_bytes).hex(' ')}")
  result = list(struct.unpack("4f", result_bytes))
  stage_set(10, f"validate {operation} output")
  _ts(f"final result decoded: {result}")
  print(f"result={result}")
  validate_result(operation, result, expected_result)
  stage_done(f"{operation} result matched {expected_result}")

if __name__ == "__main__":
  main()
