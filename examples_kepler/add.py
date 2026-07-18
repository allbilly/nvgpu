#!/usr/bin/env python3
"""GK104 macOS TinyGPU entrypoint using the shared PCIe implementation.

The RM/GMMU/FIFO/launch, firmware, cubin, and debug code is imported directly
from examples_kepler_pcie.add. This file contains only TinyGPU transport and
macOS safety/probe policy.

The TinyGPU.app service must already be running and expose /tmp/tinygpu.sock.
The service owns the DriverKit PCIe entitlement and performs BAR transactions.
This client does not load a Nouveau kernel driver or rely on Linux sysfs.
It injects the socket device into the shared launcher at startup.
Offline self-tests never connect to the socket and are safe on any platform.
Live execution remains explicitly acknowledged because a bad MMIO sequence can
hang an eGPU link; set KEPLER_LIVE_ACK=completion-abort-risk for that path.
KEPLER_RPC_TRACE is required for live macOS runs so the transaction stream is
auditable after a crash.  The shared implementation still assembles sm_30 PTX
locally when CUDA 10.2 ptxas is installed, with the checked-in cubin as fallback.
No package installation or tinygrad import is needed beyond the shared checkout.
"""
from __future__ import annotations
import os, sys, ctypes, mmap, struct, socket, subprocess, contextlib, functools, itertools, enum, atexit, select, dataclasses, collections, pathlib, threading, time, hashlib
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _path in (REPO_ROOT, os.path.join(REPO_ROOT, "ref")):
  if _path not in sys.path: sys.path.insert(0, _path)
from examples_kepler_pcie import add as shared
RemotePCIDevice = shared.RemotePCIDevice
MMIOInterface = shared.MMIOInterface
FileIOInterface = shared.FileIOInterface
ceildiv = shared.ceildiv
PAGESIZE = 0x1000
class RemoteCmd(enum.IntEnum):
  # Wire command IDs spoken by TinyGPU.app's DriverKit extension, as decoded by
  # TheTom/pascal-egpu's TinyGPUClient (which successfully reads PMC_BOOT_0 over
  # this protocol on a real eGPU).  Request: struct.pack('<BIIQQQ', cmd, dev_id,
  # bar, *args3); response: struct.unpack('<QQB', ...) = (value1, value2, status)
  # followed by `readout` payload bytes when present.
  MAP_BAR       = 1
  MAP_SYSMEM_FD = 2
  CFG_READ      = 3
  CFG_WRITE     = 4
  RESET         = 5
  SYSMEM_READ   = 9
  SYSMEM_WRITE  = 10
  MMIO_READ     = 6
  MMIO_WRITE    = 7
  MAP_SYSMEM    = 8

class RemotePCIDevice(shared.RemotePCIDevice):
  """Abstract transport for a remote PCIe GPU (TinyGPU socket, vfio, ...)."""
  def __init__(self, name, transport):
    self.name, self.transport = name, transport
  def bar_info(self, bar):
    raise NotImplementedError
  def map_bar(self, bar, fmt='B', off=0, size=None):
    raise NotImplementedError
  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    raise NotImplementedError
  def mmio_read(self, bar, offset, size):
    raise NotImplementedError
  def mmio_write(self, bar, offset, data):
    raise NotImplementedError

class RemoteMMIOInterface(MMIOInterface):
  """MMIO register window that routes reads/writes through the transport
  (no local memoryview — every access is a TinyGPU RPC)."""
  def __init__(self, pci_dev, bar, fmt='B'):
    self.pci_dev, self.bar, self.fmt = pci_dev, bar, fmt
    self.addr, self.nbytes = pci_dev.bar_info(bar)
  def __len__(self): return self.nbytes // struct.calcsize(self.fmt)
  def __getitem__(self, k):
    if isinstance(k, slice):
      start = k.start or 0
      n = (k.stop or self.nbytes) - start
      return self.pci_dev.mmio_read(self.bar, start, n)
    sz = struct.calcsize(self.fmt)
    return struct.unpack_from(self.fmt, self.pci_dev.mmio_read(self.bar, k * sz, sz))[0]
  def __setitem__(self, k, v):
    if self.fmt != 'B' and isinstance(v, (list, tuple)):
      v = b"".join(struct.pack(self.fmt, x) for x in v)
    if isinstance(k, slice):
      start = k.start or 0
      self.pci_dev.mmio_write(self.bar, start, bytes(v) if isinstance(v, (bytes, bytearray)) else v)
    else:
      sz = struct.calcsize(self.fmt)
      self.pci_dev.mmio_write(self.bar, k * sz, struct.pack(self.fmt, v))
  def view(self, offset=0, size=None, fmt=None):
    return RemoteMMIOInterface(self.pci_dev, self.bar, fmt=fmt or self.fmt)
  def read32(self, off): return self.pci_dev.mmio_read32(self.bar, off)
  def write32(self, off, val): return self.pci_dev.mmio_write32(self.bar, off, val)

def _temp_sock():
  # Match the proven examples/add.py client: one stable server/socket is reused
  # across invocations.  Spawning a new DriverKit server on every Kepler run
  # and terminating it during close caused avoidable service detach/rebind
  # churn on the Apple PCIe path.
  # Keep this literal: macOS tempfile.gettempdir() is often a per-user
  # /var/folders path, while the shared TinyGPU server contract is /tmp.
  return "/tmp/tinygpu.sock"

class APLRemotePCIDevice(RemotePCIDevice):
  """macOS: TinyGPU.app signed DriverKit extension exposes raw PCIe BAR access
  for an eGPU over a local Unix socket.  This is a faithful port of the proven
  client in examples/add.py (which drives this same hardware), adapted for
  Kepler bring-up.  The shared server must already own the stable socket; this
  crash-isolation client never starts or terminates the DriverKit service."""
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"
  APP_COMMIT = "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"

  def __init__(self, name="NV", transport=None, dev_id=0, sock_path=None, timeout_ms=2000):
    super().__init__(name, transport or "usb4")
    self.dev_id = dev_id
    self.sock_path = sock_path or os.environ.get("APL_REMOTE_SOCK", _temp_sock())
    self._sock = None
    self._server_proc = None
    self._pci_config_available = False
    self._fini_done = False
    self._sock_lock = threading.Lock()
    self._rpc_seq = 0
    self._rpc_phase = "connect"
    self._rpc_owner_thread = threading.get_ident()
    self._single_thread_rpc = os.environ.get("KEPLER_SINGLE_THREAD_RPC", "1") != "0"
    self._endpoint_frozen = False
    self._endpoint_freeze_reason = None
    self._final_rpc_budget = None
    self._trace_fd = None
    trace_path = os.environ.get("KEPLER_RPC_TRACE")
    if trace_path:
      flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
      if hasattr(os, "O_CLOEXEC"): flags |= os.O_CLOEXEC
      self._trace_fd = os.open(trace_path, flags, 0o600)
    try:
      self.set_phase("connect")
      self._connect(timeout_ms)
    except Exception:
      # __init__ failures never reach NVDevice.close().  Roll back this client
      # socket, but leave the shared signed server lifecycle alone.
      self.fini(reset_endpoint=False)
      raise

  def _connect(self, timeout_ms):
    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self._sock.settimeout(timeout_ms / 1000.0)
    connected = False
    for i in range(100):
      try:
        self._sock.connect(self.sock_path); connected = True; break
      except (ConnectionRefusedError, FileNotFoundError):
        time.sleep(0.05)
    if not connected:
      raise RuntimeError(
          f"shared TinyGPU server is not reachable at {self.sock_path}; "
          "start exactly one intended server before the cold live run")

  def _trace_record(self, record):
    fd = getattr(self, "_trace_fd", None)
    if fd is not None:
      data = (record.rstrip("\n") + "\n").encode("utf-8", "backslashreplace")
      while data:
        written = os.write(fd, data)
        if written <= 0:
          raise OSError("RPC flight recorder made no write progress")
        data = data[written:]

  @staticmethod
  def _trace_atom(value):
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace(" ", "\\x20")

  def set_phase(self, phase):
    self._rpc_phase = str(phase)
    self._trace_record(
        f"PHASE monotonic_ns={time.monotonic_ns()} thread={threading.get_ident()} "
        f"phase={self._trace_atom(self._rpc_phase)}")

  def arm_final_output_read(self, bar, offset, size):
    """Allow exactly one final BAR read after the completion semaphore."""
    if self._endpoint_frozen:
      raise RuntimeError(f"cannot arm output read after freeze: {self._endpoint_freeze_reason}")
    self._final_rpc_budget = (int(RemoteCmd.MMIO_READ), int(bar), int(offset), int(size))
    self.set_phase("output-read")

  def freeze(self, reason):
    """Reject every future protocol frame while still allowing local close."""
    if self._endpoint_frozen:
      return
    self._endpoint_frozen = True
    self._endpoint_freeze_reason = str(reason)
    self._trace_record(
        f"FREEZE monotonic_ns={time.monotonic_ns()} thread={threading.get_ident()} "
        f"phase={self._trace_atom(self._rpc_phase)} "
        f"reason={self._trace_atom(self._endpoint_freeze_reason)}")

  def _recvall(self, n):
    buf = bytearray(n)
    got = 0
    while got < n:
      cnt = self._sock.recv_into(memoryview(buf)[got:])
      if cnt == 0: raise ConnectionError("TinyGPU socket closed")
      got += cnt
    return bytes(buf)

  def _rpc(self, cmd, bar, *args, readout=0, payload=b'', has_fd=False):
    if getattr(self, "_endpoint_frozen", False):
      raise RuntimeError(
          f"endpoint RPC after freeze: cmd={cmd} bar={bar} args={args}")
    owner = getattr(self, "_rpc_owner_thread", threading.get_ident())
    if getattr(self, "_single_thread_rpc", False):
      assert threading.get_ident() == owner, (
          f"endpoint RPC from non-owner thread: owner={owner} "
          f"current={threading.get_ident()} cmd={cmd} bar={bar} args={args}")
    cmd_id = int(cmd)
    padded_args = (tuple(args) + (0, 0, 0))[:3]
    offset = int(padded_args[0])
    size = int(padded_args[1])
    budget = getattr(self, "_final_rpc_budget", None)
    if budget == "consumed":
      raise RuntimeError(
          f"endpoint RPC after final output read: cmd={cmd} bar={bar} args={args}")
    if budget is not None:
      attempted = (cmd_id, int(bar), offset, size)
      if attempted != budget:
        raise RuntimeError(
            f"RPC outside final output budget: allowed={budget} attempted={attempted}")
      # Consume before sendall: a failed final read must never be retried.
      self._final_rpc_budget = "consumed"
    with self._sock_lock:
      self._rpc_seq = getattr(self, "_rpc_seq", 0) + 1
      seq = self._rpc_seq
      start_ns = time.monotonic_ns()
      phase = getattr(self, "_rpc_phase", "unknown")
      cmd_name = cmd.name if isinstance(cmd, RemoteCmd) else RemoteCmd(cmd_id).name
      payload_hash = hashlib.sha256(payload).hexdigest() if payload else "-"
      config_offset = offset if cmd_id in (int(RemoteCmd.CFG_READ), int(RemoteCmd.CFG_WRITE)) else "-"
      common = (
          f"seq={seq} monotonic_ns={start_ns} phase={self._trace_atom(phase)} "
          f"thread={threading.get_ident()} cmd={cmd_name} cmd_id={cmd_id} "
          f"bar={bar} offset={offset} size={size} config_offset={config_offset} "
          f"args={self._trace_atom(repr(tuple(args)))} "
          f"readout={readout} payload_len={len(payload)} payload_sha256={payload_hash}")
      self._trace_record(f"BEGIN {common}")
      try:
        self._sock.sendall(struct.pack("<BIIQQQ", cmd_id, self.dev_id, bar,
                                       *padded_args) + payload)
        if payload:  # writes: server sends no response (matches examples/add.py)
          self._trace_record(
              f"END {common} status=ok bytes={len(payload)} "
              f"duration_us={(time.monotonic_ns() - start_ns) // 1000}")
          return None
        if has_fd:
          msg, anc, _, _ = self._sock.recvmsg(17, socket.CMSG_LEN(4))
          fd = struct.unpack('<i', anc[0][2][:4])[0]
        else:
          msg = self._recvall(17); fd = None
        status, value1, value2 = struct.unpack("<BQQ", msg)
        if status != 0:
          err = self._recvall(value1).decode('utf-8') if value1 > 0 else 'unknown error'
          raise RuntimeError(f"TinyGPU RPC cmd={cmd_id} bar={bar} args={args} failed: {err}")
        data = self._recvall(readout) if readout else b""
        self._trace_record(
            f"END {common} status=ok bytes={len(data)} "
            f"duration_us={(time.monotonic_ns() - start_ns) // 1000}")
        return value1, value2, data, fd
      except BaseException as exc:
        self._trace_record(
            f"END {common} status=exception "
            f"exception={self._trace_atom(type(exc).__name__ + ':' + str(exc))} "
            f"duration_us={(time.monotonic_ns() - start_ns) // 1000}")
        raise

  def bar_info(self, bar):
    v1, v2, _, _ = self._rpc(RemoteCmd.MAP_BAR, bar)
    return (v1, v2)

  def mmio_read(self, bar, offset, size):
    _, _, data, _ = self._rpc(RemoteCmd.MMIO_READ, bar, offset, size, readout=size)
    return data

  def mmio_read32(self, bar, offset):
    return struct.unpack_from("<I", self.mmio_read(bar, offset, 4))[0]

  def mmio_write(self, bar, offset, data):
    self._rpc(RemoteCmd.MMIO_WRITE, bar, offset, len(data), payload=bytes(data))

  def mmio_write32(self, bar, offset, value):
    self.mmio_write(bar, offset, struct.pack("<I", value))

  def map_bar(self, bar, fmt='B', off=0, size=None):
    return RemoteMMIOInterface(self, bar, fmt=fmt)

  def read_config(self, offset, size):
    value = self._rpc(RemoteCmd.CFG_READ, 0, offset, size)[0]
    self._pci_config_available = True
    return value

  def write_config(self, offset, value, size):
    self._rpc(RemoteCmd.CFG_WRITE, 0, offset, size, value)

  def write_config_flush(self, offset, value, size):
    self.write_config(offset, value, size)
    return self.read_config(offset, size)

  def reset(self):
    self._rpc(RemoteCmd.RESET, 0)

  def fini(self, reset_endpoint=False):
    if getattr(self, "_fini_done", False):
      return
    self._fini_done = True
    sock = getattr(self, "_sock", None)
    # Deliberately match the working RTX 3080 client in examples/add.py: close
    # only this client socket.  Do not issue Kepler-only CFG/RESET transactions
    # and do not terminate/relaunch the signed TinyGPU DriverKit server.  The
    # 11:35 panic caught the sole Python thread in recv_into after the FECS
    # thread had been joined, which isolates the rejected request to the extra
    # synchronous PCI shutdown path removed here.
    if sock:
      try:
        sock.close()
      except Exception:
        pass
    self._sock = None
    self._server_proc = None
    trace_fd = getattr(self, "_trace_fd", None)
    if trace_fd is not None:
      try:
        os.close(trace_fd)
      finally:
        self._trace_fd = None

  def __del__(self):
    # Fallback only; normal and partial-constructor paths call fini directly.
    try: self.fini()
    except Exception: pass

  def alloc_sysmem(self, size, vaddr=0, contiguous=False):
    """Allocate GPU-visible host memory.  Returns (memoryview, [bus_paddrs])
    via MAP_SYSMEM_FD + recvmsg fd + mmap — CPU-coherent, so copies go straight
    to the mmap (no SYSMEM_READ/WRITE RPCs).  Matches examples/add.py."""
    mapped_size, _, _, fd = self._rpc(RemoteCmd.MAP_SYSMEM_FD, 0, size, int(contiguous), has_fd=True)
    memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt='B')
    paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
    paddrs = [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]
    return memview, paddrs

  def sysmem_read(self, addr, size):
    _, _, data, _ = self._rpc(RemoteCmd.SYSMEM_READ, 0, addr, size, readout=size)
    return data

  def sysmem_write(self, addr, data):
    self._rpc(RemoteCmd.SYSMEM_WRITE, 0, addr, len(data), payload=bytes(data))

  @staticmethod
  def probe(sock_path=None, timeout_ms=500):
    """Return an APLRemotePCIDevice if TinyGPU is reachable, else None."""
    try:
      return APLRemotePCIDevice(sock_path=sock_path, timeout_ms=timeout_ms)
    except (OSError, RuntimeError):
      return None


class _MacPCIDeviceFactory:
  def __call__(self, dev_id=0, **kwargs): return APLRemotePCIDevice(dev_id=dev_id)

def _probe():
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe: TinyGPU.app socket is not reachable (is the eGPU connected?)")
    raise SystemExit(1)
  try:
    boot0, meta = shared._gk104_ensure_bar0_mmio(dev)
    print(f"probe: PCI_ID={meta['id32']:#010x} "
          f"COMMAND={meta['command_before']:#06x}->{meta['command_after']:#06x} "
          f"mse_was={meta['mse_before']} reset={meta['did_reset']}")
    print(f"probe: PMC_BOOT_0=0x{boot0:08x} (chip_id={(boot0 >> 20) & 0xfff})")
  finally:
    dev.fini()

def _probe_post_ownership():
  """Read the inherited Nouveau POST boundary and exit without GPU writes."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-post-ownership: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    boot0, _meta = shared._gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
    snap = shared._gk104_post_entry_probe(BAR0Dev())
    if snap["boot0"] != boot0:
      raise RuntimeError("PMC_BOOT_0 changed during POST ownership probe")
    print("probe-post-ownership: " +
          ("READY for posted Night41h" if snap["night41h_ready"] else
           "NOT posted/PRAMIN-visible; do not repeat cold BAR1 run"))
  finally:
    dev.fini()

def _probe_rom_shadow_ownership():
  """Read Nouveau's inherited PRAMIN VBIOS-source predicates; write nothing."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-rom-shadow-ownership: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    shared._gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
    snap = shared._gk104_rom_shadow_entry_probe(BAR0Dev())
    print("probe-rom-shadow-ownership: " +
          ("READY: inherited Nouveau RAMIN source is present" if
           snap["firmware_shadow_ready"] else
           "MISSING: cold firmware RAMIN source is not present"))
  finally:
    dev.fini()

def _probe_golden_preinit():
  """Compare all seven safe pre-init golden reads without GPU writes."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-golden-preinit: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    shared._gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
    snap = shared._gk104_golden_preinit_entry_probe(BAR0Dev())
    print(f"probe-golden-preinit: mismatches="
          f"{[hex(reg) for reg in snap['mismatch_regs']]}")
  finally:
    dev.fini()

def _probe_option_rom_vga_preamble():
  """A/B the proven x86-ROM VGA prefix against immediate PRAMIN state."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-option-rom-vga-preamble: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    shared._gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
      def write32(self, r, v): dev.mmio_write32(0, r, v)
      def read8(self, r): return dev.mmio_read(0, r, 1)[0]
      def write8(self, r, v): dev.mmio_write(0, r, bytes((v & 0xff,)))
    bar0 = BAR0Dev()
    before = shared._gk104_post_entry_probe(bar0)
    image = shared.nvbios_init.find_vbios_image(
        pathlib.Path(shared.DEFAULT_VBIOS).read_bytes())
    shared.nvbios_init.NvbiosInit(bar0, image).option_rom_vga_enable_prefix()
    after = shared._gk104_post_entry_probe(bar0)
    activated = bool(after["pramin_positive"] and
                     not before["pramin_positive"])
    print("probe-option-rom-vga-preamble: "
          f"before={before['pramin_word']:#010x} "
          f"after={after['pramin_word']:#010x} activated={activated}")
  finally:
    dev.fini()

def _probe_nouveau_init_io():
  """A/B the executed Palit INIT_IO special case before any other init."""
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-nouveau-init-io: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    shared._gk104_ensure_bar0_mmio(dev)
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
      def write32(self, r, v): dev.mmio_write32(0, r, v)
      def read8(self, r): return dev.mmio_read(0, r, 1)[0]
      def write8(self, r, v): dev.mmio_write(0, r, bytes((v & 0xff,)))
    bar0 = BAR0Dev()
    before = shared._gk104_post_entry_probe(bar0)
    image = shared.nvbios_init.find_vbios_image(
        pathlib.Path(shared.DEFAULT_VBIOS).read_bytes())
    init = shared.nvbios_init.NvbiosInit(bar0, image)
    if init.rd08(0x85bb) != 0x69:
      raise RuntimeError("Palit VBIOS INIT_IO opcode moved from 0x85bb")
    init.offset = 0x85bb
    init._op_io()
    if init.offset != 0x85c0:
      raise RuntimeError(f"INIT_IO ended at unexpected {init.offset:#x}")
    after = shared._gk104_post_entry_probe(bar0)
    activated = bool(after["pramin_positive"] and
                     not before["pramin_positive"])
    print("probe-nouveau-init-io: "
          f"before={before['pramin_word']:#010x} "
          f"after={after['pramin_word']:#010x} activated={activated}")
  finally:
    dev.fini()

def _probe_nouveau_base_lifecycle(*, bisect_post_scripts=False):
  """Bisect Nouveau's POST and base FB/RAM boundaries on one cold entry.

  ``bisect_post_scripts`` runs a *prefix* of BIT-I top-level scripts, then
  takes exactly one fixed-PA PRAMIN sample.  Night41t proved that retargeting
  ``0x1700`` between scripts leaves the same core NVINIT MMIO stream as
  Night41s but keeps fixed-PA ``0xfffe0000`` virgin; Night41s only sampled
  once at end-of-POST and saw data.  Mid-POST selector traffic is therefore
  forbidden.  Set ``KEPLER_POST_SCRIPT_PREFIX`` to the 1-based prefix length
  (1..N).  Optionally set ``KEPLER_NVINIT_STOP_OFFSET`` so the *last* script
  of the prefix stops before that top-level ROM offset (Night41x nested
  bisect of ``0x8fe8``).  One cold cycle tests one cut.

  Full lifecycle (no bisect): runs all BIT-I scripts, samples fixed-PA, then
  by default stops if POST already activated (H79 causal stop).  Set
  ``KEPLER_LIFECYCLE_THROUGH_RAM=1`` to continue into ``run_vbios_ram_init``
  and sample again (Night41ah preservation discriminator).  Set
  ``KEPLER_LIFECYCLE_THROUGH_LTC=1`` to also run ``_gk104_post_ram_fb_ltc``
  and sample again (Night41ai; implies through-ram).  Set
  ``KEPLER_LIFECYCLE_THROUGH_BAR=1`` to also run a one-page
  ``_gk104_init_bar1_identity`` and sample again (Night41aj; implies
  through-ltc). Optional ``KEPLER_LIFECYCLE_BAR_MAP_SIZE`` (default
  ``0x1000``).
  """
  dev = APLRemotePCIDevice.probe()
  if dev is None:
    print("probe-nouveau-base-lifecycle: TinyGPU.app socket is not reachable")
    raise SystemExit(1)
  try:
    _boot0, ensure_meta = shared._gk104_ensure_bar0_mmio(
        dev, allow_reset=False)
    if ensure_meta.get("did_reset"):
      raise RuntimeError("lifecycle probe forbids PCI reset")
    class BAR0Dev:
      def read32(self, r): return dev.mmio_read32(0, r)
      def write32(self, r, v): dev.mmio_write32(0, r, v)
      def read8(self, r): return dev.mmio_read(0, r, 1)[0]
      def write8(self, r, v): dev.mmio_write(0, r, bytes((v & 0xff,)))
    bar0 = BAR0Dev()
    before = shared._gk104_post_entry_probe(bar0)
    if (not shared._gk104_boot0_looks_live(before["boot0"]) or
        before["posted_marker"] or before["pramin_positive"]):
      raise RuntimeError(
          "lifecycle probe requires a live, unposted, PRAMIN-negative cold "
          f"entry (boot0={before['boot0']:#010x} "
          f"posted={before['posted_marker']} "
          f"pramin={before['pramin_word']:#010x}); replug without retry")
    # Night41ah+: KEPLER_LIFECYCLE_THROUGH_RAM=1 continues into RAMMAP even
    # when after-POST fixed-PA PRAMIN is already positive (H79 closed).  Strap
    # override stays forbidden on the bisect/causal-stop path; through-ram may
    # pin Palit strap 6 only after POST if 0x101000 is still unread.
    # Night41ai+: KEPLER_LIFECYCLE_THROUGH_LTC=1 continues past RAM into
    # Nouveau's post-RAM FB page + LTC init (implies through-ram).
    # Night41aj+: KEPLER_LIFECYCLE_THROUGH_BAR=1 continues past LTC into a
    # minimal BAR1 identity bootstrap (implies through-ltc).
    # Night41am+: KEPLER_LIFECYCLE_THROUGH_PMC=1 continues past BAR into
    # Nouveau nv50_mc_init-only PMC_ENABLE=0xffffffff (implies through-bar;
    # no PGOB — H93a discriminator).
    through_pmc = os.environ.get("KEPLER_LIFECYCLE_THROUGH_PMC", "0") == "1"
    through_bar = (os.environ.get("KEPLER_LIFECYCLE_THROUGH_BAR", "0") == "1" or
                   through_pmc)
    through_ltc = (os.environ.get("KEPLER_LIFECYCLE_THROUGH_LTC", "0") == "1" or
                   through_bar)
    through_ram = (os.environ.get("KEPLER_LIFECYCLE_THROUGH_RAM", "0") == "1" or
                   through_ltc)
    if (os.environ.get("KEPLER_RAMCFG_STRAP") not in (None, "") and
        not through_ram):
      raise RuntimeError(
          "lifecycle probe forbids KEPLER_RAMCFG_STRAP override; "
          "the post-POST live strap must select RAMCFG "
          "(set KEPLER_LIFECYCLE_THROUGH_RAM=1 to allow a post-POST pin)")
    def pramin_positive(words):
      return any(not shared._gk104_pramin_word_is_stub(word) and
                 word not in (0x00000000, 0xffffffff) for word in words)
    image, _bit_i, scripts = shared.vbios_init_info(shared.DEFAULT_VBIOS)
    os.environ.setdefault("KEPLER_VBIOS_I2C_TRACE", "1")
    if bisect_post_scripts:
      raw_prefix = os.environ.get("KEPLER_POST_SCRIPT_PREFIX", "").strip()
      if not raw_prefix:
        raise RuntimeError(
            "Night41t retired mid-POST 0x1700 sampling; set "
            "KEPLER_POST_SCRIPT_PREFIX to a 1-based script count "
            f"(1..{len(scripts)}) for one end-only fixed-PA sample")
      prefix = int(raw_prefix, 0)
      if prefix < 1 or prefix > len(scripts):
        raise RuntimeError(
            f"KEPLER_POST_SCRIPT_PREFIX={prefix} out of range 1..{len(scripts)}")
      raw_stop = os.environ.get("KEPLER_NVINIT_STOP_OFFSET", "").strip()
      stop_before = int(raw_stop, 0) if raw_stop else None
      chosen = scripts[:prefix]
      init = shared.nvbios_init.NvbiosInit(bar0, image, debug=False)
      init.unlock_vga_crtc()
      for index, script in enumerate(chosen):
        is_last = index + 1 == len(chosen)
        if is_last and stop_before is not None:
          init.run_script(script, stop_before=stop_before)
          print(f"probe-nouveau-post-script-bisect: "
                f"ran script[{index}]={script:#06x} "
                f"stop_before={stop_before:#x} "
                f"(ended@{init.offset:#x}; no mid-POST PRAMIN sample)",
                flush=True)
        else:
          init.run_script(script)
          print(f"probe-nouveau-post-script-bisect: "
                f"ran script[{index}]={script:#06x} "
                f"(no mid-POST PRAMIN sample)", flush=True)
      label = (f"POST prefix={prefix} last={chosen[-1]:#06x}"
               + (f" stop_before={stop_before:#x}" if stop_before is not None
                  else ""))
      words = shared._gk104_pramin_stage_snapshot(
          bar0, label, pa=0xfffe0000)
      activated = pramin_positive(words)
      stop_s = (f"{stop_before:#x}" if stop_before is not None else "none")
      print("probe-nouveau-post-script-bisect: "
            f"prefix={prefix} last={chosen[-1]:#06x} "
            f"stop_before={stop_s} "
            f"activated={activated}; "
            "intentional causal stop before RAM "
            "(single end-of-prefix 0x1700 sample only)")
      return
    shared.nvbios_init.run_vbios_init(bar0, image, scripts, debug=False)
    after_post = shared._gk104_pramin_stage_snapshot(
        bar0, "nouveau-measure after POST", pa=0xfffe0000)
    post_ok = pramin_positive(after_post)
    if post_ok and not through_ram:
      print("probe-nouveau-base-lifecycle: activated=after-post; "
            "intentional causal stop before RAM "
            "(set KEPLER_LIFECYCLE_THROUGH_RAM=1 to continue into RAMMAP)")
      return
    if through_ram:
      strap_reg = bar0.read32(0x101000)
      print(f"probe-nouveau-base-lifecycle: after-post "
            f"activated={post_ok}; through-ram; "
            f"0x101000={strap_reg:#010x}", flush=True)
      if ((strap_reg & 0x0000003c) == 0 and
          os.environ.get("KEPLER_RAMCFG_STRAP") in (None, "")):
        # Cold unread strap would select the wrong M0205/M0209 tables;
        # pin Palit golden strap 6 only for the RAMMAP phase.
        os.environ["KEPLER_RAMCFG_STRAP"] = "6"
        print("probe-nouveau-base-lifecycle: pinned "
              "KEPLER_RAMCFG_STRAP=6 for RAMMAP (0x101000 unread)",
              flush=True)
    shared.nvbios_init.run_vbios_ram_init(bar0, image, debug=False)
    after_ram = shared._gk104_pramin_stage_snapshot(
        bar0, "nouveau-measure after RAM", pa=0xfffe0000)
    ram_ok = pramin_positive(after_ram)
    after_ltc = None
    ltc_ok = False
    after_bar = None
    bar_ok = False
    bar1_dword = None
    bar1_ctl = None
    if through_ltc:
      if not ram_ok:
        print("probe-nouveau-base-lifecycle: through-ltc skipped; "
              "after-RAM fixed-PA not positive", flush=True)
      else:
        print("probe-nouveau-base-lifecycle: through-ltc; "
              "running post-RAM FB page + LTC", flush=True)
        shared._gk104_post_ram_fb_ltc(bar0)
        after_ltc = shared._gk104_pramin_stage_snapshot(
            bar0, "nouveau-measure after LTC", pa=0xfffe0000)
        ltc_ok = pramin_positive(after_ltc)
    if through_bar:
      if not ltc_ok:
        print("probe-nouveau-base-lifecycle: through-bar skipped; "
              "after-LTC fixed-PA not positive", flush=True)
      else:
        # Nouveau-shaped BAR1 identity: PRAMIN roots + 0x1704 enable.
        # Default one page (H89/H90); H91 uses KEPLER_LIFECYCLE_BAR_MAP_SIZE
        # up to 0x1000000 (16 MiB) — SPT bank max before inst@0x60000.
        map_size = int(os.environ.get("KEPLER_LIFECYCLE_BAR_MAP_SIZE",
                                      "0x1000"), 0)
        print(f"probe-nouveau-base-lifecycle: through-bar; "
              f"BAR1 identity map_size={map_size:#x}", flush=True)
        shared._gk104_init_bar1_identity(
            bar0, mapped_size=map_size, map_vram=True)
        after_bar = shared._gk104_pramin_stage_snapshot(
            bar0, "nouveau-measure after BAR1", pa=0xfffe0000)
        bar_ok = pramin_positive(after_bar)
        bar1_ctl = bar0.read32(0x001704) & 0xffffffff
        # Identity VA→PA; sample page 0, mid, and last mapped page (H91).
        n_pages = max(map_size // 0x1000, 1)
        sample_pages = [0]
        if n_pages > 1:
          sample_pages.append(n_pages // 2)
        if n_pages > 2:
          sample_pages.append(n_pages - 1)
        sample_pages = sorted(set(sample_pages))
        # TinyGPU MMIO_READ needs prior MAP_BAR (bar_info); BAR0 already mapped
        # in _gk104_ensure_bar0_mmio — Night41aj skipped this for BAR1 (H90a).
        bar1_dword = None
        page0_dword = None
        multi_ok = True
        try:
          bar1_addr, bar1_size = dev.bar_info(1)
          print(f"probe-nouveau-base-lifecycle: MAP_BAR1 "
                f"addr={bar1_addr:#x} size={bar1_size:#x}", flush=True)
          for page in sample_pages:
            pa = page * 0x1000
            # Use offset-correct PRAMIN read (stage_snapshot only hits window base).
            pr = shared._gk104_pramin_read32(bar0, pa) & 0xffffffff
            raw = bytes(dev.mmio_read(1, pa, 4))
            b1 = struct.unpack("<I", raw)[0] if len(raw) == 4 else None
            hit = b1 is not None and b1 == pr
            multi_ok = multi_ok and hit
            if page == 0:
              page0_dword, bar1_dword = pr, b1
            print(f"probe-nouveau-base-lifecycle: page{page} "
                  f"PRAMIN={pr:#010x} BAR1={b1:#010x} match={hit}"
                  if b1 is not None else
                  f"probe-nouveau-base-lifecycle: page{page} "
                  f"PRAMIN={pr:#010x} BAR1=unreadable match=False",
                  flush=True)
        except Exception as e:
          multi_ok = False
          print(f"probe-nouveau-base-lifecycle: physical BAR1 "
                f"read failed: {e}", flush=True)
        page0_s = (f"{page0_dword:#010x}" if page0_dword is not None
                   else "None")
        bar1_s = (f"{bar1_dword:#010x}" if bar1_dword is not None
                  else "unreadable")
        print(f"probe-nouveau-base-lifecycle: 0x1704={bar1_ctl:#010x} "
              f"PRAMIN[PA0]={page0_s} BAR1[0]={bar1_s} "
              f"match={multi_ok} pages={sample_pages}",
              flush=True)
    topo_before_pmc = None
    topo_after_pmc = None
    pmc_before = None
    pmc_after = None
    if through_pmc:
      if not bar_ok:
        print("probe-nouveau-base-lifecycle: through-pmc skipped; "
              "after-BAR fixed-PA not positive", flush=True)
      else:
        # Nouveau nv50_mc_init / gk104_mc.init: wr32(0x000200, 0xffffffff).
        # H93a: isolate MC full enable from PGOB (gf100_gr_oneinit).
        pmc_before = bar0.read32(0x000200) & 0xffffffff
        topo_before_pmc = bar0.read32(0x409604) & 0xffffffff
        pgraph_before = bar0.read32(0x400000) & 0xffffffff
        print(f"probe-nouveau-base-lifecycle: through-pmc; "
              f"before PMC_ENABLE={pmc_before:#010x} "
              f"topo={topo_before_pmc:#010x} "
              f"PGRAPH={pgraph_before:#010x}", flush=True)
        bar0.write32(0x000200, 0xffffffff)
        _ = bar0.read32(0x000200)  # posting read
        pmc_after = bar0.read32(0x000200) & 0xffffffff
        topo_after_pmc = bar0.read32(0x409604) & 0xffffffff
        pgraph_after = bar0.read32(0x400000) & 0xffffffff
        fecs_scratch = bar0.read32(0x409800) & 0xffffffff
        ungated = (topo_after_pmc != 0xbadf1200 and
                   (topo_after_pmc & 0xffff0000) != 0xbadf0000)
        print(f"probe-nouveau-base-lifecycle: after PMC_ENABLE="
              f"{pmc_after:#010x} topo={topo_after_pmc:#010x} "
              f"PGRAPH={pgraph_after:#010x} "
              f"0x409800={fecs_scratch:#010x} ungated={ungated}",
              flush=True)
        after_pmc_fixed = shared._gk104_pramin_stage_snapshot(
            bar0, "nouveau-measure after PMC", pa=0xfffe0000)
        print(f"probe-nouveau-base-lifecycle: after-pmc fixed-PA "
              f"preserved={pramin_positive(after_pmc_fixed)}",
              flush=True)
    # Summarize furthest stage reached.
    if through_pmc and topo_after_pmc is not None:
      ungated = (topo_after_pmc != 0xbadf1200 and
                 (topo_after_pmc & 0xffff0000) != 0xbadf0000)
      stage = ("after-pmc-ungated" if ungated else "after-pmc-still-gated")
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} "
            f"ltc={[hex(word) for word in after_ltc]} "
            f"bar={[hex(word) for word in after_bar]} "
            f"pmc={pmc_after:#010x} topo={topo_after_pmc:#010x} "
            f"activated={stage}")
    elif through_bar and after_bar is not None:
      if post_ok and ram_ok and ltc_ok and bar_ok:
        stage = "after-bar-preserved"
      elif post_ok and ram_ok and ltc_ok and not bar_ok:
        stage = "after-bar-clobbered"
      elif bar_ok:
        stage = "after-bar"
      else:
        stage = "none"
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} "
            f"ltc={[hex(word) for word in after_ltc]} "
            f"bar={[hex(word) for word in after_bar]} activated={stage}")
    elif through_ltc and after_ltc is not None:
      if post_ok and ram_ok and ltc_ok:
        stage = "after-ltc-preserved"
      elif post_ok and ram_ok and not ltc_ok:
        stage = "after-ltc-clobbered"
      elif ltc_ok:
        stage = "after-ltc"
      elif post_ok and ram_ok:
        stage = "after-ram-preserved"
      elif ram_ok:
        stage = "after-ram"
      elif post_ok:
        stage = "after-post-clobbered"
      else:
        stage = "none"
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} "
            f"ltc={[hex(word) for word in after_ltc]} activated={stage}")
    else:
      if post_ok and ram_ok:
        stage = "after-ram-preserved"
      elif ram_ok:
        stage = "after-ram"
      elif post_ok:
        stage = "after-post-clobbered"
      else:
        stage = "none"
      print("probe-nouveau-base-lifecycle: "
            f"entry={before['pramin_word']:#010x} "
            f"post={[hex(word) for word in after_post]} "
            f"ram={[hex(word) for word in after_ram]} activated={stage}")
  finally:
    dev.fini()

def main():
  shared.set_pci_transport_factory(_MacPCIDeviceFactory())
  os.environ.setdefault("KEPLER_NO_AUTO_SUDO", "1")
  # This diagnostic must observe the cold strap and forbids PCI reset.  Dispatch
  # before the normal launcher pins the known Palit strap for recovery paths.
  if "--probe-nouveau-base-lifecycle" in sys.argv:
    _probe_nouveau_base_lifecycle(); return
  if "--probe-nouveau-post-script-bisect" in sys.argv:
    _probe_nouveau_base_lifecycle(bisect_post_scripts=True); return
  # Nouveau golden mmiotrace for this Palit GTX 770 reads 0x101000=0x8040509a
  # (RAMCFG strap 6).  Cold eGPU bring-up sometimes returns 0 from that
  # register and then programs the wrong M0205/M0209 training tables; pin the
  # known strap unless the caller overrides it.
  os.environ.setdefault("KEPLER_RAMCFG_STRAP", "6")
  os.environ.setdefault("KEPLER_PMU_MEMX", "1")
  os.environ.setdefault("KEPLER_PMU_ENTER_NOWAIT", "1")
  os.environ.setdefault("KEPLER_RAM_MEMX_WR", "1")
  if "--probe" in sys.argv:
    _probe(); return
  if "--probe-post-ownership" in sys.argv:
    _probe_post_ownership(); return
  if "--probe-rom-shadow-ownership" in sys.argv:
    _probe_rom_shadow_ownership(); return
  if "--probe-golden-preinit" in sys.argv:
    _probe_golden_preinit(); return
  if "--probe-option-rom-vga-preamble" in sys.argv:
    _probe_option_rom_vga_preamble(); return
  if "--probe-nouveau-init-io" in sys.argv:
    _probe_nouveau_init_io(); return
  backend = os.environ.get("NV_BACKEND", "kepler")
  offline = any(x in sys.argv for x in (
      "--middle-selftest", "--mmiotrace-selftest",
      "--vbios-info", "--vbios-init-info", "--compare-cubin"))
  # Live TinyGPU only: one atomic PMU RAM transition.  Keep offline golden
  # selftests on KEPLER_RAM_BLOCK=0.
  if not offline:
    # Run the complete Nouveau memory transition inside one PMU script.  ENTER
    # may hide host MMIO transiently; LEAVE restores it before the reply.
    os.environ.setdefault("KEPLER_RAM_MEMX_ATOMIC", "1")
    # Night40ah: stock memx.fuc waits on FB_PAUSE during atomic ram_program.
    # Early PMU still loads with ENTER_NOWAIT; add.py reloads waits before RAM.
    # Skip ENTER+LEAVE preflight under stock wait (can hang before MC train).
    os.environ.setdefault("KEPLER_RAM_ENTER_WAIT", "1")
    os.environ.setdefault("KEPLER_RAM_ATOMIC_PREFLIGHT", "0")
    os.environ.setdefault("KEPLER_RAM_BLOCK", "atomic")
    # Never host-program GDDR5 without MEMX (kills BAR0). Soft PRAMIN live
    # skips the destructive PRAMIN window poke until BAR1 bootstrap.
    os.environ.setdefault("KEPLER_RAM_REQUIRE_MEMX", "1")
    os.environ.setdefault("KEPLER_PRAMIN_SOFT_LIVE", "1")
    # Refuse GPC-awake+PRAMIN-stub half-POST (dirty) — that path hung USB4 /
    # WindowServer.  Enclosure power-cycle required; KEPLER_ALLOW_DIRTY=1 opts in.
    os.environ.setdefault("KEPLER_REFUSE_DIRTY", "1")
    # Night10: post-MEMX bit0 then LTC/ZBC (0x100c80/0x17ea*) collapsed TinyGPU BAR0.
    # Night40aj: with KEPLER_RAM_BIT0_DEFER=1, run Nouveau post-RAM fb_init_page
    # + LTC/ZBC in golden order *before* bit0.
    os.environ.setdefault("KEPLER_POST_RAM_LTC", "1")
    # Host write to 0x4041f0 after bit0 collapses BAR0 (even no-op).
    os.environ.setdefault("KEPLER_PGRAPH_BLCG", "0")
    # Keep ENTER/XFER/LEAVE inside one autonomous PMU routine; standalone
    # ENTER hides host PMU MMIO before a separate LEAVE can be submitted.
    os.environ.setdefault("KEPLER_RAM_BIT0_DEFER", "1")
    # Full pack is OK *before* bit0 (night13 FECS ready); keep default on.
    os.environ.setdefault("KEPLER_PGRAPH_PACK", "1")
    # Literal PRAMIN 0 on XOR virgin cannot stick and hung TinyGPU (night13).
    os.environ.setdefault("KEPLER_PRAMIN_LITERAL", "0")
    # Host 0x1700 after bit0 kills BAR0 (night14); store PRAMIN via MEMX WR32.
    os.environ.setdefault("KEPLER_PRAMIN_MEMX", "1")
    # Retained for the legacy post-bit0 WR32 path; the default uses MEMIF.
    os.environ.setdefault("KEPLER_BAR1_MEMX_LITERAL", "1")
    # Store minimal BAR1 roots with PMU xdst inside autonomous ENTER/LEAVE.
    # The option name is retained for compatibility with night40o-r.
    os.environ.setdefault("KEPLER_BAR1_DIRECT_PHYS", "1")
    # macOS/TinyGPU only: the embedded PMU pad owns ENTER→xdst→LEAVE and
    # publishes DONE after host visibility is restored.  Shared/Linux never
    # enables this experimental bootstrap.
    os.environ.setdefault("KEPLER_TINYGPU_ATOMIC_BAR1", "1")
    # 16 MiB BAR1 covers bit19-safe GR/attrib; keeps MEMX PRAMIN tractable.
    os.environ.setdefault("KEPLER_BAR1_MAP_SIZE", "0x1000000")
    # Default all-MEMX (host-then-MEMX timed out on cold); opt-in host prog0.
    os.environ.setdefault("KEPLER_RAM_HOST_PROG0", "0")
    # Experimental early bit0 path: skip RAMMAP unless continuing into MEMX.
    if os.environ.get("KEPLER_RAM_PROGRAM") == "bit0-only":
      if os.environ.get("KEPLER_RAM_AFTER_BIT0") != "memx":
        os.environ.setdefault("KEPLER_RAM_INIT", "0")
  else:
    os.environ.setdefault("KEPLER_RAM_BLOCK", "0")
    os.environ.setdefault("KEPLER_RAM_REQUIRE_MEMX", "0")
  if backend != "software" and not offline:
    if os.environ.get("KEPLER_LIVE_ACK") != "completion-abort-risk":
      raise SystemExit("hardware launch refused: set KEPLER_LIVE_ACK=completion-abort-risk for an authorized TinyGPU test")
    if not os.environ.get("KEPLER_RPC_TRACE"):
      raise SystemExit("hardware launch refused: KEPLER_RPC_TRACE is required")
  shared.main()

if __name__ == "__main__":
  main()
