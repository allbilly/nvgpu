#!/usr/bin/env python3
"""FIFO + channel creation test for GT218 (G82/NV50 FIFO class).

Tests: FIFO init -> channel creation -> NOP method submission via GPFIFO.
Based on nouveau nv50.c / g84.c channel layout.
"""
from __future__ import annotations
import sys, os, struct, socket, time, enum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples_kepler'))
from nvbios_init import find_vbios_scripts, run_vbios_init

# ============================================================================
# TinyGPU transport
# ============================================================================
class RemoteCmd(enum.IntEnum):
  MAP_BAR    = 1
  CFG_READ   = 3
  CFG_WRITE  = 4
  RESET      = 5
  MMIO_READ  = 6
  MMIO_WRITE = 7

TINYGPU_SOCK = '/tmp/tinygpu.sock'

class Dev:
  def __init__(self):
    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self._sock.settimeout(10.0)
    for _ in range(100):
      try: self._sock.connect(TINYGPU_SOCK); break
      except: time.sleep(0.05)
  def _rpc(self, cmd, bar=0, a1=0, a2=0, a3=0, payload=b'', readout=0):
    self._sock.sendall(struct.pack('<BIIQQQ', int(cmd), 0, bar, a1, a2, a3) + payload)
    if payload: return None
    msg = self._recvall(17)
    status, v1, v2 = struct.unpack('<BQQ', msg)
    if status != 0:
      err = self._recvall(v1).decode() if v1 > 0 else 'unknown'
      raise RuntimeError(f'RPC failed: {err}')
    data = self._recvall(readout) if readout else b''
    return v1, v2, data
  def _recvall(self, n):
    buf = bytearray(n); got = 0
    while got < n:
      cnt = self._sock.recv_into(memoryview(buf)[got:]); got += cnt
    return bytes(buf)
  def read32(self, reg):
    _, _, d = self._rpc(RemoteCmd.MMIO_READ, 0, reg & 0xffffffff, 4, readout=4)
    return struct.unpack_from('<I', d)[0]
  def write32(self, reg, val):
    self._rpc(RemoteCmd.MMIO_WRITE, 0, reg & 0xffffffff, 4, payload=struct.pack('<I', val & 0xffffffff))
  def cfg(self, off, sz=4):
    return self._rpc(RemoteCmd.CFG_READ, 0, off, sz)[0]
  def map_bar(self, bar):
    v1, v2, _ = self._rpc(RemoteCmd.MAP_BAR, bar)
    return v1, v2
  def close(self):
    try: self._sock.close()
    except: pass

# ============================================================================
# VRAM access via PRAMIN window (BAR0 + 0x700000, window via reg 0x001700)
# ============================================================================
PRAMIN_WINDOW_REG = 0x001700
PRAMIN_BASE = 0x700000

class VRAM:
  def __init__(self, dev):
    self.dev = dev
    self._cur_window = None
  def _set_window(self, addr_base):
    page = addr_base >> 16
    if page != self._cur_window:
      self.dev.write32(PRAMIN_WINDOW_REG, page)
      self._cur_window = page
  def write32(self, vram_addr, val):
    base = vram_addr & 0xFFFFFF00000
    offset = vram_addr & 0x000000FFFFF
    self._set_window(base)
    self.dev.write32(PRAMIN_BASE + offset, val)
  def read32(self, vram_addr):
    base = vram_addr & 0xFFFFFF00000
    offset = vram_addr & 0x000000FFFFF
    self._set_window(base)
    return self.dev.read32(PRAMIN_BASE + offset)
  def write_block(self, vram_addr, data: bytes):
    for i in range(0, len(data), 4):
      val = struct.unpack_from('<I', data, i)[0] if i + 4 <= len(data) else \
            struct.unpack('<I', data[i:].ljust(4, b'\x00'))[0]
      self.write32(vram_addr + i, val)
  def read_block(self, vram_addr, size: int) -> bytes:
    data = bytearray()
    for i in range(0, size, 4):
      data.extend(struct.pack('<I', self.read32(vram_addr + i)))
    return bytes(data[:size])

# ============================================================================
# NV50 push buffer method encoding (cl506f.h)
# ============================================================================
def nvm50(subch, method, *args, inc=True):
  """Encode NV50 push buffer method.
  inc=True: incrementing; inc=False: non-incrementing.
  bits[31:29]=SEC_OP (0=inc, 2=noninc), bits[28:18]=count, bits[17:16]=TERT_OP(0),
  bits[15:13]=subch, bits[12:2]=method_dword_addr
  The method byte address goes at bits[12:0] (4-byte aligned, bits[1:0]=0).
  Hardware extracts dword addr from bits[12:2].
  """
  sec_op = 0 if inc else 2
  tert_op = 0  # INC_METHOD (GRP0) or NON_INC_METHOD (GRP2)
  count = len(args)
  hdr = (sec_op << 29) | (count << 18) | (tert_op << 16) | (subch << 13) | method
  return [hdr] + list(args)

# ============================================================================
# NV50 GPFIFO entry (cl506f.h)
# ============================================================================
def gp_entry(push_offset, length_dwords, priv=0, level=0, no_ctx_switch=0, disable=0):
  """8-byte GPFIFO entry (cl506f.h + nvif/chan506f.c).
  entry0: bit0=disable, bit1=no_context_switch, bits[31:2]=push_addr[31:2]
  entry1: bits[7:0]=addr_hi, bit8=priv, bit9=level, bits[31:10]=length_in_dwords
  length is in DWORDS (nouveau: (size >> 2) << 10), not bytes.
  """
  entry0 = (push_offset & ~3) | (no_ctx_switch << 1) | disable
  get_hi = (push_offset >> 32) & 0xFF
  entry1 = (length_dwords << 10) | (level << 9) | (priv << 8) | get_hi
  return struct.pack('<II', entry0, entry1)

# ============================================================================
# NV50 DMA object (nv50_dmaobj_bind)
# ============================================================================
def make_dma_object(start, limit, target='vram', access='rw', cls=0x00000002):
  """24-byte NV50 DMA object (nv50_dmaobj_bind format, usernv50.c).
  For non-VM target: priv=1(US), part=1(256B), comp=0(NONE), kind=0(PITCH)
  For VM target:     priv=0(VM), part=0(VM), comp=3(VM), kind=0x7f(VM)
  Constants from cl0002.h.
  """
  tgt = {'vram': 0x00010000, 'pci': 0x00020000, 'ncoh': 0x00030000, 'vm': 0}[target]
  if target == 'vm':
    acc = 0  # NV_MEM_ACCESS_VM → no access bits
    priv = 0  # NV50_DMA_V0_PRIV_VM
    comp, kind, part = 3, 0x7f, 0  # COMP_VM, KIND_VM, PART_VM
  else:
    acc = {'ro': 0x00040000, 'rw': 0x00080000}[access]
    priv = 1  # NV50_DMA_V0_PRIV_US
    comp, kind, part = 0, 0, 1  # COMP_NONE, KIND_PITCH, PART_256
  flags0 = (comp << 29) | (kind << 22) | (priv << 20) | cls | tgt | acc
  flags5 = (part << 16)
  return struct.pack('<IIIIII',
    flags0, limit & 0xFFFFFFFF, start & 0xFFFFFFFF,
    ((limit >> 32) & 0xFF) << 24 | ((start >> 32) & 0xFF),
    0, flags5)

# ============================================================================
# Constants
# ============================================================================
CHAN_ID = 1
USERD_BASE = 0xc00000
USERD_SIZE = 0x2000
# Flow control starts at 0x7c0 dwords = 0x1f00 bytes into USERD page
# But on NV50/G84, USERD maps inst block first 0x2000 bytes, control at 0x1f00
# Actually from testing: flow control is at offset 0 within USERD page
# Put=0x40, Get=0x44, Ref=0x48, GPGet=0x88, GPPut=0x8c

VRAM_BASE = 0x100000  # 1 MB into VRAM

# g84 instance block layout (all allocated within 0x10000 inst block):
# HEAD allocations:
#   0x0000: eng    (0x200 bytes, align 0)
#   0x0200: pgd    (0x4000 bytes, align 0)
#   0x4400: cache  (0x1000 bytes, align 0x400)
#   0x5400: ramfc  (0x100 bytes, align 0x100)
#   0x5500: ramht  (0x8000 bytes, align 16)
#   0xD500: (free)
# TAIL allocation:
#   0xFFE0: push DMA object (24 bytes, align 16, from tail)
INST_ENG_OFF    = 0x0000
INST_PGD_OFF    = 0x0200
INST_CACHE_OFF  = 0x4400
INST_RAMFC_OFF  = 0x5400
INST_RAMHT_OFF  = 0x5500
INST_PUSH_OFF   = 0xFFE0

# GMMU page table (1MB PGT in VRAM, identity-mapped for push buffer region)
PGT_ADDR = VRAM_BASE + 0x30000  # 1MB PGT at VRAM 0x130000

def setup_gmmu(vram, inst_addr):
  """Set up a minimal GMMU page table for identity-mapped VRAM pages.

  NV50 GMMU: 2-level page table.
  - PGD (in inst block at offset 0x0200): 11-bit index [39:29], 8-byte entries
  - PGT (1MB, 17-bit index [28:12] for 4KB pages): 8-byte entries
  PDE format: 0x003 | pgt_addr (4KB pages, 1MB PGT, VRAM target)
  PTE format: phys_addr | 0x1 (valid, VRAM aperture)
  """
  print('[gmmu] Setting up page table...')
  # Zero the PGT (1MB = 131072 entries * 8 bytes)
  # Only zero the first 512 entries (covers VA 0x00000000-0x001FF000)
  for i in range(512):
    vram.write32(PGT_ADDR + i * 8, 0)
    vram.write32(PGT_ADDR + i * 8 + 4, 0)

  # Write PDE in instance block at offset 0x0200 (pd_offset for g84)
  # PDE = 0x003 (4KB pages, 1MB PGT) | PGT_ADDR (VRAM target = 0)
  pde = 0x00000003 | PGT_ADDR
  vram.write32(inst_addr + INST_PGD_OFF, pde & 0xFFFFFFFF)
  vram.write32(inst_addr + INST_PGD_OFF + 4, (pde >> 32) & 0xFF)

  # Identity-map VRAM pages for inst (0x100000), push buf (0x110000), gpfifo (0x120000)
  # PGT index = (vaddr >> 12) & 0x1FFFF
  # PTE = phys_addr | 0x1 (valid + VRAM)
  for vaddr in range(0x100000, 0x130000, 0x1000):
    pgt_idx = (vaddr >> 12) & 0x1FFFF
    pte = vaddr | 0x1  # identity mapping: virtual = physical
    vram.write32(PGT_ADDR + pgt_idx * 8, pte & 0xFFFFFFFF)
    vram.write32(PGT_ADDR + pgt_idx * 8 + 4, (pte >> 32) & 0xFF)

  print(f'[gmmu] PGT at 0x{PGT_ADDR:08x}, PDE=0x{pde:08x}')
  print(f'[gmmu] Mapped VA 0x100000-0x12FFFF -> phys (identity)')

  # Verify PDE
  pde_r0 = vram.read32(inst_addr + INST_PGD_OFF)
  pde_r1 = vram.read32(inst_addr + INST_PGD_OFF + 4)
  print(f'[gmmu] PDE readback: 0x{pde_r1:08x}{pde_r0:08x}')

  # Verify a few PTEs
  for vaddr in [0x100000, 0x110000, 0x120000]:
    pgt_idx = (vaddr >> 12) & 0x1FFFF
    pte_r0 = vram.read32(PGT_ADDR + pgt_idx * 8)
    pte_r1 = vram.read32(PGT_ADDR + pgt_idx * 8 + 4)
    print(f'[gmmu] PTE[{pgt_idx:#x}] (VA 0x{vaddr:x}): 0x{pte_r1:08x}{pte_r0:08x}')

  # Flush GMMU TLB for all known engine IDs
  for eng_id in [0x00, 0x01, 0x06, 0x08, 0x09, 0x0a, 0x0d]:
    vram.dev.write32(0x100c80, (eng_id << 16) | 1)
    for _ in range(2000):
      if not (vram.dev.read32(0x100c80) & 0x00000001):
        break
      time.sleep(0.001)
  print('[gmmu] TLB flushed')

def fifo_init(dev):
  """nv50_fifo_init."""
  print('[fifo] Init...')
  dev.write32(0x000200, dev.read32(0x000200) & ~0x00000100)
  dev.write32(0x000200, dev.read32(0x000200) | 0x00000100)
  dev.write32(0x00250c, 0x6f3cfc34)
  dev.write32(0x002044, 0x01003fff)
  dev.write32(0x002100, 0xffffffff)  # clear intr
  dev.write32(0x002140, 0xbfffffff)  # intr enable
  for i in range(128):
    dev.write32(0x002600 + i * 4, 0)
  # Empty runlist update
  dev.write32(0x0032f4, 0)
  dev.write32(0x0032ec, 0)
  dev.write32(0x003200, 0x00000001)
  dev.write32(0x003250, 0x00000001)
  dev.write32(0x002500, 0x00000001)
  print(f'[fifo] PFIFO_ENABLE=0x{dev.read32(0x002500):08x}')

CHIPSET = 0xa8  # GT218

def gr_init(dev, ctxprog=None, ctxvals=None, ctxvals_size=0):
  """nv50_gr_init for GT218 (chipset 0xa8).
  ctxprog: list of 32-bit instructions (from ctxnv50.py)
  ctxvals: bytes buffer of default register values
  ctxvals_size: size of ctxvals buffer
  """
  print('[gr] Init...')
  # HW context switch enable
  dev.write32(0x40008c, 0x00000004)

  # Reset/enable traps
  dev.write32(0x400804, 0xc0000000)
  dev.write32(0x406800, 0xc0000000)
  dev.write32(0x400c04, 0xc0000000)
  dev.write32(0x401800, 0xc0000000)
  dev.write32(0x405018, 0xc0000000)
  dev.write32(0x402000, 0xc0000000)

  # Per-TP trap clear (chipset >= 0xa0)
  units = dev.read32(0x001540)
  print(f'[gr] Units bitmap: 0x{units:08x}')
  for i in range(16):
    if not (units & (1 << i)):
      continue
    dev.write32(0x408600 + (i << 11), 0xc0000000)
    dev.write32(0x408708 + (i << 11), 0xc0000000)
    dev.write32(0x40831c + (i << 11), 0xc0000000)

  # Interrupt enable
  dev.write32(0x400108, 0xffffffff)
  dev.write32(0x400138, 0xffffffff)
  dev.write32(0x400100, 0xffffffff)
  dev.write32(0x40013c, 0xffffffff)
  dev.write32(0x400500, 0x00010001)

  # Upload ctxprog
  if ctxprog:
    print(f'[gr] Uploading ctxprog ({len(ctxprog)} instructions)...')
    dev.write32(0x400324, 0)  # reset index
    for instr in ctxprog:
      dev.write32(0x400328, instr)
    print(f'[gr] Ctxprog uploaded, ctxvals_size={ctxvals_size}')
  else:
    print('[gr] WARNING: No ctxprog provided — GR will not work!')

  # Clear context pointers
  dev.write32(0x400824, 0)
  dev.write32(0x400828, 0)
  dev.write32(0x40082c, 0)
  dev.write32(0x400830, 0)
  dev.write32(0x40032c, 0)
  dev.write32(0x400330, 0)

  # ZCULL config (chipset 0xa8, not a0/aa/ac)
  dev.write32(0x402cc0, 0x00000000)
  dev.write32(0x402ca8, 0x00000002)

  # Zero ZCULL regions
  for i in range(8):
    dev.write32(0x402c20 + i * 0x10, 0)
    dev.write32(0x402c24 + i * 0x10, 0)
    dev.write32(0x402c28 + i * 0x10, 0)
    dev.write32(0x402c2c + i * 0x10, 0)

  print('[gr] Init complete')
  return ctxvals_size

def gr_bind_context(dev, vram, chan_id, inst_addr, ctx_addr, ctx_size):
  """Bind a GR context to the channel (g84_ectx_bind for GR engine).
  ptr0 = 0x0020 for GR engine.
  The context buffer must be in VRAM, filled with ctxvals.
  """
  print(f'[gr] Binding GR context to channel {chan_id}...')
  # eng block is at inst+0x0000 (INST_ENG_OFF)
  eng_addr = inst_addr + INST_ENG_OFF
  ptr0 = 0x0020  # GR engine
  flags = 0x00190000
  limit = ctx_addr + ctx_size - 1

  vram.write32(eng_addr + ptr0 + 0x00, flags)
  vram.write32(eng_addr + ptr0 + 0x04, limit & 0xFFFFFFFF)
  vram.write32(eng_addr + ptr0 + 0x08, ctx_addr & 0xFFFFFFFF)
  vram.write32(eng_addr + ptr0 + 0x0c,
    ((limit >> 32) & 0xFF) << 24 | (ctx_addr >> 32) & 0xFF)
  vram.write32(eng_addr + ptr0 + 0x10, 0)
  vram.write32(eng_addr + ptr0 + 0x14, 0)
  print(f'[gr] GR context: addr=0x{ctx_addr:x} size={ctx_size} limit=0x{limit:x}')

def create_channel(dev, vram, push_words=None, setup_fn=None):
  """Create a G84_CHANNEL_GPFIFO channel and submit push words via GPFIFO.
  setup_fn(vram, inst_addr) is called after RAMFC+GMMU setup, before channel bind.
  Use it to add RAMHT entries, GR context binding, extra GMMU mappings, etc.
  """
  chan_id = CHAN_ID
  inst_addr = VRAM_BASE
  push_buf_addr = VRAM_BASE + 0x10000  # 64KB push buffer
  gp_fifo_addr = VRAM_BASE + 0x20000   # GPFIFO ring (4KB = 512 entries)
  runlist_addr = VRAM_BASE + 0x21000   # Runlist

  if push_words is None:
    push_words = [0x00000000]  # NOP

  print(f'[chan] Channel {chan_id}, inst=0x{inst_addr:08x}')
  print(f'[chan] push=0x{push_buf_addr:08x}, gpfifo=0x{gp_fifo_addr:08x}')

  # Zero instance block (0x10000 bytes)
  print('[chan] Zeroing instance block...')
  for off in range(0, 0x10000, 4):
    vram.write32(inst_addr + off, 0)

  # Set up GMMU page table (identity-mapped VRAM pages)
  setup_gmmu(vram, inst_addr)

  # Write push DMA object at tail (offset 0xFFE0 in inst)
  # Try VM target with GMMU identity mapping (Tesla uses VM target for GART pushbuf)
  push_dma = make_dma_object(0, 0xFFFFFFFFFF, target='vm', access='rw')
  vram.write_block(inst_addr + INST_PUSH_OFF, push_dma)
  print(f'[chan] Push DMA at inst+0x{INST_PUSH_OFF:x}')

  # Write RAMFC at offset 0x5400 in inst
  ramfc_addr = inst_addr + INST_RAMFC_OFF
  gp_fifo_size = 0x1000  # 4KB = 512 entries
  gp_limit2 = 9  # ilog2(0x1000 / 8) = ilog2(512) = 9
  ramht_bits = 12  # order_base_2(0x8000/8) = order_base_2(0x1000) = 12

  vram.write32(ramfc_addr + 0x3c, 0x403f6078)
  vram.write32(ramfc_addr + 0x44, 0x01003fff)
  vram.write32(ramfc_addr + 0x48, INST_PUSH_OFF >> 4)  # push DMA offset >> 4
  vram.write32(ramfc_addr + 0x50, gp_fifo_addr & 0xFFFFFFFF)  # GPFIFO addr low
  vram.write32(ramfc_addr + 0x54, ((gp_fifo_addr >> 32) & 0xFFFF) | (gp_limit2 << 16))
  vram.write32(ramfc_addr + 0x60, 0x7fffffff)
  vram.write32(ramfc_addr + 0x78, 0x00000000)
  vram.write32(ramfc_addr + 0x7c, 0x30000000 | 0xfff)  # devm=0xfff
  vram.write32(ramfc_addr + 0x80, ((ramht_bits - 9) << 27) | (4 << 24) | (INST_RAMHT_OFF >> 4))
  vram.write32(ramfc_addr + 0x88, (inst_addr + INST_CACHE_OFF) >> 10)  # cache abs addr >> 10
  vram.write32(ramfc_addr + 0x98, inst_addr >> 12)  # inst abs addr >> 12

  print('[chan] RAMFC:')
  for off in [0x3c, 0x44, 0x48, 0x50, 0x54, 0x60, 0x78, 0x7c, 0x80, 0x88, 0x98]:
    print(f'  [0x{off:02x}] = 0x{vram.read32(ramfc_addr + off):08x}')

  # Write push buffer data
  for i, w in enumerate(push_words):
    vram.write32(push_buf_addr + i * 4, w)
  push_len = len(push_words) * 4
  print(f'[chan] Push buffer: {len(push_words)} words ({push_len} bytes)')

  # Zero GPFIFO ring (prevent garbage entries causing MEM_FAULT)
  for off in range(0, 0x1000, 4):
    vram.write32(gp_fifo_addr + off, 0)
  # GPFIFO entry: GET=push_buf_addr (VM addr), length in DWORDS
  entry = gp_entry(push_buf_addr, len(push_words))
  vram.write_block(gp_fifo_addr, entry)
  print(f'[chan] GPFIFO entry: GET=0x{push_buf_addr:x} len={len(push_words)} dwords')

  # Call setup_fn before binding (for RAMHT, GR context, extra GMMU, etc.)
  if setup_fn:
    setup_fn(vram, inst_addr)

  # Bind channel: g84_chan_bind writes ramfc->addr >> 8
  chan_reg = 0x002600 + chan_id * 4
  dev.write32(chan_reg, ramfc_addr >> 8)
  print(f'[chan] Bind: chan_reg = 0x{ramfc_addr >> 8:08x}')

  # Update runlist: write channel ID, commit
  vram.write32(runlist_addr, chan_id)
  dev.write32(0x0032f4, runlist_addr >> 12)
  dev.write32(0x0032ec, 1)
  for _ in range(100):
    if not (dev.read32(0x0032ec) & 0x00000100): break
    time.sleep(0.01)
  print(f'[chan] Runlist committed, 0x0032ec=0x{dev.read32(0x0032ec):08x}')

  # Start channel: set bit 31
  dev.write32(chan_reg, (ramfc_addr >> 8) | 0x80000000)
  print(f'[chan] Started: chan_reg = 0x{(ramfc_addr >> 8) | 0x80000000:08x}')

  # Clear any pending interrupts
  dev.write32(0x002100, 0xffffffff)

  # Set GPPut to trigger GPFIFO processing
  userd = USERD_BASE + chan_id * USERD_SIZE
  print(f'[chan] USERD at BAR0+0x{userd:06x}')

  # Read initial flow control state
  print('[chan] Initial USERD:')
  for name, off in [('Put', 0x40), ('Get', 0x44), ('Ref', 0x48),
                    ('GPGet', 0x88), ('GPPut', 0x8c)]:
    print(f'  {name:6s} [0x{off:02x}] = 0x{dev.read32(userd + off):08x}')

  # Write GPPut = 1 (entry index, NOT byte offset — see nvif/chan506f.c)
  dev.write32(userd + 0x8c, 1)
  print(f'[chan] Wrote GPPut=1 (entry index)')

  # Poll for completion
  for attempt in range(200):
    time.sleep(0.02)
    gp_get = dev.read32(userd + 0x88)
    gp_put = dev.read32(userd + 0x8c)
    fifo_intr = dev.read32(0x002100)
    if gp_get == gp_put or fifo_intr != 0:
      break

  print(f'[chan] After poll ({attempt+1} iterations):')
  print(f'  GPGet=0x{gp_get:08x} GPPut=0x{gp_put:08x}')
  print(f'  PFIFO_INTR=0x{fifo_intr:08x}')

  # Read all flow control fields
  print('[chan] Final USERD:')
  for name, off in [('Put', 0x40), ('Get', 0x44), ('Ref', 0x48), ('PutHi', 0x4c),
                    ('SetRef', 0x50), ('TopGet', 0x58), ('TopGetHi', 0x5c),
                    ('GetHi', 0x60), ('GPGet', 0x88), ('GPPut', 0x8c)]:
    print(f'  {name:10s} [0x{off:02x}] = 0x{dev.read32(userd + off):08x}')

  # DMA pusher error registers
  if fifo_intr & 0x1000:
    print('[chan] DMA_PUSHER error!')
    dma_get = dev.read32(0x003244)
    dma_put = dev.read32(0x003240)
    push_state = dev.read32(0x003220)
    dma_state = dev.read32(0x003228)
    ho_get = dev.read32(0x003328)
    ho_put = dev.read32(0x003320)
    ib_get = dev.read32(0x003334)
    ib_put = dev.read32(0x003330)
    err = ['NONE', 'CALL_SUBR', 'INVALID_MTHD', 'RET_SUBR',
           'INVALID_CMD', 'IB_EMPTY', 'MEM_FAULT', 'UNK'][(dma_state >> 29) & 7]
    print(f'  DMA_GET=0x{dma_get:08x} DMA_PUT=0x{dma_put:08x}')
    print(f'  HO_GET=0x{ho_get:08x} HO_PUT=0x{ho_put:08x}')
    print(f'  IB_GET=0x{ib_get:08x} IB_PUT=0x{ib_put:08x}')
    print(f'  DMA_STATE=0x{dma_state:08x} (err: {err})')
    print(f'  PUSH=0x{push_state:08x}')

  return chan_id

def main():
  dev = Dev()
  try:
    # Enable MSE
    cmd_reg = dev.cfg(0x04, 2)
    if not (cmd_reg & 0x0002):
      dev._rpc(RemoteCmd.CFG_WRITE, 0, 0x04, 2, payload=struct.pack('<H', cmd_reg | 0x0002))
    dev.map_bar(0)
    dev.write32(0x000200, 0xffffffff)
    print(f'PMC_ENABLE = 0x{dev.read32(0x000200):08x}')

    # VBIOS devinit
    with open(os.path.join(os.path.dirname(__file__), 'en210.rom'), 'rb') as f:
      image = f.read()
    scripts = find_vbios_scripts(image)
    for s in scripts:
      run_vbios_init(dev, image, scripts=[s], debug=False)
    print('Devinit complete.')

    # Verify VRAM
    vram = VRAM(dev)
    vram.write32(0x100000, 0xDEADBEEF)
    v = vram.read32(0x100000)
    print(f'VRAM test: {"PASS" if v == 0xDEADBEEF else "FAIL"} (0x{v:08x})')

    # FIFO init
    fifo_init(dev)

    # Test: SET_REFERENCE
    print('\n=== Test: SET_REFERENCE ===')
    # NV506F_SET_REFERENCE = 0x0050, subchannel 0
    test_ref = 0xDEAD
    push = nvm50(0, 0x0050, test_ref, inc=True)
    print(f'Push words: {[hex(w) for w in push]}')
    create_channel(dev, vram, push_words=push)

    # Check if reference was set
    userd = USERD_BASE + CHAN_ID * USERD_SIZE
    ref = dev.read32(userd + 0x48)
    print(f'\nReference = 0x{ref:08x} (expected 0x{test_ref:08x})')
    if ref == test_ref:
      print('PASS: SET_REFERENCE executed!')
    else:
      print('FAIL: Reference not set')

    # GR init (with ctxprog)
    print('\n=== GR Init ===')
    try:
      # Load generated ctxprog
      ctxprog_path = os.path.join(os.path.dirname(__file__), 'ctxprog.py')
      if os.path.exists(ctxprog_path):
        ctxprog_globals = {}
        with open(ctxprog_path) as f:
          exec(f.read(), ctxprog_globals)
        ctxprog = ctxprog_globals['ctxprog']
        ctxvals_size = ctxprog_globals['ctxvals_size']
        print(f'[gr] Loaded ctxprog: {len(ctxprog)} instructions, ctxvals_size=0x{ctxvals_size:x}')

        # Load ctxvals binary
        ctxvals_path = os.path.join(os.path.dirname(__file__), 'ctxvals.bin')
        with open(ctxvals_path, 'rb') as f:
          ctxvals = f.read()
        print(f'[gr] Loaded ctxvals: {len(ctxvals)} bytes')

        gr_init(dev, ctxprog=ctxprog, ctxvals=ctxvals, ctxvals_size=ctxvals_size)
      else:
        print('[gr] No ctxprog.py found, running register-only init')
        gr_init(dev)
    except Exception as e:
      import traceback
      print(f'GR init error: {e}')
      traceback.print_exc()

  finally:
    dev.close()

if __name__ == '__main__':
  main()
