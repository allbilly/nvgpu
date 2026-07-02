#!/usr/bin/env python3
import array, ctypes, functools, hashlib, os, pathlib, struct, sys, time, traceback
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "tinygrad"))
from tinygrad.device import Device
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP, NVRpcQueue
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import APLRemotePCIDevice

METHOD_NAMES = {
  0x005c: "NVC56F_SEM_ADDR_LO",
  0x02b4: "NVC6C0_SEND_PCAS_A",
  0x02c0: "NVC6C0_SEND_SIGNALING_PCAS2_B",
  0x1698: "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
  0x0020: "NVC56F_NON_STALL_INTERRUPT",
}

NV_PGSP_FALCON_ENGINE = 0x1103C0
NV_PSEC_FALCON_ENGINE = 0x8403C0
NV_FALCON_GSP_BASE = 0x110000
NV_FALCON_SEC2_BASE = 0x840000
NV_PFALCON_FALCON_OS = 0x80
NV_PFALCON_FALCON_RM = 0x84
NV_PFALCON_FALCON_HWCFG2 = 0xF4
NV_PFALCON_FALCON_CPUCTL = 0x100
NV_PFALCON_FALCON_DMACTL = 0x10C
NV_PFALCON_FALCON_DMATRFBASE = 0x110
NV_PFALCON_FALCON_DMATRFMOFFS = 0x114
NV_PFALCON_FALCON_DMATRFCMD = 0x118
NV_PFALCON_FALCON_DMATRFFBOFFS = 0x11C
NV_PFALCON_FALCON_DMATRFBASE1 = 0x128
NV_PFALCON_FALCON_EXCI = 0x18C
NV_PFALCON_FBIF_TRANSCFG0 = 0x600
NV_PFALCON_FBIF_CTL = 0x624
NV_PFALCON_FALCON_IRQSTAT = 0x650
NV_PRISCV_RISCV_CPUCTL = 0x1388
NV_PRISCV_RISCV_BCR_CTRL = 0x1668
NV_PBUS_BAR1_BLOCK = 0x1704
NV_PFB_PRI_MMU_WPR2_ADDR_HI = 0x1FA828

TINY_GSP_RPC_NAMES = {
  4: "ALLOC_MEMORY",
  47: "UNLOADING_GUEST_DRIVER",
  54: "SET_PAGE_DIRECTORY",
  72: "GSP_SET_SYSTEM_INFO",
  73: "SET_REGISTRY",
  76: "GSP_RM_CONTROL",
  103: "GSP_RM_ALLOC",
  4097: "EVENT_GSP_INIT_DONE",
  4098: "EVENT_GSP_RUN_CPU_SEQUENCER",
  4101: "EVENT_MMU_FAULT_QUEUED",
  4102: "EVENT_OS_ERROR_LOG",
  4128: "EVENT_GSP_POST_NOCAT_RECORD",
  0x80000000: "CONTINUATION_RECORD",
}

def _tiny_gsp_rpc_name(func):
  return TINY_GSP_RPC_NAMES.get(func, f"UNKNOWN_{func}")

def printable_c_strings(data, min_len=3, limit=12):
  strings, current = [], []
  for byte in bytes(data):
    if 32 <= byte < 127:
      current.append(chr(byte))
    else:
      if len(current) >= min_len:
        strings.append("".join(current))
        if len(strings) >= limit: return strings
      current = []
  if len(current) >= min_len and len(strings) < limit:
    strings.append("".join(current))
  return strings

def decode_post_nocat_record(msg):
  data = bytes(msg)
  qwords = []
  for off in range(0, min(len(data), 24), 8):
    if off + 8 <= len(data):
      qwords.append(f"0x{struct.unpack_from('<Q', data, off)[0]:x}")
  return {
    "qwords": qwords,
    "kind": f"0x{struct.unpack_from('<Q', data, 16)[0]:x}" if len(data) >= 24 else None,
    "strings": printable_c_strings(data[24:] if len(data) > 24 else data),
  }

def format_post_nocat_record_decode(msg):
  info = decode_post_nocat_record(msg)
  return (f"qwords={','.join(info['qwords']) if info['qwords'] else 'missing'} "
          f"kind={info['kind'] or 'missing'} "
          f"strings={ '|'.join(info['strings']) if info['strings'] else 'missing'}")

FALCON_WRITE_NAMES = {
  NV_PGSP_FALCON_ENGINE: "GSP_ENGINE",
  NV_PSEC_FALCON_ENGINE: "SEC2_ENGINE",
  NV_PFALCON_FALCON_OS: "OS",
  NV_PFALCON_FALCON_RM: "RM",
  NV_PFALCON_FALCON_CPUCTL: "CPUCTL",
  NV_PFALCON_FALCON_DMACTL: "DMACTL",
  NV_PFALCON_FALCON_DMATRFBASE: "DMATRFBASE",
  NV_PFALCON_FALCON_DMATRFMOFFS: "DMATRFMOFFS",
  NV_PFALCON_FALCON_DMATRFCMD: "DMATRFCMD",
  NV_PFALCON_FALCON_DMATRFFBOFFS: "DMATRFFBOFFS",
  NV_PFALCON_FALCON_DMATRFBASE1: "DMATRFBASE1",
  NV_PFALCON_FBIF_TRANSCFG0: "FBIF_TRANSCFG0",
  NV_PFALCON_FBIF_CTL: "FBIF_CTL",
  NV_PRISCV_RISCV_CPUCTL: "RISCV_CPUCTL",
  NV_PRISCV_RISCV_BCR_CTRL: "RISCV_BCR_CTRL",
}

def _trace_enabled():
  return os.environ.get("NV_ADD_TINY_TRACE", "1") != "0"

def _trace_stack_enabled():
  return os.environ.get("NV_ADD_TINY_TRACE_STACK") == "1"

def cli_arg_value(flag):
  if flag not in sys.argv: return None
  index = sys.argv.index(flag)
  if index + 1 >= len(sys.argv):
    print(f"cli_arg_error kind=missing-value flag={flag}")
    raise SystemExit(2)
  return sys.argv[index + 1]

def recommended_tiny_trace_command(script="examples/add_tiny.py"):
  flags = ["NV_ADD_TINY_TRACE=1", "NV_ADD_TINY_TRACE_STACK=1", "NV_ADD_TINY_BOOT_VALUES=1"]
  return " ".join(flags + ["python3", script])

def recommended_standalone_golden_command(script="examples/add.py"):
  flags = [
    "NV_ADD_TRANSPORT=mac-egpu", "NV_ADD_PREPARE_GOLDEN_CTX=1", "NV_ADD_BOOT_GSP=1",
    "NV_ADD_SUMMARY=1", "NV_ADD_CHECK_FRTS_BAR1=1", "NV_ADD_TRACE_GSP_BOOT=1",
    "NV_ADD_VERIFY_SEC2_INPUTS=1", "NV_ADD_TRACE_RM_ALLOC=1", "NV_ADD_TRACE_RM_STATE=1",
    "NV_ADD_TRACE_RPC=1", "NV_ADD_TRACE_RPC_READ=1", "NV_ADD_TRACE_CHANNEL=1",
    "NV_ADD_TRACE_MM_ALLOC=1", "NV_ADD_TRACE_LAUNCH_STEPS=1",
  ]
  return " ".join(flags + ["python3", script])

def recommended_standalone_stack_command(script="examples/add.py"):
  flags = [
    "NV_ADD_TRACE_RM_STACK=1", "NV_ADD_TRACE_CHANNEL_STACK=1", "NV_ADD_TRACE_LAUNCH_STACK=1",
    "NV_ADD_TRACE_FALCON=1",
  ]
  return " ".join(flags + [recommended_standalone_golden_command(script)])

def tiny_live_log_workflow_lines(script="examples/add_tiny.py", standalone_script="examples/add.py",
                                 tiny_log="tiny-golden.log", standalone_log="standalone-golden.log"):
  return [
    f"live_log_workflow script={script} standalone_script={standalone_script} standalone_log={standalone_log} tiny_log={tiny_log}",
    f"gate_command NV_ADD_TRANSPORT=mac-egpu python3 {standalone_script} --transport-preflight-plan --require-ready",
    f"tiny_bar_info_command python3 {script} --bar-info",
    f"standalone_log_command {recommended_standalone_golden_command(standalone_script)} 2>&1 | tee {standalone_log}",
    f"tiny_log_command {recommended_tiny_trace_command(script)} 2>&1 | tee {tiny_log}",
    f"compare_command python3 {standalone_script} --compare-trace-logs --standalone-log {standalone_log} --tiny-log {tiny_log}",
    "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence",
    "workflow_rule run standalone_log_command only after gate result is ready-for-gsp",
    "workflow_rule run tiny_log_command in the same eGPU session if standalone stalls or times out",
  ]

def tiny_live_stack_log_workflow_lines(script="examples/add_tiny.py", standalone_script="examples/add.py",
                                       tiny_log="tiny-stack.log", standalone_log="standalone-stack.log"):
  return [
    f"live_stack_log_workflow script={script} standalone_script={standalone_script} standalone_log={standalone_log} tiny_log={tiny_log}",
    f"gate_command NV_ADD_TRANSPORT=mac-egpu python3 {standalone_script} --transport-preflight-plan --require-ready",
    f"tiny_bar_info_command python3 {script} --bar-info",
    f"standalone_log_command {recommended_standalone_stack_command(standalone_script)} 2>&1 | tee {standalone_log}",
    f"tiny_log_command {recommended_tiny_trace_command(script)} 2>&1 | tee {tiny_log}",
    f"compare_command python3 {standalone_script} --compare-trace-logs --standalone-log {standalone_log} --tiny-log {tiny_log}",
    "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence",
    "workflow_check stack inspect trace_log_compare_stack, trace_log_compare_falcon",
    "workflow_rule use this only when Python call-path stacks are needed; it is verbose",
    "workflow_rule run standalone_log_command only after gate result is ready-for-gsp",
    "workflow_rule run tiny_log_command in the same eGPU session if standalone stalls or times out",
  ]

def print_tiny_trace_command(script="examples/add_tiny.py"):
  print(f"tiny_trace_command {recommended_tiny_trace_command(script)}")

def print_tiny_live_log_workflow(script="examples/add_tiny.py", standalone_script="examples/add.py",
                                 tiny_log="tiny-golden.log", standalone_log="standalone-golden.log"):
  for line in tiny_live_log_workflow_lines(script, standalone_script, tiny_log, standalone_log):
    print(line)

def print_tiny_live_stack_log_workflow(script="examples/add_tiny.py", standalone_script="examples/add.py",
                                       tiny_log="tiny-stack.log", standalone_log="standalone-stack.log"):
  for line in tiny_live_stack_log_workflow_lines(script, standalone_script, tiny_log, standalone_log):
    print(line)

def print_tiny_debug_help(script="examples/add_tiny.py", standalone_script="examples/add.py"):
  print(f"tiny_debug_help script={script}")
  print(f"bar_info python3 {script} --bar-info")
  print(f"trace_command python3 {script} --trace-command")
  print(f"trace_selftest python3 {script} --trace-selftest")
  print(f"live_log_workflow python3 {script} --live-log-workflow --standalone-script {standalone_script}")
  print(f"live_stack_log_workflow python3 {script} --live-stack-log-workflow --standalone-script {standalone_script}")
  print(f"compare_trace_logs python3 {standalone_script} --compare-trace-logs --standalone-log standalone-golden.log --tiny-log tiny-golden.log")
  print(f"compare_stack_trace_logs python3 {standalone_script} --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log")
  print(f"tiny_trace {recommended_tiny_trace_command(script)}")

def _tiny_symbol_name(value, prefixes):
  matches = []
  for name, symbol_value in vars(nv_gpu).items():
    if not any(name.startswith(prefix) for prefix in prefixes): continue
    if isinstance(symbol_value, int) and symbol_value == value: matches.append(name)
  return matches[0] if matches else f"0x{value:x}"

def _tiny_class_name(value):
  return _tiny_symbol_name(value, ("NV", "FERMI_", "KEPLER_", "AMPERE_", "ADA_", "BLACKWELL_", "GT200_"))

def _tiny_control_name(value):
  return _tiny_symbol_name(value, ("NV", "NVA", "NVB", "NVC"))

def _tiny_params_bytes(params):
  return b"" if params is None else bytes(params)

def _tiny_pack_rpc_rm_alloc(h_client, h_parent, h_object, h_class, params=b"", flags=0):
  params = _tiny_params_bytes(params)
  return struct.pack("<IIIIIII4x", h_client, h_parent, h_object, h_class, 0, len(params), flags) + params

def _tiny_pack_rpc_rm_control(h_client, h_object, cmd, params=b"", flags=0):
  params = _tiny_params_bytes(params)
  return struct.pack("<IIIIII", h_client, h_object, cmd, 0, len(params), flags) + params

def _tiny_memdesc(data, offset):
  if len(data) < offset + 24: return None
  base, size, address_space, cache_attrib = struct.unpack_from("<QQII", data, offset)
  return base, size, address_space, cache_attrib

def _tiny_gpfifo_desc_text(params_bytes):
  if len(params_bytes) >= 248:
    gpfifo_va, entries, flags = struct.unpack_from("<QII", params_bytes, 8)
    h_context_share, h_vaspace = struct.unpack_from("<II", params_bytes, 24)
    h_userd_memory = struct.unpack_from("<I", params_bytes, 32)[0]
    userd_offset = struct.unpack_from("<Q", params_bytes, 64)[0]
    engine_type, cid, runlist_id = struct.unpack_from("<III", params_bytes, 128)
    internal_flags = struct.unpack_from("<I", params_bytes, 244)[0]
    scalar = (f"gpfifo_va=0x{gpfifo_va:x} entries={entries} flags=0x{flags:x} "
              f"h_context_share=0x{h_context_share:x} h_vaspace=0x{h_vaspace:x} "
              f"h_userd_memory=0x{h_userd_memory:x} userd_offset=0x{userd_offset:x} "
              f"engine_type=0x{engine_type:x} cid={cid} runlist_id={runlist_id} "
              f"internal_flags=0x{internal_flags:x}")
  else:
    scalar = "gpfifo_scalars=truncated"
  fields = (("ramfc", 144), ("userd", 168), ("instance", 192), ("method", 216), ("error", 248))
  parts = [scalar]
  for name, offset in fields:
    desc = _tiny_memdesc(params_bytes, offset)
    if desc is None:
      parts.append(f"{name}=truncated")
    else:
      base, size, address_space, cache_attrib = desc
      parts.append(f"{name}=0x{base:x}/0x{size:x}/as{address_space}/ca{cache_attrib}")
  return " ".join(parts)

def _tiny_stack(label):
  if _trace_stack_enabled():
    stack = "".join(traceback.format_stack(limit=8)[:-1]).replace("\n", "\\n")
    print(f"tiny {label}_stack {stack}", flush=True)

def _falcon_state(nvdev, base):
  engine_reg = NV_PGSP_FALCON_ENGINE if base == NV_FALCON_GSP_BASE else NV_PSEC_FALCON_ENGINE
  items = {
    "engine": nvdev.rreg(engine_reg),
    "cpuctl": nvdev.rreg(base + NV_PFALCON_FALCON_CPUCTL),
    "dmactl": nvdev.rreg(base + NV_PFALCON_FALCON_DMACTL),
    "dmatrfcmd": nvdev.rreg(base + NV_PFALCON_FALCON_DMATRFCMD),
    "dmatrfbase": nvdev.rreg(base + NV_PFALCON_FALCON_DMATRFBASE),
    "dmatrfbase1": nvdev.rreg(base + NV_PFALCON_FALCON_DMATRFBASE1),
    "dmatrfmoffs": nvdev.rreg(base + NV_PFALCON_FALCON_DMATRFMOFFS),
    "dmatrffboffs": nvdev.rreg(base + NV_PFALCON_FALCON_DMATRFFBOFFS),
    "hwcfg2": nvdev.rreg(base + NV_PFALCON_FALCON_HWCFG2),
    "fbif_ctl": nvdev.rreg(base + NV_PFALCON_FBIF_CTL),
    "fbif_transcfg0": nvdev.rreg(base + NV_PFALCON_FBIF_TRANSCFG0),
    "exci": nvdev.rreg(base + NV_PFALCON_FALCON_EXCI),
    "irqstat": nvdev.rreg(base + NV_PFALCON_FALCON_IRQSTAT),
    "riscv_bcr": nvdev.rreg(base + NV_PRISCV_RISCV_BCR_CTRL),
    "riscv_cpuctl": nvdev.rreg(base + NV_PRISCV_RISCV_CPUCTL),
    "os": nvdev.rreg(base + NV_PFALCON_FALCON_OS),
    "rm": nvdev.rreg(base + NV_PFALCON_FALCON_RM),
    "wpr2_hi": nvdev.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI),
  }
  return " ".join(f"{name}=0x{value:x}" for name, value in items.items())

def _tiny_rm_state_text(nvdev):
  return (f"bar1=0x{nvdev.rreg(NV_PBUS_BAR1_BLOCK):x} "
          f"wpr2_hi=0x{nvdev.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI):x} "
          f"gsp=({_falcon_state(nvdev, NV_FALCON_GSP_BASE)}) "
          f"sec2=({_falcon_state(nvdev, NV_FALCON_SEC2_BASE)})")

def _queue_dump(label, queue):
  if not hasattr(queue, "tx"): return f"{label}=uninitialized"
  tx_view = queue.view.view(fmt='I')
  parts = [f"{label}_tx=({queue.tx.version},{queue.tx.size},{queue.tx.msgSize},{queue.tx.msgCount},"
           f"{tx_view[4]},{queue.tx.flags},{queue.tx.rxHdrOff},{queue.tx.entryOff})"]
  rx_view = getattr(queue, "rx_view", None)
  if rx_view is not None: parts.append(f"{label}_rx={rx_view[0]}")
  for slot in range(min(4, queue.tx.msgCount)):
    off = slot * queue.tx.msgSize
    elem = bytes(queue.queue_mv[off:off + 0x30])
    hdr = bytes(queue.queue_mv[off + 0x30:off + 0x50])
    checksum, seq, elem_count, padding = struct.unpack_from("<IIII", elem, 32)
    header_version, signature, length, function, result, private, sequence, union_value = struct.unpack_from("<IIIIIIII", hdr)
    parts.append(f"{label}_slot{slot}: checksum=0x{checksum:x} seq={seq} elem_count={elem_count} "
                 f"func={function} func_name={_tiny_gsp_rpc_name(function)} len={length} "
                 f"result=0x{result:x} private=0x{private:x} sig=0x{signature:x}")
  return "; ".join(parts)

def _gsp_queue_dump(gsp):
  parts = []
  if hasattr(gsp, "cmd_q"): parts.append(_queue_dump("cmd", gsp.cmd_q))
  if hasattr(gsp, "stat_q"): parts.append(_queue_dump("stat", gsp.stat_q))
  return " ".join(parts) if parts else "queues=unavailable"

def _tiny_mapping_paddrs_text(mapping):
  try:
    return ",".join(f"0x{paddr:x}/0x{span:x}" for paddr, span in mapping.paddrs[:4])
  except Exception:
    return "unavailable"

def install_tinygrad_falcon_trace():
  if getattr(NV_FLCN, "_add_tiny_trace_installed", False): return
  NV_FLCN._add_tiny_trace_installed = True

  old_palloc = NVMemoryManager.palloc
  @functools.wraps(old_palloc)
  def traced_palloc(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    result = old_palloc(self, size, align=align, zero=zero, boot=boot, ptable=ptable)
    if _trace_enabled() and os.environ.get("NV_ADD_TINY_TRACE_MM_ALLOC", "1") != "0":
      print(f"tiny mm palloc size=0x{((int(size) + 0xfff) & ~0xfff):x} align=0x{int(align):x} "
            f"zero={zero} boot={boot} ptable={ptable} -> 0x{result:x}", flush=True)
    return result
  NVMemoryManager.palloc = traced_palloc

  old_valloc = NVMemoryManager.valloc
  @functools.wraps(old_valloc)
  def traced_valloc(self, size, align=0x1000, uncached=False, contiguous=False):
    result = old_valloc(self, size, align=align, uncached=uncached, contiguous=contiguous)
    if _trace_enabled() and os.environ.get("NV_ADD_TINY_TRACE_MM_ALLOC", "1") != "0":
      print(f"tiny mm valloc size=0x{((int(size) + 0xfff) & ~0xfff):x} align=0x{int(align):x} "
            f"contiguous={contiguous} uncached={uncached} -> va=0x{result.va_addr:x} "
            f"paddrs={_tiny_mapping_paddrs_text(result)}", flush=True)
    return result
  NVMemoryManager.valloc = traced_valloc

  old_wreg = NVDev.wreg
  def traced_wreg(self, addr, value):
    if _trace_enabled():
      for base, base_name in ((NV_FALCON_GSP_BASE, "GSP"), (NV_FALCON_SEC2_BASE, "SEC2")):
        off = addr - base
        if off in FALCON_WRITE_NAMES:
          print(f"tiny wreg {base_name}.{FALCON_WRITE_NAMES[off]} addr=0x{addr:x} value=0x{value:x}", flush=True)
          break
      else:
        if addr in FALCON_WRITE_NAMES:
          print(f"tiny wreg {FALCON_WRITE_NAMES[addr]} addr=0x{addr:x} value=0x{value:x}", flush=True)
    return old_wreg(self, addr, value)
  NVDev.wreg = traced_wreg

  old_send_rpc = NVRpcQueue.send_rpc
  def traced_send_rpc(self, func, msg):
    if _trace_enabled():
      print(f"tiny send_rpc func={func} func_name={_tiny_gsp_rpc_name(func)} len={len(msg)} "
            f"sha256={hashlib.sha256(msg).hexdigest()} head={bytes(msg[:128]).hex()}", flush=True)
    return old_send_rpc(self, func, msg)
  NVRpcQueue.send_rpc = traced_send_rpc

  old_read_resp = NVRpcQueue.read_resp
  def traced_read_resp(self):
    traced = []
    traced_yields = {}
    traced_by_func = {}
    if _trace_enabled():
      try:
        rp, wp = self.rx_view[0], self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4]
        cur = rp
        while cur != wp:
          off = cur * self.tx.msgSize
          hdr = nv.rpc_message_header_v.from_buffer_copy(
            bytes(self.queue_mv[off + 0x30 : off + 0x30 + ctypes.sizeof(nv.rpc_message_header_v)]))
          msg = bytes(self.queue_mv[off + 0x50 : off + 0x50 + hdr.length])
          advance = (hdr.length + self.tx.msgSize - 1) // self.tx.msgSize
          traced.append({"rp": cur, "wp": wp, "advance": advance, "function": hdr.function, "length": hdr.length,
                         "result": hdr.rpc_result, "private": hdr.rpc_result_private, "msg": msg})
          traced_yields[(hdr.function, hashlib.sha256(msg).hexdigest())] = traced[-1]
          traced_by_func.setdefault(hdr.function, []).append(traced[-1])
          cur = (cur + advance) % self.tx.msgCount
      except Exception:
        traced = []
      for meta in traced:
        msg = meta["msg"]
        extra = (f" post_nocat={format_post_nocat_record_decode(msg)}"
                 if meta["function"] == 4128 else "")
        print(f"tiny read_rpc rp={meta['rp']} wp={meta['wp']} advance={meta['advance']} "
              f"func={meta['function']} func_name={_tiny_gsp_rpc_name(meta['function'])} len={meta['length']} "
              f"result=0x{meta['result']:x} private=0x{meta['private']:x} "
              f"sha256={hashlib.sha256(msg).hexdigest()}{extra} head={bytes(msg[:128]).hex()}", flush=True)
    for func, msg in old_read_resp(self):
      if _trace_enabled():
        msg_sha = hashlib.sha256(msg).hexdigest()
        meta = traced_yields.get((func, msg_sha))
        if meta is None and len(traced_by_func.get(func, ())) == 1:
          meta = traced_by_func[func][0]
        extra = (f" post_nocat={format_post_nocat_record_decode(msg)}"
                 if func == 4128 else "")
        if meta is None:
          print(f"tiny read_rpc_yield func={func} func_name={_tiny_gsp_rpc_name(func)} "
                f"sha256={msg_sha}{extra} head={bytes(msg[:128]).hex()}", flush=True)
        else:
          print(f"tiny read_rpc_yield rp={meta['rp']} wp={meta['wp']} advance={meta['advance']} "
                f"func={func} func_name={_tiny_gsp_rpc_name(func)} len={meta['length']} "
                f"result=0x{meta['result']:x} private=0x{meta['private']:x} "
                f"sha256={msg_sha}{extra} head={bytes(msg[:128]).hex()}", flush=True)
      yield func, msg
  NVRpcQueue.read_resp = traced_read_resp

  def wrap_gsp(name):
    old = getattr(NV_GSP, name)
    @functools.wraps(old)
    def traced(self, *args, **kwargs):
      result = old(self, *args, **kwargs)
      if _trace_enabled():
        if name == "init_rm_args":
          print(f"tiny queue rm_args=0x{self.rm_args_sysmem:x} cmd_head={bytes(self.cmd_q_view[:32]).hex()} queue_base=0x{self.cmd_q_view.off - self.cmd_q_view.off + self.cmd_q_view.off if hasattr(self.cmd_q_view, 'off') else 0:x}", flush=True)
        elif name == "init_libos_args":
          print(f"tiny libos_args=0x{self.libos_args_sysmem:x}", flush=True)
      return result
    setattr(NV_GSP, name, traced)

  for name in ("init_rm_args", "init_libos_args"):
    wrap_gsp(name)

  old_promote_ctx = NV_GSP.promote_ctx
  @functools.wraps(old_promote_ctx)
  def traced_promote_ctx(self, client, subdevice, obj, ctxbufs, bufs=None, virt=None, phys=None):
    result = old_promote_ctx(self, client, subdevice, obj, ctxbufs, bufs=bufs, virt=virt, phys=phys)
    if _trace_enabled():
      entries = []
      for buffer_id, desc in ctxbufs.items():
        use_v, use_p = (desc.virt if virt is None else virt), (desc.phys if phys is None else phys)
        mapping = result[buffer_id]
        entries.append((mapping.paddrs[0][0] if use_p else 0, mapping.va_addr if use_v else 0,
                        desc.size if use_p else 0, 0x4 if use_p else 0, buffer_id, use_p, use_p and not use_v))
      entry_text = ";".join(
        f"id={buffer_id}:phys=0x{gpu_phys_addr:x}:virt=0x{gpu_virt_addr:x}:size=0x{entry_size:x}:"
        f"attr=0x{phys_attr:x}:init={int(initialize)}:nonmapped={int(nonmapped)}"
        for gpu_phys_addr, gpu_virt_addr, entry_size, phys_attr, buffer_id, initialize, nonmapped in entries)
      payload = b"".join(struct.pack("<QQQI HBB", *entry) for entry in entries)
      print(f"tiny promote_ctx_payload client=0x{client:x} subdevice=0x{subdevice:x} object=0x{obj:x} "
            f"virt={'default' if virt is None else virt} phys={'default' if phys is None else phys} "
            f"entries={len(entries)} ids={[entry[4] for entry in entries]} "
            f"entries_sha256={hashlib.sha256(repr(entries).encode()).hexdigest()} "
            f"packed_entries_sha256={hashlib.sha256(payload).hexdigest()} entry_text={entry_text}", flush=True)
    return result
  NV_GSP.promote_ctx = traced_promote_ctx

  old_init_golden_image = NV_GSP.init_golden_image
  @functools.wraps(old_init_golden_image)
  def traced_init_golden_image(self, *args, **kwargs):
    if _trace_enabled():
      print(f"tiny golden_start priv_root=0x{getattr(self, 'priv_root', 0):x} "
            f"gpfifo_class=0x{getattr(self, 'gpfifo_class', 0):x} "
            f"compute_class=0x{getattr(self, 'compute_class', 0):x} "
            f"dma_class=0x{getattr(self, 'dma_class', 0):x}", flush=True)
      _tiny_stack("golden_start")
    result = old_init_golden_image(self, *args, **kwargs)
    if _trace_enabled():
      print(f"tiny golden_done grctx_ids={sorted(getattr(self, 'grctx_bufs', {}).keys())}", flush=True)
    return result
  NV_GSP.init_golden_image = traced_init_golden_image

  old_rpc_rm_alloc = NV_GSP.rpc_rm_alloc
  @functools.wraps(old_rpc_rm_alloc)
  def traced_rpc_rm_alloc(self, *args, **kwargs):
    h_parent = kwargs.get("hParent", args[0] if len(args) > 0 else 0)
    h_class = kwargs.get("hClass", args[1] if len(args) > 1 else 0)
    params = kwargs.get("params", args[2] if len(args) > 2 else None)
    client = kwargs.get("client", None)
    if _trace_enabled():
      params_bytes = _tiny_params_bytes(params)
      print(f"tiny rm_alloc pre client=0x{(client or self.priv_root):x} parent=0x{h_parent:x} "
            f"class=0x{h_class:x} class_name={_tiny_class_name(h_class)} params_len={len(params_bytes)} "
            f"params_sha256={hashlib.sha256(params_bytes).hexdigest()} head={params_bytes[:128].hex()}", flush=True)
      if hasattr(self, "nvdev"):
        print(f"tiny rm_alloc pre_state parent=0x{h_parent:x} class=0x{h_class:x} "
              f"class_name={_tiny_class_name(h_class)} {_tiny_rm_state_text(self.nvdev)}", flush=True)
      print(f"tiny rm_alloc pre_queues parent=0x{h_parent:x} class=0x{h_class:x} {_gsp_queue_dump(self)}", flush=True)
      _tiny_stack("rm_alloc")
    if _trace_enabled() and hasattr(self, "nvdev") and not getattr(self, "_add_tiny_pre_root_printed", False):
      self._add_tiny_pre_root_printed = True
      print(f"tiny pre-root {_tiny_rm_state_text(self.nvdev)}", flush=True)
      print(f"tiny pre-root queues {_gsp_queue_dump(self)}", flush=True)
    try:
      result = old_rpc_rm_alloc(self, *args, **kwargs)
    except Exception as exc:
      if _trace_enabled():
        print(f"tiny rm_alloc exception parent=0x{h_parent:x} class=0x{h_class:x} "
              f"class_name={_tiny_class_name(h_class)} exc={type(exc).__name__} msg={str(exc)} "
              f"{_gsp_queue_dump(self)}", flush=True)
        if hasattr(self, "nvdev"):
          print(f"tiny rm_alloc exception_state bar1=0x{self.nvdev.rreg(NV_PBUS_BAR1_BLOCK):x} "
                f"wpr2_hi=0x{self.nvdev.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI):x} "
                f"gsp=({_falcon_state(self.nvdev, NV_FALCON_GSP_BASE)}) "
                f"sec2=({_falcon_state(self.nvdev, NV_FALCON_SEC2_BASE)})", flush=True)
      raise
    if _trace_enabled():
      params_bytes = _tiny_params_bytes(params)
      client_handle = client or self.priv_root
      rpc_sha256 = hashlib.sha256(_tiny_pack_rpc_rm_alloc(client_handle, h_parent, result, h_class, params_bytes)).hexdigest()
      print(f"tiny rm_alloc post object=0x{result:x} class=0x{h_class:x} class_name={_tiny_class_name(h_class)} "
            f"params_len={len(params_bytes)} params_sha256={hashlib.sha256(params_bytes).hexdigest()} "
            f"rpc_sha256={rpc_sha256} head={params_bytes[:128].hex()}", flush=True)
      print(f"tiny rm_alloc post_queues object=0x{result:x} class=0x{h_class:x} {_gsp_queue_dump(self)}", flush=True)
      if h_class == getattr(self, "gpfifo_class", None):
        print(f"tiny gpfifo_patch post object=0x{result:x} "
              f"params_sha256={hashlib.sha256(params_bytes).hexdigest()} {_tiny_gpfifo_desc_text(params_bytes)}", flush=True)
      if h_class == getattr(self, "compute_class", None):
        print(f"tiny compute_alloc parent=0x{h_parent:x} object=0x{result:x} compute_class=0x{h_class:x} "
              f"rpc_sha256={rpc_sha256}", flush=True)
        _tiny_stack("compute_alloc")
      if h_class == getattr(self, "dma_class", None):
        print(f"tiny dma_alloc parent=0x{h_parent:x} object=0x{result:x} dma_class=0x{h_class:x} "
              f"rpc_sha256={rpc_sha256}", flush=True)
        _tiny_stack("dma_alloc")
    if _trace_enabled() and getattr(self, "_add_tiny_pre_root_printed", False) and not getattr(self, "_add_tiny_post_root_printed", False):
      self._add_tiny_post_root_printed = True
      print(f"tiny post-root queues {_gsp_queue_dump(self)}", flush=True)
    return result
  NV_GSP.rpc_rm_alloc = traced_rpc_rm_alloc

  old_rpc_rm_control = NV_GSP.rpc_rm_control
  @functools.wraps(old_rpc_rm_control)
  def traced_rpc_rm_control(self, *args, **kwargs):
    h_object = kwargs.get("hObject", args[0] if len(args) > 0 else 0)
    cmd = kwargs.get("cmd", args[1] if len(args) > 1 else 0)
    params = kwargs.get("params", args[2] if len(args) > 2 else None)
    client = kwargs.get("client", None)
    if _trace_enabled():
      params_bytes = _tiny_params_bytes(params)
      print(f"tiny rm_control pre client=0x{(client or self.priv_root):x} object=0x{h_object:x} "
            f"cmd=0x{cmd:x} cmd_name={_tiny_control_name(cmd)} params_len={len(params_bytes)} "
            f"params_sha256={hashlib.sha256(params_bytes).hexdigest()} head={params_bytes[:128].hex()}", flush=True)
      print(f"tiny rm_control pre_queues object=0x{h_object:x} cmd=0x{cmd:x} "
            f"cmd_name={_tiny_control_name(cmd)} {_gsp_queue_dump(self)}", flush=True)
      _tiny_stack("rm_control")
    result = old_rpc_rm_control(self, *args, **kwargs)
    if _trace_enabled():
      params_bytes = _tiny_params_bytes(params)
      result_bytes = _tiny_params_bytes(result)
      client_handle = client or self.priv_root
      rpc_sha256 = hashlib.sha256(_tiny_pack_rpc_rm_control(client_handle, h_object, cmd, params_bytes)).hexdigest()
      print(f"tiny rm_control post object=0x{h_object:x} cmd=0x{cmd:x} cmd_name={_tiny_control_name(cmd)} "
            f"rpc_sha256={rpc_sha256} result_len={len(result_bytes)} result_sha256={hashlib.sha256(result_bytes).hexdigest()} "
            f"head={result_bytes[:128].hex()}", flush=True)
      print(f"tiny rm_control post_queues object=0x{h_object:x} cmd=0x{cmd:x} "
            f"cmd_name={_tiny_control_name(cmd)} {_gsp_queue_dump(self)}", flush=True)
      cmd_name = _tiny_control_name(cmd)
      if "GET_WORK_SUBMIT_TOKEN" in cmd_name:
        print(f"tiny token_control object=0x{h_object:x} cmd=0x{cmd:x} rpc_sha256={rpc_sha256}", flush=True)
        _tiny_stack("token_control")
      if "GPFIFO_SCHEDULE" in cmd_name:
        print(f"tiny schedule_control object=0x{h_object:x} cmd=0x{cmd:x} rpc_sha256={rpc_sha256}", flush=True)
        _tiny_stack("schedule_control")
    return result
  NV_GSP.rpc_rm_control = traced_rpc_rm_control

  old_prep_booter = NV_FLCN.prep_booter
  @functools.wraps(old_prep_booter)
  def traced_prep_booter(self, *args, **kwargs):
    result = old_prep_booter(self, *args, **kwargs)
    if os.environ.get("NV_ADD_TINY_BOOT_VALUES") == "1":
      print(f"tiny booter img=0x{self.booter_image_paddr:x} code_off=0x{self.booter_code_off:x} "
            f"data_off=0x{self.booter_data_off:x} code_size=0x{self.booter_code_sz:x} data_size=0x{self.booter_data_sz:x} "
            f"sha256={hashlib.sha256(bytes(self.nvdev.vram.view(self.booter_image_paddr, self.booter_code_off + self.booter_code_sz + self.booter_data_sz, fmt='B')[:self.booter_code_off + self.booter_code_sz + self.booter_data_sz])).hexdigest()}",
            flush=True)
    return result
  NV_FLCN.prep_booter = traced_prep_booter

  old_init_wpr_meta = NV_GSP.init_wpr_meta
  @functools.wraps(old_init_wpr_meta)
  def traced_init_wpr_meta(self, *args, **kwargs):
    result = old_init_wpr_meta(self, *args, **kwargs)
    if os.environ.get("NV_ADD_TINY_BOOT_VALUES") == "1":
      meta = bytes(self.wpr_meta[:256])
      print(f"tiny wpr meta=0x{self.wpr_meta_sysmem:x} bootloader=0x{self.booter_bar1:x} "
            f"radix3=0x{self.gsp_radix3_addrs[0]:x} signature=0x{self.gsp_signature_bar1:x} "
            f"meta_sha256={hashlib.sha256(meta).hexdigest()}", flush=True)
      print(f"tiny wpr meta_hex={meta.hex()}", flush=True)
    return result
  NV_GSP.init_wpr_meta = traced_init_wpr_meta

  def wrap(name):
    old = getattr(NV_FLCN, name)
    @functools.wraps(old)
    def traced(self, *args, **kwargs):
      if _trace_enabled():
        print(f"tiny {name} args={args} kwargs={kwargs}", flush=True)
        if args and isinstance(args[0], int):
          if name == "execute_dma" and len(args) >= 6:
            base, cmd, dest, mem_off, src, size = args[:6]
            print(f"tiny {name} pre base=0x{base:x} cmd=0x{cmd:x} dest=0x{dest:x} "
                  f"mem_off=0x{mem_off:x} src=0x{src:x} size=0x{size:x} "
                  f"state=({_falcon_state(self.nvdev, base)})", flush=True)
          else:
            print(f"tiny {name} pre state=({_falcon_state(self.nvdev, args[0])})", flush=True)
      result = old(self, *args, **kwargs)
      if _trace_enabled():
        if args and isinstance(args[0], int):
          print(f"tiny {name} post state=({_falcon_state(self.nvdev, args[0])})", flush=True)
        else:
          print(f"tiny {name} result={result}", flush=True)
      return result
    setattr(NV_FLCN, name, traced)

  for name in ("reset", "disable_ctx_req", "execute_dma", "execute_hs", "start_cpu", "wait_cpu_halted", "init_hw"):
    wrap(name)

def open_pcie_device():
  install_tinygrad_falcon_trace()
  dev = Device["NV"]
  return dev

def print_tiny_bar_info():
  pcidev = APLRemotePCIDevice("NV", "usb4")
  bars = {idx: pcidev.bar_info(idx) for idx in (0, 1, 3)}
  pci_device = pcidev.read_config(0, 4)
  pci_subdevice = pcidev.read_config(0x2c, 4)
  pci_revision = pcidev.read_config(8, 1)
  print("tiny_bar_info "
        f"bar0=0x{bars[0][0]:x}/0x{bars[0][1]:x} "
        f"bar1=0x{bars[1][0]:x}/0x{bars[1][1]:x} "
        f"bar3=0x{bars[3][0]:x}/0x{bars[3][1]:x} "
        f"pci_device=0x{pci_device:x} pci_subdevice=0x{pci_subdevice:x} pci_revision=0x{pci_revision:x}")

def trace_selftest():
  import contextlib, io
  class FakeGsp:
    priv_root = 0xc1e00000
    compute_class = 0xc6c0
    def base_alloc(self, hParent, hClass, params, client=None): return 0xcf000001
    def base_control(self, hObject, cmd, params, client=None, extra=None): return params
  class FakeNvdev:
    def rreg(self, addr): return addr & 0xffff

  old_alloc, old_control = NV_GSP.rpc_rm_alloc, NV_GSP.rpc_rm_control
  old_init_golden_image = NV_GSP.init_golden_image
  old_installed = getattr(NV_FLCN, "_add_tiny_trace_installed", False)
  old_trace_stack = os.environ.get("NV_ADD_TINY_TRACE_STACK")
  old_send_rpc, old_read_resp = NVRpcQueue.send_rpc, NVRpcQueue.read_resp
  try:
    NV_GSP.rpc_rm_alloc = FakeGsp.base_alloc
    NV_GSP.rpc_rm_control = FakeGsp.base_control
    NV_FLCN._add_tiny_trace_installed = False
    os.environ["NV_ADD_TINY_TRACE_STACK"] = "1"
    install_tinygrad_falcon_trace()
    def traced_send_rpc_stub(self, func, msg):
      if _trace_enabled():
        print(f"tiny send_rpc func={func} func_name={_tiny_gsp_rpc_name(func)} len={len(msg)} "
              f"sha256={hashlib.sha256(msg).hexdigest()} head={bytes(msg[:128]).hex()}", flush=True)
    NVRpcQueue.send_rpc = traced_send_rpc_stub
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      fake = FakeGsp()
      fake.nvdev = FakeNvdev()
      def fake_checksum(data):
        if pad_len := (-len(data)) % 8: data += b"\x00" * pad_len
        checksum = 0
        for offset in range(0, len(data), 8):
          checksum ^= struct.unpack_from("Q", data, offset)[0]
        return (checksum >> 32) ^ (checksum & 0xffffffff)
      fake.cmd_q = fake_q = type("FakeQueue", (), {
        "tx": type("Tx", (), {"version": 0, "size": 0x5000, "msgSize": 0x1000, "msgCount": 4,
                              "flags": 1, "rxHdrOff": 0x20, "entryOff": 0x1000})(),
        "rx_view": [0],
        "_checksum": staticmethod(fake_checksum),
      })()
      fake_q.mem = bytearray(0x5000)
      fake_q.tx_view = [0, 0, 0, 0, 1]
      fake_q.view = type("View", (), {"view": lambda _self, fmt='I': fake_q.tx_view})()
      fake_q.queue_mv = memoryview(fake_q.mem)[fake_q.tx.entryOff:]
      def install_fake_response(queue, msg=b"ok", function=103, result=0, private=0):
        hdr = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, length=0x20 + len(msg),
          function=function, rpc_result=result, rpc_result_private=private)
        elem = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=1, seqNum=0)
        base = 0
        queue.queue_mv[base:base + len(bytes(elem))] = bytes(elem)
        queue.queue_mv[base + 0x30:base + 0x30 + len(bytes(hdr))] = bytes(hdr)
        queue.queue_mv[base + 0x50:base + 0x50 + len(msg)] = msg
        queue.tx_view[4] = 1
      def install_fake_event(queue):
        msg = struct.pack("<QQQ", 0, 0x12345678, 5) + b"ASSERT\0FECS_A\0GR_STATUS\0"
        hdr = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, length=0x20 + len(msg),
          function=4128, rpc_result=0, rpc_result_private=0)
        elem = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=1, seqNum=1)
        base = queue.tx.msgSize
        queue.queue_mv[base:base + len(bytes(elem))] = bytes(elem)
        queue.queue_mv[base + 0x30:base + 0x30 + len(bytes(hdr))] = bytes(hdr)
        queue.queue_mv[base + 0x50:base + 0x50 + len(msg)] = msg
      install_fake_response(fake_q)
      install_fake_event(fake_q)
      fake_q.tx_view[4] = 2
      fake.stat_q = fake.cmd_q
      NV_GSP.rpc_rm_alloc(fake, hParent=0x80, hClass=0xC56F, params=bytes(range(16)))
      NV_GSP.rpc_rm_alloc(fake, hParent=0xcf000001, hClass=fake.compute_class, params=b"")
      fake.dma_class = 0xc7b5
      NV_GSP.rpc_rm_alloc(fake, hParent=0xcf000001, hClass=fake.dma_class, params=b"")
      NV_GSP.rpc_rm_control(fake, hObject=0x2080, cmd=0x2080012B, params=bytes(range(8)))
      NV_GSP.rpc_rm_control(fake, hObject=0xcf000001, cmd=0xC36F0108, params=struct.pack("<i", -1))
      NV_GSP.rpc_rm_control(fake, hObject=0xcf000002, cmd=0xA06C0101, params=struct.pack("<I", 1))
      NVRpcQueue.send_rpc(fake_q, 103, b"abc")
      def yield_read_resp_stub(self):
        yield 103, b"ok"
      real_traced_read_resp = NVRpcQueue.read_resp
      old_read_resp_for_wrapper = real_traced_read_resp.__closure__[0].cell_contents if getattr(real_traced_read_resp, "__closure__", None) else None
      if old_read_resp_for_wrapper is not None:
        real_traced_read_resp.__closure__[0].cell_contents = yield_read_resp_stub
      list(NVRpcQueue.read_resp(fake_q))
      if old_read_resp_for_wrapper is not None:
        real_traced_read_resp.__closure__[0].cell_contents = old_read_resp_for_wrapper
      def yield_missed_read_resp_stub(self):
        yield 76, b"late"
      if old_read_resp_for_wrapper is not None:
        real_traced_read_resp.__closure__[0].cell_contents = yield_missed_read_resp_stub
      list(NVRpcQueue.read_resp(fake_q))
      if old_read_resp_for_wrapper is not None:
        real_traced_read_resp.__closure__[0].cell_contents = old_read_resp_for_wrapper
      install_fake_response(fake_q, msg=b"bad", result=0x1f, private=0x2)
      def raising_read_resp_stub(self):
        raise RuntimeError("RPC call 103 failed with result 31")
        yield
      if old_read_resp_for_wrapper is not None:
        real_traced_read_resp.__closure__[0].cell_contents = raising_read_resp_stub
      try:
        list(NVRpcQueue.read_resp(fake_q))
      except RuntimeError:
        pass
      else:
        raise AssertionError("failing tiny read_resp did not fail")
      if old_read_resp_for_wrapper is not None:
        real_traced_read_resp.__closure__[0].cell_contents = old_read_resp_for_wrapper
      print(_queue_dump("fake", fake_q), flush=True)
      class FailingGsp(FakeGsp):
        pass
      def fail_alloc(self, hParent, hClass, params, client=None):
        raise RuntimeError("tiny alloc failed")
      failing = FailingGsp()
      failing.cmd_q = fake_q
      failing.stat_q = fake_q
      NV_GSP.rpc_rm_alloc = fail_alloc
      NV_FLCN._add_tiny_trace_installed = False
      install_tinygrad_falcon_trace()
      try:
        NV_GSP.rpc_rm_alloc(failing, hParent=0xcf000001, hClass=failing.compute_class, params=b"")
      except RuntimeError:
        pass
      else:
        raise AssertionError("failing tiny rm alloc did not fail")
      def fake_init_golden_image(self):
        self.grctx_bufs = {0: object(), 2: object(), 9: object(), 10: object(), 11: object()}
      NV_GSP.init_golden_image = fake_init_golden_image
      NV_FLCN._add_tiny_trace_installed = False
      install_tinygrad_falcon_trace()
      fake.gpfifo_class = 0xC56F
      fake.dma_class = 0xC7B5
      NV_GSP.init_golden_image(fake)
      print_tiny_trace_command()
      print_tiny_live_log_workflow()
      print_tiny_live_log_workflow(standalone_script="examples/mul.py", standalone_log="mul-standalone.log", tiny_log="mul-tiny.log")
      print_tiny_debug_help()
      print_tiny_debug_help(standalone_script="examples/mul.py")
    text = buf.getvalue()
    assert "tiny rm_alloc pre" in text and "tiny rm_alloc post" in text
    assert "tiny rm_alloc pre_state parent=0x80 class=0xc56f class_name=AMPERE_CHANNEL_GPFIFO_A" in text
    assert "bar1=0x" in text and "gsp=(engine=0x" in text and "sec2=(engine=0x" in text
    assert "tiny rm_alloc pre_queues" in text and "tiny rm_alloc post_queues" in text
    assert "class_name=AMPERE_CHANNEL_GPFIFO_A" in text and "object=0xcf000001" in text
    assert f"params_sha256={hashlib.sha256(bytes(range(16))).hexdigest()}" in text
    compute_rpc_sha256 = hashlib.sha256(_tiny_pack_rpc_rm_alloc(fake.priv_root, 0xcf000001, 0xcf000001, fake.compute_class, b"")).hexdigest()
    assert f"tiny compute_alloc parent=0xcf000001 object=0xcf000001 compute_class=0xc6c0 rpc_sha256={compute_rpc_sha256}" in text
    assert "compute_alloc_stack" in text
    dma_rpc_sha256 = hashlib.sha256(_tiny_pack_rpc_rm_alloc(fake.priv_root, 0xcf000001, 0xcf000001, fake.dma_class, b"")).hexdigest()
    assert f"tiny dma_alloc parent=0xcf000001 object=0xcf000001 dma_class=0xc7b5 rpc_sha256={dma_rpc_sha256}" in text
    assert "dma_alloc_stack" in text
    assert "tiny rm_control pre" in text and "tiny rm_control post" in text
    assert "cmd_name=NV2080_CTRL_CMD_GPU_PROMOTE_CTX" in text and "rm_control_stack" in text
    assert f"result_sha256={hashlib.sha256(bytes(range(8))).hexdigest()}" in text
    token_rpc_sha256 = hashlib.sha256(_tiny_pack_rpc_rm_control(fake.priv_root, 0xcf000001, 0xC36F0108, struct.pack("<i", -1))).hexdigest()
    schedule_rpc_sha256 = hashlib.sha256(_tiny_pack_rpc_rm_control(fake.priv_root, 0xcf000002, 0xA06C0101, struct.pack("<I", 1))).hexdigest()
    assert f"tiny token_control object=0xcf000001 cmd=0xc36f0108 rpc_sha256={token_rpc_sha256}" in text
    assert f"tiny schedule_control object=0xcf000002 cmd=0xa06c0101 rpc_sha256={schedule_rpc_sha256}" in text
    assert "token_control_stack" in text and "schedule_control_stack" in text
    assert "tiny send_rpc func=103 func_name=GSP_RM_ALLOC" in text
    assert "tiny read_rpc rp=0 wp=2 advance=1 func=103 func_name=GSP_RM_ALLOC len=34 result=0x0 private=0x0" in text
    assert "tiny read_rpc_yield rp=0 wp=2 advance=1 func=103 func_name=GSP_RM_ALLOC len=34 result=0x0 private=0x0 sha256=" in text
    assert "tiny read_rpc_yield func=76 func_name=GSP_RM_CONTROL sha256=" in text
    assert "func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD" in text
    assert "post_nocat=qwords=0x0,0x12345678,0x5 kind=0x5 strings=ASSERT|FECS_A|GR_STATUS" in text
    assert "tiny rm_alloc exception parent=0xcf000001 class=0xc6c0 class_name=AMPERE_COMPUTE_A exc=RuntimeError msg=tiny alloc failed" in text
    assert "cmd_slot1: checksum=0x0 seq=1 elem_count=1 func=4128 func_name=EVENT_GSP_POST_NOCAT_RECORD" in text
    assert "tiny golden_start priv_root=0xc1e00000 gpfifo_class=0xc56f compute_class=0xc6c0 dma_class=0xc7b5" in text
    assert "golden_start_stack" in text and "tiny golden_done grctx_ids=[0, 2, 9, 10, 11]" in text
    assert "tiny_trace_command NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 examples/add_tiny.py" in text
    assert "live_log_workflow script=examples/add_tiny.py standalone_script=examples/add.py standalone_log=standalone-golden.log tiny_log=tiny-golden.log" in text
    assert "tiny_bar_info_command python3 examples/add_tiny.py --bar-info" in text
    assert "gate_command NV_ADD_TRANSPORT=mac-egpu python3 examples/add.py --transport-preflight-plan --require-ready" in text
    assert "standalone_log_command NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1 NV_ADD_BOOT_GSP=1 NV_ADD_SUMMARY=1 NV_ADD_CHECK_FRTS_BAR1=1" in text
    assert "NV_ADD_TRACE_RPC=1 NV_ADD_TRACE_RPC_READ=1" in text
    assert "tiny_log_command NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 examples/add_tiny.py 2>&1 | tee tiny-golden.log" in text
    assert "compare_command python3 examples/add.py --compare-trace-logs --standalone-log standalone-golden.log --tiny-log tiny-golden.log" in text
    assert "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence" in text
    assert "bar_info python3 examples/add_tiny.py --bar-info" in text
    stack_text = "\n".join(tiny_live_stack_log_workflow_lines())
    assert "live_stack_log_workflow script=examples/add_tiny.py standalone_script=examples/add.py standalone_log=standalone-stack.log tiny_log=tiny-stack.log" in stack_text
    assert "tiny_bar_info_command python3 examples/add_tiny.py --bar-info" in stack_text
    assert "standalone_log_command NV_ADD_TRACE_RM_STACK=1 NV_ADD_TRACE_CHANNEL_STACK=1 NV_ADD_TRACE_LAUNCH_STACK=1 NV_ADD_TRACE_FALCON=1 NV_ADD_TRANSPORT=mac-egpu" in stack_text
    assert "tiny_log_command NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 examples/add_tiny.py 2>&1 | tee tiny-stack.log" in stack_text
    assert "compare_command python3 examples/add.py --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log" in stack_text
    assert "workflow_check stack inspect trace_log_compare_stack, trace_log_compare_falcon" in stack_text
    assert "live_log_workflow script=examples/add_tiny.py standalone_script=examples/mul.py standalone_log=mul-standalone.log tiny_log=mul-tiny.log" in text
    assert "compare_command python3 examples/mul.py --compare-trace-logs --standalone-log mul-standalone.log --tiny-log mul-tiny.log" in text
    assert "workflow_rule run standalone_log_command only after gate result is ready-for-gsp" in text
    assert "workflow_rule run tiny_log_command in the same eGPU session if standalone stalls or times out" in text
    assert "tiny_debug_help script=examples/add_tiny.py" in text
    assert "trace_command python3 examples/add_tiny.py --trace-command" in text
    assert "trace_selftest python3 examples/add_tiny.py --trace-selftest" in text
    assert "live_log_workflow python3 examples/add_tiny.py --live-log-workflow --standalone-script examples/add.py" in text
    assert "live_stack_log_workflow python3 examples/add_tiny.py --live-stack-log-workflow --standalone-script examples/add.py" in text
    assert "compare_trace_logs python3 examples/add.py --compare-trace-logs --standalone-log standalone-golden.log --tiny-log tiny-golden.log" in text
    assert "compare_stack_trace_logs python3 examples/add.py --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log" in text
    assert "live_log_workflow python3 examples/add_tiny.py --live-log-workflow --standalone-script examples/mul.py" in text
    assert "live_stack_log_workflow python3 examples/add_tiny.py --live-stack-log-workflow --standalone-script examples/mul.py" in text
    assert "compare_trace_logs python3 examples/mul.py --compare-trace-logs --standalone-log standalone-golden.log --tiny-log tiny-golden.log" in text
    assert "compare_stack_trace_logs python3 examples/mul.py --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log" in text
    old_argv = sys.argv[:]
    try:
      sys.argv = ["examples/add_tiny.py", "--live-log-workflow", "--standalone-script"]
      try:
        main()
        raise AssertionError("missing standalone script argument was accepted")
      except SystemExit as exc:
        assert str(exc) == "--standalone-script requires a value"
      sys.argv = ["examples/add_tiny.py", "--live-stack-log-workflow", "--standalone-log", "custom-standalone.log", "--tiny-log", "custom-tiny.log"]
      custom_workflow = io.StringIO()
      with contextlib.redirect_stdout(custom_workflow): main()
      custom_text = custom_workflow.getvalue()
      assert "standalone_log=custom-standalone.log tiny_log=custom-tiny.log" in custom_text
      assert "tee custom-standalone.log" in custom_text and "tee custom-tiny.log" in custom_text
      assert "compare_command python3 examples/add.py --compare-trace-logs --standalone-log custom-standalone.log --tiny-log custom-tiny.log" in custom_text
      sys.argv = ["examples/add_tiny.py", "--live-log-workflow", "--standalone-log"]
      missing_value = io.StringIO()
      with contextlib.redirect_stdout(missing_value):
        try:
          main()
          raise AssertionError("missing standalone log value was accepted")
        except SystemExit as exc:
          assert exc.code == 2
      assert missing_value.getvalue().strip() == "cli_arg_error kind=missing-value flag=--standalone-log"
      sys.argv = ["examples/add_tiny.py", "--live-stack-log-workflow", "--tiny-log"]
      missing_value = io.StringIO()
      with contextlib.redirect_stdout(missing_value):
        try:
          main()
          raise AssertionError("missing tiny log value was accepted")
        except SystemExit as exc:
          assert exc.code == 2
      assert missing_value.getvalue().strip() == "cli_arg_error kind=missing-value flag=--tiny-log"
    finally:
      sys.argv = old_argv
  finally:
    NV_GSP.rpc_rm_alloc, NV_GSP.rpc_rm_control = old_alloc, old_control
    NV_GSP.init_golden_image, NVRpcQueue.send_rpc, NVRpcQueue.read_resp = old_init_golden_image, old_send_rpc, old_read_resp
    NV_FLCN._add_tiny_trace_installed = old_installed
    if old_trace_stack is None: os.environ.pop("NV_ADD_TINY_TRACE_STACK", None)
    else: os.environ["NV_ADD_TINY_TRACE_STACK"] = old_trace_stack
  print("trace_selftest=ok")

class CubinHelper:
  class Reg:
    RZ = 255
    R0 = 0; R1 = 1; R2 = 2; R3 = 3; R4 = 4; R5 = 5; R6 = 6; R7 = 7
    R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15

  class UReg:
    URZ = 63
    UR4 = 4  # only UR4 is used in our cubin

  #  cuobjdump -sass   (closed-source NVIDIA disassembler; mnemonic + bytes)
  #  cuasm sm_86       (open-source assembler, CuAsm/InsAsmRepos)
  #  denvdis data11    (open-source 128-bit SASS spec, denvdis/data11/sm86_1.txt)
  class Op:
    LDC     = 0x7a02  # LDC / LDC.64 (alias: MOV).  denvdis 0xb82 family.
    LDCU64  = 0x7ab9  # LDCU.64.                    denvdis ULDC_default 0xab9 — EXACT.
    FADD    = 0x7221  # FADD / IMAD.WIDE.           denvdis FADD_Rb 0x221 — EXACT (low12).
    FMUL    = 0x7220  # FMUL / IMAD.WIDE.           denvdis FMUL_v3 0x220 — EXACT (low12).
    LDG     = 0x7981  # LDG descriptor pre-load.    denvdis LDG_R_dARI 0x981 — EXACT (low12).
    STG     = 0x7986  # STG.E.                      denvdis STG_E 0xc86 family.
    EXIT    = 0x794d  # EXIT.                       denvdis EXIT 0x94d — EXACT (low12).
    BRA     = 0x7947  # BRA.                        denvdis BRA 0x947 — EXACT (low12).
    NOP     = 0x7918  # NOP.                        denvdis NOP 0x918 — EXACT (low12).

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
    return [f"sem_addr=0x{sem_addr:x}", f"payload={payload}", f"execute=0x{args[4]:08x}"]
  if method == 0x1698 and len(args) == 1:
    return [f"invalidate_flags=0x{args[0]:08x}"]
  if method == 0x02b4 and len(args) == 1:
    return [f"qmd_addr=0x{args[0] << 8:x}", f"qmd_addr_shifted8=0x{args[0]:x}"]
  if method == 0x02c0 and len(args) == 1:
    return [f"pcas2_action=0x{args[0]:x}"]
  return [f"arg{i}=0x{arg:08x}" for i, arg in enumerate(args)]

def round_up(x, y): return ((x + y - 1) // y) * y

def write_words(dst, offset, words):
  dst[offset:offset + len(words)] = array.array('I', words)

def submit_gpfifo(dev, words):
  cmdq_addr = dev.cmdq_allocator.alloc(len(words) * 4, 16)
  cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
  write_words(dev.cmdq, cmdq_wptr, words)

  fifo = dev.compute_gpfifo
  ring_index = fifo.put_value % fifo.entries_count
  fifo.ring[ring_index] = (cmdq_addr // 4 << 2) | (len(words) << 42) | (1 << 41)
  fifo.gpput[0] = (fifo.put_value + 1) % fifo.entries_count
  dev.gpu_mmio[0x90 // 4] = fifo.token
  fifo.put_value += 1

  entry = int(fifo.ring[ring_index])
  addr = ((entry & ((1 << 40) - 1)) >> 2) << 2
  packets = (entry >> 42) & ((1 << 20) - 1)
  print(f"  GPFIFO[{ring_index}]=0x{entry:016x} addr=0x{addr:x} packets={packets} token=0x{fifo.token:x}")
  print(f"  doorbell gpput={fifo.gpput[0]} put_value={fifo.put_value}")

def wait_signal(signal, value, timeout_ms=30000):
  start = time.perf_counter()
  while signal.value < value:
    if (time.perf_counter() - start) * 1000 > timeout_ms:
      raise RuntimeError(f"timeout waiting for timeline {value}, got {signal.value}")
    time.sleep(0.001)

def manual_launch(dev, program, out, a, b):
  kernargs = dev.kernargs_buf.offset(dev.kernargs_offset_allocator.alloc(program.kernargs_alloc_size, 8), program.kernargs_alloc_size)
  cbuf_words = program.cbuf_0 or []
  kernargs.cpu_view().view(size=len(cbuf_words) * 4, fmt='I')[:] = array.array('I', cbuf_words)
  kernargs.cpu_view().view(offset=len(cbuf_words) * 4, size=3 * 8, fmt='Q')[:] = array.array('Q', [out.va_addr, a.va_addr, b.va_addr])

  qmd_buf = kernargs.offset(round_up(program.constbufs[0][1], 1 << 8))
  qmd_buf.cpu_view().view(size=program.qmd.mv.nbytes, fmt='B')[:] = program.qmd.mv
  qmd = type(program.qmd)(dev=dev, view=qmd_buf.cpu_view())
  qmd.write(cta_raster_width=1, cta_raster_height=1, cta_raster_depth=1,
            cta_thread_dimension0=1, cta_thread_dimension1=1, cta_thread_dimension2=1)
  qmd.set_constant_buf_addr(0, kernargs.va_addr)

  wait_value = dev.timeline_value - 1
  done_value = dev.next_timeline()
  signal_addr = dev.timeline_signal.value_addr
  qmd.write(release0_enable=1, release0_address_lower=signal_addr & 0xffffffff, release0_address_upper=(signal_addr >> 32) & 0xff,
            release0_payload_lower=done_value & 0xffffffff, release0_payload_upper=done_value >> 32)

  words = build_launch_words(signal_addr, wait_value, done_value, qmd_buf.va_addr)[:12]
  print(f"submit #manual: NVComputeQueue words={len(words)}")
  for index, typ, subc, method, name, args in decode_words(words):
    print(f"  method[{index}] {name}: typ={typ} subc={subc} mthd=0x{method:x} args=[{', '.join(describe_args(method, args))}]")
  submit_gpfifo(dev, words)
  wait_signal(dev.timeline_signal, done_value)

def build_cubin(): # nvdisasm add.cubin
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

    # SASS_ARITHMETIC
    ((ch.Reg.R11 << 24) | (ch.Reg.R11 << 16) | ch.Op.FADD, 0x00000007, 0x00000000, 0x004fe200),  # FADD R11, R11, R7
    ((ch.Reg.R10 << 24) | (ch.Reg.R10 << 16) | ch.Op.FADD, 0x00000006, 0x00000000, 0x000fe200),  # FADD R10, R10, R6
    ((ch.Reg.R9  << 24) | (ch.Reg.R9  << 16) | ch.Op.FADD, 0x00000005, 0x00000000, 0x000fe200),  # FADD R9, R9, R5
    ((ch.Reg.R8  << 24) | (ch.Reg.R8 << 16) | ch.Op.FADD, 0x00000004, 0x00000000, 0x000fe200),    # FADD R8, R8, R4

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

def main():
  if "--standalone-script" in sys.argv:
    index = sys.argv.index("--standalone-script")
    if index + 1 >= len(sys.argv): raise SystemExit("--standalone-script requires a value")
    standalone_script = sys.argv[index + 1]
  else:
    standalone_script = "examples/add.py"
  if "--trace-selftest" in sys.argv:
    trace_selftest()
    return
  if "--bar-info" in sys.argv:
    print_tiny_bar_info()
    return
  if "--trace-command" in sys.argv:
    print_tiny_trace_command()
    return
  if "--live-log-workflow" in sys.argv:
    print_tiny_live_log_workflow(standalone_script=standalone_script,
                                 standalone_log=cli_arg_value("--standalone-log") or "standalone-golden.log",
                                 tiny_log=cli_arg_value("--tiny-log") or "tiny-golden.log")
    return
  if "--live-stack-log-workflow" in sys.argv:
    print_tiny_live_stack_log_workflow(standalone_script=standalone_script,
                                       standalone_log=cli_arg_value("--standalone-log") or "standalone-stack.log",
                                       tiny_log=cli_arg_value("--tiny-log") or "tiny-stack.log")
    return
  if "--debug-help" in sys.argv:
    print_tiny_debug_help(standalone_script=standalone_script)
    return
  a = (1.0, 2.0, 3.0, 4.0)
  b = (10.0, 20.0, 30.0, 40.0)
  cubin = build_cubin()
  print(f"cubin_bytes={len(cubin)} expected_result={[x + y for x, y in zip(a, b)]}")
  dev = open_pcie_device()
  print(f"device={dev.device} iface={type(dev.iface).__name__}")
  a_buf = dev.allocator.alloc(16)
  b_buf = dev.allocator.alloc(16)
  out_buf = dev.allocator.alloc(16)
  dev.allocator._copyin(a_buf, memoryview(struct.pack("4f", *a)))
  dev.allocator._copyin(b_buf, memoryview(struct.pack("4f", *b)))
  dev.allocator._copyin(out_buf, memoryview(bytes(16)))
  program = dev.runtime("E_4", cubin)
  manual_launch(dev, program, out_buf, a_buf, b_buf)
  result_bytes = bytearray(16)
  dev.allocator._copyout(memoryview(result_bytes), out_buf)
  result = list(struct.unpack("4f", result_bytes))
  print(f"result={result}")
  print("submitted rebuilt NV add kernel")


if __name__ == "__main__":
  main()
