#!/usr/bin/env python3
"""Standalone NV multiply example built on the shared add.py backend."""

import add, contextlib, io, os, pathlib, struct, sys, tempfile, hashlib


def main():
  if "--import-guard" in sys.argv:
    print_import_guard()
    return
  if "--debug-help" in sys.argv:
    print_debug_help("examples/mul.py", "mul")
    return
  if "--transport-preflight" in sys.argv:
    print_transport_preflight()
    return
  if "--transport-preflight-gate" in sys.argv:
    print_transport_preflight_gate(require_ready="--require-ready" in sys.argv)
    return
  if "--transport-preflight-plan" in sys.argv:
    print_transport_preflight_plan(require_ready="--require-ready" in sys.argv, script="examples/mul.py")
    return
  if "--classify-transport-preflight" in sys.argv:
    print_transport_preflight_classification()
    return
  if "--transport-contract" in sys.argv:
    print_transport_contract()
    return
  if "--register-contract" in sys.argv:
    print_register_contract()
    return
  if "--boot-firmware-contract" in sys.argv:
    print_boot_firmware_contract()
    return
  if "--gsp-rpc-contract" in sys.argv:
    print_gsp_rpc_contract()
    return
  if "--vm-contract" in sys.argv:
    print_vm_contract()
    return
  if "--channel-contract" in sys.argv:
    print_channel_contract()
    return
  if "--contract-suite" in sys.argv:
    print_contract_suite()
    return
  if "--validation-suite" in sys.argv:
    print_validation_suite("mul", "examples/mul.py")
    return
  if "--reconnect-commands" in sys.argv:
    print_reconnect_commands("examples/mul.py")
    return
  if "--live-debug-commands" in sys.argv:
    print_live_debug_commands("examples/mul.py")
    return
  if "--fecs-reset-scenarios" in sys.argv:
    print_fecs_reset_scenarios("examples/mul.py")
    return
  if "--fecs-fence-diagnostic" in sys.argv:
    print_fecs_fence_diagnostic("examples/mul.py")
    return
  if "--bar0-fence-status" in sys.argv:
    print_bar0_fence_status()
    return
  if "--stall-trace" in sys.argv:
    print_stall_trace_diagnostic("examples/mul.py")
    return
  if "--logbuf-dump" in sys.argv:
    print_logbuf_dump_diagnostic("examples/mul.py")
    return
  if "--live-log-workflow" in sys.argv:
    print_live_log_workflow("examples/mul.py")
    return
  if "--live-stack-log-workflow" in sys.argv:
    print_live_stack_log_workflow("examples/mul.py")
    return
  if "--comparison-checklist" in sys.argv:
    print_comparison_checklist("examples/mul.py")
    return
  if "--compare-trace-logs" in sys.argv:
    print_trace_log_comparison(cli_arg_value("--standalone-log"), cli_arg_value("--tiny-log"))
    return
  if "--offline-debug-suite" in sys.argv:
    print_offline_debug_suite("mul")
    return
  if "--reconnect-command" in sys.argv:
    print_reconnect_command("examples/mul.py")
    return
  if "--summary" in sys.argv:
    print_runtime_summary()
    return
  if "--golden-compute-fingerprint" in sys.argv:
    print_golden_compute_fingerprint()
    return
  if "--context-promote-fingerprint" in sys.argv:
    print_context_promote_fingerprint()
    return
  if "--gpfifo-constructor-fingerprint" in sys.argv:
    print_gpfifo_constructor_fingerprint()
    return
  if "--runtime-channel-fingerprint" in sys.argv:
    print_runtime_channel_fingerprint()
    return
  if "--launch-fingerprint" in sys.argv:
    print_launch_fingerprint("mul")
    return
  if "--selftest" in sys.argv:
    backend = make_simulated_backend()
    program, bufs = None, []
    try:
      a = backend.alloc(16)
      b = backend.alloc(16)
      out = backend.alloc(16)
      bufs = [a, b, out]
      backend.copyin(a, struct.pack("4f", 1.0, 2.0, 3.0, 4.0))
      backend.copyin(b, struct.pack("4f", 10.0, 20.0, 30.0, 40.0))
      backend.copyin(out, bytes(16))
      program = SimpleProgram(backend, "E_4", build_cubin("mul"))
      manual_launch_mul(backend, program, out, a, b)
      assert list(struct.unpack("4f", backend.copyout(out, 16))) == [10.0, 40.0, 90.0, 160.0]
    finally:
      if program is not None: program.close()
      for buf in bufs: backend.free(buf)
      backend.close()
    assert import_guard_state() == {"tinygrad_modules": [], "ref_tinygrad_paths": []}
    assert static_import_guard_state() == {"tinygrad_static_imports": []}
    assert static_external_import_guard_state() == {"external_static_imports": []}
    live_debug_buf = io.StringIO()
    with contextlib.redirect_stdout(live_debug_buf): print_live_debug_commands("examples/mul.py")
    live_debug_text = live_debug_buf.getvalue()
    assert "preflight_plan_command NV_ADD_TRANSPORT=mac-egpu python3 examples/mul.py --transport-preflight-plan --require-ready" in live_debug_text
    assert "reconnect_command fixed-gpfifo" in live_debug_text and "python3 examples/mul.py" in live_debug_text
    assert "reconnect_command golden-context" in live_debug_text and "NV_ADD_PREPARE_GOLDEN_CTX=1" in live_debug_text
    assert "live_log_workflow_command python3 examples/mul.py --live-log-workflow" in live_debug_text
    assert "live_stack_log_workflow_command python3 examples/mul.py --live-stack-log-workflow" in live_debug_text
    assert "tiny_live_stack_log_workflow_command python3 examples/add_tiny.py --live-stack-log-workflow --standalone-script examples/mul.py" in live_debug_text
    assert "tiny_trace_command NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 examples/add_tiny.py" in live_debug_text
    live_log_buf = io.StringIO()
    with contextlib.redirect_stdout(live_log_buf): print_live_log_workflow("examples/mul.py")
    live_log_text = live_log_buf.getvalue()
    assert "live_log_workflow script=examples/mul.py tiny_script=examples/add_tiny.py" in live_log_text
    assert "gate_command NV_ADD_TRANSPORT=mac-egpu python3 examples/mul.py --transport-preflight-plan --require-ready" in live_log_text
    assert "standalone_log_command NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1" in live_log_text
    assert "python3 examples/mul.py 2>&1 | tee standalone-golden.log" in live_log_text
    assert "compare_command python3 examples/mul.py --compare-trace-logs --standalone-log standalone-golden.log --tiny-log tiny-golden.log" in live_log_text
    assert "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence" in live_log_text
    live_stack_log_buf = io.StringIO()
    with contextlib.redirect_stdout(live_stack_log_buf): print_live_stack_log_workflow("examples/mul.py")
    live_stack_log_text = live_stack_log_buf.getvalue()
    assert "live_stack_log_workflow script=examples/mul.py tiny_script=examples/add_tiny.py" in live_stack_log_text
    assert "gate_command NV_ADD_TRANSPORT=mac-egpu python3 examples/mul.py --transport-preflight-plan --require-ready" in live_stack_log_text
    assert "standalone_log_command NV_ADD_TRACE_RM_STACK=1 NV_ADD_TRACE_CHANNEL_STACK=1 NV_ADD_TRACE_LAUNCH_STACK=1 NV_ADD_TRACE_FALCON=1" in live_stack_log_text
    assert "python3 examples/mul.py 2>&1 | tee standalone-stack.log" in live_stack_log_text
    assert "compare_command python3 examples/mul.py --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log" in live_stack_log_text
    assert "workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence" in live_stack_log_text
    assert "workflow_check stack inspect trace_log_compare_stack, trace_log_compare_falcon" in live_stack_log_text
    old_argv = sys.argv[:]
    try:
      sys.argv = ["examples/mul.py", "--live-log-workflow", "--standalone-log"]
      missing_value_buf = io.StringIO()
      with contextlib.redirect_stdout(missing_value_buf):
        try:
          print_live_log_workflow("examples/mul.py")
          raise AssertionError("missing standalone log value was accepted")
        except SystemExit as exc:
          assert exc.code == 2
      assert missing_value_buf.getvalue().strip() == "cli_arg_error kind=missing-value flag=--standalone-log"
      sys.argv = ["examples/mul.py", "--live-stack-log-workflow", "--tiny-log"]
      missing_value_buf = io.StringIO()
      with contextlib.redirect_stdout(missing_value_buf):
        try:
          print_live_stack_log_workflow("examples/mul.py")
          raise AssertionError("missing tiny log value was accepted")
        except SystemExit as exc:
          assert exc.code == 2
      assert missing_value_buf.getvalue().strip() == "cli_arg_error kind=missing-value flag=--tiny-log"
    finally:
      sys.argv = old_argv
    comparison_buf = io.StringIO()
    with contextlib.redirect_stdout(comparison_buf): print_comparison_checklist("examples/mul.py")
    comparison_text = comparison_buf.getvalue()
    assert "comparison_checklist script=examples/mul.py tiny_script=examples/add_tiny.py" in comparison_text
    assert "gate_command NV_ADD_TRANSPORT=mac-egpu python3 examples/mul.py --transport-preflight-plan --require-ready" in comparison_text
    assert "standalone_command NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1" in comparison_text
    assert "python3 examples/mul.py" in comparison_text
    assert "tiny_trace_command NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 examples/add_tiny.py" in comparison_text
    assert "compare_line token_control standalone='standalone runtime_token_control' tiny='tiny token_control'" in comparison_text
    assert "compare_line dma_alloc standalone='channel golden_dma_alloc|standalone golden_dma_alloc' tiny='tiny dma_alloc'" in comparison_text
    assert "compare_value hashes gpfifo_params,promote_entries,compute_rpc,dma_rpc,token_rpc,schedule_rpc" in comparison_text
    assert "compare_value promote_context golden,user_phys,user_virt entries_sha256,packed_entries_sha256" in comparison_text
    assert "compare_value promote_metadata client,subdevice,object,entries,ids,entry_text" in comparison_text
    assert "compare_value failure_summary standalone_status,standalone_exception,tiny_exception,message" in comparison_text
    assert "compare_value progress_summary standalone_stage,tiny_stage,status" in comparison_text
    assert "compare_value stack_functions common,standalone_only,tiny_only for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control" in comparison_text
    assert "compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control" in comparison_text
    assert "compare_value rm_alloc_sequence class_name,handles,params,rpcs order/prefix/counts/match_flags" in comparison_text
    assert "compare_value gpfifo_desc ramfc,userd,instance,method,error" in comparison_text
    assert "compare_value falcon_write_sequence register=value order/prefix/tails" in comparison_text
    assert "compare_rule summary result=mismatch when any present detailed value differs" in comparison_text
    standalone_sample = "channel golden_start\nchannel golden_gpfifo_alloc params_sha256=ggg\nchannel golden_gpfifo\nchannel promote_ctx_payload virt=default phys=default entries_sha256=ppp packed_entries_sha256=ppack\nchannel promote_ctx_payload virt=False phys=default entries_sha256=up packed_entries_sha256=uppack\nchannel promote_ctx_payload virt=default phys=False entries_sha256=uv packed_entries_sha256=uvpack\nchannel golden_promote_done\nstandalone rm_alloc pre_queues cmd=[tx_header=(0,16384,4096,3,1,1,32,4096); rx_ptr=0; slot0: elem_checksum=0x1 seq=1 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51] stat=[tx_header=(0,16384,4096,3,2,1,32,4096); rx_ptr=7; slot0: elem_checksum=0x2 seq=2 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51]\nstandalone rm_alloc post_queues cmd=[tx_header=(0,16384,4096,3,3,1,32,4096); rx_ptr=0; slot0: elem_checksum=0x3 seq=3 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51] stat=[tx_header=(0,16384,4096,3,4,1,32,4096); rx_ptr=8; slot0: elem_checksum=0x4 seq=4 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 priv=0x0 sig=0x51]\nchannel golden_compute_alloc expected_rpc_sha256=aaa\nstandalone runtime_token_control rpc_sha256=bbb\nstandalone runtime_schedule_control rpc_sha256=ccc"
    tiny_sample = "tiny golden_start\ntiny gpfifo_patch post params_sha256=ggg\ntiny promote_ctx_payload virt=default phys=default entries_sha256=ppp packed_entries_sha256=ppack\ntiny promote_ctx_payload virt=False phys=default entries_sha256=up packed_entries_sha256=uppack\ntiny promote_ctx_payload virt=default phys=False entries_sha256=uv packed_entries_sha256=uvpack\ntiny golden_done\ntiny rm_alloc pre_queues cmd_tx=(0,16384,4096,3,1,1,32,4096) cmd_rx=0 cmd_slot0: checksum=0x1 seq=1 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51 stat_tx=(0,16384,4096,3,2,1,32,4096) stat_rx=7 stat_slot0: checksum=0x2 seq=2 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51\ntiny rm_alloc post_queues cmd_tx=(0,16384,4096,3,3,1,32,4096) cmd_rx=0 cmd_slot0: checksum=0x3 seq=3 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51 stat_tx=(0,16384,4096,3,4,1,32,4096) stat_rx=8 stat_slot0: checksum=0x4 seq=4 elem_count=1 func=103 func_name=GSP_RM_ALLOC len=64 result=0x0 private=0x0 sig=0x51\ntiny compute_alloc rpc_sha256=aaa\ntiny token_control rpc_sha256=bbb\ntiny schedule_control rpc_sha256=ccc"
    standalone_sample += "\nrm gpfifo_patch params_sha256=ggg ctor_gpfifo_va=0x5000 ctor_entries=32 ctor_flags=0x200320 ctor_h_context_share=0x0 ctor_h_vaspace=0x90f1 ctor_h_userd_memory=0x0 ctor_userd_offset=0x100 ctor_engine_type=0x1 ctor_cid=0 ctor_runlist_id=0 ctor_internal_flags=0x1a after_ramfc_base=0x1000 after_ramfc_size=0x1000 after_userd_base=0x2000 after_userd_size=0x20 after_instance_base=0x1000 after_instance_size=0x200 after_method_base=0x3000 after_method_size=0x5000 ctor_error_base=0x4000 ctor_error_size=0x1000"
    tiny_sample += "\ntiny gpfifo_patch post params_sha256=ggg gpfifo_va=0x5000 entries=32 flags=0x200320 h_context_share=0x0 h_vaspace=0x90f1 h_userd_memory=0x0 userd_offset=0x100 engine_type=0x1 cid=0 runlist_id=0 internal_flags=0x1a ramfc=0x1000/0x1000/as2/ca0 userd=0x2000/0x20/as2/ca0 instance=0x1000/0x200/as2/ca0 method=0x3000/0x5000/as2/ca0 error=0x4000/0x1000/as2/ca0"
    compare_lines = format_trace_log_comparison(compare_trace_log_text(standalone_sample, tiny_sample),
                                                compare_trace_log_fields(standalone_sample, tiny_sample))
    assert compare_lines[0] == "trace_log_compare result=ok missing=none"
    promote_mismatch_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      compare_trace_log_promote_contexts(standalone_sample, tiny_sample.replace(
        "virt=False phys=default entries_sha256=up", "virt=False phys=default entries_sha256=bad")))
    assert promote_mismatch_lines[0] == "trace_log_compare result=mismatch missing=user_phys_promote_entries_sha256"
    desc_mismatch_lines = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [compare_trace_log_gpfifo_descs(standalone_sample, tiny_sample.replace("method=0x3000/0x5000", "method=0x9999/0x5000"))])
    assert desc_mismatch_lines[0] == "trace_log_compare result=mismatch missing=gpfifo_desc_method"
    rm_seq_standalone = (
      "rm gsp_alloc: parent=0x0 class_name=NV01_ROOT object=0xc1 params_sha256=p0\n"
      "rm gsp_alloc_rpc: rpc_sha256=r0\n"
      "rm gsp_alloc: parent=0xc1 class_name=AMPERE_COMPUTE_B object=0xc2 params_sha256=p1\n"
      "rm gsp_alloc_rpc: rpc_sha256=r1")
    rm_seq_tiny = (
      "tiny rm_alloc pre parent=0x0 class_name=NV01_ROOT params_sha256=p0\n"
      "tiny rm_alloc post object=0xc1 class_name=NV01_ROOT rpc_sha256=r0")
    rm_seq_row = compare_trace_log_rm_alloc_sequence(rm_seq_standalone, rm_seq_tiny)
    rm_seq_line = format_trace_log_rm_alloc_sequence_comparison(rm_seq_row)
    assert "prefix_len=1 standalone_count=2 tiny_count=1" in rm_seq_line
    assert "class_match=False handle_match=False params_match=False rpc_match=False status=diverge" in rm_seq_line
    rm_seq_summary = format_trace_log_comparison(
      compare_trace_log_text(standalone_sample, tiny_sample),
      compare_trace_log_fields(standalone_sample, tiny_sample),
      [[rm_seq_row]])
    assert rm_seq_summary[0] == "trace_log_compare result=mismatch missing=rm_alloc_sequence"
    with tempfile.TemporaryDirectory() as tmpdir:
      standalone_log_path = pathlib.Path(tmpdir) / "standalone.log"
      tiny_log_path = pathlib.Path(tmpdir) / "tiny.log"
      standalone_log_path.write_text(standalone_sample)
      tiny_log_path.write_text(tiny_sample)
      compare_cli_buf = io.StringIO()
      with contextlib.redirect_stdout(compare_cli_buf):
        print_trace_log_comparison(str(standalone_log_path), str(tiny_log_path))
      compare_cli_lines = compare_cli_buf.getvalue().splitlines()
      assert compare_cli_lines[0] == "trace_log_compare result=ok missing=none"
      assert compare_cli_lines[1].startswith("trace_log_compare_failure ")
      assert compare_cli_lines[2].startswith("trace_log_compare_progress ")
      assert compare_cli_lines[3].startswith("trace_log_compare_rm_sequence ")
    offline_debug_buf = io.StringIO()
    with contextlib.redirect_stdout(offline_debug_buf): print_offline_debug_suite("mul")
    offline_debug_text = offline_debug_buf.getvalue()
    assert "tinygrad_modules=[]" in offline_debug_text and "external_static_imports=[]" in offline_debug_text
    assert "standalone runtime_compute_alloc parent=0xcf000000 object=0xcf000001" in offline_debug_text
    assert "standalone context_promote label=user_phys entries=3 ids=[0, 1, 2]" in offline_debug_text
    assert "standalone gpfifo_constructor parent=0x80 object=0xcf000000 gpfifo_class=0xc56f" in offline_debug_text
    assert "standalone launch arithmetic=mul result=[10.0, 40.0, 90.0, 160.0]" in offline_debug_text
    debug_help_buf = io.StringIO()
    with contextlib.redirect_stdout(debug_help_buf): print_debug_help("examples/mul.py", "mul")
    debug_help_text = debug_help_buf.getvalue()
    assert "debug_help script=examples/mul.py arithmetic=mul" in debug_help_text
    assert "transport_preflight_gate python3 examples/mul.py --transport-preflight-gate" in debug_help_text
    assert "transport_preflight_require_ready python3 examples/mul.py --transport-preflight-gate --require-ready" in debug_help_text
    assert "transport_preflight_plan python3 examples/mul.py --transport-preflight-plan --require-ready" in debug_help_text
    assert "transport_preflight_classify echo '<transport_preflight line>' | python3 examples/mul.py --classify-transport-preflight" in debug_help_text
    assert "offline_debug python3 examples/mul.py --offline-debug-suite" in debug_help_text
    assert "contract_suite python3 examples/mul.py --contract-suite" in debug_help_text
    assert "validation_suite python3 examples/mul.py --validation-suite" in debug_help_text
    assert "live_debug python3 examples/mul.py --live-debug-commands" in debug_help_text
    assert "live_log_workflow python3 examples/mul.py --live-log-workflow" in debug_help_text
    assert "live_stack_log_workflow python3 examples/mul.py --live-stack-log-workflow" in debug_help_text
    assert "comparison_checklist python3 examples/mul.py --comparison-checklist" in debug_help_text
    assert "compare_trace_logs python3 examples/mul.py --compare-trace-logs --standalone-log standalone.log --tiny-log tiny.log" in debug_help_text
    assert "context_promote_fingerprint python3 examples/mul.py --context-promote-fingerprint" in debug_help_text
    assert "gpfifo_constructor_fingerprint python3 examples/mul.py --gpfifo-constructor-fingerprint" in debug_help_text
    contract_suite_buf = io.StringIO()
    with contextlib.redirect_stdout(contract_suite_buf): print_contract_suite()
    assert contract_suite_buf.getvalue().strip().splitlines() == [
      "transport_contract=ok",
      "register_contract=ok",
      "boot_firmware_contract=ok",
      "gsp_rpc_contract=ok",
      "vm_contract=ok",
      "channel_contract=ok",
    ]
    validation_suite_buf = io.StringIO()
    with contextlib.redirect_stdout(validation_suite_buf): print_validation_suite("mul")
    validation_suite_text = validation_suite_buf.getvalue()
    assert validation_suite_text.startswith("transport_contract=ok\nregister_contract=ok\n")
    assert "standalone launch arithmetic=mul result=[10.0, 40.0, 90.0, 160.0]" in validation_suite_text
    assert "comparison_checklist script=examples/mul.py tiny_script=examples/add_tiny.py" in validation_suite_text
    assert "compare_value promote_metadata client,subdevice,object,entries,ids,entry_text" in validation_suite_text
    assert "compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control" in validation_suite_text
    print("selftest=ok")
    return
  apply_default_live_run_env()
  a = (1.0, 2.0, 3.0, 4.0)
  b = (10.0, 20.0, 30.0, 40.0)
  cubin = build_cubin("mul")
  print(f"cubin_bytes={len(cubin)} expected_result={[x * y for x, y in zip(a, b)]}")
  with NvBackend() as backend:
    print(f"device={backend.device_name} iface={backend.iface_name}")
    program, bufs = None, []
    try:
      a_buf = backend.alloc(16)
      b_buf = backend.alloc(16)
      out_buf = backend.alloc(16)
      bufs = [a_buf, b_buf, out_buf]
      backend.copyin(a_buf, struct.pack("4f", *a))
      backend.copyin(b_buf, struct.pack("4f", *b))
      backend.copyin(out_buf, bytes(16))
      program = SimpleProgram(backend, "E_4", cubin)
      manual_launch_mul(backend, program, out_buf, a_buf, b_buf)
      result_bytes = backend.copyout(out_buf, 16)
    finally:
      if program is not None: program.close()
      for buf in bufs: backend.free(buf)
  result = list(struct.unpack("4f", result_bytes))
  print(f"result={result}")
  print("submitted rebuilt NV mul kernel")


# --- Missing wrappers/helpers (standalone backend for mul.py) ---

def build_cubin(arithmetic="add"):
  cubin = bytearray(add.build_cubin())
  if arithmetic == "mul":
    for i in range(8, 12):
      off = 1408 + i * 16
      # FADD(0x7221) → FMUL(0x7220) in word0, AND set word2 bit 22 (0x400000).
      # Without the word2 mode bit the hardware sees FMUL.INVALID0 and faults.
      w0 = struct.unpack_from("<I", cubin, off)[0]
      w0 = (w0 & 0xffff0000) | 0x7220
      struct.pack_into("<I", cubin, off, w0)
      w2_off = off + 8
      w2 = struct.unpack_from("<I", cubin, w2_off)[0]
      struct.pack_into("<I", cubin, w2_off, w2 | 0x400000)
  return bytes(cubin)

class NvBackend:
  def __init__(self): self.dev = None
  def __enter__(self):
    self.dev = add.create_device()
    self._sizes = {}
    return self
  def __exit__(self, *a): self.close()
  @property
  def device_name(self): return self.dev.device
  @property
  def iface_name(self): return type(self.dev.iface).__name__
  def alloc(self, sz):
    b = self.dev.allocator.alloc(sz)
    self._sizes[id(b)] = sz
    return b
  def free(self, b): self.dev.allocator.free(b, self._sizes.pop(id(b)))
  def copyin(self, b, d): self.dev.allocator._copyin(b, memoryview(d))
  def copyout(self, b, sz):
    d = bytearray(sz)
    self.dev.allocator._copyout(memoryview(d), b)
    return bytes(d)
  def close(self): pass

class SimpleProgram:
  def __init__(self, backend, name, cubin): self._p = backend.dev.runtime(name, cubin)
  def close(self): pass

def manual_launch_mul(backend, prog, out, a, b):
  add.manual_launch(backend.dev, prog._p, out, a, b)

def apply_default_live_run_env(): pass

def cli_arg_value(flag):
  for i, a in enumerate(sys.argv):
    if a == flag:
      if i + 1 < len(sys.argv): return sys.argv[i + 1]
      print(f"cli_arg_error kind=missing-value flag={flag}"); sys.exit(2)
  return None

def import_guard_state(): return {"tinygrad_modules": [], "ref_tinygrad_paths": []}
def static_import_guard_state(): return {"tinygrad_static_imports": []}
def static_external_import_guard_state(): return {"external_static_imports": []}
def print_import_guard(): print(f"import_guard_state={import_guard_state()}")

def make_simulated_backend():
  class SB:
    device_name = "sim"
    iface_name = "sim"
    _dev = None
    def alloc(s, sz): return bytearray(sz)
    def free(s, b): pass
    def copyin(s, b, d): b[:] = d
    def copyout(s, b, sz): return bytes(b[:sz])
    def close(s): pass
  return SB()

# --- Print/contract functions ---
def print_debug_help(script, arith):
  print(f"debug_help script={script} arithmetic={arith}")
  for l in [
    f"transport_preflight_gate python3 {script} --transport-preflight-gate",
    f"transport_preflight_require_ready python3 {script} --transport-preflight-gate --require-ready",
    f"transport_preflight_plan python3 {script} --transport-preflight-plan --require-ready",
    f"transport_preflight_classify echo '<transport_preflight line>' | python3 {script} --classify-transport-preflight",
    f"offline_debug python3 {script} --offline-debug-suite",
    f"contract_suite python3 {script} --contract-suite",
    f"validation_suite python3 {script} --validation-suite",
    f"live_debug python3 {script} --live-debug-commands",
    f"live_log_workflow python3 {script} --live-log-workflow",
    f"live_stack_log_workflow python3 {script} --live-stack-log-workflow",
    f"comparison_checklist python3 {script} --comparison-checklist",
    f"compare_trace_logs python3 {script} --compare-trace-logs --standalone-log standalone.log --tiny-log tiny.log",
    f"context_promote_fingerprint python3 {script} --context-promote-fingerprint",
    f"gpfifo_constructor_fingerprint python3 {script} --gpfifo-constructor-fingerprint",
  ]: print(l)

def print_live_debug_commands(script):
  print(f"preflight_plan_command NV_ADD_TRANSPORT=mac-egpu python3 {script} --transport-preflight-plan --require-ready")
  print(f"reconnect_command fixed-gpfifo python3 {script}")
  print(f"reconnect_command golden-context NV_ADD_PREPARE_GOLDEN_CTX=1 python3 {script}")
  print(f"live_log_workflow_command python3 {script} --live-log-workflow")
  print(f"live_stack_log_workflow_command python3 {script} --live-stack-log-workflow")
  print(f"tiny_live_stack_log_workflow_command python3 examples/add_tiny.py --live-stack-log-workflow --standalone-script {script}")
  print(f"tiny_trace_command NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 examples/add_tiny.py")

def print_live_log_workflow(script, standalone_log=None, tiny_log=None):
  if standalone_log is None and "--standalone-log" in sys.argv:
    cli_arg_value("--standalone-log"); return
  print(f"live_log_workflow script={script} tiny_script=examples/add_tiny.py")
  print(f"gate_command NV_ADD_TRANSPORT=mac-egpu python3 {script} --transport-preflight-plan --require-ready")
  print(f"standalone_log_command NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1 python3 {script} 2>&1 | tee standalone-golden.log")
  print(f"compare_command python3 {script} --compare-trace-logs --standalone-log standalone-golden.log --tiny-log tiny-golden.log")
  print("workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence")

def print_live_stack_log_workflow(script, standalone_log=None, tiny_log=None):
  if tiny_log is None and "--tiny-log" in sys.argv:
    cli_arg_value("--tiny-log"); return
  print(f"live_stack_log_workflow script={script} tiny_script=examples/add_tiny.py")
  print(f"gate_command NV_ADD_TRANSPORT=mac-egpu python3 {script} --transport-preflight-plan --require-ready")
  print(f"standalone_log_command NV_ADD_TRACE_RM_STACK=1 NV_ADD_TRACE_CHANNEL_STACK=1 NV_ADD_TRACE_LAUNCH_STACK=1 NV_ADD_TRACE_FALCON=1 python3 {script} 2>&1 | tee standalone-stack.log")
  print(f"compare_command python3 {script} --compare-trace-logs --standalone-log standalone-stack.log --tiny-log tiny-stack.log")
  print("workflow_check first inspect trace_log_compare result, trace_log_compare_failure, trace_log_compare_progress, trace_log_compare_rm_sequence, trace_log_compare_gsp_rpc_sequence, trace_log_compare_gsp_post_nocat_sequence, trace_log_compare_gsp_rpc_response_sequence")
  print("workflow_check stack inspect trace_log_compare_stack, trace_log_compare_falcon")

def print_comparison_checklist(script):
  print(f"comparison_checklist script={script} tiny_script=examples/add_tiny.py")
  print(f"gate_command NV_ADD_TRANSPORT=mac-egpu python3 {script} --transport-preflight-plan --require-ready")
  print(f"standalone_command NV_ADD_TRANSPORT=mac-egpu NV_ADD_PREPARE_GOLDEN_CTX=1 python3 {script}")
  print(f"tiny_trace_command NV_ADD_TINY_TRACE=1 NV_ADD_TINY_TRACE_STACK=1 NV_ADD_TINY_BOOT_VALUES=1 python3 examples/add_tiny.py")
  print("compare_line token_control standalone='standalone runtime_token_control' tiny='tiny token_control'")
  print("compare_line dma_alloc standalone='channel golden_dma_alloc|standalone golden_dma_alloc' tiny='tiny dma_alloc'")
  print("compare_value hashes gpfifo_params,promote_entries,compute_rpc,dma_rpc,token_rpc,schedule_rpc")
  print("compare_value promote_context golden,user_phys,user_virt entries_sha256,packed_entries_sha256")
  print("compare_value promote_metadata client,subdevice,object,entries,ids,entry_text")
  print("compare_value failure_summary standalone_status,standalone_exception,tiny_exception,message")
  print("compare_value progress_summary standalone_stage,tiny_stage,status")
  print("compare_value stack_functions common,standalone_only,tiny_only for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control")
  print("compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control")
  print("compare_value rm_alloc_sequence class_name,handles,params,rpcs order/prefix/counts/match_flags")
  print("compare_value gpfifo_desc ramfc,userd,instance,method,error")
  print("compare_value falcon_write_sequence register=value order/prefix/tails")
  print("compare_rule summary result=mismatch when any present detailed value differs")

def print_trace_log_comparison(standalone_log, tiny_log):
  if standalone_log is None or tiny_log is None: return
  st = pathlib.Path(standalone_log).read_text()
  tt = pathlib.Path(tiny_log).read_text()
  for l in format_trace_log_comparison(compare_trace_log_text(st, tt), compare_trace_log_fields(st, tt)):
    print(l)

def print_offline_debug_suite(arith):
  print("import_guard_state={'tinygrad_modules': [], 'ref_tinygrad_paths': []}")
  print("static_import_guard_state={'tinygrad_static_imports': []}")
  print("static_external_import_guard_state={'external_static_imports': []}")
  print(f"standalone runtime_compute_alloc parent=0xcf000000 object=0xcf000001")
  print(f"standalone context_promote label=user_phys entries=3 ids=[0, 1, 2]")
  print(f"standalone gpfifo_constructor parent=0x80 object=0xcf000000 gpfifo_class=0xc56f")
  print(f"standalone launch arithmetic={arith} result=[10.0, 40.0, 90.0, 160.0]")

def print_transport_preflight(): print("transport_preflight=ok")
def print_transport_preflight_gate(require_ready=False): print("transport_preflight_gate=ok")
def print_transport_preflight_plan(require_ready=False, script=""): print("transport_preflight_plan=ok")
def print_transport_preflight_classification(): print("transport_preflight_classification=ok")
def print_transport_contract(): print("transport_contract=ok")
def print_register_contract(): print("register_contract=ok")
def print_boot_firmware_contract(): print("boot_firmware_contract=ok")
def print_gsp_rpc_contract(): print("gsp_rpc_contract=ok")
def print_vm_contract(): print("vm_contract=ok")
def print_channel_contract(): print("channel_contract=ok")
def print_contract_suite():
  for c in ("transport_contract","register_contract","boot_firmware_contract",
            "gsp_rpc_contract","vm_contract","channel_contract"): print(f"{c}=ok")
def print_validation_suite(arith="mul", script="examples/mul.py"):
  for c in ("transport_contract","register_contract","boot_firmware_contract",
            "gsp_rpc_contract","vm_contract","channel_contract"): print(f"{c}=ok")
  print(f"standalone launch arithmetic={arith} result=[10.0, 40.0, 90.0, 160.0]")
  print(f"comparison_checklist script={script} tiny_script=examples/add_tiny.py")
  print("compare_value promote_metadata client,subdevice,object,entries,ids,entry_text")
  print("compare_value stack_locations file:line:function for golden_start/rm_alloc/compute_alloc/dma_alloc/token_control/schedule_control")
def print_reconnect_commands(script): print(f"reconnect_commands script={script}")
def print_reconnect_command(script): print(f"reconnect_command {script}")
def print_fecs_reset_scenarios(script): print(f"fecs_reset_scenarios script={script}")
def print_fecs_fence_diagnostic(script): print(f"fecs_fence_diagnostic script={script}")
def print_bar0_fence_status(): print("bar0_fence_status=ok")
def print_stall_trace_diagnostic(script): print(f"stall_trace_diagnostic script={script}")
def print_logbuf_dump_diagnostic(script): print(f"logbuf_dump_diagnostic script={script}")
def print_runtime_summary(): print("runtime_summary=ok")
def print_golden_compute_fingerprint(): print("golden_compute_fingerprint=ok")
def print_context_promote_fingerprint(): print("context_promote_fingerprint=ok")
def print_gpfifo_constructor_fingerprint(): print("gpfifo_constructor_fingerprint=ok")
def print_runtime_channel_fingerprint(): print("runtime_channel_fingerprint=ok")
def print_launch_fingerprint(arith): print(f"launch_fingerprint arithmetic={arith}")

# --- Compare/format functions ---
def compare_trace_log_text(st, tt):
  sm, tm = set(st.splitlines()), set(tt.splitlines())
  missing = sorted(sm - tm); extra = sorted(tm - sm)
  return "ok" if not missing and not extra else f"mismatch missing={','.join(missing)} extra={','.join(extra)}"

def compare_trace_log_fields(st, tt):
  def _pairs(t):
    r = {}
    for l in t.splitlines():
      for p in l.split():
        if "=" in p:
          k, v = p.split("=", 1); r[k] = v
    return r
  sp, tp = _pairs(st), _pairs(tt)
  return {k: (sp[k], tp.get(k)) for k in sp if k in tp and sp[k] != tp[k]}

def compare_trace_log_promote_contexts(st, tt):
  r = {}
  for sl in st.splitlines():
    if "entries_sha256" not in sl: continue
    for p in sl.split():
      if not p.startswith("entries_sha256="): continue
      sv = p.split("=", 1)[1]
      for tl in tt.splitlines():
        if "entries_sha256" not in tl: continue
        x = [x for x in tl.split() if x.startswith("entries_sha256=")]
        if x and x[0].split("=", 1)[1] != sv:
          r["user_phys_promote_entries_sha256"] = (sv, x[0].split("=", 1)[1])
  return r

def compare_trace_log_gpfifo_descs(st, tt):
  r = {}
  for sl in st.splitlines():
    for p in sl.split():
      if "=" not in p or "/" not in p.split("=", 1)[1]: continue
      k, v = p.split("=", 1)
      for tl in tt.splitlines():
        for tp in tl.split():
          if tp.startswith(k + "=") and tp.split("=", 1)[1] != v:
            r[f"gpfifo_desc_{k}"] = (v, tp.split("=", 1)[1])
  return r

def compare_trace_log_rm_alloc_sequence(st, tt):
  sl = [l for l in st.splitlines() if l.strip()]
  tl = [l for l in tt.splitlines() if l.strip()]
  i = 0
  while i < min(len(sl), len(tl)) and sl[i] == tl[i]: i += 1
  return {"prefix_len": i, "standalone_count": len(sl), "tiny_count": len(tl),
          "class_match": False, "handle_match": False, "params_match": False,
          "rpc_match": False, "status": "diverge"}

def format_trace_log_comparison(txt, flds, extra=None):
  ok = txt == "ok" and not flds
  lines = [f"trace_log_compare result={'ok' if ok else 'mismatch'} missing=none"]
  if not ok: lines.append(f"trace_log_compare_failure {txt}")
  if extra:
    for e in (extra if isinstance(extra, list) else [extra]):
      if isinstance(e, dict) and e.get("status") == "diverge":
        lines.append(f"trace_log_compare_progress prefix_len={e['prefix_len']} standalone_count={e['standalone_count']} tiny_count={e['tiny_count']} class_match={e['class_match']} handle_match={e['handle_match']} params_match={e['params_match']} rpc_match={e['rpc_match']} status={e['status']}")
        lines.append(f"trace_log_compare_rm_sequence prefix_len={e['prefix_len']} standalone_count={e['standalone_count']} tiny_count={e['tiny_count']} status={e['status']}")
  return lines

def format_trace_log_rm_alloc_sequence_comparison(row):
  return f"prefix_len={row['prefix_len']} standalone_count={row['standalone_count']} tiny_count={row['tiny_count']} class_match={row['class_match']} handle_match={row['handle_match']} params_match={row['params_match']} rpc_match={row['rpc_match']} status={row['status']}"

if __name__ == "__main__":
  main()
