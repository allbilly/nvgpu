#!/usr/bin/env python3
"""Standalone NV multiply example built on the shared add.py backend."""

from add import NvBackend, SimpleProgram, apply_default_live_run_env, build_cubin, cli_arg_value, compare_trace_log_fields, compare_trace_log_gpfifo_descs, compare_trace_log_promote_contexts, compare_trace_log_rm_alloc_sequence, compare_trace_log_text, format_trace_log_comparison, format_trace_log_rm_alloc_sequence_comparison, import_guard_state, make_simulated_backend, manual_launch_mul, print_boot_firmware_contract, print_channel_contract, print_comparison_checklist, print_context_promote_fingerprint, print_contract_suite, print_debug_help, print_golden_compute_fingerprint, print_gpfifo_constructor_fingerprint, print_gsp_rpc_contract, print_import_guard, print_launch_fingerprint, print_live_debug_commands, print_live_log_workflow, print_live_stack_log_workflow, print_offline_debug_suite, print_reconnect_command, print_reconnect_commands, print_register_contract, print_runtime_channel_fingerprint, print_runtime_summary, print_trace_log_comparison, print_transport_contract, print_transport_preflight, print_transport_preflight_classification, print_transport_preflight_gate, print_transport_preflight_plan, print_validation_suite, print_vm_contract, static_external_import_guard_state, static_import_guard_state
import contextlib
import io
import pathlib
import struct
import sys
import tempfile


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
    assert "standalone runtime_compute_alloc parent=0xcf000002" in offline_debug_text
    assert "standalone context_promote label=user_phys entries=3 ids=[0, 1, 2]" in offline_debug_text
    assert "standalone gpfifo_constructor parent=0xcf000000 object=0xcf000002 gpfifo_class=0xc56f" in offline_debug_text
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


if __name__ == "__main__":
  main()
