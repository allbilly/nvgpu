# TODO — `examples/add.py` on ADT-Link UT3G (RTX 3080, GA102, macOS, eGPU)

**Status: CLOSED — 2026-06-23.**

`python3 examples/add.py` and `python3 examples/middle_nv.py` (identical)
print `result=[11.0, 22.0, 33.0, 44.0]` on the RTX 3080 eGPU. The live
path imports only `from tinygrad.runtime.autogen import nv, nv_570, pci,
nv_regs, libc` (ctypes constants). `Device["NV"]`, `_load_tinygrad`, and
`from tinygrad.device/runtime/ops` are gone from the live path.

The standalone NV stack is fully vendored into a single 2 k-line file,
`examples/middle_nv.py`. The cubin is hand-assembled SM86 SASS (4× FADD,
2× LDG, 1× STG). Wall time per run: ~4 s, of which 3.87 s is GSP boot
and ~2 ms is the actual add kernel.

## Verify

```bash
# Tier 1 (offline, no eGPU required)
python3 examples/middle_nv.py --middle-selftest
#   -> middle_selftest=ok cubin_sha=54f9606... launch_words=20 rpc_checksum=0xc040404

# Tier 2 (eGPU required)
python3 examples/add.py
#   -> result=[11.0, 22.0, 33.0, 44.0]
python3 examples/add_tiny.py
#   -> result=[11.0, 22.0, 33.0, 44.0]
```

## Milestones (chronological)

| Step | Result | Fix |
|------|--------|-----|
| Vendored NV stack (helpers, MMIO, transport, NVReg, NVMemoryManager, NVRpcQueue, NV_GSP, NV_FLCN) | boot OK up to S12a | — |
| Initial S12c stall (FACS/PMU mutex timeout) | failed at compute alloc | root-caused to GSP RM waiting on PMU |
| S06b fix (correct INIT_DONE poll) | boot OK, S12a→S12b within 2× of tiny | corrected post-init poll |
| FACS verify (3-SHA diff at pre_compute_alloc) | confirmed standalone matches tiny | — |
| PMU promote path matches tiny | compute alloc OK | — |
| Context promotion (golden → user) | user compute gpfifo allocated | — |
| User-mode compute channel + GPFIFO setup | ring_size=65536, token=0x1 | — |
| `manual_launch` with hand-assembled SM86 cubin | first runs stalled at `wait_signal timeout got=0 want>=3` | pushbuffer byte-for-byte diff against `ref/tinygrad/dump/nv_add_dump.jsonl` |
| `data64` / `data64_le` swap fix | mem window args now `[hi, lo]` matching tiny | swapped definitions to match `tinygrad/helpers.py` |
| `synchronize` off-by-one fix | wait for `timeline_value-1` instead of `timeline_value` | first setup now passes |
| `timeline_value = 1` init fix | first `next_timeline()` returns 1, sem release writes 1 | tinygrad convention |
| Removed destructive `_signal_page_view[0] = 0` init | first release no longer races with host zero | — |
| `_copyout` byte-by-byte fix | `dest[i] = src.cpu_view()[i]` | fixes `memoryview assignment: lvalue and rvalue have different structures` |
| Bounded `synchronize()` (5 s timeout + diagnostics) | infinite hang → 5 s timeout with ring/gpput/sig_va snapshot | — |
| `setup_usermode()` reorder (before GPFIFO create) | `gpu_mmio` ready before any doorbell write | — |
| `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE` on channel_group | channel actually starts consuming | — |
| Two clean green eGPU sessions | `result=[11.0, 22.0, 33.0, 44.0]` | step 23 gate met |
| `add.py` (== `middle_nv.py`) | canonical standalone file | — |
| `add_middle.py` deleted | single source of truth | — |
| Deadcode cleanup | `add_middle.py` and all helper dups removed; ~120 KB total, 2 k lines | — |

## Key fixes (in order of discovery)

1. Bounded `synchronize()` (5 s timeout with diagnostic dump)
2. `setup_usermode()` reorder (before GPFIFO create)
3. Channel group `NVA06C_GPFIFO_SCHEDULE` after both gpfifos created
4. `data64` returns `(hi, lo)`, `data64_le` returns `(lo, hi)` — match tinygrad
5. `synchronize()` waits for `timeline_value - 1`
6. `timeline_value = 1` init (tinygrad convention; first `next_timeline()` returns 1)
7. Removed destructive signal-init zero-write
8. `_copyout` byte-by-byte copy via int indexing

## Files

```
nvgpu/
├── AGENT.md
├── README.md
├── TODO.md                  # this file
├── examples/
│   ├── add.py              # the standalone live path (== middle_nv.py)
│   ├── middle_nv.py        # vendored NV stack + cubin builder + live driver
│   ├── add_tiny.py         # frozen tinygrad health reference
│   └── mul.py              # multiply kernel (separate; same NV stack)
├── firmware/ga102
└── ref/                    # tinygrad source
```

## Historical debug (kept for context)

The bulk of the old TODO.md was a per-step timing analysis showing the
S12c stall. That stall is fixed. The original S12c timing table, the
3-SHA diff plan, and the "per_step_timing.py" snippet are not needed
for the current green path but are preserved here for reference.

The original `examples/add.py` and `examples/add.py.legacy` were a
non-green, 926 kB hand-mirror implementation. They have been deleted
along with the migration scratch (capture scripts, debug logs,
golden-image dump output, `add.cu`/`add.cubin` stubs).
