# Falcon State Oracle

Instruction-accurate, event-driven Falcon executor for GK104 / Kepler PMU bring-up.

## Why this exists (progress.md)

Night41d–g proved the PMU pad’s `ENTER → three xdst → LEAVE` path retains
exact MEMIF loadback (`H52`: 3/3 fragments, 40/40 bytes).  Night41s then proved
that corrected Nouveau-compatible NVINIT POST activates fixed-PA PRAMIN before
RAM (Night41s/41u).  Night41t’s mid-POST `0x1700` sampling was a virgin
artifact (H80 confirmed); H79 now needs prefix cold runs with one end sample.

This oracle answers offline questions that used to burn cold runs:

- Did the pad take a rising-edge FB_PAUSE before the first `xdst`?
- Are the Night41f VRAM targets (`0x60200` / `0x40000` / `0x50000`) exact?
- Does `call 0x34` clobber `$r0` the way `nv_iowr` does?
- How does pad ENTER differ from stock `memx_func_enter`?

## Quick start

```bash
cd /path/to/nvgpu
falcon_oracle/.venv/bin/python -m pytest falcon_oracle/tests -q
falcon_oracle/.venv/bin/python -m falcon_oracle explain \
  --corpus falcon_oracle/corpus/pmu_bar1_bootstrap
falcon_oracle/.venv/bin/python -m falcon_oracle enter-diff \
  --corpus falcon_oracle/corpus/pmu_bar1_bootstrap
falcon_oracle/.venv/bin/python -m falcon_oracle loadback \
  --corpus falcon_oracle/corpus/pmu_bar1_bootstrap
falcon_oracle/.venv/bin/python -m falcon_oracle diff-semantic \
  --corpus falcon_oracle/corpus/pmu_bar1_bootstrap
falcon_oracle/.venv/bin/python -m falcon_oracle diagnose \
  --corpus falcon_oracle/corpus/pmu_bar1_bootstrap
falcon_oracle/.venv/bin/python -m falcon_oracle memx-run \
  --fixture falcon_oracle/corpus/memx/fixtures/08_enter_leave.json
falcon_oracle/.venv/bin/python -m falcon_oracle memx-lint \
  --fixture falcon_oracle/corpus/memx/fixtures/09_enter_wr32_leave.json
falcon_oracle/.venv/bin/python -m falcon_oracle cold-gate \
  --corpus falcon_oracle/corpus/pmu_bar1_bootstrap \
  --fixture falcon_oracle/corpus/memx/fixtures/08_enter_leave.json
falcon_oracle/.venv/bin/python -m falcon_oracle entry-probe \
  --snapshot falcon_oracle/corpus/snapshots/night41h_replug_cold.json
falcon_oracle/.venv/bin/python -m falcon_oracle memx-diff \
  --left falcon_oracle/corpus/memx/fixtures/08_enter_leave.json \
  --right falcon_oracle/corpus/memx/fixtures/09_enter_wr32_leave.json
falcon_oracle/.venv/bin/python -m falcon_oracle hypotheses
falcon_oracle/.venv/bin/python -m falcon_oracle lifecycle
falcon_oracle/.venv/bin/python -m falcon_oracle plan-check \
  --plan falcon_oracle/corpus/bringup/plans/night41q_h76_only.json
```

All of the above are **offline** (fake bus / JSON fixtures). Do not point this tool at live BAR0.

## Bring-up commands

| Command | Use |
|---|---|
| `explain` | Night41 summary: rising-edge, 3 XFERs, 40-byte loadback, ENTER/LEAVE MMIO |
| `loadback` | JSON report for the 40-byte MEMIF view |
| `enter-diff` | Pad ENTER vs stock MEMX ENTER (GF119 path from `memx.fuc`) |
| `diff-semantic` | Pad interpreter vs semantic bootstrap external trace |
| `diagnose` | Triage H52 offline invariants vs live H80/H79 POST path |
| `hypotheses` | Progress.md H52–H80 board + leading discriminators |
| `lifecycle` | Nouveau/Python stage matrix; BAR1 blocked without posted PRAMIN |
| `plan-check` | Validate historical H76-only shape; REFUSE H69 / H74 / full cold PRAMIN replay |
| `memx-run` | Run MEMX semantic fixtures (ENTER/WR32/WAIT/DELAY/LEAVE) |
| `memx-lint` | H29 wait bound, ENTER ownership, DMEM capacity, Night41 order hints |
| `memx-encode` | Nouveau `0x10a1c4` wire words `(size<<16)|mthd` + data |
| `memx-diff` | First divergence between two MEMX fixtures |
| `entry-probe` | Classify a **recorded** MMIO snapshot (Night41h / H70) |
| `cold-gate` | CONDITIONAL if Falcon-offline OK; NO_GO on COLD_REPLUG snapshot |

## Architecture notes

- `call 0x34` is stock `wr32` via the PMU MMIO window (`0x7a0`/`0x7a4`/`0x7ac`)
- Direct `iowr` to `0x7e0`/`0x7e4`/`0x7c0` is FB_PAUSE SET/CLR/STATUS
- Semantic MEMX script executor covers plan fixture ladder 1–9; wire encode matches `memx.c`
- Oracle cannot identify the H79 activating script or prove BAR1 — `cold-gate` / `entry-probe` stay offline
- Never promotes `0x619f04` / `0x088050` writes as a fix (H69 closed)
- H77/H80 POST activation is confirmed with end-only sampling; next *live* step is H79 `KEPLER_POST_SCRIPT_PREFIX` bisect — not run from this tool
