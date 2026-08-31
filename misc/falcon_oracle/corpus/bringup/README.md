# Offline bring-up board (progress.md)

| File | Role |
|---|---|
| `hypotheses.json` | H52–H77 status + oracle role |
| `h76_init_io_sequence.json` | Nouveau INIT_IO 0x3c3 special-case steps |
| `h75_io_bar_prefix.json` | Multi-space capture checklist (not executable here) |
| `plans/` | Example ALLOW / REFUSE live plans for `plan-check` |

```bash
python -m falcon_oracle hypotheses
python -m falcon_oracle lifecycle
python -m falcon_oracle plan-check \
  --plan falcon_oracle/corpus/bringup/plans/night41q_h76_only.json
```
