# MEMX semantic command fixtures (plan.md §18.2 ladder 1–9)

JSON scripts for ``falcon_oracle.memx_script``. Full binary ``memx.fuc``
differential (ladder 10+) waits on a Falcon-executed MEMX corpus.

| # | File | Focus |
|---|---|---|
| 1 | `01_empty.json` | empty script |
| 2 | `02_one_wr32.json` | single WR32 |
| 3 | `03_two_wr32.json` | two WR32 commands |
| 4 | `04_wr32_group.json` | serial WR32 group |
| 5 | `05_wait_success.json` | WAIT match |
| 6 | `06_wait_timeout.json` | WAIT timeout (continues) |
| 7 | `07_delay.json` | DELAY logical steps |
| 8 | `08_enter_leave.json` | ENTER; LEAVE |
| 9 | `09_enter_wr32_leave.json` | ENTER; WR32; LEAVE |
