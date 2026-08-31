# Offline entry snapshots (progress.md)

JSON register dumps for ``falcon_oracle entry-probe``. **Never sampled from this
tool** — paste values from a one-shot live probe (or use the checked-in fixtures).

| File | Classification |
|---|---|
| `night41h_replug_cold.json` | COLD_REPLUG |
| `stub_pramin_after_memif.json` | STUB_PRAMIN |
| `posted_baseline_candidate.json` | POSTED_CANDIDATE (synthetic template) |
| `night41s_after_post.json` | Live four-word fixed-PA POST activation evidence (not an `entry-probe` register dump) |
| `night41t_mid_post_virgin.json` | Live mid-POST `0x1700` sampling kept fixed-PA virgin despite Night41s-identical core MMIO |
| `night41u_after_post.json` | Live end-only POST activation reproducing Night41s-class fixed-PA data (H80 confirmed) |
| `night41v_prefix_4.json` | Live prefix k=4 activation; abb6/ac95/acfb not required |
| `night41w_prefix_2.json` | Live prefix k=2 activation; activator ⊆ {0x87e5,0x8fe8} |
| `night41x_prefix_1.json` | Live prefix k=1 virgin; 0x8fe8 necessary after 0x87e5 |
| `night41y_8fe8_stop_9e34.json` | Live 8fe8 mid-cut virgin; activator in latter half |
| `night41z_8fe8_stop_a43c.json` | Live 8fe8 0xa43c cut positive; activator in (0x9e34,0xa43c] |
| `night41aa_8fe8_stop_a138.json` | Live 8fe8 0xa138 cut virgin; activator in (0xa138,0xa43c] |
