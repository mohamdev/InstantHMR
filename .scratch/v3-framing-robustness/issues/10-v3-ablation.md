# 10: The v3 ablation

**What to build:** measure what the v3 preset and a larger input buy. Three arms, 2 seeds each, on the rebuilt corpus's data mix: a g8h control (baseline preset on the same mix), v3 at 224 and v3 at 288. Current generation-8 recipe unless ticket 02 has already produced a winner.

**Blocked by:** 06, 07, 08 (09 descoped: Harmony4D added afterwards as its own measurement)

**Status:** launching (2026-10-04) as generation 9

- [ ] Launch commands printed in the launcher's form, with the rsync list from ticket 07 and a cluster-side `grep -c` check that the new trainer is in place
- [ ] Every checkpoint scored on EMDB-1 published protocol, on EMDB under 1.2x / 1.4x / 1.6x enlarged boxes, and on 3DPW test J14 with the adapter
- [ ] A table of accuracy against latency (from tickets 03 and 06), stating whether v3 reduces the EMDB published-protocol penalty by more than ~0.5 mm without losing on 3DPW
- [ ] Results recorded in the benchmark results and status docs

## Scope change (user, 2026-10-04)

- No control runs; reference is g8h_s0/s1. g9 vs g8 changes crops AND mix (no Harmony4D), so it is not a single-variable comparison.
- Arms: v3 @ 224 and v3 @ 288, each on HGNetV2-B4 and B0 (folds ticket 11's CPU-tier member in). Seed 0 only. **300 epochs** (user choice, account deadline), 4 chained 20 h links per run.
- Without Harmony4D now; Harmony4D (ticket 09, building on the laptop) is added later and its effect measured on its own.
- Launch recipe and prerequisites: STATUS.md, "Generation 9".

## Restart (2026-10-05)

First launch (data_ctx only, no Harmony4D) crashed at epoch 12 on an empty npz. Restarted from scratch on `data_all` (all five context splits incl. Harmony4D) with g8h's mix MIX7, so g9 vs g8h is now a cleaner comparison (crops + preset + backbone/input size). Old runs kept as `_ctxonly_ep12_*`. No separate Harmony4D-delta generation.

## Final scope (user, 2026-10-12)

14 runs: HGNetV2 B0..B6 x input {224, 288}, 600 epochs, v3, `data_all` (incl. Harmony4D), MIX7, seed 0. Folds ticket 11's model family into this generation. Dataset loader now skips unreadable samples (logged) instead of killing the run. Launch block and costs: STATUS.md "Generation 9, final form".
