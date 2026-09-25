# 10: The v3 ablation

**What to build:** measure what the v3 preset and a larger input buy. Three arms, 2 seeds each, on the rebuilt corpus's data mix: a g8h control (baseline preset on the same mix), v3 at 224 and v3 at 288. Current generation-8 recipe unless ticket 02 has already produced a winner.

**Blocked by:** 06, 07, 08, 09

**Status:** ready-for-human

- [ ] Launch commands printed in the launcher's form, with the rsync list from ticket 07 and a cluster-side `grep -c` check that the new trainer is in place
- [ ] Every checkpoint scored on EMDB-1 published protocol, on EMDB under 1.2x / 1.4x / 1.6x enlarged boxes, and on 3DPW test J14 with the adapter
- [ ] A table of accuracy against latency (from tickets 03 and 06), stating whether v3 reduces the EMDB published-protocol penalty by more than ~0.5 mm without losing on 3DPW
- [ ] Results recorded in the benchmark results and status docs
