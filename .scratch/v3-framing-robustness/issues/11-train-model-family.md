# 11: Train the model family

**What to build:** the paper's accuracy-vs-latency curve. The CPU-tier backbone(s) chosen in ticket 03, trained with the winning v3 setting from ticket 10 and the winning recipe from ticket 02, beside the NPU-tier (B4) member.

**Blocked by:** 02, 03, 10

**Status:** ready-for-human

- [ ] Each member scored on EMDB-1 published protocol and 3DPW test J14, 2 seeds
- [ ] Each member profiled on the reference phones: latency, NPU placement, fp16 parity
- [ ] The accuracy-vs-latency table recorded, with each member's deployment tier
