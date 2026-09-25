# 03: Profile candidate CPU-tier backbones on the reference phones

**What to build:** pick the CPU-tier model-family backbone from measured latency, before any training. Random-weight exports of the current architecture (continuous head, exact landmarks, bounded scales, CLIFF focal) with HGNetV2-B0, B1 and B4 (anchor) at 224 px, profiled on AI Hub.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] The B4 random-weight export's latency matches the trained g8h's within noise on the S23 CPU (63 ms) — the check that random weights are a valid latency proxy
- [ ] CPU-only latency and peak memory for B0, B1 and B4 on Galaxy S23, A73, A53, A14 and Pixel 8; S23 NPU latency and placement for each
- [ ] Every export places 100% of layers on the NPU on the S23 and passes fp16 parity
- [ ] A table recorded in the status doc and a recommended CPU-tier size against the 33 ms budget on mid-range phones (A73 / Pixel 8)
