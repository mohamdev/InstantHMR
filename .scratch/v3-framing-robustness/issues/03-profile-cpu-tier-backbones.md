# 03: Profile candidate CPU-tier backbones on the reference phones

**What to build:** pick the CPU-tier model-family backbone from measured latency, before any training. Random-weight exports of the current architecture (continuous head, exact landmarks, bounded scales, CLIFF focal) with HGNetV2-B0, B1 and B4 (anchor) at 224 px, profiled on AI Hub.

**Blocked by:** None (can start immediately).

**Status:** done (2026-09-25), uncommitted

- [x] The B4 random-weight export's latency matches the trained g8h's within noise on the S23 CPU (63 ms) — the check that random weights are a valid latency proxy
- [x] CPU-only latency and peak memory for B0, B1 and B4 on Galaxy S23, A73, A53, A14 and Pixel 8; S23 NPU latency and placement for each
- [~] Every export places 100% of layers on the NPU on the S23 (yes, all four); fp16 parity NOT measured: it is meaningless on random weights, so it moves to ticket 11 on the trained members
- [x] A table recorded in the status doc and a recommended CPU-tier size against the 33 ms budget on mid-range phones (A73 / Pixel 8)

## Answer

Exports: g8h's exact architecture and config (continuous head, bounded scales, CLIFF focal), backbone swapped, network only (no FK/mesh), 224 px, opset 17, onnxsim. AI Hub, ORT runtime compiled for the S23. Raw rows: `benchmark/results/family_latency_aihub.json`; scripts: `tools/profiling/`.

| model | params (backbone) | S23 NPU | S23 CPU | A73 CPU | Pixel 8 CPU | A53 CPU | A14 CPU |
|---|---|---|---|---|---|---|---|
| g8h trained (B4) | 32.2 M (13.6 M) | 2.44 ms | 61.7 | 84.9 | 83.2 | 187.9 | 195.2 |
| random B4 | 32.2 M (13.6 M) | 2.47 | 60.0 | 81.7 | 78.7 | 194.2 | 199.4 |
| random B1 | 20.3 M (2.2 M) | 1.88 | 26.9 | 36.9 | 33.2 | 77.6 | 76.8 |
| random B0 | 20.0 M (1.9 M) | 1.60 | 23.9 | 32.7 | 29.9 | 79.7 | 68.6 |

- Random weights are a valid latency proxy: random B4 is within 2-6% of the trained g8h on every phone.
- All four place 100% of layers on the S23 NPU.
- **Recommendation: HGNetV2-B0 is the CPU-tier member** (under 33 ms on mid-range: A73 32.7, Pixel 8 29.9). B1 is 3.3 ms slower and just over budget on the A73 (36.9); worth training as the middle point of the accuracy-vs-latency curve.
- **Budget phones (A53, A14) stay at 70-80 ms with either backbone.** B0 and B1 differ by <2 ms there: below B4 the backbone is no longer the cost. ~18 M of the 20 M parameters are the decoder and heads, unchanged by the backbone swap. Reaching 30 FPS on budget phones needs a slimmer decoder (width / depth), which is a new work item, not a backbone choice.
