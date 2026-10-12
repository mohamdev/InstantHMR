# 06: Profile a 288 px input on the reference phones

**What to build:** decide whether the 288 px arm is deployable before GPU hours are spent on it. A random-weight B4 export at 288, profiled like ticket 03.

**Blocked by:** 05

**Status:** done (2026-09-25), uncommitted

- [x] NPU latency on Galaxy S22, S23, S24 and S25; CPU latency on S23, A73, A53, A14 and Pixel 8
- [~] 100% NPU placement on all four Snapdragon phones (yes); fp16 parity deferred to trained weights (ticket 11), as in ticket 03
- [x] Result recorded next to ticket 03's table, with the 224-vs-288 latency ratio per phone

## Answer

Random-weight exports at 288 px (`tools/profiling/export_family.py --image-size 288`), same method as ticket 03. Raw rows: `benchmark/results/family_latency_aihub_288.json`.

| model @ 288 | S22 NPU | S23 NPU | S24 NPU | S25 NPU | S23 CPU | A73 CPU | Pixel 8 CPU | A53 CPU | A14 CPU |
|---|---|---|---|---|---|---|---|---|---|
| B4 | 5.5 | 3.27 | 2.31 | 1.88 | 93.3 | 119.0 | 115.0 | 288.7 | 294.2 |
| B0 | 3.7 | 2.05 | 1.31 | 1.00 | 29.3 | 38.2 | 32.1 | 93.5 | 79.0 |

Cost of 288 over 224 (same backbone): B4 x1.32 on the S23 NPU, x1.46-1.55 on CPUs; B0 x1.28 NPU, x1.07-1.23 CPU. All exports 100% on the NPU.

- **The 288 arm is deployable on the NPU tier:** B4 at 288 is 1.9-5.5 ms across Snapdragon 8 Gen 1-Elite, far inside 33 ms. Ticket 10 may train it.
- **Not for the CPU tier with B4** (93-294 ms). B0 at 288 stays near budget on mid-range CPUs (Pixel 8 32.1, A73 38.2) because below B4 the decoder, not the image, dominates cost -- a 288 CPU-tier member is plausible if 288 proves worth it.

## Addendum (2026-10-04): 384 and 448 px

Same method; 16 of 24 profiles returned (B4@448 and two B0@448 phones missing: run timed out). Raw: `benchmark/results/family_latency_aihub_384_448.json`.

| model | input | S23 NPU | S23 CPU | Pixel 8 CPU | A73 CPU | A53 CPU | A14 CPU |
|---|---|---|---|---|---|---|---|
| B4 | 384 | 4.18 | 146 | 158 | 186 | 445 | 454 |
| B0 | 384 | 2.31 | 37 | 46 | 50 | 107 | 105 |
| B0 | 448 | 2.81 | 42 | - | 59 | 130 | - |

NPU tier affords 384+ (B4 4.2 ms, 100% NPU). CPU tier is capped at 224-288 (B0 over 33 ms on mid-range from 384). Above ~270 px the current context corpus (2.0x box at 448 px) has no extra detail: 384 needs a ~640 px rebuild. Fast SAM 3D Body (Table 4): PA-MPJPE 33.5 @256 -> 31.0 @384 -> 31.0 @448, i.e. saturation at 384.
