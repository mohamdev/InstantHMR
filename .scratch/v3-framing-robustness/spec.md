# Spec: framing-robust, resolution-flexible training (v3 preset) and a deployable model family

Status: ready-for-agent

## Problem Statement

The reference model (g8h) scores 51.9 mm SMPL-24 PA-MPJPE on EMDB-1 when it is given tight boxes built from the ground-truth joints, which is the framing every training crop has. Under the published protocol, which uses EMDB's own looser and off-centre boxes, it scores 64.3 / 66.9 mm (seeds 0 / 1): a penalty of 12–15 mm that puts it level with SMPLer-X-H and about 21 mm behind CameraHMR. The penalty has grown since generation 5 (+7.0 mm) and generation 6 (+8.3 mm), and it differs by 2.6 mm between two seeds that agree to 0.3 mm on tight boxes, which is the signature of behaviour that training never constrains.

Enlarging our own boxes shows where it breaks: 1.4x the tight box costs 2.2 mm, 1.6x costs 16.2 mm. 1.6x is exactly the loosest framing training can produce, and at that point training has only ever shown black borders, because every stored crop is cut at exactly 1.2x the tight box and saved at the network's 224 px input size. No pixel of real scene outside that square exists in the corpus. The v2 preset cannot fix this: it adds zoom-in (truncation), which is the opposite direction, and its zoom-in upsamples the 224 crop, training the model on blur a real crop would not have.

The same fragility hits the product. On the phone, crops come from the NanoDet detector, whose boxes are often too large, and people often stand very close to the camera and are heavily truncated.

Separately:

- The model is underfitting. Training loss sits above validation loss, and it was still falling when the learning rate reached zero. Learning rate and schedule length have never been tuned.
- The input resolution has never been varied, because the stored crops cap it at 224.
- g8h runs in 1.4–4.1 ms on Snapdragon 8 Gen 1 and newer NPUs, but in 63–196 ms on phone CPUs, against a 33 ms (30 FPS) budget. The CPU deployment tier has no model that fits.

## Solution

The corpus is rebuilt once, from original frames, with twice the context around each person and at 448 px. A new v3 preset samples each training crop from that larger stored area with a wide, bounded range of framings: loose and off-centre like EMDB's and the detector's boxes, and tight and truncated like a person close to the camera. The surrounding area is always real scene, never black. Because the stored crops have resolution to spare, the network input can be 224 or larger without upsampling.

Accuracy is judged on the published protocol, and v3 is compared against g8h on two seeds each, at 224 and at 288. Independently, and on the existing corpus, a small learning-rate x schedule-length sweep tests whether the underfitting model gains from a better recipe. The winning recipe is then trained at smaller backbone sizes. The sizes are picked beforehand by profiling random-weight exports on the same hosted phones, so the model family gives one accuracy-vs-latency curve with a member for each deployment tier.

## User Stories

1. As the researcher, I want the crop builder to accept a context factor and an output size, so that I can store crops with real scene around each person at a resolution above the network input.
2. As the researcher, I want the builder with default flags to produce byte-identical output to today's, so that the existing corpus, the baseline and v2 presets, and every recorded result stay valid.
3. As the researcher, I want each stored crop to record where the standard 1.2x window sits inside it, so that the trainer and any verification can recover today's framing exactly.
4. As the researcher, I want a verification that cuts the 1.2x window out of a rebuilt crop and compares it with today's 224 crop, so that I can prove the rebuild shows the same person in the same place.
5. As the researcher, I want the rebuilt crop's stored geometry to map the annotation targets' 2D keypoints onto the new crop to 0.000 px, so that no label is silently shifted by the rebuild.
6. As the researcher, I want the rebuilt corpus written beside the existing one rather than over it, so that runs in flight and old checkpoints keep reading the 224 corpus.
7. As the researcher, I want the four splits whose original frames are on Jean Zay (coco, sa1b, aic, mpii) rebuilt with the existing array build job, so that the rebuild reuses tooling that already works on the cluster.
8. As the researcher, I want Harmony4D rebuilt by streaming its original zips scene by scene, so that the one split without original frames rejoins the corpus at the new framing.
9. As the researcher, I want the Harmony4D streaming route chosen from the result of a download test on the `prepost` partition, so that the rebuild runs on Jean Zay when the network allows and on my laptop when it does not.
10. As the researcher, I want each rebuilt split verified (crop count, reprojection, missing images) and archived to `$STORE`, so that the 30-day `$SCRATCH` purge costs a restore, not a rebuild.
11. As the researcher, I want the AIC and MPII original frames archived before any rebuild starts, so that the purge due around 1–2 October 2026 cannot delete the only on-cluster copy.
12. As the researcher, I want a v3 preset selectable with `--preset v3`, so that it sits beside baseline and v2 with the same interface.
13. As the researcher, I want v3 to sample the effective person box over a wide range, from tight truncation to loose framing, so that the model sees the framings that EMDB boxes, detector boxes and close-up people produce.
14. As the researcher, I want v3 to shift the box centre randomly, so that off-centre framing like EMDB's 5.7% offset is part of training.
15. As the researcher, I want the v3 sampler to keep the sampled window inside the stored context, so that black borders never appear except where the real frame itself ends.
16. As the researcher, I want the v3 crop sampled directly from the stored high-resolution crop in one resampling step, so that zoom-in never upsamples a 224 intermediate.
17. As the researcher, I want the CLIFF conditioning to describe the box actually sampled in v3, so that the camera and depth placement stays correct under the new framings.
18. As the researcher, I want the network input size to be a training option (224 by default, 288 for the resolution ablation), so that I can test whether resolution buys accuracy.
19. As the researcher, I want the input size recorded in the run config and read back when a checkpoint is loaded, so that evaluation and deployment cannot silently feed a 288 model a 224 image.
20. As the researcher, I want the validation, the benchmark harness and the inference package to build their input at the checkpoint's configured size, so that the three input builders stay in agreement.
21. As the researcher, I want the ONNX export to carry the input size in its metadata, so that the phone app configures its crop size from the model file.
22. As the researcher, I want baseline and v2 samples to stay bit-identical to the current code, so that controls in flight remain valid.
23. As the researcher, I want the v3 dataset to reject a split that lacks the rebuilt geometry, with a clear message, so that a v3 run can never train silently on 224 crops with black-border zoom-out.
24. As the researcher, I want a verification of v3 samples on real data (label alignment, CLIFF correctness, no black border inside the stored context, coverage of the configured box-scale range), so that the preset is proven before any GPU hours are spent.
25. As the researcher, I want the DDP smoke test to pass for v3 at 224 and at 288, so that the cluster job cannot abort on its second step.
26. As the researcher, I want the EMDB-1 published-protocol score (EMDB's own boxes, SMPL-24 via the adapter) reported for every new checkpoint beside the 3DPW test J14 score, so that decisions use the numbers that go in the paper.
27. As the researcher, I want to score a checkpoint on EMDB under enlarged boxes (1.2x, 1.4x, 1.6x), so that I can see whether the framing cliff has moved.
28. As the researcher, I want matched g8h controls on the rebuilt corpus's data mix, so that a v3 result is attributable to the preset and not to a changed mixture.
29. As the researcher, I want v3 at 224 and v3 at 288 trained with two seeds each, so that a change above ~0.5 mm can be told from seed noise.
30. As the researcher, I want a learning-rate x schedule-length sweep (peak learning rate {6e-4, 1.2e-3} x {300, 600} epochs) on g8h with the baseline preset on the current corpus, so that I know whether the underfitting model gains from a better recipe, independently of the rebuild.
31. As the researcher, I want the existing g8h_s0 run reused as the (6e-4, 300) cell of that sweep, so that the sweep costs three new runs, not four.
32. As the researcher, I want the sweep's winning recipe carried into the v3 runs once both are known, so that the final model family uses the best recipe on the best data.
33. As the researcher, I want random-weight exports of the current architecture with HGNetV2-B0 and B1 backbones profiled on the same five hosted phones as g8h, so that the CPU-tier backbone size is chosen from measured latency before any training.
34. As the researcher, I want each model-family member profiled for latency, NPU placement and fp16 error on QNN, so that every member satisfies the portability rule before it is trained at scale.
35. As the researcher, I want the chosen CPU-tier backbone(s) trained with the v3 recipe, so that the paper's accuracy-vs-latency curve has a member for each deployment tier.
36. As the researcher, I want every architecture or input-resolution change to pass the fp16-parity and full-NPU-placement check, so that a gain on the desktop never becomes a model that falls off the NPU on the phone.
37. As the researcher, I want the published-protocol numbers, the sweep results and the family latencies recorded in the repo's results and status docs, so that the paper's tables have a single source.
38. As the phone-app developer, I want the NPU runtime libraries for Snapdragon 8 Gen 1, Gen 3 and Elite shipped alongside Gen 2's, so that the same model runs on every Snapdragon phone that can run it.
39. As the phone-app developer, I want phones without a usable fp16 NPU (Snapdragon 888 / 778G class, Exynos, Tensor) served by the CPU-tier model instead of the fake-model fallback, so that the app works on any Android phone.

## Implementation Decisions

- **Two storage parameters, one rebuild.** The builder gains a context factor (the stored square's side as a multiple of the tight box; today 1.2) and keeps its existing output-size option. The rebuild uses context 2.0 at 448 px. The standard 1.2x window then covers ~269 px of real detail. That supports 224 and 256 input without upsampling; 288 upsamples by ~7%. Defaults reproduce today's builder byte for byte.
- **Recorded geometry.** Each rebuilt annotation stores the stored square in original-frame coordinates, next to the existing 1.2x square, so any window can be recomputed exactly. Where the stored square leaves the original frame, the builder's existing padding behaviour applies, and that padding is real frame-edge truncation, not an augmentation artefact.
- **Separate corpus.** Rebuilt splits live in their own directories, named so that the baseline and v2 presets never pick them up. The pair-index fingerprint therefore does not change for runs on the old corpus.
- **v3 box sampling.** Per sample, the effective person box is drawn as a multiple of the tight box, **log-uniformly** between 0.6x (the same truncation depth as v2's 2.0 zoom-in on a 1.2x crop) and 2.0x (the full stored context). Log-uniform keeps tight and loose framings equally likely instead of over-sampling the loose end. The centre is shifted by up to +/-15% of the box on each axis. The window is clamped to stay inside the stored context, so a large box and a large shift never combine into a black border. The bounds are config values with these defaults; they are not per-run tuning knobs. v3 keeps rotation and flip as v2 has them.
- **Wide on purpose.** The zoom-out range covers EMDB's framing (average ~1.4x equivalent, with a tail past the 1.6x cliff) and the phone detector's oversized boxes. The zoom-in range covers people close to the camera. The upper bound is the stored context, so "wide" never means synthetic black margins.
- **One resampling step.** The v3 warp maps the stored crop straight to the network input, composing the box sampling with rotation and flip. No intermediate resize to 224 happens first. This is what removes v2's upsampling blur.
- **CLIFF follows the sampled box.** v3 builds the conditioning vector from the sampled box in original-frame coordinates, in the perspective-correct form (`cliff_focal`), with the crop-centre correction for zoom already verified for v2.
- **Input size is config.** A single input-size setting drives the dataset, the validation, the benchmark harness and the inference package. It is stored in the run config, recovered by the existing config-from-checkpoint path, and stamped into ONNX metadata beside the CLIFF form. A checkpoint without it means 224.
- **Presets stay orthogonal.** v3 is a value of the existing preset option. The loss-budget option is unchanged. baseline and v2 code paths are not modified beyond what is needed to route to v3.
- **Harmony4D route.** If the `prepost` download test returns HTTP 206, Harmony4D is rebuilt on Jean Zay with the existing streaming job, passing the new builder flags and keeping no 4K originals. If it fails, the same streaming script runs on the laptop, scene by scene, deleting each zip after its crops are built, and the crops are rsynced as one tar.
- **Controls and seeds.** Each v3 arm is compared against g8h retrained on the rebuilt corpus's exact data mix, with 2 seeds per arm. The acceptance target is a meaningful reduction of the EMDB published-protocol penalty with no loss on 3DPW test J14, where "meaningful" means beyond the ~0.5 mm that 2 seeds can resolve.
- **Recipe sweep runs now.** The learning-rate x schedule-length sweep uses the current corpus and the baseline preset. It shares nothing with the rebuild and starts immediately.
- **Model family sizing.** CPU-tier backbones are chosen from random-weight exports profiled on the Galaxy S23, A73, A53, A14 and Pixel 8 (CPU only), against a 33 ms budget on mid-range phones. The NPU-tier member stays at the HGNetV2-B4 size.
- **Not reopened.** RepViT stays dropped because it overflows fp16 on the NPU. Weight-decay and dropout tuning stay dropped while the model underfits; they return if training ever overfits. No large seed sweep: seed spread is ~0.2 mm at generation 8.

## Testing Decisions

A good test here observes a component from its outside, on real data, and reports a number: pixels, millimetres, milliseconds or a byte diff. It never asserts on internal variables. Four seams, three of which already exist:

- **Crop builder output (new check).** Run the builder on a small real split. Default flags: output byte-identical to the current builder's. New flags: the 1.2x window cut from the rebuilt crop and resized to 224 matches today's crop to within JPEG and resampling noise (report PSNR). The recorded geometry reprojects the annotation targets' 2D keypoints to 0.000 px. Prior art: the builder's own reprojection validation and `verify_split.py`.
- **One training sample.** For baseline and v2: a few dozen augmented samples bit-identical against `git show HEAD`, compared on alternating calls because of the ~5e-10 cold-kernel difference in the rig. For v3: 2D labels land on the right pixels; the CLIFF vector matches the exact preimage of the crop centre (as in the `--crop-centre-fix` verification: mean, p95 and max error in px); no black pixel appears inside the stored context; the drawn box scales cover the configured range; the output size equals the configured input size. Prior art: `tools/verify_cont_head.py`, `tools/verify_exact_landmarks.py`.
- **Exported model on phones.** `tools/ddp_smoke.py` for every new configuration, then ONNX export, then AI Hub profiling for latency on the five reference phones, NPU placement (must be 100% on Snapdragon 8 Gen 1+) and fp16 error. Prior art: the AI Hub profiling in the phone-app repo and this session's portability run.
- **Benchmark harness, unchanged.** EMDB-1 with `--bbox annotated` and the SMPL-24 adapter, 3DPW test J14 with the adapter, plus EMDB under enlarged boxes (1.2x / 1.4x / 1.6x) as the framing-cliff diagnostic. Evaluating a 288 checkpoint through the harness is itself the test that the input size round-trips through the checkpoint config.

## Out of Scope

- Core ML / Apple deployment. There is no Mac or iPhone yet; it is deferred to when there is.
- int8 quantization for older Snapdragon NPUs. Those phones are served by the CPU tier.
- Decoder feedback (roadmap item 4). It follows once the v3 recipe is known and is gated by the same portability check.
- Changing the loss budget, the continuous head or the exact-landmark readout.
- Weight-decay and dropout tuning, while the model underfits.
- Rebuilding EgoHumans or EgoExo4D, which are not in the current corpus.
- The phone app's own changes (stories 38–39) beyond specifying what the model family needs from it. They are tracked in the phone-app repo.

## Further Notes

- Measurements this spec rests on (2026-09-25): EMDB-1 annotated-box scores for g8h_s0/s1 are in `benchmark/results/gen8_emdb1_annotated.json`. The box-scale diagnostic: 51.95 / 54.19 / 68.16 mm at 1.2 / 1.4 / 1.6x, stride 5. Hosted-phone latencies are from AI Hub jobs on the compiled g8h model (compile job `jpvldwoz5`): NPU S22 4.1 ms, S23 2.4 ms, S24 1.7 ms, S25 1.4 ms, S21 and A73 all-CPU. CPU: S23 63 ms, Pixel 8 82 ms, A73 84 ms, A53 188 ms, A14 196 ms.
- Storage: the 448 px rebuild is roughly 4x today's crop pixels, ~250 GB on `$SCRATCH` (400 TB, 1% used), archived as one tar per split to `$STORE` (50 TiB).
- Compute: 4,385 of 50,000 V100 hours used. A 300-epoch run is at most 4 links x 20 h x 4 GPUs = 320 GPU-hours.
- Vocabulary follows `CONTEXT.md`: reference model, annotation target, baseline preset, v2 preset, published protocol, deployment tier, model family.
