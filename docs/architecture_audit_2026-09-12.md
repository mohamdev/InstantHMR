# Generation 7 architecture audit — 12 September 2026

**The regression-space mismatch is real and worth testing. It does not establish a particular accuracy gain. The largest overlooked architectural difference is the teacher's repeated geometry-to-image feedback inside its decoder.** A full reproduction of the output head alone will leave that difference intact.

This is an assessment, not a training change. No trainer, deployment code, existing TODO, or checkpoint was modified. The quoted generation-7 results are validation statistics supplied by the user; no generation-7 checkpoint was available in the local `runs/` directory. These probes establish representation compatibility and approximation errors, not improvements from retraining.

The wrapper in `../sam3-biomechanics/video_to_pose_pipeline` loads the implementation in `../sam3-biomechanics/sam-3d-body`. I inspected that implementation and the actual `video_to_pose_pipeline/checkpoints/sam-3d-body-dinov3/model_config.yaml` and `model.ckpt`, rather than relying on generic architectural descriptions. Its model files have no local git modifications. The wrapper also has optional encoder replacement/precision/quantization paths; they are separate from the reference architecture discussed here.

Evidence: [main measurements](audit_2026-09-12/measurements.json), [follow-up measurements](audit_2026-09-12/followup_measurements.json), [representation/geometry probe](audit_2026-09-12/probe_pathway.py), [additional geometry probe](audit_2026-09-12/probe_geometry.py). Sampling used seed 20260912: 6,000 COCO annotations and 1,000 each from AIC, MPII, Harmony4D and 3DPW, selected without replacement from sorted filenames. Full mesh probes used the first 256 selected samples per dataset, in batches of 16. All computations ran on CPU; these are not GPU latency measurements. SA-1B is not available locally and was not sampled. The 3DPW probe is a representation check on annotations, not a benchmark evaluation.

**1. What reproducing the regression pathway actually means.**

The current student emits 204 MHR parameters, 45 shape coefficients and 3 camera coordinates from one linear layer on a single token. The teacher also reads pose from one token. A single global token is therefore not itself a departure from the teacher.

| Component | Current student | Actual local DINOv3 teacher |
|---|---|---|
| Pose regression head | Linear(512, 252), including shape and camera | Two linear layers with ReLU, 1024 → 1024 → 519; camera has a separate MLP |
| Initial pose | Near-zero raw parameters | Learned initial estimate; continuous neutral-pose initialization and residual prediction |
| Global orientation | Three raw Euler values | Six values → orthogonalized rotation matrix → Euler triple |
| Local pose | Raw `6:136` | 260 continuous values → 133 values, with hand/jaw masking and slicing |
| Skeleton scales | 68 freely predicted values, then bounds | 28 coefficients → fixed affine map to 68 scales |
| Finger poses | Part of the raw 204-vector | Two separate 54-value continuous blocks, converted and inserted by joint index |
| Expression | No expression head | 72 predicted values explicitly multiplied by zero |

Source: [student model](../instanthmr_distill_train/train_distill_mhr_only.py#L886), [teacher MHR head](../../sam3-biomechanics/sam-3d-body/sam_3d_body/models/heads/mhr_head.py#L39), [actual configuration](../../sam3-biomechanics/video_to_pose_pipeline/checkpoints/sam-3d-body-dinov3/model_config.yaml).

The TODO's decomposition of the 260-dimensional representation is correct: 23 triples represented with 6D, 58 scalar angles represented with sin/cos, and six raw translation-like size channels. However, it includes finger channels that the head subsequently overwrites and a jaw triple that is discarded. The six size channels reach MHR indices `130:136`; genuine local rotation parameters occupy `6:130`, which is **124 values**, not 130.

The measured case for changing representation is:

| Probe | Result | What it establishes |
|---|---|---|
| COCO root components with absolute angle > 2.8 rad | 12.84% | Large root angles are common |
| COCO poses with an axis within 0.34 rad of ±π | 21.23% | A substantial subset is near an Euler wrap boundary |
| COCO middle root angle within 0.1 rad of ±π/2 | 2.17% | Near-singular Euler configurations occur |
| COCO local rotation absolute angle, p99 | 1.188 rad | Local angles usually do not approach wrap boundaries |
| Continuous body round-trip, all 10,000 annotations | Maximum parameter difference 2.38e-7 | Existing labels can supply continuous body targets |
| Separate continuous hand round-trip | Maximum parameter difference 1.19e-7 | Existing labels also support the teacher's distinct hand parameterization |
| GT scales projected into checkpoint's scale basis | Maximum residual < 2.2e-8 across sampled datasets | The scale subspace does not exclude these GT skeletons |

These percentages differ slightly from the TODO because the sample differs. They support its qualitative observation. They are **not** the percentage of current predictions suffering a wrap error. That requires predictions paired with targets.

A prediction just below −π and a target just below +π can represent nearby orientations while incurring a large raw Euler loss. Predicting continuous rotations and using an appropriate rotation loss addresses that mechanism. [Zhou et al.](https://arxiv.org/abs/1812.07035) provides the underlying continuity result, not a promised gain for this model.

The local-pose case is less directly established by the angle histogram: those angles mostly remain away from wraps. The full pathway is still a sensible controlled experiment, especially because the scale basis and distinct hand mapping are compatible with the GT. It would be unjustified to claim that a head port alone will close the benchmark gap. Global rotation is removed by PA alignment; improvements to it principally affect orientation-sensitive errors, with only indirect benefits to PA-MPJPE.

**Corrections required before implementing TODO item 1:**

- **Fix the root inverse convention.** The teacher calls `roma.rotmat_to_euler("ZYX", R)`. Its inverse must therefore start with `R = roma.euler_to_rotmat("ZYX", p[:, 3:6])`, then concatenate the first and second **columns** of `R`. Do not pass the stored triple directly to `batch6DFromXYZ`. A simple input `[0.3, 0.5, 0.7]` returns `[0.7, 0.5, 0.3]` through that proposed inverse/forward pair. The correct construction preserves rig geometry, including sampled noncanonical Euler targets. For a physical rotation loss, respect the rig's actual extrinsic-XYZ convention; do not silently identify a teacher latent rotation matrix with the rig rotation.
- **Pad the body target correctly.** The inverse expects 133 values: use MHR `6:136` plus three zero jaw values. Passing the 204-vector or only 130 values is wrong. Reconstruct hands using `hand_joint_idxs_left/right`, the hand conversion, and the checkpoint's hand buffers. A 339-value head that simply retains the body representation's finger channels is a different pathway from the teacher's hand overwrite.
- **Separate inactive outputs from useful behavior.** 519 pose outputs reproduce the literal layout. 447 pose outputs omit the identically disabled face block while retaining both hand blocks; camera still needs its own three outputs. Neither choice requires a hand-crop encoder. Predicting 72 values that are always zeroed cannot improve accuracy.
- **Include the MLP, initial estimate and valid continuous initialization.** All-zero 6D vectors and all-zero sin/cos pairs are degenerate. Neutral rotation initialization is not the current all-zero raw-Euler initialization. The teacher also normalizes decoder outputs before its head.
- **Keep bounds compatible with the scale subspace.** PCA coefficients are not inherently bounded. Applying the existing coordinatewise tanh to the *expanded* 68 scales generally moves the result outside the teacher's affine subspace. A faithful port needs stability constraints that respect that subspace, plus separate handling of the six flexible size channels and zero root translation. Blindly retaining the old scale remap is not an exact reproduction.
- **Use a numerically stable scale inverse.** The 28×68 checkpoint matrix has effective rank 24 in float32, three exactly zero singular values, and one tiny singular value (~1.29e-7). Do not assume an orthonormal, full-rank PCA basis or use transpose as its inverse. In the 256-sample COCO follow-up, a pseudoinverse with relative cutoff 1e-5 reconstructs scales to 1.50e-8 and the mesh to <0.00016 mm maximum vertex difference. Coefficients in null directions are not identifiable; expanded-scale supervision avoids inventing targets for them.
- **Test equivalence in geometry, not only raw angles.** Euler canonicalization can change an equivalent angle triple. All sampled local-body triples round-tripped numerically here, but that is not a universal identity over arbitrary angles. Check rotations, FK and gradients near chart singularities. Returning through Euler for FK does not magically remove all numerical singularities.
- **Treat loss changes as separate experiments.** Rotation-matrix or stable angular losses avoid wrap-dependent root penalties. Canonical 6D L2 supervision is not universally invalid, as the TODO implies, but its metric differs from physical rotation error. Likewise, SmoothL1 or unsquared Euclidean losses are not bugs merely because the paper describes other losses.

**2. The most consequential missing decoder mechanism: geometry guides the next image lookup.**

The teacher's actual configuration enables intermediate predictions, intermediate supervision, 2D keypoint tokens, 3D keypoint tokens, and keypoint updates. Its decoder does the following between layers:

1. Predict an MHR mesh and camera from the current pose token.
2. Obtain the mesh keypoints and project them into the crop.
3. Bilinearly sample image features at those projected locations and add them to the corresponding 2D keypoint tokens. Out-of-image or behind-camera points are masked.
4. Embed the predicted 2D coordinates into token positional information.
5. Subtract the pelvis midpoint from predicted 3D keypoints and embed those coordinates into 70 separate 3D keypoint tokens.
6. Let the next decoder layer use those updated tokens to revise the pose.

Your four-layer decoder only exchanges generic query features and globally attends the image memory. It predicts MHR once, after the last layer. Its SimCC coordinates are never fed back into feature sampling or the pose decoder.

This provides a concrete mechanism for improving **local body articulation**: after a provisional knee or elbow estimate, the next layer can inspect the corresponding image evidence. It is my highest-priority missing architectural mechanism beyond the parameter head. The recommendation is an inference from the implementation, not an ablation result on generation 7.

There is an important distinction: the teacher has no independent freely regressed XYZ pose head in the inspected inference code, but **it does have 70 auxiliary 3D tokens**. The student's comment explaining their removal conflates these two things. Restoring an unconstrained 70×3 head alone would not reproduce their function.

Also, the inspected callback predicts against the initial pose estimate at each layer; it does **not** add each new regression to the previous layer's predicted pose. Do not substitute conventional cumulative iterative regression and call it identical. Intermediate outputs are observable in released inference code; the configuration requests intermediate supervision, but the actual training loss implementation is not released.

Sources: [intermediate prediction callback](../../sam3-biomechanics/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py#L463), [decoder loop](../../sam3-biomechanics/sam-3d-body/sam_3d_body/models/decoders/promptable_decoder.py#L153), [2D and 3D token updates](../../sam3-biomechanics/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py#L1759).

The reference decoder also uses six layers at width 1024, versus four at width 512 in the student, repeats positional information within its layers, and applies a final normalization. Attention and feed-forward dimensions differ too, so width alone does not describe the cost. Its two-way image-to-token attention is explicitly disabled: adding that is not necessary for fidelity. The MLP/final-normalization changes are inexpensive candidates; none has a measured generation-7 gain yet.

Implementation is more substantial than replacing a linear head: expose decoder stages, run differentiable FK and projection between them, update query features, and supervise useful intermediate predictions. Reuse the existing image encoding. The current deploy graph returns parameters and runs rig geometry outside that graph; moving FK and feature sampling inside decoding changes that export boundary. Validate the intended ONNX/mobile runtime before committing to the full design. A smaller two-stage prototype is a useful cost/accuracy experiment, but is not full teacher reproduction.

**3. Image features are much coarser, beyond the crop-resolution issue already noted.**

The actual DINOv3 checkpoint uses 512×512 crops and 16-pixel patches: a 32×32 image feature grid. Both current student backbones produce a **7×7** final feature grid at 224 input; I verified their forward outputs. This is 1,024 versus 49 spatial tokens. These are memory-grid sizes, not FLOP ratios.

A stored-image resolution increase and a feature-stride change solve different problems. Rebuilding crops at higher resolution restores source detail. Using a stride-16 feature map or fusing an earlier backbone stage preserves more of that detail for the decoder. At 224 input, stride 16 gives a 14×14 grid without increasing input pixels. That is a particularly relevant experiment if adding joint-local feature sampling: sampling a 7×7 map remains coarse even with a sophisticated decoder.

The teacher also has large-scale DINOv3 visual pretraining and substantially greater model capacity. Exact head mechanics cannot establish that this representation advantage has disappeared. Pose-oriented pretraining or feature distillation into the compact encoder is a plausible separate experiment. Ground-truth supervision remains the current training source; feature distillation would be an additional signal.

One correction to TODO item 2: upsampling a 224 crop does not add image information, but that does not make training a larger input **necessarily pointless**. It can change effective feature stride and optimization. The clean experiment separates source resolution from network sampling density. Also, the current dataset resizes to the network input **before** its affine augmentation; merely storing 320-pixel crops would not preserve their detail through a zoom. Warp/crop from the higher-resolution source directly into the final network input.

**4. The skeleton-only landmark approximation hides body-surface information.**

The teacher forms its 70 landmarks from both mesh vertices and skeletal joints. The student uses an affine 70×127 skeleton regressor. This approximation is accurate for most fingers but less accurate for several body-surface landmarks.

| Dataset, 256 samples each | All 70 landmarks: mean error | 30 nonfinger landmarks: mean error |
|---|---:|---:|
| COCO | 1.263 mm | 2.933 mm |
| AIC | 1.322 mm | 3.071 mm |
| MPII | 1.408 mm | 3.273 mm |
| Harmony4D | 1.616 mm | 3.750 mm |
| 3DPW annotations | 1.421 mm | 3.300 mm |

These compare two landmark operators applied to the **same GT parameters**, using the local rig and teacher checkpoint mapping. No learned student prediction enters the comparison. In COCO, the worst individual discrepancy is 38.82 mm; mean discrepancies include small-toe tips at 8.8–9.1 mm and acromion landmarks at 6.7–6.8 mm. The exact operator is also closer to the stored COCO annotations: 0.175 mm mean versus the approximation's 1.242 mm. The small nonzero exact residual means the local rig/mapping is not perfectly identical to the annotation generator.

An additional distinction matters for shape: changing the 45 shape coefficients cannot move the skeleton, but **can** move the teacher's mesh-derived landmarks. Zeroing shape on these COCO samples moves the nonfinger exact landmarks by 1.05 mm mean and up to 11.56 mm. Thus “shape has no geometric gradient” is a property of the student's skeleton-only landmark path, not of the teacher's full geometry. Generation 7's vertex loss already provides a shape gradient; this proposal adds accurate landmark locations and their gradients.

The checkpoint's first 70 keypoints reference only **468 distinct mesh vertices**. You could extend the existing subset-skinning mechanism to include those exact indices, plus any vertices needed by the current vertex loss, and apply the sparse teacher mapping. This avoids making all 18,439 vertices mandatory at every decoder stage. Implementation feasibility follows from the current subset-skinning code; its combined runtime was not benchmarked here.

This is a measurable modeling approximation worth testing. Its error is neither a guaranteed benchmark gain nor a quantity that should be subtracted from PA-MPJPE.

**5. Camera conditioning and loss definitions still differ.**

The teacher supplies per-location camera-ray embeddings to the image memory, as well as CLIFF information to the pose token. The student adds one learned projection of three CLIFF values to all queries. Dense ray conditioning gives each image feature its viewing direction explicitly; it is a plausible improvement for perspective-sensitive poses. For ideal square crops with known centered intrinsics, much of that information is already derivable from CLIFF plus grid position, so it is an inductive bias, not automatically extra information.

The teacher separately regresses a crop camera `(scale, tx, ty)` and converts it using focal length and bbox geometry to full-camera translation. Your model directly regresses translation in metres. That is another regression-pathway difference omitted from TODO item 1. Its most direct benefits would concern depth and placement rather than PA-MPJPE. Source: [camera head](../../sam3-biomechanics/sam-3d-body/sam_3d_body/models/heads/camera_head.py#L83), [ray conditioning](../../sam3-biomechanics/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py#L1027).

For the specific generation-7 command, `--crop-centre-fix` is a **no-op under baseline** because `cliff_follows_aug` remains false. The affine zoom/translation therefore does not get the same conditioning update as the v2 path. This can be tested independently of adopting every stronger v2 augmentation.

The [paper's training section](https://arxiv.org/html/2602.15989v1#S4) describes L1 keypoint losses with learned uncertainty, pelvis/wrist normalization, annotation-dependent hand weights, parameter L2 and joint-limit penalties. The released code cannot establish their full formulas or weights. Your generation-7 configuration already uses **unsquared Euclidean** 3D error, not the default SmoothL1 listed in the TODO; shape weight is 0.03 and finger weight 0.2.

Two verified loss differences deserve attention:

- **No explicit pelvis/wrist centering in the main 3D loss.** It compares `pred_kp3d - tgt_3d` directly. `fk_scale` is only a training warm-up multiplier. Rig-origin coordinates are not equivalent to subtracting each sample's predicted and target hip midpoint. In the COCO probe, the GT hip midpoint lies about 907 mm from the rig coordinate origin. Separate wrist-relative hand losses would prevent arm-placement error from dominating finger articulation supervision. Retain an additional placement-sensitive term if adding centered losses.
- **A naive joint-limit penalty conflicts with the annotations.** All 6,000 COCO samples violate at least one stored limit among the 124 local rotation parameters, using a tolerance of 1e-5 rad. About 20.23% of sampled parameter entries violate their limits; the maximum violation is 0.817 rad. There are 24 zero-width local-rotation limits. For example, `spine0_rx_flexible`, `l_talocrural_rx_flexible` and `r_subtalar_rz_flexible` have `[0,0]` limits with positive penalty weights, yet nonzero GT values. Do not apply all these limits unmodified. Use an annotation-compatible soft prior, or exclude the conflicting constraints and assess the others.

Parameter supervision also remains masked by augmentation: with the generation-7 defaults, the root parameter loss sees about 20% of samples, and local-pose, scale, shape and vertex terms see about 60%. Geometry still supervises the other samples, so this is not a complete loss of their training signal. Correctly transforming parameter targets under rotations/flips could increase coverage, but requires rig-aware transformations and is not proven to match the unreleased teacher training pipeline.

Learned uncertainty and hand-availability weighting are sensible experiments, particularly on ambiguous annotations. They need a proper probabilistic objective or regularizer; simply allowing a learned weight to fall toward zero can evade the task. Matching the teacher's stated norm is a lower priority than fixing demonstrated mismatches, and should not be bundled with the head ablation.

**6. A tiny hand encoder is feasible; body gains need a feedback mechanism.**

The teacher **shares one image encoder** across body and hand crops and uses separate body and hand decoders. Full inference runs body estimation, recrops each hand from the original image, estimates the hands, prompts the body decoder with hand wrists and body elbows, and then merges hand parameters with wrist-frame conversion and validity checks. It reuses cached body image embeddings for the prompted body refinement. It also updates selected hand scale and shape components. Source: [full inference and fusion](../../sam3-biomechanics/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py#L1197).

Your proposed separate HGNetV2-B0 encoder is a reasonable compact alternative, though it is not the teacher's weight-sharing design. The installed timm provides B0 through B6; B0 has **1,850,396 parameters** after replacing its classifier head with Identity, and yields `(B,1024,7,7)` features at 224 input. A 256-wide, two-layer hand decoder shared by left and right crops is a reasonable initial experiment, not a measured optimum. Batch the two hands together and normalize left/right conventions consistently.

The main implementation work is:

1. Obtain original-resolution hand crops and labels. In the sampled COCO body crops, the projected hand extent averages only **15.4 pixels** at 224 input; Harmony4D averages **10.5 pixels**. Enlarging those saved patches cannot recover their original detail. Extents come from annotation coordinates, not a guarantee of visibility. Use original frames or store separate hand crops, and retain mappings needed to recreate their camera coordinates.
2. Add crop jitter and annotation/visibility gates so training does not assume perfect hand boxes. The existing 2D joints can bootstrap boxes; a learned detector and confidence output are an additional option.
3. Predict hand articulation plus wrist orientation in a consistent frame, with wrist-relative geometry supervision. Existing MHR GT supports articulation targets; hand-only normalization, flips and wrist transforms still need to be implemented.
4. Convert predicted global wrist orientation into the body's local wrist frame, gate unreliable estimates, and preserve the body's fallback prediction.
5. To improve the **body pose**, feed hand evidence back into body estimation. Reproduce the teacher's wrist/elbow prompting or train a smaller body-refinement stage. Merely overwriting finger parameters does not improve hip, knee or torso estimates by kinematics.

Measured check: zeroing all 54 finger articulation parameters on 256 GT COCO samples changes the teacher's 30 nonfinger landmarks by only **0.102 mm mean**, while strongly moving fingers. The student's approximate landmark operator reports 1.298 mm body movement for the same intervention, showing some artificial finger-to-body coupling in that operator. Neither number is a measured gain from a trained hand network. SMPL conversion and body-metric adapters can also introduce small indirect effects.

Expect the strongest benefit in hand articulation and full-mesh quality. A body-J14 gain is plausible through better wrists, arm refinement or shared training, but is not guaranteed by adding a hand encoder. Matching the complete teacher fusion is substantial work; adding a basic finger-only refinement branch is much simpler. Hand-specific training diversity also matters, beyond branch size.

**7. Suggested order of experiments.**

| Order | Experiment | Main quantity it should improve or clarify |
|---|---|---|
| 1 | Correct full continuous head, distinct hand blocks, MLP/init and compatible scale constraints | Parameter fitting, orientation, scale consistency; establishes pathway baseline |
| 2 | Exact mesh-derived 70 landmarks using supported vertex subset | Removes a measured landmark approximation and restores surface-sensitive gradients |
| 3 | Geometry feedback between decoder stages, including 3D token updates | Local body articulation; strongest missing body-specific mechanism |
| 4 | Denser image features and source-aware cropping, tested separately | Local image evidence and spatial localization |
| 5 | Camera-ray conditioning and explicit pelvis/wrist-relative terms | Perspective handling and separation of articulation from placement |
| 6 | Dedicated hand crops/branch, then body refinement from wrists | Hands first; body improvement depends on refinement |

These are priorities based on verified differences and mechanisms, not a ranking measured by training. Keeping the head experiment separate from decoder feedback is compatible with fully porting the head; it makes the result interpretable. Preserve generation 7 as the control and compare 2–3 matched seeds at comparable optimization budgets. The head and exact-landmark tests can use existing annotations; high-detail recropping requires original imagery. Measure body and hand errors separately, plus mesh error and absolute placement where relevant.

The teacher's broader training mix includes additional multiview, synthetic and hand-specific datasets. Your five-source mix does not reproduce that coverage. The provided mix weights total 0.70 and the sampler renormalizes them: **SA-1B 42.86%, AIC 25.71%, Harmony4D 14.29%, COCO 12.86%, MPII 4.29%**, assuming all five are staged. Those are the actual probabilities, not 30/18/10/9/3 percent. Unlisted folders receive zero sampling weight. Source: [mixture implementation](../instanthmr_distill_train/train_distill_jz.py#L277).

Finally, the supplied numbers are 3DPW validation J14+adapter records, not published test results. The summarizer also takes independent minima across epochs/RAW/EMA, so the displayed PA/MPJPE pair may not belong to one checkpoint. Establish proximity to published methods using one selected checkpoint, the matching test split, joint/conversion convention and bbox protocol; evaluate EMDB independently. Representation round-trips and the older 62.90-versus-31.84 fitting audit do not establish the achievable improvement from this head or a numerical gap attributable to it.

To rerun the two main probes in the existing environment:

```sh
python docs/audit_2026-09-12/probe_pathway.py
python docs/audit_2026-09-12/probe_geometry.py
```

The first writes `measurements.json`; the second prints the focused follow-up results. `scale_rank` in the first JSON uses float64's default tolerance (25); the effective float32 rank (24) and stable truncated-inverse check are recorded separately. Raw parameter maxima and geometric maxima use different sample counts, as specified above. The extra limit and backbone observations are recorded in `followup_measurements.json`.
