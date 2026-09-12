# Open work

Cleared on 12 September 2026. The previous 884 lines were the 2026-09-06 audit
backlog; most items had been implemented, superseded, or disproved, and the file
had stopped being a reliable statement of what is open. Recover it with
`git show HEAD~1:docs/todo.md` if you need an old entry.

Two rules that kept the old file honest and are worth keeping:

- **State the measurement, not the intuition.** An item is worth writing down
  once there is a number attached to it — what was measured, on what data, and
  what the result was.
- **Delete an item when it lands or dies.** A file of stale entries is worse
  than an empty one, because it looks authoritative.

Reference material: `benchmark/README.md` (evaluation protocol, conversion
routes, measured floors, gen-6/7 tables), `instanthmr_distill_train/README.md`
(label source, presets, size bounds, vertex term), `CLAUDE.md` (repo traps),
`docs/accuracy_audit_2026-09-06.md` (the original audit, as a dated record).

**Architecture constraint, confirmed 2026-09-12:** real-time inference on
smartphones is a primary contribution. Keep efficient encoders such as HGNetV2
and RepViT; DINOv3 is a reference, not a proposed deployment backbone. Evaluate
architecture changes against the current model's latency, memory and operator
support on the intended phone runtime. No acceptable slowdown has been
quantified yet; do not assume a larger model or a repeated decoder is affordable
from desktop GPU timings alone. Training-only changes should be distinguished
from additions to the deployed inference path.

The [2026-09-12 architecture audit](architecture_audit_2026-09-12.md) records
code references and GT probes for the items below. Its accuracy opportunities
are hypotheses unless accompanied by a matched training ablation.

---

# 1. Reproduce the teacher's regression pathway exactly

**Status: implemented 2026-09-12, launched as generation 8.** Behind
`--cont-head`, default off; the default path is verified bit-identical (1544
state_dict keys, same sha256, forward and all 12 loss terms 0.000e+00 apart on
48 real augmented samples under both `--losses legacy` and `rebalanced`). Code
in `instanthmr_distill_train/mhr_cont.py` (the vendored conversions) and
`ContMHRHead` in `train_distill_mhr_only.py`; asset built by
`tools/build_cont_head_assets.py`; gated by `tools/verify_cont_head.py`.
Measurements are in `datasets_pipeline/jeanzay/STATUS.md`, "Generation 8".

**One deviation from §1.3 below, and it is deliberate.** That section says to
build the root inverse with `roma`'s `"ZYX"`, matching the teacher's literal
call. Measured against the rig -- `checkpoints/mhr_model.pt`, md5-identical to
the teacher's own `assets/mhr_model.pt` -- driving `model_params[3:6]` and
rigid-fitting the 125 joints it moves recovers **extrinsic XYZ**
(`batch6DFromXYZ`) to 0.00000-0.014 deg, while `roma.euler_to_rotmat("ZYX", .)`
of the same triple is **36-92 deg away**: roma's uppercase `"ZYX"` returns the
triple Z-first, so it is the reverse of the rig's. Either convention
round-trips, so both would train; this one is used because only then is the
intermediate rotation matrix the body's actual root rotation, which is what
makes the new chordal loss a physical angular error instead of a distance in a
permuted space. §1.3's own instruction -- "the two conventions must not be
silently conflated" -- is the reason.

**Decision taken 2026-09-12:** reproduce the
teacher's pathway *in full*, not a measurement-guided subset. The point is to
remove the pathway as a confound, so a partial port would leave exactly the
question it is meant to settle.

## 1.1 The problem, stated precisely

The student and SAM 3D Body feed MHR the same thing — the rig-native 204-vector —
but they **regress in different spaces**, and the student's is the harder one.

The teacher's head emits **519** numbers in a continuous space and converts them
to the 204-vector *inside the head*, before MHR sees anything. Verified in
`sam_3d_body/models/heads/mhr_head.py`, `forward()` (lines ~284-326) and
`mhr_forward()` (lines ~198-215):

| teacher head emits | dims | conversion applied inside the head | reaches MHR as |
|---|---|---|---|
| global rotation, 6D | 6 | `rot6d_to_rotmat` → `roma.rotmat_to_euler("ZYX", ·)` | 3 Euler angles |
| body pose, continuous | 260 | `compact_cont_to_model_params_body` → 133, then `[:130]` | 130 |
| shape | 45 | passes through raw | 45 |
| bone scales, PCA | 28 | `scale_mean + coeffs @ scale_comps` | 68 bone scales |
| hands (2 x 54) | 108 | `replace_hands_in_pose` into the body block | — |
| face | 72 | multiplied by 0 in `forward()` | — |

The 260-dim body space decomposes as **23 three-DoF joints x 6D (138)** +
**58 one-DoF hinges x (sin, cos) (116)** + **6 one-DoF translation channels, raw
(6)**. Those 6 raw channels are the `*_length/_width_flexible` parameters and
land at `130:136` of the 204-vector — i.e. **the continuous representation does
not bound them**, so they still need separate size constraints. Bounds on the
expanded PCA scales must respect the scale subspace; see §1.3.

The student instead emits the 204-vector directly from one `nn.Linear`:
`head_global = nn.Linear(d_model, model_params_dim + shape_dim + cam_dim)` =
204 + 45 + 3 = 252 numbers, all off a single query token
(`shared_feats[:, 0, :]`). So it regresses raw Euler angles and 68 free bone
scales where the teacher regresses 6D, sin/cos and 28 PCA coefficients.

**The teacher uses a two-layer MLP and an initial pose estimate**, whereas the
student uses a single linear layer. Reproduce the MLP and prediction relative
to that estimate at a width compatible with the mobile budget. Valid continuous
initialization matters: all-zero 6D vectors and all-zero sin/cos pairs are
degenerate. Use a valid neutral-pose encoding, including identity rotations.

## 1.2 Why this is not merely cosmetic — measured

Over 6,000 `data/sam3d_gt_coco/annotations/*.npz` ground-truth samples:

| parameter block | range | p99 \|angle\| | fraction beyond 2.8 rad |
|---|---|---|---|
| root rotation `3:6` | spans ±π: [−3.187, +3.361] | 3.13 rad | **13.59 %** |
| local joints `6:130` | max 2.908 | **1.18 rad** | **0.0016 %** |

- **22.32 %** of samples have at least one root axis within 0.34 rad of ±π,
  near a wrap boundary of the target.
- The root's middle Euler axis spans the **±π/2 singularity** (max 1.6226 vs
  π/2 = 1.5708). At gimbal lock the first and third angles become degenerate;
  the range alone does not prove a sample lies exactly there. **2.25 %** are
  within 0.1 rad of it, **6.42 %** within 0.2 rad.
- The 124 local rotation values in `6:130` essentially never wrap
  (p99 1.18 rad). This measures target angles, not an accuracy gain from changing
  their representation. The decision remains full reproduction.

**Near-boundary targets do not prove current predictions cross that boundary.**
Nor do they establish an expected millimetre improvement. That requires paired
predictions and targets, followed by matched training ablations. Root-orientation
improvements mostly affect MPJPE and orientation quality; PA alignment removes
global rotation. Any improvement in PA-MPJPE from fixing root regression is
indirect, not a guaranteed consequence of the angle histogram.

Zhou et al., *On the Continuity of Rotation Representations* (CVPR 2019) is the
reference for why a discontinuous target costs accuracy.

## 1.3 What to implement

Head emits 519 pose dims, or 447 if only the identically disabled 72-dim face
block is omitted (**record that choice**), plus a separate 3-dim camera output.
It then converts to the 204-vector with the
teacher's own functions so the two pathways are identical:

- `sam_3d_body.models.modules.rot6d_to_rotmat` + `roma.rotmat_to_euler("ZYX")`
- `sam_3d_body.models.modules.mhr_utils.compact_cont_to_model_params_body`
- `scale_mean + coeffs @ scale_comps` (both buffers are already in the
  teacher's state dict; they must be copied into the student or loaded from the
  rig)
- the separate hand conversion and `replace_hands_in_pose`, using the teacher's
  hand buffers and joint-index mappings

**The separate hand parameter blocks matter even without a hand encoder.** They
overwrite the body representation's finger channels. Omitting that overwrite is
not the same pathway. A 339-dim head retaining fingers only through the body
representation would therefore be a partial port; a dedicated hand-crop encoder
is not required to reproduce the full head.

**Correct root inverse:** do not apply `batch6DFromXYZ` directly to the stored
root triple. The teacher decodes with `roma.rotmat_to_euler("ZYX", R)`, so build
the inverse with the matching convention and concatenate matrix **columns**:

```python
R = roma.euler_to_rotmat("ZYX", mhr_model_params[:, 3:6])
root_6d = torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)
```

The rejected inverse returns `[0.7, 0.5, 0.3]` from `[0.3, 0.5, 0.7]` through
the teacher's forward conversion. For the body inverse, pass `6:136` plus three
zero jaw values to `compact_model_params_to_cont_body` (133 inputs). Hand
targets use their own joint ordering and inverse. These targets all come from
the existing GT 204-vector; no new annotation pass is needed. Test root rotation
losses against the rig's actual rotation convention as well as this head-space
round-trip; the two conventions must not be silently conflated.

**Existing scale bounds cannot simply be applied after PCA expansion.**
Coordinatewise tanh generally moves the expanded scales outside the teacher's
scale subspace. Preserve that subspace when designing bounds, for example by
constraining coefficients with validated limits; still constrain the six
flexible size channels separately and preserve zero root translation. PCA
coefficients are not inherently bounded. The checkpoint's 28x68 scale matrix
has effective float32 rank 24, so use a stable truncated pseudoinverse if
constructing coefficient targets, not its transpose. The audit's GT projection
residual is <2.2e-8; retaining the subspace is compatible with sampled GT scales.

Every FK / geometric loss stays where it is and operates on the converted 204,
with gradients flowing back through the conversion. Prefer a rotation-space
metric over raw Euler differences for root supervision. Canonical 6D L2 targets
are possible, but their metric is not physical angular error: multiple 6D
vectors encode the same rotation after orthogonalization. Changing the output
representation and changing other loss norms are distinct experiments.

Gate it behind a config flag whose default reproduces today's behaviour, and
prove bit-identity of the old path (`--preset baseline` and `--losses legacy`
are controls for runs in flight).

## 1.4 Loss and regularisation differences — configuration matters

The public SAM 3D Body release is **inference-only** — no training code, no loss
definitions. The following is from the paper (§ *Model Training*), so treat it as
the teacher's stated design, not as verified code:

> *2D/3D Keypoint Loss: We supervise 2D/3D joint locations using an L1 loss,
> incorporating learnable per-joint uncertainty to modulate the loss based on
> prediction confidence. For 3D body and hand keypoints, we normalize them with
> their respective pelvis and wrist locations before computing the loss.*
>
> *Parameter Losses: MHR parameters (pose, shape) are supervised with L2
> regression losses, and joint limit penalties are imposed to discourage
> anatomically implausible poses.*

**Already implemented in `--losses rebalanced`, including generation 7:**
`apply_rebalanced_losses` sets `kp3d_loss="euclid"` and
`w_keypoints3d=0.35`. This is unsquared Euclidean distance in metres, not
SmoothL1 and not squared L2. **Do not re-fix the default SmoothL1 behavior as if
generation 7 still used it.** The legacy/default configuration keeps SmoothL1
for reproducibility. Switching the rebalanced path to the paper's coordinatewise
L1 would be a new ablation, not completion of a missing fix.

Deltas against the student today (deferred items are not implementation tasks):

| item | teacher (paper) | student now | action |
|---|---|---|---|
| parameter loss type | **L2** on MHR parameters | `SmoothL1`, `pose_beta` 0.05 rad (`_m_beta`) | decide; SmoothL1 at beta 0.05 is nearly L1 in the operating range, not L2 |
| **joint limit penalty** | present, on the pose angles | absent; existing size bounds are not a pose penalty | do not apply stored pose limits blindly: all 6,000 audited COCO GT samples violate at least one, largely from flexible joints with `[0,0]` limits; any future prior must be annotation-compatible |
| keypoint loss type | L1 | legacy: SmoothL1; **rebalanced/gen-7: unsquared Euclidean**, already implemented | no correction pending for rebalanced |
| per-joint weighting | **learnable per-joint uncertainty** | fixed weights only (`finger_weight`, default 1.0, 0.2 in gen-6/7) | deferred with annotation-aware weighting; not current work |
| 3D keypoint normalisation | pelvis (body) / wrist (hands) before the loss | direct difference in rig-origin coordinates; `fk_scale` is only a warm-up multiplier | deferred by user on 2026-09-12; do not implement now |
| warm-up on the 3D term | yes | `kp3d_warmup_steps` = 2000 | already present |
| root vs local split | one continuous body space | `pose_split` separates `loss_pose_root` (beta 1.0) from `loss_pose` (beta 0.05) | revisit once the root is 6D |

**The loss metric is a separate bug from the output space.** Converting fixes the
network's output manifold; it does not fix the metric. A SmoothL1 in radians on
the root still scores a prediction at −π+ε against a target at π−ε as a ~2π
error when it is a tiny rotation. The 13.59 % statistic measures large GT angles,
not the frequency of this prediction error. Put the root's parameter loss on
the rotation matrix (Frobenius or a numerically stable geodesic formulation)
rather than on the Euler triple. Test finite gradients near identity and π.

Current student weight defaults, for the record: `w_pose` 1.0, `w_scale` 0.1,
`w_shape` 1.0, `w_cam` 0.1, `w_keypoints3d` 2.0, `w_keypoints2d` 10.0,
`w_3d_joints` 1e-3, `w_reproj` 0.01, `w_simcc` 1.0, `w_verts` 0.0. The teacher's
λ weights are not published.

For **generation 7**, override that defaults list with `w_shape=0.03`,
`w_keypoints3d=0.35`, `pose_beta=0.05`, `pose_split=True`,
`finger_weight=0.2`, and the explicitly requested `w_verts=0.35`.

## 1.5 Verification

1. **Bit-identity of the old path** with the flag off — diff a few dozen
   augmented samples against `git show HEAD:<file>`, comparing *alternating*
   calls (the TorchScript MHR forward differs by ~5e-10 on its first invocation
   in a process).
2. **Round-trip correctness**: test padded body targets, separate hand targets
   and the corrected root inverse on real GT. Compare rotations and FK geometry
   as well as parameters: equivalent Euler triples need not be numerically
   identical after canonicalization. Verify finite gradients and valid neutral
   initialization before training.
3. **The progress metric is the fitting gap, not 3DPW.** The student is roughly
   twice as far from its labels as it needs to be *on images it has already
   trained on*, and that — not split, conversion or selection — is where the
   ~23 mm gap to SOTA lives. Track it directly while changing the regression
   space.

   **Build the measurement; it does not exist today.** The number quoted in the
   old todo (**62.90 mm** with the student's predicted MHR params against
   **31.84 mm** with the GT params, held-out vertex error over 8,000 crops on
   five splits) came from a throwaway MLP ablation whose code is gone — no
   `z_body` / `head_SMPL` / `smpl_head` remains in the tree. It was also a
   by-product of a different question (whether an SMPL head should read the
   trunk latent or the predicted parameters; answer: 0.07 mm, irrelevant), it
   carries the MLP's own 31.84 mm floor, and it used `SMPL_MALE` throughout.
   Do not quote it next to a benchmark number.

   **Prefer a floor-free version:** run the student over a few thousand
   *training* crops, push the predicted 204-vector and the GT 204-vector through
   `MHRForwardPass` separately, and compare the 70 keypoints **root-relative,
   not Procrustes-aligned**. No learned head means no floor; no Procrustes means
   body-size and proportion errors stay visible. ~20 lines.

   **What exists today and its trap.** The epoch log's `held-in` line is a 2 %
   slice of the training corpus scored against the GT labels — the right family
   of metric. But `train_distill_jz.py:900-903` **overwrites** `Mesh_PA_MPJPE`
   and `Mesh_MPJPE` with the 3DPW numbers before they reach `history.jsonl`
   (deliberately, so selection keys off 3DPW), so those two fields carry no
   held-in information. Only `Mesh_PA_MPJPE_body` survives. Current values:
   g7r_s0 ep169 **34.15** (3DPW-val 39.93), g7h_s0 ep111 **40.52** (43.65),
   g6vv_s ep252 **43.44** (42.22), g6bv_s ep299 **41.82** (42.24) — note that
   two of the four runs are *worse* on their own training data than on 3DPW,
   which is worth explaining on its own.
4. `python tools/ddp_smoke.py --backbone <name> --w-verts <w>` before submitting
   — a changed head shape is exactly the class of change that passes every
   single-process check and then aborts DDP on step 2.
5. Seed spread is wide; anything under ~2 mm on 3DPW needs 2-3 seeds.

---

# 2. Crop resolution — the corpus is stored at the network input size

**Status:** open.

`datasets_pipeline/build_split.py --body-size` defaults to **224**, and the
stored crops are 224x224 (`data/sam3d_gt_*/images/*.png`, verified). The network
input is also 224. Two consequences:

1. **Upsampling existing crops adds no source detail.** Training at 256 or 288
   may still change feature sampling density, but evaluate that separately from
   rebuilding crops with `build_split.py --body-size <N>` from original frames.
2. **`--preset v2`'s zoom augmentation currently upsamples.** Its zoom ceiling is
   2.0, so a zoomed-in sample is interpolated from the 224 crop: up to 2x
   upsampling. The model is being trained to expect blur that a real crop at that
   scale would not have. Storing crops at 288-320 and letting augmentation crop
   *down* into 224 can address this **without changing the network at all**.
   The current dataset resizes to 224 **before** affine augmentation, so the
   implementation must instead sample the affine crop directly from the stored
   higher-resolution source. Changing stored resolution alone is insufficient.

**Keep the efficient encoder and mobile speed constraint.** Test modest input
increases (256/288) on the current HGNetV2/RepViT, or earlier-stage/stride-16
features at the same 224 input, as separate options. The current encoders emit
7x7 final features at 224; stride-16 features would give 14x14. More image tokens
also cost decoder time and memory. Benchmark the complete exported path on the
intended phone runtime before promoting either option. Do not replace the
encoder with DINOv3 to reproduce the teacher's 32x32 feature grid.

For scale: NLF uses EfficientNetV2-S at **256 px** and EfficientNetV2-L at
**384 px** (NLF paper, *Implementation details*). 384² is 2.94x the pixels of
224². The student is 40.1 M parameters total / 22.6 M backbone (`g7r_s0`,
`repvit_m2_3`); `g7h_s0` (`hgnetv2_b4`) is 31.9 M / 13.6 M. NLF-L is therefore
~3x the student in both parameters and pixel budget — the fair size comparison
is NLF-**S**, not NLF-L.

Cost to rebuild: the corpus is 4.2 M crops, and `sam3d_gt_harmony4d` has no
originals on the cluster (staged from the laptop), so a rebuild is not free.
Decide the target size once and do it in one pass.

## Verification

- Confirm stored crop size and that `build_split.py` writes what was asked.
- Re-measure 3DPW J14+adapter PA-MPJPE at matched epochs, 2-3 seeds.
- Check separately whether removing the augmentation upsampling helps at the
  *same* 224 input — that isolates the blur effect from the resolution effect.

---

# 3. Model selection

**Status:** open, low priority relative to items 1 and 2.

Selection is `min` over epochs of 3DPW **validation** J14 PA-MPJPE with the
adapter applied, from `train_distill_jz.py` via `val3dpw.py` (validation split,
12 sequences, stride 5, 1,993 person-frames, H36M GT).

**The split choice is correct — keep it.** Selecting on 3DPW test or EMDB would
turn the only clean numbers in the repo into training metrics.

**Selection noise is not a problem, measured.** The EMA validation curve has std
**0.08-0.17 mm** over the last 30 epochs, so `min` over ~300 epochs is only
0.2-0.6 mm optimistic. The val→test gap (g7r_s0: 39.93 → 41.55, **+1.62 mm**) is
genuine split difference, not a winner's curse. Older runs showing test
*better* than val by 6.5-7.2 mm (b2_s0, b3_s1, v3_s1) had their validation
scored by a pre-2026-09-05 `val3dpw.py`; those gaps are not comparable.

Two real defects:

1. **One metric, blind to most of what matters.** PA-MPJPE removes translation,
   rotation and scale; root-relative MPJPE removes translation. So `cam_trans`
   drift, body-size error, temporal jitter and identity are all invisible to the
   selector — and they are what the vid1/vid4/vid5 visual check actually
   judges. The ranking is not stable across benchmarks either: `g6vv_s` is best
   on 3DPW and under projected-GT-joint EMDB boxes, and **fourth** under EMDB's
   own boxes. Add an EMDB column and a jitter number (frame-to-frame
   acceleration; rigid bone-length CV, GT floor 1.06 %) to the selection.
2. **`summarize_runs.py` reports a pair no checkpoint achieves.**
   `datasets_pipeline/jeanzay/summarize_runs.py` reduces every METRICS column
   with `min` *independently* across epochs, and `_dpw` additionally takes
   `min(raw, ema)` per epoch. So the `g7r_s0` row "39.93 PA / 69.27 MPJPE" is PA
   from epoch 169's EMA weights and MPJPE from a different epoch, possibly
   different weights. Report the pair at the selected epoch; keep per-column
   records in a separate row if they are wanted.

---

# 4. Decoder feedback within the smartphone inference budget

**Status:** open experiment, conditional on preserving real-time smartphone
inference. User accepts an adapted version if full teacher feedback costs too
much. Do not copy the teacher's six-layer, 1024-wide decoder wholesale.

**Verified difference:** the student predicts MHR once after four decoder
layers. The teacher predicts provisional geometry between layers, projects
keypoints, samples image features at their 2D locations, and supplies updated
2D/3D coordinate embeddings to subsequent layers. It has 70 auxiliary 3D
tokens, although it has no independent XYZ regression head. Restoring a freely
regressed XYZ head alone would not reproduce this mechanism.

The expected benefit is better local articulation from inspecting image
evidence near provisional joints. No generation-7 accuracy gain or phone
latency has been measured for this change.

## Implementation and verification

- Start with one feedback update between existing decoder layers; reuse the
  encoder features and keep decoder width/depth fixed for the first ablation.
  Try a small body-joint subset and the existing cheap skeleton FK before
  considering full mesh geometry or additional 3D tokens at every layer.
- A cheaper alternative is intermediate SimCC coordinates driving feature
  sampling, without FK inside decoding. Label this as a 2D-guided approximation
  and measure it separately; it is not the teacher's geometry-derived feedback.
- Intermediate supervision can be training-only, but feature updates used by
  later layers belong to inference too. Do not assume they can be removed at
  export without testing the resulting accuracy change.
- The current parameter-output graph leaves FK outside the neural model.
  Putting FK or bilinear feature sampling inside decoding changes the export
  requirements. Check actual phone-backend support and fallbacks, end-to-end
  latency, peak memory and sustained throughput, alongside matched accuracy.
  If the change costs too much, simplify or leave it unselected; desktop FPS
  and FLOP estimates do not establish smartphone feasibility.

---

# 5. Exact 70-landmark readout using a small vertex subset

**Status:** open, scoped to extending existing subset skinning. Preserve MHR's
direct pose-parameter prediction and rig/mesh separation. First evaluate as a
training-loss change; a deployed landmark-readout change is a separate decision
subject to the smartphone speed constraint.

**What these landmarks are:** the 70 annotation keypoints supervised by
`loss_3d_native` and used for reprojection, including nose, shoulders, elbows,
knees, wrists, fingers, toes and surface landmarks such as the acromion. They
are not the rig's 127 kinematic joints or its joint-angle parameters.

The teacher first predicts pose, scale and shape parameters. MHR computes
skeletal joints by forward kinematics and skins the mesh using those joints.
Only then does a fixed checkpoint matrix read out landmark positions:

```text
predicted parameters → FK joints J → skinning → vertices V
                              J + V → fixed mapping → 70 landmark positions

teacher: K = W_joint @ J + W_vertex @ V
student: K ≈ W_fitted @ J
```

This is **not reconstructing joint angles from mesh vertices**. Angles remain
network predictions, and joints remain FK outputs. Mesh-surface landmarks
depend on shape and skinning as well as skeletal pose; replacing them with a
skeleton-only fit introduces an approximation.

Verified examples from the actual DINOv3 checkpoint's mapping:

| landmark | contributing mesh vertices | contributing skeletal joints |
|---|---:|---:|
| nose | 8 | 0 |
| left elbow | 21 | 0 |
| left knee | 0 | 1 |
| right wrist | 0 | 1 |
| left small-toe tip | 24 | 0 |

On 256 GT poses per split, the student's approximation differs from the
teacher's mapping by **2.93–3.75 mm mean over the 30 nonfinger landmarks**.
Those are operator discrepancies on identical GT geometry, not predicted
benchmark gains. The full audit includes errors against the stored annotations.

**Feasible reuse:** the teacher's first 70 mapping rows reference only **468
distinct vertices**. `MHRForwardPass._build_vertex_subset` already restricts
skinning to selected vertices, but currently chooses a farthest-point subset.
Extend it to accept the exact mapping-supported indices, or their union with
the current 595 vertex-loss indices. Do not assume the 595 already contain the
468, or that changing the subset size alone selects the right vertices.

## Implementation and verification

- Load the teacher's fixed mapping and select its first 70 rows. Split its
  vertex and joint columns, restrict vertex columns to their nonzero support,
  and remap them to the compact skinning indices. Preserve units and axes.
- Reuse one FK pass and the existing shape-dependent subset skinning; avoid
  constructing full vertices solely for this landmark readout. The existing
  shape-blend work and any other full-topology operations must still be
  accounted for when profiling; 468 indices are not an end-to-end speed claim.
- Prove landmark and gradient equivalence against the full reference mapping
  on real GT and perturbed predictions, including shape changes. Measure the
  union-subset cost at the actual training batch size.
- Gate the new loss readout so current runs remain reproducible. A training-only
  implementation adds no deployed operators, but deployment would retain the
  approximate landmarks; evaluate accuracy through that retained deployment
  path as well. If exact deployed landmarks are wanted, measure their phone
  runtime cost separately and document the changed readout in comparisons.
- Reuse this subset in decoder feedback only if item 4's latency gate passes;
  it is not a requirement to run skinning between all decoder layers.

---

# 6. Camera-ray conditioning and crop-camera parameterization

**Status:** open, focused on absolute pose and subject placement; keep the
efficient encoder and benchmark any inference overhead.

**Focal-aware conditioning is already present in generation 7** via
`--cliff-focal`. A learned projection already embeds the three conditioning
values, including crop size and offset normalized by focal length, into the
queries. Do not describe this task as simply adding focal information or adding
a learnable focal token; neither captures the remaining teacher difference.

Two distinct experiments:

1. **Spatial camera rays.** For each feature-grid position, recover its original
   image coordinate `(u, v)` through the crop transform and compute its viewing
   direction, proportional to `((u-cx)/fx, (v-cy)/fy, 1)`. Here `cx, cy` denote
   the camera principal point, not the person-box centre. The teacher encodes
   these rays and fuses them with image features. Test a lightweight fusion at
   the student's existing feature resolution; do not blindly copy the wide
   teacher module. Rays are derived from camera/crop metadata, not learned
   focal estimates. Keep augmentations, intrinsics and inference fallback
   calibration consistent. With ideal centred square crops, much of this is
   derivable from existing CLIFF plus grid position, so gains are not automatic.
2. **Separate crop-camera head.** The student directly predicts translation in
   metres; the teacher predicts crop scale and lateral offsets, then converts
   them to full-camera translation using focal length and box geometry. Assess
   that parameterization separately from ray embeddings. Check coordinate/sign
   conventions, depth stability and train/export/inference consistency.

Measure absolute camera-space error, depth/placement stability and reprojection
as well as body MPJPE. PA-MPJPE removes translation, rotation and scale and is
not the main success metric for these changes. Keep the current conditioning
and camera path available as the matched control.

**Deferred by user, 2026-09-12:** pelvis/wrist-relative supervision and
annotation-aware weighting (including the related uncertainty-weighting
proposal) are not current implementation tasks.
