# InstantHMR — training scripts

## What the labels are — read this first

**Training is supervised by the released ground-truth annotations of
[`facebook/sam-3d-body-dataset`](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**
— the human MHR fits Meta used to build SAM 3D Body. They are **not** teacher
inference. Every run in this repo's result tables, and the whole cluster corpus
(`sam3d_gt_sa1b` / `_aic` / `_harmony4d` / `_coco` / `_mpii`), is of this kind.

Distilling from the `facebook/sam-3d-body-dinov3` model is a supported
**option**, not the default: `tools/annotate_dataset.py` runs it over your own
images and is the only thing that writes `data/sam3d_distill_mix/`. Reach for it
to label imagery the released dataset does not cover, and note the cost — a
student trained on teacher output cannot exceed the teacher, while one trained on
the ground truth can.

Two naming traps that follow from the history:

- **"distill" in a filename means nothing about the labels.**
  `train_distill_mhr_only.py`, `train_distill_jz.py` and the
  `data/sam3d_distill_mix/` directory keep the name from when distillation was
  the only path. Only `annotate_dataset.py` produces teacher labels.
- **"teacher" further down this file usually means "the training target".** Where
  a number was measured against a label — parameter ranges, oracle ablations,
  the `adapter_j14_h36m_teacher.npz` fit — the label is the dataset's ground
  truth unless the text says `annotate_dataset.py` explicitly.

## Which script

Four files, and the difference matters.

| file | what it is |
|---|---|
| `train_distill_mhr_only.py` | **the trainer.** Dataset, augmentation, student architecture, MHR forward pass, all losses, metrics, the single-GPU loop. Everything of substance lives here. |
| `train_distill_jz.py` | **the multi-GPU entry point.** `import train_distill_mhr_only as T` plus what a cluster needs: DDP, mixture sampling, resumable checkpointing, held-out 3DPW validation, the pair-index cache. Run this one on a cluster. |
| `val3dpw.py` | 3DPW held-out validation used by `train_distill_jz.py`. Cross-validated against `benchmark/eval_3dpw.py` — they agree exactly. |
| `train_distill.py` | **legacy.** The original notebook port. See the warning below before using it. |

They are one trainer in two layers, not two trainers. A change to the dataset or
a loss goes in `train_distill_mhr_only.py` and both entry points pick it up; only
cluster plumbing and the CLI belong in `train_distill_jz.py`. Do not fork the
base file — that is how the ONNX bug below happened.

> **`train_distill.py` predates the SimCC 2D head.** Exporting a SimCC
> checkpoint through its ONNX path silently zeroes the 2D head, and the symptom
> is keypoints collapsing to the bounding-box centre with no error raised. Use it
> for reference, not for exporting current checkpoints.

## The `--preset` switch

`train_distill_jz.py --preset {baseline,v2}` selects the model generation.
`baseline` is the default and is bit-for-bit identical to the pre-v2 trainer, so
the two are directly comparable on the same corpus and the same seeds.

`v2` (see `apply_v2_preset`) bundles four changes from MeTRAbs (2007.07227) and
NLF (2407.07532):

| item | change | why |
|---|---|---|
| truncation | zoom ceiling 2.0, `geom_trans` 0.20 | `bbox_square` is exactly 1.2x the tight box on every split, so the old symmetric ±0.25 shaved ~2% off each side and essentially never truncated. Measured effect: images with a truncated keypoint go from 17.2% to 57.0%. |
| prerequisite A | `mask_oob_2d` | SimCC spans ±`kp2d_range` and `pred_2d` is a soft-argmax over the same bins, so a target at 1.67 is unreachable. Clamping asserts it sits at the edge and leaves a permanent saturating gradient. Mask instead; the 3D loss still supervises those joints. Do **not** widen `kp2d_range` — `bin_w = 2*range/(bins-1)`, so that degrades every in-frame joint. |
| prerequisite B | `cliff_follows_aug` | CLIFF describes the box the crop came from. Flip and rotation already moved it; `dx/dy/scale` did not, which was harmless at ±0.08 and 1.25 and is not at truncation strength. |
| occlusion | `RandomErasing` p 0.2 → 0.7 | MeTRAbs Table 7: 52.8 → 49.3 mm on top of an already-strong colour pipeline. |
| JPEG | `RandomJPEG` p 0.3, q 30–90 | The one item on NLF's list that was missing, and the most deployment-relevant — phone camera stacks emit JPEG. |
| absolute pose | `w_reproj` 0.01 → 0.5, flipped half un-masked, `loss_cam` unsquared Euclidean | MeTRAbs Table 8 and NLF §3.2 both insist on supervising the camera-frame pose. At 0.01 the term was decorative and half the batch was masked out of it. |

Two things to know when reading v2 results:

- **PA-MPJPE and root-relative MPJPE are both blind to the translation half of
  the absolute-pose change.** Procrustes removes translation; root-relative
  alignment removes it too. The reprojection and flip-unmasking parts should
  still move them. An absolute metric needs 3DPW's real intrinsics and a
  `f_true/f_assumed` correction — the corpus uses a synthetic
  `f = sqrt(H² + W²)`, ~12% off 3DPW's real focal.
- **Truncation augmentation is not expected to show on a full-body benchmark.**
  MeTRAbs's 124.7 → 77.8 mm is measured on deliberately truncated crops. It is
  in v2 because the papers already established it works, not to be re-proved.

`w_reproj = 0.5` is the least-grounded number in the preset (a 50x jump, chosen
so the term stops being decorative). Sweep it first if v2 underperforms:
`--preset v2 --w_reproj 0.1`.

## The `--losses` switch

**Orthogonal to `--preset`.** `--preset` controls the input pipeline and the
absolute-pose terms; `--losses {legacy,rebalanced}` controls the loss budget.
`legacy` is the default and is bit-identical to the pre-rebalance trainer, so
b2 / v2 runs already in flight stay valid controls. The four arms are
`baseline`, `v2`, `baseline + rebalanced`, `v2 + rebalanced`.

`rebalanced` (see `apply_rebalanced_losses` in `train_distill_mhr_only.py`)
comes from two measurements on a trained checkpoint, both reproducible from the
repo root against `data/sam3d_gt_coco`:

- **an oracle ablation** — replace one predicted group with the teacher's value
  and re-measure. Of the 37.8 mm of removable PA-MPJPE, `pose[6:136]` (the 130
  local joint angles) accounts for **34.3 mm**; the bone scales, the root and
  the 45 identity blendshapes account for **0.00 mm each**.
- **a per-term gradient norm** — back-propagate each term alone. `loss_shape`
  owned 27% of the update direction and `loss_cam` 15%, against `loss_pose`'s
  4.7% on 21% of the batch.

| item | change | why |
|---|---|---|
| shape | `w_shape` 1.0 -> 0.03 | Driving MHR params 204–248 to ±2σ moves the 127-joint skeleton by exactly `0.00e+00` cm, and `MHRForwardPass.get_joints` substitutes zeros for `shape_params` anyway. The blendshapes drive the mesh only. Kept non-zero so the deployed mesh identity stays supervised. |
| pose mask | `pose_split` | `loss_pose` was masked to `m_ident` (21% of the batch) because the dataset rotates the labels. Model params 3/4/5 map to joint-parameter slots 10/11/12 and nothing else, so `pose[6:136]` is exactly invariant to the in-plane rotation and only needs the flip mask — 21% -> 60% coverage on the parameters that carry all of the error. Root stays on `m_ident`; the two halves keep their per-parameter weight, so this changes the mask and not the root-vs-local balance. |
| pose beta | `pose_beta` 1.0 -> 0.05 | SmoothL1's beta is in radians here, so 1.0 is ~10x the p95 error and the term is a scaled MSE. |
| 3D loss | `kp3d_loss` `euclid`, `w_keypoints3d` 2.0 -> 0.35 | SmoothL1's beta of 1.0 is one **metre**, so at a 37.7 mm mean / 119 mm p95 error every sample is in the quadratic branch: 16x less gradient than L1 and 31x less than NLF's unsquared Euclidean at the same weight. The weight drops because the form under it changed. |
| fingers | `finger_weight` 0.2 | 40 of the 70 keypoints are finger joints and they carry 61% of `loss_3d_native`'s mass, while J12/J14 contain none of them and SA-1B — 55% of the batch under `--mix sqrt` — records hands as 8% observed. Applied to `loss_3d_native` and `loss_reproj`, the two FK-derived geometric terms; the 2D head's own terms are untouched. |

**The unsquared Euclidean is gentler where divergence happens, not harsher.**
`d‖x‖/dx` is a unit vector, so its gradient is flat in the error, where
SmoothL1(beta=1) grows with it. Measured against the form it replaces, at the
weights above: **0.62x** the gradient on a freshly initialised model (515 mm
error) and 6.4x once trained (60 mm). That is the opposite of the shape that
produced the LR blow-ups in `datasets_pipeline/jeanzay/STATUS.md`.

Validation also gains `Mesh_PA_MPJPE_body` / `Mesh_MPJPE_body` over the 30
non-finger keypoints, reported next to the existing metrics. **Selection is
unchanged** — and with `--val-3dpw` on, both are replaced by the 3DPW J12
numbers anyway.

## `--val-3dpw-gt` (added 2026-09-05)

3DPW validation now scores against the published-protocol ground truth — SMPL
run forward on the sequence's `poses`/`betas`, then the Human3.6M regressor —
instead of the pickles' raw `jointPositions`. Two things follow.

`ThreeDPWValSet` needs the precomputed reference next to `sequenceFiles/`, built
once by `benchmark/make_3dpw_gt.py` and rsynced to the cluster as
`$SCRATCH/instanthmr/3dpw/gt_h36m/`. It is 14 MB for all three splits. Without
it the job raises at startup rather than silently falling back.

The two references are ~13 mm apart on J12 PA-MPJPE, so **a selection metric
from before this change cannot be compared with one from after**. A resumed run
must keep the metric it started on: `--val-3dpw-gt jointpositions`, or
`DPW_GT=jointpositions` for `52_train_ddp.slurm`. `run_config.json` records
which was used.

## `--cliff-focal` (added 2026-09-06, mandatory)

**Use it on every new run.** It changes the CLIFF conditioning from
image-normalised pixels to a perspective-correct form:

```
[ atan((cx - W/2) / f),  atan((cy - H/2) / f),  box_px / f ]
```

— where the box sits in the field of view, and how large it is *angularly*.
`f` is `cam_focal_length[1]` (fy), a single scalar because the rotation
augmentation mixes the two image axes and `fx != fy` on the calibrated splits.

**Why it is not optional any more.** Under a *fixed* focal the two forms are
interchangeable, and the corpus this trainer was written for had exactly that:
every crop from `tools/annotate_dataset.py` carries the synthetic
`f = sqrt(H^2+W^2)`, i.e. `f/diag = 1.000` with **zero** spread. The rebuilt
corpus carries the datasets' own focals instead:

| split | f / diag | sd |
|---|---|---|
| `sam3d_distill_mix` (old corpus) | 1.000 | **0.000** |
| `sam3d_gt_aic` | 1.184 | 0.371 |
| `sam3d_gt_coco` | 1.049 | 0.344 |
| `sam3d_gt_mpii` | 0.961 | 0.266 |
| `sam3d_gt_harmony4d` | 0.285 | 0.010 |

With that spread the pixel form is **ambiguous**: of 4,000 COCO crops, 124 have
a near-twin under it (conditioning distance < 0.005) whose true depths differ by
**1.34 m**. The network can only predict the conditional mean, and the residual
leaves as unstable `cam_trans` and body scale. Measured on 3DPW test, models
trained on the new corpus jitter at 220-234 mm/frame² of translation
acceleration (98% of it in depth) against 148 for one trained on the
constant-focal corpus, and a ground-truth floor of 7.

**PA-MPJPE cannot see any of this** — Procrustes removes translation and scale,
and the metric moves by ~0.1 mm either way. Judge the change on `cam_trans`
stability and bone-length variance, never on the benchmark.

**Three places compute this vector and all three must agree**:
`SAM3DStudentDataset.__getitem__`, `val3dpw._preprocess` (which takes the real
`fy` from 3DPW's `cam_intrinsics`), and `instanthmr.inference.InstantHMR`.
Mismatching them raises nothing; it silently mis-places the person in depth.

### Carrying the flag past training (fixed 2026-09-06)

Until 2026-09-06 nothing outside the trainer knew about it: the checkpoint
evaluator and the ONNX exporter both built a default `DistillConfig`, so a
focal-aware model was scored and deployed with the pixel form. Two lookups
close that, and neither needs a flag in the normal case:

| consumer | how it learns the form |
|---|---|
| `train_distill_jz.py` validation | `cfg.cliff_focal` directly |
| `benchmark/eval_3dpw_ckpt.py` | `run_config.json` beside the checkpoint |
| `instanthmr.inference.InstantHMR` | `cliff_focal` in the ONNX metadata, stamped by `tools/pth_to_onnx.py` |

`T.config_from_checkpoint()` in `train_distill_mhr_only.py` is the single
implementation. It also restores `bound_scales` — from the weights, since
`--bound-scales` registers `scale_lo`/`scale_hi` buffers and their presence
*is* the flag. Both scripts print what they inferred and where it came from;
`--cliff-focal` / `--no-cliff-focal` override it for a checkpoint whose
`run_config.json` is missing or predates the flag.

**The value of `f`, once the form is known, is a separate question.** Training
and both 3DPW evaluators use the dataset's own focal — `cam_focal_length[1]`
per annotation, and 3DPW's calibrated `K[1,1]` (1969.23 px on 10 of the 12
validation sequences, 1961.85 on the two landscape ones). Deployment has no
annotation, so `instanthmr.inference` falls back to
`FOCAL_FALLBACK_DIAG * sqrt(H^2+W^2)` with `FOCAL_FALLBACK_DIAG = 1.05`, and
`demo.py --focal <px>` supplies the real one. The fallback applies to
focal-aware graphs **only**: a pixel-conditioned checkpoint was trained on a
corpus where `cam_focal_length = sqrt(H^2+W^2)` exactly, so `1.00 x diag` is
that model's camera and is left untouched.

Measured cost of getting the form wrong: **0.13 mm** J14+adapter PA-MPJPE on a
170-frame 3DPW validation probe (47.46 vs 47.33) — consistent with the ~0.1 mm
above. Like the flag itself, this matters for absolute placement and jitter,
not for the headline number.

The default is `False` and is bit-identical to the pre-flag trainer, verified
over augmented samples tensor by tensor.

## `--bound-scales` / `--anomaly-safe-fallback` (added 2026-09-06, mandatory)

**Use both on every new run.** They fix the divergence that killed all four
runs of the 2026-09-06 sweep at epochs 24-32 (`fno_s*`, `fsa_s*`), and every
earlier sweep that was "fixed" by lowering the peak LR.

### What was actually wrong

`head_global` is a plain `nn.Linear`. 77 of its outputs are **body-size**
parameters that the MHR forward kinematics multiplies along the kinematic
chain, and nothing bounded them:

| block | what it is | unbounded behaviour (measured on `checkpoints/mhr_model.pt`) |
|---|---|---|
| `0:3` | root translation | linear; +100 -> 22 m skeleton. Teacher value is **exactly 0.000e+00** in all 24,000 sampled annotations across all six splits — the person is placed by `cam_trans`. |
| `130:136` | `*_length/_width_flexible` | linear, transform weights up to 10.0; +100 -> 32 m |
| `136:204` | the 68 bone scales | **multiplicative**: +10 -> 336 m, +20 -> 703 km, +25 -> 28,300 km |

Note `130:136`. Those six sit at the end of the *pose* block but they drive
joint **translation** channels, not rotations, so they are size parameters.
Bounding only `136:204` still let a blown-up head reach a 375 m skeleton.
`6:130` are genuine rotations and stay at ~1.5 m at any magnitude — leave them
alone.

Loss on real teacher targets, with the head drifted by `d` on the size block:

| drift | total loss, unbounded | total loss, `--bound-scales` |
|---|---|---|
| 0 | 6.5 | 6.5 |
| 10 | **140** (over `anomaly_loss_threshold=100`) | 15.6 |
| 20 | **2.2e5** | 25.1 |

That is where the `4.06e7` / `8.76e13` losses in the dead logs come from.

### Why the guard made it permanent rather than catching it

An anomalous batch was discarded whole. But `loss_scale` / `loss_pose` in that
same batch are small and point straight back at the teacher — they are the
**only** force pulling the sizes out of the runaway region — so skipping
removed the fix along with the fault and quarantined those samples for good.
Two amplifiers turned that into a dead job:

* `bad` is all-reduced with `MAX`, so **one** bad sample out of 4x64 = 256
  vetoes the step for every rank. At 1% runaway samples 92% of steps are
  vetoed; at 2%, 99.4%. The run aborts when 2% of the data is bad, not the model.
* The EMA rollback (`anomaly_skip_patience`, `max_ema_rollbacks`) existed only
  in `train_distill_mhr_only`'s single-GPU loop. The DDP path had the abort and
  **no recovery at all**.

`--anomaly-safe-fallback` steps on `safe_loss_subset(losses)` — everything
except `FK_LOSS_KEYS` — instead of discarding the batch. Verified: with the
scales sitting +20 off the teacher, that gradient has cosine **+1.0000** with
`teacher - prediction` and moves **100%** of the 68 scales the right way, while
the full loss is 2.9e5. The rollback is now in the DDP loop too.

### Why lowering the LR never worked

`OneCycleLR(pct_start=0.1)` puts the peak at 10% of the run — epoch 30 of 300.
The four dead runs died at epochs 24-32. Lowering `max_lr` does not remove the
peak, it just slows the random walk toward the runaway region.

### The bound

`assets/mhr_size_bounds.npz` holds, per parameter, the union of the rig's own
`character_torch.parameter_limits` and the observed teacher range; the model
widens each by `--scale-bound-margin` (0.5) and squashes with `tanh`. The rig
bounds every one of these to `[-1.1, 1.1]`; the teacher respects that for the
scales but **not** for `130:136`, which the rig pins to `[0,0]` while the
teacher uses them up to 1.64 (`leg_length_flexible`) — hence the union.

Measured over 24,000 annotations from all six splits: **0.000000%** of teacher
targets fall outside the bound and the worst sits at `|atanh| = 0.80`, well
inside tanh's linear region, so the bound costs nothing. With the flag on, a
head deliberately blown up to `bias ~ N(0, 5000)` still yields a 1.58 m
skeleton. Exports to ONNX (matches torch to 1.2e-7).

Regenerate the asset from the rig plus the corpus if either changes; the file
also stores `rig_lo` / `rig_hi` unmodified for reference.

Both flags default to **off** and are bit-identical to the pre-flag trainer,
verified tensor by tensor. `run_config.json` records `bound_scales`,
`scale_bound_margin` and `anomaly_safe_fallback`.

### The 12-dimensional null space (not fixed, know it exists)

`parameter_transform[:, :204]` has **rank 192**. Twelve directions move the
parameters and leave the skeleton *exactly* unchanged — they pair each
`*_flexible` parameter against its `scale_*` partner (`scale_spine_length` vs
`spine_length_flexible`, `scale_hip_width` vs `hip_width_flexible`, and the
spine rotations). Along those, every geometric loss has zero gradient and only
`loss_pose` / `loss_scale` act, so the model is being asked to regress 12
dimensions of solver-arbitrary teacher noise that no image can determine.
Confirmed directly: pushing `scale_spine_length` to 20.3 along a null direction
moves the skeleton by 1.2e-7 m. The bound now keeps that wandering finite;
collapsing the redundancy properly would be a separate change.

## `--w-verts` — the surface (PVE) term (added 2026-09-07)

Every geometric loss in this trainer is blind to body shape. `get_joints` zeroes
the 45 `shape_params` before the forward pass, and that is not a shortcut — the
identity coefficients move the 127-joint skeleton by **exactly** `0.000e+00 cm`.
They are mesh-only. So `w_shape` in parameter space was the only thing
supervising them, and at `--losses rebalanced` that weight is 0.03, carrying
0.2% of the trunk gradient.

`--w-verts W` adds a term that skins the rig and compares vertices, which is the
only route by which a geometric gradient can reach shape. Measured
`d(loss)/d(shape_params)`: **8.9e-03** through vertices, **exactly 0.000e+00**
through joints.

**What it costs to ignore shape.** Running the rig on real teacher parameters
with the identity zeroed against the true identity: **5.95 mm mean, 22.68 mm max
per-vertex error**. That is a floor a shape-blind model cannot perceive, let
alone fix.

### It uses the rig's own mesh, not a separate LOD

The obvious way to make this cheap is to load `lod6.fbx` (595 vertices against
lod0's 18,439). Don't — it needs `pymomentum` to parse, which segfaults against
the torch in this environment, and lod6 is a *different topology*, so the
teacher correspondence would have to be fitted rather than being exact.

Instead `n_verts` picks a farthest-point subset of the rig's **own** 18,439
vertices and rewrites the three flattened influence tables of the rig's own
`linear_blend_skinning` — `(vertex, bone, weight)` triples — to reference only
those, under a compacted index. The skinning op is untouched, so the result is
the full mesh's answer restricted to those vertices, **verified bit-exact to
4.6e-05 cm**. Same topology, same vertex ids as the teacher: the correspondence
is exact, not fitted. The default `n_verts=595` is lod6's vertex count, chosen
so the sampling density matches what the LOD would have given.

Measured at batch 64 on an RTX 4070:

| path | fwd+bwd | vs skeleton-only |
|---|---|---|
| skeleton FK only (what the trainer already did) | 4.85 ms | — |
| + 595-vertex subset | 5.44 ms | **+0.6 ms** |
| + all 18,439 vertices | 21.6 ms | +16.7 ms |

Against a 207 ms step that is **+0.3%** for the subset and +8.1% for the full
mesh. On a real training step, end to end: **+0.9% wall time and +0 MiB peak
GPU** — the backbone activations dominate and the vertex tensors fit in slack.

### Choosing the weight

The term is an unsquared Euclidean distance in **metres**, the same form as
`loss_3d_native` under `--losses rebalanced`, so the weights are directly
comparable and `loss_verts / w_verts * 1000` reads as mean PVE in millimetres.
Per-term gradient norm on the trunk, pretrained backbone, `v2 + rebalanced`:

| term | \|grad\| | share |
|---|---|---|
| `loss_reproj` | 33.97 | 61.3% |
| `loss_verts` (at w=1.0) | 7.21 | 13.0% |
| `loss_cam` | 4.51 | 8.1% |
| `loss_3d_native` | 3.74 | 6.7% |
| `loss_pose` | 2.93 | 5.3% |
| `loss_shape` | 0.12 | **0.2%** |

**`--w-verts 0.35` is the recommended setting**: `|grad| = 2.52`, 5.0% of the
update, just under `loss_3d_native` and nowhere near `loss_reproj`. At 1.0 it
would be the second-largest term in the budget, which is not what a first trial
should do.

`loss_verts` is in `FK_LOSS_KEYS`, so `--anomaly-safe-fallback` drops it along
with the other forward-kinematics terms on an anomalous batch — a blown-up bone
scale explodes the vertices exactly as it explodes the joints. It is also
multiplied by `fk_scale`, so it ramps over `kp3d_warmup_steps` (2000) like
`loss_3d_native` and `loss_reproj`; a PVE read from the first epoch is
suppressed by that ramp, not genuinely small.

`w_verts = 0.0` is the default and disables the term **and every code path it
touches** — `MHRForwardPass` is then constructed with `n_verts=0` and builds no
subset at all. Verified: every loss term matches `git show HEAD` to the last
printed digit on 48 real augmented samples, under both `baseline/legacy` and
`v2/rebalanced`.

**There is no PVE metric yet, only the loss.** See `docs/todo.md` item 13.

## `--crop-centre-fix` (added 2026-09-07)

`--preset v2` sets `cliff_follows_aug`, which moves the CLIFF conditioning to
describe the *augmented* box. The crop-centre half of that correction was
missing a division by the zoom: the warp is `u' = s*R*(u - c) + c + t`, so
inverting it puts the visible window's centre at `-t/s`, not `-t`.

Verified against the exact preimage of the crop centre under the real
`M_total` that `cv2.warpAffine` was called with, over 400 augmented COCO crops
at v2 strength, rotation disabled so the zoom is the only variable:

| | mean | p95 | max |
|---|---|---|---|
| off (the gen-5 behaviour) | 7.73 px | 21.00 px | 34.93 px |
| **on** | **0.68 px** | 1.12 px | 1.50 px |

The residual is the bbox-rounding floor. **Default off** so generation 5 and
`--preset baseline` reproduce bit-for-bit; it is a no-op under `--preset
baseline`, which does not set `cliff_follows_aug` at all. Pass it for every new
`v2` run.

This is a wrong **input**, not a wrong label — the reprojection chain is exact
to 0.000 px — and it only affects lateral placement, which PA-MPJPE cannot see.
Do not expect the benchmark to move; expect the conditioning to stop lying.

The rotation half of the same correction is **not** settled: see `docs/todo.md`
item 11.

## `--cont-head` — the teacher's continuous regression space (added 2026-09-12)

**What it changes.** Without it, the student emits the rig-native 204-vector
straight out of one `nn.Linear`: raw Euler angles and 68 free bone scales.
SAM 3D Body does not. Its head emits a *continuous* parameterisation and
converts to the 204-vector inside the head, before MHR sees anything. This flag
reproduces that pathway.

| the head emits | dims | converted, inside the model, to |
|---|---|---|
| root rotation, 6D | 6 | 3 extrinsic-XYZ Euler angles |
| body pose, continuous | 260 | 130 joint angles |
| identity coefficients | 45 | passed through unchanged |
| bone-scale PCA coefficients | 28 | 68 bone scales |
| two hand blocks | 2 x 54 | the 54 finger channels, overwritten |

447 numbers total — the teacher's 519 minus the 72-dim face block its own
`forward()` multiplies by zero. They come from a two-layer MLP at the student's
own 512 width on LayerNorm'd pose-token features (`nn.TransformerDecoder` is
built here with `norm_first=True` layers and no final norm, so its output is
unnormalised), plus a learned initial estimate. Camera translation moves to its
own 3-dim head, unchanged in parameterisation.

**Nothing downstream changes.** The model still returns 204 MHR parameters, 45
identity coefficients, camera translation and 70 2D keypoints; every loss, the
forward kinematics, `tools/pth_to_onnx.py` and `instanthmr.inference` see the
contract they already saw. `config_from_checkpoint` reads the flag back out of
the weights (the `cont_head.*` buffers ARE the flag), so an export needs no
argument.

**Why 260 for the body.** 23 three-DoF joints x 6D (138) + 58 one-DoF hinges x
(sin, cos) (116) + 6 raw translation channels (6). Those last six are the
`*_length/_width_flexible` parameters at `130:136` — body SIZE, not rotation —
and **the continuous space does not bound them**, so they keep their own tanh
against the rig limits exactly as under `--bound-scales`.

**The root convention is the rig's, not the teacher's literal call.**
`mhr_head.forward()` decodes the root with `roma.rotmat_to_euler("ZYX", R)`.
Measured against `checkpoints/mhr_model.pt` (md5-identical to the teacher's own
`assets/mhr_model.pt`): driving `model_params[3:6]` and rigid-fitting the 125
joints it moves recovers **extrinsic XYZ** to 0.00000–0.014 deg, while roma's
`"ZYX"` of the same triple is **36–92 deg away** — roma returns the triple
Z-first, so it is the reverse of the rig's. Either convention round-trips and
both would train; the rig's is used because only then is the intermediate
rotation matrix the body's *actual* root rotation, which is what makes the
accompanying chordal loss a physical angular error rather than a distance in a
permuted space. All of this lives in `mhr_cont.py`, which is a transcription of
the teacher's own functions verified at **0.000e+00**.

**Bone scales are bounded as coefficients, and that ends the runaway.**
`--bound-scales` clamps the 68 expanded scales coordinatewise, which is wrong
for a PCA head: an arbitrary point of the 68-dim box is generally not in the
24-dimensional column space `scale_comps` spans, so the clamp silently leaves
the subspace. The bound moves onto the 28 coefficients, symmetric about zero so
an untrained head sits exactly on `scale_mean`. Measured: feeding the head
`N(0, 25)` — the regime that produced a 28,300 km skeleton and killed four runs
at epochs 24–32 — now gives a **2.66 m** skeleton, scales still in-subspace to
1.2e-07.

**The root loss changes with it** (`root_rot_loss`, set by the same flag) from
SmoothL1 on the Euler triple to the squared chordal distance
`||Rp - Rt||_F^2 / 4`, on the rotation the head built rather than one rebuilt
from the decoded Euler triple — `atan2` in the backward path is unstable, and
2.25% of corpus roots sit within 0.1 rad of the ±pi/2 gimbal. Divided by the six
root parameters the old term averaged over, so it is budget-neutral: 0.00003 vs
0.00003 at 1 deg, 0.00063 vs 0.00063 at 5 deg, and 0.333 vs 0.437 at 179 deg —
matched where it matters, saturating instead of growing.

**Cost: none worth measuring.** +0.37 M parameters, and ONNX CPU single-thread
batch 1 goes 102.03 -> 102.49 ms on `repvit_m2_3` and 64.55 -> 64.19 ms on
`hgnetv2_b4`. The 980 extra graph nodes are all elementwise on 447 numbers.

**Before using it:**

```sh
python tools/verify_cont_head.py --data_root data          # 24 gates, exits non-zero on any failure
python tools/ddp_smoke.py --backbone <name> --w-verts 0.35 --cont-head --data_root data/sam3d_gt_mpii
```

`instanthmr_distill_train/assets/mhr_cont_head.npz` (30 KB) carries the
teacher's scale and hand bases plus the measured coefficient bounds;
`tools/build_cont_head_assets.py` rebuilds it from the teacher checkpoint and
the corpus. **It is a new file and must be rsynced**, or the job dies at
startup.

Default off. With it off every tensor in the trainer is bit-identical to the
previous commit — verified: 1544 state_dict keys, same sha256, forward and all
12 loss terms 0.000e+00 apart on 48 real augmented samples, under both
`--losses legacy` and `--losses rebalanced`.

## `--exact-landmarks` — the teacher's own 70-landmark readout (added 2026-09-13)

**What it changes.** The student derives its 70 annotation keypoints from the
127 skeleton joints alone, through a fitted `(70, 127)` matrix. SAM 3D Body does
not — it reads them off the *skinned mesh* and the joints together, with a fixed
matrix from its checkpoint:

```
teacher:  K = W_joint @ J  +  W_vertex @ V
student:  K ≈ W_fitted @ J
```

**21 of the 70 landmarks have no joint contribution at all** — nose, elbows, toe
tips, acromion and other surface points — so a skeleton-only matrix cannot
represent them, and none of the 70 can move with body shape. Measured against
the teacher's readout on identical GT geometry:

| readout | all 70 | 30 non-finger | worst |
|---|---|---|---|
| fitted `(70, 127)` | 1.306 mm mean | **3.037 mm mean** | 18.05 mm |
| exact subset readout | 0.000 mm | 0.000 mm | 0.000 mm |

Those targets came out of the teacher's mapping, so this removes an operator
mismatch between the prediction and its own label. It affects `loss_3d_native`
and `loss_reproj` only.

**Why it is affordable.** The first 70 mapping rows reference just **468
distinct vertices**. `MHRForwardPass` already skins an arbitrary subset exactly
(not approximately — it rewrites the rig's three flattened influence tables and
leaves the op untouched), so this takes the **union** of those 468 with the
`--w-verts` subset. Do not assume the 595 already contain the 468: measured,
they overlap in **10**. Union is 1053 vertices, and the cost is unmeasurable —
181.2 ms for 595 alone against 180.9 ms for the union, batch 64, forward+backward.
The skinning is dominated by the forward-kinematics pass, not the vertex count.

`loss_verts` keeps averaging over exactly its own 595 farthest-point samples
with uniform weighting, via `verts_loss_pos`; averaging over the union instead
would silently reweight the term and break comparability with generations 6-8.

**It gives identity a second geometric gradient.** Resample `shape_params` and
the exact landmarks move **0.821 mm mean**; the fitted ones move exactly
**0.000 mm**. Before this, `--w-verts` was the only route by which the 45
identity coefficients saw any geometry at all.

**Verified** (`python tools/verify_exact_landmarks.py --data_root data`) against
full 18,439-vertex skinning with the full `(70, 18566)` teacher mapping:
**2.4e-04 mm** on real GT, on pose perturbed by N(0, 0.25) rad, and under
resampled identity; gradients agree to 2e-07 relative in `model_params` and
2e-09 absolute in `shape_params`. Every row of the mapping sums to exactly 1.0,
so the readout is translation-equivariant and commutes with `to_vision` — which
is why the mapping is applied after the unit/axis conversion.

**Scoring is unchanged, on purpose.** `val3dpw.py` and
`benchmark/eval_3dpw_ckpt.py` both go through `get_native_keypoints`, i.e. the
fitted readout, and neither was touched. The flag changes what the model is
trained to match, not what it is measured with, so the reported J14 PA-MPJPE
stays comparable with earlier generations and keeps measuring the path that
actually ships. **Deployment is unchanged**: the exported graph never emitted
these 3D keypoints — the consumer derives them — so no operator is added to the
phone runtime. Making the deployed readout exact is a separate decision with its
own latency measurement.

`instanthmr_distill_train/assets/mhr_landmarks70.npz` (167 KB) is **a new file
and must be rsynced**, or the job dies at startup. Default off; with it off
every loss term is bit-identical, verified across `legacy`, `rebalanced`,
`w_verts` on/off and `cont_head` on/off.

## Tried and rejected: detaching the 2D head (2026-09-06 / 09-07)

**Do not retry this without reading the failure mode.** The flag is gone from
the code; this section is the record of why.

The idea was to stop the 2D SimCC head from training the **shared trunk** —
backbone plus decoder — with one line in `InstantHMRStudent.forward`,
`feat_2d = feat_2d.detach()`. The head kept its full loss and stayed a deployed
output; only its gradient into the trunk was cut. The motivation was a real
measurement: on the converged `b3_s1` checkpoint, of the gradient reaching
backbone + decoder, `loss_2d_simcc` owned 47.6% and `loss_2d_native` 8.6%
against 20.7% for all seven 3D body-pose terms together, and the 2D gradient had
the worst signal-to-noise of any term (`|mean g| / mean |g|` 0.28 against the
pose group's 0.39) at only +0.21 per-batch cosine with it.

**It failed on both axes.** Two matched seeds each (`dt` = detached, `bno` =
control), identical mixture, LR and flags, scored at epoch 83-87 of 300 on 3DPW
test (35,463 person-frames, published-protocol GT, adapter applied) and COCO
val2017 (6,352 persons, GT boxes):

| arm | 3DPW PA-MPJPE | 3DPW MPJPE | COCO OKS AP | COCO PCK@0.05 |
|---|---|---|---|---|
| `bno` (control) | **44.45 ± 0.90** | **71.37 ± 1.04** | **39.4** | **73.5%** |
| `dt` (detached) | 45.53 ± 0.30 | 79.61 ± 3.86 | **0.00** | 3.1% |

**The 2D head does not degrade — it collapses.** Measured over 48 COCO crops,
the spread of the 70 predicted keypoints *within* one crop:

| arm | within-crop spread (x, y) | across-crop centroid spread (x, y) |
|---|---|---|
| `bno_s0` / `bno_s1` | 0.226 / 0.311, 0.251 / 0.310 | 0.123 / 0.271, 0.129 / 0.287 |
| `dt_s0` / `dt_s1` | **0.013 / 0.011**, **0.020 / 0.023** | 0.140 / 0.231, 0.155 / 0.232 |

A 15-25x collapse within the crop, on both seeds, while the across-crop spread
is unchanged — the head still tracks *where the person is*, but predicts all 70
keypoints at the same point. Hence OKS AP of exactly 0.00: no prediction ever
clears the lowest threshold.

**Why, and this is the part worth remembering.** `head_2d_feat` and
`head_2d_logits` are **shared across all 70 queries**, applied per token. For
the output to differ per keypoint, query *k*'s feature must encode keypoint *k*
— and the only thing that ever forced that was the 2D loss. Detach it and the
70 query features are shaped solely by what helps the *global* token through
self-attention, where per-keypoint identity is worth nothing. They converge on
each other and the shared head has nothing left to distinguish. Nothing in the
objective pushes it back, so more epochs do not recover it.

An earlier note in this file claimed the queries "stay alive and become free
capacity for the token the parameters are read from". Alive, yes. Still
keypoint-specific, no.

**Two secondary lessons.**

*Early epochs lied.* At epoch 9 `dt` led by 5.6 mm J14 PA. By epoch 84 it
trailed. `OneCycleLR(pct_start=0.1)` peaks at epoch 30 of 300, and an arm
spending less on the fast-converging 2D task looks good before the LR anneals.
Do not read a 300-epoch A/B before the peak.

*The entropy argument behind it was wrong.* `loss_2d_simcc` sits at 2.227 nats
against an irreducible floor of 2.112 — the entropy of its own Gaussian soft
label at `sigma = 2 * bin_w`, computed directly. That was read as "only 5.2% of
the term is reducible, so the task is 95% converged". It does not follow:
`CE = H(target) + KL(target || prediction)` and `H` is a constant with **zero
gradient**, so the ratio says nothing about how much of the *learnable* part is
left. The learnable part is the whole of `KL = 0.115` nats. Use the floor only
as a scale for reading the `simcc` column — subtract 2.112 to get what the model
is actually graded on. The gradient measurements above are independent of this
and still stand.

**If you want the trunk's capacity back, the cost is not the head.** It is
0.025 of ~10.8 GFLOPs, 0.23%. The 15.7% is the **70 decoder queries** that feed
it, and removing those is a different architecture that no run has tested. See
`docs/todo.md`.

## The reported number

One number everywhere: **J14 PA-MPJPE, adapter-applied, against the
published-protocol 3DPW ground truth.** The training log, `summarize_runs.py`,
`benchmark/eval_3dpw_ckpt.py` and a paper results table all print the same
quantity — cross-checked to the decimal (41.71 mm for `b3_s1` on 3DPW test).
J14 MPJPE rides alongside it. J12 is gone: it was the rig-clean row against the
old `jointPositions` GT, and against the H36M one it is not (those hips sit
292 mm apart against the MHR rig's ~120).

The adapter is `benchmark/results/adapter_j14_h36m_teacher.npz`, a fixed 14x70
map fitted from the teacher's labels and independent of any student, so it does
not move between checkpoints. `--val-3dpw-adapter none` scores raw MHR joints
instead; a missing file is a hard error, not a silent fallback.

The epoch log prints two lines because there are **two different datasets**:

```
held-in (teacher agreement)  RAW loss 5.09 body PA 153.5 | EMA ...
3DPW-validation  RAW J14 PA 126.0 MPJPE 283.0 | EMA J14 PA 125.7 MPJPE 284.0 mm
```

`held-in` is a 2% slice of the **training corpus** scored against the
**teacher's** labels — agreement, not generalisation, and only a training-health
signal. `3DPW-validation` is 12 held-out sequences against real **MoCap**.
Before 2026-09-06 both appeared on one line under a single "val" label, because
`Mesh_PA_MPJPE` is overwritten with the 3DPW number when `--val-3dpw` is on.

**Runs started before 2026-09-06 are not comparable**: their `J14_PA_MPJPE` in
`history.jsonl` is the raw MHR joint set, ~26 mm higher than the adapter-applied
one. `summarize_runs.py` still renders them, in the same column.

## The pair index

`SAM3DStudentDataset` enumerates the corpus with one `glob` plus a `stat()` per
crop. At 3.7M crops that is ~7.5M metadata operations, and `build_jz_loaders`
builds two dataset objects per rank — eight concurrent scans on a 4-GPU job, 78
minutes on Lustre before a single training step.

`build_pair_index()` caches that enumeration to
`<data_root>/.pair_index_v2.pkl`, rank 0 builds it behind a barrier while the
others wait, and train/val share one list. Build it ahead of time with
`--warm-cache` (which also populates `$TORCH_HOME` with the timm backbone, since
compute nodes have no internet). Cache validity is keyed on sub-folder mtimes;
`--rebuild-index` forces a rescan.

## Legacy: notebook port for `pfcalcul`

Everything below describes `train_distill.py`, the standalone version of
`notebooks/distill_transformer_decoder.ipynb`. It bundles notebook cells
**0, 1, 3, 7, 8, 10, 11 and 15** into a single script: setup, config, dataset,
student architecture, distillation loss, HMR metrics, the EMA + early-stopping
training loop, and ONNX export/quantization.

## Folder layout (on the cluster)

Project root: `/pfcalcul/work/kchalabi/envs/lstm/instanthmr_distill_train/`

```
instanthmr_distill_train/
├── train_distill.py          # the standalone training script
├── submit_train_distill.sh   # SLURM launcher (sbatch)
├── requirements.txt          # pip deps (auto-installed by the platform)
├── checkpoints/
│   └── mhr_model.pt          # <-- YOU must place this here (≈700 MB, not in git)
└── runs/                     # created at runtime: checkpoints, logs, exports
```

## Dataset layout

The script takes a **single entry-point folder** (`--data_root`) that contains
one or more sub-folders, each with its own `annotations/` and `images/`:

```
instanthmr_data/
├── sam3d_distill_coco/
│   ├── annotations/   *.npz
│   └── images/        *.jpg | *.png
├── sam3d_gt_coco/
│   ├── annotations/
│   └── images/
└── ...
```

All sub-folders are loaded and concatenated. File names may collide across
sub-folders — that is fine, each `(image, npz)` pair is stored as a full path,
so identically-named files in different sub-folders are kept as distinct
samples.

## Before launching

1. Copy `mhr_model.pt` into `checkpoints/` (or pass `--mhr_model_path`).
2. Make sure the dataset is registered as `/datasets/instanthmr_data` so that
   `datasynch_perso` can sync it onto the node.

## Launch

From `/pfcalcul/work/kchalabi/envs/lstm/`:

```sh
sh instanthmr_distill_train/submit_train_distill.sh
```

The launcher syncs `/datasets/instanthmr_data`, `cd`s into the project folder,
and runs `python3 train_distill.py --data_root ../instanthmr_data`. If
`datasynch_perso` places the data somewhere else, edit the `--data_root` value
(and the `datasynch_perso` line) in `submit_train_distill.sh`.

## Useful flags

```
python3 train_distill.py \
    --data_root ../instanthmr_data \
    --output_dir runs/distill_repvit_cliff_v2 \
    --epochs 400 --batch_size 64 --lr 3e-4 --num_workers 8 \
    --self-test     # run arch + perfect-student loss sanity checks first
    --no-resume     # ignore any existing checkpoint, train from scratch
    --no-export     # skip ONNX export/quantization after training
```

On the cluster, flags go through `EXTRA_TRAIN_ARGS` on
`datasets_pipeline/jeanzay/52_train_ddp.slurm`, not on a bare command line. The
generation-6 set is:

```
--preset {baseline|v2} --losses rebalanced \
--cliff-focal --bound-scales --anomaly-safe-fallback --crop-centre-fix \
--w-verts {0|0.35}
```

Training auto-resumes from `runs/<name>/best_student_model_v3.pth` if present.
Outputs (best checkpoints + `export/*.onnx`) land under `--output_dir`.
