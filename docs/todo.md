# Open work, from the 2026-09-06 accuracy audit

What the external audit found, what was checked against the code, and what is
left. The audit itself is `docs/accuracy_audit_2026-09-06.md`; this file is the
verdict on each item and the queue.

Every claim below was reproduced on real data before being written down. Where
a number appears, the command that produced it is named.

---

## Done (2026-09-06 / 09-07)

### 1. The ONNX exporter dropped the scale bounds — 21 mm, silently

`tools/pth_to_onnx.py` built a default `DistillConfig`, where `bound_scales` is
off, and loaded with `strict=False`. A `--bound-scales` checkpoint's `scale_lo`
/ `scale_hi` were reported as "unexpected keys" and discarded, so the exported
graph had no `tanh` remap while the head still emitted pre-`tanh` values.

**Mechanism.** With `--bound-scales` the last layer no longer emits bone scales.
It emits a raw number that `tanh` maps into the rig's legal size range. Feed
those raw numbers to the forward kinematics directly and the skeleton is
wrong — not catastrophically, because the pre-`tanh` values are small
(`|u|` mean 0.120, max 0.571 over teacher targets), but consistently.

**Measured**, reconstructing the exact pre-`tanh` values a bounded head must
emit to reproduce 64 real COCO teacher targets, then running the rig both ways:

```
MPJPE   (export vs correct), raw     : 21.0 mm
MPJPE   (export vs correct), centred : 19.9 mm
PA-MPJPE(export vs correct)          : 14.4 mm
```

On a 41.8 mm PA-MPJPE model that is a third added, with no error raised.

**Fixed by** `T.config_from_checkpoint()` in `train_distill_mhr_only.py`, which
reads `bound_scales` back out of the weights. Unexpected keys are now a hard
exit, and the exporter gates on a PyTorch↔ONNX parity check (`2.7e-05` on a
bounded graph, `3.1e-05` on a legacy one).

### 2. Training and evaluation could use different camera inputs

Three separate holes, failing in opposite ways:

* `benchmark/eval_3dpw_ckpt.py` built a default config, so `cliff_focal` was
  off — **silent**. It also loaded strictly, so a bounded checkpoint raised —
  **loud**, and the only reason no wrong number was ever published.
* `instanthmr/pipeline.py` never passed a focal to `InstantHMR`.
* `HMRPrediction.focal_length` always reported `sqrt(H^2+W^2)` even when a real
  focal was supplied, so anything unprojecting or rendering used a fake camera.

**Fixed by** separating *which conditioning form* (from `run_config.json` for
checkpoints, from ONNX metadata for graphs) from *what the focal is* (the
dataset's own value in training and both evaluators; `--focal`, else
`1.05 x diag`, at deployment). See `instanthmr_distill_train/README.md`
§`--cliff-focal` and `benchmark/README.md`.

**Measured cost of the form mismatch**: 0.13 mm J14+adapter PA-MPJPE on a
170-frame 3DPW validation probe. Small, because Procrustes removes exactly what
this corrupts. The `bound_scales` half is where the 14–21 mm lives.

---

### 3. The v2 crop-centre update forgot to divide by the zoom — fixed 2026-09-07

`train_distill_mhr_only.py` wrote `cx -= dx * crop_w / 2.0`. The augmentation
warps the crop with `u' = s*R*(u - c) + c + t`, so inverting it puts the visible
window's centre at `-t/s`: the shift has to be divided by the zoom. It
over-corrected when zooming in and under-corrected when zooming out, and it was
live in every `--preset v2` run (`geom_trans=0.20`, `geom_scale_max=2.0`).

**Fixed behind `--crop-centre-fix`** (config `crop_centre_fix`), default off so
generation 5 and `--preset baseline` stay bit-identical — verified: every loss
term matches `git show HEAD` to the last printed digit on 48 real augmented
samples, under both `baseline/legacy` and `v2/rebalanced`.

**Verified on the fix itself.** Reference is the exact preimage of the crop
centre under the real `M_total` that `cv2.warpAffine` was called with, over 400
augmented COCO crops at v2 strength with rotation disabled so the zoom is the
only variable:

| | box-centre error, mean | p95 | max |
|---|---|---|---|
| `crop_centre_fix=False` | 7.73 px | 21.00 px | 34.93 px |
| `crop_centre_fix=True` | **0.68 px** | 1.12 px | 1.50 px |

The 0.68 px residual is the bbox-rounding floor.

**With rotation on (±30°) a ~31 px residual remains against that reference, and
it is NOT this bug** — the fix moves it only 33.40 → 30.93 px. The code models
the in-plane rotation as a camera roll (it rotates `joints_3d`, `cam_trans` and
the box centre about the image centre together), for which "preimage of the crop
centre" is the wrong target, so most of that 31 px is the reference, not a
defect. Expressing the `dx`/`dy` correction in the rotated axes instead changes
the distance from the box centre to the actual body centre by 63.75 → 63.15 px,
i.e. nothing. **No second defect demonstrated; see item 11 for what is still
open.**

## Queue

Ordered by (cost of being wrong) x (cheapness of the fix). Items 3, 4 and 6 are
defects; 5 and 7 onward are opportunities the audit raised that are not bugs.

### 4. Validation drops the frames the model failed on

`val3dpw.py:179` removes non-finite predictions before scoring — necessary,
because one `inf` makes numpy's Procrustes SVD raise and that killed a healthy
20-hour job. But the count is only logged, and the surviving-subset score still
competes for "best checkpoint" at `train_distill_jz.py:895`. A checkpoint that
blows up on the hardest frames can therefore be selected over one that merely
does badly on them.

The distributed half is real but numerically trivial: `all_reduce_mean`
averages rank means rather than pooling, and the 3DPW set is stride-sharded
499/498/498/498 out of 1,993, so the bias is under 0.1% unless failures cluster
on one rank.

**Risk is low right now**: `--bound-scales` is exactly what stops the outputs
going non-finite. **Watch** `⚠️ 3DPW: N non-finite prediction(s) dropped` in
the epoch log — zero every epoch means this never fires; a rising count is a
real instability and should be investigated immediately.

**Fix when convenient**: reduce error *sums* and valid *counts* rather than
rank means, and either mark an epoch with drops ineligible for selection or
score dropped frames at a fixed penalty so the population is constant.

### 5. Score both routes to 2D keypoints on the same footing

The graph emits 2D keypoints two ways, and only one is measured today:

1. `joints_2d` — the SimCC head. This is what `benchmark/eval_coco.py` scores
   (`p.joints_2d[COCO17_FROM_MHR70]`).
2. `mhr_params` + `cam_trans` -> forward kinematics -> project with the focal.
   Governed by `loss_reproj`.

The detach experiment is what makes this worth measuring: it showed the two
routes are genuinely independent. `loss_reproj` was unchanged between the arms
to within seed noise while the SimCC head collapsed to a point, so the trunk
kept its image-plane localisation and only the readout lost it. Route 2 may
therefore be usable when route 1 is not — see
`instanthmr_distill_train/README.md` §"Tried and rejected".

The logged columns cannot settle which is better: `loss_reproj` is an L1 with
fingers down-weighted to 0.2 and `w_reproj = 0.5` folded in, while
`loss_2d_native` is a smooth-L1 at full finger weight times 10. Different
norms, different weights, both on training data.

**Add a `--from-mhr` path to `eval_coco.py`** that projects the MHR keypoints
instead of reading `joints_2d`, and report both on the same 17 COCO body
keypoints. That is the measurement that decides whether the SimCC head still
earns its place — and note the projected route needs `cam_trans` to be right,
which PA-MPJPE cannot validate, so it is not automatically the safer choice.

**If the projected route wins, "remove the 2D head" is still not the change to
make.** The head is 0.025 of ~10.8 GFLOPs — 0.23%, nothing. The cost is in the
**70 decoder queries** that feed it: 15.7% of total FLOPs. Dropping those is a
different architecture that no run has tested, and the standing hypothesis is
that they *help* the global token via self-attention. Sequence: finish the
runs, score both routes, then a separate ablation with the 70 queries removed.
(`--detach-2d-head` is gone; it was tried and rejected, see the trainer README.)

### 5b. Late divergence the loss guard cannot see (`bno_s1`, job 1813482)

**Open. This killed a healthy run at epoch 92 of 300 and nothing caught it.**

`bno_s1` reached 45.77 mm J14 PA at epoch 84 and was dead by 92, aborting on
`still stuck after 3 EMA rollback(s)`. Its twin `bno_s0`, identical config and
a different seed, skipped exactly **one** step in 100 epochs.

This is **not** the generation-4 failure. `--bound-scales` held: the anomalous
losses are 50-175, not the 4e7 / 8.8e13 of the size-parameter blowup, and the
skeleton stayed metre-scale. PA-MPJPE removes scale, so 341 mm means the
**pose** was destroyed — the 124 rotations at `mhr_params[6:130]`, which
`--bound-scales` deliberately leaves free.

It is also not the learning rate. The collapse ran at `lr 6.8e-5`, a quarter of
nominal, well past OneCycle's epoch-30 peak.

**The precursor, nine epochs early, that nobody was watching.** Epochs 75-84 ran
with `skipped 0/1562` every single epoch — no NaN, no loss over the threshold —
while the two validation branches disagreed wildly:

| epoch | RAW J14 PA | EMA J14 PA | skipped |
|---|---|---|---|
| 75 | 53.0 | 47.7 | 0 |
| 76 | 51.7 | **188.9** | 0 |
| 77 | 48.4 | 123.4 | 0 |
| 78 | 49.3 | 69.5 | 0 |
| 79 | 47.4 | 50.8 | 0 |
| 80-84 | 47-48 | 45.8-48.7 | 0 |
| **85** | **268.7** | 47.1 | 67 |
| 86 | 302.3 | 321.0 | 0 |

RAW is sampled once per epoch; the EMA integrates every step. So the epoch-76
spike says the weights took a large excursion *inside* an epoch that the
end-of-epoch snapshot missed entirely and the loss guard never saw. The EMA
decayed it out over ~3 epochs (half-life `ln2/(1-0.9998)` = 3,466 steps ≈ 2.2
epochs, which matches), everything looked fine again, and nine epochs later the
run died for real.

**`EMA / RAW` on the validation metric is a free excursion detector and is not
monitored.** Add it to the epoch line and to `summarize_runs.py`; a ratio above
~1.5 with `skipped 0` means the weights are wandering between snapshots.

**Each rollback made things immediately worse.** Both visible rollbacks are
followed by a markedly worse epoch:

| rollback | epoch after | train total | skipped |
|---|---|---|---|
| attempt 2 (end of ep 73) | 74 | 3.19 (was 2.64) | 74 |
| attempt 3 (end of ep 84) | 85 | **7.99** (was 2.60) | 67 |

Attempt 3 restored a **45.8 mm** EMA at `lr 6.8e-5` and the model was at 268 mm
one epoch later. Restoring good weights and then destroying them within 1,562
steps is not a learning-rate problem — the recovery mechanism is a prime
suspect, not just the failure.

**Ranked hypotheses, and how to test each.**

1. **The Adam restart in the rollback.** `optimizer.state.clear()` zeroes both
   moments, so the very next step is `lr * m̂/(√v̂+ε) = lr * sign(g)` for **all**
   40M parameters at once — a coordinated jump of norm `lr * √(4e7)` ≈ 0.43 at
   this LR, against the heavily damped updates steady-state Adam produces.
   *Test:* keep the moments, or re-warm the LR over ~200 steps after a
   rollback, and see whether the post-rollback epoch still degrades.
2. **The rotations are genuinely unbounded.** `--bound-scales` covers `0:3` and
   `130:204` only. `6:130` cannot blow up the skeleton's *size* but can produce
   arbitrary pose. *Test:* log `mhr_params[6:130]` percentiles per epoch; if
   they drift monotonically through epochs 75-84 the wandering is there.
3. **The guard's trigger is the wrong quantity.** It fires on
   `loss > 100 or non-finite`. Epochs 75-84 satisfied neither while the model
   visited bad regions. *Test:* trigger on gradient norm or on a weight-delta
   norm instead of, or as well as, the loss.

**Two log defects found while diagnosing this, both worth fixing:**

* `bad` is all-reduced with `ReduceOp.MAX` (`train_distill_jz.py:682`) but the
  message prints **rank 0's own** loss. That is why the log shows
  `skipping step 57 (loss 10.8)` against a threshold of 100 — a healthy rank
  vetoed by a sick one. Print the reduced maximum, or the vetoing rank.
* `📈 epoch` reports RAW and EMA side by side but nothing flags their
  divergence, which was the only visible symptom for nine epochs.

**Ruled out.** The safe-fallback path was *not* masking the collapse: `repaired`
fired **once** in the entire run, and only one epoch shows a `repaired` count.
The concern filed as item 6 is real bookkeeping sloppiness but had nothing to do
with this failure.

### 6. `n_repair` overstates successful updates

`GradScaler.step` silently skips the optimiser when gradients overflow, but
`train_distill_jz.py:715` / `:793` call `scheduler.step()` and
`ema.update_parameters()` regardless and count a repair. Consequences are
cosmetic — re-averaging unchanged weights into the EMA is harmless, and
stepping the scheduler through a skip is deliberate (freezing the LR during a
skip storm is what caused the earlier runaway). Worth making explicit so the
"repaired N" line means what it says.

### 7. Direct parameter supervision reaches 20–60% of samples

At `geom_p=0.8`, `geom_flip_p=0.5`: root rotation (`mhr_params[3:6]`) is
supervised on the 20% with no spatial augmentation; local angles, bone scales
and shape on the 60% that were not flipped. **Not a bug** — the dataset rotates
and mirrors the labels without rotating the parameters, so the masked-out
targets are genuinely wrong, and those samples still get full 3D geometric
supervision.

**The opportunity**: transform the targets properly instead of masking — mirror
the left/right parameter channels for flips, compose the roll into the root
rotation. Needs the rig's mirror mapping **verified, not assumed**; MHR's Euler
channels are not SMPL's permutation/sign convention.

### 8. Twelve redundant target dimensions

`parameter_transform[:, :204]` has rank 192. Moving 10 units along a null
direction changes the skeleton by `< 0.0002 mm`, yet `loss_pose` / `loss_scale`
still penalise the difference — 12 dimensions of solver-arbitrary teacher noise
no image can determine.

**Measure before acting**: how much teacher variance actually lies along those
directions, and how much trunk gradient they carry. The audit is explicit that
redundancy alone does not prove a cost. Do **not** "fix" it by zeroing the
`*_flexible` parameters — the teacher genuinely uses them
(`leg_length_flexible` mean 0.32, max 1.64) and zeroing shortens every leg.

### 9. The mesh is invisible to the geometry objective

`MHRForwardPass.get_joints` zeroes the 45 `shape_params` before the forward
pass; perturbing them by +2 sigma moves the 127-joint skeleton by exactly
`0.00e+00 cm`. They receive direct parameter loss (`w_shape = 0.03`) and
nothing else. So every geometric loss and every reported metric is blind to
body shape.

**Consequence for claims, today**: PA-MPJPE can improve while the mesh gets
worse. Do not say "better mesh" on the strength of a pose number.

**Half-addressed 2026-09-07 by `--w-verts`** (see item 13): the surface *term* now
exists and is the only loss whose gradient reaches `shape_params` — measured
`d(loss)/d(shape)` is `8.9e-03` through vertices and **exactly 0.000e+00**
through joints. What is still missing is the surface *metric*: nothing in
`benchmark/` reports PVE, so the claim side of this item is unchanged.

### 10. `fy` for both axes — checked, not a problem here

The conditioning uses one focal where a camera has two. Measured `fx/fy` over
300 samples per split:

| split | fx/fy |
|---|---|
| coco, aic, mpii | 1.00000 exactly |
| harmony4d | 0.98661 – 1.03656 |
| 3dpw | 0.99625 – 1.00376 |

The `bno` / `dt` mix is `aic/coco/mpii` only — perfectly isotropic — and 3DPW
differs by 0.4%. Revisit only if a new split with anisotropic pixels enters the
mix.


### 11. The rotation half of the CLIFF crop-centre correction is unverified

Item 3 fixed the zoom. The rotation is a separate question and it was **not**
resolved: with `geom_rot_deg=30` a ~31 px box-centre residual remains against a
preimage-of-crop-centre reference, and the zoom fix moves it by only 2.5 px.

Two readings, and the measurement does not separate them. Either the code is
right — it treats the in-plane rotation as a camera roll, rotating `joints_3d`,
`cam_trans` and the box centre about the image centre together, and under that
model the box is *supposed* to move, so the reference is simply wrong — or there
is a real error hiding under a reference too loose to see it. What was ruled
out: expressing the `dx`/`dy` correction in the rotated axes rather than the
unrotated ones, which changes the box-centre-to-body-centre distance from 63.75
to 63.15 px, i.e. nothing.

**To settle it** the reference has to come from the camera-roll model itself,
not from the warp: construct a synthetic camera, roll it by the augmentation
angle, and check that the conditioning the dataset emits equals the conditioning
that rolled camera would produce for the same box. Until then, do not "fix" the
rotation — the current form is not shown to be wrong.

### 12. Benchmark HGNetv2 against RepViT — after the baseline lands

Not to be run now. `repvit_m2_3` is the incumbent and generation 6 exists to
give it a solid, uncontaminated baseline on the five-split corpus; swapping the
trunk at the same time would confound both questions.

When it is time, the comparison has to cover all four axes, because a backbone
that wins on one can lose the deployment:

* **Model sizes** — which HGNetv2 variants exist and their parameter counts
  against `repvit_m2_3`'s. `--backbone` already takes any timm name, and
  `config_from_checkpoint` reads it back out of `run_config.json`, so no code
  change is needed to try one.
* **Accuracy** — J14 PA-MPJPE with the adapter, 2 to 3 seeds. The documented
  seed spread is 34–49 mm on the 70-keypoint metric, so a single run proves
  nothing.
* **Speed** — ONNX latency on the deployment target, not PyTorch training
  throughput. `tools/bench_student_arch.py` is the existing harness.
* **Capabilities** — HGNetv2 is a plain conv stack where RepViT is
  reparameterisable. Check that the export path survives (`tools/pth_to_onnx.py`
  has a torch↔ONNX parity gate that exits if the worst output differs by more
  than 1e-3) and that `backbone.forward_features` returns something the
  `feat_proj` reshape accepts — the trainer handles both 4-D and 3-D feature
  maps, but the channel count feeds `backbone_feat_dim` and that is a config
  field, not inferred.
* **On our data, locally** — the local corpus plus `checkpoints/mhr_model.pt` is
  enough to run the 1-batch overfit test and a short run per backbone before
  spending cluster time. Judge the overfit at ~1k steps.

### 13. PVE has no metric yet — only a loss

`--w-verts` supervises the surface (it is the fix half of item 9), but nothing
*reports* it. Competitors quote PVE, so a submission needs it.

The pieces are in place and the awkward part is already solved: the vertex
subset is drawn from the rig's **own** 18,439-vertex mesh, not from a separate
LOD asset, so predicted and teacher vertices share topology and vertex ids and
the correspondence is exact — no cross-topology fitting, which is what makes PVE
comparisons between rigs contentious. `MHRForwardPass.get_joints_and_vertices`
returns them.

**What is missing**: 3DPW ships SMPL, and MHR is not SMPL. A published PVE
number needs an MHR→SMPL surface correspondence with the same care the J14
adapter needed — float64 and a truncated `lstsq`, fitted on train sequences and
never refitted on the reported split. Until that exists, PVE can be reported
against the *teacher* on held-out crops, which measures distillation fidelity
and is honest as long as it is labelled as such.

**Do not raise `n_verts` for the metric.** 595 vertices is a sampling choice for
the loss; a reported PVE should use all 18,439, which costs 21.6 ms per batch-64
forward and is irrelevant at evaluation time.

### 14. Do the tanh size bounds block extreme poses? — measured, they do not

Raised because `vid4.mp4` looked poorly fitted after `--bound-scales` landed.
**The bound cannot be the cause, and here is why.**

`--bound-scales` squashes two slices through a `tanh`: `mhr_params[0:3]` (root
translation) and `[130:204]` (the six `*_flexible` size parameters and the 68
bone scales) — 77 parameters, all of them body SIZE. The 124 local joint
**rotations** at `[6:130]`, which is what a yoga pose actually is, and the root
rotation at `[3:6]` are **not bounded at all**.

Measured over 4,000 teacher annotations from each of the six local splits —
1,848,000 bounded targets in total:

| split | targets outside the bound | worst \|atanh\| | max \|rotation param\| |
|---|---|---|---|
| 3dpw | 0.0000% | 0.63 | 2.64 |
| aic | 0.0000% | 0.90 | 2.94 |
| coco | 0.0000% | 0.97 | 2.94 |
| harmony4d | 0.0000% | 0.59 | 2.56 |
| mpii | 0.0000% | 0.80 | 2.94 |
| distill_mix | 0.0000% | 1.63 | 2.84 |
| **all** | **0 / 1,848,000** | **1.63** | |

Not one target is clipped, and the worst sits at `|atanh| = 1.63`, i.e.
`tanh = 0.926` — a value a linear head reaches without saturating. The one real
cost is gradient damping, `1 - tanh²`, and on the teacher distribution that is
0.994x at the median, 0.875x at p99 and 0.733x at worst. Harmless.

**So look elsewhere for the `vid4.mp4` regression.** Two candidates, in order:

1. **A stale ONNX.** `models/instanthmr_mhr_only_ckpt90.onnx` and
   `models/instanthmr.onnx` carry **no `metadata_props` at all**, so
   `instanthmr.inference` cannot know how they were conditioned and falls back
   to the pixel-CLIFF form with a `1.00 x diag` focal. Feeding a focal-aware
   model that vector raises nothing and silently mis-places the person in depth.
   Every 2026-09-07 export (`models/instanthmr_mhr_only.onnx`, `models/gen5/*`)
   does carry `cliff_focal=true, bound_scales=true`. **Re-render `vid4.mp4` with
   a stamped export before treating this as a model regression.**
2. **The deployment focal fallback.** With no annotation to read a focal from,
   `demo.py` uses `1.05 x diag` for focal-aware graphs. If the real camera is
   far from that, depth is wrong — `--focal PX` overrides it.


---

## Evaluation hygiene the audit raised

Not code defects, but they gate a submission.

* **Sequence manifests.** The local corpus contains all 3DPW sequences in
  `sam3d_distill_mix`; cluster-trained checkpoints use a different, clean
  corpus, but the loader does not *enforce* the separation. Save a
  source-image / sequence manifest and exclude explicitly — including for any
  offline regressor or adapter fitting.
* **Adapter provenance.** The fixed teacher-derived J14 adapter is fitted on
  3DPW-train ground truth. That is target-dataset supervision and must be
  disclosed. Test it on held-out sequences; never refit on the reported split.
  The fitting script's halfway split is not an explicit sequence grouping.
* **Protocol parity.** The harness uses GT-derived boxes, its own crop
  expansion, a 60%-inside-image filter and its own person-frame list. Using the
  standard SMPL/H36M reference does not by itself make the protocol identical
  to a published one. Report GT vs detected boxes and known vs estimated
  intrinsics separately.
* **Test-set monitoring.** `--val-3dpw-test` is logged and never used for
  selection, which is right, but human design decisions can still overfit a
  number you watch every epoch.
* **Metrics.** PA-MPJPE removes translation, rotation and scale; root-relative
  MPJPE removes translation. Neither sees `cam_trans` drift, body-size error or
  jitter. Claims about stability need acceleration and bone-length-variance
  numbers, and low acceleration alone can just mean over-smoothing — compare
  against GT motion and report lag.
* **Seed spread.** The documented 34–49 mm spread is for the 70-keypoint
  metric, not for J14+adapter. Measure the spread of the metric you quote
  before calling any gap real.
