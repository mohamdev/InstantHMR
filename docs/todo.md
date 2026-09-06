# Open work, from the 2026-09-06 accuracy audit

What the external audit found, what was checked against the code, and what is
left. The audit itself is `docs/accuracy_audit_2026-09-06.md`; this file is the
verdict on each item and the queue.

Every claim below was reproduced on real data before being written down. Where
a number appears, the command that produced it is named.

---

## Done (2026-09-06)

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

## Queue

Ordered by (cost of being wrong) x (cheapness of the fix). Items 3, 4 and 6 are
defects; 5 and 7 onward are opportunities the audit raised that are not bugs.

### 3. The v2 crop-centre update forgets to divide by the zoom — **in flight**

`train_distill_mhr_only.py:678` writes `cx -= dx * crop_w / 2.0`. The
augmentation warps the crop with `u' = s*R*(u - c) + c + t`; inverting it puts
the visible window's centre at `-t/s`, so the correction needs the `/s`. It
over-corrects when zooming in and under-corrects when zooming out. Live in
every `--preset v2` run (`geom_trans=0.20`, `geom_scale_max=2.0`).

**Measured** over 972 augmented COCO/AIC/MPII samples drawn from the real v2
distribution:

| | px err mean | px p95 | deg mean | deg p95 | deg max |
|---|---|---|---|---|---|
| all | 11.7 | 33.6 | 0.61 | 1.73 | 3.21 |

0.6° of error in the bearing the network is told, on average — 4% of the crop's
own angular half-width, 13% at worst.

**It is a wrong input, not a wrong label.** Verified: feeding the teacher's own
3D joints and `cam_trans` through the reprojection loss's full geometry
(counter-rotate, un-mirror, project in the original frame, apply `aug_M`)
against the augmented 2D targets, over 64 samples with 34 flipped, gives a
residual of **0.000 px**. The augmented geometry chain is exact. And the third
conditioning channel — angular size, `b_scale / scale`, the one that sets depth
and motivated `--cliff-focal` — is already correct. The damage is confined to
lateral placement, which PA-MPJPE cannot see.

**Fix** (one character per line):

```python
cx -= dx * crop_w / (2.0 * scale)
cy -= dy * crop_w / (2.0 * scale)
```

**Do not patch this on the cluster while runs are chaining.**
`52_train_ddp.slurm` resubmits itself on walltime, so an rsync'd trainer changes
the augmentation partway through a 300-epoch run. Land it for the next
generation, and note in `STATUS.md` that generation 5 carried the uncorrected
version.

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
   (`p.joints_2d[COCO17_FROM_MHR70]`) and what `--detach-2d-head` degrades.
2. `mhr_params` + `cam_trans` -> forward kinematics -> project with the focal.
   Governed by `loss_reproj`, which the `dt` runs leave **unchanged** to within
   seed noise, so the trunk has not lost image-plane localisation — only the
   SimCC readout of it has.

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
Dropping `loss_2d_simcc` / `loss_2d_native` under `--detach-2d-head` is the one
free part — they already contribute exactly zero trunk gradient there.

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
**Consequence for work, later**: a surface term and a surface metric, with a
valid cross-topology correspondence.

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
