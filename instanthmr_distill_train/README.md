# InstantHMR Distillation — training scripts

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

Training auto-resumes from `runs/<name>/best_student_model_v3.pth` if present.
Outputs (best checkpoints + `export/*.onnx`) land under `--output_dir`.
