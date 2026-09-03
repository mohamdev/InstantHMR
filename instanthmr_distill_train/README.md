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
