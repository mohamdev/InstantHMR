# InstantHMR benchmarks

Evaluation harness for publishing numbers on `models/instanthmr_mhr_only_ckpt90.onnx`.

Two benchmarks are implemented, chosen because they are the two cheapest paths
to a credible table — one 3D, one 2D:

| Benchmark | What it measures | Data cost | Status |
|---|---|---|---|
| **3DPW** test | MPJPE / PA-MPJPE, the standard in-the-wild 3D HMR metric | 4.6 GB images + 30 MB GT, free licence form | ready — **but the test split is contaminated, see below** |
| **COCO** val2017 | OKS AP / PCK, 2D keypoint localisation in the wild | 820 MB, direct download, no registration | ready — clean |

## Why these two

**3DPW** is the benchmark essentially every monocular HMR paper reports, so it
is the one that makes your numbers comparable. It is also unusually cheap here:
the GT pickles store `jointPositions` (24 SMPL joints in world coordinates)
directly, so **no SMPL model download or body-model forward pass is needed** —
which is what makes most 3DPW harnesses painful to set up.

**COCO val2017** needs no registration at all and the joint mapping is free:
MHR70 indices 0–14 already follow COCO's ordering, with the wrists at 41/62.

### The ones deliberately skipped

- **Human3.6M** — the licence forbids redistribution, the download is manual and
  slow, and protocol variants (P1 vs P2, 17 vs 14 joints) differ per paper. High
  effort, high chance of an unfair comparison.
- **EMDB** — a genuinely good modern benchmark (EMDB-1 for pose), but needs
  registration *and* the SMPL body model *and* their toolkit to unpack. Worth
  adding as a second 3D benchmark after 3DPW is in place; `benchlib/threedpw.py`
  is the template to copy.
- **RICH** — SMPL-X GT, large download, mostly used for contact estimation.
- **MPII** — 2D only, so it adds little once COCO is in.

## Setup

```bash
# COCO — fully automatic (~820 MB)
python benchmark/download.py --coco

# 3DPW — accept the licence at
#   https://virtualhumans.mpi-inf.mpg.de/3DPW/licence.html
# then unzip sequenceFiles.zip (and imageFiles.zip if you don't have it) and check:
python benchmark/download.py --check-3dpw \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --image-root   /path/to/3DPW/imageFiles
```

## Running

```bash
# COCO val2017 (~10 min on an RTX 4070)
python benchmark/eval_coco.py \
    --onnx models/instanthmr_mhr_only_ckpt90.onnx \
    --coco-root benchmark/data/coco

# 3DPW test — add --stride 10 for a 10x faster smoke run
python benchmark/eval_3dpw.py \
    --onnx models/instanthmr_mhr_only_ckpt90.onnx \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --image-root   /path/to/3DPW/imageFiles \
    --exclude-seen data/sam3d_distill_mix/annotations data/sam3d_gt_3dpw/annotations
```

Both write a JSON report to `benchmark/results/`.

## Protocols, stated precisely

Write these into the paper — they are the details a reviewer will ask about.

**3DPW test.** All 24 test sequences, every frame where `campose_valid` is set
and the person is at least 60 % inside the image. Person boxes come from the
projected GT joints padded by `--bbox-scale` (default 1.2), so no detector is in
the loop. MPJPE is mid-hip-aligned; PA-MPJPE is Procrustes (scale + rotation +
translation). Predictions use `joints_3d_local`, the rig-local pose, so the
camera translation never enters the metric.

**COCO val2017.** Top-down with **ground-truth boxes**, one crop per annotated
person with `num_keypoints ≥ 1` and `iscrowd = 0`. These AP numbers are *not*
comparable to COCO-leaderboard AP, which includes detection. InstantHMR emits no
per-keypoint confidence, so every keypoint is scored 1.0 — read AP next to the
confidence-free `mean_OKS` and `PCK` numbers.

### The GT-joint caveat — READ THIS BEFORE QUOTING A 3DPW NUMBER

**The ground truth this harness uses is not the one 3DPW papers report on.**
`benchlib/threedpw.py` reads `jointPositions` straight from the sequence
pickles: the 24 **SMPL kinematic-tree** joints. Every published 3DPW number —
NLF's included ("obtained through the same Human3.6M-style joint regressor that
all prior works use") — instead runs SMPL *forward* on the GT `poses` / `betas`
and applies the H36M joint regressor to the resulting 6890 vertices. Those are
different landmarks, most visibly at the shoulders and hips, which is 4 of the
12 joints in J12.

That is what makes this harness cheap to set up (see "Why these two" above) and
it is fine for *tracking a run*. It is **not** comparable to a published table.

Fixing it needs two files that are not in this repo:

| file | where |
|---|---|
| `SMPL_{MALE,FEMALE,NEUTRAL}.pkl` | `smpl.is.tue.mpg.de`, free research licence. 3DPW's `genders` field is per-subject, so get the gendered pair. |
| `J_regressor_h36m.npy` | ships inside SPIN / HMR2.0 / 4D-Humans |

Then: `V = SMPL(poses, betas)` per frame, `J14 = J_regressor @ V`, and fit the
existing `--fit-adapter` linear MHR70 -> J14 map on 3DPW **train** against that
GT before reporting on test. A forward pass, not a fit.

`v_template_clothed` in the pickles cannot substitute — it is a static T-pose
surface with no blend weights, no pose blendshapes and no joint regressor, so it
cannot be posed.

This does **not** affect `instanthmr_distill_train/val3dpw.py`, which uses the
same GT for checkpoint selection: selection only needs a consistent metric, not
a comparable one.

### The joint-convention caveat — do not skip this

InstantHMR predicts MHR70 joints; 3DPW's GT is SMPL. The two rigs place several
landmarks differently, which puts a floor under the error that has nothing to do
with model quality. The harness therefore reports three rows:

- **J14** — the 14-joint LSP set most 3DPW papers use. `neck` and `head` are the
  two joints where the rigs genuinely disagree (MHR has no head-top; `nose` is
  substituted), so this row carries a systematic penalty.
- **J12** — limbs only (shoulders, elbows, wrists, hips, knees, ankles), where
  both rigs mean the same anatomical point. **This is the cleanest measure of
  the model.**
- **J14+adapter** *(optional)* — a linear MHR70 → SMPL-J14 joint regressor,
  fitted on 3DPW **train**, that absorbs the rig offset. Standard practice when
  a method's joint convention differs from the benchmark's. Fit it once, then
  apply to test:

  ```bash
  python benchmark/eval_3dpw.py --onnx ... --split train --stride 10 \
      --fit-adapter benchmark/results/adapter_j14.npz
  python benchmark/eval_3dpw.py --onnx ... --split test \
      --adapter benchmark/results/adapter_j14.npz
  ```

  Fit on train, report on test — never fit on the split you report.

### Train/test contamination — READ THIS BEFORE PUBLISHING

**COCO val2017 is clean.** All 25,594 COCO images in `data/` are
`COCO_train2014_*`. COCO's 2017 split re-partitions the 2014 data such that
val2017 is a 5k subset of val2014, which is disjoint from train2014. Verified by
image-id intersection: **0 of 5000** val2017 images appear in training.

**3DPW test is contaminated.** `data/sam3d_gt_3dpw` holds only train sequences,
but `data/sam3d_distill_mix` contains teacher-distilled crops from **all 60**
3DPW sequences — including all 24 test sequences (9,060 person-crops, ~28.5k
distinct frames across the dataset). `--exclude-seen` quantifies this: it splits
the report into frames the model saw during training and frames it did not.

Note that even the "not in training data" subset is only *frame*-level held out.
Neighbouring frames of the same person in the same scene were seen, so it still
overstates true generalisation. **To publish a 3DPW number, retrain with the 24
test sequences removed from `sam3d_distill_mix`.** The contamination-split
report tells you how large the effect is in the meantime.

## Layout

```
benchmark/
  download.py        COCO fetcher + 3DPW presence check
  eval_3dpw.py       3D: MPJPE, PA-MPJPE, PCK3D, AUC, per-sequence breakdown
  eval_coco.py       2D: OKS AP/AR, mean OKS, PCK, NME, per-keypoint breakdown
  benchlib/
    joints.py        MHR70 <-> COCO17 / SMPL24 index maps, J14 & J12 sets
    metrics.py       Procrustes alignment and the metric definitions
    threedpw.py      3DPW pickle loading, projection, box derivation
    runner.py        batched ONNX inference with overlapped image decoding
```
