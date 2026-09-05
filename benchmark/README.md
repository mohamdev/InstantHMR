# InstantHMR benchmarks

Evaluation harness for publishing numbers on `models/instanthmr_mhr_only_ckpt90.onnx`.

Two benchmarks are implemented, chosen because they are the two cheapest paths
to a credible table — one 3D, one 2D:

| Benchmark | What it measures | Data cost | Status |
|---|---|---|---|
| **3DPW** test | MPJPE / PA-MPJPE, the standard in-the-wild 3D HMR metric | 4.6 GB images + 30 MB GT + 500 MB SMPL | ready — published-protocol GT; **check contamination below** |
| **COCO** val2017 | OKS AP / PCK, 2D keypoint localisation in the wild | 820 MB, direct download, no registration | ready — clean |

## Why these two

**3DPW** is the benchmark essentially every monocular HMR paper reports, so it
is the one that makes your numbers comparable. The GT pickles store
`jointPositions` (24 SMPL joints in world coordinates) directly, which is
tempting and wrong — published numbers use SMPL run *forward* on the GT
`poses`/`betas` with the Human3.6M regressor applied to the vertices. The two
references are **46 mm apart in J12 PA-MPJPE**, so the harness builds the real
one once with `make_3dpw_gt.py` and scores against that.

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

Then build the ground truth. This needs two files that cannot be redistributed
here — put both in `benchmark/data/smpl/` (gitignored):

| file | where |
|---|---|
| `SMPL_{MALE,FEMALE}.pkl` | `smpl.is.tue.mpg.de`, free research licence. 3DPW's `genders` field is per-subject, so the gendered pair is required. |
| `J_regressor_h36m.npy` | `visiondata.cis.upenn.edu/spin/data.tar.gz` (SPIN); also ships inside HMR2.0 / 4D-Humans |

```bash
python benchmark/make_3dpw_gt.py \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --smpl-dir benchmark/data/smpl --check
```

That writes `<3DPW root>/gt_h36m/{test,validation,train}.npz` — 17 H36M joints
per person-frame in the same world frame as `jointPositions`, 14 MB for all
three splits. `--check` additionally regresses the SMPL kinematic joints and
compares them to `jointPositions`: **0.001 mm max over 74,620 person-frames**,
which is what certifies the forward pass, the gendered model choice and the
world frame all the way through.

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

# same thing for a training checkpoint instead of an exported graph
python benchmark/eval_3dpw_ckpt.py \
    --ckpt instanthmr_distill_train/runs/b3_s1/b3_s1/best_student_model_v3.pth \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --image-root   /path/to/3DPW/imageFiles --split test
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

### The ground truth — what it is, and what it used to be

The harness scores against the **published-protocol GT**: gendered SMPL run
forward on each frame's `poses`/`betas`/`trans`, then `J_regressor_h36m.npy`
applied to the 6890 vertices. This is the reference every 3DPW paper uses —
NLF's included ("obtained through the same Human3.6M-style joint regressor that
all prior works use"). `benchmark/make_3dpw_gt.py` precomputes it; `--gt
jointpositions` restores the pickles' raw `jointPositions` field, which is what
this harness used before and is **not** comparable to any published table.

The difference is not cosmetic. Over 1,437 test person-frames, the same
landmarks under the two references:

| | raw | root-relative | after Procrustes |
|---|---|---|---|
| J14 mean | 64.2 mm | 86.3 mm | 47.2 mm |
| J12 mean | 63.6 mm | 88.1 mm | 46.0 mm |
| right hip | 154.5 mm | 86.8 mm | 114.5 mm |
| head | 116.3 mm | 42.0 mm | 62.2 mm |

The hips dominate, because the two conventions disagree about where a hip is:
H36M's are marker-derived and **292 mm apart**, SMPL's kinematic ones sit near
the femoral heads, **120 mm apart**. Shoulder width goes the other way (303 vs
387 mm), and H36M's `head` is the top of the skull where SMPL's is inside it
(neck→head 192 vs 100 mm).

Two details worth keeping straight:

- **`jointPositions` is the kinematic joint, not the regressed one.** Comparing
  `J_regressor @ posed vertices` to it gives a 4.5 mm residual and looks like a
  bug in the forward pass. The joints that match are the ones carried by the
  global transforms. `make_3dpw_gt.py --check` compares those, and gets 0.001 mm.
- **`betas` is 300-dim for 48 of the 87 subject tracks and 10-dim for the other
  39, and entries 10: are all exactly zero.** So `--num-betas 10`, the published
  protocol, is also lossless here.

`v_template_clothed` in the pickles is not a substitute for any of this — it is
a static T-pose surface with no blend weights and no pose blendshapes, so it
cannot be posed. Using it makes the residual worse (17.9 mm), not better.

Selection inside `instanthmr_distill_train/val3dpw.py` follows the same default,
so **3DPW numbers from before and after this change are not comparable** — the
new GT is several mm harsher. The cache has to be rsynced next to the cluster's
`sequenceFiles/` or the trainer raises on startup.

### The joint-convention caveat — do not skip this

InstantHMR predicts MHR70 joints; the GT is now in the H36M convention. The two
place several landmarks differently, which puts a floor under the error that has
nothing to do with model quality. The harness therefore reports three rows:

- **J14** — the 14-joint LSP set most 3DPW papers use. `neck` and `head` carry
  the largest systematic penalty (MHR has no head-top; `nose` is substituted).
- **J12** — limbs only (shoulders, elbows, wrists, hips, knees, ankles). Against
  `jointpositions` this was the clean row, because MHR and SMPL agree on those
  landmarks. Against the H36M GT it is **not** clean: the hips alone move 292 vs
  120 mm apart between the conventions. Read it as a consistent tracking metric,
  not as a rig-free measure of the model.
- **J14+adapter** is therefore the row to compare against a published table,
  not an optional extra.
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
