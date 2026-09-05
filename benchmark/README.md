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
- **RICH** — the reason to skip it is not the size alone. Its GT is **SMPL-X**,
  so it needs the SMPL-X models *and* a second forward pass (different joint
  count, hand and face parameters) that none of our SMPL code covers; the test
  split alone is 125K images at 4K, against EMDB-1's few thousand; and NLF
  *trains* on RICH (the ∗ in SAM 3D Body's Table 2), so the column is not
  out-of-domain for every method in it.
- **MPII** — 2D only, so it adds little once COCO is in.

### Next: EMDB — waiting on registration (asked 2026-09-05)

**Status: blocked on the application at `emdb.ait.ethz.ch`** (a form, approved
by email). Nothing else is missing — the two blockers this file used to list,
the SMPL body model and their toolkit, are gone: `benchmark/data/smpl/` holds
the gendered models and the toolkit is only a viewer, not a format we need.

Use **EMDB-1** (17 sequences, 13.5 min), the camera-frame pose split every paper
reports. EMDB-2 (25 sequences) is global-trajectory and not what this model does.

It is structurally a simpler 3DPW, one subject per sequence:

| EMDB pkl | 3DPW equivalent |
|---|---|
| `smpl{poses_root (N,3), poses_body (N,69), betas (10,), trans}` | `poses` (N,72), `betas`, `trans` |
| `camera{intrinsics, extrinsics (N,4,4), width, height}` | `cam_intrinsics`, `cam_poses` |
| `good_frames_mask` | `campose_valid` |
| `bboxes` (N,4) xyxy — **provided** | derived from projected joints |

`betas` is 10-dim, exactly the count the published protocol uses, so
`make_3dpw_gt.py`'s forward pass drops in unchanged.

**The protocol is the 24 SMPL joints, mid-hip aligned — not the H36M
regressor.** NLF's Table 3 reads "Results on EMDB1 (24j)" and its metrics
section states that "on 3DPW and EMDB the reference point is the midpoint
between the two hip joints"; SAM 3D Body's Table 2 column header is "EMDB (24)".
Those 24 joints are what `smpl_forward` already returns as its second output.

Two things already done, so the integration is a loader and nothing else:

- `benchmark/results/adapter_smpl24_teacher.npz` — the MHR70 → SMPL-24
  conversion, fitted the same student-free way as the J14 one, from 22,209
  teacher/GT pairs on 3DPW train (where `jointPositions` **is** the SMPL-24 set,
  verified to 0.001 mm). Reads anatomically: `pelvis = 0.38 right_hip + 0.34
  left_hip`, `left_knee = 0.93 left_knee`, `left_ankle` from heel + ankle + toe.
- **The conversion floor: 25.11 mm MPJPE / 18.77 mm PA-MPJPE** on a held-out
  half. That is what an MHR-rigged model carries on a 24-joint benchmark before
  it makes a single mistake — an *upper* bound, since it also contains the
  teacher's own 3DPW error, which cannot be separated without perfect MHR GT.
  Quote it next to the result: the table it joins runs 61.7 (SAM 3D Body) to
  118.5 (HMR2.0b) MPJPE, so ~25 mm of rig overhead is material.

What is left is `benchlib/emdb.py`, ~80 lines mirroring `threedpw.py`. It was
deliberately **not** written ahead of the data: `dataset.md` does not say
whether `extrinsics` is world→camera or camera→world, and guessing wrong gives
quietly wrong numbers rather than a crash. Thirty seconds with a real pkl
settles it.

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

Then fit the MHR70 → J14 adapter once. It is student-free and dataset-wide, so
this is a one-time step, not something to repeat per checkpoint:

```bash
python benchmark/fit_adapter_mhr.py \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --out benchmark/results/adapter_j14_h36m_teacher.npz
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

# same thing for a training checkpoint instead of an exported graph. --adapter
# adds the J14+adapter row, which is the one to quote.
python benchmark/eval_3dpw_ckpt.py \
    --ckpt instanthmr_distill_train/runs/*/*/best_student_model_v3.pth \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --image-root   /path/to/3DPW/imageFiles --split test \
    --adapter benchmark/results/adapter_j14_h36m_teacher.npz
```

Both write a JSON report to `benchmark/results/`.

## Current numbers (2026-09-05)

3DPW **test**, all 35,463 person-frames, published-protocol GT, the two Jean Zay
checkpoints pulled at ~epoch 130. Full reports in
`benchmark/results/3dpw_test_{b3_s1,v3_s1}_teacher_adapter.json`.

| run | preset | J12 PA | J14 PA | **J14+adapter PA** | J14+adapter MPJPE |
|---|---|---|---|---|---|
| b3_s1 | baseline | 62.49 | 67.89 | **41.83** | 69.03 |
| v3_s1 | v2 | 63.20 | 69.05 | **43.23** | 70.59 |

**Quote the adapter row.** Raw J14 is dominated by convention, not quality:
per-joint it is 131 mm at `head` and 98/87 mm at the hips against ~47 mm at the
knees.

Two caveats before this goes in a table. The 1.4 mm gap between the two runs is
**not** a result — measured seed spread in this repo is far wider, so
baseline-vs-v2 needs 2-3 seeds per arm through the same path. And these are the
only numbers here that are genuinely held out: both checkpoints were trained on
Jean Zay, whose corpus is coco/mpii/aic/sa1b/harmony4d with `3dpw_train`
annotations-only, so all 24 test sequences are unseen. The local `data/` corpus
is **not** clean — see the contamination section.

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
- **J14+adapter** — a linear MHR70 → J14 regressor (14x70, no bias, applied to
  centred keypoints, so it is equivariant to rotation and translation and can
  only move landmarks, never fix a pose). It absorbs the rig offset and is the
  row to compare against a published table, not an optional extra.

  ```bash
  python benchmark/eval_3dpw.py --onnx ... --split train --stride 10 \
      --fit-adapter benchmark/results/adapter_j14.npz
  python benchmark/eval_3dpw.py --onnx ... --split test \
      --adapter benchmark/results/adapter_j14.npz
  ```

  Fit on train, report on test — never fit on the split you report. The two
  splits share no sequences, so this is a real holdout, and an adapter is
  specific to one model *and* one GT convention: applying b3_s1's to another
  checkpoint, or an `h36m` one to `jointpositions`, is meaningless.

  **Fit it from the teacher, not from the student.** `--fit-adapter` uses one
  checkpoint's own predictions as the input side, so the map can absorb that
  checkpoint's systematic error along with the rig offset — a different adapter
  per checkpoint, and a fair reader will call it tuning.
  `benchmark/fit_adapter_mhr.py` instead pairs the **teacher's** MHR labels in
  `data/sam3d_gt_3dpw` with the H36M GT on the same 3DPW train frames, giving
  one adapter for every checkpoint that never saw a student:

  ```bash
  python benchmark/fit_adapter_mhr.py --sequence-dir /path/to/3DPW/sequenceFiles \
      --out benchmark/results/adapter_j14_h36m_teacher.npz
  ```

  That it is a rig conversion rather than an error sponge is visible two ways.
  The teacher's own J14 error against the GT drops from **44.90 mm to 14.83 mm**
  on a held-out half — near the teacher's accuracy floor, which absorbing error
  could not reach. And every row reads anatomically:
  `right_hip = +1.16 right_hip - 0.32 left_hip` (widening SMPL's 120 mm hips
  toward H36M's 292 mm), `head = +0.57 left_ear +0.55 right_ear -0.47 neck`
  (extrapolating to the top of the skull), `right_elbow` from the cubital fossa,
  `right_shoulder` from the acromion, `right_knee = +0.84 right_knee` — nearly
  identity, because the knee is the same point in both conventions.

  The fit must also be truncated: see `RCOND` in `fit_adapter`, without which
  the map reaches the hips through +-2000 weights on near-collinear finger
  joints and stops being a joint regressor in any readable sense.

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
  make_3dpw_gt.py    SMPL forward + H36M regressor -> <3DPW root>/gt_h36m/
  fit_adapter_mhr.py MHR70 -> J14 adapter from TEACHER labels (student-free)
  eval_3dpw.py       3D from an ONNX graph: MPJPE, PA-MPJPE, PCK3D, AUC
  eval_3dpw_ckpt.py  the same, for a .pth straight out of runs/
  eval_coco.py       2D: OKS AP/AR, mean OKS, PCK, NME, per-keypoint breakdown
  benchlib/
    joints.py        MHR70 <-> COCO17 / SMPL24 / H36M17 maps, J14 & J12 sets
    metrics.py       Procrustes alignment and the metric definitions
    threedpw.py      3DPW pickle loading, the GT switch, projection, boxes
    runner.py        batched ONNX inference with overlapped image decoding
```
