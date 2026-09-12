# InstantHMR benchmarks

Evaluation harness for publishing numbers on `models/instanthmr_mhr_only_ckpt90.onnx`.

Three benchmarks are implemented — two 3D, one 2D:

| Benchmark | What it measures | Data cost | Status |
|---|---|---|---|
| **3DPW** test | MPJPE / PA-MPJPE, the standard in-the-wild 3D HMR metric | 4.6 GB images + 30 MB GT + 500 MB SMPL | ready — published-protocol GT; **check contamination below** |
| **EMDB-1** | MPJPE / PA-MPJPE on the 24 SMPL joints, the "EMDB (24)" column | 16 GB images (of a 50 GB download) + 7 MB GT | ready — **and genuinely clean**, see below |
| **COCO** val2017 | OKS AP / PCK, 2D keypoint localisation in the wild | 820 MB, direct download, no registration | ready — clean |

## Why these three

**3DPW** is the benchmark essentially every monocular HMR paper reports, so it
is the one that makes your numbers comparable. The GT pickles store
`jointPositions` (24 SMPL joints in world coordinates) directly, which is
tempting and wrong — published numbers use SMPL run *forward* on the GT
`poses`/`betas` with the Human3.6M regressor applied to the vertices. The two
references are **46 mm apart in J12 PA-MPJPE**, so the harness builds the real
one once with `make_3dpw_gt.py` and scores against that.

**EMDB-1** is the newer of the two 3D benchmarks and the one this repo can
report honestly. Every recent paper carries an "EMDB (24)" column (SAM 3D Body
Table 2, NLF Table 3, Fast SAM 3D Body Table 1), it is out-of-domain for almost
all of them, and — unlike 3DPW — **no EMDB frame appears in any corpus this
repo trains on**, so it is the only 3D number here with no contamination
caveat attached. It is structurally a simpler 3DPW: one subject per sequence,
provided person boxes, real calibration.

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

### EMDB-1 — the joint convention, settled against the data

Use **EMDB-1** (17 sequences, 24,323 frames), the camera-frame pose split every
paper reports. EMDB-2 (25 sequences) is global-trajectory and not what this
model does. Three sequences are in both; `benchlib/emdb.py` selects on the
pickles' own `emdb1` flag and asserts it finds exactly 17.

It is structurally a simpler 3DPW, one subject per sequence:

| EMDB pkl | 3DPW equivalent |
|---|---|
| `smpl{poses_root (N,3), poses_body (N,69), betas (10,), trans}` | `poses` (N,72), `betas`, `trans` |
| `camera{intrinsics, extrinsics (N,4,4), width, height}` | `cam_intrinsics`, `cam_poses` |
| `good_frames_mask` | `campose_valid` |
| `bboxes` (N,4) xyxy — **provided** | derived from projected joints |

`betas` is 10-dim, exactly the count the published protocol uses, so
`make_3dpw_gt.py`'s forward pass drops in unchanged, and EMDB's `gender` field
picks the same gendered pair 3DPW needs.

**The protocol is the 24 SMPL joints, mid-hip aligned — not the H36M
regressor.** NLF's Table 3 reads "Results on EMDB1 (24j)" and its metrics
section states that "on 3DPW and EMDB the reference point is the midpoint
between the two hip joints"; SAM 3D Body's Table 2 column header is "EMDB (24)".

**Which 24, though — and which way the extrinsics point — came from the data,
not the docs.** Both would have produced quietly wrong numbers rather than a
crash, and EMDB's own `kp2d` field settles both at once, because it is the 24
joints already projected:

| hypothesis | reprojection error vs `kp2d` |
|---|---|
| `extrinsics` as world→camera, **kinematic** joints | **0.00 px** ✅ |
| `extrinsics` as world→camera, `J_regressor @ vertices` | 1.04 px |
| `inv(extrinsics)` as world→camera | 3477 px, every joint behind the camera |

So `extrinsics` is world→camera (3DPW's sense), and "EMDB (24)" means the SMPL
**kinematic-tree** joints — `smpl_forward`'s second output — not the regressed
ones, which sit 3.8 mm away on average and 19.3 mm at worst. This is the same
distinction that bites on 3DPW, pointing the other way: there the published GT
*is* the regressed one.

`make_emdb_gt.py --check` re-runs that comparison over all 24,323 frames and
reports **0.0024 px mean / 0.095 px max**, which certifies the SMPL forward, the
gendered model choice, the beta padding and the world frame in one number.

**The MHR70 → SMPL-24 adapter is required, not optional.** Ten of SMPL's 24
joints — the three spines, the collars, the feet, the hands — have no MHR
landmark at all, so there is no raw row to fall back on.
`benchmark/results/adapter_smpl24_teacher.npz` is the student-free conversion,
fitted by `fit_adapter_mhr.py --target smpl24` from 22,209 annotation/GT pairs
on 3DPW **train** (where `jointPositions` **is** the SMPL-24 kinematic set,
verified to 0.001 mm). EMDB is a different dataset, so it is a real holdout.
It reads anatomically: `pelvis = 0.38 right_hip + 0.34 left_hip`, `left_knee =
0.93 left_knee`, `left_ankle` from heel + ankle + toe.

**The conversion floor: 25.11 mm MPJPE / 18.77 mm PA-MPJPE** on a held-out half.
That is what an MHR-rigged model carries on a 24-joint benchmark before it makes
a single mistake, using this linear conversion. Quote it next to the result: the table it joins runs 61.7 (SAM 3D Body) to 118.5 (HMR2.0b)
MPJPE, so ~25 mm of rig overhead is material.

### Two routes into SMPL space, and why the mesh one wins

The linear adapter above converts *joints*. `benchmark/eval_smpl_fit.py`
converts the **mesh** instead — the route SAM 3D Body and Fast SAM 3D Body use
for this same rig — which lowers the conversion floor ~2x and is the only way
to reach PVE at all:

| conversion | MPJPE floor | PA-MPJPE floor | PVE floor |
|---|---|---|---|
| linear MHR70 → SMPL24 adapter | 25.11 | 18.77 | not possible |
| **mesh fit** (`eval_smpl_fit.py`) | **11.44** | **10.66** | **13.61** |

Measured as a round trip through the rig: GT SMPL → fit MHR to that surface →
convert back → score against the untouched GT. Every stage is cold-started, so
this is the floor the evaluator actually operates at, not a best case.

**The direction is the part that makes it publishable.** The prediction is
fitted into SMPL and the ground truth is left exactly as the benchmark defines
it, so MPJPE/PA-MPJPE/PVE mean what they mean in every published table. The
tempting inverse — solving an MHR ground truth for EMDB and scoring in MHR
space — removes the conversion entirely and is **not** a comparable number: it
changes the joint set (MPJPE is not rig-independent), it averages error over 40
finger joints millimetres apart, and it scores against a reference we generated
with an optimiser told to look like an MHR model, so whatever MHR cannot
represent silently leaves the error term. Keep that as a diagnostic if you want
it; do not put it in the table.

**Three components, and the one that was hard.**

1. The student's MHR parameters → the rig's own 18,439-vertex mesh.
2. **Meta's official barycentric surface map**
   (`facebookresearch/MHR`, `tools/mhr_smpl_conversion/assets`, Apache-2.0) →
   the same surface resampled in SMPL topology. Copy
   `{mhr2smpl,smpl2mhr}_mapping.npz` into
   `benchmark/data/mhr_smpl_conversion/`.
3. A labelled fit for SMPL `theta`/`beta`/`trans`.

Step 2 is what makes step 3 tractable, and it is worth being explicit about
why. Fitting the surface by ICP has to *discover* the correspondence, and its
search has a local minimum that its own residual cannot detect: the mesh
settles overlapping the target while limbs are matched to the **wrong limbs**,
which measured a healthy-looking 12 mm surface residual alongside 76 mm
PA-MPJPE. Three attempts to build a correspondence here all failed — a mean
offset in camera space is not pose-invariant (60 mm), an affine combination of
the K nearest MHR vertices is ill-conditioned because a local neighbourhood on
a smooth mesh spans a tangent plane and leaves the normal unresolved (65 mm,
weights reaching 405), and a surface-frame offset is sensitive to
under-converged fits. With the official map, correspondence is **known** —
vertex i pairs with vertex i — and the failure mode does not exist.

Verify the map against this rig before trusting it: triangle ids top out at
36,871 against the rig's 36,874 faces, barycentric rows sum to 1, and the
reconstructed SMPL-topology mesh has a mean nearest-vertex spacing of 13.88 mm
against real SMPL's 14.37 mm. A map built for a different mesh would scatter
neighbouring vertices onto unrelated triangles.

**PVE is the first metric in this repo that can see body shape at all.** The
45 MHR `shape_params` move the 127-joint skeleton by exactly `0.00e+00 cm`, so
every joint metric here — MPJPE, PA-MPJPE, the adapter rows, the training
losses — is blind to identity by construction (see the note in `CLAUDE.md`).
The mesh conversion feeds those parameters into `MHRRig.vertices`, so they
reach the SMPL surface and PVE responds to them. If `--w-verts` is doing
anything, this is the benchmark row where it shows up.

One more detail in step 3: **solve the global orientation in closed form.**
Known correspondence makes that a plain Procrustes between SMPL's rest mesh and
the target. Without it, Adam starts from identity and has to walk the whole way
through axis-angle, and the same fit stalls at 40-51 mm of vertex residual
instead of 13 mm.

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
    --target j14 --out benchmark/results/adapter_j14_h36m_teacher.npz
```

### EMDB

Registration is a form at `emdb.ait.ethz.ch`, approved by email; the download is
ten per-subject zips, 50 GB. **Only EMDB-1 is needed** — 17 of the 81 sequences,
16 GB of images — so unpack selectively rather than extracting all ten zips:

```python
# python3, from the download directory
import zipfile, pickle
out = '/path/to/EMDB'
for p in range(10):
    z = zipfile.ZipFile(f'P{p}.zip')
    for n in z.namelist():                      # pass 1: the annotations
        if n.endswith('.pkl'):
            z.extract(n, out)
want = {'/'.join(f.split('/')[-3:-1])
        for f in __import__('glob').glob(out + '/P*/*/*.pkl')
        if pickle.load(open(f, 'rb'))['emdb1']}
for p in range(10):                             # pass 2: only EMDB-1 frames
    z = zipfile.ZipFile(f'P{p}.zip')
    for i in z.infolist():
        if i.filename.endswith('.jpg') and '/'.join(i.filename.split('/')[:2]) in want:
            z.extract(i, out)
```

Then build the GT — the same gendered SMPL models as 3DPW, nothing new to fetch:

```bash
python benchmark/make_emdb_gt.py --emdb-root /path/to/EMDB \
    --smpl-dir benchmark/data/smpl --check
```

That writes `<EMDB root>/gt_smpl24/emdb1.npz` (7 MB, 24 world joints per frame).
`--check` must print a reprojection error of ~0.00 px against EMDB's `kp2d`; a
larger number means the extrinsics or the joint convention moved.

The MHR70 → SMPL-24 adapter is fitted once, from 3DPW train, student-free:

```bash
python benchmark/fit_adapter_mhr.py \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --target smpl24 --out benchmark/results/adapter_smpl24_teacher.npz
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

```bash
# EMDB-1, all 17 sequences (~4 min on an RTX 4070). --adapter defaults to the
# committed SMPL-24 map, so the short form is the full protocol.
python benchmark/eval_emdb_ckpt.py \
    --ckpt instanthmr_distill_train/runs/*/*/best_student_model_v3.pth \
    --emdb-root /path/to/EMDB \
    --out benchmark/results/emdb1.json

# ...and again with EMDB's own person boxes, which is what published
# "Oracle: annotated bounding boxes" rows use. State which one you quote.
python benchmark/eval_emdb_ckpt.py --ckpt <ckpt> --emdb-root /path/to/EMDB \
    --bbox annotated
```

```bash
# EMDB-1 in SMPL space via the mesh conversion: MPJPE, PA-MPJPE and PVE.
# ~25 min for all 24,103 frames on an RTX 4070 (fit is 16 min of that).
python benchmark/eval_smpl_fit.py \
    --ckpt instanthmr_distill_train/runs/*/*/best_student_model_v3.pth \
    --emdb-root /path/to/EMDB --fit-batch 128 \
    --out benchmark/results/emdb1_smplfit.json
```

All four write a JSON report to `benchmark/results/`.

### The evaluator configures itself from the checkpoint (fixed 2026-09-06)

`eval_3dpw_ckpt.py` used to build a **default** `DistillConfig`, which silently
disagreed with any run that used one of the 2026-09-06 flags:

* `--bound-scales` puts a `tanh` remap of the 77 body-size parameters inside
  `forward()`. Exporting or scoring without it reads the head's raw pre-`tanh`
  numbers as bone scales — **21.0 mm MPJPE / 14.4 mm PA-MPJPE** of corruption,
  measured on 64 COCO teacher targets.
* `--cliff-focal` changes what the conditioning vector *means*, not the graph,
  so feeding the wrong form raises nothing.

Both are now restored per checkpoint by `T.config_from_checkpoint()`: the
bounds from the weights (`scale_lo` / `scale_hi` are the flag), the
conditioning form from the `run_config.json` beside the checkpoint. The script
prints what it inferred:

```
[cfg] bno_s0/best_student_model_v3.pth: backbone=repvit_m2_3 bound_scales=True cliff_focal=True [run_config.json]
[3DPW] test / GT=h36m: 35,463 person-frames (stride 1), CLIFF conditioning angular (real per-sequence focal)
```

Check that line. `cliff_focal=False [DEFAULT ...]` on a run you launched with
`--cliff-focal` means `run_config.json` did not travel with the checkpoint —
pass `--cliff-focal` by hand. The checkpoint load is **strict**, so a bounded
checkpoint scored through the wrong config raises instead of quietly producing
a number. Checkpoints that disagree on the conditioning cannot share one
invocation, because the crops are built once; the script says so and exits.

The focal itself comes from 3DPW's own calibration (`cam_intrinsics`,
`K[1,1]` = 1969.23 px on 10 of the 12 validation sequences, 1961.85 on the two
landscape ones) — not a heuristic. The naive `sqrt(H^2+W^2)` stand-in would be
2202.9 px, 12% high. `eval_3dpw.py` (the ONNX path) now passes the same
per-frame focal through `benchlib/runner.py`.

## Current numbers

### 3DPW (2026-09-05)

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

### EMDB-1 (2026-09-07)

All 17 EMDB-1 sequences, all 24,103 evaluatable frames (24,323 total, minus 206
where `good_frames_mask` is 0 and 14 where the subject is under 60 % inside the
frame), `b3_s1` at epoch 137 — the same checkpoint as the 3DPW table above.
Reports in `benchmark/results/emdb1_b3_s1_{gt-joints,annotated}.json`.

| person box | **SMPL24+adapter PA** | SMPL24+adapter MPJPE | PCK@50 | PCK@100 |
|---|---|---|---|---|
| projected GT joints ×1.2 (this repo's 3DPW convention) | **53.82** | 79.90 | 0.632 | 0.900 |
| EMDB's own `bboxes` (published "Oracle" protocol) | **60.69** | 87.51 | 0.582 | 0.863 |
| *rig conversion floor (teacher, held-out half)* | *18.77* | *25.11* | | |

Where that sits in the published EMDB (24) table — Fast SAM 3D Body Table 1,
which is the most recent one carrying every baseline:

| method | PA-MPJPE | MPJPE |
|---|---|---|
| Fast SAM 3D Body (oracle) | 37.3 | 64.3 |
| SAM 3D Body (oracle) | 38.2 | 61.7 |
| NLF-L +fit | 40.9 | 68.4 |
| PromptHMR | 41.0 | 71.7 |
| GENMO | 42.5 | 73.0 |
| CameraHMR | 43.3 | 70.3 |
| TRAM | 45.7 | 74.4 |
| WHAM | 50.4 | 79.7 |
| **InstantHMR b3_s1** | **53.8 / 60.7** | **79.9 / 87.5** |
| SMPLer-X-H | 64.5 | 92.7 |
| HMR2.0b | 79.2 | 118.5 |

Read that with the 18.8 mm conversion floor in mind: those baselines all predict
SMPL directly and pay no rig-conversion cost, and the floor already contains the
teacher's own error. It is not separable from the model's error without perfect
MHR ground truth, so the honest statement is "53.8 mm, of which up to 18.8 mm is
the MHR→SMPL conversion", not a corrected number.

**The box convention is worth 6.9 mm, so state it.** EMDB's own boxes are 17.5 %
larger than the projected-joint boxes (mean crop side 1248 vs 1069 px) and their
centre sits 5.7 % of a crop away, so the person lands ~15 % smaller and
off-centre in a 224² crop the model was trained to see filled. That is framing
sensitivity, not a modelling difference — but it means the row that belongs next
to the published "Oracle" numbers is the 60.69 one, and the 53.82 one is what
compares to this repo's own 3DPW numbers.

**The whole EMDB path was cross-checked against 3DPW.** Scoring the same
checkpoint on 3DPW test under the *same* GT convention (`--gt jointpositions`,
i.e. SMPL kinematic joints) gives J14 PA 55.81 / J12 PA 49.26, against EMDB-1's
60.17 / 55.55 on the identical rows. A 4-6 mm gap is what a harder dataset
looks like — EMDB has handstands, cartwheels and dancing where 3DPW mostly
walks. A wrong extrinsics sense or a mismatched joint convention would show up
as hundreds of millimetres, not four.

**EMDB is the one clean 3D number here.** The corpus is coco / mpii / aic / sa1b
/ harmony4d / 3dpw; no EMDB frame appears in any of it, at any split. Contrast
the 3DPW section below, where the local corpus contains crops from all 24 test
sequences.

Hardest sequences are the ones you would guess: `P5_42_indoor_dancing` (75.4 mm)
and `P8_68_outdoor_handstand` (75.0 mm), against 44.5 mm on `P1_14_outdoor_climb`.
Per-joint, the error is concentrated in the extremities the adapter has to
extrapolate — `left/right_hand` at 97-98 mm and the wrists at 79-82 mm, against
30-40 mm at the collars, neck, pelvis and hips.

### `--fit-iters 400` is too low by ~4x — every number below it is inflated (2026-09-11)

**The default was not converged**, and it is the reason this repo carried three
mutually inconsistent conversions. On 3DPW test the fit leaves **16.13 mm** of
surface residual at 400 iterations and **7.96 mm** at 4000; the metrics move
with it, by far more than any model change on the roadmap:

| fit-iters | residual | PVE | J14 PA | SMPL24 PA |
|---|---|---|---|---|
| 400 | 16.13 | 82.34 | 45.95 | 49.16 |
| 1500 | 8.94 | 80.96 | 42.71 | 46.56 |
| 4000 | 7.96 | 81.01 | 42.33 | 45.93 |

Worse, **the sign is not predictable**: on EMDB, PVE gets *worse* as the fit
converges (92.45 -> 93.84 -> 94.07) because at 400 iterations the fit is
accidentally regularised toward the SMPL mean shape. A number you cannot
defend without saying how long Adam ran is not a protocol. Use >= 1500, or
skip the fit entirely (next section). The EMDB table two sections down was
produced at 400 and is inflated.

### The conversion, settled: one surface, one operator on both sides

`barycentric_transfer` already returns the prediction **in SMPL topology**,
vertex i <-> SMPL vertex i. So the SMPL parameter fit is not needed to reach
either PVE or a regressed joint set — it exists only to recover the 24
*kinematic* joints, which come off the kinematic chain. Measured on identical
predictions, a converged fit converges *toward* the fit-free answer, so all it
adds in those cases is a hyperparameter with a 3-4 mm blast radius.

The protocol, in order of what the ground truth actually *is*:

| metric | GT is | conversion |
|---|---|---|
| **PVE / PA-PVE** | vertices | none — compare directly |
| **3DPW J14** | `J_regressor_h36m @ GT verts` | same regressor on **both** sides, so its bias cancels exactly |
| **EMDB SMPL-24** | kinematic joints | a vertex regressor approximates these; costs ~1.1 mm (see below) |

Nothing here is fitted by us: only the rig, Meta's published barycentric map,
and the benchmark's own regressor. There is no optimiser and no free parameter,
so the failure above cannot recur.

**The one approximation, quantified.** EMDB's 24 kinematic joints are not a
linear function of posed vertices. Regressing them from the surface carries, on
EMDB GT bodies, **10.15 mm mid-hip / 7.02 mm Procrustes** of bias. Errors add in
quadrature, so at a ~50 mm error level that costs **~1.1 mm** of PA-MPJPE:
sqrt(50.73^2 + 7.02^2) = 51.2 against 51.82 measured. Quote it; do not
reach for the fit to remove it unless you run >= 1500 iterations.

**Conversion floor, measured on EMDB (2026-09-11).** `make_mhr_gt.py`'s cached
MHR fit of EMDB's own GT, pushed through this exact forward path — the oracle
row, the best any MHR-rigged model could score:

| | MPJPE | PA-MPJPE | PVE |
|---|---|---|---|
| oracle (conversion floor) | 18.64 | **12.36** | 19.92 |

The oracle's own surface fit leaves 9.40 mm, so this is an **upper bound** on
conversion cost, not a tight estimate. At a 51.8 mm result, a 12.4 mm floor
contributes ~1.5 mm in quadrature: **conversion is not what separates this
model from SOTA.** Do not subtract it — errors do not add linearly.

### The EMDB person-box convention: quote `--bbox annotated` for published tables

Not because it scores better — it scores **8-12 mm worse**, and it changes the
ranking. Measured over all 24,103 frames, PA-MPJPE penalty for switching from
projected-GT-joint boxes to EMDB's own:

| run | gt-joints | annotated | penalty |
|---|---|---|---|
| b3_s1 | 53.16 | 60.15 | +6.99 |
| g6bv_s | 52.01 | 60.33 | +8.32 |
| g6b_s | 54.78 | 63.30 | +8.52 |
| g6v_s (v2) | 52.50 | 64.36 | **+11.86** |
| g6vv_s (v2) | 51.81 | 63.81 | **+12.00** |

Published "Oracle: annotated bounding boxes" rows use EMDB's own boxes, so that
is the only row that may sit beside them. Picking the cheaper convention while
comparing against papers that used theirs is the same error as picking a
flattering joint conversion.

**`--preset v2` is markedly more box-sensitive** (~12 mm against ~8.4 mm), which
reverses the ranking: g6vv_s is best under projected boxes and *fourth* under
EMDB's. v2 sets `cliff_follows_aug`, so the model leans harder on the CLIFF
vector describing its crop; EMDB's boxes are 17.5 % larger and 5.7 % off-centre
against the padded boxes it trained on. Truncation augmentation does not cover
this mismatch. Check both conventions before claiming a v2 win.

### EMDB-1, SMPL space via mesh conversion (2026-09-08)

All 17 sequences, all 24,103 frames, projected-GT-joint boxes, `--fit-iters 400`
— **inflated, see the fit-iters section above; kept for provenance only.**
Report in `benchmark/results/emdb1_smplfit_gt-joints.json`.

| run | epoch | MPJPE | **PA-MPJPE** | PVE | PA-PVE | mean fit residual |
|---|---|---|---|---|---|---|
| b3_s1 (baseline) | 137 | 77.38 | **52.00** | 92.25 | 62.57 | 14.61 |
| bno_s0 (v2, `--cliff-focal --bound-scales`) | 85 | 79.55 | **54.23** | 94.44 | 64.82 | 14.72 |
| *conversion floor (round trip)* | | *11.44* | *10.66* | *13.61* | | |

**The two runs are not comparable to each other.** bno_s0 is at epoch 85 against
b3_s1's 137, so this is a maturity difference as much as a configuration one —
and with one seed per arm it would prove nothing even at matched epochs (seed
spread in this repo is 34-49 mm on the 70-keypoint metric).

Against the published EMDB (24) table (Fast SAM 3D Body Table 1), b3_s1 lands
next to **WHAM** — ahead of it on MPJPE (77.4 vs 79.7) and PVE (92.3 vs 94.4),
just behind on PA-MPJPE (52.0 vs 50.4) — and well ahead of SMPLer-X-H (64.5 PA)
and HMR2.0b (79.2 PA). WHAM is a *video* method; this is single-frame.

**The mesh route beats the linear adapter on 16 of 17 sequences**, mean
−1.73 mm, with per-sequence Pearson r = 0.989 and Spearman r = 0.961 between
the two. Two conversions sharing no machinery agreeing that closely on the
ranking is the evidence that neither distorts the result. The gain is largest
where the pose is unusual — `outdoor_sitting` −4.9 mm, `outdoor_climb`
−3.8 mm — which is where a fixed linear joint map is furthest from its fitting
distribution.

Note the headline gain is only ~1.8 mm even though the conversion floor drops
18.8 -> 10.7 mm. That is arithmetic, not disappointment: floors do not subtract
linearly. Treating model and conversion error as independent, sqrt(50^2 +
18.8^2) = 53.4 against sqrt(50^2 + 10.7^2) = 51.1, so ~2.3 mm predicted against
1.8 mm observed. The real win is that **PVE exists at all**.

### Generation 6 — the `preset x --w-verts` 2x2 (2026-09-11)

Scored through the fit-free conversion above, all frames, `best_student_model_v3.pth`
(selected on 3DPW validation). All four are seed 0, `--losses rebalanced
--cliff-focal --bound-scales --anomaly-safe-fallback --crop-centre-fix`.

**3DPW test** — 35,463 person-frames, J14/H36M, published-protocol GT:

| run | preset | w_verts | ep | MPJPE | PA-MPJPE | PVE | PA-PVE | J14+adapter PA |
|---|---|---|---|---|---|---|---|---|
| g6vv_s | v2 | 0.35 | 239 | 67.71 | **42.69** | 81.01 | **57.68** | **41.77** |
| g6v_s | v2 | 0 | 225 | **67.27** | 43.16 | **80.57** | 58.07 | 42.21 |
| g6bv_s | baseline | 0.35 | 247 | 70.31 | 43.30 | 83.51 | 58.51 | 42.55 |
| g6b_s | baseline | 0 | 102 | 73.98 | 45.27 | 87.92 | 61.07 | 44.44 |
| b3_s1 (gen5) | baseline | 0 | 137 | 69.10 | 43.03 | 82.81 | 58.31 | 41.81 |
| bno_s0 (gen5) | v2 | 0 | 85 | 69.84 | 44.80 | 83.39 | 60.04 | 43.78 |

**EMDB-1** — 24,103 frames, SMPL-24 kinematic, projected-GT-joint boxes (for the
`--bbox annotated` row that belongs beside published tables, see above):

| run | preset | w_verts | ep | MPJPE | PA-MPJPE | PVE | PA-PVE |
|---|---|---|---|---|---|---|---|
| g6vv_s | v2 | 0.35 | 239 | **78.71** | **51.81** | **94.71** | **61.67** |
| g6bv_s | baseline | 0.35 | 247 | 79.20 | 52.01 | 95.01 | 61.69 |
| g6v_s | v2 | 0 | 225 | 79.44 | 52.50 | 95.29 | 62.68 |
| g6b_s | baseline | 0 | 102 | 83.31 | 54.78 | 99.82 | 64.30 |
| b3_s1 (gen5) | baseline | 0 | 137 | 78.01 | 53.16 | 94.17 | 63.66 |
| bno_s0 (gen5) | v2 | 0 | 85 | 80.55 | 55.36 | 96.82 | 65.88 |

**`--preset v2` is the win, and it lands where the mechanism says it should.**
At nearly matched epochs (g6bv_s 247 vs g6vv_s 239), 3DPW MPJPE drops **2.60 mm**
and PVE **2.50 mm** while PA-MPJPE moves only 0.61 mm. That is the signature of
v2's absolute-pose supervision — `w_reproj` 0.01 -> 0.5 and the unsquared
Euclidean `loss_cam` fix *where the body sits*, and Procrustes discards exactly
that. Read it next to the box-sensitivity finding above before shipping it.

**`--w-verts 0.35` does not deliver what it was added for.** At matched epochs it
moves PVE by -0.58 mm on EMDB and **+0.44 mm on 3DPW** — the wrong sign, net
~zero. PVE is the only metric that can see the 45 `shape_params`, so this was
the row where the surface term should have shown up. It did not, at one seed.

**g6b_s is not a fair member of the 2x2.** Its best checkpoint is epoch 102
against the others' 225-247 and its run died at 171 of a 300-epoch
`OneCycleLR`, so it never annealed. Its ~2 mm deficit is maturity.

**One seed per arm.** Every gap above except the v2 MPJPE/PVE effect is under
1 mm, well inside seed spread. Only the 2.5 mm v2 result is safe to act on.

### What these metrics cannot see

Both headline metrics are alignment-invariant, and that hides exactly the
failure a demo shows first. PA-MPJPE removes translation, rotation **and**
scale; root-relative MPJPE removes translation. So camera-translation drift,
systematic body-size error and frame-to-frame jitter are all invisible here —
a change can be worth shipping on stability and move PA-MPJPE by 0.1 mm.

3DPW is video, so measure stability directly on consecutive frames:

- **rigid bone lengths** within a person track. A femur cannot change length,
  so its variance is pure prediction noise. GT floor: 1.06%.
- **`cam_trans` acceleration**, `|t[i+1] - 2t[i] + t[i-1]|`. GT floor: 7.0
  mm/frame².

Measured on 3DPW test (35,463 person-frames, J14 + teacher adapter):

| checkpoint | PA | MPJPE | bone-len CV | cam accel | (z only) |
|---|---|---|---|---|---|
| `distill_mhr_only_ckpt90` (old corpus, *contaminated*) | 41.14 | 68.85 | 1.65% | 148.1 | 144.3 |
| b2_s0 | 44.84 | 72.69 | 1.96% | 234.4 | 229.9 |
| b3_s1 | 41.83 | 69.03 | 1.85% | 219.6 | 215.9 |
| v3_s1 | 43.23 | 70.59 | 1.79% | 224.6 | 222.1 |
| ground truth | — | — | 1.06% | 7.0 | |

Every model is 21-33x jitterier than ground truth, and the rebuilt corpus made
it ~50% worse — traced to the focal spread and addressed by `--cliff-focal`
(see `instanthmr_distill_train/README.md`). The legacy row is optimistic on
both axes: it trained on all 60 3DPW sequences.

## Protocols, stated precisely

Write these into the paper — they are the details a reviewer will ask about.

**3DPW test.** All 24 test sequences, every frame where `campose_valid` is set
and the person is at least 60 % inside the image. Person boxes come from the
projected GT joints padded by `--bbox-scale` (default 1.2), so no detector is in
the loop. MPJPE is mid-hip-aligned; PA-MPJPE is Procrustes (scale + rotation +
translation). Predictions use `joints_3d_local`, the rig-local pose, so the
camera translation never enters the metric.

**EMDB-1.** All 17 sequences of the `emdb1` split, every frame where
`good_frames_mask` is set and the subject is at least 60 % inside the image.
Metrics are over the 24 SMPL kinematic joints, aligned at the midpoint of the
two hips (SMPL indices 1 and 2, not the `pelvis` joint), reached through the
rig mesh and Meta's barycentric map, then a vertex regressor (see "The
conversion, settled" — the ~1.1 mm kinematic approximation is quoted with the
result). PA-MPJPE is Procrustes with uniform scale. PVE is vertex-to-vertex
against the GT SMPL mesh, mid-hip aligned. Person boxes: say which of the two
conventions you used, and use `--bbox annotated` for any published comparison.

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

  **Fit it from the dataset annotations, not from the student.** `--fit-adapter` uses one
  checkpoint's own predictions as the input side, so the map can absorb that
  checkpoint's systematic error along with the rig offset — a different adapter
  per checkpoint, and a fair reader will call it tuning.
  `benchmark/fit_adapter_mhr.py` instead pairs the **released dataset's** MHR
  labels in `data/sam3d_gt_3dpw` with the H36M GT on the same 3DPW train
  frames, giving one adapter for every checkpoint that never saw a student.
  (Those are `facebook/sam-3d-body-dataset` annotations built by
  `datasets_pipeline/build_split.py`; only `data/sam3d_distill_mix` is teacher
  inference. Earlier revisions of this file called them teacher labels — wrong,
  and it changed what the residual below means.)

  ```bash
  python benchmark/fit_adapter_mhr.py --sequence-dir /path/to/3DPW/sequenceFiles \
      --out benchmark/results/adapter_j14_h36m_teacher.npz
  ```

  That it is a rig conversion rather than an error sponge is visible two ways.
  The annotations' own J14 error against the GT drops from **44.90 mm to 14.83 mm**
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

**3DPW test is clean for every checkpoint trained on Jean Zay** — that corpus
is coco/mpii/aic/sa1b/harmony4d plus `3dpw_train`, so the 24 test sequences are
unseen for `b3_s1`, `v3_s1` and everything after them. The paragraph below
applies to the **local** corpus and to `distill_mhr_only_ckpt90`, which was
trained from it.

**The local `data/` corpus is contaminated.** `data/sam3d_gt_3dpw` holds only train sequences,
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
  make_emdb_gt.py    SMPL forward -> <EMDB root>/gt_smpl24/ (+ a kp2d check)
  fit_adapter_mhr.py MHR70 -> J14 / SMPL24 adapter from TEACHER labels
  eval_3dpw.py       3D from an ONNX graph: MPJPE, PA-MPJPE, PCK3D, AUC
  eval_3dpw_ckpt.py  the same, for a .pth straight out of runs/
  eval_emdb_ckpt.py  EMDB-1 for a .pth; reuses the 3DPW crop and metrics
  eval_smpl_fit.py   EMDB-1 in SMPL space via mesh conversion: +PVE, lower floor
  mhr_smpl.py        the MHR rig, differentiable SMPL, and both fit directions
  eval_coco.py       2D: OKS AP/AR, mean OKS, PCK, NME, per-keypoint breakdown
  benchlib/
    joints.py        MHR70 <-> COCO17 / SMPL24 / H36M17 maps, J14 & J12 sets
    metrics.py       Procrustes alignment and the metric definitions
    threedpw.py      3DPW pickle loading, the GT switch, projection, boxes
    emdb.py          EMDB-1 loading; extrinsics sense and the 24-joint choice
    runner.py        batched ONNX inference with overlapped image decoding
```
