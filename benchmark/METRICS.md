# What we measure, and in what space

InstantHMR predicts **MHR**. Every published 3DPW and EMDB number is defined on
**SMPL**. This document explains how we get from one to the other, exactly
which space each metric lives in, and why the reported numbers are comparable
to a published table.

Read this before quoting a number, and before adding a new one.

---

## 1. The problem in one paragraph

The model outputs 204 MHR model parameters plus 45 identity coefficients. Those
drive a 127-joint skeleton and an 18,439-vertex mesh. The benchmarks want the
**24 SMPL joints** (MPJPE, PA-MPJPE) and the **6,890 SMPL vertices** (PVE).
These are different rigs: different joint definitions, different topology, no
shared vertex. Nothing can be compared until one side is moved into the other's
space. **Which side you move is the whole ballgame** — see §5.

---

## 2. How MHR becomes SMPL

Three stages. Stage 2 is the one that makes it work.

```
  student  ──▶  MHR params        (204 + 45)
                    │
                    │  ① rig forward: identity blendshapes,
                    │     pose correctives, linear blend skinning
                    ▼
                MHR mesh          (18,439 verts)
                    │
                    │  ② official barycentric surface map
                    │     SMPL vertex i = w₁·v_a + w₂·v_b + w₃·v_c
                    │     on MHR triangle t(i)
                    ▼
             target vertices      (6,890, SMPL topology)
                    │
                    │  ③ labelled fit for θ (72), β (10), t (3)
                    │     global orientation solved in closed form
                    ▼
                SMPL params  ──▶  24 joints + 6,890 vertices
```

### ① The rig forward

`MHRRig.vertices` calls the TorchScript rig's own top-level forward, so the
mesh includes identity blendshapes and pose correctives — the same mesh the
dataset annotations describe. Not the trainer's skeleton-only path, and not its
595-vertex training subset.

Note this stage consumes the **45 `shape_params`**. Those move the 127-joint
skeleton by exactly `0.00e+00 cm`, so every joint metric in this repo is blind
to them. They *do* move the mesh, which is why PVE is the first number here
that responds to body shape at all.

### ② The surface map — where the correspondence comes from

`mhr2smpl_mapping.npz` from
[facebookresearch/MHR](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion)
(`tools/mhr_smpl_conversion/assets`, Apache-2.0). For each of the 6,890 SMPL
vertices it stores **one MHR triangle id and three barycentric weights**. So
each SMPL vertex is a fixed point on the MHR surface, and pushing a posed MHR
mesh through it resamples that same surface in SMPL topology.

It is a topology constant — indices and weights, no coordinate frame, no units
— so it applies in whatever frame you hand it.

**Why not derive it ourselves?** We tried, three ways, and all three failed:

| attempt | residual | why it fails |
|---|---|---|
| nearest MHR vertex + mean offset in camera space | 60 mm | the offset rotates with the person; it is not a constant |
| affine combination of the K nearest MHR vertices | 65 mm, weights to ±405 | a local neighbourhood on a smooth mesh spans a tangent plane, leaving the normal direction unresolved |
| offset in a local surface frame (normal + tangents) | ~10 mm std | works in principle, but sensitive to under-converged fits upstream |

**Verify the map against the rig before trusting it** (`mhr_smpl.py` does):
triangle ids top out at 36,871 against the rig's 36,874 faces, barycentric rows
sum to 1, and the reconstructed SMPL-topology mesh has a mean nearest-vertex
spacing of **13.88 mm** against real SMPL's **14.37 mm**. A map built for a
different mesh would scatter neighbouring vertices onto unrelated triangles and
that number would blow up.

### ③ The fit

Now every SMPL vertex has a known target, so this is a plain labelled
regression: minimise `‖SMPL(θ,β,t) − target‖²` plus an edge term (edge
*vectors*, which constrain local shape independently of position, down-weighted
after 30 % of the schedule so absolute positions drive the endgame).

Two details that are not optional:

- **Correspondence must be known.** Fitting the surface by ICP instead — letting
  nearest-neighbour search discover the pairing — has a local minimum that its
  own residual cannot see: the mesh settles overlapping the target while limbs
  are matched to the **wrong limbs**. Measured: 12 mm surface residual, 76 mm
  PA-MPJPE. The surface looks fine and the pose is wrong.
- **Global orientation is solved in closed form**, by Procrustes between SMPL's
  rest mesh and the target. Known correspondence makes that a three-line SVD.
  Without it the fit starts at identity and has to walk to an arbitrary
  camera-space orientation through axis-angle: it stalls at **40–51 mm** of
  vertex residual instead of **13 mm**.

Typical fit residual on real predictions: **14–16 mm** surface distance.

---

## 3. Which space each metric is computed in

**Everything is computed in SMPL space.** Predictions are converted; the ground
truth is never touched.

| metric | space | prediction side | GT side | alignment |
|---|---|---|---|---|
| **MPJPE** (SMPL24) | SMPL | 24 kinematic joints of the fitted SMPL | 24 kinematic joints of GT SMPL | mid-hip (joints 1 & 2) |
| **PA-MPJPE** (SMPL24) | SMPL | same | same | Procrustes (rotation + translation + **scale**) |
| **PVE** | SMPL | 6,890 fitted vertices | 6,890 GT vertices | mid-hip |
| **PA-PVE** | SMPL | same | same | Procrustes |
| **J14 (H36M)** | SMPL | `J_regressor_h36m` @ fitted vertices → 14 | `J_regressor_h36m` @ GT vertices → 14 | mid-hip / Procrustes |

Nothing is computed in MHR space. See §5.

### How the 24 joints are obtained — there is no regression

This is the part people expect to be complicated, and it is not. Once SMPL is
fitted, **both sides come out of the same SMPL model by the same code path**:
`SMPLModel.forward` returns the 24 kinematic-tree joints carried by the global
transforms, and it is called on the fitted parameters for the prediction and on
the GT parameters for the ground truth. No adapter, no learned map, nothing
fitted per checkpoint.

We use the **kinematic** joints, not `J_regressor @ posed vertices`. That is not
a shortcut — it is EMDB's own definition. EMDB ships a `kp2d` field, and it
reprojects the kinematic joints to **0.00 px** and the regressed ones to
**1.04 px**; in 3D the two sit 3.8 mm apart on average. 3DPW's `jointPositions`
field is the same kinematic set.

### How the 14 joints are obtained — and why an adapter is not a compromise

3DPW's published protocol is 14 joints in the Human3.6M convention. **There is
no route from MHR to that convention without a conversion, and no such route
exists for SMPL-native methods either.** This is the single most important
thing to be clear about, because it is easy to state backwards.

`J_regressor_h36m.npy` — the file NLF, CameraHMR, SPIN, HMR2.0 and everyone else
uses — **is itself a fitted linear regressor**, 17 x 6890, mapping SMPL vertices
to H36M joints. Those methods do not skip a conversion; they use one that ships
with the benchmark ecosystem, because their meshes are already SMPL. We predict
MHR, so we need one more step. That is a property of the rig, not a weakness in
the protocol, and a linear map from our native output to the benchmark's joint
convention is the same class of object theirs is.

So we report two routes, and **neither is "adapter-free"** — that phrase should
never appear next to either number:

| row | path | what it needs |
|---|---|---|
| **J14 + joint adapter** (primary) | MHR70 keypoints → fixed 14x70 linear map → J14 | the adapter, fitted once on 3DPW *train* from the dataset's own MHR annotations, never from a student, rank-truncated (`RCOND`) |
| **J14 via mesh fit** | MHR mesh → barycentric map → fitted SMPL → `J_regressor_h36m` → J14 | Meta's surface map + the standard H36M regressor |

Both convert MHR into the H36M convention. They differ only in *where* the
conversion happens — joint space or mesh space.

Evidence the joint adapter is a rig conversion and not an error sponge, which
is the only real objection to it:

- it is fitted from **dataset annotations, never from a student**, so it is one
  fixed map for every checkpoint and cannot absorb a particular model's error;
- it reads anatomically — `right_hip = +1.16 right_hip - 0.32 left_hip`
  (widening SMPL's 120 mm hips toward H36M's 292 mm), `right_knee = +0.84
  right_knee`, nearly identity because the knee is the same point in both;
- the **SMPL24 version transfers across datasets**: fitted on 3DPW train,
  applied unchanged to EMDB, where it was never fitted. A map that had
  overfitted 3DPW would not survive that.

The mesh route exists because it is the only one that also yields **PVE** and
the SMPL24 joint set, not because the adapter is suspect. Expect the two J14
numbers to differ by a few mm; report the adapter row as the headline and the
mesh row alongside it, each labelled by its path.

## 4. What PVE means (checked against the papers)

**PVE is pelvis-aligned, not Procrustes-aligned.** CameraHMR §4.3:

> "…MPJPE, PA-MPJPE, and PVE (Per Vertex Error), which measures the Euclidean
> distance (in mm) between predicted and actual 3D vertices and joints **after
> aligning the pelvis**."

NLF's appendix confirms the naming: *"MVE is analogous to MPJPE but for mesh
vertices instead of joints, similarly **P-MVE** is the Procrustes-aligned
version of it."*

So the column to put next to a published PVE is our **`PVE_mm`**, not
`PA_PVE_mm`. We report PA-PVE too, because it isolates shape from pose, but it
belongs in a different column and must be labelled `P-MVE` / `PA-PVE` if quoted.

The reference point is the **midpoint of the two hips**, not SMPL joint 0. NLF:
*"on 3DPW and EMDB the reference point is the midpoint between the two hip
joints."*

---

## 5. The direction rule — read this before inventing a new metric

The conversion runs **prediction → SMPL**. Never the other way for a reported
number.

Solving an MHR ground truth for EMDB and scoring predictions on MHR joints
would remove the conversion cost entirely, and it is *not* a comparable number:

1. **MPJPE is not rig-independent.** The published column is defined on SMPL's
   joints and SMPL's vertices. A number on MHR's 70 keypoints is a different
   quantity, not a tighter measurement of the same one.
2. **40 of MHR's 70 keypoints are finger joints millimetres apart.** Averaging
   error over them drags the mean down regardless of body accuracy.
3. **The reference would be something we generated**, with an optimiser told to
   look like an MHR model. Whatever MHR cannot represent silently leaves the
   error term.

The MHR ground truth is still worth having — `make_mhr_gt.py` builds and caches
it — but for the **oracle row**: fit MHR to the GT surface, convert it back
through the forward path, and score it. That is the best any MHR-rigged model
could achieve here, and it decomposes the error into rig cost and model error.
It is an upper bound, labelled as one.

---

## 6. Conversion floors — quote these next to results

Measured as a round trip through the rig (GT SMPL → fit MHR to that surface →
convert back → score against the untouched GT), cold-started at every stage:

| route | MPJPE floor | PA-MPJPE floor | PVE floor |
|---|---|---|---|
| linear MHR70 → SMPL24 joint adapter | 25.11 mm | 18.77 mm | not possible |
| **mesh conversion** | **11.44 mm** | **10.66 mm** | **13.61 mm** |

A floor does not subtract from a result linearly. Treating model error and
conversion error as independent, dropping the floor from 18.8 to 10.7 mm at a
~50 mm model error predicts √(50²+18.8²)=53.4 → √(50²+10.7²)=51.1, about 2.3 mm
— which is what is observed. The mesh route's real win is that **PVE exists**.

---

## 7. Running it

One command per dataset. Both write JSON to `benchmark/results/`.

```bash
# EMDB-1 in SMPL space (24,103 frames, ~25 min on an RTX 4070)
python benchmark/eval_smpl_fit.py --dataset emdb \
    --ckpt <run>/best_student_model_v3.pth \
    --emdb-root /path/to/EMDB \
    --out benchmark/results/emdb1_smplfit.json

# 3DPW test in SMPL space (35,463 person-frames, ~30 min)
python benchmark/eval_smpl_fit.py --dataset 3dpw --split test \
    --ckpt <run>/best_student_model_v3.pth \
    --sequence-dir /path/to/3DPW/sequenceFiles \
    --image-root   /path/to/3DPW/imageFiles \
    --out benchmark/results/3dpw_test_smplfit.json
```

`--ckpt` takes several checkpoints and evaluates them in one pass; the config
(`--bound-scales`, `--cliff-focal`) is restored **per checkpoint** from the
weights and `run_config.json`, so checkpoints with different flags can share an
invocation safely. Results are written after *each* checkpoint.

The legacy joint-adapter rows come from `eval_emdb_ckpt.py` and
`eval_3dpw_ckpt.py`; `eval_all.py` runs everything and prints the combined
table.

### One-time setup

```bash
# Meta's surface maps -> benchmark/data/mhr_smpl_conversion/
#   {mhr2smpl,smpl2mhr}_mapping.npz  from facebookresearch/MHR (Apache-2.0)

# EMDB 24-joint GT cache (7 MB); --check must print ~0.00 px against kp2d
python benchmark/make_emdb_gt.py --emdb-root /path/to/EMDB \
    --smpl-dir benchmark/data/smpl --check

# 3DPW H36M GT cache, for the J14 adapter rows
python benchmark/make_3dpw_gt.py --sequence-dir /path/to/3DPW/sequenceFiles \
    --smpl-dir benchmark/data/smpl --check

# MHR ground truth, cached so it is never recomputed (~25 MB each)
python benchmark/make_mhr_gt.py --dataset emdb --emdb-root /path/to/EMDB
python benchmark/make_mhr_gt.py --dataset 3dpw --split test \
    --sequence-dir /path/to/3DPW/sequenceFiles
```
