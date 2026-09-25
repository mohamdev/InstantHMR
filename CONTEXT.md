# InstantHMR

A real-time, single-image human mesh recovery model that predicts MHR rig parameters and is built to run on phones. This glossary fixes the words used about its models, training runs, evaluation and deployment.

## Language

### Models and runs

**Reference model**:
The model currently shipped to devices and used as the comparison point for every new experiment (g8h as of 2026-09-24).
_Avoid_: baseline, new baseline, the current model

**Generation**:
A numbered batch of training runs that differ from the previous batch along one or two stated axes (e.g. gen8 = continuous regression head + exact landmarks).
_Avoid_: version, round

**Arm**:
One configuration inside a generation, trained under one or more seeds (e.g. g8h = gen8 with the HGNetV2-B4 backbone).
_Avoid_: variant, experiment

**Model family**:
A set of models trained with one recipe at several backbone sizes, so that each deployment tier gets its own point on one accuracy-vs-latency curve.
_Avoid_: model zoo, variants

### Training

**Annotation target**:
The human MHR fit released for each person in `facebook/sam-3d-body-dataset`, which every current run is trained to match.
_Avoid_: reference target, teacher label, pseudo-label, GT (unqualified)

**Baseline preset**:
The original input pipeline selected by `--preset baseline`; only ever the name of a preset, never of a model.
_Avoid_: baseline (on its own), default

**v2 preset**:
The alternative input pipeline selected by `--preset v2`, whose CLIFF vector follows the augmented crop.

**v3 preset**:
The input pipeline that samples each training crop from a context crop over a wide range of framings, loose, off-centre and truncated, always surrounded by real scene.

**Context crop**:
A stored training crop covering twice the person's tight box at high resolution, from which the network's input window is sampled.
_Avoid_: big crop, raw crop, original (originals are the full source frames)

### Evaluation

**Published protocol**:
The evaluation setting whose numbers may sit beside other papers' rows: EMDB-1 with EMDB's own person boxes scored on SMPL-24 joints, and 3DPW test scored on J14 with the adapter.
_Avoid_: benchmark (unqualified), our protocol

### Deployment

**Deployment tier**:
A class of device runtime the product must support, each with its own latency target: Snapdragon NPU, Apple Core ML, and CPU (the tier that covers every other device).
_Avoid_: platform, backend, target
