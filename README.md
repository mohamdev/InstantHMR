# InstantHMR

![InstantHMR demo](models/instanthmr.gif)

A lightweight, ONNX-exportable distillation of
[`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)
for **3D human pose estimation and mesh recovery**: a RepViT-M1.5 backbone + a 9-token
cross-attention decoder + CLIFF camera conditioning. The model released by [`NaturalPad`](https://www.naturalpad.fr/) was trained to mimic the
SAM3D teacher's per-person 70-keypoint outputs from a single 224×224 crop.

InstantHMR ships as a single `.onnx` file. The demo pipeline pairs it with
**RF-DETR** for person detection — both stages are timed independently so
you always know where the latency is going. Optionally, the demo decodes
the raw `mhr_params` / `shape_params` outputs through Meta's
**[MHR body model](https://github.com/facebookresearch/MHR)** to render a
full dense body mesh in the Rerun viewer.

- **Inputs:** `image (N, 3, 224, 224)`, `cliff_cond (N, 3)`.
- **Outputs:** `mhr_params (204)`, `shape_params (45)`, `cam_trans (3)`,
  `joints_2d (70, 2)` (normalised crop coords),
  `joints_3d (70, 3)` (body-centred metres, Y-down).
- **Speed (InstantHMR ONNX alone):** ~5 ms / frame (~200 FPS) on a single
  RTX 4070 with the fp16 ONNX (CUDA EP); CPU works too (~25 FPS, depending
  on hardware). On Apple Silicon, pass `--device coreml` to use
  `CoreMLExecutionProvider`.
- **Speed (full demo, end-to-end):** the demo also runs **RF-DETR** every
  frame, and on most hardware the detector — not InstantHMR — is the
  bottleneck. Use `--detector-stride N` to run RF-DETR only every Nth
  frame and reuse the previous bbox in between (see *Performance tuning*
  below).
- **Note for RTX 50-series (Blackwell, sm_120):** stock Torch with
  CUDA 12.4 wheels does not ship sm_120 kernels and falls back to slow
  paths. Use `python install.py` which automatically pulls Torch
  cu128 (>=2.7) on Blackwell GPUs.

## Install

```bash
git clone <this-repo> instanthmr
cd instanthmr
```

Choose the path that matches your needs:

| Goal | Python | Command |
|------|--------|---------|
| Pose estimation only (joints + skeleton) | 3.11 or 3.12 | `python install.py` |
| + MHR body mesh rendering | **3.12** | `python install.py` then see [MHR body mesh](#mhr-body-mesh-rendering) |

```bash
# Pose estimation only (Python 3.11+):
conda create -n instanthmr python=3.11 -y && conda activate instanthmr

# Picks the right torch + onnxruntime wheels for your machine:
#   - Linux + NVIDIA GPU → cu128 (Blackwell/RTX 50) or cu124 (Ada/Ampere/Hopper)
#                          + onnxruntime-gpu + bundled CUDA / cuDNN runtime libs
#   - macOS              → stock torch + onnxruntime (CoreML EP included)
#   - Linux without GPU  → CPU torch + onnxruntime
python install.py
```

`python install.py --dry-run` prints the pip commands without running them.
`python install.py --force-cpu` skips GPU detection on Linux.
If you'd rather manage wheels yourself, `pip install -r requirements.txt`
still works — it installs the CPU fallback path.

The first run downloads the RF-DETR (medium) checkpoint automatically.
The InstantHMR ONNX weights are on HuggingFace — see
[Model weights](#model-weights) below.

## Run the demo

```bash
# Single image
python demo.py --image path/to/photo.jpg

# Video file
python demo.py --video path/to/clip.mp4

# Live webcam (index 0)
python demo.py --camera 0
```

`python demo.py --help` lists every flag (model path, device, detector
variant, confidence, max persons, frame skip, `.rrd` recording, …).

The demo opens a Rerun viewer with the source image + 2D skeleton, a live
RF-DETR / InstantHMR / total-latency plot, and the 3D scene with the
predicted camera frustum.

## MHR body mesh rendering

The demo can render a full dense body mesh for each detected person by
running a forward pass through Meta's **MHR (Momentum Human Rig)** body
model, using the `mhr_params (204,)` and `shape_params (45,)` outputs that
InstantHMR already produces.  Render time is displayed separately in the
Rerun latency plot and in the console.

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.12** | `pymomentum` has no Python 3.11 wheels; recreate your conda env if needed |
| NVIDIA GPU / Apple Silicon | CPU works but is much slower |
| CUDA toolkit | Must match your PyTorch wheel (Linux/Windows) |

### Step 1 — Install in one pip call

Always pass **both** requirement files to a **single** `pip install` call.
Running them separately causes pip to first choose a CUDA-13 torch for
`rfdetr`, then downgrade to CUDA-12 torch for `pymomentum`, leaving
`torchvision` broken.

```bash
conda create -n instanthmr python=3.12 -y && conda activate instanthmr
pip install -r requirements.txt -r requirements-mhr.txt
```

`requirements-mhr.txt` auto-selects the right package for your platform
(`pymomentum-gpu` on Linux/Windows, `pymomentum-cpu` on macOS).
CPU-only Linux users: see the comment at the top of `requirements-mhr.txt`.

> **⚠ Wrong `pymomentum` on PyPI**
> `pip install pymomentum` installs an **unrelated** legacy SMS library
> (pyMomentum v0.1.x by MomentumAS) — not the Meta geometry library.
> `requirements-mhr.txt` pulls in the correct package automatically.
> If you accidentally installed the wrong one first:
> ```bash
> pip uninstall pymomentum
> pip install -r requirements.txt -r requirements-mhr.txt
> ```

> **Torch/torchvision version mismatch?**
> ```bash
> python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
> # If they differ, wipe both and reinstall together:
> pip uninstall -y torch torchvision
> pip install -r requirements.txt -r requirements-mhr.txt
> ```

### Step 2 — Download the body-model assets

```bash
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
unzip assets.zip -d models/mhr_assets
```

This unpacks the body template, blend shapes, skinning weights, and pose
correctives into `models/mhr_assets/`.

### Step 3 — Run the demo

```bash
# Single image — LOD 0 (73 639 vertices, highest quality)
python demo.py --image photo.jpg --mhr-assets models/mhr_assets --mhr-lod 0

# Video file — LOD 1 (18 439 vertices, good quality)
python demo.py --video clip.mp4 --mhr-assets models/mhr_assets --mhr-lod 1

# Live camera — LOD 3 (4 899 vertices, real-time default)
python demo.py --camera 0 --mhr-assets models/mhr_assets
```

### LOD reference

| `--mhr-lod` | Vertices | Faces | Recommended use |
|:-----------:|----------|-------|-----------------|
| 0 | 73 639 | — | offline images, max quality |
| 1 | 18 439 | — | video, powerful GPU |
| 2 | 10 661 | — | video, mid-range GPU |
| **3** | **4 899** | — | **live camera (default)** |
| 4 | 2 461 | — | weak GPU / many persons |
| 5 | 971 | — | borderline real-time |
| 6 | 595 | — | debugging / stress test |

### How it works

```
InstantHMR ONNX
  └─ mhr_params (204,)  ──┐
  └─ shape_params (45,) ──┤─→  MHR.forward()  →  vertices (V, 3)
  └─ cam_trans (3,)     ──┘         + cam_trans  →  camera-space mesh
```

The 204-dim `mhr_params` are MHR **model parameters**, not a 6-D rotation
encoding. Read off `character_torch.parameter_transform.parameter_names`, the
layout is:

| slice | contents |
|---|---|
| `0:3` | `root_tx/ty/tz` — always 0 in the corpus; 1 unit = 10 cm, applied at joint 1 |
| `3:6` | `root_rx/ry/rz` — extrinsic-XYZ Euler, driving joint-parameter slots 10/11/12 and nothing else |
| `6:136` | 130 local joint angles (`spine0_rx_flexible`, `l_knee_ry`, …) |
| `136:204` | 68 bone-scale parameters |

Because the root rotation is exclusive to slots 10/11/12, `6:136` is exactly
invariant to a camera roll — which is what `pose_split` in the trainer relies
on. `shape_params (45,)` encode identity blend shapes (20 body + 20 head + 5
hand components) and drive the **mesh only**: a ±2σ probe moves the 127-joint
skeleton by `0.00e+00 cm`. Facial expression is set to neutral (zeros) since
InstantHMR does not regress it.

## Performance tuning

On every machine we've measured, RF-DETR (the detector) costs ~5–10× more
per frame than InstantHMR. The two flags below target that bottleneck.

| Flag | Effect |
| --- | --- |
| `--detector-stride N` | Run RF-DETR every Nth frame; reuse the previous bbox (slightly expanded) on the in-between frames. Stride 2–3 is the single biggest knob — typically 2–3× end-to-end FPS for slow movement, with negligible quality loss. |
| `--detector-variant nano` | Use the smallest RF-DETR; biggest win on CPU / Apple Silicon where the detector is doing all the work. |
| `--device coreml` | On macOS, route the InstantHMR ONNX through `CoreMLExecutionProvider`. |
| `--max-persons N` | Maximum number of persons processed per frame (default: **2**). See below. |
| `--no-batch-persons` | Disable batched multi-person HMR (one ONNX call per person). The default is batched. |

### `--max-persons` and why it matters

`--max-persons` does two things at once:

1. **Caps detections** — RF-DETR may find more people than you need. Setting `N`
   keeps only the `N` highest-confidence detections per frame.

2. **Fixes the ONNX batch size** — CUDA / ONNX Runtime compiles GPU kernels the
   first time a model runs at a new input shape. Because the batch dimension
   counts as part of the shape, switching from 1 person to 2 persons mid-demo
   would normally trigger a multi-second stall. The pipeline avoids this by
   **always** padding the input to exactly `N` crops (zero-filling unused slots)
   so the session always sees `batch = N`, regardless of how many people are
   actually in frame.

**Startup:** during `PosePipeline.__init__` the pipeline runs two warm-up passes
— `batch = N` and `batch = 1` (the single-person path) — so all CUDA kernels
are compiled before the first real frame. Startup cost is roughly `2 ×`
one-inference time and does **not** scale with `N`.

**Choosing `N`:**

| Scene | Recommended `--max-persons` |
|-------|----------------------------|
| Single person / portrait | 1 |
| Two people, couples, sparring | **2 (default)** |
| Small group, team sport | 4–6 |
| Crowd / many people visible | raise `N`; also consider `--detector-stride` |

Raising `N` does not slow down single-person frames at inference time (the
padded slots run on GPU in parallel), but it does proportionally increase
the memory footprint of the batch buffer and may slightly increase latency
when `N` is large and the GPU can no longer fully parallelise the extra crops.

Measured on RTX 4070 + torch 2.5 cu121 + ORT 1.25 (1080p video, 1 person, 150 frames after warm-up):

| `--detector-stride` | RF-DETR ms | HMR ms | total ms | FPS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 27.1 | 5.3 | 32.4 | **30.9** |
| 2 | 13.3 | 5.3 | 18.6 | **53.8** |
| 3 |  8.7 | 5.5 | 14.2 | **70.4** |
| 4 |  6.6 | 5.3 | 11.9 | **83.9** |

You can re-run the benchmark on your own hardware with:

```bash
python tools/bench.py --video vid1.mp4 --max-frames 150 --detector-stride 3
```

## Use it from Python

```python
from instanthmr import PosePipeline

pipeline = PosePipeline(
    onnx_path="models/instanthmr.onnx",
    device="cuda",
    detector_variant="medium",
)

result = pipeline.predict(image_rgb)
for r in result.persons:
    print(r.joints_3d_cam.shape)   # (70, 3)
    print(r.joints_2d.shape)       # (70, 2)
    print(r.mhr_params.shape)      # (204,) — MHR pose parameters
    print(r.shape_params.shape)    # (45,)  — MHR identity shape parameters
```

The 70 keypoints follow the MHR70 ordering — see
[`instanthmr/skeleton.py`](instanthmr/skeleton.py) for joint names and the
canonical skeleton edge list.

To run the MHR mesh decoder yourself:

```python
from instanthmr.mhr_renderer import MHRRenderer

renderer = MHRRenderer(assets_folder="models/mhr_assets", device="cuda", lod=3)

for r in result.persons:
    verts_local = renderer.forward(r.mhr_params, r.shape_params)  # (V, 3)
    verts_cam   = verts_local + r.cam_trans                        # camera space
    faces       = renderer.faces                                   # (F, 3) int32
```

## Model weights

The InstantHMR ONNX weights are hosted on HuggingFace:

> **<https://huggingface.co/momolesang/InstantHMR>**

Download `instanthmr.onnx` and place it under `models/`, or pass any path
via `--model`.

## Training

InstantHMR can be trained from two different label sources. **Both produce the
exact same `images/*.png` + `annotations/*.npz` layout** that
[`notebooks/distill_transformer_decoder.ipynb`](notebooks/distill_transformer_decoder.ipynb)
consumes, so the training notebook is identical either way:

1. **Distillation (default).** Run the SAM3D teacher over your own images with
   [`tools/annotate_dataset.py`](tools/annotate_dataset.py) to generate
   per-person pseudo-labels — see [`docs/annotation.md`](docs/annotation.md).
2. **Original ground-truth annotations.** Train directly on the released
   [`facebook/sam-3d-body-dataset`](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)
   — the human MHR fits used to build SAM 3D Body — converted to the same
   `.npz` schema by [`tools/parquet_to_npz.py`](tools/parquet_to_npz.py).

### Train on the original annotations (COCO example)

The dataset is gated: first **request access** on the
[dataset page](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)
and `huggingface-cli login`. The annotation parquets ship **without images**, so
you download the source images once per sub-dataset (COCO and MPII are the
easiest — a single `wget`; others may need extra access / undistortion, see the
[dataset setup guides](https://github.com/facebookresearch/sam-3d-body/tree/main/data)).

```bash
# 1. Annotations — download the coco_train split (uses Meta's sam-3d-body repo)
git clone https://github.com/facebookresearch/sam-3d-body
python sam-3d-body/data/scripts/download.py \
    --save_dir $ANN_DIR --splits coco_train

# 2. Images — COCO 2014 train (the split the parquet filenames reference)
mkdir -p $COCO_IMG_DIR && cd $COCO_IMG_DIR
wget http://images.cocodataset.org/zips/train2014.zip && unzip -q train2014.zip
cd -

# 3. Convert parquet -> per-crop images/ + annotations/ (.npz)
python tools/parquet_to_npz.py \
    --annotation_dir $ANN_DIR/coco_train \
    --image_dir      $COCO_IMG_DIR \
    --output_dir     data/sam3d_gt_coco \
    --validate

# 4. Point the notebook's config at the output and train:
#    cfg.images_dir      = "data/sam3d_gt_coco/images"
#    cfg.annotations_dir = "data/sam3d_gt_coco/annotations"
```

The converter recovers `cam_trans` from the MHR params, derives the 1.2× square
crop, skips low-quality fits (`mhr_valid`), and writes the same keys
`tools/annotate_dataset.py` does. `--validate` asserts that each person's
`joints_3d + cam_trans` reprojects onto the stored `joints_2d` to < 1 px.

### Other splits (MPII, AI Challenger, Harmony4D, EgoHumans)

Every split uses the **same converter** — the annotation schema and the MHR math
are dataset-independent (verified: 0 px reprojection and an image-centred
principal point on all of them). Only the **image preparation** differs. Download
the annotations the same way (`download.py --splits <split>`), prepare the images
as below, then run the converter into its own `data/sam3d_gt_<name>/` folder.

| Split | Image source | Prep | `--image_dir` |
|-------|--------------|------|---------------|
| `mpii_train` | `wget` from MPI servers (no registration) | unpack `mpii_human_pose_v1.tar.gz` | `$MPII_IMG_DIR` (contains `images/`) |
| `aic_train` | AI Challenger 2017 keypoint images (original site defunct — use a mirror) | arrange as `train/images/` | `$AIC_IMG_DIR` |
| `harmony4d_train` | HF `Jyun-Ting/Harmony4D` | **undistort** with Meta's `data/scripts/harmony4d/` script → `$HARMONY4D_IMG_DIR` | `$HARMONY4D_IMG_DIR` |
| `egohumans_train` | Google Drive (see setup guide) | **undistort** with Meta's `data/scripts/egohumans/` script → `$EGOHUMANS_IMG_DIR` | `$EGOHUMANS_IMG_DIR` |

```bash
# --- MPII -------------------------------------------------------------------
python sam-3d-body/data/scripts/download.py --save_dir $ANN_DIR --splits mpii_train
mkdir -p $MPII_IMG_DIR && cd $MPII_IMG_DIR
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
tar xzf mpii_human_pose_v1.tar.gz && cd -        # -> $MPII_IMG_DIR/images/
python tools/parquet_to_npz.py --annotation_dir $ANN_DIR/mpii_train \
    --image_dir $MPII_IMG_DIR --output_dir data/sam3d_gt_mpii --validate

# --- AI Challenger -----------------------------------------------------------
python sam-3d-body/data/scripts/download.py --save_dir $ANN_DIR --splits aic_train
#   place the keypoint images so that  $AIC_IMG_DIR/train/images/<hash>.jpg  exists
python tools/parquet_to_npz.py --annotation_dir $ANN_DIR/aic_train \
    --image_dir $AIC_IMG_DIR --output_dir data/sam3d_gt_aic --validate

# --- Harmony4D (needs undistortion) -----------------------------------------
python sam-3d-body/data/scripts/download.py --save_dir $ANN_DIR --splits harmony4d_train
#   1) download Jyun-Ting/Harmony4D into $HARMONY4D_DATA_DIR
#   2) undistort -> $HARMONY4D_IMG_DIR  (Meta's data/scripts/harmony4d/, see setup guide)
python tools/parquet_to_npz.py --annotation_dir $ANN_DIR/harmony4d_train \
    --image_dir $HARMONY4D_IMG_DIR --output_dir data/sam3d_gt_harmony4d --validate

# --- EgoHumans (needs undistortion) -----------------------------------------
python sam-3d-body/data/scripts/download.py --save_dir $ANN_DIR --splits egohumans_train
#   1) download EgoHumans into $EGOHUMANS_DATA_DIR
#   2) undistort -> $EGOHUMANS_IMG_DIR  (Meta's data/scripts/egohumans/, see setup guide)
python tools/parquet_to_npz.py --annotation_dir $ANN_DIR/egohumans_train \
    --image_dir $EGOHUMANS_IMG_DIR --output_dir data/sam3d_gt_egohumans --validate
```

> **Harmony4D / EgoHumans must be undistorted first.** Their annotations
> (`keypoints_2d`, `cam_int`) correspond to the **undistorted** frames produced
> by Meta's per-dataset scripts, so point `--image_dir` at the undistorted output,
> not the raw download. The image-prep details and exact undistortion commands
> live in the [dataset setup guides](https://github.com/facebookresearch/sam-3d-body/tree/main/data).

To train on several datasets at once, convert each into its own
`data/sam3d_gt_<name>/` and either point the notebook at one of them or merge the
`images/` + `annotations/` folders into a single directory — the filenames are
prefixed per dataset, so they never collide.

## Documentation

- **[`docs/architecture.md`](docs/architecture.md)** — network design
  (RepViT backbone, 9-query decoder, CLIFF condition).
- **[`docs/annotation.md`](docs/annotation.md)** — training data generation
  with the SAM3D teacher.
- **[`docs/training.md`](docs/training.md)** — distillation and ONNX export.

## License

The code in this repository is released under the **Apache License 2.0**
(see [LICENSE](LICENSE)).

The model weights distributed at
<https://huggingface.co/momolesang/InstantHMR> are released under the
[SAM license](https://github.com/facebookresearch/sam-3d-body), since
InstantHMR is a distillation of `facebook/sam-3d-body-dinov3`. Please
review the SAM and RF-DETR licenses before downstream use.
