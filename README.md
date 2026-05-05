# InstantHMR

![InstantHMR demo](models/instanthmr.gif)

A lightweight, ONNX-exportable distillation of
[`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)
for **3D human pose estimation and mesh recovery**: a RepViT-M1.5 backbone + a 9-token
cross-attention decoder + CLIFF camera conditioning. Trained to mimic the
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

## Recent improvements (RF-DETR ONNX, GPU ORT, warm-up)

These options sit on top of the default PyTorch RF-DETR + InstantHMR stack.

### RF-DETR via ONNX (`--detector-onnx`)

You can run the **person detector** through ONNXRuntime instead of PyTorch
`rfdetr` — useful when you already have an RF-DETR detection export or want
the detector on the same ORT path as InstantHMR:

```bash
python demo.py --video clip.mp4 --detector-onnx models/rf-detr-medium.onnx
```

Pre-exported RF-DETR ONNX checkpoints (nano through xxlarge, COCO/O365, segmentation variants) are available at **[PierreMarieCurie/rf-detr-onnx](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main)** on Hugging Face — download the `.onnx` you want and pass its path to `--detector-onnx`.

The ONNX graph must expose a **single** float32 image input (`NCHW`,
ImageNet-normalised RGB at the export resolution, typically square **576×576**)
and outputs named **`pred_boxes`** (cxcywh) and **`pred_logits`**, as in
Roboflow’s RF-DETR ONNX export. **`--detector-variant` is ignored** when this
flag is set.

### ONNX Runtime on NVIDIA GPUs

For InstantHMR and the optional RF-DETR ONNX detector to use the GPU, you need
the **`onnxruntime-gpu`** wheel (Linux + NVIDIA: `python install.py` installs
it together with the matching Torch CUDA stack). **Do not install** the plain
CPU package **`onnxruntime`** in the same environment as **`onnxruntime-gpu`**
— they overwrite shared files and you may end up on CPU-only providers or a
broken install. If that happens: `pip uninstall -y onnxruntime onnxruntime-gpu`
then reinstall **`onnxruntime-gpu`** (see also the comment above `onnxruntime`
in `requirements.txt`).

Verify GPU execution providers:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see **`CUDAExecutionProvider`** when CUDA libraries are available.

### Warm-up before inference (`PosePipeline.warmup`)

The first ONNXRuntime **CUDA** run often pays a large one-off cost (kernel
scheduling, allocator warm-up). **`PosePipeline.warmup(image_rgb, runs=2)`**
runs the full detector + HMR path before timed frames, resets internal
stride/caching state, and — if no person is detected on the warm-up frame —
runs one extra InstantHMR forward with a **synthetic centre crop** so the HMR
graph always executes once.

The **`demo.py`** runner calls this automatically (you will see
`Warming up inference …`); **`tools/bench.py`** warm-ups on the first video
frame and rewinds the capture so benchmarks are not dominated by cold start.

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
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
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

The 204-dim `mhr_params` encode joint rotations in 6-D representation
(34 joints × 6); `shape_params (45,)` encode identity blend shapes
(20 body + 20 head + 5 hand components). Facial expression is set to
neutral (zeros) since InstantHMR does not regress it.

## Performance tuning

On every machine we've measured, RF-DETR (the detector) costs ~5–10× more
per frame than InstantHMR. The two flags below target that bottleneck.

| Flag | Effect |
| --- | --- |
| `--detector-stride N` | Run RF-DETR every Nth frame; reuse the previous bbox (slightly expanded) on the in-between frames. Stride 2–3 is the single biggest knob — typically 2–3× end-to-end FPS for slow movement, with negligible quality loss. |
| `--detector-variant nano` | Use the smallest RF-DETR; biggest win on CPU / Apple Silicon where the detector is doing all the work. |
| `--device coreml` | On macOS, route the InstantHMR ONNX through `CoreMLExecutionProvider`. |
| `--no-batch-persons` | Disable batched multi-person HMR (one ONNX call per person). The default is batched. |

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

# Optional: RF-DETR ONNX + explicit GPU warm-up before latency-sensitive loops.
# pipeline = PosePipeline(..., detector_onnx="models/rf-detr-medium.onnx")
# pipeline.warmup(image_rgb)

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
