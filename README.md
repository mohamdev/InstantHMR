# InstantHMR

![InstantHMR demo](models/instanthmr.gif)

A lightweight, ONNX-exportable distillation of
[`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)
for **3D human pose estimation and mesh recovery**: a RepViT-M1.5 backbone + a 9-token
cross-attention decoder + CLIFF camera conditioning. Trained to mimic the
SAM3D teacher's per-person 70-keypoint outputs from a single 224×224 crop.

InstantHMR ships as a single `.onnx` file. The demo pipeline pairs it with
**RF-DETR** for person detection — both stages are timed independently so
you always know where the latency is going.

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

> **Note on the body mesh.** InstantHMR does **not** run the MHR mesh
> decoder — rendering the full body mesh is expensive and we want the
> default pipeline to stay lightweight. The raw `mhr_params` and
> `shape_params` are still exposed on every `HMRPrediction` so you can
> feed them into Meta's MHR TorchScript module from the original
> [SAM 3D Body release](https://github.com/facebookresearch/sam-3d-body)
> if you need a mesh.

## Install

```bash
git clone <this-repo> instanthmr
cd instanthmr

conda create -n instanthmr python=3.11 -y
conda activate instanthmr

# Picks the right torch + onnxruntime wheels for your machine:
#   - Linux + NVIDIA GPU → cu128 (Blackwell, RTX 50-series) or cu124 (Ada/Ampere/Hopper)
#                          + onnxruntime-gpu + bundled CUDA / cuDNN runtime libs
#   - macOS              → stock torch + onnxruntime (CoreML EP included)
#   - Linux without GPU  → CPU torch + onnxruntime
python install.py
```

`python install.py --dry-run` prints the pip commands without running
them. `python install.py --force-cpu` skips GPU detection on Linux. If
you'd rather manage wheels yourself, `pip install -r requirements.txt`
still works — it gets the CPU torch + onnxruntime path.

The first run downloads the RF-DETR (medium) checkpoint into the rfdetr
cache directory automatically. The InstantHMR ONNX weights are released
separately on HuggingFace — see [Model weights](#model-weights) below.

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
detector / HMR / total-latency plot, and the 3D scene with the predicted
camera frustum.

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

result = pipeline.predict(image_rgb)
for r in result.persons:
    print(r.joints_3d_cam.shape)   # (70, 3)
    print(r.joints_2d.shape)       # (70, 2)
    print(r.mhr_params.shape)      # (204,) — for downstream MHR mesh decoding
    print(r.shape_params.shape)    # (45,)
```

The 70 keypoints follow the MHR70 ordering — see
[`instanthmr/skeleton.py`](instanthmr/skeleton.py) for joint names and the
canonical skeleton edge list.

## Model weights

The InstantHMR ONNX weights are hosted on HuggingFace:

> **<https://huggingface.co/momolesang/InstantHMR>**

Download `instanthmr.onnx` and place it under `models/`, or pass any path
via `--model`.

## Documentation

- **[`docs/architecture.md`](docs/architecture.md)** — what the network
  looks like (RepViT backbone, 9-query decoder, CLIFF condition).
- **[`docs/annotation.md`](docs/annotation.md)** — how the training data
  was generated with the SAM3D teacher.
- **[`docs/training.md`](docs/training.md)** — how the released checkpoint
  was distilled and exported to ONNX.

## License

The code in this repository is released under the **Apache License 2.0**
(see [LICENSE](LICENSE)).

The model weights distributed at
<https://huggingface.co/momolesang/InstantHMR> are released under the
[SAM license](https://github.com/facebookresearch/sam-3d-body), since
InstantHMR is a distillation of `facebook/sam-3d-body-dinov3`. Please
review the SAM and RF-DETR licenses before downstream use.
