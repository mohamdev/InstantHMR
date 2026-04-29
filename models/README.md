# InstantHMR — model weights

A lightweight, ONNX-exportable distillation of
[`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)
for **3D human pose estimation and mesh recovery**: a RepViT-M1.5
backbone + a 9-token cross-attention decoder + CLIFF camera conditioning.
Trained to mimic the SAM3D teacher's per-person 70-keypoint outputs from
a single 224×224 crop.

Demo, training, and inference code live at the main InstantHMR
repository.

## Files

| File | Size | Purpose |
|---|---|---|
| `instanthmr.onnx` | ~77 MB | fp16 ONNX export — what the demo and `InstantHMR(...)` wrapper load. |
| `instanthmr.pth`  | ~461 MB | PyTorch checkpoint for fine-tuning / further distillation. |

Most users only need `instanthmr.onnx`. The `.pth` is included for people
who want to continue training, swap the head, or re-export with different
options — see [`docs/training.md`](https://github.com/mohamdev/InstantHMR/blob/main/docs/training.md)
in the main repo.

## ONNX I/O

- **Inputs**
  - `image` — `(N, 3, 224, 224)` float32, ImageNet-normalised RGB.
  - `cliff_cond` — `(N, 3)` float32, `[cx_norm, cy_norm, b_scale]` of the
    detector bbox in the **full** frame (`cx_norm, cy_norm ∈ [-1, 1]`,
    `b_scale = max(bw, bh) / max(W, H)`).
- **Outputs**
  - `mhr_params` — `(N, 204)` MHR pose + scale parameters.
  - `shape_params` — `(N, 45)` MHR identity blendshapes.
  - `cam_trans` — `(N, 3)` body translation in camera frame (metres).
  - `joints_2d` — `(N, 70, 2)` keypoints in normalised crop coords `[-1, 1]`.
  - `joints_3d` — `(N, 70, 3)` body-centred 3D keypoints (metres, Y-down).

The 70 keypoints follow the MHR70 ordering — see
[`instanthmr/skeleton.py`](https://github.com/mohamdev/InstantHMR/blob/main/instanthmr/skeleton.py)
for joint names and skeleton edges.

## Quick start (ONNX)

Minimal end-to-end inference with `onnxruntime`, no extra dependencies:

```python
import cv2
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 1. Load the ONNX session (CUDA if available, otherwise CPU).
sess = ort.InferenceSession(
    "instanthmr.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# 2. Load an RGB image and a person bbox [x1, y1, x2, y2].
image = cv2.cvtColor(cv2.imread("person.jpg"), cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]
x1, y1, x2, y2 = 120, 40, 380, 600   # replace with your detector output

# 3. Square 1.2x crop around the bbox, resized to 224x224.
cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
size = max(x2 - x1, y2 - y1) * 1.2
sx1, sy1 = int(cx - size / 2), int(cy - size / 2)
sx2, sy2 = int(cx + size / 2), int(cy + size / 2)
pad = [max(0, -sy1), max(0, sy2 - H), max(0, -sx1), max(0, sx2 - W)]
patch = image[max(0, sy1):min(H, sy2), max(0, sx1):min(W, sx2)]
patch = cv2.copyMakeBorder(patch, *pad, cv2.BORDER_CONSTANT, value=0)
crop  = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)

# 4. Normalise + CHW.
img = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
img = np.transpose(img, (2, 0, 1))[None]   # (1, 3, 224, 224)

# 5. CLIFF conditioning vector (full-frame coords).
cliff = np.array([
    2.0 * cx / W - 1.0,
    2.0 * cy / H - 1.0,
    max(x2 - x1, y2 - y1) / max(W, H),
], dtype=np.float32)[None]                  # (1, 3)

# 6. Run the model.
mhr_params, shape_params, cam_trans, joints_2d, joints_3d = sess.run(
    None, {"image": img, "cliff_cond": cliff}
)

print(joints_3d.shape)   # (1, 70, 3) — body-centred metres, Y-down
print(joints_2d.shape)   # (1, 70, 2) — normalised crop coords [-1, 1]
print(mhr_params.shape)  # (1, 204)   — feed into Meta's MHR module for a mesh
```

For a full pipeline (RF-DETR person detector + InstantHMR + Rerun
visualiser), use the [`PosePipeline`](https://github.com/mohamdev/InstantHMR/blob/main/instanthmr/pipeline.py)
helper from the main repository — `pip install -r requirements.txt` then
`python demo.py --image path/to/photo.jpg`.

## Body mesh

InstantHMR does **not** run the MHR mesh decoder — rendering the full
mesh is resource-heavy and the default pipeline stays lightweight. The
raw `mhr_params` and `shape_params` are exposed so you can feed them
into Meta's MHR TorchScript module from the original
[SAM 3D Body release](https://github.com/facebookresearch/sam-3d-body)
when you need vertices.

## License

These weights are released under the **SAM license**, since InstantHMR
is a distillation of `facebook/sam-3d-body-dinov3`. Refer to
<https://github.com/facebookresearch/sam-3d-body> for the full license
text and applicable use restrictions.

The accompanying source code in the InstantHMR repository is released
separately under the Apache License 2.0.
