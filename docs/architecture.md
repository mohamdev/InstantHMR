# Model architecture

InstantHMR is a 17 M-parameter network that mimics
`facebook/sam-3d-body-dinov3` on the **per-person 3D pose estimation** task,
running ~80× faster on a single GPU and exporting cleanly to ONNX / TFLite /
QNN for mobile and edge deployment.

## High level

```
   RF-DETR  ──►  square 1.2× crop  ──►  RepViT backbone  ──►  cross-attn decoder
                                                                      │
                                         CLIFF cond [cx, cy, scale] ──┘
                                                                      ▼
                                      mhr_params (204), shape (45), cam_trans (3),
                                      joints_2d (70, 2), joints_3d (70, 3)
```

## Backbone — RepViT-M1.5

`timm.create_model("repvit_m1_5", pretrained=False, features_only=True)`
gives a `(B, 512, 7, 7)` feature map at 224 input. Re-parameterised
inverted-residual blocks make this fast on both desktop and mobile CPUs.

## Decoder — 9-query cross-attention

Two `nn.TransformerDecoderLayer` blocks (`d_model=256`, 4 heads, GELU,
pre-LN) attend the flattened backbone features (49 tokens) with 9 learnable
query tokens:

| Query | Output                       | Head shape |
|-------|------------------------------|-----------:|
| 0     | camera translation (CLIFF)   | (3,) |
| 1     | pose (axis-angle joints)     | (136,) |
| 2     | scales                       | (68,) |
| 3     | identity shape blendshapes   | (45,) |
| 4–8   | torso / L arm / R arm / L leg / R leg → joints | (5×256 → 70×3) |

A fixed 2D sin-cos positional embedding is added to the 7×7 token grid.

## CLIFF camera conditioning

The camera-translation head is the only one that consumes the CLIFF
conditioning vector — `[cx_norm, cy_norm, b_scale]` of the detector bbox
(RF-DETR at runtime, YOLO during annotation) in **full-frame** coordinates.
This is what lets InstantHMR predict translation consistent with the
original perspective of the source image, without ever seeing pixels
outside the 224×224 crop.

```
cam_token         (B, 256)
cliff_cond        (B,   3)
        │             │
        └─────cat─────┘
              │
       LayerNorm + MLP
              │
              ▼
        cam_trans  (B, 3)
```

## Output coordinate frame

Both `joints_3d` and `cam_trans` are in metres in the SAM3D camera frame:

- **X right**, **Y down**, **Z forward** (right-handed, Y-down).
- `joints_3d` are body-centred — add `cam_trans` to land in camera space.
- `joints_2d` come from a dedicated 2D head, in **normalised crop coords**
  `[-1, 1]`; the demo's `instanthmr/inference.py` re-projects them to
  full-frame pixels using the bbox-derived crop transform.

## Why distill?

The DINOv3-based teacher is excellent but heavy: its tokenizer + decoder runs
at <20 FPS on a desktop GPU and doesn't fit comfortably on edge hardware.
Distillation gives us InstantHMR — a model that:

- runs at ~200 FPS on a single RTX 4070, 
- exports to ONNX / TFLite / QNN, and
- preserves the SAM3D 70-keypoint & MHR outputs, so any downstream code that
  consumed `pred_keypoints_3d` keeps working.

## Detection: RF-DETR

The demo pipeline uses **RF-DETR** (Roboflow's transformer-based detector)
for the upstream person detection step. We picked it over YOLO because:

- it has **stronger recall on small / partially-occluded persons**, which
  matters a lot once the bbox is tight enough to drive a pose model;
- its single-stage transformer head produces calibrated confidence scores,
  so the `--det-confidence` knob behaves predictably; and
- it ships with a `optimize_for_inference()` pass (TorchScript +
  fused ops) that reliably halves detector latency on a desktop GPU.

At inference the demo reports detector and HMR latency **separately**, both
in the console and as scalar plots in the Rerun viewer, so it's always
clear whether RF-DETR or InstantHMR is the bottleneck on a given machine.
The default `--detector-variant medium` is tuned for ~480p–1080p input.
Drop to `nano` / `small` for higher FPS at lower resolution, or step up to
`base` / `large` for crowded scenes.
