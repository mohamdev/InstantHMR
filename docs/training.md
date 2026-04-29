# Training

The InstantHMR training notebook is
[`notebooks/distill_transformer_decoder.ipynb`](../notebooks/distill_transformer_decoder.ipynb).
It is the source of truth — this page is a tour, not a re-implementation.

## Setup

```bash
pip install torch timm onnx tqdm loguru
jupyter notebook notebooks/distill_transformer_decoder.ipynb
```

You'll need the annotations produced by
[`docs/annotation.md`](annotation.md). The notebook expects the directory
layout:

```
<dataset_root>/
├── images/        *.png    (224×224 crops)
└── annotations/   *.npz    (teacher outputs)
```

## What the notebook does

1. **Dataset** — loads the `(image, annotation)` pairs, applies light
   augmentation (colour jitter, horizontal flip with paired-joint remapping),
   and yields the 4-tuple `(image, cliff_cond, gt_targets, mask)`.
2. **Architecture** — defines InstantHMR (RepViT-M1.5 backbone + 9-token
   cross-attention decoder) exactly as it ships in this repo's
   `instanthmr/inference.py` (the runtime path is parameter-compatible
   with the trained `.pth`).
3. **Loss** — weighted L1 on each output head:
   - mhr_params, shape_params, cam_trans (parameter regression),
   - joints_3d (mean per-joint position error in camera frame),
   - joints_2d (after re-projection back to full-frame pixels).
4. **Training loop** — AdamW with cosine schedule + warm-up, mixed precision
   on CUDA, gradient clipping at 1.0.
5. **Validation** — held-out split, MPJPE on `joints_3d` reported.
6. **Export** — at the end of training, the model is exported to
   `instanthmr.onnx` with dynamic batch axis using the same input schema
   the demo expects (see [`models/README.md`](../models/README.md)).

## Hyperparameters at a glance

| | |
|---|---|
| Backbone | RepViT-M1.5 (`timm`) |
| Decoder layers | 2 |
| `d_model` | 256 |
| Heads | 4 |
| Dropout | 0.1 |
| Image size | 224 |
| Optimiser | AdamW (β=0.9/0.999, wd=1e-2) |
| LR schedule | cosine + linear warm-up |
| Precision | mixed (fp16 forward, fp32 master) |

The exact values for the released checkpoint live in the notebook's config
cell.

## Re-exporting to ONNX

The notebook's export cell is reproduced below for reference:

```python
dummy_image = torch.randn(1, 3, 224, 224, device=device)
dummy_cliff = torch.randn(1, 3, device=device)

torch.onnx.export(
    model.eval(),
    (dummy_image, dummy_cliff),
    "instanthmr.onnx",
    input_names=["image", "cliff_cond"],
    output_names=[
        "mhr_params", "shape_params", "cam_trans",
        "joints_2d", "joints_3d",
    ],
    dynamic_axes={
        "image":      {0: "batch"},
        "cliff_cond": {0: "batch"},
        "mhr_params": {0: "batch"},
        "shape_params": {0: "batch"},
        "cam_trans":  {0: "batch"},
        "joints_2d":  {0: "batch"},
        "joints_3d":  {0: "batch"},
    },
    opset_version=17,
)
```

Verify with `onnxruntime`:

```python
import onnxruntime as ort
s = ort.InferenceSession("instanthmr.onnx",
                         providers=["CPUExecutionProvider"])
print([(i.name, i.shape) for i in s.get_inputs()])
print([(o.name, o.shape) for o in s.get_outputs()])
```

## Tips

- **Don't drift the preprocessing.** The demo's
  `InstantHMR._preprocess` (`instanthmr/inference.py`) is the exact
  inverse of the annotation crop: 1.2× square padded crop → 224 → ImageNet
  norm. Any change to either side has to be mirrored on the other.
- **Joints are body-centred.** Don't add `cam_trans` to `joints_3d` before
  computing the loss; the teacher stores them as body-centred, and
  InstantHMR's last layer is trained against that.
- **CLIFF conditioning lives in full-frame coords**, not crop coords.
  `cx_norm = 2·cx/W − 1` etc. Recomputing this from the crop will silently
  degrade the camera head.
