# Building the training dataset

InstantHMR is supervised entirely by the **SAM3D teacher**
(`facebook/sam-3d-body-dinov3`): for every detector-found person we run
the teacher and store its outputs alongside a tight 224×224 crop. The
training loop in `notebooks/distill_transformer_decoder.ipynb` then learns
to mimic those teacher outputs from the crop alone.

The bundled `tools/annotate_dataset.py` uses YOLO for the
detector during annotation. That choice is independent of the runtime
detector — InstantHMR sees the same 1.2× square crop either way, and the
demo uses RF-DETR at inference time for better recall. If you re-run the
annotation with a different detector, just preserve the crop geometry.

The annotation pipeline is `tools/annotate_dataset.py`.

## What it does

```
root_dir/                              output_dir/
├── COCO/         ─┐                   ├── images/
├── Harmony4D/     │   YOLO + SAM3D    │     COCO_<stem>_p0.png       (224×224)
├── Synthetic/     │  ──────────────►  │     COCO_<stem>_p0.npz
└── …             ─┘                   │     …
                                       └── annotations/
                                             COCO_<stem>_p0.npz
                                             …
```

For each detected person (capped at `--max_persons` per image):

1. **YOLO 11n** detects persons (`conf ≥ --confidence`, default 0.7).
2. The **full-resolution** image is passed to SAM3D with the raw bbox.
   SAM3D's CLIFF camera head sees the full frame, so `pred_cam_t` and
   `focal_length` are produced in real perspective coordinates — no manual
   camera math.
3. A 1.2× **square** crop around the bbox is resized to 224×224 and stored
   as a PNG.
4. The teacher outputs are stored as a `.npz` next to the image:

   | Key | Shape | Description |
   |-----|-------|-------------|
   | `orig_shape` | `(2,)` | `[H, W]` of the original frame |
   | `bbox` | `(4,)` | raw YOLO `[x1, y1, x2, y2]` |
   | `bbox_square` | `(4,)` | the 1.2× expanded square crop bounds |
   | `cam_focal_length` | `(2,)` | `[fx, fy]` (≈ √(H²+W²)) |
   | `cam_trans` | `(3,)` | SAM3D translation in camera frame |
   | `mhr_model_params` | `(204,)` | MHR pose + scale |
   | `shape_params` | `(45,)` | MHR identity blendshapes |
   | `joints_3d` | `(70, 3)` | SAM3D `pred_keypoints_3d` |
   | `joints_2d` | `(70, 2)` | SAM3D `pred_keypoints_2d`, in **full-frame** pixels |

The script is **resume-safe**: it scans existing `.npz` filenames at startup
and skips any image whose first crop is already on disk.

## Running it

```bash
# Setup: clone Meta's sam-3d-body and download the dinov3 teacher checkpoint.
git clone https://github.com/facebookresearch/sam-3d-body
huggingface-cli download facebook/sam-3d-body-dinov3 \
    --local-dir checkpoints/sam-3d-body-dinov3

# Annotate a tree of images
python tools/annotate_dataset.py \
    --root_dir   /data/training_images \
    --output_dir /data/cliff_annotations \
    --confidence 0.7

# Quick balanced subset (50k crops, shuffled across datasets)
python tools/annotate_dataset.py \
    --root_dir   /data/training_images \
    --output_dir /data/cliff_annotations \
    --shuffle --max_samples 50000

# Visually inspect what's getting saved
python tools/annotate_dataset.py \
    --root_dir   /data/training_images \
    --output_dir /data/cliff_annotations \
    --use_rerun
```

Useful options:

- `--shuffle` — process images in random order so a partial run still spans
  the full dataset diversity.
- `--max_samples N` — stop after N **crops** are saved (not images), counting
  whatever was already on disk.
- `--timeout 5.0` — per-image SIGALRM; rare GPU hangs don't kill the run.
- `--gc_interval 200` — periodic CUDA cache + GC reset to keep VRAM stable
  across long runs.

## Why a separate cloud script?

This is the same per-person preprocessing the demo uses at inference time
(`InstantHMR._preprocess`). Aligning the training data with the runtime
preprocessing was the single biggest factor in getting InstantHMR to match
SAM3D's accuracy: any drift between annotation crops and inference crops
shows up directly as test-time error.

## Datasets we used

InstantHMR was trained on a mix of:

- **COCO** (in-the-wild humans),
- **Harmony4D** (multi-person studio captures),
- a **synthetic dataset** of rendered humans, and
- domain-specific footage (rugby, aikido) that the model is meant to handle.

