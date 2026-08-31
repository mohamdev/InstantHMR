#!/usr/bin/env python3
"""Turn one parquet split + its source frames into a trainable dataset folder.

Output layout, per source dataset (train and test merge into one folder, which
is what data/ already does and what the training script's discovery expects):

    <data_root>/sam3d_gt_<dataset>/
        original_images/   the referenced source frames, native relative paths
        annotations/       <name>.npz
        body_crops/        <name>.png            224 square person crop
        hands_crops/       <name>_l.png  <name>_r.png
        images -> body_crops                     compat symlink, see below

`body_crops` is byte-identical to what `tools/parquet_to_npz.py` writes into
`images/`: this script imports that module's crop and annotation builders
rather than re-implementing them, so a folder built here and a folder built
there are interchangeable. The `images` symlink means the current
`SAM3DStudentDataset._find_dataset_dirs` finds the folder with no code change.

The npz keeps every key the training script already reads and adds five that
it ignores today and that a hand branch will need:

    image_relpath   (str)   exact source path — the existing `<dataset>_<stem>`
                            naming flattens "/" to "_" and is NOT invertible for
                            harmony4d / egohumans / egoexo4d / 3dpw
    joints_2d_vis   (70,)   per-keypoint visibility, dropped by parquet_to_npz.
                            It is all-ones on coco/mpii/aic/3dpw/harmony4d/
                            egohumans/egoexo4d, but on SA-1B a mean of 44 of 70
                            keypoints are exactly (0, 0) placeholders — writing
                            those as if they were observations poisons the 2D
                            loss and inflates every hand box.
    joints_3d_conf  (70,)   the matching confidence column of keypoints_3d
    hand_box_l/_r   (4,)    tight xyxy over the 21 hand keypoints, full-frame px
    hand_crop_l/_r  (4,)    the square crop actually written (NaN if skipped)
    hand_valid_l/_r ()      bool: was a crop written

Hand crops are stored UNFLIPPED. SAM3D trains a single right-hand model and
mirrors the left at inference; flipping a loaded crop is a free numpy slice, so
baking the mirror into disk would only remove a choice.

    python datasets_pipeline/build_split.py \
        --split coco_train --annotation-dir $ANN_DIR/coco_train \
        --image-dir $COCO_IMG_DIR --data-root data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from registry import dataset_for_split, image_relpath, output_dir_for_split  # noqa: E402
from tools.parquet_to_npz import (  # noqa: E402
    build_annotation, get_square_crop_padded, reproj_error,
)

# MHR70 keypoint blocks. Index 41 is right_wrist, 62 is left_wrist; each hand
# block is its 20 finger joints plus that wrist. Taken from JOINT_NAMES in
# instanthmr_distill_train/train_distill_mhr_only.py.
RIGHT_HAND_KPS = list(range(21, 42))
LEFT_HAND_KPS = list(range(42, 63))


def hand_boxes(joints_2d: np.ndarray,
               vis: np.ndarray | None = None) -> dict[str, np.ndarray | None]:
    """Tight xyxy box over each hand's 21 keypoints, in full-frame pixels.

    Returns None for a hand that is not fully visible. A partially visible hand
    cannot give a usable box: the invisible keypoints are stored as (0, 0), so
    including even one drags the box to the image corner. On SA-1B that is the
    normal case — only 0.3% of hands have all 21 keypoints marked visible.
    """
    out: dict[str, np.ndarray | None] = {}
    for side, idx in (("l", LEFT_HAND_KPS), ("r", RIGHT_HAND_KPS)):
        if vis is not None and not (vis[idx] > 0).all():
            out[side] = None
            continue
        p = joints_2d[idx]
        out[side] = np.array([p[:, 0].min(), p[:, 1].min(),
                              p[:, 0].max(), p[:, 1].max()], dtype=np.float32)
    return out


def square_box(box: np.ndarray, expand: float) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    half = max(box[2] - box[0], box[3] - box[1]) * expand / 2.0
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def enough_inside(sq: np.ndarray, h: int, w: int, min_frac: float) -> bool:
    """Reject hands that are mostly outside the frame — the crop would be padding."""
    ix1, iy1 = max(0.0, sq[0]), max(0.0, sq[1])
    ix2, iy2 = min(float(w), sq[2]), min(float(h), sq[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    area = (sq[2] - sq[0]) * (sq[3] - sq[1])
    return ((ix2 - ix1) * (iy2 - iy1)) / max(area, 1e-6) >= min_frac


def write_crop(img_rgb, box, size, path: Path) -> None:
    crop, _ = get_square_crop_padded(img_rgb, box, expand=1.0)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def place_original(src: Path, dst: Path, mode: str) -> None:
    if mode == "none" or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, help="e.g. coco_train, harmony4d_train")
    p.add_argument("--annotation-dir", required=True, type=Path)
    p.add_argument("--image-dir", required=True, type=Path,
                   help="root the parquet `image` paths are relative to")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="override the sam3d_gt_<dataset> destination")

    p.add_argument("--body-size", type=int, default=224)
    p.add_argument("--crop-format", choices=("png", "jpg"), default="png",
                   help="png keeps body_crops byte-identical to the existing "
                        "data/ tree; jpg is ~5x smaller and ~4x faster to decode")
    p.add_argument("--jpg-quality", type=int, default=95)

    p.add_argument("--hands", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hand-size", type=int, default=224)
    p.add_argument("--hand-expand", type=float, default=2.0,
                   help="square hand box = 21-keypoint span x this. 2.0 leaves "
                        "room to jitter the box inside the stored crop at "
                        "train time, which is what makes the model tolerant of "
                        "a real detector's boxes.")
    p.add_argument("--hand-min-px", type=float, default=64.0,
                   help="skip hands whose SPAN (before --hand-expand) is smaller "
                        "than this in source pixels. 64 is SAM3D's own gate.")
    p.add_argument("--hand-min-inside", type=float, default=0.6,
                   help="skip hands whose square crop is less than this fraction "
                        "inside the frame")

    p.add_argument("--originals", choices=("copy", "link", "hardlink", "none"),
                   default="copy",
                   help="copy: self-contained, prunes unreferenced source images. "
                        "link/hardlink: free, but the source root must survive.")

    p.add_argument("--image-prefix", default=None,
                   help="only rows whose `image` starts with this — the hook "
                        "stream_archives.py uses to build one archive at a time")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--keep-invalid", action="store_true",
                   help="keep rows with mhr_valid == False (dropped by default)")
    p.add_argument("--validate", action="store_true",
                   help="assert < --reproj-tol px reprojection error per person")
    p.add_argument("--reproj-tol", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = dataset_for_split(args.split)
    out_dir = args.output_dir or output_dir_for_split(args.split, args.data_root)

    dirs = {k: out_dir / k for k in
            ("original_images", "annotations", "body_crops", "hands_crops")}
    for k, d in dirs.items():
        if k == "original_images" and args.originals == "none":
            continue
        if k == "hands_crops" and not args.hands:
            continue
        d.mkdir(parents=True, exist_ok=True)

    # Let the unmodified training script discover this folder.
    compat = out_dir / "images"
    if not compat.exists():
        os.symlink("body_crops", compat)

    parquets = sorted(args.annotation_dir.glob("*.parquet"))
    if not parquets:
        sys.exit(f"no parquet files in {args.annotation_dir}")

    ext = args.crop_format
    imwrite_params = ([int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality]
                      if ext == "jpg" else [])
    existing = {p.stem for p in dirs["annotations"].glob("*.npz")}
    total = len(existing)
    st = {k: 0 for k in ("crops", "hands_l", "hands_r", "hand_too_small",
                         "hand_out_of_frame", "hand_occluded", "skipped_existing",
                         "skipped_invalid", "skipped_prefix", "missing_image",
                         "failed")}

    pbar = tqdm(parquets, desc=f"{args.split} shards", unit="shard")
    for pq_path in pbar:
        df = pd.read_parquet(pq_path)
        img_cache: dict[str, np.ndarray | None] = {}

        for _, row in df.iterrows():
            dataset, image = row["dataset"], row["image"]
            if args.image_prefix and not image.startswith(args.image_prefix):
                st["skipped_prefix"] += 1
                continue

            stem = (os.path.splitext(image)[0]
                    .replace("/", "_").replace(os.sep, "_").replace(" ", "_"))
            name = f"{dataset}_{stem}_p{int(row['person_id'])}"
            if name in existing:
                st["skipped_existing"] += 1
                continue
            if not args.keep_invalid and not bool(row["mhr_valid"]):
                st["skipped_invalid"] += 1
                continue

            relpath = image_relpath(dataset, image)
            if image not in img_cache:
                bgr = cv2.imread(str(args.image_dir / relpath))
                img_cache[image] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                    if bgr is not None else None)
            img_rgb = img_cache[image]
            if img_rgb is None:
                st["missing_image"] += 1
                continue

            h, w = img_rgb.shape[:2]
            try:
                ann = build_annotation(row, h, w)
                if args.validate:
                    err = reproj_error(ann["joints_3d"], ann["cam_trans"],
                                       ann["joints_2d"], ann["cam_focal_length"],
                                       ann["orig_shape"])
                    if err > args.reproj_tol:
                        raise ValueError(f"reproj err {err:.2f}px > {args.reproj_tol}px")

                body_crop, sq_bbox = get_square_crop_padded(img_rgb, ann["bbox"],
                                                            expand=1.2)
                body_crop = cv2.resize(body_crop, (args.body_size, args.body_size),
                                       interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(dirs["body_crops"] / f"{name}.{ext}"),
                            cv2.cvtColor(body_crop, cv2.COLOR_RGB2BGR),
                            imwrite_params)

                ann["bbox_square"] = sq_bbox
                ann["image_relpath"] = relpath
                ann["dataset"] = dataset
                kp2 = np.asarray(row["keypoints_2d"].tolist(), dtype=np.float32)
                kp3 = np.asarray(row["keypoints_3d"].tolist(), dtype=np.float32)
                vis = kp2.reshape(-1, 3)[:, 2].copy()
                ann["joints_2d_vis"] = vis
                ann["joints_3d_conf"] = kp3.reshape(-1, 4)[:, 3].copy()

                boxes = hand_boxes(ann["joints_2d"], vis)
                for side in ("l", "r"):
                    tight = boxes[side]
                    nan4 = np.full(4, np.nan, dtype=np.float32)
                    ann[f"hand_box_{side}"] = tight if tight is not None else nan4
                    ann[f"hand_crop_{side}"] = nan4
                    ann[f"hand_valid_{side}"] = np.array(False)
                    if tight is None:
                        st["hand_occluded"] += 1
                        continue
                    span = max(tight[2] - tight[0], tight[3] - tight[1])
                    sq = square_box(tight, args.hand_expand)
                    if not args.hands:
                        continue
                    if span < args.hand_min_px:
                        st["hand_too_small"] += 1
                        continue
                    if not enough_inside(sq, h, w, args.hand_min_inside):
                        st["hand_out_of_frame"] += 1
                        continue
                    write_crop(img_rgb, sq, args.hand_size,
                               dirs["hands_crops"] / f"{name}_{side}.{ext}")
                    ann[f"hand_crop_{side}"] = sq
                    ann[f"hand_valid_{side}"] = np.array(True)
                    st[f"hands_{side}"] += 1

                np.savez(dirs["annotations"] / f"{name}.npz", **ann)
                if args.originals != "none":
                    place_original(args.image_dir / relpath,
                                   dirs["original_images"] / relpath,
                                   args.originals)
            except Exception as e:  # noqa: BLE001
                st["failed"] += 1
                tqdm.write(f"[fail] {name}: {e}")
                continue

            existing.add(name)
            st["crops"] += 1
            total += 1
            if args.max_samples and total >= args.max_samples:
                pbar.close()
                report(st, total, out_dir)
                return

        pbar.set_postfix(crops=st["crops"], hands=st["hands_l"] + st["hands_r"],
                         missing=st["missing_image"])

    report(st, total, out_dir)


def report(st: dict, total: int, out_dir: Path) -> None:
    print("=" * 62)
    print(f"BUILD COMPLETE -> {out_dir}")
    for k, v in st.items():
        print(f"  {k:20s}: {v:,}")
    print(f"  {'total on disk':20s}: {total:,}")
    print("=" * 62)


if __name__ == "__main__":
    main()
