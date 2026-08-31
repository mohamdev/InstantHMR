#!/usr/bin/env python3
"""Cost a split before you download it.

Reads only the `image` and `mhr_valid` columns of the parquets — cheap, no
frames needed — and reports how many person crops and unique source frames a
split will produce, plus what each output folder will weigh.

    python datasets_pipeline/plan_disk.py --annotation-dir $ANN_DIR/aic_train
    python datasets_pipeline/plan_disk.py --annotation-dir $ANN_DIR/*  --brief

Per-file sizes are measured on the existing data/ tree, not assumed:
  224 PNG body crop ~75 KB, 224 JPEG q95 ~24 KB.
Source-frame sizes come from registry.py's measured download sizes divided by
the frame count, except where the dataset is a known fixed resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

# KB per stored file, measured on data/sam3d_gt_*.
CROP_KB = {"png": 75.0, "jpg": 24.0}
# KB per source frame, measured (aic/coco/mpii/3dpw) or estimated from the
# 4K zips (harmony4d).
FRAME_KB = {"coco": 160.0, "mpii": 210.0, "aic": 70.0, "3dpw": 250.0,
            "harmony4d": 1400.0, "egohumans": 400.0, "egoexo4d": 400.0,
            "sa1b": 700.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotation-dir", nargs="+", type=Path, required=True)
    p.add_argument("--crop-format", choices=("png", "jpg"), default="png")
    p.add_argument("--hand-yield", type=float, default=0.20,
                   help="fraction of the 2 hands per person that clear "
                        "--hand-min-px; 0.20 is what AIC measures at 64 px")
    p.add_argument("--brief", action="store_true")
    args = p.parse_args()

    ckb = CROP_KB[args.crop_format]
    tot = {"rows": 0, "frames": 0, "body": 0.0, "hand": 0.0, "orig": 0.0}
    hdr = (f"{'split':26s} {'persons':>10s} {'frames':>10s} "
           f"{'body':>8s} {'hands':>8s} {'originals':>10s} {'TOTAL':>9s}")
    print(hdr)
    print("-" * len(hdr))

    for d in args.annotation_dir:
        shards = sorted(d.glob("*.parquet"))
        if not shards:
            continue
        split = d.name
        dataset = split.split("_")[0]
        rows = 0
        frames: set[str] = set()
        for s in shards:
            t = pq.read_table(s, columns=["image", "mhr_valid"])
            im = t.column("image").to_pylist()
            mv = t.column("mhr_valid").to_pylist()
            rows += sum(mv)
            frames.update(i for i, v in zip(im, mv) if v)

        body = rows * ckb / 1e6                                   # GB
        hand = rows * 2 * args.hand_yield * ckb / 1e6
        orig = len(frames) * FRAME_KB.get(dataset, 400.0) / 1e6
        print(f"{split:26s} {rows:10,d} {len(frames):10,d} "
              f"{body:7.1f}G {hand:7.1f}G {orig:9.1f}G {body+hand+orig:8.1f}G")
        tot["rows"] += rows
        tot["frames"] += len(frames)
        tot["body"] += body
        tot["hand"] += hand
        tot["orig"] += orig

    if not args.brief:
        print("-" * len(hdr))
        print(f"{'TOTAL':26s} {tot['rows']:10,d} {tot['frames']:10,d} "
              f"{tot['body']:7.1f}G {tot['hand']:7.1f}G {tot['orig']:9.1f}G "
              f"{tot['body']+tot['hand']+tot['orig']:8.1f}G")
        print()
        print("`originals` is the dominant term everywhere except COCO. If it "
              "does not fit, build with --originals none: body_crops and "
              "hands_crops are self-contained, and the npz `image_relpath` "
              "field records exactly which frame each crop came from so the "
              "originals can be re-fetched later for the rows you still want.")


if __name__ == "__main__":
    main()
