#!/usr/bin/env python3
"""Populate original_images/ for a split that was built with --originals none.

build_split.py skips a row whose crop already exists (`skipped_existing`) BEFORE
it reaches place_original, so re-running the build with a different --originals
backfills nothing. This walks the finished annotations instead: every npz stores
`image_relpath`, the exact source path the crop came from, so the originals can
be placed without re-decoding a single image.

    python backfill_originals.py \
        --dir $DATA_ROOT/sam3d_gt_aic \
        --image-dir $IMG_ROOT/aic \
        --mode hardlink

hardlink is the right default when the split and the source frames sit on the
same filesystem: one directory entry, zero extra data blocks, and the result is
self-contained if the source tree is later deleted. It falls back to a copy
across filesystems. Use link (symlink) when the source is a permanent read-only
mirror such as $DSDIR, and copy when you want the split to stand alone.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_split import place_original  # noqa: E402


def relpaths_from_annotations(ann_dir: Path, workers: int) -> set[str]:
    """The unique set of source frames a split's npz files point at."""
    files = sorted(ann_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz under {ann_dir}")

    def one(f: Path) -> str | None:
        try:
            with np.load(f, allow_pickle=True) as d:
                if "image_relpath" not in d.files:
                    return None
                return str(d["image_relpath"])
        except Exception:  # noqa: BLE001
            return None

    out: set[str] = set()
    missing_key = 0
    with ThreadPoolExecutor(workers) as ex:
        for rp in tqdm(ex.map(one, files), total=len(files),
                       desc="reading annotations", unit="npz"):
            if rp is None:
                missing_key += 1
            else:
                out.add(rp)
    if missing_key:
        print(f"  {missing_key:,} npz had no image_relpath field "
              f"(built by an older script — those frames cannot be resolved)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, type=Path,
                   help="a built split, e.g. $DATA_ROOT/sam3d_gt_aic")
    p.add_argument("--image-dir", required=True, type=Path,
                   help="root the image_relpath values are relative to")
    p.add_argument("--mode", choices=("copy", "link", "hardlink"),
                   default="hardlink")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be placed, write nothing")
    args = p.parse_args()

    ann_dir = args.dir / "annotations"
    dst_root = args.dir / "original_images"
    if not ann_dir.is_dir():
        raise SystemExit(f"{ann_dir} does not exist")

    rels = relpaths_from_annotations(ann_dir, args.workers)
    print(f"\n{len(rels):,} unique source frames referenced")

    st = Counter()

    def place(rel: str) -> str:
        src, dst = args.image_dir / rel, dst_root / rel
        if dst.exists():
            return "already_present"
        if not src.exists():
            return "missing_source"
        if args.dry_run:
            return "would_place"
        try:
            place_original(src, dst, args.mode)
            return "placed"
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"[fail] {rel}: {e}")
            return "failed"

    with ThreadPoolExecutor(args.workers) as ex:
        for r in tqdm(ex.map(place, sorted(rels)), total=len(rels),
                      desc=f"{args.mode} originals", unit="img"):
            st[r] += 1

    print("\n" + "=" * 62)
    print(f"  {args.dir.name}  ({args.mode}{', DRY RUN' if args.dry_run else ''})")
    for k in ("placed", "would_place", "already_present",
              "missing_source", "failed"):
        if st[k]:
            print(f"  {k:<18}: {st[k]:,}")
    print("=" * 62)
    if st["missing_source"]:
        print(f"  {st['missing_source']:,} frames are not under {args.image_dir} —"
              f" check that --image-dir matches the build's --image-dir.")


if __name__ == "__main__":
    main()
