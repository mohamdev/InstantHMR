#!/usr/bin/env python3
"""Index SA-1B when it is stored as per-tar folders instead of a flat tree.

The parquet asks for a flat `sa_4409715.jpg`, but a mirror that unpacked the
1000 distribution tars in place holds `sa_000000/`, `sa_000001/`, ... instead —
which is what most cluster-wide mirrors look like. This walks the tree once and
writes an index that `build_split.py --image-index` uses to resolve names.

    python datasets_pipeline/index_sa1b.py \
        --image-dir /path/to/SegmentAnything_1B \
        --out sa1b_index.npz

SA-1B assigns ids to tars in blocks, so the index is usually just 1000
(min_id, max_id, folder) rows — a few KB. If the ranges turn out to overlap the
full id->folder table is written instead (~110 MB for 11M images). Either way
lookup is an O(log n) searchsorted.

Walking ~11M directory entries takes minutes and is single-threaded — run it as
a batch job, not on a shared login node.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ID_RE = re.compile(r"^sa_(\d+)\.jpe?g$", re.I)


def find_jpeg_dir(folder: Path, max_depth: int = 3) -> Path | None:
    """The directory inside `folder` that actually holds the jpgs.

    Mirrors differ: some put the images straight in sa_NNNNNN/, others add an
    images/ level. Probe rather than assume.
    """
    stack = [(folder, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            subdirs = []
            for e in os.scandir(d):
                if e.is_file() and ID_RE.match(e.name):
                    return d
                if e.is_dir() and depth < max_depth:
                    subdirs.append((Path(e.path), depth + 1))
            stack.extend(subdirs)
        except OSError:
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pattern", default="sa_*",
                   help="glob for the per-tar folders (default sa_*)")
    args = p.parse_args()

    folders = sorted(d for d in args.image_dir.glob(args.pattern) if d.is_dir())
    if not folders:
        sys.exit(f"no {args.pattern} folders under {args.image_dir} — is it "
                 f"already flat? Then you do not need an index.")
    print(f"{len(folders)} folders under {args.image_dir}")

    rel_dirs: list[str] = []
    ids_all: list[np.ndarray] = []
    fidx_all: list[np.ndarray] = []
    t0 = time.time()

    for i, folder in enumerate(folders):
        jdir = find_jpeg_dir(folder)
        if jdir is None:
            print(f"  [skip] {folder.name}: no sa_*.jpg found")
            continue
        ids = np.fromiter(
            (int(m.group(1)) for m in
             (ID_RE.match(e.name) for e in os.scandir(jdir)) if m),
            dtype=np.int64)
        if ids.size == 0:
            continue
        rel_dirs.append(str(jdir.relative_to(args.image_dir)))
        ids_all.append(ids)
        fidx_all.append(np.full(ids.size, len(rel_dirs) - 1, dtype=np.int32))
        if i % 50 == 0 or i == len(folders) - 1:
            n = sum(a.size for a in ids_all)
            print(f"  [{i + 1}/{len(folders)}] {n:,} images, "
                  f"{time.time() - t0:.0f}s", flush=True)

    ids = np.concatenate(ids_all)
    fidx = np.concatenate(fidx_all)
    order = np.argsort(ids, kind="stable")
    ids, fidx = ids[order], fidx[order]
    print(f"\n{ids.size:,} images across {len(rel_dirs)} folders "
          f"in {time.time() - t0:.0f}s")

    # Are the folders contiguous, disjoint id blocks? Then 1000 rows suffice.
    lo = np.array([ids[fidx == k].min() for k in range(len(rel_dirs))])
    hi = np.array([ids[fidx == k].max() for k in range(len(rel_dirs))])
    o = np.argsort(lo)
    disjoint = bool(np.all(hi[o][:-1] < lo[o][1:]))

    payload = {"dirs": np.array(rel_dirs, dtype=object)}
    if disjoint:
        payload.update(mode=np.array("ranges"), lo=lo[o], hi=hi[o],
                       folder=np.arange(len(rel_dirs))[o])
        print("folders hold disjoint id ranges -> compact range index")
    else:
        payload.update(mode=np.array("full"), ids=ids, folder=fidx)
        print("id ranges overlap -> full index")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"\nuse it with:\n"
          f"  build_split.py --split sa1b_train --image-dir {args.image_dir} \\\n"
          f"      --image-index {args.out} --originals link ...")


if __name__ == "__main__":
    main()
