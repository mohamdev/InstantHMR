#!/usr/bin/env python3
"""Fetch the benchmark data.

COCO val2017 is a direct public download and is handled fully automatically.
3DPW requires accepting a licence on the MPI page, so this script cannot fetch
it for you — it verifies what you already have and tells you what is missing.

    python benchmark/download.py --coco
    python benchmark/download.py --check-3dpw --sequence-dir ... --image-root ...
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

COCO_FILES = [
    ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
     "annotations_trainval2017.zip", "annotations/person_keypoints_val2017.json"),
    ("http://images.cocodataset.org/zips/val2017.zip",
     "val2017.zip", "val2017"),
]

THREEDPW_HELP = """
3DPW must be downloaded from the official page (free, licence form):

    https://virtualhumans.mpi-inf.mpg.de/3DPW/licence.html

Fill the form; the confirmation mail carries the download links. You need:

    sequenceFiles.zip   ~ 30 MB   <-- the ground truth (required)
    imageFiles.zip      ~ 4.6 GB  <-- the RGB frames

Unzip so the layout is:

    <3dpw_root>/sequenceFiles/{train,validation,test}/*.pkl
    <3dpw_root>/imageFiles/<sequence>/image_XXXXX.jpg
"""


def _progress(block_num, block_size, total):
    if total <= 0:
        return
    done = min(block_num * block_size, total)
    pct = 100.0 * done / total
    sys.stdout.write(f"\r    {pct:5.1f}%  ({done/1e6:.0f} / {total/1e6:.0f} MB)")
    sys.stdout.flush()


def download_coco(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for url, zip_name, marker in COCO_FILES:
        if (root / marker).exists():
            print(f"[coco] {marker} already present — skipping")
            continue
        zip_path = root / zip_name
        if not zip_path.exists():
            print(f"[coco] downloading {url}")
            urllib.request.urlretrieve(url, zip_path, _progress)
            print()
        print(f"[coco] extracting {zip_name}")
        with zipfile.ZipFile(zip_path) as z:
            # The annotations zip carries train2017 too; we only need val.
            members = [m for m in z.namelist()
                       if "train2017" not in m or "val" in m]
            z.extractall(root, members=members)
        zip_path.unlink()
        print(f"[coco] ok -> {root / marker}")
    print(f"\n[coco] ready. Use --coco-root {root}")


def check_3dpw(sequence_dir: Path | None, image_root: Path | None) -> bool:
    ok = True
    if sequence_dir is None or not sequence_dir.is_dir():
        print(f"[3dpw] MISSING sequenceFiles: {sequence_dir}")
        ok = False
    else:
        for split, expected in (("test", 24), ("validation", 12), ("train", 24)):
            pkls = list((sequence_dir / split).glob("*.pkl"))
            flag = "ok " if pkls else "MISSING"
            print(f"[3dpw] {flag} sequenceFiles/{split:<11s} "
                  f"{len(pkls)} pkl (official: {expected})")
            if split == "test" and not pkls:
                ok = False
    if image_root is None or not image_root.is_dir():
        print(f"[3dpw] MISSING imageFiles: {image_root}")
        ok = False
    else:
        seqs = [d for d in image_root.iterdir() if d.is_dir()]
        print(f"[3dpw] ok  imageFiles              {len(seqs)} sequences "
              f"(official: 60)")
    if not ok:
        print(THREEDPW_HELP)
    else:
        print("\n[3dpw] ready.")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", action="store_true", help="download COCO val2017")
    p.add_argument("--coco-root", default="benchmark/data/coco")
    p.add_argument("--check-3dpw", action="store_true")
    p.add_argument("--sequence-dir", default=None)
    p.add_argument("--image-root", default=None)
    a = p.parse_args()

    if a.coco:
        download_coco(Path(a.coco_root))
    if a.check_3dpw:
        check_3dpw(
            Path(a.sequence_dir) if a.sequence_dir else None,
            Path(a.image_root) if a.image_root else None,
        )
    if not (a.coco or a.check_3dpw):
        p.print_help()


if __name__ == "__main__":
    main()
