#!/usr/bin/env python3
"""Download parquet annotation shards from facebook/sam-3d-body-dataset.

The gated HF repo needs an accepted licence and a token (`hf auth login`, or
HF_TOKEN in the environment). Downloads are resumable and skip shards already
on disk, so re-running after a killed job is free.

    # everything for one split
    python datasets_pipeline/fetch_annotations.py --splits coco_train --out $ANN_DIR

    # a slice, for a cheap first pass on a huge split
    python datasets_pipeline/fetch_annotations.py --splits sa1b_train --out $ANN_DIR \
        --max-shards 50

    # just tell me what is available and how big it is
    python datasets_pipeline/fetch_annotations.py --list
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import HF_ANNOTATION_REPO, SPLIT_TO_DATASET  # noqa: E402


def shard_index(api: HfApi) -> dict[str, list[str]]:
    """split -> sorted list of repo-relative parquet paths."""
    out: dict[str, list[str]] = {}
    for f in api.list_repo_files(HF_ANNOTATION_REPO, repo_type="dataset"):
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        if len(parts) < 3:
            continue
        out.setdefault(parts[1], []).append(f)
    return {k: sorted(v) for k, v in sorted(out.items())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", help="e.g. coco_train harmony4d_train")
    p.add_argument("--out", type=Path, help="destination root; one folder per split")
    p.add_argument("--max-shards", type=int, default=None,
                   help="take only the first N shards of each split")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--list", action="store_true", help="print available splits and exit")
    args = p.parse_args()

    api = HfApi()
    index = shard_index(api)

    if args.list:
        print(f"{'split':26s} {'shards':>7s}  ~persons")
        for split, files in index.items():
            # ~1,750 rows/shard, measured across coco/aic/sa1b/egoexo4d/egohumans.
            print(f"{split:26s} {len(files):7d}  {len(files) * 1750:>9,d}")
        return

    if not args.splits or not args.out:
        p.error("--splits and --out are required unless --list is given")

    unknown = [s for s in args.splits if s not in index]
    if unknown:
        p.error(f"unknown split(s): {unknown}. Run --list for the full set.")
    for s in args.splits:
        if s not in SPLIT_TO_DATASET:
            print(f"[warn] {s} has no entry in registry.py; build_split.py will "
                  f"not know where its images live.", file=sys.stderr)

    for split in args.splits:
        files = index[split]
        if args.max_shards:
            files = files[: args.max_shards]
        dest = args.out / split
        dest.mkdir(parents=True, exist_ok=True)
        todo = [f for f in files if not (dest / Path(f).name).exists()]
        print(f"{split}: {len(files)} shards, {len(todo)} to download -> {dest}")
        if not todo:
            continue

        def grab(repo_file: str) -> None:
            hf_hub_download(HF_ANNOTATION_REPO, repo_file, repo_type="dataset",
                            local_dir=str(dest.parent / ".hf_staging"))
            src = dest.parent / ".hf_staging" / repo_file
            src.replace(dest / src.name)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(grab, f): f for f in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), unit="shard"):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    tqdm.write(f"[fail] {futs[fut]}: {e}")

    staging = args.out / ".hf_staging"
    if staging.exists():
        import shutil
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
