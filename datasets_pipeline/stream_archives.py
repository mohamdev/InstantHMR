#!/usr/bin/env python3
"""Build a dataset that does not fit on disk, one archive at a time.

Harmony4D is 352 GB of 4K zips and SA-1B is ~11 TB; neither can be staged
whole on a 172 GB volume. The loop below never holds more than one archive:

    download -> extract -> [undistort] -> build crops -> delete everything raw

Because build_split.py is resume-safe and silently skips rows whose image is
absent, any prefix of the archive list yields a valid, self-consistent dataset.
Stop whenever you have enough, or when the disk says so.

    # Harmony4D, scene by scene, keeping nothing but the crops
    python datasets_pipeline/stream_archives.py --dataset harmony4d \
        --split harmony4d_train --annotation-dir $ANN_DIR/harmony4d_train \
        --work-dir /scratch/h4d --data-root data

    # SA-1B, 20 tars at a time, stop after 200
    python datasets_pipeline/stream_archives.py --dataset sa1b \
        --split sa1b_train --annotation-dir $ANN_DIR/sa1b_train \
        --work-dir /scratch/sa1b --data-root data \
        --links-file ~/sa1b_links.txt --max-archives 200 --build-every 20

Anything passed after `--` goes straight to build_split.py, e.g.
`-- --crop-format jpg --hand-min-px 48 --originals none`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from registry import DATASETS, dataset_for_split  # noqa: E402


def sh(cmd: list, cwd: Path | None = None) -> None:
    print("\n+ " + " ".join(str(c) for c in cmd)
          + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    subprocess.run([str(c) for c in cmd], check=True,
                   cwd=str(cwd) if cwd else None)


def free_gb(p: Path) -> float:
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def undistort_harmony4d(raw: Path, undist: Path, scene_rel: str,
                        sam3d_repo: Path, python_bin: str) -> None:
    """Run Meta's undistortion over every take of one scene.

    `--seqs` wants the TAKE folder (train/09_karate/007_karate), not the scene:
    the script appends `exo/` to whatever it is given. See undistort.md.
    """
    scene_dir = raw / scene_rel
    takes = sorted(d.name for d in scene_dir.iterdir() if d.is_dir())
    if not takes:
        raise SystemExit(f"no takes under {scene_dir}")
    seqs = ",".join(f"{scene_rel}/{t}" for t in takes)
    scripts = sam3d_repo / "data" / "scripts"
    if not (scripts / "lib").is_dir():
        raise SystemExit(
            f"{scripts}/lib is missing — Meta's undistort script imports it but "
            f"the clone does not ship it. See datasets_pipeline/undistort.md.")
    sh([python_bin, "harmony4d/undistort_images.py",
        "--src_dir", raw.resolve(), "--dst_dir", undist.resolve(),
        "--seqs", seqs], cwd=scripts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--split", required=True)
    p.add_argument("--annotation-dir", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path,
                   help="scratch space for one archive at a time")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--archives", nargs="*", default=None)
    p.add_argument("--max-archives", type=int, default=None)
    p.add_argument("--build-every", type=int, default=1,
                   help="build after this many archives (SA-1B has flat, "
                        "unpredictable filenames so batching amortises the "
                        "parquet scan)")
    p.add_argument("--links-file", type=Path, default=None)
    p.add_argument("--min-free-gb", type=float, default=25.0,
                   help="abort before an archive that would leave less than this")
    p.add_argument("--keep-frames", action="store_true",
                   help="do not delete the extracted frames after building")
    p.add_argument("--sam3d-repo", type=Path,
                   default=Path("/home/madjel/Projects/gitpackages/sam3-biomechanics/sam-3d-body"),
                   help="for the undistortion scripts")
    p.add_argument("--undistort-python", default=sys.executable,
                   help="interpreter with pycolmap available (the "
                        "sam_3d_body_data env, not the training env)")
    p.add_argument("build_args", nargs="*",
                   help="extra args for build_split.py, after --")
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    if dataset_for_split(args.split).name != ds.name:
        raise SystemExit(f"split {args.split} does not belong to {ds.name}")

    raw = args.work_dir / "raw"
    undist = args.work_dir / "undistorted"
    image_dir = undist if ds.needs_undistort else raw

    if ds.method == "hf_zips":
        archives = list(args.archives or ds.archives)
    elif ds.method == "links_file":
        if not args.links_file:
            raise SystemExit("--links-file is required for sa1b")
        archives = [ln.split()[0] for ln in args.links_file.read_text().splitlines()
                    if len(ln.split()) == 2]
    else:
        raise SystemExit(
            f"{ds.name} arrives as one blob, not per-archive — fetch it with "
            f"fetch_images.py and run build_split.py directly.")
    if args.max_archives:
        archives = archives[: args.max_archives]

    print(f"{ds.name}: {len(archives)} archive(s), work dir {args.work_dir}")
    pending: list[str] = []

    for i, arc in enumerate(archives, 1):
        if free_gb(args.work_dir) < args.min_free_gb:
            print(f"\nSTOP: only {free_gb(args.work_dir):.0f} GB free, below "
                  f"--min-free-gb {args.min_free_gb}. "
                  f"{len(archives) - i + 1} archive(s) left; re-run to resume.")
            break

        print(f"\n===== [{i}/{len(archives)}] {arc} "
              f"({free_gb(args.work_dir):.0f} GB free) =====")
        fetch = [sys.executable, HERE / "fetch_images.py",
                 "--dataset", ds.name, "--dest", raw, "--no-keep-archive"]
        if ds.method == "hf_zips":
            fetch += ["--archives", arc]
        else:
            fetch += ["--links-file", args.links_file, "--max-archives", "1"]
        sh(fetch)

        scene_rel = arc[:-4] if arc.endswith(".zip") else None
        if ds.needs_undistort and scene_rel:
            undistort_harmony4d(raw, undist, scene_rel, args.sam3d_repo,
                                args.undistort_python)

        pending.append(arc)
        if len(pending) < args.build_every and i < len(archives):
            continue

        build = [sys.executable, HERE / "build_split.py",
                 "--split", args.split,
                 "--annotation-dir", args.annotation_dir,
                 "--image-dir", image_dir,
                 "--data-root", args.data_root]
        if ds.method == "hf_zips" and len(pending) == 1 and scene_rel:
            build += ["--image-prefix", f"{scene_rel}/"]
        build += args.build_args
        sh(build)
        pending.clear()

        if not args.keep_frames:
            for d in (raw, undist):
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)

    print(f"\nstreaming done; {free_gb(args.work_dir):.0f} GB free")


if __name__ == "__main__":
    main()
