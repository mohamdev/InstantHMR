#!/usr/bin/env python3
"""Download and extract the source frames for one SAM-3D-Body dataset.

Every dataset that CAN be fetched with a plain URL is fetched with a plain URL
(`wget -c`, so a killed job resumes). The four that cannot — AIC, 3DPW,
EgoExo4D, SA-1B — need a login or a signed licence, and this script prints the
exact commands instead of pretending.

    python datasets_pipeline/fetch_images.py --dataset coco      --dest $IMG_ROOT/coco
    python datasets_pipeline/fetch_images.py --dataset mpii      --dest $IMG_ROOT/mpii
    python datasets_pipeline/fetch_images.py --dataset harmony4d --dest $IMG_ROOT/harmony4d_raw \
        --archives train/09_karate.zip --no-keep-archive
    python datasets_pipeline/fetch_images.py --dataset sa1b --dest $IMG_ROOT/sa1b \
        --links-file ~/sa1b_links.txt --max-archives 20

`--dry-run` prints the plan and the byte counts without downloading anything.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DATASETS  # noqa: E402

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def wget(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / url.split("/")[-1].split("?")[0]
    run(["wget", "-c", "-O", str(out), url])
    return out


def extract(archive: Path, into: Path, strip_top: bool = False) -> None:
    into.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} -> {into}", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if archive.name.endswith((".tar.gz", ".tgz")) else "r"
        with tarfile.open(archive, mode) as t:
            t.extractall(into, filter="data")
    else:
        raise SystemExit(f"don't know how to extract {archive.name}")


def manual(ds_name: str, notes: str) -> None:
    print("=" * 70)
    print(f"{ds_name}: needs a credential or a signed licence — cannot wget it.")
    print("=" * 70)
    print(notes)
    print()
    print("Once the frames are on disk, point build_split.py --image-dir at the "
          "root that the parquet `image` paths are relative to and carry on.")
    sys.exit(2)


def fetch_direct(ds, dest: Path, keep: bool, dry: bool) -> None:
    for url in ds.urls:
        if dry:
            print(f"[dry-run] wget {url}")
            continue
        arc = wget(url, dest)
        extract(arc, dest)
        if not keep:
            arc.unlink()


def fetch_hf_zips(ds, dest: Path, wanted: list[str] | None,
                  max_archives: int | None, keep: bool, dry: bool) -> None:
    archives = list(wanted or ds.archives)
    unknown = [a for a in archives if a not in ds.archives]
    if unknown:
        raise SystemExit(f"unknown archive(s) {unknown}\nknown:\n  "
                         + "\n  ".join(ds.archives))
    if max_archives:
        archives = archives[:max_archives]
    for rel in archives:
        url = HF_RESOLVE.format(repo=ds.hf_repo, path=rel)
        if dry:
            print(f"[dry-run] wget {url}")
            continue
        # Harmony4D zips already carry the train/<scene>/ prefix internally.
        arc = wget(url, dest / "_archives")
        extract(arc, dest)
        if not keep:
            arc.unlink()


def fetch_links_file(ds, dest: Path, links_file: Path,
                     max_archives: int | None, keep: bool, dry: bool) -> None:
    """SA-1B: Meta hands you a two-column `name<TAB>signed-url` file."""
    lines = [ln.strip() for ln in links_file.read_text().splitlines() if ln.strip()]
    entries = []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 2 and parts[1].startswith("http"):
            entries.append((parts[0], parts[1]))
    if not entries:
        raise SystemExit(f"no `name url` pairs found in {links_file}")
    if max_archives:
        entries = entries[:max_archives]
    print(f"{len(entries)} SA-1B shards to fetch")
    for name, url in entries:
        if dry:
            print(f"[dry-run] {name}")
            continue
        arc = dest / "_archives" / name
        arc.parent.mkdir(parents=True, exist_ok=True)
        run(["wget", "-c", "-O", str(arc), url])
        # Each tar holds sa_*.jpg plus per-image json masks we do not need.
        with tarfile.open(arc) as t:
            members = [m for m in t.getmembers() if m.name.endswith(".jpg")]
            t.extractall(dest, members=members, filter="data")
        if not keep:
            arc.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--archives", nargs="*", default=None,
                   help="subset of archives (hf_zips only); default is all")
    p.add_argument("--max-archives", type=int, default=None)
    p.add_argument("--links-file", type=Path, default=None, help="SA-1B links.txt")
    p.add_argument("--keep-archive", action=argparse.BooleanOptionalAction,
                   default=False, help="keep the downloaded zip/tar after extracting")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-archives", action="store_true")
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    if args.list_archives:
        for a in ds.archives or ("(single blob)",):
            print(a)
        return

    print(f"{ds.name}: method={ds.method}  ~{ds.approx_gb or '?'} GB  "
          f"undistort={'YES' if ds.needs_undistort else 'no'}")
    if ds.notes:
        print("  " + ds.notes.replace("\n", "\n  "))
    print()

    if ds.method == "direct":
        fetch_direct(ds, args.dest, args.keep_archive, args.dry_run)
    elif ds.method == "hf_zips":
        fetch_hf_zips(ds, args.dest, args.archives, args.max_archives,
                      args.keep_archive, args.dry_run)
    elif ds.method == "links_file":
        if not args.links_file:
            manual(ds.name, ds.notes)
        fetch_links_file(ds, args.dest, args.links_file, args.max_archives,
                         args.keep_archive, args.dry_run)
    else:
        manual(ds.name, ds.notes)

    if ds.needs_undistort:
        print()
        print("!! These frames are DISTORTED. Run the undistortion step before "
              "build_split.py — see datasets_pipeline/undistort.md.")
    free = (shutil.disk_usage(args.dest).free / 1e9
            if args.dest.exists() else shutil.disk_usage(".").free / 1e9)
    print(f"done -> {args.dest}  ({free:.0f} GB free)")


if __name__ == "__main__":
    main()
