#!/usr/bin/env python3
"""Per-dataset acquisition specs for the SAM-3D-Body corpus.

One place that answers, for every split of ``facebook/sam-3d-body-dataset``:

  * how the images are obtained (a plain URL, a Hugging Face repo, a CLI that
    needs credentials, or a licence form),
  * where a parquet ``image`` string lands inside the extracted tree,
  * whether the frames need undistorting first,
  * how big it all is.

``image_relpath`` is re-exported from ``tools/parquet_to_npz.py`` rather than
re-implemented, so this module can never drift from the converter that the
existing ``data/sam3d_gt_*`` folders were built with.

Sizes are measured, not guessed:
  * COCO train2014 / MPII: HTTP Content-Length, checked 2026-08-31.
  * Harmony4D: the Hugging Face tree API for ``Jyun-Ting/Harmony4D``.
  * AIC: the OpenDataLab tarball already on this machine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.parquet_to_npz import image_relpath  # noqa: E402,F401  (re-export)

HF_ANNOTATION_REPO = "facebook/sam-3d-body-dataset"


@dataclass(frozen=True)
class Dataset:
    """Everything needed to go from a bare machine to usable frames."""

    name: str
    splits: tuple[str, ...]
    # How to get the images. "direct" = plain URLs; everything else needs a
    # human somewhere in the loop and fetch_images.py prints instructions.
    method: str                       # direct | hf_zips | opendatalab | gdrive | cli | links_file
    urls: tuple[str, ...] = ()        # for method == "direct"
    hf_repo: str = ""                 # for method == "hf_zips"
    # Where image_relpath() is rooted, relative to the extraction dir. The
    # archives usually already create this, so it is normally "".
    image_root_suffix: str = ""
    needs_undistort: bool = False
    approx_gb: float = 0.0            # download size
    # Many HPC sites mirror the big public datasets into a read-only shared
    # space. Set SHARED_DATASET_DIR and check there before downloading —
    # presence of a directory is not enough, the referenced SUBSET has to be
    # there in the layout image_relpath() expects.
    shared_mirror: str = ""
    notes: str = ""
    # Per-scene/shard archives, for streaming one at a time. Empty means the
    # dataset arrives as one blob.
    archives: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# The eight source datasets.
# --------------------------------------------------------------------------
HARMONY4D_TRAIN_ZIPS = (
    "01_hugging", "02_grappling", "03_grappling2", "04_sword_part1",
    "04_sword_part2", "05_sword2", "07_ballroom", "08_ballroom2",
    "09_karate", "10_karate2", "11_karate3", "12_mma", "13_mma2",
    "14_mma3", "15_mma4",
)
HARMONY4D_TEST_ZIPS = (
    "01_hugging", "03_grappling2", "05_sword2", "06_sword3",
    "08_ballroom2", "15_mma4", "16_mma5",
)

DATASETS: dict[str, Dataset] = {
    "coco": Dataset(
        name="coco",
        splits=("coco_train",),
        method="direct",
        urls=("http://images.cocodataset.org/zips/train2014.zip",),
        approx_gb=13.5,
        shared_mirror="COCO",
        notes="Only train2014 is referenced by coco_train — val2014/test2014 are "
              "a waste of 9 GB. Verified over every image string in shard 0.\n"
              "If your site mirrors COCO, confirm the mirror carries train2014 "
              "and not just 2017 before trusting it, then build with "
              "--originals link instead of downloading.",
    ),
    "mpii": Dataset(
        name="mpii",
        splits=("mpii_train",),
        method="direct",
        urls=("https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/"
              "mpii_human_pose_v1.tar.gz",),
        approx_gb=12.1,
        notes="Extracts to images/. Only 5,441 of ~25k images are referenced, "
              "so --originals copy reclaims ~11 GB.",
    ),
    "aic": Dataset(
        name="aic",
        splits=("aic_train",),
        method="opendatalab",
        approx_gb=16.0,
        notes="AI Challenger 2017. Only the train/images/ subtree is needed: "
              "210,000 jpgs, 16 GB. The full OpenDataLab archive is 36 GB but "
              "the rest is val, eval scripts and duplicate annotation copies.\n"
              "The original challenger.ai host is dead. If you already hold a "
              "copy, rsync train/ straight in - that is the reliable route.\n"
              "Otherwise OpenDataLab, in an ISOLATED venv: openxlab pins old "
              "requests/urllib3/tqdm and will break huggingface_hub if it "
              "shares an environment with fetch_annotations.py.\n"
              "    python -m venv ~/venvs/odl && ~/venvs/odl/bin/pip install opendatalab\n"
              "    ~/venvs/odl/bin/odl login\n"
              "    ~/venvs/odl/bin/odl get AI_Challenger -d $DEST\n"
              "The dataset name is bare - `odl get OpenDataLab/AI_Challenger` "
              "fails with `No dataset: OpenDataLab`; the org/name form is an "
              "openxlab convention, not an odl one.",
    ),
    "3dpw": Dataset(
        name="3dpw",
        splits=("3dpw_train",),
        method="cli",
        approx_gb=4.6,
        notes="Accept the licence at "
              "https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html, then "
              "download imageFiles.zip and unzip so that sequence folders "
              "(courtyard_arguing_00/, ...) sit directly under the image dir.",
    ),
    "harmony4d": Dataset(
        name="harmony4d",
        splits=("harmony4d_train", "harmony4d_test"),
        method="hf_zips",
        hf_repo="Jyun-Ting/Harmony4D",
        needs_undistort=True,
        approx_gb=351.7,
        archives=tuple(f"train/{s}.zip" for s in HARMONY4D_TRAIN_ZIPS)
        + tuple(f"test/{s}.zip" for s in HARMONY4D_TEST_ZIPS),
        notes="287 GB train + 65 GB test of 4K frames, ungated on Hugging Face. "
              "Far past any sane disk budget as a whole, so run it through "
              "stream_archives.py, which does one scene zip at a time and "
              "deletes it after the crops are written. Frames MUST be "
              "undistorted first — see undistort.md for the three known "
              "blockers in Meta's script.",
    ),
    "egohumans": Dataset(
        name="egohumans",
        splits=("egohumans_train",),
        method="gdrive",
        needs_undistort=True,
        approx_gb=107.0,
        notes="Google Drive folder 1JD963urzuzV_R_6FOVOtlx8UupwUuknR, one "
              ".tar.gz per TAKE under <sequence>/<take>.tar.gz.\n"
              "    pip install gdown\n"
              "    gdown --folder 1JD963urzuzV_R_6FOVOtlx8UupwUuknR -O $DEST\n"
              "    tar xzf 001_tagging.tar.gz --strip-components=8   # NOT optional\n"
              "The dataset advertises 550 GB but egohumans_train references "
              "only 89 takes across 7 sequences (tagging, lego, fencing, "
              "basketball, volleyball, badminton, tennis) = 272k persons over "
              "268k 4K frames, ~107 GB of referenced frames. Get the exact "
              "shopping list with:\n"
              "    probe_split.py --annotation-dir $ANN/egohumans_train --list-groups 2\n"
              "Only the exo/ cameras are referenced; ego/, colmap/ and "
              "processed_data/ in each tarball are dead weight. Undistort with "
              "sam-3d-body/data/scripts/egohumans/undistort_images.py — see "
              "undistort.md.",
    ),
    "egoexo4d": Dataset(
        name="egoexo4d",
        splits=("egoexo4d_physical_train", "egoexo4d_physical_test",
                "egoexo4d_procedure_train", "egoexo4d_procedure_test"),
        method="cli",
        approx_gb=0.0,
        notes="Needs the signed Ego-Exo4D licence (you have one pending). Then:\n"
              "    pip install ego4d && egoexo -o $DEST --parts sam_3d_body\n"
              "The sam_3d_body part is already undistorted. 610 train shards "
              "~= 1.1M person crops, and it is the one split in this corpus "
              "where hands are routinely large enough to learn from.",
    ),
    "sa1b": Dataset(
        name="sa1b",
        splits=("sa1b_train",),
        method="links_file",
        approx_gb=0.0,
        shared_mirror="SegmentAnything_1B",
        notes="1,850 annotation shards ~= 3.4M person crops — bigger than every "
              "other split combined. Accept the terms at "
              "https://ai.meta.com/datasets/segment-anything-downloads/ to get "
              "links.txt (one `sa_NNNNNN.tar <signed-url>` per line), then feed "
              "it to fetch_images.py --links-file.\n"
              "MEASURED, and it is bad news: the referenced images are spread "
              "UNIFORMLY over all 1000 tars — 994 of 1000 id-blocks are hit by "
              "just 3% of the annotations — and only 11.4% of SA-1B is "
              "referenced at all. So there is no dense subset: you pull a whole "
              "~10 GB tar to get ~1,270 useful frames (~3,400 crops), i.e. "
              "~2.3 MB of download per crop against AIC's 0.13 MB. Worse, "
              "SA-1B is the ONE split whose keypoints carry real visibility "
              "flags: a mean of 44 of 70 are literal (0,0) placeholders and "
              "only 0.3% of hands are fully visible.\n"
              "Any prefix of the tar list is a valid subset — each tar is "
              "self-contained and the builder skips rows whose image is absent "
              "— so stream it and stop whenever.\n"
              "If your site already mirrors SA-1B the download cost is zero and "
              "it becomes the largest body corpus here by far (3.4M persons, 4x "
              "everything else combined). A mirror that unpacked the 1000 tars "
              "in place needs index_sa1b.py + build_split.py --image-index, "
              "because the parquet asks for a flat sa_<id>.jpg. Its 3D "
              "keypoints and MHR params are 100% complete; only the 2D array is "
              "sparse (body 100%, feet 97%, hands 8%) — see the label trap in "
              "README.md. Near-useless for hand CROPS either way: 0.3% of hands "
              "have all 21 2D keypoints.",
    ),
}

# Which parquet split belongs to which source dataset.
SPLIT_TO_DATASET = {s: d.name for d in DATASETS.values() for s in d.splits}


def dataset_for_split(split: str) -> Dataset:
    try:
        return DATASETS[SPLIT_TO_DATASET[split]]
    except KeyError:
        raise SystemExit(
            f"unknown split {split!r}. Known splits:\n  "
            + "\n  ".join(sorted(SPLIT_TO_DATASET))
        ) from None


def output_dir_for_split(split: str, data_root: Path) -> Path:
    """`coco_train` -> `<data_root>/sam3d_gt_coco`.

    Train and test of the same dataset deliberately land in the same folder:
    that is the convention the existing data/ tree already uses, and the
    training script's dataset discovery walks one level of sub-folders.
    """
    return data_root / f"sam3d_gt_{SPLIT_TO_DATASET[split]}"


if __name__ == "__main__":
    print(f"{'dataset':12s} {'method':12s} {'GB':>7s}  splits")
    for d in DATASETS.values():
        gb = f"{d.approx_gb:.1f}" if d.approx_gb else "?"
        print(f"{d.name:12s} {d.method:12s} {gb:>7s}  {', '.join(d.splits)}")
