# datasets_pipeline

Rebuild any split of the SAM-3D-Body corpus from scratch, keeping the original
frames this time, into the layout the training code already understands.

```
data/sam3d_gt_<dataset>/
├── original_images/   referenced source frames, native relative paths
├── annotations/       <name>.npz
├── body_crops/        <name>.png              224 square person crop
├── hands_crops/       <name>_l.png  <name>_r.png
└── images -> body_crops                       compat symlink
```

`body_crops` is **byte-identical** to what `tools/parquet_to_npz.py` writes into
`images/` — verified md5-for-md5 on 120 AIC crops against the existing
`data/sam3d_gt_aic/`. `build_split.py` imports that module's crop and annotation
builders rather than re-implementing them, so the two can never drift. The
`images` symlink means `SAM3DStudentDataset` finds these folders with **no code
change**; that path is tested, not assumed.

## The scripts

| script | does |
|---|---|
| `fetch_annotations.py` | parquet shards from the gated HF repo |
| `probe_split.py` | **is this split worth its download?** — from annotations alone |
| `fetch_images.py` | source frames, per dataset |
| `build_split.py` | parquet + frames → the folder above |
| `verify_split.py` | integrity, reprojection, and how much hand signal a split carries |

Plus `plan_disk.py` (cost a split before downloading it), `stream_archives.py`
(build a dataset larger than your disk), `registry.py` (every URL and gotcha in
one place), and `undistort.md`.

Annotations are ~1 GB per split; frames are tens to thousands of GB. So the
order is always: fetch annotations → `probe_split.py` → decide → fetch frames.

## Quick start

```bash
export ANN_DIR=/scratch/sam3d_ann
export IMG_ROOT=/scratch/sam3d_img

python datasets_pipeline/fetch_annotations.py --list
python datasets_pipeline/fetch_annotations.py --splits coco_train --out $ANN_DIR
python datasets_pipeline/plan_disk.py --annotation-dir $ANN_DIR/coco_train

python datasets_pipeline/fetch_images.py --dataset coco --dest $IMG_ROOT/coco
python datasets_pipeline/build_split.py \
    --split coco_train \
    --annotation-dir $ANN_DIR/coco_train \
    --image-dir $IMG_ROOT/coco \
    --data-root data --validate

python datasets_pipeline/verify_split.py --dir data/sam3d_gt_coco
```

Everything is resume-safe: re-running skips crops already on disk.

## What each dataset costs, and how to get it

| dataset | images | route | undistort |
|---|---|---|---|
| coco | 13.5 GB | `wget` train2014.zip — **only train2014 is referenced** | no |
| mpii | 12.1 GB | `wget` mpii_human_pose_v1.tar.gz | no |
| aic | 36.4 GB | OpenDataLab CLI (login) — where this corpus came from before | no |
| 3dpw | 4.6 GB | licence form → imageFiles.zip | no |
| harmony4d | 352 GB | HF `Jyun-Ting/Harmony4D`, ungated, per-scene zips | **yes** |
| egohumans | 107 GB referenced (550 GB advertised) | Google Drive, per-take tar.gz → `gdown --folder` | **yes** |
| egoexo4d | ~8 GB (procedure) | `egoexo -o $DEST --parts sam_3d_body` (signed licence) | no |
| sa1b | ~11 TB | terms → `links.txt` → `--links-file` | no |

`fetch_images.py` handles the first two and harmony4d directly. For the rest it
prints the exact commands and exits 2 rather than pretending.

Measured output sizes, from `plan_disk.py` over the shards already on this
machine (PNG crops, 20 % hand yield):

```
split                         persons     frames     body    hands  originals     TOTAL
3dpw_train                     22,405     17,410     1.7G     0.7G       4.4G      6.7G
aic_train                     287,391    169,923    21.6G     8.6G      11.9G     42.1G
harmony4d_train               496,036    248,018    37.2G    14.9G     347.2G    399.3G
mpii_train                      7,135      5,441     0.5G     0.2G       1.1G      1.9G
```

Two things fall out of that table.

**`original_images` is the whole cost.** Everything except Harmony4D fits
comfortably: coco + mpii + aic + 3dpw is roughly 60 GB of originals and 40 GB of
crops. Harmony4D alone is 347 GB of 4K frames and does not fit in 172 GB at any
setting. Build it with `--originals none` — `body_crops` and `hands_crops` are
self-contained, and the npz records `image_relpath`, so the originals for any
subset can be re-fetched later.

**`--crop-format jpg` cuts the crops 3×** (75 KB → 24 KB per 224 crop) and
decodes ~4× faster, which also speeds up training. It is not the default only
because PNG keeps byte-identity with the existing `data/` tree. If you are
rebuilding from scratch anyway, use jpg.

## Datasets you fetch by hand (AIC, 3DPW)

Both are one archive behind a login, so you download and `rsync` them yourself.
The only thing that matters afterwards is **what `--image-dir` points at**: it
is the root that the parquet `image` strings are relative to, and that root
differs per dataset because `image_relpath()` prepends a different prefix.

**AIC** — `image` is a bare `<hash>.jpg`, and the resolver prepends
`train/images/`. So unpack until this exists:

```
$AIC_IMG_DIR/train/images/0000252aea98840a550dac9a78c476ecb9f47ffa.jpg
```

which is exactly the layout already on this machine at
`~/sam3d_data/images/aic`. Then:

```bash
python datasets_pipeline/build_split.py --split aic_train \
    --annotation-dir $ANN_DIR/aic_train --image-dir $AIC_IMG_DIR \
    --data-root data --validate
```

`imageFiles.zip`-style archives often unpack one level deeper than you expect;
if the build reports `missing_image` for everything, the root is wrong by one
directory. A single `--max-samples 20` run tells you in seconds.

**3DPW** — `image` is `<sequence>/image_00386.jpg` and the resolver prepends
nothing, so sequence folders sit directly under the root:

```
$THREEDPW_IMG_DIR/outdoors_climbing_01/image_00386.jpg
```

You need `imageFiles.zip` only. `sequenceFiles.zip` holds the original SMPL
ground truth, which this pipeline never reads — the MHR fits come from the
parquet. Keep it for the 3DPW *benchmark* (`benchmark/eval_3dpw.py`), not for
training data.

Same command, `--split 3dpw_train`. Both are small enough for
`--originals copy` (4.4 GB and 11.9 GB of referenced frames).

## EgoHumans

Google Drive, one `.tar.gz` per take, and the untar needs
`--strip-components=8` or you get eight nested empty parents:

```bash
pip install gdown
gdown --folder 1JD963urzuzV_R_6FOVOtlx8UupwUuknR -O $EGOHUMANS_DATA_DIR
cd $EGOHUMANS_DATA_DIR/01_tagging && tar xzf 001_tagging.tar.gz --strip-components=8
```

The dataset advertises 550 GB, but `egohumans_train` references only **89 takes
across 7 sequences** — 272k persons over 268k 4K frames, ~107 GB of referenced
frames. Get the exact shopping list before downloading anything:

```bash
python datasets_pipeline/probe_split.py \
    --annotation-dir $ANN_DIR/egohumans_train --list-groups 2
```

Only the `exo/` cameras are referenced; `ego/`, `colmap/` and `processed_data/`
inside each tarball are dead weight you can delete after extraction. Frames
must be undistorted (`undistort.md`), then build per take with
`--image-prefix 01_tagging/001_tagging/`, or let `stream_archives.py` do it.

Worth knowing before you spend the 107 GB: it is good body data (multi-view,
272k persons) but useless for hands — sports at distance, so only 21.5% of
hands clear 64 px and the finger articulation is 6.3 mm, the lowest in the
corpus.

## SA-1B

Accept the terms at <https://ai.meta.com/datasets/segment-anything-downloads/>
to get `links.txt` (one `sa_NNNNNN.tar <signed-url>` per line), then:

```bash
python datasets_pipeline/stream_archives.py --dataset sa1b \
    --split sa1b_train --annotation-dir $ANN_DIR/sa1b_train \
    --work-dir /scratch/sa1b --data-root data \
    --links-file ~/links.txt --max-archives 50 --build-every 10 \
    -- --originals none --crop-format jpg
```

Any prefix of the tar list is a valid dataset, so you can stop whenever. But
the measured economics are poor, and I would not start here:

- References are spread **uniformly over all 1000 tars** — 994 of 1000 id
  blocks are hit by just 3% of the annotations — so there is no dense subset to
  cherry-pick.
- Only **11.4%** of SA-1B is referenced. Each ~10 GB tar yields ~1,270 useful
  frames ≈ 3,400 crops, i.e. **~2.3 MB of download per crop** against AIC's
  0.13 MB. The full 3.4M crops cost ~11 TB.
- It is the one split with real visibility flags: a mean of **44 of 70**
  keypoints are literal `(0, 0)` placeholders, and only **0.3%** of hands are
  fully visible.

That last point is a trap for any converter. `tools/parquet_to_npz.py` drops the
visibility column (`joints_2d = kp2[:, :2]`), so a naive SA-1B build writes
those placeholders as if they were observations at the image corner — poisoning
the 2D loss and blowing every hand box out to the frame edge. `build_split.py`
stores `joints_2d_vis` / `joints_3d_conf` and refuses to cut a hand crop unless
all 21 of its keypoints are visible. Every other split is 100% visible, so this
only bites on SA-1B.

## Datasets that do not fit: `stream_archives.py`

Never holds more than one archive: download → extract → undistort → build →
delete. Any prefix of the archive list is a valid dataset, so you can stop when
the disk or the clock says so and resume later.

```bash
python datasets_pipeline/stream_archives.py --dataset harmony4d \
    --split harmony4d_train --annotation-dir $ANN_DIR/harmony4d_train \
    --work-dir /scratch/h4d --data-root data \
    --undistort-python ~/anaconda3/envs/sam_3d_body_data/bin/python \
    -- --originals none --crop-format jpg

python datasets_pipeline/stream_archives.py --dataset sa1b \
    --split sa1b_train --annotation-dir $ANN_DIR/sa1b_train \
    --work-dir /scratch/sa1b --data-root data \
    --links-file ~/sa1b_links.txt --max-archives 200 --build-every 20 \
    -- --originals none
```

`--min-free-gb` (default 25) aborts before an archive that would fill the disk.
Read `undistort.md` before the first Harmony4D run — Meta's script has three
blockers that a fresh clone hits.

## Hand crops

The hand box is free: `joints_2d` is already in full-frame pixel coordinates, so
the 21 keypoints of each hand give a tight box with no extra annotation.

- `--hand-min-px 64` (default) is SAM3D's own validity gate, applied to the raw
  keypoint span. On AIC this keeps ~17 % of hands, at a median 145 source pixels
  upsampled 1.5× to 224.
- `--hand-expand 2.0` stores margin around the hand so you can jitter the box
  *inside* the stored crop at train time. That is what makes a model tolerant of
  a real detector's boxes, and it is why cropping dynamically from originals
  buys nothing.
- Crops are stored **unflipped**. SAM3D trains one right-hand model and mirrors
  the left; flipping a loaded crop is a free numpy slice, so baking the mirror
  into disk would only remove a choice.
- `hand_valid_l` / `hand_valid_r` in the npz say whether a crop exists;
  `hand_box_*` (tight) and `hand_crop_*` (the square actually written) let you
  re-derive any tighter crop without touching the originals.

## Which split to spend money on

`probe_split.py` measured every split from its annotations. *Articulation* is
how far the labels' finger poses sit from ONE constant hand pose, in mm at the
fingertips — an upper bound on what a hand branch could learn from them.
*≥96 px* is the share of hands whose crop would be at least 96 source pixels.

| split | persons | articulation | hands ≥96 px | download |
|---|---|---|---|---|
| **egoexo4d_procedure** | ~350k | **16.1 mm** | **85.5%** | licence + CLI |
| egoexo4d_physical | ~700k | 14.5 mm | 19.3% | licence + CLI |
| harmony4d | 496k | 12.6 mm | 3.4% | 352 GB |
| sa1b | 3.4M | 8.5 mm | 0.3% visible at all | ~11 TB |
| 3dpw | 22k | 7.5 mm | 71.4% | 4.6 GB |
| mpii | 7k | 7.4 mm | 11.5% | 12.1 GB |
| aic | 287k | 6.9 mm | 16.6% | 36.4 GB |
| coco | ~42k | 6.5 mm | 3.0% | 13.5 GB |
| egohumans | 272k | 6.3 mm | 6.4% | 107 GB |

COCO, MPII and AIC have no finger keypoints and 3DPW is SMPL, so the official
MHR fits fall back on the prior — a hand branch trained on them learns a
constant. **EgoExo4D procedure** (cooking, bike repair, piano — hands doing
things close to the camera) is the only split in the corpus where both terms
are good at once: 2.5× the articulation of COCO/AIC and two thirds of hands
above 128 px. It is also cheap, ~8 GB of referenced frames for 350k persons.
That is where the licence you are waiting on pays off.

Re-run the check on your own build at any time:

```bash
python datasets_pipeline/verify_split.py --dir data/sam3d_gt_egoexo4d --hand-signal
```
