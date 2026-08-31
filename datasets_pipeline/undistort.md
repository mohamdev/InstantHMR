# Undistorting Harmony4D and EgoHumans

Both datasets ship **distorted** frames plus COLMAP intrinsics. The SAM-3D-Body
annotations are expressed against the *undistorted* frames, so skipping this
step silently misaligns every keypoint. EgoExo4D's `sam_3d_body` part is already
undistorted — nothing to do there.

Meta's scripts live in `sam-3d-body/data/scripts/{harmony4d,egohumans}/undistort_images.py`.
Three things block a fresh clone. All three were hit and fixed on 2026-06-30.

### 1. `ModuleNotFoundError: No module named 'lib'`

The script does `from lib.datasets.ego_exo_scene import EgoExoSceneVis`, and
`_init_paths.py` expects `lib/` as a sibling of `harmony4d/` — but the clone
only ships `config/`. Vendor the seven files of `lib/` (`datasets/` + `utils/`)
from <https://github.com/jyuntins/harmony4d> (`main`) into `data/scripts/lib/`.

`stream_archives.py` checks for this directory and fails loudly rather than
part-way through a 45 GB scene.

### 2. `mmcv` / `smplx` missing

`lib/datasets/ego_exo_scene.py` imports both at module level, but only uses
them for SMPL/keypoint *visualisation* — never for undistortion. Make those two
imports lazy (`try/except ImportError → None`) instead of installing two large
packages you will not call. `pycolmap` genuinely is required.

### 3. `--seqs` wants the TAKE, not the scene

The script appends `exo/` to whatever `--seqs` names, so `train/01_hugging` is
wrong (there is no `exo/` there) — it must be `train/01_hugging/001_hugging`.
One scene can hold several takes; pass them comma-separated:

```bash
ls -d $HARMONY4D_DATA_DIR/train/09_karate/*/ \
  | sed 's#.*/train/#train/#; s#/$##' | paste -sd,
```

`stream_archives.py` enumerates the takes of each scene and builds this list for
you, so the manual form is only needed for a one-off.

### Running it

Use the `sam_3d_body_data` env (it has `pycolmap`), not the training env:

```bash
cd $SAM3D_REPO/data/scripts
python harmony4d/undistort_images.py \
    --src_dir $HARMONY4D_DATA_DIR \
    --dst_dir $HARMONY4D_IMG_DIR \
    --seqs train/09_karate/007_karate
```

Validated on `01_hugging/001_hugging` (6.6k frames, 22 exo cameras).
Pass the interpreter to the streamer with `--undistort-python`.
