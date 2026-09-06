"""3DPW as a held-out validation set during training.

The default validation set is a random slice of the training corpus, which
measures agreement with the teacher on data drawn from the same distribution —
useful as a training-health signal, useless as a measure of generalisation.
3DPW is real MoCap ground truth from a different distribution, so it answers a
different and more interesting question.

Two rules that matter more than the code:

**Use the ``validation`` split (12 sequences), not ``test``.** The 24 test
sequences are what ``benchmark/eval_3dpw.py`` reports for publication. Select a
checkpoint on them and that number stops being held-out — it becomes a number
you optimised against. The validation split exists precisely so you don't have
to spend the test split on model selection.

**Never build ``3dpw_train`` into the training corpus.** Neighbouring frames of
the same person in the same scene make frame-level holdout meaningless. This is
already the situation locally, where ``sam3d_distill_mix`` contains crops from
all 60 sequences (see ``benchmark/README.md``). On Jean Zay it is not: the built
corpus is coco/mpii/aic/sa1b only, so 3DPW there is genuinely unseen — which is
what makes this worth doing on the cluster and not on the laptop.

GT is the published protocol by default — SMPL forward + the H36M regressor,
precomputed by ``benchmark/make_3dpw_gt.py`` into ``<3DPW root>/gt_h36m/`` and
rsynced next to ``sequenceFiles``. ``gt="jointpositions"`` restores the raw
pickle field the harness used before; the two are several mm apart, so numbers
from the two sources must never be compared to each other.

Selection metric is **PA-MPJPE on J12**. J12 is limbs only, where MHR and SMPL
mean the same anatomical points; J14 adds neck and head, which the two rigs
place differently and which therefore carry a systematic penalty unrelated to
model quality. Procrustes alignment additionally removes any residual
coordinate-frame convention mismatch between the rig-local prediction and 3DPW's
camera space, so the number is comparable across runs without further care.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_BENCH = Path(__file__).resolve().parent.parent / "benchmark"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from benchlib import joints as J          # noqa: E402
from benchlib import metrics as M         # noqa: E402
from benchlib import threedpw as P        # noqa: E402

INPUT_SIZE = 224
CROP_EXPAND = 1.2                          # must match instanthmr/inference.py
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MM = 1000.0


def _preprocess(image_rgb: np.ndarray, bbox: np.ndarray, h: int, w: int,
                focal: float | None = None):
    """Square 1.2x crop + ImageNet normalise + CLIFF conditioning.

    A transcription of ``instanthmr.inference.InstantHMR._preprocess``. It is
    duplicated rather than imported because that module pulls in onnxruntime,
    and this one runs inside the training loop on every rank. If the crop
    convention there ever changes, this must follow or the validation numbers
    will silently stop matching ``benchmark/eval_3dpw.py``.
    """
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    # Must match SAM3DStudentDataset.__getitem__ exactly, including which
    # variant. Feeding a model trained on one and evaluated on the other is a
    # silent distribution shift, not an error.
    if focal is not None:
        cliff = np.array([math.atan((cx - w / 2.0) / focal),
                          math.atan((cy - h / 2.0) / focal),
                          max(bw, bh) / focal], dtype=np.float32)
    else:
        cliff = np.array([2.0 * (cx / w) - 1.0,
                          2.0 * (cy / h) - 1.0,
                          max(bw, bh) / max(w, h)], dtype=np.float32)

    sq = max(bw, bh) * CROP_EXPAND
    half = sq / 2.0
    ix1, iy1 = int(math.floor(cx - half)), int(math.floor(cy - half))
    ix2, iy2 = int(math.ceil(cx - half + sq)), int(math.ceil(cy - half + sq))

    pl, pt = max(0, -ix1), max(0, -iy1)
    pr, pb = max(0, ix2 - w), max(0, iy2 - h)
    patch = image_rgb[max(0, iy1):min(h, iy2), max(0, ix1):min(w, ix2)]
    if pl or pt or pr or pb:
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    crop = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(crop.transpose(2, 0, 1)), cliff


class ThreeDPWValSet(Dataset):
    """(crop, cliff_cond, GT SMPL-24 joints) for every evaluatable person-frame.

    Person boxes come from the projected GT joints, so no detector is in the
    loop and the number isolates the pose model. ``stride`` subsamples frames:
    3DPW is 30 fps and neighbouring frames are near-duplicates, so stride 5
    keeps the signal at a fifth of the cost.
    """

    def __init__(self, sequence_dir: str, image_root: str,
                 split: str = "validation", stride: int = 5,
                 bbox_scale: float = 1.2, gt: str = "h36m",
                 gt_dir: str | None = None, cliff_focal: bool = False):
        self.samples, self.gt = P.build_samples(
            sequence_dir, image_root, split=split,
            stride=stride, bbox_scale=bbox_scale, gt=gt, gt_dir=gt_dir)
        self.gt = np.asarray(self.gt, dtype=np.float32)
        self.gt_source = gt
        self.cliff_focal = cliff_focal

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        bgr = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            # A missing frame must not kill a 20-hour job; return a zero crop
            # and flag it so the metric can drop it.
            return {"image": torch.zeros(3, INPUT_SIZE, INPUT_SIZE),
                    "cliff_cond": torch.zeros(3),
                    "gt": torch.from_numpy(self.gt[i]),
                    "ok": torch.tensor(0.0)}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        crop, cliff = _preprocess(rgb, s["bbox"], h, w,
                                  s["focal"] if self.cliff_focal else None)
        return {"image": torch.from_numpy(crop),
                "cliff_cond": torch.from_numpy(cliff),
                "gt": torch.from_numpy(self.gt[i]),
                "ok": torch.tensor(1.0)}


@torch.no_grad()
def evaluate(model, loader, mhr_module, device, use_amp: bool = True,
             gt: str = "h36m") -> dict:
    """PA-MPJPE and MPJPE in mm on J12 and J14, from the MHR forward pass.

    Predictions come from ``mhr_params`` through the same FK + (70, 127)
    regressor the training loss uses, so this measures exactly the geometry the
    model is being trained to produce.
    """
    model.eval()
    preds, gts = [], []
    n_dropped = 0
    for b in loader:
        if float(b["ok"].sum()) == 0:
            continue
        img = b["image"].to(device, non_blocking=True)
        cc = b["cliff_cond"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(img, cc)
        kp = mhr_module.get_native_keypoints(out["mhr_params"].float(),
                                             out["shape_params"].float())
        # Drop non-finite predictions BEFORE they reach the metric. The forward
        # runs under fp16 autocast, so an early-training head output above
        # 65504 is already inf by the time .float() sees it, and the MHR forward
        # kinematics turn that into inf/NaN keypoints. numpy's SVD inside
        # pa_mpjpe's Procrustes alignment then raises "SVD did not converge",
        # which killed job 1767242 (v3_s0) at the epoch-1 validation with a
        # perfectly healthy training loop behind it -- 0 skipped steps, no
        # divergence. One bad frame out of a few hundred must not cost a
        # 20-hour job.
        finite = torch.isfinite(kp).flatten(1).all(dim=1).cpu()
        keep = b["ok"].bool() & finite
        n_dropped += int((b["ok"].bool() & ~finite).sum())
        if not bool(keep.any()):
            continue
        preds.append(kp[keep.to(kp.device)].float().cpu().numpy())
        gts.append(b["gt"][keep].numpy())

    # Always return the SAME key set. train_distill_jz.all_reduce_mean builds a
    # tensor from sorted(keys) and all-reduces it, so a rank returning a short
    # dict while the others return a full one is a size mismatch -> NCCL hang.
    res: dict[str, float] = {"n": 0.0, "n_dropped": float(n_dropped)}
    for name in ("J12", "J14"):
        res[f"{name}_MPJPE"] = float("inf")
        res[f"{name}_PA_MPJPE"] = float("inf")
    if not preds:
        # inf, not 0: this epoch's 3DPW number must never look like a new best.
        return res

    pred = np.concatenate(preds) * MM          # (N, 70, 3) rig-local, mm
    gt_all = np.concatenate(gts) * MM          # (N, J, 3) camera space, mm

    res["n"] = float(len(pred))
    for name in ("J12", "J14"):
        spec = J.JOINT_SETS[name]
        p = pred[:, spec["mhr"], :]
        g = gt_all[:, J.gt_index(spec, gt), :]
        rp, rg = J.pelvis(p), J.pelvis(g)
        res[f"{name}_MPJPE"] = float(M.mpjpe(p, g, root=(rp, rg)).mean())
        res[f"{name}_PA_MPJPE"] = float(M.pa_mpjpe(p, g).mean())
    return res
