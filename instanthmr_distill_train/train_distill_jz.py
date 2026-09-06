#!/usr/bin/env python3
"""Multi-GPU (DDP) training of the InstantHMR baseline on Jean Zay.

This is a thin driver around ``train_distill_mhr_only.py``: it imports the
config, dataset, model, loss and metrics from there rather than copying them,
so the changes queued in the change plan only ever have to be made once. What
lives here is the four things the single-GPU script cannot do on a cluster.

**1. DistributedDataParallel.** Ranks come from Slurm (``SLURM_PROCID`` /
``SLURM_LOCALID`` / ``SLURM_NTASKS``) with ``MASTER_ADDR`` taken from the first
node in the allocation, so one ``srun`` line launches the whole job.

**2. Mixture control instead of caps.** SA-1B is 3.40 M of the 3.74 M crops —
91% of the corpus — so uniform sampling trains almost exclusively on it. The
existing ``per_dataset_caps`` is a ``random.sample`` applied once at dataset
construction, which *deletes* crops for the whole run. Here a
:class:`DistributedWeightedSampler` draws with replacement from per-dataset
weights, so every crop stays reachable and the mixture is a knob (``--mix``),
not a demolition. Default is weight proportional to sqrt(count), which pulls
SA-1B from 91% to ~69% while keeping COCO and MPII visible.

**3. Epochs decoupled from corpus size.** ``--samples-per-epoch`` fixes the
epoch length, so "epoch" means a checkpoint interval rather than "one pass over
3.74 M crops" (~58 k steps, ~4 h on one V100). Size it so an epoch is 30-60 min
and a preemption costs at most that.

**4. Resume by resubmitting.** Every epoch writes ``last.pth`` atomically with
the *full* state — model, EMA, optimiser, scaler, scheduler, RNG, epoch, best
metrics. Relaunching the same sbatch picks it up. Note this differs from the
single-GPU script, which resumes from ``best_student_model_v3.pth`` and so
silently discards every epoch since the last improvement.

Offline caveat: Jean Zay compute nodes have no internet, and
``InstantHMRStudent(pretrained=True)`` pulls timm weights from HuggingFace. Run
``--warm-cache`` once on a login or ``prepost`` node to populate
``$TORCH_HOME``/``$HF_HOME`` before the first training job.

    # once, on a node with the proxy
    python instanthmr_distill_train/train_distill_jz.py --warm-cache

    # then, under srun
    srun python instanthmr_distill_train/train_distill_jz.py \
        --data_root $DATA_ROOT --output_dir $RUNS_DIR/baseline
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pickle
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_distill_mhr_only as T  # noqa: E402  (sets up matplotlib/ort at import)


# ============================================================
# Distributed plumbing
# ============================================================
def setup_distributed() -> tuple[int, int, int]:
    """Initialise the process group from Slurm's environment.

    Returns ``(rank, local_rank, world_size)``. Falls back to a single-process
    group when not launched under ``srun``, so the same file runs on a laptop.
    """
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        if "MASTER_ADDR" not in os.environ:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            head = subprocess.check_output(
                ["scontrol", "show", "hostnames", nodelist], text=True
            ).split()[0]
            os.environ["MASTER_ADDR"] = head
        os.environ.setdefault("MASTER_PORT", "29500")
    else:
        rank, local_rank, world_size = 0, 0, 1

    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size,
            # A cold Lustre cache can make the first epoch's data loading slow
            # enough to trip the 30-minute default on the initial barrier.
            timeout=_dt.timedelta(minutes=90),
        )
    return rank, local_rank, world_size


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a, **kw) -> None:
    """Print from rank 0 only. Four ranks logging the same line helps nobody."""
    if is_main():
        print(*a, **kw, flush=True)


def all_reduce_mean(values: dict[str, float], device) -> dict[str, float]:
    """Average a dict of scalars across ranks so every rank sees one number."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return values
    keys = sorted(values)
    t = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return {k: float(v) for k, v in zip(keys, t.tolist())}


# ============================================================
# Mixture-aware distributed sampling
# ============================================================
def dataset_of(npz_path: Path) -> str:
    """``.../sam3d_gt_aic/annotations/xxx.npz`` -> ``sam3d_gt_aic``."""
    return npz_path.parent.parent.name


INDEX_VERSION = 2


def _index_fingerprint(data_root: Path) -> list:
    """Cheap identity for the corpus: the sub-folders and their mtimes.

    A directory's mtime changes when entries are added or removed, so this
    catches the cases that matter (a split rebuilt, a split added) at the cost
    of a handful of stat() calls rather than millions. It will NOT catch a file
    rewritten in place with the same name -- use --rebuild-index for that.
    """
    fp = []
    for ann, img in T.SAM3DStudentDataset._find_dataset_dirs(data_root):
        fp.append([ann.parent.name,
                   int(ann.stat().st_mtime_ns), int(img.stat().st_mtime_ns)])
    return sorted(fp)


def build_pair_index(data_root: str, cache: Path | None = None,
                     rebuild: bool = False, log_fn=print) -> list:
    """The (image, npz) pair list, cached to disk.

    Scanning 3.7 M crops takes over an hour on Lustre and every rank of every
    job in a chained run would otherwise repeat it. The cache stores
    ``(sub_folder, stem, image_extension)`` rather than full paths -- an order
    of magnitude smaller than pickling Path objects -- and paths are rebuilt on
    load, which is seconds rather than minutes because it touches no metadata.
    """
    root = Path(data_root)
    cache = Path(cache) if cache else root / f".pair_index_v{INDEX_VERSION}.pkl"
    fp = _index_fingerprint(root)

    if cache.exists() and not rebuild:
        try:
            with cache.open("rb") as fh:
                blob = pickle.load(fh)
            if blob.get("version") == INDEX_VERSION and blob.get("fingerprint") == fp:
                pairs = [(root / sub / "images" / f"{stem}{ext}",
                          root / sub / "annotations" / f"{stem}.npz")
                         for sub, stem, ext in blob["entries"]]
                # One stat() against millions saved. Catches a cache written by
                # an older build whose stored sub-folder does not round-trip
                # under this data_root -- silently wrong paths would otherwise
                # surface as a FileNotFoundError from a dataloader worker hours
                # into a job.
                if pairs and not pairs[0][0].exists():
                    log_fn(f"  pair index: {cache} does not resolve under "
                           f"{root} (first entry {pairs[0][0]}) — rebuilding")
                    raise FileNotFoundError(cache)
                log_fn(f"  pair index: {len(pairs):,} crops from cache {cache} "
                       f"(built {blob.get('built', '?')})")
                return pairs
            log_fn(f"  pair index: {cache} is stale (corpus changed) — rebuilding")
        except Exception as e:                      # truncated / half-written
            log_fn(f"  pair index: {cache} unreadable ({e}) — rebuilding")

    t0 = time.time()
    log_fn(f"  pair index: scanning {root} (slow — this is why it is cached)")
    ds = T.SAM3DStudentDataset(str(root), augment=False, per_dataset_caps=None)
    # relative_to(root), not .name: when data_root IS a split -- it holds
    # annotations/ and images/ directly -- .name is the root's own folder and
    # rebuilding yields <root>/<root name>/images/... For the nested layout the
    # two are identical, which is why INDEX_VERSION does not need bumping and
    # existing caches stay valid. Path / "." normalises away, so no special case.
    entries = [(str(npz.parent.parent.relative_to(root)), npz.stem, img.suffix)
               for img, npz in ds.pairs]
    log_fn(f"  pair index: {len(entries):,} crops in {time.time() - t0:.0f}s")

    try:
        tmp = cache.with_suffix(cache.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump({"version": INDEX_VERSION, "fingerprint": fp,
                         "built": time.strftime("%Y-%m-%d %H:%M"),
                         "entries": entries}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
        log_fn(f"  pair index: cached to {cache}")
    except OSError as e:
        # A read-only data root is not fatal — it just means we pay the scan
        # again next time. Say so rather than dying an hour into startup.
        log_fn(f"  pair index: could NOT cache to {cache} ({e}); "
               f"every job will repeat the scan")
    return ds.pairs


def build_mixture_weights(dataset, mix: str | None) -> tuple[torch.Tensor, str]:
    """Per-sample sampling weights implementing a target dataset mixture.

    ``mix`` is either ``"sqrt"`` / ``"uniform"`` / ``"equal"``, or an explicit
    ``name=proportion`` list (``sa1b=0.4,aic=0.35,coco=0.15,mpii=0.1``, matched
    as a substring of the split folder name). Proportions are renormalised, so
    they need not sum to 1.

    - ``uniform``: every crop equally likely — SA-1B ends up at 91%.
    - ``sqrt``: proportion proportional to sqrt(count). The usual compromise for
      a long-tailed corpus: keeps the big split dominant without erasing the
      small ones.
    - ``equal``: every split contributes the same number of samples per epoch.
      With MPII at 7 k crops that means ~50x oversampling, which augmentation
      cannot disguise — expect it to overfit MPII.
    """
    names = [dataset_of(npz) for _, npz in dataset.pairs]
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    splits = sorted(counts)

    if mix in (None, "sqrt"):
        target = {s: math.sqrt(counts[s]) for s in splits}
    elif mix == "uniform":
        target = {s: float(counts[s]) for s in splits}
    elif mix == "equal":
        target = {s: 1.0 for s in splits}
    else:
        req: dict[str, float] = {}
        for tok in mix.split(","):
            k, _, v = tok.partition("=")
            req[k.strip()] = float(v)
        target = {}
        for s in splits:
            hit = [v for k, v in req.items() if k in s]
            if len(hit) > 1:
                raise ValueError(f"--mix key matches '{s}' more than once: {req}")
            target[s] = hit[0] if hit else 0.0
        if sum(target.values()) <= 0:
            raise ValueError(f"--mix matched no split. Splits present: {splits}")

    tot = sum(target.values())
    prop = {s: target[s] / tot for s in splits}
    # Per-sample weight: a split's total probability mass divided evenly over
    # its crops. WeightedRandomSampler normalises, so absolute scale is free.
    per_sample = {s: (prop[s] / counts[s] if counts[s] else 0.0) for s in splits}
    w = torch.tensor([per_sample[n] for n in names], dtype=torch.double)

    rows = [f"  {'split':<24}{'crops':>12}{'raw %':>9}{'sampled %':>11}"]
    total = sum(counts.values())
    for s in splits:
        rows.append(f"  {s:<24}{counts[s]:>12,}{100*counts[s]/total:>8.1f}%"
                    f"{100*prop[s]:>10.1f}%")
    return w, "\n".join(rows)


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampling with replacement, sharded across ranks.

    ``WeightedRandomSampler`` has no distributed variant. Rather than sample
    per rank — which would give each rank a different mixture and a different
    epoch length — every rank draws the *same* index list from a generator
    seeded with ``(seed, epoch)``, then takes its own stride. So the global
    mixture is exactly the requested one, the shards are disjoint, and the
    sequence is reproducible from the epoch number alone, which is what makes
    resume deterministic.
    """

    def __init__(self, weights: torch.Tensor, num_samples: int,
                 num_replicas: int, rank: int, seed: int = 0):
        self.weights = weights.double()
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        # Round up so every rank gets the same count; DDP deadlocks if one rank
        # runs out of batches before the others.
        self.per_rank = int(math.ceil(num_samples / num_replicas))
        self.total = self.per_rank * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.per_rank

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100_003 + self.epoch)
        idx = torch.multinomial(self.weights, self.total, replacement=True, generator=g)
        return iter(idx[self.rank::self.num_replicas].tolist())


# ============================================================
# Data
# ============================================================
def build_jz_loaders(cfg, args, rank: int, world: int):
    """Train loader with mixture sampling, plus a small held-out val loader.

    The val split is a fixed slice of the *same* corpus, so it measures
    agreement with the teacher rather than generalisation — it is a training
    health signal, not a model-selection metric. Keep it small (``--val-frac``,
    default 2%): with 3.74 M crops even 1% is 37 k samples, and evaluating it
    every epoch is pure overhead. See ``--val-3dpw`` in the README notes for
    the held-out alternative.
    """
    geom = dict(geom_p=cfg.geom_p, geom_rot_deg=cfg.geom_rot_deg,
                geom_scale_range=cfg.geom_scale_range,
                geom_trans=cfg.geom_trans, geom_flip_p=cfg.geom_flip_p,
                geom_scale_max=cfg.geom_scale_max,
                cliff_follows_aug=cfg.cliff_follows_aug,
                cliff_focal=cfg.cliff_focal,
                occl_p=cfg.occl_p, occl_scale=cfg.occl_scale,
                jpeg_p=cfg.jpeg_p, jpeg_quality=cfg.jpeg_quality)

    # Rank 0 builds the index (or reads the cache) while the others wait, then
    # everyone loads the same cache. Building on all ranks at once would put
    # world_size concurrent metadata storms on the same filesystem, which is
    # slower than doing it once and antisocial on a shared cluster.
    if world > 1 and rank != 0:
        dist.barrier()
    pairs = build_pair_index(cfg.data_root, args.index_cache,
                             rebuild=args.rebuild_index, log_fn=log)
    if world > 1 and rank == 0:
        dist.barrier()
    if cfg.max_images is not None:
        pairs = pairs[:cfg.max_images]

    # Both datasets share one index; only the augmentation pipeline differs.
    train_ds = T.SAM3DStudentDataset(
        cfg.data_root, augment=cfg.augment, pairs=pairs, **geom)
    val_ds = T.SAM3DStudentDataset(
        cfg.data_root, augment=False, pairs=pairs, **geom)

    n = len(train_ds)
    g = torch.Generator().manual_seed(42)          # fixed: the split must not
    perm = torch.randperm(n, generator=g).tolist()  # move when --seed changes
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    weights_all, report = build_mixture_weights(train_ds, args.mix)
    log(f"\ndataset mixture ('{args.mix or 'sqrt'}'):\n{report}")

    # Zero out the val indices so the sampler can never draw them.
    w = weights_all.clone()
    w[torch.tensor(val_idx, dtype=torch.long)] = 0.0

    sampler = DistributedWeightedSampler(
        w, num_samples=args.samples_per_epoch,
        num_replicas=world, rank=rank, seed=args.seed)

    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    if cfg.num_workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kw)

    # Val is sharded by plain striding — no weighting, no replacement.
    val_subset = val_idx[rank::world]
    val_loader = DataLoader(
        torch.utils.data.Subset(val_ds, val_subset),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(1, cfg.num_workers // 2), pin_memory=True, drop_last=False)

    log(f"train pool {len(train_idx):,} crops | val {n_val:,} "
        f"({len(val_subset):,} on this rank) | "
        f"{args.samples_per_epoch:,} samples/epoch")
    return train_loader, val_loader, sampler, train_ds


# ============================================================
# Checkpointing
# ============================================================
def atomic_save_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def atomic_save(obj, path: Path) -> None:
    """Write then rename. A job killed mid-``torch.save`` must not leave a
    truncated ``last.pth`` — that would turn one lost epoch into a lost run."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_state(path: Path, *, model, ema, optimizer, scaler, scheduler,
               epoch, best, args, cfg) -> None:
    atomic_save({
        "epoch": epoch,
        "model_state_dict": unwrap(model).state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best": best,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "run": {"seed": args.seed, "epochs": args.epochs,
                "samples_per_epoch": args.samples_per_epoch,
                "batch_size": cfg.batch_size, "lr": cfg.lr, "mix": args.mix},
    }, path)


def unwrap(m):
    return m.module if isinstance(m, (DDP, AveragedModel)) else m


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def run_validation(eval_model, loader, criterion, cfg, mhr_module, device):
    eval_model.eval()
    acc = {"loss": 0.0, "Mesh_MPJPE": 0.0, "Mesh_PA_MPJPE": 0.0,
           "Mesh_PA_MPJPE_body": 0.0}
    n = 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
            preds = eval_model(batch["image"], batch["cliff_cond"])
        preds = {k: v.float() for k, v in preds.items()}
        losses = criterion(preds, batch)
        lv = losses["total_loss"].item()
        if math.isnan(lv) or math.isinf(lv):
            continue
        bm = T.evaluate_hmr_batch(preds, batch, cfg, mhr_module)
        acc["loss"] += lv
        acc["Mesh_MPJPE"] += bm.get("Mesh_MPJPE", 0.0)
        acc["Mesh_PA_MPJPE"] += bm.get("Mesh_PA_MPJPE", 0.0)
        # Reported only. Mesh_PA_MPJPE is 60% fingers by keypoint count; this is
        # the 30 body/face/foot keypoints. Selection is untouched — and when
        # --val-3dpw is on, both of these are replaced by the 3DPW J12 numbers.
        acc["Mesh_PA_MPJPE_body"] += bm.get("Mesh_PA_MPJPE_body", 0.0)
        n += 1
    out = {k: v / max(n, 1) for k, v in acc.items()}
    out["n"] = float(n)
    # Average over ranks so the early-stopping decision is identical everywhere.
    return all_reduce_mean(out, device)


# ============================================================
# Training
# ============================================================
def train(args, cfg, rank, local_rank, world):
    device = torch.device(f"cuda:{local_rank}")
    T.device = device                     # the module global some helpers read
    out_dir = Path(cfg.log_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    W = np.load(cfg.kp_regressor_path)
    mhr_module = T.MHRForwardPass(cfg.mhr_model_path, device, kp_regressor=W)
    criterion = T.DistillationLoss(cfg, mhr_module)

    train_loader, val_loader, sampler, _ = build_jz_loaders(cfg, args, rank, world)
    steps_per_epoch = len(train_loader)

    dpw_loader = dpw_test_loader = None
    if args.val_3dpw:
        if not args.val_3dpw_images:
            raise SystemExit("--val-3dpw needs --val-3dpw-images")
        from val3dpw import ThreeDPWValSet
        dpw = ThreeDPWValSet(args.val_3dpw, args.val_3dpw_images,
                             split=args.val_3dpw_split, stride=args.val_3dpw_stride,
                             gt=args.val_3dpw_gt, cliff_focal=cfg.cliff_focal)
        # Stride-shard across ranks; the metric is all-reduced afterwards.
        dpw_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dpw, list(range(rank, len(dpw), world))),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=max(1, cfg.num_workers // 2), pin_memory=True)
        log(f"3DPW {args.val_3dpw_split}: {len(dpw):,} person-frames "
            f"(stride {args.val_3dpw_stride}, GT {args.val_3dpw_gt}) "
            f"-> selection metric is J12 PA-MPJPE")

        # Optional read-only monitor on the split the paper reports. Selection
        # NEVER touches it: picking the best of ~100 epochs on the test split
        # would bank a few mm of selection luck and the published number would
        # no longer be held out. Watching it is fine; optimising against it is
        # not, so it is logged and never compared to `best`.
        if args.val_3dpw_test:
            dpw_t = ThreeDPWValSet(args.val_3dpw, args.val_3dpw_images,
                                   split="test", stride=args.val_3dpw_test_stride,
                                   gt=args.val_3dpw_gt, cliff_focal=cfg.cliff_focal)
            dpw_test_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dpw_t, list(range(rank, len(dpw_t), world))),
                batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                num_workers=max(1, cfg.num_workers // 2), pin_memory=True)
            log(f"3DPW test:       {len(dpw_t):,} person-frames "
                f"(stride {args.val_3dpw_test_stride}) -> REPORTED ONLY, "
                f"never used for checkpoint selection")

    model = T.InstantHMRStudent(cfg, pretrained=not args.no_pretrained).to(device)
    if args.sync_bn and world > 1:
        # repvit is 218 BatchNorm2d layers. At batch 64 per GPU the local stats
        # are already reliable, so this is off by default — it costs an all-
        # reduce per BN layer per step for no accuracy gain at this batch size.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False) if world > 1 else model

    param_groups = T.build_param_groups(model, cfg.weight_decay,
                                        split_wd=cfg.split_weight_decay)
    optimizer = optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.1)
    ema = AveragedModel(model, multi_avg_fn=T.make_ema_avg_fn(cfg.ema_decay),
                        use_buffers=True)

    start_epoch = 0
    best = {"raw": float("inf"), "ema": float("inf"), "overall": float("inf"),
            "no_improve": 0}
    last_path = out_dir / "last.pth"
    if last_path.exists() and not args.no_resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        prev = ck.get("run", {})
        prev_epochs = prev.get("epochs")
        extending = bool(args.extend and prev_epochs is not None
                         and args.epochs > prev_epochs)
        for k in ("epochs", "samples_per_epoch", "batch_size"):
            cur = {"epochs": args.epochs, "samples_per_epoch": args.samples_per_epoch,
                   "batch_size": cfg.batch_size}[k]
            if prev.get(k) not in (None, cur) and not (k == "epochs" and extending):
                log(f"⚠️  {k} changed {prev[k]} -> {cur} since the checkpoint. "
                    f"OneCycleLR's schedule is defined over the ORIGINAL total "
                    f"step count, so the LR curve will not match a fresh run.")
        # A changed --lr on resume is silently ignored: OneCycleLR keeps max_lr in
        # the optimiser's param groups, and optimizer.load_state_dict below
        # restores the checkpoint's. You would think you were training at the new
        # rate and actually be back at the old one. Refuse instead -- this cost a
        # 16-hour run once.
        prev_lr = prev.get("lr")
        if prev_lr is not None and abs(prev_lr - cfg.lr) > 1e-12 and not extending:
            raise SystemExit(
                f"refusing to resume: checkpoint was trained at peak lr "
                f"{prev_lr:.3e}, this invocation asks for {cfg.lr:.3e}.\n"
                f"OneCycleLR stores max_lr in the optimiser state, so resuming "
                f"would silently use {prev_lr:.3e} and ignore your --lr.\n"
                f"Use a NEW --output_dir (RUN_NAME) to start fresh at "
                f"{cfg.lr:.3e}, or delete {last_path} to restart in place.")

        unwrap(model).load_state_dict(ck["model_state_dict"])
        ema.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck["epoch"] + 1
        if extending:
            # A finished OneCycle run sits at lr ~= 0, and its state_dict carries
            # total_steps -- loading it would clobber the longer schedule built
            # above and then raise on the first step past the old total. Instead
            # keep the NEW schedule and fast-forward it to where we are, so the
            # remaining epochs follow the tail of a curve drawn for the full
            # extended length.
            #
            # Be clear about what this is: a warm restart, not the same
            # trajectory as one uninterrupted run of the longer length, because
            # the weights up to here were produced under a schedule that had
            # already annealed. Fine for squeezing more out of a converged run;
            # not a substitute for choosing the epoch count up front when two
            # arms have to be comparable.
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()
            log(f"⏩ extending {prev_epochs} -> {args.epochs} epochs. Warm restart: "
                f"schedule rebuilt over the new total and fast-forwarded to "
                f"step {start_epoch * steps_per_epoch:,}, lr "
                f"{scheduler.get_last_lr()[0]:.2e}. NOT equivalent to a single "
                f"{args.epochs}-epoch run.")
        else:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        best = ck["best"]
        if (r := ck.get("rng")):
            random.setstate(r["python"]); np.random.set_state(r["numpy"])
            torch.set_rng_state(r["torch"].cpu())
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in r["cuda"]])
            except Exception:
                pass   # a different GPU count than the run that saved it
        log(f"🔄 resumed from {last_path} at epoch {start_epoch} "
            f"(best PA-MPJPE {best['overall']:.1f} mm)")
    else:
        log("🚀 starting from scratch")

    log(f"steps/epoch {steps_per_epoch} | world {world} | "
        f"effective batch {cfg.batch_size * world} | peak lr {cfg.lr:.2e}")

    if is_main():
        # summarize_runs.py reads this for the epoch total, the preset and the
        # seed; without it every run shows "?/?" and no preset.
        atomic_save_json(out_dir / "run_config.json", {
            "preset": args.preset, "seed": args.seed, "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": cfg.batch_size, "world": world,
            "lr": cfg.lr, "lr_arg": args.lr, "lr_scaling": args.lr_scaling,
            "mix": args.mix, "steps_per_epoch": steps_per_epoch,
            "backbone": cfg.backbone, "w_reproj": cfg.w_reproj,
            "geom_scale_max": cfg.geom_scale_max, "occl_p": cfg.occl_p,
            "jpeg_p": cfg.jpeg_p, "cam_loss": cfg.cam_loss,
            "losses": args.losses, "w_shape": cfg.w_shape,
            "w_keypoints3d": cfg.w_keypoints3d, "kp3d_loss": cfg.kp3d_loss,
            "pose_split": cfg.pose_split, "pose_beta": cfg.pose_beta,
            "finger_weight": cfg.finger_weight,
            "val_3dpw_gt": args.val_3dpw_gt if args.val_3dpw else None,
            "cliff_focal": cfg.cliff_focal,
            "started": time.strftime("%Y-%m-%d %H:%M")})

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        # reproj and cam are here because --preset v2 changes both; without them
        # a regression in the absolute-pose supervision is invisible in
        # history.jsonl and you would have to guess from the total.
        run = {"total": 0.0, "3d": 0.0, "2d": 0.0, "simcc": 0.0,
               "reproj": 0.0, "cam": 0.0, "pose": 0.0}
        nb = 0
        n_skip = n_skip_run = 0
        t_epoch = time.time()
        criterion.set_step(epoch * steps_per_epoch)
        for i, batch in enumerate(train_loader):
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                preds = ddp_model(batch["image"], batch["cliff_cond"])
            preds = {k: v.float() for k, v in preds.items()}
            losses = criterion(preds, batch)
            loss = losses["total_loss"]
            lv = loss.item()

            # An anomalous step is skipped, but every rank must agree or DDP's
            # gradient all-reduce deadlocks: rank 0 skipping while rank 1 steps
            # leaves rank 1 waiting forever. Vote, then act together.
            bad = int(math.isnan(lv) or math.isinf(lv) or lv > cfg.anomaly_loss_threshold)
            if world > 1:
                flag = torch.tensor([bad], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                bad = int(flag.item())
            if bad:
                n_skip += 1
                n_skip_run += 1
                # Step the schedule ANYWAY. Skipping it freezes the LR, and a
                # too-high LR is what produces anomalies in the first place, so
                # `continue`-ing past it is a runaway: skip -> LR stays at peak
                # -> more skips -> never anneals. Measured on the first sweep:
                # by epoch 20 the scheduler had taken 8,161 of 29,678 steps and
                # the LR was pinned at 1.18e-3 instead of annealing to 4.3e-4.
                scheduler.step()
                # Drop the graph before the next forward builds another one.
                # Without this, `continue` leaves two autograd graphs alive at
                # once -- measured 1.8x peak memory, 9.3 -> 17.2 GiB, which OOMs
                # a 16 GB node.
                del preds, losses, loss
                if n_skip_run == 1 or n_skip_run % 100 == 0:
                    log(f"⚠️  skipping step {i} (loss {lv:.1f}) "
                        f"[{n_skip_run} consecutive]")
                if n_skip_run >= cfg.max_consecutive_skips:
                    raise RuntimeError(
                        f"{n_skip_run} consecutive anomalous steps at epoch "
                        f"{epoch + 1} — the run has diverged and will not "
                        f"recover. Lower --lr (or use --lr-scaling sqrt/none) "
                        f"and restart. Exiting rather than burning the "
                        f"remaining {args.epochs - epoch} epochs.")
                continue
            n_skip_run = 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update_parameters(unwrap(ddp_model))
            criterion.set_step(epoch * steps_per_epoch + i)

            run["total"] += lv
            run["3d"] += losses.get("loss_3d_native", torch.zeros(())).item()
            run["2d"] += losses.get("loss_2d_native", torch.zeros(())).item()
            run["simcc"] += losses.get("loss_2d_simcc", torch.zeros(())).item()
            run["reproj"] += losses.get("loss_reproj", torch.zeros(())).item()
            run["cam"] += losses.get("loss_cam", torch.zeros(())).item()
            # Under cfg.pose_split, loss_pose is the local half and loss_pose_root
            # the other; summing keeps history.jsonl's "pose" column comparable
            # against a --losses legacy run.
            run["pose"] += (losses.get("loss_pose", torch.zeros(())).item()
                            + losses.get("loss_pose_root", torch.zeros(())).item())
            nb += 1
            if is_main() and i % args.log_every == 0:
                log(f"  e{epoch+1}/{args.epochs} step {i}/{steps_per_epoch} "
                    f"loss {lv:.4f} lr {scheduler.get_last_lr()[0]:.2e}")

        tr = all_reduce_mean({k: v / max(nb, 1) for k, v in run.items()}, device)
        raw = run_validation(ddp_model, val_loader, criterion, cfg, mhr_module, device)
        em = run_validation(ema, val_loader, criterion, cfg, mhr_module, device)

        # 3DPW, when available, replaces the held-in metric for selection: it is
        # the only signal here that measures generalisation rather than teacher
        # agreement. The held-in numbers keep printing either way.
        dpw_raw = dpw_em = dpw_test_raw = dpw_test_em = None
        if dpw_loader is not None:
            import val3dpw
            dpw_raw = all_reduce_mean(
                val3dpw.evaluate(ddp_model, dpw_loader, mhr_module, device,
                                 cfg.use_amp, gt=args.val_3dpw_gt),
                device)
            dpw_em = all_reduce_mean(
                val3dpw.evaluate(ema, dpw_loader, mhr_module, device,
                                 cfg.use_amp, gt=args.val_3dpw_gt),
                device)
            if dpw_test_loader is not None:
                dpw_test_raw = all_reduce_mean(val3dpw.evaluate(
                    ddp_model, dpw_test_loader, mhr_module, device, cfg.use_amp,
                    gt=args.val_3dpw_gt), device)
                dpw_test_em = all_reduce_mean(val3dpw.evaluate(
                    ema, dpw_test_loader, mhr_module, device, cfg.use_amp,
                    gt=args.val_3dpw_gt), device)
            raw["Mesh_PA_MPJPE"], em["Mesh_PA_MPJPE"] = (
                dpw_raw["J12_PA_MPJPE"], dpw_em["J12_PA_MPJPE"])
            raw["Mesh_MPJPE"], em["Mesh_MPJPE"] = (
                dpw_raw["J12_MPJPE"], dpw_em["J12_MPJPE"])

        log(f"\n📈 epoch {epoch+1}/{args.epochs} | lr {scheduler.get_last_lr()[0]:.2e}"
            f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        log(f"   train  total {tr['total']:.4f} | 3d {tr['3d']:.4f} | "
            f"2d {tr['2d']:.4f} | simcc {tr['simcc']:.3f} | "
            f"reproj {tr['reproj']:.4f} | cam {tr['cam']:.4f} "
            f"| skipped {n_skip}/{steps_per_epoch} | {time.time() - t_epoch:.0f}s")
        log(f"   valRAW loss {raw['loss']:.4f} | PA-MPJPE {raw['Mesh_PA_MPJPE']:.1f} | "
            f"MPJPE {raw['Mesh_MPJPE']:.1f} mm | held-in body PA "
            f"{raw['Mesh_PA_MPJPE_body']:.1f} mm")
        log(f"   valEMA loss {em['loss']:.4f} | PA-MPJPE {em['Mesh_PA_MPJPE']:.1f} | "
            f"MPJPE {em['Mesh_MPJPE']:.1f} mm | held-in body PA "
            f"{em['Mesh_PA_MPJPE_body']:.1f} mm")
        if dpw_raw is not None:
            log(f"   3DPW-{args.val_3dpw_split} RAW  J12 PA {dpw_raw['J12_PA_MPJPE']:.1f} "
                f"MPJPE {dpw_raw['J12_MPJPE']:.1f} | J14 PA {dpw_raw['J14_PA_MPJPE']:.1f} mm")
            log(f"   3DPW-{args.val_3dpw_split} EMA  J12 PA {dpw_em['J12_PA_MPJPE']:.1f} "
                f"MPJPE {dpw_em['J12_MPJPE']:.1f} | J14 PA {dpw_em['J14_PA_MPJPE']:.1f} mm")
            # Non-finite predictions are dropped rather than crashing the metric
            # (see val3dpw.evaluate). Say so — a handful at epoch 1 is the raw
            # branch still settling, a rising count is a real instability.
            nd = dpw_raw.get("n_dropped", 0.0) + dpw_em.get("n_dropped", 0.0)
            if nd > 0:
                log(f"   ⚠️  3DPW: {nd:.0f} non-finite prediction(s) dropped "
                    f"(raw {dpw_raw.get('n_dropped', 0.0):.0f} / "
                    f"ema {dpw_em.get('n_dropped', 0.0):.0f})")
        if dpw_test_raw is not None:
            log(f"   3DPW-TEST (report only, not used for selection)")
            log(f"     RAW  J14 MPJPE {dpw_test_raw['J14_MPJPE']:.1f} | "
                f"J14 PA {dpw_test_raw['J14_PA_MPJPE']:.1f} mm")
            log(f"     EMA  J14 MPJPE {dpw_test_em['J14_MPJPE']:.1f} | "
                f"J14 PA {dpw_test_em['J14_PA_MPJPE']:.1f} mm")

        improved = False
        if raw["Mesh_PA_MPJPE"] < best["raw"]:
            best["raw"] = raw["Mesh_PA_MPJPE"]; improved = True
            if is_main():
                atomic_save({"model_state_dict": unwrap(model).state_dict(),
                             "val_pa_mpjpe": best["raw"], "source": "raw",
                             "epoch": epoch}, out_dir / "best_student_model_raw.pth")
        if em["Mesh_PA_MPJPE"] < best["ema"]:
            best["ema"] = em["Mesh_PA_MPJPE"]; improved = True
            if is_main():
                atomic_save({"model_state_dict": unwrap(ema).state_dict(),
                             "val_pa_mpjpe": best["ema"], "source": "ema",
                             "epoch": epoch}, out_dir / "best_student_model_ema.pth")
        cur = min(raw["Mesh_PA_MPJPE"], em["Mesh_PA_MPJPE"])
        if cur < best["overall"] - 1e-4:
            best["overall"] = cur
            use_ema = em["Mesh_PA_MPJPE"] <= raw["Mesh_PA_MPJPE"]
            if is_main():
                src = unwrap(ema) if use_ema else unwrap(model)
                atomic_save({"model_state_dict": src.state_dict(),
                             "val_pa_mpjpe": cur, "epoch": epoch,
                             "source": "ema" if use_ema else "raw"},
                            out_dir / "best_student_model_v3.pth")
                log(f"💾 new best {cur:.1f} mm ({'EMA' if use_ema else 'RAW'})")
        best["no_improve"] = 0 if improved else best["no_improve"] + 1

        if is_main():
            save_state(out_dir / "last.pth", model=model, ema=ema,
                       optimizer=optimizer, scaler=scaler, scheduler=scheduler,
                       epoch=epoch, best=best, args=args, cfg=cfg)
            with (out_dir / "history.jsonl").open("a") as fh:
                fh.write(json.dumps({"epoch": epoch, "train": tr,
                                     "val_raw": raw, "val_ema": em,
                                     "dpw_raw": dpw_raw, "dpw_ema": dpw_em,
                                     "dpw_test_raw": dpw_test_raw,
                                     "dpw_test_ema": dpw_test_em,
                                     "lr": scheduler.get_last_lr()[0],
                                     "skipped": n_skip,
                                     "sec": round(time.time() - t_epoch, 1),
                                     "at": time.time()}) + "\n")
        if world > 1:
            dist.barrier()   # nobody starts epoch N+1 before last.pth is on disk

        if best["no_improve"] >= cfg.early_stop_patience:
            log(f"\n🛑 early stop: no improvement for {cfg.early_stop_patience} epochs")
            break

    # The sbatch resubmits itself unless this file exists, so it must be written
    # on normal completion only — never in an exception path.
    if is_main():
        (out_dir / "DONE").write_text(
            f"best_pa_mpjpe={best['overall']:.3f}\nepochs={args.epochs}\n")
    log(f"\n🎉 done. best PA-MPJPE {best['overall']:.1f} mm -> {out_dir}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    p.add_argument("--output_dir", required=False,
                   default=os.environ.get("OUTPUT_DIR"))
    p.add_argument("--mhr_model_path", default=None)
    p.add_argument("--kp_regressor_path", default=None)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64,
                   help="PER GPU. Effective batch is this times the world size.")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak LR for a single GPU; scaled by --lr-scaling.")
    p.add_argument("--lr-scaling", choices=("linear", "sqrt", "none"), default="sqrt",
                   help="How to scale --lr with world size. 'linear' is the "
                        "standard rule for SGD-style large-batch training; "
                        "'sqrt' is gentler and sometimes better for Adam.")
    p.add_argument("--num_workers", type=int, default=9)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--samples-per-epoch", type=int, default=400_000,
                   help="Epoch length in crops. Size it so one epoch is 30-60 "
                        "min: a preemption costs at most one epoch.")
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="Held-in val fraction. This is a training-health signal "
                        "(agreement with the teacher on the same distribution), "
                        "not a generalisation measure. 0 disables it.")
    p.add_argument("--val-3dpw", default=os.environ.get("THREEDPW_SEQ"),
                   help="3DPW sequenceFiles/ directory. When given, PA-MPJPE on "
                        "J12 becomes the checkpoint-selection metric.")
    p.add_argument("--val-3dpw-images", default=os.environ.get("THREEDPW_IMG"),
                   help="3DPW imageFiles/ directory.")
    p.add_argument("--val-3dpw-split", default="validation",
                   choices=("validation", "test", "train"),
                   help="Keep this at 'validation'. Selecting on 'test' spends "
                        "the split benchmark/eval_3dpw.py reports.")
    p.add_argument("--val-3dpw-test", action="store_true",
                   help="ALSO evaluate the 24-sequence test split each epoch and "
                        "log it, so you can watch the number the paper will "
                        "report. Never used for checkpoint selection.")
    p.add_argument("--val-3dpw-test-stride", type=int, default=5)
    p.add_argument("--val-3dpw-stride", type=int, default=5,
                   help="3DPW is 30 fps; neighbouring frames are near-duplicates.")
    p.add_argument("--cliff-focal", action="store_true",
                   help="Perspective-correct CLIFF conditioning: angles off the "
                        "optical axis and angular box size instead of "
                        "image-normalised pixels. Required once the corpus "
                        "carries real per-image focals (f/diag spans 0.6-1.8), "
                        "where the pixel form is ambiguous and cam_trans and "
                        "body scale come out unstable. Applies to training AND "
                        "to the 3DPW validation crops; the two must agree.")
    p.add_argument("--val-3dpw-gt", default="h36m",
                   choices=("h36m", "jointpositions"),
                   help="h36m (default) is the published protocol and needs "
                        "gt_h36m/ next to sequenceFiles/ (see "
                        "benchmark/make_3dpw_gt.py). jointpositions is the raw "
                        "pickle field the runs before this flag selected on — "
                        "several mm apart, so a resumed run must keep its own.")
    p.add_argument("--mix", default="sqrt",
                   help="Dataset mixture: sqrt | uniform | equal | "
                        "'sa1b=0.4,aic=0.35,coco=0.15,mpii=0.1'.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_images", type=int, default=None)

    p.add_argument("--w_keypoints3d", type=float, default=None)
    p.add_argument("--w_simcc", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--ema-decay", type=float, default=None)

    p.add_argument("--sync-bn", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--extend", action="store_true",
                   help="Continue a run past the epoch count it was trained "
                        "with: rebuild the LR schedule over the new total and "
                        "fast-forward it. A warm restart, not the same as one "
                        "long run -- see the note in the resume path.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip the timm download. Compute nodes have no internet, "
                        "so run --warm-cache once on a login node instead.")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--preset", choices=["baseline", "v2"], default="baseline",
                   help="'v2' enables the wave-1 bundle: truncation, occlusion "
                        "and JPEG augmentation with the SimCC-mask and CLIFF "
                        "prerequisites, plus absolute-pose supervision. "
                        "'baseline' reproduces the pre-v2 trainer exactly.")
    p.add_argument("--w_reproj", type=float, default=None)
    p.add_argument("--losses", choices=("legacy", "rebalanced"), default="legacy",
                   help="Orthogonal to --preset, which only controls the input "
                        "pipeline and the absolute-pose terms. 'rebalanced' applies "
                        "the four loss-budget fixes (w_shape 1.0->0.03, pose split "
                        "onto the non-flipped 60%% of the batch at beta 0.05, "
                        "unsquared-Euclidean 3D loss at w 0.35, finger keypoints at "
                        "0.2). 'legacy' reproduces the pre-rebalance recipe bit for "
                        "bit, so b2/v2 runs in flight stay valid controls.")
    p.add_argument("--index-cache", default=None,
                   help="Where to cache the (image, npz) pair index. "
                        "Default: <data_root>/.pair_index_v2.pkl")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Rescan the corpus even if the cache looks current.")
    p.add_argument("--warm-cache", action="store_true",
                   help="Instantiate the backbone to populate $TORCH_HOME / "
                        "$HF_HOME, then exit. Run on a login or prepost node.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build config, dataset and sampler, print the mixture, "
                        "then exit without training.")
    return p.parse_args()


def build_cfg(args):
    cfg = T.DistillConfig()
    if args.data_root:        cfg.data_root = args.data_root
    if args.output_dir:       cfg.log_dir = args.output_dir
    if args.mhr_model_path:   cfg.mhr_model_path = args.mhr_model_path
    if args.kp_regressor_path: cfg.kp_regressor_path = args.kp_regressor_path
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.max_images = args.max_images
    cfg.epochs = args.epochs
    cfg.per_dataset_caps = {}            # replaced by the weighted sampler
    if args.backbone:               cfg.backbone = args.backbone
    if args.w_simcc is not None:    cfg.w_simcc = args.w_simcc
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.ema_decay is not None:  cfg.ema_decay = args.ema_decay

    if args.preset == "v2":
        apply_v2_preset(cfg)
    # Orthogonal axis: --preset is the input pipeline + absolute-pose terms,
    # --losses is the loss budget. Applied second so it owns w_keypoints3d.
    if args.losses == "rebalanced":
        T.apply_rebalanced_losses(cfg)
    # An explicit --w_reproj / --w_keypoints3d still wins over both.
    if args.w_reproj is not None:   cfg.w_reproj = args.w_reproj
    if args.w_keypoints3d is not None: cfg.w_keypoints3d = args.w_keypoints3d
    # Orthogonal to both presets: it changes what the network is TOLD, not what
    # it is trained on or scored against.
    cfg.cliff_focal = args.cliff_focal
    return cfg


def apply_v2_preset(cfg) -> None:
    """Wave-1 bundle: augmentation (MeTRAbs items 1/2/4) + absolute-pose (item 3).

    Deliberately one preset rather than four independent flags. MeTRAbs measures
    the augmentations cumulatively (Table 7: 58.0 -> 52.8 -> 49.3), so splitting
    them into separate runs would spend three 3-seed sweeps reproducing an
    ablation that already exists in the literature. Absolute-pose supervision
    rides along because it touches disjoint machinery -- the reprojection and
    translation terms, not the input pipeline -- so a regression stays
    attributable from the per-term loss breakdown in history.jsonl.
    """
    # --- item 1: truncation -------------------------------------------------
    cfg.geom_scale_max = 2.0          # retains the central quarter by area
    cfg.geom_trans = 0.20             # off-centre, so truncation is not symmetric
    cfg.cliff_follows_aug = True      # prerequisite B
    cfg.mask_oob_2d = True            # prerequisite A
    # --- item 2: occlusion --------------------------------------------------
    cfg.occl_p = 0.7                  # MeTRAbs Table 7
    cfg.occl_scale = (0.02, 0.30)
    # --- item 4: JPEG -------------------------------------------------------
    cfg.jpeg_p = 0.3
    cfg.jpeg_quality = (30, 90)
    # --- item 3: absolute pose ---------------------------------------------
    cfg.w_reproj = 0.5                # was 0.01, i.e. effectively off
    cfg.reproj_all_samples = True     # the flipped half now contributes
    cfg.cam_loss = "euclid"


def main():
    args = parse_args()

    if args.warm_cache:
        cfg = build_cfg(args)
        import timm
        m = timm.create_model(cfg.backbone, pretrained=True, num_classes=0)
        print(f"cached {cfg.backbone}: {sum(p.numel() for p in m.parameters())/1e6:.2f} M "
              f"params under TORCH_HOME={os.environ.get('TORCH_HOME')} "
              f"HF_HOME={os.environ.get('HF_HOME')}")
        # The pair index too: this is the other thing a compute node should
        # never have to build, and a prepost node has the walltime for it.
        build_pair_index(cfg.data_root, args.index_cache,
                         rebuild=args.rebuild_index)
        return

    rank, local_rank, world = setup_distributed()

    # Distinct per rank so workers do not draw identical augmentation streams,
    # but derived from --seed so the run stays reproducible.
    seed = args.seed * 1000 + rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    cfg = build_cfg(args)
    if args.lr_scaling == "linear":
        cfg.lr = args.lr * world
    elif args.lr_scaling == "sqrt":
        cfg.lr = args.lr * math.sqrt(world)
    else:
        cfg.lr = args.lr

    if not args.output_dir:
        raise SystemExit("--output_dir (or $OUTPUT_DIR) is required")
    if not Path(cfg.kp_regressor_path).exists():
        raise SystemExit(
            f"keypoint regressor missing: {cfg.kp_regressor_path}\n"
            f"It ships with the repo; if absent, fit it once with\n"
            f"  python instanthmr_distill_train/train_distill_mhr_only.py "
            f"--fit-regressor --data_root {cfg.data_root}")
    if not Path(cfg.mhr_model_path).exists():
        raise SystemExit(f"MHR TorchScript rig missing: {cfg.mhr_model_path}")

    log("=" * 66)
    log(f"host {socket.gethostname()} | rank {rank}/{world} (local {local_rank})")
    log(f"data      {cfg.data_root}")
    log(f"out       {cfg.log_dir}")
    log(f"backbone  {cfg.backbone} | epochs {args.epochs} | "
        f"batch {cfg.batch_size}/gpu | lr {cfg.lr:.2e} ({args.lr_scaling} x{world})")
    log(f"preset    {args.preset}"
        + (f" | trunc<={cfg.geom_scale_max} trans={cfg.geom_trans} "
           f"occl_p={cfg.occl_p} jpeg_p={cfg.jpeg_p} "
           f"w_reproj={cfg.w_reproj} cam={cfg.cam_loss}"
           if args.preset != "baseline" else ""))
    log(f"losses    {args.losses}"
        + (f" | w_shape={cfg.w_shape} pose_split={cfg.pose_split} "
           f"pose_beta={cfg.pose_beta} kp3d={cfg.kp3d_loss}@{cfg.w_keypoints3d} "
           f"finger_w={cfg.finger_weight}"
           if args.losses != "legacy" else ""))
    log(f"seed      {args.seed} | mix {args.mix} | "
        f"samples/epoch {args.samples_per_epoch:,}")
    log("=" * 66)

    if args.dry_run:
        build_jz_loaders(cfg, args, rank, world)
        log("dry run OK")
    else:
        train(args, cfg, rank, local_rank, world)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
