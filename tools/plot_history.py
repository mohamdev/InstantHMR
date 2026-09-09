#!/usr/bin/env python3
"""Plot the per-epoch curves in one or more runs' `history.jsonl`.

    python tools/plot_history.py instanthmr_distill_train/runs/g6bv_s
    python tools/plot_history.py .../g6bv_s .../g6vv_s -o /tmp/gen6.png
    python tools/plot_history.py $RUNS_DIR/g6*_s --tail 60

`summarize_runs.py` answers "which run is ahead". This answers "is it still
going down, and is it healthy" — the question you have to settle before
resubmitting a chained run for more epochs.

Six panels:

  J14 PA-MPJPE   the reported metric. EMA solid, RAW faint. The star is the
                 best epoch; if it sits at the right-hand edge the run has not
                 plateaued.
  J14 MPJPE      the same but without Procrustes, so it still sees body-size
                 and depth error that PA-MPJPE removes.
  train total    teacher agreement on the training mixture.
  learning rate  OneCycleLR peaks at 10% of the planned epochs and anneals to
                 ~0 at the end. A run stopped mid-anneal has left the whole
                 low-LR refinement phase unspent, which is where the last few
                 mm usually come from — read this panel before concluding a
                 flat metric curve means "converged".
  loss terms     every term in the epoch record, normalised to its own epoch-1
                 value so terms three orders of magnitude apart share an axis.
                 `verts` appears only on --w-verts runs.
  health         EMA/RAW ratio (left) and skipped steps per epoch (right).
                 Ratio below 1 means the average is ahead of the raw weights,
                 which is the normal state. A spike above 1 is the cheapest
                 detector there is for an intra-epoch excursion the loss guard
                 never saw — see docs/todo.md item 5b.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # write a file; there is no display on a cluster
import matplotlib.pyplot as plt

# DistillConfig.anomaly_skip_patience: consecutive anomalous steps that trigger
# an EMA rollback. An epoch reporting exactly this many skips took one.
ROLLBACK_SKIPS = 50


def find_history(p: Path) -> Path | None:
    """history.jsonl for a run, whether given the run dir or its parent.

    52_train_ddp.slurm writes to $RUNS_DIR/<name>, and train_distill_jz.py
    creates <name>/ again underneath, so the file usually sits one level deeper
    than the directory you name on the command line.
    """
    if p.is_file():
        return p
    hits = sorted(p.glob("**/history.jsonl"))
    return hits[0] if hits else None


def load(path: Path, tail: int | None):
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return rows[-tail:] if tail else rows


def series(rows, *keys, default=None):
    """rows[i][k1][k2]... as a list, with None where the record is missing it."""
    out = []
    for r in rows:
        v = r
        for k in keys:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        out.append(default if v is None else v)
    return out


def plot(runs, out: Path, target_epochs: int | None):
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    ax_pa, ax_mj, ax_tr, ax_lr, ax_terms, ax_health = axes.flat
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, rows, cfg) in enumerate(runs):
        c = colours[i % len(colours)]
        ep = [r["epoch"] + 1 for r in rows]
        label = name
        if cfg:
            label += f"  ({cfg.get('preset', '?')}"
            if cfg.get("w_verts"):
                label += f", w_verts={cfg['w_verts']}"
            label += ")"

        pa_e = series(rows, "dpw_ema", "J14_PA_MPJPE")
        pa_r = series(rows, "dpw_raw", "J14_PA_MPJPE")
        if any(v is not None for v in pa_e):
            ax_pa.plot(ep, pa_e, color=c, lw=1.8, label=label)
            ax_pa.plot(ep, pa_r, color=c, lw=0.8, alpha=0.35)
            best = min(v for v in pa_e if v is not None)
            bi = pa_e.index(best)
            ax_pa.plot(ep[bi], best, "*", color=c, ms=15, zorder=5)
            ax_pa.annotate(f"{best:.2f} @{ep[bi]}", (ep[bi], best), color=c,
                           fontsize=8, xytext=(4, -11), textcoords="offset points")

        mj = series(rows, "dpw_ema", "J14_MPJPE")
        if any(v is not None for v in mj):
            ax_mj.plot(ep, mj, color=c, lw=1.6, label=label)

        ax_tr.plot(ep, series(rows, "train", "total"), color=c, lw=1.6, label=label)
        ax_lr.plot(ep, series(rows, "lr"), color=c, lw=1.6, label=label)

        # Terms live on wildly different scales (simcc ~2.2, reproj ~5e-4), so
        # each is divided by its own first value: the shape is what matters.
        for j, k in enumerate(sorted(rows[0].get("train", {}))):
            if k == "total":
                continue
            v = series(rows, "train", k)
            if not any(v) or v[0] in (None, 0):
                continue
            ax_terms.plot(ep, [x / v[0] for x in v], lw=1.3,
                          ls=["-", "--", ":", "-."][i % 4],
                          color=colours[(j + 2) % len(colours)],
                          label=f"{k} ({name})" if len(runs) > 1 else k)

        ratio = [(a / b) if (a and b) else None for a, b in zip(pa_e, pa_r)]
        if any(v is not None for v in ratio):
            ax_health.plot(ep, ratio, color=c, lw=1.4, label=f"EMA/RAW {name}")
        skipped = series(rows, "skipped", default=0)
        ax_health.twinx().plot(ep, skipped, color=c, lw=0.9, alpha=0.4, ls=":")

        # An EMA rollback halves the LR and clears the Adam moments, so it
        # permanently reshapes the schedule -- the discontinuity it leaves in
        # the LR panel is otherwise unexplainable. It fires at exactly
        # anomaly_skip_patience (50) consecutive skips, which is why that count
        # in a single epoch is its fingerprint.
        for e, s in zip(ep, skipped):
            if s >= ROLLBACK_SKIPS:
                for ax in (ax_pa, ax_lr):
                    ax.axvline(e, color=c, ls=":", lw=1.2, alpha=0.7)
                ax_lr.annotate(f"EMA rollback ep{e}\nLR x0.5", (e, max(series(rows, "lr"))),
                               color=c, fontsize=7, xytext=(4, -20),
                               textcoords="offset points")

    for ax, title, ylab in (
            (ax_pa, "3DPW-val J14 PA-MPJPE  (EMA bold, RAW faint)", "mm"),
            (ax_mj, "3DPW-val J14 MPJPE (EMA)", "mm"),
            (ax_tr, "train total loss", "loss"),
            (ax_lr, "learning rate", "lr"),
            (ax_terms, "train loss terms, relative to epoch 1", "x epoch-1"),
            (ax_health, "health: EMA/RAW ratio (solid), skipped steps (dotted, right)", "ratio")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)

    if target_epochs:
        for ax in (ax_pa, ax_mj, ax_tr, ax_lr):
            ax.axvline(target_epochs, color="k", ls="--", lw=0.8, alpha=0.5)
        ax_lr.annotate(f"planned end (ep {target_epochs})", (target_epochs, 0),
                       fontsize=8, rotation=90, va="bottom", ha="right", alpha=0.6)

    ax_health.axhline(1.0, color="k", lw=0.7, alpha=0.4)
    for ax in (ax_pa, ax_mj, ax_tr, ax_lr, ax_health):
        ax.legend(fontsize=8)
    ax_terms.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path,
                    help="run directories (or history.jsonl paths)")
    ap.add_argument("-o", "--out", type=Path, default=Path("history.png"))
    ap.add_argument("--tail", type=int, default=None,
                    help="plot only the last N epochs")
    args = ap.parse_args()

    loaded = []
    for p in args.runs:
        h = find_history(p)
        if h is None:
            print(f"  no history.jsonl under {p} — skipped")
            continue
        rows = load(h, args.tail)
        cfg_p = h.parent / "run_config.json"
        cfg = json.loads(cfg_p.read_text()) if cfg_p.is_file() else {}
        loaded.append((p.name, rows, cfg))
        pa = [r.get("dpw_ema", {}).get("J14_PA_MPJPE") for r in rows if r.get("dpw_ema")]
        best = min(pa) if pa else float("nan")
        print(f"  {p.name:10s} {len(rows):4d} epochs, best EMA J14 PA {best:6.2f} mm"
              f"  (target {cfg.get('epochs', '?')})")
    if not loaded:
        raise SystemExit("nothing to plot")

    targets = {c.get("epochs") for _, _, c in loaded if c.get("epochs")}
    plot(loaded, args.out, targets.pop() if len(targets) == 1 else None)


if __name__ == "__main__":
    main()
