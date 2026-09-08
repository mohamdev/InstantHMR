#!/usr/bin/env python3
"""One command per checkpoint: run every benchmark route, print one table.

    python benchmark/eval_all.py --ckpt <run>/best_student_model_v3.pth

Runs four evaluations and consolidates them:

    3DPW test   J14 (H36M) + adapter        eval_3dpw_ckpt.py
    3DPW test   SMPL24 + PVE, J14 adapter-free   eval_smpl_fit.py
    EMDB-1      SMPL24 + adapter            eval_emdb_ckpt.py
    EMDB-1      SMPL24 + PVE                eval_smpl_fit.py

``--table-only`` skips the compute and re-prints the table from whatever is
already in ``benchmark/results/``, which is what you want after a run or when
adding a checkpoint to an existing comparison.

Paths default to this machine's layout; override per site. See METRICS.md for
what each row means and which space it is computed in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[])
    p.add_argument("--emdb-root", default="/home/madjel/Downloads/EMDB_root")
    p.add_argument("--sequence-dir",
                   default="/home/madjel/Downloads/sequenceFiles/sequenceFiles")
    p.add_argument("--image-root", default="/home/madjel/Downloads/imageFiles")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fit-iters", type=int, default=400)
    p.add_argument("--fit-batch", type=int, default=128)
    p.add_argument("--table-only", action="store_true")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["3dpw-adapter", "3dpw-mesh", "emdb-adapter", "emdb-mesh"])
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def group_by_conditioning(ckpts: list[str]) -> list[list[str]]:
    """Split checkpoints into groups that can share one crop pipeline.

    ``eval_3dpw_ckpt.py`` and ``eval_emdb_ckpt.py`` build the dataset once for
    all their checkpoints, so they refuse a mix of ``--cliff-focal`` settings --
    the conditioning vector means something different in each, and feeding the
    wrong one raises nothing, it just mis-places the person in depth. Runs from
    before and after 2026-09-06 therefore have to go in separate invocations,
    which this works out automatically instead of making the caller notice.
    """
    import torch
    sys.path.insert(0, str(ROOT / "instanthmr_distill_train"))
    import train_distill_mhr_only as T
    groups: dict[bool, list[str]] = {}
    for c in ckpts:
        st = torch.load(c, map_location="cpu", weights_only=False)
        cfg, _ = T.config_from_checkpoint(st.get("model_state_dict", st), c)
        groups.setdefault(bool(cfg.cliff_focal), []).append(c)
        del st
    return list(groups.values())


def merge_json(parts: list[Path], dest: Path) -> None:
    """Concatenate per-group reports into the one file the table reads."""
    out: dict = {}
    for f in parts:
        if f.is_file():
            out.update(json.loads(f.read_text()))
            f.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))


def _load(name: str) -> dict:
    f = RESULTS / name
    return json.loads(f.read_text()) if f.is_file() else {}


def _run_name(key: str) -> str:
    parts = key.split("|")[0].split("/")
    return parts[-3] if len(parts) >= 3 else key


def table() -> None:
    """The full comparison, one row per checkpoint per dataset."""
    dpw_ad = _load("3dpw_test_alladapter.json")
    dpw_me = _load("3dpw_test_smplfit.json")
    emdb_ad = _load("emdb1_alladapter.json")
    emdb_me = _load("emdb1_smplfit_gt-joints.json")

    runs: list[str] = []
    for src in (dpw_me, emdb_me, dpw_ad, emdb_ad):
        for k in src:
            n = _run_name(k)
            if n not in runs:
                runs.append(n)

    def pick(src, name, *fields):
        for k, v in src.items():
            if _run_name(k) == name:
                out = []
                for f in fields:
                    if f.startswith("@"):        # a joint_set row
                        r = next((r for r in v.get("results", [])
                                  if r["joint_set"] == f[1:]), None)
                        out.append(r)
                    else:
                        out.append(v.get(f))
                return out
        return [None] * len(fields)

    def fmt(v, nd=2):
        return "-" if v is None else f"{v:.{nd}f}"

    print("\n" + "=" * 108)
    print("3DPW test  —  all 24 sequences, published-protocol GT")
    print("=" * 108)
    print(f"{'run':<12}{'ep':>5} | {'J14 PA':>8}{'J14 MPJPE':>11}  (adapter)"
          f" | {'J14 PA':>8}{'J14 MPJPE':>11}  (via mesh fit)"
          f" | {'SMPL24 PA':>10}{'MPJPE':>8}{'PVE':>8}")
    for n in runs:
        (ep,) = pick(dpw_me, n, "epoch")
        a, = pick(dpw_ad, n, "@J14+adapter")
        m = pick(dpw_me, n, "J14_h36m_viamesh_PA_MPJPE_mm", "J14_h36m_viamesh_MPJPE_mm",
                 "PA_MPJPE_mm", "MPJPE_mm", "PVE_mm")
        print(f"{n:<12}{fmt(ep,0):>5} | "
              f"{fmt(a['PA_MPJPE_mm']) if a else '-':>8}"
              f"{fmt(a['MPJPE_mm']) if a else '-':>11}           "
              f" | {fmt(m[0]):>8}{fmt(m[1]):>11}                "
              f" | {fmt(m[2]):>10}{fmt(m[3]):>8}{fmt(m[4]):>8}")

    print("\n" + "=" * 108)
    print("EMDB-1  —  all 17 sequences, the 'EMDB (24)' protocol")
    print("=" * 108)
    print(f"{'run':<12}{'ep':>5} | {'SMPL24 PA':>10}{'MPJPE':>9}  (joint adapter)"
          f" | {'SMPL24 PA':>10}{'MPJPE':>9}{'PVE':>8}{'PA-PVE':>9}  (mesh fit)")
    for n in runs:
        (ep,) = pick(emdb_me, n, "epoch")
        a, = pick(emdb_ad, n, "@SMPL24+adapter")
        m = pick(emdb_me, n, "PA_MPJPE_mm", "MPJPE_mm", "PVE_mm", "PA_PVE_mm")
        print(f"{n:<12}{fmt(ep,0):>5} | "
              f"{fmt(a['PA_MPJPE_mm']) if a else '-':>10}"
              f"{fmt(a['MPJPE_mm']) if a else '-':>9}                 "
              f" | {fmt(m[0]):>10}{fmt(m[1]):>9}{fmt(m[2]):>8}{fmt(m[3]):>9}")
    print("=" * 108)
    print("Both J14 columns convert MHR->H36M; neither is adapter-free. We predict")
    print("MHR, so a conversion is mandatory -- the joint adapter is the primary row.")
    print("PVE is pelvis-aligned (the published definition); PA-PVE is the")
    print("Procrustes variant and is NOT what papers call PVE. See METRICS.md.")
    print("Conversion floors — mesh: 11.44 MPJPE / 10.66 PA / 13.61 PVE;")
    print("                 adapter: 25.11 MPJPE / 18.77 PA / no PVE.\n")


def main():
    args = parse_args()
    if not args.table_only:
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --table-only")
        py = [sys.executable]
        common = ["--stride", str(args.stride)]
        groups = group_by_conditioning(args.ckpt)
        if "3dpw-adapter" not in args.skip:
            parts = []
            for gi, grp in enumerate(groups):
                dst = RESULTS / f"_dpw_ad_{gi}.json"
                run(py + ["benchmark/eval_3dpw_ckpt.py", "--ckpt", *grp,
                          "--sequence-dir", args.sequence_dir,
                          "--image-root", args.image_root, "--split", "test",
                          "--adapter", "benchmark/results/adapter_j14_h36m_teacher.npz",
                          "--out", str(dst)] + common)
                parts.append(dst)
            merge_json(parts, RESULTS / "3dpw_test_alladapter.json")
        if "emdb-adapter" not in args.skip:
            parts = []
            for gi, grp in enumerate(groups):
                dst = RESULTS / f"_emdb_ad_{gi}.json"
                run(py + ["benchmark/eval_emdb_ckpt.py", "--ckpt", *grp,
                          "--emdb-root", args.emdb_root, "--out", str(dst)] + common)
                parts.append(dst)
            merge_json(parts, RESULTS / "emdb1_alladapter.json")
        mesh = ["--fit-iters", str(args.fit_iters), "--fit-batch", str(args.fit_batch)]
        if "3dpw-mesh" not in args.skip:
            run(py + ["benchmark/eval_smpl_fit.py", "--dataset", "3dpw",
                      "--split", "test", "--ckpt", *args.ckpt,
                      "--sequence-dir", args.sequence_dir,
                      "--image-root", args.image_root,
                      "--out", "benchmark/results/3dpw_test_smplfit.json"]
                + common + mesh)
        if "emdb-mesh" not in args.skip:
            run(py + ["benchmark/eval_smpl_fit.py", "--dataset", "emdb",
                      "--ckpt", *args.ckpt, "--emdb-root", args.emdb_root,
                      "--out", "benchmark/results/emdb1_smplfit_gt-joints.json"]
                + common + mesh)
    table()


if __name__ == "__main__":
    main()
