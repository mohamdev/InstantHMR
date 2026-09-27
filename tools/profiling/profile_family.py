"""Compile every .onnx in a directory for the S23 (ORT runtime) and profile it on
AI Hub: NPU on the Snapdragon 8-series phones, CPU-only on the reference phones.

    python tools/profiling/profile_family.py --dir /tmp/fam288 --image-size 288

Writes <dir>/profile.json (one row per model x device x unit) and prints rows.
"""
import argparse, json
from pathlib import Path
import qai_hub as hub

CPU = ["Samsung Galaxy S23", "Samsung Galaxy A73 5G", "Samsung Galaxy A53 5G",
       "Samsung Galaxy A14 5G", "Google Pixel 8"]
NPU = ["Samsung Galaxy S22 5G", "Samsung Galaxy S23", "Samsung Galaxy S24", "Samsung Galaxy S25"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--npu", default="S23", help="'S23' or 'all' Snapdragon 8-series phones")
    args = p.parse_args()
    s = args.image_size
    specs = dict(image=(1, 3, s, s), cliff_cond=(1, 3))
    npu = NPU if args.npu == "all" else ["Samsung Galaxy S23"]

    comp = {f.stem: hub.submit_compile_job(model=str(f), device=hub.Device("Samsung Galaxy S23"),
                                           input_specs=specs, options="--target_runtime onnx",
                                           name=f"fam-{f.stem}")
            for f in sorted(args.dir.glob("*.onnx"))}
    jobs = []
    for n, cj in comp.items():
        m = cj.get_target_model()
        if m is None:
            print(json.dumps(dict(model=n, compile_failed=cj.get_status().message)), flush=True)
            continue
        for d in npu:
            jobs.append((n, "npu", d, hub.submit_profile_job(model=m, device=hub.Device(d),
                                                             name=f"fam-{n}-npu")))
        for d in CPU:
            jobs.append((n, "cpu", d, hub.submit_profile_job(model=m, device=hub.Device(d),
                         name=f"fam-{n}-cpu", options="--compute_unit cpu")))
    out = []
    for n, u, d, j in jobs:
        row = dict(model=n, image_size=s, unit=u, device=d, job=j.job_id)
        if j.wait().success:
            prof = j.download_profile()
            summ = prof["execution_summary"]
            units = {}
            for layer in prof.get("execution_detail", []):
                units[layer.get("compute_unit")] = units.get(layer.get("compute_unit"), 0) + 1
            row.update(ms=round(summ["estimated_inference_time"] / 1000, 2),
                       peak_mb=round(summ.get("estimated_inference_peak_memory", 0) / 2**20, 1),
                       layers=units)
        else:
            row["error"] = j.get_status().message
        print(json.dumps(row), flush=True)
        out.append(row)
    json.dump(out, open(args.dir / "profile.json", "w"), indent=1)


if __name__ == "__main__":
    main()
