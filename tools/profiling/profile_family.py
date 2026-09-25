import json, sys
from pathlib import Path
import qai_hub as hub
D = Path(sys.argv[1])
CPU = ["Samsung Galaxy S23", "Samsung Galaxy A73 5G", "Samsung Galaxy A53 5G",
       "Samsung Galaxy A14 5G", "Google Pixel 8"]
specs = dict(image=(1, 3, 224, 224), cliff_cond=(1, 3))
names = ["g8h_trained_b4", "random_b4", "random_b1", "random_b0"]
comp = {n: hub.submit_compile_job(model=str(D / f"{n}.onnx"), device=hub.Device("Samsung Galaxy S23"),
                                  input_specs=specs, options="--target_runtime onnx", name=f"t03-{n}")
        for n in names}
jobs = []
for n, cj in comp.items():
    m = cj.get_target_model()
    if m is None:
        print(json.dumps(dict(model=n, compile_failed=cj.get_status().message)), flush=True); continue
    jobs.append((n, "npu", "Samsung Galaxy S23",
                 hub.submit_profile_job(model=m, device=hub.Device("Samsung Galaxy S23"), name=f"t03-{n}-npu")))
    for d in CPU:
        jobs.append((n, "cpu", d, hub.submit_profile_job(model=m, device=hub.Device(d),
                     name=f"t03-{n}-cpu", options="--compute_unit cpu")))
out = []
for n, u, d, j in jobs:
    row = dict(model=n, unit=u, device=d, job=j.job_id)
    if j.wait().success:
        p = j.download_profile(); s = p["execution_summary"]
        units = {}
        for l in p.get("execution_detail", []):
            units[l.get("compute_unit")] = units.get(l.get("compute_unit"), 0) + 1
        row.update(ms=round(s["estimated_inference_time"] / 1000, 2),
                   peak_mb=round(s.get("estimated_inference_peak_memory", 0) / 2**20, 1), layers=units)
    else:
        row["error"] = j.get_status().message
    print(json.dumps(row), flush=True); out.append(row)
json.dump(out, open(D / "profile.json", "w"), indent=1)
