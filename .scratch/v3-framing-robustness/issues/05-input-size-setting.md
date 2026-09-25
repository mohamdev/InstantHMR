# 05: Input size as a checkpoint setting (prefactor)

**What to build:** one input-size setting (default 224) drives every place that builds the network input: the training dataset, 3DPW validation, the benchmark harness and the inference package. It is saved in the run config, recovered by config-from-checkpoint (a checkpoint without it means 224), and stamped into the ONNX metadata beside the CLIFF form, so the phone app can read its crop size from the model file.

**Blocked by:** None (can start immediately).

**Status:** done (2026-09-25), uncommitted

- [x] baseline and v2 augmented samples are bit-identical against `git show HEAD`, compared on alternating calls
- [x] g8h_s0 evaluated through the harness gives the same 3DPW test J14 and EMDB-1 numbers as recorded, to the decimal
- [x] A random-weight model configured for 288 runs end to end through the 3DPW and EMDB harnesses, and its ONNX metadata reads 288
- [x] `ddp_smoke` passes at 224 and at 288

## Answer

`--image-size` on the DDP trainer (multiple of 32), recorded in run_config.json. `config_from_checkpoint` reads the size from the weights (`mem_pos_embed` grid, 7x7 = 224, 9x9 = 288) and refuses a run_config.json that contradicts it. The dataset (both trainers and the self/overfit tests), `val3dpw`, the 3DPW / EMDB / SMPL-fit harnesses, `eval_all.py` grouping, `pth_to_onnx.py` metadata (`image_size`), `instanthmr.inference` and `ddp_smoke.py` all follow it.

Evidence: `tools/verify_input_size.py` PASS (baseline and v2 0/24 samples differ from the pre-change trainer; labels and CLIFF identical at 224 and 288; 288 checkpoint round-trips through config, export metadata, inference on a real frame and both harnesses). g8h_s0 full regression identical to 4 decimals (3DPW test J14+adapter PA 40.7983, EMDB SMPL24+adapter 51.9492). `ddp_smoke` passes at 224 and 288; `--self-test` passes.

Open, outside this repo (story 21 is NOT satisfied end to end yet):
- neurocomp-app's own exporter (`tools/modelexport/pose.py`) hard-codes `INPUT_SHAPES` 224 and its metadata stamp clears all props and never writes `image_size`.
- the Android engine hard-codes `POSE_INPUT_SIZE = 224` in `Preprocess.kt`.
Both must read `image_size` before a non-224 model can ship. File this in the phone-app tracker.

Note for ticket 10: `RandomPixelate` uses absolute resolutions (32-112 px), so at 288 it degrades relatively harder than at 224, which slightly confounds a 224-vs-288 comparison.
