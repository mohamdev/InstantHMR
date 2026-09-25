# 05: Input size as a checkpoint setting (prefactor)

**What to build:** one input-size setting (default 224) drives every place that builds the network input: the training dataset, 3DPW validation, the benchmark harness and the inference package. It is saved in the run config, recovered by config-from-checkpoint (a checkpoint without it means 224), and stamped into the ONNX metadata beside the CLIFF form, so the phone app can read its crop size from the model file.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] baseline and v2 augmented samples are bit-identical against `git show HEAD`, compared on alternating calls
- [ ] g8h_s0 evaluated through the harness gives the same 3DPW test J14 and EMDB-1 numbers as recorded, to the decimal
- [ ] A random-weight model configured for 288 runs end to end through the 3DPW and EMDB harnesses, and its ONNX metadata reads 288
- [ ] `ddp_smoke` passes at 224 and at 288
