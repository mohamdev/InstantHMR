# 08: Rebuild coco, sa1b, aic and mpii as context crops on Jean Zay

**What to build:** the four splits whose original frames are on the cluster, rebuilt with the array build job at context 2.0 / 448 into new directories that the baseline and v2 presets never pick up, then verified and archived.

**Blocked by:** 01, 04

**Status:** done (2026-10-04); archive to STORE submitted

- [x] Launch commands printed in the build job's form, with the new flags and output directories
- [x] Crop counts match the current splits minus images that are missing at source (report the difference)
- [x] Verification passes per split (reprojection RMS ~0.000 px, zero missing or orphan)
- [x] Each split archived as one tar to `$STORE`
- [x] `$SCRATCH` usage recorded (expected ~250 GB total with ticket 09)

## Answer

Context crops (2.0x box, 448 px) in `$SCRATCH/instanthmr/data_ctx/sam3d_gt_{coco,mpii,aic,sa1b}`: 39,665 / 7,135 / 287,391 / 3,401,720 -- every split equal to the 224 corpus.

What it took:
- `prepost` had no usable node for a week (pp1/pp4 draining), so builds moved to `21_build_gpu.slurm` (gpu_p13, qos_gpu-dev, 2 h, 10 processes, PART/PARTS to split one split across parallel jobs).
- MPII photos exist only under `$DATA_ROOT/sam3d_gt_mpii/original_images`, not `$IMG_ROOT/mpii`.
- SA-1B jobs died OUT_OF_MEMORY at ~40 min: `build_split.py` cached every decoded frame of a parquet shard. Bounded to 8 entries (output byte-identical, `verify_context_crops.py` PASS); the next round completed in 36 min.
- Every kill was followed by a truncated-npz check on files written in the 3 min before it; none found.
