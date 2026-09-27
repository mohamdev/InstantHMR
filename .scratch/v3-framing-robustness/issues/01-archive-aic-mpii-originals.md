# 01: Archive the AIC and MPII original frames to $STORE

**What to build:** every built split has a tar in `$STORE`, so the `$SCRATCH` 30-day purge (AIC and MPII originals last read 1–2 Sept 2026, due around 1–2 Oct) costs a restore instead of a rebuild. The archive template's placeholder account has been fixed to `vsi@v100`; job 189727 was submitted with `REPO=$PWD` on the free `archive` partition.

**Blocked by:** None (can start immediately; already queued as job 189727).

**Status:** done (2026-09-26)

- [x] `hmr_archive_<jobid>.out` ends with an `ls -lh` listing five tars: aic, coco, mpii, sa1b, harmony4d
- [x] The aic and mpii tars hold real frames (hard links/copies), not links: `tar tvf` shows regular files under `original_images/`
- [x] If the job timed out on sa1b, the partial sa1b tar was deleted before resubmitting (the script skips any existing tar, including a half-written one)

## Answer

Job 189727 wrote five split tars, but `40_archive.slurm`'s unanchored `--exclude=images` also dropped every nested `images/` folder -- i.e. all AIC and MPII original photos (0 in either tar; mpii.tar 228 MB). Fixed in the template (exclude only a top-level `images` symlink, anchored; tested on a fake tree) and rsynced. The photos were archived separately by job 226766: `sam3d_gt_aic_originals.tar` 13 GB, `sam3d_gt_mpii_originals.tar` 2.6 GB. `sam3d_gt_harmony4d.tar` verified to hold all 496,036 crops. STORE total 155 GB.
