# 01: Archive the AIC and MPII original frames to $STORE

**What to build:** every built split has a tar in `$STORE`, so the `$SCRATCH` 30-day purge (AIC and MPII originals last read 1–2 Sept 2026, due around 1–2 Oct) costs a restore instead of a rebuild. The archive template's placeholder account has been fixed to `vsi@v100`; job 189727 was submitted with `REPO=$PWD` on the free `archive` partition.

**Blocked by:** None (can start immediately; already queued as job 189727).

**Status:** ready-for-human

- [ ] `hmr_archive_<jobid>.out` ends with an `ls -lh` listing five tars: aic, coco, mpii, sa1b, harmony4d
- [ ] The aic and mpii tars hold real frames (hard links/copies), not links: `tar tvf` shows regular files under `original_images/`
- [ ] If the job timed out on sa1b, the partial sa1b tar was deleted before resubmitting (the script skips any existing tar, including a half-written one)
