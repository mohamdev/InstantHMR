# 08: Rebuild coco, sa1b, aic and mpii as context crops on Jean Zay

**What to build:** the four splits whose original frames are on the cluster, rebuilt with the array build job at context 2.0 / 448 into new directories that the baseline and v2 presets never pick up, then verified and archived.

**Blocked by:** 01, 04

**Status:** ready-for-human

- [ ] Launch commands printed in the build job's form, with the new flags and output directories
- [ ] Crop counts match the current splits minus images that are missing at source (report the difference)
- [ ] Verification passes per split (reprojection RMS ~0.000 px, zero missing or orphan)
- [ ] Each split archived as one tar to `$STORE`
- [ ] `$SCRATCH` usage recorded (expected ~250 GB total with ticket 09)
