# 07: v3 preset, proven on local data

**What to build:** `--preset v3` trains from context crops. Per sample it draws the effective person box log-uniformly between 0.6x and 2.0x the tight box, shifts its centre by up to +/-15% of the box, clamps the window inside the stored context, and maps the context crop to the network input in one resampling step (with v2's rotation and flip). The CLIFF vector (perspective-correct form) describes the sampled box. The bounds are config values with these defaults.

**Blocked by:** 04, 05

**Status:** ready-for-agent

- [ ] A small local split rebuilt with ticket 04's builder (context 2.0 / 448) trains under v3 at 224 and at 288
- [ ] Verification on real samples: 2D labels land on the right pixels; CLIFF crop-centre error at the rounding floor (mean / p95 / max px, as in the crop-centre-fix check); no black pixel inside the stored context; box-scale draws cover 0.6–2.0x; output size equals the configured input size
- [ ] baseline and v2 samples bit-identical against HEAD (alternating calls)
- [ ] The v3 dataset refuses a split without context geometry, with a message naming the split
- [ ] `ddp_smoke` passes for v3 at 224 and at 288
- [ ] Files that must be rsynced to the cluster listed in the status doc
