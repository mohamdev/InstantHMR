# 07: v3 preset, proven on local data

**What to build:** `--preset v3` trains from context crops. Per sample it draws the effective person box log-uniformly between 0.6x and 2.0x the tight box, shifts its centre by up to +/-15% of the box, clamps the window inside the stored context, and maps the context crop to the network input in one resampling step (with v2's rotation and flip). The CLIFF vector (perspective-correct form) describes the sampled box. The bounds are config values with these defaults.

**Blocked by:** 04, 05

**Status:** done (2026-09-25), uncommitted

- [x] A small local split rebuilt with ticket 04's builder (context 2.0 / 448) trains under v3 at 224 and at 288
- [x] Verification on real samples: 2D labels land on the right pixels; CLIFF crop-centre error at the rounding floor (mean / p95 / max px, as in the crop-centre-fix check); no black pixel inside the stored context; box-scale draws cover 0.6–2.0x; output size equals the configured input size
- [x] baseline and v2 samples bit-identical against HEAD (alternating calls)
- [x] The v3 dataset refuses a split without context geometry, with a message naming the split
- [x] `ddp_smoke` passes for v3 at 224 and at 288
- [x] Files that must be rsynced to the cluster listed in the status doc

## Answer

Implemented as specified, with two decisions taken with the user (2026-09-25):
- **Clamp (Q22, option b):** keep the rotated window inside the stored square only on sides where the square ends inside the photo. Tail restored: P(>1.6x) 17.0% (unclamped 18.5%, all-sides clamp 10.5%), P(>1.8x) 6.6% (8.8% / 1.9%), loose windows keep their shift.
- **geom_p (Q23):** 80/20 as v2; the 20% identity windows keep root-rotation supervision. Revisit only if v3 helps.

Evidence: `tools/verify_v3_preset.py` PASS (see README section `--preset v3`). `ddp_smoke --context-crops` passes at 224 and 288. A 1-epoch `train_distill_jz.py --preset v3 --image-size 288` run on the local context split completed and recorded preset / image_size / v3 fields in run_config.json. Rsync list and cluster checks: STATUS.md, "Code ready, not launched".

Findings:
- 95% of 2.0x context squares (AIC, COCO) run past the frame; windows >1.5x are ~38% black on average (frame edge, representative of deployment). Harmony4D / SA-1B may differ; measure after the rebuild.
- Naming: `--preset v3` is unrelated to the old `v3_s{0,1,2}` runs (v2 + rebalance). Noted in STATUS.
- `apply_v3_preset` forces `cliff_focal`.
