# 04: Context-crop builder

**What to build:** the crop builder can store a context crop, a square covering a chosen multiple of the person's tight box (default 1.2, today's) at a chosen size. It records the stored square in original-frame coordinates next to the existing 1.2x square, so any window can be recomputed exactly. The rebuild setting is context 2.0 at 448 px.

**Blocked by:** None (can start immediately).

**Status:** done (2026-09-25), uncommitted

- [x] With default flags the builder's output (crops and npz) is byte-identical to the current builder's on a real local split
- [x] With context 2.0 / 448, the 1.2x window cut from the context crop and resized to 224 matches today's 224 crop of the same person (PSNR reported, mean and min)
- [x] The recorded geometry maps the annotation targets' 2D keypoints onto the context crop to 0.000 px
- [x] Where the context square leaves the frame, the existing frame-edge padding applies and is flagged as such in the npz
- [x] Works through the array build job and the streaming script unchanged apart from the new flags

## Answer

`--context C` on the builder (default 1.2 = unchanged); context builds record `bbox_context` and `context_padded`, need an explicit `--output-dir`, and refuse to mix with a folder of the other kind. `tools/verify_context_crops.py` on 200 real AIC rows: default byte-identical to HEAD; geometry within 0.5 px of the exact square (int() rounding); 1.2x window vs today's crop 42.2 dB mean / 33.4 min after a sigma-1 blur, 0.094 px max shift; keypoints unchanged, none lost; context_padded 0 wrong.

Notes for later tickets:
- At 2.0x, 195/200 AIC context squares reach past the frame edge, so most context crops contain some black frame-edge padding. Ticket 07 must treat it as real frame edge (deployment crops pad the same way), and may want to report how often v3 windows touch it.
- 42/200 rows have visible keypoints outside today's 1.2x crop: the annotation box does not always enclose all keypoints. Pre-existing.
- Today's 224 crops of people larger than ~400 px are aliased (INTER_LINEAR downscale by 3-4x). Context crops alias less. Default left unchanged.
- Tickets 08/09 must pass `--output-dir` for every context split (the builder now refuses otherwise).
