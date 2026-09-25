# 04: Context-crop builder

**What to build:** the crop builder can store a context crop, a square covering a chosen multiple of the person's tight box (default 1.2, today's) at a chosen size. It records the stored square in original-frame coordinates next to the existing 1.2x square, so any window can be recomputed exactly. The rebuild setting is context 2.0 at 448 px.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] With default flags the builder's output (crops and npz) is byte-identical to the current builder's on a real local split
- [ ] With context 2.0 / 448, the 1.2x window cut from the context crop and resized to 224 matches today's 224 crop of the same person (PSNR reported, mean and min)
- [ ] The recorded geometry maps the annotation targets' 2D keypoints onto the context crop to 0.000 px
- [ ] Where the context square leaves the frame, the existing frame-edge padding applies and is flagged as such in the npz
- [ ] Works through the array build job and the streaming script unchanged apart from the new flags
