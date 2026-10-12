# 12: Build EgoHumans as context crops on Jean Zay

**What to build:** EgoHumans (`egohumans_train`) joins the corpus as context crops (2.0x box, 448 px), the way Harmony4D does in ticket 09: download on a Jean Zay LOGIN node, undistort and build on compute nodes. It is ~272k annotated persons over ~268k 4K frames from 89 takes in 7 multi-person sports sequences (tagging, lego, fencing, basketball, volleyball, badminton, tennis): close interactions and wide framings the current corpus is thin on.

Known facts (STATUS.md 6-8, `datasets_pipeline/registry.py`, `undistort.md`):
- **Annotations are missing everywhere.** The `egohumans_train` parquets failed to download on Jean Zay (Xet proxy error, 2026-09-03) and are not on the laptop either. Hugging Face file downloads work from a login node (Harmony4D, 2026-09-25), so fetch them there first.
- **Images:** Google Drive folder `1JD963urzuzV_R_6FOVOtlx8UupwUuknR`, one `.tar.gz` per take under `<sequence>/<take>.tar.gz`. gdown file downloads ran at ~20 MB/s from Jean Zay, but folder *listing* got rate-limited (403) after repeated requests: list each of the 7 sequence folders separately, ~15 min apart if refused.
- **Only the referenced takes are needed** (~107 GB of the advertised 550 GB). Shopping list: `datasets_pipeline/probe_split.py --annotation-dir $ANN_DIR/egohumans_train --list-groups 2`, intersected with the Drive listing.
- **Unpack with `tar xzf <take>.tar.gz --strip-components=8`. Not optional:** the tarballs carry an 8-level absolute prefix. Only `exo/` cameras are referenced.
- **Undistort** with `sam-3d-body/data/scripts/egohumans/undistort_images.py`. It shares the vendored `lib/` with Harmony4D, so the pycolmap >= 3 patch (ticket 09) applies; check it in one take first.
- `stream_archives.py` does not support `gdrive`; write a per-take job like `22_h4d_scene.slurm` instead.
- Keep the output out of `data_ctx` while a generation trains from it (pair-index fingerprint, see ticket 09 / STATUS).

**Blocked by:** 09 (proves the login-download / compute-undistort route and the patched `lib/` on Jean Zay).

**Status:** ready-for-agent (job script + one verified take), then ready-for-human (full download and launch)

- [ ] `egohumans_train` parquets fetched on a login node; row count recorded
- [ ] Shopping list of referenced takes built; download size recorded
- [ ] One take downloaded, unpacked, undistorted and built; its crops checked (labels reproject to 0.000 px; a few crops visually on the person; no undistortion exceptions)
- [ ] Per-take Jean Zay job written and the remaining takes built
- [ ] Final crop count within a few hundred of the annotation row count (minus missing frames), recorded in STATUS
- [ ] Archived to `$STORE` (`sam3d_gt_egohumans_ctx.tar`)
