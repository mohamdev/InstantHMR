# 09: Rebuild Harmony4D as context crops

**What to build:** Harmony4D rejoins the corpus as context crops. Its original frames exist nowhere, so its zips are streamed scene by scene: download, undistort, build context crops, delete. The route depends on the `prepost` download test: HTTP 206 means the streaming job runs on Jean Zay (keeping no 4K originals); otherwise the same script runs on the laptop and the crops are rsynced as one tar.

**Blocked by:** None (04 done; route decided)

**Status:** ready-for-human -- route 2 (laptop)

- [x] Route chosen from the `prepost` result and recorded
- [ ] Undistortion blockers in the undistortion notes handled before the first full scene
- [ ] 496,036 train crops (± source-missing), verified and archived to `$STORE`

## Route decided (2026-09-27): laptop

`prepost` job 226661 returned `HTTP 302, 0 bytes`: Hugging Face's redirect to its file host `us.aws.cdn.hf.co` cannot be followed from Jean Zay compute nodes (the login node could, 2026-09-25; same failure as STATUS 7.1 on 2026-09-03). So stream on the laptop, scene by scene (65 GB free: download one zip, undistort, build context crops with `--context 2.0 --body-size 448 --output-dir`, delete), then rsync one tar of crops to Jean Zay's context data root.
