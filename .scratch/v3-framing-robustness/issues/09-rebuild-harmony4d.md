# 09: Rebuild Harmony4D as context crops

**What to build:** Harmony4D rejoins the corpus as context crops. Its original frames exist nowhere, so its zips are streamed scene by scene: download, undistort, build context crops, delete. The route depends on the `prepost` download test: HTTP 206 means the streaming job runs on Jean Zay (keeping no 4K originals); otherwise the same script runs on the laptop and the crops are rsynced as one tar.

**Blocked by:** 04, and the `prepost` curl result (job 190182)

**Status:** ready-for-human

- [ ] Route chosen from the `prepost` result and recorded
- [ ] Undistortion blockers in the undistortion notes handled before the first full scene
- [ ] 496,036 train crops (± source-missing), verified and archived to `$STORE`
