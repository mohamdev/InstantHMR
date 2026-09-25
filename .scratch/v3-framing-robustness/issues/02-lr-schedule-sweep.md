# 02: Learning-rate x schedule-length sweep on the reference model

**What to build:** find out whether the underfitting reference model (train loss 2.340 above val 2.302 at epoch 299; 3DPW val still falling as the LR reached zero) gains from a better recipe. 2x2 grid on g8h, current corpus, baseline preset, 1 seed per cell: peak LR {6e-4, 1.2e-3} x length {300, 600 epochs}. The (6e-4, 300) cell is the existing g8h_s0, so three new runs. Launched through the usual Slurm launcher with the generation-8 flags plus the LR/epoch overrides.

**Blocked by:** None (can start immediately).

**Status:** ready-for-human

- [ ] Launch commands printed in the launcher's form (`RUN_NAME=... SEED=... EPOCHS=... MIX=... EXTRA_TRAIN_ARGS=... sbatch`), with enough `afterany` links for 600 epochs (~8 x 20 h)
- [ ] Verified that the LR the trainer actually uses is the intended peak after its sqrt(world size) scaling, and that the one-cycle schedule spans the full 600 epochs (read from `history.jsonl` `lr` at epoch ~60 and at the end)
- [ ] Each finished run scored on the published protocol (EMDB-1 with `--bbox annotated`, SMPL-24 adapter) and on 3DPW test J14 with the adapter
- [ ] A 2x2 table recorded in the status doc, naming the winning recipe or stating that no cell beats g8h_s0 by more than ~0.5 mm
