# CLAUDE.md

Behavioural guidelines plus what you need to know about this repo before touching it.

**Tradeoff:** these guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 0. Never commit

**The user commits. Always.** Never run `git commit`, `git push`, `git merge`, or
`git rebase`. When work is ready, print the commands and let the user run them.
Keep commit messages to one or two short sentences.

`git add` and read-only git (`status`, `diff`, `log`, `show`) are fine.

---

## 0.5 Write for a human, not a compiler

**The user is a person reading a terminal, not a log parser.** Answers that are
technically correct and unreadable are failures. This has come up more than
once — take it seriously.

- **Define every name before you use it.** `scale_lo`, `m_noflip`, `fk_scale`,
  `n_skip_run` mean nothing to someone who has not just read that function. If
  a variable, flag or file has to appear, say what it *is* in plain words
  first, once.
- **Give commands in the form the user actually runs them.** This repo launches
  through `datasets_pipeline/jeanzay/52_train_ddp.slurm` with `RUN_NAME=... MIX=...
  EXTRA_TRAIN_ARGS=... sbatch`. Read the launcher and match it. A bare `srun
  python ...` line the user has never typed is not an answer.
- **Say what to run, in what order, and what happens next.** If a command is
  optional, say so. If you offer a check, say what a good result looks like.
- **Lead with the answer.** The finding first, the evidence after. Do not make
  the user read a derivation to learn whether it worked.
- **Explain the mechanism, don't just name it.** "The bone scales enter the
  kinematics multiplicatively, so +10 makes a 336 m skeleton" teaches
  something; "unbounded scale parameters caused divergence" does not.
- **Cut the ceremony.** No restating the request, no listing what you are about
  to do, no summary of the summary.

If the user says they cannot follow an answer, that is a defect to fix in the
next answer, not a preference to note.

---

## 1. Think before coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity first

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

## 3. Surgical changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports and variables that *your* changes orphaned; leave pre-existing
  dead code alone unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-driven execution

Define success criteria. Loop until verified.

- "Add validation" → "write tests for invalid inputs, then make them pass"
- "Fix the bug" → "write a test that reproduces it, then make it pass"
- "Change a loss" → "show the new term is numerically what you claim, on real data"

For multi-step tasks, state a brief plan with a verification per step.

In this repo "verify" usually means **run it on real data and show numbers**, not
read the code and assert it looks right. `data/` holds a local corpus and
`checkpoints/mhr_model.pt` is the MHR rig, so most claims about geometry, losses
or augmentation can be measured in under a minute. Do that.

---

## Where to read first

Docs are the source of truth; the code is bigger than any context window.

| order | file | for |
|---|---|---|
| 1 | `instanthmr_distill_train/README.md` | which training script to run, what `--preset v2` and `--losses rebalanced` change, the pair-index cache |
| 2 | `datasets_pipeline/jeanzay/STATUS.md` | Jean Zay cluster + data state, what is built, job templates *(git-excluded, local only)* |
| 3 | `datasets_pipeline/JEAN_ZAY.md` | site runbook: env, quotas, proxy quirks *(git-excluded, local only)* |
| 4 | `benchmark/README.md` | how published numbers are produced; the 3DPW GT (SMPL forward + H36M regressor), the MHR->J14 adapter, and why **J14+adapter** is the row to quote |
| 5 | `datasets_pipeline/README.md` | how the corpus is built, including the SA-1B visibility mask |
| 6 | `docs/architecture.md` | the student model |
| 7 | `docs/todo.md` | open defects and opportunities from the 2026-09-06 audit, each with the measurement behind it |

`datasets_pipeline/jeanzay/` and `JEAN_ZAY.md` are in `.git/info/exclude` on
purpose — they carry the login name, jump host and project code. Keep
site-specific material there and portable material in the repo docs.

## Repo-specific traps

**One trainer, two layers.** `train_distill_mhr_only.py` holds the dataset,
augmentation, model and losses. `train_distill_jz.py` imports it and adds DDP,
mixture sampling, resume and 3DPW validation. Substance changes go in the base
file so both entry points get them. **Never fork the base file** — that is how
`train_distill.py` drifted until its ONNX export silently zeroed the SimCC 2D
head, with keypoints collapsing to the bbox centre and no error raised.

**`--preset baseline` and `--losses legacy` must stay bit-identical.** Runs in
flight use them as controls. The two flags are orthogonal axes — `--preset` is
the input pipeline and the absolute-pose terms, `--losses` is the loss budget —
and both default to the old behaviour. If you change the dataset or a loss, gate
it behind a config flag whose default reproduces the old behaviour, and prove it
— diff a few dozen augmented samples against `git show HEAD:<file>`. When you
do, compare **alternating** calls: the TorchScript MHR forward differs by ~5e-10
on its first invocation in a process (cold-kernel autotuning), which reads as a
regression if you only compare once.

**Know the MHR parameter layout before touching a pose loss.** From
`character_torch.parameter_transform.parameter_names`: `0:3` root translation
(always 0 in the corpus, 1 unit = 10 cm), `3:6` root rotation (extrinsic-XYZ
Euler, driving joint-parameter slots 10/11/12 *and nothing else*), `6:136` the
130 local joint angles, `136:204` 68 bone scales. Consequences that cost a
session to re-derive: `6:136` is exactly invariant to the in-plane rotation
augmentation, so it never needed the `m_ident` mask; and the 45 `shape_params`
move the 127-joint skeleton by `0.00e+00 cm`, so they are mesh-only and every
loss and metric in the trainer is blind to them.

**The student's MHR outputs are unbounded and 77 of them are body SIZE.** The
head is a plain `nn.Linear`. `0:3` root translation (exactly `0.000e+00` in all
24,000 sampled annotations — the person is placed by `cam_trans`), `130:136`
the `*_length/_width_flexible` parameters, and `136:204` the 68 bone scales all
scale the skeleton, the last **multiplicatively along the kinematic chain**:
uniform +10 gives a 336 m skeleton, +25 gives 28,300 km. That, not the learning
rate, killed four 300-epoch runs at epochs 24-32 on 2026-09-06. Watch for the
two traps: `130:136` *look* like pose parameters (they sit inside `6:136`) but
drive joint **translation** channels, so bounding only `136:204` still reaches
375 m; and `6:130` are genuine rotations that stay at ~1.5 m at any magnitude,
so leave them alone. `--bound-scales` fixes it from
`assets/mhr_size_bounds.npz`; that file is **new and must be rsynced** or the
job dies at startup.

**The anomaly guard deletes the gradient that would fix the anomaly.** A batch
over `anomaly_loss_threshold` was skipped whole — but `loss_scale`/`loss_pose`
in that same batch are the only force pulling the sizes back, so the sample was
quarantined for good. Worse, `bad` is all-reduced with `MAX`: one bad sample in
4x64 = 256 vetoes the step for every rank, so 2% runaway samples stop 99.4% of
steps. Use `--anomaly-safe-fallback`, which steps on the non-FK terms instead
(`safe_loss_subset`, cosine +1.0000 with `teacher - prediction`). Never "fix" a
divergence here by lowering the LR alone: `OneCycleLR(pct_start=0.1)` peaks at
10% of the run, so a lower peak only moves *when* it dies.

**`parameter_transform[:, :204]` has rank 192, not 204.** Twelve directions
move the parameters and leave the skeleton *exactly* unchanged — each
`*_flexible` parameter against its `scale_*` partner, plus the spine rotations.
Along them every geometric loss has zero gradient, so the model is regressing
12 dimensions of solver-arbitrary teacher noise that no image can determine.
Do **not** "fix" this by zeroing the `*_flexible` parameters even though the rig
pins them to `[0,0]`: the teacher genuinely uses them (`leg_length_flexible`
mean 0.32, max 1.64), so zeroing shortens every leg.

**`DistillationLoss` is never `.to(device)`'d.** `run_overfit_test`,
`run_self_tests`, `train_instant_hmr` and `train_distill_jz.train` all construct
it bare, so any buffer you register lands on the CPU and raises on step 1. Build
tensors on `mhr_module.device` instead.

**3DPW GT is the published protocol, and it is a cache on disk.** The pickles'
`jointPositions` field is the 24 SMPL *kinematic* joints, which is not what any
paper reports; `benchmark/make_3dpw_gt.py` precomputes the real reference (SMPL
forward + `J_regressor_h36m`) into `<3DPW root>/gt_h36m/`, and both the harness
and `val3dpw.py` default to it. Two consequences: a job whose 3DPW directory
lacks that cache raises at startup, and **numbers from before 2026-09-05 are ~13
mm apart from later ones** — `--val-3dpw-gt jointpositions` keeps a resumed run
on its original metric.

**Anything fitted over the 70 MHR keypoints needs float64 and a truncated
`lstsq`.** 40 of the 70 are finger joints millimetres apart, so the design
matrix is near-singular. Centring in float32 leaves a 0.04 mm residual in the
exact direction centring annihilates, `lstsq` keeps it as signal, and the J14
adapter went from 37 mm to 430 mm in-sample looking like a broken GT. Even in
float64 the unregularised fit reaches landmarks through ±2000 weights on
fingers; `RCOND` in `eval_3dpw.fit_adapter` truncates it to a rank-24 map with
weights under 1 at no measurable cost.

**The CLIFF conditioning is computed in three places and they must agree.**
`SAM3DStudentDataset.__getitem__`, `val3dpw._preprocess` and
`instanthmr.inference.InstantHMR._preprocess` each build the 3-vector
separately. Feeding a model trained on one variant with the other raises
nothing — it silently mis-places the person in depth. Since 2026-09-06 the
perspective-correct variant (`--cliff-focal`, config `cliff_focal`) is
mandatory for new runs, because the rebuilt corpus carries real per-image
focals spanning `f/diag` 0.6-1.8 where the pixel form is ambiguous; the old
corpus was constant at 1.000. See `instanthmr_distill_train/README.md`.

**Never rebuild a config from `DistillConfig()` defaults to load a checkpoint.**
Two 2026-09-06 flags make the defaults wrong, in opposite ways.
`--bound-scales` puts a `tanh` remap inside `forward()`, so a default-config
export reads the head's raw pre-`tanh` numbers as bone scales — measured 21.0
mm MPJPE / 14.4 mm PA-MPJPE of silent corruption. `--cliff-focal` changes what
the conditioning vector *means*, not the graph, so a mismatch raises nothing.
Use `T.config_from_checkpoint()`: it reads the bounds out of the weights
(`scale_lo`/`scale_hi` are the flag) and the conditioning form out of the
`run_config.json` beside the checkpoint, and `tools/pth_to_onnx.py` stamps the
latter into the ONNX metadata so `instanthmr.inference` configures itself.
Deployment has no annotation to read a focal from, so it falls back to
`1.05 x diag` for focal-aware graphs and `1.00 x diag` for pixel-conditioned
ones (that IS the old corpus's synthetic focal); `demo.py --focal` overrides.

**One reported number: J14 PA-MPJPE, adapter-applied, on the H36M GT.** The
training log, `summarize_runs.py` and `benchmark/eval_3dpw_ckpt.py` all print
it and they agree to the decimal. J12 was removed on 2026-09-06 — it was the
rig-clean row against the old GT and is not against this one. Runs started
before that date recorded J14 *without* the adapter, ~26 mm higher; do not put
old and new rows in one table.

**PA-MPJPE is blind to half of what a demo shows.** Procrustes removes
translation, rotation and scale, so `cam_trans` drift, body-size error and
temporal jitter are all invisible to it — and to root-relative MPJPE for the
first two. A change can be worth shipping on stability and move the benchmark
by 0.1 mm. Measure jitter directly (frame-to-frame acceleration on 3DPW's
consecutive frames, and the variance of rigid bone lengths within a person
track, which cannot change and so is pure noise).

**Seed spread is wide.** Observed 70-keypoint PA-MPJPE varies 34–49 mm across
seeds, which is larger than several of the changes on the roadmap. A single-run
comparison proves nothing; use 2–3 seeds.

**You cannot reach Jean Zay.** No SSH, no Slurm. Print commands for the user and
read the logs they paste back. Files reach the cluster by `rsync` of *specific
files* — never the whole tree, since some files are edited on both sides.

**ONNX export constrains the model.** `torch.linalg.lstsq` and friends do not
export; use explicit small-matrix algebra. Pin `onnxruntime-gpu==1.26.0` — 1.27+
is CUDA 13 and silently falls back to CPU.
