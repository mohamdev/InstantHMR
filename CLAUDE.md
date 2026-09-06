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
