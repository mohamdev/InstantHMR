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
| 1 | `instanthmr_distill_train/README.md` | which training script to run, what `--preset v2` changes, the pair-index cache |
| 2 | `datasets_pipeline/jeanzay/STATUS.md` | Jean Zay cluster + data state, what is built, job templates *(git-excluded, local only)* |
| 3 | `datasets_pipeline/JEAN_ZAY.md` | site runbook: env, quotas, proxy quirks *(git-excluded, local only)* |
| 4 | `benchmark/README.md` | how published numbers are produced; J12 vs J14 and MHR-vs-SMPL joint conventions |
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

**`--preset baseline` must stay bit-identical.** Runs in flight use it as the
control. If you change the dataset or a loss, gate it behind a config flag whose
default reproduces the old behaviour, and prove it — diff a few dozen augmented
samples against `git show HEAD:<file>`.

**Seed spread is wide.** Observed 70-keypoint PA-MPJPE varies 34–49 mm across
seeds, which is larger than several of the changes on the roadmap. A single-run
comparison proves nothing; use 2–3 seeds.

**You cannot reach Jean Zay.** No SSH, no Slurm. Print commands for the user and
read the logs they paste back. Files reach the cluster by `rsync` of *specific
files* — never the whole tree, since some files are edited on both sides.

**ONNX export constrains the model.** `torch.linalg.lstsq` and friends do not
export; use explicit small-matrix algebra. Pin `onnxruntime-gpu==1.26.0` — 1.27+
is CUDA 13 and silently falls back to CPU.
