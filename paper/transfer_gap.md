# The transfer gap, and what closes it

CLEO on the RFdiffusion2 AME benchmark. Status 2026-10-16 16:10.
Three backbones, single seed per cell.

Training a policy against a structure predictor teaches it that predictor's
quirks. Four arms ask whether an oracle pool or a bigger batch is what keeps the
in-loop number honest.

## What the objective is

The library thesis is that a large combinatorially-compatible library can be
built by taking designs that pass, cutting each into **k = 4 fragments**, and
recombining fragments across designs. That only works if fragments are
interchangeable, so the RL objective does not score the designs — it scores
**recombined chimeras**. Each step samples a batch of sequences, cuts them,
builds `rotations * batch = 128` chimeras, folds those, and rewards chimera pass
rate.

Which means a structure predictor sits inside the training loop, and the pass
rate it reports cannot be taken at face value. Every arm is therefore measured
twice:

| | |
|---|---|
| **train level** | In-loop chimera pass rate, oracle-balanced over a sliding window. The three predictors disagree by up to 26 points on identical designs, so a single step's raw draw jitters more than the effect being measured. |
| **held-out AF3** | The same chimeras re-folded by AlphaFold3, which appears in no reward. |
| **gap** | **train level − held-out AF3.** Positive means the training number is overstating. Near zero means the policy learned something real. |

## The 2x2

Two candidate fixes. Fold with a **pool** of predictors in round-robin, so no
single one can be gamed. Or just use a **bigger batch** — GRPO normalises
advantage as `(r - mean) / (std + 1e-3)`, and a batch of 8 gives a very noisy
baseline. Nothing on disk separated the two, which is what the fourth cell is
for.

| | **RF3 only** | **3-oracle pool** |
|---|---|---|
| **batch 8** | `fragrl4` — **+0.278**<br>warm-started from a `fragrl3` checkpoint, not a matched cell | `fragmulti` — **+0.000**<br>from scratch; 226/300 steps, left unfinished |
| **batch 32** | `fragb32rf3` — **+0.028**<br>running, 26–51 of 150 steps; three early probe points, not a converged result | `fragb32` — **−0.009**<br>two of three backbones complete at 150 steps |

Mean gap over probed steps, averaged across the three backbones.

## Per backbone

| arm | batch | oracle | M0097 | M0255 | M0315 | mean |
|---|---:|---|---:|---:|---:|---:|
| `fragrl4` | 8 | RF3 | +0.299 | +0.476 | +0.060 | **+0.278** |
| `fragmulti` | 8 | pool | +0.060 | +0.042 | −0.103 | **+0.000** |
| `fragb32` | 32 | pool | +0.054 | −0.057 | −0.025 | **−0.009** |
| `fragb32rf3` | 32 | RF3 | +0.070 | +0.101 | −0.086 | **+0.028** |

Mean of `train_pass - af3_pass` over probed steps, from
`paper/figures/ame_transfer_curves.py`, which excludes probe rows a resume
overwrote. The first three arms are probed offline at steps 0/30/60/90/120
(n=5; `fragb32` M0255 at 33/45/60/90/120, M0315 n=4). `fragb32rf3` is probed
in-loop every 25 steps and currently has one or two usable points per backbone.

## Where the runs stand

| run | steps | train level | in-loop gaps | state |
|---|---:|---:|---|---|
| `fragb32` / M0097 | 151 / 150 | 0.958 | — | done |
| `fragb32` / M0255 | 126 / 150 | 0.927 | s75 +0.102 | running |
| `fragb32` / M0315 | 150 / 150 | 0.992 | s100 +0.000, s125 +0.007 | done |
| `fragb32rf3` / M0097 | 51 / 150 | 0.748 | s25 +0.070, s50 +0.070 | running |
| `fragb32rf3` / M0255 | 26 / 150 | 0.505 | s25 +0.101 | running |
| `fragb32rf3` / M0315 | 26 / 150 | 0.819 | s25 −0.086 | running |
| `fragk` (9 configs) | 0 / 200 | — | staged, validated, gated | not launched |

## Held fixed across every arm

- **policy** — LigandMPNN, GRPO, `lr 1e-4`, `N_updates 16`,
  `update_batch_size 8`, `temperature 1.0`, `seed 1`
- **objective** — `k = 4` fragments, `rotations = 4`, 128 chimeras per step,
  best-of-5 folding
- **backbones** — `M0097_1ctt_cond9_14`, `M0255_1mg5_cond17_66`,
  `M0315_1ey3_cond18_25`, all L = 180
- **hardware** — 4 GPUs per run. RF3 measures 20–24 min/step, so a 150-step arm
  is ~220 GPU-h and the three E15 arms ~660, against 284 budgeted.

## How to read it

The clean comparison is the bottom row: same batch, same k, same seed, same
backbones, one variable — `oracle: rf3` forced inside the same
`multi_oracle_from_df` call, so dispatch and column renaming stay identical. The
pool arm sits at −0.009; the RF3-only arm is at +0.028, and both of its probed
M0097 points read +0.070.

That leans toward the pool doing real work even at batch 32, which is the result
that would justify the fragment-count sweep's ~650 GPU-h per k level. Three
things stop it being a claim:

1. **M0315 runs the other way** — −0.086, on the backbone that saturates
   earliest and shows the smallest gap in every arm.
2. **The policies are not converged.** Pass rates are still climbing steeply,
   0.505 to 0.819 across backbones. A gap at step 25 against the pool arm's
   steps 30–120 is not a matched comparison. It becomes one when probes land at
   100–150.
3. **n = 1 seed per cell.** Two unseeded runs of one config on this benchmark
   gave 68.8% and 89.6% — a 21-point swing. `seed: 1` everywhere makes the arms
   comparable but says nothing about sampling error.

## Carried caveats

- **`fragrl4` is not a matched baseline.** It warm-starts from a `fragrl3`
  checkpoint, so its step 0 already carries ~450 RF3-trained steps while every
  other arm starts from scratch. It is in the table as the arm the project
  started from.
- **A quarter of each step's chimeras are parents.** `build_chimeras` maps
  chimera (rho,i) to parent `(i + rho*j) % B`, so rho=0 returns the
  unrecombined parent: 32 of 128 at every k.
- **Two sets of numbers exist in the repo.** The header comment in
  `submit_fragb32rf3.sh` quotes these gaps as +0.439 / +0.176 / −0.001. Those
  predate the bad-row exclusions and the oracle-balanced train level, and the
  current loader does not reproduce them. The figures here are what the code
  computes today.
- **Only k = 4 has been trained.** `ame_granularity.py` sweeps k = 4…36 *post
  hoc* by re-cutting designs trained at k = 4 and reports best k = 36. The
  staged sweep trains at each k, which is what separates a train/test mismatch
  from a junction-context limit.
