# Why the method is what it is

The method has converged. This file is the reasoning behind it, so a change
can be judged against the reason rather than the shape. `TRAINING_PROTOCOL.md`
says *what to run*; this says *why*, and what breaks if you drop a piece.

Each principle below has cost something to learn. The numbers are measured on
this project's own runs, not borrowed.

---

## 1. Never train and report on the same oracle

A policy trained against one structure predictor learns that predictor's
quirks, and the score it earns there is not a measurement of the protein.

| | gap (training oracle − AF3) |
|---|---|
| designs no policy optimised | −3.9 |
| RF3-trained, sequence reward | **+18.0** |
| RF3-trained, fragment reward | **+36.3** |

The predictors agree within 4 points where nobody optimised. Everything beyond
that is the policy learning the scorer.

**So:** train on a pool (`[rf3, boltz, protenix]`), report on AF3, and keep AF3
out of the objective entirely. `multi_oracle_from_df` warns if it finds `af3`
in a pool.

**Corollary — evaluation is not training.** Training pools oracles so the
policy cannot overfit one. Evaluation wants the opposite: one fixed yardstick.
A pooled evaluation would mix predictors that disagree by up to 26 points on
identical designs, so the number would depend on which oracle each sequence
drew. `evaluate.yaml` folds with AF3 and nothing else.

## 2. Optimise the thing you are actually building

A combinatorial library does not ask "does this sequence fold?" It asks "does
this fragment still work beside fragments it has never met?" Optimising the
first does not get you the second.

That is what the chimera step is for: it folds `rotations × B` recombinants of
the batch and scores each fragment by the company it keeps.

**But a fragment reward alone is not enough.** With chimera quality as the only
objective, the policy optimised fragments that travel while the sequences it
emitted degraded — **28 passing parents against 182** for the arm trained on
parent RMSD alone (M0255), and retention fell with them, 30.5% against 79.3%.
The fragment term says "your parts travel"; the parent term says "and you
fold".

> **The fix is inferred, not yet measured.** Every shuffling run on record
> (`experiments/ame/fragrl` and the `fragb32`/`bs16`/`bs64` sweeps) has
> `parent_metric` unset — the recommended configuration has never actually
> been run. What is measured is the *failure* of omitting it. Treat
> `parent_metric: ame_motif_rmsd` in `fragment_shuffle.yaml` as the leading
> hypothesis, and the first thing to test if shuffling underperforms.
> See `docs/COMPARISONS.md` for the arm-by-arm numbers.

## 3. Compare only within a batch

GRPO's advantage is `(r − mean) / (std + 1e-3)` computed **within the group**,
which is what makes a pooled oracle safe: a whole-batch offset cancels.

Two consequences, both load-bearing:

- **One group is folded by ONE oracle, never a mixture.** Otherwise part of
  each advantage is a comparison between predictors rather than between
  designs.
- **`use_avg_reward` stays off.** A running-history baseline compares this
  step's oracle against earlier steps' and injects the ~26-point disagreement
  straight into the advantage.

A per-step score is therefore **not comparable across steps**. Compare
oracle-balanced windows of `len(oracles) × m` consecutive steps.

## 4. Best-of-N is the unit, and "best" must mean what you optimise

Best-of-5 is the benchmark's own unit; a single-sample run measures something
else. Every oracle defaults to 5 with `per_sample_rows`, so every pipeline
needs a reducer or the reward tensor stops matching the batch.

The selection criterion is not a detail. For AME it is the benchmark's pass
criterion. For anything else it should be the run's own reward
(`aggregation: ${reward.reward_aggregation}`), so "best of 5" and "what we
optimise" cannot drift apart when the reward changes.

Reducing columns separately is wrong: the lowest-RMSD sample is not always the
clash-free one, so a design would pass on the strength of two different
predictions.

## 5. Measure rather than assume, and write down that you measured

Repeated pattern in this project: the assumed value was wrong and cheap to
check.

- MoMI's **slot boundaries are not recorded anywhere**. They were recovered
  from single-slot swaps against the parent: 1–40, 45–84, 89–126, 131–163,
  168–209, non-overlapping and ordered, which is a check on the method rather
  than an assumption of it.
- The **PETase libraries share 5 parts out of 3,629**. Pooling them put slots
  2 and 3 at 8.1 and 7.0 mean mutations against the true 4.4 and 4.4.
- The **SMILES spelling of each oracle** was read out of the containers. Two of
  three did not match what our own docstrings claimed.
- **Arial really is installed** and is what matplotlib resolves, which is the
  only reason editable SVG text is safe.

## 6. One run proves nothing

Two unseeded runs of one config gave control pass rates of **68.8% and 89.6%**
— a 21-point swing, larger than most effects these experiments exist to
detect. Set `seed:` for anything you intend to compare, and budget replicates.
A single-run A/B at n≈96 cannot resolve anything smaller than that swing.

## 7. Put invariants in code, not in comments

A rule that lives in prose gets broken by the next person in a hurry. The ones
that now fail loudly:

| invariant | enforced by |
|---|---|
| fold N, reduce N | `UniversalReward._check_best_of_n` |
| AF3 stays out of the training pool | warning in `multi_oracle_from_df` |
| figures are SVG | `figio.save` rejects other extensions |
| no label below 7 pt | `palette.fs()` clamps |
| Figure 3 is one library | `fig3_momi.only()` |

## 8. A broken diagnostic is worse than none

It looks like a working one. Two live examples:

- The AF3 probe raised `Missing key af3_container` on every stride and left
  its column NaN — a transfer check that silently was not running.
- The best-of-N guard read fields with `getattr`, which returns the default on
  a plain dict, so under `_convert_="all"` it found no oracle anywhere and
  passed every pipeline. It did not fail; it stopped existing.

So the probe catches its own exception and continues (a diagnostic must never
kill a 450-step run), and the guard now reads through a helper that handles
both dicts and `DictConfig`.

## 9. Keep the record, mark it as the record

Old configs are kept rather than rewritten: they say what was actually run.
`denovo_petase_single_oracle.yaml` no longer runs — it folds best-of-5 with no
reducer and the guard rejects it — and that is correct behaviour. Re-running it
should fail rather than quietly do something different from what it recorded.
