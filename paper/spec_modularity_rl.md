# Spec: training for recombinability (fragment-level GRPO)

Working doc. Proposed experiment, not yet run.

## Why

Fragment retention is currently an *accident*. We train for motif RMSD, and the
resulting designs happen to tolerate being cut into quarters and shuffled: 81.2%
of chimeras still pass, against 56.4% for untrained LigandMPNN, on all four
backbones measured so far.

Nothing in the objective asked for that. Making it the objective is the natural
next move, and it is the one thing a combinatorial library actually needs —
library size is `P^k` only if the recombinants fold.

## The idea

Shuffle the batch's fragments, fold the chimeras, and score each **fragment** by
how well the chimeras containing it do. Then push that score back into the
policy as a per-fragment advantage.

This is fragment-level credit assignment, not a sequence-level bonus. A design
with three strong fragments and one weak one currently collapses to a single
averaged scalar; here the three are reinforced and the fourth is pushed down.

## It needs no change to the loss

`grpo.py` already computes the surrogate per token:

```python
A_ = A.unsqueeze(-1) if A.dim() < r.dim() else A
per_token = torch.min(r * A_, r_clipped * A_)
```

`r` is `[B, L]`. `A` is broadcast only when it is lower-rank, so passing an
advantage of shape `[B, L]` — every position inside fragment *j* of sequence *i*
carrying that fragment's advantage — is consumed directly. The clip stays a
per-token trust region.

## Procedure, per training step

**Parents are sampled but never folded.** All credit comes from how a fragment
performs in chimeras, which is the claim being trained for; folding the parents
too would spend 40 extra predictions per step on a signal the objective does not
use.

1. Sample `B = 8` sequences.
2. Split each at `k = 4` fixed equal boundaries -> 8 fragments per slot, 32 total.
3. Build 32 chimeras by **rotation, not random draw**: for `rho = 0..3`, chimera
   `i` takes slot `j` from parent `(i + rho*j) mod 8`. Verified: 32 distinct
   chimeras, every fragment in exactly 4, none duplicated. Random draws would
   leave some fragments unscored and over-weight others, biasing the advantage.
4. Fold the 32 chimeras at **best-of-5**. Single-sample folding measures
   something the benchmark does not, and the extra samples are what hold the
   reward's variance down -- each fragment score then rests on 4 chimeras x 5
   predictions = 20 structure predictions.
5. Fragment score `s(i,j)` = fraction of its 4 chimeras that pass
   (`ame_motif_pass_and_no_clash`, best-of-5 reduced).
6. Standardise `s` **within each slot** across the batch. Slots differ
   systematically in tolerance -- one covering the motif passes less often than a
   surface loop -- so pooling them would reward position rather than quality.
7. Advantage `A[i, t] = A_frag[i, slot(t)]`, shape `[B, L]`, consumed directly by
   the existing per-token surrogate.

Cost: 32 designs/step x 150 steps = 4,800 designs, ~32 GPU-h per run, 2x the
panel arm. Three backbones ~96 GPU-h.

## What has to be built

Three pieces, none of them large, but none of them free:

**A chimera reward step.** `UniversalReward` passes one row per sampled sequence
through a chain of steps. This step receives 8 rows, builds 32 chimeras, folds
them, and returns the 8 rows carrying per-fragment columns. It is the first step
in the codebase whose folding set differs from its input set.

**A per-slot advantage path.** `self.advantage()` standardises over the whole
reward tensor. Fragment advantages must be standardised per slot and assembled
into `[B, L]`, so this needs its own path rather than reuse.

**A checkpoint metric that does not depend on parents.**
`ame_motif_rmsd_batch_mean` is computed from the sampled sequences, which are no
longer folded. Use the chimera mean instead, and note in the config that it is
not comparable to the panel runs' checkpoint metric.

## Backbones

Three, chosen for headroom and for having enough passing parents that chimeras
carry signal:

| backbone | dead% | current retention | parents |
|---|---|---|---|
| M0097_1ctt | 55 | 72.7% | 178 |
| M0255_1mg5 | 77 | 79.3% | 182 |
| M0315_1ey3 | 85 | 78.1% | 199 |

M0664 is excluded deliberately: at 94.9% retention there is almost nothing to
improve, so it cannot distinguish the arms. The hard backbones (M0732 at 8
parents, M0365 at 9) are excluded because a batch of 16 on a policy passing ~3%
yields almost no passing chimeras, so the fragment score would be mostly zeros
and carry no gradient. Those are the interesting case *later*, once this is shown
to work where signal exists.

Comparison arm is free: the existing panel runs on these three backbones are the
same configuration without the fragment term.

## Cost

32 chimeras/step × 150 steps = 4,800 designs at best-of-5, **~32 GPU-h per run,
~96 GPU-h for three** -- 2x the panel arm. Evaluation is the existing 256-design
protocol plus a 256-chimera retention fold, ~2 GPU-h per backbone.

Note the batch drops from 16 to 8, which halves the group GRPO standardises over.
If the arm underperforms, that is a confound with the fragment term. A B=8
control without the fragment term costs ~16 GPU-h per backbone and removes it;
worth running if the first result is ambiguous rather than pre-emptively.

## Predictions, pre-registered

1. Retention rises above the panel arm on all three backbones. If it does not
   move on any, the fragment advantage is not reaching the policy and the
   experiment is a null.
2. Parental pass rate does **not** collapse. Chimera pass rate is itself a
   pass-based measure, so unlike the diversity term it has no obvious degenerate
   optimum — but see the kill criterion.
3. The gain is larger on M0255 and M0315 than on M0097, following the pattern
   that policy advantage grows with backbone difficulty.

## Kill criterion

**Fragment collapse.** The policy can trivially maximise recombinability by
emitting near-identical sequences: every chimera then reproduces a parent and
passes. Track distinct fragments per slot among passing designs, and distinct
substitutions `U` over the sampled library.

If either falls below the panel arm's value at the same step, the run has found
the degenerate optimum and the term must be gated — score only chimeras that
differ from every parent by at least `d` substitutions — rather than reweighted.

Do **not** add a diversity reward to counteract it. The weight pilot measured
that: at every weight from 1 to 4, on both backbones, adding
`div_marginal_fraction` to the objective drove pass rate to 0/256 with median
motif RMSD of 5.9–11.8 Å. Diversity is a diagnostic here, not a reward.

## Open question

Chimeras are excluded from the fragment score if they reproduce a parent exactly,
but near-parents are not. Whether that matters depends on how much the policy
converges; the distinct-fragment diagnostic will show it before it becomes a
problem.
