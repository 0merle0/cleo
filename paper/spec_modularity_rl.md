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

1. Sample `B = 16` sequences as now.
2. Split each at `k = 4` fixed equal boundaries → a pool of 16 fragments per slot.
3. Build chimeras by **rotation, not random draw**: for rotation `ρ = 1..R`,
   chimera `i` takes slot `j` from sequence `(i + ρ·j) mod B`. This is a Latin-
   square design, so every fragment appears in exactly `R` chimeras and no
   fragment's score is noisier than another's. Random sampling would leave some
   fragments unscored and others over-represented, which biases the advantage.
   `R = 4` → 64 chimeras.
4. Fold the 16 originals at best-of-5 (the reported unit) and the 64 chimeras at
   **1 diffusion sample**. The chimera number is an internal relative signal for
   credit assignment, never reported, so it does not need the benchmark's
   best-of-5. This is what keeps the cost down: 16×5 + 64×1 = 144 predictions per
   step against 80 today, 1.8× rather than 5×.
5. Fragment score `s(i,j)` = fraction of the `R` chimeras containing fragment
   `(i,j)` that pass.
6. Advantage: standardise `s` **within each slot** across the batch. Slots differ
   systematically in how tolerant they are — a slot covering the motif will pass
   less often than a surface loop — and pooling them would reward position rather
   than quality.
7. Combine with the existing geometry advantage per sequence:

   ```
   A[i, t] = w_geom * A_rmsd[i]  +  w_frag * A_frag[i, slot(t)]
   ```

   `A_rmsd` is today's sequence-level advantage, broadcast flat; `A_frag` varies
   along the sequence. Start `w_geom = w_frag = 1`.

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

144 predictions/step × 150 steps ≈ 21,600 per run. At the measured RF3 rate
(~6.7 GPU-h per 1,000 designs) that is **~30 GPU-h per run, ~90 GPU-h for three**.
Evaluation is the existing 256-design protocol plus a 256-chimera retention
fold, ~2 GPU-h per backbone.

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
