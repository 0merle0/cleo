# Archived from the pooled library design story

Moved 2026-10-17. Nothing here is deleted; all of it is in git history at its
original path, and `git log --follow archive/<file>` still reaches it.

## The three wet-lab application arcs

`results_petase.tex`, `methods_petase.tex`, `heme_results.tex`,
`heme_methods.tex`, `protease_results.tex`, `protease_methods.tex`.
(The two PETase files were both named `petase.tex`, in `results/` and
`methods/`; they are prefixed here so they can coexist.)

The paper is purely computational: RL fine-tuning on the RFdiffusion2 AME
benchmark, and whether the designs it produces can be cut into fragments and
recombined. The PETase closed-loop rounds, the heme oxidation enzyme and the
protease diversity ablation are experimental applications of the method. They
are a different paper.

The heme and protease files were already orphaned — no master `.tex` inputted
them. The PETase pair was live in `main.tex` and needed three edits to remove
without breaking the build:

- `main.tex` dropped `\input{results/petase}` and `\input{methods/petase}`.
- `results/fragments.tex` said fragment recombination "is the only reason the
  PETase libraries in Section~\ref{sec:res:petase} can be the size they are".
  Left in place that is an undefined reference. The claim was rewritten to rest
  on the combinatorial argument itself, which is where its force comes from.
- `methods/overview.tex` described a pipeline "underlying all three
  applications". Restated for the computational scope.

## Superseded planning docs

`outline_computational.md` — superseded by `plan_combinatorial.md`, which says
so in its own second paragraph: the outline's headline claim (policy dominates
the diversity/pass-rate frontier) did not survive the 13-backbone panel.

## Amended 2026-10-17: the PETase data came back, the plan did not

`results_petase.tex` is a six-block **plan** for a three-round closed-loop
campaign ending "250x above the start". No part of that campaign is in this
repository, and the `250` in it was always a placeholder. It stays archived.

What does exist, and what was found after this directory was created, is the
assay data for a *library* built by the fragment-recombination procedure: 3,135
recombined PETase designs on one MoMI run, folded by AF3 in two states. That is
now `results/experimental.tex`, with the join snapshotted into
`figures/data/petase_activity_zeroshot.csv`.

The two are easy to confuse and must not be. The archived plan is about
*optimizing* an enzyme across rounds; the live section is about whether the
library is physically realizable and whether the oracle predicts function. The
archived plan's headline number cannot be computed from the live section's data
— the parent was never assayed, and the only designs above 100 were measured in
a different assay run.
