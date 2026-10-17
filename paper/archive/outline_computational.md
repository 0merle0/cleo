# CLEO — computational-only paper: outline

Working doc. Not part of the LaTeX build.

Purpose: define a paper that stands on computational results alone, so wet-lab
data later slots in as confirmation of a claim that is already complete.

Status key: **[M]** measured · **[P]** partial · **[R]** running · **[X]** not run

---

## Thesis

> The most informative library is the one with the most *diverse passing*
> sequences. Inverse-folding models cannot produce one: raising temperature buys
> diversity and destroys pass rate, lowering it buys pass rate and destroys
> diversity. RL fine-tuning moves both at once, and on some backbones produces
> passing sequences where sampling produces none. Splitting those sequences into
> fragments and recombining them converts that linear gain into a multiplicative
> one.

Two halves: **(1)** more diverse passing sequences per unit of screening effort;
**(2)** fragment recombination turns them into libraries orders of magnitude
larger than the parent set.

---

## 0. Protocol — applies to every result

**Train under RoseTTAFold3, evaluate under AlphaFold3.** Always. The policy
never sees the predictor it is scored by, so no number in the paper can be a
product of optimizing the metric being reported. This is why cross-oracle is no
longer a separate result — it is the method.

**Budget accounting.** Folding is the entire cost; MPNN sampling is
microseconds. Every arm is charged in *designs folded*, split into training and
inference, and written to `folds.json` per run so the matched-budget claims are
auditable rather than asserted.

| | folds |
|---|---|
| policy training | 2,400 (batch 16 × 150 steps) |
| policy inference | 256 |
| **policy total** | **2,656** |
| LigandMPNN baseline | 2,656 (443 at each of T = 0.1, 0.2, 0.3, 0.5, 0.7, 1.0) |

The baseline's budget is spread across the temperature sweep, not given to one
setting. Pooling temperatures reaches more distinct substitutions than any
single temperature at the same budget (measured on M0097: pooled U = 1,090 vs
781 for the best single temperature), so pooling is the baseline's strongest
play and beating it is the stronger claim.

**Objective.** Geometry and unique mutations, rank-normalized:

```
steps:  div → rf3 → ame → bo5
  ame_motif_rmsd         min   weight 1.0   rank
  div_marginal_fraction  max   weight  W    rank
```

`div` is sequence-level and must run before folding; `bo5` takes remaining
columns from the min-RMSD row, so the diversity term survives the reduction.

Note: under `normalize: rank` the `lower_bound`/`upper_bound` fields are
**deliberately ignored** by `reward.py`. They have been stripped from all
configs — left in, they read as a calibrated range that does nothing. The only
lever on diversity pressure is `weight`.

In reference-free mode (no `ref_seq`, correct for de novo backbones)
`div_total_muts` is the sequence length for every member, so
`marginal_fraction` and `marginal_count` are rank-identical (corr 1.000). The
choice between them does not matter.

---

## 1. Landscape (retained — still accurate)

**De novo enzyme structure generation.** RFdiffusion2 (41/41 active sites on its
own benchmark vs 16/41 prior SOTA), de novo serine hydrolases, metallohydrolases.
These papers own *backbone generation*. Their sequence-design step is uniformly
**ProteinMPNN at low temperature plus hard in-silico filtering**, ~96 designs
ordered. Sequence design is treated as a solved subroutine.

**RL for inverse folding.** ProteinZero, diversity-regularized DPO, BetterMPNN
(GRPO + AF metrics). Closest prior art to the mechanism and must be confronted
directly. But: general benchmarks (CATH), sequence-level metrics, no active
sites, no ligands, **no libraries**.

**Manufacturing-aware generative models** (Weinstein 2026, Nat Biotech).
Factorize the generative model to match combinatorial DNA chemistry; ~10^16
designs. Nearest prior art for the library half. They factorize *then* train on
300M observed antibodies; we train *then* factorize, because a de novo enzyme has
no natural distribution to learn from. Cite as prior art, not as a foil.

**White space: library design for de novo enzymes.** No MSA prior, starting
activity ~ 0, the fix is tens of mutations away, and current practice
deliberately destroys diversity to buy pass rate.

---

## 2. Core results

### R1 — The informative-library frontier  ★ headline

**Claim.** At matched fold budget the policy dominates the temperature frontier:
more passing sequences *and* more distinct substitutions among them.

Pilot evidence, M0097, 96 folds per arm (old budget) **[M]**:

| arm | passing | U |
|---|---|---|
| T=0.2 | 64 | 341 |
| T=0.5 | 33 | 628 |
| T=0.7 (best U) | 25 | 781 |
| T=1.0 | 4 | 356 |
| policy (RF3-trained) | 60 | **1,092** |

The sweep has an interior optimum because the axes trade off; the policy sits
above all of it, and every seed clears it individually (942 / 1,048 / 1,287).

⚠️ **Two open issues.** (a) That comparison excludes the 2,400-fold training
debt; charged it, pooled T=1.0 at equal total budget reaches U = 1,824. (b) Our
U figures come from 96 samples and are floors, not measurements. The panel
(X1) fixes both by sampling 256 and charging the baseline a matched budget.

### R2 — Rescue: dead backbones become productive  ★ highlight, not the focus

| backbone | untrained (measured) | RF3-trained → AF3 |
|---|---|---|
| M0907 | **0 / 10,000** | 23.3% (sd 3.7) |
| M0904 | **8 / 10,000** = 0.08% | 9.7% (sd 1.2) |
| M0097 | 102 / 4,000 = 2.55% | 62.2% (sd 21.9) |

Context from RFdiffusion2's published 41-site table: **~76% of backbone-slots
yield zero passing sequences**; median site 86% dead; 16/41 sites >=90% dead.
Our backbones sit at sites with 55% / 97% / 93% dead fractions — representative,
not cherry-picked. **[M]** for these three; the panel makes it a rate.

### R3 — *(absorbed into the protocol)*

Cross-oracle transfer is no longer a standalone result. Every number is
RF3-trained and AF3-evaluated by construction, so the circularity objection is
answered everywhere rather than in one section. The three-backbone transfer data
stays in the notebook as the validation that established the protocol.

### R4 — Fragments convert yield into library size  **[M arithmetic, X biology]**

At a matched 4,000-fold budget on M0097, real passing parents, 4 equal slots:

| | parents | recombinants | parts to synthesize |
|---|---|---|---|
| T=1.0 | 102 | 1.1 x 10^8 | 408 |
| **policy** | **1,433** | **4.2 x 10^12** | 5,728 |

**~39,000x the library for ~14x the synthesis**, because the product scales as
P^k while synthesis scales as k*P. Pass rate — exactly what RL improves — is the
dominant lever on library size.

⚠️ Arithmetic on an untested assumption: that recombinants still fold. **See X3.**

The 13 trained policies from the panel become the 13 fragment-library parents,
so R4 is a by-product of X1 rather than a separate campaign.

### R5 — Calibration  **[M]**

σ = 7.75 points run-to-run (E21, 10 runs), resolution floor 9.6 points at n = 5.
Stated up front; every contrast is reported against it. Rare in this literature
and worth foregrounding rather than burying.

---

## 3. Experiments

### X0 — Diversity-weight pilot  **[R] running**, ~96 GPU-h

Jobs 25866776–81. Two backbones (M0097 easy, M0907 hard) × div weight {1, 2, 4};
**w = 0 comes free from the E22 runs**, which had no diversity term. Evaluated at
n = 256 under AF3.

Gates the panel. Two facts bracket the answer and neither was chosen:
`backbone19`'s M0097 already ran with div at weight 1.0 and still reached
U = 1,045, below the pooled baseline's 1,090 — so **1.0 is known insufficient**;
and E16/E18 showed diversity-first pressure collapses pass rate — so **too much
is known to fail**. The response is non-monotone with an interior optimum.

Readout: U of the passing set at n = 256, not pass rate alone. A weight that
raises coverage while halving yield may still win on the actual objective.

A flat curve, or a peak at 1.0, is also informative — it would mean the
objective cannot buy past the baseline's coverage and R1 needs rethinking before
390 GPU-h goes into the panel.

### X1 — The R1 panel  **[X]** built and waiting on X0, ~390 GPU-h

13 backbones, **one per AME site** so the points are independent rather than
correlated draws from one site. Stratified by the published dead-backbone
fraction; hard-weighted on purpose, because the benchmark is (median site 86%
dead).

| tier | backbones |
|---|---|
| easy | M0664 (16% dead), M0097 (55%) |
| medium | M0255 (77%), M0315 (85%), M0500 (86%) |
| hard | M0078, M0058, M0732, M0050, M0092, M0365 (93–99%), M0904, M0907 |

One seed per backbone: for a population claim, more backbones beats more seeds —
seed noise averages out across targets, and the existing 3-seed runs already
carry the variance statement.

Configs are generated from each backbone's own source config with assertions on
every field (pipeline order, `structure_col`, early-stopping disabled, existence
of pdb / template / design_pdb / trb). All 13 pass; two verified through full
Hydra instantiation. Backbones whose motif atoms sit on designable residues were
checked — all such atoms are backbone-only (N/CA/C/O), so they exist regardless
of residue identity.

⚠️ All 13 must be re-run, including M0097/M0904/M0907, because those were
trained without the diversity term and are no longer comparable. Their 3-seed
variance result stays valid on its own.

Produces R1, R2, and the parents for R4.

**Also answers:** does baseline difficulty predict policy performance? With 13
backbones spanning 16-99% dead fractions we can regress policy pass rate on the
published baseline rate. A null or weak correlation is a result in its own
right — it would mean the backbones RL rescues cannot be picked in advance from
how badly sampling does on them, which matters for anyone deciding where to
spend training compute.

### X2 — Policy coverage at scale  — folded into X1

Sampling at n = 256 rather than 96 is now the panel default, which is what this
experiment existed to fix. Still a floor rather than a saturating measurement;
if the U-vs-n curve has not flattened by 256, a deeper draw on one backbone
settles it cheaply.

### X3 — Fragment retention  ★ critical path, ~10 GPU-h

Split passing designs at fixed boundaries, recombine, fold best-of-5, report
retention = recombinant pass rate / parental pass rate. Pre-registered with a
kill criterion in `results/fragments.tex`. Runs on the panel's trained policies.

Robust to a modest answer — even 1% retention leaves 4 x 10^10 against the
baseline's 10^8 at 100% — but unmeasured it is the paper's soft center.

**Order:** X0 (running) → X1 → X3.

**Cut: AF3-trained comparators.** An arm trained on the reporting oracle was
considered and dropped. Under the protocol we never report AF3-trained numbers,
so the comparator has no section to live in; and a result showing AF3-training
scores higher would only be demonstrating the reward-hacking effect the protocol
exists to remove, which needs no experiment. The M0097 pair we already have
(78.5% AF3-trained vs 62.2% RF3-trained, p = 0.14, n = 5 vs 3) is enough for a
Methods footnote quantifying the price of the held-out oracle. Quantifying that
price properly is a different paper's question.

---

## 4. Edits and fills

| item | action |
|---|---|
| `results/frontier.tex` | Rewrite to the informative-library frontier; currently a pass-rate story |
| `results/fragments.tex` | Keep pre-registration; fill Block 3 from X3 |
| `results/petase.tex` | **Cut** — wet lab |
| `supplementary/heme_*`, `protease_*` | **Cut** |
| Cross-oracle section | Fold into Methods as the protocol, not a result |
| Selection-rule arc (E9/E16/E18/E19) | Demote to one honest paragraph in Discussion — it ends unresolved and is not a result |
| Diversity wording | Must say *per unit budget* everywhere; at matched library size T=1.0 carries more substitutions (1,988 vs 1,045) and a reviewer will find it |
| `abstract`, `intro`, `conclusion` | Rewrite to the two-half thesis |
| Figures | F1 frontier across 13 backbones (R1) · F2 rescue + published dead-fraction context (R2) · F3 library size vs budget (R4) · F4 diversity-weight response (X0) |
| `main.tex` | Drop Move-3 scaffolding; two results halves plus calibration |

---

## 5. Known soft spots

- **M0097 seed variance is 21.9** against 1.2 and 3.7 on the other two
  backbones. Unexplained.
- **`backbone19` and `selection2` disagree by 35 points** on the same backbone
  and arm — 4.5σ, so not seed noise. Unexplained; must be resolved or disclosed.
- **Baseline difficulty and policy difficulty are different axes.** The
  easy/medium/hard labels are derived from the published dead-backbone fraction
  and are correct for what they measure: how hard a backbone is for baseline
  LigandMPNN. They simply do not predict how hard it is for the policy — M0907
  is harder than M0904 for sampling (0/10,000 vs 8/10,000 measured) and easier
  for the policy (23.3% vs 9.7%, and better motif RMSD). Keep the labels; they
  are honest. The open question is whether the two axes are genuinely
  uncorrelated, which the panel can answer.
- **Training debt is per-backbone.** Policies are trained on one backbone, so the
  debt amortizes over library size, not over targets — unless the panel shows
  transfer across backbones, which it is not designed to test.
- **`experiments/` is gitignored**, so all panel and pilot infrastructure lives
  only on disk.
