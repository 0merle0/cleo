# CLEO — computational-only paper: outline

Working doc. Not part of the LaTeX build.

Purpose: define a paper that stands on computational results alone, so wet-lab
data later slots in as confirmation of a claim that is already complete.

Status key: **[M]** measured · **[P]** partial · **[X]** not run

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

**Claim.** At equal sampling budget the policy dominates the entire temperature
frontier: more passing sequences *and* more distinct substitutions among them.

Measured on M0097, 96 folds per arm **[M]**:

| arm | passing | U (substitutions among passers) |
|---|---|---|
| T=0.2 | 64 | 341 |
| T=0.5 | 33 | 628 |
| T=0.7 | 25 | 781 |
| T=1.0 | 4 | 356 |
| **policy** | **86** | **1,045** |

The sweep has an interior optimum (T~0.7) because the two axes trade off. The
policy sits above all of it.

⚠️ **Open, and it decides this section.** Our U = 1,045 comes from the only 96
designs ever sampled from a trained policy — a floor, not a measurement. Charged
its 2,400-fold training debt, T=1.0 given the same total budget reaches 66
passing / U = 1,824, *above* us. Whether the policy's coverage keeps climbing with
sample count or saturates near the Chao2 estimate (~1,695) is unmeasured, and it
determines whether R1 survives. **See X1.**

### R2 — Rescue: dead backbones become productive  ★ highlight, not the focus

**Claim.** On backbones where sampling yields nothing at any affordable budget,
the policy yields usable libraries.

| backbone | untrained (measured) | trained |
|---|---|---|
| M0907 | **0 / 10,000** | 23.3% (RF3-trained), 54.2% (AF3, n=1) |
| M0904 | **8 / 10,000** = 0.08% | 9.7% (RF3-trained), 63.5% (AF3, n=1) |
| M0097 | 102 / 4,000 = 2.55% | 62.2% / 78.5% |

Context from RFdiffusion2's published 41-site table: **~76% of backbone-slots
yield zero passing sequences**; the median site has 86% dead backbones; 16/41
sites are >=90% dead. Our three sit at sites with 55% / 97% / 93% dead fractions,
so the two hard ones are representative, not cherry-picked.

**[P] n = 3 backbones.** Needs breadth to become a rate rather than an anecdote.

### R3 — The gain is not an artifact of the scoring model  **[M]**

Train against RoseTTAFold3, evaluate with AlphaFold3 — the policy never sees AF3.

| backbone | RF3-trained -> AF3-eval | untrained | ratio |
|---|---|---|---|
| M0097 | 62.2% (sd 21.9) | 2.55% | 24x |
| M0904 | 9.7% (sd 1.2) | 0.08% | 122x |
| M0907 | 23.3% (sd 3.7) | 0% | inf |

Answers the circularity objection on all three targets. Three seeds each.

### R4 — Fragments convert yield into library size  **[M arithmetic, X biology]**

At a matched 4,000-fold budget on M0097, real passing parents, 4 equal slots:

| | parents | recombinants | parts to synthesize |
|---|---|---|---|
| T=1.0 | 102 | 1.1 x 10^8 | 408 |
| **policy** | **1,433** | **4.2 x 10^12** | 5,728 |

**~39,000x the library for ~14x the synthesis**, because the product scales as
P^k while synthesis scales as k*P. Pass rate — exactly what RL improves — is the
dominant lever on library size.

⚠️ This is arithmetic on an untested assumption: that recombinants still fold.
**See X2.**

### R5 — Calibration  **[M]**

σ = 7.75 points run-to-run (E21, 10 runs), so the resolution floor is 9.6 points
at n = 5. Stated up front; every contrast in the paper is reported against it.
Rare in this literature and worth foregrounding rather than burying.

---

## 3. Experiments needed

### X1 — Policy coverage at scale  ★ critical path, ~10 GPU-h

Sample ~2,000 designs from a trained policy (not 96), fold best-of-5, plot U of
the passing set against fold budget on the same axes as the measured T=1.0
trajectory. **Decides whether R1 stands as written.** Cheap, and the paper
currently rests on an extrapolation from 86 sequences.

### X2 — Fragment retention  ★ critical path, ~10 GPU-h

Split passing designs at fixed boundaries, recombine, fold best-of-5, report
retention = recombinant pass rate / parental pass rate. Pre-registered with a
kill criterion in `results/fragments.tex`. Shares its sampling step with X1 —
**one job answers both**.

Robust to a modest answer: even 1% retention leaves 4 x 10^10 against the
baseline's 10^8 at 100%. But unmeasured it is the paper's soft center.

### X3 — Breadth, ~10-15 backbones  ~200 GPU-h

Stratified across the published dead-fraction distribution. Converts R2 from two
rescued backbones into a rate, and gives R1 a population rather than one target.

### X4 — Seed replication on headline backbones  ~60 GPU-h

AF3-trained comparators on M0904/M0907 are n = 1 against σ = 7.75. Needed only if
we want "matches AF3-training" claimed; the transfer claim in R3 does not need it.

**Order:** X1 + X2 (one job) -> X3 -> X4.

---

## 4. Edits and fills

| item | action |
|---|---|
| `results/frontier.tex` | Rewrite to the informative-library frontier; currently a pass-rate story |
| `results/fragments.tex` | Keep pre-registration; fill Block 3 from X2 |
| `results/petase.tex` | **Cut** — wet lab |
| `supplementary/heme_*`, `protease_*` | **Cut** |
| Selection-rule arc (E9/E16/E18/E19) | Demote to one honest paragraph in Discussion — it ends unresolved and is not a result |
| Diversity wording | Must say *per unit budget* everywhere; at matched library size T=1.0 carries more substitutions (1,988 vs 1,045) and a reviewer will find it |
| `abstract`, `intro`, `conclusion` | Rewrite to the two-half thesis |
| Figures | F1 frontier (R1) · F2 rescue + published dead-fraction context (R2) · F3 cross-oracle (R3) · F4 library size vs budget (R4) |
| `main.tex` | Drop Move-3 scaffolding; two results halves plus calibration |

---

## 5. Known soft spots

- **M0097 seed variance is 21.9** against 1.2 and 3.7 on the other two
  backbones. Unexplained.
- **`backbone19` and `selection2` disagree by 35 points** on the same backbone
  and arm — 4.5σ, so not seed noise. Unexplained; must be resolved or disclosed.
- **Difficulty labels do not order results**: the hard backbone beats the medium
  one on both pass rate and motif RMSD. Keep the labels for figure continuity
  only, not as an ordering of difficulty for this method.
- **Training debt is per-backbone.** Policies are trained on one backbone, so the
  debt amortizes over library size, not over targets — unless X3 shows transfer.
