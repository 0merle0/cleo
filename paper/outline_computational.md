# CLEO — computational-first paper: brainstorm + outline

Working doc. Not part of the LaTeX build. Purpose: define the computational-only
story and the experiments that make it compelling, so that wet-lab data slots in
later as confirmation of a claim that already stands on its own.

---

## 1. Landscape — what exists, and where the white space is

**(a) De novo enzyme structure generation.** RFdiffusion2 (41/41 active sites on
its own in-silico benchmark, vs 16/41 for prior SOTA), computational design of
serine hydrolases, metallohydrolases, catalytic motif scaffolding. These papers
own *backbone generation*. Their sequence-design step is uniformly
**ProteinMPNN at low temperature followed by hard in-silico filtering**, then
~96 designs ordered. Sequence design is treated as a solved subroutine.

**(b) RL for inverse folding.** ProteinZero (online RL on inverse folding;
ESMFold + self-derived ddG rewards; embedding-level diversity regularizer;
36–48% reduction in design failure rate vs ProteinMPNN/ESM-IF/InstructPLM on
CATH-4.3), diversity-regularized DPO for peptides, BetterMPNN (GRPO + AF
metrics). **This is the closest prior art to CLEO's core mechanism and must be
confronted head-on.** But: general protein benchmarks (CATH), sequence-level
metrics (designability, recovery), no active sites, no ligands, no libraries.

**(c) ML-guided directed evolution / library design.** MODIFY (co-optimizes
predicted fitness *and* library diversity), ALDE (active learning DE, 12%→93%
yield in 3 rounds), ML-guided cell-free platforms. All of these assume a
**natural enzyme with measurable starting activity and an evolutionary prior
(MSA)**, and optimize a handful of positions.

**White space:** nobody has addressed **library design for de novo enzymes.**
- No MSA / evolutionary prior → MODIFY-class methods are structurally
  inapplicable.
- Starting activity ≈ 0 → ML-guided DE has nothing to bootstrap from.
- The fix is empirically tens of mutations away → SSM and few-position
  combinatorial libraries cannot reach it.
- Current practice (MPNN T=0.1 + hard filters) *deliberately destroys
  diversity to buy in-silico pass rate*.

---

## 2. The fresh angle (three framings, in order of strength)

### Framing A — "The filter becomes the objective."
Every de novo enzyme paper generates sequences, then discards 99% of them with
AF/PLACER/geometry filters. Filtering is not search: it cannot create passing
sequences, it can only find the ones a functionally-blind prior happened to
emit, and the survivors are a small, highly correlated neighborhood of MPNN's
mode. CLEO moves those exact filters *into the sampling objective* via online
GRPO. Memorable, true, and immediately legible to the enzyme-design audience.

### Framing B — "Move the frontier, don't win a point comparison."
The central quantitative object is a **Pareto frontier of fidelity vs
diversity**, not a single win. An MPNN temperature sweep traces the incumbent
frontier; the claim is that CLEO *strictly dominates it* — more diversity at
matched pass rate, higher pass rate at matched diversity. This pre-empts the
two obvious rebuttals ("just raise the temperature", "just filter harder")
by making them the baselines.

### Framing C — "Library size is not the number of independent bets."
Introduce **library-level** metrics where the field uses sequence-level ones.
Low-temperature MPNN designs are highly correlated, so their success
probabilities are correlated, so a 10,000-member library may carry an effective
sample size in the tens. Reframing a library as a *portfolio of independent
bets* is a genuinely new contribution to how the field budgets experiments —
and it is the bridge that makes the wet-lab sections mechanistically
explainable rather than merely favorable.

**Recommended thesis:** *Function-aligned online RL converts in-silico filters
from a rejection step into a design objective, producing de novo enzyme
libraries that are simultaneously more likely to pass the community's own
fidelity criteria and far more diverse — increasing the number of independent
experimental bets per unit of screening effort.*

---

## 3. Experiment plan

Setup: generate backbones with the active-site scaffolding model (RF3/RFD2) over
the **41-site active-site scaffolding benchmark**. Full library protocol run on a
representative subset (4–6 targets spanning chemistries, ideally *including* the
wet-lab enzymes); aggregate frontier statistics reported across all 41.

### E1 — Fidelity/diversity Pareto frontier  ★ headline
- **Arms:** MPNN T ∈ {0.1, 0.2, 0.3, 0.5, 0.8}; MPNN + best-of-N rejection
  sampling; LigandMPNN where a ligand is present; filtered-SFT (fine-tune on
  the sequences that passed — the cheap competitor that must be beaten);
  ProteinZero if runnable; CLEO-GRPO; CLEO-GRPO + diversity reward.
- **Fidelity axis:** AF3 pTM/ipTM/pLDDT, scRMSD to design backbone, catalytic
  motif RMSD, ligand RMSD, PLACER preorganization. Composite "pass" =
  conjunction of thresholds **taken verbatim from published papers, fixed
  before running anything.**
- **Diversity axis:** pairwise identity distribution, unique (position, AA)
  mutation count, per-position entropy, cluster count at 90% identity,
  distance from the MPNN argmax.
- **Claim:** CLEO's curve lies strictly outside the temperature-sweep curve.

### E2 — Held-out oracle / reward-hacking control  ★ credibility
This is the #1 reviewer attack and needs its own figure panel.
- Train the reward on a *cheap* oracle (Boltz + geometry); evaluate on a
  *held-out* one (AF3 + PLACER). Improvements must survive an oracle the policy
  never saw.
- Orthogonal physics check on a subsample (Rosetta ddG / MD stability).
- Composition sanity: no degenerate low-complexity solutions, sane hydrophobic
  core, no Cys blowups, sensible surface charge.
- Explicitly withhold ≥1 fidelity metric from the reward and report it.

### E3 — Breadth across the benchmark
Whole protocol across all 41 sites. Per-target deltas + aggregate. Answers
"is this a platform or an anecdote?" — the difference between a methods paper
and a demo.

### E4 — Method ablations (why it works)
GRPO vs vanilla PG vs best-of-N vs filtered-SFT; KL weight sweep (drift vs mode
collapse); diversity reward on/off; marginal vs fractional consensus divergence
and weight sweep; sampling temperature from the *fine-tuned* policy; is the
**online** part load-bearing vs one-shot offline SFT?

### E5 — Fragment recombination layer (unique to CLEO)
- Do recombined chimeras retain parent pass rate? Pass rate vs fragment
  boundary placement; where epistasis at boundaries bites.
- Currency metric: **cost per distinct passing design** — N sampled designs →
  M library members at K% predicted pass rate.
- Baseline: recombining MPNN T=0.1 fragments (expected to have too little
  fragment diversity for combinatorial expansion to buy anything).

### E6 — Why diversity pays, argued in silico  ★ intellectual core
- **Max-of-N curves** under a held-out fitness proxy: best achieved score as a
  function of screening budget N, per arm. Diverse libraries should show
  heavier right tails — the portfolio argument, quantified.
- **Effective library size:** estimate the number of independent bets given the
  correlation structure of scores within each library. Predicted headline:
  low-temperature MPNN libraries have effective sizes orders of magnitude below
  their nominal size.

### E7 — Closed loop, retrospective / simulated (bridge to wet lab)
Retrospective on existing PETase library-1 data: surrogate trained on round 1
improves round 2 predicted activity. Plus a fully in-silico multi-round
simulation with a held-out oracle as ground truth. De-risks the wet-lab
closed-loop section before it exists.

---

## 4. Figure plan (computational-only version)

| Fig | Content |
|-----|---------|
| F1 | Concept + pipeline: filter-then-select vs reward-then-sample; fragment combinatorics |
| F2 | **Pareto frontier** — per-target panels + aggregate (E1) |
| F3 | Sequence-space maps: UMAP/MDS, mutation spectra, position entropy, distance-to-mode (E1) |
| F4 | Robustness: held-out oracles, PLACER, physics; method ablations (E2, E4) |
| F5 | Benchmark breadth: all 41 sites, per-target deltas (E3) |
| F6 | Effective library size + max-of-N budget curves — "why diversity pays" (E6) |
| F7 | Fragment recombination + retrospective closed loop (E5, E7) |

Wet-lab data later enters as F8+ (heme head-to-head, PETase closed loop,
protease diversity ablation), with F2/F6 serving as the *mechanistic
explanation* for those wins rather than a separate story.

---

## 5. Risks and soft spots

1. **Circularity / reward hacking.** Training and evaluating on AF3 metrics is
   the paper's fatal flaw if not handled. E2 is non-negotiable.
2. **ProteinZero overlap.** Already claims online RL for inverse folding with
   diversity regularization and large failure-rate reductions. Differentiation
   must be explicit and stated early: enzymes and active sites vs CATH;
   library-level vs sequence-level objective; catalytic-geometry rewards vs
   generic ddG; and a physically constructible combinatorial library. Running
   it as a baseline arm would be the strongest possible move.
3. **"Just raise the temperature."** Must be the primary baseline, not a
   footnote. Same for "just filter harder" (best-of-N).
4. **Diversity for its own sake is not a Science-level claim.** E6 is what
   converts a metric into a scientific argument.
5. **Backbone-generator provenance.** If RF3 is unpublished/in-house,
   benchmark comparability and reproducibility both suffer. Consider running
   the whole thing on published RFdiffusion2 outputs as well.
6. **Compute.** AF3 across arms × temperatures × targets × library members is
   the budget driver. Cheap oracle in-loop (Boltz) + AF3 for evaluation only
   is both cheaper *and* the correct held-out-oracle design — a rare case where
   the cheap option is the rigorous one.

**Venue reality check:** this computational-only package is a strong Nature
Methods / Nature Biotech / top-ML-venue paper. Science almost certainly still
requires the wet lab. But building it first is correct sequencing: it defines
exactly which wet-lab experiments are load-bearing and which are decoration.

---

## 6. Open questions for the user

- **RF3** — RFdiffusion3? RoseTTAFold3? In-house or published? Determines
  reproducibility framing and whether the benchmark comparison is portable.
- **Target selection** — which 4–6 enzymes get full library treatment, and do
  they overlap with heme / PETase / protease? Overlap is strongly preferred:
  it is what welds the computational and experimental halves into one paper.
- **Oracle split** — confirm Boltz in-loop, AF3 + PLACER held out for
  evaluation.
- **Compute budget** for AF3 evaluation.
- **Is PLACER accessible** in the pipeline? It is the metric the enzyme-design
  community currently trusts most for preorganization.
