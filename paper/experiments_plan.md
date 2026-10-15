# Computational experiments: CLEO vs. LigandMPNN on the RFdiffusion2 AME benchmark

Working doc. Not part of the LaTeX build. Companion to `outline_computational.md`
(landscape + framing); this file is the concrete experimental build-out.

---

## 0a. What is already public — and what it tells us before we run anything

**Sources located 2026-08-04:**

| Asset | Location | Status |
|---|---|---|
| Inference + benchmark code | `github.com/RosettaCommons/RFdiffusion2` | public |
| 41 active-site input motifs | `rf_diffusion/benchmark/input/mcsa_41/` (`M<mcsa>_<pdb>.pdb`) | public, 41 files |
| Contigs / ligands / fixed atoms per site | `rf_diffusion/benchmark/mcsa_41.json` | public, 41 entries |
| Run configs | `rf_diffusion/benchmark/configs/enzyme_bench_n41.yaml` | public |
| Diversity-adjusted success rate | `rf_diffusion/benchmark/benchmark_success_rate.ipynb` | public |
| **Per-sequence benchmark results** | Zenodo `10.5281/zenodo.17401001` → `figure3/AME_main_benchmark_results/AME_benchmark_results_all.csv` | **downloaded, 328k rows** |
| Generated backbone PDBs | `files.ipd.uw.edu/pub/rfdiffusion2/rfd2_ame_41_backbones.tar.gz` | **404 — dead link** |
| Full n41 run (fallback) | `files.ipd.uw.edu/pub/rfdiffusion2/2024-12-16_08-11-34_enzyme_bench_n41.tar` | live, **177 GB** |

Zenodo deposit is CC-BY-4.0.

### The success criterion is given to us

No threshold guessing required. Their CSV carries the composite directly:

```
chai_motif_pass_and_no_clash = chai_motif_pass AND no_clash
```
supported by `backbone_aligned_allatom_rmsd_chai_motif` and
`ligand_dist_des_ncac_min`. **We adopt this verbatim as the pre-registered
filter.** We are then evaluating against their criterion, not one of our
choosing — which removes the single easiest way to attack a comparison like
this.

### The rescue opportunity, quantified from their own data

Their deposited run is 41 sites × 100 backbones × **40 sequences** each
(164,000 folds per method; note the README says 8 sequences, the deposited data
has 40 — cite the data, not the README):

| Quantity, RFdiffusion2 arm | Value |
|---|---|
| Sequence-level pass rate | **5.8 %** (9,542 / 164,000) |
| Backbones with ≥ 1 passing sequence | 964 / 4,100 = **23.5 %** |
| **Backbones with 0 passing sequences after 40 tries** | **3,136 / 4,100 = 76.5 %** |
| Sites with < 1 % sequence pass rate | 16 / 41 |
| Per-site pass rate range | 0.02 % (M0024_1nzy) → 57.0 % (M0630_1j79) |

(RFdiffusion1 arm, for reference: 2.8 % sequence-level pass rate.)

**This is the paper's opening.** RFdiffusion2 scaffolds all 41 active sites —
that claim is theirs and it stands. But at the level of the individual design,
**76.5 % of generated backbones yield no usable sequence at all**, and 40
LigandMPNN samples per backbone is not a shallow search. The gap between "all
41 sites solved" and "three quarters of individual designs discarded" is
precisely the space this paper occupies, and it is a *sequence-search* gap, not
a backbone gap.

Two things follow immediately:

- **The compute-matched control is pre-strengthened.** "Just sample more" has
  already been run for us at n = 40 and it left 76.5 % of backbones unusable.
  Our bar is to beat 40 additional samples, not 8.
- **Site stratification is free.** Their per-site pass rates give a defensible,
  published basis for choosing the P1 subset across the difficulty range,
  rather than picking sites that flatter us.

### RESOLVED: everything is on local IPD storage

The public `rfd2_ame_41_backbones.tar.gz` URL 404s, but the file itself is on
the lab share. Nothing needs to be downloaded.

```
/net/lab/pub/rfdiffusion2/
  ame_results/rfd2_ame_41_backbones.tar.gz     12.6 GB  ← the 404'd file
  model_weights/RFD_140.pt, RFD_173.pt          2.3 GB  ← RFdiffusion2 weights
  sifs/chai.sif                                 8.4 GB  ← the Chai-1 oracle
  sifs/rfdiffusion.sif, mlfold.sif
  2024-12-16_08-11-34_enzyme_bench_n41.tar       52 GB  (apparent 177 GB, sparse)
  2025-03-17_ablation_subsampled_designs.tar    102 GB
  rfd2_active_designs.zip                        36 MB
```

**Verified join, 2026-08-04.** The tarball holds 8,200 members: 4,100 `.pdb`
backbones plus 4,100 matching `.trb` files. Member names map to the results CSV
by stripping the `rfflow_fixed-ligand_` prefix from `design_id`:

```
design_id  rfflow_fixed-ligand_run_M0024_1nzy_cond0_0-atomized-bb-True
tar member                     run_M0024_1nzy_cond0_0-atomized-bb-True.pdb
```

**4,100 / 4,100 backbones matched, 0 missing. All 3,136 rescue targets are
present and individually addressable.** The index is committed at
`paper/figures/data/rfd2_AME_backbone_index.csv` (site, design_id, pdb member,
n_pass, n_seq, rescue_target).

The `.trb` files matter as much as the PDBs — they carry the contig mapping, so
motif positions are recoverable per backbone and can be handed straight to
CLEO's `fixed_residues`. Without them, motif indices would have to be inferred.

Consequence for the paper: rescue is measured against **their** published
backbones and **their** published failure labels, using **their** success
criterion, with **their** oracle container. That is the strongest available
version of the claim, and it is now unblocked end to end.

---

## 0. Why this benchmark is the right target

RFdiffusion2's Active site Model Enzyme (AME) benchmark is public — inference
*and* benchmarking code at `RosettaCommons/RFdiffusion2` — and its protocol is
fully specified:

| Component | Their setting |
|---|---|
| Active sites | 41 (M-CSA × PARITY curated, all reactants/cofactors present) |
| Backbones per site | 100 |
| **Sequences per backbone** | **8** |
| Sequence design | LigandMPNN, ligand-aware + motif-rotamer-aware, with packing |
| Refolding oracle | Chai-1 |
| Headline result | scaffolds for all 41 sites (vs. 16 for prior methods) |
| Total folding runs | 41 × 100 × 8 = **32,800** |

Three consequences we should exploit:

1. **The baseline is a published number we can reproduce**, not a strawman we
   built. This kills the most common reviewer objection to method comparisons.
2. **8 sequences per backbone is a tiny search.** That is the opening. Their
   contribution is the backbone; sequence design is a subroutine they spend 8
   samples on. Our claim is that this subroutine is leaving a large amount on
   the table.
3. **The oracle call is the unit of cost**, and their protocol fixes it at 8 per
   backbone. Every comparison below is denominated in folds, which is
   hardware-independent and directly reproducible.

**Open item:** the README documents metric *names*
(`contig_rmsd_{a}_{b}_{s}`, `metrics.IdealizedResidueRMSD.rmsd_constellation`,
`motif_ideality_diff`) but not pass/fail *thresholds*. First task is to extract
the exact thresholds from the benchmark source and the paper, then freeze them
before running anything. We evaluate against their criteria, never ours.

**Compatibility check (do this early):** their LigandMPNN runs in
motif-rotamer-aware mode with packing. CLEO's `model_type: ligand_mpnn` path
must match that mode or the baseline is not apples-to-apples. If it cannot
match, say so explicitly in Methods and report both.

---

## 1. The measurement currency

Everything is plotted against **cumulative folds** (structure-prediction calls).
Both arms pay this cost, it dominates wall time, and it is the quantity a lab
actually budgets.

### Primary metrics

| Metric | Definition | Why it matters |
|---|---|---|
| **Unique mutations per fold** | distinct (position, AA) pairs vs. reference, ÷ folds spent | Raw search efficiency — how much sequence space you buy per unit compute |
| **Unique *passing* mutations per fold** | same, restricted to designs clearing the filter battery | The honest version. Diversity among failures is worthless |
| **Distinct passing clusters per fold** | sequence clusters at 90% identity among passing designs, ÷ folds | Number of genuinely independent solutions |
| **Folds to first pass** | per backbone | Practical: how long until this backbone is usable |
| **Pass rate** | fraction of folded sequences clearing all filters | The metric the field already uses |

### Methodological commitment

**Diversity is measured on the passing subset only.** A method can trivially win
on raw diversity by emitting garbage. Reporting diversity among *passing*
designs is the only fair comparison, and stating this up front converts a
likely reviewer attack into a credibility signal.

### Compute accounting (logged for every run)

- Folds (primary)
- GPU-hours split by component: MPNN forward / GRPO backward / oracle
- Wall time, device type, device count

Report folds as the headline and GPU-hours in supplement. GRPO carries an
upfront training cost that pure sampling does not; the crossover point is a
result, not something to hide.

---

## 2. Arms

| ID | Arm | Role |
|---|---|---|
| A0 | LigandMPNN, published AME settings, 8 seq/backbone | Reference — must reproduce their number |
| A1 | LigandMPNN T = 0.1 | Low-temperature control (current production practice) |
| A2 | LigandMPNN T ∈ {0.2, 0.3, 0.5, 0.8} | The incumbent diversity/fidelity frontier |
| A3 | Best-of-N rejection sampling at matched folds | "Just filter harder" |
| A4 | Filtered-SFT: fine-tune on sequences that passed | Cheapest competitor; must be beaten |
| A5 | CLEO-GRPO | Ours |
| A6 | CLEO-GRPO + mutation-diversity reward | Ours + explicit diversity pressure |

A2 and A3 are not optional. They are the two rebuttals every reviewer will
reach for, and the argument only lands if they are primary arms rather than
footnotes.

---

## 3. Experiments

### E1 — Matched-fold head-to-head  ★ core

Run every arm on the same backbones at the **same total fold budget**, and
report the metric table from §1 per active site plus aggregated.

Claim: at equal folds, CLEO produces more passing designs, more distinct
passing clusters, and more unique passing mutations.

Guard: budget matching must include GRPO's training folds. Training folds count
against CLEO's total. Anything else is not a fair comparison.

### E2 — Diversity yield curves  ★ the figure this project has been building toward

X-axis: cumulative folds. Three panels, each a curve per arm:

- (a) cumulative unique (position, AA) mutations
- (b) cumulative unique mutations **among passing designs**
- (c) cumulative distinct passing clusters at 90% identity

**Predicted shape, and the reason this is the paper's best figure:** low-T
LigandMPNN *saturates*. Past some fold count it returns sequences it has
already effectively produced, so curve (b) flattens — additional compute buys
nothing. CLEO's curves should keep climbing. A visible asymptote in the
incumbent method is "filtering is not search" made quantitative, in one panel.

Report the saturation fold count explicitly; it converts the thesis into a
number.

### E3 — Rescue of failed designs  ★ headline novelty

The most compelling experiment in the set, and the one nobody else can run,
because it reframes what a filter failure *means*.

Standard practice discards a backbone whose 8 sequences all fail. But that
conflates two very different things: a bad backbone, and a search that was too
shallow to find the sequence the backbone needed.

**The partition already exists** in their deposited data (§0a) — no baseline
run required:
- **FAIL** — 0 of 40 sequences pass: **3,136 backbones (76.5 %)**
- **PASS** — ≥ 1 of 40 passes: 964 backbones (23.5 %)

On the FAIL set, spend additional folds three ways:
- R1: more LigandMPNN samples (40 → 100 → 400) — *the compute-matched control*
- R2: higher-temperature LigandMPNN at matched folds
- R3: CLEO-GRPO on that backbone

**Headline number:** percentage of the 3,136 discarded backbones that CLEO
rescues, and folds spent per rescue.

Sampling design: stratify the FAIL set by site difficulty using their published
per-site pass rates. Include sites at both extremes — a method that only
rescues designs at easy sites (M0630_1j79, 57 %) is far less interesting than
one that rescues at hard ones (M0024_1nzy, 0.02 %).

R1 is load-bearing. Without it the result reads as "we spent more compute."
With it, the claim becomes: *at equal additional compute, RL search rescues
designs that additional sampling cannot.*

**Also report the converse:** does CLEO improve backbones that already passed —
more distinct solutions, better margins? A method that only helps failures is
a repair tool; one that helps both is a replacement.

**Sub-analysis — is rescuability predictable?** Correlate rescue success with
backbone properties (design-stage contig RMSD, motif burial, radius of
gyration, secondary-structure content). If a subset is genuinely unrescuable,
that is an honest and *interesting* finding: it separates "the backbone is
bad" from "the search was too shallow," which is exactly the distinction the
field currently cannot make.

### E4 — Compute efficiency

Not a standalone experiment; a cross-cutting analysis over E1–E3.

- Folds per distinct passing design, per arm
- Cumulative distinct passing designs vs. cumulative folds — **the crossover
  plot**. GRPO pays upfront and wins later; where is the break-even?
- GPU-hour breakdown by component

If the crossover happens below ~8–64 folds per backbone, CLEO is a drop-in
replacement. If it needs thousands, it is a rescue tool for hard targets. Both
are publishable; they are different papers, and E4 tells us which one we have.

### E5 — Held-out oracle  ★ credibility, non-negotiable

Train the reward on a cheap oracle, evaluate on one the policy never saw.

- **In-loop reward:** Boltz-2 (or Chai-1) + motif RMSD
- **Held-out evaluation:** AF3 + PLACER, never in the reward
- Withhold ≥ 1 metric from the composite reward and report it separately
- Orthogonal physics check (Rosetta ddG / MD) on a subsample
- Composition sanity: no low-complexity collapse, sane core packing, no Cys
  blowups

Circularity is the single most likely reason this paper gets rejected. Every
headline claim must survive an oracle swap.

### E6 — Policy transfer / amortization

Determines whether this is practical or merely possible.

- **Per-backbone policy** — fine-tune on one backbone. Best performance,
  highest cost.
- **Per-site policy** — fine-tune across the 100 backbones of one active site.
- **Generalist policy** — fine-tune across N sites, evaluate on **held-out**
  sites.

If a generalist transfers to unseen active sites, the upfront cost amortizes to
near zero and CLEO becomes a drop-in replacement for LigandMPNN in anyone's
pipeline. That is the strongest practical form of "better than vanilla MPNN for
library design," and it is worth a figure panel of its own.

### E7 — Effective library size

Estimate the number of *independent* bets each library represents, given the
correlation structure of pass/fail outcomes within it.

Predicted: low-T MPNN libraries have effective size far below nominal size,
because their members are near-duplicates whose successes and failures are
correlated. This is the deepest version of the paper's claim and the bridge to
the wet-lab sections.

### E8 — Mutation-diversity reward (optional reward term)

Add the batch mutation-diversity reward
(`cleo.design.utils.mutation_diversity`) as an optional term alongside the
benchmark metric, and run it as its own arm (A6).

Both variants are already implemented: marginal/exclusive (credit only for
mutations no other batch member carries — sparse, strong pressure) and
fractional 1/k (smoother, robust to batch size). Sweep the weight; report both.

The interesting question is not whether it raises diversity — it will — but
whether it raises **passing** diversity: unique mutations among designs that
clear the 1.5 A motif cutoff. A diversity term that buys sequence spread at the
cost of pass rate is a worse trade than simply raising temperature, and panel 2B
is where that shows up.

### E9 — Oversample cheap, fold expensive  ★ highest value per unit effort

The asymmetry the whole project runs on: **sampling from LigandMPNN is nearly
free; folding is essentially the entire cost.** The baseline spends its budget
folding 8-40 sequences per backbone chosen with no regard to redundancy — and at
low temperature many are near-duplicates, so folds are spent re-testing the same
hypothesis.

Protocol: sample far more sequences than the fold budget (10^2-10^4 per
backbone), then fold only a maximally diverse subset of size N.

- **Selection rules to compare:** random (control); greedy max-min Hamming;
  cluster at an identity threshold and take representatives; diversity subject
  to a policy-likelihood floor.
- **Budget:** N folds per backbone, identical across arms. The comparison is
  entirely about *which* N you fold.
- **Arms (2x2):** {baseline LigandMPNN, CLEO} x {random N, diverse N}. This
  factorizes selection from policy, so we learn whether the gain comes from a
  better policy, better selection, or both.

Directly optimizes the headline quantity — unique passing mutations **per
fold** — and it is cheap because the expensive half is capped by construction.
It also applies to the baseline, so it is an honest improvement offered to both
arms rather than an advantage reserved for ours. If diverse-N lifts the baseline
substantially, that is a real result and belongs in the paper either way.

Composes with E8: the diversity *reward* shapes what the policy proposes, the
diversity *selection* decides what gets folded. Either alone may suffice; the
2x2 says which.

---

## 3b. Metric calibration: status and resolution

**Gate: no pilot runs until we can compute every metric we intend to report.**

### What was checked

`ligand_dist_des_ncac_min` depends only on the design — no predictor — so it is
the one benchmark quantity checkable against their published values without
running an oracle. Our implementation does **not** reproduce it: median absolute
error 0.42 A, max 1.68 A, 0/40 exact, `no_clash` boolean agreement 38/40
(`experiments/ame/calibrate_metrics.py`).

Ruled out, none reproducing the published number:

- ligand atom subsets: all / DAD only / MG only / `partially_fixed_ligand`
  members / non-members
- excluding motif residues from the N/CA/C set; CA-only; N/CA/C/O
- the `unidealized/` PDB variant from `enzyme_bench_n41.tar` (median error 0.47 A
  — slightly worse than the idealized one)
- `metrics_cache/*.csv` in that tar holds only `IdealizedResidueRMSD.*`, not the
  ligand distance

### Resolution: compute both arms ourselves, one implementation

This turns out not to block the work, for two reasons.

1. **`no_clash` never needs recomputing.** It is constant across all 40
   sequences of a backbone, so it is a per-backbone constant we can simply read
   from their published CSV. We already do — it is how the 2,837 true rescue
   targets were defined. Nothing in the reward depends on our version of it.

2. **The comparison must be internally consistent, not externally identical.**
   Scoring CLEO designs with our implementation while taking baseline numbers
   from theirs would be an apples-to-oranges comparison regardless of whether our
   implementation matched. The fix is to recompute *both* arms with one
   implementation and one oracle.

That is now possible, and by the simplest route available: **we regenerate the
baseline ourselves.** MPNN sampling is cheap — the expensive step is folding, and
we have to fold our own designs anyway. `experiments/ame/sample_baseline.py`
samples from the vanilla LigandMPNN weights shipped in the repo
(`src/cleo/design/protein_mpnn_utils/ligand_protein_mpnn_weights.pt`) at their
published T = 0.1, with motif residues pinned from the `.trb`.

It does this by subclassing `PolicyMPNN` and removing only the training
machinery, so the baseline is **the same code path as CLEO with zero gradient
steps**. Any difference between the arms is then attributable to training rather
than to two different implementations of "sample from MPNN". Their deposited
sequences under `ligmpnn/backbones/` (32,801 files = 4,100 x 8) remain available
as a cross-check on our sampler, but are not needed for the comparison.

So:

- fold baseline sequences and CLEO sequences with the **same** predictor (AF3)
- score both with the **same** module (`rfd2_benchmark.py`)
- use their published pass/fail labels as an independent cross-check and to
  define the rescue partition, never as the basis of the head-to-head

Their Chai-1 predicted structures are not in the deposit (only `backbones/` and
`packed/` under `ligmpnn/`), so calibrating the sequence-dependent motif RMSD
against their exact values is not possible. Recomputing both arms is not a
workaround for that — it is the correct design either way.

### Remaining pre-pilot checks

- [x] Metric implementation self-consistent: scoring a design PDB against itself
      gives motif RMSD 4e-7 A with 0 missing motif atoms, so the `.trb` atom
      selection and the superposition agree with the structure
- [x] Baseline sampler reproduces the pinned motif exactly (D78/D81/R114/K138 on
      `M0058_1cju_cond4_45`, 8/8 sequences) at full backbone length
- [x] Motif RMSD sane on AF3 predictions and discriminating between backbones of
      known opposite outcome (below)
- [x] One backbone end-to-end through AF3 -> metrics -> per-sequence CSV
- [ ] Rerun `calibrate_metrics.py` if the design-PDB provenance is ever
      resolved; document the residual discrepancy in Methods if not

### Side-chain context is not a detail — it is most of the benchmark

The gate ran two backbones from `M0664_2dhn` with opposite published outcomes
(`cond29_29`, 40/40 passing under their Chai eval; `cond29_6`, 0/40), 8
sequences each, sampled at T = 0.1 and folded with AF3:

Numbers below use the **corrected** motif metric (see next section). An earlier
version of this table used a global superposition over the contig-atom subset
and reported 0/8 -> 3/8; that overstated the effect and is superseded.

| backbone | side-chain ctx | motif RMSD min / median | passing | AF3 ptm med |
|---|---|---|---|---|
| cond29_29 (their 40/40) | off | 1.17 / 1.45 | 5/8 | 0.90 |
| cond29_29 | **on** | 0.75 / **0.80** | **8/8** | 0.90 |
| cond29_6 (their 0/40) | off | 1.56 / 1.93 | 0/8 | 0.78 |
| cond29_6 | **on** | 1.65 / 2.04 | **0/8** | 0.81 |

Side-chain context takes the positive control from 5/8 to 8/8 and halves its
median RMSD, while leaving the negative control at 0/8. That asymmetry is the
useful part: the fix helps where the published data says it should and not where
it says it should not, so it is correcting a real deficiency rather than
inflating the metric. **AF3 ptm is unchanged at 0.90**, confirming the gain is
rotamer placement and not better folding — the benchmark measures side-chain
atoms, so a policy shown only the motif backbone has no reason to hold the
rotamer.

Implemented as `ligand_mpnn_use_side_chain_context`, now configurable in
`PolicyMPNN` (default 0 to preserve existing runs) and set to 1 in every
generated AME config, so CLEO does not train under a handicap its baseline
lacks. `--side-chain-context 0` is retained as an ablation: it isolates how much
of this benchmark is rotamer placement, which is a result in its own right.

### The motif metric, checked against their actual definition

Reading the published *wording* rather than inferring from the CSV schema found
two errors in our implementation, both of which made good designs look bad. The
column name decodes in `per_sequence_metrics.py` as
`{align_to}_aligned_{rmsd_to}_rmsd_{source}_{target}`, so
`backbone_aligned_allatom_rmsd_chai_motif` means:

1. **Superpose on N/CA/C/O of the motif residues only** — a *local* alignment.
   We were aligning on the entire protein backbone.
2. **RMSD over all heavy atoms of those residues** — we were using the `.trb`
   contig-atom subset (often just 2-3 tip atoms).

The paper states the same thing in prose: a success is RMSD of all heavy atoms
in the catalytic residues < 1.5 A when aligned on the backbone N, CA, C of those
catalytic residues.

Local alignment is the stricter of the two conventions — a global fit can absorb
motif error into a whole-body rotation — but scoring only tip atoms was harsher
still, and that dominated. Correcting both, and adding terminal-group symmetry
resolution (Asp/Glu/Arg/Phe/Tyr, as their `sidechain_symmetry_resolved` does):

| backbone | their published | ours, corrected metric |
|---|---|---|
| cond29_29 | 40/40 pass | **8/8** pass, median 0.80 A |
| cond29_6 | 0/40 pass | **0/8** pass, median 2.04 A |

**Both controls now agree with the published outcome.** The 3/8-vs-40/40 gap was
our metric, not our designs, and it is closed — no appeal to packing, oracle
disagreement or sample size required.

One caveat on absolute values. Their pipeline compares each prediction against
`unideal`, `packed` (the LigandMPNN-packed design) and `ref` (the native active
site); the deposited `*_chai_motif` column is one of those, most likely not the
idealized backbone we hold. So our numbers need not equal theirs
sequence-for-sequence even with the definition matched. That is acceptable for a
baseline-vs-CLEO comparison run through one pipeline, and it explains why
`calibrate_metrics.py` was never going to reproduce their values.

### Ligand placement is not part of the benchmark — resolved, not a blocker

Recorded here because it was briefly mistaken for a calibration problem. The
deposited results have exactly these columns:

```
backbone_aligned_allatom_rmsd_chai_motif, ligand_dist_des_ncac_min,
chai_motif_pass, no_clash, chai_motif_pass_and_no_clash,
benchmark, source, design_id, Number of Residue Islands
```

There is **no ligand-pose metric**. `ligand_dist_des_ncac_min` is the clash
check, nothing more. Across all 328,000 rows, `chai_motif_pass` is exactly
`rmsd < 1.5`, `no_clash` is exactly `dist > 1.5`, and the composite is exactly
their conjunction. The `chai_pocket_aligned_ligand_{i}_rmsd < 2.5` criterion
noted earlier in this plan came from their *source*, not their results: a
quantity their pipeline can compute but that they neither report nor score on.
There were never any published values to calibrate against.

The high ligand RMSD we saw (4.62 A median on the positive control) is real and
is not ours to fix:

- atom matching is complete (14/14 PH2 atoms in both structures), so it is not
  a silently shrunken comparison set
- it is insensitive to the alignment used — global backbone 4.88 A vs pocket
  4.89 A — so it is not our pocket definition, which is what was first suspected
- it is uncorrelated with motif accuracy: the best motif (1.32 A, passing) had
  the worst ligand RMSD (5.61 A); a failing motif (1.55 A) had the best (1.98 A)

So AF3 places the ligand several angstroms from the design pose more or less
independently of how well it reproduces the motif — consistent with the known
unreliability of ligand placement on de novo designs with empty MSAs, though we
have no independent evidence either way.

**Decision: reward on motif RMSD only.** `no_clash` is read from the design
(per-backbone constant, no sequence can change it). Ligand RMSD is logged as a
diagnostic under the neutral name `ligand_rmsd_max`, with no pass/fail boolean
so that no future reward can pick it up by accident, and is never reported as a
benchmark number. Optimizing a quantity that has no ground truth, under an
oracle whose ligand placement is unvalidated here, is a reward-hacking risk with
no upside for the headline claim.

Ligand placement does matter for real enzyme design even though AME does not
score it. Revisit later as an explicitly CLEO-defined metric with its own
validation — never smuggled in as though it were theirs.

---

## 4. Staging

| Phase | Work | Purpose |
|---|---|---|
| **P0** | **Metric calibration gate (§3b)** — confirm every metric we intend to report computes end-to-end; verify CLEO's LigandMPNN matches their motif-rotamer-aware + packing mode | Prove we can *measure* before claiming to *improve* |
| **P0.5** | **Pilot runs** — a few backbones per class through the full loop (sample → AF3 → metrics → reward → GRPO step) | Shake out wiring at small scale before committing GPU time |
| **P1** | E1 + E2 on a stratified subset (8–12 sites spanning chemistry and difficulty) | The core claim, cheaply |
| **P2** | E3 rescue on the same subset; E4 accounting throughout | The headline |
| **P3** | Scale to all 41 sites; E5 oracle swap; E6 transfer; E7 | Breadth and credibility |

Reproducing the full baseline is 32,800 Chai-1 folds for a *single* replicate.
Do not start there. Pick the subset on stratified difficulty using their own
published per-site results, so the subset is defensible rather than convenient.

---

## 5. Risks

1. **Circularity.** Optimizing the metric we report. E5 is the answer; it is not
   optional and should appear in the main text, not supplement.
2. **Compute.** Arms × temperatures × sites × folds multiplies fast. Cheap
   in-loop oracle + expensive held-out evaluation is both cheaper *and* more
   rigorous — the rare case where the affordable choice is the defensible one.
3. **Baseline fidelity.** If we cannot reproduce A0, every downstream comparison
   is contestable. P0 exists for this reason alone.
4. **Rescue rate could be low.** If CLEO rescues only a few percent of FAIL
   backbones, E3 stops being the headline and E2 has to carry the paper. Worth
   knowing early — run E3 on a small subset before committing to it as the
   story.
5. **LigandMPNN mode mismatch.** Motif-rotamer-aware + packing is a specific
   configuration. A mismatch here is a silent confound that a reviewer familiar
   with the benchmark will catch immediately.

---

## 6. What to log, from run one

Per generated sequence: site ID, backbone ID, arm, sampling temperature, policy
checkpoint step, sequence, all `contig_rmsd_*` metrics, pass/fail per individual
filter and composite, cumulative folds at time of generation, wall time, GPU
seconds by component.

Per-sequence fold-count provenance is what makes every curve in §1 and §3
reconstructable after the fact. Retrofitting it later means re-running
everything.

---

## 7. Status board — updated 2026-08-12

What the pilot has actually settled, per experiment. Three entries changed
materially and one is newly at risk; read those before planning compute.

| ID | Experiment | Status | What we now know |
|---|---|---|---|
| **E1** | Matched-fold head-to-head | **partial** | Done on M0097 only. At the 8-fold benchmark budget: T=0.1 → 5.0 passing / **1.0** cluster; CLEO peak → 3.1 passing / **3.1** clusters. We lose on hit rate, win 3× on independent solutions. Blocked on A2 for the frontier. |
| **E2** | Diversity yield curves | **CLEO half done** | M0097 trajectory: 3200 folds → 97 passing → 97 clusters @90 % id, 33 % mean identity, still climbing. Baseline curves need A2. |
| **E3** | Rescue | **not started** | Target picked: the three *near-miss* backbones (1.61–1.69 Å best-of-40, 0/40 published). Clean because there is no positive result of theirs to fail to reproduce. Current hint only: 4/3200 passing on two backbones where all baselines give 0. |
| **E4** | Compute efficiency | **blocked on E1/E2** | — |
| **E5** | Held-out oracle | **in progress — see §8** | Was fully circular as piloted (reward AF3 motif RMSD, evaluation AF3 motif RMSD). On the chimera objective this is now measured: training under an rf3/boltz/protenix round robin and reporting under held-out AF3. Three of four cells of the batch × pool design are done; the single-oracle large-batch cell (E15) is the one that decides whether the credit goes to the pool or to batch size. |
| **E6** | Policy transfer | **not started** | — |
| **E7** | Effective library size | **partially answered** | Among passing designs, cluster count already *equals* passing count. Effective size ≈ nominal for CLEO; for T=0.1 it is 1 regardless of nominal size. |
| **E8** | Mutation-diversity reward | **DROPPED** | Superseded: the training trajectory *is* the library, so diversity comes free without a reward term to tune and defend. Arm A6 retired. |
| **E9** | Oversample cheap, fold expensive | **answered — negative** | Greedy max-min selection is **worse than random** at every budget (800 folds: 51 as-sampled / 25 random / **6** max-min). It selects outliers, which fail. Gating on quality narrows but does not close the gap. Diversity-first acquisition is not viable; the defensible use is post-hoc subsetting of known-passing designs (`select_library.py`). |

### New experiments this pilot implies

| ID | Experiment | Why | Cost |
|---|---|---|---|
| **E10** | **Temperature sweep**, T ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, matched folds | **The blocking experiment.** Builds the incumbent frontier that every claim is measured against. If an intermediate T reaches CLEO's coverage at CLEO's pass rate there is no frontier shift and the narrative collapses. T=0.2/0.3 are the dangerous points. | med |
| **E11** | **Reward shaping**: rank-normalised + saturating at the 1.5 Å pass threshold | Fixed bounds 0.5–20 Å collapse the within-batch reward spread once median RMSD hits 2–3 Å, which is exactly when the policy starts drifting (diversity bottoms at the performance peak, then rises as pass rate falls). A hinge at threshold also stops the gradient pulling every passing design onto one mode. Expected to raise pass rate *and* diversity. | low |
| **E12** | **Early stopping / trajectory length** | Peak is at step ~40; steps 46–200 are decay. 5/5 budget-matched draws pass at peak vs 1/5 at final. | free |
| **E13** | **Consensus anchoring** — seed/anchor search at the low-T consensus | Backbone-dependent by construction: on M0097 the passing designs sit ≥104 mutations from the low-T mode, so anchoring there suppresses the win. On backbones where nothing passes at all it may supply the basin the search never finds. Run as a control on M0097, a candidate fix on M0904/M0907. | low |
| **E14** | **ESM embedding view** of the passing libraries | One-hot PCA answers "how far apart in Hamming", not "how different biochemically". ESM mean-pooled embeddings answer the second. Presentation/diagnostic only — quantitative claims stay on Hamming so figure and tables agree. | low |

### Reading of the whole board

The narrative is now **diversity at matched fidelity**, not pass rate. E10 decides
whether that narrative survives; E5 decides whether anyone believes the numbers;
E11/E12 are cheap and likely to improve every downstream result, so they should
land before the expensive arms are run.

---

## 8. Chimera objective: multi-oracle transfer — plan as of 2026-10-14

This section is the concrete build-out of **E5** for the *chimera* objective
(split passing designs into fragments, recombine, reward the recombinants).
E5 above states the principle — never report under the predictor you trained
on. What follows is what has actually been measured against it, the one cell
that is missing, and what it costs to close.

### 8.1 What is measured

Held-out AF3 pass on the in-loop chimera set versus the oracle-balanced
training level at the same step. "Gap" is training level minus held-out AF3, so
positive means the training number overstates.

| arm | config family | batch | oracles | mean gap | per backbone (M0097 / M0255 / M0315) |
|---|---|---|---|---|---|
| single-oracle, small batch | `fragrl4` | 8 | rf3 | **+0.439** | +0.28 / +0.94 / +0.09 |
| pool, small batch | `fragmulti` | 8 | rf3+boltz+protenix | **+0.176** | +0.33 / +0.20 / +0.00 |
| pool, large batch | `fragb32` | 32 | rf3+boltz+protenix | **−0.001** | −0.01 / +0.02 / −0.01 |

Both comparisons are one-factor. `fragrl4` and `fragmulti` are both
`batch_size: 8, seed: 1` and differ only in the oracle function; `fragmulti`
and `fragb32` share the pool and differ only in batch size.

Worst single case: **M0255 `fragrl4` scored 1.000 under RF3 and 0.062 under
AF3** — a 94-point gap at 22 SE. That is the project's cleanest reward-hacking
demonstration and should be a figure panel, not a footnote.

Precision caveat: the batch-8 cells are n = 32 chimeras from one step each
(2 SE 0.09–0.16); the batch-32 cells are n = 128. Directions are 3.4–22 SE and
safe, **magnitudes are not publishable at that precision** — see T1.3.

### 8.2 The missing cell, and why it decides the claim

The 2×2 has three cells. **Batch 32 with a single oracle does not exist.**

|  | batch 8 | batch 32 |
|---|---|---|
| rf3 only | +0.439 | **missing** |
| 3-oracle pool | +0.176 | −0.001 |

So the pool is only *shown* to help at batch 8, where the gap is large anyway.
Two readings survive the current data and they imply different papers:

- **Pool necessary.** The missing cell reopens the gap → multi-oracle training
  is what makes the chimera objective generalize, and batch size is a
  prerequisite rather than the cause.
- **Batch sufficient.** The missing cell transfers like `fragb32` → batch size
  alone closes the gap, the pool is redundant, and the claim must be reframed
  as "adequate batch size," not "multiple oracles."

The honest current lean is toward the second, because batch 32 closed the gap
to zero *with* the pool and nothing isolates the pool's contribution there.
**One run settles it.** Better to learn this from E15 than from a referee.

### 8.3 Experiments

#### E15 — Complete the 2×2: batch 32, single oracle  ★ decisive, run first

Forced single oracle inside the *same* `multi_oracle_from_df` call
(`oracle: rf3`), so dispatch, per-oracle rundir layout and column renaming stay
byte-identical to `fragb32`. This is a tighter pair than `fragrl4`-vs-
`fragmulti`, which swapped the oracle function itself.

- Start with **M0255** — largest single-oracle gap (94 points), so most room to
  discriminate. Then M0097. M0315 is a control, not evidence: it transfers even
  at batch 8.
- 150 steps, `seed: 1`, `af3_probe_every: 25`.
- **Cost:** from job 29035611's own rate (43.5 h for ~169 steps on 4 GPUs),
  ~15.4 min/step → ~38 h wall, **~154 GPU-h per run**.
- Staged and ready, not launched: `experiments/ame/fragb32rf3/*.yaml` and
  `submit_fragb32rf3.sh`. Diff against `fragb32` is exactly three lines —
  `N_steps: 150`, `oracle: rf3`, `af3_probe_every: 25`.

#### E16 — Oracle dose-response at batch 32

One → two → three oracles, held-out AF3, batch held at 32. The 1-oracle point
comes from E15 and the 3-oracle point already exists; only **{rf3, boltz}** is
missing. Three points on a monotone trend are much harder to attribute to batch
size than two. **Cost:** ~154 GPU-h × backbones. Do not start before E15
reports — if the pool is redundant at batch 32 this is ~600 GPU-h spent
defending a claim that needs reframing anyway.

#### E17 — Leave-one-out within the pool  ★ strongest form of E5

Train on two predictors, evaluate on the third *and* AF3: {rf3,boltz} → protenix,
{rf3,protenix} → boltz, {boltz,protenix} → rf3. This is generalization to a
structure predictor never seen, not merely to AF3, and it is the version of the
claim a sceptical reviewer will ask for. Expensive — needs new runs per arm, so
it is gated behind E15/E16.

#### T1 — Offline tier: no training, highest value per GPU-hour

All three run on data already on disk.

| ID | What | Why it matters |
|---|---|---|
| **T1.1** | AF3-vs-steps curves, stride 0/30/60/90/120, on `fragrl4` vs `fragb32` | **DONE 2026-10-14 — see 8.5.** Nine valid stride-points per arm instead of one plateau reading each. Not the overfitting curve this row predicted; see 8.5 for what it actually shows. Cost was ~8 min/probe at 32 chimeras and ~15 min at 128, not the ~28 min quoted here. |
| **T1.2** | Full predictor cross-tab at plateau on `fragrl4` | `fragrl4` trained on RF3 alone, so boltz, protenix *and* AF3 are all genuinely held out — a 3-wide held-out panel for the single-oracle arm with no retraining. The pool arms only have AF3 held out; that asymmetry is what E17 fixes. |
| **T1.3** | Pool each batch-8 cell over ~4 plateau steps → n = 128 | Brings the batch-8 cells to `fragb32` precision. Required before any magnitude in §8.1 goes in the paper. |

#### T2 — The library evaluation  ★ still the outstanding gap

Everything in §8.1 is the **in-loop** chimera set, which recombines the 32
parents of one rollout. The paper's claim is about *library* design and the 65.6
figure came from the post-hoc eval recombining **256**. Until the no-transfer-gap
result is reproduced on that wider set, the headline has not faced the test it
has to pass.

### 8.4 Risks specific to this section

Additions to §5.

6. **`seed:` is the model seed, not the diffusion seed.** Verified in
   `src/cleo/design/train_policy.py:39-46`: it seeds `random`, `numpy` and
   `torch` only — LigandMPNN sampling, GRPO rollout, init. The predictors'
   diffusion seeds are **not** set by it. `rf3_from_df`/`af3_from_df` pass no
   seed and instead *read it back* out of the `seed-<S>_sample-<K>/` output
   layout (`oracle.py:40`) as `{step_name}_seed`; `protenix_from_df` accepts
   `cfg.seeds` (`oracle.py:1041-1042`) but no config sets it; and
   `multi_oracle_from_df`'s own `seed` key only seeds the weighted oracle draw,
   which is inert under `schedule: round_robin`.

   Consequences: (a) **`seed: 1` does not make a run reproducible** — identical
   sequence sampling, different folds; do not describe these as seeded
   replicates. (b) Best-of-5 takes a max over those unseeded draws, so a
   favourable draw inflates a cell. (c) The 21-point unseeded swing recorded
   under "Seeding and Run-to-Run Variance" in `CLAUDE.md` is the **sum** of
   policy and diffusion variance; nobody has separated them. Pinning protenix
   alone would make the three oracles asymmetrically noisy, which is worse for
   the comparison than leaving all three free — so average over plateau steps
   (T1.3) instead of pinning.

7. **Every cell is n = 1 run.** All arms are `seed: 1`, which is what makes them
   comparable, but a 20-point arm difference from single runs is not resolvable
   against the variance in risk 6. Budget ≥ 2 policy seeds for anything
   headline and use the three backbones as a blocking factor.

8. **`use_avg_reward` must stay off for any multi-oracle run.** GRPO's
   `advantage()` computes `(r - mean(r)) / (std(r) + 1e-3)` within the group, so
   a whole-batch oracle offset cancels in both location and scale — that is
   *why* round-robin multi-oracle training is sound at all. Switch
   `use_avg_reward` on and the offset leaks into the advantage and every
   cross-oracle comparison breaks silently. It is absent from all current
   configs and was, until this entry, written down nowhere.

9. **M0255 does not converge in 150 steps.** It projects to ~342, with protenix
   26 points below rf3 and the spread *widening* (0.000 → 0.307). It is
   simultaneously the most informative backbone for transfer and the most
   likely to be mistaken for a failed run. Decide per-experiment whether it gets
   a longer budget or is reported as a separate case — do not let it silently
   set the step budget for the others.

10. **A resumed run overwrites its own step history.** `watch_fragb32.sh`
   resubmits after a node failure, resuming from `_last.pt` into the **same
   `output_dir`**, and the training step counter restarts at 0 on a resume (see
   `submit_fragb32.sh`, where the remaining-budget argument exists for the same
   reason). So the resumed run writes `step_0000, step_0001, ...` on top of the
   original run's directories at ~10 steps/hour. Job 29184308 (resumed M0255,
   2026-10-14 00:42) had consumed steps 0-31 before this was noticed, which
   invalidated two points of T1.1's M0255 pool panel; steps 33-164 were rescued
   to `experiments/ame/m0255_original_steps/` with the step-146 attribution row
   intact. Checkpoints live at the run root and are unaffected.

   It surfaced only because a probe happened to read a directory mid-overwrite
   and reported the missing file. **Fix before the next node failure:** direct a
   resume at a `_resume<N>` output dir, or offset its step counter by the
   checkpoint's step so numbering stays global.

11. **25% of every "chimera" set is not a chimera.** `build_chimeras` assigns
   slot j of chimera (rho, i) from parent `(i + rho*j) % B`, so rotation 0 is the
   identity and the first B of the B*R folded sequences are the parents verbatim
   -- 32 of 128 at batch 32, 8 of 32 at batch 8. (An earlier note in this
   project's records claiming "0 parent folds per step" was wrong.) Measured
   across the first 8 probes, this does **not** inflate the pooled number:
   parents pass at 0.352 and true recombinants at 0.385, a difference of -0.034
   that is well inside noise. Worth reporting as its own panel -- no parent
   advantage at matched folds is close to the modularity claim the thesis needs
   -- but any per-chimera statistic must split on provenance rather than pool.


### 8.5 T1.1 result — measured 2026-10-14

Held-out AF3 against the oracle-balanced training level, at matched strides
0/30/60/90/120 on both informative backbones.

| arm | M0097 per-stride gap | mean | M0255 per-stride gap | mean |
|---|---|---|---|---|
| `fragrl4` rf3 only, b8 | +0.476 +0.288 +0.142 +0.281 +0.309 | **+0.299** | +0.531 +0.524 +0.417 +0.635 +0.274 | **+0.476** |
| `fragb32` pool, b32 | +0.082 +0.088 +0.009 +0.111 -0.021 | **+0.054** | -- -- -0.028 -0.139 -0.094 | **-0.087** |

**The pooled large-batch arm tracks AF3 across the whole trajectory, not just at
convergence.** On M0097 it climbs training 0.082 -> 0.940 while AF3 climbs
0.000 -> 0.961 beside it, never separating by more than 0.111. The single-oracle
arm never closes to within 0.14 at any stride on either backbone.

This answers the objection the 8.1 table could not: the no-gap result is not a
lucky plateau step and the large gap is not a lucky bad one. It is nine valid
stride-points per arm rather than one reading each.

**Correction to this section's own framing.** 8.3 called T1.1 "the overfitting
figure," expecting a training score climbing while held-out transfer stalls.
That is not what it shows, on either arm. The single-oracle gap is large at
*step 0*, before meaningful training, and then moves erratically with no trend
-- M0255 swings +0.53, +0.52, +0.42, +0.64, +0.27 against a 2 SE of ~0.17. The
defensible statement is **persistent miscalibration, not overfitting**: a single
oracle disagrees with AF3 by a large and erratic margin at every point in
training, and the pooled large-batch arm does not disagree at all. That is a
stronger claim for the paper, because miscalibration present from step 0 cannot
be fixed by early stopping.

**What it still cannot say.** Both columns differ in batch size *and* oracle
count, so this does not separate the two. **E15 remains the decisive, unrun
cell** -- and is now more worth running, since the effect it would attribute is
large and visible across an entire trajectory rather than at a single step.

*(Superseded in part by 8.5.1: the matched-batch arm separates the two factors
in one direction. E15 is still needed for the other.)*

Figure: `paper/figures/ame_transfer_curves.py` ->
`figures/svg/ame_transfer_curves.svg`.

**Caveats carried forward.** Each point is n=32 or n=128 chimeras from one step,
so T1.3's pooling is still required before these magnitudes are published. The
M0255 pool panel rests on 3 points, because a resume destroyed steps 0 and 30
(risk 10). The `fragrl4` M0255 volatility of +-0.37 between adjacent strides is
itself a finding: any single-step readout of a single-oracle run can land
anywhere in a 37-point band, which is how the original n=32 attribution cells
could have misled.

### 8.5.1 Completion: the full 3x3, and the matched-batch contrast

The figure is now three arms x three backbones. The third arm, `fragmulti`
(3-oracle pool, **batch 8**, seed 1), is matched to `fragrl4` on batch size and
seed and differs only in the oracle function, which is the contrast 8.5 said it
could not make.

Mean gap per cell, over strides 0/30/60/90/120:

| arm | M0097 | M0255 | M0315 |
|---|---|---|---|
| `fragrl4` rf3 only, b8 | **+0.299** | **+0.476** | +0.060 (ceiling) |
| `fragmulti` pool, b8 | **+0.060** | **+0.042** | -0.103 |
| `fragb32` pool, b32 | +0.054 | -0.057 | -0.025 (ceiling) |

**CORRECTION, same day.** The sentence that stood here -- "at identical batch
size the pool closes the gap", from M0097 +0.299 -> +0.060 and M0255 +0.476 ->
+0.042 -- claimed a one-factor contrast that does not exist. All three
`fragrl4` configs set

    checkpoint_path: .../fragrl3/run_<backbone>_frag3/..._last.pt

while every `fragmulti` and `fragb32` config sets `checkpoint_path: null`. So
`fragrl4` and `fragmulti` differ in the oracle function **and** in
initialisation: the single-oracle arm warm-starts from a previous RF3-only
generation (~450 steps), the pool arm starts from the pretrained policy. Batch
size is held at 8, but it was never the only other factor.

What made this visible: on M0315 the RF3-only arm reads **0.983 at step 0**
while the from-scratch pool arm on the same backbone reads **0.000-0.047 across
steps 1-5** -- a span covering the whole round robin, so not single-draw
jitter. An untrained policy does not score 0.983; that run had already been
trained.

**What survives.** `fragmulti` vs `fragb32` is still clean (same pool, same
seed, both from scratch, batch 8 vs 32). The *direction* of the single-oracle
result also survives, and arguably reads stronger: a policy carrying ~450 steps
of RF3-only training shows a +0.30/+0.48 held-out gap, which is what
accumulated single-oracle optimisation looks like. What cannot be claimed is
that the pool, rather than the warm start, is what separates the two arms.

**What this costs.** The matched-level analysis below is confounded the same
way -- its 24 pairs compare a warm-started arm against from-scratch arms -- so
it does not rescue the claim either. And the step-0 framing in 8.5 ("the gap is
large at step 0, before meaningful training") is wrong for `fragrl4`: its step 0
is not before training.

**E15 is now the only clean single-vs-pool contrast available.** The staged
`fragb32rf3` configs are `checkpoint_path: null`, `batch_size: 32`, `seed: 1` --
identical to `fragb32` except for the forced oracle. It was already the decisive
cell; it is now the *only* one. This also reframes the step-0 observation of 8.5: the
single-oracle arm opens a +0.48/+0.53 gap at step 0 while the pool arm on the
same backbones at the same step shows +0.059/+0.007. A step-0 gap is therefore
**not** an intrinsic property of chimeric recombination -- if recombination
itself confused structure predictors, the pool arm would show it too. It is
specific to being scored by one predictor.

**The level confound, tested rather than assumed.** A gap is easier to keep
small when both series are small, and `fragmulti` runs 3-5x lower than
`fragrl4` at the same step. Comparing gaps at matched *training level* instead
of matched step: over all 24 cross-arm point pairs within 0.08 of each other in
training level, the pool's gap is smaller in **20 of 24**, mean reduction
**+0.188**. The four exceptions are all M0315 above level 0.97, where neither
arm has headroom to show a gap at all.

**M0315 is largely uninformative and is annotated as such on the figure.** Under
`fragrl4` it starts at 0.983 training / 0.969 AF3 -- a backbone the untrained
policy already solves. Its near-zero mean gap is a ceiling artifact, not
transfer, and it should not be counted as a third replicate of the no-gap
result. The informative backbones remain M0097 and M0255.

**What this changes.** The 2026-10-13 attribution reading -- that batch size,
not the pool, closes the gap -- is **contradicted** by the matched-batch arm and
should not be carried into the paper. The pool demonstrably closes the gap at
batch 8. What remains open is the converse: whether batch 32 *alone* would also
close it, making the pool redundant at scale. That is exactly E15, which is
unchanged as the decisive unrun cell.

**Provenance.** M0255/`fragb32` steps 33 and 45 were probed from the rescued
snapshot in `experiments/ame/m0255_original_steps/` via
`af3_probe_offline.py --metrics-dir=`, because a resume overwrote the live tree
(risk 10). Fidelity checked: snapshot and live tree return identical values
(0.394 / 0.580 / 0.680) on steps 60/90/120, which the resume had not yet
reached.

### 8.6 Order of work

1. ~~**T1.1**~~ **done** (8.5, 8.5.1). **T1.2–T1.3** next — offline and cheap.
2. **E15 on M0255**, then M0097, as soon as GPUs are free.
3. **T2**, the 256-parent library eval — independent of E15 and overdue.
4. **E16/E17** only after E15 reports, for the reason in §8.3.
