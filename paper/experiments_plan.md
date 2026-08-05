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

That is now possible: the baseline sequences are recoverable from
`ligmpnn/backbones/` in `enzyme_bench_n41.tar` (32,801 files = 4,100 x 8). So:

- fold baseline sequences and CLEO sequences with the **same** predictor (AF3)
- score both with the **same** module (`rfd2_benchmark.py`)
- use their published pass/fail labels as an independent cross-check and to
  define the rescue partition, never as the basis of the head-to-head

Their Chai-1 predicted structures are not in the deposit (only `backbones/` and
`packed/` under `ligmpnn/`), so calibrating the sequence-dependent motif RMSD
against their exact values is not possible. Recomputing both arms is not a
workaround for that — it is the correct design either way.

### Remaining pre-pilot checks

- [ ] Motif RMSD is stable and sane on a handful of AF3 predictions (correct
      magnitude, correlates with pass/fail on backbones whose published outcome
      we know)
- [ ] Rerun `calibrate_metrics.py` if the design-PDB provenance is ever
      resolved; document the residual discrepancy in Methods if not
- [ ] Confirm CLEO's `ligand_mpnn` matches their motif-rotamer-aware + packing
      mode
- [ ] One backbone end-to-end through AF3 -> metrics -> reward scalar

---

## 4. Staging

| Phase | Work | Purpose |
|---|---|---|
| **P0** | Extract and freeze filter thresholds from benchmark source; verify CLEO's LigandMPNN mode matches theirs; reproduce A0 on 3–5 sites | Prove we can hit their published numbers before claiming to beat them |
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
