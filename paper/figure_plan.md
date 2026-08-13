# Figure plan — computational paper

Working backwards from the panels that have to convince a reviewer. Each panel
lists its claim, axes, arms, the data it needs, and **what would make it
unconvincing** — the failure mode is the useful part.

Companion to `experiments_plan.md` (protocol) and `outline_computational.md`
(framing).

Two decisions this plan encodes:

- **Figure 1 costs zero compute.** It is re-analysis of the RFdiffusion2 Zenodo
  deposit. It should be built first, this week, because if it is not striking
  the whole framing needs to change before we spend GPU-hours.
- **Diversity is always measured on passing designs only.** Stated in every
  caption. Any method can win raw diversity by emitting garbage.

---

## Figure 1 — The sequence-search gap  *(free: their data, our analysis)*

**Claim:** de novo enzyme design has a sequence-search bottleneck that
site-level success metrics hide.

Framing discipline: RFdiffusion2's claim is that it scaffolds all 41 sites, and
that claim is true and stands. Our point is orthogonal and must be written that
way — at the level of the individual design, most backbones never receive a
usable sequence.

| Panel | Content | Claim |
|---|---|---|
| **1A** | Per-site sequence pass rate, 41 sites sorted, RFd2 vs RFd1 | Enormous spread: 0.02 % → 57 %. Most sites are very hard |
| **1B** | Histogram of passing sequences per backbone (0–40) | **The motivating panel.** A spike at zero holding 76.5 % of backbones |
| **1C** | Success collapse across levels: site 41/41 → backbone 23.5 % → sequence 5.8 % | The gap the paper occupies, in one figure |
| **1D** | Motif size (1–7 residues) vs. backbone success rate | **Difficulty is predicted by active-site complexity**, r = −0.81 |

**1C is the thesis panel.** Three bars, or a funnel. It reframes a solved-looking
problem as an open one without contradicting anyone.

**1D justifies the whole project.** Measured from their deposit:

| Motif residues | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Sites | 1 | 5 | 9 | 11 | 4 | 7 | 4 |
| Mean backbone success | 94 % | 61 % | 31 % | 20 % | 6 % | 4 % | 4 % |

Correlation of motif size with success: **r = −0.805** (atom count: −0.784). The
sequence-search problem gets monotonically harder exactly as active sites get
more chemically interesting. One-residue motifs essentially always work and are
not where enzyme design is hard; the 7-residue sites — real multi-residue
catalytic constellations — sit at ~4 %.

This is why the pilot forces in a 7-residue site. Rescuing a 2-residue motif is
a curiosity; rescuing a 7-residue Zn/GSH site is the result that matters.

**Unconvincing if:** the zero-spike in 1B turns out to be concentrated in a few
pathological sites rather than spread across the benchmark. Check this before
committing — if 76.5 % is really "6 sites are impossible," the story is much
weaker and Figure 1 has to be rebuilt around the sites that matter.

---

## Figure 2 — More diverse libraries at equal or better fidelity  *(PASS backbones)*

**Claim:** on backbones where LigandMPNN already succeeds, CLEO produces
substantially more distinct passing solutions for the same folding budget.

| Panel | X | Y | Claim |
|---|---|---|---|
| **2A** | Diversity of passing set (mean pairwise distance, or distinct clusters) | Pass rate | CLEO lies outside the LigandMPNN temperature frontier |
| **2B** | Cumulative folds | Cumulative **unique passing mutations** | LigandMPNN saturates; CLEO keeps climbing |
| **2C** | Cumulative folds | Distinct passing clusters @ 90 % id | Same story in units of "independent solutions" |
| **2D** | Sequence position | Per-position AA entropy, passing designs | Where the diversity lives; MPNN concentrated, CLEO spread |

**2B is the panel this project has been building toward.** The prediction worth
stating explicitly: low-temperature LigandMPNN has an **asymptote**. Past some
fold count it re-emits sequences it has effectively already produced, so the
curve flattens and further compute buys nothing. A visible plateau in the
incumbent method, next to a still-rising CLEO curve, is "filtering is not
search" rendered as a single line.

Report the saturation fold count as a number in the text.

**Arms (all panels):** LigandMPNN at their published settings; T ∈ {0.1, 0.2,
0.3, 0.5, 0.7, 1.0}; best-of-N rejection; filtered-SFT; CLEO-GRPO.

**Design change (supersedes the earlier plan): the diversity-reward arm is
dropped, and the library is the training trajectory.** Rather than adding an
explicit diversity term to the reward — a hyperparameter that has to be tuned
and then defended — every rollout sampled during training is a library
candidate. Those sequences were already folded to compute the reward, so the
pool costs nothing extra and the library is a byproduct of training. Measured
on M0097: 3200 scored designs, 97 passing, 97 distinct clusters at 90 % id,
33 % mean identity, still climbing at the end of training. This removes a
knob and makes the comparison to a temperature sweep exactly parallel — both
arms are "sample a lot, then select."

**Measured negative result, recorded so it is not re-derived.** Selecting
*which* sequences to fold by greedy max-min (farthest-point) diversity is
**worse than random** at every budget tested on the M0097 pool:

| Budget | as-sampled | random | max-min |
|---|---|---|---|
| 800 | 51 passing | 25 | 6 |
| 1600 | 85 passing | 46 | 10 |

Max-min preferentially selects outliers, and outliers are exactly the designs
least likely to pass; gating candidates to a high-quality band narrows but does
not close the gap. Diversity-first acquisition is therefore not a viable way to
spend fold budget. Its defensible use is post-hoc selection of an orderable
subset from designs *already known to pass* — implemented in
`experiments/ame/select_library.py`, which also supports an optional
anchor term that maximises distance from the low-temperature consensus.

Note this also means the diversity is essentially *free*: among passing designs
the cluster count already equals the passing count, so there is nothing for a
selection step to improve. Hits are the expensive quantity; diversity is not.

### Panel 2E — PCA of occupied sequence space *(free; script written)*

`paper/figures/ame_pca.py` writes two figures. `ame_pca.svg` projects all arms
into one shared PCA with CLEO restricted to its peak window, so three
*policies* are compared at one moment in training: the low-T arm is a tight
displaced knot, CLEO a broad cloud with passing designs throughout.
`ame_pca_drift.svg` is the honesty control — **PC1 of the pooled trajectory
correlates with training step at r = +0.92…0.94**, so most of that axis is
drift over training, not diversity available at any one step. Drift-derived
spread is real library diversity under the trajectory-as-library framing but is
*not* evidence a single policy is diverse; conflating the two would overstate
the result and is the kind of thing a referee finds. Explained variance is 5–9 %,
so the projection is a visual aid — every quantitative claim uses Hamming
distance and cluster counts.

**Unconvincing if:** the temperature sweep reaches the same diversity at the
same pass rate. Then there is no frontier shift, only a reparameterisation, and
the honest conclusion is that temperature was enough all along. **2A is
therefore the panel most likely to kill the paper, which is exactly why it must
be run first among the compute-spending panels.**

---

## Figure 3 — Rescue  *(FAIL backbones)* ★ headline

**Claim:** backbones discarded after 40 failed sequences are frequently not bad
backbones — they are under-searched, and RL search recovers them where more
sampling does not.

| Panel | Content | Claim |
|---|---|---|
| **3A** | Rescue rate vs. additional folds, per arm | CLEO rises; the compute-matched sampling control stays flat |
| **3B** | Per-backbone best motif RMSD: baseline (40 seqs) vs. CLEO, paired, with pass threshold drawn | Individual designs crossing the line |
| **3C** | Rescue rate vs. site difficulty (their published per-site pass rate) | Does it work where it is hard, or only where it was nearly working? |
| **3D** | For rescued backbones: how many *distinct* passing sequences | Rescue yields a library, not a lucky single hit |

**3A's control is the whole experiment.** R1 (more LigandMPNN samples,
40 → 100 → 400) must be plotted on the same axes. Without it this reads as "we
spent more compute." With it, the claim becomes *at equal additional compute, RL
search rescues what sampling cannot* — and their own data already shows 40
samples was not enough.

**3D matters more than it looks.** A rescue producing exactly one passing
sequence is a curiosity; one producing a diverse passing set is a library, and
ties Figure 3 back to the paper's thesis instead of leaving it as a separate
trick.

**3C is the honesty panel.** If rescue only works on backbones that were nearly
passing anyway, say so plainly — it is still a useful result, just a narrower
one. If it works at the hard end (M0024_1nzy, 0.02 % pass) that is a much
stronger claim and deserves its own callout.

**Unconvincing if:** rescue rate is low single digits. Test this early on a
small subset before committing Figure 3 as the headline.

---

## Figure 4 — Efficiency and controls

| Panel | Content | Claim |
|---|---|---|
| **4A** | Cumulative distinct passing designs vs. cumulative folds | **The crossover.** GRPO pays upfront, wins later — where is break-even? |
| **4B** | Folds per distinct passing design, by arm | The practical headline number |
| **4C** | Held-out oracle: pass rate under Chai when reward used Boltz | Not reward hacking |
| **4D** | Policy transfer: per-backbone vs. per-site vs. generalist on held-out sites | Is the upfront cost amortizable? |

**4A determines which paper this is.** Crossover below ~50 folds/backbone → CLEO
is a drop-in replacement for LigandMPNN. Crossover in the thousands → it is a
rescue tool for hard targets. Both are publishable; they are different papers.

**4C is non-negotiable.** See the oracle decision below.

---

## The oracle decision (affects every panel)

Their success criterion is `chai_motif_pass_and_no_clash`, computed with
Chai-1. If we both train and evaluate on Chai, every number is circular.

**Recommended:** in-loop reward = **Boltz** (already supported in
`cleo.design.utils.oracle.boltz_from_df`); evaluation = **Chai-1**, their
container, their criterion, never in the reward. The headline then reads: *we
rescue X % of their discarded designs under their exact published criterion,
using an oracle the policy never saw.* That is very hard to attack.

**Risk:** if Boltz and Chai disagree, the reward is a poor proxy and performance
suffers for reasons unrelated to the method. Measure the Boltz↔Chai agreement on
the existing 164,000 labelled sequences **before** launching training — that is
a cheap, decisive pre-flight check and it costs only Boltz inference on
sequences whose Chai labels we already have.

**Fallback:** if agreement is poor, run Chai in-loop and hold out AF3 + PLACER,
and report the Boltz-in-loop arm as the conservative variant. Label the circular
arm as circular; do not bury it.

---

## What this implies for target selection

Panels drive the pilot set:

- **Figure 2** needs backbones with ≥ 1 passing sequence → sample from the 964
- **Figure 3** needs 0-pass backbones → sample from the 3,136
- **3C** needs difficulty spread → sites across the published pass-rate range
- **2A/2B** need enough backbones per site for curves to be smooth, not enough
  to bankrupt us

**Pilot: 4 sites × (5 PASS + 5 FAIL) = 40 backbones.** Three chosen to span
difficulty, plus one forced complex site.

| Site | Motif | Ligands | Backbone success | Role |
|---|---|---|---|---|
| M0058_1cju | 5 res / 14 atoms | MG, DAD | 5 % | hard |
| M0255_1mg5 | 4 res / 10 atoms | ACT, NAI | 23 % | mid |
| M0664_2dhn | 2 res / 5 atoms | PH2 | 84 % | easy |
| **M0157_1qh5** | **7 res / 22 atoms** | **GSH, ZN** | 9 % | **complex active site** |

M0157_1qh5 is the only 7-residue site with enough passing backbones (9) to serve
both the diversity and the rescue panels — the other three have 1, 3 and 4. Its
motif is four His plus two Asp around a Zn/glutathione site, i.e. a genuine
multi-residue catalytic constellation rather than a two-point anchor.

**Compute is denominated in folds** (structure-prediction calls), not wall time
or GPU-hours, in every panel. Folds transfer across hardware and schedulers,
they are what the baseline's published budget is expressed in (40 per backbone),
and they are the quantity a lab actually plans around. GPU-hours go in the
supplement for completeness only.

Scale-up after the pilot survives 2A and 3A.

---

## Build order

1. **Figure 1** — now, zero compute. If 1B/1C are not striking, re-frame before spending anything.
2. **Boltz↔Chai agreement pre-flight** — cheap, decides the oracle question.
3. **Figure 3A on a small subset** — is rescue real? This is the headline; find out early.
4. **Figure 2A** — is there a frontier shift, or does temperature suffice? Most likely to kill the paper.
5. Everything else.

Deliberately front-loading the two panels most likely to invalidate the project.
