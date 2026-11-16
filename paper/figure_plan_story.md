# Figure plan, keyed to the story draft

Written 2026-11-16, against the first full prose draft. The draft makes ~26
distinct figure references, most of them written as `figure 1?` or `fig 2?`
because the panel did not exist to point at. This file resolves every one:
what the panel shows, whether the data exists, and where it comes from.

`paper/figure_plan.md` is the older, computational-only plan under a different
numbering. Where the two disagree, this file wins.

---

## 0. Four figures, as the draft has them

The draft's own numbering stands:

| | carries |
|---|---|
| **Fig 1** | why combinatorial, the method, and the AME benchmark |
| **Fig 2** | PETase: the design, and what post-training did to it |
| **Fig 3** | the experimental platform and the three libraries |
| **Fig 4** | searching a 10^12 library in four rounds |

They are built into `figures/svg/main/`, separate from the panel sources in
`figures/svg/` that get dropped into their slots.

Figure 1 is doing the most work — eight panels, and at 6.5 in wide it is
9.9 in tall. That is the cost of the four-figure format and the thing to
watch when it is placed. If it has to come down, panel (d) (cost vs library
size) is the one whose argument the text can carry on its own.

A note on panel letters: the draft writes `figure 1?` throughout because
there was nothing to point at. Figure 1's letters are now fixed, so the
prose can be pointed at them:

    (a) three landscapes (2D)   (f) diversity, per backbone
    (b) fragment shuffling      (g) motif RMSD, all backbones
    (c) the GRPO objective      (h) oracle calls vs library size
    (d) DNA cost at several k   (i) where in sequence space
    (e) pass rate, per backbone (j) how fine it can be cut

**Every arm comparison in Figure 1 is two arms: stock ProteinMPNN against
"This work".** `plain RL` -- GRPO on whole sequences, RF3 only -- is real and
is kept loadable (`ame_arms.ARMS_ALL`), but it answers a different question
(does fragment-level credit beat sequence-level credit) and a third bar in
every panel invites a three-way comparison the figure is not making.

**Panel (e) is per-backbone on purpose; (h) is pooled.** (e) is a scatter —
diversity on x, pass rate on y, one point per backbone per arm, with an
arrow for the move from stock ProteinMPNN to post-trained. It is not
averaged, because the move is not the same move on every backbone and a mean
of the three asserts something none of them does (see §7).

(h) shows ONE curve, the policy trained at each k, with a min-max band over
the three backbones. The post-hoc control — a single k=4 policy re-cut at
every k — is about train/test mismatch rather than about this claim, and
stays in `figS_posthoc`:

| k | 2 | 4 | 8 | 16 | 36 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| trained at k | 0.91 | 0.90 | 0.94 | **0.92** | 0.60 | 0.29 |

Flat through k=16, a cliff at 36.

**Keep the per-backbone points on the pooled panels.** M0255 is materially
harder than the other two (82.4% against 96.9 and 100.0 on recombinant pass;
0.406 against 0.559 and 0.832 at k=36) and it is what drives the tail. A
bare mean hides that, and M0255 at k=36/64 was still climbing at step 200,
so the pooled tail is a lower bound either way.

## 1. Every figure reference in the draft, resolved

Status key: **have** = data in hand, panel is a build job. **derive** = data
exists but needs a new reduction or a new script. **collect** = the
measurement or the run does not exist yet.

### Figure 1 — the method, and the benchmark that says it works

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 1a | **built, 3-peak mesh.** "figure 1a on the left / middle / right" | three landscapes: error-prone PCR, naive combinatorial, filtered combinatorial | **have** | `concept_library.py:draw_landscape` — toy, no data. **Redraw as 3D (see §3).** |
| 1b | **built.** "sample Nsampled new sequences (figure 1?)" | the shuffling scheme: Nbatch=16 decoded, cut into k, Nsampled=64 chimeras, every fragment seen 4x | **have** | schematic; `concept_loop.py:panel_loop` is the credit-assignment half of it |
| 1c | **built.** "INSERT GRPO OBJECTIVE HERE" | the per-fragment GRPO objective, typeset, variables labelled against 1b | **have** | equation, not a plot — belongs inside the figure so its symbols sit next to the scheme they describe |
| 1d | **built.** "on the order of 10^4 oracle calls ... can easily reach 10^12" | oracle calls spent vs library size reached, and per-variant cost | **have** | `panel_cost.py`, `concept_library.py:gene_cost/library_cost` |


| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 1e | "increase success from these backbones for sequences sampled from the library" | recombinant pass rate, **pooled over the three backbones** with the points shown | **built** | `ame_arms.py`. stock MPNN **45.1**, plain RL **53.5**, chimera+pool **90.1** |
| 1f | "increase diversity ... improving the novel of unique mutations" | rarefied distinct substitutions over *passing* members, pooled | **built — and it does not say what the draft says**, see §7 | `ame_arms.py`. 900.0 / 1342.4 / 990.6 |
| 1g | **built, M0097.** "low dimensional representation of random mutagenesis scoped libraries, naive proteinMPNN, and the post-trained model" | three arms in one embedding | **derive** | archived `ame_pca.py`. **Third arm needs building** — a random-mutagenesis library at matched size; see §2 |
| 1h | **built.** "libraries out to k=16 ... still yield strong highfidelity libraries" | retention vs k, trained-at-k vs cut-at-k, **pooled over the three backbones** | **have** | `ame_fragk.csv` — supports a *stronger* claim than the draft makes; see §4 |

### Figure 2 — the PETase, and what post-training did to it

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 2a | "designed from a theozyme alone of PET polymer docked into the active site (fig 2a)" | theozyme → RFD2 backbone → MPNN → filters → momi; the design funnel | **have** | `structures/png/momi_fold.png`, `momi_site.png`; filter thresholds are in the draft prose |
| 2b | "optimizing for these rewards drastically reduces variance and improves positioning and confidence of active site residues (Fig 2?)" | per-reward distributions, naive vs post-trained library | **derive** | needs the PETase run's own metrics; AME equivalent is `panel_vanilla.py` |
| 2c | "Visualizations of the active site atoms from naive and optimized libraries (fig2?)" | superposed catalytic atoms, two clouds | **derive** | `structures/render_momi.py` exists; needs folds for both arms |
| 2d | "ability to center the sequence space of the library to be tethered to a top-k set (figure 2?)" | library 3's sequence space centred on the top-3 measured variants vs library 2's | **have** | part names carry design intent (`dist0.topk1`, `distN.*`); compute real Hamming — **the `distN` tag disagrees with computed distance on 992/1803 parts** |

### Figure 3 — the platform and the three libraries

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 3a | "as seen depicted in figure 3a" | wet-lab pipeline strip: geneblocks → ECHO → GGA → PCR → PUREfrex → reader | **have, built** | `svg/wetlab_pipeline.svg` (Jason, `312211c`) |
| 3b | "gate samples on each replicate having at least 0.1uM ... (figure 3b)" + "background subtracted and fit with least squares (figure 3b)" | the filter cascade with counts surviving each gate | **have** | `libraries_snapshot.py` `passes_*` columns |
| 3c | "consistency amongst measured control rates seen in figure 3c" | positive-control rate per plate, per round, all three libraries | **have** | snapshot controls. Plates agree to 1–2% within a round (20.70/20.18/20.05) — **plot raw rate, not momi-normalised**, which is 19.7 by construction in most rounds |
| 3d | "parity plots of individual replicates seen in figure 3d" | replicate i vs replicate j | **collect** | the snapshot is already reduced to means; `cv_rate` survives, the three replicate values do not. Re-extract from jq01's per-round raw files |
| 3e | — (draft skips e) | one worked kinetic trace: raw fluorescence, background, least-squares fit | **have** | `library1_kinetics.py` reads the raw reader files already |
| 3f | "aggregate measurements from each of three different libraries ... figure 3f?" + "nearly 400x ... figure 3x" | all three libraries on one momi axis, by round | **have, built** | `petase_libraries_filtered.csv`; medians 1.15 / 3.69 / 20.27, max 17.1 / 66.0 / 395.0 |

### Figure 4 — searching 10^12 in four rounds

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 4a | "combine each designed fragment with the other parent fragments ... figure 4a" | single-body effect of every part, by slot | **have** | round 0 is *exactly* this — 1665 (lib 2) and 1791 (lib 3) designs, every one with 4 of 5 slots = G20 |
| 4b | "correlation between distance from top-k fragment and activity ... figure 4a" | activity vs Hamming distance of the one varied part | **have** | same rows; compute distance, do not trust `distN` |
| 4c | "binning fragments ... ~128 variants per bin ... figure 4b" | round 1 activity by bin, with the top-2 exhaustive block called out | **have** | round 1 tags `bin1`..`bin6` (19–165 each) + `top2combos` (31 of the 32 possible) |
| 4d | "using the predictors ... next 2 rounds ... figure 4x" | rounds 2–4, predicted vs measured, by selection strategy | **have** | tags `MLPgamma1/5/8`, `topkMLP`, `sampledtopkMLP`, `top30mu`, `top10mu`, `mostdiverseMuGt3/4` |
| 4e | "in many fewer evaluations (put real savings here!) than greedy would require" | policy-over-fragment-space vs exhaustive/greedy screening cost | **derive** | counts are computable from library size and rounds; the greedy comparator needs defining |
| 4f | "comparing our top designs to native PETases ... at XXx faster Kcat figure 4x" | designs vs FastPETase / ThermoPETase / LCC / Kubu / MiPa, one plate | **have** | **`jq_e0328`, and it is better than we thought — see §5** |
| 4g | "we do not find that optimization of this probe substrate transfers well to bulk PET" | probe activity vs nanoPET / pellet activity | **collect** | `jq_e0334/3_sm_hplc_nanopet_titration` and `jq_e0331/0333` HPLC exist as notebooks only. Needs extraction, or ask Jason |
| 4h | "many variants distant in sequence space that are significantly more accurate" | activity vs distance from momi, final libraries | **have** | snapshot. **Note the first library said the opposite** — activity fell with distance (`key_results.md` §7) |

### Supplementary referenced in the draft

| supp | draft text | status | note |
|---|---|---|---|
| S1 | "splitting up into fragments is actually helpful" | **collect** | the sequence-level vs fragment-level credit ablation has not been run |
| S2 | "a pool of structure prediction oracles improves ... pass rate on heldout oracle" | **have, two ways** | computational: `key_results.md` §4, pool −0.005 vs RF3-alone +0.041 on the one matched pair. **Experimental: library 2 round 2 has `af3Only` / `boltzOnly` / `modelhubOnly` / `allOracles` arms with measured activity** — 336 designs. Stronger panel than the computational one |
| S3 | "do not benefit from protein language model embeddings, and simply using one-hot ... (see supplemental figure)" | **collect** | no PLM-vs-one-hot comparison exists. `petase_model_cv_scores.csv` is AF3-vs-Rosetta features, a different question. Runnable from `cleo.optimize` + the snapshot |
| S4 | shuffling during training vs not | **collect** | the 18-run clean comparison is bundled at `/net/scratch/jgershon/cleanab_bundle` and **has never been submitted** (`MaxSubmitJobs=0`) |
| S5 | oracle metrics vs measured activity on the real libraries | **have** | `jq_e0330`, 6882 rows with `af3_his_ser_angle`, `af3_ligand_rmsd`, `af3_ptm_*`, `af3_iptm` *and* activity, keyed by library and round |

---

## 2. What has to be collected, in priority order

1. **Replicate-level values for the parity plot (3d).** Everything else about
   Figure 4 is in hand; this is the one panel with no path from the snapshot.
   The raw readers are reachable the way `library1_kinetics.py` reaches them.
2. **Bulk PET / nanoPET (4g).** The draft states a negative result — that
   probe gains do not transfer. A stated negative needs its panel. The
   notebooks exist; the extracted numbers do not.
3. **The random-mutagenesis arm for 1g.** `jq_e0307_250311_petase_library_random_mutation`
   has AF3 output for a random-mutagenesis PETase library, which may serve
   directly; otherwise it is a cheap generate-and-fold.
4. **PLM vs one-hot (S3).** Runnable today against the snapshot. Half a day.
5. **The fragment-credit ablation (S1).** A training run, and the draft leans
   on it for the central methodological choice.
6. **The shuffling comparison (S4).** Bundle is ready. Blocked on queue
   access, not on us.

---

## 3. The landscapes go 3D

The conceptual landscapes in Fig 1(a) are currently **2D filled contours** seen from
above (`contourf` + white contour lines). The draft wants 3D.

**The trap:** matplotlib's `mplot3d` is the obvious route and the wrong one.
It emits one path per surface quad — a 60x60 surface is ~3,600 filled polygons
with no axis objects — and its text is positioned by projection, so labels
shift when anything is nudged. That breaks the one rule in `docs/FIGURES.md`:
the SVG has to stay editable in Inkscape.

**The route that keeps both:** a hand-rolled isometric projection drawn as a
ridgeline — ~40 constant-y profiles, each one filled polyline, painted back to
front. It reads as a 3D surface, it is ~40 objects instead of 3,600, every
label is a real `<text>` placed in axis coordinates, and the peaks can be
reordered or recoloured by hand afterwards. Designs are projected through the
same transform so they sit on the surface.

That is `landscape3d.py`. The 2D version stays in `concept_library.py` so the
choice is reversible by swapping one import.

---

## 4. Two places the data supports a stronger claim than the draft makes

- **"libraries out to k=16" is conservative.** `ame_fragk.csv` has retention at
  k=16 of 0.910 / 0.871 / 0.984 across the three backbones — essentially flat
  from k=2. The cliff is at k=36 (0.559 / 0.406 / 0.832) and k=64. k=16 on a
  ~210-aa protein is 13-residue fragments. Say "flat through 16 and falls by
  36", which is what was measured, rather than implying 16 is the edge.
- **"~19x for MomiG20" is the derived number, and there is a measured one.**
  See §5.

## 5. The 400x claim can be made inside one plate

`jq_e0328_260202_..._top_design_vs_native/2_overnight_37c_kinetics_analysis`
measures the top designs, the native PETases, G20 **and momi WT** on one plate,
one condition, one normalisation:

| construct | x momi WT (this plate) |
|---|---:|
| best design (`...4.top14`) | **150.4** |
| 10th best design | ~122 |
| kubu_pm12 | 56.7 |
| mipa_pm19 | 48.6 |
| kubu_wt | 44.7 |
| ThermoPETase | 44.5 |
| FAST_PETase | 44.3 |
| LCC | 19.8 |
| mipa_wt | 15.5 |
| momi G20 | 12.4 |
| momi WT | 0.383 |

So, measured within one plate and needing **no cross-experiment factor at all**:

- best design / momi WT = **393x**  (the draft's "~400x", and now defensible)
- best design / FAST_PETase = **3.4x**
- best design / ThermoPETase = **3.4x**
- best design / LCC = **7.6x**  (fills the draft's "XXx")

**Three caveats that have to travel with it.** (i) It is an overnight 37 C
endpoint, n = 1 per construct — a ranking, not a kinetic constant. (ii) The
denominator is weak: momi WT at 0.383 sits close to background, which is the
whole reason `MOMI_NORMALISATION.md` exists. (iii) This plate's *internal*
G20/momi ratio is **34.0x**, not the 19.7x adopted from the two
condition-matched measurements. The two numbers answer different questions and
must not be mixed: use 19.7 for the cross-library axis in Fig 3(f), and this plate's
own ratio for anything quoted off this plate. That both routes land near 394x
for the best design is a coincidence of this dataset, not corroboration.

---

## 6. Numbers the draft leaves blank

| draft | fill with | where |
|---|---|---|
| "upwards of XX% of the library passing design filters" | 96.9 / 82.4 / 100.0 recombinant pass, three AME backbones | `key_results.md` §6 |
| "increase Kcat by X-fold" | 393x over momi WT, in-plate | §5 above |
| "surpassed these native designs at XXx faster Kcat" | 3.4x FastPETase, 7.6x LCC | §5 above |
| "~19x (check that number)" for G20 | 19.7x derived, 34.0x in the one plate measuring both | `docs/MOMI_NORMALISATION.md` |
| "~66x (check these)" library 2 | 66.0x — correct | snapshot |
| "~400x" library 3 | 395.0x on the derived axis, 393x in-plate | snapshot, §5 |
| "a total of 25 = 32 variants" | 2^5 = 32; 31 of the 32 were assayed | round 1 `top2combos` |
| "(put real savings here!)" | needs the greedy comparator defined | section 1, 4e |
| "10^4 oracle calls" | 200 steps x 32 x best-of-5 = 3.2x10^4 | `panel_cost.py` |

---

## 7. Figure 1(f) contradicts the draft, and the draft should change

The draft says post-training "increase[s] success from these backbones ...
and increase[s] diversity as well". The first half holds everywhere. The
second does not, and the panel now shows it per backbone rather than pooled.

Rarefied distinct substitutions over passing recombinants, each arm
subsampled to a common survivor count on each backbone:

| backbone | stock ProteinMPNN | This work | pass rate |
|---|---:|---:|---|
| M0097 | 853 | **1199** | 60 -> 86 |
| M0255 | 608 | **700** | 26 -> 86 |
| M0315 | **1238** | 1045 | 50 -> 98 |

**Pass rate gains on all three; diversity gains on two and falls on the
third.** M0315 trades 193 distinct substitutions for +48 points of fold
rate. The defensible sentence is: fine-tuning for recombination raises the
fraction of the library that folds on every backbone, while holding
diversity at or above stock ProteinMPNN on 2 of 3. The stronger one is not
supported.

### Which fine-tuned arm the numbers come from

`key_results.md` §6 quotes 96.9 / 82.4 / 100.0 for the chimera objective.
That is **`fragb32`**, evaluated as a 256-parent library (T2). Figure 1 uses
**`fragk4`** -- 86 / 86 / 98 -- because it is scored through the same
`recomb` pipeline as the stock arm it is compared against. Different arms;
do not mix them in one sentence.

---

## 8. Catalytic-atom spread: the measured version of the PETase claim

The PETase section claims in prose that post-training "drastically reduces
variance and improves positioning ... of active site residues". On AME that
is measurable, and it is Figure 1(g).

`rfd2_motif_rmsd` is the predicted motif's RMSD to the theozyme it was
scaffolded around. Figure 1(g) shows it as a strip, **all three backbones
pooled**, 768 designs per arm, one point per design averaged over its five
diffusion samples:

| | median | IQR | within 1.5 A | worst |
|---|---:|---:|---:|---:|
| stock ProteinMPNN | 1.84 | 1.48 - 2.24 | 26% | 53.4 |
| **this work** | **1.28** | **0.99 - 1.42** | **84%** | 12.1 |

Fine-tuning lowers motif RMSD and tightens it: the median falls 0.56 A, the
IQR narrows from 0.76 to 0.43, and the share of the library placing its
catalytic atoms within the benchmark's own 1.5 A criterion goes 26% to 84%.

Per backbone (`motif_variance.py` prints it), over EVERY design rather than
only passing ones -- conditioning on passing truncates at the threshold and
would manufacture the very reduction being claimed:

| backbone | arm | median | sd between designs | IQR | frac <= 1.5 A |
|---|---|---:|---:|---:|---:|
| M0097 | stock | 1.59 | 1.59 | 0.62 | 0.42 |
| M0097 | **this work** | 1.32 | **0.77** | **0.19** | **0.82** |
| M0255 | stock | 2.10 | 1.19 | 0.90 | 0.15 |
| M0255 | **this work** | 1.39 | **0.50** | **0.24** | **0.73** |
| M0315 | stock | 1.83 | 5.04 | 0.69 | 0.23 |
| M0315 | **this work** | 0.92 | **0.38** | **0.10** | **0.97** |

Between-design spread falls **2.1x / 2.4x / 13.4x**.

**Two things to state with it.**

0. **The pooled strip mixes three backbones of different difficulty.**
   M0255 is the hard one and M0315 the easy one, so part of the pooled
   cloud's width is that mixture rather than within-library spread. The
   simple panel is the right one for the claim; the per-backbone table above
   is what to quote if the width is questioned.
1. **The gain is between designs, not within one.** The within-design
   spread -- across the five diffusion samples of a single sequence -- goes
   0.33 to 0.42 on M0097, i.e. slightly UP. Fine-tuning tightens the
   *library*, not the oracle's reproducibility on a given sequence. That is
   the right claim and the stronger-sounding one would be wrong.
2. **M0315's stock sd of 5.04 is a tail, not a typical design.** Its worst
   stock design sits at 53 A against a 1.83 A median. Quote the IQR (0.69 ->
   0.10) as the robust statement and let the log axis carry the tail.

**This is not a per-atom RMSF**, and cannot be made into one from what is on
disk. A per-atom figure needs the predicted structures, and
`r1panel/recomb/folds/` is empty -- the stock arm's CIFs were deleted and no
tarball survives, so every `af3_path` in that arm points at a missing file.
Recovering it means re-folding 256 designs x best-of-5 x 3 backbones =
**3,840 AF3 predictions**. Worth doing if a per-atom panel is wanted; the
scalar spread above is what today's data supports.

---

## 9. The PETase anchors: two sets, and which is which

**The anchors.** `config/design/denovo_petase.yaml` lists three sequences
under `dist_to_ref_seqs`, and the reward minimises distance to the nearest
of them at **weight 10** -- the heaviest term in the aggregation. Those
three are the design target. They are all exact library-2 members, but they
are **not** the top 3 by measured activity:

| anchor | rank in library 2 | activity | round |
|---|---:|---:|---:|
| ref1 | 1 of 3,371 | 3.35 x G20 | 2 |
| ref2 | 4 of 3,371 | 2.72 x G20 | 2 |
| ref3 | 21 of 3,371 | 2.31 x G20 | 3 |

They sit 4-7 mutations apart. `petase_design.anchors()` reads them from the
config and `anchor_table()` prints this table, so a top-k-by-activity guess
cannot quietly replace ref3 with rank 2.

**The menu is uniform by design.** Library 3's part names carry a `distN`
tag: N mutations from the corresponding anchor part. Counting DISTINCT
parts per slot:

| slot | parts | reach | shape |
|---|---:|---:|---|
| 1 | 360 | 1-14 | 26-27 at every distance |
| 2 | 363 | 1-9 | 38-80, peaked at 4 |
| 3 | 360 | 1-8 | 40-48, flat |
| 4 | 354 | 1-8 | 39-50, flat |
| 5 | 366 | 1-9 | 42-71, peaked at 2 |

Figure 2(c) plots this. Parts and counts, not designs and density: a part
built into 200 assembled designs is still one part in the menu, and
counting designs reweights the menu by how often the sampler drew each
part. The flat shape is the design; what differs between slots is the
REACH, and slot 1 reaches nearly twice as far as slots 3 and 4.

**RETRACTED: the slot boundaries are fine.** This section previously said
slots 1 and 5 in `SLOTS` were mis-cut, on the evidence that recomputed
per-slot distance correlated with the `distN` tag at r = 0.96 and 0.99 for
slots 2 and 3 but only 0.09 and 0.41 for slots 1 and 5. That inference was
wrong and `fig3_momi.py` needs no fix.

Checked directly: for each slot, the positions whose residue is *determined*
by that slot's part name (constant within every part group, varying between
groups) land exactly inside the stated range, for all five slots.

| slot | `SLOTS` | positions determined by the part name |
|---|---|---|
| 1 | 1-40 | 1-40 |
| 2 | 45-84 | 45-84 |
| 3 | 89-126 | 89-126 |
| 4 | 131-163 | 131-163 |
| 5 | 168-209 | 168-209 |

**What was actually wrong is the reference set.** Library 3's parts were
generated around **five** references, not three. A part named
`i.dist0.topkN` is reference N's own fragment at slot i, and reassembling
those gives all five:

| | rank in library 2 | activity | distance to the config's three |
|---|---:|---:|---|
| topk1 | 1 of 3,371 | 3.35 x G20 | 0, 7, 4 |
| topk2 | 4 | 2.72 | 7, 0, 6 |
| topk3 | 21 | 2.31 | 4, 6, 0 |
| **topk4** | 27 | 2.24 | **22, 22, 18** |
| **topk5** | 32 | 2.21 | **16, 15, 14** |

The config's three are exactly topk1-3. topk4 and topk5 are 14-22 mutations
from all three, so they are separate corners, and any distance computed
against the config three mislabels every part generated around them --
which is exactly what produced the bad correlation at slots 1 and 5, the
two slots that carry a second `dist0.topk` part.

**So there are two anchor sets, and they are not interchangeable:**

    anchors()         3 sequences, from denovo_petase.yaml. What the
                      TRAINING REWARD tethered to, via dist_to_ref_seqs at
                      weight 10.
    design_anchors()  5 sequences, reconstructed from the library. What the
                      LIBRARY PARTS were generated around, and what every
                      distN tag is measured from.

Figure 2(c) reads the tags and is unaffected by the distinction. Any text
about what library 3 was aimed at should say which of the two it means.
