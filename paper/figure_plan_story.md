# Figure plan, keyed to the story draft

Written 2026-11-16, against the first full prose draft. The draft makes ~26
distinct figure references, most of them written as `figure 1?` or `fig 2?`
because the panel did not exist to point at. This file resolves every one:
what the panel shows, whether the data exists, and where it comes from.

`paper/figure_plan.md` is the older, computational-only plan under a different
numbering. Where the two disagree, this file wins.

---

## 0. The one renumbering this needs

The draft puts the concept, the method, the GRPO objective, the shuffling
scheme, the AME benchmark, the diversity result, the sequence-space
embedding, the k sweep and the cost/scale argument all in **Figure 1**. That
is nine panels of two different kinds of claim, and it is why every reference
reads `figure 1?`.

Split it. Everything downstream shifts by one:

| draft | becomes | carries |
|---|---|---|
| Fig 1 (concept half) | **Fig 1** | why combinatorial, and what the method is |
| Fig 1 (benchmark half) | **Fig 2** | AME: it works, and how far it scales |
| Fig 2 | **Fig 3** | PETase: the design, and what post-training did to it |
| Fig 3 | **Fig 4** | the experimental platform and the three libraries |
| Fig 4 | **Fig 5** | searching a 10^12 library in four rounds |

Five main figures. Everything else supplementary. The rest of this file uses
the new numbers.

---

## 1. Every figure reference in the draft, resolved

Status key: **have** = data in hand, panel is a build job. **derive** = data
exists but needs a new reduction or a new script. **collect** = the
measurement or the run does not exist yet.

### Figure 1 — why combinatorial, and what the method is

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 1a | "figure 1a on the left / middle / right" | three landscapes: error-prone PCR, naive combinatorial, filtered combinatorial | **have** | `concept_library.py:draw_landscape` — toy, no data. **Redraw as 3D (see §3).** |
| 1b | "sample Nsampled new sequences (figure 1?)" | the shuffling scheme: Nbatch=16 decoded, cut into k, Nsampled=64 chimeras, every fragment seen 4x | **have** | schematic; `concept_loop.py:panel_loop` is the credit-assignment half of it |
| 1c | "INSERT GRPO OBJECTIVE HERE" | the per-fragment GRPO objective, typeset, variables labelled against 1b | **have** | equation, not a plot — belongs inside the figure so its symbols sit next to the scheme they describe |
| 1d | "on the order of 10^4 oracle calls ... can easily reach 10^12" | oracle calls spent vs library size reached, and per-variant cost | **have** | `panel_cost.py`, `concept_library.py:gene_cost/library_cost` |

### Figure 2 — AME: it works, and how far it scales

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 2a | "increase success from these backbones for sequences sampled from the library" | recombinant pass rate, sampled vs plain RL vs chimera-trained, per backbone | **have** | `key_results.md` §6 table; `arm_table.py` |
| 2b | "increase diversity ... improving the novel of unique mutations" | distinct substitutions carried by *passing* members, rarefied to matched count | **have** | `ame_fragk_diversity.csv`, `key_results.md` §1 |
| 2c | "low dimensional representation of random mutagenesis scoped libraries, naive proteinMPNN, and the post-trained model" | three arms in one embedding | **derive** | archived `ame_pca.py`. **Third arm needs building** — a random-mutagenesis library at matched size; see §2 |
| 2d | "libraries out to k=16 ... still yield strong highfidelity libraries" | retention vs k, trained-at-k vs cut-at-k | **have** | `ame_fragk.csv` — supports a *stronger* claim than the draft makes; see §4 |

### Figure 3 — the PETase, and what post-training did to it

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 3a | "designed from a theozyme alone of PET polymer docked into the active site (fig 2a)" | theozyme → RFD2 backbone → MPNN → filters → momi; the design funnel | **have** | `structures/png/momi_fold.png`, `momi_site.png`; filter thresholds are in the draft prose |
| 3b | "optimizing for these rewards drastically reduces variance and improves positioning and confidence of active site residues (Fig 2?)" | per-reward distributions, naive vs post-trained library | **derive** | needs the PETase run's own metrics; AME equivalent is `panel_vanilla.py` |
| 3c | "Visualizations of the active site atoms from naive and optimized libraries (fig2?)" | superposed catalytic atoms, two clouds | **derive** | `structures/render_momi.py` exists; needs folds for both arms |
| 3d | "ability to center the sequence space of the library to be tethered to a top-k set (figure 2?)" | library 3's sequence space centred on the top-3 measured variants vs library 2's | **have** | part names carry design intent (`dist0.topk1`, `distN.*`); compute real Hamming — **the `distN` tag disagrees with computed distance on 992/1803 parts** |

### Figure 4 — the platform and the three libraries

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 4a | "as seen depicted in figure 3a" | wet-lab pipeline strip: geneblocks → ECHO → GGA → PCR → PUREfrex → reader | **have, built** | `svg/wetlab_pipeline.svg` (Jason, `312211c`) |
| 4b | "gate samples on each replicate having at least 0.1uM ... (figure 3b)" + "background subtracted and fit with least squares (figure 3b)" | the filter cascade with counts surviving each gate | **have** | `libraries_snapshot.py` `passes_*` columns |
| 4c | "consistency amongst measured control rates seen in figure 3c" | positive-control rate per plate, per round, all three libraries | **have** | snapshot controls. Plates agree to 1–2% within a round (20.70/20.18/20.05) — **plot raw rate, not momi-normalised**, which is 19.7 by construction in most rounds |
| 4d | "parity plots of individual replicates seen in figure 3d" | replicate i vs replicate j | **collect** | the snapshot is already reduced to means; `cv_rate` survives, the three replicate values do not. Re-extract from jq01's per-round raw files |
| 4e | — (draft skips e) | one worked kinetic trace: raw fluorescence, background, least-squares fit | **have** | `library1_kinetics.py` reads the raw reader files already |
| 4f | "aggregate measurements from each of three different libraries ... figure 3f?" + "nearly 400x ... figure 3x" | all three libraries on one momi axis, by round | **have, built** | `petase_libraries_filtered.csv`; medians 1.15 / 3.69 / 20.27, max 17.1 / 66.0 / 395.0 |

### Figure 5 — searching 10^12 in four rounds

| panel | draft text | shows | status | source |
|---|---|---|---|---|
| 5a | "combine each designed fragment with the other parent fragments ... figure 4a" | single-body effect of every part, by slot | **have** | round 0 is *exactly* this — 1665 (lib 2) and 1791 (lib 3) designs, every one with 4 of 5 slots = G20 |
| 5b | "correlation between distance from top-k fragment and activity ... figure 4a" | activity vs Hamming distance of the one varied part | **have** | same rows; compute distance, do not trust `distN` |
| 5c | "binning fragments ... ~128 variants per bin ... figure 4b" | round 1 activity by bin, with the top-2 exhaustive block called out | **have** | round 1 tags `bin1`..`bin6` (19–165 each) + `top2combos` (31 of the 32 possible) |
| 5d | "using the predictors ... next 2 rounds ... figure 4x" | rounds 2–4, predicted vs measured, by selection strategy | **have** | tags `MLPgamma1/5/8`, `topkMLP`, `sampledtopkMLP`, `top30mu`, `top10mu`, `mostdiverseMuGt3/4` |
| 5e | "in many fewer evaluations (put real savings here!) than greedy would require" | policy-over-fragment-space vs exhaustive/greedy screening cost | **derive** | counts are computable from library size and rounds; the greedy comparator needs defining |
| 5f | "comparing our top designs to native PETases ... at XXx faster Kcat figure 4x" | designs vs FastPETase / ThermoPETase / LCC / Kubu / MiPa, one plate | **have** | **`jq_e0328`, and it is better than we thought — see §5** |
| 5g | "we do not find that optimization of this probe substrate transfers well to bulk PET" | probe activity vs nanoPET / pellet activity | **collect** | `jq_e0334/3_sm_hplc_nanopet_titration` and `jq_e0331/0333` HPLC exist as notebooks only. Needs extraction, or ask Jason |
| 5h | "many variants distant in sequence space that are significantly more accurate" | activity vs distance from momi, final libraries | **have** | snapshot. **Note the first library said the opposite** — activity fell with distance (`key_results.md` §7) |

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

1. **Replicate-level values for the parity plot (4d).** Everything else about
   Figure 4 is in hand; this is the one panel with no path from the snapshot.
   The raw readers are reachable the way `library1_kinetics.py` reaches them.
2. **Bulk PET / nanoPET (5g).** The draft states a negative result — that
   probe gains do not transfer. A stated negative needs its panel. The
   notebooks exist; the extracted numbers do not.
3. **The random-mutagenesis arm for 2c.** `jq_e0307_250311_petase_library_random_mutation`
   has AF3 output for a random-mutagenesis PETase library, which may serve
   directly; otherwise it is a cheap generate-and-fold.
4. **PLM vs one-hot (S3).** Runnable today against the snapshot. Half a day.
5. **The fragment-credit ablation (S1).** A training run, and the draft leans
   on it for the central methodological choice.
6. **The shuffling comparison (S4).** Bundle is ready. Blocked on queue
   access, not on us.

---

## 3. The landscapes go 3D

The conceptual landscapes in 1a are currently **2D filled contours** seen from
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
must not be mixed: use 19.7 for the cross-library axis in 4f, and this plate's
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
| "(put real savings here!)" | needs the greedy comparator defined | §1, 5e |
| "10^4 oracle calls" | 200 steps x 32 x best-of-5 = 3.2x10^4 | `panel_cost.py` |
