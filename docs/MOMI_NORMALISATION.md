# Why G20 = 22.2 × momi WT

Every PETase activity number in this repo is measured against **G20** and
converted once to a multiple of **momi WT**, the enzyme the project started
from. This file is the justification for that conversion factor, because it
is a single constant that scales every reported "× momi" figure and it
disagrees with the value used elsewhere in the project.

**Factor: 22.2.** Set in `paper/figures/libraries_snapshot.py` as
`MOMI_PER_G20`.

## Why a conversion is needed at all

momi WT cannot be the measured reference. In a short plate read it sits at
or below background:

| experiment | momi WT, × G20 |
|---|---|
| e0328 round 5, 37 °C short | **−0.017** |
| e0328 round 5, 60 °C short | **−0.399** |
| e0328 overnight, 37 °C | 0.029 |
| e0328 overnight, 60 °C | 0.147 |

Negative values mean it is under the background. That is not an assay
failure — momi WT is a weak enzyme, and being hard to see above vector is
the premise of the whole campaign. But it means nothing can be normalised
to it directly. G20 is used instead, because it is the one design
comfortably above background in all three libraries, and converted once.

## The primary evidence

`jq_e0321_260108_..._pool_screen_reading_analysis` was run for exactly this
comparison. From its README:

> **Goal:** analyze plate reading data for 2nd big library pooled screen …
> as well as **WT momi, G20 momi**, top 5 from round 3 from first library,
> **FastPETase, LCC, thermoPETase** comparison

Both enzymes, one plate, one condition
(`260109_processed_filtered_data_round0_pooled_2nd_big_lib.csv`):

| | well | raw rate | × G20 | expression | CV | r² | reps |
|---|---|---|---|---|---|---|---|
| G20 | E3 | 4.494e-05 | 1.000 | 0.606 µM | 0.019 | 0.9982 | 3/3 |
| momi WT | D1 | 2.024e-06 | **0.045** | 0.787 µM | **0.004** | **0.9999** | 3/3 |

**Ratio 22.2.**

Three things make this hard to dismiss:

1. **momi WT expresses the best of every reference enzyme on the plate** —
   0.787 µM against G20's 0.606, FastPETase's 0.366 and ThermoPETase's
   0.336. Its low activity is therefore not an expression artefact; there
   was more of it than of anything it lost to.
2. **The measurement is clean** — 3/3 replicates, CV 0.4%, r² 0.9999 on the
   kinetic fit. It is the tightest row in the file.
3. **Same plate, same read, same background subtraction** as the G20 it is
   divided by, so no cross-plate bridging is involved.

## Every experiment that measured both, and what each gives

Searched all 22,410 CSVs under `jq_p07_active_learning` for files containing
both sequences; 16 matched, 8 with activity columns. After excluding
archives and duplicates, these are the independent comparison points:

| experiment | momi × G20 | **factor** | momi expression | CV |
|---|---|---|---|---|
| e0321 pool screen | 0.0450 | **22.2** | 0.787 µM | **0.004** |
| e0328 overnight 37 °C | 0.0294 | **34.0** | 0.536 µM | 0.007 |
| e0328 overnight 60 °C | 0.1473 | 6.8 | 0.469 µM | 0.110 |
| library 1 December re-test, plate 3 | 0.0585 | **17.1** | 0.139 µM | 0.013 |
| e0328 round 5, 37 °C short | **−0.017** | — | 0.826 µM | 0.010 |
| e0328 round 5, 60 °C short | **−0.399** | — | 0.927 µM | 0.122 |
| library 1, control row :1 | 0.046 | 21.6 | **0.097 µM** | — |
| library 1, control row :2 | 0.125 | 8.0 | **0.115 µM** | — |

The two short reads are unusable: momi comes out **negative**, i.e. below
background, which is the same reason it cannot be the reference in the first
place. Only the longer reads resolve it at all.

**Library 1's two rows should be discounted entirely, on expression.** Its
momi controls sit at 0.097 and 0.115 µM against the pool screen's 0.787 µM —
seven to eight times less protein, and right on the 0.1 µM filter line. By
the house rule of at least 2 of 3 replicates above 0.1 µM:

| row | replicates (µM) | passing | factor |
|---|---|---|---|
| :1 | 0.090, 0.105, 0.096 | **1 of 3 — fails** | 21.6 |
| :2 | 0.131, 0.115, 0.097 | 2 of 3 — passes | 8.0 |

So the row giving 21.6 does not clear the project's own expression filter,
and the row that does gives 8.0. Neither is a usable estimate: at ~0.1 µM
the enzyme is barely present, and library 1's plate is saturated besides.
Library 1 is where G20 was found; it is not where this factor should be
measured.

**The pattern is consistent and it does not support 13.** The three
best-conditioned measurements — CV 0.004, 0.007 and 0.013 — give **22.2,
34.0 and 17.1**, all above 13. Every value at or below 13 comes from a measurement that
is otherwise compromised: the 60 °C overnight has CV 0.110 and runs the
enzymes near their stability limit, and library 1's 8.0 is one of two control
rows that disagree by 2.7× on a saturated plate.

So 13 sits at the bottom of everything measured, and nothing clean supports
it. 22.2 is the best single measurement; 34.0 is the only other clean one and
points the same way.

Two experiments contain both sequences but cannot contribute: `e0333`
(validation set, 60 °C HPLC) and `e0340`/`e0344` are protein-purification
runs — momi and G20 have HPLC concentrations there but no activity merged in,
so `g20_norm_rate` is null for both.

## What else was on the pool-screen plate

Context for the primary evidence — e0321 was not a momi-vs-G20 experiment
alone, which is part of why it is well controlled:

| category | n | × G20 range |
|---|---|---|
| pooled library fragments (5 fragments × 4 plates) | 20 | 0.058 – 1.639 |
| top designs from library 2 round 3 | 5 | 1.066 – 1.916 |
| natural PETase benchmarks (FastPETase, ThermoPETase, LCC) | 3 | 0.924 – 1.577 |
| G20 positive control | 1 | 1.000 |
| G20 ordered as a single gene | 1 | 0.927 |
| momi WT | 1 | 0.045 |

**G20 measured two independent ways on that plate agrees to 7%** — 1.000 as
the library construct against 0.927 ordered as a single gene. That is a
direct check that the reference itself is reproducible within the
experiment the factor comes from.

## Where the 13 comes from

jq01's notebooks apply `g20_norm_rate * 13  # approximate to momi activity`.
It is a hardcoded constant, labelled approximate in its own comment, and it
appears in **the same notebook that measures momi at 0.045 × G20**. So the
13 was already in the code and was not revised when this experiment produced
the number that supersedes it.

The internal inconsistency is visible in jq01's own output: under a factor
of 13, the `momi_norm_rate` of the **momi WT row itself** reads **0.585**,
where it must be 1.0 by definition. At 22.2 it reads 1.000.

## Why library 1 cannot arbitrate

Library 1 contains momi WT as its positive control, so in principle it gives
an independent estimate. In practice its two control rows disagree too much:

| row | × G20 | implied factor |
|---|---|---|
| `1.wt_2.wt_3.wt_4.wt_5.wt:1` | 0.046 | 21.6 |
| `1.wt_2.wt_3.wt_4.wt_5.wt:2` | 0.125 | 8.0 |

**2.7× apart**, and the range covers both candidates. The mean gives 11.7,
which is why an earlier version of this analysis reported library 1 as
supporting 13 — that was reading the mean of two discordant measurements as
an estimate.

Library 1 is also the worst-conditioned experiment for this: its plate is
saturated, its background rests on n=2 negative-control wells 23% apart, and
momi sits only ~30 units above that background, so the derived ratio moves
from 11.7 to 24.0 purely on whether the background is a median or a mean.

**The one point in 13's favour:** under 13, library 1's two momi controls
average 1.12 on their own axis, close to the 1.0 they should be. Under 22.2
they average 1.9. That is a genuine argument, resting on a measurement whose
two replicates differ by 2.7×.

## What it changes

Nothing relative. The factor is a constant rescale, so every **× G20**
value, every ranking, every percentage and every library-to-library
comparison is identical at 13 or 22.2. Only the absolute "× momi" headline
moves:

| | at 13 | at 22.2 |
|---|---|---|
| G20 | 13.0 | 22.2 |
| Fast_PETase | 20.5 | 35.0 |
| library 3 best design | 260.7 | 445.1 |

**These numbers now differ from jq01's notebooks**, which still apply 13.
Any "× momi" figure taken from there is 1.71× lower than one taken from
here. That is deliberate, and it is the reason this file exists.

## What would settle it

A head-to-head of momi WT and G20 on one plate, replicated across plates and
days — the pool screen is a single plate on a single day, n=1 per enzyme
with 3 replicate wells. Repeating it would turn the strongest single
measurement into a reliable one.

Related: [COMPARISONS.md](COMPARISONS.md) ·
`paper/figures/libraries_snapshot.py` · `paper/figures/wt_benchmark.py`
