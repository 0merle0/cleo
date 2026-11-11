# Why G20 = 22.2 × momi WT

Every PETase activity number in this repo is measured against **G20** and
converted once to a multiple of **momi WT**, the enzyme the project started
from. This file is the justification for that conversion factor, because it
is a single constant that scales every reported "× momi" figure and it
disagrees with the value used elsewhere in the project.

**Factor: 19.7** — the midpoint of the two condition-matched measurements,
22.207 and 17.113. Set in `paper/figures/libraries_snapshot.py` as
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

| experiment | temp | momi × G20 | **factor** | momi expr | CV |
|---|---|---|---|---|---|
| e0321 pool screen | **22.4 °C** | 0.0450 | **22.2** | 0.787 µM | **0.004** |
| library 1 Dec re-test, plate 3 | **23.7 °C** | 0.0585 | **17.1** | 0.139 µM | 0.013 |
| e0328 overnight | 37 °C | 0.0294 | *34.0* | 0.536 µM | 0.007 |
| e0328 overnight | 60 °C | 0.1473 | *6.8* | 0.469 µM | 0.110 |
| e0328 round 5, short | 31.7 °C | **−0.017** | — | 0.826 µM | 0.010 |
| e0328 round 5, short | 60 °C | **−0.399** | — | 0.927 µM | 0.122 |
| library 1, control :1 | 30 °C | 0.046 | *21.6* | 0.097 µM | — |
| library 1, control :2 | 30 °C | 0.125 | *8.0* | 0.115 µM | — |

**Only the first two rows are condition-matched**, and they are the estimate:
both ambient (22.4 and 23.7 °C), both short kinetic reads, both on plates
where momi and G20 sit side by side. They give **22.2 and 17.1**.

Everything in italics is excluded, for a stated reason:

- **37 °C and 60 °C overnight (34.0 and 6.8).** Different temperature AND
  different duration. momi WT and G20 need not share a thermal profile —
  indeed the ratio moves from 34.0 to 6.8 between 37 °C and 60 °C, which is
  itself evidence that temperature changes it — so neither transfers to the
  ambient reads the libraries were measured at. The 60 °C point also runs
  the enzymes near their stability limit at CV 0.110.
- **The two short reads.** momi comes out negative, below background.
- **Library 1's September controls (21.6 and 8.0).** Saturated plate, and
  momi barely expressing at 0.097–0.115 µM — row :1 fails the house
  expression filter outright.

So the defensible range is **17.1 to 22.2**, and the value in use is their
**midpoint, 19.7** (mean 19.660; geometric mean 19.494, which rounds the
same way).

The midpoint rather than the better-conditioned single value, because the
two are **26% apart** and neither experiment is strong enough to override
the other. The pool screen has the better measurement — momi at 0.787 µM
against 0.139, CV 0.004 against 0.013 — but library 1's December re-test is
a clean kinetic read on its own plate with its own controls, and nothing
about it says to discard it. Quoting a midpoint says what the data supports;
quoting 22.2 would imply the other measurement did not happen.

**13 still sits below both matched measurements**, and nothing at ambient
temperature supports it.

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

## What this means on a plot

Because library 1 is normalised to momi directly while libraries 2 and 3 are
converted, **G20 does not land at the same height in all three**:

| | G20, × momi | |
|---|---|---|
| library 1, its own G20 well | **17.11** | measured against its own momi control |
| library 2 | 19.70 | derived — 1.000 × G20 × 19.7 |
| library 3 | 19.70 | derived |

That 2.6-unit gap is the 26% disagreement between the two matched
measurements, made visible. It is not an error: library 1 says G20 is 17.1 ×
momi, and we quote 19.7 because the other experiment says 22.2.

It is left visible deliberately. The alternative — normalising library 1 to
its own G20 well instead — would force all three to 19.70 and look tidier,
but it would hide the one piece of disagreement the factor rests on, and it
would make library 1's momi axis derived rather than measured, which is the
opposite of what its plate layout supports (momi is on all three plates,
G20 only on plate 3).

If a figure needs a single G20 reference line, draw it at 19.7 and expect
library 1's own G20 marker to sit slightly below it.

## What it changes

Nothing relative. The factor is a constant rescale, so every **× G20**
value, every ranking, every percentage and every library-to-library
comparison is identical at 13 or 22.2. Only the absolute "× momi" headline
moves:

| | at 13 | at 19.7 |
|---|---|---|
| G20 | 13.0 | 19.7 |
| Fast_PETase | 20.5 | 31.1 |
| library 3 best design | 260.7 | 394.9 |

**These numbers now differ from jq01's notebooks**, which still apply 13.
Any "× momi" figure taken from there is 1.51× lower than one taken from
here. That is deliberate, and it is the reason this file exists.

## What would settle it

A head-to-head of momi WT and G20 on one plate, replicated across plates and
days — the pool screen is a single plate on a single day, n=1 per enzyme
with 3 replicate wells. Repeating it would turn the strongest single
measurement into a reliable one.

Related: [COMPARISONS.md](COMPARISONS.md) ·
`paper/figures/libraries_snapshot.py` · `paper/figures/wt_benchmark.py`
