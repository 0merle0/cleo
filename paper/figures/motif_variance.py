#!/usr/bin/env python3
"""How tightly each library holds its catalytic atoms.

The claim the PETase section makes in prose -- that post-training "drastically
reduces variance and improves positioning of active site residues" -- has a
direct measurement on the AME benchmark, and this is it.

WHAT IS MEASURED, PRECISELY. `rfd2_motif_rmsd` is the RMSD of the predicted
motif atoms to the theozyme they were scaffolded around. Its SPREAD across a
library is how variably that library places its catalytic atoms. This is not
a per-atom RMSF: it is a scalar per prediction, so it says how far the motif
sits from where it should, not which atom moves most.

A true per-atom positional variance is NOT computable from what is on disk.
It needs the predicted structures, and `r1panel/recomb/folds/` is empty --
the stock-ProteinMPNN arm's CIFs were deleted and no tarball survives, so
every `af3_path` in that arm points at a file that is gone. Recovering it
means re-folding 256 designs x best-of-5 x 3 backbones = 3,840 AF3
predictions. Worth doing if a per-atom figure is wanted; not available today.

THE DECOMPOSITION. Each design is folded best-of-5, so the spread splits in
two and they mean different things:

    within-design   across the 5 diffusion samples of ONE sequence.
                    How reproducibly the oracle places the motif for a
                    sequence it has already been given.
    between-design  across the library's members.
                    How much the LIBRARY varies, which is the quantity the
                    method is supposed to move.

Reported over EVERY design, not only passing ones. Conditioning on passing
truncates the distribution at the pass threshold, which manufactures exactly
the variance reduction being claimed -- the post-trained arm passes more
often, so it would keep more of its own tail.

    uv run python paper/figures/motif_variance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
AME = CLEO / "experiments" / "ame"
sys.path.insert(0, str(HERE))

BB = {"M0097": "run_M0097_1ctt_cond9_14",
      "M0255": "run_M0255_1mg5_cond17_66",
      "M0315": "run_M0315_1ey3_cond18_25"}

ARMS = ("sampled", "chimera")
METRIC = "rfd2_motif_rmsd"


def _samples(site, arm):
    """-> per-sample scored rows (5 per design), or None."""
    st = BB[site]
    f = (AME / "r1panel" / "recomb" / f"{st}_baseline_scored_samples.csv"
         if arm == "sampled" else
         AME / "fragk" / "recomb" / f"{st}_fragk4_scored_samples.csv")
    return pd.read_csv(f) if f.exists() else None


def collect():
    """-> one row per (site, arm) with the variance decomposition.

    Within- and between-design components are variances, so they add. They
    are reported as standard deviations because the metric is an RMSD in
    angstroms and a variance in A^2 is not a quantity anyone reads.
    """
    rows = []
    for site in BB:
        for arm in ARMS:
            d = _samples(site, arm)
            if d is None:
                continue
            d = d.dropna(subset=[METRIC])
            g = d.groupby("name")[METRIC]

            # Within: pooled variance of samples about their own design mean.
            within_var = float(np.mean(g.var(ddof=1).dropna()))
            # Between: variance of the design means.
            means = g.mean()
            between_var = float(means.var(ddof=1))

            rows.append(dict(
                site=site, arm=arm,
                n_designs=int(means.size), n_samples=int(len(d)),
                mean=float(means.mean()),
                median=float(means.median()),
                sd_total=float(d[METRIC].std(ddof=1)),
                sd_within=np.sqrt(within_var),
                sd_between=np.sqrt(between_var),
                iqr=float(means.quantile(0.75) - means.quantile(0.25)),
                frac_under_1p5=float((means <= 1.5).mean()),
            ))
    return pd.DataFrame(rows)


def main():
    d = collect()
    show = d[["site", "arm", "n_designs", "mean", "median", "sd_total",
              "sd_within", "sd_between", "iqr", "frac_under_1p5"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nratio stock / post-trained, per backbone:")
    for site in BB:
        g = d[d.site == site].set_index("arm")
        if len(g) < 2:
            continue
        print(f"  {site}  sd_between {g.loc['sampled'].sd_between:.3f} -> "
              f"{g.loc['chimera'].sd_between:.3f}  "
              f"({g.loc['sampled'].sd_between / g.loc['chimera'].sd_between:.1f}x tighter)"
              f"   sd_within {g.loc['sampled'].sd_within:.3f} -> "
              f"{g.loc['chimera'].sd_within:.3f}")


if __name__ == "__main__":
    main()


def design_means():
    """-> long-form per-design motif RMSD, for plotting the distributions.

    One row per DESIGN, its RMSD averaged over the 5 diffusion samples. The
    per-sample rows are the wrong unit for a library panel: they would weight
    each design five times and mix the within-design spread, which is a
    property of the oracle, into a figure about the library.
    """
    rows = []
    for site in BB:
        for arm in ARMS:
            d = _samples(site, arm)
            if d is None:
                continue
            m = d.dropna(subset=[METRIC]).groupby("name")[METRIC].mean()
            rows.append(pd.DataFrame(dict(site=site, arm=arm,
                                          rmsd=m.to_numpy())))
    return pd.concat(rows, ignore_index=True)
