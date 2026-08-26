#!/usr/bin/env python3
"""The diversity/fidelity frontier: temperature is the baseline, CLEO is above it.

Temperature is how you buy diversity from LigandMPNN today, and it is a genuine
trade -- every point you gain in coverage costs you pass rate. Sweeping T from
0.1 to 1.0 traces that trade out as a curve, which is the honest thing for a new
method to be compared against. A method that only beat *one* temperature would
be beating a badly chosen operating point, not the baseline.

The claim this figure supports is "higher pass rate at matched diversity", and
it is deliberately narrower than "more diverse". On mean pairwise Hamming CLEO
is comparable to baseline T=0.7, not better; what CLEO changes is how often a
design at that spread folds correctly. Plotting U_k rather than Hamming makes
the supported claim the visible one.

    uv run python paper/figures/ame_frontier.py
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from ame_diversity import ARMS, BACKBONES, baseline, cleo, reference, stats
from figio import save
from palette import PALETTE

# Two series only: a neutral for the baseline curve and one accent for CLEO.
# Identity is carried by shape and direct labels as well as hue, so the panel
# survives both colourblindness and greyscale printing.
C_BASE, C_CLEO = PALETTE["gray"], PALETTE["blue"]
ARM_MARK = {"random": "o", "logprob_strat": "s", "logprob_band": "^", "logprob_top": "D"}
ARM_LABEL = {"random": "random", "logprob_strat": "strat",
             "logprob_band": "band", "logprob_top": "top"}


def collect(k):
    rows = []
    for bb in BACKBONES:
        ref = reference(bb)
        b = baseline(bb)
        for T, g in b.groupby("temperature"):
            rows.append(dict(backbone=bb, kind="baseline", label=f"T={T:g}",
                             temp=T, **stats(g, ref, k)))
        for arm in ARMS:
            d = cleo(bb, arm)
            if d is not None:
                rows.append(dict(backbone=bb, kind="cleo", label=arm,
                                 temp=float("nan"), **stats(d, ref, k)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=15, help="rarefaction depth for U_k")
    a = ap.parse_args()

    df = collect(a.k)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)

    for ax, bb in zip(axes, BACKBONES):
        d = df[df.backbone == bb]
        # Ordered by temperature, not by U_k: the line is the sweep's own path
        # through the plane, so reordering it would draw a curve nobody ran.
        # Temperatures with no passing design have no diversity to place and
        # are reported in text instead of piled on the origin.
        b = d[(d.kind == "baseline") & (d.n_pass > 0)].sort_values("temp")
        ax.plot(b.Uk, b.pass_pct, "-", color=C_BASE, lw=2, zorder=1,
                )
        for _, r in b.iterrows():
            ax.plot(r.Uk, r.pass_pct, "o", ms=5, mew=1.6, mec=C_BASE, zorder=2,
                    mfc="white" if r.n_pass >= a.k else C_BASE)
        # Ends only, pushed apart -- the low-T and high-T points sit close
        # together whenever the sweep is compressed, as it is on M0904.
        for r, off, ha in zip([b.iloc[0], b.iloc[-1]] if len(b) else [],
                              [(-7, -3), (8, -4)], ["right", "left"]):
            ax.annotate(r.label, (r.Uk, r.pass_pct), textcoords="offset points",
                        xytext=off, fontsize=7.5, color=C_BASE, ha=ha)

        for _, r in d[d.kind == "cleo"].iterrows():
            ax.plot(r.Uk, r.pass_pct, ARM_MARK[r.label], ms=8, mew=1.4, mec="white",
                    color=C_CLEO, zorder=3,
                    alpha=1.0 if r.n_pass >= a.k else 0.45)
            ax.annotate(ARM_LABEL[r.label], (r.Uk, r.pass_pct), textcoords="offset points",
                        xytext=(7, 4) if r.Uk else (7, 7), fontsize=7.5, color=C_CLEO)

        nz = d[(d.kind == "baseline") & (d.n_pass == 0)]
        if len(nz) == 6:
            note = "baseline: 0/96 at every T"
        elif len(nz):
            note = "baseline T=" + ", ".join(f"{t:g}" for t in nz.temp) + ": 0/96"
        else:
            note = ""
        if note:
            ax.annotate(note, (0.5, 0.88), xycoords="axes fraction", ha="center",
                        fontsize=7.5, color=C_BASE, style="italic")

        ax.set_title(bb.replace("run_", "").split("_cond")[0], fontsize=9)
        ax.set_xlabel(f"$U_{{{a.k}}}$  (substitutions, rarefied to {a.k} designs)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.margins(x=0.13)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("pass rate (%)", fontsize=8.5)
    axes[0].set_ylim(-4, 100)
    axes[0].plot([], [], "o", ms=5, mfc="white", mec=C_BASE, mew=1.6, ls="none",
                 label="LigandMPNN (T sweep)")
    axes[0].plot([], [], "o", color=C_CLEO, ms=8, mec="white", mew=1.4, ls="none",
                 label="CLEO (this work)")
    axes[0].plot([], [], "o", color=C_BASE, ms=5, ls="none",
                 label=f"filled: $<${a.k} passing, $U_k$ not rarefied")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "ame_frontier")
    print(df.to_string(index=False, float_format="{:.3g}".format))


if __name__ == "__main__":
    main()
