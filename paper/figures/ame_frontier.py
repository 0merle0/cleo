#!/usr/bin/env python3
"""The diversity/fidelity frontier: temperature is the baseline, CLEO is above it.

Temperature is how you buy diversity from LigandMPNN today, and it is a genuine
trade -- every point of coverage costs pass rate. Sweeping T=0.1..1.0 traces
that trade as a curve, which is the honest thing to compare a new method
against; beating a single temperature would only mean beating a badly chosen
operating point.

CLEO is shown as the `random` arm alone: sequences taken as the policy emits
them, with no selection rule applied. Selection rules move both axes, so
including them would confound "the policy is better" with "the filter is
better". The on-policy draw is the claim about the policy.

The claim is "higher pass rate at matched diversity", deliberately narrower than
"more diverse". On mean pairwise Hamming CLEO is comparable to baseline T=0.7,
not better; what changes is how often a design at that spread folds correctly.

    uv run python paper/figures/ame_frontier.py

On the x-axis: U_k, distinct (position, residue) substitutions rarefied to k
designs. Raw U grows with how many designs passed, so plotting it against pass
rate would partly plot pass rate against itself. U/n does not fix this -- it
inverts it, falling 57.4 -> 12.2 as n goes 5 -> 86 on a single fixed policy.
Rarefaction is the fix: it holds the design count constant.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from ame_diversity import BACKBONES, baseline, cleo, reference, stats
from figio import save
from palette import PALETTE

C_BASE, C_CLEO = PALETTE["gray"], PALETTE["blue"]


def collect(k, arm):
    rows = []
    for bb in BACKBONES:
        ref = reference(bb)
        for T, g in baseline(bb).groupby("temperature"):
            rows.append(dict(backbone=bb, kind="baseline", label=f"T={T:g}",
                             temp=T, **stats(g, ref, k)))
        d = cleo(bb, arm)
        if d is not None:
            rows.append(dict(backbone=bb, kind="cleo", label="CLEO",
                             temp=float("nan"), **stats(d, ref, k)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=15, help="rarefaction depth for U_k")
    ap.add_argument("--arm", default="random", help="CLEO selection arm to plot")
    a = ap.parse_args()

    df = collect(a.k, a.arm)
    # Shared x as well as y: with one CLEO point per panel, an autoscaled
    # M0907 would zoom to a 40-unit window and make its coverage look
    # comparable to a full sweep's.
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True, sharex=True)

    for ax, bb in zip(axes, BACKBONES):
        d = df[df.backbone == bb]

        # Ordered by temperature: the line is the sweep's own path through the
        # plane. Reordering by U_k would draw a curve nobody ran. Temperatures
        # that pass nothing have no diversity to place and are noted in text.
        b = d[(d.kind == "baseline") & (d.n_pass > 0)].sort_values("temp")
        ax.plot(b.Uk, b.pass_pct, "-o", color=C_BASE, lw=1.8, ms=5,
                mfc="white", mec=C_BASE, mew=1.5, zorder=2,
                label="LigandMPNN ($T$ sweep)")

        c = d[d.kind == "cleo"].iloc[0]
        ax.plot(c.Uk, c.pass_pct, "o", color=C_CLEO, ms=11, mec="white", mew=1.5,
                zorder=3, label="CLEO (on-policy)")

        # The sweep's two ends, pushed apart -- they sit close together
        # whenever the sweep is compressed, as it is on M0904.
        # Offset horizontally, not vertically: the sweep's high-T end sits
        # near the floor, where a label placed below would fall off the axis.
        for r, off, ha in zip([b.iloc[0], b.iloc[-1]] if len(b) else [],
                              [(-7, -3), (7, -3)], ["right", "left"]):
            ax.annotate(r.label, (r.Uk, r.pass_pct), textcoords="offset points",
                        xytext=off, ha=ha, fontsize=7, color=C_BASE)

        nz = d[(d.kind == "baseline") & (d.n_pass == 0)]
        if len(nz) == 6:
            ax.annotate("LigandMPNN: 0/96\nat every $T$", (0.30, 0.30), xycoords="axes fraction",
                        ha="center", fontsize=8.5, color=C_BASE, style="italic")

        ax.set_title(bb.replace("run_", "").split("_cond")[0], fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.16, y=0.14)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("pass rate (%)", fontsize=10)
    axes[0].set_ylim(-4, 100)
    # Legend goes in the emptiest panel, which is the one whose baseline
    # never passes anything.
    axes[2].legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False,
                   loc="upper left", borderpad=0, handletextpad=0.5)
    fig.supxlabel(f"$U_{{{a.k}}}$: distinct substitutions, rarefied to {a.k} passing designs",
                  fontsize=9, color="#4B5563", y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "ame_frontier")
    print(df.to_string(index=False, float_format="{:.3g}".format))


if __name__ == "__main__":
    main()
