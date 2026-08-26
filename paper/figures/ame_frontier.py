#!/usr/bin/env python3
"""The diversity/fidelity frontier, across an easy, a medium and a hard backbone.

Temperature is how you buy diversity from LigandMPNN today, and it is a genuine
trade: every point of coverage costs pass rate. The sweep T=0.1..1.0 is plotted
as a cloud of points shaded by temperature rather than joined into a curve --
the six settings are six independent operating points, and a line between them
would assert a continuity nobody measured.

CLEO is the `random` arm alone: sequences as the policy emits them, with no
selection rule applied. Selection rules move both axes, so including them would
confound "the policy is better" with "the filter is better".

Backbones are ordered by how hard the baseline finds them -- best baseline pass
rate 66.7%, 10.4%, and 0% -- which is the axis along which the result matters.
An improvement on an easy backbone is a convenience; one on a backbone where the
baseline never succeeds is a different kind of claim.

The claim is "higher pass rate at matched diversity", deliberately narrower than
"more diverse": on mean pairwise Hamming CLEO is comparable to baseline T=0.7,
not better.

    uv run python paper/figures/ame_frontier.py

x-axis is U_k, distinct (position, residue) substitutions rarefied to k passing
designs. Raw U grows with how many designs passed, so plotting it against pass
rate would partly plot pass rate against itself. Normalising as U/n does not fix
that -- it inverts it, falling 57.4 -> 12.2 as n goes 5 -> 86 on one fixed
policy. Rarefaction is the fix: it holds the design count constant.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ame_diversity import BACKBONES, baseline, cleo, reference, stats
from figio import save
from palette import PALETTE

# Temperature is a magnitude, so it gets a sequential ramp: one hue, light to
# dark. The hue is deliberately not blue -- CLEO's marker is blue, and a blue
# ramp would put the baseline's dark end in the same visual family as the thing
# it is being compared against. Orange against blue also survives the common
# colour-vision deficiencies.
T_RAMP = LinearSegmentedColormap.from_list("temp", ["#F6D5B0", "#D97A29", "#7A3D08"])
C_CLEO = PALETTE["blue"]
C_TEXT = "#4B5563"

# Ordered easy -> hard by best baseline pass rate (66.7%, 10.4%, 0%).
DIFFICULTY = ["easy", "medium", "hard"]


def collect(k, arm):
    rows = []
    for bb in BACKBONES:
        ref = reference(bb)
        for T, g in baseline(bb).groupby("temperature"):
            rows.append(dict(backbone=bb, kind="baseline", temp=T, **stats(g, ref, k)))
        d = cleo(bb, arm)
        if d is not None:
            rows.append(dict(backbone=bb, kind="cleo", temp=float("nan"),
                             **stats(d, ref, k)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=15, help="rarefaction depth for U_k")
    ap.add_argument("--arm", default="random", help="CLEO selection arm to plot")
    a = ap.parse_args()

    df = collect(a.k, a.arm)
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Shared axes: with one CLEO point per panel, an autoscaled hard panel would
    # zoom to a 40-unit window and make its coverage look like a full sweep's.
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), sharey=True, sharex=True)

    for ax, bb, tier in zip(axes, BACKBONES, DIFFICULTY):
        d = df[df.backbone == bb]
        b = d[(d.kind == "baseline") & (d.n_pass > 0)]
        ax.scatter(b.Uk, b.pass_pct, c=b.temp, cmap=T_RAMP, norm=norm, s=62,
                   edgecolor="white", lw=1.4, zorder=2)

        c = d[d.kind == "cleo"].iloc[0]
        ax.plot(c.Uk, c.pass_pct, "o", color=C_CLEO, ms=12, mec="white", mew=1.6, zorder=3)
        ax.annotate("CLEO", (c.Uk, c.pass_pct), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=9.5, color=C_CLEO,
                    fontweight="bold")

        # Temperatures that pass nothing have no diversity to place; say so in
        # words rather than piling them on the origin.
        dead = d[(d.kind == "baseline") & (d.n_pass == 0)]
        if len(dead) == 6:
            ax.annotate("LigandMPNN\n0/96 at every $T$", (0.30, 0.30),
                        xycoords="axes fraction", ha="center", fontsize=8.5,
                        color=C_TEXT, style="italic")
        elif len(dead):
            ax.annotate("$T$=" + ", ".join(f"{t:g}" for t in dead.temp) + ": 0/96",
                        (0.5, 0.93), xycoords="axes fraction", ha="center",
                        fontsize=7.5, color=C_TEXT, style="italic")

        ax.set_title(f"{tier}\n{bb.replace('run_', '').split('_cond')[0]}",
                     fontsize=10, linespacing=1.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.17, y=0.15)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("pass rate (%)", fontsize=10)
    axes[0].set_ylim(-4, 100)

    # Explicit geometry rather than colorbar(ax=...), which re-lays-out the
    # panels underneath and pushes the x-label into the tick labels.
    fig.subplots_adjust(left=0.075, right=0.875, bottom=0.20, top=0.84, wspace=0.12)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=T_RAMP),
                      cax=fig.add_axes([0.895, 0.20, 0.014, 0.64]),
                      ticks=[0.1, 0.3, 0.5, 0.7, 1.0])
    cb.set_label("LigandMPNN sampling temperature", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)

    fig.supxlabel(f"$U_{{{a.k}}}$: distinct substitutions, rarefied to {a.k} passing designs",
                  fontsize=9, color=C_TEXT, x=0.475, y=0.035)
    save(fig, "ame_frontier", tight=False)
    print(df.to_string(index=False, float_format="{:.3g}".format))


if __name__ == "__main__":
    main()
