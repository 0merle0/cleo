#!/usr/bin/env python3
"""What the fragment-count sweep looks like during training, not after it.

`ame_fragk.py` reports one number per cell -- the retention of a library built
from the final checkpoint. That hides the thing the convergence question turns
on: whether a cell had stopped improving when its checkpoint was taken. These
are the per-step histories of all eighteen runs, three backbones by
k in {2, 4, 8, 16, 36, 64}.

Two things to know about the columns before reading them.

**`reward` is useless here and is deliberately not plotted.** GRPO's advantage
is `(r - mean) / (std + 1e-3)`, centred within the rollout, so the logged
`reward` is that centred quantity and sits at ~1e-8 for every step of every
one of the eighteen runs. It is numerically zero, not flat-because-converged.
A "training curve" of it would be a straight line through the origin and would
look like a policy that never learned anything. `reward_std` is the batch
spread and does carry signal, but it is a spread, not a score.

**`frag_chimera_pass_af3` has eight points per run.** The in-loop AF3 pass is
evaluated on a sparse cadence, so it is drawn as markers rather than a line;
`frag_chimera_pass` is the dense per-step proxy and is what the curves are.

So the panels are the two dense in-loop signals:

  chimera pass   fraction of recombinants built inside the rollout that pass.
                 Rises from 0; this is the learning curve.
  chimera RMSD   geometric quality of those same recombinants. Falls; a second
                 view of the same learning that does not saturate at 1.0, so
                 it separates cells that the pass rate pins at the ceiling.

In-loop numbers are NOT comparable to the retention in `ame_fragk.py`: the
in-loop chimera pass includes unrecombined parents (rho=0), roughly a quarter
of the batch, which `make_recombinants.py` excludes. These curves are for
reading convergence, not for reading effect size.

    uv run python paper/figures/ame_fragk_curves.py [--window 11]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
AME = CLEO / "experiments" / "ame"
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import PALETTE  # noqa: E402

BB = {"M0097": "run_M0097_1ctt_cond9_14",
      "M0255": "run_M0255_1mg5_cond17_66",
      "M0315": "run_M0315_1ey3_cond18_25"}
# The k=4 anchor is a different family per backbone, and a different run
# length: 150 / 300 / 152 steps against 200 for the fragk arms. Drawn dashed
# for that reason -- it is not a matched member of the sweep.
ANCHOR = {"M0097": ("fragbsweep", "bs16"),
          "M0255": ("fragb16", "bs16"),
          "M0315": ("fragb16", "bs16")}
KS = (2, 4, 8, 16, 36, 64)
# k is ordered, so it gets one hue light->dark, not categorical colors.
RAMP = ["#A5C3DF", "#7FA5CC", "#5A87B4", "#3C6A99", "#2A4F78", "#15314D"]
COLOR = dict(zip(KS, RAMP))
METRICS = [("frag_chimera_pass", "in-loop chimera pass rate", None),
           ("frag_chimera_rmsd", r"chimera RMSD ($\AA$)", None)]
# The latest step present in all eighteen cells. Before E18b this was 149 and
# a matched-step comparison had to happen at or before it; now every cell runs
# the full 200, so the line simply marks the common endpoint.
COMMON = 199


def metrics_path(site, k):
    """-> the train_metrics.csv for the best cell on disk, or None.

    E18b trained k=4 in-family and reran M0315/k=16 as `fragk16b`, so all
    eighteen are a matched 200 steps. The borrowed anchor is kept last as a
    fallback for a tree where E18b has not run.
    """
    c = [("fragk", f"{BB[site]}_fragk{k}")]
    if k == 16:
        c.insert(0, ("fragk", f"{BB[site]}_fragk16b"))
    if k == 4:
        fam, suf = ANCHOR[site]
        c.append((fam, f"{BB[site]}_{suf}"))
    for fam, stem in c:
        p = AME / fam / stem / stem / f"{stem}_train_metrics.csv"
        if p.exists():
            return p
    return None


def load():
    """-> one long frame of every step of every cell.

    Guards the resume hazard: a resumed run restarts at step 0 and appends into
    the same tree, so a non-monotone step column means the history is two runs
    interleaved and its per-step numbers cannot be read. All eighteen are
    monotone today; this fails loudly if that changes.
    """
    out = []
    for site in BB:
        for k in KS:
            p = metrics_path(site, k)
            if p is None:
                print(f"  missing: {site} k={k}")
                continue
            d = pd.read_csv(p)
            if not d.step.is_monotonic_increasing:
                raise ValueError(
                    f"{p} has a non-monotone step column: a resume wrote into "
                    "an existing run's tree and the history is not one run.")
            d["site"], d["k"] = site, k
            out.append(d)
    return pd.concat(out, ignore_index=True)


def smooth(s, window):
    return s.rolling(window, center=True, min_periods=1).mean()


def panel(ax, d, site, col, window, first):
    """One backbone, one metric: six k levels over training steps."""
    for k in KS:
        g = d[(d.site == site) & (d.k == k)].sort_values("step")
        if not len(g):
            continue
        ls = "-"
        ax.plot(g.step, g[col], ls, color=COLOR[k], lw=0.8, alpha=0.22,
                zorder=2)
        ax.plot(g.step, smooth(g[col], window), ls, color=COLOR[k], lw=1.8,
                zorder=3)
    ax.axvline(COMMON, color=PALETTE["gray"], lw=1, ls=":", zorder=1)
    ax.set_xlim(0, 205)
    ax.set_xticks([0, 50, 100, 150, 200])
    grid(ax)
    if not first:
        ax.tick_params(labelleft=False)


def panel_af3(ax, d, site):
    """The sparse in-loop AF3 pass, as markers over the dense proxy."""
    for k in KS:
        g = d[(d.site == site) & (d.k == k)].dropna(
            subset=["frag_chimera_pass_af3"]).sort_values("step")
        if not len(g):
            continue
        ax.plot(g.step, g.frag_chimera_pass_af3, "o", ms=3.5,
                mfc=COLOR[k], mec="white", mew=0.7, zorder=4)


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#E5E7EB", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure(d, window):
    fig = plt.figure(figsize=(10.5, 6.0))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.08)
    for row, (col, lbl, _) in enumerate(METRICS):
        axes = []
        for i, site in enumerate(sorted(BB)):
            ax = fig.add_subplot(gs[row, i],
                                 sharey=axes[0] if axes else None)
            axes.append(ax)
            panel(ax, d, site, col, window, first=(i == 0))
            if col == "frag_chimera_pass":
                panel_af3(ax, d, site)
                ax.set_title(site, fontsize=10)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("training step")
        axes[0].set_ylabel(lbl)

    handles = [Line2D([], [], color=COLOR[k], lw=2, label=f"$k={k}$")
               for k in KS]
    handles.append(Line2D([], [], color=PALETTE["gray"], lw=0, marker="o",
                          ms=4, mec="white", label="in-loop AF3 pass"))
    handles.append(Line2D([], [], color=PALETTE["gray"], lw=1, ls=":",
                          label=f"step {COMMON}: common endpoint"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("fragment-count sweep, per-step training history "
                 "(all 18 cells matched at 200 steps)", fontsize=10)
    return fig


def verdict(final, last50, slope):
    """Has this cell stopped improving, in a way a bounded metric allows?

    A plain slope threshold is wrong against a ceiling: M0097/k=16 rose 3.4
    points over its final 50 steps and would be called "still climbing", but it
    did so from 96% to 99.5% and has under half a point left to gain. So a cell
    counts as converged either because it has run out of headroom, or because
    it has genuinely flattened. The 2% / 2e-4 cuts are a choice, not a
    measurement -- `headroom`, `delta50` and `slope` are printed so the call
    can be re-made.
    """
    if 1.0 - final < 0.02:
        return "ceiling"
    if abs(slope) < 2e-4 and abs(last50[-1] - last50[0]) < 0.02:
        return "flat"
    if abs(slope) < 6e-4:
        return "marginal"
    return "climbing"


def report(d, window):
    """Print the convergence read the curves are there to support."""
    rows = []
    for site in sorted(BB):
        for k in KS:
            g = d[(d.site == site) & (d.k == k)].sort_values("step")
            if len(g) < 60:
                continue
            s = smooth(g.frag_chimera_pass, window).to_numpy()
            step = g.step.to_numpy()
            last50 = s[-50:]
            slope = np.polyfit(step[-50:], last50, 1)[0]
            final = float(np.mean(s[-10:]))
            head = 1.0 - final
            rows.append(dict(
                site=site, k=k, steps=int(step[-1]) + 1,
                final=final, headroom=head,
                delta50=float(last50[-1] - last50[0]),
                slope=slope, verdict=verdict(final, last50, slope)))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\nIN-LOOP CHIMERA PASS, smoothed over {window} steps. "
          "`delta50`/`slope` are over each run's final 50 steps.\n")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nat own final step: "
          + ", ".join(f"{v}={n}" for v, n in t.verdict.value_counts().items())
          + "   (ceiling/flat = usable; marginal/climbing = not)")

    common = int(d.groupby(["site", "k"]).step.max().min())
    print(f"\nlatest step present in all {d.groupby(['site', 'k']).ngroups} "
          f"cells: {common}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=11,
                    help="centred rolling-mean window for the smoothed line")
    a = ap.parse_args()

    d = load()
    fig = figure(d, a.window)
    save(fig, "ame_fragk_curves")
    t = report(d, a.window)

    keep = ["step", "site", "k", "frag_chimera_pass", "frag_chimera_rmsd",
            "frag_chimera_pass_af3", "reward_std", "p_omit_mean"]
    out = HERE / "data" / "ame_fragk_curves.csv"
    d[keep].to_csv(out, index=False)
    t.to_csv(HERE / "data" / "ame_fragk_convergence.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
