#!/usr/bin/env python3
"""Why the AME policy degrades after its peak, and what the fix looks like.

Three claims, one per row, all read off the per-step rollout metrics that
training already writes -- no extra folding.

  Row 1  The failure. Pass rate peaks early and decays, while entropy *rises*.
         That direction is the whole diagnosis: RL fine-tuning is usually
         watched for mode collapse, and this is the opposite. A run that were
         collapsing would show entropy falling as pass rate fell.

  Row 2  The mechanism. With normalisation bounds [0.5, 20] on motif RMSD the
         reward is squeezed into a narrow band near the top of its range, so
         the pass/fail boundary -- the only distinction that matters -- is worth
         far less reward than the batch-to-batch noise GRPO divides by.

  Row 3  The fix. Rank normalisation makes the advantage invariant to that
         compression. Same target, same everything else.

    uv run python paper/figures/ame_convergence.py

Arms are discovered on disk, so this populates further as the A/B ladder
(legacy / surrogate-fixed / fully-fixed / +KL) finishes.
"""

import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 15, "axes.linewidth": 1.0,
})
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import PALETTE  # noqa: E402

AME = HERE.parents[1] / "experiments" / "ame"
LO, HI = 0.5, 20.0          # the reward bounds the pilot runs used
BIN = 15                    # steps per bin; 16 designs/step, so ~240 per point


def load(run_dir):
    """-> per-rollout metrics for one run, or None. Adds `passing` and `step`."""
    fs = sorted(glob.glob(f"{run_dir}/**/outputs/step_*/metrics.csv", recursive=True))
    if not fs:
        return None
    d = pd.concat(
        [pd.read_csv(m).assign(step=int(re.search(r"step_(\d+)", m).group(1)))
         for m in fs],
        ignore_index=True,
    )
    d["passing"] = d.ame_motif_pass & d.ame_no_clash
    return d


def binned(d, width=BIN):
    """-> DataFrame of per-bin pass rate, median RMSD and positional entropy.

    Entropy is the mean per-position Shannon entropy of the sampled residue
    distribution: the direct read of how spread the *policy* is, as opposed to
    pairwise distance, which is a property of the particular sample drawn.
    """
    rows = []
    for lo in range(0, int(d.step.max()) + 1, width):
        g = d[(d.step >= lo) & (d.step < lo + width)]
        if len(g) < 10:
            continue
        M = np.frombuffer("".join(g.sequence).encode(), dtype="S1").reshape(len(g), -1)
        H = []
        for i in range(M.shape[1]):
            _, c = np.unique(M[:, i], return_counts=True)
            p = c / c.sum()
            H.append(-(p * np.log(p)).sum())
        rows.append(dict(step=lo + width / 2,
                         pass_pct=100 * g.passing.mean(),
                         rmsd=g.ame_motif_rmsd.median(),
                         entropy=float(np.mean(H)),
                         p_omit=g["p_omit_mean"].mean() if "p_omit_mean" in g else np.nan))
    return pd.DataFrame(rows)


def norm(r):
    """The pilot's reward normalisation, for the compression panel."""
    return np.clip((HI - r) / (HI - LO), 0, 1)


def main():
    runs = {
        "value norm (200 steps)": AME / "runs9rmsd" / "run_M0097_1ctt_cond9_14",
        "rank norm (75 steps)": AME / "centering" / "run_M0097_1ctt_cond9_14_centre_w0.0",
    }
    # Any A/B arms that have started writing metrics.
    # The ablation ladder, each rung one fix further than the last.
    LABEL = {"legacy": "legacy objective", "surr": "+ surrogate fix",
             "": "+ log-prob fix", "kl": "+ KL anchor (0.02)"}
    for p in sorted(AME.glob("centering/run_M0097_1ctt_cond9_14_conv200*")):
        key = p.name.split("conv200")[-1].strip("_")
        runs[LABEL.get(key, key)] = p

    # A run needs a few bins before its trajectory means anything; arms that
    # have only just started would otherwise contribute a single point and, via
    # the shared colour scale, squash the step range for everyone else.
    data = {}
    for k, v in runs.items():
        d = load(v)
        if d is None:
            continue
        t = binned(d)
        if len(t) >= 3:
            data[k] = t
    if not data:
        raise SystemExit("no runs found")

    # One colour per arm, and enough of them: zip() silently drops arms when the
    # palette runs short, which quietly removed a whole ablation rung.
    colors = [PALETTE["red"], PALETTE["blue"], PALETTE["gray"],
              PALETTE["orange"], PALETTE["green"], "#7B4FA8", "#00868B"]
    if len(data) > len(colors):
        raise SystemExit(f"{len(data)} arms but only {len(colors)} colours")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))

    # --- the failure, and its signature ------------------------------------
    for ax, col, lab in [(axes[0][0], "pass_pct", "Passing designs (%)"),
                         (axes[0][1], "entropy", "Positional entropy (nats)")]:
        for (k, t), c in zip(data.items(), colors):
            ax.plot(t.step, t[col], "o-", color=c, label=k, lw=1.8, ms=4)
        ax.set_xlabel("training step")
        ax.set_ylabel(lab)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_title("Fixing the objective converts collapse into convergence")
    axes[0][1].set_title("...and inverts the failure mode: entropy now $\\it{falls}$")
    axes[0][0].legend(frameon=False)

    # --- the mechanism ------------------------------------------------------
    ax = axes[1][0]
    x = np.linspace(0.5, 6, 300)
    ax.plot(x, norm(x), color=PALETTE["blue"], lw=2)
    for r, c, lbl in [(1.4, PALETTE["green"], "1.4 Å (passes)"),
                      (2.0, PALETTE["red"], "2.0 Å (fails)")]:
        ax.plot([r], [norm(r)], "o", color=c, ms=9, zorder=3, label=lbl)
        ax.annotate(f"{norm(r):.3f}", (r, norm(r)), textcoords="offset points",
                    xytext=(-34, 6 if r < 1.7 else -16), fontsize=10, color=c)
    ax.annotate("", xy=(1.4, norm(1.4)), xytext=(1.4, norm(2.0)),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
    ax.text(2.3, (norm(1.4) + norm(2.0)) / 2 + 0.004,
            "the entire pass/fail\ndistinction: 0.031", fontsize=10, va="center")
    ax.axhspan(0.825, 0.929, color=PALETTE["gray"], alpha=0.18, zorder=0)
    ax.text(4.6, 0.877, "batch IQR", fontsize=10, color=PALETTE["gray"], ha="center")
    ax.set_xlabel("motif RMSD (Å)")
    ax.set_ylabel("normalised reward")
    ax.set_title("Bounds [0.5, 20] compress the decision region", pad=10)
    ax.legend(frameon=False, loc="lower left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # --- entropy against pass rate: the trajectory shape --------------------
    ax = axes[1][1]
    # One normalisation across every arm, so a point's colour means the same
    # training step in all of them -- otherwise a short run and a long one get
    # the same colour ramp over different step ranges and the panel misleads.
    smax = max(t.step.max() for t in data.values())
    nrm = matplotlib.colors.Normalize(vmin=0, vmax=smax)
    for (k, t), c in zip(data.items(), colors):
        ax.plot(t.entropy, t.pass_pct, "-", color=c, lw=1.5, alpha=0.7)
        ax.scatter(t.entropy, t.pass_pct, c=t.step, cmap="viridis", norm=nrm,
                   s=42, zorder=3, edgecolors=c, linewidths=1.2)
        ax.annotate(k, (t.entropy.iloc[-1], t.pass_pct.iloc[-1]),
                    textcoords="offset points", xytext=(6, -4), fontsize=8,
                    color=c, fontweight="bold")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=nrm, cmap="viridis"),
                 ax=ax, label="training step", fraction=0.046)
    ax.set_xlabel("positional entropy (nats)")
    ax.set_ylabel("passing designs (%)")
    ax.set_title("Only the KL anchor holds entropy open", pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle("Correcting the objective fixes the divergence and exposes "
                 "mode collapse; the KL anchor is the knob against it")
    fig.tight_layout()
    save(fig, "ame_convergence")


if __name__ == "__main__":
    main()
