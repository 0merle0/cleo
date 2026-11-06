#!/usr/bin/env python3
"""PCA of the sequence space each arm occupies on the AME pilot backbones.

One-hot encodes every sampled sequence (20 AAs x L positions), fits PCA on the
pooled set so all arms share one projection, and plots each arm's cloud with
its passing designs highlighted.

The point of the panel is the *relationship between the clouds*, not their
absolute coordinates: low-temperature LigandMPNN should appear as a single
tight knot, T=1.0 as a broad cloud with nothing passing in it, and CLEO as a
broad cloud that contains passing designs throughout. Fitting one shared PCA
is what makes that comparison legible -- per-arm PCA would put each cloud at
the origin and destroy the claim.

Two figures, because one projection cannot honestly carry both claims:

  ame_pca.svg        Peak-window policy vs. baselines. A like-for-like
                     comparison of three *policies*, all sampled at one point
                     in training.
  ame_pca_drift.svg  The full trajectory coloured by step. PC1 of the pooled
                     rollouts correlates with training step at r = +0.92..0.94
                     on all three backbones, so most of the spread along it is
                     drift over training, not diversity available at any one
                     moment. That spread is real diversity for a library built
                     from the whole run -- which is how we build ours -- but it
                     is not evidence that a single policy is diverse, and the
                     two claims must not be conflated. This panel exposes the
                     axis rather than hiding it.

    uv run python paper/figures/ame_pca.py

Sequences come from the training rollouts (already scored to compute the
reward, so free) plus the folded baseline sets.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Three panels are compressed into a 6.5in text width, so each lands at ~2.2in
# across. Type sized for the 15in canvas the figure is drawn on is illegible
# after that reduction; these are chosen to survive it.
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "figure.titlesize": 16, "axes.linewidth": 1.0,
})
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import PALETTE  # noqa: E402

AME = HERE.parents[1] / "experiments" / "ame"
RUNS, BEST = AME / "runs9rmsd", AME / "bestofn"
AAS = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AAS)}


def onehot(seqs):
    """-> (n, L*20) float array. Unknown residues become all-zero blocks."""
    L = len(seqs[0])
    X = np.zeros((len(seqs), L * len(AAS)), dtype=np.float32)
    for r, s in enumerate(seqs):
        for c, a in enumerate(s):
            if a in IDX:
                X[r, c * len(AAS) + IDX[a]] = 1.0
    return X


def load_trajectory(backbone):
    """Every rollout sampled during training for one backbone, with pass flag."""
    run = RUNS / backbone / backbone
    if not (run / "outputs").is_dir():
        run = RUNS / backbone
    rows = []
    for m in sorted((run / "outputs").glob("step_*/metrics.csv")):
        d = pd.read_csv(m)
        d["step"] = int(re.search(r"step_(\d+)", str(m)).group(1))
        rows.append(d)
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True)
    d["passing"] = d.ame_motif_pass & d.ame_no_clash
    return d[["sequence", "passing", "step"]].assign(arm="CLEO-GRPO")


def load_baseline(stem, label, backbone):
    f = BEST / f"{stem}_bo5_scored.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.backbone.str.contains(backbone.replace("run_", ""), regex=False)]
    if d.empty:
        return None
    return pd.DataFrame({"sequence": d.sequence, "passing": d.rfd2_any_pass.astype(bool),
                         "step": -1, "arm": label})


def peak_window(traj, width=10):
    """-> (lo, hi) inclusive: the `width`-step window with the most passing designs."""
    steps = sorted(traj.step.unique())
    best, key = (steps[-width], steps[-1]), -1
    for lo in steps[: max(1, len(steps) - width + 1)]:
        n = traj[(traj.step >= lo) & (traj.step < lo + width)].passing.sum()
        if n > key:
            best, key = (lo, lo + width - 1), n
    return best


def fit_projection(d):
    """Fit ONE PCA per backbone on every sequence that backbone contributes.

    Everything plotted for a backbone -- both figures, all arms, all training
    steps -- is transformed by this single projection, so points are comparable
    across panels and not only within one. Fitting per arm, or per figure, would
    place each cloud at its own origin and make the clouds impossible to compare,
    which is the whole content of the panel.

    The projection is deliberately NOT shared across backbones. Measured on the
    pooled rollouts, a global PCA puts 99 % of PC1 and 98 % of PC2 variance on
    backbone identity: the three backbones separate into distinct blobs and the
    arm differences we care about vanish inside them. Different backbones are
    different proteins, and the comparison of interest is always arm-vs-arm
    within one.
    """
    return PCA(n_components=2, random_state=0).fit(onehot(list(d.sequence)))


def _panel(ax, d, arms, title, Z):
    """Scatter one backbone's arms into a pre-fit shared projection."""
    d = d.assign(**dict(zip(("pc1", "pc2"), Z.transform(onehot(list(d.sequence))).T)))
    for arm, color in arms:
        s = d[d.arm == arm]
        if s.empty:
            continue
        f, p = s[~s.passing], s[s.passing]
        # Non-passing designs are context, not the claim: keep them faint so the
        # passing clouds -- the only ones a library would contain -- read as the
        # subject. Alpha scales up for the small baseline arms, which would
        # otherwise be invisible at the alpha that suits 3200 rollout points.
        ax.scatter(f.pc1, f.pc2, s=5, c=color, linewidths=0,
                   alpha=min(0.6, max(0.08, 40 / max(len(f), 1))),
                   rasterized=True)
        ax.scatter(p.pc1, p.pc2, s=60, c=color, alpha=0.95, linewidths=0.6,
                   edgecolors="white", label=f"{arm} ({len(p)} pass)")
    v = Z.explained_variance_ratio_
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({100 * v[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * v[1]:.1f}%)")
    ax.legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    backbones = sorted(p.name for p in RUNS.glob("run_*"))
    arms = [("LigandMPNN T=0.1", PALETTE["orange"]),
            ("LigandMPNN T=1.0", PALETTE["gray"]),
            ("CLEO-GRPO", PALETTE["blue"])]

    # Everything this backbone contributes, and the one projection fit on it.
    data, proj = {}, {}
    for bb in backbones:
        traj = load_trajectory(bb)
        if traj is None:
            continue
        parts = [traj,
                 load_baseline("repro_T0.1_seqs", "LigandMPNN T=0.1", bb),
                 load_baseline("repro_T1.0_seqs", "LigandMPNN T=1.0", bb)]
        d = pd.concat([p for p in parts if p is not None and len(p)],
                      ignore_index=True).drop_duplicates("sequence")
        data[bb] = d
        proj[bb] = fit_projection(d)

    # --- Figure 1: policy vs policy, GRPO restricted to its peak window ---
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 5.4),
                             squeeze=False)
    for ax, bb in zip(axes[0], data):
        d = data[bb]
        traj = d[d.arm == "CLEO-GRPO"]
        lo, hi = peak_window(traj)
        keep = (d.arm != "CLEO-GRPO") | ((d.step >= lo) & (d.step <= hi))
        _panel(ax, d[keep], arms, f"{bb.replace('run_', '')}\nGRPO steps {lo}-{hi}",
               proj[bb])
    fig.suptitle("Sequence space of a single policy "
                 "(faint = sampled, solid = passing)", fontsize=16)
    fig.tight_layout()
    save(fig, "ame_pca")

    # --- Figure 2: the drift axis, shown rather than hidden ---
    # Same projection as Figure 1, so the two panels can be read against each
    # other rather than each defining its own axes.
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 5.0),
                             squeeze=False)
    for ax, bb in zip(axes[0], data):
        d = data[bb]
        traj = d[d.arm == "CLEO-GRPO"]
        Z = proj[bb]
        P = Z.transform(onehot(list(traj.sequence)))
        r = np.corrcoef(P[:, 0], traj.step)[0, 1]
        sc = ax.scatter(P[:, 0], P[:, 1], c=traj.step, s=5, cmap="viridis",
                        alpha=0.5, linewidths=0, rasterized=True)
        m = traj.passing.values
        ax.scatter(P[m, 0], P[m, 1], s=30, facecolors="none",
                   edgecolors=PALETTE["red"], linewidths=0.8, label="passing")
        ax.set_title(f"{bb.replace('run_', '')}\nPC1 vs step: r = {r:+.2f}")
        ax.set_xlabel(f"PC1 ({100 * Z.explained_variance_ratio_[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * Z.explained_variance_ratio_[1]:.1f}%)")
        ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.colorbar(sc, ax=ax, label="training step", fraction=0.046)
    fig.suptitle("Most of PC1 is drift over training, not diversity available "
                 "at any one step", fontsize=16)
    fig.tight_layout()
    save(fig, "ame_pca_drift")


if __name__ == "__main__":
    main()
