#!/usr/bin/env python3
"""E19: the batch-size sweep, as an evaluation-efficiency frontier.

    uv run python paper/figures/ame_bsweep.py

Four panels, written to figures/svg/:

  ame_bsweep_frontier    pass vs FOLDS, four batch sizes, M0097   (the headline)
  ame_bsweep_backbones   b16 vs b32 on all three backbones
  ame_bsweep_thresholds  folds to first reach each level, b32 -> b16
  ame_bsweep_transfer    held-out AF3 against the training level it was reached at

Three decisions behind these, each of which changes what the figure says.

**The x axis is folds, not steps.** folds/step = rotations * batch *
num_diffusion_samples = 4 * B * 5, i.e. 160 at b8 and 1280 at b64. Plotted
against steps, the four arms sit on budgets differing 8-fold and the figure
shows the budget, not the batch size. Folds are also the quantity that costs
money, which is why the experiment was framed in them.

**Chronology is reconstructed, not assumed.** A resume restarts the training
step counter at 0 into the same outputs/ tree, so a resumed run's
train_metrics.csv is several runs concatenated and `(step+1)*folds_per_step` is
not elapsed folds. Read naively, M0255's b32 arm first reaches 0.7 at 78,080
folds and 0.9 at 36,480 -- impossible on a monotone axis, and the tell. The
loader splits at every step-counter reset, keeps the original segment, and
asserts `runtime` is monotone inside it. See experiments/ame/bsweep_frontier.py.

**Batch size is ORDERED, so it gets a sequential ramp, not four hues.** One hue
light-to-dark over b8..b64 encodes the magnitude the reader is being asked to
compare, and it avoids spending a green/red categorical pair on it. Every line
is also directly labelled at its end, so series identity never rests on colour
alone. The two-series backbone panels use the house blue/orange pair.

Partial arms are plotted as far as they have run; nothing is extrapolated.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

HERE = Path(__file__).resolve().parent
AME = HERE.parents[1] / "experiments" / "ame"
sys.path.insert(0, str(HERE))
from figio import save                                        # noqa: E402
from palette import PALETTE                                    # noqa: E402

ROT, NS = 4, 5
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
PASS_COL = "frag_chimera_pass"

# Smoothing is matched in FOLDS, not steps. A 3-step window is the minimum for
# oracle balance (round-robin over 3 predictors), but 3 steps is 480 folds at
# b8 and 1920 at b32, so a fixed step window smooths the small-batch arms four
# times less. That is not cosmetic: a noisier curve touches a threshold early by
# chance, and measured against a 3-step window every b8 and b16 crossing landed
# EARLIER than it does here (b8 to 0.7: 23,520 folds vs 28,000; b16 to 0.3:
# 12,800 vs 17,280), while b32 and b64 did not move at all. The step window was
# systematically flattering the arms the conclusion favours.
#
# So: pick the window from a target fold width, rounded to a multiple of 3.
# b64 cannot go below 3 steps without breaking oracle balance, so its window is
# 3840 folds -- the one arm that is smoothed more than the others, which makes
# it look smoother than it is.
SMOOTH_FOLDS = 1920          # = 3 steps at batch 32


def smooth_window(batch):
    """-> window in steps: a multiple of 3 spanning about SMOOTH_FOLDS folds."""
    fps = ROT * batch * NS
    return max(3, 3 * round(SMOOTH_FOLDS / (fps * 3)))


# (family, stem, batch, index of the segment that is the original run)
ARMS = {
    ("M0097", 8):  ("fragbsweep", "run_M0097_1ctt_cond9_14_bs8", 0),
    ("M0097", 16): ("fragbsweep", "run_M0097_1ctt_cond9_14_bs16", 0),
    ("M0097", 32): ("fragb32", "run_M0097_1ctt_cond9_14_fragb32", 0),
    ("M0097", 64): ("fragbsweep", "run_M0097_1ctt_cond9_14_bs64", 0),
    ("M0255", 16): ("fragb16", "run_M0255_1mg5_cond17_66_bs16", 0),
    ("M0255", 32): ("fragb32", "run_M0255_1mg5_cond17_66_fragb32", 0),
    ("M0315", 16): ("fragb16", "run_M0315_1ey3_cond18_25_bs16", 0),
    ("M0315", 32): ("fragb32", "run_M0315_1ey3_cond18_25_fragb32", 2),
}
BACKBONES = ["M0097", "M0255", "M0315"]

# Sequential ramp over an ordered variable (batch size), light -> dark.
# A sequential ramp, not four hues: batch size is ordered, so magnitude is the
# thing to encode. Steps are spaced so each is clearly darker than the last and
# the lightest still carries on white (the first ramp put b8 at #A8C4E0, which
# washed out against the grid). Every line is directly labelled as well, so
# identity never rests on colour alone.
RAMP = {8: "#8FB4D9", 16: "#4F82B4", 32: "#2E5C8A", 64: "#13283C"}
MARKER = {8: "o", 16: "s", 32: "^", 64: "D"}
GRID = dict(color="#E5E7EB", lw=0.6, zorder=0)


def kfmt(v, _pos=None):
    """40000 -> '40k'. Five-digit fold counts on every tick are unreadable."""
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def segments(m):
    cuts, prev = [0], None
    for i, v in enumerate(m.step):
        if prev is not None and v <= prev:
            cuts.append(i)
        prev = v
    cuts.append(len(m))
    return [m.iloc[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]


def curve(bb, batch):
    """-> DataFrame with folds and a 3-step oracle-balanced pass, or None."""
    spec = ARMS.get((bb, batch))
    if spec is None:
        return None
    family, stem, seg_idx = spec
    f = AME / family / stem / stem / f"{stem}_train_metrics.csv"
    if not f.exists():
        print(f"  (no metrics yet for {bb} b{batch})")
        return None
    segs = segments(pd.read_csv(f))
    if seg_idx >= len(segs):
        print(f"  ({bb} b{batch}: segment {seg_idx} absent; {len(segs)} present)")
        return None
    g = segs[seg_idx].reset_index(drop=True).copy()
    assert g.runtime.is_monotonic_increasing, \
        f"{stem}: segment {seg_idx} runtime not monotone -- segmentation is wrong"
    g["folds"] = (g.step + 1) * ROT * batch * NS
    # The oracle pool is round-robin over 3 predictors that disagree by up to
    # 26 points on identical designs, so one step is whichever was drawn. Any
    # multiple of 3 consecutive steps holds each equally often.
    w = smooth_window(batch)
    g["p3"] = g[PASS_COL].rolling(w, min_periods=w).mean()
    return g.dropna(subset=["p3"])


def first_reach(g, thr):
    h = g[g.p3 >= thr]
    return float(h.folds.iloc[0]) if len(h) else None


def style(ax, xlabel="evaluations (folds)", ylabel="chimera pass rate"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
    ax.tick_params(colors="#4B5563", labelsize=8)
    if xlabel:
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.xaxis.label.set_color("#374151")
    ax.yaxis.label.set_color("#374151")


def fig_frontier():
    """M0097: all four batch sizes on one fold axis."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    drawn = []
    for b in (8, 16, 32, 64):
        g = curve("M0097", b)
        if g is None or g.empty:
            continue
        ax.plot(g.folds, g.p3, color=RAMP[b], lw=2.0, zorder=3,
                marker=MARKER[b], markersize=0, label=f"batch {b}")
        # Direct label at the line end, so identity is not colour-alone.
        ax.annotate(f"b{b}", (g.folds.iloc[-1], g.p3.iloc[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=RAMP[b], fontsize=8, va="center", fontweight="bold")
        drawn.append(b)
    for thr in THRESHOLDS:
        ax.axhline(thr, color="#D1D5DB", lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    style(ax)
    # NOT "the only arm to reach 0.9": batch 8 crossed at 46,720 folds in its
    # last 50 steps, after reading as stalled at 0.72 for the 10k folds before
    # that. The earlier draft of this title was written off the incomplete run.
    ax.set_title("Batch 16 reaches 0.9 on 40k evaluations; batch 32 needs 55k "
                 "and batch 64 never gets there",
                 fontsize=10, color="#111827", loc="left", pad=22)
    ax.annotate("M0097. Batch 8 is cheapest up to 0.7 but needs 47k to reach 0.9.",
                (0, 1), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", fontsize=7.5, color="#6B7280")
    if len(drawn) >= 2:
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor="#374151")
    save(fig, "ame_bsweep_frontier")
    plt.close(fig)


def fig_backbones():
    """b16 against b32, one panel per backbone. Shared axes, so the panels are
    comparable to each other and not just internally."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=True, sharey=True)
    colors = {16: RAMP[16], 32: RAMP[32]}  # same ramp as the frontier figure: batch
    # size is ordered, and 16 must not be blue in one figure and orange in the next
    for ax, bb in zip(axes, BACKBONES):
        for b in (32, 16):
            g = curve(bb, b)
            if g is None or g.empty:
                continue
            ax.plot(g.folds, g.p3, color=colors[b], lw=2.0, zorder=3,
                    label=f"batch {b}")
            # Both arms plateau near 1.0, so a label centred on the line end
            # lands on the line. Push b16 up and b32 down to separate them.
            ax.annotate(f"b{b}", (g.folds.iloc[-1], g.p3.iloc[-1]),
                        xytext=(5, 7 if b == 16 else -7),
                        textcoords="offset points",
                        color=colors[b], fontsize=8, va="center",
                        fontweight="bold")
        for thr in (0.5, 0.9):
            ax.axhline(thr, color="#D1D5DB", lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.set_title(bb, fontsize=9, color="#111827", loc="left")
        style(ax, ylabel="chimera pass rate" if bb == BACKBONES[0] else "")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xlim(left=0)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right",
                   labelcolor="#374151")
    # Not "every level": M0097's 0.3 crossing is 0.78x, i.e. b16 is slower
    # there. The title says what the three panels show.
    fig.suptitle("Batch 16 converges on fewer evaluations than batch 32 on all "
                 "three backbones, by 1.2--2.0\u00d7 from 0.5 upward",
                 fontsize=10, color="#111827", x=0.005, ha="left", y=1.04)
    save(fig, "ame_bsweep_backbones")
    plt.close(fig)


def fig_thresholds():
    """Slope chart: folds to first reach each level, b32 -> b16.

    A slope per threshold per backbone shows both the direction and the size of
    the saving, which a grouped bar chart of the same numbers hides. An arm that
    never reached a level is omitted rather than drawn at the axis limit --
    plotting a censored value as a number is the thing that makes 'never' look
    like 'slow'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    for ax, bb in zip(axes, BACKBONES):
        g16, g32 = curve(bb, 16), curve(bb, 32)
        if g16 is None or g32 is None:
            continue
        censored = []
        for thr in THRESHOLDS:
            a, c = first_reach(g32, thr), first_reach(g16, thr)
            if a is None or c is None:
                censored.append(thr)
                continue
            # Ties are grey. 1.00x is "no difference", and painting it red
            # read as a regression in the first draft.
            ratio = a / c
            col = (PALETTE["green"] if ratio > 1.02
                   else PALETTE["red"] if ratio < 0.98 else PALETTE["gray"])
            ax.plot([0, 1], [a, c], color=col, lw=1.8, zorder=3,
                    marker="o", markersize=5)
            ax.annotate(f"{thr:g}  {a / c:.2f}×",
                        (1, c), xytext=(7, 0), textcoords="offset points",
                        fontsize=8, va="center", color="#374151")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["batch 32", "batch 16"], fontsize=8)
        ax.set_xlim(-0.25, 1.75)
        ax.set_title(bb, fontsize=9, color="#111827", loc="left")
        if censored:
            ax.annotate("not reached by one arm: "
                        + ", ".join(f"{t:g}" for t in censored),
                        (0, 0), xycoords="axes fraction", xytext=(0, -26),
                        textcoords="offset points", fontsize=7, color="#6B7280")
        style(ax, xlabel="",
              ylabel="folds to first reach" if bb == BACKBONES[0] else "")
        ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    fig.suptitle("Evaluations saved by halving the batch, per convergence level",
                 fontsize=10, color="#111827", x=0.005, ha="left", y=1.04)
    save(fig, "ame_bsweep_thresholds")
    plt.close(fig)


def fig_transfer():
    """Held-out AF3 pass against the oracle-balanced training level it was
    reached at. The diagonal is 'training tells the truth'; distance below it is
    the transfer gap. Both axes are the same quantity on the same scale, so this
    is one scale, not two."""
    probes = []
    for family, pat in (("fragbsweep", "*_summary.csv"), ("fragb32", "*_summary.csv")):
        for f in sorted((AME / family / "offline_probe").glob(pat)):
            d = pd.read_csv(f)
            if "af3_pass" not in d:
                continue
            d = d.dropna(subset=["af3_pass", "train_pass"])
            if d.empty:
                continue
            d["family"] = family
            d["src"] = f.stem
            probes.append(d)
    if not probes:
        print("  (no probe summaries yet; skipping transfer panel)")
        return
    p = pd.concat(probes, ignore_index=True)

    def batch_of(run):
        for (bb, b), (_, stem, _) in ARMS.items():
            if stem == run:
                return b
        return None

    p["batch"] = p.run.map(batch_of)
    p = p.dropna(subset=["batch"])
    if p.empty:
        print("  (probe summaries matched no known arm; skipping)")
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate("training = held-out", (0.17, 0.17), rotation=45, fontsize=7,
                color="#6B7280", ha="center", va="bottom")
    p["bb"] = p.run.str.extract(r"(M\d+)")
    for b in sorted(p.batch.unique()):
        q = p[p.batch == b]
        ax.scatter(q.train_pass, q.af3_pass, s=46, color=RAMP[int(b)],
                   marker=MARKER[int(b)], zorder=3, linewidths=0.8,
                   edgecolors="white", label=f"batch {int(b)}")
    # Backbone as a text tag beside each point, not a second colour: colour is
    # already spent on batch, and a second ramp on the same marks would make
    # neither readable.
    for _, r in p.iterrows():
        ax.annotate(str(r.bb).replace("M0", ""), (r.train_pass, r.af3_pass),
                    xytext=(6, -3), textcoords="offset points", fontsize=6,
                    color="#9CA3AF")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    style(ax, xlabel="training pass (oracle-balanced)",
          ylabel="held-out AF3 pass")
    ax.set_title("Points below the diagonal are where training overstates",
                 fontsize=10, color="#111827", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor="#374151")
    save(fig, "ame_bsweep_transfer")
    plt.close(fig)


def main():
    fig_frontier()
    fig_backbones()
    fig_thresholds()
    fig_transfer()


if __name__ == "__main__":
    main()
