#!/usr/bin/env python3
"""Novel mutations: what CLEO reaches that no temperature of the baseline does.

Coverage numbers say how much space a policy explores. They do not say whether
it is *new* space. This figure asks the set question instead: take the distinct
(position, residue) substitutions carried by passing designs, and compare the
baseline's set to CLEO's.

The baseline here is the temperature sweep pooled across T=0.1..1.0 -- the union
of everything LigandMPNN reaches at any setting, not one operating point. That
is the strongest form of the comparison: a substitution counts as novel only if
no temperature produced it.

Set size grows with the number of designs you draw, so panel B repeats the
count with CLEO subsampled to the baseline's exact passing-design count, over
`--draws` resamples. Without that control the whole figure would be a restated
pass rate.

    uv run python paper/figures/ame_novelty.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ame_diversity import ARMS, BACKBONES, baseline, cleo, passing, reference, subs
from figio import save
from palette import PALETTE

C_BASE, C_SHARED, C_CLEO = PALETTE["gray"], "#C7CBD1", PALETTE["blue"]
SHORT = {b: b.replace("run_", "").split("_cond")[0] for b in BACKBONES}


def collect(draws, seed):
    rng = np.random.default_rng(seed)
    pooled, per_arm = [], []
    for bb in BACKBONES:
        ref = reference(bb)
        b, c = baseline(bb), cleo(bb)
        B, C = subs(b, ref), subs(c, ref)
        nb, nc = len(passing(b)), len(passing(c))

        # Matched-count control: CLEO cut down to the baseline's design count.
        keep = passing(c)
        vals = []
        for _ in range(draws if len(keep) and nb else 0):
            sub = keep.sample(min(nb, len(keep)), random_state=int(rng.integers(1 << 31)))
            vals.append(len(subs(sub.assign(rfd2_any_pass=True), ref) - B))
        pooled.append(dict(backbone=bb, base_only=len(B - C), shared=len(B & C),
                           cleo_only=len(C - B), n_base=nb, n_cleo=nc,
                           matched_mean=np.mean(vals) if vals else np.nan,
                           matched_sd=np.std(vals) if vals else np.nan))

        for arm in ARMS:
            d = cleo(bb, arm)
            if d is None:
                continue
            S = subs(d, ref)
            per_arm.append(dict(backbone=bb, arm=arm, n=len(S), novel=len(S - B),
                                frac=100 * len(S - B) / len(S) if S else 0.0))
    return pd.DataFrame(pooled), pd.DataFrame(per_arm)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pooled, per_arm = collect(a.draws, a.seed)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 3.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # -- A: set containment, one stacked bar per backbone -------------------
    y = np.arange(len(pooled))[::-1]
    for i, (_, r) in enumerate(pooled.iterrows()):
        left = 0
        for val, col in ((r.base_only, C_BASE), (r.shared, C_SHARED), (r.cleo_only, C_CLEO)):
            if val:
                # 2px of surface between segments so the boundaries read as
                # boundaries rather than as a colour change inside one bar.
                axA.barh(y[i], val, left=left, height=0.55, color=col,
                         edgecolor="white", lw=1.6)
                if val > 0.06 * (r.base_only + r.shared + r.cleo_only):
                    axA.text(left + val / 2, y[i], f"{val:,}", ha="center", va="center",
                             fontsize=7.5, color="white" if col != C_SHARED else "#374151")
            left += val
        # The matched-count control, drawn where it belongs: inside the
        # CLEO-only segment, at the count CLEO reaches when cut to the
        # baseline's own number of passing designs. If this tick sat near the
        # segment's left edge, the novelty would be a sampling-depth artefact.
        if np.isfinite(r.matched_mean):
            xm = r.base_only + r.shared + r.matched_mean
            axA.plot([xm, xm], [y[i] - 0.30, y[i] + 0.30], color="white", lw=1.8, zorder=4)
            axA.annotate(f"matched-$n$: {r.matched_mean:,.0f}$\\pm${r.matched_sd:.0f}",
                         (xm, y[i] - 0.40), fontsize=7, color=C_CLEO, ha="center", va="top")
        else:
            axA.annotate("baseline 0/96 - nothing to match",
                         (r.cleo_only / 2, y[i] - 0.40), fontsize=7, color=C_BASE,
                         ha="center", va="top", style="italic")
    axA.set_yticks(y, [SHORT[b] for b in pooled.backbone], fontsize=8.5)
    axA.set_xlabel("distinct (position, residue) substitutions among passing designs",
                   fontsize=8.5)
    axA.tick_params(labelsize=8)
    axA.grid(axis="x", alpha=0.25, lw=0.6)
    axA.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)
    for lab, col, x in (("baseline only", C_BASE, 0.02), ("shared", C_SHARED, 0.26),
                        ("CLEO only", C_CLEO, 0.44)):
        axA.text(x, 1.06, lab, transform=axA.transAxes, fontsize=8, color=col,
                 fontweight="bold")

    # -- B: novel fraction per selection arm --------------------------------
    w = 0.8 / len(ARMS)
    x = np.arange(len(BACKBONES))
    for j, arm in enumerate(ARMS):
        d = per_arm[per_arm.arm == arm].set_index("backbone").reindex(BACKBONES)
        axB.bar(x + j * w - 0.4 + w / 2, d.frac.fillna(0), width=w * 0.86,
                color=C_CLEO, alpha=0.45 + 0.18 * j, edgecolor="white", lw=0.8,
                label=arm.replace("logprob_", ""))
    axB.set_xticks(x, [SHORT[b] for b in BACKBONES], fontsize=8.5)
    axB.set_ylabel("% of arm's substitutions\nunseen at any baseline T", fontsize=8.5)
    axB.tick_params(labelsize=8)
    axB.grid(axis="y", alpha=0.25, lw=0.6)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    axB.set_ylim(0, 118)
    for j, arm in enumerate(ARMS):
        d = per_arm[per_arm.arm == arm].set_index("backbone").reindex(BACKBONES)
        for i, bb in enumerate(BACKBONES):
            if not d.n.get(bb):   # no passing designs -> no substitutions to classify
                axB.annotate("no\npass", (i + j * w - 0.4 + w / 2, 1.5), fontsize=6.5,
                             color=C_BASE, ha="center", va="bottom", style="italic")

    axA.margins(y=0.14)
    fig.tight_layout()
    save(fig, "ame_novelty")

    pd.set_option("display.width", 200, "display.float_format", "{:.4g}".format)
    print(pooled.to_string(index=False))
    print()
    print(per_arm.to_string(index=False))


if __name__ == "__main__":
    main()
