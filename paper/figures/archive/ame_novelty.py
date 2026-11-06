#!/usr/bin/env python3
"""Novel mutations: what CLEO reaches that no temperature of the baseline does.

Coverage says how much space a policy explores. It does not say whether the
space is new. This asks the set question instead: over passing designs, compare
the distinct (position, residue) substitutions the baseline carries against the
ones CLEO carries.

The baseline is the temperature sweep pooled over T=0.1..1.0 -- the union of
everything LigandMPNN reaches at any setting, not one operating point. A
substitution counts as novel only if no temperature produced it.

CLEO is the `random` arm: sequences as the policy emits them, no selection.

**Every bar is computed at matched depth.** Set size grows with the number of
designs drawn, and the two sides are not drawn equally deep: the baseline pools
six temperatures, so it folds 576 designs to CLEO's 96 and arrives with 238
passing designs against CLEO's 86 on M0097. Comparing the raw sets would hand
the baseline a 2.8x advantage that comes from folding budget, not from the
policy. So both sides are subsampled to n = min(n_base, n_cleo) passing designs
and the three counts averaged over `--draws` resamples.

That correction is not cosmetic. On M0097 it moves CLEO-only from 470 to the
matched number below, and moves baseline-only far more.

    uv run python paper/figures/ame_novelty.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ame_diversity import BACKBONES, baseline, cleo, passing, reference, subs
from figio import save
from palette import PALETTE

C_BASE, C_SHARED, C_CLEO = PALETTE["gray"], "#C7CBD1", PALETTE["blue"]
SHORT = {b: b.replace("run_", "").split("_cond")[0] for b in BACKBONES}


def collect(arm, draws, seed):
    rng = np.random.default_rng(seed)
    out = []
    for bb in BACKBONES:
        ref = reference(bb)
        kb, kc = passing(baseline(bb)), passing(cleo(bb, arm))
        n = min(len(kb), len(kc))

        if n == 0:
            # Nothing to match against: report CLEO's full set and say so.
            C = subs(kc, ref)
            out.append(dict(backbone=bb, n_base=len(kb), n_cleo=len(kc), depth=np.nan,
                            base_only=0.0, shared=0.0, cleo_only=float(len(C)),
                            cleo_sd=np.nan))
            continue

        bo, sh, co = [], [], []
        for _ in range(draws):
            B = subs(kb.sample(n, random_state=int(rng.integers(1 << 31))), ref)
            C = subs(kc.sample(n, random_state=int(rng.integers(1 << 31))), ref)
            bo.append(len(B - C)); sh.append(len(B & C)); co.append(len(C - B))
        out.append(dict(backbone=bb, n_base=len(kb), n_cleo=len(kc), depth=n,
                        base_only=np.mean(bo), shared=np.mean(sh),
                        cleo_only=np.mean(co), cleo_sd=np.std(co)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="random", help="CLEO selection arm to plot")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = collect(a.arm, a.draws, a.seed)

    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    y = np.arange(len(df))[::-1]

    for i, (_, r) in enumerate(df.iterrows()):
        left = 0
        total = r.base_only + r.shared + r.cleo_only
        for val, col in ((r.base_only, C_BASE), (r.shared, C_SHARED), (r.cleo_only, C_CLEO)):
            if val:
                # A hairline of surface between segments, so a boundary reads as
                # a boundary and not as a colour change inside one bar.
                ax.barh(y[i], val, left=left, height=0.5, color=col,
                        edgecolor="white", lw=1.6)
                if val > 0.07 * total:
                    ax.text(left + val / 2, y[i], f"{val:,.0f}", ha="center", va="center",
                            fontsize=8, color="#374151" if col is C_SHARED else "white")
            left += val

        if np.isfinite(r.depth):
            note = f"{r.depth:.0f} passing designs per side"
            col = C_BASE
        else:
            note = f"LigandMPNN passes nothing here; our {r.n_cleo:.0f} designs, unmatched"
            col = C_CLEO
        ax.annotate(note, (0, y[i] - 0.33), fontsize=7, color=col,
                    ha="left", va="top", style="italic")

    ax.set_yticks(y, [SHORT[b] for b in df.backbone], fontsize=9)
    ax.set_xlabel("distinct (position, residue) substitutions, at matched design count",
                  fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for lab, col, x in (("LigandMPNN only", C_BASE, 0.0),
                        ("shared", C_SHARED, 0.24),
                        ("this paper only", C_CLEO, 0.40)):
        ax.text(x, 1.05, lab, transform=ax.transAxes, fontsize=8.5,
                color="#6B7280" if col is C_SHARED else col, fontweight="bold")

    fig.tight_layout()
    save(fig, "ame_novelty")
    pd.set_option("display.width", 200, "display.float_format", "{:.4g}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
