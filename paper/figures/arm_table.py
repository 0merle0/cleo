#!/usr/bin/env python3
"""Every arm that has a library evaluation, in one table.

The question this answers is "what has actually been run, and under what
conditions", which was previously only answerable by reading three markdown
files and several config trees -- and getting it wrong is easy: the same
nominal method appears under half a dozen tree names with different oracles,
batch sizes and training budgets.

The number reported is the RECOMBINANT PASS RATE: fragments from the arm's
own parents, shuffled, reassembled, folded best-of-5 through AF3 and scored
against the motif criterion. AF3 is held out of every training loop here, so
it is comparable across arms that trained differently. Retention is
deliberately NOT reported -- it is a ratio with an unstable denominator and
swings 0.767 to 1.611 across replicates of one config.

    uv run python paper/figures/arm_table.py            # grouped summary
    uv run python paper/figures/arm_table.py --full     # every cell
    uv run python paper/figures/arm_table.py --md       # markdown
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AME = HERE.parents[1] / "experiments" / "ame"

# Trees whose arm is not inferable from the config alone.
BASELINE_SUFFIX = "baseline"


def _config_for(tree, backbone_run):
    """-> the training config for this run, or None for sampled baselines."""
    hits = sorted(glob.glob(str(AME / tree / f"{backbone_run}*" / "*" / "*_config.yaml")))
    return Path(hits[0]) if hits else None


def _params(cfg):
    """-> dict of the handful of fields that make arms comparable or not."""
    if cfg is None:
        return dict(objective="sampled", oracle="-", batch=None, steps=None,
                    updates=None, seed=None)
    txt = cfg.read_text()

    def grab(pat, cast=str):
        m = re.search(pat, txt, re.M)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except ValueError:
            return None

    oracles = set(re.findall(r"(multi_oracle|af3|rf3|boltz|protenix)_from_df", txt))
    if "multi_oracle" in oracles:
        oracle = "pool"
    else:
        oracle = "+".join(sorted(oracles - {"multi_oracle"})) or "?"
    return dict(
        objective="shuffle" if "FragmentReward" in txt else "plain",
        oracle=oracle,
        batch=grab(r"^batch_size:\s*(\d+)", int),
        steps=grab(r"^N_steps:\s*(\d+)", int),
        updates=grab(r"^N_updates:\s*(\d+)", int),
        seed=grab(r"^seed:\s*(\d+)", int),
    )


def scan():
    rows = []
    for f in sorted(glob.glob(str(AME / "*" / "recomb" / "*_scored.csv"))):
        if "samples" in f:
            continue
        p = Path(f)
        tree = p.parents[1].name
        stem = p.name.replace("_scored.csv", "")
        m = re.match(r"(run_(M\d+)_\w+?_cond\d+_\d+)_(.*)$", stem)
        if not m:
            continue
        run, bb, arm_sfx = m.group(1), m.group(2), m.group(3)
        d = pd.read_csv(f)
        if "rfd2_any_pass" not in d.columns:
            continue
        par = _params(None if arm_sfx == BASELINE_SUFFIX else _config_for(tree, run))
        rows.append(dict(tree=tree, arm=arm_sfx, backbone=bb, n=len(d),
                         recomb_pass=round(100 * d.rfd2_any_pass.mean(), 1),
                         **par))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="every cell, not a summary")
    ap.add_argument("--md", action="store_true", help="markdown table")
    a = ap.parse_args()

    d = scan()
    if not len(d):
        sys.exit("no scored recombinant libraries found under experiments/ame/*/recomb/")

    pd.set_option("display.width", 220)
    if a.full:
        out = d.sort_values(["objective", "oracle", "tree", "backbone"])
        cols = ["tree", "arm", "backbone", "objective", "oracle", "batch",
                "steps", "updates", "seed", "n", "recomb_pass"]
        print(out[cols].to_markdown(index=False) if a.md
              else out[cols].to_string(index=False))
        return

    # Summary: one row per (tree, config), backbones as columns.
    # groupby, not pivot_table(dropna=False): the latter builds the full
    # cartesian product of every index level and fills it with NaN rows.
    keys = ["objective", "oracle", "tree", "batch", "steps", "updates", "seed"]
    d2 = d.copy()
    for k in keys:
        d2[k] = d2[k].fillna("-")
    piv = (d2.groupby(keys + ["backbone"], dropna=False)["recomb_pass"]
             .mean().unstack("backbone"))
    bbs = list(piv.columns)
    piv["mean"] = piv[bbs].mean(axis=1).round(1)
    piv["n_bb"] = piv[bbs].notna().sum(axis=1)
    piv = piv.dropna(subset=["mean"])
    # Only arms measured on every backbone are directly comparable; the rest
    # are shown but must not be averaged against them.
    piv = piv.sort_values(["objective", "oracle", "mean"], ascending=[True, True, False])
    piv[bbs] = piv[bbs].round(1)
    print("RECOMBINANT PASS RATE (%), AF3-scored, held out of all training\n")
    print(piv.to_markdown() if a.md else piv.to_string())
    print("\nmean/n_bb cover only the backbones that arm was evaluated on; "
          "an arm run on 1 backbone is not comparable with one run on 3.")

    # The only honest cross-arm view: arms measured on ALL of the backbones
    # that every arm shares. Averaging an 11-backbone arm against a
    # 2-backbone one compares different problems, and the backbones differ
    # enormously in difficulty (M0255 runs ~30 points below M0315).
    common = ["M0097", "M0255", "M0315"]
    have = [c for c in common if c in d.backbone.unique()]
    sub = d[d.backbone.isin(have)]
    full = (sub.groupby(["objective", "oracle", "tree"])["backbone"]
               .nunique().eq(len(have)))
    ok = full[full].index
    g = (sub.set_index(["objective", "oracle", "tree"]).loc[ok]
            .groupby(["objective", "oracle", "tree"])["recomb_pass"]
            .mean().round(1).sort_values(ascending=False))
    print(f"\n\nARMS EVALUATED ON ALL OF {have} -- mean pass rate\n")
    print(g.to_markdown() if a.md else g.to_string())
    print("\nSpread WITHIN a condition is the thing to notice before reading "
          "any difference BETWEEN conditions.")
    for cond, lab in ((("plain", "rf3"), "plain RL, rf3"),
                      (("plain", "pool"), "plain RL, pool"),
                      (("shuffle", "pool"), "shuffle, pool")):
        v = g.loc[cond] if cond in g.index.droplevel(2).unique() else None
        if v is None or not len(v):
            continue
        print(f"  {lab:16s} n={len(v)}  range {v.min():.1f}-{v.max():.1f}  "
              f"median {v.median():.1f}")


if __name__ == "__main__":
    main()
