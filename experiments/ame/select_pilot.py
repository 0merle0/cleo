#!/usr/bin/env python3
"""Select the pilot backbones: 4 difficulty classes x 3 backbones.

Units note. In the deposited results each backbone has 40 rows = 8 LigandMPNN
sequences x 5 Chai model predictions. `n_pred_pass` therefore counts passing
*predictions*, not sequences -- at most 8 sequences can pass, and a single
good sequence contributes up to 5. Their per-backbone budget is 8 sequences.

Classes (RFdiffusion2 (fixed ligand) rows only)
----------------------------------------------
    pass_   n_pred_pass >= 10       At least ~2 of the 8 sequences pass, i.e.
                                    LigandMPNN already finds something. One
                                    backbone each at 4, 5 and 6 residue islands,
                                    so motif complexity is the axis within the
                                    class. Note 6 is the ceiling: exactly one
                                    backbone in the entire benchmark has 6
                                    islands and reaches this pass rate, and
                                    none has 7.
    near    0 of 8 pass, best motif RMSD < 1.7 A
                                    Failed, but close. Cheapest rescue.
    hard    0 of 8 pass, best motif RMSD > 5.0 A
                                    Failed badly. If rescue works here it is
                                    search, not luck.

`near` and `hard` both require `no_clash`: a backbone that buries the ligand
cannot be repaired by any sequence, so including one would count a guaranteed
failure against us.

A selected backbone must also be clash-free under *our* implementation, not
only theirs. Our `ligand_dist_des_ncac_min` disagrees with the deposited value
by a median 0.38 A (max 1.76 A on this set) because they measure against the
packed/unidealized design and we hold the idealized backbone. A backbone that
we score as clashing can never report a composite pass in our own bookkeeping,
so it must be excluded regardless of what their column says. See
`verify_clash.py`.

Selection rules
---------------
1. **Distinct sites within a class.** Three backbones from one active site
   would measure that site, not the method.
2. **Prefer complex active sites.** Sorted by residue islands descending, so
   each class carries the hardest motifs it can. Note the ceiling this exposes:
   the `easy` class tops out at 4 islands because *no* 6-7 island backbone
   reaches n_pass >= 30 in the published data. That is the difficulty gradient
   (site success vs motif complexity, r = -0.805) showing up in the sampling
   frame, and it is a result, not a selection artifact.

Usage:
    uv run python experiments/ame/select_pilot.py
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = ("/tmp/claude-2230/-home-jgershon-git-cleo/"
               "2c774690-377e-44f9-a14d-3312238aa274/scratchpad/ext/zenodo/"
               "figure3/AME_main_benchmark_results/AME_benchmark_results_all.csv")
DEFAULT_SUMMARY = str(Path(DEFAULT_CSV).parent / "AME_benchmark_summary.csv")
SOURCE = "RFdiffusion2 (fixed ligand)"

NEAR_MISS_MAX = 1.7   # A; just past the 1.5 cutoff
HARD_MISS_MIN = 5.0   # A


def per_backbone(csv, summary_csv):
    d = pd.read_csv(csv)
    d = d[d.source == SOURCE]
    bb = d.groupby(["benchmark", "design_id"]).agg(
        n_pred_pass=("chai_motif_pass_and_no_clash", "sum"),
        n_pred=("chai_motif_pass_and_no_clash", "size"),
        clash_ok=("no_clash", "any"),
        best_rmsd=("backbone_aligned_allatom_rmsd_chai_motif", "min"),
    ).reset_index()

    s = pd.read_csv(summary_csv)
    s = s[s.source == SOURCE].set_index("benchmark")
    # Each backbone has 8 sequences x 5 Chai models. The deposit carries no
    # sequence index, so exact per-sequence counts are unrecoverable -- but each
    # sequence contributes at most 5 passing predictions, giving a hard lower
    # bound, and passes tend to be all-or-nothing within a sequence, making
    # n_pred_pass/5 a good sequence-equivalent.
    bb["seq_pass_min"] = -(-bb.n_pred_pass // 5)          # ceil, hard lower bound
    bb["seq_equiv"] = (bb.n_pred_pass / 5).round(1)       # of 8
    bb["islands"] = bb.benchmark.map(s["Number of Residue Islands"])
    bb["site_pass"] = bb.benchmark.map(s["chai_motif_pass_and_no_clash"])
    return bb


def _take(pool, n, sort_cols, ascending, exclude_sites=()):
    """Greedily take `n` rows, at most one per site, preferring sites unused so far."""
    pool = pool.sort_values(sort_cols, ascending=ascending)
    picked, seen = [], set()
    for prefer_new_site in (True, False):
        for _, r in pool.iterrows():
            if len(picked) == n:
                break
            if r.benchmark in seen:
                continue
            if prefer_new_site and r.benchmark in exclude_sites:
                continue
            picked.append(r)
            seen.add(r.benchmark)
        if len(picked) == n:
            break
    return pd.DataFrame(picked)


def _excluded(path):
    """design_ids that fail our own clash check; see clash_excluded.txt."""
    if not Path(path).exists():
        return set()
    out = set()
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.add(line)
    return out


def select(bb, n_each=3, exclude_file=None):
    if exclude_file:
        ex = _excluded(exclude_file)
        if ex:
            bb = bb[~bb.design_id.isin(ex)]
    fail = bb[(bb.n_pred_pass == 0) & bb.clash_ok]
    out, used = [], set()

    # passing: one backbone per island count, hardest complexity first, so the
    # class spans motif complexity rather than clustering at the easy end.
    p = bb[bb.n_pred_pass >= 10]
    picked = []
    for isl in (6, 5, 4):
        cand = p[(p.islands == isl) & ~p.benchmark.isin(used)]
        if cand.empty:
            cand = p[p.islands == isl]
        if cand.empty:
            continue
        r = cand.sort_values("n_pred_pass", ascending=False).iloc[0]
        picked.append(r)
        used.add(r.benchmark)
    out.append(pd.DataFrame(picked).assign(cls="pass"))

    for cls, pool in (("near", fail[fail.best_rmsd < NEAR_MISS_MAX]),
                      ("hard", fail[fail.best_rmsd > HARD_MISS_MIN])):
        sel = _take(pool, n_each, ["islands", "best_rmsd"], [False, False],
                    exclude_sites=used)
        sel = sel.assign(cls=cls)
        used |= set(sel.benchmark)
        out.append(sel)

    sel = pd.concat(out, ignore_index=True)
    sel["pdb"] = sel.design_id.str.replace("^rfflow_fixed-ligand_", "", regex=True) + ".pdb"
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--n-each", type=int, default=3)
    ap.add_argument("--out", default=HERE / "pilot_backbones.csv")
    ap.add_argument("--exclude", default=HERE / "clash_excluded.txt")
    a = ap.parse_args()

    bb = per_backbone(a.csv, a.summary)
    sel = select(bb, a.n_each, a.exclude)
    sel.to_csv(a.out, index=False)

    cols = ["cls", "benchmark", "islands", "site_pass", "n_pred_pass",
            "seq_equiv", "seq_pass_min", "best_rmsd", "pdb"]
    print(sel[cols].to_string(index=False))
    print(f"\n{len(sel)} backbones, {sel.benchmark.nunique()} distinct sites -> {a.out}")
    print("islands per class:",
          {c: sorted(g.islands.astype(int)) for c, g in sel.groupby("cls")})
    return sel


if __name__ == "__main__":
    main()
