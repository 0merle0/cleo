#!/usr/bin/env python3
"""Select the pilot backbones: 4 difficulty classes x 3 backbones.

Units note. In the deposited results each backbone has 40 rows = 8 LigandMPNN
sequences x 5 Chai model predictions. `n_pred_pass` therefore counts passing
*predictions*, not sequences -- at most 8 sequences can pass, and a single
good sequence contributes up to 5. Their per-backbone budget is 8 sequences.

Classes (RFdiffusion2 (fixed ligand) rows only)
----------------------------------------------
    easy    n_pred_pass >= 30       >=6 of the 8 sequences pass. LigandMPNN
                                    already solves it; question is whether we
                                    add diversity without losing fidelity.
    medium  10 <= n_pred_pass <= 29  ~2-6 of 8 pass. The most headroom.
    near    0 of 8 pass, best motif RMSD < 1.7 A
                                    Failed, but close. Cheapest rescue.
    hard    0 of 8 pass, best motif RMSD > 5.0 A
                                    Failed badly. If rescue works here it is
                                    search, not luck.

`near` and `hard` both require `no_clash`: a backbone that buries the ligand
cannot be repaired by any sequence, so including one would count a guaranteed
failure against us.

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


def select(bb, n_each=3):
    fail = bb[(bb.n_pred_pass == 0) & bb.clash_ok]
    classes = {
        "easy":   bb[bb.n_pred_pass >= 30],
        "medium": bb[bb.n_pred_pass.between(10, 29)],
        "near":   fail[fail.best_rmsd < NEAR_MISS_MAX],
        "hard":   fail[fail.best_rmsd > HARD_MISS_MIN],
    }
    out, used = [], set()
    for cls, pool in classes.items():
        # islands descending; then hardest-first within the class so we are not
        # quietly picking the friendliest member of each band.
        asc = [False, cls in ("easy", "medium")]
        sel = _take(pool, n_each, ["islands", "best_rmsd"], asc, exclude_sites=used)
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
    a = ap.parse_args()

    bb = per_backbone(a.csv, a.summary)
    sel = select(bb, a.n_each)
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
