#!/usr/bin/env python3
"""Check selected backbones are clash-free under *our* metric, not only theirs.

Why this exists. `ligand_dist_des_ncac_min` is a design-only quantity, so it
should be reproducible exactly -- and it is not. On the pilot set our values
differ from the deposit by a median 0.38 A, max 1.76 A. The cause is known:
their per-sequence metrics compare against the packed or unidealized design
(`pairs_to_compare` in `per_sequence_metrics.py` includes 'packed' and
'unideal'), while we hold the idealized `*-atomized-bb-True.pdb`.

That discrepancy is harmless for the reward, which uses motif RMSD only. It is
not harmless for selection: a backbone we score as clashing can never report a
composite pass in our own bookkeeping, so it would enter the experiment as a
guaranteed failure no matter how good the sequences are. One of the first 12
picks (M0092_1dli cond6_43, off by -1.76 A) failed exactly this way.

Usage:
    uv run python experiments/ame/verify_clash.py --targets experiments/ame/targets12
    uv run python experiments/ame/verify_clash.py --pdb <one.pdb>   # check a candidate
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
warnings.filterwarnings("ignore")

from cleo.design.utils.rfd2_benchmark import (  # noqa: E402
    _load, _protein_mask, _ligand_mask, NCAC, CLASH_CUTOFF,
)


def ligand_dist(pdb_path):
    a = _load(str(pdb_path))
    lig = a.coord[_ligand_mask(a)]
    ncac = a.coord[_protein_mask(a) & np.isin(a.atom_name, list(NCAC))]
    if not len(lig) or not len(ncac):
        return np.nan
    return float(np.linalg.norm(lig[:, None, :] - ncac[None, :, :], axis=-1).min())


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb", nargs="+", help="candidate backbone(s)")
    g.add_argument("--selection", help="pilot_backbones.csv")
    ap.add_argument("--targets", default=HERE / "targets12")
    a = ap.parse_args()

    if a.pdb:
        paths = [Path(p) for p in a.pdb]
        meta = {}
    else:
        sel = pd.read_csv(a.selection)
        paths = [Path(a.targets) / p for p in sel.pdb]
        meta = dict(zip(sel.pdb, sel.cls))

    rows = []
    for p in paths:
        d = ligand_dist(p)
        rows.append(dict(pdb=p.name, cls=meta.get(p.name, ""),
                         ours=round(d, 2), ok=bool(d > CLASH_CUTOFF)))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    bad = t[~t.ok]
    if len(bad):
        print(f"\nFAILED: {len(bad)} backbone(s) clash under our metric; replace them.")
        return 1
    print(f"\nAll {len(t)} clash-free under our metric (> {CLASH_CUTOFF} A).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
