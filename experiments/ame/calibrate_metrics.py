#!/usr/bin/env python3
"""Calibrate our benchmark-metric implementation against RFdiffusion2's published values.

`ligand_dist_des_ncac_min` is the one benchmark quantity that depends only on the
design -- no structure predictor is involved -- so it is the only metric we can
check against ground truth without running an oracle. If our geometry and
parsing layer is right, we should reproduce it exactly for every backbone.

STATUS 2026-08-04: WE DO NOT. Median absolute error 0.42 A, max 1.68 A,
boolean no_clash agreement 38/40. Until this passes, the motif-RMSD reward
(which shares the same parsing and superposition code) is not trustworthy.

Ruled out so far, none of which reproduce the published number:
  - ligand atom subset: all / DAD only / MG only / partially_fixed_ligand
    members / non-members
  - excluding motif residues from the N/CA/C set
  - CA-only instead of N/CA/C
  - N/CA/C/O instead of N/CA/C

Leading remaining hypothesis: the published metric is computed on a different
PDB than the `*-atomized-bb-True.pdb` shipped in rfd2_ame_41_backbones.tar.gz.
`per_sequence_metrics.py` distinguishes `analyze.get_design_pdb(row)` (the
MPNN-packed design) from `analyze.get_diffusion_pdb(row)` (raw diffusion
output); we may have neither. Those variants most likely live in the 52 GB
/net/lab/pub/rfdiffusion2/2024-12-16_08-11-34_enzyme_bench_n41.tar.

Usage:
    uv run python experiments/ame/calibrate_metrics.py [--csv PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleo.design.utils.rfd2_benchmark import (  # noqa: E402
    _load, _protein_mask, _ligand_mask, NCAC, CLASH_CUTOFF,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = ("/tmp/claude-2230/-home-jgershon-git-cleo/"
               "2c774690-377e-44f9-a14d-3312238aa274/scratchpad/ext/zenodo/"
               "figure3/AME_main_benchmark_results/AME_benchmark_results_all.csv")


def ligand_dist_ncac_min(pdb_path):
    des = _load(pdb_path)
    lig = des.coord[_ligand_mask(des)]
    ncac = des.coord[_protein_mask(des) & np.isin(des.atom_name, list(NCAC))]
    if not len(lig) or not len(ncac):
        return np.nan
    return float(np.linalg.norm(lig[:, None, :] - ncac[None, :, :], axis=-1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="AME_benchmark_results_all.csv")
    ap.add_argument("--targets", default=HERE / "pilot_targets.csv")
    ap.add_argument("--target-dir", default=HERE / "targets")
    a = ap.parse_args()

    truth = pd.read_csv(a.csv)
    truth = truth[truth.source.str.startswith("RFdiffusion2")]
    truth = truth.groupby("design_id").agg(
        published=("ligand_dist_des_ncac_min", "first"),
        published_no_clash=("no_clash", "first"))

    sel = pd.read_csv(a.targets)
    rows = []
    for _, r in sel.iterrows():
        mine = ligand_dist_ncac_min(Path(a.target_dir) / r.pdb)
        t = truth.loc[r.design_id]
        rows.append(dict(design_id=r.design_id, mine=mine, published=float(t.published),
                         mine_pass=mine > CLASH_CUTOFF,
                         published_pass=bool(t.published_no_clash)))
    d = pd.DataFrame(rows)
    d["abs_err"] = (d.mine - d.published).abs()

    print(f"backbones checked         : {len(d)}")
    print(f"max abs error             : {d.abs_err.max():.4f} A")
    print(f"median abs error          : {d.abs_err.median():.4f} A")
    print(f"exact matches (<0.001 A)  : {(d.abs_err < 1e-3).sum()}/{len(d)}")
    print(f"no_clash boolean agreement: {(d.mine_pass == d.published_pass).sum()}/{len(d)}")
    ok = (d.abs_err < 1e-3).all()
    print("\nCALIBRATION", "PASSED" if ok else "FAILED -- do not trust the reward yet")
    d.to_csv(HERE / "calibration_ligand_dist.csv", index=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
