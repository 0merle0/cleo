#!/usr/bin/env python3
"""Fold sequences with AF3 and score them against the RFdiffusion2 benchmark criteria.

The P0 gate: proves end-to-end that we can produce every number we intend to
report -- sample -> fold -> motif RMSD / clash / ligand placement -> reward
scalar -- before any GPU time goes into training.

Runs one backbone at a time because AF3 inputs are per-backbone (each has its
own ligand set) and the metrics need that backbone's design PDB and .trb.

    uv run python experiments/ame/fold_and_score.py --seqs baseline/baseline_seqs.csv

``--skip-run`` re-reads existing AF3 outputs and recomputes metrics only, which
makes iterating on the metric code free once the folds exist.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from backbone_to_config import AF3_CONTAINER, AF3_SCRIPT, AF3_MODEL_DIR  # noqa: E402
from cleo.design.utils.oracle import af3_from_df  # noqa: E402
from cleo.design.utils.rfd2_benchmark import rfd2_metrics_from_df  # noqa: E402


def af3_cfg(rundir, template, skip_run=False):
    return OmegaConf.create({
        "rundir": str(rundir),
        "template_path": str(template),
        "af3_container": AF3_CONTAINER,
        "af3_script": AF3_SCRIPT,
        "model_dir": AF3_MODEL_DIR,
        "skip_run": skip_run,
    })


def run_backbone(df, backbone, targets, rundir, skip_run=False, pocket_cutoff=8.0):
    """Fold and score every sequence for one backbone. -> DataFrame with metrics."""
    pdb = Path(targets) / f"{backbone}-atomized-bb-True.pdb"
    if not pdb.exists():                       # tolerate names without the suffix
        pdb = Path(targets) / f"{backbone}.pdb"
    trb = pdb.with_suffix(".trb")
    template = HERE / "templates" / f"{backbone}.json"
    for p in (pdb, trb, template):
        if not p.exists():
            raise FileNotFoundError(p)

    rundir = Path(rundir) / backbone
    rundir.mkdir(parents=True, exist_ok=True)

    folded = af3_from_df(df, af3_cfg(rundir, template, skip_run), step_name="af3")
    return rfd2_metrics_from_df(folded, {
        "design_pdb": str(pdb),
        "trb": str(trb),
        "structure_col": "af3_path",
        "pocket_cutoff": pocket_cutoff,
    }, step_name="rfd2")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", required=True, help="CSV with name, sequence, backbone")
    ap.add_argument("--targets", default=HERE / "targets")
    ap.add_argument("--rundir", default=HERE / "runs" / "fold_and_score")
    ap.add_argument("--out", default=None, help="default: alongside --seqs, *_scored.csv")
    ap.add_argument("--skip-run", action="store_true", help="reuse existing AF3 outputs")
    ap.add_argument("--pocket-cutoff", type=float, default=8.0)
    ap.add_argument("--backbone", default=None, help="restrict to one backbone")
    a = ap.parse_args()

    df = pd.read_csv(a.seqs)
    if a.backbone:
        df = df[df.backbone == a.backbone]
    if df.empty:
        raise SystemExit("no sequences selected")

    out_rows = []
    for bb, sub in df.groupby("backbone"):
        print(f"\n=== {bb}: {len(sub)} sequences ===", flush=True)
        out_rows.append(run_backbone(sub.reset_index(drop=True), bb, a.targets,
                                     a.rundir, a.skip_run, a.pocket_cutoff))
    res = pd.concat(out_rows, ignore_index=True)

    out = Path(a.out) if a.out else Path(a.seqs).with_name(Path(a.seqs).stem + "_scored.csv")
    res.to_csv(out, index=False)

    print(f"\nwrote {len(res)} rows -> {out}")
    cols = [c for c in ("rfd2_motif_rmsd", "rfd2_motif_pass", "rfd2_no_clash",
                        "rfd2_pocket_aligned_ligand_rmsd_max", "rfd2_motif_pass_and_no_clash")
            if c in res.columns]
    for bb, sub in res.groupby("backbone"):
        print(f"  {bb}:")
        for c in cols:
            v = sub[c]
            if v.dtype == bool:
                print(f"    {c:38s} {v.sum()}/{len(v)}")
            else:
                print(f"    {c:38s} min={v.min():.2f} median={v.median():.2f}")
    return res


if __name__ == "__main__":
    main()
