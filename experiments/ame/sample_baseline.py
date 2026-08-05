#!/usr/bin/env python3
"""Sample baseline LigandMPNN sequences for an RFdiffusion2 backbone.

Regenerating the baseline ourselves -- rather than reading RFdiffusion2's
deposited sequences -- is what makes the head-to-head internally consistent:
both arms then pass through one sampler, one oracle and one metric
implementation. It also sidesteps the fact that their Chai structures were
never deposited, so their per-sequence numbers cannot be reproduced exactly
regardless of how careful we are.

This deliberately reuses :class:`PolicyMPNN`'s model loading, featurization and
rollout. The baseline is then *the same code path as CLEO with zero gradient
steps*, so any difference in the results is attributable to training rather
than to two different implementations of "sample from MPNN".

Motif residues are pinned via ``fixed_residues`` from the ``.trb``; the design
PDB carries their native identities (everything else is poly-ALA), so the
rollout's ``S_true`` substitution reproduces them exactly.

Side-chain context
------------------
Defaults to **on**, matching RFdiffusion2's motif-rotamer-aware LigandMPNN. This
is not a detail. The benchmark's motif RMSD is dominated by side-chain tip
placement (Lys NZ, Glu OE2 and the like), so a policy that sees only the motif
backbone designs a pocket that has no reason to hold the rotamer. Measured on
``M0664_2dhn_cond29_29``: with context off, AF3 reproduced the fold (backbone
RMSD 0.9 A) and the motif backbone (0.15 A) while motif all-atom RMSD sat at
1.8-3.5 A against a 1.5 A cutoff -- the pocket was right and the rotamer was not.

``--side-chain-context 0`` reproduces the old behaviour and is worth keeping as
an ablation: it isolates how much of the benchmark is rotamer placement.

Usage
-----
    uv run python experiments/ame/sample_baseline.py \
        --pdb experiments/ame/targets/<backbone>.pdb --n 40 --temperature 0.1
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from backbone_to_config import parse_trb  # noqa: E402
from cleo.design.utils.policy import PolicyMPNN  # noqa: E402


class BaselineSampler(PolicyMPNN):
    """PolicyMPNN with the training machinery removed.

    The parent constructor instantiates a reward function and an optimizer and
    creates a run directory. None of that is meaningful for an untrained
    baseline, and requiring a reward config here would mean the baseline could
    not be sampled without also standing up an oracle.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_mpnn_model().eval()
        self.ligand_mpnn_use_atom_context = 1
        self.ligand_mpnn_cutoff_for_score = 8.0

    @torch.no_grad()
    def sample(self, n, batch_size=32):
        """-> list of n sequence strings."""
        seqs = []
        while len(seqs) < n:
            self.cfg.batch_size = min(batch_size, n - len(seqs))
            feat = self.featurize_pdb(self.cfg.pdb)
            h_V, h_E, E_idx = self.model.encode(feat)
            out = self.rollout(feat, h_V, h_E, E_idx)
            seqs.extend(self.get_sequences(out))
        return seqs[:n]


def build_cfg(pdb, fixed_residues, temperature, model_type="ligand_mpnn", omit_AA="CX",
              side_chain_context=1):
    return OmegaConf.create({
        "pdb": str(pdb),
        "model_type": model_type,
        "temperature": temperature,
        "omit_AA": omit_AA,
        "fixed_residues": fixed_residues,
        "batch_size": 1,
        "checkpoint_path": None,
        "ligand_mpnn_use_side_chain_context": side_chain_context,
    })


def sample_backbone(pdb, n=40, temperature=0.1, batch_size=32, model_type="ligand_mpnn",
                    omit_AA="CX", name_prefix=None, side_chain_context=1):
    """-> DataFrame[name, sequence, backbone, temperature]. Deduplicated is NOT
    applied: collision rate at low temperature is itself a reported quantity."""
    pdb = Path(pdb)
    trb = pdb.with_suffix(".trb")
    if not trb.exists():
        raise FileNotFoundError(f"no .trb beside {pdb}")
    fixed, _, _ = parse_trb(trb)

    stem = pdb.name.replace("-atomized-bb-True.pdb", "").replace(".pdb", "")
    sc = int(side_chain_context)
    prefix = name_prefix or f"base_T{temperature}_sc{sc}_{stem}"

    sampler = BaselineSampler(build_cfg(pdb, fixed, temperature, model_type, omit_AA, sc))
    seqs = sampler.sample(n, batch_size=batch_size)

    return pd.DataFrame({
        "name": [f"{prefix}_{i:04d}" for i in range(len(seqs))],
        "sequence": seqs,
        "backbone": stem,
        "temperature": temperature,
        "side_chain_context": sc,
        "arm": "baseline",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb")
    g.add_argument("--pdb-dir")
    ap.add_argument("--n", type=int, default=40, help="sequences per backbone (their budget)")
    ap.add_argument("--temperature", type=float, default=0.1, help="their published setting")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--model-type", default="ligand_mpnn", choices=["ligand_mpnn", "protein_mpnn"])
    ap.add_argument("--omit-AA", default="CX")
    ap.add_argument("--side-chain-context", type=int, default=1, choices=[0, 1],
                    help="show fixed motif rotamers to the model (RFdiffusion2's mode)")
    ap.add_argument("--out", default=None, help="output CSV (default <out-dir>/baseline_seqs.csv)")
    ap.add_argument("--out-dir", default=HERE / "baseline")
    a = ap.parse_args()

    pdbs = [Path(a.pdb)] if a.pdb else sorted(Path(a.pdb_dir).glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"no .pdb files in {a.pdb_dir}")

    dfs = [sample_backbone(p, a.n, a.temperature, a.batch_size, a.model_type, a.omit_AA,
                           side_chain_context=a.side_chain_context)
           for p in pdbs]
    df = pd.concat(dfs, ignore_index=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else out_dir / "baseline_seqs.csv"
    df.to_csv(out, index=False)

    print(f"{len(df)} sequences from {len(pdbs)} backbone(s) -> {out}")
    for bb, sub in df.groupby("backbone"):
        print(f"  {bb}: {sub.sequence.nunique()}/{len(sub)} unique")
    return df


if __name__ == "__main__":
    main()
