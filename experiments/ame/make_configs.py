#!/usr/bin/env python3
"""Derive per-backbone design parameters from the AME targets and emit CLEO configs.

For each pilot backbone this reads:
  - the .trb  -> con_hal_pdb_idx, the motif positions *in the designed backbone*,
                 which become CLEO's `fixed_residues`
  - the .pdb  -> HETATM residue names, which become the ligand list

and writes one training config per target plus a manifest.

The .trb is pickled with rf_diffusion classes in scope, so it is loaded through
a shim unpickler that stubs out anything we do not need. Only plain
arrays/lists/tuples are read.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Their published success criterion, recovered from the deposited per-sequence
# results by locating the cut points between the raw metric and its boolean:
#   chai_motif_pass = backbone_aligned_allatom_rmsd_chai_motif < 1.5 A
#   no_clash        = ligand_dist_des_ncac_min              >= 1.5 A
# no_clash is constant across all 40 sequences of a backbone, so it is a
# backbone property and NOT part of the reward -- only motif RMSD is.
MOTIF_RMSD_THRESHOLD = 1.5


class _Shim(pickle.Unpickler):
    """Unpickle .trb without rf_diffusion/torch installed."""

    def find_class(self, mod, name):
        if mod.startswith(("rf_diffusion", "torch", "openfold", "rf2aa", "ipd")):
            return type(name, (object,), {"__init__": lambda s, *a, **k: None})
        return super().find_class(mod, name)


BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def motif_from_trb(trb_path):
    """Per-atom motif definition for one design.

    Indexing note, verified against residue identities: `con_hal_pdb_idx` (NOT
    `con_hal_pdb_idx_literal`) is what indexes the output PDB. The literal
    positions are poly-ALA placeholders from the contig string; the non-literal
    positions carry the real motif side chains and match `con_ref_pdb_idx`
    element-wise.

    `atomize_indices2atomname` is keyed by `con_hal_idx0`, so the atom sets pair
    positionally with the same ordering.

    Returns (fixed_residues, motif_atoms, native_motif) where:
      fixed_residues -- residues with >=1 SIDE-CHAIN atom constrained. A residue
        constrained only on backbone atoms has free identity and must stay
        designable, otherwise we would be handing the policy an answer the
        benchmark never fixed.
      motif_atoms    -- {"A25": ["OD2","CG"], ...}, the atoms the motif RMSD is
        computed over.
    """
    t = _Shim(open(trb_path, "rb")).load()
    hal = t["con_hal_pdb_idx"]          # [(chain, resnum), ...] design numbering
    ref = t["con_ref_pdb_idx"]          # [(chain, resnum), ...] native numbering
    idx0 = list(t["con_hal_idx0"])      # 0-based, keys of atomize_indices2atomname
    a2n = t["atomize_indices2atomname"]

    motif_atoms, fixed = {}, []
    for (chain, resnum), i0 in zip(hal, idx0):
        key = f"{chain}{int(resnum)}"
        atoms = list(a2n.get(int(i0), []))
        motif_atoms[key] = atoms
        if any(a not in BACKBONE_ATOMS for a in atoms):
            fixed.append((chain, int(resnum)))

    fixed_residues = " ".join(f"{c}{i}" for c, i in sorted(fixed, key=lambda x: (x[0], x[1])))
    native = [f"{c}{int(i)}" for c, i in ref]
    return fixed_residues, motif_atoms, native


def ligands_from_pdb(pdb_path):
    """Distinct HETATM residue names, in first-seen order."""
    seen = []
    for line in open(pdb_path):
        if line.startswith("HETATM"):
            code = line[17:20].strip()
            if code and code not in seen:
                seen.append(code)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=HERE / "pilot_targets.csv")
    ap.add_argument("--target-dir", default=HERE / "targets")
    ap.add_argument("--out", default=HERE / "configs")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tdir = Path(a.target_dir)
    sel = pd.read_csv(a.targets)

    rows = []
    for _, r in sel.iterrows():
        pdb = tdir / r.pdb
        trb = pdb.with_suffix(".trb")
        fixed, motif_atoms, ref_res = motif_from_trb(trb)
        ligs = ligands_from_pdb(pdb)
        name = r.pdb.replace("-atomized-bb-True.pdb", "")

        cfg = {
            "run_name": f"ame_{r.cls}_{name}",
            "output_dir": str(HERE / "runs" / f"ame_{r.cls}_{name}"),
            "pdb": str(pdb),
            "checkpoint_path": None,
            "algorithm": "grpo",
            "batch_size": a.batch_size,
            "N_steps": a.n_steps,
            "lr": 1e-4,
            "kl_weight": 0.0,
            "N_updates": 16,
            "update_batch_size": 8,
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.28,
            # Ligand present in every AME target, so ligand_mpnn throughout.
            # NOTE: RFdiffusion2's baseline runs LigandMPNN in motif-rotamer-aware
            # mode with packing. Confirm CLEO's ligand_mpnn path matches before
            # treating the baseline comparison as apples-to-apples.
            "model_type": "ligand_mpnn",
            "temperature": 1.0,
            "omit_AA": "CX",
            "fixed_residues": fixed,
            "reward": {
                "_target_": "cleo.design.utils.reward.UniversalReward",
                "steps": [
                    {
                        # Cheap in-loop oracle. Chai-1 is deliberately held out
                        # for evaluation so the headline number is not circular.
                        "name": "boltz",
                        "target_fn": "cleo.design.utils.oracle.boltz_from_df",
                        "cfg": {"template_path": str(HERE / "templates" / f"{name}.yaml")},
                    },
                    {
                        "name": "ame",
                        "target_fn": "cleo.design.utils.ame.ame_metrics_from_df",
                        "cfg": {
                            "design_pdb": str(pdb),
                            "motif_atoms": motif_atoms,
                            "structure_col": "boltz_path",
                            "rmsd_threshold": MOTIF_RMSD_THRESHOLD,
                        },
                    },
                ],
                "reward_aggregation": [
                    {
                        "metric": "ame_motif_rmsd",
                        "lower_bound": 0.5,
                        "upper_bound": 4.0,
                        "weight": 1.0,
                        "mode": "min",
                    }
                ],
            },
        }
        (out / f"{cfg['run_name']}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        )
        rows.append(dict(benchmark=r.benchmark, cls=r.cls, name=name, pdb=str(pdb),
                         fixed_residues=fixed, native_motif=" ".join(ref_res),
                         motif_atoms=";".join(f"{k}:{'/'.join(v)}" for k, v in motif_atoms.items()),
                         ligands=",".join(ligs),
                         n_motif_res=len(motif_atoms),
                         n_fixed=len(fixed.split()),
                         n_motif_atoms=sum(len(v) for v in motif_atoms.values()),
                         n_pass_baseline=r.n_pass))

    m = pd.DataFrame(rows)
    m.to_csv(HERE / "pilot_manifest.csv", index=False)
    print(f"wrote {len(m)} configs -> {out}")
    print(f"manifest -> {HERE/'pilot_manifest.csv'}\n")
    print(m.groupby(["benchmark", "cls"]).agg(n=("name", "size"),
                                              motif_res=("n_motif_res", "first"),
                                              fixed=("n_fixed", "first"),
                                              motif_atoms=("n_motif_atoms", "first"),
                                              ligands=("ligands", "first")).to_string())


if __name__ == "__main__":
    main()
