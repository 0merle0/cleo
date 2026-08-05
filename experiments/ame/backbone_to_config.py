#!/usr/bin/env python3
"""Turn an RFdiffusion2 backbone (.pdb + .trb) into a runnable CLEO config.

Works on any of the 4,100 AME backbones -- or any RFdiffusion2 output that ships
its .trb alongside the .pdb. Emits three files per backbone:

    <out>/configs/<name>.yaml         CLEO design-training config
    <out>/templates/<name>.yaml       Boltz input template  (in-loop reward)
    <out>/templates/<name>.json       AF3 input template    (held-out eval)

Usage
-----
    # one backbone
    python backbone_to_config.py --pdb targets/run_M0157_1qh5_cond0_3-atomized-bb-True.pdb

    # a whole directory
    python backbone_to_config.py --pdb-dir targets/ --out .

What comes out of the .trb
--------------------------
`con_hal_pdb_idx` indexes the design PDB. (NOT `con_hal_pdb_idx_literal`, whose
positions are poly-ALA placeholders from the contig string -- verified by
checking that the non-literal positions carry the motif side-chain identities
and match `con_ref_pdb_idx` element-wise.)

`atomize_indices2atomname`, keyed by `con_hal_idx0`, gives the specific atoms
each motif residue is constrained on. That atom set -- not the whole residue --
is what the motif RMSD is computed over.

A residue constrained only on backbone atoms (N/CA/C/O) has free identity, so it
stays designable rather than being pinned in `fixed_residues`.
"""

import argparse
import json
import pickle
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Their published success criterion, recovered from the deposited per-sequence
# results by locating the cut points between each raw metric and its boolean:
#   chai_motif_pass = backbone_aligned_allatom_rmsd_chai_motif < 1.5 A
#   no_clash        = ligand_dist_des_ncac_min              >= 1.5 A
# no_clash is constant across all 40 sequences of a backbone, so it is a
# property of the backbone that no sequence can repair; only motif RMSD is
# optimizable and only motif RMSD enters the reward.
MOTIF_RMSD_THRESHOLD = 1.5

# Verified present on this system.
AF3_CONTAINER = "/net/software/containers/af3.sif"
AF3_SCRIPT = "/opt/alphafold3/run_alphafold.py"
AF3_MODEL_DIR = "/net/databases/alphafold"


class _Shim(pickle.Unpickler):
    """Unpickle a .trb without rf_diffusion/torch installed.

    Only plain arrays, lists and tuples are read; anything from the design stack
    is replaced with an inert placeholder.
    """

    def find_class(self, mod, name):
        if mod.startswith(("rf_diffusion", "torch", "openfold", "rf2aa", "ipd")):
            return type(name, (object,), {"__init__": lambda s, *a, **k: None})
        return super().find_class(mod, name)


def parse_trb(trb_path):
    """-> (fixed_residues, motif_atoms, native_motif). See module docstring."""
    t = _Shim(open(trb_path, "rb")).load()
    hal = t["con_hal_pdb_idx"]
    ref = t["con_ref_pdb_idx"]
    idx0 = list(t["con_hal_idx0"])
    a2n = t["atomize_indices2atomname"]

    motif_atoms, fixed = {}, []
    for (chain, resnum), i0 in zip(hal, idx0):
        key = f"{chain}{int(resnum)}"
        atoms = list(a2n.get(int(i0), []))
        motif_atoms[key] = atoms
        if any(a not in BACKBONE_ATOMS for a in atoms):
            fixed.append((chain, int(resnum)))

    fixed_residues = " ".join(f"{c}{i}" for c, i in sorted(fixed, key=lambda x: (x[0], x[1])))
    return fixed_residues, motif_atoms, [f"{c}{int(i)}" for c, i in ref]


AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_pdb(pdb_path):
    """-> (sequence, chain_id, ligand_ccd_codes). Ligands are HETATM residue names."""
    seq, seen_res, chain, ligands = [], set(), None, []
    for line in open(pdb_path):
        if line.startswith("ATOM"):
            key = (line[21], int(line[22:26]))
            if key not in seen_res:
                seen_res.add(key)
                seq.append(AA3TO1.get(line[17:20].strip(), "X"))
                chain = chain or line[21]
        elif line.startswith("HETATM"):
            code = line[17:20].strip()
            # Waters are not modelled as ligands.
            if code and code not in ligands and code != "HOH":
                ligands.append(code)
    return "".join(seq), chain or "A", ligands


def boltz_template(seq, chain, ligands):
    """Boltz YAML input: one protein chain plus one entry per ligand CCD code."""
    seqs = [{"protein": {"id": chain, "sequence": seq, "msa": "empty"}}]
    for i, lig in enumerate(ligands):
        seqs.append({"ligand": {"id": chr(ord("B") + i), "ccd": lig}})
    return {"version": 1, "sequences": seqs}


def af3_template(name, seq, chain, ligands):
    """AF3 JSON input. MSAs left empty: designed sequences have no homologs, and
    the benchmark's own refolding is single-sequence."""
    seqs = [{"protein": {"id": chain, "sequence": seq,
                         "unpairedMsa": "", "pairedMsa": "", "templates": []}}]
    for i, lig in enumerate(ligands):
        seqs.append({"ligand": {"id": chr(ord("B") + i), "ccdCodes": [lig]}})
    return {"name": name, "sequences": seqs, "modelSeeds": [1],
            "dialect": "alphafold3", "version": 2}


def build_config(name, pdb, fixed_residues, motif_atoms, out_dir, run_root,
                 n_steps=200, batch_size=32, oracle="boltz",
                 checkpoint_every_n_steps=25, ref_seq=None, diversity_weight=0.0):
    """CLEO design-training config.

    oracle="boltz" puts Boltz in the reward loop and leaves AF3/Chai for
    held-out evaluation, so headline numbers are not circular. oracle="af3"
    exists for the deliberately-circular ceiling arm; label it as such if used.
    """
    tdir = Path(out_dir) / "templates"
    if oracle == "af3":
        pred = {"name": "af3",
                "target_fn": "cleo.design.utils.oracle.af3_from_df",
                "cfg": {"rundir": str(Path(run_root) / name),
                        "template_path": str(tdir / f"{name}.json"),
                        "af3_container": AF3_CONTAINER,
                        "af3_script": AF3_SCRIPT,
                        "model_dir": AF3_MODEL_DIR}}
    else:
        pred = {"name": "boltz",
                "target_fn": "cleo.design.utils.oracle.boltz_from_df",
                "cfg": {"template_path": str(tdir / f"{name}.yaml")}}

    steps = [
        pred,
        {"name": "ame",
         "target_fn": "cleo.design.utils.rfd2_benchmark.rfd2_metrics_from_df",
         "cfg": {"design_pdb": str(pdb),
                 "trb": str(Path(pdb).with_suffix(".trb")),
                 # Keys are the catalytic residues; the metric uses all their
                 # heavy atoms, so the atom lists here only feed the secondary
                 # contigatom variant.
                 "motif_atoms": motif_atoms,
                 "structure_col": f"{pred['name']}_path"}},
    ]
    aggregation = [
        {"metric": "ame_motif_rmsd", "lower_bound": 0.5,
         "upper_bound": 6.0, "weight": 1.0, "mode": "min"},
    ]
    # Rank-normalise each term over the batch when there is more than one, so
    # `weight` actually controls influence. Rank rather than zscore because the
    # metrics' ranges move by an order of magnitude during training and fixed
    # bounds cannot track that: a step-0 batch on a 4-chain target had motif
    # RMSD 9-37 A, which a 6 A bound clipped entirely flat. Ranks are uniform
    # by construction, so both terms carry identical spread and equal weight
    # means equal influence at every stage. See UniversalReward.
    if diversity_weight:
        aggregation[0]["normalize"] = "rank"

    if diversity_weight:
        # "Unique mutations" = a (position, residue) choice carried by exactly
        # one sequence in the batch. Reference-free, so total_muts is just the
        # sequence length and `marginal_fraction` reads directly as "fraction of
        # positions where my residue is unique among my peers".
        #
        # Bounds are NOT [0,1]. GRPO consumes within-batch differences, and on a
        # 16-sequence T=1.0 batch this metric spans only ~0.13-0.25 (sd 0.029)
        # against the RMSD term's sd of ~0.34. Left on [0,1] a nominal
        # `weight: 1.0` would deliver a few percent of the actual influence.
        # Bounds set to the measured operating range, widened for headroom, so
        # equal weight means equal influence in practice. Re-measure if the
        # sampling temperature or batch size changes.
        #
        # pairwise_hamming is also logged but deliberately not used as the
        # reward: at sd 0.013 across a batch it is nearly constant and would
        # supply almost no gradient. It is the better *reporting* measure of
        # library spread.
        steps.insert(0, {
            "name": "div",
            "target_fn": "cleo.design.utils.mutation_diversity.mutation_diversity_from_df",
            "cfg": {},
        })
        aggregation.append(
            {"metric": "div_marginal_fraction", "lower_bound": 0.05,
             "upper_bound": 0.35, "weight": diversity_weight, "mode": "max",
             "normalize": "rank"})

    return {
        "run_name": name,
        "output_dir": str(Path(run_root) / name),
        "pdb": str(pdb),
        "checkpoint_path": None,
        "algorithm": "grpo",
        "batch_size": batch_size,
        "N_steps": n_steps,
        "lr": 1e-4,
        # Hard attribute access in PolicyMPNN.__init__ -- omitting it crashes at
        # startup rather than falling back to a default. Kept infrequent (10 MB
        # per snapshot) because the experimental artifact is the folded sequence
        # record accumulated during training, not the policy weights; best/last
        # are saved regardless.
        "checkpoint_every_n_steps": checkpoint_every_n_steps,
        "kl_weight": 0.0,
        "N_updates": 16,
        "update_batch_size": 8,
        "clip_eps_low": 0.2,
        "clip_eps_high": 0.28,
        # Every AME target has a ligand.
        "model_type": "ligand_mpnn",
        # Motif-rotamer-aware, as in the RFdiffusion2 baseline. Required for a
        # fair head-to-head, and required to do well at all: the benchmark's
        # motif RMSD is measured over side-chain atoms, so a policy shown only
        # the motif backbone cannot learn to hold the rotamer. Empirically this
        # is the difference between ~2-3 A and passing.
        "ligand_mpnn_use_side_chain_context": 1,
        "temperature": 1.0,
        "omit_AA": "CX",
        "fixed_residues": fixed_residues,
        "reward": {
            "_target_": "cleo.design.utils.reward.UniversalReward",
            # UniversalReward builds its per-step rundir from these; omitting
            # them leaves output_dir None and the run dies on the first reward
            # call. Interpolated so they cannot drift from the top-level values.
            "output_dir": "${output_dir}",
            "run_name": "${run_name}",
            "steps": steps,
            # Motif RMSD is the benchmark's only optimizable criterion
            # (no_clash is fixed by the backbone).
            #
            # Upper bound 6.0, not 3.0. At the T=1.0 training temperature a
            # step-0 batch spans 1.5-12.3 A with a 3.8 A median, so a 3.0 A
            # clip flattens roughly half the batch to exactly zero and the
            # policy cannot tell a 4 A design from a 12 A one. Widening keeps
            # gradient across the range the policy actually occupies early,
            # while still resolving the 1.5 A cutoff region.
            #
            # UniversalReward divides by the summed weights, so weights are
            # relative: 1.0 and 1.0 means each term contributes half.
            "reward_aggregation": aggregation,
        },
    }


def convert(pdb_path, out_dir, run_root, oracle="boltz", n_steps=200, batch_size=32,
            checkpoint_every_n_steps=25, diversity_weight=0.0):
    """One backbone -> config + Boltz template + AF3 template. Returns a summary."""
    pdb_path = Path(pdb_path)
    trb = pdb_path.with_suffix(".trb")
    if not trb.exists():
        raise FileNotFoundError(f"no .trb beside {pdb_path}")

    name = pdb_path.name.replace("-atomized-bb-True.pdb", "").replace(".pdb", "")
    fixed, motif_atoms, native = parse_trb(trb)
    seq, chain, ligands = parse_pdb(pdb_path)

    out_dir = Path(out_dir)
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    (out_dir / "templates").mkdir(parents=True, exist_ok=True)

    (out_dir / "templates" / f"{name}.yaml").write_text(
        yaml.safe_dump(boltz_template(seq, chain, ligands), sort_keys=False))
    (out_dir / "templates" / f"{name}.json").write_text(
        json.dumps(af3_template(name, seq, chain, ligands), indent=2))
    cfg = build_config(name, pdb_path, fixed, motif_atoms, out_dir, run_root,
                       n_steps=n_steps, batch_size=batch_size, oracle=oracle,
                       checkpoint_every_n_steps=checkpoint_every_n_steps,
                       ref_seq=seq, diversity_weight=diversity_weight)
    (out_dir / "configs" / f"{name}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

    return dict(name=name, length=len(seq), chain=chain,
                ligands=",".join(ligands), n_motif_res=len(motif_atoms),
                n_fixed=len(fixed.split()) if fixed else 0,
                n_motif_atoms=sum(len(v) for v in motif_atoms.values()),
                fixed_residues=fixed, native_motif=" ".join(native),
                motif_atoms=";".join(f"{k}:{'/'.join(v)}" for k, v in motif_atoms.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb", help="single backbone .pdb (expects .trb beside it)")
    g.add_argument("--pdb-dir", help="directory of backbones")
    ap.add_argument("--out", default=HERE, help="output root")
    ap.add_argument("--run-root", default=None, help="where CLEO runs write (default <out>/runs)")
    ap.add_argument("--oracle", choices=["boltz", "af3"], default="boltz",
                    help="in-loop reward oracle; boltz keeps AF3/Chai held out")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="policy snapshots are 10 MB each and are not the artifact")
    ap.add_argument("--diversity-weight", type=float, default=0.0,
                    help="weight on batch mutation diversity; 1.0 matches the RMSD term")
    a = ap.parse_args()

    out = Path(a.out)
    run_root = Path(a.run_root) if a.run_root else out / "runs"
    pdbs = [Path(a.pdb)] if a.pdb else sorted(Path(a.pdb_dir).glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"no .pdb files found in {a.pdb_dir}")

    rows = [convert(p, out, run_root, a.oracle, a.n_steps, a.batch_size,
                    a.checkpoint_every, a.diversity_weight) for p in pdbs]
    print(f"converted {len(rows)} backbone(s) -> {out}/configs, {out}/templates")
    for r in rows[:5]:
        print(f"  {r['name']}: len={r['length']} lig={r['ligands']} "
              f"motif={r['n_motif_res']}res/{r['n_motif_atoms']}atoms fixed={r['n_fixed']}")
    if len(rows) > 5:
        print(f"  ... and {len(rows)-5} more")
    return rows


if __name__ == "__main__":
    main()
