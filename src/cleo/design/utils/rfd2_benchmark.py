"""RFdiffusion2 enzyme-benchmark metrics as a CLEO reward step.

Optimizes directly for the published AME benchmark criteria, so the reward is
the thing the field already scores on rather than a proxy for it. Metric
definitions are transcribed from the RFdiffusion2 source
(``rf_diffusion/dev/benchmark.py`` and ``rf_diffusion/benchmark/per_sequence_metrics.py``)
rather than inferred.

The motif definition is read straight from the design's ``.trb``, so this works
on any RFdiffusion2 output without per-target configuration.

Criteria (names and cutoffs as published)
-----------------------------------------
``chai_motif_pass``
    ``backbone_aligned_allatom_rmsd_chai_motif < 1.5``. Superimpose the
    predicted structure onto the design over protein backbone atoms, then take
    all-atom RMSD over the constrained motif atoms. Sequence-dependent, and the
    primary optimization target.

``no_clash``
    ``ligand_dist_des_ncac_min > 1.5``. Minimum distance from any ligand atom to
    any design N/CA/C atom. **A property of the backbone alone** -- it is
    constant across all 40 sequences of a backbone in the deposited results, so
    no sequence can repair it. Reported for bookkeeping; never optimized.

Composite ``chai_motif_pass_and_no_clash`` is a strict AND. Verified against all
328,000 rows of the deposited ``AME_benchmark_results_all.csv``: the two
booleans are exactly the cutoffs above and the composite is exactly their
conjunction.

Ligand placement is NOT a benchmark criterion
---------------------------------------------
``ligand_rmsd`` below is a CLEO diagnostic, not an AME metric. The deposit
carries no ligand-pose column at all -- only ``ligand_dist_des_ncac_min``, which
is the clash check. A pocket-aligned ligand RMSD is computable from the
RFdiffusion2 source, but they neither report it nor score on it, so there are no
published values to calibrate against.

It is therefore logged and never optimized. Two reasons beyond the missing
ground truth. On ``M0664_2dhn_cond29_29`` it is uncorrelated with motif
accuracy -- the best motif (1.32 A, passing) had the worst ligand RMSD (5.61 A)
and a failing motif (1.55 A) had the best (1.98 A) -- and it is insensitive to
the alignment used (global backbone 4.88 A vs pocket 4.89 A), so the large
values reflect the predictor placing the ligand elsewhere rather than a choice
of superposition. Rewarding a quantity with no ground truth, under an oracle
whose ligand placement is unvalidated on de novo designs, is a reward-hacking
risk with no benchmark upside.

Oracle independence
-------------------
The metric is oracle-agnostic: point ``structure_col`` at whichever predictor
produced the structure. Optimizing the benchmark metric under a *cheap* in-loop
oracle (Boltz) while reporting it under the benchmark's own oracle (Chai-1) is
what keeps "we optimize the benchmark directly" from collapsing into "we
optimized the number we report".
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io as strucio

BACKBONE_ATOMS = {"N", "CA", "C", "O"}
NCAC = {"N", "CA", "C"}

MOTIF_RMSD_CUTOFF = 1.5   # fa_rmsd_cutoff; the only optimizable pass criterion
CLASH_CUTOFF = 1.5        # no_clash thresh; property of the backbone alone


class _TrbShim(pickle.Unpickler):
    """Load a .trb without the rf_diffusion stack installed."""

    def find_class(self, mod, name):
        if mod.startswith(("rf_diffusion", "torch", "openfold", "rf2aa", "ipd")):
            return type(name, (object,), {"__init__": lambda s, *a, **k: None})
        return super().find_class(mod, name)


def motif_atoms_from_trb(trb_path):
    """{"A25": ["OD2", "CG"], ...} -- constrained atoms per motif residue.

    Uses ``con_hal_pdb_idx``, which indexes the design PDB. (Not
    ``con_hal_pdb_idx_literal``, whose positions are poly-ALA placeholders from
    the contig string.) ``atomize_indices2atomname`` is keyed by
    ``con_hal_idx0``, so atom sets pair positionally.
    """
    t = _TrbShim(open(trb_path, "rb")).load()
    a2n = t["atomize_indices2atomname"]
    out = {}
    for (chain, resnum), i0 in zip(t["con_hal_pdb_idx"], t["con_hal_idx0"]):
        out[f"{chain}{int(resnum)}"] = list(a2n.get(int(i0), []))
    return out


def _load(path):
    arr = strucio.load_structure(path)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    return arr


def _protein_mask(arr):
    return struc.filter_amino_acids(arr)


def _kabsch(mobile, target):
    """Rotation+translation taking `mobile` onto `target` (both (N,3))."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return lambda x: (x - mc) @ r.T + tc


def _rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def _atom_index(arr, mask=None):
    """{(chain, resnum, atom_name): row index}"""
    idx = {}
    rng = np.arange(arr.array_length()) if mask is None else np.where(mask)[0]
    for i in rng:
        idx[(arr.chain_id[i], int(arr.res_id[i]), arr.atom_name[i])] = i
    return idx


def _paired_coords(a, b, keys):
    """Coordinates for `keys` present in both structures, plus the miss count."""
    ia, ib = _atom_index(a), _atom_index(b)
    ca, cb, missing = [], [], 0
    for k in keys:
        if k in ia and k in ib:
            ca.append(a.coord[ia[k]])
            cb.append(b.coord[ib[k]])
        else:
            missing += 1
    if not ca:
        return None, None, missing
    return np.array(ca), np.array(cb), missing


def _backbone_keys(arr):
    m = _protein_mask(arr)
    return [(arr.chain_id[i], int(arr.res_id[i]), arr.atom_name[i])
            for i in np.where(m)[0] if arr.atom_name[i] in BACKBONE_ATOMS]


def _ligand_mask(arr):
    return ~_protein_mask(arr) & (arr.element != "H") & (arr.res_name != "HOH")


def compute_metrics(design_pdb, pred_path, motif_atoms, pocket_cutoff=8.0):
    """All benchmark metrics for one predicted structure. Returns a dict."""
    des, pred = _load(design_pdb), _load(pred_path)
    out = {}

    # --- chai_motif_pass: backbone-align prediction to design, RMSD on motif atoms
    bb_keys = _backbone_keys(des)
    bd, bp, _ = _paired_coords(des, pred, bb_keys)
    if bd is None:
        return {"motif_rmsd": np.nan}
    to_design = _kabsch(bp, bd)

    motif_keys = [(r[0], int(r[1:]), a) for r, atoms in motif_atoms.items() for a in atoms]
    md, mp, miss = _paired_coords(des, pred, motif_keys)
    if md is None:
        return {"motif_rmsd": np.nan}
    out["motif_rmsd"] = _rmsd(to_design(mp), md)
    out["motif_atoms_missing"] = miss
    out["motif_pass"] = out["motif_rmsd"] < MOTIF_RMSD_CUTOFF

    # --- no_clash: design-only, min ligand-to-N/CA/C distance. Constant per
    # backbone; computed here so a run is self-describing.
    lig_des = des.coord[_ligand_mask(des)]
    ncac = des.coord[_protein_mask(des) & np.isin(des.atom_name, list(NCAC))]
    if len(lig_des) and len(ncac):
        d = np.linalg.norm(lig_des[:, None, :] - ncac[None, :, :], axis=-1)
        out["ligand_dist_des_ncac_min"] = float(d.min())
        out["no_clash"] = out["ligand_dist_des_ncac_min"] > CLASH_CUTOFF
    else:
        out["ligand_dist_des_ncac_min"] = np.nan
        out["no_clash"] = False

    # --- ligand placement: DIAGNOSTIC ONLY, not an AME criterion (see module
    # docstring). Pocket = protein backbone within `pocket_cutoff` A of any
    # design ligand atom. No pass/fail is emitted and nothing here feeds the
    # composite: there is no published value to threshold against, so a boolean
    # would invent a standard and invite it into a reward by accident.
    # `ligand_atoms_matched` is recorded because an unnoticed atom-name mismatch
    # would silently shrink the comparison set and flatter the RMSD.
    lig_pred_mask = _ligand_mask(pred)
    if len(lig_des) and lig_pred_mask.any():
        bbm = _protein_mask(des) & np.isin(des.atom_name, list(BACKBONE_ATOMS))
        bb_idx = np.where(bbm)[0]
        dmat = np.linalg.norm(des.coord[bb_idx][:, None, :] - lig_des[None, :, :], axis=-1)
        near = bb_idx[dmat.min(1) < pocket_cutoff]
        pocket_keys = [(des.chain_id[i], int(des.res_id[i]), des.atom_name[i]) for i in near]
        pd_, pp_, _ = _paired_coords(des, pred, pocket_keys)
        if pd_ is not None and len(pd_) >= 3:
            to_pocket = _kabsch(pp_, pd_)
            rmsds, n_matched, n_total = [], 0, 0
            for res in np.unique(pred.res_name[lig_pred_mask]):
                sel_p = lig_pred_mask & (pred.res_name == res)
                sel_d = _ligand_mask(des) & (des.res_name == res)
                keys = [(pred.chain_id[i], int(pred.res_id[i]), pred.atom_name[i])
                        for i in np.where(sel_p)[0]]
                names_d = {des.atom_name[i]: des.coord[i] for i in np.where(sel_d)[0]}
                a, b = [], []
                for (_, _, an), i in zip(keys, np.where(sel_p)[0]):
                    if an in names_d:
                        a.append(pred.coord[i])
                        b.append(names_d[an])
                n_matched += len(a)
                n_total += len(keys)
                if len(a) >= 1:
                    rmsds.append(_rmsd(to_pocket(np.array(a)), np.array(b)))
            if rmsds:
                out["ligand_rmsd_max"] = float(max(rmsds))
                out["ligand_atoms_matched"] = n_matched
                out["ligand_atoms_total"] = n_total

    out["motif_pass_and_no_clash"] = bool(out.get("motif_pass", False) and out.get("no_clash", False))
    return out


def rfd2_metrics_from_df(df_input, cfg, step_name="rfd2"):
    """Reward step: RFdiffusion2 benchmark metrics for each predicted structure.

    Config
    ------
    design_pdb      RFdiffusion2 backbone the sequences were designed on.
    trb             Its .trb. Defaults to design_pdb with a .trb suffix.
    structure_col   Column holding predicted structure paths, e.g. ``af3_path``
                    or ``boltz_path``.
    pocket_cutoff   Angstroms defining the pocket for ligand alignment (8.0).

    Adds ``{step_name}_{metric}`` columns. ``{step_name}_motif_rmsd`` is the
    quantity to optimize (mode: min, cutoff 1.5).
    """
    design_pdb = cfg["design_pdb"]
    trb = cfg.get("trb") or str(design_pdb).rsplit(".", 1)[0] + ".trb"
    structure_col = cfg.get("structure_col", "boltz_path")
    pocket_cutoff = float(cfg.get("pocket_cutoff", 8.0))

    if structure_col not in df_input.columns:
        raise KeyError(
            f"{step_name}: '{structure_col}' not in dataframe; available: "
            f"{sorted(df_input.columns)}. Run a structure-prediction step first."
        )

    motif_atoms = cfg.get("motif_atoms") or motif_atoms_from_trb(trb)

    records = []
    for path in df_input[structure_col]:
        try:
            records.append(compute_metrics(design_pdb, path, motif_atoms, pocket_cutoff))
        except Exception as e:  # a single bad prediction must not kill the batch
            warnings.warn(f"{step_name}: failed on {path}: {e}")
            records.append({"motif_rmsd": np.nan})

    met = pd.DataFrame(records).add_prefix(f"{step_name}_")
    met.index = df_input.index
    return pd.concat([df_input, met], axis=1)
