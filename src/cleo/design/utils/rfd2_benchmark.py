"""RFdiffusion2 enzyme-benchmark metrics as a CLEO reward step.

Optimizes directly for the published AME benchmark criteria, so the reward is
the thing the field already scores on rather than a proxy for it. Metric
definitions are transcribed from the RFdiffusion2 source
(``rf_diffusion/dev/benchmark.py`` and ``rf_diffusion/benchmark/per_sequence_metrics.py``)
rather than inferred.

The motif definition is read straight from the design's ``.trb``, so this works
on any RFdiffusion2 output without per-target configuration.

Reference structure
-------------------
We score against the design PDB we generated sequences for. Their pipeline
compares each prediction against ``unideal``, ``packed`` (the LigandMPNN-packed
design) and ``ref`` (the native active site), and the deposited
``*_chai_motif`` column corresponds to one of those -- most likely not the
idealized backbone we hold. So our absolute values need not equal theirs
sequence-for-sequence even when the definition is matched. This is fine for a
baseline-vs-CLEO comparison run through one pipeline, and is the reason
``calibrate_metrics.py`` could never reproduce their numbers exactly.

Criteria (names and cutoffs as published)
-----------------------------------------
``chai_motif_pass``
    ``backbone_aligned_allatom_rmsd_chai_motif < 1.5``. Superimpose on the
    N/CA/C/O of the **motif residues only**, then take RMSD over **all heavy
    atoms of those residues**. Sequence-dependent, and the primary optimization
    target.

    Both halves are easy to get wrong, and getting either wrong makes good
    designs look bad. The column name decodes as
    ``{align_to}_aligned_{rmsd_to}_rmsd_{source}_{target}``
    (``per_sequence_metrics.py``), with ``align_to='backbone'`` selecting
    ``['N','CA','C','O']`` per motif residue -- a *local* superposition, not a
    global one -- and ``rmsd_to='allatom'`` selecting every heavy atom of the
    residue, not the ``.trb`` contig-atom subset. The paper states it as: a
    success is RMSD of all heavy atoms in the catalytic residues < 1.5 A when
    aligned on the backbone N, CA, C of those catalytic residues.

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

# Side-chain atom pairs that are indistinguishable under a 180-degree flip of
# the terminal group. Their pipeline resolves this (`sidechain_symmetry_resolved`,
# `make_alternate_xyz_indexes`); without it an otherwise perfect carboxylate
# scores ~1 A purely from an arbitrary naming choice. Asn/Gln are deliberately
# excluded: OD1/ND2 differ in element, so the assignment is chemistry, not
# nomenclature.
SYMMETRIC_PAIRS = {
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}


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


def motif_residues_from_trb(trb_path):
    """["A62", "A66", ...] -- the catalytic residues, ignoring atom subsets.

    The published metric is over *all heavy atoms* of these residues, so only
    the residue identity matters here. ``motif_atoms_from_trb`` remains for the
    contig-atom variant.
    """
    t = _TrbShim(open(trb_path, "rb")).load()
    return [f"{c}{int(r)}" for c, r in t["con_hal_pdb_idx"]]


def _heavy_atoms_by_residue(arr, residues):
    """{"A66": ["N","CA","C","O","CB",...]} -- heavy atoms present for each residue."""
    out = {}
    for key in residues:
        chain, resnum = key[0], int(key[1:])
        m = (arr.chain_id == chain) & (arr.res_id == resnum) & (arr.element != "H")
        out[key] = sorted(set(arr.atom_name[m].tolist()))
    return out


def _residue_name(arr, key):
    m = (arr.chain_id == key[0]) & (arr.res_id == int(key[1:]))
    names = arr.res_name[m]
    return str(names[0]) if len(names) else ""


def _symmetry_variants(des, motif_residues, keys):
    """Yield each symmetry-equivalent relabelling of `keys` on the design side.

    Only the terminal-group swaps in SYMMETRIC_PAIRS are considered, and each
    residue is flipped independently, so the returned list is the product over
    residues that actually have a symmetric group. Returns a list of key lists
    parallel to `keys`.
    """
    swaps = []
    for key in motif_residues:
        pairs = SYMMETRIC_PAIRS.get(_residue_name(des, key), [])
        if pairs:
            swaps.append((key, pairs))
    variants = [list(keys)]
    for key, pairs in swaps:
        flip = {}
        for a, b in pairs:
            flip[(key[0], int(key[1:]), a)] = (key[0], int(key[1:]), b)
            flip[(key[0], int(key[1:]), b)] = (key[0], int(key[1:]), a)
        variants = variants + [[flip.get(k, k) for k in v] for v in variants]
    return variants


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
    """All benchmark metrics for one predicted structure. Returns a dict.

    ``motif_atoms`` supplies the contig-atom subset for the secondary
    ``motif_rmsd_contigatom`` variant; the headline ``motif_rmsd`` is over all
    heavy atoms of those residues, per the published definition.
    """
    des, pred = _load(design_pdb), _load(pred_path)
    out = {}
    motif_residues = list(motif_atoms)

    # --- backbone_aligned_allatom_rmsd_chai_motif.
    # Superposition is on N/CA/C/O of the MOTIF RESIDUES ONLY -- not the whole
    # protein backbone. RMSD is then over ALL heavy atoms of those residues, not
    # the .trb contig-atom subset. Both follow per_sequence_metrics.py, where the
    # column name decodes as {align_to}_aligned_{rmsd_to}_rmsd_{source}_{target}
    # with align_to='backbone' -> ['N','CA','C','O'] per motif residue and
    # rmsd_to='allatom' -> every heavy atom of the residue.
    #
    # A local alignment is much the stricter test: a global fit can absorb motif
    # error into a small whole-body rotation, whereas aligning on the motif's own
    # backbone leaves the side chains nowhere to hide.
    heavy = _heavy_atoms_by_residue(des, motif_residues)
    align_keys = [(r[0], int(r[1:]), a) for r in motif_residues
                  for a in sorted(BACKBONE_ATOMS & set(heavy[r]))]
    ad, ap, _ = _paired_coords(des, pred, align_keys)
    if ad is None or len(ad) < 3:
        return {"motif_rmsd": np.nan}
    to_design = _kabsch(ap, ad)

    allatom_keys = [(r[0], int(r[1:]), a) for r in motif_residues for a in heavy[r]]
    md, mp, miss = _paired_coords(des, pred, allatom_keys)
    if md is None:
        return {"motif_rmsd": np.nan}
    # Symmetry is resolved on the design side by relabelling, so the prediction's
    # coordinates are never permuted -- only which design atom each is compared to.
    best = None
    for variant in _symmetry_variants(des, motif_residues, allatom_keys):
        vd, vp, _ = _paired_coords(des, pred, variant)
        if vd is None:
            continue
        r = _rmsd(to_design(vp), vd)
        best = r if best is None else min(best, r)
    out["motif_rmsd"] = best if best is not None else _rmsd(to_design(mp), md)
    out["motif_atoms_missing"] = miss
    out["motif_atoms_used"] = len(md)
    out["motif_pass"] = out["motif_rmsd"] < MOTIF_RMSD_CUTOFF

    # Secondary: the constrained-atom subset under the same local alignment.
    # Kept because it is what the .trb actually pins, so a large gap between the
    # two says the constrained atoms are right and the rest of the residue is not.
    contig_keys = [(r[0], int(r[1:]), a) for r, atoms in motif_atoms.items() for a in atoms]
    cd, cp, _ = _paired_coords(des, pred, contig_keys)
    if cd is not None:
        out["motif_rmsd_contigatom"] = _rmsd(to_design(cp), cd)

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


def best_of_n_from_df(df_input, cfg, step_name="best_of_n"):
    """Reward step: collapse per-sample rows to one row per sequence, best-of-N.

    Pair this with ``af3_from_df`` run at ``num_diffusion_samples: N`` *and*
    ``per_sample_rows: true``, which emits one row per (sequence, sample). A
    reward step must return one row per sampled sequence, so something has to
    collapse those rows before aggregation -- this is that something.

    Why not let ``af3_from_df`` collapse them itself: left to its own devices it
    keeps the top-ranked sample, ranked by AF3's confidence score. That is a
    best-of-N by *confidence*, which is not the benchmark's unit and correlates
    only loosely with motif accuracy. The benchmark counts a design as a success
    if any one of its predictions clears the motif-RMSD cutoff and is clash-free,
    so the reduction has to be over the benchmark's own criterion.

    Two reductions, deliberately not the same one:

    ``{metric_prefix}_motif_rmsd``  min over samples -- the continuous training
        signal, the best geometry the sequence proved able to reach.
    ``{metric_prefix}_motif_pass_and_no_clash``  OR over samples of the
        per-prediction AND. Computed from the per-sample conjunction rather than
        from the reduced columns, because the sample with the lowest RMSD is not
        always one that avoids the clash; reducing each column separately and
        then AND-ing them would score a design as passing on the strength of two
        different predictions, and would overcount.

    Remaining columns are taken from the min-RMSD row. Row order follows first
    appearance of each group, so the collapsed frame stays aligned with the
    order sequences were sampled in -- which is what the reward tensor assumes.

    Config
    ------
    group_col      Column identifying a sequence across its samples ("name").
    metric_prefix  Prefix the metric step used ("ame").
    """
    group_col = cfg.get("group_col", "name")
    pfx = cfg.get("metric_prefix", "ame")
    rmsd_col, pass_col = f"{pfx}_motif_rmsd", f"{pfx}_motif_pass_and_no_clash"

    for col in (group_col, rmsd_col):
        if col not in df_input.columns:
            raise KeyError(
                f"{step_name}: '{col}' not in dataframe; available: "
                f"{sorted(df_input.columns)}."
            )
    if df_input[group_col].duplicated().sum() == 0:
        warnings.warn(
            f"{step_name}: every {group_col!r} is unique, so there is nothing to "
            "reduce. Set per_sample_rows: true on the af3 step (and "
            "num_diffusion_samples > 1) or drop this step."
        )
        return df_input

    d = df_input.copy()
    # NaN RMSDs are failed predictions; they must never win the idxmin, but a
    # group that is *all* failures still has to survive as a row.
    order = d[group_col].drop_duplicates().tolist()
    keep = d.loc[d.groupby(group_col)[rmsd_col].transform(
        lambda s: s.isna().all() or s.eq(s.min())).fillna(False)]
    out = keep.groupby(group_col, sort=False).head(1).set_index(group_col)

    if pass_col in d.columns:
        out[pass_col] = d.groupby(group_col)[pass_col].any()
    for c in (f"{pfx}_motif_pass", f"{pfx}_no_clash"):
        if c in d.columns:
            out[c] = d.groupby(group_col)[c].any()
    out[f"{step_name}_n_samples"] = d.groupby(group_col).size()

    return out.loc[order].reset_index()
