"""
Protenix v2 structure-prediction oracle for antigen-conditioned nanobody design.

Provides :func:`protenix_from_df`, a reward-pipeline step (same contract as
:func:`cleo.design.utils.oracle.boltz_from_df`) that folds each designed
nanobody together with its target antigen using Protenix v2 and returns
interface-confidence metrics for the reward:

    - interface ipTM           (chain_pair_iptm[nb][ag])      -> maximize
    - interface PAE (mean, A)  (chain_pair_pae_mean[nb][ag])  -> minimize
    - ptm / ranking_score / plddt / has_clash
    - epitope_overlap          (predicted interface vs intended epitope)

Protenix runs in its OWN virtualenv (torch 2.7 / numpy 2.x, incompatible with
cleo's env), so it is invoked as a subprocess via its console-script entrypoint
with ``PROTENIX_ROOT_DIR`` pointing at the pre-populated cache + checkpoint.
See ``SPEC.md`` and the project memory for the setup.

Per-target inputs (antigen sequence, cached antigen MSA dir, intended epitope
residues) may be supplied either in ``cfg`` (single-target, as in classic CLEO)
or per-row via DataFrame columns ``antigen_sequence`` / ``antigen_msa_dir`` /
``epitope_residues`` (forward-compatible with multi-target GRPO).

When ``use_msa`` is set, each designed chain gets a proper framework MSA rather
than a single-sequence query: supply the scaffold's CDR-gapped framework MSA dir
as ``framework_msa_dir`` and the chain's CDR spans as ``cdr_spans`` (cfg or df
column). :func:`_build_design_fw_msa` then writes a per-design a3m (designed
sequence as query + cached gapped homolog rows). This is ESSENTIAL to the
interface reward -- without it the oracle scores even known binders near-blind
(SPEC.md §5.4, calibrated 2026-07-02). The gapped framework MSA itself is
precomputed once per scaffold at dataset-build time.

Both VHH (one designed chain) and paired **Fv** (two designed chains, H + L) are
supported through one N-designed-chain path. A single-chain design uses the
scalar ``sequence`` / ``framework_msa_dir`` / ``cdr_spans`` columns; an Fv adds a
``design_chains`` column = ``[{length, framework_msa_dir, cdr_spans}, ...]`` in
chain order, which splits the decoded ``sequence`` into per-chain segments and
builds **one gapped framework MSA per chain** (H gapped at H1/2/3, L at L1/2/3).
The designed-vs-antigen interface metrics aggregate over all designed chains, and
epitope overlap counts antigen residues contacting *any* designed chain.
"""

import os
import json
import copy
import functools
import subprocess

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Input JSON construction
# ---------------------------------------------------------------------------
def _write_single_seq_msa(msa_dir: str, seq: str) -> str:
    """Write a depth-1 ``non_pairing.a3m`` (query only) so Protenix treats the
    chain as single-sequence WITHOUT triggering a remote MSA search."""
    os.makedirs(msa_dir, exist_ok=True)
    with open(os.path.join(msa_dir, "non_pairing.a3m"), "w") as f:
        f.write(f">query\n{seq}\n")
    return msa_dir


def _norm_spans(cdr_spans) -> tuple:
    """Normalize CDR spans to a hashable, sorted tuple of half-open (start, end)
    pairs. Accepts a dict ``{"H1": [s, e], ...}`` or a list ``[[s, e], ...]``."""
    pairs = cdr_spans.values() if isinstance(cdr_spans, dict) else cdr_spans
    return tuple(sorted((int(s), int(e)) for s, e in pairs))


@functools.lru_cache(maxsize=128)
def _gapped_fw_homolog_rows(fw_a3m: str, spans: tuple) -> tuple:
    """CDR-gapped HOMOLOG rows of a framework a3m (query row dropped).

    For every homolog row, characters aligned to a query CDR column are replaced
    by ``-`` and lowercase insertions falling inside a CDR span are dropped, so
    the MSA constrains only the framework while the (newly designed) CDRs stay
    unbiased. Cached per (framework a3m, CDR spans): the expensive gapping runs
    once per scaffold, not once per design. Returns ``((header, seq), ...)``.
    See ``SPEC.md`` §5.4 / §6.3 and ``bench/gap_cdrs.py`` (the validated logic).
    """
    def in_cdr(qi):
        return any(s <= qi < e for s, e in spans)

    lines = [l.rstrip("\n") for l in open(fw_a3m) if l.strip()]
    hdrs, seqs = lines[0::2], lines[1::2]
    out = []
    for h, s in zip(hdrs[1:], seqs[1:]):          # skip query row (row 0)
        qi, new = 0, []
        for ch in s:
            if ch.islower():                       # insertion: no query column
                if not in_cdr(qi):
                    new.append(ch)
                continue
            new.append("-" if in_cdr(qi) else ch)  # match column or gap
            qi += 1
        out.append((h, "".join(new)))
    return tuple(out)


def _build_design_fw_msa(msa_dir: str, fw_a3m: str, cdr_spans, designed_seq: str) -> str:
    """Assemble this design's framework MSA and write ``non_pairing.a3m``.

    Query row = the DESIGNED sequence (Protenix requires the a3m query to equal
    the folded chain); homolog rows = the scaffold's CDR-gapped framework rows
    (cached). The designed chain shares the scaffold backbone, so it has the same
    length as the framework query and only the query row changes per design.
    """
    homologs = _gapped_fw_homolog_rows(fw_a3m, _norm_spans(cdr_spans))
    if homologs:
        n_match = sum(1 for c in homologs[0][1] if not c.islower())
        assert n_match == len(designed_seq), (
            f"designed seq len {len(designed_seq)} != framework MSA match "
            f"columns {n_match} (framework a3m: {fw_a3m})")
    os.makedirs(msa_dir, exist_ok=True)
    with open(os.path.join(msa_dir, "non_pairing.a3m"), "w") as f:
        f.write(f">query\n{designed_seq}\n")
        for h, s in homologs:
            f.write(f"{h}\n{s}\n")
    return msa_dir


def _protein_chain(seq: str, chain_id: str, msa_dir, use_msa: bool) -> dict:
    """Build one Protenix ``proteinChain`` entry, attaching a precomputed MSA
    dir when MSA is enabled (real dir for the antigen, depth-1 for the binder)."""
    chain = {"sequence": seq, "count": 1, "id": [chain_id]}
    if use_msa and msa_dir is not None:
        # old-style field; Protenix auto-converts to paired/unpaired paths
        chain["msa"] = {"precomputed_msa_dir": msa_dir, "pairing_db": "uniref100"}
    return {"proteinChain": chain}


def _build_task(name, designed, ag_seq, ag_cid, ag_msa_dir, use_msa):
    """One Protenix task = N designed chains + one fixed antigen chain.

    ``designed`` = ``[(seq, chain_id, msa_dir), ...]`` — one entry per designed chain (a single
    VHH, or an Fv's H + L). Chains are ordered designed-first, antigen last, so the summary's
    chain-pair matrix indexes designed chains 0..N-1 and the antigen at N.
    """
    seqs = [_protein_chain(s, cid, m, use_msa) for (s, cid, m) in designed]
    seqs.append(_protein_chain(ag_seq, ag_cid, ag_msa_dir, use_msa))
    return {"sequences": seqs, "name": name}


def _designed_chain_ids(n: int, ag_cid: str) -> list:
    """Task-local chain ids for the N designed chains (``A, B, ...``), disjoint from the antigen.

    The framework file's own H/L labels are not reused here — Protenix ids are assigned locally
    so the antigen id can never collide with a designed chain (VHH: ``["A"]``; Fv: ``["A","B"]``)."""
    ids = [chr(ord("A") + i) for i in range(n)]
    if ag_cid in ids:
        raise ValueError(
            f"antigen_chain_id {ag_cid!r} collides with designed chain ids {ids}; "
            "set a different antigen_chain_id")
    return ids


def _split_seq(full: str, lengths: list) -> list:
    """Split the concatenated designed sequence into per-chain segments (in chain order)."""
    segs, i = [], 0
    for L in lengths:
        segs.append(full[i:i + L])
        i += L
    if i != len(full):
        raise ValueError(f"designed sequence length {len(full)} != sum of chain lengths {i}")
    return segs


# ---------------------------------------------------------------------------
# Multi-GPU subprocess launching
# ---------------------------------------------------------------------------
def _chunk(items, n):
    """Split ``items`` into at most ``n`` near-even chunks (drops empties)."""
    if n <= 0:
        n = 1
    q, r = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        end = start + q + (1 if i < r else 0)
        if end > start:
            out.append(items[start:end])
        start = end
    return out


def _run_protenix_jobs(cmds_envs):
    """Launch ``(cmd, env)`` pairs concurrently (one per GPU) and wait."""
    procs = []
    for cmd, env in cmds_envs:
        print(f"[protenix] launching on GPU {env.get('CUDA_VISIBLE_DEVICES')}: {cmd[:90]}...")
        procs.append((subprocess.Popen(cmd, shell=True, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT), cmd))
    for p, cmd in procs:
        out, _ = p.communicate()
        if p.returncode != 0:
            print(out.decode(errors="ignore"))
            raise RuntimeError(f"Protenix job failed (code {p.returncode}): {cmd[:120]}")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
def _summary_path(outdir, name, seed):
    return os.path.join(outdir, name, f"seed_{seed}", "predictions",
                        f"{name}_summary_confidence_sample_0.json")


def _cif_path(outdir, name, seed):
    return os.path.join(outdir, name, f"seed_{seed}", "predictions",
                        f"{name}_sample_0.cif")


def _parse_summary(summary, designed_idxs=(0,), ag_idx=1):
    """Pull interface + global confidence metrics out of a Protenix summary dict.

    The designed-vs-antigen interface is aggregated over all designed chains: **best** interface
    ipTM (max) and **lowest** interface PAE (min) across ``designed_idxs``. For a single-chain VHH
    (``designed_idxs=(0,)``, ``ag_idx=1``) this is exactly the old ``chain_pair[0][1]`` value; for
    an Fv it takes whichever of H/L makes the stronger antigen contact.
    """
    cp_iptm = summary.get("chain_pair_iptm")
    cp_pae = summary.get("chain_pair_pae_mean")
    return {
        "iptm": summary.get("iptm"),
        "ptm": summary.get("ptm"),
        "plddt": summary.get("plddt"),
        "ranking_score": summary.get("ranking_score"),
        "has_clash": int(bool(summary.get("has_clash", 0))),
        "interface_iptm": max(cp_iptm[d][ag_idx] for d in designed_idxs) if cp_iptm else None,
        "interface_pae": min(cp_pae[d][ag_idx] for d in designed_idxs) if cp_pae else None,
    }


def _epitope_overlap(cif_file, ag_cid, nb_cids, epitope_residues, cutoff=5.0):
    """Fraction of the predicted antigen interface that lands on the intended
    epitope (precision). Returns NaN if it cannot be computed.

    predicted interface = antigen residues with any atom within ``cutoff`` of any atom of **any**
    designed chain (``nb_cids`` — one for a VHH, both H and L for an Fv), in the Protenix-predicted
    structure.
    """
    try:
        import gemmi
    except Exception:
        return float("nan")
    try:
        st = gemmi.read_structure(cif_file)
        model = st[0]
        nb_names = {c for c in nb_cids if c in model}
        ag = model[ag_cid] if ag_cid in model else None
        if not nb_names or ag is None:
            # fall back to ordering: last chain = antigen, all others = designed
            chains = list(model)
            if len(chains) < 2:
                return float("nan")
            ag = chains[-1]
            nb_names = {c.name for c in chains[:-1]}

        ns = gemmi.NeighborSearch(model, st.cell, cutoff).populate()
        intended = set(int(r) for r in epitope_residues)
        predicted = set()
        for res in ag:
            hit = False
            for atom in res:
                marks = ns.find_atoms(atom.pos, "\0", radius=cutoff)
                for m in marks:
                    if m.to_cra(model).chain.name in nb_names:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                predicted.add(res.seqid.num)
        if not predicted:
            return 0.0
        return len(predicted & intended) / len(predicted)
    except Exception as e:
        print(f"[protenix] epitope_overlap failed for {cif_file}: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Main reward step
# ---------------------------------------------------------------------------
def protenix_from_df(df_input, cfg, step_name="protenix"):
    """Run Protenix v2 on (designed nanobody + antigen) and merge interface
    metrics into the DataFrame. See module docstring for cfg fields."""
    assert cfg.rundir is not None, "cfg.rundir must be set (UniversalReward sets this)"

    # --- resolve config (with sensible defaults) ---------------------------
    protenix_bin = cfg.get("protenix_bin", "/home/jgershon/git/Protenix/.venv/bin/protenix")
    protenix_root = cfg.get("protenix_root_dir", "/home/jgershon/git/protenix-data")
    model_name = cfg.get("model_name", "protenix-v2")
    seed = int(cfg.get("seed", 101))
    cycle = int(cfg.get("cycle", 4))
    n_step = int(cfg.get("n_diffusion_step", 20))   # NB: cfg.step is reserved by UniversalReward
    n_sample = int(cfg.get("n_sample", 1))
    use_msa = bool(cfg.get("use_msa", False))
    dtype = cfg.get("dtype", "bf16")
    triatt = cfg.get("triatt_kernel", "cuequivariance")
    trimul = cfg.get("trimul_kernel", "cuequivariance")
    ag_cid = cfg.get("antigen_chain_id", "L")
    cutoff = float(cfg.get("contact_cutoff", 5.0))

    workdir = os.path.join(cfg.rundir, step_name)
    indir = os.path.join(workdir, "inputs")
    outdir = os.path.join(workdir, "outputs")
    msadir = os.path.join(workdir, "msa")
    os.makedirs(indir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    names = df_input["name"].tolist()
    nb_seqs = df_input["sequence"].tolist()

    def _row_val(i, col, default):
        return df_input[col].iloc[i] if col in df_input.columns else cfg.get(col, default)

    # --- build one Protenix task per design --------------------------------
    tasks, meta = [], {}
    for i, (name, nb_seq) in enumerate(zip(names, nb_seqs)):
        ag_seq = _row_val(i, "antigen_sequence", None)
        assert ag_seq is not None, "antigen_sequence must be in cfg or df"
        ag_msa = _row_val(i, "antigen_msa_dir", cfg.get("antigen_msa_dir", None))
        epi = _row_val(i, "epitope_residues", cfg.get("epitope_residues", None))

        # Per-designed-chain (framework_msa_dir, cdr_spans, sequence segment). Multi-chain Fv
        # supplies ``design_chains`` = [{length, framework_msa_dir, cdr_spans}, ...] (chain order);
        # a single-chain VHH/legacy design falls back to the scalar columns.
        spec = _row_val(i, "design_chains", cfg.get("design_chains", None))
        if spec:
            chain_specs = json.loads(spec) if isinstance(spec, str) else spec
            segs = _split_seq(nb_seq, [int(c["length"]) for c in chain_specs])
            fw_spans = [(c.get("framework_msa_dir"), c.get("cdr_spans")) for c in chain_specs]
        else:
            segs = [nb_seq]
            fw_spans = [(_row_val(i, "framework_msa_dir", cfg.get("framework_msa_dir", None)),
                         _row_val(i, "cdr_spans", cfg.get("cdr_spans", None)))]

        cids = _designed_chain_ids(len(segs), ag_cid)
        designed = []
        for j, (seg, (fw, spans)) in enumerate(zip(segs, fw_spans)):
            msa = None
            if use_msa:
                if fw is not None and spans is not None:
                    # CDR-gapped framework MSA per chain (H gapped at H1/2/3, L at L1/2/3);
                    # query row = this chain's designed segment (Protenix requires it).
                    msa = _build_design_fw_msa(
                        os.path.join(msadir, f"{name}_d{j}"),
                        os.path.join(fw, "non_pairing.a3m"), spans, seg)
                else:
                    # no framework MSA -> single-seq (reward near-blind; SPEC §5.4, 2026-07-02)
                    msa = _write_single_seq_msa(os.path.join(msadir, f"{name}_d{j}"), seg)
            designed.append((seg, cids[j], msa))
        if use_msa and ag_msa is None:
            ag_msa = _write_single_seq_msa(os.path.join(msadir, f"{name}_ag"), ag_seq)

        tasks.append(_build_task(name, designed, ag_seq, ag_cid, ag_msa, use_msa))
        meta[name] = {"epitope": epi, "nb_cids": cids, "n_designed": len(designed)}

    # --- split across GPUs, one input JSON + one protenix process per GPU ---
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    base_env = dict(os.environ)
    base_env["PROTENIX_ROOT_DIR"] = protenix_root
    base_env.setdefault("PYTHONPATH", "")

    cmds_envs = []
    for g, group in enumerate(_chunk(tasks, n_gpus)):
        in_json = os.path.join(indir, f"batch_{g:02}.json")
        with open(in_json, "w") as f:
            json.dump(group, f)
        cmd = (
            f"{protenix_bin} pred -i {in_json} -o {outdir} "
            f"-s {seed} -n {model_name} -c {cycle} -p {n_step} -e {n_sample} "
            f"-d {dtype} --use_template false --use_msa {str(use_msa).lower()} "
            f"--triatt_kernel {triatt} --trimul_kernel {trimul}"
        )
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(g % n_gpus)
        cmds_envs.append((cmd, env))

    _run_protenix_jobs(cmds_envs)

    # --- collect metrics ----------------------------------------------------
    rows = []
    for name in names:
        sp = _summary_path(outdir, name, seed)
        rec = {"name": name}
        if os.path.exists(sp):
            with open(sp) as f:
                summary = json.load(f)
            n_des = meta[name]["n_designed"]
            rec.update(_parse_summary(summary, designed_idxs=range(n_des), ag_idx=n_des))
            cif = _cif_path(outdir, name, seed)
            rec[f"{step_name}_path"] = cif
            epi = meta[name]["epitope"]
            if epi is not None and os.path.exists(cif):
                rec["epitope_overlap"] = _epitope_overlap(
                    cif, ag_cid, meta[name]["nb_cids"], epi, cutoff)
        else:
            print(f"[protenix] WARNING: no summary for {name} at {sp}")
            rec.update({k: float("nan") for k in
                        ["iptm", "ptm", "plddt", "ranking_score", "has_clash",
                         "interface_iptm", "interface_pae"]})
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    # prefix metric columns with the step name (e.g. protenix_interface_iptm)
    df_out = df_out.rename(columns={
        c: f"{step_name}_{c}" for c in df_out.columns
        if c not in ("name", f"{step_name}_path")
    })
    return pd.merge(df_input, df_out, on="name", how="inner")
