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

When ``use_msa`` is set, the designed (antibody) chain gets a proper framework
MSA rather than a single-sequence query: supply the scaffold's CDR-gapped
framework MSA dir as ``framework_msa_dir`` and the chain's CDR spans as
``cdr_spans`` (cfg or df column). :func:`_build_design_fw_msa` then writes a
per-design a3m (designed sequence as query + cached gapped homolog rows). This
is ESSENTIAL to the interface reward -- without it the oracle scores even known
binders near-blind (SPEC.md §5.4, calibrated 2026-07-02). The gapped framework
MSA itself is precomputed once per scaffold at dataset-build time.
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


def _build_task(name, nb_seq, ag_seq, nb_cid, ag_cid, nb_msa_dir, ag_msa_dir, use_msa):
    """One Protenix task = designed nanobody chain + fixed antigen chain."""
    return {
        "sequences": [
            _protein_chain(nb_seq, nb_cid, nb_msa_dir, use_msa),
            _protein_chain(ag_seq, ag_cid, ag_msa_dir, use_msa),
        ],
        "name": name,
    }


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


def _parse_summary(summary, nb_idx=0, ag_idx=1):
    """Pull interface + global confidence metrics out of a Protenix summary dict."""
    cp_iptm = summary.get("chain_pair_iptm")
    cp_pae = summary.get("chain_pair_pae_mean")
    return {
        "iptm": summary.get("iptm"),
        "ptm": summary.get("ptm"),
        "plddt": summary.get("plddt"),
        "ranking_score": summary.get("ranking_score"),
        "has_clash": int(bool(summary.get("has_clash", 0))),
        "interface_iptm": cp_iptm[nb_idx][ag_idx] if cp_iptm else None,
        "interface_pae": cp_pae[nb_idx][ag_idx] if cp_pae else None,
    }


def _epitope_overlap(cif_file, ag_cid, nb_cid, epitope_residues, cutoff=5.0):
    """Fraction of the predicted antigen interface that lands on the intended
    epitope (precision). Returns NaN if it cannot be computed.

    predicted interface = antigen residues with any atom within ``cutoff`` of
    any nanobody atom in the Protenix-predicted structure.
    """
    try:
        import gemmi
    except Exception:
        return float("nan")
    try:
        st = gemmi.read_structure(cif_file)
        model = st[0]
        nb = model[nb_cid] if nb_cid in model else None
        ag = model[ag_cid] if ag_cid in model else None
        if nb is None or ag is None:
            # fall back to ordering: chain 0 = nanobody, chain 1 = antigen
            chains = list(model)
            if len(chains) < 2:
                return float("nan")
            nb, ag = chains[0], chains[1]

        ns = gemmi.NeighborSearch(model, st.cell, cutoff).populate()
        intended = set(int(r) for r in epitope_residues)
        predicted = set()
        for res in ag:
            hit = False
            for atom in res:
                marks = ns.find_atoms(atom.pos, "\0", radius=cutoff)
                for m in marks:
                    cra = m.to_cra(model)
                    if cra.chain.name == nb.name:
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
    nb_cid = cfg.get("nanobody_chain_id", "A")
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

        # designed-chain framework MSA (per example): CDR-gapped scaffold MSA
        # dir + this chain's CDR spans -> per-design a3m (query-row swap).
        nb_fw_msa = _row_val(i, "framework_msa_dir", cfg.get("framework_msa_dir", None))
        cdr_spans = _row_val(i, "cdr_spans", cfg.get("cdr_spans", None))

        nb_msa = None
        if use_msa:
            if nb_fw_msa is not None and cdr_spans is not None:
                nb_msa = _build_design_fw_msa(
                    os.path.join(msadir, f"{name}_nb"),
                    os.path.join(nb_fw_msa, "non_pairing.a3m"),
                    cdr_spans, nb_seq,
                )
            else:
                # no framework MSA supplied -> single-seq (reward is near-blind;
                # see SPEC.md §5.4, calibration 2026-07-02)
                nb_msa = _write_single_seq_msa(os.path.join(msadir, f"{name}_nb"), nb_seq)
            if ag_msa is None:
                ag_msa = _write_single_seq_msa(os.path.join(msadir, f"{name}_ag"), ag_seq)

        tasks.append(_build_task(name, nb_seq, ag_seq, nb_cid, ag_cid, nb_msa, ag_msa, use_msa))
        meta[name] = {"epitope": epi}

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
            rec.update(_parse_summary(summary))
            cif = _cif_path(outdir, name, seed)
            rec[f"{step_name}_path"] = cif
            epi = meta[name]["epitope"]
            if epi is not None and os.path.exists(cif):
                rec["epitope_overlap"] = _epitope_overlap(cif, ag_cid, nb_cid, epi, cutoff)
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
