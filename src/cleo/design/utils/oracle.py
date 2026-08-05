"""
Structure prediction oracles for reward pipelines.

Provides step functions that write oracle-specific input files, run
predictions across available GPUs in parallel, and collect confidence
metrics back into the sequence DataFrame.

Supported oracles:
  - Boltz  (:func:`boltz_from_df`)  — https://github.com/jwohlwend/boltz
  - AF3    (:func:`af3_from_df`)    — AlphaFold 3 via Apptainer container
"""

import os
import json
import copy
import subprocess
import numpy as np
import torch
import glob
import pandas as pd
from omegaconf import DictConfig, OmegaConf

_AF3_CONF_SUFFIX = "_summary_confidences.json"


def _af3_name_and_paths_from_output_folder(folder: str) -> tuple[str, str, str] | None:
    """
    Resolve design ``name`` and file paths for one AF3 output subfolder.

    Mlfold/AF3 may rename the *directory* (e.g. filesystem-sanitize dots) while still
    writing ``<original_name>{_summary_confidences.json}`` inside. We used to use only
    ``os.path.basename(folder)`` as the key, which breaks the merge on ``name`` and
    causes every confidence file to be "missing."
    """
    conf_paths = sorted(glob.glob(os.path.join(folder, f"*{_AF3_CONF_SUFFIX}")))
    if not conf_paths:
        return None
    if len(conf_paths) > 1:
        print(
            f"Warning: multiple *_summary_confidences.json in {folder!r}, using {conf_paths[0]!r}"
        )
    conf_json_path = conf_paths[0]
    base = os.path.basename(conf_json_path)
    if not base.endswith(_AF3_CONF_SUFFIX):
        return None
    # Filename is {name}_summary_confidences.json (name may contain dots, etc.)
    name = base[: -len(_AF3_CONF_SUFFIX)]
    cif_path = os.path.join(folder, f"{name}_model.cif")
    return name, cif_path, conf_json_path


def _af3_canonical_name_map_from_input(names: pd.Series) -> dict[str, str]:
    """``casefold()`` key -> `name` as it appears in the input / FASTA."""
    m: dict[str, str] = {}
    for n in names.astype(str):
        k = n.casefold()
        if k in m and m[k] != n:
            raise ValueError(
                f"af3: two input `name` values are identical after case-folding: {m[k]!r} and {n!r}"
            )
        m[k] = n
    return m


def _af3_canonicalize_output_names_to_input(
    df_input: pd.DataFrame, df_output: pd.DataFrame
) -> pd.DataFrame:
    """
    AF3 / mlfold3 often rewrites `name` casing in paths and filenames. Merge must use the same
    strings as `df_input['name']` or the inner-join is empty. Remap the output `name` column
    to the input spelling using case-folding.
    """
    if df_output.empty or "name" not in df_output.columns:
        return df_output
    canon = _af3_canonical_name_map_from_input(df_input["name"])
    out = df_output.copy()
    keys = out["name"].astype(str).str.casefold()
    miss = ~keys.isin(canon)
    if miss.any():
        raise RuntimeError(
            "af3: some prediction `name` values are not in the input (after case-folding), "
            f"cannot merge. Unmatched: {out.loc[miss, 'name'].head(8).tolist()!r}."
        )
    out["name"] = keys.map(canon)
    return out


def _af3_should_skip_run(cfg) -> bool:
    """
    Reuse on-disk ``{rundir}/<step>/outputs`` without running AF3 (e.g. after a failed
    downstream step). Config: ``step.cfg.skip_run: true`` or
    environment ``CLEO_AF3_SKIP_RUN=1`` / ``true`` / ``yes``.
    """
    env = os.environ.get("CLEO_AF3_SKIP_RUN", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if isinstance(cfg, (dict, DictConfig)):
        v = cfg.get("skip_run", False) if cfg is not None else False
    else:
        v = False
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _local_oracle_worker_count() -> int:
    """Parallel slots used to shard Boltz/AF3 invocations. Must match :func:`run_multi_gpu_commands`.

    If PyTorch cannot use CUDA, we still use one worker (one subprocess at a time). Relying only on
    ``device_count()`` disagrees with chunking that gates on ``is_available()`` and can break when
    the driver reports devices that PyTorch cannot use.
    """
    if not torch.cuda.is_available():
        return 1
    return max(1, int(torch.cuda.device_count()))


def _require_nonempty_commands(cmd_list) -> None:
    if len(cmd_list) == 0:
        raise RuntimeError("Oracle: no shell commands to run (empty list).")


def _fail_if_multi_gpu_visible_but_cuda_unusable() -> None:
    """Fail fast on common cluster misconfig: several GPUs assigned but PyTorch cannot use CUDA."""
    if torch.cuda.is_available():
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd or cvd.strip() in ("", "NoDevFiles"):
        return
    n_vis = sum(1 for part in cvd.split(",") if part.strip())
    if n_vis > 1:
        raise RuntimeError(
            "Oracle refuses to run: multiple GPUs are visible "
            f"(CUDA_VISIBLE_DEVICES={cvd!r}) but torch.cuda.is_available() is False. "
            "Fix driver, node image, or PyTorch build before re-submitting."
        )


def run_multi_gpu_commands(cmd_list):
    """Runs commands across up to ``min(usable GPUs, len(cmd_list))`` concurrent workers."""

    _require_nonempty_commands(cmd_list)
    _fail_if_multi_gpu_visible_but_cuda_unusable()

    desired = _local_oracle_worker_count()
    if not torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(
            "WARNING: torch.cuda.device_count()>0 but is_available() is False; "
            "driver may be misconfigured. Using 1 oracle worker; check PyTorch, driver, and node."
        )
    num_workers = max(1, min(desired, len(cmd_list)))
    if len(cmd_list) < desired:
        print(
            f"Note: {len(cmd_list)} oracle job(s) for {desired} GPU slot(s); "
            "extra slot(s) unused."
        )

    print(
        f"Running with {num_workers} concurrent worker(s) "
        f"(desired_gpu_slots={desired}, torch.cuda.is_available()={torch.cuda.is_available()})"
    )

    processes = []

    for i, cmd in enumerate(cmd_list):
        gpu_idx = i % num_workers
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        print(f"[GPU {gpu_idx}] Launching: {cmd[:100]}...")
        p = subprocess.Popen(cmd, shell=True, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((p, cmd, gpu_idx))

        if len(processes) == num_workers or i == len(cmd_list) - 1:
            for p, cmd, gpu in processes:
                stdout, stderr = p.communicate()  # wait and collect output
                if p.returncode != 0:
                    print(f"[GPU {gpu}] FAILED: {cmd[:100]}...")
                    print("stdout:\n", stdout.decode())
                    print("stderr:\n", stderr.decode())
                    raise RuntimeError(f"Job failed on GPU {gpu} with code {p.returncode}")
                else:
                    print(f"[GPU {gpu}] FINISHED: {cmd[:100]}...")
            processes = []


def chunk(_list, num_chunks):
    """splits list items into num_chunks chunks as evenly as possible"""
    n = len(_list)
    q, r = divmod(n, num_chunks)

    chunks, start = [], 0
    for i in range(min(num_chunks, n)):
        end = start + q + (i < r)
        chunks.append(_list[start:end])
        start = end

    return chunks

# For running Boltz (https://github.com/jwohlwend/boltz)
def make_boltz_command(input_folder, output_folder):
    cmd = f"boltz predict {input_folder} --out_dir {output_folder};"
    return cmd

def boltz_from_df(df_input, cfg, step_name="boltz"):
    """
    Runs Boltz and aggregates base metrics from the output.
    """
    assert cfg.rundir is not None, "rundir must be specified in cfg"

    # define and create input and output directories
    outdir = f'{cfg.rundir}/{step_name}/outputs'
    inputdir = f'{cfg.rundir}/{step_name}/inputs'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(inputdir, exist_ok=True)

    # get all sequences and names from input df
    sequences = df_input['sequence'].tolist()
    names = df_input['name'].tolist()

    # load boltz template for prediction
    template_path = cfg.template_path
    boltz_template = OmegaConf.load(template_path)


    chunk_size = _local_oracle_worker_count()
    if not torch.cuda.is_available():
        print("No usable CUDA in this process; using 1 parallel Boltz job (may be very slow).")
    else:
        print(f"Sharding Boltz across {chunk_size} GPU slot(s).")


    # build list of commands to run in parallel
    cmd_list = []
    for i, seqs_chunked in enumerate(chunk([(n,s) for n,s in zip(names,sequences)], chunk_size)):
        
        batch_folder = os.path.join(inputdir, f"batch_{i:04}")
        os.makedirs(batch_folder, exist_ok=True)

        # write input files
        for j, (name, seq) in enumerate(seqs_chunked):
            _template = copy.deepcopy(boltz_template)
            _template.sequences[0].protein.sequence = seq

            file_path = os.path.join(batch_folder, f"{name}.yaml")
            OmegaConf.save(_template, file_path)

        _cmd = make_boltz_command(batch_folder, outdir)    
        cmd_list.append(_cmd)

    
    # Write jobs file (this is just for record keeping, the actual execution is done in run_multi_gpu_commands)
    jobs_file = os.path.join(cfg.rundir, f'jobs.{step_name}.list')
    if os.path.exists(jobs_file):
        os.remove(jobs_file)
    with open(jobs_file, 'w') as f:
        f.writelines([cmd + '\n' for cmd in cmd_list])
    print(f"Wrote {len(cmd_list)} commands to {jobs_file}")

    # run commands on multiple gpus
    run_multi_gpu_commands(cmd_list)

    # collect output data
    output_folders = glob.glob(os.path.join(outdir, "*", "predictions", "*"))
    assert len(output_folders) > 0, f"No output folders found in {outdir}"

    output_data = {
        "name": [],
        f"{step_name}_path": [],
        f"{step_name}_confidence_score": [],
        f"{step_name}_ptm": [],
        f"{step_name}_iptm": [],
        f"{step_name}_ligand_iptm": [],
        f"{step_name}_protein_iptm": [],
        f"{step_name}_complex_plddt": [],
        f"{step_name}_complex_iplddt": [],
        f"{step_name}_complex_pde": [],
        f"{step_name}_complex_ipde": [],
    }

    # go through each folder and extract metrics from confidence json, and path to cif file
    for folder in output_folders:
        name = os.path.basename(folder)
        cif_path = os.path.join(folder, f"{name}_model_0.cif")
        conf_json_path = os.path.join(folder, f"confidence_{name}_model_0.json")

        with open(conf_json_path, "r") as f:
            conf_data = json.load(f)

        output_data["name"].append(name)
        output_data[f"{step_name}_path"].append(cif_path)
        output_data[f"{step_name}_confidence_score"].append(conf_data["confidence_score"])
        output_data[f"{step_name}_ptm"].append(conf_data["ptm"])
        output_data[f"{step_name}_iptm"].append(conf_data["iptm"])
        output_data[f"{step_name}_ligand_iptm"].append(conf_data["ligand_iptm"])
        output_data[f"{step_name}_protein_iptm"].append(conf_data["protein_iptm"])
        output_data[f"{step_name}_complex_plddt"].append(conf_data["complex_plddt"])
        output_data[f"{step_name}_complex_iplddt"].append(conf_data["complex_iplddt"])
        output_data[f"{step_name}_complex_pde"].append(conf_data["complex_pde"])
        output_data[f"{step_name}_complex_ipde"].append(conf_data["complex_ipde"])

    df_output = pd.DataFrame(output_data)
    df_merged = pd.merge(df_input, df_output, on='name', how='inner')

    return df_merged


# For running AlphaFold 3 via Apptainer container
def make_af3_command(input_folder, output_folder, container_path, script_path, model_dir):
    cmd = (
        f"apptainer run --nv {container_path} python {script_path}"
        f" --input_dir {input_folder} --output_dir {output_folder}"
        f" --run_data_pipeline=false --model_dir={model_dir}"
        f" --num_diffusion_samples=1"
    )
    return cmd


def af3_from_df(df_input, cfg, step_name="af3"):
    """
    Runs AlphaFold 3 and aggregates confidence metrics from the output.

    If ``step.cfg.skip_run`` is true (or ``CLEO_AF3_SKIP_RUN`` is set), the container
    is not run; existing results under ``{rundir}/{step_name}/outputs/`` are read
    and merged so downstream eval steps (``dist_to_ref`` etc.) can be re-run cheaply.
    """
    assert cfg.rundir is not None, "rundir must be specified in cfg"

    outdir = f"{cfg.rundir}/{step_name}/outputs"
    inputdir = f"{cfg.rundir}/{step_name}/inputs"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(inputdir, exist_ok=True)

    skip_run = _af3_should_skip_run(cfg)
    if skip_run:
        if not os.path.isdir(outdir):
            raise FileNotFoundError(
                f"af3 skip_run: no outputs directory at {outdir!r}. "
                "Run without skip_run once, or fix rundir / step name."
            )
        print(
            f"af3: skip_run=True (reuse only); not launching the AF3/mfold3 container. "
            f"Reading metrics from {outdir!r}."
        )
    else:
        sequences = df_input["sequence"].tolist()
        names = df_input["name"].tolist()

        with open(cfg.template_path) as f:
            af3_template = json.load(f)

        chunk_size = _local_oracle_worker_count()
        if not torch.cuda.is_available():
            print("No usable CUDA in this process; using 1 parallel AF3 job (may be very slow).")
        else:
            print(f"Sharding AF3 across {chunk_size} GPU slot(s).")

        cmd_list = []
        for i, seqs_chunked in enumerate(chunk([(n, s) for n, s in zip(names, sequences)], chunk_size)):

            batch_folder = os.path.join(inputdir, f"batch_{i:04}")
            os.makedirs(batch_folder, exist_ok=True)

            for j, (name, seq) in enumerate(seqs_chunked):
                _template = copy.deepcopy(af3_template)
                _template["name"] = name
                _template["sequences"][0]["protein"]["sequence"] = seq
                _template["modelSeeds"] = [int(np.random.randint(1, 1e5))]

                file_path = os.path.join(batch_folder, f"{name}.json")
                with open(file_path, "w") as f:
                    json.dump(_template, f, indent=4)

            _cmd = make_af3_command(
                batch_folder, outdir, cfg.af3_container, cfg.af3_script, cfg.model_dir
            )
            cmd_list.append(_cmd)

        jobs_file = os.path.join(cfg.rundir, f"jobs.{step_name}.list")
        if os.path.exists(jobs_file):
            os.remove(jobs_file)
        with open(jobs_file, "w") as f:
            f.writelines([cmd + "\n" for cmd in cmd_list])
        print(f"Wrote {len(cmd_list)} commands to {jobs_file}")

        run_multi_gpu_commands(cmd_list)

    # AF3 outputs: <outdir>/<name>/<name>_model.cif and <name>_summary_confidences.json
    output_folders = glob.glob(os.path.join(outdir, "*"))
    output_folders = [f for f in output_folders if os.path.isdir(f)]
    assert len(output_folders) > 0, f"No output folders found in {outdir}"

    output_data = {
        "name": [],
        f"{step_name}_path": [],
        f"{step_name}_ptm": [],
        f"{step_name}_iptm": [],
        f"{step_name}_ranking_score": [],
        f"{step_name}_chain_ptm_protein": [],
        f"{step_name}_chain_ptm_ligand": [],
        f"{step_name}_chain_iptm_protein": [],
        f"{step_name}_chain_iptm_ligand": [],
        f"{step_name}_fraction_disordered": [],
        f"{step_name}_has_clash": [],
        f"{step_name}_interaction_pae_min": [],
    }

    for folder in output_folders:
        paths = _af3_name_and_paths_from_output_folder(folder)
        if paths is None:
            b = os.path.basename(folder)
            print(
                f"Warning: no *_summary_confidences.json in output folder {folder!r} (basename {b!r}), skipping."
            )
            continue

        name, cif_path, conf_json_path = paths
        if not os.path.exists(conf_json_path):
            # Should be redundant if paths came from a glob that includes this file
            print(f"Warning: confidence path missing for {name}, skipping.")
            continue

        with open(conf_json_path, "r") as f:
            conf_data = json.load(f)

        output_data["name"].append(name)
        output_data[f"{step_name}_path"].append(cif_path)
        output_data[f"{step_name}_ptm"].append(conf_data["ptm"])
        output_data[f"{step_name}_iptm"].append(conf_data["iptm"])
        output_data[f"{step_name}_ranking_score"].append(conf_data["ranking_score"])
        output_data[f"{step_name}_chain_ptm_protein"].append(conf_data["chain_ptm"][0])
        output_data[f"{step_name}_chain_ptm_ligand"].append(conf_data["chain_ptm"][1])
        output_data[f"{step_name}_chain_iptm_protein"].append(conf_data["chain_iptm"][0])
        output_data[f"{step_name}_chain_iptm_ligand"].append(conf_data["chain_iptm"][1])
        output_data[f"{step_name}_fraction_disordered"].append(conf_data["fraction_disordered"])
        output_data[f"{step_name}_has_clash"].append(conf_data["has_clash"])

        # Minimum PAE across off-diagonal (inter-chain) entries of chain_pair_pae_min.
        # For a 2-chain complex this is min(mat[0][1], mat[1][0]).
        pae_mat = conf_data["chain_pair_pae_min"]
        n_chains = len(pae_mat)
        off_diag = [pae_mat[i][j] for i in range(n_chains) for j in range(n_chains) if i != j]
        output_data[f"{step_name}_interaction_pae_min"].append(min(off_diag))

    df_output = pd.DataFrame(output_data)
    df_output = _af3_canonicalize_output_names_to_input(df_input, df_output)
    df_merged = pd.merge(df_input, df_output, on="name", how="inner")
    if len(df_merged) == 0 and len(df_input) > 0:
        in_names = df_input["name"].head(5).tolist()
        out_names = df_output["name"].head(5).tolist() if not df_output.empty else []
        n_dir = len([f for f in output_folders if os.path.isdir(f)])
        raise RuntimeError(
            "af3: no rows after merging predictions into the input table "
            f"(n_input={len(df_input)}, n_parsed={len(df_output)}, output_folders={n_dir}). "
            "Usually: missing or unreadable *_summary_confidences.json, a mismatch of "
            "`name` between FASTA and `*_summary_confidences.json` prefixes, or a case/ "
            "spelling change that is not fixed by case-folding. "
            f"Example input names: {in_names!r} example parsed (canonical) names: {out_names!r}."
        )

    return df_merged
