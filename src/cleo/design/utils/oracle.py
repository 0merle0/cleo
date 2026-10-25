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
import re
import json
import copy
import subprocess
import time
import numpy as np
import torch
import glob
import pandas as pd
from omegaconf import DictConfig, OmegaConf

_AF3_CONF_SUFFIX = "_summary_confidences.json"


def _min_not_none(values):
    """min() over values, ignoring None; NaN if nothing is left.

    AF3 confidence summaries carry nulls for chains it cannot score pairwise
    (a lone metal ion, for instance). Returning NaN keeps the row -- losing an
    otherwise good prediction to a missing diagnostic would be worse than
    reporting the diagnostic as absent.
    """
    vals = [v for v in values if v is not None]
    return min(vals) if vals else float("nan")


_AF3_SAMPLE_DIR_RE = re.compile(r"seed-(\d+)_sample-(\d+)$")


def _af3_sample_paths_from_output_folder(folder: str):
    """
    Resolve every diffusion sample written for one AF3 design folder.

    With ``--num_diffusion_samples=N`` AF3 writes ``seed-<S>_sample-<K>/`` for
    K in 0..N-1 alongside the top-level files, which hold whichever sample won
    on ranking score. Scoring against the RFdiffusion2 benchmark needs all N,
    not the winner: their success criterion is best-of-40 over 8 sequences x 5
    Chai models, so a single prediction per sequence is a strictly harsher unit
    and the two rates are not comparable.

    Returns ``[(name, sample_index, seed, cif_path, conf_json_path), ...]``
    sorted by sample index, or ``[]`` if the folder has no sample subdirectories
    (``num_diffusion_samples=1`` runs predating this, which keep only the
    top-level files -- callers fall back to
    ``_af3_name_and_paths_from_output_folder`` there).
    """
    out = []
    for sub in sorted(glob.glob(os.path.join(folder, "seed-*_sample-*"))):
        m = _AF3_SAMPLE_DIR_RE.search(os.path.basename(sub))
        if not m or not os.path.isdir(sub):
            continue
        seed, sample = int(m.group(1)), int(m.group(2))
        conf_paths = sorted(glob.glob(os.path.join(sub, f"*{_AF3_CONF_SUFFIX}")))
        if not conf_paths:
            continue
        conf_json_path = conf_paths[0]
        base = os.path.basename(conf_json_path)[: -len(_AF3_CONF_SUFFIX)]
        cif_path = os.path.join(sub, f"{base}_model.cif")
        if not os.path.exists(cif_path):
            continue
        # `base` carries the seed/sample suffix; the design name does not.
        name = base[: -len(f"_seed-{seed}_sample-{sample}")]
        out.append((name, sample, seed, cif_path, conf_json_path))
    return sorted(out, key=lambda r: r[1])


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


def _run_commands_once(cmd_list):
    """Run every command across the GPU slots. -> list of (cmd, code, stderr).

    Returns the failures instead of raising on the first one. The previous
    version raised from inside the wait loop, which left the sibling processes
    of that wave running with nobody to reap them -- and, more importantly, made
    a single bad exit indistinguishable from a bad batch. Collecting them lets
    the caller retry only what actually failed.
    """
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

    failures = []
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
            # Drain the whole wave before reacting to any failure, so a bad
            # exit never strands its siblings.
            for p, cmd, gpu in processes:
                stdout, stderr = p.communicate()  # wait and collect output
                if p.returncode != 0:
                    err = stderr.decode(errors="replace")
                    print(f"[GPU {gpu}] FAILED: {cmd[:100]}...")
                    print("stdout:\n", stdout.decode(errors="replace"))
                    print("stderr:\n", err)
                    failures.append((cmd, p.returncode, err))
                else:
                    print(f"[GPU {gpu}] FINISHED: {cmd[:100]}...")
            processes = []

    return failures


def run_multi_gpu_commands(cmd_list, attempts=3):
    """Runs commands across up to ``min(usable GPUs, len(cmd_list))`` concurrent workers.

    Retries transient infrastructure faults, for the reason `rf3_from_df` and
    `protenix_from_df` already do: a training run is thousands of these calls
    over many hours, and one non-zero exit used to kill the whole run. Job
    29184308 (M0255, fragb32) died 16 h in on `Job failed on GPU 0 with code 1`
    out of `boltz_from_df` -- a transient GPU fault that said nothing about the
    designs. Because the two other oracles in that round robin retried and Boltz
    did not, one predictor in three could end a multi-day run on its own.

    Only the commands that failed are retried; the ones that already wrote their
    output are not re-run.

    Deliberately bounded, as in `rf3_from_df`: a genuine configuration error
    fails identically every time, so after `attempts` passes the error is raised
    rather than looped on.
    """
    _require_nonempty_commands(cmd_list)
    _fail_if_multi_gpu_visible_but_cuda_unusable()

    attempts = max(1, int(attempts))
    pending = list(cmd_list)

    for attempt in range(1, attempts + 1):
        failures = _run_commands_once(pending)
        if not failures:
            return
        if attempt == attempts:
            cmd, code, err = failures[0]
            raise RuntimeError(
                f"{len(failures)}/{len(pending)} oracle job(s) failed after "
                f"{attempt} attempt(s); first exited {code}: {cmd[:200]}\n{err[-2000:]}"
            )
        wait = 30 * attempt
        print(f"oracle: {len(failures)}/{len(pending)} job(s) failed on attempt "
              f"{attempt}/{attempts}; retrying those in {wait}s")
        time.sleep(wait)
        pending = [c for c, _, _ in failures]


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
def make_boltz_command(input_folder, output_folder, diffusion_samples=1,
                       no_kernels=True):
    """Build the Boltz CLI call.

    ``no_kernels`` defaults True because Boltz's optional cuequivariance kernels
    are not installed in this project's environment -- without the flag it dies
    with ``ModuleNotFoundError: No module named 'cuequivariance_torch'`` after
    loading the model. The alternative is ``uv sync --extra cuda``
    (pyproject declares ``cuda = ["boltz[cuda]"]``), which is faster but mutates
    the shared environment every other run depends on; set
    ``boltz_no_kernels: false`` once that extra is installed.
    """
    cmd = (f"boltz predict {input_folder} --out_dir {output_folder}"
           f" --diffusion_samples {int(diffusion_samples)}")
    if no_kernels:
        cmd += " --no_kernels"
    return cmd + ";"

def _boltz_input_from_af3_template(af3_template: dict, seq: str) -> dict:
    """Translate one AF3 input JSON into Boltz's YAML schema (as a dict).

    Fourth translation of the same template, for the reason the other three
    exist: a cross-oracle comparison only means something if every predictor
    folds the same complex. Boltz previously required its own hand-written YAML,
    and the AME set had 12 of them against 13 backbones -- M0255 had none, so a
    multi-oracle run there would have failed on whichever step drew Boltz.

    AF3 nests entities under ``sequences`` as ``{"protein": {...}}`` /
    ``{"ligand": {"ccdCodes": [...]}}``. Boltz uses the same ``sequences`` key
    with ``{"protein": {"id", "sequence", "msa"}}`` and
    ``{"ligand": {"id", "ccd"}}`` -- one CCD code per entity, so a multi-code
    AF3 ligand becomes several entities with suffixed chain ids.

    ``msa: empty`` is set explicitly: that is Boltz's spelling for a
    single-sequence prediction, matching --use_msa False (Protenix),
    run_data_pipeline=false (AF3) and RF3's no-MSA default.
    """
    sequences = []
    for entity in af3_template["sequences"]:
        if "protein" in entity:
            sequences.append({"protein": {"id": entity["protein"]["id"],
                                          "sequence": seq, "msa": "empty"}})
        elif "ligand" in entity:
            ccd_codes = entity["ligand"].get("ccdCodes")
            if not ccd_codes:
                raise ValueError(
                    f"boltz: ligand entity {entity['ligand'].get('id')!r} in the AF3 template "
                    "has no ccdCodes. SMILES and file-based ligands are not translated yet; add "
                    "them here rather than letting the ligand silently vanish from the complex."
                )
            base_id = entity["ligand"]["id"]
            for j, ccd in enumerate(ccd_codes):
                sequences.append({"ligand": {"id": base_id if j == 0 else f"{base_id}{j}",
                                             "ccd": ccd}})
        else:
            raise ValueError(
                f"boltz: unrecognised AF3 entity {sorted(entity)!r}; expected "
                "'protein' or 'ligand'."
            )
    return {"version": 1, "sequences": sequences}


def boltz_from_df(df_input, cfg, step_name="boltz"):
    """Runs Boltz and aggregates base metrics from the output.

    Honours ``num_diffusion_samples`` (and ``per_sample_rows``) the same way
    ``af3_from_df`` and ``rf3_from_df`` do. Until this was added Boltz silently
    ran one sample and read only ``model_0``, so a config asking all three
    oracles for best-of-5 got best-of-5 from two of them and a single draw from
    Boltz -- a reward that means something different depending on which oracle
    scored the step.
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
    # Accept either Boltz's own YAML or the authoritative AF3 JSON. The JSON
    # path is preferred: it is the one template every oracle translates from, so
    # no backbone can have a Boltz complex that differs from its AF3 one.
    template_path = cfg.template_path
    if str(template_path).endswith(".json"):
        with open(template_path) as f:
            _af3 = json.load(f)
        boltz_template = OmegaConf.create(_boltz_input_from_af3_template(_af3, ""))
        _protein_idx = next(i for i, e in enumerate(boltz_template.sequences)
                            if "protein" in e)
    else:
        boltz_template = OmegaConf.load(template_path)
        _protein_idx = 0

    # Default 5, matching af3_from_df and rf3_from_df. Best-of-5 is the AME
    # benchmark's own unit, and an oracle that quietly ran 1 sample while the
    # others ran 5 would make the reward mean different things per step.
    n_samples = int(cfg.get("num_diffusion_samples", 5))
    per_sample_rows = bool(cfg.get("per_sample_rows", n_samples > 1))


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
            _template.sequences[_protein_idx].protein.sequence = seq

            file_path = os.path.join(batch_folder, f"{name}.yaml")
            OmegaConf.save(_template, file_path)

        _cmd = make_boltz_command(batch_folder, outdir, n_samples,
                                  no_kernels=bool(cfg.get("boltz_no_kernels", True)))
        cmd_list.append(_cmd)

    
    # Write jobs file (this is just for record keeping, the actual execution is done in run_multi_gpu_commands)
    jobs_file = os.path.join(cfg.rundir, f'jobs.{step_name}.list')
    if os.path.exists(jobs_file):
        os.remove(jobs_file)
    with open(jobs_file, 'w') as f:
        f.writelines([cmd + '\n' for cmd in cmd_list])
    print(f"Wrote {len(cmd_list)} commands to {jobs_file}")

    # run commands on multiple gpus
    # `boltz_retries` mirrors `rf3_retries` and `protenix_retries`; see
    # run_multi_gpu_commands for why one non-zero exit used to end a run.
    run_multi_gpu_commands(cmd_list, attempts=int(cfg.get("boltz_retries", 3)))

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
        # One row per diffusion sample when asked for, so a downstream
        # best-of-N step sees every sample. A missing model is skipped rather
        # than faked: Boltz can write fewer models than requested.
        idxs = range(n_samples) if per_sample_rows else [0]
        for k in idxs:
            cif_path = os.path.join(folder, f"{name}_model_{k}.cif")
            conf_json_path = os.path.join(folder, f"confidence_{name}_model_{k}.json")
            if not (os.path.exists(cif_path) and os.path.exists(conf_json_path)):
                continue
            _collect_boltz_row(output_data, step_name, name, cif_path, conf_json_path)

    df_output = pd.DataFrame(output_data)
    if per_sample_rows:
        df_output[f"{step_name}_sample"] = df_output.groupby("name").cumcount()
    df_merged = pd.merge(df_input, df_output, on="name", how="inner")
    return df_merged


def _collect_boltz_row(output_data, step_name, name, cif_path, conf_json_path):
    """Append one Boltz prediction's metrics to the column dict."""
    if True:
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


# For running AlphaFold 3 via Apptainer container
def make_af3_command(input_folder, output_folder, container_path, script_path, model_dir,
                     num_diffusion_samples=1):
    cmd = (
        f"apptainer run --nv {container_path} python {script_path}"
        f" --input_dir {input_folder} --output_dir {output_folder}"
        f" --run_data_pipeline=false --model_dir={model_dir}"
        f" --num_diffusion_samples={int(num_diffusion_samples)}"
    )
    return cmd


def af3_from_df(df_input, cfg, step_name="af3"):
    """
    Runs AlphaFold 3 and aggregates confidence metrics from the output.

    If ``step.cfg.skip_run`` is true (or ``CLEO_AF3_SKIP_RUN`` is set), the container
    is not run; existing results under ``{rundir}/{step_name}/outputs/`` are read
    and merged so downstream eval steps (``dist_to_ref`` etc.) can be re-run cheaply.

    ``num_diffusion_samples`` (**default 5**) folds each sequence N times in one
    AF3 call. With ``per_sample_rows`` also set (**default true**), the returned
    frame carries one row per (sequence, sample) with ``{step_name}_sample`` and
    ``{step_name}_seed`` columns, so a downstream metric step scores all N and
    the caller aggregates (min RMSD, any-pass) itself. Left off, only the
    top-ranked sample is returned and the row count is unchanged.

    Both defaults changed from ``1``/``false`` on 2026-08-16. Best-of-5 is the
    benchmark's own unit -- RFdiffusion2 counts a design as a success if any of
    its predictions clears the cutoff -- so a single-sample run measures
    something the benchmark does not, and the two are not comparable. Configs
    predating the change are left as they are, since they record what was
    actually run; re-running one now picks up best-of-5 and
    :class:`UniversalReward` will refuse it until a ``best_of_n_from_df`` step is
    added to collapse the rows. Set ``num_diffusion_samples: 1`` to opt out
    deliberately.
    """
    assert cfg.rundir is not None, "rundir must be specified in cfg"
    n_samples = int(cfg.get("num_diffusion_samples", 5))
    per_sample_rows = bool(cfg.get("per_sample_rows", n_samples > 1))
    if per_sample_rows and n_samples < 2:
        print(f"af3: per_sample_rows with num_diffusion_samples={n_samples} is a no-op.")

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
                batch_folder, outdir, cfg.af3_container, cfg.af3_script, cfg.model_dir,
                num_diffusion_samples=n_samples,
            )
            cmd_list.append(_cmd)

        jobs_file = os.path.join(cfg.rundir, f"jobs.{step_name}.list")
        if os.path.exists(jobs_file):
            os.remove(jobs_file)
        with open(jobs_file, "w") as f:
            f.writelines([cmd + "\n" for cmd in cmd_list])
        print(f"Wrote {len(cmd_list)} commands to {jobs_file}")

        run_multi_gpu_commands(cmd_list, attempts=int(cfg.get("af3_retries", 3)))

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

    if per_sample_rows:
        output_data[f"{step_name}_sample"] = []
        output_data[f"{step_name}_seed"] = []

    # One (name, cif, conf) record per row we intend to emit.
    records = []
    for folder in output_folders:
        samples = _af3_sample_paths_from_output_folder(folder) if per_sample_rows else []
        if samples:
            records.extend(
                (name, cif, conf, sample, seed)
                for name, sample, seed, cif, conf in samples
            )
            continue
        if per_sample_rows:
            # Asked for per-sample rows but the folder has no sample subdirs.
            # Falling through to the top-level file would silently emit 1 row
            # where N are expected, so the aggregation downstream would be a
            # best-of-1 wearing a best-of-N label.
            print(
                f"Warning: no seed-*_sample-* subfolders in {folder!r}; "
                "falling back to the top-ranked prediction for this design."
            )
        paths = _af3_name_and_paths_from_output_folder(folder)
        if paths is None:
            b = os.path.basename(folder)
            print(
                f"Warning: no *_summary_confidences.json in output folder {folder!r} (basename {b!r}), skipping."
            )
            continue
        records.append((*paths, 0, -1))

    for name, cif_path, conf_json_path, sample_idx, seed in records:
        if not os.path.exists(conf_json_path):
            # Should be redundant if paths came from a glob that includes this file
            print(f"Warning: confidence path missing for {name}, skipping.")
            continue

        with open(conf_json_path, "r") as f:
            conf_data = json.load(f)

        output_data["name"].append(name)
        if per_sample_rows:
            output_data[f"{step_name}_sample"].append(sample_idx)
            output_data[f"{step_name}_seed"].append(seed)
        output_data[f"{step_name}_path"].append(cif_path)
        output_data[f"{step_name}_ptm"].append(conf_data["ptm"])
        output_data[f"{step_name}_iptm"].append(conf_data["iptm"])
        output_data[f"{step_name}_ranking_score"].append(conf_data["ranking_score"])
        # Chain 0 is the protein; every other chain is a ligand entity. With a
        # single ligand this is exactly the old `[1]` behaviour; with several
        # (metal + cofactor + substrate) it summarises the worst-confidence
        # ligand rather than silently reporting only chain B.
        output_data[f"{step_name}_chain_ptm_protein"].append(conf_data["chain_ptm"][0])
        output_data[f"{step_name}_chain_ptm_ligand"].append(
            _min_not_none(conf_data["chain_ptm"][1:]))
        output_data[f"{step_name}_chain_iptm_protein"].append(conf_data["chain_iptm"][0])
        output_data[f"{step_name}_chain_iptm_ligand"].append(
            _min_not_none(conf_data["chain_iptm"][1:]))
        output_data[f"{step_name}_fraction_disordered"].append(conf_data["fraction_disordered"])
        output_data[f"{step_name}_has_clash"].append(conf_data["has_clash"])

        # Minimum PAE across off-diagonal (inter-chain) entries of chain_pair_pae_min.
        # For a 2-chain complex this is min(mat[0][1], mat[1][0]).
        #
        # Entries can be null. AF3 emits an all-None row for chains it cannot
        # score pairwise -- single-atom ligands such as a lone metal ion. Any
        # target with a metal plus a cofactor hits this, so the Nones must be
        # dropped rather than passed to min(), which raises on None.
        pae_mat = conf_data["chain_pair_pae_min"]
        n_chains = len(pae_mat)
        off_diag = [pae_mat[i][j] for i in range(n_chains) for j in range(n_chains) if i != j]
        output_data[f"{step_name}_interaction_pae_min"].append(_min_not_none(off_diag))

    df_output = pd.DataFrame(output_data)
    df_output = _af3_canonicalize_output_names_to_input(df_input, df_output)
    if per_sample_rows and not df_output.empty:
        # Fans df_input out to one row per (sequence, diffusion sample). Sorted
        # so the row order is reproducible across runs rather than glob order.
        df_output = df_output.sort_values(["name", f"{step_name}_sample"])
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


# --------------------------------------------------------------------------- #
# RoseTTAFold3
# --------------------------------------------------------------------------- #

RF3_CONTAINER = "/net/software/containers/versions/modelhub_inference/rf3_5.sif"


def _rf3_input_from_af3_template(af3_template: dict, name: str, seq: str) -> dict:
    """Translate one AF3 input JSON into RF3's schema.

    The two formats carry the same information in different shapes. AF3 nests
    entities under ``sequences`` as ``{"protein": {...}}`` / ``{"ligand": {...}}``;
    RF3 takes a flat ``components`` list where a protein is ``{"seq", "chain_id"}``
    and a ligand is ``{"ccd_code", "chain_id"}``.

    Sharing one template between the two oracles is the point: a cross-oracle
    comparison is only meaningful if both are folding the same complex, and two
    hand-maintained templates drift. AF3 template stays authoritative.

    Ligands are passed as CCD codes, which RF3's docs rank above SMILES and
    CIF/SDF for accuracy. An AF3 ligand entity may name several CCD codes under
    one chain id; each becomes its own component, since RF3 keys components to a
    single code.

    MSA fields are dropped. The AME templates carry empty ``unpairedMsa`` /
    ``pairedMsa``, so these are single-sequence predictions on both sides.
    """
    components = []
    for entity in af3_template["sequences"]:
        if "protein" in entity:
            components.append({"seq": seq, "chain_id": entity["protein"]["id"]})
        elif "ligand" in entity:
            ccd_codes = entity["ligand"].get("ccdCodes")
            if not ccd_codes:
                raise ValueError(
                    f"rf3: ligand entity {entity['ligand'].get('id')!r} in the AF3 template has "
                    "no ccdCodes. SMILES and file-based ligands are not translated yet; add them "
                    "here rather than letting the ligand silently vanish from the complex."
                )
            for ccd in ccd_codes:
                components.append({"ccd_code": ccd, "chain_id": entity["ligand"]["id"]})
        else:
            raise ValueError(
                f"rf3: unrecognised AF3 entity {sorted(entity)!r}; expected 'protein' or 'ligand'."
            )
    return {"name": name, "components": components}


def rf3_from_df(df_input, cfg, step_name="rf3"):
    """Runs RoseTTAFold3 and aggregates confidence metrics from the output.

    Same contract as :func:`af3_from_df` -- takes ``name``/``sequence``, returns
    the input frame with ``{step_name}_path`` and confidence columns merged on
    ``name`` -- so ``rfd2_metrics_from_df`` consumes it by setting
    ``structure_col: rf3_path`` and nothing else changes.

    The reason to have it: training reward and evaluation currently come from
    the same predictor, so a gain is not separable from having learned that
    predictor's quirks. Optimising under RF3 and reporting under AF3 makes the
    two independent.

    Config
    ------
    rundir          Where inputs/ and outputs/ are written (required).
    template_path   AF3-format JSON template; translated to RF3's schema.
    rf3_container   Apptainer image (defaults to the IPD modelhub build).
    ckpt_path       Override RF3's default checkpoint.
    num_diffusion_samples   Samples per design (default 5, matching AF3).
    per_sample_rows One row per (design, sample) (default true when N > 1).
    early_stopping_plddt_threshold  **Default 0.0, which disables it.**
    num_steps       Diffusion steps (RF3's default is 50; AF3 uses 200).
    skip_run        Reuse an existing outputs/ instead of folding again.

    Note on ``early_stopping_plddt_threshold``: RF3 ships with 0.5, which makes
    it abandon a prediction it judges unpromising and write a
    ``summary_confidences.json`` with no structure beside it. That is a sensible
    default for screening and the wrong one for a reward, because the abandoned
    designs are exactly the failures the policy has to be told about -- they
    would return with no coordinates and drop out of the batch, biasing reward
    toward whatever RF3 was already confident about. Disabled here unless a
    config asks for it back.
    """
    assert cfg.rundir is not None, "rundir must be specified in cfg"
    n_samples = int(cfg.get("num_diffusion_samples", 5))
    per_sample_rows = bool(cfg.get("per_sample_rows", n_samples > 1))

    outdir = f"{cfg.rundir}/{step_name}/outputs"
    inputdir = f"{cfg.rundir}/{step_name}/inputs"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(inputdir, exist_ok=True)

    skip_run = bool(cfg.get("skip_run", False)) or bool(os.environ.get("CLEO_RF3_SKIP_RUN"))
    if skip_run:
        if not os.path.isdir(outdir):
            raise FileNotFoundError(
                f"rf3 skip_run: no outputs directory at {outdir!r}. "
                "Run without skip_run once, or fix rundir / step name."
            )
        print(f"rf3: skip_run=True (reuse only). Reading metrics from {outdir!r}.")
    else:
        with open(cfg.template_path) as f:
            af3_template = json.load(f)

        # One JSON holding every design. RF3 amortises model load across a batch
        # and distributes over available GPUs itself, so sharding the way
        # af3_from_df does would have our scheduler and RF3's fight over the
        # same devices. Startup is ~100 s against ~9 s per design, so batching
        # is most of the cost control.
        examples = [
            _rf3_input_from_af3_template(af3_template, name, seq)
            for name, seq in zip(df_input["name"], df_input["sequence"])
        ]
        input_json = os.path.join(inputdir, "inputs.json")
        with open(input_json, "w") as f:
            json.dump(examples, f, indent=2)

        overrides = [
            f"inputs='{input_json}'",
            f"out_dir='{outdir}'",
            f"diffusion_batch_size={n_samples}",
            f"early_stopping_plddt_threshold={float(cfg.get('early_stopping_plddt_threshold', 0.0))}",
        ]
        if cfg.get("ckpt_path"):
            overrides.append(f"ckpt_path='{cfg.ckpt_path}'")
        if cfg.get("num_steps"):
            overrides.append(f"num_steps={int(cfg.num_steps)}")

        container = cfg.get("rf3_container", RF3_CONTAINER)
        command = f"apptainer exec --nv {container} rf3 fold " + " ".join(overrides)
        print(f"rf3: folding {len(examples)} designs x {n_samples} samples")
        print(f"  {command}")

        # Retry transient infrastructure faults. A training run is thousands of
        # these calls over many hours, and a single failure kills it: this
        # project has lost runs to /dev/shm exhaustion on one node and to an NFS
        # `[Errno 5] Input/output error` mid-write, neither of which says
        # anything about the designs. Retries are cheap against ~80 GPU-h.
        #
        # Deliberately not unlimited: a genuine configuration error (bad
        # template, missing checkpoint) fails identically every time, so after
        # `rf3_retries` attempts the error is raised rather than looped on.
        attempts = int(cfg.get("rf3_retries", 3))
        for attempt in range(1, attempts + 1):
            try:
                subprocess.run(command, shell=True, check=True)
                break
            except subprocess.CalledProcessError:
                if attempt == attempts:
                    raise
                wait = 30 * attempt
                print(f"rf3: attempt {attempt}/{attempts} failed; retrying in {wait}s")
                time.sleep(wait)

    output_folders = [f for f in glob.glob(os.path.join(outdir, "*")) if os.path.isdir(f)]
    assert len(output_folders) > 0, f"No output folders found in {outdir}"

    # RF3 writes AF3's directory layout byte for byte -- seed-<S>_sample-<K>/ with
    # {name}_seed-<S>_sample-<K>_model.cif and _summary_confidences.json beside
    # it -- so the AF3 resolver is reused rather than reimplemented. If that ever
    # diverges, the assertion above and the merge check below are what catch it.
    records = []
    for folder in output_folders:
        samples = _af3_sample_paths_from_output_folder(folder) if per_sample_rows else []
        if samples:
            records.extend((n, cif, conf, s, sd) for n, s, sd, cif, conf in samples)
            continue
        if per_sample_rows:
            print(
                f"Warning: no seed-*_sample-* subfolders in {folder!r}; "
                "falling back to the top-ranked prediction for this design."
            )
        paths = _af3_name_and_paths_from_output_folder(folder)
        if paths is None:
            print(f"Warning: no *_summary_confidences.json in {folder!r}, skipping.")
            continue
        records.append((*paths, 0, -1))

    cols = ["name", f"{step_name}_path", f"{step_name}_ptm", f"{step_name}_iptm",
            f"{step_name}_ranking_score", f"{step_name}_plddt",
            f"{step_name}_chain_ptm_protein", f"{step_name}_chain_ptm_ligand",
            f"{step_name}_has_clash", f"{step_name}_interaction_pae_min"]
    if per_sample_rows:
        cols += [f"{step_name}_sample", f"{step_name}_seed"]
    output_data = {c: [] for c in cols}

    for name, cif_path, conf_json_path, sample_idx, seed in records:
        if not os.path.exists(conf_json_path):
            print(f"Warning: confidence path missing for {name}, skipping.")
            continue
        with open(conf_json_path) as f:
            conf = json.load(f)

        # RF3 reports `early_stopped` when it abandoned a prediction. There is
        # no structure to score in that case, and a row with a path that does
        # not exist would surface downstream as an unexplained NaN.
        if conf.get("early_stopped") or not os.path.exists(cif_path):
            print(
                f"Warning: {name} sample {sample_idx} has no structure "
                f"(early_stopped={conf.get('early_stopped')}); skipping. "
                "Set early_stopping_plddt_threshold=0.0 to keep these."
            )
            continue

        output_data["name"].append(name)
        output_data[f"{step_name}_path"].append(cif_path)
        output_data[f"{step_name}_ptm"].append(conf["ptm"])
        output_data[f"{step_name}_iptm"].append(conf["iptm"])
        output_data[f"{step_name}_ranking_score"].append(conf["ranking_score"])
        output_data[f"{step_name}_plddt"].append(conf.get("overall_plddt"))
        # Chain 0 is the protein, the rest are ligand entities; the ligand
        # column summarises the worst-confidence ligand, as af3_from_df does.
        output_data[f"{step_name}_chain_ptm_protein"].append(conf["chain_ptm"][0])
        output_data[f"{step_name}_chain_ptm_ligand"].append(
            _min_not_none(conf["chain_ptm"][1:]))
        output_data[f"{step_name}_has_clash"].append(conf["has_clash"])

        pae_mat = conf["chain_pair_pae_min"]
        n_chains = len(pae_mat)
        off_diag = [pae_mat[i][j] for i in range(n_chains) for j in range(n_chains) if i != j]
        output_data[f"{step_name}_interaction_pae_min"].append(_min_not_none(off_diag))

        if per_sample_rows:
            output_data[f"{step_name}_sample"].append(sample_idx)
            output_data[f"{step_name}_seed"].append(seed)

    df_output = pd.DataFrame(output_data)
    df_output = _af3_canonicalize_output_names_to_input(df_input, df_output)
    if per_sample_rows and not df_output.empty:
        df_output = df_output.sort_values(["name", f"{step_name}_sample"])
    df_merged = pd.merge(df_input, df_output, on="name", how="inner")
    if len(df_merged) == 0 and len(df_input) > 0:
        raise RuntimeError(
            "rf3: no rows after merging predictions into the input table "
            f"(n_input={len(df_input)}, n_parsed={len(df_output)}, "
            f"output_folders={len(output_folders)}). Usually a name mismatch between the "
            f"input JSON and {outdir!r}, or every prediction early-stopped."
        )
    return df_merged


# For running Protenix via Apptainer container
PROTENIX_CONTAINER = "/net/software/containers/pipelines/protenix.sif"
PROTENIX_PYTHON = "/opt/protenix-env/bin/protenix"


def _protenix_input_from_af3_template(af3_template: dict, name: str, seq: str) -> dict:
    """Translate one AF3 input JSON into Protenix's schema.

    Third translation of the same template, for the same reason as the RF3 one:
    a cross-oracle comparison only means something if every predictor folds the
    same complex, and hand-maintained per-oracle templates drift. The AF3
    template stays authoritative; each oracle translates from it.

    AF3 nests entities under ``sequences`` as ``{"protein": {...}}`` /
    ``{"ligand": {"ccdCodes": [...]}}``. Protenix takes the same ``sequences``
    key but spells a protein ``{"proteinChain": {"sequence", "count"}}`` and a
    ligand ``{"ligand": {"ligand": "CCD_<code>", "count"}}`` -- the ``CCD_``
    prefix is required, and Protenix's own parser rejects a bare code
    ("Please change the 'ligand' input format to 'CCD_' or 'FILE_'").

    An AF3 ligand entity may name several CCD codes under one chain id; each
    becomes its own Protenix entity, since the format carries one code per
    entity. Chain ids are dropped -- Protenix assigns its own -- which is why
    metrics must key on the structure, not on chain naming.

    MSA is not requested here; the caller passes ``--use_msa False``. The AME
    templates carry empty MSA fields, so this is single-sequence on every oracle.
    """
    sequences = []
    for entity in af3_template["sequences"]:
        if "protein" in entity:
            sequences.append({"proteinChain": {"sequence": seq, "count": 1}})
        elif "ligand" in entity:
            ccd_codes = entity["ligand"].get("ccdCodes")
            if not ccd_codes:
                raise ValueError(
                    f"protenix: ligand entity {entity['ligand'].get('id')!r} in the AF3 template "
                    "has no ccdCodes. SMILES and file-based ligands are not translated yet; add "
                    "them here rather than letting the ligand silently vanish from the complex."
                )
            for ccd in ccd_codes:
                sequences.append({"ligand": {"ligand": f"CCD_{ccd}", "count": 1}})
        else:
            raise ValueError(
                f"protenix: unrecognised AF3 entity {sorted(entity)!r}; expected "
                "'protein' or 'ligand'."
            )
    return {"name": name, "sequences": sequences}


def protenix_from_df(df_input, cfg, step_name="protenix"):
    """Runs Protenix and aggregates confidence metrics from the output.

    Same contract as :func:`af3_from_df` / :func:`rf3_from_df` -- takes
    ``name``/``sequence``, returns the input frame with ``{step_name}_path`` and
    confidence columns merged on ``name`` -- so ``rfd2_metrics_from_df``
    consumes it by setting ``structure_col: protenix_path`` and nothing else
    changes. Reads the same AF3 JSON template, translated by
    :func:`_protenix_input_from_af3_template`.

    cfg keys
    --------
    rundir                Where inputs/ and outputs/ are written (required).
    template_path         AF3-format JSON template; translated to Protenix's schema.
    num_diffusion_samples Samples per design (default 5, the AME unit).
    use_msa               Default False: the AME templates carry empty MSA
                          fields, so every oracle here is single-sequence.
    skip_run              Reuse existing outputs and recompute metrics only.

    Output layout, observed rather than assumed:
        <out>/<name>/seed_<S>/predictions/<name>_sample_<K>.cif
        <out>/<name>/seed_<S>/predictions/<name>_summary_confidence_sample_<K>.json
    which is Protenix's own shape, not AF3's, so the AF3 resolver is NOT reused.

    Protenix reports ``plddt`` on 0-100; AF3 and RF3 use 0-1 (measured: RF3
    spans 0.504-0.772 on the same designs). ``{step}_plddt`` is therefore
    divided by 100 so one reward bound means the same thing whichever oracle
    scored the step -- without this, a plddt term would weight Protenix steps
    100x. The unscaled value is kept beside it as ``{step}_plddt_raw`` so the
    rescaling is visible in the metrics table rather than implicit.
    """
    assert cfg.rundir is not None, "rundir must be specified in cfg"
    n_samples = int(cfg.get("num_diffusion_samples", 5))
    per_sample_rows = bool(cfg.get("per_sample_rows", n_samples > 1))

    outdir = f"{cfg.rundir}/{step_name}/outputs"
    inputdir = f"{cfg.rundir}/{step_name}/inputs"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(inputdir, exist_ok=True)

    skip_run = bool(cfg.get("skip_run", False)) or bool(os.environ.get("CLEO_PROTENIX_SKIP_RUN"))
    if skip_run:
        if not os.path.isdir(outdir):
            raise FileNotFoundError(
                f"protenix skip_run: no outputs directory at {outdir!r}. "
                "Run without skip_run once, or fix rundir / step name."
            )
        print(f"protenix: skip_run=True (reuse only). Reading metrics from {outdir!r}.")
    else:
        with open(cfg.template_path) as f:
            af3_template = json.load(f)

        # One JSON holding every design, as for RF3: Protenix amortises a long
        # model load (plus a one-time CCD component download) across the batch.
        examples = [
            _protenix_input_from_af3_template(af3_template, name, seq)
            for name, seq in zip(df_input["name"], df_input["sequence"])
        ]
        input_json = os.path.join(inputdir, "inputs.json")
        with open(input_json, "w") as f:
            json.dump(examples, f, indent=2)

        container = cfg.get("protenix_container", PROTENIX_CONTAINER)
        runner = cfg.get("protenix_bin", PROTENIX_PYTHON)
        command = (
            f"apptainer exec --nv {container} {runner} pred"
            f" -i {input_json} -o {outdir} -e {n_samples}"
            f" --use_msa {bool(cfg.get('use_msa', False))}"
        )
        if cfg.get("model_name"):
            command += f" -n {cfg.model_name}"
        if cfg.get("seeds"):
            command += f" -s {cfg.seeds}"
        print(f"protenix: folding {len(examples)} designs x {n_samples} samples")
        print(f"  {command}")

        # Same bounded retry as rf3_from_df, for the same reason: a training run
        # is thousands of these calls and this project has lost runs to /dev/shm
        # exhaustion and to a mid-write NFS error, neither of which says
        # anything about the designs. A real config error fails identically
        # every attempt and is raised rather than looped on.
        attempts = int(cfg.get("protenix_retries", 3))
        for attempt in range(1, attempts + 1):
            try:
                subprocess.run(command, shell=True, check=True)
                break
            except subprocess.CalledProcessError:
                if attempt == attempts:
                    raise
                wait = 30 * attempt
                print(f"protenix: attempt {attempt}/{attempts} failed; retrying in {wait}s")
                time.sleep(wait)

    rows = []
    n_cif = n_unparsed = n_noconf = 0
    for cif in glob.glob(os.path.join(outdir, "*", "seed_*", "predictions", "*_sample_*.cif")):
        n_cif += 1
        pred_dir = os.path.dirname(cif)
        stem = os.path.basename(cif)[: -len(".cif")]
        m = re.search(r"^(?P<name>.+)_sample_(?P<k>\d+)$", stem)
        if m is None:
            n_unparsed += 1
            print(f"Warning: cannot parse sample index from {cif!r}, skipping.")
            continue
        name, k = m.group("name"), int(m.group("k"))
        conf = os.path.join(pred_dir, f"{name}_summary_confidence_sample_{k}.json")
        if not os.path.exists(conf):
            n_noconf += 1
            print(f"Warning: no summary confidence beside {cif!r}, skipping.")
            continue
        seed_m = re.search(r"seed_(\d+)", pred_dir)
        rows.append((name, cif, conf, k, int(seed_m.group(1)) if seed_m else -1))

    if not rows:
        # Not an assert: `python -O` strips asserts, and this is the one failure
        # a caller cannot diagnose from the message alone. The loop above warns
        # and continues, so "nothing found" has three different causes -- no
        # .cif at all (Protenix wrote nothing, or `outdir` is wrong), names it
        # could not parse, or .cif files whose confidence JSON is missing -- and
        # they call for opposite fixes. Say which one happened.
        raise RuntimeError(
            f"No usable Protenix predictions under {outdir}: found {n_cif} .cif "
            f"file(s), of which {n_unparsed} had unparsable sample indices and "
            f"{n_noconf} had no summary-confidence JSON beside them."
            + ("" if n_cif else " No .cif files matched at all -- check that "
               "Protenix ran and that outdir points at its output root.")
        )
    if not per_sample_rows:
        # Keep one sample per design: sample_0, matching the single-sample
        # contract the other oracles honour when asked for one row.
        rows = [r for r in rows if r[3] == 0]

    cols = ["name", f"{step_name}_path", f"{step_name}_ptm", f"{step_name}_iptm",
            f"{step_name}_ranking_score", f"{step_name}_plddt",
            f"{step_name}_chain_ptm_protein", f"{step_name}_chain_ptm_ligand",
            f"{step_name}_has_clash", f"{step_name}_plddt_raw"]
    if per_sample_rows:
        cols += [f"{step_name}_sample", f"{step_name}_seed"]
    output_data = {c: [] for c in cols}

    for name, cif, conf, k, seed in rows:
        with open(conf) as f:
            c = json.load(f)
        chain_ptm = c.get("chain_ptm") or []
        output_data["name"].append(name)
        output_data[f"{step_name}_path"].append(cif)
        output_data[f"{step_name}_ptm"].append(c.get("ptm"))
        output_data[f"{step_name}_iptm"].append(c.get("iptm"))
        output_data[f"{step_name}_ranking_score"].append(c.get("ranking_score"))
        # 0-100 -> 0-1, matching AF3/RF3 so a shared reward bound is valid.
        plddt = c.get("plddt")
        output_data[f"{step_name}_plddt"].append(
            None if plddt is None else float(plddt) / 100.0)
        output_data[f"{step_name}_plddt_raw"].append(plddt)
        # chain_ptm is [protein, ligand...]; several ligands collapse to their
        # best, matching how the AF3/RF3 columns summarise a multi-ligand complex.
        output_data[f"{step_name}_chain_ptm_protein"].append(
            chain_ptm[0] if chain_ptm else None)
        output_data[f"{step_name}_chain_ptm_ligand"].append(
            max(chain_ptm[1:]) if len(chain_ptm) > 1 else None)
        output_data[f"{step_name}_has_clash"].append(c.get("has_clash"))
        if per_sample_rows:
            output_data[f"{step_name}_sample"].append(k)
            output_data[f"{step_name}_seed"].append(seed)

    df_output = pd.DataFrame(output_data)
    df_merged = pd.merge(df_input, df_output, on="name", how="inner")
    if df_merged.empty:
        raise RuntimeError(
            f"protenix: merged frame is empty -- {len(df_input)} inputs and "
            f"{len(df_output)} predictions shared no 'name'. Check that the "
            "template translation preserved design names."
        )
    return df_merged


# Oracles the multi-oracle step can draw from, by name.
_ORACLE_FNS = {
    "rf3": rf3_from_df,
    "boltz": boltz_from_df,
    "protenix": protenix_from_df,
    "af3": af3_from_df,
}


def multi_oracle_from_df(df_input, cfg, step_name="fold"):
    """Fold with ONE oracle drawn per call, so a policy cannot overfit any single
    structure predictor.

    Training against a fixed predictor buys performance that does not transfer.
    Measured on M0097, 256 designs, best-of-5: sampled sequences score 36.3%
    under RF3 and 40.2% under AF3 -- the two predictors agree (-3.9) on designs
    no policy optimised. The same comparison on RF3-trained policies gives +18.0
    (sequence-level reward) and +36.3 (fragment-level), i.e. ~22 and ~40 points
    of excess that exist only under the training oracle.

    Drawing a different oracle each rollout removes the fixed target. The whole
    group is folded by ONE oracle, never a mixture: GRPO's advantage is
    group-relative -- ``advantage()`` centres and scales within the current
    batch -- so every design in a group must be scored on the same yardstick for
    the ranking to mean anything. Mixing predictors inside a group would make
    part of each advantage a comparison between predictors rather than between
    designs. One oracle per group also costs 1x rather than Nx. (The
    across-steps part holds only while ``use_avg_reward`` is off, which it is in
    every config here; with a running-history baseline, rewards WOULD have to be
    comparable across steps and alternation would inject the oracle's scale
    shift straight into the advantage.)

    ``schedule`` picks how the per-group oracle is chosen:

    ``round_robin`` (default)
        ``oracles[train_step % len(oracles)]``. Exactly uniform by
        construction. Preferred because the seed-arithmetic draw below is not:
        measured over a real 115-step run it gave 43%/32%/25% against the
        intended thirds, leaning toward whichever oracle sits first -- and the
        lean was reproducible across seeds, so it is a property of the
        ``seed + k*step`` sequence, not noise.
    ``random``
        Independent draw per step, honouring ``oracle_weights``. Use when the
        schedule itself should be unpredictable; expect the realised split to
        wander from the nominal one at these step counts.

    Emits a stable ``{step_name}_path`` whichever oracle ran, so downstream
    metric steps set ``structure_col: fold_path`` once and never change, plus
    ``{step_name}_oracle`` recording the draw -- without which the training log
    cannot say which predictor scored which step.

    cfg keys
    --------
    oracles        Names to draw from (default ["rf3", "boltz", "protenix"]).
    schedule       "round_robin" (default) or "random".
    oracle_weights Draw probabilities for schedule="random", parallel to ``oracles``.
    oracle         Force one oracle (skips the draw); for evaluation and tests.
    seed           Optional base seed; combined with ``step`` for a reproducible
                   draw. Without it the draw follows global numpy state.
    train_step     Current training step index, set by UniversalReward. Note
                   ``cfg.step`` is the step's NAME, not a number.
    ...            Every other key is passed to the chosen oracle unchanged.
    """
    names = list(cfg.get("oracles", ["rf3", "boltz", "protenix"]))
    unknown = [n for n in names if n not in _ORACLE_FNS]
    if unknown:
        raise ValueError(
            f"multi_oracle: unknown oracle(s) {unknown}; known: {sorted(_ORACLE_FNS)}"
        )
    if not names:
        raise ValueError("multi_oracle: 'oracles' is empty; nothing to fold with.")

    forced = cfg.get("oracle")
    if forced:
        if forced not in _ORACLE_FNS:
            raise ValueError(f"multi_oracle: unknown forced oracle {forced!r}")
        choice = forced
    elif str(cfg.get("schedule", "round_robin")) == "round_robin":
        # Exactly uniform: every oracle is used once per cycle of len(oracles)
        # steps, so no predictor is over-represented however long the run.
        choice = names[int(cfg.get("train_step", 0)) % len(names)]
    else:
        weights = cfg.get("oracle_weights")
        if weights is not None:
            w = np.asarray([float(x) for x in weights], dtype=float)
            if len(w) != len(names):
                raise ValueError(
                    f"multi_oracle: {len(weights)} weights for {len(names)} oracles."
                )
            w = w / w.sum()
        else:
            w = None
        seed = cfg.get("seed")
        if seed is None:
            rng = np.random.default_rng()
        else:
            # Same seed must not pick the same oracle every step.
            rng = np.random.default_rng(
                int(seed) + 1000003 * int(cfg.get("train_step", 0)))
        choice = str(rng.choice(names, p=w))

    print(f"multi_oracle: train_step {cfg.get('train_step', '?')} folding with "
          f"{choice!r} (pool {names})")

    # Each oracle writes under its own name inside the step's rundir, so a step
    # that draws Boltz cannot collide with, or be mistaken for, a step that drew
    # RF3 -- and a `skip_run` reread finds the right tree.
    sub = _ORACLE_FNS[choice](df_input, cfg, step_name=choice)

    out = sub.rename(columns={f"{choice}_path": f"{step_name}_path"})
    out[f"{step_name}_oracle"] = choice
    if f"{choice}_sample" in out.columns:
        out = out.rename(columns={f"{choice}_sample": f"{step_name}_sample"})
    if f"{step_name}_path" not in out.columns:
        raise RuntimeError(
            f"multi_oracle: {choice} returned no {choice}_path column; cannot "
            f"expose a stable {step_name}_path."
        )
    return out
