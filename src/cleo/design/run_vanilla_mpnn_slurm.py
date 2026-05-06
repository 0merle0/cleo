"""
SLURM array driver for vanilla ProteinMPNN baseline sampling + evaluation.

Each task: (1) ``sample_from_policy`` with bundled weights, (2) ``evaluate_sequences``
using the same ``reward.steps`` as the training YAML (AF3, dist_to_ref, etc.)—same
pipeline as ``run_checkpoint_sample_eval`` after sampling.

Usage::

  cd cleo
  uv run python -m cleo.design.run_vanilla_mpnn_slurm slurm_bundle_dir=slurm_vanilla_mpnn_bundle
  sbatch slurm_vanilla_mpnn_bundle/submit_array.sh

Worker::

  uv run python -m cleo.design.run_vanilla_mpnn_slurm \\
    manifest_path=/path/to/manifest.json task_id=0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from cleo.design.run_checkpoint_sample_eval import (
    load_train_cfg,
    write_eval_hydra_config,
)

_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/design")


def _cleo_root() -> Path:
    return Path(__file__).resolve().parents[3]


CLEO_ROOT = _cleo_root()


def build_vanilla_task_list(repo_root: Path, num_batches: int) -> list[dict]:
    """Four tasks: gdf8 / ltk × T=0.1 / 0.2; each run has its own directory (like checkpoint_eval)."""
    r = repo_root.resolve()
    gdf8_yaml = r / "configs/gdf8_vhh/gdf8_vhh_lep_run10_distw0.yaml"
    ltk_yaml = r / "configs/ltk_vhh/ltk_vhh_design.yaml"
    pdb_gdf8 = r / "data/gdf8_vhh/lepionce_run10_packed41_complex.pdb"
    pdb_ltk = r / "data/ltk_vhh/ltk_vhh97_complex_HA.pdb"
    out_base = CLEO_ROOT / "cleo_runs" / "vanilla_mpnn_baseline"
    tasks: list[dict] = []
    for temp in (0.1, 0.2):
        tstr = str(temp)
        for train_yaml, pdb, label_prefix in (
            (gdf8_yaml, pdb_gdf8, "gdf8"),
            (ltk_yaml, pdb_ltk, "ltk"),
        ):
            run_label = f"{label_prefix}_T{tstr}_vanilla"
            run_dir = out_base / run_label
            tasks.append(
                {
                    "train_yaml": str(train_yaml),
                    "pdb": str(pdb),
                    "temperature": temp,
                    "output_name": run_label,
                    "num_batches": num_batches,
                    "output_dir": str(run_dir.resolve()),
                    "run_name_override": None,
                }
            )
    return tasks


def prepare_slurm_bundle(
    bundle_dir: Path,
    tasks: list[dict],
    job_name: str,
    array_max_parallel: int | None,
    slurm_partition: str,
    slurm_gres: str,
    cpus_per_task: int,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / "manifest.json"
    submit_path = bundle_dir / "submit_array.sh"

    with open(manifest_path, "w") as f:
        json.dump(tasks, f, indent=2)

    n = len(tasks)
    if n == 0:
        print("No tasks; submit script not created.", file=sys.stderr)
        return

    last_idx = n - 1
    throttle = ""
    if array_max_parallel is not None and array_max_parallel > 0:
        throttle = f"%{array_max_parallel}"

    manifest_abs = manifest_path.resolve()
    part_line = f"#SBATCH --partition={slurm_partition}\n"
    gres_line = f"#SBATCH --gres={slurm_gres}\n"
    submit_body = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{last_idx}{throttle}
{part_line}{gres_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

TASK_ID="${{SLURM_ARRAY_TASK_ID:?need SLURM array environment}}"

cd {CLEO_ROOT}
uv run --frozen python -m cleo.design.run_vanilla_mpnn_slurm \\
  manifest_path={manifest_abs} \\
  task_id="$TASK_ID"
"""

    submit_path.write_text(submit_body)
    os.chmod(submit_path, 0o755)

    print(f"Wrote {manifest_path} ({n} tasks, indices 0..{last_idx})")
    print(f"Wrote {submit_path}")
    print(f"Submit with: sbatch {submit_path}")


def _cfg_truthy_manifest(cfg: DictConfig) -> str | None:
    mp = cfg.get("manifest_path")
    if mp is None or mp == "null":
        return None
    s = str(mp).strip()
    return s if s and s.lower() != "none" else None


def resolve_task_id(cfg: DictConfig) -> int | None:
    tid = cfg.get("task_id")
    if tid is not None and str(tid).strip() not in ("null", "", "None"):
        return int(tid)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is not None and env != "":
        return int(env)
    return None


def run_worker_task(manifest_path: Path, task_id: int) -> None:
    with open(manifest_path) as f:
        tasks = json.load(f)
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"task_id {task_id} out of range (0..{len(tasks) - 1})")
    t = tasks[task_id]

    run_dir = Path(t["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    out_name = str(t["output_name"])
    train_yaml = Path(t["train_yaml"])

    print(
        f"Worker task {task_id}/{len(tasks) - 1}: {out_name}  "
        f"T={t['temperature']}\n  train_yaml: {train_yaml}\n  pdb: {t['pdb']}"
    )

    sample_cmd = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "cleo.design.sample_from_policy",
        f"baseline_train_config={train_yaml}",
        f"pdb={t['pdb']}",
        f"temperature={t['temperature']}",
        f"num_batches={int(t['num_batches'])}",
        "fragment_bounds=null",
        "batch_size=null",
        f"output_dir={run_dir}",
        f"output_name={out_name}",
    ]
    print("  $ " + " ".join(sample_cmd))
    subprocess.run(sample_cmd, cwd=str(CLEO_ROOT), check=True)

    fasta = run_dir / f"{out_name}.fasta"
    if not fasta.is_file():
        raise FileNotFoundError(f"Expected FASTA after sampling: {fasta}")

    override = t.get("run_name_override")
    if override is not None and str(override).strip() in ("null", "", "None", "~"):
        override = None
    train_cfg = load_train_cfg(train_yaml, override)

    hydra_dir = run_dir / "_hydra_configs"
    eval_yaml = hydra_dir / "eval_vanilla.yaml"
    write_eval_hydra_config(train_cfg.reward.steps, fasta, run_dir, eval_yaml)

    eval_cmd = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "cleo.design.evaluate_sequences",
        "--config-path",
        str(hydra_dir),
        "--config-name",
        "eval_vanilla",
    ]
    print("  $ " + " ".join(eval_cmd))
    subprocess.run(eval_cmd, cwd=str(CLEO_ROOT), check=True)

    csv_path = run_dir / "evaluation.csv"
    if csv_path.is_file():
        print(f"  Wrote: {csv_path}")
    else:
        print(f"  Warning: expected {csv_path} missing after evaluate_sequences.")


def _slurm_str(cfg: DictConfig, key: str, default: str) -> str:
    v = cfg.get(key)
    if v is None or str(v).strip() in ("null", "", "None", "~"):
        return default
    return str(v).strip()


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="vanilla_mpnn_slurm")
def main(cfg: DictConfig) -> None:
    if not CLEO_ROOT.is_dir():
        print(f"cleo root not found: {CLEO_ROOT}", file=sys.stderr)
        sys.exit(1)

    mp = _cfg_truthy_manifest(cfg)
    if mp is not None:
        tid = resolve_task_id(cfg)
        if tid is None:
            print(
                "Worker mode needs task_id in config or SLURM_ARRAY_TASK_ID.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_worker_task(Path(mp), tid)
        print("\nDone.")
        return

    bundle_dir = cfg.get("slurm_bundle_dir")
    if bundle_dir is None or str(bundle_dir).strip() in ("null", "", "None"):
        print(
            "Set slurm_bundle_dir=... to write the bundle, or manifest_path + task_id for worker.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(str(cfg.repo_root)).resolve()
    num_batches = int(cfg.num_batches)
    tasks = build_vanilla_task_list(repo_root, num_batches)

    throttle = cfg.get("slurm_array_throttle")
    if throttle is not None and str(throttle) not in ("null", "", "None"):
        throttle = int(throttle)
    else:
        throttle = None

    job_name = _slurm_str(cfg, "slurm_job_name", "cleo_vanilla_mpnn")
    partition = _slurm_str(cfg, "slurm_partition", "gpu-train")
    gres = _slurm_str(cfg, "slurm_gres", "gpu:l40:2")
    _cpu = cfg.get("slurm_cpus_per_task")
    if _cpu is not None and str(_cpu).strip() not in ("null", "", "None", "~"):
        cpus = int(_cpu)
    else:
        cpus = 2

    prepare_slurm_bundle(
        Path(str(bundle_dir)).resolve(),
        tasks,
        job_name,
        throttle,
        partition,
        gres,
        cpus,
    )


if __name__ == "__main__":
    main()
