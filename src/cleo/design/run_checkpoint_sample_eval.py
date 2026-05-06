"""
Hydra-driven checkpoint sampling + evaluation (same pipeline as training).

Config: ``cleo/config/design/checkpoint_sample_eval.yaml``

Usage::

  cd cleo
  uv run python -m cleo.design.run_checkpoint_sample_eval
  uv run python -m cleo.design.run_checkpoint_sample_eval dry_run=true
  uv run python -m cleo.design.run_checkpoint_sample_eval slurm_bundle_dir=/path/to/bundle

Worker (e.g. SLURM array task)::

  uv run python -m cleo.design.run_checkpoint_sample_eval \\
    manifest_path=/path/to/manifest.json task_id=0

Slurm bundle ``submit_array.sh`` defaults: ``partition=gpu-train``, ``gres=gpu:l40:2``,
``cpus-per-task=2`` (override via ``slurm_partition`` / ``slurm_gres`` / ``slurm_cpus_per_task``).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/design")

# Generated ``submit_array.sh`` uses these unless overridden in config.
DEFAULT_SLURM_PARTITION = "gpu-train"
DEFAULT_SLURM_GRES = "gpu:l40:2"
DEFAULT_SLURM_CPUS_PER_TASK = 2


def _cleo_root() -> Path:
    # .../cleo/src/cleo/design/this_file.py -> cleo repo root
    return Path(__file__).resolve().parents[3]


CLEO_ROOT = _cleo_root()


def load_train_cfg(path: Path, run_name_override: str | None) -> OmegaConf:
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    cfg = OmegaConf.load(path)
    if run_name_override is not None:
        cfg.run_name = run_name_override
    return cfg


def run_dir_for(cfg: OmegaConf) -> Path:
    run_name = str(cfg.run_name)
    base = Path(cfg.output_dir) / run_name
    candidates = [base, base / run_name]

    def has_ckpts(p: Path) -> bool:
        if not p.is_dir():
            return False
        if (p / f"{run_name}_best.pt").is_file():
            return True
        if (p / f"{run_name}_last.pt").is_file():
            return True
        return bool(list(p.glob(f"{run_name}_step_*.pt")))

    for c in candidates:
        if has_ckpts(c):
            return c
    return base


def _tag_sort_key(tag: str) -> tuple[int, str]:
    if tag == "best":
        return (1, tag)
    if tag == "last":
        return (2, tag)
    return (0, tag)


def discover_checkpoint_jobs(
    cfg: OmegaConf, min_step: int
) -> list[tuple[str, Path]]:
    run_name = str(cfg.run_name)
    rdir = run_dir_for(cfg)
    jobs: list[tuple[str, Path]] = []

    pattern = str(rdir / f"{run_name}_step_*.pt")
    for p in sorted(glob.glob(pattern)):
        base = os.path.basename(p)
        m = re.match(re.escape(run_name) + r"_step_(\d+)\.pt$", base)
        if not m:
            continue
        step = int(m.group(1))
        if step >= min_step:
            jobs.append((f"step_{step:04d}", Path(p)))

    best = rdir / f"{run_name}_best.pt"
    if best.is_file():
        jobs.append(("best", best))

    last = rdir / f"{run_name}_last.pt"
    if last.is_file():
        jobs.append(("last", last))

    jobs.sort(key=lambda t: (_tag_sort_key(t[0]), t[0]))
    return jobs


def resolve_jobs_for_run(
    run_cfg: DictConfig, train_omega: OmegaConf
) -> list[tuple[str, Path]]:
    """Use explicit ``checkpoints`` or ``auto_discover_min_step``."""
    checkpoints = run_cfg.get("checkpoints")
    if checkpoints is not None and len(checkpoints) > 0:
        jobs: list[tuple[str, Path]] = []
        for c in checkpoints:
            d = OmegaConf.to_container(c, resolve=True)
            assert isinstance(d, dict)
            jobs.append((str(d["tag"]), Path(str(d["path"]))))
        jobs.sort(key=lambda t: (_tag_sort_key(t[0]), t[0]))
        return jobs

    ads = run_cfg.get("auto_discover_min_step")
    if ads is not None:
        return discover_checkpoint_jobs(train_omega, int(ads))

    raise ValueError(
        f"Run {run_cfg.get('label')!r} needs non-empty `checkpoints` "
        "or an integer `auto_discover_min_step`."
    )


def write_eval_hydra_config(
    steps_cfg: OmegaConf,
    input_fasta: Path,
    output_dir: Path,
    dest_yaml: Path,
) -> None:
    steps_resolved = OmegaConf.create(
        OmegaConf.to_container(steps_cfg, resolve=True)
    )
    eval_cfg = OmegaConf.create(
        {
            "input_fasta": str(input_fasta.resolve()),
            "output_dir": str(output_dir.resolve()),
            "output_name": "evaluation",
            "steps": steps_resolved,
        }
    )
    dest_yaml.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(eval_cfg, dest_yaml)


def run_uv_module(module: str, overrides: list[str], dry_run: bool) -> None:
    cmd = ["uv", "run", "python", "-m", module, *overrides]
    print("  $ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(CLEO_ROOT), check=True)


def process_single_checkpoint(
    label: str,
    cfg: OmegaConf,
    tag: str,
    ckpt_path: Path,
    num_batches: int,
    dry_run: bool,
) -> None:
    rdir = run_dir_for(cfg)
    eval_base = rdir / "checkpoint_eval"
    hydra_dir = eval_base / "_hydra_configs"
    pdb = str(OmegaConf.select(cfg, "pdb"))

    sub = eval_base / tag
    sub.mkdir(parents=True, exist_ok=True)
    fasta = sub / "sampled.fasta"

    # A prior failed attempt may have left stale AF3 predictions under sub/run/.
    # Sampling re-randomizes sequence UUIDs, so old predictions become unmatchable
    # and will crash the AF3 merge step. Wipe the run dir before re-sampling.
    stale_run = sub / "run"
    if stale_run.exists() and not dry_run:
        print(f"  Removing stale eval outputs: {stale_run}")
        shutil.rmtree(stale_run)

    print(f"\n--- {label} / {tag} ---\n  checkpoint: {ckpt_path}")

    sample_overrides = [
        f"output_dir={sub}",
        "output_name=sampled",
        f"checkpoints=[{ckpt_path}]",
        f"num_batches={num_batches}",
        f"pdb={pdb}",
        "fragment_bounds=null",
    ]
    run_uv_module("cleo.design.sample_from_policy", sample_overrides, dry_run)

    if dry_run:
        return

    if not fasta.is_file():
        raise FileNotFoundError(f"Expected FASTA missing after sampling: {fasta}")

    eval_yaml = hydra_dir / f"eval_{tag}.yaml"
    write_eval_hydra_config(cfg.reward.steps, fasta, sub, eval_yaml)

    eval_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "cleo.design.evaluate_sequences",
        "--config-path",
        str(hydra_dir),
        "--config-name",
        f"eval_{tag}",
    ]
    print("  $ " + " ".join(eval_cmd))
    subprocess.run(eval_cmd, cwd=str(CLEO_ROOT), check=True)

    csv_path = sub / "evaluation.csv"
    if csv_path.is_file():
        print(f"  Wrote: {csv_path}")
    else:
        print(f"  Warning: expected CSV not found: {csv_path}")


def _run_name_override(run_cfg: DictConfig) -> str | None:
    ro = run_cfg.get("run_name_override")
    if ro is None or str(ro).strip() in ("null", "", "None", "~"):
        return None
    return str(ro)


def build_task_list(cfg: DictConfig) -> list[dict]:
    tasks: list[dict] = []
    num_batches = int(cfg.num_batches)
    for run_cfg in cfg.runs:
        train_yaml = Path(OmegaConf.to_container(run_cfg, resolve=True)["train_yaml"])
        run_override = _run_name_override(run_cfg)
        tcfg = load_train_cfg(train_yaml, run_override)
        rdir = run_dir_for(tcfg)
        jobs = resolve_jobs_for_run(run_cfg, tcfg)
        for tag, ckpt_path in jobs:
            tasks.append(
                {
                    "label": str(run_cfg.label),
                    "train_yaml": str(train_yaml.resolve()),
                    "run_name_override": run_override,
                    "tag": tag,
                    "checkpoint": str(ckpt_path.resolve()),
                    "num_batches": num_batches,
                    "run_dir": str(rdir.resolve()),
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
        print("No tasks in manifest; submit script not created.", file=sys.stderr)
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
# --frozen: avoid PyPI on compute nodes (use uv.lock + synced .venv on shared FS).
uv run --frozen python -m cleo.design.run_checkpoint_sample_eval \\
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


def run_worker_task(manifest_path: Path, task_id: int, dry_run: bool) -> None:
    with open(manifest_path) as f:
        tasks = json.load(f)
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"task_id {task_id} out of range (0..{len(tasks) - 1})")
    t = tasks[task_id]
    train_yaml = Path(t["train_yaml"])
    run_override = t.get("run_name_override")
    cfg = load_train_cfg(train_yaml, run_override)
    print(
        f"Worker task {task_id}/{len(tasks) - 1}: {t['label']} / {t['tag']}\n"
        f"  checkpoint: {t['checkpoint']}"
    )
    process_single_checkpoint(
        t["label"],
        cfg,
        t["tag"],
        Path(t["checkpoint"]),
        int(t["num_batches"]),
        dry_run,
    )


def process_run_sequential(
    run_cfg: DictConfig,
    num_batches: int,
    dry_run: bool,
) -> None:
    train_yaml = Path(OmegaConf.to_container(run_cfg, resolve=True)["train_yaml"])
    run_override = _run_name_override(run_cfg)
    cfg = load_train_cfg(train_yaml, run_override)
    run_name = str(cfg.run_name)
    rdir = run_dir_for(cfg)
    label = str(run_cfg.label)

    print(f"\n{'=' * 60}\nRun: {label}  (run_name={run_name})\n  dir: {rdir}\n{'=' * 60}")

    jobs = resolve_jobs_for_run(run_cfg, cfg)
    if not jobs:
        print("  No checkpoints resolved; skip.")
        return

    for tag, ckpt_path in jobs:
        process_single_checkpoint(
            label, cfg, tag, ckpt_path, num_batches, dry_run
        )


@hydra.main(
    version_base=None,
    config_path=_CONFIG_DIR,
    config_name="checkpoint_sample_eval",
)
def main(cfg: DictConfig) -> None:
    if not CLEO_ROOT.is_dir():
        print(f"cleo root not found: {CLEO_ROOT}", file=sys.stderr)
        sys.exit(1)

    dry_run = bool(cfg.get("dry_run", False))
    mp = _cfg_truthy_manifest(cfg)

    if mp is not None:
        tid = resolve_task_id(cfg)
        if tid is None:
            print(
                "Worker mode needs task_id in config or SLURM_ARRAY_TASK_ID.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_worker_task(Path(mp), tid, dry_run)
        print("\nDone.")
        return

    bundle_dir = cfg.get("slurm_bundle_dir")
    if bundle_dir is not None and str(bundle_dir) not in ("null", "", "None"):
        tasks = build_task_list(cfg)
        throttle = cfg.get("slurm_array_throttle")
        if throttle is not None and str(throttle) not in ("null", "", "None"):
            throttle = int(throttle)
        else:
            throttle = None
        def _slurm_str(key: str) -> str | None:
            v = cfg.get(key)
            if v is None or str(v).strip() in ("null", "", "None", "~"):
                return None
            return str(v).strip()

        slurm_partition = _slurm_str("slurm_partition") or DEFAULT_SLURM_PARTITION
        slurm_gres = _slurm_str("slurm_gres") or DEFAULT_SLURM_GRES
        _cpu = cfg.get("slurm_cpus_per_task")
        if _cpu is not None and str(_cpu).strip() not in ("null", "", "None", "~"):
            cpus_per_task = int(_cpu)
        else:
            cpus_per_task = DEFAULT_SLURM_CPUS_PER_TASK

        prepare_slurm_bundle(
            Path(str(bundle_dir)),
            tasks,
            str(cfg.slurm_job_name),
            throttle,
            slurm_partition,
            slurm_gres,
            cpus_per_task,
        )
        return

    num_batches = int(cfg.num_batches)
    for run_cfg in cfg.runs:
        process_run_sequential(run_cfg, num_batches, dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
