#!/usr/bin/env python3
"""Generate the one-factor shuffling experiment: pooled oracle, +/- shuffling.

WHY THIS EXISTS. Every shuffle-vs-no-shuffle comparison on record is
confounded. The arms that differ in objective also differ in oracle
(fragb32 pool vs r1panel rf3-only) or in training budget (shuffled arms ran
200-300 steps against the plain arms' 150), and there is one run per cell
against a within-condition spread of 52.7-93.1 for pooled shuffling and
53.5-76.7 for plain RL. Nothing in that can separate the method from the
draw.

This builds the missing experiment:

    2 objectives  x  3 backbones  x  3 seeds  =  18 runs

matched on oracle (pool, round_robin), batch (16), updates (16), lr, and
N_steps -- so the ONLY differences are the objective and the seed.

CONFIGS ARE DERIVED, NOT WRITTEN. Each one starts from a config that has
actually run on that backbone (multirr for plain, fragb32 for shuffle) and
changes only run_name, output_dir, seed and N_steps. Writing fresh configs
would mean re-deriving motif_atoms, fixed_residues, template and pdb paths
per backbone, which is exactly where a silent error would hide.

    uv run python scripts/make_clean_comparison.py --out experiments/ame/cleanab
    uv run python scripts/make_clean_comparison.py --out ... --submit

SUBMISSION IS CURRENTLY BLOCKED, and not by anything in this script. On
cluster `digs` the association jgershon/ipd has MaxSubmitJobs=0 and
MaxJobs=0 on all six partitions (cpu, cpu-bf, gpu, gpu-bf, gpu-em,
gpu-train), so every sbatch returns:

    sbatch: error: AssocMaxSubmitJobLimit
    Job violates accounting/QOS policy (job submit limit, ...)

That is an administrative cap, not a fault here -- 2,426 of 5,470
association rows on this cluster are capped the same way. sinfo and squeue
work fine, which is why the block only shows up at submit time. An admin has
to raise the limit for this account; afterwards this script is unchanged and
`--submit` queues all 18.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
AME = ROOT / "experiments" / "ame"

BACKBONES = {
    "M0097": "run_M0097_1ctt_cond9_14",
    "M0255": "run_M0255_1mg5_cond17_66",
    "M0315": "run_M0315_1ey3_cond18_25",
}
# source config per arm: one that has really run on these backbones
SOURCE = {"plain": ("multirr", "rr"), "shuffle": ("fragb32", "fragb32")}
SEEDS = (1, 2, 3)
STEPS = 300          # matched; the shuffled arms' budget, not the plain arms'

SUBMIT = """#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:large:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 7-00:00:00
#SBATCH -J {name}
#SBATCH -o {out}/{name}.log
export HYDRA_FULL_ERROR=1
cd {root}
uv run cleo-design-train --config-path {cfgdir} --config-name {name}
"""


def source_config(arm, bb_run):
    tree, _ = SOURCE[arm]
    hits = sorted((AME / tree).glob(f"{bb_run}*/*/*_config.yaml"))
    if not hits:
        sys.exit(f"no source config for {arm} on {bb_run} under {AME/tree}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for configs and runs")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--parent-metric", action="store_true",
                    help="set parent_metric on the shuffle arm. OFF by default: "
                         "the source arm (fragb32) does not set it, and it is "
                         "unproven -- fragrl2 added it and changed nothing. "
                         "Turning it on adds a second untested variable.")
    ap.add_argument("--submit", action="store_true",
                    help="actually sbatch them (default: just write files)")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    cfgdir = out / "configs"
    cfgdir.mkdir(parents=True, exist_ok=True)

    made = []
    for arm in ("plain", "shuffle"):
        for bb, bb_run in BACKBONES.items():
            src = source_config(arm, bb_run)
            for seed in a.seeds:
                name = f"{bb}_{arm}_s{seed}"
                cfg = OmegaConf.load(src)
                cfg.run_name = name
                cfg.output_dir = str(out / name)
                cfg.seed = int(seed)
                cfg.N_steps = int(a.steps)
                cfg.checkpoint_path = None        # fresh start, every arm
                if arm == "shuffle" and a.parent_metric:
                    cfg.reward.parent_metric = "ame_motif_rmsd"
                    cfg.reward.w_geom = 1.0
                    cfg.reward.w_frag = 1.0
                # output_dir is interpolated into the reward block in these
                # configs; re-point anything that still names the source tree.
                txt = OmegaConf.to_yaml(cfg, resolve=False)
                txt = txt.replace(str(src.parents[2]), str(out / name))
                (cfgdir / f"{name}.yaml").write_text(txt)
                made.append((name, arm, bb, seed))

    for name, *_ in made:
        (out / f"{name}.submit").write_text(
            SUBMIT.format(name=name, out=out, root=ROOT, cfgdir=cfgdir))

    print(f"wrote {len(made)} configs to {cfgdir}")
    print(f"  arms: plain / shuffle   backbones: {list(BACKBONES)}   "
          f"seeds: {a.seeds}   steps: {a.steps}")
    if not a.submit:
        print("\nDRY RUN -- nothing submitted. Re-run with --submit to queue them.")
        return
    for name, *_ in made:
        r = subprocess.run(["sbatch", str(out / f"{name}.submit")],
                           capture_output=True, text=True)
        print(f"  {name}: {r.stdout.strip() or r.stderr.strip()}")


if __name__ == "__main__":
    main()
