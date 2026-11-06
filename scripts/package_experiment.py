#!/usr/bin/env python3
"""Package the clean-comparison experiment so someone else can run it.

The configs `make_clean_comparison.py` writes are not portable: they carry
absolute paths under one home directory, point at inputs that live in a
gitignored tree, and write their output where only the author can write.
This turns them into a self-contained directory that another person can
drop anywhere and submit.

What it does about each problem:

INPUTS ARE COPIED IN. Each backbone needs a PDB and an AF3 JSON template.
The `.trb` is NOT copied -- 4.3 MB per backbone and never read, because
every config carries `motif_atoms` inline and `rfd2_metrics_from_df` only
falls back to the trb when they are absent. The whole bundle is ~200 KB.

PATHS BECOME RELOCATABLE. Inputs resolve through `${oc.env:CLEO_BUNDLE}`
and outputs through `${oc.env:CLEO_RUNS}`, both set by the submit scripts
from their own location. Nothing is hard-coded to a home directory, so the
bundle works wherever it is unpacked.

OUTPUTS GO SOMEWHERE WRITABLE. `CLEO_RUNS` defaults to `runs/` inside the
bundle, and `submit_all.sh` refuses to submit if that directory is not
writable rather than letting 18 jobs start and fail an hour in.

    uv run python scripts/package_experiment.py \\
        --src experiments/ame/cleanab --out /tmp/cleanab_bundle
"""

import argparse
import re
import shutil
import sys
import tarfile
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]

SUBMIT = """#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:large:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 7-00:00:00
#SBATCH -J {name}
# NOTE: sbatch parses #SBATCH lines before any shell runs, so a
# $VAR here is NOT expanded -- it would create a directory called
# "$BUNDLE". This path is relative to the SUBMITTING directory;
# submit_all.sh overrides it with an absolute one via --output.
#SBATCH -o logs/%x-%j.log

# Resolve the bundle from this script's own location, so the bundle can be
# moved or renamed without editing anything.
BUNDLE="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )/.." && pwd )"
export CLEO_BUNDLE="$BUNDLE"
export CLEO_RUNS="${{CLEO_RUNS:-$BUNDLE/runs}}"
export HYDRA_FULL_ERROR=1

mkdir -p "$CLEO_RUNS" || {{ echo "cannot create $CLEO_RUNS"; exit 1; }}
[ -w "$CLEO_RUNS" ] || {{ echo "$CLEO_RUNS is not writable"; exit 1; }}

cd "$CLEO_REPO" || {{ echo "set CLEO_REPO to your cleo checkout"; exit 1; }}

# hydra.run.dir is RELATIVE TO THE CWD by default, which is the repo -- so
# without this every job writes outputs/<date>/<time>/ into the checkout,
# 18 of them concurrently, and it fails outright on a read-only or shared
# clone. Point it at the run directory with everything else.
uv run cleo-design-train \
    --config-path "$BUNDLE/configs" \
    --config-name {name} \
    hydra.run.dir="$CLEO_RUNS/{name}/hydra"
"""

README = """# Clean comparison: does fragment shuffling help?

18 runs. Everything needed is in this directory; nothing points back at the
machine that built it.

## The question

Every shuffle-vs-no-shuffle comparison we have is confounded. Arms that
differ in objective also differ in oracle or in training budget, and there
is one run per cell against this much spread between nominally similar runs:

    pooled + shuffling   52.7, 54.7, 77.2, 93.1   (4 runs)
    pooled, no shuffling 53.2, 54.4               (2 runs)
    plain RL, rf3 only   53.5, 56.0, 76.7         (3 runs)

(mean recombinant pass rate, AF3-scored, over backbones M0097/M0255/M0315.)

The best "no shuffling" arm beats two of the four "shuffling" arms. Nothing
in that separates the method from the draw. This experiment varies **one
factor plus seed** and nothing else:

| | |
|---|---|
| varies | objective (plain vs shuffle), seed (1, 2, 3) |
| matched | oracle (pool: rf3/boltz/protenix, round-robin), batch 16, N_updates 16, lr 1e-4, N_steps 300, fresh start |
| backbones | M0097, M0255, M0315 |
| held out | AF3 is in no training loop, and must stay out |

`parent_metric` is deliberately OFF, so the only difference is shuffling.

## Running it

You need a cleo checkout and `uv`:

    git clone <cleo repo> && cd cleo && uv sync --extra cuda

Then:

    export CLEO_REPO=/path/to/your/cleo      # required
    export CLEO_RUNS=/somewhere/writable     # optional; defaults to ./runs here
    ./submit_all.sh

`submit_all.sh` checks that `CLEO_REPO` looks like a cleo checkout and that
the output and log directories are writable, and refuses to queue anything
if not -- better than 18 jobs failing an hour in. To run one:

    sbatch submit/M0097_shuffle_s1.submit

## If sbatch is rejected

    sbatch: error: AssocMaxSubmitJobLimit

means your SLURM account cannot submit, which is nothing to do with this
bundle. Check with:

    sacctmgr -n show assoc user=$USER format=Account,Partition,MaxSubmitJobs,MaxJobs

A zero needs an admin to raise it. (This is why the bundle exists: the
account that built it is capped at zero.)

## Cost

Roughly **27 GPU-h** per plain run and **~135 GPU-h** per shuffle run -- a
shuffle step folds about five times as much (rotations x batch x samples =
4 x 16 x 5 = 320 chimera folds plus 80 parent folds, against 80 for plain).
About 1,450 GPU-h in total; the shuffle runs take ~5.5 days each, which is
why the walltime is set to the 7-day partition maximum.

The oracle container paths are the IPD defaults baked into
`cleo/design/utils/oracle.py`. On another cluster, set `rf3_container`,
`protenix_container` and the `af3_probe_cfg` paths in the configs.

## Where it writes

Everything the jobs produce goes under two directories, and `submit_all.sh`
proves both are writable (by creating a file, not by trusting `[ -w ]`)
before it queues anything:

    $CLEO_RUNS/<name>/        run output, checkpoints, per-step metrics
    $CLEO_RUNS/<name>/hydra/  hydra's own run dir
    <bundle>/logs/            sbatch stdout

Nothing is written into the cleo checkout. That needed pinning:
`hydra.run.dir` is relative to the working directory by default, which is
the repo, so without the override in each submit script all 18 jobs would
write `outputs/<date>/<time>/` into your clone concurrently — and fail
outright on a read-only or shared one.

`CLEO_RUNS` defaults to `runs/` inside this bundle, which is group-writable.
If you are not in the owning group, point it anywhere you own:

    export CLEO_RUNS=/net/scratch/$USER/cleanab

## What comes back

Per run, under `$CLEO_RUNS/<name>/`:

    <name>_train_metrics.csv     per-step log
    <name>_last.pt               final checkpoint
    <name>/outputs/step_NNNN/    per-step metrics.csv

**The headline number is not in there.** It needs a library evaluation:
sample 256 parents from the final checkpoint, recombine into 256 chimeras,
fold best-of-5 through AF3, and report the recombinant pass rate. Send back
the checkpoints and `*_train_metrics.csv` and that can be run here, or run
it there with `cleo-design-sample` -> `cleo-design-resample` ->
`cleo-design-evaluate`.

Two things to flag if you see them: a job that restarts mid-run writes step
history from 0 again into the same tree, so step numbering can repeat; and
any run that ends early should be reported rather than quietly included --
one of the arms we are trying to interpret is marked `.truncated` and we
still cannot tell whether that is why it scored 38 points low.

## Contents

    configs/     18 hydra configs, paths via $CLEO_BUNDLE / $CLEO_RUNS
    inputs/      3 backbone PDBs + 3 AF3 JSON templates (~200 KB)
    submit/      18 sbatch scripts
    submit_all.sh
    logs/        sbatch stdout lands here
"""


ALL = """#!/bin/bash
# Submit every run in this bundle.
#
#   export CLEO_REPO=/path/to/your/cleo         # required
#   export CLEO_RUNS=/somewhere/writable        # optional, defaults to ./runs
#   ./submit_all.sh
set -u
BUNDLE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
: "${CLEO_REPO:?set CLEO_REPO to your cleo checkout}"
export CLEO_RUNS="${CLEO_RUNS:-$BUNDLE/runs}"

# Fail before queueing rather than an hour into 18 jobs.
#
# Every path the jobs write to is under one of these two: run output and
# checkpoints under $CLEO_RUNS/<name>/, hydra's own run dir under
# $CLEO_RUNS/<name>/hydra (pinned explicitly, since it otherwise lands in
# the repo), and sbatch logs under $BUNDLE/logs.
#
# Tested by actually creating a file, not with `[ -w ]`: on NFS with ACLs or
# a full quota, -w reports writable and the write still fails.
mkdir -p "$CLEO_RUNS" "$BUNDLE/logs" || exit 1
for d in "$CLEO_RUNS" "$BUNDLE/logs"; do
  probe="$d/.write_probe.$$"
  if ! ( : > "$probe" ) 2>/dev/null; then
    echo "NOT WRITABLE: $d"
    echo "  set CLEO_RUNS to somewhere you own, e.g. /net/scratch/\$USER/cleanab"
    exit 1
  fi
  rm -f "$probe"
done
[ -d "$CLEO_REPO/src/cleo" ] || { echo "CLEO_REPO does not look like a cleo checkout: $CLEO_REPO"; exit 1; }

n=0
for s in "$BUNDLE"/submit/*.submit; do
  CLEO_REPO="$CLEO_REPO" CLEO_RUNS="$CLEO_RUNS" \
    sbatch --output="$BUNDLE/logs/%x-%j.log" "$s" && n=$((n+1))
done
echo "submitted $n job(s); outputs under $CLEO_RUNS"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory holding configs/")
    ap.add_argument("--out", required=True, help="bundle directory to create")
    ap.add_argument("--tar", action="store_true", help="also write a .tar.gz")
    a = ap.parse_args()

    src = Path(a.src).resolve()
    cfgs = sorted((src / "configs").glob("*.yaml"))
    if not cfgs:
        sys.exit(f"no configs under {src/'configs'}")

    out = Path(a.out).resolve()
    for d in ("configs", "inputs", "submit", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)

    copied, n_in = {}, 0
    for f in cfgs:
        txt = f.read_text()

        # Copy every input the config names, and rewrite the reference.
        for m in sorted(set(re.findall(r"[\w./-]+\.(?:pdb|json)", txt))):
            p = (ROOT / m) if not m.startswith("/") else Path(m)
            if not p.exists() or "/containers/" in str(p) or "alphafold" in str(p):
                continue
            dest = out / "inputs" / p.name
            if p.name not in copied:
                shutil.copy2(p, dest)
                copied[p.name] = p
                n_in += 1
            txt = txt.replace(m, "${oc.env:CLEO_BUNDLE}/inputs/" + p.name)

        # Drop the trb reference: motif_atoms are inline, so it is never read,
        # and a dangling 4.3 MB path in a bundle is a trap.
        txt = "\n".join(l for l in txt.splitlines()
                        if not re.match(r"\s*trb:\s", l))

        cfg = OmegaConf.create(txt)
        name = f.stem
        cfg.output_dir = "${oc.env:CLEO_RUNS}/" + name
        (out / "configs" / f.name).write_text(OmegaConf.to_yaml(cfg, resolve=False))
        (out / "submit" / f"{name}.submit").write_text(
            SUBMIT.format(name=name))

    (out / "README.md").write_text(README)

    sh = out / "submit_all.sh"
    sh.write_text(ALL)
    sh.chmod(0o755)
    for s in (out / "submit").glob("*.submit"):
        s.chmod(0o755)

    # The recipient may run the bundle where it sits rather than copying it,
    # and then has to be able to write logs/ and runs/. Group-writable with
    # setgid so new files keep the group; 0o775 on dirs, 0o664 on data.
    # Without this the guards in submit_all.sh fire on a bundle that is
    # otherwise fine, which looks like a packaging bug.
    for d in ("logs", "runs"):
        t = out / d
        t.mkdir(exist_ok=True)
        t.chmod(0o2775)
    out.chmod(0o2775)
    for sub in ("configs", "inputs", "submit"):
        (out / sub).chmod(0o2775)
        for f in (out / sub).iterdir():
            f.chmod(0o775 if f.suffix == ".submit" else 0o664)
    (out / "README.md").chmod(0o664)

    print(f"bundle: {out}")
    print(f"  {len(cfgs)} configs, {n_in} input files, {len(cfgs)} submit "
          f"scripts, README.md")
    for k in sorted(copied):
        print(f"    inputs/{k}")
    if a.tar:
        tp = out.with_suffix(".tar.gz")
        with tarfile.open(tp, "w:gz") as t:
            t.add(out, arcname=out.name)
        print(f"  tarball: {tp} ({tp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
