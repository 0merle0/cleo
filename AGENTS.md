# Orientation for coding agents

Read this first, then the one file below that matches your task. Everything
here is load-bearing: the rules exist because breaking them has produced wrong
results that looked right.

| your task | read |
|---|---|
| training a policy, changing a reward or oracle | [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md), then [docs/PRINCIPLES.md](docs/PRINCIPLES.md) |
| why the method is shaped this way | [docs/PRINCIPLES.md](docs/PRINCIPLES.md) |
| what the measured results actually say | [docs/COMPARISONS.md](docs/COMPARISONS.md) |
| running jobs, SLURM, reading run output | [docs/RUNNING.md](docs/RUNNING.md) |
| any figure or plot | [docs/FIGURES.md](docs/FIGURES.md) |
| oracle details, pooling, ligands | [config/design/POOLED_ORACLE.md](config/design/POOLED_ORACLE.md) |
| repo layout, CLI, adding a project | [CLAUDE.md](CLAUDE.md) |

## Hard invariants — do not break these silently

1. **AF3 is held out.** It is the evaluator and a read-only probe, never a
   training oracle. Training pools `[rf3, boltz, protenix]`. If you put `af3`
   in a pool, every number afterwards stops being a measurement.
2. **Fold N, reduce N.** Oracles default to 5 samples; a reward step must
   return one row per sequence. Every pipeline needs a `best_of_n_from_df`.
3. **Figures are SVG, with live text.** Save via `figio.save`, never
   `fig.savefig`. Never set `svg.fonttype` back to `"path"`.
4. **Seed anything that will be compared.** Unseeded run-to-run variance here
   is ±21 points of pass rate.
5. **Don't rewrite old configs to make them run.** They record what was
   executed. If one now fails a guard, that is correct.

## Working style this repo expects

- **Run it with `uv run`.** The bare interpreter is not the project env.
- **Verify rather than assume.** Repeatedly here the assumed value was wrong
  and one command would have caught it: slot boundaries were recovered by
  measurement, two of three oracle SMILES formats did not match our own
  docstrings, the font had to be confirmed installed. If a claim can be
  checked in one command, check it before writing it down.
- **Say what you did not verify.** An unverified claim stated plainly is worse
  than a gap, because it stops the next person checking.
- **Put new invariants in code.** A rule in a comment gets broken; a guard
  fails loudly. See the table in `docs/PRINCIPLES.md` §7.
- **A broken diagnostic is worse than none** — it looks like a working one.
  Make checks fail loudly or report that they did not run.
- **Keep comments about *why*.** The code says what it does; the comment
  should say what it cost to learn, with the number.

## Fast checks before you call something done

    uv run python -c "import cleo.design.utils.oracle, cleo.design.utils.best_of_n"

    # every training config still builds its reward, under both conversion modes
    uv run python -W ignore -c "
    from omegaconf import OmegaConf; import hydra.utils as hu
    OmegaConf.register_new_resolver('hydra', lambda p:'.', replace=True)
    for n in ['denovo_petase','pooled_oracle','fragment_shuffle']:
        c=OmegaConf.load(f'config/design/{n}.yaml'); OmegaConf.resolve(c)
        hu.instantiate(c.reward); hu.instantiate(c.reward,_convert_='all'); print(n,'OK')"

    cd paper && make figures          # all six print OK
    cd paper && pdflatex clean.tex    # 0 errors, 0 overfull

There is **no test suite** in this repo yet. Verification is by the commands
above plus whatever you can check directly — if you add behaviour that
deserves a test, say so rather than assuming one exists.

## Known open questions

Do not present these as settled:

- The PETase part names carry a `dist3`–`dist9` tag that matches the computed
  Hamming distance for only 992 of 1,803 parts, always with the computed value
  higher. The computed distances are sound; the tag means something else.
- `archive/ame_three_arm.py` contradicts the Figure 2 framing (fragrl beats
  the policy arm on 0/3 backbones, 53.8% vs 76.7%).
- `best_of_n` selection for PETase uses the reward; whether that is the right
  criterion for a *reported* number rather than a training one is unsettled.
- **Fragment shuffling loses to plain sequence RL on retention (0/3) and wins
  on fragment diversity (3/3).** It is a trade, not an improvement, and
  `parent_metric` did not change it. Do not describe the shuffled pipeline as
  the better method without saying on which axis. See docs/COMPARISONS.md.
- **No run exists with the current default** (pooled oracles AND shuffling);
  every shuffling arm on record is RF3-only.
