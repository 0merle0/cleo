"""Fragment-level reward: score a design by how well its parts travel.

Standard training rewards a sequence for folding. This rewards each *fragment*
for how well the chimeras containing it fold, which is what a combinatorial
library actually needs -- library size is ``P^k`` only if the recombinants fold.

The sampled sequences are never folded. Only chimeras are, and every fragment's
score is the mean over the chimeras it landed in.

Design notes that matter
------------------------

**Rotation, not random draws.** For rotation ``rho``, chimera ``i`` takes slot
``j`` from parent ``(i + rho*j) % B``. Every fragment then appears in exactly
``R`` chimeras. Random sampling leaves some fragments unscored and over-weights
others, which biases the advantage; simulated against known fragment effects,
rotation recovers the true ranking better than random permutation at this size.

**Averaging is good enough here, regression is not better.** A fragment's score
is contaminated by which partners it drew -- with 8 parents and 4 appearances it
sees only half the possible partners per slot. Fitting an additive model to
decontaminate was simulated and is *worse* at 32 chimeras (32 observations, 32
parameters); it only wins at 64+. Simple averaging recovers the correct pairwise
ordering ~80% of the time, and ~67% once fragment interactions are included,
which is the honest ceiling. That is enough to shift a distribution, which is
all GRPO needs -- but it is a diluted signal, so prefer more steps over a bigger
batch.

**Score on RMSD, not pass/fail.** Averaging a binary over 4 chimeras gives five
possible values; standardised within a slot of 8 that produces heavy ties, and
tied advantages give no gradient. Continuous motif RMSD has no ties and is the
same metric the standard arm rewards.

**Per-slot standardisation.** Slots differ systematically in tolerance -- one
covering the catalytic motif passes less often than a surface loop -- so pooling
them would reward position rather than fragment quality.

Returning ``[B, L]`` rather than ``[B,]`` needs no change to ``grpo.py``: its
surrogate is already per-token and only broadcasts the advantage when it is
lower-rank than the importance ratio. The global standardisation GRPO then
applies is an affine map over an already per-slot-standardised tensor, so it
preserves every relative ordering.
"""

import numpy as np
import pandas as pd
import torch

from cleo.design.utils.reward import UniversalReward, get_method


def equal_bounds(length, k):
    """k equal [start, end] inclusive slots covering 0..length-1."""
    edges = [round(i * length / k) for i in range(k + 1)]
    return [(edges[i], edges[i + 1] - 1) for i in range(k)]


def build_chimeras(sequences, k, rotations):
    """-> (chimera sequences, list of [(slot, parent), ...] per chimera).

    Rotation design: chimera (rho, i) takes slot j from parent (i + rho*j) % B.
    """
    B = len(sequences)
    bounds = equal_bounds(len(sequences[0]), k)
    seqs, provenance = [], []
    for rho in range(rotations):
        for i in range(B):
            parts, prov = [], []
            for j, (s, e) in enumerate(bounds):
                p = (i + rho * j) % B
                parts.append(sequences[p][s:e + 1])
                prov.append((j, p))
            seqs.append("".join(parts))
            provenance.append(prov)
    return seqs, provenance


def chimera_metrics_from_df(df_input, cfg, step_name="frag"):
    """Reward step: fold chimeras of the batch, score each fragment by them.

    Unlike every other step in the pipeline, the set this folds is not the set
    it was given -- it receives B sampled sequences and folds B*R chimeras built
    from their fragments, returning the original B rows with one column per slot
    holding that parent's fragment score.

    Config
    ------
    k            fragments per sequence (4)
    rotations    chimeras per fragment (4)
    oracle_fn    structure-prediction step to apply to the chimeras
    oracle_cfg   its config
    metric_fn    metric step to apply (rfd2_metrics_from_df)
    metric_cfg   its config; `structure_col` must match the oracle's output
    bo_fn        best-of-N reduction applied to the chimeras
    bo_cfg       its config
    metric_col   column to average per fragment (ame_motif_rmsd)
    """
    k = int(cfg.get("k", 4))
    R = int(cfg.get("rotations", 4))
    metric_col = cfg.get("metric_col", "ame_motif_rmsd")

    sequences = df_input["sequence"].tolist()
    chim_seqs, prov = build_chimeras(sequences, k, R)

    chim = pd.DataFrame({
        "name": [f"chim_{i:04d}" for i in range(len(chim_seqs))],
        "sequence": chim_seqs,
    })

    # Fold and score the chimeras with the same machinery the standard arm uses,
    # but in their OWN rundir. Sharing the parent chain's directory means the
    # oracle parses `seq_*` outputs alongside its own `chim_*` ones and fails to
    # merge, because the parents are not in its input.
    import os as _os
    chim_rundir = _os.path.join(str(cfg.rundir), "chimeras")
    _os.makedirs(chim_rundir, exist_ok=True)
    for fn_key, cfg_key, nm in (("oracle_fn", "oracle_cfg", "rf3"),
                                ("metric_fn", "metric_cfg", "ame"),
                                ("bo_fn", "bo_cfg", "bo")):
        fn = get_method(cfg[fn_key])
        sub = cfg[cfg_key]
        sub.rundir = chim_rundir
        # Propagate the training step into the nested configs. UniversalReward
        # sets train_step on the *step's* cfg, not on these sub-configs, so an
        # alternating oracle (multi_oracle_from_df) would see train_step=0 on
        # every rollout and draw the same predictor for the whole run -- which
        # looks exactly like a working multi-oracle run and is not one.
        if "train_step" in cfg:
            sub.train_step = cfg.train_step
        chim = fn(chim, sub, step_name=sub.get("step_name", nm))

    if metric_col not in chim.columns:
        raise KeyError(
            f"{step_name}: '{metric_col}' not produced by the chimera steps; "
            f"available: {sorted(chim.columns)}"
        )

    # Fragment score = mean metric over the chimeras containing it.
    vals = chim[metric_col].to_numpy(dtype=float)
    by_name = {n: v for n, v in zip(chim["name"], vals)}
    acc = np.zeros((len(sequences), k))
    cnt = np.zeros((len(sequences), k))
    for idx, pr in enumerate(prov):
        v = by_name.get(f"chim_{idx:04d}", np.nan)
        if not np.isfinite(v):
            continue
        for j, p in pr:
            acc[p, j] += v
            cnt[p, j] += 1
    # A fragment with no scored chimera takes the batch mean, so it receives
    # zero advantage rather than an arbitrary extreme.
    with np.errstate(invalid="ignore"):
        score = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    for j in range(k):
        col = score[:, j]
        if np.isnan(col).all():
            score[:, j] = 0.0
        else:
            score[np.isnan(col), j] = np.nanmean(col)

    out = df_input.copy()
    for j in range(k):
        out[f"{step_name}_slot{j}"] = score[:, j]
    out[f"{step_name}_chimera_pass"] = float(
        chim.get("ame_motif_pass_and_no_clash", pd.Series(dtype=float)).mean()
        if "ame_motif_pass_and_no_clash" in chim.columns else np.nan
    )
    out[f"{step_name}_chimera_rmsd"] = float(np.nanmean(vals))

    # --- diagnostic: the same chimeras under AF3, on a stride ---------------
    # `{step_name}_chimera_pass` above is scored by the TRAINING oracle, and
    # that number stopped tracking transfer: M0315 climbed 96.4 -> 99.6 on RF3
    # between generations while its AF3 chimera pass went 66.8 -> 65.6. Folding
    # the same chimeras under AF3 every `af3_probe_every` steps makes the
    # transfer gap visible during the run instead of only after a full
    # train-then-evaluate cycle.
    #
    # Read-only with respect to the reward: it writes one column and touches
    # neither `score` nor `vals`. AF3 must never enter the objective here -- it
    # is the held-out evaluator, and keeping it out is the entire point of the
    # cross-oracle protocol. Cost is one extra fold per stride, so at
    # every=100 it is ~1% of a run.
    out[f"{step_name}_chimera_pass_af3"] = np.nan
    every = int(cfg.get("af3_probe_every", 0) or 0)
    step_i = int(cfg.get("train_step", -1))
    if every > 0 and step_i >= 0 and step_i % every == 0:
        try:
            from omegaconf import OmegaConf as _OC

            from cleo.design.utils.oracle import af3_from_df

            def _copy(node):
                return _OC.create(_OC.to_container(node, resolve=True))

            pdir = _os.path.join(chim_rundir, "af3probe")
            _os.makedirs(pdir, exist_ok=True)
            ocfg = _copy(cfg["oracle_cfg"])
            ocfg.rundir = pdir
            # RF3-only knob; af3_from_df does not take it.
            ocfg.pop("early_stopping_plddt_threshold", None)
            # `oracle_cfg` is the TRAINING oracle's config. Under
            # multi_oracle_from_df it carries template_path (shared by design,
            # so one template feeds every predictor) but none of AF3's own
            # container paths, so the probe raised
            # `Missing key af3_container` on every stride and the column stayed
            # NaN -- a broken diagnostic that looked like a quiet one. The
            # paths come from the config rather than a module default, because
            # a site path baked into library code is wrong on the next cluster.
            probe_cfg = cfg.get("af3_probe_cfg", None)
            if probe_cfg is None:
                raise KeyError(
                    "af3_probe_every is set but af3_probe_cfg is missing; it "
                    "must supply af3_container, af3_script and model_dir"
                )
            for _k, _v in _OC.to_container(probe_cfg, resolve=True).items():
                ocfg[_k] = _v
            probe = pd.DataFrame({"name": chim["name"].tolist(),
                                  "sequence": chim_seqs})
            probe = af3_from_df(probe, ocfg, step_name="af3")
            mcfg = _copy(cfg["metric_cfg"])
            mcfg.rundir = pdir
            mcfg.structure_col = "af3_path"
            probe = get_method(cfg["metric_fn"])(probe, mcfg, step_name="ame")
            bcfg = _copy(cfg["bo_cfg"])
            bcfg.rundir = pdir
            probe = get_method(cfg["bo_fn"])(probe, bcfg, step_name="bo")
            if "ame_motif_pass_and_no_clash" in probe.columns:
                out[f"{step_name}_chimera_pass_af3"] = float(
                    probe["ame_motif_pass_and_no_clash"].mean())
                print(f"{step_name}: AF3 probe at step {step_i}: "
                      f"chimera pass {out[f'{step_name}_chimera_pass_af3'].iloc[0]:.3f} "
                      f"vs training oracle "
                      f"{out[f'{step_name}_chimera_pass'].iloc[0]:.3f}")
        except Exception as exc:
            # A diagnostic that can kill a 450-step run is worse than no
            # diagnostic. The column stays NaN and training continues.
            print(f"{step_name}: AF3 probe failed at step {step_i}: {exc!r}")

    return out


class FragmentReward(UniversalReward):
    """UniversalReward returning a per-token advantage from fragment scores.

    The aggregation in the base class produces one scalar per sequence. Here the
    reward varies *along* the sequence: every position inside slot j carries that
    fragment's standardised score. ``grpo.py`` consumes the ``[B, L]`` tensor
    directly, since its surrogate is already per-token.

    Config adds:
        k            fragments per sequence
        frag_step    name of the chimera step whose ``_slot{j}`` columns to read
        mode         "min" if lower metric is better (motif RMSD), else "max"
    """

    def __init__(self, *args, k=4, frag_step="frag", mode="min",
                 parent_metric=None, w_geom=1.0, w_frag=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = int(k)
        self.frag_step = frag_step
        self.mode = mode
        # Parent geometry, folded alongside the chimeras. Omitting it was a
        # mistake: with chimera quality as the only objective the policy
        # optimised fragments that travel while the sequences it emits
        # degraded -- 28 passing parents against 182 for the arm trained on
        # parent RMSD alone, and retention fell with them. The parent fold
        # costs 8 extra trunk passes against 32, about 25%.
        self.parent_metric = parent_metric
        self.w_geom = float(w_geom)
        self.w_frag = float(w_frag)

    def __call__(self, step, policy_output, feature_dict, device):
        import os
        import shutil

        rundir = os.path.join(self.output_dir, self.run_name, "outputs", f"step_{step:04}")
        if os.path.exists(rundir):
            shutil.rmtree(rundir, ignore_errors=True)
        os.makedirs(rundir, exist_ok=True)

        chain_mask = feature_dict["chain_labels"] == 0
        chain_mask = chain_mask[0]
        sequences = self.get_sequences(policy_output, chain_mask=chain_mask)
        df = self.get_input_df(sequences)

        for _s in self.steps:
            fn = get_method(_s.target_fn)
            _s.cfg.rundir = rundir
            _s.cfg.step = _s.name
            # This class duplicates UniversalReward's step loop, so anything
            # added there has to be added here too. train_step is what an
            # alternating oracle uses to advance its schedule; without it
            # multi_oracle_from_df drew rf3 on every rollout -- a single-oracle
            # run that logs and plots exactly like a multi-oracle one.
            _s.cfg.train_step = int(step)
            print(f"Running step: {_s.name} using function: {_s.target_fn}")
            df = fn(df, _s.cfg, step_name=_s.name)

        k, frag_step, mode = self.k, self.frag_step, self.mode

        cols = [f"{frag_step}_slot{j}" for j in range(k)]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"FragmentReward: missing {missing}; got {sorted(df.columns)}")

        s = torch.tensor(df[cols].to_numpy(dtype=float), dtype=torch.float32)
        if mode == "min":
            s = -s                                   # lower RMSD is better

        # Standardise within slot, so a slot's intrinsic difficulty does not
        # become reward. Then expand each slot across its positions.
        s = (s - s.mean(dim=0, keepdim=True)) / (s.std(dim=0, keepdim=True) + 1e-3)

        L = policy_output["S"].shape[1]
        bounds = equal_bounds(L, k)
        A = torch.zeros((s.shape[0], L), dtype=torch.float32)
        for j, (a, b) in enumerate(bounds):
            A[:, a:b + 1] = s[:, j].unsqueeze(1)
        A = self.w_frag * A

        # Parent geometry: one scalar per sequence, flat along its length. The
        # fragment term says "your parts travel"; this says "and you fold".
        if self.parent_metric:
            if self.parent_metric not in df.columns:
                raise KeyError(
                    f"FragmentReward: parent_metric '{self.parent_metric}' not in "
                    f"{sorted(df.columns)}; add the oracle and metric steps that "
                    "produce it before the chimera step.")
            g = torch.tensor(df[self.parent_metric].to_numpy(dtype=float),
                             dtype=torch.float32)
            if mode == "min":
                g = -g
            g = (g - g.mean()) / (g.std() + 1e-3)
            A = A + self.w_geom * g.unsqueeze(1)
            log_geom = {f"parent_{self.parent_metric}": float(df[self.parent_metric].mean())}
        else:
            log_geom = {}

        # `UniversalReward` emits `<metric>_batch_mean` keys from its
        # aggregation; this class overrides __call__ and never runs that, so a
        # config carrying a checkpoint_metric copied from a standard run dies at
        # the first checkpoint. Logged keys are named here explicitly so the
        # mismatch is visible in one place.
        # Write the per-step record. UniversalReward does this; overriding
        # __call__ dropped it, which lost the provenance the panel runs have and
        # also removed the signal the pruner uses to tell a finished step from a
        # live one.
        try:
            df.to_csv(os.path.join(rundir, "metrics.csv"), index=False)
        except Exception as exc:                      # never fail a step on logging
            print(f"FragmentReward: could not write metrics.csv ({exc})")

        log = {
            f"{frag_step}_chimera_rmsd": float(df[f"{frag_step}_chimera_rmsd"].iloc[0]),
            f"{frag_step}_chimera_pass": float(df[f"{frag_step}_chimera_pass"].iloc[0]),
            "frag_score_spread": float(s.std().item()),
            **log_geom,
        }
        # The AF3 probe is the only number here that measures transfer rather
        # than training-oracle score, and this dict is enumerated by hand -- so
        # omitting it meant the probe printed to the log and reached no file,
        # leaving `check_converged.py --metric frag_chimera_pass_af3` with
        # nothing to read. NaN on non-probe steps is correct: the readers drop
        # it, and a zero would be a measurement that never happened.
        _af3 = f"{frag_step}_chimera_pass_af3"
        if _af3 in df.columns:
            log[_af3] = float(df[_af3].iloc[0])
        return A.to(device), log
