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

    # Fold and score the chimeras with the same machinery the standard arm uses.
    for fn_key, cfg_key, nm in (("oracle_fn", "oracle_cfg", "rf3"),
                                ("metric_fn", "metric_cfg", "ame"),
                                ("bo_fn", "bo_cfg", "bo")):
        fn = get_method(cfg[fn_key])
        sub = cfg[cfg_key]
        sub.rundir = cfg.rundir
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

    def __init__(self, *args, k=4, frag_step="frag", mode="min", **kwargs):
        super().__init__(*args, **kwargs)
        self.k = int(k)
        self.frag_step = frag_step
        self.mode = mode

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

        log = {
            f"{frag_step}_chimera_rmsd": float(df[f"{frag_step}_chimera_rmsd"].iloc[0]),
            f"{frag_step}_chimera_pass": float(df[f"{frag_step}_chimera_pass"].iloc[0]),
            "frag_score_spread": float(s.std().item()),
        }
        return A.to(device), log
