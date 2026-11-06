"""Best-of-N reduction for ANY metric family.

``rfd2_benchmark.best_of_n_from_df`` reduces the AME motif columns and only
those, so a target scored on anything else -- PETase catalytic geometry, say --
had nothing it could reduce and was forced to run ``num_diffusion_samples: 1``.
That is not a small loss: best-of-5 is the unit the benchmark is defined in,
and a single-sample run measures a different thing.

This module reduces by an explicit SELECTION CRITERION instead of by hard-coded
column names. Every oracle emits N rows per sequence; exactly one survives.

    - name: bo5
      target_fn: cleo.design.utils.best_of_n.best_of_n_from_df
      cfg:
        group_col: name
        # pick the sample that scores best under the run's own reward
        aggregation: ${reward.reward_aggregation}

Config
------
group_col    Column identifying a sequence across its samples ("name").
aggregation  A reward_aggregation list; the winning sample is the one with the
             highest weighted, normalised score. Interpolate the run's own
             (``${reward.reward_aggregation}``) so the choice of "best" and the
             objective cannot drift apart.
select_col   Alternative to `aggregation`: a single column to pick on.
select_mode  "min" or "max", with `select_col`. Default "max".
any_cols     Extra boolean columns OR-ed across the samples rather than taken
             from the winning row. Use for pass flags, where the question is
             "did ANY prediction succeed".
min_cols     Columns reduced by min across samples instead of taken from the
             winner (e.g. a best-achieved distance).
max_cols     Likewise, by max.

Only bounded normalisation is used for selection. `normalize: rank` and
`zscore` in a reward_aggregation are BATCH-relative -- they answer "which
design in this batch", not "which sample of this design" -- so applying them
per group would make a sample's score depend on the other sequences in the
rollout. Entries carrying them are scored on their bounds here, and a warning
says so once rather than silently changing what "best" means.
"""

import warnings

import numpy as np
import pandas as pd


def _bounded_score(values, spec):
    """-> per-row score in [0,1] for one reward_aggregation entry."""
    lo = float(spec.get("lower_bound", 0.0))
    hi = float(spec.get("upper_bound", 1.0))
    mode = str(spec.get("mode", "max"))
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    norm = (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-3)
    if mode == "max":
        s = norm
    elif mode == "min":
        s = 1.0 - norm
    elif mode == "avg":
        s = 1.0 - np.abs(norm - 0.5) * 2.0
    else:
        raise ValueError(f"best_of_n: unknown mode {mode!r} for {spec.get('metric')!r}")
    # A failed prediction must never win. It still has to survive as a row if
    # every sample of that sequence failed, which the caller handles.
    return np.where(np.isfinite(v), s, np.nan)


def _selection_score(df, cfg, step_name):
    """-> (Series of per-row scores, "higher is better")."""
    agg = cfg.get("aggregation")
    if agg:
        rank_like = [m.get("metric") for m in agg
                     if str(m.get("normalize", "") or "") in ("rank", "zscore")]
        if rank_like:
            warnings.warn(
                f"{step_name}: {rank_like} use batch-relative normalisation "
                "(rank/zscore), which cannot rank the samples of ONE sequence. "
                "They are scored on their bounds for selection only; the "
                "reward itself is unaffected.")
        missing = [m["metric"] for m in agg if m["metric"] not in df.columns]
        if missing:
            raise KeyError(
                f"{step_name}: aggregation names {missing}, which are not in the "
                f"dataframe. Put this step AFTER the metric steps that produce "
                f"them. Available: {sorted(df.columns)}")
        total = np.zeros(len(df), dtype=float)
        wsum = 0.0
        for m in agg:
            w = float(m.get("weight", 1.0))
            total = total + w * _bounded_score(df[m["metric"]], m)
            wsum += w
        return pd.Series(total / (wsum or 1.0), index=df.index)

    col = cfg.get("select_col")
    if not col:
        raise KeyError(
            f"{step_name}: give either 'aggregation' (preferred -- interpolate "
            "${reward.reward_aggregation}) or 'select_col' + 'select_mode'.")
    if col not in df.columns:
        raise KeyError(f"{step_name}: select_col {col!r} not in dataframe; "
                       f"available: {sorted(df.columns)}")
    v = pd.to_numeric(df[col], errors="coerce")
    mode = str(cfg.get("select_mode", "max"))
    if mode not in ("min", "max"):
        raise ValueError(f"{step_name}: select_mode must be 'min' or 'max', got {mode!r}")
    return -v if mode == "min" else v


def best_of_n_from_df(df_input, cfg, step_name="best_of_n"):
    """Collapse N rows per sequence to the one that scores best. -> DataFrame."""
    group_col = cfg.get("group_col", "name")
    if group_col not in df_input.columns:
        raise KeyError(f"{step_name}: '{group_col}' not in dataframe; "
                       f"available: {sorted(df_input.columns)}")

    if df_input[group_col].duplicated().sum() == 0:
        warnings.warn(
            f"{step_name}: every {group_col!r} is unique, so there is nothing to "
            "reduce. Set per_sample_rows: true and num_diffusion_samples > 1 on "
            "the oracle step, or drop this step.")
        return df_input

    d = df_input.copy()
    score = _selection_score(d, cfg, step_name)
    d["_bon_score"] = score.to_numpy(dtype=float)

    # Rows that scored NaN are failed predictions and must not win -- but a
    # sequence whose samples ALL failed still has to come through as one row,
    # or the reward tensor stops matching the batch.
    order = d[group_col].drop_duplicates().tolist()
    best = d.groupby(group_col)["_bon_score"].transform(
        lambda s: s.isna().all() or s.eq(s.max()))
    keep = d.loc[best.fillna(False)]
    out = keep.groupby(group_col, sort=False).head(1).set_index(group_col)

    # Columns where the question is "did ANY sample manage it", not "what did
    # the winner do".
    for c in cfg.get("any_cols", []) or []:
        if c in d.columns:
            out[c] = d.groupby(group_col)[c].any()
    for c in cfg.get("min_cols", []) or []:
        if c in d.columns:
            out[c] = d.groupby(group_col)[c].min()
    for c in cfg.get("max_cols", []) or []:
        if c in d.columns:
            out[c] = d.groupby(group_col)[c].max()

    out[f"{step_name}_n_samples"] = d.groupby(group_col).size()
    out = out.loc[order].reset_index()
    return out.drop(columns=["_bon_score"], errors="ignore")
