#!/usr/bin/env python3
"""
Aggregate evaluation.csv files from the checkpoint-eval manifest and optional vanilla
ProteinMPNN baseline runs, then plot metrics.

Usage (from cleo/)::
  uv run python scripts/compile_checkpoint_eval_bundle.py \\
    --manifest slurm_checkpoint_eval_bundle/manifest.json

Vanilla baselines: any ``evaluation.csv`` found under ``cleo_runs/vanilla_mpnn_baseline/``
(by default), e.g. after::

  uv run python -m cleo.design.evaluate_sequences ... output_dir=.../gdf8_T0.1_vanilla

Writes:
  - Combined CSV (default: next to manifest as ``checkpoint_eval_combined.csv``; all rows)
  - Deduplicated CSV (``checkpoint_eval_combined_unique_per_campaign.csv``): one row per
    (campaign, sequence) when using plot dedupe; disable with ``--no-output-csv-unique-per-campaign``
  - Metric stripplots in two files by source: **AF3 / structure**
    (``checkpoint_eval_metrics_af3.png``) and **sequence** (e.g. Hamming to references:
    ``checkpoint_eval_metrics_sequence.png``). The unique-per-campaign variants are
    ``..._unique_af3.png`` and ``..._unique_sequence.png`` unless disabled. GDF8/LTK
    only: ``..._gdf8_af3.png`` / ``..._gdf8_sequence.png`` (and ``..._unique_...``).
    Use ``--no-split-figures`` to skip family PNGs. Override with ``--output-figure-(af3|sequence)*``.
  - Pairwise library diversity: ``checkpoint_eval_pairwise_diversity.png`` (within each
    campaign, distribution of all pairwise Hamming distances among unique sequences, as a
    fraction of sequence length). GDF8/LTK splits: ``..._diversity_gdf8.png`` / ``..._ltk.png``.
    Use ``--no-pairwise-diversity-figure`` to skip, ``--max-sequences-diversity N`` to cap
    per-campaign subsampling (default 600).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cleo.plot_style import campaign_color_map, setup_style

_VANILLA_MANIFEST_INDEX_START = 10_000
_EXTRA_EVAL_MANIFEST_INDEX_START = 20_000
_DEFAULT_MAX_SEQUENCES_FOR_PAIRWISE = 600
_DIVERSITY_SUBSAMPLE_SEED = 42


def _repo_root() -> Path:
    """antibody_opt repo (parent of the ``cleo/`` package)."""
    return Path(__file__).resolve().parent.parent.parent


def _default_paths(
    manifest: Path,
) -> dict[str, Path]:
    b = manifest.parent
    return {
        "csv": b / "checkpoint_eval_combined.csv",
        "csv_uq": b / "checkpoint_eval_combined_unique_per_campaign.csv",
        "fig_af3": b / "checkpoint_eval_metrics_af3.png",
        "fig_seq": b / "checkpoint_eval_metrics_sequence.png",
        "fig_uq_af3": b / "checkpoint_eval_metrics_unique_af3.png",
        "fig_uq_seq": b / "checkpoint_eval_metrics_unique_sequence.png",
    }


def _family_figure_paths(manifest: Path) -> dict[str, Path]:
    b = manifest.parent
    return {
        "gdf8_all_af3": b / "checkpoint_eval_metrics_gdf8_af3.png",
        "gdf8_all_seq": b / "checkpoint_eval_metrics_gdf8_sequence.png",
        "ltk_all_af3": b / "checkpoint_eval_metrics_ltk_af3.png",
        "ltk_all_seq": b / "checkpoint_eval_metrics_ltk_sequence.png",
        "gdf8_uq_af3": b / "checkpoint_eval_metrics_unique_gdf8_af3.png",
        "gdf8_uq_seq": b / "checkpoint_eval_metrics_unique_gdf8_sequence.png",
        "ltk_uq_af3": b / "checkpoint_eval_metrics_unique_ltk_af3.png",
        "ltk_uq_seq": b / "checkpoint_eval_metrics_unique_ltk_sequence.png",
    }


def _diversity_figure_paths(manifest: Path) -> dict[str, Path]:
    b = manifest.parent
    return {
        "all": b / "checkpoint_eval_pairwise_diversity.png",
        "gdf8": b / "checkpoint_eval_pairwise_diversity_gdf8.png",
        "ltk": b / "checkpoint_eval_pairwise_diversity_ltk.png",
    }


def _filter_by_campaign_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """``prefix``: ``"gdf8"`` or ``"ltk"`` (case-insensitive)."""
    if df.empty or "campaign" not in df.columns:
        return df
    c = df["campaign"].astype(str)
    p = prefix.lower()
    m = c.str.lower().str.startswith(p)
    return df.loc[m].copy()


def load_manifest_tasks(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON list of task objects")
    return data


def evaluation_csv_for_task(task: dict) -> Path:
    run_dir = Path(task["run_dir"])
    tag = str(task["tag"])
    return run_dir / "checkpoint_eval" / tag / "evaluation.csv"


def collect_checkpoint_frames(
    tasks: list[dict], skip_missing: bool
) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for i, task in enumerate(tasks):
        label = str(task.get("label", f"task_{i}"))
        tag = str(task["tag"])
        path = evaluation_csv_for_task(task)
        if not path.is_file():
            missing.append(f"{label} / {tag} -> {path}")
            if not skip_missing:
                raise FileNotFoundError(f"Missing evaluation CSV: {path}")
            continue

        df = pd.read_csv(path)
        df.insert(0, "checkpoint_tag", tag)
        df.insert(0, "campaign", label)
        df["manifest_index"] = i
        df["eval_source"] = "checkpoint_eval"
        frames.append(df)

    return frames, missing


def collect_vanilla_frames(vanilla_root: Path | None) -> list[pd.DataFrame]:
    """Load evaluation.csv under vanilla_root (recursive), excluding nested checkpoint_eval trees."""
    if vanilla_root is None or not vanilla_root.is_dir():
        return []

    frames: list[pd.DataFrame] = []
    # Prefer stable order: path string sort
    for path in sorted(vanilla_root.rglob("evaluation.csv"), key=lambda p: str(p)):
        parts = path.parts
        if "checkpoint_eval" in parts:
            continue
        rel = path.relative_to(vanilla_root)
        if len(rel.parts) == 1:
            # evaluation.csv sitting directly in vanilla root — no per-run label
            campaign = vanilla_root.name + "_flat_eval"
        else:
            # e.g. gdf8_T0.1_vanilla/evaluation.csv -> campaign gdf8_T0.1_vanilla
            campaign = rel.parts[0]

        df = pd.read_csv(path)
        df.insert(0, "checkpoint_tag", "vanilla_mpnn")
        df.insert(0, "campaign", campaign)
        idx = _VANILLA_MANIFEST_INDEX_START + len(frames)
        df["manifest_index"] = idx
        df["eval_source"] = "vanilla_mpnn"
        frames.append(df)

    return frames


def collect_extra_eval_frames(
    roots: list[Path], eval_source: str
) -> list[pd.DataFrame]:
    """Load ``evaluation.csv`` from extra directories (e.g. gdf8 temp-sweep run folders)."""
    frames: list[pd.DataFrame] = []
    n = 0
    for root in roots:
        r = root.resolve()
        if not r.is_dir():
            continue
        for path in sorted(r.rglob("evaluation.csv"), key=lambda p: str(p)):
            if "checkpoint_eval" in path.parts:
                continue
            rel = path.relative_to(r)
            if len(rel.parts) == 1:
                campaign = f"{r.name}_flat"
            else:
                campaign = rel.parts[0]

            df = pd.read_csv(path)
            df.insert(0, "checkpoint_tag", "extra")
            df.insert(0, "campaign", campaign)
            df["manifest_index"] = _EXTRA_EVAL_MANIFEST_INDEX_START + n
            df["eval_source"] = eval_source
            frames.append(df)
            n += 1
    return frames


def _add_per_sequence_mutation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill per-sequence mutation metrics from the `sequence` column, computed within
    each campaign over the same-length rows:

    - ``mutation_diversity_total_muts_vs_consensus``: Hamming distance to the per-position
      mode-consensus of the campaign (ties broken by lexicographic AA).
    - ``mutation_diversity_pairwise_hamming_mean``: mean Hamming distance to all *other*
      rows in the same campaign.
    - ``mutation_diversity_pairwise_hamming_min``: min Hamming distance to any other row
      in the same campaign (nearest neighbour; 0 ⇒ duplicate sequence).
    - ``mutation_diversity_unique_mut_sites_vs_consensus``: campaign-level count of
      distinct ``(position, AA)`` pairs observed as a deviation from the campaign's
      consensus sequence. Broadcast: every row in the campaign gets the same value.

    Rows whose sequence length does not match the campaign's dominant length, or campaigns
    with fewer than two usable rows, are left as NaN. Uses the `mutation_diversity_`
    prefix so they route to the sequence-metric PNG via the existing classifier.
    """
    cons_col = "mutation_diversity_total_muts_vs_consensus"
    mean_col = "mutation_diversity_pairwise_hamming_mean"
    min_col = "mutation_diversity_pairwise_hamming_min"
    uniq_col = "mutation_diversity_unique_mut_sites_vs_consensus"
    if "sequence" not in df.columns or "campaign" not in df.columns:
        return df
    for c in (cons_col, mean_col, min_col, uniq_col):
        df[c] = np.nan

    length_warnings: list[tuple[str, int]] = []
    for camp, sub in df.groupby("campaign", sort=False):
        seqs = sub["sequence"].astype(str).tolist()
        lens = [len(s) for s in seqs]
        if not lens:
            continue
        L = Counter(lens).most_common(1)[0][0]
        keep_mask = np.array([ln == L for ln in lens], dtype=bool)
        n_drop = int((~keep_mask).sum())
        if n_drop:
            length_warnings.append((str(camp), n_drop))
        if keep_mask.sum() < 2:
            continue
        kept_idx = sub.index.to_numpy()[keep_mask]
        kept_seqs = [seqs[i] for i in range(len(seqs)) if keep_mask[i]]
        X = np.vstack(
            [np.frombuffer(s.encode("ascii"), dtype=np.uint8) for s in kept_seqs]
        )
        consensus = np.empty(L, dtype=np.uint8)
        for j in range(L):
            vals, counts = np.unique(X[:, j], return_counts=True)
            ties = vals[counts == counts.max()]
            consensus[j] = np.sort(ties)[0]
        d_cons = (X != consensus).sum(axis=1).astype(np.int64)

        D = (X[:, None, :] != X[None, :, :]).sum(axis=2).astype(np.int64)
        n = D.shape[0]
        row_sum = D.sum(axis=1)
        means = row_sum / max(n - 1, 1)
        D_for_min = D.copy()
        np.fill_diagonal(D_for_min, np.iinfo(np.int64).max)
        mins = D_for_min.min(axis=1)

        diff_rows, diff_cols = np.where(X != consensus)
        unique_mut_sites = int(
            len(set(zip(diff_cols.tolist(), X[diff_rows, diff_cols].tolist())))
        )

        df.loc[kept_idx, cons_col] = d_cons
        df.loc[kept_idx, mean_col] = means
        df.loc[kept_idx, min_col] = mins
        df.loc[kept_idx, uniq_col] = unique_mut_sites

    for camp, n_drop in length_warnings:
        print(
            f"Per-sequence mutation metrics: campaign {camp!r} dropped {n_drop} row(s) "
            "of non-dominant length (left as NaN).",
            file=sys.stderr,
        )
    return df


def _plot_frame_unique_sequence_per_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """
    For stripplots, one point per (campaign, sequence) so duplicate designs do not stack.
    The combined CSV is unchanged; this is only the view used for plotting.
    """
    if "sequence" not in df.columns or "campaign" not in df.columns:
        return df
    n0 = len(df)
    out = df.drop_duplicates(subset=["campaign", "sequence"], keep="first", ignore_index=True)
    if n0 > len(out):
        print(
            f"Plot: {n0} -> {len(out)} rows (unique sequence per campaign; first row kept).",
            file=sys.stderr,
        )
    return out


def metric_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "campaign",
        "checkpoint_tag",
        "manifest_index",
        "eval_source",
        "name",
        "sequence",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def is_sequence_only_metric_name(col: str) -> bool:
    """
    e.g. ``dist_to_ref_seqs_*`` from the sequence step, and batch sequence-diversity
    (``mutation_diversity_*``); not AF3 / structure prediction.
    """
    t = col.lower()
    if "dist_to_ref" in t or "ref_seq" in t:
        return True
    if "mutation_diversity" in t:
        return True
    return False


def is_structure_af3_metric_name(col: str) -> bool:
    """AF3, Boltz, mfold, or other structure-step outputs; excludes sequence-only ref distances."""
    if is_sequence_only_metric_name(col):
        return False
    t = col.lower()
    for needle in (
        "plddt",
        "pae",
        "pde",
        "ipde",
        "iplddt",
        "protein_iptm",
        "ligand_iptm",
        "chain_iptm",
        "chain_ptm",
        "ranking",
        "fraction_disordered",
        "has_clash",
        "interaction_pae",
        "confidence_score",
        "boltz_",
        "mfold3",
        "mfold",
    ):
        if needle in t:
            return True
    for suf in ("_ptm", "_iptm", "_pde", "_ipde", "_pae"):
        if t.endswith(suf) and "dist" not in t and "to_ref" not in t and "ref_seq" not in t:
            return True
    if "_complex_" in t and any(x in t for x in ("plddt", "pde", "pae", "i")):
        return True
    return False


def split_metrics_af3_sequence(
    columns: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Return ``(af3_and_structure, sequence, unclassified)`` from numeric metric column
    names. Unclassified (rare) columns are printed and appended to the structure group
    for plotting.
    """
    af3: list[str] = []
    seq: list[str] = []
    oth: list[str] = []
    for c in columns:
        if is_sequence_only_metric_name(c):
            seq.append(c)
        elif is_structure_af3_metric_name(c):
            af3.append(c)
        else:
            oth.append(c)
    return af3, seq, oth


def _stripplot_suptitle(
    df: pd.DataFrame,
    data_scope: str,
    family: str | None = None,
    metric_group: str | None = None,
) -> str:
    """
    ``data_scope``: all_rows | unique_per_campaign | standard. ``family``: gdf8 | ltk | None
    ``metric_group``: "af3" (structure) | "seq" (sequence) | None
    """
    has_vanilla = bool(
        "eval_source" in df.columns
        and (df["eval_source"] == "vanilla_mpnn").any()
    )
    if has_vanilla:
        base = "Checkpoint eval + vanilla MPNN metrics"
    else:
        base = "Checkpoint eval metrics"
    if data_scope == "unique_per_campaign" and "sequence" in df.columns:
        mid = f"{base} (unique sequence per campaign)"
    elif data_scope == "all_rows" and "sequence" in df.columns:
        mid = f"{base} (all rows, duplicate sequences shown)"
    else:
        mid = f"{base} by campaign"
    if metric_group == "af3":
        g = "AF3 & structure: "
    elif metric_group == "seq":
        g = "Sequence metrics: "
    else:
        g = ""
    mid = f"{g}{mid}"
    if family == "gdf8":
        return f"GDF8: {mid}"
    if family == "ltk":
        return f"LTK: {mid}"
    return mid


def plot_metrics(
    df: pd.DataFrame,
    metrics: list[str],
    out_path: Path,
    *,
    data_scope: str = "standard",
    family: str | None = None,
    metric_group: str | None = None,
) -> None:
    n = len(metrics)
    if n == 0:
        print("No numeric metric columns to plot.", file=sys.stderr)
        return

    setup_style()

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    n_cat = int(df["campaign"].nunique()) if "campaign" in df.columns else 1
    # Wider when many campaign columns so x labels and ticks stay readable
    w = min(max(5.0 * ncols, 1.5 + 0.42 * n_cat * ncols), 36.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(w, 3.9 * nrows + 0.2),
        squeeze=False,
    )

    campaigns = df["campaign"].astype(str)
    palette = campaign_color_map(campaigns.unique().tolist())

    for ax, metric in zip(axes.flat, metrics):
        sub = df[["campaign", metric]].dropna(subset=[metric])
        if sub.empty:
            ax.set_visible(False)
            continue
        order = sorted(sub["campaign"].astype(str).unique())
        # hue is redundant with x; dodge=True would offset markers vs tick positions.
        sns.stripplot(
            data=sub,
            x="campaign",
            y=metric,
            order=order,
            hue="campaign",
            hue_order=order,
            palette=palette,
            dodge=False,
            jitter=0.3,
            size=2.8,
            alpha=0.65,
            linewidth=0,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.margins(x=0.02)
        for tl in ax.get_xticklabels():
            tl.set_rotation(28)
            tl.set_ha("right")
            tl.set_rotation_mode("anchor")
        ax.set_title(metric, fontsize=10)

    for j in range(len(metrics), nrows * ncols):
        axes.flat[j].set_visible(False)

    title = _stripplot_suptitle(
        df, data_scope, family=family, metric_group=metric_group
    )
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=1.2, h_pad=1.0, w_pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _pairwise_hamming_upper_tril(sequences: list[str]) -> np.ndarray:
    """All pairwise Hamming counts for i < j, memory-friendly (O(n) rows, O(n) seq storage)."""
    n = len(sequences)
    if n < 2:
        return np.empty(0, dtype=np.int32)
    for s in sequences:
        s.encode("ascii", errors="strict")
    l0 = len(sequences[0])
    if not all(len(s) == l0 for s in sequences):
        raise ValueError("all sequences in a list must be the same length")
    x = np.vstack(
        [np.frombuffer(s.encode("ascii"), dtype=np.uint8) for s in sequences]
    )
    blocks: list[np.ndarray] = []
    for i in range(0, n - 1):
        diff = (x[i + 1 :] != x[i]).sum(axis=1)
        blocks.append(diff.astype(np.int32, copy=False))
    return np.concatenate(blocks, axis=0)


def _length_homogeneous_uniques(sequences: list[str]) -> tuple[list[str], int, int] | None:
    """Return (same-length uniques, that length, n_dropped) or None if <2 left."""
    u = list(dict.fromkeys(sequences))
    if len(u) < 2:
        return None
    c = Counter(len(s) for s in u)
    l_best, _ = c.most_common(1)[0]
    out = [s for s in u if len(s) == l_best]
    dropped = len(u) - len(out)
    if len(out) < 2:
        return None
    return out, l_best, dropped


def _subsample_list(seqs: list[str], k: int, rng: np.random.Generator) -> list[str]:
    if len(seqs) <= k:
        return seqs
    idx = rng.choice(len(seqs), size=k, replace=False)
    return [seqs[i] for i in sorted(idx)]


def build_pairwise_diversity_long(
    df: pd.DataFrame,
    *,
    max_seqs: int,
    seed: int = _DIVERSITY_SUBSAMPLE_SEED,
) -> tuple[pd.DataFrame, bool, list[tuple[str, int]]] | None:
    """
    One row per unordered pair: ``campaign``, ``hamming`` (int), ``frac_diff`` (0–1), ``seq_len``.

    If any campaign is subsampled (too many unique seqs), ``subsampled`` is True.
    Warnings: list of (campaign, n_dropped_length) for non-modal lengths dropped.
    """
    if "sequence" not in df.columns or "campaign" not in df.columns:
        return None

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    subsampled = False
    warnings: list[tuple[str, int]] = []

    for camp, g in df.groupby("campaign", sort=True):
        seqs = g["sequence"].dropna().astype(str).tolist()
        hom = _length_homogeneous_uniques(seqs)
        if hom is None:
            continue
        hom_s, l_best, d_len = hom
        if d_len:
            warnings.append((str(camp), d_len))
        if len(hom_s) > max_seqs:
            hom_s = _subsample_list(hom_s, max_seqs, rng)
            subsampled = True
        dvec = _pairwise_hamming_upper_tril(hom_s)
        for d in dvec:
            d_i = int(d)
            rows.append(
                {
                    "campaign": str(camp),
                    "hamming": d_i,
                    "frac_diff": d_i / float(l_best) if l_best else float("nan"),
                    "seq_len": l_best,
                }
            )

    if not rows:
        return None
    return pd.DataFrame(rows), subsampled, warnings


def _diversity_suptitle(
    family: str | None,
    *,
    subsampled: bool,
    max_seqs: int,
) -> str:
    t = "Pairwise library diversity: Hamming distance (fraction of positions differing) within each campaign; all unique pairings"
    if family == "gdf8":
        t = f"GDF8: {t}"
    if family == "ltk":
        t = f"LTK: {t}"
    if subsampled:
        t += f". Subsample: ≤{max_seqs} sequences per campaign (seed={_DIVERSITY_SUBSAMPLE_SEED})"
    return t


def plot_pairwise_diversity(
    long_df: pd.DataFrame,
    out_path: Path,
    *,
    family: str | None = None,
    subsampled: bool,
    max_seqs: int,
) -> bool:
    """``long_df`` from :func:`build_pairwise_diversity_long`. Return True if written."""
    if long_df is None or long_df.empty or "frac_diff" not in long_df.columns:
        return False

    setup_style()
    order = sorted(long_df["campaign"].astype(str).unique(), key=str)
    n_cat = len(order)
    w = min(max(5.0, 1.2 + 0.4 * n_cat), 36.0)
    fig, ax = plt.subplots(1, 1, figsize=(w, 4.2))

    palette = campaign_color_map(list(order))
    sns.violinplot(
        data=long_df,
        x="campaign",
        y="frac_diff",
        order=order,
        palette=[palette.get(c, "#5B8DB8") for c in order],
        cut=0,
        inner="box",
        scale="width",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Pairwise position difference (Hamming / length)", fontsize=10)
    for tl in ax.get_xticklabels():
        tl.set_rotation(28)
        tl.set_ha("right")
        tl.set_rotation_mode("anchor")
    ax.margins(x=0.02)
    title = _diversity_suptitle(family, subsampled=subsampled, max_seqs=max_seqs)
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.2, h_pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "slurm_checkpoint_eval_bundle"
        / "manifest.json",
        help="Path to manifest.json from the checkpoint eval bundle",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Combined CSV path (default: beside manifest)",
    )
    parser.add_argument(
        "--output-figure-af3",
        type=Path,
        default=None,
        help="AF3/structure stripplot (default: checkpoint_eval_metrics_af3.png beside manifest)",
    )
    parser.add_argument(
        "--output-figure-sequence",
        type=Path,
        default=None,
        help="Sequence (e.g. dist-to-ref) stripplot (default: checkpoint_eval_metrics_sequence.png)",
    )
    parser.add_argument(
        "--output-figure-unique-af3",
        type=Path,
        default=None,
        help="Unique/campaign: AF3/structure (default: checkpoint_eval_metrics_unique_af3.png)",
    )
    parser.add_argument(
        "--output-figure-unique-sequence",
        type=Path,
        default=None,
        help="Unique/campaign: sequence metrics (default: ..._unique_sequence.png)",
    )
    parser.add_argument(
        "--no-output-figure-unique-per-campaign",
        action="store_true",
        help="Do not write the *unique* AF3/sequence pair of figures",
    )
    parser.add_argument(
        "--no-split-figures",
        action="store_true",
        help="Do not write gdf8/ltk-only split PNGs (checkpoint_eval_metrics_*.png)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip checkpoint manifest rows with no evaluation.csv instead of failing",
    )
    parser.add_argument(
        "--vanilla-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "cleo_runs"
        / "vanilla_mpnn_baseline",
        help="Also load **/evaluation.csv under this directory (set with --no-vanilla to disable)",
    )
    parser.add_argument(
        "--no-vanilla",
        action="store_true",
        help="Do not search for vanilla MPNN baseline evaluation CSVs",
    )
    parser.add_argument(
        "--extra-eval-root",
        type=Path,
        action="append",
        default=[],
        help="Extra root directory to search for **/evaluation.csv (repeat for multiple).",
    )
    parser.add_argument(
        "--no-auto-temp-sweep",
        action="store_true",
        help="Do not auto-include <repo>/cleo_runs/gdf8_vhh/temp_sweep_w0 (finetuned T sweep) if it exists.",
    )
    parser.add_argument(
        "--no-pairwise-diversity-figure",
        action="store_true",
        help="Do not write pairwise within-campaign library diversity (Hamming) PNGs.",
    )
    parser.add_argument(
        "--max-sequences-diversity",
        type=int,
        default=_DEFAULT_MAX_SEQUENCES_FOR_PAIRWISE,
        help=(
            "Subsample to at most this many unique sequences per campaign when computing "
            f"pairwise diversity (default {_DEFAULT_MAX_SEQUENCES_FOR_PAIRWISE})."
        ),
    )
    parser.add_argument(
        "--no-plot-unique-sequences",
        action="store_true",
        help="Do not dedupe: plot one point per CSV row (include duplicate sequences).",
    )
    parser.add_argument(
        "--output-csv-unique-per-campaign",
        type=Path,
        default=None,
        help=(
            "Where to write the deduplicated (campaign+sequence) table. "
            "Default: next to the manifest as checkpoint_eval_combined_unique_per_campaign.csv. "
            "Omitted if --no-output-csv-unique-per-campaign."
        ),
    )
    parser.add_argument(
        "--no-output-csv-unique-per-campaign",
        action="store_true",
        help="Do not write the separate unique-per-campaign CSV (default is to write when using plot dedupe).",
    )
    args = parser.parse_args()
    plot_unique = not args.no_plot_unique_sequences
    split_figures = not args.no_split_figures

    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    dp = _default_paths(manifest)
    spaths = _family_figure_paths(manifest)
    dpaths = _diversity_figure_paths(manifest)
    out_csv = (args.output_csv or dp["csv"]).resolve()
    out_fig_af3 = (args.output_figure_af3 or dp["fig_af3"]).resolve()
    out_fig_seq = (args.output_figure_sequence or dp["fig_seq"]).resolve()
    out_fig_uq_af3 = (args.output_figure_unique_af3 or dp["fig_uq_af3"]).resolve()
    out_fig_uq_seq = (args.output_figure_unique_sequence or dp["fig_uq_seq"]).resolve()

    tasks = load_manifest_tasks(manifest)
    ck_frames, missing = collect_checkpoint_frames(tasks, skip_missing=args.skip_missing)

    if missing:
        print("Skipped missing checkpoint manifest CSVs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    vanilla_root = None if args.no_vanilla else args.vanilla_root.resolve()
    v_frames = collect_vanilla_frames(vanilla_root)

    if vanilla_root is not None:
        if v_frames:
            print(
                f"Loaded {len(v_frames)} vanilla baseline evaluation.csv "
                f"from {vanilla_root}",
                file=sys.stderr,
            )
        else:
            print(
                f"No evaluation.csv under {vanilla_root} (optional; run evaluate_sequences per baseline run).",
                file=sys.stderr,
            )

    extra_roots: list[Path] = [p.resolve() for p in (args.extra_eval_root or [])]
    auto_ts = _repo_root() / "cleo_runs" / "gdf8_vhh" / "temp_sweep_w0"
    if not args.no_auto_temp_sweep and auto_ts.is_dir():
        ar = auto_ts.resolve()
        if all(r != ar for r in extra_roots):
            extra_roots.append(ar)
            print(f"Auto-including finetuned temp-sweep: {ar}", file=sys.stderr)
    e_frames = collect_extra_eval_frames(extra_roots, "temp_sweep")
    if e_frames:
        print(
            f"Loaded {len(e_frames)} extra evaluation file(s) (e.g. T sweep) from {len(extra_roots)} root(s).",
            file=sys.stderr,
        )

    all_frames = ck_frames + v_frames + e_frames
    if not all_frames:
        raise RuntimeError(
            "No evaluation CSVs found (checkpoint manifest, vanilla, and extra roots all empty)."
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined = _add_per_sequence_mutation_metrics(combined)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"Wrote {len(combined)} rows -> {out_csv}")

    if plot_unique and "sequence" in combined.columns and "campaign" in combined.columns:
        plot_df = _plot_frame_unique_sequence_per_campaign(combined)
        suptitle_dedup = True
    else:
        plot_df = combined
        suptitle_dedup = False

    if (
        plot_unique
        and not args.no_output_csv_unique_per_campaign
        and "sequence" in combined.columns
        and "campaign" in combined.columns
    ):
        u_path = (args.output_csv_unique_per_campaign or dp["csv_uq"]).resolve()
        u_path.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(u_path, index=False)
        print(
            f"Wrote {len(plot_df)} rows (unique sequence per campaign) -> {u_path}"
        )

    raw_metrics = metric_columns(combined)
    m_af3, m_seq, m_other = split_metrics_af3_sequence(raw_metrics)
    if m_other:
        m_af3 = m_af3 + m_other
        print(
            "Unclassified metric columns (grouped with AF3/structure): "
            f"{m_other!s}",
            file=sys.stderr,
        )
    m_af3_c = m_af3
    m_seq_c = m_seq
    m_af3_p = f"{len(m_af3_c)}"
    m_seq_p = f"{len(m_seq_c)}"

    def _emit_bundles(
        comb: pd.DataFrame,
        p_df: pd.DataFrame,
        *,
        family: str | None,
        p_af3: Path,
        p_seq: Path,
        p_uq_af3: Path,
        p_uq_seq: Path,
    ) -> None:
        for mg, mlist, pout, p_uq in [
            ("af3", m_af3_c, p_af3, p_uq_af3),
            ("seq", m_seq_c, p_seq, p_uq_seq),
        ]:
            if not mlist:
                tag = f"{(family or 'all')!s}"
                print(
                    f"Skip {tag} {mg}-metric figure: no columns in that group.",
                    file=sys.stderr,
                )
                continue
            if not plot_unique or not suptitle_dedup:
                print(f"  → {pout} ({mg})")
                plot_metrics(
                    comb, mlist, pout, data_scope="standard", family=family, metric_group=mg
                )
            else:
                print(f"  → {pout} + {p_uq} ({mg}, {len(comb)} vs {len(p_df)} pts)")
                plot_metrics(
                    comb, mlist, pout, data_scope="all_rows", family=family, metric_group=mg
                )
                if not args.no_output_figure_unique_per_campaign:
                    plot_metrics(
                        p_df,
                        mlist,
                        p_uq,
                        data_scope="unique_per_campaign",
                        family=family,
                        metric_group=mg,
                    )

    if not m_af3_c and not m_seq_c:
        print("No plottable numeric columns after split; no metric stripplots written.", file=sys.stderr)
    else:
        if not plot_unique or not suptitle_dedup:
            print(
                f"Plotting AF3+structure: {m_af3_p} cols, sequence: {m_seq_p} numeric metrics "
                f"-> {out_fig_af3} / {out_fig_seq}"
            )
        else:
            print(
                f"Plotting ALL rows: AF3+structure {m_af3_p} cols, sequence {m_seq_p} cols; "
                f"unique: {out_fig_uq_af3} / {out_fig_uq_seq}"
            )
        _emit_bundles(
            combined,
            plot_df,
            family=None,
            p_af3=out_fig_af3,
            p_seq=out_fig_seq,
            p_uq_af3=out_fig_uq_af3,
            p_uq_seq=out_fig_uq_seq,
        )

    if split_figures and "campaign" in combined.columns:
        for name, pfx, pth in [
            ("GDF8", "gdf8", "gdf8"),
            ("LTK", "ltk", "ltk"),
        ]:
            gdf8_or_ltk = _filter_by_campaign_prefix(combined, pth)
            if gdf8_or_ltk.empty:
                print(
                    f"Skip {name}-only figure: no campaigns starting with {pth!r}.",
                    file=sys.stderr,
                )
                continue
            p_all_af3 = spaths[f"{pfx}_all_af3"]
            p_all_seq = spaths[f"{pfx}_all_seq"]
            p_uq_a = spaths[f"{pfx}_uq_af3"]
            p_uq_s = spaths[f"{pfx}_uq_seq"]
            g_u = (
                _filter_by_campaign_prefix(plot_df, pth)
                if (plot_unique and suptitle_dedup)
                else gdf8_or_ltk
            )
            _emit_bundles(
                gdf8_or_ltk,
                g_u,
                family=pth,
                p_af3=p_all_af3,
                p_seq=p_all_seq,
                p_uq_af3=p_uq_a,
                p_uq_seq=p_uq_s,
            )

    if (
        not args.no_pairwise_diversity_figure
        and "sequence" in combined.columns
        and "campaign" in combined.columns
    ):
        div_max = int(args.max_sequences_diversity)
        div_src = _plot_frame_unique_sequence_per_campaign(combined)
        div_result = build_pairwise_diversity_long(
            div_src, max_seqs=div_max, seed=_DIVERSITY_SUBSAMPLE_SEED
        )
        if div_result is None:
            print(
                "Pairwise diversity: no data (need ≥2 same-length unique seqs in some campaign); skipped.",
                file=sys.stderr,
            )
        else:
            long_d, any_sub, length_warns = div_result
            for camp, dropped in length_warns:
                print(
                    f"Pairwise diversity: campaign {camp!r} dropped {dropped} unique sequence(s) "
                    "of non-dominant length (dominant length kept for pairs).",
                    file=sys.stderr,
                )
            if plot_pairwise_diversity(
                long_d,
                dpaths["all"],
                family=None,
                subsampled=any_sub,
                max_seqs=div_max,
            ):
                print(f"Wrote pairwise diversity (all) -> {dpaths['all']}")
            if split_figures:
                for pfx, fkey, path_key in [
                    ("gdf8", "gdf8", "gdf8"),
                    ("ltk", "ltk", "ltk"),
                ]:
                    l_sub = _filter_by_campaign_prefix(long_d, pfx)
                    if l_sub.empty:
                        print(
                            f"Skip {pfx.upper()}-only pairwise diversity: no rows after prefix filter.",
                            file=sys.stderr,
                        )
                        continue
                    dest = dpaths[path_key]
                    if plot_pairwise_diversity(
                        l_sub,
                        dest,
                        family=fkey,
                        subsampled=any_sub,
                        max_seqs=div_max,
                    ):
                        print(f"Wrote pairwise diversity ({pfx}) -> {dest}")


if __name__ == "__main__":
    main()
