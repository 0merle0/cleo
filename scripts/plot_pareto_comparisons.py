"""Pareto-style comparison plots across vanilla and RL-finetuned ProteinMPNN runs.

Reads the combined CSV from compile_checkpoint_eval_bundle.py and produces
per-campaign plots emphasizing the quality/diversity tradeoff at matched
mutation budget.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUNDLE = Path("/home/jgershon/projects/antibody_opt/cleo/slurm_checkpoint_eval_bundle")
CSV = BUNDLE / "checkpoint_eval_combined.csv"
OUT = BUNDLE


def classify(row: pd.Series) -> str:
    src = row["eval_source"]
    if src == "vanilla_mpnn":
        return "vanilla"
    if src == "temp_sweep":
        return "temp_sweep"
    return "finetune"


def short_run(label: str) -> str:
    return (
        label.replace("gdf8_lep_", "")
        .replace("ltk_", "")
        .replace("_consensus", "+cons")
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["campaign_base", "run", "kind", "tag"], dropna=False)
    agg = g.agg(
        n=("af3_iptm", "size"),
        iptm=("af3_iptm", "mean"),
        ptm=("af3_ptm", "mean"),
        ipae_min=("af3_interaction_pae_min", "mean"),
        ipae_best=("af3_interaction_pae_min", "min"),
        chain_ptm_lig=("af3_chain_ptm_ligand", "mean"),
        total_muts=("mutation_diversity_total_muts", "mean"),
        dist_to_ref=("dist_to_ref_seqs_min", "mean"),
        marginal_vs_cons=("mutation_diversity_marginal_count_vs_consensus", "mean"),
        unique_sites_vs_cons=("mutation_diversity_unique_mut_sites_vs_consensus", "mean"),
        pairwise_hamming=("mutation_diversity_pairwise_hamming_mean", "mean"),
    ).reset_index()
    agg["unique_sites_per_seq"] = agg["unique_sites_vs_cons"] / agg["n"]
    return agg


def annotate_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # campaign_base = gdf8 / ltk
    df["campaign_base"] = df["campaign"].apply(
        lambda c: "gdf8" if c.startswith("gdf8") else "ltk"
    )
    df["kind"] = df.apply(classify, axis=1)

    def run_label(row):
        if row["kind"] == "vanilla":
            # campaigns look like ltk_T0.1_vanilla
            t = row["campaign"].split("_")[1]
            return f"vanilla_{t}"
        if row["kind"] == "temp_sweep":
            t = row["campaign"].split("_")[-1]
            return f"w0_{t}"
        return short_run(row["campaign"])

    df["run"] = df.apply(run_label, axis=1)
    df["tag"] = df["checkpoint_tag"]
    return df


# ordering and color palette per campaign (run-level)
GDF8_ORDER = [
    "vanilla_T0.1", "vanilla_T0.2",
    "w0_T0.1", "w0_T0.2", "w0_T0.5", "w0_T0.75",
    "w0", "w1", "distw5+cons", "distw8+cons",
]
LTK_ORDER = [
    "vanilla_T0.1", "vanilla_T0.2",
    "v2", "v3", "distw2+cons", "distw8+cons",
]

PALETTE = {
    # vanilla (greys)
    "vanilla_T0.1": "#888888",
    "vanilla_T0.2": "#bbbbbb",
    # gdf8 temp sweep (greens)
    "w0_T0.1": "#a1d99b", "w0_T0.2": "#74c476",
    "w0_T0.5": "#41ab5d", "w0_T0.75": "#238b45",
    # finetune progression (gdf8: blue→red)
    "w0": "#3182bd", "w1": "#9ecae1",
    "distw5+cons": "#fd8d3c", "distw8+cons": "#d94801",
    # ltk
    "v2": "#3182bd", "v3": "#9ecae1",
    "distw2+cons": "#fd8d3c",
}
# avoid duplicate-key clobbering: ltk distw8+cons same as gdf8
PALETTE_LTK = dict(PALETTE)
PALETTE_LTK["distw8+cons"] = "#d94801"

TAG_MARKER = {
    "step_0400": "o",
    "step_0450": "s",
    "best": "*",
    "last": "D",
    "vanilla_mpnn": "P",
    "extra": "X",
}


def _scatter(ax, sub, palette, x, y, lo_better_y=False, annotate_best=True):
    for run, run_df in sub.groupby("run"):
        color = palette.get(run, "#444444")
        for _, r in run_df.iterrows():
            marker = TAG_MARKER.get(r["tag"], "o")
            ax.scatter(
                r[x], r[y],
                color=color,
                marker=marker,
                s=180 if marker == "*" else 90,
                edgecolor="black", linewidth=0.6,
                alpha=0.9,
                label=run if marker in ("*", "P", "X") else None,
            )
    ax.set_xlabel(x)
    ax.set_ylabel(y + (" (lower=better)" if lo_better_y else ""))
    ax.grid(alpha=0.25)


def make_pareto(agg: pd.DataFrame, campaign: str, palette: dict, order: list[str], outfile: Path):
    sub = agg[agg["campaign_base"] == campaign].copy()
    if sub.empty:
        print(f"no data for {campaign}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"{campaign.upper()} — quality vs mutation budget (per run × checkpoint)", fontsize=13)

    # 1: total_muts vs iptm
    _scatter(axes[0, 0], sub, palette, "total_muts", "iptm")
    axes[0, 0].set_title("AF3 iptm vs mean mutations/seq (higher iptm better)")

    # 2: total_muts vs interaction_pae_min (lower=better)
    _scatter(axes[0, 1], sub, palette, "total_muts", "ipae_min", lo_better_y=True)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("AF3 interaction_pae_min vs mean mutations/seq")

    # 3: total_muts vs unique_sites_vs_cons (diversity)
    _scatter(axes[1, 0], sub, palette, "total_muts", "unique_sites_vs_cons")
    axes[1, 0].set_title("Unique mutation sites vs batch consensus (higher better)")

    # 4: total_muts vs marginal_vs_cons (per-seq novelty)
    _scatter(axes[1, 1], sub, palette, "total_muts", "marginal_vs_cons")
    axes[1, 1].set_title("Per-seq marginal novel mutations vs consensus")

    # build legend: run colors + tag markers
    handles = []
    for run in order:
        if run in sub["run"].values:
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=palette.get(run, "#444"), markersize=10,
                           label=run, markeredgecolor="black")
            )
    handles.append(plt.Line2D([0], [0], marker="", color="w", label=" "))
    for tag, marker in [("step_0400", "o"), ("step_0450", "s"), ("best", "*"),
                         ("last", "D"), ("vanilla_mpnn", "P"), ("extra (Tsweep)", "X")]:
        m = "X" if tag.startswith("extra") else marker
        handles.append(
            plt.Line2D([0], [0], marker=m, color="black",
                       markerfacecolor="lightgrey", markersize=11,
                       label=tag, linestyle="")
        )
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile}")


def make_quality_diversity(agg: pd.DataFrame, campaign: str, palette: dict, outfile: Path):
    """Single-panel: best checkpoint per finetune run, quality (iptm) vs diversity (unique_sites_vs_cons),
    sized by total_muts; vanilla as star anchor."""
    sub = agg[agg["campaign_base"] == campaign].copy()
    if sub.empty:
        return
    # pick representative: vanilla -> the only entry; finetune -> 'best' tag
    rep = sub[(sub["kind"] == "vanilla") | (sub["tag"] == "best") | (sub["kind"] == "temp_sweep")]

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, r in rep.iterrows():
        color = palette.get(r["run"], "#444")
        marker = "*" if r["kind"] == "finetune" else ("P" if r["kind"] == "vanilla" else "X")
        size = 60 + r["total_muts"] * 12
        ax.scatter(r["iptm"], r["unique_sites_vs_cons"], s=size, color=color,
                   marker=marker, edgecolor="black", linewidth=0.7, alpha=0.9,
                   label=r["run"])
        ax.annotate(f"{r['run']}\n({r['total_muts']:.1f}mut)",
                    (r["iptm"], r["unique_sites_vs_cons"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("AF3 iptm (mean) — higher better")
    ax.set_ylabel("Unique mutation sites vs batch consensus — higher better")
    ax.set_title(f"{campaign.upper()} — quality vs library diversity (point size = mean muts/seq)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile}")


def subsample_unique_sites(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """For each campaign_base, find min N across all (run, tag) groups,
    subsample each group to that N, then recompute unique (pos, AA) mutation
    sites vs each group's own consensus on the subsampled rows.

    Returns a DataFrame: columns [campaign_base, run, kind, tag, n_sub,
    unique_sites_sub, ipae_min_sub, iptm_sub]."""
    rng = np.random.default_rng(seed)
    rows = []
    for camp_base, camp_df in df.groupby("campaign_base"):
        # group key: run × tag (one row per sequence inside)
        groups = list(camp_df.groupby(["run", "kind", "tag"]))
        sizes = [len(g) for _, g in groups]
        n_min = min(sizes)
        print(f"  {camp_base}: subsampling all checkpoints to N={n_min} "
              f"(was {min(sizes)}..{max(sizes)} across {len(groups)} groups)")
        for (run, kind, tag), g in groups:
            sub = g.sample(n=n_min, random_state=rng.integers(0, 2**31 - 1))
            seqs = sub["sequence"].astype(str).tolist()
            lens = [len(s) for s in seqs]
            L = max(set(lens), key=lens.count)
            mask = [ln == L for ln in lens]
            kept = [s for s, m in zip(seqs, mask) if m]
            if len(kept) < 2:
                continue
            X = np.vstack([np.frombuffer(s.encode("ascii"), dtype=np.uint8) for s in kept])
            consensus = np.empty(L, dtype=np.uint8)
            for j in range(L):
                vals, counts = np.unique(X[:, j], return_counts=True)
                ties = vals[counts == counts.max()]
                consensus[j] = np.sort(ties)[0]
            diff_rows, diff_cols = np.where(X != consensus)
            unique_sites = len(set(zip(diff_cols.tolist(),
                                        X[diff_rows, diff_cols].tolist())))
            rows.append(dict(
                campaign_base=camp_base, run=run, kind=kind, tag=tag,
                n_sub=n_min,
                unique_sites_sub=unique_sites,
                ipae_min_sub=sub["af3_interaction_pae_min"].mean(),
                iptm_sub=sub["af3_iptm"].mean(),
                total_muts_sub=sub["mutation_diversity_total_muts"].mean(),
            ))
    return pd.DataFrame(rows)


def make_subsampled_plot(sub_agg: pd.DataFrame, outfile: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, camp, palette, order in [
        (axes[0], "gdf8", PALETTE, GDF8_ORDER),
        (axes[1], "ltk", PALETTE_LTK, LTK_ORDER),
    ]:
        s = sub_agg[sub_agg["campaign_base"] == camp]
        if s.empty:
            continue
        n_sub = int(s["n_sub"].iloc[0])
        for run, run_df in s.groupby("run"):
            color = palette.get(run, "#444444")
            for _, r in run_df.iterrows():
                marker = TAG_MARKER.get(r["tag"], "o")
                ax.scatter(
                    r["unique_sites_sub"], r["ipae_min_sub"],
                    color=color, marker=marker,
                    s=240 if marker == "*" else 110,
                    edgecolor="black", linewidth=0.6, alpha=0.9,
                )
        ax.set_title(f"{camp.upper()} — subsampled to N={n_sub} per checkpoint")
        ax.set_xlabel(f"Unique (pos, AA) mutations vs consensus (N={n_sub} seqs)")
        ax.set_ylabel("AF3 interaction_pae_min — mean (lower=better)")
        ax.invert_yaxis()
        ax.grid(alpha=0.25)

        handles = []
        for run in order:
            if run in s["run"].values:
                handles.append(plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=palette.get(run, "#444"), markersize=9,
                    markeredgecolor="black", label=run))
        ax.legend(handles=handles, fontsize=7, loc="best", frameon=True)

    marker_handles = [
        plt.Line2D([0], [0], marker=m, color="black",
                   markerfacecolor="lightgrey", markersize=10,
                   linestyle="", label=t)
        for t, m in [("step_0400", "o"), ("step_0450", "s"), ("best", "*"),
                     ("last", "D"), ("vanilla", "P"), ("temp_sweep", "X")]
    ]
    fig.legend(handles=marker_handles, loc="lower center",
               ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Library novelty (unique mutations at matched N) vs interface quality",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile}")


def make_unique_vs_ipae(agg: pd.DataFrame, outfile: Path):
    """Two-panel (gdf8 / ltk): unique mut sites vs consensus / N  vs  mean min-iPAE.
    One point per (run × checkpoint). Color = run, marker = checkpoint tag."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    for ax, camp, palette, order in [
        (axes[0], "gdf8", PALETTE, GDF8_ORDER),
        (axes[1], "ltk", PALETTE_LTK, LTK_ORDER),
    ]:
        sub = agg[agg["campaign_base"] == camp]
        for run, run_df in sub.groupby("run"):
            color = palette.get(run, "#444444")
            for _, r in run_df.iterrows():
                marker = TAG_MARKER.get(r["tag"], "o")
                ax.scatter(
                    r["unique_sites_per_seq"], r["ipae_min"],
                    color=color, marker=marker,
                    s=220 if marker == "*" else 110,
                    edgecolor="black", linewidth=0.6, alpha=0.9,
                )
        ax.set_title(f"{camp.upper()}")
        ax.set_xlabel("Unique (pos, AA) mutations vs consensus / N seqs")
        ax.set_ylabel("AF3 interaction_pae_min — mean (lower=better)")
        ax.invert_yaxis()
        ax.grid(alpha=0.25)

        # per-axis legend
        handles = []
        for run in order:
            if run in sub["run"].values:
                handles.append(plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=palette.get(run, "#444"), markersize=9,
                    markeredgecolor="black", label=run))
        ax.legend(handles=handles, fontsize=7, loc="best", frameon=True)

    # shared marker legend
    marker_handles = [
        plt.Line2D([0], [0], marker=m, color="black",
                   markerfacecolor="lightgrey", markersize=10,
                   linestyle="", label=t)
        for t, m in [("step_0400", "o"), ("step_0450", "s"), ("best", "*"),
                     ("last", "D"), ("vanilla", "P"), ("temp_sweep", "X")]
    ]
    fig.legend(handles=marker_handles, loc="lower center",
               ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Library novelty (unique mutations / seq) vs interface quality", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile}")


def main():
    df = pd.read_csv(CSV)
    df = annotate_runs(df)
    # ltk_T0.2_vanilla had only 22 successful samples (vs 128 elsewhere) — drop it
    n_drop = (df["campaign"] == "ltk_T0.2_vanilla").sum()
    if n_drop:
        print(f"Dropping ltk_T0.2_vanilla ({n_drop} rows) — N={n_drop} too low vs 128 elsewhere")
        df = df[df["campaign"] != "ltk_T0.2_vanilla"].reset_index(drop=True)
    agg = aggregate(df)
    agg.to_csv(OUT / "checkpoint_eval_run_summary.csv", index=False)
    print(f"Aggregated {len(agg)} (run × tag) groups -> checkpoint_eval_run_summary.csv")

    print("Pareto plots:")
    make_pareto(agg, "gdf8", PALETTE,    GDF8_ORDER, OUT / "pareto_gdf8.png")
    make_pareto(agg, "ltk",  PALETTE_LTK, LTK_ORDER,  OUT / "pareto_ltk.png")

    print("Unique-mutations-per-seq vs iPAE:")
    make_unique_vs_ipae(agg, OUT / "unique_per_seq_vs_ipae.png")

    print("Subsampled unique-mutations vs iPAE (matched N per campaign):")
    sub_agg = subsample_unique_sites(df)
    sub_agg.to_csv(OUT / "checkpoint_eval_subsampled_summary.csv", index=False)
    make_subsampled_plot(sub_agg, OUT / "unique_subsampled_vs_ipae.png")

    print("Quality vs diversity headlines:")
    make_quality_diversity(agg, "gdf8", PALETTE,    OUT / "quality_diversity_gdf8.png")
    make_quality_diversity(agg, "ltk",  PALETTE_LTK, OUT / "quality_diversity_ltk.png")

    print("\nSummary table (best ckpt + vanilla):")
    rep = agg[(agg["kind"] == "vanilla") | (agg["tag"] == "best") | (agg["kind"] == "temp_sweep")]
    cols = ["campaign_base", "run", "kind", "n", "total_muts", "dist_to_ref",
            "iptm", "ptm", "ipae_min", "ipae_best",
            "unique_sites_vs_cons", "marginal_vs_cons", "pairwise_hamming"]
    print(rep[cols].sort_values(["campaign_base", "run"]).to_string(index=False))


if __name__ == "__main__":
    main()
