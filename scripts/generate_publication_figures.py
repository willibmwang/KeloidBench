#!/usr/bin/env python3
"""Generate publication figures for SpheroScar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUB_DIR = PROJECT_ROOT / "results/publication"
FIG_DIR = PUB_DIR / "figures"

sns.set_theme(style="whitegrid", context="talk")

ABLATION_FEATURE_LABELS = {
    "profibrotic_module_only": "POSTN/profibrotic\n(1 module)",
    "modules_only": "7 modules",
    "modules_rank_only": "7 modules\n(rank)",
    "modules_rank_plus_expanded": "Rank modules\n+ expanded POSTN",
    "expanded_profibrotic_only": "Expanded\nPOSTN program",
    "published_markers_only": "Published\nmarkers",
    "low_i2_core_only": "44-gene\nlow-I² core",
    "shared_genes": "5006 genes",
    "shared_genes_plus_modules": "Genes +\nmodules",
    "stacked_ensemble": "Calibrated\nensemble",
}


def fig5_ml_ablation(pub_dir: Path, out_dir: Path) -> None:
    path = pub_dir / "keloid_binary_loso_ablation.csv"
    if not path.exists() or path.stat().st_size == 0:
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    order = [k for k in ABLATION_FEATURE_LABELS if k in set(df["feature_set"])]
    if not order:
        return
    plot_rows = []
    for feature_set in order:
        for _, row in df[df["feature_set"].eq(feature_set)].iterrows():
            plot_rows.append(
                {
                    "feature_set": feature_set,
                    "split_family": row["reporting_label"],
                    "weighted_f1": row["weighted_f1_mean"],
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {"loso": "#e76f51", "grouped": "#457b9d"}
    x_positions = np.arange(len(order))
    width = 0.35
    for i, family in enumerate(["loso", "grouped"]):
        sub = plot_df[plot_df["split_family"].eq(family)].set_index("feature_set").reindex(order)
        offset = -width / 2 if family == "loso" else width / 2
        bars = ax.bar(
            x_positions + offset,
            sub["weighted_f1"],
            width=width,
            label="LOSO (primary)" if family == "loso" else "Grouped CV (secondary)",
            color=palette[family],
            alpha=0.9,
        )
        for bar, (_, row) in zip(bars, sub.iterrows()):
            if pd.notna(row["weighted_f1"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{row['weighted_f1']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1, label="Target F1=0.8")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([ABLATION_FEATURE_LABELS[k] for k in order])
    ax.set_ylabel("Weighted F1 (keloid_binary)")
    ax.set_ylim(0, max(0.9, plot_df["weighted_f1"].max() + 0.1))
    ax.legend(loc="upper right")
    ax.set_title("Figure 5. ML prediction ablation: program vs marker vs gene (RQ2)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_ml_ablation.png", dpi=200)
    plt.close(fig)


def fig6_push08_comparison(pub_dir: Path, out_dir: Path) -> None:
    rows = []
    ablation = pub_dir / "keloid_binary_loso_ablation.csv"
    if ablation.exists():
        df = pd.read_csv(ablation)
        loso = df[df["reporting_label"].eq("loso")]
        for _, row in loso.iterrows():
            rows.append({"track": row["feature_set"], "weighted_f1": row["weighted_f1_mean"], "kind": "baseline"})
    push = pub_dir / "push08_ensemble_loso.json"
    if push.exists():
        report = json.loads(push.read_text())
        rows.append({"track": "stacked_ensemble", "weighted_f1": report["weighted_f1_mean"], "kind": "ensemble"})
    if not rows:
        return
    plot_df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2a9d8f" if k == "ensemble" else "#457b9d" for k in plot_df["kind"]]
    ax.barh(plot_df["track"], plot_df["weighted_f1"], color=colors)
    ax.axvline(0.8, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("LOSO weighted F1")
    ax.set_title("Figure 6. Push-0.8 comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_push08_comparison.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub-dir", type=Path, default=PUB_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig5_ml_ablation(args.pub_dir, args.fig_dir)
    fig6_push08_comparison(args.pub_dir, args.fig_dir)
    manifest = {
        "figures": [str(p.name) for p in sorted(args.fig_dir.glob("fig*.png"))],
        "output_dir": str(args.fig_dir),
    }
    (args.fig_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
