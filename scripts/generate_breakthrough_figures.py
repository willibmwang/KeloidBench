#!/usr/bin/env python3
"""Generate breakthrough-sprint publication figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results/breakthrough_sprint"
FIG = OUT / "figures"

sns.set_theme(style="whitegrid", context="talk")


def fig_endpoint_eligibility(training_dir: Path, fig_dir: Path) -> None:
    path = training_dir / "endpoint_eligibility_matrix.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    eps = [
        "keloid_vs_unaffected_skin",
        "keloid_vs_normal_scar",
        "keloid_vs_pathologic_scar",
    ]
    heat = []
    for _, row in df.iterrows():
        for ep in eps:
            n_pos = row.get(f"{ep}_keloid", 0)
            n_neg = row.get(f"{ep}_non_keloid", 0)
            val = np.nan
            if n_pos > 0 and n_neg > 0:
                val = n_pos / (n_pos + n_neg)
            heat.append({"accession": row["accession"], "endpoint": ep.replace("keloid_vs_", ""), "keloid_frac": val})
    plot = pd.DataFrame(heat).pivot(index="accession", columns="endpoint", values="keloid_frac")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot))))
    sns.heatmap(plot, ax=ax, cmap="RdYlBu_r", vmin=0, vmax=1, linewidths=0.3)
    ax.set_title("Endpoint eligibility (keloid fraction; blank = not evaluable)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_endpoint_eligibility.png", dpi=200)
    plt.close(fig)


def fig_ablation_waterfall(out_dir: Path, fig_dir: Path) -> None:
    path = out_dir / "breakthrough_ablations.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    order = ["endpoint_cleanup", "fibrosis_view", "scar_discriminative_view", "composition_view", "fused_experts"]
    df = df[df.ablation_stage.isin(order)].copy()
    df["ablation_stage"] = pd.Categorical(df["ablation_stage"], categories=order, ordered=True)
    df = df.sort_values("ablation_stage")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["ablation_stage"].astype(str), df["macro_f1_mean"], marker="o", linewidth=2)
    ax.fill_between(
        range(len(df)),
        df["macro_f1_lo"].fillna(df["macro_f1_mean"]),
        df["macro_f1_hi"].fillna(df["macro_f1_mean"]),
        alpha=0.2,
    )
    ax.axhline(0.85, color="black", linestyle="--", label="Gate 0.85")
    ax.set_ylabel("Macro-accession F1")
    ax.set_title("Ablation waterfall (primary skin endpoint)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_ablation_waterfall.png", dpi=200)
    plt.close(fig)


def fig_accuracy_coverage(out_dir: Path, fig_dir: Path) -> None:
    path = out_dir / "breakthrough_report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    cascade = report.get("cascade") or {}
    full = (cascade.get("full_coverage_macro_f1") or {}).get("mean")
    sel = (cascade.get("selective_macro_f1") or {}).get("mean")
    cov = cascade.get("mean_selective_coverage")
    if full is None and sel is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [1.0, cov if cov is not None else 0.6]
    ys = [full if full is not None else np.nan, sel if sel is not None else np.nan]
    ax.scatter(xs, ys, s=120)
    for x, y, lab in zip(xs, ys, ["Full coverage", "Selective"]):
        if y is not None and not np.isnan(y):
            ax.annotate(f"{lab}\nF1={y:.3f}", (x, y), textcoords="offset points", xytext=(8, 8))
    ax.axhline(0.85, color="gray", linestyle="--")
    ax.axhline(0.90, color="black", linestyle="--")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Macro-accession F1")
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs coverage (selective cascade)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_accuracy_coverage.png", dpi=200)
    plt.close(fig)


def fig_cascade_routing(out_dir: Path, fig_dir: Path) -> None:
    path = out_dir / "breakthrough_cascade_predictions.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "decision" not in df.columns:
        return
    counts = df.groupby(["accession", "decision"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Profiles")
    ax.set_title("Cascade routing / abstention by accession")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_cascade_routing.png", dpi=200)
    plt.close(fig)


def fig_lockbox(out_dir: Path, fig_dir: Path) -> None:
    path = out_dir / "breakthrough_lockbox_once.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = [r for r in data.get("results", []) if r.get("status") == "ok"]
    if not rows:
        return
    labels = [r["accession"] for r in rows]
    full = [r.get("weighted_f1", np.nan) for r in rows]
    sel = [r.get("selective_f1", np.nan) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.15, full, width=0.3, label="Full coverage")
    ax.bar(x + 0.15, sel, width=0.3, label="Selective")
    ax.axhline(0.85, color="gray", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_title("Breakthrough lockbox (one-shot)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_lockbox_once.png", dpi=200)
    plt.close(fig)


def fig_nested_primary(out_dir: Path, fig_dir: Path) -> None:
    path = out_dir / "breakthrough_nested_selected.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    skin = df[df.endpoint.eq("keloid_vs_unaffected_skin")]
    if skin.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(skin["accession"], skin["weighted_f1"], color="#2a9d8f")
    ax.axhline(0.85, color="black", linestyle="--", label="Gate 0.85")
    ax.axhline(0.75, color="gray", linestyle=":", label="Worst-fold gate 0.75")
    ax.set_ylabel("Weighted F1")
    ax.set_title("Primary nested LOSO: keloid vs unaffected skin")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_primary_nested_loso.png", dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "data/processed/training")
    p.add_argument("--fig-dir", type=Path, default=FIG)
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_endpoint_eligibility(args.training_dir, args.fig_dir)
    fig_ablation_waterfall(args.out_dir, args.fig_dir)
    fig_accuracy_coverage(args.out_dir, args.fig_dir)
    fig_cascade_routing(args.out_dir, args.fig_dir)
    fig_lockbox(args.out_dir, args.fig_dir)
    fig_nested_primary(args.out_dir, args.fig_dir)
    manifest = {"figures": sorted(p.name for p in args.fig_dir.glob("fig_*.png")), "dir": str(args.fig_dir)}
    (args.fig_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
