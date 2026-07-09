#!/usr/bin/env python3
"""Add bootstrap CIs, LOSO-primary reporting, and push-0.8 ablation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from publication_utils import bootstrap_metric_rows, split_family

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "results/baselines"
ENSEMBLE_DIR = PROJECT_ROOT / "results/ensemble_loso"
OUT_DIR = PROJECT_ROOT / "results/publication"

ABLATION_FEATURE_SETS = [
    "profibrotic_module_only",
    "modules_only",
    "modules_rank_only",
    "modules_rank_plus_expanded",
    "expanded_profibrotic_only",
    "published_markers_only",
    "low_i2_core_only",
    "shared_genes",
    "shared_genes_plus_modules",
    "stacked_ensemble",
]


def enrich_baseline_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = results[results["error"].isna()].copy() if "error" in results.columns else results.copy()
    valid["split_family"] = valid["split_name"].map(split_family)
    summary = (
        valid.groupby(["task", "feature_set", "model", "split_family", "normalization_mode"], dropna=False)[
            ["weighted_f1", "balanced_accuracy", "accuracy"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join([p for p in col if p]).strip("_") for col in summary.columns.to_flat_index()]
    ci_rows = bootstrap_metric_rows(
        valid,
        ["task", "feature_set", "model", "split_family", "normalization_mode"],
        "weighted_f1",
        n_bootstrap=500,
        random_state=13,
    )
    return summary, ci_rows


def loso_primary_table(summary: pd.DataFrame, task: str = "keloid_binary") -> pd.DataFrame:
    rows = summary[summary["task"].eq(task)].copy()
    if rows.empty:
        return rows
    sort_key = "weighted_f1_mean"
    loso = rows[rows["split_family"].eq("leave_accession_out")].sort_values(sort_key, ascending=False)
    grouped = rows[rows["split_family"].eq("grouped")].sort_values(sort_key, ascending=False)
    primary = pd.concat([loso.head(5), grouped.head(5)], ignore_index=True)
    primary["reporting_priority"] = ["primary_loso"] * len(loso.head(5)) + ["secondary_grouped"] * len(grouped.head(5))
    return primary


def loso_ablation_table(summary: pd.DataFrame, task: str = "keloid_binary") -> pd.DataFrame:
    rows = summary[summary["task"].eq(task) & summary["feature_set"].isin(ABLATION_FEATURE_SETS)].copy()
    if rows.empty:
        return rows
    sort_key = "weighted_f1_mean"
    out = []
    for feature_set in ABLATION_FEATURE_SETS:
        sub = rows[rows["feature_set"].eq(feature_set)]
        for family, label in [("leave_accession_out", "loso"), ("grouped", "grouped")]:
            fam = sub[sub["split_family"].eq(family)].sort_values(sort_key, ascending=False)
            if fam.empty:
                continue
            best = fam.iloc[0]
            out.append(
                {
                    "task": task,
                    "feature_set": feature_set,
                    "split_family": family,
                    "reporting_label": label,
                    "model": best["model"],
                    "normalization_mode": best.get("normalization_mode"),
                    "weighted_f1_mean": best.get("weighted_f1_mean"),
                    "weighted_f1_std": best.get("weighted_f1_std"),
                    "balanced_accuracy_mean": best.get("balanced_accuracy_mean"),
                    "n_splits": best.get("weighted_f1_count"),
                }
            )
    return pd.DataFrame(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--ensemble-dir", type=Path, default=ENSEMBLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = args.baseline_dir / "baseline_results.jsonl"
    if not baseline_path.exists():
        baseline_path = args.baseline_dir / "baseline_results.csv"
    if baseline_path.suffix == ".jsonl":
        baseline_results = pd.read_json(baseline_path, lines=True)
    else:
        baseline_results = pd.read_csv(baseline_path)

    summary, baseline_ci = enrich_baseline_results(baseline_results)
    primary = loso_primary_table(summary)
    ablation = loso_ablation_table(summary)

    ensemble_summary = pd.DataFrame()
    ensemble_path = args.ensemble_dir / "ensemble_loso_results.csv"
    if ensemble_path.exists():
        ensemble_results = pd.read_csv(ensemble_path)
        ensemble_results["split_family"] = ensemble_results["split_name"].map(split_family)
        ensemble_summary = (
            ensemble_results.groupby(["feature_set", "model", "normalization_mode"], dropna=False)[
                ["weighted_f1", "balanced_accuracy", "accuracy"]
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        ensemble_summary.columns = ["_".join([p for p in col if p]).strip("_") for col in ensemble_summary.columns.to_flat_index()]
        ensemble_summary.to_csv(args.out_dir / "ensemble_loso_summary_by_split_family.csv", index=False)
        stacked = ensemble_results[ensemble_results["feature_set"].eq("stacked_ensemble")]
        if not stacked.empty:
            stacked_summary = {
                "weighted_f1_mean": float(stacked["weighted_f1"].mean()),
                "weighted_f1_std": float(stacked["weighted_f1"].std()),
                "balanced_accuracy_mean": float(stacked["balanced_accuracy"].mean()),
                "n_splits": int(len(stacked)),
            }
            (args.out_dir / "push08_ensemble_loso.json").write_text(json.dumps(stacked_summary, indent=2))

    summary.to_csv(args.out_dir / "baseline_summary_by_split_family.csv", index=False)
    baseline_ci.to_csv(args.out_dir / "baseline_weighted_f1_bootstrap_ci.csv", index=False)
    primary.to_csv(args.out_dir / "keloid_binary_loso_primary.csv", index=False)
    ablation.to_csv(args.out_dir / "keloid_binary_loso_ablation.csv", index=False)

    report = {
        "primary_metric": "leave_accession_out_weighted_f1",
        "keloid_binary_loso_primary": primary.to_dict(orient="records"),
        "keloid_binary_loso_ablation": ablation.to_dict(orient="records"),
        "ensemble_summary_rows": int(len(ensemble_summary)),
    }
    (args.out_dir / "rigor_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
