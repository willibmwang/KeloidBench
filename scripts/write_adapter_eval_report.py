#!/usr/bin/env python3
"""Aggregate frozen-Qwen adapter search results and compare to baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
BASELINE_DIR = PROJECT_ROOT / "results/baselines"
RESULTS_DIR = PROJECT_ROOT / "results/frozen_qwen_adapter"

BASELINE_WEIGHTED_F1 = 0.654
BASELINE_WEIGHTED_F1_STD = 0.179
LOSO_TOLERANCE = 0.05


def split_family(split_name: str) -> str:
    if split_name.startswith("grouped_"):
        return "grouped"
    if split_name.startswith("leave_accession_out_"):
        return "leave_accession_out"
    return "other"


def load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing adapter results: {results_path}")
    rows = pd.read_json(results_path, lines=True)
    if "error" in rows.columns:
        rows = rows[rows["error"].isna()]
    rows = rows[rows["search_phase"] == "phase_c"].copy()
    if rows.empty:
        rows = pd.read_json(results_path, lines=True)
        if "error" in rows.columns:
            rows = rows[rows["error"].isna()]
    return rows


def flatten_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        test_metrics = row.get("test_metrics") or {}
        val_metrics = row.get("val_metrics") or {}
        records.append(
            {
                "config_id": row["config_id"],
                "search_phase": row.get("search_phase"),
                "encoder_type": row.get("encoder_type"),
                "encoder_layers": row.get("encoder_layers"),
                "num_prefix_tokens": row.get("num_prefix_tokens"),
                "max_epochs": row.get("max_epochs"),
                "dropout": row.get("dropout"),
                "lr": row.get("lr"),
                "split_name": row["split_name"],
                "split_family": split_family(row["split_name"]),
                "best_epoch": row.get("best_epoch"),
                "n_train": row.get("n_train"),
                "n_test": row.get("n_test"),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_weighted_f1": test_metrics.get("weighted_f1"),
                "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "test_macro_f1": test_metrics.get("macro_f1"),
                "test_auroc": test_metrics.get("auroc"),
                "val_weighted_f1": val_metrics.get("weighted_f1"),
            }
        )
    return pd.DataFrame(records)


def attach_test_composition(flat: pd.DataFrame, manifest: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    manifest = manifest.set_index("sample_id")
    split_map = {split["split_name"]: split for split in splits}
    modality_counts = []
    accession_counts = []
    for _, row in flat.iterrows():
        split = split_map.get(row["split_name"], {})
        test_ids = [sid for sid in split.get("test_sample_ids", []) if sid in manifest.index]
        test_manifest = manifest.loc[test_ids] if test_ids else manifest.iloc[0:0]
        modality_counts.append(test_manifest["processed_modality"].value_counts().to_dict())
        accession_counts.append(test_manifest["accession"].value_counts().to_dict())
    flat = flat.copy()
    flat["test_modality_counts"] = modality_counts
    flat["test_accession_counts"] = accession_counts
    return flat


def aggregate_by_family(flat: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_accuracy",
        "test_weighted_f1",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_auroc",
        "best_epoch",
    ]
    rows = []
    for (config_id, family), group in flat.groupby(["config_id", "split_family"]):
        row = {"config_id": config_id, "split_family": family, "n_splits": len(group)}
        for col in metric_cols:
            values = group[col].dropna().tolist()
            row[f"{col}_mean"] = float(np.mean(values)) if values else None
            row[f"{col}_std"] = float(np.std(values)) if values else None
        rows.append(row)
    return pd.DataFrame(rows)


def adapter_decision(summary: pd.DataFrame) -> dict:
    if summary.empty:
        return {
            "decision": "hold_qwen_decode",
            "reason": "No phase_c adapter results were available for evaluation.",
        }
    best_config = (
        summary[summary["split_family"] == "grouped"]
        .sort_values("test_weighted_f1_mean", ascending=False)
        .head(1)
    )
    if best_config.empty:
        return {
            "decision": "hold_qwen_decode",
            "reason": "No grouped-split adapter metrics were available.",
        }
    best = best_config.iloc[0]
    grouped_f1 = float(best["test_weighted_f1_mean"])
    grouped_std = float(best.get("test_weighted_f1_std") or 0.0)
    loso = summary[
        (summary["config_id"] == best["config_id"])
        & (summary["split_family"] == "leave_accession_out")
    ]
    loso_f1 = float(loso.iloc[0]["test_weighted_f1_mean"]) if not loso.empty else None

    beats_baseline = grouped_f1 > BASELINE_WEIGHTED_F1
    stable = grouped_std <= BASELINE_WEIGHTED_F1_STD
    loso_ok = loso_f1 is None or grouped_f1 - loso_f1 <= LOSO_TOLERANCE

    if beats_baseline and stable and loso_ok:
        decision = "proceed_to_prompt_conditioned_qwen"
        reason = (
            "Adapter beats the elastic-net baseline on grouped folds with comparable stability "
            "and acceptable leave-one-accession-out performance."
        )
    elif beats_baseline:
        decision = "hold_adapter_refine"
        reason = (
            "Adapter beats the baseline mean on grouped folds but remains unstable or weak "
            "under leave-one-accession-out evaluation."
        )
    else:
        decision = "hold_qwen_decode"
        reason = "Adapter did not beat the elastic-net baseline mean weighted F1 on grouped folds."

    return {
        "decision": decision,
        "reason": reason,
        "best_config_id": best["config_id"],
        "grouped_weighted_f1_mean": grouped_f1,
        "grouped_weighted_f1_std": grouped_std,
        "loso_weighted_f1_mean": loso_f1,
        "baseline_weighted_f1_mean": BASELINE_WEIGHTED_F1,
        "baseline_weighted_f1_std": BASELINE_WEIGHTED_F1_STD,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results."
    headers = df.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, flat: pd.DataFrame, summary: pd.DataFrame, decision: dict) -> None:
    best_rows = (
        flat[flat["config_id"] == decision.get("best_config_id")]
        .sort_values(["split_family", "split_name"])
        if decision.get("best_config_id")
        else flat.head(0)
    )
    lines = [
        "# Frozen-Qwen Adapter Evaluation",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Reason: {decision['reason']}",
        f"- Best config: `{decision.get('best_config_id', 'n/a')}`",
        f"- Grouped weighted F1 mean: {decision.get('grouped_weighted_f1_mean')}",
        f"- LOSO weighted F1 mean: {decision.get('loso_weighted_f1_mean')}",
        f"- Baseline target: {BASELINE_WEIGHTED_F1}",
        "",
        "## Config Summary",
        "",
        markdown_table(
            summary.sort_values(["config_id", "split_family"])[
                [
                    "config_id",
                    "split_family",
                    "n_splits",
                    "test_weighted_f1_mean",
                    "test_weighted_f1_std",
                    "test_balanced_accuracy_mean",
                    "best_epoch_mean",
                ]
            ]
        ),
        "",
        "## Best Config Per-Split Results",
        "",
        markdown_table(
            best_rows[
                [
                    "split_name",
                    "split_family",
                    "n_train",
                    "n_test",
                    "best_epoch",
                    "test_weighted_f1",
                    "test_balanced_accuracy",
                    "test_auroc",
                ]
            ]
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--results-jsonl", type=Path, default=RESULTS_DIR / "adapter_results.jsonl")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.training_dir / "profile_manifest.parquet")
    splits = json.loads((args.training_dir / "splits/splits.json").read_text())

    raw = load_results(args.results_jsonl)
    flat = flatten_metrics(raw)
    flat = attach_test_composition(flat, manifest, splits)
    summary = aggregate_by_family(flat)
    decision = adapter_decision(summary)

    flat.to_csv(args.out_dir / "adapter_eval_per_split.csv", index=False)
    summary.to_csv(args.out_dir / "adapter_eval_summary.csv", index=False)
    report = {
        "n_result_rows": int(len(raw)),
        "n_flat_rows": int(len(flat)),
        "decision": decision,
        "summary": summary.to_dict(orient="records"),
    }
    (args.out_dir / "adapter_eval_report.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(args.out_dir / "adapter_eval_report.md", flat, summary, decision)

    if decision.get("best_config_id"):
        best_config = raw[raw["config_id"] == decision["best_config_id"]].iloc[0].to_dict()
        keep = {
            k: best_config[k]
            for k in [
                "config_id",
                "encoder_type",
                "encoder_layers",
                "encoder_hidden_dim",
                "transformer_encoder_dim",
                "num_prefix_tokens",
                "max_epochs",
                "patience",
                "dropout",
                "lr",
                "batch_size",
                "feature_set",
                "model_name",
            ]
            if k in best_config
        }
        keep["decision"] = decision["decision"]
        (args.out_dir / "best_adapter_config.json").write_text(json.dumps(keep, indent=2))

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
