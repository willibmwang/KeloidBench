#!/usr/bin/env python3
"""Write the SpheroScar MVP evaluation report and next-stage decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
BASELINE_DIR = PROJECT_ROOT / "results/baselines"
OUT_DIR = PROJECT_ROOT / "results/mvp"


def load_data(training_dir: Path, baseline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(training_dir / "profile_manifest.parquet")
    results = pd.read_json(baseline_dir / "baseline_results.jsonl", lines=True)
    summary = pd.read_csv(baseline_dir / "baseline_summary.csv")
    return manifest, results, summary


def best_rows(summary: pd.DataFrame, task: str, n: int = 10) -> pd.DataFrame:
    rows = summary[summary["task"].eq(task)].copy()
    if rows.empty:
        return rows
    sort_cols = [col for col in ["weighted_f1_mean", "balanced_accuracy_mean", "accuracy_mean"] if col in rows.columns]
    return rows.sort_values(sort_cols, ascending=False).head(n)


def qwen_adapter_decision(summary: pd.DataFrame) -> dict:
    task_rows = best_rows(summary, "keloid_binary", n=20)
    if task_rows.empty:
        return {
            "decision": "do_not_train_yet",
            "reason": "No valid keloid_binary baseline results were produced.",
        }
    non_dummy = task_rows[~task_rows["model"].isin(["majority", "stratified_random"])].copy()
    dummy = task_rows[task_rows["model"].isin(["majority", "stratified_random"])].copy()
    best_non_dummy = non_dummy.iloc[0].to_dict() if not non_dummy.empty else None
    best_dummy = dummy.iloc[0].to_dict() if not dummy.empty else None
    if best_non_dummy is None:
        return {
            "decision": "do_not_train_yet",
            "reason": "Only dummy baselines produced valid keloid_binary results.",
            "best_dummy": best_dummy,
        }
    dummy_f1 = float(best_dummy.get("weighted_f1_mean", 0.0)) if best_dummy else 0.0
    model_f1 = float(best_non_dummy.get("weighted_f1_mean", 0.0))
    balanced = float(best_non_dummy.get("balanced_accuracy_mean", 0.0))
    margin = model_f1 - dummy_f1
    if margin >= 0.05 and balanced > 0.55:
        decision = "ready_for_frozen_qwen_adapter"
        reason = "Best non-dummy baseline beats dummy baselines with a useful weighted-F1 margin and balanced accuracy above chance."
    else:
        decision = "hold_qwen_adapter"
        reason = "Basic baselines have not established a strong enough leakage-safe signal for adapter training."
    return {
        "decision": decision,
        "reason": reason,
        "best_non_dummy": best_non_dummy,
        "best_dummy": best_dummy,
        "weighted_f1_margin": margin,
    }


def split_lookup(training_dir: Path) -> dict[tuple[str, str], dict]:
    splits = json.loads((training_dir / "splits/splits.json").read_text())
    return {(split["task"], split["split_name"]): split for split in splits}


def write_detailed_breakdowns(
    out_dir: Path,
    manifest: pd.DataFrame,
    results: pd.DataFrame,
    splits: dict[tuple[str, str], dict],
) -> list[dict]:
    manifest_by_id = manifest.set_index("sample_id")
    valid = results[results["error"].isna()] if "error" in results.columns else results.copy()
    rows: list[dict] = []
    for _, row in valid.iterrows():
        split = splits.get((row["task"], row["split_name"]), {})
        test_ids = [
            sample_id
            for sample_id in split.get("test_sample_ids", [])
            if sample_id in manifest_by_id.index
        ]
        test_manifest = manifest_by_id.loc[test_ids] if test_ids else manifest_by_id.iloc[0:0]
        detail = {
            "task": row["task"],
            "split_name": row["split_name"],
            "feature_set": row["feature_set"],
            "model": row["model"],
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
            "accuracy": float(row["accuracy"]),
            "weighted_f1": float(row["weighted_f1"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "auroc": None if pd.isna(row.get("auroc")) else float(row["auroc"]),
            "labels": row["labels"],
            "confusion_matrix": row["confusion_matrix"],
            "test_label_counts": row["test_label_counts"],
            "test_modality_counts": test_manifest["processed_modality"].value_counts().to_dict(),
            "test_accession_counts": test_manifest["accession"].value_counts().to_dict(),
        }
        rows.append(detail)

    (out_dir / "detailed_eval_breakdowns.json").write_text(json.dumps(rows, indent=2))
    flat_rows = []
    for row in rows:
        flat = row.copy()
        for col in [
            "labels",
            "confusion_matrix",
            "test_label_counts",
            "test_modality_counts",
            "test_accession_counts",
        ]:
            flat[col] = json.dumps(flat[col], sort_keys=True)
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(out_dir / "detailed_eval_breakdowns.csv", index=False)
    return rows


def write_markdown(
    path: Path,
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
    decision: dict,
    detailed_rows: list[dict],
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "No valid results."
        headers = df.columns.tolist()
        rows = []
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            values = []
            for value in row.tolist():
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    lines = [
        "# SpheroScar MVP Baseline Evaluation",
        "",
        "## Corpus",
        "",
        f"- Total profiles: {len(manifest)}",
        f"- Modalities: {manifest['processed_modality'].value_counts().to_dict()}",
        f"- Accessions: {manifest['accession'].value_counts().to_dict()}",
        f"- Canonical keloid labels: {manifest['keloid_binary'].value_counts().to_dict()}",
        "",
        "## Best Baselines",
        "",
    ]
    for task in ["keloid_binary", "lesional_status", "cell_type", "fibroblast_subcluster", "celltype_subcluster"]:
        rows = best_rows(summary, task, n=5)
        lines.append(f"### {task}")
        if rows.empty:
            lines.append("")
            lines.append("No valid results.")
            lines.append("")
            continue
        keep_cols = [
            "feature_set",
            "model",
            "accuracy_mean",
            "weighted_f1_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "weighted_f1_count",
        ]
        keep_cols = [col for col in keep_cols if col in rows.columns]
        lines.append("")
        lines.append(markdown_table(rows[keep_cols]))
        lines.append("")

    lines.extend(
        [
            "## Frozen-Qwen Adapter Decision",
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Reason: {decision['reason']}",
            "",
            "The frozen-Qwen adapter should only be trained after simple baselines show stable grouped-split signal.",
            "",
        ]
    )
    lines.extend(
        [
            "## Detailed Artifacts",
            "",
            "- `detailed_eval_breakdowns.json` contains split-level metrics, confusion matrices, and modality/accession composition for every valid baseline run.",
            "- `detailed_eval_breakdowns.csv` is the same information flattened for quick review.",
            f"- Detailed baseline rows: {len(detailed_rows)}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_adapter_config(path: Path, decision: dict) -> None:
    config = {
        "stage": "frozen_qwen_adapter_mvp",
        "decision": decision["decision"],
        "enabled": decision["decision"] == "ready_for_frozen_qwen_adapter",
        "decoder": "Qwen/Qwen3-1.7B",
        "trainable_components": ["expression_prefix_adapter", "optional_classification_heads"],
        "frozen_components": ["qwen_decoder"],
        "input_feature_sets": ["shared_genes_plus_modules", "modules_only"],
        "primary_task": "keloid_binary",
        "notes": decision["reason"],
    }
    path.write_text(json.dumps(config, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest, results, summary = load_data(args.training_dir, args.baseline_dir)
    detailed_rows = write_detailed_breakdowns(
        args.out_dir,
        manifest,
        results,
        split_lookup(args.training_dir),
    )
    decision = qwen_adapter_decision(summary)
    report = {
        "n_profiles": int(len(manifest)),
        "modalities": manifest["processed_modality"].value_counts().to_dict(),
        "keloid_binary": manifest["keloid_binary"].value_counts().to_dict(),
        "n_result_rows": int(len(results)),
        "n_detailed_eval_rows": int(len(detailed_rows)),
        "qwen_adapter_decision": decision,
    }
    (args.out_dir / "mvp_eval_report.json").write_text(json.dumps(report, indent=2))
    write_markdown(args.out_dir / "mvp_eval_report.md", manifest, summary, decision, detailed_rows)
    write_adapter_config(args.out_dir / "frozen_qwen_adapter_config.json", decision)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
