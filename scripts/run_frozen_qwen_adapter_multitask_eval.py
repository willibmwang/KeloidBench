#!/usr/bin/env python3
"""Evaluate one frozen-Qwen adapter config across multiple SpheroScar tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_frozen_qwen_adapter import RESULTS_DIR, TRAINING_DIR, TrainConfig, run_training

DEFAULT_TASKS = [
    "keloid_binary",
    "cell_type",
    "fibroblast_subcluster",
    "celltype_subcluster",
]

FOUNDATION_CONFIG = {
    "model_name": "Qwen/Qwen3-1.7B",
    "feature_set": "shared_genes_plus_modules",
    "encoder_type": "mlp",
    "encoder_layers": 2,
    "encoder_hidden_dim": 1024,
    "transformer_encoder_dim": 1024,
    "transformer_encoder_heads": 4,
    "transformer_gene_chunk_size": 256,
    "num_prefix_tokens": 4,
    "max_epochs": 15,
    "patience": 3,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "config_id": "foundation_mlp_L2_E15_P4",
    "search_phase": "multitask_eval",
}


def make_args(base: argparse.Namespace, task: str) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(base))
    ns.task = task
    ns.split_name = None
    ns.split_names = None
    ns.split_pattern = None
    ns.all_task_splits = False
    ns.all_keloid_binary_splits = False
    ns.wandb_run_name = f"{base.wandb_run_name}-{task}"
    return ns


def summarize_results(results_jsonl: Path, out_dir: Path) -> dict:
    rows = pd.read_json(results_jsonl, lines=True)
    if "error" in rows.columns:
        rows = rows[rows["error"].isna()]
    if rows.empty:
        summary = {"tasks": {}, "splits": []}
        (out_dir / "foundation_multitask_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    rows["test_accuracy"] = rows["test_metrics"].map(lambda m: m.get("accuracy"))
    rows["test_weighted_f1"] = rows["test_metrics"].map(lambda m: m.get("weighted_f1"))

    task_summary = (
        rows.groupby("task")
        .agg(
            n_splits=("split_name", "count"),
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            weighted_f1_mean=("test_weighted_f1", "mean"),
            weighted_f1_std=("test_weighted_f1", "std"),
        )
        .reset_index()
        .sort_values("task")
    )

    summary = {
        "config_id": rows["config_id"].iloc[0],
        "n_result_rows": int(len(rows)),
        "tasks": task_summary.to_dict(orient="records"),
        "splits": rows[
            ["task", "split_name", "n_train", "n_test", "best_epoch", "test_accuracy", "test_weighted_f1"]
        ].to_dict(orient="records"),
    }
    (out_dir / "foundation_multitask_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# Frozen-Qwen Foundation Multi-Task Evaluation", ""]
    lines.append(f"- Config: `{summary['config_id']}`")
    lines.append(f"- Result rows: {summary['n_result_rows']}")
    lines.append("")
    lines.append("| task | n_splits | accuracy_mean | weighted_f1_mean |")
    lines.append("| --- | --- | --- | --- |")
    for row in summary["tasks"]:
        lines.append(
            f"| {row['task']} | {row['n_splits']} | "
            f"{row['accuracy_mean']:.4f} | {row['weighted_f1_mean']:.4f} |"
        )
    (out_dir / "foundation_multitask_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--results-jsonl", type=Path, default=RESULTS_DIR / "foundation_multitask_results.jsonl")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--reset-results", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="SpheroScar")
    parser.add_argument("--wandb-entity", default="williamwang178243-yale-university")
    parser.add_argument("--wandb-group", default="foundation-multitask-eval")
    parser.add_argument("--wandb-run-name", default="foundation-multitask-eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_results and args.results_jsonl.exists():
        args.results_jsonl.unlink()

    train_config = TrainConfig(**{k: FOUNDATION_CONFIG[k] for k in TrainConfig.__dataclass_fields__ if k in FOUNDATION_CONFIG})

    for task in args.tasks:
        task_args = make_args(args, task)
        task_args.results_jsonl = args.results_jsonl
        task_args.save_checkpoint = False
        run_training(task_args, train_config)

    summary = summarize_results(args.results_jsonl, args.out_dir)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
