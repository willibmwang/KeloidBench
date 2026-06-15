#!/usr/bin/env python3
"""Upload completed SpheroScar baseline artifacts to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "results/baselines"
MVP_DIR = PROJECT_ROOT / "results/mvp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--mvp-dir", type=Path, default=MVP_DIR)
    parser.add_argument("--project", default="SpheroScar")
    parser.add_argument("--entity", default="williamwang178243-yale-university")
    parser.add_argument("--run-name", default="baseline-mvp-267-profiles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    summary = pd.read_csv(args.baseline_dir / "baseline_summary.csv")
    results = pd.read_csv(args.baseline_dir / "baseline_results.csv")
    report = json.loads((args.mvp_dir / "mvp_eval_report.json").read_text())

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        job_type="baseline_upload",
        config={
            "n_profiles": report["n_profiles"],
            "modalities": report["modalities"],
            "keloid_binary": report["keloid_binary"],
            "n_result_rows": report["n_result_rows"],
        },
    )
    run.summary["qwen_adapter_decision"] = report["qwen_adapter_decision"]["decision"]
    run.summary["qwen_adapter_reason"] = report["qwen_adapter_decision"]["reason"]
    best = report["qwen_adapter_decision"].get("best_non_dummy") or {}
    for metric in ["accuracy_mean", "weighted_f1_mean", "balanced_accuracy_mean", "macro_f1_mean", "auroc_mean"]:
        if metric in best:
            run.summary[f"best_keloid_binary/{metric}"] = best[metric]
    run.summary["best_keloid_binary/model"] = best.get("model")
    run.summary["best_keloid_binary/feature_set"] = best.get("feature_set")

    run.log(
        {
            "baseline_summary": wandb.Table(dataframe=summary),
            "baseline_results": wandb.Table(dataframe=results),
        }
    )
    for path in [
        args.baseline_dir / "baseline_results.csv",
        args.baseline_dir / "baseline_summary.csv",
        args.baseline_dir / "baseline_report.json",
        args.mvp_dir / "mvp_eval_report.json",
        args.mvp_dir / "mvp_eval_report.md",
        args.mvp_dir / "detailed_eval_breakdowns.json",
        args.mvp_dir / "detailed_eval_breakdowns.csv",
        args.mvp_dir / "frozen_qwen_adapter_config.json",
    ]:
        if path.exists():
            run.save(str(path))
    run.finish()
    print(f"Uploaded baseline artifacts to {args.entity}/{args.project}")


if __name__ == "__main__":
    main()
