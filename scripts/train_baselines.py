#!/usr/bin/env python3
"""Train leakage-aware baseline models for the SpheroScar MVP."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from domain_adaptation import accession_map, apply_normalization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
RESULTS_DIR = PROJECT_ROOT / "results/baselines"

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


def load_inputs(training_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict]]:
    manifest = pd.read_parquet(training_dir / "profile_manifest.parquet")
    features = {
        path.stem: pd.read_parquet(path).set_index("sample_id")
        for path in (training_dir / "features").glob("*.parquet")
    }
    splits = json.loads((training_dir / "splits/splits.json").read_text())
    return manifest, features, splits


def target_for_task(manifest: pd.DataFrame, task: str) -> pd.Series:
    if task == "keloid_binary":
        return manifest.set_index("sample_id")["keloid_binary"]
    return manifest.set_index("sample_id")["task_target"]


def model_specs(random_state: int) -> dict[str, object]:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "stratified_random": DummyClassifier(strategy="stratified", random_state=random_state),
        "elastic_net_logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(class_weight="balanced", random_state=random_state, max_iter=5000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=random_state,
        ),
        "knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=3)),
            ]
        ),
        "small_mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=0.01,
                        learning_rate_init=0.001,
                        max_iter=500,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def score_binary_auc(model, x_test: pd.DataFrame, y_test: np.ndarray) -> float | None:
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)
            if scores.shape[1] == 2:
                return float(roc_auc_score(y_test, scores[:, 1]))
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x_test)
            return float(roc_auc_score(y_test, scores))
    except Exception:
        return None
    return None


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], auc: float | None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(labels))).tolist(),
        "labels": labels,
    }
    return metrics


def split_rows(feature_table: pd.DataFrame, target: pd.Series, sample_ids: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    ids = [sample_id for sample_id in sample_ids if sample_id in feature_table.index and sample_id in target.index]
    return feature_table.loc[ids], target.loc[ids]


def run_one(
    feature_name: str,
    feature_table: pd.DataFrame,
    split: dict,
    target: pd.Series,
    random_state: int,
    *,
    manifest: pd.DataFrame | None = None,
    normalization_mode: str = "none",
) -> list[dict]:
    x_train, y_train_raw = split_rows(feature_table, target, split["train_sample_ids"])
    x_val, y_val_raw = split_rows(feature_table, target, split.get("val_sample_ids", []))
    x_test, y_test_raw = split_rows(feature_table, target, split["test_sample_ids"])
    if len(x_train) < 4 or len(x_test) < 2:
        return []
    if y_train_raw.nunique() < 2 or y_test_raw.nunique() < 2:
        return []

    if normalization_mode != "none" and manifest is not None:
        acc_map = accession_map(manifest)
        x_train, x_val, x_test = apply_normalization(
            x_train,
            x_val,
            x_test,
            x_train.index.tolist(),
            x_val.index.tolist() if len(x_val) else [],
            x_test.index.tolist(),
            acc_map,
            normalization_mode,
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train_raw, y_test_raw]).astype(str))
    y_train = label_encoder.transform(y_train_raw.astype(str))
    y_test = label_encoder.transform(y_test_raw.astype(str))
    labels = label_encoder.classes_.tolist()

    rows = []
    for model_name, model in model_specs(random_state).items():
        if model_name == "small_mlp" and feature_name == "shared_genes":
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            auc = score_binary_auc(model, x_test, y_test) if len(labels) == 2 else None
            metrics = evaluate_predictions(y_test, y_pred, labels, auc)
            rows.append(
                {
                    "task": split["task"],
                    "split_name": split["split_name"],
                    "feature_set": feature_name,
                    "normalization_mode": normalization_mode,
                    "model": model_name,
                    "n_train": int(len(x_train)),
                    "n_test": int(len(x_test)),
                    "train_label_counts": y_train_raw.value_counts().to_dict(),
                    "test_label_counts": y_test_raw.value_counts().to_dict(),
                    **metrics,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "task": split["task"],
                    "split_name": split["split_name"],
                    "feature_set": feature_name,
                    "normalization_mode": normalization_mode,
                    "model": model_name,
                    "n_train": int(len(x_train)),
                    "n_test": int(len(x_test)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[results["error"].isna()] if "error" in results.columns else results.copy()
    metric_cols = ["accuracy", "weighted_f1", "balanced_accuracy", "macro_f1", "auroc"]
    grouped = (
        valid.groupby(["task", "feature_set", "model"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = ["_".join([part for part in col if part]) for col in grouped.columns.to_flat_index()]
    return grouped.sort_values(["task", "weighted_f1_mean", "balanced_accuracy_mean"], ascending=[True, False, False])


def write_reports(results: list[dict], manifest: pd.DataFrame, out_dir: Path, wandb_run=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in results:
        converted = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                converted[key] = value.item()
            else:
                converted[key] = value
        serializable.append(converted)

    results_df = pd.DataFrame(serializable)
    results_df.to_json(out_dir / "baseline_results.jsonl", orient="records", lines=True)
    results_df.to_csv(out_dir / "baseline_results.csv", index=False)
    summary_df = summarize(results_df)
    summary_df.to_csv(out_dir / "baseline_summary.csv", index=False)

    best = {}
    if not summary_df.empty:
        for task, group in summary_df.groupby("task"):
            best[task] = group.head(10).to_dict(orient="records")
    report = {
        "n_profiles": int(len(manifest)),
        "tasks": manifest["encoder_task"].value_counts().to_dict(),
        "keloid_binary": manifest["keloid_binary"].value_counts().to_dict(),
        "n_result_rows": int(len(results_df)),
        "n_errors": int(results_df["error"].notna().sum()) if "error" in results_df.columns else 0,
        "best_by_task": best,
    }
    (out_dir / "baseline_report.json").write_text(json.dumps(report, indent=2))
    if wandb_run is not None:
        wandb_run.summary["n_profiles"] = report["n_profiles"]
        wandb_run.summary["n_result_rows"] = report["n_result_rows"]
        wandb_run.summary["n_errors"] = report["n_errors"]
        for task, rows in best.items():
            if not rows:
                continue
            best_row = rows[0]
            prefix = f"best/{task}"
            wandb_run.summary[f"{prefix}/feature_set"] = best_row.get("feature_set")
            wandb_run.summary[f"{prefix}/model"] = best_row.get("model")
            for metric in ["accuracy_mean", "weighted_f1_mean", "balanced_accuracy_mean", "macro_f1_mean"]:
                if metric in best_row:
                    wandb_run.summary[f"{prefix}/{metric}"] = best_row[metric]
        if wandb is not None:
            wandb_run.log(
                {
                    "baseline_results": wandb.Table(dataframe=results_df),
                    "baseline_summary": wandb.Table(dataframe=summary_df),
                }
            )
            for artifact_path in [
                out_dir / "baseline_results.csv",
                out_dir / "baseline_summary.csv",
                out_dir / "baseline_report.json",
            ]:
                wandb_run.save(str(artifact_path))
    print(json.dumps(report, indent=2))


def init_wandb(args: argparse.Namespace, manifest: pd.DataFrame, splits: list[dict]):
    if not args.wandb or wandb is None:
        return None
    try:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            job_type="baseline_training",
            mode=args.wandb_mode,
            config={
                "training_dir": str(args.training_dir),
                "out_dir": str(args.out_dir),
                "n_profiles": int(len(manifest)),
                "n_splits": len(splits),
                "tasks": manifest["encoder_task"].value_counts().to_dict(),
                "keloid_binary": manifest["keloid_binary"].value_counts().to_dict(),
                "feature_sets": ["modules_only", "shared_genes", "shared_genes_plus_modules"],
            },
        )
        return run
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: W&B logging disabled ({type(exc).__name__}: {exc})")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="SpheroScar")
    parser.add_argument("--wandb-entity", default="williamwang178243-yale-university")
    parser.add_argument("--wandb-run-name", default="baseline-mvp-267-profiles")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--normalization-mode",
        choices=["none", "accession_zscore", "quantile_rank"],
        default="none",
        help="Transductive per-accession normalization applied before model fitting.",
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--feature-sets", nargs="+", default=None)
    parser.add_argument(
        "--split-pattern",
        default=None,
        help="Glob pattern for split_name (e.g. leave_accession_out_*)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, features, splits = load_inputs(args.training_dir)
    if args.tasks:
        task_set = set(args.tasks)
        splits = [split for split in splits if split["task"] in task_set]
    if args.split_pattern:
        import fnmatch

        splits = [split for split in splits if fnmatch.fnmatch(split["split_name"], args.split_pattern)]
    if args.feature_sets:
        feature_set = set(args.feature_sets)
        features = {name: table for name, table in features.items() if name in feature_set}
    if not splits:
        raise SystemExit("No splits remain after filtering.")
    if not features:
        raise SystemExit("No feature sets remain after filtering.")
    wandb_run = init_wandb(args, manifest, splits)
    results = []
    for split in splits:
        target = target_for_task(manifest, split["task"])
        for feature_name, feature_table in features.items():
            results.extend(
                run_one(
                    feature_name,
                    feature_table,
                    split,
                    target,
                    args.random_state,
                    manifest=manifest,
                    normalization_mode=args.normalization_mode,
                )
            )
    write_reports(results, manifest, args.out_dir, wandb_run=wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
