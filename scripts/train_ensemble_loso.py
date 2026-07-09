#!/usr/bin/env python3
"""Calibrated soft-vote ensemble for keloid_binary LOSO with threshold tuning."""

from __future__ import annotations

import argparse
import fnmatch
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from domain_adaptation import accession_map, apply_normalization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
RESULTS_DIR = PROJECT_ROOT / "results/ensemble_loso"

DEFAULT_FEATURE_SETS = [
    "modules_rank_only",
    "modules_rank_plus_expanded",
    "profibrotic_module_only",
    "expanded_profibrotic_only",
    "published_markers_only",
    "shared_genes",
]


def load_inputs(training_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict]]:
    manifest = pd.read_parquet(training_dir / "profile_manifest.parquet")
    features = {
        path.stem: pd.read_parquet(path).set_index("sample_id")
        for path in sorted((training_dir / "features").glob("*.parquet"))
    }
    splits = json.loads((training_dir / "splits/splits.json").read_text())
    return manifest, features, splits


def split_rows(feature_table: pd.DataFrame, target: pd.Series, sample_ids: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    ids = [sample_id for sample_id in sample_ids if sample_id in feature_table.index and sample_id in target.index]
    return feature_table.loc[ids], target.loc[ids]


def base_models(random_state: int) -> dict[str, object]:
    return {
        "elastic_net": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(class_weight="balanced", random_state=random_state, max_iter=8000)),
            ]
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=random_state,
        ),
        "small_mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=0.01,
                        learning_rate_init=0.001,
                        max_iter=800,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def predict_proba_binary(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if probs.shape[1] == 2:
            return probs[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(x)
    return preds.astype(float)


def tune_threshold(y_true: np.ndarray, probs: np.ndarray, labels: list[str]) -> float:
    if len(labels) != 2:
        return 0.5
    keloid_idx = labels.index("keloid") if "keloid" in labels else 1
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (probs >= thr).astype(int)
        if keloid_idx == 0:
            pred = 1 - pred
        f1 = f1_score(y_true, pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def fit_calibrated(model, x_train: pd.DataFrame, y_train: np.ndarray, random_state: int):
    if len(np.unique(y_train)) < 2 or len(x_train) < 8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)
        return model
    cv = min(3, len(np.unique(y_train)), len(x_train) // 4)
    if cv < 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)
        return model
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        calibrated.fit(x_train, y_train)
    return calibrated


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, labels: list[str], threshold: float) -> dict:
    keloid_idx = labels.index("keloid") if "keloid" in labels else 1
    pred = (probs >= threshold).astype(int)
    if keloid_idx == 0:
        pred = 1 - pred
    auc = None
    if len(labels) == 2 and len(np.unique(y_true)) == 2:
        score = probs if keloid_idx == 1 else 1.0 - probs
        auc = float(roc_auc_score(y_true, score))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auroc": auc,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=np.arange(len(labels))).tolist(),
        "labels": labels,
    }


def run_split(
    split: dict,
    feature_tables: dict[str, pd.DataFrame],
    target: pd.Series,
    manifest: pd.DataFrame,
    *,
    normalization_mode: str,
    random_state: int,
) -> list[dict]:
    acc_map = accession_map(manifest)
    rows = []
    ensemble_test_probs = []
    val_ensemble_probs = []
    y_test = None
    labels = None

    for feature_name, feature_table in feature_tables.items():
        x_train, y_train_raw = split_rows(feature_table, target, split["train_sample_ids"])
        x_val, y_val_raw = split_rows(feature_table, target, split.get("val_sample_ids", []))
        x_test, y_test_raw = split_rows(feature_table, target, split["test_sample_ids"])
        if len(x_train) < 4 or len(x_test) < 2:
            continue
        if y_train_raw.nunique() < 2 or y_test_raw.nunique() < 2:
            continue

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
        local_labels = label_encoder.classes_.tolist()
        y_train = label_encoder.transform(y_train_raw.astype(str))
        y_val = label_encoder.transform(y_val_raw.astype(str)) if len(y_val_raw) else np.array([])
        y_test = label_encoder.transform(y_test_raw.astype(str))
        labels = local_labels

        val_probs = []
        test_probs = []
        for model_name, model in base_models(random_state).items():
            if model_name == "small_mlp" and feature_name == "shared_genes":
                continue
            fitted = fit_calibrated(model, x_train, y_train, random_state)
            if len(y_val):
                val_probs.append(predict_proba_binary(fitted, x_val))
            test_probs.append(predict_proba_binary(fitted, x_test))

        if not test_probs:
            continue

        val_mean = np.mean(val_probs, axis=0) if val_probs else None
        test_mean = np.mean(test_probs, axis=0)
        threshold = tune_threshold(y_val, val_mean, labels) if val_mean is not None and len(y_val) else 0.5
        metrics = evaluate_probs(y_test, test_mean, labels, threshold)
        rows.append(
            {
                "task": split["task"],
                "split_name": split["split_name"],
                "feature_set": feature_name,
                "model": "ensemble_member",
                "normalization_mode": normalization_mode,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                **metrics,
            }
        )
        ensemble_test_probs.append(test_mean)
        if val_mean is not None:
            val_ensemble_probs.append(val_mean)

    if ensemble_test_probs and y_test is not None and labels is not None:
        stacked_test = np.mean(np.vstack(ensemble_test_probs), axis=0)
        threshold = 0.5
        if val_ensemble_probs:
            # Reuse last y_val from final feature iteration when available.
            val_stack = np.mean(np.vstack(val_ensemble_probs), axis=0)
            if len(y_val):
                threshold = tune_threshold(y_val, val_stack, labels)
        metrics = evaluate_probs(y_test, stacked_test, labels, threshold)
        rows.append(
            {
                "task": split["task"],
                "split_name": split["split_name"],
                "feature_set": "stacked_ensemble",
                "model": "calibrated_soft_vote",
                "normalization_mode": normalization_mode,
                "n_train": int(len(split["train_sample_ids"])),
                "n_test": int(len(split["test_sample_ids"])),
                **metrics,
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[results.get("error", pd.Series(dtype=object)).isna()] if "error" in results.columns else results
    return (
        valid.groupby(["feature_set", "model", "normalization_mode"], dropna=False)[
            ["weighted_f1", "balanced_accuracy", "accuracy", "auroc"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--normalization-mode",
        choices=["accession_zscore", "quantile_rank"],
        default="quantile_rank",
    )
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--split-pattern", default="leave_accession_out_*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest, features, splits = load_inputs(args.training_dir)
    splits = [
        split
        for split in splits
        if split["task"] == "keloid_binary" and fnmatch.fnmatch(split["split_name"], args.split_pattern)
    ]
    feature_tables = {name: features[name] for name in args.feature_sets if name in features}
    if not splits or not feature_tables:
        raise SystemExit("No LOSO splits or requested feature sets found.")

    target = manifest.set_index("sample_id")["keloid_binary"]
    rows = []
    for split in splits:
        rows.extend(
            run_split(
                split,
                feature_tables,
                target,
                manifest,
                normalization_mode=args.normalization_mode,
                random_state=args.random_state,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "ensemble_loso_results.csv", index=False)
    results.to_json(args.out_dir / "ensemble_loso_results.jsonl", orient="records", lines=True)
    summary = summarize(results)
    summary.to_csv(args.out_dir / "ensemble_loso_summary.csv", index=False)
    best = summary.sort_values("weighted_f1_mean", ascending=False).head(10)
    report = {
        "n_splits": len(splits),
        "feature_sets": list(feature_tables),
        "normalization_mode": args.normalization_mode,
        "best_rows": best.to_dict(orient="records"),
    }
    (args.out_dir / "ensemble_loso_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
