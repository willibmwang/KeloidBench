#!/usr/bin/env python3
"""TabPFN MVP training with leave-one-condition-out CV on Dataset A."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/mvp.yaml"
DEFAULT_DATA = PROJECT_ROOT / "data/processed/dataset_a_keloid_spheroid.parquet"


def load_tabpfn(tabpfn_src: str):
    if not os.environ.get("TABPFN_TOKEN"):
        print("TABPFN_TOKEN is not set; using LogisticRegression baseline.")
        return LogisticRegression, "logistic_regression"
    src = Path(tabpfn_src)
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from tabpfn import TabPFNClassifier  # type: ignore

        return TabPFNClassifier, "tabpfn"
    except Exception as exc:  # noqa: BLE001
        print(f"TabPFN unavailable ({exc}); using LogisticRegression baseline.")
        return LogisticRegression, "logistic_regression"


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def prepare_drug_response(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.dropna(subset=["relative_volume_pct"]).copy()
    out = out[out["drug"] != "vehicle"]
    out["drug_response_class"] = np.where(
        out["relative_volume_pct"] <= threshold, "sensitive", "resistant"
    )
    return out


def prepare_spheroid_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["dominant_spheroid_state"]).copy()
    out["spheroid_state"] = out["dominant_spheroid_state"]
    return out


def prepare_fibrotic_state(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["fibrotic_state"]).copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "paper",
        "source_sheet",
        "condition_code",
        "drug_response_class",
        "spheroid_state",
        "fibrotic_state",
        "dominant_spheroid_state",
        "dominant_state_pct",
        "culture_format",
        "notes",
    }
    cat = ["source_dataset", "cell_source", "fb_ec_ratio", "drug", "drug_concentration_um", "tgfb1"]
    categorical_present = [c for c in cat if c in df.columns]
    numeric = []
    for col in df.columns:
        if col in exclude or col in categorical_present:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().sum() == 0:
                continue
            numeric.append(col)
    return categorical_present + numeric


def encode_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            if pd.isna(fill_value):
                fill_value = 0.0
            parts.append(df[col].fillna(fill_value).to_numpy().reshape(-1, 1))
        else:
            enc = LabelEncoder()
            parts.append(enc.fit_transform(df[col].astype(str)).reshape(-1, 1))
    return np.hstack(parts).astype(np.float64)


def looco_eval(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    clf_factory,
    model_name: str,
) -> tuple[dict, pd.DataFrame]:
    df = df.dropna(subset=[target_col]).copy()
    if df.empty or df[target_col].nunique() < 2:
        return {"model": model_name, "task": target_col, "error": "insufficient data"}, pd.DataFrame()

    cols = feature_columns(df)
    if not cols:
        return {"model": model_name, "task": target_col, "error": "no usable features"}, pd.DataFrame()
    X = encode_features(df, cols)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target_col].astype(str))
    groups = df[group_col].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    preds = np.zeros_like(y)
    n_classes = len(np.unique(y))
    probs = np.full((len(y), n_classes), np.nan)

    valid_test_indices = []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        if clf_factory is DummyClassifier:
            clf = clf_factory(strategy="most_frequent")
        elif clf_factory is LogisticRegression:
            clf = make_pipeline(StandardScaler(), clf_factory(max_iter=5000))
        else:
            clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
        if hasattr(clf, "predict_proba") and n_classes == 2:
            fold_probs = clf.predict_proba(X[test_idx])
            if fold_probs.shape[1] == 2:
                probs[test_idx] = fold_probs
        valid_test_indices.extend(test_idx.tolist())

    if not valid_test_indices:
        return {
            "model": model_name,
            "task": target_col,
            "error": "no valid leave-one-group-out folds",
        }, pd.DataFrame()

    valid_test_indices = np.array(sorted(valid_test_indices))
    y_eval = y[valid_test_indices]
    preds_eval = preds[valid_test_indices]

    metrics = {
        "model": model_name,
        "task": target_col,
        "n_samples": int(len(valid_test_indices)),
        "n_groups": int(len(np.unique(groups))),
        "features": cols,
        "accuracy": float(accuracy_score(y_eval, preds_eval)),
        "macro_f1": float(f1_score(y_eval, preds_eval, average="macro")),
    }
    if len(np.unique(y_eval)) == 2 and probs.shape[1] == 2:
        prob_eval = probs[valid_test_indices, 1]
        if not np.isnan(prob_eval).any():
            metrics["roc_auc"] = float(roc_auc_score(y_eval, prob_eval))

    predictions = df.iloc[valid_test_indices][
        [c for c in ["paper", "source_dataset", "condition_code", "cell_source", "fb_ec_ratio", "drug"] if c in df.columns]
    ].copy()
    predictions["task"] = target_col
    predictions["model"] = model_name
    predictions["y_true"] = target_encoder.inverse_transform(y_eval)
    predictions["y_pred"] = target_encoder.inverse_transform(preds_eval)
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    threshold = cfg.get("drug_response", {}).get("sensitive_threshold_pct", 70)

    Clf, model_name = load_tabpfn(cfg["tabpfn_src"])

    tasks = []
    predictions = []
    drug_df = prepare_drug_response(df, threshold)
    if not drug_df.empty:
        metrics, preds = looco_eval(
            drug_df,
            "drug_response_class",
            "condition_code",
            Clf,
            model_name,
        )
        tasks.append(metrics)
        predictions.append(preds)

    state_df = prepare_spheroid_state(df)
    if not state_df.empty and state_df["spheroid_state"].nunique() >= 2:
        metrics, preds = looco_eval(
            state_df,
            "spheroid_state",
            "condition_code",
            Clf,
            model_name,
        )
        tasks.append(metrics)
        predictions.append(preds)

    fibrotic_df = prepare_fibrotic_state(df)
    if not fibrotic_df.empty and fibrotic_df["fibrotic_state"].nunique() >= 2:
        metrics, preds = looco_eval(
            fibrotic_df,
            "fibrotic_state",
            "condition_code",
            Clf,
            model_name,
        )
        tasks.append(metrics)
        predictions.append(preds)

    if model_name != "logistic_regression":
        for task_df, target_col in [
            (drug_df, "drug_response_class"),
            (state_df, "spheroid_state"),
            (fibrotic_df, "fibrotic_state"),
        ]:
            if not task_df.empty and task_df[target_col].nunique() >= 2:
                metrics, preds = looco_eval(
                    task_df,
                    target_col,
                    "condition_code",
                    LogisticRegression,
                    "logistic_regression",
                )
                tasks.append(metrics)
                predictions.append(preds)

    out_path = args.out_dir / "mvp_metrics.json"
    with out_path.open("w") as f:
        json.dump(tasks, f, indent=2)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            args.out_dir / "predictions_looco.csv", index=False
        )
    feature_rows = []
    for metric in tasks:
        for feature in metric.get("features", []):
            feature_rows.append(
                {
                    "model": metric.get("model"),
                    "task": metric.get("task"),
                    "feature": feature,
                    "proxy": "feature_used_in_model",
                }
            )
    if feature_rows:
        pd.DataFrame(feature_rows).to_csv(
            args.out_dir / "feature_importance_proxy.csv", index=False
        )
    print(json.dumps(tasks, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
