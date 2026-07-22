#!/usr/bin/env python3
"""Nested leave-one-accession-out training for the public-cohort F1 ladder.

Outer: leave-one-accession-out.
Inner: equal-accession mean F1 on accession-blocked OOF folds for
model/feature/threshold selection (lower-tail tie-breaker). Never selects from
outer-test results.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
RESULTS_DIR = PROJECT_ROOT / "results/public_expansion_loso"
REGISTRY_PATH = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"

POSITIVE_LABEL = "keloid"
BINARY_CLASSES = ("keloid", "non_keloid")
RANK_FEATURES = {
    "modules_rank_only",
    "modules_rank_plus_expanded",
    "expanded_profibrotic_only",
    "robust_programs_only",
    "rank_programs_only",
    "rank_programs_plus_compartment",
    "donor_balanced_rank_programs",
    "scar_discriminative_rank_programs",
}
PREREGISTERED = [
    "profibrotic_module_only",
    "rank_programs_only",
    "rank_programs_plus_compartment",
    "donor_balanced_rank_programs",
    "scar_discriminative_rank_programs",
]
ENDPOINT_TARGET = {
    "keloid_binary": "keloid_binary",
    "keloid_binary_all_profiles": "keloid_binary_all_profiles",
    "clean_keloid_binary": "clean_keloid_binary",
    "fibroblast_keloid_binary": "fibroblast_keloid_binary",
    "keloid_vs_normal_scar": "keloid_vs_normal_scar",
    "keloid_vs_pathologic_scar": "keloid_vs_pathologic_scar",
    "keloid_vs_unaffected_skin": "keloid_vs_unaffected_skin",
}
BASELINE_REPORT = PROJECT_ROOT / "results/accuracy_ladder_baseline/public_expansion_loso_report.json"
BASELINE_NESTED = PROJECT_ROOT / "results/accuracy_ladder_baseline/public_expansion_nested_selected.csv"


def load_inputs(training_dir: Path):
    manifest = pd.read_parquet(training_dir / "profile_manifest.parquet")
    features = {
        path.stem: pd.read_parquet(path).set_index("sample_id")
        for path in sorted((training_dir / "features").glob("*.parquet"))
    }
    splits = json.loads((training_dir / "splits/splits.json").read_text())
    return manifest, features, splits


def split_rows(feature_table: pd.DataFrame, target: pd.Series, sample_ids: list[str]):
    ids = [sid for sid in sample_ids if sid in feature_table.index and sid in target.index]
    return feature_table.loc[ids], target.loc[ids]


def encode_binary(y_train_raw: pd.Series, y_test_raw: pd.Series):
    """Fixed class order; never fit an encoder on outer-test labels."""
    classes = list(BINARY_CLASSES)
    mapping = {label: idx for idx, label in enumerate(classes)}
    y_train = y_train_raw.astype(str).map(mapping).to_numpy()
    y_test = y_test_raw.astype(str).map(mapping).to_numpy()
    if np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError("Binary labels must be keloid/non_keloid")
    return classes, y_train.astype(int), y_test.astype(int), classes.index(POSITIVE_LABEL)


def model_specs(random_state: int) -> dict[str, object]:
    return {
        "elastic_net_logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        class_weight=None,
                        max_iter=4000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "ridge_logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        class_weight=None,
                        max_iter=4000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(class_weight=None, random_state=random_state, max_iter=8000)),
            ]
        ),
    }


def positive_proba(model, x: pd.DataFrame, positive_code: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        # Pipeline/LogisticRegression classes_ follow fit order of y codes.
        classes = list(getattr(model, "classes_", getattr(model.named_steps["clf"], "classes_", [0, 1])))
        if positive_code in classes:
            return probs[:, list(classes).index(positive_code)]
        return probs[:, 0] if positive_code == 0 else probs[:, -1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return p1 if positive_code == 1 else 1.0 - p1
    preds = model.predict(x)
    return (preds == positive_code).astype(float)


def tune_threshold(y_true: np.ndarray, probs: np.ndarray, positive_code: int, metric: str = "weighted_f1") -> float:
    best_thr, best = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 33):
        pred = np.where(probs >= thr, positive_code, 1 - positive_code)
        if metric == "macro_f1":
            score = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            score = f1_score(y_true, pred, average="weighted", zero_division=0)
        if score > best:
            best, best_thr = score, thr
    return float(best_thr)


def metrics_from_probs(y_true, probs, labels, threshold, positive_code) -> dict:
    pred = np.where(probs >= threshold, positive_code, 1 - positive_code)
    auc = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score((y_true == positive_code).astype(int), probs))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auroc": auc,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=np.arange(len(labels))).tolist(),
        "labels": labels,
    }


def donor_accession_weights(sample_ids: list[str], manifest: pd.DataFrame) -> np.ndarray:
    meta = manifest.set_index("sample_id")
    if "accession_donor_balanced_weight" in meta.columns:
        w = meta.loc[sample_ids, "accession_donor_balanced_weight"].astype(float).to_numpy()
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        return w / w.mean()
    accessions = meta.loc[sample_ids, "accession"].astype(str)
    groups = meta.loc[sample_ids, "split_group"].astype(str)
    # Equal accession * equal donor within accession.
    acc_counts = accessions.value_counts()
    donor_counts = groups.value_counts()
    w = np.array([1.0 / (acc_counts[a] * donor_counts[g] / accessions.eq(a).sum()) for a, g in zip(accessions, groups)])
    return w / w.mean()


def fit_model(model, x, y, sample_weight=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if sample_weight is None:
            model.fit(x, y)
        else:
            try:
                model.fit(x, y, clf__sample_weight=sample_weight)
            except TypeError:
                model.fit(x, y)
    return model


def _train_only_accession_zscore(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score each accession using train-accession statistics only (no test leakage)."""
    from domain_adaptation import accession_map, apply_train_only_normalization

    return apply_train_only_normalization(
        x_train,
        x_test,
        train_ids,
        test_ids,
        accession_map(manifest),
        "train_accession_zscore",
    )


def _apply_train_only_norm(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    manifest: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from domain_adaptation import accession_map, apply_train_only_normalization

    return apply_train_only_normalization(
        x_train,
        x_test,
        train_ids,
        test_ids,
        accession_map(manifest),
        mode,
    )


def _fit_platt(scores: np.ndarray, y_true: np.ndarray, positive_code: int) -> tuple[float, float]:
    """Fit logistic calibration on inner OOF scores → P(positive)."""
    from sklearn.linear_model import LogisticRegression as _LR

    y = (np.asarray(y_true) == positive_code).astype(int)
    if len(np.unique(y)) < 2:
        return 0.0, 0.0
    clf = _LR(solver="lbfgs", max_iter=1000)
    clf.fit(np.asarray(scores).reshape(-1, 1), y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def _apply_platt(scores: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    if coef == 0.0 and intercept == 0.0:
        return np.asarray(scores, dtype=float)
    z = coef * np.asarray(scores, dtype=float) + intercept
    return 1.0 / (1.0 + np.exp(-z))


def evaluate_fixed(
    *,
    feature_table,
    target,
    train_ids,
    test_ids,
    manifest,
    model_name,
    feature_name,
    weight_mode,
    random_state,
    threshold: float | None = None,
    normalization_mode: str = "none",
    calibration: str = "none",
    platt_params: tuple[float, float] | None = None,
):
    x_train, y_train_raw = split_rows(feature_table, target, train_ids)
    x_test, y_test_raw = split_rows(feature_table, target, test_ids)
    if len(x_train) < 4 or len(x_test) < 2 or y_train_raw.nunique() < 2 or y_test_raw.nunique() < 2:
        return None
    if normalization_mode not in {None, "none", ""}:
        x_train, x_test = _apply_train_only_norm(
            x_train, x_test, train_ids, test_ids, manifest, normalization_mode
        )
    labels, y_train, y_test, positive_code = encode_binary(y_train_raw, y_test_raw)
    if weight_mode in {"donor", "accession_donor"} or feature_name == "donor_balanced_rank_programs":
        weights = donor_accession_weights(x_train.index.tolist(), manifest)
    else:
        weights = None
    model = model_specs(random_state)[model_name]
    fitted = fit_model(model, x_train, y_train, sample_weight=weights)
    test_probs = positive_proba(fitted, x_test, positive_code)
    if calibration == "platt_inner_oof" and platt_params is not None:
        test_probs = _apply_platt(test_probs, platt_params[0], platt_params[1])
    thr = 0.5 if threshold is None else float(threshold)
    metrics = metrics_from_probs(y_test, test_probs, labels, thr, positive_code)
    # Donor-grain metrics: majority vote / mean prob per split_group in test.
    meta = manifest.set_index("sample_id")
    groups = meta.loc[x_test.index, "split_group"].astype(str)
    donor_true = []
    donor_prob = []
    for group, idx in groups.groupby(groups).groups.items():
        y_g = y_test[[x_test.index.get_loc(i) for i in idx]]
        p_g = test_probs[[x_test.index.get_loc(i) for i in idx]]
        # Donor label: majority; donor prob: mean.
        donor_true.append(int(np.round(y_g.mean())))
        donor_prob.append(float(np.mean(p_g)))
    donor_metrics = None
    if len(set(donor_true)) >= 1 and len(donor_true) >= 1:
        # Report donor metrics whenever ≥2 donors exist; binary AUC needs 2 classes.
        if len(set(donor_true)) == 2 and len(donor_true) >= 2:
            donor_metrics = metrics_from_probs(
                np.asarray(donor_true),
                np.asarray(donor_prob),
                labels,
                thr,
                positive_code,
            )
        else:
            # Single-class donor fold: still compute weighted F1 vs constant labels.
            donor_metrics = metrics_from_probs(
                np.asarray(donor_true),
                np.asarray(donor_prob),
                labels,
                thr,
                positive_code,
            )
    return {
        "feature_set": feature_name,
        "model": model_name,
        "weight_mode": weight_mode,
        "adaptation": "none",
        "normalization_mode": normalization_mode,
        "calibration": calibration,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_test_donors": int(groups.nunique()),
        **metrics,
        "donor_weighted_f1": None if donor_metrics is None else donor_metrics["weighted_f1"],
        "donor_macro_f1": None if donor_metrics is None else donor_metrics["macro_f1"],
        "donor_auroc": None if donor_metrics is None else donor_metrics.get("auroc"),
        "probs": test_probs,
        "y_true": y_test,
        "positive_code": positive_code,
        "label_names": labels,
        "donor_true": np.asarray(donor_true),
        "donor_prob": np.asarray(donor_prob),
    }


def _accession_f1(y_true: np.ndarray, probs: np.ndarray, threshold: float, metric: str = "weighted_f1") -> float:
    pred = np.where(probs >= threshold, 0, 1)
    if metric == "macro_f1":
        return float(f1_score(y_true, pred, average="macro", zero_division=0))
    return float(f1_score(y_true, pred, average="weighted", zero_division=0))


def tune_threshold_equal_accession(
    fold_truths: list[np.ndarray],
    fold_probs: list[np.ndarray],
    metric: str = "weighted_f1",
) -> tuple[float, float, float]:
    """Maximize equal-accession mean F1; lower-tail (10th pct) is the tie-breaker."""
    if not fold_truths:
        return 0.5, float("nan"), float("nan")
    grid = np.unique(
        np.concatenate(
            [
                np.quantile(np.concatenate(fold_probs), np.linspace(0.05, 0.95, 19)),
                np.array([0.5]),
            ]
        )
    )
    best_thr = 0.5
    best_key = (-1.0, -1.0)
    best_mean = float("nan")
    best_tail = float("nan")
    for thr in grid:
        scores = [_accession_f1(y, p, float(thr), metric=metric) for y, p in zip(fold_truths, fold_probs)]
        mean_score = float(np.mean(scores))
        lower_tail = float(np.quantile(scores, 0.10))
        key = (mean_score, lower_tail)
        if key > best_key:
            best_key = key
            best_thr = float(thr)
            best_mean = mean_score
            best_tail = lower_tail
    return best_thr, best_mean, best_tail


def inner_select_equal_accession(
    split,
    feature_tables,
    target,
    manifest,
    candidates,
    random_state,
    metric="weighted_f1",
    selection_grain: str = "profile",
):
    """Select config+threshold by equal-accession mean inner F1 (+ lower-tail tie-break).

    selection_grain:
      - profile: classic profile-level F1 (v1)
      - donor: donor-aggregated F1 within each inner accession (v2)
    """
    train_ids = split["train_sample_ids"] + split.get("val_sample_ids", [])
    meta = manifest.set_index("sample_id")
    train_ids = [i for i in train_ids if i in meta.index]
    train_acc = sorted(set(meta.loc[train_ids, "accession"].astype(str)))
    # Exclude lockbox from selection pool.
    if "lockbox" in meta.columns:
        train_acc = [a for a in train_acc if not bool(meta.loc[meta["accession"].eq(a), "lockbox"].any())]
    if len(train_acc) < 3:
        chosen = candidates[0]
        return {
            **chosen,
            "threshold": 0.5,
            "inner_score": None,
            "inner_lower_tail": None,
            "platt_params": None,
            "normalization_mode": chosen.get("normalization_mode", "none"),
            "calibration": chosen.get("calibration", "none"),
        }

    best = None
    for cand in candidates:
        fold_truths = []
        fold_probs = []
        oof_scores = []
        oof_labels = []
        norm = cand.get("normalization_mode", "none")
        calib = cand.get("calibration", "none")
        for held in train_acc:
            inner_test = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) == held]
            inner_train = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) != held]
            if "lockbox" in meta.columns:
                inner_train = [sid for sid in inner_train if not bool(meta.loc[sid, "lockbox"])]
            result = evaluate_fixed(
                feature_table=feature_tables[cand["feature_set"]],
                target=target,
                train_ids=inner_train,
                test_ids=inner_test,
                manifest=manifest,
                model_name=cand["model"],
                feature_name=cand["feature_set"],
                weight_mode=cand["weight_mode"],
                random_state=random_state,
                threshold=0.5,
                normalization_mode=norm,
                calibration="none",
            )
            if result is None:
                continue
            if selection_grain == "donor" and len(result.get("donor_true", [])) >= 1:
                fold_truths.append(result["donor_true"])
                fold_probs.append(result["donor_prob"])
            else:
                fold_truths.append(result["y_true"])
                fold_probs.append(result["probs"])
            oof_scores.append(result["probs"])
            oof_labels.append(result["y_true"])
        if not fold_truths:
            continue
        platt_params = None
        if calib == "platt_inner_oof" and oof_scores:
            scores_cat = np.concatenate(oof_scores)
            labels_cat = np.concatenate(oof_labels)
            platt_params = _fit_platt(scores_cat, labels_cat, positive_code=0)
            fold_probs = [_apply_platt(p, platt_params[0], platt_params[1]) for p in fold_probs]
        thr, mean_score, lower_tail = tune_threshold_equal_accession(fold_truths, fold_probs, metric=metric)
        key = (mean_score, lower_tail)
        if best is None or key > best[0]:
            best = (
                key,
                {
                    **cand,
                    "threshold": thr,
                    "inner_score": mean_score,
                    "inner_lower_tail": lower_tail,
                    "platt_params": platt_params,
                    "normalization_mode": norm,
                    "calibration": calib,
                    "selection_grain": selection_grain,
                },
            )
    return best[1] if best else {
        **candidates[0],
        "threshold": 0.5,
        "inner_score": None,
        "inner_lower_tail": None,
        "platt_params": None,
        "normalization_mode": candidates[0].get("normalization_mode", "none"),
        "calibration": candidates[0].get("calibration", "none"),
    }


# Backward-compatible alias used by older tests/imports.
inner_select_pooled = inner_select_equal_accession


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in [
            "weighted_f1",
            "balanced_accuracy",
            "accuracy",
            "macro_f1",
            "auroc",
            "donor_weighted_f1",
            "donor_auroc",
        ]
        if c in results.columns
    ]
    grouped = (
        results.groupby(["endpoint", "stage", "feature_set", "model", "weight_mode", "adaptation"], dropna=False)[
            metric_cols
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = ["_".join([part for part in col if part]) for col in grouped.columns.to_flat_index()]
    return grouped.sort_values(["endpoint", "weighted_f1_mean"], ascending=[True, False])


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 13) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return {
        "mean": float(np.mean(arr)),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--random-state", type=int, default=13)
    p.add_argument(
        "--endpoint",
        choices=[
            "keloid_binary",
            "keloid_binary_all_profiles",
            "clean_keloid_binary",
            "fibroblast_keloid_binary",
            "keloid_vs_normal_scar",
            "keloid_vs_pathologic_scar",
            "keloid_vs_unaffected_skin",
            "both",
            "all",
        ],
        default="all",
    )
    p.add_argument("--split-pattern", default="leave_accession_out_*")
    p.add_argument(
        "--eval-mode",
        choices=["current10", "expanded", "lockbox", "all_modes"],
        default="all_modes",
    )
    p.add_argument("--allow-group-dro", action="store_true")
    return p.parse_args()


def candidate_grid(feature_names: list[str]) -> list[dict]:
    models = ["elastic_net_logreg", "ridge_logreg", "linear_svm"]
    out = []
    for f in feature_names:
        weight = "accession_donor" if f == "donor_balanced_rank_programs" else "none"
        for m in models:
            out.append(
                {
                    "feature_set": f,
                    "model": m,
                    "weight_mode": weight,
                    "adaptation": "none",
                    "normalization_mode": "none",
                }
            )
    return out


def filter_splits(splits, endpoint, pattern, eval_mode, original_10, lockbox_accessions):
    out = []
    for s in splits:
        if s["task"] != endpoint:
            continue
        if not fnmatch.fnmatch(s["split_name"], pattern):
            continue
        acc = s["split_name"].replace("leave_accession_out_", "")
        if eval_mode == "current10" and acc not in original_10:
            continue
        if eval_mode == "lockbox" and acc not in lockbox_accessions:
            continue
        if eval_mode == "expanded" and acc in lockbox_accessions:
            continue
        out.append(s)
    return out


def run_endpoint(endpoint: str, args, manifest, features, splits, original_10, lockbox_accessions):
    target_col = ENDPOINT_TARGET[endpoint]
    if target_col not in manifest.columns:
        return [], []
    target = manifest.set_index("sample_id")[target_col]
    feature_names = [f for f in PREREGISTERED if f in features]
    if not feature_names:
        feature_names = [f for f in ["profibrotic_module_only", "modules_only", "robust_programs_only"] if f in features]
    candidates = candidate_grid(feature_names)
    modes = ["current10", "expanded", "lockbox"] if args.eval_mode == "all_modes" else [args.eval_mode]

    rows = []
    selected_rows = []
    for mode in modes:
        endpoint_splits = filter_splits(splits, endpoint, args.split_pattern, mode, original_10, lockbox_accessions)
        for split in endpoint_splits:
            # Fixed pre-registered configs at threshold 0.5
            for cand in candidates:
                result = evaluate_fixed(
                    feature_table=features[cand["feature_set"]],
                    target=target,
                    train_ids=split["train_sample_ids"],
                    test_ids=split["test_sample_ids"],
                    manifest=manifest,
                    model_name=cand["model"],
                    feature_name=cand["feature_set"],
                    weight_mode=cand["weight_mode"],
                    random_state=args.random_state,
                    threshold=0.5,
                )
                if result is None:
                    continue
                row = {
                    "endpoint": endpoint,
                    "eval_mode": mode,
                    "stage": "corrected_fixed",
                    "split_name": split["split_name"],
                    "selection": "fixed",
                    **{k: v for k, v in result.items() if k not in {"probs", "y_true", "positive_code", "label_names"}},
                }
                rows.append(row)

            # Nested selection — skip for lockbox outer folds (use frozen config later)
            if mode == "lockbox":
                continue
            chosen = inner_select_equal_accession(split, features, target, manifest, candidates, args.random_state)
            nested = evaluate_fixed(
                feature_table=features[chosen["feature_set"]],
                target=target,
                train_ids=split["train_sample_ids"],
                test_ids=split["test_sample_ids"],
                manifest=manifest,
                model_name=chosen["model"],
                feature_name=chosen["feature_set"],
                weight_mode=chosen["weight_mode"],
                random_state=args.random_state,
                threshold=chosen.get("threshold", 0.5),
            )
            if nested is None:
                continue
            row = {
                "endpoint": endpoint,
                "eval_mode": mode,
                "stage": "nested_selected",
                "split_name": split["split_name"],
                "selection": "inner_equal_accession",
                **{k: v for k, v in nested.items() if k not in {"probs", "y_true", "positive_code", "label_names"}},
            }
            rows.append(row)
            selected_rows.append(
                {
                    "endpoint": endpoint,
                    "eval_mode": mode,
                    "split_name": split["split_name"],
                    "inner_score": chosen.get("inner_score"),
                    "inner_lower_tail": chosen.get("inner_lower_tail"),
                    "threshold": chosen.get("threshold", 0.5),
                    **{k: chosen[k] for k in ("feature_set", "model", "weight_mode", "adaptation")},
                    **{k: v for k, v in nested.items() if k not in {"probs", "y_true", "positive_code", "label_names"}},
                }
            )
    return rows, selected_rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest, features, splits = load_inputs(args.training_dir)
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}
    original_10 = set(registry.get("original_10_comparator", []))
    lockbox_accessions = {e["accession"] for e in registry.get("lockbox", [])}

    if args.endpoint == "both":
        endpoints = ["keloid_binary", "clean_keloid_binary"]
    elif args.endpoint == "all":
        endpoints = list(ENDPOINT_TARGET.keys())
    else:
        endpoints = [args.endpoint]

    all_rows = []
    all_selected = []
    for endpoint in endpoints:
        rows, selected = run_endpoint(endpoint, args, manifest, features, splits, original_10, lockbox_accessions)
        all_rows.extend(rows)
        all_selected.extend(selected)

    results = pd.DataFrame(all_rows)
    selected = pd.DataFrame(all_selected)
    results.to_csv(args.out_dir / "public_expansion_loso_results.csv", index=False)
    results.to_json(args.out_dir / "public_expansion_loso_results.jsonl", orient="records", lines=True)
    selected.to_csv(args.out_dir / "public_expansion_nested_selected.csv", index=False)
    summary = summarize(results) if len(results) else pd.DataFrame()
    if len(summary):
        summary.to_csv(args.out_dir / "public_expansion_loso_summary.csv", index=False)

    # Locked comparator stats for current10 nested keloid_binary + clinical endpoints.
    report = {
        "endpoints": endpoints,
        "n_rows": int(len(results)),
        "selection_rule": "equal_accession_mean_inner_f1_with_lower_tail_tiebreak",
        "gates": {},
    }
    if len(results):
        cur = results[
            results.endpoint.eq("keloid_binary")
            & results.eval_mode.eq("current10")
            & results.stage.eq("nested_selected")
        ]
        if len(cur):
            ci = bootstrap_ci(cur["weighted_f1"].tolist())
            donor_vals = cur["donor_weighted_f1"].dropna().tolist() if "donor_weighted_f1" in cur else []
            donor_ci = bootstrap_ci(donor_vals) if donor_vals else {"mean": None, "lo": None, "hi": None}
            report["current10_nested"] = {
                "weighted_f1_macro_accession": ci,
                "donor_weighted_f1_pooled": donor_ci,
                "worst_fold": float(cur["weighted_f1"].min()),
                "balanced_accuracy_mean": float(cur["balanced_accuracy"].mean())
                if "balanced_accuracy" in cur
                else None,
                "auroc_mean": float(cur["auroc"].dropna().mean()) if "auroc" in cur else None,
                "n_folds": int(len(cur)),
            }
            report["gates"]["macro_ge_0_72"] = bool(ci["mean"] is not None and ci["mean"] >= 0.72)
            report["gates"]["donor_ge_0_70"] = bool(
                donor_ci["mean"] is not None and donor_ci["mean"] >= 0.70
            )
            report["gates"]["worst_ge_0_60"] = bool(cur["weighted_f1"].min() >= 0.60)

        scar = results[
            results.endpoint.eq("keloid_vs_normal_scar")
            & results.eval_mode.eq("expanded")
            & results.stage.eq("nested_selected")
        ]
        if len(scar):
            scar_ci = bootstrap_ci(scar["weighted_f1"].tolist())
            report["clinical_scar_nested"] = {
                "weighted_f1_macro_accession": scar_ci,
                "worst_fold": float(scar["weighted_f1"].min()),
                "n_folds": int(len(scar)),
            }
            report["gates"]["scar_n_accessions_ge_5"] = bool(len(scar) >= 5)
            report["gates"]["scar_macro_ge_0_75"] = bool(
                scar_ci["mean"] is not None and scar_ci["mean"] >= 0.75
            )
            report["gates"]["scar_worst_ge_0_60"] = bool(scar["weighted_f1"].min() >= 0.60)

        # Paired accession-level delta vs frozen baseline nested folds when available.
        if BASELINE_NESTED.exists() and len(cur):
            base = pd.read_csv(BASELINE_NESTED)
            base_cur = base[base.endpoint.eq("keloid_binary") & base.eval_mode.eq("current10")]
            if len(base_cur):
                merged = cur[["split_name", "weighted_f1"]].merge(
                    base_cur[["split_name", "weighted_f1"]],
                    on="split_name",
                    suffixes=("_new", "_base"),
                )
                if len(merged):
                    deltas = (merged["weighted_f1_new"] - merged["weighted_f1_base"]).tolist()
                    report["paired_delta_vs_baseline"] = {
                        "mean_delta_macro_f1": float(np.mean(deltas)),
                        "n_shared_folds": int(len(deltas)),
                        "baseline_report": str(BASELINE_REPORT.relative_to(PROJECT_ROOT))
                        if BASELINE_REPORT.exists()
                        else None,
                    }
                    report["gates"]["stop_if_lt_0_03_gain"] = bool(float(np.mean(deltas)) < 0.03)

    (args.out_dir / "public_expansion_loso_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
