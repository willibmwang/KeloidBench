#!/usr/bin/env python3
"""Breakthrough sprint: endpoint specialists + selective abstaining cascade.

Primary product: keloid_vs_unaffected_skin
Specialists: keloid_vs_normal_scar, keloid_vs_pathologic_scar
Selection: equal-accession mean inner F1 with lower-tail tie-break; no outer-test fallbacks.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from gene_modules import BREAKTHROUGH_FEATURE_VIEWS
from train_nested_loso import (
    BINARY_CLASSES,
    bootstrap_ci,
    donor_accession_weights,
    encode_binary,
    evaluate_fixed,
    fit_model,
    inner_select_equal_accession,
    load_inputs,
    metrics_from_probs,
    positive_proba,
    split_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint"
REGISTRY_PATH = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"

PRIMARY = "keloid_vs_unaffected_skin"
SPECIALISTS = ["keloid_vs_normal_scar", "keloid_vs_pathologic_scar"]
SENSITIVITY = ["fibroblast_keloid_binary"]
ALL_ENDPOINTS = [PRIMARY, *SPECIALISTS, *SENSITIVITY]

ENDPOINT_FEATURE_PREF = {
    PRIMARY: ["fused_multiview", "scar_discriminative", "fibrosis_only", "composition_only", "rank_programs_only"],
    "keloid_vs_normal_scar": ["scar_discriminative", "fused_multiview", "fibrosis_only", "composition_only"],
    "keloid_vs_pathologic_scar": ["scar_discriminative", "fused_multiview", "fibrosis_only", "composition_only"],
    "fibroblast_keloid_binary": ["fibrosis_only", "fused_multiview", "rank_programs_only", "composition_only"],
}

ABLATION_ORDER = [
    ("endpoint_cleanup", "rank_programs_only"),
    ("fibrosis_view", "fibrosis_only"),
    ("scar_discriminative_view", "scar_discriminative"),
    ("composition_view", "composition_only"),
    ("fused_experts", "fused_multiview"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--random-state", type=int, default=13)
    p.add_argument("--min-selective-coverage", type=float, default=0.60)
    return p.parse_args()


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
                    LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000, random_state=random_state),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(random_state=random_state, max_iter=8000)),
            ]
        ),
    }


def candidate_grid(feature_names: list[str]) -> list[dict]:
    out = []
    for f in feature_names:
        for m in ["ridge_logreg", "elastic_net_logreg", "linear_svm"]:
            out.append({"feature_set": f, "model": m, "weight_mode": "accession_donor", "adaptation": "none"})
    return out


def loso_splits(splits: list[dict], endpoint: str, lockbox: set[str], development_only: bool = True) -> list[dict]:
    out = []
    for s in splits:
        if s["task"] != endpoint:
            continue
        if not s["split_name"].startswith("leave_accession_out_"):
            continue
        acc = s["split_name"].replace("leave_accession_out_", "")
        if development_only and acc in lockbox:
            continue
        out.append(s)
    return out


def tune_selective_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_coverage: float,
) -> tuple[float, float, float]:
    """Maximize F1 among non-abstained samples subject to coverage >= min_coverage.

    Abstain when max(p, 1-p) < thr (i.e. confidence below thr).
    """
    best = (0.5, 0.0, 0.0)  # thr, f1, coverage
    for thr in np.linspace(0.50, 0.95, 19):
        conf = np.maximum(probs, 1.0 - probs)
        keep = conf >= thr
        coverage = float(keep.mean()) if len(keep) else 0.0
        if coverage < min_coverage or keep.sum() < 2:
            continue
        pred = np.where(probs[keep] >= 0.5, 0, 1)
        score = float(f1_score(y_true[keep], pred, average="weighted", zero_division=0))
        if score > best[1] or (score == best[1] and coverage > best[2]):
            best = (float(thr), score, coverage)
    return best


def run_endpoint_nested(
    endpoint: str,
    *,
    manifest: pd.DataFrame,
    features: dict,
    splits: list[dict],
    lockbox: set[str],
    random_state: int,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = endpoint
    if target_col not in manifest.columns:
        return pd.DataFrame(), pd.DataFrame()
    target = manifest.set_index("sample_id")[target_col]
    prefs = feature_names or ENDPOINT_FEATURE_PREF.get(endpoint, BREAKTHROUGH_FEATURE_VIEWS)
    available = [f for f in prefs if f in features]
    if not available:
        available = [f for f in BREAKTHROUGH_FEATURE_VIEWS if f in features]
    candidates = candidate_grid(available)
    endpoint_splits = loso_splits(splits, endpoint, lockbox, development_only=True)

    rows = []
    selected = []
    for split in endpoint_splits:
        chosen = inner_select_equal_accession(
            split, features, target, manifest, candidates, random_state
        )
        nested = evaluate_fixed(
            feature_table=features[chosen["feature_set"]],
            target=target,
            train_ids=split["train_sample_ids"],
            test_ids=split["test_sample_ids"],
            manifest=manifest,
            model_name=chosen["model"],
            feature_name=chosen["feature_set"],
            weight_mode=chosen["weight_mode"],
            random_state=random_state,
            threshold=chosen.get("threshold", 0.5),
        )
        if nested is None:
            continue
        acc = split["split_name"].replace("leave_accession_out_", "")
        # Leakage guard: outer accession must not appear in train ids.
        train_accs = set(manifest.set_index("sample_id").loc[split["train_sample_ids"], "accession"].astype(str))
        assert acc not in train_accs, f"LEAK: {acc} in train for {endpoint}"
        row = {
            "endpoint": endpoint,
            "stage": "nested_selected",
            "split_name": split["split_name"],
            "accession": acc,
            "selection": "inner_equal_accession",
            "inner_score": chosen.get("inner_score"),
            "inner_lower_tail": chosen.get("inner_lower_tail"),
            "threshold": chosen.get("threshold", 0.5),
            **{k: v for k, v in nested.items() if k not in {"probs", "y_true", "positive_code", "label_names"}},
        }
        rows.append(row)
        selected.append(row)
        # Also store prediction artifacts for cascade.
        meta = manifest.set_index("sample_id")
        test_ids = [i for i in split["test_sample_ids"] if i in nested["y_true"].index] if hasattr(nested["y_true"], "index") else split["test_sample_ids"]
        # evaluate_fixed returns numpy arrays; recover ids from feature table order.
        x_test, _ = split_rows(features[chosen["feature_set"]], target, split["test_sample_ids"])
        for sid, yt, pr in zip(x_test.index.tolist(), nested["y_true"], nested["probs"]):
            rows.append(
                {
                    "endpoint": endpoint,
                    "stage": "prediction",
                    "split_name": split["split_name"],
                    "accession": acc,
                    "sample_id": sid,
                    "donor": str(meta.loc[sid, "split_group"]) if sid in meta.index else "unknown",
                    "y_true": int(yt),
                    "prob_keloid": float(pr),
                    "feature_set": chosen["feature_set"],
                    "model": chosen["model"],
                    "threshold": chosen.get("threshold", 0.5),
                }
            )
    return pd.DataFrame([r for r in rows if r.get("stage") == "nested_selected"]), pd.DataFrame(
        [r for r in rows if r.get("stage") == "prediction"]
    )


def run_ablations(
    endpoint: str,
    *,
    manifest,
    features,
    splits,
    lockbox,
    random_state,
) -> pd.DataFrame:
    """Fixed-view ablation (ridge, thr=0.5) for waterfall plots; nested selection remains primary."""
    target = manifest.set_index("sample_id")[endpoint]
    endpoint_splits = loso_splits(splits, endpoint, lockbox, development_only=True)
    rows = []
    for stage_name, feature_name in ABLATION_ORDER:
        if feature_name not in features:
            continue
        fold_scores = []
        donor_scores = []
        for split in endpoint_splits:
            result = evaluate_fixed(
                feature_table=features[feature_name],
                target=target,
                train_ids=split["train_sample_ids"],
                test_ids=split["test_sample_ids"],
                manifest=manifest,
                model_name="ridge_logreg",
                feature_name=feature_name,
                weight_mode="accession_donor",
                random_state=random_state,
                threshold=0.5,
            )
            if result is None:
                continue
            fold_scores.append(result["weighted_f1"])
            if result.get("donor_weighted_f1") is not None:
                donor_scores.append(result["donor_weighted_f1"])
        if not fold_scores:
            continue
        ci = bootstrap_ci(fold_scores)
        rows.append(
            {
                "endpoint": endpoint,
                "ablation_stage": stage_name,
                "feature_set": feature_name,
                "macro_f1_mean": ci["mean"],
                "macro_f1_lo": ci["lo"],
                "macro_f1_hi": ci["hi"],
                "worst_fold": float(min(fold_scores)),
                "n_folds": int(len(fold_scores)),
                "donor_f1_mean": float(np.mean(donor_scores)) if donor_scores else None,
            }
        )
    return pd.DataFrame(rows)


def build_selective_cascade(
    primary_preds: pd.DataFrame,
    pathologic_preds: pd.DataFrame,
    min_coverage: float,
) -> tuple[pd.DataFrame, dict]:
    """Primary skin expert with confidence abstention; pathologic specialist as secondary route."""
    if primary_preds.empty:
        return pd.DataFrame(), {"status": "no_primary_predictions"}

    # Aggregate per accession fold from primary predictions.
    fold_rows = []
    for (acc, split_name), group in primary_preds.groupby(["accession", "split_name"]):
        y = group["y_true"].to_numpy()
        p = group["prob_keloid"].to_numpy()
        thr_conf, sel_f1, cov = tune_selective_threshold(y, p, min_coverage)
        # Full-coverage metrics at 0.5
        pred_full = np.where(p >= 0.5, 0, 1)
        full_f1 = float(f1_score(y, pred_full, average="weighted", zero_division=0))
        conf = np.maximum(p, 1.0 - p)
        keep = conf >= thr_conf
        fold_rows.append(
            {
                "accession": acc,
                "split_name": split_name,
                "full_coverage_f1": full_f1,
                "selective_f1": sel_f1,
                "selective_coverage": cov,
                "confidence_threshold": thr_conf,
                "n_test": int(len(y)),
                "n_kept": int(keep.sum()),
                "n_abstain": int((~keep).sum()),
            }
        )
        # Attach per-sample cascade decisions
        for sid, yt, pr, c in zip(group["sample_id"], y, p, conf):
            decision = "abstain" if c < thr_conf else ("keloid" if pr >= 0.5 else "non_keloid")
            route = "primary_skin"
            if decision == "keloid" and not pathologic_preds.empty:
                # Optional specialist annotation (does not override skin claim).
                sub = pathologic_preds[pathologic_preds["sample_id"].eq(sid)]
                if len(sub):
                    route = "primary_then_pathologic_specialist"
            fold_rows.append(
                {
                    "stage": "cascade_prediction",
                    "accession": acc,
                    "split_name": split_name,
                    "sample_id": sid,
                    "y_true": int(yt),
                    "prob_keloid": float(pr),
                    "confidence": float(c),
                    "decision": decision,
                    "route": route,
                    "confidence_threshold": thr_conf,
                }
            )

    fold_metrics = pd.DataFrame([r for r in fold_rows if "full_coverage_f1" in r])
    preds = pd.DataFrame([r for r in fold_rows if r.get("stage") == "cascade_prediction"])
    summary = {
        "n_folds": int(len(fold_metrics)),
        "full_coverage_macro_f1": bootstrap_ci(fold_metrics["full_coverage_f1"].tolist())
        if len(fold_metrics)
        else None,
        "selective_macro_f1": bootstrap_ci(fold_metrics["selective_f1"].tolist()) if len(fold_metrics) else None,
        "mean_selective_coverage": float(fold_metrics["selective_coverage"].mean()) if len(fold_metrics) else None,
        "worst_full_f1": float(fold_metrics["full_coverage_f1"].min()) if len(fold_metrics) else None,
        "worst_selective_f1": float(fold_metrics["selective_f1"].min()) if len(fold_metrics) else None,
    }
    return preds, {"fold_metrics": fold_metrics.to_dict(orient="records"), "summary": summary}


def majority_config(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {
            "feature_set": "fused_multiview",
            "model": "ridge_logreg",
            "weight_mode": "accession_donor",
            "threshold": 0.5,
        }
    keys = list(zip(selected["feature_set"], selected["model"], selected["weight_mode"]))
    from collections import Counter

    (feature_set, model, weight_mode), _ = Counter(keys).most_common(1)[0]
    thr = float(selected["threshold"].median())
    return {
        "feature_set": feature_set,
        "model": model,
        "weight_mode": weight_mode,
        "threshold": thr,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out_dir / "frozen_protocol.json"
    # Fall back to v1 path only when training into the default v1 directory.
    if not protocol_path.exists() and args.out_dir.resolve() == OUT_DIR.resolve():
        protocol_path = OUT_DIR / "frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}

    lockbox = {
        *{e["accession"] for e in registry.get("lockbox", [])},
        protocol.get("breakthrough_lockbox", {}).get("accession", "GSE185309"),
        "GSE212954",
    }
    lockbox.discard(None)

    manifest, features, splits = load_inputs(args.training_dir)
    # Ensure breakthrough views exist.
    missing_views = [v for v in BREAKTHROUGH_FEATURE_VIEWS if v not in features]
    if missing_views:
        raise SystemExit(
            f"Missing breakthrough feature views {missing_views}. Rebuild corpus first."
        )

    all_selected = []
    all_preds = []
    ablation_frames = []
    endpoint_reports = {}

    for endpoint in ALL_ENDPOINTS:
        print(f"=== nested LOSO {endpoint} ===")
        selected, preds = run_endpoint_nested(
            endpoint,
            manifest=manifest,
            features=features,
            splits=splits,
            lockbox=lockbox,
            random_state=args.random_state,
        )
        if not selected.empty:
            all_selected.append(selected)
            ci = bootstrap_ci(selected["weighted_f1"].tolist())
            endpoint_reports[endpoint] = {
                "macro_accession_f1": ci,
                "worst_fold": float(selected["weighted_f1"].min()),
                "n_folds": int(len(selected)),
                "frozen_config": majority_config(selected),
                "accessions": sorted(selected["accession"].unique().tolist()),
            }
        if not preds.empty:
            all_preds.append(preds)
        if endpoint == PRIMARY:
            abl = run_ablations(
                endpoint,
                manifest=manifest,
                features=features,
                splits=splits,
                lockbox=lockbox,
                random_state=args.random_state,
            )
            if not abl.empty:
                ablation_frames.append(abl)

    selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    ablation_df = pd.concat(ablation_frames, ignore_index=True) if ablation_frames else pd.DataFrame()

    primary_preds = preds_df[preds_df.endpoint.eq(PRIMARY)] if len(preds_df) else pd.DataFrame()
    path_preds = (
        preds_df[preds_df.endpoint.eq("keloid_vs_pathologic_scar")] if len(preds_df) else pd.DataFrame()
    )
    cascade_preds, cascade_report = build_selective_cascade(
        primary_preds, path_preds, args.min_selective_coverage
    )

    selected_df.to_csv(args.out_dir / "breakthrough_nested_selected.csv", index=False)
    preds_df.to_csv(args.out_dir / "breakthrough_predictions.csv", index=False)
    if len(ablation_df):
        ablation_df.to_csv(args.out_dir / "breakthrough_ablations.csv", index=False)
    if len(cascade_preds):
        cascade_preds.to_csv(args.out_dir / "breakthrough_cascade_predictions.csv", index=False)

    primary_report = endpoint_reports.get(PRIMARY, {})
    cascade_summary = cascade_report.get("summary", {})
    gates = {}
    if primary_report.get("macro_accession_f1"):
        mean_f1 = primary_report["macro_accession_f1"]["mean"]
        gates["full_coverage_macro_ge_0_85"] = bool(mean_f1 is not None and mean_f1 >= 0.85)
        gates["worst_ge_0_75"] = bool(primary_report.get("worst_fold", 0) >= 0.75)
    if cascade_summary.get("selective_macro_f1"):
        sel_mean = cascade_summary["selective_macro_f1"]["mean"]
        cov = cascade_summary.get("mean_selective_coverage")
        gates["selective_macro_ge_0_90"] = bool(sel_mean is not None and sel_mean >= 0.90)
        gates["selective_coverage_ge_0_60"] = bool(cov is not None and cov >= 0.60)

    frozen = {
        "protocol": "breakthrough_sprint_v1",
        "primary_endpoint": PRIMARY,
        "primary_config": primary_report.get("frozen_config"),
        "specialist_configs": {
            ep: endpoint_reports[ep]["frozen_config"]
            for ep in SPECIALISTS
            if ep in endpoint_reports
        },
        "cascade": {
            "min_selective_coverage": args.min_selective_coverage,
            "summary": cascade_summary,
        },
        "breakthrough_lockbox": protocol.get("breakthrough_lockbox", {}).get("accession"),
        "ladder_reserved_lockbox": protocol.get("ladder_reserved_lockbox", []),
    }
    (args.out_dir / "breakthrough_frozen_model.json").write_text(json.dumps(frozen, indent=2))

    report = {
        "protocol": "breakthrough_sprint_v1",
        "endpoints": endpoint_reports,
        "cascade": cascade_summary,
        "gates": gates,
        "claim_rule": (
            "High-accuracy claim requires nested primary full-coverage macro F1>=0.85 "
            "(or selective F1>=0.90 at coverage>=0.60) AND the same rule on the untouched "
            "breakthrough lockbox. Scar specialists are reported separately."
        ),
        "n_selected_rows": int(len(selected_df)),
        "n_prediction_rows": int(len(preds_df)),
    }
    (args.out_dir / "breakthrough_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
