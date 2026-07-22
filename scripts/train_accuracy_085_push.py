#!/usr/bin/env python3
"""Accuracy 0.85 push: primary-endpoint nested LOSO with train-only norms + ensemble.

Leakage-safe. Selection and thresholds on inner equal-accession OOF only.
Writes nested_selected / predictions under results/accuracy_085_push/<run_label>/.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from gene_modules import BREAKTHROUGH_FEATURE_VIEWS
from train_breakthrough_cascade_v2 import (
    PRIMARY,
    candidate_grid,
    loso_splits,
    majority_config,
)
from train_nested_loso import (
    bootstrap_ci,
    encode_binary,
    evaluate_fixed,
    fit_model,
    inner_select_equal_accession,
    load_inputs,
    model_specs,
    positive_proba,
    donor_accession_weights,
    split_rows,
    tune_threshold,
    _apply_platt,
    _fit_platt,
    _apply_train_only_norm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
OUT_DIR = PROJECT_ROOT / "results/accuracy_085_push"
REGISTRY_PATH = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"

PRIMARY_FEATURE_PREF = [
    "composition_only",
    "rank_programs_only",
    "fibrosis_only",
    "fused_multiview",
    "scar_discriminative",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--run-label", default="norm_ablation")
    p.add_argument("--random-state", type=int, default=13)
    p.add_argument(
        "--normalization-modes",
        nargs="+",
        default=["none", "train_accession_zscore", "train_reference_quantile", "train_combat_lite"],
    )
    p.add_argument(
        "--calibrations",
        nargs="+",
        default=["none", "platt_inner_oof"],
    )
    p.add_argument(
        "--features",
        nargs="+",
        default=[
            "composition_only",
            "rank_programs_only",
            "fibrosis_only",
            "fused_multiview",
            "low_i2_core_only",
            "published_markers_only",
        ],
    )
    p.add_argument("--ensemble-top-k", type=int, default=0, help="If >0, soft-vote top-K inner configs")
    p.add_argument("--selection-grain", choices=["donor", "profile"], default="donor")
    return p.parse_args()


def lockbox_set() -> set[str]:
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}
    lb = {
        e["accession"]
        for role in ("lockbox", "breakthrough_lockbox")
        for e in registry.get(role, [])
        if "accession" in e
    }
    lb |= {"GSE185309", "GSE212954", "GSE218007", "GSE237752", "GSE191067"}
    return lb


def _inner_oof_probs(
    *,
    split: dict,
    feature_table: pd.DataFrame,
    target: pd.Series,
    manifest: pd.DataFrame,
    model_name: str,
    feature_name: str,
    weight_mode: str,
    normalization_mode: str,
    calibration: str,
    random_state: int,
    platt_params: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    meta = manifest.set_index("sample_id")
    train_ids = [i for i in split["train_sample_ids"] if i in meta.index]
    if "lockbox" in meta.columns:
        train_ids = [i for i in train_ids if not bool(meta.loc[i, "lockbox"])]
    train_accs = sorted(set(meta.loc[train_ids, "accession"].astype(str)))
    oof_y, oof_p, oof_ids = [], [], []
    for held in train_accs:
        inner_test = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) == held]
        inner_train = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) != held]
        res = evaluate_fixed(
            feature_table=feature_table,
            target=target,
            train_ids=inner_train,
            test_ids=inner_test,
            manifest=manifest,
            model_name=model_name,
            feature_name=feature_name,
            weight_mode=weight_mode,
            random_state=random_state,
            threshold=0.5,
            normalization_mode=normalization_mode,
            calibration=calibration,
            platt_params=platt_params,
        )
        if res is None:
            continue
        oof_y.append(res["y_true"])
        oof_p.append(res["probs"])
        oof_ids.extend(inner_test)
    if not oof_y:
        return None
    return np.concatenate(oof_y), np.concatenate(oof_p)


def _score_config_on_outer(
    *,
    cfg: dict,
    split: dict,
    features: dict,
    target: pd.Series,
    manifest: pd.DataFrame,
    random_state: int,
) -> dict | None:
    return evaluate_fixed(
        feature_table=features[cfg["feature_set"]],
        target=target,
        train_ids=split["train_sample_ids"],
        test_ids=split["test_sample_ids"],
        manifest=manifest,
        model_name=cfg["model"],
        feature_name=cfg["feature_set"],
        weight_mode=cfg.get("weight_mode", "accession_donor"),
        random_state=random_state,
        threshold=cfg.get("threshold", 0.5),
        normalization_mode=cfg.get("normalization_mode", "none"),
        calibration=cfg.get("calibration", "none"),
        platt_params=cfg.get("platt_params"),
    )


def _rank_candidates_inner(
    split: dict,
    features: dict,
    target: pd.Series,
    manifest: pd.DataFrame,
    candidates: list[dict],
    random_state: int,
    selection_grain: str,
    top_k: int,
) -> list[dict]:
    """Return top-K configs ranked by inner equal-accession score (no outer labels)."""
    # Reuse existing single-best selector repeatedly by filtering already-chosen keys.
    remaining = list(candidates)
    chosen: list[dict] = []
    for _ in range(max(1, top_k)):
        if not remaining:
            break
        best = inner_select_equal_accession(
            split,
            features,
            target,
            manifest,
            remaining,
            random_state,
            selection_grain=selection_grain,
        )
        chosen.append(best)
        key = (
            best["feature_set"],
            best["model"],
            best.get("normalization_mode", "none"),
            best.get("calibration", "none"),
        )
        remaining = [
            c
            for c in remaining
            if (
                c["feature_set"],
                c["model"],
                c.get("normalization_mode", "none"),
                c.get("calibration", "none"),
            )
            != key
        ]
    return chosen


def _ensemble_outer(
    configs: list[dict],
    split: dict,
    features: dict,
    target: pd.Series,
    manifest: pd.DataFrame,
    random_state: int,
) -> dict | None:
    """Soft-vote outer-test probabilities from multiple inner-selected configs."""
    results = []
    for cfg in configs:
        res = _score_config_on_outer(
            cfg=cfg,
            split=split,
            features=features,
            target=target,
            manifest=manifest,
            random_state=random_state,
        )
        if res is not None:
            results.append((cfg, res))
    if not results:
        return None
    # Align on sample order from first result.
    base_cfg, base = results[0]
    probs = np.mean([r["probs"] for _, r in results], axis=0)
    # Threshold: median of member thresholds (inner-tuned).
    thr = float(np.median([c.get("threshold", 0.5) for c, _ in results]))
    labels = base["label_names"]
    positive_code = base["positive_code"]
    from train_nested_loso import metrics_from_probs

    metrics = metrics_from_probs(base["y_true"], probs, labels, thr, positive_code)
    # Donor grain
    meta = manifest.set_index("sample_id")
    x_test, _ = split_rows(features[base_cfg["feature_set"]], target, split["test_sample_ids"])
    groups = meta.loc[x_test.index, "split_group"].astype(str)
    donor_true, donor_prob = [], []
    for _, idx in groups.groupby(groups).groups.items():
        locs = [x_test.index.get_loc(i) for i in idx]
        donor_true.append(int(np.round(base["y_true"][locs].mean())))
        donor_prob.append(float(np.mean(probs[locs])))
    donor_metrics = None
    if len(donor_true) >= 1:
        donor_metrics = metrics_from_probs(
            np.asarray(donor_true), np.asarray(donor_prob), labels, thr, positive_code
        )
    return {
        **metrics,
        "probs": probs,
        "y_true": base["y_true"],
        "positive_code": positive_code,
        "label_names": labels,
        "n_train": base["n_train"],
        "n_test": base["n_test"],
        "n_test_donors": int(groups.nunique()),
        "donor_weighted_f1": None if donor_metrics is None else donor_metrics["weighted_f1"],
        "donor_macro_f1": None if donor_metrics is None else donor_metrics["macro_f1"],
        "donor_auroc": None if donor_metrics is None else donor_metrics.get("auroc"),
        "feature_set": "ensemble:" + "+".join(sorted({c["feature_set"] for c, _ in results})),
        "model": "soft_vote",
        "weight_mode": "accession_donor",
        "normalization_mode": "ensemble",
        "calibration": "ensemble",
        "threshold": thr,
        "ensemble_members": [
            {
                "feature_set": c["feature_set"],
                "model": c["model"],
                "normalization_mode": c.get("normalization_mode"),
                "calibration": c.get("calibration"),
                "threshold": c.get("threshold"),
                "inner_score": c.get("inner_score"),
            }
            for c, _ in results
        ],
    }


def run_primary(args: argparse.Namespace) -> dict:
    manifest, features, splits = load_inputs(args.training_dir)
    lockbox = lockbox_set()
    available = [f for f in args.features if f in features] or [
        f for f in PRIMARY_FEATURE_PREF if f in features
    ]
    candidates = candidate_grid(
        available,
        normalization_modes=args.normalization_modes,
        calibrations=args.calibrations,
    )
    endpoint_splits = loso_splits(splits, PRIMARY, lockbox)
    target = manifest.set_index("sample_id")[PRIMARY]

    selected_rows = []
    pred_rows = []
    for split in endpoint_splits:
        acc = split["split_name"].replace("leave_accession_out_", "")
        if args.ensemble_top_k and args.ensemble_top_k > 1:
            configs = _rank_candidates_inner(
                split,
                features,
                target,
                manifest,
                candidates,
                args.random_state,
                args.selection_grain,
                args.ensemble_top_k,
            )
            nested = _ensemble_outer(
                configs, split, features, target, manifest, args.random_state
            )
            chosen = {
                "feature_set": nested["feature_set"] if nested else "ensemble",
                "model": "soft_vote",
                "weight_mode": "accession_donor",
                "normalization_mode": "ensemble",
                "calibration": "ensemble",
                "threshold": nested["threshold"] if nested else 0.5,
                "inner_score": float(np.mean([c.get("inner_score") or 0 for c in configs])),
                "inner_lower_tail": None,
                "ensemble_members": nested.get("ensemble_members") if nested else [],
            }
        else:
            chosen = inner_select_equal_accession(
                split,
                features,
                target,
                manifest,
                candidates,
                args.random_state,
                selection_grain=args.selection_grain,
            )
            nested = _score_config_on_outer(
                cfg=chosen,
                split=split,
                features=features,
                target=target,
                manifest=manifest,
                random_state=args.random_state,
            )
        if nested is None:
            continue
        selected_rows.append(
            {
                "endpoint": PRIMARY,
                "stage": "nested_selected",
                "split_name": split["split_name"],
                "accession": acc,
                "selection": f"inner_equal_accession_{args.selection_grain}"
                + (f"_ensemble{args.ensemble_top_k}" if args.ensemble_top_k > 1 else ""),
                "inner_score": chosen.get("inner_score"),
                "inner_lower_tail": chosen.get("inner_lower_tail"),
                "threshold": chosen.get("threshold", nested.get("threshold", 0.5)),
                "normalization_mode": chosen.get("normalization_mode", "none"),
                "calibration": chosen.get("calibration", "none"),
                "feature_set": chosen.get("feature_set"),
                "model": chosen.get("model"),
                "weight_mode": chosen.get("weight_mode", "accession_donor"),
                "adaptation": "none",
                "n_train": nested.get("n_train"),
                "n_test": nested.get("n_test"),
                "n_test_donors": nested.get("n_test_donors"),
                "accuracy": nested.get("accuracy"),
                "weighted_f1": nested.get("weighted_f1"),
                "balanced_accuracy": nested.get("balanced_accuracy"),
                "macro_f1": nested.get("macro_f1"),
                "auroc": nested.get("auroc"),
                "confusion_matrix": nested.get("confusion_matrix"),
                "labels": nested.get("labels") or nested.get("label_names"),
                "donor_weighted_f1": nested.get("donor_weighted_f1"),
                "donor_macro_f1": nested.get("donor_macro_f1"),
                "donor_auroc": nested.get("donor_auroc"),
                "ensemble_members": json.dumps(chosen.get("ensemble_members") or []),
            }
        )
        meta = manifest.set_index("sample_id")
        feat_name = (
            chosen["feature_set"].split(":")[-1].split("+")[0]
            if str(chosen.get("feature_set", "")).startswith("ensemble:")
            else chosen["feature_set"]
        )
        if feat_name not in features:
            feat_name = available[0]
        x_test, _ = split_rows(features[feat_name], target, split["test_sample_ids"])
        for sid, yt, pr in zip(x_test.index.tolist(), nested["y_true"], nested["probs"]):
            pred_rows.append(
                {
                    "endpoint": PRIMARY,
                    "stage": "prediction",
                    "split_name": split["split_name"],
                    "accession": acc,
                    "sample_id": sid,
                    "donor": str(meta.loc[sid, "split_group"]) if sid in meta.index else "unknown",
                    "y_true": int(yt),
                    "prob_keloid": float(pr),
                    "feature_set": chosen.get("feature_set"),
                    "model": chosen.get("model"),
                    "threshold": chosen.get("threshold", nested.get("threshold", 0.5)),
                    "normalization_mode": chosen.get("normalization_mode"),
                }
            )

    selected_df = pd.DataFrame(selected_rows)
    preds_df = pd.DataFrame(pred_rows)
    run_dir = args.out_dir / args.run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(run_dir / "nested_selected.csv", index=False)
    preds_df.to_csv(run_dir / f"predictions_{PRIMARY}.csv", index=False)
    report = {
        "protocol": "accuracy_085_push_v1",
        "run_label": args.run_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "normalization_modes": args.normalization_modes,
        "calibrations": args.calibrations,
        "features": available,
        "ensemble_top_k": args.ensemble_top_k,
        "selection_grain": args.selection_grain,
        "n_folds": int(len(selected_df)),
        "frozen_config": majority_config(selected_df) if len(selected_df) else None,
        "sample_macro_mean": float(selected_df["macro_f1"].mean()) if len(selected_df) else None,
        "donor_macro_mean": float(selected_df["donor_macro_f1"].dropna().mean())
        if len(selected_df) and selected_df["donor_macro_f1"].notna().any()
        else None,
        "accessions": sorted(selected_df["accession"].unique().tolist()) if len(selected_df) else [],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    return {"run_dir": str(run_dir), "report": report}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        out = run_primary(args)
    print(json.dumps(out["report"], indent=2))


if __name__ == "__main__":
    main()
