#!/usr/bin/env python3
"""Breakthrough sprint v2: endpoint-pure nested LOSO + specimen-aware cascade.

Leakage-safe selective thresholds are tuned on inner accession-blocked OOF only.
v1 artifacts under results/breakthrough_sprint/ are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from gene_modules import BREAKTHROUGH_FEATURE_VIEWS
from train_breakthrough_cascade import (
    PRIMARY,
    SENSITIVITY,
    SPECIALISTS,
    tune_selective_threshold,
)
from train_nested_loso import (
    bootstrap_ci,
    evaluate_fixed,
    inner_select_equal_accession,
    load_inputs,
    split_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint_v2"
REGISTRY_PATH = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"

ALL_ENDPOINTS = [PRIMARY, *SPECIALISTS, *SENSITIVITY]

ENDPOINT_FEATURE_PREF = {
    PRIMARY: ["fused_multiview", "scar_discriminative", "fibrosis_only", "composition_only", "rank_programs_only"],
    "keloid_vs_normal_scar": ["scar_discriminative", "fused_multiview", "fibrosis_only", "composition_only"],
    "keloid_vs_pathologic_scar": ["scar_discriminative", "fused_multiview", "fibrosis_only", "composition_only"],
    "fibroblast_keloid_binary": ["fibrosis_only", "fused_multiview", "rank_programs_only", "composition_only"],
}

COMPARTMENT_ROUTE = {
    "bulk_tissue": PRIMARY,
    "scrna_pseudobulk": PRIMARY,
    "primary_fibroblast": "fibroblast_keloid_binary",
    "fibroblast_perturbed": None,
    "cell_line": None,
    "endothelial": None,
    "keratinocyte": None,
    "other": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--random-state", type=int, default=13)
    p.add_argument("--min-selective-coverage", type=float, default=0.60)
    p.add_argument(
        "--stage",
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help="Ablation stage: A endpoint-pure; B donor+calib; C +norm; D specimen cascade",
    )
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


def candidate_grid(
    feature_names: list[str],
    *,
    normalization_modes: list[str],
    calibrations: list[str],
) -> list[dict]:
    out = []
    for f in feature_names:
        for m in ["ridge_logreg", "elastic_net_logreg", "linear_svm"]:
            for norm in normalization_modes:
                for calib in calibrations:
                    out.append(
                        {
                            "feature_set": f,
                            "model": m,
                            "weight_mode": "accession_donor",
                            "adaptation": "none",
                            "normalization_mode": norm,
                            "calibration": calib,
                        }
                    )
    return out


def loso_splits(splits: list[dict], endpoint: str, lockbox: set[str]) -> list[dict]:
    out = []
    for s in splits:
        if s["task"] != endpoint:
            continue
        if not s["split_name"].startswith("leave_accession_out_"):
            continue
        acc = s["split_name"].replace("leave_accession_out_", "")
        if acc in lockbox:
            continue
        out.append(s)
    return out


def majority_config(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {
            "feature_set": "fused_multiview",
            "model": "ridge_logreg",
            "weight_mode": "accession_donor",
            "threshold": 0.5,
            "normalization_mode": "none",
            "calibration": "none",
        }
    keys = list(
        zip(
            selected["feature_set"],
            selected["model"],
            selected["weight_mode"],
            selected.get("normalization_mode", pd.Series(["none"] * len(selected))),
            selected.get("calibration", pd.Series(["none"] * len(selected))),
        )
    )
    (feature_set, model, weight_mode, norm, calib), _ = Counter(keys).most_common(1)[0]
    return {
        "feature_set": feature_set,
        "model": model,
        "weight_mode": weight_mode,
        "threshold": float(selected["threshold"].median()),
        "normalization_mode": norm,
        "calibration": calib,
    }


def run_endpoint_nested(
    endpoint: str,
    *,
    manifest: pd.DataFrame,
    features: dict,
    splits: list[dict],
    lockbox: set[str],
    random_state: int,
    selection_grain: str,
    normalization_modes: list[str],
    calibrations: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Nested LOSO with optional donor-grain selection and inner OOF selective thr."""
    if endpoint not in manifest.columns:
        return pd.DataFrame(), pd.DataFrame(), {}
    target = manifest.set_index("sample_id")[endpoint]
    prefs = ENDPOINT_FEATURE_PREF.get(endpoint, BREAKTHROUGH_FEATURE_VIEWS)
    available = [f for f in prefs if f in features] or [f for f in BREAKTHROUGH_FEATURE_VIEWS if f in features]
    candidates = candidate_grid(
        available, normalization_modes=normalization_modes, calibrations=calibrations
    )
    endpoint_splits = loso_splits(splits, endpoint, lockbox)

    selected_rows = []
    pred_rows = []
    inner_selective = {}  # accession -> confidence threshold from inner OOF

    for split in endpoint_splits:
        chosen = inner_select_equal_accession(
            split,
            features,
            target,
            manifest,
            candidates,
            random_state,
            selection_grain=selection_grain,
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
            normalization_mode=chosen.get("normalization_mode", "none"),
            calibration=chosen.get("calibration", "none"),
            platt_params=chosen.get("platt_params"),
        )
        if nested is None:
            continue
        acc = split["split_name"].replace("leave_accession_out_", "")
        train_accs = set(
            manifest.set_index("sample_id").loc[split["train_sample_ids"], "accession"].astype(str)
        )
        assert acc not in train_accs, f"LEAK: {acc} in train for {endpoint}"

        # Inner-OOF selective threshold (no outer labels).
        meta = manifest.set_index("sample_id")
        train_ids = [i for i in split["train_sample_ids"] if i in meta.index]
        train_acc_list = sorted(
            a
            for a in set(meta.loc[train_ids, "accession"].astype(str))
            if not (bool(meta.loc[meta["accession"].eq(a), "lockbox"].any()) if "lockbox" in meta.columns else False)
        )
        oof_y, oof_p = [], []
        for held in train_acc_list:
            inner_test = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) == held]
            inner_train = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) != held]
            if "lockbox" in meta.columns:
                inner_train = [sid for sid in inner_train if not bool(meta.loc[sid, "lockbox"])]
            inner_res = evaluate_fixed(
                feature_table=features[chosen["feature_set"]],
                target=target,
                train_ids=inner_train,
                test_ids=inner_test,
                manifest=manifest,
                model_name=chosen["model"],
                feature_name=chosen["feature_set"],
                weight_mode=chosen["weight_mode"],
                random_state=random_state,
                threshold=0.5,
                normalization_mode=chosen.get("normalization_mode", "none"),
                calibration=chosen.get("calibration", "none"),
                platt_params=chosen.get("platt_params"),
            )
            if inner_res is None:
                continue
            oof_y.append(inner_res["y_true"])
            oof_p.append(inner_res["probs"])
        sel_thr = 0.5
        if oof_y:
            y_cat = np.concatenate(oof_y)
            p_cat = np.concatenate(oof_p)
            sel_thr, _, _ = tune_selective_threshold(y_cat, p_cat, min_coverage=0.60)
        inner_selective[acc] = float(sel_thr)

        selected_rows.append(
            {
                "endpoint": endpoint,
                "stage": "nested_selected",
                "split_name": split["split_name"],
                "accession": acc,
                "selection": f"inner_equal_accession_{selection_grain}",
                "inner_score": chosen.get("inner_score"),
                "inner_lower_tail": chosen.get("inner_lower_tail"),
                "threshold": chosen.get("threshold", 0.5),
                "selective_confidence_threshold": sel_thr,
                "normalization_mode": chosen.get("normalization_mode", "none"),
                "calibration": chosen.get("calibration", "none"),
                **{
                    k: v
                    for k, v in nested.items()
                    if k
                    not in {
                        "probs",
                        "y_true",
                        "positive_code",
                        "label_names",
                        "donor_true",
                        "donor_prob",
                    }
                },
            }
        )
        x_test, _ = split_rows(features[chosen["feature_set"]], target, split["test_sample_ids"])
        for sid, yt, pr in zip(x_test.index.tolist(), nested["y_true"], nested["probs"]):
            compartment = (
                str(meta.loc[sid, "specimen_compartment"])
                if "specimen_compartment" in meta.columns and sid in meta.index
                else "unknown"
            )
            pred_rows.append(
                {
                    "endpoint": endpoint,
                    "stage": "prediction",
                    "split_name": split["split_name"],
                    "accession": acc,
                    "sample_id": sid,
                    "donor": str(meta.loc[sid, "split_group"]) if sid in meta.index else "unknown",
                    "specimen_compartment": compartment,
                    "y_true": int(yt),
                    "prob_keloid": float(pr),
                    "feature_set": chosen["feature_set"],
                    "model": chosen["model"],
                    "threshold": chosen.get("threshold", 0.5),
                    "selective_confidence_threshold": sel_thr,
                }
            )
    return pd.DataFrame(selected_rows), pd.DataFrame(pred_rows), inner_selective


def build_specimen_cascade(
    preds_by_endpoint: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    min_coverage: float,
    use_routing: bool,
) -> tuple[pd.DataFrame, dict]:
    """Route by specimen_compartment metadata; abstain via inner-tuned confidence thr."""
    primary = preds_by_endpoint.get(PRIMARY, pd.DataFrame())
    if primary.empty:
        return pd.DataFrame(), {"status": "no_primary_predictions"}
    fib = preds_by_endpoint.get("fibroblast_keloid_binary", pd.DataFrame())
    path = preds_by_endpoint.get("keloid_vs_pathologic_scar", pd.DataFrame())
    meta = manifest.set_index("sample_id")

    fold_metrics = []
    cascade_rows = []
    for (acc, split_name), group in primary.groupby(["accession", "split_name"]):
        y = group["y_true"].to_numpy()
        decisions = []
        probs_used = []
        kept_mask = []
        for _, row in group.iterrows():
            sid = row["sample_id"]
            compartment = row.get("specimen_compartment")
            if pd.isna(compartment) and sid in meta.index and "specimen_compartment" in meta.columns:
                compartment = str(meta.loc[sid, "specimen_compartment"])
            compartment = str(compartment or "bulk_tissue")
            route_ep = COMPARTMENT_ROUTE.get(compartment, PRIMARY) if use_routing else PRIMARY
            prob = float(row["prob_keloid"])
            route = "primary_skin"
            if use_routing and route_ep == "fibroblast_keloid_binary" and not fib.empty:
                sub = fib[fib["sample_id"].eq(sid)]
                if len(sub):
                    prob = float(sub.iloc[0]["prob_keloid"])
                    route = "fibroblast_specialist"
                else:
                    route = "primary_skin_fallback"
            elif use_routing and route_ep is None:
                # Compartment has no expert → abstain.
                cascade_rows.append(
                    {
                        "stage": "cascade_prediction",
                        "accession": acc,
                        "split_name": split_name,
                        "sample_id": sid,
                        "y_true": int(row["y_true"]),
                        "prob_keloid": prob,
                        "confidence": float(max(prob, 1 - prob)),
                        "decision": "abstain",
                        "route": "no_expert_for_compartment",
                        "confidence_threshold": float(row.get("selective_confidence_threshold", 0.5)),
                        "specimen_compartment": compartment,
                    }
                )
                decisions.append("abstain")
                probs_used.append(prob)
                kept_mask.append(False)
                continue

            thr_conf = float(row.get("selective_confidence_threshold", 0.5))
            conf = max(prob, 1.0 - prob)
            if conf < thr_conf:
                decision = "abstain"
            else:
                decision = "keloid" if prob >= 0.5 else "non_keloid"
                if decision == "keloid" and not path.empty:
                    if len(path[path["sample_id"].eq(sid)]):
                        route = f"{route}_then_pathologic_specialist"
            cascade_rows.append(
                {
                    "stage": "cascade_prediction",
                    "accession": acc,
                    "split_name": split_name,
                    "sample_id": sid,
                    "y_true": int(row["y_true"]),
                    "prob_keloid": prob,
                    "confidence": conf,
                    "decision": decision,
                    "route": route,
                    "confidence_threshold": thr_conf,
                    "specimen_compartment": compartment,
                }
            )
            decisions.append(decision)
            probs_used.append(prob)
            kept_mask.append(decision != "abstain")

        kept = np.asarray(kept_mask)
        coverage = float(kept.mean()) if len(kept) else 0.0
        pred_full = np.where(np.asarray(probs_used) >= 0.5, 0, 1)
        full_f1 = float(f1_score(y, pred_full, average="weighted", zero_division=0))
        if kept.sum() >= 2:
            pred_sel = np.where(np.asarray(probs_used)[kept] >= 0.5, 0, 1)
            sel_f1 = float(f1_score(y[kept], pred_sel, average="weighted", zero_division=0))
        else:
            sel_f1 = 0.0
        fold_metrics.append(
            {
                "accession": acc,
                "split_name": split_name,
                "full_coverage_f1": full_f1,
                "selective_f1": sel_f1,
                "selective_coverage": coverage,
                "n_test": int(len(y)),
                "n_kept": int(kept.sum()),
                "n_abstain": int((~kept).sum()),
            }
        )

    fold_df = pd.DataFrame(fold_metrics)
    preds = pd.DataFrame(cascade_rows)
    summary = {
        "n_folds": int(len(fold_df)),
        "full_coverage_macro_f1": bootstrap_ci(fold_df["full_coverage_f1"].tolist()) if len(fold_df) else None,
        "selective_macro_f1": bootstrap_ci(fold_df["selective_f1"].tolist()) if len(fold_df) else None,
        "mean_selective_coverage": float(fold_df["selective_coverage"].mean()) if len(fold_df) else None,
        "worst_full_f1": float(fold_df["full_coverage_f1"].min()) if len(fold_df) else None,
        "worst_selective_f1": float(fold_df["selective_f1"].min()) if len(fold_df) else None,
        "specimen_routing": bool(use_routing),
        "selective_tune_on": "inner_equal_accession_oof_only",
    }
    return preds, {"fold_metrics": fold_df.to_dict(orient="records"), "summary": summary}


def stage_config(stage: str) -> dict:
    if stage == "A":
        return {
            "selection_grain": "profile",
            "normalization_modes": ["none"],
            "calibrations": ["none"],
            "use_routing": False,
            "label": "stage_a_endpoint_pure_baseline",
        }
    if stage == "B":
        return {
            "selection_grain": "donor",
            "normalization_modes": ["none"],
            "calibrations": ["none", "platt_inner_oof"],
            "use_routing": False,
            "label": "stage_b_donor_selection_calibration",
        }
    if stage == "C":
        return {
            "selection_grain": "donor",
            "normalization_modes": ["none", "train_accession_zscore"],
            "calibrations": ["none", "platt_inner_oof"],
            "use_routing": False,
            "label": "stage_c_train_only_normalization",
        }
    return {
        "selection_grain": "donor",
        "normalization_modes": ["none", "train_accession_zscore"],
        "calibrations": ["none", "platt_inner_oof"],
        "use_routing": True,
        "label": "stage_d_specimen_routed_cascade",
    }


def run_stage(stage: str, args: argparse.Namespace, protocol: dict, lockbox: set[str]) -> dict:
    cfg = stage_config(stage)
    stage_dir = args.out_dir / "ablations" / cfg["label"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Stage {stage}: {cfg['label']} ===")

    manifest, features, splits = load_inputs(args.training_dir)
    missing = [v for v in BREAKTHROUGH_FEATURE_VIEWS if v not in features]
    if missing:
        raise SystemExit(f"Missing feature views {missing}")

    all_selected = []
    preds_by_ep: dict[str, pd.DataFrame] = {}
    endpoint_reports = {}

    for endpoint in ALL_ENDPOINTS:
        print(f"  nested LOSO {endpoint}")
        selected, preds, _ = run_endpoint_nested(
            endpoint,
            manifest=manifest,
            features=features,
            splits=splits,
            lockbox=lockbox,
            random_state=args.random_state,
            selection_grain=cfg["selection_grain"],
            normalization_modes=cfg["normalization_modes"],
            calibrations=cfg["calibrations"],
        )
        if not selected.empty:
            all_selected.append(selected)
            ci = bootstrap_ci(selected["weighted_f1"].tolist())
            donor_vals = selected["donor_weighted_f1"].dropna().tolist() if "donor_weighted_f1" in selected else []
            endpoint_reports[endpoint] = {
                "macro_accession_f1": ci,
                "worst_fold": float(selected["weighted_f1"].min()),
                "n_folds": int(len(selected)),
                "donor_f1_mean": float(np.mean(donor_vals)) if donor_vals else None,
                "frozen_config": majority_config(selected),
                "accessions": sorted(selected["accession"].unique().tolist()),
            }
        if not preds.empty:
            preds_by_ep[endpoint] = preds

    selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    cascade_preds, cascade_report = build_specimen_cascade(
        preds_by_ep,
        manifest,
        args.min_selective_coverage,
        use_routing=cfg["use_routing"],
    )
    selected_df.to_csv(stage_dir / "nested_selected.csv", index=False)
    for ep, pdf in preds_by_ep.items():
        pdf.to_csv(stage_dir / f"predictions_{ep}.csv", index=False)
    if len(cascade_preds):
        cascade_preds.to_csv(stage_dir / "cascade_predictions.csv", index=False)

    primary_report = endpoint_reports.get(PRIMARY, {})
    cascade_summary = cascade_report.get("summary", {})
    gates = {}
    interim = {}
    if primary_report.get("macro_accession_f1"):
        mean_f1 = primary_report["macro_accession_f1"]["mean"]
        worst = primary_report.get("worst_fold", 0)
        gates["full_coverage_macro_ge_0_85"] = bool(mean_f1 is not None and mean_f1 >= 0.85)
        gates["worst_ge_0_75"] = bool(worst >= 0.75)
        interim["worst_ge_0_50"] = bool(worst >= 0.50)
        interim["macro_ge_0_70"] = bool(mean_f1 is not None and mean_f1 >= 0.70)
        interim["no_zero_fold"] = bool(worst > 0.0)
    if cascade_summary.get("selective_macro_f1"):
        sel_mean = cascade_summary["selective_macro_f1"]["mean"]
        cov = cascade_summary.get("mean_selective_coverage")
        gates["selective_macro_ge_0_90"] = bool(sel_mean is not None and sel_mean >= 0.90)
        gates["selective_coverage_ge_0_60"] = bool(cov is not None and cov >= 0.60)

    report = {
        "protocol": "breakthrough_sprint_v2",
        "stage": cfg["label"],
        "stage_config": cfg,
        "endpoints": endpoint_reports,
        "cascade": cascade_summary,
        "gates": gates,
        "interim_targets": interim,
        "n_selected_rows": int(len(selected_df)),
    }
    (stage_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"stage": stage, "primary": primary_report.get("macro_accession_f1"), "worst": primary_report.get("worst_fold"), "gates": gates, "interim": interim}, indent=2))
    return report


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out_dir / "frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}

    lockbox = {
        *{e["accession"] for e in registry.get("lockbox", [])},
        protocol.get("breakthrough_lockbox", {}).get("accession"),
        protocol.get("v1_immutable", {}).get("breakthrough_lockbox_accession", "GSE185309"),
        "GSE212954",
        "GSE185309",
        "GSE218007",
    }
    lockbox.discard(None)
    # Also exclude interpretation-only additions from nested test/train via lockbox-like skip of eval?
    # They remain in corpus but should not be primary skin folds after eligibility rebuild.

    stages = ["A", "B", "C", "D"] if args.stage == "all" else [args.stage]
    summaries = []
    for st in stages:
        summaries.append(run_stage(st, args, protocol, lockbox))

    # Final stage report promoted to top-level v2 outputs.
    final = summaries[-1]
    (args.out_dir / "breakthrough_report.json").write_text(json.dumps(final, indent=2))
    frozen = {
        "protocol": "breakthrough_sprint_v2",
        "primary_endpoint": PRIMARY,
        "primary_config": final.get("endpoints", {}).get(PRIMARY, {}).get("frozen_config"),
        "specialist_configs": {
            ep: final["endpoints"][ep]["frozen_config"]
            for ep in SPECIALISTS
            if ep in final.get("endpoints", {})
        },
        "cascade": final.get("cascade"),
        "breakthrough_lockbox": protocol.get("breakthrough_lockbox", {}).get("accession"),
        "v1_lockbox_immutable": protocol.get("v1_immutable", {}).get("breakthrough_lockbox_accession"),
        "do_not_rescore_v1_lockbox": True,
    }
    (args.out_dir / "breakthrough_frozen_model.json").write_text(json.dumps(frozen, indent=2))
    (args.out_dir / "ablation_summary.json").write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "stage": s.get("stage"),
                        "primary_macro": (s.get("endpoints", {}).get(PRIMARY, {}).get("macro_accession_f1") or {}).get("mean"),
                        "worst_fold": s.get("endpoints", {}).get(PRIMARY, {}).get("worst_fold"),
                        "gates": s.get("gates"),
                        "interim_targets": s.get("interim_targets"),
                        "cascade_selective": (s.get("cascade") or {}).get("selective_macro_f1"),
                    }
                    for s in summaries
                ]
            },
            indent=2,
        )
    )
    print(json.dumps({"written": str(args.out_dir), "n_stages": len(summaries)}, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        main()
