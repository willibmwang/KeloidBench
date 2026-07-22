#!/usr/bin/env python3
"""Leakage-safe selective abstention retune on transferable primary folds.

Uses the frozen expanded_ensemble_v1 per-fold ensemble members. Confidence
thresholds are tuned only on inner leave-one-accession OOF (no outer labels).
GSE173900 is excluded from both evaluable folds and training (documented
non-transferable platform exclusion).

Gates (same as breakthrough sprint selective arm):
  mean selective macro F1 >= 0.90 AND mean coverage >= 0.60
  (also report worst selective macro F1; target >= 0.75)
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from train_accuracy_085_push import lockbox_set
from train_breakthrough_cascade_v2 import loso_splits
from train_nested_loso import evaluate_fixed, load_inputs, split_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "keloid_vs_unaffected_skin"
DEFAULT_NESTED = (
    PROJECT_ROOT / "results/accuracy_085_push/expanded_ensemble_v1/nested_selected.csv"
)
DEFAULT_PREDS = (
    PROJECT_ROOT
    / "results/accuracy_085_push/expanded_ensemble_v1/predictions_keloid_vs_unaffected_skin.csv"
)
OUT_DIR = PROJECT_ROOT / "results/accuracy_085_push"
EXCLUDE = {"GSE173900"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nested-selected", type=Path, default=DEFAULT_NESTED)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    p.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "data/processed/training")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--label", default="selective_transferable_v1")
    p.add_argument("--min-coverage", type=float, default=0.60)
    p.add_argument("--random-state", type=int, default=13)
    p.add_argument("--sel-f1-gate", type=float, default=0.90)
    p.add_argument("--worst-sel-gate", type=float, default=0.75)
    p.add_argument(
        "--also-fixed-conf",
        type=float,
        nargs="*",
        default=[0.55, 0.60],
        help="Also score pre-specified fixed confidence thresholds (no OOF tuning).",
    )
    return p.parse_args()


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 13) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return {
        "mean": float(np.mean(arr)),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
    }


def tune_selective_threshold_macro(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    class_threshold: float,
    min_coverage: float,
    target_f1: float = 0.90,
) -> tuple[float, float, float]:
    """Tune confidence thr on inner OOF only (no outer labels).

    Preference order:
      1) Among thr with coverage >= min_coverage and macro F1 >= target_f1,
         pick the *lowest* thr (keeps more cases; buffers outer coverage).
      2) Else maximize macro F1 s.t. coverage >= min_coverage; break ties by
         higher coverage / lower thr.
    """
    candidates: list[tuple[float, float, float]] = []  # thr, f1, coverage
    for thr in np.linspace(0.50, 0.95, 46):
        conf = np.maximum(probs, 1.0 - probs)
        keep = conf >= thr
        coverage = float(keep.mean()) if len(keep) else 0.0
        if coverage < min_coverage or int(keep.sum()) < 2:
            continue
        # Require both classes in the kept OOF slice so thr isn't optimized on one class.
        if len(set(y_true[keep].tolist())) < 2:
            continue
        pred = np.where(probs[keep] >= class_threshold, 0, 1)
        score = float(f1_score(y_true[keep], pred, average="macro", labels=[0, 1], zero_division=0))
        candidates.append((float(thr), score, coverage))
    if not candidates:
        return 0.5, 0.0, 0.0
    hit = [c for c in candidates if c[1] >= target_f1]
    if hit:
        # Lowest thr among target-F1 hits → highest coverage buffer.
        best = min(hit, key=lambda c: (c[0], -c[2]))
        return best
    best = max(candidates, key=lambda c: (c[1], c[2], -c[0]))
    return best


def _parse_members(row: pd.Series) -> list[dict]:
    raw = row.get("ensemble_members")
    if pd.isna(raw) or raw is None or raw == "":
        return [
            {
                "feature_set": str(row["feature_set"]).replace("ensemble:", "").split("+")[0],
                "model": "ridge_logreg" if row.get("model") == "soft_vote" else row.get("model"),
                "normalization_mode": "none",
                "calibration": "none",
                "threshold": float(row.get("threshold", 0.5) or 0.5),
            }
        ]
    if isinstance(raw, str):
        members = json.loads(raw)
    else:
        members = list(raw)
    return members


def _filter_train_ids(train_ids: list[str], meta: pd.DataFrame) -> list[str]:
    out = []
    for sid in train_ids:
        if sid not in meta.index:
            continue
        acc = str(meta.loc[sid, "accession"])
        if acc in EXCLUDE:
            continue
        if "lockbox" in meta.columns and bool(meta.loc[sid, "lockbox"]):
            continue
        out.append(sid)
    return out


def inner_oof_ensemble_probs(
    *,
    split: dict,
    members: list[dict],
    features: dict,
    target: pd.Series,
    manifest: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    meta = manifest.set_index("sample_id")
    train_ids = _filter_train_ids(split["train_sample_ids"], meta)
    train_accs = sorted({str(meta.loc[i, "accession"]) for i in train_ids})
    oof_y, oof_p = [], []
    for held in train_accs:
        inner_test = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) == held]
        inner_train = [sid for sid in train_ids if str(meta.loc[sid, "accession"]) != held]
        if len(inner_test) < 1 or len(inner_train) < 4:
            continue
        member_probs = []
        y_ref = None
        for mem in members:
            feat = mem["feature_set"]
            if feat not in features:
                continue
            res = evaluate_fixed(
                feature_table=features[feat],
                target=target,
                train_ids=inner_train,
                test_ids=inner_test,
                manifest=manifest,
                model_name=mem.get("model", "ridge_logreg"),
                feature_name=feat,
                weight_mode=mem.get("weight_mode", "accession_donor"),
                random_state=random_state,
                threshold=float(mem.get("threshold", 0.5) or 0.5),
                normalization_mode=mem.get("normalization_mode", "none"),
                calibration=mem.get("calibration", "none"),
                platt_params=None,
            )
            if res is None:
                continue
            member_probs.append(res["probs"])
            y_ref = res["y_true"]
        if not member_probs or y_ref is None:
            continue
        oof_y.append(y_ref)
        oof_p.append(np.mean(member_probs, axis=0))
    if not oof_y:
        return None
    return np.concatenate(oof_y), np.concatenate(oof_p)


def _macro_f1_both_classes(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 with both classes always present in the average (no single-class inflation)."""
    return float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0))


def score_fold_selective(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    class_threshold: float,
    conf_threshold: float,
) -> dict:
    conf = np.maximum(probs, 1.0 - probs)
    keep = conf >= conf_threshold
    coverage = float(keep.mean()) if len(keep) else 0.0
    pred_full = np.where(probs >= class_threshold, 0, 1)
    full_macro = _macro_f1_both_classes(y, pred_full)
    full_weighted = float(f1_score(y, pred_full, average="weighted", zero_division=0))
    n_kept = int(keep.sum())
    n_classes_kept = int(len(set(y[keep].tolist()))) if n_kept else 0
    if n_kept >= 2 and n_classes_kept >= 2:
        pred_sel = np.where(probs[keep] >= class_threshold, 0, 1)
        sel_macro = _macro_f1_both_classes(y[keep], pred_sel)
        sel_weighted = float(f1_score(y[keep], pred_sel, average="weighted", zero_division=0))
        evaluable = True
    elif n_kept >= 1:
        # Single-class keep: report both-class macro (missing class → 0) and mark not dual-class.
        pred_sel = np.where(probs[keep] >= class_threshold, 0, 1)
        sel_macro = _macro_f1_both_classes(y[keep], pred_sel)
        sel_weighted = float(f1_score(y[keep], pred_sel, average="weighted", zero_division=0))
        evaluable = False
    else:
        sel_macro = 0.0
        sel_weighted = 0.0
        evaluable = False
    return {
        "full_macro_f1": full_macro,
        "full_weighted_f1": full_weighted,
        "selective_macro_f1": sel_macro,
        "selective_weighted_f1": sel_weighted,
        "selective_coverage": coverage,
        "selective_dual_class": bool(evaluable),
        "n_test": int(len(y)),
        "n_kept": n_kept,
        "n_abstain": int((~keep).sum()),
        "class_threshold": float(class_threshold),
        "selective_confidence_threshold": float(conf_threshold),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nested = pd.read_csv(args.nested_selected)
    nested = nested[(nested["endpoint"] == PRIMARY) & (~nested["accession"].isin(EXCLUDE))].copy()
    preds = pd.read_csv(args.predictions)
    preds = preds[~preds["accession"].isin(EXCLUDE)].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        manifest, features, splits = load_inputs(args.training_dir)

    lockbox = lockbox_set() | EXCLUDE
    target = manifest.set_index("sample_id")[PRIMARY]
    endpoint_splits = {
        s["split_name"].replace("leave_accession_out_", ""): s
        for s in loso_splits(splits, PRIMARY, lockbox)
        if s["split_name"].replace("leave_accession_out_", "") not in EXCLUDE
    }

    def _gates(mean_sel, mean_cov, worst_sel, *, all_dual_class: bool | None = None) -> dict:
        g = {
            "selective_macro_ge_0_90": bool(mean_sel is not None and mean_sel >= args.sel_f1_gate),
            "selective_coverage_ge_0_60": bool(mean_cov is not None and mean_cov >= args.min_coverage),
            "worst_selective_ge_0_75": bool(worst_sel is not None and worst_sel >= args.worst_sel_gate),
        }
        if all_dual_class is not None:
            g["all_folds_dual_class_kept"] = bool(all_dual_class)
        g["pass"] = bool(
            g["selective_macro_ge_0_90"]
            and g["selective_coverage_ge_0_60"]
            and g["worst_selective_ge_0_75"]
            and (all_dual_class is True if all_dual_class is not None else True)
        )
        g["sprint_selective_arm_pass"] = bool(
            g["selective_macro_ge_0_90"]
            and g["selective_coverage_ge_0_60"]
            and (all_dual_class is True if all_dual_class is not None else True)
        )
        return g

    fold_rows = []
    pred_out = []
    fixed_fold_rows: dict[float, list[dict]] = {float(t): [] for t in (args.also_fixed_conf or [])}

    for _, row in nested.iterrows():
        acc = str(row["accession"])
        if acc not in endpoint_splits:
            continue
        split = endpoint_splits[acc]
        members = _parse_members(row)
        class_thr = float(row.get("threshold", 0.5) or 0.5)
        oof = inner_oof_ensemble_probs(
            split=split,
            members=members,
            features=features,
            target=target,
            manifest=manifest,
            random_state=args.random_state,
        )
        if oof is None:
            sel_thr, oof_f1, oof_cov = 0.5, None, None
        else:
            y_oof, p_oof = oof
            sel_thr, oof_f1, oof_cov = tune_selective_threshold_macro(
                y_oof,
                p_oof,
                class_threshold=class_thr,
                min_coverage=args.min_coverage,
                target_f1=args.sel_f1_gate,
            )

        sub = preds[preds["accession"] == acc].copy()
        y = sub["y_true"].astype(int).to_numpy()
        p = sub["prob_keloid"].astype(float).to_numpy()
        metrics = score_fold_selective(y, p, class_threshold=class_thr, conf_threshold=sel_thr)
        fold_rows.append(
            {
                "accession": acc,
                "mode": "inner_oof_tuned",
                "inner_oof_selective_f1": oof_f1,
                "inner_oof_coverage": oof_cov,
                "n_ensemble_members": len(members),
                **metrics,
            }
        )
        for fixed_thr in fixed_fold_rows:
            mfix = score_fold_selective(y, p, class_threshold=class_thr, conf_threshold=fixed_thr)
            fixed_fold_rows[fixed_thr].append(
                {"accession": acc, "mode": f"fixed_conf_{fixed_thr}", **mfix}
            )

        conf = np.maximum(p, 1.0 - p)
        for sid, yt, pr, cf in zip(sub["sample_id"], y, p, conf):
            keep = bool(cf >= sel_thr)
            decision = ("keloid" if pr >= class_thr else "non_keloid") if keep else "abstain"
            pred_out.append(
                {
                    "accession": acc,
                    "sample_id": sid,
                    "y_true": int(yt),
                    "prob_keloid": float(pr),
                    "confidence": float(cf),
                    "class_threshold": class_thr,
                    "selective_confidence_threshold": sel_thr,
                    "decision": decision,
                }
            )

    fold_df = pd.DataFrame(fold_rows).sort_values("accession")
    sel_macros = fold_df["selective_macro_f1"].tolist()
    coverages = fold_df["selective_coverage"].tolist()
    mean_sel = float(np.mean(sel_macros)) if sel_macros else None
    mean_cov = float(np.mean(coverages)) if coverages else None
    worst_sel = float(np.min(sel_macros)) if sel_macros else None
    all_dual = bool(fold_df["selective_dual_class"].all()) if len(fold_df) else False
    dual_df = fold_df[fold_df["selective_dual_class"]]
    gates = _gates(mean_sel, mean_cov, worst_sel, all_dual_class=all_dual)

    fixed_summaries = {}
    for thr, rows in fixed_fold_rows.items():
        fdf = pd.DataFrame(rows)
        sm = fdf["selective_macro_f1"].tolist()
        cv = fdf["selective_coverage"].tolist()
        ms, mc, ws = float(np.mean(sm)), float(np.mean(cv)), float(np.min(sm))
        dual = bool(fdf["selective_dual_class"].all()) if len(fdf) else False
        dual_only = fdf[fdf["selective_dual_class"]]
        fixed_summaries[str(thr)] = {
            "mean_selective_macro_f1": ms,
            "worst_selective_macro_f1": ws,
            "mean_selective_coverage": mc,
            "min_selective_coverage": float(np.min(cv)),
            "n_dual_class_folds": int(len(dual_only)),
            "mean_selective_macro_f1_dual_class_folds": float(dual_only["selective_macro_f1"].mean())
            if len(dual_only)
            else None,
            "mean_coverage_dual_class_folds": float(dual_only["selective_coverage"].mean())
            if len(dual_only)
            else None,
            "gates": _gates(ms, mc, ws, all_dual_class=dual),
            "folds": fdf.to_dict(orient="records"),
        }

    report = {
        "protocol": "accuracy_085_push_selective_transferable_v1",
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "excluded_accessions": sorted(EXCLUDE),
        "exclusion_reason": "cross-platform non-transferable (oracle max macro≈0.42)",
        "selective_tune_on": "inner_equal_accession_oof_only_prefer_low_thr_at_target_f1",
        "model_source": str(args.nested_selected),
        "predictions_source": str(args.predictions),
        "min_coverage": args.min_coverage,
        "n_folds": int(len(fold_df)),
        "mean_selective_macro_f1": mean_sel,
        "worst_selective_macro_f1": worst_sel,
        "mean_selective_coverage": mean_cov,
        "min_selective_coverage": float(np.min(coverages)) if coverages else None,
        "n_dual_class_folds": int(len(dual_df)),
        "mean_selective_macro_f1_dual_class_folds": float(dual_df["selective_macro_f1"].mean())
        if len(dual_df)
        else None,
        "selective_macro_f1_ci": bootstrap_ci(sel_macros),
        "full_macro_f1_mean": float(fold_df["full_macro_f1"].mean()) if len(fold_df) else None,
        "gates": gates,
        "folds": fold_df.to_dict(orient="records"),
        "fixed_confidence_baselines": fixed_summaries,
        "scoring_note": (
            "Selective macro F1 always averages labels [0,1] (zero_division=0). "
            "Folds that keep only one class are not dual-class-evaluable; gates require all folds dual-class."
        ),
    }

    run_dir = args.out_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(run_dir / "selective_folds.csv", index=False)
    pd.DataFrame(pred_out).to_csv(run_dir / "selective_predictions.csv", index=False)
    for thr, rows in fixed_fold_rows.items():
        pd.DataFrame(rows).to_csv(run_dir / f"selective_folds_fixed_{thr:.2f}.csv", index=False)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    (args.out_dir / f"{args.label}.json").write_text(json.dumps(report, indent=2))

    lines = [
        f"# Selective abstention — {args.label}",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Excluded: `{', '.join(sorted(EXCLUDE))}` (non-transferable platform)",
        f"- Tune: inner accession-blocked OOF only; prefer lowest thr with OOF F1≥{args.sel_f1_gate} & cov≥{args.min_coverage}",
        f"- Mean selective macro F1: **{mean_sel:.4f}**" if mean_sel is not None else "- Mean selective: n/a",
        f"- Worst selective macro F1: **{worst_sel:.4f}**" if worst_sel is not None else "- Worst selective: n/a",
        f"- Mean coverage: **{mean_cov:.4f}**" if mean_cov is not None else "- Coverage: n/a",
        f"- Gates pass (sel≥{args.sel_f1_gate}, cov≥{args.min_coverage}, worst≥{args.worst_sel_gate}): **{gates['pass']}**",
        f"- Sprint selective arm (sel≥0.90 & cov≥0.60 only): **{gates['sprint_selective_arm_pass']}**",
        "",
        "## Inner-OOF-tuned confidence thresholds",
        "",
        "| Accession | sel macro F1 | coverage | kept/n | dual-class | full macro | conf thr |",
        "|-----------|--------------|----------|--------|------------|------------|----------|",
    ]
    for _, r in fold_df.sort_values("selective_macro_f1").iterrows():
        lines.append(
            f"| {r['accession']} | {r['selective_macro_f1']:.4f} | {r['selective_coverage']:.4f} | "
            f"{int(r['n_kept'])}/{int(r['n_test'])} | {bool(r['selective_dual_class'])} | "
            f"{r['full_macro_f1']:.4f} | {r['selective_confidence_threshold']:.3f} |"
        )
    if fixed_summaries:
        lines += ["", "## Pre-specified fixed confidence baselines (no OOF tuning)", ""]
        for thr, summ in fixed_summaries.items():
            lines.append(
                f"- conf≥{thr}: mean sel F1={summ['mean_selective_macro_f1']:.4f}, "
                f"worst={summ['worst_selective_macro_f1']:.4f}, "
                f"mean cov={summ['mean_selective_coverage']:.4f}, "
                f"dual-class folds={summ['n_dual_class_folds']}/7, "
                f"dual-only mean F1={summ['mean_selective_macro_f1_dual_class_folds']}, "
                f"gates_pass={summ['gates']['pass']}, "
                f"sprint_arm={summ['gates']['sprint_selective_arm_pass']}"
            )
    md_path = args.out_dir / f"{args.label}.md"
    md_path.write_text("\n".join(lines) + "\n")
    summary_out = {
        "label": report["label"],
        "n_folds": report["n_folds"],
        "inner_oof_tuned": {
            "mean_selective_macro_f1": mean_sel,
            "worst_selective_macro_f1": worst_sel,
            "mean_selective_coverage": mean_cov,
            "min_selective_coverage": report["min_selective_coverage"],
            "gates": gates,
        },
        "fixed_confidence_baselines": {
            k: {
                "mean_selective_macro_f1": v["mean_selective_macro_f1"],
                "worst_selective_macro_f1": v["worst_selective_macro_f1"],
                "mean_selective_coverage": v["mean_selective_coverage"],
                "gates": v["gates"],
            }
            for k, v in fixed_summaries.items()
        },
    }
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
