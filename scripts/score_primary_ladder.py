#!/usr/bin/env python3
"""Pre-registered scoreboard for keloid_vs_unaffected_skin >=0.85 push.

Primary metric (frozen):
  - Donor-grain macro F1 under leave-one-accession-out (LOSO).
  - Single-donor accessions: sample-grain macro F1 (donor-grain is trivial).
  - Success gate: mean macro F1 >= 0.85 AND every evaluable fold >= 0.75.

Never moves goalposts; reads nested_selected.csv / predictions and emits a
JSON + Markdown scoreboard.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "keloid_vs_unaffected_skin"
OUT_DIR = PROJECT_ROOT / "results/accuracy_085_push"
STAGE_B_NESTED = (
    PROJECT_ROOT
    / "results/breakthrough_sprint_v2/ablations/stage_b_donor_selection_calibration/nested_selected.csv"
)
STAGE_B_PREDS = (
    PROJECT_ROOT
    / "results/breakthrough_sprint_v2/ablations/stage_b_donor_selection_calibration"
    / "predictions_keloid_vs_unaffected_skin.csv"
)
TRAINING_MANIFEST = PROJECT_ROOT / "data/processed/training/profile_manifest.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nested-selected", type=Path, default=STAGE_B_NESTED)
    p.add_argument("--predictions", type=Path, default=STAGE_B_PREDS)
    p.add_argument("--manifest", type=Path, default=TRAINING_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--label", default="baseline")
    p.add_argument("--mean-gate", type=float, default=0.85)
    p.add_argument("--worst-gate", type=float, default=0.75)
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


def _n_donors_from_manifest(manifest: pd.DataFrame, accession: str) -> int:
    sub = manifest[manifest["accession"].astype(str) == str(accession)]
    if "split_group" in sub.columns:
        return int(sub["split_group"].astype(str).nunique())
    if "patient_id" in sub.columns:
        return int(sub["patient_id"].astype(str).nunique())
    return int(len(sub))


def _donors_are_class_pure(manifest: pd.DataFrame, accession: str) -> bool:
    """False when any donor/group carries both keloid and non_keloid labels.

    Paired lesion vs non-lesional designs (e.g. GSE92566 / GSE158395) are
    sample-labeled contrasts; collapsing to donor majority is undefined.
    """
    sub = manifest[manifest["accession"].astype(str) == str(accession)].copy()
    if "keloid_vs_unaffected_skin" in sub.columns:
        sub = sub[sub["keloid_vs_unaffected_skin"].isin(["keloid", "non_keloid"])]
    if sub.empty:
        return True
    gcol = "split_group" if "split_group" in sub.columns else "patient_id"
    label_col = (
        "keloid_vs_unaffected_skin"
        if "keloid_vs_unaffected_skin" in sub.columns
        else None
    )
    if label_col is None:
        return True
    for _, g in sub.groupby(gcol):
        if g[label_col].nunique() > 1:
            return False
    return True


def fold_score_from_nested_row(
    row: pd.Series,
    n_donors: int,
    *,
    class_pure_donors: bool = True,
) -> dict:
    """Donor-grain when multi-donor and class-pure; else sample-grain."""
    sample_macro = float(row["macro_f1"]) if pd.notna(row.get("macro_f1")) else None
    donor_macro = float(row["donor_macro_f1"]) if pd.notna(row.get("donor_macro_f1")) else None
    sample_weighted = float(row["weighted_f1"]) if pd.notna(row.get("weighted_f1")) else None
    donor_weighted = float(row["donor_weighted_f1"]) if pd.notna(row.get("donor_weighted_f1")) else None
    use_sample = (n_donors <= 1) or (not class_pure_donors)
    primary = sample_macro if use_sample else (donor_macro if donor_macro is not None else sample_macro)
    primary_w = sample_weighted if use_sample else (donor_weighted if donor_weighted is not None else sample_weighted)
    if n_donors <= 1:
        grain = "sample"
    elif not class_pure_donors:
        grain = "sample_mixed_donor"
    else:
        grain = "donor"
    return {
        "accession": str(row["accession"]),
        "n_donors": int(n_donors),
        "grain": grain,
        "macro_f1": primary,
        "weighted_f1": primary_w,
        "sample_macro_f1": sample_macro,
        "donor_macro_f1": donor_macro,
        "n_test": int(row["n_test"]) if pd.notna(row.get("n_test")) else None,
        "feature_set": row.get("feature_set"),
        "model": row.get("model"),
        "normalization_mode": row.get("normalization_mode"),
        "calibration": row.get("calibration"),
        "threshold": float(row["threshold"]) if pd.notna(row.get("threshold")) else None,
    }


def score_from_predictions(
    preds: pd.DataFrame,
    manifest: pd.DataFrame,
    threshold_by_acc: dict[str, float] | None = None,
) -> list[dict]:
    """Recompute donor/sample-grain macro F1 from prediction rows."""
    meta = manifest.set_index("sample_id")
    folds = []
    for acc, sub in preds.groupby(preds["accession"].astype(str)):
        thr = 0.5
        if threshold_by_acc and acc in threshold_by_acc:
            thr = float(threshold_by_acc[acc])
        elif "threshold" in sub.columns and sub["threshold"].notna().any():
            thr = float(sub["threshold"].iloc[0])
        y = sub["y_true"].astype(int).to_numpy()
        p = sub["prob_keloid"].astype(float).to_numpy()
        pred = (p >= thr).astype(int)
        sample_macro = float(f1_score(y, pred, average="macro", zero_division=0))
        # Donor grain
        if "donor" in sub.columns:
            donors = sub["donor"].astype(str)
        else:
            sids = sub["sample_id"].astype(str)
            donors = pd.Series(
                [
                    str(meta.loc[sid, "split_group"]) if sid in meta.index else sid
                    for sid in sids
                ],
                index=sub.index,
            )
        donor_true, donor_pred = [], []
        for _, idx in donors.groupby(donors).groups.items():
            yi = y[[sub.index.get_loc(i) for i in idx]]
            pi = p[[sub.index.get_loc(i) for i in idx]]
            donor_true.append(int(np.round(yi.mean())))
            donor_pred.append(int(np.mean(pi) >= thr))
        n_donors = len(donor_true)
        donor_macro = (
            float(f1_score(donor_true, donor_pred, average="macro", zero_division=0))
            if n_donors >= 1
            else None
        )
        single = n_donors <= 1
        # Detect mixed-class donors from prediction truths when possible.
        class_pure = True
        for _, idx in donors.groupby(donors).groups.items():
            yi = y[[sub.index.get_loc(i) for i in idx]]
            if len(set(yi.tolist())) > 1:
                class_pure = False
                break
        use_sample = single or (not class_pure)
        if single:
            grain = "sample"
        elif not class_pure:
            grain = "sample_mixed_donor"
        else:
            grain = "donor"
        folds.append(
            {
                "accession": acc,
                "n_donors": n_donors,
                "grain": grain,
                "macro_f1": sample_macro if use_sample else donor_macro,
                "sample_macro_f1": sample_macro,
                "donor_macro_f1": donor_macro,
                "n_test": int(len(sub)),
                "threshold": thr,
            }
        )
    return folds


def build_scoreboard(
    folds: list[dict],
    *,
    label: str,
    mean_gate: float,
    worst_gate: float,
    source: dict,
) -> dict:
    macros = [f["macro_f1"] for f in folds if f.get("macro_f1") is not None]
    ci = bootstrap_ci(macros)
    worst = float(min(macros)) if macros else None
    mean = float(np.mean(macros)) if macros else None
    gates = {
        "mean_macro_ge_0_85": bool(mean is not None and mean >= mean_gate),
        "worst_ge_0_75": bool(worst is not None and worst >= worst_gate),
        "pass": bool(
            mean is not None
            and worst is not None
            and mean >= mean_gate
            and worst >= worst_gate
        ),
    }
    return {
        "protocol": "accuracy_085_push_v1",
        "endpoint": PRIMARY,
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_definition": {
            "grain": "donor_primary; sample_fallback for single-donor OR mixed-class within-donor contrasts",
            "aggregation": "macro_f1_mean_across_LOSO_folds",
            "mean_gate": mean_gate,
            "worst_gate": worst_gate,
            "full_coverage": True,
        },
        "source": source,
        "n_folds": len(folds),
        "folds": sorted(folds, key=lambda r: (r.get("macro_f1") is None, r.get("macro_f1", 0))),
        "mean_macro_f1": mean,
        "worst_fold_macro_f1": worst,
        "bootstrap_ci": ci,
        "gates": gates,
    }


def write_markdown(board: dict, path: Path) -> None:
    lines = [
        f"# Accuracy 0.85 push — {board['label']}",
        "",
        f"- Generated: `{board['generated_at']}`",
        f"- Endpoint: `{board['endpoint']}`",
        f"- Mean macro F1: **{board['mean_macro_f1']:.4f}**" if board["mean_macro_f1"] is not None else "- Mean: n/a",
        f"- Worst fold: **{board['worst_fold_macro_f1']:.4f}**" if board["worst_fold_macro_f1"] is not None else "- Worst: n/a",
        f"- Gates pass: **{board['gates']['pass']}** (mean≥{board['metric_definition']['mean_gate']}, worst≥{board['metric_definition']['worst_gate']})",
        "",
        "| Accession | Grain | n_donors | macro F1 | sample | donor |",
        "|-----------|-------|----------|----------|--------|-------|",
    ]
    for f in board["folds"]:
        lines.append(
            f"| {f['accession']} | {f['grain']} | {f.get('n_donors')} | "
            f"{f.get('macro_f1'):.4f} | "
            f"{(f.get('sample_macro_f1') if f.get('sample_macro_f1') is not None else float('nan')):.4f} | "
            f"{(f.get('donor_macro_f1') if f.get('donor_macro_f1') is not None else float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nested = pd.read_csv(args.nested_selected)
    primary = nested[nested["endpoint"] == PRIMARY].copy()
    if primary.empty:
        raise SystemExit(f"No {PRIMARY} rows in {args.nested_selected}")
    manifest = pd.read_parquet(args.manifest) if args.manifest.exists() else pd.DataFrame()

    folds = []
    for _, row in primary.iterrows():
        acc = str(row["accession"])
        n_donors = int(row["n_test_donors"]) if pd.notna(row.get("n_test_donors")) else 1
        if len(manifest) and "accession" in manifest.columns:
            n_donors = max(n_donors, _n_donors_from_manifest(manifest, acc))
        # Prefer nested's reported n_test_donors for the fold definition.
        n_donors = int(row["n_test_donors"]) if pd.notna(row.get("n_test_donors")) else n_donors
        pure = True
        if len(manifest) and "accession" in manifest.columns:
            pure = _donors_are_class_pure(manifest, acc)
        folds.append(fold_score_from_nested_row(row, n_donors, class_pure_donors=pure))

    # Optionally cross-check with predictions.
    pred_check = None
    if args.predictions.exists():
        preds = pd.read_csv(args.predictions)
        thr_map = {str(r["accession"]): float(r["threshold"]) for _, r in primary.iterrows()}
        pred_check = score_from_predictions(preds, manifest, thr_map)

    board = build_scoreboard(
        folds,
        label=args.label,
        mean_gate=args.mean_gate,
        worst_gate=args.worst_gate,
        source={
            "nested_selected": str(args.nested_selected),
            "predictions": str(args.predictions) if args.predictions.exists() else None,
            "predictions_crosscheck": pred_check,
        },
    )
    json_path = args.out_dir / f"{args.label}.json"
    md_path = args.out_dir / f"{args.label}.md"
    json_path.write_text(json.dumps(board, indent=2))
    write_markdown(board, md_path)
    print(
        json.dumps(
            {
                "written": str(json_path),
                "mean": board["mean_macro_f1"],
                "worst": board["worst_fold_macro_f1"],
                "gates": board["gates"],
                "n_folds": board["n_folds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
