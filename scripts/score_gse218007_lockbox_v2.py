#!/usr/bin/env python3
"""One-shot score of v2 lockbox GSE218007 (fibroblast route). Never enters selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from expression_processing import write_expression_artifacts
from preprocess_public_keloid import download_cohort, load_registry, process_cohort
from product_features import build_product_feature_views
from train_nested_loso import positive_proba

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DIR = PROJECT_ROOT / "results/product_friday"
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint_v2/lockbox_gse218007"
RAW_DIR = PROJECT_ROOT / "data/raw/public_keloid"
ENDPOINT = "fibroblast_keloid_binary"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product-dir", type=Path, default=PRODUCT_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--registry", type=Path, default=PROJECT_ROOT / "data/raw/public_keloid_cohorts.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    marker = args.out_dir / "scored_once.json"
    if marker.exists():
        raise SystemExit(f"Already scored once: {marker}. Refusing to rescore.")

    registry = load_registry(args.registry)
    entries = [e for e in registry.get("breakthrough_lockbox", []) if e["accession"] == "GSE218007"]
    if not entries:
        # Fallback: construct from known URL.
        entries = [
            {
                "accession": "GSE218007",
                "role": "lockbox",
                "modality": "microarray",
                "platform_id": "GPL23126",
                "download": {
                    "series_matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE218nnn/GSE218007/matrix/GSE218007_series_matrix.txt.gz"
                },
            }
        ]
    entry = entries[0]
    paths = download_cohort(entry, RAW_DIR, force=False, include_reserved_expression=True)
    result = process_cohort(entry, paths)
    if result is None or result[0] is None:
        raise SystemExit(f"GSE218007 process failed: {result}")
    meta, expr, summary = result
    # Persist processed lockbox artifacts separately (not merged into train).
    write_expression_artifacts(
        out_dir=args.out_dir / "processed",
        prefix="gse218007_lockbox",
        metadata=meta,
        expr=expr,
        dataset_summaries=[summary],
    )

    product = json.loads((args.product_dir / "product_manifest.json").read_text())
    if ENDPOINT not in product["endpoints"]:
        raise SystemExit(f"Product missing {ENDPOINT}")
    bundle = joblib.load(PROJECT_ROOT / product["endpoints"][ENDPOINT]["model_path"])
    model = bundle["model"]
    mmeta = bundle["meta"]

    # Align expression rows with metadata sample order.
    expr = expr.copy()
    expr.index = meta["sample_id"].astype(str).tolist()
    views = build_product_feature_views(expr)
    feat_name = mmeta["feature_set"]
    feats = views[feat_name] if feat_name in views else views["fibrosis_only"]
    xs = pd.DataFrame(index=feats.index)
    for col in mmeta["feature_columns"]:
        if col in feats.columns:
            xs[col] = feats[col]
        else:
            xs[col] = float(mmeta["feature_medians"].get(col, 0.0))
    xs = xs.fillna(0.0)
    probs = positive_proba(model, xs, mmeta["positive_code"])
    thr = float(mmeta["config"].get("threshold", 0.5))
    pred = np.where(probs >= thr, "keloid", "non_keloid")
    conf = np.maximum(probs, 1 - probs)
    y_true = meta.set_index("sample_id").loc[xs.index, "keloid_vs_normal"].astype(str)
    y_true = y_true.map(lambda v: "keloid" if v == "keloid" else "non_keloid")
    y_bin = (y_true == "keloid").astype(int).to_numpy()
    pred_bin = (pred == "keloid").astype(int)
    metrics = {
        "n": int(len(xs)),
        "n_donors": int(meta["patient_id"].nunique()),
        "weighted_f1": float(f1_score(y_bin, pred_bin, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_bin, pred_bin, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_bin, probs)) if len(set(y_bin)) == 2 else None,
        "threshold": thr,
        "endpoint": ENDPOINT,
        "note": "Fibroblast culture lockbox; not a tissue-primary claim.",
    }
    # Donor-grain
    donor = meta.set_index("sample_id").loc[xs.index, "patient_id"].astype(str)
    donor_true, donor_prob = [], []
    for d, idx in donor.groupby(donor).groups.items():
        donor_true.append(int(np.round(y_bin[[xs.index.get_loc(i) for i in idx]].mean())))
        donor_prob.append(float(np.mean(probs[[xs.index.get_loc(i) for i in idx]])))
    if len(set(donor_true)) == 2:
        metrics["donor_macro_f1"] = float(
            f1_score(donor_true, (np.asarray(donor_prob) >= thr).astype(int), average="macro", zero_division=0)
        )
        metrics["donor_auroc"] = float(roc_auc_score(donor_true, donor_prob))
    else:
        metrics["donor_macro_f1"] = None
        metrics["donor_auroc"] = None

    pred_df = pd.DataFrame(
        {
            "sample_id": xs.index,
            "patient_id": donor.values,
            "y_true": y_true.values,
            "prob_keloid": probs,
            "pred_label": pred,
            "confidence": conf,
        }
    )
    pred_df.to_csv(args.out_dir / "predictions.csv", index=False)
    payload = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "accession": "GSE218007",
        "protocol": "breakthrough_sprint_v2_lockbox_once",
        "summary": summary,
        "metrics": metrics,
        "do_not_rescore": True,
    }
    marker.write_text(json.dumps(payload, indent=2))
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# GSE218007 v2 lockbox (scored once)",
                "",
                f"- Endpoint: `{ENDPOINT}` (fibroblast route)",
                f"- Macro F1: **{metrics['macro_f1']:.3f}** (n={metrics['n']})",
                f"- Donor macro F1: {metrics['donor_macro_f1']}",
                "- Immutable after this write; do not rescore.",
                "",
            ]
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
