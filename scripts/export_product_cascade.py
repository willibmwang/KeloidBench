#!/usr/bin/env python3
"""Export a shippable Stage-B cascade product (joblib weights + claim card).

Fits frozen Stage-B endpoint configs on all eligible training profiles.
Does not touch lockboxes or reopen architecture search.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_nested_loso import (
    BINARY_CLASSES,
    encode_binary,
    fit_model,
    load_inputs,
    model_specs,
    positive_proba,
    donor_accession_weights,
)
from train_breakthrough_cascade import PRIMARY, SPECIALISTS, SENSITIVITY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_B_REPORT = (
    PROJECT_ROOT
    / "results/breakthrough_sprint_v2/ablations/stage_b_donor_selection_calibration/report.json"
)
OUT_DIR = PROJECT_ROOT / "results/product_friday"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "data/processed/training")
    p.add_argument("--stage-b-report", type=Path, default=STAGE_B_REPORT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--random-state", type=int, default=13)
    return p.parse_args()


def _eligible_ids(manifest: pd.DataFrame, endpoint: str) -> list[str]:
    y = manifest.set_index("sample_id")[endpoint].astype(str)
    keep = y.isin(BINARY_CLASSES)
    # Never train on lockbox / reserved accessions.
    roles = manifest.set_index("sample_id").get("cohort_role")
    if roles is not None:
        keep &= ~roles.astype(str).str.contains("lockbox", case=False, na=False)
    acc = manifest.set_index("sample_id")["accession"].astype(str)
    keep &= ~acc.isin({"GSE185309", "GSE212954", "GSE218007", "GSE237752", "GSE191067"})
    return y.index[keep].tolist()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.stage_b_report.read_text())
    endpoints = report["endpoints"]
    manifest, features, _ = load_inputs(args.training_dir)

    product = {
        "product": "SpheroScar_molecular_triage_friday",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_stage": "stage_b_donor_selection_calibration",
        "claim_gates_pass": False,
        "honest_ceiling": {
            "primary_macro_f1": endpoints[PRIMARY]["macro_accession_f1"]["mean"],
            "primary_worst_fold": endpoints[PRIMARY]["worst_fold"],
            "decision": "EXECUTE_MATCHED_COHORT_PROTOCOL",
        },
        "endpoints": {},
        "selective_confidence_threshold_default": 0.60,
    }

    models_dir = args.out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for endpoint in [PRIMARY, *SPECIALISTS, *SENSITIVITY]:
        if endpoint not in endpoints or endpoint not in manifest.columns:
            continue
        cfg = endpoints[endpoint]["frozen_config"]
        feat_name = cfg["feature_set"]
        if feat_name not in features:
            print(f"skip {endpoint}: missing feature view {feat_name}")
            continue
        ids = _eligible_ids(manifest, endpoint)
        x = features[feat_name].loc[[i for i in ids if i in features[feat_name].index]]
        y_raw = manifest.set_index("sample_id").loc[x.index, endpoint]
        if y_raw.nunique() < 2 or len(x) < 8:
            print(f"skip {endpoint}: insufficient labeled rows ({len(x)})")
            continue
        labels, y, _, positive_code = encode_binary(y_raw, y_raw)
        weights = None
        if cfg.get("weight_mode") in {"donor", "accession_donor"}:
            weights = donor_accession_weights(x.index.tolist(), manifest)
        model = model_specs(args.random_state)[cfg["model"]]
        fitted = fit_model(model, x, y, sample_weight=weights)
        # Store train feature medians for missing-column imputation at score time.
        artifact = {
            "endpoint": endpoint,
            "config": cfg,
            "feature_set": feat_name,
            "feature_columns": x.columns.tolist(),
            "feature_medians": x.median(axis=0).fillna(0.0).to_dict(),
            "labels": labels,
            "positive_label": "keloid",
            "positive_code": int(positive_code),
            "n_train": int(len(x)),
            "train_accessions": sorted(
                manifest.set_index("sample_id").loc[x.index, "accession"].astype(str).unique().tolist()
            ),
        }
        joblib.dump({"model": fitted, "meta": artifact}, models_dir / f"{endpoint}.joblib")
        product["endpoints"][endpoint] = {
            **artifact,
            "model_path": str((models_dir / f"{endpoint}.joblib").relative_to(PROJECT_ROOT)),
            "nested_macro_f1": endpoints[endpoint]["macro_accession_f1"]["mean"],
            "nested_worst_fold": endpoints[endpoint]["worst_fold"],
        }
        print(json.dumps({"exported": endpoint, "n_train": artifact["n_train"], "features": feat_name}, indent=2))

    (args.out_dir / "product_manifest.json").write_text(json.dumps(product, indent=2))
    claim = args.out_dir / "CLAIM_CARD.md"
    claim.write_text(
        "\n".join(
            [
                "# SpheroScar Friday product — claim card",
                "",
                f"- Built: `{product['frozen_at']}`",
                "- Source: breakthrough v2 **Stage B** (donor selection + calibration)",
                f"- Primary nested macro F1: **{product['honest_ceiling']['primary_macro_f1']:.3f}**",
                f"- Primary worst fold: **{product['honest_ceiling']['primary_worst_fold']:.3f}**",
                "- Claim gates (full≥0.85 or selective≥0.90@≥60%, worst≥0.75): **FAIL**",
                "- Decision: **EXECUTE_MATCHED_COHORT_PROTOCOL**",
                "",
                "## What this product is",
                "Molecular triage CLI: primary keloid-vs-unaffected-skin + scar specialists + fibroblast route,",
                "with selective abstention. Honest research prototype — not a clinical diagnostic.",
                "",
                "## What this product is not",
                "- Not a validated ≥0.85 accuracy claim",
                "- Does not rescore v1 lockbox GSE185309",
                "- Does not mix culture/stiffness into tissue primary",
                "",
                "## Run",
                "```bash",
                "python scripts/score_cascade_product.py --features-parquet <view.parquet> --sample-ids ...",
                "# or from HGNC expression wide CSV:",
                "python scripts/score_cascade_product.py --expression-csv genes_x_samples.csv --transpose",
                "```",
                "",
                "## Endpoints shipped",
                *[f"- `{ep}`: nested macro F1={meta['nested_macro_f1']:.3f}, n_train={meta['n_train']}" for ep, meta in product["endpoints"].items()],
                "",
            ]
        )
    )
    print(json.dumps({"written": str(args.out_dir), "n_endpoints": len(product["endpoints"])}, indent=2))


if __name__ == "__main__":
    main()
