#!/usr/bin/env python3
"""Export the frozen KeloidBench demo as browser-native JSON for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from demo_utils import (  # noqa: E402
    CURATED_EXAMPLES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    NON_TRANSFERABLE_ACCESSIONS,
    PRIMARY_ENDPOINT,
    PROGRAM_LABELS,
    SOURCE_RECORDS,
    TRANSFERABLE_ACCESSIONS,
    load_feature_views,
    load_loso_evidence,
    load_product_bundle,
    load_profile_manifest,
    program_score_table,
    score_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "website/assets/data/demo-data.json",
    )
    return parser.parse_args()


def clean_value(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return [
        {str(key): clean_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def main() -> None:
    args = parse_args()
    manifest = load_profile_manifest().set_index("sample_id")
    views = load_feature_views()
    _, bundle = load_product_bundle(PRIMARY_ENDPOINT)
    fused = views["fused_multiview"].copy()
    live = views[bundle["meta"]["feature_set"]].copy()

    eligible = manifest[
        manifest[PRIMARY_ENDPOINT].isin(["keloid", "non_keloid"])
        & manifest["accession"].isin(TRANSFERABLE_ACCESSIONS | NON_TRANSFERABLE_ACCESSIONS)
    ]
    sample_ids = [sample_id for sample_id in eligible.index if sample_id in fused.index]

    matrix = fused.loc[sample_ids].astype(float)
    coordinates = PCA(n_components=2, random_state=13).fit_transform(
        StandardScaler().fit_transform(matrix)
    )
    pca = []
    for sample_id, coordinate in zip(sample_ids, coordinates, strict=True):
        row = eligible.loc[sample_id]
        pca.append(
            {
                "sample_id": str(sample_id),
                "pc1": float(coordinate[0]),
                "pc2": float(coordinate[1]),
                "reference_label": str(row[PRIMARY_ENDPOINT]),
                "accession": str(row["accession"]),
                "transfer_group": (
                    "non-transferable platform"
                    if str(row["accession"]) in NON_TRANSFERABLE_ACCESSIONS
                    else "transferable subset"
                ),
            }
        )

    reference_table = matrix.copy()
    reference_table["reference_label"] = eligible.loc[sample_ids, PRIMARY_ENDPOINT].to_numpy()
    references = reference_table.groupby("reference_label").median(numeric_only=True)

    slug_by_sample = {
        item["sample_id"]: slug for slug, item in CURATED_EXAMPLES.items()
    }
    profiles = []
    for sample_id in sample_ids:
        row = eligible.loc[sample_id]
        scored = score_sample(
            live.loc[[sample_id]],
            bundle,
            is_expression=False,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        program_rows = program_score_table(fused.loc[[sample_id]])
        program_records = []
        for _, program in program_rows.iterrows():
            name = str(program["program"])
            program_records.append(
                {
                    "program": name,
                    "display_name": str(program["display_name"]),
                    "score": float(program["score"]),
                    "direction": str(program["direction"]),
                    "keloid_median": float(references.loc["keloid", name]),
                    "unaffected_median": float(references.loc["non_keloid", name]),
                }
            )
        profiles.append(
            {
                "sample_id": str(sample_id),
                "accession": str(row["accession"]),
                "reference_label": str(row[PRIMARY_ENDPOINT]),
                "modality": str(row.get("modality", "unknown")),
                "platform_id": str(row.get("platform_id", "unknown")),
                "patient_id": str(row.get("patient_id", "unknown")),
                "curated_slug": slug_by_sample.get(str(sample_id)),
                "prob_keloid": float(scored["prob_keloid"]),
                "confidence": float(scored["confidence"]),
                "pred_label": str(scored["pred_label"]),
                "decision": str(scored["decision"]),
                "decision_threshold": float(scored["decision_threshold"]),
                "confidence_threshold": float(scored["confidence_threshold"]),
                "programs": program_records,
                "contributions": [
                    {key: clean_value(value) for key, value in contribution.items()}
                    for contribution in scored["contributions"]
                ],
            }
        )

    predictions, selected, transferable_report, selective_report = load_loso_evidence()
    folds = {}
    for accession, frame in predictions.groupby("accession", sort=True):
        fold_row = selected[selected["accession"] == accession].iloc[0]
        folds[str(accession)] = {
            "accession": str(accession),
            "transfer_group": (
                "non-transferable platform"
                if str(accession) in NON_TRANSFERABLE_ACCESSIONS
                else "transferable subset"
            ),
            "threshold": float(fold_row["threshold"]),
            "macro_f1": float(fold_row["macro_f1"]),
            "accuracy": float(fold_row["accuracy"]),
            "coverage_055": float((frame["confidence"] >= 0.55).mean()),
            "predictions": records(
                frame[
                    [
                        "sample_id",
                        "true_label",
                        "prob_keloid",
                        "pred_label",
                        "confidence",
                        "selective_decision",
                        "correct",
                    ]
                ].sort_values("prob_keloid")
            ),
        }

    fixed = selective_report["fixed_confidence_baselines"]["0.55"]
    payload = {
        "generated_from": "versioned repository artifacts",
        "endpoint": PRIMARY_ENDPOINT,
        "headline": {
            "transferable_studies": 7,
            "profiles": 92,
            "donors": 39,
            "loso_macro_f1": float(transferable_report["mean_macro_f1"]),
            "pooled_accuracy": 77 / 92,
            "pooled_accuracy_count": "77/92",
            "selective_macro_f1": float(fixed["mean_selective_macro_f1"]),
            "selective_coverage": float(fixed["mean_selective_coverage"]),
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        },
        "artifact": {
            "stage": "stage_b_donor_selection_calibration",
            "model": str(bundle["meta"]["config"]["model"]),
            "feature_set": str(bundle["meta"]["feature_set"]),
            "class_threshold": float(bundle["meta"]["config"]["threshold"]),
            "feature_columns": list(bundle["meta"]["feature_columns"]),
        },
        "curated_examples": CURATED_EXAMPLES,
        "program_labels": PROGRAM_LABELS,
        "sources": SOURCE_RECORDS,
        "profiles": profiles,
        "pca": pca,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} ({len(profiles)} profiles, {len(folds)} folds)")


if __name__ == "__main__":
    main()
