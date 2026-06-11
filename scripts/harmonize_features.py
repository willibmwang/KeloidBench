#!/usr/bin/env python3
"""Harmonize Dataset A/B/C into integrated MVP feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed"
SPHEROID_PROCESSED = PROCESSED / "spheroid"
MICROARRAY_SIGNATURES = PROCESSED / "microarray/signatures"
RESULTS = PROJECT_ROOT / "results"

MODULES = {
    "ECM_score": ["COL1A1", "COL3A1"],
    "TGFb_score": ["TGFB1", "TGFB3"],
    "hypoxia_vascular_score": ["HIF1A"],
    "remodeling_score": ["MMP14", "ADAM12", "HTRA1", "CTHRC1"],
}


def add_choi_module_scores(a: pd.DataFrame) -> pd.DataFrame:
    out = a.copy()
    for score_name, genes in MODULES.items():
        available = [gene for gene in genes if gene in out.columns]
        out[f"choi_{score_name}"] = out[available].mean(axis=1) if available else np.nan
    module_cols = [f"choi_{name}" for name in MODULES]
    out["choi_qpcr_activity_score"] = out[module_cols].mean(axis=1)
    return out


def add_public_signature_context(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    out = a.copy()
    score_cols = [
        "ECM_score",
        "myofibroblast_score",
        "TGFb_score",
        "hypoxia_vascular_score",
        "remodeling_score",
        "keloid_activity_score",
    ]
    keloid_ref = c[c["keloid_vs_normal"] == "keloid"] if "keloid_vs_normal" in c.columns else c
    for col in score_cols:
        if col in keloid_ref.columns:
            out[f"public_keloid_{col}_mean"] = float(keloid_ref[col].mean())
            out[f"public_keloid_{col}_std"] = float(keloid_ref[col].std())
    return out


def morphology_transfer_diagnostics(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a_morph = a.dropna(subset=["spheroid_area_um2"]).copy()
    if a_morph.empty or b.empty:
        return pd.DataFrame()
    b_area_mean = b["area"].mean()
    b_area_std = b["area"].std()
    out = a_morph[
        ["paper", "condition_code", "cell_source", "fb_ec_ratio", "replicate_id", "spheroid_area_um2"]
    ].copy()
    out["source_dataset"] = "choi_vs_bodenmiller_morphology"
    out["bodenmiller_area_mean"] = b_area_mean
    out["bodenmiller_area_std"] = b_area_std
    out["area_z_vs_bodenmiller"] = (out["spheroid_area_um2"] - b_area_mean) / b_area_std
    out["morphology_size_bin_vs_bodenmiller"] = pd.cut(
        out["area_z_vs_bodenmiller"],
        bins=[-np.inf, -1, 1, np.inf],
        labels=["small", "within_reference", "large"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    spheroid_dir = args.processed_dir / "spheroid"
    signature_dir = args.processed_dir / "microarray/signatures"
    dataset_a = pd.read_parquet(spheroid_dir / "dataset_a_keloid_spheroid.parquet")
    dataset_b = pd.read_parquet(spheroid_dir / "dataset_b_morphology.parquet")
    dataset_c = pd.read_parquet(signature_dir / "dataset_c_keloid_signature.parquet")

    a_aug = add_public_signature_context(add_choi_module_scores(dataset_a), dataset_c)
    a_aug_path = spheroid_dir / "dataset_a_c_augmented.parquet"
    a_aug.to_parquet(a_aug_path, index=False)

    transfer = morphology_transfer_diagnostics(dataset_a, dataset_b)
    transfer_path = spheroid_dir / "dataset_b_to_a_morphology_transfer.parquet"
    transfer.to_parquet(transfer_path, index=False)

    correlations = {}
    if "choi_qpcr_activity_score" in a_aug.columns and "public_keloid_keloid_activity_score_mean" in a_aug.columns:
        subset = a_aug.dropna(subset=["choi_qpcr_activity_score"])
        correlations["n_choi_qpcr_rows"] = int(len(subset))
        correlations["public_keloid_activity_score_mean"] = float(
            a_aug["public_keloid_keloid_activity_score_mean"].dropna().iloc[0]
        )
        correlations["choi_qpcr_activity_score_mean"] = float(subset["choi_qpcr_activity_score"].mean())

    summary = {
        "dataset_a_rows": int(len(dataset_a)),
        "dataset_b_rows": int(len(dataset_b)),
        "dataset_c_rows": int(len(dataset_c)),
        "dataset_a_c_augmented": str(a_aug_path),
        "dataset_b_to_a_transfer": str(transfer_path),
        "correlations": correlations,
    }
    with (args.results_dir / "harmonization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
