#!/usr/bin/env python3
"""Build Dataset A from Choi source data and Dirand qualitative controls."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data/processed"
CHOI_SCRIPT = PROJECT_ROOT / "scripts/02_extract_choi.py"


def build_dirand_seed() -> pd.DataFrame:
    """Create the Dirand control table described in the project plan.

    Dirand et al. report that keloid fibroblasts lose fibrotic features when moved
    from 2D monolayer into fibroblast-only 3D spheroids and lose TGF-beta1
    sensitivity. These rows are intentionally provenance-marked as qualitative
    controls rather than digitized replicate measurements.
    """
    records = []
    for cell_source in ["KF", "NDF"]:
        for culture_format in ["2D_monolayer", "3D_spheroid"]:
            for tgfb1 in ["none", "present"]:
                if cell_source == "KF" and culture_format == "2D_monolayer":
                    fibrotic_state = "active_fibrotic"
                    alpha_sma = 1.0 if tgfb1 == "none" else 1.25
                    collagen_activity = 1.0 if tgfb1 == "none" else 1.2
                else:
                    fibrotic_state = "deactivated"
                    alpha_sma = 0.25 if culture_format == "3D_spheroid" else 0.4
                    collagen_activity = 0.3 if culture_format == "3D_spheroid" else 0.5

                records.append(
                    {
                        "paper": "dirand_2023",
                        "source_dataset": "dirand_seed",
                        "source_sheet": "paper_qualitative_control",
                        "condition_code": f"Dirand_{cell_source}_{culture_format}_{tgfb1}",
                        "cell_source": cell_source,
                        "culture_format": culture_format,
                        "fb_ec_ratio": "1:0",
                        "tgfb1": tgfb1,
                        "replicate_id": 1,
                        "fibrotic_state": fibrotic_state,
                        "alpha_SMA": alpha_sma,
                        "COL1A1_COL3A1_ratio": collagen_activity,
                        "notes": (
                            "Qualitative control from Dirand et al. 2023: KF spheroids "
                            "deactivate and lose TGF-beta1 sensitivity."
                        ),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-choi", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_choi:
        subprocess.run([sys.executable, str(CHOI_SCRIPT), "--out-dir", str(args.out_dir)], check=True)

    dataset_a_path = args.out_dir / "dataset_a_keloid_spheroid.parquet"
    choi = pd.read_parquet(dataset_a_path)

    dirand = build_dirand_seed()
    dirand_path = args.out_dir / "dirand_seed_labels.parquet"
    dirand.to_parquet(dirand_path, index=False)

    combined = pd.concat([choi, dirand], ignore_index=True, sort=False)
    combined.to_parquet(dataset_a_path, index=False)

    summary_path = args.out_dir / "dataset_a_summary.csv"
    summary = (
        combined.assign(source_dataset=combined.get("source_dataset", pd.Series(index=combined.index)).fillna("choi"))
        .groupby(["paper", "source_dataset"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    summary.to_csv(summary_path, index=False)

    print(f"dirand_seed: {len(dirand)} rows -> {dirand_path}")
    print(f"dataset_a_combined: {len(combined)} rows -> {dataset_a_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
