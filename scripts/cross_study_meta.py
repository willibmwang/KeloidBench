#!/usr/bin/env python3
"""Cross-study meta-analysis for keloid reproducibility (rebuild if outputs missing)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
OUT_DIR = PROJECT_ROOT / "results/publication/meta"


def derive_core_signature(meta_df: pd.DataFrame, *, fdr: float, max_i2: float, min_studies: int) -> pd.DataFrame:
    meta_df = meta_df.copy()
    meta_df["tier"] = "non_significant"
    core = (
        meta_df["fdr"].le(fdr)
        & meta_df["i_squared"].le(max_i2)
        & meta_df["n_studies"].ge(min_studies)
    )
    meta_df.loc[core, "tier"] = "core_reproducible"
    fragile = meta_df["fdr"].le(fdr) & ~core
    meta_df.loc[fragile, "tier"] = "study_specific_or_fragile"
    return meta_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--max-i2", type=float, default=0.5)
    parser.add_argument("--min-studies", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "gene_meta_analysis.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
    else:
        per_study = args.out_dir / "per_study_effect_sizes.csv"
        if not per_study.exists():
            raise SystemExit("Missing meta-analysis inputs; existing outputs not found.")
        study_df = pd.read_csv(per_study)
        pooled = (
            study_df.groupby("gene")
            .agg(
                pooled_effect=("effect_size", "mean"),
                i_squared=("i_squared", "mean"),
                n_studies=("accession", "nunique"),
                fdr=("fdr", "min"),
            )
            .reset_index()
        )
        meta_df = derive_core_signature(pooled, fdr=args.fdr, max_i2=args.max_i2, min_studies=args.min_studies)
        meta_df.to_csv(meta_path, index=False)

    meta_df = derive_core_signature(meta_df, fdr=args.fdr, max_i2=args.max_i2, min_studies=args.min_studies)
    core_genes = meta_df.loc[meta_df["tier"].eq("core_reproducible"), "gene"].astype(str).tolist()
    (args.out_dir / "core_signature_genes.txt").write_text("\n".join(core_genes) + ("\n" if core_genes else ""))
    summary = {
        "n_genes": int(len(meta_df)),
        "n_core_genes": int(len(core_genes)),
        "core_genes": core_genes,
    }
    (args.out_dir / "meta_analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
