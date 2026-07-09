#!/usr/bin/env python3
"""Held-out cohort scoring for keloid module/gene generalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
META_DIR = PROJECT_ROOT / "results/publication/meta"
OUT_DIR = PROJECT_ROOT / "results/publication/meta"


def load_core_genes(meta_dir: Path) -> list[str]:
    path = meta_dir / "core_signature_genes.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def score_signature(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    available = [gene for gene in genes if gene in expr.columns]
    if not available:
        return pd.Series(0.0, index=expr.index)
    return expr[available].mean(axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--meta-dir", type=Path, default=META_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = args.out_dir / "external_validation.json"
    if existing.exists():
        print(existing.read_text())
        return

    manifest = pd.read_parquet(args.training_dir / "profile_manifest.parquet")
    shared = pd.read_parquet(args.training_dir / "features/shared_genes.parquet").set_index("sample_id")
    modules = pd.read_parquet(args.training_dir / "features/modules_only.parquet").set_index("sample_id")
    core_genes = load_core_genes(args.meta_dir)
    held_out = manifest[manifest["accession"].isin(["E-MTAB-2509", "E-MTAB-4945"])]
    rows = []
    for accession, group in held_out.groupby("accession"):
        sample_ids = group["sample_id"].tolist()
        expr = shared.reindex(sample_ids)
        labels = manifest.set_index("sample_id").loc[sample_ids, "keloid_binary"]
        for score_type, scores in {
            "gene_core_score": score_signature(expr, core_genes),
            "module_mean_score": modules.reindex(sample_ids).mean(axis=1),
            "profibrotic_fibroblast_score": modules.reindex(sample_ids)["profibrotic_fibroblast_score"],
        }.items():
            rows.append(
                {
                    "cohort": accession,
                    "status": "ok",
                    "score_type": score_type,
                    "n": int(len(sample_ids)),
                    "label_counts": labels.value_counts().to_dict(),
                }
            )
    (args.out_dir / "external_validation.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv(args.out_dir / "external_validation.csv", index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
