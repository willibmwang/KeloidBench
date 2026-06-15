"""Shared helpers for SpheroScar expression preprocessing scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "ECM_score": ["COL1A1", "COL3A1", "FN1"],
    "myofibroblast_score": ["ACTA2", "TAGLN", "MYL9"],
    "TGFb_score": ["TGFB1", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3"],
    "hypoxia_vascular_score": ["HIF1A", "PECAM1", "VWF", "KDR"],
    "remodeling_score": ["MMP14", "ADAM12", "HTRA1", "CTHRC1"],
    "profibrotic_fibroblast_score": ["POSTN"],
    "antifibrotic_fibroblast_score": ["IGFBP2"],
}

METADATA_COLUMNS = [
    "sample_id",
    "accession",
    "sample_title",
    "platform_id",
    "modality",
    "source_dataset",
    "disease_domain",
    "disease_label",
    "keloid_vs_normal",
    "lesional_status",
    "scar_type",
    "cell_type",
    "treatment",
    "patient_id",
    "contrast_type",
    "eligible_for_keloid_pretraining",
    "encoder_task",
    "encoder_prompt",
    "encoder_response",
]


def normalize_gene_symbol(raw: object) -> str | None:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "na", "---", "null", "none"}:
        return None
    gene = re.sub(r"[^A-Za-z0-9_.-]", "", text).upper()
    if not gene or gene in {"NA", "NAN", "NULL", "GENE", "SYMBOL"}:
        return None
    return gene


def make_unique_index(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for value in values:
        base = value
        counts[base] = counts.get(base, 0) + 1
        out.append(base if counts[base] == 1 else f"{base}__{counts[base]}")
    return out


def zscore_by_accession(expr: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    out = expr.copy()
    for _, idx in metadata.groupby("accession").groups.items():
        block = out.loc[idx]
        means = block.mean(axis=0, skipna=True)
        stds = block.std(axis=0, skipna=True).replace(0, np.nan)
        out.loc[idx] = (block - means) / stds
    return out.fillna(0.0)


def add_module_scores(metadata: pd.DataFrame, expr: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    for score_name, genes in MODULES.items():
        available = [gene for gene in genes if gene in expr.columns]
        out[score_name] = expr[available].mean(axis=1).values if available else np.nan
    out["fibrotic_activity_score"] = out[list(MODULES)].mean(axis=1, skipna=True)
    return out


def write_jsonl(path: Path, metadata: pd.DataFrame, expr: pd.DataFrame, max_genes: int) -> None:
    variances = expr.var(axis=0).sort_values(ascending=False)
    selected_genes = variances.head(max_genes).index.tolist()
    with path.open("w") as handle:
        for idx, row in metadata.iterrows():
            values = expr.loc[idx, selected_genes]
            gene_values = {
                gene: float(value)
                for gene, value in values.items()
                if np.isfinite(value) and value != 0
            }
            payload = {
                "sample_id": row["sample_id"],
                "accession": row["accession"],
                "task": row["encoder_task"],
                "prompt": row["encoder_prompt"],
                "response": row["encoder_response"],
                "genes": gene_values,
            }
            handle.write(json.dumps(payload) + "\n")


def write_expression_artifacts(
    *,
    out_dir: Path,
    prefix: str,
    metadata: pd.DataFrame,
    expr: pd.DataFrame,
    dataset_summaries: list[dict],
    skipped: list[dict] | None = None,
    jsonl_top_genes: int = 2048,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata.reset_index(drop=True)
    expr = expr.reset_index(drop=True)
    expr = expr.reindex(sorted(expr.columns), axis=1)
    expr = zscore_by_accession(expr, metadata)
    metadata = add_module_scores(metadata, expr)

    for col in METADATA_COLUMNS:
        if col not in metadata.columns:
            metadata[col] = "unknown"

    gene_cols = expr.columns.tolist()
    module_cols = list(MODULES) + ["fibrotic_activity_score"]
    wide = pd.concat([metadata[METADATA_COLUMNS + module_cols], expr], axis=1)

    metadata_path = out_dir / f"{prefix}_sample_metadata.parquet"
    wide_path = out_dir / f"{prefix}_expression_wide.parquet"
    npy_path = out_dir / f"{prefix}_X.npy"
    labels_path = out_dir / f"{prefix}_encoder_labels.csv"
    vocab_path = out_dir / f"{prefix}_gene_vocab.txt"
    jsonl_path = out_dir / f"{prefix}_encoder_decoder_samples.jsonl"
    summary_path = out_dir / f"{prefix}_summary.json"

    metadata.to_parquet(metadata_path, index=False)
    wide.to_parquet(wide_path, index=False)
    np.save(npy_path, expr.to_numpy(dtype=np.float32))
    metadata[["sample_id", "accession", "encoder_task", "encoder_prompt", "encoder_response"]].to_csv(
        labels_path,
        index=False,
    )
    vocab_path.write_text("\n".join(gene_cols) + "\n")
    write_jsonl(jsonl_path, metadata, expr, jsonl_top_genes)

    summary = {
        "n_samples": int(len(metadata)),
        "n_genes": int(len(gene_cols)),
        "accessions": dataset_summaries,
        "skipped": skipped or [],
        "outputs": {
            "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
            "wide": str(wide_path.relative_to(PROJECT_ROOT)),
            "matrix": str(npy_path.relative_to(PROJECT_ROOT)),
            "labels": str(labels_path.relative_to(PROJECT_ROOT)),
            "gene_vocab": str(vocab_path.relative_to(PROJECT_ROOT)),
            "jsonl": str(jsonl_path.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
