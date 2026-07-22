"""Shared helpers for SpheroScar expression preprocessing scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gene_modules import MODULE_COLUMNS, MODULE_GENES, MODULES

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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
    "eval_grain",
    "cohort_role",
    "allow_accession_level_grouping",
    "lockbox",
]


def is_hgnc_like_symbol(gene: str) -> bool:
    """Reject RefSeq/Ensembl accessions that are not gene symbols."""
    if not gene:
        return False
    if gene.startswith(("NM_", "NR_", "XM_", "XR_", "NP_", "XP_", "ENSG", "ENST", "ENSP")):
        return False
    if gene.isdigit():
        return False
    return bool(re.match(r"^[A-Z][A-Z0-9.-]*$", gene))


def symbol_from_gene_assignment(raw: object) -> str | None:
    """Parse Affymetrix-style gene_assignment fields.

    Format is typically:
      ACCESSION // SYMBOL // description // cytoband // entrez /// ...
    Prefer the SYMBOL field rather than the first whitespace/RefSeq token.
    """
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "na", "---", "null", "none", "---"}:
        return None
    for block in re.split(r"\s*///\s*", text):
        parts = [part.strip() for part in re.split(r"\s*//\s*", block)]
        if len(parts) >= 2:
            candidate = normalize_gene_symbol(parts[1])
            if candidate and is_hgnc_like_symbol(candidate):
                return candidate
        for part in parts:
            candidate = normalize_gene_symbol(part)
            if candidate and is_hgnc_like_symbol(candidate):
                return candidate
    return None


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
    for score_name, genes in MODULE_GENES.items():
        available = [gene for gene in genes if gene in expr.columns]
        out[score_name] = expr[available].mean(axis=1).values if available else np.nan
    score_cols = list(MODULE_GENES)
    out["fibrotic_activity_score"] = out[score_cols].mean(axis=1, skipna=True)
    return out


def module_coverage_report(expr: pd.DataFrame, accession: str | None = None) -> dict:
    """Report mapped gene coverage and variance for each program."""
    rows = []
    for score_name, genes in MODULE_GENES.items():
        available = [gene for gene in genes if gene in expr.columns]
        if available:
            block = expr[available]
            nonzero = int((block.fillna(0.0).abs() > 0).any(axis=0).sum())
            variance = float(block.var(axis=0, skipna=True).fillna(0.0).sum())
        else:
            nonzero = 0
            variance = 0.0
        rows.append(
            {
                "accession": accession,
                "module": score_name,
                "n_genes_defined": len(genes),
                "n_genes_present": len(available),
                "n_genes_nonzero": nonzero,
                "variance_sum": variance,
                "genes_present": ",".join(available),
            }
        )
    return {
        "accession": accession,
        "modules": rows,
        "n_modules_with_signal": int(sum(1 for row in rows if row["variance_sum"] > 0)),
        "all_zero_programs": bool(all(row["variance_sum"] <= 0 for row in rows)),
    }


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
    module_cols = list(MODULE_COLUMNS)
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
            "metadata": _rel(metadata_path),
            "wide": _rel(wide_path),
            "matrix": _rel(npy_path),
            "labels": _rel(labels_path),
            "gene_vocab": _rel(vocab_path),
            "jsonl": _rel(jsonl_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
