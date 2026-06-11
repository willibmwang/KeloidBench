#!/usr/bin/env python3
"""Build Dataset C: public keloid transcriptomic module scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data/raw/microarray/keloid_geo_legacy"
OUT_DIR = PROJECT_ROOT / "data/processed/microarray/signatures"

GEO_ACCESSIONS = ["GSE7980", "GSE44270", "GSE145725"]
ARRAYEXPRESS_ACCESSIONS = ["E-MTAB-2509", "E-MTAB-4945"]

MODULES = {
    "ECM_score": ["COL1A1", "COL3A1", "FN1"],
    "myofibroblast_score": ["ACTA2"],
    "TGFb_score": ["TGFB1", "TGFB3", "TGFBR2"],
    "hypoxia_vascular_score": ["HIF1A", "PECAM1", "VWF"],
    "remodeling_score": ["MMP14", "ADAM12", "HTRA1", "CTHRC1"],
}
MARKERS = sorted({gene for genes in MODULES.values() for gene in genes})


def infer_label(title: str) -> str:
    lower = title.lower()
    if "keloid" in lower:
        return "keloid"
    if "normal" in lower or "healthy" in lower:
        return "normal"
    if "scar" in lower:
        return "scar"
    return "unknown"


def _annotation_text(gpl_table: pd.DataFrame) -> pd.Series:
    columns = [c for c in ["Gene Symbol", "gene_assignment", "GENE_SYMBOL", "Symbol", "gene_symbol"] if c in gpl_table.columns]
    if not columns:
        columns = [c for c in gpl_table.columns if "gene" in c.lower() or "symbol" in c.lower()]
    if not columns:
        return pd.Series("", index=gpl_table.index)
    return gpl_table[columns].fillna("").astype(str).agg(" ".join, axis=1)


def marker_probe_map(gpl_table: pd.DataFrame) -> dict[str, list[str]]:
    probe_col = "ID" if "ID" in gpl_table.columns else gpl_table.columns[0]
    text = _annotation_text(gpl_table)
    mapping: dict[str, list[str]] = {}
    for gene in MARKERS:
        exact_patterns = [
            f"// {gene} //",
            f"({gene})",
            f" {gene} ",
        ]
        mask = text.apply(lambda x: any(pattern in f" {x} " for pattern in exact_patterns))
        probes = gpl_table.loc[mask, probe_col].astype(str).tolist()
        if probes:
            mapping[gene] = probes
    return mapping


def expression_by_gene(gse) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    expr = gse.pivot_samples("VALUE")
    expr.index = expr.index.astype(str)

    probe_mapping: dict[str, list[str]] = {}
    for gpl in gse.gpls.values():
        probe_mapping.update(marker_probe_map(gpl.table))

    gene_expr = {}
    for gene, probes in probe_mapping.items():
        available = [probe for probe in probes if probe in expr.index]
        if not available:
            continue
        gene_expr[gene] = expr.loc[available].astype(float).mean(axis=0)
    return pd.DataFrame(gene_expr), probe_mapping


def score_accession(accession: str, raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    gse = GEOparse.get_GEO(geo=accession, destdir=str(raw_dir), silent=True)
    gene_expr, probe_mapping = expression_by_gene(gse)
    if gene_expr.empty:
        return pd.DataFrame(), {"accession": accession, "status": "no_marker_genes"}

    z = (gene_expr - gene_expr.mean(axis=0)) / gene_expr.std(axis=0).replace(0, np.nan)
    sample_rows = []
    for sample_id in z.index:
        gsm = gse.gsms[sample_id]
        title = " ".join(gsm.metadata.get("title", [""]))
        row = {
            "source_dataset": "keloid_geo",
            "accession": accession,
            "sample_id": sample_id,
            "sample_title": title,
            "keloid_vs_normal": infer_label(title),
        }
        for module_name, genes in MODULES.items():
            available = [gene for gene in genes if gene in z.columns]
            row[module_name] = float(z.loc[sample_id, available].mean()) if available else np.nan
        score_cols = list(MODULES.keys())
        row["keloid_activity_score"] = float(np.nanmean([row[c] for c in score_cols]))
        for gene in MARKERS:
            if gene in gene_expr.columns:
                row[gene] = float(gene_expr.loc[sample_id, gene])
        sample_rows.append(row)

    return pd.DataFrame(sample_rows), {
        "accession": accession,
        "status": "ok",
        "n_samples": len(sample_rows),
        "mapped_genes": sorted(gene_expr.columns.tolist()),
        "probe_counts": {gene: len(probes) for gene, probes in probe_mapping.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    summary = []
    for accession in GEO_ACCESSIONS:
        try:
            table, info = score_accession(accession, args.raw_dir)
            summary.append(info)
            if not table.empty:
                tables.append(table)
                table.to_parquet(args.out_dir / f"{accession}_keloid_signature.parquet", index=False)
        except Exception as exc:  # noqa: BLE001
            summary.append({"accession": accession, "status": "error", "error": str(exc)})

    dataset_c = pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()
    dataset_c.to_parquet(args.out_dir / "dataset_c_keloid_signature.parquet", index=False)
    dataset_c.to_parquet(args.out_dir / "keloid_activity_score.parquet", index=False)

    arrayexpress_manifest = pd.DataFrame(
        {
            "accession": ARRAYEXPRESS_ACCESSIONS,
            "source_url": [f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{acc}" for acc in ARRAYEXPRESS_ACCESSIONS],
            "status": "recorded_for_followup",
        }
    )
    arrayexpress_manifest.to_csv(args.raw_dir / "arrayexpress_manifest.csv", index=False)

    with (args.out_dir / "dataset_c_summary.json").open("w") as f:
        json.dump(
            {
                "geo": summary,
                "arrayexpress": arrayexpress_manifest.to_dict(orient="records"),
                "n_rows": int(len(dataset_c)),
            },
            f,
            indent=2,
        )

    print(f"dataset_c_keloid_signature: {len(dataset_c)} rows")
    print(args.out_dir / "dataset_c_keloid_signature.parquet")


if __name__ == "__main__":
    main()
