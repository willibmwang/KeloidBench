#!/usr/bin/env python3
"""Preprocess bulk RNA-seq matrices into SpheroScar encoder-ready artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from expression_processing import module_coverage_report, normalize_gene_symbol, write_expression_artifacts
from ensembl_map import load_ensembl_symbol_map, strip_ensembl_version
from gene_modules import ALL_PINNED_GENES, MODULE_GENES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data/raw/bulk_rnaseq"
OUT_DIR = PROJECT_ROOT / "data/processed/bulk_rnaseq"


def _read_matrix(path: Path, sep: str, gene_col: str, *, ensembl_map: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, compression="infer")
    df = df.rename(columns={gene_col: "gene"})
    if ensembl_map is not None:
        mapped = []
        for raw in df["gene"].astype(str):
            key = strip_ensembl_version(raw)
            mapped.append(ensembl_map.get(key) or normalize_gene_symbol(raw))
        df["gene"] = mapped
    else:
        df["gene"] = df["gene"].map(normalize_gene_symbol)
    df = df.dropna(subset=["gene"])
    value_cols = [col for col in df.columns if col != "gene"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = df.groupby("gene")[value_cols].mean().T
    expr.index.name = "sample_id"
    return np.log1p(expr.clip(lower=0))


def _metadata_row(accession: str, sample_id: str, task: str, response: str, **extra) -> dict:
    prompt = {
        "keloid_vs_normal": "Given this bulk RNA-seq expression profile, predict whether the sample is keloid or normal. Answer:",
        "lesional_status": "Given this keloid bulk RNA-seq expression profile, predict the lesional status. Answer:",
        "scar_differential": "Given this scar bulk RNA-seq expression profile, predict the scar type. Answer:",
    }.get(task, "Given this bulk RNA-seq expression profile, predict the biological state. Answer:")
    row = {
        "sample_id": sample_id,
        "accession": accession,
        "sample_title": sample_id,
        "platform_id": "bulk_rnaseq",
        "modality": "bulk_rnaseq",
        "source_dataset": "bulk_rnaseq",
        "disease_domain": "keloid",
        "disease_label": response,
        "keloid_vs_normal": "unknown",
        "lesional_status": "unknown",
        "scar_type": "unknown",
        "cell_type": "bulk_tissue",
        "treatment": "none",
        "patient_id": "unknown",
        "contrast_type": "sample",
        "eligible_for_keloid_pretraining": True,
        "encoder_task": task,
        "encoder_prompt": prompt,
        "encoder_response": response,
    }
    row.update(extra)
    return row


def process_gse158395(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = raw_dir / "GSE158395_exp_fin_geo.csv.gz"
    expr = _read_matrix(path, sep=",", gene_col="Unnamed: 0")
    rows = []
    for sample_id in expr.index:
        parts = sample_id.split("_")
        if "_Keloid_LS_" in sample_id:
            response = "lesional"
            task = "lesional_status"
            disease = "keloid"
            lesion = "lesional"
        elif "_Keloid_NL_" in sample_id:
            response = "non_lesional"
            task = "lesional_status"
            disease = "keloid"
            lesion = "non_lesional"
        else:
            response = "normal"
            task = "keloid_vs_normal"
            disease = "normal"
            lesion = "control"
        rows.append(
            _metadata_row(
                "GSE158395",
                sample_id,
                task,
                response,
                disease_label=disease,
                keloid_vs_normal="keloid" if disease == "keloid" else "normal",
                lesional_status=lesion,
                patient_id=parts[0],
            )
        )
    meta = pd.DataFrame(rows)
    return meta, expr, {"accession": "GSE158395", "status": "ok", "n_samples": int(len(meta)), "n_genes": int(expr.shape[1]), "labels": meta["encoder_response"].value_counts().to_dict()}


def process_gse188952(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = raw_dir / "GSE188952_Processed_FPKM.tsv.gz"
    expr = _read_matrix(path, sep="\t", gene_col="Symbol")
    rows = []
    for sample_id in expr.index:
        lower = sample_id.lower()
        if lower.startswith("keloid"):
            response = "keloid"
            task = "scar_differential"
            scar = "keloid"
        elif lower.startswith("hypertrophicscar"):
            response = "hypertrophic_scar"
            task = "scar_differential"
            scar = "hypertrophic_scar"
        else:
            response = "normotrophic_scar"
            task = "scar_differential"
            scar = "normotrophic_scar"
        rows.append(
            _metadata_row(
                "GSE188952",
                sample_id,
                task,
                response,
                scar_type=scar,
                # Do not collapse hypertrophic/normotrophic scar into "normal" skin.
                keloid_vs_normal="keloid" if scar == "keloid" else "unknown",
            )
        )
    meta = pd.DataFrame(rows)
    return meta, expr, {"accession": "GSE188952", "status": "ok", "n_samples": int(len(meta)), "n_genes": int(expr.shape[1]), "labels": meta["encoder_response"].value_counts().to_dict()}


def process_sun_burns(raw_dir: Path, ensembl_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = raw_dir / "Sun et al Burns.txt"
    expr = _read_matrix(path, sep="\t", gene_col="ensembl_gene_id", ensembl_map=ensembl_map)
    rows = []
    for sample_id in expr.index:
        if sample_id.startswith("Treat_K_"):
            response = "keloid"
            kvn = "keloid"
        else:
            response = "normal"
            kvn = "normal"
        rows.append(
            _metadata_row(
                "Sun_Burns",
                sample_id,
                "keloid_vs_normal",
                response,
                disease_label=response,
                keloid_vs_normal=kvn,
            )
        )
    meta = pd.DataFrame(rows)
    coverage = module_coverage_report(expr, accession="Sun_Burns")
    if coverage["all_zero_programs"]:
        raise RuntimeError(
            "Sun_Burns: all program features have zero variance after Ensembl→symbol mapping."
        )
    return (
        meta,
        expr,
        {
            "accession": "Sun_Burns",
            "status": "ok",
            "n_samples": int(len(meta)),
            "n_genes": int(expr.shape[1]),
            "n_modules_with_signal": coverage["n_modules_with_signal"],
            "labels": meta["encoder_response"].value_counts().to_dict(),
        },
    )


def process(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensembl_map = load_ensembl_symbol_map(allow_download=not args.skip_ensembl_download)
    processors = [
        process_gse158395,
        process_gse188952,
        lambda raw_dir: process_sun_burns(raw_dir, ensembl_map),
    ]
    metas = []
    exprs = []
    summaries = []
    coverage_rows = []
    for fn in processors:
        meta, expr, summary = fn(args.raw_dir)
        coverage = module_coverage_report(expr, accession=summary["accession"])
        coverage_rows.extend(coverage["modules"])
        metas.append(meta)
        exprs.append(expr)
        summaries.append(summary)
        print(
            f"{summary['accession']}: {summary['n_samples']} samples, {summary['n_genes']} genes, "
            f"{coverage['n_modules_with_signal']}/{len(MODULE_GENES)} modules with signal"
        )

    metadata = pd.concat(metas, ignore_index=True, sort=False)
    expr = pd.concat(exprs, ignore_index=True, sort=False).fillna(0.0)
    if expr.shape[1] > args.max_genes:
        variances = expr.var(axis=0).sort_values(ascending=False)
        selected = list(
            dict.fromkeys(
                [
                    *variances.head(args.max_genes).index.tolist(),
                    *[g for g in ALL_PINNED_GENES if g in expr.columns],
                ]
            )
        )
        expr = expr[selected]
    skipped = [
        {"file": "GSE125022_List_all-genes_RNAseq_PGS-ANOVA_Keloid-vs-ctrl_no-filter.txt.gz", "reason": "differential gene list, not sample-level expression"},
        {"file": "GSE125022_List_DEGs_RNAseq_PGS-ANOVA_Keloid-vs-ctrl_p0.001_514-genes.txt.gz", "reason": "filtered DEG list, not sample-level expression"},
        {"file": "GSE125022_List_DARs_keloid-vs-ctrl_gene-body+5K-TSS_1597-regions.txt.gz", "reason": "ATAC/DAR annotation list, not sample-level RNA expression"},
        {"file": "GSE125022_RAW.tar", "reason": "large raw archive; not needed for encoder-ready matrix"},
        {"file": "GSE145725_RAW.tar", "reason": "microarray raw archive already represented in legacy signatures"},
    ]
    summary = write_expression_artifacts(
        out_dir=args.out_dir,
        prefix="bulk_rnaseq",
        metadata=metadata,
        expr=expr,
        dataset_summaries=summaries,
        skipped=skipped,
        jsonl_top_genes=args.jsonl_top_genes,
    )
    coverage_path = args.out_dir / "bulk_rnaseq_module_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    summary["module_coverage"] = str(coverage_path.relative_to(PROJECT_ROOT))
    (args.out_dir / "bulk_rnaseq_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--jsonl-top-genes", type=int, default=2048)
    parser.add_argument("--max-genes", type=int, default=10000)
    parser.add_argument("--skip-ensembl-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
