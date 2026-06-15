#!/usr/bin/env python3
"""Preprocess scRNA/spatial-derived keloid assets into encoder-ready artifacts.

This script currently emits sample-level artifacts for files that already expose
gene x sample expression tables. Large Seurat/RData and 10x archives are
inventoried in the summary for later AnnData conversion.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as scipy_io

from expression_processing import normalize_gene_symbol, write_expression_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data/raw/scrna_spatial"
OUT_DIR = PROJECT_ROOT / "data/processed/scrna_spatial"
GSE163973_SUFFIX = {"NF1": "1", "NF2": "2", "NF3": "3", "KF1": "4", "KF2": "5", "KF3": "6"}
GSE163973_FIB_BARCODE_SUFFIX = {"KF1": "1", "KF2": "2", "KF3": "3", "NF1": "4", "NF2": "5", "NF3": "6"}
GSE163973_SAMPLE_ALIASES = {
    "KL1": "KF1",
    "KL2": "KF2",
    "KL3": "KF3",
    "NS1": "NF1",
    "NS2": "NF2",
    "NS3": "NF3",
}


def read_gse175866(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = raw_dir / "GSE175866_RNAseq_keloid_fibroblast_exp.M.csv.gz"
    df = pd.read_csv(path, compression="infer")
    df["gene"] = df["genenames"].map(normalize_gene_symbol)
    df = df.dropna(subset=["gene"])
    value_cols = ["FPKM.KF", "FPKM.KZ"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = df.groupby("gene")[value_cols].mean().T
    expr.index = ["GSE175866_KF_other_keloid_fibroblasts", "GSE175866_KZ_CD266pos_CD9neg_fibroblasts"]
    expr = np.log1p(expr.clip(lower=0))
    rows = [
        {
            "sample_id": "GSE175866_KF_other_keloid_fibroblasts",
            "accession": "GSE175866",
            "sample_title": "KF: human keloid other fibroblasts",
            "platform_id": "GPL24676",
            "modality": "scrna_pseudobulk",
            "source_dataset": "scrna_spatial",
            "disease_domain": "keloid",
            "disease_label": "keloid",
            "keloid_vs_normal": "keloid",
            "lesional_status": "lesional",
            "scar_type": "keloid",
            "cell_type": "other_keloid_fibroblast",
            "treatment": "none",
            "patient_id": "pooled",
            "contrast_type": "cell_state",
            "eligible_for_keloid_pretraining": True,
            "encoder_task": "fibroblast_state",
            "encoder_prompt": "Given this keloid fibroblast pseudobulk expression profile, predict the fibroblast state. Answer:",
            "encoder_response": "other_keloid_fibroblast",
        },
        {
            "sample_id": "GSE175866_KZ_CD266pos_CD9neg_fibroblasts",
            "accession": "GSE175866",
            "sample_title": "KZ: human keloid CD266+/CD9- fibroblasts",
            "platform_id": "GPL24676",
            "modality": "scrna_pseudobulk",
            "source_dataset": "scrna_spatial",
            "disease_domain": "keloid",
            "disease_label": "keloid",
            "keloid_vs_normal": "keloid",
            "lesional_status": "lesional",
            "scar_type": "keloid",
            "cell_type": "CD266pos_CD9neg_fibroblast",
            "treatment": "none",
            "patient_id": "pooled",
            "contrast_type": "cell_state",
            "eligible_for_keloid_pretraining": True,
            "encoder_task": "fibroblast_state",
            "encoder_prompt": "Given this keloid fibroblast pseudobulk expression profile, predict the fibroblast state. Answer:",
            "encoder_response": "CD266pos_CD9neg_fibroblast",
        },
    ]
    meta = pd.DataFrame(rows)
    summary = {
        "accession": "GSE175866",
        "status": "ok",
        "n_samples": int(len(meta)),
        "n_genes": int(expr.shape[1]),
        "labels": meta["encoder_response"].value_counts().to_dict(),
    }
    return meta, expr, summary


def _metadata_row(
    *,
    sample_id: str,
    accession: str,
    sample_title: str,
    modality: str,
    disease_label: str,
    keloid_vs_normal: str,
    scar_type: str,
    cell_type: str,
    patient_id: str,
    task: str,
    response: str,
    n_cells: int,
    contrast_type: str = "pseudobulk",
) -> dict:
    prompts = {
        "keloid_vs_normal": "Given this single-cell pseudobulk expression profile, predict whether the originating tissue is keloid or normal scar. Answer:",
        "cell_type": "Given this single-cell pseudobulk expression profile, predict the dominant cell type. Answer:",
        "fibroblast_subcluster": (
            "Given this keloid fibroblast pseudobulk expression profile, predict the fibroblast subcluster. Answer:"
        ),
        "celltype_subcluster": (
            "Given this single-cell pseudobulk expression profile, predict the cell-type subcluster. Answer:"
        ),
        "out_of_domain_fibrotic_state": "Given this skin single-cell pseudobulk expression profile, predict the out-of-domain fibrotic state. Answer:",
    }
    return {
        "sample_id": sample_id,
        "accession": accession,
        "sample_title": sample_title,
        "platform_id": "GPL24676",
        "modality": modality,
        "source_dataset": "scrna_spatial",
        "disease_domain": "keloid",
        "disease_label": disease_label,
        "keloid_vs_normal": keloid_vs_normal,
        "lesional_status": "lesional" if keloid_vs_normal == "keloid" else "control",
        "scar_type": scar_type,
        "cell_type": cell_type,
        "treatment": "none",
        "patient_id": patient_id,
        "contrast_type": contrast_type,
        "eligible_for_keloid_pretraining": True,
        "encoder_task": task,
        "encoder_prompt": prompts[task],
        "encoder_response": response,
        "n_cells_pseudobulk": n_cells,
    }


def _read_inner_10x_archive(outer: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[str, list[str], list[str], object]:
    sample_key = member.name.split("_matrix", maxsplit=1)[0].split("_")[-1]
    nested = outer.extractfile(member)
    if nested is None:
        raise ValueError(f"Could not read {member.name}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        shutil.copyfileobj(nested, tmp)
        tmp.flush()
        with tarfile.open(tmp.name, "r:gz") as inner:
            members = {Path(m.name).name: m for m in inner.getmembers()}
            with gzip.open(inner.extractfile(members["features.tsv.gz"]), "rt") as handle:
                genes = []
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    gene = normalize_gene_symbol(parts[1] if len(parts) > 1 else parts[0])
                    genes.append(gene or parts[0])
            with gzip.open(inner.extractfile(members["barcodes.tsv.gz"]), "rt") as handle:
                barcodes = [line.strip() for line in handle if line.strip()]
            with gzip.open(inner.extractfile(members["matrix.mtx.gz"]), "rb") as handle:
                matrix = scipy_io.mmread(handle).tocsc()
    return sample_key, genes, barcodes, matrix


def _read_10x_members_from_tar(outer: tarfile.TarFile, prefix: str) -> tuple[list[str], list[str], object]:
    matching = {
        kind: m
        for m in outer.getmembers()
        for basename in [Path(m.name).name]
        for kind in ["features", "barcodes", "matrix"]
        if f"_{prefix}_{kind}" in basename or basename.startswith(f"{prefix}_{kind}")
    }
    with gzip.open(outer.extractfile(matching["features"]), "rt") as handle:
        genes = []
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            gene = normalize_gene_symbol(parts[1] if len(parts) > 1 else parts[0])
            genes.append(gene or parts[0])
    with gzip.open(outer.extractfile(matching["barcodes"]), "rt") as handle:
        barcodes = [line.strip() for line in handle if line.strip()]
    with gzip.open(outer.extractfile(matching["matrix"]), "rb") as handle:
        matrix = scipy_io.mmread(handle).tocsc()
    return genes, barcodes, matrix


def _log_cpm_from_matrix(matrix: object, columns: np.ndarray | list[int] | None = None) -> np.ndarray:
    selected = matrix if columns is None else matrix[:, columns]
    summed = np.asarray(selected.sum(axis=1)).ravel().astype(np.float64)
    cpm = summed / max(float(summed.sum()), 1.0) * 1_000_000.0
    return np.log1p(cpm)


def _slug_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def read_gse163973_pseudobulk(raw_dir: Path, min_cells: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_tar = raw_dir / "GSE163973_RAW.tar"
    meta = pd.read_csv(raw_dir / "GSE163973_integrate.all.NS.all.KL_cell.meta.data.csv.gz")
    barcode_col = meta.columns[0]
    meta = meta.rename(columns={barcode_col: "cell_barcode"})
    meta["cell_barcode"] = meta["cell_barcode"].astype(str)

    fib_meta = pd.read_csv(raw_dir / "GSE163973_integrate.all.NS.all.KL.fib_cell.meta.data.csv.gz")
    fib_barcode_col = fib_meta.columns[0]
    fib_meta = fib_meta.rename(columns={fib_barcode_col: "cell_barcode"})
    fib_meta["cell_barcode"] = fib_meta["cell_barcode"].astype(str)

    rows = []
    vectors = []
    vector_ids = []
    gene_index: list[str] | None = None
    with tarfile.open(raw_tar) as outer:
        for member in outer.getmembers():
            if not member.name.endswith(".tar.gz"):
                continue
            sample_key, genes, barcodes, matrix = _read_inner_10x_archive(outer, member)
            sample_key = GSE163973_SAMPLE_ALIASES.get(sample_key, sample_key)
            suffix = GSE163973_SUFFIX[sample_key]
            cell_ids = [barcode.replace("-1", f"_{suffix}") for barcode in barcodes]
            sub_meta = meta[meta["dataset"].eq(sample_key)].set_index("cell_barcode")
            aligned = sub_meta.reindex(cell_ids)
            fib_sub_meta = fib_meta[fib_meta["dataset"].eq(sample_key)].set_index("cell_barcode")
            fib_suffix = GSE163973_FIB_BARCODE_SUFFIX[sample_key]
            fib_cell_ids = [f"{cell_id.rsplit('_', 1)[0]}_{fib_suffix}" for cell_id in cell_ids]
            fib_aligned = fib_sub_meta.reindex(fib_cell_ids)
            if gene_index is None:
                gene_index = genes
            sample_type = "keloid" if sample_key.startswith("KF") else "normal_scar"
            scar_type = "keloid" if sample_type == "keloid" else "normal_scar"
            cell_types = aligned["cellType"].dropna().astype(str).unique()
            for cell_type in cell_types:
                cell_type = str(cell_type)
                valid_idx = np.flatnonzero(aligned["cellType"].astype(str).to_numpy() == cell_type)
                if len(valid_idx) < min_cells:
                    continue
                sample_id = f"GSE163973_{sample_key}_{cell_type}"
                vector_ids.append(sample_id)
                vectors.append(_log_cpm_from_matrix(matrix, valid_idx))
                rows.append(
                    _metadata_row(
                        sample_id=sample_id,
                        accession="GSE163973",
                        sample_title=f"{sample_key} {cell_type} pseudobulk",
                        modality="scrna_pseudobulk",
                        disease_label=sample_type,
                        keloid_vs_normal="keloid" if sample_type == "keloid" else "normal",
                        scar_type=scar_type,
                        cell_type=cell_type,
                        patient_id=sample_key,
                        task="cell_type",
                        response=cell_type,
                        n_cells=len(valid_idx),
                    )
                )

            seurat_clusters = pd.to_numeric(aligned["seurat_clusters"], errors="coerce")
            cell_types_series = aligned["cellType"].astype(str)
            for cell_type in sorted(cell_types_series.dropna().unique()):
                cell_type = str(cell_type)
                if cell_type.lower() in {"unknown", "nan", ""}:
                    continue
                cell_type_mask = cell_types_series.to_numpy() == cell_type
                for cluster_id in sorted(seurat_clusters[cell_type_mask].dropna().astype(int).unique()):
                    cluster_mask = cell_type_mask & (seurat_clusters.to_numpy() == cluster_id)
                    valid_idx = np.flatnonzero(cluster_mask)
                    if len(valid_idx) < min_cells:
                        continue
                    cell_slug = _slug_label(cell_type)
                    response = f"{cell_slug}_cluster_{int(cluster_id)}"
                    sample_id = f"GSE163973_{sample_key}_{response}"
                    vector_ids.append(sample_id)
                    vectors.append(_log_cpm_from_matrix(matrix, valid_idx))
                    rows.append(
                        _metadata_row(
                            sample_id=sample_id,
                            accession="GSE163973",
                            sample_title=f"{sample_key} {cell_type} cluster {cluster_id} pseudobulk",
                            modality="scrna_pseudobulk",
                            disease_label=sample_type,
                            keloid_vs_normal="keloid" if sample_type == "keloid" else "normal",
                            scar_type=scar_type,
                            cell_type=cell_type,
                            patient_id=sample_key,
                            task="celltype_subcluster",
                            response=response,
                            n_cells=len(valid_idx),
                            contrast_type="cell_state",
                        )
                    )

            fib_seurat = pd.to_numeric(fib_aligned["seurat_clusters"], errors="coerce")
            fib_clusters = fib_seurat.dropna().astype(int).unique()
            for cluster_id in sorted(fib_clusters):
                valid_idx = np.flatnonzero(fib_seurat.to_numpy() == cluster_id)
                if len(valid_idx) < min_cells:
                    continue
                response = f"fib_cluster_{int(cluster_id)}"
                sample_id = f"GSE163973_{sample_key}_{response}"
                vector_ids.append(sample_id)
                vectors.append(_log_cpm_from_matrix(matrix, valid_idx))
                rows.append(
                    _metadata_row(
                        sample_id=sample_id,
                        accession="GSE163973",
                        sample_title=f"{sample_key} fibroblast subcluster {cluster_id} pseudobulk",
                        modality="scrna_pseudobulk",
                        disease_label=sample_type,
                        keloid_vs_normal="keloid" if sample_type == "keloid" else "normal",
                        scar_type=scar_type,
                        cell_type="fibroblast",
                        patient_id=sample_key,
                        task="fibroblast_subcluster",
                        response=response,
                        n_cells=len(valid_idx),
                        contrast_type="cell_state",
                    )
                )

            all_valid = np.flatnonzero(aligned["cellType"].notna().to_numpy())
            if len(all_valid) >= min_cells:
                sample_id = f"GSE163973_{sample_key}_all_cells"
                vector_ids.append(sample_id)
                vectors.append(_log_cpm_from_matrix(matrix, all_valid))
                rows.append(
                    _metadata_row(
                        sample_id=sample_id,
                        accession="GSE163973",
                        sample_title=f"{sample_key} all annotated cells pseudobulk",
                        modality="scrna_pseudobulk",
                        disease_label=sample_type,
                        keloid_vs_normal="keloid" if sample_type == "keloid" else "normal",
                        scar_type=scar_type,
                        cell_type="all_cells",
                        patient_id=sample_key,
                        task="keloid_vs_normal",
                        response="keloid" if sample_type == "keloid" else "normal_scar",
                        n_cells=len(all_valid),
                    )
                )

    assert gene_index is not None
    expr = pd.DataFrame(vectors, index=vector_ids, columns=gene_index)
    expr = expr.T.groupby(level=0).mean().T
    metadata = pd.DataFrame(rows)
    fib_rows = metadata[metadata["encoder_task"].eq("fibroblast_subcluster")]
    celltype_rows = metadata[metadata["encoder_task"].eq("celltype_subcluster")]
    summary = {
        "accession": "GSE163973",
        "status": "ok_pseudobulk",
        "n_source_cells": int(len(meta)),
        "n_source_fibroblast_cells": int(len(fib_meta)),
        "n_pseudobulk_samples": int(len(metadata)),
        "n_fibroblast_subcluster_samples": int(len(fib_rows)),
        "n_celltype_subcluster_samples": int(len(celltype_rows)),
        "n_genes": int(expr.shape[1]),
        "min_cells_per_pseudobulk": int(min_cells),
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "fibroblast_subcluster_labels": fib_rows["encoder_response"].value_counts().to_dict(),
        "celltype_subcluster_labels": celltype_rows["encoder_response"].value_counts().to_dict(),
        "source_samples": metadata["patient_id"].value_counts().to_dict(),
    }
    return metadata, expr, summary


def read_gse181297_pseudobulk(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_tar = raw_dir / "GSE181297_RAW.tar"
    sample_labels = {
        "Ke01": ("keloid", "keloid", "keloid", "scRNA"),
        "Ke02": ("keloid", "keloid", "keloid", "scRNA"),
        "Pt1": ("keloid", "keloid", "keloid", "spatial_visium"),
        "Pt2": ("keloid", "keloid", "keloid", "spatial_visium"),
        "NS02": ("normal_scar", "normal", "normal_scar", "scRNA"),
        "NSV1": ("adjacent_normal", "normal", "adjacent_normal", "spatial_visium"),
        "NSV2": ("adjacent_normal", "normal", "adjacent_normal", "spatial_visium"),
    }
    rows = []
    vectors = []
    vector_ids = []
    gene_index: list[str] | None = None
    with tarfile.open(raw_tar) as outer:
        for sample_key, (disease_label, kvn, scar_type, platform_kind) in sample_labels.items():
            genes, barcodes, matrix = _read_10x_members_from_tar(outer, sample_key)
            if gene_index is None:
                gene_index = genes
            sample_id = f"GSE181297_{sample_key}_all_barcodes"
            vector_ids.append(sample_id)
            vectors.append(_log_cpm_from_matrix(matrix))
            rows.append(
                _metadata_row(
                    sample_id=sample_id,
                    accession="GSE181297",
                    sample_title=f"{sample_key} all {platform_kind} barcodes pseudobulk",
                    modality="spatial_pseudobulk" if platform_kind == "spatial_visium" else "scrna_pseudobulk",
                    disease_label=disease_label,
                    keloid_vs_normal=kvn,
                    scar_type=scar_type,
                    cell_type="all_barcodes",
                    patient_id=sample_key,
                    task="keloid_vs_normal",
                    response="keloid" if kvn == "keloid" else disease_label,
                    n_cells=len(barcodes),
                )
            )
    assert gene_index is not None
    expr = pd.DataFrame(vectors, index=vector_ids, columns=gene_index)
    expr = expr.T.groupby(level=0).mean().T
    metadata = pd.DataFrame(rows)
    summary = {
        "accession": "GSE181297",
        "status": "ok_pseudobulk",
        "n_source_barcodes": int(metadata["n_cells_pseudobulk"].sum()),
        "n_pseudobulk_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "source_samples": metadata["patient_id"].value_counts().to_dict(),
    }
    return metadata, expr, summary


def read_gse160536_all_cells(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = raw_dir / "GSE160536_counts_ProcessingData.csv.gz"
    gene_sums: dict[str, float] = {}
    n_barcodes = 0
    with gzip.open(path, "rt", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        n_barcodes = len(header)
        for row in reader:
            if not row:
                continue
            gene = normalize_gene_symbol(row[0])
            if not gene:
                continue
            total = 0.0
            for value in row[1:]:
                if value:
                    try:
                        total += float(value)
                    except ValueError:
                        continue
            gene_sums[gene] = gene_sums.get(gene, 0.0) + total
    genes = list(gene_sums)
    summed = np.array([gene_sums[gene] for gene in genes], dtype=np.float64)
    cpm = summed / max(float(summed.sum()), 1.0) * 1_000_000.0
    expr = pd.DataFrame([np.log1p(cpm)], index=["GSE160536_scleroderma_all_cells"], columns=genes)
    expr = expr.T.groupby(level=0).mean().T
    metadata = pd.DataFrame(
        [
            _metadata_row(
                sample_id="GSE160536_scleroderma_all_cells",
                accession="GSE160536",
                sample_title="All uploaded localized scleroderma scRNA barcodes pseudobulk",
                modality="scrna_pseudobulk",
                disease_label="scleroderma",
                keloid_vs_normal="unknown",
                scar_type="out_of_domain_fibrotic_skin",
                cell_type="all_cells",
                patient_id="pooled_unknown_barcode_mapping",
                task="out_of_domain_fibrotic_state",
                response="scleroderma",
                n_cells=n_barcodes,
            )
        ]
    )
    summary = {
        "accession": "GSE160536",
        "status": "ok_all_cells_pseudobulk",
        "n_source_barcodes": int(n_barcodes),
        "n_pseudobulk_samples": 1,
        "n_genes": int(expr.shape[1]),
        "labels": {"scleroderma": 1},
        "limitation": "barcodes could not be assigned to the six GEO samples from uploaded files, so only pooled out-of-domain pseudobulk is emitted",
    }
    return metadata, expr, summary


def summarize_metadata_tables(raw_dir: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(raw_dir.glob("*cell.meta.data.csv.gz")):
        df = pd.read_csv(path, compression="infer")
        out_path = out_dir / path.name.replace(".csv.gz", ".summary.json")
        summary = {
            "file": path.name,
            "n_cells": int(len(df)),
            "columns": df.columns.tolist(),
        }
        for col in ["dataset", "orig.ident", "cellType", "seurat_clusters"]:
            if col in df.columns:
                counts = df[col].astype(str).value_counts().head(50).to_dict()
                summary[f"{col}_counts"] = counts
        out_path.write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
    return summaries


def tar_preview(path: Path, limit: int = 20) -> list[str]:
    with tarfile.open(path) as tf:
        return tf.getnames()[:limit]


def process(args: argparse.Namespace) -> None:
    meta, expr, dataset_summary = read_gse175866(args.raw_dir)
    dataset_summaries = [dataset_summary]
    if args.include_gse163973:
        gse163973_meta, gse163973_expr, gse163973_summary = read_gse163973_pseudobulk(
            args.raw_dir,
            min_cells=args.min_cells_per_pseudobulk,
        )
        meta = pd.concat([meta, gse163973_meta], ignore_index=True, sort=False)
        expr = pd.concat([expr, gse163973_expr], ignore_index=True, sort=False).fillna(0.0)
        dataset_summaries.append(gse163973_summary)
    else:
        dataset_summaries.append({"accession": "GSE163973", "status": "metadata_only"})
    if args.include_gse181297:
        gse181297_meta, gse181297_expr, gse181297_summary = read_gse181297_pseudobulk(args.raw_dir)
        meta = pd.concat([meta, gse181297_meta], ignore_index=True, sort=False)
        expr = pd.concat([expr, gse181297_expr], ignore_index=True, sort=False).fillna(0.0)
        dataset_summaries.append(gse181297_summary)
    else:
        dataset_summaries.append({"accession": "GSE181297", "status": "raw_archive_only"})
    if args.include_gse160536:
        gse160536_meta, gse160536_expr, gse160536_summary = read_gse160536_all_cells(args.raw_dir)
        meta = pd.concat([meta, gse160536_meta], ignore_index=True, sort=False)
        expr = pd.concat([expr, gse160536_expr], ignore_index=True, sort=False).fillna(0.0)
        dataset_summaries.append(gse160536_summary)
    else:
        dataset_summaries.append({"accession": "GSE160536", "status": "raw_counts_only"})
    if expr.shape[1] > args.max_genes:
        variances = expr.var(axis=0).sort_values(ascending=False)
        module_genes = {
            "COL1A1",
            "COL3A1",
            "FN1",
            "ACTA2",
            "TAGLN",
            "MYL9",
            "TGFB1",
            "TGFB3",
            "TGFBR1",
            "TGFBR2",
            "SMAD2",
            "SMAD3",
            "HIF1A",
            "PECAM1",
            "VWF",
            "KDR",
            "MMP14",
            "ADAM12",
            "HTRA1",
            "CTHRC1",
            "POSTN",
            "IGFBP2",
        }
        selected = list(dict.fromkeys([*variances.head(args.max_genes).index.tolist(), *[g for g in module_genes if g in expr.columns]]))
        expr = expr[selected]
    metadata_summaries = summarize_metadata_tables(args.raw_dir, args.out_dir / "metadata_inventory")
    dataset_summaries = [
        {**summary, "metadata_tables": metadata_summaries}
        if summary.get("accession") == "GSE163973"
        else summary
        for summary in dataset_summaries
    ]
    skipped = [
        {
            "file": "GSE163973_integrate.all.NS.all.KL*.Rdata.gz",
            "reason": "large Seurat/RData objects not required because matching 10x matrices were processed directly",
        },
        {
            "file": "GSE202203_RawCounts_gene_3207.tsv.gz and GSE202203_TPM_Raw_gene_3207.tsv.gz",
            "reason": "breast tumor out-of-domain dataset; not included in keloid scRNA/spatial corpus",
        },
    ]
    raw_tar = args.raw_dir / "GSE181297_RAW.tar"
    if raw_tar.exists():
        skipped.append({"file": raw_tar.name, "preview_entries": tar_preview(raw_tar)})

    summary = write_expression_artifacts(
        out_dir=args.out_dir,
        prefix="scrna_spatial",
        metadata=meta,
        expr=expr,
        dataset_summaries=dataset_summaries,
        skipped=skipped,
        jsonl_top_genes=args.jsonl_top_genes,
    )
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--jsonl-top-genes", type=int, default=2048)
    parser.add_argument("--max-genes", type=int, default=10000)
    parser.add_argument("--min-cells-per-pseudobulk", type=int, default=30)
    parser.add_argument("--include-gse163973", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-gse181297", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-gse160536", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
