#!/usr/bin/env python3
"""Download and preprocess public keloid cohorts for the F1 improvement ladder.

Development cohorts may enter nested selection. Lockbox cohorts are processed
but must never be used for feature/model/threshold selection.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from expression_processing import (
    module_coverage_report,
    normalize_gene_symbol,
    write_expression_artifacts,
)
from ensembl_map import load_ensembl_symbol_map, strip_ensembl_version
from entrez_map import fetch_entrez_symbols
from gene_modules import ALL_PINNED_GENES, MODULE_GENES
from preprocess_microarray import (
    PLATFORM_DIR,
    expression_to_genes,
    load_platform_mapping,
    parse_series_matrix,
)
from refseq_map import fetch_refseq_symbols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"
RAW_DIR = PROJECT_ROOT / "data/raw/public_keloid"
OUT_DIR = PROJECT_ROOT / "data/processed/public_keloid"


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text())


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    print(f"Downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return dest


def download_cohort(
    entry: dict,
    raw_dir: Path,
    force: bool = False,
    include_reserved_expression: bool = False,
) -> dict[str, Path]:
    accession = entry["accession"]
    acc_dir = raw_dir / accession
    paths: dict[str, Path] = {}
    download = dict(entry.get("download", {}))
    if include_reserved_expression and entry.get("download_after_freeze"):
        after = entry["download_after_freeze"]
        download["supplementary"] = list(download.get("supplementary", [])) + list(
            after.get("supplementary", [])
        )
    if "series_matrix" in download:
        dest = acc_dir / f"{accession}_series_matrix.txt.gz"
        try:
            paths["series_matrix"] = download_file(download["series_matrix"], dest, force=force)
        except RuntimeError as exc:
            print(f"Warning: {exc}")
    for url in download.get("supplementary", []):
        name = url.rstrip("/").split("/")[-1]
        dest = acc_dir / name
        try:
            paths[name] = download_file(url, dest, force=force)
        except RuntimeError as exc:
            print(f"Warning: {exc}")
    # Also index any already-present files in the accession directory.
    if acc_dir.exists():
        for path in acc_dir.iterdir():
            if path.is_file():
                paths.setdefault(path.name, path)
    return paths


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _base_meta(
    *,
    accession: str,
    sample_id: str,
    platform_id: str,
    modality: str,
    disease_label: str,
    keloid_vs_normal: str,
    scar_type: str,
    patient_id: str,
    task: str,
    response: str,
    contrast_type: str,
    cell_type: str = "bulk_tissue",
    lesional_status: str = "unknown",
    treatment: str = "none",
    cohort_role: str = "development",
    allow_accession_level_grouping: bool = False,
) -> dict:
    prompts = {
        "keloid_vs_normal": "Given this expression profile, predict whether the sample is keloid or normal. Answer:",
        "scar_differential": "Given this scar expression profile, predict the scar type. Answer:",
        "stiffness_response": "Given this fibroblast expression profile, predict the ECM stiffness condition. Answer:",
        "wound_susceptibility": "Given this skin expression profile, predict keloid susceptibility / wound timepoint. Answer:",
    }
    return {
        "sample_id": sample_id,
        "accession": accession,
        "sample_title": sample_id,
        "platform_id": platform_id,
        "modality": modality,
        "source_dataset": "public_keloid",
        "disease_domain": "keloid",
        "disease_label": disease_label,
        "keloid_vs_normal": keloid_vs_normal,
        "lesional_status": lesional_status,
        "scar_type": scar_type,
        "cell_type": cell_type,
        "treatment": treatment,
        "patient_id": patient_id,
        "contrast_type": contrast_type,
        "eligible_for_keloid_pretraining": cohort_role != "lockbox",
        "encoder_task": task,
        "encoder_prompt": prompts.get(task, "Given this expression profile, predict the biological state. Answer:"),
        "encoder_response": response,
        "cohort_role": cohort_role,
        "allow_accession_level_grouping": allow_accession_level_grouping,
        "lockbox": cohort_role == "lockbox",
    }


def _infer_binary_from_text(text: str) -> tuple[str, str, str] | None:
    t = text.lower()
    if "immature" in t and "scar" in t:
        return "immature_scar", "non_keloid", "immature_scar"
    if "hypertrophic" in t:
        return "hypertrophic_scar", "non_keloid", "hypertrophic_scar"
    if "normotrophic" in t:
        return "normotrophic_scar", "non_keloid", "normotrophic_scar"
    if "keloid" in t or re.search(r"\bkdf\b", t) or re.search(r"\bkf\b", t):
        return "keloid", "keloid", "keloid"
    if "normal" in t or "control" in t or re.search(r"\bndf\b", t) or re.search(r"\bnf\b", t):
        return "normal", "normal", "normal"
    return None


def process_series_matrix_cohort(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    path = paths.get("series_matrix")
    if path is None or not path.exists():
        return None
    series = parse_series_matrix(path)
    if series.expression.empty or series.expression.shape[0] == 0 or series.expression.shape[1] == 0:
        raise RuntimeError(f"{entry['accession']}: empty series matrix expression table")
    probe_map = load_platform_mapping(series.platform_id, PLATFORM_DIR, allow_download=True)
    gene_expr = expression_to_genes(series.expression, probe_map, series.platform_id)
    # expression_to_genes returns samples x genes.
    if gene_expr.shape[0] == 0:
        raise RuntimeError(f"{entry['accession']}: no gene-mapped expression rows")
    gene_expr = gene_expr.reset_index(drop=True)
    meta = series.metadata.reset_index(drop=True)
    if len(meta) != len(gene_expr):
        # Align on sample_id order from expression columns when series matrix was transposed.
        if list(series.expression.columns) == list(meta["sample_id"]):
            gene_expr.index = meta["sample_id"].tolist()
            gene_expr = gene_expr.reset_index(drop=True)
        else:
            raise RuntimeError(
                f"{entry['accession']}: metadata/expression length mismatch "
                f"(meta={len(meta)}, expr={len(gene_expr)}, raw={series.expression.shape})"
            )

    rows = []
    for idx, row in meta.iterrows():
        text_bits = " ".join(str(v) for v in row.values if pd.notna(v))
        inferred = _infer_binary_from_text(text_bits)
        patient = str(row.get("patient_id", row.get("donor", "unknown")))
        if patient.lower() in {"nan", "none", ""}:
            patient = "unknown"
        # Try title-based donor ids for GSE121618-like naming.
        title = str(row.get("title", row.get("sample_title", row["sample_id"])))
        donor_match = re.search(r"(\d{3,4}[A-Za-z]+|\b[KN]\d+\b|donor[_\s-]?\d+)", title, re.I)
        if patient == "unknown" and donor_match:
            patient = _slug(donor_match.group(1))
        if inferred is None:
            disease, kn, scar = "unknown", "unknown", "unknown"
            task, response = "keloid_vs_normal", "unknown"
        else:
            disease, kn, scar = inferred
            if disease in {"immature_scar", "hypertrophic_scar", "normotrophic_scar"}:
                task, response = "scar_differential", disease
            else:
                task, response = "keloid_vs_normal", disease
        rows.append(
            _base_meta(
                accession=entry["accession"],
                sample_id=str(row["sample_id"]),
                platform_id=series.platform_id,
                modality=entry.get("modality", "microarray"),
                disease_label=disease,
                keloid_vs_normal=kn if kn in {"keloid", "normal"} else ("keloid" if disease == "keloid" else "unknown"),
                scar_type=scar,
                patient_id=patient,
                task=task,
                response=response,
                contrast_type="sample",
                cohort_role=entry.get("role", "development").replace("development_optional", "development"),
                allow_accession_level_grouping=bool(entry.get("allow_accession_level_grouping", False)),
            )
        )
    metadata = pd.DataFrame(rows)
    # Keep only labeled samples for primary use; retain unknown as excluded later.
    coverage = module_coverage_report(gene_expr, accession=entry["accession"])
    if coverage["all_zero_programs"]:
        raise RuntimeError(f"{entry['accession']}: all program features zero after mapping")
    summary = {
        "accession": entry["accession"],
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(gene_expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": entry.get("role"),
        "labels": metadata["encoder_response"].value_counts(dropna=False).to_dict(),
    }
    return metadata, gene_expr, summary


def process_gse113619(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    path = None
    for key, candidate in paths.items():
        if "normalized" in key.lower() or ("raw" in key.lower() and key.endswith(".csv.gz")):
            path = candidate
            break
    if path is None:
        return process_series_matrix_cohort(entry, paths)
    df = pd.read_csv(path, compression="infer")
    gene_col = df.columns[0]
    df = df.rename(columns={gene_col: "gene_id"})
    ids = df["gene_id"].astype(str).str.strip()
    # First column is Entrez Gene IDs (1, 10, 100, ...), not Ensembl/symbols.
    if ids.str.fullmatch(r"\d+").mean() > 0.8:
        entrez_map = fetch_entrez_symbols(ids.tolist())
        mapped = [entrez_map.get(i, "") for i in ids.tolist()]
        n_mapped = sum(1 for g in mapped if g)
        if n_mapped < 100:
            raise RuntimeError(f"GSE113619: Entrez→symbol map too sparse ({n_mapped}/{len(mapped)})")
    else:
        ensembl_map = load_ensembl_symbol_map()
        mapped = []
        for raw in ids.tolist():
            key = strip_ensembl_version(raw)
            mapped.append(ensembl_map.get(key) or normalize_gene_symbol(raw) or "")
    df["gene"] = [normalize_gene_symbol(g) if g else None for g in mapped]
    df = df.dropna(subset=["gene"])
    if df.empty:
        raise RuntimeError("GSE113619: no genes remained after ID mapping")
    value_cols = [c for c in df.columns if c not in {"gene", "gene_id"}]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = df.groupby("gene")[value_cols].mean().T
    if expr.size == 0:
        raise RuntimeError("GSE113619: empty expression after mapping")
    if float(np.nanmax(expr.to_numpy())) > 100:
        expr = np.log1p(expr.clip(lower=0))
    rows = []
    for sample_id in expr.index.astype(str):
        # Names like K4-1st / N1-2nd
        m = re.match(r"^([KN])(\d+)-(1st|2nd)", sample_id)
        if not m:
            m = re.match(r"^([KN])(\d+)_(1st|2nd)", sample_id)
        if m:
            group, donor_num, timepoint = m.group(1), m.group(2), m.group(3)
            prone = group == "K"
            patient = f"{group}{donor_num}"
            disease = "keloid_prone" if prone else "control"
            kn = "unknown"  # susceptibility-only; never enter keloid tissue binary
            response = f"{disease}_{'day0' if timepoint == '1st' else 'day42'}"
            task = "wound_susceptibility"
        else:
            patient, disease, kn, response, task = "unknown", "unknown", "unknown", "unknown", "wound_susceptibility"
        rows.append(
            _base_meta(
                accession="GSE113619",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL16791"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type="unknown",
                patient_id=patient,
                task=task,
                response=response,
                contrast_type="wound_timepoint",
                treatment="none",
                cohort_role="development",
            )
        )
    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE113619")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE113619: all program features zero after Entrez→symbol mapping")
    summary = {
        "accession": "GSE113619",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "gene_id_type": "entrez",
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse246562(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """KDF/NDF stiffness RNA-seq; keep only sample columns like 'K1 8kPa' / 'N3 214kPa'."""
    count_path = None
    for key, path in paths.items():
        if "Raw_counts" in key or "counts" in key.lower():
            count_path = path
            break
    if count_path is None:
        return process_series_matrix_cohort(entry, paths)
    df = pd.read_csv(count_path, compression="infer")
    gene_col = df.columns[0]
    df = df.rename(columns={gene_col: "gene"})
    df["gene"] = df["gene"].map(normalize_gene_symbol)
    df = df.dropna(subset=["gene"])
    sample_re = re.compile(r"^[KN]\d+\s+\d+\s*kPa$", re.I)
    # Strip trailing spaces that GEO count matrices sometimes leave on headers.
    rename = {c: str(c).strip() for c in df.columns if c != "gene"}
    df = df.rename(columns=rename)
    value_cols = [c for c in df.columns if c != "gene" and sample_re.match(str(c))]
    if len(value_cols) < 2:
        raise RuntimeError(f"GSE246562: expected K*/N* kPa sample columns, found {df.columns.tolist()}")
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = np.log1p(df.groupby("gene")[value_cols].mean().T.clip(lower=0))
    rows = []
    for sample_id in expr.index.astype(str):
        text = sample_id.strip()
        m = re.match(r"^([KN])(\d+)\s+(\d+)\s*kPa$", text, re.I)
        if not m:
            continue
        group, donor_num, kpa = m.group(1).upper(), m.group(2), m.group(3)
        patient = f"{group}{donor_num}"
        stiffness = f"{kpa}kPa"
        disease, kn = ("keloid", "keloid") if group == "K" else ("normal", "normal")
        rows.append(
            _base_meta(
                accession="GSE246562",
                sample_id=text,
                platform_id=entry.get("platform_id", "GPL18573"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease,
                patient_id=patient,
                task="stiffness_response",
                response=stiffness,
                contrast_type="stiffness_pair",
                cell_type="fibroblast",
                treatment=stiffness,
                cohort_role="development",
            )
        )
    if not rows:
        raise RuntimeError("GSE246562: no labeled stiffness samples after column filter")
    metadata = pd.DataFrame(rows)
    expr = expr.loc[metadata["sample_id"].tolist()].reset_index(drop=True)
    coverage = module_coverage_report(expr, accession="GSE246562")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE246562: all program features zero after sample-column filter")
    summary = {
        "accession": "GSE246562",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr, summary


def process_gse121618(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Keloid endothelial cells (KECs) vs normal endothelial cells (NECs)."""
    path = paths.get("series_matrix")
    if path is None or not path.exists():
        return None
    series = parse_series_matrix(path)
    if series.expression.empty:
        raise RuntimeError("GSE121618: empty series matrix expression table")
    probe_map = load_platform_mapping(series.platform_id, PLATFORM_DIR, allow_download=True)
    gene_expr = expression_to_genes(series.expression, probe_map, series.platform_id).reset_index(drop=True)
    meta = series.metadata.reset_index(drop=True)
    if len(meta) != len(gene_expr):
        raise RuntimeError(f"GSE121618: meta/expr mismatch ({len(meta)} vs {len(gene_expr)})")
    rows = []
    for _, row in meta.iterrows():
        title = str(row.get("title", row["sample_id"]))
        source = str(row.get("source_name_ch1", row.get("source_name", ""))).lower()
        tissue_bits = " ".join(
            str(row.get(c, "")) for c in row.index if "characteristic" in str(c).lower() or c == "title"
        ).lower()
        if "kec" in source or "keloid" in tissue_bits:
            disease, kn, response = "keloid", "keloid", "keloid"
        elif "nec" in source or ("skin tissue" in tissue_bits and "keloid" not in tissue_bits):
            disease, kn, response = "normal", "normal", "normal"
        else:
            disease, kn, response = "unknown", "unknown", "unknown"
        # Source_name is authoritative when present (KECs / NECs).
        if "kec" in source:
            disease, kn, response = "keloid", "keloid", "keloid"
        elif "nec" in source:
            disease, kn, response = "normal", "normal", "normal"
        donor_m = re.search(r"(\d{3,4})", title)
        patient = _slug(donor_m.group(1) + ("K" if disease == "keloid" else "N")) if donor_m else _slug(title)
        rows.append(
            _base_meta(
                accession="GSE121618",
                sample_id=str(row["sample_id"]),
                platform_id=series.platform_id,
                modality="microarray",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=patient,
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="endothelial",
                cohort_role="development",
            )
        )
    metadata = pd.DataFrame(rows)
    classes = set(metadata["keloid_vs_normal"])
    if not ({"keloid"} & classes) or not ({"normal"} & classes):
        raise RuntimeError(
            f"GSE121618: expected both classes, got {metadata['encoder_response'].value_counts().to_dict()}"
        )
    coverage = module_coverage_report(gene_expr, accession="GSE121618")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE121618: all program features zero after mapping")
    summary = {
        "accession": "GSE121618",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(gene_expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, gene_expr, summary


def process_gse173900(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Asian keloid vs control tissue; KC*=control, KL*=keloid."""
    count_path = None
    for key, path in paths.items():
        if "gene_count_matrix" in key or "count" in key.lower():
            count_path = path
            break
    if count_path is None or not count_path.exists():
        raise RuntimeError("GSE173900: gene count matrix missing")
    df = pd.read_csv(count_path, compression="infer")
    gene_col = df.columns[0]
    df = df.rename(columns={gene_col: "gene"})
    ensembl_map = load_ensembl_symbol_map()
    mapped = []
    for raw in df["gene"].astype(str):
        key = strip_ensembl_version(raw)
        mapped.append(ensembl_map.get(key) or normalize_gene_symbol(raw) or "")
    df["gene"] = [g if g else None for g in mapped]
    df = df.dropna(subset=["gene"])
    value_cols = [c for c in df.columns if c != "gene" and re.match(r"^(KC|KL)\d+$", str(c), re.I)]
    if len(value_cols) < 2:
        raise RuntimeError(f"GSE173900: expected KC*/KL* columns, got {df.columns.tolist()}")
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = np.log1p(df.groupby("gene")[value_cols].mean().T.clip(lower=0))
    # Series matrix maps titles → tissue labels.
    title_to_tissue: dict[str, str] = {}
    if "series_matrix" in paths:
        try:
            series = parse_series_matrix(paths["series_matrix"])
            for _, row in series.metadata.iterrows():
                title = str(row.get("title", ""))
                m = re.search(r"\b(KC|KL)\d+\b", title, re.I)
                text = " ".join(str(v) for v in row.values if pd.notna(v)).lower()
                if m:
                    title_to_tissue[m.group(0).upper()] = "keloid" if "keloid" in text else "control"
        except Exception:  # noqa: BLE001
            title_to_tissue = {}
    rows = []
    for sample_id in expr.index.astype(str):
        sid = sample_id.upper()
        tissue = title_to_tissue.get(sid)
        if tissue is None:
            tissue = "control" if sid.startswith("KC") else ("keloid" if sid.startswith("KL") else "unknown")
        if tissue == "keloid":
            disease, kn, response = "keloid", "keloid", "keloid"
        elif tissue == "control":
            disease, kn, response = "normal", "normal", "normal"
        else:
            disease, kn, response = "unknown", "unknown", "unknown"
        rows.append(
            _base_meta(
                accession="GSE173900",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL21697"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=_slug(sample_id),
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="bulk_tissue",
                cohort_role=entry.get("role", "development").replace("development_optional", "development"),
            )
        )
    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE173900")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE173900: all program features zero")
    summary = {
        "accession": "GSE173900",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": entry.get("role", "development"),
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse190626(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """3 keloid + 3 normal tissue TPM matrix (xlsx)."""
    xlsx_path = None
    for key, path in paths.items():
        if key.endswith(".xlsx") or "gene_expression" in key.lower():
            xlsx_path = path
            break
    if xlsx_path is None or not xlsx_path.exists():
        raise RuntimeError("GSE190626: gene_expression xlsx missing")
    df = pd.read_excel(xlsx_path)
    gene_col = "gene_symbol" if "gene_symbol" in df.columns else df.columns[0]
    tpm_cols = [c for c in df.columns if str(c).lower().startswith("tpm_")]
    if len(tpm_cols) < 2:
        raise RuntimeError(f"GSE190626: no TPM columns in {df.columns.tolist()}")
    genes = df[gene_col].map(lambda g: normalize_gene_symbol(str(g))).astype(str)
    mat = df[tpm_cols].apply(pd.to_numeric, errors="coerce")
    mat.index = genes
    expr = np.log1p(mat.groupby(level=0).mean().T.clip(lower=0))
    # Rename tpm_Keloid1 → Keloid1
    expr.index = [re.sub(r"^tpm_", "", str(i), flags=re.I) for i in expr.index]
    rows = []
    for sample_id in expr.index.astype(str):
        low = sample_id.lower()
        if "keloid" in low:
            disease, kn, response = "keloid", "keloid", "keloid"
        elif "normal" in low:
            disease, kn, response = "normal", "normal", "normal"
        else:
            disease, kn, response = "unknown", "unknown", "unknown"
        rows.append(
            _base_meta(
                accession="GSE190626",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=_slug(sample_id),
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="bulk_tissue",
                cohort_role=entry.get("role", "development").replace("development_optional", "development"),
            )
        )
    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE190626")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE190626: all program features zero")
    summary = {
        "accession": "GSE190626",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": entry.get("role", "development"),
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse185309(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Breakthrough lockbox: keloid keratinocytes (KK*) vs normal keratinocytes (NK*)."""
    count_path = None
    for key, path in paths.items():
        if "counts_keratinocytes" in key or "counts" in key.lower():
            count_path = path
            break
    if count_path is None or not count_path.exists():
        return None, None, {
            "accession": "GSE185309",
            "status": "skipped_reserved_until_freeze",
            "role": "lockbox",
            "reason": "counts_keratinocytes matrix missing",
        }
    df = pd.read_csv(count_path, compression="infer")
    gene_col = df.columns[0]
    df = df.rename(columns={gene_col: "gene"})
    ensembl_map = load_ensembl_symbol_map()
    mapped = [
        ensembl_map.get(strip_ensembl_version(str(g))) or normalize_gene_symbol(str(g)) or None
        for g in df["gene"].tolist()
    ]
    df["gene"] = mapped
    df = df.dropna(subset=["gene"])
    value_cols = [c for c in df.columns if c != "gene" and re.match(r"^(KK|NK)_\d+", str(c), re.I)]
    if len(value_cols) < 2:
        raise RuntimeError(f"GSE185309: expected KK*/NK* columns, got {df.columns.tolist()}")
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = np.log1p(df.groupby("gene")[value_cols].mean().T.clip(lower=0))
    # Clean sample IDs: KK_1_sorted.bam -> KK_1
    expr.index = [re.sub(r"_sorted\.bam$", "", str(i), flags=re.I) for i in expr.index]
    rows = []
    for sample_id in expr.index.astype(str):
        sid = sample_id.upper()
        if sid.startswith("KK"):
            disease, kn, response = "keloid", "keloid", "keloid"
        elif sid.startswith("NK"):
            disease, kn, response = "normal", "normal", "normal"
        else:
            disease, kn, response = "unknown", "unknown", "unknown"
        donor_m = re.search(r"(KK|NK)_(\d+)", sample_id, re.I)
        patient = f"{donor_m.group(1).upper()}_{donor_m.group(2)}" if donor_m else _slug(sample_id)
        rows.append(
            _base_meta(
                accession="GSE185309",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=patient,
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="keratinocyte",
                cohort_role="lockbox",
            )
        )
    metadata = pd.DataFrame(rows)
    classes = set(metadata["keloid_vs_normal"])
    if not ({"keloid"} & classes) or not ({"normal"} & classes):
        raise RuntimeError(f"GSE185309: expected both classes, got {metadata['encoder_response'].value_counts().to_dict()}")
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE185309")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE185309: all program features zero")
    summary = {
        "accession": "GSE185309",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "lockbox",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr.reset_index(drop=True), summary


def _read_gse212954_expression_table(xlsx_path: Path) -> pd.DataFrame:
    """Parse GSE212954 Expression_Gene.xlsx (header buried under legend rows)."""
    raw = pd.read_excel(xlsx_path, header=None)
    header_idx = None
    for i in range(min(30, len(raw))):
        vals = [str(v).strip() for v in raw.iloc[i].tolist()]
        if "Track_id" in vals and "Gene_Name" in vals:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("GSE212954: could not locate Track_id/Gene_Name header row")
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_idx].tolist()]
    df = df.dropna(how="all")
    return df


def process_gse212954(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Lockbox keloid center/margin vs normal; expression from Gene xlsx."""
    xlsx_path = None
    for key, path in paths.items():
        if "Expression_Gene" in key or key.endswith(".xlsx"):
            xlsx_path = path
            break
    if xlsx_path is None or not xlsx_path.exists():
        # Expression is reserved until protocol freeze; metadata-only downloads are allowed earlier.
        return None, None, {
            "accession": "GSE212954",
            "status": "skipped_reserved_until_freeze",
            "role": "lockbox",
            "reason": "Expression_Gene.xlsx not present; download only after nested protocol freeze",
        }
    df = _read_gse212954_expression_table(xlsx_path)
    # Prefer HGNC Gene_Name; fall back to Ensembl Track_id (with Gencode-style versions).
    if "Gene_Name" not in df.columns:
        raise RuntimeError(f"GSE212954: Gene_Name missing; columns={df.columns.tolist()}")
    sample_cols = []
    for c in df.columns:
        if c in {"Track_id", "Gene_Name", "Locus", "Strand", "Gene_Type"}:
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
            sample_cols.append(c)
    if len(sample_cols) < 2:
        raise RuntimeError(f"GSE212954: could not find sample columns in {df.columns.tolist()[:20]}")
    ensembl_map = load_ensembl_symbol_map()
    mapped = []
    for _, row in df.iterrows():
        sym = normalize_gene_symbol(row.get("Gene_Name"))
        if not sym and "Track_id" in df.columns:
            key = strip_ensembl_version(str(row.get("Track_id")))
            sym = ensembl_map.get(key) or normalize_gene_symbol(key)
        mapped.append(sym)
    mat = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    mat.index = mapped
    mat = mat.loc[[g for g in mat.index if g]]
    # Values are FPKM; still non-negative. log1p for parity with other bulk cohorts.
    expr = np.log1p(mat.groupby(level=0).mean().T.clip(lower=0))
    # Align names to series titles when possible.
    title_by_token: dict[str, tuple[str, str]] = {}
    if "series_matrix" in paths:
        try:
            series = parse_series_matrix(paths["series_matrix"])
            for _, row in series.metadata.iterrows():
                title = str(row.get("title", row["sample_id"]))
                text = " ".join(str(v) for v in row.values if pd.notna(v)).lower()
                if "normal" in text:
                    label = "normal"
                elif "keloid" in text:
                    label = "keloid"
                else:
                    label = "unknown"
                donor_m = re.search(r"([NMC])-?K(\d+)", title, re.I)
                donor = f"K{donor_m.group(2)}" if donor_m else _slug(title)
                title_by_token[title] = (label, donor)
                title_by_token[_slug(title)] = (label, donor)
        except Exception:  # noqa: BLE001
            title_by_token = {}
    rows = []
    for sample_id in expr.index.astype(str):
        key = sample_id if sample_id in title_by_token else _slug(sample_id)
        if key in title_by_token:
            label, donor = title_by_token[key]
        else:
            low = sample_id.lower()
            if low.startswith("n") or "normal" in low:
                label, donor = "normal", _slug(sample_id)
            elif "keloid" in low or low.startswith("c") or low.startswith("m"):
                label, donor = "keloid", _slug(sample_id)
            else:
                label, donor = "unknown", _slug(sample_id)
        disease = label
        kn = label if label in {"keloid", "normal"} else "unknown"
        rows.append(
            _base_meta(
                accession="GSE212954",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL20301"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=donor,
                task="keloid_vs_normal",
                response=disease if disease != "unknown" else "unknown",
                contrast_type="sample",
                cell_type="bulk_tissue",
                cohort_role="lockbox",
            )
        )
    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE212954")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE212954: all program features zero")
    summary = {
        "accession": "GSE212954",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "lockbox",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse245660(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Sample-level RNA-seq from GSE245660_RAW.tar (series matrix has no expression)."""
    tar_path = None
    for key, path in paths.items():
        if key.endswith(".tar") or path.name.endswith("_RAW.tar"):
            tar_path = path
            break
    if tar_path is None or not tar_path.exists():
        raise RuntimeError("GSE245660: RAW.tar missing; DEG CSV is not a sample matrix")

    series_titles: dict[str, str] = {}
    if "series_matrix" in paths:
        try:
            series = parse_series_matrix(paths["series_matrix"])
            for _, row in series.metadata.iterrows():
                series_titles[str(row["sample_id"])] = str(row.get("title", row["sample_id"]))
        except Exception:  # noqa: BLE001
            series_titles = {}

    sample_exprs: dict[str, pd.Series] = {}
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.isfile() or "expression_values" not in member.name:
                continue
            base = Path(member.name).name
            gsm_m = re.match(r"^(GSM\d+)_", base)
            if not gsm_m:
                continue
            gsm = gsm_m.group(1)
            handle = tf.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            if base.endswith(".gz") or raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            df = pd.read_csv(io.BytesIO(raw))
            gene_col = "Name" if "Name" in df.columns else df.columns[0]
            value_col = "TPM" if "TPM" in df.columns else ("RPKM" if "RPKM" in df.columns else "Total exon reads")
            if value_col not in df.columns:
                raise RuntimeError(f"GSE245660: no expression column in {base}")
            genes = df[gene_col].map(lambda g: normalize_gene_symbol(str(g))).astype(str)
            values = pd.to_numeric(df[value_col], errors="coerce")
            series_vals = pd.Series(values.to_numpy(), index=genes).groupby(level=0).mean()
            sample_exprs[gsm] = series_vals

    if not sample_exprs:
        raise RuntimeError("GSE245660: no sample expression tables in RAW.tar")

    expr = pd.DataFrame(sample_exprs).T.fillna(0.0)
    if float(np.nanmax(expr.to_numpy())) > 100:
        expr = np.log1p(expr.clip(lower=0))

    rows = []
    for sample_id in expr.index.astype(str):
        title = series_titles.get(sample_id, sample_id)
        text = f"{sample_id} {title}".lower()
        if "immature" in text:
            disease, kn, scar, response = "immature_scar", "unknown", "immature_scar", "immature_scar"
        elif "keloid" in text:
            disease, kn, scar, response = "keloid", "keloid", "keloid", "keloid"
        else:
            disease, kn, scar, response = "unknown", "unknown", "unknown", "unknown"
        # Donor from title tokens s1/k1 etc.
        donor_m = re.search(r"\b([sk]\d+)\b", title.lower()) or re.search(r"_(s\d+|k\d+)_", sample_id.lower())
        patient = _slug(donor_m.group(1)) if donor_m else sample_id
        rows.append(
            _base_meta(
                accession="GSE245660",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=scar,
                patient_id=patient,
                task="scar_differential",
                response=response,
                contrast_type="sample",
                cohort_role="development",
            )
        )
    # Fix keloid labels using member filenames when title parse failed.
    with tarfile.open(tar_path) as tf:
        name_by_gsm = {}
        for member in tf.getmembers():
            gsm_m = re.match(r"^(GSM\d+)_(.+)$", Path(member.name).name)
            if gsm_m:
                name_by_gsm[gsm_m.group(1)] = gsm_m.group(2).lower()
    for i, row in enumerate(rows):
        gsm = row["sample_id"]
        fname = name_by_gsm.get(gsm, "")
        if rows[i]["disease_label"] == "unknown":
            if fname.startswith("k") or "_k" in f"_{fname}":
                rows[i].update(
                    {
                        "disease_label": "keloid",
                        "keloid_vs_normal": "keloid",
                        "scar_type": "keloid",
                        "encoder_response": "keloid",
                    }
                )
            elif fname.startswith("s") or "_s" in f"_{fname}":
                rows[i].update(
                    {
                        "disease_label": "immature_scar",
                        "keloid_vs_normal": "unknown",
                        "scar_type": "immature_scar",
                        "encoder_response": "immature_scar",
                    }
                )
        if rows[i]["patient_id"] in {gsm, "unknown"}:
            dm = re.search(r"(^|_)([sk]\d+)(_|$)", fname)
            if dm:
                rows[i]["patient_id"] = _slug(dm.group(2))

    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE245660")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE245660: all program features zero after RAW.tar ingest")
    summary = {
        "accession": "GSE245660",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "source": "RAW.tar",
    }
    return metadata, expr.reset_index(drop=True), summary


def _gse191067_label(title: str) -> tuple[str, str, str, str]:
    """Map sample titles HK / HK-NS / HNS / HNSR to disease labels."""
    t = title.strip().upper()
    if t.startswith("HK-NS") or t.startswith("HKNS"):
        return "adjacent_normal", "adjacent_normal", "adjacent_normal", "adjacent_normal"
    if re.fullmatch(r"HK\d+", t):
        return "keloid", "keloid", "keloid", "keloid"
    if t.startswith("HNSR"):
        return "normal_scar", "normal_scar", "normal_scar", "normal_scar"
    if t.startswith("HNS"):
        return "normal", "normal", "normal", "normal"
    return "unknown", "unknown", "unknown", "unknown"


def process_gse191067(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Donor/sample-level pseudobulk from the all-cells UMI matrix.

    Matrix is genes × ~1.8e5 barcodes. One streaming pass accumulates per-sample
    UMI sums in gene chunks to keep peak memory manageable. Cached under
    data/processed/public_keloid/cache/ after the first successful pass.
    """
    umi_path = None
    for key, path in paths.items():
        if "UMI.matrix" in key or "UMI" in key:
            umi_path = path
            break
    if umi_path is None or not umi_path.exists():
        raise RuntimeError("GSE191067: UMI matrix missing")

    cache_dir = OUT_DIR / "cache" / "GSE191067"
    cache_meta = cache_dir / "metadata.parquet"
    cache_expr = cache_dir / "expression.parquet"
    cache_summary = cache_dir / "summary.json"
    if cache_meta.exists() and cache_expr.exists() and cache_summary.exists():
        print("GSE191067: loading cached donor/sample pseudobulk")
        metadata = pd.read_parquet(cache_meta)
        expr = pd.read_parquet(cache_expr)
        summary = json.loads(cache_summary.read_text())
        summary["source"] = "UMI.matrix_pseudobulk_cache"
        return metadata, expr, summary

    header = pd.read_csv(umi_path, compression="infer", nrows=0)
    gene_col = str(header.columns[0])
    barcode_cols = [str(c) for c in header.columns if str(c) != gene_col]
    prefix_re = re.compile(r"^(HK-NS\d+|HNSR\d+|HK\d+|HNS\d+)_", re.I)
    groups: dict[str, list[str]] = {}
    for col in barcode_cols:
        m = prefix_re.match(col)
        if not m:
            continue
        title = m.group(1)
        # Normalize case while keeping HK-NS hyphenation.
        if title.upper().startswith("HK-NS"):
            title = "HK-NS" + re.sub(r"^HK-NS", "", title, flags=re.I)
        else:
            title = title.upper()
        groups.setdefault(title, []).append(col)

    if len(groups) < 2:
        raise RuntimeError(f"GSE191067: could not parse sample prefixes from barcodes (n_barcodes={len(barcode_cols)})")

    sample_ids = sorted(groups)
    sample_index = {s: i for i, s in enumerate(sample_ids)}
    ensembl_map = load_ensembl_symbol_map()
    gene_names: list[str] = []
    sum_blocks: list[np.ndarray] = []
    print(f"GSE191067: streaming pseudobulk for {len(sample_ids)} samples / {len(barcode_cols)} cells")

    for chunk in pd.read_csv(umi_path, compression="infer", chunksize=800):
        raw_genes = chunk[gene_col].astype(str).tolist()
        mapped = [
            ensembl_map.get(strip_ensembl_version(g)) or normalize_gene_symbol(g) or strip_ensembl_version(g)
            for g in raw_genes
        ]
        gene_names.extend(mapped)
        acc = np.zeros((len(chunk), len(sample_ids)), dtype=np.float64)
        for sample, cols in groups.items():
            j = sample_index[sample]
            acc[:, j] = chunk[cols].to_numpy(dtype=np.float64, copy=False).sum(axis=1)
        sum_blocks.append(acc)

    gene_sums = np.vstack(sum_blocks) if sum_blocks else np.zeros((0, len(sample_ids)))
    pb = pd.DataFrame(gene_sums, index=gene_names, columns=sample_ids)
    pb = pb.groupby(level=0).sum()
    expr = np.log1p(pb.clip(lower=0).T)

    rows = []
    for title in expr.index.astype(str):
        disease, kn, scar, response = _gse191067_label(title)
        rows.append(
            _base_meta(
                accession="GSE191067",
                sample_id=title,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="scrna_pseudobulk",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=scar,
                patient_id=_slug(title),
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="all_cells_pseudobulk",
                cohort_role="lockbox",
                allow_accession_level_grouping=False,
            )
        )
    metadata = pd.DataFrame(rows)
    expr = expr.loc[metadata["sample_id"].tolist()].reset_index(drop=True)

    coverage = module_coverage_report(expr, accession="GSE191067")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE191067: all program features zero after UMI pseudobulk")
    summary = {
        "accession": "GSE191067",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "lockbox",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "n_cells_per_sample": {k: len(v) for k, v in groups.items()},
        "source": "UMI.matrix_pseudobulk",
        "note": "Barcode prefixes (HNS1/HNS2/HNS3) used as sample IDs; GEO titles HNS3/HNS4/HNS6 differ.",
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(cache_meta, index=False)
    expr.to_parquet(cache_expr, index=False)
    cache_summary.write_text(json.dumps(summary, indent=2))
    print(f"GSE191067: cached pseudobulk -> {cache_dir}")
    return metadata, expr, summary


def _is_sample_column(name: str) -> bool:
    text = str(name).strip()
    low = text.lower()
    if low in {"gene", "gene_id", "genes", "symbol", "id", "ensembl", "gene_dbxref", "description", "go_id", "go_term", "pathway", "pathway_description", "tf_family"}:
        return False
    if low.startswith(("go_", "path", "desc", "tf_", "gene_")):
        return False
    # Keep typical sample-like names.
    if re.search(r"(ctrl|control|keloid|normal|scar|donor|patient|kdf|ndf|sample)", low):
        return True
    # GSM accessions
    if re.match(r"^GSM\d+$", text, re.I):
        return True
    # Generic short sample tokens with digits
    if re.match(r"^[A-Za-z]+\d+$", text):
        return True
    return False


def process_count_matrix_cohort(
    entry: dict,
    paths: dict[str, Path],
    *,
    gene_col_guess: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    count_path = None
    for key, path in paths.items():
        if key == "series_matrix":
            continue
        if path.suffixes[-2:] == [".txt", ".gz"] or path.name.endswith(".csv.gz") or path.suffix == ".gz":
            count_path = path
            break
    if count_path is None:
        return process_series_matrix_cohort(entry, paths)
    # Detect delimiter.
    open_fn = gzip.open if str(count_path).endswith(".gz") else open
    with open_fn(count_path, "rt", errors="replace") as handle:
        first = handle.readline()
    sep = "\t" if first.count("\t") > first.count(",") else ","
    df = pd.read_csv(count_path, sep=sep, compression="infer")
    gene_col = gene_col_guess or df.columns[0]
    df = df.rename(columns={gene_col: "gene"})
    ensembl_map = load_ensembl_symbol_map()
    mapped = []
    for raw in df["gene"].astype(str):
        key = strip_ensembl_version(raw)
        mapped.append(ensembl_map.get(key) or normalize_gene_symbol(raw))
    df["gene"] = mapped
    df = df.dropna(subset=["gene"])
    value_cols = [c for c in df.columns if c != "gene" and _is_sample_column(c)]
    if not value_cols:
        # Fallback: numeric-only columns.
        for c in df.columns:
            if c == "gene":
                continue
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
                value_cols.append(c)
    if not value_cols:
        raise RuntimeError(f"{entry['accession']}: no sample columns found in count matrix")
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = np.log1p(df.groupby("gene")[value_cols].mean().T.clip(lower=0))
    # Metadata from series matrix if available.
    series_meta = None
    if "series_matrix" in paths:
        try:
            series_meta = parse_series_matrix(paths["series_matrix"]).metadata.set_index("sample_id")
        except Exception:  # noqa: BLE001
            series_meta = None
    rows = []
    for sample_id in expr.index.astype(str):
        text = sample_id
        if series_meta is not None and sample_id in series_meta.index:
            text = " ".join(str(v) for v in series_meta.loc[sample_id].values if pd.notna(v))
        inferred = _infer_binary_from_text(text) or _infer_binary_from_text(sample_id)
        if inferred is None:
            # Explicit Ctrl*/control naming common in count matrices.
            low = sample_id.lower()
            if low.startswith("ctrl") or "control" in low:
                inferred = ("normal", "normal", "normal")
            elif "keloid" in low:
                inferred = ("keloid", "keloid", "keloid")
        if inferred is None:
            disease, kn, scar = "unknown", "unknown", "unknown"
        else:
            disease, kn, scar = inferred
        patient = "unknown"
        donor_m = re.search(r"(donor[_\s-]?\d+|[KN]\d+|patient[_\s-]?\d+|ctrl\d+|keloid\d+)", text, re.I)
        if donor_m:
            patient = _slug(donor_m.group(1))
        elif re.match(r"^(ctrl|keloid|normal|kdf|ndf)\d+$", sample_id, re.I):
            patient = _slug(sample_id)
        # Small public cohorts often lack independent donor metadata; allow accession grouping when needed.
        allow_group = bool(entry.get("allow_accession_level_grouping", False))
        if patient == "unknown":
            allow_group = True
        kn_final = kn if kn in {"keloid", "normal", "non_keloid"} else ("keloid" if disease == "keloid" else ("normal" if disease == "normal" else "unknown"))
        rows.append(
            _base_meta(
                accession=entry["accession"],
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "unknown"),
                modality=entry.get("modality", "bulk_rnaseq"),
                disease_label=disease,
                keloid_vs_normal=kn_final,
                scar_type=scar,
                patient_id=patient,
                task=entry.get("native_task", "keloid_vs_normal"),
                response=disease if disease != "unknown" else "unknown",
                contrast_type="sample",
                cohort_role=entry.get("role", "development").replace("development_optional", "development"),
                allow_accession_level_grouping=allow_group,
            )
        )
    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession=entry["accession"])
    if coverage["all_zero_programs"]:
        raise RuntimeError(f"{entry['accession']}: all program features zero after mapping")
    # Pin program genes into vocab by ensuring columns exist.
    for gene in ALL_PINNED_GENES:
        if gene not in expr.columns:
            expr[gene] = 0.0
    summary = {
        "accession": entry["accession"],
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": entry.get("role"),
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse210434(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Scar fibroblast RNA-seq: normal scar / hypertrophic scar / keloid (n=9)."""
    tar_path = None
    for key, path in paths.items():
        if key.endswith(".tar") or path.name.endswith("_RAW.tar"):
            tar_path = path
            break
    if tar_path is None or not tar_path.exists():
        raise RuntimeError("GSE210434: RAW.tar missing")

    sample_exprs: dict[str, pd.Series] = {}
    label_by_gsm: dict[str, tuple[str, str, str]] = {}
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            base = Path(member.name).name
            m = re.match(r"^(GSM\d+)_([NHK])(\d+)\.txt\.gz$", base, re.I)
            if not m:
                continue
            gsm, code, donor_n = m.group(1), m.group(2).upper(), m.group(3)
            if code == "N":
                disease, scar = "normal_scar", "normal_scar"
            elif code == "H":
                disease, scar = "hypertrophic_scar", "hypertrophic_scar"
            else:
                disease, scar = "keloid", "keloid"
            handle = tf.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            df = pd.read_csv(io.BytesIO(raw), sep="\t")
            gene_col = "Gene symbol" if "Gene symbol" in df.columns else df.columns[0]
            value_col = [c for c in df.columns if c != gene_col][0]
            genes = df[gene_col].map(lambda g: normalize_gene_symbol(str(g)))
            values = pd.to_numeric(df[value_col], errors="coerce")
            series_vals = pd.Series(values.to_numpy(), index=genes).groupby(level=0).mean()
            series_vals = series_vals[series_vals.index.notna()]
            sample_exprs[gsm] = series_vals
            label_by_gsm[gsm] = (disease, scar, f"{code}{donor_n}")

    if len(sample_exprs) < 6:
        raise RuntimeError(f"GSE210434: expected 9 sample tables, got {len(sample_exprs)}")

    expr = pd.DataFrame(sample_exprs).T.fillna(0.0)
    if float(np.nanmax(expr.to_numpy())) > 100:
        expr = np.log1p(expr.clip(lower=0))

    rows = []
    for sample_id in expr.index.astype(str):
        disease, scar, donor = label_by_gsm[sample_id]
        kn = "keloid" if disease == "keloid" else "unknown"
        rows.append(
            _base_meta(
                accession="GSE210434",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=scar,
                patient_id=_slug(donor),
                task="scar_differential",
                response=disease,
                contrast_type="sample",
                cell_type="fibroblast",
                cohort_role="development",
            )
        )
    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE210434")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE210434: all program features zero")
    summary = {
        "accession": "GSE210434",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "source": "RAW.tar",
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse218007(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Affymetrix Clariom S RMA-GENE series matrix; fibroblast keloid vs control skin lockbox."""
    result = process_series_matrix_cohort(entry, paths)
    if result is None or result[0] is None:
        return result
    meta, expr, summary = result
    series = parse_series_matrix(paths["series_matrix"])
    title_by_id = series.metadata.set_index("sample_id")["title"].astype(str).to_dict() if "title" in series.metadata.columns else {}
    # Titles look like "Keloid NOD Sou (SOU22_CP)" / "Control Skin Maj (MAJ23_PAP)".
    donors = []
    for sid in meta["sample_id"].astype(str):
        title = title_by_id.get(sid, sid)
        m = re.search(r"\(([A-Za-z]{2,4}\d{2})_", title)
        donors.append(_slug(m.group(1)) if m else "unknown")
    meta = meta.copy()
    meta["patient_id"] = donors
    meta["cell_type"] = "fibroblast"
    meta["cohort_role"] = entry.get("role", "lockbox")
    meta["lockbox"] = meta["cohort_role"].astype(str).eq("lockbox")
    meta["eligible_for_keloid_pretraining"] = ~meta["lockbox"]
    summary = {
        **summary,
        "n_donors": int(meta["patient_id"].nunique()),
        "donors": sorted(meta["patient_id"].unique().tolist()),
        "note": "Fibroblast culture lockbox; score via fibroblast_keloid_binary, not tissue primary.",
    }
    return meta, expr, summary


def process_gse178562(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Human JUN fibroblast RNA-seq: HTS / keloid / normal skin / scar (exclude JUN-KO)."""
    path = None
    for key, candidate in paths.items():
        if "Jun_Human_RNA" in key or "Jun_Human_RNA" in candidate.name:
            path = candidate
            break
    if path is None:
        for candidate in paths.values():
            if "Human_RNA" in candidate.name:
                path = candidate
                break
    if path is None or not path.exists():
        raise RuntimeError("GSE178562: Jun_Human_RNA-seq matrix missing")

    # GEO matrix: header = sample columns; first column = RefSeq IDs (unnamed).
    df = pd.read_csv(path, sep="\t", compression="infer", index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    refseq = df.index.astype(str)
    keep = [c for c in df.columns if "KO" not in c.upper()]
    if len(keep) < 4:
        raise RuntimeError(f"GSE178562: expected non-KO human columns, got {df.columns.tolist()}")
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    ref_map = fetch_refseq_symbols(refseq.tolist())
    genes = [ref_map.get(r) or None for r in refseq.tolist()]
    mapped = df.copy()
    mapped["gene"] = genes
    mapped = mapped.dropna(subset=["gene"])
    expr = np.log1p(mapped.groupby("gene")[keep].mean().T.clip(lower=0))

    label_map = {
        "HTS": ("hypertrophic_scar", "unknown", "hypertrophic_scar"),
        "KLD": ("keloid", "keloid", "keloid"),
        "NS": ("normal", "normal", "normal"),
        "SCR": ("normal_scar", "unknown", "normal_scar"),
    }
    rows = []
    for sample_id in expr.index.astype(str):
        stem = sample_id.split(".")[0].upper()
        if stem not in label_map:
            continue
        disease, kn, scar = label_map[stem]
        rows.append(
            _base_meta(
                accession="GSE178562",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL16791"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=scar,
                patient_id=_slug(stem),  # technical-replicate grain; one donor/arm
                task="scar_differential",
                response=disease,
                contrast_type="sample",
                cell_type="fibroblast",
                cohort_role=entry.get("role", "development"),
                allow_accession_level_grouping=True,
            )
        )
    metadata = pd.DataFrame(rows)
    expr = expr.loc[metadata["sample_id"].tolist()].reset_index(drop=True)
    coverage = module_coverage_report(expr, accession="GSE178562")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE178562: all program features zero after RefSeq mapping")
    summary = {
        "accession": "GSE178562",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": entry.get("role", "development"),
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "note": "JUN non-KO human fibroblasts only; KO arms excluded; tech-reps share arm donor id.",
    }
    return metadata, expr, summary


def process_gse303591(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Bulk KDF vs NDF; keep untreated control (without AA) replicates only."""
    xlsx_path = None
    for key, path in paths.items():
        if "readcounts" in key.lower() or key.endswith(".xlsx"):
            xlsx_path = path
            break
    if xlsx_path is None or not xlsx_path.exists():
        raise RuntimeError("GSE303591: counts xlsx missing")
    df = pd.read_excel(xlsx_path, sheet_name="Counts")
    gene_col = "gene_symbol" if "gene_symbol" in df.columns else df.columns[0]
    # Prefer control columns; fall back to all KDF_/NDF_ if naming differs.
    value_cols = [c for c in df.columns if re.match(r"^(KDF|NDF)_C_\d+$", str(c), re.I)]
    if len(value_cols) < 4:
        value_cols = [c for c in df.columns if re.match(r"^(KDF|NDF)_", str(c), re.I) and "_AA_" not in str(c)]
    if len(value_cols) < 4:
        raise RuntimeError(f"GSE303591: expected KDF_C_*/NDF_C_* columns, got {df.columns.tolist()}")
    genes = df[gene_col].map(lambda g: normalize_gene_symbol(str(g)))
    mat = df[value_cols].apply(pd.to_numeric, errors="coerce")
    mat.index = genes
    mat = mat.loc[[g for g in mat.index if g]]
    expr = np.log1p(mat.groupby(level=0).mean().T.clip(lower=0))
    rows = []
    for sample_id in expr.index.astype(str):
        if sample_id.upper().startswith("KDF"):
            disease, kn, response = "keloid", "keloid", "keloid"
        elif sample_id.upper().startswith("NDF"):
            disease, kn, response = "normal", "normal", "normal"
        else:
            disease, kn, response = "unknown", "unknown", "unknown"
        rows.append(
            _base_meta(
                accession="GSE303591",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=_slug(sample_id.rsplit("_", 1)[0]),
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="fibroblast",
                treatment="none",
                cohort_role="development",
            )
        )
    metadata = pd.DataFrame(rows)
    if set(metadata["encoder_response"]) < {"keloid", "normal"}:
        raise RuntimeError(f"GSE303591: expected both classes, got {metadata['encoder_response'].value_counts().to_dict()}")
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE303591")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE303591: all program features zero")
    summary = {
        "accession": "GSE303591",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse282479(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Keloid vs normal fibroblasts; keep untreated columns only (K odd / ST odd)."""
    count_path = None
    for key, path in paths.items():
        if "VitD_counts" in key or key.endswith(".csv.gz"):
            count_path = path
            break
    if count_path is None or not count_path.exists():
        raise RuntimeError("GSE282479: VitD_counts.csv.gz missing")
    df = pd.read_csv(count_path)
    gene_col = df.columns[0]
    # Series order alternates Untreated/Paricalcitol within K* then ST*.
    untreated = []
    for c in df.columns:
        if c == gene_col:
            continue
        m = re.match(r"^(K|ST)(\d+)_", str(c), re.I)
        if not m:
            continue
        idx = int(m.group(2))
        if idx % 2 == 1:  # 1,3,5,... untreated in published matrix order
            untreated.append(c)
    if len(untreated) < 4:
        raise RuntimeError(f"GSE282479: expected untreated K*/ST* columns, got {df.columns.tolist()}")
    ensembl_map = load_ensembl_symbol_map()
    mapped = []
    for raw in df[gene_col].astype(str):
        key = strip_ensembl_version(raw)
        mapped.append(ensembl_map.get(key) or normalize_gene_symbol(raw) or None)
    mat = df[untreated].apply(pd.to_numeric, errors="coerce")
    mat.index = mapped
    mat = mat.loc[[g for g in mat.index if g and not str(g).startswith("ENS")]]
    expr = np.log1p(mat.groupby(level=0).mean().T.clip(lower=0))
    # Clean sample ids to short tokens.
    expr.index = [re.sub(r"_sorted\.bam$", "", str(i), flags=re.I) for i in expr.index]
    rows = []
    for sample_id in expr.index.astype(str):
        if sample_id.upper().startswith("K"):
            disease, kn, response = "keloid", "keloid", "keloid"
        elif sample_id.upper().startswith("ST"):
            disease, kn, response = "normal", "normal", "normal"
        else:
            disease, kn, response = "unknown", "unknown", "unknown"
        rows.append(
            _base_meta(
                accession="GSE282479",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=_slug(sample_id),
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="fibroblast",
                treatment="none",
                cohort_role="development",
            )
        )
    metadata = pd.DataFrame(rows)
    if set(metadata["encoder_response"]) < {"keloid", "normal"}:
        raise RuntimeError(f"GSE282479: expected both classes, got {metadata['encoder_response'].value_counts().to_dict()}")
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE282479")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE282479: all program features zero")
    summary = {
        "accession": "GSE282479",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse181316(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """All-barcode tissue pseudobulk from 10x scRNA (keloid / healthy skin / scar)."""
    tar_path = None
    for key, path in paths.items():
        if path.name.endswith("_RAW.tar") or key.endswith(".tar"):
            tar_path = path
            break
    if tar_path is None or not tar_path.exists():
        raise RuntimeError("GSE181316: RAW.tar missing")

    import tempfile

    from scipy import io as scipy_io

    import gzip

    sample_exprs: dict[str, pd.Series] = {}
    label_info: dict[str, tuple[str, str, str, str]] = {}
    # title token -> (disease, kn, scar, patient)
    with tarfile.open(tar_path) as tf, tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Group members by GSM prefix.
        by_gsm: dict[str, dict[str, object]] = {}
        for member in tf.getmembers():
            if not member.isfile():
                continue
            base = Path(member.name).name
            m = re.match(r"^(GSM\d+)_(.+?)_(barcodes|features|matrix)\.(tsv|mtx)\.gz$", base, re.I)
            if not m:
                continue
            gsm, token, kind = m.group(1), m.group(2), m.group(3).lower()
            by_gsm.setdefault(gsm, {"token": token})
            by_gsm[gsm][kind] = member
        for gsm, parts in sorted(by_gsm.items()):
            token = str(parts["token"]).lower()
            if not {"barcodes", "features", "matrix"} <= set(parts):
                continue
            for kind in ("barcodes", "features", "matrix"):
                member = parts[kind]
                dest = tmp_path / Path(member.name).name
                with tf.extractfile(member) as src, dest.open("wb") as out:
                    out.write(src.read())
            features_path = tmp_path / Path(parts["features"].name).name
            matrix_path = tmp_path / Path(parts["matrix"].name).name
            # 10x mtx is genes x cells; mmread needs an open gzip stream.
            with gzip.open(matrix_path, "rb") as handle:
                mat = scipy_io.mmread(handle).tocsr()
            feats = pd.read_csv(features_path, sep="\t", header=None, compression="gzip")
            # columns: id, symbol, type
            symbols = feats.iloc[:, 1].map(lambda g: normalize_gene_symbol(str(g)))
            # Sum across cells → pseudobulk counts.
            counts = mat.sum(axis=1).A.ravel()
            series = pd.Series(counts, index=symbols)
            series = series[series.index.astype(str).str.len() > 0]
            series = series.groupby(level=0).sum()
            sample_id = f"GSE181316_{gsm}_{parts['token']}"
            sample_exprs[sample_id] = series
            if token.startswith("keloid"):
                # keloid_3L / keloid_3R share a donor.
                donor = re.sub(r"[lr]$", "", token, flags=re.I)
                label_info[sample_id] = ("keloid", "keloid", "keloid", donor)
            elif token.startswith("skin"):
                label_info[sample_id] = ("normal", "normal", "normal", token)
            elif token.startswith("scar"):
                label_info[sample_id] = ("normal_scar", "unknown", "normal_scar", token)
            else:
                label_info[sample_id] = ("unknown", "unknown", "unknown", token)

    if len(sample_exprs) < 5:
        raise RuntimeError(f"GSE181316: expected >=5 pseudobulks, got {len(sample_exprs)}")
    expr = pd.DataFrame(sample_exprs).T.fillna(0.0)
    expr = np.log1p(expr.clip(lower=0))
    rows = []
    for sample_id in expr.index.astype(str):
        disease, kn, scar, donor = label_info[sample_id]
        task = "scar_differential" if scar == "normal_scar" else "keloid_vs_normal"
        rows.append(
            _base_meta(
                accession="GSE181316",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL20301"),
                modality="scrna_pseudobulk",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=scar,
                patient_id=_slug(donor),
                task=task,
                response=disease,
                contrast_type="sample",
                cell_type="all_barcodes",
                cohort_role=entry.get("role", "development"),
                allow_accession_level_grouping=False,
            )
        )
    metadata = pd.DataFrame(rows)
    coverage = module_coverage_report(expr.reset_index(drop=True), accession="GSE181316")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE181316: all program features zero after pseudobulk")
    summary = {
        "accession": "GSE181316",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": entry.get("role", "development"),
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "note": "All-barcode tissue pseudobulk; keloid_3L/3R share donor.",
    }
    return metadata, expr.reset_index(drop=True), summary


def process_gse232079(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Keloid vs normal fibroblast lines; DMSO vehicle only (exclude corin)."""
    count_path = None
    for key, path in paths.items():
        if key.endswith("counts.txt.gz") or "counts.txt" in key:
            count_path = path
            break
    if count_path is None or not count_path.exists():
        raise RuntimeError("GSE232079: counts.txt.gz missing")
    df = pd.read_csv(count_path, sep="\t")
    gene_col = df.columns[0]
    value_cols = [c for c in df.columns if c != gene_col and "DMSO" in str(c)]
    if len(value_cols) < 4:
        raise RuntimeError(f"GSE232079: expected DMSO columns, got {df.columns.tolist()[:12]}")
    ensembl_map = load_ensembl_symbol_map()
    mapped = []
    for raw in df[gene_col].astype(str):
        key = strip_ensembl_version(raw)
        mapped.append(ensembl_map.get(key) or normalize_gene_symbol(raw) or None)
    mat = df[value_cols].apply(pd.to_numeric, errors="coerce")
    mat.index = mapped
    mat = mat.loc[[g for g in mat.index if g and not str(g).startswith("ENS")]]
    expr = np.log1p(mat.groupby(level=0).mean().T.clip(lower=0))
    rows = []
    for sample_id in expr.index.astype(str):
        low = sample_id.lower()
        # Series characteristics: PCS/PNF = normal; CRL-1762 and PK* = keloid.
        if low.startswith("pk") or "crl-1762" in low:
            disease, kn, response = "keloid", "keloid", "keloid"
            donor = re.match(r"^(pk\d+|crl-1762)", low)
        elif low.startswith("pnf") or low.startswith("pcs-"):
            disease, kn, response = "normal", "normal", "normal"
            donor = re.match(r"^(pnf\d+|pcs-[\d-]+)", low)
        else:
            disease, kn, response = "unknown", "unknown", "unknown"
            donor = None
        rows.append(
            _base_meta(
                accession="GSE232079",
                sample_id=sample_id,
                platform_id=entry.get("platform_id", "GPL24676"),
                modality="bulk_rnaseq",
                disease_label=disease,
                keloid_vs_normal=kn,
                scar_type=disease if disease in {"keloid", "normal"} else "unknown",
                patient_id=_slug(donor.group(1) if donor else sample_id),
                task="keloid_vs_normal",
                response=response,
                contrast_type="sample",
                cell_type="fibroblast",
                treatment="none",
                cohort_role="development",
            )
        )
    metadata = pd.DataFrame(rows)
    metadata = metadata[metadata["encoder_response"].isin(["keloid", "normal"])].reset_index(drop=True)
    expr = expr.loc[metadata["sample_id"]].reset_index(drop=True)
    if set(metadata["encoder_response"]) < {"keloid", "normal"}:
        raise RuntimeError(f"GSE232079: expected both classes, got {metadata['encoder_response'].value_counts().to_dict()}")
    coverage = module_coverage_report(expr, accession="GSE232079")
    if coverage["all_zero_programs"]:
        raise RuntimeError("GSE232079: all program features zero")
    summary = {
        "accession": "GSE232079",
        "status": "ok",
        "n_samples": int(len(metadata)),
        "n_genes": int(expr.shape[1]),
        "n_modules_with_signal": coverage["n_modules_with_signal"],
        "role": "development",
        "labels": metadata["encoder_response"].value_counts().to_dict(),
        "note": "DMSO vehicle only; corin-treated samples excluded",
    }
    return metadata, expr, summary


def process_cohort(entry: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    accession = entry["accession"]
    if not paths:
        return None
    try:
        if accession == "GSE113619":
            return process_gse113619(entry, paths)
        if accession == "GSE246562":
            return process_gse246562(entry, paths)
        if accession == "GSE245660":
            return process_gse245660(entry, paths)
        if accession == "GSE121618":
            return process_gse121618(entry, paths)
        if accession == "GSE173900":
            return process_gse173900(entry, paths)
        if accession == "GSE190626":
            return process_gse190626(entry, paths)
        if accession == "GSE185309":
            return process_gse185309(entry, paths)
        if accession == "GSE212954":
            return process_gse212954(entry, paths)
        if accession == "GSE210434":
            return process_gse210434(entry, paths)
        if accession == "GSE303591":
            return process_gse303591(entry, paths)
        if accession == "GSE282479":
            return process_gse282479(entry, paths)
        if accession == "GSE232079":
            return process_gse232079(entry, paths)
        if accession == "GSE218007":
            return process_gse218007(entry, paths)
        if accession == "GSE178562":
            return process_gse178562(entry, paths)
        if accession == "GSE181316":
            return process_gse181316(entry, paths)
        if accession in {"GSE237752"}:
            return process_count_matrix_cohort(entry, paths)
        if entry.get("modality") == "microarray" or accession in {"GSE7980"}:
            return process_series_matrix_cohort(entry, paths)
        if accession == "GSE191067":
            return process_gse191067(entry, paths)
        return process_count_matrix_cohort(entry, paths)
    except Exception as exc:  # noqa: BLE001
        return None, None, {"accession": accession, "status": "error", "error": str(exc), "role": entry.get("role")}


def merge_into_existing_public(
    out_dir: Path,
    new_metas: list[pd.DataFrame],
    new_exprs: list[pd.DataFrame],
    new_summaries: list[dict],
    skipped: list[dict],
    registry: dict,
    jsonl_top_genes: int = 2048,
) -> dict:
    """Append newly processed accessions into existing public_keloid artifacts."""
    wide_path = out_dir / "public_keloid_expression_wide.parquet"
    meta_path = out_dir / "public_keloid_sample_metadata.parquet"
    if not wide_path.exists() or not meta_path.exists():
        raise RuntimeError("merge-existing requires prior public_keloid parquet artifacts")
    base_meta = pd.read_parquet(meta_path)
    existing = pd.read_parquet(wide_path)
    vocab_path = out_dir / "public_keloid_gene_vocab.txt"
    vocab = [line.strip() for line in vocab_path.read_text().splitlines() if line.strip()] if vocab_path.exists() else []
    score_cols = [c for c in existing.columns if c.endswith("_score") or c == "fibrotic_activity_score"]
    gene_cols = [c for c in existing.columns if c not in set(base_meta.columns) | set(score_cols)]
    if not gene_cols and vocab:
        gene_cols = vocab
    base_expr = existing.set_index("sample_id").reindex(columns=gene_cols).fillna(0.0)

    add_meta = pd.concat(new_metas, ignore_index=True, sort=False)
    add_expr = pd.concat(new_exprs, ignore_index=True, sort=False).fillna(0.0)
    add_expr.index = add_meta["sample_id"].tolist()
    all_genes = sorted(set(gene_cols) | set(add_expr.columns))
    base_expr = base_expr.reindex(columns=all_genes, fill_value=0.0)
    add_expr = add_expr.reindex(columns=all_genes, fill_value=0.0)

    drop_acc = set(add_meta["accession"].astype(str).unique())
    keep = ~base_meta["accession"].astype(str).isin(drop_acc)
    merged_meta = pd.concat([base_meta.loc[keep].reset_index(drop=True), add_meta], ignore_index=True, sort=False)
    # Vectorized row alignment (avoid O(n_samples) tiny concat on wide gene matrices).
    kept_ids = base_meta.loc[keep, "sample_id"].astype(str).tolist()
    base_kept = base_expr.reindex(index=kept_ids).fillna(0.0)
    merged_expr = pd.concat([base_kept, add_expr], axis=0).fillna(0.0)
    merged_expr = merged_expr.reindex(index=merged_meta["sample_id"].astype(str).tolist()).fillna(0.0)
    merged_expr = merged_expr.reset_index(drop=True)

    roles_path = out_dir / "public_keloid_cohort_roles.json"
    roles = json.loads(roles_path.read_text()) if roles_path.exists() else {}
    prev_summaries = [{"accession": a, "role": "development"} for a in roles.get("development", [])]
    prev_summaries += [{"accession": a, "role": "lockbox"} for a in roles.get("lockbox", [])]
    # Drop replaced accessions from previous summary list.
    prev_summaries = [s for s in prev_summaries if s["accession"] not in drop_acc]
    summary = write_expression_artifacts(
        out_dir=out_dir,
        prefix="public_keloid",
        metadata=merged_meta,
        expr=merged_expr,
        dataset_summaries=prev_summaries + new_summaries,
        skipped=list(roles.get("skipped", [])) + skipped,
        jsonl_top_genes=jsonl_top_genes,
    )
    for s in new_summaries:
        role = s.get("role", "development")
        acc = s["accession"]
        if role == "lockbox":
            roles.setdefault("lockbox", [])
            if acc not in roles["lockbox"]:
                roles["lockbox"].append(acc)
            roles["development"] = [a for a in roles.get("development", []) if a != acc]
        else:
            roles.setdefault("development", [])
            if acc not in roles["development"]:
                roles["development"].append(acc)
    roles["skipped"] = list(roles.get("skipped", [])) + skipped
    roles["original_10_comparator"] = registry.get("original_10_comparator", [])
    roles_path.write_text(json.dumps(roles, indent=2))
    summary["cohort_roles"] = str(roles_path.resolve().relative_to(PROJECT_ROOT) if roles_path.resolve().is_relative_to(PROJECT_ROOT) else roles_path)
    (out_dir / "public_keloid_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def process(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for key in ("development", "lockbox", "breakthrough_lockbox"):
        for e in registry.get(key, []):
            # Normalize breakthrough_lockbox role for processors that key off role.
            if key == "breakthrough_lockbox" and "role" not in e:
                e = {**e, "role": "lockbox"}
            entries.append(e)
    if args.roles:
        entries = [e for e in entries if e.get("role") in args.roles or e["accession"] in args.accessions]
    if args.accessions:
        entries = [e for e in entries if e["accession"] in args.accessions]

    metas = []
    exprs = []
    summaries = []
    skipped = []

    for entry in entries:
        if entry.get("role") == "interpretation_only":
            skipped.append({"accession": entry["accession"], "reason": "interpretation_only"})
            continue
        print(f"=== {entry['accession']} ({entry.get('role')}) ===")
        paths = download_cohort(
            entry,
            args.raw_dir,
            force=args.force_download,
            include_reserved_expression=bool(args.include_reserved_expression),
        )
        result = process_cohort(entry, paths)
        if result is None:
            skipped.append({"accession": entry["accession"], "reason": "no_usable_files", "role": entry.get("role")})
            continue
        meta, expr, summary = result
        if meta is None:
            skipped.append(summary)
            print(f"Failed: {summary}")
            continue
        # Validate optional GSE7980: require at least some donor IDs or explicit allow flag + both classes.
        if entry["accession"] == "GSE7980":
            n_known = int((~meta["patient_id"].astype(str).str.lower().isin(["unknown", "nan", "none"])).sum())
            classes = set(meta["keloid_vs_normal"].unique())
            if n_known == 0 and not entry.get("allow_accession_level_grouping", False):
                skipped.append({"accession": "GSE7980", "reason": "no_donor_metadata"})
                continue
            if not ({"keloid"} & classes) or not ({"normal", "non_keloid"} & classes):
                # keep if disease_label has both
                dl = set(meta["disease_label"].unique())
                if not ({"keloid"} & dl) or not ({"normal", "control"} & dl):
                    skipped.append({"accession": "GSE7980", "reason": "invalid_binary_labels", "labels": meta["encoder_response"].value_counts().to_dict()})
                    continue
        coverage = module_coverage_report(expr, accession=entry["accession"])
        if coverage["all_zero_programs"]:
            skipped.append({"accession": entry["accession"], "reason": "all_zero_programs"})
            continue
        print(
            f"{entry['accession']}: {len(meta)} samples, {expr.shape[1]} genes, "
            f"{coverage['n_modules_with_signal']}/{len(MODULE_GENES)} modules with signal"
        )
        metas.append(meta)
        exprs.append(expr)
        summaries.append(summary)

    if not metas:
        raise RuntimeError("No public keloid cohorts were processed successfully.")

    if args.merge_existing:
        summary = merge_into_existing_public(
            out_dir=args.out_dir,
            new_metas=metas,
            new_exprs=exprs,
            new_summaries=summaries,
            skipped=skipped,
            registry=registry,
            jsonl_top_genes=args.jsonl_top_genes,
        )
        print(json.dumps(summary, indent=2))
        return

    metadata = pd.concat(metas, ignore_index=True, sort=False)
    expr = pd.concat(exprs, ignore_index=True, sort=False).fillna(0.0)
    summary = write_expression_artifacts(
        out_dir=args.out_dir,
        prefix="public_keloid",
        metadata=metadata,
        expr=expr,
        dataset_summaries=summaries,
        skipped=skipped,
        jsonl_top_genes=args.jsonl_top_genes,
    )
    # Persist role flags on metadata for corpus filtering.
    role_path = args.out_dir / "public_keloid_cohort_roles.json"
    role_path.write_text(
        json.dumps(
            {
                "development": [s["accession"] for s in summaries if s.get("role") in {"development", "development_optional"}],
                "lockbox": [s["accession"] for s in summaries if s.get("role") == "lockbox"],
                "skipped": skipped,
                "original_10_comparator": registry.get("original_10_comparator", []),
            },
            indent=2,
        )
    )
    summary["cohort_roles"] = str(role_path.relative_to(PROJECT_ROOT))
    (args.out_dir / "public_keloid_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--accessions", nargs="*", default=[])
    p.add_argument("--roles", nargs="*", default=[])
    p.add_argument("--force-download", action="store_true")
    p.add_argument(
        "--include-reserved-expression",
        action="store_true",
        help="Download download_after_freeze assets (prospective lockbox expression).",
    )
    p.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly processed accessions into existing public_keloid artifacts (do not wipe).",
    )
    p.add_argument("--jsonl-top-genes", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
