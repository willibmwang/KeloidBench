#!/usr/bin/env python3
"""Preprocess uploaded GEO microarray series matrices for SpheroScar.

Outputs are designed for the lightweight expression encoder-decoder path:

- a sample metadata table
- a shared gene vocabulary
- a dense sample x gene NumPy matrix
- a wide parquet table with identical gene columns across accessions
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd

from expression_processing import (
    METADATA_COLUMNS,
    add_module_scores,
    is_hgnc_like_symbol,
    module_coverage_report,
    normalize_gene_symbol,
    symbol_from_gene_assignment,
    write_expression_artifacts,
    zscore_by_accession,
)
from gene_modules import MODULE_COLUMNS, MODULE_GENES, MODULES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw/microarray"
OUT_DIR = DATA_DIR / "processed/microarray"
PLATFORM_DIR = RAW_DIR / "platforms"

ACCESSIONS = ["GSE7890", "GSE92566", "GSE90051", "GSE44270", "GSE3189", "GSE145725"]
EXTRA_SERIES_DIRS = [DATA_DIR / "raw/bulk_rnaseq"]


@dataclass
class SeriesMatrix:
    accession: str
    platform_id: str
    metadata: pd.DataFrame
    expression: pd.DataFrame


def clean_geo_value(value: str) -> str:
    return value.strip().strip('"')


def split_geo_line(line: str) -> list[str]:
    return [clean_geo_value(part) for part in next(csv.reader([line], delimiter="\t"))]


def series_matrix_path(accession: str, data_dir: Path) -> Path:
    primary = data_dir / f"{accession}_series_matrix.txt.gz"
    if primary.exists():
        return primary
    for extra_dir in EXTRA_SERIES_DIRS:
        candidate = extra_dir / f"{accession}_series_matrix.txt.gz"
        if candidate.exists():
            return candidate
    return primary


def parse_series_matrix(path: Path) -> SeriesMatrix:
    accession = path.name.split("_", maxsplit=1)[0]
    platform_id = "unknown"
    sample_meta: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    table_lines: list[str] = []
    in_table = False

    with gzip.open(path, "rt", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table_lines.append(line)
                continue

            if line.startswith("!Series_platform_id"):
                values = split_geo_line(line)
                if len(values) > 1:
                    platform_id = values[1]
            elif line.startswith("!Sample_characteristics_ch1"):
                characteristics.append(split_geo_line(line)[1:])
            elif line.startswith("!Sample_characteristics_ch2"):
                sample_meta["source_name_ch2"] = split_geo_line(line)[1:]
            elif line.startswith("!Sample_"):
                values = split_geo_line(line)
                key = values[0].replace("!Sample_", "").lower()
                sample_meta[key] = values[1:]

    if not table_lines:
        raise ValueError(f"No expression table found in {path}")

    header = split_geo_line(table_lines[0])
    sample_ids = header[1:]
    rows = [split_geo_line(line) for line in table_lines[1:] if line]
    expression = pd.DataFrame(rows)
    expression.columns = ["ID_REF", *sample_ids]
    expression = expression.set_index("ID_REF")
    expression = expression.apply(pd.to_numeric, errors="coerce")

    metadata = pd.DataFrame({"sample_id": sample_ids})
    for key, values in sample_meta.items():
        if len(values) == len(sample_ids):
            metadata[key] = values

    for char_values in characteristics:
        if len(char_values) != len(sample_ids):
            continue
        parsed_keys = [value.split(":", maxsplit=1)[0].strip().lower() if ":" in value else "characteristic" for value in char_values]
        if len(set(parsed_keys)) == 1:
            key = re.sub(r"[^a-z0-9]+", "_", parsed_keys[0]).strip("_")
            metadata[key] = [value.split(":", maxsplit=1)[1].strip() if ":" in value else value for value in char_values]
        else:
            col = f"characteristics_{len([c for c in metadata.columns if c.startswith('characteristics_')]) + 1}"
            metadata[col] = char_values

    metadata["accession"] = accession
    metadata["platform_id"] = platform_id
    metadata["modality"] = "microarray"
    metadata["source_dataset"] = "microarray"
    return SeriesMatrix(accession=accession, platform_id=platform_id, metadata=metadata, expression=expression)


def _annotation_text(gpl_table: pd.DataFrame) -> pd.Series:
    symbol_cols = [
        col
        for col in (
            "Gene Symbol",
            "GENE_SYMBOL",
            "Symbol",
            "gene_symbol",
            "Gene symbol",
            "ILMN_Gene",
            "GENE",
        )
        if col in gpl_table.columns
    ]
    empty = {"", "nan", "na", "---", "null", "none"}

    def _nonempty(series: pd.Series) -> pd.Series:
        text = series.fillna("").astype(str).str.strip()
        return text.where(~text.str.lower().isin(empty), "")

    assignment = (
        _nonempty(gpl_table["gene_assignment"])
        if "gene_assignment" in gpl_table.columns
        else pd.Series("", index=gpl_table.index)
    )
    if symbol_cols:
        symbols = _nonempty(gpl_table[symbol_cols[0]])
        # Prefer Gene Symbol when present; fall back to gene_assignment (GPL6244).
        return symbols.where(symbols.ne(""), assignment)
    if "gene_assignment" in gpl_table.columns:
        return assignment
    fallback_cols = [col for col in gpl_table.columns if "gene" in col.lower() or "symbol" in col.lower()]
    if not fallback_cols:
        return pd.Series("", index=gpl_table.index)
    return gpl_table[fallback_cols].fillna("").astype(str).agg(" /// ".join, axis=1)


def _mapping_from_annotation_table(table: pd.DataFrame) -> dict[str, str]:
    if table.empty:
        return {}
    probe_col = "ID" if "ID" in table.columns else table.columns[0]
    text = _annotation_text(table)
    mapping: dict[str, str] = {}
    for probe, annotation in zip(table[probe_col].astype(str), text, strict=False):
        gene = symbol_from_gene_assignment(annotation)
        if gene is None:
            # Fallback for platforms that store bare symbols.
            gene = normalize_gene_symbol(annotation)
            if gene and not is_hgnc_like_symbol(gene):
                gene = None
        if gene:
            mapping[str(probe)] = gene
    return mapping


def load_platform_mapping_from_soft(path: Path) -> dict[str, str]:
    """Parse a local GPL SOFT annotation table without GEOparse."""
    if not path.exists():
        return {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if header is None:
                if line.startswith("ID\t"):
                    header = line.rstrip("\n").split("\t")
                continue
            if line.startswith("!platform_table_end"):
                break
            if line.startswith("!"):
                continue
            rows.append(line.rstrip("\n").split("\t"))
    if not header or not rows:
        return {}
    n_cols = len(header)
    # Pad/truncate ragged SOFT rows so Gene Symbol / gene_assignment stay aligned.
    fixed = [row[:n_cols] + [""] * max(0, n_cols - len(row)) for row in rows]
    table = pd.DataFrame(fixed, columns=header)
    keep_cols = ["ID"] + [
        col
        for col in (
            "Gene Symbol",
            "GENE_SYMBOL",
            "Symbol",
            "gene_symbol",
            "Gene symbol",
            "ILMN_Gene",
            "GENE",
            "gene_assignment",
        )
        if col in table.columns
    ]
    table = table[keep_cols]
    ids = table["ID"].astype(str).str.strip()
    table = table[ids.ne("") & ~ids.str.startswith("!") & ~ids.str.lower().eq("nan")]
    return _mapping_from_annotation_table(table)


def load_platform_mapping(platform_id: str, platform_dir: Path, allow_download: bool) -> dict[str, str]:
    platform_dir.mkdir(parents=True, exist_ok=True)
    local_soft = platform_dir / f"{platform_id}.txt"
    local_map = load_platform_mapping_from_soft(local_soft)
    if local_map:
        return local_map
    if not allow_download:
        return {}
    try:
        gpl = GEOparse.get_GEO(geo=platform_id, destdir=str(platform_dir), silent=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load {platform_id} annotation ({exc}); using probe IDs.")
        return {}

    table = gpl.table.copy()
    return _mapping_from_annotation_table(table)


def expression_to_genes(expression: pd.DataFrame, probe_to_gene: dict[str, str], platform_id: str) -> pd.DataFrame:
    mapped = expression.copy()
    if probe_to_gene:
        gene_index = pd.Series(mapped.index.astype(str), index=mapped.index).map(probe_to_gene)
        mapped = mapped.loc[gene_index.notna()].copy()
        mapped.index = gene_index[gene_index.notna()].values
    else:
        mapped.index = [f"{platform_id}_{probe}" for probe in mapped.index.astype(str)]
    # Multiple probes per gene are averaged per sample.
    gene_expr = mapped.groupby(mapped.index).mean()
    return gene_expr.T


def infer_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    parsed_cell_type = (
        metadata["cell_type"].copy()
        if "cell_type" in metadata.columns
        else pd.Series("unknown", index=metadata.index)
    )
    out = metadata.copy()
    if "sample_title" not in out.columns:
        out["sample_title"] = out.get("title", out["sample_id"])
    out["metadata_cell_type"] = parsed_cell_type
    out["disease_domain"] = "keloid"
    out["disease_label"] = "unknown"
    out["keloid_vs_normal"] = "unknown"
    out["lesional_status"] = "unknown"
    out["scar_type"] = "unknown"
    out["cell_type"] = "unknown"
    out["treatment"] = "none"
    out["patient_id"] = "unknown"
    out["contrast_type"] = "sample"
    out["eligible_for_keloid_pretraining"] = True

    for idx, row in out.iterrows():
        accession = str(row["accession"])
        title = str(row.get("title", row.get("sample_title", "")))
        source = str(row.get("source_name_ch1", ""))
        disorder = str(row.get("disorder", ""))
        condition = str(row.get("condition", ""))
        tissue = str(row.get("tissue", ""))
        sample_cell_type = str(row.get("metadata_cell_type", ""))
        # Protocol fields often describe all groups in the study. Restrict label
        # inference to sample-specific fields to avoid leakage across labels.
        text = f"{title} {source} {disorder} {condition} {tissue} {sample_cell_type}".lower()

        if accession == "GSE3189":
            text = f"{text} {row.get('characteristic', '')} {row.get('description', '')}".lower()
            out.at[idx, "disease_domain"] = "melanoma"
            out.at[idx, "eligible_for_keloid_pretraining"] = False
            if "melanoma" in text:
                out.at[idx, "disease_label"] = "melanoma"
            elif "nevus" in text:
                out.at[idx, "disease_label"] = "nevus"
            elif "normal" in text:
                out.at[idx, "disease_label"] = "normal"
            continue

        if accession == "GSE7890":
            out.at[idx, "cell_type"] = "fibroblast"
            if "hydrocortisone" in text or "+hydrocortisone" in text:
                out.at[idx, "treatment"] = "hydrocortisone" if "+hydrocortisone" in text or "treated" in text else "no_hydrocortisone"
            if disorder.lower() == "keloid" or title.lower().startswith("keloid"):
                out.at[idx, "disease_label"] = "keloid"
                out.at[idx, "keloid_vs_normal"] = "keloid"
            elif "normal scar" in text:
                out.at[idx, "disease_label"] = "normal_scar"
                out.at[idx, "keloid_vs_normal"] = "normal"
                out.at[idx, "scar_type"] = "normal_scar"
            elif "normal" in text:
                out.at[idx, "disease_label"] = "normal"
                out.at[idx, "keloid_vs_normal"] = "normal"
            patient = row.get("patient", None)
            if pd.isna(patient) or not str(patient).strip():
                patient = row.get("strain", None)
            if pd.notna(patient) and str(patient).strip():
                out.at[idx, "patient_id"] = str(patient).strip()
            continue

        if accession == "GSE44270":
            if "keloid" in title.lower():
                out.at[idx, "disease_label"] = "keloid"
                out.at[idx, "keloid_vs_normal"] = "keloid"
                out.at[idx, "lesional_status"] = "lesional"
            elif "non-lesional" in title.lower() or "non-lesion" in title.lower():
                out.at[idx, "disease_label"] = "non_lesional"
                out.at[idx, "keloid_vs_normal"] = "normal"
                out.at[idx, "lesional_status"] = "non_lesional"
            elif "control" in title.lower():
                out.at[idx, "disease_label"] = "normal"
                out.at[idx, "keloid_vs_normal"] = "normal"
                out.at[idx, "lesional_status"] = "control"
            if "keratinocyte" in title.lower():
                out.at[idx, "cell_type"] = "keratinocyte"
            elif "fibroblast" in title.lower():
                out.at[idx, "cell_type"] = "fibroblast"
            continue

        if accession == "GSE92566":
            out.at[idx, "disease_label"] = "keloid"
            out.at[idx, "keloid_vs_normal"] = "keloid"
            out.at[idx, "cell_type"] = "bulk_skin"
            if "new keloid formation" in title.lower():
                out.at[idx, "lesional_status"] = "new_keloid_formation"
            elif "non-lesion" in title.lower() or source.lower() == "non-lesion":
                out.at[idx, "lesional_status"] = "non_lesional"
            elif "lesion" in title.lower() or source.lower() == "lesion":
                out.at[idx, "lesional_status"] = "lesional"
            patient = row.get("patient", None)
            if pd.notna(patient) and str(patient).strip():
                out.at[idx, "patient_id"] = str(patient).strip()
            continue

        if accession == "GSE145725":
            out.at[idx, "cell_type"] = "fibroblast"
            out.at[idx, "source_dataset"] = "microarray"
            title_lower = title.lower()
            if title_lower.startswith("keloid"):
                out.at[idx, "disease_label"] = "keloid"
                out.at[idx, "keloid_vs_normal"] = "keloid"
            elif title_lower.startswith("normal"):
                out.at[idx, "disease_label"] = "normal"
                out.at[idx, "keloid_vs_normal"] = "normal"
            patient_match = re.search(r"fibrobalst_([0-9]+)", title_lower)
            if patient_match:
                prefix = "keloid" if title_lower.startswith("keloid") else "normal"
                out.at[idx, "patient_id"] = f"{prefix}_fibroblast_{patient_match.group(1)}"
            continue

        if "hydrocortisone" in text:
            out.at[idx, "treatment"] = "hydrocortisone" if "+hydrocortisone" in text or "treated" in text else "no_hydrocortisone"
        if "keloid" in text:
            out.at[idx, "disease_label"] = "keloid"
            out.at[idx, "keloid_vs_normal"] = "keloid"
        if "normal scar" in text:
            out.at[idx, "disease_label"] = "normal_scar"
            out.at[idx, "keloid_vs_normal"] = "normal"
            out.at[idx, "scar_type"] = "normal_scar"
        elif "control" in text or "normal" in text:
            out.at[idx, "disease_label"] = "normal"
            out.at[idx, "keloid_vs_normal"] = "normal"

        if "non-lesion" in text or "non-lesional" in text:
            out.at[idx, "lesional_status"] = "non_lesional"
            if out.at[idx, "disease_label"] == "unknown":
                out.at[idx, "disease_label"] = "non_lesional"
        elif "lesion" in text or "keloid lesion" in text:
            out.at[idx, "lesional_status"] = "lesional"

        if "new keloid formation" in text:
            out.at[idx, "lesional_status"] = "new_keloid_formation"
            out.at[idx, "disease_label"] = "keloid"
            out.at[idx, "keloid_vs_normal"] = "keloid"

        if "fibroblast" in text:
            out.at[idx, "cell_type"] = "fibroblast"
        elif "keratinocyte" in text:
            out.at[idx, "cell_type"] = "keratinocyte"
        elif "skin" in text:
            out.at[idx, "cell_type"] = "bulk_skin"

        patient = row.get("patient", None)
        if pd.isna(patient) or not str(patient).strip():
            patient = row.get("strain", None)
        if pd.notna(patient) and str(patient).strip():
            out.at[idx, "patient_id"] = str(patient).strip()
        else:
            match = re.search(r"patient\s*([0-9]+)", text)
            if match:
                out.at[idx, "patient_id"] = f"patient_{match.group(1)}"

        if accession == "GSE90051":
            out.at[idx, "contrast_type"] = "paired_keloid_over_normal_log_ratio"
            out.at[idx, "disease_label"] = "keloid_over_normal"
            out.at[idx, "keloid_vs_normal"] = "contrast"
            out.at[idx, "cell_type"] = "bulk_skin"

    out["encoder_task"] = out.apply(encoder_task_for_row, axis=1)
    out["encoder_prompt"] = out["encoder_task"].map(prompt_for_task)
    out["encoder_response"] = out.apply(response_for_row, axis=1)
    return out


def encoder_task_for_row(row: pd.Series) -> str:
    if not bool(row.get("eligible_for_keloid_pretraining", True)):
        return "out_of_domain_disease_state"
    if row.get("contrast_type") == "paired_keloid_over_normal_log_ratio":
        return "keloid_normal_contrast"
    if row.get("lesional_status") not in {None, "unknown"}:
        return "lesional_status"
    return "keloid_vs_normal"


def prompt_for_task(task: str) -> str:
    prompts = {
        "keloid_vs_normal": "Given this microarray expression profile, predict whether the sample is keloid or normal. Answer:",
        "lesional_status": "Given this keloid microarray expression profile, predict the lesional status. Answer:",
        "keloid_normal_contrast": "Given this paired keloid-over-normal expression contrast, identify the contrast type. Answer:",
        "out_of_domain_disease_state": "Given this out-of-domain skin expression profile, predict the disease state. Answer:",
    }
    return prompts.get(task, "Given this expression profile, predict the biological state. Answer:")


def response_for_row(row: pd.Series) -> str:
    task = row.get("encoder_task")
    if task == "lesional_status":
        return str(row.get("lesional_status", "unknown"))
    if task == "keloid_normal_contrast":
        return "keloid_over_normal"
    if task == "out_of_domain_disease_state":
        return str(row.get("disease_label", "unknown"))
    return str(row.get("keloid_vs_normal", "unknown"))


def process(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    PLATFORM_DIR.mkdir(parents=True, exist_ok=True)

    sample_tables = []
    expression_tables = []
    dataset_summaries = []
    coverage_rows = []
    platform_mappings: dict[str, dict[str, str]] = {}

    for accession in args.accessions:
        path = series_matrix_path(accession, args.data_dir)
        if not path.exists():
            dataset_summaries.append({"accession": accession, "status": "missing", "path": str(path)})
            continue

        series = parse_series_matrix(path)
        if series.platform_id not in platform_mappings:
            platform_mappings[series.platform_id] = load_platform_mapping(
                series.platform_id,
                PLATFORM_DIR,
                allow_download=not args.skip_platform_download,
            )
        gene_expr = expression_to_genes(series.expression, platform_mappings[series.platform_id], series.platform_id)
        metadata = infer_labels(series.metadata).reset_index(drop=True)
        gene_expr = gene_expr.reset_index(drop=True)
        coverage = module_coverage_report(gene_expr, accession=accession)
        coverage_rows.extend(coverage["modules"])
        if coverage["all_zero_programs"]:
            raise RuntimeError(
                f"{accession}: all program features have zero variance after gene mapping "
                f"(platform={series.platform_id}, mapped_probes={len(platform_mappings[series.platform_id])})."
            )

        sample_tables.append(metadata)
        expression_tables.append(gene_expr)
        dataset_summaries.append(
            {
                "accession": accession,
                "status": "ok",
                "platform_id": series.platform_id,
                "n_samples": int(len(metadata)),
                "n_probe_rows": int(series.expression.shape[0]),
                "n_gene_features": int(gene_expr.shape[1]),
                "n_mapped_probes": int(len(platform_mappings[series.platform_id])),
                "n_modules_with_signal": coverage["n_modules_with_signal"],
                "labels": metadata["encoder_response"].value_counts(dropna=False).to_dict(),
            }
        )
        print(
            f"{accession}: {len(metadata)} samples, {gene_expr.shape[1]} gene features, "
            f"{coverage['n_modules_with_signal']}/{len(MODULE_GENES)} modules with signal"
        )

    if not sample_tables:
        raise RuntimeError("No microarray datasets were processed.")

    metadata = pd.concat(sample_tables, axis=0, ignore_index=True, sort=False)
    expr = pd.concat(expression_tables, axis=0, ignore_index=True, sort=False).fillna(0.0)
    summary = write_expression_artifacts(
        out_dir=args.out_dir,
        prefix="microarray",
        metadata=metadata,
        expr=expr,
        dataset_summaries=dataset_summaries,
        jsonl_top_genes=args.jsonl_top_genes,
    )
    coverage_path = args.out_dir / "microarray_module_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    summary["module_coverage"] = str(coverage_path.relative_to(PROJECT_ROOT))
    (args.out_dir / "microarray_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--accessions", nargs="+", default=ACCESSIONS)
    parser.add_argument("--skip-platform-download", action="store_true")
    parser.add_argument("--jsonl-top-genes", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
