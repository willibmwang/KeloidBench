#!/usr/bin/env python3
"""Reusable data extraction entry point for spheroid fibrosis datasets."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = PROJECT_ROOT / "data/raw/spheroid/choi/supplementary_choi.xlsx"
OUT_DIR = PROJECT_ROOT / "data/processed/spheroid"

STATE_NAMES = [
    "aggregate",
    "pre_compaction",
    "compact_spheroid",
    "pre_regression",
    "regression",
]

DRUG_LAYOUT = [
    ("triamcinolone", 1),
    ("5FU", 4),
    ("bleomycin", 7),
]


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of files written by an extractor."""

    name: str
    rows: int
    path: Path


@dataclass(frozen=True)
class DatasetExtractor:
    """Registry entry for a reusable dataset extractor."""

    name: str
    description: str
    run: Callable[[argparse.Namespace], list[ExtractionResult]]


def parse_condition(code: str) -> dict:
    code = str(code).strip()
    match = re.match(r"^([AKN])(\d?)\(F(\d+)E(\d+)\)$", code)
    if not match:
        return {
            "condition_code": code,
            "cell_source": "mixed",
            "fb_ec_ratio": "NA",
        }

    prefix, patient_num, fibroblast_count, endothelial_count = match.groups()
    if prefix == "A":
        cell_source = "ATCC_KF"
    elif prefix == "N":
        cell_source = "NDF"
    else:
        cell_source = f"K{patient_num or '1'}"

    return {
        "condition_code": code,
        "cell_source": cell_source,
        "fb_ec_ratio": f"{fibroblast_count}:{endothelial_count}",
    }


def base_metadata(sheet: str, **kwargs) -> dict:
    return {"paper": "choi_2024", "source_sheet": sheet, **kwargs}


def write_parquet(df: pd.DataFrame, path: Path) -> ExtractionResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return ExtractionResult(path.stem, len(df), path)


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def melt_state_composition(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Figure 1c / 4b: state percentage by condition x day."""
    header = df.iloc[0]
    days = df.iloc[1]
    records = []
    col = 1
    while col < df.shape[1]:
        condition = header.iloc[col]
        if pd.isna(condition) or not str(condition).startswith(("A(", "K", "N(")):
            col += 1
            continue

        block_end = col
        while block_end < df.shape[1] and (
            pd.isna(header.iloc[block_end]) or header.iloc[block_end] == condition
        ):
            block_end += 1

        meta = parse_condition(condition)
        for c in range(col, block_end):
            day_label = days.iloc[c]
            if pd.isna(day_label) or not str(day_label).startswith("Day"):
                continue
            day = int(str(day_label).replace("Day ", ""))
            for row in range(2, df.shape[0]):
                state = df.iloc[row, 0]
                value = df.iloc[row, c]
                if pd.isna(state) or pd.isna(value):
                    continue
                records.append(
                    {
                        **base_metadata(sheet, **meta, day=day),
                        "state_name": str(state).lower().replace(" ", "_"),
                        "state_pct": float(value),
                    }
                )
        col = block_end

    return pd.DataFrame(records)


def melt_area_timeseries(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Figure 1d: per-replicate spheroid area."""
    header = df.iloc[0]
    days = df.iloc[1]
    records = []
    col = 1
    while col < df.shape[1]:
        condition = header.iloc[col]
        if pd.isna(condition) or not str(condition).startswith(("A(", "K", "N(")):
            col += 1
            continue

        block_end = col
        while block_end < df.shape[1] and (
            pd.isna(header.iloc[block_end]) or header.iloc[block_end] == condition
        ):
            block_end += 1

        meta = parse_condition(condition)
        for c in range(col, block_end):
            day_label = days.iloc[c]
            if pd.isna(day_label) or not str(day_label).startswith("Day"):
                continue
            day = int(str(day_label).replace("Day ", ""))
            for row in range(2, df.shape[0]):
                replicate = df.iloc[row, 0]
                value = df.iloc[row, c]
                if pd.isna(replicate) or pd.isna(value):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            day=day,
                            replicate_id=int(replicate),
                            culture_format="3D_spheroid",
                        ),
                        "spheroid_area_um2": float(value),
                    }
                )
        col = block_end

    return pd.DataFrame(records)


def melt_propagation(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Figure 3c: propagation area by day block."""
    row0 = df.iloc[0]
    row1 = df.iloc[1]
    records = []
    col = 1
    while col < df.shape[1]:
        day_label = row0.iloc[col]
        if pd.isna(day_label) or not str(day_label).startswith("Day"):
            col += 1
            continue

        day = int(str(day_label).replace("Day ", ""))
        block_end = col
        while block_end < df.shape[1] and (
            pd.isna(row0.iloc[block_end]) or row0.iloc[block_end] == day_label
        ):
            block_end += 1

        for c in range(col, block_end):
            condition = row1.iloc[c]
            if pd.isna(condition):
                continue
            meta = parse_condition(condition)
            for row in range(2, df.shape[0]):
                replicate = df.iloc[row, 0]
                value = df.iloc[row, c]
                if pd.isna(replicate) or pd.isna(value):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            day=day,
                            replicate_id=int(replicate),
                            culture_format="3D_spheroid",
                        ),
                        "propagation_area": float(value),
                    }
                )
        col = block_end

    return pd.DataFrame(records)


def melt_qpcr(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Figure 5: qPCR relative expression blocks."""
    genes = ["COL1A1", "COL3A1", "TGFB1", "TGFB3", "HIF1A", "MMP14", "HTRA1", "ADAM12", "CTHRC1"]
    records = []

    gene_cols: list[tuple[str, int]] = []
    for col in range(df.shape[1]):
        value = df.iloc[0, col]
        if isinstance(value, str) and value in genes:
            gene_cols.append((value, col))

    for gene, start_col in gene_cols:
        condition_col = start_col - 1
        layout_row = df.iloc[1, start_col : start_col + 8].tolist()
        has_spheroid_label = any(
            str(value).strip().lower() == "spheroid" for value in layout_row if not pd.isna(value)
        )
        spheroid_offset = 4 if has_spheroid_label else 3

        for row in range(3, df.shape[0]):
            condition = df.iloc[row, condition_col]
            if pd.isna(condition) or not str(condition).startswith(("A(", "K")):
                continue

            meta = parse_condition(str(condition))
            monolayer_values = [df.iloc[row, start_col + i] for i in range(3)]
            spheroid_values = [
                df.iloc[row, start_col + spheroid_offset + i]
                for i in range(3)
                if start_col + spheroid_offset + i < df.shape[1]
            ]
            while len(spheroid_values) < 3:
                spheroid_values.append(float("nan"))

            for replicate_id, value in enumerate(monolayer_values, start=1):
                if pd.isna(value):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            replicate_id=replicate_id,
                            culture_format="2D_monolayer",
                        ),
                        "gene": gene,
                        "expression_relative": float(value),
                    }
                )

            for replicate_id, value in enumerate(spheroid_values, start=1):
                if pd.isna(value):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            replicate_id=replicate_id,
                            culture_format="3D_spheroid",
                        ),
                        "gene": gene,
                        "expression_relative": float(value),
                    }
                )

    return pd.DataFrame(records)


def _is_replicate(value) -> bool:
    if pd.isna(value):
        return False
    try:
        int(float(value))
        return True
    except (TypeError, ValueError):
        return False


def melt_drug_response(df: pd.DataFrame, sheet: str, assay: str) -> pd.DataFrame:
    """Figure 6b (3D volume) or 6c (2D viability)."""
    records = []
    row = 0
    while row < df.shape[0]:
        value = df.iloc[row, 1] if df.shape[1] > 1 else None
        if isinstance(value, str) and "(" in value and "F" in value and "E" in value:
            condition = value
            meta = parse_condition(condition)
            data_row = row + 3
            while data_row < df.shape[0]:
                replicate = df.iloc[data_row, 0]
                if not _is_replicate(replicate):
                    break

                for drug_name, start_col in DRUG_LAYOUT:
                    for offset, concentration_um in enumerate([0, 10, 100]):
                        col = start_col + offset
                        if col >= df.shape[1]:
                            continue
                        response_value = df.iloc[data_row, col]
                        if pd.isna(response_value):
                            continue

                        record = {
                            **base_metadata(
                                sheet,
                                **meta,
                                replicate_id=int(float(replicate)),
                                culture_format="3D_spheroid" if assay == "3d" else "2D_monolayer",
                                drug="vehicle" if concentration_um == 0 else drug_name,
                                drug_concentration_um=concentration_um,
                            ),
                        }
                        if assay == "3d":
                            record["relative_volume_pct"] = float(response_value)
                        else:
                            record["viability_2d_pct"] = float(response_value)
                        records.append(record)
                data_row += 1
            row = data_row
        else:
            row += 1

    return pd.DataFrame(records)


def melt_figure_s1(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Supplementary Figure S1 regression points."""
    records = []
    proportions = df.iloc[0, 1:].tolist()
    for row in range(1, df.shape[0]):
        point_id = df.iloc[row, 0]
        if pd.isna(point_id):
            continue
        for col, proportion in enumerate(proportions, start=1):
            value = df.iloc[row, col]
            if pd.isna(value):
                continue
            records.append(
                {
                    **base_metadata(sheet, replicate_id=int(point_id)),
                    "fibroblast_proportion": float(proportion),
                    "gene_expression_ratio_pct": float(value),
                }
            )

    return pd.DataFrame(records)


def pivot_qpcr_wide(qpcr: pd.DataFrame) -> pd.DataFrame:
    if qpcr.empty:
        return qpcr

    index_cols = [
        "paper",
        "source_sheet",
        "condition_code",
        "cell_source",
        "fb_ec_ratio",
        "culture_format",
        "replicate_id",
    ]
    wide = qpcr.pivot_table(
        index=index_cols,
        columns="gene",
        values="expression_relative",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def build_choi_unified_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge Choi tables into one row per condition-replicate for modeling."""
    morphology = tables.get("morphology_area", pd.DataFrame())
    drug3d = tables.get("drug_response_3d", pd.DataFrame())
    qpcr = tables.get("qpcr", pd.DataFrame())
    state = pd.concat(
        [
            tables.get("state_composition_atcc", pd.DataFrame()),
            tables.get("state_composition_patient", pd.DataFrame()),
        ],
        ignore_index=True,
    )

    qpcr_wide = pivot_qpcr_wide(qpcr)

    morph_day4 = pd.DataFrame()
    if not morphology.empty:
        morph_day4 = (
            morphology[morphology["day"] == 4]
            .groupby(
                ["condition_code", "cell_source", "fb_ec_ratio", "replicate_id"],
                as_index=False,
            )["spheroid_area_um2"]
            .mean()
        )

    state_day4 = pd.DataFrame()
    if not state.empty:
        state_dom = (
            state.sort_values("state_pct", ascending=False)
            .groupby(["condition_code", "cell_source", "fb_ec_ratio", "day"], as_index=False)
            .first()[
                [
                    "condition_code",
                    "cell_source",
                    "fb_ec_ratio",
                    "day",
                    "state_name",
                    "state_pct",
                ]
            ]
        )
        state_day4 = state_dom[state_dom["day"] == 4].rename(
            columns={"state_name": "dominant_spheroid_state", "state_pct": "dominant_state_pct"}
        )

    unified = drug3d.copy()
    if not morph_day4.empty:
        unified = unified.merge(
            morph_day4,
            on=["condition_code", "cell_source", "fb_ec_ratio", "replicate_id"],
            how="outer",
        )
    if not qpcr_wide.empty:
        qpcr_spheroid = qpcr_wide[qpcr_wide["culture_format"] == "3D_spheroid"]
        unified = unified.merge(
            qpcr_spheroid.drop(columns=["paper", "source_sheet", "culture_format"], errors="ignore"),
            on=["condition_code", "cell_source", "fb_ec_ratio", "replicate_id"],
            how="left",
        )
    if not state_day4.empty:
        unified = unified.merge(
            state_day4.drop(columns=["day"], errors="ignore"),
            on=["condition_code", "cell_source", "fb_ec_ratio"],
            how="left",
        )

    if "relative_volume_pct" in unified.columns:
        unified["drug_response_class"] = pd.cut(
            unified["relative_volume_pct"],
            bins=[-float("inf"), 70, 85, float("inf")],
            labels=["sensitive", "intermediate", "resistant"],
        )

    if not qpcr_wide.empty:
        qpcr_spheroid = qpcr_wide[qpcr_wide["culture_format"] == "3D_spheroid"].copy()
        marker_cols = [
            col
            for col in [
                "COL1A1",
                "COL3A1",
                "TGFB1",
                "TGFB3",
                "HIF1A",
                "MMP14",
                "HTRA1",
                "ADAM12",
                "CTHRC1",
            ]
            if col in qpcr_spheroid.columns
        ]
        if marker_cols:
            qpcr_spheroid["mean_qpcr_expression"] = qpcr_spheroid[marker_cols].mean(axis=1)
            qpcr_spheroid["fibrotic_state"] = "active_fibrotic"
            qpcr_spheroid["source_dataset"] = "choi_qpcr"
            qpcr_features = qpcr_spheroid[
                [
                    "paper",
                    "source_sheet",
                    "source_dataset",
                    "condition_code",
                    "cell_source",
                    "fb_ec_ratio",
                    "culture_format",
                    "replicate_id",
                    "fibrotic_state",
                    "mean_qpcr_expression",
                    *marker_cols,
                ]
            ]
            unified = pd.concat([unified, qpcr_features], ignore_index=True, sort=False)

    return unified


def extract_choi_tables(xlsx: Path, out_dir: Path) -> tuple[dict[str, pd.DataFrame], list[ExtractionResult]]:
    """Extract all supported Choi supplementary Excel sheets."""
    parsers = {
        "Figure 1c": ("state_composition_atcc", lambda df: melt_state_composition(df, "Figure 1c")),
        "Figure 1d": ("morphology_area", lambda df: melt_area_timeseries(df, "Figure 1d")),
        "Figure 3c": ("propagation", lambda df: melt_propagation(df, "Figure 3c")),
        "Figure 4b": ("state_composition_patient", lambda df: melt_state_composition(df, "Figure 4b")),
        "Figure 5": ("qpcr", lambda df: melt_qpcr(df, "Figure 5")),
        "Figure 6b": ("drug_response_3d", lambda df: melt_drug_response(df, "Figure 6b", "3d")),
        "Figure 6c": ("drug_response_2d", lambda df: melt_drug_response(df, "Figure 6c", "2d")),
        "Figure S1": ("regression_s1", lambda df: melt_figure_s1(df, "Figure S1")),
    }

    tables: dict[str, pd.DataFrame] = {}
    results: list[ExtractionResult] = []
    for sheet_name, (key, parser) in parsers.items():
        sheet = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
        extracted = parser(sheet)
        tables[key] = extracted
        result = write_parquet(extracted, out_dir / f"choi_{key}.parquet")
        results.append(result)
        print(f"{key}: {result.rows} rows -> {result.path}")

    summary_path = out_dir / "choi_extraction_summary.csv"
    pd.DataFrame(
        [(result.name.replace("choi_", ""), result.rows, repo_relative(result.path)) for result in results],
        columns=["table", "n_rows", "path"],
    ).to_csv(summary_path, index=False)
    results.append(ExtractionResult("choi_extraction_summary", len(results), summary_path))

    return tables, results


def build_dirand_seed() -> pd.DataFrame:
    """Create qualitative Dirand controls for fibrotic-state supervision."""
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


def run_choi(args: argparse.Namespace) -> list[ExtractionResult]:
    tables, results = extract_choi_tables(args.xlsx, args.out_dir)
    unified = build_choi_unified_dataset(tables)
    result = write_parquet(unified, args.out_dir / "dataset_a_keloid_spheroid.parquet")
    results.append(result)
    print(f"choi_unified: {result.rows} rows -> {result.path}")
    return results


def run_dataset_a(args: argparse.Namespace) -> list[ExtractionResult]:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExtractionResult] = []
    if args.skip_choi:
        dataset_a_path = args.out_dir / "dataset_a_keloid_spheroid.parquet"
        choi = pd.read_parquet(dataset_a_path)
        choi = choi[choi.get("paper", pd.Series(index=choi.index)) != "dirand_2023"].copy()
    else:
        results.extend(run_choi(args))
        choi = pd.read_parquet(args.out_dir / "dataset_a_keloid_spheroid.parquet")

    if args.skip_dirand:
        combined = choi
    else:
        dirand = build_dirand_seed()
        dirand_result = write_parquet(dirand, args.out_dir / "dirand_seed_labels.parquet")
        results.append(dirand_result)
        combined = pd.concat([choi, dirand], ignore_index=True, sort=False)
        print(f"dirand_seed: {dirand_result.rows} rows -> {dirand_result.path}")

    dataset_a_result = write_parquet(combined, args.out_dir / "dataset_a_keloid_spheroid.parquet")
    results.append(dataset_a_result)

    summary_path = args.out_dir / "dataset_a_summary.csv"
    summary = (
        combined.assign(
            source_dataset=combined.get("source_dataset", pd.Series(index=combined.index)).fillna("choi")
        )
        .groupby(["paper", "source_dataset"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    summary.to_csv(summary_path, index=False)
    results.append(ExtractionResult("dataset_a_summary", len(summary), summary_path))

    print(f"dataset_a_combined: {dataset_a_result.rows} rows -> {dataset_a_result.path}")
    print(f"summary -> {summary_path}")
    return results


EXTRACTORS = {
    "choi": DatasetExtractor(
        name="choi",
        description="Extract Choi supplementary Excel sheets and Choi-only Dataset A table.",
        run=run_choi,
    ),
    "dataset_a": DatasetExtractor(
        name="dataset_a",
        description="Build Dataset A from Choi source data plus Dirand qualitative controls.",
        run=run_dataset_a,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract reusable spheroid fibrosis datasets. Add future datasets by "
            "registering a DatasetExtractor in EXTRACTORS."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(EXTRACTORS),
        default="dataset_a",
        help="Dataset extractor to run.",
    )
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Choi supplementary Excel file.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Processed data output directory.")
    parser.add_argument("--skip-choi", action="store_true", help="Reuse existing Choi table when building Dataset A.")
    parser.add_argument("--skip-dirand", action="store_true", help="Omit Dirand qualitative controls.")
    parser.add_argument("--list-datasets", action="store_true", help="List available extractors and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        for extractor in EXTRACTORS.values():
            print(f"{extractor.name}: {extractor.description}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extractor = EXTRACTORS[args.dataset]
    results = extractor.run(args)
    print(f"{extractor.name}: wrote {len(results)} artifacts")


if __name__ == "__main__":
    main()
