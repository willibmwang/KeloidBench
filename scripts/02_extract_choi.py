#!/usr/bin/env python3
"""Parse Choi et al. 2024 Supplementary Data 1 (xlsx) into long-format tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = PROJECT_ROOT / "data/raw/choi/supplementary_choi.xlsx"
OUT_DIR = PROJECT_ROOT / "data/processed"

CONDITION_RE = re.compile(r"^([AKN]\d?)\(F(\d+)E(\d+)\)$|^([AKN])\(F(\d+)E(\d+)\)$")

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


def parse_condition(code: str) -> dict:
    code = str(code).strip()
    m = re.match(r"^([AKN])(\d?)\(F(\d+)E(\d+)\)$", code)
    if not m:
        return {
            "condition_code": code,
            "cell_source": "mixed",
            "fb_ec_ratio": "NA",
        }
    prefix, patient_num, f, e = m.groups()
    if prefix == "A":
        cell_source = "ATCC_KF"
    elif prefix == "N":
        cell_source = "NDF"
    else:
        cell_source = f"K{patient_num or '1'}"
        if prefix == "K" and patient_num:
            cell_source = f"K{patient_num}"
    return {
        "condition_code": code,
        "cell_source": cell_source,
        "fb_ec_ratio": f"{f}:{e}",
    }


def base_metadata(sheet: str, **kwargs) -> dict:
    return {"paper": "choi_2024", "source_sheet": sheet, **kwargs}


def melt_state_composition(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Figure 1c / 4b: state percentage by condition x day."""
    header = df.iloc[0]
    days = df.iloc[1]
    records = []
    col = 1
    while col < df.shape[1]:
        cond = header.iloc[col]
        if pd.isna(cond) or not str(cond).startswith(("A(", "K", "N(")):
            col += 1
            continue
        block_end = col
        while block_end < df.shape[1] and (
            pd.isna(header.iloc[block_end]) or header.iloc[block_end] == cond
        ):
            block_end += 1
        meta = parse_condition(cond)
        for c in range(col, block_end):
            day_label = days.iloc[c]
            if pd.isna(day_label) or not str(day_label).startswith("Day"):
                continue
            day = int(str(day_label).replace("Day ", ""))
            for r in range(2, df.shape[0]):
                state = df.iloc[r, 0]
                val = df.iloc[r, c]
                if pd.isna(state) or pd.isna(val):
                    continue
                records.append(
                    {
                        **base_metadata(sheet, **meta, day=day),
                        "state_name": str(state).lower().replace(" ", "_"),
                        "state_pct": float(val),
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
        cond = header.iloc[col]
        if pd.isna(cond) or not str(cond).startswith(("A(", "K", "N(")):
            col += 1
            continue
        block_end = col
        while block_end < df.shape[1] and (
            pd.isna(header.iloc[block_end]) or header.iloc[block_end] == cond
        ):
            block_end += 1
        meta = parse_condition(cond)
        for c in range(col, block_end):
            day_label = days.iloc[c]
            if pd.isna(day_label) or not str(day_label).startswith("Day"):
                continue
            day = int(str(day_label).replace("Day ", ""))
            for r in range(2, df.shape[0]):
                rep = df.iloc[r, 0]
                val = df.iloc[r, c]
                if pd.isna(rep) or pd.isna(val):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            day=day,
                            replicate_id=int(rep),
                            culture_format="3D_spheroid",
                        ),
                        "spheroid_area_um2": float(val),
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
            cond = row1.iloc[c]
            if pd.isna(cond):
                continue
            meta = parse_condition(cond)
            for r in range(2, df.shape[0]):
                rep = df.iloc[r, 0]
                val = df.iloc[r, c]
                if pd.isna(rep) or pd.isna(val):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            day=day,
                            replicate_id=int(rep),
                            culture_format="3D_spheroid",
                        ),
                        "propagation_area": float(val),
                    }
                )
        col = block_end
    return pd.DataFrame(records)


def melt_qpcr(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Figure 5: qPCR relative expression blocks."""
    genes = ["COL1A1", "COL3A1", "TGFB1", "TGFB3", "HIF1A", "MMP14", "HTRA1", "ADAM12", "CTHRC1"]
    gene_starts = [1, 9, 17, 1, 9, 17, 1, 9, 17]  # placeholder, detect dynamically
    records = []

    gene_cols: list[tuple[str, int]] = []
    for c in range(df.shape[1]):
        val = df.iloc[0, c]
        if isinstance(val, str) and val in genes:
            gene_cols.append((val, c))

    for gene, start_col in gene_cols:
        cond_col = start_col - 1
        layout_row = df.iloc[1, start_col : start_col + 8].tolist()
        has_sph_label = any(str(v).strip().lower() == "spheroid" for v in layout_row if not pd.isna(v))
        sph_offset = 4 if has_sph_label else 3
        for r in range(3, df.shape[0]):
            cond = df.iloc[r, cond_col]
            if pd.isna(cond) or not str(cond).startswith(("A(", "K")):
                continue
            meta = parse_condition(str(cond))
            mono_vals = [df.iloc[r, start_col + i] for i in range(3)]
            sph_vals = [
                df.iloc[r, start_col + sph_offset + i]
                for i in range(3)
                if start_col + sph_offset + i < df.shape[1]
            ]
            while len(sph_vals) < 3:
                sph_vals.append(float("nan"))
            for i, v in enumerate(mono_vals, start=1):
                if pd.isna(v):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            replicate_id=i,
                            culture_format="2D_monolayer",
                        ),
                        "gene": gene,
                        "expression_relative": float(v),
                    }
                )
            for i, v in enumerate(sph_vals, start=1):
                if pd.isna(v):
                    continue
                records.append(
                    {
                        **base_metadata(
                            sheet,
                            **meta,
                            replicate_id=i,
                            culture_format="3D_spheroid",
                        ),
                        "gene": gene,
                        "expression_relative": float(v),
                    }
                )
    return pd.DataFrame(records)


def _is_replicate(val) -> bool:
    if pd.isna(val):
        return False
    try:
        int(float(val))
        return True
    except (TypeError, ValueError):
        return False


def melt_drug_response(df: pd.DataFrame, sheet: str, assay: str) -> pd.DataFrame:
    """Figure 6b (3D volume) or 6c (2D viability)."""
    records = []
    r = 0
    while r < df.shape[0]:
        val = df.iloc[r, 1] if df.shape[1] > 1 else None
        if isinstance(val, str) and "(" in val and "F" in val and "E" in val:
            cond = val
            meta = parse_condition(cond)
            data_row = r + 3
            while data_row < df.shape[0]:
                rep = df.iloc[data_row, 0]
                if not _is_replicate(rep):
                    break
                for drug_name, start_col in DRUG_LAYOUT:
                    for offset, conc_um in enumerate([0, 10, 100]):
                        col = start_col + offset
                        if col >= df.shape[1]:
                            continue
                        v = df.iloc[data_row, col]
                        if pd.isna(v):
                            continue
                        row = {
                            **base_metadata(
                                sheet,
                                **meta,
                                replicate_id=int(float(rep)),
                                culture_format=(
                                    "3D_spheroid" if assay == "3d" else "2D_monolayer"
                                ),
                                drug="vehicle" if conc_um == 0 else drug_name,
                                drug_concentration_um=conc_um,
                            ),
                        }
                        if assay == "3d":
                            row["relative_volume_pct"] = float(v)
                        else:
                            row["viability_2d_pct"] = float(v)
                        records.append(row)
                data_row += 1
            r = data_row
        else:
            r += 1
    return pd.DataFrame(records)


def melt_figure_s1(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Supplementary Figure S1 regression points."""
    records = []
    proportions = df.iloc[0, 1:].tolist()
    for r in range(1, df.shape[0]):
        point_id = df.iloc[r, 0]
        if pd.isna(point_id):
            continue
        for c, prop in enumerate(proportions, start=1):
            val = df.iloc[r, c]
            if pd.isna(val):
                continue
            records.append(
                {
                    **base_metadata(sheet, replicate_id=int(point_id)),
                    "fibroblast_proportion": float(prop),
                    "gene_expression_ratio_pct": float(val),
                }
            )
    return pd.DataFrame(records)


def pivot_qpcr_wide(qpcr: pd.DataFrame) -> pd.DataFrame:
    if qpcr.empty:
        return qpcr
    idx = [
        "paper",
        "source_sheet",
        "condition_code",
        "cell_source",
        "fb_ec_ratio",
        "culture_format",
        "replicate_id",
    ]
    wide = qpcr.pivot_table(
        index=idx, columns="gene", values="expression_relative", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide


def build_unified_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge key tables into one row per condition-replicate for modeling."""
    morph = tables.get("morphology_area", pd.DataFrame())
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

    morph_day4 = (
        morph[morph["day"] == 4]
        .groupby(
            ["condition_code", "cell_source", "fb_ec_ratio", "replicate_id"],
            as_index=False,
        )["spheroid_area_um2"]
        .mean()
    )

    state_dom = pd.DataFrame()
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
        qpcr_sph = qpcr_wide[qpcr_wide["culture_format"] == "3D_spheroid"]
        unified = unified.merge(
            qpcr_sph.drop(columns=["paper", "source_sheet", "culture_format"], errors="ignore"),
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
        qpcr_sph = qpcr_wide[qpcr_wide["culture_format"] == "3D_spheroid"].copy()
        marker_cols = [
            c
            for c in [
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
            if c in qpcr_sph.columns
        ]
        if marker_cols:
            qpcr_sph["mean_qpcr_expression"] = qpcr_sph[marker_cols].mean(axis=1)
            qpcr_sph["fibrotic_state"] = "active_fibrotic"
            qpcr_sph["source_dataset"] = "choi_qpcr"
            qpcr_features = qpcr_sph[
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(args.xlsx)

    parsers = {
        "Figure 1c": ("state_composition_atcc", lambda d: melt_state_composition(d, "Figure 1c")),
        "Figure 1d": ("morphology_area", lambda d: melt_area_timeseries(d, "Figure 1d")),
        "Figure 3c": ("propagation", lambda d: melt_propagation(d, "Figure 3c")),
        "Figure 4b": ("state_composition_patient", lambda d: melt_state_composition(d, "Figure 4b")),
        "Figure 5": ("qpcr", lambda d: melt_qpcr(d, "Figure 5")),
        "Figure 6b": ("drug_response_3d", lambda d: melt_drug_response(d, "Figure 6b", "3d")),
        "Figure 6c": ("drug_response_2d", lambda d: melt_drug_response(d, "Figure 6c", "2d")),
        "Figure S1": ("regression_s1", lambda d: melt_figure_s1(d, "Figure S1")),
    }

    tables: dict[str, pd.DataFrame] = {}
    summary = []
    for sheet_name, (key, fn) in parsers.items():
        df = pd.read_excel(args.xlsx, sheet_name=sheet_name, header=None)
        out = fn(df)
        tables[key] = out
        path = args.out_dir / f"choi_{key}.parquet"
        out.to_parquet(path, index=False)
        summary.append((key, len(out), path))
        print(f"{key}: {len(out)} rows -> {path}")

    unified = build_unified_dataset(tables)
    unified_path = args.out_dir / "dataset_a_keloid_spheroid.parquet"
    unified.to_parquet(unified_path, index=False)
    print(f"unified: {len(unified)} rows -> {unified_path}")

    pd.DataFrame(summary, columns=["table", "n_rows", "path"]).to_csv(
        args.out_dir / "choi_extraction_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
