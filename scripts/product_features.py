"""Build Stage-B product feature views from a sample×gene expression matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gene_modules import (
    ALL_RANK_PROGRAM_SETS,
    COMPOSITION_COLUMNS,
    COMPOSITION_PROGRAM_SETS,
    FIBROSIS_VIEW_COLUMNS,
    RANK_PROGRAM_COLUMNS,
    RANK_PROGRAM_SETS,
    SCAR_DISCRIMINATIVE_COLUMNS,
    SCAR_DISCRIMINATIVE_PROGRAM_SETS,
)


def _mean_rank(ranks: pd.Series, genes: list[str]) -> float | None:
    available = [g for g in genes if g in ranks.index]
    if not available:
        return None
    return float(ranks[available].mean())


def coverage_adjusted_from_expr(expr: pd.DataFrame, program_sets: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sample_id, row in expr.iterrows():
        ranks = pd.to_numeric(row, errors="coerce").rank(pct=True, method="average")
        out = {"sample_id": str(sample_id)}
        for name, genes in program_sets.items():
            pos_genes = list(genes.get("positive", []))
            neg_genes = list(genes.get("negative", []))
            present = [g for g in (*pos_genes, *neg_genes) if g in ranks.index]
            min_genes = int(genes.get("min_genes_present", 1) or 1)
            if len(present) < min_genes:
                out[name] = 0.0
                continue
            pos = _mean_rank(ranks, pos_genes)
            neg = _mean_rank(ranks, neg_genes)
            if pos is None and neg is None:
                out[name] = 0.0
            elif neg is None:
                out[name] = float(pos)
            elif pos is None:
                out[name] = float(-neg)
            else:
                out[name] = float(pos - neg)
        rows.append(out)
    return pd.DataFrame(rows).set_index("sample_id")


def build_product_feature_views(expr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return feature views matching breakthrough training names."""
    fused = coverage_adjusted_from_expr(expr, ALL_RANK_PROGRAM_SETS)
    scar = coverage_adjusted_from_expr(
        expr, {**RANK_PROGRAM_SETS, **SCAR_DISCRIMINATIVE_PROGRAM_SETS}
    )
    composition = coverage_adjusted_from_expr(expr, COMPOSITION_PROGRAM_SETS)
    fibrosis_cols = [c for c in FIBROSIS_VIEW_COLUMNS if c in fused.columns]
    scar_cols = [c for c in [*RANK_PROGRAM_COLUMNS, *SCAR_DISCRIMINATIVE_COLUMNS] if c in scar.columns]
    return {
        "fibrosis_only": fused[fibrosis_cols].copy(),
        "composition_only": composition[[c for c in COMPOSITION_COLUMNS if c in composition.columns]].copy(),
        "scar_discriminative": scar[scar_cols].copy(),
        "rank_programs_only": fused[[c for c in RANK_PROGRAM_COLUMNS if c in fused.columns]].copy(),
        "fused_multiview": fused.copy(),
    }
