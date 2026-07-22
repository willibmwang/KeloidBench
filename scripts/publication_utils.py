"""Shared helpers for SpheroScar publication reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gene_modules import EXPANDED_PROFIBROTIC_GENES, MODULE_GENES

__all__ = [
    "MODULE_GENES",
    "EXPANDED_PROFIBROTIC_GENES",
    "split_family",
    "bootstrap_metric_rows",
]


def split_family(split_name: str) -> str:
    if split_name.startswith("leave_accession_out_"):
        return "leave_accession_out"
    if split_name.startswith("grouped_"):
        return "grouped"
    if split_name.startswith("leave_source_out_"):
        return "leave_source_out"
    if split_name.startswith("leave_patient_out_"):
        return "leave_patient_out"
    return "other"


def bootstrap_metric_rows(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    *,
    n_bootstrap: int = 500,
    random_state: int = 13,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        values = group[metric_col].dropna().to_numpy()
        if len(values) == 0:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        boot = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boot.append(float(np.mean(sample)))
        boot_arr = np.asarray(boot)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "metric": metric_col,
                "mean": float(np.mean(values)),
                "ci_low": float(np.quantile(boot_arr, 0.025)),
                "ci_high": float(np.quantile(boot_arr, 0.975)),
                "n": int(len(values)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
