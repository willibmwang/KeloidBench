"""Transductive cross-study normalization for keloid_binary baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def accession_map(manifest: pd.DataFrame) -> dict[str, str]:
    indexed = manifest.set_index("sample_id")
    return indexed["accession"].astype(str).to_dict()


def group_ids_by_accession(sample_ids: list[str], accession_by_sample: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        acc = accession_by_sample.get(sample_id, "unknown")
        groups.setdefault(acc, []).append(sample_id)
    return groups


def zscore_within_accessions(
    features: pd.DataFrame,
    sample_ids: list[str],
    accession_by_sample: dict[str, str],
) -> pd.DataFrame:
    """Transductive per-accession z-score (each study uses its own mean/std)."""
    out = features.copy()
    for _acc, sids in group_ids_by_accession(sample_ids, accession_by_sample).items():
        present = [sid for sid in sids if sid in out.index]
        if not present:
            continue
        block = out.loc[present]
        means = block.mean(axis=0, skipna=True)
        stds = block.std(axis=0, skipna=True).replace(0, np.nan)
        out.loc[present] = ((block - means) / stds).fillna(0.0)
    return out


def quantile_rank_within_accessions(
    features: pd.DataFrame,
    sample_ids: list[str],
    accession_by_sample: dict[str, str],
) -> pd.DataFrame:
    """Transductive per-accession rank-percentile normalization."""
    out = features.copy()
    for _acc, sids in group_ids_by_accession(sample_ids, accession_by_sample).items():
        present = [sid for sid in sids if sid in out.index]
        if len(present) < 2:
            if present:
                out.loc[present] = 0.5
            continue
        block = out.loc[present]
        ranked = block.rank(axis=0, pct=True, method="average")
        out.loc[present] = ranked.fillna(0.5)
    return out


def _combine_splits(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate train/val/test without losing sample_ids."""
    parts = [df for df in frames if df is not None and len(df) > 0]
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts)
    # Deduplicate on index (same sample_id), not row values.
    return combined.loc[~combined.index.duplicated(keep="first")]


def apply_normalization(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    accession_by_sample: dict[str, str],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mode == "none":
        return x_train, x_val, x_test
    if mode == "accession_zscore":
        fn = zscore_within_accessions
    elif mode == "quantile_rank":
        fn = quantile_rank_within_accessions
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    all_ids = list(dict.fromkeys(train_ids + val_ids + test_ids))
    combined = _combine_splits(x_train, x_val, x_test)
    normalized = fn(combined, all_ids, accession_by_sample)
    return (
        normalized.reindex(x_train.index),
        normalized.reindex(x_val.index) if len(x_val) else x_val,
        normalized.reindex(x_test.index),
    )
