"""Cross-study normalization for keloid baselines.

Includes both historical *transductive* helpers and leakage-safe *train-only*
reference quantile / location-scale (ComBat-lite) transforms used by the
accuracy_085_push campaign.
"""

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


# ---------------------------------------------------------------------------
# Train-only (leakage-safe) transforms for nested LOSO
# ---------------------------------------------------------------------------


def train_only_accession_zscore(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    accession_by_sample: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score each train accession in-place; map test onto global train stats.

    Unseen test accessions use the pooled train mean/std (no test labels used).
    """
    out_train = x_train.copy()
    out_test = x_test.copy()
    global_mean = out_train.mean(axis=0, skipna=True)
    global_std = out_train.std(axis=0, skipna=True).replace(0, np.nan)
    stats: dict[str, tuple[pd.Series, pd.Series]] = {}
    for acc, sids in group_ids_by_accession(train_ids, accession_by_sample).items():
        present = [sid for sid in sids if sid in out_train.index]
        if len(present) < 2:
            continue
        block = out_train.loc[present]
        means = block.mean(axis=0, skipna=True)
        stds = block.std(axis=0, skipna=True).replace(0, np.nan)
        stats[str(acc)] = (means, stds)
        out_train.loc[present] = ((block - means) / stds).fillna(0.0)
    for sid in out_test.index:
        acc = accession_by_sample.get(str(sid), "unknown")
        means, stds = stats.get(acc, (global_mean, global_std))
        out_test.loc[sid] = ((out_test.loc[sid] - means) / stds).fillna(0.0)
    return out_train.fillna(0.0), out_test.fillna(0.0)


def _reference_quantiles(x_train: pd.DataFrame, n_quantiles: int = 101) -> dict[str, np.ndarray]:
    """Pooled train feature-wise empirical quantiles (reference distribution)."""
    qs = np.linspace(0.0, 1.0, n_quantiles)
    refs: dict[str, np.ndarray] = {}
    for col in x_train.columns:
        vals = pd.to_numeric(x_train[col], errors="coerce").dropna().to_numpy()
        if len(vals) < 2:
            refs[col] = np.full(n_quantiles, 0.0)
        else:
            refs[col] = np.quantile(vals, qs)
    return refs


def _map_to_reference(block: pd.DataFrame, refs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Map each column of `block` onto the reference quantile distribution."""
    out = block.copy()
    qs = np.linspace(0.0, 1.0, next(iter(refs.values())).shape[0]) if refs else np.linspace(0, 1, 101)
    for col in out.columns:
        ref = refs.get(col)
        vals = pd.to_numeric(out[col], errors="coerce")
        if ref is None or vals.notna().sum() < 2:
            out[col] = vals.fillna(0.0)
            continue
        # Rank within the block → interpolate onto reference quantiles.
        ranks = vals.rank(method="average", pct=True)
        mapped = np.interp(ranks.fillna(0.5).to_numpy(), qs, ref)
        out[col] = mapped
    return out


def train_only_reference_quantile(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    accession_by_sample: dict[str, str],
    n_quantiles: int = 101,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a pooled train quantile reference; map each accession onto it.

    Train accessions are mapped using their own within-accession ranks.
    The held-out test accession is mapped using *its* within-accession ranks
    onto the *train* reference (labels unused).
    """
    refs = _reference_quantiles(x_train, n_quantiles=n_quantiles)
    out_train = x_train.copy()
    for _acc, sids in group_ids_by_accession(train_ids, accession_by_sample).items():
        present = [sid for sid in sids if sid in out_train.index]
        if not present:
            continue
        out_train.loc[present] = _map_to_reference(out_train.loc[present], refs)
    out_test = _map_to_reference(x_test.copy(), refs)
    return out_train.fillna(0.0), out_test.fillna(0.0)


def train_only_combat_lite(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    accession_by_sample: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Location/scale batch correction with accession as batch (train-fit only).

    Estimates a grand mean/var on train, plus per-accession location/scale.
    For the held-out test accession, estimate its location/scale from unlabeled
    test features only and standardize onto the train grand distribution.
    This is a simplified parametric ComBat (no empirical Bayes shrinkage).
    """
    out_train = x_train.copy().astype(float)
    out_test = x_test.copy().astype(float)
    grand_mean = out_train.mean(axis=0, skipna=True)
    grand_std = out_train.std(axis=0, skipna=True).replace(0, np.nan)

    # Standardize train accessions onto grand distribution.
    for _acc, sids in group_ids_by_accession(train_ids, accession_by_sample).items():
        present = [sid for sid in sids if sid in out_train.index]
        if len(present) < 2:
            # Fall back to grand stats.
            block = out_train.loc[present]
            out_train.loc[present] = ((block - grand_mean) / grand_std).fillna(0.0)
            continue
        block = out_train.loc[present]
        b_mean = block.mean(axis=0, skipna=True)
        b_std = block.std(axis=0, skipna=True).replace(0, np.nan)
        standardized = (block - b_mean) / b_std
        out_train.loc[present] = (standardized * grand_std + grand_mean).fillna(0.0)

    # Map test accession onto grand train distribution.
    if len(out_test) >= 2:
        b_mean = out_test.mean(axis=0, skipna=True)
        b_std = out_test.std(axis=0, skipna=True).replace(0, np.nan)
        standardized = (out_test - b_mean) / b_std
        out_test = (standardized * grand_std + grand_mean).fillna(0.0)
    else:
        out_test = ((out_test - grand_mean) / grand_std).fillna(0.0)
    return out_train.fillna(0.0), out_test.fillna(0.0)


def apply_train_only_normalization(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    accession_by_sample: dict[str, str],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch leakage-safe normalization modes used by nested LOSO."""
    if mode in {None, "none", ""}:
        return x_train, x_test
    if mode == "train_accession_zscore":
        return train_only_accession_zscore(
            x_train, x_test, train_ids, test_ids, accession_by_sample
        )
    if mode in {"train_reference_quantile", "reference_quantile"}:
        return train_only_reference_quantile(
            x_train, x_test, train_ids, test_ids, accession_by_sample
        )
    if mode in {"train_combat_lite", "combat_lite"}:
        return train_only_combat_lite(
            x_train, x_test, train_ids, test_ids, accession_by_sample
        )
    raise ValueError(f"Unknown train-only normalization mode: {mode}")
