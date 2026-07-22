#!/usr/bin/env python3
"""Score samples with the Friday product cascade (joblib Stage-B models)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from expression_processing import normalize_gene_symbol
from product_features import build_product_feature_views
from train_nested_loso import positive_proba

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DIR = PROJECT_ROOT / "results/product_friday"
PRIMARY = "keloid_vs_unaffected_skin"
ROUTE = {
    "bulk_tissue": PRIMARY,
    "scrna_pseudobulk": PRIMARY,
    "primary_fibroblast": "fibroblast_keloid_binary",
    "fibroblast": "fibroblast_keloid_binary",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product-dir", type=Path, default=PRODUCT_DIR)
    p.add_argument("--features-parquet", type=Path, default=None, help="Prebuilt feature table with sample_id index/col")
    p.add_argument("--feature-set", default=None, help="Override feature view name when using --features-parquet")
    p.add_argument("--expression-csv", type=Path, default=None, help="HGNC expression matrix")
    p.add_argument("--transpose", action="store_true", help="CSV is genes x samples (will transpose)")
    p.add_argument("--sample-meta-csv", type=Path, default=None, help="Optional CSV with sample_id,specimen_compartment")
    p.add_argument("--endpoint", default=None, help="Score a single endpoint instead of cascade routing")
    p.add_argument("--confidence-threshold", type=float, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def _load_expression(path: Path, transpose: bool) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if transpose:
        df = df.T
    df.columns = [normalize_gene_symbol(str(c)) or str(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Collapse duplicate gene symbols.
    df = df.T.groupby(level=0).mean().T
    return df


def _align_features(x: pd.DataFrame, columns: list[str], medians: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=x.index)
    for col in columns:
        if col in x.columns:
            out[col] = pd.to_numeric(x[col], errors="coerce")
        else:
            out[col] = float(medians.get(col, 0.0))
    return out.fillna(0.0)


def score_endpoint(bundle: dict, x: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    meta = bundle["meta"]
    xs = _align_features(x, meta["feature_columns"], meta["feature_medians"])
    p = positive_proba(model, xs, meta["positive_code"])
    thr = float(meta["config"].get("threshold", 0.5))
    pred = np.where(p >= thr, "keloid", "non_keloid")
    conf = np.maximum(p, 1 - p)
    return pd.DataFrame(
        {
            "sample_id": xs.index.astype(str),
            "endpoint": meta["endpoint"],
            "prob_keloid": p,
            "pred_label": pred,
            "confidence": conf,
            "decision_threshold": thr,
        }
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.product_dir / "product_manifest.json").read_text())
    conf_thr = args.confidence_threshold
    if conf_thr is None:
        conf_thr = float(manifest.get("selective_confidence_threshold_default", 0.60))

    feature_views: dict[str, pd.DataFrame] = {}
    if args.expression_csv is not None:
        expr = _load_expression(args.expression_csv, args.transpose)
        feature_views = build_product_feature_views(expr)
    elif args.features_parquet is not None:
        feat = pd.read_parquet(args.features_parquet)
        if "sample_id" in feat.columns:
            feat = feat.set_index("sample_id")
        name = args.feature_set or "composition_only"
        feature_views[name] = feat
    else:
        raise SystemExit("Provide --expression-csv or --features-parquet")

    meta = None
    if args.sample_meta_csv is not None:
        meta = pd.read_csv(args.sample_meta_csv)
        if "sample_id" not in meta.columns:
            raise SystemExit("sample-meta-csv needs sample_id")

    rows = []
    endpoints = [args.endpoint] if args.endpoint else list(manifest["endpoints"])
    for endpoint in endpoints:
        if endpoint not in manifest["endpoints"]:
            continue
        info = manifest["endpoints"][endpoint]
        bundle = joblib.load(PROJECT_ROOT / info["model_path"])
        feat_name = info["feature_set"]
        # Fall back across views if exact view missing (expression path).
        x = feature_views.get(feat_name)
        if x is None and args.feature_set:
            x = feature_views.get(args.feature_set)
        if x is None:
            x = next(iter(feature_views.values()))
        scored = score_endpoint(bundle, x)
        rows.append(scored)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Cascade routing summary (one row per sample) when multi-endpoint.
    if args.endpoint is None and meta is not None and len(out):
        wide = []
        for sid, sub in out.groupby("sample_id"):
            compartment = "bulk_tissue"
            mrow = meta.loc[meta["sample_id"].astype(str) == str(sid)]
            if len(mrow) and "specimen_compartment" in mrow.columns:
                compartment = str(mrow.iloc[0]["specimen_compartment"])
            route = ROUTE.get(compartment, PRIMARY)
            if route not in sub["endpoint"].values:
                route = PRIMARY if PRIMARY in sub["endpoint"].values else sub["endpoint"].iloc[0]
            r = sub.loc[sub["endpoint"] == route].iloc[0]
            decision = r["pred_label"] if r["confidence"] >= conf_thr else "abstain"
            wide.append(
                {
                    "sample_id": sid,
                    "specimen_compartment": compartment,
                    "route_endpoint": route,
                    "prob_keloid": r["prob_keloid"],
                    "pred_label": r["pred_label"],
                    "confidence": r["confidence"],
                    "decision": decision,
                    "confidence_threshold": conf_thr,
                }
            )
        cascade = pd.DataFrame(wide)
    else:
        cascade = out.copy()
        if len(cascade):
            cascade["decision"] = np.where(cascade["confidence"] >= conf_thr, cascade["pred_label"], "abstain")

    out_path = args.out_csv or (args.product_dir / "score_output.csv")
    cascade.to_csv(out_path, index=False)
    print(json.dumps({"written": str(out_path), "n_rows": int(len(cascade))}, indent=2))


if __name__ == "__main__":
    main()
