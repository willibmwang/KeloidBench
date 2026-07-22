#!/usr/bin/env python3
"""Freeze one nested current10 config and evaluate lockbox cohorts once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from expression_processing import write_expression_artifacts
from preprocess_public_keloid import download_cohort, load_registry, process_cohort
from train_nested_loso import BINARY_CLASSES, evaluate_fixed, load_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"
PUBLIC_DIR = PROJECT_ROOT / "data/processed/public_keloid"
RAW_DIR = PROJECT_ROOT / "data/raw/public_keloid"
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/public_expansion_loso")
    p.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "data/processed/training")
    p.add_argument(
        "--accessions",
        nargs="*",
        default=None,
        help="Optional lockbox accession filter (default: all registry lockbox).",
    )
    p.add_argument(
        "--download-reserved-lockbox",
        action="store_true",
        help="Download download_after_freeze assets, merge reserved lockbox into artifacts, rebuild corpus, then score.",
    )
    return p.parse_args()


def majority_config(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {
            "feature_set": "profibrotic_module_only",
            "model": "elastic_net_logreg",
            "weight_mode": "none",
            "threshold": 0.5,
        }
    keys = list(zip(selected["feature_set"], selected["model"], selected["weight_mode"]))
    (feature_set, model, weight_mode), _ = Counter(keys).most_common(1)[0]
    thr = float(selected["threshold"].median()) if "threshold" in selected.columns else 0.5
    return {
        "feature_set": feature_set,
        "model": model,
        "weight_mode": weight_mode,
        "threshold": thr,
    }


def merge_reserved_lockbox(accessions: set[str]) -> list[dict]:
    """Download/process reserved lockbox cohorts and merge into existing public_keloid artifacts."""
    registry = load_registry(REGISTRY_PATH)
    wide_path = PUBLIC_DIR / "public_keloid_expression_wide.parquet"
    if not wide_path.exists():
        raise SystemExit("public_keloid_expression_wide.parquet missing; run preprocess_public_keloid first")

    existing = pd.read_parquet(wide_path)
    meta_cols = [c for c in existing.columns if c in {
        "sample_id", "accession", "sample_title", "platform_id", "modality", "source_dataset",
        "disease_domain", "disease_label", "keloid_vs_normal", "lesional_status", "scar_type",
        "cell_type", "treatment", "patient_id", "contrast_type", "eligible_for_keloid_pretraining",
        "encoder_task", "encoder_prompt", "encoder_response", "eval_grain", "cohort_role",
        "allow_accession_level_grouping", "lockbox",
    } or c.endswith("_score") or c == "fibrotic_activity_score"]
    # Prefer dedicated metadata file when present.
    meta_path = PUBLIC_DIR / "public_keloid_sample_metadata.parquet"
    if meta_path.exists():
        base_meta = pd.read_parquet(meta_path)
    else:
        base_meta = existing[[c for c in meta_cols if c in existing.columns]].copy()

    vocab = [line.strip() for line in (PUBLIC_DIR / "public_keloid_gene_vocab.txt").read_text().splitlines() if line.strip()]
    base_expr = existing.set_index("sample_id").reindex(columns=vocab).fillna(0.0)

    summaries = []
    new_metas = []
    new_exprs = []
    for entry in registry.get("lockbox", []):
        acc = entry["accession"]
        if acc not in accessions:
            continue
        if (base_meta["accession"].astype(str) == acc).any():
            summaries.append({"accession": acc, "status": "already_present"})
            continue
        print(f"=== reserved lockbox ingest {acc} ===")
        paths = download_cohort(entry, RAW_DIR, force=False, include_reserved_expression=True)
        result = process_cohort(entry, paths)
        if result is None:
            summaries.append({"accession": acc, "status": "no_usable_files"})
            continue
        meta, expr, summary = result
        if meta is None:
            summaries.append(summary)
            continue
        new_metas.append(meta)
        new_exprs.append(expr)
        summaries.append(summary)

    if not new_metas:
        print("No new reserved lockbox cohorts to merge.")
        return summaries

    add_meta = pd.concat(new_metas, ignore_index=True, sort=False)
    add_expr = pd.concat(new_exprs, ignore_index=True, sort=False).fillna(0.0)
    # Align gene columns with existing vocab union.
    all_genes = sorted(set(vocab) | set(add_expr.columns))
    base_expr = base_expr.reindex(columns=all_genes, fill_value=0.0)
    add_expr = add_expr.reindex(columns=all_genes, fill_value=0.0)
    # Drop any stale rows for these accessions then append.
    keep = ~base_meta["accession"].isin(add_meta["accession"].unique())
    merged_meta = pd.concat([base_meta.loc[keep].reset_index(drop=True), add_meta], ignore_index=True, sort=False)
    add_expr_idx = add_expr.copy()
    add_expr_idx.index = add_meta["sample_id"].tolist()
    base_expr_idx = base_expr.copy()
    expr_parts = []
    for sid in merged_meta["sample_id"]:
        if sid in add_expr_idx.index:
            expr_parts.append(add_expr_idx.loc[[sid]])
        else:
            expr_parts.append(base_expr_idx.loc[[sid]])
    merged_expr = pd.concat(expr_parts, ignore_index=True).fillna(0.0)

    roles_path = PUBLIC_DIR / "public_keloid_cohort_roles.json"
    roles = json.loads(roles_path.read_text()) if roles_path.exists() else {}
    prev_summaries = [{"accession": a, "role": "lockbox"} for a in roles.get("lockbox", [])]
    prev_summaries += [{"accession": a, "role": "development"} for a in roles.get("development", [])]
    write_expression_artifacts(
        out_dir=PUBLIC_DIR,
        prefix="public_keloid",
        metadata=merged_meta,
        expr=merged_expr,
        dataset_summaries=prev_summaries + summaries,
        skipped=roles.get("skipped", []),
        jsonl_top_genes=2048,
    )
    roles.setdefault("lockbox", [])
    for acc in add_meta["accession"].unique():
        if acc not in roles["lockbox"]:
            roles["lockbox"].append(acc)
    roles_path.write_text(json.dumps(roles, indent=2))

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "scripts")}
    subprocess.check_call(
        [PYTHON, str(PROJECT_ROOT / "scripts/build_training_corpus.py")],
        cwd=PROJECT_ROOT,
        env=env,
    )
    return summaries


def main() -> None:
    args = parse_args()
    registry = json.loads(REGISTRY_PATH.read_text())
    lockbox = {e["accession"] for e in registry.get("lockbox", [])}
    if args.accessions:
        lockbox = set(args.accessions)
    original_10 = set(registry.get("original_10_comparator", []))

    selected_path = args.results_dir / "public_expansion_nested_selected.csv"
    protocol_path = args.results_dir / "frozen_protocol.json"
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    cur = (
        selected[selected.eval_mode.eq("current10") & selected.endpoint.eq("keloid_binary")]
        if len(selected)
        else selected
    )
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        frozen = protocol.get("frozen_config") or majority_config(cur)
    else:
        frozen = majority_config(cur)
        protocol = {
            "protocol": "accuracy_evidence_ladder_v1",
            "frozen_config": frozen,
            "source": "majority vote over current10 nested_selected keloid_binary folds",
        }
        args.results_dir.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol, indent=2))

    ingest_summaries = []
    if args.download_reserved_lockbox:
        print("=== merging reserved lockbox expression into corpus ===")
        ingest_summaries = merge_reserved_lockbox(lockbox)

    manifest, features, splits = load_inputs(args.training_dir)
    if frozen["feature_set"] not in features:
        raise SystemExit(f"Frozen feature set missing: {frozen['feature_set']}")
    target = manifest.set_index("sample_id")["keloid_binary"]

    rows = []
    for split in splits:
        if split["task"] != "keloid_binary":
            continue
        if not split["split_name"].startswith("leave_accession_out_"):
            continue
        acc = split["split_name"].replace("leave_accession_out_", "")
        if acc not in lockbox:
            continue
        result = evaluate_fixed(
            feature_table=features[frozen["feature_set"]],
            target=target,
            train_ids=split["train_sample_ids"],
            test_ids=split["test_sample_ids"],
            manifest=manifest,
            model_name=frozen["model"],
            feature_name=frozen["feature_set"],
            weight_mode=frozen["weight_mode"],
            random_state=13,
            threshold=frozen["threshold"],
        )
        if result is None:
            rows.append({"accession": acc, "status": "skipped_invalid_split"})
            continue
        rows.append(
            {
                "accession": acc,
                "status": "ok",
                "endpoint": "keloid_binary",
                "eval_mode": "lockbox_frozen_once",
                "feature_set": frozen["feature_set"],
                "model": frozen["model"],
                "weight_mode": frozen["weight_mode"],
                "threshold": frozen["threshold"],
                "weighted_f1": result["weighted_f1"],
                "macro_f1": result["macro_f1"],
                "auroc": result["auroc"],
                "balanced_accuracy": result["balanced_accuracy"],
                "n_test": result["n_test"],
                "n_test_donors": result["n_test_donors"],
                "donor_weighted_f1": result["donor_weighted_f1"],
                "classes": list(BINARY_CLASSES),
                "prospective": acc == "GSE212954",
            }
        )

    out = {
        "frozen_config": frozen,
        "source": protocol.get(
            "source",
            "majority vote over current10 nested_selected keloid_binary folds",
        ),
        "original_10_comparator": sorted(original_10),
        "lockbox_accessions": sorted(lockbox),
        "ingest_summaries": ingest_summaries,
        "results": rows,
        "claim_rule": (
            "Broad 0.80 may be claimed only if current10 nested macro-accession F1 and "
            "a fresh multi-donor untouched lockbox both meet the preregistered gate."
        ),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "lockbox_frozen_once.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(args.results_dir / "lockbox_frozen_once.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
