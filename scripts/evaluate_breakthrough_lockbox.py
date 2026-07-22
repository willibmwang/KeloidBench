#!/usr/bin/env python3
"""Score the breakthrough lockbox once under the frozen cascade protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_public_keloid import download_cohort, load_registry, process_cohort
from expression_processing import write_expression_artifacts
from train_nested_loso import BINARY_CLASSES, evaluate_fixed, load_inputs
from train_breakthrough_cascade import tune_selective_threshold, PRIMARY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint"
PROTOCOL_PATH = OUT_DIR / "breakthrough_frozen_model.json"
SPRINT_PROTOCOL = OUT_DIR / "frozen_protocol.json"
PUBLIC_DIR = PROJECT_ROOT / "data/processed/public_keloid"
RAW_DIR = PROJECT_ROOT / "data/raw/public_keloid"
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=OUT_DIR)
    p.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "data/processed/training")
    p.add_argument("--download-lockbox", action="store_true")
    p.add_argument("--accession", default=None, help="Override lockbox accession")
    return p.parse_args()


def discover_supplementary(accession: str) -> list[str]:
    """Best-effort GEO supplementary listing via FTP directory index."""
    import re
    import urllib.request

    prefix = accession[:6] + "nnn" if accession.startswith("GSE") and len(accession) >= 6 else accession
    # GSE185309 -> GSE185nnn
    m = re.match(r"(GSE\d{1,3})", accession)
    if m:
        stem = m.group(1)
        # pad style used by NCBI: GSE185nnn for GSE185309
        digits = accession[3:]
        if len(digits) >= 3:
            prefix = f"GSE{digits[:-3]}nnn" if len(digits) > 3 else f"GSE{digits}nnn"
            # Actually NCBI uses first 3 digits of number: GSE185309 -> GSE185nnn
            num = accession[3:]
            prefix = "GSE" + num[: max(1, len(num) - 3)] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/suppl/"
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not list {url}: {exc}")
        return []
    files = sorted(set(re.findall(rf"href=\"({accession}[^\"]+)\"", html)))
    return [url + f for f in files if not f.endswith("/")]


def ingest_lockbox(accession: str) -> dict:
    """Download and merge breakthrough lockbox into public_keloid artifacts."""
    from preprocess_public_keloid import process_cohort

    num = accession[3:]
    folder = "GSE" + num[: len(num) - 3] + "nnn" if len(num) > 3 else accession
    series = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder}/{accession}/matrix/{accession}_series_matrix.txt.gz"
    urls = discover_supplementary(accession)
    preferred = [
        u
        for u in urls
        if any(tok in u.lower() for tok in ["count", "tpm", "fpkm", "expression", "gene", ".txt", ".csv", ".tsv", ".xlsx"])
        and "raw.tar" not in u.lower()
    ]
    entry = {
        "accession": accession,
        "role": "lockbox",
        "platform_id": "GPL24676",
        "modality": "bulk_rnaseq",
        "download": {"series_matrix": series, "supplementary": preferred or urls[:5]},
        "allow_accession_level_grouping": False,
    }
    print(json.dumps({"ingest_entry": {**entry, "n_suppl": len(entry["download"]["supplementary"])}}, indent=2))
    paths = download_cohort(entry, RAW_DIR, force=False, include_reserved_expression=True)
    result = process_cohort(entry, paths)
    if result is None:
        return {"accession": accession, "status": "no_usable_files"}
    meta, expr, summary = result
    if meta is None:
        return summary

    # Force lockbox role.
    meta = meta.copy()
    meta["cohort_role"] = "lockbox"
    meta["lockbox"] = True
    meta["eligible_for_keloid_pretraining"] = False

    wide_path = PUBLIC_DIR / "public_keloid_expression_wide.parquet"
    meta_path = PUBLIC_DIR / "public_keloid_sample_metadata.parquet"
    if not wide_path.exists():
        write_expression_artifacts(
            out_dir=PUBLIC_DIR,
            prefix="public_keloid",
            metadata=meta,
            expr=expr,
            dataset_summaries=[summary],
            skipped=[],
        )
    else:
        existing_meta = pd.read_parquet(meta_path) if meta_path.exists() else pd.read_parquet(wide_path)
        vocab = [
            line.strip()
            for line in (PUBLIC_DIR / "public_keloid_gene_vocab.txt").read_text().splitlines()
            if line.strip()
        ]
        existing = pd.read_parquet(wide_path)
        base_expr = existing.set_index("sample_id").reindex(columns=vocab).fillna(0.0)
        if (existing_meta["accession"].astype(str) == accession).any():
            return {"accession": accession, "status": "already_present", **summary}
        all_genes = sorted(set(vocab) | set(expr.columns))
        base_expr = base_expr.reindex(columns=all_genes, fill_value=0.0)
        add_expr = expr.reindex(columns=all_genes, fill_value=0.0)
        add_expr.index = meta["sample_id"].tolist()
        keep = ~existing_meta["accession"].astype(str).eq(accession)
        merged_meta = pd.concat([existing_meta.loc[keep].reset_index(drop=True), meta], ignore_index=True, sort=False)
        parts = []
        for sid in merged_meta["sample_id"]:
            if sid in add_expr.index:
                parts.append(add_expr.loc[[sid]])
            else:
                parts.append(base_expr.loc[[sid]])
        merged_expr = pd.concat(parts, ignore_index=True).fillna(0.0)
        write_expression_artifacts(
            out_dir=PUBLIC_DIR,
            prefix="public_keloid",
            metadata=merged_meta,
            expr=merged_expr,
            dataset_summaries=[{"accession": accession, "role": "lockbox", **summary}],
            skipped=[],
        )

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "scripts")}
    subprocess.check_call([PYTHON, str(PROJECT_ROOT / "scripts/build_training_corpus.py")], cwd=PROJECT_ROOT, env=env)
    summary["status"] = "ok"
    return summary


def main() -> None:
    args = parse_args()
    frozen = json.loads(PROTOCOL_PATH.read_text()) if PROTOCOL_PATH.exists() else {}
    sprint = json.loads(SPRINT_PROTOCOL.read_text()) if SPRINT_PROTOCOL.exists() else {}
    accession = args.accession or frozen.get("breakthrough_lockbox") or sprint.get("breakthrough_lockbox", {}).get(
        "accession"
    )
    if not accession:
        raise SystemExit("No breakthrough lockbox accession configured")

    # Guard: never score before protocol freeze.
    if not SPRINT_PROTOCOL.exists():
        raise SystemExit("frozen_protocol.json missing; refuse to open lockbox expression")

    ingest = None
    if args.download_lockbox:
        print(f"=== downloading breakthrough lockbox {accession} ===")
        ingest = ingest_lockbox(accession)

    manifest, features, splits = load_inputs(args.training_dir)
    cfg = frozen.get("primary_config") or {
        "feature_set": "fused_multiview",
        "model": "ridge_logreg",
        "weight_mode": "accession_donor",
        "threshold": 0.5,
    }
    if cfg["feature_set"] not in features:
        # Fallback to available view.
        for alt in ["scar_discriminative", "rank_programs_only", "fibrosis_only"]:
            if alt in features:
                cfg["feature_set"] = alt
                break
    if PRIMARY not in manifest.columns:
        raise SystemExit(f"{PRIMARY} missing from manifest")
    target = manifest.set_index("sample_id")[PRIMARY]

    rows = []
    for split in splits:
        if split["task"] != PRIMARY:
            continue
        if not split["split_name"].startswith("leave_accession_out_"):
            continue
        acc = split["split_name"].replace("leave_accession_out_", "")
        if acc != accession:
            continue
        result = evaluate_fixed(
            feature_table=features[cfg["feature_set"]],
            target=target,
            train_ids=split["train_sample_ids"],
            test_ids=split["test_sample_ids"],
            manifest=manifest,
            model_name=cfg["model"],
            feature_name=cfg["feature_set"],
            weight_mode=cfg["weight_mode"],
            random_state=13,
            threshold=cfg["threshold"],
        )
        if result is None:
            rows.append({"accession": acc, "status": "skipped_invalid_split"})
            continue
        y = result["y_true"]
        p = result["probs"]
        thr_conf, sel_f1, cov = tune_selective_threshold(y, p, 0.60)
        rows.append(
            {
                "accession": acc,
                "status": "ok",
                "endpoint": PRIMARY,
                "eval_mode": "breakthrough_lockbox_once",
                **cfg,
                "weighted_f1": result["weighted_f1"],
                "macro_f1": result["macro_f1"],
                "auroc": result["auroc"],
                "balanced_accuracy": result["balanced_accuracy"],
                "donor_weighted_f1": result["donor_weighted_f1"],
                "selective_f1": sel_f1,
                "selective_coverage": cov,
                "confidence_threshold": thr_conf,
                "n_test": result["n_test"],
                "n_test_donors": result["n_test_donors"],
                "classes": list(BINARY_CLASSES),
            }
        )

    out = {
        "frozen_config": cfg,
        "lockbox_accession": accession,
        "ingest": ingest,
        "results": rows,
        "claim_rule": frozen.get("cascade", {}),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "breakthrough_lockbox_once.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(args.results_dir / "breakthrough_lockbox_once.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
