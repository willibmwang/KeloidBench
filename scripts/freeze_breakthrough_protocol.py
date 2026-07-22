#!/usr/bin/env python3
"""Freeze the breakthrough-sprint protocol before any new lockbox expression access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"
CANDIDATES = PROJECT_ROOT / "data/raw/public_keloid_candidate_registry.json"
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint"
GENE_MODULES = PROJECT_ROOT / "scripts/gene_modules.py"
TRAIN_NESTED = PROJECT_ROOT / "scripts/train_nested_loso.py"


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    candidates = json.loads(CANDIDATES.read_text()) if CANDIDATES.exists() else {}

    # GSE212954 is reserved by the accuracy ladder (expression still withheld at freeze time,
    # but the ladder will consume it). Breakthrough uses an independent metadata-adjudicated lockbox.
    breakthrough_lockbox = {
        "accession": "GSE185309",
        "role": "breakthrough_lockbox",
        "rationale": (
            "Multi-donor bulk RNA-seq of keratinocyte gene expression in keloid vs normal skin "
            "(n≈17). Independent of accuracy-ladder reserved lockbox GSE212954. "
            "Expression withheld until this protocol hash exists."
        ),
        "endpoint_fit": ["keloid_vs_unaffected_skin"],
        "download_after_freeze": {
            "series_matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185309/matrix/GSE185309_series_matrix.txt.gz",
            "supplementary": [
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185309/suppl/"
            ],
        },
        "expression_status": "withheld_until_protocol_freeze",
    }
    backup_lockbox = {
        "accession": "GSE218007",
        "role": "breakthrough_lockbox_backup",
        "rationale": "Larger keloid transcriptome array cohort (n≈29); use only if GSE185309 fails coverage.",
        "expression_status": "withheld",
    }

    protocol = {
        "protocol": "breakthrough_sprint_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "product": {
            "primary": "keloid_vs_unaffected_skin",
            "specialists": ["keloid_vs_normal_scar", "keloid_vs_pathologic_scar"],
            "sensitivity": ["fibroblast_keloid_binary"],
            "cascade": "specimen_aware_mixture_of_experts_with_abstention",
        },
        "claim_gates": {
            "full_coverage_macro_accession_f1": 0.85,
            "selective_macro_accession_f1": 0.90,
            "selective_min_coverage": 0.60,
            "worst_accession_f1": 0.75,
            "lockbox_same_rule": True,
            "selection": "equal_accession_mean_inner_f1_with_lower_tail_tiebreak",
            "no_outer_test_fallback": True,
        },
        "feature_views": [
            "fibrosis_only",
            "scar_discriminative",
            "composition_only",
            "fused_multiview",
        ],
        "model_grid": ["ridge_logreg", "elastic_net_logreg", "linear_svm"],
        "selective_prediction": {
            "labels": ["keloid", "non_keloid", "abstain"],
            "tune_on": "inner_equal_accession_f1",
            "coverage_constraint": ">=0.60",
            "report_nonselective_alongside": True,
        },
        "development_accessions": sorted(
            {e["accession"] for e in registry.get("development", []) if e.get("role") == "development"}
            | set(registry.get("original_10_comparator", []))
            | {"GSE173900", "GSE190626", "GSE121618"}
        ),
        "interpretation_only": [e["accession"] for e in registry.get("interpretation_only", [])],
        "ladder_reserved_lockbox": ["GSE212954"],
        "historical_lockbox_comparators": ["GSE237752", "GSE191067"],
        "breakthrough_lockbox": breakthrough_lockbox,
        "breakthrough_lockbox_backup": backup_lockbox,
        "metadata_candidates_prioritized": [
            "GSE185309",
            "GSE218007",
            "GSE90051",
            "GSE237755",
            "GSE210434",
        ],
        "immutable_comparators": {
            "accuracy_ladder_baseline": "results/accuracy_ladder_baseline/",
            "original_10": registry.get("original_10_comparator", []),
        },
        "hashes": {
            "public_keloid_cohorts.json": file_sha256(REGISTRY),
            "public_keloid_candidate_registry.json": file_sha256(CANDIDATES),
            "gene_modules.py": file_sha256(GENE_MODULES),
            "train_nested_loso.py": file_sha256(TRAIN_NESTED),
        },
        "discovery_snapshot": {
            "n_candidates": candidates.get("n_candidates"),
            "n_novel": candidates.get("n_novel"),
            "generated_at": candidates.get("generated_at"),
        },
        "ablation_sequence": [
            "endpoint_cleanup",
            "fibrosis_view",
            "scar_discriminative_view",
            "composition_view",
            "fused_experts",
            "selective_cascade",
        ],
    }
    out = args.out_dir / "frozen_protocol.json"
    out.write_text(json.dumps(protocol, indent=2))
    # Also write a lightweight cohort adjudication table.
    adj = {
        "breakthrough_lockbox": breakthrough_lockbox["accession"],
        "ladder_reserved": ["GSE212954"],
        "do_not_download_before_hash": [breakthrough_lockbox["accession"], "GSE212954", "GSE218007"],
        "development_priority_skin": ["GSE173900", "GSE190626", "GSE121618", "GSE90051"],
        "development_priority_scar": ["GSE188952", "GSE245660", "GSE210434"],
    }
    (args.out_dir / "cohort_adjudication.json").write_text(json.dumps(adj, indent=2))
    print(json.dumps({"written": str(out), "lockbox": breakthrough_lockbox["accession"]}, indent=2))


if __name__ == "__main__":
    main()
