#!/usr/bin/env python3
"""Freeze breakthrough sprint v2 (endpoint-pure) without mutating v1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"
CANDIDATES = PROJECT_ROOT / "data/raw/public_keloid_candidate_registry.json"
LEDGER = PROJECT_ROOT / "results/breakthrough_sprint_v2/cohort_decision_ledger.json"
V1_DIR = PROJECT_ROOT / "results/breakthrough_sprint"
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint_v2"
GENE_MODULES = PROJECT_ROOT / "scripts/gene_modules.py"
TRAIN_NESTED = PROJECT_ROOT / "scripts/train_nested_loso.py"
BUILD_CORPUS = PROJECT_ROOT / "scripts/build_training_corpus.py"


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
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}

    v1_proto = {}
    v1_lock = {}
    if (V1_DIR / "frozen_protocol.json").exists():
        v1_proto = json.loads((V1_DIR / "frozen_protocol.json").read_text())
    if (V1_DIR / "breakthrough_lockbox_once.json").exists():
        v1_lock = json.loads((V1_DIR / "breakthrough_lockbox_once.json").read_text())

    protocol = {
        "protocol": "breakthrough_sprint_v2",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "parent_protocol": "breakthrough_sprint_v1",
        "v1_immutable": {
            "directory": "results/breakthrough_sprint/",
            "frozen_protocol": "results/breakthrough_sprint/frozen_protocol.json",
            "lockbox_once": "results/breakthrough_sprint/breakthrough_lockbox_once.json",
            "breakthrough_lockbox_accession": v1_proto.get("breakthrough_lockbox", {}).get(
                "accession", "GSE185309"
            ),
            "lockbox_already_scored": bool(v1_lock),
            "do_not_rescore_v1_lockbox": True,
        },
        "product": {
            "primary": "keloid_vs_unaffected_skin",
            "primary_compartment": "bulk_tissue_or_whole_skin",
            "specialists": ["keloid_vs_normal_scar", "keloid_vs_pathologic_scar"],
            "sensitivity": ["fibroblast_keloid_binary"],
            "cascade": "specimen_aware_mixture_of_experts_with_inner_abstention",
        },
        "claim_gates": {
            "full_coverage_macro_accession_f1": 0.85,
            "selective_macro_accession_f1": 0.90,
            "selective_min_coverage": 0.60,
            "worst_accession_f1": 0.75,
            "lockbox_same_rule": True,
            "public_data_interim_targets": {
                "remove_catastrophic_inversion": True,
                "worst_fold_f1_ge": 0.50,
                "stable_macro_f1_ge": 0.70,
            },
            "selection": "equal_accession_mean_donor_f1_with_lower_tail_tiebreak",
            "no_outer_test_fallback": True,
        },
        "feature_views": [
            "fibrosis_only",
            "scar_discriminative",
            "composition_only",
            "fused_multiview",
        ],
        "model_grid": ["ridge_logreg", "elastic_net_logreg", "linear_svm"],
        "normalization_grid": ["none", "train_accession_zscore"],
        "calibration": ["none", "platt_inner_oof"],
        "selective_prediction": {
            "labels": ["keloid", "non_keloid", "abstain"],
            "tune_on": "inner_equal_accession_oof_only",
            "coverage_constraint": ">=0.60",
            "report_nonselective_alongside": True,
        },
        "endpoint_purity": {
            "ledger": "results/breakthrough_sprint_v2/cohort_decision_ledger.json",
            "primary_skin_include": ledger.get("primary_skin_include", []),
            "primary_skin_exclude": ledger.get("primary_skin_exclude", []),
            "interpretation_only_additions": ledger.get("interpretation_only_additions", []),
        },
        "development_accessions": sorted(
            {
                e["accession"]
                for e in registry.get("development", [])
                if e.get("role") == "development"
            }
            | set(registry.get("original_10_comparator", []))
            | {"GSE173900", "GSE190626", "GSE210434", "GSE245660"}
            - set(ledger.get("interpretation_only_additions", []))
        ),
        "interpretation_only": sorted(
            {e["accession"] for e in registry.get("interpretation_only", [])}
            | set(ledger.get("interpretation_only_additions", []))
        ),
        "ladder_reserved_lockbox": ["GSE212954"],
        "historical_lockbox_comparators": ["GSE237752", "GSE191067", "GSE185309"],
        "breakthrough_lockbox": {
            "accession": "GSE218007",
            "role": "breakthrough_lockbox_v2",
            "rationale": (
                "Independent of v1 lockbox GSE185309 (already scored once; immutable). "
                "Multi-donor Affymetrix keloid transcriptome (n≈29). Expression withheld "
                "until v2 nested protocol is frozen and public-data interim targets are assessed. "
                "Requires CEL→symbol processing before one-shot score."
            ),
            "endpoint_fit": ["keloid_vs_unaffected_skin"],
            "expression_status": "withheld_until_v2_go",
            "download_after_freeze": {
                "series_matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE218nnn/GSE218007/matrix/GSE218007_series_matrix.txt.gz",
                "supplementary": [
                    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE218nnn/GSE218007/suppl/GSE218007_RAW.tar"
                ],
            },
        },
        "targeted_ingest_priority": [
            "GSE218007",
            "GSE307504",
            "GSE125022",
            "GSE90051",
            "GSE44270",
        ],
        "immutable_comparators": {
            "accuracy_ladder_baseline": "results/accuracy_ladder_baseline/",
            "breakthrough_sprint_v1": "results/breakthrough_sprint/",
            "original_10": registry.get("original_10_comparator", []),
        },
        "hashes": {
            "public_keloid_cohorts.json": file_sha256(REGISTRY),
            "public_keloid_candidate_registry.json": file_sha256(CANDIDATES),
            "cohort_decision_ledger.json": file_sha256(LEDGER),
            "gene_modules.py": file_sha256(GENE_MODULES),
            "train_nested_loso.py": file_sha256(TRAIN_NESTED),
            "build_training_corpus.py": file_sha256(BUILD_CORPUS),
        },
        "discovery_snapshot": {
            "n_candidates": candidates.get("n_candidates"),
            "n_novel": candidates.get("n_novel"),
            "generated_at": candidates.get("generated_at"),
        },
        "ablation_sequence": [
            "stage_a_endpoint_pure_baseline",
            "stage_b_donor_selection_calibration",
            "stage_c_train_only_normalization",
            "stage_d_specimen_routed_cascade",
        ],
        "go_no_go": {
            "if_public_macro_below_0_85": "execute_matched_cohort_protocol",
            "matched_cohort_protocol": "manuscript/matched_cohort_protocol.md",
            "do_not_architecture_chase_below_interim": True,
        },
    }
    out = args.out_dir / "frozen_protocol.json"
    out.write_text(json.dumps(protocol, indent=2))
    adj = {
        "protocol": "breakthrough_sprint_v2",
        "v1_lockbox_immutable": "GSE185309",
        "v2_lockbox_withheld": "GSE218007",
        "ladder_reserved": ["GSE212954"],
        "primary_skin_include": protocol["endpoint_purity"]["primary_skin_include"],
        "primary_skin_exclude": protocol["endpoint_purity"]["primary_skin_exclude"],
    }
    (args.out_dir / "cohort_adjudication.json").write_text(json.dumps(adj, indent=2))
    print(json.dumps({"written": str(out), "v2_lockbox": "GSE218007", "v1_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
