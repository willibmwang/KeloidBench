#!/usr/bin/env python3
"""Preregistered cohort decision ledger for breakthrough sprint v2.

Decisions are biological / design-based (compartment, treatment, donors),
not chosen from outer-fold F1 scores.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = PROJECT_ROOT / "data/processed/training/sample_provenance_audit.csv"
ELIGIBILITY = PROJECT_ROOT / "data/processed/training/endpoint_eligibility_matrix.csv"
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint_v2"

# Hard decisions for endpoint-pure v2 (pre-registered before retrain).
DECISIONS: dict[str, dict] = {
    "GSE282479": {
        "role": "fibroblast_sensitivity_only",
        "compartment": "primary_fibroblast",
        "primary_skin": "exclude",
        "rationale": "Isolated fibroblasts forced into tissue skin endpoint caused complete LOSO inversion in v1.",
    },
    "GSE303591": {
        "role": "interpretation_only",
        "compartment": "cell_line",
        "primary_skin": "exclude",
        "rationale": "Two dermal cell lines with technical replicates, not independent patient donors.",
    },
    "GSE232079": {
        "role": "interpretation_only",
        "compartment": "cell_line",
        "primary_skin": "exclude",
        "rationale": "Commercial / immortalized fibroblast lines under DMSO vehicle; cell-line domain.",
    },
    "GSE246562": {
        "role": "auxiliary_stiffness_only",
        "compartment": "primary_fibroblast",
        "primary_skin": "exclude",
        "rationale": "Stiffness perturbation arm; keep for stiffness_response only.",
    },
    "GSE121618": {
        "role": "compartment_specialist_endothelial",
        "compartment": "endothelial",
        "primary_skin": "exclude",
        "rationale": "Keloid vs normal endothelial cells; not whole-skin tissue.",
    },
    "GSE145725": {
        "role": "fibroblast_sensitivity_only",
        "compartment": "primary_fibroblast",
        "primary_skin": "exclude",
        "rationale": "Fibroblast cultures; route to fibroblast_keloid_binary, not tissue skin.",
    },
    "E-MTAB-2509": {
        "role": "fibroblast_sensitivity_only",
        "compartment": "primary_fibroblast",
        "primary_skin": "exclude",
        "rationale": "Fibroblast cultures with drug arms; untreated fibroblast sensitivity only.",
    },
    "GSE7890": {
        "role": "fibroblast_sensitivity_only",
        "compartment": "primary_fibroblast",
        "primary_skin": "exclude",
        "rationale": "Fibroblast cultures; normals are normal_scar and half are hydrocortisone-treated.",
    },
    "GSE44270": {
        "role": "audit_pending_donor_recovery",
        "compartment": "mixed_fibroblast_keratinocyte",
        "primary_skin": "exclude",
        "rationale": "Single pseudo-donor + mixed compartments; exclude until donor IDs recovered.",
    },
    "Sun_Burns": {
        "role": "development_tissue_caution",
        "compartment": "bulk_tissue",
        "primary_skin": "include_with_single_donor_flag",
        "rationale": "Bulk tissue keloid vs normal but n_donors=1; keep with accession-level grouping caveat.",
    },
    "GSE163973": {
        "role": "scar_and_celltype_auxiliary",
        "compartment": "scrna_pseudobulk",
        "primary_skin": "exclude",
        "rationale": "NF labeled normal_scar; use for scar/cell-type tasks, not unaffected-skin primary.",
    },
    "GSE188952": {
        "role": "scar_specialist",
        "compartment": "bulk_tissue",
        "primary_skin": "exclude",
        "rationale": "Scar-differential design (keloid/HTS/normotrophic); single donor.",
    },
    "GSE210434": {
        "role": "scar_specialist",
        "compartment": "primary_fibroblast",
        "primary_skin": "exclude",
        "rationale": "Scar-triad fibroblasts; scar specialists only.",
    },
    "GSE245660": {
        "role": "scar_specialist",
        "compartment": "bulk_tissue",
        "primary_skin": "exclude",
        "rationale": "Keloid vs immature scar tissue; pathologic-scar specialist.",
    },
    "GSE173900": {
        "role": "development_tissue",
        "compartment": "bulk_tissue",
        "primary_skin": "include",
        "rationale": "Multi-donor bulk tissue keloid vs control skin.",
    },
    "GSE190626": {
        "role": "development_tissue",
        "compartment": "bulk_tissue",
        "primary_skin": "include",
        "rationale": "Multi-donor bulk tissue keloid vs normal skin.",
    },
    "E-MTAB-4945": {
        "role": "development_tissue_row_filter",
        "compartment": "mixed_keep_tissue_rows",
        "primary_skin": "include_tissue_rows_only",
        "rationale": "Keep whole_skin/dermis/epidermis rows; drop fibroblast/keratinocyte culture rows for primary.",
    },
    "GSE181297": {
        "role": "development_tissue_pseudobulk",
        "compartment": "scrna_all_barcodes",
        "primary_skin": "include",
        "rationale": "Donor-level all-barcode pseudobulk approximating tissue mix.",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prov = pd.read_csv(PROVENANCE) if PROVENANCE.exists() else pd.DataFrame()
    elig = pd.read_csv(ELIGIBILITY) if ELIGIBILITY.exists() else pd.DataFrame()

    rows = []
    for acc, decision in sorted(DECISIONS.items()):
        n = int((prov["accession"] == acc).sum()) if len(prov) else 0
        el = elig[elig["accession"] == acc].iloc[0].to_dict() if len(elig) and acc in set(elig["accession"]) else {}
        rows.append(
            {
                "accession": acc,
                "n_profiles_in_corpus": n,
                **decision,
                "v1_skin_evaluable": bool(el.get("keloid_vs_unaffected_skin_evaluable", False)),
                "v1_normal_scar_evaluable": bool(el.get("keloid_vs_normal_scar_evaluable", False)),
                "v1_pathologic_scar_evaluable": bool(el.get("keloid_vs_pathologic_scar_evaluable", False)),
            }
        )

    ledger = {
        "protocol": "breakthrough_sprint_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "primary_skin_compartment": "bulk_tissue_or_whole_skin",
            "fibroblast_sensitivity_compartment": "primary_fibroblast_unperturbed",
            "exclude_from_primary": [
                "cell_line",
                "endothelial",
                "stiffness_response",
                "drug_perturbation",
                "scar_differential_as_skin",
            ],
            "selection_not_based_on_outer_f1": True,
        },
        "decisions": rows,
        "primary_skin_include": sorted(
            a for a, d in DECISIONS.items() if str(d["primary_skin"]).startswith("include")
        ),
        "primary_skin_exclude": sorted(
            a for a, d in DECISIONS.items() if d["primary_skin"] == "exclude"
        ),
        "interpretation_only_additions": sorted(
            a for a, d in DECISIONS.items() if d["role"] == "interpretation_only"
        ),
    }
    out = args.out_dir / "cohort_decision_ledger.json"
    out.write_text(json.dumps(ledger, indent=2))
    pd.DataFrame(rows).to_csv(args.out_dir / "cohort_decision_ledger.csv", index=False)
    print(json.dumps({"written": str(out), "n_decisions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
