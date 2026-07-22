#!/usr/bin/env python3
"""Build the SpheroScar baseline training corpus.

This script creates:
- a unified profile manifest with canonical targets
- harmonized feature tables for modules, shared genes, and both
- task-specific grouped split manifests
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from gene_modules import (
    ALL_PINNED_GENES,
    ALL_RANK_PROGRAM_SETS,
    BREAKTHROUGH_FEATURE_VIEWS,
    COMPARTMENT_FEATURE_COLUMNS,
    COMPOSITION_COLUMNS,
    COMPOSITION_PROGRAM_SETS,
    EXPANDED_PROFIBROTIC_GENES,
    FIBROSIS_VIEW_COLUMNS,
    MODULE_COLUMNS,
    MODULE_GENES,
    PREREGISTERED_FEATURE_SETS,
    RANK_PROGRAM_COLUMNS,
    RANK_PROGRAM_SETS,
    ROBUST_PROGRAM_GENES,
    SCAR_DISCRIMINATIVE_COLUMNS,
    SCAR_DISCRIMINATIVE_PROGRAM_SETS,
)
from publication_utils import EXPANDED_PROFIBROTIC_GENES as _EXPANDED_ALIAS  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUT_DIR = PROCESSED_DIR / "training"
SIG_DIR = PROJECT_ROOT / "data/raw/signatures"
CORE_GENES_PATH = PROJECT_ROOT / "results/publication/meta/core_signature_genes.txt"
PUBLIC_ROLES_PATH = PROCESSED_DIR / "public_keloid/public_keloid_cohort_roles.json"
ORIGINAL_10_PATH = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"

METADATA_COLUMNS = [
    "sample_id",
    "accession",
    "sample_title",
    "platform_id",
    "modality",
    "source_dataset",
    "disease_domain",
    "disease_label",
    "keloid_vs_normal",
    "lesional_status",
    "scar_type",
    "cell_type",
    "treatment",
    "patient_id",
    "contrast_type",
    "eligible_for_keloid_pretraining",
    "encoder_task",
    "encoder_prompt",
    "encoder_response",
    "eval_grain",
    "cohort_role",
    "allow_accession_level_grouping",
    "lockbox",
]

POSITIVE_LABELS = {"keloid", "lesional"}
NEGATIVE_LABELS = {
    "normal",
    "normal_scar",
    "adjacent_normal",
    "non_lesional",
    "control",
    "hypertrophic_scar",
    "normotrophic_scar",
    "immature_scar",
}
CLEAN_POSITIVE = {"keloid"}
CLEAN_NEGATIVE = {"normal", "adjacent_normal", "normal_scar", "control"}
NORMAL_SCAR_LABELS = {"normal_scar", "normotrophic_scar"}
PATHOLOGIC_SCAR_LABELS = {"hypertrophic_scar", "immature_scar"}
UNAFFECTED_SKIN_LABELS = {"normal", "adjacent_normal", "control"}
# Susceptibility / wound-response cohorts are not lesion-status classifiers.
SUSCEPTIBILITY_TASKS = {"wound_susceptibility"}
OOD_ACCESSIONS = {"GSE3189", "GSE160536"}
EXTERNAL_FIBROSIS_ACCESSIONS = {"GSE32537", "GSE48149", "GSE58095"}
TREATED_TOKENS = {"hydrocortisone", "rapamycin", "tacrolimus"}
FIBROBLAST_TOKENS = {"fibroblast", "fibroblasts", "dermal_fibroblast", "kdf", "ndf"}
KERATINOCYTE_TOKENS = {"keratinocyte", "keratinocytes", "epithelial"}
ENDOTHELIAL_TOKENS = {"endothelial", "endothelium", "lymphatic_endothelial"}
TISSUE_TOKENS = {
    "bulk_tissue",
    "whole_skin",
    "dermis",
    "epidermis",
    "skin",
    "all_barcodes",
    "all_cells",
    "tissue",
}
# Cell-line / commercial-line accessions excluded from disease endpoints (v2 ledger).
CELL_LINE_ACCESSIONS = {"GSE303591", "GSE232079"}
# Accessions excluded from primary tissue-skin endpoint (compartment / design mismatch).
PRIMARY_SKIN_EXCLUDE_ACCESSIONS = {
    "GSE282479",
    "GSE303591",
    "GSE232079",
    "GSE246562",
    "GSE121618",
    "GSE145725",
    "E-MTAB-2509",
    "GSE7890",
    "GSE44270",
    "GSE163973",
    "GSE188952",
    "GSE210434",
    "GSE245660",
    "GSE178562",
}
STIFFNESS_TASKS = {"stiffness_response"}
LEDGER_PATH = PROJECT_ROOT / "results/breakthrough_sprint_v2/cohort_decision_ledger.json"
ANNOTATION_SAMPLE_TOKENS = {
    "probe",
    "chromosome",
    "start",
    "end",
    "feature",
    "description",
    "distance",
    "type",
    "id",
    "strand",
    "orientation",
}


def recompute_module_scores(manifest: pd.DataFrame, expression_parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Recompute program scores from mapped expression rather than inherited metadata."""
    rows = []
    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        modality = row["processed_modality"]
        expr = expression_parts[modality]
        if sample_id not in expr.index:
            continue
        sample_expr = expr.loc[sample_id]
        out = {"sample_id": sample_id}
        score_vals = []
        for score_name, genes in MODULE_GENES.items():
            available = [gene for gene in genes if gene in sample_expr.index]
            value = float(pd.to_numeric(sample_expr[available], errors="coerce").mean()) if available else np.nan
            out[score_name] = value
            score_vals.append(value)
        out["fibrotic_activity_score"] = float(np.nanmean(score_vals)) if score_vals else np.nan
        rows.append(out)
    modules = pd.DataFrame(rows)
    for col in MODULE_COLUMNS:
        if col not in modules.columns:
            modules[col] = 0.0
    modules[MODULE_COLUMNS] = modules[MODULE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return modules


def within_sample_rank_program_scores(
    manifest: pd.DataFrame,
    expression_parts: dict[str, pd.DataFrame],
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """ssGSEA-style within-sample percentile ranks averaged over gene sets."""
    rows = []
    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        modality = row["processed_modality"]
        expr = expression_parts[modality]
        if sample_id not in expr.index:
            continue
        sample_expr = pd.to_numeric(expr.loc[sample_id], errors="coerce")
        ranks = sample_expr.rank(pct=True, method="average")
        out = {"sample_id": sample_id}
        for score_name, genes in gene_sets.items():
            available = [gene for gene in genes if gene in ranks.index]
            out[score_name] = float(ranks[available].mean()) if available else 0.5
        rows.append(out)
    return pd.DataFrame(rows)


def _mean_rank(ranks: pd.Series, genes: list[str]) -> float | None:
    available = [gene for gene in genes if gene in ranks.index]
    if not available:
        return None
    return float(ranks[available].mean())


def coverage_adjusted_rank_programs(
    manifest: pd.DataFrame,
    expression_parts: dict[str, pd.DataFrame],
    program_sets: dict[str, dict],
) -> pd.DataFrame:
    """Single-sample rank scores with positive-minus-negative contrasts."""
    rows = []
    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        modality = row["processed_modality"]
        expr = expression_parts[modality]
        if sample_id not in expr.index:
            continue
        ranks = pd.to_numeric(expr.loc[sample_id], errors="coerce").rank(pct=True, method="average")
        out = {"sample_id": sample_id}
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
    table = pd.DataFrame(rows)
    for col in program_sets:
        if col not in table.columns:
            table[col] = 0.0
    return table


def accession_coverage_audit(
    manifest: pd.DataFrame,
    modules: pd.DataFrame,
    expression_parts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # Avoid colliding with inherited module columns from modality wide tables.
    meta_cols = [
        "sample_id",
        "accession",
        "processed_modality",
        "keloid_binary",
        "is_out_of_domain",
        "patient_id",
    ]
    merged = manifest[meta_cols].merge(modules, on="sample_id", how="left")
    rows = []
    eligible = merged[merged["keloid_binary"].isin(["keloid", "non_keloid"]) & ~merged["is_out_of_domain"]]
    for accession, group in eligible.groupby("accession"):
        modality = str(group["processed_modality"].iloc[0])
        expr = expression_parts[modality]
        present_ids = [sid for sid in group["sample_id"] if sid in expr.index]
        block = expr.reindex(present_ids)
        module_block = group.set_index("sample_id")[MODULE_COLUMNS]
        rows.append(
            {
                "accession": accession,
                "modality": modality,
                "n_profiles": int(len(group)),
                "n_keloid": int((group["keloid_binary"] == "keloid").sum()),
                "n_non_keloid": int((group["keloid_binary"] == "non_keloid").sum()),
                "n_patients": int(group["patient_id"].nunique()),
                "n_unknown_patients": int(group["patient_id"].astype(str).str.lower().isin(["unknown", "nan", "none"]).sum()),
                "n_module_genes_present": int(sum(1 for g in ALL_PINNED_GENES if g in block.columns)),
                "module_variance_sum": float(module_block.var(axis=0, skipna=True).fillna(0.0).sum()),
                "all_zero_programs": bool(float(module_block.var(axis=0, skipna=True).fillna(0.0).sum()) <= 0),
            }
        )
    return pd.DataFrame(rows)


def _label_values(row: pd.Series) -> set[str]:
    return {
        str(row.get("disease_label", "")).lower(),
        str(row.get("keloid_vs_normal", "")).lower(),
        str(row.get("scar_type", "")).lower(),
        str(row.get("encoder_response", "")).lower(),
    }


def _is_treated(row: pd.Series) -> bool:
    """True for drug / stiffness / intervention arms that should leave disease endpoints."""
    if str(row.get("encoder_task", "")) in STIFFNESS_TASKS:
        return True
    treatment = str(row.get("treatment", "none")).strip().lower()
    if treatment in {"none", "unknown", "nan", ""}:
        return False
    if treatment.startswith("no_"):
        return False
    # kPa stiffness and named drug tokens are treated interventions for disease endpoints.
    if re.search(r"\d+\s*kpa", treatment):
        return True
    if any(tok in treatment for tok in TREATED_TOKENS):
        return True
    return True


def specimen_compartment(row: pd.Series) -> str:
    """Coarse specimen compartment for endpoint purity / routing."""
    acc = str(row.get("accession", ""))
    if acc in CELL_LINE_ACCESSIONS:
        return "cell_line"
    cell = str(row.get("cell_type", "")).strip().lower()
    task = str(row.get("encoder_task", ""))
    if task in STIFFNESS_TASKS or re.search(r"\d+\s*kpa", str(row.get("treatment", "")).lower()):
        return "fibroblast_perturbed"
    if any(tok in cell for tok in ENDOTHELIAL_TOKENS):
        return "endothelial"
    if any(tok in cell for tok in KERATINOCYTE_TOKENS):
        return "keratinocyte"
    if any(tok in cell for tok in FIBROBLAST_TOKENS):
        return "primary_fibroblast"
    if any(tok == cell or tok in cell for tok in TISSUE_TOKENS):
        return "bulk_tissue"
    modality = str(row.get("modality", "")).lower()
    if "scrna" in modality or "pseudobulk" in modality:
        if cell in {"", "unknown", "nan"} or "barcode" in cell:
            return "bulk_tissue"
        return "scrna_pseudobulk"
    if cell in {"", "unknown", "nan", "none"}:
        # Default unknown bulk microarray/RNA-seq rows to tissue for historical cohorts.
        return "bulk_tissue"
    return "other"


def is_tissue_for_primary_skin(row: pd.Series) -> bool:
    """Whole-skin / tissue rows allowed on keloid_vs_unaffected_skin (v2 purity)."""
    acc = str(row.get("accession", ""))
    if acc in CELL_LINE_ACCESSIONS:
        return False
    # Mixed cohort: keep only tissue rows.
    if acc == "E-MTAB-4945":
        return specimen_compartment(row) == "bulk_tissue"
    if acc in PRIMARY_SKIN_EXCLUDE_ACCESSIONS:
        return False
    return specimen_compartment(row) == "bulk_tissue"


def is_primary_fibroblast_unperturbed(row: pd.Series) -> bool:
    if str(row.get("accession", "")) in CELL_LINE_ACCESSIONS:
        return False
    if _is_treated(row):
        return False
    return specimen_compartment(row) == "primary_fibroblast"


def clean_binary(row: pd.Series) -> str:
    """High-confidence keloid vs normal/adjacent-normal/normal-scar only."""
    if bool(row.get("is_out_of_domain", False)):
        return "exclude"
    if str(row.get("encoder_task", "")) in SUSCEPTIBILITY_TASKS:
        return "exclude"
    if _is_treated(row):
        return "exclude"
    values = _label_values(row)
    # Exclude scar-differential and lesional contrasts from the clean endpoint.
    if values & {"hypertrophic_scar", "normotrophic_scar", "immature_scar", "lesional", "non_lesional"}:
        return "exclude"
    if values & CLEAN_POSITIVE:
        return "keloid"
    if values & CLEAN_NEGATIVE:
        return "non_keloid"
    return "exclude"


def keloid_vs_normal_scar(row: pd.Series) -> str:
    """Keloid vs normal/normotrophic scar only."""
    if bool(row.get("is_out_of_domain", False)):
        return "exclude"
    if str(row.get("encoder_task", "")) in SUSCEPTIBILITY_TASKS:
        return "exclude"
    if _is_treated(row):
        return "exclude"
    scar = str(row.get("scar_type", "")).lower()
    values = _label_values(row)
    if scar in NORMAL_SCAR_LABELS or values & NORMAL_SCAR_LABELS:
        return "non_keloid"
    if scar == "keloid" or (values & CLEAN_POSITIVE and str(row.get("encoder_task", "")) == "scar_differential"):
        # Only keep keloid samples from scar-differential designs for this endpoint.
        if str(row.get("encoder_task", "")) == "scar_differential" or scar == "keloid":
            if values & PATHOLOGIC_SCAR_LABELS or values & UNAFFECTED_SKIN_LABELS:
                return "exclude"
            return "keloid"
    return "exclude"


def keloid_vs_pathologic_scar(row: pd.Series) -> str:
    """Keloid vs hypertrophic/immature scar (differential diagnosis)."""
    if bool(row.get("is_out_of_domain", False)):
        return "exclude"
    if str(row.get("encoder_task", "")) in SUSCEPTIBILITY_TASKS:
        return "exclude"
    if _is_treated(row):
        return "exclude"
    scar = str(row.get("scar_type", "")).lower()
    values = _label_values(row)
    if scar in PATHOLOGIC_SCAR_LABELS or values & PATHOLOGIC_SCAR_LABELS:
        return "non_keloid"
    if scar == "keloid" or (values & CLEAN_POSITIVE and str(row.get("encoder_task", "")) == "scar_differential"):
        if str(row.get("encoder_task", "")) == "scar_differential" or scar == "keloid":
            return "keloid"
    return "exclude"


def keloid_vs_unaffected_skin(row: pd.Series) -> str:
    """Keloid vs unaffected / adjacent / control skin (not scar tissue).

    v2 endpoint purity: only unperturbed bulk/whole-skin tissue compartments.
    Fibroblast cultures, cell lines, endothelial, and stiffness arms are excluded.

    Lesional-status cohorts (e.g. GSE158395): keloid lesion biopsies count as keloid;
    patient non-lesional and healthy-control skin count as non_keloid. Scar tissues
    remain excluded.
    """
    if bool(row.get("is_out_of_domain", False)):
        return "exclude"
    if str(row.get("encoder_task", "")) in SUSCEPTIBILITY_TASKS | STIFFNESS_TASKS:
        return "exclude"
    if _is_treated(row):
        return "exclude"
    if not is_tissue_for_primary_skin(row):
        return "exclude"
    values = _label_values(row)
    if values & (NORMAL_SCAR_LABELS | PATHOLOGIC_SCAR_LABELS):
        return "exclude"
    if str(row.get("encoder_task", "")) == "scar_differential":
        return "exclude"
    disease = str(row.get("disease_label", "")).lower()
    kn = str(row.get("keloid_vs_normal", "")).lower()
    les = str(row.get("lesional_status", "")).lower()
    # Explicit disease labels win over encoder_response tokens like "lesional".
    if disease == "keloid" or kn == "keloid":
        if les == "non_lesional":
            return "non_keloid"
        return "keloid"
    if disease in UNAFFECTED_SKIN_LABELS or kn in {"normal", "non_keloid"}:
        return "non_keloid"
    if values & CLEAN_POSITIVE and "lesional" not in values:
        return "keloid"
    if values & UNAFFECTED_SKIN_LABELS:
        return "non_keloid"
    # encoder_response alone saying "lesional" without disease keloid → exclude
    return "exclude"


@dataclass(frozen=True)
class ModalityArtifact:
    name: str
    prefix: str
    directory: Path

    @property
    def wide_path(self) -> Path:
        return self.directory / f"{self.prefix}_expression_wide.parquet"

    @property
    def vocab_path(self) -> Path:
        return self.directory / f"{self.prefix}_gene_vocab.txt"


ARTIFACTS = [
    ModalityArtifact("microarray", "microarray", PROCESSED_DIR / "microarray"),
    ModalityArtifact("arrayexpress", "arrayexpress", PROCESSED_DIR / "arrayexpress"),
    ModalityArtifact("bulk_rnaseq", "bulk_rnaseq", PROCESSED_DIR / "bulk_rnaseq"),
    ModalityArtifact("scrna_spatial", "scrna_spatial", PROCESSED_DIR / "scrna_spatial"),
    ModalityArtifact("external_fibrosis", "external_fibrosis", PROCESSED_DIR / "external_fibrosis"),
    ModalityArtifact("public_keloid", "public_keloid", PROCESSED_DIR / "public_keloid"),
]


def read_vocab(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def is_gene_symbol(feature: str) -> bool:
    if feature.startswith("ENSG"):
        return False
    if re.match(r"^GPL\d+_", feature):
        return False
    return bool(re.match(r"^[A-Z][A-Z0-9.-]*$", feature))


def canonical_binary(row: pd.Series) -> str:
    if bool(row.get("is_out_of_domain", False)):
        return "exclude"
    # Historical broad endpoint keeps scar/lesional negatives, but not susceptibility-only tasks.
    if str(row.get("encoder_task", "")) in SUSCEPTIBILITY_TASKS:
        return "exclude"
    values = _label_values(row)
    if values & POSITIVE_LABELS:
        return "keloid"
    if values & NEGATIVE_LABELS:
        return "non_keloid"
    return "exclude"


def task_target(row: pd.Series) -> str:
    task = str(row.get("encoder_task", "unknown"))
    if task == "keloid_vs_normal":
        return row["keloid_binary"]
    if task == "lesional_status":
        value = str(row.get("lesional_status", "unknown"))
        return value if value and value != "unknown" else str(row.get("encoder_response", "unknown"))
    if task in {"cell_type", "fibroblast_state", "fibroblast_subcluster", "celltype_subcluster", "scar_differential"}:
        return str(row.get("encoder_response", "unknown"))
    if task.startswith("out_of_domain"):
        return str(row.get("encoder_response", "unknown"))
    return str(row.get("encoder_response", "unknown"))


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "nan", "none", ""}:
        return False
    return default


def make_group_id(row: pd.Series) -> str:
    patient = str(row.get("patient_id", "unknown")).strip()
    if patient and patient.lower() not in {"unknown", "nan", "none"}:
        return f"{row['modality']}|{row['accession']}|{patient}"
    if _as_bool(row.get("allow_accession_level_grouping", False)):
        return f"{row['modality']}|{row['accession']}|accession"
    return f"{row['modality']}|{row['accession']}|unknown_donor"


def is_fibroblast_compartment(row: pd.Series) -> bool:
    cell = str(row.get("cell_type", "")).strip().lower()
    if any(tok in cell for tok in KERATINOCYTE_TOKENS):
        return False
    if any(tok in cell for tok in FIBROBLAST_TOKENS):
        return True
    # Bulk tissue / all_cells / unknown keep for sensitivity only when not keratinocyte.
    return cell in {"bulk_tissue", "all_cells", "unknown", "", "nan"}


def native_aux_target(row: pd.Series) -> str:
    task = str(row.get("encoder_task", ""))
    if task in {
        "lesional_status",
        "scar_differential",
        "wound_susceptibility",
        "stiffness_response",
        "cell_type",
        "fibroblast_subcluster",
        "celltype_subcluster",
    }:
        return str(row.get("encoder_response", "unknown"))
    return "exclude"


def load_artifacts(max_shared_genes: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict, dict[str, pd.DataFrame]]:
    manifest_parts = []
    expression_parts = {}
    vocabs = {}

    for artifact in ARTIFACTS:
        if not artifact.wide_path.exists():
            continue
        wide = pd.read_parquet(artifact.wide_path)
        vocab = read_vocab(artifact.vocab_path)
        metadata_cols = [col for col in METADATA_COLUMNS + MODULE_COLUMNS if col in wide.columns]
        metadata = wide[metadata_cols].copy()
        for col in METADATA_COLUMNS:
            if col not in metadata.columns:
                if col in {"allow_accession_level_grouping", "lockbox"}:
                    metadata[col] = False
                elif col == "eval_grain":
                    metadata[col] = "profile"
                elif col == "cohort_role":
                    metadata[col] = "existing"
                else:
                    metadata[col] = "unknown"
        metadata["processed_modality"] = artifact.name
        metadata["artifact_prefix"] = artifact.prefix
        metadata["artifact_wide_path"] = str(artifact.wide_path.relative_to(PROJECT_ROOT))
        metadata["artifact_vocab_path"] = str(artifact.vocab_path.relative_to(PROJECT_ROOT))
        expr_cols = [col for col in vocab if col in wide.columns]
        expression_parts[artifact.name] = wide[["sample_id", *expr_cols]].set_index("sample_id")
        vocabs[artifact.name] = expr_cols
        manifest_parts.append(metadata)

    if not manifest_parts:
        raise RuntimeError("No modality artifacts found for corpus build.")

    manifest = pd.concat(manifest_parts, ignore_index=True, sort=False)
    manifest["profile_index"] = np.arange(len(manifest))
    manifest["allow_accession_level_grouping"] = manifest["allow_accession_level_grouping"].map(
        lambda v: _as_bool(v, False)
    )
    manifest["lockbox"] = manifest["lockbox"].map(lambda v: _as_bool(v, False))
    # Existing cohorts without explicit eval_grain: keep all profiles.
    missing_grain = manifest["eval_grain"].astype(str).isin(["unknown", "nan", ""])
    manifest.loc[missing_grain, "eval_grain"] = "profile"

    manifest["is_external_fibrosis"] = (
        manifest["processed_modality"].eq("external_fibrosis")
        | manifest["accession"].isin(EXTERNAL_FIBROSIS_ACCESSIONS)
        | manifest["disease_domain"].astype(str).str.lower().isin({"ipf", "pulmonary_fibrosis", "scleroderma"})
    )
    manifest["is_out_of_domain"] = (
        manifest["accession"].isin(OOD_ACCESSIONS)
        | manifest["encoder_task"].astype(str).str.startswith("out_of_domain")
        | manifest["disease_domain"].astype(str).str.lower().isin({"melanoma", "scleroderma"})
        | manifest["is_external_fibrosis"]
    )
    # Drop annotation-like sample IDs that slipped through count-matrix parsers.
    bad_sample = manifest["sample_id"].astype(str).str.strip().str.lower().isin(ANNOTATION_SAMPLE_TOKENS)
    bad_sample |= manifest["sample_id"].astype(str).str.contains(r"strand|orientation|chromosome", case=False, na=False)
    if bad_sample.any():
        manifest = manifest.loc[~bad_sample].copy()

    manifest["specimen_compartment"] = manifest.apply(specimen_compartment, axis=1)
    manifest["keloid_binary"] = manifest.apply(canonical_binary, axis=1)
    manifest["clean_keloid_binary"] = manifest.apply(clean_binary, axis=1)
    manifest["keloid_vs_normal_scar"] = manifest.apply(keloid_vs_normal_scar, axis=1)
    manifest["keloid_vs_pathologic_scar"] = manifest.apply(keloid_vs_pathologic_scar, axis=1)
    manifest["keloid_vs_unaffected_skin"] = manifest.apply(keloid_vs_unaffected_skin, axis=1)
    manifest["task_target"] = manifest.apply(task_target, axis=1)
    manifest["native_aux_target"] = manifest.apply(native_aux_target, axis=1)
    manifest["is_fibroblast_compartment"] = manifest.apply(is_fibroblast_compartment, axis=1)
    manifest["is_tissue_for_primary_skin"] = manifest.apply(is_tissue_for_primary_skin, axis=1)
    # Sensitivity endpoint: unperturbed primary fibroblasts only (not cell lines / tissue mix).
    manifest["fibroblast_keloid_binary"] = manifest["keloid_binary"]
    fib_ok = manifest.apply(is_primary_fibroblast_unperturbed, axis=1)
    manifest.loc[~fib_ok, "fibroblast_keloid_binary"] = "exclude"
    # Broad eval grain for scRNA: prefer donor_tissue; retain all historical profiles under
    # keloid_binary_all_profiles for immutable comparator, while keloid_binary uses donor grain when present.
    manifest["keloid_binary_all_profiles"] = manifest["keloid_binary"]
    has_donor_grain = manifest["eval_grain"].eq("donor_tissue")
    if has_donor_grain.any():
        # For accessions that emit donor_tissue profiles, exclude celltype pseudobulks from primary broad.
        donor_accessions = set(manifest.loc[has_donor_grain, "accession"])
        mask = manifest["accession"].isin(donor_accessions) & ~manifest["eval_grain"].eq("donor_tissue")
        manifest.loc[mask, "keloid_binary"] = "exclude"

    # Legacy cohorts historically used accession-level grouping when patient_id was unknown.
    # New public_keloid cohorts must set allow_accession_level_grouping explicitly in the registry.
    legacy = ~manifest["processed_modality"].eq("public_keloid")
    unknown_patient = manifest["patient_id"].astype(str).str.lower().isin(["unknown", "nan", "none", ""])
    manifest.loc[legacy & unknown_patient, "allow_accession_level_grouping"] = True

    manifest["split_group"] = manifest.apply(make_group_id, axis=1)
    manifest["source_group"] = manifest["split_group"]

    # Fail primary builds when public_keloid donor identity is unknowable without an explicit flag.
    unknown_donor = manifest["split_group"].astype(str).str.endswith("|unknown_donor")
    primary_mask = (
        manifest["keloid_binary"].isin(["keloid", "non_keloid"])
        & ~manifest["is_out_of_domain"]
        & ~manifest["lockbox"]
        & manifest["processed_modality"].eq("public_keloid")
    )
    bad = manifest[primary_mask & unknown_donor]
    if len(bad):
        bad_acc = sorted(bad["accession"].unique())
        raise RuntimeError(
            "Primary keloid_binary profiles have unknown donor IDs without "
            f"allow_accession_level_grouping for accessions: {', '.join(bad_acc)}"
        )

    # Public development disease cohorts used for keloid_binary must contain both classes.
    pub = manifest[
        manifest["processed_modality"].eq("public_keloid")
        & ~manifest["lockbox"].astype(bool)
        & manifest["encoder_task"].eq("keloid_vs_normal")
        & manifest["keloid_binary"].isin(["keloid", "non_keloid"])
    ]
    for accession, group in pub.groupby("accession"):
        classes = set(group["keloid_binary"])
        if classes and classes != {"keloid", "non_keloid"}:
            raise RuntimeError(
                f"Public cohort {accession} is single-class for keloid_binary after labeling: {classes}"
            )

    # Donor-balanced weights: within accession, weight = 1 / (n_profiles_in_donor).
    donor_counts = manifest.groupby(["accession", "split_group"])["sample_id"].transform("count").astype(float)
    manifest["donor_balanced_weight"] = np.where(donor_counts > 0, 1.0 / donor_counts, 1.0)
    # Accession-equalizing multiplier so each accession contributes equally in aggregate.
    acc_weight_sum = manifest.groupby("accession")["donor_balanced_weight"].transform("sum").replace(0, np.nan)
    manifest["accession_donor_balanced_weight"] = manifest["donor_balanced_weight"] / acc_weight_sum
    manifest["accession_donor_balanced_weight"] = manifest["accession_donor_balanced_weight"].fillna(1.0)

    symbol_presence: dict[str, set[str]] = {}
    for modality, genes in vocabs.items():
        for gene in genes:
            if is_gene_symbol(gene):
                symbol_presence.setdefault(gene, set()).add(modality)
    shared_genes = sorted([gene for gene, mods in symbol_presence.items() if len(mods) >= 2])
    if len(shared_genes) > max_shared_genes:
        variance_scores = []
        for gene in shared_genes:
            vals = []
            for expr in expression_parts.values():
                if gene in expr.columns:
                    vals.append(expr[gene])
            score = pd.concat(vals).var(skipna=True) if vals else 0.0
            variance_scores.append((gene, float(score)))
        shared_genes = [gene for gene, _ in sorted(variance_scores, key=lambda item: item[1], reverse=True)[:max_shared_genes]]
        shared_genes = sorted(shared_genes)

    pinned_genes = sorted(set(ALL_PINNED_GENES) & set(symbol_presence))
    shared_genes = sorted(set(shared_genes) | set(pinned_genes))

    feature_tables = build_feature_tables(manifest, expression_parts, shared_genes)
    published_genes = load_published_marker_genes(SIG_DIR)
    core_genes = load_low_i2_core_genes(CORE_GENES_PATH)
    vocab_report = {
        "shared_genes": shared_genes,
        "published_marker_genes": published_genes,
        "low_i2_core_genes": core_genes,
        "expanded_profibrotic_genes": EXPANDED_PROFIBROTIC_GENES,
        "robust_program_genes": ROBUST_PROGRAM_GENES,
        "rank_program_sets": RANK_PROGRAM_SETS,
        "scar_discriminative_program_sets": SCAR_DISCRIMINATIVE_PROGRAM_SETS,
        "all_rank_program_sets": ALL_RANK_PROGRAM_SETS,
        "compartment_feature_columns": COMPARTMENT_FEATURE_COLUMNS,
        "preregistered_feature_sets": PREREGISTERED_FEATURE_SETS,
        "modality_gene_counts": {name: len(genes) for name, genes in vocabs.items()},
        "n_shared_genes": len(shared_genes),
    }
    return manifest, feature_tables, vocab_report, expression_parts


def load_published_marker_genes(sig_dir: Path) -> list[str]:
    genes: set[str] = set()
    if not sig_dir.exists():
        return []
    for path in sorted(sig_dir.glob("*.tsv")):
        df = pd.read_csv(path, sep="\t")
        if "gene" not in df.columns:
            continue
        genes.update(str(g).strip().upper() for g in df["gene"].dropna().unique())
    return sorted(genes)


def load_low_i2_core_genes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]


def subset_gene_features(shared: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    available = [gene for gene in genes if gene in shared.columns]
    if not available:
        return shared[["sample_id"]].copy()
    table = shared[["sample_id", *available]].copy()
    table[available] = table[available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return table


def rank_normalize_modules(modules: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    merged = modules.merge(manifest[["sample_id", "accession"]], on="sample_id", how="left")
    cols = [col for col in MODULE_COLUMNS if col in merged.columns]
    ranked = merged[["sample_id"]].copy()
    for col in cols:
        ranked[col] = merged.groupby("accession", dropna=False)[col].rank(pct=True, method="average")
    ranked[cols] = ranked[cols].fillna(0.5)
    return ranked


def expanded_profibrotic_score(
    manifest: pd.DataFrame,
    expression_parts: dict[str, pd.DataFrame],
    genes: list[str],
) -> pd.DataFrame:
    rows = []
    meta = manifest.set_index("sample_id")
    for sample_id, row in meta.iterrows():
        modality = row["processed_modality"]
        expr = expression_parts[modality]
        if sample_id not in expr.index:
            continue
        sample_expr = expr.loc[sample_id]
        available = [gene for gene in genes if gene in sample_expr.index]
        if not available:
            score = 0.0
        else:
            accession_ids = meta.index[meta["accession"].eq(row["accession"])]
            acc_block = expr.reindex(accession_ids)[available]
            ranks = acc_block.rank(axis=0, pct=True, method="average")
            if sample_id in ranks.index:
                score = float(ranks.loc[sample_id].mean(skipna=True))
            else:
                score = 0.5
        rows.append({"sample_id": sample_id, "expanded_profibrotic_score": score})
    return pd.DataFrame(rows)


def build_feature_tables(
    manifest: pd.DataFrame,
    expression_parts: dict[str, pd.DataFrame],
    shared_genes: list[str],
) -> dict[str, pd.DataFrame]:
    modules = recompute_module_scores(manifest, expression_parts)
    # Keep recomputed program scores as the sole source of truth on the manifest.
    stale = [col for col in MODULE_COLUMNS if col in manifest.columns]
    if stale:
        manifest.drop(columns=stale, inplace=True)
    for col in MODULE_COLUMNS:
        manifest[col] = manifest["sample_id"].map(modules.set_index("sample_id")[col]).fillna(0.0)

    shared_rows = []
    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        modality = row["processed_modality"]
        expr = expression_parts[modality]
        values = expr.reindex(index=[sample_id], columns=shared_genes).fillna(0.0)
        values.insert(0, "sample_id", sample_id)
        shared_rows.append(values.reset_index(drop=True))
    shared = pd.concat(shared_rows, ignore_index=True, sort=False)
    shared[shared_genes] = shared[shared_genes].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    shared_plus_modules = shared.merge(modules, on="sample_id", how="left")
    modules_rank = rank_normalize_modules(modules, manifest)
    expanded = expanded_profibrotic_score(manifest, expression_parts, EXPANDED_PROFIBROTIC_GENES)
    robust = within_sample_rank_program_scores(manifest, expression_parts, ROBUST_PROGRAM_GENES)
    rank_programs = coverage_adjusted_rank_programs(manifest, expression_parts, RANK_PROGRAM_SETS)
    scar_programs = coverage_adjusted_rank_programs(
        manifest, expression_parts, {**RANK_PROGRAM_SETS, **SCAR_DISCRIMINATIVE_PROGRAM_SETS}
    )
    composition = coverage_adjusted_rank_programs(manifest, expression_parts, COMPOSITION_PROGRAM_SETS)
    fused = coverage_adjusted_rank_programs(manifest, expression_parts, ALL_RANK_PROGRAM_SETS)
    published_genes = load_published_marker_genes(SIG_DIR)
    core_genes = load_low_i2_core_genes(CORE_GENES_PATH)

    feature_tables = {
        "modules_only": modules,
        "modules_rank_only": modules_rank,
        "shared_genes": shared,
        "shared_genes_plus_modules": shared_plus_modules,
        "profibrotic_module_only": modules[["sample_id", "profibrotic_fibroblast_score"]].copy(),
        "expanded_profibrotic_only": expanded,
        "robust_programs_only": robust,
        "rank_programs_only": rank_programs[["sample_id", *RANK_PROGRAM_COLUMNS]].copy(),
    }
    rank_plus_comp = rank_programs[
        ["sample_id", *RANK_PROGRAM_COLUMNS]
    ].copy()
    # Explicit compartment covariates are already inside RANK_PROGRAM_COLUMNS; keep named alias.
    feature_tables["rank_programs_plus_compartment"] = rank_plus_comp
    # Donor-balanced feature table is identical features; weights applied at train time.
    feature_tables["donor_balanced_rank_programs"] = rank_programs[["sample_id", *RANK_PROGRAM_COLUMNS]].copy()
    scar_cols = [*RANK_PROGRAM_COLUMNS, *SCAR_DISCRIMINATIVE_COLUMNS]
    feature_tables["scar_discriminative_rank_programs"] = scar_programs[["sample_id", *scar_cols]].copy()
    # Breakthrough multi-view feature tables (versioned names from gene_modules).
    fibrosis_cols = [c for c in FIBROSIS_VIEW_COLUMNS if c in fused.columns]
    feature_tables["fibrosis_only"] = fused[["sample_id", *fibrosis_cols]].copy()
    feature_tables["scar_discriminative"] = scar_programs[["sample_id", *scar_cols]].copy()
    feature_tables["composition_only"] = composition[["sample_id", *COMPOSITION_COLUMNS]].copy()
    fused_cols = [*RANK_PROGRAM_COLUMNS, *SCAR_DISCRIMINATIVE_COLUMNS, *COMPOSITION_COLUMNS]
    feature_tables["fused_multiview"] = fused[["sample_id", *[c for c in fused_cols if c in fused.columns]]].copy()
    modules_rank_plus_expanded = modules_rank.merge(expanded, on="sample_id", how="left")
    feature_tables["modules_rank_plus_expanded"] = modules_rank_plus_expanded
    if published_genes:
        feature_tables["published_markers_only"] = subset_gene_features(shared, published_genes)
    if core_genes:
        feature_tables["low_i2_core_only"] = subset_gene_features(shared, core_genes)
    return feature_tables


def valid_task_rows(manifest: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "keloid_binary":
        rows = manifest[manifest["keloid_binary"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        # Lockbox never enters training selection folds as train data from other scripts;
        # they still get leave-one-accession-out test folds for one-shot evaluation.
        rows["target"] = rows["keloid_binary"]
        return rows
    if task == "keloid_binary_all_profiles":
        rows = manifest[manifest["keloid_binary_all_profiles"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["keloid_binary_all_profiles"]
        return rows
    if task == "clean_keloid_binary":
        rows = manifest[manifest["clean_keloid_binary"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["clean_keloid_binary"]
        return rows
    if task == "fibroblast_keloid_binary":
        rows = manifest[manifest["fibroblast_keloid_binary"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["fibroblast_keloid_binary"]
        return rows
    if task == "keloid_vs_normal_scar":
        rows = manifest[manifest["keloid_vs_normal_scar"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["keloid_vs_normal_scar"]
        return rows
    if task == "keloid_vs_pathologic_scar":
        rows = manifest[manifest["keloid_vs_pathologic_scar"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["keloid_vs_pathologic_scar"]
        return rows
    if task == "keloid_vs_unaffected_skin":
        rows = manifest[manifest["keloid_vs_unaffected_skin"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["keloid_vs_unaffected_skin"]
        return rows
    if task in {
        "lesional_status",
        "scar_differential",
        "wound_susceptibility",
        "stiffness_response",
    }:
        rows = manifest[manifest["encoder_task"].eq(task)].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["native_aux_target"]
        rows = rows[~rows["target"].isin(["unknown", "exclude", "nan", ""])]
        return rows
    rows = manifest[manifest["encoder_task"].eq(task)].copy()
    if task != "out_of_domain_disease_state":
        rows = rows[~rows["is_out_of_domain"]]
    rows["target"] = rows["task_target"]
    rows = rows[~rows["target"].isin(["unknown", "exclude", "nan", ""])]
    return rows


def add_split(splits: list[dict], name: str, task: str, rows: pd.DataFrame, train_idx, val_idx, test_idx) -> None:
    train = rows.iloc[list(train_idx)]
    val = rows.iloc[list(val_idx)]
    test = rows.iloc[list(test_idx)]
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        return
    splits.append(
        {
            "split_name": name,
            "task": task,
            "train_sample_ids": train["sample_id"].tolist(),
            "val_sample_ids": val["sample_id"].tolist(),
            "test_sample_ids": test["sample_id"].tolist(),
            "train_label_counts": train["target"].value_counts().to_dict(),
            "val_label_counts": val["target"].value_counts().to_dict(),
            "test_label_counts": test["target"].value_counts().to_dict(),
            "grouping": "split_group",
        }
    )


def build_splits(manifest: pd.DataFrame, tasks: list[str], n_repeats: int) -> list[dict]:
    splits: list[dict] = []
    for task in tasks:
        rows = valid_task_rows(manifest, task).reset_index(drop=True)
        if len(rows) < 6 or rows["target"].nunique() < 2:
            continue
        n_groups = rows["split_group"].nunique()
        n_splits = min(5, n_groups)
        if n_splits >= 3:
            for seed in range(n_repeats):
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                folds = list(cv.split(rows, rows["target"], rows["split_group"]))
                for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
                    train_val = rows.iloc[train_val_idx].reset_index(drop=True)
                    if train_val["target"].nunique() < 2 or train_val["split_group"].nunique() < 2:
                        continue
                    val_splits = min(3, train_val["split_group"].nunique())
                    inner = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=seed + 100)
                    inner_train_idx, val_idx = next(inner.split(train_val, train_val["target"], train_val["split_group"]))
                    train_idx = train_val_idx[inner_train_idx]
                    absolute_val_idx = train_val_idx[val_idx]
                    add_split(
                        splits,
                        f"grouped_seed{seed}_fold{fold_idx}",
                        task,
                        rows,
                        train_idx,
                        absolute_val_idx,
                        test_idx,
                    )

        if task in {
            "keloid_binary",
            "keloid_binary_all_profiles",
            "clean_keloid_binary",
            "fibroblast_keloid_binary",
            "keloid_vs_normal_scar",
            "keloid_vs_pathologic_scar",
            "keloid_vs_unaffected_skin",
            "lesional_status",
            "scar_differential",
            "wound_susceptibility",
            "stiffness_response",
        }:
            for accession in sorted(rows["accession"].unique()):
                test_idx = rows.index[rows["accession"].eq(accession)].to_numpy()
                not_test = ~rows["accession"].eq(accession)
                if "lockbox" in rows.columns:
                    not_lockbox_train = ~rows["lockbox"].astype(bool)
                    train_idx = rows.index[not_test & not_lockbox_train].to_numpy()
                else:
                    train_idx = rows.index[not_test].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                train_rows = rows.iloc[train_idx]
                val_idx = train_idx[:0]
                if train_rows["split_group"].nunique() >= 3 and train_rows["target"].nunique() >= 2:
                    inner = StratifiedGroupKFold(n_splits=min(3, train_rows["split_group"].nunique()), shuffle=True, random_state=17)
                    inner_train_idx, inner_val_idx = next(inner.split(train_rows, train_rows["target"], train_rows["split_group"]))
                    val_idx = train_idx[inner_val_idx]
                    train_idx = train_idx[inner_train_idx]
                add_split(splits, f"leave_accession_out_{accession}", task, rows, train_idx, val_idx, test_idx)

        if task in {"cell_type", "keloid_binary"}:
            sc_rows = rows[rows["accession"].isin(["GSE163973", "GSE181297"])].reset_index(drop=True)
            for group in sorted(sc_rows["split_group"].unique()):
                test_idx = sc_rows.index[sc_rows["split_group"].eq(group)].to_numpy()
                train_idx = sc_rows.index[~sc_rows["split_group"].eq(group)].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                add_split(splits, f"leave_source_out_{group.replace('|', '_')}", task, sc_rows, train_idx, [], test_idx)

        if task == "fibroblast_subcluster":
            fib_rows = rows[rows["accession"].eq("GSE163973")].reset_index(drop=True)
            for group in sorted(fib_rows["split_group"].unique()):
                test_idx = fib_rows.index[fib_rows["split_group"].eq(group)].to_numpy()
                train_idx = fib_rows.index[~fib_rows["split_group"].eq(group)].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                add_split(
                    splits,
                    f"leave_patient_out_{group.replace('|', '_')}",
                    task,
                    fib_rows,
                    train_idx,
                    [],
                    test_idx,
                )

        if task == "celltype_subcluster":
            sub_rows = rows[rows["accession"].eq("GSE163973")].reset_index(drop=True)
            for group in sorted(sub_rows["split_group"].unique()):
                test_idx = sub_rows.index[sub_rows["split_group"].eq(group)].to_numpy()
                train_idx = sub_rows.index[~sub_rows["split_group"].eq(group)].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                add_split(
                    splits,
                    f"leave_patient_out_{group.replace('|', '_')}",
                    task,
                    sub_rows,
                    train_idx,
                    [],
                    test_idx,
                )
    return splits


def write_outputs(
    manifest: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    vocab_report: dict,
    splits: list[dict],
    out_dir: Path,
    coverage: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = out_dir / "features"
    split_dir = out_dir / "splits"
    feature_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest.to_parquet(out_dir / "profile_manifest.parquet", index=False)
    manifest.to_csv(out_dir / "profile_manifest.csv", index=False)
    provenance_cols = [
        c
        for c in [
            "sample_id",
            "accession",
            "patient_id",
            "platform_id",
            "modality",
            "processed_modality",
            "cell_type",
            "specimen_compartment",
            "disease_label",
            "keloid_vs_normal",
            "scar_type",
            "lesional_status",
            "treatment",
            "encoder_task",
            "encoder_response",
            "contrast_type",
            "eval_grain",
            "cohort_role",
            "lockbox",
            "allow_accession_level_grouping",
            "split_group",
            "keloid_binary",
            "keloid_binary_all_profiles",
            "keloid_vs_normal_scar",
            "keloid_vs_pathologic_scar",
            "keloid_vs_unaffected_skin",
            "clean_keloid_binary",
            "fibroblast_keloid_binary",
            "is_tissue_for_primary_skin",
        ]
        if c in manifest.columns
    ]
    manifest[provenance_cols].to_csv(out_dir / "sample_provenance_audit.csv", index=False)
    # Endpoint eligibility matrix: mutually exclusive specimen/comparator strata.
    endpoint_cols = [
        c
        for c in [
            "keloid_vs_unaffected_skin",
            "keloid_vs_normal_scar",
            "keloid_vs_pathologic_scar",
            "clean_keloid_binary",
            "fibroblast_keloid_binary",
            "keloid_binary",
        ]
        if c in manifest.columns
    ]
    eligibility_rows = []
    for accession, group in manifest.groupby("accession"):
        row = {
            "accession": accession,
            "n_profiles": int(len(group)),
            "n_donors": int(group["split_group"].nunique()) if "split_group" in group else None,
            "lockbox": bool(group["lockbox"].astype(bool).any()) if "lockbox" in group else False,
            "cohort_role": str(group["cohort_role"].iloc[0]) if "cohort_role" in group else "unknown",
            "encoder_task": str(group["encoder_task"].mode().iloc[0]) if "encoder_task" in group else "unknown",
        }
        for ep in endpoint_cols:
            labels = group[ep]
            n_pos = int((labels == "keloid").sum())
            n_neg = int((labels == "non_keloid").sum())
            row[f"{ep}_keloid"] = n_pos
            row[f"{ep}_non_keloid"] = n_neg
            row[f"{ep}_evaluable"] = bool(n_pos > 0 and n_neg > 0)
        eligibility_rows.append(row)
    eligibility = pd.DataFrame(eligibility_rows).sort_values("accession")
    eligibility.to_csv(out_dir / "endpoint_eligibility_matrix.csv", index=False)
    for name, table in feature_tables.items():
        table.to_parquet(feature_dir / f"{name}.parquet", index=False)
        feature_cols = [col for col in table.columns if col != "sample_id"]
        (feature_dir / f"{name}_columns.txt").write_text("\n".join(feature_cols) + "\n")
    (feature_dir / "feature_report.json").write_text(json.dumps(vocab_report, indent=2))
    (split_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    if coverage is not None:
        coverage.to_csv(out_dir / "accession_module_coverage.csv", index=False)

    original_10 = []
    if ORIGINAL_10_PATH.exists():
        original_10 = json.loads(ORIGINAL_10_PATH.read_text()).get("original_10_comparator", [])
    summary = {
        "n_profiles": int(len(manifest)),
        "modalities": manifest["processed_modality"].value_counts().to_dict(),
        "accessions": manifest["accession"].value_counts().to_dict(),
        "tasks": manifest["encoder_task"].value_counts().to_dict(),
        "keloid_binary": manifest["keloid_binary"].value_counts().to_dict(),
        "keloid_binary_all_profiles": manifest["keloid_binary_all_profiles"].value_counts().to_dict(),
        "clean_keloid_binary": manifest["clean_keloid_binary"].value_counts().to_dict(),
        "fibroblast_keloid_binary": manifest["fibroblast_keloid_binary"].value_counts().to_dict(),
        "keloid_vs_normal_scar": manifest["keloid_vs_normal_scar"].value_counts().to_dict()
        if "keloid_vs_normal_scar" in manifest.columns
        else {},
        "keloid_vs_pathologic_scar": manifest["keloid_vs_pathologic_scar"].value_counts().to_dict()
        if "keloid_vs_pathologic_scar" in manifest.columns
        else {},
        "keloid_vs_unaffected_skin": manifest["keloid_vs_unaffected_skin"].value_counts().to_dict()
        if "keloid_vs_unaffected_skin" in manifest.columns
        else {},
        "n_lockbox_profiles": int(manifest["lockbox"].astype(bool).sum()) if "lockbox" in manifest.columns else 0,
        "n_external_fibrosis_excluded": int(manifest["is_external_fibrosis"].sum()) if "is_external_fibrosis" in manifest.columns else 0,
        "original_10_comparator": original_10,
        "preregistered_feature_sets": PREREGISTERED_FEATURE_SETS,
        "breakthrough_feature_views": BREAKTHROUGH_FEATURE_VIEWS,
        "feature_sets": {name: int(table.shape[1] - 1) for name, table in feature_tables.items()},
        "n_splits": len(splits),
        "split_tasks": pd.Series([split["task"] for split in splits]).value_counts().to_dict() if splits else {},
        "coverage_failures": coverage[coverage["all_zero_programs"]]["accession"].tolist() if coverage is not None else [],
        "endpoint_eligibility": {
            ep: {
                "n_evaluable_accessions": int(eligibility[f"{ep}_evaluable"].sum())
                if f"{ep}_evaluable" in eligibility.columns
                else 0
            }
            for ep in endpoint_cols
        },
    }
    (out_dir / "training_corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-shared-genes", type=int, default=5000)
    parser.add_argument("--split-repeats", type=int, default=3)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "keloid_binary",
            "keloid_binary_all_profiles",
            "clean_keloid_binary",
            "fibroblast_keloid_binary",
            "keloid_vs_normal_scar",
            "keloid_vs_pathologic_scar",
            "keloid_vs_unaffected_skin",
            "lesional_status",
            "cell_type",
            "fibroblast_subcluster",
            "celltype_subcluster",
            "scar_differential",
            "wound_susceptibility",
            "stiffness_response",
            "out_of_domain_disease_state",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, feature_tables, vocab_report, expression_parts = load_artifacts(args.max_shared_genes)
    modules = feature_tables["modules_only"]
    coverage = accession_coverage_audit(manifest, modules, expression_parts)
    failures = coverage[coverage["all_zero_programs"]]["accession"].tolist()
    if failures:
        raise RuntimeError(
            "Evaluable accessions with all-zero program features: "
            + ", ".join(failures)
            + ". Fix gene-symbol mapping before building the training corpus."
        )
    splits = build_splits(manifest, args.tasks, args.split_repeats)
    write_outputs(manifest, feature_tables, vocab_report, splits, args.out_dir, coverage=coverage)


if __name__ == "__main__":
    main()
