"""Shared, testable utilities for the KeloidBench interactive demo.

The demo deliberately separates deterministic inference from generated prose.
All user-facing explanations are built from the evidence packet returned here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from expression_processing import METADATA_COLUMNS, normalize_gene_symbol  # noqa: E402
from gene_modules import ALL_RANK_PROGRAM_SETS, ALL_PINNED_GENES  # noqa: E402
from product_features import build_product_feature_views  # noqa: E402
from score_cascade_product import score_endpoint  # noqa: E402

PRIMARY_ENDPOINT = "keloid_vs_unaffected_skin"
TRANSFERABLE_ACCESSIONS = {
    "E-MTAB-4945",
    "GSE158395",
    "GSE181297",
    "GSE181316",
    "GSE190626",
    "GSE92566",
    "Sun_Burns",
}
NON_TRANSFERABLE_ACCESSIONS = {"GSE173900"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.55

# Curated, frozen public examples for the guided demo. Each one exercises a
# distinct product behavior in the exported Stage-B scorer.
CURATED_EXAMPLES: dict[str, dict[str, str]] = {
    "confident_keloid": {
        "sample_id": "GSE181316_GSM5494687_keloid_3R",
        "title": "Confident keloid-like profile",
        "short_title": "Keloid-like",
        "description": "A transferable tissue profile with a high live keloid score.",
        "presenter_note": "Start here to introduce the end-to-end workflow.",
        "icon": "●",
    },
    "confident_unaffected": {
        "sample_id": "GSE181316_GSM5494683_skin_7",
        "title": "Confident unaffected-like profile",
        "short_title": "Unaffected-like",
        "description": "A transferable skin profile with a high unaffected score.",
        "presenter_note": "Use this to contrast the biological program pattern.",
        "icon": "●",
    },
    "abstention": {
        "sample_id": "N20_Normal_Normal_N20",
        "title": "Confidence-based abstention",
        "short_title": "Abstention",
        "description": "A borderline profile that the interface deliberately withholds.",
        "presenter_note": "Use this to demonstrate calibrated human oversight.",
        "icon": "◆",
    },
    "transfer_failure": {
        "sample_id": "KL3",
        "title": "Known transfer failure",
        "short_title": "Transfer warning",
        "description": "A non-transferable platform example with an incorrect live prediction.",
        "presenter_note": "Use this to show that failures remain visible and actionable.",
        "icon": "▲",
    },
}

SOURCE_RECORDS: dict[str, dict[str, str]] = {
    "GSE181316": {
        "study_title": "Schwann cells contribute to keloid formation",
        "study_design": "10x single-cell RNA-seq of four keloid samples, one healthy-skin sample, and three normal-scar samples; KeloidBench uses all-barcode tissue pseudobulk.",
        "paper_url": "https://pubmed.ncbi.nlm.nih.gov/35278628/",
        "paper_citation": "Direder et al., Matrix Biology (2022), DOI 10.1016/j.matbio.2022.03.001",
        "geo_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181316",
    },
    "GSE158395": {
        "study_title": "RNA Sequencing Keloid Transcriptome Associates Keloids With Th2, Th1, Th17/Th22, and JAK3-Skewing",
        "study_design": "Bulk RNA-seq of lesional and non-lesional keloid skin and healthy control skin from African American participants.",
        "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7719808/",
        "paper_citation": "Wu et al., Frontiers in Immunology (2020), DOI 10.3389/fimmu.2020.597741",
        "geo_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158395",
    },
    "GSE173900": {
        "study_title": "WNT5A drives interleukin-6-dependent epithelial-mesenchymal transition via the JAK/STAT pathway in keloid pathogenesis",
        "study_design": "Bulk tissue RNA-seq of five keloid and four normal-skin profiles from an Asian tissue cohort.",
        "paper_url": "https://pubmed.ncbi.nlm.nih.gov/36225328/",
        "paper_citation": "Lee et al., Burns & Trauma (2022), PMID 36225328",
        "geo_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173900",
    },
    "GSE190626": {
        "study_title": "Single-Cell Sequencing Analysis and Weighted Co-Expression Network Analysis Identified TNC as a Keloid Biomarker",
        "study_design": "Bulk RNA-seq validation matrix comprising three keloid and three paired normal-skin profiles.",
        "paper_url": "https://doi.org/10.3389/fimmu.2021.783907",
        "paper_citation": "Xie et al., Frontiers in Immunology (2021), DOI 10.3389/fimmu.2021.783907",
        "geo_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190626",
    },
}

PROGRAM_LABELS = {
    "ecm_profibrotic_rank": "ECM / profibrotic",
    "remodeling_rank": "Matrix remodeling",
    "yap_taz_mechano_rank": "YAP/TAZ mechanotransduction",
    "wound_inflammation_rank": "Wound inflammation",
    "fibroblast_identity_rank": "Fibroblast identity",
    "keratinocyte_compartment_rank": "Keratinocyte compartment",
    "immune_activation_rank": "Immune activation",
    "jak_stat_th2_rank": "JAK/STAT and Th2",
    "epidermal_keratinization_rank": "Epidermal keratinization",
    "lipid_metabolism_rank": "Lipid metabolism",
    "sensory_neural_rank": "Sensory / neural",
    "hypoxia_vascular_rank": "Hypoxia / vascular",
    "fibroblast_composition_rank": "Fibroblast composition",
    "keratinocyte_composition_rank": "Keratinocyte composition",
    "immune_composition_rank": "Immune composition",
    "endothelial_composition_rank": "Endothelial composition",
}


def normalize_expression_table(frame: pd.DataFrame, *, transpose: bool = False) -> pd.DataFrame:
    """Return a numeric sample-by-HGNC expression matrix with duplicates collapsed."""
    expr = frame.copy()
    if transpose:
        expr = expr.T
    expr.index = expr.index.astype(str)
    metadata = set(METADATA_COLUMNS) | {
        "sample_id",
        "accession",
        "sample_title",
        "platform_id",
        "modality",
        "source_dataset",
    }
    expr = expr.drop(columns=[column for column in expr.columns if str(column) in metadata])
    expr.columns = [normalize_gene_symbol(str(c)) or str(c).strip() for c in expr.columns]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.T.groupby(level=0).mean().T.astype(float)
    return expr.dropna(axis=1, how="all").fillna(0.0)


def program_coverage(expression_columns: list[str] | pd.Index) -> pd.DataFrame:
    """Compute transparent gene coverage for every versioned rank program."""
    available = {str(g) for g in expression_columns}
    rows: list[dict[str, Any]] = []
    for name, spec in ALL_RANK_PROGRAM_SETS.items():
        positive = list(spec.get("positive", []))
        negative = list(spec.get("negative", []))
        expected = list(dict.fromkeys([*positive, *negative]))
        present = [g for g in expected if g in available]
        missing = [g for g in expected if g not in available]
        minimum = int(spec.get("min_genes_present", 1) or 1)
        rows.append(
            {
                "program": name,
                "display_name": PROGRAM_LABELS.get(name, name),
                "present_genes": len(present),
                "expected_genes": len(expected),
                "coverage": len(present) / len(expected) if expected else 0.0,
                "minimum_required": minimum,
                "usable": len(present) >= minimum,
                "missing": ", ".join(missing) if missing else "—",
            }
        )
    return pd.DataFrame(rows)


def expression_qc(expression: pd.DataFrame, accession: str = "uploaded") -> dict[str, Any]:
    coverage = program_coverage(expression.columns)
    pinned_present = len(set(expression.columns).intersection(ALL_PINNED_GENES))
    if accession in NON_TRANSFERABLE_ACCESSIONS:
        transfer_status = "non-transferable"
        transfer_note = "This platform failed the documented cross-study transfer check."
    elif accession in TRANSFERABLE_ACCESSIONS:
        transfer_status = "represented"
        transfer_note = "This accession is represented in the transferable public-study evaluation."
    else:
        transfer_status = "unverified"
        transfer_note = "Transferability is unverified for this uploaded or unseen cohort."
    return {
        "n_samples": int(len(expression)),
        "n_genes": int(expression.shape[1]),
        "pinned_genes_present": int(pinned_present),
        "pinned_genes_expected": int(len(ALL_PINNED_GENES)),
        "usable_programs": int(coverage["usable"].sum()),
        "total_programs": int(len(coverage)),
        "coverage_table": coverage,
        "transfer_status": transfer_status,
        "transfer_note": transfer_note,
    }


def load_product_bundle(
    endpoint: str = PRIMARY_ENDPOINT,
    product_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    product_dir = product_dir or PROJECT_ROOT / "results/product_friday"
    manifest = json.loads((product_dir / "product_manifest.json").read_text())
    info = manifest["endpoints"][endpoint]
    bundle = joblib.load(PROJECT_ROOT / info["model_path"])
    return manifest, bundle


def load_feature_views() -> dict[str, pd.DataFrame]:
    feature_dir = PROJECT_ROOT / "data/processed/training/features"
    return {
        name: pd.read_parquet(feature_dir / f"{name}.parquet").set_index("sample_id")
        for name in ("composition_only", "fibrosis_only", "rank_programs_only", "fused_multiview")
    }


def load_profile_manifest() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data/processed/training/profile_manifest.parquet")


def primary_example_ids(manifest: pd.DataFrame) -> list[str]:
    eligible = manifest[
        manifest[PRIMARY_ENDPOINT].isin(["keloid", "non_keloid"])
        & manifest["accession"].isin(TRANSFERABLE_ACCESSIONS | NON_TRANSFERABLE_ACCESSIONS)
    ].copy()
    return eligible.sort_values(["accession", "sample_id"])["sample_id"].astype(str).tolist()


def raw_expression_for_sample(sample_id: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a lightweight expression-shaped row for public-demo QC.

    Public examples already have versioned program features. The UI only needs
    their source expression vocabulary for coverage checks, so reading a full
    10–100 MB cohort matrix would add latency without changing the score.
    """
    matches = manifest[manifest["sample_id"].astype(str) == str(sample_id)]
    if matches.empty:
        raise KeyError(f"Unknown sample: {sample_id}")
    relative = Path(str(matches.iloc[0]["artifact_wide_path"]))
    source = PROJECT_ROOT / relative
    schema_columns = pq.read_schema(source).names
    metadata = set(METADATA_COLUMNS) | {
        "sample_id",
        "accession",
        "sample_title",
        "platform_id",
        "modality",
        "source_dataset",
    }
    gene_columns = [
        normalize_gene_symbol(name) or name
        for name in schema_columns
        if name not in metadata
    ]
    gene_columns = list(dict.fromkeys(gene_columns))
    return pd.DataFrame(np.zeros((1, len(gene_columns))), index=[sample_id], columns=gene_columns)


def _linear_contributions(bundle: dict[str, Any], aligned_features: pd.DataFrame) -> pd.DataFrame:
    """Return transparent linear contributions directed toward P(keloid)."""
    model = bundle["model"]
    meta = bundle["meta"]
    if not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return pd.DataFrame(columns=["feature", "display_name", "value", "contribution"])
    classifier = model.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["feature", "display_name", "value", "contribution"])
    transformed = aligned_features.to_numpy(dtype=float)
    if "scale" in model.named_steps:
        transformed = model.named_steps["scale"].transform(aligned_features)
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    # sklearn's binary linear score points toward classes_[1]. Keloid is code 0.
    classes = list(getattr(classifier, "classes_", [0, 1]))
    positive_code = int(meta["positive_code"])
    direction = 1.0 if len(classes) > 1 and classes[1] == positive_code else -1.0
    values = aligned_features.iloc[0].to_numpy(dtype=float)
    contributions = direction * transformed[0] * coefficients
    return pd.DataFrame(
        {
            "feature": aligned_features.columns,
            "display_name": [PROGRAM_LABELS.get(c, c) for c in aligned_features.columns],
            "value": values,
            "contribution": contributions,
        }
    ).sort_values("contribution", ascending=False)


def score_sample(
    expression_or_features: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    is_expression: bool,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Score one sample and return a single structured evidence record."""
    meta = bundle["meta"]
    if is_expression:
        views = build_product_feature_views(expression_or_features)
        feature_view = views[meta["feature_set"]]
    else:
        feature_view = expression_or_features
    aligned = pd.DataFrame(index=feature_view.index)
    for column in meta["feature_columns"]:
        aligned[column] = pd.to_numeric(
            feature_view[column] if column in feature_view else meta["feature_medians"].get(column, 0.0),
            errors="coerce",
        )
    aligned = aligned.fillna(pd.Series(meta["feature_medians"])).fillna(0.0)
    scored = score_endpoint(bundle, aligned).iloc[0].to_dict()
    confidence = float(scored["confidence"])
    scored["decision"] = scored["pred_label"] if confidence >= confidence_threshold else "abstain"
    scored["confidence_threshold"] = float(confidence_threshold)
    scored["feature_set"] = meta["feature_set"]
    scored["model_name"] = meta["config"]["model"]
    scored["artifact_stage"] = "stage_b_donor_selection_calibration"
    scored["contributions"] = _linear_contributions(bundle, aligned).to_dict(orient="records")
    scored["features"] = {
        key: float(value) for key, value in aligned.iloc[0].to_dict().items()
    }
    return scored


def program_score_table(feature_row: pd.DataFrame) -> pd.DataFrame:
    row = feature_row.iloc[0]
    records = [
        {
            "program": column,
            "display_name": PROGRAM_LABELS.get(column, column),
            "score": float(value),
            "direction": "positive" if value > 0 else "negative" if value < 0 else "neutral",
        }
        for column, value in row.items()
        if column in PROGRAM_LABELS
    ]
    return pd.DataFrame(records).sort_values("score", ascending=False)


def build_evidence_packet(
    score: dict[str, Any],
    qc: dict[str, Any],
    programs: pd.DataFrame,
    *,
    accession: str,
    sample_label: str,
) -> dict[str, Any]:
    top_support = sorted(score.get("contributions", []), key=lambda x: x["contribution"], reverse=True)[:3]
    top_against = sorted(score.get("contributions", []), key=lambda x: x["contribution"])[:3]
    return {
        "system": "KeloidBench",
        "intended_use": "research-use cross-cohort transcriptomic triage",
        "not_for": "clinical diagnosis or treatment selection",
        "sample": {"sample_id": sample_label, "accession": accession},
        "quality_control": {
            key: value
            for key, value in qc.items()
            if key != "coverage_table"
        },
        "prediction": {
            key: score[key]
            for key in (
                "prob_keloid",
                "pred_label",
                "decision",
                "confidence",
                "decision_threshold",
                "confidence_threshold",
                "feature_set",
                "model_name",
                "artifact_stage",
            )
        },
        "top_program_scores": programs.head(6).to_dict(orient="records"),
        "top_supporting_contributions": top_support,
        "top_opposing_contributions": top_against,
        "evidence_scope": {
            "transferable_loso_mean_macro_f1": 0.918,
            "transferable_profiles": 92,
            "transferable_studies": 7,
            "pooled_held_out_accuracy": 77 / 92,
            "warning": "The live Stage-B artifact and the later nested LOSO ensemble are distinct evidence objects.",
        },
    }


def deterministic_summary(packet: dict[str, Any]) -> str:
    """Faithful fallback narrative when no language-model endpoint is configured."""
    pred = packet["prediction"]
    qc = packet["quality_control"]
    if pred["decision"] == "abstain":
        opening = (
            f"KeloidBench abstained because model confidence ({pred['confidence']:.3f}) "
            f"did not reach the operational threshold ({pred['confidence_threshold']:.2f})."
        )
    else:
        opening = (
            f"KeloidBench returned a {pred['decision'].replace('_', ' ')} research-triage result "
            f"with P(keloid)={pred['prob_keloid']:.3f} and model confidence={pred['confidence']:.3f}."
        )
    support = packet["top_supporting_contributions"]
    if support:
        names = ", ".join(item["display_name"] for item in support[:3])
        evidence = f"The strongest model contributions toward keloid were {names}."
    else:
        evidence = "No linear contribution breakdown was available for this artifact."
    transfer = (
        f"Transferability status is {qc['transfer_status']}: {qc['transfer_note']}"
    )
    return " ".join(
        [
            opening,
            evidence,
            transfer,
            "This is a research prototype and must not be interpreted as a clinical diagnosis.",
        ]
    )


def _copilot_prompt(packet: dict[str, Any]) -> str:
    return (
        "You are the KeloidBench research copilot. Write a concise, technically accurate "
        "interpretation for a transcriptomics researcher. Use only the JSON evidence below. "
        "Do not add mechanisms, citations, diagnoses, treatment advice, or numbers absent from "
        "the evidence. Explicitly state when the decision is abstain and preserve the research-use "
        "limitation. Use at most 130 words.\n\nEVIDENCE_JSON:\n"
        + json.dumps(packet, indent=2)
    )


def generate_grounded_summary(packet: dict[str, Any]) -> tuple[str, str]:
    """Call an optional OpenAI-compatible endpoint; otherwise return faithful fallback prose.

    Configure with KELOIDBENCH_LLM_API_URL, KELOIDBENCH_LLM_MODEL, and optionally
    KELOIDBENCH_LLM_API_KEY. The endpoint is never allowed to alter the score.
    """
    url = os.environ.get("KELOIDBENCH_LLM_API_URL", "").strip()
    model = os.environ.get("KELOIDBENCH_LLM_MODEL", "").strip()
    if not url or not model:
        return deterministic_summary(packet), "deterministic_fallback"
    # Never transmit raw expression or sample identifiers to a remote narrative
    # endpoint. The model receives only derived evidence and a redacted sample key.
    external_packet = json.loads(json.dumps(packet))
    if "sample" in external_packet:
        external_packet["sample"]["sample_id"] = "redacted"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("KELOIDBENCH_LLM_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": _copilot_prompt(external_packet)}],
            "temperature": 0,
            "max_tokens": 220,
        },
        timeout=45,
    )
    response.raise_for_status()
    body = response.json()
    return str(body["choices"][0]["message"]["content"]).strip(), "grounded_llm"


def load_loso_evidence() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    root = PROJECT_ROOT / "results/accuracy_085_push"
    run = root / "expanded_ensemble_v1"
    predictions = pd.read_csv(run / f"predictions_{PRIMARY_ENDPOINT}.csv")
    selected = pd.read_csv(run / "nested_selected.csv")
    transferable = json.loads((root / "transferable_excl_GSE173900.json").read_text())
    selective = json.loads((root / "selective_transferable_v1/report.json").read_text())
    predictions["true_label"] = np.where(predictions["y_true"].astype(int) == 0, "keloid", "non_keloid")
    predictions["pred_label"] = np.where(
        predictions["prob_keloid"] >= predictions["threshold"], "keloid", "non_keloid"
    )
    predictions["confidence"] = np.maximum(
        predictions["prob_keloid"], 1.0 - predictions["prob_keloid"]
    )
    predictions["selective_decision"] = np.where(
        predictions["confidence"] >= DEFAULT_CONFIDENCE_THRESHOLD,
        predictions["pred_label"],
        "abstain",
    )
    predictions["correct"] = predictions["true_label"] == predictions["pred_label"]
    return predictions, selected, transferable, selective
