"""Canonical fibroblast program gene sets for SpheroScar.

All preprocessing, corpus construction, and publication reporting should import
from this module so program definitions stay synchronized.
"""

from __future__ import annotations

MODULE_GENES: dict[str, list[str]] = {
    "ECM_score": ["COL1A1", "COL3A1", "FN1"],
    "myofibroblast_score": ["ACTA2", "TAGLN", "MYL9"],
    "TGFb_score": ["TGFB1", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3"],
    "hypoxia_vascular_score": ["HIF1A", "PECAM1", "VWF", "KDR"],
    "remodeling_score": ["MMP14", "ADAM12", "HTRA1", "CTHRC1"],
    "profibrotic_fibroblast_score": ["POSTN", "CTHRC1", "COMP", "ASPN", "ADAM12", "TGFBI"],
    "antifibrotic_fibroblast_score": ["IGFBP2"],
}

# Backward-compatible alias used by older preprocessing scripts.
MODULES = MODULE_GENES

MODULE_COLUMNS = [
    *MODULE_GENES.keys(),
    "fibrotic_activity_score",
]

EXPANDED_PROFIBROTIC_GENES = [
    "POSTN",
    "CTHRC1",
    "COMP",
    "ASPN",
    "ADAM12",
    "TGFBI",
    "COL11A1",
    "SFRP2",
    "SFRP4",
]

# Fixed before outer LOSO evaluation (robust-program ladder).
ROBUST_PROGRAM_GENES: dict[str, list[str]] = {
    "ECM_score": MODULE_GENES["ECM_score"],
    "profibrotic_fibroblast_score": MODULE_GENES["profibrotic_fibroblast_score"],
    "remodeling_score": MODULE_GENES["remodeling_score"],
    "TGFb_score": MODULE_GENES["TGFb_score"],
    "yap_taz_mechano_score": ["CTGF", "CYR61", "ANKRD1", "AMOTL2", "TEAD1"],
    "wound_healing_score": ["IL6", "CXCL8", "CCL2", "ICAM1", "VCAM1", "SERPINE1"],
    "mesenchymal_score": ["VIM", "SNAI1", "SNAI2", "ZEB1", "TWIST1", "CDH2"],
}

# Literature-fixed single-sample rank programs with optional negative controls.
# Scores are coverage-adjusted positive-minus-negative contrasts.
RANK_PROGRAM_SETS: dict[str, dict] = {
    "ecm_profibrotic_rank": {
        "positive": ["POSTN", "CTHRC1", "COMP", "ASPN", "ADAM12", "TGFBI", "COL1A1", "COL3A1", "FN1", "COL11A1"],
        "negative": ["KRT1", "KRT10", "KRT14", "DMKN"],
        "source": "literature_fixed_v1",
        "min_genes_present": 4,
    },
    "remodeling_rank": {
        "positive": ["MMP14", "ADAM12", "HTRA1", "CTHRC1", "MMP2", "LOX", "LOXL2"],
        "negative": ["KRT1", "KRT10"],
        "source": "literature_fixed_v1",
        "min_genes_present": 3,
    },
    "yap_taz_mechano_rank": {
        "positive": ["CTGF", "CYR61", "ANKRD1", "AMOTL2", "TEAD1", "WWTR1", "YAP1"],
        "negative": ["KRT14", "DMKN"],
        "source": "literature_fixed_v1",
        "min_genes_present": 3,
    },
    "wound_inflammation_rank": {
        "positive": ["IL6", "CXCL8", "CCL2", "ICAM1", "VCAM1", "SERPINE1", "TNF", "IL1B"],
        "negative": ["KRT1", "KRT10"],
        "source": "literature_fixed_v1",
        "min_genes_present": 3,
    },
    "fibroblast_identity_rank": {
        "positive": ["PDGFRA", "LUM", "DCN", "COL1A2", "FBLN1", "THY1"],
        "negative": ["KRT1", "KRT14", "KRT5", "DMKN"],
        "source": "literature_fixed_v1",
        "min_genes_present": 3,
    },
    "keratinocyte_compartment_rank": {
        "positive": ["KRT1", "KRT10", "KRT14", "KRT5", "DMKN", "DSC1"],
        "negative": ["PDGFRA", "LUM", "COL1A1"],
        "source": "literature_fixed_v1",
        "min_genes_present": 3,
    },
}

# Scar-discriminative / clinical-contrast programs (prospective panel; not derived from lockbox).
SCAR_DISCRIMINATIVE_PROGRAM_SETS: dict[str, dict] = {
    "immune_activation_rank": {
        "positive": ["CD3D", "CD8A", "GZMB", "PRF1", "HLA-DRA", "CD74", "CCL5", "CXCL9", "CXCL10"],
        "negative": ["COL1A1", "POSTN"],
        "source": "msigdb_immune_literature_v1",
        "min_genes_present": 3,
    },
    "jak_stat_th2_rank": {
        "positive": ["IL4R", "IL13", "CCL11", "CCL17", "CCL18", "JAK3", "STAT6", "OX40L", "ICOS"],
        "negative": ["KRT1", "KRT10"],
        "source": "keloid_th2_literature_v1",
        "min_genes_present": 3,
    },
    "epidermal_keratinization_rank": {
        "positive": ["KRT1", "KRT10", "KRT14", "FLG", "LOR", "IVL", "DMKN", "DSC1"],
        "negative": ["POSTN", "COL1A1", "ACTA2"],
        "source": "msigdb_keratinization_v1",
        "min_genes_present": 3,
    },
    "lipid_metabolism_rank": {
        "positive": ["LGMN", "FABP4", "FABP5", "PPARG", "ACSL1", "SCD", "FASN", "DGAT2"],
        "negative": ["KRT14", "DMKN"],
        "source": "keloid_lipid_literature_v1",
        "min_genes_present": 3,
    },
    "sensory_neural_rank": {
        "positive": ["NTRK1", "NTRK2", "NGFR", "RET", "TAC1", "CALCA", "TRPV1", "PIEZO2"],
        "negative": ["KRT1", "PDGFRA"],
        "source": "keloid_sensory_literature_v1",
        "min_genes_present": 2,
    },
    "hypoxia_vascular_rank": {
        "positive": ["HIF1A", "VEGFA", "PECAM1", "VWF", "KDR", "ANGPT1", "EDN1"],
        "negative": ["KRT1", "KRT10"],
        "source": "literature_fixed_v1",
        "min_genes_present": 3,
    },
}

COMPARTMENT_FEATURE_COLUMNS = [
    "fibroblast_identity_rank",
    "keratinocyte_compartment_rank",
]

# Explicit composition proxies for breakthrough multi-view fusion (not disease labels).
COMPOSITION_PROGRAM_SETS: dict[str, dict] = {
    "fibroblast_composition_rank": {
        "positive": ["PDGFRA", "LUM", "DCN", "COL1A2", "FBLN1", "THY1", "DPT"],
        "negative": ["KRT1", "KRT14", "CD3D", "CD68"],
        "source": "composition_proxy_v1",
        "min_genes_present": 3,
    },
    "keratinocyte_composition_rank": {
        "positive": ["KRT1", "KRT10", "KRT14", "KRT5", "DMKN", "DSC1", "FLG"],
        "negative": ["PDGFRA", "LUM", "COL1A1", "CD3D"],
        "source": "composition_proxy_v1",
        "min_genes_present": 3,
    },
    "immune_composition_rank": {
        "positive": ["CD3D", "CD8A", "CD68", "CD74", "HLA-DRA", "LYZ", "CSF1R"],
        "negative": ["COL1A1", "KRT1", "POSTN"],
        "source": "composition_proxy_v1",
        "min_genes_present": 3,
    },
    "endothelial_composition_rank": {
        "positive": ["PECAM1", "VWF", "KDR", "CDH5", "ENG"],
        "negative": ["KRT1", "PDGFRA", "COL1A1"],
        "source": "composition_proxy_v1",
        "min_genes_present": 2,
    },
}

FIBROSIS_VIEW_COLUMNS = [
    "ecm_profibrotic_rank",
    "remodeling_rank",
    "yap_taz_mechano_rank",
    "wound_inflammation_rank",
]

RANK_PROGRAM_COLUMNS = list(RANK_PROGRAM_SETS.keys())
SCAR_DISCRIMINATIVE_COLUMNS = list(SCAR_DISCRIMINATIVE_PROGRAM_SETS.keys())
COMPOSITION_COLUMNS = list(COMPOSITION_PROGRAM_SETS.keys())
ALL_RANK_PROGRAM_SETS = {
    **RANK_PROGRAM_SETS,
    **SCAR_DISCRIMINATIVE_PROGRAM_SETS,
    **COMPOSITION_PROGRAM_SETS,
}

ALL_PINNED_GENES = sorted(
    {
        *{gene for genes in MODULE_GENES.values() for gene in genes},
        *EXPANDED_PROFIBROTIC_GENES,
        *{gene for genes in ROBUST_PROGRAM_GENES.values() for gene in genes},
        *{
            gene
            for prog in ALL_RANK_PROGRAM_SETS.values()
            for gene in (*prog.get("positive", []), *prog.get("negative", []))
        },
    }
)

# Pre-registered feature candidates for the public-cohort ladder (frozen before lockbox).
PREREGISTERED_FEATURE_SETS = [
    "profibrotic_module_only",
    "rank_programs_only",
    "rank_programs_plus_compartment",
    "donor_balanced_rank_programs",
    "scar_discriminative_rank_programs",
]

# Breakthrough sprint multi-view feature sets (frozen in breakthrough protocol).
BREAKTHROUGH_FEATURE_VIEWS = [
    "fibrosis_only",
    "scar_discriminative",
    "composition_only",
    "fused_multiview",
]
