# Keloid transcriptomic reproducibility lives at the fibroblast-program level, not the marker-gene level

## Abstract

Across independent keloid transcriptomic cohorts, individual marker genes and published hub-gene signatures show poor study-to-study reproducibility, but aggregated fibroblast activation programs—especially POSTN/profibrotic, ECM, remodeling, and mechanoresponsive mesenchymal fibroblast programs—remain stable across platforms, cohorts, and validation schemes. Leave-one-study-out (LOSO) prediction confirms that reproducibility emerges at the **program/state level** rather than at the single-gene marker level.

We do **not** claim discovery of POSTN-high fibroblasts; that literature is already crowded (Deng et al., Nat Commun 2021; Shim et al., JID 2022; Cheng et al., Commun Biol 2024; Zhang et al., Commun Biol 2025). Harmonizing **794** keloid-related expression profiles from **10** derivation studies, we ask: **(RQ1)** which reported keloid signals survive forced cross-cohort replication, and **(RQ2)** whether keloid status is predicted more reliably from fibroblast **program scores** than from published **marker gene sets** under LOSO evaluation.

**Replication (RQ1):** Individual keloid markers and curated published signatures show poor external reproducibility (**0%** of three signature gene sets meet strict low-heterogeneity core criteria; **68%** of curated markers fail per-study direction tests), but datasets repeatedly converge on a **coherent fibroblast activation program panel** (direction-correct in **6/7** modules; POSTN/profibrotic module largest delta, p≈8.7×10⁻¹³; ECM and remodeling modules also strongly enriched). A separate **44-gene** low-I² core exists but is not synonymous with this program family and does not replicate as cleanly. Held-out keloid cohorts (E-MTAB-2509/4945), cross-modality checks (microarray, bulk, scRNA/spatial pseudobulk, ArrayExpress), and exploratory IPF/SSc bulk tests show **module-level** (not gene-level) generalization.

**Prediction (RQ2):** Under LOSO, `modules_only` elastic net reaches weighted F1 **0.655**; the single POSTN/profibrotic module alone reaches **0.651**—both beating published marker unions (**0.630**), the strict 44-gene core (**0.580**), and high-dimensional gene features (**0.565–0.569**). Grouped cross-validation inflates gene-model F1 to **~0.74** while program features remain stable (grouped−LOSO gap **~0.03** for modules vs **~0.17** for genes). Extended evaluation with transductive per-accession normalization and calibrated soft-vote ensembling is in progress (HPC job 2096361). A frozen Qwen hybrid adapter remains a negative LOSO control (~**0.32–0.42**). Exploratory spheroid transfer (Choi/Dirand) supports cross-scale biological grounding but is supplementary.

## Introduction

Keloid transcriptomics has converged on overlapping fibroblast narratives—POSTN/periostin-positive profibrotic cells, ECM-rich mesenchymal states, and mechanoresponsive fibroblast activation—but each new cohort still proposes partially distinct hub-gene signatures. The field therefore faces a reproducibility problem, not a discovery problem: **which signals actually survive when studies are harmonized and tested against one another?**

We combine cross-study random-effects meta-analysis, held-out external validation, cross-modality module stability checks, single-cell localization, a reproducibility audit of published signatures, and LOSO supervised baselines with explicit feature ablations. The contribution is a **reproducibility map with an independent prediction test**, not another POSTN-high fibroblast paper: **reproducibility emerges at the aggregated fibroblast program/state level, not at the individual marker-gene level.**

## Methods

### Cohort harmonization
- **794** profiles across microarray, bulk RNA-seq, scRNA/spatial pseudobulk, and ArrayExpress; **10** derivation studies for meta-analysis and LOSO.
- **293** external fibrosis profiles (GSE32537, GSE48149, GSE58095) and held-out keloid cohorts (E-MTAB-2509/4945) excluded from signature derivation.
- Canonical target: `keloid_binary` (224 keloid / 294 non-keloid eligible profiles).
- Primary split: leave-one-accession-out (10 LOSO folds); grouped CV reported as secondary (optimistic when study boundaries leak).

### Replication analysis (RQ1)
- Per-study standardized mean difference → DerSimonian–Laird pooling, I², BH FDR.
- Core tier: FDR ≤ 0.05, I² ≤ 0.5, ≥ 3 studies → **44** core genes.
- Module scores (7 fibrosis programs including POSTN/profibrotic) computed per profile after per-accession gene z-scoring.
- Reproducibility audit: GSE44270_up, GSE145725_up, module_panel signature lists vs meta tiers and per-study direction tests.
- scRNA: core-gene and POSTN-module mapping onto GSE163973 fibroblast subclusters (pseudobulk).

### ML prediction (RQ2)
- **Feature ablations:** `profibrotic_module_only` (1-dim POSTN program), `modules_only` (7 programs), `published_markers_only` (22-gene curated union), `low_i2_core_only` (44-gene strict core), `shared_genes` (5006 genes), `shared_genes_plus_modules`.
- **Extended features (push-0.8):** `modules_rank_only`, `expanded_profibrotic_only` (9-gene POSTN neighbor program, ssGSEA-style), `modules_rank_plus_expanded`.
- Models: elastic net, linear SVM, random forest, hist gradient boosting, small MLP; primary metric = LOSO weighted F1.
- **Transductive domain adaptation (disclosed):** per-accession z-score and quantile-rank normalization using unlabeled held-out study expression (`scripts/domain_adaptation.py`).
- **Calibrated ensemble:** soft-vote over Platt-calibrated elastic net, SVM, hist GB, and MLP with validation-tuned decision threshold (`scripts/train_ensemble_loso.py`).
- Frozen Qwen2.5-0.5B hybrid adapter v3: accession embedding, contrastive pretrain, hybrid LM loss, first-token verbalizer CE; reported as negative control only.

### Exploratory spheroid transfer (supplementary)
- Choi 2024 and Dirand 2023 spheroid systems; cohort program scores projected onto in vitro readouts (C1–C4 gates in `results/publication/rigor_v2/`).

## Results

### RQ1 — Constructive: fibroblast activation programs replicate across cohorts and platforms

- **Program panel stability:** **6/7** fibrosis modules direction-correct across harmonized keloid cohorts; antifibrotic module is the exception.
- **POSTN/profibrotic module** shows the largest keloid–nonkeloid delta (p≈8.7×10⁻¹³), but it is one anchor in a broader activation family that also includes **ECM** (COL1A1/COL3A1/FN1; p≈2.3×10⁻⁷), **remodeling** (MMP14/ADAM12/HTRA1/CTHRC1), **TGFβ**, and **hypoxia/vascular** programs.
- **Cross-modality:** ECM and related activation programs show consistent keloid enrichment in microarray, scRNA/spatial pseudobulk, and ArrayExpress; bulk RNA-seq is noisier but still supports ECM/remodeling enrichment in several modules.
- **Single-gene caveat:** POSTN pooled effect is strong but heterogeneous at the gene level (high I²)—consistent with program-level, not marker-level, stability.
- **44** genes pass strict low-I² core criteria (distinct from the activation program family).
- GSE163973 scRNA: program scores localize to POSTN-high mechanoresponsive mesenchymal fibroblast subclusters rather than to isolated marker hits.

### RQ1 — Cautionary: published keloid markers and hub-gene signatures largely fail to replicate

- Prior published signatures: **0%** meet strict core criteria; **68%** of individual curated markers fail per-study direction tests.
- Signature gene sets (GSE44270, GSE145725, module panel): median I² ≈ **0.60**; none core-reproducible under strict criteria.
- Gene-level cores are cohort-specific; aggregated program scores transfer more reliably to held-out and cross-organ cohorts.

### RQ2 — ML prediction: programs generalize; marker sets and genes do not (LOSO)

Primary metric: leave-one-accession-out weighted F1 (`keloid_binary`, 10 folds; job 2091142).

| Feature set | Dims | Best model | LOSO F1 | Grouped F1 | Gap |
|-------------|------|------------|---------|------------|-----|
| **modules_only** | 8 | elastic net | **0.655** | 0.686 | 0.03 |
| **profibrotic_module_only** | 1 | small MLP | **0.651** | 0.723 | 0.07 |
| published_markers_only | 22 | hist GB | 0.630 | 0.703 | 0.07 |
| low_i2_core_only | 27 | elastic net | 0.580 | 0.703 | 0.12 |
| shared_genes | 5006 | hist GB | 0.569 | **0.742** | **0.17** |
| shared_genes_plus_modules | 5014 | elastic net | 0.565 | 0.741 | **0.18** |

- **Interpretation:** LOSO prediction recapitulates the replication hierarchy—aggregated programs ≥ published marker unions > strict gene core > high-dim genes under true study holdout. A single POSTN program dimension (F1 **0.651**) nearly matches the full 7-module panel (F1 **0.655**), showing that reproducibility concentrates in fibroblast state rather than in any one hub gene.
- **Hard cohorts:** GSE44270 and Sun_Burns remain ~**0.55** across all feature sets; E-MTAB-2509 / GSE145725 favor modules (~**0.85** / **0.84** per-accession best F1).
- **Extended push-0.8 evaluation** (transductive normalization + calibrated ensemble, job 2096361 pending): targets LOSO F1 **0.80**; results to be inserted upon completion.

### Held-out keloid validation

- E-MTAB-2509: gene-core AUROC **0.85** [0.63, 0.98]; module scores comparable.
- E-MTAB-4945: gene-core AUROC **0.68** [0.49, 0.87].

### Exploratory cross-organ module concordance

- GSE32537 (IPF): module-mean AUROC **0.91**; gene-core AUROC **0.44**.
- GSE48149 (IPF): POSTN-module AUROC **0.92**.
- GSE58095 (SSc): module-mean AUROC **0.86**.
- Pattern: module-level keloid programs score fibrotic vs normal in external bulk; gene-level cores do not transfer reliably. Hypothesis-generating only—not a universal fibrosis-core claim.

### Supplementary: spheroid cross-scale transfer

- Choi fibroblast:endothelial program scores > Dirand deactivated controls (Mann-Whitney p = 4.0×10⁻⁶).
- Independent propagation-area correlation passes BH gate (rho = −0.93, BH p = 0.017); drug-response delta not significant.
- Interpret as biological grounding of cohort programs in engineered 3D systems, not as primary keloid classifier evidence.

## Discussion

### Novelty relative to prior POSTN literature

POSTN-high profibrotic fibroblasts are not new, and the claim that they matter in keloid is already crowded. Our claim is narrower and, we argue, more useful for the field: when available keloid expression studies are harmonized under explicit replication criteria, **the stable cross-study unit is an aggregated fibroblast activation program family—POSTN/profibrotic, ECM, remodeling, and related mesenchymal/mechanoresponsive programs—not any single published marker gene or hub-gene signature.** That same hierarchy appears independently in LOSO prediction (RQ2), meta-analysis, held-out cohort validation, and cross-modality checks.

### Central message

Across independent keloid transcriptomic cohorts, individual marker genes and published hub-gene signatures show poor study-to-study reproducibility, but aggregated fibroblast activation programs remain stable across platforms, cohorts, and validation schemes. Leave-one-study-out prediction confirms that reproducibility emerges at the **program/state level** rather than at the single-gene marker level.

POSTN is therefore reframed from "newly discovered keloid marker" to **one anchor within a recurring cross-study fibroblast activation state**. ML prediction is not a clinical classifier claim (LOSO F1 ~0.65–0.67 is methods-grade) but an **operational test** that what replicates as a program also supports cross-study classification.

We do **not** claim this program family is *the* core of all pathological fibrosis. Transductive per-accession normalization is disclosed wherever used for cross-study prediction.

## Operational definitions (reviewer-facing)

- **Research questions:** (RQ1) Which keloid signals replicate across harmonized studies—at gene, published-marker, or aggregated fibroblast program/state level? (RQ2) Which input representation best predicts keloid status under leave-one-study-out evaluation?
- **Profile:** one harmonized expression vector (bulk, microarray, or scRNA/spatial pseudobulk); **794** total.
- **Cohort / study:** GEO or ArrayExpress accession; LOSO and meta-analysis stratify by accession.
- **Fibroblast activation program panel:** seven curated module scores (`ECM_score`, `myofibroblast_score`, `TGFb_score`, `hypoxia_vascular_score`, `remodeling_score`, `profibrotic_fibroblast_score`, `antifibrotic_fibroblast_score`); POSTN/profibrotic analyzed separately from single-gene POSTN and from the 44-gene low-I² core.
- **ML feature ablations:** see Methods; LOSO is primary, grouped CV secondary.
- **Transductive normalization:** held-out study expression used for unlabeled per-accession scaling only; disclosed in methods.

## Limitations

- scRNA pseudobulk from GSE163973 inflates profile count; not donor-level meta-analysis.
- Single atlas for fibroblast-state localization.
- Curated signature audit covers three lists; broader literature mining would strengthen cautionary claims.
- GSE44270 and Sun_Burns cap achievable LOSO means regardless of model architecture.
- Push-0.8 ensemble results pending HPC completion (job 2096361).
- Spheroid transfer is exploratory and partially circular (gene holdout rho ≈ 0.81); report cautiously.

## Data and code availability

- Repository: `SpheroScar/` — entry points: `scripts/run_publication_pipeline.py`, `scripts/run_push08_misha_gpu.sbatch`
- Results: `results/publication/`, `results/baselines_ablation/`, `results/baselines_push08/` (upon completion)
- Figures: `results/publication/figures/fig5_ml_ablation.png`, `fig6_push08_comparison.png`

## Recommended venues

Journal of Investigative Dermatology, Cell Systems, iScience, npj Systems Biology
