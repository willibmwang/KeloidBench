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
- **Extended push-0.8 evaluation** (transductive normalization + calibrated ensemble, job 2130265 recovery ladder complete): targets LOSO F1 **0.80**; results to be inserted upon completion.

### RQ2 recovery (input-corrected + nested LOSO)

After repairing GPL6244/`gene_assignment` symbol parsing, GPL570 Affymetrix probe-set IDs, and Sun/Burns Ensembl→symbol mapping, previously all-zero program cohorts regained nonzero program variance. Coverage failures after rebuild: **none**.

**Primary endpoint** (broad profile-level `keloid_binary`, frozen 10-accession LOSO):
- Locked corrected_fixed best: **profibrotic_module_only / elastic_net_logreg (weight=none, adapt=none): F1=0.685**
- Cohort-balanced best: **profibrotic_module_only / linear_svm (weight=accession_class, adapt=none): F1=0.662**
- Nested inner-LOSO selected mean weighted F1 (outer folds): **0.612**

**Secondary endpoint** (separately named `clean_keloid_binary` / high-confidence keloid vs normal):
- Locked corrected_fixed best: **robust_programs_only / linear_svm (weight=none, adapt=none): F1=0.721**
- Cohort-balanced best: **robust_programs_only / ridge_logreg (weight=accession_class, adapt=none): F1=0.750**
- Nested selected mean weighted F1: **0.658**

Hard-cohort program coverage after repair:
- GSE188952: module_variance_sum=4.8977, n_module_genes_present=45, all_zero=False
- GSE44270: module_variance_sum=4.6501, n_module_genes_present=45, all_zero=False
- GSE7890: module_variance_sum=3.2512, n_module_genes_present=45, all_zero=False
- Sun_Burns: module_variance_sum=5.1154, n_module_genes_present=45, all_zero=False

Hard-cohort predictive recovery (outer-fold best within corrected_fixed):
- GSE44270 (best corrected_fixed config): F1=0.564 (modules_rank_only / linear_svm)
- Sun_Burns (best corrected_fixed config): F1=0.884 (modules_only / elastic_net_logreg)
- GSE188952 (best corrected_fixed config): F1=0.838 (published_markers_only / linear_svm)
- GSE7890 (best corrected_fixed config): F1=0.788 (robust_programs_only / linear_svm)

Broad-endpoint 0.80 was **not** achieved under the frozen outer LOSO protocol. The clean secondary metric is reported separately and does not replace the broad primary result. Limited adaptation (CORAL / train quantile) did not improve the locked broad mean above corrected program baselines; architecture chase was stopped once corrected programs remained below 0.70 with GSE44270 near chance.


### RQ2 public-cohort expansion (prospective ladder)

Public keloid cohorts were pre-registered in `data/raw/public_keloid_cohorts.json` under the accuracy evidence ladder. Development cohorts may enter nested selection; prospective lockbox expression (GSE212954) is downloaded only after protocol freeze and scored once. External fibrosis / IPF profiles are excluded from keloid-negative training. GSE125022 sample-level RNA-seq is unavailable from GEO RAW (ATAC-only) and remains skipped.

**Corpus audit**
- Profiles: **869**; external-fibrosis excluded count: **n/a**; lockbox profiles: **18**
- Ingested public accessions: **GSE113619, GSE121618, GSE173900, GSE190626, GSE191067, GSE237752, GSE245660**
- Coverage failures: **none**
- Results directory: `results/accuracy_ladder`

**Primary endpoint semantics (non-interchangeable)**
- Historical comparator (immutable): original-10 nested estimate frozen at baseline macro F1=**0.718**.
- Broad updated primary: donor-aware `keloid_binary`; all-profile sensitivity: `keloid_binary_all_profiles`.
- Clinical scar endpoint: `keloid_vs_normal_scar` (keloid vs normal/normotrophic scar only).
- Pathologic-scar differential: `keloid_vs_pathologic_scar` (hypertrophic/immature; not merged with normal scar).
- Unaffected-skin endpoint: `keloid_vs_unaffected_skin`; fibroblast sensitivity: `fibroblast_keloid_binary`.

**Estimates (do not conflate historical / post-development / prospective)**
- Post-hoc current-cohort fixed candidate (prior recovery ladder): F1=**0.685**
- Post-hoc ensemble candidate: F1=**0.691**
- Post-development nested current-10 macro-accession F1: **0.713** (worst fold **0.526**, GSE7890); paired Δ vs frozen baseline: **-0.005**
- Nested clean current-10 mean F1: **0.656**
- Nested fibroblast-compartment current-10 mean F1: **0.706**
- Clinical scar nested expanded mean F1: **0.495** (worst **0.257**)
- Unaffected-skin nested expanded mean F1: **0.734**
- Inner selection rule: equal-accession mean F1 with lower-tail tie-break (no outer-test fallbacks).

**Frozen lockbox (one-shot)**
- Frozen config: `rank_programs_only / elastic_net_logreg / thr=0.5299569779466491`
- lockbox evaluation pending or no lockbox cohorts ingested

**Claim status**
- Improvement gate (≥0.72 macro, donor≥0.70, worst≥0.60): **FAIL**
- Broad 0.80 claim under frozen outer LOSO + untouched lockbox: **NO**

Interpretation remains program-level reproducibility across heterogeneous keloid contrasts; endpoint heterogeneity and independent-cohort scarcity—not classifier depth—set the prediction ceiling. Residual hard fold for broad original-10 remains scar-differential biology (e.g. GSE188952).


### RQ2 breakthrough sprint (skin-first selective cascade)

The breakthrough sprint reframes prediction as a **molecular triage** product rather than a universal keloid-vs-anything classifier. Primary endpoint is `keloid_vs_unaffected_skin`; `keloid_vs_normal_scar` and `keloid_vs_pathologic_scar` are separate specialists. A selective cascade may abstain when confidence is low. Accuracy-ladder outputs remain immutable comparators; ladder lockbox GSE212954 is not reused as the breakthrough lockbox. After scar-endpoint starvation, development training was expanded with public GEO cohorts (GSE210434 scar triad; GSE303591 / GSE282479 / GSE232079 keloid-vs-normal fibroblasts; GSE246562 stiffness auxiliary) without touching the breakthrough lockbox one-shot score.

**Protocol**
- Frozen protocol: `results/breakthrough_sprint/frozen_protocol.json` (`breakthrough_sprint_v1`)
- Breakthrough lockbox (expression withheld until freeze): **GSE185309**
- Feature views: fibrosis-only, scar-discriminative, composition-only, fused multi-view
- Follow-up sampling protocol: `manuscript/matched_cohort_protocol.md`

**Primary nested estimates (do not conflate with broad original-10)**
- Full-coverage nested macro-accession F1: **0.611** (worst fold **0.000**, n=12)
- Selective cascade macro F1: **0.645** at mean coverage **0.809**
- Frozen primary config: `composition_only / ridge_logreg / thr=0.5910754312673769`

**Specialists**
- Normal-scar specialist nested folds: **4**
- Pathologic-scar specialist nested folds: **3**
- These are reported separately and are not merged into the skin product claim.

**Breakthrough lockbox (one-shot)**
- GSE185309: full F1=0.526, selective F1=0.580 (coverage=0.706)

**Claim status**
- Sprint gate (full≥0.85 or selective≥0.90@≥60% coverage, worst≥0.75): **FAIL**
- High-accuracy claim with untouched breakthrough lockbox: **NO**
- Broad original-10 0.80 claim remains governed by the accuracy-ladder protocol and is not implied by skin-endpoint gains.


### RQ2 breakthrough sprint v2 (endpoint-pure ceiling + matched-cohort decision)

Public GEO expansion alone did not clear the sprint gates. v1 nested primary macro F1 was **0.611** with worst fold **0.000** (complete inversion on fibroblast culture GSE282479). v2 froze an endpoint-pure protocol that removes culture lines, stiffness arms, and endothelial cohorts from `keloid_vs_unaffected_skin`, adds donor-level nested selection / inner-OOF abstention, and keeps the v1 lockbox (**GSE185309**) immutable.

**v2 nested evidence (tissue-only primary folds)**
- Best stage (donor selection + calibration): macro F1 **0.713**, worst fold **0.430**, n=5
- Zero-fold inversion removed; selective coverage gate still the only consistently passing sprint gate
- Cohorts removed from primary skin: E-MTAB-2509, GSE121618, GSE145725, GSE232079, GSE246562, GSE282479, GSE303591
- v1 lockbox one-shot (immutable): GSE185309: full F1=0.526, selective F1=0.580

**Go / no-go**
- Decision: **EXECUTE_MATCHED_COHORT_PROTOCOL**
- Claim gates (full≥0.85 or selective≥0.90@≥60%, worst≥0.75): **FAIL**
- Interpretation: compartment-matched public data lifts the floor (~0.61→~0.71) but does not support a high-accuracy claim; further architecture chase on heterogeneous GEO is discontinued.

**Prospective matched cohort (next required evidence)**
- Protocol: `manuscript/matched_cohort_protocol.md`
- Ops package: `manuscript/matched_cohort/` (checklist, enrollment CRF, lockbox assigner, power plan)
- Design: ≥20 donors × 4 arms (keloid / HTS / normotrophic scar / unaffected skin), one bulk RNA-seq assay, 50% donor lockbox reserved before scoring
- Same success criteria as the breakthrough sprint; public-GEO-frozen model is the primary scorer
- Friday product packaging: `PRODUCT.md` / `results/product_friday/` (Stage-B joblib cascade + CLI). v2 fibroblast lockbox GSE218007 one-shot macro F1≈0.33 (culture domain fail). Scar specialist refresh with GSE178562: pathologic≈0.42, normal-scar≈0.33.

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
