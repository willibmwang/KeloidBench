# Matched Multi-Arm Cohort Protocol (Follow-up)

Preregistered follow-up cohort for external confirmation of the breakthrough sprint cascade.
This protocol is written before new prospective sampling and is not used to tune models on public GEO data.

**Status (2026-07-20):** breakthrough v2 go/no-go = `EXECUTE_MATCHED_COHORT_PROTOCOL`  
(`results/breakthrough_sprint_v2/go_no_go_decision.json`). Public tissue-only nested macro F1≈0.71 with worst≈0.43 is below claim gates; architecture chase on heterogeneous GEO is stopped.

## Objective
Provide an independent, multi-donor, multi-arm transcriptomic cohort that can support:
1. Primary product: keloid vs unaffected skin
2. Specialist: keloid vs hypertrophic (pathologic) scar
3. Specialist: keloid vs normotrophic (normal) scar
4. Selective abstention operating characteristics

## Design
- Arms (same assay, same site annotation rules):
  - Keloid lesion
  - Hypertrophic scar
  - Normotrophic / mature scar
  - Unaffected skin (anatomically matched when feasible)
- Target size: ≥20 donors per arm (≥80 profiles), preferably paired within donors when clinically available.
- Grain: donor-level primary metrics; profile-level as sensitivity.
- Stratify / record: anatomical site, duration, ethnicity/self-identified ancestry, sex, age, recurrence history, prior treatments (exclude treated cultures from primary endpoints).

## Assay
- Bulk RNA-seq with a single library prep / sequencing protocol across all arms.
- Optional matched fibroblast cultures as sensitivity only (never mixed silently with tissue).
- Gene symbols mapped to HGNC; program coverage must meet frozen `gene_modules.py` minima.

## Splits and lockbox
- Randomly reserve 50% of donors as an untouched lockbox before any feature/model selection on the new cohort.
- Development half may be used only for calibration transfer checks; the public-GEO-frozen model remains the primary scorer.
- No lockbox expression may enter nested selection.
- Assignment tool (run once after enrollment closes):  
  `python scripts/assign_matched_cohort_lockbox.py --enrollment-csv <final_enrollment.csv>`
- Paired multi-arm donors share one `donor_id` across rows; split is donor-grain
  (all samples for a donor stay development or lockbox together).

## Endpoints (non-interchangeable)
- `keloid_vs_unaffected_skin`
- `keloid_vs_pathologic_scar`
- `keloid_vs_normal_scar`
- Cascade with abstention using the frozen confidence rule from  
  `results/breakthrough_sprint_v2/breakthrough_frozen_model.json` (fallback: v1 freeze)

## Success criteria (same as breakthrough sprint)
- Full-coverage nested/external macro-accession (or macro-donor) F1 ≥ 0.85, or
- Selective F1 ≥ 0.90 at coverage ≥ 0.60
- Worst evaluable arm/donor fold F1 ≥ 0.75
- High-accuracy claim requires the untouched half-lockbox to meet the same rule

## Exclusions
- Intervention-only, single-donor, selected-subpopulation, and susceptibility-only samples
- Drug/radiation/stretch cultures as primary disease labels (allowed only as mechanistic validation)

## Operations package
| Artifact | Path |
|----------|------|
| Sampling checklist | `manuscript/matched_cohort/sampling_checklist.md` |
| Enrollment CRF template | `manuscript/matched_cohort/enrollment_template.csv` |
| Power / n plan | `results/matched_cohort/power_sample_size.md` |
| Lockbox assigner | `scripts/assign_matched_cohort_lockbox.py` |
| Public-side optional follow-ups | `manuscript/matched_cohort/public_followups.md` |
