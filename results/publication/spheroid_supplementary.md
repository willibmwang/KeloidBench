# Spheroid Qualitative Supplementary (Demoted)

Spheroid morphology and sparse qPCR (COL1A1, COL3A1, TGFB1) are reported as **qualitative concordance only**. No transcriptome-level cross-scale transfer claim is made.

## Non-circular v2 results (authoritative)

- morphology_area: rho=0.261904761904762, BH_p=0.7272945410917817, passes=False
- propagation_area: rho=-0.9285714285714287, BH_p=0.016796640671865627, passes=True
- compact_state_pct: rho=0.05042278237853517, BH_p=0.7370525894821037, passes=False
- pre_regression_pct: rho=0.07864732029550758, BH_p=0.7272945410917817, passes=False
- regression_pct: rho=-0.15118578920369086, BH_p=0.7272945410917817, passes=False
- drug_3d_relative_volume: rho=-0.3333333333333334, BH_p=0.7272945410917817, passes=False

## Circularity audit (limitation)

- Gene holdout rho: None
- Interpretation: High cross-half correlation; interpret cautiously.

## Removed claims

- Do not cite v1 feasibility spike rho=0.99 (circular).
- Do not claim cohort transcriptome programs transfer to spheroid RNA-seq (no public spheroid RNA-seq).