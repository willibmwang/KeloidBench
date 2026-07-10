# SpheroScar Rigor v2 Decision Memo

## Decision: `proceed_transfer_paper`

At least one independent spheroid phenotype is significantly associated with the cohort-projected program score after BH correction with CI excluding zero.

## Strict gate definition

Transfer thesis holds only if >=1 independent phenotype has BH-adjusted p < 0.05 and bootstrap CI for Spearman rho excludes 0.

## Non-circular results

- C1 structural contrasts pass: True
- C2 independent phenotype pass: True (1 BH-significant hits)
- C4 cohort interpretability pass: True

### C2 independent correlations

- morphology_area: rho=0.262, perm_p=0.5433, bh_p=0.7272945410917817, passes=False
- propagation_area: rho=-0.929, perm_p=0.002799, bh_p=0.016796640671865627, passes=True
- compact_state_pct: rho=0.050, perm_p=0.7371, bh_p=0.7370525894821037, passes=False
- pre_regression_pct: rho=0.079, perm_p=0.6023, bh_p=0.7272945410917817, passes=False
- regression_pct: rho=-0.151, perm_p=0.6061, bh_p=0.7272945410917817, passes=False
- drug_3d_relative_volume: rho=-0.333, perm_p=0.4217, bh_p=0.7272945410917817, passes=False

### C1 structural contrasts

- fb_endothelial_vs_fb_only: delta=-11.182541062405127, perm_p=0.07238552289542091, passes=False
- fb_only_2D_vs_3D: delta=-50.12157385616605, perm_p=0.10137972405518897, passes=False
- fb_ec_2D_vs_3D: delta=-22.164468737921613, perm_p=0.0001999600079984003, passes=True

### Circularity audit

- Gene holdout rho: 0.8070588235294118
- Gene holdout permutation p: 0.0001999600079984003
- High cross-half correlation suggests residual self-reference; interpret cautiously.

## Recommendations

- Proceed with cross-scale transfer manuscript framing.

## Note on v1 artifacts
Earlier `results/publication/feasibility_spike.*` used circular C1/C2 tests and is retained for provenance only; do not cite those headline numbers.
