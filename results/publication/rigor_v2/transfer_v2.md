# SpheroScar Spheroid Transfer (Rigor v2)

Non-circular analysis: program score and evaluation targets are separated.

- Program model: `elastic_net_logreg_modules_only`

## C1: Within-Choi structural contrasts

- fb_endothelial_vs_fb_only: delta=-11.183, perm_p=0.07239, passes=False
- fb_only_2D_vs_3D: delta=-50.122, perm_p=0.1014, passes=False
- fb_ec_2D_vs_3D: delta=-22.164, perm_p=0.0002, passes=True

## C2: Independent phenotype correlations

- morphology_area: rho=0.262 [-0.671, 0.974], perm_p=0.5433, bh_p=0.7272945410917817, passes=False
- propagation_area: rho=-0.929 [-1.000, -0.531], perm_p=0.002799, bh_p=0.016796640671865627, passes=True
- compact_state_pct: rho=0.050 [-0.263, 0.341], perm_p=0.7371, bh_p=0.7370525894821037, passes=False
- pre_regression_pct: rho=0.079 [-0.202, 0.365], perm_p=0.6023, bh_p=0.7272945410917817, passes=False
- regression_pct: rho=-0.151 [-0.672, 0.429], perm_p=0.6061, bh_p=0.7272945410917817, passes=False
- drug_3d_relative_volume: rho=-0.333 [-0.842, 0.798], perm_p=0.4217, bh_p=0.7272945410917817, passes=False

## Dirand qualitative anchor

Dirand reported separately; not mixed with Choi program-score scale.

## Circularity audit

- Gene holdout rho: 0.8070588235294118
- Gene holdout perm_p: 0.0001999600079984003
- Interpretation: High cross-half correlation suggests residual self-reference; interpret cautiously.
