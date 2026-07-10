# SpheroScar Spheroid Transfer Feasibility

- Overall gate: `proceed`
- Program model: `elastic_net_logreg_modules_only`

## C1: Spheroid system separation

- Choi fibroblast:endothelial mean program score: 10.5590
- Dirand fibroblast-only mean program score: -0.5000
- Mann-Whitney p: 4.042e-06
- Permutation p (active vs Dirand): 0.0216
- Passes gate: True

## C2: qPCR / morphology correlation

- internal_qpcr_activity: rho=0.990 [0.964, 0.997], p=3.298e-42, n=50
- morphology_area: rho=0.224 [-0.316, 0.667], p=0.3425, n=20
- compact_state_pct: rho=0.050 [-0.285, 0.346], p=0.728, n=50
- Passes gate: True

## C3: Drug response prediction

- Program R2: 0.004448237026344892
- Baseline R2: 0.0
- Delta R2: 0.004448237026344892
- Permutation p: 0.4642678660669665
- Passes gate: False
