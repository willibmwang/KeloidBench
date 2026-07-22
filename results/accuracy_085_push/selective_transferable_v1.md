# Selective abstention — selective_transferable_v1

- Generated: `2026-07-21T16:10:33.707332+00:00`
- Excluded: `GSE173900` (non-transferable platform)
- Tune: inner accession-blocked OOF only; prefer lowest thr with OOF F1≥0.9 & cov≥0.6
- Mean selective macro F1: **0.6512**
- Worst selective macro F1: **0.0000**
- Mean coverage: **0.5430**
- Gates pass (sel≥0.9, cov≥0.6, worst≥0.75): **False**
- Sprint selective arm (sel≥0.90 & cov≥0.60 only): **False**

## Inner-OOF-tuned confidence thresholds

| Accession | sel macro F1 | coverage | kept/n | dual-class | full macro | conf thr |
|-----------|--------------|----------|--------|------------|------------|----------|
| GSE181297 | 0.0000 | 0.0000 | 0/6 | False | 1.0000 | 0.700 |
| GSE181316 | 0.5000 | 0.2000 | 1/5 | False | 1.0000 | 0.750 |
| GSE190626 | 0.5000 | 0.5000 | 3/6 | False | 1.0000 | 0.730 |
| E-MTAB-4945 | 0.5581 | 1.0000 | 30/30 | True | 0.5581 | 0.500 |
| GSE158395 | 1.0000 | 0.7692 | 10/13 | True | 0.8375 | 0.560 |
| GSE92566 | 1.0000 | 0.5714 | 4/7 | True | 1.0000 | 0.690 |
| Sun_Burns | 1.0000 | 0.7600 | 19/25 | True | 0.8264 | 0.560 |

## Pre-specified fixed confidence baselines (no OOF tuning)

- conf≥0.55: mean sel F1=0.9287, worst=0.6516, mean cov=0.8039, dual-class folds=7/7, dual-only mean F1=0.9286593931755223, gates_pass=False, sprint_arm=True
- conf≥0.6: mean sel F1=0.8571, worst=0.5000, mean cov=0.6543, dual-class folds=5/7, dual-only mean F1=1.0, gates_pass=False, sprint_arm=False
