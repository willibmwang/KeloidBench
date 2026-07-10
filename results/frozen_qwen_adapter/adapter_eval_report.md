# Frozen-Qwen Adapter Evaluation

## Reporting priority

- Primary metric: leave-one-accession-out weighted F1.
- Secondary metric: grouped weighted F1.
- Bootstrap CIs: `results/publication/rigor_stats.md`

## Decision

- Decision: `hold_qwen_decode`
- Reason: Adapter did not beat the elastic-net baseline mean weighted F1 on grouped folds.
- Best config: `attentive_pool_L2_E15_P8_D01_LR00001_TD1024`
- Grouped weighted F1 mean: 0.6000123003961889
- LOSO weighted F1 mean: 0.31982198835264586
- Baseline target: 0.654

## Config Summary

| config_id | split_family | n_splits | test_weighted_f1_mean | test_weighted_f1_std | test_balanced_accuracy_mean | best_epoch_mean |
| --- | --- | --- | --- | --- | --- | --- |
| attentive_pool_L2_E15_P8_D005_LR00001_TD512 | grouped | 15 | 0.5413 | 0.1362 | 0.5872 | 3.6667 |
| attentive_pool_L2_E15_P8_D005_LR00001_TD512 | leave_accession_out | 8 | 0.4211 | 0.1445 | 0.5161 | 4.5000 |
| attentive_pool_L2_E15_P8_D01_LR00001_TD1024 | grouped | 15 | 0.6000 | 0.1552 | 0.6621 | 5.2000 |
| attentive_pool_L2_E15_P8_D01_LR00001_TD1024 | leave_accession_out | 8 | 0.3198 | 0.1484 | 0.4828 | 4.7500 |

## Best Config Per-Split Results

| split_name | split_family | n_train | n_test | best_epoch | test_weighted_f1 | test_balanced_accuracy | test_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_seed0_fold0 | grouped | 127 | 25 | 1 | 0.5578 | 0.5243 | 0.6597 |
| grouped_seed0_fold1 | grouped | 82 | 13 | 3 | 0.6657 | 0.7727 | 0.7727 |
| grouped_seed0_fold2 | grouped | 118 | 31 | 3 | 0.6380 | 0.5864 | 0.4182 |
| grouped_seed0_fold3 | grouped | 148 | 20 | 1 | 0.1815 | 0.5000 | 0.7143 |
| grouped_seed0_fold4 | grouped | 55 | 100 | 8 | 0.6701 | 0.6396 | 0.7217 |
| grouped_seed1_fold0 | grouped | 87 | 62 | 6 | 0.5072 | 0.6548 | 0.6286 |
| grouped_seed1_fold1 | grouped | 121 | 12 | 5 | 0.7552 | 0.8125 | 0.9688 |
| grouped_seed1_fold2 | grouped | 124 | 26 | 4 | 0.6406 | 0.6538 | 0.7988 |
| grouped_seed1_fold3 | grouped | 86 | 50 | 13 | 0.7441 | 0.7167 | 0.7700 |
| grouped_seed1_fold4 | grouped | 86 | 39 | 5 | 0.6846 | 0.6738 | 0.7620 |
| grouped_seed2_fold0 | grouped | 75 | 25 | 4 | 0.5018 | 0.6750 | 0.5900 |
| grouped_seed2_fold1 | grouped | 83 | 27 | 6 | 0.7450 | 0.7228 | 0.8913 |
| grouped_seed2_fold2 | grouped | 71 | 25 | 3 | 0.6481 | 0.7022 | 0.5882 |
| grouped_seed2_fold3 | grouped | 130 | 28 | 5 | 0.7158 | 0.7188 | 0.7344 |
| grouped_seed2_fold4 | grouped | 73 | 84 | 11 | 0.3449 | 0.5776 | 0.6651 |
| leave_accession_out_GSE145725 | leave_accession_out | 126 | 19 | 4 | 0.2591 | 0.2667 | 0.1667 |
| leave_accession_out_GSE158395 | leave_accession_out | 140 | 13 | 5 | 0.5852 | 0.5952 | 0.7143 |
| leave_accession_out_GSE163973 | leave_accession_out | 74 | 53 | 9 | 0.5307 | 0.5507 | 0.6336 |
| leave_accession_out_GSE181297 | leave_accession_out | 145 | 7 | 1 | 0.2571 | 0.5000 | 0.2500 |
| leave_accession_out_GSE188952 | leave_accession_out | 147 | 12 | 6 | 0.1667 | 0.5000 | 0.2500 |
| leave_accession_out_GSE44270 | leave_accession_out | 126 | 32 | 3 | 0.2663 | 0.5000 | 0.5000 |
| leave_accession_out_GSE7890 | leave_accession_out | 108 | 19 | 3 | 0.3383 | 0.4500 | 0.3167 |
| leave_accession_out_Sun_Burns | leave_accession_out | 146 | 25 | 7 | 0.1552 | 0.5000 | 0.5294 |
