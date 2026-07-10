# SpheroScar Rigor Statistics

LOSO (leave-one-accession-out) is the primary reporting metric; grouped folds are secondary.

## keloid_binary LOSO-primary baselines

- primary_loso: elastic_net_logreg / modules_only / leave_accession_out -> weighted F1 0.65481072514
- primary_loso: small_mlp / profibrotic_module_only / leave_accession_out -> weighted F1 0.6514269607800001
- primary_loso: linear_svm / modules_only / leave_accession_out -> weighted F1 0.64867474559
- primary_loso: hist_gradient_boosting / modules_only / leave_accession_out -> weighted F1 0.63954467339
- primary_loso: elastic_net_logreg / profibrotic_module_only / leave_accession_out -> weighted F1 0.63639650604
- secondary_grouped: elastic_net_logreg / shared_genes / grouped -> weighted F1 0.7421497560333333
- secondary_grouped: elastic_net_logreg / shared_genes_plus_modules / grouped -> weighted F1 0.7405813894066667
- secondary_grouped: linear_svm / profibrotic_module_only / grouped -> weighted F1 0.7232408596533334
- secondary_grouped: elastic_net_logreg / profibrotic_module_only / grouped -> weighted F1 0.7188386211933333
- secondary_grouped: elastic_net_logreg / low_i2_core_only / grouped -> weighted F1 0.7033724114800001

## keloid_binary LOSO feature ablations

- LOSO: modules_only / elastic_net_logreg -> weighted F1 0.65481072514
- LOSO: profibrotic_module_only / small_mlp -> weighted F1 0.6514269607800001
- LOSO: published_markers_only / hist_gradient_boosting -> weighted F1 0.63024972441
- LOSO: low_i2_core_only / elastic_net_logreg -> weighted F1 0.58005826362
- LOSO: shared_genes / hist_gradient_boosting -> weighted F1 0.5693932659500001
- LOSO: shared_genes_plus_modules / elastic_net_logreg -> weighted F1 0.5649178263600001

## Bootstrap CI rows

- Baseline CI rows: 94
- Adapter CI rows: 4
