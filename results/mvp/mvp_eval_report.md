# SpheroScar MVP Baseline Evaluation

## Corpus

- Total profiles: 501
- Modalities: {'scrna_spatial': 230, 'microarray': 154, 'arrayexpress': 67, 'bulk_rnaseq': 50}
- Accessions: {'GSE163973': 220, 'GSE3189': 70, 'E-MTAB-4945': 40, 'GSE44270': 32, 'E-MTAB-2509': 27, 'Sun_Burns': 25, 'GSE7890': 19, 'GSE145725': 19, 'GSE158395': 13, 'GSE188952': 12, 'GSE90051': 7, 'GSE92566': 7, 'GSE181297': 7, 'GSE175866': 2, 'GSE160536': 1}
- Canonical keloid labels: {'keloid': 224, 'non_keloid': 199, 'exclude': 78}

## Best Baselines

### keloid_binary

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| shared_genes_plus_modules | elastic_net_logreg | 0.6527 | 0.6326 | 0.6303 | 0.5866 | 25 |
| modules_only | elastic_net_logreg | 0.6245 | 0.6223 | 0.6396 | 0.5852 | 25 |
| modules_only | linear_svm | 0.6199 | 0.6185 | 0.6305 | 0.5778 | 25 |
| modules_only | random_forest | 0.6231 | 0.6149 | 0.6232 | 0.5670 | 25 |
| shared_genes | elastic_net_logreg | 0.6488 | 0.6119 | 0.6274 | 0.5687 | 25 |

### lesional_status

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| modules_only | random_forest | 0.7917 | 0.7459 | 0.7831 | 0.7316 | 18 |
| shared_genes_plus_modules | random_forest | 0.7932 | 0.7387 | 0.7646 | 0.7108 | 18 |
| modules_only | small_mlp | 0.7416 | 0.7194 | 0.7121 | 0.6297 | 11 |
| shared_genes_plus_modules | elastic_net_logreg | 0.6180 | 0.5974 | 0.6103 | 0.5119 | 18 |
| modules_only | knn | 0.6853 | 0.5880 | 0.6103 | 0.5317 | 18 |

### cell_type

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| shared_genes | elastic_net_logreg | 0.9800 | 0.9753 | 0.9780 | 0.9681 | 21 |
| shared_genes_plus_modules | elastic_net_logreg | 0.9800 | 0.9753 | 0.9780 | 0.9681 | 21 |
| shared_genes | linear_svm | 0.9768 | 0.9717 | 0.9751 | 0.9655 | 21 |
| shared_genes_plus_modules | linear_svm | 0.9768 | 0.9717 | 0.9751 | 0.9655 | 21 |
| shared_genes | random_forest | 0.9634 | 0.9548 | 0.9621 | 0.9483 | 21 |

### fibroblast_subcluster

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| shared_genes | linear_svm | 0.8918 | 0.8640 | 0.8874 | 0.8595 | 21 |
| shared_genes_plus_modules | linear_svm | 0.8918 | 0.8640 | 0.8874 | 0.8595 | 21 |
| shared_genes | random_forest | 0.8781 | 0.8427 | 0.8750 | 0.8401 | 21 |
| shared_genes_plus_modules | random_forest | 0.8635 | 0.8265 | 0.8615 | 0.8225 | 21 |
| shared_genes | elastic_net_logreg | 0.7101 | 0.6588 | 0.7055 | 0.6491 | 21 |

### celltype_subcluster

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| shared_genes | linear_svm | 0.9719 | 0.9655 | 0.9702 | 0.9587 | 21 |
| shared_genes_plus_modules | linear_svm | 0.9719 | 0.9655 | 0.9702 | 0.9587 | 21 |
| shared_genes | random_forest | 0.9567 | 0.9457 | 0.9541 | 0.9360 | 21 |
| shared_genes_plus_modules | random_forest | 0.9541 | 0.9424 | 0.9517 | 0.9349 | 21 |
| shared_genes | elastic_net_logreg | 0.9301 | 0.9254 | 0.9281 | 0.9176 | 21 |

## Frozen-Qwen Adapter Decision

- Decision: `ready_for_frozen_qwen_adapter`
- Reason: Best non-dummy baseline beats dummy baselines with a useful weighted-F1 margin and balanced accuracy above chance.

The frozen-Qwen adapter should only be trained after simple baselines show stable grouped-split signal.

## Rigor Statistics

- LOSO (leave-one-accession-out) is the primary reporting metric.
- Grouped folds are reported as secondary.
- See `results/publication/rigor_stats.md` for bootstrap confidence intervals.

## Detailed Artifacts

- `detailed_eval_breakdowns.json` contains split-level metrics, confusion matrices, and modality/accession composition for every valid baseline run.
- `detailed_eval_breakdowns.csv` is the same information flattened for quick review.
- Detailed baseline rows: 2424
