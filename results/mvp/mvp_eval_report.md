# SpheroScar MVP Baseline Evaluation

## Corpus

- Total profiles: 267
- Modalities: {'microarray': 154, 'scrna_spatial': 63, 'bulk_rnaseq': 50}
- Accessions: {'GSE3189': 70, 'GSE163973': 53, 'GSE44270': 32, 'Sun_Burns': 25, 'GSE7890': 19, 'GSE145725': 19, 'GSE158395': 13, 'GSE188952': 12, 'GSE90051': 7, 'GSE92566': 7, 'GSE181297': 7, 'GSE175866': 2, 'GSE160536': 1}
- Canonical keloid labels: {'keloid': 106, 'non_keloid': 83, 'exclude': 78}

## Best Baselines

### keloid_binary

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| shared_genes_plus_modules | elastic_net_logreg | 0.6745 | 0.6536 | 0.6508 | 0.6178 | 23 |
| modules_only | linear_svm | 0.6293 | 0.6157 | 0.5936 | 0.5695 | 23 |
| modules_only | elastic_net_logreg | 0.6234 | 0.6110 | 0.5890 | 0.5660 | 23 |
| shared_genes | elastic_net_logreg | 0.6354 | 0.5928 | 0.6218 | 0.5605 | 23 |
| shared_genes_plus_modules | hist_gradient_boosting | 0.6133 | 0.5892 | 0.6034 | 0.5565 | 23 |

### lesional_status

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| modules_only | random_forest | 0.7917 | 0.7459 | 0.7831 | 0.7316 | 18 |
| shared_genes_plus_modules | random_forest | 0.7932 | 0.7387 | 0.7646 | 0.7108 | 18 |
| modules_only | small_mlp | 0.7416 | 0.7194 | 0.7121 | 0.6297 | 11 |
| shared_genes_plus_modules | elastic_net_logreg | 0.6291 | 0.6041 | 0.6196 | 0.5157 | 18 |
| modules_only | knn | 0.6853 | 0.5880 | 0.6103 | 0.5317 | 18 |

### cell_type

| feature_set | model | accuracy_mean | weighted_f1_mean | balanced_accuracy_mean | macro_f1_mean | weighted_f1_count |
| --- | --- | --- | --- | --- | --- | --- |
| shared_genes | elastic_net_logreg | 0.9449 | 0.9344 | 0.9427 | 0.9153 | 21 |
| shared_genes_plus_modules | elastic_net_logreg | 0.9449 | 0.9344 | 0.9427 | 0.9153 | 21 |
| shared_genes | linear_svm | 0.9396 | 0.9219 | 0.9375 | 0.9165 | 21 |
| shared_genes_plus_modules | linear_svm | 0.9396 | 0.9219 | 0.9375 | 0.9165 | 21 |
| shared_genes | random_forest | 0.9125 | 0.8875 | 0.9080 | 0.8759 | 21 |

## Frozen-Qwen Adapter Decision

- Decision: `ready_for_frozen_qwen_adapter`
- Reason: Best non-dummy baseline beats dummy baselines with a useful weighted-F1 margin and balanced accuracy above chance.

The frozen-Qwen adapter should only be trained after simple baselines show stable grouped-split signal.

## Detailed Artifacts

- `detailed_eval_breakdowns.json` contains split-level metrics, confusion matrices, and modality/accession composition for every valid baseline run.
- `detailed_eval_breakdowns.csv` is the same information flattened for quick review.
- Detailed baseline rows: 1412
