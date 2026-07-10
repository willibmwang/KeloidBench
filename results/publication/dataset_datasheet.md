# SpheroScar Dataset Datasheet

- Profiles: 794
- Modalities: {'external_fibrosis': 293, 'scrna_spatial': 230, 'microarray': 154, 'arrayexpress': 67, 'bulk_rnaseq': 50}
- Keloid binary: {'non_keloid': 294, 'exclude': 276, 'keloid': 224}
- Fibrotic binary: {'fibrotic': 422, 'non_fibrotic': 294, 'exclude': 78}

## Held-out validation accessions

- E-MTAB-2509
- E-MTAB-4945
- GSE32537
- GSE48149
- GSE58095

## LOSO benchmark support (primary)

- Best LOSO keloid model: linear_svm / shared_genes (weighted F1=0.683 ± 0.181)
- LOSO keloid F1 bootstrap CI: 0.600 [0.507, 0.695] (random_forest / modules_only)

## Adapter comparison (negative control)

- Frozen Qwen adapter underperforms module/elastic-net baselines on LOSO keloid tasks.

## Limitations

- Keloid signature derivation excludes held-out accessions (E-MTAB-2509/4945) and external fibrosis cohorts.
- Cross-disease validation uses fibrotic-vs-non-fibrotic labels; organ and platform differ from keloid skin cohorts.
- scRNA pseudobulk dominates sample count; interpret meta-analysis study weights accordingly.