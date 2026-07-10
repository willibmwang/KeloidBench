# SpheroScar Program Interpretation (C4)

## Program model choice

- Primary: `elastic_net_logreg` on `shared_genes_plus_modules`
- Primary grouped weighted F1: 0.6325899793492481
- Secondary: `frozen_qwen_adapter`
- Secondary grouped weighted F1: 0.6000123003961889

## Module direction checks

- ECM: delta=0.363, p=2.321e-07, matches expectation=True
- myofibroblast: delta=0.182, p=0.004105, matches expectation=True
- TGFb: delta=0.206, p=7.151e-09, matches expectation=True
- hypoxia_vascular: delta=0.226, p=0.003325, matches expectation=True
- remodeling: delta=0.365, p=3.559e-07, matches expectation=True
- profibrotic_fibroblast: delta=0.533, p=8.735e-13, matches expectation=True
- antifibrotic_fibroblast: delta=0.119, p=0.1207, matches expectation=False

## Top logistic weights

- profibrotic_fibroblast_score: weight=0.6516
- hypoxia_vascular_score: weight=0.3975
- ECM_score: weight=0.3828
- remodeling_score: weight=-0.1546
- TGFb_score: weight=0.0778
- myofibroblast_score: weight=-0.0744
- antifibrotic_fibroblast_score: weight=0.0591

## scRNA fibroblast subcluster localization (top 5)

- fib_cluster_9: core_score=0.540, POSTN_module=-0.08244940597561749
- fib_cluster_3: core_score=0.352, POSTN_module=2.2137650209691055
- fib_cluster_10: core_score=0.312, POSTN_module=1.768147220961431
- fib_cluster_7: core_score=0.310, POSTN_module=1.1007183216531629
- fib_cluster_2: core_score=0.310, POSTN_module=1.3481500597976772