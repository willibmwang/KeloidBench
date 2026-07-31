# PI share — SpheroScar primary training tables

## Send these
1. **Headline model table (recommended):** `transferable_7studies_headline.csv`
   - 92 profiles / 7 studies / labels + fused multiview program features
   - This is the table behind the ~0.92 mean macro F1 claim (GSE173900 excluded)

2. **Labels only (smallest):** `transferable_7studies_manifest_only.csv`

3. **Full primary development (includes non-transferable GSE173900):** `primary_development_all8studies.csv`

## What the columns mean
- `keloid_vs_unaffected_skin`: training label (`keloid` / `non_keloid`)
- `accession`: GEO/ArrayExpress study (LOSO unit)
- `split_group` / `patient_id`: donor grouping
- `*_rank` columns: low-dimensional fibroblast/composition program features used by the classifiers

## Not included (too large / local-only)
Raw gene matrices and GEO RAW.tar downloads. Features are the actual ML inputs.

## Repo
https://github.com/willibmwang/SpheroScar
