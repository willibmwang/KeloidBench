# Matched cohort sampling checklist

Use with `manuscript/matched_cohort_protocol.md` and `enrollment_template.csv`.

## Before first collection
- [ ] IRB / ethics approval recorded
- [ ] Consent form covers bulk RNA-seq + optional culture + de-identified GEO deposition
- [ ] Single bulk RNA-seq SOP frozen (kit, RIN threshold, library prep, sequencing depth)
- [ ] Arm labels locked: `keloid` | `hypertrophic_scar` | `normotrophic_scar` | `unaffected_skin`
- [ ] Enrollment template path decided; one coordinator owns `donor_id` uniqueness
- [ ] Target: ≥20 donors/arm; prefer paired keloid↔unaffected when clinically available

## Per donor / sample
- [ ] `donor_id` assigned before tissue leaves OR
- [ ] Arm chosen using clinical criteria (not molecular results)
- [ ] Anatomical site + duration + ancestry/sex/age recorded
- [ ] Prior treatments listed; if drug/radiation/laser within washout window → flag `primary_exclude=yes`
- [ ] Snap-freeze / RNAlater per SOP; aliquot barcode = `rna_sample_id`
- [ ] Optional fibroblast culture started only as **sensitivity** aliquot (never mixed into tissue primary)

## Weekly QC
- [ ] Arm counts dashboard (≥20 each before closing enrollment)
- [ ] RIN / library-fail rate tracked; failed libs do not free the donor for reassignment across arms
- [ ] No molecular predictions used to re-label arms

## After enrollment closes (before any model scoring)
- [ ] Run `scripts/assign_matched_cohort_lockbox.py --enrollment-csv ...` **once**
- [ ] Freeze `results/matched_cohort/lockbox_assignment.json` (hash enrollment CSV)
- [ ] Do **not** download/score lockbox expression until public-GEO-frozen model is fixed
- [ ] Development half: calibration-transfer checks only; nested selection stays public-GEO frozen

## Scoring rules
- [ ] Primary scorer = frozen public model (`results/breakthrough_sprint_v2/` or successor freeze)
- [ ] Endpoints scored separately: skin / pathologic scar / normal scar
- [ ] Success = same gates as breakthrough sprint + untouched lockbox half
