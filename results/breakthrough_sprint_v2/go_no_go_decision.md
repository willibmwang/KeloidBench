# Breakthrough v2 go / no-go

- Decision: **EXECUTE_MATCHED_COHORT_PROTOCOL**
- Primary macro F1: `0.7132277869569251`
- Worst fold F1: `0.43001443001442996`
- Claim gates pass: `False`
- Interim targets pass: `False`

## Rationale
Corrected public-data benchmark remains materially below claim gates (macro=0.7132277869569251, worst=0.43001443001442996). Stop architecture chasing and execute manuscript/matched_cohort_protocol.md (single assay, matched compartments, adequate independent donors/arm, reserved validation split).

## Constraints
- Do not rescore v1 lockbox GSE185309.
- v2 lockbox GSE218007 remains withheld until CEL processing + go decision SCORE_NEW_LOCKBOX.
- Matched cohort protocol: `manuscript/matched_cohort_protocol.md`.
