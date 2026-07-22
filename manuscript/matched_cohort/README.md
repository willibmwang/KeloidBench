# Matched cohort operations package

Preregistered follow-up after breakthrough v2 go/no-go = `EXECUTE_MATCHED_COHORT_PROTOCOL`.

## Documents
| File | Purpose |
|------|---------|
| `../matched_cohort_protocol.md` | Preregistered design, endpoints, success gates |
| `sampling_checklist.md` | Pre-collection → post-enrollment ops checklist |
| `enrollment_template.csv` | CRF columns + example rows (paired donor allowed) |
| `public_followups.md` | Optional public GEO side-work (no architecture chase) |

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/matched_cohort_power.py` | Sample-size / Wilson CI planning → `results/matched_cohort/` |
| `scripts/assign_matched_cohort_lockbox.py` | One-shot donor-grain 50% lockbox after enrollment closes |

## Generated artifacts
| Path | Notes |
|------|-------|
| `results/matched_cohort/power_sample_size.md` | Planning memo (20/arm, 50% lockbox) |
| `results/matched_cohort/example_lockbox_demo/` | Synthetic smoke-test assignment (not real enrollment) |

## Workflow
1. Freeze assay SOP + IRB; enroll with `enrollment_template.csv` columns.
2. Hit ≥20 unique donors per arm (paired keloid↔unaffected preferred).
3. Close enrollment → run lockbox assigner **once** (no `--force` in production).
4. Score with frozen public model under `results/breakthrough_sprint_v2/`; never feed lockbox into selection.
5. Claim only if sprint gates pass on the untouched lockbox half.
