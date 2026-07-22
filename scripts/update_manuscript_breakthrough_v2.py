#!/usr/bin/env python3
"""Insert / refresh the breakthrough v2 + matched-cohort section in manuscript/draft.md."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT = PROJECT_ROOT / "manuscript/draft.md"
V1 = PROJECT_ROOT / "results/breakthrough_sprint"
V2 = PROJECT_ROOT / "results/breakthrough_sprint_v2"


def fmt(x):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    v1_report = json.loads((V1 / "breakthrough_report.json").read_text()) if (V1 / "breakthrough_report.json").exists() else {}
    v1_lock = json.loads((V1 / "breakthrough_lockbox_once.json").read_text()) if (V1 / "breakthrough_lockbox_once.json").exists() else {}
    v2_abl = json.loads((V2 / "ablation_summary.json").read_text()) if (V2 / "ablation_summary.json").exists() else {}
    go = json.loads((V2 / "go_no_go_decision.json").read_text()) if (V2 / "go_no_go_decision.json").exists() else {}
    elig = json.loads((V2 / "eligibility_before_after.json").read_text()) if (V2 / "eligibility_before_after.json").exists() else {}

    v1_primary = v1_report.get("endpoints", {}).get("keloid_vs_unaffected_skin", {})
    v1_mean = (v1_primary.get("macro_accession_f1") or {}).get("mean")
    v1_worst = v1_primary.get("worst_fold")
    best = v2_abl.get("best_stage") or {}
    lock_line = "n/a"
    for row in v1_lock.get("results", []):
        if row.get("status") == "ok":
            lock_line = (
                f"{row.get('accession')}: full F1={fmt(row.get('weighted_f1'))}, "
                f"selective F1={fmt(row.get('selective_f1'))}"
            )
            break

    section = f"""
### RQ2 breakthrough sprint v2 (endpoint-pure ceiling + matched-cohort decision)

Public GEO expansion alone did not clear the sprint gates. v1 nested primary macro F1 was **{fmt(v1_mean)}** with worst fold **{fmt(v1_worst)}** (complete inversion on fibroblast culture GSE282479). v2 froze an endpoint-pure protocol that removes culture lines, stiffness arms, and endothelial cohorts from `keloid_vs_unaffected_skin`, adds donor-level nested selection / inner-OOF abstention, and keeps the v1 lockbox (**GSE185309**) immutable.

**v2 nested evidence (tissue-only primary folds)**
- Best stage (donor selection + calibration): macro F1 **{fmt(best.get('primary_macro'))}**, worst fold **{fmt(best.get('worst_fold'))}**, n={best.get('n_folds', 'n/a')}
- Zero-fold inversion removed; selective coverage gate still the only consistently passing sprint gate
- Cohorts removed from primary skin: {', '.join(elig.get('removed_from_primary_skin', [])) or 'see ledger'}
- v1 lockbox one-shot (immutable): {lock_line}

**Go / no-go**
- Decision: **{go.get('decision', 'n/a')}**
- Claim gates (full≥0.85 or selective≥0.90@≥60%, worst≥0.75): **{'PASS' if go.get('claim_gates_pass') else 'FAIL'}**
- Interpretation: compartment-matched public data lifts the floor (~0.61→~0.71) but does not support a high-accuracy claim; further architecture chase on heterogeneous GEO is discontinued.

**Prospective matched cohort (next required evidence)**
- Protocol: `manuscript/matched_cohort_protocol.md`
- Ops package: `manuscript/matched_cohort/` (checklist, enrollment CRF, lockbox assigner, power plan)
- Design: ≥20 donors × 4 arms (keloid / HTS / normotrophic scar / unaffected skin), one bulk RNA-seq assay, 50% donor lockbox reserved before scoring
- Same success criteria as the breakthrough sprint; public-GEO-frozen model is the primary scorer
"""

    draft = DRAFT.read_text() if DRAFT.exists() else ""
    marker = "### RQ2 breakthrough sprint v2 (endpoint-pure ceiling + matched-cohort decision)"
    old_marker = "### RQ2 breakthrough sprint (skin-first selective cascade)"
    if marker in draft:
        start = draft.index(marker)
        rest = draft[start + len(marker) :]
        cut = len(rest)
        for token in ["\n### ", "\n## "]:
            idx = rest.find(token)
            if idx != -1:
                cut = min(cut, idx)
        draft = draft[:start] + section.strip() + "\n\n" + rest[cut:].lstrip("\n")
    elif old_marker in draft:
        start = draft.index(old_marker)
        rest = draft[start:]
        cut = len(rest)
        for token in ["\n### ", "\n## "]:
            idx = rest.find(token, 1)
            if idx != -1:
                cut = min(cut, idx)
        # Keep v1 section, append v2 after it.
        insert_at = start + cut
        draft = draft[:insert_at] + "\n\n" + section.strip() + "\n\n" + draft[insert_at:].lstrip("\n")
    else:
        draft = draft.rstrip() + "\n\n" + section.strip() + "\n"

    DRAFT.write_text(draft)
    print(json.dumps({"updated": str(DRAFT), "decision": go.get("decision"), "best_macro": best.get("primary_macro")}, indent=2))


if __name__ == "__main__":
    main()
