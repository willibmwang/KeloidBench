#!/usr/bin/env python3
"""Insert / refresh the breakthrough-sprint section in manuscript/draft.md."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT = PROJECT_ROOT / "manuscript/draft.md"
OUT = PROJECT_ROOT / "results/breakthrough_sprint"


def fmt(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.3f}"


def main() -> None:
    report = json.loads((OUT / "breakthrough_report.json").read_text()) if (OUT / "breakthrough_report.json").exists() else {}
    frozen = json.loads((OUT / "breakthrough_frozen_model.json").read_text()) if (OUT / "breakthrough_frozen_model.json").exists() else {}
    lockbox = json.loads((OUT / "breakthrough_lockbox_once.json").read_text()) if (OUT / "breakthrough_lockbox_once.json").exists() else {}
    proto = json.loads((OUT / "frozen_protocol.json").read_text()) if (OUT / "frozen_protocol.json").exists() else {}

    primary = report.get("endpoints", {}).get("keloid_vs_unaffected_skin", {})
    cascade = report.get("cascade") or {}
    gates = report.get("gates", {})
    cfg = frozen.get("primary_config") or {}

    lock_lines = ""
    for row in lockbox.get("results", []):
        if row.get("status") != "ok":
            lock_lines += f"- {row.get('accession')}: {row.get('status')}\n"
        else:
            lock_lines += (
                f"- {row['accession']}: full F1={fmt(row.get('weighted_f1'))}, "
                f"selective F1={fmt(row.get('selective_f1'))} "
                f"(coverage={fmt(row.get('selective_coverage'))})\n"
            )
    if not lock_lines:
        lock_lines = "- Breakthrough lockbox not yet scored\n"

    full_mean = (primary.get("macro_accession_f1") or {}).get("mean")
    sel_mean = (cascade.get("selective_macro_f1") or {}).get("mean")
    cov = cascade.get("mean_selective_coverage")
    gate_pass = bool(gates.get("full_coverage_macro_ge_0_85") or (
        gates.get("selective_macro_ge_0_90") and gates.get("selective_coverage_ge_0_60")
    )) and bool(gates.get("worst_ge_0_75", False))
    # Lockbox same-rule check
    lock_ok = False
    ok_rows = [r for r in lockbox.get("results", []) if r.get("status") == "ok"]
    if gate_pass and ok_rows:
        lock_ok = all(
            (float(r.get("weighted_f1", 0)) >= 0.85)
            or (float(r.get("selective_f1", 0)) >= 0.90 and float(r.get("selective_coverage", 0)) >= 0.60)
            for r in ok_rows
        )

    section = f"""
### RQ2 breakthrough sprint (skin-first selective cascade)

The breakthrough sprint reframes prediction as a **molecular triage** product rather than a universal keloid-vs-anything classifier. Primary endpoint is `keloid_vs_unaffected_skin`; `keloid_vs_normal_scar` and `keloid_vs_pathologic_scar` are separate specialists. A selective cascade may abstain when confidence is low. Accuracy-ladder outputs remain immutable comparators; ladder lockbox GSE212954 is not reused as the breakthrough lockbox. After scar-endpoint starvation, development training was expanded with public GEO cohorts (GSE210434 scar triad; GSE303591 / GSE282479 / GSE232079 keloid-vs-normal fibroblasts; GSE246562 stiffness auxiliary) without touching the breakthrough lockbox one-shot score.

**Protocol**
- Frozen protocol: `results/breakthrough_sprint/frozen_protocol.json` (`{proto.get('protocol', 'breakthrough_sprint_v1')}`)
- Breakthrough lockbox (expression withheld until freeze): **{proto.get('breakthrough_lockbox', {}).get('accession', frozen.get('breakthrough_lockbox', 'n/a'))}**
- Feature views: fibrosis-only, scar-discriminative, composition-only, fused multi-view
- Follow-up sampling protocol: `manuscript/matched_cohort_protocol.md`

**Primary nested estimates (do not conflate with broad original-10)**
- Full-coverage nested macro-accession F1: **{fmt(full_mean)}** (worst fold **{fmt(primary.get('worst_fold'))}**, n={primary.get('n_folds', 'n/a')})
- Selective cascade macro F1: **{fmt(sel_mean)}** at mean coverage **{fmt(cov)}**
- Frozen primary config: `{cfg.get('feature_set', '?')} / {cfg.get('model', '?')} / thr={cfg.get('threshold', '?')}`

**Specialists**
- Normal-scar specialist nested folds: **{report.get('endpoints', {}).get('keloid_vs_normal_scar', {}).get('n_folds', 0)}**
- Pathologic-scar specialist nested folds: **{report.get('endpoints', {}).get('keloid_vs_pathologic_scar', {}).get('n_folds', 0)}**
- These are reported separately and are not merged into the skin product claim.

**Breakthrough lockbox (one-shot)**
{lock_lines}
**Claim status**
- Sprint gate (full≥0.85 or selective≥0.90@≥60% coverage, worst≥0.75): **{'PASS' if gate_pass else 'FAIL'}**
- High-accuracy claim with untouched breakthrough lockbox: **{'YES' if lock_ok else 'NO'}**
- Broad original-10 0.80 claim remains governed by the accuracy-ladder protocol and is not implied by skin-endpoint gains.
"""

    draft = DRAFT.read_text() if DRAFT.exists() else ""
    marker = "### RQ2 breakthrough sprint (skin-first selective cascade)"
    if marker in draft:
        start = draft.index(marker)
        rest = draft[start + len(marker) :]
        cut = len(rest)
        for token in ["\n### ", "\n## "]:
            idx = rest.find(token)
            if idx != -1:
                cut = min(cut, idx)
        draft = draft[:start] + section.strip() + "\n\n" + rest[cut:].lstrip("\n")
    else:
        # Insert after public-cohort expansion section if present.
        anchor = "### RQ2 public-cohort expansion (prospective ladder)"
        if anchor in draft:
            start = draft.index(anchor)
            rest = draft[start:]
            cut = len(rest)
            for token in ["\n### ", "\n## "]:
                # skip the anchor itself
                idx = rest.find(token, 1)
                if idx != -1:
                    cut = min(cut, idx)
            insert_at = start + cut
            draft = draft[:insert_at] + "\n\n" + section.strip() + "\n\n" + draft[insert_at:].lstrip("\n")
        else:
            draft = draft.rstrip() + "\n\n" + section.strip() + "\n"

    DRAFT.write_text(draft)
    print(json.dumps({"updated": str(DRAFT), "gate_pass": gate_pass, "lock_ok": lock_ok}, indent=2))


if __name__ == "__main__":
    main()
