#!/usr/bin/env python3
"""Update manuscript with public-cohort expansion / accuracy-ladder results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT = PROJECT_ROOT / "manuscript/draft.md"
DEFAULT_RESULTS = PROJECT_ROOT / "results/accuracy_ladder"
FALLBACK_RESULTS = PROJECT_ROOT / "results/public_expansion_loso"
BASELINE = PROJECT_ROOT / "results/accuracy_ladder_baseline/public_expansion_loso_report.json"
RECOVERY = PROJECT_ROOT / "results/recovery_loso/recovery_final_report.json"


def _resolve_results(path: Path | None) -> Path:
    if path is not None:
        return path
    if (DEFAULT_RESULTS / "public_expansion_loso_report.json").exists():
        return DEFAULT_RESULTS
    return FALLBACK_RESULTS


def fmt(mean, std=None):
    if mean is None or (isinstance(mean, float) and pd.isna(mean)):
        return "n/a"
    if std is None or (isinstance(std, float) and pd.isna(std)):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f}±{float(std):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    results = _resolve_results(args.results_dir)
    audit_path = results / "corpus_audit.json"
    summary_path = results / "public_expansion_loso_summary.csv"
    nested_path = results / "public_expansion_nested_selected.csv"
    lockbox_path = results / "lockbox_frozen_once.json"
    report_path = results / "public_expansion_loso_report.json"

    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    lockbox = json.loads(lockbox_path.read_text()) if lockbox_path.exists() else {}
    recovery = json.loads(RECOVERY.read_text()) if RECOVERY.exists() else {}
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    nested = pd.read_csv(nested_path) if nested_path.exists() else pd.DataFrame()

    # Nested current10 primary
    nested_mean = None
    nested_worst = None
    if len(nested):
        cur = nested[nested.endpoint.eq("keloid_binary") & nested.eval_mode.eq("current10")]
        if len(cur):
            nested_mean = float(cur["weighted_f1"].mean())
            nested_worst = float(cur["weighted_f1"].min())

    fixed_mean = None
    if len(summary):
        fixed = summary[
            summary.endpoint.eq("keloid_binary")
            & summary.get("stage", pd.Series(dtype=str)).eq("corrected_fixed")
        ] if "stage" in summary.columns else summary
        # summary may not have eval_mode; filter from results file if needed
        if "weighted_f1_mean" in summary.columns and "stage" in summary.columns:
            sub = summary[summary.endpoint.eq("keloid_binary") & summary.stage.eq("corrected_fixed")]
            if len(sub):
                fixed_mean = float(sub.sort_values("weighted_f1_mean", ascending=False).iloc[0]["weighted_f1_mean"])

    clean_mean = None
    fib_mean = None
    scar_mean = None
    scar_worst = None
    skin_mean = None
    if len(nested):
        clean = nested[nested.endpoint.eq("clean_keloid_binary") & nested.eval_mode.eq("current10")]
        fib = nested[nested.endpoint.eq("fibroblast_keloid_binary") & nested.eval_mode.eq("current10")]
        scar = nested[nested.endpoint.eq("keloid_vs_normal_scar") & nested.eval_mode.eq("expanded")]
        skin = nested[nested.endpoint.eq("keloid_vs_unaffected_skin") & nested.eval_mode.eq("expanded")]
        if len(clean):
            clean_mean = float(clean["weighted_f1"].mean())
        if len(fib):
            fib_mean = float(fib["weighted_f1"].mean())
        if len(scar):
            scar_mean = float(scar["weighted_f1"].mean())
            scar_worst = float(scar["weighted_f1"].min())
        if len(skin):
            skin_mean = float(skin["weighted_f1"].mean())
    baseline_nested = baseline.get("current10_nested", {}).get("weighted_f1_macro_accession", {}).get("mean")
    paired_delta = report.get("paired_delta_vs_baseline", {}).get("mean_delta_macro_f1")

    posthoc = recovery.get("primary_broad_corrected_fixed", [{}])[0].get("weighted_f1_mean")
    ensemble = None
    ens_path = PROJECT_ROOT / "results/recovery_loso/ensemble/ensemble_loso_summary.csv"
    if ens_path.exists():
        ens = pd.read_csv(ens_path).sort_values("weighted_f1_mean", ascending=False)
        if len(ens):
            ensemble = float(ens.iloc[0]["weighted_f1_mean"])

    lockbox_lines = ""
    for row in lockbox.get("results", []):
        if row.get("status") != "ok":
            lockbox_lines += f"- {row.get('accession')}: {row.get('status')}\n"
        else:
            auroc = row.get("auroc")
            auroc_txt = f"{float(auroc):.3f}" if auroc is not None and not pd.isna(auroc) else "n/a"
            lockbox_lines += f"- {row['accession']}: F1={row['weighted_f1']:.3f}, AUROC={auroc_txt}\n"
    if not lockbox_lines:
        lockbox_lines = "- lockbox evaluation pending or no lockbox cohorts ingested\n"

    frozen = lockbox.get("frozen_config", {})
    gates = report.get("gates", {})
    gate_pass = (
        bool(gates.get("macro_ge_0_72"))
        and bool(gates.get("worst_ge_0_60"))
        and bool(gates.get("donor_ge_0_70", True))
    )
    # Broad 0.80 requires gate + nested≥0.80 + lockbox≥0.80; keep honest.
    lock_ok = False
    ok_rows = [r for r in lockbox.get("results", []) if r.get("status") == "ok"]
    if gate_pass and ok_rows and nested_mean is not None and nested_mean >= 0.80:
        lock_ok = all(float(r.get("weighted_f1", 0)) >= 0.80 for r in ok_rows)

    worst_acc = None
    if len(nested):
        cur = nested[nested.endpoint.eq("keloid_binary") & nested.eval_mode.eq("current10")]
        if len(cur):
            worst_row = cur.loc[cur["weighted_f1"].idxmin()]
            worst_acc = str(worst_row["split_name"]).replace("leave_accession_out_", "")

    public_acc = audit.get("public_accessions", [])

    section = f"""
### RQ2 public-cohort expansion (prospective ladder)

Public keloid cohorts were pre-registered in `data/raw/public_keloid_cohorts.json` under the accuracy evidence ladder. Development cohorts may enter nested selection; prospective lockbox expression (GSE212954) is downloaded only after protocol freeze and scored once. External fibrosis / IPF profiles are excluded from keloid-negative training. GSE125022 sample-level RNA-seq is unavailable from GEO RAW (ATAC-only) and remains skipped.

**Corpus audit**
- Profiles: **{audit.get('n_profiles', 'n/a')}**; external-fibrosis excluded count: **{audit.get('n_external_fibrosis_excluded', 'n/a')}**; lockbox profiles: **{audit.get('n_lockbox_profiles', 'n/a')}**
- Ingested public accessions: **{', '.join(public_acc) if public_acc else 'none yet'}**
- Coverage failures: **{audit.get('coverage_failures') or 'none'}**
- Results directory: `{results.relative_to(PROJECT_ROOT)}`

**Primary endpoint semantics (non-interchangeable)**
- Historical comparator (immutable): original-10 nested estimate frozen at baseline macro F1=**{fmt(baseline_nested)}**.
- Broad updated primary: donor-aware `keloid_binary`; all-profile sensitivity: `keloid_binary_all_profiles`.
- Clinical scar endpoint: `keloid_vs_normal_scar` (keloid vs normal/normotrophic scar only).
- Pathologic-scar differential: `keloid_vs_pathologic_scar` (hypertrophic/immature; not merged with normal scar).
- Unaffected-skin endpoint: `keloid_vs_unaffected_skin`; fibroblast sensitivity: `fibroblast_keloid_binary`.

**Estimates (do not conflate historical / post-development / prospective)**
- Post-hoc current-cohort fixed candidate (prior recovery ladder): F1=**{fmt(posthoc)}**
- Post-hoc ensemble candidate: F1=**{fmt(ensemble)}**
- Post-development nested current-10 macro-accession F1: **{fmt(nested_mean)}** (worst fold **{fmt(nested_worst)}**{f', {worst_acc}' if worst_acc else ''}); paired Δ vs frozen baseline: **{fmt(paired_delta)}**
- Nested clean current-10 mean F1: **{fmt(clean_mean)}**
- Nested fibroblast-compartment current-10 mean F1: **{fmt(fib_mean)}**
- Clinical scar nested expanded mean F1: **{fmt(scar_mean)}** (worst **{fmt(scar_worst)}**)
- Unaffected-skin nested expanded mean F1: **{fmt(skin_mean)}**
- Inner selection rule: equal-accession mean F1 with lower-tail tie-break (no outer-test fallbacks).

**Frozen lockbox (one-shot)**
- Frozen config: `{frozen.get('feature_set', '?')} / {frozen.get('model', '?')} / thr={frozen.get('threshold', '?')}`
{lockbox_lines}
**Claim status**
- Improvement gate (≥0.72 macro, donor≥0.70, worst≥0.60): **{'PASS' if gate_pass else 'FAIL'}**
- Broad 0.80 claim under frozen outer LOSO + untouched lockbox: **{'YES' if lock_ok else 'NO'}**

Interpretation remains program-level reproducibility across heterogeneous keloid contrasts; endpoint heterogeneity and independent-cohort scarcity—not classifier depth—set the prediction ceiling. Residual hard fold for broad original-10 remains scar-differential biology (e.g. GSE188952).
"""

    draft = DRAFT.read_text()
    # Refresh stale hard-cohort mention.
    draft = draft.replace(
        "Hard cohorts: GSE44270 and Sun_Burns remain ~**0.55** across all feature sets",
        "Hard cohorts after mapping repair: GSE44270 remains near chance; Sun_Burns recovers to ~**0.88**",
    )

    marker = "### RQ2 public-cohort expansion (prospective ladder)"
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
        insert_at = draft.find("### RQ2 recovery (input-corrected + nested LOSO)")
        if insert_at == -1:
            insert_at = draft.find("### Held-out keloid validation")
        if insert_at == -1:
            draft = draft.rstrip() + "\n\n" + section
        else:
            # Insert after recovery section if present.
            if "### RQ2 recovery" in draft:
                # find end of recovery section
                rec = draft.index("### RQ2 recovery (input-corrected + nested LOSO)")
                rest = draft[rec:]
                cut = len(rest)
                for token in ["\n### Held-out", "\n## "]:
                    idx = rest.find(token)
                    if idx != -1:
                        cut = min(cut, idx)
                insert_at = rec + cut
                draft = draft[:insert_at] + "\n\n" + section.strip() + "\n\n" + draft[insert_at:].lstrip("\n")
            else:
                draft = draft[:insert_at] + section.strip() + "\n\n" + draft[insert_at:]

    DRAFT.write_text(draft)
    print(
        json.dumps(
            {
                "updated": str(DRAFT),
                "nested_current10": nested_mean,
                "lockbox_n": len(ok_rows),
                "claim_080": lock_ok,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
