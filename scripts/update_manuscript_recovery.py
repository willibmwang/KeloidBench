#!/usr/bin/env python3
"""Update manuscript with locked recovery LOSO results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT = PROJECT_ROOT / "manuscript/draft.md"
REPORT = PROJECT_ROOT / "results/recovery_loso/recovery_final_report.json"
SUMMARY = PROJECT_ROOT / "results/recovery_loso/recovery_loso_summary.csv"
NESTED = PROJECT_ROOT / "results/recovery_loso/recovery_nested_selected.csv"
COVERAGE = PROJECT_ROOT / "data/processed/training/accession_module_coverage.csv"


def fmt_row(row: pd.Series) -> str:
    return (
        f"{row.get('feature_set', '?')} / {row.get('model', '?')} "
        f"(weight={row.get('weight_mode', 'none')}, adapt={row.get('adaptation', 'none')}): "
        f"F1={float(row['weighted_f1_mean']):.3f}"
    )


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit("Recovery summary not ready yet.")

    summary = pd.read_csv(SUMMARY)
    coverage = pd.read_csv(COVERAGE) if COVERAGE.exists() else pd.DataFrame()
    nested_sel = pd.read_csv(NESTED) if NESTED.exists() else pd.DataFrame()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}

    broad_fixed = summary[
        summary["endpoint"].eq("keloid_binary") & summary["stage"].eq("corrected_fixed")
    ].sort_values("weighted_f1_mean", ascending=False)
    clean_fixed = summary[
        summary["endpoint"].eq("clean_keloid_binary") & summary["stage"].eq("corrected_fixed")
    ].sort_values("weighted_f1_mean", ascending=False)
    broad_bal = summary[
        summary["endpoint"].eq("keloid_binary") & summary["stage"].eq("cohort_balanced")
    ].sort_values("weighted_f1_mean", ascending=False)
    clean_bal = summary[
        summary["endpoint"].eq("clean_keloid_binary") & summary["stage"].eq("cohort_balanced")
    ].sort_values("weighted_f1_mean", ascending=False)

    if broad_fixed.empty:
        raise SystemExit("No corrected_fixed broad results.")

    best_broad = broad_fixed.iloc[0]
    best_clean = clean_fixed.iloc[0] if len(clean_fixed) else None
    best_broad_bal = broad_bal.iloc[0] if len(broad_bal) else None
    best_clean_bal = clean_bal.iloc[0] if len(clean_bal) else None

    nested_broad = float("nan")
    nested_clean = float("nan")
    if len(nested_sel):
        if (nested_sel.endpoint == "keloid_binary").any():
            nested_broad = float(nested_sel.loc[nested_sel.endpoint.eq("keloid_binary"), "weighted_f1"].mean())
        if (nested_sel.endpoint == "clean_keloid_binary").any():
            nested_clean = float(nested_sel.loc[nested_sel.endpoint.eq("clean_keloid_binary"), "weighted_f1"].mean())

    zero_fail = coverage[coverage["all_zero_programs"]]["accession"].tolist() if len(coverage) else []
    hard = (
        coverage[coverage["accession"].isin(["GSE44270", "Sun_Burns", "GSE188952", "GSE7890"])]
        if len(coverage)
        else pd.DataFrame()
    )

    hard_lines = ""
    for item in report.get("hard_fold_bests", []):
        if item.get("stage") != "corrected_fixed":
            continue
        hard_lines += (
            f"- {item['accession']} (best corrected_fixed config): "
            f"F1={item['weighted_f1']:.3f} ({item['feature_set']} / {item['model']})\n"
        )

    claimed = bool(report.get("stop_rule", {}).get("broad_0_80_claimed", False))
    claim_text = (
        "Broad-endpoint 0.80 was **not** achieved under the frozen outer LOSO protocol."
        if not claimed
        else "Broad-endpoint 0.80 was achieved under the frozen outer LOSO protocol."
    )

    section = f"""
### RQ2 recovery (input-corrected + nested LOSO)

After repairing GPL6244/`gene_assignment` symbol parsing, GPL570 Affymetrix probe-set IDs, and Sun/Burns Ensembl→symbol mapping, previously all-zero program cohorts regained nonzero program variance. Coverage failures after rebuild: **{zero_fail or "none"}**.

**Primary endpoint** (broad profile-level `keloid_binary`, frozen 10-accession LOSO):
- Locked corrected_fixed best: **{fmt_row(best_broad)}**
- Cohort-balanced best: **{fmt_row(best_broad_bal) if best_broad_bal is not None else "n/a"}**
- Nested inner-LOSO selected mean weighted F1 (outer folds): **{nested_broad:.3f}**

**Secondary endpoint** (separately named `clean_keloid_binary` / high-confidence keloid vs normal):
- Locked corrected_fixed best: **{fmt_row(best_clean) if best_clean is not None else "pending"}**
- Cohort-balanced best: **{fmt_row(best_clean_bal) if best_clean_bal is not None else "n/a"}**
- Nested selected mean weighted F1: **{nested_clean:.3f}**

Hard-cohort program coverage after repair:
"""
    if len(hard):
        for _, row in hard.iterrows():
            section += (
                f"- {row['accession']}: module_variance_sum={row['module_variance_sum']:.4f}, "
                f"n_module_genes_present={int(row['n_module_genes_present'])}, "
                f"all_zero={bool(row['all_zero_programs'])}\n"
            )
    else:
        section += "- coverage table unavailable\n"

    if hard_lines:
        section += "\nHard-cohort predictive recovery (outer-fold best within corrected_fixed):\n" + hard_lines

    section += (
        f"\n{claim_text} The clean secondary metric is reported separately and does not replace "
        "the broad primary result. Limited adaptation (CORAL / train quantile) did not improve the "
        "locked broad mean above corrected program baselines; architecture chase was stopped once "
        "corrected programs remained below 0.70 with GSE44270 near chance.\n"
    )

    draft = DRAFT.read_text()
    marker = "### RQ2 recovery (input-corrected + nested LOSO)"
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
        insert_at = draft.find("### Held-out keloid validation")
        if insert_at == -1:
            draft = draft.rstrip() + "\n\n" + section
        else:
            draft = draft[:insert_at] + section.strip() + "\n\n" + draft[insert_at:]

    for old in (
        "job 2096361 pending",
        "job 2091391 pending",
        "job 2109215 recovery complete",
    ):
        draft = draft.replace(old, "job 2130265 recovery ladder complete")
    DRAFT.write_text(draft)
    print(
        json.dumps(
            {
                "updated": str(DRAFT),
                "best_broad_corrected_fixed": float(best_broad["weighted_f1_mean"]),
                "best_clean_corrected_fixed": float(best_clean["weighted_f1_mean"]) if best_clean is not None else None,
                "nested_broad_mean": nested_broad,
                "nested_clean_mean": nested_clean,
                "broad_0_80_claimed": claimed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
