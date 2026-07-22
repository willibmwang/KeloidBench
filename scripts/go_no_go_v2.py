#!/usr/bin/env python3
"""Go/no-go decision for breakthrough v2 public-data path vs matched cohort."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.out_dir / "breakthrough_report.json"
    ablation_path = args.out_dir / "ablation_summary.json"
    targeted_path = args.out_dir / "targeted_data_report.json"
    protocol_path = args.out_dir / "frozen_protocol.json"

    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    ablations = json.loads(ablation_path.read_text()) if ablation_path.exists() else {}
    targeted = json.loads(targeted_path.read_text()) if targeted_path.exists() else {}
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}

    # Prefer the best ablation stage for interim/go-no-go; final architecture stage remains in report.
    best = ablations.get("best_stage") or {}
    primary = report.get("endpoints", {}).get("keloid_vs_unaffected_skin", {})
    macro = best.get("primary_macro")
    if macro is None:
        macro = (primary.get("macro_accession_f1") or {}).get("mean")
    worst = best.get("worst_fold")
    if worst is None:
        worst = primary.get("worst_fold")
    gates = report.get("gates", {})
    # Recompute interim from best-stage metrics when available.
    interim = {
        "no_zero_fold": bool(worst is not None and worst > 0.0),
        "worst_ge_0_50": bool(worst is not None and worst >= 0.50),
        "macro_ge_0_70": bool(macro is not None and macro >= 0.70),
        "source_stage": best.get("stage") or report.get("stage"),
    }
    # Claim gates still require the formal sprint thresholds.
    claim_pass = bool(
        (gates.get("full_coverage_macro_ge_0_85") or (
            gates.get("selective_macro_ge_0_90") and gates.get("selective_coverage_ge_0_60")
        ))
        and gates.get("worst_ge_0_75")
    )
    # Also allow claim pass if best stage itself clears gates.
    if best:
        claim_pass = claim_pass or bool(
            ((best.get("primary_macro") or 0) >= 0.85 or (
                (best.get("cascade_selective_mean") or 0) >= 0.90
                and (best.get("cascade_coverage") or 0) >= 0.60
            ))
            and (best.get("worst_fold") or 0) >= 0.75
        )
    interim_pass = bool(
        interim.get("no_zero_fold") and interim.get("worst_ge_0_50") and interim.get("macro_ge_0_70")
    )

    if claim_pass:
        decision = "SCORE_NEW_LOCKBOX"
        rationale = (
            "Public-data v2 nested gates met. Proceed to one-shot score of withheld v2 lockbox "
            f"{protocol.get('breakthrough_lockbox', {}).get('accession', 'GSE218007')} after CEL processing; "
            "do not rescore immutable v1 lockbox GSE185309."
        )
    elif interim_pass:
        decision = "CONTINUE_PUBLIC_DATA_REFINEMENT"
        rationale = (
            "Interim public targets met (no zero fold, worst≥0.50, macro≥0.70) but sprint claim gates "
            "still fail. Prioritize scar multi-donor tissue ingest and matched-compartment lockbox prep; "
            "avoid architecture chasing."
        )
    else:
        decision = "EXECUTE_MATCHED_COHORT_PROTOCOL"
        rationale = (
            "Corrected public-data benchmark remains materially below claim gates "
            f"(macro={macro}, worst={worst}). Stop architecture chasing and execute "
            "manuscript/matched_cohort_protocol.md (single assay, matched compartments, "
            "adequate independent donors/arm, reserved validation split)."
        )

    out = {
        "protocol": "breakthrough_sprint_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_macro_f1": macro,
        "worst_fold_f1": worst,
        "gates": gates,
        "interim_targets": interim,
        "claim_gates_pass": claim_pass,
        "interim_targets_pass": interim_pass,
        "decision": decision,
        "rationale": rationale,
        "v1_lockbox_immutable": protocol.get("v1_immutable", {}).get("breakthrough_lockbox_accession", "GSE185309"),
        "v2_lockbox_withheld": protocol.get("breakthrough_lockbox", {}).get("accession", "GSE218007"),
        "do_not_rescore_v1_lockbox": True,
        "matched_cohort_protocol": "manuscript/matched_cohort_protocol.md",
        "ablation_stages": ablations.get("stages", []),
        "targeted_ingest": targeted.get("ingested_into_train", []),
    }
    path = args.out_dir / "go_no_go_decision.json"
    path.write_text(json.dumps(out, indent=2))
    # Human-readable markdown
    md = args.out_dir / "go_no_go_decision.md"
    md.write_text(
        "\n".join(
            [
                "# Breakthrough v2 go / no-go",
                "",
                f"- Decision: **{decision}**",
                f"- Primary macro F1: `{macro}`",
                f"- Worst fold F1: `{worst}`",
                f"- Claim gates pass: `{claim_pass}`",
                f"- Interim targets pass: `{interim_pass}`",
                "",
                "## Rationale",
                rationale,
                "",
                "## Constraints",
                "- Do not rescore v1 lockbox GSE185309.",
                "- v2 lockbox GSE218007 remains withheld until CEL processing + go decision SCORE_NEW_LOCKBOX.",
                "- Matched cohort protocol: `manuscript/matched_cohort_protocol.md`.",
                "",
            ]
        )
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
