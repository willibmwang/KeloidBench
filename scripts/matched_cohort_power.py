#!/usr/bin/env python3
"""Power / sample-size helpers for the matched multi-arm cohort.

Conservative planning numbers derived from breakthrough v2 public nested
performance (best Stage B macro F1 ≈0.71, worst ≈0.43) and the claim gates
(macro ≥0.85 or selective ≥0.90 @ ≥60% coverage; worst ≥0.75).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results/matched_cohort"


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def donors_for_se(target_se: float, p: float = 0.75) -> int:
    """Approx donors for binomial SE of a donor-level accuracy proxy."""
    # SE ≈ sqrt(p(1-p)/n) => n ≈ p(1-p)/SE^2
    n = p * (1 - p) / (target_se**2)
    return int(math.ceil(n))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--donors-per-arm", type=int, default=20)
    p.add_argument("--lockbox-fraction", type=float, default=0.50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = ["keloid", "hypertrophic_scar", "normotrophic_scar", "unaffected_skin"]
    n_arm = args.donors_per_arm
    n_total = n_arm * len(arms)
    n_lock = int(round(n_arm * args.lockbox_fraction))
    n_dev = n_arm - n_lock

    # Planning scenarios for primary keloid vs unaffected skin (equal arms).
    scenarios = []
    for label, p_hat in [
        ("v2_public_best_macro_proxy", 0.71),
        ("interim_target", 0.75),
        ("claim_gate", 0.85),
        ("selective_gate_proxy", 0.90),
    ]:
        # Evaluable primary contrast uses keloid + unaffected only.
        n_primary = 2 * n_arm
        n_lb_primary = 2 * n_lock
        lo, hi = wilson_ci(p_hat, n_primary)
        lo_lb, hi_lb = wilson_ci(p_hat, n_lb_primary)
        scenarios.append(
            {
                "label": label,
                "assumed_donor_accuracy_proxy": p_hat,
                "n_primary_donors": n_primary,
                "wilson95_full": [lo, hi],
                "n_lockbox_primary_donors": n_lb_primary,
                "wilson95_lockbox": [lo_lb, hi_lb],
            }
        )

    # Worst-fold gate: with leave-one-donor-out on lockbox half, need enough donors
    # so a single hard donor cannot dominate the claim.
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_ref": "manuscript/matched_cohort_protocol.md",
        "arms": arms,
        "recommended": {
            "donors_per_arm": n_arm,
            "profiles_if_one_per_donor": n_total,
            "lockbox_fraction": args.lockbox_fraction,
            "lockbox_donors_per_arm": n_lock,
            "development_donors_per_arm": n_dev,
            "rationale": (
                "≥20/arm matches the preregistered protocol and yields ≥10 lockbox donors/arm "
                "at 50% reserve. Public v2 nested macro≈0.71 with worst≈0.43 implies the "
                "binding constraint is hard-donor generalization, not mean performance alone."
            ),
        },
        "sensitivity_upsizing": {
            "donors_per_arm_30": {
                "total": 120,
                "lockbox_per_arm": 15,
                "note": "Use if site can expand; better worst-donor stability.",
            },
            "se_0_05_donors_at_p0_75": donors_for_se(0.05, 0.75),
            "se_0_07_donors_at_p0_75": donors_for_se(0.07, 0.75),
        },
        "scenarios": scenarios,
        "assay_constraints": [
            "Single bulk RNA-seq library/sequencing protocol across all arms",
            "Exclude treated cultures from primary endpoints",
            "Optional matched fibroblast cultures = sensitivity only",
        ],
        "success_criteria": {
            "full_macro_f1": 0.85,
            "selective_f1": 0.90,
            "selective_coverage": 0.60,
            "worst_fold_f1": 0.75,
            "lockbox_same_rule": True,
        },
    }
    (args.out_dir / "power_sample_size.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Matched cohort sample-size plan",
        "",
        f"- Recommended: **{n_arm} donors/arm** ({n_total} total), **{int(args.lockbox_fraction*100)}% lockbox** → {n_lock}/arm reserved.",
        f"- Primary contrast evaluable donors (keloid + unaffected): **{2*n_arm}** full / **{2*n_lock}** lockbox.",
        "- Public v2 evidence: best nested macro F1≈0.71, worst≈0.43 → prioritize donor diversity over more technical replicates.",
        "",
        "## Scenarios (Wilson 95% around accuracy proxy)",
    ]
    for s in scenarios:
        md.append(
            f"- `{s['label']}` p={s['assumed_donor_accuracy_proxy']}: "
            f"full {s['wilson95_full'][0]:.2f}–{s['wilson95_full'][1]:.2f} (n={s['n_primary_donors']}); "
            f"lockbox {s['wilson95_lockbox'][0]:.2f}–{s['wilson95_lockbox'][1]:.2f} (n={s['n_lockbox_primary_donors']})"
        )
    (args.out_dir / "power_sample_size.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"written": str(args.out_dir / "power_sample_size.json"), "donors_per_arm": n_arm}, indent=2))


if __name__ == "__main__":
    main()
