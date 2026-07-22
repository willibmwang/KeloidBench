#!/usr/bin/env python3
"""Assign untouched lockbox donors for the matched cohort (run once after enrollment).

Input CSV must include at least: donor_id, arm
Optional: anatomical_site, sex, ancestry_self_id, age_years, rna_sample_id, ...

Rules:
- Multiple rows per donor_id are allowed (paired arms / replicate aliquots).
- Split is assigned once per unique donor_id and propagated to all rows.
- Stratify by the frozenset of arms that donor contributes (keeps paired
  keloid↔unaffected blocks together in the same split).
- Writes a frozen assignment JSON/CSV; refuses to overwrite unless --force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results/matched_cohort"
ARMS = {"keloid", "hypertrophic_scar", "normotrophic_scar", "unaffected_skin"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enrollment-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--lockbox-fraction", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assign_path = args.out_dir / "lockbox_assignment.csv"
    meta_path = args.out_dir / "lockbox_assignment.json"
    if assign_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {assign_path}; pass --force if intentional.")

    df = pd.read_csv(args.enrollment_csv)
    need = {"donor_id", "arm"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Enrollment CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["donor_id"] = df["donor_id"].astype(str).str.strip()
    df["arm"] = df["arm"].astype(str).str.strip().str.lower()
    bad_arms = sorted(set(df["arm"]) - ARMS)
    if bad_arms:
        raise SystemExit(f"Unknown arms: {bad_arms}; expected {sorted(ARMS)}")
    if df.duplicated(subset=["donor_id", "arm", "rna_sample_id"] if "rna_sample_id" in df.columns else ["donor_id", "arm"]).any():
        raise SystemExit("Duplicate donor_id+arm(+rna_sample_id) rows detected.")

    # Unique donors stratified by arm-set (paired blocks stay intact).
    donor_arms = (
        df.groupby("donor_id")["arm"]
        .apply(lambda s: "|".join(sorted(set(s))))
        .to_dict()
    )
    rng = np.random.default_rng(args.seed)
    role_by_donor: dict[str, str] = {}
    strata: dict[str, list[str]] = {}
    for donor, arm_key in donor_arms.items():
        strata.setdefault(arm_key, []).append(donor)
    for arm_key, donors in strata.items():
        donors = list(donors)
        rng.shuffle(donors)
        n_lb = max(1, int(round(len(donors) * args.lockbox_fraction))) if len(donors) else 0
        lockbox = set(donors[:n_lb])
        for d in donors:
            role_by_donor[d] = "lockbox" if d in lockbox else "development"

    out = df.copy()
    out["split_role"] = out["donor_id"].map(role_by_donor)
    out = out.sort_values(["arm", "split_role", "donor_id"])
    out.to_csv(assign_path, index=False)

    raw = args.enrollment_csv.read_bytes()
    donor_roles = pd.Series(role_by_donor)
    meta = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "enrollment_csv": str(args.enrollment_csv),
        "enrollment_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": args.seed,
        "lockbox_fraction": args.lockbox_fraction,
        "n_rows": int(len(out)),
        "n_unique_donors": int(len(role_by_donor)),
        "n_lockbox_donors": int((donor_roles == "lockbox").sum()),
        "n_development_donors": int((donor_roles == "development").sum()),
        "strata_arm_sets": {k: len(v) for k, v in strata.items()},
        "counts_rows": {
            f"{a}|{r}": int(v) for (a, r), v in out.groupby(["arm", "split_role"]).size().items()
        },
        "counts_donors": {
            role: int((donor_roles == role).sum()) for role in ("development", "lockbox")
        },
        "rules": [
            "Split is donor-grain: all rows for a donor inherit the same split_role.",
            "Paired multi-arm donors are stratified by their arm-set and kept intact.",
            "Lockbox donors are untouched until public-GEO-frozen model is applied once.",
            "No lockbox expression may enter nested selection or threshold tuning.",
            "Development half may be used only for calibration-transfer checks.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "written": str(assign_path),
                "meta": str(meta_path),
                "n_unique_donors": meta["n_unique_donors"],
                "counts_rows": meta["counts_rows"],
                "counts_donors": meta["counts_donors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
