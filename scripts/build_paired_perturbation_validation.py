#!/usr/bin/env python3
"""Bounded paired-donor / perturbation validation (representation only; not disease labels)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING = PROJECT_ROOT / "data/processed/training"
OUT = PROJECT_ROOT / "results/breakthrough_sprint"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(TRAINING / "profile_manifest.parquet")
    fused_path = TRAINING / "features/fused_multiview.parquet"
    if not fused_path.exists():
        fused_path = TRAINING / "features/scar_discriminative_rank_programs.parquet"
    feats = pd.read_parquet(fused_path).set_index("sample_id")
    feature_cols = list(feats.columns)

    # Paired-like: same patient_id with keloid and non-keloid under skin endpoint.
    rows = []
    if "keloid_vs_unaffected_skin" in manifest.columns:
        sub = manifest[manifest["keloid_vs_unaffected_skin"].isin(["keloid", "non_keloid"])].copy()
        for patient, group in sub.groupby("patient_id"):
            if str(patient).lower() in {"unknown", "nan", "none", ""}:
                continue
            labels = set(group["keloid_vs_unaffected_skin"])
            if labels != {"keloid", "non_keloid"}:
                continue
            k_ids = group.loc[group["keloid_vs_unaffected_skin"].eq("keloid"), "sample_id"]
            n_ids = group.loc[group["keloid_vs_unaffected_skin"].eq("non_keloid"), "sample_id"]
            for kid in k_ids:
                for nid in n_ids:
                    if kid not in feats.index or nid not in feats.index:
                        continue
                    delta = feats.loc[kid, feature_cols] - feats.loc[nid, feature_cols]
                    rows.append(
                        {
                            "patient_id": patient,
                            "accession": group["accession"].iloc[0],
                            "keloid_sample": kid,
                            "control_sample": nid,
                            "mean_abs_delta": float(np.abs(delta).mean()),
                            "signed_mean_delta": float(delta.mean()),
                            **{f"delta_{c}": float(delta[c]) for c in feature_cols[:12]},
                        }
                    )
    paired = pd.DataFrame(rows)
    if len(paired):
        paired.to_csv(OUT / "paired_donor_deltas.csv", index=False)

    # Perturbation / stiffness: directional check only.
    pert = manifest[manifest["encoder_task"].isin(["stiffness_response", "wound_susceptibility"])].copy()
    pert_summary = {
        "n_paired_deltas": int(len(paired)),
        "n_perturbation_profiles": int(len(pert)),
        "perturbation_tasks": pert["encoder_task"].value_counts().to_dict() if len(pert) else {},
        "note": (
            "Paired deltas and perturbation profiles are representation diagnostics only; "
            "they are never used as synthetic disease labels for the breakthrough cascade."
        ),
    }
    (OUT / "paired_perturbation_validation.json").write_text(json.dumps(pert_summary, indent=2))
    print(json.dumps(pert_summary, indent=2))


if __name__ == "__main__":
    main()
