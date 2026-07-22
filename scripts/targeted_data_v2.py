#!/usr/bin/env python3
"""Attempt high-value targeted ingest for breakthrough v2 (hard inclusion gates)."""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results/breakthrough_sprint_v2"
RAW = PROJECT_ROOT / "data/raw/public_keloid"

CANDIDATES = [
    {
        "accession": "GSE218007",
        "why": "Multi-donor Affymetrix keloid transcriptome; v2 withheld lockbox.",
        "gate": "Requires CEL→symbol pipeline (R/oligo) or published processed matrix.",
        "action": "probe",
    },
    {
        "accession": "GSE307504",
        "why": "Pathological scar atlas for scar specialists.",
        "gate": "Need human multi-donor scar-type RNA with recoverable sample matrix.",
        "action": "probe",
    },
    {
        "accession": "GSE125022",
        "why": "Prior skipped recovery; RNA may need SRA.",
        "gate": "GEO RAW is ATAC-only; SRA reprocess required.",
        "action": "skip_documented",
    },
    {
        "accession": "GSE90051",
        "why": "Matched patient contrasts.",
        "gate": "Paired log-ratio only unless deconvolved to sample-level.",
        "action": "skip_documented",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def probe_gse218007() -> dict:
    raw_tar = RAW / "GSE218007" / "GSE218007_RAW.tar"
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE218nnn/GSE218007/suppl/GSE218007_RAW.tar"
    status = {
        "accession": "GSE218007",
        "url": url,
        "local_tar": str(raw_tar),
        "tar_present": raw_tar.exists(),
        "ingestible_now": False,
        "reason": "",
    }
    try:
        import shutil
        import urllib.request

        raw_tar.parent.mkdir(parents=True, exist_ok=True)
        if not raw_tar.exists() or raw_tar.stat().st_size < 1000:
            # Download only filelist via HEAD-equivalent: fetch first members listing by streaming is heavy.
            # Instead record that CEL pipeline is missing.
            status["reason"] = "CEL-only Affymetrix GPL23126; no R/oligo in env; do not ingest raw CELs into train."
            return status
        with tarfile.open(raw_tar) as tf:
            names = [Path(m.name).name for m in tf.getmembers() if m.isfile()][:5]
        status["sample_files_head"] = names
        status["reason"] = "RAW present but CEL processing unavailable; keep withheld as v2 lockbox."
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"probe failed: {exc}"
    return status


def probe_gse307504() -> dict:
    import urllib.request

    acc = "GSE307504"
    folder = "GSE307nnn"
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder}/{acc}/"
    status = {"accession": acc, "ingestible_now": False, "suppl": [], "reason": ""}
    try:
        with urllib.request.urlopen(base + "suppl/", timeout=60) as handle:
            html = handle.read().decode("utf-8", errors="replace")
        import re

        files = re.findall(rf'href="({acc}[^"]+)"', html)
        status["suppl"] = files[:20]
        # Prefer count/TPM matrices over RAW scRNA.
        matrices = [f for f in files if any(tok in f.lower() for tok in ["count", "tpm", "fpkm", "expression", "csv", "txt"])]
        if matrices and not any(f.endswith(".tar") and "RAW" in f for f in matrices):
            status["candidate_matrices"] = matrices
            status["reason"] = "Found candidate processed matrices; manual label audit required before ingest."
        else:
            status["reason"] = (
                "Suppl listing is empty or RAW/scRNA-dominated; defer until multi-donor bulk scar matrix confirmed."
            )
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"GEO probe failed: {exc}"
    return status


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "Hard inclusion gates; no indiscriminate GEO expansion.",
        "probes": {
            "GSE218007": probe_gse218007(),
            "GSE307504": probe_gse307504(),
            "GSE125022": {
                "accession": "GSE125022",
                "ingestible_now": False,
                "reason": "Previously documented: GEO RAW ATAC-only; needs SRA RNA reprocess.",
            },
            "GSE90051": {
                "accession": "GSE90051",
                "ingestible_now": False,
                "reason": "Paired log-ratio contrasts; sample-level expression not available.",
            },
        },
        "ingested_into_train": [],
        "note": (
            "No new development cohorts meet automated hard gates in this pass. "
            "v2 lockbox GSE218007 remains withheld. Scar specialist gains continue to rely on "
            "GSE210434 / GSE245660 / GSE188952 already in corpus."
        ),
    }
    out = args.out_dir / "targeted_data_report.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({"written": str(out), "ingested": results["ingested_into_train"]}, indent=2))


if __name__ == "__main__":
    main()
