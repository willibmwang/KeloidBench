#!/usr/bin/env python3
"""Collect local seed files and source manifests for the spheroid MVP."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = {
    "choi_article": "https://doi.org/10.1038/s42003-024-07194-2",
    "dirand_article": "https://doi.org/10.3390/biomedicines11092350",
    "bodenmiller_zenodo": "https://doi.org/10.5281/zenodo.4271910",
    "spheroscan_training_zenodo": "https://doi.org/10.5281/zenodo.7555467",
    "spheroscan_external_test_zenodo": "https://doi.org/10.5281/zenodo.8211845",
    "GSE7980": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7980",
    "GSE44270": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44270",
    "GSE145725": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145725",
    "E-MTAB-2509": "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2509",
    "E-MTAB-4945": "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-4945",
}


def copy_if_present(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    root = args.project_root
    copied = {
        "choi_pdf": copy_if_present(root / "spheroid_2.pdf", root / "data/raw/choi/spheroid_2.pdf"),
        "choi_supp_pdf": copy_if_present(
            root / "supplementary_choi.pdf", root / "data/raw/choi/supplementary_choi.pdf"
        ),
        "choi_supp_xlsx": copy_if_present(
            root / "supplementary_choi.xlsx", root / "data/raw/choi/supplementary_choi.xlsx"
        ),
        "dirand_pdf": copy_if_present(root / "spheroid_1.pdf", root / "data/raw/dirand/spheroid_1.pdf"),
    }

    status = {}
    for name, url in SOURCE_MANIFEST.items():
        try:
            response = requests.head(url, allow_redirects=True, timeout=20)
            status[name] = {"url": url, "status_code": response.status_code, "final_url": response.url}
        except Exception as exc:  # noqa: BLE001
            status[name] = {"url": url, "error": str(exc)}

    manifest = {"copied_local_files": copied, "source_urls": status}
    out_path = root / "data/raw/source_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
