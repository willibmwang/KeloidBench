#!/usr/bin/env python3
"""Build Dataset B public spheroid morphology table.

The Bodenmiller processed Zenodo file is a multi-GB ZIP. This script reads the ZIP
central directory via HTTP range requests and extracts only small metadata tables.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import zlib
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data/processed"
RAW_DIR = PROJECT_ROOT / "data/raw/bodenmiller"
BODENMILLER_ZIP = "https://zenodo.org/api/records/4271910/files/phys_analysis_export_v3.zip/content"
SPHEROSCAN_RECORDS = {
    "spheroscan_training": "https://zenodo.org/api/records/7555467",
    "spheroscan_external_test": "https://zenodo.org/api/records/8211845",
}


def _remote_zip_entries(url: str) -> tuple[str, dict[str, dict]]:
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    final_url = head.url
    size = int(head.headers["content-length"])

    tail_start = max(0, size - 70000)
    tail = requests.get(
        final_url,
        headers={"Range": f"bytes={tail_start}-{size - 1}"},
        timeout=60,
    )
    tail.raise_for_status()
    data = tail.content
    eocd_idx = data.rfind(b"PK\x05\x06")
    if eocd_idx < 0:
        raise RuntimeError("Could not locate ZIP end-of-central-directory record")

    eocd = data[eocd_idx : eocd_idx + 22]
    fields = struct.unpack("<4s4H2LH", eocd)
    central_dir_size = fields[5]
    central_dir_offset = fields[6]

    central = requests.get(
        final_url,
        headers={
            "Range": f"bytes={central_dir_offset}-{central_dir_offset + central_dir_size - 1}"
        },
        timeout=120,
    )
    central.raise_for_status()

    entries: dict[str, dict] = {}
    pos = 0
    cd = central.content
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        info = struct.unpack("<4s6H3L5H2L", cd[pos : pos + 46])
        method = info[4]
        compressed_size = info[8]
        uncompressed_size = info[9]
        fname_len = info[10]
        extra_len = info[11]
        comment_len = info[12]
        local_offset = info[16]
        name = cd[pos + 46 : pos + 46 + fname_len].decode("utf-8", "replace")
        entries[name] = {
            "method": method,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        }
        pos += 46 + fname_len + extra_len + comment_len
    return final_url, entries


def _extract_remote_zip_member(final_url: str, entry: dict) -> bytes:
    local_offset = entry["local_offset"]
    header = requests.get(
        final_url,
        headers={"Range": f"bytes={local_offset}-{local_offset + 1024}"},
        timeout=60,
    )
    header.raise_for_status()
    local_header = header.content
    info = struct.unpack("<4s5H3L2H", local_header[:30])
    fname_len = info[9]
    extra_len = info[10]
    data_start = local_offset + 30 + fname_len + extra_len
    data_end = data_start + entry["compressed_size"] - 1

    payload = requests.get(
        final_url,
        headers={"Range": f"bytes={data_start}-{data_end}"},
        timeout=120,
    )
    payload.raise_for_status()
    if entry["method"] == 8:
        return zlib.decompress(payload.content, -15)
    if entry["method"] == 0:
        return payload.content
    raise RuntimeError(f"Unsupported ZIP compression method: {entry['method']}")


def build_bodenmiller_table(raw_dir: Path) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_url, entries = _remote_zip_entries(BODENMILLER_ZIP)
    manifest = pd.DataFrame(
        [
            {
                "path": name,
                **entry,
            }
            for name, entry in entries.items()
        ]
    )
    manifest.to_csv(raw_dir / "phys_analysis_export_v3_manifest.csv", index=False)

    if "image_meta.csv" not in entries:
        raise RuntimeError("Bodenmiller image_meta.csv was not found in the archive")

    image_meta_bytes = _extract_remote_zip_member(final_url, entries["image_meta.csv"])
    (raw_dir / "image_meta.csv").write_bytes(image_meta_bytes)
    image_meta = pd.read_csv(io.BytesIO(image_meta_bytes))

    area = image_meta["image_shape_h"].astype(float) * image_meta["image_shape_w"].astype(float)
    diameter = (4 * area / math.pi) ** 0.5
    out = pd.DataFrame(
        {
            "source_dataset": "bodenmiller_zanotelli_2020",
            "image_id": image_meta["image_id"],
            "condition_code": image_meta["condition_name"],
            "cell_line": image_meta["cellline"],
            "treatment": image_meta.get("condition_name"),
            "concentration": image_meta.get("concentration"),
            "time_point": image_meta.get("time_point"),
            "plate_id": image_meta.get("plate_id"),
            "well_name": image_meta.get("well_name"),
            "area": area,
            "diameter": diameter,
            "image_shape_h": image_meta["image_shape_h"],
            "image_shape_w": image_meta["image_shape_w"],
            "morphology_quality_label": "valid",
        }
    )
    return out


def fetch_spheroscan_manifest(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset, url in SPHEROSCAN_RECORDS.items():
        record = requests.get(url, timeout=30)
        record.raise_for_status()
        data = record.json()
        for file_info in data.get("files", []):
            rows.append(
                {
                    "source_dataset": dataset,
                    "title": data.get("metadata", {}).get("title"),
                    "file_key": file_info.get("key"),
                    "size_bytes": file_info.get("size"),
                    "download_url": file_info.get("links", {}).get("self"),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(raw_dir / "spheroscan_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    morphology = build_bodenmiller_table(args.raw_dir)
    out_path = args.out_dir / "dataset_b_morphology.parquet"
    morphology.to_parquet(out_path, index=False)

    spheroscan_manifest = fetch_spheroscan_manifest(PROJECT_ROOT / "data/raw/spheroscan")
    summary = {
        "dataset_b_rows": int(len(morphology)),
        "bodenmiller_source": BODENMILLER_ZIP,
        "spheroscan_records": spheroscan_manifest.to_dict(orient="records"),
        "note": "SpheroScan image archives are recorded in the manifest; images are not downloaded by default.",
    }
    with (args.out_dir / "dataset_b_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"dataset_b_morphology: {len(morphology)} rows -> {out_path}")
    print(f"spheroscan_manifest: {len(spheroscan_manifest)} rows")


if __name__ == "__main__":
    main()
