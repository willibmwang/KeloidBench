"""Entrez Gene ID → HGNC symbol mapping with a local cache."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "data/processed/public_keloid/entrez_symbol_cache.json"


def load_entrez_symbol_map(cache_path: Path = DEFAULT_CACHE) -> dict[str, str]:
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if isinstance(data, dict) and data:
                return {str(k): str(v) for k, v in data.items()}
        except Exception:  # noqa: BLE001
            pass
    return {}


def save_entrez_symbol_map(mapping: dict[str, str], cache_path: Path = DEFAULT_CACHE) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def fetch_entrez_symbols(entrez_ids: list[str], cache_path: Path = DEFAULT_CACHE, batch_size: int = 500) -> dict[str, str]:
    """Resolve Entrez IDs via MyGene POST, merging into the local cache."""
    mapping = load_entrez_symbol_map(cache_path)
    missing = sorted({str(i).strip() for i in entrez_ids if str(i).strip().isdigit() and str(i).strip() not in mapping})
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        body = urllib.parse.urlencode({"ids": ",".join(batch), "fields": "symbol"}).encode()
        req = urllib.request.Request(
            "https://mygene.info/v3/gene",
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as handle:
                payload = json.loads(handle.read().decode())
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Entrez lookup failed for batch starting {batch[0]}: {exc}")
            time.sleep(1.0)
            continue
        if isinstance(payload, dict):
            payload = [payload]
        for row in payload:
            if not isinstance(row, dict):
                continue
            eid = str(row.get("query") or row.get("_id") or "").strip()
            symbol = str(row.get("symbol") or "").strip().upper()
            if eid and symbol and symbol not in {"NAN", "NONE", "NA"}:
                mapping[eid] = symbol
        time.sleep(0.15)
    if mapping:
        save_entrez_symbol_map(mapping, cache_path)
    return mapping
