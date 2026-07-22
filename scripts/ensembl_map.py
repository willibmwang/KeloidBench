"""Ensembl gene ID → HGNC symbol mapping with a versioned local cache."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "data/processed/bulk_rnaseq/ensembl_symbol_cache.json"

# Minimal fallback for program genes if network/cache unavailable.
_FALLBACK = {
    "ENSG00000108821": "COL1A1",
    "ENSG00000168542": "COL3A1",
    "ENSG00000115414": "FN1",
    "ENSG00000107796": "ACTA2",
    "ENSG00000149591": "TAGLN",
    "ENSG00000101335": "MYL9",
    "ENSG00000105329": "TGFB1",
    "ENSG00000119699": "TGFB3",
    "ENSG00000106799": "TGFBR1",
    "ENSG00000163513": "TGFBR2",
    "ENSG00000175387": "SMAD2",
    "ENSG00000166949": "SMAD3",
    "ENSG00000100644": "HIF1A",
    "ENSG00000261371": "PECAM1",
    "ENSG00000154734": "VWF",
    "ENSG00000128052": "KDR",
    "ENSG00000157227": "MMP14",
    "ENSG00000148848": "ADAM12",
    "ENSG00000166033": "HTRA1",
    "ENSG00000168517": "CTHRC1",
    "ENSG00000133110": "POSTN",
    "ENSG00000115461": "IGFBP2",
    "ENSG00000105664": "COMP",
    "ENSG00000106819": "ASPN",
    "ENSG00000120708": "TGFBI",
    "ENSG00000060718": "COL11A1",
    "ENSG00000145423": "SFRP2",
    "ENSG00000135547": "SFRP4",
    "ENSG00000118523": "CTGF",
    "ENSG00000142871": "CYR61",
    "ENSG00000176571": "ANKRD1",
    "ENSG00000114019": "AMOTL2",
    "ENSG00000187079": "TEAD1",
    "ENSG00000136244": "IL6",
    "ENSG00000169429": "CXCL8",
    "ENSG00000108691": "CCL2",
    "ENSG00000090339": "ICAM1",
    "ENSG00000162692": "VCAM1",
    "ENSG00000163453": "SERPINE1",
    "ENSG00000026025": "VIM",
    "ENSG00000124216": "SNAI1",
    "ENSG00000019549": "SNAI2",
    "ENSG00000148516": "ZEB1",
    "ENSG00000122641": "TWIST1",
    "ENSG00000170558": "CDH2",
}


def strip_ensembl_version(gene_id: str) -> str:
    """Normalize Ensembl IDs, including Gencode-style suffixes like ENSG....14_2."""
    text = str(gene_id).strip().upper()
    match = re.match(r"^(ENS[A-Z]*G?\d+)", text)
    if match:
        return match.group(1)
    return re.sub(r"\.\d+$", "", text)


def _download_ensembl_map(timeout: int = 120) -> dict[str, str]:
    """Fetch Ensembl gene id → symbol via BioMart XML query."""
    import urllib.parse

    query = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="hsapiens_gene_ensembl" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="hgnc_symbol"/>
    </Dataset>
</Query>"""
    url = "https://www.ensembl.org/biomart/martservice?query=" + urllib.parse.quote(query)
    mapping: dict[str, str] = {}
    with urllib.request.urlopen(url, timeout=timeout) as handle:
        for raw in handle:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or "\t" not in line:
                continue
            ensembl_id, symbol = line.split("\t", 1)
            ensembl_id = strip_ensembl_version(ensembl_id)
            symbol = symbol.strip().upper()
            if ensembl_id.startswith("ENSG") and symbol and re.match(r"^[A-Z][A-Z0-9.-]*$", symbol):
                mapping[ensembl_id] = symbol
    if len(mapping) < 1000:
        raise RuntimeError(f"BioMart returned too few mappings ({len(mapping)})")
    return mapping


def load_ensembl_symbol_map(cache_path: Path = DEFAULT_CACHE, allow_download: bool = True) -> dict[str, str]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        mapping = payload.get("mapping", payload)
        return {strip_ensembl_version(k): str(v).upper() for k, v in mapping.items()}

    mapping = dict(_FALLBACK)
    if allow_download:
        try:
            mapping = _download_ensembl_map()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "source": "ensembl_biomart_hsapiens_gene_ensembl",
                        "n_mappings": len(mapping),
                        "mapping": mapping,
                    },
                    indent=2,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Ensembl download failed ({exc}); using fallback module-gene map.")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "source": "fallback_module_genes",
                        "n_mappings": len(mapping),
                        "mapping": mapping,
                    },
                    indent=2,
                )
            )
    return mapping


def map_ensembl_ids_to_symbols(ids: list[str], mapping: dict[str, str] | None = None) -> list[str | None]:
    mapping = mapping or load_ensembl_symbol_map()
    out: list[str | None] = []
    for gene_id in ids:
        key = strip_ensembl_version(gene_id)
        out.append(mapping.get(key))
    return out
