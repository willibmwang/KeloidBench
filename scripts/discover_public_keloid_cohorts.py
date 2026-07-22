#!/usr/bin/env python3
"""Reproducible GEO metadata discovery for keloid public cohorts.

Writes a frozen candidate registry under data/raw/ for auditability. Does not
download expression matrices and never touches reserved lockbox expression.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data/raw/public_keloid_candidate_registry.json"
REGISTRY = PROJECT_ROOT / "data/raw/public_keloid_cohorts.json"

# Bounded Entrez esearch query; results are metadata-only candidates.
ENTREZ_SEARCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    "?db=gds&term=keloid[All+Fields]+AND+gse[Entry+Type]&retmax=200&retmode=json"
)
ENTREZ_SUMMARY = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    "?db=gds&retmode=json&id={ids}"
)

KELOID_RE = re.compile(r"\bkeloid", re.I)
SCAR_RE = re.compile(r"\b(hypertrophic|normotrophic|immature)\s*scar|\bscar\b", re.I)
EXCLUDE_RE = re.compile(r"\b(miRNA|microRNA|ChIP|ATAC|methylation|methylome)\b", re.I)


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_known() -> dict[str, str]:
    known: dict[str, str] = {}
    if REGISTRY.exists():
        reg = json.loads(REGISTRY.read_text())
        for bucket in ("development", "lockbox", "interpretation_only", "skipped_recovery"):
            for entry in reg.get(bucket, []):
                acc = entry.get("accession")
                if acc:
                    known[acc] = entry.get("role", bucket)
        for acc in reg.get("original_10_comparator", []):
            known.setdefault(acc, "original_10")
    return known


def summarize_candidates(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    # Batch in chunks of 40.
    out = []
    for i in range(0, len(ids), 40):
        chunk = ids[i : i + 40]
        payload = _fetch_json(ENTREZ_SUMMARY.format(ids=",".join(chunk)))
        result = payload.get("result", {})
        for uid in chunk:
            row = result.get(uid)
            if not isinstance(row, dict):
                continue
            accession = str(row.get("accession", "")).strip()
            if not accession.startswith("GSE"):
                continue
            title = str(row.get("title", ""))
            summary = str(row.get("summary", ""))
            text = f"{title} {summary}"
            if not KELOID_RE.search(text):
                continue
            if EXCLUDE_RE.search(text) and not re.search(r"\bRNA-?seq|expression|transcriptom", text, re.I):
                continue
            n_samples = row.get("n_samples") or row.get("samples")
            try:
                n_samples = int(n_samples)
            except (TypeError, ValueError):
                n_samples = None
            out.append(
                {
                    "accession": accession,
                    "gds_id": uid,
                    "title": title,
                    "n_samples": n_samples,
                    "taxon": row.get("taxon"),
                    "gdstype": row.get("gdstype"),
                    "pdat": row.get("pdat"),
                    "scar_mention": bool(SCAR_RE.search(text)),
                    "summary_snippet": summary[:400],
                }
            )
    # Deduplicate by accession (keep first).
    seen = set()
    deduped = []
    for row in out:
        if row["accession"] in seen:
            continue
        seen.add(row["accession"])
        deduped.append(row)
    return sorted(deduped, key=lambda r: r["accession"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--offline-fallback", action="store_true", help="Write registry stub without network.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    known = load_known()
    candidates: list[dict] = []
    error = None
    if not args.offline_fallback:
        try:
            search = _fetch_json(ENTREZ_SEARCH)
            ids = search.get("esearchresult", {}).get("idlist", [])
            candidates = summarize_candidates(ids)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            args.offline_fallback = True

    if args.offline_fallback and not candidates:
        # Deterministic seed list from the accuracy ladder plan when Entrez is unavailable.
        candidates = [
            {
                "accession": acc,
                "title": title,
                "n_samples": n,
                "scar_mention": scar,
                "summary_snippet": note,
                "source": "plan_seed",
            }
            for acc, title, n, scar, note in [
                ("GSE125022", "Keloid bulk RNA/ATAC derivatives", None, False, "RAW is ATAC-only; RNA needs SRA."),
                ("GSE173900", "Asian keloid vs control tissue RNA-seq", 9, False, "Development priority."),
                ("GSE190626", "Keloid vs normal skin TPM", 6, False, "Development priority."),
                ("GSE212954", "Keloid center/margin vs normal", 11, False, "Reserved lockbox."),
                ("GSE121618", "KEC vs NEC microarray", 11, False, "Label repair cohort."),
            ]
        ]

    for row in candidates:
        role = known.get(row["accession"])
        row["known_role"] = role
        row["novel_candidate"] = role is None

    # Deduplicate shared BioProject / publication when fields exist.
    registry = {
        "protocol": "accuracy_evidence_ladder_discovery_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": "keloid[All Fields] AND gse[Entry Type]",
        "n_candidates": len(candidates),
        "n_novel": sum(1 for c in candidates if c.get("novel_candidate")),
        "error": error,
        "candidates": candidates,
        "dedup_rules": [
            "Prefer unique GSE accession.",
            "Collapse shared BioProject/publication/donors when annotated in GEO summary.",
            "Do not promote single-donor or intervention-only studies to primary LOSO.",
        ],
        "reserved_lockbox": ["GSE212954"],
        "skipped_recovery": ["GSE125022"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2))
    print(json.dumps({"out": str(args.out), "n_candidates": len(candidates), "error": error}, indent=2))


if __name__ == "__main__":
    main()
