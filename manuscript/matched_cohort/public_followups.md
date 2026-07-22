# Public-data follow-ups (secondary to matched cohort)

Optional while the matched cohort is enrolled. Do not reopen architecture chase or retune on v1 lockbox GSE185309.

## 1. GSE218007 (v2 fibroblast lockbox) — DONE once
- Ingested via Affymetrix RMA-GENE **series matrix** (no CEL/oligo required).
- Scored once under Stage-B `fibroblast_keloid_binary`: macro F1≈0.33, donor AUROC inverted.
- Artifacts: `results/breakthrough_sprint_v2/lockbox_gse218007/` (`scored_once.json`).
- Interpretation: culture/regional keloid fibroblast domain does not transfer from public fibrosis programs; reinforces matched tissue cohort.

## 2. Scar-comparator GEO — partial
- **GSE178562** human non-KO scar triad (HTS/keloid/NS/scar) ingested → specialists refreshed.
  - Pathologic-scar nested macro lifted ~0.16→**0.42** (worst 0→0.23).
  - Normal-scar nested macro dropped ~0.57→**0.33** (new hard fold).
- **GSE307504** still deferred (scRNA Seurat-only, no bulk matrix).
- **GSE181540** Excel malformed / not ingestible as-is.

## 3. Explicit non-goals
- No more indiscriminate GEO keloid cell-line expansion
- No rescoring of GSE185309 or GSE218007
- No mixing fibroblast cultures into tissue primary endpoint
