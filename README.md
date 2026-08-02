# KeloidBench GitHub Pages site

This directory is a static, browser-native export of the KeloidBench
demonstration. It contains no backend service and makes no inference API calls.
Processed expression matrices uploaded through the page are parsed and scored
locally in the visitor's browser; file contents are not transmitted.

The interactive public-profile explorer is driven by
`assets/data/demo-data.json`, which is exported from versioned repository
artifacts with:

```bash
python3 scripts/export_static_site_data.py
```

The export contains the eligible public profiles, program scores, frozen
Stage-B outputs, PCA coordinates, stored nested LOSO predictions, versioned
program definitions, and the frozen linear parameters needed for local browser
inference. It does not contain raw expression matrices.

The uploader accepts CSV or TSV matrices with samples in rows or genes in rows.
Inputs must be processed numeric expression with HGNC gene symbols; raw FASTQ,
CEL, and probe-ID-only matrices require preprocessing before upload. Up to 200
samples or 30 MB can be analyzed per browser session. Uploaded cohorts are
always marked transfer-unverified, and incomplete required-program coverage
forces an abstention.

For local preview:

```bash
python3 -m http.server 8765 --directory website
```

The `gh-pages` branch contains the contents of this directory at its root and
is intended to be served at <https://willibmwang.github.io/KeloidBench/>.
