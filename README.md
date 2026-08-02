# KeloidBench GitHub Pages site

This directory is a static, browser-native export of the KeloidBench
demonstration. It contains no backend service and makes no inference API calls.

The interactive public-profile explorer is driven by
`assets/data/demo-data.json`, which is exported from versioned repository
artifacts with:

```bash
python3 scripts/export_static_site_data.py
```

The export contains the eligible public profiles, program scores, frozen
Stage-B outputs, PCA coordinates, and stored nested LOSO predictions. It does
not contain raw expression matrices.

For local preview:

```bash
python3 -m http.server 8765 --directory website
```

The `gh-pages` branch contains the contents of this directory at its root and
is intended to be served at <https://willibmwang.github.io/KeloidBench/>.
