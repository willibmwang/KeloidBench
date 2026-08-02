# KeloidBench interactive demo

[![Open the demo in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/willibmwang/KeloidBench/tree/keloidbench-demo?quickstart=1)

The badge runs this interface entirely through GitHub. After selecting
**Create codespace**, dependency installation and Streamlit startup are
automatic; GitHub opens the forwarded demo in a browser tab.

This Streamlit application demonstrates five connected parts of KeloidBench:

1. expression input, program coverage, and transferability checks;
2. the biological program-score feature table;
3. live scoring, linear contributions, and confidence-based abstention;
4. evidence-grounded narrative generation; and
5. study-held-out predictions, including the known non-transferable cohort.

The live scorer uses the exported Stage-B joblib artifact in
`results/product_friday/`. The later 0.918 transferable-cohort result is shown
from stored nested LOSO predictions. The app labels these as distinct evidence
objects rather than attributing the later metric to the older frozen artifact.

## Run

```bash
KB_PY=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
"$KB_PY" -m pip install -r requirements.txt
"$KB_PY" demo/launch_demo.py
```

The launcher highlights the curated stories in the terminal before opening the
website. Start directly on a particular behavior with:

```bash
"$KB_PY" demo/launch_demo.py --example confident_keloid
"$KB_PY" demo/launch_demo.py --example confident_unaffected
"$KB_PY" demo/launch_demo.py --example abstention
"$KB_PY" demo/launch_demo.py --example transfer_failure
```

Use `"$KB_PY" demo/launch_demo.py --list-examples` for presenter cues without
starting a server. The original `streamlit run demo/app.py` command remains
supported.

## Deploy to Streamlit Community Cloud

This directory includes a cloud-specific `requirements.txt`, while the root
`.streamlit/config.toml` supplies the public-site theme. In Streamlit Community
Cloud choose:

- repository: the GitHub repository containing this branch;
- branch: `main`;
- entrypoint: `demo/app.py`; and
- app subdomain: `keloidbench` if available.

That produces a shareable `https://keloidbench.streamlit.app`-style URL. A
separately purchased domain such as `keloidbench.ai` can later redirect to that
public URL or front a container deployment that supports custom domains.

The app includes public profiles already available in the repository. It also
accepts numeric CSV expression matrices. By default, rows are samples and
columns are HGNC gene symbols; use the transpose toggle for genes-by-samples
files.

## What the graphics contain

The opening stories are not synthetic mockups. Selecting one drives four linked
views from repository artifacts:

- a PCA map calculated from the real fused program-feature table;
- a molecular fingerprint comparing the selected profile with keloid and
  unaffected medians from the seven-study transferable subset;
- the frozen Stage-B probability, confidence rule, and linear feature
  contributions; and
- stored outer-fold predictions from the later nested LOSO evaluation.

The complete program-score profile can be downloaded as CSV from the Biology
tab. The exact structured evidence shown to the optional narrative model can be
downloaded as JSON from the Copilot tab.

## Public-study provenance

The curated stories retain their accession and link back to both GEO and the
original publication in the Start here tab:

- `GSE181316`: Direder et al., *Schwann cells contribute to keloid formation*,
  Matrix Biology (2022), DOI `10.1016/j.matbio.2022.03.001`;
- `GSE158395`: Wu et al., *RNA Sequencing Keloid Transcriptome Associates
  Keloids With Th2, Th1, Th17/Th22, and JAK3-Skewing*, Frontiers in Immunology
  (2020), DOI `10.3389/fimmu.2020.597741`; and
- `GSE173900`: Lee et al., *WNT5A drives interleukin-6-dependent
  epithelial-mesenchymal transition via the JAK/STAT pathway in keloid
  pathogenesis*, Burns & Trauma (2022), PMID `36225328`.

## Optional grounded language model

The demo always works without an external model by returning a deterministic,
evidence-faithful summary. To enable the constrained generative layer, point it
at an OpenAI-compatible chat-completions endpoint:

```bash
export KELOIDBENCH_LLM_API_URL="https://your-endpoint.example/v1/chat/completions"
export KELOIDBENCH_LLM_MODEL="your-model-name"
export KELOIDBENCH_LLM_API_KEY="..."  # optional for local endpoints
streamlit run demo/app.py
```

Only the derived, structured evidence packet is sent; raw expression and sample
identifiers are never transmitted to the narrative endpoint. The language model
cannot alter the classifier probability, class threshold, confidence, or
abstention result.
