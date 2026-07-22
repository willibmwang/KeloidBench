#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-acc-ladder
#SBATCH --partition=week
#SBATCH --account=dijk
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/accuracy_ladder results/accuracy_ladder_baseline

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== SpheroScar accuracy evidence ladder ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "start: $(date -Is)"

# Preserve immutable baseline comparator if not already present.
if [[ ! -f results/accuracy_ladder_baseline/public_expansion_loso_report.json ]]; then
  if [[ -f results/public_expansion_loso/public_expansion_loso_report.json ]]; then
    cp -n results/public_expansion_loso/public_expansion_loso_report.json results/accuracy_ladder_baseline/ || true
    cp -n results/public_expansion_loso/public_expansion_nested_selected.csv results/accuracy_ladder_baseline/ || true
    cp -n results/public_expansion_loso/lockbox_frozen_once.json results/accuracy_ladder_baseline/ || true
    cp -n results/public_expansion_loso/corpus_audit.json results/accuracy_ladder_baseline/ || true
  fi
fi

echo "--- unit tests ---"
"${PYTHON}" -m unittest tests.test_recovery_correctness -v

echo "--- Stage A0: reproducible GEO discovery (metadata only) ---"
"${PYTHON}" scripts/discover_public_keloid_cohorts.py \
  --out data/raw/public_keloid_candidate_registry.json \
  || "${PYTHON}" scripts/discover_public_keloid_cohorts.py --offline-fallback \
    --out data/raw/public_keloid_candidate_registry.json

echo "--- Stage A: label/provenance corrections (bulk + public) ---"
"${PYTHON}" scripts/preprocess_bulk_rnaseq.py
"${PYTHON}" scripts/preprocess_public_keloid.py \
  --roles development development_optional lockbox

echo "--- Stage B: rebuild corpus with clinical endpoints ---"
"${PYTHON}" scripts/preprocess_scrna_spatial.py || true
"${PYTHON}" scripts/build_training_corpus.py

echo "--- corpus/metadata audit ---"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import pandas as pd
manifest = pd.read_parquet("data/processed/training/profile_manifest.parquet")
summary = json.loads(Path("data/processed/training/training_corpus_summary.json").read_text())
audit = {
    "n_profiles": int(len(manifest)),
    "keloid_binary": summary.get("keloid_binary"),
    "keloid_binary_all_profiles": summary.get("keloid_binary_all_profiles"),
    "keloid_vs_normal_scar": summary.get("keloid_vs_normal_scar"),
    "keloid_vs_pathologic_scar": summary.get("keloid_vs_pathologic_scar"),
    "keloid_vs_unaffected_skin": summary.get("keloid_vs_unaffected_skin"),
    "clean_keloid_binary": summary.get("clean_keloid_binary"),
    "fibroblast_keloid_binary": summary.get("fibroblast_keloid_binary"),
    "n_lockbox_profiles": summary.get("n_lockbox_profiles"),
    "feature_sets": summary.get("feature_sets"),
    "coverage_failures": summary.get("coverage_failures"),
    "public_accessions": sorted(
        manifest.loc[manifest.processed_modality.eq("public_keloid"), "accession"].unique().tolist()
    ),
    "gse212954_in_manifest": bool((manifest["accession"] == "GSE212954").any()),
    "gse173900_in_manifest": bool((manifest["accession"] == "GSE173900").any()),
    "gse190626_in_manifest": bool((manifest["accession"] == "GSE190626").any()),
}
Path("results/accuracy_ladder/corpus_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
assert not summary.get("coverage_failures"), summary.get("coverage_failures")
# Prospective lockbox expression must not enter development selection.
assert not audit["gse212954_in_manifest"], "GSE212954 expression leaked into corpus before freeze"
assert audit["gse173900_in_manifest"] and audit["gse190626_in_manifest"]
PY

echo "--- Stage C/D: nested LOSO with scar panel + equal-accession selection ---"
"${PYTHON}" scripts/train_nested_loso.py \
  --endpoint all \
  --eval-mode all_modes \
  --out-dir results/accuracy_ladder

echo "--- Freeze protocol before touching GSE212954 expression ---"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import pandas as pd
from collections import Counter
root = Path(".")
results = root / "results/accuracy_ladder"
selected = pd.read_csv(results / "public_expansion_nested_selected.csv")
cur = selected[selected.eval_mode.eq("current10") & selected.endpoint.eq("keloid_binary")]
keys = list(zip(cur["feature_set"], cur["model"], cur["weight_mode"])) if len(cur) else []
if keys:
    (feature_set, model, weight_mode), _ = Counter(keys).most_common(1)[0]
    thr = float(cur["threshold"].median())
else:
    feature_set, model, weight_mode, thr = "scar_discriminative_rank_programs", "ridge_logreg", "none", 0.5
protocol = {
    "protocol": "accuracy_evidence_ladder_v1",
    "frozen_at_stage": "before_GSE212954_expression_download",
    "endpoints": [
        "keloid_binary",
        "keloid_binary_all_profiles",
        "keloid_vs_normal_scar",
        "keloid_vs_pathologic_scar",
        "keloid_vs_unaffected_skin",
        "clean_keloid_binary",
        "fibroblast_keloid_binary",
    ],
    "selection_rule": "equal_accession_mean_inner_f1_with_lower_tail_tiebreak",
    "frozen_config": {
        "feature_set": feature_set,
        "model": model,
        "weight_mode": weight_mode,
        "threshold": thr,
    },
    "reserved_lockbox": ["GSE212954"],
    "historical_lockbox_comparators": ["GSE237752", "GSE191067"],
}
results.mkdir(parents=True, exist_ok=True)
(results / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2))
print(json.dumps(protocol, indent=2))
PY

echo "--- Download/score prospective lockbox GSE212954 once ---"
"${PYTHON}" scripts/evaluate_lockbox_frozen.py \
  --results-dir results/accuracy_ladder \
  --training-dir data/processed/training \
  --download-reserved-lockbox \
  --accessions GSE212954

echo "--- update manuscript ---"
"${PYTHON}" scripts/update_manuscript_public_expansion.py \
  --results-dir results/accuracy_ladder || \
"${PYTHON}" scripts/update_manuscript_public_expansion.py

echo "done: $(date -Is)"
