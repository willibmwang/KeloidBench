#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-pub-finish
#SBATCH --partition=week
#SBATCH --account=dijk
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/public_expansion_loso
PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== repair public ingest + lockbox finish ==="
echo "start: $(date -Is)"

"${PYTHON}" scripts/preprocess_public_keloid.py
"${PYTHON}" scripts/build_training_corpus.py

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
    "clean_keloid_binary": summary.get("clean_keloid_binary"),
    "fibroblast_keloid_binary": summary.get("fibroblast_keloid_binary"),
    "n_external_fibrosis_excluded": summary.get("n_external_fibrosis_excluded"),
    "n_lockbox_profiles": summary.get("n_lockbox_profiles"),
    "feature_sets": summary.get("feature_sets"),
    "coverage_failures": summary.get("coverage_failures"),
    "public_accessions": sorted(manifest.loc[manifest.processed_modality.eq("public_keloid"), "accession"].unique().tolist()),
    "roles": json.loads(Path("data/processed/public_keloid/public_keloid_cohort_roles.json").read_text()) if Path("data/processed/public_keloid/public_keloid_cohort_roles.json").exists() else {},
}
Path("results/public_expansion_loso/corpus_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
lb = manifest[manifest["lockbox"].astype(bool)]
print("lockbox label counts:\n", lb.groupby(["accession","keloid_binary"]).size())
PY

# Re-run nested only if lockbox/binary composition changed enough to need new splits.
"${PYTHON}" scripts/train_nested_loso.py \
  --endpoint all \
  --eval-mode all_modes \
  --out-dir results/public_expansion_loso

"${PYTHON}" scripts/evaluate_lockbox_frozen.py \
  --results-dir results/public_expansion_loso \
  --training-dir data/processed/training

"${PYTHON}" scripts/update_manuscript_public_expansion.py
echo "done: $(date -Is)"
