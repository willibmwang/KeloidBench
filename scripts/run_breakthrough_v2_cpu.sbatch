#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-btv2
#SBATCH --partition=week
#SBATCH --account=dijk
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/breakthrough_sprint_v2/ablations

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== SpheroScar breakthrough v2 lift ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "start: $(date -Is)"

echo "--- unit tests ---"
"${PYTHON}" -m unittest tests.test_recovery_correctness -v

echo "--- audit + freeze v2 (preserve v1) ---"
"${PYTHON}" scripts/audit_cohort_decisions_v2.py
"${PYTHON}" scripts/freeze_breakthrough_protocol_v2.py
test -f results/breakthrough_sprint/frozen_protocol.json
test -f results/breakthrough_sprint/breakthrough_lockbox_once.json
test -f results/breakthrough_sprint_v2/frozen_protocol.json

echo "--- rebuild endpoint-pure corpus ---"
"${PYTHON}" scripts/build_training_corpus.py

echo "--- eligibility after purity ---"
"${PYTHON}" - <<'PY'
import pandas as pd
elig=pd.read_csv('data/processed/training/endpoint_eligibility_matrix.csv')
skin=elig[elig.keloid_vs_unaffected_skin_evaluable]
print('primary skin evaluable', sorted(skin.accession.tolist()))
print(skin[['accession','n_profiles','n_donors','keloid_vs_unaffected_skin_keloid','keloid_vs_unaffected_skin_non_keloid']].to_string(index=False))
# Guard: harmful culture-line accessions must not be primary-skin evaluable
bad={'GSE282479','GSE303591','GSE232079','GSE246562'}
assert not (bad & set(skin.accession)), bad & set(skin.accession)
print('OK: culture/stiffness cohorts excluded from primary skin')
PY

echo "--- staged nested ablations A-D ---"
"${PYTHON}" scripts/train_breakthrough_cascade_v2.py --out-dir results/breakthrough_sprint_v2 --stage all

echo "--- targeted data probe ---"
"${PYTHON}" scripts/targeted_data_v2.py

echo "--- go / no-go ---"
"${PYTHON}" scripts/go_no_go_v2.py

echo "done: $(date -Is)"
