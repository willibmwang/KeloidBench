#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-rec-finish
#SBATCH --partition=week
#SBATCH --account=dijk
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/recovery_loso
PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== finish ensemble + report ==="
echo "start: $(date -Is)"

"${PYTHON}" scripts/train_ensemble_loso.py \
  --normalization-mode none \
  --feature-sets modules_only profibrotic_module_only robust_programs_only \
  --out-dir results/recovery_loso/ensemble

"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("results/recovery_loso")
summary = pd.read_csv(root / "recovery_loso_summary.csv")
baseline = pd.read_csv(root / "baselines_raw/baseline_summary.csv")
selected = pd.read_csv(root / "recovery_nested_selected.csv")
coverage = pd.read_csv("data/processed/training/accession_module_coverage.csv")
corpus = json.loads(Path("data/processed/training/training_corpus_summary.json").read_text())
ens_sum_path = root / "ensemble/ensemble_loso_summary.csv"
if not ens_sum_path.exists():
    ens_sum_path = root / "ensemble/ensemble_summary.csv"
ens = pd.read_csv(ens_sum_path) if ens_sum_path.exists() else pd.DataFrame()

def top(df, endpoint=None, n=5):
    out = df
    if endpoint is not None and "endpoint" in out.columns:
        out = out[out.endpoint.eq(endpoint)]
    return out.sort_values("weighted_f1_mean", ascending=False).head(n)

hard = ["GSE44270", "Sun_Burns", "GSE188952", "GSE7890"]
res = pd.read_csv(root / "recovery_loso_results.csv")
hard_rows = []
for stage in ["corrected_fixed", "cohort_balanced", "limited_adaptation"]:
    sub = res[(res.stage == stage) & (res.endpoint == "keloid_binary")]
    for h in hard:
        rows = sub[sub.split_name.str.contains(h, na=False)]
        if rows.empty:
            continue
        best = rows.sort_values("weighted_f1", ascending=False).iloc[0]
        hard_rows.append(
            {
                "stage": stage,
                "accession": h,
                "weighted_f1": float(best.weighted_f1),
                "auroc": float(best.auroc) if pd.notna(best.auroc) else None,
                "feature_set": best.feature_set,
                "model": best.model,
            }
        )

best_broad_fixed = top(summary[summary.stage.eq("corrected_fixed")], "keloid_binary", 1)
best_clean_fixed = top(summary[summary.stage.eq("corrected_fixed")], "clean_keloid_binary", 1)
nested_means = (
    selected.groupby("endpoint")["weighted_f1"].agg(["mean", "median", "count"]).reset_index().to_dict(orient="records")
)

report = {
    "corpus": {
        "n_profiles": corpus.get("n_profiles"),
        "keloid_binary": corpus.get("keloid_binary"),
        "clean_keloid_binary": corpus.get("clean_keloid_binary"),
        "coverage_failures": corpus.get("coverage_failures", []),
    },
    "coverage": coverage.to_dict(orient="records"),
    "primary_broad_corrected_fixed": best_broad_fixed.to_dict(orient="records"),
    "secondary_clean_corrected_fixed": best_clean_fixed.to_dict(orient="records"),
    "best_baselines_raw": top(baseline, n=5).to_dict(orient="records"),
    "nested_selected_means": nested_means,
    "hard_fold_bests": hard_rows,
    "best_ensemble": ens.sort_values("weighted_f1_mean", ascending=False).head(5).to_dict(orient="records") if len(ens) else [],
    "stop_rule": {
        "broad_0_80_claimed": False,
        "reason": (
            "Best frozen outer LOSO corrected_fixed broad weighted F1 is below 0.70; "
            "GSE44270 remains near chance after gene-mapping repair. "
            "Architecture chase stopped per recovery plan."
        ),
    },
}
(root / "recovery_final_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)[:5000])
PY

"${PYTHON}" scripts/update_manuscript_recovery.py
echo "done: $(date -Is)"
