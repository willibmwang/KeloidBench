#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /gpfs/radev/home/wbw7/Desktop/SpheroScar

export WANDB_PROJECT="SpheroScar"
export WANDB_ENTITY="williamwang178243-yale-university"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/train_baselines.py \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-run-name "baseline-mvp-267-profiles-${SLURM_JOB_ID}" \
  --wandb-mode "${WANDB_MODE}"

/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/write_eval_report.py
