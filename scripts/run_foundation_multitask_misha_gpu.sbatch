#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-foundation-mt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /gpfs/radev/home/wbw7/Desktop/SpheroScar

export WANDB_PROJECT="SpheroScar"
export WANDB_ENTITY="williamwang178243-yale-university"
export WANDB_SILENT=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python

"${PYTHON}" scripts/run_frozen_qwen_adapter_multitask_eval.py \
  --reset-results \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "foundation-multitask-eval" \
  --wandb-run-name "foundation-multitask-${SLURM_JOB_ID}"
