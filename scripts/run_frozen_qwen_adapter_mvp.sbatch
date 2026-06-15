#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-qwen-adapter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /gpfs/radev/home/wbw7/Desktop/SpheroScar

export WANDB_PROJECT="SpheroScar"
export WANDB_ENTITY="williamwang178243-yale-university"
export WANDB_SILENT=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/train_frozen_qwen_adapter_mvp.py \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-run-name "frozen-qwen-adapter-mvp-${SLURM_JOB_ID}" \
  --epochs 20 \
  --batch-size 8
