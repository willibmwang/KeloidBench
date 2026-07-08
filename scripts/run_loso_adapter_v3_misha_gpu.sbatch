#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-loso-v3
#SBATCH --partition=gpu
#SBATCH --account=dijk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/frozen_qwen_adapter

export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_SILENT=true

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python

echo "=== SpheroScar adapter v3 LOSO (hybrid + contrastive) ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "gpu:  ${CUDA_VISIBLE_DEVICES:-none}"
echo "start: $(date -Is)"
nvidia-smi -L 2>/dev/null || true

bash scripts/run_loso_adapter_v3.sh

echo "done: $(date -Is)"
