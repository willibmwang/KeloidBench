#!/usr/bin/env python3
"""Backward-compatible wrapper around the multi-split frozen-Qwen adapter trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_frozen_qwen_adapter import RESULTS_DIR, TRAINING_DIR, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--feature-set", default="shared_genes_plus_modules")
    parser.add_argument("--split-name", default="grouped_seed0_fold0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-prefix-tokens", type=int, default=4)
    parser.add_argument("--encoder-hidden-dim", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="SpheroScar")
    parser.add_argument("--wandb-entity", default="williamwang178243-yale-university")
    parser.add_argument("--wandb-run-name", default="frozen-qwen-adapter-mvp-267-profiles")
    args = parser.parse_args()
    args.encoder_type = "mlp"
    args.max_epochs = args.epochs
    args.patience = 3
    args.transformer_encoder_dim = 1024
    args.transformer_encoder_heads = 4
    args.transformer_gene_chunk_size = 256
    args.config_id = "mvp_wrapper"
    args.search_phase = "mvp"
    args.results_jsonl = None
    args.split_pattern = None
    args.split_names = None
    args.all_keloid_binary_splits = False
    args.save_checkpoint = True
    args.wandb_group = None
    return args


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
