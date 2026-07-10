#!/usr/bin/env python3
"""Staged hyperparameter search for frozen-Qwen keloid_binary adapters."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_frozen_qwen_adapter import RESULTS_DIR, TRAINING_DIR, TrainConfig, run_training

ANCHOR_SPLITS = [
    "grouped_seed0_fold4",
    "grouped_seed0_fold0",
    "grouped_seed0_fold3",
]

PHASE_A_ENCODER_TYPES = ["mlp", "tiny_transformer", "attentive_pool"]
PHASE_A_ENCODER_LAYERS = [2, 3, 4]
PHASE_A_MAX_EPOCHS = [10, 15]

PHASE_B_PREFIX_TOKENS = [2, 4, 8]
PHASE_B_DROPOUT = [0.05, 0.1, 0.2]
PHASE_B_LR = [5e-5, 1e-4]
PHASE_B_TRANSFORMER_DIM = [512, 1024]


def config_id_from_dict(config: dict) -> str:
    parts = [
        config["encoder_type"],
        f"L{config['encoder_layers']}",
        f"E{config['max_epochs']}",
        f"P{config['num_prefix_tokens']}",
        f"D{str(config['dropout']).replace('.', '')}",
        f"LR{str(config['lr']).replace('.', '')}",
    ]
    if config["encoder_type"] != "mlp":
        parts.append(f"TD{config['transformer_encoder_dim']}")
    return "_".join(parts)


def base_config() -> dict:
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        "feature_set": "shared_genes_plus_modules",
        "encoder_type": "mlp",
        "encoder_layers": 2,
        "encoder_hidden_dim": 1024,
        "transformer_encoder_dim": 1024,
        "transformer_encoder_heads": 4,
        "transformer_gene_chunk_size": 256,
        "num_prefix_tokens": 4,
        "max_epochs": 15,
        "patience": 3,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "dropout": 0.1,
    }


def phase_a_configs() -> list[dict]:
    configs = []
    for encoder_type, layers, max_epochs in itertools.product(
        PHASE_A_ENCODER_TYPES,
        PHASE_A_ENCODER_LAYERS,
        PHASE_A_MAX_EPOCHS,
    ):
        cfg = base_config()
        cfg.update(
            {
                "encoder_type": encoder_type,
                "encoder_layers": layers,
                "max_epochs": max_epochs,
                "search_phase": "phase_a",
            }
        )
        cfg["config_id"] = config_id_from_dict(cfg)
        configs.append(cfg)
    return configs


def rank_configs(results_path: Path, phase: str, top_k: int) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame()
    rows = pd.read_json(results_path, lines=True)
    if "error" in rows.columns:
        rows = rows[(rows["search_phase"] == phase) & rows["error"].isna()]
    else:
        rows = rows[rows["search_phase"] == phase]
    if rows.empty:
        return rows
    grouped = (
        rows.groupby("config_id")
        .agg(
            mean_val_weighted_f1=("best_val_weighted_f1", "mean"),
            mean_test_weighted_f1=("test_metrics", lambda s: np.mean([m["weighted_f1"] for m in s])),
            mean_best_epoch=("best_epoch", "mean"),
            n_splits=("split_name", "count"),
        )
        .reset_index()
        .sort_values(["mean_val_weighted_f1", "mean_test_weighted_f1"], ascending=False)
    )
    return grouped.head(top_k)


def phase_b_configs(top_configs: list[dict]) -> list[dict]:
    refined: list[dict] = []
    seen: set[str] = set()
    for base in top_configs:
        for prefix, dropout, lr in itertools.product(PHASE_B_PREFIX_TOKENS, PHASE_B_DROPOUT, PHASE_B_LR):
            cfg = deepcopy(base)
            cfg.update(
                {
                    "num_prefix_tokens": prefix,
                    "dropout": dropout,
                    "lr": lr,
                    "search_phase": "phase_b",
                }
            )
            if cfg["encoder_type"] != "mlp":
                for dim in PHASE_B_TRANSFORMER_DIM:
                    cfg_dim = deepcopy(cfg)
                    cfg_dim["transformer_encoder_dim"] = dim
                    cfg_dim["config_id"] = config_id_from_dict(cfg_dim)
                    if cfg_dim["config_id"] not in seen:
                        seen.add(cfg_dim["config_id"])
                        refined.append(cfg_dim)
            else:
                cfg["config_id"] = config_id_from_dict(cfg)
                if cfg["config_id"] not in seen:
                    seen.add(cfg["config_id"])
                    refined.append(cfg)
    return refined


def make_namespace(
    args: argparse.Namespace,
    config: dict,
    split_names: list[str],
    results_jsonl: Path,
) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.training_dir = args.training_dir
    ns.out_dir = args.out_dir
    ns.model_name = config.get("model_name", "Qwen/Qwen3-1.7B")
    ns.feature_set = config.get("feature_set", "shared_genes_plus_modules")
    ns.encoder_type = config["encoder_type"]
    ns.encoder_layers = config["encoder_layers"]
    ns.encoder_hidden_dim = config["encoder_hidden_dim"]
    ns.transformer_encoder_dim = config["transformer_encoder_dim"]
    ns.transformer_encoder_heads = config["transformer_encoder_heads"]
    ns.transformer_gene_chunk_size = config["transformer_gene_chunk_size"]
    ns.num_prefix_tokens = config["num_prefix_tokens"]
    ns.max_epochs = config["max_epochs"]
    ns.patience = config.get("patience", 3)
    ns.batch_size = config["batch_size"]
    ns.lr = config["lr"]
    ns.weight_decay = config["weight_decay"]
    ns.dropout = config["dropout"]
    ns.config_id = config["config_id"]
    ns.search_phase = config["search_phase"]
    ns.split_names = ",".join(split_names)
    ns.split_name = None
    ns.split_pattern = None
    ns.all_keloid_binary_splits = False
    ns.results_jsonl = results_jsonl
    ns.wandb = args.wandb
    ns.wandb_project = args.wandb_project
    ns.wandb_entity = args.wandb_entity
    ns.wandb_group = args.wandb_group or "adapter-keloid-binary-search"
    ns.wandb_run_name = f"{config['search_phase']}-{config['config_id']}"
    ns.save_checkpoint = config.get("search_phase") == "phase_c"
    return ns


def run_config_batch(args: argparse.Namespace, config: dict, split_names: list[str]) -> None:
    ns = make_namespace(args, config, split_names, args.results_jsonl)
    train_config = TrainConfig(**{k: config[k] for k in TrainConfig.__dataclass_fields__ if k in config})
    run_training(ns, train_config)


def all_keloid_binary_split_names(training_dir: Path) -> list[str]:
    from train_frozen_qwen_adapter import load_keloid_binary_splits

    return [split["split_name"] for split in load_keloid_binary_splits(training_dir, all_keloid_binary_splits=True)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--results-jsonl", type=Path, default=RESULTS_DIR / "adapter_results.jsonl")
    parser.add_argument("--phase", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--top-k-phase-b", type=int, default=3)
    parser.add_argument("--top-k-phase-c", type=int, default=2)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="SpheroScar")
    parser.add_argument("--wandb-entity", default="williamwang178243-yale-university")
    parser.add_argument("--wandb-group", default="adapter-keloid-binary-search")
    parser.add_argument("--reset-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_results and args.results_jsonl.exists():
        args.results_jsonl.unlink()

    search_summary: dict = {"phase_a": [], "phase_b": [], "phase_c": []}

    if args.phase in {"a", "all"}:
        for config in phase_a_configs():
            run_config_batch(args, config, ANCHOR_SPLITS)
        ranked = rank_configs(args.results_jsonl, "phase_a", args.top_k_phase_b)
        search_summary["phase_a"] = ranked.to_dict(orient="records")
        (args.out_dir / "phase_a_rankings.json").write_text(json.dumps(search_summary["phase_a"], indent=2))

    top_for_b = []
    if args.phase in {"b", "all"}:
        ranked = rank_configs(args.results_jsonl, "phase_a", args.top_k_phase_b)
        if ranked.empty:
            raise RuntimeError("Phase B requested but no Phase A rankings were found")
        rows = pd.read_json(args.results_jsonl, lines=True)
        for config_id in ranked["config_id"].tolist():
            sample = rows[rows["config_id"] == config_id].iloc[0].to_dict()
            base = {k: sample[k] for k in base_config().keys() if k in sample}
            base["search_phase"] = "phase_a"
            top_for_b.append(base)
        for config in phase_b_configs(top_for_b):
            run_config_batch(args, config, ANCHOR_SPLITS)
        ranked_b = rank_configs(args.results_jsonl, "phase_b", args.top_k_phase_c)
        search_summary["phase_b"] = ranked_b.to_dict(orient="records")
        (args.out_dir / "phase_b_rankings.json").write_text(json.dumps(search_summary["phase_b"], indent=2))

    if args.phase in {"c", "all"}:
        ranked = rank_configs(args.results_jsonl, "phase_b", args.top_k_phase_c)
        if ranked.empty:
            ranked = rank_configs(args.results_jsonl, "phase_a", args.top_k_phase_c)
        if ranked.empty:
            raise RuntimeError("Phase C requested but no ranked configs were found")
        rows = pd.read_json(args.results_jsonl, lines=True)
        all_splits = all_keloid_binary_split_names(args.training_dir)
        for config_id in ranked["config_id"].tolist():
            sample = rows[rows["config_id"] == config_id].iloc[0].to_dict()
            config = {k: sample[k] for k in base_config().keys() if k in sample}
            config["search_phase"] = "phase_c"
            config["config_id"] = config_id
            run_config_batch(args, config, all_splits)
        search_summary["phase_c"] = ranked.to_dict(orient="records")
        (args.out_dir / "phase_c_winners.json").write_text(json.dumps(search_summary["phase_c"], indent=2))

    summary_rows = []
    if args.results_jsonl.exists():
        df = pd.read_json(args.results_jsonl, lines=True)
        valid = df[df["error"].isna()] if "error" in df.columns else df
        if not valid.empty:
            summary_rows = (
                valid.groupby(["search_phase", "config_id", "encoder_type"])
                .agg(
                    mean_test_weighted_f1=("test_metrics", lambda s: np.mean([m["weighted_f1"] for m in s])),
                    std_test_weighted_f1=("test_metrics", lambda s: np.std([m["weighted_f1"] for m in s])),
                    mean_best_epoch=("best_epoch", "mean"),
                    n_splits=("split_name", "count"),
                )
                .reset_index()
                .sort_values(["search_phase", "mean_test_weighted_f1"], ascending=[True, False])
                .to_dict(orient="records")
            )
    (args.out_dir / "adapter_search_summary.json").write_text(json.dumps(summary_rows, indent=2))
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "adapter_summary.csv", index=False)
    print(json.dumps({"search_summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
