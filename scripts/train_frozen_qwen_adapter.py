#!/usr/bin/env python3
"""Train frozen-Qwen expression adapters across keloid_binary splits.

Trains only the expression prefix adapter and classification head while keeping
Qwen frozen. Supports multi-split evaluation, early stopping, and multiple
encoder families (mlp, tiny_transformer, attentive_pool).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCDEC_SCRIPTS = PROJECT_ROOT.parent / "Encoder_Decoder_LLM/scripts"
sys.path.insert(0, str(ENCDEC_SCRIPTS))

from encoder_decoder_qwen3 import EncoderDecoderQwen3ForCausalLM  # noqa: E402

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

TRAINING_DIR = PROJECT_ROOT / "data/processed/training"
RESULTS_DIR = PROJECT_ROOT / "results/frozen_qwen_adapter"


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    feature_set: str = "shared_genes_plus_modules"
    encoder_type: str = "mlp"
    encoder_layers: int = 2
    encoder_hidden_dim: int = 1024
    transformer_encoder_dim: int = 1024
    transformer_encoder_heads: int = 4
    transformer_gene_chunk_size: int = 256
    num_prefix_tokens: int = 4
    max_epochs: int = 15
    patience: int = 3
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    config_id: str = "default"
    search_phase: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_task_splits(
    training_dir: Path,
    task: str,
    *,
    split_name: str | None = None,
    split_pattern: str | None = None,
    all_task_splits: bool = False,
    split_names: list[str] | None = None,
) -> list[dict]:
    splits = json.loads((training_dir / "splits/splits.json").read_text())
    candidates = [split for split in splits if split["task"] == task]
    if split_name:
        candidates = [split for split in candidates if split["split_name"] == split_name]
    elif split_names:
        wanted = set(split_names)
        candidates = [split for split in candidates if split["split_name"] in wanted]
    elif split_pattern:
        candidates = [
            split for split in candidates if fnmatch.fnmatch(split["split_name"], split_pattern)
        ]
    elif not all_task_splits:
        candidates = [split for split in candidates if split["split_name"].startswith("grouped_")]
    candidates = sorted(candidates, key=lambda split: split["split_name"])
    if not candidates:
        raise ValueError(f"No {task} splits matched the requested filters")
    return candidates


def load_keloid_binary_splits(
    training_dir: Path,
    *,
    split_name: str | None = None,
    split_pattern: str | None = None,
    all_keloid_binary_splits: bool = False,
    split_names: list[str] | None = None,
) -> list[dict]:
    return load_task_splits(
        training_dir,
        "keloid_binary",
        split_name=split_name,
        split_pattern=split_pattern,
        all_task_splits=all_keloid_binary_splits,
        split_names=split_names,
    )


def load_split_data(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    split: dict,
) -> tuple[dict, dict]:
    manifest = manifest.set_index("sample_id")
    features = features.set_index("sample_id")
    task = str(split["task"])
    ids = {
        "train": [sample_id for sample_id in split["train_sample_ids"] if sample_id in features.index],
        "val": [sample_id for sample_id in split["val_sample_ids"] if sample_id in features.index],
        "test": [sample_id for sample_id in split["test_sample_ids"] if sample_id in features.index],
    }
    if task == "keloid_binary":
        target = manifest["keloid_binary"]
        eligible = target.isin(["keloid", "non_keloid"])
    else:
        target = manifest["task_target"]
        eligible = manifest["encoder_task"].eq(task) & ~target.isin(["unknown", "exclude", "nan", ""])
    for name, sample_ids in ids.items():
        ids[name] = [sample_id for sample_id in sample_ids if eligible.get(sample_id, False)]

    if len(ids["train"]) < 4 or len(ids["test"]) < 2:
        raise ValueError(
            f"Split {split['split_name']} has insufficient train/test samples: "
            f"train={len(ids['train'])}, test={len(ids['test'])}"
        )

    train_labels = target.loc[ids["train"]].astype(str)
    label_encoder = LabelEncoder().fit(train_labels)
    scaler = StandardScaler().fit(features.loc[ids["train"]].to_numpy(dtype=np.float32))

    arrays: dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]] = {}
    for name, sample_ids in ids.items():
        if not sample_ids:
            arrays[name] = (
                torch.empty((0, features.shape[1]), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
                [],
            )
            continue
        x = scaler.transform(features.loc[sample_ids].to_numpy(dtype=np.float32))
        y = label_encoder.transform(target.loc[sample_ids].astype(str))
        arrays[name] = (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            sample_ids,
        )

    metadata = {
        "task": task,
        "split_name": split["split_name"],
        "labels": label_encoder.classes_.tolist(),
        "input_dim": int(features.shape[1]),
        "n_train": len(ids["train"]),
        "n_val": len(ids["val"]),
        "n_test": len(ids["test"]),
    }
    return arrays, metadata


def metrics_from_logits(logits: torch.Tensor, y_true: torch.Tensor, labels: list[str]) -> dict:
    if len(y_true) == 0:
        return {
            "accuracy": None,
            "weighted_f1": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "auroc": None,
        }
    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    y_np = y_true.detach().cpu().numpy()
    pred = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y_np, pred)),
        "weighted_f1": float(f1_score(y_np, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_np, pred)),
        "macro_f1": float(f1_score(y_np, pred, average="macro", zero_division=0)),
    }
    if len(labels) == 2 and len(set(y_np.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_np, probs[:, 1]))
    else:
        out["auroc"] = None
    return out


@torch.no_grad()
def evaluate(
    model: EncoderDecoderQwen3ForCausalLM,
    dataset: TensorDataset,
    batch_size: int,
    labels: list[str],
    device: torch.device,
) -> dict:
    if len(dataset) == 0:
        return metrics_from_logits(torch.empty((0, len(labels))), torch.empty((0,)), labels)
    model.eval()
    logits_all = []
    labels_all = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        x = x.to(device)
        logits = model.classify(x)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())
    return metrics_from_logits(torch.cat(logits_all), torch.cat(labels_all), labels)


def reset_trainable_weights(model: EncoderDecoderQwen3ForCausalLM) -> None:
    """Reinitialize adapter/classification weights between independent split fits."""
    for module in [model.mlp_encoder, model.classification_head]:
        if module is None:
            continue
        for submodule in module.modules():
            if hasattr(submodule, "reset_parameters"):
                submodule.reset_parameters()
    for name, param in model.named_parameters():
        if not name.startswith("llm.") and "context_pool_queries" in name:
            init_std = getattr(model.config, "initializer_range", 0.02)
            torch.nn.init.normal_(param, mean=0.0, std=init_std)


def build_model(config: TrainConfig, input_dim: int, num_classes: int, device: torch.device):
    kwargs = {
        "model_name_or_path": config.model_name,
        "encoder_input_dim": input_dim,
        "num_prefix_tokens": config.num_prefix_tokens,
        "encoder_num_layers": config.encoder_layers,
        "encoder_dropout": config.dropout,
        "encoder_type": config.encoder_type,
        "freeze_llm": True,
        "num_classes": num_classes,
        "transformer_encoder_heads": config.transformer_encoder_heads,
        "transformer_gene_chunk_size": config.transformer_gene_chunk_size,
    }
    if config.encoder_type == "mlp":
        kwargs["encoder_intermediate_dim"] = config.encoder_hidden_dim
    else:
        kwargs["transformer_encoder_dim"] = config.transformer_encoder_dim
    return EncoderDecoderQwen3ForCausalLM(**kwargs).to(device)


def train_one_split(
    model: EncoderDecoderQwen3ForCausalLM,
    arrays: dict,
    metadata: dict,
    config: TrainConfig,
    device: torch.device,
    wandb_run=None,
) -> dict:
    train_x, train_y, _ = arrays["train"]
    val_x, val_y, _ = arrays["val"]
    test_x, test_y, test_ids = arrays["test"]
    labels = metadata["labels"]

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y) if len(val_x) else None
    test_ds = TensorDataset(test_x, test_y)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / config.batch_size))

    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(x)
            loss = F.cross_entropy(logits.float(), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(
            model,
            val_ds or train_ds,
            config.batch_size,
            labels,
            device,
        )
        score = float(val_metrics["weighted_f1"] or -1.0)
        if score > best_val:
            best_val = score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
                if not key.startswith("llm.")
            }
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log({f"{metadata['split_name']}/{key}": value for key, value in row.items()})

        if epochs_without_improvement >= config.patience:
            stopped_early = True
            break

    if best_state is not None:
        current = model.state_dict()
        current.update({key: value.to(device) for key, value in best_state.items()})
        model.load_state_dict(current, strict=False)

    val_metrics = evaluate(model, val_ds or train_ds, config.batch_size, labels, device)
    test_metrics = evaluate(model, test_ds, config.batch_size, labels, device)
    return {
        "split_name": metadata["split_name"],
        "labels": labels,
        "n_train": metadata["n_train"],
        "n_val": metadata["n_val"],
        "n_test": metadata["n_test"],
        "test_sample_ids": test_ids,
        "best_epoch": best_epoch,
        "best_val_weighted_f1": best_val,
        "stopped_early": stopped_early,
        "n_steps": steps_per_epoch * best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "best_state": best_state,
    }


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def init_wandb(args: argparse.Namespace, config: TrainConfig, split_names: list[str]):
    if not args.wandb or wandb is None:
        return None
    os.environ.setdefault("WANDB_SILENT", "true")
    run_config = config.to_dict()
    run_config["split_names"] = split_names
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        group=args.wandb_group,
        job_type="frozen_qwen_adapter",
        config=run_config,
    )


def run_training(args: argparse.Namespace, config: TrainConfig | None = None) -> list[dict]:
    config = config or TrainConfig(
        model_name=args.model_name,
        feature_set=args.feature_set,
        encoder_type=args.encoder_type,
        encoder_layers=args.encoder_layers,
        encoder_hidden_dim=args.encoder_hidden_dim,
        transformer_encoder_dim=args.transformer_encoder_dim,
        transformer_encoder_heads=args.transformer_encoder_heads,
        transformer_gene_chunk_size=args.transformer_gene_chunk_size,
        num_prefix_tokens=args.num_prefix_tokens,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        config_id=args.config_id,
        search_phase=args.search_phase,
    )

    split_names = None
    if args.split_names:
        split_names = [name.strip() for name in args.split_names.split(",") if name.strip()]

    task = getattr(args, "task", None) or "keloid_binary"
    all_task_splits = args.all_task_splits or (
        task == "keloid_binary" and args.all_keloid_binary_splits
    )
    splits = load_task_splits(
        args.training_dir,
        task,
        split_name=args.split_name,
        split_pattern=args.split_pattern,
        all_task_splits=all_task_splits,
        split_names=split_names,
    )
    manifest = pd.read_parquet(args.training_dir / "profile_manifest.parquet")
    features = pd.read_parquet(args.training_dir / f"features/{config.feature_set}.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = init_wandb(args, config, [split["split_name"] for split in splits])
    results: list[dict] = []
    shared_model: EncoderDecoderQwen3ForCausalLM | None = None
    input_dim = int(features.set_index("sample_id").shape[1])

    for split in splits:
        try:
            arrays, metadata = load_split_data(manifest, features, split)
        except ValueError as exc:
            results.append(
                {
                    "task": split["task"],
                    "config_id": config.config_id,
                    "search_phase": config.search_phase,
                    "split_name": split["split_name"],
                    "error": str(exc),
                    **config.to_dict(),
                }
            )
            continue

        if shared_model is None:
            shared_model = build_model(config, input_dim, len(metadata["labels"]), device)
        elif (
            shared_model.encoder_input_dim != input_dim
            or shared_model.classification_head is None
            or shared_model.classification_head.out_features != len(metadata["labels"])
        ):
            shared_model = build_model(config, input_dim, len(metadata["labels"]), device)
        else:
            reset_trainable_weights(shared_model)

        split_result = train_one_split(shared_model, arrays, metadata, config, device, wandb_run)
        row = {
            "stage": "frozen_qwen_adapter",
            "task": split["task"],
            "config_id": config.config_id,
            "search_phase": config.search_phase,
            **config.to_dict(),
            **{k: v for k, v in split_result.items() if k not in {"history", "best_state"}},
        }
        results.append(row)

        if args.save_checkpoint and split_result.get("best_state"):
            ckpt_dir = args.out_dir / "checkpoints" / config.config_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(split_result["best_state"], ckpt_dir / f"{split['split_name']}.pt")

    if args.results_jsonl:
        append_jsonl(args.results_jsonl, results)

    if wandb_run is not None:
        valid = [row for row in results if "error" not in row]
        if valid:
            for metric in ["accuracy", "weighted_f1", "balanced_accuracy", "macro_f1", "auroc"]:
                values = [
                    row["test_metrics"][metric]
                    for row in valid
                    if row.get("test_metrics", {}).get(metric) is not None
                ]
                if values:
                    wandb_run.summary[f"test_{metric}_mean"] = float(np.mean(values))
            wandb_run.summary["n_splits"] = len(valid)
            wandb_run.summary["mean_best_epoch"] = float(np.mean([row["best_epoch"] for row in valid]))
        wandb_run.finish()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if len(splits) == 1 and results and "error" not in results[0]:
        (args.out_dir / "frozen_qwen_adapter_report.json").write_text(
            json.dumps(results[0], indent=2, default=str)
        )

    print(json.dumps(results, indent=2, default=str))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--results-jsonl", type=Path, default=RESULTS_DIR / "adapter_results.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--feature-set", default="shared_genes_plus_modules")
    parser.add_argument("--task", default="keloid_binary")
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--split-names", default=None, help="Comma-separated split names")
    parser.add_argument("--split-pattern", default=None)
    parser.add_argument("--all-task-splits", action="store_true")
    parser.add_argument("--all-keloid-binary-splits", action="store_true")
    parser.add_argument("--encoder-type", choices=["mlp", "tiny_transformer", "attentive_pool"], default="mlp")
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-hidden-dim", type=int, default=1024)
    parser.add_argument("--transformer-encoder-dim", type=int, default=1024)
    parser.add_argument("--transformer-encoder-heads", type=int, default=4)
    parser.add_argument("--transformer-gene-chunk-size", type=int, default=256)
    parser.add_argument("--num-prefix-tokens", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--config-id", default="default")
    parser.add_argument("--search-phase", default="manual")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="SpheroScar")
    parser.add_argument("--wandb-entity", default="williamwang178243-yale-university")
    parser.add_argument("--wandb-run-name", default="frozen-qwen-adapter")
    parser.add_argument("--wandb-group", default=None)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
