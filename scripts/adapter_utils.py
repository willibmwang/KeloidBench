"""Shared helpers for SpheroScar expression adapter training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SMALL_LLM = "Qwen/Qwen2.5-0.5B"


def resolve_encdec_scripts() -> Path:
    """Return path to encoder_decoder_qwen3.py (TLM preferred, legacy fallback)."""
    candidates = [
        PROJECT_ROOT.parent / "TLM/scripts",
        PROJECT_ROOT.parent / "Encoder_Decoder_LLM/scripts",
    ]
    for path in candidates:
        if (path / "encoder_decoder_qwen3.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise FileNotFoundError(
        "Could not find encoder_decoder_qwen3.py in TLM/scripts or Encoder_Decoder_LLM/scripts"
    )


def suggest_encoder_type(input_dim: int, requested: str) -> str:
    """Low-dimensional module inputs should not use gene-chunk transformers."""
    if input_dim <= 32 and requested in {"tiny_transformer", "attentive_pool"}:
        return "mlp"
    return requested


KELQID_PROMPT = (
    "Given a harmonized skin scar expression profile, classify keloid status. "
    "Answer with exactly one token: keloid or non_keloid."
)


class CohortAugmenter(nn.Module):
    """Concatenate a learned cohort embedding (modality or accession) to each vector."""

    def __init__(self, input_dim: int, n_categories: int, embed_dim: int = 16):
        super().__init__()
        self.base_dim = input_dim
        self.embed = nn.Embedding(n_categories, embed_dim)
        self.output_dim = input_dim + embed_dim

    def forward(self, x: torch.Tensor, cohort_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed(cohort_ids.to(x.device))
        return torch.cat([x, emb.to(dtype=x.dtype)], dim=-1)


ModalityAugmenter = CohortAugmenter


def build_prompt_batch(
    tokenizer,
    batch_size: int,
    device: torch.device,
    prompt: str = KELQID_PROMPT,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def build_lm_batch(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
    *,
    max_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize prompt+response pairs with LM labels masked on prompt tokens."""
    full_texts = [f"{prompt}{response}" for prompt, response in zip(prompts, responses)]
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:] = -100

    for row_idx, (prompt, response) in enumerate(zip(prompts, responses)):
        if not response:
            continue
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        start = len(prompt_ids)
        end = min(start + len(response_ids), input_ids.shape[1])
        if start < end:
            labels[row_idx, start:end] = input_ids[row_idx, start:end]
    return input_ids, attention_mask, labels


def pool_prefix_embeddings(model, x: torch.Tensor) -> torch.Tensor:
    """Mean-pool encoded prefix tokens into one vector per sample."""
    prefix, prefix_attention = model._encode_cell_prefixes(cell_embeddings=x)
    weights = prefix_attention.to(device=prefix.device, dtype=prefix.dtype).unsqueeze(-1)
    return (prefix * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss on pooled prefix embeddings."""
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())

    z = F.normalize(embeddings.float(), dim=1)
    sim = torch.matmul(z, z.T) / temperature
    label_matrix = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    label_matrix.fill_diagonal_(False)

    losses = []
    for row_idx in range(z.shape[0]):
        pos_mask = label_matrix[row_idx]
        if not pos_mask.any():
            continue
        row_logits = sim[row_idx]
        pos_logits = row_logits[pos_mask]
        denom_logits = row_logits[row_idx != torch.arange(z.shape[0], device=z.device)]
        if denom_logits.numel() == 0:
            continue
        pos_log_prob = pos_logits - torch.logsumexp(denom_logits, dim=0)
        losses.append(-pos_log_prob.mean())

    if not losses:
        return embeddings.new_zeros(())
    return torch.stack(losses).mean()
