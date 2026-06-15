#!/usr/bin/env python3
"""Build the SpheroScar baseline training corpus.

This script creates:
- a unified profile manifest with canonical targets
- harmonized feature tables for modules, shared genes, and both
- task-specific grouped split manifests
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUT_DIR = PROCESSED_DIR / "training"

MODULE_COLUMNS = [
    "ECM_score",
    "myofibroblast_score",
    "TGFb_score",
    "hypoxia_vascular_score",
    "remodeling_score",
    "profibrotic_fibroblast_score",
    "antifibrotic_fibroblast_score",
    "fibrotic_activity_score",
]

METADATA_COLUMNS = [
    "sample_id",
    "accession",
    "sample_title",
    "platform_id",
    "modality",
    "source_dataset",
    "disease_domain",
    "disease_label",
    "keloid_vs_normal",
    "lesional_status",
    "scar_type",
    "cell_type",
    "treatment",
    "patient_id",
    "contrast_type",
    "eligible_for_keloid_pretraining",
    "encoder_task",
    "encoder_prompt",
    "encoder_response",
]

POSITIVE_LABELS = {"keloid", "lesional"}
NEGATIVE_LABELS = {
    "normal",
    "normal_scar",
    "adjacent_normal",
    "non_lesional",
    "control",
    "hypertrophic_scar",
    "normotrophic_scar",
}
OOD_ACCESSIONS = {"GSE3189", "GSE160536"}


@dataclass(frozen=True)
class ModalityArtifact:
    name: str
    prefix: str
    directory: Path

    @property
    def wide_path(self) -> Path:
        return self.directory / f"{self.prefix}_expression_wide.parquet"

    @property
    def vocab_path(self) -> Path:
        return self.directory / f"{self.prefix}_gene_vocab.txt"


ARTIFACTS = [
    ModalityArtifact("microarray", "microarray", PROCESSED_DIR / "microarray"),
    ModalityArtifact("bulk_rnaseq", "bulk_rnaseq", PROCESSED_DIR / "bulk_rnaseq"),
    ModalityArtifact("scrna_spatial", "scrna_spatial", PROCESSED_DIR / "scrna_spatial"),
]


def read_vocab(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def is_gene_symbol(feature: str) -> bool:
    if feature.startswith("ENSG"):
        return False
    if re.match(r"^GPL\d+_", feature):
        return False
    return bool(re.match(r"^[A-Z][A-Z0-9.-]*$", feature))


def canonical_binary(row: pd.Series) -> str:
    if bool(row.get("is_out_of_domain", False)):
        return "exclude"
    values = {
        str(row.get("disease_label", "")).lower(),
        str(row.get("keloid_vs_normal", "")).lower(),
        str(row.get("scar_type", "")).lower(),
        str(row.get("encoder_response", "")).lower(),
    }
    if values & POSITIVE_LABELS:
        return "keloid"
    if values & NEGATIVE_LABELS:
        return "non_keloid"
    return "exclude"


def task_target(row: pd.Series) -> str:
    task = str(row.get("encoder_task", "unknown"))
    if task == "keloid_vs_normal":
        return row["keloid_binary"]
    if task == "lesional_status":
        value = str(row.get("lesional_status", "unknown"))
        return value if value and value != "unknown" else str(row.get("encoder_response", "unknown"))
    if task in {"cell_type", "fibroblast_state", "fibroblast_subcluster", "celltype_subcluster", "scar_differential"}:
        return str(row.get("encoder_response", "unknown"))
    if task.startswith("out_of_domain"):
        return str(row.get("encoder_response", "unknown"))
    return str(row.get("encoder_response", "unknown"))


def make_group_id(row: pd.Series) -> str:
    patient = str(row.get("patient_id", "unknown")).strip()
    if patient and patient.lower() not in {"unknown", "nan", "none"}:
        return f"{row['modality']}|{row['accession']}|{patient}"
    return f"{row['modality']}|{row['accession']}"


def load_artifacts(max_shared_genes: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[str]]]:
    manifest_parts = []
    expression_parts = {}
    vocabs = {}

    for artifact in ARTIFACTS:
        wide = pd.read_parquet(artifact.wide_path)
        vocab = read_vocab(artifact.vocab_path)
        metadata_cols = [col for col in METADATA_COLUMNS + MODULE_COLUMNS if col in wide.columns]
        metadata = wide[metadata_cols].copy()
        metadata["processed_modality"] = artifact.name
        metadata["artifact_prefix"] = artifact.prefix
        metadata["artifact_wide_path"] = str(artifact.wide_path.relative_to(PROJECT_ROOT))
        metadata["artifact_vocab_path"] = str(artifact.vocab_path.relative_to(PROJECT_ROOT))
        expr_cols = [col for col in vocab if col in wide.columns]
        expression_parts[artifact.name] = wide[["sample_id", *expr_cols]].set_index("sample_id")
        vocabs[artifact.name] = expr_cols
        manifest_parts.append(metadata)

    manifest = pd.concat(manifest_parts, ignore_index=True, sort=False)
    manifest["profile_index"] = np.arange(len(manifest))
    manifest["is_out_of_domain"] = (
        manifest["accession"].isin(OOD_ACCESSIONS)
        | manifest["encoder_task"].astype(str).str.startswith("out_of_domain")
        | manifest["disease_domain"].astype(str).str.lower().isin({"melanoma", "scleroderma"})
    )
    manifest["keloid_binary"] = manifest.apply(canonical_binary, axis=1)
    manifest["task_target"] = manifest.apply(task_target, axis=1)
    manifest["split_group"] = manifest.apply(make_group_id, axis=1)
    manifest["source_group"] = manifest["split_group"]

    symbol_presence: dict[str, set[str]] = {}
    for modality, genes in vocabs.items():
        for gene in genes:
            if is_gene_symbol(gene):
                symbol_presence.setdefault(gene, set()).add(modality)
    shared_genes = sorted([gene for gene, mods in symbol_presence.items() if len(mods) >= 2])
    if len(shared_genes) > max_shared_genes:
        variance_scores = []
        for gene in shared_genes:
            vals = []
            for expr in expression_parts.values():
                if gene in expr.columns:
                    vals.append(expr[gene])
            score = pd.concat(vals).var(skipna=True) if vals else 0.0
            variance_scores.append((gene, float(score)))
        shared_genes = [gene for gene, _ in sorted(variance_scores, key=lambda item: item[1], reverse=True)[:max_shared_genes]]
        shared_genes = sorted(shared_genes)

    feature_tables = build_feature_tables(manifest, expression_parts, shared_genes)
    vocab_report = {
        "shared_genes": shared_genes,
        "modality_gene_counts": {name: len(genes) for name, genes in vocabs.items()},
        "n_shared_genes": len(shared_genes),
    }
    return manifest, feature_tables, vocab_report


def build_feature_tables(
    manifest: pd.DataFrame,
    expression_parts: dict[str, pd.DataFrame],
    shared_genes: list[str],
) -> dict[str, pd.DataFrame]:
    modules = manifest[["sample_id", *[col for col in MODULE_COLUMNS if col in manifest.columns]]].copy()
    for col in MODULE_COLUMNS:
        if col not in modules.columns:
            modules[col] = 0.0
    modules[MODULE_COLUMNS] = modules[MODULE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    shared_rows = []
    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        modality = row["processed_modality"]
        expr = expression_parts[modality]
        values = expr.reindex(index=[sample_id], columns=shared_genes).fillna(0.0)
        values.insert(0, "sample_id", sample_id)
        shared_rows.append(values.reset_index(drop=True))
    shared = pd.concat(shared_rows, ignore_index=True, sort=False)
    shared[shared_genes] = shared[shared_genes].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    shared_plus_modules = shared.merge(modules, on="sample_id", how="left")
    return {
        "modules_only": modules,
        "shared_genes": shared,
        "shared_genes_plus_modules": shared_plus_modules,
    }


def valid_task_rows(manifest: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "keloid_binary":
        rows = manifest[manifest["keloid_binary"].isin(["keloid", "non_keloid"])].copy()
        rows = rows[~rows["is_out_of_domain"]]
        rows["target"] = rows["keloid_binary"]
        return rows
    rows = manifest[manifest["encoder_task"].eq(task)].copy()
    if task != "out_of_domain_disease_state":
        rows = rows[~rows["is_out_of_domain"]]
    rows["target"] = rows["task_target"]
    rows = rows[~rows["target"].isin(["unknown", "exclude", "nan", ""])]
    return rows


def add_split(splits: list[dict], name: str, task: str, rows: pd.DataFrame, train_idx, val_idx, test_idx) -> None:
    train = rows.iloc[list(train_idx)]
    val = rows.iloc[list(val_idx)]
    test = rows.iloc[list(test_idx)]
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        return
    splits.append(
        {
            "split_name": name,
            "task": task,
            "train_sample_ids": train["sample_id"].tolist(),
            "val_sample_ids": val["sample_id"].tolist(),
            "test_sample_ids": test["sample_id"].tolist(),
            "train_label_counts": train["target"].value_counts().to_dict(),
            "val_label_counts": val["target"].value_counts().to_dict(),
            "test_label_counts": test["target"].value_counts().to_dict(),
            "grouping": "split_group",
        }
    )


def build_splits(manifest: pd.DataFrame, tasks: list[str], n_repeats: int) -> list[dict]:
    splits: list[dict] = []
    for task in tasks:
        rows = valid_task_rows(manifest, task).reset_index(drop=True)
        if len(rows) < 6 or rows["target"].nunique() < 2:
            continue
        n_groups = rows["split_group"].nunique()
        n_splits = min(5, n_groups)
        if n_splits >= 3:
            for seed in range(n_repeats):
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                folds = list(cv.split(rows, rows["target"], rows["split_group"]))
                for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
                    train_val = rows.iloc[train_val_idx].reset_index(drop=True)
                    if train_val["target"].nunique() < 2 or train_val["split_group"].nunique() < 2:
                        continue
                    val_splits = min(3, train_val["split_group"].nunique())
                    inner = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=seed + 100)
                    inner_train_idx, val_idx = next(inner.split(train_val, train_val["target"], train_val["split_group"]))
                    train_idx = train_val_idx[inner_train_idx]
                    absolute_val_idx = train_val_idx[val_idx]
                    add_split(
                        splits,
                        f"grouped_seed{seed}_fold{fold_idx}",
                        task,
                        rows,
                        train_idx,
                        absolute_val_idx,
                        test_idx,
                    )

        if task in {"keloid_binary", "lesional_status"}:
            for accession in sorted(rows["accession"].unique()):
                test_idx = rows.index[rows["accession"].eq(accession)].to_numpy()
                train_idx = rows.index[~rows["accession"].eq(accession)].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                train_rows = rows.iloc[train_idx]
                val_idx = train_idx[:0]
                if train_rows["split_group"].nunique() >= 3 and train_rows["target"].nunique() >= 2:
                    inner = StratifiedGroupKFold(n_splits=min(3, train_rows["split_group"].nunique()), shuffle=True, random_state=17)
                    inner_train_idx, inner_val_idx = next(inner.split(train_rows, train_rows["target"], train_rows["split_group"]))
                    val_idx = train_idx[inner_val_idx]
                    train_idx = train_idx[inner_train_idx]
                add_split(splits, f"leave_accession_out_{accession}", task, rows, train_idx, val_idx, test_idx)

        if task in {"cell_type", "keloid_binary"}:
            sc_rows = rows[rows["accession"].isin(["GSE163973", "GSE181297"])].reset_index(drop=True)
            for group in sorted(sc_rows["split_group"].unique()):
                test_idx = sc_rows.index[sc_rows["split_group"].eq(group)].to_numpy()
                train_idx = sc_rows.index[~sc_rows["split_group"].eq(group)].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                add_split(splits, f"leave_source_out_{group.replace('|', '_')}", task, sc_rows, train_idx, [], test_idx)

        if task == "fibroblast_subcluster":
            fib_rows = rows[rows["accession"].eq("GSE163973")].reset_index(drop=True)
            for group in sorted(fib_rows["split_group"].unique()):
                test_idx = fib_rows.index[fib_rows["split_group"].eq(group)].to_numpy()
                train_idx = fib_rows.index[~fib_rows["split_group"].eq(group)].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                add_split(
                    splits,
                    f"leave_patient_out_{group.replace('|', '_')}",
                    task,
                    fib_rows,
                    train_idx,
                    [],
                    test_idx,
                )

        if task == "celltype_subcluster":
            sub_rows = rows[rows["accession"].eq("GSE163973")].reset_index(drop=True)
            for group in sorted(sub_rows["split_group"].unique()):
                test_idx = sub_rows.index[sub_rows["split_group"].eq(group)].to_numpy()
                train_idx = sub_rows.index[~sub_rows["split_group"].eq(group)].to_numpy()
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                add_split(
                    splits,
                    f"leave_patient_out_{group.replace('|', '_')}",
                    task,
                    sub_rows,
                    train_idx,
                    [],
                    test_idx,
                )
    return splits


def write_outputs(manifest: pd.DataFrame, feature_tables: dict[str, pd.DataFrame], vocab_report: dict, splits: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = out_dir / "features"
    split_dir = out_dir / "splits"
    feature_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest.to_parquet(out_dir / "profile_manifest.parquet", index=False)
    manifest.to_csv(out_dir / "profile_manifest.csv", index=False)
    for name, table in feature_tables.items():
        table.to_parquet(feature_dir / f"{name}.parquet", index=False)
        feature_cols = [col for col in table.columns if col != "sample_id"]
        (feature_dir / f"{name}_columns.txt").write_text("\n".join(feature_cols) + "\n")
    (feature_dir / "feature_report.json").write_text(json.dumps(vocab_report, indent=2))
    (split_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    summary = {
        "n_profiles": int(len(manifest)),
        "modalities": manifest["processed_modality"].value_counts().to_dict(),
        "accessions": manifest["accession"].value_counts().to_dict(),
        "tasks": manifest["encoder_task"].value_counts().to_dict(),
        "keloid_binary": manifest["keloid_binary"].value_counts().to_dict(),
        "feature_sets": {name: int(table.shape[1] - 1) for name, table in feature_tables.items()},
        "n_splits": len(splits),
        "split_tasks": pd.Series([split["task"] for split in splits]).value_counts().to_dict() if splits else {},
    }
    (out_dir / "training_corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-shared-genes", type=int, default=5000)
    parser.add_argument("--split-repeats", type=int, default=3)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "keloid_binary",
            "lesional_status",
            "cell_type",
            "fibroblast_subcluster",
            "celltype_subcluster",
            "scar_differential",
            "out_of_domain_disease_state",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, feature_tables, vocab_report = load_artifacts(args.max_shared_genes)
    splits = build_splits(manifest, args.tasks, args.split_repeats)
    write_outputs(manifest, feature_tables, vocab_report, splits, args.out_dir)


if __name__ == "__main__":
    main()
