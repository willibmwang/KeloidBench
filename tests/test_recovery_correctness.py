#!/usr/bin/env python3
"""Unit tests for SpheroScar recovery-critical correctness checks."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from domain_adaptation import apply_normalization  # noqa: E402
from ensembl_map import strip_ensembl_version  # noqa: E402
from expression_processing import is_hgnc_like_symbol, symbol_from_gene_assignment  # noqa: E402
from train_ensemble_loso import (  # noqa: E402
    evaluate_probs,
    positive_class_proba,
    summarize,
    tune_threshold,
)


class GeneAssignmentTests(unittest.TestCase):
    def test_strip_ensembl_gencode_suffix(self):
        self.assertEqual(strip_ensembl_version("ENSG00000000003.14_2"), "ENSG00000000003")
        self.assertEqual(strip_ensembl_version("ENSG00000000003.14"), "ENSG00000000003")
        self.assertEqual(strip_ensembl_version("ENSG00000000003"), "ENSG00000000003")

    def test_gpl6244_style_assignment_prefers_symbol(self):
        raw = "NM_001135934 // POSTN // periostin, osteoblast specific factor // 13q13.3 // 10631"
        self.assertEqual(symbol_from_gene_assignment(raw), "POSTN")
        self.assertFalse(is_hgnc_like_symbol("NM_001135934"))
        self.assertTrue(is_hgnc_like_symbol("POSTN"))

    def test_soft_parser_keeps_affymetrix_and_numeric_ids(self):
        import tempfile
        from preprocess_microarray import load_platform_mapping_from_soft

        soft = (
            "^PLATFORM = GPL_TEST\n"
            "!platform_table_begin\n"
            "ID\tGene Symbol\tgene_assignment\n"
            "1007_s_at\tDDR1\t\n"
            "7971077\t\tNM_001135934 // POSTN // periostin // 13q13.3 // 10631\n"
            "!platform_table_end\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GPL_TEST.txt"
            path.write_text(soft)
            mapping = load_platform_mapping_from_soft(path)
        self.assertEqual(mapping.get("1007_s_at"), "DDR1")
        self.assertEqual(mapping.get("7971077"), "POSTN")


class EnsembleSemanticsTests(unittest.TestCase):
    def test_positive_class_proba_alignment(self):
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
        y_raw = pd.Series(["keloid"] * 20 + ["non_keloid"] * 20)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        self.assertEqual(list(le.classes_), ["keloid", "non_keloid"])
        model = LogisticRegression(max_iter=2000, random_state=0).fit(x, y)
        probs = positive_class_proba(model, x, le)
        pred_from_proba = np.where(probs >= 0.5, le.transform(["keloid"])[0], le.transform(["non_keloid"])[0])
        pred_model = model.predict(x)
        self.assertGreaterEqual((pred_from_proba == pred_model).mean(), 0.9)

    def test_threshold_tuning_perfect_separator(self):
        y_true = np.array([0, 0, 1, 1])  # keloid=0
        probs_keloid = np.array([0.9, 0.8, 0.1, 0.2])
        thr = tune_threshold(y_true, probs_keloid, positive_code=0)
        metrics = evaluate_probs(y_true, probs_keloid, ["keloid", "non_keloid"], thr, positive_code=0)
        self.assertGreater(metrics["weighted_f1"], 0.9)

    def test_summarize_flattens_columns(self):
        df = pd.DataFrame(
            [
                {
                    "feature_set": "modules_only",
                    "model": "ensemble_member",
                    "normalization_mode": "none",
                    "weighted_f1": 0.7,
                    "balanced_accuracy": 0.6,
                    "accuracy": 0.65,
                    "auroc": 0.8,
                },
                {
                    "feature_set": "modules_only",
                    "model": "ensemble_member",
                    "normalization_mode": "none",
                    "weighted_f1": 0.5,
                    "balanced_accuracy": 0.55,
                    "accuracy": 0.6,
                    "auroc": 0.7,
                },
            ]
        )
        summary = summarize(df)
        self.assertIn("weighted_f1_mean", summary.columns)


class SplitAndNormTests(unittest.TestCase):
    def test_no_accidental_double_rank_identity_on_ranks(self):
        # Quantile rank of already-uniform ranks should remain approximately ordered.
        x_train = pd.DataFrame({"m": [0.1, 0.5, 0.9]}, index=["a", "b", "c"])
        x_val = pd.DataFrame({"m": [0.2]}, index=["d"])
        x_test = pd.DataFrame({"m": [0.8]}, index=["e"])
        acc = {k: "GSE1" for k in ["a", "b", "c", "d", "e"]}
        tr, va, te = apply_normalization(
            x_train,
            x_val,
            x_test,
            list(x_train.index),
            list(x_val.index),
            list(x_test.index),
            acc,
            "quantile_rank",
        )
        self.assertEqual(list(tr.index), ["a", "b", "c"])
        self.assertTrue(tr.loc["a", "m"] < tr.loc["c", "m"])


class CorpusSemanticsTests(unittest.TestCase):
    def test_external_fibrosis_excluded_from_keloid_binary(self):
        from build_training_corpus import canonical_binary, clean_binary

        row = pd.Series(
            {
                "is_out_of_domain": True,
                "disease_label": "control",
                "keloid_vs_normal": "normal",
                "scar_type": "unknown",
                "encoder_response": "control",
                "treatment": "none",
            }
        )
        self.assertEqual(canonical_binary(row), "exclude")
        self.assertEqual(clean_binary(row), "exclude")

    def test_donor_grouping_requires_flag_for_unknown(self):
        from build_training_corpus import make_group_id

        known = pd.Series(
            {"modality": "bulk", "accession": "GSE1", "patient_id": "D1", "allow_accession_level_grouping": False}
        )
        unknown_allowed = pd.Series(
            {"modality": "bulk", "accession": "GSE1", "patient_id": "unknown", "allow_accession_level_grouping": True}
        )
        unknown_blocked = pd.Series(
            {"modality": "bulk", "accession": "GSE1", "patient_id": "unknown", "allow_accession_level_grouping": False}
        )
        self.assertEqual(make_group_id(known), "bulk|GSE1|D1")
        self.assertTrue(make_group_id(unknown_allowed).endswith("|accession"))
        self.assertTrue(make_group_id(unknown_blocked).endswith("|unknown_donor"))

    def test_rank_program_positive_minus_negative(self):
        from build_training_corpus import coverage_adjusted_rank_programs
        from gene_modules import RANK_PROGRAM_SETS

        genes = sorted({g for prog in RANK_PROGRAM_SETS.values() for g in prog["positive"] + prog["negative"]})
        # Fibroblast-like sample: high POSTN/COL1A1, low KRT.
        values = {g: 0.1 for g in genes}
        for g in ["POSTN", "COL1A1", "CTHRC1", "PDGFRA", "LUM"]:
            if g in values:
                values[g] = 10.0
        for g in ["KRT1", "KRT10", "KRT14"]:
            if g in values:
                values[g] = 0.01
        expr = pd.DataFrame([values], index=["s1"])
        manifest = pd.DataFrame(
            [{"sample_id": "s1", "processed_modality": "bulk_rnaseq", "accession": "GSE_TEST"}]
        )
        scores = coverage_adjusted_rank_programs(manifest, {"bulk_rnaseq": expr}, RANK_PROGRAM_SETS)
        self.assertGreater(float(scores.loc[0, "ecm_profibrotic_rank"]), 0.0)
        self.assertLess(float(scores.loc[0, "keratinocyte_compartment_rank"]), 0.0)

    def test_fixed_binary_label_order(self):
        from train_nested_loso import BINARY_CLASSES, encode_binary

        y_train = pd.Series(["non_keloid", "keloid", "keloid"])
        y_test = pd.Series(["keloid", "non_keloid"])
        codes, y_tr, y_te, pos = encode_binary(y_train, y_test)
        self.assertEqual(codes, list(BINARY_CLASSES))
        self.assertEqual(pos, 0)
        self.assertEqual(y_tr.tolist(), [1, 0, 0])

    def test_clinical_endpoints_are_non_interchangeable(self):
        from build_training_corpus import (
            keloid_vs_normal_scar,
            keloid_vs_pathologic_scar,
            keloid_vs_unaffected_skin,
        )

        normal_scar = pd.Series(
            {
                "is_out_of_domain": False,
                "encoder_task": "scar_differential",
                "disease_label": "normotrophic_scar",
                "keloid_vs_normal": "unknown",
                "scar_type": "normotrophic_scar",
                "encoder_response": "normotrophic_scar",
                "treatment": "none",
            }
        )
        hypertrophic = normal_scar.copy()
        hypertrophic["scar_type"] = "hypertrophic_scar"
        hypertrophic["disease_label"] = "hypertrophic_scar"
        hypertrophic["encoder_response"] = "hypertrophic_scar"
        skin = pd.Series(
            {
                "is_out_of_domain": False,
                "encoder_task": "keloid_vs_normal",
                "disease_label": "normal",
                "keloid_vs_normal": "normal",
                "scar_type": "normal",
                "encoder_response": "normal",
                "treatment": "none",
                "cell_type": "bulk_tissue",
                "accession": "GSE173900",
                "modality": "bulk_rnaseq",
            }
        )
        fibroblast = skin.copy()
        fibroblast["cell_type"] = "fibroblast"
        fibroblast["accession"] = "GSE282479"
        self.assertEqual(keloid_vs_normal_scar(normal_scar), "non_keloid")
        self.assertEqual(keloid_vs_pathologic_scar(hypertrophic), "non_keloid")
        self.assertEqual(keloid_vs_normal_scar(hypertrophic), "exclude")
        self.assertEqual(keloid_vs_unaffected_skin(skin), "non_keloid")
        self.assertEqual(keloid_vs_unaffected_skin(fibroblast), "exclude")
        self.assertEqual(keloid_vs_normal_scar(skin), "exclude")

    def test_susceptibility_excluded_from_lesion_endpoints(self):
        from build_training_corpus import canonical_binary, clean_binary, keloid_vs_unaffected_skin

        row = pd.Series(
            {
                "is_out_of_domain": False,
                "encoder_task": "wound_susceptibility",
                "disease_label": "control",
                "keloid_vs_normal": "normal",
                "scar_type": "unknown",
                "encoder_response": "control",
                "treatment": "none",
            }
        )
        self.assertEqual(canonical_binary(row), "exclude")
        self.assertEqual(clean_binary(row), "exclude")
        self.assertEqual(keloid_vs_unaffected_skin(row), "exclude")

    def test_scar_discriminative_programs_versioned(self):
        from gene_modules import (
            PREREGISTERED_FEATURE_SETS,
            SCAR_DISCRIMINATIVE_PROGRAM_SETS,
        )

        self.assertIn("scar_discriminative_rank_programs", PREREGISTERED_FEATURE_SETS)
        for name, prog in SCAR_DISCRIMINATIVE_PROGRAM_SETS.items():
            self.assertIn("source", prog, msg=name)
            self.assertIn("min_genes_present", prog, msg=name)
            self.assertGreaterEqual(len(prog.get("positive", [])), 2, msg=name)

    def test_equal_accession_selection_tiebreak(self):
        from train_nested_loso import tune_threshold_equal_accession

        # Accession A separates at 0.6; accession B separates at 0.4.
        fold_y = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]
        fold_p = [np.array([0.9, 0.8, 0.2, 0.1]), np.array([0.7, 0.6, 0.3, 0.2])]
        thr, mean_score, lower_tail = tune_threshold_equal_accession(fold_y, fold_p)
        self.assertTrue(0.0 <= thr <= 1.0)
        self.assertGreaterEqual(mean_score, 0.9)
        self.assertGreaterEqual(lower_tail, 0.9)

    def test_lockbox_reserved_in_registry(self):
        registry = json.loads((ROOT / "data/raw/public_keloid_cohorts.json").read_text())
        lockbox = {e["accession"] for e in registry.get("lockbox", [])}
        self.assertIn("GSE212954", lockbox)
        entry = next(e for e in registry["lockbox"] if e["accession"] == "GSE212954")
        self.assertIn("download_after_freeze", entry)
        self.assertNotIn(
            "Expression_Gene.xlsx",
            json.dumps(entry.get("download", {})),
        )
        skipped = {e["accession"] for e in registry.get("skipped_recovery", [])}
        self.assertIn("GSE125022", skipped)

    def test_breakthrough_feature_views_defined(self):
        from gene_modules import (
            BREAKTHROUGH_FEATURE_VIEWS,
            COMPOSITION_PROGRAM_SETS,
            FIBROSIS_VIEW_COLUMNS,
        )

        self.assertEqual(
            BREAKTHROUGH_FEATURE_VIEWS,
            ["fibrosis_only", "scar_discriminative", "composition_only", "fused_multiview"],
        )
        self.assertGreaterEqual(len(FIBROSIS_VIEW_COLUMNS), 3)
        for name, prog in COMPOSITION_PROGRAM_SETS.items():
            self.assertIn("source", prog, msg=name)
            self.assertIn("min_genes_present", prog, msg=name)

    def test_breakthrough_protocol_withholds_lockbox(self):
        proto_path = ROOT / "results/breakthrough_sprint/frozen_protocol.json"
        self.assertTrue(proto_path.exists(), "freeze breakthrough protocol before tests")
        proto = json.loads(proto_path.read_text())
        self.assertEqual(proto["protocol"], "breakthrough_sprint_v1")
        self.assertEqual(proto["product"]["primary"], "keloid_vs_unaffected_skin")
        lb = proto["breakthrough_lockbox"]["accession"]
        self.assertEqual(lb, "GSE185309")
        self.assertEqual(proto["breakthrough_lockbox"]["expression_status"], "withheld_until_protocol_freeze")
        self.assertIn("GSE212954", proto["ladder_reserved_lockbox"])
        # Freeze must precede one-shot lockbox scoring (no expression access before protocol hash).
        lock_path = ROOT / "results/breakthrough_sprint/breakthrough_lockbox_once.json"
        if lock_path.exists():
            self.assertLessEqual(proto_path.stat().st_mtime, lock_path.stat().st_mtime)
            lock = json.loads(lock_path.read_text())
            self.assertEqual(lock.get("lockbox_accession"), lb)
        else:
            # Pre-score: lockbox expression must still be absent from the training corpus.
            manifest = ROOT / "data/processed/training/profile_manifest.parquet"
            if manifest.exists():
                import pandas as pd

                df = pd.read_parquet(manifest, columns=["accession"])
                self.assertNotIn(lb, set(df["accession"].astype(str)))

    def test_selective_threshold_respects_coverage(self):
        from train_breakthrough_cascade import tune_selective_threshold

        y = np.array([0, 0, 0, 1, 1, 1])
        # Highly confident separator.
        p = np.array([0.95, 0.92, 0.91, 0.05, 0.08, 0.04])
        thr, f1, cov = tune_selective_threshold(y, p, min_coverage=0.60)
        self.assertGreaterEqual(cov, 0.60)
        self.assertGreaterEqual(f1, 0.9)
        self.assertGreaterEqual(thr, 0.5)

    def test_treated_excluded_from_clean_and_skin(self):
        from build_training_corpus import clean_binary, keloid_vs_unaffected_skin

        row = pd.Series(
            {
                "is_out_of_domain": False,
                "encoder_task": "keloid_vs_normal",
                "disease_label": "keloid",
                "keloid_vs_normal": "keloid",
                "scar_type": "keloid",
                "encoder_response": "keloid",
                "treatment": "hydrocortisone",
            }
        )
        self.assertEqual(clean_binary(row), "exclude")
        self.assertEqual(keloid_vs_unaffected_skin(row), "exclude")


if __name__ == "__main__":
    unittest.main()

