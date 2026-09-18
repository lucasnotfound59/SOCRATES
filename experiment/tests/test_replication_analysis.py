import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import experiment.replication_analysis as replication_analysis_module

from experiment.replication_analysis import (
    data_quality_table,
    ece,
    evidence_tier,
    load_human_trials,
    load_model_attempts,
    summarize_groups,
    type2_auroc,
    validate_model_attempts,
    fit_metad,
    cluster_bootstrap,
    bootstrap_all,
    _mratio_plot_data,
    _ranked_models,
    plot_bayes_mratio_vs_errors,
    _strict_csv_bool,
    _strict_csv_number,
    _validate_formal_model_labels,
    _validate_manifest_input_digests,
    validate_analysis_artifacts,
)
from experiment.replication_hmetad import aggregate_results, refresh_run_manifest


MODELS = [
    "repl-gemma4-e2b-q4km", "repl-gemma4-e4b-q4km",
    "repl-gemma4-26b-a4b-qat", "repl-qwen3-1.7b-q8",
    "repl-qwen3-4b-q4km", "repl-qwen3-14b-q4km",
]


def make_model_attempts(models=6, items=400, samples=4):
    rows = []
    for model in MODELS[:models]:
        for item_no in range(items):
            for sample_idx in range(samples):
                rows.append({
                    "model": model, "language": "zh", "item_id": f"I{item_no:03d}",
                    "sample_idx": sample_idx, "gold": "True", "parsed_answer": "True",
                    "correct": "True", "confidence": 5, "parse_ok": "True",
                    "type": "常规", "subtype": "基础", "difficulty": "易",
                    "finish_reason": "stop", "error": "",
                })
    frame = pd.DataFrame(rows)
    # pandas 3 may infer Arrow string columns; tests intentionally mutate
    # protocol flags to booleans, so keep fixture fields assignment-compatible.
    for column in ("parse_ok", "parsed_answer", "finish_reason", "error"):
        frame[column] = frame[column].astype(object)
    return frame


def make_human_master():
    return pd.DataFrame([
        {"model": "human", "language": "zh", "status": "ok", "participant_id": "p1",
         "item_id": "I001", "sample_idx": 0, "gold": "True", "parsed_answer": "True",
         "correct": "True", "confidence": 4, "type": "常规", "subtype": "基础", "difficulty": "易"},
        {"model": "gpt", "language": "en", "status": "ok", "participant_id": "p2",
         "item_id": "I002", "sample_idx": 0, "gold": "False", "parsed_answer": "False",
         "correct": "True", "confidence": 3, "type": "常规", "subtype": "基础", "difficulty": "易"},
        {"model": "human", "language": "zh", "status": "fail_parse", "participant_id": "p3",
         "item_id": "I003", "sample_idx": 0, "gold": "True", "parsed_answer": "",
         "correct": "", "confidence": "", "type": "常规", "subtype": "基础", "difficulty": "易"},
    ])


def write_csv(frame):
    temp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    temp.close()
    path = Path(temp.name)
    frame.to_csv(path, index=False)
    return path


class ReplicationInputTests(unittest.TestCase):
    def test_csv_summary_helpers_reject_coercive_strings(self):
        with self.assertRaisesRegex(ValueError, "finite number"):
            _strict_csv_number("1.337", "summary.m_ratio_mean")
        with self.assertRaisesRegex(ValueError, "boolean"):
            _strict_csv_bool("True", "summary.reliable")

    def test_formal_model_label_contract_rejects_substitution(self):
        labels = [
            "repl-gemma4-e2b-q4km", "repl-gemma4-e4b-q4km",
            "repl-gemma4-26b-a4b-qat", "repl-qwen3-1.7b-q8",
            "repl-qwen3-4b-q4km", "repl-qwen3-14b-q4km",
        ]
        self.assertEqual(_validate_formal_model_labels(labels), set(labels))
        with self.assertRaisesRegex(ValueError, "(expected 6 models|formal model labels)"):
            _validate_formal_model_labels([*labels[:-1], "repl-qwen3-32b-q4km"])

    def test_manifest_input_digest_contract_rejects_changed_bytes(self):
        model_path = write_csv(pd.DataFrame({"value": [1]}))
        human_path = write_csv(pd.DataFrame({"value": [2]}))
        self.addCleanup(model_path.unlink)
        self.addCleanup(human_path.unlink)
        import hashlib
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        inputs = {"model": {"sha256": digest(model_path)}, "human": {"sha256": digest(human_path)}}
        _validate_manifest_input_digests(inputs, model_path, human_path)
        model_path.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "model input sha256"):
            _validate_manifest_input_digests(inputs, model_path, human_path)
    def test_model_loader_keeps_invalid_attempts_out_of_metrics(self):
        attempts = make_model_attempts()
        attempts.loc[0, "parse_ok"] = False
        attempts.loc[0, "parsed_answer"] = ""
        path = write_csv(attempts)
        self.addCleanup(path.unlink)
        all_rows, valid = load_model_attempts(path)
        self.assertEqual(len(all_rows), 9600)
        self.assertEqual(len(valid), 9599)
        self.assertNotIn(all_rows.iloc[0].name, valid.index)
        self.assertTrue((valid["cluster_id"] == valid["item_id"]).all())

    def test_validation_rejects_duplicate_formal_key(self):
        attempts = make_model_attempts()
        attempts.iloc[1] = attempts.iloc[0]
        with self.assertRaisesRegex(ValueError, "unique task keys"):
            validate_model_attempts(attempts)

    def test_formal_input_requires_registered_labels_and_complete_item_sample_design(self):
        attempts = make_model_attempts()
        attempts.loc[0, "model"] = "substituted-model"
        with self.assertRaisesRegex(ValueError, "(expected 6 models|formal model labels)"):
            validate_model_attempts(attempts)

        attempts = make_model_attempts()
        attempts.loc[0, "item_id"] = "I400"
        with self.assertRaisesRegex(ValueError, "400 items"):
            validate_model_attempts(attempts)

        attempts = make_model_attempts()
        attempts.loc[0, "sample_idx"] = 4
        with self.assertRaisesRegex(ValueError, "sample_idx"):
            validate_model_attempts(attempts)

        attempts = make_model_attempts()
        attempts.loc[0, "item_id"] = "I400"
        attempts.loc[0, "sample_idx"] = 0
        with self.assertRaisesRegex(ValueError, "400 items"):
            validate_model_attempts(attempts)

    def test_formal_input_requires_common_cross_model_item_metadata(self):
        attempts = make_model_attempts()
        attempts.loc[400:403, "type"] = "学科"
        with self.assertRaisesRegex(ValueError, "metadata.*consistent"):
            validate_model_attempts(attempts)
        attempts = make_model_attempts()
        attempts.loc[1, "type"] = "学科"
        with self.assertRaisesRegex(ValueError, "metadata.*unique"):
            validate_model_attempts(attempts)

    def test_human_loader_returns_only_valid_chinese_human_rows(self):
        path = write_csv(make_human_master())
        self.addCleanup(path.unlink)
        clean = load_human_trials(path)
        self.assertEqual(len(clean), 1)
        self.assertTrue((clean["model"] == "human").all())
        self.assertTrue((clean["language"] == "zh").all())
        self.assertEqual(clean.iloc[0]["conf"], 4)
        self.assertEqual(clean.iloc[0]["stated"], 0.875)
        self.assertEqual(clean.iloc[0]["cluster_id"], "p1")
        self.assertNotIn("participant_id", clean.columns)
        self.assertIn("cluster_id", clean.columns)

    def test_quality_table_classifies_invalid_rows(self):
        attempts = make_model_attempts()
        attempts.loc[0, "parse_ok"] = False
        attempts.loc[0, "parsed_answer"] = ""
        attempts.loc[1, "parse_ok"] = False
        attempts.loc[1, "finish_reason"] = "length"
        attempts.loc[2, "parse_ok"] = False
        attempts.loc[2, "error"] = "timeout"
        quality = data_quality_table(attempts)
        self.assertEqual(list(quality.columns), ["model", "n_attempted", "n_valid", "n_invalid", "coverage", "n_truncated", "n_api_error", "n_parse_failure"])
        self.assertEqual(quality["n_attempted"].sum(), 9600)
        self.assertEqual(quality["n_invalid"].sum(), 3)
        self.assertEqual(quality["n_truncated"].sum(), 1)
        self.assertEqual(quality["n_api_error"].sum(), 1)
        self.assertEqual(quality["n_parse_failure"].sum(), 1)

    def test_correct_is_derived_from_answer_and_gold(self):
        attempts = make_model_attempts()
        attempts.loc[0, "correct"] = "False"  # deliberately disagree
        path = write_csv(attempts)
        self.addCleanup(path.unlink)
        _, valid = load_model_attempts(path)
        self.assertTrue(bool(valid.loc[0, "correct"]))

    def test_validation_rejects_canonical_duplicate_and_blank_keys(self):
        attempts = make_model_attempts()
        attempts.loc[1, "item_id"] = "  " + attempts.loc[0, "item_id"] + "  "
        attempts.loc[1, "sample_idx"] = attempts.loc[0, "sample_idx"]
        with self.assertRaisesRegex(ValueError, "unique task keys"):
            validate_model_attempts(attempts)
        attempts = make_model_attempts()
        attempts.loc[0, "item_id"] = " "
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_model_attempts(attempts)

    def test_human_loader_rejects_missing_participant_id(self):
        human = make_human_master()
        human.loc[0, "participant_id"] = " "
        path = write_csv(human)
        self.addCleanup(path.unlink)
        clean = load_human_trials(path)
        self.assertEqual(len(clean), 0)


class ReplicationMetricTests(unittest.TestCase):
    def test_summary_orders_formal_models_by_family_and_parameter_size(self):
        models = [
            "local-qwen3-14b", "local-gemma-26b-a4b-qat", "human",
            "local-qwen3-4b", "local-gemma-e4b", "local-gemma-e2b",
            "local-qwen3-1.7b",
        ]
        data = pd.DataFrame({
            "model": models, "conf": [1] * len(models), "stated": [0.5] * len(models),
            "correct": [True] * len(models),
        })
        self.assertEqual(summarize_groups(data)["model"].tolist(), [
            "human", "local-gemma-e2b", "local-gemma-e4b",
            "local-gemma-26b-a4b-qat", "local-qwen3-1.7b",
            "local-qwen3-4b", "local-qwen3-14b",
        ])

    def test_summary_orders_actual_formal_replication_labels(self):
        labels = [
            "repl-qwen3-14b-q4km", "repl-gemma4-26b-a4b-qat",
            "repl-qwen3-4b-q4km", "repl-gemma4-e4b-q4km",
            "repl-qwen3-1.7b-q8", "repl-gemma4-e2b-q4km",
        ]
        data = pd.DataFrame({
            "model": labels, "conf": [1] * len(labels),
            "stated": [0.5] * len(labels), "correct": [True] * len(labels),
        })
        self.assertEqual(summarize_groups(data)["model"].tolist(), [
            "repl-gemma4-e2b-q4km", "repl-gemma4-e4b-q4km",
            "repl-gemma4-26b-a4b-qat", "repl-qwen3-1.7b-q8",
            "repl-qwen3-4b-q4km", "repl-qwen3-14b-q4km",
        ])

    def test_ece_uses_binary_task_confidence_mapping(self):
        group = pd.DataFrame({"conf": [1, 1, 5, 5], "correct": [True, False, True, False]})
        self.assertAlmostEqual(ece(group), 0.25)

    def test_type2_auroc_rewards_higher_confidence_on_correct_trials(self):
        group = pd.DataFrame({"conf": [5, 4, 2, 1], "correct": [True, True, False, False]})
        self.assertAlmostEqual(type2_auroc(group), 1.0)

    def test_evidence_tiers_match_registered_thresholds(self):
        self.assertEqual(evidence_tier(30), "data-driven")
        self.assertEqual(evidence_tier(10), "regularized")
        self.assertEqual(evidence_tier(9), "prior-dominated")


def make_clustered_group(cluster_sizes=(4, 4, 4)):
    rows = []
    for cluster, size in enumerate(cluster_sizes):
        for trial in range(size):
            rows.append({"cluster_id": f"c{cluster}", "gold": "True",
                         "correct": trial % 2 == 0, "conf": (trial % 5) + 1,
                         "stated": ((trial % 5) + 1) / 5})
    return pd.DataFrame(rows)


def make_metad_group(n_wrong=29):
    n = 100
    correct = [False] * n_wrong + [True] * (n - n_wrong)
    return pd.DataFrame({"gold": ["True"] * n, "correct": correct,
                         "conf": [1] * n_wrong + [5] * (n - n_wrong),
                         "stated": [0.5] * n_wrong + [1.0] * (n - n_wrong),
                         "cluster_id": [f"c{i // 4}" for i in range(n)]})


class ReplicationBootstrapTests(unittest.TestCase):
    def test_bootstrap_resamples_whole_clusters(self):
        group = make_clustered_group((40, 40, 40))
        observed = []

        def fake_fit(sample):
            observed.append(sample.groupby("cluster_id").size().tolist())
            return {"m_ratio": 0.75}

        cluster_bootstrap(group, nboot_metrics=4, nboot_mratio=4,
                          seed=20260918, fit_fn=fake_fit)
        self.assertEqual(len(observed), 5)  # point fit + four bootstrap fits
        self.assertTrue(all(all(size % 40 == 0 for size in draw) for draw in observed[1:]))

    def test_fit_result_carries_error_count_and_evidence_tier(self):
        result = fit_metad(make_metad_group(n_wrong=29))
        self.assertEqual(result["n_wrong"], 29)
        self.assertEqual(result["evidence_tier"], "regularized")
        self.assertTrue(result["fit_status"])

    def test_failed_partial_metad_result_keeps_all_estimates_nan(self):
        group = make_metad_group(n_wrong=29)
        bad_result = pd.DataFrame({"dprime": [1.0], "meta_d": [2.0],
                                   "m_ratio": [3.0]})
        with patch("metadpy.mle.metad", return_value=bad_result):
            result = fit_metad(group)
        self.assertTrue(all(pd.isna(result[key]) for key in
                            ("dprime", "meta_d", "m_ratio", "m_diff")))
        self.assertIn("KeyError", result["fit_status"])

    def test_bootstrap_is_deterministic_and_keeps_model_order(self):
        data = pd.concat([make_metad_group(30).assign(model="b"),
                          make_metad_group(30).assign(model="a")], ignore_index=True)
        first = bootstrap_all(data, nboot_metrics=4, nboot_mratio=2, seed=7)
        second = bootstrap_all(data, nboot_metrics=4, nboot_mratio=2, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["model"].tolist(), ["a", "b"])


class ReplicationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build a privacy-safe formal-sized audit fixture when no run exists."""
        cls._fixture_temp = tempfile.TemporaryDirectory()
        root = Path(cls._fixture_temp.name)
        cls._fixture_analysis = root / "analysis"
        cls._fixture_model = root / "model.csv"
        cls._fixture_human = root / "human.csv"
        model = make_model_attempts()
        model.loc[:2, "parse_ok"] = False
        model.loc[:2, "parsed_answer"] = ""
        hallucination_model = model["item_id"].isin({"I000", "I001", "I002", "I003"})
        model.loc[hallucination_model, "type"] = "幻觉"
        model.loc[model["item_id"].isin({"I000", "I001"}), "gold"] = "False"
        model.loc[model["item_id"].isin({"I000", "I001"}), "parsed_answer"] = "False"
        model.loc[model["item_id"].isin({"I002", "I003"}), "gold"] = "True"
        model.loc[model["item_id"].isin({"I002", "I003"}), "parsed_answer"] = "True"
        model.to_csv(cls._fixture_model, index=False)
        human = pd.DataFrame([{
            "model": "human", "language": "zh", "status": "ok",
            "participant_id": f"fixture-p{index % 17}", "item_id": f"H{index:04d}",
            "sample_idx": 0,
            "gold": "False" if index < 100 else ("True" if index < 200 else "True"),
            "parsed_answer": "False" if index < 100 else ("True" if index < 200 else "True"),
            "confidence": 4, "type": "幻觉" if index < 200 else "常规",
            "subtype": "基础", "difficulty": "易",
        } for index in range(1647)])
        human.to_csv(cls._fixture_human, index=False)
        from experiment.replication_analysis import run_analysis, load_human_trials, load_model_attempts
        run_analysis(cls._fixture_model, cls._fixture_human, cls._fixture_analysis,
                     nboot_metrics=2, nboot_mratio=1)
        manifest_path = cls._fixture_analysis / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["bootstrap"] = {"nboot_metrics": 800, "nboot_mratio": 250}
        manifest["bootstrap_counts"] = {"metrics": 800, "mratio": 250}
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        from experiment.replication_hmetad import aggregate_results
        digest = {
            "model": __import__("hashlib").sha256(cls._fixture_model.read_bytes()).hexdigest(),
            "human": __import__("hashlib").sha256(cls._fixture_human.read_bytes()).hexdigest(),
        }
        _, valid_model = load_model_attempts(cls._fixture_model)
        human_valid = load_human_trials(cls._fixture_human)
        source_frames = {"human": human_valid}
        source_frames.update({model: valid_model.loc[valid_model["model"].eq(model)] for model in MODELS})
        counts = {model: len(frame) for model, frame in source_frames.items()}
        group_dir = cls._fixture_analysis / "hmetad" / "groups"
        group_dir.mkdir(parents=True, exist_ok=True)
        labels = ["human", *MODELS]
        for label in labels:
            record = {
                "model": label, "status": "complete", "n": counts[label],
                "n_wrong": int((~source_frames[label]["correct"].astype(bool)).sum()),
                "error_count": int((~source_frames[label]["correct"].astype(bool)).sum()),
                "evidence_tier": evidence_tier(int((~source_frames[label]["correct"].astype(bool)).sum())),
                "draws_requested": 800, "chains_requested": 2, "draws": 4,
                "chains": 2, "seed": 20260918, "input_digests": digest,
                "m_ratio_mean": 1.0, "m_ratio_median": 1.0, "m_ratio_sd": 0.1,
                "mr_mean": 1.0, "mr_median": 1.0, "post_sd": 0.1,
                "hdi_lo": 0.8, "hdi_hi": 1.2, "hdi_width": 0.4,
                "rhat": 1.01, "n_divergent": 0, "divergence_rate": 0.0,
                "reliable": True,
            }
            write_path = group_dir / (label.replace("/", "_") + ".json")
            write_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        aggregate_results(labels, group_dir,
                         output_path=cls._fixture_analysis / "hmetad_summary.csv",
                         report_path=cls._fixture_analysis / "analysis_report_zh.md")
        refresh_run_manifest(cls._fixture_analysis, draws=800, chains=2, seed=20260918)

    @classmethod
    def tearDownClass(cls):
        cls._fixture_temp.cleanup()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "analysis"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _copy_formal_audit_fixture(self):
        target = Path(self.temp_dir.name) / "formal-analysis"
        shutil.copytree(self._fixture_analysis, target)
        return target, self._fixture_model, self._fixture_human

    def _refresh_fixture_manifest(self, target):
        refresh_run_manifest(target, draws=800, chains=2, seed=20260918)

    def _validate(self, target, model_path, human_path):
        """Use the same validator with a cheap exact test bootstrap fixture."""
        original_bootstrap = replication_analysis_module.bootstrap_all

        def test_bootstrap(data, **_kwargs):
            return original_bootstrap(data, nboot_metrics=2, nboot_mratio=1, seed=20260918)

        with patch.object(replication_analysis_module, "bootstrap_all", side_effect=test_bootstrap):
            return validate_analysis_artifacts(target, model_path, human_path)

    def test_audit_rejects_failed_group_without_digest_and_with_wrong_n(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record.update({"status": "failed", "n": 1, "n_wrong": None,
                       "error_type": "TestFailure", "error_message": "synthetic"})
        record.pop("input_digests", None)
        group_path.write_text(json.dumps(record), encoding="utf-8")
        summary_path = target / "hmetad_summary.csv"
        summary = pd.read_csv(summary_path)
        for column in ("status", "n_wrong", "error_type", "error_message"):
            summary[column] = summary[column].astype(object)
        summary.loc[summary["model"].eq("human"), ["status", "n", "n_wrong", "error_type", "error_message"]] = [
            "failed", 1, pd.NA, "TestFailure", "synthetic"
        ]
        summary.to_csv(summary_path, index=False)
        comparison_path = target / "human_model_summary.csv"
        comparison = pd.read_csv(comparison_path)
        comparison.loc[comparison["model"].eq("human"), "n"] = 1
        comparison.to_csv(comparison_path, index=False)
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "Bayesian (group counts disagree with loaded source inputs|input digest mismatch)"):
            self._validate(target, model_path, human_path)

    def test_audit_rejects_nan_complete_diagnostic(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record["m_ratio_mean"] = "NaN"
        group_path.write_text(json.dumps(record), encoding="utf-8")
        summary_path = target / "hmetad_summary.csv"
        summary = pd.read_csv(summary_path)
        summary.loc[summary["model"].eq("human"), "m_ratio_mean"] = pd.NA
        summary.to_csv(summary_path, index=False)
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "m_ratio_mean must be a finite number"):
            self._validate(target, model_path, human_path)

    def test_audit_rejects_nullable_complete_reliable(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record["reliable"] = None
        group_path.write_text(json.dumps(record), encoding="utf-8")
        summary_path = target / "hmetad_summary.csv"
        summary = pd.read_csv(summary_path)
        summary["reliable"] = summary["reliable"].astype(object)
        summary.loc[summary["model"].eq("human"), "reliable"] = pd.NA
        summary.to_csv(summary_path, index=False)
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "reliable must be a boolean"):
            self._validate(target, model_path, human_path)

    def test_audit_rejects_json_numeric_string_m_ratio(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record["m_ratio_mean"] = str(record["m_ratio_mean"])
        group_path.write_text(json.dumps(record), encoding="utf-8")
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "m_ratio_mean must be a finite number"):
            self._validate(target, model_path, human_path)

    def test_audit_rejects_json_numeric_string_rhat(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record["rhat"] = str(record["rhat"])
        group_path.write_text(json.dumps(record), encoding="utf-8")
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "rhat must be a finite number"):
            self._validate(target, model_path, human_path)

    def test_audit_rejects_json_boolean_string_reliable(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record["reliable"] = "True"
        group_path.write_text(json.dumps(record), encoding="utf-8")
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "reliable must be a boolean"):
            self._validate(target, model_path, human_path)

    def test_audit_rejects_failed_record_with_stale_summary_diagnostics(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record.update({"status": "failed", "error_type": "TestFailure", "error_message": "synthetic"})
        for field in ("m_ratio_mean", "m_ratio_median", "m_ratio_sd", "mr_mean", "mr_median",
                      "post_sd", "hdi_lo", "hdi_hi", "hdi_width", "rhat", "n_divergent",
                      "divergence_rate", "reliable"):
            record.pop(field, None)
        group_path.write_text(json.dumps(record), encoding="utf-8")
        summary_path = target / "hmetad_summary.csv"
        summary = pd.read_csv(summary_path)
        for column in ("status", "n_wrong", "error_count", "evidence_tier", "error_type", "error_message"):
            summary[column] = summary[column].astype(object)
        summary.loc[summary["model"].eq("human"), ["status", "n_wrong", "error_count", "evidence_tier", "error_type", "error_message"]] = [
            "failed", 0, 0, "prior-dominated", "TestFailure", "synthetic"
        ]
        summary.to_csv(summary_path, index=False)
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "unexpectedly present in summary"):
            self._validate(target, model_path, human_path)

    def test_audit_rejects_explicit_null_failed_diagnostic(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record.update({"status": "failed", "error_type": "TestFailure", "error_message": "synthetic"})
        for field in ("draws", "chains"):
            record.pop(field, None)
        record["m_ratio_mean"] = None
        group_path.write_text(json.dumps(record), encoding="utf-8")
        summary_path = target / "hmetad_summary.csv"
        summary = pd.read_csv(summary_path)
        for column in ("status", "n_wrong", "error_count", "evidence_tier", "draws", "chains", "error_type", "error_message", "m_ratio_mean"):
            summary[column] = summary[column].astype(object)
        summary.loc[summary["model"].eq("human"), ["status", "n_wrong", "error_count", "evidence_tier", "draws", "chains", "error_type", "error_message", "m_ratio_mean"]] = [
            "failed", 0, 0, "prior-dominated", pd.NA, pd.NA, "TestFailure", "synthetic", pd.NA
        ]
        summary.to_csv(summary_path, index=False)
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "m_ratio_mean is explicitly null"):
            self._validate(target, model_path, human_path)

    def test_audit_accepts_legitimate_failed_group_after_aggregate_read(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        failed = {
            "model": "human", "status": "failed", "n": 1647,
            "n_wrong": 0, "error_count": 0, "evidence_tier": "prior-dominated",
            "draws_requested": 800, "chains_requested": 2,
            "seed": 20260918, "input_digests": record["input_digests"],
            "error_type": "SamplingError", "error_message": "synthetic retryable failure",
        }
        group_path.write_text(json.dumps(failed), encoding="utf-8")
        aggregate_results(
            ["human", "repl-gemma4-e2b-q4km", "repl-gemma4-e4b-q4km",
             "repl-gemma4-26b-a4b-qat", "repl-qwen3-1.7b-q8",
             "repl-qwen3-4b-q4km", "repl-qwen3-14b-q4km"],
            target / "hmetad/groups", output_path=target / "hmetad_summary.csv",
            report_path=target / "analysis_report_zh.md",
        )
        # Re-read the aggregate CSV as production validation does; nullable
        # mixed columns become float64 (e.g. complete 800 -> 800.0).
        self._refresh_fixture_manifest(target)
        self._validate(target, model_path, human_path)

    def test_csv_integer_helper_rejects_fractional_and_string_values(self):
        from experiment.replication_analysis import _strict_csv_int
        self.assertEqual(_strict_csv_int(800.0, "summary.draws"), 800)
        with self.assertRaisesRegex(ValueError, "integer"):
            _strict_csv_int(800.5, "summary.draws")
        with self.assertRaisesRegex(ValueError, "integer"):
            _strict_csv_int("800", "summary.draws")

    def test_audit_rejects_self_consistent_counts_that_disagree_with_source(self):
        target, model_path, human_path = self._copy_formal_audit_fixture()
        group_path = target / "hmetad/groups/human.json"
        record = json.loads(group_path.read_text(encoding="utf-8"))
        record["n"] = 1
        group_path.write_text(json.dumps(record), encoding="utf-8")
        summary_path = target / "hmetad_summary.csv"
        summary = pd.read_csv(summary_path)
        summary.loc[summary["model"].eq("human"), "n"] = 1
        summary.to_csv(summary_path, index=False)
        comparison_path = target / "human_model_summary.csv"
        comparison = pd.read_csv(comparison_path)
        comparison.loc[comparison["model"].eq("human"), "n"] = 1
        comparison.to_csv(comparison_path, index=False)
        self._refresh_fixture_manifest(target)
        with self.assertRaisesRegex(ValueError, "loaded source inputs"):
            self._validate(target, model_path, human_path)

    def test_run_analysis_writes_isolated_complete_artifacts(self):
        from experiment.replication_analysis import run_analysis

        model_path = write_csv(make_model_attempts(6, 400, 4))
        human_path = write_csv(make_human_master())
        self.addCleanup(model_path.unlink)
        self.addCleanup(human_path.unlink)
        paths = run_analysis(
            model_path=model_path,
            human_path=human_path,
            output_dir=self.output_dir,
            nboot_metrics=8,
            nboot_mratio=0,
        )
        required = {
            "data_quality.csv", "model_summary.csv", "human_model_summary.csv",
            "by_type.csv", "by_difficulty.csv", "confidence_distribution.csv",
            "reliability.csv", "hallucination_breakdown.csv",
            "hallucination_sdt.csv", "metad_summary.csv",
            "bootstrap_summary.csv", "tables.md", "analysis_report_zh.md",
            "run_manifest.json",
        }
        self.assertTrue(required.issubset({path.name for path in paths.values()}))
        comparison = pd.read_csv(self.output_dir / "human_model_summary.csv")
        self.assertNotIn("participant_id", comparison.columns)
        for name in required:
            self.assertTrue((self.output_dir / name).exists() or name.startswith("figures/"))
        figures = self.output_dir / "figures"
        self.assertEqual(
            {path.name for path in figures.glob("*.png")},
            {
                "model_accuracy_ece.png", "human_model_accuracy_ece.png",
                "accuracy_by_type.png", "mratio_evidence.png",
                "reliability_curves.png", "hallucination_breakdown.png",
            },
        )
        manifest = json.loads((self.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["random_seed"], 20260918)
        self.assertIn("run_manifest.json", manifest["artifacts"])
        self.assertIn("sha256", manifest["inputs"]["model"])
        report = (self.output_dir / "analysis_report_zh.md").read_text(encoding="utf-8")
        for phrase in ("无效/非响应记录共 0 条", "六模型", "pending", "ceiling effects", "consciousness claim"):
            self.assertIn(phrase, report)
        for model in MODELS:
            self.assertIn(model, report)
        ranking_text = report.split("完整准确率排名（高到低）：", 1)[1].split("。", 1)[0]
        self.assertEqual(ranking_text, " > ".join(MODELS))

    def test_report_rankings_include_all_models_in_metric_order(self):
        summary = pd.DataFrame({
            "model": ["model-0", "model-1", "model-2", "model-3", "model-4", "model-5"],
            "accuracy": [0.70, 0.95, 0.80, 0.65, 0.90, 0.75],
            "ece": [0.30, 0.05, 0.20, 0.35, 0.10, 0.25],
        })
        self.assertEqual(
            _ranked_models(summary, "accuracy"),
            ["model-1", "model-4", "model-2", "model-5", "model-0", "model-3"],
        )
        self.assertEqual(
            _ranked_models(summary, "ece", ascending=True),
            ["model-1", "model-4", "model-2", "model-5", "model-0", "model-3"],
        )

    def test_mratio_plot_data_keeps_order_and_bootstrap_intervals(self):
        metad = pd.DataFrame({
            "model": ["model-2", "model-0", "model-1"],
            "m_ratio": [0.8, 1.1, 0.9],
            "evidence_tier": ["regularized", "data-driven", "prior-dominated"],
        })
        bootstrap = pd.DataFrame({
            "model": ["model-0", "model-1", "model-2"],
            "m_ratio_lo": [0.9, 0.7, 0.6],
            "m_ratio_hi": [1.3, 1.1, 1.0],
        })
        result = _mratio_plot_data(metad, bootstrap)
        self.assertEqual(result["model"].tolist(), ["model-2", "model-0", "model-1"])
        self.assertEqual(result["x"].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(result["m_ratio_lo"].tolist(), [0.6, 0.9, 0.7])
        self.assertEqual(result["m_ratio_hi"].tolist(), [1.0, 1.3, 1.1])

    def test_output_guard_and_resume_contract(self):
        from experiment.replication_analysis import run_analysis

        model_path = write_csv(make_model_attempts())
        human_path = write_csv(make_human_master())
        self.addCleanup(model_path.unlink)
        self.addCleanup(human_path.unlink)
        run_analysis(model_path, human_path, self.output_dir, nboot_metrics=0, nboot_mratio=0)
        with self.assertRaises(FileExistsError):
            run_analysis(model_path, human_path, self.output_dir, nboot_metrics=0, nboot_mratio=0)
        (self.output_dir / "hmetad").mkdir()
        marker = self.output_dir / "hmetad" / "group.csv"
        marker.write_text("keep", encoding="utf-8")
        run_analysis(model_path, human_path, self.output_dir, nboot_metrics=0, nboot_mratio=0, resume=True)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        forbidden = Path(__file__).resolve().parents[1] / "results" / "analysis"
        with self.assertRaises(ValueError):
            run_analysis(model_path, human_path, forbidden, nboot_metrics=0, nboot_mratio=0)

    def test_validate_only_emits_success_marker_after_gate(self):
        from experiment.replication_analysis import main
        with patch(
            "experiment.replication_analysis.validate_analysis_artifacts",
            return_value={"model_rows": 9600},
        ) as audit:
            with patch("builtins.print") as printer:
                result = main([
                    "--validate-only", "--model-results", "model.csv",
                    "--human-master", "human.csv", "--output-dir", str(self.output_dir),
                ])
        self.assertEqual(result, 0)
        audit.assert_called_once()
        self.assertEqual(printer.call_args_list[-1].args, ("ANALYSIS_AUDIT_OK",))


class BayesMratioFigureTests(unittest.TestCase):
    """The all-groups figure must plot every group and never drop or reorder one."""

    @staticmethod
    def _frame():
        return pd.DataFrame({
            "model": ["human", "model-a", "model-b", "model-c"],
            "n_wrong": [437, 0, 12, 40],
            "mr_mean": [1.334, 1.071, 0.300, 0.800],
            "hdi_lo": [1.160, 0.926, 0.100, 0.700],
            "hdi_hi": [1.522, 1.231, 0.500, 0.900],
            "reliable": [True, False, False, True],
        })

    def test_plots_every_group_ordered_by_error_count(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "fig.png"
            path = plot_bayes_mratio_vs_errors(self._frame(), out)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_rejects_frame_missing_required_columns(self):
        frame = self._frame().drop(columns=["hdi_hi"])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "missing columns"):
                plot_bayes_mratio_vs_errors(frame, Path(temp) / "fig.png")

    def test_rejects_frame_without_a_finite_estimate(self):
        frame = self._frame()
        frame["mr_mean"] = float("nan")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "no groups with a finite"):
                plot_bayes_mratio_vs_errors(frame, Path(temp) / "fig.png")

    def test_reliable_flag_tolerates_csv_booleans(self):
        frame = self._frame()
        frame["reliable"] = ["True", "False", "False", "True"]
        with tempfile.TemporaryDirectory() as temp:
            path = plot_bayes_mratio_vs_errors(frame, Path(temp) / "fig.png")
            self.assertTrue(path.is_file())

    def test_missing_reliable_column_is_treated_as_unreliable(self):
        frame = self._frame().drop(columns=["reliable"])
        with tempfile.TemporaryDirectory() as temp:
            path = plot_bayes_mratio_vs_errors(frame, Path(temp) / "fig.png")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
