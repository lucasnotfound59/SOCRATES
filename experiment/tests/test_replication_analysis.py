import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

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
)


MODELS = [f"model-{i}" for i in range(6)]


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
    return pd.DataFrame(rows)


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
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "analysis"

    def tearDown(self):
        self.temp_dir.cleanup()

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
        for phrase in ("三条非响应记录", "六模型", "pending", "ceiling effects", "consciousness claim"):
            self.assertIn(phrase, report)

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
