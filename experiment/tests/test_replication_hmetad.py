import json
import subprocess
import sys
import tempfile
import unittest
import platform
from pathlib import Path
from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd

from experiment.replication_analysis import evidence_tier

from experiment.replication_hmetad import (
    aggregate_results,
    pending_models,
    refresh_run_manifest,
    run_group,
    summarize_idata,
    update_report_bayesian,
    write_group_json,
)
from experiment.pytensor_compat import compatibility_status
import experiment.pytensor_compat as pytensor_compat


def make_fake_idata():
    posterior = {
        "meta_d": np.array([[1.0, 1.1, 0.9, 1.05], [1.05, 1.0, 0.95, 1.0]]),
        "d1": np.ones((2, 4)),
    }
    sample_stats = {"diverging": np.zeros((2, 4), dtype=bool)}
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)


def make_group(model="human", n_wrong=23):
    n = 40
    return pd.DataFrame({
        "model": [model] * n,
        "gold": ["True"] * n,
        "correct": [False] * n_wrong + [True] * (n - n_wrong),
        "conf": [1] * n_wrong + [5] * (n - n_wrong),
    })


def make_complete_record(model="human", digests=None):
    record = {
        "model": model, "status": "complete", "n": 40, "n_wrong": 23,
        "error_count": 23, "evidence_tier": "regularized",
        "draws_requested": 800, "chains_requested": 2, "draws": 4,
        "chains": 2, "seed": 20260918, "m_ratio_mean": 1.0,
        "m_ratio_median": 1.0, "m_ratio_sd": 0.1, "mr_mean": 1.0,
        "mr_median": 1.0, "post_sd": 0.1, "hdi_lo": 0.8,
        "hdi_hi": 1.2, "hdi_width": 0.4, "rhat": 1.01,
        "n_divergent": 0, "divergence_rate": 0.0, "reliable": True,
    }
    if digests is not None:
        record["input_digests"] = dict(digests)
    return record


class HMetaSummaryTests(unittest.TestCase):
    def test_pytensor_compatibility_does_not_falsify_platform_identity(self):
        before = platform.mac_ver()
        status = compatibility_status()
        self.assertEqual(platform.mac_ver(), before)
        self.assertIn("helper_version", status)
        self.assertIn("activated", status)

    def test_pytensor_compatibility_removes_only_ld64_and_is_idempotent(self):
        class FakeCompiler:
            @staticmethod
            def compile_args(march_flags=True):
                return ["-O2", "-ld64", "-pipe", "-march=native"] if march_flags else ["-O2", "-ld64", "-pipe"]

        initial = {
            "helper_version": pytensor_compat.COMPATIBILITY_VERSION,
            "activated": False, "reason": "not_configured",
            "pytensor_version": "not-installed", "platform": "darwin", "macos_major": None,
        }
        with patch.object(pytensor_compat, "_STATUS", initial), \
             patch.object(pytensor_compat.sys, "platform", "darwin"), \
             patch.object(pytensor_compat.platform, "mac_ver", return_value=("15.0", ())), \
             patch.object(pytensor_compat.importlib.metadata, "version", return_value="2.38.3"), \
             patch.object(pytensor_compat, "_load_compiler", return_value=FakeCompiler):
            first = pytensor_compat.configure_pytensor_compatibility()
            self.assertTrue(first["activated"])
            self.assertEqual(FakeCompiler.compile_args(), ["-O2", "-pipe", "-march=native"])
            self.assertEqual(FakeCompiler.compile_args(march_flags=False), ["-O2", "-pipe"])
            second = pytensor_compat.configure_pytensor_compatibility()
            self.assertEqual(second["reason"], "removed-generated-ld64")
            self.assertEqual(FakeCompiler.compile_args(), ["-O2", "-pipe", "-march=native"])

    def test_direct_script_entrypoint_supports_registered_command(self):
        script = Path(__file__).parents[1] / "replication_hmetad.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Resumable Bayesian HMeta-d runner", result.stdout)

    def test_summary_reports_hdi_rhat_divergences_and_evidence(self):
        result = summarize_idata(make_fake_idata(), n_wrong=23)
        self.assertEqual(result["evidence_tier"], "regularized")
        self.assertIn("hdi_lo", result)
        self.assertIn("hdi_hi", result)
        self.assertIn("rhat", result)
        self.assertIn("n_divergent", result)
        self.assertEqual(result["draws"], 4)
        self.assertEqual(result["chains"], 2)

    def test_run_group_adapter_passes_registered_sampler_contract(self):
        calls = []

        def fake_sampler(**kwargs):
            calls.append(kwargs)
            return object(), make_fake_idata()

        result = run_group(make_group(), draws=12, chains=2, seed=7, sampler=fake_sampler)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["model"], "human")
        self.assertEqual(result["draws"], 4)  # posterior metadata is authoritative
        self.assertEqual(calls[0]["nRatings"], 5)
        self.assertEqual(calls[0]["num_samples"], 12)
        self.assertEqual(calls[0]["num_chains"], 2)
        self.assertEqual(calls[0]["random_seed"], 7)

    def test_model_builder_adapter_passes_seed_to_actual_sampling_callable(self):
        builder_calls = []
        sample_calls = []

        def fake_builder(**kwargs):
            builder_calls.append(kwargs)
            return object()

        def fake_sampling(**kwargs):
            sample_calls.append(kwargs)
            return make_fake_idata()

        result = run_group(
            make_group(), draws=12, chains=2, seed=41,
            sampler=fake_builder, sampling_call=fake_sampling,
        )
        self.assertEqual(result["status"], "complete")
        self.assertFalse(builder_calls[0]["sample_model"])
        self.assertEqual(sample_calls[0]["random_seed"], 41)
        self.assertEqual(sample_calls[0]["draws"], 12)

    def test_ratio_rhat_can_fail_when_meta_d_rhat_would_pass(self):
        rng = np.random.default_rng(3)
        meta_d = rng.normal(1.0, 0.02, size=(4, 100))
        d1 = np.asarray([1.0, 1.5, 0.8, 1.2])[:, None] * np.ones((4, 100))
        idata = az.from_dict(
            posterior={"meta_d": meta_d, "d1": d1},
            sample_stats={"diverging": np.zeros((4, 100), dtype=bool)},
        )
        result = summarize_idata(idata, n_wrong=30)
        self.assertGreater(result["rhat"], 1.05)
        self.assertFalse(result["reliable"])

    def test_preprocessing_failure_is_returned_as_failed_result(self):
        result = run_group(pd.DataFrame({"model": ["broken"]}), 12, 2, 7,
                           sampler=lambda **kwargs: self.fail("sampler must not run"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertIn("correct", result["error_message"])

    def test_model_extraction_failure_is_failed_and_never_samples(self):
        sampler = lambda **kwargs: self.fail("sampler must not run")
        with patch("experiment.replication_hmetad._group_model",
                   side_effect=RuntimeError("model extraction failed")):
            result = run_group(make_group(), 12, 2, 7, sampler=sampler)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["model"], "unknown")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIn("model extraction failed", result["error_message"])

    def test_evidence_and_reliability_boundaries_are_strict(self):
        self.assertEqual(evidence_tier(9), "prior-dominated")
        self.assertEqual(evidence_tier(10), "regularized")
        self.assertEqual(evidence_tier(29), "regularized")
        self.assertEqual(evidence_tier(30), "data-driven")

        with patch("experiment.replication_hmetad._ratio_rhat", return_value=1.05):
            self.assertFalse(summarize_idata(make_fake_idata(), 30)["reliable"])
        with patch("experiment.replication_hmetad._ratio_rhat", return_value=1.049999):
            self.assertTrue(summarize_idata(make_fake_idata(), 30)["reliable"])

        posterior = {"meta_d": np.ones((2, 500)), "d1": np.ones((2, 500))}
        exact = az.from_dict(
            posterior=posterior,
            sample_stats={"diverging": np.array([[True] * 50 + [False] * 450,
                                                    [False] * 500])},
        )
        below = az.from_dict(
            posterior=posterior,
            sample_stats={"diverging": np.array([[True] * 49 + [False] * 451,
                                                    [False] * 500])},
        )
        with patch("experiment.replication_hmetad._ratio_rhat", return_value=1.0):
            self.assertFalse(summarize_idata(exact, 30)["reliable"])
            self.assertTrue(summarize_idata(below, 30)["reliable"])

    def test_completed_group_is_reused_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            group_dir = Path(temp)
            write_group_json(group_dir, make_complete_record("human"))
            pending = pending_models(["human", "model-a"], group_dir)
            self.assertEqual(pending, ["model-a"])

    def test_resume_requeues_complete_group_when_settings_change(self):
        with tempfile.TemporaryDirectory() as temp:
            group_dir = Path(temp)
            record = make_complete_record("human")
            record.update({"seed": 1, "draws_requested": 800, "chains_requested": 2})
            write_group_json(group_dir, record)
            self.assertEqual(pending_models(["human"], group_dir,
                                            draws=900, chains=2, seed=1), ["human"])

    def test_resume_requeues_when_input_digest_is_missing_or_changed(self):
        expected = {"model": "model-digest", "human": "human-digest"}
        with tempfile.TemporaryDirectory() as temp:
            group_dir = Path(temp)
            write_group_json(group_dir, make_complete_record("human", expected))
            self.assertEqual(
                pending_models(["human"], group_dir, draws=800, chains=2,
                               seed=20260918, input_digests=expected), [],
            )
            self.assertEqual(
                pending_models(["human"], group_dir, draws=800, chains=2,
                               seed=20260918, input_digests={"model": "changed", "human": "human-digest"}),
                ["human"],
            )
            write_group_json(group_dir, make_complete_record("legacy"))
            self.assertEqual(
                pending_models(["legacy"], group_dir, draws=800, chains=2,
                               seed=20260918, input_digests=expected),
                ["legacy"],
            )


class HMetaArtifactTests(unittest.TestCase):
    def test_refresh_manifest_enumerates_bayesian_outputs_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            analysis = Path(temp)
            (analysis / "hmetad" / "groups").mkdir(parents=True)
            (analysis / "run_manifest.json").write_text(
                json.dumps({"random_seed": 20260918}), encoding="utf-8"
            )
            (analysis / "hmetad_summary.csv").write_text("model,status\nhuman,complete\n", encoding="utf-8")
            (analysis / "hmetad" / "groups" / "human.json").write_text(
                '{"model":"human","status":"complete"}\n', encoding="utf-8"
            )
            refresh_run_manifest(analysis, draws=800, chains=2, seed=20260918)
            manifest = json.loads((analysis / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("hmetad_summary.csv", manifest["artifacts"])
            self.assertIn("hmetad/groups/human.json", manifest["artifact_hashes"])
            self.assertEqual(manifest["bayesian"]["draws"], 800)
            self.assertIsNone(manifest["artifact_hashes"]["run_manifest.json"])

    def test_aggregate_retains_failed_and_missing_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            group_dir = Path(temp) / "groups"
            write_group_json(group_dir, model="human", status="complete",
                             n_wrong=23, m_ratio_mean=1.0, hdi_lo=0.8,
                             hdi_hi=1.2, rhat=1.01, n_divergent=0,
                             draws=3, chains=2, reliable=True)
            write_group_json(group_dir, model="model-a", status="failed",
                             n_wrong=2, error_type="ValueError", error_message="bad")
            frame = aggregate_results(["human", "model-a", "model-b"], group_dir)
            self.assertEqual(frame["model"].tolist(), ["human", "model-a", "model-b"])
            self.assertEqual(frame.set_index("model").loc["model-a", "status"], "failed")
            self.assertEqual(frame.set_index("model").loc["model-b", "status"], "missing")

    def test_aggregate_orders_formal_groups_human_then_family_and_size(self):
        labels = [
            "repl-qwen3-14b-q4km", "repl-gemma4-26b-a4b-qat",
            "human", "repl-qwen3-1.7b-q8", "repl-gemma4-e4b-q4km",
            "repl-qwen3-4b-q4km", "repl-gemma4-e2b-q4km",
        ]
        with tempfile.TemporaryDirectory() as temp:
            group_dir = Path(temp) / "groups"
            for label in labels:
                write_group_json(group_dir, model=label, status="failed",
                                 error_type="TestError", error_message="test")
            frame = aggregate_results(labels, group_dir)
            self.assertEqual(frame["model"].tolist(), [
                "human", "repl-gemma4-e2b-q4km", "repl-gemma4-e4b-q4km",
                "repl-gemma4-26b-a4b-qat", "repl-qwen3-1.7b-q8",
                "repl-qwen3-4b-q4km", "repl-qwen3-14b-q4km",
            ])

    def test_group_json_is_valid_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_group_json(Path(temp), model="human", status="complete", value=1)
            self.assertEqual(json.loads(path.read_text())["value"], 1)
            write_group_json(Path(temp), model="human", status="failed", value=2)
            self.assertEqual(json.loads(path.read_text())["status"], "failed")
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_report_update_preserves_surrounding_sections(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "analysis_report_zh.md"
            report.write_text("# Report\n\n## Before\nkeep\n\n## Bayesian 状态\n\npending\n\n## After\nkeep-after\n", encoding="utf-8")
            frame = pd.DataFrame([{"model": "human", "status": "missing"}])
            update_report_bayesian(report, frame)
            text = report.read_text(encoding="utf-8")
            self.assertIn("## Before\nkeep", text)
            self.assertIn("## After\nkeep-after", text)
            self.assertIn("`human`：`missing`", text)


if __name__ == "__main__":
    unittest.main()
