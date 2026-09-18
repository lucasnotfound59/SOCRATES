import json
import tempfile
import unittest
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from experiment.replication_hmetad import (
    aggregate_results,
    pending_models,
    run_group,
    summarize_idata,
    write_group_json,
)


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


class HMetaSummaryTests(unittest.TestCase):
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

    def test_completed_group_is_reused_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            group_dir = Path(temp)
            write_group_json(group_dir, model="human", status="complete")
            pending = pending_models(["human", "model-a"], group_dir)
            self.assertEqual(pending, ["model-a"])


class HMetaArtifactTests(unittest.TestCase):
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

    def test_group_json_is_valid_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_group_json(Path(temp), model="human", status="complete", value=1)
            self.assertEqual(json.loads(path.read_text())["value"], 1)
            write_group_json(Path(temp), model="human", status="failed", value=2)
            self.assertEqual(json.loads(path.read_text())["status"], "failed")
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
