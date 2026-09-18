import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment.replication_analysis import (
    data_quality_table,
    load_human_trials,
    load_model_attempts,
    validate_model_attempts,
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
        all_rows, valid = load_model_attempts(write_csv(attempts))
        self.assertEqual(len(all_rows), 9600)
        self.assertEqual(len(valid), 9599)
        self.assertNotIn(all_rows.iloc[0].name, valid.index)

    def test_validation_rejects_duplicate_formal_key(self):
        attempts = make_model_attempts()
        attempts.iloc[1] = attempts.iloc[0]
        with self.assertRaisesRegex(ValueError, "unique task keys"):
            validate_model_attempts(attempts)

    def test_human_loader_returns_only_valid_chinese_human_rows(self):
        clean = load_human_trials(write_csv(make_human_master()))
        self.assertTrue((clean["model"] == "human").all())
        self.assertTrue((clean["language"] == "zh").all())
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

