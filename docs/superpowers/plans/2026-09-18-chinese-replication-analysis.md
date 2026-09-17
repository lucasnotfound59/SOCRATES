# SOCRATES Chinese Replication Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an isolated, reproducible analysis pipeline for the six Chinese open-weight model replications, then produce both a model-only analysis and a Chinese-human-versus-model comparison without changing the original paper analysis outputs.

**Architecture:** Add one focused core module for data validation, metrics, clustered bootstrap, tables, figures, and report assembly, plus one resumable Bayesian runner. Both consume the archived replication CSV and the existing private human master table, and write only under `results/replication_zh_2026-09-18/analysis/`.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, matplotlib, metadpy, PyMC, ArviZ, unittest.

## Global Constraints

- Do not modify or overwrite `experiment/results/analysis/`, `experiment/results/figures/`, or `experiment/results/master_long.csv`.
- The formal model input must contain exactly 9,600 unique `model × language × item_id × sample_idx` task keys and exactly 1,600 attempts per model.
- Only valid parsed responses enter metric denominators; every invalid response remains represented in `data_quality.csv`.
- Human outputs must be aggregate-only and must never contain `participant_id` or individual-level rows.
- Confidence mapping is exactly `{1: 0.50, 2: 0.625, 3: 0.75, 4: 0.875, 5: 1.00}`.
- Human bootstrap clusters are participants; model bootstrap clusters are item IDs.
- Evidence tiers are `data-driven` for at least 30 errors, `regularized` for 10–29 errors, and `prior-dominated` below 10 errors.
- Bayesian HMeta-d is auxiliary; near-ceiling estimates cannot be described as evidence that a model possesses metacognition or consciousness.
- Use a fixed random seed of `20260918` and record bootstrap/sample counts in outputs.
- Generated analysis artifacts remain uncommitted until the user reviews them; code, tests, and documentation may be committed in narrowly scoped commits.

---

## File Structure

- Create `experiment/replication_analysis.py`: input contracts, cleaning, descriptive metrics, MLE meta-d′, clustered bootstrap, output tables, figures, Markdown tables, report, and CLI.
- Create `experiment/replication_hmetad.py`: resumable per-group Bayesian HMeta-d runs, diagnostics, and aggregation.
- Create `experiment/tests/test_replication_analysis.py`: deterministic unit and integration tests on synthetic data; no human source rows leave temporary memory/files.
- Create `experiment/tests/test_replication_hmetad.py`: Bayesian result summarization and resume-contract tests with sampler injection; tests do not run MCMC.
- Create `results/replication_zh_2026-09-18/analysis/`: generated CSV, Markdown, JSON, and figure artifacts; do not commit before review.
- Do not modify the existing English/human pipeline files.

---

### Task 1: Input contracts, validation, and privacy-safe cleaning

**Files:**
- Create: `experiment/replication_analysis.py`
- Create: `experiment/tests/test_replication_analysis.py`

**Interfaces:**
- Produces: `load_model_attempts(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `load_human_trials(path: Path) -> pd.DataFrame`
- Produces: `validate_model_attempts(attempts: pd.DataFrame) -> None`
- Produces: `data_quality_table(attempts: pd.DataFrame) -> pd.DataFrame`
- Clean model and human frames expose `model`, `language`, `item_id`, `sample_idx`, `gold`, `parsed_answer`, `correct`, `conf`, `stated`, `type`, `subtype`, `difficulty`, and internal `cluster_id`.

- [ ] **Step 1: Write failing input-contract tests**

Add synthetic helpers and tests that establish the key behavior:

```python
class ReplicationInputTests(unittest.TestCase):
    def test_model_loader_keeps_invalid_attempts_out_of_metrics(self):
        attempts = make_model_attempts(models=6, items=400, samples=4)
        attempts.loc[0, "parse_ok"] = False
        attempts.loc[0, "parsed_answer"] = ""
        all_rows, valid = load_model_attempts(write_csv(attempts))
        self.assertEqual(len(all_rows), 9600)
        self.assertEqual(len(valid), 9599)
        self.assertNotIn(all_rows.iloc[0].name, valid.index)

    def test_validation_rejects_duplicate_formal_key(self):
        attempts = make_model_attempts(models=6, items=400, samples=4)
        attempts.iloc[1] = attempts.iloc[0]
        with self.assertRaisesRegex(ValueError, "unique task keys"):
            validate_model_attempts(attempts)

    def test_human_loader_returns_only_valid_chinese_human_rows(self):
        human = make_human_master()
        clean = load_human_trials(write_csv(human))
        self.assertTrue((clean["model"] == "human").all())
        self.assertTrue((clean["language"] == "zh").all())
        self.assertNotIn("participant_id", clean.columns)
        self.assertIn("cluster_id", clean.columns)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis.ReplicationInputTests -v
```

Expected: import failure because `experiment.replication_analysis` does not exist.

- [ ] **Step 3: Implement constants and strict normalization**

Implement these contracts in `experiment/replication_analysis.py`:

```python
CONFIDENCE_MAP = {1: 0.50, 2: 0.625, 3: 0.75, 4: 0.875, 5: 1.00}
FORMAL_KEY = ["model", "language", "item_id", "sample_idx"]
EXPECTED_MODELS = 6
EXPECTED_ATTEMPTS_PER_MODEL = 1600
EXPECTED_TOTAL_ATTEMPTS = 9600
RANDOM_SEED = 20260918

def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}

def evidence_tier(n_wrong: int) -> str:
    if n_wrong >= 30:
        return "data-driven"
    if n_wrong >= 10:
        return "regularized"
    return "prior-dominated"
```

`validate_model_attempts` must reject wrong totals, duplicate keys, non-Chinese rows, a model count other than six, or any per-model attempt count other than 1,600. `load_model_attempts` must preserve all attempts in the first return value and return only rows with valid parse, binary gold/answer, and confidence 1–5 in the second. Set `cluster_id = item_id` for models.

`load_human_trials` must filter `human`, `zh`, and `status == ok`, validate binary answer/gold and confidence, set `cluster_id` from `participant_id`, then remove `participant_id` before returning or writing any public table.

- [ ] **Step 4: Implement the quality table**

Return one row per model with these exact columns:

```python
[
    "model", "n_attempted", "n_valid", "n_invalid", "coverage",
    "n_truncated", "n_api_error", "n_parse_failure"
]
```

Classify `finish_reason == length` as truncated, non-empty `error` as API error, and other invalid attempts as parse failures. Assert `n_valid + n_invalid == n_attempted` for each row.

- [ ] **Step 5: Run tests and the existing dashboard tests**

Run:

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis.ReplicationInputTests \
  experiment.tests.test_progress_dashboard -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit only Task 1 code and tests**

```bash
git add -- experiment/replication_analysis.py experiment/tests/test_replication_analysis.py
git commit -m "feat: validate Chinese replication inputs"
```

---

### Task 2: Descriptive, calibration, hallucination, and SDT tables

**Files:**
- Modify: `experiment/replication_analysis.py`
- Modify: `experiment/tests/test_replication_analysis.py`

**Interfaces:**
- Consumes: cleaned frames from Task 1.
- Produces: `ece(group: pd.DataFrame) -> float`
- Produces: `type2_auroc(group: pd.DataFrame) -> float`
- Produces: `summarize_groups(data: pd.DataFrame) -> pd.DataFrame`
- Produces: `summarize_factor(data: pd.DataFrame, factor: str, levels: Sequence[str]) -> pd.DataFrame`
- Produces: `confidence_distribution(data: pd.DataFrame) -> pd.DataFrame`
- Produces: `reliability_table(data: pd.DataFrame) -> pd.DataFrame`
- Produces: `hallucination_breakdown(data: pd.DataFrame) -> pd.DataFrame`
- Produces: `hallucination_sdt(data: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write metric tests with hand-computable values**

```python
class ReplicationMetricTests(unittest.TestCase):
    def test_ece_uses_binary_task_confidence_mapping(self):
        group = pd.DataFrame({
            "conf": [1, 1, 5, 5],
            "correct": [True, False, True, False],
        })
        self.assertAlmostEqual(ece(group), 0.25)

    def test_type2_auroc_rewards_higher_confidence_on_correct_trials(self):
        group = pd.DataFrame({
            "conf": [5, 4, 2, 1],
            "correct": [True, True, False, False],
        })
        self.assertAlmostEqual(type2_auroc(group), 1.0)

    def test_evidence_tiers_match_registered_thresholds(self):
        self.assertEqual(evidence_tier(30), "data-driven")
        self.assertEqual(evidence_tier(10), "regularized")
        self.assertEqual(evidence_tier(9), "prior-dominated")
```

- [ ] **Step 2: Run the new tests and verify metric functions are missing**

Run:

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis.ReplicationMetricTests -v
```

Expected: failure naming the first undefined metric function.

- [ ] **Step 3: Implement group and factor summaries**

`summarize_groups` must return deterministic rows sorted as `human` first, then model family and numeric size. Include exactly:

```python
[
    "model", "n", "accuracy", "mean_confidence", "mean_stated",
    "overconfidence", "ece", "type2_auroc", "n_wrong",
    "conf5_n", "conf5_accuracy", "conf5_wrong", "evidence_tier"
]
```

`summarize_factor` must emit long-form rows with `model`, factor name, `n`, `accuracy`, `mean_confidence`, `overconfidence`, and `ece`. Produce separate calls for type levels `常规/学科/陷阱/幻觉` and difficulty levels `易/中/难`.

- [ ] **Step 4: Implement reliability and hallucination analyses**

`reliability_table` emits one row per observed model/confidence bin with `model`, `confidence`, `stated`, `accuracy`, and `n`.

`hallucination_breakdown` emits fictional (`gold == False`) and real-obscure (`gold == True`) counts, accuracy, confidence, and their gaps. `hallucination_sdt` reuses the original log-linear correction:

```python
hit_rate = (hits + 0.5) / (n_fictional + 1)
false_alarm_rate = (false_alarms + 0.5) / (n_real + 1)
dprime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(false_alarm_rate))
```

- [ ] **Step 5: Run all analysis unit tests**

Run:

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis -v
```

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add -- experiment/replication_analysis.py experiment/tests/test_replication_analysis.py
git commit -m "feat: add Chinese replication summary metrics"
```

---

### Task 3: MLE meta-d′ and clustered bootstrap

**Files:**
- Modify: `experiment/replication_analysis.py`
- Modify: `experiment/tests/test_replication_analysis.py`

**Interfaces:**
- Consumes: cleaned combined data from Task 1 and metrics from Task 2.
- Produces: `fit_metad(group: pd.DataFrame) -> dict[str, float | int | str | bool]`
- Produces: `cluster_bootstrap(group: pd.DataFrame, nboot_metrics: int, nboot_mratio: int, seed: int, fit_fn: Callable = fit_metad) -> dict[str, float | int]`
- Produces: `bootstrap_all(data: pd.DataFrame, ...) -> pd.DataFrame`

- [ ] **Step 1: Write tests for evidence labeling and cluster resampling**

```python
class ReplicationBootstrapTests(unittest.TestCase):
    def test_bootstrap_resamples_whole_clusters(self):
        group = make_clustered_group(cluster_sizes=[4, 4, 4])
        observed = []
        def fake_fit(sample):
            observed.append(sample.groupby("cluster_id").size().tolist())
            return {"m_ratio": 0.75}
        cluster_bootstrap(group, nboot_metrics=4, nboot_mratio=4,
                          seed=20260918, fit_fn=fake_fit)
        self.assertTrue(all(all(size % 4 == 0 for size in draw) for draw in observed))

    def test_fit_result_carries_error_count_and_evidence_tier(self):
        result = fit_metad(make_metad_group(n_wrong=29))
        self.assertEqual(result["n_wrong"], 29)
        self.assertEqual(result["evidence_tier"], "regularized")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis.ReplicationBootstrapTests -v
```

Expected: failure because `fit_metad` and `cluster_bootstrap` are undefined.

- [ ] **Step 3: Implement MLE using the existing metadpy contract**

Use this exact input conversion:

```python
metad_input = pd.DataFrame({
    "Stimuli": (group["gold"] == "True").astype(int),
    "Accuracy": group["correct"].astype(int),
    "Confidence": group["conf"].astype(int),
})
result = metad(
    data=metad_input,
    nRatings=5,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
)
```

Return `dprime`, `meta_d`, `m_ratio`, `m_diff`, `n`, `n_wrong`, `evidence_tier`, and a non-empty `fit_status`. A fitting failure produces NaN metrics and records the exception class/message; it must not remove the group.

- [ ] **Step 4: Implement deterministic cluster bootstrap**

Sample `cluster_id` values with replacement and concatenate all rows belonging to every sampled cluster. Compute 95% percentile intervals for accuracy, ECE, overconfidence, type-2 AUROC, and M-ratio. Run M-ratio bootstrap only for `data-driven` groups; record `mratio_boot_status = skipped_insufficient_errors` for other tiers.

Use `nboot_metrics=800` and `nboot_mratio=250` in the formal run. Unit tests use four iterations and injected `fit_fn` to stay fast.

- [ ] **Step 5: Run the module tests and MLE smoke test**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis -v
```

Expected: all tests pass and no cache writes target the user home directory.

- [ ] **Step 6: Commit Task 3**

```bash
git add -- experiment/replication_analysis.py experiment/tests/test_replication_analysis.py
git commit -m "feat: add clustered metacognition estimates"
```

---

### Task 4: CLI, artifact generation, figures, and Chinese report

**Files:**
- Modify: `experiment/replication_analysis.py`
- Modify: `experiment/tests/test_replication_analysis.py`

**Interfaces:**
- Consumes: all non-Bayesian tables from Tasks 1–3.
- Produces: `run_analysis(model_path: Path, human_path: Path, output_dir: Path, nboot_metrics: int = 800, nboot_mratio: int = 250) -> dict[str, Path]`
- Produces: required CSVs, `tables.md`, `analysis_report_zh.md`, `run_manifest.json`, and PNG figures.

- [ ] **Step 1: Write an end-to-end temporary-directory test**

```python
class ReplicationArtifactTests(unittest.TestCase):
    def test_run_analysis_writes_isolated_complete_artifacts(self):
        paths = run_analysis(
            model_path=write_csv(make_model_attempts(6, 400, 4)),
            human_path=write_csv(make_human_master()),
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
```

- [ ] **Step 2: Run the artifact test and verify failure**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis.ReplicationArtifactTests -v
```

Expected: failure because `run_analysis` is undefined.

- [ ] **Step 3: Implement CSV, Markdown, JSON, and figure writers**

Write only inside the supplied `output_dir`. `run_manifest.json` must contain input paths, SHA-256 digests, row counts, random seed, bootstrap counts, package versions, start/end timestamps, and every artifact path.

Generate these figures under `analysis/figures/`:

```text
model_accuracy_ece.png
human_model_accuracy_ece.png
accuracy_by_type.png
mratio_evidence.png
reliability_curves.png
hallucination_breakdown.png
```

Plot model estimates in a stable family/size order, place humans first in matched figures, show 95% intervals when available, and visibly distinguish `data-driven`, `regularized`, and `prior-dominated` M-ratio groups.

- [ ] **Step 4: Implement generated narrative with fixed interpretation rules**

`analysis_report_zh.md` must include:

1. data quality and three retained nonresponses;
2. six-model accuracy/calibration ranking;
3. Chinese human/model comparison;
4. type/difficulty and hallucination breakdown;
5. M-ratio results grouped by evidence tier;
6. a Bayesian status section that says `pending` until Task 5 aggregation succeeds;
7. limitations: unequal sampling structures, ceiling effects, quantization/model-family confounding, behavioral-only interpretation, and no consciousness claim.

Generate sentences from table values rather than embedding manually copied numbers. `tables.md` must format the model-only summary, matched summary, per-type accuracy, hallucination SDT, and M-ratio evidence tables.

- [ ] **Step 5: Add CLI and isolated-path guards**

Use these CLI arguments:

```text
--model-results PATH
--human-master PATH
--output-dir PATH
--nboot-metrics INT
--nboot-mratio INT
--seed INT
--resume
--validate-only
```

Reject `output_dir` if it resolves to `experiment/results/analysis` or `experiment/results/figures`. Refuse to overwrite an existing analysis unless `--resume` is passed; on resume, rewrite deterministic non-Bayesian outputs but preserve `hmetad/` per-group files.

- [ ] **Step 6: Run all non-Bayesian tests**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_analysis \
  experiment.tests.test_progress_dashboard -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add -- experiment/replication_analysis.py experiment/tests/test_replication_analysis.py
git commit -m "feat: generate Chinese replication analysis artifacts"
```

---

### Task 5: Resumable Bayesian HMeta-d runner

**Files:**
- Create: `experiment/replication_hmetad.py`
- Create: `experiment/tests/test_replication_hmetad.py`

**Interfaces:**
- Consumes: validated combined data from `experiment.replication_analysis`.
- Produces: `summarize_idata(idata: az.InferenceData) -> dict[str, float | int]`
- Produces: `run_group(group: pd.DataFrame, draws: int, chains: int, seed: int, sampler: Callable = hmetad) -> dict[str, object]`
- Produces: one atomic JSON result per group under `analysis/hmetad/groups/`.
- Produces: `analysis/hmetad_summary.csv` and updates Bayesian section of `analysis_report_zh.md`.

- [ ] **Step 1: Write Bayesian summary and resume tests without sampling**

```python
class HMetaSummaryTests(unittest.TestCase):
    def test_summary_reports_hdi_rhat_divergences_and_evidence(self):
        result = summarize_idata(make_fake_idata(), n_wrong=23)
        self.assertEqual(result["evidence_tier"], "regularized")
        self.assertIn("hdi_lo", result)
        self.assertIn("hdi_hi", result)
        self.assertIn("rhat", result)
        self.assertIn("n_divergent", result)

    def test_completed_group_is_reused_on_resume(self):
        write_group_json(self.group_dir, model="human", status="complete")
        pending = pending_models(["human", "model-a"], self.group_dir)
        self.assertEqual(pending, ["model-a"])
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m unittest \
  experiment.tests.test_replication_hmetad -v
```

Expected: import failure because `experiment.replication_hmetad` does not exist.

- [ ] **Step 3: Create an isolated Python 3.11 analysis environment**

Run:

```bash
uv venv --python /opt/anaconda3/bin/python /private/tmp/socrates-analysis-venv
uv pip install --python /private/tmp/socrates-analysis-venv/bin/python \
  -r experiment/requirements.txt
```

Expected: Python 3.11 environment with importable `metadpy`, `pymc`, and `arviz`. Do not install into the system Python 3.14 interpreter or mutate the Conda base environment.

- [ ] **Step 4: Implement sampler adapter and diagnostics**

Call `metadpy.bayesian.hmetad` with `nRatings=5`, `num_chains=2`, the requested draws, and deterministic seeds. Summarize posterior M-ratio as `meta_d / d1` (or `d` if the posterior uses that name). Store mean, median, SD, 95% HDI, R-hat, divergent count, draws, chains, error count, evidence tier, runtime, and status.

Write each group result first to a temporary sibling file, flush it, then replace the final JSON atomically. A failed group writes `status=failed` and its exception type/message so resume can retry it.

- [ ] **Step 5: Implement CLI modes**

Support:

```text
--model NAME       run one group
--all              run every incomplete group serially
--aggregate        combine completed group JSON files
--resume           skip complete group files
--draws INT        default 800
--chains INT       default 2
```

Aggregation must retain failed/missing groups as explicit rows. Reliability requires at least 10 errors, `rhat < 1.05`, fewer than 5% divergent transitions, and finite HDI bounds; evidence tier remains determined only by error count.

- [ ] **Step 6: Run Bayesian unit tests in the isolated environment**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /private/tmp/socrates-analysis-venv/bin/python -m unittest \
  experiment.tests.test_replication_hmetad -v
```

Expected: all tests pass without running MCMC.

- [ ] **Step 7: Commit Task 5**

```bash
git add -- experiment/replication_hmetad.py experiment/tests/test_replication_hmetad.py
git commit -m "feat: add resumable Bayesian replication analysis"
```

---

### Task 6: Formal analysis run and artifact validation

**Files:**
- Generate: `results/replication_zh_2026-09-18/analysis/**`
- Do not commit generated artifacts before user review.

**Interfaces:**
- Consumes: formal replication CSV and private human master data.
- Produces: the complete artifact set defined in the approved specification.

- [ ] **Step 1: Hash protected old outputs and formal inputs**

Run:

```bash
find experiment/results/analysis experiment/results/figures -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > /private/tmp/socrates-old-analysis-before.sha256
shasum -a 256 \
  results/replication_zh_2026-09-18/results_local_zh_2026-09-18.csv \
  experiment/results/master_long.csv
```

Expected: two input hashes print, and the protected-output snapshot is non-empty.

- [ ] **Step 2: Run the formal non-Bayesian pipeline**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /private/tmp/socrates-analysis-venv/bin/python \
  experiment/replication_analysis.py \
  --model-results results/replication_zh_2026-09-18/results_local_zh_2026-09-18.csv \
  --human-master experiment/results/master_long.csv \
  --output-dir results/replication_zh_2026-09-18/analysis \
  --nboot-metrics 800 \
  --nboot-mratio 250 \
  --seed 20260918
```

Expected: exit code 0, 9,600 attempted model tasks, 9,597 valid model results, and all required non-Bayesian artifacts.

- [ ] **Step 3: Run Bayesian groups resumably**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /private/tmp/socrates-analysis-venv/bin/python \
  experiment/replication_hmetad.py \
  --all --resume --draws 800 --chains 2 \
  --model-results results/replication_zh_2026-09-18/results_local_zh_2026-09-18.csv \
  --human-master experiment/results/master_long.csv \
  --output-dir results/replication_zh_2026-09-18/analysis
```

Expected: one complete or explicit failed JSON per group; rerunning the same command skips complete groups.

- [ ] **Step 4: Aggregate Bayesian results and refresh report**

Run:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /private/tmp/socrates-analysis-venv/bin/python \
  experiment/replication_hmetad.py \
  --aggregate \
  --model-results results/replication_zh_2026-09-18/results_local_zh_2026-09-18.csv \
  --human-master experiment/results/master_long.csv \
  --output-dir results/replication_zh_2026-09-18/analysis
```

Expected: `hmetad_summary.csv` contains seven rows and the report no longer says Bayesian status is pending; failed diagnostics remain explicit.

- [ ] **Step 5: Run artifact and numerical audits**

Run the test suite, then execute the module's `--validate-only` mode:

```bash
XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /private/tmp/socrates-analysis-venv/bin/python -m unittest discover \
  -s experiment/tests -p 'test_*.py' -v

XDG_CACHE_HOME=/private/tmp/socrates-xdg-cache \
MPLCONFIGDIR=/private/tmp/socrates-mpl-cache \
PYTHONPATH=. /private/tmp/socrates-analysis-venv/bin/python \
  experiment/replication_analysis.py \
  --validate-only \
  --model-results results/replication_zh_2026-09-18/results_local_zh_2026-09-18.csv \
  --human-master experiment/results/master_long.csv \
  --output-dir results/replication_zh_2026-09-18/analysis
```

Expected: all tests pass and validation prints `ANALYSIS_AUDIT_OK`.

- [ ] **Step 6: Verify the old pipeline was untouched**

Run:

```bash
find experiment/results/analysis experiment/results/figures -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > /private/tmp/socrates-old-analysis-after.sha256
diff -u \
  /private/tmp/socrates-old-analysis-before.sha256 \
  /private/tmp/socrates-old-analysis-after.sha256
```

Expected: `diff` exits 0 with no output.

- [ ] **Step 7: Inspect every generated figure**

Open the six PNG files and check clipped labels, model ordering, legibility, confidence intervals, evidence-tier styling, and numerical agreement with their source CSVs. Any figure defect is fixed in code, tests rerun, and the complete figure set regenerated.

- [ ] **Step 8: Review conclusions against the evidence tiers**

Read `analysis_report_zh.md` and verify that every numeric claim is traceable to a generated table; `regularized` and `prior-dominated` groups carry explicit caveats; language matching is described accurately; and no sentence infers consciousness or an internal metacognitive mechanism.

- [ ] **Step 9: Present results for user review**

Provide links to `analysis_report_zh.md`, `tables.md`, `human_model_summary.csv`, `metad_summary.csv`, `hmetad_summary.csv`, and the figures directory. Summarize the strongest result, the largest caveat, and any Bayesian diagnostic failures. Do not push or update public-facing artifacts without a new user instruction.
