# SOCRATES

**S**elf-knowledge **O**f **C**onfidence: **R**ating **A**nd **T**esting **E**pistemic **S**ensitivity — a symmetric comparison of **metacognitive calibration in humans and large language models**.

> *"The only thing I know is that I know nothing."* — the question this project asks empirically: when a human and an LLM answer the same questions and each report how confident they are, how similar is their ability to **know what they don't know**, and where does it diverge?

---

## What this is

An empirical study that puts **humans and LLMs as parallel respondents** on the exact same set of True/False questions. Each party gives an answer **and** a confidence rating, using an identical 5-point scale. We then compare, at the behavioral level:

- **Calibration** — how well stated confidence matches actual accuracy (Expected Calibration Error).
- **Metacognitive efficiency** — `meta-d′ / d′` (M-ratio), which controls for raw ability so a strong and a weak respondent can be compared fairly.
- **Where they diverge** — broken down by question type and difficulty, with special attention to hallucination-triggering items.

This is a framework-and-adjudication study; it does **not** propose a new model. It measures behavior, and is explicit that behavioral calibration is not the same as internal metacognition (see the paper's Discussion).

---

## Repository layout

```
.
├── 题库_ItemBank_v2_400.xlsx     # The item bank: 400 bilingual T/F items
├── memory.md                    # Project context / decisions log
├── experiment/
│   ├── run_experiment.py        # Model runner (local + API, stateless, resumable)
│   ├── sample_items.py          # Human questionnaire sampler (balanced by type × difficulty)
│   ├── models.yaml              # Run config + model list
│   ├── standard_prompt.md       # The fixed prompt (human & model versions, bilingual)
│   ├── models_to_test.md        # Checklist of models across vendors
│   ├── README_LMStudio.md       # Local-model (LM Studio) setup guide
│   ├── requirements.txt
│   ├── results/                 # Collected model responses (CSV)
│   └── human_samples/           # Generated participant questionnaires + answer key
└── .gitignore                   # Ignores .env (keys), private notes, junk
```

---

## The item bank

`题库_ItemBank_v2_400.xlsx` — **400 items**, bilingual (中文 / English), objective True/False.

| Type | Count | What it probes |
|---|---|---|
| Ordinary (常规) | 100 | Everyday & encyclopedic knowledge; famous myths as a hard variant |
| Discipline (学科) | 100 | Physics / Biology / Statistics / History / Economics |
| Trap (陷阱) | 100 | Original pseudoscience items (+ a few "sounds-like-a-myth-but-true" decoys) |
| Hallucination (幻觉) | 100 | Fictional entities (answer = False) mixed with real-but-obscure facts (answer = True) |

Each item is difficulty-stratified (易/中/难) and the answer key is balanced (≈176 True / 224 False). Trap and hallucination items are original to avoid training-data contamination.

---

## Setup

```bash
cd experiment
python3 -m pip install -r requirements.txt   # openai, pyyaml, openpyxl
```

**API keys** go in `experiment/.env` (auto-loaded, git-ignored — never committed):

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
GEMINI_API_KEY=
```

**Local models** are served through [LM Studio](https://lmstudio.ai)'s OpenAI-compatible endpoint (`http://localhost:1234/v1`). See `experiment/README_LMStudio.md`.

---

## Running the model experiment

A single OpenAI-compatible client drives both local (LM Studio) and hosted API models — only `base_url` / `model_id` differ. Each question is an **independent, stateless request** (no conversation history), sampled multiple times per model.

```bash
# Parser self-test (no model calls)
python3 run_experiment.py --selftest

# Small validation run first
python3 run_experiment.py --models deepseek-v4-flash --output-file results_deepseek.csv --limit 5

# Full run for one model
python3 run_experiment.py --models deepseek-v4-flash --output-file results_deepseek.csv --concurrency 8

# Check completion status without running
python3 run_experiment.py --models local-gemma-e2b --status

# Re-run only failed / empty cells (clean, backs up .bak)
python3 run_experiment.py --models local-gemma-e2b --retry-failed
```

Key behaviors:
- **Resumable** — reruns skip already-completed `(model, language, item, sample)` cells; interrupt and continue freely.
- **Completeness check** — reports `done / expected` per model and whether the whole set is finished. By default only *valid* responses (parsed + no error) count as done.
- **Records** the model that actually served each response (`served_model`), reasoning-token count, and finish reason — so silent mislabeling or truncation is detectable.
- **Local models: run one at a time** with `--models`, matching what's loaded in LM Studio.

Config (`models.yaml`) — current defaults: English-only, `n_samples: 4`, `concurrency: 5`, `max_tokens: 1024`.

---

## Generating human questionnaires

`sample_items.py` draws a per-participant subset that is **uniform across every (type × difficulty) cell**, so each participant sees a balanced spread. Type/difficulty/answers are **not** shown to participants (kept in a separate answer key).

```bash
# 8 English questionnaires, 3 items per cell (= 36 items each), reproducible
python3 sample_items.py --n 8 --per-cell 3 --language en --seed 42
```

Outputs to `human_samples/`: `participant_XX.xlsx` (the form), `answer_key.csv` (for scoring), and `sampling_report.txt`.

---

## Method highlights

- **Confidence scale (identical for humans & models):** 1 pure guess · 2 not very sure · 3 half-and-half · 4 fairly sure · 5 very sure, reported *after* the answer (retrospective).
- **ECE mapping:** confidence levels map to 50 / 62.5 / 75 / 87.5 / 100 % — the floor is 50 % because a random guess on a T/F item is right half the time.
- **Group-level meta-d′** via hierarchical Bayesian estimation (HMeta-d), not per-participant (each person answers too few items).
- **Reasoning models** are run in their native (thinking) mode with a large token budget; reasoning length is logged as a covariate.

---

## Preliminary observation

On this bank, capable models sit near the accuracy ceiling (e.g., a frontier model at ≈99.5 % with **zero** high-confidence errors), which limits calibration/meta-d′ estimation at the strong end. The one signal that survives across models is a **confidence dip on fictional (hallucination) items** — early evidence of boundary awareness. Humans, by contrast, make enough errors for their calibration to be measurable — so recruiting a **range of human ability** (not just top performers) is important.

---

## Security

`experiment/.env` holds API keys and is git-ignored. **Never commit it.** Verify with `git status` before every commit.

## Status

Item bank complete · local Gemma-4 gradient (E2B/E4B/26B-A4B) collected · API models in progress · human data collection starting.
