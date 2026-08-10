# SOCRATES README Results Update Design

## Goal

Replace the repository's preliminary, English-only README with a concise bilingual project overview grounded in the completed experiment. The README should let a new reader understand the research question, experimental design, main findings, limitations, repository contents, and reproduction path without overstating what the study proves.

## Audience and language

- Primary audience: research mentors, students, reviewers, and developers visiting the public GitHub repository.
- Use section-level bilingual presentation: English first, followed by concise Chinese.
- Keep commands, file trees, metric notation, and model identifiers single-copy to avoid unnecessary duplication.

## Content structure

1. Bilingual title and one-paragraph project positioning.
2. A compact “Results at a glance / 核心结果” table covering sample size, model coverage, accuracy, ECE, M-ratio, hallucination-item performance, and hypothesis adjudication.
3. Four tracked result figures with bilingual captions:
   - `experiment/results/figures/c_accuracy_by_type.png`
   - `experiment/results/figures/c_mratio.png`
   - `experiment/results/figures/c_hallucination_acc.png`
   - `experiment/results/figures/c_errors_ceiling.png`
4. Bilingual research question, study design, and scope boundary.
5. Concise repository layout, setup, model-run, and questionnaire-generation instructions.
6. Bilingual limitations and final project status.

## Evidence and wording constraints

- Treat `experiment/results/analysis/stats_digest.md` and `papers/Final_Paper_EN.md` as the numerical authorities.
- Report 46 human participants, 1,647 valid human responses, and 20 model configurations.
- State the primary language limitation: humans answered Chinese items and models answered English items.
- Describe findings as behavioral confidence-calibration evidence. Do not claim the study demonstrates an internal metacognitive mechanism in LLMs.
- Explain that M-ratio is reliable only for groups with enough errors; do not interpret near-ceiling model estimates as evidence of strong metacognition.
- Use aggregate human results only. Do not expose participant-level data, identities, API keys, or private local paths.

## Git scope and validation

- README implementation scope: `README.md` only; reuse figures already tracked by Git.
- Exclude untracked archives, poster assets, defense materials, and the current unstaged `memory.md` change.
- Validate all referenced image paths and local Markdown links.
- Cross-check each numerical claim against the statistics digest and final paper.
- Run `git diff --check -- README.md`, inspect the scoped diff, and verify staged files before committing.
- Push the current `main` branch. The push will include the existing local commit `c1eb65a` plus the new README commit.

## Success criteria

- GitHub renders the README and all four figures without broken paths.
- English and Chinese readers can identify the question, design, results, limitations, and reproduction steps.
- The README no longer describes data collection or model evaluation as still in progress.
- No unrelated or sensitive files enter the README commit.
