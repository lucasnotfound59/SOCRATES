# Bilingual SOCRATES README Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the preliminary English-only README with a concise bilingual overview of the completed SOCRATES experiment and publish the verified update to GitHub.

**Architecture:** Keep `README.md` as the single reader-facing entry point and reuse four result figures already tracked under `experiment/results/figures/`. Source every numerical claim from the statistics digest and final paper, retain the existing reproduction commands, and separate behavioral findings from claims about internal metacognition.

**Tech Stack:** GitHub-flavored Markdown, existing PNG result figures, local Git, GitHub remote `origin`.

## Global Constraints

- Use section-level bilingual presentation: English first, followed by concise Chinese.
- Keep commands, file trees, metric notation, and model identifiers single-copy.
- Treat `experiment/results/analysis/stats_digest.md` and `papers/Final_Paper_EN.md` as numerical authorities.
- Report only aggregated human results: 46 participants and 1,647 valid responses.
- Report 20 model configurations and identify the human-Chinese/model-English comparison as the primary limitation.
- Describe behavioral confidence calibration only; do not infer an internal LLM metacognitive mechanism.
- Interpret M-ratio only where error counts make it estimable; preserve the ceiling-effect caveat.
- Do not expose participant-level data, identities, API keys, or private local paths.
- Do not stage the unstaged `memory.md` change, archives, poster assets, defense materials, or other unrelated files.

---

### Task 1: Rewrite the reader-facing README

**Files:**
- Modify: `README.md`
- Read: `experiment/results/analysis/stats_digest.md`
- Read: `papers/Final_Paper_EN.md`
- Reuse: `experiment/results/figures/c_accuracy_by_type.png`
- Reuse: `experiment/results/figures/c_mratio.png`
- Reuse: `experiment/results/figures/c_hallucination_acc.png`
- Reuse: `experiment/results/figures/c_errors_ceiling.png`

**Interfaces:**
- Consumes: validated aggregate statistics and existing Git-tracked PNG paths.
- Produces: one GitHub-renderable bilingual `README.md` with no new image assets.

- [ ] **Step 1: Reconfirm source facts before editing**

Run:

```bash
sed -n '1,220p' experiment/results/analysis/stats_digest.md
rg -n '46|1,647|20 model|0\.735|0\.086|1\.37|0\.269|0\.85|language|ceiling' papers/Final_Paper_EN.md
```

Expected: the digest and paper agree on the participant count, response count, group count, headline estimates, language limitation, and ceiling limitation.

- [ ] **Step 2: Replace the preliminary README with the approved bilingual structure**

Write these sections in order:

1. `# SOCRATES` with the expanded project name and a bilingual one-paragraph scope statement.
2. `## Results at a glance / 核心结果` with a compact table containing:
   - humans: 46 participants, 1,647 valid Chinese-questionnaire responses, accuracy 0.735;
   - models: 20 English-only configurations, most accuracy 0.97–1.00;
   - ECE: humans 0.086 [0.062, 0.113], significantly above 17/19 models by difference bootstrap;
   - M-ratio: humans 1.37 [1.21, 1.54], measurable weak models 0.31–0.69 with non-overlapping intervals;
   - fictional hallucination items: humans 0.269, models 0.85–1.00;
   - verdict: H1 rejected, H2 rejected within the measurable range, H3 reversed/not supported.
3. A short synthesis stating that human and model failure modes differ in kind.
4. Four figure blocks using relative paths and bilingual captions:

```markdown
![Accuracy by item type / 按题型准确率](experiment/results/figures/c_accuracy_by_type.png)
![Metacognitive efficiency / 元认知效率](experiment/results/figures/c_mratio.png)
![Fictional-item accuracy / 虚构题准确率](experiment/results/figures/c_hallucination_acc.png)
![Error-count ceiling / 错误数天花板](experiment/results/figures/c_errors_ceiling.png)
```

5. `## Research question and scope / 研究问题与范围` defining the behavioral construct and its internal-mechanism boundary.
6. `## Study design / 实验设计` covering 400 bilingual True/False items, four item types, three difficulty levels, the shared five-level retrospective confidence scale, four model samples per item, ECE, and M-ratio.
7. `## Repository layout / 仓库结构` reflecting currently tracked analysis and result directories.
8. `## Reproduction / 复现` retaining parser self-test, validation run, full run, status, retry, and questionnaire-generation commands.
9. `## Limitations / 局限` covering language confounding, model ceiling, behavioral-only inference, N=46 single-population human sample, and incomplete vendor coverage.
10. `## Status / 状态` marking data collection, core analysis, robustness checks, paper, defense materials, and posters complete without claiming frozen future-work ideas are approved.
11. `## Security and privacy / 安全与隐私` reiterating that `.env` and participant-level data must not be committed.

Expected: the old “Preliminary observation” and “API models in progress · human data collection starting” wording is absent.

- [ ] **Step 3: Check structure, links, and stale wording**

Run:

```bash
rg -n '^## ' README.md
rg -n 'Preliminary|in progress|starting|≈99\.5|zero high-confidence errors' README.md
test -f experiment/results/figures/c_accuracy_by_type.png
test -f experiment/results/figures/c_mratio.png
test -f experiment/results/figures/c_hallucination_acc.png
test -f experiment/results/figures/c_errors_ceiling.png
```

Expected: bilingual sections are present, stale wording returns no matches, and all four `test -f` commands exit 0.

- [ ] **Step 4: Validate the Markdown diff**

Run:

```bash
git diff --check -- README.md
git diff -- README.md
```

Expected: `git diff --check` exits 0; the diff changes only `README.md`, uses aggregate data, and contains no secret or participant-level material.

### Task 2: Commit and publish the scoped update

**Files:**
- Stage: `README.md`
- Do not stage: every other modified or untracked path.

**Interfaces:**
- Consumes: the validated `README.md` from Task 1.
- Produces: one scoped README commit pushed to `origin/main`, together with the already-local commits preceding it.

- [ ] **Step 1: Verify GitHub tooling and repository state**

Run:

```bash
gh --version
gh auth status
git status --short --branch
git log --oneline origin/main..HEAD
git remote get-url origin
```

Expected: `gh` is installed and authenticated; the remote is `https://github.com/lucasnotfound59/SOCRATES`; unrelated worktree files remain visible but unstaged.

- [ ] **Step 2: Stage only the README and verify the index**

Run:

```bash
git add -- README.md
git diff --cached --check
git diff --cached --name-only
git diff --cached --stat
```

Expected: the staged name list contains exactly `README.md`, and the cached whitespace check exits 0.

- [ ] **Step 3: Commit the README update**

Run:

```bash
git commit -m "docs: publish bilingual experiment results"
```

Expected: one commit is created containing only `README.md`.

- [ ] **Step 4: Reconfirm commit scope and push**

Run:

```bash
git show --stat --oneline --decorate --no-renames HEAD
git status --short --branch
git push -u origin main
```

Expected: the latest commit contains only `README.md`; unrelated files remain uncommitted; `origin/main` advances through `c1eb65a`, the approved design/plan commits, and the README commit.

- [ ] **Step 5: Verify the remote branch**

Run:

```bash
git fetch origin main
git rev-parse HEAD
git rev-parse origin/main
```

Expected: the two commit hashes are identical.
