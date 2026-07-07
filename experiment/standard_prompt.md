# 标准 Prompt 模板 / Standard Prompt Template

> 用于元认知校准研究。核心原则:**人类版与模型版的任务措辞与 5 档置信度用词完全一致**,以保证行为层对称可比。模型版仅额外加一行机器可解析的输出格式说明(人类版换成"选择"),这不改变任务本身。
>
> 关键约束:①判断题 True/False;②作答后(retrospective)报告置信度;③5 档 verbalized;④每题独立、**不保留上下文记忆**(每次调用只含 system+当题 user);⑤固定模板,`{statement}` 为唯一变量。

---

## 1. 模型版 — 英文 / Model version — English

**System:**
```
You are answering a series of independent True/False questions. For each statement, first decide whether it is True or False, and then report how confident you are in that answer. Rely only on your own knowledge. You will not be told whether you are correct, and each question is independent of the others.
```

**User:**
```
Statement: {statement}

First decide whether the statement is True or False.
Then rate your confidence in that answer using exactly one of these five levels:
1 = pure guess
2 = not very sure
3 = half-and-half
4 = fairly sure
5 = very sure

Respond with ONLY a raw JSON object — no markdown, no code fences, no extra text — in exactly this format:
{"answer": "True or False", "confidence": 1}
```

---

## 2. 模型版 — 中文 / Model version — Chinese

**System:**
```
你正在回答一系列相互独立的判断题。对每个陈述,请先判断它是"对(True)"还是"错(False)",然后报告你对这个答案的确信程度。仅依据你自己的知识作答。你不会被告知答案是否正确,每道题之间相互独立。
```

**User:**
```
陈述:{statement}

请先判断该陈述是"对(True)"还是"错(False)"。
然后用以下五档中的恰好一档,报告你的确信程度:
1 = 完全靠猜 (pure guess)
2 = 不太确定 (not very sure)
3 = 一半一半 (half-and-half)
4 = 比较确定 (fairly sure)
5 = 非常确定 (very sure)

只输出纯 JSON 对象 —— 不要 markdown、不要代码围栏、不要任何多余文字 —— 严格采用如下格式:
{"answer": "True 或 False", "confidence": 1}
```

---

## 3. 人类版 / Human version（问卷用,措辞与模型版一致)

题目与 5 档用词与上方**完全相同**,仅把"输出 JSON"替换为"勾选":

> 陈述:{statement}
>
> 你的判断:☐ 对(True)　☐ 错(False)
>
> 你的确信程度(勾选一项):
> ☐ 1 完全靠猜　☐ 2 不太确定　☐ 3 一半一半　☐ 4 比较确定　☐ 5 非常确定

英文问卷同理,把上方英文 5 档原样列为勾选项。

---

## 4. 置信度 → ECE 数值映射(分析阶段用,不出现在 prompt 中)

| 档位 | 用词 | 映射置信度 |
|---|---|---|
| 1 | pure guess / 完全靠猜 | 50% |
| 2 | not very sure / 不太确定 | 62.5% |
| 3 | half-and-half / 一半一半 | 75% |
| 4 | fairly sure / 比较确定 | 87.5% |
| 5 | very sure / 非常确定 | 100% |

下限取 50%:判断题瞎猜期望正确率为 50%(见论文 §3.3)。

---

## 5. 设计说明 / Notes

- **retrospective**:JSON 中 `answer` 在前、`confidence` 在后,要求模型先定答案再评估信心,与人类"先答后评"一致。
- **对称性**:人类只能主观报告,故模型也用 verbalized 主观报告而非 token 概率。token 概率作为**辅助分析**单独收集(见 `run_experiment.py` 的 `capture_logprobs`),不进主对比。
- **prompt 敏感性控制**:所有模型、所有题共用此固定模板,不因题而变措辞。
- **语言**:中、英各跑一遍;被试母语版本用于与人类对比,跨语言差异作为额外分析。
- `{statement}`:对应题库 xlsx 的"题干(中文)"或"Statement (EN)"列。
