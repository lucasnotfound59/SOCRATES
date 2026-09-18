# 中文复现分析报告

## 数据质量

模型输入审计共 9600 条尝试，其中 9597 条进入指标分母；质量表保留全部原始记录，无效/非响应记录共 3 条；质量表保留这些记录并单独标记，而不是静默删除，具体类别见 `data_quality.csv`。该数量与协议预期的三条非响应记录一致。

## 六模型准确率与校准

按预设的家族/规模顺序展示六个模型；准确率最高的是 repl-gemma4-26b-a4b-qat，ECE 最低的是 repl-gemma4-26b-a4b-qat。六模型平均准确率为 0.954，平均 ECE 为 0.053。这些排名只描述观测行为，不等同于模型内部过程。
完整准确率排名（高到低）：repl-gemma4-26b-a4b-qat > repl-qwen3-14b-q4km > repl-qwen3-4b-q4km > repl-gemma4-e2b-q4km > repl-gemma4-e4b-q4km > repl-qwen3-1.7b-q8。
完整 ECE 排名（低到高）：repl-gemma4-26b-a4b-qat > repl-qwen3-4b-q4km > repl-qwen3-14b-q4km > repl-gemma4-e4b-q4km > repl-gemma4-e2b-q4km > repl-qwen3-1.7b-q8。

## 中文人类与模型比较

在相同中文任务与五档置信度编码下，人类行的准确率为 0.735、ECE 为 0.086；六模型平均值分别为 0.954 和 0.053。人类记录按参与者聚类，模型记录按题目聚类，因此该对照用于描述性比较，不应被解释为完全相同抽样结构下的因果差异。

## 题型、难度与幻觉

分层表中观测到的最高题型-模型组合为：repl-gemma4-26b-a4b-qat 在 常规 上准确率 1.000；最高难度-模型组合为：repl-gemma4-26b-a4b-qat 在 易 上准确率 1.000。完整难度分层结果见 `by_difficulty.csv`；幻觉题 SDT 的最高 d′ 组合为：repl-gemma4-26b-a4b-qat 的 d′=5.236；准确率差异最大的组合为：human 的真实-虚构准确率差为 0.462。虚构命题与真实冷僻命题的准确率、置信度差异见`hallucination_breakdown.csv`，SDT 校正值见 `hallucination_sdt.csv`。

## M-ratio 结果与证据等级

- `data-driven`：human（M-ratio=1.370，错误数=437）, repl-gemma4-e2b-q4km（M-ratio=0.152，错误数=92）, repl-gemma4-e4b-q4km（M-ratio=0.259，错误数=94）, repl-qwen3-1.7b-q8（M-ratio=0.344，错误数=171）, repl-qwen3-4b-q4km（M-ratio=0.458，错误数=55）；可作数据驱动比较。
- `regularized`：repl-qwen3-14b-q4km（M-ratio=0.527，错误数=23）；仅作透明报告，不据此下元认知强弱结论。
- `prior-dominated`：repl-gemma4-26b-a4b-qat（M-ratio=0.920，错误数=5）；仅作透明报告，不据此下元认知强弱结论。

MLE 点估计和聚类 bootstrap 区间均保留；M-ratio bootstrap 区间仅在数据驱动且拟合可估计的组中报告，regularized/prior-dominated 组的区间留空；接近满分的组可能没有稳定的错误结构，因此这些组不会被包装成确定的元认知结论。

## Bayesian 状态

Bayesian HMeta-d 已处理 7 组；失败 0 组；缺失 0 组。
证据等级仍只由错误数决定：`data-driven`（≥30）、`regularized`（10–29）、`prior-dominated`（<10）。
可靠性另外要求错误数至少 10、R-hat < 1.05、发散比例 < 5%，且 95% HDI 边界有限；不满足时仅透明报告，不作强结论。
- `human`：M-ratio=1.337，95% HDI [1.147, 1.519]，R-hat=1.000，发散=0，evidence=`data-driven`，reliable=`True`。
- `repl-gemma4-e2b-q4km`：M-ratio=0.367，95% HDI [0.187, 0.544]，R-hat=1.006，发散=20，evidence=`data-driven`，reliable=`True`。
- `repl-gemma4-e4b-q4km`：M-ratio=0.399，95% HDI [0.241, 0.550]，R-hat=1.002，发散=96，evidence=`data-driven`，reliable=`False`。
- `repl-gemma4-26b-a4b-qat`：M-ratio=1.079，95% HDI [0.900, 1.244]，R-hat=1.007，发散=563，evidence=`prior-dominated`，reliable=`False`。
- `repl-qwen3-1.7b-q8`：M-ratio=0.405，95% HDI [0.245, 0.544]，R-hat=1.004，发散=23，evidence=`data-driven`，reliable=`True`。
- `repl-qwen3-4b-q4km`：M-ratio=0.539，95% HDI [0.409, 0.654]，R-hat=1.002，发散=10，evidence=`data-driven`，reliable=`True`。
- `repl-qwen3-14b-q4km`：M-ratio=0.650，95% HDI [0.519, 0.782]，R-hat=1.002，发散=274，evidence=`regularized`，reliable=`False`。

## 局限性

- 人类与模型具有不等的 sampling structures（人类按参与者、模型按题目聚类）。
- ceiling effects 会令近满分组的 M-ratio 估计不稳定。
- quantization/model-family confounding 使规模趋势不能单独归因于参数量。
- 所有结果都是 behavioral-only interpretation，不推断不可观测的内部机制。
- 本研究不作 consciousness claim，也不把置信度行为等同于意识。
