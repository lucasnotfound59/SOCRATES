# 论文审查报告 — Final_Paper_EN_Chinese_Replication.md

审查日期：2026-09-18
审查对象：`paper_revision_zh_replication/Final_Paper_EN_Chinese_Replication.md`（209 行，GPT 重写版）
对照基准：`papers/Final_Paper_EN.md`（原稿）+ `results/replication_zh_2026-09-18/`（分析产物）
格式要求：可编辑 Word、1.5 倍行距、Times New Roman 12pt、正文 ≤5000 词、APA 7

---

## 1. 数值核验：全部通过

建立 `audit_paper_claims.py`（122 项断言），逐条把论文里的数字回算到源 CSV。**结论：论文中每一个数字都正确。**

| 核验类别 | 结果 |
|---|---|
| 摘要头条数字（ECE .086、acc .735、M-ratio 1.370、d′ .001、c .613） | ✅ |
| Table 1 全表（7 行 × n/acc/ECE/errors/MLE/tier/Bayesian/HDI 上下界） | ✅ |
| 队列均值（中文 acc .954、ECE .053、range .008–.101、errors 55–171） | ✅ |
| Type-2 AUC（人类 .768 [.742,.792]，中文模型 .510–.725） | ✅ |
| 发散率（Gemma E4B 6.0%、Qwen3 14B 17.1%）与 reliable 标志 | ✅ |
| 幻觉子集（人类虚构 .269/真实 .731；中文 d′ 1.58–5.24；c −.99 至 −.07） | ✅ |
| **新声称的 53/47 虚构/真实拆分** | ✅ 与 `题库_ItemBank_v2_400.xlsx` 完全一致（100 题：F=53, T=47） |
| 英文队列（ECE 显著高于 17/19；未显著的正是最弱两个；d′ 1.87–5.62；acc 17/19 ≥.97） | ✅ |
| 人类分题型 acc（.799/.862/.767/.509） | ✅ |

**特别确认**：论文引用的置信区间来源正确且自洽 —— 人类 ECE CI [.062,.113] 与 M-ratio CI [1.205,1.543] 来自**原始英文分析**；中文队列的数字来自**本次复现分析**。两者对同一人类样本给出同量级的区间估计，数值正确。

> 注：`audit_paper_numbers.py`（第一版）报出的 7 处「不符」经复查**全部是审查脚本自身的缺陷**（列表比较、未剔除 prior-dominated 组、用了本次运行的 CI 去比论文引用的原始 CI），不是论文错误。

---

## 2. 格式合规（对照 5000 词要求）

| 区域 | 词数 | 判定 |
|---|---:|---|
| Abstract | 240 | ✅ APA 7 上限 250 |
| Keywords | 12 | ✅ |
| **正文（Related Work → Conclusion，纯散文，剔除表格/图注/标题）** | **3,721** | ✅ **限 5000，余量 1,279 词（26%）** |
| 正文含表格与图注 | 4,152 | ✅ 仍在上限内 |
| References | 460 | ✅ 不计入正文 |
| 全文 | 5,374 | — |

结论：**字数要求已满足，且余量充足，不需要压缩。** 这与原稿（约 11,900 词，超标两倍多）相比是重大改善。

其他可读性指标：211 句，平均 17.6 词/句，仅 1 句超过 45 词（Conclusion 首句，49 词，建议拆分）。

### 标题超长（APA 建议 ≤12 词）
当前标题 **19 词**："Knowing What You Don't Know: Behavioral Confidence Calibration in Humans and Large Language Models Across English- and Chinese-Prompted Cohorts"。APA 7 建议标题不超过 12 词左右。可压缩，例如：
- "Knowing What You Don't Know: Confidence Calibration in Humans and Two LLM Cohorts"（13 词）
- "Confidence Calibration in Humans and Two LLM Cohorts: Knowing What You Don't Know"（13 词）
副标题可把语言信息移入摘要。**提示**：原稿标题同样偏长，若模板对标题另有要求请以模板为准。

### 表格与图片位置
APA 7 建议表格与图片**置于参考文献之后**，每表/图单独一页，正文中以 "(see Table 1)" 引导；也可嵌入首次提及处（多数期刊接受）。当前 md 为嵌入正文。转 Word 时按投稿要求二选一，若走 APA 严格版需把 Table 1 与 Figure 1–3 移到 References 之后。

### 提交 Word 时需落实（当前 md 无法体现）
- Times New Roman 12pt 全文；**1.5 倍行距**（APA 7 允许 1.5 或双倍）
- 页边距 1 英寸；正文左对齐（非两端对齐）；首行缩进 0.5 英寸
- **页码**在右上角
- 参考文献表悬挂缩进 0.5 英寸
- 表格用 APA 表格式（表号、斜体标题、表上三线表/无竖线）——当前 md 表格转 Word 需重排
- 数字 0–1 省略前导零（论文已统一为 `.086` 形式 ✅，转换时勿被自动更正改回 `0.086`）

---

## 3. 参考文献：存在性已核实，但有 2 处引用缺失

12 条文献**全部经联网核实存在**，作者、年份、卷期、页码、DOI 与论文标注**一致**：

| 文献 | 核实结果 |
|---|---|
| Cacioli (2026) arXiv 2603.25112 | ✅ 存在于 arXiv / HF Papers / Semantic Scholar |
| Griot et al. (2025) Nat Commun 16:642 | ✅ PubMed 确认 2025-01-14;16(1):642，DOI 正确 |
| Cash et al. (2025) Mem Cogn DOI 10.3758/s13421-025-01755-4 | ✅ PubMed 确认在线预发表 |
| Steyvers et al. (2025) Nat Mach Intell 7:221–231 | ✅ Nature 官网确认 |
| Kadavath et al. (2022) arXiv 2207.05221 | ✅ |
| Geng et al. (2024) NAACL pp.6577–6595 | ✅ ACL Anthology 确认 |
| Fleming & Lau (2014) Front Hum Neurosci 8:443 | ✅ |
| Flavell (1979) Am Psychol 34(10):906–911 | ✅ DOI 正确 |
| Jacovi & Goldberg (2020) ACL pp.4198–4205 | ✅ |
| Turpin et al. (2023) NeurIPS 36:74952–74965 | ✅ |
| Scholten et al. (2024) arXiv 2408.05568 | ✅ |
| Yin et al. (2023) Findings ACL pp.8653–8665 | ✅ |

### ❌ 必须修复：Cacioli 与 Geng 列在文末但正文从未引用
APA 7 要求参考文献表**只收录正文引用过的文献**（每一处引用都必须有对应条目，反之亦然）。

- `Geng, J., et al. (2024)` 未在正文任何位置出现
- `Cacioli, J.-P. (2026)` 未在正文任何位置出现

> 对比：原稿正文有 "Geng et al. (2024) map the wider methodological spectrum…" 与 "Cacioli (2026) provides an especially close methodological comparison…" 两处引用，重写时正文段落被删除但条目留在表里。**这是重写遗留的脱节。**

两种修法（择一）：
1. 在正文补回引用。最自然的两处：
   - Related Work §"Confidence and Self-Knowledge in Language Models" 讲 calibration 方法谱系处补 Geng et al. (2024)；
   - 同节讲 Type-2/meta-d′ 处补 Cacioli (2026)（顺带可保留原稿那个有价值的限定：meta-d′ 在无二选一 Type-1 决策的开放式问答中不成立，故本文的真/假任务才适用）。
2. 或从参考文献表删除这两条。

### 次要 APA 问题
- **Guo et al. (2017) 与 Turpin et al. (2023) 无 DOI/URL**。PMLR 与 NeurIPS 论文集本身不提供 DOI，这属可接受的例外；但若严格按 APA 7「有 URL 则给 URL」，可补 PMLR 链接与 arXiv 链接。
- 参考文献 12 条远少于原稿 18 条。被删的 Peng et al. (2024)（Tong Test）、Ackerman (2025)、Cash et al. (2025)、Rudin (2019)、Hart (1965) 均**在正文中也已无引用**，因此不构成孤儿条目，属一致性删除 —— 但见第 5 节内容损失。

---

## 4. 数字与文本一致性问题

### 4.1 模型命名在表格与正文之间不一致（须统一）
表格用带量化后缀的全名，正文全部简写，读者需自行对应：

| 表格标签 | 正文标签 |
|---|---|
| `Gemma 4 E2B Q4_K_M` | `Gemma E2B` |
| `Gemma 4 E4B Q4_K_M` | `Gemma E4B` |
| `Gemma 4 26B-A4B QAT` | `Gemma 26B` |
| `Qwen3 1.7B Q8_0` | `Qwen3 1.7B` |
| `Qwen3 4B Q4_K_M` | `Qwen3 4B` |
| `Qwen3 14B Q4_K_M` | `Qwen3 14B` |

**额外风险**：英文队列里也有 `local-gemma-e2b`、`local-qwen3-1.7b` 等同名配置。正文写 "Gemma E2B" 时，读者无法确定指英文队列还是中文队列的那个。第 100 行 "the weakest local Gemma E2B and Qwen3 1.7B configurations" 讲的是**英文队列**，但用词与讲中文队列时完全相同。

建议：首次出现时定义简称（如 "Gemma 4 E2B Q4_K_M (hereafter Gemma E2B-CN)"），或正文一律用全名，至少在同句涉及两个队列时加限定词。

### 4.2 同一人类组出现两套置信区间（须加注）
- 第 100 行：人类 ECE CI = **[.062, .113]**
- 第 129 行：人类 M-ratio CI = **[1.205, 1.543]**

这两组来自**原始英文分析**（该分析的 ECE CI 用 trial 级重采样），而中文队列的数字来自**本次复现分析**（人类 CI 为 participant 聚类：[.0629,.1101] 与 [1.1997,1.5377]）。两者数值接近、结论相同，引用哪个都站得住，但**同一篇论文里对同一人类组混用两个分析批次**会让细心的审稿人困惑。

建议：在 §Measures 或首次出现处加一句说明（"Human intervals are reported from the original analysis; the Chinese-prompted cohort is analyzed in a separate pipeline"），或统一改用本次复现分析的聚类 CI。

### 4.3 图注中 "M-Ratio" 大小写与正文 "M-ratio" 不一致
第 139 行图 2 标题为 "MLE M-Ratio Estimates by Evidence Tier"，正文一律写 "M-ratio"。统一为 `M-ratio`。

### 4.4 一处易被误读的数字
第 96 行 "Gemma 26B produced only five errors" —— 该组是 prior-dominated（<10），但同表仍列出其 MLE M-ratio = .920。表注已说明 "not used for substantive interpretation"，正文第 133 行也重申，**处理是得当的**。仅提示：`.920` 与人类 `1.370` 同处一列且量级相近，建议在图 2 或表 1 中对该行加视觉标记（如灰色/斜体），避免读者误把先验主导值当作可比估计。

---

## 5. 内容与论证

### 优点
- 三问式结构（accuracy & calibration → sensitivity → 幻觉子集）清晰，且与数据能力匹配。
- "多队列 + 分strata报告" 的设计陈述诚实，反复声明不构成语言因果实验。
- 对 M-ratio 的解读保留了正确的谨慎口径（>1 不等于内省，≈1 不等于存在 faculty）。
- 新增的 Type-2 AUC 交叉验证（§Confidence Sensitivity 末段）**很值得保留** —— 它给出了"低 ECE 但 AUC≈.510"的强论据，说明聚合校准与错误分辨是两回事，这是原稿没有的亮点。
- 53/47 不平衡拆分的显式说明（第 56 行）是恰当的诚实披露，且对"一律答 False"的替代解释有实质约束力。

### 问题
**(a) §Scope of Inference 自称"analyzed as separate strata throughout"，但方法上并未分strata。**
第 48 行称两队列 "analyzed as separate strata throughout the paper"，但实际做法是"对英文队列做配对 bootstrap 差值检验，对中文队列只报点估计与区间"，并没有对中文队列施加分层分析。建议改为更准确的表述，例如 "reported separately and never pooled"。

**(b) 结论丢失了中文队列这一核心贡献的显著性。**
第 183 行 Conclusion 只说 "Humans and models differed in observable failure profiles"。但本文相对原稿的**最大增量**是：在语言匹配条件下，人类 M-ratio 仍高于所有可估模型、且没有任何模型的可估估计超过 1。这一句应当进入 Conclusion 与 Abstract 的显要位置（Abstract 目前有，Conclusion 偏弱）。

**(c) 删除了原稿的可复现性事实披露。**
原稿 §3.6 有 "several hosted endpoints used in the original collection are now retired or inaccessible… exact generative reproduction is not guaranteed"。重写版第 175 行只说 "differences in … model availability"，把这条**具体且重要**的限制弱化成了泛泛之词。建议恢复该事实陈述。

**(d) "model realization" 含义不明。**
第 10 行与第 175 行用 "model realization" 表述队列差异，读者难以理解具体所指（推测指 checkpoint/量化实现）。建议改为具体措辞：checkpoint、quantization scheme、serving runtime。

**(e) 缺少对 53/47 不平衡影响人类检验的说明。**
人类幻觉子集只有 197 条虚构 / 212 条真实条目（人类每人只做 36 题的子集采样）。第 147 行称 ".269 is well below the .50 chance benchmark"，这一判断本身正确，但**论文没有报告该检验的统计量或样本数**；原稿有 "binomial p = 6.6 × 10⁻¹¹" 及题目级稳健性（82% 虚构题低于 50%、剔除最差 3 题后 acc = .296）。建议补回 p 值与至少一句题目级稳健性，否则 "well below chance" 是一个无统计支撑的断言。

---

## 6. 语法与文风

整体质量高，无语病级别错误。自动化检查结果：

- **无**重复词（"the the" 类）、**无**标点前空格、**无**句首 And/But/So、**无**英式/美式拼写混用（behavior/analyze 统一）
- 第一人称单数/复数使用规范（仅 1 处 "we"，在 §4.7 附近，指代可接受）
- 数字风格统一：正文一律省略前导零（`.086`、`.735`），符合 APA 7
- 211 句中仅 1 句超 45 词

可优化的小项：
1. **第 183 行 Conclusion 首句 49 词**，建议拆为两句。
2. 第 56 行 "This composition is important: it is not an exactly balanced 50/50 signal-detection set." —— "composition" 略含糊，建议 "The 53/47 imbalance is important"。
3. 第 169 行 "The main result is not that one respondent class possesses a mental faculty that the other lacks." —— "respondent class" 在 APA 语境下偏生硬，建议 "one group"。
4. 第 14 行 "drafting assistance, and decision support" 与第 14 行同段 "decision support" 语义重叠于 LLM 用途列举，可精简。
5. 第 171 行 "Gemma E2B illustrates the distinction." 与前文第 135 行内容**几乎重复**（同一论据在 Results 与 Discussion 各讲一遍）。建议 Discussion 处改为回指（"As noted above, Gemma E2B…"）以省字数。

---

## 7. 待办清单（按优先级）

| # | 项目 | 类型 | 必要 |
|---|---|---|---|
| 1 | 补回或删除 Geng (2024)、Cacioli (2026) 的正文引用 | APA 合规 | **必须** |
| 2 | 统一模型命名（表格全名 vs 正文简称），消除跨队列歧义 | 一致性 | **必须** |
| 3 | 说明人类两套 CI 的来源 | 透明度 | **必须** |
| 4 | 补回人类虚构题检验的 p 值与题目级稳健性 | 论证完整性 | 建议 |
| 5 | 恢复 "hosted endpoints retired" 的可复现性披露 | 内容完整性 | 建议 |
| 6 | 修正 "separate strata throughout" 的表述 | 准确性 | 建议 |
| 7 | Conclusion 强化中文队列的增量发现 | 论证 | 建议 |
| 8 | 图注 "M-Ratio" → "M-ratio"；prior-dominated 行加视觉标记 | 细节 | 建议 |
| 9 | 拆分 Conclusion 首句；合并重复段落 | 文风 | 可选 |
| 10 | 转 Word 时落实 1.5 倍行距 / TNR 12 / 悬挂缩进 / 页码 | 格式 | 转档时 |

---

## 附：本次审查建立的可复现脚本

- `audit_paper_numbers.py` —— 表格数值与四个源 CSV 的交叉一致性检查
- `audit_paper_claims.py` —— 122 项正文声称逐条回算

---

## 8. 图片核验（9 张 PNG 逐张读图 + 按坐标轴刻度做像素测量）

论文当前只嵌入 3 张图（Figure 1–3），路径全部可解析。核验结论如下。

### 8.1 与论文声称一致 ✅

| 图 | 核验结果 |
|---|---|
| `reliability_curves.png`（未嵌入论文） | 人类曲线**每一点都在对角线下方**（conf .50→acc .38、.625→.57、.75→.65、.875→.77、1.0→.93），完美过度自信特征；模型曲线更平更高（.79–1.00）。对角线确为 y=x（拟合斜率 0.999）。 |
| `hallucination_breakdown.png`（Figure 3） | 人类虚构 .267 / 真实 .730（论文 .269/.731 ✅）；六模型两半均高（.77–1.00）。 |
| `accuracy_by_type.png`（未嵌入） | 人类 常规 .798 / 学科 .862 / 陷阱 .767 / **幻觉 .508** —— 与论文 .799/.862/.767/.509 完全一致，且**只在幻觉类崩塌**，中文标签渲染正常无缺字。 |
| `mratio_evidence.png`（Figure 2） | M-ratio=1.0 虚线存在；三个证据层级颜色可区分且图例具名；Gemma E2B/E4B 的须线下端**确实被下轴截断且无下端帽**，与论文所述 −.189/−.206 落在可见区外一致。 |
| `c_ece.png`（英文队列，未嵌入） | 标题 "humans worst calibrated"；人类 0.086 [0.061, 0.113] 确为最长条与最宽 CI，支撑原稿 17/19 的说法。 |

### 8.2 已据核验结果修正论文图注 ✅

1. **Figure 1 的 ECE 面板与 accuracy 面板共用 0–1.0 轴**，而全部 ECE < .11，导致 7 根 ECE 柱被压缩在底部约 11% 区域内，**人类 .086 与 Qwen3 1.7B .101 的差异肉眼不可分辨**。
   → 已在图注补明"两面板共用 0–1.0 轴、ECE 差异被压缩、请以 Table 1 为准"，并补明"本队列中人类为 ECE 第二高、并非最高"。
2. **Figure 2 图例为三个层级都画了误差线符号，但 regularized 与 prior-dominated 两点实际没有区间**（论文按设计省略），图例暗示了图上没有的东西。
   → 已在图注补明图例符号不适用于这两点，并把被截断的 −.189/−.206 写进图注（读者无法从图上读数）。

### 8.3 经复核**不成立**的两项告警

- **"Figure 4 最大值 0.997 与图不符（图上只到 0.99）"** —— 不成立。我核对了绘图代码（`replication_analysis.py:1193-1224`），`human_model_accuracy_ece.png` 与 `accuracy_by_type.png` 分别调用 `summarize_groups` 与 `summarize_factor`，两者取自**同一份 `matched` 数据**，Gemma 26B 总体准确率就是 **0.996871**（1593/1598，5 错）。像素测量读数偏低约 0.009，属测量误差，非图数据错误。**论文的 0.997 正确。**
- **"Figure 4 中 qwen3-4b 与 qwen3-14b 的 ECE 完全相同，疑重复"** —— 不成立。源数据为 .0427 与 .0434，两者本就极接近，在 0–1.0 轴上自然是同一像素行。

### 8.4 图片层面仍建议改进（不影响论文数值正确性）

| # | 问题 | 建议 |
|---|---|---|
| 1 | Figure 1 的 ECE 面板应使用独立轴（如 0–0.15），否则论文依赖的"人类 vs Qwen3 1.7B ECE"对比在图上不可读 | 改绘图代码 |
| 2 | 多处"零余量"：`accuracy_by_type` 中 26B 四类均 .997、`hallucination_breakdown` 中多根柱 =1.00，柱顶与上框齐平像被裁切 | y 轴上限改 1.05 |
| 3 | `reliability_curves` 中钉在 1.0 的模型标记被上框切成半圆；多条模型曲线在顶部重叠（红/棕/粉/绿/橄榄难以分辨） | 放宽上限或错开线型 |
| 4 | 不确定度报告不一致：只有 Figure 1 与 `c_ece` 有误差棒，by-type、hallucination、c_accuracy 系列均无 | 至少给关键对比加区间 |
| 5 | `c_accuracy.png` x 轴从 0.40 起，使 19 个模型在 .92–1.00 间看似相同，同时夸大微小差异 | 若用于论文，考虑分段轴或表格化 |
| 6 | `c_accuracy_by_type.png` 的 0.5 虚线未在图例具名（同类图 `c_hallucination_acc` 已标 "chance"） | 补标签 |

### 8.5 两个需要你亲自确认的问题

1. **`experiment/results/figures/` 中存在两张近似重复的分题型图**：`c_accuracy_by_type.png`（人类幻觉 .508）与更旧的 `fig6_accuracy_by_type.png`（人类幻觉约 .475）。**若误用旧版会把论文里的 .509 讲成 .475。** 建议删掉或明确标注旧版作废。
2. **英文队列图中的模型名（gpt-5.5、gpt-5.4-mini、qwen3.7-max/plus、deepseek-v4-pro/flash）需你确认是真实运行结果**，非占位/合成数据。它们与当前公开模型命名不符（可能因项目设定在 2026 年）。本审查无法从仓库内自证其来源，但论文正是引用这批数据得出 "17/19" 的核心结论，建议保留好运行日志与 `served_model` 记录以备查。
