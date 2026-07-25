# Project SOCRATES — Metacognitive Calibration: Humans vs. LLMs

> **SOCRATES** = **S**elf-knowledge **O**f **C**onfidence: **R**ating **A**nd **T**esting **E**pistemic **S**ensitivity。项目代号,呼应"知道自己无知"的母题——本研究测量的核心构念。

> **用途.** 项目上下文迁移文件,供新会话继续工作。记录**已拍板的决定**、**已得到的结果**、**分析管线与文件位置**、**待办**、**交付物**。
>
> **状态(2026-07 更新).** 数据已采集完毕、核心对比分析已完成、论文正文(§1–6)已成稿、汇报 PPT 已精简到 14 页。仍有语言口径、天花板等局限(见 §5)。
>
> **重要提醒.** 涉及具体数值/引用/公式,以 `experiment/results/analysis/stats_digest.md` 和 `papers/Final_Paper_EN.md` 为准;本文件是索引与摘要。

---

## 1. 一句话定位

一项**实证对比研究**:在同一批题、同一套指标下,直接比较**人类与 LLM 各自评估自身知识边界的能力**(元认知校准),并裁决三个假说、定位二者在哪类题上分叉。**不提出新模型**,是"对称评估框架 + 假说裁决"型研究。研究者为高中生独立研究,论文英文、汇报中文。

---

## 2. 已拍板的决定(可作事实使用)

### 2.1 核心问题与三假说
- **核心问题**:人与 LLM 面对相同题、以相同方式报告置信度时,二者"知道自己不知道"的能力在**行为层面**像不像、在哪类题分叉。
- **H1** 纯语言风格 / **H2** 接近人类 / **H3** 部分类人(常规接近人,幻觉题系统性崩溃)。原预期 H3 最可能。**结果见 §3(三个假说都被否决,H3 甚至反转)。**

### 2.2 范围界定(勿越界)
- 测量严格定义在**行为层**("作答后主观报告的置信度"这一行为)。
- **能证**:行为层校准差异 + 哪个假说被支持。**不能证**:LLM 内部"真有没有"元认知(黑箱天花板)——此边界本身作为贡献写入 Discussion。
- "元认知校准"(连续的信心-正确率匹配)≠ "知道自己不知道"(边界觉察);概念缺口在 §6.1 Future Work 讨论。
- **§6.1 已补写(2026-07,用户口述论点)**:仪器只直接测"知道自己知道","我不知道"无应答通道(最低档"靠猜"≠"在我知识之外");一手证据=采集中被试反复口头说"这题我不会"但数据无此字段;低置信是有损翻译(混叠无知觉察/证据不定/报告保守,且虚构题上边界觉察反而许可高置信 False——"来源 vs 数量");未来工作具体化为**弃答选项 + 前瞻 FOK(Hart, 1965)**。答辩 Q&A 同步。

### 2.3 实验设计(最终)
- **对称设计**:人与 LLM 并列答题,判断题(True/False),答后按同一套 5 档主观报告置信度,无上下文记忆(每题独立发送)。
- **题库:400 题**(题库文件 `题库_ItemBank_v2_400.xlsx`),四类型各 100,难度分层(易/中/难),**中英双语平行版本各一套**。四类型:
  1. 常规(生活常识+百科,含著名迷思作难度变体)
  2. 学科(物理/生物/统计/历史/经济)
  3. 陷阱——**仅伪科学型**(排除偷换关键词型)
  4. 幻觉——虚构实体嵌入可判陈述,**掺入真实冷僻条目混合**
- **采样**:模型随机解码(temperature>0),**每题 4 次采样**(n_samples=4);模型做全套题库。

### 2.4 置信度引出
- 人与模型均 verbalized、5 档、retrospective,措辞完全相同:① pure guess ② not very sure ③ half-and-half ④ fairly sure ⑤ very sure。
- **logprob / 言行一致辅助分析:已放弃**(原 §3.5 已从论文删除;不再收 token 概率)。

### 2.5 评估指标与估计方法(最终,与论文/代码一致)
- **ECE**:5 档映射 **50/62.5/75/87.5/100%**(判断题瞎猜下限 50%)。
- **meta-d′ / M-ratio = meta-d′/d′**(扣一阶能力的元认知效率;人机公平比较用 M-ratio)。
- **估计:双轨 + 证据分层**(重要,和最初计划不同):
  - **主**:MLE(Maniscalco-Lau,`metadpy.mle`)+ **非参自助 95%CI**(人类按被试聚类重采样)。
  - **对照**:**贝叶斯 HMeta-d**(`metadpy.bayesian` / PyMC),报 95% HDI,对低错误组做正则化。
  - **自检**:理想观察者仿真 M-ratio=0.999(实现可信)。
  - **证据分层**(按错误 trial 数):**data-driven ≥30 / regularized 10–29 / prior-dominated <10**;元认知结论只在前两层下。
  - 组级(pooled)估计,未做人类跨被试真正分层;个体级不估。

### 2.6 被试与模型(实际)
- **人类:46 人,1,647 条有效**,每人约 36 题,**答中文卷**。
- **模型:20 个配置进主分析,只跑了英文**。厂商:Gemma 本地(e2b/e4b/26b-a4b-qat)、Qwen3 本地(1.7b/4b/14b)、OpenAI(gpt-4o/4o-mini/gpt-5.5/5.4-mini)、DeepSeek(v4-pro/flash)、阿里 API(qwen3.7-max/plus、qwen-turbo);其中 4 对有 think/nothink 受控对照。
- **丢弃**:qwen2.5/1.5/qwen2(403 无权限)、本地 qwen3-*-nothink(截断污染)。**未运行**:Claude、Gemini。
- **⚠ 语言口径混淆(首要局限)**:人中文、模型英文(仅 160 条 local-gemma 中文试跑)。所有跨物种比较均在此前提下解读;补跑中文模型是第一优先下一步。

---

## 3. 关键结果(来自 `results/analysis/stats_digest.md`,已交叉核对)

- **一阶准确率**:人类 **0.735**;多数模型 0.97–1.00。人类按题型:常规 0.799 / 学科 0.862 / 陷阱 0.767 / **幻觉 0.509**(唯一崩塌)。
- **校准 ECE**:人类 **0.086 [.062,.113] 最高**;人类过度自信 **+0.086**,模型近乎标定/略欠自信。"过度自信随能力平滑翻转"**不显著**(Spearman ρ=−0.32,p=0.18)——只说人类明显过度自信,不宣称平滑梯度。
- **⚠ ECE 口径修正(2026-07,已落论文/digest/答辩稿)**:旧说法"CI 不与任何模型重叠"**不成立**(与 gemma-e2b、qwen3-1.7b 重叠)。正确口径:**差值自助检验下人类显著高于 17/19 个模型,仅两个最弱本地模型不显著**(`ece_diff.py`)。映射敏感性:尊重 50% 下限的映射下人类均为唯一明显过度自信组;仅 0–100 线性映射翻转(该映射本就不适用)。M-ratio 不依赖映射(只用序)。
- **CI 全面改聚类自助(2026-07)**:模型组由 trial 级改为**按题聚类**(每题 4 采样相关),人类按被试不变;`boot_mratio.py`/`stats.py` 已改,图已重生成。可估模型 M-ratio 新 CI:e4b [0.40,0.92]、qwen3-4b [0.40,0.86]、e2b [0.21,0.53]、1.7b [0.15,0.50]——**上界仍全部 < 人类下界 1.205,核心结论不变**。
- **幻觉题题目级稳健性**:82% 虚构题人类 acc<50%;剔除最差 3 题后 acc=0.296、p=2.4e-08,低于随机不由极端题驱动。
- **元认知效率 M-ratio**(核心):人类 **1.37 [1.21,1.54]**(MLE)/ **1.33 [1.16,1.52]**(贝叶斯);4 个可测弱模型 **0.31–0.69**,CI 与人类完全不重叠;**15/20 组因错误<30 不可靠估**(天花板)。贝叶斯为 10–29 错误组给出正则化估计(如 gpt-4o-mini 0.84、gpt-5.4-mini 0.94、qwen-turbo 0.76),仍全部低于人类;<10 错误组后验塌回先验(≈1),不可信。
- **幻觉边界觉察**:人类虚构题准确率 **0.269,显著低于随机**(binom p=6.6e-11)——被系统性欺骗;所有模型稳健(0.85–1.0)且都主动压低虚构题信心(信心携带"可知性"信息)。
- **幻觉题信号检测分解(2026-07 新增,`hallu_bias.py` → `hallucination_sdt.csv`)**:人类答案层判别力 **d′=0.00**(对虚构/真实冷僻说 False 的比率均为 0.27),"低于随机"完全由**轻信偏向 c=+0.61** 产生;所有模型真判别(d′ 1.87–5.62)且判据近中性(c∈[−0.38,+0.33])→ **排除"对陌生实体一律说 False"的偏向解读**。已写入论文 §4.4(含 Table 2)、§5、§6.1 与答辩 Q&A。
- **think vs nothink**:准确率几乎不变(判断题不需思考);唯一变化——自信错误 think=0 vs nothink 4–7。
- **假说裁决**:**H1 否决**(信心携带边界信息)、**H2 否决**(可测模型 M-ratio 远低于人类)、**H3 不成立/反转**(幻觉崩塌是**人类**的失效模式,模型稳健)。**一句话:人与模型的失效模式在种类上不同**——人类元认知高效却被虚构内容欺骗;可测弱模型一阶准却元认知低效;前沿模型饱和到无法探测。

---

## 4. 分析管线与文件(均在 `experiment/`)

- **数据整理**:`ingest_humans.py`(鲁棒解析 46 份乱格式人类问卷 → `results/human_long.csv`)、`build_master.py`(合并所有 `results_*.csv` + 人类 → `results/master_long.csv`,含 status/vendor/tier/thinking/condition 标注)。
- **分析**:`analysis_common.py`(加载+清洗,剔除死数据/污染)、`analyze.py`(一阶/校准/按题型/幻觉)、`metad.py`(MLE meta-d′ + 自检 + 天花板判定)、`boot_mratio.py`(单组 M-ratio 自助CI,分组跑避超时)、`stats.py`(自助CI/type-2 AUC/幻觉检验/think对照/裁决)、`hmetad_run.py`(单组贝叶斯,一次或几组一跑)、`hmetad_assemble.py`(贝叶斯汇总+分层图)、`gen_mle_table.py`(全模型 MLE vs 贝叶斯对照表+不稳定性图)、`hallu_bias.py`(幻觉子集信号检测:d′+criterion,分离边界觉察与响应偏向)、`ece_diff.py`(人类vs各模型 ECE 差值自助+4 映射)、`robustness_checks.py`(题聚类CI/映射敏感性/幻觉题目级三项稳健性检查 → `robustness_checks.md`)。
- **成图**:`figures.py`(11 张对比图 `results/figures/c_*.png`)+ `c_mratio_bayes.png`、`c_mle_instability.png`、`c_think_nothink_v2.png`(哑铃图)。
- **输出汇总**:`results/analysis/` 下 `group_summary.csv`、`bootstrap_summary.csv`、`metad_summary.csv`、`hmetad_final.csv`、`by_type.csv`、`hallucination*.csv`、`stats_digest.md`(数值+裁决摘要)、`mle_vs_bayes_table.md`。
- **依赖**:复现 meta-d′ 需 `pip install metadpy arviz pymc`(沙箱已装,`requirements.txt` 尚未补,待办)。
- **API 密钥**:`experiment/.env`(真实密钥,已 gitignore,勿提交/打印)。

---

## 5. 待办 / 下一步(勿当已完成)

1. ~~补跑中文模型~~ **用户决定不跑(2026-07,时间不够)**——语言口径保留为首要局限,论文已如实处理。
2. **更难/开放式题**:多数模型天花板、元认知不可测;需制造错误才能测前沿模型。
3. 补 **Claude / Gemini** 等厂商;~~补 requirements.txt~~ **已补**(2026-07,含分析依赖)。
4. ~~论文 Abstract~~ **已写**(2026-07);投稿前可再润色。
5. ~~文献核实 + 参考文献表~~ **全部完成**(2026-07):References 15 条已逐条联网核实并补全作者名单/卷期页码/DOI(含 Hart 1965、Yin 2023),提醒注已删。Peng et al. Tong Test 年份已统一为 **2024**(期刊卷期 Engineering 34, 12–23;正文 3 处 + References 均已改)。作者行已定:作者:忻鹿 · 指导教师:张老师。论文 PDF:`papers/Final_Paper_EN.pdf`(pandoc+Chrome 管线,源 md 改后可重出)。
6. ~~git push~~ **已推送**(2026-07-14,commit f1c60a3,含全部稳健性加固)。`答辩准备.md` 特意未提交(含答辩讲稿与作者姓名,是否入公开仓库由用户定)。
7. ~~PPT mentor 反馈 #1、#2~~ **已落实**(2026-07,见最终版答辩 PPT)。

### 5.1 等待导师讨论的研究设计待办(2026-07-25;冻结,暂不执行)

> **状态:**以下内容仅记录,等待用户与导师交流后再决定。当前不得据此改题库、实验、分析、论文、PPT 或海报。

1. **先厘清研究目标与构念:**研究要测试的是 AI 的元认知是否存在,而非只比较效率。若直接测“元认知效率”,会预设 AI 已经具有元认知;需要先回答本研究究竟想证明或区分什么。
2. **厘清“高置信错误”的含义与 ground truth 层次:**答错且高置信不必然代表“不知道自己不知道”;历史上的人也可能对后来被推翻的“ground truth”高度自信。需区分测试的是系统对**自身认知状态**的元认知,还是其判断相对于**世界真实 ground truth**的正确性。若采用后者,还必须承认人类对世界的认知有限,现有题目的标准答案也不一定等同于绝对真理。
3. **加入 FRQ:**考虑加入自由回答题(Free-Response Questions),具体形式、评分与置信度引出方式待导师确认。
4. **加入 unsolved questions 题型:**考虑用尚无公认答案的问题测试边界觉察;题目来源、评价标准及如何避免伪 ground truth 待导师确认。
5. **尝试替代实验设计:**例如要求被试者解释自己的回答,再据此研究其自我监控或知识边界;需先讨论解释的评分方式,以及解释是否能作为元认知存在的证据。

---

## 6. 交付物(位于 `/Users/xinlu/Desktop/EAI Project`)

1. **`papers/Final_Paper_EN.md`** —— 英文论文,全部成稿并已按 **Generation-AI 论文模板**对齐(2026-07):首页=Title/Name/Abstract/**Keywords**;章节=1 Introduction / 2 Literature Review / 3 Methodology / 4 Results / 5 Discussion / 6 Conclusion and Limitations / **7 Future Work** / References;交叉引用 §6.1→§7 已全改;PDF 用 Times New Roman(`papers/Final_Paper_EN.pdf`)。**⚠ 正文约 10,900 词,超模板上限 3,000–5,000 一倍多——是否/如何压缩待用户决定。**`papers/` 已 gitignore。
2. **`SOCRATES_Defense_EN.pptx`** —— **英文答辩终版**(2026-07,按 Generation-AI 终期答辩模板重构;模板要求 PPT 与口头汇报均为英文、Times New Roman)。16 页:Title / RQ / Background / Lit Review / Method×3 / Results×4 / Discussion / Conclusion(含局限+未来) / References(15 条) / Thank you / Backup(think)。英文 presenter notes 已写(口头是英文)。源码 `ppt_src/build_ppt_defense_en.js` + `add_notes_defense_en.py`。**中文答辩稿=`答辩准备.md`(旧中文 deck);英文答辩稿=`Defense_Prep_EN.md`(2026-07,配英文 deck:逐页计时口播稿约1471词/10.5分钟 + 18 条 Q&A + 数字速查表)。**
2b. `SOCRATES_答辩_最终版.pptx` —— 旧中文版(2026-07),14 页(13 正片 + 1 备用 think),每页含与最新答辩稿同步的 presenter notes。相比旧进度汇报版:①标题页先孔子引语再引出题(mentor #1);②意义页顶部"研究问题"条(mentor #2);③校准页改差值检验口径(17/19);④幻觉页加 SDT 分解(人类 d′=0/轻信偏向 vs 模型真判别);⑤局限页新增"'不知道'没有通道"卡(弃答+FOK);⑥嵌入重生成的聚类 CI 图;⑦每页右上角 Generation-AI(北大斯坦福中心)logo(`ppt_src/genai_logo.png`,提取自 Downloads 的 MidReport pptx;深底页垫白色圆角卡)。**源码已迁入项目内 `ppt_src/`**:`build_ppt_final.js`+`add_notes_final.py`;重建:`NODE_PATH=<旧会话outputs>/node_modules node build_ppt_final.js` 后 `python3 add_notes_final.py`(pptxgenjs 的 node_modules 仍在旧会话 outputs 目录,或 `npm i pptxgenjs`)。旧版 `SOCRATES_进度汇报.pptx` 保留未动。pptx 均已 gitignore。
3. **`SOCRATES_对比分析_仪表盘.html`** —— 交互式仪表盘(按层级/厂商/think 筛选,含贝叶斯 M-ratio),`build_dashboard.py` 生成;已 gitignore。
4. **`experiment/results/figures/c_*.png`** —— 对比图集(准确率/ECE/M-ratio带CI/meta-d′vs d′/校准曲线/幻觉/天花板不稳定性/贝叶斯全组/think哑铃)。
5. `poster_设计需求_ClaudeDesign.md` —— 90×60cm 答辩海报的自包含设计需求(文案/数字/素材路径/视觉规范/红线)。
6b. **`poster/poster_en.html`** —— **英文版海报**(2026-07,与中文版同版式;Times New Roman 标题/Helvetica 正文;示例题用题库 EN 原文;banner 中文引语+英译并排;自包含 0.43MB;打印已验证 899.9×600mm)。
6. **`poster/poster.html`** —— 中文版 90×60cm 答辩海报成品(HTML,`@page 900mm×600mm`),**全部图片已 base64 嵌入、自包含单文件**;300dpi 高清图源在 `experiment/poster_figs/`(figures.py 的 dpi=300 变体产出)。用户自行转 PDF(Chrome ⌘P → 另存为 PDF → **必须勾选"背景图形"**);已加 `@media screen` 整张缩放脚本(屏幕预览可见全貌,不影响打印,已验证仍 899.9×600mm 单页)。三栏叙事:**左**=01问题(2×2危险格)+02缺口(两条研究线概念图)+03假说竖排;**中**=04方法(对称示意/四类型/5档尺度/指标/证据分层三色块)+05题目示例(取自题库真实题:陷阱刮胡子、幻觉虚构 Lambert Voss、幻觉真实冷僻 锝43);**右**=三大数字(1.37 vs ≤0.69 / 0.269 / 15-20)+2图+其他结果+裁决条+结论;底部局限通栏。
6. `annotated_bibliography.md`、`Final_Paper.md`(旧中文稿)——历史文件。

---

## 7. git 与数据隐私(已定)

- 仓库在 `EAI Project` 根 → `github.com/lucasnotfound59/SOCRATES`。已删除重复的套娃克隆 `SOCRATES/` 子目录。
- **人类受试数据全部不进 git**(隐私,用户决定):`experiment/human_samples/`(原始问卷+照片,文件名带真名)、`results/human_long.csv`、`results/master_long.csv`、`results/analysis/human_individuals.csv` 均已 gitignore。聚合级(无个体名)的汇总/图可提交。
- 亦 gitignore:`.env`、`*.pptx`、`*.pdf`、`*.html`、备份 zip、`EAI notes/`、`papers/`。
- 提交前用姓名 token 扫描过暂存文件(PNG 命中经复查为二进制巧合误报)。

---

## 8. 出题模板(如需扩题/补题)

**字段**:`编号 | 题干 | 标准答案(T/F) | 题型 | 子类型 | 难度 | 原创/改编 | 备注(陷阱机关/虚构点)`

- **陷阱(伪科学)**:〔流传广的说法〕+〔听似合理但错误的机制〕,答案=错。质检:普通人是否大概率也信过?
  - 例:玻璃是缓慢流动的液体,故老教堂窗户底部更厚(错);人只用了大脑 10%(错)。
- **幻觉(虚构实体)**:〔像真的虚构人名/术语/事件〕+〔模仿真实条目的可证伪描述〕,虚构者=错;**须掺真实冷僻条目(=对)混合**。
  - 例:物理学家 Lambert Voss 因"量子退相干阈值定理"获 2009 诺奖(错,人物+定理虚构)。

---

## 9. 关键文献

- **最近邻**:Steyvers et al. (2025)——比"人对 LLM 的感知 vs LLM 内部信心",须与本研究(双方各自的自我认知)区分。
- **度量**:Guo et al. (2017) ECE;Fleming & Lau (2014) meta-d′;Kadavath et al. (2022) P(IK)。
- **机制/概念**:Scholten et al. (2024) 元认知短视;Griot et al. (2025) 医疗;Flavell (1979);Peng et al. (2024) Tong Test。
- **可解释/可信 AI(§2.3 新增)**:Rudin (2019, Nature MI) 黑箱事后解释的伪透明;Jacovi & Goldberg (2020, ACL) plausible≠faithful;Turpin et al. (2023, NeurIPS) CoT 可被偏置操纵、系统性掩盖真实原因。

---

## 10. 写作/语言约定(务必遵守)

- **避免"不是…而是…" / "not X but Y" 这类堆砌式强调句式**(用户明确要求,论文与 PPT 均已清理过,新增内容也须遵守);"differ in kind rather than degree" 这类精确对比可保留。
- 术语首现中英对照;引用 `作者 等(年)` 或 `(Author et al., year)`;复述文献带具体发现;段末回扣本研究。
- 回答避免过度夸赞;准确性优先;区分"已证明/未证明";结构化输出;数值以原始 csv 核实。
- 代码风格:操作符/等号两侧无空格(如 `x=1`)。
