**Annotated Bibliography**

Metacognitive Calibration in Humans and Large Language Models

*"Knowing What You Don't Know"*

**使用说明。** 以下引用按可获取信息整理为 APA 7
格式。提交前请逐条核实完整作者列表、卷期页码与
DOI------部分预印本的作者全名与最终发表信息可能有出入。每条包含四部分：Citation、Summary、What
I can use、What I still
need。最后一部分串起来，基本就勾勒出本研究尚需自己填补的缺口。

**A. 最接近本研究的工作**

**1. Steyvers et al. (2025) --- 人对 LLM 的感知信心 vs 模型内部信心**

**Citation.** Steyvers, M., et al. (2025). What large language models
know and what people think they know. *Nature Machine Intelligence,
7*(2), 221--231.
[[https://doi.org/10.1038/s42256-024-00976-7]{.underline}](https://doi.org/10.1038/s42256-024-00976-7)

**Summary.** 提出并量化了两个概念："calibration gap"------人对 LLM
答案的信心与模型自身内部信心之间的差距；以及"discrimination
gap"------人和模型区分对错答案能力的差距。实验发现用户倾向高估 LLM
准确性，尤其在看到默认解释时；较长的解释会进一步抬高用户信心，即使并未提升准确率。

**What I can use.** Related Work 的核心对照，也是 Methods 中 ECE / AUROC
度量的直接先例（它在人和模型两侧都算了校准误差）。它的"信任校准"动机可直接支撑我
Introduction 的现实意义部分。

**What I still need.** 关键：它测的是"人对 LLM 答案的感知信心" vs"LLM
内部信心"，而不是"人自己做题的元认知" vs"LLM 做题的元认知"。它把人当成
LLM 的"读者"，我把人当成和 LLM
并列的答题者。所以它恰恰留出了我的缝隙------我仍需一个让人和模型面对同一批题、各自报告自己置信度的对称设计。

**2. Yin et al. (2023) --- LLM 能否识别自己不会的问题**

**Citation.** Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., & Huang, X.
(2023). Do large language models know what they don't know? *arXiv
preprint* arXiv:2305.18153.
[[https://arxiv.org/abs/2305.18153]{.underline}](https://arxiv.org/abs/2305.18153)

**Summary.** 评估 LLM
的"自我知识"（self-knowledge），即识别哪些问题超出自己能力、无法回答的能力。发现模型在识别"不可回答问题"上与人类仍有明显差距。

**What I can use.** 标题几乎就是我的研究问题，必须在 Introduction
引用以定位本文。它的"识别不可回答问题"可作为我"未知的未知"题型的设计依据之一。

**What I still need.**
它聚焦"能不能识别无法回答"，偏二元判断；我需要的是连续的置信度-正确率匹配（校准曲线），粒度更细。且它没有人类基线，我仍需自己采集人类数据做对称比较。

**3. Kadavath et al. (2022) --- LLM 自评与 P(IK) 的奠基工作**

**Citation.** Kadavath, S., et al. (2022). Language models (mostly) know
what they know. *arXiv preprint* arXiv:2207.05221.
[[https://arxiv.org/abs/2207.05221]{.underline}](https://arxiv.org/abs/2207.05221)

**Summary.** 研究 LLM
能否评估自身主张的有效性、并预测自己能答对哪些问题。发现较大的模型在格式恰当时校准良好；可训练模型预测
P(True) 与 P(IK)；但 P(IK) 在新任务上的校准会变差。

**What I can use.** 校准方法论的基石，Methods 中置信度引出与 ECE
的直接参照。它"换新任务就崩"的发现，是我设计 held-out /
幻觉触发题型的有力依据------也支持 H3。

**What I still need.**
它纯测模型、无人类基线，且"格式恰当时校准良好"这一结论依赖特定提示格式------我仍需控制
prompt 敏感性，并把人放进同一框架做对称对照。

**4. Griot et al. (2025) --- LLM 在医疗推理中缺乏可靠元认知**

**Citation.** Griot, M., Hemptinne, C., Vanderdonckt, J., & Yuksel, D.
(2025). Large language models lack essential metacognition for reliable
medical reasoning. *Nature Communications, 16*(1), 642.

**Summary.** 在医疗推理任务上系统检验 LLM
的元认知，发现模型缺乏可靠的自我监控------难以恰当识别自己知识的边界，在高风险情境下仍可能过度自信。

**What I can use.** 为 Introduction
的"高风险场景风险"提供权威实证支撑（Nature 子刊），也为 H1/H3
提供正面证据。可在 Discussion 讨论领域特异性。

**What I still need.**
它限定在医疗领域，结论未必泛化到一般常识题；且同样无人类对照。我需要确认题型覆盖更广，并自己建立人类基线。

**B. 度量与方法**

**5. Fleming & Lau (2014) --- 人类元认知的测量方法**

**Citation.** Fleming, S. M., & Lau, H. C. (2014). How to measure
metacognition. *Frontiers in Human Neuroscience, 8*, 443.

**Summary.**
综述人类元认知的测量方法，核心贡献是用信号检测论框架（meta-d′）把"元认知敏感度"与"一阶任务表现"解耦------避免把"答得准"误当成"自知得准"。

**What I can use.** 可能比 ECE 更适合我的对称设计：meta-d′
提供一个能同时套在人和 LLM
两侧、且控制了一阶能力差异的"监控敏感度"度量。建议在 Methods 里和 ECE
并列考虑。

**What I still need.** 它是为人类心理实验设计的，把 meta-d′ 迁移到 LLM
上是否合理、如何实现，文献里尚无成熟做法------这需要我自己论证迁移的有效性，可能成为一个方法贡献点（也是潜在审稿风险点）。

**6. Guo et al. (2017) --- 现代神经网络的校准与 ECE**

**Citation.** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
On calibration of modern neural networks. *Proceedings of the 34th
International Conference on Machine Learning (ICML)*, 1321--1330.
[[https://arxiv.org/abs/1706.04599]{.underline}](https://arxiv.org/abs/1706.04599)

**Summary.**
发现现代深度网络虽准确率高，却普遍校准差、系统性过度自信；引入（普及了）期望校准误差（ECE）作为度量，并提出
temperature scaling 等后处理校准方法。

**What I can use.** ECE 的标准出处，Methods
必引。"高准确但过度自信"的发现可作为我预期 LLM 行为的理论先验。

**What I still need.** 它针对的是图像分类等传统模型，不是生成式
LLM，也完全不涉及人类对照。我需要说明 ECE 用于 LLM
文字任务时的适配（如何分箱、置信度如何引出）。

**7. Geng et al. (2024) --- LLM 置信度估计与校准综述**

**Citation.** Geng, J., Cai, F., Wang, Y., Koeppl, H., Nakov, P., &
Gurevych, I. (2024). A survey of confidence estimation and calibration
in large language models. *Proceedings of NAACL-HLT 2024*, 6577--6595.

**Summary.** 系统综述 LLM
置信度估计与校准的方法谱系：温度缩放、verbalized
uncertainty（让模型用语言/数字直接说出置信度）、基于一致性/集成的方法等。

**What I can use.** 帮我在 Methods 里选定并论证置信度引出方式（尤其
verbalized confidence，因为要和人类的"声称置信度"对齐）。也为 Related
Work 提供方法地图。

**What I still need.**
综述只罗列方法、不做人机对比。我仍需自己确定一个对人和模型都公平的单一引出方式，并处理它已指出的
verbalized confidence 的离散化/不稳定问题。

**C. LLM 元认知的局限（支撑假设）**

**8. Scholten et al. (2024) --- LLM 的"元认知短视"**

**Citation.** Scholten, F., Rebholz, T. R., & Hütter, M. (2024).
Metacognitive myopia in large language models. *arXiv preprint*
arXiv:2408.05568.
[[https://arxiv.org/abs/2408.05568]{.underline}](https://arxiv.org/abs/2408.05568)

**Summary.** 借用人类认知偏差概念"元认知短视"，论证 LLM
在整合信息时过度依赖表面/高频信息，缺乏对信息来源与可靠性的元层面加工。

**What I can use.** 为 H1/H3 提供机制性解释，也呼应我"common sense ≠
correct"的论点------可放进 Discussion 解释 LLM
为何在"大家都这么说"处最自信。

**What I still need.**
它是概念迁移+演示，非严格的校准量化，也无人类对照实验。我需要把这个直觉变成可测的校准崩溃，并用人类基线确认这是模型特有还是人也有。

**9. Cash, Oppenheimer & Christie (2024) --- 测 LLM 置信度判断的准确性**

**Citation.** Cash, T. N., Oppenheimer, D. M., & Christie, S. (2024).
Quantifying UncertAInty: Testing the accuracy of LLMs' confidence
judgments. *PsyArXiv*. （preprint------提交前请核实最新发表状态与链接）

**Summary.** 用认知心理学的范式检验 LLM
置信度判断的准确性，把元认知测量工具应用到模型上。

**What I can use.** 方法上是"把人类元认知范式搬到
LLM"的近邻先例，可借鉴其任务/计分思路，并在 Related Work 中与之对比。

**What I still need.**
需核实它是否包含人类对照（若无，正是我的对称设计的价值）；也需确认其题型是否覆盖我的陷阱/幻觉触发类。

**D. 概念与理论根基**

**10. Flavell (1979) --- 元认知概念的源头**

**Citation.** Flavell, J. H. (1979). Metacognition and cognitive
monitoring: A new area of cognitive--developmental inquiry. *American
Psychologist, 34*(10), 906--911.

**Summary.**
首次系统提出"元认知"框架，区分元认知知识、元认知体验与认知监控，奠定了"对自身认知的认知"这一研究领域。

**What I can use.** Introduction
定义元认知、并把"知道自己不知道"锚定为监控功能时的奠基引用。也支撑我"元认知是求知前提"的意义论述。

**What I still need.** 它是理论框架、非测量或 AI
工作。从它到"可量化的校准"之间的操作化，需要靠 Fleming &
Lau（度量）和我自己的实验来搭桥。

**11. Peng et al. (2024) --- Tong Test（本研究的学理出发点）**

**Citation.** Peng, Y., Han, J., Zhang, Z., Fan, L., Liu, T., Qi, S.,
Feng, X., Ma, Y., Wang, Y., & Zhu, S.-C. (2024). The Tong test:
Evaluating artificial general intelligence through dynamic embodied
physical and social interactions. *Engineering, 34*, 12--22.
[[https://doi.org/10.1016/j.eng.2023.07.006]{.underline}](https://doi.org/10.1016/j.eng.2023.07.006)

**Summary.** 观点性文章，主张 AGI
评估应植根于动态具身的物理与社会交互（DEPSI），提出能力-价值双系统的五级评估框架，并论证当前
LLM 因缺乏自驱动、价值、因果理解与自我认知而不应被视为 AGI。

**What I can use.** 本研究的概念起点：它指出 LLM
缺乏真正的自我认知，我把其中一个可严格测量的面向（元认知校准）操作化。适合放在
Introduction / Discussion 作为大框架锚点。

**What I still need.**
它是宏大蓝图、非可执行实验，也未给出元认知的具体量化方案。它只提供"为什么这个面向重要"，而"如何测、与人怎么比"全部需要我自己设计------这恰是本研究的落点。

**综合缺口：用完以上文献后，本研究仍需自己提供的**

把每条的"still need"汇总，指向三个必须由我自己完成的部分：

1.  **对称设计。**现有工作要么只测
    LLM（Kadavath、Yin、Griot、Guo），要么把人当作 LLM
    的"读者"（Steyvers）。没有人让人和 LLM
    作为并列答题者、面对同一批题、各自报告置信度------这是本研究的方法核心。

2.  **统一度量。**需要一个对人和模型都公平的度量。ECE 是标配，但
    meta-d′（Fleming &
    Lau）能控制一阶能力差异，更适合对称比较；把它迁移到 LLM
    是我的潜在贡献，也是潜在风险点。

3.  **区分性题型。**现有文献多用通用 QA
    数据集；我需要自建包含陷阱题、幻觉触发题、已知未知 vs
    未知未知的题库，才能让 H1 / H2 / H3 的预测真正分开。

**提醒：**以上文献支撑的是"行为层面的校准差异"。没有任何一篇能让我宣称证明了
LLM
内部"真的有没有"元认知------这一黑箱边界对所有这些工作同样成立，应在我的
Discussion 中明确声明，并作为本研究诚实的范围界定。
