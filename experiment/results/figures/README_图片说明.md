# 图片使用说明 / Figure usage notes

**更新：2026-09-18**

## 当前有效（论文引用请用这些）

`c_` 前缀系列为 2026-07-14 重新生成版（聚类自助 CI 修正后）：

| 文件 | 内容 |
|---|---|
| `c_accuracy.png` | 一阶准确率（英文队列 + 人类） |
| `c_ece.png` | ECE + 95% CI（英文队列 + 人类） |
| `c_accuracy_by_type.png` | 分题型准确率（人类幻觉 **.508**） |
| `c_hallucination_acc.png` | 幻觉题虚构/真实冷僻准确率（含 chance 线） |
| `c_mratio.png` | M-ratio + 95% CI（可估组） |
| `c_mratio_bayes.png` | 贝叶斯 HMeta-d M-ratio（全组） |
| `c_metad_vs_dprime.png` | meta-d′ vs d′ |
| `c_reliability.png` | 可靠性曲线 |
| `c_overconfidence.png` | 过度自信 vs 准确率 |
| `c_errors_ceiling.png` | 天花板效应：错误数分布 |
| `c_mle_instability.png` | MLE M-ratio vs 错误数 |
| `c_think_nothink.png` / `c_think_nothink_v2.png` | think vs nothink |
| `c_mratio_bayes_vs_errors.png` | **新增(2026-09-18)**:全部 19 个英文队列模型的贝叶斯 M-ratio vs 错误数，按证据层级分层显示，含人类估计与其 HDI 下界参考线。论文 Figure 4。由 `plot_bayes_mratio_vs_errors()` 生成。 |

中文复现队列的图片在
`results/replication_zh_2026-09-18/figures/`（论文 Figure 1–3 用的是这三张）。

## ⚠️ 已作废（请勿使用）

| 文件 | 作废原因 |
|---|---|
| `OUTDATED_do_not_use_fig6_accuracy_by_type.png` | 2026-07-10 旧版，人类幻觉题准确率为 **~.475**，但权威值为 **.5086**。已重命名为 `OUTDATED_` 前缀以防误用。 |
| `fig1_accuracy_by_group.png` `fig2_confidence_distribution.png` `fig3_ece_by_group.png` `fig4_reliability_curves.png` `fig5_accuracy_by_difficulty.png` `fig7_overconfidence_vs_accuracy.png` `fig8_think_vs_nothink.png` `fig9_hallucination_confidence.png` | 2026-07-10 旧系列，已被 `c_` 系列取代（聚类自助 CI 修正前的版本）。 |

## 数值权威来源

任何数字请以以下文件为准，不要从图片读数：

- 英文队列：`experiment/results/analysis/`（`group_summary.csv`、`by_type.csv`、`bootstrap_summary.csv`、`ece_diff.csv`、`hallucination_sdt.csv`、`boot_mratio_itemclust.csv`）
- 中文复现队列：`results/replication_zh_2026-09-18/`（`human_model_summary.csv`、`by_type.csv`、`bootstrap_summary.csv`、`metad_summary.csv`、`hmetad_summary.csv`、`hallucination_sdt.csv`）

回归审计：`paper_revision_zh_replication/audit_paper_claims.py`（122 项声称逐条回算）
