# -*- coding: utf-8 -*-
"""汇总 HMeta-d 贝叶斯结果:按错误数分可信层级,与 MLE 对照,出全组图。"""
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "results", "analysis"); F = os.path.join(HERE, "results", "figures")
b = pd.read_csv(os.path.join(A, "hmetad_summary.csv")).set_index("model")
mle = pd.read_csv(os.path.join(A, "metad_summary.csv")).set_index("model")
gs = pd.read_csv(os.path.join(A, "group_summary.csv")).set_index("model")

def tier(nw):
    if nw >= 30: return "data-driven"       # 数据可支撑
    if nw >= 10: return "regularized"        # 贝叶斯正则化,可参考
    return "prior-dominated"                 # 先验主导,不可信(≈先验~1)
b["evidence"] = b.n_wrong.map(tier)
b["mle_m_ratio"] = mle["m_ratio"]
b["vendor"] = gs["vendor"]; b["tier"] = gs["tier"]
b = b.sort_values("n_wrong")
b.to_csv(os.path.join(A, "hmetad_final.csv"))

pd.set_option("display.width", 200)
print("== HMeta-d 贝叶斯 M-ratio(按错误数排序;与 MLE 对照)==")
print(b[["n_wrong", "mr_mean", "hdi_lo", "hdi_hi", "width", "evidence", "mle_m_ratio", "rhat"]].round(3).to_string())
print("\n可信层级计数:"); print(b.evidence.value_counts().to_string())

# ---- 图:全部 20 组,按证据层级着色,HDI 区间 ----
COL = {"data-driven": "#1f77b4", "regularized": "#7fb3d5", "prior-dominated": "#c9c9c9"}
bb = b.sort_values("mr_mean")
y = np.arange(len(bb))
colors = ["#d62728" if m == "human" else COL[bb.loc[m, "evidence"]] for m in bb.index]
plt.figure(figsize=(9, 7))
plt.barh(y, bb.mr_mean.values, color=colors)
for i, m in enumerate(bb.index):
    plt.plot([bb.loc[m, "hdi_lo"], bb.loc[m, "hdi_hi"]], [i, i], color="k", lw=1)
    plt.text(0.02, i, f"e={int(bb.loc[m,'n_wrong'])}", va="center", fontsize=7, color="#333")
plt.axvline(1.0, ls="--", c="gray", lw=1)
plt.yticks(y, bb.index, fontsize=8)
plt.xlabel("Bayesian M-ratio = meta-d′/d′  (95% HDI)")
from matplotlib.patches import Patch
leg = [Patch(color="#d62728", label="human"),
       Patch(color="#1f77b4", label="data-driven (≥30 errors)"),
       Patch(color="#7fb3d5", label="regularized (10–29 errors)"),
       Patch(color="#c9c9c9", label="prior-dominated (<10 errors, ≈ prior)")]
plt.legend(handles=leg, fontsize=8, loc="lower right")
plt.title("HMeta-d (Bayesian) M-ratio for all groups\n"
          "grey bars sit at the prior (~1): too few errors to be informative", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(F, "c_mratio_bayes.png"), bbox_inches="tight"); plt.close()
print("\n图:results/figures/c_mratio_bayes.png")
