# -*- coding: utf-8 -*-
"""生成 全模型 MLE vs 贝叶斯 M-ratio 对照(markdown 表 + 不稳定性图)。"""
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "results", "analysis"); F = os.path.join(HERE, "results", "figures")
hm = pd.read_csv(os.path.join(A, "hmetad_final.csv")).set_index("model")
mle = pd.read_csv(os.path.join(A, "metad_summary.csv")).set_index("model")
hm["mle_dprime"] = mle["dprime"]; hm["mle_metad"] = mle["meta_d"]
hm = hm.sort_values("n_wrong", ascending=False)

# ---- markdown 表 ----
LAB = {"data-driven": "data-driven ✓", "regularized": "regularized ~", "prior-dominated": "prior-dominated ✗"}
lines = ["| Group | Errors | d′ | MLE meta-d′ | **MLE M-ratio** | **Bayesian M-ratio [95% HDI]** | Evidence |",
         "|---|---:|---:|---:|---:|---|---|"]
for m, r in hm.iterrows():
    name = "**human**" if m == "human" else m
    lines.append(f"| {name} | {int(r.n_wrong)} | {r.mle_dprime:.2f} | {r.mle_metad:.2f} | "
                 f"{r.mle_m_ratio:.2f} | {r.mr_mean:.2f} [{r.hdi_lo:.2f}, {r.hdi_hi:.2f}] | {LAB[r.evidence]} |")
md = "\n".join(lines)
open(os.path.join(A, "mle_vs_bayes_table.md"), "w").write(md)
print(md)

# ---- 不稳定性图:MLE M-ratio vs 错误数(对数轴)----
COL = {"data-driven": "#1f77b4", "regularized": "#7fb3d5", "prior-dominated": "#c9c9c9"}
plt.figure(figsize=(9, 5.5))
mods = hm.drop(index="human", errors="ignore")
for m, r in mods.iterrows():
    plt.scatter(max(r.n_wrong, 0.5), r.mle_m_ratio, s=80, color=COL[r.evidence],
                edgecolor="k", lw=.5, zorder=3)
plt.axvline(30, ls="--", c="red", lw=1, label="reliability threshold (30 errors)")
plt.axhline(1.0, ls=":", c="gray", lw=1)
plt.xscale("log")
plt.xlabel("Number of error trials (log scale)")
plt.ylabel("MLE M-ratio (forced point estimate)")
plt.title("Why the MLE M-ratio is unusable for most models:\n"
          "with <30 errors it scatters randomly between ~0.3 and ~1.0")
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color="#1f77b4", label="≥30 errors (usable)"),
                    Patch(color="#7fb3d5", label="10–29 errors"),
                    Patch(color="#c9c9c9", label="<10 errors (unusable)"),
                    plt.Line2D([0],[0],ls="--",c="red",label="30-error threshold")],
           fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(F, "c_mle_instability.png"), bbox_inches="tight"); plt.close()
print("\n图:results/figures/c_mle_instability.png")
