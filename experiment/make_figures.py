#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — 从 master_long.csv 生成 PPT/论文用图(PNG)。
只用 status=='ok';自动剔除可用数据太少的模型(死数据)。图用英文标签避免字体问题。
出:results/figures/*.png
"""
import os, sys
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
CMAP = {1: .50, 2: .625, 3: .75, 4: .875, 5: 1.0}   # 置信度→数值(判断题下限50%)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})

d = pd.read_csv(os.path.join(RES, "master_long.csv"))
d = d[d.status == "ok"].copy()
d["correct"] = d.correct.astype(str) == "True"
d["conf"] = pd.to_numeric(d.confidence, errors="coerce")
d = d[d.conf.isin([1, 2, 3, 4, 5])]
d["stated"] = d.conf.map(CMAP)

# 剔除死数据 / 污染:每模型可用行数 < 500 的丢掉;本地 qwen3 nothink 已知污染,排除
counts = d.groupby("model").size()
keep = counts[counts >= 500].index.tolist()
CONTAMINATED = {"local-qwen3-1.7b-nothink", "local-qwen3-4b-nothink", "local-qwen3-14b-nothink"}
keep = [m for m in keep if m not in CONTAMINATED]
d = d[d.model.isin(keep)].copy()

# 颜色:人类红,其余按厂商
VCOL = {"human": "#d62728", "openai": "#2ca02c", "deepseek": "#1f77b4",
        "alibaba": "#9467bd", "google(local)": "#ff7f0e", "other": "#8c564b"}
def color(m):
    return VCOL.get(d[d.model == m].vendor.iloc[0], "#7f7f7f")

def ece(g):
    tot = len(g); e = 0
    for c, sub in g.groupby("conf"):
        e += abs(sub.correct.mean() - CMAP[c]) * len(sub)
    return e / tot

# 每模型汇总
rows = []
for m, g in d.groupby("model"):
    rows.append(dict(model=m, vendor=g.vendor.iloc[0], tier=g.tier.iloc[0],
                     n=len(g), acc=g.correct.mean(), mean_conf=g.conf.mean(),
                     ece=ece(g), overconf=g.stated.mean() - g.correct.mean(),
                     conf5_wrong=int((~g[g.conf == 5].correct).sum())))
S = pd.DataFrame(rows).set_index("model")
S.to_csv(os.path.join(RES, "figure_summary.csv"))
order = S.sort_values("acc").index.tolist()   # 按准确率排序

def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close()
    print("  ✓", name)

# ---------- 图1:各组准确率 ----------
plt.figure(figsize=(10, 6))
y = range(len(order)); accs = [S.loc[m, "acc"] for m in order]
plt.barh(list(y), accs, color=[color(m) for m in order])
plt.yticks(list(y), order, fontsize=9)
plt.axvline(0.5, ls="--", c="gray", lw=1, label="chance (0.5)")
plt.xlabel("Accuracy"); plt.title("Accuracy by group (humans vs LLMs)")
plt.xlim(0.4, 1.02); plt.legend()
for i, m in enumerate(order):
    plt.text(S.loc[m, "acc"] + .005, i, f"{S.loc[m,'acc']:.2f}", va="center", fontsize=8)
savefig("fig1_accuracy_by_group.png")

# ---------- 图2:置信度分布(人类 vs 代表模型)----------
reps = [x for x in ["human", "local-gemma-e2b", "gpt-4o", "gpt-5.5", "qwen3.7-max"] if x in S.index]
plt.figure(figsize=(10, 5))
w = 0.15
for j, m in enumerate(reps):
    g = d[d.model == m]; dist = [(g.conf == c).mean() for c in [1, 2, 3, 4, 5]]
    plt.bar([c + (j - len(reps) / 2) * w for c in range(1, 6)], dist, width=w,
            label=m, color=color(m))
plt.xticks(range(1, 6), ["1 guess", "2", "3", "4", "5 very sure"])
plt.ylabel("Proportion of responses"); plt.title("Confidence distribution")
plt.legend(fontsize=9)
savefig("fig2_confidence_distribution.png")

# ---------- 图3:ECE ----------
plt.figure(figsize=(10, 6))
eces = [S.loc[m, "ece"] for m in order]
plt.barh(list(range(len(order))), eces, color=[color(m) for m in order])
plt.yticks(range(len(order)), order, fontsize=9)
plt.xlabel("Expected Calibration Error (lower = better)")
plt.title("Calibration error by group")
savefig("fig3_ece_by_group.png")

# ---------- 图4:可靠性/校准曲线 ----------
plt.figure(figsize=(7, 7))
plt.plot([.5, 1], [.5, 1], "k--", lw=1, label="perfect calibration")
for m in reps:
    g = d[d.model == m]; xs, ys = [], []
    for c in [1, 2, 3, 4, 5]:
        sub = g[g.conf == c]
        if len(sub) >= 5:
            xs.append(CMAP[c]); ys.append(sub.correct.mean())
    plt.plot(xs, ys, "-o", color=color(m), label=m, ms=5)
plt.xlabel("Stated confidence (mapped)"); plt.ylabel("Actual accuracy")
plt.title("Reliability curves"); plt.legend(fontsize=9); plt.xlim(.45, 1.02); plt.ylim(.3, 1.02)
savefig("fig4_reliability_curves.png")

# ---------- 图5:按难度准确率(人类 vs 模型均值)----------
diffs = ["易", "中", "难"]; labels = ["easy", "medium", "hard"]
hum = d[d.model == "human"]; mod = d[d.model != "human"]
plt.figure(figsize=(7, 5))
hv = [hum[hum.difficulty == x].correct.mean() for x in diffs]
mv = [mod[mod.difficulty == x].correct.mean() for x in diffs]
x = np.arange(3)
plt.bar(x - .2, hv, .4, label="Humans", color=VCOL["human"])
plt.bar(x + .2, mv, .4, label="LLMs (avg)", color="#4477aa")
plt.xticks(x, labels); plt.ylabel("Accuracy"); plt.ylim(0, 1.05)
plt.axhline(.5, ls="--", c="gray", lw=1); plt.title("Accuracy by difficulty")
plt.legend()
savefig("fig5_accuracy_by_difficulty.png")

# ---------- 图6:按题型准确率(突出幻觉题)----------
types = ["常规", "学科", "陷阱", "幻觉"]; tlabels = ["Ordinary", "Discipline", "Trap", "Hallucination"]
plt.figure(figsize=(8, 5))
hv = [hum[hum.type == t].correct.mean() for t in types]
mv = [mod[mod.type == t].correct.mean() for t in types]
x = np.arange(4)
plt.bar(x - .2, hv, .4, label="Humans", color=VCOL["human"])
plt.bar(x + .2, mv, .4, label="LLMs (avg)", color="#4477aa")
plt.xticks(x, tlabels); plt.ylabel("Accuracy"); plt.ylim(0, 1.05)
plt.axhline(.5, ls="--", c="gray", lw=1); plt.title("Accuracy by item type")
plt.legend()
savefig("fig6_accuracy_by_type.png")

# ---------- 图7:过度自信 vs 准确率(关键图)----------
plt.figure(figsize=(8, 6))
for m in S.index:
    plt.scatter(S.loc[m, "acc"], S.loc[m, "overconf"], color=color(m), s=70,
                edgecolor="k", lw=.5, zorder=3)
    plt.annotate(m, (S.loc[m, "acc"], S.loc[m, "overconf"]), fontsize=7,
                 xytext=(3, 3), textcoords="offset points")
plt.axhline(0, c="gray", lw=1)
plt.xlabel("Accuracy (first-order ability)")
plt.ylabel("Overconfidence  (stated confidence − accuracy)")
plt.title("Overconfidence vs. ability:\nweak respondents overconfident, strong ones slightly under")
savefig("fig7_overconfidence_vs_accuracy.png")

# ---------- 图8:think vs nothink(API 干净对照)----------
pairs = [("qwen3.7-max", "qwen3.7-max-nothink"), ("qwen3.7-plus", "qwen3.7-plus-nothink"),
         ("deepseek-v4-pro", "deepseek-v4-pro-nothink"), ("deepseek-v4-flash", "deepseek-v4-flash-nothink")]
pairs = [(a, b) for a, b in pairs if a in S.index and b in S.index]
if pairs:
    plt.figure(figsize=(9, 5))
    x = np.arange(len(pairs))
    tk = [S.loc[a, "conf5_wrong"] for a, b in pairs]
    nt = [S.loc[b, "conf5_wrong"] for a, b in pairs]
    plt.bar(x - .2, tk, .4, label="think", color="#4477aa")
    plt.bar(x + .2, nt, .4, label="no-think", color="#ee6677")
    plt.xticks(x, [a for a, b in pairs], rotation=15, fontsize=9)
    plt.ylabel("High-confidence errors (conf=5 & wrong)")
    plt.title("Think vs no-think: confident errors")
    plt.legend()
    savefig("fig8_think_vs_nothink.png")

# ---------- 图9:幻觉题 虚构 vs 真实 的信心 ----------
h = d[d.type == "幻觉"].copy()
h["kind"] = np.where(h.gold == "False", "fictional (should=False)", "real-obscure (should=True)")
grp = [x for x in ["human", "gpt-5.5", "deepseek-v4-pro", "qwen3.7-max"] if x in S.index]
plt.figure(figsize=(8, 5))
x = np.arange(len(grp)); w = .35
fic = [h[(h.model == m) & (h.kind.str.startswith("fictional"))].conf.mean() for m in grp]
real = [h[(h.model == m) & (h.kind.str.startswith("real"))].conf.mean() for m in grp]
plt.bar(x - w/2, fic, w, label="fictional (unknowable)", color="#cc6677")
plt.bar(x + w/2, real, w, label="real-obscure", color="#44aa99")
plt.xticks(x, grp, fontsize=9); plt.ylabel("Mean confidence (1-5)")
plt.title("Hallucination items: confidence on fictional vs real entities")
plt.legend()
savefig("fig9_hallucination_confidence.png")

print(f"\n完成。图在 {FIG}/  |  每模型汇总在 results/figure_summary.csv")
print("纳入分析的组:", ", ".join(order))
