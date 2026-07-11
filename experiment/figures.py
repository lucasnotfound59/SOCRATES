# -*- coding: utf-8 -*-
"""对比分析成图(读 results/analysis/*.csv)。英文标签避免字体问题。出 results/figures/c_*.png。"""
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "results", "analysis")
F = os.path.join(HERE, "results", "figures"); os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})
HUMAN = "#d62728"; MODEL = "#3b6ea5"; ACCENT = "#e0a458"

gs = pd.read_csv(os.path.join(A, "group_summary.csv")).set_index("model")
bs = pd.read_csv(os.path.join(A, "bootstrap_summary.csv")).set_index("model")
md = pd.read_csv(os.path.join(A, "metad_summary.csv")).set_index("model")
rel = pd.read_csv(os.path.join(A, "reliability.csv"))
byt = pd.read_csv(os.path.join(A, "by_type.csv")).set_index("model")
hl = pd.read_csv(os.path.join(A, "hallucination.csv")).set_index("model")
tn = pd.read_csv(os.path.join(A, "think_nothink.csv"))
order = gs.index.tolist()
cols = [HUMAN if m == "human" else MODEL for m in order]

def save(n):
    plt.tight_layout(); plt.savefig(os.path.join(F, n), bbox_inches="tight"); plt.close()
    print("  ✓", n)

# 1) 准确率
plt.figure(figsize=(9, 6))
y = range(len(order))
plt.barh(list(y), gs.acc.values, color=cols)
plt.yticks(list(y), order, fontsize=8); plt.axvline(.5, ls="--", c="gray", lw=1, label="chance")
plt.xlim(.4, 1.02); plt.xlabel("Accuracy"); plt.title("First-order accuracy: humans vs LLMs")
plt.legend(); save("c_accuracy.png")

# 2) ECE + 95%CI
plt.figure(figsize=(9, 6))
err = np.abs(np.vstack([bs.ece - bs.ece_lo, bs.ece_hi - bs.ece]))
plt.barh(list(y), bs.ece.loc[order].values, color=cols,
         xerr=err[:, [order.index(m) for m in order]] if False else None)
# 逐条画误差棒(顺序对齐)
for i, m in enumerate(order):
    plt.plot([bs.loc[m, "ece_lo"], bs.loc[m, "ece_hi"]], [i, i], color="k", lw=1)
plt.yticks(list(y), order, fontsize=8); plt.xlabel("Expected Calibration Error (95% CI)")
plt.title("Calibration error (lower = better): humans worst calibrated"); save("c_ece.png")

# 3) M-ratio(可估组)+ 95%CI —— 关键图
est = bs[bs.m_ratio.notna()].sort_values("m_ratio")
plt.figure(figsize=(8, 4.6))
yy = range(len(est))
c2 = [HUMAN if m == "human" else MODEL for m in est.index]
plt.barh(list(yy), est.m_ratio.values, color=c2)
for i, m in enumerate(est.index):
    plt.plot([est.loc[m, "m_lo"], est.loc[m, "m_hi"]], [i, i], color="k", lw=1.2)
plt.axvline(1.0, ls="--", c="gray", lw=1, label="M-ratio = 1 (meta-d′ = d′)")
plt.yticks(list(yy), est.index, fontsize=9)
plt.xlabel("M-ratio = meta-d′ / d′  (95% CI)")
plt.title("Metacognitive efficiency (only groups with enough errors)\n"
          "Humans efficient (>1); measurable models inefficient (<1)")
plt.legend(fontsize=9); save("c_mratio.png")

# 4) meta-d′ vs d′(可估组)
e = md[md.estimable == True]
plt.figure(figsize=(6.4, 6))
lim = [0, max(e.dprime.max(), e.meta_d.max()) * 1.1]
plt.plot(lim, lim, "k--", lw=1, label="meta-d′ = d′")
for m in e.index:
    col = HUMAN if m == "human" else MODEL
    plt.scatter(e.loc[m, "dprime"], e.loc[m, "meta_d"], s=90, color=col, edgecolor="k", lw=.5, zorder=3)
    plt.annotate(m, (e.loc[m, "dprime"], e.loc[m, "meta_d"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
plt.xlabel("d′ (first-order sensitivity)"); plt.ylabel("meta-d′ (metacognitive sensitivity)")
plt.title("meta-d′ vs d′: human above identity, models below"); plt.legend(); save("c_metad_vs_dprime.png")

# 5) 可靠性曲线(代表组)
reps = [m for m in ["human", "local-gemma-e2b", "local-qwen3-1.7b", "gpt-4o", "gpt-5.5", "qwen3.7-max"] if m in order]
plt.figure(figsize=(6.4, 6.2))
plt.plot([.5, 1], [.5, 1], "k--", lw=1, label="perfect calibration")
for m in reps:
    s = rel[rel.model == m]
    col = HUMAN if m == "human" else None
    plt.plot(s.stated, s.acc, "-o", ms=5, label=m, color=col)
plt.xlim(.45, 1.02); plt.ylim(.3, 1.03); plt.xlabel("Stated confidence (mapped)")
plt.ylabel("Actual accuracy"); plt.title("Reliability curves"); plt.legend(fontsize=8); save("c_reliability.png")

# 6) 过度自信 vs 准确率(描述性)
plt.figure(figsize=(8, 6))
for m in order:
    col = HUMAN if m == "human" else MODEL
    plt.scatter(gs.loc[m, "acc"], gs.loc[m, "overconf"], s=70, color=col, edgecolor="k", lw=.5, zorder=3)
    if m in ("human",) or gs.loc[m, "overconf"] > .01 or gs.loc[m, "acc"] < .95:
        plt.annotate(m, (gs.loc[m, "acc"], gs.loc[m, "overconf"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.axhline(0, c="gray", lw=1); plt.xlabel("Accuracy"); plt.ylabel("Overconfidence (stated − accuracy)")
plt.title("Overconfidence vs accuracy (descriptive; ρ=−0.32, p=0.18 n.s.)"); save("c_overconfidence.png")

# 7) 按题型准确率:人类 vs 模型均值
types = ["常规", "学科", "陷阱", "幻觉"]; tl = ["Ordinary", "Discipline", "Trap", "Hallucination"]
hum_acc = [byt.loc["human", f"{t}_acc"] for t in types]
mod_acc = [byt.drop("human")[f"{t}_acc"].mean() for t in types]
x = np.arange(4)
plt.figure(figsize=(7.5, 5))
plt.bar(x - .2, hum_acc, .4, label="Humans", color=HUMAN)
plt.bar(x + .2, mod_acc, .4, label="LLMs (avg)", color=MODEL)
plt.axhline(.5, ls="--", c="gray", lw=1); plt.xticks(x, tl); plt.ylim(0, 1.05)
plt.ylabel("Accuracy"); plt.title("Accuracy by item type: humans collapse only on Hallucination")
plt.legend(); save("c_accuracy_by_type.png")

# 8) 幻觉题:虚构 vs 真实 准确率(人类 below chance)
grp = [m for m in ["human", "local-gemma-e2b", "local-qwen3-1.7b", "local-gemma-e4b", "gpt-4o", "gpt-5.5", "qwen3.7-max"] if m in hl.index]
x = np.arange(len(grp)); w = .38
plt.figure(figsize=(9, 5))
plt.bar(x - w/2, [hl.loc[m, "fic_acc"] for m in grp], w, label="Fictional (unknowable, should=False)", color="#cc6677")
plt.bar(x + w/2, [hl.loc[m, "real_acc"] for m in grp], w, label="Real-obscure (should=True)", color="#44aa99")
plt.axhline(.5, ls="--", c="gray", lw=1, label="chance")
plt.xticks(x, grp, rotation=20, fontsize=8, ha="right"); plt.ylim(0, 1.05); plt.ylabel("Accuracy")
plt.title("Hallucination items: only humans fall below chance on fictional entities"); plt.legend(fontsize=8)
save("c_hallucination_acc.png")

# 9) 错误数(天花板):为何多数模型 meta-d′ 不可估
plt.figure(figsize=(9, 6))
nw = gs["n_wrong"].loc[order]
barcol = [HUMAN if m == "human" else (ACCENT if bs.loc[m, "m_ratio"] == bs.loc[m, "m_ratio"] else MODEL) for m in order]
plt.barh(list(y), nw.values, color=barcol)
plt.axvline(30, ls="--", c="red", lw=1, label="min errors for reliable meta-d′ (30)")
plt.yticks(list(y), order, fontsize=8); plt.xlabel("Number of error trials")
plt.title("Ceiling effect: most models have too few errors to estimate metacognition\n(orange = estimable)")
plt.legend(); save("c_errors_ceiling.png")

# 10) think vs nothink:自信错误
if len(tn):
    x = np.arange(len(tn));
    plt.figure(figsize=(8, 5))
    plt.bar(x - .2, tn.conf5wrong_think, .4, label="think", color=MODEL)
    plt.bar(x + .2, tn.conf5wrong_nothink, .4, label="no-think", color="#ee6677")
    plt.xticks(x, tn.pair, rotation=12, fontsize=9); plt.ylabel('High-confidence errors (conf=5 & wrong)')
    plt.title("Think vs no-think: disabling thinking adds a few confident errors"); plt.legend()
    save("c_think_nothink.png")

print("完成,图在 results/figures/c_*.png")
