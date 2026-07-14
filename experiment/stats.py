# -*- coding: utf-8 -*-
"""统计检验:自助法CI、type-2 AUC(天花板稳健)、幻觉题检验、think对照、H1/H2/H3裁决。"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, binomtest, spearmanr
from metadpy.mle import metad
from analysis_common import load_clean, ece, group_order, CMAP, OUT

rng = np.random.default_rng(42)
d = load_clean()
order = group_order(d)

# ---------- type-2 AUC:正确 trial 的置信度高于错误 trial 的概率(天花板也可算,需≥1错)----------
def auroc2(g):
    c = g[g.correct].conf.values; w = g[~g.correct].conf.values
    if len(c) == 0 or len(w) == 0: return np.nan
    U = mannwhitneyu(c, w, alternative="greater").statistic
    return U / (len(c) * len(w))

# ---------- meta-d′ 拟合(用于自助)----------
def fit_mratio(g):
    df = pd.DataFrame({"Stimuli": (g.gold == "True").astype(int),
                       "Accuracy": g.correct.astype(int), "Confidence": g.conf.astype(int)})
    try:
        r = metad(data=df, nRatings=5, stimuli="Stimuli", accuracy="Accuracy",
                  confidence="Confidence"); return float(r["m_ratio"].iloc[0])
    except Exception:
        return np.nan

ESTIMABLE = ["human", "local-gemma-e2b", "local-qwen3-1.7b", "local-qwen3-4b", "local-gemma-e4b"]
NBOOT_E = 800

def boot_ci(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if len(v) > 10 else (np.nan, np.nan)

rows = []
for m in order:
    g = d[d.model == m].reset_index(drop=True)
    # 聚类自助:人类按被试聚类;模型按题目聚类(每题 4 个采样相关,trial 级重采样会低估不确定性)
    clust_col = "participant_id" if m == "human" else "item_id"
    clust_idx = [g.index[g[clust_col] == c].values for c in g[clust_col].unique()]
    def draw():
        return g.loc[np.concatenate([clust_idx[i] for i in
                     rng.integers(0, len(clust_idx), len(clust_idx))])]
    # ECE 自助(聚类重采样)
    e_lo, e_hi = boot_ci([ece(draw()) for _ in range(NBOOT_E)])
    rows.append(dict(model=m, tier=g.tier.iloc[0], vendor=g.vendor.iloc[0],
                     n=len(g), acc=g.correct.mean(),
                     overconf=g.stated.mean() - g.correct.mean(),
                     ece=ece(g), ece_lo=e_lo, ece_hi=e_hi,
                     auroc2=auroc2(g), estimable=(m in ESTIMABLE)))
B = pd.DataFrame(rows).set_index("model")
# 并入分组预算的 M-ratio 自助CI(boot_mratio.py 产出)
mr = pd.read_csv(os.path.join(OUT, "boot_mratio.csv")).set_index("model")
B = B.join(mr[["m_ratio", "m_lo", "m_hi"]])
B.to_csv(os.path.join(OUT, "bootstrap_summary.csv"))

# ---------- 能力↔过度自信 的翻转(跨组)----------
rho, p_rho = spearmanr(B.acc, B.overconf)

# ---------- 幻觉题:虚构 vs 真实 信心检验 + 人类below-chance ----------
h = d[d.type == "幻觉"]
htests = []
for m in order:
    g = h[h.model == m]
    fic = g[g.gold == "False"]; real = g[g.gold == "True"]
    if len(fic) < 5 or len(real) < 5: continue
    U = mannwhitneyu(real.conf, fic.conf, alternative="greater")
    bt = binomtest(int(fic.correct.sum()), len(fic), 0.5)  # 虚构题是否偏离随机
    htests.append(dict(model=m, fic_acc=fic.correct.mean(), fic_conf=fic.conf.mean(),
                       real_conf=real.conf.mean(), conf_drop=real.conf.mean() - fic.conf.mean(),
                       U_p=U.pvalue, fic_vs_chance_p=bt.pvalue,
                       below_chance=(fic.correct.mean() < 0.5)))
pd.DataFrame(htests).set_index("model").to_csv(os.path.join(OUT, "hallucination_tests.csv"))

# ---------- think vs nothink 受控对照 ----------
pairs = [("qwen3.7-max", "qwen3.7-max-nothink"), ("qwen3.7-plus", "qwen3.7-plus-nothink"),
         ("deepseek-v4-pro", "deepseek-v4-pro-nothink"),
         ("deepseek-v4-flash", "deepseek-v4-flash-nothink")]
tk = []
for a, b in pairs:
    if a not in B.index or b not in B.index: continue
    ga, gb = d[d.model == a], d[d.model == b]
    tk.append(dict(pair=a, acc_think=ga.correct.mean(), acc_nothink=gb.correct.mean(),
                   conf5wrong_think=int((~ga[ga.conf == 5].correct).sum()),
                   conf5wrong_nothink=int((~gb[gb.conf == 5].correct).sum()),
                   ece_think=ece(ga), ece_nothink=ece(gb),
                   auroc2_think=auroc2(ga), auroc2_nothink=auroc2(gb)))
pd.DataFrame(tk).to_csv(os.path.join(OUT, "think_nothink.csv"), index=False)

# ---------- 打印摘要 ----------
pd.set_option("display.width", 200)
print("== 各组:ECE / type-2 AUC / M-ratio(含95%自助CI)==")
show = B[["tier", "acc", "overconf", "ece", "ece_lo", "ece_hi", "auroc2", "m_ratio", "m_lo", "m_hi"]]
print(show.round(3).to_string())
print(f"\n能力↔过度自信 Spearman ρ = {rho:.3f} (p={p_rho:.4f}) —— 负值=越强越不过度自信")
print("\n== 幻觉题 虚构vs真实(conf_drop=真实−虚构信心;below_chance=虚构题是否低于随机)==")
print(pd.DataFrame(htests).set_index("model")[["fic_acc","conf_drop","below_chance","fic_vs_chance_p"]].round(3).to_string())
hum = [t for t in htests if t["model"] == "human"][0]
print(f"\n人类虚构题准确率 {hum['fic_acc']:.3f},vs 随机 p={hum['fic_vs_chance_p']:.2e} "
      f"({'显著低于随机' if hum['below_chance'] else '未低于随机'})")
print("\n== think vs nothink ==")
print(pd.DataFrame(tk).round(3).to_string(index=False))
