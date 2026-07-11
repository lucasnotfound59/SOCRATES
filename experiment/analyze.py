# -*- coding: utf-8 -*-
"""核心一阶 + 校准分析。输出整洁 csv 到 results/analysis/。"""
import os, pandas as pd, numpy as np
from analysis_common import load_clean, ece, group_order, OUT, CMAP

d = load_clean()
order = group_order(d)
TYPES = ["常规", "学科", "陷阱", "幻觉"]
DIFF  = ["易", "中", "难"]

# ---------- 1) 每组总览 ----------
rows = []
for m, g in d.groupby("model"):
    conf5 = g[g.conf == 5]
    rows.append(dict(
        model=m, vendor=g.vendor.iloc[0], tier=g.tier.iloc[0], thinking=bool(g.thinking.iloc[0]),
        condition=g.condition.iloc[0], n=len(g),
        acc=g.correct.mean(), mean_conf=g.conf.mean(), stated=g.stated.mean(),
        overconf=g.stated.mean() - g.correct.mean(), ece=ece(g),
        conf5_n=len(conf5), conf5_acc=conf5.correct.mean() if len(conf5) else np.nan,
        conf5_wrong=int((~conf5.correct).sum()),
        n_wrong=int((~g.correct).sum()), err_rate=1 - g.correct.mean(),
    ))
summ = pd.DataFrame(rows).set_index("model").loc[order]
summ.to_csv(os.path.join(OUT, "group_summary.csv"))

# ---------- 2) 按类型 / 难度 准确率 & 置信度 ----------
def by_factor(col, levels):
    recs = []
    for m, g in d.groupby("model"):
        r = {"model": m}
        for lv in levels:
            s = g[g[col] == lv]
            r[f"{lv}_acc"] = s.correct.mean() if len(s) else np.nan
            r[f"{lv}_conf"] = s.conf.mean() if len(s) else np.nan
            r[f"{lv}_overconf"] = (s.stated.mean() - s.correct.mean()) if len(s) else np.nan
            r[f"{lv}_n"] = len(s)
        recs.append(r)
    return pd.DataFrame(recs).set_index("model").loc[order]

by_factor("type", TYPES).to_csv(os.path.join(OUT, "by_type.csv"))
by_factor("difficulty", DIFF).to_csv(os.path.join(OUT, "by_difficulty.csv"))

# ---------- 3) 置信度分布 ----------
dist = []
for m, g in d.groupby("model"):
    r = {"model": m}
    for c in [1, 2, 3, 4, 5]:
        r[f"p_conf{c}"] = (g.conf == c).mean()
    dist.append(r)
pd.DataFrame(dist).set_index("model").loc[order].to_csv(os.path.join(OUT, "confidence_dist.csv"))

# ---------- 4) 可靠性(每组每档:声称概率 vs 实际正确率)----------
rel = []
for m, g in d.groupby("model"):
    for c in [1, 2, 3, 4, 5]:
        s = g[g.conf == c]
        if len(s):
            rel.append(dict(model=m, conf=c, stated=CMAP[c], acc=s.correct.mean(), n=len(s)))
pd.DataFrame(rel).to_csv(os.path.join(OUT, "reliability.csv"), index=False)

# ---------- 5) 幻觉题:虚构(gold=False)vs 真实冷僻(gold=True)----------
h = d[d.type == "幻觉"]
hall = []
for m, g in h.groupby("model"):
    fic = g[g.gold == "False"]; real = g[g.gold == "True"]
    hall.append(dict(model=m,
        fic_n=len(fic), fic_acc=fic.correct.mean() if len(fic) else np.nan,
        fic_conf=fic.conf.mean() if len(fic) else np.nan,
        real_n=len(real), real_acc=real.correct.mean() if len(real) else np.nan,
        real_conf=real.conf.mean() if len(real) else np.nan,
        conf_gap=(real.conf.mean() - fic.conf.mean()) if len(fic) and len(real) else np.nan))
pd.DataFrame(hall).set_index("model").reindex([m for m in order if m in h.model.unique()]
    ).to_csv(os.path.join(OUT, "hallucination.csv"))

# ---------- 6) 人类个体(每被试)分布,用于组内异质性 ----------
hum = d[d.model == "human"]
ind = []
for pid, g in hum.groupby("participant_id"):
    if len(g) < 20: continue
    ind.append(dict(participant_id=pid, n=len(g), acc=g.correct.mean(),
                    mean_conf=g.conf.mean(), overconf=g.stated.mean() - g.correct.mean(),
                    ece=ece(g), conf5_wrong=int((~g[g.conf == 5].correct).sum())))
pd.DataFrame(ind).to_csv(os.path.join(OUT, "human_individuals.csv"), index=False)

print("== 组总览(按层级+准确率)==")
show = summ[["tier", "n", "acc", "mean_conf", "stated", "overconf", "ece", "err_rate", "conf5_wrong"]]
pd.set_option("display.width", 200, "display.max_rows", 60)
print(show.round(3).to_string())
print(f"\n纳入 {len(order)} 组;人类被试 {len(ind)} 名。输出 → results/analysis/")
