# -*- coding: utf-8 -*-
"""三项稳健性检查(均不需重跑模型):
1) 模型组 M-ratio 自助CI 改为按题目(item_id)聚类重采样——每题 4 个采样相关,
   trial 级重采样会低估不确定性;与人类"按被试聚类"对称。
2) ECE 档位映射敏感性:主映射 50/62.5/75/87.5/100 之外换 3 种映射重算,
   检验"人类校准最差"的结论是否依赖映射选择。
3) 幻觉虚构题题目级检查:人类低于随机是否由少数极端题驱动
   (按题准确率分布 + 剔除最差 10% 题后重算 binomial p)。
输出:results/analysis/robustness_checks.md + boot_mratio_itemclust.csv
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import binomtest
from metadpy.mle import metad
from analysis_common import load_clean, OUT

d=load_clean()
lines=["# 稳健性检查(robustness_checks.py 产出)\n"]

# ---------- 1) 按题聚类的 M-ratio 自助CI(4 个可估模型) ----------
NB=200
def fit_mratio(gg):
    df=pd.DataFrame({"Stimuli": (gg.gold=="True").astype(int),
                     "Accuracy": gg.correct.astype(int),
                     "Confidence": gg.conf.astype(int)})
    try:
        r=metad(data=df, nRatings=5, stimuli="Stimuli", accuracy="Accuracy",
                confidence="Confidence")
        return float(r["m_ratio"].iloc[0])
    except Exception:
        return np.nan

rows=[]
lines.append("## 1) 模型 M-ratio 自助CI:按题目聚类 vs 原 trial 级\n")
for m in ["local-gemma-e4b", "local-qwen3-4b", "local-gemma-e2b", "local-qwen3-1.7b"]:
    g=d[d.model==m].reset_index(drop=True)
    rng=np.random.default_rng(abs(hash(m))%(2**32))
    item_idx=[g.index[g.item_id==i].values for i in g.item_id.unique()]
    vals=[]
    for _ in range(NB):
        pick=rng.integers(0, len(item_idx), len(item_idx))
        vals.append(fit_mratio(g.loc[np.concatenate([item_idx[i] for i in pick])]))
    vals=np.array([v for v in vals if np.isfinite(v)])
    lo, hi=np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)
    point=fit_mratio(g)
    rows.append(dict(model=m, m_ratio=point, m_lo=lo, m_hi=hi, nboot=len(vals)))
    lines.append(f"- {m}: M-ratio={point:.3f} 题聚类95%CI[{lo:.3f}, {hi:.3f}] (n={len(vals)})")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "boot_mratio_itemclust.csv"), index=False)
prev=pd.read_csv(os.path.join(OUT, "boot_mratio.csv")).set_index("model")
for r in rows:
    o=prev.loc[r["model"]]
    lines.append(f"  (原 trial 级 [{o.m_lo:.3f}, {o.m_hi:.3f}])")
lines.append("- 人类 CI(按被试聚类,不变):[1.205, 1.543]。判定:若所有模型题聚类上界仍 <1.205,结论不变。\n")

# ---------- 2) ECE 映射敏感性 ----------
MAPS={"主映射 50/62.5/75/87.5/100": {1:.50,2:.625,3:.75,4:.875,5:1.0},
      "线性满幅 0/25/50/75/100":     {1:.00,2:.25,3:.50,4:.75,5:1.0},
      "居中 55/65/75/85/95":         {1:.55,2:.65,3:.75,4:.85,5:.95},
      "顶部保守 50/62.5/75/87.5/97.5":{1:.50,2:.625,3:.75,4:.875,5:.975}}
def ece_with(sub, cmap):
    tot=len(sub); e=0.0
    for c, g in sub.groupby("conf"):
        e+=abs(g.correct.mean()-cmap[c])*len(g)
    return e/tot
lines.append("## 2) ECE 映射敏感性(各映射下:人类 ECE / 模型最大 ECE / 人类是否仍最差)\n")
for name, cmap in MAPS.items():
    per={m: ece_with(g, cmap) for m, g in d.groupby("model")}
    h=per.pop("human"); mx=max(per.values()); mxm=max(per, key=per.get)
    worst=h>mx
    lines.append(f"- {name}: 人类 {h:.3f} vs 模型最大 {mx:.3f}({mxm}) → 人类最差: {worst}")
lines.append("")

# ---------- 3) 幻觉虚构题题目级检查(人类) ----------
h=d[(d.model=="human")&(d.type=="幻觉")&(d.gold=="False")]
by_item=h.groupby("item_id").agg(n=("correct","size"), acc=("correct","mean"))
by_item=by_item[by_item.n>=2]
lines.append("## 3) 人类虚构题:题目级分布(是否少数极端题驱动)\n")
lines.append(f"- 覆盖 {len(by_item)} 道虚构题(每题≥2 人作答);按题准确率:中位数 {by_item.acc.median():.3f},"
             f" 均值 {by_item.acc.mean():.3f}")
lines.append(f"- 低于 50% 的题占 {(by_item.acc<0.5).mean():.1%};全对的题 {(by_item.acc==1).sum()} 道,"
             f"全错的题 {(by_item.acc==0).sum()} 道")
n_drop=max(1, int(np.ceil(len(by_item)*0.10)))
worst_items=by_item.acc.nsmallest(n_drop).index
kept=h[~h.item_id.isin(worst_items)]
k=int(kept.correct.sum()); n=len(kept)
p=binomtest(k, n, 0.5, alternative="less").pvalue
lines.append(f"- 剔除最差 {n_drop} 题({n_drop/len(by_item):.0%})后:acc={k/n:.3f}(n={n}),"
             f" binomial p={p:.2e} → 仍{'显著' if p<0.001 else '不显著'}低于随机\n")

out=os.path.join(OUT, "robustness_checks.md")
open(out, "w").write("\n".join(lines))
print("\n".join(lines))
