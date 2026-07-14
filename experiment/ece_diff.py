# -*- coding: utf-8 -*-
"""人类 vs 每个模型的 ECE 差值自助检验(比较边缘 CI 是否重叠是错误做法;
正确做法:对差值 Δ=ECE_human−ECE_model 做自助分布,看 95%CI 是否排除 0)。
聚类:人类按被试,模型按题目。输出 results/analysis/ece_diff.csv。
另附:4 种档位映射下的差值检验(映射敏感性)。
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from analysis_common import load_clean, OUT

NB=800
d=load_clean()
MAPS={"primary": {1:.50,2:.625,3:.75,4:.875,5:1.0},
      "linear_full": {1:.00,2:.25,3:.50,4:.75,5:1.0},
      "centered": {1:.55,2:.65,3:.75,4:.85,5:.95},
      "top_conservative": {1:.50,2:.625,3:.75,4:.875,5:.975}}

def ece_with(sub, cmap):
    tot=len(sub); e=0.0
    for c, g in sub.groupby("conf"):
        e+=abs(g.correct.mean()-cmap[c])*len(g)
    return e/tot

def boot_eces(g, clust_col, rng):
    idx=[g.index[g[clust_col]==c].values for c in g[clust_col].unique()]
    out={k: [] for k in MAPS}
    for _ in range(NB):
        gg=g.loc[np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])]
        for k, cmap in MAPS.items():
            out[k].append(ece_with(gg, cmap))
    return {k: np.array(v) for k, v in out.items()}

rng=np.random.default_rng(42)
gh=d[d.model=="human"].reset_index(drop=True)
hb=boot_eces(gh, "participant_id", rng)

rows=[]
for m in sorted(d.model.unique()):
    if m=="human":
        continue
    gm=d[d.model==m].reset_index(drop=True)
    mb=boot_eces(gm, "item_id", rng)
    row=dict(model=m)
    for k, cmap in MAPS.items():
        diff=hb[k]-mb[k]
        row[f"d_{k}"]=ece_with(gh, cmap)-ece_with(gm, cmap)
        row[f"d_{k}_lo"]=np.percentile(diff, 2.5)
        row[f"d_{k}_hi"]=np.percentile(diff, 97.5)
        row[f"sig_{k}"]=np.percentile(diff, 2.5)>0
    rows.append(row)

out=pd.DataFrame(rows).set_index("model")
out.to_csv(os.path.join(OUT, "ece_diff.csv"))
for k in MAPS:
    n_sig=int(out[f"sig_{k}"].sum())
    worst=out[f"d_{k}"].idxmin()
    print(f"{k}: 人类显著高于 {n_sig}/{len(out)} 个模型;最小差值组={worst} "
          f"Δ={out.loc[worst, f'd_{k}']:.4f} CI[{out.loc[worst, f'd_{k}_lo']:.4f}, {out.loc[worst, f'd_{k}_hi']:.4f}]")
