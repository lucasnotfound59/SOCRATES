# -*- coding: utf-8 -*-
"""对单个组做 M-ratio 自助CI,追加写 results/analysis/boot_mratio.csv。
用法:python3 boot_mratio.py <model> [nboot]  —— 分组跑以避开单次调用超时。"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from metadpy.mle import metad
from analysis_common import load_clean, OUT

m = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 250
rng = np.random.default_rng(abs(hash(m)) % (2**32))
d = load_clean(); g = d[d.model == m].reset_index(drop=True)
# 聚类自助:人类按被试聚类;模型按题目聚类(每题 4 个采样相关,trial 级重采样会低估不确定性)
clust_col = "participant_id" if m == "human" else "item_id"
clust_idx = [g.index[g[clust_col] == c].values for c in g[clust_col].unique()]

def draw():
    return g.loc[np.concatenate([clust_idx[i] for i in
                 rng.integers(0, len(clust_idx), len(clust_idx))])]

def fit(gg):
    df = pd.DataFrame({"Stimuli": (gg.gold == "True").astype(int),
                       "Accuracy": gg.correct.astype(int),
                       "Confidence": gg.conf.astype(int)})
    try:
        r = metad(data=df, nRatings=5, stimuli="Stimuli", accuracy="Accuracy",
                  confidence="Confidence"); return float(r["m_ratio"].iloc[0])
    except Exception:
        return np.nan

vals = np.array([x for x in (fit(draw()) for _ in range(NB)) if np.isfinite(x)])
lo, hi = np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)
point = fit(g)
outp = os.path.join(OUT, "boot_mratio.csv")
row = pd.DataFrame([dict(model=m, m_ratio=point, m_lo=lo, m_hi=hi, nboot=len(vals))])
if os.path.exists(outp):
    prev = pd.read_csv(outp); prev = prev[prev.model != m]; row = pd.concat([prev, row])
row.to_csv(outp, index=False)
print(f"{m}: M-ratio={point:.3f}  95%CI[{lo:.3f}, {hi:.3f}]  (n={len(vals)})")
