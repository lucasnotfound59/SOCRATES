# -*- coding: utf-8 -*-
"""单个组的贝叶斯 meta-d′(HMeta-d 的单被试情形),给出后验均值 + 95% HDI。
低/零错误组会得到很宽的区间 —— 这正是"诚实的不确定性"。追加写 hmetad_summary.csv。
用法:python3 hmetad_run.py <model> [num_samples]"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, arviz as az
from analysis_common import load_clean, OUT
from metadpy.bayesian import hmetad

m = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
d = load_clean(); g = d[d.model == m]
n_wrong = int((~g.correct).sum())
df = pd.DataFrame({"Stimuli": (g.gold == "True").astype(int),
                   "Accuracy": g.correct.astype(int),
                   "Confidence": g.conf.astype(int)})

def summarize(idata):
    post = idata.posterior
    d1 = post["d1"].values.flatten() if "d1" in post else post["d"].values.flatten()
    metad = post["meta_d"].values.flatten()
    mr = metad / d1
    lo, hi = az.hdi(mr, hdi_prob=0.95)
    # R-hat 收敛诊断 + 发散数(少错误组会大量发散 → 先验主导,不可信)
    try:
        rhat = float(az.rhat(idata)["meta_d"].values)
    except Exception:
        rhat = np.nan
    try:
        ndiv = int(idata.sample_stats["diverging"].values.sum())
    except Exception:
        ndiv = -1
    return np.mean(mr), np.median(mr), lo, hi, np.std(mr), rhat, ndiv

try:
    res = hmetad(data=df, nRatings=5, stimuli="Stimuli", accuracy="Accuracy",
                 confidence="Confidence", num_samples=NS, num_chains=2, output="model")
    idata = res[1] if isinstance(res, (tuple, list)) else res
    mean, med, lo, hi, sd, rhat, ndiv = summarize(idata)
    # 可信性:错误≥10 且发散占比低 → 可信;否则先验主导
    reliable = (n_wrong >= 10) and (ndiv < 40) and (rhat < 1.05)
    row = dict(model=m, n_wrong=n_wrong, mr_mean=round(mean, 3), mr_median=round(med, 3),
               hdi_lo=round(lo, 3), hdi_hi=round(hi, 3), post_sd=round(sd, 3),
               rhat=round(rhat, 3), n_div=ndiv, width=round(hi - lo, 3), reliable=reliable)
except Exception as e:
    row = dict(model=m, n_wrong=n_wrong, mr_mean=np.nan, mr_median=np.nan,
               hdi_lo=np.nan, hdi_hi=np.nan, post_sd=np.nan, rhat=np.nan, width=np.nan,
               error=str(e)[:100])

outp = os.path.join(OUT, "hmetad_summary.csv")
r = pd.DataFrame([row])
if os.path.exists(outp):
    prev = pd.read_csv(outp); prev = prev[prev.model != m]
    r = pd.concat([prev, r], ignore_index=True)
r.to_csv(outp, index=False)
print(f"{m}: 错误{n_wrong}  M-ratio={row.get('mr_mean')}  "
      f"95%HDI[{row.get('hdi_lo')}, {row.get('hdi_hi')}]  发散={row.get('n_div')}  "
      f"Rhat={row.get('rhat')}  可信={row.get('reliable')}")
