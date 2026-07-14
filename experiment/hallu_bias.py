# -*- coding: utf-8 -*-
"""幻觉题响应偏向分析:把"边界觉察能力"与"对陌生实体的响应偏向"分开。

信号检测框架(仅幻觉题子集):
- signal=虚构条目(gold=False);response="False"=判为虚构。
- hit=虚构题答 False;false alarm=真实冷僻题答 False。
- d′=z(H)−z(FA):区分虚构与真实冷僻的敏感度(真正的边界觉察)。
- c=−0.5·(z(H)+z(FA)):响应偏向。c<0=倾向对陌生实体说"不存在"(False 偏向);
  c>0=倾向说"存在/True"(轻信偏向)。
极端比率用 log-linear 校正(加 0.5,分母加 1)。
输出:results/analysis/hallucination_sdt.csv
"""
import numpy as np, pandas as pd, os
from scipy.stats import norm
from analysis_common import load_clean, OUT

d=load_clean()
h=d[d.type=="幻觉"]

rows=[]
for m, g in h.groupby("model"):
    fic=g[g.gold=="False"]; real=g[g.gold=="True"]
    if not len(fic) or not len(real):
        continue
    # log-linear 校正,避免 0/1 比率下 z 无穷
    hit=(fic.parsed_answer=="False").sum(); n_s=len(fic)
    fa=(real.parsed_answer=="False").sum(); n_n=len(real)
    H=(hit+0.5)/(n_s+1); FA=(fa+0.5)/(n_n+1)
    zH=norm.ppf(H); zFA=norm.ppf(FA)
    rows.append(dict(model=m, n_fic=n_s, n_real=n_n,
                     fic_acc=fic.correct.mean(), real_acc=real.correct.mean(),
                     hit_rate=H, fa_rate=FA,
                     dprime=zH-zFA, criterion=-0.5*(zH+zFA)))

out=pd.DataFrame(rows).set_index("model").sort_values("dprime", ascending=False)
out.to_csv(os.path.join(OUT, "hallucination_sdt.csv"))
print(out.round(3).to_string())
