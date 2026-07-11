# -*- coding: utf-8 -*-
"""每组估计 d′ / meta-d′ / M-ratio(Maniscalco-Lau MLE,经 metadpy)。
天花板(错误 trial 过少)标记为不可靠。--selftest 跑理想观察者验证。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from metadpy.mle import metad
from analysis_common import load_clean, group_order, OUT

MIN_ERRORS = 30   # 错误 trial 少于此 → meta-d′ 不可靠(天花板)

def fit_group(g):
    """g: 含 gold(True/False)、correct(bool)、conf(1-5) 的一组。返回 dict。"""
    df = pd.DataFrame({
        "Stimuli":   (g.gold == "True").astype(int),
        "Accuracy":  g.correct.astype(int),
        "Confidence": g.conf.astype(int),
    })
    n_wrong = int((df.Accuracy == 0).sum())
    err = df[df.Accuracy == 0]
    estimable = (n_wrong >= MIN_ERRORS) and (err.Confidence.nunique() >= 2)
    out = dict(n=len(df), n_wrong=n_wrong, acc=df.Accuracy.mean(),
               estimable=estimable, dprime=np.nan, meta_d=np.nan,
               m_ratio=np.nan, m_diff=np.nan)
    try:
        r = metad(data=df, nRatings=5, stimuli="Stimuli",
                  accuracy="Accuracy", confidence="Confidence")
        out.update(dprime=float(r["dprime"].iloc[0]), meta_d=float(r["meta_d"].iloc[0]),
                   m_ratio=float(r["m_ratio"].iloc[0]), m_diff=float(r["m_diff"].iloc[0]))
    except Exception as e:
        out["error"] = str(e)[:80]
    return out

def selftest():
    rng = np.random.default_rng(0); N = 20000
    def sim(mn):
        s = np.r_[np.zeros(N), np.ones(N)].astype(int)
        x = np.r_[rng.normal(0, 1, N), rng.normal(1.5, 1, N)]
        acc = ((x > 0).astype(int) == s).astype(int)
        xc = np.abs(x + rng.normal(0, mn, 2 * N))
        q = np.quantile(xc, np.linspace(0, 1, 6)[1:-1])
        conf = np.digitize(xc, q) + 1
        return pd.DataFrame({"Stimuli": s, "Accuracy": acc, "Confidence": conf})
    print("== 自检:理想观察者 M-ratio 应≈1,加噪后应下降 ==")
    for name, mn in [("理想", 0.0), ("轻噪", 0.7), ("重噪", 1.6)]:
        r = metad(data=sim(mn), nRatings=5, stimuli="Stimuli",
                  accuracy="Accuracy", confidence="Confidence")
        print(f"  {name}: d′={float(r['dprime'].iloc[0]):.3f}  "
              f"meta-d′={float(r['meta_d'].iloc[0]):.3f}  M-ratio={float(r['m_ratio'].iloc[0]):.3f}")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit()

    d = load_clean()
    order = group_order(d)
    rows = []
    for m in order:
        g = d[d.model == m]
        r = fit_group(g); r["model"] = m
        r["vendor"] = g.vendor.iloc[0]; r["tier"] = g.tier.iloc[0]
        r["thinking"] = bool(g.thinking.iloc[0]); r["condition"] = g.condition.iloc[0]
        rows.append(r)
    S = pd.DataFrame(rows).set_index("model")
    cols = ["tier", "n", "n_wrong", "acc", "dprime", "meta_d", "m_ratio", "m_diff", "estimable"]
    S[cols + [c for c in S.columns if c not in cols]].to_csv(os.path.join(OUT, "metad_summary.csv"))

    pd.set_option("display.width", 200)
    print("== meta-d′ / M-ratio(estimable=False 为天花板,数字仅供参考)==")
    print(S[cols].round(3).to_string())
    est = S[S.estimable]
    print(f"\n可靠估计的组:{len(est)} / {len(S)}")
    print("其中 M-ratio<1(元认知弱于一阶):",
          ", ".join(f"{m}({v:.2f})" for m, v in est.m_ratio.items() if v < 1))
