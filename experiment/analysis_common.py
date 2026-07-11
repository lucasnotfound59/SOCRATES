# -*- coding: utf-8 -*-
"""共享:加载 master_long,清洗,统一分组标注。供 analyze.py / metad.py / stats.py 使用。"""
import os, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results")
OUT  = os.path.join(RES, "analysis"); os.makedirs(OUT, exist_ok=True)
FIG  = os.path.join(RES, "figures");  os.makedirs(FIG, exist_ok=True)

# 5 档主观置信度 → 概率(判断题瞎猜下限 50%)
CMAP = {1: .50, 2: .625, 3: .75, 4: .875, 5: 1.0}

# 已知污染:本地 qwen3 无思考版截断率高(LM Studio 未真正关闭思考)→ 主分析剔除
CONTAMINATED = {"local-qwen3-1.7b-nothink", "local-qwen3-4b-nothink", "local-qwen3-14b-nothink"}
MIN_OK = 500   # 有效行低于此的组视为死数据丢弃

# 能力层级排序(用于出图/排版)
TIER_ORDER = {"human": 0, "weak": 1, "unspecified": 2, "mid": 3, "strong": 4, "frontier": 5}

def load_clean(drop_contaminated=True, drop_nothink_local=True):
    d = pd.read_csv(os.path.join(RES, "master_long.csv"), low_memory=False)
    d = d[d.status == "ok"].copy()
    # 统一类型
    d["correct"] = d.correct.astype(str).str.strip().isin(["True", "true", "1", "1.0"])
    d["gold"]    = d.gold.astype(str).str.strip()
    d["parsed_answer"] = d.parsed_answer.astype(str).str.strip()
    d["conf"]    = pd.to_numeric(d.confidence, errors="coerce")
    d = d[d.conf.isin([1, 2, 3, 4, 5])]
    d = d[d.gold.isin(["True", "False"])]
    d = d[d.parsed_answer.isin(["True", "False"])]
    d["conf"]    = d.conf.astype(int)
    d["stated"]  = d.conf.map(CMAP)
    # 丢死数据 / 污染
    if drop_contaminated:
        d = d[~d.model.isin(CONTAMINATED)]
    if drop_nothink_local:
        d = d[~d.model.str.startswith("local-qwen3") | ~d.model.str.endswith("-nothink")]
    counts = d.groupby("model").size()
    keep = counts[counts >= MIN_OK].index
    d = d[d.model.isin(keep)].copy()
    # 干净的层级排序键
    d["tier_rank"] = d.tier.map(TIER_ORDER).fillna(2).astype(int)
    return d

def ece(sub):
    """Expected Calibration Error:声称置信度(映射)与实际正确率的加权偏离。"""
    tot = len(sub); e = 0.0
    for c, g in sub.groupby("conf"):
        e += abs(g.correct.mean() - CMAP[c]) * len(g)
    return e / tot if tot else np.nan

def group_order(d):
    """按 (tier_rank, acc) 排序的组名列表。"""
    agg = d.groupby("model").agg(tier_rank=("tier_rank", "first"),
                                 acc=("correct", "mean"))
    return agg.sort_values(["tier_rank", "acc"]).index.tolist()

VENDOR_COLOR = {"human": "#d62728", "openai": "#2ca02c", "deepseek": "#1f77b4",
                "alibaba": "#9467bd", "google(local)": "#ff7f0e", "other": "#8c564b"}
def vendor_color(d, m):
    v = d[d.model == m].vendor.iloc[0]
    return VENDOR_COLOR.get(v, "#7f7f7f")
