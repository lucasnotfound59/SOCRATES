#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py — 汇总所有 results_*.csv 为分析就绪的主数据集。
- 给每行打 status:ok / fail_truncated(用了过多token没答上来)/ fail_ratelimit(限流,可重跑)
  / fail_access(无权限,需开通)/ fail_error(其它) / fail_parse。
- 去重(同 model×语言×题×采样,ok 优先)。
- 标注 vendor / thinking / condition(think|nothink|na)/ base_model / tier。
- 只保留英文(主分析语言);输出 master_long.csv(每次调用一行)+ master_summary.csv(每模型 QA)。
分析时只用 status=='ok' 的行;fail_* 作为稳健性/覆盖率单独报告。
"""
import pandas as pd, glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
# 被取代的旧文件不纳入(数据已并入 results_local.csv)
SKIP = {"results.csv", "results_run1.csv", "results_gemma4-e4b.csv", "results_gemma4-26B-A4B.csv",
        "master_long.csv", "master_summary.csv"}

def classify(r):
    if str(r.get("parse_ok")).strip().lower() == "true":
        return "ok"                                   # 答上来了(即便 finish_reason=length 也算答上)
    raw = str(r.get("raw_response", "")); fr = str(r.get("finish_reason", "")).strip()
    err = str(r.get("error", "")).strip()
    if fr == "length" or raw.startswith("[EMPTY") or raw.startswith("[TRUNCATED"):
        return "fail_truncated"                       # 用了过多token / 被截断,没答上来
    low = err.lower()
    if err:
        if "429" in err or "rate limit" in low or "timeout" in low or "timed out" in low:
            return "fail_ratelimit"                   # 瞬时,重跑可补
        if "403" in err or "401" in err or "access denied" in low or "invalid_api_key" in low:
            return "fail_access"                      # 权限/密钥,需修
        return "fail_error"
    return "fail_parse"                               # 有内容但解析不出

def vendor(m):
    if m == "human": return "human"
    if m.startswith("local-gemma") or m.startswith("gemma"): return "google(local)"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"): return "openai"
    if m.startswith("deepseek"): return "deepseek"
    if m.startswith("qwen"): return "alibaba"
    return "other"

def thinking_of(m):
    if m == "human": return False
    if m.endswith("-nothink"): return False
    if m.startswith(("qwen2.5", "qwen-turbo")): return False          # 无思考模式
    if m.startswith(("gpt-4o", "gpt-4.1")): return False              # 非推理老模型
    return True                                                       # gemma-4 / gpt-5系 / deepseek-v4 / qwen3.7 默认思考

def condition(m):
    base = m.replace("-nothink", "")
    # 只有成对跑了 think/nothink 的模型才算受控条件
    paired = {"qwen3.7-max", "qwen3.7-plus", "deepseek-v4-pro", "deepseek-v4-flash"}
    if base in paired:
        return "nothink" if m.endswith("-nothink") else "think"
    return "na"

def base_model(m):
    return m.replace("-nothink", "")

TIER = {
    "local-gemma-e2b":"weak","qwen2.5-0.5b":"weak","qwen2.5-1.5b":"weak","qwen2.5-3b":"weak",
    "local-gemma-e4b":"mid","qwen2.5-7b":"mid","gpt-4o-mini":"mid","gpt-4.1-mini":"mid",
    "qwen-turbo":"mid","gpt-5.4-mini":"mid","deepseek-v4-flash":"mid","deepseek-v4-flash-nothink":"mid",
    "local-gemma-26b-a4b-qat":"strong","gpt-4o":"strong","gpt-4.1":"strong",
    "gpt-5.5":"frontier","deepseek-v4-pro":"frontier","deepseek-v4-pro-nothink":"frontier",
    "qwen3.7-max":"frontier","qwen3.7-max-nothink":"frontier","qwen3.7-plus":"frontier","qwen3.7-plus-nothink":"frontier",
}

frames = []
for f in sorted(glob.glob(os.path.join(RES, "*.csv"))):
    if os.path.basename(f) in SKIP: continue
    df = pd.read_csv(f)
    if "model" not in df.columns: continue
    df["__file"] = os.path.basename(f)
    frames.append(df)
all_ = pd.concat(frames, ignore_index=True)

all_["status"] = all_.apply(classify, axis=1)
all_["vendor"] = all_.model.map(vendor)
all_["thinking"] = all_.model.map(thinking_of)
all_["condition"] = all_.model.map(condition)
all_["base_model"] = all_.model.map(base_model)
all_["tier"] = all_.model.map(lambda m: "human" if m == "human" else TIER.get(m, "unspecified"))

# participant_id:人类每人不同,模型为空。去重键必须含它,否则同一题的多个被试会被并成一行。
if "participant_id" not in all_.columns:
    all_["participant_id"] = ""
all_["participant_id"] = all_["participant_id"].fillna("")

# 去重:同 (model,participant_id,language,item_id,sample_idx),ok 优先,其次保留最后一条
all_["__ok"] = (all_.status == "ok").astype(int)
all_ = (all_.sort_values(["__ok"])            # ok(1) 排后
             .drop_duplicates(["model", "participant_id", "language", "item_id", "sample_idx"], keep="last")
             .drop(columns="__ok"))

# 主分析:英文模型 + 中文人类(人类只答了中文卷;语言差异在分析中如实标注)
master = all_[(all_.language == "en") | (all_.model == "human")].copy()
master.to_csv(os.path.join(RES, "master_long.csv"), index=False)

# 每模型 QA 汇总
def summ(g):
    n=len(g); ok=(g.status=="ok").sum()
    return pd.Series({
        "n":n,"ok":ok,"ok_rate":round(ok/n,3),
        "fail_truncated":(g.status=="fail_truncated").sum(),
        "fail_ratelimit":(g.status=="fail_ratelimit").sum(),
        "fail_access":(g.status=="fail_access").sum(),
        "fail_other":(g.status.isin(["fail_error","fail_parse"])).sum(),
        "vendor":g.vendor.iloc[0],"thinking":g.thinking.iloc[0],
        "condition":g.condition.iloc[0],"tier":g.tier.iloc[0],
    })
s = master.groupby("model").apply(summ).reset_index()
s.to_csv(os.path.join(RES, "master_summary.csv"), index=False)

pd.set_option("display.width",220,"display.max_rows",100,"display.max_columns",20)
print("master_long.csv 行数(英文):", len(master))
print("\n每模型 QA 汇总:")
print(s.sort_values(["tier","model"]).to_string(index=False))
zh=(all_.language=="zh").sum()
print(f"\n(另有 {zh} 条中文数据未纳入主分析文件)")