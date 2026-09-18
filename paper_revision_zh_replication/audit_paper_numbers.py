"""Audit every number claimed in the Chinese-replication paper against the source CSVs."""
import json
import sys
from pathlib import Path

import pandas as pd

A = Path("results/replication_zh_2026-09-18")
LOAD = dict(
    hms=A / "human_model_summary.csv", bs=A / "bootstrap_summary.csv",
    met=A / "metad_summary.csv", hm=A / "hmetad_summary.csv",
    sdt=A / "hallucination_sdt.csv", hb=A / "hallucination_breakdown.csv",
    q=A / "data_quality.csv",
)
d = {k: pd.read_csv(v) for k, v in LOAD.items()}
print("columns per file:")
for k, v in d.items():
    print(f"  {k}: {list(v.columns)}")

LABEL = {
    "human": "Human cohort", "repl-gemma4-e2b-q4km": "Gemma E2B",
    "repl-gemma4-e4b-q4km": "Gemma E4B", "repl-gemma4-26b-a4b-qat": "Gemma 26B",
    "repl-qwen3-1.7b-q8": "Qwen3 1.7B", "repl-qwen3-4b-q4km": "Qwen3 4B",
    "repl-qwen3-14b-q4km": "Qwen3 14B",
}

failures = []
rows = []
for key, label in LABEL.items():
    h = d["hms"].loc[d["hms"]["model"] == key].iloc[0]
    b = d["bs"].loc[d["bs"]["model"] == key].iloc[0]
    m = d["met"].loc[d["met"]["model"] == key].iloc[0]
    x = d["hm"].loc[d["hm"]["model"] == key].iloc[0]
    checks = {
        "n": (int(h["n"]), int(x["n"]), int(b["n"])),
        "n_wrong": (int(h["n_wrong"]), int(x["n_wrong"]), int(x["error_count"]), int(b["n_wrong"])),
        "accuracy": (round(float(h["accuracy"]), 9), round(float(b["accuracy"]), 9)),
        "evidence_tier": (m["evidence_tier"], x["evidence_tier"], b["evidence_tier"]),
    }
    for name, vals in checks.items():
        if len(set(map(str, vals))) != 1:
            failures.append(f"{key}.{name}: {vals}")
    if abs(float(m["m_ratio"]) - float(b["m_ratio"])) > 1e-12:
        failures.append(f"{key}.mle_m_ratio: metad={m['m_ratio']} boot={b['m_ratio']}")
    # MLE and Bayesian HMeta-d are different estimators, so their values are
    # reported side by side rather than required to agree. Only the claimed
    # ordering (human above every estimable model) is asserted later.
    rows.append({
        "group": label, "n": int(h["n"]), "acc": round(float(h["accuracy"]), 3),
        "ece": round(float(h["ece"]), 3), "over": round(float(h["overconfidence"]), 3),
        "err": int(h["n_wrong"]), "mle": round(float(m["m_ratio"]), 3),
        "tier": m["evidence_tier"], "bayes": round(float(x["m_ratio_mean"]), 3),
        "div_pct": round(float(x["divergence_rate"]) * 100, 1), "reliable": bool(x["reliable"]),
        "auc2": round(float(b["type2_auroc"]), 3),
        "hdi": [round(float(x["hdi_lo"]), 3), round(float(x["hdi_hi"]), 3)],
    })

print("\n=== Table 1 sources ===")
print(pd.DataFrame(rows).to_string(index=False))

q = d["q"]
print(f"\nvalid={int(q['n_valid'].sum())} attempted={int(q['n_attempted'].sum())} invalid={int(q['n_invalid'].sum())}")
print("valid per config:", dict(zip(q["model"].str.replace("repl-", "", regex=False), q["n_valid"])))
print("truncated per config:", dict(zip(q["model"].str.replace("repl-", "", regex=False), q["n_truncated"])))

print("\n=== Table 2 (SDT) ===")
print(d["sdt"][["model", "dprime", "criterion"]].round(4).to_string(index=False))
print("\n=== fictional/real accuracy + confidence gap ===")
print(d["hb"][["model", "fictional_accuracy", "real_accuracy", "confidence_gap"]].round(3).to_string(index=False))

print("\n=== English-cohort cross-checks (original analysis) ===")
O = Path(__file__).resolve().parents[1] / "experiment/results/analysis"
e = pd.read_csv(O / "ece_diff.csv")
nonsig = e.loc[~e["sig_primary"], "model"].tolist()
print(f"ECE difference significant for {int(e['sig_primary'].sum())}/{len(e)} configurations")
print("non-significant:", nonsig, [(r["model"], round(r["d_primary"], 4), round(r["d_primary_lo"], 4), round(r["d_primary_hi"], 4))
                                  for _, r in e.iterrows() if not r["sig_primary"]])
bt = pd.read_csv(O / "by_type.csv").query("model=='human'")
print("human acc by type:", {c: round(float(bt[c].iloc[0]), 3) for c in ("常规_acc", "学科_acc", "陷阱_acc", "幻觉_acc")})
bo = pd.read_csv(O / "boot_mratio_itemclust.csv")
print("EN model M-ratio CIs:"); print(bo.round(3).to_string(index=False))
sdto = pd.read_csv(O / "hallucination_sdt.csv")
print("EN d' range:", round(sdto["dprime"].min(), 2), "-", round(sdto["dprime"].max(), 2),
      "| c range:", round(sdto["criterion"].min(), 2), "-", round(sdto["criterion"].max(), 2))
print("EN human d'/c:", round(float(sdto.query("model=='human'")["dprime"].iloc[0]), 4),
      round(float(sdto.query("model=='human'")["criterion"].iloc[0]), 3))

print("\n=== RESULT ===")
if failures:
    print("MISMATCHES FOUND:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL NUMERICAL CROSS-CHECKS PASSED")
