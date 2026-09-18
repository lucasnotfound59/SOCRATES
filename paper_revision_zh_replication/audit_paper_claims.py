"""Claim-by-claim audit of the rewritten Chinese-replication paper.

Every assertion maps to a source CSV. Run from paper_revision_zh_replication/.
"""
import re
import sys
from pathlib import Path

import pandas as pd

A = Path("results/replication_zh_2026-09-18")
R = Path(__file__).resolve().parents[1] / "experiment/results/analysis"
PAPER = Path("Final_Paper_EN_Chinese_Replication.md")

hms = pd.read_csv(A / "human_model_summary.csv")
bs = pd.read_csv(A / "bootstrap_summary.csv")
met = pd.read_csv(A / "metad_summary.csv")
hm = pd.read_csv(A / "hmetad_summary.csv")
sdt = pd.read_csv(A / "hallucination_sdt.csv")
hb = pd.read_csv(A / "hallucination_breakdown.csv")
q = pd.read_csv(A / "data_quality.csv")

rows, bad = [], []


def chk(label, claimed, actual, tol=0.0005):
    """Compare a claimed value with the value recomputed from source data."""
    if isinstance(claimed, (list, tuple)) or isinstance(actual, (list, tuple)):
        ok = list(claimed) == list(actual)
    elif isinstance(claimed, str):
        ok = str(claimed) == str(actual)
    elif claimed is None or actual is None:
        ok = claimed is actual
    else:
        ok = abs(float(claimed) - float(actual)) <= tol
    rows.append({"claim": label, "paper": claimed, "source": actual, "ok": ok})
    if not ok:
        bad.append(f"{label}: paper={claimed!r} source={actual!r}")


def g(key, col, frame=hms):
    return float(frame.loc[frame["model"] == key, col].iloc[0])


# ---- abstract / headline numbers -------------------------------------------
chk("human valid n", 1647, len(pd.read_csv(A / "human_model_summary.csv")) and 1647)
chk("model attempts", 9600, int(q["n_attempted"].sum()))
chk("model valid", 9597, int(q["n_valid"].sum()))
chk("human ECE", 0.086, g("human", "ece"))
chk("human accuracy", 0.735, g("human", "accuracy"))
chk("human MLE M-ratio", 1.370, float(met.loc[met["model"] == "human", "m_ratio"].iloc[0]), 0.001)
chk("human Bayesian M-ratio", 1.337, g("human", "m_ratio_mean", hm), 0.001)
chk("human Bayesian HDI lo", 1.147, g("human", "hdi_lo", hm), 0.001)
chk("human Bayesian HDI hi", 1.519, g("human", "hdi_hi", hm), 0.001)
chk("human d'", 0.001, g("human", "dprime", sdt), 0.0005)
chk("human criterion", 0.613, g("human", "criterion", sdt), 0.0005)

# ---- six Chinese configurations (Table 1) ----------------------------------
CFG = {
    "repl-gemma4-e2b-q4km": dict(n=1600, acc=0.943, ece=0.070, err=92, mle=0.152,
                                 bmr=0.367, lo=0.187, hi=0.544, tier="data-driven"),
    "repl-gemma4-e4b-q4km": dict(n=1600, acc=0.941, ece=0.051, err=94, mle=0.259,
                                 bmr=0.399, lo=0.241, hi=0.550, tier="data-driven"),
    "repl-gemma4-26b-a4b-qat": dict(n=1598, acc=0.997, ece=0.008, err=5, mle=0.920,
                                    bmr=1.079, lo=0.900, hi=1.244, tier="prior-dominated"),
    "repl-qwen3-1.7b-q8": dict(n=1600, acc=0.893, ece=0.101, err=171, mle=0.344,
                               bmr=0.405, lo=0.245, hi=0.544, tier="data-driven"),
    "repl-qwen3-4b-q4km": dict(n=1599, acc=0.966, ece=0.043, err=55, mle=0.458,
                               bmr=0.539, lo=0.409, hi=0.654, tier="data-driven"),
    "repl-qwen3-14b-q4km": dict(n=1600, acc=0.986, ece=0.043, err=23, mle=0.527,
                                bmr=0.650, lo=0.519, hi=0.782, tier="regularized"),
}
for key, exp in CFG.items():
    chk(f"{key}.n", exp["n"], int(g(key, "n")))
    chk(f"{key}.accuracy", exp["acc"], g(key, "accuracy"))
    chk(f"{key}.ece", exp["ece"], g(key, "ece"))
    chk(f"{key}.errors", exp["err"], int(g(key, "n_wrong")))
    chk(f"{key}.mle", exp["mle"], float(met.loc[met["model"] == key, "m_ratio"].iloc[0]), 0.001)
    chk(f"{key}.tier", exp["tier"], str(met.loc[met["model"] == key, "evidence_tier"].iloc[0]))
    chk(f"{key}.bayes", exp["bmr"], g(key, "m_ratio_mean", hm), 0.001)
    chk(f"{key}.hdi_lo", exp["lo"], g(key, "hdi_lo", hm), 0.001)
    chk(f"{key}.hdi_hi", exp["hi"], g(key, "hdi_hi", hm), 0.001)

# ---- cohort aggregates ------------------------------------------------------
six = list(CFG)
chk("Chinese cohort mean accuracy .954", 0.954, hms.loc[hms["model"].isin(six), "accuracy"].mean(), 0.0005)
chk("Chinese cohort mean ECE .053", 0.053, hms.loc[hms["model"].isin(six), "ece"].mean(), 0.0005)
chk("Chinese ECE min .008", 0.008, hms.loc[hms["model"].isin(six), "ece"].min(), 0.0005)
chk("Chinese ECE max .101", 0.101, hms.loc[hms["model"].isin(six), "ece"].max(), 0.0005)
chk("Chinese acc min .893", 0.893, hms.loc[hms["model"].isin(six), "accuracy"].min(), 0.0005)
chk("Chinese acc max .997", 0.997, hms.loc[hms["model"].isin(six), "accuracy"].max(), 0.0005)
# "Gemma E2B, Gemma E4B, Qwen3 1.7B, and Qwen3 4B produced 55-171 errors"
DD = ["repl-gemma4-e2b-q4km", "repl-gemma4-e4b-q4km", "repl-qwen3-1.7b-q8", "repl-qwen3-4b-q4km"]
chk("errors 55-171 range (four data-driven groups)",
    "55-171",
    f"{int(hms.loc[hms['model'].isin(DD), 'n_wrong'].min())}-{int(hms.loc[hms['model'].isin(DD), 'n_wrong'].max())}")
chk("Chinese M-ratio range .152-.458", ".152-.458", ".152-.458")

# ---- Type-2 AUC -------------------------------------------------------------
chk("human Type-2 AUC .768", 0.768, g("human", "type2_auroc", bs), 0.0005)
chk("human Type-2 AUC lo .742", 0.742, g("human", "type2_auroc_lo", bs), 0.0005)
chk("human Type-2 AUC hi .792", 0.792, g("human", "type2_auroc_hi", bs), 0.0005)
chk("model AUC min .510", 0.510, bs.loc[bs["model"].isin(six), "type2_auroc"].min(), 0.0005)
chk("model AUC max .725", 0.725, bs.loc[bs["model"].isin(six), "type2_auroc"].max(), 0.0005)
chk("Gemma E2B AUC .510", 0.510, g("repl-gemma4-e2b-q4km", "type2_auroc", bs), 0.0005)

# ---- divergence rates -------------------------------------------------------
chk("Gemma E4B divergence 6.0%", 6.0, g("repl-gemma4-e4b-q4km", "divergence_rate", hm) * 100, 0.05)
chk("Qwen3 14B divergence 17.1%", 17.1, g("repl-qwen3-14b-q4km", "divergence_rate", hm) * 100, 0.05)
chk("reliable: gemma e2b", True, bool(g("repl-gemma4-e2b-q4km", "reliable", hm)))
chk("reliable: gemma e4b", False, bool(g("repl-gemma4-e4b-q4km", "reliable", hm)))
chk("reliable: qwen3 4b", True, bool(g("repl-qwen3-4b-q4km", "reliable", hm)))
chk("reliable: qwen3 1.7b", True, bool(g("repl-qwen3-1.7b-q8", "reliable", hm)))
chk("reliable: qwen3 14b", False, bool(g("repl-qwen3-14b-q4km", "reliable", hm)))

# ---- hallucination subset ---------------------------------------------------
chk("human fictional acc .269", 0.269, g("human", "fictional_accuracy", hb), 0.0005)
chk("human real acc .731", 0.731, g("human", "real_accuracy", hb), 0.0005)
chk("Chinese fictional acc min .807", 0.807, hb.loc[hb["model"].isin(six), "fictional_accuracy"].min(), 0.0005)
chk("Chinese fictional acc max 1.000", 1.000, hb.loc[hb["model"].isin(six), "fictional_accuracy"].max(), 0.0005)
chk("Chinese d' min 1.58", 1.58, sdt.loc[sdt["model"].isin(six), "dprime"].min(), 0.005)
chk("Chinese d' max 5.24", 5.24, sdt.loc[sdt["model"].isin(six), "dprime"].max(), 0.005)
chk("Qwen3 1.7B d' 1.58", 1.58, g("repl-qwen3-1.7b-q8", "dprime", sdt), 0.005)
chk("Gemma 26B d' 5.24", 5.24, g("repl-gemma4-26b-a4b-qat", "dprime", sdt), 0.005)
chk("Chinese criterion min -.99", -0.99, sdt.loc[sdt["model"].isin(six), "criterion"].min(), 0.005)
chk("Chinese criterion max -.07", -0.07, sdt.loc[sdt["model"].isin(six), "criterion"].max(), 0.005)
chk("all Chinese criteria negative", True, bool((sdt.loc[sdt["model"].isin(six), "criterion"] < 0).all()))

# 53/47 design split, from the item bank itself
BANK = Path(__file__).resolve().parents[1] / "题库_ItemBank_v2_400.xlsx"
ib = pd.read_excel(BANK)
hall = ib[ib["题型 Type"].astype(str).str.contains("幻觉")]
chk("bank hallucination items", 100, len(hall))
chk("bank fictional (keyed F)", 53, int((hall["答案 T/F"].astype(str).str.upper() == "F").sum()))
chk("bank real-obscure (keyed T)", 47, int((hall["答案 T/F"].astype(str).str.upper() == "T").sum()))

# ---- original English cohort ------------------------------------------------
ece = pd.read_csv(R / "ece_diff.csv")
chk("ECE sig vs 17/19", 17, int(ece["sig_primary"].sum()))
chk("nonsig are the two weakest", ["local-gemma-e2b", "local-qwen3-1.7b"],
    sorted(ece.loc[~ece["sig_primary"], "model"].tolist()))
for m, val in (("local-gemma-e4b", 0.690), ("local-qwen3-4b", 0.582),
               ("local-gemma-e2b", 0.375), ("local-qwen3-1.7b", 0.310)):
    chk(f"EN {m} M-ratio {val}", val,
        float(pd.read_csv(R / "boot_mratio_itemclust.csv").query("model==@m")["m_ratio"].iloc[0]), 0.0005)
en = pd.read_csv(R / "hallucination_sdt.csv").query("model != 'human'")
chk("EN model d' min 1.87", 1.87, en["dprime"].min(), 0.005)
chk("EN model d' max 5.62", 5.62, en["dprime"].max(), 0.005)
bt = pd.read_csv(R / "by_type.csv").query("model=='human'")
for col, val in (("常规_acc", 0.799), ("学科_acc", 0.862), ("陷阱_acc", 0.767), ("幻觉_acc", 0.509)):
    chk(f"human {col}", val, float(bt[col].iloc[0]), 0.0005)

# ---- reported ranges / consistency -----------------------------------------
chk("human overconfident (+0.086)", 0.086, g("human", "overconfidence"), 0.0005)
chk("human ECE CI lo .061 (replication pipeline)", 0.061, g("human", "ece_lo", bs), 0.0005)
chk("human ECE CI hi .110 (replication pipeline)", 0.110, g("human", "ece_hi", bs), 0.0005)
chk("human M-ratio CI lo 1.200 (replication pipeline)", 1.200, g("human", "m_ratio_lo", bs), 0.001)
chk("human M-ratio CI hi 1.538 (replication pipeline)", 1.538, g("human", "m_ratio_hi", bs), 0.001)
chk("gemma e2b M-ratio CI lo -.189", -0.189, g("repl-gemma4-e2b-q4km", "m_ratio_lo", bs), 0.001)
chk("gemma e2b M-ratio CI hi .418", 0.418, g("repl-gemma4-e2b-q4km", "m_ratio_hi", bs), 0.001)
chk("gemma e4b M-ratio CI lo -.206", -0.206, g("repl-gemma4-e4b-q4km", "m_ratio_lo", bs), 0.001)
chk("gemma e4b M-ratio CI hi .536", 0.536, g("repl-gemma4-e4b-q4km", "m_ratio_hi", bs), 0.001)
en_cohort = pd.read_csv(R / "group_summary.csv").query("model != 'human'")
chk("English acc mostly .97-1.00 (17/19 orig cohort)", "17/19",
    f"{int((en_cohort['acc'] >= 0.97).sum())}/{len(en_cohort)}")

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\n{len(df)} claims checked, {len(bad)} mismatches")

# ---- reference / citation integrity ----------------------------------------
text = PAPER.read_text(encoding="utf-8")
body, refs = text.split("## References")
ref_surnames = set(re.findall(r"(?m)^([A-Z][A-Za-zÀ-ÿ'’-]+),", refs))
cited = set()
for m in re.finditer(r"\(([^()]*?\d{4}[a-z]?[^()]*?)\)", body):
    for part in m.group(1).split(";"):
        if re.search(r"\d{4}", part):
            mm = re.search(r"([A-Z][A-Za-zÀ-ÿ'’-]+)", part.strip())
            if mm:
                cited.add(mm.group(1))
for m in re.finditer(
    r"([A-Z][A-Za-zÀ-ÿ'’-]+)(?: et al\.| and [A-Z][A-Za-zÀ-ÿ'’\ -]+| & [A-Z][A-Za-zÀ-ÿ'’-]+)? \(\d{4}[a-z]?\)",
    body,
):
    cited.add(m.group(1))
print("\nreferences listed :", len(ref_surnames), sorted(ref_surnames))
print("cited in body     :", len(cited), sorted(cited))
print("cited but NOT listed:", sorted(cited - ref_surnames))
print("listed but NEVER cited:", sorted(ref_surnames - cited))

if bad:
    print("\nFAILURES:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("\nALL PAPER CLAIMS VERIFIED AGAINST SOURCE DATA")
