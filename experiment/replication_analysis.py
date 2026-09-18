"""Input contracts and privacy-safe normalization for the Chinese replication."""

from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


CONFIDENCE_MAP = {1: 0.50, 2: 0.625, 3: 0.75, 4: 0.875, 5: 1.00}
FORMAL_KEY = ["model", "language", "item_id", "sample_idx"]
EXPECTED_MODELS = 6
EXPECTED_ATTEMPTS_PER_MODEL = 1600
EXPECTED_TOTAL_ATTEMPTS = 9600
RANDOM_SEED = 20260918

_BINARY = {"true", "false"}
_PUBLIC_COLUMNS = [
    "model", "language", "item_id", "sample_idx", "gold", "parsed_answer",
    "correct", "conf", "stated", "type", "subtype", "difficulty", "cluster_id",
]


def ece(group: pd.DataFrame) -> float:
    """Expected calibration error using the task's five-bin probability map."""
    if group.empty:
        return float("nan")
    conf = pd.to_numeric(group["conf"], errors="coerce")
    correct = group["correct"].astype(bool)
    total = len(group)
    error = 0.0
    for level, rows in group.loc[conf.isin(CONFIDENCE_MAP)].groupby(conf):
        error += abs(float(rows["correct"].astype(bool).mean()) - CONFIDENCE_MAP[int(level)]) * len(rows)
    return error / total


def type2_auroc(group: pd.DataFrame) -> float:
    """Probability that a correct trial has higher confidence than an error."""
    correct = group.loc[group["correct"].astype(bool), "conf"].dropna().to_numpy()
    wrong = group.loc[~group["correct"].astype(bool), "conf"].dropna().to_numpy()
    if not len(correct) or not len(wrong):
        return float("nan")
    comparisons = (correct[:, None] > wrong[None, :]).sum()
    ties = (correct[:, None] == wrong[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(correct) * len(wrong)))


def _mean(group: pd.DataFrame, column: str) -> float:
    return float(group[column].mean()) if len(group) else float("nan")


def _model_sort_key(model: object) -> tuple:
    name = str(model)
    if name == "human":
        return (0, "", -1, "")
    # Formal local replication labels encode the model version and parameter
    # size in the same identifier (e.g. qwen3-14b, gemma-e4b, and
    # gemma-26b-a4b-qat).  Only the first parameter-size token is the size;
    # quantization/active-expert suffixes must not affect ordering.
    base = re.sub(r"-nothink$", "", name)
    if base.startswith("local-gemma-"):
        family = "gemma"
    elif base.startswith("local-qwen3-"):
        family = "qwen3"
    else:
        family = re.sub(r"(?:^|[-_])(?:e)?\d+(?:\.\d+)?b", "-size", base)
    size_match = re.search(r"(?:^|[-_])(?:e)?(\d+(?:\.\d+)?)b(?:[-_]|$)", base)
    numeric = float(size_match.group(1)) if size_match else float("inf")
    return (1, family, numeric, name)


def _ordered_models(data: pd.DataFrame) -> list[str]:
    return sorted(data["model"].dropna().astype(str).unique(), key=_model_sort_key)


def summarize_groups(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "n", "accuracy", "mean_confidence", "mean_stated",
               "overconfidence", "ece", "type2_auroc", "n_wrong", "conf5_n",
               "conf5_accuracy", "conf5_wrong", "evidence_tier"]
    rows = []
    for model in _ordered_models(data):
        group = data.loc[data["model"].astype(str).eq(model)]
        wrong = ~group["correct"].astype(bool)
        conf5 = group.loc[pd.to_numeric(group["conf"], errors="coerce").eq(5)]
        accuracy = float(group["correct"].astype(bool).mean()) if len(group) else float("nan")
        n_wrong = int(wrong.sum())
        rows.append({
            "model": model, "n": len(group), "accuracy": accuracy,
            "mean_confidence": _mean(group, "conf"), "mean_stated": _mean(group, "stated"),
            "overconfidence": _mean(group, "stated") - accuracy, "ece": ece(group),
            "type2_auroc": type2_auroc(group), "n_wrong": n_wrong,
            "conf5_n": len(conf5),
            "conf5_accuracy": float(conf5["correct"].astype(bool).mean()) if len(conf5) else float("nan"),
            "conf5_wrong": int((~conf5["correct"].astype(bool)).sum()),
            "evidence_tier": evidence_tier(n_wrong),
        })
    return pd.DataFrame(rows, columns=columns)


def summarize_factor(data: pd.DataFrame, factor: str, levels: Sequence[str]) -> pd.DataFrame:
    columns = ["model", factor, "n", "accuracy", "mean_confidence", "overconfidence", "ece"]
    rows = []
    for model in _ordered_models(data):
        model_data = data.loc[data["model"].astype(str).eq(model)]
        for level in levels:
            group = model_data.loc[model_data[factor].eq(level)]
            accuracy = float(group["correct"].astype(bool).mean()) if len(group) else float("nan")
            mean_stated = _mean(group, "stated")
            rows.append({"model": model, factor: level, "n": len(group),
                         "accuracy": accuracy, "mean_confidence": _mean(group, "conf"),
                         "overconfidence": mean_stated - accuracy, "ece": ece(group)})
    return pd.DataFrame(rows, columns=columns)


def confidence_distribution(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", *[f"p_conf{level}" for level in CONFIDENCE_MAP]]
    rows = []
    for model in _ordered_models(data):
        group = data.loc[data["model"].astype(str).eq(model)]
        conf = pd.to_numeric(group["conf"], errors="coerce")
        rows.append({"model": model, **{f"p_conf{level}": float(conf.eq(level).mean()) for level in CONFIDENCE_MAP}})
    return pd.DataFrame(rows, columns=columns)


def reliability_table(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "confidence", "stated", "accuracy", "n"]
    rows = []
    for model in _ordered_models(data):
        group = data.loc[data["model"].astype(str).eq(model)]
        for confidence in sorted(pd.to_numeric(group["conf"], errors="coerce").dropna().unique()):
            bin_rows = group.loc[pd.to_numeric(group["conf"], errors="coerce").eq(confidence)]
            rows.append({"model": model, "confidence": int(confidence),
                         "stated": CONFIDENCE_MAP.get(int(confidence), np.nan),
                         "accuracy": float(bin_rows["correct"].astype(bool).mean()), "n": len(bin_rows)})
    return pd.DataFrame(rows, columns=columns)


def hallucination_breakdown(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "n_fictional", "fictional_accuracy", "fictional_confidence",
               "n_real", "real_accuracy", "real_confidence", "accuracy_gap", "confidence_gap"]
    hallucination = data.loc[data["type"].eq("幻觉")]
    rows = []
    for model in _ordered_models(hallucination):
        group = hallucination.loc[hallucination["model"].astype(str).eq(model)]
        fictional = group.loc[group["gold"].astype(str).str.lower().eq("false")]
        real = group.loc[group["gold"].astype(str).str.lower().eq("true")]
        f_acc, r_acc = _mean_bool(fictional, "correct"), _mean_bool(real, "correct")
        rows.append({"model": model, "n_fictional": len(fictional), "fictional_accuracy": f_acc,
                     "fictional_confidence": _mean(fictional, "conf"), "n_real": len(real),
                     "real_accuracy": r_acc, "real_confidence": _mean(real, "conf"),
                     "accuracy_gap": r_acc - f_acc, "confidence_gap": _mean(real, "conf") - _mean(fictional, "conf")})
    return pd.DataFrame(rows, columns=columns)


def _mean_bool(group: pd.DataFrame, column: str) -> float:
    return float(group[column].astype(bool).mean()) if len(group) else float("nan")


def hallucination_sdt(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "n_fictional", "n_real", "hit_rate", "false_alarm_rate", "dprime", "criterion"]
    hallucination = data.loc[data["type"].eq("幻觉")]
    rows = []
    for model in _ordered_models(hallucination):
        group = hallucination.loc[hallucination["model"].astype(str).eq(model)]
        fictional = group.loc[group["gold"].astype(str).str.lower().eq("false")]
        real = group.loc[group["gold"].astype(str).str.lower().eq("true")]
        if not len(fictional) or not len(real):
            continue
        hits = (fictional["parsed_answer"].astype(str).str.lower() == "false").sum()
        false_alarms = (real["parsed_answer"].astype(str).str.lower() == "false").sum()
        hit_rate = (hits + 0.5) / (len(fictional) + 1)
        false_alarm_rate = (false_alarms + 0.5) / (len(real) + 1)
        zh, zfa = norm.ppf(hit_rate), norm.ppf(false_alarm_rate)
        rows.append({"model": model, "n_fictional": len(fictional), "n_real": len(real),
                     "hit_rate": hit_rate, "false_alarm_rate": false_alarm_rate,
                     "dprime": zh - zfa, "criterion": -0.5 * (zh + zfa)})
    return pd.DataFrame(rows, columns=columns)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def evidence_tier(n_wrong: int) -> str:
    if n_wrong >= 30:
        return "data-driven"
    if n_wrong >= 10:
        return "regularized"
    return "prior-dominated"


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].astype("string").fillna("").str.strip()


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    parse_ok = _text(frame, "parse_ok").str.lower().isin({"true", "1", "1.0"})
    gold = _text(frame, "gold").str.lower().isin(_BINARY)
    answer = _text(frame, "parsed_answer").str.lower().isin(_BINARY)
    confidence = frame.get("confidence", frame.get("conf", pd.Series(index=frame.index)))
    conf = pd.to_numeric(confidence, errors="coerce")
    return parse_ok & gold & answer & conf.isin(CONFIDENCE_MAP)


def _normalise(frame: pd.DataFrame, cluster_from: str = "item_id") -> pd.DataFrame:
    out = frame.copy()
    out["model"] = _text(out, "model")
    out["language"] = _text(out, "language").str.lower()
    for col in ("item_id", "type", "subtype", "difficulty", "parsed_answer", "gold"):
        out[col] = _text(out, col)
    out["gold"] = out["gold"].str.title()
    out["parsed_answer"] = out["parsed_answer"].str.title()
    # Accuracy is an observed property of the normalized answer and gold;
    # never trust a producer-supplied ``correct`` field.
    out["correct"] = out["gold"].eq(out["parsed_answer"])
    confidence = out.get("confidence", out.get("conf", pd.Series(index=out.index)))
    out["conf"] = pd.to_numeric(confidence, errors="coerce")
    out["stated"] = out["conf"].map(CONFIDENCE_MAP)
    out["sample_idx"] = pd.to_numeric(out["sample_idx"], errors="coerce").astype("Int64")
    if cluster_from == "participant_id":
        out["cluster_id"] = _text(out, "participant_id")
    else:
        out["cluster_id"] = out["item_id"]
    return out


def validate_model_attempts(attempts: pd.DataFrame) -> None:
    missing = [c for c in FORMAL_KEY if c not in attempts.columns]
    if missing:
        raise ValueError(f"missing formal key columns: {missing}")
    if len(attempts) != EXPECTED_TOTAL_ATTEMPTS:
        raise ValueError(f"expected {EXPECTED_TOTAL_ATTEMPTS} attempts, got {len(attempts)}")
    canonical = attempts.copy()
    canonical["model"] = _text(canonical, "model")
    canonical["language"] = _text(canonical, "language").str.lower()
    canonical["item_id"] = _text(canonical, "item_id")
    canonical["sample_idx"] = pd.to_numeric(canonical["sample_idx"], errors="coerce")
    if canonical[FORMAL_KEY].isna().any().any() or canonical[FORMAL_KEY].eq("").any().any():
        raise ValueError("formal task keys must be non-empty")
    if canonical[FORMAL_KEY].duplicated().any():
        raise ValueError("formal task keys must be unique task keys")
    if canonical["language"].ne("zh").any():
        raise ValueError("model attempts must be Chinese rows")
    if canonical["model"].nunique() != EXPECTED_MODELS:
        raise ValueError(f"expected {EXPECTED_MODELS} models")
    counts = canonical.groupby("model", dropna=False).size()
    if not (counts == EXPECTED_ATTEMPTS_PER_MODEL).all():
        raise ValueError(f"each model must have {EXPECTED_ATTEMPTS_PER_MODEL} attempts")


def load_model_attempts(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(Path(path), low_memory=False)
    validate_model_attempts(raw)
    all_rows = _normalise(raw)
    valid = all_rows.loc[_valid_mask(raw)].copy()
    return all_rows, valid


def load_human_trials(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(Path(path), low_memory=False)
    selected = raw.loc[
        (_text(raw, "model").eq("human"))
        & (_text(raw, "language").str.lower().eq("zh"))
        & (_text(raw, "status").str.lower().eq("ok"))
    ].copy()
    participant = _text(selected, "participant_id")
    selected = selected.loc[participant.ne("")]
    # Human status is the parse gate; unlike model rows, some human exports do
    # not carry a separate parse_ok column.
    human_valid = (
        _text(selected, "gold").str.lower().isin(_BINARY)
        & _text(selected, "parsed_answer").str.lower().isin(_BINARY)
        & pd.to_numeric(selected.get("confidence", selected.get("conf", pd.Series(index=selected.index))), errors="coerce").isin(CONFIDENCE_MAP)
    )
    selected = selected.loc[human_valid]
    out = _normalise(selected, cluster_from="participant_id")
    return out[_PUBLIC_COLUMNS].copy()


def data_quality_table(attempts: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_mask(attempts)
    finish = _text(attempts, "finish_reason").str.lower()
    has_error = _text(attempts, "error").ne("")
    rows = []
    for model, group in attempts.groupby("model", dropna=False):
        idx = group.index
        invalid = ~valid.loc[idx]
        truncated = invalid & finish.loc[idx].eq("length")
        api_error = invalid & ~truncated & has_error.loc[idx]
        parse_failure = invalid & ~truncated & ~has_error.loc[idx]
        n_attempted = len(group)
        n_valid = int(valid.loc[idx].sum())
        n_invalid = int(invalid.sum())
        assert n_valid + n_invalid == n_attempted
        rows.append({
            "model": model, "n_attempted": n_attempted, "n_valid": n_valid,
            "n_invalid": n_invalid, "coverage": n_valid / n_attempted if n_attempted else 0.0,
            "n_truncated": int(truncated.sum()), "n_api_error": int(api_error.sum()),
            "n_parse_failure": int(parse_failure.sum()),
        })
    return pd.DataFrame(rows, columns=[
        "model", "n_attempted", "n_valid", "n_invalid", "coverage",
        "n_truncated", "n_api_error", "n_parse_failure",
    ])
