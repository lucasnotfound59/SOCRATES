"""Input contracts and privacy-safe normalization for the Chinese replication.

The public analysis entry point in this module deliberately writes only to a
caller-supplied directory.  The legacy ``experiment/results`` tree is not an
output location for this pipeline.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


CONFIDENCE_MAP = {1: 0.50, 2: 0.625, 3: 0.75, 4: 0.875, 5: 1.00}
FORMAL_KEY = ["model", "language", "item_id", "sample_idx"]
EXPECTED_MODELS = 6
EXPECTED_ATTEMPTS_PER_MODEL = 1600
EXPECTED_TOTAL_ATTEMPTS = 9600
RANDOM_SEED = 20260918
FORMAL_MODEL_LABELS = frozenset({
    "repl-gemma4-e2b-q4km", "repl-gemma4-e4b-q4km",
    "repl-gemma4-26b-a4b-qat", "repl-qwen3-1.7b-q8",
    "repl-qwen3-4b-q4km", "repl-qwen3-14b-q4km",
})

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
        return (0, 0, -1.0, "")
    # Formal local replication labels encode the model version and parameter
    # size in the same identifier (e.g. qwen3-14b, gemma-e4b, and
    # gemma-26b-a4b-qat).  Only the first parameter-size token is the size;
    # quantization/active-expert suffixes must not affect ordering.
    base = re.sub(r"-nothink$", "", name)
    size_match = re.search(r"(?:^|[-_])(?:e)?(\d+(?:\.\d+)?)b(?:[-_]|$)", base)
    numeric = float(size_match.group(1)) if size_match else float("inf")
    lowered = base.lower()
    if "gemma" in lowered:
        family_rank = 0
    elif "qwen" in lowered:
        family_rank = 1
    else:
        family_rank = 2
    return (1, family_rank, numeric, name)


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


def fit_metad(group: pd.DataFrame) -> dict[str, float | int | str | bool]:
    """Fit MLE meta-d-prime for one cleaned group.

    The error/evidence metadata is returned even when ``metadpy`` cannot fit
    the contingency table.  This is important for ceiling groups: a failed
    estimate is still an observed group, not a missing row.
    """
    n = int(len(group))
    correct = group["correct"].astype(int)
    n_wrong = int((correct == 0).sum())
    out: dict[str, float | int | str | bool] = {
        "dprime": float("nan"), "meta_d": float("nan"),
        "m_ratio": float("nan"), "m_diff": float("nan"),
        "n": n, "n_wrong": n_wrong,
        "evidence_tier": evidence_tier(n_wrong), "fit_status": "not_run",
    }
    try:
        # Keep this conversion identical to the registered metadpy contract.
        metad_input = pd.DataFrame({
            "Stimuli": (group["gold"] == "True").astype(int),
            "Accuracy": group["correct"].astype(int),
            "Confidence": group["conf"].astype(int),
        })
        from metadpy.mle import metad
        result = metad(data=metad_input, nRatings=5, stimuli="Stimuli",
                       accuracy="Accuracy", confidence="Confidence")
        row = result.iloc[0] if hasattr(result, "iloc") else result
        # Convert all values before mutating the output so a partial/bad
        # metadpy result cannot leave a misleading partially successful fit.
        fitted = {key: float(row[key]) for key in ("dprime", "meta_d", "m_ratio", "m_diff")}
        out.update(fitted)
        out["fit_status"] = "ok"
    except Exception as exc:  # fitting failures must not drop the group
        out["fit_status"] = f"{type(exc).__name__}: {exc}"
    return out


def _cluster_draw(group: pd.DataFrame, cluster_ids: np.ndarray,
                  rng: np.random.Generator) -> pd.DataFrame:
    """Draw as many clusters as observed, retaining every row per cluster."""
    sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
    pieces = [group.loc[group["cluster_id"].eq(cluster)] for cluster in sampled]
    return pd.concat(pieces, ignore_index=True) if pieces else group.iloc[0:0].copy()


def _percentile(values: list[float], q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if len(finite) else float("nan")


def cluster_bootstrap(group: pd.DataFrame, nboot_metrics: int = 800,
                      nboot_mratio: int = 250, seed: int = RANDOM_SEED,
                      fit_fn: Callable = fit_metad) -> dict[str, float | int | str]:
    """Compute point metrics and deterministic whole-cluster percentile CIs."""
    if "cluster_id" not in group.columns:
        raise ValueError("cluster_bootstrap requires cluster_id")
    if nboot_metrics < 0 or nboot_mratio < 0:
        raise ValueError("bootstrap counts must be non-negative")
    clusters = group["cluster_id"].dropna().unique()
    if not len(clusters):
        raise ValueError("cluster_bootstrap requires at least one cluster")
    rng = np.random.default_rng(seed)
    point_fit = fit_fn(group)
    n_wrong = int((group["correct"].astype(bool) == 0).sum())
    tier = evidence_tier(n_wrong)
    accuracy = float(group["correct"].astype(bool).mean()) if len(group) else float("nan")
    overconfidence = (_mean(group, "stated") - accuracy)
    result: dict[str, float | int | str] = {
        "n": int(len(group)), "n_wrong": n_wrong,
        "accuracy": accuracy, "ece": ece(group),
        "overconfidence": overconfidence, "type2_auroc": type2_auroc(group),
        "dprime": point_fit.get("dprime", float("nan")),
        "meta_d": point_fit.get("meta_d", float("nan")),
        "m_ratio": point_fit.get("m_ratio", float("nan")),
        "m_diff": point_fit.get("m_diff", float("nan")),
        "evidence_tier": tier, "fit_status": point_fit.get("fit_status", "unknown"),
    }
    metric_values = {key: [] for key in ("accuracy", "ece", "overconfidence", "type2_auroc")}
    for _ in range(nboot_metrics):
        sample = _cluster_draw(group, clusters, rng)
        acc = float(sample["correct"].astype(bool).mean())
        metric_values["accuracy"].append(acc)
        metric_values["ece"].append(ece(sample))
        metric_values["overconfidence"].append(_mean(sample, "stated") - acc)
        metric_values["type2_auroc"].append(type2_auroc(sample))
    for key, values in metric_values.items():
        result[f"{key}_lo"] = _percentile(values, 2.5)
        result[f"{key}_hi"] = _percentile(values, 97.5)

    if tier != "data-driven":
        result["mratio_boot_status"] = "skipped_insufficient_errors"
        result["m_ratio_lo"] = result["m_ratio_hi"] = float("nan")
    else:
        m_values = []
        for _ in range(nboot_mratio):
            draw_fit = fit_fn(_cluster_draw(group, clusters, rng))
            value = draw_fit.get("m_ratio", float("nan"))
            try:
                m_values.append(float(value))
            except (TypeError, ValueError):
                m_values.append(float("nan"))
        result["m_ratio_lo"] = _percentile(m_values, 2.5)
        result["m_ratio_hi"] = _percentile(m_values, 97.5)
        result["mratio_boot_status"] = "ok" if np.isfinite(result["m_ratio_lo"]) else "insufficient_finite_fits"
    # Short aliases match the historical analysis tables.
    result["acc"] = result["accuracy"]
    result["acc_lo"] = result["accuracy_lo"]
    result["acc_hi"] = result["accuracy_hi"]
    result["overconf"] = result["overconfidence"]
    result["overconf_lo"] = result["overconfidence_lo"]
    result["overconf_hi"] = result["overconfidence_hi"]
    result["auroc2"] = result["type2_auroc"]
    result["auroc2_lo"] = result["type2_auroc_lo"]
    result["auroc2_hi"] = result["type2_auroc_hi"]
    result["m_lo"] = result["m_ratio_lo"]
    result["m_hi"] = result["m_ratio_hi"]
    return result


def bootstrap_all(data: pd.DataFrame, nboot_metrics: int = 800,
                  nboot_mratio: int = 250, seed: int = RANDOM_SEED,
                  fit_fn: Callable = fit_metad) -> pd.DataFrame:
    """Run clustered bootstrap per model, retaining the established order."""
    rows = []
    for model in _ordered_models(data):
        group = data.loc[data["model"].astype(str).eq(model)].copy()
        row = cluster_bootstrap(group, nboot_metrics=nboot_metrics,
                                nboot_mratio=nboot_mratio, seed=seed,
                                fit_fn=fit_fn)
        row["model"] = model
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    columns = ["model"] + [c for c in rows[0] if c != "model"]
    return pd.DataFrame(rows).reindex(columns=columns)


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
    missing = [c for c in (*FORMAL_KEY, "gold", "type", "subtype", "difficulty") if c not in attempts.columns]
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
    _validate_formal_model_labels(canonical["model"])
    if canonical["sample_idx"].isna().any() or not np.equal(
        canonical["sample_idx"].to_numpy(dtype=float),
        np.floor(canonical["sample_idx"].to_numpy(dtype=float)),
    ).all():
        raise ValueError("sample_idx must contain integer values")
    counts = canonical.groupby("model", dropna=False).size()
    if not (counts == EXPECTED_ATTEMPTS_PER_MODEL).all():
        raise ValueError(f"each model must have {EXPECTED_ATTEMPTS_PER_MODEL} attempts")
    # The formal file is a complete 400-item x 4-sample design for each of
    # the six registered model labels.  Counting rows alone would allow a
    # coordinated substitution of item IDs or sample indices to pass.
    for model, group in canonical.groupby("model", sort=False):
        item_counts = group.groupby("item_id", sort=False).size()
        if len(item_counts) != 400 or not item_counts.eq(4).all():
            raise ValueError(f"{model} must contain exactly 400 items with four samples each")
        sample_sets = group.groupby("item_id", sort=False)["sample_idx"].agg(lambda values: tuple(sorted(values.astype(int))))
        if not sample_sets.map(lambda values: values == (0, 1, 2, 3)).all():
            raise ValueError(f"{model} items must contain sample_idx values 0,1,2,3 exactly")
    # Cross-condition design: all six models must receive the same item set,
    # and item-level metadata must agree across model conditions.  Answers,
    # confidence, and parse outcomes are condition-specific and are not part
    # of this invariance check.
    canonical["gold"] = _text(canonical, "gold").str.lower()
    for column in ("type", "subtype", "difficulty"):
        canonical[column] = _text(canonical, column)
    reference_model = sorted(FORMAL_MODEL_LABELS, key=_model_sort_key)[0]
    reference = canonical.loc[canonical["model"].eq(reference_model)]
    reference_items = set(reference["item_id"])
    reference_meta = reference.groupby("item_id", sort=False)[["language", "gold", "type", "subtype", "difficulty"]].first()
    for model in sorted(FORMAL_MODEL_LABELS, key=_model_sort_key):
        group = canonical.loc[canonical["model"].eq(model)]
        if set(group["item_id"]) != reference_items:
            raise ValueError("all formal models must share the same item_id design")
        for item_id, item_rows in group.groupby("item_id", sort=False):
            if item_rows[["language", "gold", "type", "subtype", "difficulty"]].nunique(dropna=False).gt(1).any():
                raise ValueError("formal item metadata must be unique within each model/item")
        meta = group.groupby("item_id", sort=False)[["language", "gold", "type", "subtype", "difficulty"]].first()
        if not meta.sort_index().equals(reference_meta.sort_index()):
            raise ValueError("formal item metadata must be consistent across models")


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
    for model in _ordered_models(attempts):
        group = attempts.loc[attempts["model"].astype(str).eq(model)]
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


# ---------------------------------------------------------------------------
# Artifact pipeline

_CSV_ARTIFACTS = (
    "data_quality.csv", "model_summary.csv", "human_model_summary.csv",
    "by_type.csv", "by_difficulty.csv", "confidence_distribution.csv",
    "reliability.csv", "hallucination_breakdown.csv", "hallucination_sdt.csv",
    "metad_summary.csv", "bootstrap_summary.csv",
)
_FIGURE_NAMES = (
    "model_accuracy_ece.png", "human_model_accuracy_ece.png",
    "accuracy_by_type.png", "mratio_evidence.png", "reliability_curves.png",
    "hallucination_breakdown.png",
)


def _safe_output_dir(output_dir: Path) -> Path:
    """Resolve an output path and reject the legacy analysis locations."""
    resolved = Path(output_dir).expanduser().resolve()
    experiment = Path(__file__).resolve().parent
    legacy_root = (experiment / "results").resolve()
    forbidden = {
        (legacy_root / "analysis").resolve(),
        (legacy_root / "figures").resolve(),
    }
    # Reject the whole legacy results tree, including subdirectories whose
    # names could otherwise evade the exact-path guard.
    if resolved in forbidden or legacy_root in resolved.parents:
        raise ValueError(
            "output_dir must be isolated; experiment/results/analysis and "
            "experiment/results/figures are forbidden"
        )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "metadpy", "pymc", "arviz", "pytensor"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _strict_json_bool(value: object, label: str) -> bool:
    """Accept only a native JSON boolean from a group record."""
    if type(value) is bool:
        return value
    raise ValueError(f"{label} must be a boolean")


def _strict_csv_bool(value: object, label: str) -> bool:
    """Accept only pandas/native boolean scalars parsed from CSV."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{label} must be a boolean")


def _strict_json_int(value: object, label: str) -> int:
    """Accept only a native JSON integer from a group record."""
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _strict_csv_int(value: object, label: str) -> int:
    """Accept parsed integer scalars and integral float cells from CSV."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{label} must be an integer")


def _strict_json_number(value: object, label: str) -> float:
    """Accept only native JSON int/float values, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _strict_csv_number(value: object, label: str) -> float:
    """Accept finite pandas/numpy numeric scalars parsed from CSV."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _validate_formal_model_labels(labels: Sequence[object]) -> set[str]:
    """Require exactly the six registered replication model labels."""
    actual = {str(label) for label in labels}
    if actual != set(FORMAL_MODEL_LABELS):
        missing = sorted(FORMAL_MODEL_LABELS - actual)
        unexpected = sorted(actual - FORMAL_MODEL_LABELS)
        raise ValueError(f"formal model labels disagree: missing={missing}, unexpected={unexpected}")
    return actual


def _validate_manifest_input_digests(
    inputs: dict[str, object], model_path: Path, human_path: Path
) -> None:
    """Require the current formal input bytes to match the recorded manifest."""
    for label, path in (("model", Path(model_path)), ("human", Path(human_path))):
        expected_digest = inputs.get(label, {}).get("sha256") if isinstance(inputs.get(label), dict) else None
        if not expected_digest or expected_digest != _sha256(path.resolve()):
            raise ValueError(f"manifest {label} input sha256 does not match current input")


def _assert_recomputed_table(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    """Compare a deterministic artifact with values recomputed from inputs.

    CSV round-tripping changes integer/nullable dtypes, so the comparison is
    deliberately column- and cell-based while remaining strict about row and
    column order.  Numeric NaN is the only accepted missing-value equivalence.
    """
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ValueError(f"{label} schema/row order differs from current inputs")
    for column in expected.columns:
        left, right = actual[column], expected[column]
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            left_missing, right_missing = pd.isna(left_value), pd.isna(right_value)
            if left_missing or right_missing:
                if not (left_missing and right_missing):
                    raise ValueError(f"{label}.{column}[{index}] differs from current inputs")
                continue
            if isinstance(right_value, (int, float, np.integer, np.floating)) and not isinstance(right_value, (bool, np.bool_)):
                try:
                    left_number = float(left_value)
                    right_number = float(right_value)
                except (TypeError, ValueError):
                    raise ValueError(f"{label}.{column}[{index}] is not numeric")
                if not np.isclose(left_number, right_number, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"{label}.{column}[{index}] differs from current inputs")
            elif str(left_value) != str(right_value):
                raise ValueError(f"{label}.{column}[{index}] differs from current inputs")


_REQUIRED_FORMAL_ARTIFACTS = {
    "data_quality.csv", "model_summary.csv", "human_model_summary.csv",
    "by_type.csv", "by_difficulty.csv", "confidence_distribution.csv",
    "reliability.csv", "hallucination_breakdown.csv", "hallucination_sdt.csv",
    "metad_summary.csv", "bootstrap_summary.csv", "hmetad_summary.csv",
    "tables.md", "analysis_report_zh.md", "run_manifest.json",
}
_REQUIRED_FIGURES = {
    "model_accuracy_ece.png", "human_model_accuracy_ece.png",
    "accuracy_by_type.png", "mratio_evidence.png", "reliability_curves.png",
    "hallucination_breakdown.png",
}


def validate_analysis_artifacts(
    output_dir: Path,
    model_path: Path,
    human_path: Path,
) -> dict[str, object]:
    """Validate the complete formal artifact set and cross-table contracts."""
    output = _safe_output_dir(Path(output_dir))
    missing = [name for name in sorted(_REQUIRED_FORMAL_ARTIFACTS) if not (output / name).is_file()]
    missing.extend(f"figures/{name}" for name in sorted(_REQUIRED_FIGURES) if not (output / "figures" / name).is_file())
    if missing:
        raise ValueError(f"missing required artifacts: {', '.join(missing)}")
    for figure in _REQUIRED_FIGURES:
        data = (output / "figures" / figure).read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"invalid PNG artifact: {figure}")

    all_model, valid_model = load_model_attempts(Path(model_path))
    human = load_human_trials(Path(human_path))
    quality = pd.read_csv(output / "data_quality.csv")
    model_summary = pd.read_csv(output / "model_summary.csv")
    comparison = pd.read_csv(output / "human_model_summary.csv")
    metad = pd.read_csv(output / "metad_summary.csv")
    bootstrap = pd.read_csv(output / "bootstrap_summary.csv")
    hmetad = pd.read_csv(output / "hmetad_summary.csv")
    if len(all_model) != EXPECTED_TOTAL_ATTEMPTS or len(valid_model) != 9597 or len(human) != 1647:
        raise ValueError("input row counts do not match formal protocol")
    if int(quality["n_attempted"].sum()) != EXPECTED_TOTAL_ATTEMPTS or int(quality["n_valid"].sum()) != len(valid_model):
        raise ValueError("data_quality.csv counts do not match loaded inputs")
    if len(model_summary) != EXPECTED_MODELS or len(comparison) != 7 or len(metad) != 7 or len(bootstrap) != 7 or len(hmetad) != 7:
        raise ValueError("formal table row counts do not match protocol")
    model_names = _validate_formal_model_labels(model_summary["model"])
    expected_model_order = sorted(FORMAL_MODEL_LABELS, key=_model_sort_key)
    expected_comparison_order = ["human", *expected_model_order]
    if model_summary["model"].astype(str).tolist() != expected_model_order:
        raise ValueError("model_summary rows are not in the registered formal order")
    for label, frame in (("human_model_summary", comparison), ("metad_summary", metad),
                         ("bootstrap_summary", bootstrap), ("hmetad_summary", hmetad)):
        if frame["model"].astype(str).tolist() != expected_comparison_order:
            raise ValueError(f"{label} rows are not in human-then-formal order")
    comparison_names = set(comparison["model"].astype(str))
    if comparison_names != model_names | {"human"} or set(metad["model"].astype(str)) != comparison_names:
        raise ValueError("model names disagree across comparison tables")
    if set(bootstrap["model"].astype(str)) != comparison_names or set(hmetad["model"].astype(str)) != comparison_names:
        raise ValueError("model names disagree across bootstrap/Bayesian tables")
    comparison_n = comparison.set_index("model")["n"].astype(int)
    hmetad_n = hmetad.set_index("model")["n"].astype(int)
    if not comparison_n.equals(hmetad_n.reindex(comparison_n.index)):
        raise ValueError("Bayesian group counts disagree with human_model_summary.csv")
    source_counts = {"human": len(human)}
    source_wrong = {"human": int((~human["correct"].astype(bool)).sum())}
    source_counts.update({
        model: int((valid_model["model"].astype(str) == model).sum())
        for model in model_names
    })
    source_wrong.update({
        model: int((~valid_model.loc[valid_model["model"].astype(str).eq(model), "correct"].astype(bool)).sum())
        for model in model_names
    })
    if any(int(comparison_n[model]) != source_counts[model] for model in comparison_names):
        raise ValueError("Bayesian group counts disagree with loaded source inputs")

    # Recompute every deterministic non-Bayesian table from the fresh input
    # bytes.  Manifest/hash consistency alone cannot detect a coherent edit
    # to a table followed by a refreshed manifest.
    matched = pd.concat([valid_model, human], ignore_index=True, sort=False)
    metad_rows = []
    for model_name in _ordered_models(matched):
        fit = fit_metad(matched.loc[matched["model"].astype(str).eq(model_name)])
        fit["model"] = model_name
        metad_rows.append(fit)
    expected_tables = {
        "data_quality.csv": data_quality_table(all_model),
        "model_summary.csv": summarize_groups(valid_model),
        "human_model_summary.csv": summarize_groups(matched),
        "by_type.csv": summarize_factor(matched, "type", ["常规", "学科", "陷阱", "幻觉"]),
        "by_difficulty.csv": summarize_factor(matched, "difficulty", ["易", "中", "难"]),
        "confidence_distribution.csv": confidence_distribution(matched),
        "reliability.csv": reliability_table(matched),
        "hallucination_breakdown.csv": hallucination_breakdown(matched),
        "hallucination_sdt.csv": hallucination_sdt(matched),
        "metad_summary.csv": pd.DataFrame(metad_rows, columns=[
            "model", "dprime", "meta_d", "m_ratio", "m_diff", "n", "n_wrong",
            "evidence_tier", "fit_status",
        ]),
    }
    actual_tables = {
        "data_quality.csv": quality, "model_summary.csv": model_summary,
        "human_model_summary.csv": comparison,
        "by_type.csv": pd.read_csv(output / "by_type.csv"),
        "by_difficulty.csv": pd.read_csv(output / "by_difficulty.csv"),
        "confidence_distribution.csv": pd.read_csv(output / "confidence_distribution.csv"),
        "reliability.csv": pd.read_csv(output / "reliability.csv"),
        "hallucination_breakdown.csv": pd.read_csv(output / "hallucination_breakdown.csv"),
        "hallucination_sdt.csv": pd.read_csv(output / "hallucination_sdt.csv"),
        "metad_summary.csv": metad,
    }
    for name, expected in expected_tables.items():
        _assert_recomputed_table(actual_tables[name], expected, name)

    # The fixed formal bootstrap is reproducible from the current input
    # bytes; never cache this expected frame because a mutable cache could
    # itself become an audit bypass.
    expected_bootstrap = bootstrap_all(matched, nboot_metrics=800, nboot_mratio=250, seed=RANDOM_SEED)
    _assert_recomputed_table(bootstrap, expected_bootstrap, "bootstrap_summary.csv")

    # Every emitted model-bearing CSV follows the same canonical sequence;
    # checking only sets would allow swapped rows to remain numerically
    # self-consistent.
    for name, frame, order in (
        ("data_quality.csv", quality, expected_model_order),
        ("by_type.csv", actual_tables["by_type.csv"], expected_comparison_order),
        ("by_difficulty.csv", actual_tables["by_difficulty.csv"], expected_comparison_order),
        ("confidence_distribution.csv", actual_tables["confidence_distribution.csv"], expected_comparison_order),
        ("reliability.csv", actual_tables["reliability.csv"], expected_comparison_order),
        ("hallucination_breakdown.csv", actual_tables["hallucination_breakdown.csv"], expected_comparison_order),
        ("hallucination_sdt.csv", actual_tables["hallucination_sdt.csv"], expected_comparison_order),
    ):
        if "model" not in frame.columns:
            raise ValueError(f"{name} is missing model ordering column")
        observed = frame["model"].astype(str).drop_duplicates().tolist()
        if observed != order:
            raise ValueError(f"{name} rows are not in canonical model order")

    expected_bootstrap_order = expected_comparison_order
    if bootstrap["model"].astype(str).tolist() != expected_bootstrap_order:
        raise ValueError("bootstrap_summary rows are not in formal order")
    for _, row in bootstrap.iterrows():
        model_name = str(row["model"])
        expected_row = expected_tables["human_model_summary.csv"].set_index("model").loc[model_name]
        for column in ("n", "n_wrong", "evidence_tier"):
            if column not in bootstrap or str(row[column]) != str(expected_row[column]):
                raise ValueError(f"bootstrap {model_name}.{column} disagrees with current inputs")
        expected_metad = expected_tables["metad_summary.csv"].set_index("model").loc[model_name]
        point_pairs = {
            "accuracy": expected_row["accuracy"], "ece": expected_row["ece"],
            "overconfidence": expected_row["overconfidence"],
            "type2_auroc": expected_row["type2_auroc"],
            "dprime": expected_metad["dprime"], "meta_d": expected_metad["meta_d"],
            "m_ratio": expected_metad["m_ratio"], "m_diff": expected_metad["m_diff"],
            "fit_status": expected_metad["fit_status"],
        }
        for column, expected_value in point_pairs.items():
            actual_value = row.get(column)
            if isinstance(expected_value, str):
                if str(actual_value) != expected_value:
                    raise ValueError(f"bootstrap {model_name}.{column} disagrees with current inputs")
            elif pd.isna(expected_value):
                if pd.notna(actual_value):
                    raise ValueError(f"bootstrap {model_name}.{column} disagrees with current inputs")
            elif pd.isna(actual_value) or not np.isclose(float(actual_value), float(expected_value), rtol=1e-10, atol=1e-12):
                raise ValueError(f"bootstrap {model_name}.{column} disagrees with current inputs")
    if not {"n", "n_wrong", "evidence_tier", "mratio_boot_status"}.issubset(bootstrap.columns):
        raise ValueError("bootstrap_summary is missing its reproducibility contract columns")
    for _, row in bootstrap.iterrows():
        tier = str(row["evidence_tier"])
        status = str(row["mratio_boot_status"])
        lo, hi = pd.to_numeric(row.get("m_ratio_lo"), errors="coerce"), pd.to_numeric(row.get("m_ratio_hi"), errors="coerce")
        if tier != "data-driven" and (status != "skipped_insufficient_errors" or pd.notna(lo) or pd.notna(hi)):
            raise ValueError("non-data-driven M-ratio intervals must be explicitly skipped and blank")
        if tier == "data-driven" and status == "ok" and (pd.isna(lo) or pd.isna(hi) or float(lo) > float(hi)):
            raise ValueError("data-driven M-ratio interval is not estimable/ordered")

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("random_seed") != RANDOM_SEED or manifest.get("bootstrap") != {"nboot_metrics": 800, "nboot_mratio": 250}:
        raise ValueError("manifest seed/bootstrap settings do not match formal protocol")
    inputs = manifest.get("inputs", {})
    if inputs.get("model", {}).get("valid_row_count") != 9597 or inputs.get("human", {}).get("valid_row_count") != 1647:
        raise ValueError("manifest input counts do not match formal protocol")
    _validate_manifest_input_digests(inputs, Path(model_path), Path(human_path))
    bayesian = manifest.get("bayesian", {})
    group_names = sorted(p.stem for p in (output / "hmetad" / "groups").glob("*.json"))
    expected_bayesian_order = ["human", *sorted(FORMAL_MODEL_LABELS, key=_model_sort_key)]
    if bayesian.get("draws") != 800 or bayesian.get("chains") != 2 or bayesian.get("seed") != RANDOM_SEED or bayesian.get("groups") != expected_bayesian_order or len(group_names) != 7:
        raise ValueError("manifest Bayesian settings/groups are incomplete")
    report_text = (output / "analysis_report_zh.md").read_text(encoding="utf-8")
    report_section = re.search(r"(?ms)^## Bayesian 状态\n.*?(?=^## |\Z)", report_text)
    report_models = re.findall(r"^- `([^`]+)`：", report_section.group(0) if report_section else "", flags=re.MULTILINE)
    if report_models != expected_bayesian_order:
        raise ValueError("Bayesian report rows are not in canonical model order")
    group_records: dict[str, dict[str, object]] = {}
    for group_path in sorted((output / "hmetad" / "groups").glob("*.json")):
        record = json.loads(group_path.read_text(encoding="utf-8"))
        model = str(record.get("model", ""))
        if not model or model in group_records:
            raise ValueError("Bayesian group JSONs have missing or duplicate model labels")
        group_records[model] = record
    if set(group_records) != comparison_names:
        raise ValueError("Bayesian group JSON order/labels disagree with formal order")
    hmetad_by_model = hmetad.set_index("model")
    expected_input_digests = {
        "model": inputs["model"].get("sha256"),
        "human": inputs["human"].get("sha256"),
    }
    core_fields = ("n", "n_wrong", "error_count", "draws", "chains", "seed")
    diagnostic_fields = (
        "m_ratio_mean", "hdi_lo", "hdi_hi", "rhat", "n_divergent", "divergence_rate",
        "reliable", "evidence_tier",
    )
    for model in expected_bayesian_order:
        record = group_records[model]
        row = hmetad_by_model.loc[model]
        status = str(record.get("status", ""))
        if status not in {"complete", "failed"} or str(row["status"]) != status:
            raise ValueError(f"Bayesian status mismatch for {model}")
        if record.get("input_digests") != expected_input_digests:
            raise ValueError(f"Bayesian input digest mismatch for {model}")
        for field in core_fields:
            record_has_field = field in record
            record_value = record[field] if record_has_field else None
            summary_value = row[field] if field in hmetad.columns else None
            # Failed preprocessing records legitimately have no posterior
            # draws, chains, or error count; every field they do provide is
            # still checked against the summary rather than ignored.
            if not record_has_field:
                if status == "complete":
                    raise ValueError(f"Bayesian group is missing {field}: {model}")
                if field in {"n_wrong", "error_count"}:
                    raise ValueError(f"Bayesian group is missing source-bound {field}: {model}")
                if not pd.isna(summary_value):
                    raise ValueError(f"Bayesian {field} unexpectedly present in summary for {model}")
                continue
            if record_value is None:
                raise ValueError(f"Bayesian {field} is explicitly null for {model}")
            record_int = _strict_json_int(record_value, f"{model}.{field}")
            if pd.isna(summary_value) or _strict_csv_int(summary_value, f"summary {model}.{field}") != record_int:
                raise ValueError(f"Bayesian {field} mismatch for {model}")
            if field in {"n_wrong", "error_count"} and record_int != source_wrong[model]:
                raise ValueError(f"Bayesian {field} disagrees with current input errors for {model}")
        if "evidence_tier" not in record:
            raise ValueError(f"Bayesian group is missing source-bound evidence_tier: {model}")
        if str(record["evidence_tier"]) != evidence_tier(source_wrong[model]):
            raise ValueError(f"Bayesian evidence_tier disagrees with current input errors for {model}")
        if status == "failed":
            if not str(record.get("error_type", "")).strip() or not str(record.get("error_message", "")).strip():
                raise ValueError(f"failed Bayesian group lacks error details: {model}")
            for field in ("error_type", "error_message"):
                if str(row[field]) != str(record[field]):
                    raise ValueError(f"Bayesian {field} mismatch for {model}")
            for field in diagnostic_fields:
                record_has_field = field in record
                record_value = record[field] if record_has_field else None
                summary_value = row[field] if field in hmetad.columns else None
                if not record_has_field:
                    if not pd.isna(summary_value):
                        raise ValueError(f"Bayesian {field} unexpectedly present in summary for {model}")
                    continue
                if record_value is None:
                    raise ValueError(f"Bayesian {field} is explicitly null for {model}")
                if field == "reliable":
                    if _strict_json_bool(record_value, f"{model}.{field}") != _strict_csv_bool(summary_value, f"summary {model}.{field}"):
                        raise ValueError(f"Bayesian {field} mismatch for {model}")
                elif field == "evidence_tier":
                    if str(summary_value) != str(record_value):
                        raise ValueError(f"Bayesian {field} mismatch for {model}")
                elif not np.isclose(_strict_json_number(record_value, f"{model}.{field}"), _strict_csv_number(summary_value, f"summary {model}.{field}")):
                    raise ValueError(f"Bayesian {field} mismatch for {model}")
            continue
        for field in diagnostic_fields:
            if field not in record or field not in hmetad.columns:
                raise ValueError(f"Bayesian group is missing {field}: {model}")
            if field == "reliable":
                if _strict_json_bool(record[field], f"{model}.{field}") != _strict_csv_bool(row[field], f"summary {model}.{field}"):
                    raise ValueError(f"Bayesian {field} mismatch for {model}")
            elif field == "evidence_tier":
                if not str(record[field]).strip() or str(row[field]).strip() != str(record[field]):
                    raise ValueError(f"Bayesian {field} mismatch for {model}")
            elif not np.isclose(_strict_json_number(record[field], f"{model}.{field}"), _strict_csv_number(row[field], f"summary {model}.{field}")):
                raise ValueError(f"Bayesian {field} mismatch for {model}")
    versions = manifest.get("package_versions", {})
    if any(versions.get(name) in (None, "not-installed") for name in ("pymc", "arviz", "pytensor", "metadpy")):
        raise ValueError("manifest is missing Bayesian package versions")
    compatibility = manifest.get("runtime_compatibility", {})
    if not compatibility.get("helper_version") or "activated" not in compatibility:
        raise ValueError("manifest is missing runtime compatibility record")
    hashes = manifest.get("artifact_hashes", {})
    artifacts = manifest.get("artifacts", {})
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if set(artifacts) != actual_files or set(hashes) != actual_files:
        raise ValueError("manifest does not enumerate every generated artifact")
    for relative, digest in hashes.items():
        if relative == "run_manifest.json":
            if digest is not None:
                raise ValueError("manifest self-hash must be null")
        elif digest != _sha256(output / relative):
            raise ValueError(f"artifact hash mismatch: {relative}")
    return {
        "model_rows": len(all_model), "model_valid_rows": len(valid_model),
        "human_valid_rows": len(human), "model_groups": len(model_summary),
        "comparison_groups": len(comparison), "bayesian_groups": len(hmetad),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        # Keep the artifact usable on minimal environments without tabulate.
        columns = [str(c) for c in frame.columns]
        lines = ["| " + " | ".join(columns) + " |",
                 "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
        return "\n".join(lines)


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA" if value is None or pd.isna(value) else str(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _top_model(summary: pd.DataFrame, metric: str, ascending: bool = False) -> str:
    finite = summary.loc[pd.to_numeric(summary[metric], errors="coerce").notna()]
    if finite.empty:
        return "暂无"
    return str(finite.sort_values(metric, ascending=ascending, kind="stable").iloc[0]["model"])


def _ranked_models(summary: pd.DataFrame, metric: str, ascending: bool = False) -> list[str]:
    """Return every model in a stable numeric ranking, with missing values last."""
    if metric not in summary or "model" not in summary:
        return []
    ranked = summary[["model", metric]].copy()
    ranked["_value"] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked["_finite"] = ranked["_value"].notna()
    ranked = ranked.sort_values(
        ["_finite", "_value"], ascending=[False, ascending], kind="stable"
    )
    return ranked["model"].astype(str).tolist()


def _mratio_plot_data(
    metad_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame
) -> pd.DataFrame:
    """Join point estimates and CIs in the fixed report/model order."""
    columns = ["model", "x", "m_ratio", "m_ratio_lo", "m_ratio_hi", "evidence_tier"]
    if metad_summary.empty:
        return pd.DataFrame(columns=columns)
    frame = metad_summary[["model", "m_ratio", "evidence_tier"]].copy()
    intervals = (
        bootstrap_summary[["model", "m_ratio_lo", "m_ratio_hi"]].copy()
        if {"model", "m_ratio_lo", "m_ratio_hi"}.issubset(bootstrap_summary.columns)
        else pd.DataFrame(columns=["model", "m_ratio_lo", "m_ratio_hi"])
    )
    frame = frame.merge(intervals, on="model", how="left", sort=False)
    frame["x"] = np.arange(len(frame), dtype=float)
    return frame[columns]


def _write_tables(
    path: Path,
    model_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
    by_type: pd.DataFrame,
    hallucination: pd.DataFrame,
    metad_summary: pd.DataFrame,
) -> Path:
    sections = [
        "# 中文复现分析表格\n",
        "## 六模型核心指标\n\n" + _markdown_table(model_summary),
        "## 中文人类与模型对照\n\n" + _markdown_table(matched_summary),
        "## 按题型准确率\n\n" + _markdown_table(
            by_type[["model", "type", "n", "accuracy"]]
        ),
        "## 幻觉题 SDT\n\n" + _markdown_table(hallucination),
        "## M-ratio 与证据等级\n\n" + _markdown_table(
            metad_summary[["model", "n", "n_wrong", "m_ratio", "evidence_tier", "fit_status"]]
        ),
    ]
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def _write_report(
    path: Path,
    quality: pd.DataFrame,
    model_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
    by_type: pd.DataFrame,
    by_difficulty: pd.DataFrame,
    hallucination: pd.DataFrame,
    hallucination_sdt_frame: pd.DataFrame,
    metad_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> Path:
    invalid = int(pd.to_numeric(quality.get("n_invalid", 0), errors="coerce").fillna(0).sum())
    valid = int(pd.to_numeric(quality.get("n_valid", 0), errors="coerce").fillna(0).sum())
    attempted = int(pd.to_numeric(quality.get("n_attempted", 0), errors="coerce").fillna(0).sum())
    accuracy_leader = _top_model(model_summary, "accuracy")
    ece_leader = _top_model(model_summary, "ece", ascending=True)
    accuracy_ranking = _ranked_models(model_summary, "accuracy", ascending=False)
    ece_ranking = _ranked_models(model_summary, "ece", ascending=True)
    human = matched_summary.loc[matched_summary["model"].eq("human")]
    human_acc = _fmt(human.iloc[0]["accuracy"]) if len(human) else "NA"
    human_ece = _fmt(human.iloc[0]["ece"]) if len(human) else "NA"
    model_acc = _fmt(model_summary["accuracy"].mean()) if len(model_summary) else "NA"
    model_ece = _fmt(model_summary["ece"].mean()) if len(model_summary) else "NA"

    tier_lines = []
    for tier in ("data-driven", "regularized", "prior-dominated"):
        rows = metad_summary.loc[metad_summary["evidence_tier"].eq(tier)]
        if rows.empty:
            tier_lines.append(f"- `{tier}`：本次没有组落入该等级。")
        else:
            values = ", ".join(
                f"{row.model}（M-ratio={_fmt(row.m_ratio)}，错误数={int(row.n_wrong)}）"
                for row in rows.itertuples()
            )
            caveat = "可作数据驱动比较" if tier == "data-driven" else "仅作透明报告，不据此下元认知强弱结论"
            tier_lines.append(f"- `{tier}`：{values}；{caveat}。")

    type_rows = by_type.loc[pd.to_numeric(by_type["accuracy"], errors="coerce").notna()]
    type_best = "暂无"
    if not type_rows.empty:
        best = type_rows.sort_values("accuracy", ascending=False, kind="stable").iloc[0]
        type_best = f"{best['model']} 在 {best['type']} 上准确率 {_fmt(best['accuracy'])}"
    difficulty_rows = by_difficulty.loc[pd.to_numeric(by_difficulty["accuracy"], errors="coerce").notna()]
    difficulty_best = "暂无"
    if not difficulty_rows.empty:
        best = difficulty_rows.sort_values("accuracy", ascending=False, kind="stable").iloc[0]
        difficulty_best = f"{best['model']} 在 {best['difficulty']} 上准确率 {_fmt(best['accuracy'])}"
    hall_rows = hallucination_sdt_frame.loc[hallucination_sdt_frame["model"].ne("human")]
    hall_best = "暂无"
    if not hall_rows.empty:
        best = hall_rows.sort_values("dprime", ascending=False, kind="stable").iloc[0]
        hall_best = f"{best['model']} 的 d′={_fmt(best['dprime'])}"
    hall_breakdown = "暂无"
    if len(hallucination):
        best = hallucination.sort_values("accuracy_gap", ascending=False, kind="stable").iloc[0]
        hall_breakdown = f"{best['model']} 的真实-虚构准确率差为 {_fmt(best['accuracy_gap'])}"

    lines = [
        "# 中文复现分析报告",
        "",
        "## 数据质量",
        "",
        f"模型输入审计共 {attempted} 条尝试，其中 {valid} 条进入指标分母；质量表保留全部原始记录，"
        f"无效/非响应记录共 {invalid} 条；质量表保留这些记录并单独标记，而不是静默删除，"
        "具体类别见 `data_quality.csv`。"
        + ("该数量与协议预期的三条非响应记录一致。" if invalid == 3 else "若正式输入预期为三条非响应记录，应据此审查输入质量。"),
        "",
        "## 六模型准确率与校准",
        "",
        f"按预设的家族/规模顺序展示六个模型；准确率最高的是 {accuracy_leader}，ECE 最低的是 {ece_leader}。"
        f"六模型平均准确率为 {model_acc}，平均 ECE 为 {model_ece}。这些排名只描述观测行为，"
        "不等同于模型内部过程。",
        f"完整准确率排名（高到低）：{' > '.join(accuracy_ranking) or '暂无'}。",
        f"完整 ECE 排名（低到高）：{' > '.join(ece_ranking) or '暂无'}。",
        "",
        "## 中文人类与模型比较",
        "",
        f"在相同中文任务与五档置信度编码下，人类行的准确率为 {human_acc}、ECE 为 {human_ece}；"
        f"六模型平均值分别为 {model_acc} 和 {model_ece}。人类记录按参与者聚类，模型记录按题目聚类，"
        "因此该对照用于描述性比较，不应被解释为完全相同抽样结构下的因果差异。",
        "",
        "## 题型、难度与幻觉",
        "",
        f"分层表中观测到的最高题型-模型组合为：{type_best}；最高难度-模型组合为：{difficulty_best}。"
        "完整难度分层结果见 `by_difficulty.csv`；"
        f"幻觉题 SDT 的最高 d′ 组合为：{hall_best}；准确率差异最大的组合为：{hall_breakdown}。"
        "虚构命题与真实冷僻命题的准确率、置信度差异见"
        "`hallucination_breakdown.csv`，SDT 校正值见 `hallucination_sdt.csv`。",
        "",
        "## M-ratio 结果与证据等级",
        "",
        *tier_lines,
        "",
        "MLE 点估计和聚类 bootstrap 区间均保留；M-ratio bootstrap 区间仅在数据驱动且拟合可估计的组中报告，"
        "regularized/prior-dominated 组的区间留空；接近满分的组可能没有稳定的错误结构，"
        "因此这些组不会被包装成确定的元认知结论。",
        "",
        "## Bayesian 状态",
        "",
        "pending：Bayesian HMeta-d 汇总属于后续任务；本报告只呈现非 Bayesian 的 MLE 与 bootstrap 结果。",
        "",
        "## 局限性",
        "",
        "- 人类与模型具有不等的 sampling structures（人类按参与者、模型按题目聚类）。",
        "- ceiling effects 会令近满分组的 M-ratio 估计不稳定。",
        "- quantization/model-family confounding 使规模趋势不能单独归因于参数量。",
        "- 所有结果都是 behavioral-only interpretation，不推断不可观测的内部机制。",
        "- 本研究不作 consciousness claim，也不把置信度行为等同于意识。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _plot_artifacts(
    figure_dir: Path,
    model_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
    by_type: pd.DataFrame,
    metad_summary: pd.DataFrame,
    reliability: pd.DataFrame,
    hallucination: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # The report is Chinese; use an installed CJK font when available and
    # retain DejaVu as a portable fallback for headless CI.
    matplotlib.rcParams["font.sans-serif"] = ["STSong", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def save(name: str) -> None:
        target = figure_dir / name
        plt.tight_layout()
        plt.savefig(target, dpi=140)
        plt.close()
        paths[name] = target

    def accuracy_ece(frame: pd.DataFrame, title: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = frame["model"].astype(str).tolist() if len(frame) else []
        x = np.arange(len(labels))
        values = pd.to_numeric(frame.get("accuracy", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        errors = np.zeros((2, len(values)))
        if len(bootstrap_summary):
            boot = bootstrap_summary.set_index("model")
            for idx, label in enumerate(labels):
                if label in boot.index and "accuracy_lo" in boot and "accuracy_hi" in boot and np.isfinite(values[idx]):
                    lo, hi = float(boot.loc[label, "accuracy_lo"]), float(boot.loc[label, "accuracy_hi"])
                    if np.isfinite(lo) and np.isfinite(hi):
                        errors[:, idx] = [max(0.0, values[idx] - lo), max(0.0, hi - values[idx])]
        axes[0].bar(x, values, yerr=errors if len(values) else None, capsize=3, color="#277da1")
        axes[0].set_ylim(0, 1); axes[0].set_ylabel("Accuracy"); axes[0].set_xticks(x, labels, rotation=45, ha="right")
        ece_values = pd.to_numeric(frame.get("ece", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        ece_errors = np.zeros((2, len(ece_values)))
        if len(bootstrap_summary):
            boot = bootstrap_summary.set_index("model")
            for idx, label in enumerate(labels):
                if label in boot.index and "ece_lo" in boot and "ece_hi" in boot and np.isfinite(ece_values[idx]):
                    lo, hi = float(boot.loc[label, "ece_lo"]), float(boot.loc[label, "ece_hi"])
                    if np.isfinite(lo) and np.isfinite(hi):
                        ece_errors[:, idx] = [max(0.0, ece_values[idx] - lo), max(0.0, hi - ece_values[idx])]
        axes[1].bar(x, ece_values, yerr=ece_errors if len(ece_values) else None, capsize=3, color="#f9844a")
        # ECE is an absolute calibration error on a small scale, whereas
        # accuracy spans the full 0-1 range.  Sharing one axis crushes every
        # ECE bar into the bottom tenth of the panel and makes the
        # human-versus-model difference unreadable, so the ECE panel gets its
        # own limit, scaled to the data with headroom for the interval caps.
        ece_top = 0.1
        if len(ece_values):
            finite_hi = np.where(np.isfinite(ece_values), ece_values, np.nan) + ece_errors[1]
            finite_hi = finite_hi[np.isfinite(finite_hi)]
            if len(finite_hi):
                ece_top = max(0.1, float(finite_hi.max()) * 1.15)
        axes[1].set_ylim(0, ece_top); axes[1].set_ylabel("ECE"); axes[1].set_xticks(x, labels, rotation=45, ha="right")
        fig.suptitle(title)

    accuracy_ece(model_summary, "Model accuracy and calibration")
    save("model_accuracy_ece.png")
    accuracy_ece(matched_summary, "Chinese human and model comparison")
    save("human_model_accuracy_ece.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    types = list(dict.fromkeys(by_type["type"].astype(str))) if len(by_type) else []
    models = matched_summary["model"].astype(str).tolist() if len(matched_summary) else []
    width = 0.8 / max(len(models), 1)
    for idx, model in enumerate(models):
        rows = by_type.loc[by_type["model"].eq(model)].set_index("type")
        values = [float(rows.loc[level, "accuracy"]) if level in rows.index and pd.notna(rows.loc[level, "accuracy"]) else np.nan for level in types]
        ax.bar(np.arange(len(types)) + idx * width, values, width=width, label=model)
    ax.set_xticks(np.arange(len(types)) + width * max(len(models) - 1, 0) / 2, types)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by item type")
    if models:
        fig.subplots_adjust(right=0.76)
        ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    save("accuracy_by_type.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"data-driven": "#2a9d8f", "regularized": "#e9c46a", "prior-dominated": "#e76f51"}
    mratio_frame = _mratio_plot_data(metad_summary, bootstrap_summary)
    for tier in ("data-driven", "regularized", "prior-dominated"):
        rows = mratio_frame.loc[mratio_frame["evidence_tier"].eq(tier)].copy()
        rows["m_ratio"] = pd.to_numeric(rows["m_ratio"], errors="coerce")
        rows["m_ratio_lo"] = pd.to_numeric(rows["m_ratio_lo"], errors="coerce")
        rows["m_ratio_hi"] = pd.to_numeric(rows["m_ratio_hi"], errors="coerce")
        rows = rows.loc[rows["m_ratio"].notna()]
        if len(rows):
            lower = (rows["m_ratio"] - rows["m_ratio_lo"]).clip(lower=0).fillna(0)
            upper = (rows["m_ratio_hi"] - rows["m_ratio"]).clip(lower=0).fillna(0)
            ax.errorbar(rows["x"], rows["m_ratio"], yerr=np.vstack([lower, upper]), fmt="o",
                        color=colors[tier], label=tier, capsize=3, markersize=6)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_ylim(bottom=0); ax.set_ylabel("M-ratio"); ax.set_title("M-ratio by evidence tier")
    ax.set_xticks(mratio_frame["x"], mratio_frame["model"], rotation=45, ha="right")
    if len(mratio_frame):
        # Bootstrap intervals are computed only for the data-driven tier, so
        # the legend states which tiers carry an interval instead of drawing
        # an interval glyph for a point that has none.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f"{name} (interval)" if name == "data-driven" else f"{name} (point only)"
                            for name in labels], fontsize="small", loc="upper left")
    save("mratio_evidence.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in reliability["model"].astype(str).unique() if len(reliability) else []:
        rows = reliability.loc[reliability["model"].eq(model)].sort_values("confidence")
        ax.plot(rows["stated"], rows["accuracy"], marker="o", label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="perfect calibration")
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0, 1); ax.set_xlabel("Stated confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability curves"); ax.legend(fontsize="small", ncol=2)
    save("reliability_curves.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    if len(hallucination):
        labels = hallucination["model"].astype(str).tolist(); x = np.arange(len(labels)); width = 0.36
        ax.bar(x - width / 2, hallucination["fictional_accuracy"], width, label="fictional")
        ax.bar(x + width / 2, hallucination["real_accuracy"], width, label="real obscure")
        ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Hallucination breakdown")
    if len(hallucination):
        ax.legend()
    save("hallucination_breakdown.png")
    return paths


def plot_bayes_mratio_vs_errors(
    bayes: pd.DataFrame,
    output_path: Path,
    *,
    human_label: str = "human",
    title: str = "Bayesian M-ratio against error count",
) -> Path:
    """Plot Bayesian M-ratio for every group against its error count.

    A figure that shows only the data-driven groups hides how the remaining
    groups behave and why they were excluded.  This view places every group on
    one axis so the instability at low error counts is visible rather than
    asserted: filled markers with intervals are groups that satisfied the
    prespecified Bayesian diagnostics, and open markers are groups whose fits
    were prior-dominated or divergent and therefore carry no interval.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"model", "n_wrong", "mr_mean", "hdi_lo", "hdi_hi"}
    missing = sorted(required - set(bayes.columns))
    if missing:
        raise ValueError(f"bayes frame is missing columns: {missing}")
    frame = bayes.copy()
    for column in ("n_wrong", "mr_mean", "hdi_lo", "hdi_hi"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "reliable" in frame.columns:
        frame["reliable"] = frame["reliable"].astype(str).str.lower().isin({"true", "1", "1.0"})
    else:
        frame["reliable"] = False
    frame = frame.loc[frame["mr_mean"].notna()].sort_values(
        ["n_wrong", "model"], ascending=[True, True]
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no groups with a finite Bayesian M-ratio to plot")

    human = frame.loc[frame["model"].astype(str).eq(human_label)]
    rest = frame.loc[~frame["model"].astype(str).eq(human_label)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    # Visual reference: the human estimate and the lower edge of its interval.
    if len(human):
        human_row = human.iloc[0]
        human_band = float(human_row["hdi_lo"])
        ax.axhline(float(human_row["mr_mean"]), color="#c1121f", linewidth=1.1,
                   linestyle="-", label=f"{human_label} M-ratio")
        ax.axhline(human_band, color="#c1121f", linewidth=1.0, linestyle=":",
                   label=f"{human_label} 95% HDI lower bound")
    ax.axhline(1.0, color="0.5", linewidth=0.9, linestyle="--", label="M-ratio = 1")

    x = np.arange(len(rest), dtype=float)
    ok = rest["reliable"].to_numpy()
    for mask, label, filled in ((ok, "Bayesian diagnostics passed", True),
                                (~ok, "prior-dominated or divergent", False)):
        if not mask.any():
            continue
        rows = rest.loc[mask]
        means = rows["mr_mean"].to_numpy(dtype=float)
        lower = np.clip(means - rows["hdi_lo"].to_numpy(dtype=float), 0, None)
        upper = np.clip(rows["hdi_hi"].to_numpy(dtype=float) - means, 0, None)
        lower = np.where(np.isfinite(lower), lower, 0.0)
        upper = np.where(np.isfinite(upper), upper, 0.0)
        if filled:
            ax.errorbar(x[mask], means, yerr=np.vstack([lower, upper]), fmt="o",
                        color="#2a9d8f", ecolor="#2a9d8f", capsize=3, markersize=6, label=label)
        else:
            # Hollow grey markers: the interval is not interpreted for these groups.
            ax.errorbar(x[mask], means, yerr=np.vstack([lower, upper]), fmt="o",
                        mfc="white", mec="0.45", color="0.45", ecolor="0.75",
                        capsize=3, markersize=6, label=label)

    # Put the error count in the tick label itself: the whole point of the
    # figure is that position on the x-axis encodes how much evidence each
    # group contributes.
    tick_labels = [
        f"{row.model}\n({int(row.n_wrong)} errors)" for row in rest.itertuples(index=False)
    ]
    ax.set_xticks(x, tick_labels, rotation=45, ha="right")
    ax.set_ylabel("Bayesian M-ratio")
    ax.set_title(title)
    ax.set_xlabel("Groups ordered by number of error trials (fewest first)")
    ax.legend(fontsize="small", loc="lower right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def run_analysis(
    model_path: Path,
    human_path: Path,
    output_dir: Path,
    nboot_metrics: int = 800,
    nboot_mratio: int = 250,
    seed: int = RANDOM_SEED,
    resume: bool = False,
) -> dict[str, Path]:
    """Run the non-Bayesian Chinese replication and write isolated artifacts."""
    if nboot_metrics < 0 or nboot_mratio < 0:
        raise ValueError("bootstrap counts must be non-negative")
    output = _safe_output_dir(Path(output_dir))
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"analysis output exists; pass resume=True: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_path, human_path = Path(model_path).expanduser().resolve(), Path(human_path).expanduser().resolve()
    started = datetime.now(timezone.utc)
    all_model, model = load_model_attempts(model_path)
    human = load_human_trials(human_path)
    quality = data_quality_table(all_model)
    matched = pd.concat([model, human], ignore_index=True, sort=False)
    model_summary = summarize_groups(model)
    matched_summary = summarize_groups(matched)
    by_type = summarize_factor(matched, "type", ["常规", "学科", "陷阱", "幻觉"])
    by_difficulty = summarize_factor(matched, "difficulty", ["易", "中", "难"])
    confidence = confidence_distribution(matched)
    reliability = reliability_table(matched)
    hallucination = hallucination_breakdown(matched)
    sdt = hallucination_sdt(matched)
    metad_rows = []
    for model_name in _ordered_models(matched):
        fit = fit_metad(matched.loc[matched["model"].astype(str).eq(model_name)])
        fit["model"] = model_name
        metad_rows.append(fit)
    metad_columns = ["model", "dprime", "meta_d", "m_ratio", "m_diff", "n", "n_wrong", "evidence_tier", "fit_status"]
    metad_summary = pd.DataFrame(metad_rows, columns=metad_columns)
    bootstrap = bootstrap_all(matched, nboot_metrics=nboot_metrics, nboot_mratio=nboot_mratio, seed=seed)

    artifacts: dict[str, Path] = {}
    frames = {
        "data_quality.csv": quality, "model_summary.csv": model_summary,
        "human_model_summary.csv": matched_summary, "by_type.csv": by_type,
        "by_difficulty.csv": by_difficulty, "confidence_distribution.csv": confidence,
        "reliability.csv": reliability, "hallucination_breakdown.csv": hallucination,
        "hallucination_sdt.csv": sdt, "metad_summary.csv": metad_summary,
        "bootstrap_summary.csv": bootstrap,
    }
    for name, frame in frames.items():
        artifacts[name] = _write_csv(frame, output / name)
    artifacts["tables.md"] = _write_tables(output / "tables.md", model_summary, matched_summary, by_type, sdt, metad_summary)
    artifacts["analysis_report_zh.md"] = _write_report(output / "analysis_report_zh.md", quality, model_summary, matched_summary, by_type, by_difficulty, hallucination, sdt, metad_summary, bootstrap)
    figure_paths = _plot_artifacts(output / "figures", model_summary, matched_summary, by_type, metad_summary, reliability, hallucination, bootstrap)
    artifacts.update({f"figures/{name}": path for name, path in figure_paths.items()})
    ended = datetime.now(timezone.utc)
    manifest_path = output / "run_manifest.json"
    # Include the manifest itself in the artifact index so consumers can
    # enumerate every generated file from one machine-readable record.
    artifacts["run_manifest.json"] = manifest_path
    human_raw = pd.read_csv(human_path, low_memory=False)
    human_row_count = int(len(human_raw))
    artifact_index = {name: str(path.resolve()) for name, path in artifacts.items()}
    manifest = {
        "inputs": {
            "model": {"path": str(model_path), "sha256": _sha256(model_path), "row_count": int(len(all_model)), "valid_row_count": int(len(model))},
            "human": {"path": str(human_path), "sha256": _sha256(human_path), "row_count": human_row_count, "valid_row_count": int(len(human))},
        },
        "input_paths": {"model": str(model_path), "human": str(human_path)},
        "input_digests": {"model": _sha256(model_path), "human": _sha256(human_path)},
        "row_counts": {"model": int(len(all_model)), "human": human_row_count},
        "random_seed": int(seed),
        "bootstrap": {"nboot_metrics": int(nboot_metrics), "nboot_mratio": int(nboot_mratio)},
        "bootstrap_counts": {"metrics": int(nboot_metrics), "mratio": int(nboot_mratio)},
        "package_versions": _package_versions(), "started_at": started.isoformat(), "ended_at": ended.isoformat(),
        "start_timestamp": started.isoformat(), "end_timestamp": ended.isoformat(),
        "artifacts": artifact_index,
        "artifact_paths": artifact_index,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate isolated Chinese replication analysis artifacts")
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--human-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nboot-metrics", type=int, default=800)
    parser.add_argument("--nboot-mratio", type=int, default=250)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _safe_output_dir(args.output_dir)
    if args.validate_only:
        audit = validate_analysis_artifacts(args.output_dir, args.model_results, args.human_master)
        print(json.dumps(audit, ensure_ascii=False))
        print("ANALYSIS_AUDIT_OK")
        return 0
    paths = run_analysis(args.model_results, args.human_master, args.output_dir,
                         nboot_metrics=args.nboot_metrics, nboot_mratio=args.nboot_mratio,
                         seed=args.seed, resume=args.resume)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
