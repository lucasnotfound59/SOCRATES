"""Input contracts and privacy-safe normalization for the Chinese replication."""

from pathlib import Path

import pandas as pd


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
    out["correct"] = out["correct"].map(parse_bool) if "correct" in out else False
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
    if attempts[FORMAL_KEY].duplicated().any():
        raise ValueError("formal task keys must be unique task keys")
    if attempts["language"].astype(str).str.strip().str.lower().ne("zh").any():
        raise ValueError("model attempts must be Chinese rows")
    if attempts["model"].nunique() != EXPECTED_MODELS:
        raise ValueError(f"expected {EXPECTED_MODELS} models")
    counts = attempts.groupby("model", dropna=False).size()
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
