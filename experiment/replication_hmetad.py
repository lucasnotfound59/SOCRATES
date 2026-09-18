"""Resumable Bayesian HMeta-d analysis for the Chinese replication.

The sampler is deliberately kept behind :func:`run_group`.  The public
analysis pipeline can therefore run validation and aggregation without ever
starting MCMC, while a completed group is a small, independently replaceable
JSON artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd

if __package__:
    from .pytensor_compat import compatibility_status, configure_pytensor_compatibility
    # Package import for ``python -m experiment.replication_hmetad`` and tests.
    from .replication_analysis import (
        _model_sort_key, _package_versions, _sha256, evidence_tier, load_human_trials,
        load_model_attempts,
    )
else:  # pragma: no cover - exercised by direct script execution
    from pytensor_compat import compatibility_status, configure_pytensor_compatibility
    # The registered formal command invokes this file by path.  In that mode
    # Python puts ``experiment/`` on sys.path and relative imports are invalid.
    from replication_analysis import (
        _model_sort_key, _package_versions, _sha256, evidence_tier, load_human_trials,
        load_model_attempts,
    )

configure_pytensor_compatibility()

try:  # Keep import errors out of non-sampling utility use.
    from metadpy.bayesian import hmetad
except Exception:  # pragma: no cover - exercised only in incomplete environments
    hmetad = None


GROUP_DIRNAME = "hmetad/groups"
SUMMARY_NAME = "hmetad_summary.csv"


def _rhat(idata: az.InferenceData, variable: str) -> float:
    try:
        diagnostic = az.rhat(idata, var_names=[variable])
        values = np.asarray(diagnostic[variable].values, dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if len(finite) else float("nan")
    except Exception:
        return float("nan")


def _ratio_rhat(ratio: np.ndarray) -> float:
    """Compute R-hat on the chain/draw-preserving M-ratio samples."""
    try:
        ratio_idata = az.from_dict(posterior={"m_ratio": ratio})
        return _rhat(ratio_idata, "m_ratio")
    except Exception:
        return float("nan")


def _divergences(idata: az.InferenceData) -> tuple[int, float]:
    try:
        values = np.asarray(idata.sample_stats["diverging"].values, dtype=bool)
    except Exception:
        return -1, float("nan")
    count = int(values.sum())
    total = int(values.size)
    return count, (count / total if total else float("nan"))


def summarize_idata(idata: az.InferenceData, n_wrong: int = 0) -> dict[str, float | int | str | bool]:
    """Summarize posterior M-ratio and diagnostics without rerunning MCMC.

    ``metadpy`` has used both ``d1`` and ``d`` for the type-1 sensitivity
    variable.  Both are accepted; the ratio is always computed sample by
    sample before its mean, interval, and diagnostics are reported.
    """
    denominator = "d1" if "d1" in idata.posterior else "d"
    meta_d_array = np.asarray(idata.posterior["meta_d"].values, dtype=float)
    d_array = np.asarray(idata.posterior[denominator].values, dtype=float)
    if meta_d_array.shape != d_array.shape:
        raise ValueError("posterior meta_d and d/d1 have different sample counts")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_array = meta_d_array / d_array
    finite = ratio_array[np.isfinite(ratio_array)]
    if len(finite):
        mean = float(np.mean(finite))
        median = float(np.median(finite))
        sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        hdi = np.asarray(az.hdi(finite, hdi_prob=0.95), dtype=float).reshape(-1)
        hdi_lo, hdi_hi = float(hdi[0]), float(hdi[-1])
    else:
        mean = median = sd = hdi_lo = hdi_hi = float("nan")

    n_divergent, divergence_rate = _divergences(idata)
    posterior = idata.posterior
    chains = int(posterior.sizes.get("chain", 0))
    draws = int(posterior.sizes.get("draw", 0))
    rhat = _ratio_rhat(ratio_array)
    wrong = int(n_wrong)
    reliable = bool(
        wrong >= 10
        and np.isfinite(rhat)
        and rhat < 1.05
        and np.isfinite(divergence_rate)
        and divergence_rate < 0.05
        and np.isfinite(hdi_lo)
        and np.isfinite(hdi_hi)
    )
    return {
        "n_wrong": wrong,
        "error_count": wrong,
        "evidence_tier": evidence_tier(wrong),
        "m_ratio_mean": mean,
        "m_ratio_median": median,
        "m_ratio_sd": sd,
        "mr_mean": mean,
        "mr_median": median,
        "post_sd": sd,
        "hdi_lo": hdi_lo,
        "hdi_hi": hdi_hi,
        "hdi_width": hdi_hi - hdi_lo if np.isfinite(hdi_lo) and np.isfinite(hdi_hi) else float("nan"),
        "rhat": rhat,
        "n_divergent": n_divergent,
        "n_div": n_divergent,
        "divergence_rate": divergence_rate,
        "width": hdi_hi - hdi_lo if np.isfinite(hdi_lo) and np.isfinite(hdi_hi) else float("nan"),
        "draws": draws,
        "chains": chains,
        "reliable": reliable,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _group_model(group: pd.DataFrame) -> str:
    if "model" in group.columns and len(group):
        values = group["model"].dropna().astype(str).unique()
        if len(values):
            return values[0]
    return str(group.attrs.get("model", "unknown"))


def _n_wrong(group: pd.DataFrame) -> int:
    if "correct" not in group:
        raise ValueError("group is missing correct")
    return int((~group["correct"].map(_as_bool)).sum())


def _sampler_frame(group: pd.DataFrame) -> pd.DataFrame:
    required = {"gold", "correct"}
    missing = required.difference(group.columns)
    if missing:
        raise ValueError(f"group is missing required columns: {sorted(missing)}")
    confidence_column = "conf" if "conf" in group.columns else "confidence"
    if confidence_column not in group.columns:
        raise ValueError("group is missing conf")
    gold = group["gold"].map(_as_bool).astype(int)
    correct = group["correct"].map(_as_bool).astype(int)
    confidence = pd.to_numeric(group[confidence_column], errors="raise").astype(int)
    if not confidence.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("confidence must contain only ratings 1 through 5")
    return pd.DataFrame({"Stimuli": gold, "Accuracy": correct, "Confidence": confidence})


def run_group(
    group: pd.DataFrame,
    draws: int,
    chains: int,
    seed: int,
    sampler: Callable = hmetad,
    sampling_call: Callable | None = None,
    input_digests: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run one group and return a JSON-ready result record.

    Sampling failures are represented as ``status=failed`` so a later
    ``--resume`` can retry them.  Persistence is intentionally separate and
    handled by :func:`write_group_json`.
    """
    if draws <= 0 or chains <= 0:
        raise ValueError("draws and chains must be positive")
    started = time.perf_counter()
    model_error = None
    try:
        model = _group_model(group)
    except Exception as exc:
        model = "unknown"
        model_error = exc
    base: dict[str, object] = {
        "model": model,
        "status": "failed",
        "n": int(len(group)),
        "draws_requested": int(draws),
        "chains_requested": int(chains),
        "seed": int(seed),
    }
    if input_digests is not None:
        base["input_digests"] = dict(input_digests)
    if model_error is not None:
        base.update({
            "error_type": type(model_error).__name__,
            "error_message": str(model_error),
            "reliable": False,
            "runtime_s": float(time.perf_counter() - started),
            "runtime": float(time.perf_counter() - started),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        return base
    try:
        wrong = _n_wrong(group)
        base.update({"n_wrong": wrong, "error_count": wrong, "evidence_tier": evidence_tier(wrong)})
        if sampler is None:
            raise ImportError("metadpy.bayesian.hmetad is unavailable")
        sampler_kwargs = {
            "data": _sampler_frame(group), "nRatings": 5,
            "stimuli": "Stimuli", "accuracy": "Accuracy", "confidence": "Confidence",
            "num_samples": int(draws), "num_chains": int(chains), "output": "model",
        }
        if sampling_call is not None or sampler is hmetad:
            # metadpy currently accepts **kwargs but drops them when it calls
            # its internal PyMC sampler. Build the model without sampling,
            # then pass the seed directly to the actual sampling callable.
            built = sampler(**sampler_kwargs, sample_model=False)
            model_object = built[0] if isinstance(built, (tuple, list)) else built
            if sampling_call is None:
                import pymc as pm
                sampling_call = pm.sample
            sampled = sampling_call(
                model=model_object, draws=int(draws), chains=int(chains),
                random_seed=int(seed), return_inferencedata=True,
            )
        else:
            # Injected test samplers may already return an InferenceData. Keep
            # this path as the small public adapter seam.
            sampled = sampler(**sampler_kwargs, random_seed=int(seed))
        idata = sampled[1] if isinstance(sampled, (tuple, list)) else sampled
        diagnostics = summarize_idata(idata, n_wrong=wrong)
        base.update(diagnostics)
        base["status"] = "complete"
    except Exception as exc:
        base.update({
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "reliable": False,
        })
    base["runtime_s"] = float(time.perf_counter() - started)
    base["runtime"] = base["runtime_s"]
    base["completed_at"] = datetime.now(timezone.utc).isoformat()
    return base


def _safe_model_name(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model)).strip("._")
    return safe or "unknown"


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def group_json_path(group_dir: Path, model: str) -> Path:
    return Path(group_dir) / f"{_safe_model_name(model)}.json"


def write_group_json(
    group_dir: Path,
    result: Mapping[str, Any] | None = None,
    *,
    model: str | None = None,
    status: str | None = None,
    **fields: Any,
) -> Path:
    """Atomically replace one group JSON, flushing the temporary sibling first."""
    payload: dict[str, Any] = dict(result or {})
    if model is not None:
        payload["model"] = model
    if "model" not in payload:
        raise ValueError("group result requires model")
    if status is not None:
        payload["status"] = status
    payload.update(fields)
    directory = Path(group_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = group_json_path(directory, str(payload["model"]))
    serializable = _json_value(payload)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=f".{target.stem}.", suffix=".tmp", delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return target


def _read_group_json(group_dir: Path, model: str) -> dict[str, Any] | None:
    path = group_json_path(group_dir, model)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"model": model, "status": "failed", "error_type": "InvalidJSON"}
    if not isinstance(value, dict):
        return {"model": model, "status": "failed", "error_type": "InvalidJSON"}
    value.setdefault("model", model)
    return value


def pending_models(
    models: Sequence[str], group_dir: Path, *, draws: int | None = None,
    chains: int | None = None, seed: int | None = None,
    input_digests: Mapping[str, str] | None = None,
) -> list[str]:
    """Return models absent, failed, or complete under different run settings."""
    pending = []
    for model in models:
        result = _read_group_json(Path(group_dir), str(model))
        reusable = result is not None and result.get("status") == "complete"
        requested = {"draws_requested": draws, "chains_requested": chains, "seed": seed}
        for key, expected in requested.items():
            if expected is not None and (result is None or result.get(key) != expected):
                reusable = False
        if input_digests is not None and (result is None or result.get("input_digests") != dict(input_digests)):
            # Missing digests are deliberately unsafe: an old result is never
            # reused against an unverified input file.
            reusable = False
        if not reusable:
            pending.append(str(model))
    return pending


_SUMMARY_COLUMNS = [
    "model", "status", "n", "n_wrong", "error_count", "evidence_tier",
    "m_ratio_mean", "m_ratio_median", "m_ratio_sd", "mr_mean", "mr_median",
    "post_sd", "hdi_lo", "hdi_hi", "hdi_width", "rhat", "n_divergent",
    "divergence_rate", "draws", "chains", "seed", "reliable", "runtime_s",
    "error_type", "error_message",
]


def aggregate_results(
    models: Sequence[str],
    group_dir: Path,
    *,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> pd.DataFrame:
    """Combine complete, failed, and missing group records in requested order."""
    rows: list[dict[str, Any]] = []
    for model in sorted((str(model) for model in models), key=_model_sort_key):
        result = _read_group_json(Path(group_dir), str(model))
        if result is None:
            result = {"model": str(model), "status": "missing", "error_type": "MissingGroup"}
        rows.append(result)
    frame = pd.DataFrame(rows)
    for column in _SUMMARY_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame.reindex(columns=_SUMMARY_COLUMNS + [c for c in frame.columns if c not in _SUMMARY_COLUMNS])
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    if report_path is not None:
        update_report_bayesian(Path(report_path), frame)
    return frame


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.3f}"


def update_report_bayesian(path: Path, summary: pd.DataFrame) -> Path:
    """Replace only the Bayesian section of the Chinese report."""
    complete = summary.loc[summary["status"].eq("complete")] if len(summary) else summary
    failed = summary.loc[summary["status"].eq("failed")] if len(summary) else summary
    missing = summary.loc[summary["status"].eq("missing")] if len(summary) else summary
    lines = [
        "## Bayesian 状态",
        "",
        f"Bayesian HMeta-d 已处理 {len(complete)} 组；失败 {len(failed)} 组；缺失 {len(missing)} 组。",
        "证据等级仍只由错误数决定：`data-driven`（≥30）、`regularized`（10–29）、`prior-dominated`（<10）。",
        "可靠性另外要求错误数至少 10、R-hat < 1.05、发散比例 < 5%，且 95% HDI 边界有限；不满足时仅透明报告，不作强结论。",
    ]
    for row in complete.itertuples(index=False):
        lines.append(
            f"- `{row.model}`：M-ratio={_fmt(row.m_ratio_mean)}，95% HDI [{_fmt(row.hdi_lo)}, {_fmt(row.hdi_hi)}]，"
            f"R-hat={_fmt(row.rhat)}，发散={row.n_divergent if pd.notna(row.n_divergent) else 'NA'}，"
            f"evidence=`{row.evidence_tier}`，reliable=`{row.reliable}`。"
        )
    for row in failed.itertuples(index=False):
        message = getattr(row, "error_message", "") or getattr(row, "error_type", "")
        lines.append(f"- `{row.model}`：`failed`（{message}）；可通过 `--resume` 重试。")
    for row in missing.itertuples(index=False):
        lines.append(f"- `{row.model}`：`missing`；尚未生成组结果。")
    replacement = "\n".join(lines) + "\n\n"
    old = path.read_text(encoding="utf-8") if path.exists() else "# 中文复现分析报告\n\n"
    match = re.search(r"(?ms)^## Bayesian 状态\n.*?(?=^## |\Z)", old)
    if match:
        text = old[:match.start()] + replacement + old[match.end():]
    else:
        text = old.rstrip() + "\n\n" + replacement
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def refresh_run_manifest(
    analysis_dir: Path,
    *,
    draws: int,
    chains: int,
    seed: int,
) -> Path:
    """Extend the non-Bayesian manifest with every Bayesian output and hash."""
    analysis_dir = Path(analysis_dir)
    manifest_path = analysis_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing non-Bayesian manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: dict[str, str] = {}
    artifact_hashes: dict[str, str | None] = {}
    for path in sorted(p for p in analysis_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(analysis_dir).as_posix()
        artifacts[relative] = str(path.resolve())
        # The manifest cannot contain its own final hash without becoming
        # self-referential; all other generated files are content-hashed.
        artifact_hashes[relative] = None if relative == "run_manifest.json" else _sha256(path)
    manifest["artifacts"] = artifacts
    manifest["artifact_paths"] = artifacts
    manifest["artifact_hashes"] = artifact_hashes
    expected_order = ["human", *sorted(
        (p.stem for p in (analysis_dir / GROUP_DIRNAME).glob("*.json") if p.stem != "human"),
        key=_model_sort_key,
    )]
    manifest["bayesian"] = {
        "draws": int(draws), "chains": int(chains), "seed": int(seed),
        "groups": expected_order,
    }
    manifest["runtime_compatibility"] = compatibility_status()
    manifest["package_versions"] = _package_versions()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.data is not None:
        data = pd.read_csv(args.data, low_memory=False)
        if "model" not in data.columns:
            raise ValueError("--data must contain model")
        return data
    if args.model_results is None or args.human_master is None:
        raise ValueError("provide --data or both --model-results and --human-master")
    _, model = load_model_attempts(args.model_results)
    human = load_human_trials(args.human_master)
    return pd.concat([model, human], ignore_index=True, sort=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable Bayesian HMeta-d runner")
    parser.add_argument("--analysis-dir", "--output-dir", dest="analysis_dir", type=Path, default=Path("analysis"))
    parser.add_argument("--data", type=Path, help="validated combined CSV")
    parser.add_argument("--model-results", type=Path)
    parser.add_argument("--human-master", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    actions = int(args.model is not None) + int(args.all) + int(args.aggregate)
    if actions != 1:
        raise SystemExit("choose exactly one of --model, --all, or --aggregate")
    analysis_dir = Path(args.analysis_dir)
    group_dir = analysis_dir / GROUP_DIRNAME
    input_digests: dict[str, str] = {}
    if args.model_results is not None and args.human_master is not None:
        input_digests = {
            "model": _sha256(Path(args.model_results).expanduser().resolve()),
            "human": _sha256(Path(args.human_master).expanduser().resolve()),
        }
    elif args.data is not None:
        input_digests = {"data": _sha256(Path(args.data).expanduser().resolve())}
    data = None
    if args.model is not None or args.all:
        data = _load_data(args)
        models = sorted(data["model"].dropna().astype(str).drop_duplicates().tolist(), key=_model_sort_key)
        selected = [str(args.model)] if args.model is not None else models
        unknown = [model for model in selected if model not in set(models)]
        if unknown:
            raise SystemExit(f"unknown model(s): {', '.join(unknown)}")
        if args.resume:
            selected = [model for model in selected if model in pending_models(
                selected, group_dir, draws=args.draws, chains=args.chains, seed=args.seed,
                input_digests=input_digests,
            )]
        by_model = {model: data.loc[data["model"].astype(str).eq(model)].copy() for model in models}
        for model in selected:
            result = run_group(
                by_model[model], args.draws, args.chains, args.seed,
                input_digests=input_digests,
            )
            write_group_json(group_dir, result)
    elif args.data is not None or (args.model_results is not None and args.human_master is not None):
        data = _load_data(args)
        models = sorted(data["model"].dropna().astype(str).drop_duplicates().tolist(), key=_model_sort_key)
    else:
        summary_path = analysis_dir / "human_model_summary.csv"
        if not summary_path.exists():
            raise SystemExit("--aggregate needs --data/input paths or human_model_summary.csv")
        models = pd.read_csv(summary_path)["model"].astype(str).tolist()
    aggregate_results(
        models,
        group_dir,
        output_path=analysis_dir / SUMMARY_NAME,
        report_path=analysis_dir / "analysis_report_zh.md",
    )
    refresh_run_manifest(
        analysis_dir, draws=args.draws, chains=args.chains, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
