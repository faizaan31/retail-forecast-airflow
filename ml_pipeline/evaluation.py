"""Hold-out backtesting and error metrics (MAPE, RMSE, sMAPE, volume-weighted RMSSE-style score)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SimpleExponentialSmoothingOptimized

from ml_pipeline import config

logger = logging.getLogger(__name__)


def mape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def naive_last_value_rmse(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Naive forecast: every step equals the last observed training value."""
    baseline = np.full(shape=y_test.shape, fill_value=float(y_train[-1]), dtype=float)
    return rmse(y_test.astype(float), baseline)


def holdout_per_series(
    series_df: pd.DataFrame,
    *,
    horizon: int,
    min_train: int,
) -> dict[str, Any] | None:
    df = series_df.sort_values("ds").reset_index(drop=True)
    if len(df) < min_train + horizon:
        return None

    train = df.iloc[:-horizon].copy()
    actual = df.iloc[-horizon:].copy()
    train_panel = train[["unique_id", "ds", "y"]]
    uid = str(train["unique_id"].iloc[0])

    model = StatsForecast(
        models=[SimpleExponentialSmoothingOptimized()],
        freq="D",
        n_jobs=1,
    )
    model.fit(train_panel)
    forecasts = model.predict(h=horizon)
    value_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
    if not value_cols:
        return None
    model_col = value_cols[0]
    merged = actual.merge(forecasts[["ds", model_col]], on="ds", how="inner")
    if len(merged) < horizon:
        return None

    y_true = merged["y"].to_numpy(dtype=float)
    y_hat = merged[model_col].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)

    model_rmse = rmse(y_true, y_hat)
    naive_err = naive_last_value_rmse(y_train, y_true)
    rmsse = model_rmse / max(naive_err, 1e-9)
    volume_weight = float(np.sum(np.abs(y_train))) + 1e-6

    return {
        "unique_id": uid,
        "mape_pct": mape_pct(y_true, y_hat),
        "rmse": model_rmse,
        "smape_pct": smape_pct(y_true, y_hat),
        "rmsse_vs_naive_last": float(rmsse),
        "volume_weight": volume_weight,
    }


def wrmsse_style_aggregate(rows: list[dict[str, Any]]) -> float:
    """M5-style idea: weight each series RMSSE by a volume proxy (sum of abs training demand)."""

    if not rows:
        return float("nan")
    numerator = sum(float(r["volume_weight"]) * float(r["rmsse_vs_naive_last"]) for r in rows)
    denominator = sum(float(r["volume_weight"]) for r in rows)
    return float(numerator / denominator) if denominator else float("nan")


def run_holdout_evaluation(
    panel: pd.DataFrame,
    *,
    horizon: int = 7,
    min_train_rows: int = 14,
    max_series: int = 150,
) -> dict[str, Any]:
    required = {"unique_id", "ds", "y"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel must contain columns {required}")

    working = panel.copy()
    working["ds"] = pd.to_datetime(working["ds"])
    working["y"] = pd.to_numeric(working["y"], errors="coerce").fillna(0.0)

    head_uids = working["unique_id"].drop_duplicates().head(max_series)
    working = working[working["unique_id"].isin(head_uids)]

    per_series: list[dict[str, Any]] = []
    for _, chunk in working.groupby("unique_id", sort=False):
        row = holdout_per_series(chunk, horizon=horizon, min_train=min_train_rows)
        if row is not None:
            per_series.append(row)

    macro = {
        "series_evaluated": len(per_series),
        "horizon": horizon,
        "mean_mape_pct": float(np.nanmean([r["mape_pct"] for r in per_series]))
        if per_series
        else float("nan"),
        "mean_rmse": float(np.nanmean([r["rmse"] for r in per_series])) if per_series else float("nan"),
        "mean_smape_pct": float(np.nanmean([r["smape_pct"] for r in per_series]))
        if per_series
        else float("nan"),
        "wrmsse_style_volume_weighted_rmsse": wrmsse_style_aggregate(per_series),
        "median_rmsse_vs_naive": float(np.nanmedian([r["rmsse_vs_naive_last"] for r in per_series]))
        if per_series
        else float("nan"),
    }
    return {"macro": macro, "per_series_sample": per_series[:20]}


def write_evaluation_report(panel_path: Path | None = None) -> Path:
    """Load the training panel parquet, run hold-out metrics, write JSON next to model artifacts."""

    path = panel_path if panel_path is not None else config.WORKING_DIR / "training_panel.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing training panel at {path}")
    panel = pd.read_parquet(path)
    report = run_holdout_evaluation(panel)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote backtest metrics -> %s", config.EVALUATION_REPORT_PATH)
    return config.EVALUATION_REPORT_PATH
