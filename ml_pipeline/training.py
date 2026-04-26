"""Training, evaluation, and artifact export for the StatsForecast baseline."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SimpleExponentialSmoothingOptimized

from ml_pipeline import config

logger = logging.getLogger(__name__)


def _processed_parquet_path() -> Path:
    return config.PROCESSED_DIR / "m5_nixtla_long.parquet"


def _training_frame_path() -> Path:
    return config.WORKING_DIR / "training_panel.parquet"


def _validate_schema(frame: pd.DataFrame) -> None:
    missing = config.REQUIRED_NIXTLA_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Frame is missing required columns: {sorted(missing)}")


def build_training_frame(
    processed_path: Optional[Path] = None,
    per_series_rows: Optional[int] = None,
    row_cap: Optional[int] = None,
) -> pd.DataFrame:
    """Load processed parquet, trim history per series, enforce dtypes."""
    path = Path(processed_path) if processed_path is not None else _processed_parquet_path()
    if not path.is_file():
        raise FileNotFoundError(f"Run preprocessing first; missing {path}")

    frame = pd.read_parquet(path)
    _validate_schema(frame)

    cap = per_series_rows if per_series_rows is not None else config.DEFAULT_PER_SERIES_HISTORY
    trimmed = (
        frame.sort_values(["unique_id", "ds"])
        .groupby("unique_id", as_index=False, sort=False)
        .tail(cap)
    )

    max_rows = row_cap if row_cap is not None else config.DEFAULT_MAX_TRAINING_ROWS
    if len(trimmed) > max_rows:
        trimmed = trimmed.head(max_rows).copy()

    trimmed["ds"] = pd.to_datetime(trimmed["ds"])
    trimmed["y"] = pd.to_numeric(trimmed["y"], errors="coerce").fillna(0.0)

    config.WORKING_DIR.mkdir(parents=True, exist_ok=True)
    trimmed.to_parquet(_training_frame_path(), index=False)
    logger.info(
        "Training frame ready: rows=%s series=%s",
        f"{len(trimmed):,}",
        f"{trimmed['unique_id'].nunique():,}",
    )
    return trimmed


def fit_and_forecast(
    horizon: Optional[int] = None,
    training_frame: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, StatsForecast]:
    """Fit SES per series and return in-sample forecasts for the next *horizon* steps."""
    panel = (
        training_frame
        if training_frame is not None
        else pd.read_parquet(_training_frame_path())
    )
    _validate_schema(panel)

    h = horizon if horizon is not None else config.DEFAULT_FORECAST_HORIZON
    model = StatsForecast(
        models=[SimpleExponentialSmoothingOptimized()],
        freq="D",
        n_jobs=1,
    )
    model.fit(panel)
    forecasts = model.predict(h=h)
    forecasts = _normalize_forecast_frame(forecasts, panel)
    return forecasts, model


def _normalize_forecast_frame(forecasts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Ensure `unique_id` is a plain column and the index is not ambiguous."""

    frame = forecasts.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    if "unique_id" in frame.columns and frame.index.names and "unique_id" in frame.index.names:
        frame = frame.reset_index(drop=True)
    if "unique_id" in frame.columns:
        return frame

    series_ids = sorted(panel["unique_id"].drop_duplicates().astype(str).tolist())
    n_series = len(series_ids)
    if n_series == 0:
        return frame
    if len(frame) % n_series != 0:
        raise ValueError("Cannot infer unique_id mapping from forecast row count.")
    chunk = len(frame) // n_series
    frame.insert(0, "unique_id", np.repeat(np.array(series_ids, dtype=object), chunk))
    return frame


def forecast_quality_snapshot(forecasts: pd.DataFrame) -> dict[str, Any]:
    """Lightweight stats for Airflow logs (not a full backtest)."""
    if "unique_id" not in forecasts.columns:
        raise ValueError("forecast_quality_snapshot expects a unique_id column")
    cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
    summary: dict[str, Any] = {
        "forecast_rows": len(forecasts),
        "series_count": int(forecasts["unique_id"].nunique()),
        "horizon_steps": int(forecasts.groupby("unique_id").size().median()) if len(forecasts) else 0,
        "model_columns": cols,
    }
    if cols:
        summary["mean_yhat"] = float(pd.to_numeric(forecasts[cols[0]], errors="coerce").mean())
    return summary


def persist_artifacts(forecasts: pd.DataFrame, fitted: StatsForecast) -> dict[str, Any]:
    """Write model bundle + parquet snapshots to the shared models volume."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "forecaster": fitted,
        "fitted_at": pd.Timestamp.utcnow().isoformat(),
        "horizon_trained": config.DEFAULT_FORECAST_HORIZON,
    }
    joblib.dump(bundle, config.MODEL_BUNDLE_PATH)
    forecasts.to_parquet(config.FORECAST_SNAPSHOT_PATH, index=False)
    logger.info("Saved model bundle -> %s", config.MODEL_BUNDLE_PATH)
    logger.info("Saved forecasts -> %s", config.FORECAST_SNAPSHOT_PATH)
    return {"model_path": str(config.MODEL_BUNDLE_PATH), "rows": len(forecasts)}


def promote_forecasts_to_deploy() -> Path:
    """Copy the latest forecast snapshot to the deploy location consumed by downstream jobs."""
    if not config.FORECAST_SNAPSHOT_PATH.is_file():
        raise FileNotFoundError(f"No forecast snapshot at {config.FORECAST_SNAPSHOT_PATH}")
    config.DEPLOYED_FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.FORECAST_SNAPSHOT_PATH, config.DEPLOYED_FORECAST_PATH)
    logger.info("Promoted forecasts -> %s", config.DEPLOYED_FORECAST_PATH)
    return config.DEPLOYED_FORECAST_PATH
