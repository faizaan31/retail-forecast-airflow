"""FastAPI service that loads the trained StatsForecast bundle from the shared volume."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_pipeline import config

logger = logging.getLogger(__name__)

_forecaster: Any = None
_bundle_meta: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _forecaster, _bundle_meta
    path = config.MODEL_BUNDLE_PATH
    if not path.is_file():
        logger.warning("Model bundle missing at %s — /predict will return 503", path)
        _forecaster = None
        _bundle_meta = {}
    else:
        raw = joblib.load(path)
        _forecaster = raw.get("forecaster")
        _bundle_meta = {k: v for k, v in raw.items() if k != "forecaster"}
        logger.info("Loaded StatsForecast bundle from %s", path)
    config.API_LOG_DIR.mkdir(parents=True, exist_ok=True)
    yield
    _forecaster = None


app = FastAPI(
    title="Retail demand forecast API",
    version="1.0.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    unique_id: str = Field(..., min_length=1, description="Series key from the training panel")
    horizon: int = Field(28, ge=1, le=90, description="Days ahead to simulate")


@app.get("/health")
def health() -> dict[str, str]:
    status = "ok" if _forecaster is not None else "degraded"
    return {"status": status}


@app.post("/predict")
def predict(body: ForecastRequest) -> dict[str, Any]:
    if _forecaster is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Run the training DAG at least once.",
        )
    try:
        full = _forecaster.predict(h=body.horizon)
    except Exception as exc:  # noqa: BLE001 — surface training skew to callers
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    series = full[full["unique_id"] == body.unique_id].copy()
    if series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown unique_id '{body.unique_id}' for this model run.",
        )

    log_path = config.PREDICTION_LOG_PATH
    existing = pd.read_parquet(log_path) if log_path.is_file() else pd.DataFrame()
    combined = pd.concat([existing, series], ignore_index=True)
    combined.to_parquet(log_path, index=False)

    value_cols = [c for c in series.columns if c not in ("unique_id", "ds")]
    preview = series[["ds"] + value_cols].head(5).to_dict(orient="records")
    return {
        "status": "ok",
        "rows_returned": int(len(series)),
        "unique_id": body.unique_id,
        "logged_to": str(log_path),
        "sample": preview,
        "bundle_meta": _bundle_meta,
    }
