from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from ml_pipeline import config
from ml_pipeline.training import build_training_frame, fit_and_forecast, persist_artifacts


def _bootstrap_artifacts() -> str:
    raw = Path(config.RAW_DATA_DIR)
    days = 22
    calendar = pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, days + 1)],
            "date": pd.date_range("2024-01-01", periods=days, freq="D").strftime("%Y-%m-%d"),
        }
    )
    calendar.to_csv(raw / "calendar.csv", index=False)
    row = {
        "id": "SKU_1",
        "item_id": "ITEM_API",
        "dept_id": "D1",
        "cat_id": "C1",
        "store_id": "S1",
        "state_id": "ST",
    }
    for d_idx in range(1, days + 1):
        row[f"d_{d_idx}"] = float(d_idx % 4)
    pd.DataFrame([row]).to_csv(raw / "sales_train_validation.csv", index=False)

    from ml_pipeline.preprocessing import run_preprocess

    run_preprocess(max_items=1, raw_dir=raw, output_dir=config.PROCESSED_DIR)
    panel = build_training_frame(per_series_rows=20, row_cap=500)
    forecasts, fitted = fit_and_forecast(horizon=5, training_frame=panel)
    persist_artifacts(forecasts, fitted)
    return str(panel["unique_id"].iloc[0])


def test_health_and_predict() -> None:
    uid = _bootstrap_artifacts()
    from ml_pipeline.api.app import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post("/predict", json={"unique_id": uid, "horizon": 5})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["unique_id"] == uid


def test_predict_unknown_series() -> None:
    _bootstrap_artifacts()
    from ml_pipeline.api.app import app

    with TestClient(app) as client:
        response = client.post("/predict", json={"unique_id": "does-not-exist", "horizon": 3})
        assert response.status_code == 404
