from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_pipeline import config
from ml_pipeline.preprocessing import run_preprocess
from ml_pipeline.evaluation import write_evaluation_report
from ml_pipeline.training import (
    build_training_frame,
    fit_and_forecast,
    forecast_quality_snapshot,
    persist_artifacts,
)


def _seed_raw(tmp_raw: Path) -> None:
    days = 25
    calendar = pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, days + 1)],
            "date": pd.date_range("2024-01-01", periods=days, freq="D").strftime("%Y-%m-%d"),
        }
    )
    calendar.to_csv(tmp_raw / "calendar.csv", index=False)
    rows = []
    for sku in ("ITEM_A", "ITEM_B"):
        row = {
            "id": f"id_{sku}",
            "item_id": sku,
            "dept_id": "D1",
            "cat_id": "C1",
            "store_id": "S1",
            "state_id": "ST",
        }
        for d_idx in range(1, days + 1):
            row[f"d_{d_idx}"] = float((d_idx + hash(sku)) % 6)
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_raw / "sales_train_validation.csv", index=False)


def test_training_pipeline_smoke() -> None:
    raw = Path(config.RAW_DATA_DIR)
    _seed_raw(raw)
    run_preprocess(max_items=2, raw_dir=raw, output_dir=config.PROCESSED_DIR)

    panel = build_training_frame(per_series_rows=18, row_cap=500)
    assert not panel.empty

    forecasts, fitted = fit_and_forecast(horizon=4, training_frame=panel)
    assert not forecasts.empty
    snap = forecast_quality_snapshot(forecasts)
    assert snap["forecast_rows"] > 0

    meta = persist_artifacts(forecasts, fitted)
    assert Path(meta["model_path"]).is_file()
    assert config.FORECAST_SNAPSHOT_PATH.is_file()

    report_path = write_evaluation_report()
    assert report_path.is_file()
    assert "macro" in report_path.read_text(encoding="utf-8")
