from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_pipeline import config
from ml_pipeline.preprocessing import run_preprocess


def _write_minimal_m5_raw(raw_dir: Path) -> None:
    days = 20
    calendar = pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, days + 1)],
            "date": pd.date_range("2024-01-01", periods=days, freq="D").strftime("%Y-%m-%d"),
        }
    )
    calendar.to_csv(raw_dir / "calendar.csv", index=False)

    id_cols = {
        "id": ["SKU_1"],
        "item_id": ["ITEM_A"],
        "dept_id": ["DEPT_1"],
        "cat_id": ["CAT_1"],
        "store_id": ["TX_1"],
        "state_id": ["TX"],
    }
    row = {k: id_cols[k][0] for k in id_cols}
    for d_idx in range(1, days + 1):
        row[f"d_{d_idx}"] = float(d_idx % 5)
    pd.DataFrame([row]).to_csv(raw_dir / "sales_train_validation.csv", index=False)


def test_run_preprocess_emits_long_parquet() -> None:
    raw = config.RAW_DATA_DIR
    _write_minimal_m5_raw(raw)
    output = run_preprocess(max_items=1, raw_dir=raw, output_dir=config.PROCESSED_DIR)
    assert output.is_file()
    frame = pd.read_parquet(output)
    assert set(frame.columns) >= {"unique_id", "ds", "y"}
    assert frame["unique_id"].nunique() == 1
