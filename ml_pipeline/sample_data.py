"""
Emit tiny synthetic M5-shaped CSVs under data/raw/ for a zero-download smoke test.

Run on the host before `docker compose up`:

    python -m ml_pipeline.sample_data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw"


def write_minimal_dataset() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    days = 30
    calendar = pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, days + 1)],
            "date": pd.date_range("2024-01-01", periods=days, freq="D").strftime("%Y-%m-%d"),
        }
    )
    calendar.to_csv(RAW / "calendar.csv", index=False)

    id_cols = {
        "id": ["SKU_1", "SKU_2"],
        "item_id": ["ITEM_A", "ITEM_B"],
        "dept_id": ["DEPT_1", "DEPT_1"],
        "cat_id": ["CAT_1", "CAT_1"],
        "store_id": ["TX_1", "TX_1"],
        "state_id": ["TX", "TX"],
    }
    rows = []
    for i in range(2):
        row = {k: id_cols[k][i] for k in id_cols}
        for d_idx in range(1, days + 1):
            row[f"d_{d_idx}"] = max(0, (d_idx % 7) + i * 2)
        rows.append(row)

    sales = pd.DataFrame(rows)
    sales.to_csv(RAW / "sales_train_validation.csv", index=False)
    print(f"Wrote sample M5-shaped files to {RAW}")


if __name__ == "__main__":
    write_minimal_dataset()
