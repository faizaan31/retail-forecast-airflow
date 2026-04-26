"""Turn wide M5-style sales tables into long-format data for StatsForecast / Nixtla."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_pipeline import config

logger = logging.getLogger(__name__)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected dataset file at {path}. "
            "Add Kaggle M5 CSVs under data/raw/ or run: python -m ml_pipeline.sample_data"
        )


def run_preprocess(
    max_items: Optional[int] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Load sales + calendar, melt daily columns to long form, emit Nixtla schema parquet.

    Parameters
    ----------
    max_items
        Limit distinct item_id values for faster iteration (None = all items).
    """
    raw = Path(raw_dir) if raw_dir is not None else config.RAW_DATA_DIR
    out_dir = Path(output_dir) if output_dir is not None else config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sales_path = raw / "sales_train_validation.csv"
    calendar_path = raw / "calendar.csv"
    _require_file(sales_path)
    _require_file(calendar_path)

    item_cap = max_items if max_items is not None else config.DEFAULT_ITEM_SAMPLE

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)

    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [c for c in sales.columns if c not in id_columns]

    if item_cap:
        chosen_items = sales["item_id"].drop_duplicates().head(item_cap)
        sales = sales[sales["item_id"].isin(chosen_items)].copy()

    long = sales.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="y",
    )

    cal_subset = calendar[["d", "date"]].rename(columns={"date": "ds"})
    long = long.merge(cal_subset, on="d", how="left").drop(columns=["d"])

    long = long.rename(columns={"item_id": "unique_id"})
    long["ds"] = pd.to_datetime(long["ds"], errors="coerce")
    long["y"] = pd.to_numeric(long["y"], errors="coerce").fillna(0.0)

    nixtla_ready = long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    output_path = out_dir / "m5_nixtla_long.parquet"
    nixtla_ready.to_parquet(output_path, index=False)

    series_count = nixtla_ready["unique_id"].nunique()
    logger.info(
        "Preprocessing finished: rows=%s series=%s -> %s",
        f"{len(nixtla_ready):,}",
        f"{series_count:,}",
        output_path,
    )
    return output_path
