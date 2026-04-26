"""Reusable ML and data utilities for the M5-style forecasting Airflow stack."""

import os

# StatsForecast 1.7+: keep series id as a column (avoids ambiguous index/column state on predict).
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
