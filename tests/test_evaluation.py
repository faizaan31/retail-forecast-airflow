from __future__ import annotations

import numpy as np
import pandas as pd

from ml_pipeline.evaluation import (
    mape_pct,
    rmse,
    run_holdout_evaluation,
    smape_pct,
    wrmsse_style_aggregate,
)


def test_mape_rmse_sanity() -> None:
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([10.0, 18.0, 33.0])
    assert mape_pct(y, p) >= 0.0
    assert rmse(y, p) > 0.0
    assert smape_pct(y, p) >= 0.0


def test_holdout_runs_on_synthetic_panel() -> None:
    rows = []
    rng = np.random.default_rng(0)
    for uid in ("A", "B"):
        for step in range(40):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": pd.Timestamp("2024-01-01") + pd.Timedelta(days=step),
                    "y": float(5 + rng.normal(0, 0.5) + step * 0.02),
                }
            )
    panel = pd.DataFrame(rows)
    report = run_holdout_evaluation(panel, horizon=5, min_train_rows=12, max_series=10)
    assert report["macro"]["series_evaluated"] >= 1
    assert not np.isnan(report["macro"]["mean_rmse"])


def test_wrmsse_aggregate_weights() -> None:
    rows = [
        {"volume_weight": 2.0, "rmsse_vs_naive_last": 1.0},
        {"volume_weight": 2.0, "rmsse_vs_naive_last": 2.0},
    ]
    assert abs(wrmsse_style_aggregate(rows) - 1.5) < 1e-9
