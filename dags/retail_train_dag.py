"""Train StatsForecast (SES), persist bundle for the API, promote parquet snapshot."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "start_date": datetime(2024, 1, 1),
}


def _materialize_training_panel() -> None:
    from ml_pipeline.training import build_training_frame

    logging.basicConfig(level=logging.INFO)
    build_training_frame()


def _fit_and_export() -> None:
    from ml_pipeline.training import fit_and_forecast, persist_artifacts

    logging.basicConfig(level=logging.INFO)
    forecasts, fitted = fit_and_forecast()
    persist_artifacts(forecasts, fitted)


def _log_forecast_profile() -> None:
    from ml_pipeline import config
    from ml_pipeline.training import forecast_quality_snapshot

    logging.basicConfig(level=logging.INFO)
    frame = pd.read_parquet(config.FORECAST_SNAPSHOT_PATH)
    stats = forecast_quality_snapshot(frame)
    logger.info("Forecast snapshot: %s", stats)


def _write_backtest_report() -> None:
    from ml_pipeline.evaluation import write_evaluation_report

    logging.basicConfig(level=logging.INFO)
    out = write_evaluation_report()
    logger.info("Backtest report path: %s", out)


def _promote_artifact() -> None:
    from ml_pipeline.training import promote_forecasts_to_deploy

    logging.basicConfig(level=logging.INFO)
    promote_forecasts_to_deploy()


with DAG(
    dag_id="retail_train_forecast",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["retail", "m5", "training"],
    doc_md=__doc__,
) as dag:
    build_panel = PythonOperator(
        task_id="build_training_panel",
        python_callable=_materialize_training_panel,
    )
    train = PythonOperator(
        task_id="fit_statsforecast",
        python_callable=_fit_and_export,
    )
    backtest = PythonOperator(
        task_id="holdout_backtest_metrics",
        python_callable=_write_backtest_report,
    )
    profile = PythonOperator(
        task_id="forecast_snapshot_stats",
        python_callable=_log_forecast_profile,
    )
    promote = PythonOperator(
        task_id="promote_forecast_file",
        python_callable=_promote_artifact,
    )

    build_panel >> train >> backtest >> profile >> promote
