"""Call the forecast API using a series id from the latest promoted parquet, then tail the log."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from ml_pipeline import config

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

API_BASE = "http://forecast-api:8000"


def _call_inference_service() -> None:
    logging.basicConfig(level=logging.INFO)
    deployed = Path(config.DEPLOYED_FORECAST_PATH)
    if not deployed.is_file():
        raise FileNotFoundError(
            f"Missing {deployed}. Run retail_train_forecast before this DAG."
        )
    sample_id = str(pd.read_parquet(deployed, columns=["unique_id"])["unique_id"].iloc[0])
    url = f"{API_BASE}/predict"
    response = requests.post(
        url,
        json={"unique_id": sample_id, "horizon": 28},
        timeout=180,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Inference failed: %s", response.text)
        raise exc
    payload = response.json()
    logger.info("Inference OK: %s", payload.get("rows_returned"))


def _print_recent_request_log() -> None:
    logging.basicConfig(level=logging.INFO)
    log_path = Path(config.PREDICTION_LOG_PATH)
    if not log_path.is_file():
        logger.warning("No prediction log yet at %s", log_path)
        return
    tail = pd.read_parquet(log_path).tail(5)
    logger.info("Latest logged rows:\n%s", tail.to_string(index=False))


with DAG(
    dag_id="retail_batch_infer",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["retail", "inference"],
    doc_md=__doc__,
) as dag:
    infer = PythonOperator(
        task_id="http_predict_sample_series",
        python_callable=_call_inference_service,
    )
    tail_log = PythonOperator(
        task_id="tail_prediction_log",
        python_callable=_print_recent_request_log,
    )
    infer >> tail_log
