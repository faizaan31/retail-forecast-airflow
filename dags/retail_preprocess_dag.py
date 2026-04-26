"""One-shot preprocessing: wide M5-style sales → Nixtla long parquet."""

from __future__ import annotations

import logging
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

logger = logging.getLogger(__name__)


def _run_preprocessing() -> None:
    from ml_pipeline.preprocessing import run_preprocess

    logging.basicConfig(level=logging.INFO)
    run_preprocess()


with DAG(
    dag_id="retail_preprocess",
    start_date=datetime(2024, 1, 1),
    schedule="@once",
    catchup=False,
    tags=["retail", "m5", "preprocess"],
    doc_md=__doc__,
) as dag:
    PythonOperator(
        task_id="nixtla_long_format",
        python_callable=_run_preprocessing,
    )
