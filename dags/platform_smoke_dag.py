"""Lightweight checks that Airflow can write under the shared data directory."""

from __future__ import annotations

import logging
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from ml_pipeline import config

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "start_date": datetime(2024, 1, 1),
}


def _write_smoke_marker() -> None:
    logging.basicConfig(level=logging.INFO)
    marker_dir = config.WORKING_DIR / "smoke_checks"
    marker_dir.mkdir(parents=True, exist_ok=True)
    path = marker_dir / "airflow_write_ok.txt"
    path.write_text("scheduler can write here\n", encoding="utf-8")
    logger.info("Wrote smoke marker -> %s", path)


with DAG(
    dag_id="platform_smoke_test",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["platform", "smoke"],
    doc_md=__doc__,
) as dag:
    PythonOperator(
        task_id="verify_data_volume",
        python_callable=_write_smoke_marker,
    )
