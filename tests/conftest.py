from __future__ import annotations

import pytest

from ml_pipeline import config


@pytest.fixture(autouse=True)
def isolated_airflow_layout(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every test gets its own fake /opt/airflow tree so parquet + joblib paths stay hermetic."""

    root = tmp_path_factory.mktemp("airflow_home")
    for relative in (
        "data/raw",
        "data/processed",
        "data/working",
        "models/deployed",
        "outputs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config.rebase_paths(root)
    yield
