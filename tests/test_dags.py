from __future__ import annotations

import ast
from pathlib import Path


def test_dag_files_are_valid_python() -> None:
    root = Path(__file__).resolve().parents[1] / "dags"
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))


def test_dag_bag_loads_when_airflow_installed() -> None:
    import pytest

    try:
        from airflow.models.dagbag import DagBag
    except ImportError:
        pytest.skip("apache-airflow is not installed (syntax-only check runs in CI)")

    root = Path(__file__).resolve().parents[1]
    bag = DagBag(dag_folder=str(root / "dags"), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    assert bag.dags, "expected at least one DAG"
