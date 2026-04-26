"""Central paths and tuning knobs used across DAGs, training, and the API."""

from pathlib import Path

# Container layout (matches docker-compose volume mounts)
AIRFLOW_HOME = Path("/opt/airflow")
DATA_ROOT = AIRFLOW_HOME / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
WORKING_DIR = DATA_ROOT / "working"

MODELS_DIR = AIRFLOW_HOME / "models"
MODEL_BUNDLE_PATH = MODELS_DIR / "statsforecast_bundle.joblib"
FORECAST_SNAPSHOT_PATH = MODELS_DIR / "latest_forecasts.parquet"
DEPLOYED_FORECAST_PATH = MODELS_DIR / "deployed" / "forecasts.parquet"
EVALUATION_REPORT_PATH = MODELS_DIR / "backtest_metrics.json"

API_LOG_DIR = AIRFLOW_HOME / "outputs"
PREDICTION_LOG_PATH = API_LOG_DIR / "prediction_requests.parquet"


def rebase_paths(airflow_home: Path) -> None:
    """Redirect all volume paths (used by tests and local scripts)."""

    global AIRFLOW_HOME, DATA_ROOT, RAW_DATA_DIR, PROCESSED_DIR, WORKING_DIR
    global MODELS_DIR, MODEL_BUNDLE_PATH, FORECAST_SNAPSHOT_PATH, DEPLOYED_FORECAST_PATH
    global EVALUATION_REPORT_PATH, API_LOG_DIR, PREDICTION_LOG_PATH

    home = Path(airflow_home)
    AIRFLOW_HOME = home
    DATA_ROOT = AIRFLOW_HOME / "data"
    RAW_DATA_DIR = DATA_ROOT / "raw"
    PROCESSED_DIR = DATA_ROOT / "processed"
    WORKING_DIR = DATA_ROOT / "working"
    MODELS_DIR = AIRFLOW_HOME / "models"
    MODEL_BUNDLE_PATH = MODELS_DIR / "statsforecast_bundle.joblib"
    FORECAST_SNAPSHOT_PATH = MODELS_DIR / "latest_forecasts.parquet"
    DEPLOYED_FORECAST_PATH = MODELS_DIR / "deployed" / "forecasts.parquet"
    EVALUATION_REPORT_PATH = MODELS_DIR / "backtest_metrics.json"
    API_LOG_DIR = AIRFLOW_HOME / "outputs"
    PREDICTION_LOG_PATH = API_LOG_DIR / "prediction_requests.parquet"

REQUIRED_NIXTLA_COLUMNS = frozenset({"unique_id", "ds", "y"})

# Preprocessing: cap distinct items so local runs stay fast (raise for full M5)
DEFAULT_ITEM_SAMPLE = 500
# Training: max rows passed to StatsForecast after per-series head()
DEFAULT_MAX_TRAINING_ROWS = 120_000
DEFAULT_PER_SERIES_HISTORY = 400
DEFAULT_FORECAST_HORIZON = 28
