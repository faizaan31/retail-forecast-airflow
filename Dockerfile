FROM apache/airflow:3.1.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    "statsforecast==1.7.7" \
    "pandas>=2.1,<3" \
    "numpy>=1.26,<3" \
    "pyarrow>=14" \
    "joblib>=1.3" \
    "cloudpickle>=3" \
    "requests>=2.31"

ENV PYTHONPATH=/opt/airflow \
    NIXTLA_ID_AS_COL=1
