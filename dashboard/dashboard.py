"""Minimal Streamlit view over inference logs and promoted forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

LOG_PATH = Path("/opt/airflow/outputs/prediction_requests.parquet")
DEPLOYED_PATH = Path("/opt/airflow/models/deployed/forecasts.parquet")

st.set_page_config(page_title="Forecast monitor", layout="wide")
st.title("Retail demand — forecast monitor")
st.caption("Reads artifacts written by Airflow + the FastAPI inference service.")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Latest promoted forecasts")
    if DEPLOYED_PATH.is_file():
        deployed = pd.read_parquet(DEPLOYED_PATH)
        st.metric("Rows", len(deployed))
        st.dataframe(deployed.head(50), use_container_width=True)
    else:
        st.info("No deployed parquet yet. Run **retail_train_forecast** first.")

with col_b:
    st.subheader("API request log")
    if LOG_PATH.is_file():
        logged = pd.read_parquet(LOG_PATH)
        st.metric("Logged rows", len(logged))
        st.dataframe(logged.tail(50), use_container_width=True)
    else:
        st.info("No API calls logged yet. Run **retail_batch_infer** or POST to `/predict`.")
