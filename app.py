import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Niño 3.4 Anomaly Forecast",
    page_icon="🌊",
    layout="wide",
)

DATA_FILE = "nino_data.csv"
MODEL_FILE = "nino_final_model.pkl"

FEATURE_COLUMNS = [
    "ANOM_CURRENT",
    "ANOM_1_MONTH_AGO",
    "ANOM_2_MONTHS_AGO",
    "ANOM_3_MONTHS_AGO",
]

st.title("Niño 3.4 Anomaly Forecast")
st.write(
    "Explore historical Niño 3.4 sea-surface temperature anomalies and view "
    "a one-month-ahead estimate from a Linear Regression model."
)
st.caption(
    "This is an experimental statistical forecast, not an official ENSO alert "
    "or a substitute for official climate monitoring."
)

# ----------------------------
# Load files
# ----------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    required_columns = {"YR", "MON", "ANOM.3"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    data = data.copy()
    data["YR"] = pd.to_numeric(data["YR"], errors="coerce")
    data["MON"] = pd.to_numeric(data["MON"], errors="coerce")
    data["ANOM.3"] = pd.to_numeric(data["ANOM.3"], errors="coerce")
    data = data.dropna(subset=["YR", "MON", "ANOM.3"])
    data["YR"] = data["YR"].astype(int)
    data["MON"] = data["MON"].astype(int)
    data = data[data["MON"].between(1, 12)]
    data["Date"] = pd.to_datetime(
        dict(year=data["YR"], month=data["MON"], day=1),
        errors="coerce",
    )
    data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return data


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    df = load_data(DATA_FILE)
    model = load_model(MODEL_FILE)
except FileNotFoundError as exc:
    st.error(
        f"Could not find `{exc.filename}`. Make sure both "
        "`nino34_data.csv` and `nino34_final_model.pkl` are in the same "
        "GitHub repository folder as `app.py`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the data or model: {exc}")
    st.stop()

if len(df) < 4:
    st.error("The dataset needs at least four valid monthly observations.")
    st.stop()

# ----------------------------
# Latest observation and forecast
# ----------------------------
latest = df.iloc[-1]
latest_four = df["ANOM.3"].tail(4).to_list()

# tail(4) is oldest-to-newest; the model expects current month first.
features = pd.DataFrame()
    [[latest_four[3], latest_four[2], latest_four[1], latest_four[0]]],
