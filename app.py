
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Niño 3.4 Forecasting App",
    page_icon="🌊",
    layout="wide"
)

st.title("Niño 3.4 Forecasting Dashboard")

st.write(
    "Explore historical Niño 3.4 anomalies and forecast "
    "the following month's anomaly."
)

# -----------------------------
# Load data and saved model
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("nino_data.csv")

    required_columns = ["YR", "MON", "ANOM.3"]

    if not all(col in data.columns for col in required_columns):
        raise ValueError(
            "The CSV must contain YR, MON, and ANOM.3 columns."
        )

    data["YR"] = pd.to_numeric(data["YR"], errors="coerce")
    data["MON"] = pd.to_numeric(data["MON"], errors="coerce")
    data["ANOM.3"] = pd.to_numeric(
        data["ANOM.3"], errors="coerce"
    )

    data = data.dropna(subset=required_columns).copy()

    data["DATE"] = pd.to_datetime(
        {
            "year": data["YR"].astype(int),
            "month": data["MON"].astype(int),
            "day": 1
        },
        errors="coerce"
    )

    data = data.dropna(subset=["DATE"])
    data = data.sort_values("DATE").reset_index(drop=True)

    return data


@st.cache_resource
def load_model():
    return joblib.load("nino_final_model.pkl")


try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error(f"Could not load the data or model: {error}")
    st.stop()


# -----------------------------
# Prepare model features
# -----------------------------
FEATURE_COLUMNS = [
    "ANOM_CURRENT",
    "ANOM_1_MONTH_AGO",
    "ANOM_2_MONTHS_AGO",
    "ANOM_3_MONTHS_AGO"
]

df_model = df[["DATE", "ANOM.3"]].copy()

df_model["ANOM_CURRENT"] = df_model["ANOM.3"]
df_model["ANOM_1_MONTH_AGO"] = df_model["ANOM.3"].shift(1)
df_model["ANOM_2_MONTHS_AGO"] = df_model["ANOM.3"].shift(2)
df_model["ANOM_3_MONTHS_AGO"] = df_model["ANOM.3"].shift(3)

df_model["TARGET_NEXT_MONTH"] = df_model["ANOM.3"].shift(-1)
df_model["TARGET_DATE"] = df_model["DATE"].shift(-1)

df_model = df_model.dropna().reset_index(drop=True)

X = df_model[FEATURE_COLUMNS]
y = df_model["TARGET_NEXT_MONTH"]


# -----------------------------
# Historical actual vs predicted
# -----------------------------
st.header("Historical Actual vs Predicted Anomalies")

st.write(
    "This table compares actual anomalies with predictions "
    "for the final 20% of the dataset. The comparison model "
    "is trained on earlier months only."
)

split_index = int(len(df_model) * 0.8)

if split_index < 1 or split_index >= len(df_model):
    st.error("There is not enough data to create the comparison.")
    st.stop()

comparison_model = LinearRegression()
comparison_model.fit(
    X.iloc[:split_index],
    y.iloc[:split_index]
)

historical_predictions = comparison_model.predict(
    X.iloc[split_index:]
)

comparison = pd.DataFrame({
    "Month": df_model["TARGET_DATE"].iloc[split_index:].values,
    "Actual Anomaly (°C)": y.iloc[split_index:].values,
    "Predicted Anomaly (°C)": historical_predictions
})

comparison["Month"] = pd.to_datetime(comparison["Month"])

comparison = comparison.sort_values("Month", ascending=False)

st.dataframe(
    comparison.style.format({
        "Actual Anomaly (°C)": "{:+.3f}",
        "Predicted Anomaly (°C)": "{:+.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Historical comparison chart
chart_data = comparison.sort_values("Month").set_index("Month")

st.line_chart(
    chart_data[
        ["Actual Anomaly (°C)", "Predicted Anomaly (°C)"]
    ]
)


# -----------------------------
# Manual next-month forecast
# -----------------------------
st.header("Predict the Next Month")

st.write(
    "Enter the latest four monthly Niño 3.4 anomaly values, "
    "starting with the most recent month."
)

latest_four = df["ANOM.3"].tail(4).tolist()

if len(latest_four) < 4:
    st.error("At least four months of data are needed.")
    st.stop()

with st.form("forecast_form"):
    col1, col2 = st.columns(2)

    with col1:
        current_value = st.number_input(
            "Latest month's anomaly (°C)",
            value=float(latest_four[3]),
            format="%.3f"
        )

        one_month_ago = st.number_input(
            "One month ago (°C)",
            value=float(latest_four[2]),
            format="%.3f"
        )

    with col2:
        two_months_ago = st.number_input(
            "Two months ago (°C)",
            value=float(latest_four[1]),
            format="%.3f"
        )

        three_months_ago = st.number_input(
            "Three months ago (°C)",
            value=float(latest_four[0]),
            format="%.3f"
        )

    submitted = st.form_submit_button("Predict Next Month")

if submitted:
    input_data = pd.DataFrame(
        [[
            current_value,
            one_month_ago,
            two_months_ago,
            three_months_ago
        ]],
        columns=FEATURE_COLUMNS
    )

    try:
        forecast = float(model.predict(input_data)[0])

        st.subheader("Forecast Result")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Predicted Anomaly",
                f"{forecast:+.3f} °C"
            )

        with metric_col2:
            change = forecast - current_value
            st.metric(
                "Change from Latest Month",
                f"{change:+.3f} °C"
            )

        if forecast > 0:
            st.write(
                "The predicted anomaly is positive, indicating "
                "warmer-than-average conditions."
            )
        elif forecast < 0:
            st.write(
                "The predicted anomaly is negative, indicating "
                "cooler-than-average conditions."
            )
        else:
            st.write(
                "The predicted anomaly is approximately zero, "
                "indicating conditions close to the reference average."
            )

        st.caption(
            "This is a model estimate, not an official ENSO classification."
        )

    except Exception as error:
        st.error(f"Could not generate the forecast: {error}")


# Latest observations

st.header("Recent Observations")

recent_data = df[["DATE", "ANOM.3"]].tail(12).copy()
recent_data = recent_data.rename(
    columns={
        "DATE": "Month",
        "ANOM.3": "Actual Anomaly (°C)"
    }
)

st.dataframe(
    recent_data.sort_values("Month", ascending=False).style.format({
        "Actual Anomaly (°C)": "{:+.3f}"
    }),
    use_container_width=True,
    hide_index=True
)
