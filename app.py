
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Niño 3.4 Experimental Anomaly Forecast",
    page_icon="🌊",
    layout="wide"
)
# ==========================================
# 2. LOAD DATA AND MODEL
# ==========================================
@st.cache_data
def load_data():
    data = pd.read_csv("nino_data.csv")

    required = ["YR", "MON", "ANOM.3"]
    if not all(col in data.columns for col in required):
        raise ValueError(
            "Dataset must contain YR, MON, and ANOM.3 columns."
        )

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required).copy()
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
    st.error(f"Unable to load the dataset or model: {error}")
    st.stop()


# ==========================================
# 3. PREPARE HISTORICAL MODEL FEATURES
# ==========================================
FEATURE_COLUMNS = [
    "ANOM_CURRENT",
    "ANOM_1_MONTH_AGO",
    "ANOM_2_MONTHS_AGO",
    "ANOM_3_MONTHS_AGO"
]

model_data = df[["DATE", "ANOM.3"]].copy()

model_data["ANOM_CURRENT"] = model_data["ANOM.3"]
model_data["ANOM_1_MONTH_AGO"] = model_data["ANOM.3"].shift(1)
model_data["ANOM_2_MONTHS_AGO"] = model_data["ANOM.3"].shift(2)
model_data["ANOM_3_MONTHS_AGO"] = model_data["ANOM.3"].shift(3)

model_data["TARGET"] = model_data["ANOM.3"].shift(-1)
model_data["TARGET_DATE"] = model_data["DATE"].shift(-1)
model_data = model_data.dropna().reset_index(drop=True)

X = model_data[FEATURE_COLUMNS]
y = model_data["TARGET"]

if len(model_data) < 10:
    st.error("There is not enough historical data for model evaluation.")
    st.stop()

split = int(len(model_data) * 0.8)

comparison_model = LinearRegression()
comparison_model.fit(X.iloc[:split], y.iloc[:split])

test_predictions = comparison_model.predict(X.iloc[split:])

comparison = pd.DataFrame({
    "Month": model_data["TARGET_DATE"].iloc[split:].values,
    "Actual Anomaly (°C)": y.iloc[split:].values,
    "Predicted Anomaly (°C)": test_predictions
})

comparison["Month"] = pd.to_datetime(comparison["Month"])


# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Niño 3.4 Explorer")
st.sidebar.write("Explore historical data and forecasts.")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Historical Trends",
        "Forecast",
        "Model Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Data source: NOAA")
st.sidebar.caption("For educational purposes")


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "Overview":

    st.title("Niño 3.4 experimental Anomaly Forecast
")

    st.write(
        "Explore monthly sea-surface temperature anomalies "
        "in the Niño 3.4 region of the central-eastern "
        "equatorial Pacific Ocean."
    )

    latest = df.iloc[-1]
    latest_date = latest["DATE"].strftime("%b %Y")
    latest_value = float(latest["ANOM.3"])

    latest_four = df["ANOM.3"].tail(4).tolist()

    input_values = pd.DataFrame(
        [[latest_four[3], latest_four[2],
          latest_four[1], latest_four[0]]],
        columns=FEATURE_COLUMNS
    )

    forecast_value = float(model.predict(input_values)[0])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Observed Anomaly",
        f"{latest_value:+.2f} °C",
        help=f"Latest observation in the dataset: {latest_date}"
    )

    col2.metric(
        "Next-Month Forecast",
        f"{forecast_value:+.2f} °C"
    )

    col3.metric(
        "Dataset Coverage",
        f"{int(df['YR'].min())}–{int(df['YR'].max())}"
    )

    st.divider()

    st.subheader("Historical Temperature Anomalies")

    st.write(
        "The chart below shows how the anomaly has changed "
        "over time. Use the year slider to explore a specific period."
    )

    min_year = int(df["YR"].min())
    max_year = int(df["YR"].max())

    selected_years = st.slider(
        "Select year range",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, max_year - 30), max_year),
        step=1,
        key="overview_years"
    )

    chart_df = df[
        df["YR"].between(selected_years[0], selected_years[1])
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        chart_df["DATE"],
        chart_df["ANOM.3"],
        linewidth=1.2,
        label="Observed anomaly"
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly (°C)")
    ax.set_title("Monthly Niño 3.4 Temperature Anomalies")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Positive values indicate warmer-than-average conditions; "
        "negative values indicate cooler-than-average conditions."
    )

    st.caption(
        "This experimental forecast is not an official El Niño "
        "or La Niña declaration."
    )


# ==========================================
# PAGE 2: HISTORICAL TRENDS
# ==========================================
elif page == "Historical Trends":

    st.title("Historical Trends")

    st.write(
        "Explore the monthly anomaly record and select the years "
        "you want to examine."
    )

    min_year = int(df["YR"].min())
    max_year = int(df["YR"].max())

    selected_years = st.slider(
        "Choose years to display",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
        key="historical_years"
    )

    filtered = df[
        df["YR"].between(selected_years[0], selected_years[1])
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        filtered["DATE"],
        filtered["ANOM.3"],
        linewidth=1.2
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly (°C)")
    ax.set_title(
        f"Niño 3.4 Anomalies: "
        f"{selected_years[0]}–{selected_years[1]}"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Monthly Observations")

    display_data = filtered[["DATE", "ANOM.3"]].copy()
    display_data["DATE"] = display_data["DATE"].dt.strftime("%b %Y")
    display_data = display_data.rename(
        columns={
            "DATE": "Month",
            "ANOM.3": "Actual Anomaly (°C)"
        }
    )

    st.dataframe(
        display_data.sort_values("Month", ascending=False).style.format(
            {"Actual Anomaly (°C)": "{:+.3f}"}
        ),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 3: MANUAL FORECAST
# ==========================================
elif page == "Forecast":

    st.title("Forecast the Next Month")

    st.write(
        "Enter four consecutive monthly anomalies, starting "
        "with the most recent month. The model will estimate "
        "the following month's anomaly."
    )

    latest_four = df["ANOM.3"].tail(4).tolist()

    if len(latest_four) < 4:
        st.error("At least four monthly observations are required.")
        st.stop()

    with st.form("forecast_form"):

        col1, col2 = st.columns(2)

        with col1:
            current = st.number_input(
                "Latest month (°C)",
                value=float(latest_four[3]),
                format="%.3f"
            )

            one_ago = st.number_input(
                "One month ago (°C)",
                value=float(latest_four[2]),
                format="%.3f"
            )

        with col2:
            two_ago = st.number_input(
                "Two months ago (°C)",
                value=float(latest_four[1]),
                format="%.3f"
            )

            three_ago = st.number_input(
                "Three months ago (°C)",
                value=float(latest_four[0]),
                format="%.3f"
            )

        submitted = st.form_submit_button(
            "Predict Next Month",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[current, one_ago, two_ago, three_ago]],
            columns=FEATURE_COLUMNS
        )

        try:
            prediction = float(model.predict(input_data)[0])
            change = prediction - current

            st.divider()
            st.subheader("Forecast Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Anomaly",
                f"{prediction:+.3f} °C"
            )

            col2.metric(
                "Change from Latest Month",
                f"{change:+.3f} °C"
            )

            if prediction > 0:
                st.write(
                    "The forecast is positive, indicating "
                    "warmer-than-average conditions."
                )
            elif prediction < 0:
                st.write(
                    "The forecast is negative, indicating "
                    "cooler-than-average conditions."
                )
            else:
                st.write(
                    "The forecast is approximately zero, indicating "
                    "conditions close to the reference average."
                )

            st.caption(
                "This is an experimental model estimate, not an "
                "official ENSO classification or warning."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ==========================================
# PAGE 4: MODEL EVALUATION
# ==========================================
elif page == "Model Evaluation":

    st.title("Actual vs Predicted Anomalies")

    st.write(
        "This section compares predictions with actual observations "
        "in the final 20% of the historical dataset. The comparison "
        "model is trained on earlier data."
    )

    actual = y.iloc[split:].to_numpy()
    predicted = test_predictions

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    col1, col2 = st.columns(2)

    col1.metric("Mean Absolute Error", f"{mae:.3f} °C")
    col2.metric("Root Mean Squared Error", f"{rmse:.3f} °C")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        comparison["Month"],
        comparison["Actual Anomaly (°C)"],
        label="Actual anomaly",
        linewidth=1.2
    )

    ax.plot(
        comparison["Month"],
        comparison["Predicted Anomaly (°C)"],
        label="Predicted anomaly",
        linewidth=1.2
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Anomaly (°C)")
    ax.set_title("Actual vs Predicted: Test Period")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Prediction Table")

    display_comparison = comparison.copy()
    display_comparison["Month"] = (
        display_comparison["Month"].dt.strftime("%b %Y")
    )

    st.dataframe(
        display_comparison.sort_values("Month", ascending=False).style.format({
            "Actual Anomaly (°C)": "{:+.3f}",
            "Predicted Anomaly (°C)": "{:+.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "MAE measures average absolute prediction error. "
        "RMSE penalizes larger errors more strongly. "
        "Lower values generally indicate better predictions."
    )
