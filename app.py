
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
    "Explore historical Niño 3.4 anomalies and view a "
    "one-month-ahead estimate from a Linear Regression model."
)
st.caption(
    "Experimental statistical forecast, not an official ENSO alert."
)

# Load and prepare the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)

    required = {"YR", "MON", "ANOM.3"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(sorted(missing))}. "
            f"Available columns: {list(data.columns)}"
        )

    data["YR"] = pd.to_numeric(data["YR"], errors="coerce")
    data["MON"] = pd.to_numeric(data["MON"], errors="coerce")
    data["ANOM.3"] = pd.to_numeric(data["ANOM.3"], errors="coerce")
    data = data.dropna(subset=["YR", "MON", "ANOM.3"]).copy()
    data["YR"] = data["YR"].astype(int)
    data["MON"] = data["MON"].astype(int)
    data = data[data["MON"].between(1, 12)]

    data["Date"] = pd.to_datetime(
        dict(year=data["YR"], month=data["MON"], day=1),
        errors="coerce",
    )
    data = data.dropna(subset=["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

if len(df) < 4:
    st.error("At least four monthly observations are required.")
    st.stop()

# Prepare the four most recent monthly values
latest = df.iloc[-1]
latest_four = df["ANOM.3"].tail(4).tolist()

features = pd.DataFrame(
    [[
        latest_four[3],
        latest_four[2],
        latest_four[1],
        latest_four[0],
    ]],
    columns=FEATURE_COLUMNS,
)

# Generate next-month forecast
try:
    forecast = float(model.predict(features)[0])
except Exception as exc:
    st.error(f"Could not generate the forecast: {exc}")
    st.stop()

latest_value = float(latest["ANOM.3"])
latest_date = latest["Date"]
forecast_date = latest_date + pd.offsets.MonthBegin(1)
change = forecast - latest_value

# Summary metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest observed anomaly",
    f"{latest_value:+.2f} °C",
)

col2.metric(
    f"Forecast for {forecast_date.strftime('%B %Y')}",
    f"{forecast:+.3f} °C",
    delta=f"{change:+.3f} °C vs latest observation",
)

col3.metric(
    "Latest data month",
    latest_date.strftime("%B %Y"),
)

st.divider()

# Historical chart
st.subheader("Historical Niño 3.4 anomalies")

years = sorted(df["YR"].unique().tolist())

if len(years) > 1:
    first_year = int(years[0])
    last_year = int(years[-1])
    start_default = max(first_year, last_year - 20)

    selected_years = st.slider(
        "Choose years to display",
        min_value=first_year,
        max_value=last_year,
        value=(start_default, last_year),
    )

    chart_df = df[
        df["YR"].between(selected_years[0], selected_years[1])
    ]
else:
    chart_df = df

fig, ax = plt.subplots(figsize=(12, 4.5))

ax.plot(
    chart_df["Date"],
    chart_df["ANOM.3"],
    linewidth=1.2,
    label="Observed anomaly",
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    label="Reference average",
)

ax.set_xlabel("Date")
ax.set_ylabel("Anomaly (°C)")
ax.set_title("Monthly Niño 3.4 temperature anomaly")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Forecast interpretation
st.subheader("Forecast interpretation")

if change > 0:
    st.write(
        f"The model predicts an increase of {change:.3f} °C "
        "compared with the latest observed anomaly."
    )
elif change < 0:
    st.write(
        f"The model predicts a decrease of {abs(change):.3f} °C "
        "compared with the latest observed anomaly."
    )
else:
    st.write("The model predicts no change.")

st.warning(
    "One forecast does not establish whether El Niño or La Niña "
    "conditions will occur. Official assessments also consider "
    "persistence and atmospheric indicators."
)

with st.expander("View latest 12 observations"):
    recent = df.tail(12)[["Date", "ANOM.3"]].copy()
    recent["Date"] = recent["Date"].dt.strftime("%B %Y")
    recent = recent.rename(
        columns={
            "Date": "Month",
            "ANOM.3": "Observed anomaly (°C)",
        }
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)

st.caption(
    "Forecast generated using the saved Linear Regression model "
    "and the four latest observations in the dataset."
)
