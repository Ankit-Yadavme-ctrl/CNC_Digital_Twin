from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="CNC Digital Twin Dashboard",
    page_icon="CNC",
    layout="wide",
)

st_autorefresh(interval=2000, key="cnc_dashboard_refresh")

st.title("CNC Digital Twin - Cloud Dashboard")
st.caption("Layer 6 | AI-Based Status Classification")

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_FILE = BASE_DIR / "data" / "cnc_hivemq_data.csv"
MODEL_FILE = BASE_DIR / "ml" / "cnc_failure_model.pkl"

REQUIRED_COLUMNS = [
    "temperature",
    "vibration",
    "spindle_rpm",
    "motor_current",
    "tool_wear",
]

# Load the model safely.
model = None
model_loaded = False
if MODEL_FILE.exists():
    try:
        model = joblib.load(MODEL_FILE)
        model_loaded = True
    except Exception as exc:
        st.warning(f"AI model could not be loaded: {exc}")
else:
    st.info("AI model file is unavailable. Sensor monitoring can continue without AI predictions.")

# Prefer live MQTT data; fall back to CSV when MQTT has no readings.
df = pd.DataFrame()
data_source = "No data"
try:
    from dashboard.cloud_mqtt import get_live_data

    live_df = get_live_data()
    if live_df is not None and not live_df.empty:
        df = live_df.copy()
        data_source = "Live MQTT"
except Exception as exc:
    st.warning(f"Live MQTT is unavailable: {exc}")

if df.empty and CSV_FILE.exists():
    try:
        csv_df = pd.read_csv(CSV_FILE)
        if not csv_df.empty:
            df = csv_df.copy()
            data_source = "CSV fallback"
            st.info("Showing CSV data because live MQTT readings are not currently available.")
    except Exception as exc:
        st.warning(f"Could not read the fallback CSV file: {exc}")

if df.empty:
    st.warning("Waiting for CNC sensor readings. Check the MQTT connection and publisher.")
    st.caption(f"Data source: {data_source}")
    st.stop()

missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing:
    st.error("Sensor data is missing required fields: " + ", ".join(missing))
    st.stop()

for column in REQUIRED_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")
df = df.dropna(subset=REQUIRED_COLUMNS).copy()

if df.empty:
    st.error("Sensor readings were received, but required sensor values are not numeric.")
    st.stop()

latest = df.iloc[-1]
machine_id = str(latest.get("machine_id", "CNC_01"))
timestamp = str(latest.get("timestamp", "Timestamp unavailable"))
status = str(latest.get("status", "UNKNOWN")).upper()

st.write(f"**Machine:** {machine_id}")
st.caption(f"Last sensor reading: {timestamp} | Data source: {data_source}")

if status == "NORMAL":
    st.success("MACHINE STATUS: NORMAL")
elif status == "WARNING":
    st.warning("MACHINE STATUS: WARNING")
elif status == "CRITICAL":
    st.error("MACHINE STATUS: CRITICAL")
else:
    st.info(f"MACHINE STATUS: {status}")

st.subheader("Live Machine Parameters")
temperature = float(latest["temperature"])
vibration = float(latest["vibration"])
spindle_rpm = float(latest["spindle_rpm"])
motor_current = float(latest["motor_current"])
tool_wear = float(latest["tool_wear"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Temperature", f"{temperature:.2f} deg C")
c2.metric("Vibration", f"{vibration:.3f} g")
c3.metric("Spindle RPM", f"{spindle_rpm:.0f} RPM")
c4.metric("Motor Current", f"{motor_current:.2f} A")
c5.metric("Tool Wear", f"{tool_wear:.2f} %")

st.subheader("AI-Based Status Classification")
if model_loaded:
    input_data = pd.DataFrame(
        [[temperature, vibration, spindle_rpm, motor_current, tool_wear]],
        columns=REQUIRED_COLUMNS,
    )
    try:
        prediction = str(model.predict(input_data)[0]).upper()
        if prediction in {"0", "NORMAL"}:
            ai_status = "NORMAL"
        elif prediction in {"1", "WARNING"}:
            ai_status = "WARNING"
        else:
            ai_status = prediction

        normal_probability = None
        warning_probability = None
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            probabilities = model.predict_proba(input_data)[0]
            for label, probability in zip(model.classes_, probabilities):
                label = str(label).upper()
                if label in {"0", "NORMAL"}:
                    normal_probability = float(probability) * 100
                elif label in {"1", "WARNING"}:
                    warning_probability = float(probability) * 100

        a1, a2, a3 = st.columns(3)
        a1.metric("AI Status", ai_status)
        a2.metric(
            "Normal Probability",
            f"{normal_probability:.1f}%" if normal_probability is not None else "N/A",
        )
        a3.metric(
            "Warning Probability",
            f"{warning_probability:.1f}%" if warning_probability is not None else "N/A",
        )

        if ai_status == "NORMAL":
            st.success("AI alert: The model classifies the current reading as NORMAL.")
        elif ai_status == "WARNING":
            st.warning(
                "AI alert: The model classifies the current reading as WARNING. "
                "Review the sensor readings and inspect the machine if appropriate."
            )
        else:
            st.info(f"AI model output: {ai_status}")

        st.caption(
            "Model probabilities are model outputs, not validated real-world failure probabilities."
        )
    except Exception as exc:
        st.error(f"AI prediction failed: {exc}")
else:
    st.info("AI prediction is disabled because the trained model is unavailable.")

st.subheader("Machine Parameter History")
chart_data = df.tail(100).copy()
if "timestamp" in chart_data.columns:
    parsed_timestamp = pd.to_datetime(chart_data["timestamp"], errors="coerce")
    if parsed_timestamp.notna().any():
        chart_data["timestamp"] = parsed_timestamp
        chart_data = chart_data.dropna(subset=["timestamp"])
        chart_data = chart_data.set_index("timestamp")

chart_columns = [column for column in REQUIRED_COLUMNS if column in chart_data.columns]
if not chart_data.empty and chart_columns:
    st.line_chart(chart_data[chart_columns], use_container_width=True)
else:
    st.info("Not enough valid history is available to draw a chart yet.")

st.subheader("Recent Sensor Data")
st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
st.caption("Dashboard refresh interval: 2 seconds.")
