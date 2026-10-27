from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CNC Digital Twin Dashboard",
    page_icon="⚙️",
    layout="wide",
)

st_autorefresh(interval=2000, key="cnc_dashboard_refresh")

# Custom styling: polished cards, status badges, and spacing.
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.2rem;
        max-width: 1500px;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43 0%, #243b53 65%, #334e68 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.15rem;
        margin: 0 0 .25rem 0;
    }
    .hero p {
        color: #d9e2ec;
        margin: 0;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .22);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, .035);
        min-height: 120px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: .95rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .section-note {
        color: var(--text-color);
        opacity: .72;
        font-size: .9rem;
        margin-top: -.45rem;
        margin-bottom: .8rem;
    }
    .status-pill {
        padding: .8rem 1rem;
        border-radius: 12px;
        font-weight: 650;
        border: 1px solid rgba(128, 128, 128, .2);
        margin: .4rem 0 1rem 0;
    }
    .status-normal {
        background: rgba(22, 163, 74, .12);
        color: #15803d;
    }
    .status-warning {
        background: rgba(217, 119, 6, .13);
        color: #b45309;
    }
    .status-critical {
        background: rgba(220, 38, 38, .12);
        color: #b91c1c;
    }
    .ai-panel {
        border: 1px solid rgba(128, 128, 128, .25);
        border-radius: 16px;
        padding: 1rem 1.1rem .6rem 1.1rem;
        margin-bottom: .75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>⚙️ CNC Digital Twin</h1>
        <p>AI-driven smart manufacturing · Live machine monitoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# FILE PATHS AND SENSOR CONFIGURATION
# --------------------------------------------------
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

# --------------------------------------------------
# LOAD AI MODEL
# --------------------------------------------------
model = None
model_loaded = False

if MODEL_FILE.exists():
    try:
        model = joblib.load(MODEL_FILE)
        model_loaded = True
    except Exception as exc:
        st.warning(f"AI model could not be loaded: {exc}")
else:
    st.info(
        "AI model is unavailable. Sensor monitoring can continue "
        "without AI predictions."
    )

# --------------------------------------------------
# GET LIVE MQTT DATA; FALL BACK TO CSV IF NECESSARY
# --------------------------------------------------
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
            st.info(
                "Live MQTT readings are unavailable. "
                "Displaying locally stored CSV data."
            )
    except Exception as exc:
        st.warning(f"Could not read the fallback CSV file: {exc}")

if df.empty:
    st.warning(
        "Waiting for CNC sensor readings. "
        "Check the MQTT connection and sensor publisher."
    )
    st.caption(f"Data source: {data_source}")
    st.stop()

# --------------------------------------------------
# VALIDATE SENSOR DATA
# --------------------------------------------------
missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing:
    st.error("Sensor data is missing required fields: " + ", ".join(missing))
    st.stop()

for column in REQUIRED_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=REQUIRED_COLUMNS).copy()
if df.empty:
    st.error(
        "Sensor readings were received, but the required "
        "sensor values are not numeric."
    )
    st.stop()

# --------------------------------------------------
# LATEST SENSOR READING
# --------------------------------------------------
latest = df.iloc[-1]
machine_id = str(latest.get("machine_id", "CNC_01"))
timestamp = str(latest.get("timestamp", "Timestamp unavailable"))
sensor_status = str(latest.get("status", "UNKNOWN")).upper()

temperature = float(latest["temperature"])
vibration = float(latest["vibration"])
spindle_rpm = float(latest["spindle_rpm"])
motor_current = float(latest["motor_current"])
tool_wear = float(latest["tool_wear"])

# --------------------------------------------------
# MACHINE INFORMATION
# --------------------------------------------------
meta1, meta2, meta3 = st.columns([1, 1, 2])
with meta1:
    st.markdown(f"**Machine ID**  \n`{machine_id}`")
with meta2:
    st.markdown(f"**Data source**  \n{data_source}")
with meta3:
    st.markdown(f"**Last sensor reading**  \n{timestamp}")

st.divider()

# --------------------------------------------------
# SENSOR-REPORTED MACHINE STATUS
# --------------------------------------------------
st.subheader("Machine Health")

status_classes = {
    "NORMAL": ("status-normal", "🟢 SENSOR STATUS: NORMAL"),
    "WARNING": ("status-warning", "🟠 SENSOR STATUS: WARNING"),
    "CRITICAL": ("status-critical", "🔴 SENSOR STATUS: CRITICAL"),
}
status_class, status_text = status_classes.get(
    sensor_status,
    ("", f"Sensor-reported status: {sensor_status}"),
)
st.markdown(
    f'<div class="status-pill {status_class}">{status_text}</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LIVE SENSOR METRICS
# --------------------------------------------------
st.subheader("Live Machine Parameters")
st.markdown(
    '<div class="section-note">Latest available values from the sensor stream</div>',
    unsafe_allow_html=True,
)

row1 = st.columns(3)
row1[0].metric("🌡️ Temperature", f"{temperature:.1f} °C")
row1[1].metric("〰️ Vibration", f"{vibration:.3f} g")
row1[2].metric("⚙️ Spindle Speed", f"{spindle_rpm:,.0f} RPM")

row2 = st.columns(2)
row2[0].metric("🔌 Motor Current", f"{motor_current:.2f} A")
row2[1].metric("🛠️ Tool Wear", f"{tool_wear:.2f} %")

st.divider()

# --------------------------------------------------
# AI-BASED STATUS CLASSIFICATION
# --------------------------------------------------
st.subheader("AI Health Assessment")
st.markdown(
    '<div class="section-note">Model classification based on the latest sensor reading</div>',
    unsafe_allow_html=True,
)

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
                score = float(probability) * 100
                if label in {"0", "NORMAL"}:
                    normal_probability = score
                elif label in {"1", "WARNING"}:
                    warning_probability = score

        panel_class = (
            "status-normal" if ai_status == "NORMAL"
            else "status-warning" if ai_status == "WARNING"
            else ""
        )
        st.markdown('<div class="ai-panel">', unsafe_allow_html=True)

        ai1, ai2, ai3 = st.columns(3)
        with ai1:
            st.metric("AI Predicted Status", ai_status)
        with ai2:
            st.metric(
                "Normal Score",
                f"{normal_probability:.1f}%" if normal_probability is not None else "N/A",
            )
        with ai3:
            st.metric(
                "Warning Score",
                f"{warning_probability:.1f}%" if warning_probability is not None else "N/A",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if normal_probability is not None:
            st.caption("Normal classification score")
            st.progress(max(0.0, min(1.0, normal_probability / 100)))
        if warning_probability is not None:
            st.caption("Warning classification score")
            st.progress(max(0.0, min(1.0, warning_probability / 100)))

        if ai_status == "NORMAL":
            st.success("AI assessment: the model classifies the current reading as NORMAL.")
        elif ai_status == "WARNING":
            st.warning(
                "AI alert: the model classifies the current reading as WARNING. "
                "Review the readings and inspect the machine if appropriate."
            )
        else:
            st.info(f"AI model output: {ai_status}")

        st.caption(
            "Important: these are classification scores from a model trained on "
            "synthetic data. They are not validated real-world failure probabilities. "
            "Sensor-reported status and AI-predicted status may differ because they "
            "can use different classification rules."
        )

    except Exception as exc:
        st.error(f"AI prediction failed: {exc}")
else:
    st.info("AI prediction is disabled because the trained model is unavailable.")

st.divider()

# --------------------------------------------------
# MACHINE PARAMETER HISTORY
# --------------------------------------------------
st.subheader("Machine Parameter History")
chart_data = df.tail(100).copy()

if "timestamp" in chart_data.columns:
    parsed_timestamp = pd.to_datetime(chart_data["timestamp"], errors="coerce")
    if parsed_timestamp.notna().any():
        chart_data["timestamp"] = parsed_timestamp
        chart_data = chart_data.dropna(subset=["timestamp"])
        chart_data = chart_data.set_index("timestamp")

if not chart_data.empty:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Temperature", "Vibration", "Spindle RPM", "Motor Current", "Tool Wear"]
    )
    chart_tabs = [
        (tab1, "temperature", "Temperature (°C)"),
        (tab2, "vibration", "Vibration (g)"),
        (tab3, "spindle_rpm", "Spindle RPM"),
        (tab4, "motor_current", "Motor Current (A)"),
        (tab5, "tool_wear", "Tool Wear (%)"),
    ]

    for tab, column, label in chart_tabs:
        with tab:
            if column in chart_data.columns:
                st.caption(f"Showing the latest {len(chart_data)} readings")
                st.line_chart(chart_data[[column]], y_label=label)
            else:
                st.info(f"{label} data is unavailable.")
else:
    st.info("Not enough valid history is available to draw charts yet.")

# --------------------------------------------------
# RECENT SENSOR DATA
# --------------------------------------------------
st.subheader("Recent Sensor Data")
st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

st.caption("Dashboard auto-refresh interval: 2 seconds.")
