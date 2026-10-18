from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
from streamlit_autorefresh import st_autorefresh


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="CNC Digital Twin - Cloud",
    page_icon="CNC",
    layout="wide"
)


# ==========================================
# LIVE AUTO REFRESH
# ==========================================

st_autorefresh(
    interval=2000,
    key="cnc_live_refresh"
)


# ==========================================
# TITLE
# ==========================================

st.title("CNC Digital Twin - Cloud Dashboard")

st.subheader("Layer 6: AI-Based Status Classification")


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_FILE = BASE_DIR / "data" / "cnc_hivemq_data.csv"

MODEL_FILE = BASE_DIR / "ml" / "cnc_failure_model.pkl"


# ==========================================

# ==========================================
# LOAD ML MODEL
# ==========================================

model = None
model_loaded = False

if MODEL_FILE.exists():
    try:
        model = joblib.load(MODEL_FILE)
        model_loaded = True
    except Exception as e:
        st.warning(f"AI model could not be loaded: {e}")
else:
    st.info(
        "AI model is not available in this deployment. "
        "Live sensor monitoring can continue, but AI classification is disabled."
    )


# ==========================================
# READ LIVE MQTT DATA, WITH CSV FALLBACK

from dashboard.cloud_mqtt import get_live_data

try:
    df = get_live_data()
except Exception as e:
    st.warning(f"MQTT connection unavailable: {e}")
    df = pd.DataFrame()

if df.empty and CSV_FILE.exists():
    try:
        df = pd.read_csv(CSV_FILE)
        st.info("Showing CSV data because live MQTT data is unavailable.")
    except Exception as e:
        st.warning(f"Could not read CSV file: {e}")
        df = pd.DataFrame()

if df.empty:
    st.info("Waiting for live CNC sensor readings from HiveMQ.")# ==========================================

if df.empty:

    st.warning(
        "Waiting for cloud sensor data..."
    )

else:

    # ==========================================
    # LATEST DATA
    # ==========================================

    latest = df.iloc[-1]


    # ==========================================
    # SENSOR VALUES
    # ==========================================

    temperature = float(
        latest["temperature"]
    )

    vibration = float(
        latest["vibration"]
    )

    spindle_rpm = float(
        latest["spindle_rpm"]
    )

    motor_current = float(
        latest["motor_current"]
    )

    tool_wear = float(
        latest["tool_wear"]
    )


    # ==========================================
    # MACHINE INFORMATION
    # ==========================================

    machine_id = str(
        latest["machine_id"]
    )

    timestamp = str(
        latest["timestamp"]
    )


    st.write(
        f"**Machine:** {machine_id}"
    )

    st.caption(
        f"Last sensor reading: {timestamp}"
    )


    # ==========================================
    # MACHINE STATUS
    # ==========================================

    status = str(
        latest["status"]
    ).upper()


    if status == "NORMAL":

        st.success(
            "ðŸŸ¢ MACHINE STATUS: NORMAL"
        )

    elif status == "WARNING":

        st.warning(
            "ðŸŸ¡ MACHINE STATUS: WARNING"
        )

    else:

        st.error(
            "ðŸ”´ MACHINE STATUS: CRITICAL"
        )


    # ==========================================
    # SENSOR METRICS
    # ==========================================

    st.subheader("ðŸ“Š Live Machine Parameters")


    col1, col2, col3, col4, col5 = st.columns(5)


    col1.metric(
        "ðŸŒ¡ï¸ Temperature",
        f"{temperature:.2f} Â°C"
    )


    col2.metric(
        "ðŸ“³ Vibration",
        f"{vibration:.3f} g"
    )


    col3.metric(
        "ðŸ”„ Spindle RPM",
        f"{int(spindle_rpm)}"
    )


    col4.metric(
        "âš¡ Motor Current",
        f"{motor_current:.2f} A"
    )


    col5.metric(
        "ðŸ› ï¸ Tool Wear",
        f"{tool_wear:.2f} %"
    )


    # ==========================================
    # AI PREDICTIVE MAINTENANCE
    # ==========================================

    st.subheader(
        "ðŸ¤– AI-Based Status Classification"
    )


    if model_loaded:

        # ======================================
        # MODEL INPUT
        # ======================================

        input_data = pd.DataFrame(
            [[
                temperature,
                vibration,
                spindle_rpm,
                motor_current,
                tool_wear
            ]],
            columns=[
                "temperature",
                "vibration",
                "spindle_rpm",
                "motor_current",
                "tool_wear"
            ]
        )


        # ======================================
        # AI PREDICTION
        # ======================================

        try:

            prediction_raw = model.predict(
                input_data
            )[0]


            prediction_text = str(
                prediction_raw
            ).upper()


            if prediction_text in [
                "WARNING",
                "1"
            ]:

                ai_status = "WARNING"

            elif prediction_text in [
                "NORMAL",
                "0"
            ]:

                ai_status = "NORMAL"

            else:

                ai_status = prediction_text


        except Exception as e:

            ai_status = "ERROR"

            st.error(
                f"AI prediction error: {e}"
            )


        # ======================================
        # AI PROBABILITY
        # ======================================

        normal_probability = 0.0

        warning_probability = 0.0


        try:

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = model.classes_


            for cls, probability in zip(
                classes,
                probabilities
            ):

                cls_text = str(
                    cls
                ).upper()


                if cls_text in [
                    "NORMAL",
                    "0"
                ]:

                    normal_probability = float(
                        probability
                    )


                elif cls_text in [
                    "WARNING",
                    "1"
                ]:

                    warning_probability = float(
                        probability
                    )


        except Exception:

            if ai_status == "WARNING":

                normal_probability = 0.0

                warning_probability = 1.0

            else:

                normal_probability = 1.0

                warning_probability = 0.0


        # ======================================
        # HEALTH / FAILURE RISK
        # ======================================

        machine_health = (
            normal_probability * 100
        )

        failure_risk = (
            warning_probability * 100
        )


        # ======================================
        # AI METRICS
        # ======================================

        ai_col1, ai_col2, ai_col3 = st.columns(3)


        ai_col1.metric(
            "ðŸ¤– AI Status Classification",
            ai_status
        )


        ai_col2.metric(
            "â¤ï¸ Normal Status Probability",
            f"{machine_health:.1f}%"
        )


        ai_col3.metric(
            "âš ï¸ Warning Status Probability",
            f"{failure_risk:.1f}%"
        )


        # ======================================
        # AI ALERT
        # ======================================

        if ai_status == "NORMAL":

            st.success(
                "ðŸŸ¢ AI ALERT: Machine operating normally."
            )

        elif ai_status == "WARNING":

            st.error(
                "ðŸš¨ AI ALERT: "
                "CNC machine may require attention!"
            )

            st.warning(
                f"âš ï¸ Warning Status Probability: "
                f"{failure_risk:.1f}%"
            )

            st.info(
                "Recommended action: Check "
                "temperature, vibration, "
                "motor current and tool wear."
            )

        else:

            st.warning(
                f"âš ï¸ AI returned: {ai_status}"
            )


    else:

        st.warning(
            "âš ï¸ AI model unavailable."
        )


    # ==========================================
    # GRAPHS
    # ==========================================

    st.subheader(
        "ðŸ“ˆ Machine Parameter History"
    )


    chart_data = df.tail(50).copy()


    if "timestamp" in chart_data.columns:

        chart_data["timestamp"] = pd.to_datetime(
            chart_data["timestamp"],
            errors="coerce"
        )

        chart_data = chart_data.dropna(
            subset=["timestamp"]
        )

        chart_data = chart_data.set_index(
            "timestamp"
        )


    available_columns = [
        column
        for column in [
            "temperature",
            "vibration",
            "spindle_rpm",
            "motor_current",
            "tool_wear"
        ]
        if column in chart_data.columns
    ]


    if available_columns:

        st.line_chart(
            chart_data[available_columns]
        )


    # ==========================================
    # RECENT DATA
    # ==========================================

    st.subheader(
        "ðŸ“‹ Recent Sensor Data"
    )


    st.dataframe(
        df.tail(10),
        width="stretch"
    )


    # ==========================================
    # LIVE INDICATOR
    # ==========================================

    st.caption(
        "ðŸ”´ LIVE â€” Dashboard refreshes every 2 seconds"
    )
