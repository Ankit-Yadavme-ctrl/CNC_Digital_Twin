from pathlib import Path
from collections import deque
import json
import os
import threading

import pandas as pd
import streamlit as st
import joblib
from streamlit_autorefresh import st_autorefresh

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="CNC Digital Twin - Cloud",
    page_icon="🏭",
    layout="wide"
)

st_autorefresh(
    interval=2000,
    key="cnc_live_refresh"
)


# ==========================================
# TITLE
# ==========================================

st.title("🏭 CNC Digital Twin — Cloud Dashboard")
st.subheader("Layer 6: AI Predictive Maintenance")


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_FILE = BASE_DIR / "data" / "cnc_hivemq_data.csv"
MODEL_FILE = BASE_DIR / "ml" / "cnc_failure_model.pkl"


# ==========================================
# HIVE MQ SETTINGS
# ==========================================

def mqtt_setting(name, default=None):

    try:
        if "mqtt" in st.secrets and name in st.secrets["mqtt"]:
            return st.secrets["mqtt"][name]

    except Exception:
        pass

    return os.getenv(
        "MQTT_" + name.upper(),
        default
    )


MQTT_HOST = mqtt_setting("host")

MQTT_PORT = int(
    mqtt_setting("port", 8883)
)

MQTT_USERNAME = mqtt_setting("username")

MQTT_PASSWORD = mqtt_setting("password")

MQTT_TOPIC = mqtt_setting(
    "topic",
    "factory/cnc/CNC_01/sensors"
)


# ==========================================
# LIVE MQTT STORE
# ==========================================

@st.cache_resource
def live_store():

    return {
        "latest": None,
        "history": deque(maxlen=200),
        "connected": False,
        "error": None,
        "client": None,
        "lock": threading.Lock()
    }


live = live_store()


# ==========================================
# MQTT CONNECT
# ==========================================

def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties=None
):

    try:

        rc = getattr(
            reason_code,
            "value",
            reason_code
        )

        with live["lock"]:

            live["connected"] = (
                rc == 0
            )

            live["error"] = (
                None
                if rc == 0
                else f"MQTT connection failed: {rc}"
            )

        if rc == 0:

            client.subscribe(
                MQTT_TOPIC,
                qos=0
            )

    except Exception as e:

        with live["lock"]:

            live["connected"] = False

            live["error"] = str(e)


# ==========================================
# MQTT DISCONNECT
# ==========================================

def on_disconnect(
    client,
    userdata,
    disconnect_flags,
    reason_code,
    properties=None
):

    with live["lock"]:

        live["connected"] = False


# ==========================================
# MQTT MESSAGE
# ==========================================

def on_message(
    client,
    userdata,
    msg
):

    try:

        payload = msg.payload.decode(
            "utf-8"
        )

        data = json.loads(payload)

        numeric_fields = [
            "temperature",
            "vibration",
            "spindle_rpm",
            "motor_current",
            "tool_wear"
        ]

        for field in numeric_fields:

            if field in data:

                data[field] = float(
                    data[field]
                )

        data["machine_id"] = str(
            data.get(
                "machine_id",
                "CNC_01"
            )
        )

        data["status"] = str(
            data.get(
                "status",
                "NORMAL"
            )
        ).upper()

        data["timestamp"] = str(
            data.get(
                "timestamp",
                pd.Timestamp.utcnow()
            )
        )

        with live["lock"]:

            live["latest"] = data

            live["history"].append(
                dict(data)
            )

            live["error"] = None

    except Exception as e:

        with live["lock"]:

            live["error"] = (
                f"Invalid MQTT message: {e}"
            )


# ==========================================
# START MQTT CLIENT
# ==========================================

@st.cache_resource
def start_mqtt():

    if mqtt is None:

        return None

    if not all(
        [
            MQTT_HOST,
            MQTT_USERNAME,
            MQTT_PASSWORD
        ]
    ):

        return None

    try:

        client = mqtt.Client(
            callback_api_version=(
                mqtt.CallbackAPIVersion.VERSION2
            ),
            client_id=(
                f"cnc-streamlit-{os.getpid()}"
            ),
            protocol=mqtt.MQTTv311
        )

        client.username_pw_set(
            MQTT_USERNAME,
            MQTT_PASSWORD
        )

        client.tls_set()

        client.on_connect = on_connect

        client.on_disconnect = on_disconnect

        client.on_message = on_message

        client.connect(
            MQTT_HOST,
            MQTT_PORT,
            keepalive=60
        )

        client.loop_start()

        live["client"] = client

        return client

    except Exception as e:

        with live["lock"]:

            live["connected"] = False

            live["error"] = (
                f"Could not start MQTT: {e}"
            )

        return None


mqtt_client = start_mqtt()


# ==========================================
# LOAD ML MODEL
# ==========================================

try:

    model = joblib.load(
        MODEL_FILE
    )

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        f"ML model could not be loaded: {e}"
    )


# ==========================================
# GET LIVE DATA
# ==========================================

with live["lock"]:

    latest_live = (
        dict(live["latest"])
        if live["latest"]
        else None
    )

    history_live = list(
        live["history"]
    )

    mqtt_connected = (
        live["connected"]
    )

    mqtt_error = (
        live["error"]
    )


# ==========================================
# MQTT STATUS
# ==========================================

if mqtt is None:

    st.error(
        "paho-mqtt is not installed. "
        "Add paho-mqtt to requirements.txt."
    )

elif not all(
    [
        MQTT_HOST,
        MQTT_USERNAME,
        MQTT_PASSWORD
    ]
):

    st.warning(
        "⚠️ HiveMQ credentials are not "
        "configured in Streamlit Secrets."
    )

elif mqtt_connected:

    st.success(
        f"🟢 LIVE MQTT CONNECTED — "
        f"{MQTT_TOPIC}"
    )

else:

    st.warning(
        "🟡 Connecting to HiveMQ Cloud..."
    )

    if mqtt_error:

        st.caption(
            mqtt_error
        )


# ==========================================
# CSV FALLBACK
# ==========================================

df = pd.DataFrame()


if latest_live is not None:

    latest = latest_live

    df = pd.DataFrame(
        history_live
    )

elif CSV_FILE.exists():

    try:

        df = pd.read_csv(
            CSV_FILE
        )

    except Exception as e:

        st.error(
            f"Could not read CSV file: {e}"
        )


# ==========================================
# NO DATA
# ==========================================

if latest_live is None and df.empty:

    st.info(
        "⏳ Waiting for live HiveMQ sensor data..."
    )

    st.stop()


# ==========================================
# LATEST DATA
# ==========================================

if latest_live is None:

    latest = df.iloc[-1].to_dict()


# ==========================================
# REQUIRED COLUMNS
# ==========================================

required = [
    "temperature",
    "vibration",
    "spindle_rpm",
    "motor_current",
    "tool_wear"
]


missing = [
    column
    for column in required
    if column not in latest
]


if missing:

    st.error(
        "Missing sensor columns: "
        + ", ".join(missing)
    )

    st.stop()


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


machine_id = str(
    latest.get(
        "machine_id",
        "CNC_01"
    )
)


timestamp = str(
    latest.get(
        "timestamp",
        "N/A"
    )
)


status = str(
    latest.get(
        "status",
        "NORMAL"
    )
).upper()


# ==========================================
# MACHINE INFORMATION
# ==========================================

st.write(
    f"**Machine:** {machine_id}"
)

st.caption(
    f"Last sensor reading: {timestamp}"
)


# ==========================================
# MACHINE STATUS
# ==========================================

if status == "NORMAL":

    st.success(
        "🟢 MACHINE STATUS: NORMAL"
    )

elif status == "WARNING":

    st.warning(
        "🟡 MACHINE STATUS: WARNING"
    )

else:

    st.error(
        "🔴 MACHINE STATUS: CRITICAL"
    )


# ==========================================
# LIVE MACHINE PARAMETERS
# ==========================================

st.subheader(
    "📊 Live Machine Parameters"
)


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "🌡️ Temperature",
    f"{temperature:.2f} °C"
)


c2.metric(
    "📳 Vibration",
    f"{vibration:.3f} g"
)


c3.metric(
    "🔄 Spindle RPM",
    f"{int(spindle_rpm)}"
)


c4.metric(
    "⚡ Motor Current",
    f"{motor_current:.2f} A"
)


c5.metric(
    "🛠️ Tool Wear",
    f"{tool_wear:.2f} %"
)


# ==========================================
# AI PREDICTIVE MAINTENANCE
# ==========================================

st.subheader(
    "🤖 AI Predictive Maintenance"
)


if model_loaded:

    input_data = pd.DataFrame(
        [
            [
                temperature,
                vibration,
                spindle_rpm,
                motor_current,
                tool_wear
            ]
        ],
        columns=required
    )


    # ======================================
    # AI PREDICTION
    # ======================================

    try:

        prediction = str(
            model.predict(
                input_data
            )[0]
        ).upper()


        if prediction in [
            "WARNING",
            "1"
        ]:

            ai_status = "WARNING"

        elif prediction in [
            "NORMAL",
            "0"
        ]:

            ai_status = "NORMAL"

        else:

            ai_status = prediction


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


        for cls, probability in zip(
            model.classes_,
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
    # HEALTH AND RISK
    # ======================================

    health = (
        normal_probability * 100
    )

    risk = (
        warning_probability * 100
    )


    a1, a2, a3 = st.columns(3)


    a1.metric(
        "🤖 AI Prediction",
        ai_status
    )


    a2.metric(
        "❤️ Machine Health",
        f"{health:.1f}%"
    )


    a3.metric(
        "⚠️ Failure Risk",
        f"{risk:.1f}%"
    )


    # ======================================
    # AI ALERT
    # ======================================

    if ai_status == "NORMAL":

        st.success(
            "🟢 AI ALERT: "
            "Machine operating normally."
        )


    elif ai_status == "WARNING":

        st.error(
            "🚨 AI ALERT: "
            "CNC machine may require attention!"
        )

        st.warning(
            f"⚠️ Failure Risk: "
            f"{risk:.1f}%"
        )

        st.info(
            "Recommended action: Check "
            "temperature, vibration, "
            "motor current and tool wear."
        )


else:

    st.warning(
        "⚠️ AI model unavailable."
    )


# ==========================================
# MACHINE PARAMETER HISTORY
# ==========================================

st.subheader(
    "📈 Machine Parameter History"
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


available = [
    column
    for column in required
    if column in chart_data.columns
]


if available:

    st.line_chart(
        chart_data[available]
    )


# ==========================================
# RECENT SENSOR DATA
# ==========================================

st.subheader(
    "📋 Recent Sensor Data"
)


st.dataframe(
    df.tail(10),
    use_container_width=True
)


# ==========================================
# LIVE INDICATOR
# ==========================================

if mqtt_connected:

    st.caption(
        "🔴 LIVE — HiveMQ data received. "
        "Dashboard refreshes every 2 seconds."
    )

else:

    st.caption(
        "🟡 Waiting for HiveMQ live data. "
        "Dashboard refreshes every 2 seconds."
    )
