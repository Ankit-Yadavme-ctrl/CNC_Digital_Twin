# LIVE VERSION - replace cloud_dashboard.py with this file
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

st.set_page_config(page_title="CNC Digital Twin - Cloud", page_icon="🏭", layout="wide")
st_autorefresh(interval=2000, key="cnc_live_refresh")

st.title("🏭 CNC Digital Twin — Cloud Dashboard")
st.subheader("Layer 6: AI Predictive Maintenance")

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_FILE = BASE_DIR / "data" / "cnc_hivemq_data.csv"
MODEL_FILE = BASE_DIR / "ml" / "cnc_failure_model.pkl"

# Streamlit Cloud Secrets:
# [mqtt]
# host = "YOUR_HIVEMQ_HOST"
# port = 8883
# username = "YOUR_HIVEMQ_USERNAME"
# password = "YOUR_HIVEMQ_PASSWORD"
# topic = "factory/cnc/CNC_01/sensors"

def mqtt_setting(name, default=None):
    try:
        if "mqtt" in st.secrets and name in st.secrets["mqtt"]:
            return st.secrets["mqtt"][name]
    except Exception:
        pass
    return os.getenv("MQTT_" + name.upper(), default)

MQTT_HOST = mqtt_setting("host")
MQTT_PORT = int(mqtt_setting("port", 8883))
MQTT_USERNAME = mqtt_setting("username")
MQTT_PASSWORD = mqtt_setting("password")
MQTT_TOPIC = mqtt_setting("topic", "factory/cnc/CNC_01/sensors")

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

def on_connect(client, userdata, flags, reason_code, properties=None):
    rc = getattr(reason_code, "value", reason_code)
    with live["lock"]:
        live["connected"] = (rc == 0)
        live["error"] = None if rc == 0 else f"MQTT connection failed: {rc}"
    if rc == 0:
        client.subscribe(MQTT_TOPIC, qos=0)

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    with live["lock"]:
        live["connected"] = False

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        for field in ["temperature", "vibration", "spindle_rpm", "motor_current", "tool_wear"]:
            if field in data:
                data[field] = float(data[field])
        data["machine_id"] = str(data.get("machine_id", "CNC_01"))
        data["status"] = str(data.get("status", "NORMAL")).upper()
        data["timestamp"] = str(data.get("timestamp", pd.Timestamp.utcnow()))
        with live["lock"]:
            live["latest"] = data
            live["history"].append(dict(data))
            live["error"] = None
    except Exception as e:
        with live["lock"]:
            live["error"] = f"Invalid MQTT message: {e}"

@st.cache_resource
def start_mqtt():
    if mqtt is None or not all([MQTT_HOST, MQTT_USERNAME, MQTT_PASSWORD]):
        return None
    try:
