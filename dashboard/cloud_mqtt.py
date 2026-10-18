import json
import ssl
import threading

import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st


@st.cache_resource
def start_mqtt():
    config = st.secrets["mqtt"]

    host = config.get("host") or config.get("broker")
    port = int(config.get("port", 8883))
    username = config["username"]
    password = config["password"]
    topic = config.get(
        "topic",
        "factory/cnc/CNC_01/sensors",
    )

    records = []
    lock = threading.Lock()

    def on_message(client, userdata, message):
        try:
            data = json.loads(message.payload.decode("utf-8"))

            required = [
                "temperature",
                "vibration",
                "spindle_rpm",
                "motor_current",
                "tool_wear",
            ]

            if not all(key in data for key in required):
                return

            data.setdefault("timestamp", "")
            data.setdefault("machine_id", "CNC_01")
            data.setdefault("status", "NORMAL")

            with lock:
                records.append(data)

                if len(records) > 500:
                    del records[:-500]

        except (ValueError, UnicodeDecodeError):
            return

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2
    )

    client.username_pw_set(username, password)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.on_message = on_message

    client.connect(host, port, keepalive=60)
    client.subscribe(topic)
    client.loop_start()

    return client, records, lock


def get_live_data():
    client, records, lock = start_mqtt()

    with lock:
        snapshot = list(records)

    return pd.DataFrame(snapshot)