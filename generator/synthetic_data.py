import json
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt


# -----------------------------
# MQTT SETTINGS
# -----------------------------
BROKER = "localhost"
PORT = 1883
TOPIC = "factory/cnc/CNC_01/sensors"


# -----------------------------
# CREATE MQTT CLIENT
# -----------------------------
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

print("Connecting to MQTT broker...")

client.connect(BROKER, PORT, 60)

print("Connected successfully!")
print()
print("======================================")
print(" CNC MILLING MACHINE DATA GENERATOR")
print("======================================")
print("Machine ID :", "CNC_01")
print("MQTT Topic :", TOPIC)
print("Publishing every 1 second...")
print("Press CTRL+C to stop.")
print()


# -----------------------------
# GENERATE DATA CONTINUOUSLY
# -----------------------------
while True:

    # CNC machine parameters
    temperature = round(random.uniform(40, 55), 2)

    vibration = round(random.uniform(0.10, 0.50), 3)

    spindle_rpm = random.randint(2400, 2600)

    motor_current = round(random.uniform(3.5, 5.5), 2)

    tool_wear = round(random.uniform(5, 20), 2)


    # -----------------------------
    # MACHINE STATUS
    # -----------------------------
    if temperature > 52 or vibration > 0.45:
        status = "WARNING"
    else:
        status = "NORMAL"


    # -----------------------------
    # CREATE CNC DATA
    # -----------------------------
    data = {

        "timestamp":
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "machine_id":
            "CNC_01",

        "temperature":
            temperature,

        "vibration":
            vibration,

        "spindle_rpm":
            spindle_rpm,

        "motor_current":
            motor_current,

        "tool_wear":
            tool_wear,

        "status":
            status
    }


    # -----------------------------
    # CONVERT TO JSON
    # -----------------------------
    message = json.dumps(data)


    # -----------------------------
    # SEND DATA THROUGH MQTT
    # -----------------------------
    client.publish(TOPIC, message)


    # -----------------------------
    # DISPLAY DATA
    # -----------------------------
    print(message)


    # Wait 1 second
    time.sleep(1)