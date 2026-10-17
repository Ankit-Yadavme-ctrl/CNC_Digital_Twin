import csv
import json
from pathlib import Path

import paho.mqtt.client as mqtt


# MQTT settings
BROKER = "localhost"
PORT = 1883
TOPIC = "factory/cnc/CNC_01/sensors"


# CSV file location
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_FILE = BASE_DIR / "data" / "cnc_machine_data.csv"


# CSV columns
FIELDS = [
    "timestamp",
    "machine_id",
    "temperature",
    "vibration",
    "spindle_rpm",
    "motor_current",
    "tool_wear",
    "status"
]


# Create CSV file and header if it doesn't exist
CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists():

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=FIELDS)

        writer.writeheader()


# When connected to MQTT
def on_connect(client, userdata, flags, reason_code, properties):

    print("Connected to MQTT broker!")

    print("Subscribed to:", TOPIC)

    client.subscribe(TOPIC)


# When a message arrives
def on_message(client, userdata, message):

    try:

        data = json.loads(message.payload.decode("utf-8"))

        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:

            writer = csv.DictWriter(file, fieldnames=FIELDS)

            writer.writerow(data)

        print("Saved to CSV:", data)

    except Exception as error:

        print("Error:", error)


# Create MQTT client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

client.on_connect = on_connect
client.on_message = on_message


print("===================================")
print(" MQTT → CSV")
print("===================================")
print("Connecting to MQTT broker...")


client.connect(BROKER, PORT, 60)


print("Waiting for CNC data...")
print("CSV file:", CSV_FILE)


# Keep receiving messages
client.loop_forever()