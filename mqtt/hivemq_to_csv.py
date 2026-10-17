
import csv
import json
import os
from pathlib import Path

import paho.mqtt.client as mqtt

BROKER = os.getenv("HIVEMQ_BROKER", "")
PORT = int(os.getenv("HIVEMQ_PORT", "8883"))
USERNAME = os.getenv("HIVEMQ_USERNAME", "")
PASSWORD = os.getenv("HIVEMQ_PASSWORD", "")
TOPIC = "factory/cnc/CNC_01/sensors"

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_FILE = BASE_DIR / "data" / "cnc_hivemq_data.csv"

FIELDS = [
    "timestamp",
    "machine_id",
    "temperature",
    "vibration",
    "spindle_rpm",
    "motor_current",
    "tool_wear",
    "status",
]


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to HiveMQ Cloud")
        client.subscribe(TOPIC)
        print("Subscribed to:", TOPIC)
    else:
        print("Connection failed:", reason_code)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))

        # Validate the incoming message before writing it.
        missing = [field for field in FIELDS if field not in data]
        if missing:
            print("Skipping message; missing fields:", missing)
            return

        CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Create a clean header if the file is missing, empty,
        # or does not have the expected header.
        needs_header = True
        if CSV_FILE.exists() and CSV_FILE.stat().st_size > 0:
            try:
                with CSV_FILE.open("r", newline="", encoding="utf-8") as f:
                    first_row = next(csv.reader(f), [])
                needs_header = first_row != FIELDS
            except (OSError, csv.Error):
                needs_header = True

        if needs_header:
            # Preserve the old file rather than silently discarding it.
            if CSV_FILE.exists() and CSV_FILE.stat().st_size > 0:
                backup = CSV_FILE.with_name("cnc_hivemq_data_backup.csv")
                if not backup.exists():
                    CSV_FILE.replace(backup)
                else:
                    print("Backup already exists:", backup)
                    print("Please move the malformed CSV aside manually.")
                    return

            with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

        with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writerow({field: data[field] for field in FIELDS})

        print("Saved sensor reading:", data.get("timestamp"))

    except (ValueError, OSError, csv.Error) as e:
        print("CSV/message error:", e)


if not all([BROKER, USERNAME, PASSWORD]):
    raise RuntimeError(
        "Set HIVEMQ_BROKER, HIVEMQ_USERNAME and HIVEMQ_PASSWORD "
        "as environment variables before starting this receiver."
    )

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set()
client.on_connect = on_connect
client.on_message = on_message

print("Connecting to HiveMQ Cloud...")
client.connect(BROKER, PORT, 60)
client.loop_forever()
