import json
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt


BROKER = "fd161802694643e682ca4aa728cf4025.s1.eu.hivemq.cloud"
PORT = 8883

USERNAME = "cnc_user"
PASSWORD = "Yadav12@"

TOPIC = "factory/cnc/CNC_01/sensors"


def on_connect(client, userdata, flags, reason_code, properties):
    print("CONNECTED TO HIVEMQ CLOUD")
    print("Topic:", TOPIC)
    print()


def on_disconnect(client, userdata, flags, reason_code, properties):
    print("MQTT DISCONNECTED")
    print("Reason:", reason_code)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

client.username_pw_set(USERNAME, PASSWORD)

client.tls_set()

client.on_connect = on_connect
client.on_disconnect = on_disconnect


print("Connecting to HiveMQ Cloud...")

client.connect(BROKER, PORT, 60)

client.loop_start()

# Wait for connection
time.sleep(3)


try:

    while True:

        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": "CNC_01",
            "temperature": round(random.uniform(40, 55), 2),
            "vibration": round(random.uniform(0.10, 0.50), 3),
            "spindle_rpm": random.randint(2400, 2600),
            "motor_current": round(random.uniform(3.5, 5.5), 2),
            "tool_wear": round(random.uniform(5, 20), 2)
        }

        if data["temperature"] > 52 or data["vibration"] > 0.45:
            data["status"] = "WARNING"
        else:
            data["status"] = "NORMAL"

        message = json.dumps(data)

        if client.is_connected():

            info = client.publish(
                TOPIC,
                message,
                qos=0
            )

            info.wait_for_publish()

            print("Published:", message)

        else:

            print("Waiting for MQTT connection...")

        time.sleep(2)


except KeyboardInterrupt:

    print("\nStopping generator...")

finally:

    client.loop_stop()
    client.disconnect()

    print("Disconnected.")