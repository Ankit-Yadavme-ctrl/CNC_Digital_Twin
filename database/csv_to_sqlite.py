import sqlite3
import csv
import time
from pathlib import Path


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_FILE = BASE_DIR / "data" / "cnc_hivemq_data.csv"

DB_FILE = BASE_DIR / "database" / "cnc_digital_twin.db"


# ==========================================
# DATABASE
# ==========================================

connection = sqlite3.connect(DB_FILE)

cursor = connection.cursor()


# ==========================================
# CREATE TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS sensor_data (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    timestamp TEXT,

    machine_id TEXT,

    temperature REAL,

    vibration REAL,

    spindle_rpm INTEGER,

    motor_current REAL,

    tool_wear REAL,

    status TEXT
)
""")


connection.commit()


print("======================================")
print(" CNC DIGITAL TWIN DATABASE")
print("======================================")
print("Database:", DB_FILE)
print()


# ==========================================
# TRACK PROCESSED ROWS
# ==========================================

processed_rows = 0


# ==========================================
# CSV → SQLITE
# ==========================================

while True:

    try:

        if CSV_FILE.exists():

            with open(
                CSV_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                rows = list(csv.DictReader(file))


            new_rows = rows[processed_rows:]


            for row in new_rows:

                cursor.execute(
                    """
                    INSERT INTO sensor_data
                    (
                        timestamp,
                        machine_id,
                        temperature,
                        vibration,
                        spindle_rpm,
                        motor_current,
                        tool_wear,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,

                    (
                        row["timestamp"],
                        row["machine_id"],
                        float(row["temperature"]),
                        float(row["vibration"]),
                        int(row["spindle_rpm"]),
                        float(row["motor_current"]),
                        float(row["tool_wear"]),
                        row["status"]
                    )
                )


                connection.commit()


                print(
                    "Inserted:",
                    row["timestamp"],
                    "|",
                    row["machine_id"],
                    "|",
                    row["status"]
                )


            processed_rows = len(rows)


        else:

            print("CSV file not found...")


    except Exception as error:

        print("Error:", error)


    time.sleep(2)