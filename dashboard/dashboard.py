import time
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="CNC Digital Twin",
    page_icon="🏭",
    layout="wide"
)


# -----------------------------------
# TITLE
# -----------------------------------

st.title("🏭 CNC Milling Machine Dashboard")

st.subheader("AI-Driven Digital Twin — Layer 1")

st.write("Synthetic Data → MQTT → CSV → Dashboard")


# -----------------------------------
# CSV FILE
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_FILE = BASE_DIR / "data" / "cnc_machine_data.csv"


# -----------------------------------
# DASHBOARD PLACEHOLDER
# -----------------------------------

placeholder = st.empty()


# -----------------------------------
# LIVE DASHBOARD
# -----------------------------------

while True:

    with placeholder.container():

        if not CSV_FILE.exists():

            st.error("CSV file not found!")

        else:

            df = pd.read_csv(CSV_FILE)

            if df.empty:

                st.warning("Waiting for CNC data...")

            else:

                # Latest machine reading
                latest = df.iloc[-1]


                # -----------------------------------
                # MACHINE STATUS
                # -----------------------------------

                status = str(latest["status"])


                if status == "NORMAL":

                    st.success("🟢 MACHINE STATUS: NORMAL")

                elif status == "WARNING":

                    st.warning("🟡 MACHINE STATUS: WARNING")

                else:

                    st.error("🔴 MACHINE STATUS: CRITICAL")


                st.write(
                    f"Machine: **{latest['machine_id']}**"
                )


                # -----------------------------------
                # METRICS
                # -----------------------------------

                col1, col2, col3, col4, col5 = st.columns(5)


                col1.metric(
                    "🌡️ Temperature",
                    f"{latest['temperature']} °C"
                )


                col2.metric(
                    "📳 Vibration",
                    f"{latest['vibration']} g"
                )


                col3.metric(
                    "🔄 Spindle RPM",
                    f"{int(latest['spindle_rpm'])}"
                )


                col4.metric(
                    "⚡ Motor Current",
                    f"{latest['motor_current']} A"
                )


                col5.metric(
                    "🛠️ Tool Wear",
                    f"{latest['tool_wear']} %"
                )


                # -----------------------------------
                # GRAPHS
                # -----------------------------------

                st.subheader("📈 Machine Parameters")


                chart_data = df.tail(60).copy()


                chart_data["timestamp"] = pd.to_datetime(
                    chart_data["timestamp"]
                )


                chart_data = chart_data.set_index("timestamp")


                st.line_chart(
                    chart_data[
                        [
                            "temperature",
                            "vibration",
                            "spindle_rpm",
                            "motor_current",
                            "tool_wear"
                        ]
                    ]
                )


                # -----------------------------------
                # RECENT DATA
                # -----------------------------------

                st.subheader("📋 Recent CNC Data")

                st.dataframe(
                    df.tail(10),
                    use_container_width=True
                )


    # Refresh every 2 seconds
    time.sleep(2)