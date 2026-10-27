# CNC Digital Twin — AI-Driven Smart Manufacturing

A student project demonstrating real-time CNC machine monitoring using MQTT, Python, Streamlit, and machine learning.

## Features

- **Live sensor monitoring:** Receives simulated CNC sensor readings through HiveMQ Cloud using MQTT.
- **Interactive dashboard:** Displays temperature, vibration, spindle RPM, motor current, tool wear, and machine status.
- **AI-based classification:** Uses a trained machine-learning model to classify readings as NORMAL or WARNING.
- **Live visualization:** Displays sensor metrics, prediction results, and historical charts.
- **Data logging:** Supports storing incoming telemetry locally in CSV and SQLite.
- **Offline research:** Includes separate experiments using the PHM2010 cutting-tool wear dataset.

## System Architecture

1. A Python simulator generates synthetic CNC sensor readings.
2. The simulator publishes messages to HiveMQ Cloud using MQTT.
3. The Streamlit dashboard subscribes to the configured MQTT topic.
4. Incoming readings are processed and displayed in the dashboard.
5. An AI classification model predicts NORMAL or WARNING.
6. Separate scripts support local data storage and offline tool-wear experiments.

**MQTT topic:** `factory/cnc/CNC_01/sensors`

## Technologies

Python, Streamlit, Pandas, NumPy, scikit-learn, Joblib, Matplotlib, Paho MQTT, HiveMQ Cloud, SQLite, Git, and GitHub.

## Project Structure

```text
CNC_Digital_Twin/
├── app.py
├── requirements.txt
├── README.md
├── dashboard/
│   ├── cloud_dashboard.py
│   ├── cloud_mqtt.py
│   └── login.py
├── generator/
│   └── synthetic_data_hivemq.py
├── mqtt/
│   └── hivemq_to_csv.py
├── database/
│   └── csv_to_sqlite.py
├── ml/
│   ├── train_model.py
│   └── test_model.py
├── models/
├── screenshots/
└── analysis/
```

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CNC_Digital_Twin
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure HiveMQ Cloud credentials

The simulator reads these environment variables:

- `HIVEMQ_BROKER`
- `HIVEMQ_PORT`
- `HIVEMQ_USERNAME`
- `HIVEMQ_PASSWORD`

Configure them in the terminal used to run the simulator.

### 4. Configure Streamlit secrets

Create `.streamlit/secrets.toml` locally with the following structure:

```toml
dashboard_username = "YOUR_DASHBOARD_USERNAME"
dashboard_password = "YOUR_DASHBOARD_PASSWORD"

[mqtt]
host = "YOUR_CLUSTER_HOST"
port = 8883
username = "YOUR_HIVEMQ_USERNAME"
password = "YOUR_HIVEMQ_PASSWORD"
topic = "factory/cnc/CNC_01/sensors"
```

Replace the placeholders with your own credentials. Never commit this file or share its contents publicly.

### 5. Start the sensor simulator

In a terminal where the HiveMQ environment variables are configured:

```bash
python generator/synthetic_data_hivemq.py
```

### 6. Start the dashboard

In another terminal:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit.

The dashboard also requires its configured AI model and valid MQTT credentials.

## Dashboard Screenshot

![CNC Digital Twin Live Dashboard](screenshots/dashboard-live.png)

## Important Limitations

- Sensor readings are simulated; the current prototype is not connected to a physical CNC machine.
- The classification model is trained using generated data with rule-based labels. Its accuracy does not establish real-world CNC failure prediction accuracy.
- The PHM2010 tool-wear experiments are a separate offline research component.
- Experimental tool-wear models require further validation before industrial use.
- The login mechanism is intended for demonstration and is not production-grade authentication.
- AI predictions should not be used alone to make real maintenance decisions.

## Security

- Keep HiveMQ credentials and dashboard passwords private.
- Never commit `.streamlit/secrets.toml`, local databases, or private datasets.
- Review model and data files before publishing the repository.
- Configure secrets separately in any cloud deployment.

## Future Improvements

- Validate the AI model using real CNC sensor data.
- Integrate physical sensors into the MQTT pipeline.
- Improve authentication and model monitoring.
- Add automated tests and reproducible model evaluation.
- Improve fault detection and maintenance alerting.