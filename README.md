# 🏭 AI-Driven Digital Twin for Smart Factory Operations

An IoT and AI-based Digital Twin system for real-time monitoring and predictive maintenance of a CNC machine.

## 🎯 Project Objective

The system creates a digital representation of a CNC machine to monitor operating parameters, store historical data, and support AI-based predictive maintenance.

## 🔄 System Architecture

```text
CNC Machine / Simulator
        ↓
Sensor Data Generation
        ↓
MQTT
        ↓
HiveMQ Cloud
        ↓
Node-RED
   ↙           ↘
SQLite       Dashboard
                ↓
          Streamlit Cloud
                ↓
            AI / ML Model
                ↓
       Machine Health & Risk
```

## 📊 Parameters Monitored

| Parameter | Description |
|---|---|
| 🌡️ Temperature | CNC machine temperature |
| 📳 Vibration | Machine vibration level |
| 🔄 Spindle RPM | Spindle rotational speed |
| ⚡ Motor Current | Motor current consumption |
| 🛠️ Tool Wear | Tool wear percentage |
| 🚦 Status | NORMAL / WARNING / CRITICAL |

## 📡 MQTT Communication

**Broker:** HiveMQ Cloud

**Topic:**
```text
factory/cnc/CNC_01/sensors
```

Example message:

```json
{
  "timestamp": "2026-10-01T06:06:01.574Z",
  "machine_id": "CNC_01",
  "temperature": 55.41,
  "vibration": 0.244,
  "spindle_rpm": 2404,
  "motor_current": 5.89,
  "tool_wear": 12.4,
  "status": "WARNING"
}
```

## 🔧 Node-RED

Node-RED receives MQTT messages, processes the data, displays machine parameters, and stores readings in SQLite.

```text
MQTT IN
   ↓
JSON / Function
   ├── Temperature
   ├── Vibration
   ├── Spindle RPM
   ├── Motor Current
   ├── Tool Wear
   ├── Machine Status
   └── SQLite Database
```

## 🗄️ Database

SQLite stores historical CNC sensor data in:

```text
cnc_sensor_data
```

Typical fields:

```text
timestamp
machine_id
temperature
vibration
spindle_rpm
motor_current
tool_wear
status
```

## 🤖 AI Predictive Maintenance

The Machine Learning model uses:

```text
Temperature
Vibration
Spindle RPM
Motor Current
Tool Wear
```

to predict machine condition such as:

```text
NORMAL
WARNING
```

The dashboard can display AI prediction, machine health, failure risk, and recommended maintenance action.

## 📈 Streamlit Dashboard

Features include:

- Live machine parameters
- Machine status
- MQTT connection status
- AI prediction
- Machine health
- Failure risk
- Sensor history
- Recent sensor data
- Parameter charts

## 🛠️ Technologies Used

- Python
- MQTT
- HiveMQ Cloud
- Node-RED
- SQLite
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Streamlit Cloud
- ESP32 / CNC simulator
- Git & GitHub

## 📁 Project Structure

```text
CNC_Digital_Twin/
│
├── app.py
├── dashboard/
│   ├── login.py
│   └── cloud_dashboard.py
├── data/
│   └── cnc_hivemq_data.csv
├── database/
│   └── cnc_digital_twin.db
├── ml/
│   └── cnc_failure_model.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CNC_Digital_Twin
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run Streamlit

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## ▶️ Run Node-RED

```bash
node-red
```

Open:

```text
http://localhost:1880
```

Dashboard:

```text
http://localhost:1880/dashboard
```

## 🔐 Streamlit Secrets

Never upload your real HiveMQ password to GitHub.

In Streamlit Cloud → **Manage app → Settings → Secrets**:

```toml
[mqtt]
host = "YOUR_HIVEMQ_HOST"
port = 8883
username = "YOUR_MQTT_USERNAME"
password = "YOUR_MQTT_PASSWORD"
topic = "factory/cnc/CNC_01/sensors"
```

## 🧪 Live Data Flow

```text
CNC Simulator / ESP32
        ↓
MQTT Publish
        ↓
HiveMQ Cloud
        ↓
Node-RED MQTT IN
        ↓
Data Processing
        ↓
SQLite + Dashboard
        ↓
Streamlit
        ↓
AI Prediction
```

## 🔮 Future Scope

- Real CNC sensor integration
- ESP32 sensor acquisition
- Multiple CNC monitoring
- Advanced anomaly detection
- Deep Learning
- Remaining Useful Life prediction
- Automated maintenance alerts
- Cloud database
- 3D CNC visualization
- Edge AI

## 📌 Project Status

The prototype includes MQTT communication, HiveMQ Cloud integration, Node-RED processing, SQLite storage, CNC parameter monitoring, Machine Learning prediction, and a Streamlit dashboard.

## 👨‍💻 Project

**Project Title:** AI-Driven Digital Twin for Smart Factory Operations

**Application:** CNC Machine Monitoring & Predictive Maintenance

**Domains:** Artificial Intelligence, IoT, Digital Twin, Smart Manufacturing, Predictive Maintenance, Industrial Automation

## ⭐ Conclusion

This project combines IoT, MQTT, cloud communication, data processing, machine learning, and visualization to create a Digital Twin of a CNC machine for real-time monitoring and AI-assisted predictive maintenance.
