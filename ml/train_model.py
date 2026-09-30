import sqlite3
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Database
DB_PATH = r"database\cnc_digital_twin.db"

# Load valid labeled data
conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    temperature,
    vibration,
    spindle_rpm,
    motor_current,
    tool_wear,
    status
FROM cnc_sensor_data
WHERE
    temperature IS NOT NULL
    AND vibration IS NOT NULL
    AND spindle_rpm IS NOT NULL
    AND motor_current IS NOT NULL
    AND tool_wear IS NOT NULL
    AND status IN ('NORMAL', 'WARNING')
"""

df = pd.read_sql_query(query, conn)
conn.close()

print("Valid records:", len(df))
print("\nStatus distribution:")
print(df["status"].value_counts())

# Features
X = df[
    [
        "temperature",
        "vibration",
        "spindle_rpm",
        "motor_current",
        "tool_wear"
    ]
]

# Target
y = df["status"]

# Convert labels
y = y.map({
    "NORMAL": 0,
    "WARNING": 1
})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL TRAINING COMPLETE")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NORMAL", "WARNING"]
    )
)

# Save model
joblib.dump(model, "ml/cnc_failure_model.pkl")

print("\nModel saved:")
print("ml/cnc_failure_model.pkl")