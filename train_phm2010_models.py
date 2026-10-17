
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]

# Load and validate data
df = pd.read_csv(DATA_PATH)

required_columns = ["cut", *TARGETS]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Use sensor-derived features only.
# Exclude cut, source_file, sample count, and wear labels.
feature_columns = [
    col for col in df.columns
    if col not in ["cut", "source_file", "n_samples", *TARGETS]
]

if len(feature_columns) != 42:
    raise ValueError(
        f"Expected 42 sensor features, found {len(feature_columns)}."
    )

df = df.sort_values("cut").reset_index(drop=True)

if df[feature_columns + TARGETS].isna().any().any():
    raise ValueError("Missing values found in features or targets.")

if not np.isfinite(df[feature_columns + TARGETS].to_numpy()).all():
    raise ValueError("Non-finite values found in features or targets.")

# Chronological split: first 70% train, next 15% validation,
# final 15% held out for final testing.
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"Dataset rows: {n}")
print(f"Sensor features: {len(feature_columns)}")
print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(val)}")
print(f"Final test rows: {len(test)}")

results = []

for target in TARGETS:
    y_train = train[target]
    y_test = test[target]

    # Baseline 1: predict the training-set mean.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(train[feature_columns], y_train)
    baseline_pred = baseline.predict(test[feature_columns])

    # Baseline 2: use cut number alone.
    cut_model = LinearRegression()
    cut_model.fit(train[["cut"]], y_train)
    cut_pred = cut_model.predict(test[["cut"]])

    # Model 1: linear regression using sensor features.
    linear_model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    linear_model.fit(train[feature_columns], y_train)
    linear_pred = linear_model.predict(test[feature_columns])

    # Model 2: random forest using sensor features.
    forest_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    forest_model.fit(train[feature_columns], y_train)
    forest_pred = forest_model.predict(test[feature_columns])

    models = {
        "Training Mean Baseline": baseline_pred,
        "Cut Number Baseline": cut_pred,
        "Linear Regression (Sensors)": linear_pred,
        "Random Forest (Sensors)": forest_pred,
    }

    for model_name, predictions in models.items():
        results.append({
            "target": target,
            "model": model_name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
            "test_rows": len(test),
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["target", "MAE"])

output_path = RESULTS_DIR / "phm2010_model_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nFINAL TEST RESULTS")
print(results_df.round(3).to_string(index=False))
print(f"\nResults saved to: {output_path}")
print("\nImportant: the final test set was not used to fit the models.")
