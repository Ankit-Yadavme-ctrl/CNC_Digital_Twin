
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "processed" / "phm2010_c1_features.csv"
OUTPUT_DIR = PROJECT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "phm2010_tool_wear_model.joblib"
METRICS_FILE = OUTPUT_DIR / "phm2010_model_metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "phm2010_test_predictions.csv"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    # Average flank wear across all three flutes.
    df["average_wear"] = df[["flute_1", "flute_2", "flute_3"]].mean(axis=1)

    # Only use sensor-derived features. Never use cut number, filename,
    # wear labels, or the target as model inputs.
    feature_columns = [
        c for c in df.columns
        if c.startswith("channel_")
        and c.rsplit("_", 1)[-1] in
        {"mean", "std", "rms", "min", "max", "abs_mean"}
    ]

    if len(feature_columns) != 42:
        raise ValueError(
            f"Expected 42 sensor features, found {len(feature_columns)}"
        )

    if df[feature_columns + ["average_wear"]].isna().any().any():
        raise ValueError("Missing values found in features or target.")

    X = df[feature_columns]
    y = df["average_wear"]

    # Chronological split: earlier cuts for training, later cuts for testing.
    split_index = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    predictions = pd.DataFrame({
        "cut": df["cut"].iloc[split_index:].to_numpy(),
        "actual_average_wear": y_test.to_numpy()
    })

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)

        results[name] = {
            "MAE": float(mae),
            "RMSE": rmse,
            "R2": float(r2)
        }

        predictions[name.replace(" ", "_").lower() + "_prediction"] = pred

        print(f"\n{name}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")

    # Select the lowest test MAE for this initial comparison.
    # This is exploratory, not a final independent benchmark.
    best_name = min(results, key=lambda name: results[name]["MAE"])
    best_model = models[best_name]

    joblib.dump({
        "model": best_model,
        "feature_columns": feature_columns,
        "target": "average_wear",
        "target_unit": "10^-3 mm",
        "training_cutter": "C1",
        "training_cuts": int(split_index),
        "test_cuts": int(len(df) - split_index),
        "selected_model": best_name
    }, MODEL_FILE)

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    report = {
        "dataset": str(DATA_FILE),
        "rows": int(len(df)),
        "features": feature_columns,
        "target": "mean(flute_1, flute_2, flute_3)",
        "target_unit": "10^-3 mm",
        "split": "chronological 80% train / 20% test",
        "train_cuts": int(split_index),
        "test_cuts": int(len(df) - split_index),
        "results": results,
        "best_by_test_MAE": best_name,
        "warning": (
            "Exploratory single-cutter evaluation. Test-based model selection "
            "can be optimistic; validate on other labeled cutters before "
            "claiming generalization."
        )
    }

    METRICS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTraining and comparison complete.")
    print(f"Selected model: {best_name}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved test predictions: {PREDICTIONS_FILE}")
    print("\nFirst five test predictions:")
    print(predictions.head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
