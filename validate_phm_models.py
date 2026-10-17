
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "processed" / "phm2010_c1_features.csv"
REPORT_DIR = PROJECT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)
    df["average_wear"] = df[["flute_1", "flute_2", "flute_3"]].mean(axis=1)

    feature_columns = [
        c for c in df.columns
        if c.startswith("channel_")
        and c.rsplit("_", 1)[-1]
        in {"mean", "std", "rms", "min", "max", "abs_mean"}
    ]

    if df[feature_columns + ["average_wear"]].isna().any().any():
        raise ValueError("Missing values detected.")

    # Expanding chronological windows: each test period comes after its training period.
    windows = [
        (126, 189),
        (189, 252),
        (252, 315),
    ]

    rows = []
    prediction_rows = []

    for train_end, test_end in windows:
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]

        model_specs = [
            ("Mean baseline", ["average_wear"], "mean"),
            ("Cut number only", ["cut"], "linear"),
            ("Sensors only", feature_columns, "linear"),
            ("Cut number + sensors", ["cut"] + feature_columns, "linear"),
        ]

        for name, columns, kind in model_specs:
            if kind == "mean":
                model = DummyRegressor(strategy="mean")
            else:
                model = LinearRegression()

            X_train = train[columns]
            X_test = test[columns]
            y_train = train["average_wear"]
            y_test = test["average_wear"]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            scores = evaluate(y_test, pred)
            rows.append({
                "train_cuts": f"1-{train_end}",
                "test_cuts": f"{train_end + 1}-{test_end}",
                "model": name,
                **scores,
            })

            for cut, actual, estimate in zip(test["cut"], y_test, pred):
                prediction_rows.append({
                    "train_end": train_end,
                    "test_cut": int(cut),
                    "model": name,
                    "actual_average_wear": float(actual),
                    "predicted_average_wear": float(estimate),
                })

        # Random Forest with the sensor features only
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[feature_columns], train["average_wear"])
        pred = model.predict(test[feature_columns])

        rows.append({
            "train_cuts": f"1-{train_end}",
            "test_cuts": f"{train_end + 1}-{test_end}",
            "model": "Random Forest (sensors)",
            **evaluate(test["average_wear"], pred),
        })

        for cut, actual, estimate in zip(test["cut"], test["average_wear"], pred):
            prediction_rows.append({
                "train_end": train_end,
                "test_cut": int(cut),
                "model": "Random Forest (sensors)",
                "actual_average_wear": float(actual),
                "predicted_average_wear": float(estimate),
            })

        print(f"Finished training through cut {train_end}; tested through {test_end}.")

    results = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)

    results_file = REPORT_DIR / "phm2010_chronological_validation.csv"
    predictions_file = REPORT_DIR / "phm2010_validation_predictions.csv"

    results.to_csv(results_file, index=False)
    predictions.to_csv(predictions_file, index=False)

    print("\nValidation results (lower MAE/RMSE is better; higher R2 is better):")
    print(results.round(3).to_string(index=False))
    print(f"\nResults saved to: {results_file}")
    print(f"Predictions saved to: {predictions_file}")
    print("\nNote: These are chronological tests on cutter C1 only.")

if __name__ == "__main__":
    main()
