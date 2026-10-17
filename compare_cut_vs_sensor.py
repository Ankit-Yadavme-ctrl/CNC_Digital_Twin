
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]
EXCLUDED = [
    "cut", "source_file", "n_samples",
    "flute_1", "flute_2", "flute_3"
]


def score(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def make_sensor_models():
    return {
        "Ridge (sensors)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest (sensors)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Extra Trees (sensors)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    if len(df) != 315:
        raise ValueError(f"Expected 315 rows, found {len(df)}")

    features = [c for c in df.columns if c not in EXCLUDED]

    train = df[df["cut"] <= 220].copy()
    validation = df[df["cut"].between(221, 267)].copy()
    development = df[df["cut"] <= 267].copy()
    test = df[df["cut"].between(268, 315)].copy()

    rows = []
    predictions = []

    for target in TARGETS:
        models = make_sensor_models()
        validation_mae = {}

        # Select the sensor model using validation only.
        for name, model in models.items():
            model.fit(train[features], train[target])
            pred = model.predict(validation[features])
            validation_mae[name] = mean_absolute_error(
                validation[target], pred
            )

        best_sensor_name = min(validation_mae, key=validation_mae.get)

        # Baseline 1: fit a linear trend using cut number.
        cut_model = LinearRegression()
        cut_model.fit(train[["cut"]], train[target])
        cut_val_pred = cut_model.predict(validation[["cut"]])
        cut_validation_mae = mean_absolute_error(
            validation[target], cut_val_pred
        )

        # Refit both selected approaches on cuts 1-267.
        cut_model.fit(development[["cut"]], development[target])
        cut_test_pred = cut_model.predict(test[["cut"]])

        sensor_model = make_sensor_models()[best_sensor_name]
        sensor_model.fit(development[features], development[target])
        sensor_test_pred = sensor_model.predict(test[features])

        # Baseline 2: predict the training-development mean.
        mean_test_pred = np.full(
            len(test), development[target].mean()
        )

        approaches = {
            "Cut-number Linear": cut_test_pred,
            best_sensor_name: sensor_test_pred,
            "Development Mean": mean_test_pred,
        }

        for name, pred in approaches.items():
            rows.append({
                "target": target,
                "model": name,
                "validation_MAE": (
                    cut_validation_mae
                    if name == "Cut-number Linear"
                    else validation_mae.get(name, np.nan)
                ),
                **score(test[target], pred),
                "test_cuts": len(test),
            })

            for cut, actual, predicted in zip(
                test["cut"], test[target], pred
            ):
                predictions.append({
                    "target": target,
                    "model": name,
                    "cut": int(cut),
                    "actual_wear": float(actual),
                    "predicted_wear": float(predicted),
                })

        print(
            f"{target}: best sensor model = {best_sensor_name}; "
            f"sensor validation MAE = {validation_mae[best_sensor_name]:.3f}; "
            f"cut-number validation MAE = {cut_validation_mae:.3f}"
        )

    results = pd.DataFrame(rows)
    prediction_df = pd.DataFrame(predictions)

    results_path = RESULTS_DIR / "cut_vs_sensor_test_comparison.csv"
    predictions_path = RESULTS_DIR / "cut_vs_sensor_test_predictions.csv"

    results.to_csv(results_path, index=False)
    prediction_df.to_csv(predictions_path, index=False)

    print("\nSAME-TEST-CUT COMPARISON")
    print(
        results.sort_values(["target", "MAE"])
        .round(3).to_string(index=False)
    )

    print("\nSaved files:")
    print(results_path)
    print(predictions_path)

    print(
        "\nAll approaches are evaluated on cuts 268-315. "
        "Model selection uses validation cuts 221-267."
    )


if __name__ == "__main__":
    main()
