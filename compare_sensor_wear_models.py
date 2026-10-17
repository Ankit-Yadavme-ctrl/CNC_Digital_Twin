
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def make_models():
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Extra Trees": Pipeline([
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
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    if len(df) != 315:
        raise ValueError(f"Expected 315 rows; found {len(df)}")

    feature_columns = [
        column for column in df.columns
        if column not in EXCLUDED
    ]

    if len(feature_columns) != 42:
        raise ValueError(
            f"Expected 42 sensor features; found {len(feature_columns)}"
        )

    if df[TARGETS].isna().any().any():
        raise ValueError("Wear labels contain missing values.")

    train = df[df["cut"] <= 220]
    validation = df[df["cut"].between(221, 267)]
    test = df[df["cut"].between(268, 315)]

    X_train = train[feature_columns]
    X_validation = validation[feature_columns]
    X_test = test[feature_columns]

    validation_results = []
    test_results = []
    prediction_rows = []

    for target in TARGETS:
        models = make_models()
        validation_scores = {}

        for name, model in models.items():
            model.fit(X_train, train[target])
            prediction = model.predict(X_validation)

            scores = calculate_metrics(
                validation[target], prediction
            )
            validation_scores[name] = scores["MAE"]

            validation_results.append({
                "target": target,
                "model": name,
                **scores,
            })

        # Choose model by validation MAE only.
        selected_name = min(
            validation_scores, key=validation_scores.get
        )

        # Refit selected model on train + validation.
        development = pd.concat(
            [train, validation]
        ).sort_values("cut")

        selected_model = make_models()[selected_name]
        selected_model.fit(
            development[feature_columns],
            development[target],
        )

        test_prediction = selected_model.predict(X_test)
        test_scores = calculate_metrics(
            test[target], test_prediction
        )

        test_results.append({
            "target": target,
            "selected_model": selected_name,
            **test_scores,
            "test_cuts": len(test),
        })

        for cut, actual, predicted in zip(
            test["cut"], test[target], test_prediction
        ):
            prediction_rows.append({
                "target": target,
                "cut": int(cut),
                "actual_wear": float(actual),
                "predicted_wear": float(predicted),
                "selected_model": selected_name,
            })

        print(
            f"{target}: selected {selected_name} "
            f"using validation MAE "
            f"{validation_scores[selected_name]:.3f}"
        )

    validation_df = pd.DataFrame(validation_results)
    test_df = pd.DataFrame(test_results)
    predictions_df = pd.DataFrame(prediction_rows)

    validation_path = RESULTS_DIR / "sensor_model_validation.csv"
    test_path = RESULTS_DIR / "sensor_model_final_test.csv"
    predictions_path = RESULTS_DIR / "sensor_model_test_predictions.csv"

    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nVALIDATION RESULTS")
    print(
        validation_df.sort_values(["target", "MAE"])
        .round(3).to_string(index=False)
    )

    print("\nFINAL TEST RESULTS")
    print(test_df.round(3).to_string(index=False))

    print("\nSaved files:")
    print(validation_path)
    print(test_path)
    print(predictions_path)

    print(
        "\nImportant: The final test is held out from model selection. "
        "These are PHM2010 sensor-feature experiments, not live "
        "CNC sensor predictions."
    )


if __name__ == "__main__":
    main()
