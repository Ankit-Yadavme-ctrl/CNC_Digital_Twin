from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]

def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    feature_columns = [
        c for c in df.columns
        if c not in ["cut", "source_file", "n_samples", *TARGETS]
    ]

    if len(df) != 315 or len(feature_columns) != 42:
        raise ValueError(
            f"Unexpected dataset: {len(df)} rows, "
            f"{len(feature_columns)} sensor features."
        )

    if df[feature_columns + TARGETS].isna().any().any():
        raise ValueError("Missing features or wear labels detected.")

    if not np.isfinite(df[feature_columns + TARGETS].to_numpy()).all():
        raise ValueError("Non-finite values detected.")

    # Chronological train, validation, and untouched final test.
    train = df[df["cut"] <= 220]
    validation = df[df["cut"].between(221, 267)]
    test = df[df["cut"].between(268, 315)]

    candidates = {
        "Training Mean": (
            lambda: DummyRegressor(strategy="mean"),
            [],
        ),
        "Cut Number Linear": (
            lambda: LinearRegression(),
            ["cut"],
        ),
        "Sensor Ridge": (
            lambda: make_pipeline(
                StandardScaler(),
                Ridge(alpha=10.0),
            ),
            feature_columns,
        ),
        "Sensor Random Forest": (
            lambda: RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            feature_columns,
        ),
        "Sensor Extra Trees": (
            lambda: ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            feature_columns,
        ),
    }

    validation_rows = []
    test_rows = []
    prediction_rows = []

    print(f"Dataset rows: {len(df)}")
    print(f"Sensor features: {len(feature_columns)}")

    for target in TARGETS:
        validation_scores = []

        for name, (factory, columns) in candidates.items():
            model = factory()
            actual_columns = columns if columns else feature_columns

            model.fit(train[actual_columns], train[target])
            val_pred = model.predict(validation[actual_columns])

            scores = calculate_metrics(validation[target], val_pred)

            validation_rows.append({
                "target": target,
                "model": name,
                **scores,
            })

            validation_scores.append(
                (scores["MAE"], name, factory, actual_columns)
            )

        # Choose the model using validation MAE only.
        _, selected_name, selected_factory, selected_columns = min(
            validation_scores, key=lambda item: item[0]
        )

        # Refit on training + validation data.
        development = pd.concat([train, validation]).sort_values("cut")
        final_model = selected_factory()
        final_model.fit(development[selected_columns], development[target])

        # Evaluate once on the final 48 cuts.
        test_pred = final_model.predict(test[selected_columns])
        scores = calculate_metrics(test[target], test_pred)

        test_rows.append({
            "target": target,
            "selected_model": selected_name,
            **scores,
            "test_cuts": len(test),
        })

        for cut, actual, predicted in zip(
            test["cut"], test[target], test_pred
        ):
            prediction_rows.append({
                "target": target,
                "cut": int(cut),
                "actual_wear": float(actual),
                "predicted_wear": float(predicted),
                "selected_model": selected_name,
            })

        print(f"{target}: selected {selected_name}")

    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    validation_file = RESULTS_DIR / "corrected_validation_scores.csv"
    test_file = RESULTS_DIR / "corrected_final_test_scores.csv"
    predictions_file = RESULTS_DIR / "corrected_final_test_predictions.csv"

    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    print("\nVALIDATION RESULTS")
    print(validation_df.sort_values(["target", "MAE"]).round(3).to_string(index=False))

    print("\nFINAL TEST RESULTS")
    print(test_df.round(3).to_string(index=False))

    print("\nFiles saved:")
    print(validation_file)
    print(test_file)
    print(predictions_file)

    print("\nNote: Poor test scores mean these predictions are not yet reliable.")


if __name__ == "__main__":
    main()