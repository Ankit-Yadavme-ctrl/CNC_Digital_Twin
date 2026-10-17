
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]


def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def fit_predict_linear(train, future, target):
    model = LinearRegression()
    model.fit(train[["cut"]], train[target])
    return model.predict(future[["cut"]])



def fit_predict_piecewise(train, future, target):
    boundaries = [0, 110, 220, 315]
    predictions = np.full(len(future), np.nan, dtype=float)

    for i in range(3):
        lower = boundaries[i]
        upper = boundaries[i + 1]

        region_train = train[
            (train["cut"] > lower) & (train["cut"] <= upper)
        ]

        region_future_mask = (
            (future["cut"] > lower) & (future["cut"] <= upper)
        )

        if not region_future_mask.any():
            continue

        if region_train.empty:
            previous_lower = boundaries[i - 1] if i > 0 else 0
            region_train = train[
                (train["cut"] > previous_lower)
                & (train["cut"] <= lower)
            ]

        if region_train.empty:
            raise ValueError(f"No training data for region {i + 1}")

        model = LinearRegression()
        model.fit(region_train[["cut"]], region_train[target])

        predictions[region_future_mask.to_numpy()] = model.predict(
            future.loc[region_future_mask, ["cut"]]
        )

    if np.isnan(predictions).any():
        raise ValueError("Some future cuts were not assigned a region.")

    return predictions


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    if len(df) != 315:
        raise ValueError(f"Expected 315 cuts; found {len(df)}.")

    train = df[df["cut"] <= 220].copy()
    validation = df[df["cut"].between(221, 267)].copy()
    test = df[df["cut"].between(268, 315)].copy()

    validation_results = []
    test_results = []
    prediction_rows = []

    for target in TARGETS:
        # Compare on the same validation period.
        linear_val = fit_predict_linear(train, validation, target)
        piecewise_val = fit_predict_piecewise(train, validation, target)

        linear_mae = mean_absolute_error(
            validation[target], linear_val
        )
        piecewise_mae = mean_absolute_error(
            validation[target], piecewise_val
        )

        validation_results.extend([
            {
                "target": target,
                "model": "Linear",
                **metrics(validation[target], linear_val),
            },
            {
                "target": target,
                "model": "Piecewise Linear",
                **metrics(validation[target], piecewise_val),
            },
        ])

        # Select using validation MAE only.
        if piecewise_mae < linear_mae:
            selected_name = "Piecewise Linear"
            predictor = fit_predict_piecewise
        else:
            selected_name = "Linear"
            predictor = fit_predict_linear

        # Refit using all development data, then test once.
        development = pd.concat(
            [train, validation]
        ).sort_values("cut")

        test_pred = predictor(development, test, target)
        scores = metrics(test[target], test_pred)

        test_results.append({
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

        print(
            f"{target}: validation selected {selected_name} "
            f"(linear MAE={linear_mae:.3f}, "
            f"piecewise MAE={piecewise_mae:.3f})"
        )

    validation_df = pd.DataFrame(validation_results)
    test_df = pd.DataFrame(test_results)
    predictions_df = pd.DataFrame(prediction_rows)

    validation_file = RESULTS_DIR / "piecewise_validation_scores.csv"
    test_file = RESULTS_DIR / "piecewise_final_test_scores.csv"
    predictions_file = RESULTS_DIR / "piecewise_final_test_predictions.csv"

    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    print("\nVALIDATION RESULTS")
    print(
        validation_df.sort_values(["target", "MAE"])
        .round(3).to_string(index=False)
    )

    print("\nFINAL TEST RESULTS")
    print(test_df.round(3).to_string(index=False))

    print("\nSaved files:")
    print(validation_file)
    print(test_file)
    print(predictions_file)

    print(
        "\nNote: The piecewise model uses cut number only, "
        "not live sensor signals."
    )


if __name__ == "__main__":
    main()
