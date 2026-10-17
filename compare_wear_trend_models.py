
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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


def make_model(degree):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    if len(df) != 315:
        raise ValueError(f"Expected 315 cuts, found {len(df)}")

    # Train: cuts 1-220
    # Validation: cuts 221-267
    # Final test: cuts 268-315
    train = df[df["cut"] <= 220]
    validation = df[df["cut"].between(221, 267)]
    test = df[df["cut"].between(268, 315)]

    results = []
    predictions = []

    print("Linear vs Quadratic Wear Trend")
    print(f"Training cuts: {len(train)}")
    print(f"Validation cuts: {len(validation)}")
    print(f"Final test cuts: {len(test)}")

    for target in TARGETS:
        validation_scores = []

        for degree, name in [
            (1, "Linear"),
            (2, "Quadratic"),
        ]:
            model = make_model(degree)
            model.fit(train[["cut"]], train[target])

            val_pred = model.predict(validation[["cut"]])
            val_scores = metrics(validation[target], val_pred)

            results.append({
                "target": target,
                "model": name,
                "evaluation": "validation",
                **val_scores,
            })

            validation_scores.append(
                (val_scores["MAE"], degree, name)
            )

        # Select model using validation MAE only.
        _, selected_degree, selected_name = min(
            validation_scores, key=lambda row: row[0]
        )

        # Refit selected model using train + validation.
        development = pd.concat([train, validation])
        final_model = make_model(selected_degree)
        final_model.fit(development[["cut"]], development[target])

        test_pred = final_model.predict(test[["cut"]])
        test_scores = metrics(test[target], test_pred)

        results.append({
            "target": target,
            "model": selected_name,
            "evaluation": "final_test",
            **test_scores,
        })

        for cut, actual, predicted in zip(
            test["cut"], test[target], test_pred
        ):
            predictions.append({
                "target": target,
                "cut": int(cut),
                "actual_wear": float(actual),
                "predicted_wear": float(predicted),
                "selected_model": selected_name,
            })

        print(f"{target}: selected {selected_name}")

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions)

    results_file = RESULTS_DIR / "linear_quadratic_comparison.csv"
    predictions_file = RESULTS_DIR / "linear_quadratic_predictions.csv"

    results_df.to_csv(results_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    print("\nRESULTS")
    print(results_df.round(3).to_string(index=False))

    print(f"\nSaved results: {results_file}")
    print(f"Saved predictions: {predictions_file}")
    print("\nFinal test scores are for cuts 268-315.")
    print("These are trend models, not sensor-based predictions.")


if __name__ == "__main__":
    main()
