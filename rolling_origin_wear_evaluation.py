
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

# Every evaluation predicts a future block of 20 cuts.
INITIAL_TRAIN_END = 150
BLOCK_SIZE = 20


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def predict_linear(train, future, target):
    model = LinearRegression()
    model.fit(train[["cut"]], train[target])
    return model.predict(future[["cut"]])


def predict_piecewise(train, future, target):
    """
    Fit independent linear trends for cuts 1-110 and 111 onward.
    If a segment has too few observations, use all available
    training observations rather than fitting an unstable line.
    """
    predictions = np.empty(len(future), dtype=float)

    for position, (_, row) in enumerate(future.iterrows()):
        cut = row["cut"]

        if cut <= 110:
            segment = train[train["cut"] <= 110]
        else:
            segment = train[train["cut"] > 110]

        if len(segment) < 10:
            segment = train

        model = LinearRegression()
        model.fit(segment[["cut"]], segment[target])

        predictions[position] = model.predict(
            pd.DataFrame({"cut": [cut]})
        )[0]

    return predictions


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    if len(df) != 315 or df["cut"].duplicated().any():
        raise ValueError("Expected 315 rows with unique cut numbers.")

    if df[TARGETS].isna().any().any():
        raise ValueError("Wear labels contain missing values.")

    all_results = []
    all_predictions = []

    for target in TARGETS:
        block_start = INITIAL_TRAIN_END + 1

        while block_start <= int(df["cut"].max()):
            block_end = min(
                block_start + BLOCK_SIZE - 1,
                int(df["cut"].max()),
            )

            # Strict chronological split: training precedes prediction.
            train = df[df["cut"] < block_start].copy()
            future = df[
                df["cut"].between(block_start, block_end)
            ].copy()

            if train.empty or future.empty:
                break

            model_predictions = {
                "Linear": predict_linear(train, future, target),
                "Piecewise Linear": predict_piecewise(
                    train, future, target
                ),
            }

            for model_name, predicted in model_predictions.items():
                actual = future[target].to_numpy()
                scores = calculate_metrics(actual, predicted)

                all_results.append({
                    "target": target,
                    "model": model_name,
                    "train_end_cut": int(train["cut"].max()),
                    "prediction_start": int(future["cut"].min()),
                    "prediction_end": int(future["cut"].max()),
                    "n_predictions": len(future),
                    **scores,
                })

                for cut, actual_value, predicted_value in zip(
                    future["cut"], actual, predicted
                ):
                    all_predictions.append({
                        "target": target,
                        "model": model_name,
                        "train_end_cut": int(train["cut"].max()),
                        "cut": int(cut),
                        "actual_wear": float(actual_value),
                        "predicted_wear": float(predicted_value),
                    })

            block_start = block_end + 1

    results = pd.DataFrame(all_results)
    predictions = pd.DataFrame(all_predictions)

    results_path = RESULTS_DIR / "rolling_origin_block_scores.csv"
    predictions_path = RESULTS_DIR / "rolling_origin_predictions.csv"
    summary_path = RESULTS_DIR / "rolling_origin_summary.csv"

    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    summary = (
        results.groupby(["target", "model"])
        .agg(
            mean_block_MAE=("MAE", "mean"),
            mean_block_RMSE=("RMSE", "mean"),
            blocks=("MAE", "count"),
        )
        .reset_index()
    )

    # Also calculate pooled errors over all rolling predictions.
    pooled = (
        predictions.assign(
            absolute_error=lambda x: (
                x["actual_wear"] - x["predicted_wear"]
            ).abs(),
            squared_error=lambda x: (
                x["actual_wear"] - x["predicted_wear"]
            ) ** 2,
        )
        .groupby(["target", "model"])
        .agg(
            pooled_MAE=("absolute_error", "mean"),
            pooled_RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
    )

    summary = summary.merge(pooled, on=["target", "model"])
    summary.to_csv(summary_path, index=False)

    print("\nROLLING-ORIGIN BLOCK SCORES")
    print(results.round(3).to_string(index=False))

    print("\nOVERALL ROLLING-ORIGIN SUMMARY")
    print(summary.round(3).to_string(index=False))

    print("\nSaved files:")
    print(results_path)
    print(predictions_path)
    print(summary_path)

    print(
        "\nNote: These models use cut number only, not sensor features. "
        "This evaluates wear-label trend prediction, not a sensor-based "
        "real-time predictive-maintenance system."
    )


if __name__ == "__main__":
    main()
