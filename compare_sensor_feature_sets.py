
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]

# Small, interpretable feature groups selected from the earlier
# training-only correlation analysis.
FEATURE_SETS = {
    "All 42 sensors": None,
    "Force features": [
        "force_y_N_max",
        "force_y_N_rms",
        "force_z_N_mean",
        "force_z_N_max",
    ],
    "Vibration features": [
        "vibration_x_g_rms",
        "vibration_y_g_rms",
        "vibration_z_g_rms",
    ],
    "Force and vibration": [
        "force_y_N_max",
        "force_z_N_mean",
        "force_z_N_max",
        "vibration_x_g_rms",
        "vibration_y_g_rms",
        "vibration_z_g_rms",
    ],
    "Acoustic emission": [
        "ae_rms_V_mean",
        "ae_rms_V_abs_mean",
        "ae_rms_V_rms",
        "ae_rms_V_std",
    ],
    "Candidate per flute": None,
}

EXCLUDED = {
    "cut", "source_file", "n_samples",
    "flute_1", "flute_2", "flute_3"
}


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def main():
    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    if len(df) != 315:
        raise ValueError(f"Expected 315 rows, found {len(df)}")

    all_features = [c for c in df.columns if c not in EXCLUDED]

    train = df[df["cut"] <= 220].copy()
    validation = df[df["cut"].between(221, 267)].copy()
    development = df[df["cut"] <= 267].copy()
    test = df[df["cut"].between(268, 315)].copy()

    candidate_sets = dict(FEATURE_SETS)

    candidate_sets["Candidate per flute"] = {
        "flute_1": [
            "ae_rms_V_abs_mean",
            "ae_rms_V_mean",
            "force_y_N_max",
            "force_y_N_rms",
        ],
        "flute_2": [
            "force_y_N_max",
            "force_z_N_mean",
            "force_z_N_max",
        ],
        "flute_3": [
            "force_z_N_mean",
            "force_y_N_max",
            "force_z_N_max",
        ],
    }

    rows = []
    prediction_rows = []

    for target in TARGETS:
        for set_name, definition in candidate_sets.items():
            if set_name == "All 42 sensors":
                features = all_features
            elif set_name == "Candidate per flute":
                features = definition[target]
            else:
                features = definition

            missing = sorted(set(features) - set(df.columns))
            if missing:
                raise ValueError(
                    f"Missing features for {set_name}: {missing}"
                )

            # Fit on training cuts and select feature set on validation.
            model = make_model()
            model.fit(train[features], train[target])
            validation_pred = model.predict(validation[features])

            validation_score = metrics(
                validation[target], validation_pred
            )

            # Refit the same feature set on development cuts, then test.
            final_model = make_model()
            final_model.fit(development[features], development[target])
            test_pred = final_model.predict(test[features])

            test_score = metrics(test[target], test_pred)

            rows.append({
                "target": target,
                "feature_set": set_name,
                "feature_count": len(features),
                "features": ", ".join(features),
                "validation_MAE": validation_score["MAE"],
                "validation_RMSE": validation_score["RMSE"],
                "validation_R2": validation_score["R2"],
                "test_MAE": test_score["MAE"],
                "test_RMSE": test_score["RMSE"],
                "test_R2": test_score["R2"],
            })

            for cut, actual, predicted in zip(
                test["cut"], test[target], test_pred
            ):
                prediction_rows.append({
                    "target": target,
                    "feature_set": set_name,
                    "cut": int(cut),
                    "actual_wear": float(actual),
                    "predicted_wear": float(predicted),
                })

    results = pd.DataFrame(rows)

    # This validation-only ranking determines which feature set to prefer.
    results["validation_rank"] = results.groupby("target")[
        "validation_MAE"
    ].rank(method="min")

    results_path = RESULTS_DIR / "sensor_feature_set_comparison.csv"
    predictions_path = RESULTS_DIR / "sensor_feature_set_predictions.csv"

    results.to_csv(results_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    for target in TARGETS:
        subset = results[results["target"] == target].sort_values(
            "validation_MAE"
        )

        print(f"\n{target}: RANKED BY VALIDATION MAE")
        print(
            subset[[
                "feature_set", "feature_count",
                "validation_MAE", "validation_R2",
                "test_MAE", "test_R2"
            ]].round(3).to_string(index=False)
        )

    print("\nSaved:")
    print(results_path)
    print(predictions_path)
    print(
        "\nUse validation MAE to choose feature sets. "
        "Test results are for final evaluation, not further tuning."
    )


if __name__ == "__main__":
    main()
