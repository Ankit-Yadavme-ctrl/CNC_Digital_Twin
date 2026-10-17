
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]
EXCLUDED = [
    "cut", "source_file", "n_samples",
    "flute_1", "flute_2", "flute_3"
]


def main():
    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)
    features = [c for c in df.columns if c not in EXCLUDED]

    train = df[df["cut"] <= 220].copy()
    test = df[df["cut"].between(268, 315)].copy()

    # Standardized mean differences: training versus test.
    drift_rows = []

    for feature in features:
        train_values = train[feature].dropna()
        test_values = test[feature].dropna()

        pooled_std = np.sqrt(
            (train_values.var(ddof=1) + test_values.var(ddof=1)) / 2
        )

        if pooled_std == 0 or pd.isna(pooled_std):
            difference = 0.0
        else:
            difference = (
                test_values.mean() - train_values.mean()
            ) / pooled_std

        drift_rows.append({
            "feature": feature,
            "train_mean": train_values.mean(),
            "test_mean": test_values.mean(),
            "standardized_mean_difference": difference,
            "absolute_drift": abs(difference),
        })

    drift = pd.DataFrame(drift_rows).sort_values(
        "absolute_drift", ascending=False
    )

    drift_path = RESULTS_DIR / "sensor_drift_importance.csv"
    drift.to_csv(drift_path, index=False)

    # Feature importance is measured on training data only.
    importance_rows = []

    for target in TARGETS:
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        model.fit(train[features], train[target])

        importances = model.named_steps["model"].feature_importances_

        for feature, importance in zip(features, importances):
            importance_rows.append({
                "target": target,
                "feature": feature,
                "training_feature_importance": importance,
            })

    importance = pd.DataFrame(importance_rows)
    importance_path = RESULTS_DIR / "sensor_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\nTOP 15 FEATURES BY DISTRIBUTION DRIFT")
    print(
        drift[[
            "feature", "train_mean", "test_mean",
            "standardized_mean_difference"
        ]].head(15).round(3).to_string(index=False)
    )

    print("\nTOP 10 TRAINING FEATURE IMPORTANCES PER FLUTE")
    for target in TARGETS:
        print(f"\n{target}")
        subset = importance[importance["target"] == target]
        subset = subset.sort_values(
            "training_feature_importance", ascending=False
        )
        print(
            subset[["feature", "training_feature_importance"]]
            .head(10).round(4).to_string(index=False)
        )

    print("\nSaved files:")
    print(drift_path)
    print(importance_path)

    print(
        "\nImportant: drift measures distribution changes, not causation. "
        "Tree feature importance measures model reliance, not proof "
        "that a feature causes wear."
    )


if __name__ == "__main__":
    main()
