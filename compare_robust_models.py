
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "processed" / "phm2010_c1_features.csv"
OUTPUT_FILE = PROJECT / "reports" / "phm2010_robust_model_comparison.csv"

def metrics(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }

def main():
    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)
    df["average_wear"] = df[["flute_1", "flute_2", "flute_3"]].mean(axis=1)

    features = [
        c for c in df.columns
        if c.startswith("channel_")
        and c.rsplit("_", 1)[-1]
        in {"mean", "std", "rms", "min", "max", "abs_mean"}
    ]

    windows = [(126, 189), (189, 252), (252, 315)]
    results = []

    for train_end, test_end in windows:
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]

        candidates = {
            "Ridge (sensors)": (
                make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                features,
            ),
            "Random Forest (sensors)": (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
                features,
            ),
            "Extra Trees (sensors)": (
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
                features,
            ),
            "Cut number only": (
                make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                ["cut"],
            ),
        }

        for name, (model, columns) in candidates.items():
            model.fit(train[columns], train["average_wear"])
            pred = model.predict(test[columns])

            results.append({
                "train_cuts": f"1-{train_end}",
                "test_cuts": f"{train_end + 1}-{test_end}",
                "model": name,
                **metrics(test["average_wear"], pred),
            })

        print(f"Finished test window {train_end + 1}-{test_end}")

    result_df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nRobust-model comparison:")
    print(result_df.round(3).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
