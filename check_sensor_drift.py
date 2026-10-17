
from pathlib import Path

import numpy as np
import pandas as pd

path = (
    Path(__file__).resolve().parent
    / "data_processed"
    / "phm2010_c1_features.csv"
)

df = pd.read_csv(path).sort_values("cut").reset_index(drop=True)

features = [
    c for c in df.columns
    if c not in ["cut", "source_file", "n_samples",
                 "flute_1", "flute_2", "flute_3"]
]

train = df.iloc[:220]
test = df.iloc[267:315]

rows = []

for feature in features:
    train_mean = train[feature].mean()
    train_std = train[feature].std()

    test_mean = test[feature].mean()

    # Standardized mean difference; avoid division by zero.
    if pd.isna(train_std) or train_std == 0:
        difference = np.nan
    else:
        difference = (test_mean - train_mean) / train_std

    rows.append({
        "feature": feature,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "standardized_difference": difference
    })

result = pd.DataFrame(rows)
result["abs_difference"] = result["standardized_difference"].abs()
result = result.sort_values("abs_difference", ascending=False)

print("TOP 15 SENSOR FEATURES WITH THE LARGEST DISTRIBUTION SHIFT")
print(result.head(15).round(3).to_string(index=False))

print("\nFull report saved to results/sensor_drift_report.csv")
output = Path(__file__).resolve().parent / "results" / "sensor_drift_report.csv"
output.parent.mkdir(exist_ok=True)
result.to_csv(output, index=False)
