
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "processed" / "phm2010_c1_features.csv"
OUTPUT_DIR = PROJECT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

features = [
    c for c in df.columns
    if c.startswith("channel_")
    and c.rsplit("_", 1)[-1]
    in {"mean", "std", "rms", "min", "max", "abs_mean"}
]

train = df.iloc[:252]
test = df.iloc[252:]

rows = []

for feature in features:
    train_mean = train[feature].mean()
    test_mean = test[feature].mean()
    train_std = train[feature].std()
    test_std = test[feature].std()

    rows.append({
        "feature": feature,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "train_std": train_std,
        "test_std": test_std,
        "mean_difference": test_mean - train_mean,
        "absolute_mean_difference": abs(test_mean - train_mean),
    })

result = pd.DataFrame(rows).sort_values(
    "absolute_mean_difference", ascending=False
)

output_file = OUTPUT_DIR / "phm2010_feature_distribution_comparison.csv"
result.to_csv(output_file, index=False)

print("Largest train/test feature-mean differences:")
print(result.head(15).round(4).to_string(index=False))
print(f"\nFull report saved to: {output_file}")
