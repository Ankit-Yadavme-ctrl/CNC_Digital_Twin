
from pathlib import Path

import pandas as pd

base = Path(__file__).resolve().parent
data_path = base / "data_processed" / "phm2010_c1_features.csv"

df = pd.read_csv(data_path).sort_values("cut").reset_index(drop=True)

columns = [
    "cut",
    "vibration_x_g_rms",
    "force_x_N_std",
    "force_x_N_rms",
    "vibration_y_g_rms",
    "flute_1",
    "flute_2",
    "flute_3",
]

print("SELECTED FEATURES AT KEY CUTS")
print(df.loc[df["cut"].isin([1, 50, 100, 150, 200, 220, 240, 267, 268, 290, 315]), columns]
      .round(3).to_string(index=False))

print("\nAVERAGES BY CUT PERIOD")
df["period"] = pd.cut(
    df["cut"],
    bins=[0, 55, 110, 165, 220, 267, 315],
    labels=["1-55", "56-110", "111-165", "166-220", "221-267", "268-315"]
)

summary = df.groupby("period", observed=True)[
    ["vibration_x_g_rms", "force_x_N_std", "force_x_N_rms",
     "vibration_y_g_rms", "flute_1", "flute_2", "flute_3"]
].mean()

print(summary.round(3).to_string())
