
from pathlib import Path
import pandas as pd

folder = Path("data/raw/phm2010/c1")

files = [
    "c_1_305.csv",
    "c_1_309.csv",
    "c_1_315.csv",
]

channel_names = [
    "channel_1", "channel_2", "channel_3",
    "channel_4", "channel_5", "channel_6", "channel_7",
]

for filename in files:
    path = folder / filename
    data = pd.read_csv(path, header=None, names=channel_names)

    print(f"\n{'=' * 20} {filename} {'=' * 20}")
    print("Rows:", len(data))
    print(data.agg(["mean", "std", "min", "median", "max"]).round(3))
