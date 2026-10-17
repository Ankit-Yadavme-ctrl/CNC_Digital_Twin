
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "processed" / "phm2010_c1_features.csv"
OUTPUT_DIR = PROJECT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE).sort_values("cut")
df["average_wear"] = df[["flute_1", "flute_2", "flute_3"]].mean(axis=1)

plt.figure(figsize=(12, 6))

for column in ["flute_1", "flute_2", "flute_3"]:
    plt.plot(df["cut"], df[column], label=column)

plt.plot(
    df["cut"],
    df["average_wear"],
    label="Average wear",
    linewidth=2.5
)

plt.axvline(252, linestyle="--", label="Train/test boundary")
plt.xlabel("Cut number")
plt.ylabel("Wear measurement (verify units in dataset documentation)")
plt.title("PHM 2010 C1 — Tool Wear Across 315 Cuts")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = OUTPUT_DIR / "phm2010_c1_wear_trend.png"
plt.savefig(output_file, dpi=160)
print(f"Plot saved to: {output_file}")

plt.show()
