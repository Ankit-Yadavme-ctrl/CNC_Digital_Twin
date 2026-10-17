
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "processed" / "phm2010_c1_features.csv"
OUTPUT_DIR = PROJECT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE).sort_values("cut")

df["average_wear"] = df[["flute_1", "flute_2", "flute_3"]].mean(axis=1)

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

signals = [
    ("channel_1_rms", "Channel 1 RMS"),
    ("channel_2_rms", "Channel 2 RMS"),
    ("channel_3_rms", "Channel 3 RMS"),
    ("average_wear", "Average tool wear"),
]

for ax, (column, title) in zip(axes, signals):
    ax.plot(df["cut"], df[column])
    ax.axvline(252, linestyle="--", label="Train/test boundary")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Cut number")
fig.suptitle("Sensor Distribution Shift and Tool Wear — PHM 2010 C1")
fig.tight_layout()

output_file = OUTPUT_DIR / "phm2010_sensor_shift.png"
fig.savefig(output_file, dpi=160)
print(f"Saved graph: {output_file}")
plt.show()
