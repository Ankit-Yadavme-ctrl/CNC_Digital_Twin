
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
PREDICTIONS_FILE = (
    PROJECT / "reports" / "phm2010_validation_predictions.csv"
)
OUTPUT_DIR = PROJECT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS_FILE)

models = [
    "Cut number only",
    "Sensors only",
    "Cut number + sensors",
    "Random Forest (sensors)",
]

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

windows = [
    (126, 189),
    (189, 252),
    (252, 315),
]

for ax, (train_end, test_end) in zip(axes, windows):
    period = df[
        (df["train_end"] == train_end)
        & (df["test_cut"] <= test_end)
    ]

    actual = (
        period[["test_cut", "actual_average_wear"]]
        .drop_duplicates("test_cut")
        .sort_values("test_cut")
    )

    ax.plot(
        actual["test_cut"],
        actual["actual_average_wear"],
        label="Actual wear",
        linewidth=2.5,
    )

    for model_name in models:
        subset = period[period["model"] == model_name].sort_values("test_cut")

        if not subset.empty:
            ax.plot(
                subset["test_cut"],
                subset["predicted_average_wear"],
                label=model_name,
                alpha=0.85,
            )

    ax.set_title(f"Train cuts 1–{train_end}; test cuts {train_end + 1}–{test_end}")
    ax.set_ylabel("Average wear (dataset units)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Cut number")
fig.suptitle("PHM 2010 C1: Actual vs Predicted Tool Wear", fontsize=14)
fig.tight_layout()

output_file = OUTPUT_DIR / "phm2010_actual_vs_predicted.png"
fig.savefig(output_file, dpi=160)
print(f"Plot saved to: {output_file}")

plt.show()
