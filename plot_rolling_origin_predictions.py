
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "results" / "rolling_origin_predictions.csv"
OUTPUT_DIR = BASE_DIR / "results" / "rolling_origin_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Prediction file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

for target in ["flute_1", "flute_2", "flute_3"]:
    subset = df[df["target"] == target]

    plt.figure(figsize=(12, 6))

    actual = (
        subset[["cut", "actual_wear"]]
        .drop_duplicates("cut")
        .sort_values("cut")
    )

    plt.plot(
        actual["cut"],
        actual["actual_wear"],
        marker="o",
        markersize=3,
        label="Actual wear",
    )

    for model_name in ["Linear", "Piecewise Linear"]:
        model_data = (
            subset[subset["model"] == model_name]
            .sort_values("cut")
        )

        plt.plot(
            model_data["cut"],
            model_data["predicted_wear"],
            marker=".",
            markersize=3,
            label=f"{model_name} prediction",
        )

    plt.xlabel("Cut number")
    plt.ylabel("Wear label")
    plt.title(f"{target.replace('_', ' ').title()}: Actual vs Predicted Wear")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{target}_rolling_origin.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")

print("\nAll plots generated successfully.")
