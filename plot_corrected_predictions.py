from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "corrected_final_test_predictions.csv"
OUTPUT_DIR = RESULTS_DIR / "prediction_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "target", "cut", "actual_wear", "predicted_wear"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for target in ["flute_1", "flute_2", "flute_3"]:
        part = df[df["target"] == target].sort_values("cut")

        if part.empty:
            print(f"Skipping {target}: no predictions found.")
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(
            part["cut"],
            part["actual_wear"],
            marker="o",
            markersize=3,
            label="Actual wear",
        )
        plt.plot(
            part["cut"],
            part["predicted_wear"],
            marker=".",
            markersize=4,
            linestyle="--",
            label="Predicted wear",
        )

        plt.xlabel("Cut number")
        plt.ylabel("Wear label (original dataset units)")
        plt.title(f"{target.replace('_', ' ').title()}: Actual vs Predicted")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = OUTPUT_DIR / f"{target}_actual_vs_predicted.png"
        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"Saved: {output_file}")

    print("\nAll available prediction plots have been generated.")


if __name__ == "__main__":
    main()