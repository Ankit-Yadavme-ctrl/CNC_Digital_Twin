from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]

REGIONS = [
    ("Cuts 1-110", 1, 110),
    ("Cuts 111-220", 111, 220),
    ("Cuts 221-315", 221, 315),
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).sort_values("cut").reset_index(drop=True)

    results = []

    for target in TARGETS:
        for name, start, end in REGIONS:
            region = df[df["cut"].between(start, end)].sort_values("cut")

            if len(region) < 2:
                continue

            first_cut = region.iloc[0]
            last_cut = region.iloc[-1]

            wear_change = last_cut[target] - first_cut[target]
            cut_interval = last_cut["cut"] - first_cut["cut"]
            wear_rate = wear_change / cut_interval

            results.append({
                "flute": target,
                "region": name,
                "start_cut": int(first_cut["cut"]),
                "end_cut": int(last_cut["cut"]),
                "start_wear": first_cut[target],
                "end_wear": last_cut[target],
                "wear_change": wear_change,
                "average_increase_per_cut": wear_rate,
            })

    report = pd.DataFrame(results)
    output_file = RESULTS_DIR / "wear_rate_analysis.csv"
    report.to_csv(output_file, index=False)

    print("\nWEAR RATE ANALYSIS")
    print(report.round(4).to_string(index=False))
    print(f"\nReport saved to: {output_file}")

    print(
        "\nNote: These are average label increases per cut, "
        "not direct measurements of instantaneous wear rate."
    )


if __name__ == "__main__":
    main()