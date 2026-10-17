
from pathlib import Path
import re
import numpy as np
import pandas as pd

# Project paths
PROJECT = Path(__file__).resolve().parent
RAW_DIR = PROJECT / "data" / "raw" / "phm2010"
SENSOR_DIR = RAW_DIR / "c1"
WEAR_FILE = RAW_DIR / "c1_wear.csv"
OUTPUT_DIR = PROJECT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "phm2010_c1_features.csv"

# Seven channels in the PHM 2010 sensor recordings.
# Confirm their exact physical names and units against the dataset documentation.
CHANNELS = [
    "channel_1", "channel_2", "channel_3",
    "channel_4", "channel_5", "channel_6", "channel_7"
]


def extract_features(file_path):
    """Read one sensor recording and calculate statistical features."""
    data = pd.read_csv(file_path, header=None)

    if data.shape[1] != 7:
        raise ValueError(
            f"{file_path.name}: expected 7 columns, found {data.shape[1]}"
        )

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError(f"{file_path.name} contains no valid numeric rows.")

    result = {
        "n_samples": len(data),
        "source_file": file_path.name
    }

    for index, channel in enumerate(CHANNELS):
        values = data.iloc[:, index].to_numpy(dtype=float)

        result[f"{channel}_mean"] = float(np.mean(values))
        result[f"{channel}_std"] = float(np.std(values))
        result[f"{channel}_rms"] = float(
            np.sqrt(np.mean(np.square(values)))
        )
        result[f"{channel}_min"] = float(np.min(values))
        result[f"{channel}_max"] = float(np.max(values))
        result[f"{channel}_abs_mean"] = float(np.mean(np.abs(values)))

    return result


def main():
    if not SENSOR_DIR.is_dir():
        raise FileNotFoundError(f"Sensor directory not found: {SENSOR_DIR}")

    if not WEAR_FILE.is_file():
        raise FileNotFoundError(f"Wear-label file not found: {WEAR_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wear = pd.read_csv(WEAR_FILE)

    required_columns = {"cut", "flute_1", "flute_2", "flute_3"}
    if not required_columns.issubset(wear.columns):
        raise ValueError(
            f"Unexpected wear columns. Found: {list(wear.columns)}"
        )

    if wear["cut"].duplicated().any():
        raise ValueError("Duplicate cut numbers found in wear labels.")

    wear["cut"] = pd.to_numeric(wear["cut"], errors="raise").astype(int)

    files = []
    for path in SENSOR_DIR.glob("*.csv"):
        match = re.fullmatch(r"c_1_(\d+)", path.stem)
        if match:
            files.append((int(match.group(1)), path))

    files.sort(key=lambda item: item[0])

    if not files:
        raise RuntimeError("No sensor files matching c_1_NNN.csv were found.")

    file_cuts = [cut for cut, _ in files]
    if len(file_cuts) != len(set(file_cuts)):
        raise ValueError("Duplicate cut numbers found in sensor filenames.")

    label_cuts = set(wear["cut"].tolist())
    file_cut_set = set(file_cuts)

    missing_labels = sorted(file_cut_set - label_cuts)
    missing_files = sorted(label_cuts - file_cut_set)

    if missing_labels or missing_files:
        raise ValueError(
            f"Cut mapping mismatch. Missing labels: {missing_labels[:10]}; "
            f"missing sensor files: {missing_files[:10]}"
        )

    records = []

    for position, (cut, path) in enumerate(files, start=1):
        features = extract_features(path)
        features["cut"] = cut
        records.append(features)

        if position % 25 == 0 or position == len(files):
            print(f"Processed {position}/{len(files)} sensor files")

    features_df = pd.DataFrame(records)

    # Join by cut number, not by row order.
    result = features_df.merge(
        wear[["cut", "flute_1", "flute_2", "flute_3"]],
        on="cut",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    if not (result["_merge"] == "both").all():
        raise ValueError("Some sensor recordings did not match wear labels.")

    result = result.drop(columns=["_merge"])
    result = result.sort_values("cut")
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature extraction completed.")
    print(f"Sensor files processed: {len(files)}")
    print(f"Output rows: {len(result)}")
    print(f"Output columns: {len(result.columns)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nFirst five rows:")
    print(result.head().to_string(index=False))
    print("\nWear-label summary:")
    print(result[["flute_1", "flute_2", "flute_3"]].describe())


if __name__ == "__main__":
    main()
