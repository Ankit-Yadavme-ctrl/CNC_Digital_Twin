from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
# Set this to the folder containing c_1_001.csv ... c_1_315.csv.
# Example: r"C:\Users\YourName\Downloads\c1"

RAW_DIR = Path(r"C:\Users\Asus\Downloads\PHM2010\c1")


# Set this to the location of c1_wear.csv.
# Example: r"C:\Users\YourName\Downloads\c1_wear.csv"k

WEAR_FILE = Path(r"C:\Users\Asus\Downloads\PHM2010\c1_wear.csv")


# Output is saved beside this script in a folder named data_processed.
OUTPUT_DIR = Path(__file__).resolve().parent / "data_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "phm2010_c1_features.csv"

CHANNELS = [
    "force_x_N", "force_y_N", "force_z_N",
    "vibration_x_g", "vibration_y_g", "vibration_z_g",
    "ae_rms_V"
]

def get_cut_number(path: Path) -> int:
    match = re.search(r"c_1_(\d+)", path.stem.lower())
    if not match:
        raise ValueError(f"Cannot determine cut number from filename: {path.name}")
    return int(match.group(1))

def read_signal(path: Path) -> pd.DataFrame:
    # PHM2010 signal files are headerless, with seven numeric columns.
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(CHANNELS):
        raise ValueError(
            f"{path.name} has {df.shape[1]} columns; expected 7. "
            "Check whether the file includes a header or a different format."
        )
    df.columns = CHANNELS
    for col in CHANNELS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def extract_features(path: Path) -> dict:
    df = read_signal(path)
    row = {
        "cut": get_cut_number(path),
        "source_file": path.name,
        "n_samples": int(len(df))
    }

    for col in CHANNELS:
        x = df[col].dropna().to_numpy(dtype=np.float64)
        if len(x) == 0:
            raise ValueError(f"No valid numeric values in {path.name}, channel {col}")

        row[f"{col}_mean"] = float(np.mean(x))
        row[f"{col}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        row[f"{col}_rms"] = float(np.sqrt(np.mean(np.square(x))))
        row[f"{col}_min"] = float(np.min(x))
        row[f"{col}_max"] = float(np.max(x))
        row[f"{col}_abs_mean"] = float(np.mean(np.abs(x)))

    return row

def main():
    if not RAW_DIR.exists():
        raise SystemExit(
            f"\nRAW_DIR does not exist:\n{RAW_DIR}\n\n"
            "Edit RAW_DIR near the top of this script to point to the folder "
            "containing c_1_001.csv through c_1_315.csv."
        )
    if not WEAR_FILE.exists():
        raise SystemExit(
            f"\nWEAR_FILE does not exist:\n{WEAR_FILE}\n\n"
            "Edit WEAR_FILE near the top of this script to point to c1_wear.csv."
        )

    raw_files = sorted(
        [p for p in RAW_DIR.glob("c_1_*.csv")],
        key=get_cut_number
    )
    if not raw_files:
        raise SystemExit(f"No c_1_*.csv files found in: {RAW_DIR}")

    print(f"Found {len(raw_files)} raw sensor files.")
    print("Extracting features; this may take a while on a laptop...")

    feature_rows = []
    for i, path in enumerate(raw_files, start=1):
        print(f"[{i}/{len(raw_files)}] {path.name}")
        feature_rows.append(extract_features(path))

    features = pd.DataFrame(feature_rows).sort_values("cut").reset_index(drop=True)

    wear = pd.read_csv(WEAR_FILE)
    required_wear_columns = {"cut", "flute_1", "flute_2", "flute_3"}
    missing = required_wear_columns - set(wear.columns)
    if missing:
        raise SystemExit(
            f"Wear file is missing columns: {sorted(missing)}. "
            f"Found columns: {wear.columns.tolist()}"
        )

    wear["cut"] = pd.to_numeric(wear["cut"], errors="raise").astype(int)
    for col in ["flute_1", "flute_2", "flute_3"]:
        wear[col] = pd.to_numeric(wear[col], errors="coerce")

    if wear["cut"].duplicated().any():
        raise SystemExit("c1_wear.csv contains duplicate cut numbers; stopping to avoid a bad merge.")

    merged = features.merge(wear, on="cut", how="left", validate="one_to_one", indicator=True)

    unmatched = merged.loc[merged["_merge"] != "both", "cut"].tolist()
    merged = merged.drop(columns="_merge")

    if unmatched:
        raise SystemExit(
            f"No wear labels found for these cuts: {unmatched[:20]}. "
            "Check cut numbering before training."
        )

    missing_labels = merged[["flute_1", "flute_2", "flute_3"]].isna().sum().to_dict()
    if any(missing_labels.values()):
        raise SystemExit(f"Missing wear labels after merge: {missing_labels}")

    merged.to_csv(OUTPUT_FILE, index=False)

    print("\n========== DATASET CHECK ==========")
    print(f"Raw files processed: {len(raw_files)}")
    print(f"Feature rows: {len(features)}")
    print(f"Wear-label rows: {len(wear)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Feature columns (including cut/source): {features.shape[1]}")
    print(f"Output: {OUTPUT_FILE}")
    print("Wear label columns: flute_1, flute_2, flute_3")
    print("\nFirst five rows:")
    print(merged[["cut", "source_file", "flute_1", "flute_2", "flute_3"]].head().to_string(index=False))

    if len(raw_files) != 315:
        print(
            f"\nWARNING: Found {len(raw_files)} raw files, not 315. "
            "Check whether all cuts c_1_001.csv through c_1_315.csv are present."
        )
    else:
        print("\nAll 315 raw cuts were processed and matched to wear labels.")

if __name__ == "__main__":
    main()
