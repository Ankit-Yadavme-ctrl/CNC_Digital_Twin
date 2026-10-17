
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_processed" / "phm2010_c1_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["flute_1", "flute_2", "flute_3"]
EXCLUDED = [
    "cut", "source_file", "n_samples",
    "flute_1", "flute_2", "flute_3"
]


def main():
    df = pd.read_csv(DATA_FILE).sort_values("cut")
    train = df[df["cut"] <= 220].copy()
    features = [c for c in df.columns if c not in EXCLUDED]

    rows = []

    for target in TARGETS:
        for feature in features:
            # Ordinary correlation on training cuts.
            raw_corr = train[feature].corr(train[target])

            # Correlation after removing linear cut progression
            # from both the feature and wear label.
            cut = train["cut"]
            feature_residual = (
                train[feature]
                - (train[feature].mean()
                   + train[feature].cov(cut) / cut.var()
                   * (cut - cut.mean()))
            )
            target_residual = (
                train[target]
                - (train[target].mean()
                   + train[target].cov(cut) / cut.var()
                   * (cut - cut.mean()))
            )

            adjusted_corr = feature_residual.corr(target_residual)

            rows.append({
                "target": target,
                "feature": feature,
                "raw_correlation": raw_corr,
                "correlation_after_linear_cut_adjustment": adjusted_corr,
            })

    result = pd.DataFrame(rows)
    result["absolute_adjusted_correlation"] = (
        result["correlation_after_linear_cut_adjustment"].abs()
    )

    output = RESULTS_DIR / "sensor_wear_correlations.csv"
    result.to_csv(output, index=False)

    for target in TARGETS:
        subset = result[result["target"] == target]

        print(f"\n{target}: TOP 10 RAW CORRELATIONS")
        print(
            subset.assign(
                abs_raw=subset["raw_correlation"].abs()
            ).sort_values("abs_raw", ascending=False)[
                ["feature", "raw_correlation"]
            ].head(10).round(3).to_string(index=False)
        )

        print(f"\n{target}: TOP 10 AFTER LINEAR CUT ADJUSTMENT")
        print(
            subset.sort_values(
                "absolute_adjusted_correlation", ascending=False
            )[
                ["feature", "correlation_after_linear_cut_adjustment"]
            ].head(10).round(3).to_string(index=False)
        )

    print(f"\nSaved: {output}")
    print(
        "\nExploratory analysis only. Adjusted correlations remove "
        "linear cut trends, not nonlinear trends or all confounding."
    )


if __name__ == "__main__":
    main()
