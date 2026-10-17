
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data" / "processed" / "phm2010_c1_features.csv"
OUT = PROJECT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA).sort_values("cut").reset_index(drop=True)
df["average_wear"] = df[["flute_1", "flute_2", "flute_3"]].mean(axis=1)

features = [
    c for c in df.columns
    if c.startswith("channel_")
    and c.rsplit("_", 1)[-1]
    in {"mean", "std", "rms", "min", "max", "abs_mean"}
]

train = df.iloc[:252]
test = df.iloc[252:]

models = {
    "Ridge (sensors)": (
        make_pipeline(StandardScaler(), Ridge(alpha=10.0)), features
    ),
    "Random Forest (sensors)": (
        RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ), features
    ),
    "Extra Trees (sensors)": (
        ExtraTreesRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ), features
    ),
    "Cut number only": (
        make_pipeline(StandardScaler(), Ridge(alpha=10.0)), ["cut"]
    ),
}

plt.figure(figsize=(12, 6))
plt.plot(
    test["cut"], test["average_wear"],
    label="Actual wear", linewidth=3
)

for name, (model, columns) in models.items():
    model.fit(train[columns], train["average_wear"])
    predictions = model.predict(test[columns])
    plt.plot(test["cut"], predictions, label=name)

plt.xlabel("Cut number")
plt.ylabel("Average wear (dataset units)")
plt.title("C1 Final Test Window: Cuts 253–315")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = OUT / "phm2010_final_window_comparison.png"
plt.savefig(path, dpi=160)
print(f"Saved graph: {path}")
plt.show()
