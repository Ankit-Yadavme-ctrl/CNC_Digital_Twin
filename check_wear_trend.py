
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

data_path = (
    Path(__file__).resolve().parent
    / "data_processed"
    / "phm2010_c1_features.csv"
)

df = pd.read_csv(data_path).sort_values("cut").reset_index(drop=True)

train = df.iloc[:220]
test = df.iloc[267:315]

for target in ["flute_1", "flute_2", "flute_3"]:
    print(f"\n{target}")

    for degree in [1, 2]:
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression()
        )

        model.fit(train[["cut"]], train[target])
        predictions = model.predict(test[["cut"]])

        mae = mean_absolute_error(test[target], predictions)
        rmse = np.sqrt(mean_squared_error(test[target], predictions))
        r2 = r2_score(test[target], predictions)

        print(
            f"Degree {degree}: "
            f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}"
        )

print("\nDiagnostic complete. No existing model files were changed.")
