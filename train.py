import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_PATH = Path("data/car_listings.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "car_price_pipeline.joblib"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Remove rows with missing critical fields (if any)
    df = df.dropna(subset=["make","model","year","mileage_km","fuel_type","transmission","owner_count","location_city","engine_cc","power_bhp","seats","price"])
    return df

def build_pipeline(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                              ("model", model)])
    return pipeline

def main():
    df = load_data()

    target = "price"
    categorical = ["make","model","fuel_type","transmission","location_city"]
    numeric = ["year","mileage_km","owner_count","engine_cc","power_bhp","seats"]

    X = df[categorical + numeric]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(categorical, numeric)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Test MAE:  ₹{mae:,.0f}")
    print(f"Test RMSE: ₹{rmse:,.0f}")
    print(f"R² Score:  {r2:.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
