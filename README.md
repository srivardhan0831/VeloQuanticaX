# Car Price Estimator (Flask + scikit-learn)

A minimal, production-ready example that trains a regression model to predict used car prices and serves predictions with a Flask web app.

## Features
- End-to-end ML pipeline with `ColumnTransformer` + `OneHotEncoder` + `RandomForestRegressor`
- Clean separation: `train.py` for training, `app.py` for serving
- Sample dataset included (`data/car_listings.csv`) so you can train immediately
- Form-based HTML UI (no JavaScript required)
- Model metrics printed after training (MAE, RMSE, R^2)
- Saves the trained pipeline to `models/car_price_pipeline.joblib`

## Quickstart

```bash
# 1) (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train the model
python train.py

# 4) Run the web app (Flask)
python app.py
```

Then open: http://127.0.0.1:5000

## Data Schema

The sample CSV includes these columns:
- `make` (str): e.g., "Toyota", "Honda", "Hyundai", "Maruti"
- `model` (str): e.g., "Corolla", "City", "i20", "Swift"
- `year` (int): first registration year
- `mileage_km` (int): odometer reading in kilometers
- `fuel_type` (str): Petrol/Diesel/CNG/Electric/Hybrid
- `transmission` (str): Manual/Automatic
- `owner_count` (int): number of previous owners
- `location_city` (str): city/region (e.g., "Hyderabad", "Bengaluru")
- `engine_cc` (int): engine displacement in cc (approximate)
- `power_bhp` (float): engine power (approximate)
- `seats` (int): seating capacity
- `price` (int): **target** in INR

> Replace `data/car_listings.csv` with your own data using the same column names for best results.

## Notes
- RandomForest is robust and easy to start with; for even better accuracy, try GradientBoosting, XGBoost, LightGBM, or CatBoost.
- If you change columns or add new ones, update the lists in `train.py` accordingly.
- Always retrain after changing features or data distribution.
