from flask import Flask, render_template, request
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

app = Flask(__name__)
MODEL_PATH = Path("models/car_price_pipeline.joblib")

# Load model at startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Please run `python train.py` first to create models/car_price_pipeline.joblib"
    )
pipeline = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # Collect input from form with defaults
    payload = {
        "make": form.get("make","").strip(),
        "model": form.get("model","").strip(),
        "year": int(form.get("year", 2018)),
        "mileage_km": int(form.get("mileage_km", 40000)),
        "fuel_type": form.get("fuel_type","Petrol").strip(),
        "transmission": form.get("transmission","Manual").strip(),
        "owner_count": int(form.get("owner_count", 1)),
        "location_city": form.get("location_city","Hyderabad").strip(),
        "engine_cc": int(form.get("engine_cc", 1200)),
        "power_bhp": float(form.get("power_bhp", 85.0)),
        "seats": int(form.get("seats", 5)),
    }

    # Convert payload to a DataFrame for the pipeline
    feature_order = [
        "make","model","fuel_type","transmission","location_city",
        "year","mileage_km","owner_count","engine_cc","power_bhp","seats"
    ]
    X = pd.DataFrame([{k: payload[k] for k in feature_order}])  # <-- FIXED

    # Predict price
    pred_price = pipeline.predict(X)[0]

    # Round to nearest thousand for display
    pred_price_disp = int(np.round(pred_price / 1000.0) * 1000)

    return render_template("result.html",
                           pred_price=pred_price_disp,
                           details=payload)

if __name__ == "__main__":
    app.run(debug=True)
