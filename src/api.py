from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="House Price Prediction API",
    description="ML API for predicting California house prices",
    version="1.0.0"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# INPUT DATA MODEL
# --------------------------------------------------

class HouseFeatures(BaseModel):

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# --------------------------------------------------
# ROOT / HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running",
        "model": "Random Forest",
        "version": "1.0.0"
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_price(data: HouseFeatures):

    input_data = pd.DataFrame(
        [[
            data.MedInc,
            data.HouseAge,
            data.AveRooms,
            data.AveBedrms,
            data.Population,
            data.AveOccup,
            data.Latitude,
            data.Longitude
        ]],
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 4),
        "price_unit": "$100,000"
    }