
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load trained models
crop_model = joblib.load("crop_loss_model.pkl")
energy_model = joblib.load("energy_load_model.pkl")
transport_model = joblib.load("transport_delay_model.pkl")

@app.get("/")
def home():
    return {"message": "Climate API is running"}

@app.post("/predict")
def predict(data: dict):
    temp = data["temp"]
    soil_moisture = data["soil_moisture"]
    ndvi = data["ndvi"]
    humidity = data["humidity"]

    features = np.array([[temp, soil_moisture, ndvi, humidity]])

    crop_loss = crop_model.predict(features)[0]
    energy_load = energy_model.predict(features)[0]
    transport_delay = transport_model.predict(features)[0]

    return {
        "crop_loss": float(crop_loss),
        "energy_load": float(energy_load),
        "transport_delay": float(transport_delay)
    }

