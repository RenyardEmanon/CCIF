import numpy as np
import pandas as pd

n = 1000

data = pd.DataFrame({
    "temp": np.random.uniform(25, 50, n),
    "soil_moisture": np.random.uniform(0, 100, n),
    "ndvi": np.random.uniform(0, 1, n),
    "humidity": np.random.uniform(20, 90, n),
})

# Simulated relationships
data["crop_loss"] = 0.5*data["temp"] - 0.3*data["soil_moisture"] - 20*data["ndvi"]
data["energy_load"] = 0.7*data["temp"] + 0.2*data["humidity"]
data["transport_delay"] = 0.4*data["temp"] - 0.2*data["ndvi"]

data = data.clip(lower=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = data[["temp", "soil_moisture", "ndvi", "humidity"]]

models = {}

for target in ["crop_loss", "energy_load", "transport_delay"]:
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    models[target] = model

import joblib

for name, model in models.items():
    joblib.dump(model, f"{name}_model.pkl")

data.to_csv("climate_data.csv", index=False)

print(data.head())

import os
print(os.getcwd())

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
def predict(data1: dict):
    temp = data1["temp"]
    soil_moisture = data1["soil_moisture"]
    ndvi = data1["ndvi"]
    humidity = data1["humidity"]

    features = np.array([[temp, soil_moisture, ndvi, humidity]])

    crop_loss = crop_model.predict(features)[0]
    energy_load = energy_model.predict(features)[0]
    transport_delay = transport_model.predict(features)[0]

    return {
        "crop_loss": float(crop_loss),
        "energy_load": float(energy_load),
        "transport_delay": float(transport_delay)
    }