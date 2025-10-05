from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model, label encoder and feature list
model = joblib.load("../models/xgb_koi.joblib")
label_encoder = joblib.load("../models/label_encoder.joblib")
feature_list = joblib.load("../build/feature_list.joblib")

feature_info = {
    "koi_period": {
        "label": "Orbital Period [days]",
        "type": "number",
        "min": 0,
        "max": 1000,
        "default": 30,
    },
    "koi_period_err1": {
        "label": "Orbital Period Upper Unc. [days]",
        "type": "number",
        "min": 0,
        "max": 1,
        "default": 0.001,
    },
    "koi_period_err2": {
        "label": "Orbital Period Lower Unc. [days]",
        "type": "number",
        "min": -1,
        "max": 0,
        "default": -0.001,
    },
    "koi_time0bk": {
        "label": "Transit Epoch [BKJD]",
        "type": "number",
        "min": 100,
        "max": 200,
        "default": 135,
    },
    "koi_time0bk_err1": {
        "label": "Transit Epoch Upper Unc. [BKJD]",
        "type": "number",
        "min": 0,
        "max": 1,
        "default": 0.01,
    },
    "koi_time0bk_err2": {
        "label": "Transit Epoch Lower Unc. [BKJD]",
        "type": "number",
        "min": -1,
        "max": 0,
        "default": -0.01,
    },
    "koi_impact": {
        "label": "Impact Parameter",
        "type": "number",
        "min": 0,
        "max": 2,
        "default": 0.5,
    },
    "koi_impact_err1": {
        "label": "Impact Parameter Upper Unc.",
        "type": "number",
        "min": 0,
        "max": 1,
        "default": 0.1,
    },
    "koi_impact_err2": {
        "label": "Impact Parameter Lower Unc.",
        "type": "number",
        "min": -1,
        "max": 0,
        "default": -0.1,
    },
    "koi_duration": {
        "label": "Transit Duration [hrs]",
        "type": "number",
        "min": 0,
        "max": 24,
        "default": 4,
    },
    "koi_duration_err1": {
        "label": "Transit Duration Upper Unc. [hrs]",
        "type": "number",
        "min": 0,
        "max": 5,
        "default": 0.5,
    },
    "koi_duration_err2": {
        "label": "Transit Duration Lower Unc. [hrs]",
        "type": "number",
        "min": -5,
        "max": 0,
        "default": -0.5,
    },
    "koi_depth": {
        "label": "Transit Depth [ppm]",
        "type": "number",
        "min": 0,
        "max": 100000,
        "default": 1000,
    },
    "koi_depth_err1": {
        "label": "Transit Depth Upper Unc. [ppm]",
        "type": "number",
        "min": 0,
        "max": 1000,
        "default": 100,
    },
    "koi_depth_err2": {
        "label": "Transit Depth Lower Unc. [ppm]",
        "type": "number",
        "min": -1000,
        "max": 0,
        "default": -100,
    },
    "koi_prad": {
        "label": "Planetary Radius [Earth radii]",
        "type": "number",
        "min": 0,
        "max": 100,
        "default": 2,
    },
    "koi_prad_err1": {
        "label": "Planetary Radius Upper Unc. [Earth radii]",
        "type": "number",
        "min": 0,
        "max": 10,
        "default": 0.5,
    },
    "koi_prad_err2": {
        "label": "Planetary Radius Lower Unc. [Earth radii]",
        "type": "number",
        "min": -10,
        "max": 0,
        "default": -0.5,
    },
    "koi_teq": {
        "label": "Equilibrium Temperature [K]",
        "type": "number",
        "min": 0,
        "max": 3000,
        "default": 1000,
    },
    "koi_insol": {
        "label": "Insolation Flux [Earth flux]",
        "type": "number",
        "min": 0,
        "max": 10000,
        "default": 100,
    },
    "koi_insol_err1": {
        "label": "Insolation Flux Upper Unc. [Earth flux]",
        "type": "number",
        "min": 0,
        "max": 1000,
        "default": 10,
    },
    "koi_insol_err2": {
        "label": "Insolation Flux Lower Unc. [Earth flux]",
        "type": "number",
        "min": -1000,
        "max": 0,
        "default": -10,
    },
    "koi_model_snr": {
        "label": "Transit Signal-to-Noise",
        "type": "number",
        "min": 0,
        "max": 1000,
        "default": 50,
    },
    "koi_steff": {
        "label": "Stellar Effective Temperature [K]",
        "type": "number",
        "min": 2000,
        "max": 10000,
        "default": 5000,
    },
    "koi_steff_err1": {
        "label": "Stellar Effective Temperature Upper Unc. [K]",
        "type": "number",
        "min": 0,
        "max": 1000,
        "default": 100,
    },
    "koi_steff_err2": {
        "label": "Stellar Effective Temperature Lower Unc. [K]",
        "type": "number",
        "min": -1000,
        "max": 0,
        "default": -100,
    },
    "koi_slogg": {
        "label": "Stellar Surface Gravity [log10(cm/s**2)]",
        "type": "number",
        "min": 0,
        "max": 6,
        "default": 4.5,
    },
    "koi_slogg_err1": {
        "label": "Stellar Surface Gravity Upper Unc. [log10(cm/s**2)]",
        "type": "number",
        "min": 0,
        "max": 1,
        "default": 0.1,
    },
    "koi_slogg_err2": {
        "label": "Stellar Surface Gravity Lower Unc. [log10(cm/s**2)]",
        "type": "number",
        "min": -1,
        "max": 0,
        "default": -0.1,
    },
    "koi_srad": {
        "label": "Stellar Radius [Solar radii]",
        "type": "number",
        "min": 0,
        "max": 50,
        "default": 1,
    },
    "koi_srad_err1": {
        "label": "Stellar Radius Upper Unc. [Solar radii]",
        "type": "number",
        "min": 0,
        "max": 5,
        "default": 0.2,
    },
    "koi_srad_err2": {
        "label": "Stellar Radius Lower Unc. [Solar radii]",
        "type": "number",
        "min": -5,
        "max": 0,
        "default": -0.2,
    },
    "ra": {
        "label": "RA [decimal degrees]",
        "type": "number",
        "min": 0,
        "max": 360,
        "default": 290,
    },
    "dec": {
        "label": "Dec [decimal degrees]",
        "type": "number",
        "min": -90,
        "max": 90,
        "default": 45,
    },
    "koi_kepmag": {
        "label": "Kepler-band [mag]",
        "type": "number",
        "min": 0,
        "max": 20,
        "default": 15,
    },
    "koi_ror": {
        "label": "Planet-Star Radius Ratio",
        "type": "number",
        "min": 0,
        "max": 1,
        "default": 0.1,
    },
    "koi_smet": {
        "label": "Stellar Metallicity [Fe/H]",
        "type": "number",
        "min": -2,
        "max": 1,
        "default": 0.0,
    },
}


filtered_feature_info = {
    key: feature_info[key]
    for key in feature_list
    if key in feature_info and not (key.startswith("koi_fpflag_") or key == "koi_score")
}

# Warn if feature_list still contains leakage features
leakage_keys = [
    k for k in feature_list if k.startswith("koi_fpflag_") or k == "koi_score"
]
if leakage_keys:
    print(
        f"[warn] Excluding leakage features from UI/inference input: {leakage_keys}. "
        "Ensure you retrain the model without them to eliminate hidden leakage."
    )


class KOIFeatures(BaseModel):
    koi_period: float
    koi_period_err1: float
    koi_period_err2: float
    koi_time0bk: float
    koi_time0bk_err1: float
    koi_time0bk_err2: float
    koi_impact: float
    koi_impact_err1: float
    koi_impact_err2: float
    koi_duration: float
    koi_duration_err1: float
    koi_duration_err2: float
    koi_depth: float
    koi_depth_err1: float
    koi_depth_err2: float
    koi_prad: float
    koi_prad_err1: float
    koi_prad_err2: float
    koi_teq: float
    koi_insol: float
    koi_insol_err1: float
    koi_insol_err2: float
    koi_model_snr: float
    koi_steff: float
    koi_steff_err1: float
    koi_steff_err2: float
    koi_slogg: float
    koi_slogg_err1: float
    koi_slogg_err2: float
    koi_srad: float
    koi_srad_err1: float
    koi_srad_err2: float
    ra: float
    dec: float
    koi_kepmag: float
    koi_ror: float
    koi_smet: float


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "feature_info": filtered_feature_info}
    )


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()

    features = {}
    for key, info in filtered_feature_info.items():
        if info["type"] == "checkbox":
            features[key] = 1.0 if form_data.get(key) else 0.0
        else:
            features[key] = float(form_data.get(key, info["default"]))

    data = pd.DataFrame([features])

    # Feature Engineering
    data["log_period"] = np.log(data["koi_period"])
    data["dur_over_per"] = data["koi_duration"] / (data["koi_period"] * 24)
    data["depth_sqrt_dur"] = np.sqrt(data["koi_depth"]) / data["koi_duration"]

    # Ensure columns are in the correct order
    data = data[feature_list]

    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)

    prediction_label = label_encoder.inverse_transform(prediction)[0]
    confidence = max(prediction_proba[0]).item()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "feature_info": filtered_feature_info,
            "prediction": prediction_label,
            "confidence": f"{confidence * 100:.1f}%",
            "user_input": features,
        },
    )
