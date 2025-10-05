from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import RedirectResponse
from fastapi import Query
import io
import csv
import uuid
import os
import json
from typing import Optional, Dict


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

def _read_table_like(raw: bytes, filename: Optional[str]) -> pd.DataFrame:
    """
    Robustly read CSV/TSV/JSON with various delimiters, encodings, and comment lines.
    Tries JSON (lines and standard), then CSV with inferred and explicit seps,
    and finally a skip-bad-lines fallback.
    """
    name = (filename or "").lower()

    # JSON first
    if name.endswith(".json"):
        # Try JSON Lines, then normal JSON
        for kwargs in ({"lines": True}, {}):
            try:
                return pd.read_json(io.BytesIO(raw), **kwargs)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail="Could not read JSON (tried lines and standard).")

    # CSV/TSV variants
    # Order: infer sep, then comma, semicolon, tab; all with python engine and BOM-safe encoding
    candidates = (
        dict(sep=None, engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep=",",  engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep=";",  engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep="\t", engine="python", encoding="utf-8-sig", comment="#"),
    )
    bio = io.BytesIO(raw)
    for kwargs in candidates:
        try:
            bio.seek(0)
            return pd.read_csv(bio, **kwargs)
        except Exception:
            continue

    # Last resort: skip malformed rows so user still gets something back
    try:
        bio.seek(0)
        return pd.read_csv(
            bio,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
            comment="#",
            on_bad_lines="skip",  # pandas >=1.3
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file as CSV/JSON: {e}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce to numeric, create engineered cols to mirror training/web,
    and order columns to match feature_list. Missing values left as NaN
    (XGBoost can handle NaNs).
    """
    df = df.copy()

    # try to coerce any non-numeric
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # engineer features if raw cols are present
    if {"koi_period", "koi_duration", "koi_depth"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["log_period"] = np.log(df["koi_period"].replace(0, np.nan))
            df["dur_over_per"] = df["koi_duration"] / (df["koi_period"] * 24)
            df["depth_sqrt_dur"] = np.sqrt(df["koi_depth"]) / df["koi_duration"].replace(0, np.nan)

    # make sure all expected columns exist (create missing as NaN)
    for col in feature_list:
        if col not in df.columns:
            df[col] = np.nan

    # order columns exactly as training
    df = df[feature_list]

    return df


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

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = _read_table_like(raw, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    X = build_features(df)
    preds = model.predict(X)
    probs = model.predict_proba(X).max(axis=1)

    df_out = df.copy()
    df_out["prediction"] = label_encoder.inverse_transform(preds)
    df_out["confidence"] = np.round(probs * 100, 2)

    exports_dir = os.path.join("static", "exports")
    os.makedirs(exports_dir, exist_ok=True)
    fname = f"predictions_{uuid.uuid4().hex[:8]}.csv"
    save_path = os.path.join(exports_dir, fname)
    df_out.to_csv(save_path, index=False)

    preview_html = df_out.head(20).to_html(classes="table table-sm", index=False, border=0)

    # Store temporary info in memory (optional improvement — e.g. using a cache or session)
    # For now, just redirect with query params
    return RedirectResponse(
        url=f"/?batch_rows={len(df_out)}&batch_file={fname}", status_code=303
    )

@app.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request,
    batch_rows: int = Query(None),
    batch_file: str = Query(None)
):
    batch = None
    if batch_rows and batch_file:
        file_path = f"static/exports/{batch_file}"
        if os.path.exists(file_path):
            df_out = pd.read_csv(file_path)
            preview_html = df_out.head(20).to_html(classes="table table-sm", index=False, border=0)
            batch = {
                "rows": batch_rows,
                "download_href": f"/{file_path}",
                "preview_html": preview_html,
            }

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "feature_info": filtered_feature_info, "batch": batch},
    )

# --- Spectrometry (Macedo/Zalewski) artifacts ---
try:
    macedo_model = joblib.load("../models_macedo/lightgbm_macedo.joblib")
    macedo_label_encoder = joblib.load("../models_macedo/label_encoder.joblib")
    macedo_feature_list = joblib.load("../models_macedo/feature_list.joblib")  # post-cleaning feature list
except FileNotFoundError:
    macedo_model = None
    macedo_label_encoder = None
    macedo_feature_list = None

def build_spectro_feature_info(cols) -> Dict[str, Dict]:
    # Simple numeric inputs with default 0; customize labels here if you want
    return {c: {"label": c, "type": "number", "default": 0} for c in (cols or [])}

# Build fields only if artifacts are present
spectro_feature_info = build_spectro_feature_info(macedo_feature_list)

def prepare_spectro_X(df: pd.DataFrame) -> pd.DataFrame:
    if macedo_feature_list is None:
        # Should never get here because routes guard; keep a safe fallback
        return df.copy()
    df = df.copy()

    # Coerce numeric; non-numeric -> NaN
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only known features; add missing with NaN
    for col in macedo_feature_list:
        if col not in df.columns:
            df[col] = np.nan

    X = df[macedo_feature_list].copy()

    # Replace inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaNs with column medians (safe default)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    return X

@app.get("/spectro", response_class=HTMLResponse)
async def spectro_root(
    request: Request,
    batch_rows: Optional[int] = Query(None),
    batch_file: Optional[str] = Query(None),
):
    spectro_batch = None
    if batch_rows is not None and batch_file:
        file_path = f"static/exports/{batch_file}"
        if os.path.exists(file_path):
            df_out = pd.read_csv(file_path)
            preview_html = df_out.head(20).to_html(classes="table table-sm", index=False, border=0)
            spectro_batch = {
                "rows": int(batch_rows),
                "download_href": f"/{file_path}",
                "preview_html": preview_html,
            }

    return templates.TemplateResponse(
        "spectro.html",
        {
            "request": request,
            "spectro_feature_info": {},   # << hide manual form
            "spectro_batch": spectro_batch,
            "spectro_user_input": None,
        },
    )


@app.post("/spectro", response_class=HTMLResponse)
async def spectro_predict(request: Request):
    if macedo_model is None or macedo_label_encoder is None or macedo_feature_list is None:
        raise HTTPException(status_code=503, detail="Spectrometry model not available. Please train it first.")

    form_data = await request.form()

    # Collect inputs; fall back to defaults
    features = {}
    for key, info in spectro_feature_info.items():
        val = form_data.get(key, info["default"])
        try:
            features[key] = float(val)
        except Exception:
            features[key] = np.nan

    X = pd.DataFrame([features])
    X = prepare_spectro_X(X)

    preds = macedo_model.predict(X)
    proba = macedo_model.predict_proba(X) if hasattr(macedo_model, "predict_proba") else None
    conf = float(np.max(proba, axis=1)[0]) if proba is not None else 1.0

    pred_label = macedo_label_encoder.inverse_transform(preds)[0]

    return templates.TemplateResponse(
        "spectro.html",
        {
            "request": request,
            "spectro_feature_info": spectro_feature_info,
            "spectro_prediction": pred_label,
            "spectro_confidence": f"{conf * 100:.1f}%",
            "spectro_user_input": features,
        },
    )

@app.post("/spectro/upload")
async def spectro_upload(request: Request, file: UploadFile = File(...)):
    if macedo_model is None or macedo_label_encoder is None or macedo_feature_list is None:
        raise HTTPException(status_code=503, detail="Spectrometry model not available. Please train it first.")

    raw = await file.read()
    try:
        # spectrometry expects CSV; reuse _read_table_like if you want broader support
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    X = prepare_spectro_X(df)

    try:
        preds = macedo_model.predict(X)
        probs = macedo_model.predict_proba(X).max(axis=1) if hasattr(macedo_model, "predict_proba") else np.ones(len(X))
    except Exception as e:
        cols_list = ", ".join(list(df.columns)[:20]) + ("..." if len(df.columns) > 20 else "")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}. Parsed columns: [{cols_list}]")

    df_out = df.copy()
    df_out["prediction"] = macedo_label_encoder.inverse_transform(preds)
    df_out["confidence"] = np.round(probs * 100, 2)

    exports_dir = os.path.join("static", "exports")
    os.makedirs(exports_dir, exist_ok=True)
    fname = f"spectro_predictions_{uuid.uuid4().hex[:8]}.csv"
    save_path = os.path.join(exports_dir, fname)
    df_out.to_csv(save_path, index=False)

    return RedirectResponse(
        url=f"/spectro?batch_rows={len(df_out)}&batch_file={fname}",
        status_code=303,
    )
