# AstroNets — NASA Space Apps Challenge (Hackathon Project)

**Team:** Timothy Roch • Cyrus Yip • Ariyan Khayer

AstroNets automates exoplanet triage using AI so researchers, students, and enthusiasts can quickly classify likely planets from open NASA data. We implement the two cornerstone techniques used in exoplanet discovery—**transits** and **radial velocity (spectrometry)**—and expose them through a clean **FastAPI + Jinja2** web app supporting single-entry and batch predictions.
 [![Batch upload demo (Project's Presentation)](https://img.youtube.com/vi/B25vS75lQCY/0.jpg)](https://www.youtube.com/watch?v=B25vS75lQCY)

---

## What AstroNets does

* **Transit classification (KOI data, XGBoost):** predicts **Confirmed / Candidate / False Positive** from Kepler Objects of Interest (KOI) features, with physics-motivated engineered features.
* **Spectrometry classification (LightGBM):** ingests Macedo/Zalewski-style **spectral feature CSVs** and predicts classes with confidence.
* **Web interface:**

  * Manual form for a single KOI → instant prediction + confidence
  * **CSV/JSON** batch upload (KOI features) → downloadable results
  * **Spectrometry CSV** upload → cleaned, aligned, predicted, and exported

---

## Problem & solution (hackathon focus)

* **Problem:** Kepler/K2/TESS missions yield massive candidate tables. Manual vetting is slow and error-prone; many signals are false positives.
* **Our solution:** a dual ML pipeline:

  * **Transit method (light dimming):** detect periodic dips when a planet passes in front of its star. We model this with **XGBoost** on KOI features and engineered transit descriptors.
  * **Radial velocity via spectrometry (light shifting):** detect tiny **Doppler shifts** (blue when approaching, red when receding) and spectral line changes that indicate stellar wobble and, during transits, atmospheric absorption. We model this with **LightGBM** on spectral feature vectors.

---

## Project Structure

```
.
├── KOI_data.csv                     # Kepler Objects of Interest dataset
├── NasaExoplanetArchive.csv         # NASA Exoplanet Archive dataset
├── k2pandc_dataset_exoplanet.csv    # K2 exoplanet dataset
├── lightcurve_dataset.csv           # Light curve dataset
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata (Poetry/PDM)
├── uv.lock                          # Lock file for reproducible installs

├── build/                           # Training artifacts (KOI data)
│   ├── Readme.md
│   ├── X.parquet                    # Cleaned, encoded feature matrix
│   ├── y.csv                        # Labels for KOI data
│   ├── feature_list.joblib          # Saved feature column list
│   └── groups.csv                   # Grouped data for validation

├── models/                          # Trained KOI (transit) model
│   ├── xgb_koi.joblib               # XGBoost classifier
│   └── label_encoder.joblib         # Encodes “Confirmed”, “Candidate”, “False Positive”

├── models_macedo/                   # Trained spectrometry model
│   ├── lightgbm_macedo.joblib       # LightGBM spectrometry classifier
│   ├── label_encoder.joblib         # Label encoder for spectrometry labels
│   └── feature_list.joblib          # Saved spectral feature order

├── plots/
│   └── feature_importance.png       # Feature importance visualization (XGBoost)

├── src/                             # Core ML training scripts
│   ├── extract_data.py              # Processes raw KOI data into X/y
│   ├── train.py                     # Trains the XGBoost transit model
│   ├── visualize.py                 # Feature visualization utilities
│   └── RadVelocity/
│       └── train_lightgbm_macedo.py # Trains the LightGBM spectrometry model

├── web/                             # FastAPI + Jinja2 web interface
│   ├── main.py                      # Main FastAPI app
│   ├── templates/                   # Frontend HTML templates
│   │   ├── index.html               # Main UI (form + batch uploads)
│   │   └── info.html                # Info/documentation page
│   └── static/                      # Static files (images, exports)
│       ├── planet.jpg               # UI background/image
│       └── exports/                 # User prediction results (auto-generated CSVs)

└── index.html                       # (Optional) Root or legacy landing page
```

---


## Install & run

### 1) Create environment & install dependencies

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -U pip

# requirements
pip install -r requirements.txt

# and minimal set
pip install fastapi uvicorn jinja2 python-multipart 
```

### 2) Prepare data & train (first run)

If you already have `models/` and `build/` populated, skip to **Run the web app**.

```bash
# build training features 
cd src
python extract_data.py

# train transit model (XGBoost)
python train.py

# (optional) train spectrometry model (LightGBM)
python RadVelocity/train_lightgbm_macedo.py --csv lightcurve_dataset.csv
# artifacts saved to ../models_macedo
```

Artifacts created:

* `build/X.parquet`, `build/y.csv`, `build/feature_list.joblib` (KOI)
* `models/xgb_koi.joblib`, `models/label_encoder.joblib`
* `models_macedo/lightgbm_macedo.joblib`, `models_macedo/label_encoder.joblib`, `models_macedo/feature_list.joblib`

### 3) Run the web app

```bash
# from repo root
cd web
uvicorn main:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## The models

### Transit (KOI) → **XGBoost**

* **Inputs (examples used by UI and scripts):**
  `koi_period`, `koi_duration`, `koi_depth`, `koi_ror`, `koi_impact`,
  `koi_model_snr`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_smet`, `koi_kepmag`, RA/Dec, plus uncertainties where available.
* **Engineered features (mirrored in both training and web inference):**

  * `log_period = log(koi_period)`
  * `dur_over_per = koi_duration / (koi_period * 24)`
  * `depth_sqrt_dur = sqrt(koi_depth) / koi_duration`
* **Why XGBoost:** robust on tabular data, handles missing values, learns nonlinear interactions; outputs class probabilities used as **confidence**.

### Spectrometry (radial velocity & atmospheric signals) → **LightGBM**

* **Input:** Macedo/Zalewski-style spectral feature CSVs (last column `label` during training).
* **Cleaning (training & inference):** numeric coercion, inf→NaN, drop all-NaN and constant columns at train time; median-fill; align to saved `feature_list`.
* **Why LightGBM:** fast on high-dimensional numeric features; class weighting for imbalance; probability outputs for confidence.

---

## User interface (what you can do on the site)

The `templates/index.html` renders three main sections:

1. **Manual KOI prediction (form)**

   * Enter KOI/stellar features → click **Predict**
   * See **Prediction** (Confirmed/Candidate/False) and **Confidence**
   * Form fields are sourced from `filtered_feature_info` in `main.py`, ordered to match the trained `feature_list`.
     
   [![Manual entry demo (transit)](https://img.youtube.com/vi/S-hOxUzVjtI/0.jpg)](https://www.youtube.com/watch?v=S-hOxUzVjtI)

2. **KOI Batch predictions (CSV/JSON uploader)**

   * Upload a table with KOI columns (names matching the form/KOI schema).
   * The server:

     * robustly parses CSV/TSV/JSON (`_read_table_like`)
     * builds features + engineered columns (`build_features`)
     * predicts for each row, saves to `web/static/exports/predictions_*.csv`
   * The page shows a preview and a **Download CSV** button.
     
     [![Batch upload demo (KOI & spectrometry)](https://img.youtube.com/vi/nXfRZbtsmvQ/0.jpg)](https://www.youtube.com/watch?v=nXfRZbtsmvQ)

3. **Spectrometry predictions (CSV uploader)**

   * Upload a **spectrometry CSV** (Macedo/Zalewski feature set).
   * The server:

     * coerces numerics, aligns to `models_macedo/feature_list.joblib`, median fills
     * predicts using LightGBM; saves `spectro_predictions_*.csv` in `static/exports`
   * You get a preview and a **Download CSV** button.


---

## Endpoints (already implemented in your `main.py`)

* `GET /` — render home; optionally shows batch preview if query params present
* `POST /` — manual KOI prediction (form)
* `POST /upload` — KOI batch predictions (CSV/JSON)
* `GET /spectro` — spectrometry page with optional preview
* `POST /spectro` — manual spectrometry prediction (when fields are exposed)
* `POST /spectro/upload` — spectrometry batch predictions (CSV)
* `GET /info` — documentation page (template provided separately)

---

## Reproducibility & evaluation

* **Splits:** stratified 80/20 train/test in both models.
* **Reports:** `classification_report`, `confusion_matrix`, and accuracy printed to console after training.
* **Schema control:** `feature_list.joblib` ensures inference columns match training order; web code aligns and adds missing columns (NaN → handled/filled).

---

## Notes & guardrails

* The app removes **leakage features** (e.g., `koi_fpflag_*`, `koi_score`) from UI input. If they appear in your `feature_list`, retrain without them.
* Parquet export requires **pyarrow**.
* Spectrometry model is optional; if artifacts aren’t present, the app returns a clear message on `/spectro/upload`.
* Exports are placed under `web/static/exports/` and linked back to the UI.

---

## Troubleshooting

* **`FileNotFoundError` for models:** Ensure you ran training or copied artifacts into `models/`, `build/`, and `models_macedo/` (relative to `web/`).
* **`to_parquet` errors:** `pip install pyarrow`.
* **CSV parse errors:** the uploader tries multiple parsers; if it still fails, check encoding, delimiter, or BOM.
* **Wrong columns:** batch files must use the same column names as the form/KOI schema. Spectrometry CSV must have columns matching the saved `models_macedo/feature_list.joblib` (order doesn’t matter; the server reindexes).

---

## License & attribution

* Built during the **NASA Space Apps Challenge in Montreal** hackathon by **Timothy Roch, Cyrus Yip, and Ariyan Khayer**.
* Data: NASA Kepler KOI tables and Macedo/Zalewski spectral feature datasets (follow their licenses/usage terms).
* Code license: MIT

---

### One-liner run (after training)

```bash
cd web && uvicorn main:app --reload --port 8000
```

Happy planet hunting! 🌍🔭
