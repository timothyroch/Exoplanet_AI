import os
import numpy as np
import pandas as pd
import joblib

# ---- 1) Load datasets ----
# Load KOI and K2/Exoplanet datasets (note the paths one level up from src)
KOI_PATH = "../KOI_data.csv"                 # <-- Kepler Objects of Interest (KOI)
K2_PATH = "../k2pandc_dataset_exoplanet.csv" # <-- K2 and Exoplanet Archive data

print("[extract_data] Loading datasets...")
koi_df = pd.read_csv(KOI_PATH, comment="#")
k2_df = pd.read_csv(K2_PATH, comment="#")

print(f"  Loaded KOI dataset with {len(koi_df):,} rows")
print(f"  Loaded K2/Exoplanet dataset with {len(k2_df):,} rows")

# ---- 2) Labels ----
label_map = {"CONFIRMED": "Confirmed", "CANDIDATE": "Candidate", "FALSE POSITIVE": "False"}

# --- KOI labels ---
koi_df = koi_df[koi_df["koi_disposition"].notna()].copy()
koi_df["label"] = koi_df["koi_disposition"].map(label_map)
koi_df = koi_df[koi_df["label"].isin(["Confirmed", "Candidate", "False"])].copy()

# --- K2 labels ---
k2_df = k2_df[k2_df["disposition"].notna()].copy()
k2_df["label"] = k2_df["disposition"].map(label_map)
k2_df = k2_df[k2_df["label"].isin(["Confirmed", "Candidate"])].copy()  # K2 dataset doesn't include explicit False Positives

# ---- 3) Base features (KOI cumulative commonly has these) ----
# For KOI
requested_features = [
    "koi_period", "koi_duration", "koi_depth", "koi_ror", "koi_impact",
    "koi_score", "koi_model_snr",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_steff", "koi_slogg", "koi_smet", "koi_kepmag",
]

available = [c for c in requested_features if c in koi_df.columns]
missing = [c for c in requested_features if c not in koi_df.columns]

if missing:
    print(f"[extract_data] Missing columns in KOI (will be filled as NaN and handled by XGBoost): {missing}")

# Build X starting from available columns
X_koi = koi_df[available].copy()

# Add any missing columns as NaN so downstream code has a consistent schema
for c in missing:
    X_koi[c] = np.nan

# ---- Map K2 dataset columns to KOI equivalents ----
k2_df = k2_df.rename(columns={
    "pl_orbper": "koi_period",
    "pl_eqt": "koi_teq",
    "pl_insol": "koi_insol",
    "st_teff": "koi_steff",
    "st_logg": "koi_slogg",
    "st_met": "koi_smet",
    "sy_vmag": "koi_kepmag",
})

# Build matching K2 feature table
requested_features_k2 = [
    "koi_period", "koi_steff", "koi_slogg", "koi_smet", "koi_kepmag",
    "koi_teq", "koi_insol"
]

X_k2 = k2_df[[c for c in requested_features_k2 if c in k2_df.columns]].copy()

# ---- Align schemas between KOI and K2 ----
for c in requested_features:
    if c not in X_k2.columns:
        X_k2[c] = np.nan

# ---- 4) Optional engineered features (cheap & helpful) ----

# This line creates a new feature called log_period, which is the natural 
# logarithm of the planet’s orbital period (koi_period)
# np.log1p(x) = log(1 + x) — it’s numerically safer when x is small or zero
X_koi["log_period"] = np.log1p(koi_df["koi_period"])
X_k2["log_period"] = np.log1p(k2_df["koi_period"])

# This is the fraction of time the object is transiting compared to its full orbital period
# Planets typically have short transit durations relative to their period.
# False positives (e.g., binary stars) may show different duration-to-period ratios.

with np.errstate(divide="ignore", invalid="ignore"):
# np.errstate(divide="ignore", invalid="ignore") prevents annoying divide-by-zero warnings.
# .replace([np.inf, -np.inf], np.nan) ensures that divisions like duration / 0 become NaN, 
# not infinite — XGBoost can handle NaNs
    X_koi["dur_over_per"] = (koi_df["koi_duration"] / koi_df["koi_period"]).replace([np.inf, -np.inf], np.nan)
    X_k2["dur_over_per"] = np.nan  # K2 dataset lacks duration info

# Creates another derived feature combining transit depth (how much light dims) and duration.
# A rough proxy for signal strength — deeper and longer transits usually mean a stronger or 
# more easily detectable signal.
X_koi["depth_sqrt_dur"] = koi_df["koi_depth"] * np.sqrt(np.clip(koi_df["koi_duration"], 0, None))
X_k2["depth_sqrt_dur"] = np.nan  # missing in K2

# Ensure fp flags are ints (XGBoost handles ints/bools fine)
for c in ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]:
    if c in X_koi.columns:
        X_koi[c] = X_koi[c].astype("Int64")  # nullable int is fine
    if c in X_k2.columns:
        X_k2[c] = X_k2[c].astype("Int64")

# ---- Combine both datasets ----
X_koi["source"] = "Kepler"
X_k2["source"] = "K2/Exoplanet"

X = pd.concat([X_koi, X_k2], ignore_index=True)
y = pd.concat([koi_df["label"], k2_df["label"]], ignore_index=True)

print(f"[extract_data] Combined dataset shape: {X.shape}")
print(y.value_counts())

# ---- 5) Save artifacts for train/visualize ----
# Create the output directory and export all artifacts for downstream use:
#   - X.parquet: feature matrix (includes engineered features and consistent schema)
#   - y.csv: target labels (Confirmed / Candidate / False)
#   - feature_list.joblib: list of feature names to ensure consistent column order
#   - groups.csv (optional): kepid identifiers for group-based CV / leakage-safe splits

os.makedirs("../build", exist_ok=True)
X.to_parquet("../build/X.parquet", index=False)
y.to_csv("../build/y.csv", index=False)
feature_list = list(X.columns)
joblib.dump(feature_list, "../build/feature_list.joblib")

# Optional: save groups to enable leakage-safe split later
if "kepid" in koi_df.columns:
    koi_df[["kepid"]].to_csv("../build/groups.csv", index=False)

print("[extract_data] Saved:")
print("  ../build/X.parquet")
print("  ../build/y.csv")
print("  ../build/feature_list.joblib")
if "kepid" in koi_df.columns:
    print("  ../build/groups.csv")
print("[extract_data] Done.")
