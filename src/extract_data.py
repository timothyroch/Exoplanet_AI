# extract_data.py
import os
import numpy as np
import pandas as pd
import joblib

# ---- 1) Load KOI (note the path one level up from src) ----
DF_PATH = "../KOI_data.csv"   # <-- adjust if needed
df = pd.read_csv(DF_PATH, comment="#")

# ---- 2) Labels ----
label_map = {"CONFIRMED": "Confirmed", "CANDIDATE": "Candidate", "FALSE POSITIVE": "False"}
df = df[df["koi_disposition"].notna()].copy()
df["label"] = df["koi_disposition"].map(label_map)
df = df[df["label"].isin(["Confirmed", "Candidate", "False"])].copy()

# ---- 3) Base features (KOI cumulative commonly has these) ----
requested_features = [
    "koi_period", "koi_duration", "koi_depth", "koi_ror", "koi_impact",
    "koi_score", "koi_model_snr",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_steff", "koi_slogg", "koi_smet", "koi_kepmag",
]

available = [c for c in requested_features if c in df.columns]
missing = [c for c in requested_features if c not in df.columns]

if missing:
    print(f"[extract_data] Missing columns (will be filled as NaN and handled by XGBoost): {missing}")

# Build X starting from available columns
X = df[available].copy()

# Add any missing columns as NaN so downstream code has a consistent schema
for c in missing:
    X[c] = np.nan

# ---- 4) Optional engineered features (cheap & helpful) ----
X["log_period"] = np.log1p(df["koi_period"])
with np.errstate(divide="ignore", invalid="ignore"):
    X["dur_over_per"] = (df["koi_duration"] / df["koi_period"]).replace([np.inf, -np.inf], np.nan)
X["depth_sqrt_dur"] = df["koi_depth"] * np.sqrt(np.clip(df["koi_duration"], 0, None))

# Ensure fp flags are ints (XGBoost handles ints/bools fine)
for c in ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]:
    if c in X.columns:
        X[c] = X[c].astype("Int64")  # nullable int is fine

y = df["label"].copy()

# ---- 5) Save artifacts for train/visualize ----
os.makedirs("../build", exist_ok=True)
X.to_parquet("../build/X.parquet", index=False)
y.to_csv("../build/y.csv", index=False)
feature_list = list(X.columns)
joblib.dump(feature_list, "../build/feature_list.joblib")

# Optional: save groups to enable leakage-safe split later
if "kepid" in df.columns:
    df[["kepid"]].to_csv("../build/groups.csv", index=False)

print("[extract_data] Saved:")
print("  ../build/X.parquet")
print("  ../build/y.csv")
print("  ../build/feature_list.joblib")
if "kepid" in df.columns:
    print("  ../build/groups.csv")
