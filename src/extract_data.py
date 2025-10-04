# extract_data.py
import os

import joblib
import numpy as np
import pandas as pd

# ---- 1) Load KOI (note the path one level up from src) ----
DF_PATH = "../KOI_data.csv"  # <-- adjust if needed
df = pd.read_csv(DF_PATH, comment="#")

# ---- 2) Labels ----
label_map = {
    "CONFIRMED": "Confirmed",
    "CANDIDATE": "Candidate",
    "FALSE POSITIVE": "False",
}
df = df[df["koi_disposition"].notna()].copy()
df["label"] = df["koi_disposition"].map(label_map)
df = df[df["label"].isin(["Confirmed", "Candidate", "False"])].copy()

# ---- 3) Whitelisted physical / observational features ----
# Keep only features that would be known at (or near) initial vetting time.
# Excluded: post-vetting flags (koi_fpflag_*), disposition score (koi_score).
WHITELIST_FEATURES = [
    "koi_period",
    "koi_duration",  # hours
    "koi_depth",  # ppm
    "koi_ror",
    "koi_impact",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_kepmag",
]

available = [c for c in WHITELIST_FEATURES if c in df.columns]
missing = [c for c in WHITELIST_FEATURES if c not in df.columns]

if missing:
    print(
        f"[extract_data] Missing columns (will be filled as NaN and handled by XGBoost): {missing}"
    )

# Build X starting from available columns
X = df[available].copy()

# Add any missing columns as NaN so downstream code has a consistent schema
for c in missing:
    X[c] = np.nan

# ---- 4) Engineered features (align with web/main.py inference logic) ----
if "koi_period" in X.columns:
    X["log_period"] = np.log(df["koi_period"].replace(0, np.nan))
else:
    X["log_period"] = np.nan

if "koi_duration" in X.columns and "koi_period" in X.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        X["dur_over_per"] = (df["koi_duration"] / (df["koi_period"] * 24)).replace(
            [np.inf, -np.inf], np.nan
        )
else:
    X["dur_over_per"] = np.nan

if "koi_depth" in X.columns and "koi_duration" in X.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        X["depth_sqrt_dur"] = np.sqrt(df["koi_depth"]) / df["koi_duration"].replace(
            0, np.nan
        )
else:
    X["depth_sqrt_dur"] = np.nan

# Fill any NaNs from engineering with median (fallback to 0 if all NaN)
for col in ["log_period", "dur_over_per", "depth_sqrt_dur"]:
    if X[col].isna().any():
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)

y = df["label"].copy()

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
if "kepid" in df.columns:
    df[["kepid"]].to_csv("../build/groups.csv", index=False)

print("[extract_data] Saved:")
print("  ../build/X.parquet")
print("  ../build/y.csv")
print("  ../build/feature_list.joblib")
if "kepid" in df.columns:
    print("  ../build/groups.csv")
