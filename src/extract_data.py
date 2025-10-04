import os
import numpy as np
import pandas as pd
import joblib

# ---- 1) Load datasets ----
KOI_PATH = "../KOI_data.csv"
K2_PATH = "../k2pandc_dataset_exoplanet.csv"

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
k2_df = k2_df[k2_df["label"].isin(["Confirmed", "Candidate"])].copy()

# ---- 3) Base features ----
requested_features = [
    "koi_period", "koi_duration", "koi_depth", "koi_ror", "koi_impact",
    "koi_score", "koi_model_snr",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_steff", "koi_slogg", "koi_smet", "koi_kepmag",
]

available = [c for c in requested_features if c in koi_df.columns]
missing = [c for c in requested_features if c not in koi_df.columns]

if missing:
    print(f"[extract_data] Missing columns in KOI (will be filled as NaN): {missing}")

X_koi = koi_df[available].copy()
for c in missing:
    X_koi[c] = np.nan

# ---- Map K2 dataset columns ----
k2_df = k2_df.rename(columns={
    "pl_orbper": "koi_period",
    "pl_eqt": "koi_teq",
    "pl_insol": "koi_insol",
    "st_teff": "koi_steff",
    "st_logg": "koi_slogg",
    "st_met": "koi_smet",
    "sy_vmag": "koi_kepmag",
})

requested_features_k2 = [
    "koi_period", "koi_steff", "koi_slogg", "koi_smet", "koi_kepmag",
    "koi_teq", "koi_insol"
]

X_k2 = k2_df[[c for c in requested_features_k2 if c in k2_df.columns]].copy()

# Align schemas
for c in requested_features:
    if c not in X_k2.columns:
        X_k2[c] = np.nan

# ---- 4) Engineered features ----
X_koi["log_period"] = np.log1p(koi_df["koi_period"])
X_k2["log_period"] = np.log1p(k2_df["koi_period"])

with np.errstate(divide="ignore", invalid="ignore"):
    X_koi["dur_over_per"] = (koi_df["koi_duration"] / koi_df["koi_period"]).replace([np.inf, -np.inf], np.nan)
    X_k2["dur_over_per"] = np.nan

X_koi["depth_sqrt_dur"] = koi_df["koi_depth"] * np.sqrt(np.clip(koi_df["koi_duration"], 0, None))
X_k2["depth_sqrt_dur"] = np.nan

for c in ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]:
    if c in X_koi.columns:
        X_koi[c] = X_koi[c].astype("Int64")
    if c in X_k2.columns:
        X_k2[c] = X_k2[c].astype("Int64")

# ---- Combine datasets ----
X_koi["source"] = "Kepler"
X_k2["source"] = "K2/Exoplanet"

X = pd.concat([X_koi, X_k2], ignore_index=True)
y = pd.concat([koi_df["label"], k2_df["label"]], ignore_index=True)
X["source"] = X["source"].map({"Kepler": 0.0, "K2/Exoplanet": 1.0}).astype("float32")
print("[extract_data] source counts:\n", X["source"].value_counts())
print(f"[extract_data] Combined dataset shape: {X.shape}")
print(y.value_counts())

# ---- Build aligned groups ----
if "kepid" in koi_df.columns:
    groups_koi = koi_df["kepid"].reset_index(drop=True)
else:
    groups_koi = pd.Series(np.arange(len(koi_df)), name="kepid")

groups_k2 = pd.Series(np.arange(len(k2_df)) + 10_000_000, name="kepid")
groups_all = pd.concat([groups_koi, groups_k2], ignore_index=True)
assert len(groups_all) == len(X), "Groups length must match X/y length"

# ---- 5) Save artifacts ----
os.makedirs("../build", exist_ok=True)
X.to_parquet("../build/X.parquet", index=False)
y.to_csv("../build/y.csv", index=False)
feature_list = list(X.columns)
joblib.dump(feature_list, "../build/feature_list.joblib")
groups_all.to_csv("../build/groups.csv", index=False)

print("[extract_data] Saved:")
print("  ../build/X.parquet")
print("  ../build/y.csv")
print("  ../build/feature_list.joblib")
print("  ../build/groups.csv")
print("[extract_data] Done.")
