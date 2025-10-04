import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# === Load processed data ===
X = pd.read_parquet("../build/X.parquet")
y = pd.read_csv("../build/y.csv")["label"]

# === Feature Engineering (must mirror web main.py) ===
if (
    "koi_period" in X.columns
    and "koi_duration" in X.columns
    and "koi_depth" in X.columns
):
    X["log_period"] = np.log(X["koi_period"].replace(0, np.nan))
    # Handle potential divide-by-zero with safe replacement
    with np.errstate(divide="ignore", invalid="ignore"):
        X["dur_over_per"] = X["koi_duration"] / (X["koi_period"] * 24)
    X["depth_sqrt_dur"] = np.sqrt(X["koi_depth"]) / X["koi_duration"].replace(0, np.nan)
    # Replace any inf/nan from engineering with safe values (median fallback)
    for col in ["log_period", "dur_over_per", "depth_sqrt_dur"]:
        if col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
else:
    print(
        "[warn] Required columns for feature engineering missing; engineered features skipped."
    )

# === Encode labels to integers 0..K-1 ===
le = LabelEncoder()
y_enc = le.fit_transform(y)

# (optional) show mapping in logs
print("[train] Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# === Capture final feature list (order preserved) ===
feature_list = list(X.columns)

# === Model ===
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    learning_rate=0.05,
    n_estimators=400,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
)

# === Train ===
xgb.fit(X_train, y_train)

# === Predict & Evaluate ===
y_pred = xgb.predict(X_test)

# Reports with human-readable class names
target_names = list(le.classes_)
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
print("Confusion Matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))

# === Save model, label encoder, and feature list ===
os.makedirs("../models", exist_ok=True)
os.makedirs("../build", exist_ok=True)
joblib.dump(xgb, "../models/xgb_koi.joblib")
joblib.dump(le, "../models/label_encoder.joblib")
joblib.dump(feature_list, "../build/feature_list.joblib")
print(
    "✅ Saved: ../models/xgb_koi.joblib, ../models/label_encoder.joblib, ../build/feature_list.joblib"
)
print(f"[train] Final feature count: {len(feature_list)}")
