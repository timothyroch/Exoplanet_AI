import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, log_loss

# --- Load processed data ---
X = pd.read_parquet("../build/X.parquet")
y = pd.read_csv("../build/y.csv")["label"]

# Load groups if present and aligned
groups = None
groups_path = "../build/groups.csv"
if os.path.exists(groups_path):
    gdf = pd.read_csv(groups_path)
    col = "kepid" if "kepid" in gdf.columns else ("group" if "group" in gdf.columns else None)
    if col is not None:
        groups = gdf[col]

# --- Ensure labels are present ---
mask = y.notna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)
if groups is not None:
    groups = groups.loc[mask].reset_index(drop=True)

# --- Encode labels ---
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("[train] Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# --- Split train/test ---
use_group_split = groups is not None and len(groups) == len(X)
if use_group_split:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y_enc, groups=groups))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y_enc[tr_idx], y_enc[te_idx]
else:
    if groups is not None and len(groups) != len(X):
        print(f"[train] Warning: groups length {len(groups)} != X length {len(X)}; using standard split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

# --- Model base params (conservative; no early stopping) ---
base_params = dict(
    objective="multi:softprob",
    num_class=len(le.classes_),
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
)

xgb = XGBClassifier(**base_params)

# --- CV over n_estimators (since we can't use early stopping) ---
# Use a modest grid to keep runtime reasonable. Adjust if you want more/less training.
param_grid = {
    "n_estimators": [200, 400, 800, 1200, 1600, 2000]
}

# Use stratified K-fold for multiclass
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Negative log loss (because GridSearchCV maximizes the score)
neg_logloss = make_scorer(lambda yt, yp: -log_loss(yt, yp, labels=np.arange(len(le.classes_))), needs_proba=True)

gs = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=neg_logloss,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,   # retrain on full training set with best params
)

print("[train] Tuning n_estimators via 3-fold CV...")
gs.fit(X_train, y_train)

print(f"[train] Best params: {gs.best_params_}")
print(f"[train] Best CV logloss: {-gs.best_score_:.5f}")

best_model = gs.best_estimator_

# --- Evaluate ---
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=list(le.classes_), digits=4))
print("Confusion Matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))

# --- Save artifacts ---
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/xgb_koi.joblib")
joblib.dump(le, "../models/label_encoder.joblib")
print("Saved: ../models/xgb_koi.joblib and ../models/label_encoder.joblib")
