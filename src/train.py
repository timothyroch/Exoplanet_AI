import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Load processed data ===
X = pd.read_parquet("../build/X.parquet")          
y = pd.read_csv("../build/y.csv")["label"]
obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    print("[train] Dropping non-numeric columns:", obj_cols)
    X = X.drop(columns=obj_cols)

# === Encode labels to integers 0..K-1 ===
le = LabelEncoder()
y_enc = le.fit_transform(y)                        

# (optional) show mapping in logs
print("[train] Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

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
print("Confusion Matrix (rows=true, cols=pred):\n",
      confusion_matrix(y_test, y_pred))

# === Save model and label encoder ===
os.makedirs("../models", exist_ok=True)
joblib.dump(xgb, "../models/xgb_koi.joblib")
joblib.dump(le, "../models/label_encoder.joblib")
print("Saved: ../models/xgb_koi.joblib and ../models/label_encoder.joblib")
