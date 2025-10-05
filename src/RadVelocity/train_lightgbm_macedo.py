import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier


def main():
    ap = argparse.ArgumentParser(description="Train LightGBM on Macedo/Zalewski feature CSVs")
    ap.add_argument("--csv", required=True, help="Path to all_global.csv or all_local.csv")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--models-dir", default="../models_macedo",
                    help="Where to save model artifacts (default ../models_macedo)")
    args = ap.parse_args()

    print(f"[load] Reading {args.csv} ...")
    df = pd.read_csv(args.csv)

    # Expect last column named 'label' based on the dataset description
    if "label" not in df.columns:
        raise ValueError("Could not find a 'label' column in the CSV. Please confirm the file format.")

    # Separate features/labels
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].copy()
    y = df["label"].astype(str)

    print(f"[shape] X: {X.shape}, y: {y.shape}")
    print("[label] sample counts:\n", y.value_counts())

    # --- Make sure features are numeric and usable ---
    # 1) Force numeric (non-numeric -> NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # 2) Clean up inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # 3) Drop columns that are all NaN
    all_nan = X.columns[X.isna().all(axis=0)]
    if len(all_nan):
        print(f"[clean] Dropping all-NaN columns: {len(all_nan)}")
        X = X.drop(columns=all_nan)

    # 4) Drop constant columns (no variance)
    const_cols = X.columns[(X.nunique(dropna=False) <= 1)]
    if len(const_cols):
        print(f"[clean] Dropping constant columns: {len(const_cols)}")
        X = X.drop(columns=const_cols)

    # 5) Fill remaining NaNs with column medians (safe default)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    print("[clean] Final shape after sanity checks:", X.shape)
    print("[clean] Sample dtypes:", X.dtypes.iloc[:10].to_dict())

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("[label] mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.random_state, stratify=y_enc
    )

    # Class weights (inverse frequency)
    counts = np.bincount(y_train)
    class_weight = {cls: (len(y_train) / (cnt + 1e-9)) for cls, cnt in enumerate(counts)}
    print("[class_weight]", class_weight)

    # LightGBM model
    is_multiclass = len(le.classes_) > 2
    clf = LGBMClassifier(
        objective="multiclass" if is_multiclass else "binary",
        num_class=len(le.classes_) if is_multiclass else None,
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.random_state,
        class_weight=class_weight  # works for binary or multiclass (dict of class_index -> weight)
    )

    # Some LightGBM builds support early_stopping in the sklearn API; guard it
    fit_kwargs = {}
    fit_params = getattr(clf.fit, "__code__", None)
    if fit_params and "early_stopping_rounds" in fit_params.co_varnames:
        fit_kwargs.update({
            "eval_set": [(X_test, y_test)],
            "eval_metric": "multi_logloss" if is_multiclass else "logloss",
            "early_stopping_rounds": 50,
            "verbose": False
        })

    print("[train] Fitting LightGBM...")
    clf.fit(X_train, y_train, **fit_kwargs)

    # Evaluate
    y_pred = clf.predict(X_test)
    target_names = list(le.classes_)
    print("\n=== Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    print("Confusion Matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save artifacts
    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "lightgbm_macedo.joblib")
    le_path = os.path.join(args.models_dir, "label_encoder.joblib")
    feat_path = os.path.join(args.models_dir, "feature_list.joblib")

    joblib.dump(clf, model_path)
    joblib.dump(le, le_path)
    joblib.dump(list(X.columns), feat_path)  # save the post-cleaning feature list

    print(f"[save] Model: {model_path}")
    print(f"[save] LabelEncoder: {le_path}")
    print(f"[save] Feature list: {feat_path}")


if __name__ == "__main__":
    main()
