"""
Hour 1 — smallest possible end-to-end version.
Crude, but running: load -> minimal clean -> train -> one AUC number.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

# ---- Load ----
df = pd.read_csv("data/diabetic_data.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ---- Target: binary readmission <30 days ----
df["target"] = (df["readmitted"] == "<30").astype(int)
print(f"Positive rate: {df['target'].mean():.3f}")

# ---- Minimal cleaning (crude, revisit in hours 2-4) ----
# Drop known-bad columns: near-all-missing, zero-variance, or leakage-risk IDs
drop_cols = [
    "encounter_id", "patient_nbr", "readmitted",
    "weight", "payer_code", "medical_specialty",  # >50% missing, defer decision
    "examide", "citoglipton",  # zero variance in this dataset
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Replace '?' placeholder with NaN, then simplest possible handling: mode-fill categoricals
df = df.replace("?", np.nan)

# Keep race and gender and age as our subgroup axes for later — don't drop them
subgroup_cols = ["race", "gender", "age"]

X = df.drop(columns=["target"])
y = df["target"]

# Simple encoding: categoricals -> category codes (crude; revisit encoding in hours 2-4)
cat_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category").cat.codes  # -1 for NaN, fine for a crude pass

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# ---- Split (stratified) ----
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---- Train: plain logistic regression ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf.fit(X_train_s, y_train)

# ---- The one aggregate number ----
probs = clf.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\n=== AGGREGATE RESULT ===")
print(f"Test AUC: {auc:.4f}")

# Save everything hour 2-4 needs so we don't retrain from scratch
import joblib
joblib.dump(
    {
        "model": clf, "scaler": scaler, "X_test": X_test, "y_test": y_test,
        "probs": probs, "idx_test": idx_test, "subgroup_cols": subgroup_cols,
        "df_raw": df.loc[idx_test, subgroup_cols],  # raw (unencoded) subgroup labels for test set
    },
    "hour1_artifacts.joblib",
)
print("Saved hour1_artifacts.joblib")
