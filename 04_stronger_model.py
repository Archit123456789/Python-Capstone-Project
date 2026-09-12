"""
Experiment 1 — does the age gradient survive a stronger model?
Same pipeline, swap logistic regression for LightGBM (much higher capacity).
If the gap persists, it's a property of the data/task, not a weak-model artifact.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

RANDOM_STATE = 42
N_BOOT = 1000
rng = np.random.default_rng(RANDOM_STATE)

df = pd.read_csv("data/diabetic_data.csv")
df["target"] = (df["readmitted"] == "<30").astype(int)

drop_cols = ["encounter_id", "patient_nbr", "readmitted", "weight", "payer_code",
             "medical_specialty", "examide", "citoglipton"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df = df.replace("?", np.nan)

subgroup_cols = ["race", "gender", "age"]
X = df.drop(columns=["target"])
y = df["target"]

cat_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")  # LightGBM handles categoricals natively, no manual encoding needed

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    random_state=RANDOM_STATE, verbosity=-1
)
clf.fit(X_train, y_train, categorical_feature=cat_cols)

probs = clf.predict_proba(X_test)[:, 1]
agg_auc = roc_auc_score(y_test, probs)
print(f"LightGBM aggregate AUC: {agg_auc:.4f}  (logistic regression was 0.644)")

y_test_r = y_test.reset_index(drop=True)
probs_r = pd.Series(probs).reset_index(drop=True)
subgroups = df.loc[idx_test, subgroup_cols].reset_index(drop=True)

def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOT):
    n = len(y_true)
    if n < 20 or y_true.nunique() < 2:
        return np.nan, np.nan, np.nan
    point = roc_auc_score(y_true, y_prob)
    boots = []
    yt, yp = y_true.values, y_prob.values
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = yt[idx], yp[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi

AGE_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]
print(f"\n{'AGE BAND':15s}{'n':>8s}{'AUC (LGBM)':>12s}{'95% CI':>20s}")
lgbm_age_results = []
for g in AGE_ORDER:
    mask = (subgroups["age"] == g).values
    n = mask.sum()
    if n < 100:
        continue
    point, lo, hi = bootstrap_auc_ci(y_test_r[mask], probs_r[mask])
    lgbm_age_results.append({"group": g, "n": n, "auc": point, "ci_lo": lo, "ci_hi": hi})
    print(f"{g:15s}{n:8d}{point:12.3f}   [{lo:.3f}, {hi:.3f}]")

lgbm_age_df = pd.DataFrame(lgbm_age_results)
joblib.dump({"agg_auc": agg_auc, "age_results": lgbm_age_df, "probs": probs, "y_test": y_test_r,
             "subgroups": subgroups}, "lgbm_results.joblib")
print("\nSaved lgbm_results.joblib")
