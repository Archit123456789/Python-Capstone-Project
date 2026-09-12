"""
Upgrade 1 -- test the mechanism directly instead of through the age proxy.
Original claim was based on correlating AUC against diagnosis count across
only 9 age-band data points. This reruns the audit binned DIRECTLY by
number_diagnoses (up to 16 bins, most with thousands of patients) -- a much
stronger test of the same claim, and one that doesn't route through age at all.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
N_BOOT = 1000
rng = np.random.default_rng(RANDOM_STATE)

lgbm = joblib.load("lgbm_results.joblib")
y_test = lgbm["y_test"]
probs = pd.Series(lgbm["probs"]).reset_index(drop=True)

# Need number_diagnoses for the test set -- reload and re-align via the same split
df = pd.read_csv("data/diabetic_data.csv")
df["target"] = (df["readmitted"] == "<30").astype(int)
drop_cols = ["encounter_id", "patient_nbr", "readmitted", "weight", "payer_code",
             "medical_specialty", "examide", "citoglipton"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df = df.replace("?", np.nan)

from sklearn.model_selection import train_test_split
X = df.drop(columns=["target"])
y = df["target"]
_, _, _, _, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
num_diag = df.loc[idx_test, "number_diagnoses"].reset_index(drop=True)

def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOT):
    n = len(y_true)
    if n < 30 or y_true.nunique() < 2:
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

# Bin: 1-8 individually, 9 alone (huge mass point), 10+ grouped (small n)
def bin_diag(n):
    if n <= 8:
        return str(int(n))
    elif n == 9:
        return "9"
    else:
        return "10+"

bins = num_diag.apply(bin_diag)
order = [str(i) for i in range(1, 10)] + ["10+"]

print(f"{'diag_bin':10s}{'n':>8s}{'AUC':>10s}{'95% CI':>20s}")
rows = []
for b in order:
    mask = (bins == b).values
    n = mask.sum()
    if n < 30:
        print(f"{b:10s}{n:8d}  (skipped, n<30)")
        continue
    point, lo, hi = bootstrap_auc_ci(y_test[mask], probs[mask])
    rows.append({"diag_bin": b, "n_diagnoses": int(b) if b != "10+" else 12, "n": n, "auc": point, "ci_lo": lo, "ci_hi": hi})
    print(f"{b:10s}{n:8d}{point:10.3f}   [{lo:.3f}, {hi:.3f}]")

result_df = pd.DataFrame(rows)
r = result_df["n_diagnoses"].corr(result_df["auc"])
print(f"\nCorrelation (AUC vs number_diagnoses, direct binning, n={len(result_df)} bins): r = {r:.3f}")
print(f"Total patients covered by this analysis: {result_df['n'].sum()} (vs. 9 age-band points previously)")

joblib.dump({"result_df": result_df, "r": r}, "diagnosis_binning_results.joblib")
result_df.to_csv("diagnosis_binning_results.csv", index=False)
print("\nSaved diagnosis_binning_results.joblib / .csv")
