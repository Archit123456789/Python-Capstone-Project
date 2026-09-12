"""
Upgrade 2 -- replace the crude "number of diagnoses" proxy with a real,
clinically validated severity measure: the Charlson Comorbidity Index
(Deyo 1992 ICD-9-CM adaptation), computed from diag_1/diag_2/diag_3.

WHY THE DIABETES COMPONENT IS EXCLUDED -- READ BEFORE USING THIS SCORE:
The Charlson index normally includes two diabetes-related categories
("diabetes without complication", weight 1; "diabetes with complication",
weight 2). This cohort's inclusion criterion (per the original dataset
documentation) is that every encounter has a diabetes diagnosis somewhere
in the patient's record. Diabetes is therefore not an independent
comorbidity that varies across patients here -- it is the population-
defining condition, present at ~100% by construction.

Empirical check: only 37.4% of encounters show an explicit 250.xx code in
diag_1-3 (38,024 / 101,766), despite diabetes being true for essentially the
whole cohort. That gap is not real prevalence variation -- it's an artifact
of which diagnosis happened to be coded first, second, or third for a given
encounter (diag_1-3 are the top 3 recorded diagnoses, not an exhaustive
list). Including the diabetes component here would therefore inject
diagnosis-ORDERING noise, not real between-patient signal, and would
misrepresent the score as differentiating patients on a dimension that's
actually constant. Excluding it is a deliberate scope decision, not an
oversight.

LIMITATION: because only the top 3 diagnoses are available (not the full
coded history), this Charlson score is a lower bound / undercount relative
to a full-chart Charlson calculation. Treat it as a relative severity
ranking within this dataset, not an absolute clinical score.
"""
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_BOOT = 1000
rng = np.random.default_rng(RANDOM_STATE)


def _in_range(code_num, lo, hi):
    return lo <= code_num <= hi


def charlson_category(code):
    """Map one ICD-9 code to a Charlson category weight. Diabetes excluded (see module docstring)."""
    if pd.isna(code):
        return None
    code = str(code).strip()
    if code in ("", "?"):
        return None
    if code.startswith("V") or code.startswith("E"):
        return None  # V/E codes are supplementary classifications, not disease categories here

    try:
        num = float(code)
    except ValueError:
        return None

    # (category_name, weight, ranges as (lo, hi))
    rules = [
        ("MI", 1, [(410, 410.99), (412, 412.99)]),
        ("CHF", 1, [(428, 428.99)]),
        ("PVD", 1, [(441, 441.99), (443.9, 443.9), (785.4, 785.4)]),
        ("CVD", 1, [(430, 438.99)]),
        ("Dementia", 1, [(290, 290.99)]),
        ("COPD", 1, [(490, 496.99), (500, 505.99), (506.4, 506.4)]),
        ("Rheumatic", 1, [(710.0, 710.1), (710.4, 710.4), (714.0, 714.2), (714.81, 714.81), (725, 725.99)]),
        ("PepticUlcer", 1, [(531, 534.99)]),
        ("MildLiver", 1, [(571.2, 571.2), (571.5, 571.6), (571.4, 571.49)]),
        # Diabetes categories intentionally omitted -- see module docstring
        ("Hemiplegia", 2, [(342, 342.99), (344.1, 344.1)]),
        ("Renal", 2, [(582, 582.99), (583, 583.99), (585, 585.99), (586, 586.99), (588, 588.99)]),
        ("Malignancy", 2, [(140, 172.99), (174, 195.99), (200, 208.99)]),
        ("ModSevereLiver", 3, [(572.2, 572.8), (456.0, 456.2)]),
        ("MetastaticTumor", 6, [(196, 199.99)]),
        ("AIDS", 6, [(42, 44.99)]),
    ]

    matched = []
    for name, weight, ranges in rules:
        for lo, hi in ranges:
            if _in_range(num, lo, hi):
                matched.append((name, weight))
                break
    return matched


def charlson_score_for_row(diag1, diag2, diag3):
    seen = {}
    for code in (diag1, diag2, diag3):
        matches = charlson_category(code)
        if matches:
            for name, weight in matches:
                seen[name] = weight  # dedupe by category, not by which diag field
    return sum(seen.values())


print("Computing Charlson scores from diag_1/diag_2/diag_3 (this takes a moment)...")
df = pd.read_csv("data/diabetic_data.csv")
df = df.replace("?", np.nan)
df["charlson"] = [
    charlson_score_for_row(d1, d2, d3)
    for d1, d2, d3 in zip(df["diag_1"], df["diag_2"], df["diag_3"])
]
print(df["charlson"].value_counts().sort_index())

# ---- Align to the same test split used throughout ----
df["target"] = (pd.read_csv("data/diabetic_data.csv")["readmitted"] == "<30").astype(int)
drop_cols = ["encounter_id", "patient_nbr", "readmitted", "weight", "payer_code",
             "medical_specialty", "examide", "citoglipton"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["target", "charlson"])
y = df["target"]
_, _, _, _, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
charlson_test = df.loc[idx_test, "charlson"].reset_index(drop=True)

lgbm = joblib.load("lgbm_results.joblib")
y_test = lgbm["y_test"]
probs = pd.Series(lgbm["probs"]).reset_index(drop=True)


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


def bin_charlson(c):
    if c == 0:
        return "0"
    elif c <= 2:
        return "1-2"
    elif c <= 4:
        return "3-4"
    else:
        return "5+"


bins = charlson_test.apply(bin_charlson)
order = ["0", "1-2", "3-4", "5+"]

print(f"\n{'charlson_bin':14s}{'n':>8s}{'AUC':>10s}{'95% CI':>20s}")
rows = []
for b in order:
    mask = (bins == b).values
    n = mask.sum()
    if n < 30:
        print(f"{b:14s}{n:8d}  (skipped, n<30)")
        continue
    point, lo, hi = bootstrap_auc_ci(y_test[mask], probs[mask])
    rows.append({"charlson_bin": b, "n": n, "auc": point, "ci_lo": lo, "ci_hi": hi})
    print(f"{b:14s}{n:8d}{point:10.3f}   [{lo:.3f}, {hi:.3f}]")

result_df = pd.DataFrame(rows)
joblib.dump({"result_df": result_df, "charlson_test": charlson_test, "y_test": y_test, "probs": probs},
            "charlson_results.joblib")
result_df.to_csv("charlson_results.csv", index=False)
print("\nSaved charlson_results.joblib / .csv")
