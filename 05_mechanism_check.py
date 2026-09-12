"""
Experiment 2 — mechanism check.
Does the elderly AUC drop trace to less/noisier information being available
for older patients (fewer labs/meds logged, more missing data, more diagnoses
crowding the signal)? Or is the model just seeing genuinely harder cases?
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/diabetic_data.csv")
df = df.replace("?", np.nan)

AGE_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]

print(f"{'AGE BAND':12s}{'n':>7s}{'lab_proc':>10s}{'n_meds':>8s}{'n_diag':>8s}{'n_emerg':>9s}{'n_inpat':>9s}{'diag_miss%':>11s}")
rows = []
for g in AGE_ORDER:
    sub = df[df["age"] == g]
    n = len(sub)
    row = {
        "age": g, "n": n,
        "avg_lab_procedures": sub["num_lab_procedures"].mean(),
        "avg_medications": sub["num_medications"].mean(),
        "avg_num_diagnoses": sub["number_diagnoses"].mean(),
        "avg_emergency_visits": sub["number_emergency"].mean(),
        "avg_inpatient_visits": sub["number_inpatient"].mean(),
        "diag3_missing_pct": sub["diag_3"].isna().mean() * 100,
    }
    rows.append(row)
    print(f"{g:12s}{n:7d}{row['avg_lab_procedures']:10.1f}{row['avg_medications']:8.1f}"
          f"{row['avg_num_diagnoses']:8.1f}{row['avg_emergency_visits']:9.2f}"
          f"{row['avg_inpatient_visits']:9.2f}{row['diag3_missing_pct']:11.1f}")

mech_df = pd.DataFrame(rows)
mech_df.to_csv("mechanism_check.csv", index=False)

print("\n--- Correlation of each factor with age-band AUC (from LightGBM run) ---")
import joblib
lgbm = joblib.load("lgbm_results.joblib")
age_auc = lgbm["age_results"].set_index("group")["auc"]
merged = mech_df.set_index("age").join(age_auc)
merged = merged.dropna()
print(merged[["avg_num_diagnoses", "avg_inpatient_visits", "avg_emergency_visits", "diag3_missing_pct", "auc"]]
      .corr()["auc"].sort_values())
merged.to_csv("mechanism_vs_auc.csv")
