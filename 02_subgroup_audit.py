"""
Hours 2-4 — the actual investigation.
Does the aggregate AUC mask per-subgroup gaps? Bootstrap CIs separate
real gaps from small-sample noise. Also check calibration per subgroup.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
N_BOOT = 1000
rng = np.random.default_rng(RANDOM_STATE)

art = joblib.load("hour1_artifacts.joblib")
y_test = art["y_test"].reset_index(drop=True)
probs = pd.Series(art["probs"]).reset_index(drop=True)
subgroups = art["df_raw"].reset_index(drop=True)  # race, gender, age (raw labels)

AGE_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]


def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOT):
    """Bootstrap 95% CI for AUC. Returns (point_estimate, lo, hi)."""
    n = len(y_true)
    if n < 20 or y_true.nunique() < 2:
        return np.nan, np.nan, np.nan
    point = roc_auc_score(y_true, y_prob)
    boots = []
    y_true_arr = y_true.values
    y_prob_arr = y_prob.values
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y_true_arr[idx], y_prob_arr[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, pb))
    if len(boots) < 50:
        return point, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def audit_axis(axis_name, min_n=100):
    print(f"\n{'='*60}\nSUBGROUP AXIS: {axis_name}\n{'='*60}")
    rows = []
    groups = subgroups[axis_name].unique()
    if axis_name == "age":
        groups = [g for g in AGE_ORDER if g in groups]
    for g in groups:
        mask = subgroups[axis_name] == g
        n = mask.sum()
        if n < min_n:
            print(f"  {g}: n={n} (skipped, below min_n={min_n})")
            continue
        yt, yp = y_test[mask], probs[mask]
        point, lo, hi = bootstrap_auc_ci(yt, yp)
        rows.append({"group": g, "n": n, "pos_rate": yt.mean(), "auc": point, "ci_lo": lo, "ci_hi": hi})
        print(f"  {g:20s} n={n:6d}  pos_rate={yt.mean():.3f}  AUC={point:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
    return pd.DataFrame(rows)


def calibration_table(y_true, y_prob, n_bins=10):
    """Reliability table: predicted-prob bin vs observed frequency."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        n = mask.sum()
        if n == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        rows.append({"bin": b, "n": n, "mean_predicted": conf, "observed_freq": acc, "gap": abs(conf - acc)})
    tbl = pd.DataFrame(rows)
    ece = (tbl["gap"] * tbl["n"]).sum() / tbl["n"].sum()
    return tbl, ece


# ---- Aggregate baseline (the "one number" everyone reports) ----
agg_point, agg_lo, agg_hi = bootstrap_auc_ci(y_test, probs)
print(f"AGGREGATE (baseline claim): AUC={agg_point:.3f}  95% CI=[{agg_lo:.3f}, {agg_hi:.3f}]")
agg_tbl, agg_ece = calibration_table(y_test, probs)
print(f"AGGREGATE ECE: {agg_ece:.4f}")

# ---- Per-subgroup audit ----
age_results = audit_axis("age")
gender_results = audit_axis("gender")
race_results = audit_axis("race", min_n=200)  # smaller groups here, be stricter

# ---- Calibration per subgroup for the axis most likely to show something ----
print(f"\n{'='*60}\nCALIBRATION BY AGE BAND\n{'='*60}")
cal_results = []
for g in AGE_ORDER:
    mask = (subgroups["age"] == g).values
    if mask.sum() < 100:
        continue
    tbl, ece = calibration_table(y_test[mask].values, probs[mask].values)
    cal_results.append({"age_band": g, "n": mask.sum(), "ece": ece})
    print(f"  {g:12s} n={mask.sum():6d}  ECE={ece:.4f}")
cal_df = pd.DataFrame(cal_results)

# ---- Save everything for the plotting/writeup pass ----
joblib.dump(
    {
        "agg": (agg_point, agg_lo, agg_hi, agg_ece),
        "age_results": age_results, "gender_results": gender_results,
        "race_results": race_results, "cal_df": cal_df,
        "agg_cal_tbl": agg_tbl,
    },
    "audit_results.joblib",
)
print("\nSaved audit_results.joblib")
