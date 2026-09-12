"""
Hour 5 — evaluation plots.
Two figures: (1) per-subgroup AUC with bootstrap CI whiskers, by age band
(2) reliability diagrams: aggregate vs youngest vs oldest age band.
"""
import numpy as np
import matplotlib.pyplot as plt
import joblib

art = joblib.load("audit_results.joblib")
agg_point, agg_lo, agg_hi, agg_ece = art["agg"]
age_results = art["age_results"]
agg_cal_tbl = art["agg_cal_tbl"]

# ---- Figure 1: AUC by age band with 95% CI whiskers ----
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(age_results))
aucs = age_results["auc"].values
lo = age_results["auc"].values - age_results["ci_lo"].values
hi = age_results["ci_hi"].values - age_results["auc"].values

ax.bar(x, aucs, yerr=[lo, hi], capsize=4, color="#4C72B0", alpha=0.85, label="Per age-band AUC (95% CI)")
ax.axhline(agg_point, color="#C44E52", linestyle="--", linewidth=2, label=f"Aggregate AUC = {agg_point:.3f}")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (AUC=0.5)")
ax.set_xticks(x)
ax.set_xticklabels(age_results["group"].values, rotation=45, ha="right")
ax.set_ylabel("AUC")
ax.set_title("Model discrimination collapses with age\n(aggregate AUC masks this gradient)")
ax.set_ylim(0.45, 1.0)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("fig1_auc_by_age.png", dpi=150)
print("Saved fig1_auc_by_age.png")

# ---- Figure 2: Reliability diagrams (aggregate, youngest, oldest) ----
import pandas as pd

art2 = joblib.load("hour1_artifacts.joblib")
y_test = art2["y_test"].reset_index(drop=True)
probs = pd.Series(art2["probs"]).reset_index(drop=True)
subgroups = art2["df_raw"].reset_index(drop=True)


def calibration_table(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        n = mask.sum()
        if n == 0:
            continue
        rows.append({"mean_predicted": y_prob[mask].mean(), "observed_freq": y_true[mask].mean(), "n": n})
    return pd.DataFrame(rows)


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

panels = [
    ("Aggregate (all patients)", np.ones(len(y_test), dtype=bool)),
    ("Age [20-30) — best discrimination", (subgroups["age"] == "[20-30)").values),
    ("Age [70-80) — worst discrimination", (subgroups["age"] == "[70-80)").values),
]

for ax, (title, mask) in zip(axes, panels):
    tbl = calibration_table(y_test[mask].values, probs[mask].values)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(tbl["mean_predicted"], tbl["observed_freq"], "-", color="#4C72B0", alpha=0.5, zorder=1)
    sizes = 15 + 200 * (tbl["n"] / tbl["n"].max())  # bigger marker = more data behind that bin
    ax.scatter(tbl["mean_predicted"], tbl["observed_freq"], s=sizes, color="#4C72B0",
               alpha=0.85, zorder=2, label="Model (size = bin n)")
    ax.set_title(f"{title}\n(n={mask.sum()})", fontsize=10)
    ax.set_xlabel("Mean predicted probability")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Observed frequency")
fig.suptitle("Calibration holds even where discrimination fails", y=1.02)
fig.tight_layout()
fig.savefig("fig2_reliability_diagrams.png", dpi=150, bbox_inches="tight")
print("Saved fig2_reliability_diagrams.png")
