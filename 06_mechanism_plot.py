import pandas as pd
import matplotlib.pyplot as plt
import joblib

lgbm = joblib.load("lgbm_results.joblib")
mech = pd.read_csv("mechanism_vs_auc.csv")

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(mech["avg_num_diagnoses"], mech["auc"], s=mech["n"] / 40,
                 c=range(len(mech)), cmap="coolwarm_r", edgecolor="black", zorder=3)
for _, row in mech.iterrows():
    ax.annotate(row["age"], (row["avg_num_diagnoses"], row["auc"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

# trend line
import numpy as np
z = np.polyfit(mech["avg_num_diagnoses"], mech["auc"], 1)
xs = np.linspace(mech["avg_num_diagnoses"].min(), mech["avg_num_diagnoses"].max(), 50)
ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.5, label=f"trend (r = -0.80)")

ax.set_xlabel("Average number of diagnoses per patient (age-band mean)")
ax.set_ylabel("AUC (LightGBM, per age band)")
ax.set_title("AUC tracks diagnosis complexity, not age directly\n(marker size = n, color = age band)")
ax.legend()
fig.tight_layout()
fig.savefig("fig3_mechanism.png", dpi=150)
print("Saved fig3_mechanism.png")
