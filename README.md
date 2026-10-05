# Subgroup Generalization Audit: Hospital Readmission Prediction

## 1. What does this do?
It trains a standard readmission-risk model on real hospital data, then checks whether the model's headline accuracy number holds for every patient group, or whether a good-looking average is hiding a large gap for one group.

## 2. How do I run it?
Clone, install, and run the scripts in order **from the repository folder** (they read `data/diabetic_data.csv` relative to it):

```bash
git clone https://github.com/Archit123456789/Python-Capstone-Project.git
cd Python-Capstone-Project
python3 -m pip install -r requirements.txt
python3 01_baseline.py            # crude end-to-end baseline, one AUC number
python3 02_subgroup_audit.py      # bootstrap CI audit by age / gender / race, plus calibration
python3 03_plots.py               # evaluation figures
python3 04_stronger_model.py      # does the gap survive a stronger model?
python3 05_mechanism_check.py     # first mechanism hypothesis (diagnosis complexity)
python3 06_mechanism_plot.py
python3 07_diagnosis_binning.py   # direct test of that hypothesis at full resolution
python3 08_charlson_index.py      # second direct test (Charlson comorbidity index)
```
Each script needs the previous ones to have run (they pass results along as `.joblib` files, which are generated and not stored in the repo).

* **Data:** `data/diabetic_data.csv` (UCI "Diabetes 130-US hospitals", 101,766 encounters) is included. No account or API key is needed.
* **LightGBM:** `04_stronger_model.py` uses LightGBM when it imports. If it does not (on macOS this usually means the `libomp` system library is missing: `brew install libomp`), the script automatically falls back to scikit-learn's `HistGradientBoostingClassifier` and says so on the first line of output. The fallback gives a slightly higher aggregate AUC (0.688 vs 0.657) and shows the same age gap.
* Last verified from a fresh clone on 2026-10-05: all eight scripts run start to finish.

## 3. What did you find?
**Aggregate:** logistic regression gets AUC 0.644, 95% CI [0.632, 0.656], well calibrated overall (ECE 0.009).

**That number hides an age gradient.** Discrimination is good for young patients and weak for the oldest:

| Age band | n | AUC | 95% CI |
|---|---|---|---|
| [20-30) | 324 | 0.808 | [0.722, 0.874] |
| [60-70) | 4,547 | 0.624 | [0.601, 0.650] |
| [80-90) | 3,414 | 0.603 | [0.574, 0.630] |
| [90-100) | 576 | 0.581 | [0.508, 0.657] |

The intervals for [20-30) and [90-100) do not overlap, so this is not sample-size noise. The youngest group is itself small, so I lean on the whole downward trend across the middle bands (thousands of patients each), not only the two ends.

**Gender:** no meaningful gap (0.648 vs 0.640; intervals overlap).
**Race:** Caucasian 0.642 and African American 0.645, no gap. Hispanic (n=404, AUC 0.738 [0.662, 0.812]) and Other (n=276, [0.557, 0.825]) have few readmissions and wide intervals, and Asian (n=123) fell below the audit threshold, so I do not draw conclusions about them.

**Calibration:** for the oldest groups the model stays well calibrated on average (ECE 0.009 at 70-90, 0.018 at 90-100) while discriminating poorly: it is not confidently wrong, it just does not separate cases. Young groups have higher calibration error (0.04 to 0.06), but with few patients in the high-probability bins.

**The gap is not a weak-model artifact.** LightGBM raises the aggregate AUC only from 0.644 to 0.657 and reproduces the gradient (20-30: 0.800 [0.725, 0.870]; 90-100: 0.598 [0.533, 0.662]; intervals do not overlap). The scikit-learn fallback model (aggregate 0.688) does too.

**What I could not explain, and a claim I retracted.** Age-band AUC correlates with average number of diagnoses at r = -0.80 over nine age-band points (`05_mechanism_check.py`), which suggested that diagnosis complexity drives the gap. I first reported that as the cause. Testing it directly (`07`, `08`) does not support it:

* Binning patients by their actual diagnosis count, r = -0.795 over 10 bins, but that is carried entirely by two tiny bins (n=53 with AUC 0.993; n=33 with AUC 0.344). Among bins with at least 500 patients, r = +0.07: AUC sits between 0.635 and 0.683 with no trend.
* Charlson comorbidity index (diabetes excluded): AUC 0.664, 0.643, 0.599, 0.651 for scores 0, 1-2, 3-4, 5+. It is not monotonic, and the last two intervals overlap the others.

So **the cause of the age gap is unexplained.** The correlation across age bands shows that age and diagnosis count move together, not that one explains the other, which is why the claim was retracted.

**Headline:** an aggregate AUC of 0.64 hides a real gap by age (about 0.81 for patients in their twenties vs about 0.58 for those over 90, non-overlapping intervals), it persists with a much stronger model, and I have not found its cause.

## 4. What would you do next, given more time?
* **Split by patient.** These scripts use a random row-level split, but about 30% of encounters come from repeat patients, so the same person can land in train and test and inflate scores. The extended version ([readmission-subgroup-audit](https://github.com/Archit123456789/readmission-subgroup-audit)) splits by patient: AUC fell from 0.688 to 0.676 and the age gap persisted.
* **Replicate on a second, more recent dataset.** This data ends in 2008, and nothing here shows the gap appears elsewhere.
* **Test other candidate causes directly** (for example how complete patients' records are, or admission source) with the same intervals-first rule, and run repeated splits so intervals reflect split variance.
* **Try to close the gap** (reweighting, group-specific thresholds). Either outcome would be informative.
