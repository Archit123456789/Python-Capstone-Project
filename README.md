# Subgroup Generalization Audit — Hospital Readmission Prediction

## 1. What does this do?
It trains a standard readmission-risk model on real hospital data, then checks whether the model's headline accuracy number actually holds up for every patient group — or whether it's hiding a large gap for one group behind a good-looking average.

## 2. How do I run it?
```bash
pip install -r requirements.txt
python3 01_baseline.py          # hour 1: crude end-to-end baseline, one AUC number
python3 02_subgroup_audit.py    # hours 2-4: bootstrap CI audit by age/gender/race
python3 03_plots.py             # hour 5: evaluation figures
python3 04_stronger_model.py    # extension: does the gap survive a stronger model? (LightGBM)
python3 05_mechanism_check.py   # extension: what actually drives the gap? (diagnosis complexity)
```
Data: `data/diabetic_data.csv` (UCI "Diabetes 130-US hospitals" dataset, 101,766 encounters) is included in this repo. No API key or account needed.

## 3. What did you find?
The aggregate model looks fine (AUC = 0.644, 95% CI [0.632, 0.656]) — but that single number hides a real age gradient. The model discriminates well for patients under 30 (AUC ≈ 0.81) and is barely better than chance for patients over 80 (AUC ≈ 0.58–0.60); the confidence intervals for the youngest and oldest groups don't overlap, so this isn't sample-size noise. Gender showed no meaningful gap (0.648 vs 0.640). Race showed no gap between the two largest groups (Caucasian, African American); Hispanic/Other/Asian subgroups were too small (n=123–404) to draw a real conclusion, and I'm reporting that honestly rather than claiming a finding off a small sample.

The more precise diagnosis: for elderly patients, the model stays well-calibrated on average (ECE ≈ 0.009–0.012) even though it can't discriminate between individuals well (low AUC) — it's not confidently wrong, it's just not separating cases. For young patients it's the opposite: good discrimination, noisier calibration in the sparse high-probability bins (small n there, so treat with caution).

**The gap isn't a weak-model artifact.** Swapping logistic regression for LightGBM (much higher capacity) barely moves the aggregate number (0.644 → 0.657) and reproduces the identical age gradient, so this is a property of the task, not the model.

**The real driver is diagnosis complexity, not age itself.** Age-band AUC correlates with average number of diagnoses at -0.80 — patients accumulate more diagnoses with age (2.7 in the youngest band vs. 7.9 in the oldest), and more concurrent diagnoses makes individual readmission risk genuinely noisier to predict. A `diag_3`-missing rate of 60% in the youngest band looked like a data-quality problem at first, but it isn't: young patients simply don't have a third diagnosis to record, so the missingness is a proxy for lower complexity, not lost information. This reframes the finding from "the model is worse for old patients" (an age story) to "the model is worse for multimorbid patients" (a complexity story, and a more clinically actionable one — this would motivate flagging high-diagnosis-count patients for a different triage path regardless of age).

**Headline: aggregate AUC of 0.64 hides a real gap that tracks diagnosis complexity, not model choice — patients with few diagnoses are predicted well (AUC ≈ 0.80) and patients with many concurrent diagnoses are predicted barely better than chance (AUC ≈ 0.60), and this persists even with a much stronger model.**

## 4. What would you do next, given more time?
- Test whether conditioning directly on diagnosis count (rather than age) as the stratification variable produces an even cleaner separation than age does — age may just be a proxy for the real driver
- Extend the race audit with a larger sample or an external dataset to get real conclusions for the underrepresented subgroups instead of wide, inconclusive CIs
- Try a model that explicitly handles multimorbidity structure (e.g., diagnosis-code embeddings or a model that treats the diagnosis list as a set rather than three flat columns) to see if the complexity-driven gap can be closed
