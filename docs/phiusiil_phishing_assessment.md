# PhiUSIIL Phishing URL Dataset — Final Project Fit Assessment

**Dataset:** PhiUSIIL Phishing URL Dataset (UCI id 967)
**URL:** https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
**Size:** 235,795 instances · 54 features · no missing values · CC BY 4.0 · donated 2024
**Class balance:** 134,850 legitimate vs. 100,945 phishing (~57/43)
**Target:** `label` — 1 = legitimate, 0 = phishing (binary classification)

Verdict up front: **it works for the final project** — a strong, timely M4 + M6 option — but it must be designed around a leakage gotcha (see below).

---

## Requirement coverage

| Requirement | Fit | Notes |
|---|---|---|
| Introduction | ✅ | Phishing detection — timely cybersecurity story |
| Data Cleaning/Prep | ⚠️ moderate | No missing values, so light on "messy data." Real prep: drop `FILENAME`, decide how to handle raw `URL`/`Domain` strings, encode high-cardinality `TLD`, scale features, train/test split |
| Exploratory Data Analysis | ✅ | Good statistical EDA: URL-length / HTTPS / TLD-legitimacy distributions, phishing-vs-legit comparisons, correlations |
| Model Selection | ✅ | Logistic regression + XGBoost/RandomForest + a neural net — easily 2+ algorithms |
| Model Analysis | ✅ | ROC/AUC, precision/recall, confusion matrix, feature importance |
| Conclusion & Recommendations | ✅ | Deploy-as-a-filter recommendations |
| Appendix (Jupyter notebook) | ✅ | |
| ≥ 2 algorithms | ✅ | |
| Non-medical | ✅ | Cybersecurity |

---

## Module fit (AAI-501)

- **M3** (evaluation metrics) — ✅
- **M4** (supervised classification + XGBoost + ML lifecycle) — ✅ **strong**
- **M6** (deep learning) — ✅ either a tabular DNN, or a **character-level CNN/RNN on the raw URL string** (a real, impressive technique that fits the CNN/RNN module)
- **M5** (time series) — ❌ no time dimension, so it does **not** touch the forecasting module (less breadth than Bike Sharing; similar to Telco Churn)
- **M7** (reinforcement learning) — ❌

---

## ⚠️ Key caveat: data leakage / "too easy"

This dataset is **trivially separable** if every feature is used:

- **`URLSimilarityIndex`** is near-leaky — it essentially encodes how close a URL is to known-legitimate URLs; models reach ~99–100% accuracy on it alone.
- The **raw `URL` / `Domain` strings** can also leak (the model memorizes specific domains).

If the team dumps all 54 features into a model, the result is ~100% accuracy and a **shallow Model Analysis**, which grades poorly.

**The fix turns this into a strength:** deliberately drop `URLSimilarityIndex` and the raw string columns, model the harder/realistic problem, and discuss the leakage and generalization in the report. That is exactly the "validity of your model" analysis the rubric rewards.

---

## Trade-offs vs. the other two shortlisted options

- **PhiUSIIL** — timely + deep-learning showcase, but **no M5**, lighter cleaning, requires careful leakage handling.
- **Bike Sharing** — widest module coverage (M4 → M5 → M6).
- **Telco Churn** — best business story + safest all-around fit.

**Bottom line:** solid pick if the team wants a security / deep-learning flavor and is willing to handle the leakage properly (drop `URLSimilarityIndex` + raw strings and study the realistic problem).
