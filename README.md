# Phishing URL Detection — AAI-501 Final Team Project (Group 2)

**Group 2:** Pranav Dhinakar, Jason Justice, Guna Pasupathy

Detecting phishing URLs on the **PhiUSIIL** dataset, built deliberately
around the dataset's biggest weakness — it is *too easy* — to produce an
honest, realistic study rather than a meaningless 100%-accuracy headline.

## The problem

Phishing URLs impersonate legitimate sites to steal credentials. We frame
this as a binary classification problem on the
[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
(UCI id 967): **235,795 URLs, 54 features, ~57% legitimate / 43% phishing**
— mild imbalance, not the severe skew often assumed.

The catch that shapes the whole project: **PhiUSIIL is trivially separable
as delivered.** The UCI-provided `URLSimilarityIndex` is a similarity score
against known-legitimate URLs — it essentially already contains the answer,
and a model using it alone reaches ~99.6% accuracy. Worse, the *remaining*
content/structural features as a set still reach ~99.9%. Dumping all 54
features into a model yields ~100% accuracy and nothing to analyze.

So we treat that as the design constraint, not a result: **drop the leaky
signal, model the harder realistic problem, and be explicit about what is
actually learnable.** This directly answers our proposal feedback — a unique
twist beyond the raw dataset, a state-of-the-art comparison, a
business-relevant metric, and appropriate (mild) imbalance handling.

## Approach — the "twist"

- **Leakage-clean tabular models** — Logistic Regression, Random Forest,
  and XGBoost on a vetted, non-leaky feature set.
- **Engineered features** — domain/URL Shannon entropy,
  brand-impersonation edit distance, punycode and URL-shortener flags.
- **Character-level CNN on the raw URL string** — leakage-immune by
  construction (it never sees `URLSimilarityIndex` or any engineered
  feature); the harder, more honest comparison point.
- **Clustering** of phishing URLs into unsupervised attack "families."
- **Cost-sensitive threshold layer** — tuned to an asymmetric
  false-negative / false-positive cost instead of a default 0.5 cutoff.

## Notebooks

Run in order; each hands artifacts to the next via `data/*.json`.

### `code/01_load_leakage_eda.ipynb` — data prep, leakage audit, EDA
Loads the dataset, checks class balance and duplicates, then runs a
three-stage **leakage audit**: (1) `URLSimilarityIndex`, (2) "too-perfect"
binary flags such as `IsHTTPS` (100% of legitimate rows use HTTPS — a
collection artifact), and (3) repeated domains and the train/test split
strategy (`GroupShuffleSplit` on `Domain`). Closes with feature-vs-label
EDA and saves the agreed **safe-feature manifest**
(`01_feature_columns.json`) consumed by the other two notebooks.

### `code/02_classical_models.ipynb` — classical models & cost analysis
Builds the engineered "twist" features, then trains and compares **Logistic
Regression, Random Forest, and XGBoost**. Includes a class-imbalance
ablation (class weighting vs. SMOTE), hyperparameter tuning
(`RandomizedSearchCV`), feature-importance analysis, a top-8-feature reduced
model, and a **cost-sensitive decision layer** that sweeps the threshold to
minimize expected cost per 1,000 URLs — plus a discussion of why the "right"
metric depends on the deployment scenario (browser warning vs. mail
gateway).

### `code/03_charcnn_clustering_sota.ipynb` — deep learning, clustering, SOTA
Trains a **character-level CNN on the raw URL string only** (parallel 3/5/7
convolutions over character embeddings) — the leakage-immune, realistic
regime. Then runs **unsupervised clustering** (KMeans with elbow/silhouette
selection, cross-checked with HDBSCAN) to surface phishing attack families,
and closes with a **state-of-the-art comparison** placing our results
against the published literature (full comparison table and citations live
in this notebook).

## Key findings (kept honest)

- On the leakage-clean tabular features, all three classifiers **saturate at
  ~99.99% accuracy** — the dataset is genuinely easy, so the model *ranking*
  is not the interesting result.
- The **URL-only character CNN (~99.85%)** is the more meaningful number: it
  cannot see any leaky or engineered feature, yet nearly matches the tabular
  models — evidence that the separability lives in the raw URL strings
  themselves.
- The **engineered features add little** on the easy tabular set (they rank
  low in importance) — reported as an honest negative result.
- Clustering reveals **soft attack-family structure**, not clean-cut groups.
- Our numbers sit squarely in the literature's **99.5–100% band** for this
  dataset, consistent with published critiques of phishing-benchmark
  leakage.

## Getting started (macOS)

We use [uv](https://docs.astral.sh/uv/) for reproducible environments.

```bash
./init.sh                          # installs uv, syncs deps, registers the Jupyter kernel
uv run python code/fetch_data.py   # caches the dataset to data/phiusiil.csv (run once)
./start_jupyter.sh                 # launches Jupyter Lab
```

Inside a notebook, select the **"AAI-501 (Group 2)"** kernel. Full setup,
everyday `uv` commands, and the linting/notebook workflow are in
**[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**.

## Repository layout

```
code/    three notebooks (run in order) + fetch_data.py
docs/    DEVELOPMENT.md (setup) and proposal.pdf
data/    dataset + generated artifacts (git-ignored; recreated by fetch_data.py + notebooks)
```

## Reference

Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile
empowered phishing URL detection framework based on similarity index and
incremental learning. *Computers & Security, 136*, 103545.
https://doi.org/10.1016/j.cose.2023.103545

The full state-of-the-art comparison and its citations are in
`code/03_charcnn_clustering_sota.ipynb`.
