# AAI-501 Final Team Project — Group 2

## Project Overview

**Phishing URL detection** on the [PhiUSIIL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
(UCI id 967) — 235,795 URLs, 54 features, ~57% legitimate / 43% phishing.

The dataset is trivially separable as-is (`URLSimilarityIndex` alone hits
~99.6% accuracy), so the project is deliberately built around that fact:
drop the leaky signal, model the realistic problem, and add work that goes
beyond what the dataset hands us. This directly answers the professor's
feedback — a unique twist, a state-of-the-art comparison, a
business-relevant metric, and appropriate (mild, 57/43) imbalance
handling. See **[docs/project_plan.md](docs/project_plan.md)** for the full
plan and **[docs/phiusiil_phishing_assessment.md](docs/phiusiil_phishing_assessment.md)**
for the dataset assessment.

**Approach**
- **Leakage-clean tabular models** — Logistic Regression, Random Forest,
  XGBoost on a vetted feature set, with class-weighting vs. SMOTE ablation.
- **Engineered features** (the "twist") — domain/URL Shannon entropy,
  brand-impersonation edit distance, punycode and URL-shortener flags.
- **Character-level deep model** on the raw URL string — leakage-immune,
  realistic-difficulty comparison point.
- **Clustering** of phishing rows into attack "families" (unsupervised).
- **Cost-sensitive threshold layer** tuned to a business-relevant metric
  rather than the default 0.5 cutoff.

**Notebooks** (`code/`)
- `01.ipynb` — data loading, leakage audit, EDA ✅
- `02.ipynb` — classical models, tuning, imbalance ablation, cost layer ✅
- `03.ipynb` — clustering + character-level DL + SOTA comparison

## Getting Started

New to the repo? Set up your environment in one step (macOS):

```bash
./init.sh                          # installs uv, syncs dependencies, registers the Jupyter kernel
uv run python code/fetch_data.py   # caches the dataset to data/phiusiil.csv (run once)
./start_jupyter.sh                 # launches Jupyter Lab
```

See **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** for full setup instructions,
everyday `uv` commands, and the shared-notebook workflow.

## Project Checklist

### Module 1 — Team Setup
- [x] Complete the final team project survey (by end of Week 1)
- [x] Exchange contact info with all team members (Canvas, USD Email, or Slack)
- [x] Create GitHub repository and add all team members as collaborators
- [x] Add a `README.md` to the repository
- [x] Agree on a communication cadence for the rest of the project

### Module 3 — Proposal *(Due: Assignment 3.3, 11:59 PM)*
- [x] Identify an AI-driven problem and dataset (≥1000 examples, not used in coursework)
- [x] Select at least two AI/ML algorithm types to investigate (e.g., Classification + Clustering)
- [x] Write 1–2 page proposal including:
  - [x] Clear problem statement
  - [x] Brief discussion of problem and intended algorithms/system
  - [x] Identification of specific related course topics
  - [x] Expected system behaviors and problem types the algorithms handle
  - [x] Issues/challenges you expect to focus on
  - [x] Reference list (APA 7) of books/papers/articles to inform the project
- [x] One team member submits the proposal to Canvas

### Module 4 — Status Update *(Due: Assignment 4.3, 11:59 PM)*
- [ ] Submit the Final Team Project Status Update Form

### Modules 4–6 — Implementation
- [x] Set up project structure and development environment
- [x] Acquire and perform initial data preprocessing (leakage audit + EDA, notebook 01)
- [x] Implement first AI/ML algorithm (classification: LogReg / RandomForest / XGBoost, notebook 02)
- [ ] Implement second AI/ML algorithm (different type — clustering, notebook 03)
- [ ] Run experiments and comparisons between algorithms
- [x] Apply parameter tuning and/or feature selection as appropriate (RandomizedSearchCV + top-8, notebook 02)
- [x] Generate summary statistics and visualizations of findings (notebook 02)
- [ ] Follow PEP 8 style guide throughout all Python code
- [ ] Commit code regularly to GitHub (all members contributing)

### Module 7 — Final Deliverables *(Due: Assignment 7.2, 11:59 PM)*

#### Presentation
- [ ] Each member prepares their equal portion of the presentation
- [ ] Record the final presentation (20–30 minutes total, good audio quality)
- [ ] Upload recording to YouTube or Vimeo
- [ ] Add video link to the title page of the slide deck
- [ ] Finalize and export presentation slides

#### Paper (~10 pages, APA 7 format)
- [ ] Write project purpose, goals, and scope (with references)
- [ ] Clearly specify each AI algorithm used
- [ ] Include analysis, evaluation, and critique of algorithms and implementation
- [ ] Present empirical comparison results graphically
- [ ] Write appendix listing each member's detailed contributions

#### Code
- [ ] Clean, document, and organize all source code
- [ ] Add GitHub repository link to the paper

#### Submission
- [ ] One team member submits to Canvas:
  - [ ] Final Project Presentation Slides (with video link)
  - [ ] Final Project Paper (with GitHub link)
  - [ ] Complete source code (or GitHub link)
- [ ] Each member submits the Peer Evaluation form individually (separate Canvas link)

---
