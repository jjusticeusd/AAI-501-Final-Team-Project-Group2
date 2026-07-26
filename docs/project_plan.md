# Project Plan — Phishing URL Detection (PhiUSIIL)

Status: draft for team review. Responds directly to the professor's feedback
on our proposal (unique twist beyond the raw dataset, SOTA comparison,
business-relevant metric, imbalance handling). Nothing here is implemented
yet — this is the plan to discuss and adjust before we start coding in
`code/01-03.ipynb`.

## 0. Recap: why this needs a plan, not just "train a classifier"

`docs/phiusiil_phishing_assessment.md` already flagged the core problem:
PhiUSIIL is trivially separable if we use every feature as-is.
`URLSimilarityIndex` is a UCI-provided similarity score against known
legitimate URLs — it essentially already contains the answer, so any model
that uses it lands near 100% accuracy with nothing left to say. The raw
`URL`/`Domain` strings have the same issue (models memorize specific
domains rather than learning generalizable patterns). The professor's
feedback about a "unique twist" and reproducing state-of-the-art results
both point at the same fix: don't hand the model the answer, and be honest
about what's actually learnable from realistic signal.

Also worth noting up front, re: the imbalance comment — the professor
assumed "highly imbalanced," but PhiUSIIL is actually ~57% legitimate /
43% phishing (134,850 / 100,945), which is mild imbalance, not severe. We
should say this explicitly in the report rather than silently applying
heavy-handed rebalancing a mild-imbalance problem doesn't need — see §4.
Real phishing traffic, unlike this benchmark, is nowhere near balanced
(see §7) — worth flagging as a limitation of any accuracy claim here.

**Update after running notebook 01**: the leakage problem is bigger than
just `URLSimilarityIndex`. Dropping it and training on the other 49
"clean" features still reaches ~99.9% accuracy — *higher* than the leaky
column alone (~99.6%). So the dataset is close to trivially separable on
its content/structural features as a *set*, not on one column. §7 found
the same pattern independently: every reproduction of PhiUSIIL we could
find, across unrelated algorithm families, lands in a 99.5–100% band.
That's corroborating evidence, not just our own artifact. Practically:
we report the full "clean-features" model as the easy/optimistic regime
and the URL-string-only character-level model (§5c) as the harder,
more realistic regime, and discuss both rather than picking one number
to headline.

Notebook 01 also turned up a few more things worth recording here
(details and numbers are in the notebook):

- **`IsHTTPS` is a collection artifact.** 100% of legitimate rows use
  HTTPS — zero exceptions — vs. ~49% of phishing rows. "No HTTPS =
  phishing" is never wrong on this dataset, which says more about
  where the legitimate URLs were collected than about the real web.
  Several other content flags are nearly as lopsided (`HasSocialNet`
  0.5% vs. 79.5%, `HasCopyrightInfo` 5.7% vs. 80.8%). We keep these
  features for the tabular models but call them out in the report —
  dropping the six most lopsided flags only moves the quick-model
  accuracy from 99.9% to 99.7%, so the easiness isn't hiding in a few
  columns anyway.
- **425 duplicate URLs** (always with the same label). Notebook 02
  drops URL duplicates before splitting so the same URL can't land in
  both train and test.
- **Domain repetition is real but doesn't inflate scores**: 9% of rows
  share a `Domain` with another row, but a domain-grouped split scores
  the same as a naive random split (0.9995 vs. 0.9992). We use the
  grouped split anyway as the safer setup — see §1.
- **599 IP-address rows have junk "TLDs"** (the last octet of the IP,
  e.g. `123`), all phishing. Data-quality quirk worth a sentence in
  the report.
- **`TLDLegitimateProb` is safe to use.** If UCI had computed it from
  this dataset's own labels it would be target leakage; notebook 01
  checked, and its correlation with the per-TLD share of legitimate
  labels in our data is only ~0.18, so it came from an outside source.

## 1. Data prep & leakage handling (Notebook 01)

- `FILENAME` turns out not to exist in the `ucimlrepo` version of the
  dataset (only the Kaggle CSV has it), so there's nothing to drop
  there. Do drop the 425 duplicate URLs before splitting.
- Drop `URLSimilarityIndex` — the leaky feature. Documented in notebook
  01 with the before/after experiment: 99.6% accuracy on that single
  column alone vs. 99.9% on everything else, reported as the
  justification for excluding it.
- Hold `URL` and `Domain` out of the tabular feature set (they're
  effectively identifiers), but don't discard them — they're the raw
  material for the engineered features in §2 and the input to the
  character-level model in §5c.
- `TLD` is high-cardinality categorical (695 values, and 599 rows have
  junk numeric "TLDs" from IP-address URLs) — too many to one-hot
  encode, so we rely on `TLDLegitimateProb` instead. Notebook 01
  verified it isn't secretly computed from this dataset's labels
  (corr ~0.18 with the per-TLD label share), so that's safe.
- Split strategy: **domain-grouped split** (`GroupShuffleSplit` on
  `Domain`), after dropping duplicate URLs. Notebook 01 answered the
  "does `Domain` repeat?" question: yes, 5,526 domains repeat (9% of
  rows), but the grouped split scores essentially the same as a naive
  stratified one (0.9995 vs. 0.9992) — no inflation, we just keep the
  grouped split because it's the safer choice and costs nothing.
  Report both numbers in notebook 02. One caveat: exact `Domain`
  matching under-groups subdomains (the top repeats are all gateways
  to the same IPFS service: `ipfs.io`, `gateway.ipfs.io`,
  `cf-ipfs.com`, ...); grouping by registered domain via `tldextract`
  is the stricter option if we decide we want it.
- No missing values per the UCI description, so this section is short —
  consistent with the assessment doc's note that cleaning is "moderate"
  fit for this dataset.

## 2. New engineered features (the core "twist")

PhiUSIIL already ships many derived features (lexical ratios, TLD risk,
HTML/content features like `NoOfJS`, `HasPasswordField`, `NoOfImage`,
etc.). To have something genuinely new to say, our engineered features
should target signal the existing columns don't already encode:

- **Shannon entropy** of the domain and of the full URL — phishing domains
  generated by DGA-style or randomized registration tend to have higher
  character entropy than legitimate ones.
- **Brand-impersonation distance** — Levenshtein/edit-distance from the
  registered domain to a small curated list (~100–200) of well-known
  brand domains (banks, payment processors, major tech/social platforms).
  A small edit distance to a well-known brand, combined with the domain
  *not* actually being that brand, is a classic typosquatting signal.
  We'll bundle a static CSV of reference domains rather than depend on a
  live lookup, for reproducibility.
- **Homograph / punycode flag** — whether the domain starts with `xn--`
  (IDN homograph attacks are a known phishing technique not obviously
  captured elsewhere in the feature set).
- **URL shortener flag** — domain matches a small known list of shortener
  services (bit.ly, tinyurl, etc.).
- Fold these into the "important features" analysis the team proposal
  already promised, i.e., check whether these new features rank highly
  in feature importance, not just whether they exist.

## 3. Class imbalance handling (Notebook 01/02)

Given the *actual* (mild) imbalance:

- Primary approach: **class weighting** (`class_weight="balanced"` for
  Logistic Regression / Random Forest, `scale_pos_weight` for XGBoost) —
  cheap, doesn't synthesize data, and is the standard first move for
  mild imbalance.
- Run **SMOTE (or ADASYN)** as an explicit ablation, not a default —
  compare class-weighted vs. SMOTE-resampled results and discuss the
  known risk that SMOTE can overfit on tabular data with redundant/
  correlated engineered features (relevant here since many PhiUSIIL
  columns are correlated ratios of each other).
- Use **stratified k-fold CV** throughout so imbalance doesn't distort
  hyperparameter selection.
- Report PR-AUC alongside ROC-AUC — PR-AUC is the more informative curve
  under class imbalance and is standard practice in this literature (to
  be backed with citations once the literature review lands, §7).

## 4. Model roster (Notebooks 02–03)

| # | Model | Type | Purpose |
|---|---|---|---|
| 1 | Logistic Regression (class-weighted) | Classification, linear | Interpretable baseline |
| 2 | Random Forest | Classification, ensemble | Non-linear baseline + feature importance |
| 3 | XGBoost (tuned) | Classification, ensemble | Strongest tabular model, primary comparison point vs. literature |
| 4 | Character-level CNN/BiLSTM on raw URL string only | Classification, deep learning | Leakage-immune second modeling paradigm (M6 requirement); trained on *just* the URL string, no engineered features at all |
| 5 | KMeans / HDBSCAN on phishing-only subset | Clustering, unsupervised | Second algorithm *type* (not just second classifier family); discovers attack "families" — see §5b |

Items 1–3 all use the leakage-clean tabular feature set from §1–2.
Hyperparameter tuning via `RandomizedSearchCV`/`GridSearchCV` with
stratified CV, tuned toward the business metric chosen in §6, not
accuracy.

### 5a. "Simple model" check (already promised in the proposal)

Take the top-K features by importance from the best tabular model (e.g.,
top 5–8) and retrain a simple model using only those. If it comes close
to the full model, that's a real, actionable finding worth a paragraph in
the report (cheaper feature collection in a production filter).

### 5b. Clustering — attack-family discovery

Cluster the **phishing-labeled rows only** on structural/content features
(same leakage-clean set, minus label) with KMeans (elbow + silhouette to
pick k) and/or HDBSCAN for comparison. Profile the resulting clusters by
their centroids/feature distributions and give them descriptive names,
e.g., "IP-address URLs without HTTPS," "brand-impersonation typosquats,"
"content-mimicking pages with credential forms." Visualize with
PCA/t-SNE. This is kept as a standalone unsupervised deliverable (EDA +
its own findings) rather than feeding cluster IDs back into the
classifier, to avoid a second leakage-review cycle — we can revisit
feeding it back as a feature if time allows.

### 5c. Character-level deep learning on the raw URL

Tokenize the URL string at the character level, embed, pass through a
small CNN or BiLSTM, sigmoid output — trained with class weights. Compare
its metrics against the tabular models. This model literally cannot see
`URLSimilarityIndex` or any engineered feature, so it's a clean second
data point on "how hard is this problem really," and it's directly
comparable to the lexical/NLP-style approach in Sahingoz et al. (2019)
from our own proposal's reference list.

## 5d. Cost-sensitive decision layer

Rather than picking a metric by convention, quantify it:

- Define illustrative but sourced costs: cost of a false negative (missed
  phishing → credential theft / fraud exposure) vs. cost of a false
  positive (blocked legitimate site → user friction, support tickets,
  lost trust). We'll anchor these to public industry figures (e.g., IBM's
  Cost of a Data Breach report for the FN side) and state clearly that
  the FP-side numbers are illustrative estimates, not sourced.
- Sweep the decision threshold, compute expected cost per 1,000 URLs at
  each threshold, and pick the cost-minimizing operating point instead of
  the default 0.5.
- Present a **3-tier policy** using two thresholds: allow / warn
  (interstitial) / block — closer to how real browser and email phishing
  filters actually behave than a single binary cutoff.

## 6. Business-relevant metric

**Revised, now that §7's research is in — the working hypothesis below
was wrong to assert a single universal answer.** The literature actually
splits by deployment context:

- **Whittaker, Ryner, & Nazif (2010)** — the actual Google Safe Browsing
  classifier paper (NDSS 2010) — explicitly designs for **precision
  over recall**: they accept catching "only" >90% of phishing pages in
  exchange for a low false-positive rate, because wrongly blocking a
  legitimate site is highly visible, erodes user trust, and generates
  support/legal blowback. This is a browser-level, defense-in-depth
  layer, not the only line of defense.
- The intuitive counter-case — a mail/endpoint gateway that silently
  drops or quarantines suspected phishing before a human ever sees it —
  plausibly favors recall instead, since a missed phishing email that
  reaches an inbox risks direct credential theft/fraud. We didn't find
  a rigorous cost-quantification study specific to phishing to cite for
  this side; it's argued from the same FN/FP logic as the browser case,
  just with the asymmetry flipped.

So the "right" metric is a **function of deployment scenario**, not a
fixed rule — which is exactly why §5d's cost-sensitive threshold layer
matters more than picking one metric by convention. Plan:

- State a specific assumed deployment scenario in the report (we'll
  frame it as a **browser-integrated warning system**, matching
  Whittaker et al.'s real-world precedent and PhiUSIIL's own framing as
  a URL-level detector) and justify metric choice against *that* choice
  explicitly, citing Whittaker et al. (2010) for the precision-first
  argument.
- Report **PR-AUC** (Davis & Goadrich, 2006, ICML; Saito & Rehmsmeier,
  2015, *PLOS ONE* 10(3):e0118432 — both establish PR-AUC over ROC-AUC
  as the more informative curve under class imbalance) as the
  threshold-independent headline number regardless of scenario.
- Use §5d's cost sweep to show *both* directions: report metrics at a
  precision-first operating point (Google-style) and a recall-first
  operating point (gateway-style) side by side, rather than asserting
  one is correct. This turns "which metric matters" into an actual
  result instead of a stated assumption.
- Plain accuracy stays a secondary/sanity-check number only — not
  meaningless here (57/43 is mild, not severe, imbalance) but not the
  headline either.

## 7. Literature & SOTA comparison

*Numbers below are from a research pass across public sources; several
figures come from secondary citations because the primary paper
(ScienceDirect) wasn't directly fetchable, and are flagged accordingly.
Treat unverified figures as directional, not exact, until we can check
a primary copy (e.g., via the university library).*

**PhiUSIIL, the dataset itself** — Prasad, A., & Chandra, S. (2024).
PhiUSIIL: A diverse security profile empowered phishing URL detection
framework based on similarity index and incremental learning.
*Computers & Security*, 136, 103545.
https://doi.org/10.1016/j.cose.2023.103545. Secondary sources report
~99.2% accuracy (incremental-learning setup) to ~99.8% (batch setup),
with one source claiming up to 99.97% / AUC = 1.00 for the best
configuration — **sources disagree on the exact number and which
algorithm produced it; unverified against the primary text.**
Independent reproductions on this exact dataset (Kaggle notebooks and
smaller papers, aggregated/unverified individually) consistently land
in the same 99.5–100% band across very different algorithm families
(Random Forest, Decision Tree, KNN, Logistic Regression, XGBoost, a
plain FCNN, even a BERT variant) — that near-universal ceiling across
unrelated model types, regardless of which single column is dropped,
is exactly what our own EDA found in notebook 01 (dropping
`URLSimilarityIndex` alone still leaves ~99.9% accuracy on the
remaining features). Pentapalli, Salisbury, Riep, & Cohen (2025), *A
Gradient-Optimized TSK Fuzzy Framework for Explainable Phishing
Detection*, arXiv:2504.18636, independently excluded
`URLSimilarityIndex` for the same reason, calling it "a strong proxy
signal for legitimacy." Dalton et al. (2025), *PhreshPhish*,
arXiv:2507.10854, separately critiques phishing benchmark datasets in
general for leakage and unrealistic base rates producing "overly
optimistic performance results" — directly consistent with the case
we're making for treating PhiUSIIL's ceiling numbers cautiously and
reporting the URL-only model (§5c) as the harder, more honest
comparison point.

**Broader lexical/URL-only literature** (the comparison point for our
character-level model, §5c):
- Sahingoz, Buber, Demir, & Diri (2019). Machine learning based
  phishing detection from URLs. *Expert Systems with Applications*,
  117, 345–357. 73,575 URLs; Random Forest on 39 NLP-based lexical
  features reached **97.98% accuracy** — corroborated across sources,
  our most solid comparison number for a "no page content, URL text
  only" baseline.
- Mohammad, Thabtah, & McCluskey (2014). *Neural Computing and
  Applications*, 25, 443–458. Secondary sources disagree (92.2–92.5%
  vs. 96.1%, the latter possibly misattributed to a different
  classifier in that paper) — **could not verify against the original;
  cite with a hedge or drop this number if we can't confirm it.**
- Character-level / deep models on raw URLs: Yerima & Alzaylaee (2020),
  arXiv:2004.03960, char-CNN, 98.2% / F1 0.976. Maneriker et al.
  (2021), *URLTran*, arXiv:2106.05256, BERT on subword-tokenized URLs,
  ~0.96 accuracy. A MiniLM-CNN-LSTM hybrid (2024/25, *Technologies*,
  MDPI), 98.98%. Dubey et al. (2025), arXiv:2512.16717, char-CNN +
  LightGBM ensemble, 99.82%. These bracket a realistic **97–99%** range
  for our §5c model — if we land meaningfully below ~97% we should
  double-check preprocessing/tokenization before concluding the
  architecture itself is weak.
- Vrbančič, Fister Jr., & Podgorelec (2020). Datasets for phishing
  websites detection. *Data in Brief*, 33, 106438 — confirmed as a
  data-descriptor paper only (two datasets released, 58,645 and 88,647
  instances); it reports no baseline classifier results of its own, so
  it's a references-list citation, not a benchmark comparison point.

**Metric conventions**: see the rewritten §6 — Whittaker, Ryner, &
Nazif (2010, NDSS) is the key citation for precision-first design in a
real deployed phishing classifier (Google Safe Browsing). Davis &
Goadrich (2006, ICML, pp. 233–240) and Saito & Rehmsmeier (2015, *PLOS
ONE*) are the general-ML citations for PR-AUC over ROC-AUC under
imbalance. Das, Baki, El Aassal, Verma, & Dunbar, *SoK: A Comprehensive
Reappraisal of Content-Based Phishing Detection Research*,
arXiv:1911.00953, flags the "base-rate fallacy" — worth citing when we
discuss why benchmark-dataset balance (57/43 here) doesn't match
production traffic skew (one source in this research pass cited real
deployments as skewed up to ~1:50,000 legitimate:phishing; URLTran's
authors downsampled to ~1:20 for training). This is useful ammunition
for the report's discussion of §3's imbalance framing — the *dataset*
is mildly imbalanced, but we should explicitly note that real traffic
is not, and that's a limitation of any benchmark-dataset accuracy claim
(ours included).

**Imbalance handling in this literature**: Omari, Taoussi, & Oukhatar
(2025), *IJACSA*, 16(2), compared RUS/ROS/SMOTE variants with XGBoost
on phishing URLs; best (SMOTE-NC + XGBoost) reached 98.0%
precision / 98.5% recall — a data point for our own SMOTE-vs-class-
weighting ablation in §3.

**Clustering precedent** (justifies §5b as more than a novelty add-on):
Feng, Qiao, Ye, & Zhang (2022), *PeerJ Computer Science*, 8, e868 —
k-medoid clustering on DOM/CSS structural features found 1,598 phishing
"kits" clustering by brand target (Yahoo/Google/Dropbox/PayPal), 90.1%
TPR. Althobaiti et al. (2023), IEEE Access — Mean Shift/DBSCAN on
email+URL features to identify phishing campaigns. Zia & Kalidass
(2025), arXiv:2502.13171 — LSH clustering on lexical URL tokens, 93%
detection on PhishStorm. These directly support framing §5b's clusters
as "attack families," which is standard practice in this literature,
not something we're inventing.

## 8. Notebook structure & division of labor (proposed — adjust freely)

- **`01.ipynb`** — data loading, leakage audit/fix, EDA, feature
  engineering (§1–2).
- **`02.ipynb`** — classical models: LR / RF / XGBoost, tuning, imbalance
  ablation (§3–4, §5a), cost-sensitive threshold analysis (§5d), business
  metric writeup (§6).
- **`03.ipynb`** — clustering / attack-family analysis (§5b) and the
  character-level deep learning model (§5c), plus the SOTA comparison
  table (§7).

Three teammates map naturally onto the three notebooks; happy to adjust
based on who wants to own the deep-learning vs. clustering piece.

## 9. Tooling gap

`pyproject.toml` currently has dependencies from an unrelated template
(langchain, google-genai, llama-index, osmnx, pulp, wikipedia, networkx)
and is missing what this project actually needs: `pandas`, `numpy`,
`scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE), a plotting library
already present (`matplotlib`, could add `seaborn`), `shap` (feature
importance/interpretability), and a DL framework (`torch` or
`tensorflow`) for §5c. Will clean this up when we start implementation.

## 10. Open questions for the team

1. Confirm the twist scope: engineered features (foundation, doing
   regardless) + clustering + char-level DL + cost-sensitive layer — all
   four, per your call — is that still the right amount of scope given
   the Module 4–6 timeline, or should we treat any of these as "if time
   allows"?
2. Division of labor above is a guess — who wants which notebook?
3. ~~Domain-grouped train/test split needs checking~~ — done in
   notebook 01: domains do repeat (9% of rows) but don't inflate the
   score; grouped split adopted anyway (§1).
