# Notebook 02 - Classical Models (Jason Justice)
# Exported from 02_classical_models.ipynb (code cells only, in order).

# ===== Code cell 1 ========================================
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ===== Code cell 2 ========================================
DATA_DIR = Path("../data")
df = pd.read_csv(DATA_DIR / "phiusiil.csv")
manifest = json.loads((DATA_DIR / "01_feature_columns.json").read_text())

clean_features = manifest["clean_numeric_features"]
print(f"rows: {len(df):,}")
print(f"clean features from notebook 01: {len(clean_features)}")

# ===== Code cell 3 ========================================
def shannon_entropy(s):
    if not s:
        return 0.0
    counts = np.unique(list(s), return_counts=True)[1]
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


SUFFIX_STOP = {
    "www",
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "co",
    "uk",
    "io",
    "de",
    "ru",
    "info",
    "biz",
    "app",
    "xyz",
    "online",
    "site",
    "web",
}
SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
}
BRANDS = [
    "paypal",
    "apple",
    "microsoft",
    "google",
    "amazon",
    "facebook",
    "instagram",
    "netflix",
    "linkedin",
    "whatsapp",
    "outlook",
    "office365",
    "chase",
    "wellsfargo",
    "bankofamerica",
    "citibank",
    "hsbc",
    "barclays",
    "americanexpress",
    "capitalone",
    "santander",
    "visa",
    "mastercard",
    "coinbase",
    "binance",
    "blockchain",
    "metamask",
    "dropbox",
    "adobe",
    "yahoo",
    "gmail",
    "icloud",
    "spotify",
    "steam",
    "roblox",
    "twitter",
    "tiktok",
    "snapchat",
    "ebay",
    "walmart",
    "target",
    "fedex",
    "ups",
    "dhl",
    "usps",
    "irs",
    "hmrc",
    "docusign",
    "onedrive",
    "zoom",
    "twitch",
    "discord",
    "telegram",
    "reddit",
    "github",
    "wordpress",
    "shopify",
    "stripe",
    "venmo",
    "zelle",
    "cashapp",
    "revolut",
    "aliexpress",
    "alibaba",
    "samsung",
    "xfinity",
    "verizon",
    "tmobile",
    "vodafone",
    "booking",
    "airbnb",
    "uber",
    "lyft",
    "doordash",
    "instacart",
]


def domain_labels(domain):
    return [
        p
        for p in domain.lower().split(".")
        if p not in SUFFIX_STOP and len(p) >= 3
    ]


def registered_domain(domain):
    d = domain.lower()
    return d[4:] if d.startswith("www.") else d

# ===== Code cell 4 ========================================
dom = df["Domain"].astype(str)

df["url_entropy"] = df["URL"].astype(str).map(shannon_entropy)
df["domain_entropy"] = dom.map(shannon_entropy)
df["is_punycode"] = dom.str.contains("xn--").astype(int)
df["is_shortener"] = dom.map(registered_domain).isin(SHORTENERS).astype(int)

# brand_distance: min normalized Levenshtein from any significant domain
# label to a brand. Compute once per unique label (fast in C), then map.
labels_per_row = dom.map(domain_labels)
unique_labels = sorted({lab for labs in labels_per_row for lab in labs})
dist = process.cdist(
    unique_labels, BRANDS, scorer=Levenshtein.normalized_distance
)
label_min = dict(zip(unique_labels, dist.min(axis=1)))


def brand_distance(labs):
    return min((label_min[x] for x in labs), default=1.0)


df["brand_distance"] = labels_per_row.map(brand_distance).astype(float)

engineered = [
    "url_entropy",
    "domain_entropy",
    "brand_distance",
    "is_punycode",
    "is_shortener",
]
print(df[engineered].describe().round(3).to_string())

# ===== Code cell 5 ========================================
phish = df["label"] == 0
by_class = pd.DataFrame(
    {
        "phishing_median": df.loc[phish, engineered].median(),
        "legit_median": df.loc[~phish, engineered].median(),
    }
)
print(by_class.round(3).to_string())

cont = ["url_entropy", "domain_entropy", "brand_distance"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, col in zip(axes, cont):
    ax.hist(
        df.loc[~phish, col],
        bins=40,
        alpha=0.7,
        density=True,
        color="tab:blue",
        label="legitimate",
    )
    ax.hist(
        df.loc[phish, col],
        bins=40,
        alpha=0.7,
        density=True,
        color="tab:orange",
        label="phishing",
    )
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.show()

# ===== Code cell 6 ========================================
features = clean_features + engineered

before = len(df)
model_df = df.drop_duplicates(subset="URL").reset_index(drop=True)
print(f"dropped {before - len(model_df):,} duplicate-URL rows")

X = model_df[features]
y = 1 - model_df["label"]  # 1 = phishing (positive class)
groups = model_df["Domain"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print(f"train: {len(X_train):,}   test: {len(X_test):,}")
print(f"phishing rate  train={y_train.mean():.3f}  test={y_test.mean():.3f}")

# ===== Code cell 7 ========================================
def xgb(**kw):
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    params.update(kw)
    return XGBClassifier(**params)


def evaluate(name, model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
    }


n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())

models = {
    "LogReg": Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": xgb(scale_pos_weight=n_neg / n_pos),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    rows.append(evaluate(name, model, X_test, y_test))

results = pd.DataFrame(rows).set_index("model")
print(results.round(4).to_string())

# ===== Code cell 8 ========================================
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (name, model) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["legit", "phish"],
        yticklabels=["legit", "phish"],
    )
    ax.set_title(name)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
plt.tight_layout()
plt.show()

# ===== Code cell 9 ========================================
ablation = {
    "none": xgb(),
    "scale_pos_weight": xgb(scale_pos_weight=n_neg / n_pos),
    "SMOTE": ImbPipeline(
        [
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", xgb()),
        ]
    ),
}

rows = []
for name, model in ablation.items():
    model.fit(X_train, y_train)
    rows.append(evaluate(name, model, X_test, y_test))

print(pd.DataFrame(rows).set_index("model").round(4).to_string())

# ===== Code cell 10 ========================================
cv = StratifiedGroupKFold(n_splits=5)
param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.1, 0.3],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}
search = RandomizedSearchCV(
    xgb(scale_pos_weight=n_neg / n_pos),
    param_distributions=param_dist,
    n_iter=10,
    scoring="average_precision",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
search.fit(X_train, y_train, groups=groups_train)
best = search.best_estimator_

print("best params:", search.best_params_)
print(f"CV PR-AUC: {search.best_score_:.4f}")
tuned = evaluate("XGBoost (tuned)", best, X_test, y_test)
print(pd.DataFrame([tuned]).set_index("model").round(4).to_string())

# ===== Code cell 11 ========================================
importances = pd.Series(best.feature_importances_, index=features).sort_values(
    ascending=False
)

plt.figure(figsize=(8, 5))
importances.head(15)[::-1].plot.barh(color="tab:blue")
plt.xlabel("XGBoost gain importance")
plt.title("Top 15 features (tuned XGBoost)")
plt.tight_layout()
plt.show()

print(f"engineered-feature ranks (of {len(features)} features):")
for f in engineered:
    rank = int(importances.index.get_loc(f)) + 1
    print(f"  {f:16s} rank {rank:2d}   gain {importances[f]:.4f}")

# ===== Code cell 12 ========================================
sample = X_test.sample(min(20000, len(X_test)), random_state=RANDOM_STATE)
perm = permutation_importance(
    best,
    sample,
    y_test.loc[sample.index],
    scoring="average_precision",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
perm_imp = pd.Series(perm.importances_mean, index=features)
print("top 10 by permutation importance (PR-AUC drop):")
print(perm_imp.sort_values(ascending=False).head(10).round(4).to_string())

# ===== Code cell 13 ========================================
K = 8
top_k = importances.head(K).index.tolist()
print(f"top {K} features: {top_k}")

simple = xgb(scale_pos_weight=n_neg / n_pos)
simple.fit(X_train[top_k], y_train)
simple_res = evaluate(f"XGBoost (top {K})", simple, X_test[top_k], y_test)

compare = pd.DataFrame([tuned, simple_res]).set_index("model")
print(compare.round(4).to_string())

# ===== Code cell 14 ========================================
# Illustrative per-event costs (see markdown above).
COST_FN = 500.0
COST_FP = 5.0

proba = best.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)
n = len(y_test)


def cost_per_1k(t):
    pred = (proba >= t).astype(int)
    fn = int(((pred == 0) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    return (COST_FN * fn + COST_FP * fp) / n * 1000


cost_curve = np.array([cost_per_1k(t) for t in thresholds])
best_t = float(thresholds[cost_curve.argmin()])

plt.figure(figsize=(8, 4))
plt.plot(thresholds, cost_curve, color="tab:blue")
plt.axvline(best_t, color="tab:red", ls="--", label=f"min-cost t={best_t:.2f}")
plt.axvline(0.5, color="gray", ls=":", label="default t=0.50")
plt.xlabel("threshold (flag phishing if proba >= t)")
plt.ylabel("expected cost per 1,000 URLs ($)")
plt.title("Cost-sensitive threshold selection")
plt.legend()
plt.tight_layout()
plt.show()

print(
    f"min-cost threshold: {best_t:.2f}  "
    f"(${cost_curve.min():.2f} per 1k) vs "
    f"${cost_per_1k(0.5):.2f} at default 0.5"
)

# ===== Code cell 15 ========================================
prec, rec, thr = precision_recall_curve(y_test, proba)


def operating_point(target, mode):
    metric = prec[:-1] if mode == "precision" else rec[:-1]
    other = rec[:-1] if mode == "precision" else prec[:-1]
    ok = metric >= target
    if not ok.any():
        return None
    idx = np.where(ok)[0]
    pick = idx[np.argmax(other[idx])]
    return thr[pick], prec[pick], rec[pick]


for name, mode in [
    ("precision-first (prec>=0.99)", "precision"),
    ("recall-first (recall>=0.99)", "recall"),
]:
    r = operating_point(0.99, mode)
    if r:
        t, p, rc = r
        print(f"{name:30s} t={t:.3f}  precision={p:.4f}  recall={rc:.4f}")

# 3-tier policy
t_block, t_warn = 0.90, 0.50
tier = np.where(
    proba >= t_block,
    "block",
    np.where(proba >= t_warn, "warn", "allow"),
)
print()
print("3-tier policy on the test set:")
print(pd.Series(tier).value_counts().to_string())

# ===== Code cell 16 ========================================
all_results = pd.concat(
    [results, pd.DataFrame([tuned, simple_res]).set_index("model")]
)
out = {
    "notebook": "02",
    "positive_class": "phishing",
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "engineered_features": engineered,
    "metrics": all_results.round(4).reset_index().to_dict(orient="records"),
    "min_cost_threshold": best_t,
}
(DATA_DIR / "02_classical_results.json").write_text(json.dumps(out, indent=2))
print("saved data/02_classical_results.json")

