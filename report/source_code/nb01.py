# Notebook 01 - Load, Leakage, EDA (Guna Pasupathy)
# Exported from 01_load_leakage_eda.ipynb (code cells only, in order).

# ===== Code cell 1 ========================================
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("../data/phiusiil.csv")

# ===== Code cell 2 ========================================
df = pd.read_csv(DATA_PATH)
y = df["label"]

print(f"shape: {df.shape}")
print(f"missing values: {df.isna().sum().sum()}")
print()
print("column types:")
print(df.dtypes.value_counts())
df.head(3)

# ===== Code cell 3 ========================================
counts = y.value_counts().sort_index()

plt.figure(figsize=(5, 4))
plt.bar(
    ["phishing (0)", "legitimate (1)"],
    counts.values,
    color=["tab:orange", "tab:blue"],
)
for i, c in enumerate(counts.values):
    plt.text(i, c + 2000, f"{c:,} ({c / len(df):.1%})", ha="center")
plt.ylabel("rows")
plt.title("Class balance")
plt.ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.show()

# ===== Code cell 4 ========================================
dup_url_extra = int(df["URL"].duplicated().sum())
dups = df[df["URL"].duplicated(keep=False)]
conflicts = dups.groupby("URL")["label"].nunique()

print(f"exact duplicate rows: {df.duplicated().sum()}")
print(f"repeated URLs (extra rows): {dup_url_extra}")
print(f"repeated URLs with conflicting labels: {(conflicts > 1).sum()}")

# ===== Code cell 5 ========================================
print(df.groupby("label")["URLSimilarityIndex"].describe())

n_100 = ((df["URLSimilarityIndex"] == 100) & (y == 0)).sum()
print(f"\nphishing rows that also score exactly 100: {n_100}")

# ===== Code cell 6 ========================================
bins = np.linspace(0, 100, 41)

plt.figure(figsize=(8, 4))
plt.hist(
    df.loc[y == 1, "URLSimilarityIndex"],
    bins=bins,
    alpha=0.7,
    color="tab:blue",
    label="legitimate (1)",
)
plt.hist(
    df.loc[y == 0, "URLSimilarityIndex"],
    bins=bins,
    alpha=0.7,
    color="tab:orange",
    label="phishing (0)",
)
plt.yscale("log")
plt.xlabel("URLSimilarityIndex")
plt.ylabel("count (log scale)")
plt.title("URLSimilarityIndex by class")
plt.legend()
plt.tight_layout()
plt.show()

# ===== Code cell 7 ========================================
def quick_logreg(X, y, groups=None):
    """Train a basic logistic regression, return test acc and F1."""
    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # keep all rows of the same group on one side of the split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), f1_score(y_test, preds)


numeric_cols = df.select_dtypes("number").columns
safe_cols = [
    c for c in numeric_cols if c not in ("URLSimilarityIndex", "label")
]

acc1, f11 = quick_logreg(df[["URLSimilarityIndex"]], y)
acc2, f12 = quick_logreg(df[safe_cols], y)

print(f"URLSimilarityIndex alone: acc={acc1:.4f}  f1={f11:.4f}")
print(f"all other numeric cols:   acc={acc2:.4f}  f1={f12:.4f}")

# ===== Code cell 8 ========================================
binary_cols = [
    c for c in numeric_cols if c != "label" and set(df[c].unique()) <= {0, 1}
]

rates = df.groupby("label")[binary_cols].mean().T
rates.columns = ["phishing_rate", "legit_rate"]
rates["gap"] = (rates["legit_rate"] - rates["phishing_rate"]).abs()
rates = rates.sort_values("gap", ascending=False)

print(f"{len(binary_cols)} binary feature columns")
rates.head(10).round(3)

# ===== Code cell 9 ========================================
giveaways = rates.index[:6].tolist()
print("dropping:", giveaways)

acc3, f13 = quick_logreg(df[[c for c in safe_cols if c not in giveaways]], y)
print(f"without giveaway flags: acc={acc3:.4f}  f1={f13:.4f}")

# ===== Code cell 10 ========================================
domain_counts = df["Domain"].value_counts()
repeated = domain_counts[domain_counts > 1]

print(f"unique domains: {len(domain_counts):,} / {len(df):,} rows")
print(
    f"domains appearing more than once: {len(repeated):,} "
    f"({repeated.sum():,} rows, {repeated.sum() / len(df):.1%})"
)
domain_counts.head(10)

# ===== Code cell 11 ========================================
acc4, f14 = quick_logreg(df[safe_cols], y, groups=df["Domain"])

print(f"normal random split:  acc={acc2:.4f}  f1={f12:.4f}")
print(f"domain-grouped split: acc={acc4:.4f}  f1={f14:.4f}")

# ===== Code cell 12 ========================================
print(f"unique TLDs: {df['TLD'].nunique()}")
print(df["TLD"].value_counts().head(10))

weird = df["TLD"].astype(str).str.fullmatch(r"\d+")
is_ip = df.loc[weird, "IsDomainIP"] == 1
print(f"\nrows where the 'TLD' is just a number: {weird.sum()}")
print(f"of those, IsDomainIP == 1: {is_ip.sum()}")
print(f"labels of those rows: {df.loc[weird, 'label'].unique()}")
df.loc[weird, ["Domain", "TLD", "label"]].head(3)

# ===== Code cell 13 ========================================
legit_share_per_tld = df.groupby("TLD")["label"].mean()
legit_share = df["TLD"].map(legit_share_per_tld)

corr = legit_share.corr(df["TLDLegitimateProb"])
print(f"corr with per-TLD legit share in this data: {corr:.3f}")

# ===== Code cell 14 ========================================
corrs = df[safe_cols].corrwith(y)
top15 = corrs.reindex(corrs.abs().sort_values().index).tail(15)
colors = ["tab:blue" if v > 0 else "tab:red" for v in top15]

plt.figure(figsize=(8, 6))
plt.barh(top15.index, top15.values, color=colors)
plt.axvline(0, color="gray", linewidth=1)
plt.xlabel("correlation with label (positive = legitimate)")
plt.title("Top 15 safe features by correlation with label")
plt.tight_layout()
plt.show()

# ===== Code cell 15 ========================================
non_binary = [c for c in safe_cols if c not in binary_cols]

medians = df.groupby("label")[non_binary].median().T
medians.columns = ["phishing_median", "legit_median"]
print(f"{len(non_binary)} non-binary numeric features")
medians.round(3)

# ===== Code cell 16 ========================================
feats = [
    "URLLength",
    "DomainLength",
    "DegitRatioInURL",
    "SpacialCharRatioInURL",
    "LineOfCode",
    "NoOfJS",
    "NoOfImage",
    "NoOfExternalRef",
]

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.flat, feats):
    # clip extreme outliers so the plots stay readable
    clipped = df[col].clip(upper=df[col].quantile(0.99))
    ax.hist(
        clipped[y == 1],
        bins=30,
        alpha=0.6,
        color="tab:blue",
        density=True,
        label="legitimate",
    )
    ax.hist(
        clipped[y == 0],
        bins=30,
        alpha=0.6,
        color="tab:orange",
        density=True,
        label="phishing",
    )
    ax.set_title(col, fontsize=10)
    ax.set_yticks([])
axes[0, 0].legend()
fig.suptitle("Feature distributions by class (clipped at 99th pct)")
plt.tight_layout()
plt.show()

# ===== Code cell 17 ========================================
corr_matrix = df[safe_cols].corr()
upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
pairs = upper.stack()
high = pairs[pairs.abs() > 0.8].sort_values(ascending=False)

print("feature pairs with |corr| > 0.8:")
print(high.round(3))

# ===== Code cell 18 ========================================
top12 = corrs.abs().sort_values(ascending=False).head(12).index

plt.figure(figsize=(9, 7))
sns.heatmap(
    df[list(top12)].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
)
plt.title("Correlations among the 12 strongest safe features")
plt.tight_layout()
plt.show()

# ===== Code cell 19 ========================================
LEAKY_COLS = ["URLSimilarityIndex", "URL", "Domain", "Title"]

clean_features = [
    c for c in df.columns if c not in LEAKY_COLS + ["label", "TLD"]
]

manifest = {
    "target": "label",
    "dropped_leaky": LEAKY_COLS,
    "dropped_identifier": [],
    "categorical_raw": ["TLD"],
    "drop_duplicate_urls": True,
    "split_group_column": "Domain",
    "clean_numeric_features": clean_features,
}

out_path = Path("../data/01_feature_columns.json")
out_path.write_text(json.dumps(manifest, indent=2))
print(f"saved {len(clean_features)} feature names to {out_path}")

