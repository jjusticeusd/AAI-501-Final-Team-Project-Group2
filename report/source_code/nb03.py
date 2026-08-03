# Notebook 03 - CharCNN, Clustering, SOTA (Pranav Dhinakar)
# Exported from 03_charcnn_clustering_sota.ipynb (code cells only, in order).

# ===== Code cell 1 ========================================
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Section-specific heavy deps (torch, tldextract, hdbscan, rapidfuzz)
# are imported at the top of their own sections to keep setup light.

DATA_PATH = Path("../data/phiusiil.csv")
MANIFEST_PATH = Path("../data/01_feature_columns.json")

# ===== Code cell 2 ========================================
df = pd.read_csv(DATA_PATH)
manifest = json.loads(MANIFEST_PATH.read_text())

target = manifest["target"]
clean_features = manifest["clean_numeric_features"]
group_col = manifest["split_group_column"]

# Same pre-split cleaning NB 02 uses, so our numbers are comparable.
# We keep URL and Domain as columns (raw material for the char model,
# engineered features and grouping) even though they're not features.
if manifest["drop_duplicate_urls"]:
    before = len(df)
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    print(f"dropped {before - len(df)} duplicate-URL rows")

y = df[target]

print(f"rows: {len(df):,}")
print(f"clean feature count: {len(clean_features)}")
print(f"group column for splits: {group_col}")

# ===== Code cell 3 ========================================
import tldextract

# eTLD+1, e.g. "gateway.ipfs.io" -> "ipfs.io". Uses a bundled public
# suffix list; suppress the network refresh for reproducibility.
extractor = tldextract.TLDExtract(suffix_list_urls=())


def registered_domain(host: str) -> str:
    ext = extractor(str(host))
    # fall back to the raw host for IP-address URLs (no eTLD+1)
    return ext.top_domain_under_public_suffix or str(host)


df["RegDomain"] = df["Domain"].map(registered_domain)

n_exact = df["Domain"].nunique()
n_reg = df["RegDomain"].nunique()
print(f"exact Domain groups:     {n_exact:,}")
print(f"registered-domain groups: {n_reg:,}")
print(f"collapsed by {n_exact - n_reg:,} groups")
print()
# how many rows share a registered domain with another row?
reg_counts = df["RegDomain"].value_counts()
reg_repeated = reg_counts[reg_counts > 1]
print(
    f"rows sharing a registered domain: "
    f"{reg_repeated.sum():,} ({reg_repeated.sum() / len(df):.1%})"
)
reg_counts.head(10)

# ===== Code cell 4 ========================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def quick_logreg(X, y, groups):
    """Basic logistic regression under a grouped split; return acc, F1."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), f1_score(y_test, preds)


X = df[clean_features]
acc_exact, f1_exact = quick_logreg(X, y, groups=df["Domain"])
acc_reg, f1_reg = quick_logreg(X, y, groups=df["RegDomain"])

print(f"exact-Domain split:      acc={acc_exact:.4f}  f1={f1_exact:.4f}")
print(f"registered-domain split: acc={acc_reg:.4f}  f1={f1_reg:.4f}")

# ===== Code cell 5 ========================================
# optional: treat multi-tenant PaaS suffixes as public, so
# victim-a.web.app and victim-b.web.app become separate groups
extractor_strict = tldextract.TLDExtract(
    suffix_list_urls=(), include_psl_private_domains=True
)
df["RegDomainStrict"] = df["Domain"].map(
    lambda h: extractor_strict(str(h)).top_domain_under_public_suffix or str(h)
)
print(f"strict (PaaS-aware) groups: {df['RegDomainStrict'].nunique():,}")

# ===== Code cell 6 ========================================
acc_strict, f1_strict = quick_logreg(X, y, groups=df["RegDomainStrict"])
print(f"PaaS-aware split: acc={acc_strict:.4f}  f1={f1_strict:.4f}")

# ===== Code cell 7 ========================================
# Build a character vocabulary from the training URLs only (fit on
# train to avoid leaking test-set characters into the vocab). Reserve
# 0 for padding and 1 for unknown/out-of-vocab characters.
MAX_LEN = 200  # covers ~99th pct of URL lengths; longer URLs truncated

urls = df["URL"].astype(str).values
labels = df[target].values

lengths = np.array([len(u) for u in urls])
print(
    f"URL length: median={np.median(lengths):.0f}  "
    f"95th={np.percentile(lengths, 95):.0f}  "
    f"99th={np.percentile(lengths, 99):.0f}  max={lengths.max()}"
)
print(
    f"URLs longer than MAX_LEN={MAX_LEN}: "
    f"{(lengths > MAX_LEN).sum()} ({(lengths > MAX_LEN).mean():.2%})"
)

# ===== Code cell 8 ========================================
from collections import Counter


# Phishing = positive, matching NB 02. df is already dup-URL-dropped and
# index-reset (Section 1), so this split reproduces NB 02's test set.
y_phish = (1 - df[target]).to_numpy()
groups = df["Domain"].to_numpy()
urls = df["URL"].astype(str).to_numpy()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y_phish, groups=groups))

url_train, url_test = urls[train_idx], urls[test_idx]
y_train, y_test = y_phish[train_idx], y_phish[test_idx]

print(f"train: {len(train_idx):,}   test: {len(test_idx):,}")
print(f"phishing rate  train={y_train.mean():.3f}  test={y_test.mean():.3f}")

# Character vocabulary, fit on TRAIN urls only. 0 = <pad>, 1 = <unk>.
char_counts = Counter()
for u in url_train:
    char_counts.update(u)

MIN_FREQ = 5  # chars rarer than this in train collapse to <unk>
vocab = {"<pad>": 0, "<unk>": 1}
for ch, cnt in char_counts.most_common():
    if cnt >= MIN_FREQ:
        vocab[ch] = len(vocab)

print(f"distinct chars in train: {len(char_counts)}")
print(f"vocab size (>= {MIN_FREQ} occ, + pad/unk): {len(vocab)}")

# ===== Code cell 9 ========================================
def encode(url, vocab, max_len=MAX_LEN):
    ids = [vocab.get(c, 1) for c in url[:max_len]]  # 1 = <unk>
    ids += [0] * (max_len - len(ids))  # 0 = <pad>
    return ids


X_train_ids = np.array([encode(u, vocab) for u in url_train], dtype=np.int64)
X_test_ids = np.array([encode(u, vocab) for u in url_test], dtype=np.int64)

trunc = (np.array([len(u) for u in url_train]) > MAX_LEN).mean()
print(f"X_train_ids: {X_train_ids.shape}   X_test_ids: {X_test_ids.shape}")
print(f"train URLs truncated at {MAX_LEN}: {trunc:.2%}")
print("example row (first 40 ids):", X_train_ids[0][:40])

# ===== Code cell 10 ========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device: {device}")

# Inner grouped train/val split; val only monitors convergence / early
# stopping. Grouped so no domain straddles train and val either.
groups_train = groups[train_idx]
inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
tr_i, va_i = next(inner.split(X_train_ids, y_train, groups=groups_train))

Xtr = torch.tensor(X_train_ids[tr_i], dtype=torch.long)
ytr = torch.tensor(y_train[tr_i], dtype=torch.float32)
Xva = torch.tensor(X_train_ids[va_i], dtype=torch.long)
yva = torch.tensor(y_train[va_i], dtype=torch.float32)
Xte = torch.tensor(X_test_ids, dtype=torch.long)
yte = torch.tensor(y_test, dtype=torch.float32)

BATCH = 512
train_loader = DataLoader(
    TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True
)
val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=BATCH)
test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=BATCH)

# class weight for phishing (positive): n_neg / n_pos on the train split
n_pos = float(ytr.sum())
n_neg = float(len(ytr) - ytr.sum())
pos_weight = torch.tensor([n_neg / n_pos], device=device)
print(
    f"train/val: {len(tr_i):,}/{len(va_i):,}   pos_weight={n_neg / n_pos:.3f}"
)


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=48,
        n_filters=128,
        kernels=(3, 5, 7),
        dropout=0.4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, n_filters, k, padding=k // 2)
                for k in kernels
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernels), 1)

    def forward(self, x):
        e = self.embed(x).permute(0, 2, 1)  # (B, embed, L)
        feats = [
            F.relu(conv(e)).max(dim=2).values  # global max-pool
            for conv in self.convs
        ]
        h = self.dropout(torch.cat(feats, dim=1))
        return self.fc(h).squeeze(1)  # (B,) logits


model = CharCNN(vocab_size=len(vocab)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params:,}")

# ===== Code cell 11 ========================================
from sklearn.metrics import average_precision_score

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 6


@torch.no_grad()
def predict_proba(loader):
    model.eval()
    probs = []
    for xb, _ in loader:
        logits = model(xb.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


best_val_ap, best_state = -1.0, None
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    train_loss = total / len(Xtr)

    val_p = predict_proba(val_loader)
    val_ap = average_precision_score(yva.numpy(), val_p)
    val_f1 = f1_score(yva.numpy(), (val_p >= 0.5).astype(int))
    flag = ""
    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        flag = "  <- best"
    print(
        f"epoch {epoch}  train_loss={train_loss:.4f}  "
        f"val_PR_AUC={val_ap:.4f}  val_F1={val_f1:.4f}{flag}"
    )

model.load_state_dict(best_state)
print(f"\nrestored best val PR-AUC = {best_val_ap:.4f}")

# ===== Code cell 12 ========================================
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

test_proba = predict_proba(test_loader)
test_pred = (test_proba >= 0.5).astype(int)

char_result = {
    "model": "CharCNN (URL only)",
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred),
    "recall": recall_score(y_test, test_pred),
    "f1": f1_score(y_test, test_pred),
    "roc_auc": roc_auc_score(y_test, test_proba),
    "pr_auc": average_precision_score(y_test, test_proba),
}
print(pd.Series(char_result).to_string())

cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(4, 3.4))
sns.heatmap(
    cm,
    annot=True,
    fmt=",d",
    cmap="Blues",
    cbar=False,
    xticklabels=["legit", "phish"],
    yticklabels=["legit", "phish"],
)
plt.title("CharCNN (URL only): test")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.tight_layout()
plt.show()

# ===== Code cell 13 ========================================
# Cluster the PHISHING rows only, on the leakage-clean feature set from
# NB 01 (label held out). Goal: find structure within phishing,
# "attack families", not to separate phishing from legit.
phish_mask = (df[target] == 0).to_numpy()
Xp_raw = df.loc[phish_mask, clean_features].copy()

# Drop zero-variance columns within the phishing subset (constant here,
# so they carry no clustering signal and would clutter the profile).
nunique = Xp_raw.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
Xp_raw = Xp_raw.drop(columns=const_cols)
cluster_features = Xp_raw.columns.tolist()

scaler = StandardScaler()
Xp = scaler.fit_transform(Xp_raw)

print(f"phishing rows: {Xp.shape[0]:,}")
print(
    f"clustering features: {Xp.shape[1]} "
    f"(dropped {len(const_cols)} constant: {const_cols})"
)

# ===== Code cell 14 ========================================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ks = range(2, 11)
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbl = km.fit_predict(Xp)
    inertias.append(km.inertia_)
    # silhouette on a subsample; full O(n^2) is infeasible at ~100k rows
    sil = silhouette_score(Xp, lbl, sample_size=10000, random_state=42)
    silhouettes.append(sil)
    print(f"k={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(ks), inertias, "o-", color="tab:blue")
ax1.set_xlabel("k")
ax1.set_ylabel("inertia (within-cluster SS)")
ax1.set_title("Elbow")
ax2.plot(list(ks), silhouettes, "o-", color="tab:orange")
ax2.set_xlabel("k")
ax2.set_ylabel("silhouette (10k sample)")
ax2.set_title("Silhouette")
plt.tight_layout()
plt.show()

# ===== Code cell 15 ========================================
K_FAMILIES = 5

# Dominant structure is binary (k=2, silhouette 0.44); characterize it
# briefly, then use k=5 for interpretable families.
km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xp)
sizes2 = pd.Series(km2.labels_).value_counts().sort_index()
print("k=2 dominant split sizes:")
for c, n in sizes2.items():
    print(f"  cluster {c}: {n:,} ({n / len(Xp):.1%})")

km5 = KMeans(n_clusters=K_FAMILIES, n_init=10, random_state=42)
phish_labels = km5.fit_predict(Xp)

profile = Xp_raw.copy()
profile["cluster"] = phish_labels
sizes = pd.Series(phish_labels).value_counts().sort_index()
print(f"\nk={K_FAMILIES} family sizes:")
for c, n in sizes.items():
    print(f"  cluster {c}: {n:,} ({n / len(Xp):.1%})")

# Interpretable raw-unit profile (only columns present in the feature set)
shortlist = [
    "IsDomainIP",
    "IsHTTPS",
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "DegitRatioInURL",
    "SpacialCharRatioInURL",
    "LineOfCode",
    "NoOfJS",
    "NoOfImage",
    "NoOfExternalRef",
    "NoOfCSS",
    "HasSocialNet",
    "HasCopyrightInfo",
    "HasDescription",
    "HasPasswordField",
    "HasSubmitButton",
    "TLDLegitimateProb",
]
shortlist = [c for c in shortlist if c in cluster_features]
means = profile.groupby("cluster")[shortlist].mean().T
means.columns = [f"c{c}" for c in means.columns]
print(f"\nper-cluster mean (raw units), {len(shortlist)} features:")
print(means.round(2).to_string())

# ===== Code cell 16 ========================================
# KMeans centroids live in standardized space, so each value is that
# cluster's mean in SDs from the phishing-wide mean. Large |z| = what
# makes the family distinctive.
centroids = pd.DataFrame(km5.cluster_centers_, columns=cluster_features)
for c in range(K_FAMILIES):
    row = centroids.loc[c]
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(6)
    print(f"cluster {c} (n={sizes[c]:,}), top distinguishing features:")
    for feat, z in top.items():
        arrow = "↑" if z > 0 else "↓"
        print(f"    {feat:24s} {z:+.2f} {arrow}")
    print()

# ===== Code cell 17 ========================================
from sklearn.decomposition import PCA

FAMILY_NAMES = {
    0: "empty-shell (68%)",
    1: "outlier long-URL (5)",
    2: "IP-host / obfuscated (1%)",
    3: "credential forms (12%)",
    4: "HTTP lookalike stubs (19%)",
}

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Xp)
ev = pca.explained_variance_ratio_
print(
    f"PC1+PC2 explained variance: {ev[0]:.1%} + {ev[1]:.1%} = {ev.sum():.1%}"
)

# Subsample for a readable scatter; fit was on all rows.
rng = np.random.default_rng(42)
idx = rng.choice(len(coords), size=min(15000, len(coords)), replace=False)

plt.figure(figsize=(8, 6))
for c in range(K_FAMILIES):
    m = phish_labels[idx] == c
    plt.scatter(
        coords[idx][m, 0],
        coords[idx][m, 1],
        s=4,
        alpha=0.4,
        label=FAMILY_NAMES[c],
    )
plt.xlabel(f"PC1 ({ev[0]:.0%} var)")
plt.ylabel(f"PC2 ({ev[1]:.0%} var)")
plt.title("Phishing families: PCA projection (15k sample)")
plt.legend(markerscale=3, fontsize=8, loc="best")
plt.tight_layout()
plt.show()

# ===== Code cell 18 ========================================
from sklearn.cluster import HDBSCAN

# Density-based cross-check on a subsample (HDBSCAN is heavier than
# KMeans). Unlike KMeans it doesn't force every point into a cluster;
# it can label sparse outliers as noise (-1), which is the honest test
# of how much crisp structure exists and whether the 5-row KMeans
# micro-cluster survives.
sub = rng.choice(len(Xp), size=min(25000, len(Xp)), replace=False)
hdb = HDBSCAN(min_cluster_size=250, min_samples=10)
hdb_labels = hdb.fit_predict(Xp[sub])

n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
noise = (hdb_labels == -1).mean()
print(f"HDBSCAN on {len(sub):,}-row sample")
print(f"clusters found: {n_clusters}   noise: {noise:.1%}")
print()
sizes_hdb = pd.Series(hdb_labels).value_counts().sort_index()
for c, n in sizes_hdb.items():
    tag = " (noise)" if c == -1 else ""
    print(f"  label {c:>2}: {n:,} ({n / len(sub):.1%}){tag}")

# ===== Code cell 19 ========================================
# NB 02's saved tabular metrics (phishing = positive, same exact-Domain
# split as this notebook, so the rows below are directly comparable).
nb02_path = Path("../data/02_classical_results.json")
assert nb02_path.exists(), (
    "Run NB 02 first; it saves 02_classical_results.json this table needs."
)
nb02 = json.loads(nb02_path.read_text())
assert nb02["positive_class"] == "phishing"

tab = pd.DataFrame(nb02["metrics"]).set_index("model")

# Append this notebook's CharCNN result (live variable from above).
char_row = (
    pd.Series(char_result).drop("model").to_frame(char_result["model"]).T
)
metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
our = pd.concat([tab, char_row])[metric_cols].astype(float)

print(
    f"NB 02 test rows: {nb02['n_test']:,}   "
    f"CharCNN test rows: {len(y_test):,}  (shared split)\n"
)
print(our.round(4).to_string())

# Graphical empirical comparison (rubric asks for this explicitly).
# Zoom the y-axis since every model is near the ceiling.
plot_metrics = ["accuracy", "recall", "pr_auc"]
lo = float(our[plot_metrics].min().min())

ax = our[plot_metrics].plot.bar(
    figsize=(11, 4.5),
    width=0.82,
    color=["tab:blue", "tab:orange", "tab:green"],
)
ax.set_ylim(max(0.0, lo - 0.01), 1.001)
ax.set_ylabel("score")
ax.set_title("Our models: key metrics (test set, phishing = positive)")
ax.legend(loc="lower right")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# ===== Code cell 20 ========================================
nb03_out = {
    "notebook": "03",
    "positive_class": "phishing",
    "split": {
        "grouping": "exact Domain (matches NB 02)",
        "test_size": 0.2,
        "random_state": 42,
        "n_test": int(len(y_test)),
        "grouping_invariance": {
            "exact_domain": 0.9995,
            "paas_aware_etld1": 0.9993,
            "naive_etld1": 0.9990,
        },
    },
    "charcnn": {
        "description": "char-level CNN on raw URL string only, no "
        "engineered/content/similarity features",
        "max_len": int(MAX_LEN),
        "vocab_size": int(len(vocab)),
        "n_params": int(n_params),
        "metrics": {
            k: float(v) for k, v in char_result.items() if k != "model"
        },
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    },
    "clustering": {
        "algorithm": "KMeans",
        "k": int(K_FAMILIES),
        "n_phishing_rows": int(Xp.shape[0]),
        "silhouette_k2": 0.4373,
        "silhouette_k5": 0.1540,
        "families": {
            FAMILY_NAMES[c]: int(sizes[c]) for c in range(K_FAMILIES)
        },
        "hdbscan_check": {
            "sample_size": int(len(sub)),
            "clusters": int(n_clusters),
            "noise_fraction": float(noise),
        },
    },
    "our_models_vs_literature": {
        "our_models": our.round(4)
        .reset_index()
        .rename(columns={"index": "model"})
        .to_dict(orient="records"),
        "phiusiil_published_band": "99.5-100%",
        "url_only_literature_band": "97-99% (other corpora)",
    },
}

out_path = Path("../data/03_charcnn_cluster_results.json")
out_path.write_text(json.dumps(nb03_out, indent=2))
print(f"saved {out_path}")
print(
    f"  charcnn: {nb03_out['charcnn']['metrics']['accuracy']:.4f} acc, "
    f"{nb03_out['charcnn']['metrics']['pr_auc']:.4f} PR-AUC"
)
print(f"  families: {nb03_out['clustering']['families']}")

