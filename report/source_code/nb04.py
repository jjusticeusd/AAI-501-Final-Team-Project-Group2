# Notebook 04 - Validity and Error Analysis (Guna Pasupathy)
# Exported from 04.ipynb (code cells only, in order).

# ===== Code cell 1 ========================================
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy.stats import binomtest
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

manifest = json.loads(Path("../data/feature_columns.json").read_text())
clean = manifest["clean_numeric_features"]

df = pd.read_csv("../data/phiusiil.csv")
model_df = df.drop_duplicates(subset="URL").reset_index(drop=True)
y = model_df["label"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(model_df, y, groups=model_df["Domain"]))
y_te = y.iloc[te].values
print(f"train {len(tr):,} / test {len(te):,}")

# ===== Code cell 2 ========================================
URL_COLS = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]


def fit_eval(X, name):
    """Train notebook-02-style XGBoost, return preds + confusion."""
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr = y.iloc[tr]
    m = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    is_phish = pred == 0
    was_phish = y_te == 0
    tp = int((is_phish & was_phish).sum())
    fp = int((is_phish & ~was_phish).sum())
    fn = int((~is_phish & was_phish).sum())
    tn = int((~is_phish & ~was_phish).sum())
    acc = (tp + tn) / len(y_te)
    print(
        f"{name:28s} acc={acc:.4f}  FP={fp:3d}  FN={fn:3d}  "
        f"prec={tp / (tp + fp):.4f}  rec={tp / (tp + fn):.4f}"
    )
    return m, pred, (tp, fp, fn, tn)


m_full, p_full, cm_full = fit_eval(model_df[clean], "full clean (49)")
m_url, p_url, cm_url = fit_eval(model_df[URL_COLS], "URL-only (21)")

# ===== Code cell 3 ========================================
def entropy(s):
    counts = pd.Series(list(s)).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


BRANDS = [
    "google",
    "facebook",
    "youtube",
    "amazon",
    "apple",
    "microsoft",
    "paypal",
    "netflix",
    "instagram",
    "whatsapp",
    "linkedin",
    "twitter",
    "ebay",
    "chase",
    "wellsfargo",
    "bankofamerica",
    "citibank",
    "hsbc",
    "americanexpress",
    "dropbox",
    "adobe",
    "yahoo",
    "outlook",
    "office",
    "icloud",
    "steam",
    "spotify",
    "coinbase",
    "binance",
    "walmart",
    "usps",
    "fedex",
    "dhl",
    "irs",
    "att",
    "verizon",
]
SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rb.gy",
    "shorturl.at",
}


def sld(domain):
    """second-level label: mail.foo.com -> foo"""
    parts = domain.lower().removeprefix("www.").split(".")
    return parts[-2] if len(parts) >= 2 else parts[0]


def brand_dist(domain):
    s = sld(domain)
    return min(Levenshtein.normalized_distance(s, b) for b in BRANDS)


eng = pd.DataFrame(index=model_df.index)
eng["url_entropy"] = model_df["URL"].str.lower().map(entropy)
eng["domain_entropy"] = model_df["Domain"].str.lower().map(entropy)
eng["brand_distance"] = model_df["Domain"].map(brand_dist)
eng["is_punycode"] = (
    model_df["Domain"].str.lower().str.contains("xn--").astype(int)
)
host = model_df["Domain"].str.lower().str.removeprefix("www.")
eng["is_shortener"] = host.isin(SHORTENERS).astype(int)

m_urlp, p_urlp, cm_urlp = fit_eval(
    pd.concat([model_df[URL_COLS], eng], axis=1),
    "URL-only + engineered (26)",
)
m_nh, p_nh, cm_nh = fit_eval(
    model_df[[c for c in URL_COLS if c != "IsHTTPS"]],
    "URL-only, no IsHTTPS (20)",
)

imp = pd.Series(
    m_urlp.feature_importances_, index=URL_COLS + list(eng.columns)
)
ranks = imp.rank(ascending=False, method="min").astype(int)
print("\nengineered feature importance ranks (out of 26):")
print(ranks[eng.columns].to_string())

# ===== Code cell 4 ========================================
rng = np.random.default_rng(42)


def boot_ci(err, n_boot=2000):
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    accs = 1 - err[idx].mean(axis=1)
    return np.percentile(accs, [2.5, 97.5])


err_full = (p_full != y_te).astype(int)
err_url = (p_url != y_te).astype(int)

lo, hi = boot_ci(err_full)
print(f"full model acc:     {1 - err_full.mean():.5f} [{lo:.5f}, {hi:.5f}]")
lo, hi = boot_ci(err_url)
print(f"URL-only model acc: {1 - err_url.mean():.5f} [{lo:.5f}, {hi:.5f}]")

only_url = int(((err_full == 0) & (err_url == 1)).sum())
only_full = int(((err_full == 1) & (err_url == 0)).sum())
p_val = binomtest(min(only_url, only_full), only_url + only_full).pvalue
print(
    f"\nMcNemar: URL-only-errors={only_url}, "
    f"full-only-errors={only_full}, p={p_val:.2e}"
)

# ===== Code cell 5 ========================================
def rates(cm):
    tp, fp, fn, tn = cm
    return tp / (tp + fn), fp / (fp + tn), tn + fp


tpr_f, fpr_f, n_leg = rates(cm_full)
tpr_u, fpr_u, _ = rates(cm_url)
fpr_f_up = 3 / n_leg if fpr_f == 0 else fpr_f

pis = np.logspace(-5, np.log10(0.43), 300)


def prec_curve(tpr, fpr):
    return np.where(fpr == 0, 1.0, pis * tpr / (pis * tpr + (1 - pis) * fpr))


plt.figure(figsize=(8, 4.5))
plt.plot(
    pis,
    prec_curve(tpr_f, fpr_f_up),
    color="tab:blue",
    label="full model (pessimistic: 0 FPs -> rule of three)",
)
plt.plot(
    pis,
    prec_curve(tpr_u, fpr_u),
    color="tab:orange",
    label="URL-only model (12 observed FPs)",
)
plt.axvline(0.43, color="gray", linestyle=":", linewidth=1)
plt.text(0.43, 0.05, " benchmark (43%)", fontsize=8, rotation=90)
plt.axvline(1e-4, color="gray", linestyle=":", linewidth=1)
plt.text(1e-4, 0.05, " realistic (1:10,000)", fontsize=8, rotation=90)
plt.xscale("log")
plt.xlabel("phishing prevalence in traffic (log scale)")
plt.ylabel("precision of the phishing flag")
plt.title("Benchmark precision does not survive realistic base rates")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("precision at selected prevalences:")
for pi in (0.43, 1e-2, 1e-3, 1e-4):
    pf = pi * tpr_f / (pi * tpr_f + (1 - pi) * fpr_f_up)
    pu = pi * tpr_u / (pi * tpr_u + (1 - pi) * fpr_u)
    print(
        f"  prevalence {pi:>7}: full(pessimistic)={pf:.3f}  URL-only={pu:.3f}"
    )

# ===== Code cell 6 ========================================
te_df = model_df.iloc[te]
missed = te_df[(p_url == 1) & (y_te == 0)]
caught = te_df[(p_url == 0) & (y_te == 0)]

prof = [
    "IsHTTPS",
    "TLDLegitimateProb",
    "URLLength",
    "SpacialCharRatioInURL",
    "NoOfDegitsInURL",
]
print(f"missed phishing URLs: {len(missed)}")
print("\nmedians, missed vs caught phishing:")
print(
    pd.DataFrame(
        {"missed": missed[prof].median(), "caught": caught[prof].median()}
    ).round(3)
)
print("\nexamples of missed URLs:")
for u in missed["URL"].head(10):
    print("  " + u[:90])

# ===== Code cell 7 ========================================
import urllib.request
from collections import Counter
from urllib.parse import urlsplit

EXT_DIR = Path("../data/external")
EXT_DIR.mkdir(exist_ok=True)
BASE = "https://raw.githubusercontent.com/ebubekirbbr/pdd/master/input/"
for fname in ("data_legitimate_36400.json", "data_phishing_37175.json"):
    if not (EXT_DIR / fname).exists():
        print("downloading", fname)
        urllib.request.urlretrieve(BASE + fname, EXT_DIR / fname)

TLD_PROB = df.groupby("TLD")["TLDLegitimateProb"].first().to_dict()
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def str_entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * np.log2(c / n) for c in Counter(s).values())


def featurize(url):
    """~20 lexical features computed from the raw URL text only."""
    u = url.strip().lower()
    if "://" not in u:
        return None
    parts = urlsplit(u)
    hostname = parts.netloc.split("@")[-1].split(":")[0]
    if not hostname:
        return None
    rest = parts.path + ("?" + parts.query if parts.query else "")
    labels = hostname.split(".")
    sld_ = labels[-2] if len(labels) >= 2 else labels[0]
    tld = labels[-1]
    n = len(u)
    letters = sum(c.isalpha() for c in u)
    digits = sum(c.isdigit() for c in u)
    return {
        "url_len": n,
        "host_len": len(hostname),
        "path_len": len(rest),
        "n_host_labels": len(labels),
        "tld_len": len(tld),
        "tld_legit_prob": TLD_PROB.get(tld, 0.0),
        "is_ip": int(bool(IP_RE.match(hostname))),
        "n_hyphens_host": hostname.count("-"),
        "letter_ratio": letters / n,
        "digit_ratio": digits / n,
        "special_ratio": (n - letters - digits) / n,
        "digits_in_host": sum(c.isdigit() for c in hostname),
        "n_equals": u.count("="),
        "n_qmark": u.count("?"),
        "n_amp": u.count("&"),
        "n_pct": u.count("%"),
        "url_entropy": str_entropy(u),
        "host_entropy": str_entropy(hostname),
        "brand_edit": min(
            Levenshtein.normalized_distance(sld_, b) for b in BRANDS
        ),
        "brand_substr": int(
            any(b in hostname for b in BRANDS) and sld_ not in BRANDS
        ),
    }


F_phi = pd.DataFrame([featurize(u) for u in model_df["URL"]])

ext_leg = json.loads((EXT_DIR / "data_legitimate_36400.json").read_text())
ext_phi = json.loads((EXT_DIR / "data_phishing_37175.json").read_text())
rows, ext_labels, seen = [], [], set()
for urls, lab in ((ext_leg, 1), (ext_phi, 0)):
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        f = featurize(u)
        if f is not None:
            rows.append(f)
            ext_labels.append(lab)
F_ext = pd.DataFrame(rows)
y_ext = np.array(ext_labels)
print(
    f"external usable: {len(F_ext):,} "
    f"(legit {(y_ext == 1).sum():,} / "
    f"phish {(y_ext == 0).sum():,})"
)

# ===== Code cell 8 ========================================
def fit_transfer(cols, name):
    m = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y.iloc[tr] == 0).sum() / (y.iloc[tr] == 1).sum(),
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )
    m.fit(F_phi.iloc[tr][cols], y.iloc[tr])

    def rep(pred, truth, where):
        is_p, was_p = pred == 0, truth == 0
        tp = int((is_p & was_p).sum())
        fp = int((is_p & ~was_p).sum())
        fn = int((~is_p & was_p).sum())
        tn = int((~is_p & ~was_p).sum())
        acc = (tp + tn) / len(truth)
        print(
            f"  {where:22s} acc={acc:.4f} "
            f"prec={tp / max(tp + fp, 1):.4f} "
            f"rec={tp / max(tp + fn, 1):.4f}  FP={fp} FN={fn}"
        )
        return (tp, fp, fn, tn)

    print(name)
    rep(m.predict(F_phi.iloc[te][cols]), y.iloc[te].values, "PhiUSIIL test")
    ext_cm = rep(m.predict(F_ext[cols]), y_ext, "Sahingoz external")
    return m, ext_cm


ALL_LEX = list(F_phi.columns)
m_lex, ext_cm_lex = fit_transfer(ALL_LEX, "full lexical (20 feats):")
imp_lex = pd.Series(m_lex.feature_importances_, index=ALL_LEX)
print("\ntop importances:", imp_lex.nlargest(4).round(3).to_dict())
print(
    "median path_len - PhiUSIIL legit:",
    float(F_phi["path_len"][y.values == 1].median()),
    "| PhiUSIIL phish:",
    float(F_phi["path_len"][y.values == 0].median()),
    "| external legit:",
    float(F_ext["path_len"][y_ext == 1].median()),
)

# ===== Code cell 9 ========================================
HOST_ONLY = [
    "host_len",
    "n_host_labels",
    "tld_len",
    "tld_legit_prob",
    "is_ip",
    "n_hyphens_host",
    "digits_in_host",
    "host_entropy",
    "brand_edit",
    "brand_substr",
]
m_host, ext_cm_host = fit_transfer(HOST_ONLY, "host-only (10 feats):")

# ===== Code cell 10 ========================================
results = {
    "confusion": {
        "full": cm_full,
        "url_only": cm_url,
        "url_plus_engineered": cm_urlp,
        "url_no_https": cm_nh,
    },
    "mcnemar_p": float(p_val),
    "precision_at_prevalence": {
        str(pi): {
            "full_pessimistic": float(
                pi * tpr_f / (pi * tpr_f + (1 - pi) * fpr_f_up)
            ),
            "url_only": float(pi * tpr_u / (pi * tpr_u + (1 - pi) * fpr_u)),
        }
        for pi in (0.43, 1e-2, 1e-3, 1e-4)
    },
    "n_missed_url_only": int(len(missed)),
    "transfer_sahingoz": {
        "full_lexical": ext_cm_lex,
        "host_only": ext_cm_host,
        "n_external": int(len(y_ext)),
    },
}
Path("../data/nb04_results.json").write_text(json.dumps(results, indent=2))
print("saved ../data/nb04_results.json")
