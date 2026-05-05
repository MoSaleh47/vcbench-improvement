"""
generate_pr_curve.py
====================
Generates Figure 1 (Precision-Recall curve) for the NeurIPS 2026 paper
"Failure Modes and Calibration of Local LLMs on Imbalanced VC Prediction".

Usage (from the project repo root, with vcbench_final_public.csv present):
    python generate_pr_curve.py

Outputs:
    pr_curve.pdf   — vector figure ready for \includegraphics in LaTeX
    pr_curve.png   — raster version for quick preview

Requirements:
    pip install scikit-learn matplotlib numpy pandas scipy

What the script does
--------------------
1. Loads the public VCBench split and applies the same stratified 80/20
   split (seed=42) used in all experiments.
2. Fits a Logistic Regression on TF-IDF features of anonymised_prose on
   the train split (class_weight='balanced').
3. Scores the 900-row validation split to obtain a continuous probability
   for each profile, then plots the full precision-recall curve with
   PR-AUC in the legend.
4. Adds point markers for:
      • All-SUCCESS  (precision = base_rate, recall = 1.0)
      • All-FAILURE  (precision = 0, recall = 0)
      • LLM systems from the paper (hard labels → single PR point each)
5. Adds a shaded "no-skill" reference line at precision = base_rate.
6. Saves pr_curve.pdf and pr_curve.png.

LLM operating points (from Tables 2 & 3 in the paper, 120-sample subset)
-------------------------------------------------------------------------
These are hard-label operating points; they appear as scatter points, not
curves, because we only have the final predictions rather than scores.

If you rerun LLM experiments with log-probability outputs, replace the
llm_points dict with continuous scores and add a proper PR curve per model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------

DATA_PATH   = "vcbench_final_public.csv"   # adjust path if needed
SEED        = 42
TEST_SIZE   = 0.20          # 20 % → 900-row validation pool
SAMPLE_SIZE = 120           # subset used for LLM experiments
POS_LABEL   = 1             # integer label after mapping SUCCESS→1

# LLM hard-label operating points  {label: (precision, recall, marker, color)}
# Values taken from Table 2 (seed=42, 120-sample) of the paper.
llm_points = {
    "qwen3:32b\nFew-Shot (PPR 40.8%)": (0.1633, 0.7273, "^",  "#e07b39"),
    "qwen3:32b\nVanilla (PPR 21.7%)":  (0.1923, 0.4545, "s",  "#c0392b"),
    "GGUF Vanilla\n(PPR 1.7%)":        (0.5000, 0.0909, "D",  "#8e44ad"),
}

# Trivial baselines (computed analytically from the 120-sample set)
BASE_RATE_120 = 11 / 120   # ~0.0917

# ---------------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------------

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find '{DATA_PATH}'. "
        "Run this script from the directory containing vcbench_final_public.csv."
    )

# Normalise column names
df.columns = df.columns.str.strip().str.lower()

required = {"anonymised_prose", "success"}
missing  = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

df["label"] = df["success"].astype(int)

# ---------------------------------------------------------------------------
# 2.  Train / validation split (mirrors paper protocol)
# ---------------------------------------------------------------------------

train_df, val_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
)

X_train = train_df["anonymised_prose"].fillna("").values
y_train = train_df["label"].values
X_val   = val_df["anonymised_prose"].fillna("").values
y_val   = val_df["label"].values

base_rate_900 = y_val.mean()
print(f"Validation set : {len(y_val)} profiles, {y_val.sum()} positives "
      f"(base rate {base_rate_900:.3f})")

# ---------------------------------------------------------------------------
# 3.  Fit LR-TF-IDF pipeline
# ---------------------------------------------------------------------------

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                              sublinear_tf=True)),
    ("clf",   LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=1_000, random_state=SEED)),
])

pipe.fit(X_train, y_train)
y_scores = pipe.predict_proba(X_val)[:, 1]   # probability of SUCCESS

precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
pr_auc = auc(recall, precision)

# Best F0.5 threshold on the full 900-row validation set
best_f05, best_thr = 0.0, 0.5
for thr in thresholds:
    preds = (y_scores >= thr).astype(int)
    tp = ((preds == 1) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    fn = ((preds == 0) & (y_val == 1)).sum()
    if tp + fp == 0 or tp + fn == 0:
        continue
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    denom = (1 + 0.5**2) * prec * rec
    numer = (0.5**2 * prec + rec)
    f05 = denom / numer if numer > 0 else 0.0
    if f05 > best_f05:
        best_f05, best_thr = f05, thr

lr_preds = (y_scores >= best_thr).astype(int)
lr_tp = ((lr_preds == 1) & (y_val == 1)).sum()
lr_fp = ((lr_preds == 1) & (y_val == 0)).sum()
lr_fn = ((lr_preds == 0) & (y_val == 1)).sum()
lr_prec = lr_tp / (lr_tp + lr_fp) if (lr_tp + lr_fp) > 0 else 0.0
lr_rec  = lr_tp / (lr_tp + lr_fn) if (lr_tp + lr_fn) > 0 else 0.0
print(f"LR-TF-IDF (threshold={best_thr:.3f}): "
      f"F0.5={best_f05:.4f}, P={lr_prec:.4f}, R={lr_rec:.4f}, "
      f"TP={lr_tp}, FP={lr_fp}, FN={lr_fn}, "
      f"PPR={100*(lr_tp+lr_fp)/len(y_val):.1f}%")

# ---------------------------------------------------------------------------
# 4.  Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 5.0))

# No-skill baseline (horizontal line at base rate)
ax.axhline(y=base_rate_900, color="grey", linewidth=1.0,
           linestyle="--", label=f"No-skill baseline (base rate {base_rate_900:.3f})")

# LR-TF-IDF PR curve
ax.plot(recall, precision, color="#2980b9", linewidth=2.0,
        label=f"LR-TF-IDF (PR-AUC = {pr_auc:.4f})")

# LR operating point
ax.scatter([lr_rec], [lr_prec], color="#2980b9", s=80, zorder=5,
           marker="o", edgecolors="white", linewidths=0.8)

# All-SUCCESS point
ax.scatter([1.0], [base_rate_900], color="#27ae60", s=90, zorder=5,
           marker="*", label=f"All-SUCCESS (P={base_rate_900:.3f}, R=1.0)")

# All-FAILURE point — at origin, annotate separately
ax.scatter([0.0], [0.0], color="#7f8c8d", s=60, zorder=5,
           marker="x", label="All-FAILURE (P=0, R=0)")

# LLM hard-label points
for label, (prec_v, rec_v, mkr, col) in llm_points.items():
    ax.scatter([rec_v], [prec_v], color=col, s=90, zorder=5,
               marker=mkr, label=label, edgecolors="white", linewidths=0.8)
    # Wilson CI bars for GGUF (only 2 predictions — show horizontal CI on precision)
    if "GGUF" in label:
        ax.errorbar([rec_v], [prec_v],
                    yerr=[[prec_v - 0.09], [0.91 - prec_v]],
                    fmt="none", color=col, capsize=4, linewidth=1.2,
                    label="GGUF precision Wilson 95% CI [0.09, 0.91]")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — VCBench Public Validation\n"
             "(LR-TF-IDF: 900-sample; LLM points: 120-sample)",
             fontsize=11)
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.01, 1.05)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.6)

plt.tight_layout()
plt.savefig("pr_curve.pdf", bbox_inches="tight", dpi=300)
plt.savefig("pr_curve.png", bbox_inches="tight", dpi=200)
print("Saved: pr_curve.pdf  pr_curve.png")
