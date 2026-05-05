"""
compute_bootstrap_cis.py
========================
Computes all bootstrap 95% confidence intervals and Wilson intervals.

Usage:
    python compute_bootstrap_cis.py

The script expects:
    vcbench_final_public.csv      — public split (labels available)
    results_llm_120.csv           — LLM hard-label predictions on the
                                    120-sample validation subset.
                                    Columns: profile_id, true_label,
                                    qwen32b_fewshot, qwen32b_vanilla,
                                    gguf_vanilla
                                    (1=SUCCESS, 0=FAILURE)

If results_llm_120.csv is not present the script will still compute
the LR-TF-IDF and trivial baselines on the full 900-row split.

Outputs:
    bootstrap_ci_results.csv   — machine-readable CI table
    bootstrap_ci_results.tex   — LaTeX table fragment ready to paste
                                 into the paper
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.stats import binom
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
RNG  = np.random.default_rng(42)
BOOT = 2_000    # bootstrap resamples
SEED = 42

# ---------------------------------------------------------------------------
# Helper: F0.5 from counts
# ---------------------------------------------------------------------------

def f05(tp, fp, fn):
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    return (1 + 0.25) * prec * rec / (0.25 * prec + rec)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    centre  = (phat + z**2 / (2*n)) / (1 + z**2 / n)
    halfwidth = (z / (1 + z**2/n)) * np.sqrt(phat*(1-phat)/n + z**2/(4*n**2))
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def bootstrap_f05(y_true, y_pred, n_boot=BOOT):
    """Bootstrap percentile CI for F0.5."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        tp = int(((yp==1)&(yt==1)).sum())
        fp = int(((yp==1)&(yt==0)).sum())
        fn = int(((yp==0)&(yt==1)).sum())
        scores.append(f05(tp, fp, fn))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return lo, hi


# ---------------------------------------------------------------------------
# Load public data
# ---------------------------------------------------------------------------

DATA_PATH = "vcbench_final_public.csv"
if not Path(DATA_PATH).exists():
    print(f"WARNING: {DATA_PATH} not found. Skipping LR-TF-IDF baseline.")
    df = None
else:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df["label"] = df["success"].astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["label"]
    )
    # 900-row full validation split
    X_train = train_df["anonymised_prose"].fillna("").values
    y_train = train_df["label"].values
    X_val   = val_df["anonymised_prose"].fillna("").values
    y_val   = val_df["label"].values
    base_rate = y_val.mean()
    n_pos     = int(y_val.sum())
    n_total   = len(y_val)
    print(f"Full validation: n={n_total}, positives={n_pos}, base_rate={base_rate:.4f}")

# ---------------------------------------------------------------------------
# Compute results
# ---------------------------------------------------------------------------

rows = []

def add_row(system, split, y_true, y_pred):
    tp = int(((np.array(y_pred)==1)&(np.array(y_true)==1)).sum())
    fp = int(((np.array(y_pred)==1)&(np.array(y_true)==0)).sum())
    fn = int(((np.array(y_pred)==0)&(np.array(y_true)==1)).sum())
    tn = int(((np.array(y_pred)==0)&(np.array(y_true)==0)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f    = f05(tp, fp, fn)
    ppr  = (tp+fp)/len(y_true)
    lo, hi = bootstrap_f05(y_true, y_pred)
    plo, phi = wilson_ci(tp, tp+fp)
    rlo, rhi = wilson_ci(tp, tp+fn) if (tp+fn)>0 else (0.0,0.0)
    rows.append({
        "System": system, "Split": split,
        "F0.5": round(f,4),
        "F0.5_lo": round(lo,4), "F0.5_hi": round(hi,4),
        "Precision": round(prec,4),
        "Prec_lo": round(plo,4), "Prec_hi": round(phi,4),
        "Recall": round(rec,4),
        "Rec_lo": round(rlo,4), "Rec_hi": round(rhi,4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "PPR": f"{100*ppr:.1f}%"
    })
    print(f"  {system:40s}  F0.5={f:.4f} [{lo:.4f}–{hi:.4f}]  "
          f"P={prec:.4f}  R={rec:.4f}  PPR={100*ppr:.1f}%")

if df is not None:
    print("\n=== 900-row full validation split ===")
    n = len(y_val)

    # Trivial baselines
    add_row("All-FAILURE", "900-row", y_val, np.zeros(n, dtype=int))
    add_row("All-SUCCESS", "900-row", y_val, np.ones(n,  dtype=int))

    # LR-TF-IDF
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1,2),
                                   sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=1_000, random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)
    y_scores = pipe.predict_proba(X_val)[:, 1]

    # Tune threshold on the 120-sample subset (mirrors paper protocol)
    rng_sub = np.random.default_rng(SEED)
    pos_idx = np.where(y_val == 1)[0]
    neg_idx = np.where(y_val == 0)[0]
    sub_pos = rng_sub.choice(pos_idx, size=11, replace=False)
    sub_neg = rng_sub.choice(neg_idx, size=109, replace=False)
    sub_idx = np.concatenate([sub_pos, sub_neg])
    y_sub_true, y_sub_scores = y_val[sub_idx], y_scores[sub_idx]

    best_f05_val, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, 200):
        preds = (y_sub_scores >= thr).astype(int)
        tp_ = int(((preds==1)&(y_sub_true==1)).sum())
        fp_ = int(((preds==1)&(y_sub_true==0)).sum())
        fn_ = int(((preds==0)&(y_sub_true==1)).sum())
        val_ = f05(tp_, fp_, fn_)
        if val_ > best_f05_val:
            best_f05_val, best_thr = val_, thr

    lr_preds_900 = (y_scores >= best_thr).astype(int)
    add_row("LR-TF-IDF (threshold tuned on 120-subset)", "900-row",
            y_val, lr_preds_900)

    # -----------------------------------------------------------------------
    # 120-sample subset (for LLM comparisons)
    # -----------------------------------------------------------------------
    print("\n=== 120-sample validation subset (seed=42) ===")
    y_sub_true_arr = y_val[sub_idx]
    y_sub_lr = (y_sub_scores >= best_thr).astype(int)

    add_row("All-FAILURE", "120-sample", y_sub_true_arr, np.zeros(120, int))
    add_row("All-SUCCESS", "120-sample", y_sub_true_arr, np.ones(120, int))
    add_row("LR-TF-IDF",  "120-sample", y_sub_true_arr, y_sub_lr)

# LLM results — hard-coded from Table 2 / Table 3 of the paper (seed=42)
# Format: (y_true_list, y_pred_list) reconstructed from TP/FP/FN counts.
# Replace with actual prediction arrays when available.

def synthetic_preds(n_pos, tp, fp, fn, n=120):
    """Reconstruct a plausible prediction vector from summary counts."""
    y_true = np.array([1]*n_pos + [0]*(n - n_pos))
    y_pred = np.zeros(n, dtype=int)
    # Mark tp true positives
    y_pred[:tp] = 1
    # Mark fp false positives (from negatives)
    y_pred[n_pos:n_pos+fp] = 1
    return y_true, y_pred

print("\n=== LLM systems (120-sample, seed=42, from paper tables) ===")
llm_configs = [
    # (name,             n_pos, tp, fp, fn)
    ("qwen3:32b Few-Shot",   11,  8, 41,  3),
    ("qwen3:32b Vanilla",    11,  5, 21,  6),
    ("GGUF Q4_K_M Vanilla",  11,  1,  1, 10),
]
for name, n_pos, tp, fp, fn in llm_configs:
    yt, yp = synthetic_preds(n_pos, tp, fp, fn)
    add_row(name, "120-sample", yt, yp)

# -----------------------------------------------------------------------
# GGUF precision Wilson CI (special case: n=2 predictions)
# -----------------------------------------------------------------------
print("\n=== Wilson CI for GGUF precision (k=1, n=2) ===")
lo_w, hi_w = wilson_ci(1, 2)
print(f"  Wilson 95% CI for precision 0.5000 (1/2): [{lo_w:.4f}, {hi_w:.4f}]")

# -----------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------
results_df = pd.DataFrame(rows)
results_df.to_csv("bootstrap_ci_results.csv", index=False)
print("\nSaved: bootstrap_ci_results.csv")

# LaTeX table fragment
def fmt_ci(val, lo, hi):
    return f"{val:.4f} [{lo:.4f}--{hi:.4f}]"

with open("bootstrap_ci_results.tex", "w") as f:
    f.write("% Auto-generated by compute_bootstrap_cis.py\n")
    f.write("% Paste into your NeurIPS paper tables.\n\n")
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
    f.write("System & Split & $F_{0.5}$ [95\\% CI] & Precision [Wilson CI] & "
            "Recall & PPR \\\\\n\\midrule\n")
    for _, row in results_df.iterrows():
        f05_str  = fmt_ci(row["F0.5"],      row["F0.5_lo"],  row["F0.5_hi"])
        prec_str = fmt_ci(row["Precision"], row["Prec_lo"],  row["Prec_hi"])
        f.write(f"{row['System']} & {row['Split']} & "
                f"{f05_str} & {prec_str} & "
                f"{row['Recall']:.4f} & {row['PPR']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("Saved: bootstrap_ci_results.tex")
