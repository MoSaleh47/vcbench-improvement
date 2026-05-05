import os, csv, time, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

load_dotenv()

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
DATASET_PATH = "vcbench_final_public.csv"   # full 4500 rows
SAMPLE_SIZE  = 120
RESULTS_FILE = "results.csv"
MODEL        = "qwen/qwen3-32b"
SEED         = 42

# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
def load_data():
    print("\n[1/5] Loading VCBench dataset...")
    df = pd.read_csv(DATASET_PATH)

    print(f"  Total rows   : {len(df)}")
    print(f"  Success rate : {df['success'].mean():.1%}")
    print(f"  Columns      : {list(df.columns)}")

    # Use anonymised_prose as input text, success as label
    df = df[["anonymised_prose", "success", "industry"]].copy()
    df["success"] = df["success"].astype(int)
    df = df.dropna(subset=["anonymised_prose"])

    # Stratified val split (mimics the paper's evaluation setup)
    _, val = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["success"]
    )

    if SAMPLE_SIZE and SAMPLE_SIZE < len(val):
        # Keep class balance in the sample without relying on pandas groupby.apply
        # preserving grouping columns.
        sampled = []
        for _, group in val.groupby("success"):
            n = int(round(SAMPLE_SIZE * len(group) / len(val)))
            if n:
                sampled.append(group.sample(n=min(n, len(group)), random_state=SEED))
        val = pd.concat(sampled).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Using        : {len(val)} profiles ({val['success'].sum()} successes, {(val['success']==0).sum()} failures)")
    return val.reset_index(drop=True)

# ══════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════
EX = """EXAMPLE 1 — FAILURE:
"This founder leads a startup in the Healthcare Technology industry.
Education: * BS in Biology (Institution QS rank 350)
Professional experience: * Research Assistant for 2-3 years * Sales Rep for 1-2 years * Co-Founder for 3-4 years (11-50 employees). No prior exits."
→ FAILURE

EXAMPLE 2 — SUCCESS:
"This founder leads a startup in the Software Development industry.
Education: * MS in Computer Science (Institution QS rank 8)
Professional experience: * Software Engineer for 3-4 years (10001+ employees) * VP Engineering for 2-3 years (1001-5000 employees) * Co-Founder, CEO for 5-6 years. One prior acquisition: $50M-$150M."
→ SUCCESS"""

PROMPTS = {
    "vanilla": """You are a venture capital analyst. Predict whether this founder will be a major success.

SUCCESS = company raised >$500M, OR had an IPO/acquisition >$500M within 8 years of founding.
FAILURE = did not achieve the above.
Important: only ~9% of founders succeed. Be conservative and precise.

Founder Profile:
{profile}

Reply with ONE word only — SUCCESS or FAILURE:""",

    "cot": """You are a venture capital analyst. Think carefully step by step, then predict.

SUCCESS = raised >$500M OR IPO/acquisition >$500M within 8 years.
Only ~9% of founders succeed. False positives (backing failures) are very costly.

Founder Profile:
{profile}

Step 1 - Positive signals (strong education, big-company experience, prior exits, domain fit):
Step 2 - Risk factors (weak track record, no exits, wrong industry, short tenures):
Step 3 - Base rate check: Given only 9% succeed, is the evidence here strong enough to call SUCCESS?
Final answer (SUCCESS or FAILURE):""",

    "few_shot": f"""You are a venture capital analyst. Study the examples, then predict the new founder.

SUCCESS = raised >$500M OR IPO/acquisition >$500M. Base rate: ~9%.

{EX}

Now predict this founder:
{{profile}}

Reply with ONE word only — SUCCESS or FAILURE:""",

    "hybrid": f"""You are a venture capital analyst. Study the examples, then reason step by step before predicting.

SUCCESS = raised >$500M OR IPO/acquisition >$500M within 8 years.
Base rate: ~9%. Precision matters — don't over-predict success.

{EX}

Now analyze this founder:
{{profile}}

Step 1 - Positive signals:
Step 2 - Risk factors:
Step 3 - Base rate check (9% threshold):
Final answer (SUCCESS or FAILURE):"""
}

# ══════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════
def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Add GROQ_API_KEY to your .env file! Get it free at console.groq.com")
    return OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1"
    )

def predict(client, profile_text, prompt_template, temperature=0.0):
    prompt = prompt_template.format(profile=profile_text)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=350,
            )
            raw = r.choices[0].message.content.strip().upper()
            hits = re.findall(r'\b(SUCCESS|FAILURE)\b', raw)
            return hits[-1] if hits else "FAILURE"
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [API error] {e} — defaulting to FAILURE")
                return "FAILURE"

# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════
def score(y_true, preds):
    yp = [1 if p == "SUCCESS" else 0 for p in preds]
    p  = precision_score(y_true, yp, zero_division=0)
    r  = recall_score(y_true, yp, zero_division=0)
    f  = fbeta_score(y_true, yp, beta=0.5, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
    return dict(f05=round(f,4), precision=round(p,4), recall=round(r,4),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                n_predicted_success=int(tp+fp))

def log(label, prompt, temp, m, n):
    row = {"time": datetime.now().strftime("%H:%M"), "experiment": label,
           "prompt": prompt, "temperature": temp, **m, "n_total": n}
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)

def run(client, df, prompt_name, temp, label=None):
    label = label or f"{prompt_name} | T={temp}"
    preds = [
        predict(client, str(row["anonymised_prose"]), PROMPTS[prompt_name], temp)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=label, leave=False)
    ]
    m = score(df["success"].tolist(), preds)
    print(f"  {label:40s}  F0.5={m['f05']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  TP={m['tp']} FP={m['fp']}")
    log(label, prompt_name, temp, m, len(df))
    return m["f05"]

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  VCBench — Samir Adam Mahamat Saleh")
    print("=" * 55)

    df     = load_data()
    client = get_client()
    print(f"\n[2/5] DeepSeek client ready (model: {MODEL})\n")

    # ── Phase 1: Compare 4 prompts at T=0.0 ───────────────
    print("[3/5] PHASE 1 — Prompt Strategy Comparison (T=0.0)")
    print("-" * 55)
    scores1 = {}
    for p in ["vanilla", "cot", "few_shot", "hybrid"]:
        scores1[p] = run(client, df, p, 0.0)

    best_prompt = max(scores1, key=scores1.get)
    print(f"\n  ★ Best prompt: '{best_prompt}'  F0.5={scores1[best_prompt]:.4f}")

    # ── Phase 2: Temperature sweep with best prompt ────────
    print(f"\n[4/5] PHASE 2 — Temperature Tuning (prompt='{best_prompt}')")
    print("-" * 55)
    scores2 = {}
    for t in [0.0, 0.1, 0.3, 0.5]:
        scores2[t] = run(client, df, best_prompt, t, f"{best_prompt} | T={t}")

    best_temp = max(scores2, key=scores2.get)
    print(f"\n  ★ Best temperature: T={best_temp}  F0.5={scores2[best_temp]:.4f}")

    # ── Final leaderboard ──────────────────────────────────
    print("\n[5/5] FINAL LEADERBOARD")
    print("=" * 55)
    res = (pd.read_csv(RESULTS_FILE)
           .drop_duplicates("experiment")
           .sort_values("f05", ascending=False))
    print(res[["experiment","f05","precision","recall","tp","fp","n_total"]].to_string(index=False))

    print("\n  ─── Paper baselines ───")
    print("  GPT-4o (paper)        F0.5 = 0.251")
    print("  DeepSeek-V3 (paper)   F0.5 = 0.118")
    print("  Human VCs (Tier-1)    Precision ≈ 5.5%")

    best = res.iloc[0]
    print(f"\n  🏆 THE BEST: {best['experiment']}")
    print(f"     F0.5={best['f05']:.4f}  Precision={best['precision']:.4f}  Recall={best['recall']:.4f}")
    print(f"\n  ✓ Full results → {RESULTS_FILE}")
