import os, csv, time, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

load_dotenv()

SAMPLE_SIZE  = 200        # start with 200. Change to None for the full dataset later
RESULTS_FILE = "results.csv"
MODEL        = "deepseek-chat"
SEED         = 42

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
def load_data():
    print("\n[1/5] Loading VCBench dataset...")
    ds  = load_dataset("cloudcatcher2/VCBench", split="train")
    df  = ds.to_pandas()
    print(f"  Columns: {list(df.columns)}")
    print(f"  Total: {len(df)} profiles")

    # Auto-detect text column
    text_col = next((c for c in ["prose","text","profile","description"] if c in df.columns), None)
    if text_col is None:
        str_cols = [c for c in df.columns if df[c].dtype == object and c != "success"]
        df["prose"] = df[str_cols].astype(str).agg(" | ".join, axis=1)
        text_col = "prose"

    # Auto-detect label column
    label_col = next((c for c in ["success","label","outcome"] if c in df.columns), None)

    df = df.rename(columns={text_col: "prose", label_col: "success"})
    df["success"] = df["success"].astype(int)
    print(f"  Success rate: {df['success'].mean():.1%}")

    _, val = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["success"])
    if SAMPLE_SIZE and SAMPLE_SIZE < len(val):
        val = val.sample(SAMPLE_SIZE, random_state=SEED)
    print(f"  Using {len(val)} profiles ({val['success'].sum()} successes)")
    return val.reset_index(drop=True)

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
EX = """EXAMPLE 1 — FAILURE:
"Healthcare startup. BS Biology QS rank 350. Research Assistant 2yr, Sales Rep 1yr, Co-Founder 3yr (11-50 employees). No prior exits."
→ FAILURE

EXAMPLE 2 — SUCCESS:
"Software startup. MS Computer Science QS rank 8. Software Engineer 3yr at large tech (>10k employees), VP Engineering 2yr (1k-5k employees), Co-Founder CEO 5yr. One acquisition: $50M-$150M."
→ SUCCESS"""

PROMPTS = {
"vanilla": """You are a venture capital analyst. Predict founder success.
SUCCESS = raised >$500M OR IPO/acquisition >$500M within 8 years.
Only ~9% succeed. Be conservative.
Profile: {profile}
Reply with ONE word: SUCCESS or FAILURE""",

"cot": """You are a venture capital analyst. Think step by step.
SUCCESS = raised >$500M OR IPO/acquisition >$500M. Only ~9% succeed.
Profile: {profile}
Step 1 - Positive signals:
Step 2 - Risk factors:
Step 3 - Given 9% base rate, is evidence strong enough?
Final answer (SUCCESS or FAILURE):""",

"few_shot": f"""You are a venture capital analyst. Learn from examples then predict.
SUCCESS = raised >$500M OR IPO/acquisition >$500M. Base rate ~9%.
{EX}
New profile: {{profile}}
ONE word only: SUCCESS or FAILURE""",

"hybrid": f"""You are a venture capital analyst. Learn from examples then reason step by step.
SUCCESS = raised >$500M OR IPO/acquisition >$500M. Base rate ~9%.
{EX}
New profile: {{profile}}
Step 1 - Positive signals:
Step 2 - Risk factors:
Step 3 - Base rate check:
Final answer (SUCCESS or FAILURE):"""
}

# ─── API ──────────────────────────────────────────────────────────────────────
def get_client():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key or key == "paste_your_key_here":
        raise ValueError("Add your DeepSeek API key to .env! Get it free at platform.deepseek.com")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")

def predict(client, profile, template, temp=0.0):
    prompt = template.format(profile=profile)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=temp, max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = r.choices[0].message.content.strip().upper()
            hits = re.findall(r'\b(SUCCESS|FAILURE)\b', raw)
            return hits[-1] if hits else "FAILURE"
        except:
            time.sleep(2 ** attempt)
    return "FAILURE"

# ─── METRICS ──────────────────────────────────────────────────────────────────
def score(y_true, preds):
    yp = [1 if p == "SUCCESS" else 0 for p in preds]
    p = precision_score(y_true, yp, zero_division=0)
    r = recall_score(y_true, yp, zero_division=0)
    f = fbeta_score(y_true, yp, beta=0.5, zero_division=0)
    tn,fp,fn,tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
    return dict(f05=round(f,4), precision=round(p,4), recall=round(r,4),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def log_result(label, prompt, temp, m, n):
    row = {"time": datetime.now().strftime("%H:%M"), "experiment": label,
           "prompt": prompt, "temperature": temp, **m, "n_total": n}
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)

def run(client, df, prompt_name, temp, label=None):
    label = label or f"{prompt_name} T={temp}"
    preds = [predict(client, str(r["prose"]), PROMPTS[prompt_name], temp)
             for _, r in tqdm(df.iterrows(), total=len(df), desc=label, leave=False)]
    m = score(df["success"].tolist(), preds)
    print(f"  {label:35s} → F0.5={m['f05']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  TP={m['tp']} FP={m['fp']}")
    log_result(label, prompt_name, temp, m, len(df))
    return m["f05"]

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  VCBench — Samir Adam Mahamat Saleh")
    print("="*55)

    df     = load_data()
    client = get_client()

    # Phase 1 — Compare all 4 prompts at T=0.0
    print("\n[3/5] PHASE 1: Prompt Strategy Comparison (T=0.0)")
    print("-"*55)
    scores1 = {p: run(client, df, p, 0.0) for p in ["vanilla","cot","few_shot","hybrid"]}
    best_prompt = max(scores1, key=scores1.get)
    print(f"\n  ★ Best prompt: '{best_prompt}'  F0.5={scores1[best_prompt]:.4f}")

    # Phase 2 — Temperature sweep with best prompt
    print(f"\n[4/5] PHASE 2: Temperature Tuning (prompt='{best_prompt}')")
    print("-"*55)
    scores2 = {t: run(client, df, best_prompt, t, f"{best_prompt} T={t}") for t in [0.0,0.1,0.3,0.5]}
    best_temp = max(scores2, key=scores2.get)
    print(f"\n  ★ Best temp: T={best_temp}  F0.5={scores2[best_temp]:.4f}")

    # Final leaderboard
    print("\n[5/5] FINAL LEADERBOARD")
    print("="*55)
    res = pd.read_csv(RESULTS_FILE).drop_duplicates("experiment").sort_values("f05", ascending=False)
    print(res[["experiment","f05","precision","recall","tp","fp"]].to_string(index=False))
    print("\n  Paper baselines:")
    print("  GPT-4o (paper)        F0.5 = 0.251")
    print("  DeepSeek-V3 (paper)   F0.5 = 0.118")
    print(f"\n  YOUR BEST: {res.iloc[0]['experiment']}")
    print(f"  F0.5 = {res.iloc[0]['f05']:.4f}")
    print(f"\n  ✓ Saved → {RESULTS_FILE}  (open this for your paper!)")