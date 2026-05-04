import os, re, time
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

load_dotenv()

MODEL        = "qwen/qwen3-32b"
SAMPLE_SIZE  = 120
RESULTS_FILE = "results_test.csv"
SEED_SPLIT   = 42          # SAME seed → same 80/20 split as before
SEED_SAMPLE  = 99          # DIFFERENT seed → fresh sample from train portion
DATASET_PATH = "vcbench_final_public.csv"

# Best config from validation
BEST_PROMPT_NAME = "few_shot"
BEST_TEMPERATURE = 0.0

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

BEST_PROMPT = f"""You are a venture capital analyst. Study the examples, then predict the new founder.

SUCCESS = raised >$500M OR IPO/acquisition >$500M. Base rate: ~9%.

{EX}

Now predict this founder:
{{profile}}

Reply with ONE word only — SUCCESS or FAILURE:"""

def load_test_data():
    print("Loading VCBench dataset...")
    df = pd.read_csv(DATASET_PATH)
    df["success"] = df["success"].astype(int)
    df = df[["anonymised_prose", "success", "industry"]].dropna(subset=["anonymised_prose"])

    # SAME split as experiments — get the 80% train portion
    train, _ = train_test_split(df, test_size=0.2, random_state=SEED_SPLIT, stratify=df["success"])

    # Sample 120 from train with a DIFFERENT seed (never seen during prompt tuning)
    n_success = int(round(SAMPLE_SIZE * train["success"].mean()))
    n_failure = SAMPLE_SIZE - n_success

    success_rows = train[train["success"] == 1].sample(n=n_success, random_state=SEED_SAMPLE)
    failure_rows = train[train["success"] == 0].sample(n=n_failure, random_state=SEED_SAMPLE)
    test = pd.concat([success_rows, failure_rows]).sample(frac=1, random_state=SEED_SAMPLE).reset_index(drop=True)

    print(f"Test set: {len(test)} profiles ({test['success'].sum()} successes, {(test['success']==0).sum()} failures)")
    return test

def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Add GROQ_API_KEY to your .env!")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

def predict(client, profile_text, temperature=0.0):
    prompt = BEST_PROMPT.format(profile=profile_text)
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
            if "429" in str(e):
                print(f"\n  [Rate limit] Waiting 60s...")
                time.sleep(60)
            elif attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return "FAILURE"

if __name__ == "__main__":
    print("=" * 55)
    print("  VCBench — FINAL TEST SET EVALUATION")
    print(f"  Config: {BEST_PROMPT_NAME} | T={BEST_TEMPERATURE}")
    print("=" * 55)

    df_test = load_test_data()
    client  = get_client()

    preds = [
        predict(client, str(row["anonymised_prose"]), BEST_TEMPERATURE)
        for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Evaluating on test set")
    ]

    y_true = df_test["success"].tolist()
    yp = [1 if p == "SUCCESS" else 0 for p in preds]
    p  = precision_score(y_true, yp, zero_division=0)
    r  = recall_score(y_true, yp, zero_division=0)
    f  = fbeta_score(y_true, yp, beta=0.5, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()

    print("\n" + "=" * 55)
    print("  TEST SET RESULTS")
    print("=" * 55)
    print(f"  F0.5      : {f:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Predicted SUCCESS: {tp+fp} / {len(df_test)}")

    print("\n  ─── Comparison ───")
    print(f"  Validation F0.5    : 0.1188  (few_shot T=0.0)")
    print(f"  Test F0.5          : {f:.4f}  ← this goes in the paper")
    print(f"  DeepSeek-V3 (paper): 0.1180")
    print(f"  GPT-4o (paper)     : 0.2510")

    # Save
    pd.DataFrame([{
        "split": "test", "prompt": BEST_PROMPT_NAME, "temperature": BEST_TEMPERATURE,
        "f05": round(f,4), "precision": round(p,4), "recall": round(r,4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_total": len(df_test)
    }]).to_csv(RESULTS_FILE, index=False)
    print(f"\n  ✓ Saved → {RESULTS_FILE}")