import os, re, time
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

load_dotenv()

MODEL            = "qwen/qwen3-32b"
RESULTS_FILE     = "results_private_test.csv"
DATASET_PATH     = "vcbench_final_private.csv"
SAMPLE_SIZE      = 200
SEED             = 42

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

def load_private_test():
    print("Loading private test set...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Total rows   : {len(df)}")
    print(f"  Columns      : {list(df.columns)}")

    # Check if labels exist
    if "success" not in df.columns:
        print("  ⚠️  No 'success' column found — blind test set (no labels)")
        has_labels = False
    else:
        df["success"] = df["success"].astype(int)
        print(f"  Success rate : {df['success'].mean():.1%}")
        has_labels = True

    df = df.dropna(subset=["anonymised_prose"])

    # Stratified sample if labels exist, otherwise random
    if has_labels and SAMPLE_SIZE < len(df):
        n_success = int(round(SAMPLE_SIZE * df["success"].mean()))
        n_failure = SAMPLE_SIZE - n_success
        success_rows = df[df["success"] == 1].sample(n=n_success, random_state=SEED)
        failure_rows = df[df["success"] == 0].sample(n=n_failure, random_state=SEED)
        df = pd.concat([success_rows, failure_rows]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"  Sampled      : {len(df)} profiles ({df['success'].sum()} successes)")
    elif SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
        print(f"  Sampled      : {len(df)} profiles (no labels)")

    return df, has_labels

def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Add GROQ_API_KEY to your .env!")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

def predict(client, profile_text):
    prompt = BEST_PROMPT.format(profile=profile_text)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                extra_body={"thinking": False} 
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
    print("  VCBench — PRIVATE TEST SET EVALUATION")
    print("=" * 55)

    df, has_labels = load_private_test()
    client = get_client()

    preds = [
        predict(client, str(row["anonymised_prose"]))
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Private test")
    ]

    # Save raw predictions always
    df["prediction"] = preds
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n  ✓ Predictions saved → {RESULTS_FILE}")

    if has_labels:
        y_true = df["success"].tolist()
        yp     = [1 if p == "SUCCESS" else 0 for p in preds]
        p  = precision_score(y_true, yp, zero_division=0)
        r  = recall_score(y_true, yp, zero_division=0)
        f  = fbeta_score(y_true, yp, beta=0.5, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()

        print("\n" + "=" * 55)
        print("  PRIVATE TEST SET RESULTS")
        print("=" * 55)
        print(f"  F0.5       : {f:.4f}")
        print(f"  Precision  : {p:.4f}")
        print(f"  Recall     : {r:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Pred SUCCESS: {tp+fp} / {len(df)} ({(tp+fp)/len(df):.1%})")
        print(f"\n  ─── Paper baselines ───")
        print(f"  GPT-4o     : 0.2510")
        print(f"  DeepSeek-V3: 0.1180")
        print(f"  Your model : {f:.4f}")
    else:
        n_success_pred = sum(1 for p in preds if p == "SUCCESS")
        print(f"\n  Blind test — no labels available")
        print(f"  Predicted SUCCESS: {n_success_pred} / {len(df)} ({n_success_pred/len(df):.1%})")
        print(f"  Predictions saved to {RESULTS_FILE}")