import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from tqdm import tqdm

from test_real_data import BEST_PROMPT, DATASET_PATH, SEED


DEFAULT_MODEL = "hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_RESULTS_FILE = "results_ollama_qwen3_30b_a3b_q4_k_m_private_full.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the private VCBench data through a local Ollama model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of rows to sample. Use 0 for the full dataset.",
    )
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def normalize_base_url(base_url):
    base_url = base_url.rstrip("/")
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def unique_output_path(path):
    output = Path(path)
    if not output.exists():
        return output
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output.with_name(f"{output.stem}_{stamp}{output.suffix}")


def load_private_test(dataset_path, sample_size, seed):
    print("Loading private test set...")
    df = pd.read_csv(dataset_path)
    print(f"  Total rows   : {len(df)}")
    print(f"  Columns      : {list(df.columns)}")

    has_labels = "success" in df.columns
    if has_labels:
        df["success"] = df["success"].astype(int)
        print(f"  Success rate : {df['success'].mean():.1%}")
    else:
        print("  No 'success' column found - blind test set")

    df = df.dropna(subset=["anonymised_prose"]).reset_index(drop=True)

    if sample_size and sample_size < len(df):
        if has_labels:
            n_success = int(round(sample_size * df["success"].mean()))
            n_failure = sample_size - n_success
            success_rows = df[df["success"] == 1].sample(n=n_success, random_state=seed)
            failure_rows = df[df["success"] == 0].sample(n=n_failure, random_state=seed)
            df = (
                pd.concat([success_rows, failure_rows])
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )
            print(f"  Sampled      : {len(df)} profiles ({df['success'].sum()} successes)")
        else:
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
            print(f"  Sampled      : {len(df)} profiles (no labels)")
    else:
        print(f"  Using        : {len(df)} profiles")

    return df, has_labels


def predict(base_url, model, profile_text):
    prompt = BEST_PROMPT.format(profile=profile_text)
    for attempt in range(3):
        try:
            response = httpx.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"/no_think\n{prompt}"}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 100},
                },
                timeout=300,
            )
            response.raise_for_status()
            raw = response.json()["message"]["content"].strip().upper()
            hits = re.findall(r"\b(SUCCESS|FAILURE)\b", raw)
            return hits[-1] if hits else "FAILURE"
        except Exception as exc:
            if attempt < 2:
                wait_s = 2**attempt
                print(f"\n  [Ollama error] {exc} - retrying in {wait_s}s")
                time.sleep(wait_s)
            else:
                print(f"\n  [Ollama error] {exc} - defaulting to FAILURE")
                return "FAILURE"


def print_metrics(df, preds):
    y_true = df["success"].tolist()
    yp = [1 if pred == "SUCCESS" else 0 for pred in preds]
    precision = precision_score(y_true, yp, zero_division=0)
    recall = recall_score(y_true, yp, zero_division=0)
    f05 = fbeta_score(y_true, yp, beta=0.5, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()

    print("\n" + "=" * 55)
    print("  PRIVATE TEST SET RESULTS")
    print("=" * 55)
    print(f"  F0.5       : {f05:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Pred SUCCESS: {tp + fp} / {len(df)} ({(tp + fp) / len(df):.1%})")


def main():
    args = parse_args()
    output_path = unique_output_path(args.results_file)
    base_url = normalize_base_url(args.base_url)

    print("=" * 55)
    print("  VCBench - Ollama private test set")
    print("=" * 55)
    print(f"  Model        : {args.model}")
    print(f"  Base URL     : {base_url}")
    print(f"  Results file : {output_path}")

    df, has_labels = load_private_test(args.dataset, args.sample_size, args.seed)

    preds = [
        predict(base_url, args.model, str(row["anonymised_prose"]))
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Private test")
    ]

    df["prediction"] = preds
    df["model"] = args.model
    df.to_csv(output_path, index=False)
    print(f"\n  Predictions saved -> {output_path}")

    if has_labels:
        print_metrics(df, preds)
    else:
        n_success_pred = sum(1 for pred in preds if pred == "SUCCESS")
        print("\n  Blind test - no labels available")
        print(f"  Predicted SUCCESS: {n_success_pred} / {len(df)} ({n_success_pred / len(df):.1%})")


if __name__ == "__main__":
    main()
