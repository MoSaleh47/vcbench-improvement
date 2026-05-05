import argparse
import os
import re
import time

import httpx
import pandas as pd

import run_experiments as exp


DEFAULT_MODEL = "hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_RESULTS_FILE = "results_ollama_qwen3_30b_a3b_q4_k_m_public_120.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the public VCBench prompt experiments against a local Ollama model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--dataset", default=exp.DATASET_PATH)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)
    parser.add_argument(
        "--prompts",
        default="vanilla,cot,few_shot,hybrid",
        help="Comma-separated prompt names to compare in phase 1.",
    )
    parser.add_argument(
        "--skip-temp-sweep",
        action="store_true",
        help="Run only the prompt comparison phase.",
    )
    parser.add_argument(
        "--temperatures",
        default="0.0,0.1,0.3,0.5",
        help="Comma-separated temperatures for phase 2.",
    )
    return parser.parse_args()


def normalize_base_url(base_url):
    base_url = base_url.rstrip("/")
    return base_url[:-3] if base_url.endswith("/v1") else base_url


class OllamaClient:
    def __init__(self, base_url, model):
        self.base_url = normalize_base_url(base_url)
        self.model = model

    def complete(self, prompt, temperature, max_tokens=350):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"/no_think\n{prompt}"}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["message"]["content"]


def predict(client, profile_text, prompt_template, temperature=0.0):
    prompt = prompt_template.format(profile=profile_text)
    for attempt in range(3):
        try:
            raw = client.complete(prompt, temperature, max_tokens=350).strip().upper()
            hits = re.findall(r"\b(SUCCESS|FAILURE)\b", raw)
            return hits[-1] if hits else "FAILURE"
        except Exception as exc:
            if attempt < 2:
                wait_s = 2**attempt
                print(f"  [Ollama error] {exc} - retrying in {wait_s}s")
                time.sleep(wait_s)
            else:
                print(f"  [Ollama error] {exc} - defaulting to FAILURE")
                return "FAILURE"


def main():
    args = parse_args()
    prompt_names = [p.strip() for p in args.prompts.split(",") if p.strip()]
    temperatures = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    unknown = [p for p in prompt_names if p not in exp.PROMPTS]
    if unknown:
        raise ValueError(f"Unknown prompt name(s): {', '.join(unknown)}")

    exp.MODEL = args.model
    exp.DATASET_PATH = args.dataset
    exp.SAMPLE_SIZE = args.sample_size if args.sample_size > 0 else None
    exp.RESULTS_FILE = args.results_file
    exp.predict = predict

    print("=" * 55)
    print("  VCBench - Ollama public experiment run")
    print("=" * 55)
    print(f"  Model        : {exp.MODEL}")
    print(f"  Base URL     : {normalize_base_url(args.base_url)}")
    print(f"  Results file : {exp.RESULTS_FILE}")

    df = exp.load_data()
    client = OllamaClient(args.base_url, args.model)
    print(f"\n[2/5] Ollama client ready (model: {exp.MODEL})\n")

    print("[3/5] PHASE 1 - Prompt Strategy Comparison (T=0.0)")
    print("-" * 55)
    scores1 = {}
    for prompt_name in prompt_names:
        scores1[prompt_name] = exp.run(client, df, prompt_name, 0.0)

    best_prompt = max(scores1, key=scores1.get)
    print(f"\n  Best prompt: '{best_prompt}'  F0.5={scores1[best_prompt]:.4f}")

    if args.skip_temp_sweep:
        print("\n[4/5] PHASE 2 - skipped")
    else:
        print(f"\n[4/5] PHASE 2 - Temperature Tuning (prompt='{best_prompt}')")
        print("-" * 55)
        scores2 = {}
        for temp in temperatures:
            scores2[temp] = exp.run(client, df, best_prompt, temp, f"{best_prompt} | T={temp}")

        best_temp = max(scores2, key=scores2.get)
        print(f"\n  Best temperature: T={best_temp}  F0.5={scores2[best_temp]:.4f}")

    print("\n[5/5] FINAL LEADERBOARD")
    print("=" * 55)
    res = (
        pd.read_csv(exp.RESULTS_FILE)
        .drop_duplicates("experiment")
        .sort_values("f05", ascending=False)
    )
    cols = ["experiment", "f05", "precision", "recall", "tp", "fp", "n_total"]
    print(res[cols].to_string(index=False))
    print(f"\n  Full results -> {exp.RESULTS_FILE}")


if __name__ == "__main__":
    main()
