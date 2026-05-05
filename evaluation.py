"""Evaluate legacy JSON-style VCBench prediction files.

The current final analysis uses the dedicated result CSVs and bootstrap script.
This helper remains for older outputs produced under vanilla_llm_testing_results/.
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score


def safe_eval(prediction_json):
    if pd.isna(prediction_json):
        return None

    text = str(prediction_json).strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return json.loads(text)


def evaluate_file(path: Path) -> dict:
    df = pd.read_csv(path)
    predictions = []

    for _, row in df.iterrows():
        parsed = safe_eval(row["prediction"])
        if not parsed:
            prediction = "no"
        else:
            prediction = str(parsed.get("prediction", "no"))

        if prediction.lower() == "yes":
            predictions.append(1)
        elif prediction.lower() == "no":
            predictions.append(0)
        else:
            raise ValueError(f"Invalid prediction in {path.name}: {prediction}")

    y_true = df["success"].astype(int)
    return {
        "file": path.name,
        "n": len(df),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f05": fbeta_score(y_true, predictions, beta=0.5, zero_division=0),
        "pred_success": int(sum(predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="vanilla_llm_testing_results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"No legacy result directory found: {results_dir}")
        return

    rows = [evaluate_file(path) for path in sorted(results_dir.glob("*.csv"))]
    if not rows:
        print(f"No CSV files found in {results_dir}")
        return

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
