"""Quick dataset and result summaries for the VCBench project."""

from pathlib import Path

import pandas as pd


DATASETS = [
    "vcbench_final_public.csv",
    "vcbench_public.csv",
    "vcbench_final_public_sample100.csv",
    "vcbench_final_private.csv",
]

RESULTS = [
    "results.csv",
    "results_test.csv",
    "results_ollama_qwen3_32b_public_120.csv",
    "results_ollama_qwen3_30b_a3b_q4_k_m_public_120.csv",
    "results_ollama_qwen3_30b_a3b_q4_k_m_private_full.csv",
]


def summarize_csv(path: Path) -> None:
    if not path.exists():
        print(f"- {path.name}: missing")
        return

    df = pd.read_csv(path)
    print(f"- {path.name}: {len(df):,} rows, {len(df.columns)} columns")

    if "success" in df.columns:
        labels = df["success"].dropna().astype(int)
        if not labels.empty:
            print(f"  success rate: {labels.mean():.2%} ({labels.sum():,}/{len(labels):,})")

    if "prediction" in df.columns:
        counts = df["prediction"].value_counts(dropna=False)
        print("  predictions:")
        for label, count in counts.items():
            print(f"    {label}: {count:,} ({count / len(df):.1%})")

    if "anonymised_prose" in df.columns:
        prose = df["anonymised_prose"].dropna().astype(str)
        if not prose.empty:
            lengths = prose.str.len()
            print(f"  prose length mean/median: {lengths.mean():.0f}/{lengths.median():.0f}")


def main() -> None:
    print("VCBench data files")
    print("==================")
    for name in DATASETS:
        summarize_csv(Path(name))

    print("\nVCBench result files")
    print("====================")
    for name in RESULTS:
        summarize_csv(Path(name))


if __name__ == "__main__":
    main()
