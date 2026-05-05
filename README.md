# VCBench Local LLM Calibration Study

This repository is the final project package for a VCBench experiment on local
LLMs, rare-event prediction, and calibration failure modes in venture-capital
screening.

VCBench asks a model to read an anonymized founder profile and predict one
binary label:

- `SUCCESS`: the founder's company raised more than `$500M`, or had an IPO or
  acquisition above `$500M` within eight years.
- `FAILURE`: the company did not reach one of those outcomes.

The public split has an approximately 9% success rate. Because the task is
highly imbalanced, the final analysis treats `F0.5` as necessary but not
sufficient: every model result should be read together with precision, recall,
confusion-matrix counts, predicted-positive rate, trivial baselines, and
uncertainty intervals.

## Final Takeaway

The strongest result is not "prompt engineering solves VCBench." The defensible
claim is:

> Local open-weight LLMs can be pushed from severe overprediction toward more
> conservative behavior, but small-sample `F0.5` gains are unstable unless they
> are interpreted with calibration diagnostics and baselines.

The final project is best positioned as a workshop-style negative result and
evaluation-methodology study, not as a NeurIPS main-track performance claim.

## Repository Layout

```text
.
|-- README.md
|-- PAPER_DRAFT_NEURIPS.md
|-- NEURIPS_REVIEW.md
|-- requirements.txt
|-- .env.example
|-- run_experiments.py
|-- run_experiments_ollama.py
|-- test_real_data.py
|-- test_real_data_ollama.py
|-- compute_bootstrap_cis.py
|-- generate_pr_curve.py
|-- data.py
|-- evaluation.py
|-- bootstrap_ci_results.csv
|-- bootstrap_ci_results.tex
|-- pr_curve.pdf
|-- pr_curve.png
|-- vcbench_final_public.csv
|-- vcbench_public.csv
|-- vcbench_final_public_sample100.csv
|-- results.csv
|-- results_test.csv
|-- results_ollama_qwen3_32b_public_120.csv
|-- results_ollama_qwen3_30b_a3b_q4_k_m_public_120.csv
|-- core/
|-- llms/
```

The `neurips_latex/` directory is ignored and intentionally excluded from the
final repository package. The final paper text is kept in
`PAPER_DRAFT_NEURIPS.md`.

Private data, full private prediction files, virtual environments, caches, and
local secrets are ignored by git.

## Setup

Create and activate a local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For hosted API scripts, copy `.env.example` to `.env` and fill the relevant
key. The local Ollama scripts do not require an API key.

```text
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## Ollama Model

The final local model used for the latest Ollama runs is:

```text
hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
```

Pull it with:

```powershell
ollama pull hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
```

The local scripts use `http://127.0.0.1:11434`, which was more reliable than
`localhost` in the Windows environment used for this project.

For Qwen reasoning models, the Ollama wrappers force no-thinking mode with both
`/no_think` and the JSON option `think: false`.

## Main Scripts

- `run_experiments.py`: original hosted Groq/OpenAI-compatible public
  validation runner. It defines the canonical prompt templates.
- `run_experiments_ollama.py`: local Ollama public validation runner. It imports
  the same prompt templates from `run_experiments.py`.
- `test_real_data.py`: original hosted private/blind prediction runner.
- `test_real_data_ollama.py`: local Ollama private/blind prediction runner. It
  imports the same `BEST_PROMPT` from `test_real_data.py`.
- `compute_bootstrap_cis.py`: computes trivial baselines, LR-TF-IDF baselines,
  bootstrap `F0.5` intervals, and Wilson intervals.
- `generate_pr_curve.py`: generates the final precision-recall figure.
- `data.py`: prints quick dataset/result summaries.
- `evaluation.py`: evaluates legacy JSON-style outputs from
  `vanilla_llm_testing_results/` when that directory exists.

## Reproduce Final Analysis Artifacts

Recompute the bootstrap and Wilson interval table:

```powershell
.\.venv\Scripts\python.exe compute_bootstrap_cis.py
```

Outputs:

- `bootstrap_ci_results.csv`
- `bootstrap_ci_results.tex`

Regenerate the precision-recall figure:

```powershell
.\.venv\Scripts\python.exe generate_pr_curve.py
```

Outputs:

- `pr_curve.pdf`
- `pr_curve.png`

## Run Local Public Validation

Fast public validation run with the current local model and only the Vanilla
prompt:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts vanilla --skip-temp-sweep
```

Run the private-script prompt on the public sample:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts few_shot --skip-temp-sweep
```

Full local prompt comparison:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts vanilla,cot,few_shot,hybrid --skip-temp-sweep
```

`cot` and `hybrid` can be very slow on large local Qwen models.

## Run Private/Blind Predictions

The full private/blind CSV is intentionally ignored by git. If it exists locally,
run:

```powershell
.\.venv\Scripts\python.exe test_real_data_ollama.py --sample-size 0
```

The private file used in this workspace had 4,500 rows and no `success` column,
so the output is a prediction file rather than a scored benchmark result.

## Final Results

### Public 120-Sample LLM Runs

| Model | Prompt | T | F0.5 | Precision | Recall | TP | FP | Pred+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3:32b` | Few-Shot | 0.0 | 0.1932 | 0.1633 | 0.7273 | 8 | 41 | 40.8% |
| `qwen3:32b` | Vanilla | 0.0 | 0.2174 | 0.1923 | 0.4545 | 5 | 21 | 21.7% |
| Qwen3-30B-A3B GGUF | Vanilla | 0.0 | 0.2632 | 0.5000 | 0.0909 | 1 | 1 | 1.7% |

The GGUF Vanilla result has the best observed `F0.5` among the local LLM rows,
but it is based on only two positive predictions. Its Wilson 95% precision
interval is wide: `0.0945` to `0.9055`.

### Baselines and Confidence Intervals

From `bootstrap_ci_results.csv`:

| System | Split | F0.5 [95% CI] | Precision [Wilson CI] | Recall | PPR |
| --- | --- | ---: | ---: | ---: | ---: |
| All-FAILURE | 120-sample | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 1.0000] | 0.0000 | 0.0% |
| All-SUCCESS | 120-sample | 0.1120 [0.0515, 0.1807] | 0.0917 [0.0520, 0.1567] | 1.0000 | 100.0% |
| LR-TF-IDF | 120-sample | 0.7143 [0.3125, 0.9091] | 0.8333 [0.4365, 0.9699] | 0.4545 | 5.0% |
| qwen3:32b Few-Shot | 120-sample | 0.1932 [0.0794, 0.3125] | 0.1633 [0.0851, 0.2904] | 0.7273 | 40.8% |
| qwen3:32b Vanilla | 120-sample | 0.2174 [0.0556, 0.3884] | 0.1923 [0.0851, 0.3788] | 0.4545 | 21.7% |
| GGUF Q4_K_M Vanilla | 120-sample | 0.2632 [0.0000, 0.6250] | 0.5000 [0.0945, 0.9055] | 0.0909 | 1.7% |

The LR-TF-IDF baseline is useful as a sanity check, but the 120-sample number is
still high-variance and should not be sold as a final leaderboard result.

### Private/Blind Prediction Distribution

The local private/blind GGUF run used the repository's Few-Shot-style
`BEST_PROMPT` and produced:

| Prediction | Count | Rate |
| --- | ---: | ---: |
| SUCCESS | 333 | 7.4% |
| FAILURE | 4167 | 92.6% |

Because the private file has no labels, this is not a scored result.

## Paper Files

- `PAPER_DRAFT_NEURIPS.md`: final Markdown paper draft.
- `NEURIPS_REVIEW.md`: final review and readiness notes.

The LaTeX folder is ignored so the repository stays focused on reproducible code
and final artifacts. If a LaTeX submission is needed later, rebuild it from the
Markdown draft using the official conference template.

## Reproducibility Caveats

- The 120-sample LLM results contain only 11 positive examples.
- Local Ollama results can vary with quantization, hardware, Ollama version, and
  decoding settings.
- The LLM CI script reconstructs hard-label vectors from summary counts for
  rows where per-example predictions were not stored.
- The private/blind prediction file is unscored because private labels are not
  available in this workspace.
- Hardware specifications for the local Ollama machine were not recoverable
  from this environment and should be added in any formal submission.

## Ethical Note

Founder-success prediction is a sensitive application. Models may over-weight
prestige signals such as elite universities, prior big-tech employment,
geography, or prior access to capital. This repository is benchmark research,
not a deployable investment system. Any real use would require fairness audits,
human oversight, calibration monitoring, and strict limits on automated
decision-making.
