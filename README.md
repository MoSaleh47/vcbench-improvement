# VCBench Local LLM Experiments

This repository contains experiments for evaluating large language models on
VCBench, a venture-capital founder success prediction benchmark. The task is to
read an anonymized founder profile and predict one binary label:

- `SUCCESS`: the founder's company raised more than $500M, or had an IPO or
  acquisition above $500M within eight years.
- `FAILURE`: the company did not reach one of those outcomes.

The public VCBench split is highly imbalanced, with about a 9% success rate.
For that reason, the main metric is `F0.5`, which weights precision more than
recall. This matches the venture-capital setting where false positives are
expensive: a model that marks too many weak founders as successes is not useful
as a screening tool.

## What We Did

The original starter kit was extended into a local LLM experimentation workflow.
The main additions are:

- Prompt engineering experiments over four prompt styles:
  `vanilla`, `cot`, `few_shot`, and `hybrid`.
- Local Ollama inference so experiments can run without sending founder data to
  hosted APIs.
- Support for Qwen models served by Ollama, including:
  `qwen3:32b` and `hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`.
- No-thinking inference for Qwen models using both Ollama's `think: false`
  option and the `/no_think` control token.
- Separate result CSVs for public validation runs and private/blind prediction
  runs.
- A Pandas 3-compatible fix to the stratified public sampler in
  `run_experiments.py`.

The current local default model is:

```text
hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
```

This model was chosen after the first local `qwen3:32b` run was slow,
especially on chain-of-thought prompts.

## Repository Layout

```text
.
|-- run_experiments.py
|-- run_experiments_ollama.py
|-- test_real_data.py
|-- test_real_data_ollama.py
|-- evaluation.py
|-- data.py
|-- vcbench_final_public.csv
|-- vcbench_final_private.csv
|-- results*.csv
|-- requirements.txt
|-- README.md
```

Important files:

- `run_experiments.py`: original public-split experiment script using the Groq
  OpenAI-compatible API. It defines the canonical prompt templates used by the
  local wrapper.
- `run_experiments_ollama.py`: local Ollama version of the public experiment
  runner. It imports the same prompts from `run_experiments.py`.
- `test_real_data.py`: original private/blind prediction script using the Groq
  API.
- `test_real_data_ollama.py`: local Ollama version for full private/blind
  prediction. It imports the same `BEST_PROMPT` from `test_real_data.py`.
- `requirements.txt`: Python dependencies for the project.
- `results_ollama_qwen3_30b_a3b_q4_k_m_public_120.csv`: public 120-sample
  validation result for the current local GGUF model.
- `results_ollama_qwen3_30b_a3b_q4_k_m_private_full.csv`: full private/blind
  predictions for the current local GGUF model.
- `PAPER_DRAFT_NEURIPS.md`: publication-facing NeurIPS-style paper draft based
  on the latest local experiment progression.
- `NEURIPS_REVIEW.md`: NeurIPS-style reviewer critique of the current draft.
- `neurips_latex/`: LaTeX source formatted for the official NeurIPS template.

## Environment Setup

The original `venv/` in this workspace was broken because it pointed to a
Python executable that no longer existed. A fresh project-local `.venv/` was
created and ignored by git.

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If creating the environment from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project uses:

- `pandas`
- `numpy`
- `scikit-learn`
- `openai`
- `httpx`
- `python-dotenv`
- `tqdm`
- `pydantic`
- `pydantic-settings`

## Ollama Setup

Install and run Ollama, then pull the current default model:

```powershell
ollama pull hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
```

Check installed models:

```powershell
ollama list
```

The local scripts use:

```text
http://127.0.0.1:11434
```

instead of `localhost`, because `127.0.0.1` worked reliably in this Windows
environment while `localhost` sometimes produced connection errors.

## Prompt Templates

The Ollama public runner uses exactly the same prompt templates as
`run_experiments.py`:

- `vanilla`: direct zero-shot prediction with the 9% base-rate warning.
- `cot`: structured reasoning over positive signals, risks, and base-rate
  check.
- `few_shot`: two in-context examples, one failure and one success.
- `hybrid`: few-shot examples plus structured reasoning.

`test_real_data_ollama.py` uses the same private-test prompt as
`test_real_data.py`, because it imports `BEST_PROMPT` directly from that file.

For Qwen models, the local wrappers prepend:

```text
/no_think
```

and also send:

```json
{"think": false}
```

to the Ollama native chat endpoint. This keeps the runs in no-thinking mode and
prevents the model from spending most of its token budget on hidden or explicit
reasoning text.

## Running Public Experiments

Fast public validation run with only the `vanilla` prompt:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts vanilla --skip-temp-sweep
```

Run the same prompt used by the private script:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts few_shot --skip-temp-sweep
```

Run all prompt strategies, but skip the temperature sweep:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts vanilla,cot,few_shot,hybrid --skip-temp-sweep
```

Run the full original public experiment, including prompt comparison and
temperature tuning:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120
```

Warning: `cot` and `hybrid` can be very slow on large local Qwen models. In one
run, a `cot` row took more than a minute. For practical iteration, use
`--prompts vanilla --skip-temp-sweep` first.

## Running Private/Blind Predictions

Run the full private/blind dataset with the local default Ollama model:

```powershell
.\.venv\Scripts\python.exe test_real_data_ollama.py --sample-size 0
```

The output defaults to:

```text
results_ollama_qwen3_30b_a3b_q4_k_m_private_full.csv
```

If a file with that name already exists, `test_real_data_ollama.py` automatically
adds a timestamp to avoid overwriting it.

Run a small private smoke test:

```powershell
.\.venv\Scripts\python.exe test_real_data_ollama.py --sample-size 20 --results-file smoke_private_ollama.csv
```

## Hosted API Scripts

The original scripts still exist:

```powershell
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe test_real_data.py
```

Those scripts expect `GROQ_API_KEY` in `.env` and call:

```text
https://api.groq.com/openai/v1
```

The local Ollama scripts do not require an API key.

## Current Results

### Qwen3-32B via local Ollama

File:

```text
results_ollama_qwen3_32b_public_120.csv
```

Rows currently recorded:

| Prompt | Temperature | F0.5 | Precision | Recall | TP | FP | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| few_shot | 0.0 | 0.1932 | 0.1633 | 0.7273 | 8 | 41 | 120 |
| vanilla | 0.0 | 0.2174 | 0.1923 | 0.4545 | 5 | 21 | 120 |

The `cot` run was stopped because it was too slow locally.

### Qwen3-30B-A3B GGUF via local Ollama

File:

```text
results_ollama_qwen3_30b_a3b_q4_k_m_public_120.csv
```

Rows currently recorded:

| Prompt | Temperature | F0.5 | Precision | Recall | TP | FP | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 0.0 | 0.2632 | 0.5000 | 0.0909 | 1 | 1 | 120 |

This run is much more conservative than the earlier `qwen3:32b` run. It only
predicted two successes in the 120-profile validation sample.

### Private/Blind Predictions

File:

```text
results_ollama_qwen3_30b_a3b_q4_k_m_private_full.csv
```

This file contains 4,500 private/blind predictions. The private file used here
does not include a `success` label, so this is not a scored evaluation. It is a
submission-style prediction file.

Current prediction distribution:

| Prediction | Count | Rate |
| --- | ---: | ---: |
| SUCCESS | 333 | 7.4% |
| FAILURE | 4167 | 92.6% |

## Evaluation Method

The public experiment runner:

1. Loads `vcbench_final_public.csv`.
2. Keeps `anonymised_prose`, `success`, and `industry`.
3. Uses a stratified 80/20 split with `SEED = 42`.
4. Samples a class-balanced validation subset when `--sample-size` is set.
5. Runs one or more prompts.
6. Parses the last `SUCCESS` or `FAILURE` token from the model output.
7. Computes precision, recall, F0.5, TP, FP, FN, TN, and predicted-success
   count.
8. Appends metrics to the selected result CSV.

The private/blind runner:

1. Loads `vcbench_final_private.csv`.
2. Uses all rows when `--sample-size 0`.
3. Runs the imported `BEST_PROMPT`.
4. Saves the original rows plus `prediction` and `model`.
5. Computes metrics only if a `success` column is present.

## Notes on the Draft Paper

Two paper drafts were reviewed locally:

- `VC_Bench-1.pdf`: earlier positive framing, arguing that Qwen3-32B nearly
  closes the gap with DeepSeek-V3 through prompt engineering.
- `VC_Bench-2.pdf`: stronger revised framing, arguing that prompt engineering
  mostly exposes an overprediction/calibration failure on an imbalanced task.

The second framing is more scientifically robust because it compares against
trivial baselines and reports predicted-positive rate, not only F0.5.

## Troubleshooting

### `venv` points to a missing Python

Use `.venv`, not `venv`:

```powershell
.\.venv\Scripts\python.exe --version
```

### Ollama connection refused

Confirm Ollama is running:

```powershell
ollama list
```

Use:

```text
http://127.0.0.1:11434
```

instead of `localhost`.

### Qwen prints reasoning despite no-thinking mode

The wrappers already use both:

```text
/no_think
```

and:

```json
{"think": false}
```

If changing scripts manually, keep both.

### Runs are too slow

Start with:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts vanilla --skip-temp-sweep
```

Avoid `cot` and `hybrid` for first-pass local runs.

### Result file appends old runs

`run_experiments_ollama.py` appends metric rows to its result file. To isolate a
new run, pass a new output path:

```powershell
.\.venv\Scripts\python.exe run_experiments_ollama.py --sample-size 120 --prompts vanilla --skip-temp-sweep --results-file results_new_run.csv
```

## Reproducibility Caveats

- Local Ollama outputs can vary across model quantizations, hardware, Ollama
  versions, and decoding settings.
- Public validation results are based on small samples, often with only about 11
  positive examples in a 120-row sample.
- The private/blind file in this workspace does not expose labels, so the full
  private output is a prediction file rather than a scored benchmark result.
- The local GGUF model and the earlier `qwen3:32b` runs should not be mixed in a
  paper table unless they are clearly identified as separate model settings.

## Ethical Considerations

Founder-success prediction is a sensitive application. Models may over-weight
prestige signals such as elite universities, large-company experience, geography,
or prior access to capital. Any real deployment would require fairness auditing,
human oversight, and careful limits on how predictions are used. The current
experiments should be treated as benchmark research, not as a production VC
screening system.
