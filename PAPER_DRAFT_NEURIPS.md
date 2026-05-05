# Failure Modes and Calibration of Local LLMs on Imbalanced VC Prediction

## Abstract

Venture-capital screening is an imbalanced rare-event prediction problem: only a
small fraction of founders produce extreme outcomes, while false positives waste
analyst attention and capital. We study this setting with VCBench, a benchmark
for predicting founder success from anonymized pre-founding profiles. Starting
from a prompt-engineering baseline over Qwen3-32B, we reframe the study around
failure modes and calibration under class imbalance.

We evaluate local Ollama inference with no-thinking Qwen models, including
`qwen3:32b` and the quantized `hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`. Our main
finding is that prompt engineering mostly changes the predicted-positive rate
rather than reliably improving discrimination. On a 120-profile public
validation sample, local `qwen3:32b` Few-Shot reaches `F0.5 = 0.1932` by
predicting `SUCCESS` for 40.8% of profiles, while local `qwen3:32b` Vanilla
reaches `F0.5 = 0.2174` with a lower but still high 21.7% predicted-positive
rate. A quantized Qwen3-30B-A3B GGUF Vanilla run reaches `F0.5 = 0.2632` with
precision 0.5000, but this comes from only two positive predictions and one true
positive; the Wilson 95% interval for precision is 0.0945 to 0.9055.

On the 4,500-profile private/blind file, the GGUF model with the repository's
private Few-Shot-style prompt predicts `SUCCESS` for 333 founders (7.4%), close
to the benchmark base rate, but labels are unavailable and this is therefore a
submission-style prediction file rather than a scored result. These results
suggest that local open-weight LLMs can be made more conservative, but
small-sample `F0.5` gains must be interpreted alongside predicted-positive rate,
trivial baselines, and confidence intervals.

## 1. Introduction

Early-stage venture-capital prediction is difficult because the signal is sparse,
the labels are noisy, and successful outcomes are rare. A practically useful
screening model should not merely identify many successful founders; it must do
so while keeping false positives low enough to reduce the downstream review
burden. VCBench formalizes this task by asking models to classify anonymized
founder profiles as `SUCCESS` or `FAILURE`, with success defined as raising more
than `$500M` or achieving an IPO/acquisition above `$500M` within eight years.

The original VCBench framing uses `F0.5` as the main metric because precision is
more important than recall in capital allocation. However, `F0.5` alone can hide
important behavior. A model that predicts `SUCCESS` for nearly every founder can
obtain nonzero recall and a superficially plausible `F0.5` while providing
little screening utility. Conversely, a model that predicts very few successes
can obtain high precision on a small sample while missing almost all true
positives.

This project began as a prompt-engineering study on Qwen3-32B. The first draft
framed Few-Shot prompting as nearly closing the gap with larger proprietary
models. Subsequent analysis showed that this interpretation was too optimistic:
the apparent gains were driven by overprediction. We therefore revise the
scientific question from "Can prompting close the frontier-model gap?" to "How
do prompts and local model variants move the precision-recall and calibration
tradeoff on VCBench?"

The core contribution is empirical and methodological: we document how model
choice and inference settings can shift predicted-positive rate from extreme
overprediction toward the benchmark base rate, but the evidence remains
small-sample and should be judged through calibration diagnostics rather than
`F0.5` alone. In its current form, the paper is best positioned as a workshop or
benchmark-analysis submission rather than a main-track performance claim.

## 2. Contributions

1. A reproducible local Ollama workflow for VCBench using the same prompt
   templates as the hosted API experiments.
2. Documentation of no-thinking local Qwen inference and its practical value for
   runtime and output parsing.
3. Public 120-sample validation results for local `qwen3:32b` and
   `hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`.
4. A 4,500-row private/blind prediction distribution for the GGUF model, without
   claiming unscored private-set performance.
5. Calibration-oriented diagnostics: predicted-positive rate, trivial baselines,
   bootstrap `F0.5` intervals, Wilson precision intervals, and a PR-curve figure
   anchored by an LR-TF-IDF baseline.

## 3. Data and Task

We use `vcbench_final_public.csv`, which contains 4,500 anonymized founder
profiles with a success rate of about 9%. Each row includes `anonymised_prose`,
structured background columns, industry, and the binary label `success`.

For public validation, the scripts use a stratified 80/20 split with seed 42,
followed by evaluation on a stratified 120-row sample from the validation pool.
The 120-row sample contains 11 positives and 109 negatives. Metrics include
`F0.5`, precision, recall, TP, FP, FN, TN, and predicted-success count.

We also run predictions on `vcbench_final_private.csv`. The local copy used here
contains 4,500 rows but no `success` labels. Therefore, the private/blind run is
a prediction-generation exercise and not a scored evaluation.

## 4. Models and Inference

The original hosted baseline used Qwen3-32B through a Groq OpenAI-compatible
API. That run evaluated four prompt strategies: Vanilla, Chain-of-Thought,
Few-Shot, and Hybrid. Later analysis showed that the apparent Few-Shot gain was
driven by a very high predicted-positive rate and did not clearly outperform
trivial high-positive baselines.

The local experiments use Ollama's native chat endpoint at
`http://127.0.0.1:11434/api/chat`. The public local runner imports `PROMPTS`
from `run_experiments.py`. The private local runner imports `BEST_PROMPT` from
`test_real_data.py`. Thus, local scripts preserve the prompt content while
changing the inference backend.

Qwen reasoning models can spend substantial time producing thinking traces.
During local experiments, this made reasoning prompts slow and sometimes
returned empty assistant content through OpenAI-compatible routes. The local
scripts therefore use Ollama native chat and disable thinking with both
`/no_think` and `think: false`. The label parser extracts the last occurrence of
`SUCCESS` or `FAILURE`.

## 5. Prompt Strategies

- `vanilla`: direct prediction with the success definition, 9% base-rate warning,
  and instruction to be conservative.
- `cot`: explicit positive-signal, risk-factor, and base-rate reasoning steps.
- `few_shot`: two examples, one failure and one success.
- `hybrid`: few-shot examples plus structured reasoning.

The local full-private script uses the Few-Shot-style `BEST_PROMPT` from
`test_real_data.py`, not the public Vanilla prompt.

## 6. Results

### Reference Baselines and Uncertainty

On the 120-row public validation sample, an all-`FAILURE` classifier obtains
`F0.5 = 0`, while an all-`SUCCESS` classifier obtains `F0.5 = 0.1120` with
precision 0.0917 and recall 1.0. These baselines are weak but important: any
prompt that mostly predicts `SUCCESS` can appear to recover recall while adding
little screening value.

The small number of positives also makes precision estimates unstable. For
example, the GGUF Vanilla row's precision is 0.5000, but its Wilson 95% interval
is 0.0945 to 0.9055 because it is based on two positive predictions.

### Public Validation: Local Qwen3-32B

| Model | Prompt | T | F0.5 | Precision | Recall | TP | FP | Pred+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3:32b` | Few-Shot | 0.0 | 0.1932 | 0.1633 | 0.7273 | 8 | 41 | 40.8% |
| `qwen3:32b` | Vanilla | 0.0 | 0.2174 | 0.1923 | 0.4545 | 5 | 21 | 21.7% |

The Vanilla prompt outperformed Few-Shot in `F0.5` because it reduced false
positives substantially, even though it found fewer true positives. This is a
calibration effect: both prompts remain far above the 9% public base rate in
predicted-positive rate.

### Public Validation: Local Qwen3-30B-A3B GGUF

| Model | Prompt | T | F0.5 | Precision | Recall | TP | FP | FN | Pred+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B GGUF | Vanilla | 0.0 | 0.2632 | 0.5000 | 0.0909 | 1 | 1 | 10 | 1.7% |

This is the highest `F0.5` observed among the latest local LLM rows, but it
should be interpreted as evidence of a conservative calibration shift, not as a
robust performance claim.

### Structured Baseline and PR Curve

The final repository adds a lightweight LR-TF-IDF baseline and precision-recall
curve. On the full 900-row validation split, the LR-TF-IDF threshold tuned on the
120-subset obtains `F0.5 = 0.3333` with precision 0.4167, recall 0.1852, and a
4.0% predicted-positive rate. On the 120-sample subset, LR-TF-IDF obtains
`F0.5 = 0.7143`, but the confidence interval is wide (`0.3125` to `0.9091`) and
the result should be treated as a sanity-check baseline rather than a final
leaderboard claim.

The generated figure `pr_curve.pdf` visualizes the LR-TF-IDF PR curve, trivial
baselines, and hard-label LLM operating points.

### Private/Blind Prediction Distribution

The full private/blind file contains 4,500 profiles and no labels. The local
GGUF run used the repository's private Few-Shot-style `BEST_PROMPT` and produced:

| Prediction | Count | Rate |
| --- | ---: | ---: |
| SUCCESS | 333 | 7.4% |
| FAILURE | 4167 | 92.6% |

This distribution is close to the public benchmark base rate of about 9%.
However, without labels, we cannot compute `F0.5`, precision, recall, or compare
against official private baselines.

## 7. Discussion

First, prompt engineering alone is not the main story. The same family of
prompts can produce very different predicted-positive rates depending on model
backend, quantization, and inference settings. This makes calibration diagnostics
essential.

Second, no-thinking local inference is practically important. It reduces
uncontrolled reasoning output and makes local evaluation more feasible. It also
prevents a mismatch where the model spends the token budget on reasoning while
returning no usable classification content.

Third, the GGUF model appears more conservative than the earlier local
`qwen3:32b` setting. This is promising for VC screening because extreme
overprediction is harmful. But conservatism can also become underprediction: the
public validation row recalls only 1 of 11 successes. A useful system must
balance this tradeoff across larger samples.

## 8. Limitations

- The public LLM validation results are based on a 120-row sample.
- The strongest GGUF public result has only one true positive and one false
  positive.
- CoT and Hybrid were not fully rerun under the latest local GGUF setup because
  local reasoning prompts were too slow for practical iteration.
- The private/blind predictions are unscored because labels are unavailable.
- The CI script reconstructs LLM hard-label vectors from summary counts where
  row-level predictions were not stored.
- The LR-TF-IDF 120-sample result is useful for diagnosis but high variance.
- Hardware specifications for the local Ollama machine were not available from
  this workspace.

## 9. Broader Impact

Automated VC screening can reinforce existing inequities if models over-weight
prestige signals such as elite universities, prior big-tech employment, or
geographic access to startup ecosystems. Even anonymized profiles can retain
strong socioeconomic proxies. The present work should therefore be interpreted
as benchmark research, not as a deployable investment system. Any deployment
would require fairness audits, human oversight, calibration monitoring, and a
clear policy that model outputs are decision support rather than automated
funding decisions.

## 10. Conclusion

The latest experiments shift the conclusion from "prompt engineering can make
open-weight LLMs competitive on VCBench" to a more defensible claim:
calibration determines whether local LLMs are useful for rare-event VC
screening. Local no-thinking Qwen inference can move predictions from severe
overprediction toward more conservative behavior, and the current GGUF
private/blind run predicts `SUCCESS` at a rate close to the public base rate.
However, the evidence is not yet strong enough for a final performance claim.

For a NeurIPS main-track submission, the next required step would be a larger
and more statistically grounded evaluation: full public split or repeated
stratified splits, stronger structured baselines, confidence intervals for all
headline metrics, and an explicit comparison to official VCBench reference
systems. In its present form, this is a more credible workshop paper: a compact
negative-result and evaluation-methodology note on LLM failure modes in
imbalanced, high-stakes prediction tasks.

## Reproducibility Checklist

- Public data: `vcbench_final_public.csv`.
- Local inference scripts: `run_experiments_ollama.py` and
  `test_real_data_ollama.py`.
- Bootstrap/CI script: `compute_bootstrap_cis.py`.
- PR-curve script: `generate_pr_curve.py`.
- Current local model: `hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`.
- Inference backend: Ollama native chat endpoint.
- No-thinking controls: `/no_think` and `think: false`.
- Public sample size: 120.
- Public sample positives: 11.
- Seed: 42.
- Final artifacts: `bootstrap_ci_results.csv`, `bootstrap_ci_results.tex`,
  `pr_curve.pdf`, and `pr_curve.png`.
