# Calibration, Not Prompting, Determines Local LLM Utility on VCBench

## Abstract

Venture-capital screening is an imbalanced rare-event prediction problem: only a
small fraction of founders produce extreme outcomes, while false positives are
costly because they waste analyst attention and capital. We study this setting
using VCBench, a benchmark for predicting founder success from anonymized
pre-founding profiles. Starting from a prompt-engineering baseline over
Qwen3-32B, we extend the evaluation to local Ollama inference with no-thinking
Qwen models, including `qwen3:32b` and the quantized
`hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`. Our main finding is that prompt
engineering mostly changes calibration rather than reliably improving
discrimination. On a 120-profile public validation sample, local `qwen3:32b`
Few-Shot reaches F0.5 = 0.1932 by predicting SUCCESS for 40.8% of profiles,
while local `qwen3:32b` Vanilla reaches F0.5 = 0.2174 with a lower but still
high 21.7% predicted-positive rate. In contrast, the optimized Qwen3-30B-A3B
GGUF Vanilla run reaches F0.5 = 0.2632 with precision 0.5000, but this comes
from only two positive predictions and one true positive. On the 4,500-profile
private/blind file, the same GGUF setup predicts SUCCESS for 333 founders
(7.4%), close to the benchmark base rate, but labels are unavailable and this is
therefore a submission-style prediction file rather than a scored result. These
results suggest that local open-weight LLMs can be made more conservative, but
small-sample F0.5 gains must be interpreted alongside predicted-positive rate,
trivial baselines, and confidence intervals. We argue that VCBench-style
evaluation should report calibration-oriented diagnostics by default.

## 1 Introduction

Early-stage venture-capital prediction is difficult because the signal is sparse,
the labels are noisy, and successful outcomes are rare. A practically useful
screening model should not merely identify many successful founders; it must do
so while keeping false positives low enough to reduce the downstream review
burden. VCBench formalizes this task by asking models to classify anonymized
founder profiles as `SUCCESS` or `FAILURE`, with success defined as raising more
than $500M or achieving an IPO/acquisition above $500M within eight years.

The original VCBench framing uses F0.5 as the main metric because precision is
more important than recall in capital allocation. However, F0.5 alone can hide
important behavior. A model that predicts SUCCESS for nearly every founder can
obtain nonzero recall and a superficially plausible F0.5 while providing little
screening utility. Conversely, a model that predicts very few successes can
obtain high precision on a small sample while missing almost all true positives.

This project began as a prompt-engineering study on Qwen3-32B. The first draft
framed Few-Shot prompting as nearly closing the gap with larger proprietary
models. Subsequent analysis showed that this interpretation was too optimistic:
the apparent gains were driven by overprediction. We therefore revised the
scientific question from "Can prompting close the frontier-model gap?" to "How
do prompts and local model variants move the precision-recall and calibration
tradeoff on VCBench?"

This paper reports the latest stage of the work. We evaluate the same prompt
templates under local Ollama inference, disable Qwen thinking mode to reduce
latency and uncontrolled reasoning output, and compare two local Qwen settings.
The core contribution is empirical and methodological: we show that model choice
and inference settings can shift the predicted-positive rate from extreme
overprediction toward the benchmark base rate, but the evidence remains
small-sample and should be judged through calibration diagnostics rather than
F0.5 alone.

## 2 Contributions

1. We provide a reproducible local Ollama workflow for VCBench using the same
   prompt templates as the hosted API experiments.
2. We document the effect of no-thinking local Qwen inference on runtime and
   output parsing.
3. We report public 120-sample validation results for local `qwen3:32b` and
   `hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`.
4. We report the prediction distribution of a 4,500-row private/blind run for
   the GGUF model, while explicitly avoiding unscored private-performance claims.
5. We argue that predicted-positive rate, trivial baselines, and uncertainty
   estimates should be mandatory for rare-event LLM classification benchmarks.

## 3 Related Work

LLMs have recently been applied to financial decision-making, trading agents,
and future-event forecasting. These applications are vulnerable to look-ahead
bias when models use information that would not have been available at the
decision time. VCBench addresses this concern by anonymizing founder profiles and
restricting inputs to pre-founding information.

The present work is closest to VCBench and to live forecasting benchmarks such
as YCBench and FutureX. Unlike benchmarks focused primarily on frontier-model
accuracy, this study emphasizes evaluation pathology under class imbalance. The
central issue is not whether an LLM can produce a label, but whether the label
distribution is calibrated enough to provide screening value.

## 4 Data and Task

We use the public VCBench split available in this repository:

```text
vcbench_final_public.csv
```

It contains 4,500 anonymized founder profiles with a success rate of about 9%.
Each row includes `anonymised_prose`, structured background columns, industry,
and the binary label `success`.

For public validation, we use the existing script protocol:

1. Stratified 80/20 split with `SEED = 42`.
2. Evaluation on a stratified 120-row sample from the validation pool.
3. The 120-row sample contains 11 positives and 109 negatives.
4. Metrics include F0.5, precision, recall, TP, FP, FN, TN, and predicted-success
   count.

We also run predictions on:

```text
vcbench_final_private.csv
```

The local copy used here contains 4,500 rows but no `success` labels. Therefore,
the private/blind run is a prediction-generation exercise and not a scored
evaluation.

## 5 Models and Inference

### 5.1 Hosted API Baseline

The original experiment used Qwen3-32B through a Groq OpenAI-compatible API.
That run evaluated four prompt strategies:

- Vanilla
- Chain-of-Thought (CoT)
- Few-Shot
- Hybrid

The earlier draft reported that Few-Shot at temperature 0.0 achieved held-out
F0.5 = 0.1028 with high recall. Later analysis showed this result was caused by
very high predicted-positive rate and did not clearly outperform trivial
high-positive baselines.

### 5.2 Local Ollama Inference

We added local inference scripts:

```text
run_experiments_ollama.py
test_real_data_ollama.py
```

The public local runner imports `PROMPTS` directly from `run_experiments.py`.
The private local runner imports `BEST_PROMPT` directly from
`test_real_data.py`. Therefore, the local scripts use the same prompt content as
the API scripts; only the inference backend changes.

Local inference uses Ollama's native chat endpoint:

```text
http://127.0.0.1:11434/api/chat
```

We use `127.0.0.1` rather than `localhost` because it was more reliable in the
Windows environment used for the experiments.

### 5.3 No-Thinking Mode

Qwen reasoning models can spend substantial time producing thinking traces.
During local experiments, this made CoT-style prompts especially slow and made
some OpenAI-compatible responses return empty assistant content while reasoning
appeared in a separate field. We therefore use Ollama native chat and disable
thinking with both:

```text
/no_think
```

and:

```json
{"think": false}
```

The final label parser extracts the last occurrence of `SUCCESS` or `FAILURE`
from the model output.

## 6 Prompt Strategies

The four public prompt strategies are:

- `vanilla`: direct prediction with the success definition, 9% base-rate
  warning, and instruction to be conservative.
- `cot`: explicit positive-signal, risk-factor, and base-rate reasoning steps.
- `few_shot`: two examples, one failure and one success.
- `hybrid`: few-shot examples plus structured reasoning.

The local full-private script currently uses the same Few-Shot-style
`BEST_PROMPT` as `test_real_data.py`.

## 7 Results

### 7.1 Public Validation: Local Qwen3-32B

The first local Ollama run used `qwen3:32b`. Full CoT evaluation was stopped
because it was too slow locally; one CoT row took about 75 seconds. Completed
public 120-sample results are shown below.

| Model | Prompt | T | F0.5 | Precision | Recall | TP | FP | Pred+ Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3:32b | few_shot | 0.0 | 0.1932 | 0.1633 | 0.7273 | 8 | 41 | 40.8% |
| qwen3:32b | vanilla | 0.0 | 0.2174 | 0.1923 | 0.4545 | 5 | 21 | 21.7% |

The Vanilla prompt outperformed Few-Shot in F0.5 because it reduced false
positives substantially, even though it found fewer true positives.

### 7.2 Public Validation: Local Qwen3-30B-A3B GGUF

The next local run used:

```text
hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
```

The completed public validation row is:

| Model | Prompt | T | F0.5 | Precision | Recall | TP | FP | Pred+ Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B-GGUF Q4_K_M | vanilla | 0.0 | 0.2632 | 0.5000 | 0.0909 | 1 | 1 | 1.7% |

This is the highest F0.5 observed in the latest local public sample, but it
should be interpreted carefully. The result depends on only two positive
predictions and one true positive. It is evidence of a conservative calibration
shift, not yet evidence of robust superiority.

### 7.3 Private/Blind Prediction Distribution

The full private/blind file contains 4,500 profiles and no labels. The local
GGUF run produced:

| Prediction | Count | Rate |
| --- | ---: | ---: |
| SUCCESS | 333 | 7.4% |
| FAILURE | 4167 | 92.6% |

This distribution is close to the public benchmark base rate of about 9%.
However, without labels, we cannot compute F0.5, precision, recall, or compare
against official private baselines.

## 8 Discussion

The latest results support three conclusions.

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
overprediction is harmful. But conservatism can also become underprediction:
the public validation row recalls only 1 of 11 successes. A useful system must
balance this tradeoff across larger samples.

## 9 NeurIPS-Style Evaluation Against Acceptance Criteria

### Quality

The code path is reproducible and the local scripts preserve the original prompt
templates. The current weakness is statistical power. A 120-row sample with 11
positives is too small to support strong model-ranking claims. The results
should be extended to the full public validation pool or repeated over multiple
random seeds with confidence intervals.

### Clarity

The revised framing is clear: this is a calibration and evaluation study, not a
claim that prompting solves VC prediction. The paper should keep a strict
separation between:

- hosted Qwen3-32B API results,
- local `qwen3:32b` results,
- local quantized Qwen3-30B-A3B GGUF results,
- private/blind unscored predictions.

### Significance

The problem is significant because imbalanced high-stakes classification appears
in many applied ML domains. The practical lesson is broader than VCBench:
reporting F0.5 without predicted-positive rate and trivial baselines can lead to
misleading conclusions.

### Originality

The work does not introduce a new algorithm. Its originality lies in diagnosing
and documenting an evaluation failure mode for LLM-based rare-event
classification, then showing that local model and inference choices can shift
calibration dramatically.

## 10 Limitations

- The public validation results are based on a 120-row sample.
- The strongest GGUF public result has only one true positive and one false
  positive.
- CoT and Hybrid were not fully rerun under the latest local GGUF setup because
  local reasoning prompts were too slow for practical iteration.
- The private/blind predictions are unscored because labels are unavailable.
- The study has not yet evaluated multiple seeds, confidence intervals, or full
  public-split bootstrap estimates.
- The structured-feature ML baselines discussed in the earlier draft need to be
  fully reproduced and included in the repository before they can support a
  publication claim.

## 11 Broader Impact

Automated VC screening can reinforce existing inequities if models over-weight
prestige signals such as elite universities, prior big-tech employment, or
geographic access to startup ecosystems. Even anonymized profiles can retain
strong socioeconomic proxies. The present work should therefore be interpreted
as benchmark research, not as a deployable investment system. Any deployment
would require fairness audits, human oversight, calibration monitoring, and a
clear policy that model outputs are decision support rather than automated
funding decisions.

## 12 Conclusion

The latest experiments shift the paper's conclusion from "prompt engineering can
make open-weight LLMs competitive on VCBench" to a more defensible claim:
calibration determines whether local LLMs are useful for rare-event VC
screening. Local no-thinking Qwen inference can move predictions from severe
overprediction toward much more conservative behavior, and the current GGUF
private/blind run predicts SUCCESS at a rate close to the public base rate.
However, the evidence is not yet strong enough for a final performance claim.

For a NeurIPS-quality submission, the next required step is a larger and more
statistically grounded evaluation: full public split or repeated stratified
splits, trivial baselines in every table, confidence intervals, and a fully
reproducible structured-ML baseline. With those additions, the work could be a
credible negative-result and evaluation-methodology paper for LLMs on
imbalanced, high-stakes prediction tasks.

## Reproducibility Checklist

- Code for local inference: `run_experiments_ollama.py`,
  `test_real_data_ollama.py`.
- Public data: `vcbench_final_public.csv`.
- Private/blind prediction file: `vcbench_final_private.csv`.
- Current local model: `hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`.
- Inference backend: Ollama native chat endpoint.
- No-thinking controls: `/no_think` and `think: false`.
- Public sample size: 120.
- Public sample positives: 11.
- Seed: 42.
- Main result files:
  `results_ollama_qwen3_32b_public_120.csv`,
  `results_ollama_qwen3_30b_a3b_q4_k_m_public_120.csv`,
  `results_ollama_qwen3_30b_a3b_q4_k_m_private_full.csv`.
