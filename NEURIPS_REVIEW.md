# Final Review and Submission Readiness Notes

This is the final internal review for the VCBench local LLM project after the
repo cleanup and analysis pass.

## Executive Summary

The project is now coherent as a workshop-style benchmark-analysis paper. The
strongest scientific contribution is the diagnosis of a failure mode: on a
severely imbalanced VC prediction task, prompt-engineered LLMs can look
reasonable under `F0.5` while providing poor screening utility because they
predict `SUCCESS` far too often.

The final framing is stronger than the original positive-result story. The
paper should not claim that Qwen prompting closes the gap with frontier models.
It should claim that calibration diagnostics are mandatory for interpreting LLM
results on imbalanced, high-stakes classification tasks.

## What Improved

- The title and paper draft now focus on failure modes and calibration.
- The final README separates code, data, results, and paper artifacts.
- The LaTeX folder is ignored and excluded from the final repository package.
- The final analysis includes trivial all-`FAILURE` and all-`SUCCESS` baselines.
- Bootstrap `F0.5` intervals and Wilson precision intervals are included in
  `bootstrap_ci_results.csv`.
- A precision-recall figure is included as `pr_curve.pdf` and `pr_curve.png`.
- The public GGUF Vanilla result is no longer conflated with the private
  Few-Shot-style prompt.
- The private/blind output is correctly described as unscored.

## Final Evidence

### LLM Operating Points

| Model | Prompt | F0.5 | Precision | Recall | TP | FP | Pred+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3:32b` | Few-Shot | 0.1932 | 0.1633 | 0.7273 | 8 | 41 | 40.8% |
| `qwen3:32b` | Vanilla | 0.2174 | 0.1923 | 0.4545 | 5 | 21 | 21.7% |
| Qwen3-30B-A3B GGUF | Vanilla | 0.2632 | 0.5000 | 0.0909 | 1 | 1 | 1.7% |

The GGUF row is more conservative, but it is based on only two positive
predictions. It is a calibration signal, not a robust model-ranking result.

### Baseline Context

| System | Split | F0.5 [95% CI] | Precision [Wilson CI] | Recall | PPR |
| --- | --- | ---: | ---: | ---: | ---: |
| All-SUCCESS | 120-sample | 0.1120 [0.0515, 0.1807] | 0.0917 [0.0520, 0.1567] | 1.0000 | 100.0% |
| LR-TF-IDF | 120-sample | 0.7143 [0.3125, 0.9091] | 0.8333 [0.4365, 0.9699] | 0.4545 | 5.0% |
| GGUF Q4_K_M Vanilla | 120-sample | 0.2632 [0.0000, 0.6250] | 0.5000 [0.0945, 0.9055] | 0.0909 | 1.7% |

The LR-TF-IDF baseline is useful because it shows that simple structured
evaluation can dominate hard-label LLM points on the sampled split. It should be
presented carefully because the 120-sample result has a wide confidence
interval.

## Recommended Submission Position

### Workshop Paper: Reasonable

This version can work as a workshop or class-project submission if the claim is
kept modest:

> We audit local LLM behavior on VCBench and show that calibration, not prompt
> wording alone, determines whether the model has screening utility under severe
> class imbalance.

### NeurIPS Main Track: Not Ready

The current evidence is not enough for a NeurIPS main-track claim. A main-track
version would still need:

- Full public validation evaluation or repeated stratified splits.
- Row-level saved predictions for every model/prompt.
- Stronger structured baselines with locked feature pipelines.
- Confidence intervals for every headline metric.
- Hardware, Ollama version, and model artifact metadata.
- Clear comparison to official VCBench reference systems.
- Expanded fairness and responsible-use discussion.

## Remaining Risks

- The LLM confidence intervals are reconstructed from summary counts, not
  original row-level predictions.
- The private/blind file has no labels, so the 333/4,500 `SUCCESS` count cannot
  be scored.
- The local GGUF and `qwen3:32b` settings should never be collapsed into one
  "Qwen" result.
- The hosted Groq scripts and local Ollama scripts use related but distinct
  inference stacks.
- The final project contains some legacy scripts retained for reproducibility;
  the README identifies the main path to use.

## Final Recommendation

Submit the work as a compact negative-result/evaluation-methodology paper. Lead
with the calibration failure mode, not with the best `F0.5` row. The project is
now much more honest and much more useful because it shows how easy it is to
misread LLM performance on an imbalanced financial benchmark.
