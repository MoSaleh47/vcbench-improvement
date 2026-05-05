# NeurIPS-Style Review of the VCBench Draft

Draft reviewed: `VC_Bench-2.pdf`

Context draft used for comparison: `VC_Bench-1.pdf`

Review rubric sources:

- NeurIPS 2015 evaluation criteria:
  https://neurips.cc/Conferences/2015/PaperInformation/EvaluationCriteria
- NeurIPS 2025 reviewer guidelines:
  https://neurips.cc/Conferences/2025/ReviewerGuidelines

## Summary

The paper studies prompt engineering for Qwen3-32B on VCBench, an imbalanced
founder-success classification benchmark. It evaluates four prompting strategies
(`vanilla`, `cot`, `few_shot`, and `hybrid`) and a temperature sweep. The key
claim is negative: the best prompt-engineered LLM configuration achieves a
held-out F0.5 of 0.1028, but this is below trivial high-positive-rate baselines
such as Always-SUCCESS and below the DeepSeek-V3 result reported by the VCBench
paper. The paper argues that prompt engineering primarily shifts the
precision-recall tradeoff toward harmful overprediction and that F0.5 should be
reported with predicted-positive rate and trivial baselines on rare-event tasks.

The revised draft is substantially stronger than the earlier `VC_Bench-1.pdf`.
The earlier draft framed the same result as nearly closing the gap with
DeepSeek-V3, while the current draft more honestly shows that the high recall is
mostly caused by near-universal SUCCESS predictions. This reframing is the main
scientific value of the work.

## Strengths

- The paper identifies a practically important failure mode: prompt-engineered
  LLMs can achieve superficially plausible F0.5 scores while providing almost no
  useful screening because they predict SUCCESS for most examples.
- Reporting predicted-positive rate is a strong addition. In the validation
  table, the best Few-Shot prompt predicts SUCCESS for 94.2% of profiles, which
  makes the reported recall much easier to interpret.
- The comparison against Always-SUCCESS, Always-FAILURE, random base-rate, and
  random matched-rate baselines is exactly the right instinct for a severely
  imbalanced benchmark.
- The negative-result framing is significant for applied ML: the paper does not
  just say "our prompt did poorly"; it argues that metric reporting can obscure
  non-discriminative behavior.
- The broader-impact discussion correctly flags prestige bias and the risk of
  automating VC screening without fairness audits.
- Including structured-feature ML baselines is promising because it challenges
  the assumption that LLM prompting is the right tool for this task.

## Weaknesses

### Quality

The experimental evidence is suggestive but not yet strong enough for a NeurIPS
main-track claim. The validation set has only 120 examples and roughly 11
positive labels, so small changes in TP/FP counts can visibly move F0.5. The
paper acknowledges this limitation, but the main claims would be much stronger
with confidence intervals, bootstrapping, and multiple random seeds.

The held-out test construction is also confusing. The draft says prompt tuning
uses a 20% validation pool and final evaluation samples a disjoint `n=200` from
the 80% training pool. This is technically disjoint from prompt tuning, but
calling it a "held-out test set" may mislead readers because it is still drawn
from the public split and not the official private VCBench evaluation. I would
call it an "internal public test split" unless the official private labels are
used.

The structured ML baselines are under-specified. The paper reports that logistic
regression, random forest, and gradient boosting reach F0.5 of 0.14-0.21 on a
900-example split, but it does not provide enough details to reproduce the
features, hyperparameters, train/test protocol, or exact per-model table. Since
this comparison is important to the paper's conclusion, it needs a full table
and enough implementation detail to be auditable.

The model setup needs to be frozen. The draft says Qwen3-32B via Groq API, while
the current repository also contains local Ollama runs for
`hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`. These should not be mixed. If the paper
is about Qwen3-32B via Groq, keep all tables and claims tied to that setting. If
the newer local GGUF results are used, the draft should explicitly identify the
model, quantization, Ollama version, and no-thinking settings.

### Clarity

The main narrative is clear and much improved over the first draft. The title,
abstract, and conclusion all align around the same message: prompt engineering
does not solve the imbalanced classification problem and can hide
overprediction.

However, the draft has several presentation issues that would hurt a NeurIPS
review:

- Some citations are malformed, e.g. "VC context et al. [2025a]" and "VCBench
  et al. [2025a]".
- The references mix author names and "et al." placeholders in a way that makes
  related work look unfinished.
- Tables extracted from the PDF are dense and hard to read; they should include
  all key counts, especially `FN`, `TN`, and `Pred+ Rate`, consistently.
- The term "trivial high-positive-rate baselines" should be defined before it
  appears in the abstract, or rephrased for readers who have not yet seen the
  baseline section.

### Significance

The paper addresses a real and difficult applied ML problem: rare-event
prediction under severe class imbalance in a high-stakes financial setting. The
most significant contribution is not a new algorithm, but an evaluation insight:
LLM benchmark results on imbalanced tasks can look meaningful unless compared
to simple baselines and predicted-positive rates.

This is aligned with NeurIPS application-paper criteria: the task is real,
difficult, and practically relevant. The work would be more significant if the
authors showed that the overprediction failure generalizes across multiple
open-weight models, not only Qwen3-32B.

### Originality

The individual components are not novel: prompt engineering, temperature
ablation, F0.5, and trivial baselines are standard. The originality lies in
applying these checks to VCBench and exposing the gap between nominal F0.5 and
practical screening usefulness. That is a valid contribution, especially as a
negative empirical study, but the current evidence base is narrow.

## Questions for the Authors

1. Can you evaluate the best prompts over the full public split, or at least over
   multiple stratified splits with confidence intervals? With only 11 positives
   in the validation sample, the current ranking of prompts may be unstable.

2. Can you clarify the final test protocol? Is the `n=200` final test an
   internal public split, the official private set, or something else? Please
   avoid calling it "strict held-out" unless it is fully separated from all
   development decisions.

3. Can you provide a complete structured-baseline table with exact features,
   hyperparameters, train/test split, class weighting, and confidence intervals?
   These baselines are central to the claim that classical ML is currently more
   useful than prompting.

4. Can you add a calibration analysis beyond predicted-positive rate, such as
   thresholded logits/probabilities if available, self-reported confidence
   calibration, or a selective-classification curve? The paper's diagnosis is
   calibration-related, so a calibration plot would strengthen it.

5. Can you separate results for hosted Qwen3-32B from local quantized Qwen runs?
   A reviewer needs to know exactly which model, quantization, inference server,
   and thinking/no-thinking mode generated each table.

Clear criteria that would raise my score: full-split or multi-seed results,
reproducible ML baselines, corrected split terminology, and one additional
open-weight model showing the same overprediction pattern.

Clear criteria that would lower my score: if the structured baselines cannot be
reproduced, if the final test is not actually disjoint from prompt selection, or
if the trivial-baseline comparison changes materially under a larger evaluation.

## Limitations and Broader Impact

The draft includes a useful limitations section and a reasonable broader-impact
section. I would not flag the paper for ethics review on the basis of the text
alone, but I would ask the authors to expand the fairness discussion. In
particular, the paper should say which protected or proxy attributes are absent,
which proxies remain in the data, and why prestige-based features can produce
unequal access to funding even when demographic variables are removed.

The draft should also be explicit that the system is not deployment-ready. The
current results are better interpreted as a benchmark audit than as an automated
investment tool.

## NeurIPS Scores

- Quality: 2 / 4 (fair)
- Clarity: 3 / 4 (good)
- Significance: 3 / 4 (good)
- Originality: 3 / 4 (good)
- Overall: 3 / 6 (borderline reject)
- Confidence: 3 / 5 (fairly confident)

## Overall Recommendation

Borderline reject in its current form, but with a clear path to borderline
accept or accept.

The paper has a good core idea and the revised framing is much more credible
than the earlier positive-results framing. The main blocker is experimental
strength: NeurIPS reviewers will likely want larger evaluations, uncertainty
estimates, better reproducibility for the structured baselines, and precise
separation between public validation, internal test, and private/blind
prediction.

If the authors address those issues, the work could become a valuable negative
result and evaluation-methodology paper for LLMs on imbalanced, high-stakes
classification tasks.
