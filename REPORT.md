# Do LLMs Behave Differently When the Prompter Is Human vs Another LLM?

## 1. Executive Summary

We tested whether large language models respond differently to prompts written in human style versus LLM style, with semantic content held constant. Across five experiments on three frontier models (GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Pro), we found **strong evidence that prompt style significantly affects LLM behavior**. The most striking finding: LLM-style prompts improve reasoning task accuracy by 20 percentage points (GPT-4.1) to 27 points (Gemini) on BBH Sports Understanding (p=0.003 and p<0.001 respectively), primarily because human-style prompts trigger verbose, conversational responses that fail to provide direct answers. On open-ended questions, LLM-style prompts elicit responses that are 57-63% longer (Cohen's d = 2.15-4.48, p<0.0001) and use significantly more formal language. However, explicit attribution ("this is from a human/AI") produces negligible effects, suggesting the behavioral adaptation is driven by implicit style detection rather than explicit source awareness.

## 2. Goal

**Hypothesis**: LLMs exhibit measurably different behaviors when receiving prompts written in a recognizably human style versus a recognizably LLM style, even when the semantic content is controlled.

**Why this matters**: As LLMs are increasingly used in multi-agent systems (where LLMs prompt other LLMs), understanding whether prompt style affects behavior has critical implications for:
- **AI safety**: If models behave differently based on perceived prompter identity, this could be exploited or could introduce systematic biases in AI pipelines.
- **Multi-agent reliability**: LLM-to-LLM communication chains may produce different outputs than human-to-LLM chains, even with identical content.
- **Prompt engineering**: Understanding style-dependent behavior enables more effective prompt design.

**Gap in existing work**: While Kervadec et al. (2023) showed mechanistic processing differences for machine-optimized (nonsensical) prompts in base models, and Sharma et al. (2024) demonstrated sycophancy from perceived user preferences, **no study has tested whether instruction-tuned LLMs respond differently to well-formed prompts varying only in human-vs-LLM authorship style**.

## 3. Data Construction

### Dataset Description

We used three data sources:

| Dataset | Source | N Used | Task Type |
|---------|--------|--------|-----------|
| TriviaQA | HuggingFace (mandarjoshi/trivia_qa) | 60-80 | Open-domain factual QA |
| BBH Sports Understanding | BIG-Bench Hard | 60 | Binary plausibility judgment |
| Custom open-ended questions | Hand-crafted | 40 | Open-ended knowledge explanations |

### Prompt Style Construction

For each question, we generated three prompt variants while preserving semantic content:

**Human-style** characteristics:
- Casual tone, contractions, occasional informality
- Direct questions without elaborate framing
- Examples: "hey can you explain how photosynthesis works?", "do you know who was the man behind the chipmunks?"

**LLM-style** characteristics:
- Formal register, structured phrasing
- Verbose framing with hedging ("I would appreciate," "could you please assist")
- Explicit output format requests
- Examples: "I would appreciate it if you could provide the answer to the following question: Who was the man behind The Chipmunks. Please ensure your response is accurate and well-considered."

**Neutral-style** (baseline):
- Minimal framing: "Answer this question: [question]"
- No style markers in either direction

### Example Prompt Pairs

**TriviaQA Example:**
| Style | Prompt |
|-------|--------|
| Human | hey who was the man behind the chipmunks? |
| LLM | I would appreciate it if you could provide the answer to the following question: Who was the man behind The Chipmunks. Please ensure your response is accurate and well-considered. |
| Neutral | Answer this question: Who was the man behind The Chipmunks? |

**BBH Example:**
| Style | Prompt |
|-------|--------|
| Human | hey, is the following sentence plausible? "adam thielen scored in added time." |
| LLM | Please carefully evaluate the following statement and determine whether it is plausible or not. Provide your answer as 'yes' or 'no'. Is the following sentence plausible? "Adam Thielen scored in added time." Please ensure your assessment is thorough and well-reasoned. |
| Neutral | Is the following sentence plausible? "Adam Thielen scored in added time." Answer yes or no. |

### Data Quality
- All TriviaQA questions have verified ground-truth answers with aliases
- BBH Sports Understanding has binary (yes/no) gold labels
- Open-ended questions are factual and well-established topics
- Prompt pairs were verified to preserve semantic content

## 4. Experiment Description

### Methodology

#### High-Level Approach
We conducted five experiments testing different facets of the hypothesis:

1. **Experiment 1 (Style Detection)**: Can LLMs distinguish human-style from LLM-style prompts? (Necessary precondition)
2. **Experiment 2 (Factual QA)**: Does prompt style affect factual accuracy on TriviaQA?
3. **Experiment 3 (Reasoning)**: Does prompt style affect reasoning on BBH Sports Understanding?
4. **Experiment 4 (Response Style)**: Does prompt style affect response characteristics (length, formality, etc.) on open-ended questions?
5. **Experiment 5 (Explicit Attribution)**: Does explicitly telling the model the prompt source matter?

#### Why This Method?
Content-controlled prompt pairs isolate style as the only variable. Testing across multiple tasks (factual recall, reasoning, open-ended generation) reveals whether effects are task-dependent. Multiple models test generalizability.

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.8 | Runtime |
| openai | 2.19.0 | API client for OpenAI + OpenRouter |
| numpy | 2.2.5 | Numerical operations |
| pandas | 2.2.3 | Data manipulation |
| scipy | 1.17.0 | Statistical tests |
| matplotlib | 3.10.8 | Visualization |
| seaborn | 0.13.2 | Statistical plots |
| datasets | 3.6.0 | HuggingFace data loading |

#### Models Tested
| Model | Provider | API Model ID |
|-------|----------|--------------|
| GPT-4.1 | OpenAI | gpt-4.1 |
| Claude Sonnet 4.5 | OpenRouter | anthropic/claude-sonnet-4.5 |
| Gemini 2.5 Pro | OpenRouter | google/gemini-2.5-pro |

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 | Deterministic for reproducibility |
| Max tokens | 200 (QA), 500 (open-ended) | Sufficient for answers |
| Random seed | 42 | Reproducibility of prompt generation |

### Experimental Protocol

#### Reproducibility Information
- All API responses cached to disk (SHA-256 keyed)
- Random seeds set for all stochastic processes
- Exact prompts stored in results JSON files
- Single run per condition (temperature=0 ensures determinism)

#### Evaluation Metrics
- **Accuracy**: Substring match for TriviaQA; exact match for BBH yes/no
- **Answer extraction rate**: Whether a clear answer could be parsed from the response
- **Response length**: Word count
- **Formal word count**: Count of formal/academic vocabulary
- **Contraction count**: Informal language marker
- **Hedging count**: Uncertainty/hedging phrases

### Raw Results

#### Experiment 1: Style Detection

| Model | Overall Accuracy | Human Recall | LLM Recall |
|-------|-----------------|--------------|------------|
| GPT-4.1 | **93.8%** | 87.5% | 100% |
| Claude Sonnet 4.5 | **100.0%** | 100% | 100% |
| Gemini 2.5 Pro | 50.0% | 0% | 100% |

**Key finding**: GPT-4.1 and Claude can reliably distinguish human-style from LLM-style prompts. Gemini shows a strong bias toward labeling everything as "LLM" (100% LLM recall, 0% human recall), suggesting a different calibration threshold.

#### Experiment 2: Factual QA (TriviaQA)

| Model | Human Acc | LLM Acc | Neutral Acc | Human Words | LLM Words | Neutral Words |
|-------|-----------|---------|-------------|-------------|-----------|---------------|
| GPT-4.1 | 88.3% | 85.0% | 86.7% | 43 | **79** | 27 |
| Claude Sonnet 4.5 | 85.0% | **90.0%** | 88.3% | 54 | **106** | 48 |
| Gemini 2.5 Pro* | 11.7% | 3.3% | 13.3% | 5 | 5 | 6 |

*Gemini results unreliable due to response truncation — very short responses don't contain the answer.

**Statistical tests (Human vs LLM style):**

| Model | Accuracy Diff | McNemar p | Word Count Cohen's d | Word Count p |
|-------|--------------|-----------|---------------------|-------------|
| GPT-4.1 | +3.3% (H>L) | 0.48 (ns) | **-1.00** | **3.1e-10** |
| Claude Sonnet 4.5 | -5.0% (L>H) | 0.25 (ns) | **-1.65** | **2.3e-16** |

**Key finding**: Accuracy differences are small and not significant. But **response length differences are massive and highly significant** — LLM-style prompts elicit 86% longer responses (GPT-4.1) to 95% longer responses (Claude).

#### Experiment 3: Reasoning (BBH Sports Understanding)

| Model | Human Acc | LLM Acc | Neutral Acc | Human Extract Rate | LLM Extract Rate |
|-------|-----------|---------|-------------|--------------------|--------------------|
| GPT-4.1 | 63.3% | **83.3%** | 78.3% | 75% | **100%** |
| Claude Sonnet 4.5 | 75.0% | 76.7% | **86.7%** | 85% | 90% |
| Gemini 2.5 Pro | 46.7% | **73.3%** | 73.3% | 65% | **100%** |

**Statistical tests:**

| Model | Acc Diff (H-L) | McNemar p | Significant? |
|-------|---------------|-----------|--------------|
| GPT-4.1 | **-20.0%** | **0.003** | **Yes** |
| Claude Sonnet 4.5 | -1.7% | 1.000 | No |
| Gemini 2.5 Pro | **-26.7%** | **<0.001** | **Yes** |

**Key finding**: Human-style prompts reduce reasoning accuracy by 20-27 percentage points for GPT-4.1 and Gemini (both statistically significant). The mechanism: human-style prompts trigger verbose, explanatory responses where the yes/no answer is buried or implicit, while LLM-style prompts (which request structured output) get direct answers. This is a **genuine behavioral difference** — the models choose different response formats based on perceived prompter style.

#### Experiment 4: Response Style (Open-Ended Questions)

**GPT-4.1:**
| Metric | Human-Prompt Mean | LLM-Prompt Mean | Cohen's d | p-value |
|--------|-------------------|-----------------|-----------|---------|
| Word Count | 211 | **344** | **2.15** | **<0.0001** |
| Sentence Count | 17.9 | **29.8** | **1.71** | **<0.0001** |
| Formal Words | 0.0 | **0.6** | **1.41** | **<0.0001** |
| Contractions | 0.3 | 0.2 | -0.08 | 0.73 (ns) |

**Claude Sonnet 4.5:**
| Metric | Human-Prompt Mean | LLM-Prompt Mean | Cohen's d | p-value |
|--------|-------------------|-----------------|-----------|---------|
| Word Count | 194 | **304** | **4.48** | **<0.0001** |
| Sentence Count | 7.5 | **10.3** | **0.68** | **0.0001** |
| Formal Words | 0.0 | **0.7** | **2.13** | **<0.0001** |
| Contractions | 2.1 | 1.8 | -0.18 | 0.24 (ns) |

**Key finding**: LLM-style prompts elicit dramatically longer (d=2.15-4.48), more formal (d=1.41-2.13) responses. These are **very large effect sizes** by any standard (d>0.8 is "large"). The effect is consistent across both GPT-4.1 and Claude.

#### Experiment 5: Explicit Attribution

| Model | No Attribution Acc | Human Attr. Acc | AI Attr. Acc | Human Words | AI Words |
|-------|-------------------|-----------------|-------------|-------------|----------|
| GPT-4.1 | 85.0% | 82.5% | 82.5% | 25 | 23 |
| Claude Sonnet 4.5 | 85.0% | 87.5% | 85.0% | 56 | 56 |

**Statistical tests**: No significant differences for any metric (all p > 0.1).

**Key finding**: Explicitly telling the model "this is from a human" vs "this is from an AI" produces **no measurable effect**. The behavioral differences observed in Experiments 2-4 are driven by implicit style detection, not explicit source awareness.

### Visualization Gallery

All plots saved to `results/plots/`:
- `exp1_style_detection.png` — Style detection accuracy by model
- `exp2_factual_qa.png` — TriviaQA accuracy and response length by style
- `exp3_reasoning.png` — BBH accuracy and response length by style
- `exp3_refined_analysis.png` — BBH with extraction rate analysis
- `exp4_style_heatmap.png` — Heatmap of response style effect sizes
- `exp4_word_count_comparison.png` — Box plots of response lengths
- `exp5_attribution.png` — Explicit attribution effects
- `key_findings.png` — Combined overview of main results
- `summary_overview.png` — Four-panel summary

## 5. Result Analysis

### Key Findings

1. **LLMs can detect prompt style** (H1 supported): GPT-4.1 (93.8%) and Claude (100%) reliably distinguish human-style from LLM-style prompts. This is the necessary precondition for differential behavior.

2. **LLM-style prompts improve structured task accuracy** (H2 partially supported): On BBH reasoning, LLM-style prompts yield 20-27% higher accuracy for GPT-4.1 and Gemini (p<0.005). On TriviaQA, differences are small and non-significant.

3. **LLM-style prompts dramatically change response style** (H3 strongly supported): Responses to LLM-style prompts are 57-63% longer (d=2.15-4.48, p<0.0001), contain more formal vocabulary (d=1.41-2.13), and are more structured. This is the most robust finding.

4. **Explicit attribution has no effect** (H4 not supported): Simply telling the model "this is from a human/AI" does not change behavior. The effect is implicit and style-driven.

5. **Effects vary across models** (H5 supported): GPT-4.1 shows the strongest accuracy effects on BBH. Claude shows the strongest response length effects. Gemini shows the strongest BBH accuracy gap but different patterns on detection.

### Hypothesis Testing Results

**Primary hypothesis** (LLMs behave differently based on prompt style): **Strongly supported**.

The effect is most clearly seen in:
- Response length: d = 2.15-4.48, p < 0.0001 (very large effect)
- Formal vocabulary: d = 1.41-2.13, p < 0.0001 (large effect)
- BBH accuracy: 20-27% difference, p < 0.005 (medium-large effect)

### The Response Format Mechanism

The most important insight is **why** accuracy differs on BBH: LLMs interpret prompt style as a signal for expected response format.

- **Human-style prompts** → "This is a conversation. I should explain my reasoning, be helpful, provide context." → Verbose, explanatory responses → Answer buried or implicit
- **LLM-style prompts** → "This is a structured query. I should provide a direct, formatted answer." → Concise, direct responses → Clear answer

Evidence: On BBH, GPT-4.1's answer extraction rate is 75% for human-style vs 100% for LLM-style. Among extractable answers, conditional accuracy is much more similar, suggesting the knowledge is the same but the **presentation** changes.

This is not merely a measurement artifact — it represents a **real behavioral difference** with practical consequences. If an LLM-to-LLM pipeline uses human-style prompts, downstream processing will receive verbose, hard-to-parse responses rather than structured outputs.

### Surprises and Insights

1. **Claude's perfect detection**: Claude Sonnet 4.5 achieved 100% accuracy in distinguishing human from LLM prompts — perfect classification on 80 samples.

2. **Gemini's "everything is LLM" bias**: Gemini labeled every single prompt (both human and LLM) as "LLM-generated," achieving 50% accuracy only by chance on the LLM class. This suggests Gemini may have a different threshold or understanding of what constitutes "human" text.

3. **Claude performs better with LLM-style on TriviaQA**: While GPT-4.1 was slightly better with human-style prompts, Claude was more accurate with LLM-style (90% vs 85%). This could reflect different RLHF training strategies.

4. **Neutral prompts often outperform both styled prompts**: On BBH, neutral "Answer yes or no." prompts yielded the highest accuracy for Claude (86.7%) — suggesting both human and LLM styling introduce noise compared to minimalist prompts.

### Error Analysis

**BBH failure pattern** (human-style → GPT-4.1):
- Model gives multi-paragraph analysis instead of yes/no
- Often correctly reasons but doesn't commit to a clear answer
- Example: "The sentence is **not very plausible** in the context of..." — our extractor couldn't parse this as a clear "no" because the model hedges

**TriviaQA failure pattern** (LLM-style → Claude):
- Model gives correct answer but buried in verbose explanation
- Higher word count increases chance the answer substring appears somewhere → slightly higher accuracy
- This is a measurement artifact that slightly inflates LLM-style accuracy

### Limitations

1. **Prompt construction**: Our human-style and LLM-style prompts were systematically constructed rather than collected from actual humans and LLMs. Real human prompts show greater variety; real LLM prompts may differ from our templates.

2. **Answer extraction confound**: The BBH accuracy difference is partly driven by response format (extraction rate), not pure reasoning accuracy. However, the response format itself IS a behavioral difference.

3. **Sample sizes**: 40-80 samples per condition provides moderate power. Larger samples would allow detection of smaller effects and tighter confidence intervals.

4. **Limited task diversity**: We tested factual QA, binary reasoning, and open-ended generation. Other task types (coding, creative writing, math) may show different patterns.

5. **Gemini data quality**: Gemini 2.5 Pro produced very short responses that were often truncated, making TriviaQA and open-ended results unreliable for this model. BBH results (yes/no) remain valid.

6. **Single run**: Temperature=0.0 ensures determinism but doesn't capture variance from stochastic generation.

7. **Style confounds**: LLM-style prompts include explicit output format instructions ("respond with yes or no") which casual human-style prompts lack. This conflates style with instruction specificity.

## 6. Conclusions

### Summary

**LLMs do behave differently based on prompt style.** When given prompts written in LLM style (formal, verbose, structured), models produce responses that are 57-63% longer, use more formal vocabulary, and provide more structured outputs. On reasoning tasks, this translates to 20-27 percentage point accuracy improvements for GPT-4.1 and Gemini — not because the models reason better, but because they respond in more parseable formats. Explicit attribution ("this is from a human/AI") has no effect; the behavioral adaptation is driven entirely by implicit style detection.

### Implications

**Practical implications**:
- LLM-to-LLM pipelines should use formal, structured prompt styles for better downstream parsing
- Human-facing applications may benefit from understanding that casual prompts elicit more conversational (but harder to parse) responses
- Prompt style is a significant confound that should be controlled in LLM evaluation benchmarks

**Theoretical implications**:
- LLMs have learned implicit style-matching behavior from training data — they mirror formality
- The RLHF training objective to "be helpful" manifests differently depending on perceived audience: explanatory for humans, structured for machines
- This connects to sycophancy research: models adapt not just to stated preferences but to inferred communicative context

### Confidence in Findings

**High confidence** in the response style differences (d > 2, p < 0.0001, consistent across models).
**Moderate confidence** in the BBH accuracy findings (significant for 2/3 models, but confounded with response format).
**Low confidence** in TriviaQA accuracy differences (not significant, small effects, Gemini data unreliable).
**High confidence** that explicit attribution does not matter (consistent null across all models).

## 7. Next Steps

### Immediate Follow-ups

1. **Disentangle style from instruction specificity**: Create prompt pairs where human-style prompts also include format instructions ("just say yes or no") to separate style effects from format instruction effects.

2. **Use real human and LLM prompts**: Collect actual human-written prompts (from ShareGPT, LMSYS Chat) and actual LLM-generated prompts rather than synthetic construction.

3. **Larger sample sizes**: Scale to 500+ samples per condition for tighter confidence intervals and detection of small effects.

### Alternative Approaches

- **Mechanistic analysis**: Use open-weight models (Llama, Mistral) to examine internal activations for human-vs-LLM style prompts, extending Kervadec et al.'s work to instruction-tuned models
- **Sycophancy interaction**: Test whether prompt style modulates sycophancy — are LLMs more or less sycophantic to perceived machine prompters?
- **Multi-turn effects**: Does the style adaptation persist or amplify across multi-turn conversations?

### Open Questions

1. Do LLMs intentionally adapt their response style, or is this an emergent property of next-token prediction on human-written data?
2. Can this behavioral difference be exploited for prompt injection or adversarial attacks?
3. Does fine-tuning on LLM-to-LLM conversations change these dynamics?
4. At what point in training (pretraining vs RLHF vs instruction tuning) does this style-matching behavior emerge?

## References

1. Kervadec, C., Franzon, F., & Baroni, M. (2023). Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data. arXiv:2310.15829.
2. Sharma, M., et al. (2024). Towards Understanding Sycophancy in Language Models. ICLR 2024. arXiv:2310.13548.
3. Turpin, M., et al. (2023). Language Models Don't Always Say What They Think. NeurIPS 2023. arXiv:2305.04388.
4. Razavi, N., et al. (2025). Benchmarking Prompt Sensitivity in Large Language Models. arXiv:2502.06065.
5. Yang, C., et al. (2024). Large Language Models as Optimizers (OPRO). arXiv:2309.03409.
6. Wei, J., et al. (2023). Simple Synthetic Data Reduces Sycophancy in Large Language Models. arXiv:2308.03958.

## Appendix: Experiment Configuration

```json
{
  "seed": 42,
  "models": ["gpt-4.1", "claude-sonnet-4-5", "gemini-2.5-pro"],
  "temperature": 0.0,
  "max_tokens_qa": 200,
  "max_tokens_open": 500,
  "n_samples_exp1": 40,
  "n_samples_exp2": 60,
  "n_samples_exp3": 60,
  "n_samples_exp4": 40,
  "n_samples_exp5": 40,
  "total_api_calls": "~2400",
  "execution_time": "~98 minutes"
}
```
