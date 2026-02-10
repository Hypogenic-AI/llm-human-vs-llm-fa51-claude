# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: "Do LLMs behave differently when the prompter is human vs another LLM?" Resources include papers, datasets, and code repositories.

## Papers
Total papers downloaded: 24

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Unnatural Language Processing | Kervadec et al. | 2023 | `papers/2310.15829_*.pdf` | **CORE**: Mechanistic diff between human/machine prompts |
| 2 | Towards Understanding Sycophancy | Sharma et al. | 2024 | `papers/2310.13548_*.pdf` | **CORE**: Sycophancy across 5 AI assistants |
| 3 | Language Models Don't Always Say What They Think | Turpin et al. | 2023 | `papers/2305.04388_*.pdf` | **CORE**: Unfaithful CoT, hidden biasing features |
| 4 | Benchmarking Prompt Sensitivity | Razavi et al. | 2025 | `papers/2502.06065_*.pdf` | **CORE**: PromptSET dataset, sensitivity prediction |
| 5 | OPRO: LLMs as Optimizers | Yang et al. | 2024 | `papers/2309.03409_*.pdf` | **CORE**: LLM-generated prompts outperform human |
| 6 | POSIX: Prompt Sensitivity Index | Razavi et al. | 2024 | `papers/2410.02185_*.pdf` | Quantitative sensitivity measurement |
| 7 | Simple Synthetic Data Reduces Sycophancy | Wei et al. | 2023 | `papers/2308.03958_*.pdf` | Sycophancy as trainable behavior |
| 8 | Accounting for Sycophancy in Uncertainty | - | 2024 | `papers/2410.14746_*.pdf` | Sycophancy and model confidence |
| 9 | TRUTH DECAY: Multi-Turn Sycophancy | - | 2025 | `papers/2503.11656_*.pdf` | Multi-turn sycophancy dynamics |
| 10 | Not Your Typical Sycophant | - | 2026 | `papers/2601.15436_*.pdf` | Novel sycophancy evaluation |
| 11 | Sycophancy: Causes and Mitigations | - | 2024 | `papers/2411.15287_*.pdf` | Survey of sycophancy |
| 12 | Yes-Men to Truth-Tellers | - | 2024 | `papers/2409.01658_*.pdf` | Pinpoint tuning for sycophancy |
| 13 | Sycophancy to Subtlety Pipeline | - | 2024 | `papers/2401.10580_*.pdf` | Subtle sycophancy |
| 14 | Persona is a Double-edged Sword | - | 2024 | `papers/2408.08631_*.pdf` | Persona effects on reasoning |
| 15 | Persona to Personalization Survey | - | 2024 | `papers/2404.18231_*.pdf` | RPLA survey |
| 16 | Two Tales of Persona in LLMs | - | 2024 | `papers/2406.01171_*.pdf` | Persona survey |
| 17 | Multi-Agent Debate | Du et al. | 2023 | `papers/2305.19118_*.pdf` | LLM-to-LLM debate |
| 18 | ChatEval: Multi-Agent Evaluation | Chan et al. | 2023 | `papers/2308.07201_*.pdf` | LLM evaluating LLM |
| 19 | Paraphrasing Evades AI Detectors | Krishna et al. | 2023 | `papers/2303.13408_*.pdf` | Human vs AI text detection |
| 20 | Human-AI Comparative Prompt Sensitivity | - | 2025 | `papers/2504.12408_*.pdf` | Human vs AI prompt comparison |
| 21 | PromptBench: Robustness of LLMs | Zhu et al. | 2023 | `papers/2306.04528_*.pdf` | Prompt robustness framework |
| 22 | Model-Written Evaluations | Perez et al. | 2022 | `papers/2212.09251_*.pdf` | LLM-generated evaluations |
| 23 | Hallucination Snowball | - | 2023 | `papers/2305.13534_*.pdf` | Cascading LLM errors |
| 24 | PIMMUR Principles | Zhou et al. | 2025 | `papers/2501.10868_*.pdf` | LLM society validity |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 6

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TriviaQA | HuggingFace | 500 examples | Open-domain QA | `datasets/triviaqa_sample/` | Factual QA with deterministic answers |
| MMLU | HuggingFace | 200 examples | Multi-choice QA | `datasets/mmlu_sample/` | 57 academic subjects |
| BBH Sports | HuggingFace | 250 examples | Plausibility judgment | `datasets/bbh_sports_understanding/` | Binary classification |
| BBH Snarks | HuggingFace | 178 examples | Sarcasm detection | `datasets/bbh_snarks/` | Pragmatic reasoning |
| HH-RLHF | HuggingFace | 500 examples | Human preferences | `datasets/hh_rlhf_sample/` | Real human-AI conversations |
| Sycophancy Eval | GitHub repo | 3 JSONL files | Sycophancy evaluation | `code/sycophancy-eval/datasets/` | From Sharma et al. (2024) |

See `datasets/README.md` for download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| sycophancy-eval | github.com/meg-tong/sycophancy-eval | Sycophancy evaluation framework | `code/sycophancy-eval/` | Includes eval datasets |
| OPRO | github.com/google-deepmind/opro | LLM prompt optimization | `code/opro/` | Generate LLM-style prompts |
| prompt-sensitivity | github.com/Narabzad/prompt-sensitivity | Prompt sensitivity prediction | `code/prompt-sensitivity/` | PromptSET generation tools |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used Semantic Scholar API with multiple targeted queries covering: sycophancy, prompt sensitivity, machine-generated prompts, multi-agent LLM interaction, persona effects
2. Used arXiv API for complementary searches
3. Cross-referenced citations from key papers to find additional relevant work
4. Focused on papers from 2022-2026 for recency, plus seminal earlier works

### Selection Criteria
- **Core papers**: Directly study human vs. machine prompt differences, or sycophancy/behavioral adaptation
- **Supporting papers**: Provide methodology, datasets, or theoretical grounding
- **Excluded**: Papers only about AI text detection without behavioral implications; papers about prompt engineering without style/source analysis

### Challenges Encountered
- The exact research question (does prompt source identity affect LLM behavior?) has not been directly studied - this IS the research gap
- Sycophancy-eval dataset had parsing issues with HuggingFace loader but data is available in the cloned repo
- LMSYS Chat 1M requires authentication; could be valuable for human prompt style analysis
- Most mechanistic work (Kervadec et al.) used base models only; instruction-tuned models are understudied

### Gaps and Workarounds
- **No direct human-vs-LLM prompt comparison dataset exists**: Will need to construct this by having LLMs paraphrase human prompts and vice versa
- **Sycophancy data focuses on explicit preference signals**: Our research needs more subtle style-based signals
- **Mechanistic analysis tools not available for API-only models**: May need to focus on behavioral (output) analysis for commercial models

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset(s)**:
   - **TriviaQA + BBH** for factual/reasoning tasks (clean accuracy measurement)
   - **Sycophancy Eval** for behavioral adaptation measurement
   - **Custom prompt pairs** (generate using OPRO and prompt-sensitivity tools)

2. **Baseline methods**:
   - Same content, human-written prompt (control)
   - Same content, LLM-paraphrased prompt (treatment 1)
   - Same content, LLM-optimized prompt (treatment 2 - from OPRO)
   - Explicit prompt source disclosure ("A human asks you..." vs "An AI system asks you...")

3. **Evaluation metrics**:
   - Task accuracy (primary)
   - Response length, formality, verbosity
   - Sycophancy rate (using Sharma et al. methodology)
   - Consistency across prompt variations (using POSIX methodology)
   - Stylistic features of responses (lexical diversity, hedging, etc.)

4. **Code to adapt/reuse**:
   - `sycophancy-eval` for sycophancy measurement framework
   - `prompt-sensitivity` for generating prompt variations
   - `opro` for generating LLM-optimized prompts
   - Standard LLM APIs (OpenAI, Anthropic, HuggingFace) for model access

5. **Experimental flow**:
   1. Collect human-written prompts from existing QA datasets
   2. Generate LLM-written equivalents using paraphrasing
   3. Have LLMs classify prompts as human/AI (test detectability)
   4. Send both versions to multiple LLMs, measure behavioral differences
   5. Test with explicit source attribution ("This prompt was written by a human/AI")
   6. Analyze response differences across accuracy, style, and behavioral metrics
