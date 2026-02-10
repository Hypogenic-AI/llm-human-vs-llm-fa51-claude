# Do LLMs Behave Differently When the Prompter Is Human vs Another LLM?

## Overview

This project investigates whether large language models exhibit different behaviors when receiving prompts written in human style versus LLM style, with semantic content held constant. We tested GPT-4.1, Claude Sonnet 4.5, and Gemini 2.5 Pro across five experiments covering style detection, factual QA, reasoning, response style, and explicit attribution.

## Key Findings

- **LLMs can detect prompt style**: GPT-4.1 (93.8%) and Claude (100%) reliably classify prompts as human- or LLM-written.
- **LLM-style prompts dramatically increase response length**: 57-63% longer responses on open-ended questions (Cohen's d = 2.15-4.48, p < 0.0001).
- **LLM-style prompts improve structured task accuracy**: 20-27 percentage point improvement on BBH reasoning for GPT-4.1 and Gemini (p < 0.005), because models give direct answers instead of verbose explanations.
- **Response formality increases with LLM-style prompts**: More formal vocabulary (d = 1.41-2.13), more sentences, more structured formatting.
- **Explicit attribution has no effect**: Telling the model "this is from a human/AI" produces no measurable behavioral change.

## Reproduction

### Environment Setup
```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install openai httpx numpy pandas scipy matplotlib seaborn datasets tqdm
```

### Required API Keys
```bash
export OPENAI_API_KEY="your-key"
export OPENROUTER_KEY="your-key"
```

### Run Experiments
```bash
source .venv/bin/activate
python src/run_experiments.py    # ~90 min, ~$30 in API costs
python src/analyze_results.py   # Generate plots and statistics
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Pre-gathered literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── api_client.py            # Unified API client (OpenAI + OpenRouter)
│   ├── prompt_generator.py      # Human/LLM/neutral prompt generation
│   ├── run_experiments.py       # Main experiment runner (5 experiments)
│   └── analyze_results.py       # Statistical analysis and visualization
├── results/
│   ├── config.json              # Experiment configuration
│   ├── exp1_style_detection.json
│   ├── exp2_factual_qa.json
│   ├── exp3_reasoning.json
│   ├── exp4_response_style.json
│   ├── exp5_attribution.json
│   ├── api_cache/               # Cached API responses
│   └── plots/                   # All generated visualizations
├── datasets/                    # Pre-downloaded datasets
├── papers/                      # Downloaded research papers
└── code/                        # Cloned reference repositories
```

## Citation

This research builds on work by Kervadec et al. (2023), Sharma et al. (2024), Turpin et al. (2023), Razavi et al. (2025), and Yang et al. (2024). See REPORT.md for full references.
