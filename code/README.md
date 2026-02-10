# Cloned Repositories

## Repo 1: sycophancy-eval
- **URL**: https://github.com/meg-tong/sycophancy-eval
- **Purpose**: Evaluation framework for measuring sycophancy in AI assistants. Includes evaluation datasets and utility functions.
- **Location**: `code/sycophancy-eval/`
- **Key files**:
  - `datasets/feedback.jsonl` - Feedback sycophancy evaluation data
  - `datasets/answer.jsonl` - Answer sycophancy evaluation data
  - `datasets/are_you_sure.jsonl` - Challenge/swaying evaluation data
  - `utils.py` - Utility functions for running evaluations
  - `example.ipynb` - Example notebook showing evaluation workflow
- **Paper**: Sharma et al. (2024) "Towards Understanding Sycophancy in Language Models" (ICLR 2024)
- **Notes**: Contains the exact evaluation datasets and code used in the sycophancy paper. Can be adapted to test whether human vs. LLM prompt style affects sycophantic behavior.

## Repo 2: OPRO (Optimization by Prompting)
- **URL**: https://github.com/google-deepmind/opro
- **Purpose**: Framework for using LLMs to optimize prompts. Demonstrates that LLM-generated prompts can outperform human-designed ones.
- **Location**: `code/opro/`
- **Key files**:
  - `opro/` - Main source code for the optimization framework
  - `data/` - Task data and example prompts
  - `README.md` - Setup and usage instructions
- **Paper**: Yang et al. (2024) "Large Language Models as Optimizers"
- **Notes**: Can be used to generate LLM-optimized prompts for comparison against human-written ones. The optimized prompts have distinctive non-human stylistic features that may trigger different model behavior.

## Repo 3: prompt-sensitivity
- **URL**: https://github.com/Narabzad/prompt-sensitivity
- **Purpose**: Benchmark for prompt sensitivity prediction task. Includes dataset generation, evaluation baselines, and PromptSET dataset.
- **Location**: `code/prompt-sensitivity/`
- **Key files**:
  - `prompt_set/` - PromptSET dataset and generation code
  - `baselines/` - Baseline methods for prompt sensitivity prediction
  - `generate_variations.sh` - Script for generating prompt variations
  - `requirements.txt` - Dependencies
- **Paper**: Razavi et al. (2025) "Benchmarking Prompt Sensitivity in Large Language Models"
- **Notes**: Provides tools for generating semantically equivalent prompt variations and measuring their effectiveness. Directly useful for generating human-style vs. LLM-style prompt pairs.
