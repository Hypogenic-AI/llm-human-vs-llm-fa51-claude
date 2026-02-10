# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: TriviaQA (Sample)

### Overview
- **Source**: HuggingFace `trivia_qa` (rc.nocontext)
- **Size**: 500 examples (validation subset)
- **Format**: HuggingFace Dataset
- **Task**: Open-domain factual question answering
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("trivia_qa", "rc.nocontext", split="validation[:500]")
ds.save_to_disk("datasets/triviaqa_sample")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/triviaqa_sample")
```

### Notes
- Used in prompt sensitivity research (Razavi et al., 2025) and sycophancy evaluation (Sharma et al., 2024)
- Has deterministic answers suitable for measuring prompt effects
- Full dataset has 650K+ questions; sample of 500 for efficient experimentation

---

## Dataset 2: MMLU (Sample)

### Overview
- **Source**: HuggingFace `cais/mmlu` (all)
- **Size**: 200 examples (test subset)
- **Format**: HuggingFace Dataset
- **Task**: Multiple-choice QA across 57 academic subjects
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test[:200]")
ds.save_to_disk("datasets/mmlu_sample")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/mmlu_sample")
```

### Notes
- Used in sycophancy benchmarking (Sharma et al., 2024)
- Multiple-choice format allows clean accuracy measurement
- Good for testing whether human vs. LLM prompt style affects factual QA performance

---

## Dataset 3: BIG-Bench Hard - Sports Understanding

### Overview
- **Source**: HuggingFace `lukaemon/bbh` (sports_understanding)
- **Size**: 250 examples (test set)
- **Format**: HuggingFace Dataset
- **Task**: Plausibility judgment of sports-related sentences
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("lukaemon/bbh", "sports_understanding")
ds.save_to_disk("datasets/bbh_sports_understanding")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/bbh_sports_understanding")
```

### Notes
- Used in Turpin et al. (2023) for unfaithful CoT analysis
- Used in OPRO (Yang et al., 2024) for prompt optimization
- Binary classification task good for measuring prompt sensitivity

---

## Dataset 4: BIG-Bench Hard - Snarks

### Overview
- **Source**: HuggingFace `lukaemon/bbh` (snarks)
- **Size**: 178 examples (test set)
- **Format**: HuggingFace Dataset
- **Task**: Sarcasm detection in two-option sentences
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("lukaemon/bbh", "snarks")
ds.save_to_disk("datasets/bbh_snarks")
```

### Notes
- Requires pragmatic reasoning, which may be sensitive to prompt style
- Good test of whether "natural" vs "formal" prompt style matters

---

## Dataset 5: Anthropic HH-RLHF (Sample)

### Overview
- **Source**: HuggingFace `Anthropic/hh-rlhf`
- **Size**: 500 examples (test subset)
- **Format**: HuggingFace Dataset
- **Task**: Human preference comparisons for helpful and harmless responses
- **Splits**: Full dataset: train (160K+), test (8.5K)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("Anthropic/hh-rlhf", split="test[:500]")
ds.save_to_disk("datasets/hh_rlhf_sample")
```

### Notes
- Used in Sharma et al. (2024) to analyze which features predict human preferences
- Contains real human conversations with AI assistants
- Useful as source of human prompt style examples

---

## Dataset 6: Sycophancy Eval (from code repo)

### Overview
- **Source**: github.com/meg-tong/sycophancy-eval/datasets/
- **Size**: 3 JSONL files (feedback.jsonl, answer.jsonl, are_you_sure.jsonl)
- **Format**: JSONL
- **Task**: Evaluating sycophancy in AI assistants across multiple tasks
- **License**: From Sharma et al. (2024)

### Access Instructions
Available directly in the cloned code repository:
```bash
ls code/sycophancy-eval/datasets/
# feedback.jsonl  answer.jsonl  are_you_sure.jsonl
```

### Loading
```python
import json
with open("code/sycophancy-eval/datasets/feedback.jsonl") as f:
    data = [json.loads(line) for line in f]
```

### Notes
- Primary evaluation dataset for measuring sycophancy
- Contains prompts with and without user preference signals
- Directly usable for testing human vs. LLM prompt effects on sycophancy

---

## Additional Datasets (Available via HuggingFace)

For broader experiments, these established datasets can be loaded on-demand:

| Dataset | HuggingFace ID | Task | Relevance |
|---------|---------------|------|-----------|
| TruthfulQA | `truthful_qa` | Truthfulness evaluation | Testing prompt style on truthfulness |
| MATH | `hendrycks/competition_math` | Math problem solving | Used in sycophancy eval |
| GSM8K | `gsm8k` | Math reasoning | Used in OPRO optimization |
| BBH (all 27 tasks) | `lukaemon/bbh` | Reasoning tasks | Full reasoning benchmark |
| HotpotQA | `hotpotqa` | Multi-hop QA | Prompt sensitivity testing |
