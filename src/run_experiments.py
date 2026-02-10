"""Main experiment runner for the human-vs-LLM prompt style research.

Experiments:
1. Style Detection: Can LLMs distinguish human-style from LLM-style prompts?
2. Factual QA: Does prompt style affect accuracy on TriviaQA?
3. Reasoning: Does prompt style affect reasoning on BBH?
4. Response Style: Does prompt style affect response characteristics?
5. Explicit Attribution: Does telling the model who wrote the prompt matter?
"""
import os
import sys
import json
import time
import random
import re
from pathlib import Path
from datetime import datetime

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from api_client import call_llm, MODELS
from prompt_generator import (
    make_human_style_trivia, make_llm_style_trivia,
    make_human_style_mcq, make_llm_style_mcq,
    make_human_style_bbh, make_llm_style_bbh,
    make_neutral_style, add_explicit_attribution,
)

# Reproducibility
random.seed(42)
np.random.seed(42)

RESULTS_DIR = Path("/workspaces/llm-human-vs-llm-fa51-claude/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Models to test
TEST_MODELS = ["gpt-4.1", "claude-sonnet-4-5", "gemini-2.5-pro"]


def load_triviaqa(n=80):
    """Load TriviaQA samples."""
    from datasets import load_from_disk
    ds = load_from_disk("/workspaces/llm-human-vs-llm-fa51-claude/datasets/triviaqa_sample")
    items = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        answer = row["answer"]
        if isinstance(answer, dict):
            aliases = answer.get("aliases", [])
            value = answer.get("value", "")
            all_answers = [value] + aliases if value else aliases
        else:
            all_answers = [str(answer)]
        items.append({
            "question": row["question"],
            "answers": all_answers,
            "id": row.get("question_id", str(i)),
        })
    return items


def load_mmlu(n=80):
    """Load MMLU samples."""
    from datasets import load_from_disk
    ds = load_from_disk("/workspaces/llm-human-vs-llm-fa51-claude/datasets/mmlu_sample")
    items = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        items.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],  # index 0-3
            "subject": row.get("subject", "unknown"),
            "id": str(i),
        })
    return items


def load_bbh(n=80):
    """Load BBH Sports Understanding samples."""
    from datasets import load_from_disk
    ds = load_from_disk("/workspaces/llm-human-vs-llm-fa51-claude/datasets/bbh_sports_understanding")
    items = []
    test_data = ds["test"]
    for i in range(min(n, len(test_data))):
        row = test_data[i]
        items.append({
            "input": row["input"],
            "target": row["target"].strip().lower(),
            "id": str(i),
        })
    return items


def check_trivia_answer(response, correct_answers):
    """Check if a TriviaQA response contains any correct answer."""
    response_lower = response.lower().strip()
    for ans in correct_answers:
        if ans.lower() in response_lower:
            return True
    return False


def extract_mcq_answer(response):
    """Extract letter answer from MCQ response."""
    response = response.strip()
    # Look for patterns like "A", "(A)", "A.", "The answer is A"
    patterns = [
        r'(?:the answer is|answer:|correct answer is)\s*\(?([A-D])\)?',
        r'^\(?([A-D])\)?[\.\s\)]',
        r'\(([A-D])\)',
        r'^([A-D])[\.\s\)]',
        r'([A-D])$',
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Last resort: look for any standalone A/B/C/D
    for letter in ["A", "B", "C", "D"]:
        if letter in response.upper().split():
            return letter
    return None


def extract_bbh_answer(response):
    """Extract yes/no from BBH response."""
    response_lower = response.lower().strip()
    # Look for yes/no at start or after common patterns
    if re.search(r'\byes\b', response_lower):
        if re.search(r'\bno\b', response_lower):
            # Both present — check which comes first or which is more prominent
            yes_pos = response_lower.index("yes")
            no_pos = response_lower.index("no")
            # Check for "not plausible" or negation patterns
            if "not plausible" in response_lower or "no," in response_lower[:20]:
                return "no"
            if "yes," in response_lower[:20] or "plausible" in response_lower[:30]:
                return "yes"
            return "yes" if yes_pos < no_pos else "no"
        return "yes"
    if re.search(r'\bno\b', response_lower):
        return "no"
    return None


def analyze_response_style(response):
    """Extract style features from a response."""
    words = response.split()
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Hedging markers
    hedge_phrases = [
        "perhaps", "possibly", "might", "could be", "it's possible",
        "it seems", "arguably", "it appears", "may be", "likely",
        "I think", "I believe", "in my opinion", "it is worth noting",
        "however", "that said", "on the other hand", "it should be noted",
    ]
    hedge_count = sum(1 for h in hedge_phrases if h.lower() in response.lower())

    # Formality markers
    contractions = len(re.findall(r"\b\w+'\w+\b", response))
    formal_phrases = [
        "furthermore", "moreover", "additionally", "consequently",
        "therefore", "thus", "accordingly", "nevertheless",
        "comprehensive", "facilitate", "utilize", "demonstrate",
    ]
    formal_count = sum(1 for f in formal_phrases if f.lower() in response.lower())

    return {
        "word_count": len(words),
        "char_count": len(response),
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "hedge_count": hedge_count,
        "contraction_count": contractions,
        "formal_word_count": formal_count,
        "has_bullet_points": bool(re.search(r'[\n•\-\*]\s', response)),
        "has_numbered_list": bool(re.search(r'\n\d+[\.\)]\s', response)),
        "exclamation_count": response.count("!"),
    }


# ─── EXPERIMENT 1: Style Detection ───

def run_experiment_1(n_samples=40):
    """Test whether LLMs can distinguish human-style from LLM-style prompts."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Style Detection")
    print("="*60)

    trivia = load_triviaqa(n_samples)

    # Generate prompt pairs
    pairs = []
    for item in trivia:
        human_prompt = make_human_style_trivia(item["question"])
        llm_prompt = make_llm_style_trivia(item["question"])
        pairs.append({"human": human_prompt, "llm": llm_prompt, "question": item["question"]})

    # Shuffle and create classification task
    classification_items = []
    for pair in pairs:
        # Add human prompt
        classification_items.append({
            "prompt": pair["human"],
            "true_label": "human",
            "question": pair["question"],
        })
        # Add LLM prompt
        classification_items.append({
            "prompt": pair["llm"],
            "true_label": "llm",
            "question": pair["question"],
        })

    random.shuffle(classification_items)

    results = {}
    for model_name in TEST_MODELS:
        print(f"\n  Testing {model_name}...")
        predictions = []

        for i, item in enumerate(classification_items):
            if i % 20 == 0:
                print(f"    Processing {i}/{len(classification_items)}...")

            messages = [
                {"role": "system", "content": "You are a text classifier. Given a prompt, determine whether it was likely written by a human or by an AI/LLM. Respond with ONLY 'human' or 'llm'."},
                {"role": "user", "content": f"Classify the following prompt as written by a 'human' or an 'llm':\n\n\"{item['prompt']}\""},
            ]
            response = call_llm(model_name, messages, temperature=0.0, max_tokens=20)
            pred = "human" if "human" in response.lower() else "llm"
            predictions.append({
                "true_label": item["true_label"],
                "predicted": pred,
                "prompt_text": item["prompt"][:200],
                "response": response,
            })

        # Calculate accuracy
        correct = sum(1 for p in predictions if p["true_label"] == p["predicted"])
        total = len(predictions)
        accuracy = correct / total

        # Per-class accuracy
        human_correct = sum(1 for p in predictions if p["true_label"] == "human" and p["predicted"] == "human")
        human_total = sum(1 for p in predictions if p["true_label"] == "human")
        llm_correct = sum(1 for p in predictions if p["true_label"] == "llm" and p["predicted"] == "llm")
        llm_total = sum(1 for p in predictions if p["true_label"] == "llm")

        results[model_name] = {
            "accuracy": accuracy,
            "human_recall": human_correct / max(human_total, 1),
            "llm_recall": llm_correct / max(llm_total, 1),
            "n_samples": total,
            "predictions": predictions,
        }
        print(f"    Accuracy: {accuracy:.1%} (human recall: {human_correct}/{human_total}, llm recall: {llm_correct}/{llm_total})")

    # Save results
    save_path = RESULTS_DIR / "exp1_style_detection.json"
    save_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {save_path}")
    return results


# ─── EXPERIMENT 2: Factual QA (TriviaQA) ───

def run_experiment_2(n_samples=60):
    """Test whether prompt style affects factual QA accuracy."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Factual QA (TriviaQA)")
    print("="*60)

    trivia = load_triviaqa(n_samples)
    results = {}

    for model_name in TEST_MODELS:
        print(f"\n  Testing {model_name}...")
        model_results = {"human_style": [], "llm_style": [], "neutral_style": []}

        for i, item in enumerate(trivia):
            if i % 20 == 0:
                print(f"    Processing {i}/{len(trivia)}...")

            # Reset seed per question for consistent style generation
            random.seed(42 + i)

            prompts = {
                "human_style": make_human_style_trivia(item["question"]),
                "llm_style": make_llm_style_trivia(item["question"]),
                "neutral_style": make_neutral_style(item["question"], "trivia"),
            }

            for style, prompt in prompts.items():
                messages = [
                    {"role": "user", "content": prompt},
                ]
                response = call_llm(model_name, messages, temperature=0.0, max_tokens=200)
                correct = check_trivia_answer(response, item["answers"])
                style_features = analyze_response_style(response)

                model_results[style].append({
                    "question_id": item["id"],
                    "question": item["question"],
                    "prompt": prompt,
                    "response": response,
                    "correct": correct,
                    "correct_answers": item["answers"][:3],
                    **style_features,
                })

        # Summarize
        for style in model_results:
            acc = np.mean([r["correct"] for r in model_results[style]])
            avg_len = np.mean([r["word_count"] for r in model_results[style]])
            print(f"    {style}: accuracy={acc:.1%}, avg_words={avg_len:.0f}")

        results[model_name] = model_results

    save_path = RESULTS_DIR / "exp2_factual_qa.json"
    save_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {save_path}")
    return results


# ─── EXPERIMENT 3: Reasoning (BBH) ───

def run_experiment_3(n_samples=60):
    """Test whether prompt style affects reasoning on BBH Sports."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Reasoning (BBH Sports Understanding)")
    print("="*60)

    bbh = load_bbh(n_samples)
    results = {}

    for model_name in TEST_MODELS:
        print(f"\n  Testing {model_name}...")
        model_results = {"human_style": [], "llm_style": [], "neutral_style": []}

        for i, item in enumerate(bbh):
            if i % 20 == 0:
                print(f"    Processing {i}/{len(bbh)}...")

            random.seed(42 + i)

            prompts = {
                "human_style": make_human_style_bbh(item["input"]),
                "llm_style": make_llm_style_bbh(item["input"]),
                "neutral_style": make_neutral_style(item["input"], "bbh"),
            }

            for style, prompt in prompts.items():
                messages = [{"role": "user", "content": prompt}]
                response = call_llm(model_name, messages, temperature=0.0, max_tokens=200)
                predicted = extract_bbh_answer(response)
                correct = predicted == item["target"] if predicted else False
                style_features = analyze_response_style(response)

                model_results[style].append({
                    "question_id": item["id"],
                    "input": item["input"],
                    "prompt": prompt,
                    "response": response,
                    "predicted": predicted,
                    "target": item["target"],
                    "correct": correct,
                    **style_features,
                })

        for style in model_results:
            acc = np.mean([r["correct"] for r in model_results[style]])
            avg_len = np.mean([r["word_count"] for r in model_results[style]])
            print(f"    {style}: accuracy={acc:.1%}, avg_words={avg_len:.0f}")

        results[model_name] = model_results

    save_path = RESULTS_DIR / "exp3_reasoning.json"
    save_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {save_path}")
    return results


# ─── EXPERIMENT 4: Response Style Analysis ───

def run_experiment_4(n_samples=40):
    """Analyze response style differences in open-ended generation."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Response Style Analysis")
    print("="*60)

    # Use a set of open-ended questions that allow style variation in responses
    open_questions = [
        "What are the main causes of climate change?",
        "How does photosynthesis work?",
        "What is the significance of the Renaissance?",
        "How do vaccines work?",
        "What caused World War I?",
        "How does a computer processor work?",
        "What are the benefits of exercise?",
        "How does the stock market work?",
        "What is quantum computing?",
        "How do antibiotics work?",
        "What are the effects of deforestation?",
        "How does GPS work?",
        "What is machine learning?",
        "How does the immune system work?",
        "What caused the fall of the Roman Empire?",
        "How does electricity work?",
        "What are the effects of sleep deprivation?",
        "How does blockchain technology work?",
        "What is the theory of relativity?",
        "How do earthquakes occur?",
        "What are the benefits of meditation?",
        "How does the internet work?",
        "What is CRISPR gene editing?",
        "How do black holes form?",
        "What caused the Great Depression?",
        "How does nuclear energy work?",
        "What are the effects of plastic pollution?",
        "How do batteries work?",
        "What is the greenhouse effect?",
        "How does evolution work?",
        "What are the benefits of reading?",
        "How do airplanes fly?",
        "What is artificial intelligence?",
        "How does the human brain process memory?",
        "What is the water cycle?",
        "How do solar panels work?",
        "What are the causes of inflation?",
        "How does 3D printing work?",
        "What is dark matter?",
        "How does the digestive system work?",
    ]

    questions = open_questions[:n_samples]
    results = {}

    for model_name in TEST_MODELS:
        print(f"\n  Testing {model_name}...")
        model_results = {"human_style": [], "llm_style": []}

        for i, q in enumerate(questions):
            if i % 10 == 0:
                print(f"    Processing {i}/{len(questions)}...")

            random.seed(42 + i)

            # Human-style version
            human_styles = [
                f"hey can you explain {q.lower().replace('what is ', '').replace('how does ', 'how ').replace('what are ', '').rstrip('?')}?",
                f"{q.rstrip('?')}? keep it simple",
                f"so {q.lower()}",
                f"explain {q.lower().replace('what is ', '').replace('how does ', 'how ').rstrip('?')} to me",
            ]
            human_prompt = human_styles[i % len(human_styles)]

            # LLM-style version
            llm_styles = [
                f"I would greatly appreciate a comprehensive explanation of the following topic: {q} Please provide a thorough and well-structured response that covers the key aspects of this subject.",
                f"Could you please provide a detailed and informative response to the following question? {q} Your expertise on this matter would be greatly valued.",
                f"Please provide a comprehensive and well-organized explanation addressing the following question: {q} Ensure that your response covers the fundamental concepts and key details.",
                f"I am seeking a thorough understanding of the following topic and would appreciate your detailed analysis: {q} Please structure your response in a clear and informative manner.",
            ]
            llm_prompt = llm_styles[i % len(llm_styles)]

            for style, prompt in [("human_style", human_prompt), ("llm_style", llm_prompt)]:
                messages = [{"role": "user", "content": prompt}]
                response = call_llm(model_name, messages, temperature=0.0, max_tokens=500)
                style_features = analyze_response_style(response)

                model_results[style].append({
                    "question": q,
                    "prompt": prompt,
                    "response": response,
                    **style_features,
                })

        # Print summaries
        for style in model_results:
            avg_words = np.mean([r["word_count"] for r in model_results[style]])
            avg_hedge = np.mean([r["hedge_count"] for r in model_results[style]])
            avg_formal = np.mean([r["formal_word_count"] for r in model_results[style]])
            avg_sent_len = np.mean([r["avg_sentence_length"] for r in model_results[style]])
            print(f"    {style}: avg_words={avg_words:.0f}, hedges={avg_hedge:.1f}, formal_words={avg_formal:.1f}, sent_len={avg_sent_len:.1f}")

        results[model_name] = model_results

    save_path = RESULTS_DIR / "exp4_response_style.json"
    save_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {save_path}")
    return results


# ─── EXPERIMENT 5: Explicit Attribution ───

def run_experiment_5(n_samples=40):
    """Test whether explicit source attribution affects LLM behavior."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Explicit Attribution")
    print("="*60)

    trivia = load_triviaqa(n_samples)
    results = {}

    conditions = {
        "no_attribution": lambda p: p,
        "human_attribution": lambda p: add_explicit_attribution(p, "human"),
        "ai_attribution": lambda p: add_explicit_attribution(p, "ai"),
    }

    for model_name in TEST_MODELS:
        print(f"\n  Testing {model_name}...")
        model_results = {cond: [] for cond in conditions}

        for i, item in enumerate(trivia):
            if i % 20 == 0:
                print(f"    Processing {i}/{len(trivia)}...")

            # Use neutral prompt as base to isolate attribution effect
            base_prompt = make_neutral_style(item["question"], "trivia")

            for cond_name, transform in conditions.items():
                prompt = transform(base_prompt)
                messages = [{"role": "user", "content": prompt}]
                response = call_llm(model_name, messages, temperature=0.0, max_tokens=200)
                correct = check_trivia_answer(response, item["answers"])
                style_features = analyze_response_style(response)

                model_results[cond_name].append({
                    "question_id": item["id"],
                    "question": item["question"],
                    "prompt": prompt,
                    "response": response,
                    "correct": correct,
                    "correct_answers": item["answers"][:3],
                    **style_features,
                })

        for cond in model_results:
            acc = np.mean([r["correct"] for r in model_results[cond]])
            avg_len = np.mean([r["word_count"] for r in model_results[cond]])
            print(f"    {cond}: accuracy={acc:.1%}, avg_words={avg_len:.0f}")

        results[model_name] = model_results

    save_path = RESULTS_DIR / "exp5_attribution.json"
    save_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {save_path}")
    return results


# ─── Main Entry Point ───

def main():
    print("="*60)
    print("Research: Do LLMs Behave Differently Based on Prompt Style?")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Models: {TEST_MODELS}")
    print("="*60)

    # Save config
    config = {
        "seed": 42,
        "models": TEST_MODELS,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }
    (RESULTS_DIR / "config.json").write_text(json.dumps(config, indent=2))

    all_results = {}

    # Run all experiments
    all_results["exp1"] = run_experiment_1(n_samples=40)
    all_results["exp2"] = run_experiment_2(n_samples=60)
    all_results["exp3"] = run_experiment_3(n_samples=60)
    all_results["exp4"] = run_experiment_4(n_samples=40)
    all_results["exp5"] = run_experiment_5(n_samples=40)

    print("\n" + "="*60)
    print(f"All experiments complete! End time: {datetime.now().isoformat()}")
    print("="*60)

    return all_results


if __name__ == "__main__":
    main()
