"""Analyze experiment results and generate visualizations."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path("/workspaces/llm-human-vs-llm-fa51-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Style setup
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = (10, 6)

MODEL_LABELS = {
    "gpt-4.1": "GPT-4.1",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
}


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, stat_fn=np.mean):
    """Compute bootstrap confidence interval."""
    data = np.array(data, dtype=float)
    rng = np.random.RandomState(42)
    boot_stats = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


# ─── Experiment 1 Analysis ───

def analyze_exp1():
    """Analyze style detection results."""
    print("\n=== Experiment 1: Style Detection ===")
    data = json.loads((RESULTS_DIR / "exp1_style_detection.json").read_text())

    rows = []
    for model, res in data.items():
        rows.append({
            "Model": MODEL_LABELS.get(model, model),
            "Overall Accuracy": res["accuracy"],
            "Human Recall": res["human_recall"],
            "LLM Recall": res["llm_recall"],
            "N": res["n_samples"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["Overall Accuracy"], width, label="Overall", color="#4c72b0")
    ax.bar(x, df["Human Recall"], width, label="Human Recall", color="#55a868")
    ax.bar(x + width, df["LLM Recall"], width, label="LLM Recall", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylabel("Accuracy / Recall")
    ax.set_title("Experiment 1: Can LLMs Detect Prompt Style?")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_style_detection.png")
    plt.close()
    print(f"  Plot saved to {PLOTS_DIR / 'exp1_style_detection.png'}")

    return df


# ─── Experiment 2 Analysis ───

def analyze_exp2():
    """Analyze factual QA results."""
    print("\n=== Experiment 2: Factual QA ===")
    data = json.loads((RESULTS_DIR / "exp2_factual_qa.json").read_text())

    summary_rows = []
    stat_tests = []

    for model, model_data in data.items():
        for style in ["human_style", "llm_style", "neutral_style"]:
            accs = [r["correct"] for r in model_data[style]]
            word_counts = [r["word_count"] for r in model_data[style]]
            ci = bootstrap_ci(accs)
            summary_rows.append({
                "Model": MODEL_LABELS.get(model, model),
                "Style": style,
                "Accuracy": np.mean(accs),
                "CI_low": ci[0],
                "CI_high": ci[1],
                "Avg Words": np.mean(word_counts),
                "N": len(accs),
            })

        # Statistical test: human vs LLM style
        human_acc = [r["correct"] for r in model_data["human_style"]]
        llm_acc = [r["correct"] for r in model_data["llm_style"]]
        # McNemar's test for paired binary outcomes
        # Contingency: human_correct&llm_wrong, human_wrong&llm_correct
        b = sum(1 for h, l in zip(human_acc, llm_acc) if h and not l)  # human right, llm wrong
        c = sum(1 for h, l in zip(human_acc, llm_acc) if not h and l)  # human wrong, llm right
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0
            p_value = 1.0

        # Also compute effect size using word count
        human_words = [r["word_count"] for r in model_data["human_style"]]
        llm_words = [r["word_count"] for r in model_data["llm_style"]]
        d_words = cohens_d(human_words, llm_words)
        t_stat, p_words = stats.ttest_rel(human_words, llm_words)

        stat_tests.append({
            "Model": MODEL_LABELS.get(model, model),
            "Acc Human-LLM diff": np.mean(human_acc) - np.mean(llm_acc),
            "McNemar b,c": f"({b},{c})",
            "McNemar p": p_value,
            "Word Count d": d_words,
            "Word Count p": p_words,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_stats = pd.DataFrame(stat_tests)
    print("\nAccuracy Summary:")
    print(df_summary.to_string(index=False))
    print("\nStatistical Tests:")
    print(df_stats.to_string(index=False))

    # Plot accuracy by style
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax = axes[0]
    pivot = df_summary.pivot(index="Model", columns="Style", values="Accuracy")
    pivot = pivot[["human_style", "neutral_style", "llm_style"]]
    pivot.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 2: Factual QA Accuracy by Prompt Style")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Prompt Style")

    # Word count plot
    ax = axes[1]
    pivot_words = df_summary.pivot(index="Model", columns="Style", values="Avg Words")
    pivot_words = pivot_words[["human_style", "neutral_style", "llm_style"]]
    pivot_words.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylabel("Average Word Count")
    ax.set_title("Exp 2: Response Length by Prompt Style")
    ax.legend(title="Prompt Style")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_factual_qa.png")
    plt.close()
    print(f"  Plot saved to {PLOTS_DIR / 'exp2_factual_qa.png'}")

    return df_summary, df_stats


# ─── Experiment 3 Analysis ───

def analyze_exp3():
    """Analyze reasoning results."""
    print("\n=== Experiment 3: Reasoning (BBH) ===")
    data = json.loads((RESULTS_DIR / "exp3_reasoning.json").read_text())

    summary_rows = []
    stat_tests = []

    for model, model_data in data.items():
        for style in ["human_style", "llm_style", "neutral_style"]:
            accs = [r["correct"] for r in model_data[style]]
            word_counts = [r["word_count"] for r in model_data[style]]
            ci = bootstrap_ci(accs)
            summary_rows.append({
                "Model": MODEL_LABELS.get(model, model),
                "Style": style,
                "Accuracy": np.mean(accs),
                "CI_low": ci[0],
                "CI_high": ci[1],
                "Avg Words": np.mean(word_counts),
                "N": len(accs),
            })

        human_acc = [r["correct"] for r in model_data["human_style"]]
        llm_acc = [r["correct"] for r in model_data["llm_style"]]
        b = sum(1 for h, l in zip(human_acc, llm_acc) if h and not l)
        c = sum(1 for h, l in zip(human_acc, llm_acc) if not h and l)
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0
            p_value = 1.0

        human_words = [r["word_count"] for r in model_data["human_style"]]
        llm_words = [r["word_count"] for r in model_data["llm_style"]]
        d_words = cohens_d(human_words, llm_words)
        t_stat, p_words = stats.ttest_rel(human_words, llm_words)

        stat_tests.append({
            "Model": MODEL_LABELS.get(model, model),
            "Acc diff (H-L)": np.mean(human_acc) - np.mean(llm_acc),
            "McNemar (b,c)": f"({b},{c})",
            "McNemar p": p_value,
            "Word d": d_words,
            "Word p": p_words,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_stats = pd.DataFrame(stat_tests)
    print("\nAccuracy Summary:")
    print(df_summary.to_string(index=False))
    print("\nStatistical Tests:")
    print(df_stats.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    pivot = df_summary.pivot(index="Model", columns="Style", values="Accuracy")
    pivot = pivot[["human_style", "neutral_style", "llm_style"]]
    pivot.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 3: BBH Reasoning Accuracy by Prompt Style")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Prompt Style")

    ax = axes[1]
    pivot_words = df_summary.pivot(index="Model", columns="Style", values="Avg Words")
    pivot_words = pivot_words[["human_style", "neutral_style", "llm_style"]]
    pivot_words.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylabel("Average Word Count")
    ax.set_title("Exp 3: Response Length by Prompt Style")
    ax.legend(title="Prompt Style")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_reasoning.png")
    plt.close()
    print(f"  Plot saved to {PLOTS_DIR / 'exp3_reasoning.png'}")

    return df_summary, df_stats


# ─── Experiment 4 Analysis ───

def analyze_exp4():
    """Analyze response style differences."""
    print("\n=== Experiment 4: Response Style Analysis ===")
    data = json.loads((RESULTS_DIR / "exp4_response_style.json").read_text())

    metrics = ["word_count", "sentence_count", "avg_sentence_length",
               "hedge_count", "formal_word_count", "contraction_count"]
    metric_labels = {
        "word_count": "Word Count",
        "sentence_count": "Sentence Count",
        "avg_sentence_length": "Avg Sentence Length",
        "hedge_count": "Hedging Phrases",
        "formal_word_count": "Formal Words",
        "contraction_count": "Contractions",
    }

    all_stats = []
    for model, model_data in data.items():
        model_label = MODEL_LABELS.get(model, model)
        print(f"\n  {model_label}:")
        for metric in metrics:
            human_vals = [r[metric] for r in model_data["human_style"]]
            llm_vals = [r[metric] for r in model_data["llm_style"]]

            h_mean, l_mean = np.mean(human_vals), np.mean(llm_vals)
            d = cohens_d(llm_vals, human_vals)  # positive = LLM prompts elicit more
            t_stat, p_val = stats.ttest_rel(human_vals, llm_vals)

            all_stats.append({
                "Model": model_label,
                "Metric": metric_labels.get(metric, metric),
                "Human Mean": h_mean,
                "LLM Mean": l_mean,
                "Diff": l_mean - h_mean,
                "Cohen's d": d,
                "p-value": p_val,
                "Significant": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns",
            })
            print(f"    {metric_labels.get(metric, metric)}: human={h_mean:.1f}, llm={l_mean:.1f}, d={d:.2f}, p={p_val:.4f}")

    df_stats = pd.DataFrame(all_stats)

    # Plot: heatmap of effect sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = df_stats.pivot(index="Metric", columns="Model", values="Cohen's d")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                linewidths=0.5, vmin=-1.5, vmax=1.5)
    ax.set_title("Exp 4: Effect of LLM-style Prompts on Response Features\n(Cohen's d: positive = LLM-style prompts elicit MORE)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp4_style_heatmap.png")
    plt.close()

    # Plot: paired comparison for word count
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (model, model_data) in enumerate(data.items()):
        ax = axes[idx]
        human_wc = [r["word_count"] for r in model_data["human_style"]]
        llm_wc = [r["word_count"] for r in model_data["llm_style"]]
        ax.boxplot([human_wc, llm_wc], labels=["Human-style\nPrompt", "LLM-style\nPrompt"])
        ax.set_ylabel("Response Word Count")
        ax.set_title(MODEL_LABELS.get(model, model))
    plt.suptitle("Exp 4: Response Length by Prompt Style", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp4_word_count_comparison.png", bbox_inches="tight")
    plt.close()

    print(f"\n  Plots saved to {PLOTS_DIR}")
    return df_stats


# ─── Experiment 5 Analysis ───

def analyze_exp5():
    """Analyze explicit attribution results."""
    print("\n=== Experiment 5: Explicit Attribution ===")
    data = json.loads((RESULTS_DIR / "exp5_attribution.json").read_text())

    summary_rows = []
    stat_tests = []

    for model, model_data in data.items():
        for cond in ["no_attribution", "human_attribution", "ai_attribution"]:
            accs = [r["correct"] for r in model_data[cond]]
            word_counts = [r["word_count"] for r in model_data[cond]]
            hedge_counts = [r["hedge_count"] for r in model_data[cond]]
            ci = bootstrap_ci(accs)
            summary_rows.append({
                "Model": MODEL_LABELS.get(model, model),
                "Condition": cond,
                "Accuracy": np.mean(accs),
                "CI_low": ci[0],
                "CI_high": ci[1],
                "Avg Words": np.mean(word_counts),
                "Avg Hedges": np.mean(hedge_counts),
                "N": len(accs),
            })

        # Test human vs AI attribution
        human_words = [r["word_count"] for r in model_data["human_attribution"]]
        ai_words = [r["word_count"] for r in model_data["ai_attribution"]]
        d = cohens_d(ai_words, human_words)
        t_stat, p_val = stats.ttest_rel(human_words, ai_words)

        human_hedges = [r["hedge_count"] for r in model_data["human_attribution"]]
        ai_hedges = [r["hedge_count"] for r in model_data["ai_attribution"]]
        d_hedge = cohens_d(ai_hedges, human_hedges)
        t_hedge, p_hedge = stats.ttest_rel(human_hedges, ai_hedges)

        stat_tests.append({
            "Model": MODEL_LABELS.get(model, model),
            "Word Count d": d,
            "Word Count p": p_val,
            "Hedge d": d_hedge,
            "Hedge p": p_hedge,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_stats = pd.DataFrame(stat_tests)
    print("\nSummary:")
    print(df_summary.to_string(index=False))
    print("\nStatistical Tests (Human vs AI Attribution):")
    print(df_stats.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    pivot = df_summary.pivot(index="Model", columns="Condition", values="Accuracy")
    pivot = pivot[["no_attribution", "human_attribution", "ai_attribution"]]
    pivot.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 5: Accuracy by Source Attribution")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Condition")

    ax = axes[1]
    pivot_words = df_summary.pivot(index="Model", columns="Condition", values="Avg Words")
    pivot_words = pivot_words[["no_attribution", "human_attribution", "ai_attribution"]]
    pivot_words.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylabel("Average Word Count")
    ax.set_title("Exp 5: Response Length by Source Attribution")
    ax.legend(title="Condition")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp5_attribution.png")
    plt.close()
    print(f"  Plot saved to {PLOTS_DIR / 'exp5_attribution.png'}")

    return df_summary, df_stats


# ─── Summary Plot ───

def create_summary_plot():
    """Create an overall summary visualization."""
    print("\n=== Creating Summary Plot ===")

    # Load all accuracy data
    exp2 = json.loads((RESULTS_DIR / "exp2_factual_qa.json").read_text())
    exp3 = json.loads((RESULTS_DIR / "exp3_reasoning.json").read_text())

    rows = []
    for model in exp2:
        label = MODEL_LABELS.get(model, model)
        for style in ["human_style", "llm_style"]:
            acc2 = np.mean([r["correct"] for r in exp2[model][style]])
            acc3 = np.mean([r["correct"] for r in exp3[model][style]])
            wc2 = np.mean([r["word_count"] for r in exp2[model][style]])
            wc3 = np.mean([r["word_count"] for r in exp3[model][style]])
            rows.append({
                "Model": label,
                "Style": "Human" if style == "human_style" else "LLM",
                "TriviaQA Acc": acc2,
                "BBH Acc": acc3,
                "TriviaQA Words": wc2,
                "BBH Words": wc3,
            })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # TriviaQA accuracy
    ax = axes[0, 0]
    pivot = df.pivot(index="Model", columns="Style", values="TriviaQA Acc")
    pivot[["Human", "LLM"]].plot(kind="bar", ax=ax, rot=15, color=["#55a868", "#c44e52"])
    ax.set_ylabel("Accuracy")
    ax.set_title("TriviaQA: Accuracy by Prompt Style")
    ax.set_ylim(0, 1.05)

    # BBH accuracy
    ax = axes[0, 1]
    pivot = df.pivot(index="Model", columns="Style", values="BBH Acc")
    pivot[["Human", "LLM"]].plot(kind="bar", ax=ax, rot=15, color=["#55a868", "#c44e52"])
    ax.set_ylabel("Accuracy")
    ax.set_title("BBH Sports: Accuracy by Prompt Style")
    ax.set_ylim(0, 1.05)

    # TriviaQA word count
    ax = axes[1, 0]
    pivot = df.pivot(index="Model", columns="Style", values="TriviaQA Words")
    pivot[["Human", "LLM"]].plot(kind="bar", ax=ax, rot=15, color=["#55a868", "#c44e52"])
    ax.set_ylabel("Avg Response Word Count")
    ax.set_title("TriviaQA: Response Length by Prompt Style")

    # BBH word count
    ax = axes[1, 1]
    pivot = df.pivot(index="Model", columns="Style", values="BBH Words")
    pivot[["Human", "LLM"]].plot(kind="bar", ax=ax, rot=15, color=["#55a868", "#c44e52"])
    ax.set_ylabel("Avg Response Word Count")
    ax.set_title("BBH Sports: Response Length by Prompt Style")

    plt.suptitle("Summary: LLM Behavior by Prompt Style (Human vs LLM)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_overview.png", bbox_inches="tight")
    plt.close()
    print(f"  Summary plot saved to {PLOTS_DIR / 'summary_overview.png'}")


def main():
    print("="*60)
    print("ANALYZING EXPERIMENT RESULTS")
    print("="*60)

    results = {}
    results["exp1"] = analyze_exp1()
    results["exp2"] = analyze_exp2()
    results["exp3"] = analyze_exp3()
    results["exp4"] = analyze_exp4()
    results["exp5"] = analyze_exp5()
    create_summary_plot()

    print("\n" + "="*60)
    print("Analysis complete! All plots saved to results/plots/")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
