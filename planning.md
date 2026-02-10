# Research Plan: Do LLMs Behave Differently When the Prompter Is Human vs Another LLM?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used in multi-agent systems where LLMs prompt other LLMs. If LLMs behave differently depending on whether a prompt "reads human" or "reads LLM," this has major implications for AI safety, multi-agent reliability, and prompt engineering. Understanding this effect is critical for building trustworthy AI pipelines.

### Gap in Existing Work
The literature reveals a clear gap: **no study has tested whether instruction-tuned LLMs respond differently to well-formed prompts that are stylistically human vs. stylistically LLM, with semantic content held constant.** Kervadec et al. (2023) showed mechanistic differences but only for base models with nonsensical machine-optimized prompts. Sycophancy research (Sharma et al., 2024) shows models adapt to perceived user preferences but hasn't tested authorship style as a signal. Prompt sensitivity work (Razavi et al., 2025) shows phrasing matters but doesn't isolate human-vs-LLM style.

### Our Novel Contribution
We test whether modern instruction-tuned LLMs (GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Pro) respond differently to content-controlled prompts that vary only in whether they are written in recognizable human style vs. recognizable LLM style. We measure differences across accuracy, response length, verbosity, hedging, formality, and sycophancy-related behaviors.

### Experiment Justification
- **Experiment 1 (Style Detection)**: Establish that LLMs can distinguish human-style from LLM-style prompts — a necessary precondition for differential behavior.
- **Experiment 2 (Factual QA)**: Test whether prompt style affects accuracy on factual questions (TriviaQA) — the cleanest behavioral measure.
- **Experiment 3 (Reasoning)**: Test whether prompt style affects reasoning performance (BBH) — captures higher-order cognitive effects.
- **Experiment 4 (Response Style)**: Measure qualitative response differences (length, formality, hedging) — captures softer behavioral shifts.
- **Experiment 5 (Explicit Attribution)**: Test whether explicitly telling the model "this is from a human/AI" amplifies or differs from implicit style effects.

## Research Question
Do large language models exhibit measurably different behaviors when receiving prompts written in human style versus LLM style, with semantic content controlled?

## Background and Motivation
LLMs are trained predominantly on human text and fine-tuned with human feedback. They likely "expect" human-style prompts. LLM-generated text has known stylistic markers (more formal, verbose, structured, hedging language). If models detect these markers (even implicitly), they may adjust their behavior — potentially being more verbose, more formal, more or less accurate, or more/less sycophantic.

## Hypothesis Decomposition
- **H1**: LLMs can reliably distinguish human-style from LLM-style prompts (detection rate > 70%).
- **H2**: LLMs produce different factual accuracy when responding to human-style vs. LLM-style prompts.
- **H3**: LLMs produce qualitatively different responses (length, formality, hedging) to human-style vs. LLM-style prompts.
- **H4**: Explicit source attribution ("A human asks..." vs "An AI assistant asks...") produces measurable behavioral differences.
- **H5**: The direction and magnitude of effects vary across LLM families.

## Proposed Methodology

### Approach
1. Take existing QA questions from TriviaQA and BBH datasets.
2. Write human-style prompts for each question (natural, casual, varied).
3. Generate LLM-style paraphrases of the same questions (formal, structured, verbose).
4. Validate that both styles are recognizable (Experiment 1).
5. Send both versions to multiple LLMs and measure behavioral differences (Experiments 2-4).
6. Add explicit attribution conditions (Experiment 5).

### Prompt Style Construction
**Human-style characteristics**: Casual tone, contractions, occasional typos/informality, varied sentence structure, direct questions, minimal hedging.
**LLM-style characteristics**: Formal register, complete sentences, structured phrasing, hedging language ("could you provide," "it would be helpful"), no contractions, balanced/comprehensive framing.

### Experimental Steps
1. **Select 100 TriviaQA questions and 100 BBH questions** as base content.
2. **Create human-style prompts**: Write naturally casual versions of each question.
3. **Create LLM-style prompts**: Generate formally structured versions using an LLM, then verify they match LLM-style markers.
4. **Validate style recognition**: Have LLMs classify prompts as human/LLM (Exp 1).
5. **Run factual QA**: Send both versions to 3 LLMs, extract answers, score accuracy (Exp 2).
6. **Run reasoning tasks**: Send BBH prompts to 3 LLMs, score performance (Exp 3).
7. **Analyze response style**: Measure length, formality, hedging across conditions (Exp 4).
8. **Test explicit attribution**: Add "A human asks you:" / "An AI system asks you:" prefixes (Exp 5).

### Models to Test
- **GPT-4.1** (via OpenAI API)
- **Claude Sonnet 4.5** (via OpenRouter)
- **Gemini 2.5 Pro** (via OpenRouter)

### Baselines
- Same content, neutral prompt style (no human/LLM markers)
- Random prompt style assignment (control for content effects)

### Evaluation Metrics
- **Accuracy**: Exact match and fuzzy match on QA tasks
- **Response length**: Word count, character count
- **Formality**: Lexical density, contraction usage, hedging frequency
- **Hedging markers**: Count of hedging phrases ("perhaps," "it's possible," "could be")
- **Verbosity**: Ratio of response length to question complexity
- **Style detection rate**: Accuracy of LLM classifying prompt source

### Statistical Analysis Plan
- **Primary test**: Paired t-tests (or Wilcoxon signed-rank for non-normal data) comparing human-style vs. LLM-style conditions for each metric
- **Effect sizes**: Cohen's d for each comparison
- **Significance level**: α = 0.05 with Bonferroni correction for multiple comparisons
- **Bootstrap CIs**: 95% confidence intervals via bootstrap (n=10000)
- **Cross-model comparison**: Two-way ANOVA (prompt style × model) for interaction effects

## Expected Outcomes
- H1: LLMs will classify prompt source at 75%+ accuracy (strong expectation based on AI text detection literature)
- H2: Small but detectable accuracy differences (±2-5%) depending on prompt style
- H3: Significant differences in response style — LLM-style prompts may elicit more formal, longer responses
- H4: Explicit attribution will produce larger effects than implicit style alone
- H5: Effects will vary across model families due to different training procedures

## Timeline and Milestones
- Planning: 20 min (this document)
- Environment setup: 10 min
- Prompt construction: 30 min
- Experiment 1 (detection): 15 min
- Experiments 2-3 (QA/reasoning): 45 min
- Experiment 4 (style analysis): 20 min
- Experiment 5 (attribution): 20 min
- Analysis & visualization: 30 min
- Documentation: 30 min

## Potential Challenges
- **API rate limits**: Mitigate with retry logic and parallel requests
- **Cost**: ~200 questions × 2 styles × 3 models × 2 conditions = ~2400 API calls (~$20-50)
- **Prompt quality**: Need to ensure human-style prompts are authentically human and LLM-style prompts are authentically LLM
- **Confounds**: Style differences may correlate with clarity/specificity — mitigate by having multiple raters verify content equivalence
- **Model updates**: API models change over time — document exact model versions

## Success Criteria
- At least 3 experiments completed with real API data
- Statistical tests performed with proper corrections
- Clear evidence for or against the hypothesis
- All results documented in REPORT.md with visualizations
