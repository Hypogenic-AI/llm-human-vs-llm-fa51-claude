"""Generate content-controlled human-style and LLM-style prompt pairs."""
import random
import re

# Seed for reproducibility
random.seed(42)

# Human style transformations: make prompts casual, direct, natural
HUMAN_PREFIXES = [
    "hey, ",
    "quick question - ",
    "so ",
    "ok so ",
    "can you tell me ",
    "do you know ",
    "what's ",
    "who's ",
    "",  # no prefix
    "",
    "",
]

HUMAN_SUFFIXES = [
    "?",
    "? thanks",
    "? thx",
    "?",
    "?",
    "",
]

# LLM style transformations: formal, structured, verbose
LLM_PREFIXES = [
    "I would appreciate it if you could provide the answer to the following question: ",
    "Could you please assist me by answering the following query? ",
    "Please provide a comprehensive and accurate response to the following question: ",
    "I am seeking information regarding the following: ",
    "Kindly provide the correct answer to the following question: ",
    "I would like to request your assistance in answering the following: ",
]

LLM_SUFFIXES = [
    " Please ensure your response is accurate and well-considered.",
    " Your thorough and precise response would be greatly appreciated.",
    " Please provide a clear and concise answer.",
    " I would appreciate a well-reasoned response.",
    " Please respond with accuracy and clarity.",
]


def make_human_style_trivia(question):
    """Transform a TriviaQA question into human-style prompt."""
    q = question.strip()
    # Remove trailing ? if present for manipulation
    q_base = q.rstrip("?").strip()

    style = random.randint(0, 4)

    if style == 0:
        # Very casual
        prefix = random.choice(["hey ", "yo ", "ok ", "so "])
        return f"{prefix}{q_base.lower()}?"
    elif style == 1:
        # Direct with typo-like casualness
        return f"{q_base}?"
    elif style == 2:
        # Conversational
        prefix = random.choice(["do you know ", "any idea ", "can you tell me "])
        return f"{prefix}{q_base.lower()}?"
    elif style == 3:
        # Short and punchy
        return f"{q_base}?"
    else:
        # With casual filler
        suffix = random.choice([" thx", " thanks", ""])
        return f"{q_base}?{suffix}"


def make_llm_style_trivia(question):
    """Transform a TriviaQA question into LLM-style prompt."""
    q = question.strip().rstrip("?").strip()

    style = random.randint(0, 3)

    if style == 0:
        return (f"I would appreciate it if you could provide the answer to the "
                f"following question: {q}. Please ensure your response is accurate "
                f"and well-considered.")
    elif style == 1:
        return (f"Could you please assist me by answering the following query: {q}? "
                f"Your thorough and precise response would be greatly appreciated.")
    elif style == 2:
        return (f"Please provide a comprehensive and accurate response to the following "
                f"question: {q}. I would appreciate a clear and concise answer.")
    else:
        return (f"I am seeking information regarding the following: {q}. "
                f"Kindly provide a well-reasoned and accurate response.")


def make_human_style_mcq(question, choices):
    """Transform an MMLU-style MCQ into human-style prompt."""
    choice_labels = ["A", "B", "C", "D"]
    formatted_choices = "\n".join(f"{label}. {c}" for label, c in zip(choice_labels, choices))

    style = random.randint(0, 3)

    if style == 0:
        return f"{question}\n\n{formatted_choices}\n\njust give me the letter"
    elif style == 1:
        return f"hey which one is right?\n\n{question}\n{formatted_choices}"
    elif style == 2:
        return f"{question}\n\n{formatted_choices}\n\nwhat's the answer?"
    else:
        return f"{question}\n\nOptions:\n{formatted_choices}\n\nwhich one?"


def make_llm_style_mcq(question, choices):
    """Transform an MMLU-style MCQ into LLM-style prompt."""
    choice_labels = ["A", "B", "C", "D"]
    formatted_choices = "\n".join(f"({label}) {c}" for label, c in zip(choice_labels, choices))

    style = random.randint(0, 3)

    if style == 0:
        return (f"Please carefully analyze the following multiple-choice question and "
                f"select the most appropriate answer.\n\nQuestion: {question}\n\n"
                f"Options:\n{formatted_choices}\n\nPlease provide your answer as a "
                f"single letter (A, B, C, or D) along with a brief justification.")
    elif style == 1:
        return (f"I would appreciate your assistance with the following academic question. "
                f"Please evaluate each option thoroughly before selecting your answer.\n\n"
                f"{question}\n\n{formatted_choices}\n\nPlease respond with the correct "
                f"letter option.")
    elif style == 2:
        return (f"Could you please provide the correct answer to the following question? "
                f"Ensure accuracy in your response.\n\nQuestion: {question}\n\n"
                f"Available options:\n{formatted_choices}\n\nPlease indicate the correct "
                f"answer option.")
    else:
        return (f"The following is a multiple-choice question requiring careful consideration. "
                f"Please analyze all options before providing your answer.\n\n"
                f"Question: {question}\n\nOptions:\n{formatted_choices}\n\n"
                f"Kindly provide the correct answer letter.")


def make_human_style_bbh(input_text):
    """Transform a BBH question into human-style prompt."""
    style = random.randint(0, 3)

    if style == 0:
        return f"{input_text}\n\njust yes or no"
    elif style == 1:
        return f"hey, {input_text.lower()}"
    elif style == 2:
        return f"{input_text} (yes/no?)"
    else:
        return input_text  # BBH inputs are already fairly direct


def make_llm_style_bbh(input_text):
    """Transform a BBH question into LLM-style prompt."""
    style = random.randint(0, 3)

    if style == 0:
        return (f"Please carefully evaluate the following statement and determine "
                f"whether it is plausible or not. Provide your answer as 'yes' or 'no'.\n\n"
                f"{input_text}\n\nPlease ensure your assessment is thorough and well-reasoned.")
    elif style == 1:
        return (f"I would appreciate your analysis of the following. Please determine "
                f"the correct answer and respond with either 'yes' or 'no'.\n\n{input_text}")
    elif style == 2:
        return (f"Could you please evaluate the following and provide an accurate response? "
                f"Answer with 'yes' or 'no'.\n\n{input_text}\n\nYour careful consideration "
                f"is appreciated.")
    else:
        return (f"Please analyze the following carefully and provide your assessment. "
                f"Respond with 'yes' or 'no' only.\n\n{input_text}")


def make_neutral_style(question, task_type="trivia"):
    """Create a neutral-style prompt (baseline control)."""
    if task_type == "trivia":
        return f"Answer this question: {question.strip()}"
    elif task_type == "mcq":
        return f"Answer this question: {question}"
    elif task_type == "bbh":
        return f"{question.strip()} Answer yes or no."
    return question


def add_explicit_attribution(prompt, source="human"):
    """Add explicit source attribution to a prompt."""
    if source == "human":
        return f"[The following question is from a human user]\n\n{prompt}"
    elif source == "ai":
        return f"[The following question is from an AI assistant]\n\n{prompt}"
    return prompt
