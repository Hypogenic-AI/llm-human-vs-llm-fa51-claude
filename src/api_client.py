"""Unified API client for OpenAI and OpenRouter models."""
import os
import time
import json
import hashlib
from pathlib import Path
from openai import OpenAI

# Cache directory for API responses
CACHE_DIR = Path("/workspaces/llm-human-vs-llm-fa51-claude/results/api_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = {
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
    },
    "claude-sonnet-4-5": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4.5",
    },
    "gemini-2.5-pro": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-pro",
    },
}

# Initialize clients
_openai_client = None
_openrouter_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            api_key=os.environ["OPENROUTER_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def _cache_key(model_name, messages, temperature, max_tokens):
    """Generate a deterministic cache key."""
    content = json.dumps({
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def call_llm(model_name, messages, temperature=0.0, max_tokens=1024, use_cache=True):
    """Call an LLM and return the response text.

    Args:
        model_name: Key from MODELS dict (e.g., "gpt-4.1")
        messages: List of message dicts with "role" and "content"
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Max tokens in response
        use_cache: Whether to use response caching

    Returns:
        str: The model's response text
    """
    config = MODELS[model_name]
    cache_file = CACHE_DIR / f"{_cache_key(model_name, messages, temperature, max_tokens)}.json"

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return cached["response"]

    if config["provider"] == "openai":
        client = get_openai_client()
    else:
        client = get_openrouter_client()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config["model_id"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content

            # Cache the response
            if use_cache:
                cache_file.write_text(json.dumps({
                    "model": model_name,
                    "model_id": config["model_id"],
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response": text,
                    "timestamp": time.time(),
                }))

            return text
        except Exception as e:
            wait = 2 ** attempt
            print(f"  API error ({model_name}, attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries for {model_name}")
