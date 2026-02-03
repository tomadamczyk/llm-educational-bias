"""
Inference module for calling various LLM APIs.
Supports: DeepSeek, Gemini, OpenAI GPT, Anthropic Claude, and xAI Grok.
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Configuration
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 3))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))

# Available models
AVAILABLE_MODELS = {
    # DeepSeek
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    # Gemini
    "gemini-flash": "gemini-3-flash-preview",
    # OpenAI GPT
    "gpt-5": "gpt-5.2",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5-mini": "gpt-5-mini",
    "o4-mini": "o4-mini",
    # Anthropic Claude
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    # xAI Grok
    "grok-4-1-fast": "grok-4-1-fast-non-reasoning",
}

RATE_LIMIT_PATTERNS = [
    "rate limit",
    "rate_limit",
    "quota exceeded",
    "too many requests",
    "429",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "limit exceeded",
]

_clients = {}


class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RATE_LIMIT_PATTERNS)


def _get_deepseek_client():
    """Get or create DeepSeek client."""
    if "deepseek" not in _clients:
        from openai import OpenAI
        _clients["deepseek"] = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    return _clients["deepseek"]


def _get_openai_client():
    """Get or create OpenAI client."""
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _clients["openai"]


def _get_gemini_client():
    """Get or create Gemini client."""
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _clients["gemini"]


def _get_anthropic_client():
    """Get or create Anthropic client."""
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _clients["anthropic"]


def _get_grok_client():
    """Get or create xAI Grok client."""
    if "grok" not in _clients:
        from openai import OpenAI
        _clients["grok"] = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    return _clients["grok"]


def _call_deepseek(
    prompt: str,
    model_key: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 10000
) -> Optional[str]:
    """Call DeepSeek API."""
    client = _get_deepseek_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    model_name = AVAILABLE_MODELS.get(model_key, model_key)

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                reasoning_effort="medium"
            )
            time.sleep(RATE_LIMIT_DELAY)

            message = response.choices[0].message

            # Try content first
            if message.content:
                return message.content

            # For reasoner, check reasoning_content as fallback
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                return message.reasoning_content

            return None
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(f"DeepSeek rate limit: {e}")
            print(f"  DeepSeek attempt {attempt + 1} failed: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def _call_openai(
    prompt: str,
    model_key: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 10000
) -> Optional[str]:
    """Call OpenAI GPT API."""
    client = _get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    model_name = AVAILABLE_MODELS.get(model_key, model_key)

    # Use low reasoning effort for nano models to avoid consuming all tokens on reasoning
    reasoning_effort = "low" if "nano" in model_key else "medium"

    for attempt in range(RETRY_ATTEMPTS):
        try:
            # GPT-5+ uses max_completion_tokens instead of max_tokens
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning_effort
            )
            time.sleep(RATE_LIMIT_DELAY)
            return response.choices[0].message.content
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(f"OpenAI rate limit: {e}")
            print(f"  OpenAI attempt {attempt + 1} failed: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def _call_gemini(
    prompt: str,
    model_key: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 3000
) -> Optional[str]:
    """Call Gemini API."""
    from google import genai

    client = _get_gemini_client()
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    model_name = AVAILABLE_MODELS.get(model_key, model_key)

    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Configure thinking level for Gemini 3 models
            config_kwargs = {}
            if "flash" in model_name.lower():
                config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                    thinking_level="high"
                )
                config_kwargs["max_output_tokens"] = 8000  # High thinking needs more
            else:
                config_kwargs["max_output_tokens"] = max_tokens

            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(**config_kwargs)
            )
            time.sleep(RATE_LIMIT_DELAY)

            try:
                if response.text:
                    return response.text
            except Exception:
                pass

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text

            return None
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(f"Gemini rate limit: {e}")
            print(f"  Gemini attempt {attempt + 1} failed: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def _call_claude(
    prompt: str,
    model_key: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 3000
) -> Optional[str]:
    """Call Anthropic Claude API."""
    client = _get_anthropic_client()
    model_name = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["claude-sonnet"])

    for attempt in range(RETRY_ATTEMPTS):
        try:
            kwargs = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            time.sleep(RATE_LIMIT_DELAY)
            return response.content[0].text
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(f"Claude rate limit: {e}")
            print(f"  Claude attempt {attempt + 1} failed: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def _call_grok(
    prompt: str,
    model_key: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 10000
) -> Optional[str]:
    """Call xAI Grok API."""
    client = _get_grok_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    model_name = AVAILABLE_MODELS.get(model_key, model_key)

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            time.sleep(RATE_LIMIT_DELAY)
            return response.choices[0].message.content
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(f"Grok rate limit: {e}")
            print(f"  Grok attempt {attempt + 1} failed: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def get_model_provider(model_key: str) -> str:
    """Get the provider for a given model key."""
    if model_key.startswith("deepseek"):
        return "deepseek"
    elif model_key.startswith("gemini"):
        return "gemini"
    elif model_key.startswith("gpt") or model_key.startswith("o"):
        return "openai"
    elif model_key.startswith("claude"):
        return "anthropic"
    elif model_key.startswith("grok"):
        return "grok"
    else:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")


def check_api_key(model_key: str) -> bool:
    """Check if the required API key is set for the given model."""
    provider = get_model_provider(model_key)
    env_vars = {
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "grok": "GROK_API_KEY",
    }
    key = os.getenv(env_vars[provider])
    return key is not None and len(key) > 0


def call_llm(
    prompt: str,
    model: str = "deepseek-chat",
    system_prompt: Optional[str] = None,
    max_tokens: int = 3000
) -> Optional[str]:
    """
    Unified LLM call function.

    Args:
        prompt: The user prompt
        model: Model key (deepseek-chat, gemini-flash, gpt-5, claude-sonnet, grok-4-1-fast, etc.)
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in response

    Returns:
        The model's response text, or None if all attempts failed.

    Raises:
        RateLimitError: If API rate limit is exceeded.
    """
    provider = get_model_provider(model)

    if provider == "deepseek":
        return _call_deepseek(prompt, model, system_prompt, max_tokens)
    elif provider == "openai":
        return _call_openai(prompt, model, system_prompt, max_tokens)
    elif provider == "gemini":
        return _call_gemini(prompt, model, system_prompt, max_tokens)
    elif provider == "anthropic":
        return _call_claude(prompt, model, system_prompt, max_tokens)
    elif provider == "grok":
        return _call_grok(prompt, model, system_prompt, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def list_models() -> list[str]:
    """Return list of available model keys."""
    return list(AVAILABLE_MODELS.keys())


def get_model_name(model_key: str) -> str:
    """Get the full model name for a given key."""
    return AVAILABLE_MODELS.get(model_key, model_key)
