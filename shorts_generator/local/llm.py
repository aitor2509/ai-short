"""Local LLM backend — OpenAI, Gemini, Groq, or LM Studio (fully local, free), selected by
LLM_PROVIDER (single) or LLM_PROVIDER_CASCADE (fallback chain)."""
import time
from typing import Optional

from ..config import (
    GEMINI_MODEL,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_PROVIDER_CASCADE,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
    OPENAI_MODEL,
    require_gemini_key,
    require_groq_key,
    require_openai_key,
)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3.0


def _with_retries(fn, *args, **kwargs):
    """Retry transient provider errors (503 overloaded, rate limits, timeouts)
    with exponential backoff. Doesn't retry on auth/config errors (401, 404,
    invalid key) — those need a human to fix the .env, not a retry."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            transient = any(s in msg for s in ("503", "unavailable", "rate limit", "429", "timeout", "overloaded"))
            last_error = e
            if not transient or attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"[llm] transient error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait:.0f}s: {e}", flush=True)
            time.sleep(wait)
    raise last_error  # pragma: no cover


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_groq_llm(prompt: str) -> str:
    """Groq backend used by --mode local when LLM_PROVIDER=groq.

    Groq's API is OpenAI-compatible, so this reuses the same 'openai'
    package — just pointed at Groq's base_url instead of api.openai.com.
    Free tier, very fast inference (LPU hardware, not GPU).
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local (also used for Groq). Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_groq_key(), base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    return response.text or ""


def call_lmstudio_llm(prompt: str) -> str:
    """LM Studio backend used by --mode local when LLM_PROVIDER=lmstudio.

    Fully local, zero cost per call — LM Studio's local server (started
    with `lms server start`) also exposes an OpenAI-compatible API, so this
    reuses the same 'openai' package pointed at localhost instead of a
    cloud endpoint. No real API key needed; any non-empty string works.
    Speed and quality depend entirely on your own hardware.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local (also used for LM Studio). Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key="lm-studio", base_url=LMSTUDIO_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not reach LM Studio at {LMSTUDIO_BASE_URL}. Make sure `lms server start` "
            f"is running and a model is loaded (`lms load {LMSTUDIO_MODEL}`). Original error: {e}"
        ) from e
    return response.choices[0].message.content or ""


_PROVIDER_FUNCS = {
    "openai": call_openai_llm,
    "gemini": call_gemini_llm,
    "groq": call_groq_llm,
    "lmstudio": call_lmstudio_llm,
}

# Error signatures that mean "this provider is out of quota right now" —
# distinct from _with_retries' "transient" list (503/overloaded/timeout),
# which is about a provider being briefly flaky and worth retrying itself.
# A quota error means retrying THIS provider is pointless; only then do we
# advance the cascade to the next one.
_QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "rate_limit", "tokens per day")


def _is_quota_error(e: Exception) -> bool:
    return any(m in str(e).lower() for m in _QUOTA_MARKERS)


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider.

    If LLM_PROVIDER_CASCADE is set (e.g. "groq,gemini,lmstudio"), tries each
    provider in order and only advances to the next one when the current one
    is out of quota — a bad key, a real bug, or an unknown provider name
    still raises immediately instead of silently hopping providers.
    Without LLM_PROVIDER_CASCADE, behaves exactly as before: a single fixed
    provider from LLM_PROVIDER.
    """
    chain = LLM_PROVIDER_CASCADE or [(LLM_PROVIDER or "openai").strip().lower()]

    last_error: Optional[Exception] = None
    for i, provider in enumerate(chain):
        fn = _PROVIDER_FUNCS.get(provider)
        if fn is None:
            raise RuntimeError(
                f"Unknown provider {provider!r} in LLM_PROVIDER/LLM_PROVIDER_CASCADE. "
                f"Use one of: {', '.join(_PROVIDER_FUNCS)}."
            )
        try:
            return _with_retries(fn, prompt)
        except Exception as e:
            last_error = e
            is_last_in_chain = i == len(chain) - 1
            if is_last_in_chain or not _is_quota_error(e):
                raise
            print(f"[llm] {provider} out of quota, falling back to '{chain[i + 1]}': {e}", flush=True)

    raise last_error  # pragma: no cover — loop always returns or raises above
