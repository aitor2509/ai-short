import os

from dotenv import load_dotenv

load_dotenv()

MUAPI_API_KEY = os.getenv("MUAPI_API_KEY", "").strip()
MUAPI_BASE_URL = os.getenv("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")

POLL_INTERVAL_SECONDS = float(os.getenv("MUAPI_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("MUAPI_POLL_TIMEOUT", "600"))

# Local-mode (--mode local) settings — only consulted when running offline.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").strip()
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
# Optional: comma-separated fallback chain, e.g. "groq,gemini,lmstudio". When
# set, call_local_llm() tries each provider in order and only advances to the
# next one on a quota/rate-limit error (429, RESOURCE_EXHAUSTED, "quota",
# "rate limit") — not on any error, so a bad API key or a real bug still
# surfaces instead of silently hopping providers. Leave unset to keep the
# old single-provider behavior (LLM_PROVIDER alone).
LLM_PROVIDER_CASCADE = [
    p.strip().lower() for p in os.getenv("LLM_PROVIDER_CASCADE", "").split(",") if p.strip()
]
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")  # auto / cpu / cuda
# beam_size=5 (faster-whisper default) is accurate but slow on CPU-only hardware
# (no CUDA GPU acceleration). beam_size=1 (greedy decoding) is roughly 3-5x
# faster with only a small accuracy trade-off — the right default when
# device=cpu, since a 2-hour video at beam_size=5 can take 10+ hours.
LOCAL_WHISPER_BEAM_SIZE = int(os.getenv("LOCAL_WHISPER_BEAM_SIZE", "1"))
# 0 = let ctranslate2 auto-detect; explicit value avoids the model silently
# under-using available CPU cores on some Windows setups.
LOCAL_WHISPER_CPU_THREADS = int(os.getenv("LOCAL_WHISPER_CPU_THREADS", str(os.cpu_count() or 4)))
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "output")

# Which engine transcribes audio in --mode local:
#   "whisper" (default) — faster-whisper running on YOUR machine. Free, fully
#      offline, but CPU-bound on non-NVIDIA hardware — a 2-hour video can take
#      hours without a CUDA GPU.
#   "groq"    — Groq's hosted Whisper API. Free tier (2,000 requests/day, no
#      credit card, console.groq.com/keys), and dramatically faster because it
#      runs on Groq's LPU hardware instead of your CPU (roughly 1h of audio in
#      a few seconds). Needs GROQ_API_KEY + internet.
LOCAL_TRANSCRIBE_PROVIDER = os.getenv("LOCAL_TRANSCRIBE_PROVIDER", "whisper").strip().lower()
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
# Groq caps uploads at 25MB per file. Audio is re-encoded to mono 16kHz/48kbps
# before upload (small + exactly what Whisper wants internally), then split
# into chunks of this length so even a multi-hour video never hits that cap.
GROQ_WHISPER_CHUNK_SECONDS = int(os.getenv("GROQ_WHISPER_CHUNK_SECONDS", "900"))

# Module 4 — Google Drive upload (service account, headless).
GDRIVE_SERVICE_ACCOUNT_FILE = os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE", "service-secrets.json")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", "").strip()

# Pre-filter signals (shorts_generator/local/signals.py) — only the comment
# timestamps signal needs this. Free tier: console.cloud.google.com, enable
# "YouTube Data API v3", create an API key (no OAuth needed, just a key).
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

# VAD (Voice Activity Detection) settings for faster-whisper
# Default threshold is 0.5; lower = more sensitive, higher = less sensitive
# Default min_speech_duration_ms is 250ms; increase to avoid tiny false positives
# Default min_silence_duration_ms is 2000ms; increase to avoid splitting mid-sentence
# DISABLED by default because VAD is too aggressive on mixed speech/music content
LOCAL_WHISPER_VAD_FILTER = os.getenv("LOCAL_WHISPER_VAD_FILTER", "false").strip().lower() == "true"
_vad_params_env = os.getenv("LOCAL_WHISPER_VAD_PARAMETERS", "")
if _vad_params_env:
    import json
    LOCAL_WHISPER_VAD_PARAMETERS = json.loads(_vad_params_env)
else:
    # Match faster-whisper defaults when VAD is enabled
    LOCAL_WHISPER_VAD_PARAMETERS = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }


def require_api_key() -> str:
    if not MUAPI_API_KEY:
        raise RuntimeError(
            "MUAPI_API_KEY is not set. Add it to your .env file or export it as an env var."
        )
    return MUAPI_API_KEY


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Local mode needs an OpenAI key for highlight ranking. "
            "Add it to your .env or export it, or switch back to --mode api."
        )
    return OPENAI_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Local mode needs a Gemini key when LLM_PROVIDER=gemini. "
            "Add it to your .env or export it, or switch LLM_PROVIDER back to openai."
        )
    return GEMINI_API_KEY


def require_groq_key() -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Local mode needs a Groq key when LLM_PROVIDER=groq. "
            "Get a free one at console.groq.com/keys, add it to your .env, "
            "or switch LLM_PROVIDER to openai/gemini."
        )
    return GROQ_API_KEY


def require_gdrive_folder_id() -> str:
    if not GDRIVE_FOLDER_ID:
        raise RuntimeError(
            "GDRIVE_FOLDER_ID is not set. Add it to your .env file (see .env.example) "
            "or drop --upload-drive."
        )
    return GDRIVE_FOLDER_ID
