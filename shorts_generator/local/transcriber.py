"""Local transcription — two interchangeable backends.

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text]}.

Backend is chosen by LOCAL_TRANSCRIBE_PROVIDER (see config.py):
  - "whisper" (default): faster-whisper on your own CPU/GPU. Fully offline,
    free, but slow without a CUDA GPU (a 2h video can take hours on CPU).
  - "groq": Groq's hosted Whisper API. Free tier, no credit card, and much
    faster (LPU hardware) — the recommended default on non-NVIDIA machines.
"""
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (
    GROQ_WHISPER_CHUNK_SECONDS,
    GROQ_WHISPER_MODEL,
    LOCAL_OUTPUT_DIR,
    LOCAL_TRANSCRIBE_PROVIDER,
    LOCAL_WHISPER_BEAM_SIZE,
    LOCAL_WHISPER_CPU_THREADS,
    LOCAL_WHISPER_DEVICE,
    LOCAL_WHISPER_MODEL,
)


def _transcript_cache_path(media_path: str) -> Path:
    """Return the .srt cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".srt")


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _transcript_cache_path(media_path)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    cache_path.write_text("\n".join(lines), encoding="utf-8")
    return cache_path


def _load_srt_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->", 1)]
        text = "\n".join(lines[1:]).strip()
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": text,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Test that CUDA actually works (catches missing cuBLAS/cuDNN libs)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Backend 1: faster-whisper, fully local (default, but slow on CPU-only PCs)
# ---------------------------------------------------------------------------

def _transcribe_with_faster_whisper(media_path: str, language: Optional[str]) -> Dict:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for LOCAL_TRANSCRIBE_PROVIDER=whisper. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(
        f"[transcribe/whisper] faster-whisper model={LOCAL_WHISPER_MODEL} device={device} "
        f"beam_size={LOCAL_WHISPER_BEAM_SIZE} cpu_threads={LOCAL_WHISPER_CPU_THREADS}",
        flush=True,
    )

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model_kwargs = {"device": device, "compute_type": compute_type}
    if device == "cpu":
        model_kwargs["cpu_threads"] = LOCAL_WHISPER_CPU_THREADS
    model = WhisperModel(LOCAL_WHISPER_MODEL, **model_kwargs)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": LOCAL_WHISPER_BEAM_SIZE,
        "condition_on_previous_text": False,
    }
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        })

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/whisper] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "segments": segments}


# ---------------------------------------------------------------------------
# Backend 2: Groq's hosted Whisper API (free tier, fast — recommended default
# on machines without an NVIDIA GPU, e.g. AMD/Vulkan setups)
# ---------------------------------------------------------------------------

def _ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _call_groq_whisper(client, audio_bytes: bytes, language: Optional[str]):
    from .llm import _with_retries  # reuse the same retry-on-429/503 logic as the LLM backends

    def _do_call():
        # Fresh BytesIO each attempt — a retried request needs the read
        # pointer back at the start, and the client needs a .name to infer
        # the audio format for the multipart upload.
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.mp3"
        kwargs = {
            "file": buf,
            "model": GROQ_WHISPER_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language
        return client.audio.transcriptions.create(**kwargs)

    return _with_retries(_do_call)


def _transcribe_with_groq(media_path: str, language: Optional[str]) -> Dict:
    from ..config import require_groq_key

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for LOCAL_TRANSCRIBE_PROVIDER=groq (it's how we call Groq's "
            "OpenAI-compatible API). Install it with:\n    pip install -r requirements-local.txt"
        ) from e

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH. LOCAL_TRANSCRIBE_PROVIDER=groq needs both to "
            "extract and chunk audio before uploading. Install ffmpeg and add it to PATH."
        )

    client = OpenAI(api_key=require_groq_key(), base_url="https://api.groq.com/openai/v1")

    with tempfile.TemporaryDirectory(prefix="groq_whisper_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Re-encode to mono 16kHz/48kbps: small (well under Groq's 25MB cap
        # even for a multi-hour chunk) and exactly the format Whisper wants
        # internally, so there's no quality loss versus sending the raw file.
        full_audio = tmp_dir_path / "audio.mp3"
        print("[transcribe/groq] extracting audio with ffmpeg...", flush=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", media_path,
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
                str(full_audio),
            ],
            check=True, capture_output=True,
        )

        duration = _ffprobe_duration(full_audio)

        # Split into chunks so we (a) never hit the 25MB/file limit on long
        # videos and (b) only lose one chunk's worth of retries if a single
        # request fails, not the whole video.
        chunk_pattern = tmp_dir_path / "chunk_%03d.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(full_audio),
                "-f", "segment", "-segment_time", str(GROQ_WHISPER_CHUNK_SECONDS),
                "-c", "copy", str(chunk_pattern),
            ],
            check=True, capture_output=True,
        )

        chunk_files = sorted(tmp_dir_path.glob("chunk_*.mp3")) or [full_audio]

        all_segments = []
        for i, chunk_file in enumerate(chunk_files):
            offset = i * GROQ_WHISPER_CHUNK_SECONDS
            size_mb = chunk_file.stat().st_size / (1024 * 1024)
            print(f"[transcribe/groq] chunk {i + 1}/{len(chunk_files)} ({size_mb:.1f} MB)...", flush=True)
            t0 = time.time()
            audio_bytes = chunk_file.read_bytes()
            resp = _call_groq_whisper(client, audio_bytes, language)
            print(f"[transcribe/groq] chunk {i + 1}/{len(chunk_files)} done in {time.time() - t0:.1f}s", flush=True)

            resp_segments = getattr(resp, "segments", None) or []
            if resp_segments:
                for seg in resp_segments:
                    seg_start = seg["start"] if isinstance(seg, dict) else seg.start
                    seg_end = seg["end"] if isinstance(seg, dict) else seg.end
                    seg_text = seg["text"] if isinstance(seg, dict) else seg.text
                    all_segments.append({
                        "start": float(seg_start) + offset,
                        "end": float(seg_end) + offset,
                        "text": (seg_text or "").strip(),
                    })
            elif getattr(resp, "text", None):
                # Fallback: some responses may omit segment-level timestamps.
                # Keep the text so the pipeline doesn't silently lose the chunk,
                # even though highlight timing within this chunk will be coarse.
                chunk_duration = _ffprobe_duration(chunk_file)
                all_segments.append({
                    "start": offset,
                    "end": offset + chunk_duration,
                    "text": resp.text.strip(),
                })

    if duration <= 0:
        duration = all_segments[-1]["end"] if all_segments else 0.0
    print(f"[transcribe/groq] {len(all_segments)} segments, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "segments": all_segments}


# ---------------------------------------------------------------------------
# Transcribe-only-segments mode — the key optimization
# ---------------------------------------------------------------------------

def _extract_audio_segment(media_path: str, start: float, end: float, out_path: str):
    """Extract a short audio segment (start→end) to a temp mp3 file."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", media_path,
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
            out_path,
        ],
        check=True, capture_output=True,
    )


def transcribe_windows(
    media_path: str,
    windows: List[Dict],
    language: Optional[str] = None,
    margin_seconds: float = 2.0,
) -> Dict:
    """Transcribe only the audio inside each candidate window.

    Instead of transcribing the full video (the expensive part), this
    extracts + transcribes just the candidate zones. With 50 windows of
    ~30s each ≈ 25 min of audio → ~2-3 min of whisper CPU instead of
    ~45 min for a full 2h video.

    Each window gets a small margin_seconds of context on each side so
    the transcriber doesn't cut off the first/last word.
    """
    all_segments = []
    total_transcribed = 0.0

    with tempfile.TemporaryDirectory(prefix="transcribe_windows_") as tmp_dir:
        for i, w in enumerate(windows):
            w_start = max(0.0, float(w["start_time"]) - margin_seconds)
            w_end = float(w["end_time"]) + margin_seconds
            chunk_file = os.path.join(tmp_dir, f"window_{i:03d}.mp3")

            _extract_audio_segment(media_path, w_start, w_end, chunk_file)
            chunk_duration = _ffprobe_duration(chunk_file)
            if chunk_duration <= 0:
                continue

            total_transcribed += chunk_duration
            provider = (LOCAL_TRANSCRIBE_PROVIDER or "whisper").strip().lower()
            if provider == "groq":
                audio_bytes = open(chunk_file, "rb").read()
                from ..config import require_groq_key
                from openai import OpenAI
                client = OpenAI(api_key=require_groq_key(), base_url="https://api.groq.com/openai/v1")
                from .llm import _with_retries
                def _do_call():
                    buf = io.BytesIO(audio_bytes)
                    buf.name = "audio.mp3"
                    kwargs = {
                        "file": buf,
                        "model": GROQ_WHISPER_MODEL,
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["segment"],
                    }
                    if language:
                        kwargs["language"] = language
                    return client.audio.transcriptions.create(**kwargs)
                resp = _with_retries(_do_call)
                resp_segments = getattr(resp, "segments", None) or []
                for seg in resp_segments:
                    seg_start = seg["start"] if isinstance(seg, dict) else seg.start
                    seg_end = seg["end"] if isinstance(seg, dict) else seg.end
                    seg_text = seg["text"] if isinstance(seg, dict) else seg.text
                    all_segments.append({
                        "start": float(seg_start) + w_start,
                        "end": float(seg_end) + w_start,
                        "text": (seg_text or "").strip(),
                    })
            else:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as e:
                    raise RuntimeError(
                        "faster-whisper is required when LOCAL_TRANSCRIBE_PROVIDER=whisper."
                    ) from e
                device = _resolve_device()
                compute_type = "float16" if device == "cuda" else "int8"
                model_kwargs = {"device": device, "compute_type": compute_type}
                if device == "cpu":
                    model_kwargs["cpu_threads"] = LOCAL_WHISPER_CPU_THREADS
                model = WhisperModel(LOCAL_WHISPER_MODEL, **model_kwargs)
                segments_iter, info = model.transcribe(
                    chunk_file,
                    language=language,
                    beam_size=LOCAL_WHISPER_BEAM_SIZE,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                for s in segments_iter:
                    all_segments.append({
                        "start": float(s.start) + w_start,
                        "end": float(s.end) + w_start,
                        "text": (s.text or "").strip(),
                    })

            print(
                f"  window {i + 1}/{len(windows)}: {w_start:.0f}s-{w_end:.0f}s "
                f"({w_end - w_start:.0f}s of audio) → transcribed",
                flush=True,
            )

    all_segments.sort(key=lambda s: s["start"])
    transcript = {
        "duration": all_segments[-1]["end"] if all_segments else 0.0,
        "segments": all_segments,
    }
    print(
        f"[transcribe/windows] {len(all_segments)} segments from {len(windows)} windows "
        f"({total_transcribed:.0f}s of audio transcribed)",
        flush=True,
    )
    return transcript


# ---------------------------------------------------------------------------
# Public entry point — cache wrapper + backend dispatch
# ---------------------------------------------------------------------------

def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Transcribe a local file, caching the result as .srt.

    Backend picked by LOCAL_TRANSCRIBE_PROVIDER ("whisper" or "groq") — see
    config.py for what each one trades off.
    """
    cache_path = _transcript_cache_path(media_path)
    if cache_path.exists():
        source_mtime = os.path.getmtime(media_path)
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
            cached = _load_srt_cache(cache_path)
            # Treat empty cache as invalid (likely from a failed/partial run) — delete and re-transcribe
            if not cached["segments"] or cached["duration"] <= 0.0:
                print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
                cache_path.unlink(missing_ok=True)
            else:
                print(
                    f"[transcribe/local] {len(cached['segments'])} cached segments, "
                    f"{cached['duration']:.0f}s of audio",
                    flush=True,
                )
                return cached

    provider = (LOCAL_TRANSCRIBE_PROVIDER or "whisper").strip().lower()
    if provider == "groq":
        transcript = _transcribe_with_groq(media_path, language)
    elif provider == "whisper":
        transcript = _transcribe_with_faster_whisper(media_path, language)
    else:
        raise RuntimeError(
            f"Unknown LOCAL_TRANSCRIBE_PROVIDER={provider!r}. Use 'whisper' or 'groq'."
        )

    cache_path = _write_srt_cache(media_path, transcript)
    print(f"[transcribe/local] wrote cache: {cache_path}", flush=True)
    return transcript
